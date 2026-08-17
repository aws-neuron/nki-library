# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for the Sparse Attention Indexer entry kernel.

Contains:
  * Constants (``P_MAX``, ``CACHE_TILE``).
  * ``SAIConfig`` dataclass + ``validate_sai_config``.
  * ``topk_over_score`` — pads the per-S-tile score row, stages it to HBM, and
    runs hardware ``nisa.topk`` over a 16-partition snake (8 queries per call)
    to emit the per-query top-k position indices.
"""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...core.utils.kernel_assert import kernel_assert

# Constants
P_MAX = 128
"""Maximum partition dimension (tile size for sequence dimension)."""

CACHE_TILE = 512
"""Tile size for iterating over KV cache positions during the score loop.

Set to PSUM_FMAX_FP32 = 512 (the max free dim for nc_matmul_mx with fp32
dst). Each activation/scalar_tensor_tensor call processes 4x more free
elements than the original CACHE_TILE=128, reducing the inner-loop
iteration count by 4x for end_pos >= 512."""


# Shared config + validation
@dataclass(frozen=True)
class SAIConfig(nl.NKIObject):
    """Sparse Attention Indexer configuration."""

    S: int = None
    dim: int = None
    q_lora_rank: int = None
    n_heads: int = None
    head_dim: int = None
    rope_head_dim: int = None
    index_topk: int = None
    start_pos: int = None
    use_hadamard: bool = False
    batch_size: int = 1


def validate_sai_config(cfg, M):
    """Common shape/dtype validation for the SAI entry kernel.

    The indexer always runs MX (block-32 fp8) Q/K/W projections, so the MX
    shape constraints below are unconditional.
    """
    kernel_assert(cfg.S * cfg.batch_size == M, f"S*batch_size must equal M={M}")
    kernel_assert(cfg.dim % P_MAX == 0 or cfg.dim <= P_MAX, "dim must be <= 128 or multiple of 128")
    kernel_assert(
        cfg.q_lora_rank % P_MAX == 0 or cfg.q_lora_rank <= P_MAX, "q_lora_rank must be <= 128 or multiple of 128"
    )
    kernel_assert(cfg.head_dim <= P_MAX, f"head_dim must be <= {P_MAX}")
    kernel_assert(cfg.rope_head_dim <= cfg.head_dim, "rope_head_dim must be <= head_dim")
    kernel_assert(cfg.rope_head_dim % 2 == 0, "rope_head_dim must be even")
    kernel_assert(cfg.index_topk >= 8, "index_topk must be >= 8 (DVE min)")
    # SBUF-stack budget: the per-batch loop keeps a persistent weights_full_sb
    # [P_MAX, num_s_tiles_per_batch, n_heads] plus interleave_degree=2 double-
    # buffered per-S-tile buffers (score_row, mask, k_full, ...). Past
    # num_s_tiles_per_batch == 14 the high-water mark overflows the stack
    num_s_tiles_per_batch = (cfg.S + P_MAX - 1) // P_MAX
    kernel_assert(
        num_s_tiles_per_batch <= 14,
        "S too large: ceil(S / 128) must be <= 14 (SBUF-stack budget); larger S needs seq tiling",
    )
    # MX path quantizes activations along the contraction dim with block 32
    # (= P_MAX / 4 partitions x 4 free). nc_matmul_mx consumes 512 contraction
    # elements per call (P_MAX_partitions x H_PACK), so the wq_b/wk projection
    # contraction dims must be a multiple of 512.
    kernel_assert(cfg.dim % 512 == 0, "MXFP8 requires dim divisible by 512")
    kernel_assert(cfg.q_lora_rank % 512 == 0, "MXFP8 requires q_lora_rank divisible by 512")
    # quantize_mx requires the partition dim of its input (= head_dim) to be a
    # multiple of 32 with at least 32 valid rows. head_dim < 128 is a follow-up.
    kernel_assert(cfg.head_dim >= 128, "MXFP8 requires head_dim >= 128")


"""
Topk over the score row

``nisa.topk`` ranks one 16-partition group at a time (8 groups per call), each
over ``n`` elements in a 16-partition snake. SAI packs 8 queries per call (one
per group), so a full S-tile of 128 queries takes NUM_TOPK_BATCHES (= 16)
calls. T is ``end_pos`` rounded up to a multiple of 16 with ``-inf`` padding
so padded positions never win topk. The score row is padded and staged to HBM
once (``_pad_and_persist_score``), then gathered per call into the snake src.
"""

# Topk batch geometry: 16 chunks per query × 8 queries per topk batch = 128 = P_MAX.
TOPK_QUERY_CHUNKS = 16
"""Number of chunks per query. Each topk-batch partition holds one chunk."""

TOPK_QUERIES_PER_BATCH = 8
"""Number of queries packed into one topk batch (8 partitions of source data)."""

NUM_TOPK_BATCHES = P_MAX // TOPK_QUERIES_PER_BATCH  # = 16
"""Number of topk batches per S-tile (each batch handles 8 of the 128 queries)."""


def round_up_to_chunks(end_pos: int) -> int:
    """Round ``end_pos`` up to a multiple of ``TOPK_QUERY_CHUNKS`` (= 16)."""
    return ((end_pos + TOPK_QUERY_CHUNKS - 1) // TOPK_QUERY_CHUNKS) * TOPK_QUERY_CHUNKS


def _pad_and_persist_score(sbm, score_row_sb, S, end_pos, T, score_padded_hbm, s_tile_idx, part_offset=0):
    """Pad ``score_row_sb`` to ``[S, T]`` with -inf and persist to HBM.

    Writes ``score_padded_hbm[s_tile_idx, part_offset : part_offset + S, :]``.
    Padding cols past ``end_pos`` are -inf so padded positions never win topk.

    ``part_offset`` (default 0) places this write at a partition offset within
    the s-tile slot — used by the seq-sharded variant so each LNC core writes
    its own disjoint query half (core ``c`` -> partitions ``[c*S, c*S + S)``) and
    later reads back exactly that range for its own topk batches (no cross-core
    sync). ``S`` is the local query count (== P_MAX for the un-sharded kernel).
    """
    kernel_assert(T % TOPK_QUERY_CHUNKS == 0, f"T must be a multiple of {TOPK_QUERY_CHUNKS}, got T={T}")
    score_padded_sb = sbm.alloc_stack((P_MAX, T), nl.float32)
    nisa.memset(dst=score_padded_sb[:S, :T], value=float("-inf"))
    nisa.tensor_copy(dst=score_padded_sb[:S, :end_pos], src=score_row_sb[:S, :end_pos])
    nisa.dma_copy(
        dst=score_padded_hbm[s_tile_idx, part_offset : part_offset + S, :T],
        src=score_padded_sb[:S, :T],
        dge_mode=dge_mode.swdge,
    )


def _gather_batch(dst_sb, batch_idx, score_padded_hbm, s_tile_idx, T, src_x):
    # Contiguous-chunk gather (HW-safe DMA AP): partition (16g + p) of group
    # g holds query (8t+g)'s contiguous chunk p, i.e.
    #   src[16g + p, c] = score[8t+g, p*src_x + c].
    # A round-robin (stride-16) free axis fails hardware DMA lowering
    # ("outer dim not divisible by 128"); a contiguous free axis (stride 1)
    # lowers fine, matching the proven relayout AP. topk finds the same
    # top-k by VALUE regardless of arrangement — only the returned snake
    # index i must be remapped to the actual position:
    #   actual_pos(i) = (i % 16) * src_x + (i // 16)
    # (done by the consumer / test, since topk emits snake indices).
    # NOTE: hoisted to MODULE level (was a nested inner fn) — NKI forbids
    # direct calls to inner functions (only fori_loop/while_loop bodies).
    q_base = batch_idx * TOPK_QUERIES_PER_BATCH  # 8 * batch_idx
    nisa.dma_copy(
        dst=dst_sb[0:P_MAX, 0:src_x],
        src=score_padded_hbm.ap(
            pattern=[
                [T, TOPK_QUERIES_PER_BATCH],  # g: stride T, count 8 (partition ×16)
                [src_x, TOPK_QUERY_CHUNKS],  # p: stride src_x, count 16 (partition low)
                [1, src_x],  # c: stride 1, count src_x (free, contiguous)
            ],
            offset=(s_tile_idx * P_MAX + q_base) * T,
        ),
        dge_mode=dge_mode.swdge,
    )


def topk_over_score(
    sbm,
    score_row_sb,
    S,
    end_pos,
    index_topk,
    score_padded_hbm,
    topk_idx_hbm,
    s_tile_idx,
    topk_flat_hbm=None,
    s_tile_base_q=0,
    topk_tiled_hbm=None,
    batch_start=0,
    batch_end=NUM_TOPK_BATCHES,
    part_offset=0,
):
    """Run hardware ``nisa.topk`` over the S-tile's score row.

    ``nisa.topk`` ranks one 16-partition group at a time (8 groups per call),
    each over ``n`` elements in a 16-partition *snake*: element ``i`` lives at
    partition ``base + i % 16``, column ``i // 16`` (``base = group * 16``).
    Output value ``j`` lands at partition ``base + j % 16``, column ``j // 16``,
    ascending.

    SAI has ``P_MAX`` queries per S-tile, each needing the top ``index_topk``
    of ``end_pos`` positions. We pack 8 queries per call (one per group), so a
    full S-tile takes ``NUM_TOPK_BATCHES`` (= 16) calls.

    Per call ``t`` (queries ``8t .. 8t+7``):
      * Gather a round-robin snake ``src[16g + r, c] = score[8t+g, c*16 + r]``
        from the padded-score HBM staging tensor (one AP DMA, f32->bf16 cast).
      * ``nisa.topk(val, idx, src, n=end_pos)`` -> per-group top-``index_topk``.
      * Emit the per-query top-k, in one of the output formats (below).

    Output format — SNAKE (default) vs FLAT vs TILED:
      * ``topk_flat_hbm is None`` and ``topk_tiled_hbm is None`` (SNAKE): persist
        ``idx`` (snake layout, uint32) to ``topk_idx_hbm[s_tile, t]``. The
        consumer must remap the snake index ``i`` to the actual position
        ``(i%16)*src_x + (i//16)`` (``src_x = T/16``). Used by the fused mega
        kernel, which decodes the snake itself.
      * ``topk_flat_hbm is not None`` (FLAT): decode the snake index -> position
        IN SBUF (``pos = (i&15)*src_x + (i>>4)``, int32) and scatter each query's
        k positions directly to ``topk_flat_hbm[q, :]`` (``q = s_tile_base_q +
        8t + g``). This FUSES the former torch/standalone snake->flat decode into
        the indexer: no snake HBM write, no round-trip, and the decode rides the
        kernel's LNC token sharding. ``topk_idx_hbm`` is ignored in this mode.
        This is what ``sparse_mla_latent_attn_cte``'s ``[B, S, K]`` int32
        ``topk_indices`` contract expects.
      * ``topk_tiled_hbm is not None`` (TILED / split-fix): decode as FLAT but
        skip the per-query scatter and write the whole batch's decoded positions
        in the indexer's NATURAL partition-tiled layout in ONE contiguous DMA.

    ``batch_start``/``batch_end``/``part_offset`` restrict this call to a query
    sub-range for the seq-sharded variant (each LNC core owns a disjoint query
    half); the defaults reproduce the single-core all-queries loop.

    Args:
        score_row_sb [P_MAX, end_pos] f32 — already mask-added.
        index_topk: number of top positions to select per query (= k).
        score_padded_hbm [num_S_tiles_total, P_MAX, T] f32 — staging tensor.
        topk_idx_hbm [num_S_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk]
            uint32 — SNAKE output (used only when ``topk_flat_hbm`` and
            ``topk_tiled_hbm`` are both None); ``[s_tile, t, 16g + j%16, j//16]``
            = the SNAKE index ``i`` of query ``8t+g``'s j-th largest score.
        s_tile_idx: S-tile index across the whole kernel call.
        topk_flat_hbm [S_flat, index_topk] int32 or None — FLAT output. When
            given, decoded positions are written to row ``q`` directly and
            ``topk_idx_hbm`` is unused.
        s_tile_base_q: global query-row offset of this S-tile's first query
            (= ``s_tile_idx * P_MAX`` for a single-batch call), used to index
            ``topk_flat_hbm`` rows in FLAT mode.
        topk_tiled_hbm [num_S_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk//16]
            int32 or None — TILED (split-fix) output.
    """
    T = round_up_to_chunks(end_pos)
    src_x = T // TOPK_QUERY_CHUNKS  # ceil(end_pos / 16)
    k = index_topk
    emit_flat = topk_flat_hbm is not None
    emit_tiled = topk_tiled_hbm is not None

    sbm.open_scope(name="topk")
    # part_offset places this core's S local queries at their global partition rows
    # in the s-tile slot, so the batch-indexed gather below (q_base = 8*batch_idx,
    # a GLOBAL query offset) reads exactly the rows this core wrote. batch_start/end
    # restrict the 16-batch loop to this core's query half (seq-sharded variant); the
    # defaults (0, NUM_TOPK_BATCHES, 0) reproduce the single-core all-queries loop.
    _pad_and_persist_score(sbm, score_row_sb, S, end_pos, T, score_padded_hbm, s_tile_idx, part_offset=part_offset)

    # Software-pipeline the 16-batch topk loop: the gather DMA feeds nisa.topk
    # (GPSIMD), which then feeds the decode (Vector). Double-buffer src and
    # prefetch batch t+1's gather BEFORE batch t's topk, so the next gather DMA
    # overlaps this batch's topk + decode (they were fully serial per batch — the
    # ~49us DMA/sync half of the topk tail). Two persistent src buffers, alternated
    # by parity; topk consumes src_bufs[t%2] after the prefetch fills [(t+1)%2].
    src_bufs = [
        sbm.alloc_stack((P_MAX, src_x), nl.bfloat16),
        sbm.alloc_stack((P_MAX, src_x), nl.bfloat16),
    ]
    _gather_batch(src_bufs[0], batch_start, score_padded_hbm, s_tile_idx, T, src_x)

    for batch_idx in range(batch_start, batch_end):
        sbm.open_scope(name=f"topk_b{batch_idx}")
        q_base = batch_idx * TOPK_QUERIES_PER_BATCH  # 8 * batch_idx

        # Prefetch next batch's gather into the alternate buffer so its DMA
        # overlaps this batch's topk (GPSIMD) + decode (Vector).
        if batch_idx + 1 < batch_end:
            _gather_batch(src_bufs[(batch_idx + 1) % 2], batch_idx + 1, score_padded_hbm, s_tile_idx, T, src_x)
        src_sb = src_bufs[(batch_idx - batch_start) % 2]

        # n = T (= src_x * 16): rank ALL snake slots. Padded positions
        # [end_pos, T) are -inf and never win. Using n=end_pos would leave
        # snake slots unconsidered AND (when end_pos < T) misalign the
        # contiguous-chunk mapping.
        val_sb = sbm.alloc_stack((P_MAX, k), nl.bfloat16)
        idx_sb = sbm.alloc_stack((P_MAX, k), nl.uint32)
        nisa.topk(
            val_dst=val_sb[:P_MAX, :k],
            idx_dst=idx_sb[:P_MAX, :k],
            src=src_sb[:P_MAX, :src_x],
            n=T,
        )

        if not emit_flat and not emit_tiled:
            # SNAKE output: persist the raw snake indices; consumer remaps.
            nisa.dma_copy(
                dst=topk_idx_hbm[s_tile_idx, batch_idx, :P_MAX, :k],
                src=idx_sb[:P_MAX, :k],
                dge_mode=dge_mode.swdge,
            )
        else:
            # FLAT or TILED output: decode snake index -> position IN SBUF.
            hi_sb = sbm.alloc_stack((P_MAX, k), nl.uint32)
            lo_sb = sbm.alloc_stack((P_MAX, k), nl.uint32)
            pos_sb = sbm.alloc_stack((P_MAX, k), nl.int32)
            nisa.tensor_scalar(
                hi_sb[:P_MAX, :k], idx_sb[:P_MAX, :k], op0=nl.right_shift, operand0=4, engine=nisa.engine.vector
            )
            nisa.tensor_scalar(
                lo_sb[:P_MAX, :k], idx_sb[:P_MAX, :k], op0=nl.bitwise_and, operand0=15, engine=nisa.engine.vector
            )
            nisa.tensor_scalar(
                lo_sb[:P_MAX, :k], lo_sb[:P_MAX, :k], op0=nl.multiply, operand0=src_x, engine=nisa.engine.vector
            )
            nisa.tensor_tensor(
                dst=pos_sb[:P_MAX, :k],
                data1=lo_sb[:P_MAX, :k],
                data2=hi_sb[:P_MAX, :k],
                op=nl.add,
                engine=nisa.engine.vector,
            )

            c_cnt = k // TOPK_QUERY_CHUNKS
            if emit_tiled:
                # SPLIT-FIX write (paired with topk_tiled read in
                # sparse_mla_latent_attn_cte).
                nisa.dma_copy(
                    dst=topk_tiled_hbm[s_tile_idx, batch_idx, :P_MAX, :c_cnt],
                    src=pos_sb[:P_MAX, :c_cnt],
                    dge_mode=dge_mode.swdge,
                )
            else:
                # Scatter each query's k decoded positions to topk_flat_hbm[q, :].
                for g in range(TOPK_QUERIES_PER_BATCH):
                    q = s_tile_base_q + q_base + g
                    if q >= s_tile_base_q + S:
                        continue
                    nisa.dma_copy(
                        dst=topk_flat_hbm.ap(
                            pattern=[[1, TOPK_QUERY_CHUNKS], [TOPK_QUERY_CHUNKS, c_cnt]],
                            offset=q * k,
                        ),
                        src=pos_sb.ap(
                            pattern=[[k, TOPK_QUERY_CHUNKS], [1, c_cnt]],
                            offset=g * TOPK_QUERY_CHUNKS * k,
                        ),
                        dge_mode=dge_mode.swdge,
                    )
        sbm.close_scope()

    sbm.close_scope()
