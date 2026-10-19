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
from ..mla.deepseek.mla_common_cte import MlaPrecision

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


def validate_sai_config(cfg, M, precision=MlaPrecision.MX):
    """Common shape/dtype validation for the SAI entry kernel.

    ``precision`` selects the Q/K projection format. The MX shape constraints at the bottom
    (contraction dims divisible by 512, head_dim >= 128) come from ``nc_matmul_mx``'s 4-packing
    and ``quantize_mx``'s block-32 partition requirement; the bf16 path uses plain
    ``nc_matmul``, which only needs the contraction to tile at the 128-partition cap.
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
        num_s_tiles_per_batch <= 64,  # EXPERIMENT: was 14
        "S too large: ceil(S / 128) must be <= 14 (SBUF-stack budget); larger S needs seq tiling",
    )
    if precision.is_bf16():
        """
        BF16 projections tile the contraction at the 128-partition cap with no packing, so both
        contraction dims only need to be multiples of 128 (GLM-MoE-DSA's dim=6144 and
        q_lora_rank=2048 satisfy either bound; the looser one matters for TP shards). head_dim
        is still capped at 128 above by the shared check, and the score matmul wants it full.
        """
        kernel_assert(cfg.dim % P_MAX == 0, f"bf16 requires dim divisible by {P_MAX}")
        kernel_assert(cfg.q_lora_rank % P_MAX == 0, f"bf16 requires q_lora_rank divisible by {P_MAX}")
        return
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

    if T >= TOPK_N_MAX:
        topk_over_score_chunked(
            sbm,
            S,
            end_pos,
            index_topk,
            score_padded_hbm,
            s_tile_idx,
            topk_flat_hbm,
            s_tile_base_q=s_tile_base_q,
            batch_start=batch_start,
            batch_end=batch_end,
            part_offset=part_offset,
        )
        return
    src_x = T // TOPK_QUERY_CHUNKS  # ceil(end_pos / 16)
    k = index_topk
    emit_flat = topk_flat_hbm is not None
    emit_tiled = topk_tiled_hbm is not None

    sbm.open_scope(name=f"topk{_u()}")
    if score_row_sb != None:
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
        sbm.open_scope(name=f"topk_b{batch_idx}{_u()}")
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


# nisa.topk ranks at most this many elements per 16-partition snake (doc: 8 <= n < 65536).
TOPK_N_MAX = 65536
# PSUM free width for an fp32 moving operand; the one-hot lookup tiles its winners by this.
_PSUM_MOVING_F32 = 512


_UNIQ = [0]


def _u():
    """Monotonic suffix for ``alloc_stack`` names.

    Names must be unique across the WHOLE trace -- ``close_scope`` releases a scope's addresses
    but not its tensor names -- and every loop here is Python-unrolled, so a bare literal name
    collides on the second iteration. The toolchain began enforcing this, so hand-written index
    suffixes were failing one name per compile; a trace-order counter makes collisions
    structurally impossible instead.
    """
    _UNIQ[0] += 1
    return f"_{_UNIQ[0]}"


def pick_topk_segments(T: int) -> int:
    """Smallest legal segment count for a two-stage top-k over ``T`` padded positions.

    ``nisa.topk`` caps ``n`` (elements per 16-partition snake) at ``TOPK_N_MAX``, so a context
    wider than that must be split into ``NSEG`` segments, each ranked separately and then
    merged. Fewer segments is strictly better: the ``4*n`` term of the instruction's
    ``4*n + K`` cost is invariant under the split (the same elements get ranked either way),
    so the only thing NSEG multiplies is the per-call ``+K``. Hence "smallest legal".

    NSEG must also divide ``TOPK_QUERIES_PER_BATCH`` (8): a 128-partition tile holds 8
    independent snakes, and the split spends NSEG of them on positions, leaving ``8 // NSEG``
    for queries. And ``T // NSEG`` must stay a multiple of ``TOPK_QUERY_CHUNKS`` so each
    segment's snake is whole.

    Returns 1 when ``T`` already fits, i.e. the caller should use the single-stage path.
    """
    for nseg in (1, 2, 4, 8):
        seg = T // nseg
        if T % nseg == 0 and seg < TOPK_N_MAX and seg % TOPK_QUERY_CHUNKS == 0:
            return nseg
    kernel_assert(
        False,
        f"two-stage topk needs T/NSEG < {TOPK_N_MAX} with NSEG <= {TOPK_QUERIES_PER_BATCH} "
        f"(one snake per position segment); T={T} exceeds what 8 segments can cover "
        f"({TOPK_QUERIES_PER_BATCH * TOPK_N_MAX}).",
    )
    return TOPK_QUERIES_PER_BATCH


def _gather_stage1_snakes(dst_sb, q_base, score_padded_hbm, s_tile_idx, T, nseg, qpc, seg, src_x_seg):
    """Gather ``qpc`` queries x ``nseg`` position segments into the 8 snakes of one tile.

    Group ``g`` of the tile (partitions ``16g .. 16g+15``) is assigned query ``g // nseg`` and
    position segment ``g % nseg``, so the 8 independent top-k the instruction already performs
    per issue cover (query, segment) pairs instead of 8 queries. That is the whole trick: the
    position split costs nothing to build because it IS the group wiring.

    The free axis stays contiguous (stride 1) for the same reason as :func:`_gather_batch` -- a
    round-robin free axis fails hardware DMA lowering. So within a segment, snake element ``i``
    is chunk ``i % 16``, offset ``i // 16``, and the position decode carries the matching
    ``(i % 16) * src_x_seg + (i // 16)`` remap plus the segment's base.

    Partition index is the row-major product of the leading three dims:
    ``j*(nseg*16) + s*16 + p`` = ``16*(j*nseg + s) + p`` = ``16g + p``.
    """
    nisa.dma_copy(
        dst=dst_sb[0:P_MAX, 0:src_x_seg],
        src=score_padded_hbm.ap(
            pattern=[
                [T, qpc],  # j: query, stride T (one padded score row per query)
                [seg, nseg],  # s: position segment within the query's row
                [src_x_seg, TOPK_QUERY_CHUNKS],  # p: contiguous chunk -> partition low bits
                [1, src_x_seg],  # c: column (free, contiguous)
            ],
            offset=(s_tile_idx * P_MAX + q_base) * T,
        ),
        dge_mode=dge_mode.swdge,
    )


def _decode_stage1_positions(sbm, idx_s1, k, nseg, qpc, seg, src_x_seg):
    """Snake indices -> GLOBAL cache positions.

    Stage-1 group ``g = j*nseg + s`` ranked segment ``s``, so its snake index ``i`` means
    position ``s*seg + (i % 16) * src_x_seg + (i // 16)``. The first two terms are the remap the
    single-stage FLAT path also does; ``s*seg`` is what the position split adds.

    THREE full-width passes, not five. ``tensor_scalar`` applies a second operator at no extra
    cost ("performing one operator has the same performance cost as performing two"), so the
    shift fuses with the segment-base add, and the mask fuses with the stride multiply. Vector is
    the kernel's bottleneck engine and these are bitvec operators, which pin the instruction to
    Vector with integer operands -- so they cannot be moved to Scalar and cutting passes is the
    only lever available here.

    The segment base is derived from the partition index (``s = (p >> 4) & (nseg-1)``, exact
    because nseg is a power of two) rather than per-group slices: a 16-partition access must
    start at partition 0/32/64/96, so slicing ``[16g, 16g+16)`` is not even legal. It stays
    integer so it can ride the bitvec instruction, and it is a [P,1] column so the whole thing is
    one broadcast.

    Positions come out float32: the one-hot lookup's ``dma_transpose`` needs matching src/dst
    dtypes against its fp32 destination, and a cache position (< 2**24) is exact in fp32.
    """
    pos_s1 = sbm.alloc_stack((P_MAX, k), nl.float32, name=f"cand_pos{_u()}")
    hi = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s1_hi{_u()}")
    lo = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s1_lo{_u()}")

    # Segment base per partition -- [P,1], so these three are negligible next to the k-wide passes.
    pidx = sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"s1_pidx{_u()}")
    segoff_u = sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"s1_segoff{_u()}")
    nisa.iota(dst=pidx, pattern=[[0, 1]], channel_multiplier=1)
    nisa.tensor_scalar(
        segoff_u[:P_MAX, 0:1],
        pidx[:P_MAX, 0:1],
        op0=nl.right_shift,
        operand0=4,
        op1=nl.bitwise_and,
        operand1=nseg - 1,
    )
    nisa.tensor_scalar(segoff_u[:P_MAX, 0:1], segoff_u[:P_MAX, 0:1], op0=nl.multiply, operand0=seg)
    # float32 twin: tensor_scalar's fused arithmetic form requires operand1 to be float32. This is
    # a [P,1] column, so the extra cast is negligible next to the k-wide passes below.
    segoff_f = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"s1_segoff_f{_u()}")
    nisa.tensor_copy(dst=segoff_f[:P_MAX, 0:1], src=segoff_u[:P_MAX, 0:1])

    # (i >> 4) + seg_base, and (i & 15) * src_x_seg -- one instruction each.
    # A tensor_scalar cannot mix a bitvec op0 with an arithmetic op1 ("Unsupported bitvec op"),
    # so the shift and mask stand alone; only the arithmetic pair fuses. 4 passes, down from 5.
    nisa.tensor_scalar(hi[:P_MAX, :k], idx_s1[:P_MAX, :k], op0=nl.right_shift, operand0=4, engine=nisa.engine.vector)
    nisa.tensor_scalar(lo[:P_MAX, :k], idx_s1[:P_MAX, :k], op0=nl.bitwise_and, operand0=15, engine=nisa.engine.vector)
    nisa.tensor_scalar(
        pos_s1[:P_MAX, :k],
        lo[:P_MAX, :k],
        op0=nl.multiply,
        operand0=src_x_seg,
        op1=nl.add,
        operand1=segoff_f[:P_MAX, 0:1],
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(dst=pos_s1[:P_MAX, :k], data1=pos_s1[:P_MAX, :k], data2=hi[:P_MAX, :k], op=nl.add)
    return pos_s1


def _relayout_candidates(sbm, val_s1, pos_s1, k, nseg, qpc):
    """Regroup stage-1's 8 per-(query, segment) results into one snake per QUERY.

    Stage 1 leaves group ``g = j*nseg + s`` in partitions ``[16g, 16g+16)``, each holding its
    own ``k`` winners snake-arranged (rank ``r`` at partition ``16g + r%16``, column ``r//16``).
    Stage 2 must rank all ``nseg*k`` of query ``j``'s candidates as a SINGLE snake, i.e. in
    partitions ``[16j, 16j+16)``.

    Choosing candidate slot ``e = s*k + r`` makes the move a pure partition-block copy: since
    ``k % TOPK_QUERY_CHUNKS == 0``, ``e % 16 == r % 16`` (low partition bits are preserved) and
    ``e // 16 == s*(k//16) + r//16`` (the segment only shifts the column block). So each of the
    8 groups is one contiguous ``[16, k//16]`` rectangle moved to a column offset -- no
    intra-partition shuffle, no transpose.
    """
    k_cols = k // TOPK_QUERY_CHUNKS
    src_x2 = nseg * k_cols
    cand_v = sbm.alloc_stack((P_MAX, src_x2), nl.bfloat16, name=f"cand_v{_u()}")
    cand_p = sbm.alloc_stack((P_MAX, src_x2), nl.float32, name=f"cand_p{_u()}")

    for j in range(qpc):
        for s in range(nseg):
            g = j * nseg + s
            src_p = nl.ds(g * TOPK_QUERY_CHUNKS, TOPK_QUERY_CHUNKS)
            dst_p = nl.ds(j * TOPK_QUERY_CHUNKS, TOPK_QUERY_CHUNKS)
            cols = nl.ds(s * k_cols, k_cols)
            # Cross-partition move => DMA (SBUF->SBUF); tensor_copy cannot change partition.
            # One DMA per (query, segment): a single ap() cannot express this regroup, because the
            # destination partition 16j + p needs two partition axes (strides 16 and 1) and ap()
            # requires the partition step to equal the tensor's free dimension size.
            nisa.dma_copy(dst=cand_v[dst_p, cols], src=val_s1[src_p, 0:k_cols])
            nisa.dma_copy(dst=cand_p[dst_p, cols], src=pos_s1[src_p, 0:k_cols])
    return cand_v, cand_p


def make_onehot_consts(sbm):
    """Loop-invariant operands for :func:`_lookup_positions_onehot`.

    Built once per s-tile and passed in: the partition ramp and the two ones-vectors do not
    depend on the query group, and rebuilding them inside the per-group loop emitted them
    ``S/qpc`` times (64x at 128k), which is pure instruction-count bloat.
    """
    iota_u = sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"oh_iota_u{_u()}")
    iota_f = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"oh_iota_f{_u()}")
    nisa.iota(dst=iota_u, pattern=[[0, 1]], channel_multiplier=1)
    nisa.tensor_copy(dst=iota_f, src=iota_u)
    ones_row_bf = sbm.alloc_stack((1, P_MAX), nl.bfloat16, name=f"oh_ones_row{_u()}")
    nisa.memset(dst=ones_row_bf, value=1.0)
    ones_col = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"oh_ones_col{_u()}")
    nisa.memset(dst=ones_col, value=1.0)
    return iota_f, ones_row_bf, ones_col


def _lookup_positions_onehot(sbm, cand_p, r_idx, k, qpc, src_x2, topk_flat_hbm, q_row_base, q_row_hi, consts):
    """``out[j, w] = cand_p[query j, slot(r_idx[j, w])]`` via one-hot matmuls on the PE array.

    Deliberately NOT ``nisa.nc_n_gather``, even though a per-partition gather is the obvious
    primitive. ``chunked_topk`` documents that gather as unusable for exactly this lookup: its
    ucode signals completion before its stores are visible, and four escalating fence
    constructions (a flow-dependent Vector consumer 500 ns downstream, an SBUF->SBUF DMA fence,
    a GpSIMD copy fence the compiler copy-propagated away, and an unfoldable cast fence) each
    still read stale data. It is also only 8 columns per issue -- 256 issues at k=2048.

    Three shape rules keep this cheap:

    1. **Flatten the winners to one partition row.** A winner tile is [16, k/16] (snake), so
       both indices of the lookup vary per element. Flattened to ``[1, k]`` the one-hot
       broadcast has a single moving operand and the tile finishes in ``k/512`` PSUM-width
       passes; looping the 16 partitions instead would multiply every matmul by 16.
    2. **Split the slot the way the snake already splits it**: ``col = e // 16``,
       ``row = e % 16``. A column select then a 16-wide row select, never one one-hot over all
       ``nseg*k`` slots.
    3. **Block the column select at 128.** ``src_x2`` reaches 512 at 128k, past the partition
       dim a one-hot can span, so the select runs per 128-column block and ACCUMULATES in PSUM.
       Exactly one block holds the true column, so the sum selects it -- no masking needed.

    Cost: ``2 + ceil(src_x2/128)`` matmuls per 512 winners per query.
    """
    C = k // TOPK_QUERY_CHUNKS  # winner columns per query
    n_blk = (src_x2 + P_MAX - 1) // P_MAX
    iota_f, ones_row_bf, ones_col = consts

    flat_pos_i = sbm.alloc_stack((1, k), nl.int32, name=f"oh_flat_pos_i{_u()}")

    for j in range(qpc):
        qp0 = j * TOPK_QUERY_CHUNKS
        # Per-block transpose of this query's candidate positions: column index on partitions
        # (the select contracts over it), the snake's 16 rows on the free axis.
        # align=32: a dma_transpose destination must sit on a 32B boundary (the XBar
        # descriptor verifier rejects an unaligned output offset).
        cp_t = sbm.alloc_stack((P_MAX, n_blk, TOPK_QUERY_CHUNKS), nl.float32, name=f"oh_cpt{_u()}", align=32)
        for b in range(n_blk):
            w = min(P_MAX, src_x2 - b * P_MAX)
            nisa.dma_transpose(
                dst=cp_t[0:w, b, 0:TOPK_QUERY_CHUNKS],
                src=cand_p[nl.ds(qp0, TOPK_QUERY_CHUNKS), nl.ds(b * P_MAX, w)],
            )

        # Flatten this query's k winner indices to one row: winner w sits at partition
        # qp0 + w%16, column w//16, so a partition-minor AP walks them in w order.
        flat_i = sbm.alloc_stack((1, k), nl.uint32, name=f"oh_flat_i{_u()}")
        for rp in range(TOPK_QUERY_CHUNKS):
            nisa.dma_copy(
                dst=flat_i[0:1, nl.ds(rp * C, C)],
                src=r_idx[nl.ds(qp0 + rp, 1), 0:C],
            )

        flat_pos = sbm.alloc_stack((1, k), nl.float32, name=f"oh_flat_pos{_u()}")
        col_u = sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.uint32, name=f"oh_colu{_u()}")
        row_u = sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.uint32, name=f"oh_rowu{_u()}")
        col_bf = sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.bfloat16, name=f"oh_colbf{_u()}")
        row_bf = sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.bfloat16, name=f"oh_rowbf{_u()}")
        col_hot = sbm.alloc_stack((P_MAX, _PSUM_MOVING_F32), nl.float32, name=f"oh_colhot{_u()}")
        row_sel = sbm.alloc_stack((P_MAX, _PSUM_MOVING_F32), nl.float32, name=f"oh_rowsel{_u()}")

        for w0 in range(0, k, _PSUM_MOVING_F32):
            n_w = min(_PSUM_MOVING_F32, k - w0)
            wt = nl.ds(w0, n_w)
            nisa.tensor_scalar(col_u[0:1, 0:n_w], flat_i[0:1, wt], op0=nl.right_shift, operand0=4)
            nisa.tensor_scalar(row_u[0:1, 0:n_w], flat_i[0:1, wt], op0=nl.bitwise_and, operand0=15)
            nisa.tensor_copy(dst=row_bf[0:1, 0:n_w], src=row_u[0:1, 0:n_w])

            """
            Column select, accumulated over 128-column blocks.

            The block base is subtracted BEFORE the bf16 cast, so every value that reaches the
            comparison is a block-LOCAL column in [0, 128). nc_matmul's operand must be bf16, and
            bf16 has 8 mantissa bits -- integers are exact only to 256. src_x2 = nseg * k/16
            reaches 512 at the target index_topk of 2048, so casting a GLOBAL column index would
            round anything above 256 to a neighbour and the one-hot would select the wrong column.
            That is invisible at index_topk 512 (src_x2 = 128, all columns exact), which is why
            this only showed up at the shipped k.

            Columns outside block b land outside [0, 128) after the subtraction and so match no
            partition; a rounded value >= 256 cannot round into [0, 128) either. Exactly one block
            therefore contributes, and the PSUM accumulation across blocks selects it.
            """
            m_ps = nl.ndarray((P_MAX, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
            for b in range(n_blk):
                w = min(P_MAX, src_x2 - b * P_MAX)
                # Subtract straight into the bf16 matmul operand -- the cast on the result is
                # free, so the tensor_copy this used to need is one pass removed.
                nisa.tensor_scalar(dst=col_bf[0:1, 0:n_w], data=col_u[0:1, 0:n_w], op0=nl.subtract, operand0=b * P_MAX)
                col_b = nl.ndarray((P_MAX, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=col_b[0:w, 0:n_w], stationary=ones_row_bf[0:1, 0:w], moving=col_bf[0:1, 0:n_w])
                nisa.tensor_scalar(
                    dst=col_hot[0:w, 0:n_w], data=col_b[0:w, 0:n_w], op0=nl.equal, operand0=iota_f[0:w, 0:1]
                )
                nisa.nc_matmul(
                    dst=m_ps[0:TOPK_QUERY_CHUNKS, 0:n_w],
                    stationary=cp_t[0:w, b, 0:TOPK_QUERY_CHUNKS],
                    moving=col_hot[0:w, 0:n_w],
                    accumulate=(b > 0),
                )

            # Row select over the snake's 16 rows, then collapse the partition axis.
            row_b = nl.ndarray((P_MAX, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=row_b[0:TOPK_QUERY_CHUNKS, 0:n_w],
                stationary=ones_row_bf[0:1, 0:TOPK_QUERY_CHUNKS],
                moving=row_bf[0:1, 0:n_w],
            )
            nisa.tensor_scalar(
                dst=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
                data=row_b[0:TOPK_QUERY_CHUNKS, 0:n_w],
                op0=nl.equal,
                operand0=iota_f[0:TOPK_QUERY_CHUNKS, 0:1],
            )
            nisa.tensor_tensor(
                dst=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
                data1=m_ps[0:TOPK_QUERY_CHUNKS, 0:n_w],
                data2=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
                op=nl.multiply,
            )
            sel = nl.ndarray((1, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=sel[0:1, 0:n_w],
                stationary=ones_col[0:TOPK_QUERY_CHUNKS, 0:1],
                moving=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
            )
            nisa.tensor_copy(dst=flat_pos[0:1, wt], src=sel[0:1, 0:n_w])

        # Cast to the int32 FLAT contract and write this query's k positions in one DMA.
        # No scatter back to the [16, C] snake: the FLAT consumer takes a flat unordered row.
        nisa.tensor_copy(dst=flat_pos_i[0:1, 0:k], src=flat_pos[0:1, 0:k])
        if q_row_base + j < q_row_hi:
            nisa.dma_copy(dst=topk_flat_hbm[nl.ds(q_row_base + j, 1), 0:k], src=flat_pos_i[0:1, 0:k])


def topk_over_score_chunked(
    sbm,
    S,
    end_pos,
    index_topk,
    score_padded_hbm,
    s_tile_idx,
    topk_flat_hbm,
    s_tile_base_q=0,
    batch_start=0,
    batch_end=NUM_TOPK_BATCHES,
    part_offset=0,
):
    """Two-stage top-k for contexts past ``nisa.topk``'s ``n < 65536`` ceiling.

    Stage 1 spends the tile's 8 snakes on POSITION segments rather than 8 queries, so each
    snake ranks ``T / NSEG`` positions (see :func:`pick_topk_segments`); stage 2 merges the
    ``NSEG * k`` survivors per query. Correct because a global top-k element is necessarily in
    the top-k of whichever segment holds it.

    Perf notes:
      * The instruction's ``4*n`` term is invariant under the split -- the same elements are
        ranked either way -- so the split multiplies only the per-call ``+K``. NSEG is chosen
        as small as legality allows for that reason.
      * Stage 2 batches ``qpc`` queries per issue (one snake each), so the merge costs
        ``S/qpc`` issues per s-tile rather than one per query.
      * No descending sort: attention gathers all k selected positions, so their order is
        irrelevant. That skips the bitonic network chunked_topk needs for its sorted contract.

    Emits FLAT decoded positions only. The single-stage SNAKE contract cannot be reproduced --
    it hands the consumer a raw snake index to remap with one global ``src_x``, and after a
    position split there is no single such mapping. FLAT is what
    ``sparse_mla_latent_attn_cte``'s ``topk_indices`` wants anyway.
    """
    T = round_up_to_chunks(end_pos)
    k = index_topk
    nseg = pick_topk_segments(T)
    qpc = TOPK_QUERIES_PER_BATCH // nseg
    seg = T // nseg
    src_x_seg = seg // TOPK_QUERY_CHUNKS
    k_cols = k // TOPK_QUERY_CHUNKS
    src_x2 = nseg * k_cols
    n2 = nseg * k

    kernel_assert(topk_flat_hbm != None, "chunked topk emits FLAT positions; pass emit_flat_topk=True")
    kernel_assert(k % TOPK_QUERY_CHUNKS == 0, f"chunked topk needs k % {TOPK_QUERY_CHUNKS} == 0, got k={k}")
    kernel_assert(n2 < TOPK_N_MAX, f"stage-2 merge needs nseg*k < {TOPK_N_MAX}, got {n2}")
    kernel_assert(k <= n2, f"topk needs K <= n; k={k}, n2={n2}")

    sbm.open_scope(name=f"topk_chunked{_u()}")
    consts = make_onehot_consts(sbm)
    """
    Query range, in ABSOLUTE s-tile rows.

    ``S`` is this core's LOCAL query count (the caller passes ``sc_S``), while q_lo/q_hi index the
    s-tile's rows absolutely, so the upper bound is ``part_offset + S`` -- the end of the range
    this core owns -- NOT ``S``. Clamping with ``S`` gave core 1 ``range(64, 64)``: an empty loop,
    so it emitted nothing and its rows read back as zeros. That looked segment-selective rather
    than missing, because a zero passes segment 0's ``[0, seg)`` range check and fails segment
    1's, and it was invisible at LNC=1 where part_offset is 0 and S is the full 128.
    """
    q_lo = batch_start * TOPK_QUERIES_PER_BATCH
    q_hi = min(batch_end * TOPK_QUERIES_PER_BATCH, part_offset + S)

    for q0 in range(q_lo, q_hi, qpc):
        sbm.open_scope(name=f"chunked_q{q0}{_u()}")

        # ---- Stage 1: one snake per (query, position segment) ----
        s1_src = sbm.alloc_stack((P_MAX, src_x_seg), nl.bfloat16, name=f"s1_src{_u()}")
        _gather_stage1_snakes(s1_src, q0, score_padded_hbm, s_tile_idx, T, nseg, qpc, seg, src_x_seg)

        val_s1 = sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"s1_val{_u()}")
        idx_s1 = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s1_idx{_u()}")
        nisa.topk(val_dst=val_s1[:P_MAX, :k], idx_dst=idx_s1[:P_MAX, :k], src=s1_src[:P_MAX, :src_x_seg], n=seg)

        pos_s1 = _decode_stage1_positions(sbm, idx_s1, k, nseg, qpc, seg, src_x_seg)
        cand_v, cand_p = _relayout_candidates(sbm, val_s1, pos_s1, k, nseg, qpc)

        # ---- Stage 2: merge each query's nseg*k candidates ----
        val_s2 = sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"s2_val{_u()}")
        idx_s2 = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s2_idx{_u()}")
        nisa.topk(val_dst=val_s2[:P_MAX, :k], idx_dst=idx_s2[:P_MAX, :k], src=cand_v[:P_MAX, :src_x2], n=n2)

        _lookup_positions_onehot(
            sbm, cand_p, idx_s2, k, qpc, src_x2, topk_flat_hbm, s_tile_base_q + q0, s_tile_base_q + q_hi, consts
        )

        sbm.close_scope()
    sbm.close_scope()
