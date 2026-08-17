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

"""DeepSeek Sparse Attention Indexer — MX projections with a BF16 score matmul.

Hybrid variant that keeps the Q/K/W projections in ``nc_matmul_mx`` but scores
in bf16 (no Q/K quantization, no Hadamard rotation); see the kernel docstring
``Notes:`` for the equivalence derivation and rationale.
"""

import math
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.logging import get_logger
from ...core.utils.tensor_view import TensorView
from ...core.utils.tiled_range import TiledRange
from .sparse_attention_indexer_bf16_helpers import (
    fused_layernorm_rope_k,
)
from .sparse_attention_indexer_mx_bf16score_helpers import (
    persist_k_bf16,
    score_against_cache_bf16,
    transpose_k_for_cache,
)
from .sparse_attention_indexer_mx_helpers import (
    H_PACK,
    k_and_weights_projection_mx,
    load_wk_mx_weights,
    q_projection_mx,
    w_projection_mx_batch,
)
from .sparse_attention_indexer_utils import (
    NUM_TOPK_BATCHES,
    P_MAX,
    TOPK_QUERIES_PER_BATCH,
    TOPK_QUERY_CHUNKS,
    SAIConfig,
    round_up_to_chunks,
    topk_over_score,
    validate_sai_config,
)


@nki.jit
def sparse_attention_indexer_mx_bf16score(
    x: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    wk: nl.NkiTensor,
    k_norm_gamma: nl.NkiTensor,
    k_norm_beta: nl.NkiTensor,
    weights_proj: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    mask: nl.NkiTensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    index_topk: int,
    start_pos: int,
    use_hadamard: bool = False,
    batch_size: int = 1,
    # MX weight/scale for the Q/K/W projections.
    wq_b_scale: Optional[nl.NkiTensor] = None,
    wk_scale: Optional[nl.NkiTensor] = None,
    k_scale_cache: Optional[nl.NkiTensor] = None,  # ignored (bf16 K cache)
    x_non_mx: Optional[nl.NkiTensor] = None,  # pre-cast bf16 x for W-proj
    x_mx_data: Optional[nl.NkiTensor] = None,  # pre-quantized x for K-proj
    x_mx_scale: Optional[nl.NkiTensor] = None,
    qr_qtz_hbm: Optional[nl.NkiTensor] = None,  # pre-quantized qr (only Q path)
    qr_scale_hbm: Optional[nl.NkiTensor] = None,
    # Context-parallel phased split (see Args).
    phase: str = "all",
    k_seq_out_hbm: Optional[nl.NkiTensor] = None,
    end_pos_arg: Optional[int] = None,
    # topk output format (see Args); default = snake.
    emit_flat_topk: bool = False,
    emit_tiled_topk: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """DeepSeek Sparse Attention Indexer — MX projections + BF16 score.

    Supported/optimal usage: head_dim == 128; dim and q_lora_rank multiples of
    512; batch_size == 1 and S a multiple of P_MAX (=128) on the validated
    v3_long/h64 configs; LNC sharding degree up to 2.

    Dimensions:
        B: batch size (= batch_size).
        S: query sequence length per batch.
        M: B * S (total query rows, = x.shape[0]).
        dim: hidden size (= x.shape[1]).
        q_lora_rank: compressed-query rank (derived from qr_qtz_hbm layout).
        n_heads: number of indexer heads.
        head_dim: per-head dimension (== 128).
        rope_head_dim: RoPE-rotated slice of head_dim.
        end_pos: start_pos + S (KV positions scored per query).
        P_MAX: partition tile size (= 128).
        NUM_TOPK_BATCHES: topk calls per S-tile (= 16).

    Args:
        x: [B*S, dim] hidden states.
        qr_qtz_hbm: [num_s_tiles, P_MAX, q_lora_rank // (P_MAX*H_PACK), P_MAX]
            uint32 — pre-quantized (fp8x4-as-uint32) compressed query from the
            upstream QKV kernel.
        qr_scale_hbm: matching uint8 block-32 e8m0 scales for qr_qtz_hbm.
        wq_b: [q_lora_rank // 4, n_heads * head_dim] fp8_e4m3fn_x4.
        wq_b_scale: [q_lora_rank // 32, n_heads * head_dim] uint8 e8m0.
        wk: [dim // 4, head_dim] fp8_e4m3fn_x4 (contraction dim on partition).
        wk_scale: [dim // 32, head_dim] uint8 e8m0.
        k_norm_gamma, k_norm_beta: [head_dim] LayerNorm gamma/beta.
        weights_proj: [n_heads, dim] per-head weights projection (always bf16).
        cos, sin: [S, rope_head_dim // 2] RoPE caches.
        k_cache: [B, head_dim, max_seq_len] bf16 (mutable).
        k_scale_cache: ignored (no MX scales in bf16 path).
        mask: [B*S, end_pos] attention mask.
        index_topk: number of top positions to select per query (topk k).
        x_non_mx: pre-cast bf16 view of x for W-projection (skips the f32->bf16
            cast).
        x_mx_data, x_mx_scale: pre-quantized x for K-projection (skips the
            in-kernel swizzle + quantize_mx).
        qr_qtz_hbm, qr_scale_hbm: pre-quantized qr latent from an upstream QKV
            kernel (indexer skips its own qr transpose+norm+quantize). This is
            the only Q input path — qr is never quantized in-kernel.
        phase: context-parallel phased split.
            "all" (default): fused self-attention (queries == keys == S).
            "kproj": project K for this rank's S shard only; write seq-major
                [S, head_dim] to k_seq_out_hbm, return (that, None), no score.
            "score": skip K-proj; score this rank's S queries vs the pre-gathered
                full k_cache [B, head_dim, end_pos_arg=S_kv] over [0, S_kv).
        k_seq_out_hbm: [S, head_dim] bf16 K output for phase="kproj".
        end_pos_arg: key range for phase="score" (defaults to start_pos + S).
        emit_flat_topk: decode the snake topk to per-query positions in-kernel
            and return flat [B*S, index_topk] int32 (the sparse_mla_latent_attn
            topk_indices contract). Fuses the out-of-kernel snake->flat decode.
        emit_tiled_topk: split-fix output — skip the per-query scatter and return
            positions in the natural partition-tiled layout
            [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, index_topk // 16] int32, read
            per-query by the paired attention kernel. Mutually exclusive with
            emit_flat_topk; both default False (snake output, decoded by the
            consumer, e.g. the fused mega kernel).

    Returns:
        index_score: [B*S, end_pos] f32 — pre-topk per-position scores.
        topk_idx: [num_S_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk] uint32
            — hardware-topk output. ``topk_idx[s_tile, t, 16g + j%16, j//16]``
            holds the position (0-based in ``[0, end_pos)``) of query
            ``8t + g``'s j-th largest score (ascending). ``num_S_tiles_total =
            batch_size * ceil(S / P_MAX)``.

    Notes:
        - Q/K/W projections still use ``nc_matmul_mx`` for speed, but the score
          matmul uses bf16 nc_matmul (no Q/K quantization, no Hadamard
          rotation). This removes the Q swizzle + quantize_mx + Hadamard chain
          from the per-S-tile critical path.
        - K cache stores bf16 instead of fp8x4 (no need for MX format if
          downstream consumers use this same kernel; otherwise downstream must
          coordinate).
        - Algorithm equivalent to the MX variant via Hadamard's orthogonality:
          ``q_H @ (k_H).T = q @ H @ H.T @ k.T = q @ k.T`` so dropping Hadamard
          is mathematically a no-op. The MX quantization that Hadamard exists to
          support is also dropped on the score path. Score precision improves
          (bf16 score matmul vs MX-rounded score matmul).

    Pseudocode:
        for b in range(batch_size):
            q_full = q_projection_mx(qr_qtz_hbm, wq_b)  # LNC-sharded on output tiles
            weights_full = w_projection_mx_batch(x, weights_proj)
            for s_tile in S-tiles:
                k = k_and_weights_projection_mx(x, wk)  # per S-tile K projection
                k = fused_layernorm_rope_k(k)           # LayerNorm + RoPE
                write k to k_cache (bf16)
                score = score_against_cache_bf16(q_full, k_cache, weights_full)
                score += mask
                index_score <- score
                topk_idx <- topk_over_score(score, index_topk)  # hardware nisa.topk
    """
    M, dim = x.shape
    S = M // batch_size
    # qr is always supplied pre-quantized via qr_qtz_hbm (an upstream QKV kernel).
    # Derive q_lora_rank from its layout [num_s_tiles, P_MAX, num_K_tiles, P_MAX]
    # (num_K_tiles = q_lora_rank // (P_MAX * H_PACK)).
    q_lora_rank = qr_qtz_hbm.shape[2] * P_MAX * H_PACK
    end_pos = end_pos_arg if (phase == "score" and end_pos_arg is not None) else start_pos + S

    # bf16-score path: Hadamard is mathematically a no-op when there's no
    # MX quantization between rotation and matmul, so it's dropped here.
    cfg = SAIConfig(
        S=S,
        dim=dim,
        q_lora_rank=q_lora_rank,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_topk=index_topk,
        start_pos=start_pos,
        use_hadamard=False,
        batch_size=batch_size,
    )
    validate_sai_config(cfg, M)

    T = round_up_to_chunks(end_pos)
    num_s_tiles_per_batch = (S + P_MAX - 1) // P_MAX
    num_s_tiles_total = batch_size * num_s_tiles_per_batch

    total_out = n_heads * head_dim

    index_score_hbm = nl.ndarray((M, end_pos), dtype=nl.float32, buffer=nl.shared_hbm)
    """
    Hardware-topk output: per S-tile, per topk-batch (8 queries), snake-encoded
    top-index_topk position indices. topk_idx_hbm[s_tile, t, 16g + j%16, j//16]
    = position (0-based in [0, end_pos)) of query (8t+g)'s j-th largest score.
    """
    topk_idx_hbm = nl.ndarray(
        (num_s_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
    )
    # FLAT topk output (emit_flat_topk=True): decoded per-query cache positions
    # [B*S, index_topk] int32 — the sparse_mla_latent_attn_cte topk_indices
    # contract. Written directly by topk_over_score; the snake topk_idx_hbm is
    # then unused (kept allocated for the shared stage signature; DCE'd).
    topk_flat_hbm = nl.ndarray((M, index_topk), dtype=nl.int32, buffer=nl.shared_hbm) if emit_flat_topk else None
    # TILED topk output (emit_tiled_topk=True): decoded positions in the natural
    # partition tile [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, index_topk//16] int32.
    # Query (s_tile*128 + t*8 + g) owns partitions [16g,16g+16) x cols [0,K//16)
    # of block [s_tile, t]. Attention reads that [16, K//16] tile per query.
    topk_tiled_hbm = (
        nl.ndarray(
            (num_s_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk // TOPK_QUERY_CHUNKS),
            dtype=nl.int32,
            buffer=nl.shared_hbm,
        )
        if emit_tiled_topk
        else None
    )
    """
    Padded-score HBM staging buffer used by the topk relayout gather. Lives only
    for the duration of the kernel; the topk replacement will eliminate this
    HBM round-trip entirely.
    """
    score_padded_hbm = nl.ndarray(
        (num_s_tiles_total, P_MAX, T),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    """
    bf16 q_full_hbm so the chunk_loop's Q DMA delivers bf16 directly,
    skipping the per-chunk f32->bf16 cast on Scalar engine (was 128
    ops × 9 us = ~1.2 ms occupancy on Scalar, which is also the relu
    binder). Q-proj writes bf16 PSUM→SBUF tensor_copy upstream.
    """
    q_full_hbm = nl.ndarray((batch_size, S, total_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        logger=get_logger("SAI_mx"),
    )
    _sai_indexer_bf16score_stage(
        sbm,
        x,
        None,
        wq_b,
        wk,
        k_norm_gamma,
        k_norm_beta,
        weights_proj,
        cos,
        sin,
        k_cache,
        mask,
        n_heads,
        head_dim,
        rope_head_dim,
        index_topk,
        start_pos,
        batch_size,
        wq_b_scale,
        wk_scale,
        x_non_mx,
        x_mx_data,
        x_mx_scale,
        index_score_hbm,
        topk_idx_hbm,
        score_padded_hbm,
        q_full_hbm,
        qr_qtz_hbm=qr_qtz_hbm,
        qr_scale_hbm=qr_scale_hbm,
        end_pos=end_pos,
        phase=phase,
        k_seq_out_hbm=k_seq_out_hbm,
        topk_flat_hbm=topk_flat_hbm,
        topk_tiled_hbm=topk_tiled_hbm,
        # Seq-shard the score+topk phase across LNC cores by query (the performant
        # path). score[q,:] depends only on query q, so cores split disjoint query
        # rows / topk batches with no cross-core barrier; K-proj stays replicated.
        seq_shard_score=True,
    )
    # TILED (split-fix) -> partition-tiled positions; FLAT -> [B*S, index_topk];
    # else snake [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, index_topk].
    if emit_tiled_topk:
        topk_out = topk_tiled_hbm
    elif emit_flat_topk:
        topk_out = topk_flat_hbm
    else:
        topk_out = topk_idx_hbm
    if phase == "kproj":
        return k_seq_out_hbm, topk_out
    return index_score_hbm, topk_out


def _sai_indexer_bf16score_stage(
    sbm,
    x,
    qr,
    wq_b,
    wk,
    k_norm_gamma,
    k_norm_beta,
    weights_proj,
    cos,
    sin,
    k_cache,
    mask,
    n_heads,
    head_dim,
    rope_head_dim,
    index_topk,
    start_pos,
    batch_size,
    wq_b_scale,
    wk_scale,
    x_non_mx,
    x_mx_data,
    x_mx_scale,
    index_score_hbm,
    topk_idx_hbm,
    score_padded_hbm,
    q_full_hbm,
    qr_qtz_hbm=None,
    qr_scale_hbm=None,
    end_pos=None,
    phase="all",
    k_seq_out_hbm=None,
    topk_flat_hbm=None,
    topk_tiled_hbm=None,
    seq_shard_score=False,
):
    """SAI MX-projection + BF16-score body as an sbm-taking stage.

    Same logic as the ``sparse_attention_indexer_mx_bf16score`` entry, but driven by a
    caller-provided ``sbm`` and caller-allocated HBM outputs so a fused mega-kernel can
    invoke it as a composable stage (qkv -> indexer -> convert -> attn -> oproj). The
    @nki.jit entry above is a thin wrapper that allocates the HBM and calls this.

    ``end_pos`` decouples the KEY range from the local query count S: standalone self-
    attention uses end_pos = start_pos + S (default), but under Context Parallelism each
    rank scores its S LOCAL queries (at global offset start_pos = c*S) against the FULL
    gathered K sequence (end_pos = cp_degree * S) held in a full-width k_cache. The score
    path (score_against_cache_bf16, score_row [S, end_pos], k_cache load over end_pos)
    already handles S != end_pos; only this derivation was hardwired.

    ``phase`` splits the fused K-projection-and-score loop so a CP parent can gather K
    across ranks BETWEEN the two halves:
      * ``"all"`` (default): standalone / single-core — K-proj -> persist to k_cache ->
        score against (current + prior) cache, exactly as before.
      * ``"kproj"``: only project + LayerNorm + RoPE this rank's K. If ``k_seq_out_hbm``
        is given, write the seq-major ``[S, head_dim]`` K there (for a collective_dim=0
        HBM all-gather to ``[cp*S, head_dim]``) instead of the transposed k_cache; q/w
        projection, score, mask and topk are skipped.
      * ``"score"``: skip K-proj/persist; run q/w projection and score this rank's S local
        queries against the FULL pre-populated ``k_cache[head_dim, end_pos]`` (the parent
        transpose-loads the gathered K into it first), then mask + topk. The full-cache
        load path (``k_current_T_sb=None``) is used since no K is recomputed locally.
    """
    do_kproj = phase in ("all", "kproj")
    do_score = phase in ("all", "score")
    M, dim = x.shape
    S = M // batch_size
    # qr may be None on the pre-quantized path (qr_qtz_hbm supplied by an upstream stage);
    # then derive q_lora_rank from qr_qtz_hbm [num_s_tiles, P_MAX, num_K_tiles, P_MAX].
    if qr is not None:
        q_lora_rank = qr.shape[1]
    else:
        q_lora_rank = qr_qtz_hbm.shape[2] * P_MAX * H_PACK
    if end_pos is None:
        end_pos = start_pos + S
    rope_half = rope_head_dim // 2
    combined_scale = (1.0 / math.sqrt(n_heads)) * (1.0 / math.sqrt(head_dim))

    num_s_tiles_per_batch = (S + P_MAX - 1) // P_MAX

    _, num_shards, shard_id = get_verified_program_sharding_info("SAI", max_sharding=2)

    total_out = n_heads * head_dim
    PSUM_FMAX = 512
    num_out_tiles_all = (total_out + PSUM_FMAX - 1) // PSUM_FMAX
    q_num_shards = min(num_shards, num_out_tiles_all)
    q_shard_id = shard_id

    sbm.open_scope(name="sai_root")

    # K LayerNorm gamma/beta — only the K-projection phase needs them.
    gamma_sb = None
    beta_sb = None
    if do_kproj:
        gamma_sb = sbm.alloc_stack((P_MAX, head_dim), nl.float32)
        beta_sb = sbm.alloc_stack((P_MAX, head_dim), nl.float32)
        gamma_hbm_view = TensorView(k_norm_gamma.reshape((1, head_dim))).broadcast(dim=0, size=P_MAX)
        beta_hbm_view = TensorView(k_norm_beta.reshape((1, head_dim))).broadcast(dim=0, size=P_MAX)
        nisa.dma_copy(dst=gamma_sb, src=gamma_hbm_view.get_view())
        nisa.dma_copy(dst=beta_sb, src=beta_hbm_view.get_view())

    # No Hadamard needed in the bf16-score path: H @ H.T = I, so rotating
    # both Q and K and then doing a bf16 matmul is identical to the un-rotated
    # bf16 matmul. The MX entry needs Hadamard because FP8 quantization
    # benefits from magnitude redistribution; this entry skips quantization.
    kernel_assert(head_dim == 128, f"head_dim=128 only, got {head_dim}")

    for b_idx in nl.sequential_range(batch_size):
        b_offset = b_idx * S

        # Q-projection is only consumed by the score path. In the "kproj" phase
        # (CP: project this rank's K for the cross-rank gather) it is skipped.
        if do_score:
            sbm.open_scope(name="batch_q")
            # Q-proj is head-sharded (output-tile shard, full M=128 PE util) — this
            # beats query-sharding Q, which drops M to 64 (half array) and, being on
            # the critical path before the score loop, costs more than the cheap
            # cross-core barrier it would remove (measured: query-shard Q 219->242us).
            q_projection_mx(
                sbm,
                qr,
                b_offset,
                S,
                wq_b,
                wq_b_scale,
                q_lora_rank,
                n_heads,
                head_dim,
                num_shards=q_num_shards,
                shard_id=q_shard_id,
                q_out_hbm=q_full_hbm[b_idx, :, :],
                qr_qtz_hbm=qr_qtz_hbm,
                qr_scale_hbm=qr_scale_hbm,
            )
            sbm.close_scope()
        # NOTE: core_barrier deferred until just before chunk_loop reads
        # q_full_hbm. Q-proj is LNC-sharded; both cores must finish before
        # the chunk loop reads. Deferring the barrier lets W-proj batch
        # and wk hoist run in parallel with Q-proj completion on the
        # other core.

        """
        Hoist wk MX weights so the per-S-tile loop only re-quantizes
        activations and dispatches matmuls. The hoist allocation lives in a
        dedicated outer scope ("batch_b_hoist") so the inner per-S-tile
        ``increment_section`` doesn't reset the stack pointer back over the
        hoisted allocation. Putting the hoist in the same scope as the
        S-tile loop lets ``increment_section`` reset to the scope's
        starting addr, which clobbers the hoisted weights at S-tile 1
        (root cause of the multi-S-tile NaN we hit on TRN3).
        """
        sbm.open_scope(name="batch_b_hoist")
        # wk MX weights feed K-projection — only the K-proj phase needs them.
        wk_fp8_sb = None
        wk_scale_sb_loaded = None
        if do_kproj:
            wk_fp8_sb, wk_scale_sb_loaded = load_wk_mx_weights(
                sbm,
                wk,
                wk_scale,
                dim,
                head_dim,
            )
        # weights_proj hoist + batch W-projection feed the score's per-head
        # weighted accumulate — only the score phase needs them.
        wp_tile_T_hoist = None
        weights_full_sb = None
        if do_score:
            """
            Hoist weights_proj into SBUF in matmul-ready layout [P_MAX, num_dim_tiles, n_heads]
            bf16 with dim on partition. weights_proj HBM is bf16 so
            dma_transpose lands directly in bf16 SBUF — no staging f32
            buffer or per-dim-tile bf16 cast needed.
            """
            DIM_TILE_OUTER = P_MAX
            num_dim_tiles_outer = (dim + DIM_TILE_OUTER - 1) // DIM_TILE_OUTER
            wp_tile_T_hoist = sbm.alloc_stack(
                (P_MAX, num_dim_tiles_outer, n_heads),
                nl.bfloat16,
            )
            for dim_tile_idx in nl.affine_range(num_dim_tiles_outer):
                dim_start_outer = dim_tile_idx * DIM_TILE_OUTER
                dim_tile_sz_outer = min(DIM_TILE_OUTER, dim - dim_start_outer)
                nisa.dma_transpose(
                    dst=wp_tile_T_hoist[:dim_tile_sz_outer, dim_tile_idx, :n_heads],
                    src=weights_proj[:n_heads, dim_start_outer : dim_start_outer + dim_tile_sz_outer],
                )

            """
            Run W-projection at batch level (mirrors q_projection_mx pattern).
            Outputs all S-tiles' weights into weights_full_sb at once, so the
            per-S-tile loop only does K-projection and the score loop. This
            takes the W-proj nc_matmul row off the per-S-tile critical path.
            """
            weights_full_sb = sbm.alloc_stack(
                (P_MAX, num_s_tiles_per_batch, n_heads),
                nl.float32,
            )
            w_projection_mx_batch(
                sbm,
                x,
                b_offset,
                S,
                dim,
                n_heads,
                combined_scale,
                wp_tile_T_hoist=wp_tile_T_hoist,
                x_non_mx=x_non_mx,
                weights_full_sb=weights_full_sb,
            )

        # Sync Q-proj completion across LNC cores before chunk_loop reads
        # q_full_hbm. Deferring this from after Q-proj lets W-proj batch
        # and wk hoist execute concurrently with Q-proj completion on the
        # other LNC core. No-op / skipped at LNC degree 1 (core_barrier requires
        # >= 2 programs); single-core has no cross-core exchange to fence.
        if do_score and num_shards >= 2:
            nisa.core_barrier(q_full_hbm, (0, 1))

        """
        Double-buffer the S-tile loop so consecutive S-tiles don't reuse
        the same physical SBUF for k_sb / k_processed / score_row_sb /
        k_T_full_x4_sb / etc. This lets S-tile N+1's K-projection +
        layernorm + Hadamard overlap with S-tile N's score loop (the
        critical_dep_type=Engine chain we see in the profile breaks
        when those buffers are physically distinct).
        NOTE: interleave_degree=2 doubles per-S-tile SBUF buffers
        (score_row, mask, k_full, score_padded). For end_pos >= 4096
        this exceeds the 245KB/partition budget. Drop to il=1 for
        large S; smaller S benefits from il=2.
        """
        outer_il = 1 if end_pos >= 4096 else 2
        sbm.open_scope(interleave_degree=outer_il, name="batch_b")
        local_s_tile_idx = 0

        for s_tile in TiledRange(S, P_MAX):
            row_start = b_offset + s_tile.start_offset
            tile_size = s_tile.size
            cache_pos = start_pos + s_tile.start_offset

            cos_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32)
            sin_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32)
            nisa.dma_copy(
                dst=cos_sb[:tile_size, :rope_half],
                src=cos[s_tile.start_offset : s_tile.start_offset + tile_size, :rope_half],
            )
            nisa.dma_copy(
                dst=sin_sb[:tile_size, :rope_half],
                src=sin[s_tile.start_offset : s_tile.start_offset + tile_size, :rope_half],
            )

            # ---- K-projection phase: project + LayerNorm + RoPE this rank's K. ----
            if do_kproj:
                k_sb = sbm.alloc_stack((P_MAX, head_dim), nl.float32)
                # skip_w=True: W-projection ran at batch level (score phase) or is
                # not needed at all (kproj-only phase); this call is K-proj only.
                k_and_weights_projection_mx(
                    sbm,
                    x,
                    row_start,
                    tile_size,
                    wk_fp8_sb,
                    wk_scale_sb_loaded,
                    weights_proj,
                    dim,
                    head_dim,
                    n_heads,
                    combined_scale,
                    x_non_mx=x_non_mx,
                    x_mx_data=x_mx_data,
                    x_mx_scale=x_mx_scale,
                    wp_tile_T_hoist=wp_tile_T_hoist,
                    s_tile_idx=local_s_tile_idx + b_idx * num_s_tiles_per_batch,
                    k_sb=k_sb,
                    weights_sb=None,
                    skip_w=True,
                )

                # No K Hadamard: K stays as LayerNorm + RoPE'd bf16.
                k_processed = sbm.alloc_stack((P_MAX, head_dim), nl.float32)
                fused_layernorm_rope_k(
                    sbm, k_sb, gamma_sb, beta_sb, cos_sb, sin_sb, k_processed, tile_size, head_dim, rope_head_dim
                )
                # Cast f32 K to bf16 for cache writeback + score matmul.
                k_processed_bf16 = sbm.alloc_stack((P_MAX, head_dim), nl.bfloat16)
                nisa.tensor_copy(
                    dst=k_processed_bf16[:tile_size, :head_dim],
                    src=k_processed[:tile_size, :head_dim],
                    engine=nisa.engine.scalar,
                )

                if k_seq_out_hbm is not None:
                    # CP K-proj: emit seq-major [S, head_dim] K so the parent can
                    # HBM all-gather (collective_dim=0) into [cp*S, head_dim]. No
                    # transpose needed — this is the natural [S P, head_dim F] layout.
                    nisa.dma_copy(
                        dst=k_seq_out_hbm[row_start : row_start + tile_size, :head_dim],
                        src=k_processed_bf16[:tile_size, :head_dim],
                    )
                else:
                    # Standalone / "all" phase: transpose to [head_dim P, S F] and
                    # persist to the transposed k_cache. Shared with the score matmul.
                    k_T_current_sb = transpose_k_for_cache(
                        sbm,
                        k_processed_bf16,
                        S=tile_size,
                        head_dim=head_dim,
                    )
                    persist_k_bf16(
                        sbm,
                        k_T_current_sb,
                        k_cache,
                        b_idx=b_idx,
                        cache_pos=cache_pos,
                        S=tile_size,
                        head_dim=head_dim,
                    )

            # ---- Score phase: q RoPE + per-head bf16 score vs the K cache. ----
            if do_score:
                # Seq-shard the score+topk across LNC cores: each core handles a
                # contiguous half of this S-tile's queries (core c -> local rows
                # [c*sc_S, c*sc_S + sc_S)). K-proj stays REPLICATED (both cores build
                # the full k_cache) since every query scores against ALL keys. The
                # score row for a query depends only on that query (no cross-query
                # reduction), so no cross-core sync is needed. The current kernel
                # (seq_shard_score=False) runs the whole tile on both cores redundantly.
                if seq_shard_score and num_shards >= 2:
                    sc_S = (tile_size + num_shards - 1) // num_shards
                    sc_start = shard_id * sc_S
                    sc_S = max(0, min(sc_S, tile_size - sc_start))
                else:
                    sc_S = tile_size
                    sc_start = 0

                if sc_S > 0:
                    # cos/sin and weights for THIS core's query rows must live at
                    # partition 0 (score_against_cache_bf16 allocates a fresh 0-based
                    # q_chunk_sb and RoPE-broadcasts cos against it). weights_full_sb is
                    # 0-based per row, so re-DMA a 0-based cos/sin slice for [sc_start,
                    # sc_start+sc_S); a partition-offset SBUF view (cos_sb[sc_start:])
                    # would misalign the RoPE tensor_tensor base partitions.
                    if seq_shard_score and num_shards >= 2:
                        cos_shard_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32)
                        sin_shard_sb = sbm.alloc_stack((P_MAX, rope_half), nl.float32)
                        nisa.dma_copy(
                            dst=cos_shard_sb[:sc_S, :rope_half],
                            src=cos[s_tile.start_offset + sc_start : s_tile.start_offset + sc_start + sc_S, :rope_half],
                        )
                        nisa.dma_copy(
                            dst=sin_shard_sb[:sc_S, :rope_half],
                            src=sin[s_tile.start_offset + sc_start : s_tile.start_offset + sc_start + sc_S, :rope_half],
                        )
                        # weights_full_sb rows are query-local; core rows [sc_start,+sc_S)
                        # need to land at partition 0 for the 0-based score. Copy them.
                        weights_sb = sbm.alloc_stack((P_MAX, n_heads), nl.float32)
                        nisa.tensor_copy(
                            dst=weights_sb[:sc_S, :n_heads],
                            src=weights_full_sb[sc_start : sc_start + sc_S, local_s_tile_idx, :n_heads],
                            engine=nisa.engine.scalar,
                        )
                    else:
                        cos_shard_sb = cos_sb
                        sin_shard_sb = sin_sb
                        weights_sb = weights_full_sb[:P_MAX, local_s_tile_idx, :n_heads]
                    score_row_sb = sbm.alloc_stack((P_MAX, end_pos), nl.bfloat16)
                    q_hbm_tile_view = q_full_hbm[
                        b_idx, s_tile.start_offset + sc_start : s_tile.start_offset + sc_start + sc_S, :total_out
                    ]

                    # "all" phase forwards the just-transposed current K (avoids the
                    # HBM-write-commit-before-read race + a redundant load). CP "score"
                    # phase has no local K (k_cache is fully pre-populated by the gather),
                    # so load the full cache from HBM (k_current_T_sb=None).
                    if phase == "all":
                        score_against_cache_bf16(
                            sbm,
                            q_hbm_tile_view,
                            cos_shard_sb,
                            sin_shard_sb,
                            k_cache,
                            weights_sb=weights_sb,
                            score_row_sb=score_row_sb,
                            b_idx=b_idx,
                            S=sc_S,
                            end_pos=end_pos,
                            n_heads=n_heads,
                            head_dim=head_dim,
                            rope_head_dim=rope_head_dim,
                            k_current_T_sb=k_T_current_sb,
                            current_start=cache_pos,
                            current_size=tile_size,
                        )
                    else:
                        score_against_cache_bf16(
                            sbm,
                            q_hbm_tile_view,
                            cos_shard_sb,
                            sin_shard_sb,
                            k_cache,
                            weights_sb=weights_sb,
                            score_row_sb=score_row_sb,
                            b_idx=b_idx,
                            S=sc_S,
                            end_pos=end_pos,
                            n_heads=n_heads,
                            head_dim=head_dim,
                            rope_head_dim=rope_head_dim,
                        )

                    mask_sb = sbm.alloc_stack((P_MAX, end_pos), nl.bfloat16)
                    nisa.dma_copy(
                        dst=mask_sb[:sc_S, :end_pos],
                        src=mask[row_start + sc_start : row_start + sc_start + sc_S, :end_pos],
                    )
                    nisa.tensor_tensor(
                        dst=score_row_sb[:sc_S, :end_pos],
                        data1=score_row_sb[:sc_S, :end_pos],
                        data2=mask_sb[:sc_S, :end_pos],
                        op=nl.add,
                    )

                    nisa.dma_copy(
                        dst=index_score_hbm[row_start + sc_start : row_start + sc_start + sc_S, :end_pos],
                        src=score_row_sb[:sc_S, :end_pos],
                    )

                    # Hardware topk over the score row. Each core writes its queries
                    # to their global partition rows of the s-tile's score_padded slot
                    # (part_offset=sc_start) and runs only its own topk batches
                    # (batch_start..batch_end over the 8-query groups it owns).
                    global_s_tile_idx = b_idx * num_s_tiles_per_batch + local_s_tile_idx
                    b_start = sc_start // TOPK_QUERIES_PER_BATCH
                    b_end = (sc_start + sc_S + TOPK_QUERIES_PER_BATCH - 1) // TOPK_QUERIES_PER_BATCH
                    topk_over_score(
                        sbm,
                        score_row_sb,
                        sc_S,
                        end_pos,
                        index_topk,
                        score_padded_hbm=score_padded_hbm,
                        topk_idx_hbm=topk_idx_hbm,
                        s_tile_idx=global_s_tile_idx,
                        topk_flat_hbm=topk_flat_hbm,
                        s_tile_base_q=global_s_tile_idx * P_MAX,
                        topk_tiled_hbm=topk_tiled_hbm,
                        batch_start=b_start,
                        batch_end=b_end,
                        part_offset=sc_start,
                    )

            local_s_tile_idx += 1
            sbm.increment_section()

        sbm.close_scope()  # batch_b
        sbm.close_scope()  # batch_b_hoist

    sbm.close_scope()
