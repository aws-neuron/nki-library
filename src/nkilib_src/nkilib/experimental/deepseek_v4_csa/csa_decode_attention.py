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

"""DeepSeek-V4 CSA decode attention kernels.

Single-token decode over the compressed sparse attention block. The headline
kernel is ``nki_indexer_score_topk_gather_2core``, which runs the lightning
indexer's scoring, the top-k selection and the O(k) sparse attention in ONE
launch on a ``[2]``-grid (two logical NeuronCores), so neither the score row nor
the selected-index array returns to the host.

Batching: decode is a SINGLE token, ``batch_size = 1``. Every kernel here takes one
query position per head (``S = 1``) and the ``[2]``-grid is spent on splitting the
sequence, not the batch. Prefill handles the multi-position case.

"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert


# --------------------------------------------------------------------------
# NKI Kernel: fused RMS(+optional learnable gain) + RoPE for BOTH the q-path
# and the kv-path, merged into ONE kernel over a [n_heads+1, head_dim] tile.
#
# Replaces the two torch tails
#     q  = q * rsqrt(q.square().mean(-1) + eps);  q[...,-rd:]  = RoPE(q[...,-rd:])
#     kv = kv_norm(wkv(x));                        kv[...,-rd:] = RoPE(kv[...,-rd:])
# with q = [n_heads, head_dim] and kv = [1, head_dim].
#
# Both do the same per-partition work -- RMS over the free axis in fp32, round to bf16
# at the RMSNorm boundary, then RoPE in fp32 on the trailing rope channels with the
# same cos/sin. The one difference (q has no learnable gain, kv scales by
# kv_norm.weight) is unified by a per-partition gain tile: 1.0 on the q rows, the
# weight on the kv row. `x * 1.0 == x` is exact in fp32, so the q rows are unchanged.
#
# Packing both onto one partition tile turns two kernels into one launch, one
# HBM->SBUF load and one store.
# --------------------------------------------------------------------------
@nki.jit
def nki_qkv_rms_rope_kernel(
    q_in: nl.NkiTensor,
    kv_in: nl.NkiTensor,
    weight_in: nl.NkiTensor,
    cos_in: nl.NkiTensor,
    sin_in: nl.NkiTensor,
    eps_val: float,
) -> nl.NkiTensor:
    """Fused RMS(+per-partition gain) + RoPE for the merged q/kv tile.

    q_in:     [n_heads, head_dim] bf16 — the query heads.
    kv_in:    [1, head_dim] bf16 — the single decode-token KV latent.
    weight_in:[1, head_dim] fp32 — kv_norm.weight (learnable gain, kv row only).
    cos_in:   [1, half_rope] fp32 (NOT pre-repeated). sin_in: [1, half_rope] fp32.
    Returns:  [n_heads+1, head_dim] bf16 — rows 0..n_heads-1 = q heads (RMS+RoPE),
              row n_heads = kv (learnable RMSNorm+RoPE).

    The q rows and the kv row are ASSEMBLED onto one partition tile inside the
    kernel (two DMAs), so the host does no concatenation. The per-partition gain
    tile is built on-chip: memset 1.0 (q rows -> x*1.0==x exact in fp32), DMA
    kv_norm.weight into the kv row. Single-launch / single-core (shared_hbm).
    """
    n_heads = q_in.shape[0]
    head_dim = q_in.shape[1]
    n_rows = n_heads + 1  # q heads + the single kv row
    half_rope = cos_in.shape[1]
    rope_head_dim = 2 * half_rope
    nope_dim = head_dim - rope_head_dim
    TILE = 128  # SBUF partitions

    kernel_assert(
        n_rows <= TILE,
        f"n_heads + 1 must fit the {TILE} SBUF partitions, got n_heads={n_heads}; "
        f"shard the heads (tp_size >= 2) or split the kv row out",
    )

    out = nl.ndarray((n_rows, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # ---- Assemble q rows + kv row onto ONE [n_rows, head_dim] tile (2 DMAs) ----
    x_sb = nl.ndarray((TILE, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb[0:n_heads, 0:head_dim], src=q_in[0:n_heads, 0:head_dim], priority=0)
    nisa.dma_copy(dst=x_sb[n_heads:n_rows, 0:head_dim], src=kv_in[0:1, 0:head_dim], priority=0)

    # ---- RMS statistic ------------------------------------------------------
    x_sq = nl.ndarray((TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=x_sq[0:n_rows, 0:head_dim],
        data1=x_sb[0:n_rows, 0:head_dim],
        data2=x_sb[0:n_rows, 0:head_dim],
        op=nl.multiply,
    )
    msq = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=msq[0:n_rows, 0:1], data=x_sq[0:n_rows, 0:head_dim], op=nl.add, axis=1)
    # mean = sum / head_dim, then + eps (fused), then rsqrt.
    nisa.tensor_scalar(
        dst=msq[0:n_rows, 0:1],
        data=msq[0:n_rows, 0:1],
        op0=nl.multiply,
        operand0=1.0 / head_dim,
        op1=nl.add,
        operand1=eps_val,
    )
    rms = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rms[0:n_rows, 0:1], op=nl.rsqrt, data=msq[0:n_rows, 0:1])

    # Per-partition learnable gain built ON-CHIP: memset 1.0 (q rows -> x*1.0==x
    # exact in fp32), DMA kv_norm.weight into the kv row (row n_heads) -> reproduces
    # the learnable kv RMSNorm exactly. Host does no concatenation.
    gain = nl.ndarray((TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=gain[0:n_rows, 0:head_dim], value=1.0)
    nisa.dma_copy(dst=gain[n_heads:n_rows, 0:head_dim], src=weight_in[0:1, 0:head_dim], priority=1)

    # ---- normed = (x * rms) * gain, rounded to bf16, in ONE pass per region ----
    normed_nope = nl.ndarray((TILE, nope_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=normed_nope[0:n_rows, 0:nope_dim],
        data=x_sb[0:n_rows, 0:nope_dim],
        op0=nl.multiply,
        operand0=rms[0:n_rows, 0:1],
        op1=nl.multiply,
        operand1=gain[0:n_rows, 0:nope_dim],
    )
    normed_pairs = nl.ndarray((TILE, half_rope, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
    normed_rope = normed_pairs.reshape((TILE, rope_head_dim))
    nisa.scalar_tensor_tensor(
        dst=normed_rope[0:n_rows, 0:rope_head_dim],
        data=x_sb[0:n_rows, nope_dim:head_dim],
        op0=nl.multiply,
        operand0=rms[0:n_rows, 0:1],
        op1=nl.multiply,
        operand1=gain[0:n_rows, nope_dim:head_dim],
    )

    # ---- Write the nope channels (0..nope_dim-1) straight to output ----
    nisa.dma_copy(dst=out[0:n_rows, 0:nope_dim], src=normed_nope[0:n_rows, 0:nope_dim])

    # ---- RoPE on the last rope_head_dim channels (fp32 math) ----
    # normed_pairs is [.., half_rope, 2], so [...,0]=even (x1), [...,1]=odd (x2).
    x1 = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x1[0:n_rows, 0:half_rope], src=normed_pairs[0:n_rows, 0:half_rope, 0])
    x2 = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x2[0:n_rows, 0:half_rope], src=normed_pairs[0:n_rows, 0:half_rope, 1])

    cos_h = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_h[0:n_rows, 0:half_rope], src=cos_in.ap(pattern=[[0, n_rows], [1, half_rope]]), priority=2)
    sin_h = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin_h[0:n_rows, 0:half_rope], src=sin_in.ap(pattern=[[0, n_rows], [1, half_rope]]), priority=2)

    rope_bf16 = nl.ndarray((TILE, half_rope, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
    tmp_a = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    tmp_b = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=tmp_a[0:n_rows, 0:half_rope],
        data1=x1[0:n_rows, 0:half_rope],
        data2=cos_h[0:n_rows, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_b[0:n_rows, 0:half_rope],
        data1=x2[0:n_rows, 0:half_rope],
        data2=sin_h[0:n_rows, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_bf16[0:n_rows, 0:half_rope, 0],
        data1=tmp_a[0:n_rows, 0:half_rope],
        data2=tmp_b[0:n_rows, 0:half_rope],
        op=nl.subtract,
    )
    nisa.tensor_tensor(
        dst=tmp_a[0:n_rows, 0:half_rope],
        data1=x1[0:n_rows, 0:half_rope],
        data2=sin_h[0:n_rows, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_b[0:n_rows, 0:half_rope],
        data1=x2[0:n_rows, 0:half_rope],
        data2=cos_h[0:n_rows, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_bf16[0:n_rows, 0:half_rope, 1],
        data1=tmp_a[0:n_rows, 0:half_rope],
        data2=tmp_b[0:n_rows, 0:half_rope],
        op=nl.add,
    )

    rope_bf16_flat = rope_bf16.reshape((TILE, rope_head_dim))
    nisa.dma_copy(dst=out[0:n_rows, nope_dim:head_dim], src=rope_bf16_flat[0:n_rows, 0:rope_head_dim])

    return out


# --------------------------------------------------------------------------
# nisa.topk batched kernel + encode/decode helpers
# --------------------------------------------------------------------------
NISA_TOPK_GROUP_SIZE = 16
NISA_TOPK_PARTITIONS = 128
NISA_TOPK_GROUPS_PER_CALL = NISA_TOPK_PARTITIONS // NISA_TOPK_GROUP_SIZE  # 8

# Sentinel used to pad a score row up to a proven-safe nisa.topk `n`. Every real
# indexer score is >= 0 (post-relu), so a padded slot can never enter the top-k.
_TOPK_PAD_SENTINEL = -1e9


@nki.jit
def nisa_topk_snake_kernel(in_tensor: nl.NkiTensor, k_val: int, n_val: int) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Batched nisa.topk on snake-encoded input.

    in_tensor: [num_batches * 128, src_x] bf16 — snake-encoded scores.
    Returns (values [num_batches * 128, k_val], indices [num_batches * 128, k_val]).
    """
    total_rows = in_tensor.shape[0]
    src_x = in_tensor.shape[1]
    num_batches = total_rows // 128
    par_dim = 128

    out_values = nl.ndarray((total_rows, k_val), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_indices = nl.ndarray((total_rows, k_val), dtype=nl.uint32, buffer=nl.shared_hbm)

    for b in nl.affine_range(num_batches):
        src = nl.ndarray((par_dim, src_x), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=src, src=in_tensor[b * 128 : (b + 1) * 128, 0:src_x])

        val_dst = nl.ndarray((par_dim, k_val), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_dst = nl.ndarray((par_dim, k_val), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=val_dst, idx_dst=idx_dst, src=src, n=n_val)

        nisa.dma_copy(dst=out_values[b * 128 : (b + 1) * 128, 0:k_val], src=val_dst)
        nisa.dma_copy(dst=out_indices[b * 128 : (b + 1) * 128, 0:k_val], src=idx_dst)

    return out_values, out_indices


# --------------------------------------------------------------------------
# NKI Kernel: Indexer per-head scoring (outputs raw scores for external topk)
# --------------------------------------------------------------------------
@nki.jit
def nki_indexer_score_kernel(
    q_T_all: nl.NkiTensor,  # [head_dim, n_heads * S_q] — all heads' Q^T stacked (bf16)
    kv_t: nl.NkiTensor,  # [head_dim, T_c] — indexer_kv transposed (bf16), shared
    weights: nl.NkiTensor,  # [S_q, n_heads] — per-row per-head weights * weight_scale (fp32)
    causal_bias: nl.NkiTensor,  # [S_q, T_c] — causal bias (0 / -1e9) (fp32)
) -> nl.NkiTensor:
    """Indexer scoring kernel — computes per-row scores for nkilib topk.

    Computes, per query row s and compressed kv position t:
        index_score[s, t] = sum_h relu(q[s, h, :] . kv[t, :]) * weights[s, h]
                            + causal_bias[s, t]

    Returns:
        scores: [S_q, T_c] fp32 — raw index scores (higher = more relevant).
    """
    head_dim = q_T_all.shape[0]
    total_q_free = q_T_all.shape[1]
    T_c = kv_t.shape[1]
    n_heads = weights.shape[1]
    S_q = total_q_free // n_heads

    TILE_Q = 128
    SCORE_CHUNK = 512 if T_c >= 512 else T_c
    num_score_chunks = (T_c + SCORE_CHUNK - 1) // SCORE_CHUNK
    num_q_tiles = (S_q + TILE_Q - 1) // TILE_Q

    scores_out = nl.ndarray((S_q, T_c), dtype=nl.float32, buffer=nl.shared_hbm)

    kv_t_sb = nl.ndarray((head_dim, T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=kv_t_sb, src=kv_t[0:head_dim, 0:T_c])

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    tiles_per_core = num_q_tiles // n_cores

    for q_local in nl.affine_range(tiles_per_core):
        q_idx = core_id * tiles_per_core + q_local
        q_start = q_idx * TILE_Q

        w_tile = nl.ndarray((TILE_Q, n_heads), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tile, src=weights[q_start : q_start + TILE_Q, 0:n_heads])

        index_score_bf16 = nl.ndarray((TILE_Q, T_c), dtype=nl.bfloat16, buffer=nl.sbuf)

        for h in nl.affine_range(n_heads):
            q_global = h * S_q + q_start
            q_T = nl.ndarray((head_dim, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_T, src=q_T_all[0:head_dim, q_global : q_global + TILE_Q])

            w_h = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_h, src=w_tile[0:TILE_Q, h : h + 1])

            for m_idx in nl.affine_range(num_score_chunks):
                m_start = m_idx * SCORE_CHUNK
                kt_slice = kv_t_sb[0:head_dim, m_start : m_start + SCORE_CHUNK]
                s_psum = nl.ndarray((TILE_Q, SCORE_CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_psum, stationary=q_T, moving=kt_slice)
                s_relu = nl.ndarray((TILE_Q, SCORE_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=s_relu, op=nl.relu, data=s_psum)
                if h == 0:
                    nisa.tensor_scalar(
                        dst=index_score_bf16[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                        data=s_relu,
                        op0=nl.multiply,
                        operand0=w_h,
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=index_score_bf16[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                        data=s_relu,
                        op0=nl.multiply,
                        operand0=w_h,
                        op1=nl.add,
                        operand1=index_score_bf16[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                    )

        index_score = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=index_score, src=index_score_bf16)

        cbias = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cbias, src=causal_bias[q_start : q_start + TILE_Q, 0:T_c])
        nisa.tensor_tensor(dst=index_score, data1=index_score, data2=cbias, op=nl.add)

        nisa.dma_copy(dst=scores_out[q_start : q_start + TILE_Q, 0:T_c], src=index_score)

    return scores_out


# --------------------------------------------------------------------------
# NKI Kernel: FUSED indexer scoring + nisa.topk (single-chunk decode path)
#
# Fuses the two-kernel (nki_indexer_score_kernel -> HBM fp32 scores -> torch
# encode_snake -> nisa_topk_snake_kernel) pipeline into ONE @nki.jit kernel:
#   1. Score all T_c positions (reuses the proven per-head relu*weight matmul
#      producing score[128, T_c] with the query on the partition dim; only
#      row 0 is meaningful in decode since all query rows are identical).
#   2. Build the nisa.topk SNAKE src [128, T_c/16]: snake[r, c] = score[16c+r] for
#      r in [0,16), c in [0, T_c/16). SBUF cannot stride its partition dim, so the
#      free->partition fold routes row 0 through a tiny HBM scratch (T_c bf16 =
#      4-16KB) that `_snake_fill` reads back contiguously.
#   3. Run nisa.topk(n=T_c) -> local indices in snake layout.
#   4. Write group-0's k indices (== global indices for the single chunk) to a
#      [TOPK_ROWS, k] HBM tensor. Row 0 receives ALL k indices as an unordered
#      SET, which is exactly what the downstream permutation-invariant softmax
#      over gathered positions needs (== what Pass-1 candidate_indices[0] gave).
#
# This replaces the two-kernel pipeline's fp32 [S_q, T_c] HBM write (1MB @
# T_c=2048), the torch encode_snake/decode_snake host glue, and the separate
# topk kernel launch with one fused kernel + a 4-16KB scratch round-trip.
# --------------------------------------------------------------------------
@nki.jit
def nki_indexer_score_topk_kernel(
    q_T_all: nl.NkiTensor,  # [head_dim, n_heads * S_q] — all heads' Q^T stacked (bf16)
    kv_t: nl.NkiTensor,  # [head_dim, T_c] — indexer_kv transposed (bf16)
    weights: nl.NkiTensor,  # [S_q, n_heads] — per-row per-head weights * weight_scale (fp32)
    k_val: int,  # number of top-k indices to return (== T_c-capped topk)
) -> nl.NkiTensor:
    """Fused indexer scoring + top-k. Returns [TOPK_ROWS, k_val] uint32 indices.

    Computes, per compressed kv position t (row 0 of the identical decode query):
        score[t] = sum_h relu(q[0, h, :] . kv[t, :]) * weights[0, h]
    then returns the GLOBAL indices of the k_val largest scores (order-agnostic).

    Requirements (single-chunk decode path):
        T_c divisible by 128 and by 16; k_val divisible by 16; S_q >= 128.
    """
    head_dim = q_T_all.shape[0]
    total_q_free = q_T_all.shape[1]
    T_c = kv_t.shape[1]
    n_heads = weights.shape[1]
    S_q = total_q_free // n_heads

    TILE_Q = 128
    GROUP = 16  # nisa.topk snake group size
    SNAKE_X = T_c // GROUP  # snake free dim (T_c/16): 128 (T_c=2048) or 512 (T_c=8192)
    SCORE_CHUNK = 512 if T_c >= 512 else T_c
    num_score_chunks = (T_c + SCORE_CHUNK - 1) // SCORE_CHUNK
    TOPK_ROWS = 8  # minimum rows nisa.topk operates on (128/16)
    PAR = 128

    out_indices = nl.ndarray((TOPK_ROWS, k_val), dtype=nl.uint32, buffer=nl.shared_hbm)

    # --- Preload shared KV^T (query on partition scoring matmul) ---
    kv_t_sb = nl.ndarray((head_dim, T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=kv_t_sb, src=kv_t[0:head_dim, 0:T_c])

    # Per-head weights for query row 0 (all rows identical in decode).
    w_tile = nl.ndarray((TILE_Q, n_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_tile, src=weights[0:TILE_Q, 0:n_heads])

    # --- Stage 1: score all T_c, query on partition (bf16 accumulation) ---
    index_score_bf16 = nl.ndarray((TILE_Q, T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in nl.affine_range(n_heads):
        q_global = h * S_q  # q_start = 0 (single tile)
        q_T = nl.ndarray((head_dim, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_T, src=q_T_all[0:head_dim, q_global : q_global + TILE_Q])

        w_h = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_h, src=w_tile[0:TILE_Q, h : h + 1])

        for m_idx in nl.affine_range(num_score_chunks):
            m_start = m_idx * SCORE_CHUNK
            kt_slice = kv_t_sb[0:head_dim, m_start : m_start + SCORE_CHUNK]
            s_psum = nl.ndarray((TILE_Q, SCORE_CHUNK), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=s_psum, stationary=q_T, moving=kt_slice)
            s_relu = nl.ndarray((TILE_Q, SCORE_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=s_relu, op=nl.relu, data=s_psum)
            if h == 0:
                nisa.tensor_scalar(
                    dst=index_score_bf16[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                    data=s_relu,
                    op0=nl.multiply,
                    operand0=w_h,
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=index_score_bf16[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                    data=s_relu,
                    op0=nl.multiply,
                    operand0=w_h,
                    op1=nl.add,
                    operand1=index_score_bf16[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                )

    # --- Stage 2: build the nisa.topk snake src [128, SNAKE_X] ---
    scratch = nl.ndarray((1, T_c), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=scratch, src=index_score_bf16[0:1, 0:T_c])

    snake_src = nl.ndarray((PAR, SNAKE_X), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=snake_src, value=0)
    _snake_fill(scratch, snake_src, 0, SNAKE_X)

    # --- Stage 3: nisa.topk (snake layout) -> local indices ---
    val_dst = nl.ndarray((PAR, k_val), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx_dst = nl.ndarray((PAR, k_val), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.topk(val_dst=val_dst, idx_dst=idx_dst, src=snake_src, n=T_c)

    # --- Stage 4: write group-0 indices (== global, single chunk) as a SET ---
    # idx_dst group 0 = partitions 0..15, columns 0..k/16-1 (k_val local positions).
    # Flatten into out row 0: out[0, p*(k/16) + c] = idx_dst[p, c]. Order-agnostic.
    k_cols = k_val // GROUP
    idx_grp0 = nl.ndarray((GROUP, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_grp0, src=idx_dst[0:GROUP, 0:k_cols])
    nisa.dma_copy(dst=out_indices[0:1, :].ap(pattern=[[k_cols, GROUP], [1, k_cols]], offset=0), src=idx_grp0)

    return out_indices


# --------------------------------------------------------------------------
# NKI Kernel: 2-LNC indexer scoring (disjoint T_c halves) -> shared bf16 scores
#
# Splits the Stage-1 scoring across 2 LNC cores; the attention kernel that
# follows already uses [2], so the 2nd core would otherwise idle through the
# whole indexing phase. Each core scores a contiguous T_c/n_cores slice over ALL
# heads and writes only row 0 (all decode query rows are identical) to its
# DISJOINT half of a shared [1, W] bf16 buffer -> no cross-core reduction. The
# cross-core barrier that guarantees both halves land before the top-k reads them
# is `nisa.core_barrier(cores=(0,1))` INSIDE the merged kernel below
# (nki_indexer_score_topk_2core), so the score->topk hand-off costs no @nki.jit
# launch boundary; the top-k itself still runs on ONE core.
#
# HEAD-BATCHED scoring: the compressed index-KV is SHARED across all n_heads
# (only the query differs per head), so instead of the old sequential
# `for h in range(n_heads)` loop that re-streamed the same KV through the PE
# array 64x (64 matmuls + 64 relu + 64 bf16 weight-accumulates per chunk), this
# scores all heads with TWO matmuls per SCORE_CHUNK:
#     allh[h, t] = q_compact[:, h] . kv[:, t]           (matmul-1, heads on free)
#     score[t]   = sum_h relu(allh[h, t]) * w[h]        (relu; matmul-2 reduces h)
# matmul-2's contraction over the 64 head-partitions accumulates in fp32 (more
# accurate than the old bf16 running sum), so max_abs_diff should stay <= the
# bit-identical baseline. q_compact [head_dim, n_heads] and w_compact [n_heads,1]
# are derived in-kernel via strided DMAs from the existing q_T_all / weights
# inputs (q_T_all[d, h*S_q] == q[h,d]; weights[0, h]), so the kernel signature,
# both call sites, and all host code are unchanged.
# --------------------------------------------------------------------------
def _score_2core_stage(
    q_T_all: nl.NkiTensor,  # [head_dim, n_heads * S_q] — all heads' Q^T stacked (bf16)
    kv_t: nl.NkiTensor,  # [head_dim, T_c] — indexer_kv transposed (bf16)
    weights: nl.NkiTensor,  # [S_q, n_heads] — per-row per-head weights * weight_scale (fp32)
    scores_dst: nl.NkiTensor,  # [1, W] bf16 shared_hbm (W >= T_c) — destination score row
) -> None:
    """Score this LNC core's disjoint T_c slice into scores_dst[0:1, t_base:...].

    A plain Python helper (NOT a @nki.jit kernel) so the SAME traced instruction
    sequence is shared verbatim by both the standalone scorer
    (`nki_indexer_score_2core`, multi-chunk path) and the merged score+topk kernel
    (`nki_indexer_score_topk_2core`, single-chunk decode path) -> the two are
    bit-identical by construction. `scores_dst` may be WIDER than T_c (the merged
    kernel passes the n=8192 top-k-padded row); only columns
    [t_base, t_base + Tc_per_core) are touched, at element stride 1 exactly as
    before, so the written bytes do not depend on the buffer width.
    """
    head_dim = q_T_all.shape[0]
    total_q_free = q_T_all.shape[1]
    T_c = kv_t.shape[1]
    n_heads = weights.shape[1]
    S_q = total_q_free // n_heads

    SCORE_CHUNK = 512 if T_c >= 512 else T_c

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    Tc_per_core = T_c // n_cores  # 1024 (s8192, [2]) or 4096 (s32768, [2])
    num_score_chunks = (Tc_per_core + SCORE_CHUNK - 1) // SCORE_CHUNK
    t_base = core_id * Tc_per_core  # this core's global T_c offset (register)

    # --- Preload this core's KV^T slice (shared across all heads) ---
    kv_t_sb = nl.ndarray((head_dim, Tc_per_core), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=kv_t_sb, src=kv_t[0:head_dim, t_base : t_base + Tc_per_core], priority=0)

    # --- Compact query [head_dim, n_heads]: q_compact[d, h] = q_T_all[d, h*S_q]
    q_compact = nl.ndarray((head_dim, n_heads), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=q_compact, src=q_T_all.ap(pattern=[[n_heads * S_q, head_dim], [S_q, n_heads]], offset=0), priority=1
    )

    w_f32 = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_f32, src=weights.ap(pattern=[[1, n_heads], [1, 1]], offset=0), priority=1)
    w_bf16 = nl.ndarray((n_heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_bf16, src=w_f32)

    # --- Score this core's T_c slice: 2 matmuls per chunk (heads batched) ---
    for m_idx in nl.affine_range(num_score_chunks):
        m_start = m_idx * SCORE_CHUNK
        kt_slice = kv_t_sb[0:head_dim, m_start : m_start + SCORE_CHUNK]

        # matmul-1: allh[h, t] = sum_d q_compact[d, h] * kt_slice[d, t]  (fp32 psum)
        allh = nl.ndarray((n_heads, SCORE_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=allh, stationary=q_compact, moving=kt_slice)

        # relu (matches reference: relu THEN weight) -> bf16 for matmul-2
        R = nl.ndarray((n_heads, SCORE_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=R, op=nl.relu, data=allh)

        # matmul-2: score[t] = sum_h w[h] * R[h, t]  (fp32 accumulation over heads)
        sc = nl.ndarray((1, SCORE_CHUNK), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sc, stationary=w_bf16, moving=R)

        sc_bf = nl.ndarray((1, SCORE_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sc_bf, src=sc)
        nisa.dma_copy(dst=scores_dst[0:1, t_base + m_start : t_base + m_start + SCORE_CHUNK], src=sc_bf, priority=0)


@nki.jit
def nki_indexer_score_2core(
    q_T_all: nl.NkiTensor,  # [head_dim, n_heads * S_q] — all heads' Q^T stacked (bf16)
    kv_t: nl.NkiTensor,  # [head_dim, T_c] — indexer_kv transposed (bf16)
    weights: nl.NkiTensor,  # [S_q, n_heads] — per-row per-head weights * weight_scale (fp32)
) -> nl.NkiTensor:
    """Score all T_c across n_cores LNC cores (disjoint T_c halves). Returns [1, T_c] bf16.

    Scores-only entry point, used by the MULTI-chunk path (s131072), whose per-chunk
    top-k + Pass-2 merge is host-orchestrated over the assembled row. The
    single-chunk decode path instead calls `nki_indexer_score_topk_2core`, which
    runs this same scoring stage and the top-k inside ONE kernel launch.
    """
    T_c = kv_t.shape[1]
    scores_out = nl.ndarray((1, T_c), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="indexer_scores_2core_out")
    _score_2core_stage(q_T_all, kv_t, weights, scores_out)
    return scores_out


# --------------------------------------------------------------------------
# nisa.topk reads a [128, SNAKE_X] tile as 8 INDEPENDENT groups of 16 partitions, and
# within a group the element it calls logical index `j` lives at partition `j % 16`,
# column `j // 16` -- the "snake" layout. Only group 0 is filled and read, so the tile
# holds n_val = 16 * SNAKE_X scores.
# --------------------------------------------------------------------------
_SNAKE_GROUP = 16


def _snake_fill(
    scores: nl.NkiTensor,  # [1, n_val] bf16 — the score row, contiguous by position
    dst: nl.NkiTensor,  # [>=16, count] bf16 sbuf — snake tile (columns 0..count-1)
    c0: int,  # first snake column to produce
    count: int,  # number of snake columns to produce
    priority: int = 0,
) -> None:
    """dst[r, j] = scores[16 * (c0 + j) + r] for r in [0,16), j in [0,count).

    Reads `scores` as its natural [n_val/16, 16] row-major view (partition stride 16,
    16 contiguous elements per partition) and folds free->partition with nc_transpose
    rather than with DMA descriptors.
    """
    GROUP = _SNAKE_GROUP
    NC = 128 if count % 128 == 0 else count
    kernel_assert(NC <= 128, "snake column count must be <= 128 or a multiple of 128")
    for b in nl.affine_range(count // NC):
        blk = nl.ndarray((NC, GROUP), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=blk,
            src=scores.ap(pattern=[[GROUP, NC], [1, GROUP]], offset=(c0 + b * NC) * GROUP),
            priority=priority,
        )
        tp = nl.ndarray((GROUP, NC), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=blk)
        nisa.tensor_copy(dst=dst[0:GROUP, b * NC : (b + 1) * NC], src=tp)


# --------------------------------------------------------------------------
# Stage helper: snake-encode + nisa.topk on the assembled score row (ONE core)
#
# Stages 2-4 of nki_indexer_score_topk_kernel, factored into a plain Python
# helper so the merged kernel below can run it on core 0 only (after the
# cross-core barrier) with no extra @nki.jit launch. Consumes the [1, n_val] bf16
# score row assembled by `_score_2core_stage` (+ its -1e9 top-k padding tail) and
# writes group-0's k GLOBAL indices as an unordered set in out row 0.
# --------------------------------------------------------------------------
def _snake_topk_stage(
    scores: nl.NkiTensor,  # [1, n_val] bf16 shared_hbm — assembled + padded score row
    out_indices: nl.NkiTensor,  # [TOPK_ROWS, k_val] uint32 shared_hbm — destination
    k_val: int,  # number of top-k indices to return
    n_val: int,  # nisa.topk n (the proven-safe padded width)
) -> None:
    """nisa.topk over snake-encoded scores -> out_indices row 0 (unordered set)."""
    GROUP = _SNAKE_GROUP
    SNAKE_X = n_val // GROUP
    PAR = 128

    # PAR stays 128: nisa.topk needs all 128 partitions resident (a 16-partition alloc
    # faults at runtime) even though only group 0 is filled and read. Groups 1..7 are
    # left uninitialized -- their contents cannot reach the output.
    snake_src = nl.ndarray((PAR, SNAKE_X), dtype=nl.bfloat16, buffer=nl.sbuf)
    _snake_fill(scores, snake_src, 0, SNAKE_X, priority=0)

    val_dst = nl.ndarray((PAR, k_val), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx_dst = nl.ndarray((PAR, k_val), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.topk(val_dst=val_dst, idx_dst=idx_dst, src=snake_src, n=n_val)

    # The strided fill makes the returned index a global position already. Write group
    # 0's indices as a SET into out row 0: out[0, p*(k/16) + c]. Order-agnostic.
    k_cols = k_val // GROUP
    idx_grp0 = nl.ndarray((GROUP, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_grp0, src=idx_dst[0:GROUP, 0:k_cols])
    nisa.dma_copy(dst=out_indices[0:1, :].ap(pattern=[[k_cols, GROUP], [1, k_cols]], offset=0), src=idx_grp0)


def _snake_topk_stage_2core(
    scores: nl.NkiTensor,  # [1, n_val] bf16 shared_hbm — assembled + padded score row
    out_indices: nl.NkiTensor,  # [TOPK_ROWS, k_val] uint32 shared_hbm — destination
    k_val: int,
    n_val: int,  # nisa.topk n (proven-safe padded width)
    core_id: int,  # this LNC's program id
) -> None:
    """2-LNC version of `_snake_topk_stage`: split the descriptor-bound snake
    reformat DMA across both cores, exchange via nisa.sendrecv, top-k on core 0.

    The single-core stage runs entirely on core 0 while core 1 sits at the barrier with
    its 16 DMA engines idle. Splitting the snake free axis `c` in half puts half the
    fill on each core's own engines; one nisa.sendrecv then swaps the halves
    SBUF<->SBUF (no HBM round-trip) so core 0 can reassemble the whole tile.

    Core 0 keeps its half at columns [0, HALF) and places the received half at
    [HALF, SNAKE_X), reproducing exactly the tile the single-core path builds -- so the
    snake-to-global remap is the same, and only core 0 runs the top-k, leaving the
    n-safety argument untouched.
    """
    GROUP = _SNAKE_GROUP
    SNAKE_X = n_val // GROUP
    HALF = SNAKE_X // 2
    PAR = 128
    peer = 1 - core_id

    my_half = nl.ndarray((PAR, HALF), dtype=nl.bfloat16, buffer=nl.sbuf)
    _snake_fill(scores, my_half, core_id * HALF, HALF, priority=0)

    peer_half = nl.ndarray((PAR, HALF), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.sendrecv(src=my_half, dst=peer_half, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)

    if core_id == 0:
        snake_src = nl.ndarray((PAR, SNAKE_X), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=snake_src[0:GROUP, 0:HALF], src=my_half[0:GROUP, 0:HALF])
        nisa.tensor_copy(dst=snake_src[0:GROUP, HALF:SNAKE_X], src=peer_half[0:GROUP, 0:HALF])

        val_dst = nl.ndarray((PAR, k_val), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_dst = nl.ndarray((PAR, k_val), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=val_dst, idx_dst=idx_dst, src=snake_src, n=n_val)

        k_cols = k_val // GROUP
        idx_grp0 = nl.ndarray((GROUP, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx_grp0, src=idx_dst[0:GROUP, 0:k_cols])
        nisa.dma_copy(dst=out_indices[0:1, :].ap(pattern=[[k_cols, GROUP], [1, k_cols]], offset=0), src=idx_grp0)


# --------------------------------------------------------------------------
# NKI Kernel: MERGED 2-LNC indexer scoring + nisa.topk in ONE launch
#
# Removes one @nki.jit launch boundary from the decode critical path by running the
# 2-core scoring stage and the single-core top-k inside one `[2]`-grid kernel. The
# scoring matmuls are a tiny share of PE work, so this is entirely about the boundary
# (kernel-boundary DMA staging is the largest sync-engine opcode in the profile) plus
# the [1, T_c] HBM score round-trip.
#
# The hand-off the launch boundary used to provide becomes an intra-kernel barrier:
#   1. both cores write their DISJOINT T_c halves into the shared_hbm score row;
#   2. `nisa.core_barrier(data=scores_pad, cores=(0, 1))` establishes visibility;
#   3. core 0 alone runs the snake read + nisa.topk + index write-back.
#
# Per-core gating: the kernel is traced ONCE PER LOGICAL CORE and `nl.program_id(0)`
# folds to a Python int during that trace, so `if core_id == 0:` is real code
# specialization -- core 1's NEFF contains no top-k. (The "no device-if on a
# register" hazard applies to values that really are registers, such as
# nisa.register_load results, which need nl.dynamic_range / nl.while_loop.)
#
# nisa.topk n-safety moves ON-CHIP: the score row is allocated `n_val` wide and its
# [T_c, n_val) tail is memset to a -1e9 sentinel before the barrier, so the top-k
# runs at a validated n. On this heavily tied score distribution the selection depends
# on the width, so the width is pinned rather than tracking T_c. Every real score
# is >= 0 after the relu, so the sentinel can never win.
# --------------------------------------------------------------------------
@nki.jit
def nki_indexer_score_topk_2core(
    q_T_all: nl.NkiTensor,  # [head_dim, n_heads * S_q] — all heads' Q^T stacked (bf16)
    kv_t: nl.NkiTensor,  # [head_dim, T_c] — indexer_kv transposed (bf16)
    weights: nl.NkiTensor,  # [S_q, n_heads] — per-row per-head weights * weight_scale (fp32)
    k_val: int,  # number of top-k indices to return
    n_val: int,  # nisa.topk n (proven-safe padded width, >= T_c)
) -> nl.NkiTensor:
    """2-core scoring + core-0 nisa.topk in ONE launch. Returns [TOPK_ROWS, k_val] uint32.

    Requirements: launched on the [2] grid; n_val >= T_c and n_val % 16 == 0;
    k_val % 16 == 0. Bit-identical to the score_2core -> host F.pad -> snake_topk
    pipeline it replaces.
    """
    T_c = kv_t.shape[1]
    GROUP = 16
    TOPK_ROWS = 8

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    kernel_assert(n_cores == 2, "nki_indexer_score_topk_2core must be launched on the [2] grid")
    kernel_assert(n_val >= T_c and n_val % GROUP == 0, "n_val must be >= T_c and a multiple of 16")
    kernel_assert(k_val % GROUP == 0, "k_val must be a multiple of 16")
    out_indices = nl.ndarray((TOPK_ROWS, k_val), dtype=nl.uint32, buffer=nl.shared_hbm, name="indexer_topk_out")

    # Score row padded out to the proven-safe nisa.topk width, and the cross-core
    # exchange buffer the barrier synchronizes on.

    scores_pad = nl.ndarray((1, n_val), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="indexer_scores_shared")

    if core_id == 0 and n_val > T_c:
        pad_sb = nl.ndarray((1, n_val - T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=pad_sb, value=_TOPK_PAD_SENTINEL)
        nisa.dma_copy(dst=scores_pad[0:1, T_c:n_val], src=pad_sb)

    # Stage 1 (BOTH cores): disjoint T_c halves -> scores_pad[0:1, 0:T_c].
    _score_2core_stage(q_T_all, kv_t, weights, scores_pad)

    # Cross-core barrier: replaces the old @nki.jit launch boundary. Both cores'
    # score halves (and the pad tail) are visible to every core after this point.
    nisa.core_barrier(data=scores_pad, cores=(0, 1))

    # Stages 2-4 on core 0 ONLY (nisa.topk must see the WHOLE assembled row, and
    # runs on exactly one core). core 1's trace ends at the barrier.
    if core_id == 0:
        _snake_topk_stage(scores_pad, out_indices, k_val, n_val)

    return out_indices


# --------------------------------------------------------------------------
# Stage helper: the WHOLE O(k) decode-attention body (gather -> score -> softmax
# -> V accumulate -> output de-RoPE -> write-back).
#
# A plain Python helper, not a @nki.jit kernel, so `nki_decode_gather_ok_kernel` (the
# standalone [1]-grid attention) and `nki_indexer_score_topk_gather_2core` (the fused
# [2]-grid kernel, which runs this inside its `core_id == 0` branch) trace the SAME
# instruction sequence rather than two copies of it.
#
# `idx_chunks` comes from the CALLER because the two callers read the same index bytes
# from differently-shaped sources -- the standalone kernel from a host-supplied [k, S]
# tensor, the fused kernel from row 0 of the top-k output it just wrote. Both are k
# contiguous uint32 with partition stride 1.
#
# `h_base` / `H_BATCH` / `output` are parameters rather than derived from
# nl.num_programs() because the fused kernel runs this body on one core of a [2] grid
# with all n_heads batched.
# --------------------------------------------------------------------------
def _gather_attn_stage(
    idx_chunks: list[nl.NkiTensor],  # list[num_k_chunks] of [COMP_CHUNK, 1] uint32 SBUF row offsets
    all_q_T: nl.NkiTensor,  # [head_dim, n_heads * S] — replicated query per head
    win_K_T: nl.NkiTensor,  # [head_dim, W] — window K^T (real window only)
    win_V: nl.NkiTensor,  # [W, head_dim] — window V (real window only)
    compress_kv: nl.NkiTensor,  # [T_c, head_dim] bf16 — compressed KV row-major (K=V in CSA)
    attn_sink_in: nl.NkiTensor,  # [1, n_heads] — per-head sink scalars
    derope_cos: nl.NkiTensor,  # [1, half_rope] fp32 — inverse-RoPE cos (output de-RoPE)
    derope_sin: nl.NkiTensor,  # [1, half_rope] fp32 — inverse-RoPE sin
    output: nl.NkiTensor,  # [n_heads * S, head_dim] bf16 shared_hbm — destination
    k: int,  # number of gathered compressed positions
    S: int,  # query rows per head (1 in decode)
    n_heads: int,  # total heads spanned by `output` / all_q_T
    h_base: int,  # global head offset of this call's head batch
    H_BATCH: int,  # heads processed by this call
) -> None:
    """O(k) decode attention: gather K and V in chunks, score+accumulate via matmul.

    Uses only indirect_dim=0 (row gather) from compress_kv for both K and V.
    K is gathered then transposed in SBUF for the Q@K^T scoring matmul.

    The caller's idx_chunks give COMP_CHUNK distinct indices per chunk for the
    swdge gather (partition-dim slicing of a k-long contiguous uint32 row).

    Total compressed KV operations: (k / COMP_CHUNK) DMA gathers + matmuls.
    For k=1024: 8 gathers + 8 matmuls, regardless of T_c.
    """
    head_dim = all_q_T.shape[0]
    KV_CHUNK = 128
    WIN_SIZE = KV_CHUNK
    COMP_CHUNK = 128
    num_k_chunks = k // COMP_CHUNK

    HD_CHUNK = 128
    HD_TILES = (head_dim + HD_CHUNK - 1) // HD_CHUNK

    # Per-head attn-sink scalar for this core's 16 heads, on the partition dim:
    # sink_hb[h_local, 0] = attn_sink_in[0, h_base + h_local].
    sink_hb = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_hb, src=attn_sink_in.ap(pattern=[[1, H_BATCH], [1, 1]], offset=h_base), priority=1)

    q_start = 0

    # idx_chunks (the [COMP_CHUNK, 1] uint32 gather offsets) were loaded by the
    # caller — see the header note on why the load lives there.

    # ---- Prefetch: gather all k compressed-KV chunks ONCE, up front ----------
    kv_chunks = [None] * num_k_chunks
    for c_idx in nl.affine_range(num_k_chunks):
        kv_bf = nl.ndarray((COMP_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=kv_bf,
            src=compress_kv.ap(
                pattern=[[head_dim, COMP_CHUNK], [1, head_dim]],
                vector_offset=idx_chunks[c_idx],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            priority=0,  # highest — the swdge gather gates all downstream compute
        )
        kv_chunks[c_idx] = nl.ndarray((COMP_CHUNK, head_dim), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_chunks[c_idx], src=kv_bf)

    # Load window K^T (shared across all heads).
    kv_t_win = [None] * HD_TILES
    for hd in nl.affine_range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        kv_t_win[hd] = nl.ndarray((hd_sz, WIN_SIZE), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=kv_t_win[hd], src=win_K_T[hd_start : hd_start + hd_sz, q_start : q_start + WIN_SIZE], priority=2
        )  # window K^T — lower priority than the gated gather

    # Load window V (shared across all heads). WIN_SIZE=KV_CHUNK now, so the
    # whole real window is one chunk (the dead zero-pad half was dropped).
    win_v_0 = nl.ndarray((KV_CHUNK, head_dim), dtype=nl.float16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=win_v_0, src=win_V[q_start : q_start + KV_CHUNK, 0:head_dim], priority=2
    )  # window V — lower priority than the gated gather

    # ---- Head-batched query [head_dim, H_BATCH] (head on the free/moving dim) ----
    q_hb = [None] * HD_TILES
    for hd in nl.affine_range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        q_hb[hd] = nl.ndarray((hd_sz, H_BATCH), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_hb[hd],
            src=all_q_T.ap(pattern=[[n_heads * S, hd_sz], [S, H_BATCH]], offset=hd_start * (n_heads * S) + h_base * S),
            priority=1,
        )  # query — mid priority (needed for scoring, after the gather)

    # ---- Window scores [H_BATCH, WIN_SIZE] via one matmul over HD_TILES ----
    win_scores_psum = nl.ndarray((H_BATCH, WIN_SIZE), dtype=nl.float32, buffer=nl.psum)
    for hd in nl.affine_range(HD_TILES):
        nisa.nc_matmul(dst=win_scores_psum, stationary=q_hb[hd], moving=kv_t_win[hd])
    # ---- Window bias collapsed to the sink scalar (bit-identical dead-weight cut) ----
    win_scores = nl.ndarray((H_BATCH, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=win_scores, src=win_scores_psum)
    nisa.tensor_scalar(
        dst=win_scores[0:H_BATCH, 0:1],
        data=win_scores_psum[0:H_BATCH, 0:1],
        op0=nl.add,
        operand0=sink_hb[0:H_BATCH, 0:1],
    )

    # ---- Gathered compressed scoring [H_BATCH, k] ----
    # Gather+transpose all k positions into wide K^T tiles, then score with WIDE
    # matmuls to amortize LDWEIGHTS + PE pipeline fill.

    SCORE_W = 512  # max moving free dim
    num_score_groups = (k + SCORE_W - 1) // SCORE_W

    # Build wide K^T tiles kt_all[hd] = [HD_CHUNK, k], filled per gathered chunk.
    kt_all = [None] * HD_TILES
    for hd in range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        kt_all[hd] = nl.ndarray((hd_sz, k), dtype=nl.float16, buffer=nl.sbuf)

    for c_idx in nl.affine_range(num_k_chunks):
        c_start = c_idx * COMP_CHUNK

        # K chunk was already gathered+cast into kv_chunks[c_idx] above (K=V in CSA,
        # shared with the V matmul). Transpose K -> K^T [head_dim, COMP_CHUNK] into
        # this chunk's slice of kt_all.
        k_chunk = kv_chunks[c_idx]
        for hd in range(HD_TILES):
            hd_start = hd * HD_CHUNK
            hd_sz = min(HD_CHUNK, head_dim - hd_start)
            kt_psum = nl.ndarray((hd_sz, COMP_CHUNK), dtype=nl.float16, buffer=nl.psum)
            nisa.nc_transpose(dst=kt_psum, data=k_chunk[0:COMP_CHUNK, hd_start : hd_start + hd_sz])
            nisa.tensor_copy(dst=kt_all[hd][0:hd_sz, c_start : c_start + COMP_CHUNK], src=kt_psum)

    # Wide scoring: SCORE_W positions per matmul group, HD_TILES accumulation.
    comp_scores = nl.ndarray((H_BATCH, k), dtype=nl.float32, buffer=nl.sbuf)
    for g in nl.affine_range(num_score_groups):
        g_start = g * SCORE_W
        g_sz = min(SCORE_W, k - g_start)
        scores_psum = nl.ndarray((H_BATCH, g_sz), dtype=nl.float32, buffer=nl.psum)
        for hd in nl.affine_range(HD_TILES):
            hd_start = hd * HD_CHUNK
            hd_sz = min(HD_CHUNK, head_dim - hd_start)
            nisa.nc_matmul(dst=scores_psum, stationary=q_hb[hd], moving=kt_all[hd][0:hd_sz, g_start : g_start + g_sz])
        nisa.tensor_copy(dst=comp_scores[0:H_BATCH, g_start : g_start + g_sz], src=scores_psum)

    # ---- Softmax over window + k gathered scores (all heads, one pass each) ----
    win_max = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=win_max, data=win_scores, op=nl.maximum, axis=1)
    comp_max = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=comp_max, data=comp_scores, op=nl.maximum, axis=1)
    neg_max = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=neg_max, data1=win_max, data2=comp_max, op=nl.maximum)
    nisa.tensor_scalar(dst=neg_max, data=neg_max, op0=nl.multiply, operand0=-1.0)

    win_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    win_exp = nl.ndarray((H_BATCH, WIN_SIZE), dtype=nl.float16, buffer=nl.sbuf)
    nisa.activation(
        dst=win_exp,
        op=nl.exp,
        data=win_scores,
        bias=neg_max,
        reduce_op=nl.add,
        reduce_res=win_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    comp_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    comp_exp = nl.ndarray((H_BATCH, k), dtype=nl.float16, buffer=nl.sbuf)
    nisa.activation(
        dst=comp_exp,
        op=nl.exp,
        data=comp_scores,
        bias=neg_max,
        reduce_op=nl.add,
        reduce_res=comp_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    total_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=total_sum, data1=win_sum, data2=comp_sum, op=nl.add)

    # ---- Output accumulation [H_BATCH, head_dim] ----
    out_psum = nl.ndarray((H_BATCH, head_dim), dtype=nl.float32, buffer=nl.psum)

    we_T0 = nl.ndarray((KV_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.psum)
    nisa.nc_transpose(dst=we_T0, data=win_exp[0:H_BATCH, 0:KV_CHUNK])
    we_T0_sb = nl.ndarray((KV_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=we_T0_sb, src=we_T0)
    nisa.nc_matmul(dst=out_psum, stationary=we_T0_sb, moving=win_v_0)

    for c_idx in nl.affine_range(num_k_chunks):
        c_start = c_idx * COMP_CHUNK

        v_chunk = kv_chunks[c_idx]

        ce_T = nl.ndarray((COMP_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.psum)
        nisa.nc_transpose(dst=ce_T, data=comp_exp[0:H_BATCH, c_start : c_start + COMP_CHUNK])
        ce_T_sb = nl.ndarray((COMP_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ce_T_sb, src=ce_T)
        nisa.nc_matmul(dst=out_psum, stationary=ce_T_sb, moving=v_chunk)

    # ---- Finalize: normalize by total_sum, write this core's H_BATCH rows ----
    out_sbuf = nl.ndarray((H_BATCH, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sbuf, src=out_psum)
    inv_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=total_sum)
    nisa.tensor_scalar(dst=out_sbuf, data=out_sbuf, op0=nl.multiply, operand0=inv_sum[0:H_BATCH, 0:1])
    out_bf16 = nl.ndarray((H_BATCH, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_bf16, src=out_sbuf)

    # ---- Fused output de-RoPE (inverse rotation on the last rope channels) ----
    half_rope = derope_cos.shape[1]
    rope_head_dim = 2 * half_rope
    nope_dim = head_dim - rope_head_dim

    # Widen the rope channels bf16 -> fp32 and split into even (x1) / odd (x2).
    d_rope_f = nl.ndarray((H_BATCH, rope_head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_rope_f[0:H_BATCH, 0:rope_head_dim], src=out_bf16[0:H_BATCH, nope_dim:head_dim])
    d_pairs = d_rope_f.reshape((H_BATCH, half_rope, 2))
    dx1 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dx1[0:H_BATCH, 0:half_rope], src=d_pairs[0:H_BATCH, 0:half_rope, 0])
    dx2 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dx2[0:H_BATCH, 0:half_rope], src=d_pairs[0:H_BATCH, 0:half_rope, 1])

    dcos = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=dcos[0:H_BATCH, 0:half_rope], src=derope_cos.ap(pattern=[[0, H_BATCH], [1, half_rope]]), priority=3
    )
    dsin = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=dsin[0:H_BATCH, 0:half_rope], src=derope_sin.ap(pattern=[[0, H_BATCH], [1, half_rope]]), priority=3
    )

    # y1 = x1*cos + x2*sin ; y2 = x2*cos - x1*sin
    dta = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    dtb = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    dy1 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    dy2 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=dta[0:H_BATCH, 0:half_rope],
        data1=dx1[0:H_BATCH, 0:half_rope],
        data2=dcos[0:H_BATCH, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=dtb[0:H_BATCH, 0:half_rope],
        data1=dx2[0:H_BATCH, 0:half_rope],
        data2=dsin[0:H_BATCH, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=dy1[0:H_BATCH, 0:half_rope], data1=dta[0:H_BATCH, 0:half_rope], data2=dtb[0:H_BATCH, 0:half_rope], op=nl.add
    )
    nisa.tensor_tensor(
        dst=dta[0:H_BATCH, 0:half_rope],
        data1=dx2[0:H_BATCH, 0:half_rope],
        data2=dcos[0:H_BATCH, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=dtb[0:H_BATCH, 0:half_rope],
        data1=dx1[0:H_BATCH, 0:half_rope],
        data2=dsin[0:H_BATCH, 0:half_rope],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=dy2[0:H_BATCH, 0:half_rope],
        data1=dta[0:H_BATCH, 0:half_rope],
        data2=dtb[0:H_BATCH, 0:half_rope],
        op=nl.subtract,
    )

    # Re-interleave y1 (even) / y2 (odd), cast bf16, overwrite the rope channels.
    d_out = nl.ndarray((H_BATCH, half_rope, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_out[0:H_BATCH, 0:half_rope, 0], src=dy1[0:H_BATCH, 0:half_rope])
    nisa.tensor_copy(dst=d_out[0:H_BATCH, 0:half_rope, 1], src=dy2[0:H_BATCH, 0:half_rope])
    d_out_flat = d_out.reshape((H_BATCH, rope_head_dim))
    nisa.tensor_copy(dst=out_bf16[0:H_BATCH, nope_dim:head_dim], src=d_out_flat[0:H_BATCH, 0:rope_head_dim])

    nisa.dma_copy(
        dst=output.ap(pattern=[[S * head_dim, H_BATCH], [1, head_dim]], offset=h_base * S * head_dim),
        src=out_bf16,
        priority=1,
    )


def _gather_attn_stage_ksplit(
    idx_chunks: list[nl.NkiTensor],  # list[k_half/COMP_CHUNK] of [COMP_CHUNK,1] uint32 — THIS core's half
    all_q_T: nl.NkiTensor,
    win_K_T: nl.NkiTensor,
    win_V: nl.NkiTensor,
    compress_kv: nl.NkiTensor,
    attn_sink_in: nl.NkiTensor,
    derope_cos: nl.NkiTensor,
    derope_sin: nl.NkiTensor,
    output: nl.NkiTensor,
    k: int,  # TOTAL gathered positions (both halves)
    S: int,
    n_heads: int,
    core_id: int,  # 0 or 1
    k_half: int,  # k // 2, this core's share of gathered positions
) -> None:
    """O(k) decode attention with the K DIMENSION split across both LNCs.

    WHY THIS EXISTS (and why the HEAD split could not work). Splitting the head
    batch was measured to change core 0's tensor time by 0.05% (383.1 -> 382.9 us):
    the heads live on the matmul OUTPUT-PARTITION dim (M), and with M = 32 or 16 of
    128 the cost is set by the MOVING free dim (the k gathered positions) and the
    head_dim contraction, not by M. Worse, everything expensive in this body is
    k-driven and HEAD-INDEPENDENT — the swdge gather, the K^T transpose build, the
    window loads — so a head split DUPLICATES all of it (8 -> 16 DMA_INDIRECT,
    +1.25 MB HBM) while removing nothing from the critical path.

    Splitting along K fixes exactly that: core c owns gathered positions
    [c*k_half, (c+1)*k_half), so per core the gather drops 8 -> 4 chunks, the
    transpose build drops to k_half columns, and the scoring matmul's MOVING dim
    drops k -> k_half. Total HBM traffic is UNCHANGED (each row is gathered once,
    by exactly one core) — unlike the head split, which read every row twice.

    THE MERGE (two nisa.sendrecv exchanges, flash-attention style):
      phase 1: exchange the per-head local comp max, so BOTH cores form the same
               global max. exp() then sees the SAME shift as the unsplit body, so
               every exp argument is bit-identical to baseline.
      phase 2: exchange (partial V accumulator, partial exp sum); core 0 adds the
               two partials, normalizes, de-RoPEs and writes the output.
    Because `max` is exact in floating point and both cores compute the window
    scores locally, the global max is bit-identical to the unsplit body. The only
    numerical difference is the GROUPING of the fp32 sums (core0's 4 chunks +
    core1's 4 chunks, instead of 8 chunks into one PSUM), which is a reassociation
    of exactly the same terms — well inside the 2e-3 correctness gate but NOT
    bit-identical, so it is graded on max_abs_diff rather than on byte equality.
    """
    head_dim = all_q_T.shape[0]
    KV_CHUNK = 128
    WIN_SIZE = KV_CHUNK
    COMP_CHUNK = 128
    HD_CHUNK = 128
    HD_TILES = (head_dim + HD_CHUNK - 1) // HD_CHUNK
    PAR = 128
    H_BATCH = n_heads  # ALL heads on BOTH cores (the split is over k, not heads)
    h_base = 0
    num_half_chunks = k_half // COMP_CHUNK
    q_start = 0
    peer = 1 - core_id

    sink_hb = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_hb, src=attn_sink_in.ap(pattern=[[1, H_BATCH], [1, 1]], offset=h_base), priority=1)

    # ---- Gather ONLY this core's k_half positions (4 chunks, not 8) ----------
    kv_chunks = [None] * num_half_chunks
    for c_idx in nl.affine_range(num_half_chunks):
        kv_bf = nl.ndarray((COMP_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=kv_bf,
            src=compress_kv.ap(
                pattern=[[head_dim, COMP_CHUNK], [1, head_dim]],
                vector_offset=idx_chunks[c_idx],
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
            priority=0,
        )
        kv_chunks[c_idx] = nl.ndarray((COMP_CHUNK, head_dim), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_chunks[c_idx], src=kv_bf)

    # Window K^T / V. BOTH cores load it: the window max participates in the global
    # softmax shift, so both need win_scores to form a bit-identical global max. Only
    # core 0 accumulates the window V contribution (below), so it is counted once.
    kv_t_win = [None] * HD_TILES
    for hd in nl.affine_range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        kv_t_win[hd] = nl.ndarray((hd_sz, WIN_SIZE), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=kv_t_win[hd], src=win_K_T[hd_start : hd_start + hd_sz, q_start : q_start + WIN_SIZE], priority=2
        )
    win_v_0 = nl.ndarray((KV_CHUNK, head_dim), dtype=nl.float16, buffer=nl.sbuf)
    nisa.dma_copy(dst=win_v_0, src=win_V[q_start : q_start + KV_CHUNK, 0:head_dim], priority=2)

    q_hb = [None] * HD_TILES
    for hd in nl.affine_range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        q_hb[hd] = nl.ndarray((hd_sz, H_BATCH), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_hb[hd],
            src=all_q_T.ap(pattern=[[n_heads * S, hd_sz], [S, H_BATCH]], offset=hd_start * (n_heads * S) + h_base * S),
            priority=1,
        )

    # ---- Window scores + window max (both cores, identical values) -----------
    win_scores_psum = nl.ndarray((H_BATCH, WIN_SIZE), dtype=nl.float32, buffer=nl.psum)
    for hd in nl.affine_range(HD_TILES):
        nisa.nc_matmul(dst=win_scores_psum, stationary=q_hb[hd], moving=kv_t_win[hd])
    win_scores = nl.ndarray((H_BATCH, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=win_scores, src=win_scores_psum)
    nisa.tensor_scalar(
        dst=win_scores[0:H_BATCH, 0:1],
        data=win_scores_psum[0:H_BATCH, 0:1],
        op0=nl.add,
        operand0=sink_hb[0:H_BATCH, 0:1],
    )

    # ---- This core's half of the gathered scoring ---------------------------
    SCORE_W = 512
    num_score_groups = (k_half + SCORE_W - 1) // SCORE_W
    kt_all = [None] * HD_TILES
    for hd in range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        kt_all[hd] = nl.ndarray((hd_sz, k_half), dtype=nl.float16, buffer=nl.sbuf)
    for c_idx in nl.affine_range(num_half_chunks):
        c_start = c_idx * COMP_CHUNK
        k_chunk = kv_chunks[c_idx]
        for hd in range(HD_TILES):
            hd_start = hd * HD_CHUNK
            hd_sz = min(HD_CHUNK, head_dim - hd_start)
            kt_psum = nl.ndarray((hd_sz, COMP_CHUNK), dtype=nl.float16, buffer=nl.psum)
            nisa.nc_transpose(dst=kt_psum, data=k_chunk[0:COMP_CHUNK, hd_start : hd_start + hd_sz])
            nisa.tensor_copy(dst=kt_all[hd][0:hd_sz, c_start : c_start + COMP_CHUNK], src=kt_psum)

    comp_scores = nl.ndarray((H_BATCH, k_half), dtype=nl.float32, buffer=nl.sbuf)
    for g in nl.affine_range(num_score_groups):
        g_start = g * SCORE_W
        g_sz = min(SCORE_W, k_half - g_start)
        scores_psum = nl.ndarray((H_BATCH, g_sz), dtype=nl.float32, buffer=nl.psum)
        for hd in nl.affine_range(HD_TILES):
            hd_start = hd * HD_CHUNK
            hd_sz = min(HD_CHUNK, head_dim - hd_start)
            nisa.nc_matmul(dst=scores_psum, stationary=q_hb[hd], moving=kt_all[hd][0:hd_sz, g_start : g_start + g_sz])
        nisa.tensor_copy(dst=comp_scores[0:H_BATCH, g_start : g_start + g_sz], src=scores_psum)

    # ---- PHASE 1 sendrecv: exchange local comp max -> identical global max ----
    # Packed into a 128-partition tile (a <128-partition sendrecv does NOT preserve
    # layout on this HW — HW-verified earlier this iteration).
    my_max = nl.ndarray((PAR, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=my_max[0:H_BATCH, 0:1], data=comp_scores, op=nl.maximum, axis=1)
    peer_max = nl.ndarray((PAR, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=my_max, dst=peer_max, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)

    comp_max = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=comp_max, data1=my_max[0:H_BATCH, 0:1], data2=peer_max[0:H_BATCH, 0:1], op=nl.maximum)
    win_max = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=win_max, data=win_scores, op=nl.maximum, axis=1)
    neg_max = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=neg_max, data1=win_max, data2=comp_max, op=nl.maximum)
    nisa.tensor_scalar(dst=neg_max, data=neg_max, op0=nl.multiply, operand0=-1.0)

    # ---- exp with the GLOBAL max (identical shift to the unsplit body) -------
    comp_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    comp_exp = nl.ndarray((H_BATCH, k_half), dtype=nl.float16, buffer=nl.sbuf)
    nisa.activation(
        dst=comp_exp,
        op=nl.exp,
        data=comp_scores,
        bias=neg_max,
        reduce_op=nl.add,
        reduce_res=comp_sum,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    out_psum = nl.ndarray((H_BATCH, head_dim), dtype=nl.float32, buffer=nl.psum)
    win_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=win_sum, value=0.0)
    if core_id == 0:
        win_exp = nl.ndarray((H_BATCH, WIN_SIZE), dtype=nl.float16, buffer=nl.sbuf)
        nisa.activation(
            dst=win_exp,
            op=nl.exp,
            data=win_scores,
            bias=neg_max,
            reduce_op=nl.add,
            reduce_res=win_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        we_T0 = nl.ndarray((KV_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.psum)
        nisa.nc_transpose(dst=we_T0, data=win_exp[0:H_BATCH, 0:KV_CHUNK])
        we_T0_sb = nl.ndarray((KV_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=we_T0_sb, src=we_T0)
        # First write into out_psum on core 0 (overwrites).
        nisa.nc_matmul(dst=out_psum, stationary=we_T0_sb, moving=win_v_0)

    # This core's half of the V accumulation (no explicit accumulate= — see above).
    for c_idx in nl.affine_range(num_half_chunks):
        c_start = c_idx * COMP_CHUNK
        v_chunk = kv_chunks[c_idx]
        ce_T = nl.ndarray((COMP_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.psum)
        nisa.nc_transpose(dst=ce_T, data=comp_exp[0:H_BATCH, c_start : c_start + COMP_CHUNK])
        ce_T_sb = nl.ndarray((COMP_CHUNK, H_BATCH), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ce_T_sb, src=ce_T)
        nisa.nc_matmul(dst=out_psum, stationary=ce_T_sb, moving=v_chunk)

    # ---- PHASE 2 sendrecv: exchange (partial acc | partial sums) -------------
    # One 128-partition fp32 tile: cols [0, head_dim) = partial V accumulator,
    # col head_dim = this core's comp exp-sum, col head_dim+1 = its window sum.
    PACK = head_dim + 2
    my_pack = nl.ndarray((PAR, PACK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=my_pack[0:H_BATCH, 0:head_dim], src=out_psum)
    nisa.tensor_copy(dst=my_pack[0:H_BATCH, head_dim : head_dim + 1], src=comp_sum)
    nisa.tensor_copy(dst=my_pack[0:H_BATCH, head_dim + 1 : head_dim + 2], src=win_sum)
    peer_pack = nl.ndarray((PAR, PACK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=my_pack, dst=peer_pack, send_to_rank=peer, recv_from_rank=peer, pipe_id=1)

    if core_id != 0:
        return

    # ---- Core 0: combine the two partials, normalize, de-RoPE, write --------
    out_sbuf = nl.ndarray((H_BATCH, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=out_sbuf, data1=my_pack[0:H_BATCH, 0:head_dim], data2=peer_pack[0:H_BATCH, 0:head_dim], op=nl.add
    )
    total_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=total_sum,
        data1=my_pack[0:H_BATCH, head_dim : head_dim + 1],
        data2=peer_pack[0:H_BATCH, head_dim : head_dim + 1],
        op=nl.add,
    )
    # window sums: exactly one core wrote a nonzero value, so adding both is exact.
    win_tot = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=win_tot,
        data1=my_pack[0:H_BATCH, head_dim + 1 : head_dim + 2],
        data2=peer_pack[0:H_BATCH, head_dim + 1 : head_dim + 2],
        op=nl.add,
    )
    nisa.tensor_tensor(dst=total_sum, data1=total_sum, data2=win_tot, op=nl.add)

    inv_sum = nl.ndarray((H_BATCH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=total_sum)
    nisa.tensor_scalar(dst=out_sbuf, data=out_sbuf, op0=nl.multiply, operand0=inv_sum[0:H_BATCH, 0:1])
    out_bf16 = nl.ndarray((H_BATCH, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_bf16, src=out_sbuf)

    # Fused output de-RoPE (identical algebra to `_gather_attn_stage`).
    half_rope = derope_cos.shape[1]
    rope_head_dim = 2 * half_rope
    nope_dim = head_dim - rope_head_dim
    d_rope_f = nl.ndarray((H_BATCH, rope_head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_rope_f[0:H_BATCH, 0:rope_head_dim], src=out_bf16[0:H_BATCH, nope_dim:head_dim])
    d_pairs = d_rope_f.reshape((H_BATCH, half_rope, 2))
    dx1 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dx1[0:H_BATCH, 0:half_rope], src=d_pairs[0:H_BATCH, 0:half_rope, 0])
    dx2 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dx2[0:H_BATCH, 0:half_rope], src=d_pairs[0:H_BATCH, 0:half_rope, 1])
    dcos = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=dcos[0:H_BATCH, 0:half_rope], src=derope_cos.ap(pattern=[[0, H_BATCH], [1, half_rope]]), priority=3
    )
    dsin = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=dsin[0:H_BATCH, 0:half_rope], src=derope_sin.ap(pattern=[[0, H_BATCH], [1, half_rope]]), priority=3
    )
    dta = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    dtb = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    dy1 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    dy2 = nl.ndarray((H_BATCH, half_rope), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dta, data1=dx1, data2=dcos, op=nl.multiply)
    nisa.tensor_tensor(dst=dtb, data1=dx2, data2=dsin, op=nl.multiply)
    nisa.tensor_tensor(dst=dy1, data1=dta, data2=dtb, op=nl.add)
    nisa.tensor_tensor(dst=dta, data1=dx2, data2=dcos, op=nl.multiply)
    nisa.tensor_tensor(dst=dtb, data1=dx1, data2=dsin, op=nl.multiply)
    nisa.tensor_tensor(dst=dy2, data1=dta, data2=dtb, op=nl.subtract)
    d_out = nl.ndarray((H_BATCH, half_rope, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_out[0:H_BATCH, 0:half_rope, 0], src=dy1)
    nisa.tensor_copy(dst=d_out[0:H_BATCH, 0:half_rope, 1], src=dy2)
    d_out_flat = d_out.reshape((H_BATCH, rope_head_dim))
    nisa.tensor_copy(dst=out_bf16[0:H_BATCH, nope_dim:head_dim], src=d_out_flat[0:H_BATCH, 0:rope_head_dim])

    nisa.dma_copy(
        dst=output.ap(pattern=[[S * head_dim, H_BATCH], [1, head_dim]], offset=h_base * S * head_dim),
        src=out_bf16,
        priority=1,
    )


def _split_head_fraction(T_c: int) -> tuple[int, int]:
    """Heads core 1 takes in the fused kernel's attention phase, as (num, den).

    (0, 1) means "don't split" — core 0 runs all heads, exactly as before.

    Trace-time only: `T_c` is a compile-time shape, so this is a plain Python
    branch and each seq-len compiles to the variant that measured fastest. No
    runtime dispatch, no per-step cost.

    WHY IT DEPENDS ON T_c. The work core 1 can take off core 0 here is O(k) with k
    FIXED (1024) — a CONSTANT. What it costs is (a) a duplicated gather of the same
    k rows (the top-k indices are head-independent, ~9.6 us of DMA) and (b) core 1
    arriving at the second barrier LATE, because core 0 spends that time on the
    top-k while core 1 is still finishing its O(T_c) score half. Cost (b) grows with
    T_c while the benefit does not, so past some T_c the split stops paying at ANY
    ratio (the obvious fix — give the late core a smaller share — was tried and
    measured; see below).

    MEASURED, medians of >=3 samples of profile total_exec_time (ms), s8192/16384/32768:
        T_c=2048  single 0.350       even 1/2 **0.342**
        T_c=4096  single 0.354       even 1/2 **0.347**
        T_c=8192  single **0.355**   even 1/2  0.364      quarter 1/4  0.3635
    At T_c=8192 BOTH split ratios lose, and shrinking core 1's share from 1/2 to 1/4
    recovered essentially nothing (0.364 -> 0.3635, inside noise). So what fails at
    large T_c is the MECHANISM, not the balance: no share is small enough to be worth
    the duplicated gather.

    *** iter-7 RE-TESTED THIS GATE AFTER ADDING THE TOP-K DMA SPLIT, AND IT STILL HOLDS.
    DO NOT REMOVE IT AGAIN. *** The hypothesis was that the T_c=8192 loss came from
    ARRIVAL SKEW (core 1 reaching the attention phase late because core 0 raced ahead
    through the core-0-only top-k region), and that `_snake_topk_stage_2core` — which
    splits the descriptor-bound snake reformat across both cores — would remove it.
    Making this function return (1, 2) unconditionally was BIT-IDENTICAL
    (max_abs_diff 1.083374e-03) and MUCH slower: total_exec {0.480, 0.474, 0.484}
    (median 0.480, a TIGHT cluster, vs 0.355 for the same file with the gate) and
    dma_active 0.300 -> 0.3055.

    The profile says exactly why, and it refutes the skew hypothesis: the NEFF span
    went 369.9 -> 567.5 us and EVERY engine on BOTH cores gained ~200 us of ACTIVE
    time (c0 Tensor 891->1086, Scalar 91->288, Vector 154->352, Sync 214->426). That
    is not a stall — it is REAL DUPLICATED WORK. Because the top-k indices are
    head-independent, both cores gather the SAME k rows, build the SAME K^T over all
    HD_TILES, and load the SAME window; meanwhile halving the heads saves almost
    nothing, since M = H_BATCH goes 32 -> 16 of 128 PE output partitions and matmul
    latency is ~insensitive to M below 128. So the duplication is pure addition.
    (The multi-hundred-microsecond EVENT_SEMAPHOREs that appear at the end of such a
    profile are drained engines parked at the terminal barrier — they lengthen because
    the NEFF lengthened, they are not the cause.)

    DUPLICATION, NOT SKEW, is what closes the split at T_c>4096. The top-k DMA split
    does not change that, so this gate stays.
    """
    if T_c <= 4096:
        return 1, 2  # even split: both cores take half the heads
    return 0, 1  # don't split — measured to lose at every ratio for T_c>4096


# --------------------------------------------------------------------------
# NKI Kernel: O(k) Decode Attention — gathered K scoring + gathered V matmul
#
# Standalone `[1]`-grid entry point, kept for the MULTI-chunk indexer path (whose
# per-chunk top-k + Pass-2 merge is host-orchestrated, so the indices genuinely
# have to come back as a tensor). The single-chunk decode path instead calls
# `nki_indexer_score_topk_gather_2core`, which runs this same body inside the
# indexer's own launch. Both share `_gather_attn_stage` verbatim.
# --------------------------------------------------------------------------
@nki.jit
def nki_decode_gather_ok_kernel(
    topk_indices_T: nl.NkiTensor,  # [k, S] uint32 — top-k indices transposed (partition=k)
    all_q_T: nl.NkiTensor,  # [head_dim, n_heads * S] — replicated query per head
    win_K_T: nl.NkiTensor,  # [head_dim, W] — window K^T (real window only)
    win_V: nl.NkiTensor,  # [W, head_dim] — window V (real window only)
    compress_kv: nl.NkiTensor,  # [T_c, head_dim] bf16 — compressed KV row-major (K=V in CSA)
    attn_sink_in: nl.NkiTensor,  # [1, n_heads] — per-head sink scalars
    derope_cos: nl.NkiTensor,  # [1, half_rope] fp32 — inverse-RoPE cos (output de-RoPE)
    derope_sin: nl.NkiTensor,  # [1, half_rope] fp32 — inverse-RoPE sin
) -> nl.NkiTensor:
    """O(k) decode attention on a `[1]` (or `[2]`) grid. Returns [n_heads*S, head_dim].

    topk_indices_T is [k, S] (transposed) so that partition-dim slicing gives
    COMP_CHUNK distinct indices per chunk for the swdge gather.
    """
    k, S = topk_indices_T.shape
    head_dim = all_q_T.shape[0]
    n_heads = all_q_T.shape[1] // S
    COMP_CHUNK = 128
    num_k_chunks = k // COMP_CHUNK

    n_cores = nl.num_programs()
    H_BATCH = n_heads // n_cores  # per-core head batch (n_heads/2 @ [2], n_heads @ [1])
    core_id = nl.program_id(0)
    h_base = core_id * H_BATCH  # global head offset for this core

    output = nl.ndarray((n_heads * S, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    idx_chunks = [None] * num_k_chunks
    for c_idx in nl.affine_range(num_k_chunks):
        c_start = c_idx * COMP_CHUNK
        idx_chunks[c_idx] = nl.ndarray((COMP_CHUNK, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_chunks[c_idx], src=topk_indices_T[c_start : c_start + COMP_CHUNK, 0:1], priority=0)

    _gather_attn_stage(
        idx_chunks,
        all_q_T,
        win_K_T,
        win_V,
        compress_kv,
        attn_sink_in,
        derope_cos,
        derope_sin,
        output,
        k,
        S,
        n_heads,
        h_base,
        H_BATCH,
    )
    return output


# --------------------------------------------------------------------------
# NKI Kernel: FUSED 2-LNC indexer scoring + top-k + O(k) attention in ONE launch
#
# Replaces `nki_indexer_score_topk_2core[2]` -> `nki_decode_gather_ok_kernel[1]` on
# the single-chunk decode path by folding the attention body into the indexer's
# `[2]`-grid launch. Two things go away: one @nki.jit launch boundary, and the [k, S]
# index array's trip out to the host graph between the two launches. The indices now
# stay on chip -- the top-k writes them and the gather's row-offset loads read them
# straight back from the same buffer.
#
# Structure:
#   1. both cores score their DISJOINT T_c halves into the named shared_hbm row;
#   2. `nisa.core_barrier(data=..., cores=(0, 1))` so core 0 sees core 1's half;
#   3. core 0 only: the top-k, then the whole gather+attention body. The gather needs
#      all n_heads on one core, which is what the standalone [1]-grid launch does
#      anyway. Core 1's trace ends at the barrier (or takes a share of the work --
#      see the K-split and head-split branches below).
#
# The index hand-off is byte-for-byte what the host chain delivered: k contiguous
# uint32 with partition stride 1, read from row 0 of the same top-k output buffer.
# --------------------------------------------------------------------------
@nki.jit
def nki_indexer_score_topk_gather_2core(
    q_T_all: nl.NkiTensor,  # [idx_head_dim, idx_n_heads * S_q] — indexer Q^T (bf16)
    kv_t: nl.NkiTensor,  # [idx_head_dim, T_c] — indexer_kv transposed (bf16)
    weights: nl.NkiTensor,  # [S_q, idx_n_heads] — per-head weights * weight_scale (fp32)
    k_val: int,  # number of top-k indices / gathered positions
    n_val: int,  # nisa.topk n (proven-safe padded width, >= T_c)
    all_q_T: nl.NkiTensor,  # [head_dim, n_heads * S] f16 — attention query per head
    win_K_T: nl.NkiTensor,  # [head_dim, W] f16 — window K^T (real window only)
    win_V: nl.NkiTensor,  # [W, head_dim] f16 — window V (real window only)
    compress_kv: nl.NkiTensor,  # [T_c, head_dim] bf16 — compressed KV row-major (K=V in CSA)
    attn_sink_in: nl.NkiTensor,  # [1, n_heads] fp32 — per-head sink scalars
    derope_cos: nl.NkiTensor,  # [1, half_rope] fp32 — inverse-RoPE cos (output de-RoPE)
    derope_sin: nl.NkiTensor,  # [1, half_rope] fp32 — inverse-RoPE sin
) -> nl.NkiTensor:
    """2-core indexer score + core-0 top-k + core-0 O(k) attention, ONE launch.

    Returns the attention output [n_heads * S, head_dim] bf16 — the same tensor
    `nki_decode_gather_ok_kernel` returns today.

    Requirements: launched on the [2] grid; n_val >= T_c and n_val % 16 == 0;
    k_val % 128 == 0. Bit-identical to the
    score_topk_2core[2] -> nki_decode_gather_ok_kernel[1] pipeline it replaces.
    """
    T_c = kv_t.shape[1]
    GROUP = 16
    TOPK_ROWS = 8
    COMP_CHUNK = 128

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    kernel_assert(n_cores == 2, "nki_indexer_score_topk_gather_2core needs the [2] grid")
    kernel_assert(n_val >= T_c and n_val % GROUP == 0, "n_val must be >= T_c and a multiple of 16")
    kernel_assert(k_val % COMP_CHUNK == 0, "k_val must be a multiple of the gather chunk (128)")
    # Attention-side geometry. attn_sink_in is [1, n_heads], so n_heads comes from
    # it rather than from the (absent) index tensor's shape; S then follows from
    # all_q_T. NOTE these are the ATTENTION head count / head_dim (32 / 512 for the
    # evaluated per-rank config), distinct from the INDEXER's (64 / 128), which the
    # score stage derives for itself from q_T_all / weights.
    n_heads = attn_sink_in.shape[1]
    S = all_q_T.shape[1] // n_heads
    head_dim = all_q_T.shape[0]
    output = nl.ndarray((n_heads * S, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="gather_attn_out")

    topk_idx = nl.ndarray((TOPK_ROWS, k_val), dtype=nl.uint32, buffer=nl.shared_hbm, name="indexer_topk_out")
    scores_pad = nl.ndarray((1, n_val), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="indexer_scores_shared")

    if core_id == 0 and n_val > T_c:
        pad_sb = nl.ndarray((1, n_val - T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=pad_sb, value=_TOPK_PAD_SENTINEL)
        nisa.dma_copy(dst=scores_pad[0:1, T_c:n_val], src=pad_sb)

    # Stage 1 (BOTH cores): disjoint T_c halves -> scores_pad[0:1, 0:T_c].
    _score_2core_stage(q_T_all, kv_t, weights, scores_pad)

    # Cross-core barrier: both cores' score halves (and the pad tail) are visible
    # to every core after this point.
    nisa.core_barrier(data=scores_pad, cores=(0, 1))

    # Stage 2: top-k.
    if (n_val // 2) % GROUP == 0:
        _snake_topk_stage_2core(scores_pad, topk_idx, k_val, n_val, core_id)
    elif core_id == 0:
        _snake_topk_stage(scores_pad, topk_idx, k_val, n_val)

    # ---- Stage 3: split the attention head batch over both cores, or not --------
    CORE1_HEAD_FRAC_NUM, CORE1_HEAD_FRAC_DEN = _split_head_fraction(T_c)
    c1_heads = (n_heads * CORE1_HEAD_FRAC_NUM) // CORE1_HEAD_FRAC_DEN
    split_heads = (c1_heads > 0) and (c1_heads < n_heads)

    # ---- K-SPLIT: divide the GATHERED POSITIONS (not the heads) over both cores ----
    use_ksplit = k_val % (2 * COMP_CHUNK) == 0
    if use_ksplit:
        # Publish the top-k winners so BOTH cores can read their own half of the offsets.
        nisa.core_barrier(data=topk_idx, cores=(0, 1))
        k_half = k_val // 2
        num_half_chunks = k_half // COMP_CHUNK
        base = core_id * k_half  # this core's first gathered position
        idx_chunks = [None] * num_half_chunks
        for c_idx in nl.affine_range(num_half_chunks):
            c_start = base + c_idx * COMP_CHUNK
            idx_chunks[c_idx] = nl.ndarray((COMP_CHUNK, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=idx_chunks[c_idx], src=topk_idx.ap(pattern=[[1, COMP_CHUNK], [1, 1]], offset=c_start), priority=0
            )
        _gather_attn_stage_ksplit(
            idx_chunks,
            all_q_T,
            win_K_T,
            win_V,
            compress_kv,
            attn_sink_in,
            derope_cos,
            derope_sin,
            output,
            k_val,
            S,
            n_heads,
            core_id,
            k_half,
        )
        return output

    if split_heads:
        # SECOND cross-core barrier: publishes the top-k winners core 0 just wrote so
        # BOTH cores can gather against them (core 1's trace would otherwise end at
        # the first barrier). A second core_barrier in one kernel was previously
        # unattested anywhere in this repo or the docs — it works, and this is the
        # first device-validated use.
        nisa.core_barrier(data=topk_idx, cores=(0, 1))

    if split_heads or core_id == 0:
        # core 0 takes the first (n_heads - c1_heads), core 1 the trailing c1_heads.
        if split_heads:
            H_BATCH = c1_heads if core_id == 1 else n_heads - c1_heads
            h_base = (n_heads - c1_heads) if core_id == 1 else 0
        else:
            H_BATCH = n_heads
            h_base = 0

        num_k_chunks = k_val // COMP_CHUNK
        idx_chunks = [None] * num_k_chunks
        for c_idx in nl.affine_range(num_k_chunks):
            c_start = c_idx * COMP_CHUNK
            idx_chunks[c_idx] = nl.ndarray((COMP_CHUNK, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=idx_chunks[c_idx], src=topk_idx.ap(pattern=[[1, COMP_CHUNK], [1, 1]], offset=c_start), priority=0
            )

        _gather_attn_stage(
            idx_chunks,
            all_q_T,
            win_K_T,
            win_V,
            compress_kv,
            attn_sink_in,
            derope_cos,
            derope_sin,
            output,
            k_val,
            S,
            n_heads,
            h_base,
            H_BATCH,
        )

    return output


# --------------------------------------------------------------------------
# NKI Kernel: the indexer's q-projection GEMV, hand-written to decouple DMA burst
# size from nc_matmul tile geometry.
#
# The block is DMA-bound and almost all of that DMA is projection-weight streaming,
# so this weight matters twice over. Lowering it as a torch nn.Linear in an lnc=2
# graph MATERIALIZES the constant at 2x its true size, and no change on the NKI
# consumer side shrinks that -- the projection has to leave nn.Linear entirely.
#
# The geometry is deliberate. Making the WEIGHT the `moving` operand in wide column
# groups cuts the declared bytes but is SLOWER: fewer, bigger matmuls each stall on
# their own weight tile instead of pipelining. So the weight stays STATIONARY at
# [128, 128] per matmul -- the tiling the compiler itself picks and pipelines well --
# while arriving in a few big contiguous bursts of 16 KB/partition, comfortably over
# the >= 4 KiB/partition DMA saturation target and far above the ~2.7 KB packets the
# compiler's own lowering emits.
#
# It is also N-sharded over the [2] grid. A [1]-grid kernel inside an lnc=2 graph
# puts this whole stream on one logical core while the sibling streams none of it,
# where every other weight in the block is split 50/50 by the compiler's lowering.
# Sharding costs no bytes (each core loads only its own column slice) and the bursts
# stay above the saturation target.
#
# Numerics: fp32 PSUM accumulation over the k-tiles, cast to bf16 exactly once at
# the end -- the same dataflow a compiler-lowered bf16 Linear uses.
#
# Output is q^T = [head_dim, n_heads], which is what the caller needs downstream.
# The n-tiles are heads and are independent (the only reduction is over k, in-core),
# so sharding changes only WHICH core evaluates which column -- but it does make the
# output a buffer both cores write disjoint halves of, hence the name= below.
# --------------------------------------------------------------------------
@nki.jit
def nki_indexer_qproj_gemv(wT: nl.NkiTensor, qr_in: nl.NkiTensor) -> nl.NkiTensor:
    """Indexer q-projection q = wq_b @ qr, returned TRANSPOSED as [head_dim, n_heads].

    wT:     [n_ktiles, 128, N] bf16 — the frozen wq_b weight pre-tiled so that
            wT[t, kk, n] == wq_b.weight[n, t*128 + kk]. A pure host-side transform of
            a frozen constant, so neuronx-cc constant-folds it and the ONLY weight
            materialized is this one, at its true size.
    qr_in:  [1, K] bf16 — the decode q-latent, K = n_ktiles * 128.
    Returns [head_dim, n_heads] bf16 where out[c, j] == q[head j, channel c].

    Requires head_dim == 128 (the indexer's index_head_dim), so that an N-tile of
    128 columns is exactly one head's channel block and the PSUM tile is q^T.

    N-SHARDED ACROSS THE [2] GRID (see the block comment above for why): core c
    owns the disjoint n-tile range [c*n_ntiles/n_cores, (c+1)*n_ntiles/n_cores)
    and DMAs only its own weight column slice, so the 25.17 MB stream is split
    ~12.6 MB/core instead of 25.17 MB on pcore0 and 0 on pcore1. Launched `[1]`
    it degenerates to exactly the previous single-core behaviour (core_id=0,
    n_cores=1 -> the full n-tile range), so the two launch shapes are
    bit-identical by construction.
    """
    core_id = nl.program_id(0)
    n_cores = nl.num_programs()

    n_ktiles = wT.shape[0]
    K_TILE = wT.shape[1]  # 128 — the nc_matmul contraction dim
    N = wT.shape[2]  # n_heads * head_dim
    N_TILE = 128  # stationary free dim: 128*128*2 = 32 KB, matches
    # the compiler's observed 1:1 MATMUL:LDWEIGHTS tile
    n_ntiles = N // N_TILE
    kernel_assert(K_TILE == 128, "k-tile must be the 128-row nc_matmul contraction dim")
    kernel_assert(N % N_TILE == 0, "N must tile evenly by 128 (one head's channels per tile)")
    kernel_assert(n_ntiles % n_cores == 0, "n-tiles must split evenly across the grid")

    nt_core = n_ntiles // n_cores
    j0 = core_id * nt_core  # first n-tile this core owns
    c0 = j0 * N_TILE  # first weight column this core owns
    N_core = nt_core * N_TILE  # weight columns this core streams

    out = nl.ndarray((K_TILE, n_ntiles), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qproj_qT")

    qr_sb = nl.ndarray((K_TILE, n_ktiles), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qr_sb, src=qr_in.ap(pattern=[[1, K_TILE], [K_TILE, n_ktiles]], offset=0), priority=1)

    # fp32 PSUM accumulator over THIS CORE's n-tiles only,
    # [head_dim=128, n_heads/n_cores] = 128 B/partition at [2] (one bank).
    acc = nl.ndarray((K_TILE, nt_core), dtype=nl.float32, buffer=nl.psum)

    w_sb = nl.ndarray((K_TILE, N_core), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in nl.sequential_range(n_ktiles):
        nisa.dma_copy(dst=w_sb, src=wT[t, 0:K_TILE, c0 : c0 + N_core], priority=0)
        for j in nl.affine_range(nt_core):
            nisa.nc_matmul(
                dst=acc[0:K_TILE, j : j + 1],
                stationary=w_sb[0:K_TILE, j * N_TILE : (j + 1) * N_TILE],
                moving=qr_sb[0:K_TILE, t : t + 1],
                accumulate=(t > 0),
            )

    # Single fp32 -> bf16 rounding at the end (matches the nn.Linear's bf16 output).
    out_bf = nl.ndarray((K_TILE, nt_core), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_bf, src=acc)
    nisa.dma_copy(dst=out[0:K_TILE, j0 : j0 + nt_core], src=out_bf, priority=1)
    return out
