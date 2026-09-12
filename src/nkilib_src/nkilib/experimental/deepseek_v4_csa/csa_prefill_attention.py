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

"""DeepSeek-V4 CSA prefill kernels.

The prefill counterpart of ``csa_decode_attention``. Prefill processes ``S``
query positions at once rather than one, so the partition axis carries the
SEQUENCE and every kernel here tiles over it; the decode kernels instead put
heads or compressed positions on partitions, because their query is a single
token.

Five kernels, in the order the block calls them:

``nki_rms_rope_kernel``
    Fused RMS(+optional learnable gain) + RoPE over a ``[S_rows, head_dim]`` tile.
    One kernel covers three call sites via trace-time flags: the q-path
    (``gain_in=None``), the kv-path (``gain_in=kv_norm.weight``) and the output
    de-RoPE (``do_rms=0, inverse=1``, rotation only).

``nki_compressor_core_kernel``
    Gated pooling over the ``2 * compress_ratio`` overlapped slots, then RMSNorm
    over ``head_dim``, then RoPE -- the compressor that folds raw tokens into
    compressed cache positions.

``nki_indexer_score_mask_kernel``
    Lightning-indexer scoring plus top-k selection, emitting a ``0 / -1e9``
    additive mask. The threshold comes from a fixed number of bisection rounds
    rather than a sort, so the whole selection stays on-chip and unrolled.

``nki_fused_csa_attn_kernel``
    Mask-predicated sparse attention over ``[window | compressed]``, used for the
    leading ``split_pos`` positions where every compressed position is still
    within the causal frontier.

``nki_gather_csa_attn_kernel``
    The same math with a per-tile COMPILE-TIME causal bound on the compressed
    loop, used for the trailing positions. The static bound is what removes the
    sequential dynamic-range device loop, the online-softmax rescaling and all
    indirect DMA, leaving every loop unrolled and pipelinable.

As in the decode file, ``priority=`` is a NeuronCore-v4 (trn3) DMA
class-of-service hint that changes no byte and no MAC.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from ...core.utils.kernel_assert import kernel_assert


# --------------------------------------------------------------------------
# The prefill analogue of csa_nki_model_decode_block.nki_qkv_rms_rope_kernel.
# That kernel is decode-shaped: it packs the n_heads q rows + the 1 kv row of a
# SINGLE token onto one [n_heads+1, head_dim] partition tile. Prefill has S rows
# per head, so the partition dim is the SEQUENCE and the kernel tiles over it,
# processing [TILE_S, head_dim] blocks with cos/sin sliced per tile (decode
# broadcasts one position with a stride-0 .ap()).
#
# Replaces this XLA chain, which materialized ~6 full [B,S,H,D] temporaries:
#   q * rsqrt(q.square().mean(-1) + eps)          (per-head RMS, no gain)
#   cat([x[..., :-rd], rope(x[..., -rd:])], -1)    (RoPE on the trailing rd dims)
# `gain_in=None` gives the no-learnable-gain q variant; passing kv_norm.weight
# gives the learnable-gain kv/RMSNorm variant. `inverse=1` negates sin for the
# output de-RoPE, and `do_rms=0` skips the norm (de-RoPE is rotation only).
# --------------------------------------------------------------------------
@nki.jit
def nki_rms_rope_kernel(
    x_in: nl.NkiTensor,
    cos_in: nl.NkiTensor,
    sin_in: nl.NkiTensor,
    gain_in: nl.NkiTensor | None,
    eps_val: float,
    do_rms: int = 1,
    inverse: int = 0,
    heads: int = 1,
    in_head_major: int = 1,
    out_head_major: int = 1,
    do_rope: int = 1,
) -> nl.NkiTensor:
    """Fused RMS(+optional gain) + RoPE over a [S_rows, head_dim] tile.

    x_in:   [S_rows, head_dim] bf16 — rows are (head-major) sequence positions.
    cos_in/sin_in: [S, half_rope] fp32 — per-POSITION rotation; with ``heads > 1``
            the table is indexed by position alone, so the caller does not repeat
            it per head.
    gain_in:[1, head_dim] fp32 or None — learnable RMSNorm gain, broadcast.
    Returns:[S_rows, head_dim] bf16 — nope channels passthrough, rope rotated.

    ``heads``/``in_head_major``/``out_head_major`` select the LAYOUT of the multi-head
    q and de-RoPE tensors, independently on each side:

    * head-major is ``[heads * S, head_dim]``, row ``h * S + s``;
    * query-major (``*_head_major=0``) is ``[S, heads * head_dim]``, row ``s`` and
      column block ``h``.

    """
    if in_head_major:
        head_dim = x_in.shape[1]
        S = x_in.shape[0] // heads
    else:
        head_dim = x_in.shape[1] // heads
        S = x_in.shape[0]
    S_rows = S * heads
    # do_rope=0 is the norm-only variant (the q-latent RMSNorm, which has no rotation):
    # every channel passes through the nope path and cos_in/sin_in are unused, so the
    # caller passes None rather than a dummy table.
    half_rope = cos_in.shape[1] if do_rope else 0
    rope_head_dim = 2 * half_rope
    nope_dim = head_dim - rope_head_dim
    TILE = 128  # partition tile (v4 hard cap)
    kernel_assert(S % TILE == 0, f"S={S} must be a multiple of {TILE}")
    n_tiles = S // TILE
    rows = TILE

    if out_head_major:
        out = nl.ndarray((S_rows, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    else:
        out = nl.ndarray((S, heads * head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    if gain_in is not None:
        gain = nl.ndarray((TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gain[0:TILE, 0:head_dim], src=gain_in.ap(pattern=[[0, TILE], [1, head_dim]]), priority=1)

    # SPMD across the (head, position-tile) space, which has no cross-tile dependency:
    # every tile reads its own rows and writes its own rows. Left grid-less this kernel ran
    # on ONE core, leaving the second core idle across this kernel's four launches while the
    # first carried the work on its Vector engine. Heads split first when they divide the grid
    # (the q and de-RoPE paths, 32 or 128 heads); otherwise the position tiles do (the kv
    # path, heads=1). A 1-core grid takes the same path with n_cores=1 and is unchanged.
    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    if heads % n_cores == 0:
        heads_per_core = heads // n_cores
        tiles_lo = 0
        tiles_hi = n_tiles
    else:
        kernel_assert(
            n_tiles % n_cores == 0,
            f"neither heads={heads} nor n_tiles={n_tiles} divides the {n_cores}-core grid",
        )
        heads_per_core = heads
        tiles_lo = 0
        tiles_hi = n_tiles // n_cores

    for h_local in nl.affine_range(heads_per_core):
        h = core_id * heads_per_core + h_local if heads % n_cores == 0 else h_local
        for ts_local in nl.affine_range(tiles_hi - tiles_lo):
            ts = ts_local if heads % n_cores == 0 else core_id * tiles_hi + ts_local
            s0 = ts * TILE
            cos_row = s0
            src_row = h * S + s0 if in_head_major else s0
            src_col = 0 if in_head_major else h * head_dim
            dst_row = h * S + s0 if out_head_major else s0
            dst_col = 0 if out_head_major else h * head_dim
            # priority=0: this load gates the whole RMS+RoPE chain below.
            x_sb = nl.ndarray((TILE, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_sb[0:rows, 0:head_dim],
                src=x_in[src_row : src_row + rows, src_col : src_col + head_dim],
                priority=0,
            )
            x_f32 = nl.ndarray((TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x_f32[0:rows, 0:head_dim], src=x_sb[0:rows, 0:head_dim])

            if do_rms:
                # mean(x^2) over the free axis -> *1/head_dim + eps (fused) -> rsqrt.
                x_sq = nl.ndarray((TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=x_sq[0:rows, 0:head_dim],
                    data1=x_f32[0:rows, 0:head_dim],
                    data2=x_f32[0:rows, 0:head_dim],
                    op=nl.multiply,
                )
                msq = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=msq[0:rows, 0:1], data=x_sq[0:rows, 0:head_dim], op=nl.add, axis=1)
                nisa.tensor_scalar(
                    dst=msq[0:rows, 0:1],
                    data=msq[0:rows, 0:1],
                    op0=nl.multiply,
                    operand0=1.0 / head_dim,
                    op1=nl.add,
                    operand1=eps_val,
                )
                rms = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rms[0:rows, 0:1], op=nl.rsqrt, data=msq[0:rows, 0:1])
                nisa.tensor_scalar(
                    dst=x_f32[0:rows, 0:head_dim],
                    data=x_f32[0:rows, 0:head_dim],
                    op0=nl.multiply,
                    operand0=rms[0:rows, 0:1],
                )
                if gain_in is not None:
                    nisa.tensor_tensor(
                        dst=x_f32[0:rows, 0:head_dim],
                        data1=x_f32[0:rows, 0:head_dim],
                        data2=gain[0:rows, 0:head_dim],
                        op=nl.multiply,
                    )

            # Cast at the RMSNorm output boundary (the reference casts back here).
            normed = nl.ndarray((TILE, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=normed[0:rows, 0:head_dim], src=x_f32[0:rows, 0:head_dim])

            # nope channels pass straight through.
            if nope_dim > 0:
                nisa.dma_copy(
                    dst=out[dst_row : dst_row + rows, dst_col : dst_col + nope_dim], src=normed[0:rows, 0:nope_dim]
                )

            if do_rope:
                # ---- RoPE on the trailing rope_head_dim channels (fp32 math) ----
                rope_f = nl.ndarray((TILE, rope_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rope_f[0:rows, 0:rope_head_dim], src=normed[0:rows, nope_dim:head_dim])
                # View as [.., half_rope, 2]: [...,0]=even (x1), [...,1]=odd (x2).
                rope_pairs = rope_f.reshape((TILE, half_rope, 2))
                x1 = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x1[0:rows, 0:half_rope], src=rope_pairs[0:rows, 0:half_rope, 0])
                x2 = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x2[0:rows, 0:half_rope], src=rope_pairs[0:rows, 0:half_rope, 1])

                # priority=2: cos/sin are consumed LAST (only by the rotation), so they
                # yield DMA bandwidth to the loads the pipeline stalls on first.
                cos_h = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=cos_h[0:rows, 0:half_rope], src=cos_in[cos_row : cos_row + rows, 0:half_rope], priority=2)
                sin_h = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=sin_h[0:rows, 0:half_rope], src=sin_in[cos_row : cos_row + rows, 0:half_rope], priority=2)

                # y1 = x1*cos - x2*sin ; y2 = x1*sin + x2*cos   (inverse negates sin, so
                # the signs swap: y1 = x1*cos + x2*sin ; y2 = -x1*sin + x2*cos)
                tmp_a = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                tmp_b = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                y1 = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                y2 = nl.ndarray((TILE, half_rope), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=tmp_a[0:rows, 0:half_rope],
                    data1=x1[0:rows, 0:half_rope],
                    data2=cos_h[0:rows, 0:half_rope],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=tmp_b[0:rows, 0:half_rope],
                    data1=x2[0:rows, 0:half_rope],
                    data2=sin_h[0:rows, 0:half_rope],
                    op=nl.multiply,
                )
                if inverse:
                    nisa.tensor_tensor(
                        dst=y1[0:rows, 0:half_rope],
                        data1=tmp_a[0:rows, 0:half_rope],
                        data2=tmp_b[0:rows, 0:half_rope],
                        op=nl.add,
                    )
                else:
                    nisa.tensor_tensor(
                        dst=y1[0:rows, 0:half_rope],
                        data1=tmp_a[0:rows, 0:half_rope],
                        data2=tmp_b[0:rows, 0:half_rope],
                        op=nl.subtract,
                    )
                nisa.tensor_tensor(
                    dst=tmp_a[0:rows, 0:half_rope],
                    data1=x1[0:rows, 0:half_rope],
                    data2=sin_h[0:rows, 0:half_rope],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=tmp_b[0:rows, 0:half_rope],
                    data1=x2[0:rows, 0:half_rope],
                    data2=cos_h[0:rows, 0:half_rope],
                    op=nl.multiply,
                )
                if inverse:
                    nisa.tensor_tensor(
                        dst=y2[0:rows, 0:half_rope],
                        data1=tmp_b[0:rows, 0:half_rope],
                        data2=tmp_a[0:rows, 0:half_rope],
                        op=nl.subtract,
                    )
                else:
                    nisa.tensor_tensor(
                        dst=y2[0:rows, 0:half_rope],
                        data1=tmp_a[0:rows, 0:half_rope],
                        data2=tmp_b[0:rows, 0:half_rope],
                        op=nl.add,
                    )

                # Re-interleave y1 (even) / y2 (odd), cast bf16, write out.
                rope_out = nl.ndarray((TILE, half_rope, 2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rope_out[0:rows, 0:half_rope, 0], src=y1[0:rows, 0:half_rope])
                nisa.tensor_copy(dst=rope_out[0:rows, 0:half_rope, 1], src=y2[0:rows, 0:half_rope])
                rope_flat = rope_out.reshape((TILE, rope_head_dim))
                rope_bf16 = nl.ndarray((TILE, rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rope_bf16[0:rows, 0:rope_head_dim], src=rope_flat[0:rows, 0:rope_head_dim])
                nisa.dma_copy(
                    dst=out[dst_row : dst_row + rows, dst_col + nope_dim : dst_col + head_dim],
                    src=rope_bf16[0:rows, 0:rope_head_dim],
                )

    return out


# --------------------------------------------------------------------------
# NKI Kernel: Compressor gated-pooling + RMSNorm + RoPE core
# --------------------------------------------------------------------------
@nki.jit
def nki_compressor_core_kernel(
    kv8: nl.NkiTensor,  # [T_c, ratio2, head_dim] — overlapped kv slots (bf16)
    score8: nl.NkiTensor,  # [T_c, ratio2, head_dim] — overlapped gate scores, NO ape (bf16)
    norm_weight: nl.NkiTensor,  # [1, head_dim] — RMSNorm gain (fp32)
    cos_rep: nl.NkiTensor,  # [T_c, rope_head_dim] — per-pair cos, each value repeated (fp32)
    sin_rep: nl.NkiTensor,  # [T_c, rope_head_dim] — per-pair sin, each value repeated (fp32)
    eps: float,  # RMSNorm epsilon
    ape: nl.NkiTensor | None = None,  # [ratio2, head_dim] — per-slot gate bias (fp32)
    hadamard: nl.NkiTensor | None = None,  # [head_dim, head_dim] — orthonormal rotation (bf16)
) -> nl.NkiTensor:
    """Gated pooling over the size-(2*ratio) axis, RMSNorm over head_dim, then RoPE.

    Per compressed position t and channel c:
        w[t, j, c] = softmax_j(score8[t, j, c])
        pooled[t, c] = sum_j kv8[t, j, c] * w[t, j, c]
    Then RMSNorm(pooled.to(bf16)) over c, and RoPE on the last rope_head_dim dims.

    Layout: partition = compressed positions (tiled by 128), free = head_dim channels.
    The softmax over the slot axis is computed independently per (position, channel).

    When ``hadamard`` is given the whole row is finally rotated by it (the indexer's
    compressor does this so the channels it scores are decorrelated). Being orthonormal
    it needs the nope and roped halves together, so they are assembled in SBUF and the
    row is written once after the rotation instead of half at a time.

    Returns:
        out: [T_c, head_dim] bf16 — normalized, roped, optionally rotated compressed kv.
    """
    T_c, ratio2, head_dim = kv8.shape
    rope_head_dim = cos_rep.shape[1]
    nope_dim = head_dim - rope_head_dim
    half_rope = rope_head_dim // 2

    TILE_P = 128
    num_tiles = (T_c + TILE_P - 1) // TILE_P

    out = nl.ndarray((T_c, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # The rotation operand is the same for every position tile, so it is loaded once for
    # the whole kernel rather than per tile. It has to fit on partitions because the row
    # is transposed to put the contracted channel axis there.
    h_sb = None
    if hadamard is not None:
        kernel_assert(head_dim <= 128, "hadamard rotation requires head_dim <= 128")
        h_sb = nl.ndarray((head_dim, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=h_sb, src=hadamard[0:head_dim, 0:head_dim])

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    tiles_per_core = num_tiles // n_cores

    for t_local in nl.affine_range(tiles_per_core):
        t_idx = core_id * tiles_per_core + t_local
        p_start = t_idx * TILE_P
        p_sz = min(TILE_P, T_c - p_start)

        # RMSNorm gain replicated to all partitions of this tile via a
        # partition-stride-0 DMA from the single HBM gain row.
        gain = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gain, src=norm_weight.ap(pattern=[[0, p_sz], [1, head_dim]]))

        # --- Load all slots for this position tile ---
        # The operands arrive BF16 and are widened here rather than on the host. They come
        # from a bf16 F.linear, so bf16 -> fp32 is exact and this is bit-identical to
        # taking them pre-widened -- but it halves what crosses HBM and, more to the point,
        # stops the host from materializing the fp32 copies at all (537 MB for the
        # projection plus 2 x 134 MB for the overlapped slots, at seq_len 32768).
        kv_slots = [None] * ratio2
        score_slots = [None] * ratio2
        for j in nl.affine_range(ratio2):
            kv_bf = nl.ndarray((p_sz, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kv_bf, src=kv8[p_start : p_start + p_sz, j, 0:head_dim])
            kv_slots[j] = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv_slots[j], src=kv_bf)
            score_bf = nl.ndarray((p_sz, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=score_bf, src=score8[p_start : p_start + p_sz, j, 0:head_dim])
            score_slots[j] = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=score_slots[j], src=score_bf)
            if ape is not None:
                # The per-slot gate bias is added HERE, in fp32, so the bf16 handoff above
                # stays exact: the host would otherwise add an fp32 parameter to a bf16
                # projection output, promoting the whole tensor to fp32 before the kernel
                # ever sees it. Row-invariant, so one stride-0 partition broadcast.
                ape_sb = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                ape_row = ape[j : j + 1, 0:head_dim]
                nisa.dma_copy(dst=ape_sb, src=ape_row.ap(pattern=[[0, p_sz], [1, head_dim]]))
                nisa.tensor_tensor(dst=score_slots[j], data1=score_slots[j], data2=ape_sb, op=nl.add)

        # --- Softmax over the slot axis (per position & channel) ---
        # Elementwise max across the ratio2 slots.
        slot_max = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slot_max, src=score_slots[0])
        for j in range(1, ratio2):
            nisa.tensor_tensor(dst=slot_max, data1=slot_max, data2=score_slots[j], op=nl.maximum)

        # exp(score - max) per slot, and accumulate the denominator.
        neg_max = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max, data=slot_max, op0=nl.multiply, operand0=-1.0)

        exp_slots = [None] * ratio2
        denom = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(ratio2):
            exp_slots[j] = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=exp_slots[j], data1=score_slots[j], data2=neg_max, op=nl.add)
            nisa.activation(dst=exp_slots[j], op=nl.exp, data=exp_slots[j])
            if j == 0:
                nisa.tensor_copy(dst=denom, src=exp_slots[0])
            else:
                nisa.tensor_tensor(dst=denom, data1=denom, data2=exp_slots[j], op=nl.add)

        inv_denom = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_denom, data=denom)

        # --- Weighted sum: pooled = sum_j kv_j * (exp_j / denom) ---
        pooled = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(ratio2):
            # weight_j = exp_j * inv_denom (softmax); pooled += kv_j * weight_j
            nisa.tensor_tensor(dst=prod, data1=exp_slots[j], data2=inv_denom, op=nl.multiply)
            nisa.tensor_tensor(dst=prod, data1=prod, data2=kv_slots[j], op=nl.multiply)
            if j == 0:
                nisa.tensor_copy(dst=pooled, src=prod)
            else:
                nisa.tensor_tensor(dst=pooled, data1=pooled, data2=prod, op=nl.add)

        # --- Cast pooled to bf16 (matches reference kv.to(bf16) before norm) ---
        pooled_bf16 = nl.ndarray((p_sz, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pooled_bf16, src=pooled)
        # Re-widen to fp32 for the norm compute (reference RMSNorm computes in fp32).
        x_norm = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_norm, src=pooled_bf16)

        # --- RMSNorm over head_dim (free axis) ---
        sq = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=x_norm, data2=x_norm, op=nl.multiply)
        msq = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=msq, data=sq, op=nl.add, axis=1)
        # mean = sum / head_dim, then rsqrt(mean + eps).
        nisa.tensor_scalar(dst=msq, data=msq, op0=nl.multiply, operand0=1.0 / head_dim, op1=nl.add, operand1=eps)
        rms = nl.ndarray((p_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rms, op=nl.rsqrt, data=msq)
        # normed = x * rms (broadcast over free) * gain (broadcast over partition).
        normed = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=normed, data=x_norm, op0=nl.multiply, operand0=rms)
        nisa.tensor_tensor(dst=normed, data1=normed, data2=gain, op=nl.multiply)

        # Cast to bf16 (RMSNorm output dtype), then RoPE reads bf16 -> fp32.
        normed_bf16 = nl.ndarray((p_sz, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=normed_bf16, src=normed)

        # --- The nope part (channels 0..nope_dim-1) ---
        # Unrotated it goes straight to HBM. Rotated it must meet the roped half first,
        # so it is staged in an SBUF row that the matmul below consumes whole.
        full_bf16 = None
        if hadamard is None:
            nisa.dma_copy(dst=out[p_start : p_start + p_sz, 0:nope_dim], src=normed_bf16[0:p_sz, 0:nope_dim])
        else:
            full_bf16 = nl.ndarray((p_sz, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=full_bf16[0:p_sz, 0:nope_dim], src=normed_bf16[0:p_sz, 0:nope_dim])

        # --- RoPE on the last rope_head_dim channels ---
        # Load cos/sin (already repeated per pair) for these positions.
        cos_sb = nl.ndarray((p_sz, rope_head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cos_sb, src=cos_rep[p_start : p_start + p_sz, 0:rope_head_dim])
        sin_sb = nl.ndarray((p_sz, rope_head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sin_sb, src=sin_rep[p_start : p_start + p_sz, 0:rope_head_dim])

        # Widen the rope channels back to fp32 (reference computes RoPE in fp32).
        rope_f = nl.ndarray((p_sz, rope_head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rope_f, src=normed_bf16[0:p_sz, nope_dim:head_dim])
        # View as [p_sz, half_rope, 2] so [...,0]=even (x1), [...,1]=odd (x2).
        rope_pairs = rope_f.reshape((p_sz, half_rope, 2))
        x1 = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x1, src=rope_pairs[0:p_sz, 0:half_rope, 0])
        x2 = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x2, src=rope_pairs[0:p_sz, 0:half_rope, 1])

        cos_pairs = cos_sb.reshape((p_sz, half_rope, 2))
        sin_pairs = sin_sb.reshape((p_sz, half_rope, 2))
        cos_h = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cos_h, src=cos_pairs[0:p_sz, 0:half_rope, 0])
        sin_h = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sin_h, src=sin_pairs[0:p_sz, 0:half_rope, 0])

        # y1 = x1*cos - x2*sin ; y2 = x1*sin + x2*cos
        tmp_a = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        tmp_b = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        y1 = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        y2 = nl.ndarray((p_sz, half_rope), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp_a, data1=x1, data2=cos_h, op=nl.multiply)
        nisa.tensor_tensor(dst=tmp_b, data1=x2, data2=sin_h, op=nl.multiply)
        nisa.tensor_tensor(dst=y1, data1=tmp_a, data2=tmp_b, op=nl.subtract)
        nisa.tensor_tensor(dst=tmp_a, data1=x1, data2=sin_h, op=nl.multiply)
        nisa.tensor_tensor(dst=tmp_b, data1=x2, data2=cos_h, op=nl.multiply)
        nisa.tensor_tensor(dst=y2, data1=tmp_a, data2=tmp_b, op=nl.add)

        # Re-interleave y1 (even) and y2 (odd) into [p_sz, half_rope, 2] then bf16.
        rope_out = nl.ndarray((p_sz, half_rope, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rope_out[0:p_sz, 0:half_rope, 0], src=y1)
        nisa.tensor_copy(dst=rope_out[0:p_sz, 0:half_rope, 1], src=y2)
        rope_out_flat = rope_out.reshape((p_sz, rope_head_dim))
        rope_out_bf16 = nl.ndarray((p_sz, rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rope_out_bf16, src=rope_out_flat)

        if hadamard is None:
            nisa.dma_copy(dst=out[p_start : p_start + p_sz, nope_dim:head_dim], src=rope_out_bf16)
        else:
            # --- Rotate the assembled row: out[s, d] = sum_c row[s, c] * H[c, d] ---
            # nc_matmul contracts over the PARTITION axis, so the row tile is transposed
            # once to put c there; then row^T as the STATIONARY operand with H moving
            # lands [s, d] directly, with no second transpose to undo. The bf16 operands
            # accumulate into an fp32 PSUM and round once on the way out, which is what
            # `bf16 @ bf16` does on XLA -- so this matches the reference bit-for-bit in
            # intent, not just approximately.
            nisa.tensor_copy(dst=full_bf16[0:p_sz, nope_dim:head_dim], src=rope_out_bf16)
            row_t_psum = nl.ndarray((head_dim, p_sz), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=row_t_psum, data=full_bf16[0:p_sz, 0:head_dim])
            row_t = nl.ndarray((head_dim, p_sz), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=row_t, src=row_t_psum)
            rot_psum = nl.ndarray((p_sz, head_dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=rot_psum, stationary=row_t, moving=h_sb)
            rot_bf16 = nl.ndarray((p_sz, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rot_bf16, src=rot_psum)
            nisa.dma_copy(dst=out[p_start : p_start + p_sz, 0:head_dim], src=rot_bf16)

    return out


# --------------------------------------------------------------------------
# NKI Kernel: Indexer per-head scoring + binary-search top-k mask
# --------------------------------------------------------------------------
@nki.jit
def nki_indexer_score_mask_kernel(
    q_T_all: nl.NkiTensor,  # [head_dim, n_heads * S_q] — all heads' Q^T stacked (bf16)
    kv_t: nl.NkiTensor,  # [head_dim, T_c] — indexer_kv transposed (bf16), shared
    weights: nl.NkiTensor,  # [S_q, n_heads] — per-row per-head weights * weight_scale (fp32)
    causal_bias: nl.NkiTensor,  # [S_q, T_c] — causal bias (0 / -1e9) (fp32)
    k: int,  # top-k count
) -> nl.NkiTensor:
    """Indexer scoring + top-k selection mask in a single NKI kernel.

    Computes, per query row s and compressed kv position t:
        index_score[s, t] = sum_h relu(q[s, h, :] . kv[t, :]) * weights[s, h]
                            + causal_bias[s, t]
    then finds, per row, a threshold via 10 iterations of bisection (matching the
    reference _build_mask_from_scores), and builds:
        sel_mask[s, t] = 0     if index_score[s, t] >= threshold[s]
                         -1e9   otherwise

    Returns:
        sel_mask: [S_q, T_c] fp32 — selection mask (0 / -1e9).
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

    NEG_INF = -1e9

    sel_mask = nl.ndarray((S_q, T_c), dtype=nl.float32, buffer=nl.shared_hbm)

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

        nisa.memset(dst=index_score_bf16, value=0)
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

        # Binary search for per-row threshold (matches reference _build_mask_from_scores)
        hi = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=hi, op=nl.maximum, data=index_score, axis=1)
        is_valid = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_valid, data=index_score, op0=nl.greater, operand0=-1e8)
        score_minus_hi = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=score_minus_hi, data=index_score, op0=nl.subtract, operand0=hi)
        masked_for_min = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=masked_for_min, data1=score_minus_hi, data2=is_valid, op=nl.multiply)
        nisa.tensor_scalar(dst=masked_for_min, data=masked_for_min, op0=nl.add, operand0=hi)
        lo = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=lo, op=nl.minimum, data=masked_for_min, axis=1)

        mid = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
        count = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
        ge_mid = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        ge_count_flag = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
        for _it in range(9):
            nisa.tensor_tensor(dst=mid, data1=lo, data2=hi, op=nl.add)
            nisa.tensor_scalar(dst=mid, data=mid, op0=nl.multiply, operand0=0.5)
            nisa.tensor_scalar_reduce(
                dst=ge_mid, data=index_score, op0=nl.greater_equal, operand0=mid, reduce_op=nl.add, reduce_res=count
            )
            nisa.tensor_scalar(dst=ge_count_flag, data=count, op0=nl.greater_equal, operand0=float(k))
            mid_minus_lo = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mid_minus_lo, data1=mid, data2=lo, op=nl.subtract)
            nisa.tensor_tensor(dst=mid_minus_lo, data1=mid_minus_lo, data2=ge_count_flag, op=nl.multiply)
            nisa.tensor_tensor(dst=lo, data1=lo, data2=mid_minus_lo, op=nl.add)
            lt_count_flag = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=lt_count_flag, data=ge_count_flag, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0
            )
            mid_minus_hi = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mid_minus_hi, data1=mid, data2=hi, op=nl.subtract)
            nisa.tensor_tensor(dst=mid_minus_hi, data1=mid_minus_hi, data2=lt_count_flag, op=nl.multiply)
            nisa.tensor_tensor(dst=hi, data1=hi, data2=mid_minus_hi, op=nl.add)

        # Build mask: sel = (score >= lo) ? 0 : -1e9
        sel = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sel, data=index_score, op0=nl.greater_equal, operand0=lo)
        nisa.tensor_scalar(dst=sel, data=sel, op0=nl.multiply, operand0=-NEG_INF, op1=nl.add, operand1=NEG_INF)
        nisa.dma_copy(dst=sel_mask[q_start : q_start + TILE_Q, 0:T_c], src=sel)

    return sel_mask


# --------------------------------------------------------------------------
# NKI Kernel: Fused CSA Attention with PSum Accumulation + V Preloading
# --------------------------------------------------------------------------


@nki.jit
def nki_fused_csa_attn_kernel(
    compress_sel: nl.NkiTensor,  # [S, T_c] — selection mask (0/-inf), shared across heads
    all_q_T: nl.NkiTensor,  # [head_dim, n_heads * S] — all heads' Q^T stacked
    all_K_T: nl.NkiTensor,  # [head_dim, S + W + T_c] — concatenated K^T (shared)
    all_V: nl.NkiTensor,  # [S + W + T_c, head_dim] — concatenated V (shared)
    win_bias_base_in: nl.NkiTensor,  # [S, 256] — base window bias (0/-1e9), shared across heads
    win_bias_sink_in: nl.NkiTensor,  # [S, 256] — sink indicator (0/1), shared across heads
    attn_sink_in: nl.NkiTensor,  # [1, n_heads] — per-head sink scalars
) -> nl.NkiTensor:
    S, T_c = compress_sel.shape
    head_dim = all_q_T.shape[0]
    total_q_free = all_q_T.shape[1]
    n_heads = total_q_free // S
    W = 128
    TILE_Q = 128
    KV_CHUNK = 128
    COMP_V_CHUNK = min(KV_CHUNK, T_c)
    SCORE_CHUNK = min(512, T_c)
    WIN_SIZE = 2 * KV_CHUNK
    num_q_tiles = S // TILE_Q
    num_c_chunks = T_c // COMP_V_CHUNK
    num_score_chunks = T_c // SCORE_CHUNK
    H_BATCH = 16 if T_c >= 512 else 8
    num_h_batches = n_heads // H_BATCH

    HD_CHUNK = 128
    HD_TILES = (head_dim + HD_CHUNK - 1) // HD_CHUNK

    output = nl.ndarray((n_heads * S, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    kv_comp_offset = S + W

    all_comp_kt = [None] * HD_TILES
    for hd in range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        all_comp_kt[hd] = nl.ndarray((hd_sz, T_c), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=all_comp_kt[hd], src=all_K_T[hd_start : hd_start + hd_sz, kv_comp_offset : kv_comp_offset + T_c]
        )

    comp_v = []
    for i in range(num_c_chunks):
        c_start = i * COMP_V_CHUNK
        v_tile = nl.ndarray((COMP_V_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_tile, src=all_V[kv_comp_offset + c_start : kv_comp_offset + c_start + COMP_V_CHUNK, 0:head_dim]
        )
        comp_v.append(v_tile)

    # Preload per-head attn_sink scalars, replicated to all TILE_Q partitions
    # using partition-stride-0 DMA so each partition has the full n_heads vector.
    attn_sink_sb = nl.ndarray((TILE_Q, n_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=attn_sink_sb, src=attn_sink_in.ap(pattern=[[0, TILE_Q], [1, n_heads]]))

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    tiles_per_core = num_q_tiles // n_cores

    for q_local in nl.affine_range(tiles_per_core):
        q_idx = core_id * tiles_per_core + q_local
        q_start = q_idx * TILE_Q

        comp_mask = nl.ndarray((TILE_Q, T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=comp_mask, src=compress_sel[q_start : q_start + TILE_Q, 0:T_c])

        kv_t_win = [None] * HD_TILES
        for hd in range(HD_TILES):
            hd_start = hd * HD_CHUNK
            hd_sz = min(HD_CHUNK, head_dim - hd_start)
            kv_t_win[hd] = nl.ndarray((hd_sz, WIN_SIZE), dtype=nl.float16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kv_t_win[hd], src=all_K_T[hd_start : hd_start + hd_sz, q_start : q_start + WIN_SIZE])

        win_v_0 = nl.ndarray((KV_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=win_v_0, src=all_V[q_start : q_start + KV_CHUNK, 0:head_dim])
        win_v_1 = nl.ndarray((KV_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=win_v_1, src=all_V[q_start + KV_CHUNK : q_start + WIN_SIZE, 0:head_dim])

        # Load base and sink_ind ONCE per tile (shared across all heads).
        bias_base_tile = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_base_tile, src=win_bias_base_in[q_start : q_start + TILE_Q, 0:WIN_SIZE])
        bias_sink_tile = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_sink_tile, src=win_bias_sink_in[q_start : q_start + TILE_Q, 0:WIN_SIZE])

        for hb_idx in nl.affine_range(num_h_batches):
            q_T = [None] * H_BATCH
            for h_local in nl.affine_range(H_BATCH):
                h = hb_idx * H_BATCH + h_local
                q_global = h * S + q_start
                q_T[h_local] = [None] * HD_TILES
                for hd in nl.affine_range(HD_TILES):
                    hd_start = hd * HD_CHUNK
                    hd_sz = min(HD_CHUNK, head_dim - hd_start)
                    q_T[h_local][hd] = nl.ndarray((hd_sz, TILE_Q), dtype=nl.float16, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=q_T[h_local][hd], src=all_q_T[hd_start : hd_start + hd_sz, q_global : q_global + TILE_Q]
                    )

            # === Compute window biases inline for H_BATCH heads ===
            win_bias = [None] * H_BATCH
            for h_local in nl.affine_range(H_BATCH):
                h = hb_idx * H_BATCH + h_local
                win_bias[h_local] = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=win_bias[h_local],
                    data=bias_sink_tile,
                    op0=nl.multiply,
                    operand0=attn_sink_sb[0:TILE_Q, h : h + 1],
                    op1=nl.add,
                    operand1=bias_base_tile,
                )

            # === Compute window scores for H_BATCH heads ===
            win_scores = [None] * H_BATCH
            for h_local in nl.affine_range(H_BATCH):
                win_scores_psum = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.psum)
                for hd in nl.affine_range(HD_TILES):
                    nisa.nc_matmul(dst=win_scores_psum, stationary=q_T[h_local][hd], moving=kv_t_win[hd])
                win_scores[h_local] = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=win_scores[h_local], data1=win_scores_psum, data2=win_bias[h_local], op=nl.add)

            # === Compute compressed scores for H_BATCH heads ===
            comp_scores = [None] * H_BATCH
            for h_local in nl.affine_range(H_BATCH):
                comp_scores[h_local] = nl.ndarray((TILE_Q, T_c), dtype=nl.float32, buffer=nl.sbuf)
                for m_idx in nl.affine_range(num_score_chunks):
                    m_start = m_idx * SCORE_CHUNK
                    scores_chunk_psum = nl.ndarray((TILE_Q, SCORE_CHUNK), dtype=nl.float32, buffer=nl.psum)
                    for hd in nl.affine_range(HD_TILES):
                        kt_slice = all_comp_kt[hd][0 : all_comp_kt[hd].shape[0], m_start : m_start + SCORE_CHUNK]
                        nisa.nc_matmul(dst=scores_chunk_psum, stationary=q_T[h_local][hd], moving=kt_slice)
                    nisa.tensor_tensor(
                        dst=comp_scores[h_local][0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                        data1=scores_chunk_psum,
                        data2=comp_mask[0:TILE_Q, m_start : m_start + SCORE_CHUNK],
                        op=nl.add,
                    )

            # === Unified global max softmax for H_BATCH heads ===
            total_sum = [None] * H_BATCH
            win_exp = [None] * H_BATCH
            comp_exp = [None] * H_BATCH
            for h_local in nl.affine_range(H_BATCH):
                win_max = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=win_max, data=win_scores[h_local], op=nl.maximum, axis=1)
                comp_max = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=comp_max, data=comp_scores[h_local], op=nl.maximum, axis=1)
                neg_max = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=neg_max, data1=win_max, data2=comp_max, op=nl.maximum)
                nisa.tensor_scalar(dst=neg_max, data=neg_max, op0=nl.multiply, operand0=-1.0)

                win_sum = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                win_exp[h_local] = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(
                    dst=win_exp[h_local],
                    op=nl.exp,
                    data=win_scores[h_local],
                    bias=neg_max,
                    reduce_op=nl.add,
                    reduce_res=win_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                comp_sum = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                comp_exp[h_local] = nl.ndarray((TILE_Q, T_c), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(
                    dst=comp_exp[h_local],
                    op=nl.exp,
                    data=comp_scores[h_local],
                    bias=neg_max,
                    reduce_op=nl.add,
                    reduce_res=comp_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                total_sum[h_local] = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=total_sum[h_local], data1=win_sum, data2=comp_sum, op=nl.add)

            # === V multiply with PSum accumulation for H_BATCH heads ===
            for h_local in nl.affine_range(H_BATCH):
                out_psum = nl.ndarray((TILE_Q, head_dim), dtype=nl.float32, buffer=nl.psum)

                scores_T_psum = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=scores_T_psum, data=win_exp[h_local][0:TILE_Q, 0:KV_CHUNK])
                scores_T_sb = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores_T_sb, src=scores_T_psum)
                nisa.nc_matmul(dst=out_psum, stationary=scores_T_sb, moving=win_v_0)

                scores_T_psum = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=scores_T_psum, data=win_exp[h_local][0:TILE_Q, KV_CHUNK:WIN_SIZE])
                scores_T_sb = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores_T_sb, src=scores_T_psum)
                nisa.nc_matmul(dst=out_psum, stationary=scores_T_sb, moving=win_v_1)

                for c_idx in nl.affine_range(num_c_chunks):
                    c_start = c_idx * COMP_V_CHUNK
                    scores_T_psum = nl.ndarray((COMP_V_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=scores_T_psum, data=comp_exp[h_local][0:TILE_Q, c_start : c_start + COMP_V_CHUNK]
                    )
                    scores_T_sb = nl.ndarray((COMP_V_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=scores_T_sb, src=scores_T_psum)
                    nisa.nc_matmul(dst=out_psum, stationary=scores_T_sb, moving=comp_v[c_idx])

                # Finalize: single copy from PSum, divide by sum, write output
                out_sbuf = nl.ndarray((TILE_Q, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_sbuf, src=out_psum)

                inv_sum = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=inv_sum, op=nl.reciprocal, data=total_sum[h_local])
                nisa.tensor_scalar(dst=out_sbuf, data=out_sbuf, op0=nl.multiply, operand0=inv_sum)

                h = hb_idx * H_BATCH + h_local
                q_global = h * S + q_start
                out_bf16 = nl.ndarray((TILE_Q, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_bf16, src=out_sbuf)
                nisa.dma_copy(dst=output[q_global : q_global + TILE_Q, 0:head_dim], src=out_bf16)

    return output


# --------------------------------------------------------------------------
# NKI Kernel: Full gather-based sparse attention (indirect DMA)
# --------------------------------------------------------------------------


@nki.jit
def nki_gather_csa_attn_kernel(
    topk_sel_bias: nl.NkiTensor,  # [S, T_c] bfloat16 — selection bias (0/-inf), causal-masked
    all_q_T: nl.NkiTensor,  # [head_dim, n_heads * S] — all heads' Q^T stacked (bf16)
    all_K_T_win: nl.NkiTensor,  # [head_dim, S + W] — window K^T (padded)
    all_V_win: nl.NkiTensor,  # [S + W, head_dim] — window V (padded)
    compress_kv_T: nl.NkiTensor,  # [head_dim, T_c] bf16 — compressed KV transposed (for scoring)
    compress_kv: nl.NkiTensor,  # [T_c, head_dim] bf16 — compressed KV row-major (for V)
    win_bias_base_in: nl.NkiTensor,  # [S, 256] — base window bias
    win_bias_sink_in: nl.NkiTensor,  # [S, 256] — sink indicator
    attn_sink_in: nl.NkiTensor,  # [1, n_heads] — per-head sink scalars
    split_pos: int,  # global position offset of this second half
    ratio: int,  # compression ratio
) -> nl.NkiTensor:
    """Sparse attention with static causal-bound + global-max softmax + mask predication.

    Mirrors the dense kernel's math (global-max softmax over window + compressed,
    sel_bias added as additive -1e9 predication before exp, then V-multiply and
    normalize) but caps the compressed-chunk loop at a *compile-time* per-tile
    causal bound. This removes the sequential dynamic_range device loop, the
    online-softmax rescaling, and all indirect DMA — every loop is unrolled and
    pipelinable by the compiler.

    q_idx is a compile-time Python int (static_range), so causal_chunks[q_idx] is
    known at trace time. topk_sel_bias already encodes causal masking, so processing
    columns [0, causal_chunks*COMP_V_CHUNK) with sel_bias predication is exact:
    every selected position lies within the causal frontier, and unselected /
    beyond-causal positions are -1e9 -> exp -> 0 -> contribute nothing.
    """
    S = topk_sel_bias.shape[0]
    head_dim = all_q_T.shape[0]
    T_c = compress_kv_T.shape[1]
    n_heads = all_q_T.shape[1] // S
    TILE_Q = 128
    KV_CHUNK = 128
    COMP_V_CHUNK = min(KV_CHUNK, T_c)
    SCORE_CHUNK = min(512, T_c)
    WIN_SIZE = 2 * KV_CHUNK
    num_q_tiles = S // TILE_Q
    num_c_chunks = T_c // COMP_V_CHUNK
    H_BATCH = 16
    num_h_batches = n_heads // H_BATCH
    HD_CHUNK = 128
    HD_TILES = (head_dim + HD_CHUNK - 1) // HD_CHUNK

    output = nl.ndarray((n_heads * S, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    attn_sink_sb = nl.ndarray((TILE_Q, n_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=attn_sink_sb, src=attn_sink_in.ap(pattern=[[0, TILE_Q], [1, n_heads]]))

    # Preload full compressed K^T (shared across all Q tiles and heads).
    all_comp_kt = [None] * HD_TILES
    for hd in range(HD_TILES):
        hd_start = hd * HD_CHUNK
        hd_sz = min(HD_CHUNK, head_dim - hd_start)
        all_comp_kt[hd] = nl.ndarray((hd_sz, T_c), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(dst=all_comp_kt[hd], src=compress_kv_T[hd_start : hd_start + hd_sz, 0:T_c])

    # Preload full compressed V chunks (shared across all Q tiles and heads).
    comp_v = [None] * num_c_chunks
    for i in range(num_c_chunks):
        c_start = i * COMP_V_CHUNK
        comp_v[i] = nl.ndarray((COMP_V_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=comp_v[i], src=compress_kv[c_start : c_start + COMP_V_CHUNK, 0:head_dim])

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    # Split work on head-batch dimension so both cores process the same Q tile.
    hb_per_core = num_h_batches // n_cores

    for q_idx in nl.static_range(num_q_tiles):
        q_start = q_idx * TILE_Q

        # Per-tile compile-time causal bound (CEIL-DIV — rounding up is always safe;
        # extra chunks are fully -1e9-masked in sel_bias so they contribute 0).
        global_q_end = split_pos + (q_idx + 1) * TILE_Q
        causal_chunks = min(num_c_chunks, (global_q_end + (ratio * COMP_V_CHUNK) - 1) // (ratio * COMP_V_CHUNK))
        comp_cols = causal_chunks * COMP_V_CHUNK
        num_score_chunks = (comp_cols + SCORE_CHUNK - 1) // SCORE_CHUNK

        # Load sel_bias for this tile's causal columns only (plain contiguous DMA).
        comp_mask = nl.ndarray((TILE_Q, comp_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=comp_mask, src=topk_sel_bias[q_start : q_start + TILE_Q, 0:comp_cols])

        # Load window data (shared across heads for this Q tile).
        kv_t_win = [None] * HD_TILES
        for hd in nl.affine_range(HD_TILES):
            hd_start = hd * HD_CHUNK
            hd_sz = min(HD_CHUNK, head_dim - hd_start)
            kv_t_win[hd] = nl.ndarray((hd_sz, WIN_SIZE), dtype=nl.float16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kv_t_win[hd], src=all_K_T_win[hd_start : hd_start + hd_sz, q_start : q_start + WIN_SIZE])

        win_v_0 = nl.ndarray((KV_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=win_v_0, src=all_V_win[q_start : q_start + KV_CHUNK, 0:head_dim])
        win_v_1 = nl.ndarray((KV_CHUNK, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=win_v_1, src=all_V_win[q_start + KV_CHUNK : q_start + WIN_SIZE, 0:head_dim])

        bias_base_tile = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_base_tile, src=win_bias_base_in[q_start : q_start + TILE_Q, 0:WIN_SIZE])
        bias_sink_tile = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias_sink_tile, src=win_bias_sink_in[q_start : q_start + TILE_Q, 0:WIN_SIZE])

        for hb_local in nl.affine_range(hb_per_core):
            hb_idx = core_id * hb_per_core + hb_local
            q_T = [None] * H_BATCH
            for h_local in nl.affine_range(H_BATCH):
                h = hb_idx * H_BATCH + h_local
                q_global = h * S + q_start
                q_T[h_local] = [None] * HD_TILES
                for hd in nl.affine_range(HD_TILES):
                    hd_start = hd * HD_CHUNK
                    hd_sz = min(HD_CHUNK, head_dim - hd_start)
                    q_T[h_local][hd] = nl.ndarray((hd_sz, TILE_Q), dtype=nl.float16, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=q_T[h_local][hd], src=all_q_T[hd_start : hd_start + hd_sz, q_global : q_global + TILE_Q]
                    )

            # Staged per-head processing (mirrors the dense kernel) so the compiler
            # can pipeline the Tensor-Engine score/V matmuls of one head against the
            # Vector-Engine exp/reduce of another. Only small per-head exp buffers and
            # scalar sums are retained as lists; the big [128, comp_cols] fp32
            # comp_scores buffer is transient (consumed to produce comp_exp), so SBUF
            # stays bounded even at H_BATCH=16 (comp_exp bf16 list = 16 * comp_cols * 2B).
            win_exp_all = [None] * H_BATCH
            comp_exp_all = [None] * H_BATCH
            total_sum_all = [None] * H_BATCH

            # --- Stage A: scores + unified global-max softmax per head ---
            for h_local in nl.affine_range(H_BATCH):
                h = hb_idx * H_BATCH + h_local

                win_bias_h = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=win_bias_h,
                    data=bias_sink_tile,
                    op0=nl.multiply,
                    operand0=attn_sink_sb[0:TILE_Q, h : h + 1],
                    op1=nl.add,
                    operand1=bias_base_tile,
                )
                win_scores_psum = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.psum)
                for hd in nl.affine_range(HD_TILES):
                    nisa.nc_matmul(dst=win_scores_psum, stationary=q_T[h_local][hd], moving=kv_t_win[hd])
                win_scores_h = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=win_scores_h, data1=win_scores_psum, data2=win_bias_h, op=nl.add)
                win_max = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=win_max, data=win_scores_h, op=nl.maximum, axis=1)

                comp_scores = nl.ndarray((TILE_Q, comp_cols), dtype=nl.float32, buffer=nl.sbuf)
                for m_idx in nl.affine_range(num_score_chunks):
                    m_start = m_idx * SCORE_CHUNK
                    m_sz = min(SCORE_CHUNK, comp_cols - m_start)
                    scores_chunk_psum = nl.ndarray((TILE_Q, m_sz), dtype=nl.float32, buffer=nl.psum)
                    for hd in nl.affine_range(HD_TILES):
                        kt_slice = all_comp_kt[hd][0 : all_comp_kt[hd].shape[0], m_start : m_start + m_sz]
                        nisa.nc_matmul(dst=scores_chunk_psum, stationary=q_T[h_local][hd], moving=kt_slice)
                    nisa.tensor_tensor(
                        dst=comp_scores[0:TILE_Q, m_start : m_start + m_sz],
                        data1=scores_chunk_psum,
                        data2=comp_mask[0:TILE_Q, m_start : m_start + m_sz],
                        op=nl.add,
                    )
                comp_max = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=comp_max, data=comp_scores, op=nl.maximum, axis=1)

                neg_max = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=neg_max, data1=win_max, data2=comp_max, op=nl.maximum)
                nisa.tensor_scalar(dst=neg_max, data=neg_max, op0=nl.multiply, operand0=-1.0)

                win_sum = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                win_exp_all[h_local] = nl.ndarray((TILE_Q, WIN_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(
                    dst=win_exp_all[h_local],
                    op=nl.exp,
                    data=win_scores_h,
                    bias=neg_max,
                    reduce_op=nl.add,
                    reduce_res=win_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                comp_sum = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                comp_exp_all[h_local] = nl.ndarray((TILE_Q, comp_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(
                    dst=comp_exp_all[h_local],
                    op=nl.exp,
                    data=comp_scores,
                    bias=neg_max,
                    reduce_op=nl.add,
                    reduce_res=comp_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                total_sum_all[h_local] = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=total_sum_all[h_local], data1=win_sum, data2=comp_sum, op=nl.add)

            # --- Stage B: V multiply + normalize + write per head ---
            for h_local in nl.affine_range(H_BATCH):
                h = hb_idx * H_BATCH + h_local
                out_psum = nl.ndarray((TILE_Q, head_dim), dtype=nl.float32, buffer=nl.psum)

                scores_T_psum = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=scores_T_psum, data=win_exp_all[h_local][0:TILE_Q, 0:KV_CHUNK])
                scores_T_sb = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores_T_sb, src=scores_T_psum)
                nisa.nc_matmul(dst=out_psum, stationary=scores_T_sb, moving=win_v_0)

                scores_T_psum2 = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=scores_T_psum2, data=win_exp_all[h_local][0:TILE_Q, KV_CHUNK:WIN_SIZE])
                scores_T_sb2 = nl.ndarray((KV_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=scores_T_sb2, src=scores_T_psum2)
                nisa.nc_matmul(dst=out_psum, stationary=scores_T_sb2, moving=win_v_1)

                for c_idx in nl.affine_range(causal_chunks):
                    c_start = c_idx * COMP_V_CHUNK
                    exp_T_psum = nl.ndarray((COMP_V_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=exp_T_psum, data=comp_exp_all[h_local][0:TILE_Q, c_start : c_start + COMP_V_CHUNK]
                    )
                    exp_T_sb = nl.ndarray((COMP_V_CHUNK, TILE_Q), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=exp_T_sb, src=exp_T_psum)
                    nisa.nc_matmul(dst=out_psum, stationary=exp_T_sb, moving=comp_v[c_idx])

                out_sbuf = nl.ndarray((TILE_Q, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_sbuf, src=out_psum)
                inv_sum = nl.ndarray((TILE_Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=inv_sum, op=nl.reciprocal, data=total_sum_all[h_local])
                nisa.tensor_scalar(dst=out_sbuf, data=out_sbuf, op0=nl.multiply, operand0=inv_sum)

                q_global = h * S + q_start
                out_bf16 = nl.ndarray((TILE_Q, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_bf16, src=out_sbuf)
                nisa.dma_copy(dst=output[q_global : q_global + TILE_Q, 0:head_dim], src=out_bf16)

    return output


# --------------------------------------------------------------------------
# NKI Kernel: TRUE sparse prefill attention (per-query indirect gather)
#
# The other two prefill attention kernels are dense-plus-mask: they score every
# causal compressed column and predicate the unselected ones to -1e9. That is
# correct but computes `causal_cols / k` more score positions than the model needs
# (4.3x at seq_len=32768, 128x at 1M).
#
# This kernel instead gathers each query's `k` SELECTED compressed rows with an
# indirect DMA and scores only those, so its compressed cost is O(k) and
# independent of context length.
#
# WHY IT NEEDS n_heads ON THE PARTITION DIM. The dense kernels put QUERIES on the
# matmul output-partition dim and loop heads, which lets 128 queries share one
# moving K^T operand -- and that sharing is exactly what per-query selection
# breaks, because each query wants different columns. So this kernel transposes the
# roles: heads on the output partitions, one query at a time, the gathered K^T as
# the moving operand. That makes the stationary tile [head_dim_chunk, n_heads], so
# it is only efficient when n_heads is large: at n_heads=128 it fills all 128
# output partitions, at n_heads=32 it wastes three quarters of them. Hence this
# kernel is for the SEQUENCE-PARALLEL sharding (all heads local, queries split
# across ranks, compressed KV replicated), not the head-parallel sharding.
#
# The window is a per-query causal SLICE rather than a masked 256-column block, so
# no additive window bias is needed: query p reads window columns [p, p+W) and the
# window is the causal W-column slice ENDING AT the query's own position.
# --------------------------------------------------------------------------
@nki.jit
def nki_prefill_sparse_attn_kernel(
    topk_idx_T: nl.NkiTensor,  # [k, S] uint32 — per-query selected compressed positions
    all_q: nl.NkiTensor,  # [S * n_heads, head_dim] f16 — QUERY-MAJOR: one query's heads contiguous
    all_K_T_win: nl.NkiTensor,  # [head_dim, S + W] f16 — window K^T (padded)
    all_V_win: nl.NkiTensor,  # [S + W, head_dim] f16 — window V (padded)
    compress_kv: nl.NkiTensor,  # [T_c, head_dim] f16 — FULL compressed KV, replicated per rank
    attn_sink_in: nl.NkiTensor,  # [1, n_heads] f32 — read for n_heads only; see the window note
) -> nl.NkiTensor:
    """O(k) sparse prefill attention. Returns [n_heads * S, head_dim] bf16.

    Requires n_heads == 128 (one full matmul output-partition tile) and
    k % 128 == 0. `S` here is the number of queries THIS launch covers.
    """
    head_dim = all_q.shape[1]
    k_val = topk_idx_T.shape[0]
    S = topk_idx_T.shape[1]
    n_heads = attn_sink_in.shape[1]
    W = 128
    COMP_CHUNK = 128
    HD_CHUNK = 128
    HD_TILES = head_dim // HD_CHUNK
    num_chunks = k_val // COMP_CHUNK

    kernel_assert(n_heads == 128, "sparse prefill needs all 128 heads on one rank (sequence-parallel sharding)")
    kernel_assert(k_val % COMP_CHUNK == 0, "k must be a multiple of the gather chunk (128)")
    kernel_assert(head_dim % HD_CHUNK == 0, "head_dim must be a multiple of 128")

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    kernel_assert(S % n_cores == 0, "query count must divide across the launch grid")
    s_per_core = S // n_cores
    s_start = core_id * s_per_core

    output = nl.ndarray((n_heads * S, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="sparse_prefill_out")

    for q_local in nl.static_range(s_per_core):
        q = s_start + q_local  # this core's query, a trace-time constant
        p = q  # window slice is pre-positioned by the caller, so p is tile-relative

        q_hb = [None] * HD_TILES
        for hd in range(HD_TILES):
            q_hb[hd] = nl.ndarray((HD_CHUNK, n_heads), dtype=nl.float16, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=q_hb[hd],
                src=all_q.ap(
                    pattern=[[head_dim, n_heads], [1, HD_CHUNK]], offset=q * n_heads * head_dim + hd * HD_CHUNK
                ),
            )

        # ---- indirect gather of this query's k selected compressed rows ----
        kv_chunks = [None] * num_chunks
        for c in nl.affine_range(num_chunks):
            idx = nl.ndarray((COMP_CHUNK, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=topk_idx_T.ap(pattern=[[S, COMP_CHUNK], [1, 1]], offset=c * COMP_CHUNK * S + q))
            kv_chunks[c] = nl.ndarray((COMP_CHUNK, head_dim), dtype=nl.float16, buffer=nl.sbuf)
            nisa.memset(dst=kv_chunks[c], value=0)
            nisa.dma_copy(
                dst=kv_chunks[c],
                src=compress_kv.ap(pattern=[[head_dim, COMP_CHUNK], [1, head_dim]], vector_offset=idx, indirect_dim=0),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=oob_mode.skip,
                priority=0,
            )

        # ---- window K^T / V: the causal W-column slice for THIS query ----
        win_kt = [None] * HD_TILES
        for hd in range(HD_TILES):
            hd_start = hd * HD_CHUNK
            win_kt[hd] = nl.ndarray((HD_CHUNK, W), dtype=nl.float16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=win_kt[hd], src=all_K_T_win[hd_start : hd_start + HD_CHUNK, p + 1 : p + 1 + W], priority=2
            )
        win_v = nl.ndarray((W, head_dim), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(dst=win_v, src=all_V_win[p + 1 : p + 1 + W, 0:head_dim], priority=2)

        # ---- window scores ----
        win_ps = nl.ndarray((n_heads, W), dtype=nl.float32, buffer=nl.psum)
        for hd in nl.affine_range(HD_TILES):
            nisa.nc_matmul(dst=win_ps, stationary=q_hb[hd], moving=win_kt[hd])
        win_scores = nl.ndarray((n_heads, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=win_scores, src=win_ps)

        # ---- compressed scores over the k gathered positions ONLY ----
        comp_scores = nl.ndarray((n_heads, k_val), dtype=nl.float32, buffer=nl.sbuf)
        for c in nl.affine_range(num_chunks):
            c0 = c * COMP_CHUNK
            ps = nl.ndarray((n_heads, COMP_CHUNK), dtype=nl.float32, buffer=nl.psum)
            for hd in nl.affine_range(HD_TILES):
                hd_start = hd * HD_CHUNK
                tp = nl.ndarray((HD_CHUNK, COMP_CHUNK), dtype=nl.float16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=kv_chunks[c][0:COMP_CHUNK, hd_start : hd_start + HD_CHUNK])
                kt_c = nl.ndarray((HD_CHUNK, COMP_CHUNK), dtype=nl.float16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=kt_c, src=tp)
                nisa.nc_matmul(dst=ps, stationary=q_hb[hd], moving=kt_c)
            nisa.tensor_copy(dst=comp_scores[0:n_heads, c0 : c0 + COMP_CHUNK], src=ps)

        # ---- one global-max softmax over [window | gathered] ----
        win_max = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=win_max, data=win_scores, op=nl.maximum, axis=1)
        comp_max = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=comp_max, data=comp_scores, op=nl.maximum, axis=1)
        neg_max = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=neg_max, data1=win_max, data2=comp_max, op=nl.maximum)
        nisa.tensor_scalar(dst=neg_max, data=neg_max, op0=nl.multiply, operand0=-1.0)

        win_sum = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        win_exp = nl.ndarray((n_heads, W), dtype=nl.float16, buffer=nl.sbuf)
        nisa.activation(
            dst=win_exp,
            op=nl.exp,
            data=win_scores,
            bias=neg_max,
            reduce_op=nl.add,
            reduce_res=win_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        comp_sum = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        comp_exp = nl.ndarray((n_heads, k_val), dtype=nl.float16, buffer=nl.sbuf)
        nisa.activation(
            dst=comp_exp,
            op=nl.exp,
            data=comp_scores,
            bias=neg_max,
            reduce_op=nl.add,
            reduce_res=comp_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        # ---- V accumulation: window slice first, then the gathered chunks ----
        out_psum = nl.ndarray((n_heads, head_dim), dtype=nl.float32, buffer=nl.psum)
        we_T = nl.ndarray((W, n_heads), dtype=nl.float16, buffer=nl.psum)
        nisa.nc_transpose(dst=we_T, data=win_exp)
        we_T_sb = nl.ndarray((W, n_heads), dtype=nl.float16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=we_T_sb, src=we_T)
        nisa.nc_matmul(dst=out_psum, stationary=we_T_sb, moving=win_v)
        for c in nl.affine_range(num_chunks):
            c0 = c * COMP_CHUNK
            ce_T = nl.ndarray((COMP_CHUNK, n_heads), dtype=nl.float16, buffer=nl.psum)
            nisa.nc_transpose(dst=ce_T, data=comp_exp[0:n_heads, c0 : c0 + COMP_CHUNK])
            ce_T_sb = nl.ndarray((COMP_CHUNK, n_heads), dtype=nl.float16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ce_T_sb, src=ce_T)
            nisa.nc_matmul(dst=out_psum, stationary=ce_T_sb, moving=kv_chunks[c])

        # ---- normalize by the shared denominator and write out head-major ----
        total = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=total, data1=win_sum, data2=comp_sum, op=nl.add)
        inv = nl.ndarray((n_heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv, op=nl.reciprocal, data=total)
        o_f32 = nl.ndarray((n_heads, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_f32, src=out_psum)
        nisa.tensor_scalar(dst=o_f32, data=o_f32, op0=nl.multiply, operand0=inv)
        o_bf = nl.ndarray((n_heads, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_bf, src=o_f32)
        nisa.dma_copy(dst=output.ap(pattern=[[S * head_dim, n_heads], [1, head_dim]], offset=q * head_dim), src=o_bf)

    return output


# --------------------------------------------------------------------------
# NKI Kernel: per-query top-k over the indexer scores, entirely on-chip
#
# The block used to build nisa.topk's snake layout on the HOST: reshape/transpose
# [S_q, T_c] into [S_q * 128, T_c/16] and hand it back to a topk kernel that
# returned [S_q * 128, k] values AND indices. Only 1/128 of that output is ever
# read (snake group 0 = 16 partitions x k/16 columns), so at S_q = 7168, T_c = 8192
# the stage moved ~6.5 GB through HBM to deliver 29 MB of indices.
#
# Here the snake tile is built in SBUF from the score rows directly, with the same
# nc_transpose fold `_snake_fill` uses in the decode indexer. HBM traffic becomes
# read S_q * T_c bf16 + write S_q * k uint32 -- 117 MB + 29 MB at those shapes.
#
# --------------------------------------------------------------------------
_SNAKE_GROUP = 16
_SNAKE_GROUPS = 8
_SNAKE_PAR = 128
_SNAKE_NEG = -1.0e30
"""Sentinel for snake positions that hold no score. Must be strictly below every real
score, including the indexer's -1e9 causal mask, so padding can never be selected."""


@nki.jit
def nki_prefill_topk_kernel(
    scores: nl.NkiTensor,  # [S_q, T_c] bf16 — indexer scores, causal bias already folded in
    k_val: int,  # top-k count; multiple of 16
    n_val: int,  # nisa.topk width; T_c padded up to a proven-safe width
) -> nl.NkiTensor:
    """Per-query top-k positions. Returns [S_q, k_val] uint32 GLOBAL compressed positions.

    The returned index is a global position because the snake fill satisfies
    ``snake[16 * g + r, c] == scores[base + g, 16 * c + r]``, which is exactly
    nisa.topk's per-group index encoding.

    The k winners of a row are an UNORDERED SET: nisa.topk emits each snake partition's
    winners in ascending position order, not by value. The gather that consumes them is
    order-agnostic.
    """
    S_q = scores.shape[0]
    T_c = scores.shape[1]
    GROUP = _SNAKE_GROUP
    GROUPS = _SNAKE_GROUPS
    PAR = _SNAKE_PAR
    snake_x = n_val // GROUP
    live_x = T_c // GROUP
    k_cols = k_val // GROUP

    kernel_assert(T_c % (GROUP * 128) == 0, "T_c must be a multiple of 16*128 for the snake fold")
    kernel_assert(k_val % GROUP == 0, "k must be a multiple of the snake group size")
    kernel_assert(n_val >= T_c, "topk width must cover T_c")

    out = nl.ndarray((S_q, k_val), dtype=nl.uint32, buffer=nl.shared_hbm, name="prefill_topk_idx")

    core_id = nl.program_id(0)
    n_cores = nl.num_programs()
    kernel_assert(S_q % (GROUPS * n_cores) == 0, "score rows must divide into 8-row tiles across the grid")
    tiles_per_core = S_q // (GROUPS * n_cores)
    tile_base = core_id * tiles_per_core

    snake = nl.ndarray((PAR, snake_x), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=snake, value=_SNAKE_NEG)

    val = nl.ndarray((PAR, k_val), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx = nl.ndarray((PAR, k_val), dtype=nl.uint32, buffer=nl.sbuf)

    for t_local in nl.sequential_range(tiles_per_core):
        base = (tile_base + t_local) * GROUPS

        for b in nl.static_range(live_x // 128):
            blk = nl.ndarray((128, PAR), dtype=nl.bfloat16, buffer=nl.sbuf)
            for g in nl.static_range(GROUPS):
                nisa.dma_copy(
                    dst=blk[0:128, g * GROUP : (g + 1) * GROUP],
                    src=scores.ap(pattern=[[GROUP, 128], [1, GROUP]], offset=(base + g) * T_c + b * 128 * GROUP),
                )
            tp = nl.ndarray((PAR, 128), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=blk)
            nisa.tensor_copy(dst=snake[0:PAR, b * 128 : (b + 1) * 128], src=tp)

        nisa.topk(val_dst=val, idx_dst=idx, src=snake, n=n_val)

        nisa.dma_copy(
            dst=out.ap(pattern=[[k_val, GROUPS], [k_cols, GROUP], [1, k_cols]], offset=base * k_val),
            src=idx[0:PAR, 0:k_cols],
        )

    return out
