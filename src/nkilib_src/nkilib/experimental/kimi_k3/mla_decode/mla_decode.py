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

"""Kimi-K3 dense MLA decode with output gate, out-absorb, and o-projection.

The kernel consumes pre-transposed attention operands, computes dense latent
attention over prior and current tokens, applies ``W_UV``, multiplies a
caller-computed sigmoid gate in natural ``[token, head * value]`` order, and
then applies the per-rank output projection.

The prior cache is dense: ``key_pos`` and ``sparse_add`` must both be ``None``.
The maximum prior length is guarded because the resident latent, scores, and
softmax weights scale linearly with context and share Trn3 SBUF.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.entry_trace import trace_kernel_entry
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ....core.utils.tiled_range import TiledRange

_P_MAX = 128
_F_MAX = 512
_NEG_INF = -30000.0

# Measured on Trn3 LNC1 with the Kimi production head shape (B=2, Nh=3).
# S=17152 has no SBUF spills; S=17280 is the first 128-token step that
# spills (3072 bytes/partition).
_MAX_DENSE_PRIOR = 17152


@nki.jit
def kimi_k3_mla_decode(
    q_nope_absorbed_T: nl.NkiTensor,
    q_pe_T: nl.NkiTensor,
    ckv_prior_T: nl.NkiTensor,
    kpe_prior_T: nl.NkiTensor,
    ckv_cur_T: nl.NkiTensor,
    kpe_cur_T: nl.NkiTensor,
    positions: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    gate: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    softmax_scale: float,
    key_pos: Optional[nl.NkiTensor] = None,
    sparse_add: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """Return Kimi-K3's gated per-rank MLA decode output ``[B, H]``.

    Args:
        q_nope_absorbed_T: BF16 absorbed query ``[C, B, Nh]``.
        q_pe_T: BF16 unrotated 64-wide query slice ``[R, B, Nh]``.
        ckv_prior_T: BF16 dense prior latent cache ``[C, B, S]``.
        kpe_prior_T: BF16 dense prior 64-wide cache ``[R, B, S]``.
        ckv_cur_T: BF16 current latent ``[C, B, 1]``.
        kpe_cur_T: BF16 current 64-wide key ``[R, B, 1]``.
        positions: INT32 absolute decode positions ``[B]``.
        out_absorb_w: BF16 ``W_UV`` weights ``[Nh, V, C]``.
        gate: BF16 sigmoid gate ``[B, Nh*V]`` in head-major order.
        o_proj_w: BF16 per-rank output weights ``[Nh*V, H]``.
        softmax_scale: Score scale, ``192**-0.5`` for Kimi-K3.
        key_pos: Must be ``None``; Kimi uses dense sequential cache slots.
        sparse_add: Must be ``None``; Kimi has no sparse indexer mask.

    Returns:
        BF16 per-rank output ``[B, H]``, before the TP all-reduce.

    Notes:
        This is a Trn3 LNC1 decode kernel. Prior lengths above
        ``_MAX_DENSE_PRIOR`` are rejected instead of overflowing SBUF.
    """
    trace_kernel_entry("kimi_k3_mla_decode", locals())

    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "kimi_k3_mla_decode supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "kimi_k3_mla_decode requires Trn3 or newer")
    kernel_assert(key_pos is None, "Kimi-K3 MLA decode requires key_pos=None")
    kernel_assert(sparse_add is None, "Kimi-K3 MLA decode requires sparse_add=None")

    C = q_nope_absorbed_T.shape[0]
    B = q_nope_absorbed_T.shape[1]
    Nh = q_nope_absorbed_T.shape[2]
    R = q_pe_T.shape[0]
    S = ckv_prior_T.shape[2]
    V = out_absorb_w.shape[1]
    H = o_proj_w.shape[1]

    kernel_assert(q_nope_absorbed_T.dtype == nl.bfloat16, "q_nope_absorbed_T must be BF16")
    kernel_assert(q_pe_T.dtype == nl.bfloat16, "q_pe_T must be BF16")
    kernel_assert(ckv_prior_T.dtype == nl.bfloat16, "ckv_prior_T must be BF16")
    kernel_assert(kpe_prior_T.dtype == nl.bfloat16, "kpe_prior_T must be BF16")
    kernel_assert(ckv_cur_T.dtype == nl.bfloat16, "ckv_cur_T must be BF16")
    kernel_assert(kpe_cur_T.dtype == nl.bfloat16, "kpe_cur_T must be BF16")
    kernel_assert(out_absorb_w.dtype == nl.bfloat16, "out_absorb_w must be BF16")
    kernel_assert(gate.dtype == nl.bfloat16, "gate must be BF16")
    kernel_assert(o_proj_w.dtype == nl.bfloat16, "o_proj_w must be BF16")
    kernel_assert(positions.dtype == nl.int32, "positions must be INT32")

    kernel_assert(0 < B <= _P_MAX, f"Expected 1 <= B <= {_P_MAX}, got B={B}")
    kernel_assert(0 < Nh <= _P_MAX, f"Expected 1 <= Nh <= {_P_MAX}, got Nh={Nh}")
    kernel_assert(0 < R <= _P_MAX, f"Expected 1 <= R <= {_P_MAX}, got R={R}")
    kernel_assert(0 < V <= _P_MAX, f"Expected 1 <= V <= {_P_MAX}, got V={V}")
    kernel_assert(0 < S <= _MAX_DENSE_PRIOR, f"Expected 1 <= S <= {_MAX_DENSE_PRIOR}, got S={S}")
    kernel_assert(q_pe_T.shape == (R, B, Nh), f"Unexpected q_pe_T shape {q_pe_T.shape}")
    kernel_assert(ckv_prior_T.shape == (C, B, S), f"Unexpected ckv_prior_T shape {ckv_prior_T.shape}")
    kernel_assert(kpe_prior_T.shape == (R, B, S), f"Unexpected kpe_prior_T shape {kpe_prior_T.shape}")
    kernel_assert(ckv_cur_T.shape == (C, B, 1), f"Unexpected ckv_cur_T shape {ckv_cur_T.shape}")
    kernel_assert(kpe_cur_T.shape == (R, B, 1), f"Unexpected kpe_cur_T shape {kpe_cur_T.shape}")
    kernel_assert(positions.shape == (B,), f"Unexpected positions shape {positions.shape}")
    kernel_assert(out_absorb_w.shape == (Nh, V, C), f"Unexpected out_absorb_w shape {out_absorb_w.shape}")
    kernel_assert(gate.shape == (B, Nh * V), f"Unexpected gate shape {gate.shape}")
    kernel_assert(o_proj_w.shape[0] == Nh * V, f"Expected o_proj_w rows={Nh * V}, got {o_proj_w.shape[0]}")

    Sk = S + 1
    nC = div_ceil(C, _P_MAX)
    output = nl.ndarray((B, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Batched softmax-weighted latent, laid out as [C tile, head, batch].
    xT_all = nl.ndarray((_P_MAX, nC * Nh * B), dtype=nl.bfloat16, buffer=nl.sbuf)

    pos_all_i = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_all_i[0:1, 0:B], src=positions[0:B].reshape((1, B)))
    pos_all_f = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_all_f[0:1, 0:B], src=pos_all_i[0:1, 0:B])

    ones_col = nl.ndarray((1, _P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col[0:1, 0:Nh], value=1.0)

    # W_UV is static across batch rows. Store each [C, V] transpose once.
    oaT_sb_all = nl.ndarray((_P_MAX, nC * Nh * V), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in range(Nh):
        for ct in TiledRange(C, _P_MAX):
            cs, ce, csz = ct.start_offset, ct.end_offset, ct.size
            ci = cs // _P_MAX
            oa_sb = nl.ndarray((_P_MAX, V), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=oa_sb[0:V, 0:csz], src=out_absorb_w[h, 0:V, cs:ce])
            oaT_psum = nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=oaT_psum[0:csz, 0:V], data=oa_sb[0:V, 0:csz])
            base = (h * nC + ci) * V
            nisa.tensor_copy(dst=oaT_sb_all[0:csz, base : base + V], src=oaT_psum[0:csz, 0:V])

    # Queries are shared by every score chunk.
    q_nope_all = nl.ndarray((_P_MAX, nC * B * Nh), dtype=nl.bfloat16, buffer=nl.sbuf)
    for ct in TiledRange(C, _P_MAX):
        cs, ce, csz = ct.start_offset, ct.end_offset, ct.size
        ci = cs // _P_MAX
        nisa.dma_copy(
            dst=q_nope_all[0:csz, ci * B * Nh : (ci + 1) * B * Nh],
            src=q_nope_absorbed_T[cs:ce, :, :],
        )
    q_pe_all = nl.ndarray((_P_MAX, B * Nh), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_pe_all[0:R, 0 : B * Nh], src=q_pe_T[0:R, :, :])

    for b in range(B):
        scores_sb = nl.ndarray((_P_MAX, Sk), dtype=nl.float32, buffer=nl.sbuf)
        kv_resident = nl.ndarray((_P_MAX, nC * Sk), dtype=nl.bfloat16, buffer=nl.sbuf)

        for ct in TiledRange(C, _P_MAX):
            cs, ce, csz = ct.start_offset, ct.end_offset, ct.size
            ci = cs // _P_MAX
            kv_base = ci * Sk
            nisa.dma_copy(
                dst=kv_resident[0:csz, kv_base : kv_base + S],
                src=ckv_prior_T[cs:ce, b, 0:S],
            )
            nisa.dma_copy(
                dst=kv_resident[0:csz, kv_base + S : kv_base + Sk],
                src=ckv_cur_T[cs:ce, b, 0:1],
            )

        # Score chunks. The dense causal iota and 64-wide keys are chunk-local,
        # avoiding context-sized mask and kpe buffers.
        for sct in TiledRange(Sk, _F_MAX):
            js, je, jsz = sct.start_offset, sct.end_offset, sct.size
            scores_psum = nl.ndarray((_P_MAX, _F_MAX), dtype=nl.float32, buffer=nl.psum)
            first_mm = True
            for ct in TiledRange(C, _P_MAX):
                cs, ce, csz = ct.start_offset, ct.end_offset, ct.size
                ci = cs // _P_MAX
                kv_base = ci * Sk
                nisa.nc_matmul(
                    dst=scores_psum[0:Nh, 0:jsz],
                    stationary=q_nope_all[
                        0:csz,
                        ci * B * Nh + b * Nh : ci * B * Nh + b * Nh + Nh,
                    ],
                    moving=kv_resident[0:csz, kv_base + js : kv_base + je],
                    accumulate=not first_mm,
                )
                first_mm = False

            kpe_chunk = nl.ndarray((_P_MAX, _F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            prior_je = je if je <= S else S
            if prior_je > js:
                nisa.dma_copy(
                    dst=kpe_chunk[0:R, 0 : prior_je - js],
                    src=kpe_prior_T[0:R, b, js:prior_je],
                )
            if je > S:
                nisa.dma_copy(
                    dst=kpe_chunk[0:R, S - js : S - js + 1],
                    src=kpe_cur_T[0:R, b, 0:1],
                )
            nisa.nc_matmul(
                dst=scores_psum[0:Nh, 0:jsz],
                stationary=q_pe_all[0:R, b * Nh : b * Nh + Nh],
                moving=kpe_chunk[0:R, 0:jsz],
                accumulate=not first_mm,
            )
            nisa.tensor_scalar(
                dst=scores_sb[0:Nh, js:je],
                data=scores_psum[0:Nh, 0:jsz],
                op0=nl.multiply,
                operand0=softmax_scale,
            )

            if prior_je > js:
                prior_width = prior_je - js
                slot_i = nl.ndarray((1, _F_MAX), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(
                    dst=slot_i[0:1, 0:prior_width],
                    pattern=[[1, prior_width]],
                    offset=js,
                    channel_multiplier=0,
                )
                slot_f = nl.ndarray((1, _F_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=slot_f[0:1, 0:prior_width], src=slot_i[0:1, 0:prior_width])
                diff = nl.ndarray((1, _F_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=diff[0:1, 0:prior_width],
                    data=slot_f[0:1, 0:prior_width],
                    op0=nl.subtract,
                    operand0=pos_all_f[0:1, b : b + 1],
                )
                keep = nl.ndarray((1, _F_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=keep[0:1, 0:prior_width],
                    data=diff[0:1, 0:prior_width],
                    op0=nl.less,
                    operand0=0.0,
                )
                pred = nl.ndarray((_P_MAX, _F_MAX), dtype=nl.int8, buffer=nl.sbuf)
                if Nh == 1:
                    nisa.tensor_copy(dst=pred[0:1, 0:prior_width], src=keep[0:1, 0:prior_width])
                else:
                    bcast_psum = nl.ndarray((_P_MAX, _F_MAX), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=bcast_psum[0:Nh, 0:prior_width],
                        stationary=ones_col[0:1, 0:Nh],
                        moving=keep[0:1, 0:prior_width],
                    )
                    nisa.tensor_copy(
                        dst=pred[0:Nh, 0:prior_width],
                        src=bcast_psum[0:Nh, 0:prior_width],
                    )
                nisa.select_reduce(
                    dst=scores_sb[0:Nh, js:prior_je],
                    predicate=pred[0:Nh, 0:prior_width],
                    on_true=scores_sb[0:Nh, js:prior_je],
                    on_false=_NEG_INF,
                )

        # Chunked fp32 softmax. Store BF16 exponentials and defer the scalar
        # denominator until after the weighted-latent matmul.
        neg_max = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        sum_sb = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_sb = nl.ndarray((_P_MAX, Sk), dtype=nl.bfloat16, buffer=nl.sbuf)
        chunk_max = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        first_chunk = True
        for sct in TiledRange(Sk, _F_MAX):
            js, je = sct.start_offset, sct.end_offset
            nisa.tensor_reduce(
                dst=chunk_max[0:Nh, 0:1],
                op=nl.maximum,
                data=scores_sb[0:Nh, js:je],
                axis=1,
                negate=True,
            )
            if first_chunk:
                nisa.tensor_copy(dst=neg_max[0:Nh, 0:1], src=chunk_max[0:Nh, 0:1])
                first_chunk = False
            else:
                nisa.tensor_tensor(
                    dst=neg_max[0:Nh, 0:1],
                    data1=neg_max[0:Nh, 0:1],
                    data2=chunk_max[0:Nh, 0:1],
                    op=nl.minimum,
                )

        first_chunk = True
        for sct in TiledRange(Sk, _F_MAX):
            js, je = sct.start_offset, sct.end_offset
            nisa.activation(
                dst=w_sb[0:Nh, js:je],
                op=nl.exp,
                data=scores_sb[0:Nh, js:je],
                bias=neg_max[0:Nh, 0:1],
                reduce_op=nl.add,
                reduce_res=sum_sb[0:Nh, 0:1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce if first_chunk else nisa.reduce_cmd.reduce,
            )
            first_chunk = False
        reciprocal = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=reciprocal[0:Nh, 0:1], data=sum_sb[0:Nh, 0:1])

        # Weighted latent. The feature-major resident cache is transposed one
        # sequence tile at a time for the Sk contraction.
        x_psum = nl.ndarray((_P_MAX, C), dtype=nl.float32, buffer=nl.psum)
        first_x_mm = True
        for st in TiledRange(Sk, _P_MAX):
            ss, se, ssz = st.start_offset, st.end_offset, st.size
            wT_psum = nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=wT_psum[0:ssz, 0:Nh], data=w_sb[0:Nh, ss:se])
            wT_sb = nl.ndarray((_P_MAX, Nh), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wT_sb[0:ssz, 0:Nh], src=wT_psum[0:ssz, 0:Nh])
            kv_token_major = nl.ndarray((_P_MAX, C), dtype=nl.bfloat16, buffer=nl.sbuf)
            for ct in TiledRange(C, _P_MAX):
                cs, ce, csz = ct.start_offset, ct.end_offset, ct.size
                ci = cs // _P_MAX
                kv_base = ci * Sk
                slabT_psum = nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=slabT_psum[0:ssz, 0:csz],
                    data=kv_resident[0:csz, kv_base + ss : kv_base + se],
                )
                nisa.tensor_copy(
                    dst=kv_token_major[0:ssz, cs:ce],
                    src=slabT_psum[0:ssz, 0:csz],
                )
            nisa.nc_matmul(
                dst=x_psum[0:Nh, 0:C],
                stationary=wT_sb[0:ssz, 0:Nh],
                moving=kv_token_major[0:ssz, 0:C],
                accumulate=not first_x_mm,
            )
            first_x_mm = False

        x_sb = nl.ndarray((_P_MAX, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=x_sb[0:Nh, 0:C],
            data=x_psum[0:Nh, 0:C],
            op0=nl.multiply,
            operand0=reciprocal[0:Nh, 0:1],
        )
        for ct in TiledRange(C, _P_MAX):
            cs, ce, csz = ct.start_offset, ct.end_offset, ct.size
            ci = cs // _P_MAX
            xT_psum = nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=xT_psum[0:csz, 0:Nh], data=x_sb[0:Nh, cs:ce])
            for h in range(Nh):
                col = (ci * Nh + h) * B + b
                nisa.tensor_copy(
                    dst=xT_all[0:csz, col : col + 1],
                    src=xT_psum[0:csz, h : h + 1],
                )

    # W_UV out-absorb for all batch rows.
    attn_all = nl.ndarray((_P_MAX, B * Nh), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in range(Nh):
        attn_psum = nl.ndarray((_P_MAX, B), dtype=nl.float32, buffer=nl.psum)
        first_attn_mm = True
        for ct in TiledRange(C, _P_MAX):
            cs, csz = ct.start_offset, ct.size
            ci = cs // _P_MAX
            base = (h * nC + ci) * V
            nisa.nc_matmul(
                dst=attn_psum[0:V, 0:B],
                stationary=oaT_sb_all[0:csz, base : base + V],
                moving=xT_all[0:csz, (ci * Nh + h) * B : (ci * Nh + h + 1) * B],
                accumulate=not first_attn_mm,
            )
            first_attn_mm = False
        nisa.tensor_copy(
            dst=attn_all[0:V, h * B : (h + 1) * B],
            src=attn_psum[0:V, 0:B],
        )

    # Gate AFTER out-absorb and BEFORE o-projection. Transposing each natural
    # [B, V] head slice produces the same [V, head*B + token] layout as attn_all.
    for h in range(Nh):
        gate_sb = nl.ndarray((_P_MAX, V), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gate_sb[0:B, 0:V],
            src=gate[0:B, h * V : (h + 1) * V],
        )
        gateT_psum = nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=gateT_psum[0:V, 0:B], data=gate_sb[0:B, 0:V])
        gateT_sb = nl.ndarray((_P_MAX, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gateT_sb[0:V, 0:B], src=gateT_psum[0:V, 0:B])
        nisa.tensor_tensor(
            dst=attn_all[0:V, h * B : (h + 1) * B],
            data1=attn_all[0:V, h * B : (h + 1) * B],
            data2=gateT_sb[0:V, 0:B],
            op=nl.multiply,
        )

    # Batched o-projection. Tile weights instead of retaining the whole
    # [Nh*V, H] matrix in SBUF, preserving context capacity without rereading
    # any weight element.
    out_sb = nl.ndarray((_P_MAX, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    for ht in TiledRange(H, _F_MAX):
        hs, he, hsz = ht.start_offset, ht.end_offset, ht.size
        out_psum = nl.ndarray((_P_MAX, _F_MAX), dtype=nl.float32, buffer=nl.psum)
        first_out_mm = True
        for h in range(Nh):
            o_proj_tile = nl.ndarray((_P_MAX, _F_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=o_proj_tile[0:V, 0:hsz],
                src=o_proj_w[h * V : (h + 1) * V, hs:he],
            )
            nisa.nc_matmul(
                dst=out_psum[0:B, 0:hsz],
                stationary=attn_all[0:V, h * B : (h + 1) * B],
                moving=o_proj_tile[0:V, 0:hsz],
                accumulate=not first_out_mm,
            )
            first_out_mm = False
        nisa.tensor_copy(dst=out_sb[0:B, hs:he], src=out_psum[0:B, 0:hsz])

    nisa.dma_copy(dst=output[0:B, 0:H], src=out_sb[0:B, 0:H])
    return output
