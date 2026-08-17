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

"""Specialized MLP TKG kernel for the Llama3-70B high-batch static-FP8 config.

Megatron-style tensor-parallel (I-shard) MLP TKG for B*S=256, H=8192, I=3584,
STATIC FP8 weights, RMS_NORM, SiLU, LNC=2.  Instead of T-sharding (both cores
redundantly stream the ENTIRE gate/up/down weights -> memory-bound), the
intermediate dimension I is split across the 2 cores: each core loads only its
I/2 weight slice (HALF the per-core weight DMA), computes a PARTIAL down output,
and a single cross-core reduce-scatter recombines the two partials.

Activation FP8 quant matches the golden's STATIC-quant contract for this config:
the gate/up input uses a dynamic per-token amax scale (which never saturates
here, so it equals the golden up to fp8 rounding), while the down input uses the
PROVIDED static down_in_scale with a +/-FP8_MAX clamp (the intermediate saturates
hard against that tiny scale, so a dynamic scale would diverge from the golden).
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import BufferManager
from ...utils.common_types import NormType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import (
    div_ceil,
    get_nl_act_fn_from_type,
    get_program_sharding_info,
)
from ...utils.tensor_view import TensorView
from ..mlp_parameters import MLPParameters

_PMAX = 128


def _load_scalar_scale(scale_hbm, out_sb):
    """Broadcast scalar scale[0,0] from a [128,1] HBM tensor to a [128,1] SBUF tile."""
    # partition stride 0 -> every partition reads element [0,0]
    nisa.dma_copy(
        dst=out_sb[0:_PMAX, 0:1],
        src=scale_hbm.ap(pattern=[[0, _PMAX], [0, 1]], offset=0),
    )


def _mlp_tkg_ishard(params, output_tensor_hbm, T, H, I, H0, H1, n_prgs, prg_id, out_dtype, do_norm, act_fn):
    """Megatron-style tensor-parallel MLP TKG: shard I across 2 cores.

    Each core loads the full hidden but only its I/2 weight slice, computes a
    partial down output, and a cross-core reduce-scatter recombines the two
    partials. Halves per-core weight DMA (the memory bottleneck).
    """
    FP8_MAX = 240.0
    I_local = I // n_prgs
    i_start = prg_id * I_local
    n_I = I_local // _PMAX
    n_pair_I = n_I // 2
    TB = T // _PMAX
    n_pair_h = H1 // 2

    hidden = params.hidden_tensor
    gate_w = params.gate_proj_weights_tensor
    up_w = params.up_proj_weights_tensor
    down_w = params.down_proj_weights_tensor

    gate_bias = params.bias_params.gate_proj_bias_tensor
    up_bias = params.bias_params.up_proj_bias_tensor
    down_bias = params.bias_params.down_proj_bias_tensor

    # ---- weight (per-tensor) dequant scales ----
    gscale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    uscale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    dscale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _load_scalar_scale(params.quant_params.gate_w_scale, gscale)
    _load_scalar_scale(params.quant_params.up_w_scale, uscale)
    _load_scalar_scale(params.quant_params.down_w_scale, dscale)

    # Both activation quantizations use the checkpoint's calibrated static scale
    # plus a clamp. STATIC quantization is defined to clamp, and a dynamic
    # per-token amax scale cannot clamp by construction, so it would compute a
    # different function than the reference.
    din_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _load_scalar_scale(params.quant_params.down_in_scale, din_scale)
    # Reciprocal is folded into the gate/up quant multiplier below.
    guin_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _load_scalar_scale(params.quant_params.gate_up_in_scale, guin_scale)
    guin_inv = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=guin_inv[0:_PMAX, 0:1], data=guin_scale[0:_PMAX, 0:1])
    # Fold 1/down_in_scale into the up dequant so inter_t is pre-scaled and the
    # down quant is a single fused clamp (fold into up, not gate: silu is nonlinear).
    uscale_din = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=uscale_din[0:_PMAX, 0:1], data=din_scale[0:_PMAX, 0:1])
    nisa.tensor_tensor(uscale_din[0:_PMAX, 0:1], uscale_din[0:_PMAX, 0:1], uscale[0:_PMAX, 0:1], nl.multiply)

    # per-token, per-token-block activation dequant scales (gate/up: dynamic amax)
    gate_in_scale = nl.ndarray((_PMAX, TB), dtype=nl.float32, buffer=nl.sbuf)
    combined_g = nl.ndarray((_PMAX, TB), dtype=nl.float32, buffer=nl.sbuf)
    combined_u = nl.ndarray((_PMAX, TB), dtype=nl.float32, buffer=nl.sbuf)
    # down dequant = down_w_scale * down_in_scale (static, per-tensor -> one column)
    combined_d = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(combined_d[0:_PMAX, 0:1], dscale[0:_PMAX, 0:1], din_scale[0:_PMAX, 0:1], nl.multiply)

    # hidden_sb[h0, tb, h1, t] = fp8 normed hidden; extra h1 plane (index H1) ships
    # the per-token hidden quant scale alongside the transposed block in one sendrecv.
    hidden_sb = nl.ndarray((H0, TB, H1 + 1, _PMAX), dtype=nl.float8_e4m3, buffer=nl.sbuf)

    # Pre-alloc the gate weight ring and kick off the first WRING slices now, so they
    # load during the DMA-idle hidden transpose + cross-core exchange below.
    I_CHUNK = 512
    n_ic = div_ceil(I_local, I_CHUNK)
    gw_dr_pat = [[I, H0], [H0 * I, 2], [1, I_local]]
    # WRING-deep ring so the DMA builds a prefetch lead that keeps the PE fed through
    # the gate/up stream; reused for the up pass, so SBUF stays flat at WRING slices.
    WRING = 16
    # Prefetch distance must be STRICTLY less than the ring depth: at distance ==
    # WRING, nxt % WRING == j % WRING, so the prefetch DMA overwrites the very slot
    # iteration j's matmul is about to read (H-pair j contracted against H-pair j+WRING).
    WLEAD = WRING - 1
    n_pre = min(WLEAD, n_pair_h)
    pre_gw_0 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_1 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_2 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_3 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_4 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_5 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_6 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_7 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_8 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_9 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_10 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_11 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_12 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_13 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_14 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw_15 = nl.ndarray((H0, 2, I_local), dtype=gate_w.dtype, buffer=nl.sbuf)
    pre_gw = [
        pre_gw_0,
        pre_gw_1,
        pre_gw_2,
        pre_gw_3,
        pre_gw_4,
        pre_gw_5,
        pre_gw_6,
        pre_gw_7,
        pre_gw_8,
        pre_gw_9,
        pre_gw_10,
        pre_gw_11,
        pre_gw_12,
        pre_gw_13,
        pre_gw_14,
        pre_gw_15,
    ]
    for s in range(n_pre):
        nisa.dma_copy(
            dst=pre_gw[s][0:H0, 0:2, 0:I_local], src=gate_w.ap(pattern=gw_dr_pat, offset=s * 2 * H0 * I + i_start)
        )

    if do_norm:
        gamma = params.norm_params.normalization_weights_tensor  # [1, H]
        gamma_f = nl.ndarray((H0, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        # Load gamma once as a [1, H] bf16 row and broadcast across partitions on
        # the PE (cheaper than a [128, H] broadcast DMA on the bandwidth-bound queue).
        gamma_row = nl.ndarray((1, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=gamma_row[0:1, 0:H], src=gamma.ap(pattern=[[0, 1], [1, H]], offset=0))
        ones_bc = nl.ndarray((1, H0), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(ones_bc, value=1.0)
        for gc in range(0, H, 512):
            gce = min(gc + 512, H)
            gtp = nl.ndarray((H0, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=gtp[0:H0, 0 : gce - gc], stationary=ones_bc[0:1, 0:H0], moving=gamma_row[0:1, gc:gce])
            nisa.tensor_copy(dst=gamma_f[0:H0, gc:gce], src=gtp[0:H0, 0 : gce - gc])

    # ---- hidden load / norm / fp8 quant / transpose ----
    # Shard the transpose: each core transposes only its own token block, then the
    # two cores exchange transposed slices so both hold the full hidden_sb.
    my_b = prg_id
    other_b = 1 - prg_id
    for tb in range(TB):
        if tb != my_b:
            continue
        t0 = tb * _PMAX
        t_sz = _PMAX
        ht = nl.ndarray((_PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ht[0:t_sz, 0:H], src=hidden.ap(pattern=[[H, t_sz], [1, H]], offset=t0 * H))
        # RMS-norm + per-token amax fp8 quant; the quant scale absorbs the norm
        # recip factor, so the separate normed=ht*recip pass over H cancels.
        recip = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        amax = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        absh = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
        if do_norm:
            w = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
            # Chunk-pipeline the gamma-multiply (Vector) and abs/amax (Scalar) so
            # they overlap, collapsing two serial full-H passes into ~one.
            NCH = 8
            CW = H // NCH
            amax_parts = nl.ndarray((_PMAX, NCH), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(NCH):
                cs = c * CW
                ce = cs + CW
                nisa.tensor_tensor(
                    w[0:t_sz, cs:ce], ht[0:t_sz, cs:ce], gamma_f[0:t_sz, cs:ce], nl.multiply, engine=nisa.engine.vector
                )
                nisa.activation(
                    absh[0:t_sz, cs:ce],
                    op=nl.abs,
                    data=w[0:t_sz, cs:ce],
                    reduce_op=nl.max,
                    reduce_res=amax_parts[0:t_sz, c : c + 1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
            nisa.tensor_reduce(amax[0:t_sz, 0:1], op=nl.max, data=amax_parts[0:t_sz, 0:NCH], axis=1, keepdims=True)
            src_tile = w
        else:
            # per-token fp8 quant of hidden tile (amax over the tile)
            nisa.activation(
                absh[0:t_sz, 0:H],
                op=nl.abs,
                data=ht[0:t_sz, 0:H],
                reduce_op=nl.max,
                reduce_res=amax[0:t_sz, 0:1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            src_tile = ht

        # NOTE: amax is dead once the static scale replaced it as the quant
        # multiplier. Left in place here rather than removed in a correctness fix;
        # a follow-up should drop it and the abs/reduce work that feeds it.
        nisa.tensor_scalar(dst=amax[0:t_sz, 0:1], data=amax[0:t_sz, 0:1], op0=nl.add, operand0=1e-12)
        # quant multiplier = 1/gate_up_in_scale, times the RMS reciprocal when the
        # norm is fused in.
        inv_scale = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if do_norm:
            sq_d = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
            ssum_d = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                sq_d[0:t_sz, 0:H],
                op=nl.square,
                data=ht[0:t_sz, 0:H],
                reduce_op=nl.add,
                reduce_res=ssum_d[0:t_sz, 0:1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            nisa.activation(recip[0:t_sz, 0:1], op=nl.rsqrt, data=ssum_d[0:t_sz, 0:1], scale=1.0 / H, bias=params.eps)
            # quant multiplier = recip / gate_up_in_scale
            nisa.tensor_tensor(inv_scale[0:t_sz, 0:1], recip[0:t_sz, 0:1], guin_inv[0:_PMAX, 0:1], nl.multiply)
        else:
            nisa.tensor_copy(dst=inv_scale[0:t_sz, 0:1], src=guin_inv[0:_PMAX, 0:1])
        qhid_b = nl.ndarray((_PMAX, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        QG = 4  # h1-slices per quant chunk
        for c0 in range(0, H1, QG):
            c1 = min(c0 + QG, H1)
            cs = c0 * H0
            ce = c1 * H0
            nisa.tensor_scalar(
                dst=qhid_b[0:t_sz, cs:ce],
                data=src_tile[0:t_sz, cs:ce],
                op0=nl.multiply,
                operand0=inv_scale[0:t_sz, 0:1],
                op1=nl.minimum,
                operand1=FP8_MAX,
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                dst=qhid_b[0:t_sz, cs:ce],
                data=qhid_b[0:t_sz, cs:ce],
                op0=nl.maximum,
                operand0=-FP8_MAX,
                engine=nisa.engine.vector,
            )
            for h1 in range(c0, c1):
                hs = h1 * H0
                tp = nl.ndarray((H0, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:H0, 0:t_sz], data=qhid_b[0:t_sz, hs : hs + H0])
                nisa.tensor_copy(dst=hidden_sb[0:H0, tb, h1, 0:t_sz], src=tp[0:H0, 0:t_sz])
        # Dequant by the same static per-tensor scale. The RMS reciprocal is absent
        # here because it was folded into the quant multiplier above.
        nisa.tensor_copy(dst=gate_in_scale[0:t_sz, tb : tb + 1], src=guin_scale[0:_PMAX, 0:1])
        nisa.tensor_tensor(
            combined_g[0:t_sz, tb : tb + 1], gscale[0:t_sz, 0:1], gate_in_scale[0:t_sz, tb : tb + 1], nl.multiply
        )
        nisa.tensor_tensor(
            combined_u[0:t_sz, tb : tb + 1], uscale_din[0:t_sz, 0:1], gate_in_scale[0:t_sz, tb : tb + 1], nl.multiply
        )

    # Pack the per-token hidden quant scale (fp32) into the extra plane so it rides
    # with the transposed hidden in one sendrecv (4 fp8 cols reinterpreted as 1 fp32).
    my_scale_slot = (
        TensorView(hidden_sb)
        .slice(1, my_b, my_b + 1)
        .slice(2, H1, H1 + 1)
        .slice(3, 0, 4)
        .reinterpret_cast(nl.float32)
        .get_view()
    )
    my_scale_src = TensorView(gate_in_scale).slice(1, my_b, my_b + 1).reshape_dim(1, (1, 1, 1)).get_view()
    nisa.tensor_copy(dst=my_scale_slot, src=my_scale_src)

    # exchange transposed hidden slice + packed hidden quant scale in ONE sendrecv
    nisa.sendrecv(
        src=hidden_sb[0:H0, my_b, 0 : H1 + 1, 0:_PMAX],
        dst=hidden_sb[0:H0, other_b, 0 : H1 + 1, 0:_PMAX],
        send_to_rank=other_b,
        recv_from_rank=other_b,
        pipe_id=1,
    )
    # unpack the received scale back into gate_in_scale[:, other_b]
    other_scale_slot = (
        TensorView(hidden_sb)
        .slice(1, other_b, other_b + 1)
        .slice(2, H1, H1 + 1)
        .slice(3, 0, 4)
        .reinterpret_cast(nl.float32)
        .get_view()
    )
    other_scale_dst = TensorView(gate_in_scale).slice(1, other_b, other_b + 1).reshape_dim(1, (1, 1, 1)).get_view()
    nisa.tensor_copy(dst=other_scale_dst, src=other_scale_slot)
    # recompute combined gate/up dequant scales for the received (other) block
    nisa.tensor_tensor(
        combined_g[0:_PMAX, other_b : other_b + 1],
        gscale[0:_PMAX, 0:1],
        gate_in_scale[0:_PMAX, other_b : other_b + 1],
        nl.multiply,
    )
    nisa.tensor_tensor(
        combined_u[0:_PMAX, other_b : other_b + 1],
        uscale_din[0:_PMAX, 0:1],
        gate_in_scale[0:_PMAX, other_b : other_b + 1],
        nl.multiply,
    )

    # ---- gate / up projections (weights streamed once, both token blocks) ----

    gate_sb = nl.ndarray((_PMAX, TB, I_local), dtype=nl.float32, buffer=nl.sbuf)
    up_sb = nl.ndarray((_PMAX, TB, I_local), dtype=nl.float32, buffer=nl.sbuf)

    # pass_id 0 -> gate (apply activation), 1 -> up (no activation)
    for pass_id in range(2):
        w_tensor = gate_w if pass_id == 0 else up_w
        out_sb = gate_sb if pass_id == 0 else up_sb
        combined = combined_g if pass_id == 0 else combined_u
        bias = gate_bias if pass_id == 0 else up_bias
        do_act = pass_id == 0
        # both passes share the same WRING-deep buffer ring
        gw = pre_gw
        ps = nl.ndarray((_PMAX, TB, n_ic, I_CHUNK), dtype=nl.float32, buffer=nl.psum)
        if pass_id != 0:
            for s in range(min(WLEAD, n_pair_h)):
                nisa.dma_copy(
                    dst=gw[s][0:H0, 0:2, 0:I_local], src=w_tensor.ap(pattern=gw_dr_pat, offset=s * 2 * H0 * I + i_start)
                )
        for j in range(n_pair_h):
            gwh = gw[j % WRING]
            nxt = j + WLEAD
            if nxt < n_pair_h:
                nisa.dma_copy(
                    dst=gw[nxt % WRING][0:H0, 0:2, 0:I_local],
                    src=w_tensor.ap(pattern=gw_dr_pat, offset=nxt * 2 * H0 * I + i_start),
                )
            for tb in range(TB):
                for ic in range(n_ic):
                    ic0 = ic * I_CHUNK
                    ic_sz = min(I_CHUNK, I_local - ic0)
                    nisa.nc_matmul(
                        dst=ps[0:_PMAX, tb, ic, 0:ic_sz],
                        stationary=hidden_sb[0:H0, tb, 2 * j : 2 * j + 2, 0:_PMAX],
                        moving=gwh[0:H0, 0:2, ic0 : ic0 + ic_sz],
                        perf_mode=nisa.matmul_perf_mode.double_row,
                    )
        for tb in range(TB):
            for ic in range(n_ic):
                ic0 = ic * I_CHUNK
                ic_sz = min(I_CHUNK, I_local - ic0)
                if bias is None and do_act:
                    nisa.activation(
                        out_sb[0:_PMAX, tb, ic0 : ic0 + ic_sz],
                        op=act_fn,
                        data=ps[0:_PMAX, tb, ic, 0:ic_sz],
                        scale=combined[0:_PMAX, tb : tb + 1],
                    )
                else:
                    nisa.tensor_scalar(
                        dst=out_sb[0:_PMAX, tb, ic0 : ic0 + ic_sz],
                        data=ps[0:_PMAX, tb, ic, 0:ic_sz],
                        op0=nl.multiply,
                        operand0=combined[0:_PMAX, tb : tb + 1],
                    )
                    if bias is not None:
                        bb = nl.ndarray((_PMAX, I_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(
                            dst=bb[0:_PMAX, 0:ic_sz],
                            src=bias.ap(pattern=[[0, _PMAX], [1, ic_sz]], offset=i_start + ic0),
                        )
                        nisa.tensor_tensor(
                            out_sb[0:_PMAX, tb, ic0 : ic0 + ic_sz],
                            out_sb[0:_PMAX, tb, ic0 : ic0 + ic_sz],
                            bb[0:_PMAX, 0:ic_sz],
                            nl.add,
                        )
                    if do_act:
                        nisa.activation(
                            out_sb[0:_PMAX, tb, ic0 : ic0 + ic_sz],
                            op=act_fn,
                            data=out_sb[0:_PMAX, tb, ic0 : ic0 + ic_sz],
                        )

    # ---- inter = act(gate) * up ; fp8 quant per token ; transpose to [I, T] ----
    # tb=0 is computed fully; tb=1's transpose is deferred into the down loop so it
    # overlaps tb=0's down matmuls (hiding the Vector quant behind PE work).
    inter_sb = nl.ndarray((_PMAX, n_I, TB, _PMAX), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    # persistent bf16 quantized inter for the DEFERRED block (tb=1)
    inter_qb_def = nl.ndarray((_PMAX, I_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    for tb in range(TB):
        inter_t = nl.ndarray((_PMAX, I_local), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            inter_t[0:_PMAX, 0:I_local], gate_sb[0:_PMAX, tb, 0:I_local], up_sb[0:_PMAX, tb, 0:I_local], nl.multiply
        )
        # Down fp8 quant: inter_t is already inter/down_in_scale (folded into the up
        # dequant), so this is a single fused clamp to +/-FP8_MAX.
        if tb == 0:
            # tb=0: quant + transpose now (chunk transpose over next chunk's quant).
            inter_qb = nl.ndarray((_PMAX, I_local), dtype=nl.bfloat16, buffer=nl.sbuf)
            QGI = 4
            for c0 in range(0, n_I, QGI):
                c1 = min(c0 + QGI, n_I)
                cs = c0 * _PMAX
                ce = c1 * _PMAX
                nisa.tensor_scalar(
                    dst=inter_qb[0:_PMAX, cs:ce],
                    data=inter_t[0:_PMAX, cs:ce],
                    op0=nl.minimum,
                    operand0=FP8_MAX,
                    op1=nl.maximum,
                    operand1=-FP8_MAX,
                )
                for it in range(c0, c1):
                    s0 = it * _PMAX
                    tp = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp[0:_PMAX, 0:_PMAX], data=inter_qb[0:_PMAX, s0 : s0 + _PMAX])
                    nisa.tensor_copy(dst=inter_sb[0:_PMAX, it, tb, 0:_PMAX], src=tp[0:_PMAX, 0:_PMAX])
        else:
            # tb=1: quant only; transpose deferred into the down loop to overlap tb=0 matmuls.
            nisa.tensor_scalar(
                dst=inter_qb_def[0:_PMAX, 0:I_local],
                data=inter_t[0:_PMAX, 0:I_local],
                op0=nl.minimum,
                operand0=FP8_MAX,
                op1=nl.maximum,
                operand1=-FP8_MAX,
            )

    # ---- down projection -> PARTIAL out[t, h] ; reduce-scatter ; store ----
    H_CHUNK = 512
    n_hchunk = div_ceil(H, H_CHUNK)
    dw_pat = [[H, _PMAX], [_PMAX * H, n_I], [1, H_CHUNK]]

    my_b = prg_id
    other_b = 1 - prg_id

    GROUP = 4  # H chunks per reduce-scatter group
    CH = GROUP * H_CHUNK
    n_group = div_ceil(n_hchunk, GROUP)

    part_g0 = nl.ndarray((_PMAX, TB, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
    part_g1 = nl.ndarray((_PMAX, TB, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
    part_g = [part_g0, part_g1]
    recv_g0 = nl.ndarray((_PMAX, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
    recv_g1 = nl.ndarray((_PMAX, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
    recv_g = [recv_g0, recv_g1]

    # dw ring: two ping-pong sets of GROUP buffers; both token blocks reuse the same
    # group-resident dw slices (down weights streamed once).
    dw_ring = []
    for _dwi in range(2 * GROUP):
        dw_ring.append(nl.ndarray((_PMAX, n_I, H_CHUNK), dtype=down_w.dtype, buffer=nl.sbuf))
    # prefetch group 0's dw chunks
    for cc in range(GROUP):
        hc = cc
        if hc < n_hchunk:
            hb = hc * H_CHUNK
            h_sz = min(H_CHUNK, H - hb)
            nisa.dma_copy(
                dst=dw_ring[cc][0:_PMAX, 0:n_I, 0:h_sz], src=down_w.ap(pattern=dw_pat, offset=i_start * H + hb)
            )

    for g in range(n_group):
        base = (g % 2) * GROUP
        pbuf = part_g[g % 2]
        gb = g * CH
        g_sz = min(CH, H - gb)
        # prefetch NEXT group's dw chunks into the other ping-pong set
        if g + 1 < n_group:
            nbase = ((g + 1) % 2) * GROUP
            for cc in range(GROUP):
                hc = (g + 1) * GROUP + cc
                if hc < n_hchunk:
                    hb = hc * H_CHUNK
                    h_sz = min(H_CHUNK, H - hb)
                    nisa.dma_copy(
                        dst=dw_ring[nbase + cc][0:_PMAX, 0:n_I, 0:h_sz],
                        src=down_w.ap(pattern=dw_pat, offset=i_start * H + hb),
                    )

        # ---- tb=0 down matmuls (issue while tb=1's quant is still on Vector) ----
        for cc in range(GROUP):
            hc = g * GROUP + cc
            if hc >= n_hchunk:
                continue
            hb = hc * H_CHUNK
            h_sz = min(H_CHUNK, H - hb)
            loc = cc * H_CHUNK
            dw = dw_ring[base + cc]
            down_ps = nl.ndarray((_PMAX, H_CHUNK), dtype=nl.float32, buffer=nl.psum)
            for j in range(n_pair_I):
                nisa.nc_matmul(
                    dst=down_ps[0:_PMAX, 0:h_sz],
                    stationary=inter_sb[0:_PMAX, 2 * j : 2 * j + 2, 0, 0:_PMAX],
                    moving=dw[0:_PMAX, 2 * j : 2 * j + 2, 0:h_sz],
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )
            nisa.tensor_scalar(
                dst=pbuf[0:_PMAX, 0, loc : loc + h_sz],
                data=down_ps[0:_PMAX, 0:h_sz],
                op0=nl.multiply,
                operand0=combined_d[0:_PMAX, 0:1],
            )

        # ---- deferred tb=1 transpose (group 0 only): overlapped by the tb=0
        # matmuls above, so tb=1's quant->transpose is off the critical path ----
        if g == 0:
            for it in range(n_I):
                s0 = it * _PMAX
                tp = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:_PMAX, 0:_PMAX], data=inter_qb_def[0:_PMAX, s0 : s0 + _PMAX])
                nisa.tensor_copy(dst=inter_sb[0:_PMAX, it, 1, 0:_PMAX], src=tp[0:_PMAX, 0:_PMAX])

        # ---- tb=1 down matmuls (reuse this group's resident dw) ----
        for cc in range(GROUP):
            hc = g * GROUP + cc
            if hc >= n_hchunk:
                continue
            hb = hc * H_CHUNK
            h_sz = min(H_CHUNK, H - hb)
            loc = cc * H_CHUNK
            dw = dw_ring[base + cc]
            down_ps = nl.ndarray((_PMAX, H_CHUNK), dtype=nl.float32, buffer=nl.psum)
            for j in range(n_pair_I):
                nisa.nc_matmul(
                    dst=down_ps[0:_PMAX, 0:h_sz],
                    stationary=inter_sb[0:_PMAX, 2 * j : 2 * j + 2, 1, 0:_PMAX],
                    moving=dw[0:_PMAX, 2 * j : 2 * j + 2, 0:h_sz],
                    perf_mode=nisa.matmul_perf_mode.double_row,
                )
            nisa.tensor_scalar(
                dst=pbuf[0:_PMAX, 1, loc : loc + h_sz],
                data=down_ps[0:_PMAX, 0:h_sz],
                op0=nl.multiply,
                operand0=combined_d[0:_PMAX, 0:1],
            )

        # ---- reduce-scatter for this group: send OTHER core's block partial,
        # recv MY block's partial from the other core (this group's H slice) ----
        rbuf = recv_g[g % 2]
        nisa.sendrecv(
            src=pbuf[0:_PMAX, other_b, 0:g_sz],
            dst=rbuf[0:_PMAX, 0:g_sz],
            send_to_rank=other_b,
            recv_from_rank=other_b,
            pipe_id=g % 2,
        )

        # ---- add + bias + cast + store this group's chunks ----
        for cc in range(GROUP):
            hc = g * GROUP + cc
            if hc >= n_hchunk:
                continue
            hb = hc * H_CHUNK
            h_sz = min(H_CHUNK, H - hb)
            loc = cc * H_CHUNK
            out_full = nl.ndarray((_PMAX, H_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                out_full[0:_PMAX, 0:h_sz],
                pbuf[0:_PMAX, my_b, loc : loc + h_sz],
                rbuf[0:_PMAX, loc : loc + h_sz],
                nl.add,
            )
            if down_bias is not None:
                db = nl.ndarray((_PMAX, H_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=db[0:_PMAX, 0:h_sz], src=down_bias.ap(pattern=[[0, _PMAX], [1, h_sz]], offset=hb))
                nisa.tensor_tensor(out_full[0:_PMAX, 0:h_sz], out_full[0:_PMAX, 0:h_sz], db[0:_PMAX, 0:h_sz], nl.add)
            out_sb = nl.ndarray((_PMAX, H_CHUNK), dtype=out_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb[0:_PMAX, 0:h_sz], src=out_full[0:_PMAX, 0:h_sz])
            nisa.dma_copy(
                dst=output_tensor_hbm.ap(pattern=[[H, _PMAX], [1, h_sz]], offset=my_b * _PMAX * H + hb),
                src=out_sb[0:_PMAX, 0:h_sz],
            )

    return [output_tensor_hbm]


def mlp_tkg_llama3_70b_high_batch(
    params: MLPParameters,
    output_tensor_hbm,
    output_stored_add_tensor_hbm,
    sbm: Optional[BufferManager] = None,
) -> list:
    """Entry point for the Llama3-70B high-batch I-shard MLP TKG kernel.

    Dispatched from ``mlp_tkg`` for the specialized config (B*S=256, H=8192,
    I=3584, STATIC FP8, RMS_NORM, SiLU, LNC=2).  Derives the kernel dims from
    ``params`` and runs the I-sharded pipeline in ``_mlp_tkg_ishard``.
    """
    T = params.batch_size * params.sequence_len
    H = params.hidden_size
    I = params.intermediate_size
    H0 = _PMAX
    H1 = H // H0
    n_I = div_ceil(I, _PMAX)

    _, n_prgs, prg_id = get_program_sharding_info()
    out_dtype = params.output_dtype
    do_norm = params.norm_params.normalization_type != NormType.NO_NORM
    act_fn = get_nl_act_fn_from_type(params.activation_fn)

    # Backstop guards for the I-shard double_row layout (the caller only dispatches
    # matching configs here).
    kernel_assert(H % H0 == 0, "H must be a multiple of 128")
    kernel_assert(I % _PMAX == 0, "I must be a multiple of 128 for this implementation")
    kernel_assert(n_prgs == 2, "high-batch I-shard kernel requires LNC=2")
    kernel_assert(T % _PMAX == 0 and T // _PMAX == n_prgs, "requires T == n_prgs * 128")
    kernel_assert(
        I % (n_prgs * _PMAX) == 0 and (I // (n_prgs * _PMAX)) % 2 == 0,
        "requires I divisible by n_prgs*256 (double_row)",
    )

    return _mlp_tkg_ishard(
        params,
        output_tensor_hbm,
        T,
        H,
        I,
        H0,
        H1,
        n_prgs,
        prg_id,
        out_dtype,
        do_norm,
        act_fn,
    )
