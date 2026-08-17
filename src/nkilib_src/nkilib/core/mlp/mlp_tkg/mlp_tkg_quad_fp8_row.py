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

"""MLP TKG kernel implementation for ROW_MX (row-wise FP8) quantization."""

import nki.isa as nisa
import nki.language as nl

from ...quantization.fp8_quantize import row_quantization
from ...subkernels.norm_tkg_utils import _DGE_MODE_NONE, pe_transpose
from ...subkernels.rmsnorm_tkg import rmsnorm_tkg_th
from ...utils.allocator import SbufManager
from ...utils.common_types import HiddenLayout
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type, get_verified_program_sharding_info
from ...utils.logging import get_logger
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import (
    BS_TILE_SIZE,
    MLPParameters,
    mlpp_has_normalization,
    mlpp_has_rms_normalization,
    mlpp_store_fused_add,
)
from .down_projection_mx_shard_H import down_projection_mx_tp_shard_H
from .gate_up_projection_mx_shard_H import gate_up_projection_mx_tp_shard_H
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_mx_utils import (
    alloc_dummy_scale_tile,
    load_down_bias_mx,
    load_down_weight,
    load_gate_up_bias_mx,
    load_gate_up_weight,
)
from .mlp_tkg_rmsnorm import rmsnorm_tkg
from .projection_mx_constants import ProjConfig

# SBUF budget reserved for RMSNorm / outer-level SbufManager (200 KiB).
_RMSNORM_SBUF_BUDGET_BYTES = 200 * 1024

# I-dimension tile size for ROW_MX path.
# Chosen to balance SBUF occupancy (gate/up/down weight tiles + activations + per-token
# dequant scale cache) against DMA bandwidth; empirically 1536 gives the best QoR
# across supported model configs.
_I_TILE_SIZE = 1536


class RowMxPreallocBuffers(nl.NKIObject):
    """Pre-allocated SBUF buffers for ROW_MX, reused across calls."""

    def __init__(self):
        self.dummy_scale_tile = None
        self.gate_w_dequant_sb = None
        self.up_w_dequant_sb = None
        self.down_w_dequant_sb = None
        self.down_out_sb = None  # Set by impl when skip_output_store=True
        self.inp_qtz_cache_buf = None  # Set by caller; impl writes quantized input here on first I-tile
        self.inp_dequant_scale_cache_buf = None  # Set by caller; impl writes ROW_MX dequant scale here on first I-tile


def _mlp_tkg_row_mx_impl(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
    name_prefix: str = "",
    sbm: SbufManager = None,
    mx_buf: RowMxPreallocBuffers = None,
    inp_qtz_in: nl.NkiTensor = None,
    skip_norm: bool = False,
    skip_output_store: bool = False,
    activation_op=None,
) -> list[nl.NkiTensor]:
    """
    MLP TKG kernel with ROW_MX (row-wise / dynamic) FP8 quantization.

    Uses software row-wise (dynamic) FP8 quantization via
    ``row_quantization()``, combined with dummy MX scales set to 127.

    Activation quantization (ROW_MX / row-wise / dynamic):
        Per-token scale computed at runtime::

            absmax = max(abs(hidden), dim=-1)          # bf16[BxS]
            dequant_scale = absmax / MAXVAL            # bf16[BxS]
            quant_scale   = 1 / dequant_scale          # bf16[BxS]
            quantized_input = hidden * quant_scale     # fp8[BxS, H]

        ``input_dequant_scale`` shape: ``[_pmax, BxS, 1]`` (rank-3, per-token).

    Args:
        params (MLPParameters): MLP configuration with FP8 quantized weights.
        output_tensor_hbm (nl.NkiTensor): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.NkiTensor): Optional fused add output in HBM

    Returns:
        list[nl.NkiTensor]:
            - [output_tensor_hbm] when store_output_in_sbuf=False
            - [down_out_sb] when store_output_in_sbuf=True
    """
    io_dtype = params.hidden_tensor.dtype

    # Validate inputs
    kernel_assert(
        params.quant_params.is_quant_row_mx(),
        "mlp_tkg_quad_fp8_row requires ROW_MX quantization",
    )
    kernel_assert(
        not mlpp_has_normalization(params) or mlpp_has_rms_normalization(params),
        "mlp_tkg_quad_fp8_row only supports RMSNorm or no normalization",
    )

    # Compute kernel dimensions
    dims = MLPTKGConstants.calculate_constants(params)

    # MX quantization constants from dims (_pmax=128, _q_width=4)
    _pmax = dims._pmax
    _q_width = dims._q_width

    # Flag for contiguous x4 gate/up weight packing
    use_contiguous_x4 = params.use_contiguous_x4_gate_up

    # Section 1: Normalization (Optional)
    hidden_tensor = params.hidden_tensor
    if not skip_norm and mlpp_has_normalization(params):
        if mlpp_has_rms_normalization(params):
            if not use_contiguous_x4:
                # Software quant path: rmsnorm_tkg produces sharded output [H0, T, H1_shard]
                H1_shard = dims.H_per_shard // _pmax
                rmsnorm_out = nl.ndarray((dims.H0, dims.T * H1_shard), dtype=io_dtype, buffer=nl.sbuf)
                old_prefix = sbm.get_name_prefix()
                sbm.set_name_prefix(name_prefix)
                rmsnorm_out, rmsnorm_layout = rmsnorm_tkg(
                    input=params.hidden_tensor,
                    gamma=params.norm_params.normalization_weights_tensor,
                    output=rmsnorm_out,
                    hidden_scale=1.0 / dims.H,
                    eps=params.eps,
                    hidden_dim_tp=True,
                    sbm=sbm,
                )
                sbm.set_name_prefix(old_prefix)
                if rmsnorm_layout == HiddenLayout.H0_H1_T:
                    # _th path outputs [H0, H1_shard, T] — permute to [H0, T, H1_shard]
                    rmsnorm_permuted = nl.ndarray((_pmax, dims.T, H1_shard), dtype=io_dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=rmsnorm_permuted,
                        src=rmsnorm_out.permute(dims=[0, 2, 1]),
                    )
                    hidden_tensor = rmsnorm_permuted
                else:
                    hidden_tensor = rmsnorm_out
            else:
                n_H512_sharded = dims.H_per_shard // (_pmax * _q_width)
                rmsnorm_out = nl.ndarray((_pmax, n_H512_sharded * _q_width, dims.T), dtype=io_dtype, buffer=nl.sbuf)
                norm_weights = params.norm_params.normalization_weights_tensor
                eps = params.eps
                hidden_flat = params.hidden_tensor
                if len(hidden_flat.shape) == 3 and hidden_flat.buffer != nl.sbuf:
                    hidden_flat = hidden_flat.flatten_dims(start_dim=0, end_dim=1)
                elif hidden_flat.buffer == nl.sbuf:
                    kernel_assert(False, "contiguous_x4 rmsnorm from SBUF input not yet supported")
                rmsnorm_sbm = SbufManager(0, _RMSNORM_SBUF_BUDGET_BYTES, get_logger("rmsnorm_cx4"), use_auto_alloc=True)
                rmsnorm_sbm.set_name_prefix(name_prefix)
                rmsnorm_tkg_th(
                    input_hbm=hidden_flat,
                    gamma=norm_weights,
                    output=rmsnorm_out,
                    num_H_shards=dims.num_shards,
                    hidden_actual=dims.H,
                    eps=eps,
                    sbm=rmsnorm_sbm,
                    use_contiguous_x4=True,
                )
                hidden_tensor = rmsnorm_out
        else:
            kernel_assert(False, "mlp_tkg_quad_fp8_row only supports RMSNorm, LayerNorm is not supported")

    # Section 2: Input Quantization and x4 Packing
    n_H512_tile_sharded = dims.H_per_shard // (_pmax * _q_width)
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))
    T_padded = div_ceil(dims.T, _q_width) * _q_width

    H_sharded = dims.H // dims.num_shards

    dummy_scale_tile = mx_buf.dummy_scale_tile

    inp_qtz = None

    if inp_qtz_in != None:
        # Caller provided pre-quantized input — skip quantization entirely
        inp_qtz = inp_qtz_in
        # ROW_MX: load cached per-token dequant scale from previous I-tile
        input_dequant_scale = mx_buf.inp_dequant_scale_cache_buf
    elif params.input_in_sbuf or mlpp_has_rms_normalization(params):
        # SBUF path
        # When rmsnorm is active (non-contiguous_x4): hidden_tensor is [H0, T, H1_shard] (already sharded).
        # When input_in_sbuf without rmsnorm: hidden_tensor is [H0, T, H1] (full H, needs slice).
        input_already_sharded = mlpp_has_rms_normalization(params) and not use_contiguous_x4

        if use_contiguous_x4 and mlpp_has_rms_normalization(params):
            H1_shard_cx4 = n_H512_tile_sharded * _q_width
            hidden_permuted = nl.ndarray((_pmax, dims.T, H1_shard_cx4), dtype=io_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=hidden_permuted,
                src=hidden_tensor.permute(dims=[0, 2, 1]),
            )
            hidden_tensor = hidden_permuted

        quantized_input, input_dequant_scale_raw = row_quantization(
            hidden_tensor,
            output_dtype=nl.float8_e4m3fn,
        )
        # Pad dequant_scale from [_pmax, T, 1] to [_pmax, T_padded, 1]
        if T_padded > dims.T:
            input_dequant_scale = nl.ndarray((_pmax, T_padded, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=input_dequant_scale, value=0.0)
            nisa.tensor_copy(
                dst=input_dequant_scale[:, : dims.T, :],
                src=input_dequant_scale_raw,
            )
        else:
            input_dequant_scale = input_dequant_scale_raw

        if not use_contiguous_x4:
            if input_already_sharded:
                # hidden_tensor is already [H0, T, H1_shard] from rmsnorm —
                # reshape directly with n_H512_tile_sharded, no slice needed.
                qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_tile_sharded, _q_width))
                qtz_x4_4d = qtz_4d.view(nl.float8_e4m3fn_x4)
                qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_tile_sharded))

                # Permute [H0, T, n_H512_sharded] → [H0, n_H512_sharded, T]
                src_perm_x4 = qtz_x4.permute(dims=[0, 2, 1])
            else:
                # input_in_sbuf without rmsnorm: hidden_tensor is [H0, T, H1] (full H).
                # Reinterpret fp8 → fp8_x4, slice for shard, permute on x4 data.
                n_H512_total = n_H512_tile_sharded * dims.num_shards
                qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_total, _q_width))
                qtz_x4_4d = qtz_4d.view(nl.float8_e4m3fn_x4)
                qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_total))

                # Slice for shard, then permute — all on x4 (4× fewer free-dim elements)
                src_perm_x4 = (
                    qtz_x4.slice(
                        dim=2,
                        start=dims.shard_id * n_H512_tile_sharded,
                        end=(dims.shard_id + 1) * n_H512_tile_sharded,
                    ).permute(dims=[0, 2, 1])  # [H0, n_H512_sharded, T]
                )

            inp_qtz = nl.ndarray((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            nisa.memset(dst=inp_qtz, value=0)
            nisa.tensor_copy(
                dst=inp_qtz[:, :, : dims.T],
                src=src_perm_x4,
            )
        else:
            qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_tile_sharded, _q_width))
            qtz_x4_4d = qtz_4d.view(nl.float8_e4m3fn_x4)
            qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_tile_sharded))

            src_perm_x4 = qtz_x4.permute(dims=[0, 2, 1])

            inp_qtz = nl.ndarray((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            nisa.memset(dst=inp_qtz, value=0)
            nisa.tensor_copy(
                dst=inp_qtz[:, :, : dims.T],
                src=src_perm_x4,
            )

    else:
        # HBM path: load → quantize → x4 packing
        H1_shard = dims.H_per_shard // _pmax
        hidden_tensor = hidden_tensor.reshape((dims.T, dims.H))

        if not use_contiguous_x4:
            input_view = (
                hidden_tensor.reshape_dim(dim=1, shape=[dims.num_shards, H1_shard, _pmax])
                .permute(dims=[3, 0, 1, 2])
                .select(dim=2, index=dims.shard_id)
            )
            input_sb = nl.ndarray((_pmax, dims.T, H1_shard), dtype=io_dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=input_view, dst=input_sb)

            quantized_input, input_dequant_scale_raw = row_quantization(
                input_sb,
                output_dtype=nl.float8_e4m3fn,
            )
            if T_padded > dims.T:
                input_dequant_scale = nl.ndarray((_pmax, T_padded, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=input_dequant_scale, value=0.0)
                nisa.tensor_copy(
                    dst=input_dequant_scale[:, : dims.T, :],
                    src=input_dequant_scale_raw,
                )
            else:
                input_dequant_scale = input_dequant_scale_raw

            qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_tile_sharded, _q_width))
            qtz_x4_4d = qtz_4d.view(nl.float8_e4m3fn_x4)
            qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_tile_sharded))

            src_perm_x4 = qtz_x4.permute(dims=[0, 2, 1])

            inp_qtz = nl.ndarray((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            nisa.memset(dst=inp_qtz, value=0)
            nisa.tensor_copy(
                dst=inp_qtz[:, :, : dims.T],
                src=src_perm_x4,
            )
        else:
            # contiguous_x4 (HBM no-norm): contiguous load + on-chip transpose + quantize
            H_shard_local = _pmax * H1_shard

            input_hbm_shard = hidden_tensor.slice(
                dim=1, start=dims.shard_id * H_shard_local, end=(dims.shard_id + 1) * H_shard_local
            )
            input_th_sb = nl.ndarray((dims.T, H_shard_local), dtype=io_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=input_th_sb, src=input_hbm_shard, dge_mode=_DGE_MODE_NONE)

            src_4d_hbm = input_th_sb.reshape_dim(dim=1, shape=[n_H512_tile_sharded, _pmax, _q_width])
            input_th_perm = nl.ndarray((dims.T, H_shard_local), dtype=io_dtype, buffer=nl.sbuf)
            dst_4d_hbm = input_th_perm.reshape_dim(dim=1, shape=[_pmax, n_H512_tile_sharded, _q_width])
            nisa.tensor_copy(
                dst=dst_4d_hbm,
                src=src_4d_hbm.permute(dims=[0, 2, 1, 3]),
            )

            perm_src_3d_hbm = input_th_perm.reshape_dim(dim=1, shape=[_pmax, n_H512_tile_sharded * _q_width])
            cx4_xpose_tile = nl.ndarray((_pmax, dims.T, n_H512_tile_sharded * _q_width), dtype=io_dtype, buffer=nl.sbuf)
            pe_transpose(
                src=perm_src_3d_hbm,
                dst=cx4_xpose_tile,
                tile_size=_pmax,
                dtype=io_dtype,
                sbm=sbm,
            )

            quantized_input, input_dequant_scale_raw = row_quantization(
                cx4_xpose_tile,
                output_dtype=nl.float8_e4m3fn,
            )
            if T_padded > dims.T:
                input_dequant_scale = nl.ndarray((_pmax, T_padded, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=input_dequant_scale, value=0.0)
                nisa.tensor_copy(
                    dst=input_dequant_scale[:, : dims.T, :],
                    src=input_dequant_scale_raw,
                )
            else:
                input_dequant_scale = input_dequant_scale_raw

            qtz_4d = quantized_input.reshape((_pmax, dims.T, n_H512_tile_sharded, _q_width))
            qtz_x4_4d = qtz_4d.view(nl.float8_e4m3fn_x4)
            qtz_x4 = qtz_x4_4d.reshape((_pmax, dims.T, n_H512_tile_sharded))
            src_perm_x4 = qtz_x4.permute(dims=[0, 2, 1])

            inp_qtz = nl.ndarray((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            nisa.memset(dst=inp_qtz, value=0)
            nisa.tensor_copy(
                dst=inp_qtz[:, :, : dims.T],
                src=src_perm_x4,
            )

    # Cache quantized input for reuse across I-tiles (first I-tile only)
    if mx_buf.inp_qtz_cache_buf != None and inp_qtz_in == None:
        cache_buf = mx_buf.inp_qtz_cache_buf
        if T_padded > dims.T:
            nisa.memset(dst=cache_buf, value=0)
        nisa.tensor_copy(dst=cache_buf[:, :, : dims.T], src=inp_qtz[:, :, : dims.T])

    # Cache ROW_MX dequant scale for reuse across I-tiles (first I-tile only)
    if mx_buf.inp_dequant_scale_cache_buf != None and inp_qtz_in == None:
        cache_buf = mx_buf.inp_dequant_scale_cache_buf
        if cache_buf.shape[1] > T_padded:
            nisa.memset(dst=cache_buf, value=0.0)
            nisa.tensor_copy(dst=cache_buf[:, :T_padded, :], src=input_dequant_scale)
        else:
            nisa.tensor_copy(dst=cache_buf, src=input_dequant_scale)

    # Create ProjConfig
    proj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=T_padded,
        n_prgs=dims.num_shards,
        prg_id=dims.shard_id,
        name_prefix=name_prefix,
    )

    # Section 3: Gate Projection with MXFP
    gate_bias_sb = load_gate_up_bias_mx(params.bias_params.gate_proj_bias_tensor, dims.I, _pmax, n_I512_tile, _q_width)

    gate_w_dequant_sb = mx_buf.gate_w_dequant_sb

    gate_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=inp_qtz,
        hidden_scale_sb=dummy_scale_tile,
        weight_qtz=params.gate_proj_weights_tensor,
        weight_scale=dummy_scale_tile,
        bias_sb=gate_bias_sb if gate_bias_sb != None else None,
        cfg=proj_cfg,
        w_dequant_scale=gate_w_dequant_sb,
        input_dequant_scale=input_dequant_scale,
        activation_op=activation_op,
        pre_quantized=True,
    )

    # Apply activation function to gate output (skip if fused via activation_op)
    if activation_op == None:
        nisa.activation(
            dst=gate_out_sb,
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=gate_out_sb,
        )

    # Section 4: Up Projection with MXFP
    up_bias_sb = load_gate_up_bias_mx(params.bias_params.up_proj_bias_tensor, dims.I, _pmax, n_I512_tile, _q_width)

    up_w_dequant_sb = mx_buf.up_w_dequant_sb

    up_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=inp_qtz,
        hidden_scale_sb=dummy_scale_tile,
        weight_qtz=params.up_proj_weights_tensor,
        weight_scale=dummy_scale_tile,
        bias_sb=up_bias_sb if up_bias_sb != None else None,
        cfg=proj_cfg,
        w_dequant_scale=up_w_dequant_sb,
        input_dequant_scale=input_dequant_scale,
        pre_quantized=True,
    )

    # Section 5: Element-wise Multiply
    intermediate_sb = gate_out_sb
    nisa.tensor_tensor(
        dst=intermediate_sb,
        data1=gate_out_sb,
        data2=up_out_sb,
        op=nl.multiply,
    )

    # Section 6: Down Projection with MXFP
    down_weights = params.down_proj_weights_tensor
    down_bias = params.bias_params.down_proj_bias_tensor

    down_bias_sb = load_down_bias_mx(down_bias, dims.num_shards, dims.H1_shard, dims.H0, dims.shard_id)

    # ROW_MX: row_quantization on intermediate
    inter_4d = intermediate_sb.reshape((_pmax, n_I512_tile, T_padded, _q_width))

    # Permute [_pmax, n_I512, T_padded, _q_width] → [_pmax, T_padded, n_I512, _q_width]
    inter_permuted = nl.ndarray(
        (_pmax, T_padded, n_I512_tile, _q_width),
        dtype=inter_4d.dtype,
        buffer=nl.sbuf,
    )
    src_perm = inter_4d.permute(dims=[0, 2, 1, 3])
    nisa.tensor_copy(dst=inter_permuted, src=src_perm)

    # Reshape to rank-3 [_pmax, T_padded, n_I512*_q_width] for row_quantization
    inter_3d = inter_permuted.reshape((_pmax, T_padded, n_I512_tile * _q_width))
    quantized_3d, inter_dequant_scale = row_quantization(
        inter_3d,
        output_dtype=nl.float8_e4m3fn,
    )

    # Reshape to [_pmax, T_padded, n_I512, _q_width] fp8, reinterpret_cast to fp8_x4
    quantized_4d = quantized_3d.reshape((_pmax, T_padded, n_I512_tile, _q_width))
    quantized_x4 = quantized_4d.view(nl.float8_e4m3fn_x4)

    # Permute [_pmax, T_padded, n_I512] → [_pmax, n_I512, T_padded] fp8_x4
    inter_qtz = nl.ndarray(
        (_pmax, n_I512_tile, T_padded),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
    )
    src_perm_back = (quantized_x4.reshape((_pmax, T_padded, n_I512_tile))).permute(dims=[0, 2, 1])
    nisa.tensor_copy(dst=inter_qtz, src=src_perm_back)

    down_w_dequant_sb = mx_buf.down_w_dequant_sb

    down_out_sb = down_projection_mx_tp_shard_H(
        inter_sb=inter_qtz,
        weight=down_weights,
        weight_scale=dummy_scale_tile,
        bias_sb=down_bias_sb,
        cfg=proj_cfg,
        sbm=sbm,
        partial_output=not params.store_output_in_sbuf,
        pre_quantized=True,
        pre_quantized_scale=dummy_scale_tile,
        w_dequant_scale=down_w_dequant_sb,
        input_dequant_scale=inter_dequant_scale,
    )

    # Section 7: Output Transpose and Storage
    if skip_output_store:
        mx_buf.down_out_sb = down_out_sb
        return [output_tensor_hbm]

    if not params.store_output_in_sbuf:
        B, S, H = output_tensor_hbm.shape
        output_tensor_hbm = output_tensor_hbm.flatten_dims(start_dim=0, end_dim=1)

        output_hbm_view = output_tensor_hbm.slice(
            dim=1, start=dims.shard_id * dims.H_per_shard, end=(dims.shard_id + 1) * dims.H_per_shard
        )

        down_out_view = down_out_sb.slice(dim=2, start=0, end=dims.T)
        output_sb = nl.ndarray(
            (dims.T, dims.H_per_shard),
            dtype=output_tensor_hbm.dtype,
            buffer=nl.sbuf,
            name=f"{name_prefix}tkg_mlp_output_sb",
        )

        for h1_tile_idx in range(dims.H1_shard):
            psum_idx = h1_tile_idx % dims._psum_bmax
            tp_psum = nl.ndarray(
                (dims.T, dims.H0),
                dtype=output_tensor_hbm.dtype,
                buffer=nl.psum,
                name=f"{name_prefix}transpose_output_{h1_tile_idx}",
            )
            nisa.nc_transpose(dst=tp_psum, data=down_out_view.select(dim=1, index=h1_tile_idx))
            interleave_copy(
                dst=output_sb.slice(dim=1, start=h1_tile_idx * dims.H0, end=(h1_tile_idx + 1) * dims.H0),
                src=tp_psum,
                index=h1_tile_idx,
            )

        nisa.dma_copy(
            dst=output_hbm_view,
            src=output_sb,
        )

        output_tensor_hbm = output_tensor_hbm.reshape((B, S, H))

        return (
            [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
        )

    else:
        return [down_out_sb, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [down_out_sb]


def mlp_tkg_quad_fp8_row(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
) -> list[nl.NkiTensor]:
    """
    ROW_MX (row-wise / dynamic FP8) MLP TKG wrapper that tiles along both BxS and I.

    Splits the ``B * S`` token axis into ``BS_TILE_SIZE`` chunks and the intermediate
    ``I`` axis into ``_I_TILE_SIZE`` chunks. For each I-tile, gate/up/down weight slices
    are DMAed into SBUF once. On the first I-tile, the per-token dequant scale and the
    quantized input are both cached in SBUF so that subsequent I-tiles can reuse them
    without re-quantization. Partial I-tile results are accumulated into the output (in
    SBUF when ``store_output_in_sbuf=True``, otherwise via HBM read-modify-write).

    Args:
        params (MLPParameters): MLP configuration. Must report ROW_MX quantization.
        output_tensor_hbm (nl.NkiTensor): [B, S, H], Output tensor in HBM.
        output_stored_add_tensor_hbm (nl.NkiTensor): Optional fused-add output in HBM.

    Returns:
        list[nl.NkiTensor]:
            - [output_tensor_hbm] when ``store_fused_add_result`` is False.
            - [output_tensor_hbm, output_stored_add_tensor_hbm] when fused-add storage
              is enabled.
            - [output_sb_accum] (SBUF tensor) when ``store_output_in_sbuf=True``.

    Notes:
        - ``store_output_in_sbuf`` and ``input_in_sbuf`` require ``B * S <= BS_TILE_SIZE``.
        - I-tile size is ``_I_TILE_SIZE`` (default 1536).
        - Row-wise dequant scale (``[_pmax, T_padded, 1]``) is cached alongside the
          quantized input (``[_pmax, n_H512_tile_sharded, T_padded]``) per T-tile.
        - This function is an undecorated sub-kernel dispatched from ``mlp_tkg_mx``.
    """

    T = params.batch_size * params.sequence_len
    H = params.hidden_size
    tile_size = min(BS_TILE_SIZE, T)

    if params.store_output_in_sbuf or params.input_in_sbuf:
        kernel_assert(T <= tile_size, "store_output_in_sbuf/input_in_sbuf requires BxS <= BS_TILE_SIZE")

    if not params.input_in_sbuf:
        hidden = params.hidden_tensor.reshape((T, H))

    B, S, H_out = output_tensor_hbm.shape
    if not params.store_output_in_sbuf:
        output_hbm_view = output_tensor_hbm.reshape((B * S, H_out))

    I = params.intermediate_size

    # ROW_MX: I-tiling path
    _, lnc, shard_id = get_verified_program_sharding_info("mlp_tkg", (0, 1))
    num_shards = lnc if not params.shard_on_h_disabled else 1
    n_I_tiles = div_ceil(I, _I_TILE_SIZE)

    # Hardware-constant shorthands; match projection_mx_constants / mlp_parameters
    _pmax = 128
    _q_width = 4
    H_sharded = H // num_shards
    H1_shard = H_sharded // _pmax
    n_H512_tile_sharded = H_sharded // (_pmax * _q_width)
    T_padded = div_ceil(tile_size, _q_width) * _q_width

    sbm = SbufManager(0, _RMSNORM_SBUF_BUDGET_BYTES, get_logger("mlp_tkg_row_mx"))
    sbm.set_name_prefix("mlp_")
    sbm.open_scope()

    # Save original HBM references for I-tile slicing
    gate_w_hbm = params.gate_proj_weights_tensor
    up_w_hbm = params.up_proj_weights_tensor
    down_w_hbm = params.down_proj_weights_tensor

    # Convert 6D scalar fp8 gate/up weights to flat 3D scalar fp8.
    # [128, n_H512, n_I512, 4, 128, 4] → [128, n_H512, I_padded*4] scalar fp8
    if len(gate_w_hbm.shape) == 6:
        n_H512_w = gate_w_hbm.shape[1]
        I_padded = gate_w_hbm.shape[2] * gate_w_hbm.shape[3] * gate_w_hbm.shape[4]
        gate_w_hbm = gate_w_hbm.reshape((_pmax, n_H512_w, I_padded * _q_width))
        up_w_hbm = up_w_hbm.reshape((_pmax, n_H512_w, I_padded * _q_width))

    # Convert 4D scalar fp8 down weights to flat 3D scalar fp8.
    # [128_I, n_I512, H, 4] → [128_I, n_I512, H*4] scalar fp8
    if len(down_w_hbm.shape) == 4:
        p_I_d = down_w_hbm.shape[0]
        n_I512_d = down_w_hbm.shape[1]
        down_w_hbm = down_w_hbm.reshape((p_I_d, n_I512_d, H * _q_width))

    gate_bias_hbm = params.bias_params.gate_proj_bias_tensor
    up_bias_hbm = params.bias_params.up_proj_bias_tensor
    original_I = I

    # Cache quantized input and per-token dequant scale per T-tile
    n_T_tiles = div_ceil(T, tile_size)
    inp_qtz_cache = []
    inp_dequant_scale_cache = []
    for t_tile_idx in range(n_T_tiles):
        inp_qtz_cache.append(
            sbm.alloc_stack((_pmax, n_H512_tile_sharded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, align=32)
        )
        inp_dequant_scale_cache.append(sbm.alloc_stack((_pmax, T_padded, 1), dtype=nl.float32, buffer=nl.sbuf))

    # Pre-allocate persistent MX buffers
    mx_buf = RowMxPreallocBuffers()

    mx_buf.dummy_scale_tile = alloc_dummy_scale_tile(sbm, _pmax)

    mx_buf.gate_w_dequant_sb = sbm.alloc_stack(params.quant_params.gate_w_scale.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mx_buf.gate_w_dequant_sb, src=params.quant_params.gate_w_scale)
    mx_buf.up_w_dequant_sb = sbm.alloc_stack(params.quant_params.up_w_scale.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mx_buf.up_w_dequant_sb, src=params.quant_params.up_w_scale)
    mx_buf.down_w_dequant_sb = sbm.alloc_stack(params.quant_params.down_w_scale.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mx_buf.down_w_dequant_sb, src=params.quant_params.down_w_scale)

    # Persistent SBUF accumulator for store_output_in_sbuf
    if params.store_output_in_sbuf:
        output_sb_accum = sbm.alloc_stack((tile_size, H_sharded), dtype=nl.bfloat16, buffer=nl.sbuf, align=32)

    # Save full dequant scale tensors for per-I-tile slicing
    gate_w_dequant_full = mx_buf.gate_w_dequant_sb
    up_w_dequant_full = mx_buf.up_w_dequant_sb

    for i_tile in TiledRange(I, _I_TILE_SIZE):
        sbm.open_scope()

        i_tile_n_I512 = div_ceil(i_tile.size, _pmax * _q_width)

        # Slice gate/up weight dequant scales to current I-tile's column range
        if gate_w_dequant_full.shape[1] > 1:
            col_start = (i_tile.start_offset // (_pmax * _q_width)) * _q_width
            col_end = col_start + i_tile_n_I512 * _q_width
            mx_buf.gate_w_dequant_sb = gate_w_dequant_full[:, col_start:col_end]
            mx_buf.up_w_dequant_sb = up_w_dequant_full[:, col_start:col_end]

        # Load gate/up/down weight slices for this I-tile
        gate_w_sb = load_gate_up_weight(sbm, gate_w_hbm, i_tile, n_H512_tile_sharded, num_shards, shard_id, _pmax)
        up_w_sb = load_gate_up_weight(sbm, up_w_hbm, i_tile, n_H512_tile_sharded, num_shards, shard_id, _pmax)
        down_w_sb = load_down_weight(
            sbm, down_w_hbm, i_tile, H_sharded, num_shards, shard_id, original_I, _pmax, _q_width
        )
        params.gate_proj_weights_tensor = gate_w_sb
        params.up_proj_weights_tensor = up_w_sb
        params.down_proj_weights_tensor = down_w_sb
        params.intermediate_size = i_tile.size

        # Slice gate/up biases to current I-tile's I512 range
        i512_start = i_tile.start_offset // (_pmax * _q_width)
        if params.bias_params.gate_proj_bias_tensor != None and n_I_tiles > 1:
            params.bias_params.gate_proj_bias_tensor = gate_bias_hbm[:, i512_start : i512_start + i_tile_n_I512, :]
        if params.bias_params.up_proj_bias_tensor != None and n_I_tiles > 1:
            params.bias_params.up_proj_bias_tensor = up_bias_hbm[:, i512_start : i512_start + i_tile_n_I512, :]

        # Inner loop: T-tiles
        for bxs_tile in TiledRange(T, tile_size):
            sbm.open_scope()

            params.batch_size = 1
            params.sequence_len = bxs_tile.size
            if not params.input_in_sbuf:
                params.hidden_tensor = hidden[bxs_tile.start_offset : bxs_tile.end_offset, :].reshape(
                    (1, bxs_tile.size, H)
                )

            if i_tile.index == 0:
                inp_qtz_in = None
                skip_norm = False
                mx_buf.inp_qtz_cache_buf = inp_qtz_cache[bxs_tile.index] if inp_qtz_cache else None
                mx_buf.inp_dequant_scale_cache_buf = (
                    inp_dequant_scale_cache[bxs_tile.index] if inp_dequant_scale_cache else None
                )
            else:
                inp_qtz_in = inp_qtz_cache[bxs_tile.index] if inp_qtz_cache else None
                skip_norm = True if inp_qtz_cache else False
                mx_buf.inp_qtz_cache_buf = None
                mx_buf.inp_dequant_scale_cache_buf = (
                    inp_dequant_scale_cache[bxs_tile.index] if inp_dequant_scale_cache else None
                )

            _mlp_tkg_row_mx_impl(
                params,
                output_tensor_hbm,
                output_stored_add_tensor_hbm,
                name_prefix=f"i{i_tile.index}_t{bxs_tile.index}_",
                sbm=sbm,
                mx_buf=mx_buf,
                inp_qtz_in=inp_qtz_in,
                skip_norm=skip_norm,
                skip_output_store=True,
                activation_op=None,
            )

            # Transpose down_out_sb [H0, H1_shard, T] → [T, H_shard]
            down_out_sb = mx_buf.down_out_sb
            down_out_view = down_out_sb.slice(dim=2, start=0, end=bxs_tile.size)
            output_sb = sbm.alloc_stack((bxs_tile.size, H_sharded), dtype=nl.bfloat16, buffer=nl.sbuf, align=32)

            for h1_tile_idx in range(H1_shard):
                tp_psum = nl.ndarray((bxs_tile.size, _pmax), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp_psum, data=down_out_view.select(dim=1, index=h1_tile_idx))
                interleave_copy(
                    dst=output_sb.slice(dim=1, start=h1_tile_idx * _pmax, end=(h1_tile_idx + 1) * _pmax),
                    src=tp_psum,
                    index=h1_tile_idx,
                )

            # Accumulate partial I-tile results
            if params.store_output_in_sbuf:
                if i_tile.index == 0:
                    nisa.tensor_copy(dst=output_sb_accum, src=output_sb)
                else:
                    nisa.tensor_tensor(dst=output_sb_accum, data1=output_sb_accum, data2=output_sb, op=nl.add)
            else:
                out_hbm_tile = output_hbm_view.slice(dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset)
                out_hbm_shard = out_hbm_tile.slice(dim=1, start=shard_id * H_sharded, end=(shard_id + 1) * H_sharded)
                if i_tile.index == 0:
                    nisa.dma_copy(dst=out_hbm_shard, src=output_sb)
                else:
                    existing_sb = sbm.alloc_stack((bxs_tile.size, H_sharded), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=existing_sb, src=out_hbm_shard)
                    nisa.tensor_tensor(dst=output_sb, data1=output_sb, data2=existing_sb, op=nl.add)
                    nisa.dma_copy(dst=out_hbm_shard, src=output_sb)

            sbm.close_scope()
        sbm.close_scope()  # I-tile scope

    if params.store_output_in_sbuf:
        return [output_sb_accum, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_sb_accum]
    return [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
