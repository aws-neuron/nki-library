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

"""MLP TKG kernel implementation for hardware MX (MXFP) quantization."""

import nki.isa as nisa
import nki.language as nl

from ...subkernels.rmsnorm_tkg import rmsnorm_tkg
from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
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
    load_down_bias_mx,
    load_gate_up_bias_mx,
)
from .mlp_tkg_utils import _layout_adapter_hbm, _layout_adapter_sb
from .projection_mx_constants import ProjConfig

# SBUF budget reserved for RMSNorm in the hw-MX implementation (200 KiB).
_RMSNORM_SBUF_BUDGET_BYTES = 200 * 1024


def _mlp_tkg_hw_mx_impl(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
    name_prefix: str = "",
) -> list[nl.NkiTensor]:
    """
    MLP TKG kernel with hardware MX (MXFP) quantization.

    Uses ``nisa.quantize_mx`` for hardware-accelerated MXFP quantization
    with real MX block-level scale factors (uint8).

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
        params.quant_params.is_quant_mx(),
        "mlp_tkg_quad_fp8_mx requires MX quantization",
    )
    kernel_assert(
        not mlpp_has_normalization(params) or mlpp_has_rms_normalization(params),
        "mlp_tkg_quad_fp8_mx only supports RMSNorm or no normalization",
    )

    # Compute kernel dimensions
    dims = MLPTKGConstants.calculate_constants(params)

    # MX quantization constants from dims (_pmax=128, _q_width=4)
    _pmax = dims._pmax
    _q_width = dims._q_width

    # Section 1: Normalization (Optional)
    hidden_tensor = params.hidden_tensor
    if mlpp_has_normalization(params):
        if mlpp_has_rms_normalization(params):
            rmsnorm_out = nl.ndarray((dims.H0, dims.T, dims.H1), dtype=io_dtype, buffer=nl.sbuf)
            norm_weights = params.norm_params.normalization_weights_tensor
            eps = params.eps
            rmsnorm_sbm = SbufManager(
                sb_lower_bound=0,
                sb_upper_bound=_RMSNORM_SBUF_BUDGET_BYTES,
                logger=get_logger("mlp_tkg_hw_mx_rmsnorm"),
                use_auto_alloc=True,
            )
            rmsnorm_sbm.set_name_prefix(name_prefix)
            rmsnorm_out = rmsnorm_tkg(
                input=params.hidden_tensor,
                gamma=norm_weights,
                output=rmsnorm_out,
                eps=eps,
                hidden_dim_tp=True,
                single_core_forced=True,
                sbm=rmsnorm_sbm,
            )
            hidden_tensor = rmsnorm_out
        else:
            kernel_assert(False, "mlp_tkg_quad_fp8_mx only supports RMSNorm, LayerNorm is not supported")

    # Section 2: Input Quantization (Hardware MX)
    n_H512_tile_sharded = dims.H_per_shard // (_pmax * _q_width)
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))
    T_padded = div_ceil(dims.T, _q_width) * _q_width

    # ── MX path: existing hardware quantization ──
    input_sb_shfl = None

    if params.input_in_sbuf or mlpp_has_rms_normalization(params):
        input_sb_shfl = _layout_adapter_sb(hidden_tensor, n_prgs=dims.num_shards, prg_id=dims.shard_id)
    else:
        hidden_tensor = hidden_tensor.reshape((dims.T, dims.H))
        input_sb_shfl = _layout_adapter_hbm(hidden_tensor, n_prgs=dims.num_shards, prg_id=dims.shard_id)

    # Allocate quantized tensors for mxfp8 format
    inp_qtz = nl.ndarray(
        (_pmax, n_H512_tile_sharded * T_padded),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name=f"{name_prefix}input_quantized",
    )
    inp_scale = nl.ndarray(
        inp_qtz.shape,
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name_prefix}input_scale",
    )

    # Quantize input from bf16 to mxfp8
    input_flat = input_sb_shfl.reshape((_pmax, n_H512_tile_sharded * T_padded * _q_width))
    nisa.quantize_mx(dst=inp_qtz, src=input_flat, dst_scale=inp_scale)

    # Reshape to tiled format for matmul operations
    inp_qtz = inp_qtz.reshape((_pmax, n_H512_tile_sharded, T_padded))
    inp_scale = inp_scale.reshape(inp_qtz.shape)

    # ---------------- Create ProjConfig ----------------
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

    gate_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=inp_qtz,
        hidden_scale_sb=inp_scale,
        weight_qtz=params.gate_proj_weights_tensor,
        weight_scale=params.quant_params.gate_w_scale,
        bias_sb=gate_bias_sb if gate_bias_sb != None else None,
        cfg=proj_cfg,
    )

    # MX path: activation only (no dequant needed, real MX scales used in matmul)
    nisa.activation(
        dst=gate_out_sb,
        op=get_nl_act_fn_from_type(params.activation_fn),
        data=gate_out_sb,
    )

    # Section 4: Up Projection with MXFP
    up_bias_sb = load_gate_up_bias_mx(params.bias_params.up_proj_bias_tensor, dims.I, _pmax, n_I512_tile, _q_width)

    up_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=inp_qtz,
        hidden_scale_sb=inp_scale,
        weight_qtz=params.up_proj_weights_tensor,
        weight_scale=params.quant_params.up_w_scale,
        bias_sb=up_bias_sb if up_bias_sb != None else None,
        cfg=proj_cfg,
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
    down_scale = params.quant_params.down_w_scale
    down_bias = params.bias_params.down_proj_bias_tensor

    down_bias_sb = load_down_bias_mx(down_bias, dims.num_shards, dims.H1_shard, dims.H0, dims.shard_id)

    down_out_sb = down_projection_mx_tp_shard_H(
        inter_sb=intermediate_sb,
        weight=down_weights,
        weight_scale=down_scale,
        bias_sb=down_bias_sb,
        cfg=proj_cfg,
        sbm=None,
        partial_output=not params.store_output_in_sbuf,
    )

    # Section 7: Output Transpose and Storage
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


def mlp_tkg_quad_fp8_mx(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
) -> list[nl.NkiTensor]:
    """
    Hardware-MX MLP TKG wrapper that tiles along the BxS dimension.

    Splits the ``B * S`` token axis into ``BS_TILE_SIZE`` chunks and invokes
    ``_mlp_tkg_hw_mx_impl`` for each tile. No I-tiling is required because the
    hardware ``nisa.quantize_mx`` path processes the full intermediate width per tile.

    Args:
        params (MLPParameters): MLP configuration. Must report MX quantization.
        output_tensor_hbm (nl.NkiTensor): [B, S, H], Output tensor in HBM.
        output_stored_add_tensor_hbm (nl.NkiTensor): Optional fused-add output in HBM.

    Returns:
        list[nl.NkiTensor]:
            - [output_tensor_hbm] when ``store_fused_add_result`` is False.
            - [output_tensor_hbm, output_stored_add_tensor_hbm] when fused-add storage
              is enabled.

    Notes:
        - ``store_output_in_sbuf`` and ``input_in_sbuf`` require ``B * S <= BS_TILE_SIZE``.
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

    # Hardware MX: no I-tiling needed, T-tile only
    for bxs_tile in TiledRange(T, tile_size):
        params.batch_size = 1
        params.sequence_len = bxs_tile.size
        if not params.input_in_sbuf:
            params.hidden_tensor = hidden[bxs_tile.start_offset : bxs_tile.end_offset, :].reshape((1, bxs_tile.size, H))
        output_tile = output_hbm_view.slice(dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset).expand_dim(
            dim=0
        )
        _mlp_tkg_hw_mx_impl(
            params,
            output_tile,
            output_stored_add_tensor_hbm,
            name_prefix=f"bxs_{bxs_tile.index}_",
        )
    return [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
