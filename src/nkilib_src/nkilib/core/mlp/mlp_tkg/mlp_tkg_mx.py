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

"""MLP TKG kernel implementation for token generation scenarios with optional normalization"""

import nki.isa as nisa
import nki.language as nl

from ...subkernels.rmsnorm_tkg import rmsnorm_tkg
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
from ...utils.tensor_view import TensorView
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_normalization,
    mlpp_has_rms_normalization,
    mlpp_store_fused_add,
)
from .down_projection_mx_shard_H import down_projection_mx_tp_shard_H
from .gate_up_projection_mx_shard_H import gate_up_projection_mx_tp_shard_H
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_utils import _layout_adapter_hbm, _layout_adapter_sb
from .projection_mx_constants import ProjConfig


def mlp_tkg_mx(
    params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
) -> list[nl.ndarray]:
    """
    MLP TKG kernel with MXFP quantization support (MXFP4 and MXFP8).

    This implementation quantizes weights to MXFP4 or MXFP8 formats while
    maintaining accuracy within 5% relative tolerance. Supports H-dimension
    sharding across gate/up/down projections, MXFP row-wise quantization,
    and optional bias addition.

    Dimensions:
        H: Hidden dimension size (must be divisible by 512)
        I: Intermediate dimension size
        T: Sequence length (BxS, padded to multiple of 4)
        B: Batch size
        S: Sequence length per batch

    Args:
        params (MLPParameters): MLP configuration with MXFP quantized weights.
            - gate_proj_weights_tensor: [_pmax, n_H512_tile, I] in MXFP4/8
            - up_proj_weights_tensor: [_pmax, n_H512_tile, I] in MXFP4/8
            - down_proj_weights_tensor: [I_p, ceil(I/512), H] in MXFP4/8
            - quant_params.gate_w_scale: uint8 scale factors
            - quant_params.up_w_scale: uint8 scale factors
            - quant_params.down_w_scale: uint8 scale factors
        output_tensor_hbm (nl.ndarray): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.ndarray): Optional fused add output in HBM

    Returns:
        list[nl.ndarray]:
            - [output_tensor_hbm] when store_output_in_sbuf=False
            - [down_out_sb] when store_output_in_sbuf=True

    Notes:
        - Input activations are quantized online to MXFP8 format
        - Supports RMSNorm but not LayerNorm
        - Does not support fused_add or column tiling
        - H must be divisible by 512 for proper quantization alignment
        - T is padded to multiple of 4 for quantization requirements

    Pseudocode:
        # Optional normalization
        if has_normalization:
            hidden = rmsnorm(hidden)

        # Layout adaptation and quantization
        hidden_shuffled = layout_adapter(hidden)  # [_pmax, n_H512, T, 4]
        hidden_qtz, hidden_scale = quantize_mx(hidden_shuffled)  # MXFP8

        # Gate projection
        gate_out = matmul_mx(hidden_qtz, gate_weights_mx, hidden_scale, gate_scale)
        if gate_bias:
            gate_out += gate_bias
        gate_out = activation(gate_out)

        # Up projection
        up_out = matmul_mx(hidden_qtz, up_weights_mx, hidden_scale, up_scale)
        if up_bias:
            up_out += up_bias

        # Element-wise multiply
        intermediate = gate_out * up_out

        # Down projection
        output = matmul_mx(intermediate, down_weights_mx, down_scale)
        if down_bias:
            output += down_bias

        # Transpose and store
        output = transpose(output)  # [H0, H1, T] → [T, H]
    """
    io_dtype = params.hidden_tensor.dtype

    # Validate inputs
    kernel_assert(
        params.quant_params.is_quant_mx(),
        "mlp_tkg_mx requires MXFP quantization (QuantizationType.MX)",
    )
    kernel_assert(
        not mlpp_has_normalization(params) or mlpp_has_rms_normalization(params),
        "mlp_tkg_mx only supports RMSNorm or no normalization",
    )

    # Compute kernel dimensions
    dims = MLPTKGConstants.calculate_constants(params)

    # Get MX quantization constants from dims
    _pmax = dims._pmax  # Partition dimension (128)
    _q_width = dims._q_width  # Quantization tile width (4)
    _q_height = dims._q_height  # Quantization tile height (8)

    # ============================================================
    # Section 1: Normalization (Optional)
    # ============================================================
    hidden_tensor = params.hidden_tensor
    if mlpp_has_normalization(params):
        if mlpp_has_rms_normalization(params):
            rmsnorm_out = nl.ndarray((dims.H0, dims.T, dims.H1), dtype=io_dtype, buffer=nl.sbuf)
            norm_weights = params.norm_params.normalization_weights_tensor
            eps = params.eps
            rmsnorm_out = rmsnorm_tkg(
                input=params.hidden_tensor,
                gamma=norm_weights,
                output=rmsnorm_out,
                eps=eps,
                hidden_dim_tp=True,
                single_core_forced=True,
            )
            hidden_tensor = rmsnorm_out
        else:
            kernel_assert(False, "mlp_tkg_mx only supports RMSNorm, LayerNorm is not supported")

    # ============================================================
    # Section 2: Layout Adaptation and Input Quantization
    # ============================================================
    """
    Convert input to quantizable layout and quantize to mxfp8.
    Calculate tiling dimensions for H and I.
    """
    n_H512_tile_sharded = dims.H_per_shard // (_pmax * _q_width)  # Number of 512-element tiles in H dimension
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))  # Number of 512-element tiles in I dimension
    T_padded = div_ceil(dims.T, 4) * 4  # Pad T to multiple of 4 for quantization

    # Use layout adapter to get quantizable layout for Gate/Up projection
    # Output is always bf16[_pmax, n_H512_tile_sharded, T_padded, _q_width] @ SBUF
    input_sb_shfl = None

    if params.input_in_sbuf or mlpp_has_rms_normalization(params):
        # Input already in SBUF, use SBUF layout adapter
        input_sb_shfl = _layout_adapter_sb(hidden_tensor, n_prgs=dims.num_shards, prg_id=dims.shard_id)
    else:
        # Input in HBM, use HBM layout adapter (includes DMA load)
        hidden_tensor = hidden_tensor.reshape((dims.T, dims.H))
        input_sb_shfl = _layout_adapter_hbm(hidden_tensor, n_prgs=dims.num_shards, prg_id=dims.shard_id)

    """
    Allocate quantized tensors for mxfp8 format.
    """
    inp_qtz = nl.ndarray(
        (_pmax, n_H512_tile_sharded * T_padded),
        dtype=nl.float8_e4m3fn_x4,  # MXFP8 format (4 elements packed)
        buffer=nl.sbuf,
        name="input_quantized",
    )
    inp_scale = nl.ndarray(
        inp_qtz.shape,
        dtype=nl.uint8,  # Scale factors for each quantization tile
        buffer=nl.sbuf,
        name="input_scale",
    )

    # Quantize input from bf16 to mxfp8
    input_flat = input_sb_shfl.reshape((_pmax, n_H512_tile_sharded * T_padded * _q_width))
    nisa.quantize_mx(dst=inp_qtz, src=input_flat, dst_scale=inp_scale)

    # Reshape to tiled format for matmul operations
    inp_qtz = inp_qtz.reshape((_pmax, n_H512_tile_sharded, T_padded))
    inp_scale = inp_scale.reshape(inp_qtz.shape)

    # ---------------- Create ProjConfig ----------------
    # Configuration object for projection operations with H-dimension sharding
    proj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=T_padded,
        n_prgs=dims.num_shards,
        prg_id=dims.shard_id,
    )

    # ============================================================
    # Section 3: Gate Projection with MXFP
    # ============================================================
    # Compute: gate_out = hidden @ gate_weight + gate_bias
    # Allocate and load gate bias in SBUF if present
    gate_bias_sb = None
    if params.bias_params.gate_proj_bias_tensor is not None:
        # Gate bias format in HBM: bf16[I_p, ceil(I/512), 4] where I_p = I//4 if I <= 512 else _pmax
        # We need to load and reshape to [_pmax, n_I512_tile, _q_width] for projection
        gate_bias_sb = nl.ndarray((_pmax, n_I512_tile, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)

        if dims.I < 512:
            # When I<512, bias HBM is not padded, so pad it in SBUF
            nisa.memset(dst=gate_bias_sb[:, 0, :], value=0.0)
            nisa.dma_copy(
                dst=gate_bias_sb[: dims.I // 4, :, :],
                src=params.bias_params.gate_proj_bias_tensor,
            )
        else:
            # I >= 512, bias is already padded in HBM
            nisa.dma_copy(
                dst=gate_bias_sb,
                src=params.bias_params.gate_proj_bias_tensor,
            )

    # Perform gate projection using MXFP quantized weights (MXFP4 or MXFP8)
    gate_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=TensorView(inp_qtz),
        hidden_scale_sb=TensorView(inp_scale),
        weight_qtz=TensorView(params.gate_proj_weights_tensor),
        weight_scale=TensorView(params.quant_params.gate_w_scale),
        bias_sb=TensorView(gate_bias_sb) if gate_bias_sb is not None else None,
        cfg=proj_cfg,
    )

    # Apply activation function to gate output
    nisa.activation(
        dst=gate_out_sb,
        op=get_nl_act_fn_from_type(params.activation_fn),
        data=gate_out_sb,
    )

    # ============================================================
    # Section 4: Up Projection with MXFP
    # ============================================================
    # Compute: up_out = hidden @ up_weight + up_bias
    # Allocate and load up bias in SBUF if present
    up_bias_sb = None
    if params.bias_params.up_proj_bias_tensor is not None:
        # Up bias format in HBM: bf16[I_p, ceil(I/512), 4] where I_p = I//4 if I <= 512 else _pmax
        # We need to load and reshape to [_pmax, n_I512_tile, _q_width] for projection
        up_bias_sb = nl.ndarray((_pmax, n_I512_tile, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)

        if dims.I < 512:
            # When I<512, bias HBM is not padded, so pad it in SBUF
            nisa.memset(dst=up_bias_sb[:, 0, :], value=0.0)
            nisa.dma_copy(
                dst=up_bias_sb[: dims.I // 4, :, :],
                src=params.bias_params.up_proj_bias_tensor,
            )
        else:
            # I >= 512, bias is already padded in HBM
            nisa.dma_copy(
                dst=up_bias_sb,
                src=params.bias_params.up_proj_bias_tensor,
            )

    # Perform up projection using MXFP quantized weights (MXFP4 or MXFP8)
    up_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=TensorView(inp_qtz),
        hidden_scale_sb=TensorView(inp_scale),
        weight_qtz=TensorView(params.up_proj_weights_tensor),
        weight_scale=TensorView(params.quant_params.up_w_scale),
        bias_sb=TensorView(up_bias_sb) if up_bias_sb is not None else None,
        cfg=proj_cfg,
    )

    # ============================================================
    # Section 5: Element-wise Multiply
    # ============================================================
    # Compute: intermediate = activation(gate_out) * up_out
    # Reuse gate_out_sb buffer for intermediate result
    intermediate_sb = gate_out_sb
    nisa.tensor_tensor(
        dst=intermediate_sb,
        data1=gate_out_sb,
        data2=up_out_sb,
        op=nl.multiply,
    )

    # ============================================================
    # Section 6: Down Projection with MXFP
    # ============================================================
    # Compute: output = intermediate @ down_weight + down_bias
    down_weights = params.down_proj_weights_tensor
    down_scale = params.quant_params.down_w_scale
    down_bias = params.bias_params.down_proj_bias_tensor

    # Allocate and load down projection bias in SBUF if present
    down_bias_sb = None
    if down_bias is not None:
        # Reshape to separate shards: [1, H] -> [dims.num_shards, H1_shard, H0]
        # This works because H = dims.num_shards * H1_shard * H0
        down_bias = down_bias.reshape((dims.num_shards, dims.H1_shard, dims.H0))

        # Select this shard's bias portion using TensorView
        sharded_down_bias_hbm_view = TensorView(down_bias).select(dim=0, index=dims.shard_id)

        # Allocate SBUF buffer for bias with correct layout [H0, H1_shard]
        down_bias_sb = nl.ndarray((dims.H0, dims.H1_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        down_bias_sb_view = TensorView(down_bias_sb)

        # dma transpose requirement : 4D AP
        while sharded_down_bias_hbm_view.get_dim() < 4:
            sharded_down_bias_hbm_view = sharded_down_bias_hbm_view.expand_dim(1)
        while down_bias_sb_view.get_dim() < 4:
            down_bias_sb_view = down_bias_sb_view.expand_dim(1)

        # DMA copy the selected shard's bias to SBUF
        # Result is [H0, H1_shard] ready for use in down projection
        nisa.dma_transpose(
            dst=down_bias_sb_view.get_view(),
            src=sharded_down_bias_hbm_view.get_view(),
        )

    # Perform down projection using MXFP quantized weights (MXFP4 or MXFP8)
    # partial_output=True skips LNC sync when output stays in SBUF (for debugging/inspection)
    down_out_sb = down_projection_mx_tp_shard_H(
        inter_sb=intermediate_sb,
        weight=down_weights,
        weight_scale=down_scale,
        bias_sb=down_bias_sb,
        cfg=proj_cfg,
        partial_output=not params.store_output_in_sbuf,  # Skip sync if keeping output in SBUF
    )

    # ============================================================
    # Section 7: Output Transpose and Storage
    # ============================================================
    if not params.store_output_in_sbuf:
        # Store output to HBM with transpose from [H0, H1, T] to [T, H]

        # Reshape to 2D tensor for easier indexing
        B, S, H = output_tensor_hbm.shape
        output_tensor_hbm = output_tensor_hbm.reshape((B * S, H))

        # Create view for this shard's portion of H dimension
        output_hbm_view = TensorView(output_tensor_hbm).slice(
            dim=1, start=dims.shard_id * dims.H_per_shard, end=(dims.shard_id + 1) * dims.H_per_shard
        )

        # Transpose output from [H0, H1, T] to [T, H] layout
        down_out_view = TensorView(down_out_sb).slice(dim=2, start=0, end=dims.T)
        output_sb = nl.ndarray(
            (dims.T, dims.H_per_shard),
            dtype=output_tensor_hbm.dtype,
            buffer=nl.sbuf,
            name="tkg_mlp_output_sb",
        )
        output_sb_view = TensorView(output_sb)

        # Transpose each H1 tile using nc_transpose and interleave_copy
        for h1_tile_idx in range(dims.H1_shard):
            psum_idx = h1_tile_idx % dims._psum_bmax
            tp_psum = nl.ndarray(
                (dims.T, dims.H0),
                dtype=output_tensor_hbm.dtype,
                buffer=nl.psum,
                name=f"transpose_output_{h1_tile_idx}",
            )
            # Transpose [H0, T] to [T, H0]
            nisa.nc_transpose(dst=tp_psum, data=down_out_view.select(dim=1, index=h1_tile_idx).get_view())
            # Copy transposed tile to output buffer with interleaving
            interleave_copy(
                dst=output_sb_view.slice(
                    dim=1, start=h1_tile_idx * dims.H0, end=(h1_tile_idx + 1) * dims.H0
                ).get_view(),
                src=tp_psum,
                index=h1_tile_idx,
            )

        # DMA copy transposed output to HBM
        nisa.dma_copy(
            dst=output_hbm_view.get_view(),
            src=output_sb_view.get_view(),
        )

        # Reshape back to 3D tensor
        output_tensor_hbm = output_tensor_hbm.reshape((B, S, H))

        return (
            [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
        )

    else:
        # Keep output in SBUF (for debugging or when caller will handle HBM storage)
        return [down_out_sb, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [down_out_sb]
