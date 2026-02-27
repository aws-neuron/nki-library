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

"""All-expert MoE token generation implementation that processes all experts sequentially."""

import nki
import nki.isa as nisa
import nki.language as nl
import nki.tensor as ntensor

from ...mlp.mlp_parameters import (
    MLPBiasParameters,
    MLPParameters,
    MLPQuantizationParameters,
)

# MLP utils
from ...mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstants
from ...mlp.mlp_tkg.mlp_tkg_down_projection import process_down_projection
from ...mlp.mlp_tkg.mlp_tkg_gate_up_projection import process_gate_up_projection
from ...mlp.mlp_tkg.mlp_tkg_utils import input_norm_load, transpose_store
from ...utils.allocator import SbufManager

# common utils
from ...utils.common_types import ExpertAffinityScaleMode, GateUpDim, QuantizationType
from ...utils.kernel_assert import kernel_assert
from ...utils.logging import get_logger
from ...utils.tensor_view import TensorView
from .moe_tkg_utils import reshape_scale_for_mlp, safe_tensor_view


def _all_expert_moe_tkg(
    params: MLPParameters,
    output: nl.ndarray,
) -> nl.ndarray:
    """
    All-expert Mixture of Experts (MoE) kernel for token generation (TKG).

    Processes all experts sequentially, computing MLP projections for each expert
    and accumulating results weighted by expert affinities.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        output (nl.ndarray): Output tensor to store the final result.

    Returns:
        output (nl.ndarray): Output tensor with accumulated expert results.

    Notes:
        - Column tiling is not supported; only swapped LHS/RHS projection is used.

    Pseudocode:
        input_sb[H0, T, H1] = normalize(hidden_tensor[T, H])
        output_temp[H0, H1_shard, T] = zeros()

        for expert_idx in range(E):
            gate_w[I, H], up_w[I, H], down_w[H, I] = weights[expert_idx]

            # Gate-Up projection: act_fn(gate(x)) * up(x)
            gate_up[I0, I1, T] = gate_up_proj(input_sb[H0, T, H1], gate_w, up_w)

            # Down projection
            down[H0, H1_shard, T] = down_proj(gate_up[I0, I1, T], down_w)

            # Scale by affinity and accumulate
            if affinity_scaling_mode == POST_SCALE:
                affinity[H0, T] = broadcast(expert_affinities[T, expert_idx])
                down[H0, H1_shard, T] *= affinity[H0, T]

            if expert_idx == 0:
                output_temp[H0, H1_shard, T] = down[H0, H1_shard, T]
            else:
                output_temp[H0, H1_shard, T] += down[H0, H1_shard, T]

        output[T, H] = transpose(output_temp[H0, H1_shard, T])
    """
    io_dtype = params.hidden_tensor.dtype
    expert_affinities = params.expert_params.expert_affinities
    expert_affinities_in_sbuf = expert_affinities.buffer == nl.sbuf

    if params.use_tkg_gate_up_proj_column_tiling or params.use_tkg_down_proj_column_tiling:
        kernel_assert(False, "Column tiling is not supported in all-expert MLP kernel, only swapped LHS/RHS projection")

    # Check if input is already in SBUF
    hidden_in_sbuf = params.hidden_tensor.buffer == nl.sbuf

    dims = MLPTKGConstants.calculate_constants(params)

    H = params.hidden_tensor.shape[-1]
    """
    Calibrate weight tile calculations and remove auto allocation workaround.
    
    Note that now it is always use the auto allocation since expert_affinities is always in SBUF.
    """
    use_auto_alloc = H >= 16 * 1024 or hidden_in_sbuf or expert_affinities_in_sbuf
    sbm = SbufManager(0, 200000, get_logger("all_expert_moe_tkg"), use_auto_alloc=use_auto_alloc)
    allocator = sbm.alloc_stack

    sbm.open_scope()
    allocator = sbm.alloc_stack

    # Load expert affinity
    if expert_affinities_in_sbuf:
        expertAffinityLoc = expert_affinities
    else:
        expertAffinityLoc = allocator(
            (dims.T, dims.E), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="expertAffinityLoc"
        )
        nisa.dma_copy(dst=expertAffinityLoc[: dims.T, : dims.E], src=expert_affinities[: dims.T, : dims.E])

    # Load input in shape of [128(H0), T, H//128(H1)]
    if hidden_in_sbuf:
        # Input is already in SBUF with shape [H0, T, H1_shard] (pre-sliced per shard)
        input_sb = params.hidden_tensor
    else:
        # Load from HBM to SBUF
        input_sb = allocator(
            [dims.H0, dims.T, dims.H1],
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="input_sb",
        )
        input_norm_load(params.hidden_tensor, input_sb, params, dims, sbm=sbm)

    # Allocate SBUF location to accumulate output
    output_temp = allocator((dims.H0, dims.H1_shard, dims.T), dtype=io_dtype, name=f"temp_output_sbuf", buffer=nl.sbuf)
    # TODO: support column tile
    # TODO: determine if accumulation buffer needs to be float32

    # Allocate SBUF locations for gate/up projection results
    gate_up_sb = allocator(
        (dims.I0, dims.num_total_128_tiles_per_I, dims.T), dtype=nl.float32, name=f"gate_up_sbuf", buffer=nl.sbuf
    )  # Note precision is f32 here.
    gate_output = gate_up_sb

    # Allocate SBUF locations for down results
    down_sb = allocator((dims.H0, dims.H1_shard, dims.T), dtype=io_dtype, name=f"down_sbuf", buffer=nl.sbuf)

    # Allocate identity tensor
    identity_hbm = nl.shared_constant(ntensor.identity(dims.T, nl.int8))
    identity_sb = allocator((dims.T, dims.T), dtype=io_dtype, buffer=nl.sbuf, name="identity_sb")
    nisa.dma_copy(dst=identity_sb, src=identity_hbm)

    # Wrap weight/bias tensors in TensorView for slicing
    gate_proj_weights_view = TensorView(params.gate_proj_weights_tensor)
    up_proj_weights_view = TensorView(params.up_proj_weights_tensor)
    down_proj_weights_view = TensorView(params.down_proj_weights_tensor)

    gate_proj_bias_view = safe_tensor_view(params.bias_params.gate_proj_bias_tensor)
    up_proj_bias_view = safe_tensor_view(params.bias_params.up_proj_bias_tensor)
    down_proj_bias_view = safe_tensor_view(params.bias_params.down_proj_bias_tensor)

    gate_w_scale_view = safe_tensor_view(params.quant_params.gate_w_scale)
    up_w_scale_view = safe_tensor_view(params.quant_params.up_w_scale)
    down_w_scale_view = safe_tensor_view(params.quant_params.down_w_scale)
    gate_up_in_scale_view = safe_tensor_view(params.quant_params.gate_up_in_scale)
    down_in_scale_view = safe_tensor_view(params.quant_params.down_in_scale)

    memory_safe_degree = 2 if dims.T * dims.H * dims.I <= 32 * 3072 * 1024 else 1
    sbm.open_scope(interleave_degree=memory_safe_degree)
    for expertIdx in range(dims.E):
        sbm.set_name_prefix(f"expert{expertIdx}_")

        # Pre-slice weights for this expert
        expert_gate_w = gate_proj_weights_view.select(dim=0, index=expertIdx)
        expert_up_w = up_proj_weights_view.select(dim=0, index=expertIdx)
        expert_down_w = down_proj_weights_view.select(dim=0, index=expertIdx)

        # Handle fused weights (if gate/up are fused in dim=1)
        if len(expert_gate_w.shape) > 2:
            expert_gate_w = expert_gate_w.select(dim=1, index=GateUpDim.GATE.value)
            expert_up_w = expert_up_w.select(dim=1, index=GateUpDim.UP.value)

        # Pre-slice biases for this expert
        expert_gate_b = None
        expert_up_b = None
        expert_down_b = None

        if gate_proj_bias_view != None:
            expert_gate_b = gate_proj_bias_view.select(dim=0, index=expertIdx)
            if len(expert_gate_b.shape) > 1:
                expert_gate_b = expert_gate_b.select(dim=0, index=GateUpDim.GATE.value)

        if up_proj_bias_view != None:
            expert_up_b = up_proj_bias_view.select(dim=0, index=expertIdx)
            if len(expert_up_b.shape) > 1:
                expert_up_b = expert_up_b.select(dim=0, index=GateUpDim.UP.value)

        if down_proj_bias_view != None:
            expert_down_b = down_proj_bias_view.select(dim=0, index=expertIdx)

        # Temporarily update params with pre-sliced weights and biases for this expert
        params.gate_proj_weights_tensor = expert_gate_w
        params.up_proj_weights_tensor = expert_up_w
        params.down_proj_weights_tensor = expert_down_w
        params.bias_params = MLPBiasParameters(
            gate_proj_bias_tensor=expert_gate_b,
            up_proj_bias_tensor=expert_up_b,
            down_proj_bias_tensor=expert_down_b,
        )

        if params.quant_params.quantization_type != QuantizationType.NONE:
            params.quant_params = _select_quant_scales(
                params.quant_params,
                gate_w_scale_view,
                up_w_scale_view,
                down_w_scale_view,
                gate_up_in_scale_view,
                down_in_scale_view,
                expertIdx,
            )

        # Gate Up projection
        gate_tile_info = process_gate_up_projection(
            hidden=input_sb,
            output=gate_output,
            params=params,
            dims=dims,
            sbm=sbm,
        )

        # Compute POST_SCALE

        if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            expert_affinities_sb = allocator(
                (dims._pmax, dims.T), dtype=io_dtype, buffer=nl.sbuf, name="expert_affinities_sb"
            )
            # transpose expert affinity and broadcast across partition dim
            affinityTpPsum = nl.ndarray(
                (1, dims.T), dtype=nl.float32, buffer=nl.psum, address=None if use_auto_alloc else (0, 0)
            )
            expertAffinityLoc_cast = allocator(
                (dims.T, 1), dtype=io_dtype, buffer=nl.sbuf, name="expertAffinityLoc_cast"
            )
            nisa.activation(
                dst=expertAffinityLoc_cast[: dims.T, :],
                op=nl.copy,
                data=expertAffinityLoc[: dims.T, nl.ds(expertIdx, 1)],
            )
            nisa.nc_matmul(
                dst=affinityTpPsum[:, :],
                stationary=expertAffinityLoc_cast,
                moving=identity_sb[: dims.T, : dims.T],
            )
            nisa.tensor_copy(
                src=affinityTpPsum[:, :],
                dst=expert_affinities_sb[:1, : dims.T],
            )
            for partition_group_idx in range(4):
                nisa.nc_stream_shuffle(
                    dst=expert_affinities_sb[nl.ds(32 * partition_group_idx, 32), : dims.T],
                    src=expert_affinities_sb[:1, : dims.T],
                    shuffle_mask=[0] * 32,
                )

        gate_up_sb_casted = allocator(
            (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
            dtype=io_dtype,
            name=f"gate_up_sbuf_with_io_dtype",
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=gate_up_sb_casted, src=gate_up_sb)
        (
            process_down_projection(
                hidden=gate_up_sb_casted,
                output=down_sb,
                params=params,
                dims=dims,
                gate_tile_info=gate_tile_info,
                sbm=sbm,
            ),
        )

        # Apply affinity and accumulate to SB
        if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            for shard in range(dims.H1_shard):
                nisa.tensor_tensor(
                    data1=down_sb[: dims.H0, shard, : dims.T],
                    data2=expert_affinities_sb,
                    op=nl.multiply,
                    dst=down_sb[: dims.H0, shard, : dims.T],
                )
        if expertIdx == 0:
            nisa.tensor_copy(
                dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, 0 : dims.T],
                src=down_sb[0 : dims.H0, 0 : dims.H1_shard, 0 : dims.T],
            )
        else:
            nisa.tensor_tensor(
                data1=down_sb[0 : dims.H0, 0 : dims.H1_shard, 0 : dims.T],
                data2=output_temp[0 : dims.H0, 0 : dims.H1_shard, 0 : dims.T],
                op=nl.add,
                dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, 0 : dims.T],
            )

        sbm.increment_section()
    sbm.close_scope()
    sbm.set_name_prefix("")

    # Store output
    output_in_sbuf = output.buffer == nl.sbuf
    if output_in_sbuf:
        # Transpose output_temp [H0, H1_shard, T] -> [H0, T, H1_shard] for SBUF output
        for h1 in range(dims.H1_shard):
            nisa.tensor_copy(dst=output[:, :, h1], src=output_temp[:, h1, :])
    else:
        transpose_store(output_temp, output, dims, params.output_dtype, sbm)

    sbm.close_scope()
    return output


def _select_quant_scales(
    quant_params: MLPQuantizationParameters,
    gate_w_scale_view: TensorView,
    up_w_scale_view: TensorView,
    down_w_scale_view: TensorView,
    gate_up_in_scale_view: TensorView,
    down_in_scale_view: TensorView,
    expertIdx: int,
):
    """
    Select and reshape quantization scales for a specific expert.

    Args:
        quant_params (MLPQuantizationParameters): Quantization parameters.
        gate_w_scale_view (TensorView): Gate weight scale tensor view.
        up_w_scale_view (TensorView): Up weight scale tensor view.
        down_w_scale_view (TensorView): Down weight scale tensor view.
        gate_up_in_scale_view (TensorView): Gate/up input scale tensor view.
        down_in_scale_view (TensorView): Down input scale tensor view.
        expertIdx (int): Expert index to select scales for.

    Returns:
        MLPQuantizationParameters: Quantization parameters with scales for the specified expert.
    """
    quantization_type = quant_params.quantization_type
    expert_gate_w_scale = None
    if gate_w_scale_view != None:
        expert_gate_w_scale = gate_w_scale_view.select(dim=0, index=expertIdx).select(dim=0, index=GateUpDim.GATE.value)
        expert_gate_w_scale = reshape_scale_for_mlp(expert_gate_w_scale)

    expert_up_w_scale = None
    if up_w_scale_view != None:
        expert_up_w_scale = up_w_scale_view.select(dim=0, index=expertIdx).select(dim=0, index=GateUpDim.UP.value)
        expert_up_w_scale = reshape_scale_for_mlp(expert_up_w_scale)

    expert_down_w_scale = None
    if down_w_scale_view != None:
        expert_down_w_scale = down_w_scale_view.select(dim=0, index=expertIdx)
        expert_down_w_scale = reshape_scale_for_mlp(expert_down_w_scale)

    expert_gate_up_in_scale = None
    if gate_up_in_scale_view != None:
        expert_gate_up_in_scale = reshape_scale_for_mlp(gate_up_in_scale_view.select(dim=0, index=expertIdx))

    expert_down_in_scale = None
    if down_in_scale_view != None:
        expert_down_in_scale = reshape_scale_for_mlp(down_in_scale_view.select(dim=0, index=expertIdx))

    return MLPQuantizationParameters(
        quantization_type=quantization_type,
        gate_w_scale=expert_gate_w_scale,
        up_w_scale=expert_up_w_scale,
        down_w_scale=expert_down_w_scale,
        gate_up_in_scale=expert_gate_up_in_scale,
        down_in_scale=expert_down_in_scale,
        clipping_bound=quant_params.clipping_bound if quant_params != None else None,
    )
