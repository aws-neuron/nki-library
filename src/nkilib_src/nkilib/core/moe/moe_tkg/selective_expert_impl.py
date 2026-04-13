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

"""Selective-expert MoE token generation implementation that processes only top-K selected experts per token."""

import nki.isa as nisa
import nki.language as nl

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

# common utils
from ...utils.allocator import SbufManager
from ...utils.common_types import ExpertAffinityScaleMode, GateUpDim
from ...utils.kernel_helpers import get_verified_program_sharding_info
from ...utils.logging import get_logger
from ...utils.tensor_view import TensorView
from .moe_tkg_utils import (
    broadcast_token_affinity,
    gather_expert_affinities,
    reshape_scale_for_mlp,
)


def _selective_expert_moe_tkg(
    params: MLPParameters,
    output: nl.ndarray,
) -> nl.ndarray:
    """
    Selective-expert Mixture of Experts (MoE) kernel for token generation (TKG).

    Processes only the top-K selected experts for each token, computing MLP projections
    for the selected experts and accumulating results weighted by expert affinities.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        output (nl.ndarray): Output tensor to store the final result.

    Returns:
        output (nl.ndarray): Output tensor with accumulated expert results.

    Notes:
        - Processes tokens sequentially, experts selectively based on top-K indices
        - Uses TensorView for dynamic expert weight selection
        - Column tiling is disabled for this implementation
        - SBUF I/O mode is supported

    Pseudocode:
        input_sb[H0, T, H1] = normalize(hidden_tensor[T, H])
        output_temp[H0, H1_shard, T] = zeros()

        # Gather expert affinities for efficient access
        gathered_affinities = gather_expert_affinities(expert_affinities, expert_index)

        for token_idx in range(T):
            token_affinities = broadcast_token_affinity(gathered_affinities, token_idx)

            for k in range(K):  # top-K experts
                expert_idx = expert_index[token_idx, k]
                gate_w[I, H], up_w[I, H], down_w[H, I] = weights[expert_idx]

                # Gate-Up projection: act_fn(gate(x)) * up(x)
                gate_up[I0, I1, 1] = gate_up_proj(input_sb[H0, token_idx:token_idx+1, H1], gate_w, up_w)

                # Down projection
                down[H0, H1_shard] = down_proj(gate_up[I0, I1, 1], down_w)

                # Scale by affinity if POST_SCALE
                if affinity_scaling_mode == POST_SCALE:
                    down[H0, H1_shard] *= token_affinities[k]

                # Accumulate results for this token
                if k == 0:
                    output_temp[H0, H1_shard, token_idx] = down[H0, H1_shard]
                else:
                    output_temp[H0, H1_shard, token_idx] += down[H0, H1_shard]

        output[T, H] = transpose(output_temp[H0, H1_shard, T])
    """

    # Check if input is already in SBUF
    hidden_in_sbuf = params.hidden_tensor.buffer == nl.sbuf

    # TODO: Calibrate weight tile calculations and remove auto allocation workaround
    H = params.hidden_tensor.shape[-1]
    need_auto_alloc = H >= 16 * 1024 or hidden_in_sbuf
    sbm = SbufManager(0, 200 * 1024, get_logger("selective_expert_moe_tkg"), use_auto_alloc=need_auto_alloc)
    sbm.open_scope()

    io_dtype = params.hidden_tensor.dtype
    expert_index_input = params.expert_params.expert_index
    expert_affinities = params.expert_params.expert_affinities
    gate_up_weights = params.gate_proj_weights_tensor

    program_sharding_info = get_verified_program_sharding_info("moe_tkg", (0, 1))
    num_shards = program_sharding_info[1]
    shard_id = program_sharding_info[2]

    T = expert_index_input.shape[0]
    I = gate_up_weights.shape[-1]
    shard_on_T = True

    # Disable shard_on_T when:
    # 1. T == 1: Only one token, no benefit from sharding on this dimension
    # 2. H * I >= 3072 * 1536: Big config has mlp tkg tile size calculation bug (NKL-1013)
    if T == 1 or H * I >= 3072 * 1536:
        shard_on_T = False

    # For odd T, use ceiling division: core 0 gets T//2, core 1 gets T - T//2
    if shard_on_T:
        T_first_shard = T // num_shards
        T_second_shard = T - T_first_shard
        T_per_shard = T_first_shard if shard_id == 0 else T_second_shard
        T_offset = 0 if shard_id == 0 else T_first_shard
    else:
        T_per_shard = T
        T_offset = 0

    params.shard_on_h_disabled = shard_on_T
    dims = MLPTKGConstants.calculate_constants(params)

    # Load input in shape of [128(H0), T, H//128(H1)]
    if hidden_in_sbuf:
        # Input is already in SBUF
        input_sb = params.hidden_tensor
    else:
        # TODO: only load for local tokens
        input_sb = sbm.alloc_stack(
            [dims.H0, T, dims.H1_shard],
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="input_sb",
        )
        input_norm_load(params.hidden_tensor, input_sb, params, dims, sbm=sbm)

    # Allocate SBUF location to accumulate output
    output_temp = sbm.alloc_stack(
        (dims.H0, dims.H1_shard, T_per_shard),
        dtype=io_dtype,
        name=f"temp_output_sbuf",
        buffer=nl.sbuf,
    )

    # Allocate SBUF locations for gate/up projection result, for each token
    gate_up_output = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, dims.K),
        dtype=io_dtype,
        name=f"intermediate_state_sbuf",
        buffer=nl.sbuf,
    )

    # Allocate SBUF locations for down result
    down_output_list = []
    for expert_k_idx in range(dims.K):
        down_sb = sbm.alloc_stack(
            (dims.H0, dims.H1_shard), dtype=io_dtype, name=f"down_sbuf_{expert_k_idx}", buffer=nl.sbuf
        )
        down_output_list.append(down_sb)

    # Reshape gate_up weights from [E, H, 2, I] to [E, H, 2 * I]
    E, H, i_2, I = gate_up_weights.shape
    gate_up_weights = gate_up_weights.reshape((E, H, I * i_2))

    # Load expert index
    if expert_index_input.buffer == nl.sbuf:
        expert_idx = expert_index_input
    else:
        expert_idx = sbm.alloc_stack(
            (dims.T, dims.K),
            dtype=expert_index_input.dtype,
            name=f"expert_idx_sbuf",
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=expert_idx, src=expert_index_input[0 : dims.T, 0 : dims.K])  # indices have to be in SBUF

    expert_affinities_sb = sbm.alloc_stack(
        (dims._pmax, dims.E),
        dtype=expert_affinities.dtype,
        name=f"expert_affinities_sb",
        buffer=nl.sbuf,
    )
    # Load expert affinity
    if expert_affinities.buffer == nl.sbuf:
        nisa.memset(expert_affinities_sb, value=0.0)
        nisa.tensor_copy(dst=expert_affinities_sb[0 : dims.T, 0 : dims.E], src=expert_affinities)
    else:
        # Prefetch expertIndices (Up to 128 tokens input)
        nisa.dma_copy(
            dst=expert_affinities_sb[0 : dims.T, 0 : dims.E],
            src=expert_affinities[0 : dims.T, 0 : dims.E],
        )

    # Gather expert affinities using utility function
    gathered_affinities_sb = gather_expert_affinities(expert_affinities_sb, expert_idx, dims, sbm)
    params.use_tkg_gate_up_proj_column_tiling = False
    params.use_tkg_down_proj_column_tiling = False

    initial_gate_proj_weights_tensor = params.gate_proj_weights_tensor
    initial_up_proj_weights_tensor = params.up_proj_weights_tensor
    initial_down_proj_weights_tensor = params.down_proj_weights_tensor

    initial_mlp_bias_params = params.bias_params
    initial_mlp_quant_params = params.quant_params

    memory_safe_degree = 2
    if shard_on_T:
        memory_safe_degree = 2 if dims.H * dims.I < 3072 * 1024 else 1

    # convert dims.T to 1 to compute output by each token
    dims.T = 1

    for local_token_idx in range(T_per_shard):
        global_token_idx = local_token_idx + T_offset
        sbm.set_name_prefix(f"T{global_token_idx}_")
        # Load Expert Affinities per Token using utility function
        expert_affinity_sb = sbm.alloc_stack(
            (dims._pmax, dims.K),
            dtype=expert_affinities.dtype,
            buffer=nl.sbuf,
            name=f"expert_affinity_sb",
        )
        broadcast_token_affinity(expert_affinity_sb, gathered_affinities_sb, global_token_idx, dims, sbm)

        sbm.open_scope(interleave_degree=memory_safe_degree)
        for expert_k_idx in range(dims.K):
            sbm.set_name_prefix(f"T{global_token_idx}_K{expert_k_idx}_")
            # Gate Up projection

            # Change back to scalar_offset=expert_idx[global_token_idx, expert_k_idx], after NKI-333 is fixed
            expert_id_scalar_offset = expert_idx.ap(
                pattern=[[dims.K, 1], [1, 1]], offset=global_token_idx * dims.K + expert_k_idx
            )
            params.gate_proj_weights_tensor = (
                TensorView(initial_gate_proj_weights_tensor)
                .select(dim=0, index=expert_id_scalar_offset)
                .select(dim=1, index=GateUpDim.GATE.value)
            )

            params.up_proj_weights_tensor = (
                TensorView(initial_up_proj_weights_tensor)
                .select(dim=0, index=expert_id_scalar_offset)
                .select(dim=1, index=GateUpDim.UP.value)
            )

            params.down_proj_weights_tensor = TensorView(initial_down_proj_weights_tensor).select(
                dim=0, index=expert_id_scalar_offset
            )

            gate_proj_bias_tensor_view = None
            up_proj_bias_tensor_view = None
            down_proj_bias_tensor_view = None
            if initial_mlp_bias_params.gate_proj_bias_tensor != None:
                gate_proj_bias_tensor_view = (
                    TensorView(initial_mlp_bias_params.gate_proj_bias_tensor)
                    .select(dim=0, index=expert_id_scalar_offset)
                    .select(dim=0, index=GateUpDim.GATE.value)
                )

            if initial_mlp_bias_params.up_proj_bias_tensor != None:
                up_proj_bias_tensor_view = (
                    TensorView(initial_mlp_bias_params.up_proj_bias_tensor)
                    .select(dim=0, index=expert_id_scalar_offset)
                    .select(dim=0, index=GateUpDim.UP.value)
                )

            if initial_mlp_bias_params.down_proj_bias_tensor != None:
                down_proj_bias_tensor_view = TensorView(initial_mlp_bias_params.down_proj_bias_tensor).select(
                    dim=0, index=expert_id_scalar_offset
                )

            params.bias_params = MLPBiasParameters(
                gate_proj_bias_tensor=gate_proj_bias_tensor_view,
                up_proj_bias_tensor=up_proj_bias_tensor_view,
                down_proj_bias_tensor=down_proj_bias_tensor_view,
            )

            params.quant_params = _select_quant_scales(
                initial_mlp_quant_params,
                expert_id_scalar_offset,
            )

            gate_tile_info = process_gate_up_projection(
                hidden=input_sb[:, global_token_idx : global_token_idx + 1, :],
                output=gate_up_output[:, :, expert_k_idx : expert_k_idx + 1],
                params=params,
                dims=dims,
                sbm=sbm,
            )

            # Down projection
            down_sb = down_output_list[expert_k_idx]
            process_down_projection(
                hidden=gate_up_output[:, :, expert_k_idx : expert_k_idx + 1],
                output=down_sb,
                params=params,
                dims=dims,
                gate_tile_info=gate_tile_info,
                sbm=sbm,
            )

            if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
                # Apply affinity and accumulate to SB
                nisa.tensor_scalar(
                    dst=down_sb,
                    data=down_sb,
                    op0=nl.multiply,
                    operand0=expert_affinity_sb[:, expert_k_idx],
                )
            if expert_k_idx == 0:
                nisa.tensor_copy(dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx], src=down_sb)
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx],
                    data1=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx],
                    data2=down_sb,
                    op=nl.add,
                )

            sbm.increment_section()
        sbm.close_scope()

    # Save output result
    sbm.set_name_prefix("")

    dims.T = T_per_shard

    # Store output
    if output.buffer == nl.sbuf:
        # Transpose output_temp [H0, H1_shard, T_per_shard] -> [H0, T, H1_shard] for SBUF output
        for h1_idx in range(dims.H1_shard):
            nisa.tensor_copy(dst=output[:, T_offset : T_offset + T_per_shard, h1_idx], src=output_temp[:, h1_idx, :])
    else:
        transpose_store(output_temp, output, dims, params.output_dtype, sbm, T_offset)

    sbm.close_scope()
    return output


def _select_quant_scales(quant_params: MLPQuantizationParameters, expert_id_offset: nl.ndarray):
    """
    Select and reshape quantization scales for a specific expert.

    Args:
        quant_params (MLPQuantizationParameters): Quantization parameters.
        expert_id_offset (nl.ndarray): Expert ID offset for selecting scales.

    Returns:
        MLPQuantizationParameters: Quantization parameters with scales for the specified expert.
    """
    gate_w_scale_view = None
    up_w_scale_view = None
    down_w_scale_view = None
    gate_up_in_scale_view = None
    down_in_scale_view = None

    if quant_params.gate_w_scale != None:
        gate_w_scale_view = (
            TensorView(quant_params.gate_w_scale)
            .select(dim=0, index=expert_id_offset)
            .select(dim=0, index=GateUpDim.GATE.value)
        )
        gate_w_scale_view = reshape_scale_for_mlp(gate_w_scale_view)

    if quant_params.up_w_scale != None:
        up_w_scale_view = (
            TensorView(quant_params.up_w_scale)
            .select(dim=0, index=expert_id_offset)
            .select(dim=0, index=GateUpDim.UP.value)
        )
        up_w_scale_view = reshape_scale_for_mlp(up_w_scale_view)

    if quant_params.down_w_scale != None:
        down_w_scale_view = TensorView(quant_params.down_w_scale).select(dim=0, index=expert_id_offset)
        down_w_scale_view = reshape_scale_for_mlp(down_w_scale_view)

    if quant_params.gate_up_in_scale != None:
        gate_up_in_scale_view = TensorView(quant_params.gate_up_in_scale).select(dim=0, index=expert_id_offset)
        gate_up_in_scale_view = reshape_scale_for_mlp(gate_up_in_scale_view)

    if quant_params.down_in_scale != None:
        down_in_scale_view = TensorView(quant_params.down_in_scale).select(dim=0, index=expert_id_offset)
        down_in_scale_view = reshape_scale_for_mlp(down_in_scale_view)

    return MLPQuantizationParameters(
        quantization_type=quant_params.quantization_type,
        gate_w_scale=gate_w_scale_view,
        up_w_scale=up_w_scale_view,
        down_w_scale=down_w_scale_view,
        gate_up_in_scale=gate_up_in_scale_view,
        down_in_scale=down_in_scale_view,
        clipping_bound=quant_params.clipping_bound,
    )
