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

"""This module implements the Mixture of Experts (MoE) token generation kernel with support for all-expert and selective-expert modes."""

import math
from typing import Optional

import nki
import nki.language as nl

# MLP utils
from ...mlp.mlp_parameters import MLPExpertParameters, MLPParameters

# common utils
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode, NormType, QuantizationType
from ...utils.kernel_assert import kernel_assert
from .all_expert_impl import _all_expert_moe_tkg
from .all_expert_mx_impl import _all_expert_moe_tkg_mx
from .moe_tkg_affinity_masking import mask_expert_affinities
from .selective_expert_impl import _selective_expert_moe_tkg
from .selective_expert_mx_impl import _selective_expert_moe_tkg_mxfp4


def moe_tkg(
    hidden_input: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    is_all_expert: bool,
    rank_id: Optional[nl.ndarray] = None,
    expert_gate_up_bias: Optional[nl.ndarray] = None,
    expert_down_bias: Optional[nl.ndarray] = None,
    expert_gate_up_weights_scale: Optional[nl.ndarray] = None,
    expert_down_weights_scale: Optional[nl.ndarray] = None,
    gate_up_input_scale: Optional[nl.ndarray] = None,
    down_input_scale: Optional[nl.ndarray] = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: Optional[nl.ndarray] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype=None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output_in_sbuf: bool = False,
) -> nl.ndarray:
    """
    Mixture of Experts (MoE) MLP token generation kernel.

    Performs MoE computation with support for both all-expert and selective-expert modes.
    Supports various quantization types including FP8 row/static quantization and MxFP4.
    Optimized for token generation scenarios with T <= 128 (except MX all-expert mode).

    Supported input data types: bfloat16, float16, float4_e2m1fn_x4 (MxFP4).

    Dimensions:
        T: Number of tokens (batch_size * seq_len)
        H: Hidden dimension
        I: Intermediate dimension
        E: Number of global experts
        E_L: Number of local experts processed by this kernel
        K: Top-k experts per token
        I_p: I//4 if I <= 512 else 128

    Args:
        hidden_input (nl.ndarray): [T, H] in HBM or [H0, T, H1] in SBUF, Input hidden states tensor.
        expert_gate_up_weights (nl.ndarray): [E_L, H, 2, I] for bf16/fp16 or [E_L, 128, 2, ceil(H/512), I] for MxFP4,
            Fused gate and up projection weights.
        expert_down_weights (nl.ndarray): [E_L, I, H] for bf16/fp16 or [E_L, I_p, ceil(I/512), H] for MxFP4,
            Down projection weights.
        expert_affinities (nl.ndarray): [T, E], Expert routing weights/affinities. For all-expert mode with
            affinity scaling, this will be sliced to [T, E_L] internally.
        expert_index (nl.ndarray): [T, K], Top-K expert indices per token.
        is_all_expert (bool): If True, process all experts for all tokens; otherwise, process only selected
            top-k experts.
        rank_id (nl.ndarray, optional): [1, 1], Rank ID tensor specifying which worker processes experts
            [E_L * rank_id, E_L * (rank_id + 1)). Required for all-expert mode with affinity scaling enabled.
        expert_gate_up_bias (nl.ndarray, optional): [E_L, 2, I] for non-MX or [E_L, I_p, 2, ceil(I/512), 4]
            for MX, Bias for gate/up projections.
        expert_down_bias (nl.ndarray, optional): [E_L, H], Bias for down projection.
        expert_gate_up_weights_scale (nl.ndarray, optional): [E_L, 2, I] for FP8 row quantization, [E_L, 2, 1] for
            FP8 static quantization, or [E_L, 128/8, 2, ceil(H/512), I] for MxFP4, Quantization scales for
            gate/up weights.
        expert_down_weights_scale (nl.ndarray, optional): [E_L, H] for FP8 row quantization, [E_L, 1] for FP8 static
            quantization, or [E_L, I_p/8, ceil(I/512), H] for MxFP4, Quantization scales for down weights.
        gate_up_input_scale (nl.ndarray, optional): [E_L, 1], FP8 dequantization scales for gate/up input.
            Used for static quantization.
        down_input_scale (nl.ndarray, optional): [E_L, 1], FP8 dequantization scales for down input. Used for
            static quantization.
        mask_unselected_experts (bool): Whether to apply expert affinity masking based on expert_index. When
            True, affinities are masked to zero for experts not selected by each token. Only used in all-expert
            mode with affinity scaling. (default: False)
        expert_affinities_eager (nl.ndarray, optional): [T, K], Eager expert affinities. Not used in
            all_expert mode.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): When to apply affinity scaling. Supported
            values: NO_SCALE, POST_SCALE. (default: NO_SCALE)
        activation_fn (ActFnType): Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to None; if None, uses hidden_input dtype.
        gate_clamp_upper_limit (float, optional): Upper bound value to clamp gate projection results.
        gate_clamp_lower_limit (float, optional): Lower bound value to clamp gate projection results.
        up_clamp_upper_limit (float, optional): Upper bound value to clamp up projection results.
        up_clamp_lower_limit (float, optional): Lower bound value to clamp up projection results.
        output_in_sbuf (bool): If True, allocate output in SBUF with same shape as hidden_input. If False
            (default), allocate output in HBM with shape [T, H].

    Returns:
        output (nl.ndarray): [T, H] or same shape as hidden_input if output_in_sbuf=True, Output tensor with
            MoE computation results.

    Notes:
        - T <= 128 (batch_size * seq_len must be <= 128, except for MX all-expert mode)
        - PRE_SCALE and PRE_SCALE_DELAYED modes are not supported
        - Column tiling is disabled for MoE kernels
        - Static quantization is not currently supported

    Pseudocode:
        # Mask expert affinities if needed (all-expert mode with affinity scaling)
        if is_all_expert and expert_affinities_scaling_mode != NO_SCALE:
            masked_expert_affinities = mask_expert_affinities(expert_affinities, expert_index, rank_id)

        # Process experts
        output = zeros([T, H])
        for each expert (all-expert) or selected expert (selective-expert):
            gate_proj_out = hidden_states @ gate_weights
            act_gate_proj = activation_fn(gate_proj_out)
            up_proj_out = hidden_states @ up_weights
            intermediate = act_gate_proj * up_proj_out
            expert_out = intermediate @ down_weights
            if expert_affinities_scaling_mode == POST_SCALE:
                expert_out *= affinity
            output += expert_out
    """
    # TODO: Currently only supports mxfp4 (nl.float4_e2m1fn_x4). Add mxfp8 dtype when implemented.
    _MX_SUPPORTED_DTYPES = (nl.float4_e2m1fn_x4,)
    is_mx_kernel = expert_gate_up_weights.dtype in _MX_SUPPORTED_DTYPES

    if is_mx_kernel:
        kernel_assert(
            expert_gate_up_weights_scale != None and expert_down_weights_scale != None,
            "Scales must be set when using MX weights",
        )

    # For all-expert mode with affinity scaling, mask expert affinities based on rank_id
    masked_expert_affinities = expert_affinities
    if is_all_expert and expert_affinities_scaling_mode != ExpertAffinityScaleMode.NO_SCALE:
        kernel_assert(rank_id != None, "rank_id is required for all-expert mode with affinity scaling")

        # Get dimensions for masking
        E_L = expert_gate_up_weights.shape[0]
        T = expert_index.shape[0]
        K = expert_index.shape[1]

        masked_expert_affinities = mask_expert_affinities(
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            rank_id=rank_id,
            E_L=E_L,
            T=T,
            K=K,
            io_dtype=hidden_input.dtype,
            mask_unselected_experts=mask_unselected_experts,
        )

    expert_params = MLPExpertParameters(
        expert_affinities=masked_expert_affinities,
        expert_index=expert_index,
        expert_affinities_eager=expert_affinities_eager,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
    )

    output_dtype = hidden_input.dtype if output_dtype == None else output_dtype
    quant_type = QuantizationType.NONE
    if is_mx_kernel:
        quant_type = QuantizationType.MX
    elif gate_up_input_scale != None and down_input_scale != None:
        quant_type = QuantizationType.STATIC
    elif expert_gate_up_weights_scale != None and expert_down_weights_scale != None:
        quant_type = QuantizationType.ROW

    mlp_params = MLPParameters(
        hidden_tensor=hidden_input,
        gate_proj_weights_tensor=expert_gate_up_weights,
        up_proj_weights_tensor=expert_gate_up_weights,
        down_proj_weights_tensor=expert_down_weights,
        activation_fn=activation_fn,
        normalization_type=NormType.NO_NORM,
        gate_proj_bias_tensor=expert_gate_up_bias,
        up_proj_bias_tensor=expert_gate_up_bias,
        down_proj_bias_tensor=expert_down_bias,
        gate_w_scale=expert_gate_up_weights_scale,
        up_w_scale=expert_gate_up_weights_scale,
        down_w_scale=expert_down_weights_scale,
        gate_up_in_scale=gate_up_input_scale,
        down_in_scale=down_input_scale,
        output_dtype=output_dtype,
        use_tkg_gate_up_proj_column_tiling=False,
        use_tkg_down_proj_column_tiling=False,
        shard_on_k=is_mx_kernel and not is_all_expert,
        expert_params=expert_params,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        quantization_type=quant_type,
    )

    T = mlp_params.sequence_len
    H = mlp_params.hidden_size
    # Allocate output tensor
    if output_in_sbuf:
        output = nl.ndarray(hidden_input.shape, dtype=output_dtype, buffer=nl.sbuf, name="output_sb")
    else:
        output = nl.ndarray((T, H), dtype=output_dtype, buffer=nl.shared_hbm)

    # Limitations on current implementations
    # TODO: relax all of the following restrictions
    kernel_assert(
        T <= 128 or (is_mx_kernel and is_all_expert),
        "currently only batch size * seq len <= 128 is supported (except for MX all-expert mode)",
    )

    kernel_assert(
        expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        "PRE_SCALE_DELAYED option is only applicable in CTE expert_mlp case",
    )

    kernel_assert(
        expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE,
        "[Expert MLP kernel] kernel does not support pre-scale mode now",
    )

    kernel_assert(
        gate_up_input_scale == None and down_input_scale == None,
        "static quantization are not supported in MoE TKG kernel",
    )

    if is_all_expert:
        kernel_assert(
            expert_affinities_eager == None,
            "[Expert MLP kernel] all_expert flavor does not support expert_affinities eager mode",
        )
        if is_mx_kernel:
            _all_expert_moe_tkg_mx(mlp_params, output)
        else:
            _all_expert_moe_tkg(mlp_params, output)
    else:
        if is_mx_kernel:
            _selective_expert_moe_tkg_mxfp4(mlp_params, output)
        else:
            _selective_expert_moe_tkg(mlp_params, output)
    return output
