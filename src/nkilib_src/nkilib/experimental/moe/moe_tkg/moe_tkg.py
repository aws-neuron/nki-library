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

"""Experimental MoE TKG: uses nkiprimitives-based selective expert MX implementation."""

from typing import Optional

import nki.language as nl

# MLP utils
from ....core.mlp.mlp_parameters import MLPExpertParameters, MLPParameters, get_T_from_hidden_input
from ....core.moe.moe_tkg.all_expert_impl import _all_expert_moe_tkg
from ....core.moe.moe_tkg.all_expert_mx_impl import _all_expert_moe_tkg_mx
from ....core.moe.moe_tkg.moe_tkg_affinity_masking import mask_expert_affinities
from ....core.moe.moe_tkg.selective_expert_impl import _selective_expert_moe_tkg

# common utils
from ....core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
    NormType,
    QuantizationType,
)
from ....core.utils.kernel_assert import kernel_assert
from .selective_expert_mx_primitives import _selective_expert_moe_tkg_mxfp4_primitives

# Constants
_SUPPORTED_MX_DTYPES = (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4)
_MOE_TKG_ERROR_PREFIX = "[MoE TKG Kernel]"


def moe_tkg(
    hidden_input: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    is_all_expert: bool,
    rank_id: Optional[nl.NkiTensor] = None,
    expert_gate_up_bias: Optional[nl.NkiTensor] = None,
    expert_down_bias: Optional[nl.NkiTensor] = None,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor] = None,
    expert_down_weights_scale: Optional[nl.NkiTensor] = None,
    hidden_input_scale: Optional[nl.NkiTensor] = None,
    expert_gate_up_input_scale: Optional[nl.NkiTensor] = None,
    expert_down_input_scale: Optional[nl.NkiTensor] = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: Optional[nl.NkiTensor] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype=None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output_in_sbuf: bool = False,
    is_all_expert_dynamic: bool = False,
    block_size: int = None,
    input_dequant_scale: Optional[nl.NkiTensor] = None,
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED,
) -> nl.NkiTensor:
    """
    Experimental MoE TKG kernel. Identical to core except selective-expert MX
    dispatches to the nkiprimitives-based implementation.
    """

    # Extract quantization type
    quant_type, is_mx_kernel = _extract_quantization_type(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
    )

    # For all-expert mode with affinity scaling, mask expert affinities based on rank_id
    if (
        is_all_expert
        and expert_affinities_scaling_mode != ExpertAffinityScaleMode.NO_SCALE
        and all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED
    ):
        kernel_assert(
            rank_id != None, f"{_MOE_TKG_ERROR_PREFIX} rank_id is required for all-expert mode with affinity scaling"
        )

        # Get dimensions for masking
        E_L = expert_gate_up_weights.shape[0]
        T = get_T_from_hidden_input(hidden_input, hidden_input_scale)
        K = expert_index.shape[-1]

        masked_expert_affinities = mask_expert_affinities(
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            rank_id=rank_id,
            E_L=E_L,
            T=T,
            K=K,
            io_dtype=expert_affinities.dtype,
            mask_unselected_experts=mask_unselected_experts,
            output_in_sbuf=not is_all_expert_dynamic,
        )
    else:
        masked_expert_affinities = expert_affinities

    # Initialize config objects
    expert_params = MLPExpertParameters(
        expert_affinities=masked_expert_affinities,
        expert_index=expert_index,
        expert_affinities_eager=expert_affinities_eager,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        is_all_expert_dynamic=is_all_expert_dynamic,
        all_to_all_v_strategy=all_to_all_v_strategy,
        block_size=block_size,
    )

    output_dtype = hidden_input.dtype if output_dtype == None else output_dtype

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
        gate_up_in_scale=expert_gate_up_input_scale,
        down_in_scale=expert_down_input_scale,
        hidden_input_scale=hidden_input_scale,
        output_dtype=output_dtype,
        use_tkg_gate_up_proj_column_tiling=False,
        use_tkg_down_proj_column_tiling=False,
        shard_on_h_disabled=is_mx_kernel and not is_all_expert,
        expert_params=expert_params,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        quantization_type=quant_type,
    )

    T = mlp_params.sequence_len
    H = mlp_params.hidden_size

    # Validate inputs
    _validate_moe_tkg_inputs(
        T=T,
        is_all_expert=is_all_expert,
        is_all_expert_dynamic=is_all_expert_dynamic,
        block_size=block_size,
        is_mx_kernel=is_mx_kernel,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        hidden_input_scale=hidden_input_scale,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
        expert_affinities_eager=expert_affinities_eager,
    )

    # Allocate output tensor
    if output_in_sbuf:
        output = nl.ndarray(hidden_input.shape, dtype=output_dtype, buffer=nl.sbuf, name="output_sb")
    else:
        output = nl.ndarray((T, H), dtype=output_dtype, buffer=nl.shared_hbm)

    # Dispatch to expert MLP implementation
    if is_all_expert:
        if is_mx_kernel:
            _all_expert_moe_tkg_mx(mlp_params, output)
        else:
            _all_expert_moe_tkg(mlp_params, output)
    else:
        if is_mx_kernel:
            # EXPERIMENTAL: use nkiprimitives-based implementation
            _selective_expert_moe_tkg_mxfp4_primitives(mlp_params, output)
        else:
            _selective_expert_moe_tkg(mlp_params, output)

    return output


def _extract_quantization_type(
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor],
    expert_down_weights_scale: Optional[nl.NkiTensor],
    expert_gate_up_input_scale: Optional[nl.NkiTensor],
    expert_down_input_scale: Optional[nl.NkiTensor],
) -> tuple[QuantizationType, bool]:
    quant_type = QuantizationType.NONE
    is_mx_kernel = False
    if expert_gate_up_weights.dtype in _SUPPORTED_MX_DTYPES:
        quant_type = QuantizationType.MX
        is_mx_kernel = True
    elif expert_gate_up_input_scale != None and expert_down_input_scale != None:
        quant_type = QuantizationType.STATIC
    elif expert_gate_up_weights_scale != None and expert_down_weights_scale != None:
        quant_type = QuantizationType.ROW

    return quant_type, is_mx_kernel


def _validate_moe_tkg_inputs(
    T: int,
    is_all_expert: bool,
    is_all_expert_dynamic: bool,
    block_size: int,
    is_mx_kernel: bool,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor],
    expert_down_weights_scale: Optional[nl.NkiTensor],
    hidden_input_scale: Optional[nl.NkiTensor],
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    expert_gate_up_input_scale: Optional[nl.NkiTensor],
    expert_down_input_scale: Optional[nl.NkiTensor],
    expert_affinities_eager: Optional[nl.NkiTensor],
) -> None:
    if is_mx_kernel:
        kernel_assert(
            expert_gate_up_weights_scale != None and expert_down_weights_scale != None,
            f"{_MOE_TKG_ERROR_PREFIX} Scales must be set when using MX weights",
        )

    if is_all_expert_dynamic:
        kernel_assert(
            is_all_expert,
            f"{_MOE_TKG_ERROR_PREFIX} is_all_expert_dynamic=True requires is_all_expert=True, but got {is_all_expert=}",
        )
        kernel_assert(
            block_size != None,
            f"{_MOE_TKG_ERROR_PREFIX} is_all_expert_dynamic=True requires block_size != None, but got {block_size=}",
        )

    kernel_assert(
        hidden_input_scale == None or (is_mx_kernel and is_all_expert),
        f"{_MOE_TKG_ERROR_PREFIX} hidden_input_scale is only supported with MX weights in all-expert mode",
    )

    kernel_assert(
        T <= 128 or is_all_expert,
        f"{_MOE_TKG_ERROR_PREFIX} Currently only batch size * seq len <= 128 is supported (except for all-expert mode)",
    )

    kernel_assert(
        expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        f"{_MOE_TKG_ERROR_PREFIX} PRE_SCALE_DELAYED option is only applicable in CTE expert_mlp case",
    )

    kernel_assert(
        expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE,
        f"{_MOE_TKG_ERROR_PREFIX} Kernel does not support pre-scale mode",
    )

    kernel_assert(
        expert_gate_up_input_scale == None and expert_down_input_scale == None,
        f"{_MOE_TKG_ERROR_PREFIX} Static quantization is not supported in MoE TKG kernel",
    )

    if is_all_expert:
        kernel_assert(
            expert_affinities_eager == None,
            f"{_MOE_TKG_ERROR_PREFIX} expert_affinities eager mode not supported with is_all_expert=True",
        )
