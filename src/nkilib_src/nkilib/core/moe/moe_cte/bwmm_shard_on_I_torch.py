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
"""PyTorch reference implementations for bwmm_shard_on_I kernels.

Provides dispatch-ready torch_refs for:
- blockwise_mm_baseline_shard_intermediate
- blockwise_mm_baseline_shard_intermediate_hybrid
- blockwise_mm_shard_intermediate_dropping

Each function matches its kernel's signature exactly, then delegates to
_moe_cte_torch_ref_impl with the appropriate BWMMFunc variant.
"""

from typing import Optional

import torch

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from .bwmm_func import BWMMFunc
from .moe_cte_torch import _moe_cte_torch_ref_impl
from .moe_cte_utils import SkipMode


def blockwise_mm_baseline_shard_intermediate_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    block_size: int,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    gate_up_hidden_scale: Optional[torch.Tensor] = None,
    down_hidden_scale: Optional[torch.Tensor] = None,
    is_block_quant: bool = False,
    is_per_tensor: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.PRE_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
) -> dict:
    """Torch reference for blockwise_mm_baseline_shard_intermediate."""
    return _moe_cte_torch_ref_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=BWMMFunc.SHARD_ON_INTERMEDIATE,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        gate_up_hidden_scale=gate_up_hidden_scale,
        down_hidden_scale=down_hidden_scale,
        is_block_quant=is_block_quant,
        is_per_tensor=is_per_tensor,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        accumulation_dtype=accumulation_dtype,
        skip_gate_proj=skip_gate_proj,
    )


def blockwise_mm_baseline_shard_intermediate_hybrid_torch_ref(
    conditions: torch.Tensor,
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    block_size: int,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    num_static_block: Optional[int] = None,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    gate_up_activations_T: Optional[torch.Tensor] = None,
    down_activations: Optional[torch.Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.PRE_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    accumulation_dtype=None,
) -> dict:
    """Torch reference for blockwise_mm_baseline_shard_intermediate_hybrid."""
    return _moe_cte_torch_ref_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=BWMMFunc.SHARD_ON_INTERMEDIATE_HW,
        conditions=conditions,
        num_static_block=num_static_block,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        accumulation_dtype=accumulation_dtype,
    )


def blockwise_mm_shard_intermediate_dropping_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    block_size: int,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.PRE_SCALE,
    expert_affinity_multiply_on_I: bool = False,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    accumulation_dtype=None,
) -> dict:
    """Torch reference for blockwise_mm_shard_intermediate_dropping."""
    return _moe_cte_torch_ref_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        accumulation_dtype=accumulation_dtype,
    )
