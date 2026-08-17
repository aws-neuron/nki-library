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

"""PyTorch reference for blockwise_mm_shard_intermediate_mx and _hybrid kernels."""

from typing import Any, Optional

import nki.language as nl
import torch

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from .bwmm_mx_torch_common import bwmm_mx_blockwise_loop
from .moe_cte_utils import SkipMode


def blockwise_mm_shard_intermediate_mx_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    block_size: int = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype: Any = nl.bfloat16,
    weight_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.PRE_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    use_block128_scales: bool = False,
) -> dict:
    """PyTorch reference for blockwise_mm_shard_intermediate_mx. Signature matches kernel.

    ``use_block128_scales`` is a kernel-side scale-layout flag with no effect on the
    reference math: the test wrapper expands block-128 scales to the native layout
    before this ref runs, so the parameter is accepted (for signature parity) and ignored.
    """
    return bwmm_mx_blockwise_loop(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        block_size=block_size,
        activation_function=activation_function,
        skip_dma=skip_dma,
        weight_dtype=weight_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        conditions=None,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        separate_outputs=False,
    )


def blockwise_mm_shard_intermediate_mx_hybrid_torch_ref(
    conditions: torch.Tensor,
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    block_size: int = None,
    num_static_block: Optional[int] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype: Any = nl.bfloat16,
    weight_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    use_block128_scales: bool = False,
) -> dict:
    """PyTorch reference for blockwise_mm_shard_intermediate_mx_hybrid. Signature matches kernel.

    ``use_block128_scales`` is a kernel-side scale-layout flag with no effect on the
    reference math: the test wrapper expands block-128 scales to the native layout
    before this ref runs, so the parameter is accepted (for signature parity) and ignored.
    """
    return bwmm_mx_blockwise_loop(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        block_size=block_size,
        activation_function=activation_function,
        skip_dma=skip_dma,
        weight_dtype=weight_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        conditions=conditions,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        separate_outputs=False,
    )
