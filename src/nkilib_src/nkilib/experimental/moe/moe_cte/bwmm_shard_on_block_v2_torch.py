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
"""PyTorch reference implementations for bwmm_shard_on_block_v2 kernels.

Provides dispatch-ready torch_refs for:
- bwmm_shard_on_block
- bwmm_shard_on_block_hybrid

Each function matches its kernel's signature exactly, then delegates to
_moe_cte_torch_ref_impl with the appropriate BWMMFunc variant.
"""

from typing import Any, Optional

import torch

from ....core.moe.moe_cte.bwmm_func import BWMMFunc
from ....core.moe.moe_cte.moe_cte_torch import _moe_cte_torch_ref_impl
from ....core.moe.moe_cte.moe_cte_utils import BlockShardStrategy, SkipMode
from ....core.utils.common_types import ActFnType, ExpertAffinityScaleMode


def bwmm_shard_on_block_torch_ref(
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
    down_activations: Optional[torch.Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    n_block_per_iter: int = 1,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    sbm=None,
    num_static_block: Optional[int] = None,
    total_n_blocks: Optional[int] = None,
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    non_overlapping_shards: bool = False,
) -> dict:
    """Torch reference for bwmm_shard_on_block (v2)."""
    # Parameters consumed only by the NKI kernel's allocation/scheduling logic
    _ = sbm, total_n_blocks, non_overlapping_shards, block_sharding_strategy

    return _moe_cte_torch_ref_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=BWMMFunc.SHARD_ON_BLOCK_V2,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        down_activations=down_activations,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        n_block_per_iter=n_block_per_iter,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        num_static_block=num_static_block,
        down_bias_tp_degree=down_bias_tp_degree,
        down_bias_tp_rank=down_bias_tp_rank,
    )


def bwmm_shard_on_block_hybrid_torch_ref(
    conditions: torch.Tensor,
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
    down_activations: Optional[torch.Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    n_block_per_iter: int = 1,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    non_overlapping_shards: bool = False,
) -> dict:
    """Torch reference for bwmm_shard_on_block_hybrid."""
    # Parameters consumed only by the NKI kernel's scheduling logic
    _ = non_overlapping_shards, block_sharding_strategy

    return _moe_cte_torch_ref_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=BWMMFunc.SHARD_ON_BLOCK_HW,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        down_activations=down_activations,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        n_block_per_iter=n_block_per_iter,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        down_bias_tp_degree=down_bias_tp_degree,
        down_bias_tp_rank=down_bias_tp_rank,
    )
