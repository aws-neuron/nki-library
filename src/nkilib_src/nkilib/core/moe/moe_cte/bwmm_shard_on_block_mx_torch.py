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

"""PyTorch reference for bwmm_shard_on_block_mx kernel."""

from typing import Any, Optional

import nki.language as nl
import torch

from ...mlp.mlp_parameters import MLPQuantizationParameters
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from .bwmm_mx_torch_common import bwmm_mx_blockwise_loop
from .moe_cte_utils import SkipMode


def bwmm_shard_on_block_mx_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    conditions: Optional[torch.Tensor] = None,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    ep_rank: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = 55,
    top_k: int = 1,
    ep_degree: int = 1,
    gate_up_activations_T: Optional[torch.Tensor] = None,
    down_activations: Optional[torch.Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = nl.bfloat16,
    weight_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    use_packed_scales: bool = False,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_up_in_scale: Optional[torch.Tensor] = None,
    down_in_scale: Optional[torch.Tensor] = None,
) -> dict:
    """PyTorch reference for bwmm_shard_on_block_mx. Signature matches kernel exactly."""
    # ep_rank only affects the packed-affinity ROW layout (the kernel derives its gather offset
    # ep_rank * E_local from it); the reference receives expert_affinities_masked already sliced to this
    # rank's local experts, so it is a no-op here. Present to satisfy the kernel<->ref signature contract.
    _ = ep_rank
    quant_params = None
    if quantization_type == QuantizationType.STATIC_MX:
        # STATIC_MX reuses gate_up_proj_scale / down_proj_scale to carry the per-expert
        # weight scales (gate/up packed [E, 2, 1], down [E, 1]). Split the packed gate/up
        # tensor into separate [E, 1] gate and up views (idx 0 = gate, idx 1 = up).
        quant_params = MLPQuantizationParameters(
            quantization_type=quantization_type,
            gate_w_scale=gate_up_proj_scale[:, 0],
            up_w_scale=gate_up_proj_scale[:, 1],
            down_w_scale=down_proj_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            clipping_bound=0.0,
        )
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
        separate_outputs=is_tensor_update_accumulating,
        quant_params=quant_params,
    )
