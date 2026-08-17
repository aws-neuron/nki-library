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

"""PyTorch reference implementation of the blockwise MoE MXFP8 backward kernel.

Delegates to the BF16 MoE backward golden (``blockwise_mm_bwd_torch_ref``) with
fixed configuration: SiLU activation, AFFINITY_ON_I, skip_token=True, no clamping.
The math is identical — only quantization-aware arguments differ.
"""

import torch

from ...moe.bwd.blockwise_mm_backward_torch import blockwise_mm_bwd_torch_ref
from ...moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    SkipMode,
)
from ...mxfp_utils.mxfp8_utils.common_dataclasses import SwizzleMode


def blockwise_mm_bwd_mxfp8_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    output_hidden_states_grad: torch.Tensor,
    block_size: int,
    gate_up_proj_act_checkpoint_T: torch.Tensor = None,
    gate_act_checkpoint_T: torch.Tensor = None,
    intermediate_checkpoint_T: torch.Tensor = None,
    scaled_intermediate_checkpoint_T: torch.Tensor = None,
    down_proj_act_checkpoint=None,
    gate_up_weight_scales=None,
    gate_up_weight_is_swizzled: bool = False,
    down_weight_scales=None,
    down_weight_is_swizzled: bool = False,
    phase1_config=None,
    phase2_config=None,
    phase3_config=None,
    phase4_config=None,
    fp8_x4_dtype=None,
    spill_reload: bool = False,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
    shard_option=None,
    affinity_option=None,
    compute_dtype=None,
    skip_dma=None,
    skip_grad_initialization: bool = False,
    single_expert_dense: bool = False,
    fast_dma_transpose: bool = False,
    output_grad_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    down_weight_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    d_gate_up_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    gate_up_weight_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    d_gate_up_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    hidden_states_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    output_grad_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    scaled_intermediate_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    accumulate_hidden_states_grad: bool = True,
    clamp_limits=None,
    activation_type=None,
    bias: bool = False,
    phase3_transpose_mode=None,
    phase4_transpose_mode=None,
) -> dict:
    """PyTorch reference for ``blockwise_mm_bwd_mxfp8``.

    Thin wrapper around the BF16 MoE backward golden. Hardware/quantization
    arguments are accepted for signature compatibility but do not affect the result.
    """
    block_to_expert_1d = block_to_expert.reshape(-1)
    return blockwise_mm_bwd_torch_ref(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        down_proj_act_checkpoint=None,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert_1d,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        skip_dma=SkipMode(True, False),
        affinity_option=AffinityOption.AFFINITY_ON_I,
        activation_type=ActFnType.SiLU,
        clamp_limits=clamp_limits,
        bias=bias,
        is_tensor_update_accumulating=accumulate_hidden_states_grad,
    )
