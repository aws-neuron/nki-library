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

from typing import Optional

import torch

from ...moe.bwd.blockwise_mm_backward_torch import blockwise_mm_bwd_torch_ref
from ...moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    SkipMode,
)
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from .config import MXFP8MOEBwdConfig


def blockwise_mm_bwd_mxfp8_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    output_hidden_states_grad: torch.Tensor,
    block_size: int = 4096,
    gate_up_proj_act_checkpoint_T: torch.Tensor = None,
    gate_act_checkpoint_T: torch.Tensor = None,
    intermediate_checkpoint_T: torch.Tensor = None,
    scaled_intermediate_checkpoint_T: torch.Tensor = None,
    down_proj_act_checkpoint=None,
    config: Optional[MXFP8MOEBwdConfig] = None,
    gate_up_weight_scales=None,
    down_weight_scales=None,
    num_valid_tokens=None,
) -> dict:
    """PyTorch reference for ``blockwise_mm_bwd_mxfp8``.

    Thin wrapper around the BF16 MoE backward golden. The parameter set matches the
    kernel entry exactly; only ``config``'s ``clamp_limits`` / ``bias`` /
    ``accumulate_hidden_states_grad`` / ``single_expert_dense`` fields and the tensor
    inputs affect the result. The remaining hardware/quantization knobs do not change
    the reference math.
    """
    if config is None:
        config = MXFP8MOEBwdConfig()
    T = hidden_states.shape[0]
    gate_up_orientation = (
        config.gate_up_weight_td.orientation if config.gate_up_weight_td is not None else TensorOrientation.F_BY_K
    )
    down_orientation = (
        config.down_weight_td.orientation if config.down_weight_td is not None else TensorOrientation.F_BY_K
    )
    if gate_up_weight_scales is None and gate_up_orientation == TensorOrientation.K_BY_F:
        gate_up_proj_weight = gate_up_proj_weight.permute(0, 3, 2, 1).contiguous()
    if down_weight_scales is None and down_orientation == TensorOrientation.K_BY_F:
        down_proj_weight = down_proj_weight.permute(0, 2, 1).contiguous()
    # The public checkpoint contract is token-major [N, 2, B, I_TP], while the
    # shared BF16 golden consumes [N, 2, I_TP, B].
    gate_up_proj_act_checkpoint_T = gate_up_proj_act_checkpoint_T.permute(0, 1, 3, 2).contiguous()
    if config.execution_mode.is_dense:
        # The kernel takes single-entry dummy routing tensors on the dense path and
        # derives every block from the contiguous token order. The shared BF16 golden
        # is routed-only, so materialize the equivalent routing here: the identity
        # token map over N*B == T positions, all blocks on expert 0.
        token_position_to_id = torch.arange(T, dtype=torch.int32)
        block_to_expert_1d = torch.zeros(T // block_size, dtype=torch.int32)
    else:
        block_to_expert_1d = block_to_expert.reshape(-1)
    # Accept-and-ignore: the dynamic block loop only changes WHICH blocks the kernel
    # computes, not the math. Padding tokens carry zero affinity, so the blocks the
    # kernel skips contribute nothing to the weight gradients, and the reference
    # matches there. Their hidden-states-grad rows are undefined on device, so the
    # comparator comes in masked to num_valid_tokens.
    _ignored = num_valid_tokens
    del _ignored
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
        clamp_limits=config.clamp_limits,
        bias=config.bias,
        is_tensor_update_accumulating=config.accumulate_hidden_states_grad,
    )
