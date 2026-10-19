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

"""Static MXFP8 SwiGLU MLP backward entry point."""

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ...moe_mxfp8._config_helpers import copy_config
from ...moe_mxfp8.bwd.blockwise_mm_backward_mxfp8 import blockwise_mm_bwd_mxfp8
from ...moe_mxfp8.bwd.config import MXFP8MOEBwdConfig
from ...moe_mxfp8.config_helpers import ExecutionMode
from .._common_utils import _get_mlp_block_size


def mlp_backward_mxfp8_nki(
    hidden_states: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    output_hidden_states_grad: nl.NkiTensor,
    gate_up_proj_act_checkpoint_T: nl.NkiTensor,
    config: MXFP8MOEBwdConfig = None,
    gate_up_weight_scales: nl.NkiTensor = None,
    down_weight_scales: nl.NkiTensor = None,
) -> tuple:
    """Return gradients for ``hidden``, gate/up weight, and down weight.

    BF16 weights use ``[H, 2, I]`` and ``[I, H]``. With scales, packed
    MXFP8 weights use backward layouts ``[2*I/4, H]`` and ``[H/4, I]``.
    Forward and backward use two blocks of T/2 tokens. The checkpoint
    is the forward output ``[2, 2, T/2, I]``. The wrapper derives the
    block size from T before calling the shared MoE kernel. T must be a
    positive multiple of 256.
    """
    config = MXFP8MOEBwdConfig(execution_mode=ExecutionMode.MLP) if config is None else copy_config(config)
    kernel_assert(config.execution_mode.is_dense, "MLP backward requires contiguous single-expert execution")
    config.execution_mode = ExecutionMode.MLP
    block_size = _get_mlp_block_size(hidden_states.shape[0])

    H = hidden_states.shape[1]
    I = gate_up_proj_act_checkpoint_T.shape[3]
    if gate_up_weight_scales is None:
        kernel_assert(
            gate_up_proj_weight.shape == (H, 2, I),
            f"BF16 gate_up_proj_weight must have shape [H={H}, 2, I={I}]",
        )
        gate_up_weight = gate_up_proj_weight.reshape((1, H, 2, I))
    else:
        kernel_assert(
            gate_up_proj_weight.shape == (2 * I // 4, H),
            f"MXFP8 gate_up_proj_weight must have shape [2*I/4={2 * I // 4}, H={H}]",
        )
        gate_up_weight = gate_up_proj_weight.reshape((1, 2 * I // 4, H))
        gate_up_weight_scales = gate_up_weight_scales.reshape(
            (1, gate_up_weight_scales.shape[0], gate_up_weight_scales.shape[1])
        )
    if down_weight_scales is None:
        kernel_assert(
            down_proj_weight.shape == (I, H),
            f"BF16 down_proj_weight must have shape [I={I}, H={H}]",
        )
        down_weight = down_proj_weight.reshape((1, I, H))
    else:
        kernel_assert(
            down_proj_weight.shape == (H // 4, I),
            f"MXFP8 down_proj_weight must have shape [H/4={H // 4}, I={I}]",
        )
        down_weight = down_proj_weight.reshape((1, H // 4, I))
        down_weight_scales = down_weight_scales.reshape((1, down_weight_scales.shape[0], down_weight_scales.shape[1]))

    hidden_grad, gate_up_grad, down_grad = blockwise_mm_bwd_mxfp8(
        hidden_states=hidden_states,
        expert_affinities_masked=None,
        gate_up_proj_weight=gate_up_weight,
        down_proj_weight=down_weight,
        token_position_to_id=None,
        block_to_expert=None,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        config=config,
        gate_up_weight_scales=gate_up_weight_scales,
        down_weight_scales=down_weight_scales,
    )
    return hidden_grad, gate_up_grad.reshape((H, 2, I)), down_grad.reshape((I, H))
