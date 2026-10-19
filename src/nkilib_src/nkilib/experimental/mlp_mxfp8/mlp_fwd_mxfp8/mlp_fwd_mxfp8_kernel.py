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

"""Static MXFP8 SwiGLU MLP forward entry point."""

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ...moe_mxfp8._config_helpers import copy_config
from ...moe_mxfp8.config_helpers import ExecutionMode
from ...moe_mxfp8.fwd.blockwise_mm_forward_mxfp8 import blockwise_mm_fwd_mxfp8
from ...moe_mxfp8.fwd.config import MXFP8MOEFwdConfig
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from .._common_utils import _get_mlp_block_size


def mlp_forward_mxfp8_nki(
    hidden_states: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    config: MXFP8MOEFwdConfig = None,
    gate_up_weight_scales: nl.NkiTensor = None,
    down_weight_scales: nl.NkiTensor = None,
) -> tuple:
    """Compute a SwiGLU MLP with one contiguous token block per core.

    BF16 weights default to the shared forward/backward layout ``[H, 2, I]``
    and ``[I, H]``. Setting both weight descriptors to F_BY_K accepts
    ``[I, 2, H]`` and ``[H, I]`` for DMA transpose loading. With scales,
    packed MXFP8 weights use forward layouts
    ``[2*H/4, I]`` and ``[I/4, H]``. The returned checkpoint has shape
    ``[2, 2, T/2, I]`` and feeds
    :func:`mlp_backward_mxfp8_nki`.

    The computation is ``(SiLU(hidden @ gate) * (hidden @ up)) @ down``.
    No routing or expert affinities are used. Forward always computes two
    blocks of T/2 tokens, including the checkpoint. The wrapper derives
    the block size from T before calling the shared MoE kernel.
    T must be a positive multiple of 256 so each core's block is tile-aligned.
    """
    config = MXFP8MOEFwdConfig(execution_mode=ExecutionMode.MLP) if config is None else copy_config(config)
    kernel_assert(config.execution_mode.is_dense, "MLP forward requires contiguous single-expert execution")
    config.execution_mode = ExecutionMode.MLP
    block_size = _get_mlp_block_size(hidden_states.shape[0])

    H = hidden_states.shape[1]
    if down_weight_scales is not None:
        I = down_proj_weight.shape[0] * 4
    elif config.down_weight_td is not None and config.down_weight_td.orientation == TensorOrientation.F_BY_K:
        I = down_proj_weight.shape[1]
    else:
        I = down_proj_weight.shape[0]
    if gate_up_weight_scales is None:
        gate_up_orientation = (
            TensorOrientation.K_BY_F if config.gate_up_weight_td is None else config.gate_up_weight_td.orientation
        )
        gate_up_f_by_k = gate_up_orientation == TensorOrientation.F_BY_K
        gate_up_shape = (I, 2, H) if gate_up_f_by_k else (H, 2, I)
        kernel_assert(
            gate_up_proj_weight.shape == gate_up_shape,
            f"BF16 gate_up_proj_weight must have shape {gate_up_shape}",
        )
        gate_up_weight = gate_up_proj_weight.reshape((1, *gate_up_shape))
        kernel_assert(
            gate_up_orientation in (TensorOrientation.K_BY_F, TensorOrientation.F_BY_K),
            "MLP BF16 gate/up orientation must be K_BY_F or F_BY_K",
        )
    else:
        kernel_assert(
            gate_up_proj_weight.shape == (2 * H // 4, I),
            f"MXFP8 gate_up_proj_weight must have shape [2*H/4={2 * H // 4}, I={I}]",
        )
        gate_up_weight = gate_up_proj_weight.reshape((1, 2 * H // 4, I))
        gate_up_weight_scales = gate_up_weight_scales.reshape(
            (1, gate_up_weight_scales.shape[0], gate_up_weight_scales.shape[1])
        )

    if down_weight_scales is None:
        down_orientation = (
            TensorOrientation.K_BY_F if config.down_weight_td is None else config.down_weight_td.orientation
        )
        down_f_by_k = down_orientation == TensorOrientation.F_BY_K
        down_shape = (H, I) if down_f_by_k else (I, H)
        kernel_assert(
            down_proj_weight.shape == down_shape,
            f"BF16 down_proj_weight must have shape {down_shape}",
        )
        down_weight = down_proj_weight.reshape((1, *down_shape))
        kernel_assert(
            down_orientation in (TensorOrientation.K_BY_F, TensorOrientation.F_BY_K),
            "MLP BF16 down orientation must be K_BY_F or F_BY_K",
        )
    else:
        kernel_assert(
            down_proj_weight.shape == (I // 4, H),
            f"MXFP8 down_proj_weight must have shape [I/4={I // 4}, H={H}]",
        )
        down_weight = down_proj_weight.reshape((1, I // 4, H))
        down_weight_scales = down_weight_scales.reshape((1, down_weight_scales.shape[0], down_weight_scales.shape[1]))

    return blockwise_mm_fwd_mxfp8(
        hidden_states=hidden_states,
        expert_affinities_masked=None,
        gate_up_proj_weight=gate_up_weight,
        down_proj_weight=down_weight,
        token_position_to_id=None,
        block_to_expert=None,
        block_size=block_size,
        config=config,
        gate_up_weight_scales=gate_up_weight_scales,
        down_weight_scales=down_weight_scales,
    )
