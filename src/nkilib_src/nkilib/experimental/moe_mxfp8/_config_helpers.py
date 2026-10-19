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

"""Shared keys and data-free config helpers for MXFP8 MoE and MLP."""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Optional

import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ..mxfp_utils.mxfp8_utils.common_dataclasses import QuantScheme, TensorOrientation
from .config_helpers import ExecutionMode


class WeightOrientation(Enum):
    """MXFP8 model weight layout shared by forward and backward."""

    BWD_NATURAL = "backward_natural"
    FWD_NATURAL = "forward_natural"


@dataclass(frozen=True)
class MoEInputShape(nl.NKIObject):
    """MLP weight shapes omit the singleton expert axis."""

    hidden_states_shape: tuple
    gate_up_weight_shape: tuple
    down_weight_shape: tuple
    block_size: int = 4096
    routing_shapes: Optional[tuple] = None
    weight_scale_shapes: Optional[tuple] = None
    num_valid_tokens_shape: Optional[tuple] = None


@dataclass(frozen=True)
class MoEConfigKey(nl.NKIObject):
    input_shape: MoEInputShape
    weight_dtype: tuple
    weight_orientation: tuple
    quant_scheme: QuantScheme
    dma_transpose_enabled: bool
    execution_mode: ExecutionMode


def make_pair(value):
    """Normalize a shared value or a gate/up and down pair."""
    if isinstance(value, tuple):
        kernel_assert(len(value) == 2, "Expected a gate/up and down pair")
        return value
    return value, value


def copy_config(value):
    """Copy data-free config records, including nested phase configs."""
    if is_dataclass(value):
        values = {}
        for field in fields(value):
            if field.init:
                values[field.name] = copy_config(getattr(value, field.name))
        return type(value)(**values)
    return value


def input_shape_from_tensors(
    hidden_states,
    gate_up_weight,
    down_weight,
    block_size,
    execution_mode,
    token_position_to_id=None,
    block_to_expert=None,
    gate_up_scales=None,
    down_scales=None,
    num_valid_tokens=None,
):
    mlp = execution_mode == ExecutionMode.MLP
    gu = gate_up_weight.shape[1:] if mlp else gate_up_weight.shape
    down = down_weight.shape[1:] if mlp else down_weight.shape
    gu_scales = None if gate_up_scales == None else (gate_up_scales.shape[1:] if mlp else gate_up_scales.shape)
    dn_scales = None if down_scales == None else (down_scales.shape[1:] if mlp else down_scales.shape)
    routing = None
    if execution_mode == ExecutionMode.ROUTED_MOE:
        routing = (token_position_to_id.shape, block_to_expert.shape)
    return MoEInputShape(
        hidden_states.shape,
        gu,
        down,
        block_size,
        routing,
        None if gu_scales == None and dn_scales == None else (gu_scales, dn_scales),
        None if num_valid_tokens == None else num_valid_tokens.shape,
    )


def make_moe_config_key(input_shape, weight_dtype, config, backward=False):
    """Use tensor facts and the caller's config before default resolution."""
    orientations = []
    packed = input_shape.weight_scale_shapes != None
    default_orientation = TensorOrientation.F_BY_K
    if not backward and (packed or config.execution_mode == ExecutionMode.MLP):
        default_orientation = TensorOrientation.K_BY_F
    for descriptor in (config.gate_up_weight_td, config.down_weight_td):
        # Packed tensors have a fixed physical layout, independent of descriptor defaults.
        orientation = default_orientation if packed or descriptor == None else descriptor.orientation
        backward_natural = orientation == (TensorOrientation.F_BY_K if backward else TensorOrientation.K_BY_F)
        orientations.append(WeightOrientation.BWD_NATURAL if backward_natural else WeightOrientation.FWD_NATURAL)
    if backward:
        scheme = QuantScheme._1x32
        for phase in (config.phase1_config, config.phase2_config, config.phase3_config, config.phase4_config):
            if phase != None:
                scheme = phase.quant_scheme
                break
    else:
        scheme = QuantScheme._1x32
        if not config.use_1x32_pe_swizzle:
            scheme = QuantScheme.WRAPX
        if input_shape.weight_scale_shapes != None:
            scheme = QuantScheme._1x32 if config.execution_mode.is_dense else QuantScheme.WRAPX
            for descriptor in (config.gate_up_weight_td, config.down_weight_td):
                if descriptor != None:
                    scheme = descriptor.quant_scheme
                    break
    return MoEConfigKey(
        input_shape,
        make_pair(weight_dtype),
        tuple(orientations),
        scheme,
        config.dma_transpose_enabled,
        config.execution_mode,
    )
