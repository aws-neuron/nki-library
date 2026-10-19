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

"""Public API for the DeepSeek-V3.2 specialized blockwise MoE context-encoding matmul kernel."""

from ._utils import (
    ActivationQuantMode,
    BlockShardStrategy,
    Configs,
    InputTensors,
    QuantConfig,
    ScaleFormat,
    SkipMode,
    WeightQuantMode,
)
from .bwmm_block_mx import bwmm_block_mx
from .bwmm_block_mx_torch import bwmm_block_mx_torch_ref

__all__ = [
    "ActivationQuantMode",
    "BlockShardStrategy",
    "Configs",
    "InputTensors",
    "QuantConfig",
    "ScaleFormat",
    "SkipMode",
    "WeightQuantMode",
    "bwmm_block_mx",
    "bwmm_block_mx_torch_ref",
]
