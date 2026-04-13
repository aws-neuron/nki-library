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

"""Shared PyTorch utility functions for MoE CTE torch reference implementations."""

import torch
import torch.nn.functional as F

from ...utils.common_types import ActFnType


def torch_act_fn(x: torch.Tensor, act_fn: ActFnType) -> torch.Tensor:
    """Apply activation function to tensor.

    Args:
        x (torch.Tensor): Input tensor
        act_fn (ActFnType): Activation function type

    Returns:
        torch.Tensor: Activated tensor
    """
    if act_fn == ActFnType.SiLU:
        return F.silu(x)
    elif act_fn == ActFnType.GELU:
        return F.gelu(x)
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        return F.gelu(x, approximate="tanh")
    elif act_fn == ActFnType.Swish:
        return x * torch.sigmoid(1.702 * x)
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
