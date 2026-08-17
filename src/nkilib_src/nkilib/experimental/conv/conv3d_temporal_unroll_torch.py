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

"""PyTorch reference for conv3d_temporal_unroll (delegates to conv3d_torch_ref)."""

from typing import Optional

import torch

from ...core.utils.common_types import ActFnType
from .conv3d_torch import conv3d_torch_ref


def conv3d_temporal_unroll_torch_ref(
    x_in: torch.Tensor,
    filters: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
) -> dict[str, torch.Tensor]:
    """PyTorch reference for conv3d_temporal_unroll.

    Signature mirrors conv3d_temporal_unroll (which does not support batchnorm
    fusion) and delegates to the shared conv3d reference.
    """
    return conv3d_torch_ref(
        x_in=x_in,
        filters=filters,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        activation_fn=activation_fn,
        lnc_shard=lnc_shard,
    )
