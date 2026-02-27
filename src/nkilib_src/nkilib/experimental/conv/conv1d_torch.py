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

"""PyTorch reference implementation for 1D convolution kernel testing."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.utils.common_types import ActFnType


def conv1d_torch_ref(
    x_in: torch.Tensor,
    filters: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of 1D convolution kernel.

    Args:
        x_in (torch.Tensor): Input tensor of shape [B, C_in, L]
        filters (torch.Tensor): Filter weights of shape [K, C_in, C_out]
        bias (Optional[torch.Tensor]): Optional bias tensor of shape [C_out]
        stride (int): Stride for convolution. Default 1.
        padding (tuple[int, int]): Tuple of (left_pad, right_pad). Default (0, 0).
        dilation (int): Dilation factor for dilated convolution. Default 1.
        activation_fn (Optional[ActFnType]): Optional activation function type. Default None.
        lnc_shard (bool): Whether LNC sharding is enabled (unused in reference). Default False.

    Returns:
        dict[str, torch.Tensor]: Dictionary with key "out" containing output tensor of shape
            [B, C_out, L_out] where L_out = (L + pad_left + pad_right - dilation * (K - 1) - 1) // stride + 1
    """
    K, C_in, C_out = filters.shape
    pad_left, pad_right = padding

    # Handle asymmetric padding by manually padding the input
    if pad_left != pad_right:
        x_in = F.pad(x_in, (pad_left, pad_right), mode="constant", value=0)
        conv_padding = 0
    else:
        conv_padding = pad_left

    # Create nn.Conv1d module
    conv = nn.Conv1d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=K,
        stride=stride,
        padding=conv_padding,
        dilation=dilation,
        bias=(bias is not None),
    )

    # Set weights: transpose from [K, C_in, C_out] to [C_out, C_in, K]
    with torch.no_grad():
        conv.weight.copy_(filters.permute(2, 1, 0))
        if bias is not None:
            conv.bias.copy_(bias)

    # Perform convolution
    output = conv(x_in)

    # Apply activation function if specified
    if activation_fn is not None:
        if activation_fn == ActFnType.SiLU:
            output = F.silu(output)
        elif activation_fn == ActFnType.GELU:
            output = F.gelu(output)
        elif activation_fn == ActFnType.GELU_Tanh_Approx:
            output = F.gelu(output, approximate="tanh")
        elif activation_fn == ActFnType.Swish:
            output = F.silu(output)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    return {"out": output.detach()}
