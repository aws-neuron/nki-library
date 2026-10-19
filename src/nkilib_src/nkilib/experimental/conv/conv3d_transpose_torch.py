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

"""PyTorch reference implementation for the 3D transposed convolution kernel."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.utils.common_types import ActFnType


def _filters_to_co_ci_kdhw(filters: torch.Tensor, filter_shape: str, flip_filter_taps_in_kernel: bool) -> torch.Tensor:
    """
    Convert the kernel's filter tensor to the ConvTranspose3d weight layout
    (C_in, C_out, K_d, K_h, K_w), which expects taps in forward orientation.

    flip_filter_taps_in_kernel mirrors the kernel flag and indicates who flips the taps:
      - True:  the kernel reverses taps internally, so `filters` is passed in natural
               orientation here and needs no flip.
      - False: filters is already spatially flipped by the caller.

    The channel axes are then permuted to (C_in, C_out, K_d, K_h, K_w).
    """
    spatially_unflip = not flip_filter_taps_in_kernel
    if filter_shape == "KDHW_CI_CO":
        # (K_d, K_h, K_w, C_in, C_out) -> (C_in, C_out, K_d, K_h, K_w)
        w = filters.flip(dims=(0, 1, 2)) if spatially_unflip else filters
        return w.permute(3, 4, 0, 1, 2).contiguous()
    if filter_shape == "KDHW_CO_CI":
        # (K_d, K_h, K_w, C_out, C_in) -> (C_in, C_out, K_d, K_h, K_w)
        w = filters.flip(dims=(0, 1, 2)) if spatially_unflip else filters
        return w.permute(4, 3, 0, 1, 2).contiguous()
    raise ValueError(f"Unsupported filter_shape '{filter_shape}'")


def conv3d_transpose_torch_ref(
    x_in: torch.Tensor,
    filters: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    output_padding: tuple[int, int, int] = (0, 0, 0),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
    filter_shape: str = "KDHW_CI_CO",
    flip_filter_taps_in_kernel: bool = False,
    sbm=None,
    use_auto_allocation: bool = False,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of the 3D transposed convolution kernel.

    Uses torch.nn.ConvTranspose3d, optionally with output_padding.

    Args:
        x_in (torch.Tensor): Input tensor of shape [B, C_in, D, H, W]
        filters (torch.Tensor): Filter weights. Their spatial orientation depends on
            flip_filter_taps_in_kernel (see below). Shape depends on filter_shape:
                - "KDHW_CI_CO": [K_d, K_h, K_w, C_in, C_out]
                - "KDHW_CO_CI": [K_d, K_h, K_w, C_out, C_in]
        bias (Optional[torch.Tensor]): Optional bias of shape [C_out].
        stride, padding, dilation: Forward-conv hyperparameters.
        output_padding: Extra size added to the output spatial dims.
        activation_fn: Optional activation applied after ConvTranspose3d.
        lnc_shard: Does nothing. Will be deprecated next release.
        filter_shape: Storage layout for filters.
        flip_filter_taps_in_kernel: When True, the kernel reverses filter taps internally, so
        filters is supplied in forward orientation. When False, filters must already be spatially flipped by the caller.

    Returns:
        dict[str, torch.Tensor]: {"out": output tensor [B, C_out, D_out, H_out, W_out]}
    """
    w_ci_co_kdhw = _filters_to_co_ci_kdhw(filters, filter_shape, flip_filter_taps_in_kernel)
    C_in, C_out, K_d, K_h, K_w = w_ci_co_kdhw.shape
    pad_d_lo, pad_d_hi, pad_h_lo, pad_h_hi, pad_w_lo, pad_w_hi = padding
    if pad_d_lo != pad_d_hi or pad_h_lo != pad_h_hi or pad_w_lo != pad_w_hi:
        raise ValueError(f"conv3d_transpose_torch_ref only supports symmetric padding per dimension, got {padding}")
    conv_padding = (pad_d_lo, pad_h_lo, pad_w_lo)

    conv_t = nn.ConvTranspose3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=(K_d, K_h, K_w),
        stride=stride,
        padding=conv_padding,
        output_padding=output_padding,
        dilation=dilation,
        bias=(bias is not None),
    )

    with torch.no_grad():
        conv_t.weight.copy_(w_ci_co_kdhw)
        if bias is not None:
            conv_t.bias.copy_(bias)

    output = conv_t(x_in)

    if activation_fn is not None:
        if activation_fn == ActFnType.SiLU:
            output = F.silu(output)
        elif activation_fn == ActFnType.GELU:
            output = F.gelu(output)
        elif activation_fn == ActFnType.GELU_Tanh_Approx:
            output = F.gelu(output, approximate="tanh")
        elif activation_fn == ActFnType.Swish:
            output = F.silu(output)
        elif activation_fn == ActFnType.ReLU:
            output = F.relu(output)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    return {"out": output.detach()}
