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

"""PyTorch reference implementation for 3D average pooling."""

import torch
import torch.nn.functional as F


def avg_pooling_3d_torch_ref(
    src_tensor: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    data_format="NCDHW",
    output_format=None,
) -> torch.Tensor:
    """Reference 3D average pool matching torch.nn.AvgPool3d.

    Uses count_include_pad=True: padded positions contribute zero and the divisor is
    the full window product, the convention the kernel folds into its finalize scale.

    Args:
        src_tensor: 5-D tensor.  (B, C, D, H, W) when data_format="NCDHW", or
            (B, D, H, W, C) when data_format="NDHWC".  Pooling is applied
            independently per (batch, channel) over the (D, H, W) axes.
        kernel_size: int or (kD, kH, kW) pooling window.
        stride: int or (sD, sH, sW); defaults to kernel_size.
        padding: int or (pD, pH, pW) implicit zero padding.
        data_format: input layout, "NCDHW" (default) or "NDHWC".
        output_format: output layout, "NCDHW" or "NDHWC"; defaults to data_format.

    Returns:
        Pooled tensor of the input dtype, laid out in output_format.
    """
    if stride is None:
        stride = kernel_size
    if output_format is None:
        output_format = data_format

    # Bring channels-last to channels-first so F.avg_pool3d acts on (N, C, D, H, W)
    x = src_tensor
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)

    y = F.avg_pool3d(
        x.to(torch.float32),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=True,
    )

    if output_format == "NDHWC":
        y = y.permute(0, 2, 3, 4, 1)
    return y.contiguous().to(src_tensor.dtype)
