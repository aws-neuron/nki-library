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

"""PyTorch reference implementation for 3D average pooling backward."""

import torch
import torch.nn.functional as F


def avg_pooling_3d_backward_torch_ref(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    data_format="NCDHW",
    output_format=None,
) -> torch.Tensor:
    """Reference backward of 3D average pool via torch autograd.

    Uses count_include_pad=True to match the forward kernel, so every output spreads its
    gradient over the full window product.

    Returns:
        grad_input in data_format layout, input dtype.
    """
    if stride is None:
        stride = kernel_size
    if output_format is None:
        output_format = data_format

    x = input_tensor
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    x = x.to(torch.float32).contiguous().requires_grad_(True)

    y = F.avg_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=True)

    go = grad_output.to(torch.float32)
    if output_format == "NDHWC":
        go = go.permute(0, 4, 1, 2, 3)
    y.backward(go.contiguous())

    gin = x.grad  # (B, C, D, H, W)
    if data_format == "NDHWC":
        gin = gin.permute(0, 2, 3, 4, 1)
    return gin.contiguous().to(input_tensor.dtype)
