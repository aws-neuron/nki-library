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

"""PyTorch reference implementation for dynamic elementwise add kernel."""

import torch


def dynamic_elementwise_add_torch_ref(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference implementation of dynamic elementwise addition.

    This is a reference implementation for testing the NKI dynamic_elementwise_add kernel.
    It computes the same mathematical operation using PyTorch.

    Args:
        input_a (torch.Tensor): [M, H], First input tensor.
        input_b (torch.Tensor): [M, H], Second input tensor, same shape as input_a.

    Returns:
        torch.Tensor: [M, H], Elementwise sum of input_a and input_b.

    Notes:
        - The num_m_tiles parameter is not needed here since the torch reference
          operates on the full tensors without tiling.
    """
    return input_a + input_b
