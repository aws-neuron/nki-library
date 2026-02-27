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

"""Test utilities for router top-K tests."""

from test.integration.nkilib.utils.dtype_helper import dt

import torch


def router_topk_tensor_gen(name: str, shape, dtype):
    """
    Generate test tensor with values from [-0.1, 0.1] for router top-K testing.

    Args:
        name (str): Tensor name (unused, for compatibility)
        shape: Shape of the tensor to generate
        dtype: Data type for the tensor

    Returns:
        numpy.ndarray: Generated tensor with uniform random values in [-0.1, 0.1]

    Notes:
        - Uses thread-safe Generator for reproducibility
        - Range chosen because x.T @ w may go into sigmoid activation
    """
    generator = torch.Generator()
    generator.manual_seed(0)
    tensor = torch.empty(shape, dtype=torch.float32).uniform_(-0.1, 0.1, generator=generator)
    return dt.static_cast(tensor.numpy(), dtype)
