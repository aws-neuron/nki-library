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

"""PyTorch reference implementations for RNG kernels."""

import torch

NUM_RNG_SEEDS = 6


def get_rng_state_gpsimd_torch_ref(tensor_state: torch.Tensor) -> dict:
    """
    PyTorch reference for get_rng_state_gpsimd.

    Returns a zero tensor of the same shape since RNG state retrieval
    is hardware-specific and cannot be replicated in PyTorch.

    Args:
        tensor_state (torch.Tensor): [1, NUM_RNG_SEEDS], dtype uint32, input state tensor.

    Returns:
        dict: {"output_0": zeros} placeholder output.
    """
    return {"output_0": torch.zeros_like(tensor_state)}


def set_rng_state_gpsimd_torch_ref(tensor_state: torch.Tensor) -> dict:
    """
    PyTorch reference for set_rng_state_gpsimd.

    Returns the input state viewed as int32 since the kernel echoes back
    the seeds that were set with a .view(int32).

    Args:
        tensor_state (torch.Tensor): [1, NUM_RNG_SEEDS], dtype uint32, input state tensor.

    Returns:
        dict: {"output_0": tensor_state} echoed back as int32.
    """
    return {"output_0": tensor_state.view(torch.int32)}


def generate_random_torch_ref(output: torch.Tensor, n_elements: int) -> dict:
    """
    PyTorch reference for generate_random.

    Generates random int32 values for shape validation. Values will differ
    from the NKI kernel output since different RNG engines are used.

    Args:
        output (torch.Tensor): [1, n_elements], dtype int32, reference output tensor.
        n_elements (int): Number of random int32 values to generate.

    Returns:
        dict: {"output_0": result} with n_elements random int32s.
    """
    result = torch.randint(low=-(2**31), high=2**31 - 1, size=(1, n_elements), dtype=torch.int32)
    return {"output_0": result}


def generate_random_fast_torch_ref(output: torch.Tensor, n_elements: int) -> dict:
    """
    PyTorch reference for generate_random_fast.

    Generates random int32 values for shape validation. Values will differ from the NKI
    kernel output since different RNG engines are used (and generate_random_fast uses 128
    interleaved GPSIMD streams), so this only validates shape/dtype and statistical
    uniformity, not the exact stream.

    Args:
        output (torch.Tensor): [1, n_elements], dtype int32, reference output tensor.
        n_elements (int): Number of random int32 values to generate.

    Returns:
        dict: {"output_0": result} with n_elements random int32s.
    """
    result = torch.randint(low=-(2**31), high=2**31 - 1, size=(1, n_elements), dtype=torch.int32)
    return {"output_0": result}
