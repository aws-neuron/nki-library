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

"""Tests for the ModularAllocator (modular_allocator.py).

Host-side tests cover the trivial address get/set accessors. The allocation
logic in ``alloc_sbuf_tensor`` / ``_allocate_recursive`` builds ``nl.ndarray``
tensors with explicit addresses and therefore needs a backend, so it is driven
by a small ``@nki.jit`` kernel through ``UnitTestFramework``. The kernel walks
every allocation shape:

  * single tensor        (block_dim None, no alignment)
  * modular 1D           (explicit num_free_tiles + alignment)
  * 1D with default tiles (num_free_tiles defaulted from block_dim)
  * nested 2D            (multi-depth recursion)
"""

from typing import final

import nki
import nki.isa as nisa
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.modular_allocator import ModularAllocator

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Host-side (pure-Python) tests
# =============================================================================


@pytest.mark.fast
def test_address_get_set():
    """get/set_current_address round-trips the internal pointer."""
    alloc = ModularAllocator(initial_address=10)
    assert alloc.get_current_address() == 10
    alloc.set_current_address(64)
    assert alloc.get_current_address() == 64


# =============================================================================
# Simulator test (allocation logic that builds nl.ndarray)
# =============================================================================


@nki.jit
def kernel_modular_alloc(input_data, output):
    """Exercise single / modular / nested modular allocation shapes.

    Data flows input -> single -> tiles[0] -> tiles2[0] -> grid[0][0] -> output,
    so the result is an identity copy of the input.
    """
    alloc = ModularAllocator(initial_address=0)

    # Single tensor: block_dim defaulted to None, no alignment.
    single = alloc.alloc_sbuf_tensor(shape=input_data.shape, dtype=input_data.dtype)
    nisa.dma_copy(dst=single, src=input_data)  # ty: ignore[invalid-argument-type]

    # Modular 1D: explicit num_free_tiles and alignment (circular buffer).
    tiles = alloc.alloc_sbuf_tensor(
        shape=input_data.shape,
        dtype=input_data.dtype,
        block_dim=[4],
        num_free_tiles=[2],
        align_to=32,
    )
    nisa.tensor_copy(dst=tiles[0], src=single)  # ty: ignore[not-subscriptable, invalid-argument-type]

    # 1D with num_free_tiles defaulted from block_dim.
    tiles2 = alloc.alloc_sbuf_tensor(
        shape=input_data.shape,
        dtype=input_data.dtype,
        block_dim=[3],
    )
    nisa.tensor_copy(dst=tiles2[0], src=tiles[0])  # ty: ignore[not-subscriptable]

    # Nested 2D: exercises multi-depth recursion.
    grid = alloc.alloc_sbuf_tensor(
        shape=input_data.shape,
        dtype=input_data.dtype,
        block_dim=[2, 2],
        num_free_tiles=[2, 2],
    )
    nisa.tensor_copy(dst=grid[0][0], src=tiles2[0])  # ty: ignore[not-subscriptable]

    nisa.dma_copy(dst=output, src=grid[0][0])  # ty: ignore[not-subscriptable]
    return output


def _identity_ref(input_data, output):
    import torch

    return {"output": torch.clone(input_data)}


@final
@pytest_test_metadata(name="ModularAllocator")
@pytest_marks(["modular_allocator"])
class TestModularAllocatorSimulation:
    """Simulator test running the modular allocation logic inside a kernel."""

    @pytest.mark.fast
    def test_single_modular_and_nested_allocation(self, test_manager, platform_target):
        """Single, modular 1D, defaulted-tiles, and nested 2D allocation."""

        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(128, 256).astype(np.float32)
            return {
                "input_data": input_data,
                "output.must_alias_input": np.zeros_like(input_data),
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((128, 256), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_modular_alloc,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
