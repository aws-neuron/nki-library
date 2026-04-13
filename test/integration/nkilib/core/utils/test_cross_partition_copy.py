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

"""Integration tests for cross_partition_copy utility.

Tests the cross_partition_copy function which copies data between SBUF tensors
at arbitrary partition offsets. Validates all three internal code paths:
  - Case 1: dst aligned to 32-partition boundary → direct tensor_copy
  - Case 2: copy stays within single quadrant → _shuffle_within_quadrant
  - Case 3: cross-quadrant copy → shuffle first portion + remaining chunks
"""

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.cross_partition_copy import cross_partition_copy

# =============================================================================
# Test Kernel
# =============================================================================


@nki.jit
def kernel_cross_partition_copy(
    input_data,
    output,
    src_start_partition: int,
    dst_start_partition: int,
    num_partitions_to_copy: int,
    free_dim_size: int,
    total_partitions: int,
):
    """Test kernel that exercises cross_partition_copy and returns the result.

    Loads input_data from HBM into an SBUF src tensor, initializes an SBUF dst
    tensor to zeros, calls cross_partition_copy, and stores dst back to HBM.
    """
    # Load input from HBM to SBUF
    src = nl.ndarray((total_partitions, free_dim_size), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=input_data)

    # Initialize dst in SBUF to zeros
    dst = nl.ndarray((total_partitions, free_dim_size), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.memset(dst=dst, value=0.0)

    # Call the utility under test
    cross_partition_copy(
        src=src,
        dst=dst,
        src_start_partition=src_start_partition,
        dst_start_partition=dst_start_partition,
        num_partitions_to_copy=num_partitions_to_copy,
        free_dim_size=free_dim_size,
    )

    # Store result back to HBM
    nisa.dma_copy(dst=output, src=dst)
    return output


# =============================================================================
# Torch Reference
# =============================================================================


def cross_partition_copy_torch_ref(
    input_data,
    output,
    src_start_partition,
    dst_start_partition,
    num_partitions_to_copy,
    free_dim_size,
    total_partitions,
):
    """Torch reference: simple slice copy from src to dst."""
    result = torch.zeros_like(input_data)
    result[dst_start_partition : dst_start_partition + num_partitions_to_copy, :free_dim_size] = input_data[
        src_start_partition : src_start_partition + num_partitions_to_copy, :free_dim_size
    ]
    return {"output": result}


# =============================================================================
# Tests
# =============================================================================


@final
@pytest_test_metadata(name="CrossPartitionCopy", pytest_marks=["cross_partition_copy"])
class TestCrossPartitionCopy:
    """Tests for cross_partition_copy utility with full data validation."""

    _ABBREVS = {
        "src_start": "ss",
        "dst_start": "ds",
        "num_partitions": "np",
        "free_dim_size": "fd",
        "total_partitions": "tp",
    }

    @pytest.mark.fast
    @pytest_parametrize(
        "src_start,dst_start,num_partitions,free_dim_size,total_partitions",
        [
            # --- Both aligned (Case 1: direct tensor_copy) ---
            (0, 0, 32, 8, 128),
            (0, 64, 32, 8, 128),
            (0, 0, 64, 8, 64),
            # --- Aligned src, unaligned dst within quadrant (Case 2: shuffle) ---
            (0, 5, 10, 8, 128),
            (0, 16, 8, 8, 64),
            # --- Aligned src, unaligned dst cross-quadrant (Case 3) ---
            (0, 28, 20, 8, 128),
            (0, 60, 10, 8, 96),
            (0, 1, 63, 8, 64),
            # --- Unaligned src + aligned dst (src normalization + Case 1) ---
            (5, 0, 16, 8, 128),
            (33, 64, 16, 8, 128),
            (17, 0, 15, 8, 32),
            # --- Unaligned src + unaligned dst within quadrant ---
            (5, 10, 8, 8, 128),
            (3, 2, 4, 8, 32),
            # --- Unaligned src + unaligned dst cross-quadrant ---
            (5, 28, 20, 8, 128),
            (10, 55, 30, 8, 96),
            # --- Unaligned src spanning two src quadrants ---
            (28, 64, 32, 8, 128),
            (25, 5, 20, 8, 64),
            # --- Various free_dim sizes ---
            (0, 5, 10, 2, 128),
            (5, 0, 16, 32, 128),
            (0, 28, 20, 128, 128),
        ],
        abbrevs=_ABBREVS,
    )
    def test_cross_partition_copy(
        self, test_manager, platform_target, src_start, dst_start, num_partitions, free_dim_size, total_partitions
    ):
        """Test cross_partition_copy with various src/dst offsets and sizes."""

        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(total_partitions, free_dim_size).astype(np.float32)
            return {
                "input_data": input_data,
                "output.must_alias_input": np.zeros_like(input_data),
                "src_start_partition": src_start,
                "dst_start_partition": dst_start,
                "num_partitions_to_copy": num_partitions,
                "free_dim_size": free_dim_size,
                "total_partitions": total_partitions,
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((total_partitions, free_dim_size), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_cross_partition_copy,
            torch_ref=torch_ref_wrapper(cross_partition_copy_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
