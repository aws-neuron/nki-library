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

"""Integration tests for indexed_flatten kernel."""

from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.indexed_flatten import indexed_flatten


def indexed_flatten_reference(
    input_tensor: np.ndarray,
    f_len: int,
    output_len: int,
    row_offsets: np.ndarray,
    row_offsets_start: int = 0,
    padding_val: int = -1,
) -> np.ndarray:
    """
    NumPy reference implementation of indexed_flatten.

    For input_tensor of shape [E, T], reshapes to [E, T//f_len, f_len] and writes
    each row's data into output at the specified row_offsets (block offsets).

    Args:
        input_tensor: Input array of shape [E, T]
        f_len: Block size in free dimension
        output_len: Length of output array
        row_offsets: Array of block offsets for each row
        row_offsets_start: Starting index into row_offsets array
        padding_val: Value for unwritten positions

    Returns:
        Output array of shape [output_len,]
    """
    E, T = input_tensor.shape
    partitions_per_row = T // f_len

    output = np.full((output_len,), padding_val, dtype=input_tensor.dtype)
    output_blocks = output.reshape(output_len // f_len, f_len)
    input_reshaped = input_tensor.reshape(E, partitions_per_row, f_len)

    for row_idx in range(E):
        block_offset = row_offsets[row_offsets_start + row_idx]
        for partition_idx in range(partitions_per_row):
            out_block_idx = block_offset + partition_idx
            if out_block_idx < output_len // f_len:
                output_blocks[out_block_idx, :] = input_reshaped[row_idx, partition_idx, :]

    return output


def generate_kernel_inputs(
    T: int,
    E: int,
    f_len: int,
    row_offsets: list[int],
    output_len: int,
    padding_val: int,
    row_offsets_start: int = 0,
) -> dict[str, np.ndarray]:
    """
    Generate indexed_flatten kernel inputs from parameters.

    Args:
        T (int): Number of elements per row
        E (int): Number of rows
        f_len (int): Block size in free dimension
        row_offsets (list[int]): List of block offsets
        output_len (int): Length of output array
        padding_val (int): Value for unwritten positions
        row_offsets_start (int): Starting index into row_offsets array

    Returns:
        dict[str, np.ndarray]: Dictionary with input_tensor, row_offsets, and optionally row_offsets_start arrays
    """
    np.random.seed(42)
    input_tensor = np.arange(E * T, dtype=np.int32).reshape(E, T)
    row_offsets_arr = np.array(row_offsets, dtype=np.int32)
    result = {
        "input_tensor": input_tensor,
        "row_offsets": row_offsets_arr,
    }
    if row_offsets_start != 0:
        result["row_offsets_start"] = np.array([row_offsets_start], dtype=np.int32)
    return result


@pytest_test_metadata(
    name="IndexedFlatten",
    pytest_marks=["indexed_flatten"],
)
@final
class TestIndexedFlattenKernel:
    """Test class for indexed_flatten kernel."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "T,E,f_len",
        [
            # Basic cases
            (4096, 1, 128),
            (4096, 2, 128),
            (4096, 2, 256),
            (4096, 3, 128),
            # Medium E
            (10240, 16, 128),
            (10240, 17, 128),
            # Large T
            (65536, 4, 128),
            (65536, 4, 256),
            # Large E (up to 256)
            (4096, 64, 128),
            (4096, 128, 128),
            (4096, 256, 128),
        ],
    )
    def test_indexed_flatten_fast(
        self,
        test_manager: Orchestrator,
        T,
        E,
        f_len,
    ):
        """Fast compile-only tests with minimal coverage."""
        # Compute derived parameters
        partitions_per_row = T // f_len
        row_offsets = [partitions_per_row * idx for idx in range(E)]
        output_len = E * T + 1024  # Extra padding
        output_len = ((output_len + 127) // 128) * 128  # Align to P_MAX
        padding_val = -1

        kernel_input = generate_kernel_inputs(T, E, f_len, row_offsets, output_len, padding_val)
        input_tensor = kernel_input["input_tensor"]
        row_offsets_arr = kernel_input["row_offsets"]

        def golden_generator():
            return {
                "flattened_array": indexed_flatten_reference(
                    input_tensor, f_len, output_len, row_offsets_arr, 0, padding_val
                )
            }

        test_manager.execute(
            KernelArgs(
                kernel_func=indexed_flatten,
                compiler_input=CompilerArgs(logical_nc_config=2),
                kernel_input={
                    "input_tensor": input_tensor,
                    "f_len": f_len,
                    "output_len": output_len,
                    "row_offsets": row_offsets_arr,
                    "row_offsets_start": None,
                    "padding_val": padding_val,
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=golden_generator,
                        output_ndarray={"flattened_array": np.zeros(output_len, dtype=np.int32)},
                    ),
                    absolute_accuracy=0,
                    relative_accuracy=0,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "T,E,f_len,N,row_offsets_start",
        [
            # E=2, start at index 2 (read offsets[2:4])
            (4096, 2, 128, 4, 2),
            # E=3, start at index 1 (read offsets[1:4])
            (4096, 3, 128, 5, 1),
            # E=8, start at index 8 (read offsets[8:16])
            (10240, 8, 128, 16, 8),
        ],
    )
    def test_indexed_flatten_row_offsets_start(
        self,
        test_manager: Orchestrator,
        T,
        E,
        f_len,
        N,
        row_offsets_start,
    ):
        """Test row_offsets_start parameter for tensor-parallel MoE use cases."""
        partitions_per_row = T // f_len
        # Create row_offsets with N entries (more than E)
        row_offsets = [partitions_per_row * idx for idx in range(N)]
        output_len = N * T + 1024
        output_len = ((output_len + 127) // 128) * 128
        padding_val = -1

        kernel_input = generate_kernel_inputs(T, E, f_len, row_offsets, output_len, padding_val, row_offsets_start)
        input_tensor = kernel_input["input_tensor"]
        row_offsets_arr = kernel_input["row_offsets"]
        row_offsets_start_arr = kernel_input["row_offsets_start"]

        def golden_generator():
            return {
                "flattened_array": indexed_flatten_reference(
                    input_tensor, f_len, output_len, row_offsets_arr, row_offsets_start, padding_val
                )
            }

        test_manager.execute(
            KernelArgs(
                kernel_func=indexed_flatten,
                compiler_input=CompilerArgs(logical_nc_config=2),
                kernel_input={
                    "input_tensor": input_tensor,
                    "f_len": f_len,
                    "output_len": output_len,
                    "row_offsets": row_offsets_arr,
                    "row_offsets_start": row_offsets_start_arr,
                    "padding_val": padding_val,
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=golden_generator,
                        output_ndarray={"flattened_array": np.zeros(output_len, dtype=np.int32)},
                    ),
                    absolute_accuracy=0,
                    relative_accuracy=0,
                ),
            )
        )
