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

"""Integration tests for find_nonzero_indices kernel."""

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
from nkilib_src.nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices


def find_nonzero_indices_reference(
    input_tensor: np.ndarray, col_start_id: int = None, n_cols: int = None
) -> dict[str, np.ndarray]:
    """
    NumPy reference implementation of find_nonzero_indices.

    Args:
        input_tensor: [T, E] input array
        col_start_id: Starting column index (if specified, only process n_cols columns starting from this index)
        n_cols: Number of columns to process

    Returns:
        dict with 'indices' and 'nonzero_counts' arrays
    """
    T, E = input_tensor.shape

    if col_start_id is not None:
        E_out = n_cols
        start_col = col_start_id
    else:
        E_out = E
        start_col = 0

    indices = np.full((E_out, T), -1, dtype=np.int32)
    nonzero_counts = np.zeros(E_out, dtype=np.int32)

    for e in range(E_out):
        col = input_tensor[:, start_col + e]
        nz_indices = np.nonzero(col)[0]
        count = len(nz_indices)
        nonzero_counts[e] = count
        if count > 0:
            indices[e, :count] = nz_indices

    return {"indices": indices, "nonzero_counts": nonzero_counts}


def generate_sparse_input(T: int, E: int, top_k: int, dtype: np.dtype, seed: int = 42) -> np.ndarray:
    """Generate sparse input tensor matching the original test's sparsity pattern.

    Args:
        T: Number of tokens (cols in output)
        E: Number of experts (columns in input)
        top_k: Controls sparsity - K = T * top_k // E nonzeros per expert
        dtype: Data type for the tensor
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    K = T * top_k // E
    flattened_input = np.random.permutation(np.concatenate([np.ones(K * E), np.zeros((T - K) * E)]))
    input_tensor = flattened_input.reshape(E, T).T.astype(dtype)
    if dtype == np.float32:
        input_tensor[input_tensor > 0] = 0.1
    return input_tensor


@pytest_test_metadata(
    name="FindNonzeroIndices",
    pytest_marks=["find_nonzero_indices"],
)
@final
class TestFindNonzeroIndicesKernel:
    """Test class for find_nonzero_indices kernel."""

    # TODO: Switch to coverage_parametrize when derive_pytest_test_id() in feature_flag_helper.py
    # is updated to sanitize special characters like < > ' from dtype class names.

    # Tests WITHOUT col_start_id (full tensor processing)
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "T,E,top_k,chunk_size,in_dtype",
        [
            (4096, 128, 8, 4096, np.int32),
            (4096, 128, 8, 4096, np.float32),
            (10240, 128, 4, 10240, np.int32),
            (10240, 128, 4, 10240, np.float32),
            (65536, 128, 4, 16384, np.int32),
            (65536, 128, 4, 16384, np.float32),
        ],
    )
    def test_find_nonzero_indices_fast(
        self,
        test_manager: Orchestrator,
        T,
        E,
        top_k,
        chunk_size,
        in_dtype,
    ):
        """Fast tests for full tensor processing."""
        input_tensor = generate_sparse_input(T, E, top_k=top_k, dtype=in_dtype)

        def golden_generator():
            return find_nonzero_indices_reference(input_tensor)

        test_manager.execute(
            KernelArgs(
                kernel_func=find_nonzero_indices,
                compiler_input=CompilerArgs(logical_nc_config=2),
                kernel_input={
                    "input_tensor": input_tensor,
                    "chunk_size": chunk_size,
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=golden_generator,
                        output_ndarray={
                            "indices": np.zeros((E, T), dtype=np.int32),
                            "nonzero_counts": np.zeros(E, dtype=np.int32),
                        },
                    ),
                    absolute_accuracy=0,
                    relative_accuracy=0,
                ),
            )
        )

    # Tests WITH col_start_id and n_cols (subset processing)
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "T,E,top_k,E_offset,E_local,chunk_size,in_dtype",
        [
            (4096, 128, 8, 0, 4, 4096, np.int32),
            (4096, 128, 8, 0, 4, 4096, np.float32),
            (10240, 128, 4, 16, 16, 10240, np.int32),
            (10240, 128, 4, 16, 16, 10240, np.float32),
            (65536, 128, 4, 32, 16, 16384, np.int32),
            (65536, 128, 4, 32, 16, 16384, np.float32),
            (10240, 128, 4, 120, 8, 10240, np.float32),
        ],
    )
    def test_find_nonzero_indices_with_col_start_fast(
        self,
        test_manager: Orchestrator,
        T,
        E,
        top_k,
        E_offset,
        E_local,
        chunk_size,
        in_dtype,
    ):
        """Fast tests with col_start_id and n_cols for subset processing."""
        input_tensor = generate_sparse_input(T, E, top_k=top_k, dtype=in_dtype)
        col_start_id = np.array([E_offset]).astype(np.int32)

        def golden_generator():
            return find_nonzero_indices_reference(input_tensor, col_start_id=E_offset, n_cols=E_local)

        test_manager.execute(
            KernelArgs(
                kernel_func=find_nonzero_indices,
                compiler_input=CompilerArgs(
                    logical_nc_config=2,
                ),
                kernel_input={
                    "input_tensor": input_tensor,
                    "col_start_id": col_start_id,
                    "n_cols": E_local,
                    "chunk_size": chunk_size,
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=golden_generator,
                        output_ndarray={
                            "indices": np.zeros((E_local, T), dtype=np.int32),
                            "nonzero_counts": np.zeros(E_local, dtype=np.int32),
                        },
                    ),
                    absolute_accuracy=0,
                    relative_accuracy=0,
                ),
            )
        )
