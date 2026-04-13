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

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from nkilib_src.nkilib.core.subkernels.find_nonzero_indices_torch import find_nonzero_indices_torch_ref


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


def _output_tensor_descriptor(kernel_input):
    """Generate output tensor descriptors for find_nonzero_indices."""
    T = kernel_input["input_tensor"].shape[0]
    C_out = kernel_input.get("n_cols", kernel_input["input_tensor"].shape[1])
    return {
        "indices": np.zeros((C_out, T), dtype=np.int32),
        "nonzero_counts": np.zeros(C_out, dtype=np.int32),
    }


_DTYPE_MAP = {"int32": np.int32, "float32": np.float32}

# fmt: off
FULL_PARAM_NAMES = \
    "T, E, top_k, chunk_size, in_dtype"
FULL_TEST_PARAMS = [
    (4096,  128, 8, 4096,  "int32"),
    (4096,  128, 8, 4096,  "float32"),
    (10240, 128, 4, 10240, "int32"),
    (10240, 128, 4, 10240, "float32"),
    (65536, 128, 4, 16384, "int32"),
    (65536, 128, 4, 16384, "float32"),
]

SUBSET_PARAM_NAMES = \
    "T, E, top_k, E_offset, E_local, chunk_size, in_dtype"
SUBSET_TEST_PARAMS = [
    (4096,  128, 8, 0,   4,  4096,  "int32"),
    (4096,  128, 8, 0,   4,  4096,  "float32"),
    (10240, 128, 4, 16,  16, 10240, "int32"),
    (10240, 128, 4, 16,  16, 10240, "float32"),
    (65536, 128, 4, 32,  16, 16384, "int32"),
    (65536, 128, 4, 32,  16, 16384, "float32"),
    (10240, 128, 4, 120, 8,  10240, "float32"),
]
# fmt: on


@pytest_test_metadata(
    name="FindNonzeroIndices",
    pytest_marks=["find_nonzero_indices"],
)
class TestFindNonzeroIndicesKernel:
    """Test class for find_nonzero_indices kernel."""

    @pytest.mark.fast
    @pytest_parametrize(FULL_PARAM_NAMES, FULL_TEST_PARAMS)
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

        def input_generator(test_config):
            return {
                "input_tensor": generate_sparse_input(T, E, top_k=top_k, dtype=_DTYPE_MAP[in_dtype]),
                "chunk_size": chunk_size,
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=find_nonzero_indices,
            torch_ref=torch_ref_wrapper(find_nonzero_indices_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensor_descriptor,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=2), atol=0, rtol=0)

    @pytest.mark.fast
    @pytest_parametrize(SUBSET_PARAM_NAMES, SUBSET_TEST_PARAMS)
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

        def input_generator(test_config):
            input_tensor = generate_sparse_input(T, E, top_k=top_k, dtype=_DTYPE_MAP[in_dtype])
            col_start_id = np.array([E_offset]).astype(np.int32)
            return {
                "input_tensor": input_tensor,
                "col_start_id": col_start_id,
                "n_cols": E_local,
                "chunk_size": chunk_size,
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=find_nonzero_indices,
            torch_ref=torch_ref_wrapper(find_nonzero_indices_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensor_descriptor,
        )
        framework.run_test(test_config=None, compiler_args=CompilerArgs(logical_nc_config=2), atol=0, rtol=0)
