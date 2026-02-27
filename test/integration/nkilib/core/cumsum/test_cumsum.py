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

"""Integration tests for cumsum kernel."""

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import FilterResult, assert_negative_test_case
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.core.cumsum import cumsum


def filter_invalid_combinations(batch, hidden, ndim, seq_len, dtype=None):
    """Filter out invalid parameter combinations.

    For 2D inputs, seq_len doesn't matter - only run one combination.
    """
    if ndim == 2 and seq_len != 1:
        return FilterResult.REDUNDANT
    return FilterResult.VALID


def generate_kernel_inputs(batch, hidden, ndim, seq_len, dtype):
    """Generate cumsum kernel inputs from parameters."""
    generate_tensor = gaussian_tensor_generator()
    shape = (batch, seq_len, hidden) if ndim == 3 else (batch, hidden)
    return {"x": generate_tensor(name="x", shape=shape, dtype=dtype)}


def golden_cumsum_np(x):
    """NumPy reference implementation of cumsum."""
    result = np.cumsum(x.astype(np.float32), axis=-1)
    return {"y": result.astype(x.dtype)}


@pytest_test_metadata(
    name="Cumsum",
    pytest_marks=["cumsum"],
)
@final
class TestCumsumKernel:
    """Test class for cumsum kernel."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        batch=[1, 64, 128],
        hidden=[256, 1024, 2048],
        ndim=[2, 3],
        seq_len=[1, 4],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=filter_invalid_combinations,
        coverage="singles",
        enable_automatic_boundary_tests=False,  # disable because kernel doesn't have hard constraints
    )
    def test_cumsum_fast(self, test_manager: Orchestrator, batch, hidden, ndim, seq_len, dtype, is_negative_test_case):
        """Fast compile-only tests with minimal coverage."""
        kernel_input = generate_kernel_inputs(batch, hidden, ndim, seq_len, dtype)
        x = kernel_input["x"]
        is_bf16 = dtype == ml_dtypes.bfloat16
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=cumsum,
                    compiler_input=CompilerArgs(),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=lambda: golden_cumsum_np(x),
                            output_ndarray={"y": np.zeros(x.shape, dtype=x.dtype)},
                        ),
                        absolute_accuracy=1e-2 if hidden > 5000 else 1e-3,
                        relative_accuracy=1e-2 if is_bf16 else 1e-3,
                    ),
                )
            )

    @pytest.mark.coverage_parametrize(
        batch=[1, 64, 128, 256, 512, 1024, 2048],
        hidden=[256, 512, 1024, 2048, 4096, 6144, 8192],
        ndim=[2, 3],
        seq_len=[1, 4, 8, 10],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,  # disable because kernel doesn't have hard constraints
    )
    def test_cumsum_sweep(self, test_manager: Orchestrator, batch, hidden, ndim, seq_len, dtype, is_negative_test_case):
        """Full sweep tests with pairwise coverage."""
        kernel_input = generate_kernel_inputs(batch, hidden, ndim, seq_len, dtype)
        x = kernel_input["x"]
        is_bf16 = dtype == ml_dtypes.bfloat16
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=cumsum,
                    compiler_input=CompilerArgs(),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=lambda: golden_cumsum_np(x),
                            output_ndarray={"y": np.zeros(x.shape, dtype=x.dtype)},
                        ),
                        absolute_accuracy=1e-2 if hidden > 5000 else 1e-3,
                        relative_accuracy=1e-2 if is_bf16 else 1e-3,
                    ),
                )
            )
