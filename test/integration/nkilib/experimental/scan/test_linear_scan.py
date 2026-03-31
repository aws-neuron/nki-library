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

"""Integration tests for linear_scan kernel."""

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
from nkilib_src.nkilib.experimental.scan import linear_scan


def filter_invalid_combinations(batch, P, L, dtype, use_initial):
    """Filter out invalid parameter combinations.

    All combinations are valid for linear_scan.
    """
    return FilterResult.VALID


def generate_kernel_inputs(batch, P, L, dtype, use_initial):
    """Generate linear_scan kernel inputs from parameters."""
    gen = gaussian_tensor_generator()
    # Generate raw decay values and apply sigmoid for stability (keep in (0, 1))
    decay_raw = gen(name="decay_raw", shape=(batch, P, L), dtype=np.float32)
    decay = (1.0 / (1.0 + np.exp(-decay_raw))).astype(dtype)
    data = gen(name="data", shape=(batch, P, L), dtype=dtype)
    inputs = {"decay": decay, "data": data}
    if use_initial:
        inputs["initial"] = gen(name="initial", shape=(batch, P, 1), dtype=dtype)
    return inputs


def golden_linear_scan_np(decay, data, initial=None):
    """NumPy reference implementation of linear scan.

    Computes result[t] = decay[t] * result[t-1] + data[t] sequentially.
    """
    orig_shape = decay.shape
    if decay.ndim == 2:
        decay = decay[np.newaxis]
        data = data[np.newaxis]
        if initial is not None:
            initial = initial[np.newaxis]

    outer, P, L = decay.shape
    result = np.zeros_like(decay, dtype=np.float32)

    if initial is not None:
        state = initial[:, :, 0].astype(np.float32)
    else:
        state = np.zeros((outer, P), dtype=np.float32)

    for t in range(L):
        state = decay[:, :, t].astype(np.float32) * state + data[:, :, t].astype(np.float32)
        result[:, :, t] = state

    final_state = state[:, :, np.newaxis]
    return {
        "result": result.reshape(orig_shape).astype(decay.dtype),
        "final_state": final_state.astype(np.float32),
    }


@pytest_test_metadata(
    name="LinearScan",
    pytest_marks=["linear_scan"],
)
@final
class TestLinearScanKernel:
    """Test class for linear_scan kernel."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        batch=[1, 4],
        P=[64, 128, 256],
        L=[128, 512, 2048],
        dtype=[np.float32, ml_dtypes.bfloat16],
        use_initial=[True, False],
        filter=filter_invalid_combinations,
        coverage="singles",
        enable_automatic_boundary_tests=False,
    )
    def test_linear_scan_fast(self, test_manager: Orchestrator, batch, P, L, dtype, use_initial, is_negative_test_case):
        """Fast compile-only tests with minimal coverage."""
        kernel_input = generate_kernel_inputs(batch, P, L, dtype, use_initial)
        decay = kernel_input["decay"]
        data = kernel_input["data"]
        initial = kernel_input.get("initial", None)
        is_bf16 = dtype == ml_dtypes.bfloat16
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=linear_scan,
                    compiler_input=CompilerArgs(),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=lambda: golden_linear_scan_np(decay, data, initial),
                            output_ndarray={
                                "result": np.zeros(decay.shape, dtype=decay.dtype),
                                "final_state": np.zeros(
                                    tuple(list(decay.shape[:-1]) + [1]), dtype=np.float32
                                ),
                            },
                        ),
                        absolute_accuracy=1e-2 if L > 2048 else 1e-3,
                        relative_accuracy=1e-2 if is_bf16 else 1e-3,
                    ),
                )
            )

    @pytest.mark.coverage_parametrize(
        batch=[1, 2, 8],
        P=[32, 64, 128, 256, 512],
        L=[64, 256, 512, 1024, 2048, 4096],
        dtype=[np.float32, ml_dtypes.bfloat16],
        use_initial=[True, False],
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_linear_scan_sweep(
        self, test_manager: Orchestrator, batch, P, L, dtype, use_initial, is_negative_test_case
    ):
        """Full sweep tests with pairwise coverage."""
        kernel_input = generate_kernel_inputs(batch, P, L, dtype, use_initial)
        decay = kernel_input["decay"]
        data = kernel_input["data"]
        initial = kernel_input.get("initial", None)
        is_bf16 = dtype == ml_dtypes.bfloat16
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=linear_scan,
                    compiler_input=CompilerArgs(),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=lambda: golden_linear_scan_np(decay, data, initial),
                            output_ndarray={
                                "result": np.zeros(decay.shape, dtype=decay.dtype),
                                "final_state": np.zeros(
                                    tuple(list(decay.shape[:-1]) + [1]), dtype=np.float32
                                ),
                            },
                        ),
                        absolute_accuracy=1e-2 if L > 2048 else 1e-3,
                        relative_accuracy=1e-2 if is_bf16 else 1e-3,
                    ),
                )
            )
