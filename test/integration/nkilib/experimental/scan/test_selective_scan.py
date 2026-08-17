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

"""Integration tests for selective scan kernel."""

from typing import final

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.scan import selective_scan
from nkilib_src.nkilib.experimental.scan.selective_scan_torch import selective_scan_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_selective_scan_inputs(batch, channels, seq_len, state_size, dtype, use_D, use_initial_state):
    """Generate selective scan kernel inputs from parameters."""
    gen = gaussian_tensor_generator()

    x = gen(name="x", shape=(batch, channels, seq_len), dtype=dtype)

    # dt should be positive and small for numerical stability
    dt_raw = gen(name="dt_raw", shape=(batch, channels, seq_len), dtype=np.float32)
    dt = np.abs(dt_raw).astype(dtype) * 0.1

    # A should be negative for stable recurrence
    A_raw = gen(name="A", shape=(channels, state_size), dtype=np.float32)
    A = -np.abs(A_raw).astype(dtype) * 0.5

    B_mat = gen(name="B", shape=(batch, state_size, seq_len), dtype=dtype)
    C_mat = gen(name="C", shape=(batch, state_size, seq_len), dtype=dtype)

    inputs = {"x": x, "dt": dt, "A": A, "B": B_mat, "C": C_mat}

    if use_D:
        inputs["D"] = gen(name="D", shape=(channels,), dtype=dtype)
    if use_initial_state:
        init_raw = gen(name="initial_state", shape=(batch, channels, state_size), dtype=dtype)
        inputs["initial_state"] = init_raw * 0.1

    return inputs


def filter_invalid_combinations(batch, channels, seq_len, state_size, use_D, use_initial_state, dtype=None):
    """Filter out invalid parameter combinations."""
    return FilterResult.VALID


@pytest_test_metadata(
    name="SelectiveScan",
    pytest_marks=["selective_scan"],
)
@final
class TestSelectiveScanKernel:
    """Test class for selective scan kernel."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        batch=[1, 2],
        channels=[128, 256],
        seq_len=[256, 1024],
        state_size=[8, 16],
        use_D=[True, False],
        use_initial_state=[True, False],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=filter_invalid_combinations,
        coverage="singles",
        enable_automatic_boundary_tests=False,
    )
    def test_selective_scan_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        channels,
        seq_len,
        state_size,
        use_D,
        use_initial_state,
        dtype,
        is_negative_test_case,
    ):
        """Fast compile-only tests with minimal coverage."""
        is_bf16 = dtype == ml_dtypes.bfloat16
        abs_acc = 1e-1 if seq_len > 2048 else 5e-2
        rel_acc = 5e-2 if is_bf16 else 1e-2

        def input_generator(test_config):
            return generate_selective_scan_inputs(
                batch,
                channels,
                seq_len,
                state_size,
                dtype,
                use_D,
                use_initial_state,
            )

        def output_tensors(kernel_input):
            x = kernel_input["x"]
            return {
                "y": np.zeros((batch, channels, seq_len), dtype=x.dtype),
                "final_state": np.zeros((batch, channels, state_size), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=selective_scan,
            torch_ref=torch_ref_wrapper(selective_scan_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=abs_acc,
            rtol=rel_acc,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.coverage_parametrize(
        batch=[1, 2, 4, 8, 16],
        channels=[128, 256, 512, 768, 2560],
        seq_len=[128, 512, 1024, 2048, 4096],
        state_size=[8, 16, 32, 64],
        use_D=[True, False],
        use_initial_state=[True, False],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_selective_scan_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        channels,
        seq_len,
        state_size,
        use_D,
        use_initial_state,
        dtype,
        is_negative_test_case,
    ):
        """Full sweep tests with pairwise coverage."""
        is_bf16 = dtype == ml_dtypes.bfloat16
        abs_acc = 1e-1 if seq_len > 2048 else 5e-2
        rel_acc = 5e-2 if is_bf16 else 1e-2

        def input_generator(test_config):
            return generate_selective_scan_inputs(
                batch,
                channels,
                seq_len,
                state_size,
                dtype,
                use_D,
                use_initial_state,
            )

        def output_tensors(kernel_input):
            x = kernel_input["x"]
            return {
                "y": np.zeros((batch, channels, seq_len), dtype=x.dtype),
                "final_state": np.zeros((batch, channels, state_size), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=selective_scan,
            torch_ref=torch_ref_wrapper(selective_scan_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=abs_acc,
            rtol=rel_acc,
            is_negative_test=is_negative_test_case,
        )
