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

from test.utils.common_dataclasses import CompilerArgs
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.core.cumsum import cumsum
from nkilib_src.nkilib.core.cumsum.cumsum_torch import cumsum_torch_ref


def filter_invalid_combinations(batch, hidden, ndim, seq_len, dtype=None):
    """Filter out invalid parameter combinations.

    For 2D inputs, seq_len doesn't matter - only run one combination.
    """
    if ndim == 2 and seq_len != 1:
        return FilterResult.REDUNDANT
    return FilterResult.VALID


def _generate_inputs(batch, hidden, ndim, seq_len, dtype):
    """Generate cumsum kernel inputs from parameters."""
    np.random.seed(42)
    shape = (batch, seq_len, hidden) if ndim == 3 else (batch, hidden)
    return {"x": np.random.randn(*shape).astype(dtype)}


def _output_tensors(kernel_input):
    """Generate output tensor descriptors."""
    return {"output_0": np.zeros_like(kernel_input["x"])}


# fmt: off
FAST_PARAM_NAMES = \
    "batch, hidden, ndim, seq_len, dtype"
FAST_TEST_PARAMS = [
    pytest.param(1,   256,  2, 1, np.float32,         id="1_256_2_1_float32"),
    pytest.param(64,  1024, 3, 4, ml_dtypes.bfloat16, id="64_1024_3_4_bfloat16"),
    pytest.param(128, 2048, 2, 1, np.float32,         id="128_2048_2_1_float32"),
]
# fmt: on


@pytest_test_metadata(
    name="Cumsum",
    pytest_marks=["cumsum"],
)
class TestCumsumKernel:
    """Test class for cumsum kernel."""

    @pytest.mark.fast
    @pytest.mark.parametrize(FAST_PARAM_NAMES, FAST_TEST_PARAMS)
    def test_cumsum_fast(self, test_manager: Orchestrator, batch, hidden, ndim, seq_len, dtype):
        """Fast compile-only tests with minimal coverage."""
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return _generate_inputs(batch, hidden, ndim, seq_len, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cumsum,
            torch_ref=torch_ref_wrapper(cumsum_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(),
            atol=1e-2 if hidden > 5000 else 1e-3,
            rtol=1e-2 if is_bf16 else 1e-3,
        )

    @pytest.mark.coverage_parametrize(
        batch=[1, 64, 128, 256, 512, 1024, 2048],
        hidden=[256, 512, 1024, 2048, 4096, 6144, 8192],
        ndim=[2, 3],
        seq_len=[1, 4, 8, 10],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_cumsum_sweep(self, test_manager: Orchestrator, batch, hidden, ndim, seq_len, dtype, is_negative_test_case):
        """Full sweep tests with pairwise coverage."""
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return _generate_inputs(batch, hidden, ndim, seq_len, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cumsum,
            torch_ref=torch_ref_wrapper(cumsum_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(),
            atol=1e-2 if hidden > 5000 else 1e-3,
            rtol=1e-2 if is_bf16 else 1e-3,
            is_negative_test=is_negative_test_case,
        )
