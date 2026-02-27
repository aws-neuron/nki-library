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

"""
Test suite for cascaded max kernel using UnitTestFramework.

Key Features:
- Test Structure: Uses UnitTestFramework with torch reference validation
- Reference Implementation: cascaded_max_torch_ref provides golden reference
- Validation: Framework handles comparison between hardware and reference
- Parameterized Tests: Unit tests with various batch sizes, sequence lengths, and vocabulary sizes
- Multiple Data Types: Support for float32

Test Coverage:
- Unit Tests: 26 different parameter combinations covering various tensor shapes
- Edge Cases: Single batch/sequence scenarios and larger vocabulary sizes up to 16K
"""

from test.utils.common_dataclasses import CompilerArgs
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.max.cascaded_max import cascaded_max
from nkilib_src.nkilib.core.max.cascaded_max_torch import cascaded_max_torch_ref


@pytest_test_metadata(
    name="Cascaded Max",
    pytest_marks=["max", "cascaded"],
)
class TestCascadedMaxKernel:
    @staticmethod
    def generate_inputs(batch: int, seqlen: int, vocab_size: int, dtype):
        """Generate input tensor for cascaded max kernel."""
        np.random.seed(42)
        return {"input_tensor": np.random.randn(batch, seqlen, vocab_size).astype(dtype)}

    @staticmethod
    def output_tensor_descriptor(kernel_input: dict):
        """Define output tensor shapes."""
        input_tensor = kernel_input["input_tensor"]
        batch, seqlen, _ = input_tensor.shape
        return {
            "max_values": np.zeros((batch, seqlen, 1), dtype=input_tensor.dtype),
            "max_indices": np.zeros((batch, seqlen, 1), dtype=np.int32),
        }

    # fmt: off
    cascaded_max_unit_params = "lnc_degree, batch, seqlen, vocab_size, dtype"
    cascaded_max_unit_perms = [
        # Llama 3 76B before global gather
        [1, 8, 5, 4058, nl.float32],
        [2, 8, 5, 4058, nl.float32],
        [1, 4, 5, 4058, nl.float32],
        [2, 5, 5, 4058, nl.float32],

        # # Llama 3 76B after global gather
        [1, 4, 5, 8192, nl.float32],
        [1, 8, 5, 8192, nl.float32],
        [2, 8, 5, 8192, nl.float32],
        [2, 5, 5, 8192, nl.float32],

        # Functionality tests
        # nominal
        [1, 1, 1, 3168, nl.float32],
        [2, 1, 1, 3168, nl.float32],

        # Vocab size generalization
        [2, 1, 1, 256, nl.float32],
        [2, 1, 1, 16000, nl.float32],

        # Max stage num batch sizes
        [2, 3, 1, 3168, nl.float32],
        [2, 7, 1, 3168, nl.float32],
        [2, 8, 1, 3168, nl.float32],

        # Medium stage num batch sizes
        [2, 10, 1, 3168, nl.float32],
        [2, 16, 1, 3168, nl.float32],
        [2, 32, 1, 3168, nl.float32],
        [2, 63, 1, 3168, nl.float32],
        [2, 65, 1, 3168, nl.float32],
        [2, 99, 1, 3168, nl.float32],
        [2, 128, 1, 3168, nl.float32],
        [2, 256, 1, 3168, nl.float32],
        [2, 1, 1, 3168, nl.float32],

        # Mixed tests
        [1, 1, 7, 3999,  nl.float32],
        [1, 1, 63, 3999, nl.float32],
        [2, 1, 127, 3999, nl.float32],
        [1, 1, 127, 3999, nl.float32],
    ]
    # fmt: on

    @pytest.mark.fast
    @pytest.mark.parametrize(cascaded_max_unit_params, cascaded_max_unit_perms)
    def test_cascaded_max_unit(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        vocab_size: int,
        dtype,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return self.generate_inputs(batch, seqlen, vocab_size, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cascaded_max,
            torch_ref=torch_ref_wrapper(cascaded_max_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree),
            rtol=1e-5,
            atol=1e-5,
        )

    def filter_combinations(lnc_degree, batch, seqlen, vocab_size, dtype):
        if 128 < batch * seqlen < 1:
            return FilterResult.INVALID
        if vocab_size > 2**14:
            return FilterResult.INVALID

    @pytest.mark.coverage_parametrize(
        lnc_degree=BoundedRange([1, 2], boundary_values=[]),
        batch=BoundedRange([1, 8, 32, 128], boundary_values=[]),
        seqlen=BoundedRange([1, 5], boundary_values=[]),
        vocab_size=BoundedRange([256, 3168, 8192, 16384], boundary_values=[]),
        dtype=BoundedRange([nl.float32, nl.bfloat16], boundary_values=[]),
        filter=filter_combinations,
        coverage="pairs",
    )
    @pytest.mark.paramterize
    def test_cascaded_max_sweep(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        vocab_size: int,
        dtype,
        is_negative_test_case: bool,
    ):
        from test.utils.coverage_parametrized_tests import assert_negative_test_case

        with assert_negative_test_case(is_negative_test_case):

            def input_generator(test_config, input_tensor_def=None):
                return self.generate_inputs(batch, seqlen, vocab_size, dtype)

            framework = UnitTestFramework(
                test_manager=test_manager,
                kernel_entry=cascaded_max,
                torch_ref=torch_ref_wrapper(cascaded_max_torch_ref),
                kernel_input_generator=input_generator,
                output_tensor_descriptor=self.output_tensor_descriptor,
            )
            framework.run_test(
                test_config=None,
                compiler_args=CompilerArgs(logical_nc_config=lnc_degree),
                rtol=1e-5,
                atol=1e-5,
            )
