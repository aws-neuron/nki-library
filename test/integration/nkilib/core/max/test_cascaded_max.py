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

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.max.cascaded_max import cascaded_max
from nkilib_src.nkilib.core.max.cascaded_max_torch import cascaded_max_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@pytest_test_metadata(name="Cascaded Max")
@pytest_marks(["max", "cascaded"])
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
    _ABBREVS = {"lnc_degree": "lnc", "batch": "b", "seqlen": "s", "vocab_size": "v", "dtype": "dt"}
    cascaded_max_unit_perms = [
        # Llama 3 76B before global gather
        [1, 8, 5, 4058, nl.float32],
        [2, 8, 5, 4058, nl.float32],
        [1, 4, 5, 4058, nl.float32],
        [2, 5, 5, 4058, nl.float32],

        # # Llama 3 76B after global gather
        pytest.param(1, 4, 5, 8192, nl.float32, marks=pytest.mark.fast),
        [1, 8, 5, 8192, nl.float32],
        [2, 8, 5, 8192, nl.float32],
        [2, 5, 5, 8192, nl.float32],

        # Functionality tests
        # nominal
        [1, 1, 1, 3168, nl.float32],
        [2, 1, 1, 3168, nl.float32],

        # Vocab size generalization
        [2, 1, 1, 256, nl.float32],
        pytest.param(2, 1, 1, 16000, nl.float32, marks=pytest.mark.fast),

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

    @pytest_parametrize(cascaded_max_unit_params, cascaded_max_unit_perms, abbrevs=_ABBREVS)
    def test_cascaded_max_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
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
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target),
            rtol=1e-5,
            atol=1e-5,
        )

    @staticmethod
    def _bf16_top_of_vocab_input(vocab_size: int):
        """Single-row bf16 input whose unique argmax sits at vocab_size-1.

        vocab_size-1 (e.g. 383) is not representable in bf16: the mantissa is
        exact only through 256, and the spacing in [256, 512) is 2, so 383
        rounds UP to 384 == vocab_size, one past the last valid token. An index
        path that runs in the bf16 logit dtype therefore returns an out-of-range
        index; the fp32 index path is exact and returns 383.
        """
        x = np.full((1, 1, vocab_size), -1.0).astype(nl.bfloat16)
        x[0, 0, vocab_size - 1] = 5.0
        return {"input_tensor": x}

    @pytest.mark.fast
    def test_cascaded_max_bf16_index_not_rounded(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """Argmax at a bf16-unrepresentable index must not round out of range.

        Regression guard for the fp32 index-path fix: with a bf16 logit tensor
        whose unique max is at index 383, a bf16 index path rounds the result up
        to 384 (out of range). Fails without the fix, passes with it.
        """
        vocab_size = 384

        def input_generator(test_config, input_tensor_def=None):
            return self._bf16_top_of_vocab_input(vocab_size)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=cascaded_max,
            torch_ref=torch_ref_wrapper(cascaded_max_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
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
    def test_cascaded_max_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
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
                compiler_args=CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target),
                rtol=1e-5,
                atol=1e-5,
            )
