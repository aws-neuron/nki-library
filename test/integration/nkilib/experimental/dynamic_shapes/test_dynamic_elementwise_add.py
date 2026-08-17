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

"""Integration tests for dynamic elementwise add kernel."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.dynamic_shapes import dynamic_elementwise_add
from nkilib_src.nkilib.experimental.dynamic_shapes.dynamic_elementwise_add_torch import (
    dynamic_elementwise_add_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

P_MAX = 128


def generate_inputs(m_dim, h_dim):
    """Generate random bf16 input tensors and num_m_tiles for the kernel."""
    np.random.seed(42)
    input_a = np.random.randn(m_dim, h_dim).astype(ml_dtypes.bfloat16)
    input_b = np.random.randn(m_dim, h_dim).astype(ml_dtypes.bfloat16)
    num_m_tiles = np.array([[m_dim // P_MAX]], dtype=np.int32)
    return {"input_a": input_a, "input_b": input_b, "num_m_tiles": num_m_tiles}


@pytest_test_metadata(name="DynamicElementwiseAdd")
@pytest_marks(["dynamic_elementwise_add"])
class TestDynamicElementwiseAdd:
    """Test class for dynamic elementwise add kernel."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        m_dim=[128, 256, 1024],
        h_dim=[512, 1024, 2048],
        coverage="singles",
        enable_automatic_boundary_tests=False,
    )
    def test_dynamic_elementwise_add_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        m_dim,
        h_dim,
        is_negative_test_case,
    ):
        """Fast tests: shape sweep over M and H dimensions."""

        def input_generator(test_config):
            return generate_inputs(m_dim, h_dim)

        def output_tensors(kernel_input):
            return {"out": np.zeros(kernel_input["input_a"].shape, dtype=kernel_input["input_a"].dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=dynamic_elementwise_add,
            torch_ref=torch_ref_wrapper(dynamic_elementwise_add_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2,
            atol=1e-3,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        m_dim=[128, 256],
        h_dim=[512, 1024],
        coverage="singles",
        enable_automatic_boundary_tests=False,
    )
    def test_dynamic_elementwise_add_commutativity(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        m_dim,
        h_dim,
        is_negative_test_case,
    ):
        """Verify add(a, b) produces the same result as add(b, a)."""

        def input_generator(test_config):
            return generate_inputs(m_dim, h_dim)

        def output_tensors(kernel_input):
            return {"out": np.zeros(kernel_input["input_a"].shape, dtype=kernel_input["input_a"].dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=dynamic_elementwise_add,
            torch_ref=torch_ref_wrapper(dynamic_elementwise_add_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.0,
            atol=0.0,
            is_negative_test=is_negative_test_case,
        )
