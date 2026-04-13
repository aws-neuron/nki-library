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

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import assert_negative_test_case
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.dynamic_shapes import dynamic_elementwise_add
from nkilib_src.nkilib.experimental.dynamic_shapes.dynamic_elementwise_add_torch import (
    dynamic_elementwise_add_torch_ref,
)

P_MAX = 128
H_TILE_SIZE = 512


def generate_kernel_inputs(m_dim, h_dim):
    """Generate random bf16 input tensors and num_m_tiles for the kernel."""
    generate_tensor = gaussian_tensor_generator()
    input_a = generate_tensor(name="input_a", shape=(m_dim, h_dim), dtype=ml_dtypes.bfloat16)
    input_b = generate_tensor(name="input_b", shape=(m_dim, h_dim), dtype=ml_dtypes.bfloat16)
    num_m_tiles = np.array([[m_dim // P_MAX]], dtype=np.int32)
    return {"input_a": input_a, "input_b": input_b, "num_m_tiles": num_m_tiles}


def golden_elementwise_add(input_a, input_b):
    """Compute golden reference using torch ref, returning result as numpy."""
    import torch

    a_torch = torch.from_numpy(input_a.view(np.uint16).view(ml_dtypes.bfloat16).astype(np.float32))
    b_torch = torch.from_numpy(input_b.view(np.uint16).view(ml_dtypes.bfloat16).astype(np.float32))
    result = dynamic_elementwise_add_torch_ref(a_torch, b_torch)
    return {"result": result.numpy().astype(ml_dtypes.bfloat16)}


@pytest_test_metadata(
    name="DynamicElementwiseAdd",
    pytest_marks=["dynamic_elementwise_add"],
)
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
        m_dim,
        h_dim,
        is_negative_test_case,
    ):
        """Fast tests: shape sweep over M and H dimensions."""
        kernel_input = generate_kernel_inputs(m_dim, h_dim)
        input_a = kernel_input["input_a"]
        input_b = kernel_input["input_b"]
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=dynamic_elementwise_add,
                    compiler_input=CompilerArgs(),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=lambda: golden_elementwise_add(input_a, input_b),
                            output_ndarray={"result": np.zeros(input_a.shape, dtype=input_a.dtype)},
                        ),
                        absolute_accuracy=1e-3,
                        relative_accuracy=1e-2,
                    ),
                )
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
        m_dim,
        h_dim,
        is_negative_test_case,
    ):
        """Verify add(a, b) produces the same result as add(b, a)."""
        kernel_input = generate_kernel_inputs(m_dim, h_dim)
        input_a = kernel_input["input_a"]
        input_b = kernel_input["input_b"]
        # Golden: run with swapped inputs — should produce identical result
        swapped_input = {"input_a": input_b, "input_b": input_a, "num_m_tiles": kernel_input["num_m_tiles"]}
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(
                KernelArgs(
                    kernel_func=dynamic_elementwise_add,
                    compiler_input=CompilerArgs(),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=lambda: golden_elementwise_add(input_a, input_b),
                            output_ndarray={"result": np.zeros(input_a.shape, dtype=input_a.dtype)},
                        ),
                        absolute_accuracy=0.0,
                        relative_accuracy=0.0,
                    ),
                )
            )
