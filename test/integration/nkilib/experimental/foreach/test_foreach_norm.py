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

"""Integration tests for foreach norm kernels (L1, L2, Linf)."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.foreach.foreach_norm import (
    l1_norm_kernel,
    l2_norm_kernel,
    linf_norm_kernel,
)
from nkilib_src.nkilib.experimental.foreach.foreach_norm_torch import (
    l1_norm_torch_ref,
    l2_norm_torch_ref,
    linf_norm_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

TEST_SHAPES = [(128,), (256,), (1024,), (4096,), (16384,), (32, 32), (4, 8, 32)]
BOUNDARY_SHAPES = [
    (127,),
    (129,),
    (255,),
    (257,),
    # Single tile boundary
    (128, 16384 - 1),
    (128, 16384 + 1),
    # Two tile boundary
    (256, 16384 - 1),
    (256, 16384 + 1),
    # Multi-tile (K > 2)
    (128, 16384 * 3),
    (128, 16384 * 4),
]


def _generate_inputs(shape):
    np.random.seed(42)
    data = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    numel = int(np.prod(shape))
    return {"data": data, "numel": numel}


def _output_tensors(kernel_input):
    return {"out": np.zeros((1, 1), dtype=ml_dtypes.bfloat16)}


@pytest_test_metadata(
    name="ForeachNorm",
    pytest_marks=["foreach_norm"],
)
class TestForeachNorm:
    """Integration tests for L1, L2, and Linf norm NKI kernels."""

    @pytest.mark.fast
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_l2_norm(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=l2_norm_kernel,
            torch_ref=torch_ref_wrapper(l2_norm_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(shape),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_l1_norm(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=l1_norm_kernel,
            torch_ref=torch_ref_wrapper(l1_norm_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(shape),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_linf_norm(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=linf_norm_kernel,
            torch_ref=torch_ref_wrapper(linf_norm_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(shape),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("shape", BOUNDARY_SHAPES)
    def test_l2_norm_boundary(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        """Boundary sizes: non-multiples of P_MAX to exercise tail path."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=l2_norm_kernel,
            torch_ref=torch_ref_wrapper(l2_norm_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(shape),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )
