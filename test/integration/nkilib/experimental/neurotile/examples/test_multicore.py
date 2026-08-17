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
"""Integration tests for tutorials in test/docs/neurotile/examples/06_multicore/.

Block-sharded and interleaved kernels are launched at LNC=2 to match the
tutorial's `kernel[2](...)` invocation. Single-core kernels run at LNC=1.
"""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._06_multicore import (
    _01_tensor_add as add_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._06_multicore import (
    _01_tensor_add_torch as add_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._06_multicore import (
    _02_matmul as matmul_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._06_multicore import (
    _02_matmul_torch as matmul_refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Tensor-add tutorials use M=512, N=1024 bf16.
_ADD_M, _ADD_N = 512, 1024


def _add_inputs(_):
    np.random.seed(42)
    return {
        "a": np.random.rand(_ADD_M, _ADD_N).astype(ml_dtypes.bfloat16),
        "b": np.random.rand(_ADD_M, _ADD_N).astype(ml_dtypes.bfloat16),
    }


def _add_outputs(ki):
    return {"out": np.zeros_like(ki["a"])}


# Matmul tutorials use M=256, K=256, N=512.
_MM_M, _MM_K, _MM_N = 256, 256, 512


def _mm_inputs(_):
    np.random.seed(42)
    return {
        "lhsT": np.random.rand(_MM_K, _MM_M).astype(ml_dtypes.bfloat16),
        "rhs": np.random.rand(_MM_K, _MM_N).astype(ml_dtypes.bfloat16),
    }


def _mm_outputs(_kernel_input):
    return {"out": np.zeros((_MM_M, _MM_N), dtype=ml_dtypes.bfloat16)}


def _run(test_manager, platform_target, kernel, ref, inputs, outputs, logical_nc_config=1):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel,
        torch_ref=torch_ref_wrapper(ref),
        kernel_input_generator=inputs,
        output_tensor_descriptor=outputs,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=logical_nc_config),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest_marks(["neurotile"])
class TestNeurotileTensorAdd:
    """Tutorials in 01_tensor_add.py."""

    @pytest.mark.fast
    def test_add_single_core(self, test_manager: Orchestrator, platform_target: Platforms):
        _run(
            test_manager,
            platform_target,
            add_mod.tensor_add_single_core,
            add_refs.tensor_add_single_core_torch_ref,
            _add_inputs,
            _add_outputs,
            logical_nc_config=1,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (add_mod.tensor_add_block_sharded, add_refs.tensor_add_block_sharded_torch_ref),
            (add_mod.tensor_add_interleaved, add_refs.tensor_add_interleaved_torch_ref),
        ],
    )
    def test_lnc2(self, test_manager: Orchestrator, platform_target: Platforms, kernel, ref):
        _run(
            test_manager,
            platform_target,
            kernel,
            ref,
            _add_inputs,
            _add_outputs,
            logical_nc_config=2,
        )


@pytest_marks(["neurotile"])
class TestNeurotileMulticoreMatmul:
    """Tutorials in 02_matmul.py."""

    @pytest.mark.fast
    def test_matmul_single_core(self, test_manager: Orchestrator, platform_target: Platforms):
        _run(
            test_manager,
            platform_target,
            matmul_mod.matmul_single_core,
            matmul_refs.matmul_single_core_torch_ref,
            _mm_inputs,
            _mm_outputs,
            logical_nc_config=1,
        )

    @pytest.mark.fast
    def test_block_sharded_lnc2(self, test_manager: Orchestrator, platform_target: Platforms):
        _run(
            test_manager,
            platform_target,
            matmul_mod.matmul_block_sharded,
            matmul_refs.matmul_block_sharded_torch_ref,
            _mm_inputs,
            _mm_outputs,
            logical_nc_config=2,
        )
