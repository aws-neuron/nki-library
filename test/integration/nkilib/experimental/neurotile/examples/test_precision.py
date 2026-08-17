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
"""Integration tests for tutorials in test/docs/neurotile/examples/04_precision/."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._04_precision import (
    _01_mixed_precision as mp_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._04_precision import (
    _01_mixed_precision_torch as mp_refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Tutorial uses M=256, K=256, N=512 (fp32 inputs, kernel rounds to bf16 internally).
_M, _K, _N = 256, 256, 512


def _matmul_inputs(_):
    np.random.seed(42)
    return {
        "A_hbm": np.random.rand(_K, _M).astype(np.float32),  # AT shape [K, M]
        "B_hbm": np.random.rand(_K, _N).astype(np.float32),
    }


@pytest_marks(["neurotile"])
class TestNeurotileMixedPrecision:
    """Tutorials in 01_mixed_precision.py."""

    @pytest.mark.fast
    def test_fp32_output(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mp_mod.mixed_precision_matmul_fp32_output,
            torch_ref=torch_ref_wrapper(mp_refs.mixed_precision_matmul_fp32_output_torch_ref),
            kernel_input_generator=_matmul_inputs,
            output_tensor_descriptor=lambda _: {"out": np.zeros((_M, _N), dtype=np.float32)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.fast
    def test_bf16_output(self, test_manager, platform_target):
        import ml_dtypes

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mp_mod.mixed_precision_matmul_bf16_output,
            torch_ref=torch_ref_wrapper(mp_refs.mixed_precision_matmul_bf16_output_torch_ref),
            kernel_input_generator=_matmul_inputs,
            output_tensor_descriptor=lambda _: {"out": np.zeros((_M, _N), dtype=ml_dtypes.bfloat16)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
        )
