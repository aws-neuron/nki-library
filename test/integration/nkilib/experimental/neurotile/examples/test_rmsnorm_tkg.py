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
"""Integration test for the neurotile rmsnorm_tkg_nt example kernel.

Tests both LNC=1 and LNC=2 paths. The kernel only shards across cores when
BxS > 18 and BxS % 2 == 0, so the LNC=2 case uses a larger BxS to actually
exercise the sendrecv-exchange path.
"""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples.kernels.rmsnorm.tkg import (
    rmsnorm_tkg_nt as kernel_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples.kernels.rmsnorm.tkg import (
    rmsnorm_tkg_nt_torch as refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Hidden dim must be divisible by H0=128.
_H = 512
_H0 = 128
_H1 = _H // _H0


def _make_inputs(B, S_tkg):
    """Build a kernel_input_generator for given (B, S_tkg)."""

    def _gen(_):
        np.random.seed(42)
        return {
            "hidden": np.random.randn(B, S_tkg, _H).astype(ml_dtypes.bfloat16),
            "gamma": np.random.randn(1, _H).astype(ml_dtypes.bfloat16),
            "eps": 1e-6,
            "H_actual": None,
        }

    return _gen


def _make_outputs(B, S_tkg):
    bxs = B * S_tkg

    def _gen(_kernel_input):
        return {"out": np.zeros((_H0, bxs, _H1), dtype=ml_dtypes.bfloat16)}

    return _gen


@pytest_marks(["neurotile"])
class TestNeurotileRMSNormTKG:
    """RMSNorm TKG kernel — LNC=1 and LNC=2 variants."""

    @pytest.mark.fast
    def test_lnc1(self, test_manager: Orchestrator, platform_target: Platforms):
        # Small BxS — LNC=1 path.
        B, S_tkg = 1, 4
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_mod.rmsnorm_tkg_kernel,
            torch_ref=torch_ref_wrapper(refs.rmsnorm_tkg_torch_ref),
            kernel_input_generator=_make_inputs(B, S_tkg),
            output_tensor_descriptor=_make_outputs(B, S_tkg),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_lnc2(self, test_manager: Orchestrator, platform_target: Platforms):
        # BxS > 18 and BxS % 2 == 0 to engage the sendrecv-exchange shard path.
        B, S_tkg = 1, 32
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_mod.rmsnorm_tkg_kernel,
            torch_ref=torch_ref_wrapper(refs.rmsnorm_tkg_torch_ref),
            kernel_input_generator=_make_inputs(B, S_tkg),
            output_tensor_descriptor=_make_outputs(B, S_tkg),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-2,
            atol=1e-2,
        )
