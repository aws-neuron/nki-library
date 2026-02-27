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

"""Tests for RMSNorm TKG kernel using UnitTestFramework."""

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg
from nkilib_src.nkilib.core.subkernels.rmsnorm_torch import rmsnorm_tkg_torch_ref, rmsnorm_tkg_torch_ref_lnc1


def generate_inputs(batch, seqlen, hidden, hidden_actual, hidden_dim_tp, dtype):
    """Generate input tensors for rmsnorm_tkg test."""
    np.random.seed(42)
    return {
        "input": np.random.randn(batch, seqlen, hidden).astype(dtype),
        "gamma": np.random.randn(1, hidden).astype(dtype),
        "output.must_alias_input": np.zeros((128, batch * seqlen, hidden // 128), dtype=dtype),
        "hidden_actual": hidden_actual,
        "hidden_dim_tp": hidden_dim_tp,
    }


# fmt: off
RMSNORM_TKG_PARAMS = "lnc_degree, batch, seqlen, hidden, hidden_actual, hidden_dim_tp, dtype"
RMSNORM_TKG_TEST_CASES = [
    (1, 1, 1, 8192, None, False, np.float16),
    (1, 1, 8, 8192, None, False, np.float16),
    (1, 1, 1, 4096, 3072, False, np.float16),
    (1, 1, 4, 2048, 2048, False, np.float16),
    (1, 1, 4, 2048, 1920, False, np.float16),
    # LNC2 batch 1
    (2, 1, 1, 5120, None, False, np.float16),
    (2, 1, 1, 8192, None, False, np.float16),
    # LNC2 higher batch
    (2, 2, 8, 8192, None, False, np.float16),
    (2, 4, 8, 8192, None, False, np.float16),
    (2, 128, 1, 8192, None, False, np.float16),
    (2, 4, 1, 5120, None, False, np.float16),
    # LNC2 BxS tiling
    (2, 128, 5, 3072, None, False, np.float16),
    (2, 256, 5, 3072, None, False, np.float16),
    (2, 63, 32, 3072, None, False, np.float16),
    # LNC2 higher hidden
    (2, 1, 1, 16384, None, False, np.float16),
    (2, 1, 1, 4096, 3072, False, np.float16),
    # Sharding threshold
    (2, 2, 8, 16384, None, False, np.float16),
    (2, 2, 8, 3072, 2880, False, np.float16),
    # hidden_dim_tp = True
    (1, 1, 1, 4096, 3072, True, np.float16),
    (2, 1, 1, 5120, None, True, np.float16),
    (2, 128, 1, 8192, None, True, np.float16),
    (2, 2, 8, 3072, 2880, True, np.float16),
]
# fmt: on


@pytest_test_metadata(
    name="RMSNorm TKG",
    pytest_marks=["rmsnorm", "tkg"],
)
class TestRmsNormTKGKernel:
    """Test class for RMSNorm TKG kernel using UnitTestFramework."""

    @pytest.mark.fast
    @pytest.mark.parametrize(RMSNORM_TKG_PARAMS, RMSNORM_TKG_TEST_CASES)
    def test_rmsnorm_tkg_unit(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        hidden_actual,
        hidden_dim_tp: bool,
        dtype,
    ):
        """Test rmsnorm_tkg using UnitTestFramework."""

        def input_generator(test_config, input_tensor_def=None):
            return generate_inputs(batch, seqlen, hidden, hidden_actual, hidden_dim_tp, dtype)

        def output_tensors(kernel_input):
            return {"out": np.zeros((128, batch * seqlen, hidden // 128), dtype=dtype)}

        # Select torch_ref based on lnc_degree
        torch_ref = rmsnorm_tkg_torch_ref if lnc_degree == 2 and not hidden_dim_tp else rmsnorm_tkg_torch_ref_lnc1

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_tkg,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree),
            rtol=2e-2,
            atol=1e-5,
        )
