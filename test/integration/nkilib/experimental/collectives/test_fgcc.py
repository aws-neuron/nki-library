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
"""Tests for fused all-gather + compute matmul (FGCC) kernel."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.collectives.fgcc import (
    allgather_compute_matmul,
)
from nkilib_src.nkilib.experimental.collectives.fgcc_torch import (
    allgather_compute_matmul_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

PARAM_NAMES = "m, K, N, dtype, tp_degree, lnc, force_hbm_cc"
TEST_PARAMS = [
    # Basic LNC2 tests
    (1024, 4096, 4096, nl.bfloat16, 4, 2, False),
    # NKILIB-796: The following tests timeout on trn2.3xl pipeline instances (32 cores)
    # because they require more cores than available. These tests need trn2.48xl instances.
    # Re-enable when dedicated high-core-count test infrastructure is available.
    #
    # (256, 1024, 1024, nl.bfloat16, 4, 2, False),
    # (512, 2048, 2048, nl.bfloat16, 4, 2, False),
    # # TP16 tests (requires 16 cores)
    # (512, 1024, 1024, nl.bfloat16, 16, 2, False),
    # (2048, 4096, 8192, nl.bfloat16, 16, 2, False),
    # # TP64 tests (requires 64 cores)
    # (512, 1024, 1024, nl.bfloat16, 64, 2, False),
    # (512, 1024, 1024, nl.bfloat16, 64, 1, False),
    # (512, 8192, 32768, nl.bfloat16, 64, 2, False),
    # # Force HBM mode
    # (1024, 2048, 2048, nl.bfloat16, 4, 2, True),
    # (256, 1024, 1024, nl.bfloat16, 4, 2, True),
    # (256, 1024, 1024, nl.bfloat16, 4, 1, True),
    # Float32 tests
    # (256, 1024, 1024, nl.float32, 4, 2, False),
    # LNC1 tests
    # (256, 1024, 1024, nl.bfloat16, 4, 1, False),
]
_ABBREVS = {"tp_degree": "tp", "force_hbm_cc": "hbm"}


@pytest_test_metadata(name="FGCC")
@pytest_marks(["collectives", "fgcc"])
class TestFgcc:
    """Test class for fused all-gather + compute matmul kernel."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS, abbrevs=_ABBREVS)
    def test_allgather_compute_matmul(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        m: int,
        K: int,
        N: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
        force_hbm_cc: bool,
    ):
        """Test fused all-gather + compute matmul (FGCC) kernel."""
        np.random.seed(42)
        num_groups = 1
        lhs_global = np.random.randn(m * tp_degree, K).astype(dtype)
        rhs_global = np.random.randn(K, N).astype(dtype)

        def create_inputs(rank_id: int):
            return {
                "lhs": lhs_global[rank_id * m : (rank_id + 1) * m, :],
                "rhs": rhs_global[:, rank_id * (N // tp_degree) : (rank_id + 1) * (N // tp_degree)],
                "tp_degree": tp_degree,
                "num_groups": num_groups,
                "force_hbm_cc": force_hbm_cc,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=allgather_compute_matmul,
            torch_ref=allgather_compute_matmul_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=tp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            output_keys=["result"],
            rtol=1e-2,
            atol=1e-2,
        )
