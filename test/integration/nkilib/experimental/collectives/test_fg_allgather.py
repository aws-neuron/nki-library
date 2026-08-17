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
"""Tests for fine-grained ring-based all-gather kernel."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.collectives.fg_allgather import (
    fine_grained_allgather,
)
from nkilib_src.nkilib.experimental.collectives.fg_allgather_torch import (
    fine_grained_allgather_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

PARAM_NAMES = "m, K, dtype, tp_degree, lnc, force_hbm_cc"
TEST_PARAMS = [
    # Basic LNC2 tests
    (1024, 4096, nl.bfloat16, 4, 2, False),
    # NKILIB-795: The following cases failed in time out from trn2.3xl fleet in pipeline
    # while passed on trn2.48xl instance
    #
    #
    # (256, 2048, nl.bfloat16, 16, 2, False),
    # (512, 8192, nl.bfloat16, 4, 2, False),
    # Force HBM mode
    # (512, 8192, nl.bfloat16, 4, 2, True),
    # Float32
    # (1024, 4096, nl.float32, 16, 2, False),
    # LNC1
    # (1024, 4096, nl.bfloat16, 4, 1, False),
]
_ABBREVS = {"tp_degree": "tp", "force_hbm_cc": "hbm"}


@pytest_test_metadata(name="FgAllgather")
@pytest_marks(["collectives", "FgAllgather"])
class TestFgAllgather:
    """Test class for fine-grained ring-based all-gather kernel."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS, abbrevs=_ABBREVS)
    def test_fine_grained_allgather(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        m: int,
        K: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
        force_hbm_cc: bool,
    ):
        """Test fine-grained ring-based all-gather kernel."""
        np.random.seed(42)
        num_groups = 1
        M = m * tp_degree
        # Global tensor: [M, K] — each rank owns rows [rank*m : (rank+1)*m]
        lhs_global = np.random.randn(M, K).astype(dtype)

        def create_inputs(rank_id: int):
            return {
                "lhs": lhs_global[rank_id * m : (rank_id + 1) * m, :],
                "tp_degree": tp_degree,
                "num_groups": num_groups,
                "force_hbm_cc": force_hbm_cc,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=fine_grained_allgather,
            torch_ref=fine_grained_allgather_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=tp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            output_keys=["result"],
            rtol=1e-3,
            atol=1e-3,
        )
