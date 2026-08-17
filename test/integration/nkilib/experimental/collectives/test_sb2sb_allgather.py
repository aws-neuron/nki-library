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
"""Tests for SBUF-to-SBUF All-Gather kernels."""

import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.sb2sb_allgather import (
    allgather_sb2sb,
    allgather_sb2sb_tiled,
)
from nkilib_src.nkilib.experimental.collectives.sb2sb_allgather_torch import (
    allgather_sb2sb_tiled_torch_ref,
    allgather_sb2sb_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

SB2SB_PARAM_NAMES = "m, k, dtype, tp_degree"
# These run at lnc=1, so physical cores == tp_degree. tp_degree > 8 needs more
# than a 3xl host's 8 cores, so those configs are marked high_rank to route them
# to 48xl hosts; tp_degree <= 8 stays in the default lane.
SB2SB_TEST_PARAMS = [
    # Basic tests
    (128, 512, nl.bfloat16, 8),
    (64, 1024, nl.bfloat16, 8),
    (128, 2048, nl.bfloat16, 8),
    pytest.param(96, 512, nl.bfloat16, 16, marks=pytest.mark.high_rank),
    # dtype variations
    (128, 512, np.float32, 8),
    (64, 1024, np.float16, 8),
    # Different TP degrees
    pytest.param(128, 256, nl.bfloat16, 64, marks=pytest.mark.high_rank),
    pytest.param(128, 256, nl.bfloat16, 32, marks=pytest.mark.high_rank),
    # Non-power-of-2 k
    pytest.param(128, 384, nl.bfloat16, 16, marks=pytest.mark.high_rank),
]

TILED_PARAM_NAMES = "m, k, dtype, tp_degree, lnc"
# Physical cores == tp_degree * lnc. Configs needing > 8 cores are marked
# high_rank to route them to 48xl hosts; the rest stay in the default lane.
TILED_TEST_PARAMS = [
    # Single tile cases (m <= 128)
    (128, 512, nl.bfloat16, 4, 2),
    (64, 1024, nl.bfloat16, 4, 2),
    # Multi-tile cases (m > 128, m % 128 == 0)
    (256, 512, nl.bfloat16, 8, 1),
    (256, 512, nl.bfloat16, 4, 2),
    (512, 1024, nl.bfloat16, 8, 1),
    pytest.param(512, 1024, nl.bfloat16, 8, 2, marks=pytest.mark.high_rank),
    # dtype variations
    (256, 512, np.float32, 8, 1),
    (512, 1024, np.float16, 4, 2),
    pytest.param(256, 512, nl.bfloat16, 8, 2, marks=pytest.mark.high_rank),
]
_ABBREVS = {"tp_degree": "tp"}


def _run_sb2sb_allgather_test(
    test_manager, platform_target, kernel_entry, torch_ref, m, k, dtype, tp_degree, lnc, output_keys
):
    """Shared test logic for SBUF-to-SBUF all-gather kernels."""
    np.random.seed(42)
    x_global = np.random.randn(tp_degree, m, k).astype(dtype)
    replica_groups = ReplicaGroup([list(range(tp_degree))])

    def create_inputs(rank_id: int):
        return {
            "inp": x_global[rank_id],
            "replica_groups": replica_groups,
            "tp_degree": tp_degree,
        }

    CollectiveUnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel_entry,
        torch_ref=torch_ref,
        per_rank_input_generator=create_inputs,
        collective_ranks=tp_degree,
    ).run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
        output_keys=output_keys,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skip_simulation
class TestSb2sbAllgather:
    """Test class for SBUF-to-SBUF all-gather kernels."""

    @pytest.mark.fast
    @pytest_parametrize(SB2SB_PARAM_NAMES, SB2SB_TEST_PARAMS, abbrevs=_ABBREVS)
    def test_allgather_sb2sb(
        self, test_manager: Orchestrator, platform_target: Platforms, m: int, k: int, dtype: np.dtype, tp_degree: int
    ):
        """Test basic SBUF-to-SBUF all-gather kernel."""
        _run_sb2sb_allgather_test(
            test_manager,
            platform_target,
            allgather_sb2sb,
            allgather_sb2sb_torch_ref,
            m,
            k,
            dtype,
            tp_degree,
            lnc=1,
            output_keys=["out"],
        )

    @pytest.mark.fast
    @pytest_parametrize(TILED_PARAM_NAMES, TILED_TEST_PARAMS, abbrevs=_ABBREVS)
    def test_allgather_sb2sb_tiled(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        m: int,
        k: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
    ):
        """Test tiled SBUF-to-SBUF all-gather kernel with LNC support."""
        _run_sb2sb_allgather_test(
            test_manager,
            platform_target,
            allgather_sb2sb_tiled,
            allgather_sb2sb_tiled_torch_ref,
            m,
            k,
            dtype,
            tp_degree,
            lnc,
            output_keys=["result"],
        )
