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
"""Integration tests for the CSA tensor-parallel output all-reduce.

The reduction is an ordinary sum; what these tests are actually for is everything
the kernel does around it, because each of those has a failure mode that produces a
plausible-looking but HALF-REDUCED answer rather than an error:

* The collective's ``src``/``dst`` must be freshly allocated ``shared_hbm`` buffers
  with an explicit ``name=``, and a collective cannot read or write an IO tensor
  directly -- hence the staging copies in and out.
* It must launch on the ``[2]`` grid to match the block's ``lnc=2`` context. A
  ``[1]``-grid collective inside that graph fails outright, and hand-splitting it
  into two ``program_id``-sliced collectives instead wires only one slice across the
  ranks and silently leaves the other unreduced.

Each rank contributes DIFFERENT data, so a rank whose contribution never arrived
changes the sum. Ranks are given equal-magnitude values rather than a distinguishing
scale, so no single rank dominates and a dropped contribution cannot hide inside
rounding.
"""

from typing import final

import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_tp_all_reduce import nki_tp_all_reduce_kernel
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_tp_all_reduce_torch import nki_tp_all_reduce_torch_ref

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

pytestmark = pytest.mark.platforms(exclude=list(set(Platforms) - {Platforms.TRN3, Platforms.TRN3_A0}))

# The block reshapes its [B, S, dim] partial to a balanced [P, F] tile before
# reducing. P = 128 with dim = 7168 gives F = 56, which is the production shape.
_PARTITIONS = 128
_FREE = 56


@final
@pytest_test_metadata(name="DeepSeek-V4 CSA TP All-Reduce")
@pytest_marks(["collectives", "deepseek_v4_csa"])
@pytest.mark.skip_simulation
@pytest.mark.high_rank
class TestCsaTpAllReduce:
    """The 2-LNC collective that sums the head-parallel output partials across ranks."""

    _PARAMS = "collective_ranks, logical_nc_config"
    # 4 ranks x 2 LNC is the production topology; 2 ranks is the cheap smoke case.
    _CASES = [(4, 2), (2, 2)]
    _ABBREVS = {"collective_ranks": "ranks", "logical_nc_config": "lnc"}

    @pytest.mark.fast
    @pytest_parametrize(_PARAMS, _CASES, abbrevs=_ABBREVS)
    def test_tp_all_reduce(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collective_ranks: int,
        logical_nc_config: int,
    ):
        """Sum distinct per-rank partials and check every rank returns the full sum."""
        rng = np.random.default_rng(42)
        # Distinct data per rank, at comparable magnitudes, so the sum depends on
        # every rank without any one of them dominating.
        per_rank = rng.standard_normal((collective_ranks, _PARTITIONS, _FREE)).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": per_rank[rank_id], "replica_group": replica_group}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_tp_all_reduce_kernel,
            torch_ref=nki_tp_all_reduce_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
            inference_args=InferenceArgs(collective_ranks=collective_ranks, enable_determinism_check=True, num_runs=10),
        )
