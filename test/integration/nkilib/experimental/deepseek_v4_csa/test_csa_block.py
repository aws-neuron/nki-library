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
"""End-to-end tests for the whole CSA attention block.

The per-kernel tests in this directory grade each ``@nki.jit`` kernel against a CPU
reference in isolation. That leaves the block itself untested, and the block is where
a distinct class of bug lives: the torch projections around the kernels, the
trace-time dispatch that picks WHICH kernel a given shape uses, the head-parallel
weight sharding, and whether the kernels compose at all once ``torch_neuronx.trace``
compiles them together rather than one at a time.

That last one is not hypothetical. The XLA trace path rejects Python constructs the
standalone kernel tests happily compile, so a block-level compile failure can sit
behind a fully green per-kernel suite.

Each test traces every head-parallel rank, sums the partials on the host, and grades
the result against a 128-head CPU golden -- so a sharding mistake surfaces as a
numeric failure rather than as a plausible-looking number. The collective is covered
separately by ``test_csa_tp_all_reduce``; here ``replica_ranks=None``, which isolates
the block compute.
"""

import os
from typing import final

import pytest

from test.utils.common_dataclasses import Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata

pytestmark = pytest.mark.platforms(exclude=list(set(Platforms) - {Platforms.TRN3, Platforms.TRN3_A0}))

# Unlike the per-kernel tests, these do not go through the Orchestrator -- the block is
# an nn.Module, so it is traced with torch_neuronx IN THIS PROCESS. That means they only
# run where pytest itself is on a Trainium host, not when the run ships kernels to a
# remote fleet host, so skip rather than fail when there is no local device.
_HAS_LOCAL_DEVICE = os.path.exists("/dev/neuron0")


@final
@pytest_test_metadata(name="DeepSeek-V4 CSA Attention Block")
@pytest_marks(["deepseek_v4_csa"])
@pytest.mark.skip_simulation
@pytest.mark.high_rank
@pytest.mark.slow
class TestCsaBlock:
    """The full block: torch projections + every CSA kernel, graded end to end."""

    _PARAMS = "phase, seq_len, tp_size"
    # tp_size=4 is the production sharding (128 heads -> 32/rank). seq_len picks the
    # dispatch path: 8192 gives T_c=2048 (fused single-chunk indexer), 32768 gives
    # T_c=8192, which is where n_val == T_c and the sentinel pad is skipped.
    _CASES = [
        ("decode", 8192, 4),
        ("decode", 32768, 4),
        ("prefill", 8192, 4),
    ]
    _ABBREVS = {"seq_len": "s", "tp_size": "tp"}

    @pytest.mark.skipif(not _HAS_LOCAL_DEVICE, reason="in-process tracing needs a local Neuron device")
    @pytest_parametrize(_PARAMS, _CASES, abbrevs=_ABBREVS)
    def test_block_matches_cpu_golden(self, phase: str, seq_len: int, tp_size: int):
        """Trace each rank's block, host-sum the partials, grade against the golden."""
        pytest.importorskip("torch_neuronx", reason="block tracing needs torch_neuronx")

        from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_block import run_sequential
        from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_common import CSAConfigFull

        passed = run_sequential(phase, CSAConfigFull(seq_len=seq_len), tp_size)
        assert passed, f"{phase} block at seq_len={seq_len}, tp_size={tp_size} exceeded its tolerance"
