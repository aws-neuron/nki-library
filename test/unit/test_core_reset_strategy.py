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
"""Unit tests for CoreResetStrategy: the single NEURON_RT_RESET_CORES decision point."""

import json
import os

import pytest

from test.utils.core_lock_manager import CoreAllocation
from test.utils.core_reset_strategy import CoreResetStrategy, _allocation_physical_cores


def _alloc(logical_core_ids: list[int], lnc_config: int = 2) -> CoreAllocation:
    return CoreAllocation(host_id="fakehost", logical_core_ids=logical_core_ids, lnc_config=lnc_config)


@pytest.fixture(autouse=True)
def _cleanup_state_files(tmp_path):
    """Remove any /tmp state files created during the test."""
    uid = f"test_{tmp_path.name}"
    yield
    CoreResetStrategy.cleanup_session_locks(uid)


def _exclusive_strategy(tmp_path) -> CoreResetStrategy:
    uid = f"test_{tmp_path.name}"
    return CoreResetStrategy(exclusive_run=True, testrun_uid=uid, ssh_alias="host1")


class TestAllocationPhysicalCores:
    """_allocation_physical_cores inverts logical->physical for the given LNC."""

    def test_lnc2_expands_each_logical_to_two_physical(self) -> None:
        assert _allocation_physical_cores([0, 1], lnc_config=2) == [0, 1, 2, 3]

    def test_lnc1_is_identity(self) -> None:
        assert _allocation_physical_cores([2], lnc_config=1) == [2]


class TestShouldResetCores:
    """Non-exclusive always resets; exclusive resets on first use,
    prior failure, or LNC transition, and skips when clean and matching."""

    def test_non_exclusive_always_resets(self) -> None:
        strategy = CoreResetStrategy(exclusive_run=False)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is True
        # Still True after a recorded clean run: non-exclusive ignores state.
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=False)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is True

    def test_first_use_forces_reset(self, tmp_path) -> None:
        strategy = _exclusive_strategy(tmp_path)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is True

    def test_prior_failure_forces_reset(self, tmp_path) -> None:
        strategy = _exclusive_strategy(tmp_path)
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=True)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is True

    def test_lnc_transition_forces_reset(self, tmp_path) -> None:
        strategy = _exclusive_strategy(tmp_path)
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=False)
        assert strategy.should_reset_cores(_alloc([0], lnc_config=1), lnc_config=1) is True

    def test_clean_same_lnc_skips_reset(self, tmp_path) -> None:
        strategy = _exclusive_strategy(tmp_path)
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=False)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is False

    def test_one_dirty_core_in_multi_core_allocation_resets(self, tmp_path) -> None:
        strategy = _exclusive_strategy(tmp_path)
        strategy.mark_test_complete(_alloc([0, 1]), lnc_config=2, failed=False)
        strategy.mark_test_complete(_alloc([1]), lnc_config=2, failed=True)  # only logical core 1 fails
        assert strategy.should_reset_cores(_alloc([0, 1]), lnc_config=2) is True
        # The untouched core alone stays clean.
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is False

    def test_exclusive_requires_state_path(self) -> None:
        with pytest.raises(ValueError):
            CoreResetStrategy(exclusive_run=True)


class TestShouldResetPurity:
    """should_reset_cores is a pure read: no state file created or mutated."""

    def test_read_does_not_create_state_file(self, tmp_path) -> None:
        uid = f"test_{tmp_path.name}"
        strategy = CoreResetStrategy(exclusive_run=True, testrun_uid=uid, ssh_alias="host1")
        state_path = CoreResetStrategy.build_state_path(uid, "host1")
        strategy.should_reset_cores(_alloc([0]), lnc_config=2)
        assert not os.path.exists(state_path)

    def test_read_does_not_mutate_state_file(self, tmp_path) -> None:
        uid = f"test_{tmp_path.name}"
        strategy = CoreResetStrategy(exclusive_run=True, testrun_uid=uid, ssh_alias="host1")
        state_path = CoreResetStrategy.build_state_path(uid, "host1")
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=False)
        with open(state_path) as f:
            before = json.load(f)
        strategy.should_reset_cores(_alloc([0]), lnc_config=2)
        strategy.should_reset_cores(_alloc([0, 1]), lnc_config=1)
        with open(state_path) as f:
            assert json.load(f) == before


class TestMarkTestComplete:
    """mark_test_complete records each physical core; no-op when not exclusive."""

    def test_round_trip_clean_then_failed(self, tmp_path) -> None:
        strategy = _exclusive_strategy(tmp_path)
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=False)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is False
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=True)
        assert strategy.should_reset_cores(_alloc([0]), lnc_config=2) is True

    def test_records_physical_cores_at_lnc2(self, tmp_path) -> None:
        uid = f"test_{tmp_path.name}"
        strategy = CoreResetStrategy(exclusive_run=True, testrun_uid=uid, ssh_alias="host1")
        state_path = CoreResetStrategy.build_state_path(uid, "host1")
        strategy.mark_test_complete(_alloc([0, 1]), lnc_config=2, failed=False)
        with open(state_path) as f:
            state = json.load(f)
        # Logical [0, 1] at LNC2 backs physical [0, 1, 2, 3].
        assert state == {str(core): {"lnc": 2, "failed": False} for core in range(4)}

    def test_no_op_when_not_exclusive(self) -> None:
        strategy = CoreResetStrategy(exclusive_run=False)
        strategy.mark_test_complete(_alloc([0]), lnc_config=2, failed=True)
        # Non-exclusive strategy has no state path; nothing written.

    def test_state_shared_across_strategy_instances(self, tmp_path) -> None:
        """Two instances over the same state file (xdist workers) see each other's records."""
        uid = f"test_{tmp_path.name}"
        writer = CoreResetStrategy(exclusive_run=True, testrun_uid=uid, ssh_alias="host1")
        reader = CoreResetStrategy(exclusive_run=True, testrun_uid=uid, ssh_alias="host1")
        writer.mark_test_complete(_alloc([0]), lnc_config=2, failed=False)
        assert reader.should_reset_cores(_alloc([0]), lnc_config=2) is False
