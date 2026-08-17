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
"""Unit tests for core_lock_manager module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from test.utils.core_lock_client import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    DEFAULT_LOCKING_PROTOCOL_VERSION,
)
from test.utils.core_lock_manager import (
    AllocationStatus,
    CoreLockManager,
    InsufficientCoreCountError,
    LockAcquisitionError,
    LockVersionError,
    calculate_total_needed_physical_cores,
    check_lock_version,
)
from test.utils.metrics_collector import MetricName, NoopMetricsCollector
from test.utils.scripts import remote_lock_scripts
from test.utils.scripts.remote_lock_scripts import LockResult, LockStatus


class TestLockVersionError:
    """Tests for LockVersionError exception."""

    def test_error_message(self):
        """Test that error message contains version information."""
        error = LockVersionError(required_version=99, current_version=2)
        assert "99" in str(error)
        assert "2" in str(error)
        assert error.required_version == 99
        assert error.current_version == 2
        assert error.retryable is True


class TestCheckLockVersion:
    """Tests for the check_lock_version guard (pure version comparison)."""

    def test_passes_when_equal(self):
        """Returns None (does not raise) when host version == client's supported version."""
        assert check_lock_version(DEFAULT_LOCKING_PROTOCOL_VERSION) is None

    def test_passes_when_host_requires_lower(self):
        """Returns None (does not raise) when the host requires an older version."""
        assert check_lock_version(DEFAULT_LOCKING_PROTOCOL_VERSION - 1) is None

    def test_fails_when_client_too_old(self):
        """Raise LockVersionError when the host requires a newer version than the client."""
        with pytest.raises(LockVersionError) as exc_info:
            check_lock_version(99)

        assert exc_info.value.required_version == 99
        assert exc_info.value.current_version == DEFAULT_LOCKING_PROTOCOL_VERSION


class TestCalculateTotalNeededPhysicalCores:
    """Tests for calculate_total_needed_physical_cores (collectives_ranks * lnc_config)."""

    def test_lnc2_multiplies_ranks_by_two(self):
        assert calculate_total_needed_physical_cores(collectives_ranks=4, lnc_config=2) == 8

    def test_lnc1_equals_ranks(self):
        assert calculate_total_needed_physical_cores(collectives_ranks=4, lnc_config=1) == 4

    def test_single_rank(self):
        assert calculate_total_needed_physical_cores(collectives_ranks=1, lnc_config=2) == 2

    def test_zero_ranks(self):
        assert calculate_total_needed_physical_cores(collectives_ranks=0, lnc_config=2) == 0


class TestPhysicalToLogicalCores:
    """Tests for CoreLockManager._physical_to_logical_cores."""

    def test_lnc2_conversion(self):
        """Test LNC2: 2 physical cores = 1 logical core."""
        result = CoreLockManager._physical_to_logical_cores([0, 1, 2, 3], lnc_config=2)
        assert result == [0, 1]

    def test_lnc2_offset(self):
        """Test LNC2 with offset physical cores."""
        result = CoreLockManager._physical_to_logical_cores([4, 5, 6, 7], lnc_config=2)
        assert result == [2, 3]

    def test_lnc1_conversion(self):
        """Test LNC1: 1 physical core = 1 logical core."""
        result = CoreLockManager._physical_to_logical_cores([0, 1, 2], lnc_config=1)
        assert result == [0, 1, 2]

    def test_lnc1_offset(self):
        """Test LNC1 with offset physical cores."""
        result = CoreLockManager._physical_to_logical_cores([4, 5, 6, 7], lnc_config=1)
        assert result == [4, 5, 6, 7]

    def test_misaligned_raises(self):
        """Test that misaligned physical cores raise assertion."""
        with pytest.raises(AssertionError) as exc_info:
            CoreLockManager._physical_to_logical_cores([1, 2, 3, 4], lnc_config=2)
        assert "contiguous and aligned" in str(exc_info.value)


class TestDrain:
    """Tests for CoreLockManager.drain method."""

    def _make_manager(self) -> CoreLockManager:
        mock_executor = MagicMock()
        mgr = CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=mock_executor,
            host_locking_version=2,
        )
        return mgr

    @patch("test.utils.core_lock_client.drain")
    def test_drain_returns_max_lock_expiry(self, mock_drain):
        """drain() returns max_lock_expiry from result file."""
        mock_drain.return_value = LockResult(status=LockStatus.DRAINED, max_lock_expiry=1234567890)
        mgr = self._make_manager()
        assert mgr.drain(timeout_seconds=1800) == 1234567890

    @patch("test.utils.core_lock_client.drain")
    def test_drain_returns_zero_when_no_active_locks(self, mock_drain):
        """drain() returns 0 when no active locks."""
        mock_drain.return_value = LockResult(status=LockStatus.DRAINED, max_lock_expiry=0)
        mgr = self._make_manager()
        assert mgr.drain(timeout_seconds=1800) == 0

    @patch("test.utils.core_lock_client.drain")
    def test_drain_returns_zero_on_failure(self, mock_drain):
        """drain() returns 0 when helper reports error."""
        mock_drain.return_value = LockResult(status=LockStatus.ERROR, message="test error")
        mgr = self._make_manager()
        assert mgr.drain(timeout_seconds=1800) == 0


class TestDisableDrain:
    """Tests for CoreLockManager.disable_drain method."""

    def _make_manager(self) -> CoreLockManager:
        mock_executor = MagicMock()
        mgr = CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=mock_executor,
            host_locking_version=2,
        )
        return mgr

    @patch("test.utils.core_lock_client.undrain")
    def test_disable_drain_succeeds(self, mock_undrain):
        """disable_drain() calls undrain helper."""
        mock_undrain.return_value = LockResult(status=LockStatus.UNDRAINED)
        mgr = self._make_manager()
        mgr.disable_drain()  # Should not raise

    @patch("test.utils.core_lock_client.undrain")
    def test_disable_drain_warns_on_failure(self, mock_undrain):
        """disable_drain() logs warning on failure."""
        mock_undrain.return_value = LockResult(status=LockStatus.ERROR, message="test error")
        mgr = self._make_manager()
        mgr.disable_drain()  # Should not raise, just warn


class TestEntryId:
    """Tests for CoreLockManager entry_id / caller_id minting."""

    def _make_manager(self, collector) -> CoreLockManager:
        return CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=collector,
            executor=MagicMock(),
            host_locking_version=2,
        )

    def test_distinct_entry_ids(self):
        """Two managers get distinct entry_id values."""
        mgr1 = self._make_manager(NoopMetricsCollector())
        mgr2 = self._make_manager(NoopMetricsCollector())
        assert mgr1.entry_id != mgr2.entry_id

    def test_caller_id_unknown_when_test_name_none(self):
        """caller_id ends in ':unknown' when collector.test_name is None."""
        collector = MagicMock()
        collector.test_name = None
        mgr = self._make_manager(collector)
        assert mgr._caller_id == f"{mgr.entry_id}:unknown"

    def test_caller_id_unknown_when_test_name_absent(self):
        """caller_id ends in ':unknown' when collector lacks test_name."""
        mgr = self._make_manager(NoopMetricsCollector())
        assert mgr._caller_id.endswith(":unknown")

    def test_caller_id_uses_test_name_when_present(self):
        """caller_id is f'{entry_id}:{test_name}' when test_name is present."""
        collector = MagicMock()
        collector.test_name = "my_test"
        mgr = self._make_manager(collector)
        assert mgr._caller_id == f"{mgr.entry_id}:my_test"

    def test_entry_id_stable_across_reads(self):
        """entry_id property returns the same value across repeated reads."""
        mgr = self._make_manager(NoopMetricsCollector())
        assert mgr.entry_id == mgr.entry_id == mgr._entry_id


class TestAcquireOutcome:
    """Tests for queue-aware CoreLockManager.acquire -> AllocationOutcome mapping."""

    def _make_manager(self) -> CoreLockManager:
        return CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=MagicMock(),
            host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
        )

    @patch("test.utils.core_lock_client.poll")
    def test_allocated_maps_to_logical_and_physical(self, mock_poll):
        """ALLOCATED wire result maps to AllocationOutcome with logical+physical cores."""
        mock_poll.return_value = LockResult(status=LockStatus.ALLOCATED, cores=[0, 1, 2, 3], expiry=123)
        mgr = self._make_manager()
        outcome = mgr.acquire(num_logical_cores=2, lnc_config=2)
        assert outcome.status == AllocationStatus.ALLOCATED
        assert outcome.physical_cores == [0, 1, 2, 3]
        assert outcome.logical_cores == [0, 1]
        assert mgr._current_expiry == 123
        # poll is called once with this manager's entry_id and ready=True default.
        _, kwargs = mock_poll.call_args
        args = mock_poll.call_args[0]
        assert mgr.entry_id in args
        assert kwargs.get("caller_id") == mgr._caller_id

    @patch("test.utils.core_lock_client.poll")
    def test_in_queue_maps_to_queued(self, mock_poll):
        """IN_QUEUE wire result maps to QUEUED with position + worst_case_eta."""
        mock_poll.return_value = LockResult(status=LockStatus.IN_QUEUE, position=2, worst_case_eta=42)
        mgr = self._make_manager()
        outcome = mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert outcome.status == AllocationStatus.QUEUED
        assert outcome.position == 2
        assert outcome.worst_case_eta == 42
        assert outcome.logical_cores is None

    @patch("test.utils.core_lock_client.poll")
    def test_draining_distinct_from_queued(self, mock_poll):
        """DRAINING wire result maps to a DRAINING outcome (NOT collapsed to QUEUED/None)."""
        mock_poll.return_value = LockResult(status=LockStatus.DRAINING, position=0, worst_case_eta=7)
        mgr = self._make_manager()
        outcome = mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert outcome.status == AllocationStatus.DRAINING
        assert outcome.status != AllocationStatus.QUEUED
        assert outcome.position == 0
        assert outcome.worst_case_eta == 7

    @patch("test.utils.core_lock_client.poll")
    def test_ready_flag_forwarded(self, mock_poll):
        """acquire(ready=False) forwards ready=False to the poll wire call."""
        mock_poll.return_value = LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1)
        mgr = self._make_manager()
        mgr.acquire(num_logical_cores=1, lnc_config=2, ready=False)
        assert False in mock_poll.call_args[0]


class TestDequeueProbe:
    """Tests for CoreLockManager.dequeue / probe delegation."""

    def _make_manager(self) -> CoreLockManager:
        return CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=MagicMock(),
            host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
        )

    @patch("test.utils.core_lock_client.dequeue")
    def test_dequeue_delegates_with_entry_id(self, mock_dequeue):
        """dequeue() calls the client with this manager's entry_id and caller_id."""
        mock_dequeue.return_value = LockResult(status=LockStatus.RELEASED)
        mgr = self._make_manager()
        mgr.dequeue()
        args = mock_dequeue.call_args[0]
        kwargs = mock_dequeue.call_args[1]
        assert mgr.entry_id in args
        assert kwargs.get("caller_id") == mgr._caller_id


class TestProbe:
    """Tests for CoreLockManager.probe -> AllocationOutcome mapping (metric-neutral)."""

    def _make_manager(self, collector=None) -> CoreLockManager:
        return CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=collector or NoopMetricsCollector(),
            executor=MagicMock(),
            host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
        )

    @patch("test.utils.core_lock_client.probe")
    def test_in_queue_maps_to_queued_with_eta(self, mock_probe):
        """IN_QUEUE probe result maps to a QUEUED outcome carrying worst_case_eta."""
        mock_probe.return_value = LockResult(status=LockStatus.IN_QUEUE, worst_case_eta=42)
        mgr = self._make_manager()
        outcome = mgr.probe(num_logical_cores=1, lnc_config=2)
        assert outcome.status == AllocationStatus.QUEUED
        assert outcome.worst_case_eta == 42

    @patch("test.utils.core_lock_client.probe")
    def test_draining_maps_to_draining_with_eta(self, mock_probe):
        """DRAINING probe result maps to a DRAINING outcome carrying worst_case_eta."""
        mock_probe.return_value = LockResult(status=LockStatus.DRAINING, worst_case_eta=7)
        mgr = self._make_manager()
        outcome = mgr.probe(num_logical_cores=1, lnc_config=2)
        assert outcome.status == AllocationStatus.DRAINING
        assert outcome.worst_case_eta == 7

    @patch("test.utils.core_lock_client.probe")
    def test_probe_is_metric_neutral(self, mock_probe):
        """probe() touches no contention/position counters, never calls _note_enqueued,
        and records no metric (read-only peek joins no queue)."""
        mock_probe.return_value = LockResult(status=LockStatus.IN_QUEUE, worst_case_eta=99)
        collector = MagicMock()
        collector.test_name = "test_probe"
        mgr = self._make_manager(collector)
        before_no_cores = mgr._no_cores_count
        before_contention_ts = mgr._first_contention_ts
        with patch.object(mgr, "_note_enqueued") as spy_enqueue:
            mgr.probe(num_logical_cores=1, lnc_config=2)
        assert mgr._no_cores_count == before_no_cores
        assert mgr._first_contention_ts == before_contention_ts
        spy_enqueue.assert_not_called()
        # No metric of any kind recorded by the probe path.
        collector.record_metric.assert_not_called()
        collector.record_timer.assert_not_called()
        collector.timer.assert_not_called()

    @patch("test.utils.core_lock_client.probe")
    def test_over_capacity_raises_without_client_call(self, mock_probe):
        """A request exceeding host capacity raises InsufficientCoreCountError and never
        reaches the client probe wrapper."""
        mgr = self._make_manager()
        with pytest.raises(InsufficientCoreCountError):
            mgr.probe(num_logical_cores=5, lnc_config=2)  # 10 physical > 8
        mock_probe.assert_not_called()

    @patch("test.utils.core_lock_client.probe")
    def test_default_timeout_matches_acquire_default(self, mock_probe):
        """probe()'s default timeout_seconds equals acquire's default."""
        mock_probe.return_value = LockResult(status=LockStatus.IN_QUEUE, worst_case_eta=1)
        mgr = self._make_manager()
        mgr.probe(num_logical_cores=1, lnc_config=2)
        assert DEFAULT_LOCK_TIMEOUT_SECONDS in mock_probe.call_args[0]

    @patch("test.utils.core_lock_client.probe")
    def test_rpc_error_wrapped_as_lock_acquisition_error(self, mock_probe):
        """A client probe RPC exception is blanket-wrapped as LockAcquisitionError."""
        mock_probe.side_effect = RuntimeError("ssh boom")
        mgr = self._make_manager()
        with pytest.raises(LockAcquisitionError):
            mgr.probe(num_logical_cores=1, lnc_config=2)

    @patch("test.utils.core_lock_client.probe")
    def test_error_status_raises_retryable_lock_acquisition_error(self, mock_probe):
        """An ERROR LockResult raises a retryable LockAcquisitionError carrying the
        host and the 'Lock helper error' prefix (the helper-error branch)."""
        mock_probe.return_value = LockResult(status=LockStatus.ERROR, message="helper boom")
        mgr = self._make_manager()
        with pytest.raises(LockAcquisitionError) as exc_info:
            mgr.probe(num_logical_cores=1, lnc_config=2)
        assert "test-host" in str(exc_info.value)
        assert "Lock helper error" in str(exc_info.value)
        assert exc_info.value.retryable is True

    @patch("test.utils.core_lock_client.probe")
    def test_unexpected_status_raises_lock_acquisition_error(self, mock_probe):
        """A status the read-only probe should never emit (ALLOCATED) falls through
        to the unknown-status branch and raises LockAcquisitionError. This is the
        contract that lets soft-join treat such a result as a swallowed probe
        failure."""
        mock_probe.return_value = LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=123)
        mgr = self._make_manager()
        with pytest.raises(LockAcquisitionError) as exc_info:
            mgr.probe(num_logical_cores=1, lnc_config=2)
        assert "Unexpected lock status" in str(exc_info.value)


# =============================================================================
# Integration test (no hardware): a fake executor invokes the REAL helper verbs
# against a temp locks.json, driven through CoreLockManager.acquire. Proves the
# manager enqueues then commits (QUEUED -> ... -> ALLOCATED) across successive
# acquire(ready=True) calls without any SSH.
# =============================================================================


class _FakeExecutor:
    """Fake RemoteExecutor that runs the deployed helper locally on a temp file."""

    def __init__(self, helpers_file: str, lock_file: str, locks_json: str) -> None:
        self._helpers_file = helpers_file
        self._lock_file = lock_file
        self._locks_json = locks_json

    def call_function(self, func, **kwargs):
        kwargs = dict(kwargs)
        kwargs["helpers_file"] = self._helpers_file
        kwargs["lock_file"] = self._lock_file
        args = list(kwargs["args"])
        args[0] = self._locks_json  # redirect remote locks.json -> temp file
        kwargs["args"] = args
        return func(**kwargs)


def test_manager_acquire_enqueue_then_commit(tmp_path):
    """Manager enqueues on first poll then commits on a later poll (QUEUED -> ALLOCATED)."""
    helpers_file = str(remote_lock_scripts.__file__)
    executor = _FakeExecutor(
        helpers_file,
        str(tmp_path / "atomic_lock"),
        str(tmp_path / "locks.json"),
    )
    mgr = CoreLockManager(
        "test-host",
        total_physical_cores=8,
        collector=NoopMetricsCollector(),
        executor=executor,
        host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
    )

    # Pre-occupy all cores so the contending caller cannot commit -> enqueues.
    blocker = CoreLockManager(
        "test-host",
        total_physical_cores=8,
        collector=NoopMetricsCollector(),
        executor=executor,
        host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
    )
    blocked = None
    for _ in range(5):
        blocked = blocker.acquire(num_logical_cores=4, lnc_config=2)
        if blocked.status == AllocationStatus.ALLOCATED:
            break
    assert blocked is not None and blocked.status == AllocationStatus.ALLOCATED

    # First poll: cores busy -> manager enqueues, reports QUEUED.
    first = mgr.acquire(num_logical_cores=4, lnc_config=2)
    assert first.status == AllocationStatus.QUEUED
    assert first.position == 0

    # Release the blocker's cores; the head's next polls open a window and commit.
    blocker.release(blocked.physical_cores)

    final = None
    for _ in range(5):
        final = mgr.acquire(num_logical_cores=4, lnc_config=2)
        if final.status == AllocationStatus.ALLOCATED:
            break
    assert final is not None
    assert final.status == AllocationStatus.ALLOCATED
    assert final.physical_cores is not None
    assert len(final.physical_cores) == 8


def test_manager_sole_entry_bump_surfaces_nonzero_bump_count(tmp_path, monkeypatch):
    """In-process integration: a sole ready=False head with free cores bumps to the
    empty tail each time its commit window lapses. The head never leaves position 0
    (a 0->0 bump invisible to position-delta detection), yet the explicit wire bump
    signal surfaces a non-zero bump count to the manager -- impossible before this fix.
    """
    helpers_file = str(remote_lock_scripts.__file__)
    executor = _FakeExecutor(
        helpers_file,
        str(tmp_path / "atomic_lock"),
        str(tmp_path / "locks.json"),
    )
    mgr = CoreLockManager(
        "test-host",
        total_physical_cores=8,
        collector=NoopMetricsCollector(),
        executor=executor,
        host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
    )

    # Controllable clock shared by the manager and the in-process helper (both call
    # time.time()), so the COMMIT_WINDOW can lapse without real waiting.
    clock = {"t": 1_700_000_000}
    monkeypatch.setattr(time, "time", lambda: clock["t"])

    # ready=False never fast-paths even though cores are free -> enqueue at head.
    first = mgr.acquire(num_logical_cores=4, lnc_config=2, ready=False)
    assert first.status == AllocationStatus.QUEUED
    assert first.position == 0

    # Next poll opens a commit window for the sole head (no bump yet).
    mgr.acquire(num_logical_cores=4, lnc_config=2, ready=False)

    # Each cycle: advance past COMMIT_WINDOW so the open window lapses; reconcile
    # bumps the sole head to the (empty) tail -- it stays at position 0 -- then
    # reopens a fresh window. The wire bump flag fires each time.
    for _ in range(3):
        clock["t"] += remote_lock_scripts.COMMIT_WINDOW + 1
        out = mgr.acquire(num_logical_cores=4, lnc_config=2, ready=False)
        assert out.status == AllocationStatus.QUEUED
        assert out.position == 0  # sole entry never leaves the head

    assert mgr._bump_count >= 3


# =============================================================================
# Queue-fairness observability metrics. A spy collector
# asserts each metric is emitted with the right name/unit at the right transition.
# =============================================================================


class TestQueueFairnessMetrics:
    """Tests for the queue-fairness metrics emitted by CoreLockManager."""

    def _make_manager(self, collector) -> CoreLockManager:
        return CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=collector,
            executor=MagicMock(),
            host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
        )

    def _spy_collector(self) -> MagicMock:
        collector = MagicMock()
        collector.test_name = "test_fairness"
        return collector

    @patch("test.utils.core_lock_client.poll")
    def test_queue_metrics_after_enqueue(self, mock_poll):
        """A queued-then-committed attempt records position/eta-at-enqueue and QueueWaitTime."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=3, worst_case_eta=42),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        first = mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert first.status == AllocationStatus.QUEUED
        second = mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert second.status == AllocationStatus.ALLOCATED

        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_QUEUE_POSITION_AT_ENQUEUE, 3, "Count")
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_ETA_AT_ENQUEUE, 42, "Seconds")
        # QueueWaitTime is a timer recorded once at the commit.
        wait_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_QUEUE_WAIT_TIME
        ]
        assert len(wait_calls) == 1
        assert wait_calls[0].args[1] >= 0.0

    @patch("test.utils.core_lock_client.poll")
    def test_eta_overrun_recorded_when_wait_exceeds_estimate(self, mock_poll, monkeypatch):
        """When the actual queue wait exceeds the first quoted worst-case ETA, the
        EtaOverrunTime metric records the excess (actual wait - first ETA)."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=3, worst_case_eta=10),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        clock = {"t": 1_700_000_000.0}
        monkeypatch.setattr(time, "time", lambda: clock["t"])
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.QUEUED
        clock["t"] += 25.0  # simulated inter-poll waits: actual wait 25s vs 10s ETA
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.ALLOCATED
        overrun_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_ETA_OVERRUN_TIME
        ]
        assert len(overrun_calls) == 1
        assert overrun_calls[0].args[1] == pytest.approx(15.0)

    @patch("test.utils.core_lock_client.poll")
    def test_eta_overrun_zero_when_within_estimate(self, mock_poll, monkeypatch):
        """When the actual queue wait is shorter than the first quoted ETA, the
        EtaOverrunTime metric is clamped to zero (never negative)."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=3, worst_case_eta=30),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        clock = {"t": 1_700_000_000.0}
        monkeypatch.setattr(time, "time", lambda: clock["t"])
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.QUEUED
        clock["t"] += 5.0  # actual wait 5s, well within the 30s ETA
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.ALLOCATED
        overrun_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_ETA_OVERRUN_TIME
        ]
        assert len(overrun_calls) == 1
        assert overrun_calls[0].args[1] == 0.0

    @patch("test.utils.core_lock_client.poll")
    def test_bump_count_increments_on_explicit_signal(self, mock_poll):
        """An explicit ``bumped=True`` signal is counted as a bump."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        for _ in range(3):
            mgr.acquire(num_logical_cores=1, lnc_config=2)
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_BUMP_COUNT, 1, "Count")

    @patch("test.utils.core_lock_client.poll")
    def test_three_sole_entry_bumps_emit_warning(self, mock_poll, caplog):
        """Three explicit bump signals on a SOLE entry (position stays 0) reach the
        >=3 livelock warning. The 0->0 sole-entry bump was invisible to the old
        position-delta heuristic, so this scenario was impossible to catch before.
        """
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump 1
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump 2
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump 3 -> warn
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        with caplog.at_level("WARNING"):
            for _ in range(4):
                mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert any("bumped 3 times" in r.message for r in caplog.records)
        assert mgr._bump_count == 3

    @patch("test.utils.core_lock_client.poll")
    def test_reenqueue_without_bump_signal_not_counted(self, mock_poll, caplog):
        """A position increase WITHOUT an explicit bump signal (e.g. prune ->
        re-enqueue) must NOT be counted as a bump or warn."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=2, worst_case_eta=1),  # re-enqueued, not bumped
            LockResult(status=LockStatus.IN_QUEUE, position=3, worst_case_eta=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        with caplog.at_level("WARNING"):
            for _ in range(3):
                mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert not any("bump livelock" in r.message for r in caplog.records)
        assert mgr._bump_count == 0

    @patch("test.utils.core_lock_client.poll")
    def test_no_warning_below_three_bumps(self, mock_poll, caplog):
        """Fewer than 3 bumps must NOT emit the livelock warning."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump 1
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump 2
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        with caplog.at_level("WARNING"):
            for _ in range(3):
                mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert not any("bump livelock" in r.message for r in caplog.records)
        assert mgr._bump_count == 2

    @patch("test.utils.core_lock_client.poll")
    def test_reenqueue_signal_counted_distinctly_from_bump(self, mock_poll):
        """An explicit ``re_enqueued=True`` signal increments the re-enqueue count
        and emits CORE_LOCK_REENQUEUE_COUNT, but NOT the bump count."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=2, worst_case_eta=1, re_enqueued=True),  # re-enqueue
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        for _ in range(3):
            mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert mgr._reenqueue_count == 1
        assert mgr._bump_count == 0
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_REENQUEUE_COUNT, 1, "Count")
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_BUMP_COUNT, 0, "Count")

    @patch("test.utils.core_lock_client.poll")
    def test_bump_signal_does_not_increment_reenqueue(self, mock_poll):
        """An explicit ``bumped=True`` signal increments the bump count but NOT the
        re-enqueue count -- the two signals are mutually distinct."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1, bumped=True),  # bump
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        for _ in range(3):
            mgr.acquire(num_logical_cores=1, lnc_config=2)
        assert mgr._bump_count == 1
        assert mgr._reenqueue_count == 0
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_BUMP_COUNT, 1, "Count")
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_REENQUEUE_COUNT, 0, "Count")

    @patch("test.utils.core_lock_client.poll")
    def test_drain_wait_time_recorded(self, mock_poll):
        """DrainWaitTime is recorded by record_contention_metrics after a DRAINING poll."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.DRAINING, position=0, worst_case_eta=7),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        mgr.acquire(num_logical_cores=1, lnc_config=2)
        mgr.acquire(num_logical_cores=1, lnc_config=2)
        mgr.record_contention_metrics()
        drain_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_DRAIN_WAIT_TIME
        ]
        assert len(drain_calls) == 1
        assert drain_calls[0].args[1] >= 0.0

    @patch("test.utils.core_lock_client.poll")
    def test_in_queue_does_not_set_drain_ts(self, mock_poll):
        """An IN_QUEUE poll must NOT arm the drain timer: after an IN_QUEUE poll
        followed by an ALLOCATED commit, DrainWaitTime is recorded as 0.0 while the
        no-cores count and contention wait are still recorded."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=7),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        mgr.acquire(num_logical_cores=1, lnc_config=2)
        mgr.acquire(num_logical_cores=1, lnc_config=2)
        mgr.record_contention_metrics()
        drain_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_DRAIN_WAIT_TIME
        ]
        contention_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_CONTENTION_WAIT_TIME
        ]
        # IN_QUEUE never arms the drain timer -> DrainWaitTime is exactly 0.0.
        assert len(drain_calls) == 1
        assert drain_calls[0].args[1] == 0.0
        # The contention path is still recorded for the queued poll.
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_NO_CORES_COUNT, 1, "Count")
        assert len(contention_calls) == 1
        assert contention_calls[0].args[1] >= 0.0

    @patch("test.utils.core_lock_client.poll")
    def test_contention_wait_is_wall_span_including_inter_poll_sleeps(self, mock_poll, monkeypatch):
        """ContentionWaitTime must reflect the full wall span from the first contended
        poll to the terminal (commit), INCLUDING the inter-poll sleeps that elapse
        in the host loop between acquire() calls -- not the summed per-poll RPC
        durations (which are ~0 here since the poll is mocked)."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=2, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=1, worst_case_eta=1),
            LockResult(status=LockStatus.IN_QUEUE, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        period = remote_lock_scripts.POLL_PERIOD
        clock = {"t": 1_700_000_000.0}
        monkeypatch.setattr(time, "time", lambda: clock["t"])
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        # 3 contended polls spaced POLL_PERIOD apart (the mocked RPC takes 0 wall
        # time; the clock only advances during the simulated inter-poll sleep).
        for _ in range(3):
            mgr.acquire(num_logical_cores=1, lnc_config=2)
            clock["t"] += period
        mgr.acquire(num_logical_cores=1, lnc_config=2)  # ALLOCATED commit
        mgr.record_contention_metrics()
        contention_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_CONTENTION_WAIT_TIME
        ]
        assert len(contention_calls) == 1
        # Span = first contended poll -> terminal = 3 * POLL_PERIOD. The summed
        # RPC duration would be ~0, which this assertion explicitly rejects.
        assert contention_calls[0].args[1] == pytest.approx(3 * period)
        assert contention_calls[0].args[1] > period

    @patch("test.utils.core_lock_client.poll")
    def test_drain_wait_is_wall_span_including_inter_poll_sleeps(self, mock_poll, monkeypatch):
        """DrainWaitTime must reflect the full wall span across draining polls,
        including inter-poll sleeps, not the summed per-poll RPC durations."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.DRAINING, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.DRAINING, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.DRAINING, position=0, worst_case_eta=1),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        period = remote_lock_scripts.POLL_PERIOD
        clock = {"t": 1_700_000_000.0}
        monkeypatch.setattr(time, "time", lambda: clock["t"])
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        for _ in range(3):
            mgr.acquire(num_logical_cores=1, lnc_config=2)
            clock["t"] += period
        mgr.acquire(num_logical_cores=1, lnc_config=2)  # ALLOCATED commit
        mgr.record_contention_metrics()
        drain_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_DRAIN_WAIT_TIME
        ]
        assert len(drain_calls) == 1
        assert drain_calls[0].args[1] == pytest.approx(3 * period)
        assert drain_calls[0].args[1] > period

    @patch("test.utils.core_lock_client.poll")
    def test_never_contended_fast_path_records_zero_wait_spans(self, mock_poll, monkeypatch):
        """A first-poll ALLOCATED commit was never contended/draining, so both wait
        spans must be exactly 0 even when wall time elapses afterwards."""
        mock_poll.return_value = LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1)
        clock = {"t": 1_700_000_000.0}
        monkeypatch.setattr(time, "time", lambda: clock["t"])
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.ALLOCATED
        clock["t"] += 123.0  # wall time passes, but never contended
        mgr.record_contention_metrics()
        contention_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_CONTENTION_WAIT_TIME
        ]
        drain_calls = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_DRAIN_WAIT_TIME
        ]
        assert len(contention_calls) == 1
        assert len(drain_calls) == 1
        assert contention_calls[0].args[1] == 0.0
        assert drain_calls[0].args[1] == 0.0

    @patch("test.utils.core_lock_client.poll")
    def test_abandon_flush_records_queue_and_contention_metrics(self, mock_poll):
        """F11: record_contention_metrics on an abandoned (never-ALLOCATED) attempt
        flushes BOTH the queue-fairness commit metrics and the contention
        counters, so a starved attempt is observable."""
        mock_poll.return_value = LockResult(status=LockStatus.IN_QUEUE, position=4, worst_case_eta=99)
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        # Two queued polls (no commit), then abandon -> flush.
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.QUEUED
        assert mgr.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.QUEUED
        mgr.record_contention_metrics()
        # Contention counters.
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_NO_CORES_COUNT, 2, "Count")
        # Queue-fairness commit metrics flushed on the abandon path too.
        collector.record_metric.assert_any_call(MetricName.CORE_LOCK_QUEUE_POSITION_AT_ENQUEUE, 4, "Count")
        queue_wait = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_QUEUE_WAIT_TIME
        ]
        contention_wait = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_CONTENTION_WAIT_TIME
        ]
        assert len(queue_wait) == 1
        assert len(contention_wait) == 1

    @patch("test.utils.core_lock_client.poll")
    def test_commit_metrics_not_double_emitted_when_contention_flush_follows_commit(self, mock_poll):
        """On the success path, the commit-time flush and the host-level
        record_contention_metrics call must not double-emit the queue metrics."""
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=3, worst_case_eta=42),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        collector = self._spy_collector()
        mgr = self._make_manager(collector)
        mgr.acquire(num_logical_cores=1, lnc_config=2)
        mgr.acquire(num_logical_cores=1, lnc_config=2)  # ALLOCATED -> commit flush
        mgr.record_contention_metrics()  # success-path host-level flush
        queue_wait = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_QUEUE_WAIT_TIME
        ]
        bump = [c for c in collector.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_BUMP_COUNT]
        assert len(queue_wait) == 1  # emitted exactly once despite two flush points
        assert len(bump) == 1


# =============================================================================
# Manager owner identity + per-instance counter isolation (F2 support).
#
# The host layer binds its cached manager's validity to the manager's owning
# collector (exposed read-only via the `collector` property) so a fresh manager
# is built whenever the collector changes. These tests pin that property and the
# fact that distinct managers carry independent entry_ids and fairness counters.
# =============================================================================


class TestManagerOwnerIdentity:
    """CoreLockManager.collector property and per-instance counter isolation."""

    def _make_manager(self, collector) -> CoreLockManager:
        return CoreLockManager(
            "test-host",
            total_physical_cores=8,
            collector=collector,
            executor=MagicMock(),
            host_locking_version=DEFAULT_LOCKING_PROTOCOL_VERSION,
        )

    def test_collector_property_returns_owning_collector(self):
        """The read-only collector property exposes the manager's owner."""
        collector = NoopMetricsCollector()
        mgr = self._make_manager(collector)
        assert mgr.collector is collector

    def test_distinct_managers_have_distinct_entry_ids(self):
        """Each manager mints its own per-attempt entry_id."""
        mgr1 = self._make_manager(NoopMetricsCollector())
        mgr2 = self._make_manager(NoopMetricsCollector())
        assert mgr1.entry_id != mgr2.entry_id

    @patch("test.utils.core_lock_client.poll")
    def test_fresh_manager_counters_start_at_zero(self, mock_poll):
        """A brand-new manager emits per-attempt (not cumulative) contention counters.

        Simulates a prior attempt accumulating a rejection, then proves a second
        manager bound to its own collector reports a fresh zero count.
        """
        mock_poll.side_effect = [
            LockResult(status=LockStatus.IN_QUEUE, position=1, worst_case_eta=5),
            LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1),
        ]
        col1 = MagicMock()
        col1.test_name = "attempt-1"
        mgr1 = self._make_manager(col1)
        assert mgr1.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.QUEUED
        assert mgr1.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.ALLOCATED
        mgr1.record_contention_metrics()
        col1.record_metric.assert_any_call(MetricName.CORE_LOCK_NO_CORES_COUNT, 1, "Count")

        # A fresh manager (new attempt) starts counters at zero and attributes
        # to its own collector only.
        mock_poll.side_effect = [LockResult(status=LockStatus.ALLOCATED, cores=[0, 1], expiry=1)]
        col2 = MagicMock()
        col2.test_name = "attempt-2"
        mgr2 = self._make_manager(col2)
        assert mgr2.acquire(num_logical_cores=1, lnc_config=2).status == AllocationStatus.ALLOCATED
        mgr2.record_contention_metrics()
        col2.record_metric.assert_any_call(MetricName.CORE_LOCK_NO_CORES_COUNT, 0, "Count")
        # The first collector never received the second attempt's metrics.
        no_cores_for_col1 = [
            c for c in col1.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_NO_CORES_COUNT
        ]
        assert all(c.args[1] == 1 for c in no_cores_for_col1)
