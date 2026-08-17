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
"""Unit tests for host_management retry logic and error reporting."""

import contextlib
import json
import os
import subprocess
import tempfile
from contextlib import closing
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from paramiko import SSHException

from test.utils.common_dataclasses import Platforms, ResolvedHost, TargetHost
from test.utils.core_reset_strategy import CoreResetStrategy
from test.utils.exceptions import FleetEmptyError, InferenceException, LocalExecutionException, TimeoutException
from test.utils.host_management import Host, HostManager, LocalHost, detect_local_neuron_devices
from test.utils.host_state import HostRecord, HostState, HostStateStore, OwnerHostStateStore
from test.utils.metrics_collector import MetricName


def _initialized_store(base_dir, target_hosts, cores_by_alias=None):
    """Build a HostStateStore over base_dir and initialize it from a TargetHost
    list — the bootstrap the factory (make_host_manager) does — and return it to
    inject into a HostManager. An empty list leaves the store untouched (local /
    plugin-provisioned, where the controller initializes instead).

    ``cores_by_alias`` (optional) stamps each host's probed physical-core capacity so
    capacity-based routing has real counts to rank by; unspecified hosts default to a
    uniform 64 cores so plain balancing tests need not care about capacity."""
    store = HostStateStore(base_dir)
    if target_hosts:
        store.initialize([ResolvedHost(ssh_host=th.ssh_host, host_type=th.host_type) for th in target_hosts])
        cores = cores_by_alias or {th.ssh_host: 64 for th in target_hosts}
        store.set_physical_cores(cores)
    return store


def _poison_platform(base_dir, platform, poisoned=True):
    """Poison (or un-poison) ``platform`` in the state file at ``base_dir``.

    Poisoning is a controller/refresher action, so it goes through an OwnerHostStateStore —
    the manager's own store is the worker-side base store, which (correctly) can't poison.
    A worker's claim then observes the poison via the shared file, exactly as in a real run.
    """
    OwnerHostStateStore(base_dir).set_poisoned(platform, poisoned)


class TestHostManagerRetryErrorReporting:
    """Test that host errors are captured and reported when all hosts fail."""

    def setup_method(self):
        self._patchers = []

    def teardown_method(self):
        # Reverse order: patches stacked on the same target must be undone LIFO, or the
        # target is left pointing at an intermediate mock instead of the real object.
        for p in reversed(self._patchers):
            p.stop()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_host_manager(self, temp_dir):
        """Create a HostManager with mocked hosts for testing."""
        return self._build_manager_with_hosts(temp_dir, num_hosts=2)

    def test_error_details_included_when_all_retries_exhausted(self, mock_host_manager):
        """Test that error details from each host are included in final exception."""
        mock_collector = MagicMock()

        with pytest.raises(InferenceException) as exc_info:
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collective_ranks=1,
                    lnc_config=1,
                    collector=mock_collector,
                    connection_failure_cap=2,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        # Simulate different errors on each host
                        host_id = host.get_host_id()
                        if host_id == "host1":
                            raise TimeoutError("Connection timed out after 30 seconds")
                        else:
                            raise SSHException("Authentication failed")

        # Verify error message contains details from both hosts
        error_message = str(exc_info.value)
        print("\n=== Test 1: All Retries Exhausted ===")
        print(error_message)
        print("=" * 50)
        assert "Connection error after 2 attempts" in error_message
        assert "host1" in error_message or "host2" in error_message
        assert "Errors from each host:" in error_message

    def test_error_details_included_when_no_hosts_available(self, mock_host_manager):
        """Test that previous errors are included when __get_host_assignment__ raises."""
        mock_collector = MagicMock()

        # First, manually fail host1 with a specific error
        with pytest.raises(InferenceException) as exc_info:
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collective_ranks=1,
                    lnc_config=1,
                    collector=mock_collector,
                    connection_failure_cap=3,
                )
            ) as host_generator:
                attempt = 0
                for host_attempt in host_generator:
                    with host_attempt as host:
                        attempt += 1
                        host_id = host.get_host_id()
                        # Fail both hosts - after 2 failures, no hosts will be available
                        raise OSError(f"Network unreachable on {host_id}")

        # Verify error message contains the enriched details
        error_message = str(exc_info.value)
        print("\n=== Test 2: No Hosts Available ===")
        print(error_message)
        print("=" * 50)
        assert "Errors from" in error_message
        assert "OSError" in error_message or "Network unreachable" in error_message

    def test_single_host_failure_shows_specific_error(self, temp_dir):
        """single host fails, error details should be shown."""
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=1)
        mock_collector = MagicMock()

        with pytest.raises(InferenceException) as exc_info:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collective_ranks=1,
                    lnc_config=1,
                    collector=mock_collector,
                    connection_failure_cap=3,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:  # noqa: F841
                        raise TimeoutError("SSH connection timed out during inference")

        # Verify the specific error is included
        error_message = str(exc_info.value)
        print("\n=== Test 3: Single Host Failure ===")
        print(error_message)
        print("=" * 50)
        assert "host1" in error_message
        assert "TimeoutError" in error_message
        assert "SSH connection timed out during inference" in error_message
        assert "Errors from previous host attempts:" in error_message

    def test_patience_rotation_is_transient_does_not_mark_failed(self, mock_host_manager):
        """A QueuePatienceRotation (busy FIFO queue) is transient: the host is NOT marked
        unavailable and stays eligible, and the loop keeps rotating until it succeeds elsewhere."""
        from test.utils.exceptions import QueuePatienceRotation

        mock_collector = MagicMock()

        with patch.object(
            mock_host_manager, "mark_host_unavailable", wraps=mock_host_manager.mark_host_unavailable
        ) as mark_unavailable:
            first_host_id = None
            successful_host_id = None
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collective_ranks=1,
                    lnc_config=1,
                    collector=mock_collector,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        host_id = host.get_host_id()
                        if first_host_id is None:
                            first_host_id = host_id
                            # Busy queue on the first host: transient, not a failure.
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        if host_id == first_host_id:
                            # Keep rotating until we land on a DIFFERENT host.
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        successful_host_id = host_id

        # The busy host was NEVER marked unavailable and stayed eligible; the loop
        # advanced to a different host and succeeded.
        assert mark_unavailable.call_count == 0
        assert mock_host_manager.get_failed_host_count() == 0
        assert first_host_id is not None
        assert successful_host_id is not None
        assert successful_host_id != first_host_id

    def test_acquisition_runs_until_deadline_not_attempt_count(self, temp_dir):
        """Under sustained busy queues the loop rotates until a WALL-CLOCK deadline
        (not a fixed attempt count), never marks hosts failed, and finally raises a
        DISTINCT deadline diagnostic."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=2)
        mock_collector = MagicMock()

        clock = {"t": 1_700_000_000.0}

        def fake_time():
            return clock["t"]

        def fake_sleep(dt):
            clock["t"] += dt

        deadline_seconds = 100.0
        backoff = 5.0
        rotations = 0
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            rotations += 1
                            raise QueuePatienceRotation("queue busy")

        msg = str(exc_info.value)
        # Far more rotations than the old fixed cap of 3.
        assert rotations > 3
        assert rotations >= int(deadline_seconds / backoff) - 1
        # No host was ever marked unavailable for being merely busy.
        assert manager.get_failed_host_count() == 0
        # The diagnostic is DISTINCT from the other two terminal messages.
        assert "deadline" in msg.lower()
        assert "No available hosts" not in msg
        assert "Connection error after" not in msg

    def test_connection_failures_still_capped_and_mark_unavailable(self, temp_dir):
        """Genuine connection failures still mark hosts unavailable and stop at the cap."""
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=5)
        mock_collector = MagicMock()

        with patch.object(manager, "mark_host_unavailable", wraps=manager.mark_host_unavailable) as mark_unavail:
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        connection_failure_cap=3,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            raise OSError("network unreachable")

        msg = str(exc_info.value)
        assert "Connection error after 3 attempts" in msg
        assert mark_unavail.call_count == 3
        assert manager.get_failed_host_count() == 3

    def test_host_rotation_count_recorded_on_failure(self, mock_host_manager):
        """Each host rotation records CoreLockHostRotationCount as a delta of 1.0."""
        from test.utils.metrics_collector import MetricName

        mock_collector = MagicMock()
        successful_host_id = None
        iteration = 0
        with closing(
            mock_host_manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collective_ranks=1,
                lnc_config=1,
                collector=mock_collector,
                connection_failure_cap=2,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    iteration += 1
                    if iteration == 1:
                        # Fail the first host, succeed on the next.
                        raise TimeoutError("SSH timed out")
                    successful_host_id = host.get_host_id()
        assert successful_host_id is not None

        rotation_calls = [
            c
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        assert len(rotation_calls) == 1
        assert rotation_calls[0].args[1] == 1.0

    def _build_manager_with_hosts(self, temp_dir, num_hosts):
        """Build a HostManager backed by ``num_hosts`` mocked TRN2 hosts.

        Hosts are seeded into the state store; SshHost is patched so the manager
        builds alias-carrying mocks lazily on assignment rather than opening real
        connections.
        """
        aliases = [f"host{i}" for i in range(1, num_hosts + 1)]
        target_hosts = [TargetHost(ssh_host=a, host_type=Platforms.TRN2) for a in aliases]

        def fake_make(alias, **_kwargs):
            return MagicMock(get_host_id=MagicMock(return_value=alias), connection=MagicMock())

        patcher = patch("test.utils.host_management.SshHost", side_effect=fake_make)
        patcher.start()
        self._patchers.append(patcher)
        return HostManager(
            state_store=_initialized_store(temp_dir, target_hosts),
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=False,
        )

    def test_host_rotation_count_datapoints_sum_to_n(self, temp_dir):
        """N rotations must emit N datapoints of 1.0 that sum to N (not N(N+1)/2)."""
        from test.utils.metrics_collector import MetricName

        n_rotations = 3
        # Need enough distinct hosts: each failure marks a host failed and
        # rotates to a fresh one, so num_hosts must exceed n_rotations.
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=n_rotations + 1)

        mock_collector = MagicMock()
        successful_host_id = None
        iteration = 0
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collective_ranks=1,
                lnc_config=1,
                collector=mock_collector,
                connection_failure_cap=n_rotations + 1,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    iteration += 1
                    if iteration <= n_rotations:
                        # Fail the first ``n_rotations`` hosts, succeed after.
                        raise TimeoutError("SSH timed out")
                    successful_host_id = host.get_host_id()
        assert successful_host_id is not None

        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        # One datapoint per rotation, each exactly 1.0 ...
        assert len(rotation_values) == n_rotations
        assert all(v == 1.0 for v in rotation_values)
        # ... so EMF sum/avg aggregations reflect the true rotation count.
        assert sum(rotation_values) == n_rotations
        # Guard against a regression to the cumulative-count emission.
        assert sum(rotation_values) != n_rotations * (n_rotations + 1) / 2

    def test_core_lock_exhaustion_marks_host_unavailable(self, mock_host_manager):
        # Core-lock acquisition that keeps erroring on a host surfaces as a plain OSError
        # (see SshHost.get_core_allocation). It is routed through the same connection-fault
        # branch: the host is marked unavailable (recoverable) and the attempt rotates on.
        mock_collector = MagicMock()

        with pytest.raises(InferenceException, match="Connection error after 2 attempts"):
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    collective_ranks=1,
                    lnc_config=1,
                    connection_failure_cap=2,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        raise OSError(
                            f"[{host.get_host_id()}] Lock acquisition failed after 10 consecutive retryable errors"
                        )

        # Every host hit was marked unavailable (recoverable), not permanently removed.
        rows = {h.resolved.ssh_host: h for h in mock_host_manager.state_store.read().hosts}
        assert rows["host1"].available is False
        assert rows["host2"].available is False

    def test_unavailable_host_is_revived_by_resolution(self, mock_host_manager):
        # Recoverable end-to-end: a host taken out for a host-level failure comes back
        # when a later resolution lists it as good-to-use (no sticky removal).
        mock_host_manager.mark_host_unavailable("host1")
        # A resolution is a controller/refresher action, so it reconciles through an owner
        # store over the same file; the worker's base store (on the manager) then reads it.
        OwnerHostStateStore(mock_host_manager.state_store.base_dir).apply_resolved(
            [ResolvedHost(ssh_host="host1", host_type=Platforms.TRN2)]
        )
        row = {h.resolved.ssh_host: h for h in mock_host_manager.state_store.read().hosts}["host1"]
        assert row.available is True


class _PatienceRotatingHost(Host):
    """Concrete ``Host`` whose ``get_core_allocation`` always raises
    ``QueuePatienceRotation``. Because the real ``Host.execute_command`` opens
    ``with self.get_core_allocation(...)`` with no try/except, the exception
    propagates out of ``execute_command`` -- exactly the production path that the
    retry loop's ``context_manager_wrapper`` must treat as transient.
    """

    def __init__(self, host_id: str):
        super().__init__(CoreResetStrategy(exclusive_run=False))
        self._host_id = host_id

    def get_host_id(self) -> str:
        return self._host_id

    def get_core_allocation(
        self,
        collector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = 0,
    ):
        from test.utils.exceptions import QueuePatienceRotation

        raise QueuePatienceRotation(f"queue ETA exceeds patience on {self._host_id}")

    # Remaining abstract methods are never reached in these tests.
    def _get_debug_output_dir(self, target_directory: str) -> str:
        return target_directory

    def _run_command(self, command, target_directory, neuron_env, collector) -> str:
        return ""

    def _collect_artifacts(self, target_directory, stdout, get_list_of_files_to_copy, collector) -> str:
        return target_directory

    def prepare_host(self, target_directory, collector, skip_remote_cleanup=False, force_local_cleanup=False):
        return contextlib.nullcontext()

    def _run_post_lock_command(self, command, target_directory, collector) -> str:
        return ""

    def get_neuron_device_info(self):
        return []

    def get_total_physical_cores(self) -> int:
        return 64


class TestHostRotationUntilDeadlineIntegration(TestHostManagerRetryErrorReporting):
    """Seam/regression test exercising the two combined behaviors TOGETHER through the
    real ``HostManager`` host-selection (``__get_host_assignment__`` + the state store's
    availability tracking) machinery -- only the wall clock and the per-attempt outcome
    are faked.

    1. ``QueuePatienceRotation`` (busy FIFO queue ETA exceeds patience) is transient: the
       host is NOT marked unavailable, NOT counted toward the connection-failure cap, the
       rotation metric is recorded, and rotation continues across hosts until a wall-clock
       deadline.
    2. Genuine connection failures still mark hosts unavailable and stop at
       ``connection_failure_cap``.
    """

    def _patched_clock(self):
        """Return (time_fn, sleep_fn) over an in-test fake monotonic clock."""
        clock = {"t": 1_700_000_000.0}

        def fake_time():
            return clock["t"]

        def fake_sleep(dt):
            clock["t"] += dt

        return fake_time, fake_sleep

    @staticmethod
    def _seed_hosts(manager, make_host):
        """Pre-populate the manager's lazy host cache with concrete hosts keyed by the
        store's aliases, so ``__get_host_assignment__`` returns these instances instead
        of lazily building a mock. ``make_host(alias)`` builds the host for each alias."""
        aliases = [h.resolved.ssh_host for h in manager.state_store.read().hosts]
        manager.target_hosts = {a: make_host(a) for a in aliases}
        return aliases

    def test_a_sustained_busy_reselects_hosts_and_terminates_on_deadline(self, temp_dir):
        """(a) Every attempt is busy: the loop rotates far past ``num_hosts`` (re-selecting
        hosts that stayed eligible), never marks a host failed, never raises
        "No available hosts", and terminates with the DISTINCT deadline diagnostic."""
        from test.utils.exceptions import QueuePatienceRotation

        num_hosts = 3
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        mock_collector = MagicMock()

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 100.0
        backoff = 5.0
        expected_min_rotations = int(deadline_seconds / backoff) - 1  # ~19

        attempted_hosts = []
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            attempted_hosts.append(host.get_host_id())
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        rotations = len(attempted_hosts)
        # Rotated far more than the host count -> hosts were RE-selected, proving they
        # stayed eligible the whole time.
        assert rotations > num_hosts
        assert rotations >= expected_min_rotations
        # Some host must have been chosen more than once (re-selection).
        assert max(attempted_hosts.count(h) for h in set(attempted_hosts)) > 1
        # Only real hosts from the pool were ever selected.
        assert set(attempted_hosts).issubset(set(manager.target_hosts.keys()))
        # Merely-busy hosts were NEVER excluded.
        assert manager.get_failed_host_count() == 0
        # Distinct deadline termination, not the other two terminal messages.
        msg = str(exc_info.value)
        assert "deadline" in msg.lower()
        assert "No available hosts" not in msg
        assert "Connection error after" not in msg

    def test_b_connection_failures_exclude_hosts_and_stop_at_cap(self, temp_dir):
        """(b) Genuine connection failures still exclude hosts (mark them unavailable) and
        stop at ``connection_failure_cap`` with the connection-error message."""
        connection_failure_cap = 3
        num_hosts = 3
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        mock_collector = MagicMock()

        with patch.object(manager, "mark_host_unavailable", wraps=manager.mark_host_unavailable) as mark_failed:
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        connection_failure_cap=connection_failure_cap,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            raise OSError("network unreachable")

        msg = str(exc_info.value)
        assert f"Connection error after {connection_failure_cap} attempts" in msg
        assert "deadline" not in msg.lower()
        assert mark_failed.call_count == connection_failure_cap
        assert manager.get_failed_host_count() == connection_failure_cap
        all_rows = manager.state_store.read().hosts
        unavailable = {h.resolved.ssh_host for h in all_rows if not h.available}
        assert unavailable.issubset({h.resolved.ssh_host for h in all_rows})

    def test_c_rotation_metric_recorded_once_per_patience_rotation(self, temp_dir):
        """(c) In the sustained-busy scenario, ``CoreLockHostRotationCount`` is recorded
        exactly once per patience rotation, each datapoint == 1.0."""
        from test.utils.exceptions import QueuePatienceRotation
        from test.utils.metrics_collector import MetricName

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=3)
        mock_collector = MagicMock()

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 100.0
        backoff = 5.0

        rotations = 0
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException):
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            rotations += 1
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        # One datapoint per rotation, each exactly 1.0, and no failure metrics.
        assert len(rotation_values) == rotations
        assert all(v == 1.0 for v in rotation_values)
        assert manager.get_failed_host_count() == 0

    def test_d_busy_host_is_reselectable_and_loop_can_succeed(self, temp_dir):
        """(d) A host that patience-rotates stays eligible; a later attempt succeeds and the
        context manager exits cleanly without any host being marked failed."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=3)
        mock_collector = MagicMock()

        attempted_hosts = []
        successful_host_id = None
        with patch.object(manager, "mark_host_unavailable", wraps=manager.mark_host_unavailable) as mark_failed:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collective_ranks=1,
                    lnc_config=1,
                    collector=mock_collector,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        host_id = host.get_host_id()
                        attempted_hosts.append(host_id)
                        if len(attempted_hosts) <= 2:
                            # Busy queue on the first couple of attempts: transient.
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        # Notify success simply by not raising.
                        successful_host_id = host_id

        assert successful_host_id is not None
        assert len(attempted_hosts) >= 3
        # No host was excluded for being merely busy; the loop ended on success.
        assert mark_failed.call_count == 0
        assert manager.get_failed_host_count() == 0

    def test_e_patience_rotation_through_execute_command_is_transient(self, temp_dir):
        """Gap #1: drive ``QueuePatienceRotation`` THROUGH the real
        ``Host.execute_command``/``get_core_allocation`` (not raised directly in the
        consumer body). The first two attempts call ``execute_command`` -- whose
        ``get_core_allocation`` raises -- and the propagated exception must be treated
        as transient (no host marked failed); a later attempt then succeeds."""
        from test.utils.metrics_collector import MetricName

        num_hosts = 3
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        # Seed concrete hosts that exercise the real execute_command ->
        # get_core_allocation propagation path (keyed by the store's aliases).
        self._seed_hosts(manager, _PatienceRotatingHost)
        mock_collector = MagicMock()

        attempted_hosts = []
        successful_host_id = None
        with patch.object(manager, "mark_host_unavailable", wraps=manager.mark_host_unavailable) as mark_failed:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collective_ranks=1,
                    lnc_config=1,
                    collector=mock_collector,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        attempted_hosts.append(host.get_host_id())
                        if len(attempted_hosts) <= 2:
                            # Real production path: the exception originates inside
                            # get_core_allocation and propagates out of execute_command.
                            host.execute_command("cmd", "/tmp", mock_collector, 1, 2)
                        else:
                            successful_host_id = host.get_host_id()

        # The loop advanced past the transient rotations and succeeded.
        assert successful_host_id is not None
        assert len(attempted_hosts) >= 3
        # Transient: no host was marked failed and none excluded.
        assert mark_failed.call_count == 0
        assert manager.get_failed_host_count() == 0
        # The two propagated rotations were recorded as rotation datapoints.
        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        assert len(rotation_values) == 2
        assert all(v == 1.0 for v in rotation_values)

    def test_f_mixed_patience_and_connection_failures_only_genuine_count_to_cap(self, temp_dir):
        """Gap #2: interleave patience rotations and genuine connection failures in one
        acquisition. Only genuine failures count toward ``connection_failure_cap`` and
        mark hosts failed, while the rotation metric is recorded for ALL non-success
        attempts; termination is the connection-error message, NOT the deadline."""
        from test.utils.exceptions import QueuePatienceRotation
        from test.utils.metrics_collector import MetricName

        connection_failure_cap = 3
        # Enough hosts so genuine-failure exclusions never exhaust the pool.
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=6)
        mock_collector = MagicMock()

        # 2 patience rotations + 3 genuine failures (interleaved). Terminates on the
        # 3rd genuine failure (the cap), so all 5 outcomes are consumed.
        outcomes = ["patience", "oserror", "patience", "oserror", "oserror"]
        idx = {"i": 0}

        with patch.object(manager, "mark_host_unavailable", wraps=manager.mark_host_unavailable) as mark_failed:
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        connection_failure_cap=connection_failure_cap,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            kind = outcomes[idx["i"]]
                            idx["i"] += 1
                            if kind == "patience":
                                raise QueuePatienceRotation("queue ETA exceeds patience")
                            raise OSError("network unreachable")

        msg = str(exc_info.value)
        assert f"Connection error after {connection_failure_cap} attempts" in msg
        assert "deadline" not in msg.lower()
        # Only the 3 genuine failures marked hosts failed / counted toward the cap.
        assert mark_failed.call_count == connection_failure_cap
        assert manager.get_failed_host_count() == connection_failure_cap
        # All 5 non-success attempts (2 patience + 3 genuine) recorded a rotation datapoint.
        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        assert len(rotation_values) == len(outcomes)
        assert all(v == 1.0 for v in rotation_values)

    def test_g_patience_branch_sleeps_with_configured_backoff(self, temp_dir):
        """Gap #3: the patience branch must call ``time.sleep(backoff_seconds)`` with the
        configured backoff (guards against a busy-loop if the backoff is removed). Patch
        ``time.sleep`` with a mock and advance ``time.time`` independently so the loop
        still terminates at the deadline."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_hosts(temp_dir, num_hosts=3)
        mock_collector = MagicMock()

        backoff = 5.0
        deadline_seconds = 10.0
        # time.time advances on its own (independent of the mocked sleep) so the
        # wall-clock deadline is reached regardless of whether sleep blocks.
        base = 1_700_000_000.0
        ticks = iter([base + i * 3.0 for i in range(1000)])
        sleep_mock = MagicMock()

        with (
            patch("test.utils.host_management.time.time", lambda: next(ticks)),
            patch("test.utils.host_management.time.sleep", sleep_mock),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:  # noqa: F841
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        assert "deadline" in str(exc_info.value).lower()
        # The patience branch backed off at least once, always with the configured value.
        assert sleep_mock.call_count > 0
        assert all(call.args == (backoff,) for call in sleep_mock.call_args_list)

    def test_h_soft_join_over_patience_rotates_before_upload(self, temp_dir):
        """Integration seam: a soft-join over-patience rotation must propagate out of the
        ``with host_attempt as host:`` body BEFORE ``prepare_host`` (the artifact upload)
        runs, rotate to another host WITHOUT marking any host failed, and be counted as a
        rotation. This mirrors the orchestrator body order
        (``soft_join_queue`` -> ``prepare_host``): if soft-join raises
        ``QueuePatienceRotation``, ``prepare_host`` must never be reached."""
        from test.utils.exceptions import QueuePatienceRotation
        from test.utils.metrics_collector import MetricName

        num_hosts = 2
        manager = self._build_manager_with_hosts(temp_dir, num_hosts=num_hosts)
        mock_collector = MagicMock()

        # Every host's soft_join_queue raises over-patience; prepare_host is a shared spy
        # that must NEVER be called on the rotation path (proves no upload happened).
        prepare_spy = MagicMock(return_value=contextlib.nullcontext())

        def make_host(alias):
            host = MagicMock(get_host_id=MagicMock(return_value=alias))
            host.soft_join_queue.side_effect = QueuePatienceRotation("queue ETA exceeds patience")
            host.prepare_host = prepare_spy
            return host

        self._seed_hosts(manager, make_host)

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 50.0
        backoff = 5.0
        # ~9 rotations before the wall-clock deadline trips.
        expected_min_rotations = int(deadline_seconds / backoff) - 1

        attempted_hosts = []
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException) as exc_info:
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=1,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            attempted_hosts.append(host.get_host_id())
                            # Mirror the orchestrator body order exactly: soft-join the
                            # FIFO queue BEFORE uploading via prepare_host. soft_join
                            # raises over-patience, so prepare_host must be unreachable.
                            host.soft_join_queue(
                                collector=mock_collector,
                                collective_ranks=1,
                                lnc_config=2,
                            )
                            host.prepare_host(
                                target_directory="/tmp",
                                collector=mock_collector,
                            )

        # (1) The upload was NEVER attempted on the rotation path.
        prepare_spy.assert_not_called()
        # (2) An over-patience rotation is transient, not a host failure.
        assert manager.get_failed_host_count() == 0
        # (3) The rotation metric was recorded exactly once per rotation (each value 1.0).
        rotation_values = [
            c.args[1]
            for c in mock_collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_HOST_ROTATION_COUNT
        ]
        rotations = len(attempted_hosts)
        assert rotations >= expected_min_rotations
        assert len(rotation_values) == rotations
        assert all(v == 1.0 for v in rotation_values)
        # (4) Only real pool hosts were attempted and at least one stayed eligible and was
        # re-selected (rotation kept hosts in the pool, no exclusion).
        assert set(attempted_hosts).issubset(set(manager.target_hosts.keys()))
        assert max(attempted_hosts.count(h) for h in set(attempted_hosts)) > 1
        # Distinct deadline termination (not "No available hosts" / connection error).
        msg = str(exc_info.value)
        assert "deadline" in msg.lower()
        assert "No available hosts" not in msg
        assert "Connection error after" not in msg

    def _build_manager_with_mixed_cores(self, temp_dir, host_cores: dict[str, int]):
        """Build a HostManager over a state store whose TRN2 host rows carry the given
        persisted core counts (a heterogeneous pool with an ineligible/too-small host
        alongside eligible ones). SshHost is patched so claimed hosts are alias mocks."""
        target_hosts = [TargetHost(ssh_host=a, host_type=Platforms.TRN2) for a in host_cores]

        def fake_make(alias, **_kwargs):
            return MagicMock(get_host_id=MagicMock(return_value=alias), connection=MagicMock())

        patcher = patch("test.utils.host_management.SshHost", side_effect=fake_make)
        patcher.start()
        self._patchers.append(patcher)
        return HostManager(
            state_store=_initialized_store(temp_dir, target_hosts, cores_by_alias=dict(host_cores)),
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=False,
        )

    def test_rotation_reset_fires_with_ineligible_host_in_pool(self, temp_dir):
        """Heterogeneous pool: two eligible same-size hosts + one too-small
        (ineligible) host. Both eligible hosts must be attempted MORE THAN ONCE
        across rotations -- proving the busy-set reset fired and re-enabled them --
        rather than the loop sticking on one host. Under the OLD all-targets
        comparison the reset can never fire (the too-small host never enters
        ``busy_hosts``), so the loop would re-pick a single host every iteration."""
        from test.utils.exceptions import QueuePatienceRotation

        # big1/big2 eligible (>= 32 needed); small ineligible (8 < 32).
        manager = self._build_manager_with_mixed_cores(temp_dir, {"big1": 64, "big2": 64, "small": 8})
        mock_collector = MagicMock()

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 100.0
        backoff = 5.0

        attempted_hosts = []
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException):
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=32,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            attempted_hosts.append(host.get_host_id())
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        # The too-small host is never eligible and must never be attempted.
        assert "small" not in attempted_hosts
        # BOTH eligible hosts were attempted more than once: the reset re-enabled
        # them so the fleet kept rotating instead of sticking on a single host.
        assert attempted_hosts.count("big1") > 1
        assert attempted_hosts.count("big2") > 1
        # No merely-busy host was ever marked failed.
        assert manager.get_failed_host_count() == 0

    def test_rotation_reset_ignores_ineligible_hosts(self, temp_dir):
        """Once both eligible hosts are busy, the reset clears ``busy_hosts`` even
        though an ineligible (too-small) host still exists in ``target_hosts`` --
        so the loop keeps rotating between the two eligible hosts until the
        deadline and never attempts the too-small host. Under the OLD all-targets
        comparison this would stick on one host."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager_with_mixed_cores(temp_dir, {"big1": 64, "big2": 64, "small": 8})
        mock_collector = MagicMock()

        fake_time, fake_sleep = self._patched_clock()
        deadline_seconds = 100.0
        backoff = 5.0

        attempted_hosts = []
        with (
            patch("test.utils.host_management.time.time", fake_time),
            patch("test.utils.host_management.time.sleep", fake_sleep),
        ):
            with pytest.raises(InferenceException):
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collective_ranks=32,
                        lnc_config=1,
                        collector=mock_collector,
                        deadline_seconds=deadline_seconds,
                        backoff_seconds=backoff,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt as host:
                            attempted_hosts.append(host.get_host_id())
                            raise QueuePatienceRotation("queue ETA exceeds patience")

        # The rotation stays entirely within the eligible set...
        assert set(attempted_hosts) == {"big1", "big2"}
        # ...and both eligible hosts get a roughly balanced share, proving the loop
        # re-rotates between them rather than deterministically re-picking one.
        assert attempted_hosts.count("big1") > 1
        assert attempted_hosts.count("big2") > 1


class TestSizeBasedHostRouting:
    """Explicit coverage for size-based host routing: hosts whose physical-core
    capacity is smaller than the requested ``collective_ranks x lnc_config`` are
    filtered out of host assignment so a request is only routed to a host big
    enough to satisfy it."""

    def setup_method(self):
        self._patchers = []

    def teardown_method(self):
        # Reverse order: patches stacked on the same target must be undone LIFO, or the
        # target is left pointing at an intermediate mock instead of the real object.
        for p in reversed(self._patchers):
            p.stop()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def _build_manager(self, temp_dir, host_cores: dict[str, int]):
        """Build a HostManager over a state store whose TRN2 host rows carry the given
        persisted physical-core counts (the single source of truth capacity routing
        ranks by). SshHost is patched so claimed hosts are alias-carrying mocks."""
        target_hosts = [TargetHost(ssh_host=a, host_type=Platforms.TRN2) for a in host_cores]

        def fake_make(alias, **_kwargs):
            return MagicMock(get_host_id=MagicMock(return_value=alias), connection=MagicMock())

        patcher = patch("test.utils.host_management.SshHost", side_effect=fake_make)
        patcher.start()
        self._patchers.append(patcher)
        return HostManager(
            state_store=_initialized_store(temp_dir, target_hosts, cores_by_alias=dict(host_cores)),
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=False,
        )

    def _set_depths(self, manager, depths: dict[str, int]):
        """Directly set per-host work_queue_depth in the store (test seam)."""
        manager.state_store._update(
            lambda state: HostState(
                hosts=[
                    replace(rec, work_queue_depth=depths.get(rec.resolved.ssh_host, rec.work_queue_depth))
                    for rec in state.hosts
                ],
                poisoned=state.poisoned,
            )
        )

    def _depths(self, manager) -> dict[str, int]:
        return {rec.resolved.ssh_host: rec.work_queue_depth for rec in manager.state_store.read().hosts}

    def test_too_small_host_is_filtered_out(self, temp_dir):
        """A host with fewer physical cores than needed is never selected; the
        only sufficiently large host is returned."""
        manager = self._build_manager(temp_dir, {"small": 4, "big": 64})
        host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=16)
        assert host.get_host_id() == "big"

    def test_boundary_exactly_enough_is_eligible_one_short_is_not(self, temp_dir):
        """A host with exactly the needed core count is eligible; one core short is
        excluded (the filter is ``>=``)."""
        manager = self._build_manager(temp_dir, {"exact": 16, "short": 15})
        host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=16)
        assert host.get_host_id() == "exact"

    def test_all_hosts_too_small_raises_with_size_breakdown(self, temp_dir):
        """When every matching host is too small, the diagnostic distinguishes a
        sizing mismatch from a failure: it reports the requested core count and
        that the hosts are 'too small' (with the largest available), NOT 'unavailable'."""
        manager = self._build_manager(temp_dir, {"h1": 4, "h2": 8})
        with pytest.raises(FleetEmptyError, match="No available hosts") as exc:
            manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=16)
        msg = str(exc.value)
        assert "needing 16 physical cores" in msg
        assert "2 too small" in msg
        assert "largest available 8 cores" in msg
        assert "unavailable" not in msg

    def test_all_hosts_unavailable_raises_with_unavailable_breakdown(self, temp_dir):
        """When every matching, large-enough host has been marked unavailable, the
        diagnostic reports them as 'unavailable' and NOT as a sizing problem."""
        manager = self._build_manager(temp_dir, {"h1": 64, "h2": 64})
        manager.mark_host_unavailable("h1")
        manager.mark_host_unavailable("h2")
        with pytest.raises(FleetEmptyError, match="No available hosts") as exc:
            manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=16)
        msg = str(exc.value)
        assert "2 unavailable" in msg
        assert "too small" not in msg

    def test_mixed_unavailable_and_too_small_breakdown(self, temp_dir):
        """A pool where one host is unavailable and another is too small reports BOTH
        reasons distinctly rather than collapsing to a single cause."""
        manager = self._build_manager(temp_dir, {"down": 64, "tiny": 4})
        manager.mark_host_unavailable("down")
        with pytest.raises(FleetEmptyError, match="No available hosts") as exc:
            manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=16)
        msg = str(exc.value)
        assert "1 unavailable" in msg
        assert "1 too small" in msg

    def test_no_matching_platform_raises_distinct_message(self, temp_dir):
        """When no host of the requested platform exists, the diagnostic says the
        type is absent from the pool -- not that hosts were unavailable or too small."""
        manager = self._build_manager(temp_dir, {"h1": 64})
        with pytest.raises(FleetEmptyError, match="no host of platform trn1 exists") as exc:
            manager.__get_host_assignment__(Platforms.TRN1, num_of_physical_cores_needed=16)
        msg = str(exc.value)
        assert "unavailable" not in msg
        assert "too small" not in msg

    def test_retry_routes_using_ranks_times_lnc_requirement(self, temp_dir):
        """``get_host_assignment_with_retry`` derives the requirement as
        ``collective_ranks * lnc_config`` (8 * 2 = 16) and routes only to a host
        large enough; the 8-core host is excluded."""
        manager = self._build_manager(temp_dir, {"small": 8, "big": 64})
        mock_collector = MagicMock()

        chosen = []
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=mock_collector,
                collective_ranks=8,
                lnc_config=2,
                backoff_seconds=0,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    chosen.append(host.get_host_id())
        assert chosen == ["big"]

    def test_retry_lnc1_only_multiplies_by_one(self, temp_dir):
        """With lnc1 the requirement equals ``collective_ranks``; a host with that
        exact count is eligible."""
        manager = self._build_manager(temp_dir, {"small": 3, "exact": 4})
        mock_collector = MagicMock()

        chosen = []
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=mock_collector,
                collective_ranks=4,
                lnc_config=1,
                backoff_seconds=0,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    chosen.append(host.get_host_id())
        assert chosen == ["exact"]

    def test_assignment_and_release_account_for_full_core_count(self, temp_dir):
        """Assignment adds the requested physical-core count to the host's queue
        depth (proportional load, not a flat +1), and ``release_host`` subtracts
        exactly that amount."""
        manager = self._build_manager(temp_dir, {"big": 64})
        needed = 16

        host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=needed)
        assert self._depths(manager)["big"] == needed

        manager.release_host(host, needed)
        assert self._depths(manager)["big"] == 0

    def test_pick_prefers_lower_load_ratio_not_absolute_depth(self, temp_dir):
        """Selection picks the single least-loaded host by load ratio
        (``work_queue_depth / num_physical_cores``), not by absolute queue depth:
        a large host carrying a higher absolute depth but a lower ratio wins over a
        small host with a lower absolute depth but a higher ratio."""
        manager = self._build_manager(temp_dir, {"big1": 64, "big2": 64, "big3": 64, "small": 8})
        # big hosts: depth 32 -> ratio 0.5; small: depth 8 -> ratio 1.0. The bigs carry
        # MORE absolute work yet a LOWER ratio, so the ratio (not absolute depth) must win.
        self._set_depths(manager, {"big1": 32, "big2": 32, "big3": 32, "small": 8})

        host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=1)
        assert host.get_host_id() != "small"
        assert host.get_host_id() in {"big1", "big2", "big3"}
        # Deterministic: identical state resolves to the same host on repeat (no randomness).
        manager.release_host(host, 1)
        again = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=1)
        assert again.get_host_id() == host.get_host_id()

    def test_assignments_spread_in_proportion_to_capacity(self, temp_dir):
        """Repeated single-core assignments drain a large host proportionally more than a
        small one: with a 128-core and an 8-core host (16:1 capacity) the 128-core host
        receives ~16x the work -- its share of total capacity."""
        manager = self._build_manager(temp_dir, {"big": 128, "small": 8})
        counts = {"big": 0, "small": 0}
        for _ in range(136):  # one full capacity sweep (128 + 8)
            host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=1)
            counts[host.get_host_id()] += 1
        # big holds 128/136 = 94% of capacity -> it must absorb the large majority.
        assert counts["big"] > counts["small"] * 10
        # small still gets a fair, capacity-proportional (non-zero) slice.
        assert counts["small"] > 0

    def test_selection_spreads_across_tied_hosts_over_runs(self, temp_dir):
        """A pure tie (equal-size hosts all at depth 0, as right after startup) must NOT
        always resolve to the same host: initialize() shuffles the append order that the
        ranking falls back on, so first-claim load spreads across independent runs instead
        of hammering the first-listed host. Ranking itself is still deterministic once
        loads/capacities differ (see the capacity-proportional and release tests)."""
        picks = set()
        for _ in range(30):
            # Each iteration uses its own store dir so every run starts from identical
            # (fresh) tied state — the only variable is initialize()'s shuffle.
            with tempfile.TemporaryDirectory() as iter_dir:
                manager = self._build_manager(iter_dir, {"a": 64, "b": 64, "c": 64})
                host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=1)
                picks.add(host.get_host_id())
        # Over 30 runs across 3 hosts, a single-host result is astronomically unlikely
        # (3 * (1/3)^30) unless the shuffle regressed. All picks stay within the pool.
        assert len(picks) > 1, "tied-host selection should spread across runs, not pin to one host"
        assert picks <= {"a", "b", "c"}

    def test_idle_small_does_not_steal_from_spare_big(self, temp_dir):
        """An idle small host must NOT act as a "ratio-0 magnet". Ranking by the
        POST-placement load ratio, an idle 8-core host (post-ratio (0+2)/8=0.25)
        loses to a 128-core host already holding 6 cores (post-ratio
        (6+2)/128=0.0625), because the big host still has proportionally more
        headroom."""
        manager = self._build_manager(temp_dir, {"small": 8, "big": 128})
        self._set_depths(manager, {"big": 6, "small": 0})

        host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=2)
        assert host.get_host_id() == "big"

    def test_small_is_selected_once_big_is_sufficiently_loaded(self, temp_dir):
        """Smalls are not starved outright: once the big host is heavily loaded its
        post-placement ratio climbs past the small's, so the small host wins. Big at
        depth 120/128 -> post (120+2)/128=0.953 loses to small 0 -> (0+2)/8=0.25."""
        manager = self._build_manager(temp_dir, {"small": 8, "big": 128})
        self._set_depths(manager, {"big": 120, "small": 0})

        host = manager.__get_host_assignment__(Platforms.TRN2, num_of_physical_cores_needed=2)
        assert host.get_host_id() == "small"


class TestSshHostTotalPhysicalCores:
    """Explicit coverage for SshHost.get_total_physical_cores(): physical cores =
    sum over devices of ``len(neuroncore_ids) * logical_neuroncore_config``,
    computed once and cached."""

    def test_sums_cores_across_devices(self, tmp_path):
        host = _make_ssh_host(tmp_path)
        host._total_physical_cores = None  # force a fresh computation
        devices = [_make_mock_device([0, 1, 2, 3], lnc_config=2), _make_mock_device([0, 1], lnc_config=2)]
        with patch.object(host, "get_neuron_device_info", return_value=devices) as mock_info:
            # (4 cores * lnc2) + (2 cores * lnc2) = 8 + 4 = 12
            assert host.get_total_physical_cores() == 12
            mock_info.assert_called_once()

    def test_lnc1_devices(self, tmp_path):
        host = _make_ssh_host(tmp_path)
        host._total_physical_cores = None
        devices = [_make_mock_device([0, 1, 2, 3], lnc_config=1)]
        with patch.object(host, "get_neuron_device_info", return_value=devices):
            assert host.get_total_physical_cores() == 4

    def test_result_is_cached(self, tmp_path):
        """A second call reuses the cached value and does not re-query devices."""
        host = _make_ssh_host(tmp_path)
        host._total_physical_cores = None
        devices = [_make_mock_device([0, 1, 2, 3], lnc_config=2)]
        with patch.object(host, "get_neuron_device_info", return_value=devices) as mock_info:
            first = host.get_total_physical_cores()
            second = host.get_total_physical_cores()
        assert first == second == 8
        mock_info.assert_called_once()


SAMPLE_NEURON_LS_OUTPUT = json.dumps(
    [
        {
            "neuron_device": 0,
            "bdf": "00:1e.0",
            "cpu_affinity": "0-3",
            "numa_node": "0",
            "connected_to": None,
            "nc_count": 2,
            "memory_size": 34359738368,
            "neuroncore_ids": [0, 1],
            "neuron_processes": [],
            "instance_type": "trn2.48xlarge",
        }
    ]
)


class TestDetectLocalNeuronDevices:
    """Tests for detect_local_neuron_devices()."""

    def setup_method(self):
        # Clear lru_cache between tests so each test gets a fresh call
        from test.utils.host_management import _run_neuron_ls

        _run_neuron_ls.cache_clear()

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_true_when_devices_found(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=0, stdout=SAMPLE_NEURON_LS_OUTPUT)
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is True

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_when_no_devices(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]")
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_when_neuron_ls_not_installed(self, mock_run, _mock_isfile):
        mock_run.side_effect = FileNotFoundError("neuron-ls not found")
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_on_timeout(self, mock_run, _mock_isfile):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="neuron-ls", timeout=10)
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_false_on_nonzero_exit(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is False

    def test_returns_false_when_binary_missing(self):
        """Binary check should short-circuit without spawning subprocess."""
        with patch("test.utils.host_management.os.path.isfile", return_value=False):
            assert detect_local_neuron_devices("/nonexistent/path") is False


def _make_mock_device(core_ids: list[int], lnc_config: int = 2) -> MagicMock:
    """Create a mock NeuronDeviceInfo with the given logical core IDs."""
    device = MagicMock()
    device.neuroncore_ids = core_ids
    device.nc_count = len(core_ids)
    device.logical_neuroncore_config = lnc_config
    return device


class TestLocalHostCoreAllocation:
    """Tests for LocalHost.get_core_allocation() with file-based locking."""

    @pytest.fixture(autouse=True)
    def _patch_neuron_ls(self):
        # get_total_physical_cores() reads neuron-ls directly via _run_neuron_ls;
        # 4 logical cores * lnc2 = 8 physical cores total.
        with patch(
            "test.utils.host_management._run_neuron_ls",
            return_value=[_make_mock_device([0, 1, 2, 3])],
        ):
            yield

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def local_host(self, temp_dir):
        return LocalHost(
            "/opt/aws/neuron/bin", "localhost", temp_dir, core_reset_strategy=CoreResetStrategy(exclusive_run=False)
        )

    def test_allocates_correct_cores(self, local_host):
        """Core allocation returns the requested number of logical cores."""
        mock_collector = MagicMock()
        mock_collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1, 2, 3])]):
            with local_host.get_core_allocation(collector=mock_collector, collective_ranks=2, lnc_config=2) as alloc:
                assert len(alloc.logical_core_ids) == 2
                assert alloc.lnc_config == 2
                assert alloc.host_id == "localhost"

    def test_releases_cores_after_use(self, local_host):
        """Cores are released back to the pool after the context manager exits."""
        mock_collector = MagicMock()
        mock_collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1, 2, 3])]):
            with local_host.get_core_allocation(collector=mock_collector, collective_ranks=2, lnc_config=2):
                pass

            # After exiting, all physical cores should be free — allocate all 4 logical (8 physical)
            with local_host.get_core_allocation(collector=mock_collector, collective_ranks=4, lnc_config=2) as alloc:
                assert len(alloc.logical_core_ids) == 4

    def test_lnc1_and_lnc2_dont_overlap(self, local_host):
        """LNC1 and LNC2 allocations must not share physical cores."""
        mock_collector = MagicMock()
        mock_collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )

        # 4 logical cores at LNC2 = 8 physical cores total
        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1, 2, 3])]):
            # Allocate 1 logical core at LNC2 = 2 physical cores
            with local_host.get_core_allocation(
                collector=mock_collector, collective_ranks=1, lnc_config=2
            ) as alloc_lnc2:
                lnc2_physical = set(range(alloc_lnc2.logical_core_ids[0] * 2, alloc_lnc2.logical_core_ids[0] * 2 + 2))

                # Allocate 1 logical core at LNC1 = 1 physical core, must not overlap
                with local_host.get_core_allocation(
                    collector=mock_collector, collective_ranks=1, lnc_config=1
                ) as alloc_lnc1:
                    lnc1_physical = {alloc_lnc1.logical_core_ids[0]}
                    assert lnc2_physical.isdisjoint(lnc1_physical), (
                        f"Physical cores overlap: LNC2={lnc2_physical}, LNC1={lnc1_physical}"
                    )


class TestLocalHostExecuteCommand:
    """Tests for LocalHost.execute_command()."""

    @pytest.fixture(autouse=True)
    def _patch_neuron_ls(self):
        # get_total_physical_cores() reads neuron-ls directly via _run_neuron_ls;
        # 2 logical cores * lnc2 = 4 physical cores total.
        with patch(
            "test.utils.host_management._run_neuron_ls",
            return_value=[_make_mock_device([0, 1])],
        ):
            yield

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def local_host(self, temp_dir):
        return LocalHost(
            "/opt/aws/neuron/bin", "localhost", temp_dir, core_reset_strategy=CoreResetStrategy(exclusive_run=False)
        )

    def _mock_collector(self):
        collector = MagicMock()
        collector.timer = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        )
        return collector

    @patch("test.utils.host_management.subprocess.run")
    def test_successful_execution_sets_env_vars(self, mock_run, local_host, temp_dir):
        """Verify env vars are set correctly on local execution."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n")
        collector = self._mock_collector()

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1])]):
            local_host.execute_command(
                command="echo hello",
                target_directory=temp_dir,
                collector=collector,
                collective_ranks=1,
                lnc_config=2,
            )

        # Check subprocess was called with correct env vars
        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert "NEURON_RT_VISIBLE_CORES" in env
        assert env["NEURON_LOGICAL_NC_CONFIG"] == "2"
        assert env["NEURON_RT_ENABLE_OCP"] == "1"

    @patch("test.utils.host_management.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run, local_host, temp_dir):
        """Non-zero exit code raises LocalExecutionException."""
        mock_run.return_value = MagicMock(returncode=1, stdout="error output", stderr="some error")
        collector = self._mock_collector()

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1])]):
            with pytest.raises(LocalExecutionException):
                local_host.execute_command(
                    command="false",
                    target_directory=temp_dir,
                    collector=collector,
                    collective_ranks=1,
                    lnc_config=2,
                )

    @patch("test.utils.host_management.subprocess.run")
    def test_artifact_collection(self, mock_run, local_host, temp_dir):
        """do_copy_artifacts moves files into infer_result/ subdirectory."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output.npy\n")
        collector = self._mock_collector()

        # Create a file in target_directory that should be moved
        test_file = os.path.join(temp_dir, "output.npy")
        with open(test_file, "w") as f:
            f.write("data")

        with patch.object(local_host, "get_neuron_device_info", return_value=[_make_mock_device([0, 1])]):
            result_path = local_host.execute_command(
                command="echo output.npy",
                target_directory=temp_dir,
                collector=collector,
                collective_ranks=1,
                lnc_config=2,
                do_copy_artifacts=True,
                get_list_of_files_to_copy=lambda stdout: ["output.npy"],
            )

        assert result_path == os.path.join(temp_dir, "infer_result")
        assert os.path.exists(os.path.join(result_path, "output.npy"))


# =============================================================================
# SshHost.get_core_allocation poll-loop hardening + caller patience rotation.
#
# Covers: POLL_PERIOD as the poll-period default, the
# max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX) <= STALE_THRESHOLD
# retry-budget invariant,
# the pure should_rotate predicate, QUEUED+ETA>patience -> dequeue + rotate,
# DRAINING stay-and-probe (no rotate), and dequeue-on-abandon (timeout/give-up).
# =============================================================================

import test.utils.host_management as host_management  # noqa: E402
from test.utils.core_lock_manager import (  # noqa: E402
    AllocationOutcome,
    AllocationStatus,
    CoreAllocation,
    CoreLockManager,
    LockAcquisitionError,
)
from test.utils.exceptions import QueuePatienceRotation  # noqa: E402
from test.utils.host_management import (  # noqa: E402
    DEFAULT_PATIENCE_SECONDS,
    FAST_POLL_PERIOD,
    POLL_JITTER_MAX,
    POLL_PERIOD,
    STALE_THRESHOLD,
    SshHost,
    select_poll_base,
    should_rotate,
)
from test.utils.metrics_collector import NoopMetricsCollector  # noqa: E402
from test.utils.remote_executor import RemoteExecutorError  # noqa: E402
from test.utils.scripts import remote_lock_scripts  # noqa: E402

# A realistic unix epoch for the integration clock. Starting the fake clock here
# (rather than ~0) ensures the helper's RELATIVE worst_case_eta is exercised
# independent of the absolute wall clock: an absolute-timestamp ETA at this
# epoch (~1.7e9) would dwarf DEFAULT_PATIENCE_SECONDS and force instant rotation.
_REALISTIC_EPOCH = 1_700_000_000.0


class _FakeClock:
    """Deterministic clock: time() returns the current value; sleep() advances it.

    Patched over the global ``time`` module so BOTH the poll loop
    (host_management) and the real lock helper (remote_lock_scripts) observe the
    same monotonic, sleep-driven time. Integration tests start it at a realistic
    epoch (``_REALISTIC_EPOCH``) rather than ~0: the helper now returns
    ``worst_case_eta`` as a RELATIVE wait (seconds from now), so the absolute
    clock value must be irrelevant to the DEFAULT_PATIENCE_SECONDS threshold. Starting
    at a real epoch proves that (with an absolute-timestamp ETA every caller
    would instantly exceed patience and rotate).
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class _RecordingClock(_FakeClock):
    """A _FakeClock that records every sleep duration in call order.

    Lets wiring tests assert the exact base+jitter the poll loop passes to
    ``time.sleep`` per iteration while still advancing the clock so the loop
    terminates deterministically.
    """

    def __init__(self, start: float = 0.0) -> None:
        super().__init__(start)
        self.sleeps: list[float] = []

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        super().sleep(dt)


def _cm_collector() -> MagicMock:
    """A MagicMock collector whose timer() is a usable context manager."""
    collector = MagicMock()
    collector.test_name = "test-poll-loop"
    collector.timer = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False)))
    return collector


def _make_ssh_host(
    tmp_path, total_physical_cores: int = 8, patience_seconds: int = DEFAULT_PATIENCE_SECONDS
) -> SshHost:
    """Construct an SshHost without any real SSH; locking version/core count pre-seeded."""
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("")
    host = SshHost(
        ssh_alias="fakehost",
        test_base_path=str(tmp_path),
        remote_neuron_install_dir="/opt/aws/neuron/bin",
        ssh_config_path=str(ssh_config),
        s3_config=MagicMock(),
        patience_seconds=patience_seconds,
    )
    host._total_physical_cores = total_physical_cores
    host._host_locking_version = 3
    return host


class _FakeManager:
    """Stand-in for CoreLockManager that replays a scripted outcome sequence.

    Once the sequence is exhausted it repeats the final outcome, so the test is
    robust to the exact number of poll iterations the loop performs.
    """

    def __init__(self, outcomes, probe_outcomes=None, release_error=None):
        self._outcomes = list(outcomes)
        # When set, release() raises this exception to exercise the best-effort
        # release teardown path (locks auto-expire, so a failed release is
        # swallowed after logging + recording a failure metric).
        self._release_error = release_error
        # Separate scripted sequence for the read-only probe pre-screen. When
        # None, probe returns a benign under-patience outcome so non-gate tests
        # fall through to acquire(ready=False).
        self._probe_outcomes = list(probe_outcomes) if probe_outcomes is not None else None
        self.acquire_calls = 0
        self.probe_calls = 0
        self.dequeue_calls = 0
        self.metrics_calls = 0
        self.released: list[list[int]] = []
        # Ordered log of terminal-teardown calls so tests can assert that the
        # abandon flush records metrics BEFORE the slot is freed.
        self.events: list[str] = []

    def probe(self, num_logical_cores, lnc_config, timeout_seconds=None):  # noqa: ARG002
        self.probe_calls += 1
        if self._probe_outcomes is None:
            return AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=0)
        item = self._probe_outcomes[0] if len(self._probe_outcomes) == 1 else self._probe_outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def acquire(self, num_logical_cores, lnc_config, ready=True):  # noqa: ARG002
        self.acquire_calls += 1
        item = self._outcomes[0] if len(self._outcomes) == 1 else self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def dequeue(self) -> None:
        self.dequeue_calls += 1
        self.events.append("dequeue")

    def release(self, core_ids) -> None:
        self.released.append(list(core_ids))
        if self._release_error is not None:
            raise self._release_error

    def record_contention_metrics(self) -> None:
        self.metrics_calls += 1
        self.events.append("metrics")


class TestShouldRotate:
    """The pure caller-patience predicate."""

    def test_patience_seconds_is_180(self) -> None:
        assert DEFAULT_PATIENCE_SECONDS == 180

    def test_rotate_when_eta_exceeds_patience_and_not_draining(self) -> None:
        assert (
            should_rotate(DEFAULT_PATIENCE_SECONDS + 1, draining=False, patience_seconds=DEFAULT_PATIENCE_SECONDS)
            is True
        )

    def test_no_rotate_when_draining_even_if_eta_large(self) -> None:
        # ETA is meaningless mid-drain; stay-and-probe, never rotate.
        assert (
            should_rotate(DEFAULT_PATIENCE_SECONDS + 1, draining=True, patience_seconds=DEFAULT_PATIENCE_SECONDS)
            is False
        )

    def test_no_rotate_when_eta_is_none(self) -> None:
        assert should_rotate(None, draining=False, patience_seconds=DEFAULT_PATIENCE_SECONDS) is False

    def test_no_rotate_when_eta_within_patience(self) -> None:
        assert (
            should_rotate(DEFAULT_PATIENCE_SECONDS - 1, draining=False, patience_seconds=DEFAULT_PATIENCE_SECONDS)
            is False
        )
        assert (
            should_rotate(DEFAULT_PATIENCE_SECONDS, draining=False, patience_seconds=DEFAULT_PATIENCE_SECONDS) is False
        )

    def test_configured_patience_threshold_is_honored(self) -> None:
        # A custom (non-default) threshold governs the rotate boundary.
        assert should_rotate(51, draining=False, patience_seconds=50) is True
        assert should_rotate(50, draining=False, patience_seconds=50) is False
        # An ETA above the default but below a larger configured patience must not rotate.
        assert should_rotate(DEFAULT_PATIENCE_SECONDS + 1, draining=False, patience_seconds=10_000) is False

    def test_negative_patience_disables_rotation(self) -> None:
        # Negative patience disables the mechanic entirely, regardless of ETA.
        assert should_rotate(10_000, draining=False, patience_seconds=-1) is False

    def test_zero_patience_disables_rotation(self) -> None:
        # Zero is not a positive threshold, so rotation stays disabled.
        assert should_rotate(10_000, draining=False, patience_seconds=0) is False


class TestPatienceConfiguration:
    """Plumbing for the configurable host-rotation patience threshold."""

    def _capture_ssh_host_kwargs(self, tmp_path, **manager_kwargs):
        """Build a HostManager (SshHost patched out) and lazily materialize one host, so
        the SshHost the manager constructs in _get_host is captured. Returns its kwargs."""
        target_hosts = [TargetHost(ssh_host="host1", host_type=Platforms.TRN2)]
        with patch("test.utils.host_management.SshHost") as mock_host:
            manager = HostManager(
                state_store=_initialized_store(str(tmp_path), target_hosts),
                neuron_installation_path="/opt/aws/neuron/bin",
                ssh_config_path="~/.ssh/config",
                testrun_uid="test-run",
                needs_local_host=False,
                **manager_kwargs,
            )
            manager._get_host("host1")  # lazily builds (and captures) the SshHost
        return mock_host.call_args.kwargs

    def test_configured_patience_propagates_to_ssh_host(self, tmp_path) -> None:
        kwargs = self._capture_ssh_host_kwargs(tmp_path, host_rotation_patience_seconds=42)
        assert kwargs["patience_seconds"] == 42

    def test_none_patience_falls_back_to_default(self, tmp_path) -> None:
        # Regression: the unset CLI flag resolves to None; HostManager must
        # coerce it to the default rather than propagating None (which would
        # crash should_rotate on the int>None comparison).
        kwargs = self._capture_ssh_host_kwargs(tmp_path, host_rotation_patience_seconds=None)
        assert kwargs["patience_seconds"] == DEFAULT_PATIENCE_SECONDS

    def test_default_when_arg_omitted(self, tmp_path) -> None:
        kwargs = self._capture_ssh_host_kwargs(tmp_path)
        assert kwargs["patience_seconds"] == DEFAULT_PATIENCE_SECONDS

    def test_ssh_host_stores_patience(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path, patience_seconds=99)
        assert host.patience_seconds == 99

    def test_exclusive_run_threads_reset_strategy_into_ssh_host(self, tmp_path) -> None:
        # _get_host builds the per-host CoreResetStrategy from the manager's exclusive +
        # testrun_uid (see __build_reset_strategy__). An exclusive strategy is state-backed
        # and namespaced by testrun_uid + alias; assert it reaches the SshHost so exclusive
        # per-capture reset isn't silently inert.
        kwargs = self._capture_ssh_host_kwargs(tmp_path, exclusive=True)
        strategy = kwargs["core_reset_strategy"]
        assert strategy._exclusive_run is True
        # Exclusive path requires (and uses) testrun_uid + ssh_alias; a state path proves both were threaded.
        assert strategy._state_path is not None
        assert "test-run" in strategy._state_path and "host1" in strategy._state_path

    def test_shared_run_reset_strategy_is_non_exclusive(self, tmp_path) -> None:
        # Default (shared) run: the strategy always resets, with no state file.
        kwargs = self._capture_ssh_host_kwargs(tmp_path)
        strategy = kwargs["core_reset_strategy"]
        assert strategy._exclusive_run is False
        assert strategy._state_path is None


class TestSelectPollBase:
    """The pure poll-cadence selector for a queued caller."""

    def test_unknown_position_uses_slow(self) -> None:
        assert select_poll_base(None) == POLL_PERIOD

    def test_front_position_uses_fast(self) -> None:
        assert select_poll_base(0) == FAST_POLL_PERIOD

    def test_one_slot_from_front_uses_fast(self) -> None:
        assert select_poll_base(1) == FAST_POLL_PERIOD

    def test_position_two_uses_slow(self) -> None:
        assert select_poll_base(2) == POLL_PERIOD

    def test_large_position_uses_slow(self) -> None:
        assert select_poll_base(50) == POLL_PERIOD

    def test_negative_position_uses_fast(self) -> None:
        # The rule is literal `position <= 1`, so a negative position is fast.
        assert select_poll_base(-1) == FAST_POLL_PERIOD

    def test_injectable_fast_and_slow(self) -> None:
        assert select_poll_base(0, fast=2, slow=9) == 2
        assert select_poll_base(5, fast=2, slow=9) == 9


class TestPollLoopConstants:
    """POLL_PERIOD is the single source of truth for the poll-period default."""

    def test_fast_poll_period_is_one_and_below_slow(self) -> None:
        assert FAST_POLL_PERIOD == 1
        assert FAST_POLL_PERIOD < POLL_PERIOD

    def test_poll_period_default_reflects_constant(self) -> None:
        import inspect

        for cls in (host_management.Host, host_management.LocalHost, SshHost):
            sig = inspect.signature(cls.get_core_allocation)
            assert sig.parameters["poll_period_seconds"].default == POLL_PERIOD

    def test_retry_budget_fits_within_stale_threshold(self) -> None:
        # The loop tolerates 10 consecutive retryable errors spaced at
        # POLL_PERIOD + up to POLL_JITTER_MAX (plus RPC latency); that full
        # budget must not exceed STALE_THRESHOLD, with headroom over jitter.
        max_retryable_errors = 10
        assert max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX) <= STALE_THRESHOLD


class TestSshHostPollLoop:
    """Unit tests for the SshHost.get_core_allocation poll loop with a faked manager."""

    @contextlib.contextmanager
    def _run(self, host, fake_mgr, clock=None, **kwargs):
        clock = clock or _FakeClock()
        with (
            patch.object(host_management, "CoreLockManager", return_value=fake_mgr),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=_cm_collector(), **kwargs) as alloc:
                yield alloc

    def test_draining_keeps_polling_then_allocates(self, tmp_path) -> None:
        """DRAINING does not abort, does not rotate, and is not a retryable error."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                # Large ETA while draining must NOT rotate (patience suppressed).
                AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=99999),
                AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0, 1],
                    physical_cores=[0, 1, 2, 3],
                ),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=2, lnc_config=2) as alloc:
            assert isinstance(alloc, CoreAllocation)
            assert alloc.logical_core_ids == [0, 1]
        # Polled through DRAINING + QUEUED before ALLOCATED; never dequeued (committed).
        assert fake_mgr.acquire_calls == 3
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.released == [[0, 1, 2, 3]]

    def test_queued_within_patience_keeps_polling_then_allocates(self, tmp_path) -> None:
        """QUEUED with ETA within patience must NOT rotate; it keeps polling."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=DEFAULT_PATIENCE_SECONDS),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.acquire_calls == 2
        assert fake_mgr.dequeue_calls == 0

    def test_poll_base_position_one_uses_fast_cadence(self, tmp_path) -> None:
        """Wiring: a near-front waiter (position=1) sleeps the FAST base + jitter."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == FAST_POLL_PERIOD + POLL_JITTER_MAX

    def test_poll_base_position_two_uses_slow_cadence(self, tmp_path) -> None:
        """Wiring: a deeper waiter (position=2) sleeps the SLOW base + jitter."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=2, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == POLL_PERIOD + POLL_JITTER_MAX

    def test_poll_base_draining_front_uses_fast_cadence_no_status_gating(self, tmp_path) -> None:
        """Wiring: a DRAINING front waiter (position=0) still fast-polls.

        Proves the fast cadence is driven purely by position and is NOT
        suppressed while the host is draining (no status gating).
        """
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == FAST_POLL_PERIOD + POLL_JITTER_MAX

    def test_poll_base_honors_caller_slow_period(self, tmp_path) -> None:
        """Wiring: a deep waiter's slow base honors the caller's poll_period_seconds,
        proving slow=poll_period_seconds is threaded into select_poll_base."""
        host = _make_ssh_host(tmp_path)
        custom_slow = POLL_PERIOD + 7
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=5, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(
                host,
                fake_mgr,
                clock=clock,
                collective_ranks=1,
                lnc_config=2,
                poll_period_seconds=custom_slow,
            ):
                pass
        assert clock.sleeps[0] == custom_slow + POLL_JITTER_MAX

    def test_stale_outcome_does_not_fast_spin_on_retryable_error(self, tmp_path) -> None:
        """Regression: a successful near-front poll (QUEUED position=0) must NOT leave a
        stale near-front position that makes the NEXT (retryable-error) iteration
        fast-spin. The successful poll fast-polls; the post-error poll must use the
        SLOW cadence (an error is not evidence the caller is near the front).

        Before the fix the except branch did not reset ``outcome``, so sleeps[1]
        would be FAST+jitter, burning the 10-error budget ~3.6x faster.
        """
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10),
                LockAcquisitionError("transient"),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        # Successful near-front poll fast-polls (unchanged)...
        assert clock.sleeps[0] == FAST_POLL_PERIOD + POLL_JITTER_MAX
        # ...but the post-error iteration falls back to the SLOW cadence.
        assert clock.sleeps[1] == POLL_PERIOD + POLL_JITTER_MAX

    def test_none_position_routes_to_slow_through_loop(self, tmp_path) -> None:
        """Wiring: a QUEUED outcome with an unknown position (None) sleeps the SLOW
        base + jitter through the real loop (not just at the pure-helper level)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(status=AllocationStatus.QUEUED, position=None, worst_case_eta=10),
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        clock = _RecordingClock()
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                pass
        assert clock.sleeps[0] == POLL_PERIOD + POLL_JITTER_MAX

    def test_dequeue_on_retryable_giveup(self, tmp_path) -> None:
        """After max_retryable_errors consecutive retryable errors, give up + dequeue."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager([LockAcquisitionError("transient") for _ in range(10)])
        with pytest.raises(OSError):
            with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2):
                pass
        assert fake_mgr.acquire_calls == 10
        assert fake_mgr.dequeue_calls == 1
        assert fake_mgr.released == []  # never committed

    def test_full_retry_budget_elapses_within_stale_threshold(self, tmp_path) -> None:
        """The full retry budget (10 retryable errors spaced at the REAL max gap of
        POLL_PERIOD + POLL_JITTER_MAX + RPC latency) elapses strictly inside
        STALE_THRESHOLD, so an actively-retrying caller refreshes last_seen_ts
        before the prune could ever fire (F5 liveness guarantee)."""
        host = _make_ssh_host(tmp_path)
        clock = _FakeClock(start=0.0)
        rpc_latency = 0.3

        class _SlowFailingManager(_FakeManager):
            def acquire(self, *a, **k):
                clock.sleep(rpc_latency)  # simulate the RPC round-trip per poll
                return super().acquire(*a, **k)

        fake_mgr = _SlowFailingManager([LockAcquisitionError("transient") for _ in range(10)])
        # Force jitter to its ceiling so every inter-poll gap is the worst case.
        with patch("random.uniform", return_value=POLL_JITTER_MAX):
            with pytest.raises(OSError):
                with self._run(host, fake_mgr, clock=clock, collective_ranks=1, lnc_config=2):
                    pass
        assert fake_mgr.acquire_calls == 10
        assert fake_mgr.dequeue_calls == 1
        # Even at max jitter + RPC latency, the whole budget fits with headroom.
        assert clock.time() < STALE_THRESHOLD

    def test_dequeue_on_timeout(self, tmp_path) -> None:
        """A wall-clock timeout while still QUEUED (within patience) dequeues + raises."""
        host = _make_ssh_host(tmp_path)
        # ETA within patience so the loop does NOT rotate; it polls until the
        # injected clock advances past the timeout budget, then dequeues.
        fake_mgr = _FakeManager([AllocationOutcome(status=AllocationStatus.QUEUED, position=2, worst_case_eta=10)])
        with pytest.raises(TimeoutException, match="Unable to allocate"):
            with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2, timeout_seconds=30):
                pass
        assert fake_mgr.acquire_calls >= 2  # polled repeatedly before timing out
        assert fake_mgr.dequeue_calls == 1
        assert fake_mgr.released == []  # never committed

    def test_no_dequeue_after_successful_commit(self, tmp_path) -> None:
        """A fast ALLOCATED must not trigger dequeue; only release on context exit."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(
                    status=AllocationStatus.ALLOCATED,
                    logical_cores=[0],
                    physical_cores=[0, 1],
                ),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.released == [[0, 1]]

    def test_over_patience_queued_keeps_polling_and_does_not_rotate(self, tmp_path) -> None:
        """Post-commit (get_core_allocation) no longer rotates on an over-patience
        QUEUED outcome: the gate moved to soft_join_queue (pre-upload). Once
        committed we keep polling until ALLOCATED with NO QueuePatienceRotation.
        The success path flushes contention metrics exactly once (not as an
        abandon) and never dequeues."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [
                AllocationOutcome(
                    status=AllocationStatus.QUEUED, position=5, worst_case_eta=DEFAULT_PATIENCE_SECONDS + 1
                ),
                AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1]),
            ]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        # Kept polling through the over-patience QUEUED, then committed: no rotation,
        # no dequeue, metrics flushed exactly once on success (not as an abandon).
        assert fake_mgr.acquire_calls == 2
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.events == ["metrics"]
        assert fake_mgr.metrics_calls == 1

    def test_retryable_giveup_and_timeout_abandons_flush_metrics(self, tmp_path) -> None:
        """F11: the retryable give-up and wall-clock timeout abandon paths also flush
        contention metrics before dequeuing."""
        # Retryable give-up.
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager([LockAcquisitionError("transient") for _ in range(10)])
        with pytest.raises(OSError):
            with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2):
                pass
        assert fake_mgr.events == ["metrics", "dequeue"]
        assert fake_mgr.metrics_calls == 1

        # Wall-clock timeout (within-patience QUEUED that never gets cores).
        host2 = _make_ssh_host(tmp_path)
        fake_mgr2 = _FakeManager([AllocationOutcome(status=AllocationStatus.QUEUED, position=2, worst_case_eta=10)])
        with pytest.raises(TimeoutException, match="Unable to allocate"):
            with self._run(host2, fake_mgr2, collective_ranks=1, lnc_config=2, timeout_seconds=30):
                pass
        assert fake_mgr2.events == ["metrics", "dequeue"]
        assert fake_mgr2.metrics_calls == 1

    def test_success_flushes_metrics_once_without_dequeue(self, tmp_path) -> None:
        """The ALLOCATED success path still flushes metrics exactly once and never
        dequeues (success and abandon are mutually exclusive)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])]
        )
        with self._run(host, fake_mgr, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.events == ["metrics"]
        assert fake_mgr.metrics_calls == 1
        assert fake_mgr.dequeue_calls == 0


class TestSshHostReleaseFailureHandling:
    """Best-effort core release on context exit.

    Core reservations are time-bound (auto-expire), so a failed ``release`` on
    context teardown must never propagate out of ``get_core_allocation`` and
    mask the real work result. Instead it is logged at warning and a
    ``CoreLockReleaseFailedCount`` datapoint is recorded.
    """

    @contextlib.contextmanager
    def _run(self, host, fake_mgr, collector, clock=None, **kwargs):
        """Like TestSshHostPollLoop._run but with a caller-supplied collector so
        the test can assert on record_metric calls."""
        clock = clock or _FakeClock()
        with (
            patch.object(host_management, "CoreLockManager", return_value=fake_mgr),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=collector, **kwargs) as alloc:
                yield alloc

    def _release_failed_count(self, collector) -> int:
        return sum(
            1
            for c in collector.record_metric.call_args_list
            if c.args and c.args[0] == MetricName.CORE_LOCK_RELEASE_FAILED_COUNT
        )

    def test_release_failure_is_swallowed_and_records_failure_metric(self, tmp_path) -> None:
        """A raising release() must not propagate; it records exactly one
        CoreLockReleaseFailedCount datapoint and still tears the manager down."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])],
            release_error=RuntimeError("boom during unlock"),
        )
        collector = _cm_collector()
        # The context manager exits cleanly despite release() raising.
        with self._run(host, fake_mgr, collector, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        # release was attempted with the committed physical cores...
        assert fake_mgr.released == [[0, 1]]
        # ...the failure was recorded exactly once...
        assert self._release_failed_count(collector) == 1
        # ...and the cached manager is dropped even though release failed.
        assert host._core_lock_manager is None

    def test_successful_release_records_no_failure_metric(self, tmp_path) -> None:
        """The happy path releases cores and records NO failure metric."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])],
        )
        collector = _cm_collector()
        with self._run(host, fake_mgr, collector, collective_ranks=1, lnc_config=2) as alloc:
            assert alloc.logical_core_ids == [0]
        assert fake_mgr.released == [[0, 1]]
        assert self._release_failed_count(collector) == 0
        assert host._core_lock_manager is None

    def test_body_exception_propagates_and_release_failure_does_not_mask_it(self, tmp_path) -> None:
        """If the with-body raises AND release() also raises, the body's exception
        must surface (release failure is swallowed) while the failure metric is
        still recorded."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])],
            release_error=RuntimeError("boom during unlock"),
        )
        collector = _cm_collector()
        with pytest.raises(ValueError, match="work failed"):
            with self._run(host, fake_mgr, collector, collective_ranks=1, lnc_config=2):
                raise ValueError("work failed")
        # release was still attempted on teardown and its failure recorded.
        assert fake_mgr.released == [[0, 1]]
        assert self._release_failed_count(collector) == 1
        assert host._core_lock_manager is None


class _HelperExecutor:
    """Fake RemoteExecutor running the real lock helper against a temp locks.json.

    Mirrors RemoteExecutor.call_function: redirects helper/lock paths and the
    remote locks.json path (args[0]) to local temp files, then runs the real
    helper verb so the deployed helper module is genuinely exercised. An optional
    on_call hook can mutate the locks.json between polls.
    """

    def __init__(self, helpers_file, lock_file, locks_json, on_call=None) -> None:
        self._helpers_file = helpers_file
        self._lock_file = lock_file
        self._locks_json = locks_json
        self._on_call = on_call
        self.calls = 0

    def call_function(self, func, **kwargs):
        self.calls += 1
        if self._on_call is not None:
            self._on_call(self.calls, self._locks_json)
        kwargs = dict(kwargs)
        kwargs["helpers_file"] = self._helpers_file
        kwargs["lock_file"] = self._lock_file
        args = list(kwargs["args"])
        args[0] = self._locks_json
        kwargs["args"] = args
        return func(**kwargs)


class TestSshHostPollLoopIntegration:
    """No-hardware integration: real helper + temp locks.json through get_core_allocation.

    A shared _FakeClock is patched over the global ``time`` module so both the
    poll loop and the real helper share one sleep-driven clock. It starts at a
    realistic epoch (``_REALISTIC_EPOCH``): the helper returns ``worst_case_eta``
    as a RELATIVE wait, so lock expiries are written epoch-relative and the
    DEFAULT_PATIENCE_SECONDS threshold is exercised independent of the absolute clock.
    """

    def _paths(self, tmp_path):
        return (
            str(remote_lock_scripts.__file__),
            str(tmp_path / "atomic_lock"),
            str(tmp_path / "locks.json"),
        )

    def _write_locks(self, locks_json, locked_until=None, draining_until=None, ncores=8):
        import json as _json

        state = {"version": 3, "physical_neuron_cores_lock_timeout": {}}
        if locked_until is not None:
            state["physical_neuron_cores_lock_timeout"] = {str(i): locked_until for i in range(ncores)}
        if draining_until is not None:
            state["draining_enabled_with_timeout"] = draining_until
        with open(locks_json, "w") as f:
            _json.dump(state, f)

    def _read_queue(self, locks_json):
        import json as _json

        with open(locks_json) as f:
            return _json.load(f).get("queue", [])

    def test_enqueues_polls_then_yields_once_cores_free(self, tmp_path) -> None:
        """Within-patience QUEUED caller stays, then commits once cores free."""
        import json as _json

        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Cores busy only briefly (ETA ~100s <= patience) so the caller does not
        # rotate; freed before the second poll so the head commits. locked_until
        # is epoch-relative so the helper's RELATIVE ETA is ~100s regardless of
        # the (realistic) absolute wall clock.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100)

        def free_cores_before_second_poll(call_n, path):
            if call_n == 2:
                with open(path) as f:
                    state = _json.load(f)
                state["physical_neuron_cores_lock_timeout"] = {}
                with open(path, "w") as f:
                    _json.dump(state, f)

        executor = _HelperExecutor(helpers_file, lock_file, locks_json, on_call=free_cores_before_second_poll)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(
                collector=_cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=300
            ) as alloc:
                assert isinstance(alloc, CoreAllocation)
                assert len(alloc.logical_core_ids) == 2

        assert executor.calls >= 2  # enqueued, polled, then committed
        assert self._read_queue(locks_json) == []  # committed + released -> empty

    def test_eta_over_patience_rotates_without_enqueue(self, tmp_path) -> None:
        """A deep queue pushes the soft-join caller's probed ETA past patience ->
        the read-only probe pre-screen in soft_join_queue raises
        QueuePatienceRotation BEFORE any enqueue (no dequeue, no churn): the
        queue keeps only the seeded waiters."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores busy for a full hold window; helper uses the shared clock.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 60)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            # Seed three full-window waiters AHEAD of our caller so the forward
            # simulation stacks 3 x 60s -> our ETA (~240s) exceeds patience.
            ahead = [
                CoreLockManager(
                    "fakehost",
                    total_physical_cores=8,
                    collector=NoopMetricsCollector(),
                    executor=executor,
                    host_locking_version=3,
                )
                for _ in range(3)
            ]
            for mgr in ahead:
                assert mgr.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert len(self._read_queue(locks_json)) == 3

            host = _make_ssh_host(tmp_path)
            with (
                patch.object(host, "_get_remote_executor", return_value=executor),
                patch.object(host, "get_instance_type", return_value="trn1"),
            ):
                with pytest.raises(QueuePatienceRotation, match="rotating host"):
                    host.soft_join_queue(collector=_cm_collector(), collective_ranks=4, lnc_config=2)

        # The probe is read-only: no enqueue, no dequeue -> only the three seeded
        # waiters remain (our caller never joined the queue).
        assert len(self._read_queue(locks_json)) == 3

    def test_stale_entry_pruned_then_reenqueued_records_reenqueue_not_bump(self, tmp_path) -> None:
        """F14 (no-hardware integration): an entry whose last_seen_ts goes stale (no
        poll for > STALE_THRESHOLD) is pruned by reconcile and transparently
        re-enqueued at the tail on its next poll with re_enqueued=True. The manager
        records a non-zero reenqueue count and a ZERO bump count -- proving a
        prune->re-enqueue is observable and distinct from a bump."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores busy so the sole caller queues (no fast-path / commit) until
        # we free them, keeping it alive in the queue across the stale gap.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100000)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            spy = _cm_collector()
            mgr = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=spy,
                executor=executor,
                host_locking_version=3,
            )
            # Poll #1: enqueue at tail (first-ever -> not a re-enqueue).
            assert mgr.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert mgr._reenqueue_count == 0
            assert len(self._read_queue(locks_json)) == 1

            # Let last_seen_ts go stale past STALE_THRESHOLD (a long upload).
            clock.sleep(STALE_THRESHOLD + POLL_PERIOD)

            # Poll #2: reconcile prunes the stale entry, then this poll re-enqueues
            # it at the tail with re_enqueued=True.
            assert mgr.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert mgr._reenqueue_count == 1
            assert mgr._bump_count == 0

            # Free the cores; the caller re-enqueues and commits on a later poll
            # (the queue is rebuilt fresh), flushing the per-attempt metrics on commit.
            self._write_locks(locks_json, locked_until=None)
            final = None
            for _ in range(5):
                final = mgr.acquire(num_logical_cores=4, lnc_config=2)
                if final.status == AllocationStatus.ALLOCATED:
                    break
            assert final is not None and final.status == AllocationStatus.ALLOCATED

        reenqueue = [c for c in spy.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_REENQUEUE_COUNT]
        assert reenqueue and reenqueue[-1].args[1] == 1
        bump = [c for c in spy.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_BUMP_COUNT]
        assert bump and bump[-1].args[1] == 0

    def test_draining_stays_and_probes_no_rotate(self, tmp_path) -> None:
        """A draining host is never rotated; the caller stays and keeps probing."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Drain far beyond the run; cores all busy so no fast-path either.
        self._write_locks(
            locks_json,
            locked_until=int(_REALISTIC_EPOCH) + 100000,
            draining_until=int(_REALISTIC_EPOCH) + 100000,
        )
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with pytest.raises(TimeoutException, match="Unable to allocate"):
                with host.get_core_allocation(
                    collector=_cm_collector(), collective_ranks=4, lnc_config=2, timeout_seconds=30
                ):
                    pass

        # Stayed and probed repeatedly (not a single-shot rotation), then the
        # wall-clock timeout dequeued the slot.
        assert executor.calls >= 3
        assert self._read_queue(locks_json) == []

    def test_timeout_calls_dequeue_and_clears_queue(self, tmp_path) -> None:
        """Within-patience QUEUED caller that never gets cores times out + dequeues."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Cores free at 120s (ETA ~120 <= patience, so no rotate) but the run
        # times out at 30s, so the caller never commits.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 120)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with pytest.raises(TimeoutException, match="Unable to allocate"):
                with host.get_core_allocation(
                    collector=_cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=30
                ):
                    pass

        # The poll enqueued us; the timeout path must dequeue so the entry is gone.
        assert self._read_queue(locks_json) == []

    def test_retrying_caller_within_budget_not_pruned_keeps_position(self, tmp_path) -> None:
        """A caller that misses several polls within the retry budget (its RPCs fail
        at the max gap, so last_seen_ts never refreshes) is NOT pruned by a
        concurrent reconcile and keeps its FIFO position behind the head (F5)."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores busy so both callers queue rather than committing.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100000)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            head = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )
            ours = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )
            assert head.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            assert ours.acquire(num_logical_cores=4, lnc_config=2).status == AllocationStatus.QUEUED
            # FIFO: head enqueued first, ours behind it.
            assert [e["entry_id"] for e in self._read_queue(locks_json)] == [head.entry_id, ours.entry_id]

            # Simulate our caller missing its whole retry budget: advance the clock
            # by 10 errors at the REAL max spacing (POLL_PERIOD + POLL_JITTER_MAX)
            # WITHOUT our manager refreshing last_seen_ts. Stay strictly inside
            # STALE_THRESHOLD -- the prune must not fire for us mid-budget.
            max_retryable_errors = 10
            budget = max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX)
            assert budget < STALE_THRESHOLD
            clock.t += budget

            # The head polls again -> runs reconcile/prune over the temp locks.json.
            head.acquire(num_logical_cores=4, lnc_config=2)

        # Our entry survived the reconcile and kept its FIFO position behind head.
        assert [e["entry_id"] for e in self._read_queue(locks_json)] == [head.entry_id, ours.entry_id]

    def test_transient_executor_reset_recovers_without_failing_host(self, tmp_path) -> None:
        """Poll-loop tolerance contract: a SINGLE transient executor error during
        acquisition (the kind a channel reset produces) surfaces as a retryable
        LockAcquisitionError; the loop retries, the next acquire succeeds, the
        attempt completes ALLOCATED, the host is NOT marked failed, and only a tiny
        slice of the 10-error retry budget is consumed.

        Scope note: this pins the poll-loop/retry-budget integration, not the
        executor's channel rebuild itself. The transient error is injected via a
        fake executor; the production RemoteExecutor self-heal (a nulled channel is
        rebuilt on the next call) is guarded directly at the unit level by
        test_remote_executor.test_call_reconnects_after_channel_crash. Together they
        cover the bench root cause: before the self-heal a reset left the cached
        executor permanently closed, so EVERY subsequent acquire raised and burned
        all 10 retries -> OSError -> failed host.
        """
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _RecordingClock(start=_REALISTIC_EPOCH)
        # Cores free from the start so a healthy acquire commits immediately; the
        # only thing standing between the caller and ALLOCATED is the transient
        # reset on the first executor use.
        self._write_locks(locks_json, locked_until=None)
        real_executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        class _TransientResetExecutor:
            """Raises a channel-reset RemoteExecutorError on its FIRST call (a transient
            reset), then self-heals: every later call delegates to the real helper
            executor, exactly as RemoteExecutor.call() rebuilds a crashed channel on
            demand before serving the next request."""

            def __init__(self, inner) -> None:
                self._inner = inner
                self.calls = 0
                self.raises = 0

            def call_function(self, func, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    self.raises += 1
                    raise RemoteExecutorError("Executor is closed")
                return self._inner.call_function(func, **kwargs)

        executor = _TransientResetExecutor(real_executor)
        host = _make_ssh_host(tmp_path)

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            # No OSError / TimeoutException escapes -> the host is never marked
            # failed by context_manager_wrapper; acquisition SUCCEEDS post-reset.
            with host.get_core_allocation(
                collector=_cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=300
            ) as alloc:
                assert isinstance(alloc, CoreAllocation)
                assert len(alloc.logical_core_ids) == 2

        # The transient reset was injected exactly once.
        assert executor.raises == 1
        # The poll loop absorbed that single transient error with only a couple of
        # inter-poll sleeps (the recovered acquire enqueues then commits) -- nowhere
        # near the max_retryable_errors budget of 10. Had the channel stayed dead
        # (OLD behavior) every acquire would raise, forcing 10 retries -> OSError and
        # a failed host; reaching ALLOCATED at all proves recovery.
        assert len(clock.sleeps) < 10
        # Committed + released -> a clean FIFO slot (no abandoned dequeue).
        assert self._read_queue(locks_json) == []


# =============================================================================
# Soft-join during artifact upload.
#
# Covers: SshHost.soft_join_queue enqueues via acquire(ready=False) and caches
# the manager; get_core_allocation reuses the SAME cached manager (same
# entry_id, no second construction); soft_join_queue swallows errors
# (best-effort); base/LocalHost soft_join_queue is a no-op; and an end-to-end,
# no-hardware integration proving the upload-window slot reservation preserves
# FIFO ordering against a real locks.json + helper.
# =============================================================================


class _RecordingManager:
    """Fake CoreLockManager that records acquire(ready=...) calls and replays outcomes."""

    def __init__(self, entry_id: str = "entry-abc") -> None:
        self.entry_id = entry_id
        # Mirrors CoreLockManager.collector so SshHost._ensure_core_lock_manager's
        # collector-identity reuse guard can be exercised with this fake.
        self.collector = None
        self.acquire_ready_args: list[bool] = []
        self.probe_calls = 0
        self._seq: list = []
        self.dequeue_calls = 0
        self.released: list[list[int]] = []

    def probe(self, num_logical_cores, lnc_config, timeout_seconds=None):  # noqa: ARG002
        # Benign under-patience peek so the caching/reuse tests proceed to
        # acquire(ready=False).
        self.probe_calls += 1
        return AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=0)

    def acquire(self, num_logical_cores, lnc_config, ready=True):  # noqa: ARG002
        self.acquire_ready_args.append(ready)
        if self._seq:
            item = self._seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10)

    def record_contention_metrics(self) -> None:
        pass

    def release(self, core_ids) -> None:
        self.released.append(list(core_ids))

    def dequeue(self) -> None:
        self.dequeue_calls += 1


class TestSoftJoinQueue:
    """SshHost.soft_join_queue + manager caching."""

    def test_soft_join_enqueues_ready_false_and_caches_manager(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager()
        ctor = MagicMock(return_value=rec)
        with (
            patch.object(host_management, "CoreLockManager", ctor),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
        ):
            host.soft_join_queue(collector=_cm_collector(), collective_ranks=2, lnc_config=2)

        # Exactly one ready=False enqueue; manager cached on the host.
        assert rec.probe_calls == 1
        assert rec.acquire_ready_args == [False]
        assert ctor.call_count == 1
        assert host._core_lock_manager is rec

    def test_get_core_allocation_reuses_soft_joined_manager(self, tmp_path) -> None:
        """get_core_allocation must reuse the cached manager (same entry_id, no rebuild)."""
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager(entry_id="entry-xyz")
        ctor = MagicMock(return_value=rec)
        clock = _FakeClock()
        # Same collector across soft-join and the real allocation (one attempt):
        # the reuse guard binds the cached manager to its owning collector, so
        # rec must report that same collector to be reused.
        collector = _cm_collector()
        rec.collector = collector
        with (
            patch.object(host_management, "CoreLockManager", ctor),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            host.soft_join_queue(collector=collector, collective_ranks=1, lnc_config=2)
            entry_after_soft_join = host._core_lock_manager.entry_id
            # Now the real allocation commits ready=True on the SAME manager.
            rec._seq = [AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=[0], physical_cores=[0, 1])]
            with host.get_core_allocation(collector=collector, collective_ranks=1, lnc_config=2) as alloc:
                assert alloc.logical_core_ids == [0]

        # No second construction; same entry_id; first poll ready=False, then ready=True.
        assert ctor.call_count == 1
        assert entry_after_soft_join == "entry-xyz"
        assert rec.acquire_ready_args == [False, True]
        assert rec.released == [[0, 1]]
        # Cache cleared on teardown so the next attempt starts fresh.
        assert host._core_lock_manager is None

    def test_soft_join_swallows_exceptions(self, tmp_path) -> None:
        """Best-effort: any failure during soft-join is logged and swallowed."""
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager()
        rec._seq = [LockAcquisitionError("boom")]
        with (
            patch.object(host_management, "CoreLockManager", MagicMock(return_value=rec)),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
        ):
            # Must NOT raise even though acquire raises.
            host.soft_join_queue(collector=_cm_collector(), collective_ranks=1, lnc_config=2)
        assert rec.probe_calls == 1
        assert rec.acquire_ready_args == [False]

    def _soft_join(self, host, fake_mgr, **kwargs) -> None:
        """Drive soft_join_queue with a scripted fake manager (no real SSH)."""
        with (
            patch.object(host_management, "CoreLockManager", MagicMock(return_value=fake_mgr)),
            patch.object(host, "_get_remote_executor", return_value=MagicMock()),
            patch.object(host, "get_instance_type", return_value="trn1"),
        ):
            host.soft_join_queue(collector=_cm_collector(), **kwargs)

    def test_soft_join_over_patience_queued_rotates_without_enqueue(self, tmp_path) -> None:
        """An over-patience QUEUED probe rotates the host BEFORE any enqueue: it
        raises QueuePatienceRotation (a TimeoutException) without ever calling
        acquire or dequeue, and no contention metrics are flushed (none have
        accrued yet — the host was never joined)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [],
            probe_outcomes=[
                AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=DEFAULT_PATIENCE_SECONDS + 1)
            ],
        )
        with pytest.raises(QueuePatienceRotation, match="rotating host") as exc_info:
            self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert isinstance(exc_info.value, TimeoutException)
        # Exactly one probe, zero enqueue, zero dequeue, no contention flush.
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 0
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_over_patience_retains_cached_manager(self, tmp_path) -> None:
        """After an over-patience probe rotation the cached manager is RETAINED:
        a patience rotation does not mark the host failed, so a re-selection of
        this same host reuses the unused/enqueued entry (or transparently
        re-enqueues) on the next attempt — soft-join never drops the cache."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [],
            probe_outcomes=[
                AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=DEFAULT_PATIENCE_SECONDS + 1)
            ],
        )
        with pytest.raises(QueuePatienceRotation, match="rotating host"):
            self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        # No enqueue and no dequeue (probe never joined the queue) ...
        assert fake_mgr.acquire_calls == 0
        assert fake_mgr.dequeue_calls == 0
        # ... and the cached manager is retained for the next attempt.
        assert host._core_lock_manager is fake_mgr

    def test_soft_join_within_patience_queued_does_not_raise(self, tmp_path) -> None:
        """A QUEUED probe whose ETA is within patience falls through to a single
        ready=False enqueue (no dequeue, no raise)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=DEFAULT_PATIENCE_SECONDS)],
            probe_outcomes=[AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=DEFAULT_PATIENCE_SECONDS)],
        )
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_draining_over_patience_does_not_raise(self, tmp_path) -> None:
        """A DRAINING probe never rotates even with an over-patience ETA
        (should_rotate returns False while draining), so it falls through to
        acquire(ready=False)."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=DEFAULT_PATIENCE_SECONDS)],
            probe_outcomes=[
                AllocationOutcome(status=AllocationStatus.DRAINING, worst_case_eta=DEFAULT_PATIENCE_SECONDS + 1)
            ],
        )
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_draining_invokes_should_rotate_with_draining_true(self, tmp_path) -> None:
        """A DRAINING probe outcome must route the draining state into
        should_rotate (draining=True), making should_rotate the single
        suppression point rather than the status guard short-circuiting."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=DEFAULT_PATIENCE_SECONDS)],
            probe_outcomes=[
                AllocationOutcome(status=AllocationStatus.DRAINING, worst_case_eta=DEFAULT_PATIENCE_SECONDS + 1)
            ],
        )
        calls: list[bool] = []

        def _spy(worst_case_eta, draining, patience_seconds):
            calls.append(draining)
            return should_rotate(worst_case_eta, draining, patience_seconds=patience_seconds)

        with patch.object(host_management, "should_rotate", _spy):
            self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert calls == [True]

    def test_soft_join_probe_error_is_swallowed_without_acquire(self, tmp_path) -> None:
        """A probe RPC error is swallowed (no raise) and never reaches acquire or
        dequeue — the hard acquire in get_core_allocation takes over."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager([], probe_outcomes=[LockAcquisitionError("boom")])
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 0
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_acquire_error_is_swallowed_without_dequeue(self, tmp_path) -> None:
        """With an under-patience probe, an acquire RPC error is swallowed (no
        raise) and never triggers a dequeue."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [LockAcquisitionError("boom")],
            probe_outcomes=[AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=0)],
        )
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_unexpected_status_probe_error_is_swallowed_without_acquire(self, tmp_path) -> None:
        """An ERROR/unexpected-status probe surfaces from CoreLockManager.probe as a
        LockAcquisitionError; soft-join ignores it best-effort (no raise) and,
        because probe and acquire share one try, the round never reaches acquire
        or dequeue."""
        host = _make_ssh_host(tmp_path)
        fake_mgr = _FakeManager(
            [],
            probe_outcomes=[LockAcquisitionError("[test-host] Unexpected lock status: LockStatus.ALLOCATED")],
        )
        # Must NOT raise even though probe raises.
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 0
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

    def test_soft_join_patience_disabled_falls_through_to_acquire(self, tmp_path) -> None:
        """With patience disabled (patience_seconds <= 0) should_rotate returns
        False, so an over-patience probe ETA does NOT rotate: the round falls
        through to a single acquire(ready=False) with no QueuePatienceRotation
        and no dequeue."""
        host = _make_ssh_host(tmp_path, patience_seconds=-1)
        fake_mgr = _FakeManager(
            [AllocationOutcome(status=AllocationStatus.QUEUED, position=1, worst_case_eta=DEFAULT_PATIENCE_SECONDS)],
            probe_outcomes=[
                AllocationOutcome(status=AllocationStatus.QUEUED, worst_case_eta=DEFAULT_PATIENCE_SECONDS + 1)
            ],
        )
        # Over-patience ETA but rotation disabled: must NOT raise.
        self._soft_join(host, fake_mgr, collective_ranks=1, lnc_config=2)
        assert fake_mgr.probe_calls == 1
        assert fake_mgr.acquire_calls == 1
        assert fake_mgr.dequeue_calls == 0

    def test_base_and_local_soft_join_is_noop(self, tmp_path) -> None:
        """LocalHost inherits the base no-op; it must not enqueue or error."""
        # LocalHost does not override the base no-op (uses the file-lock path).
        assert LocalHost.soft_join_queue is host_management.Host.soft_join_queue
        local = LocalHost(
            "/opt/aws/neuron/bin",
            "localhost",
            str(tmp_path),
            core_reset_strategy=CoreResetStrategy(exclusive_run=False),
        )
        assert local.soft_join_queue(collector=MagicMock(), collective_ranks=2, lnc_config=2) is None


class TestSoftJoinIntegration(TestSshHostPollLoopIntegration):
    """No-hardware integration: soft-join slot reservation preserves FIFO order."""

    def test_soft_join_reserves_upload_window_slot_and_preserves_fifo(self, tmp_path) -> None:
        """A soft-joins (ready=False) at slot 0 without committing even though cores
        are free; B (ready=True) cannot jump ahead; A then commits (ready=True)."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=0.0)
        self._write_locks(locks_json)  # all 8 cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr_a = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )
            mgr_b = CoreLockManager(
                "fakehost",
                total_physical_cores=8,
                collector=NoopMetricsCollector(),
                executor=executor,
                host_locking_version=3,
            )

            # A soft-joins during "upload": enqueues at position 0 but does NOT
            # commit, even though all cores are free (ready=False never grabs cores).
            out_a = mgr_a.acquire(num_logical_cores=4, lnc_config=2, ready=False)
            assert out_a.status == AllocationStatus.QUEUED
            assert out_a.position == 0
            queue = self._read_queue(locks_json)
            assert [e["entry_id"] for e in queue] == [mgr_a.entry_id]
            # No cores claimed by the soft-join (slot is metadata-only).
            import json as _json

            with open(locks_json) as f:
                assert _json.load(f).get("physical_neuron_cores_lock_timeout", {}) == {}

            # B arrives ready-to-commit but must NOT jump ahead of A's reserved slot.
            out_b = mgr_b.acquire(num_logical_cores=4, lnc_config=2, ready=True)
            assert out_b.status == AllocationStatus.QUEUED
            assert [e["entry_id"] for e in self._read_queue(locks_json)] == [mgr_a.entry_id, mgr_b.entry_id]

            # A finishes uploading and commits on the SAME entry_id -> ALLOCATED.
            out_a2 = mgr_a.acquire(num_logical_cores=4, lnc_config=2, ready=True)
            assert out_a2.status == AllocationStatus.ALLOCATED
            assert len(out_a2.logical_cores) == 4


# =============================================================================
# Per-attempt CoreLockManager lifetime (F2).
#
# The HostManager keeps ONE SshHost per alias for the whole pytest-worker
# session. Before the fix, SshHost cached its CoreLockManager once per host and
# never cleared it, so the manager's entry_id, captured collector, and latched
# fairness counters bled across every later test on the
# same host. These tests pin the fix: the cached manager is bound to its owning
# collector and dropped on teardown, so each allocation attempt gets a fresh
# manager (new entry_id, rebound collector, reset counters) while soft-join ->
# commit reuse within one attempt (same collector) is preserved.
# =============================================================================


class _RecordingCollector(NoopMetricsCollector):
    """Captures record_metric/record_timer calls so per-collector attribution is observable."""

    def __init__(self, test_name: str = "rec") -> None:
        self.test_name = test_name
        self.metrics: list[tuple] = []
        self.timers: list[tuple] = []

    def record_metric(self, name, value, unit: str = "None") -> None:  # noqa: D102
        self.metrics.append((name, value, unit))

    def record_timer(self, name, duration_seconds) -> None:  # noqa: D102
        self.timers.append((name, duration_seconds))


class TestManagerLifetimePerAttempt(TestSshHostPollLoopIntegration):
    """No-hardware integration proving per-attempt manager lifetime (F2)."""

    def _metric_values(self, collector, name):
        return [v for (n, v, _u) in collector.metrics if n == name]

    def test_manager_resets_across_attempts_on_same_host(self, tmp_path) -> None:
        """Two attempts on the SAME SshHost (different collectors) each get a fresh
        manager: distinct entry_id, rebound collector, and per-attempt counters
        recorded on the owning collector only.

        Before the fix this was impossible: the cached manager's fairness counters
        latched after attempt #1, attributing attempt #2's metrics onto
        attempt #1's collector. This exercises the exact cross-test bleed F2
        describes over the real helper + a temp locks.json.
        """
        import json as _json

        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Attempt #1: cores busy with ETA ~100s (<= patience, so no rotation) ->
        # the caller enqueues, then cores are freed before the 2nd poll so it
        # commits THROUGH the queue.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100)

        def free_cores_on_second_call(call_n, path):
            if call_n == 2:
                with open(path) as f:
                    state = _json.load(f)
                state["physical_neuron_cores_lock_timeout"] = {}
                with open(path, "w") as f:
                    _json.dump(state, f)

        executor = _HelperExecutor(helpers_file, lock_file, locks_json, on_call=free_cores_on_second_call)
        host = _make_ssh_host(tmp_path)
        col1 = _RecordingCollector("attempt-1")
        col2 = _RecordingCollector("attempt-2")

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=col1, collective_ranks=4, lnc_config=2, timeout_seconds=300) as a1:
                entry1 = host._core_lock_manager.entry_id
                assert host._core_lock_manager.collector is col1
                assert len(a1.logical_core_ids) == 4
            # Teardown drops the cache so the next attempt cannot reuse it.
            assert host._core_lock_manager is None

            # Attempt #2: freed cores -> commits THROUGH the queue on a later poll.
            with host.get_core_allocation(collector=col2, collective_ranks=4, lnc_config=2, timeout_seconds=300) as a2:
                entry2 = host._core_lock_manager.entry_id
                assert host._core_lock_manager.collector is col2
                assert len(a2.logical_core_ids) == 4
            assert host._core_lock_manager is None

        # Distinct entry_id minted per attempt (not once per host).
        assert entry1 != entry2
        # Each attempt's contention counter is recorded exactly once on its OWN
        # collector (proving the manager + collector binding reset per attempt).
        assert len(self._metric_values(col1, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1
        assert len(self._metric_values(col2, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1

    def test_distinct_collectors_get_distinct_managers_no_cross_attribution(self, tmp_path) -> None:
        """Two back-to-back attempts (different collectors) on the same host
        mint distinct managers/entry_ids; neither collector receives the other's metrics."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free -> instant ALLOCATED
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)
        host = _make_ssh_host(tmp_path)
        col1 = _RecordingCollector("attempt-1")
        col2 = _RecordingCollector("attempt-2")

        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            with host.get_core_allocation(collector=col1, collective_ranks=4, lnc_config=2) as a1:
                mgr1 = host._core_lock_manager
                entry1 = mgr1.entry_id
                assert mgr1.collector is col1
                assert len(a1.logical_core_ids) == 4
            with host.get_core_allocation(collector=col2, collective_ranks=4, lnc_config=2) as a2:
                mgr2 = host._core_lock_manager
                entry2 = mgr2.entry_id
                assert mgr2.collector is col2
                assert len(a2.logical_core_ids) == 4

        assert mgr1 is not mgr2
        assert entry1 != entry2
        # Each attempt's metrics are attributed to its own collector only.
        assert len(self._metric_values(col1, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1
        assert len(self._metric_values(col2, MetricName.CORE_LOCK_NO_CORES_COUNT)) == 1


# =============================================================================
# End-to-end manager <-> helper integration under concurrent reservations (IT-4 / T9).
#
# These tests drive one or more REAL CoreLockManager instances (and, where the
# stay/rotate decision is under test, the real SshHost poll loop) against the
# REAL lock helper (remote_lock_scripts) over a single shared temp locks.json
# via _HelperExecutor. They prove the concurrent per-entry reservation model
# behaves correctly end-to-end: disjoint concurrent landing, soft-join entry_id
# reuse, manager-layer W-lazy (a ready waiter is not blocked by an uploading
# predecessor), patience that stays on a within-patience concurrent ETA,
# long-upload bump + eventual commit, RESERVE_EXPIRED observability in the event
# log, and drain-then-grant in FIFO order.
#
# T9: the manager consumes the SAME wire LockResult surface as before, so no
# manager source change is required -- these tests pass against the unmodified
# manager/poll loop, confirming transparency.
# =============================================================================


class TestManagerConcurrentReservationE2E(TestSshHostPollLoopIntegration):
    """Multiple managers over one shared locks.json under concurrent reservations."""

    def _mk_mgr(self, executor) -> CoreLockManager:
        """A real CoreLockManager bound to the shared helper executor (8-core host)."""
        return CoreLockManager(
            "fakehost",
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=executor,
            host_locking_version=3,
        )

    def _read_events(self, locks_json) -> list:
        """Read the helper's daily-rotated event log(s) beside locks.json."""
        import glob
        import json as _json

        d = os.path.dirname(locks_json)
        events = []
        for fn in sorted(glob.glob(os.path.join(d, "events-*.log"))):
            with open(fn) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(_json.loads(line))
        return events

    def _free_all_cores(self, locks_json) -> None:
        """Clear all live core locks while preserving the queue/drain fields."""
        import json as _json

        with open(locks_json) as f:
            state = _json.load(f)
        state["physical_neuron_cores_lock_timeout"] = {}
        with open(locks_json, "w") as f:
            _json.dump(state, f)

    def _clear_drain(self, locks_json) -> None:
        """Disable drain while preserving the queue (does not wipe the file)."""
        import json as _json

        with open(locks_json) as f:
            state = _json.load(f)
        state.pop("draining_enabled_with_timeout", None)
        with open(locks_json, "w") as f:
            _json.dump(state, f)

    def _assert_disjoint(self, allocations) -> None:
        """No physical core may appear in two managers' allocations simultaneously."""
        seen: set[int] = set()
        for cores in allocations:
            block = set(cores or [])
            overlap = block & seen
            assert not overlap, f"physical core double-allocated across managers: {sorted(overlap)}"
            seen |= block

    def test_multiple_managers_land_concurrently_disjoint(self, tmp_path) -> None:
        """N managers (distinct entry_ids) over ONE locks.json each request w-cores
        that together fit (4 x 2 cores on an 8-core host). Each soft-joins
        (ready=False) then commits (ready=True). Every manager reaches ALLOCATED on
        DISJOINT physical cores, no core is double-allocated, and the helper event
        log contains only ENQUEUE/COMMIT (no ACQUIRE -- the fast path is gone)."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all 8 cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            managers = [self._mk_mgr(executor) for _ in range(4)]
            # Soft-join: each anchors a FIFO slot (ready=False) without grabbing cores.
            for mgr in managers:
                out = mgr.acquire(num_logical_cores=1, lnc_config=2, ready=False)
                assert out.status == AllocationStatus.QUEUED
            # Four distinct entry_ids enqueued, no cores claimed by the soft-joins.
            queue = self._read_queue(locks_json)
            assert {e["entry_id"] for e in queue} == {m.entry_id for m in managers}
            import json as _json

            with open(locks_json) as f:
                assert _json.load(f).get("physical_neuron_cores_lock_timeout", {}) == {}

            # Commit: one ready=True poll per manager (the soft-joined entry already
            # exists, so reconcile reserves its block and it commits the same poll).
            allocations = []
            for mgr in managers:
                out = mgr.acquire(num_logical_cores=1, lnc_config=2, ready=True)
                assert out.status == AllocationStatus.ALLOCATED
                assert out.physical_cores is not None and len(out.physical_cores) == 2
                allocations.append(out.physical_cores)

        # Every manager landed on a disjoint pair; all 8 cores used exactly once.
        self._assert_disjoint(allocations)
        assert sorted(c for block in allocations for c in block) == list(range(8))

        # Event log: only ENQUEUE + COMMIT transitions -- the fast-path ACQUIRE is gone.
        events = self._read_events(locks_json)
        kinds = {e["event"] for e in events}
        assert "ACQUIRE" not in kinds
        assert kinds == {"ENQUEUE", "COMMIT"}
        assert sum(e["event"] == "ENQUEUE" for e in events) == 4
        assert sum(e["event"] == "COMMIT" for e in events) == 4

    def test_soft_join_reuses_entry_id_through_commit(self, tmp_path) -> None:
        """One manager soft-joins (ready=False -> QUEUED) then commits
        (ready=True -> ALLOCATED) on the SAME entry_id: no re-enqueue on the happy
        path (re_enqueued/bump counters stay 0) and the queue never holds two
        entries for it."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr = self._mk_mgr(executor)
            entry_id = mgr.entry_id

            # Poll #1: soft-join (ready=False) -> IN_QUEUE, exactly one queue entry.
            out1 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out1.status == AllocationStatus.QUEUED
            queue = self._read_queue(locks_json)
            assert [e["entry_id"] for e in queue] == [entry_id]

            # Poll #2: commit (ready=True) on the SAME cached entry_id -> ALLOCATED.
            out2 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out2.status == AllocationStatus.ALLOCATED
            assert out2.physical_cores is not None and len(out2.physical_cores) == 4

            # Same entry_id throughout; no re-enqueue and no bump on the happy path.
            assert mgr.entry_id == entry_id
            assert mgr._reenqueue_count == 0
            assert mgr._bump_count == 0
            # Committed + removed -> queue empty (never two entries for this id).
            assert self._read_queue(locks_json) == []

        # Event log: a single ENQUEUE for this entry_id then its COMMIT; no
        # second ENQUEUE, no BUMP/PRUNE/RESERVE_EXPIRED (no re-enqueue happened).
        events = self._read_events(locks_json)
        enqueues = [e for e in events if e["event"] == "ENQUEUE" and e.get("entry_id") == entry_id]
        commits = [e for e in events if e["event"] == "COMMIT" and str(e.get("caller", "")).startswith(entry_id)]
        assert len(enqueues) == 1
        assert len(commits) == 1
        assert not any(e["event"] in {"BUMP", "PRUNE", "RESERVE_EXPIRED"} for e in events)

    def test_ready_waiter_not_blocked_by_uploading_manager(self, tmp_path) -> None:
        """Manager A soft-joins and stays ready=False (a long upload); manager B
        (behind A) soft-joins then polls ready=True. B reaches ALLOCATED while A is
        still not-ready (manager-layer W-lazy: a ready waiter backfills past a
        not-ready predecessor), then A commits onto disjoint cores."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr_a = self._mk_mgr(executor)
            mgr_b = self._mk_mgr(executor)

            # A soft-joins at the head and stays not-ready (uploading).
            out_a = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out_a.status == AllocationStatus.QUEUED
            assert out_a.position == 0
            # B soft-joins behind A, then commits ready=True.
            out_b0 = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out_b0.status == AllocationStatus.QUEUED
            assert out_b0.position == 1
            out_b = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_b.status == AllocationStatus.ALLOCATED
            assert out_b.physical_cores is not None and len(out_b.physical_cores) == 4

            # A is STILL in the queue and STILL not-ready: B backfilled past it.
            queue = self._read_queue(locks_json)
            a_entry = next(e for e in queue if e["entry_id"] == mgr_a.entry_id)
            assert a_entry.get("ready", False) is False

            # A finishes uploading and commits on the SAME entry_id -> disjoint cores.
            out_a2 = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a2.status == AllocationStatus.ALLOCATED
            assert out_a2.physical_cores is not None and len(out_a2.physical_cores) == 4

        self._assert_disjoint([out_b.physical_cores, out_a2.physical_cores])

    def test_patience_stays_on_host_with_concurrent_eta(self, tmp_path) -> None:
        """A board where the OLD serial ETA would exceed patience (so the caller
        would rotate) but the NEW concurrent ETA is within patience because the
        caller fits alongside its predecessor. The manager stays (does not rotate)
        and eventually ALLOCATEs onto cores disjoint from its predecessor."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All 8 cores busy until +170s. Concurrent ETA for a 4-core caller that
        # backfills past a 4-core predecessor is ~170s (<= patience 180s). The OLD
        # serial model would stack the predecessor's 60s hold on top (~230s),
        # exceeding patience and forcing a rotation.
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 170)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            pred = self._mk_mgr(executor)
            caller = self._mk_mgr(executor)

            # Predecessor anchors the head (4 cores). Caller enqueues behind it.
            assert pred.acquire(num_logical_cores=2, lnc_config=2, ready=True).status == AllocationStatus.QUEUED
            out = caller.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out.status == AllocationStatus.QUEUED
            assert out.position == 1
            # Concurrent-aware ETA is within patience -> the manager would NOT rotate.
            assert out.worst_case_eta is not None
            assert out.worst_case_eta <= DEFAULT_PATIENCE_SECONDS
            assert not should_rotate(out.worst_case_eta, draining=False, patience_seconds=DEFAULT_PATIENCE_SECONDS)
            # The serial-model ETA (predecessor hold stacked on the lock) would have
            # exceeded patience, proving the difference is the concurrent model.
            serial_eta = out.worst_case_eta + 60
            assert serial_eta > DEFAULT_PATIENCE_SECONDS

            # Cores free; both reserve disjoint blocks and commit in FIFO order.
            self._free_all_cores(locks_json)
            out_pred = pred.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_caller = caller.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_pred.status == AllocationStatus.ALLOCATED
            assert out_caller.status == AllocationStatus.ALLOCATED

        self._assert_disjoint([out_pred.physical_cores, out_caller.physical_cores])

    def test_long_upload_bumps_then_reenqueues(self, tmp_path) -> None:
        """A manager that stays ready=False past its one-shot readiness grace is
        bumped to the tail by reconcile (a sole-entry bump is a tail re-enqueue).
        The manager observes the bump (bump_count increments; reenqueue_count stays
        0 -- a bump is distinct from a stale-prune re-enqueue) and ultimately still
        commits once ready."""
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr = self._mk_mgr(executor)

            # Poll #1: soft-join at the head (ready=False); no grace opened yet.
            assert mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False).status == AllocationStatus.QUEUED
            # Poll #2: reconcile opens the head's one-shot readiness grace.
            out2 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out2.status == AllocationStatus.QUEUED
            assert mgr._bump_count == 0

            # Let the readiness grace lapse (still well within STALE_THRESHOLD so the
            # entry is bumped to the tail, NOT pruned).
            clock.sleep(remote_lock_scripts.COMMIT_WINDOW + 1)

            # Poll #3: reconcile bumps the stuck not-ready head to the tail.
            out3 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=False)
            assert out3.status == AllocationStatus.QUEUED
            assert mgr._bump_count >= 1
            assert mgr._reenqueue_count == 0

            # Poll #4: now ready -> reserve + commit on the same entry_id.
            out4 = mgr.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out4.status == AllocationStatus.ALLOCATED
            assert out4.physical_cores is not None and len(out4.physical_cores) == 4

        # The bump is observable in the event log; no prune occurred.
        events = self._read_events(locks_json)
        assert any(e["event"] == "BUMP" and e.get("entry_id") == mgr.entry_id for e in events)
        assert not any(e["event"] == "PRUNE" for e in events)
        # Single allocation -> trivially no double-allocation.
        self._assert_disjoint([out4.physical_cores])

    def test_vanished_reserver_reserve_expired_in_event_log(self, tmp_path) -> None:
        """A vanisher V is granted a reservation (by a follower's reconcile) then
        stops polling. Advancing the clock past its commit deadline makes reconcile
        reclaim the tiles and log RESERVE_EXPIRED (observability = event log, not a
        manager metric). A live manager B behind it then commits the freed cores
        once V is finally pruned."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json)  # all cores free
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            vanisher = self._mk_mgr(executor)  # wants ALL 8 cores -> blocks the follower
            follower = self._mk_mgr(executor)  # wants 4 cores, behind the vanisher

            # V enqueues at the head (ready, unreserved).
            assert vanisher.acquire(num_logical_cores=4, lnc_config=2, ready=True).status == AllocationStatus.QUEUED
            # B's poll runs reconcile, which reserves V its 8-core block; B is a
            # barrier behind it. V now holds a reservation it must commit soon.
            out_b = follower.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_b.status == AllocationStatus.QUEUED
            v_entry = next(e for e in self._read_queue(locks_json) if e["entry_id"] == vanisher.entry_id)
            assert v_entry.get("reserved_region")

            # V vanishes. Advance past its commit deadline; B's next poll reclaims
            # the reservation (RESERVE_EXPIRED) -- V keeps its FIFO slot for now.
            clock.sleep(remote_lock_scripts.COMMIT_WINDOW + 1)
            out_b2 = follower.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_b2.status == AllocationStatus.QUEUED
            events = self._read_events(locks_json)
            assert any(e["event"] == "RESERVE_EXPIRED" and e.get("entry_id") == vanisher.entry_id for e in events)

            # B keeps polling (refreshing its own liveness). V never polls, so once
            # it goes stale it is pruned and B commits the freed cores.
            out = out_b2
            for _ in range(20):
                clock.sleep(POLL_PERIOD)
                out = follower.acquire(num_logical_cores=2, lnc_config=2, ready=True)
                if out.status == AllocationStatus.ALLOCATED:
                    break
            assert out.status == AllocationStatus.ALLOCATED
            assert out.physical_cores is not None and len(out.physical_cores) == 4

        # V is gone from the queue and a PRUNE was logged for it.
        assert all(e["entry_id"] != vanisher.entry_id for e in self._read_queue(locks_json))
        events = self._read_events(locks_json)
        assert any(e["event"] == "PRUNE" and e.get("entry_id") == vanisher.entry_id for e in events)
        # The committed cores are B's alone (V vanished) -> no double-allocation.
        self._assert_disjoint([out.physical_cores])

    def test_drain_keeps_managers_polling_then_grants(self, tmp_path) -> None:
        """While draining, managers receive DRAINING and hold their FIFO positions;
        once drain is disabled they commit in FIFO order onto disjoint cores."""
        import random

        random.seed(42)
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # All cores free but the host is draining (far beyond the test).
        self._write_locks(locks_json, draining_until=int(_REALISTIC_EPOCH) + 100000)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json)

        with patch("time.time", clock.time), patch("time.sleep", clock.sleep):
            mgr_a = self._mk_mgr(executor)
            mgr_b = self._mk_mgr(executor)

            # Both enqueue while draining: DRAINING (grants withheld), FIFO positions.
            out_a = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_b = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a.status == AllocationStatus.DRAINING and out_a.position == 0
            assert out_b.status == AllocationStatus.DRAINING and out_b.position == 1

            # Re-poll: still draining, positions preserved (stay-and-probe, no grant).
            out_a2 = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_b2 = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a2.status == AllocationStatus.DRAINING and out_a2.position == 0
            assert out_b2.status == AllocationStatus.DRAINING and out_b2.position == 1

            # Disable drain -> next polls commit in FIFO order (head first).
            self._clear_drain(locks_json)
            out_a3 = mgr_a.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            out_b3 = mgr_b.acquire(num_logical_cores=2, lnc_config=2, ready=True)
            assert out_a3.status == AllocationStatus.ALLOCATED
            assert out_b3.status == AllocationStatus.ALLOCATED

        self._assert_disjoint([out_a3.physical_cores, out_b3.physical_cores])


class _TimerRecordingCollector(NoopMetricsCollector):
    """Collector spy that captures timed metric names. The recoverable-host wait uses
    ``collector.timer(...)``, whose context records via ``record_timer`` on exit — a bare
    MagicMock's ``timer()`` wouldn't run that context, so we inherit the real ``timer`` (it
    returns a _TimerContext bound to this collector) and just capture the name."""

    def __init__(self):
        self.timed: list[str] = []

    def record_timer(self, name: str, duration_seconds: float) -> None:
        self.timed.append(name)


class TestHostStateAssignment:
    """Tests for HostManager assignment against the host-state store."""

    def setup_method(self):
        self._patchers = []

    def teardown_method(self):
        # Reverse order: patches stacked on the same target must be undone LIFO, or the
        # target is left pointing at an intermediate mock instead of the real object.
        for p in reversed(self._patchers):
            p.stop()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def _make_manager(self, temp_dir, target_hosts, needs_local_host=None, hosts_recoverable=False):
        def fake_make(alias, **_kwargs):
            return MagicMock(get_host_id=MagicMock(return_value=alias), connection=MagicMock())

        # Patch SshHost so lazily-built hosts yield alias-carrying mocks rather
        # than opening real connections.
        patcher = patch("test.utils.host_management.SshHost", side_effect=fake_make)
        patcher.start()
        self._patchers.append(patcher)
        # Build + initialize the store and inject it, mirroring the factory (which needs a
        # local host when there are no static hosts — i.e. a local hardware run).
        resolved_needs_local = (len(target_hosts) < 1) if needs_local_host is None else needs_local_host
        return HostManager(
            state_store=_initialized_store(temp_dir, target_hosts),
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=resolved_needs_local,
            hosts_recoverable=hosts_recoverable,
        )

    def _assign(self, manager, platform=Platforms.TRN2, collector=None, num_of_physical_cores_needed=1):
        # __get_host_assignment__ takes a real collector (Noop when metrics are off); default
        # to one rather than forwarding None so this helper matches the production contract.
        collector = collector or NoopMetricsCollector()
        return manager.__get_host_assignment__(platform, num_of_physical_cores_needed, collector=collector)

    def test_no_available_host_raises(self, temp_dir):
        # Non-recoverable pool (default): an empty claim fails immediately, no wait. The
        # no-hosts terminal is a FleetEmptyError (same kind as the poisoned path), not a
        # bare Exception.
        manager = self._make_manager(temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)])
        manager.mark_host_unavailable("h1")
        with patch("test.utils.host_management.time.sleep") as sleep:
            with pytest.raises(FleetEmptyError, match="No available hosts for platform trn2"):
                self._assign(manager)
        sleep.assert_not_called()  # recoverable=False → no waiting

    def test_recoverable_pool_waits_then_succeeds_when_host_appears(self, temp_dir):
        # Recoverable pool: empty claim waits; a background source adds a host during
        # the wait (simulated in the sleep stub), and the retry then succeeds.
        manager = self._make_manager(temp_dir, [], needs_local_host=False, hosts_recoverable=True)

        def add_host_during_wait(_seconds):
            _add_host(manager.state_store, "newhost", Platforms.TRN2)

        with patch("test.utils.host_management.time.sleep", side_effect=add_host_during_wait) as sleep:
            host = self._assign(manager)
        assert host.get_host_id() == "newhost"
        sleep.assert_called_once()  # appeared on the first wait

    def test_recoverable_pool_polls_until_poisoned_then_fails_fast(self, temp_dir):
        # Recoverable pool, nothing online yet: the worker polls (no per-worker deadline)
        # until the background re-resolver poisons the pool, then fails fast. Here that
        # re-resolver is simulated by poisoning on the 3rd poll's sleep.
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        manager.mark_host_unavailable("h1")

        polls = {"n": 0}

        def poison_on_third_poll(_seconds):
            polls["n"] += 1
            if polls["n"] >= 3:
                _poison_platform(temp_dir, Platforms.TRN2)

        with patch("test.utils.host_management.time.sleep", side_effect=poison_on_third_poll) as sleep:
            with pytest.raises(FleetEmptyError, match="perpetually empty"):
                self._assign(manager)
        # Polled repeatedly (no fixed attempt cap) until poison landed.
        assert sleep.call_count == 3

    def test_recoverable_wait_times_out_if_re_resolver_never_advances(self, temp_dir):
        # Backstop for a dead re-resolver: nothing ever comes online AND poison never lands
        # (the background re-resolver stopped advancing). The wait must not spin forever —
        # it hits the deadline and raises a DISTINCT TimeoutException (not FleetEmptyError:
        # the fleet was never declared empty). Drive a fake clock so the deadline trips
        # without real waiting; each sleep advances time past the budget.
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        manager.mark_host_unavailable("h1")  # never recovered; poison never set

        clock = {"t": 1000.0}

        def advance(seconds):
            clock["t"] += seconds

        with (
            patch("test.utils.host_management.time.time", side_effect=lambda: clock["t"]),
            patch("test.utils.host_management.time.sleep", side_effect=advance),
        ):
            with pytest.raises(TimeoutException, match="re-resolver likely stopped advancing"):
                self._assign(manager)
        # The pool was poisoned: a claim that finds no host must NOT wait — it raises
        # FleetEmptyError immediately (responsibility for availability moved to the
        # background re-resolver; workers stop rediscovering a known-empty fleet).
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        manager.mark_host_unavailable("h1")
        _poison_platform(temp_dir, Platforms.TRN2)
        with patch("test.utils.host_management.time.sleep") as sleep:
            with pytest.raises(FleetEmptyError, match="perpetually empty"):
                self._assign(manager)
        sleep.assert_not_called()  # poisoned -> no wait at all

    def test_poison_landing_mid_wait_short_circuits(self, temp_dir):
        # Not poisoned at claim time, so the worker starts waiting; the background
        # re-resolver poisons the fleet during the first wait (simulated in the sleep
        # stub). The next cycle's poison check short-circuits with FleetEmptyError instead
        # of burning the remaining wait budget.
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        manager.mark_host_unavailable("h1")

        def poison_during_wait(_seconds):
            _poison_platform(temp_dir, Platforms.TRN2)

        with patch("test.utils.host_management.time.sleep", side_effect=poison_during_wait) as sleep:
            with pytest.raises(FleetEmptyError, match="perpetually empty"):
                self._assign(manager)
        sleep.assert_called_once()  # bailed after the first wait, not the full budget

    def test_unpoisoned_recoverable_wait_still_succeeds(self, temp_dir):
        # Regression guard: the poison check must not break the normal recoverable
        # path — an un-poisoned pool that gets a host during the wait still succeeds.
        manager = self._make_manager(temp_dir, [], needs_local_host=False, hosts_recoverable=True)

        def add_host_during_wait(_seconds):
            _add_host(manager.state_store, "newhost", Platforms.TRN2)

        with patch("test.utils.host_management.time.sleep", side_effect=add_host_during_wait):
            host = self._assign(manager)
        assert host.get_host_id() == "newhost"

    def test_recoverable_wait_records_metrics_on_success(self, temp_dir):
        # The recoverable-wait path must record RECOVERABLE_HOST_WAIT_TIME and
        # _ATTEMPTS so a stuck pool is visible in metrics. Previously this branch
        # (guarded by `if collector is not None`) had zero test coverage. A host
        # appears on the first wait -> one attempt recorded.
        manager = self._make_manager(temp_dir, [], needs_local_host=False, hosts_recoverable=True)

        def add_host_during_wait(_seconds):
            _add_host(manager.state_store, "newhost", Platforms.TRN2)

        collector = _TimerRecordingCollector()
        with patch("test.utils.host_management.time.sleep", side_effect=add_host_during_wait):
            self._assign(manager, collector=collector)

        assert MetricName.RECOVERABLE_HOST_WAIT_TIME in collector.timed

    def test_recoverable_wait_records_metrics_when_poisoned(self, temp_dir):
        # When the wait ends by poisoning (fail fast), the wait-time metric is still recorded
        # (the timer records on __exit__, so it fires even though the wait raises).
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        manager.mark_host_unavailable("h1")
        collector = _TimerRecordingCollector()

        polls = {"n": 0}

        def poison_on_second_poll(_seconds):
            polls["n"] += 1
            if polls["n"] >= 2:
                _poison_platform(temp_dir, Platforms.TRN2)

        with patch("test.utils.host_management.time.sleep", side_effect=poison_on_second_poll):
            with pytest.raises(FleetEmptyError, match="perpetually empty"):
                self._assign(manager, collector=collector)
        assert MetricName.RECOVERABLE_HOST_WAIT_TIME in collector.timed

    def test_lazily_builds_host_added_by_re_resolver(self, temp_dir):
        # Manager initialized with only h1; a background re-resolution later adds h2 to the file.
        manager = self._make_manager(temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)])
        manager.mark_host_unavailable("h1")  # force selection onto h2
        _add_host(manager.state_store, "h2", Platforms.TRN2)
        host = self._assign(manager)
        # h2 wasn't in the original target_hosts; it must be built on demand.
        assert host.get_host_id() == "h2"
        assert manager.target_hosts["h2"] is host

    def test_needs_local_host_sets_up_the_local_host(self, temp_dir):
        # A local hardware run (needs_local_host=True) detects the platform, probes core count,
        # and adds the local host to the state store.
        with patch("test.utils.host_management.detect_local_platform", return_value=Platforms.TRN2):
            with patch("test.utils.host_management.LocalHost.get_total_physical_cores", return_value=8):
                manager = self._make_manager(temp_dir, [], needs_local_host=True)
        assert (
            HostRecord(resolved=ResolvedHost(manager.LOCAL_HOST_ID, Platforms.TRN2, num_physical_cores=8))
            in manager.state_store.read().hosts
        )

    def test_local_host_not_set_up_when_not_needed(self, temp_dir):
        # Compile-only / trace-only / simulation local run: the run never claims a host, so
        # make_host_manager passes needs_local_host=False. No local-host setup runs — so neuron-ls
        # is never invoked (it would fail on a non-hardware host) and no store row is added.
        with patch("test.utils.host_management.detect_local_platform") as detect:
            with patch("test.utils.host_management.LocalHost.get_total_physical_cores") as cores:
                manager = self._make_manager(temp_dir, [], needs_local_host=False)
        detect.assert_not_called()  # no platform detection...
        cores.assert_not_called()  # ...and no capacity probe (both run neuron-ls)
        assert manager.LOCAL_HOST_ID not in manager.target_hosts  # no local host registered
        assert manager.state_store.read() is None  # no store row initialized


def _add_host(store, alias, host_type, num_physical_cores=64):
    """Test seam: inject a host (as a background re-resolution would) via the store's private
    primitive. Defaults to 64 cores so the host is eligible for typical (small) requests."""
    store._update(
        lambda state: HostState(
            hosts=[
                *state.hosts,
                HostRecord(
                    resolved=ResolvedHost(ssh_host=alias, host_type=host_type, num_physical_cores=num_physical_cores)
                ),
            ],
            poisoned=state.poisoned,
        )
    )
