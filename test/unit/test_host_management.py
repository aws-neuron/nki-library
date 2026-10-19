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

import collections
import contextlib
import json
import os
import random
import re
import subprocess
import tempfile
from contextlib import closing
from dataclasses import replace
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from paramiko import SSHException

from test.utils.common_dataclasses import Platforms, ResolvedHost, TargetHost
from test.utils.exceptions import (
    FleetEmptyError,
    InferenceException,
    RequestTooLargeError,
    TimeoutException,
)
from test.utils.host_management import (
    Host,
    HostManager,
    LocalHost,
    SshHostGroup,
    _run_neuron_ls,
    detect_local_neuron_devices,
    detect_local_platform,
)
from test.utils.host_state import (
    HostGroup,
    HostRecord,
    HostState,
    HostStateStore,
    OwnerHostStateStore,
    temporary_random_seed,
)
from test.utils.metrics_collector import MetricName

pytestmark = pytest.mark.usefixtures("controller_environment")


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


def _set_ceiling(base_dir, ceiling):
    """Set the per-platform provisionable ceiling in the state file at ``base_dir``.

    Writing the ceiling is a controller action (via apply_resolved), so it goes through an
    OwnerHostStateStore — matching how a real run publishes it — while preserving the existing
    host rows. A worker's claim then observes the ceiling via the shared file.
    """
    owner = OwnerHostStateStore(base_dir)
    current = owner.read()
    hosts = [rec.resolved for rec in current.hosts] if current else []
    owner.apply_resolved(hosts, provisionable_ceiling=ceiling)


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
        ranks by). Host construction is patched so claimed hosts are alias-carrying SshHost mocks."""
        target_hosts = [TargetHost(ssh_host=a, host_type=Platforms.TRN2) for a in host_cores]

        def fake_get_host(manager, alias):
            host = manager.target_hosts.get(alias)
            if host is None:
                host = create_autospec(SshHost, instance=True)
                host.get_host_id.return_value = alias
                manager.target_hosts[alias] = host
            return host

        patcher = patch.object(HostManager, "_get_host", fake_get_host)
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

    @staticmethod
    def _member(
        alias: str, logical_host_id: str, rank: int, *, size: int, depth: int = 0, cores: int = 128
    ) -> HostRecord:
        """Build a ``cores``-core TRN2 HostRecord belonging to multi-node deployment ``logical_host_id``."""
        resolved = ResolvedHost(
            ssh_host=alias,
            host_type=Platforms.TRN2,
            num_physical_cores=cores,
            logical_host_id=logical_host_id,
            logical_host_rank=rank,
            logical_host_size=size,
        )
        return HostRecord(resolved=resolved, work_queue_depth=depth)

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

    def test_retry_releases_exactly_the_claimed_members(self, temp_dir):
        """A multi-node claim is released from the members it reserved, even if another member of
        the same deployment joins the pool while the attempt is held."""
        manager = self._build_manager(temp_dir, {})
        owner = OwnerHostStateStore(temp_dir)
        owner._update(
            lambda state: HostState(hosts=[self._member("m0", "L", 0, size=3), self._member("m1", "L", 1, size=3)])
        )

        chosen = []
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=MagicMock(),
                collective_ranks=130,
                lnc_config=1,
                backoff_seconds=0,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    chosen.append(host.get_host_id())
                    assert self._depths(manager) == {"m0": 130, "m1": 130}
                    owner._update(
                        lambda state: replace(state, hosts=[*state.hosts, self._member("m2", "L", 2, size=3, depth=64)])
                    )
        assert chosen == ["m0"]
        assert self._depths(manager) == {"m0": 0, "m1": 0, "m2": 64}

    def test_retry_releases_claim_when_host_construction_fails(self, temp_dir):
        """A claim is released even when building the host connection for it fails."""
        manager = self._build_manager(temp_dir, {})
        owner = OwnerHostStateStore(temp_dir)
        owner._update(
            lambda state: HostState(hosts=[self._member("m0", "L", 0, size=2), self._member("m1", "L", 1, size=2)])
        )

        with patch.object(HostManager, "_get_host", side_effect=ValueError("bad host config")):
            with pytest.raises(ValueError, match="bad host config"):
                with closing(
                    manager.get_host_assignment_with_retry(
                        platform_target=Platforms.TRN2,
                        collector=MagicMock(),
                        collective_ranks=130,
                        lnc_config=1,
                        backoff_seconds=0,
                    )
                ) as host_generator:
                    for host_attempt in host_generator:
                        with host_attempt:
                            pass
        assert self._depths(manager) == {"m0": 0, "m1": 0}

    def test_patience_rotation_excludes_busy_multinode_group(self, temp_dir):
        """A busy multi-node group is excluded from the next pick, and the busy set resets once
        every eligible group has been tried, so rotation alternates between the groups."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager(temp_dir, {})
        owner = OwnerHostStateStore(temp_dir)

        # B's seeded depth of 1 makes A the strict first pick whenever neither group is excluded;
        # each attempt's claim is released before the next pick, so it never affects the order.
        seeded = {"a0": 0, "a1": 0, "b0": 1, "b1": 1}
        owner._update(
            lambda state: HostState(
                hosts=[
                    self._member("a0", "A", 0, size=2, depth=seeded["a0"]),
                    self._member("a1", "A", 1, size=2, depth=seeded["a1"]),
                    self._member("b0", "B", 0, size=2, depth=seeded["b0"]),
                    self._member("b1", "B", 1, size=2, depth=seeded["b1"]),
                ]
            )
        )

        chosen = []
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=MagicMock(),
                collective_ranks=130,
                lnc_config=1,
                backoff_seconds=0,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as host:
                    chosen.append(host.get_host_id())
                    if len(chosen) <= 3:
                        raise QueuePatienceRotation("queue ETA exceeds patience")
        assert chosen == ["a0", "b0", "a0", "b0"]
        assert self._depths(manager) == seeded

    @staticmethod
    def _three_of_four(depth: int = 0) -> list[HostRecord]:
        return [TestSizeBasedHostRouting._member(f"t{rank}", "T", rank, size=4, depth=depth) for rank in range(3)]

    # 129 ranks at LNC2 (258 physical cores). The idle 2-member deployment I has 256 cores per member:
    # it fits 129 physical cores per member at LNC1 but cannot split 129 logical ranks at LNC2. The
    # busier 3-member deployments V and W (128 cores each) host 43 ranks (86 physical) per member.
    _LNC_RANKS = 129

    def _unsplittable_pair(self) -> list[HostRecord]:
        return [self._member(f"i{rank}", "I", rank, size=2, cores=256) for rank in range(2)]

    def _valid_triple(self, name: str, depth: int) -> list[HostRecord]:
        return [self._member(f"{name}{rank}", name.upper(), rank, size=3, depth=depth) for rank in range(3)]

    def test_retry_propagates_lnc_through_claim_and_rotation(self, temp_dir):
        """At LNC2 the idle deployment I is never picked, and the rotation's busy-set reset counts
        only V and W as eligible, so rotation alternates V, W, V, W. Dropping LNC from the claim
        picks I first; dropping it from the eligibility query keeps I in the eligible set, so the
        reset never fires and the fallback re-picks the less busy V."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager(temp_dir, {})
        OwnerHostStateStore(temp_dir)._update(
            lambda state: HostState(
                hosts=[*self._unsplittable_pair(), *self._valid_triple("v", 1), *self._valid_triple("w", 2)]
            )
        )
        store = manager.state_store
        chosen = []
        with (
            patch.object(store, "eligible_host_aliases", wraps=store.eligible_host_aliases) as eligible,
            closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=MagicMock(),
                    collective_ranks=self._LNC_RANKS,
                    lnc_config=2,
                    backoff_seconds=0,
                )
            ) as host_generator,
        ):
            for host_attempt in host_generator:
                with host_attempt as host:
                    chosen.append(host.get_host_id())
                    if len(chosen) <= 3:
                        raise QueuePatienceRotation("queue ETA exceeds patience")
        assert chosen == ["v0", "w0", "v0", "w0"]
        # The forwarded LNC is the caller's (one argument pin; the selection above is the behaviour).
        assert {call.kwargs["lnc_config"] for call in eligible.call_args_list} == {2}
        assert self._depths(manager)["i0"] == self._depths(manager)["i1"] == 0

    def test_retry_diagnostic_explains_unsplittable_group(self, temp_dir):
        manager = self._build_manager(temp_dir, {})
        OwnerHostStateStore(temp_dir)._update(lambda state: HostState(hosts=self._three_of_four()))
        with pytest.raises(FleetEmptyError, match="cannot evenly split 128 logical ranks"):
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=MagicMock(),
                    collective_ranks=128,
                    lnc_config=2,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt:
                        pass
        assert self._depths(manager) == {"t0": 0, "t1": 0, "t2": 0}

    def test_recoverable_wait_propagates_lnc(self, temp_dir):
        """The recoverable wait claims with the caller's LNC: the idle unsplittable I is never
        claimed, and the 3-member V added during the wait is."""
        manager = self._build_manager(temp_dir, {})
        manager.hosts_recoverable = True
        owner = OwnerHostStateStore(temp_dir)
        owner._update(lambda state: HostState(hosts=self._unsplittable_pair()))
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) == 1:
                owner._update(lambda s: replace(s, hosts=[*s.hosts, *self._valid_triple("v", 1)]))

        with patch.object(host_management.time, "sleep", side_effect=sleep):
            unit = manager._claim_unit(Platforms.TRN2, 2 * self._LNC_RANKS, lnc_config=2)
        assert unit.member_ids == ("v0", "v1", "v2")
        assert len(sleeps) == 1
        depths = self._depths(manager)
        assert (depths["i0"], depths["i1"]) == (0, 0)
        assert (depths["v0"], depths["v1"], depths["v2"]) == (259, 259, 259)

    def test_recoverable_timeout_explains_unsplittable_group(self, temp_dir):
        """A wait that can never succeed names the placement reason, not only a stalled re-resolver."""
        manager = self._build_manager(temp_dir, {})
        manager.hosts_recoverable = True
        manager._RECOVERABLE_CLAIM_DEADLINE_SECONDS = 0
        OwnerHostStateStore(temp_dir)._update(lambda state: HostState(hosts=self._unsplittable_pair()))
        with pytest.raises(TimeoutException, match="whose 2 hosts cannot evenly split 129 logical ranks"):
            manager._claim_unit(Platforms.TRN2, 2 * self._LNC_RANKS, lnc_config=2)
        assert self._depths(manager) == {"i0": 0, "i1": 0}

    @staticmethod
    def _record_patience_rotation_flags(manager, rotations: int) -> list[tuple[str, bool]]:
        """Run the retry loop, rotating on patience ``rotations`` times; return each attempt's
        (target id, rotate_on_patience)."""
        from test.utils.exceptions import QueuePatienceRotation

        real_get_target = manager._get_target
        flags: list[tuple[str, bool]] = []

        def get_target(unit, *, rotate_on_patience=True):
            flags.append((unit.host_id, rotate_on_patience))
            return real_get_target(unit, rotate_on_patience=rotate_on_patience)

        with patch.object(manager, "_get_target", side_effect=get_target):
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=MagicMock(),
                    collective_ranks=130,
                    lnc_config=1,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt:
                        if len(flags) <= rotations:
                            raise QueuePatienceRotation("queue ETA exceeds patience")
        return flags

    def test_sole_deployment_joins_without_patience_rotation(self, temp_dir):
        """With no other eligible unit, a group never rotates on patience: it would re-pick itself."""
        manager = self._two_member_deployment(temp_dir)
        assert self._record_patience_rotation_flags(manager, rotations=0) == [("m0", False)]

    def test_group_rotates_on_patience_only_while_another_unit_is_untried(self, temp_dir):
        """The first group rotates to the untried one; the last untried group joins. After a full
        sweep the busy set resets, so the next pass rotates once more before joining."""
        manager = self._build_manager(temp_dir, {})
        OwnerHostStateStore(temp_dir)._update(
            lambda state: HostState(
                hosts=[
                    self._member("a0", "A", 0, size=2),
                    self._member("a1", "A", 1, size=2),
                    self._member("b0", "B", 0, size=2, depth=1),
                    self._member("b1", "B", 1, size=2, depth=1),
                ]
            )
        )
        flags = self._record_patience_rotation_flags(manager, rotations=2)
        assert flags == [("a0", True), ("b0", False), ("a0", True)]

    def _two_member_deployment(self, temp_dir):
        manager = self._build_manager(temp_dir, {})
        OwnerHostStateStore(temp_dir)._update(
            lambda state: HostState(hosts=[self._member("m1", "L", 1, size=2), self._member("m0", "L", 0, size=2)])
        )
        return manager

    def test_get_target_rejects_non_ssh_member(self, temp_dir):
        manager = self._build_manager(temp_dir, {})
        manager.target_hosts["m1"] = MagicMock(spec=LocalHost)
        group_unit = HostGroup(members=(self._member("m0", "L", 0, size=2), self._member("m1", "L", 1, size=2)))
        with pytest.raises(TypeError, match="m1"):
            manager._get_target(group_unit)

    def test_retry_marks_tagged_member_unavailable_not_primary(self, temp_dir):
        """A failure tagged with a member id marks that member unavailable, leaves the primary
        available, and the error history names the member."""
        manager = self._two_member_deployment(temp_dir)
        error = OSError("connection reset")
        error.failed_host_id = "m1"  # ty: ignore[unresolved-attribute]
        with pytest.raises(InferenceException) as exc:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=MagicMock(),
                    collective_ranks=130,
                    lnc_config=1,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt:
                        raise error
        available = {rec.resolved.ssh_host: rec.available for rec in manager.state_store.read().hosts}
        assert available == {"m0": True, "m1": False}
        assert "m1: OSError: connection reset" in str(exc.value)

    def test_tagged_member_failure_clears_busy_primary(self, temp_dir):
        """After a busy group fails on a tagged member, its primary is no longer deprioritized."""
        from test.utils.exceptions import QueuePatienceRotation

        manager = self._build_manager(temp_dir, {})
        owner = OwnerHostStateStore(temp_dir)
        # L's lower depth makes it the first pick whenever it is not deprioritized.
        owner._update(
            lambda state: HostState(
                hosts=[
                    self._member("m0", "L", 0, size=2),
                    self._member("m1", "L", 1, size=2),
                    self._member("k0", "K", 0, size=2, depth=1),
                    self._member("k1", "K", 1, size=2, depth=1),
                ]
            )
        )

        def set_available(aliases: set[str], available: bool) -> None:
            owner._update(
                lambda state: replace(
                    state,
                    hosts=[
                        replace(rec, available=available) if rec.resolved.ssh_host in aliases else rec
                        for rec in state.hosts
                    ],
                )
            )

        real_claim = manager.state_store.claim_least_busy
        excludes: list[set[str]] = []

        def claim(platform, cores, exclude=None, **kwargs):
            excludes.append(set(exclude or ()))
            if len(excludes) == 2:
                # K drops out after L rotated, so L is the only eligible pick despite being busy.
                set_available({"k0"}, False)
            elif len(excludes) == 3:
                # Both hosts come back before the next pick.
                set_available({"k0", "m1"}, True)
            return real_claim(platform, cores, exclude=exclude, **kwargs)

        error = OSError("connection reset")
        error.failed_host_id = "m1"  # ty: ignore[unresolved-attribute]
        chosen = []
        with patch.object(manager.state_store, "claim_least_busy", side_effect=claim):
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=MagicMock(),
                    collective_ranks=130,
                    lnc_config=1,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        chosen.append(host.get_host_id())
                        if len(chosen) == 1:
                            raise QueuePatienceRotation("queue ETA exceeds patience")
                        if len(chosen) == 2:
                            raise error
        assert excludes == [set(), {"m0"}, set()]
        assert chosen == ["m0", "m0", "m0"]

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


class TestUniqueRemoteDirName:
    """The remote work directory is unique per client (16 random hex), bounded by NAME_MAX,
    and fails loudly for a test name that cannot take the suffix."""

    def test_16_hex_suffix(self):
        assert re.fullmatch(r"out-test_short_[0-9a-f]{16}", unique_remote_dir_name("out-test_short"))

    def test_two_calls_differ_even_in_the_same_process(self):
        # Two sandboxed clients share os.getpid(); the suffix must not depend on it.
        assert unique_remote_dir_name("out-test_short") != unique_remote_dir_name("out-test_short")

    def test_longest_possible_local_name_fits(self):
        # construct_test_output_directory_name caps the local name at 200 characters;
        # the L0-separation path appends "_l0_separated". Both must take the suffix.
        base = "x" * 200 + "_l0_separated"
        assert len(unique_remote_dir_name(base).encode()) <= 255

    def test_exactly_at_name_max(self):
        base = "x" * (255 - 1 - 16)
        assert len(unique_remote_dir_name(base).encode()) == 255

    def test_fails_when_suffix_does_not_fit(self):
        base = "x" * (255 - 16)  # one byte too many for "_" + 16 hex
        with pytest.raises(ValueError, match="Test name is too long") as e:
            unique_remote_dir_name(base)
        assert base in str(e.value) and "255" in str(e.value)

    def test_name_max_counts_bytes_not_characters(self):
        base = "\u00e9" * 120  # 120 characters but 240 bytes: no room for the suffix
        with pytest.raises(ValueError, match="Test name is too long"):
            unique_remote_dir_name(base)

    def test_prepare_host_uses_the_unique_name(self, tmp_path):
        host = _make_ssh_host(tmp_path)
        remote_dir = MagicMock()
        host._comm = MagicMock()
        host._comm.directory.return_value = remote_dir
        with patch.object(host, "__install_prerequisites__"):
            local = tmp_path / "out-test_some_case"
            local.mkdir()
            with host.prepare_host(str(local), MagicMock()) as remote_full_path:
                pass
        assert re.fullmatch(r"/tmp/neuronx-cc/tests/out-test_some_case_[0-9a-f]{16}", remote_full_path)
        assert "_pid" not in remote_full_path
        remote_dir.upload.assert_called_once()
        remote_dir.cleanup.assert_called_once()


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


# --output-format json carries instance_type once under instance_info, not on each device.
SAMPLE_NEURON_LS_INSTANCE_OUTPUT = json.dumps(
    {
        "instance_info": {
            "instance_type": "trn3-dev1.48xlarge",
            "instance_id": "i-05aa9dd8c8eba4ae4",
            "logical_neuroncore_config": 2,
            "ultraserver_info": {"ultraserver_id": "456", "ultraserver_size": 4, "ultraserver_instance_index": 2},
        },
        "devices": [
            {k: v for k, v in device.items() if k != "instance_type"} for device in json.loads(SAMPLE_NEURON_LS_OUTPUT)
        ],
    }
)


class TestLocalNeuronLsFlagFallback:
    """Most hosts run a neuron-ls that predates --output-format and exits non-zero on it, so the
    local invocation must retry with the legacy flag before giving up."""

    def setup_method(self):
        _run_neuron_ls.cache_clear()

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_uses_the_new_flag_when_supported(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=0, stdout=SAMPLE_NEURON_LS_INSTANCE_OUTPUT)
        devices = _run_neuron_ls("/opt/aws/neuron/bin")
        assert devices is not None and [d.neuron_device for d in devices] == [0]
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0][1:] == ["--output-format", "json"]

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_falls_back_to_the_legacy_flag(self, mock_run, _mock_isfile):
        mock_run.side_effect = [
            MagicMock(returncode=2, stdout=""),  # older neuron-ls rejects --output-format
            MagicMock(returncode=0, stdout=SAMPLE_NEURON_LS_OUTPUT),
        ]
        devices = _run_neuron_ls("/opt/aws/neuron/bin")
        assert devices is not None and [d.neuron_device for d in devices] == [0]
        assert [c.args[0][1:] for c in mock_run.call_args_list] == [["--output-format", "json"], ["--json-output"]]

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_returns_none_when_no_flag_is_supported(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=2, stdout="")
        assert _run_neuron_ls("/opt/aws/neuron/bin") is None
        assert mock_run.call_count == 2

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_detects_devices_from_the_instance_format(self, mock_run, _mock_isfile):
        mock_run.return_value = MagicMock(returncode=0, stdout=SAMPLE_NEURON_LS_INSTANCE_OUTPUT)
        assert detect_local_neuron_devices("/opt/aws/neuron/bin") is True


class TestDetectLocalPlatform:
    """The local platform comes from instance_type, which the two neuron-ls formats place differently."""

    def setup_method(self):
        _run_neuron_ls.cache_clear()

    @pytest.mark.parametrize(
        "stdout,expected",
        [
            (SAMPLE_NEURON_LS_OUTPUT, Platforms.TRN2),  # legacy: per-device instance_type
            (SAMPLE_NEURON_LS_INSTANCE_OUTPUT, Platforms.TRN3),  # structured: instance_info only
        ],
    )
    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_detects_platform_in_either_format(self, mock_run, _mock_isfile, stdout, expected):
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        assert detect_local_platform("/opt/aws/neuron/bin") == expected

    @patch("test.utils.host_management.os.path.isfile", return_value=True)
    @patch("test.utils.host_management.subprocess.run")
    def test_unknown_instance_type_yields_none(self, mock_run, _mock_isfile):
        stdout = json.dumps(
            {
                "instance_info": {"instance_type": "inf2.48xlarge"},
                "devices": json.loads(SAMPLE_NEURON_LS_INSTANCE_OUTPUT)["devices"],
            }
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        assert detect_local_platform("/opt/aws/neuron/bin") is None


class TestDetectLocalNeuronDevices:
    """Tests for detect_local_neuron_devices()."""

    def setup_method(self):
        # Clear lru_cache between tests so each test gets a fresh call
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


class TestTemporaryRandomSeed:
    """Tests for the temporary_random_seed context manager."""

    def setup_method(self) -> None:
        """Set a deterministic seed before each test."""
        random.seed(42)

    def teardown_method(self) -> None:
        """Reseed from system entropy so tests don't leak deterministic state."""
        random.seed()

    def test_restores_original_state(self) -> None:
        """RNG state before and after the context manager should be identical."""
        state_before = random.getstate()
        with temporary_random_seed(999):
            random.random()
        assert random.getstate() == state_before

    def test_choice_deterministic_with_same_seed(self) -> None:
        """Same temporary seed produces the same random.choice result."""
        items = ["host_a", "host_b", "host_c"]

        with temporary_random_seed(123):
            first = random.choice(items)

        with temporary_random_seed(123):
            second = random.choice(items)

        assert first == second

    def test_outer_sequence_unchanged_with_choice(self) -> None:
        """random.choice inside the context manager should not alter the outer RNG sequence."""
        expected_next = random.choice(["a", "b", "c"])

        random.seed(42)
        with temporary_random_seed(999):
            random.choice(["a", "b", "c"])
        assert random.choice(["a", "b", "c"]) == expected_next


def _make_mock_device(core_ids: list[int], lnc_config: int = 2) -> MagicMock:
    """Create a mock NeuronDeviceInfo with the given logical core IDs."""
    device = MagicMock()
    device.neuroncore_ids = core_ids
    device.nc_count = len(core_ids)
    device.logical_neuroncore_config = lnc_config
    return device


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
    unique_remote_dir_name,
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
    tmp_path,
    total_physical_cores: int = 8,
    patience_seconds: int = DEFAULT_PATIENCE_SECONDS,
    ssh_alias: str = "fakehost",
) -> SshHost:
    """Construct an SshHost without any real SSH; locking version/core count pre-seeded."""
    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("")
    host = SshHost(
        ssh_alias=ssh_alias,
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
        # Set by _ctor_returning, like CoreLockManager binds its collector.
        self.collector = None
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

    def acquire(self, num_logical_cores, lnc_config, ready=True, hold=False, hold_cap_seconds=None):  # noqa: ARG002
        self.acquire_calls += 1
        item = self._outcomes[0] if len(self._outcomes) == 1 else self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def has_enqueued(self) -> bool:
        return self.acquire_calls > 0

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


def _ctor_returning(fake) -> MagicMock:
    """CoreLockManager stand-in returning ``fake`` bound to the collector it is built with."""

    def _build(ssh_alias, total_physical_cores, collector, **kwargs):  # noqa: ARG001
        fake.collector = collector
        return fake

    return MagicMock(side_effect=_build)


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

    def test_transport_threads_into_ssh_host(self, tmp_path) -> None:
        # _get_host threads the manager's transport choice into each SshHost.
        kwargs = self._capture_ssh_host_kwargs(tmp_path, transport="ssh")
        assert kwargs["transport"] == "ssh"


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
            patch.object(host_management, "CoreLockManager", _ctor_returning(fake_mgr)),
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
            patch.object(host_management, "CoreLockManager", _ctor_returning(fake_mgr)),
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

    def acquire(self, num_logical_cores, lnc_config, ready=True, hold=False, hold_cap_seconds=None):  # noqa: ARG002
        self.acquire_ready_args.append(ready)
        if self._seq:
            item = self._seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=10)

    @property
    def has_enqueued(self) -> bool:
        return bool(self.acquire_ready_args)

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
        ctor = _ctor_returning(rec)
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
            soft_join_mgr = host._core_lock_manager
            assert soft_join_mgr is not None, "soft join caches a core lock manager"
            entry_after_soft_join = soft_join_mgr.entry_id
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
            patch.object(host_management, "CoreLockManager", _ctor_returning(rec)),
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
            patch.object(host_management, "CoreLockManager", _ctor_returning(fake_mgr)),
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
        # The retry loop's end-of-attempt abandon keeps it too and sends no dequeue.
        host.abandon_queue()
        assert host._core_lock_manager is fake_mgr
        assert fake_mgr.dequeue_calls == 0
        assert fake_mgr.metrics_calls == 0

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
        )
        assert local.soft_join_queue(collector=MagicMock(), collective_ranks=2, lnc_config=2) is None

    def test_base_and_local_abandon_queue_is_noop(self, tmp_path) -> None:
        assert LocalHost.abandon_queue is host_management.Host.abandon_queue
        assert LocalHost("/opt/aws/neuron/bin", "localhost", str(tmp_path)).abandon_queue() is None

    def test_abandon_queue_swallows_metric_flush_error_and_drops_manager(self, tmp_path, caplog) -> None:
        host = _make_ssh_host(tmp_path)
        assert host.abandon_queue() is None
        manager = MagicMock()
        manager.record_contention_metrics.side_effect = RuntimeError("flush broke")
        host._core_lock_manager = manager
        with caplog.at_level("WARNING"):
            host.abandon_queue()
        assert any("flush broke" in r.getMessage() for r in caplog.records)
        assert host._core_lock_manager is None

    def test_lock_signal_counts_read_the_cached_manager(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path)
        assert host._lock_signal_counts() == (0, 0)
        host._core_lock_manager = MagicMock(bump_count=2, reenqueue_count=1)
        assert host._lock_signal_counts() == (2, 1)


_UNKNOWN_KWARG_ERROR = "poll() got an unexpected keyword argument 'hold_cap_seconds'"


class _FailingFirstExecutor(_HelperExecutor):
    """A _HelperExecutor whose first ``failures`` calls raise like a helper that rejects an argument."""

    def __init__(self, *args, failures: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._failures = failures

    def call_function(self, func, **kwargs):
        if self._failures > 0:
            self._failures -= 1
            raise RemoteExecutorError(f"TypeError: {_UNKNOWN_KWARG_ERROR}")
        return super().call_function(func, **kwargs)


class TestSshHostHeldPoll:
    """Held ``_poll_queue`` against the real helper: hold cap, assembly signal, capability gate."""

    _paths = TestSshHostPollLoopIntegration._paths
    _write_locks = TestSshHostPollLoopIntegration._write_locks
    _read_queue = TestSshHostPollLoopIntegration._read_queue

    def _host(self, tmp_path, executor_cls=_HelperExecutor, patience_seconds=DEFAULT_PATIENCE_SECONDS, **kwargs):
        helpers_file, lock_file, locks_json = self._paths(tmp_path)
        executor = executor_cls(helpers_file, lock_file, locks_json, **kwargs)
        host = _make_ssh_host(tmp_path, patience_seconds=patience_seconds)
        return host, executor, locks_json

    @contextlib.contextmanager
    def _wired(self, host, executor, clock):
        with (
            patch.object(host, "_get_remote_executor", return_value=executor),
            patch.object(host, "get_instance_type", return_value="trn1"),
            patch("time.sleep", clock.sleep),
            patch("time.time", clock.time),
        ):
            yield

    def test_held_poll_sends_patience_as_hold_cap(self, tmp_path) -> None:
        host, executor, locks_json = self._host(tmp_path, patience_seconds=321)
        self._write_locks(locks_json)
        with self._wired(host, executor, _FakeClock(start=_REALISTIC_EPOCH)):
            host._poll_queue(_cm_collector(), 2, 2, ready=False, hold=True)
        (entry,) = self._read_queue(locks_json)
        assert entry["hold"] is True
        assert entry["hold_cap"] == 321

    @pytest.mark.parametrize("patience_seconds", [0, -1])
    def test_non_positive_patience_sends_default_hold_cap(self, tmp_path, patience_seconds) -> None:
        """Patience <= 0 disables rotation; it is not a usable cap, so the default patience is sent."""
        host, executor, locks_json = self._host(tmp_path, patience_seconds=patience_seconds)
        self._write_locks(locks_json)
        with self._wired(host, executor, _FakeClock(start=_REALISTIC_EPOCH)):
            host._poll_queue(_cm_collector(), 2, 2, ready=False, hold=True)
        (entry,) = self._read_queue(locks_json)
        assert entry["hold"] is True
        assert entry["hold_cap"] == DEFAULT_PATIENCE_SECONDS

    def test_non_held_poll_sends_no_hold_cap(self, tmp_path) -> None:
        host, executor, locks_json = self._host(tmp_path, patience_seconds=321)
        self._write_locks(locks_json)
        with self._wired(host, executor, _FakeClock(start=_REALISTIC_EPOCH)):
            host._poll_queue(_cm_collector(), 2, 2, ready=False)
        (entry,) = self._read_queue(locks_json)
        assert "hold_cap" not in entry

    def test_assembled_once_held_region_is_free(self, tmp_path) -> None:
        host, executor, locks_json = self._host(tmp_path)
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100)
        collector = _cm_collector()
        with self._wired(host, executor, clock):
            first = host._poll_queue(collector, 2, 2, ready=False, hold=True)
            clock.sleep(5)
            busy = host._poll_queue(collector, 2, 2, ready=False, hold=True)
            with open(locks_json) as f:
                state = json.load(f)
            state["physical_neuron_cores_lock_timeout"] = {}
            with open(locks_json, "w") as f:
                json.dump(state, f)
            clock.sleep(5)
            free = host._poll_queue(collector, 2, 2, ready=False, hold=True)
        assert (first.assembled, busy.assembled, free.assembled) == (False, False, True)
        assert free.status == AllocationStatus.QUEUED

    def test_unknown_argument_redeploys_once_and_retries(self, tmp_path, caplog) -> None:
        host, executor, locks_json = self._host(tmp_path, executor_cls=_FailingFirstExecutor, failures=1)
        self._write_locks(locks_json)
        init = MagicMock(return_value=3)
        with (
            self._wired(host, executor, _FakeClock(start=_REALISTIC_EPOCH)),
            patch.object(host_management.lock_client, "initialize_and_deploy", init),
            caplog.at_level("WARNING"),
        ):
            outcome = host._poll_queue(_cm_collector(), 2, 2, ready=False, hold=True)
        init.assert_called_once_with(executor)
        assert outcome.status == AllocationStatus.QUEUED
        (entry,) = self._read_queue(locks_json)
        assert entry["hold"] is True
        assert any("fakehost" in r.getMessage() and "redeploy" in r.getMessage() for r in caplog.records)

    def test_second_unknown_argument_failure_propagates(self, tmp_path) -> None:
        host, executor, locks_json = self._host(tmp_path, executor_cls=_FailingFirstExecutor, failures=2)
        self._write_locks(locks_json)
        init = MagicMock(return_value=3)
        with (
            self._wired(host, executor, _FakeClock(start=_REALISTIC_EPOCH)),
            patch.object(host_management.lock_client, "initialize_and_deploy", init),
            pytest.raises(LockAcquisitionError, match="unexpected keyword argument"),
        ):
            host._poll_queue(_cm_collector(), 2, 2, ready=False, hold=True)
        init.assert_called_once_with(executor)
        assert self._read_queue(locks_json) == []

    def test_non_held_unknown_argument_does_not_redeploy(self, tmp_path) -> None:
        host, executor, locks_json = self._host(tmp_path, executor_cls=_FailingFirstExecutor, failures=1)
        self._write_locks(locks_json)
        init = MagicMock(return_value=3)
        with (
            self._wired(host, executor, _FakeClock(start=_REALISTIC_EPOCH)),
            patch.object(host_management.lock_client, "initialize_and_deploy", init),
            pytest.raises(LockAcquisitionError, match="unexpected keyword argument"),
        ):
            host._poll_queue(_cm_collector(), 2, 2, ready=False)
        init.assert_not_called()

    def test_other_lock_errors_do_not_redeploy(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path)
        rec = _RecordingManager()
        rec._seq = [LockAcquisitionError("flock timed out after 30s")]
        init = MagicMock(return_value=3)
        with (
            patch.object(host, "_attempt_manager", return_value=rec),
            patch.object(host_management.lock_client, "initialize_and_deploy", init),
            pytest.raises(LockAcquisitionError, match="flock timed out"),
        ):
            host._poll_queue(_cm_collector(), 2, 2, ready=False, hold=True)
        init.assert_not_called()


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
            assert out_a2.logical_cores is not None, "an ALLOCATED outcome carries its logical cores"
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
                mgr1 = host._core_lock_manager
                assert mgr1 is not None, "an active allocation caches its core lock manager"
                entry1 = mgr1.entry_id
                assert mgr1.collector is col1
                assert len(a1.logical_core_ids) == 4
            # Teardown drops the cache so the next attempt cannot reuse it.
            assert host._core_lock_manager is None

            # Attempt #2: freed cores -> commits THROUGH the queue on a later poll.
            with host.get_core_allocation(collector=col2, collective_ranks=4, lnc_config=2, timeout_seconds=300) as a2:
                mgr2 = host._core_lock_manager
                assert mgr2 is not None, "an active allocation caches its core lock manager"
                entry2 = mgr2.entry_id
                assert mgr2.collector is col2
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
                assert mgr1 is not None, "an active allocation caches its core lock manager"
                entry1 = mgr1.entry_id
                assert mgr1.collector is col1
                assert len(a1.logical_core_ids) == 4
            with host.get_core_allocation(collector=col2, collective_ranks=4, lnc_config=2) as a2:
                mgr2 = host._core_lock_manager
                assert mgr2 is not None, "an active allocation caches its core lock manager"
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

    def test_recoverable_request_over_ceiling_fails_fast_without_waiting(self, temp_dir):
        # The bug this fixes: on a recoverable pool whose largest host can never fit the request
        # (8-core hosts, 32-core request), the claim must fail fast with RequestTooLargeError
        # instead of polling until the 1800s deadline and then misreporting a dead re-resolver.
        # No sleep should happen — waiting is futile.
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        # The 8-core host IS present and available, just too small. Shrink it from the default 64
        # to match the fleet's 8-core ceiling.
        manager.state_store.set_physical_cores({"h1": 8})
        _set_ceiling(temp_dir, {Platforms.TRN2: 8})
        with patch("test.utils.host_management.time.sleep") as sleep:
            with pytest.raises(RequestTooLargeError, match="largest host the fleet can provision"):
                self._assign(manager, num_of_physical_cores_needed=32)
        sleep.assert_not_called()  # impossible request -> no wait at all

    def test_non_recoverable_request_over_ceiling_fails_fast(self, temp_dir):
        # Same guard on the non-recoverable path: a too-large request is a RequestTooLargeError
        # (a FleetEmptyError subtype) naming the sizing cause, not a generic "no available hosts".
        manager = self._make_manager(temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)])
        manager.state_store.set_physical_cores({"h1": 8})  # present but too small
        _set_ceiling(temp_dir, {Platforms.TRN2: 8})
        with pytest.raises(RequestTooLargeError, match="Request needs 32 physical cores"):
            self._assign(manager, num_of_physical_cores_needed=32)

    def test_recoverable_request_within_ceiling_still_waits(self, temp_dir):
        # Regression guard: the ceiling check must not disturb the normal recoverable wait. A
        # request that FITS the ceiling (8 <= 8) but has no host up yet still waits and succeeds
        # when the background re-resolver adds a host.
        manager = self._make_manager(temp_dir, [], needs_local_host=False, hosts_recoverable=True)
        _set_ceiling(temp_dir, {Platforms.TRN2: 8})

        def add_host_during_wait(_seconds):
            _add_host(manager.state_store, "newhost", Platforms.TRN2, num_physical_cores=8)

        with patch("test.utils.host_management.time.sleep", side_effect=add_host_during_wait) as sleep:
            host = self._assign(manager, num_of_physical_cores_needed=8)
        assert host.get_host_id() == "newhost"
        sleep.assert_called_once()

    def test_over_ceiling_reported_with_history_in_retry_loop(self, temp_dir):
        # Through the full retry entrypoint (get_host_assignment_with_retry), a too-large request
        # surfaces as the RequestTooLargeError (caught by the loop's `except FleetEmptyError` and
        # re-raised) — cleanly terminal, not retried across hosts until the deadline.
        manager = self._make_manager(
            temp_dir, [TargetHost(ssh_host="h1", host_type=Platforms.TRN2)], hosts_recoverable=True
        )
        manager.state_store.set_physical_cores({"h1": 8})  # present but too small for 32 cores
        _set_ceiling(temp_dir, {Platforms.TRN2: 8})
        with patch("test.utils.host_management.time.sleep"):
            with pytest.raises(RequestTooLargeError):
                # 16 ranks x lnc2 = 32 cores, more than any 8-core host can serve.
                for _host_ctx in manager.get_host_assignment_with_retry(
                    Platforms.TRN2, NoopMetricsCollector(), collective_ranks=16, lnc_config=2
                ):
                    pass

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
    primitive. Defaults to 64 cores so the host is eligible for typical (small) requests.

    Preserves the existing fleet metadata (poisoned / under_capacity / provisionable_ceiling)
    so injecting a host mid-wait doesn't silently clear a verdict a test set up."""
    store._update(
        lambda state: HostState(
            hosts=[
                *state.hosts,
                HostRecord(
                    resolved=ResolvedHost(ssh_host=alias, host_type=host_type, num_physical_cores=num_physical_cores)
                ),
            ],
            poisoned=state.poisoned,
            under_capacity=state.under_capacity,
            provisionable_ceiling=state.provisionable_ceiling,
        )
    )


class TestNeuronEnvPassthrough:
    """Environment variables registered for passthrough reach the remote workload."""

    @pytest.fixture
    def local_host(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalHost("/opt/aws/neuron/bin", "localhost", tmpdir)

    @staticmethod
    def _build_env(local_host):
        return local_host._build_neuron_env(
            core_allocation=CoreAllocation(host_id="localhost", logical_core_ids=[0], lnc_config=2),
            lnc_config=2,
            unique_port=61234,
            debug_output_dir="/tmp/debug",
            reset_cores=False,
        )

    def test_registered_variable_is_forwarded_when_set(self, local_host):
        with patch.dict(host_management.os.environ, {"NEURON_RT_TEST_PASSTHROUGH": "/some/path"}):
            with patch.object(host_management, "ENV_PASSTHROUGH", {"NEURON_RT_TEST_PASSTHROUGH"}):
                env = self._build_env(local_host)
        assert env["NEURON_RT_TEST_PASSTHROUGH"] == "/some/path"

    def test_registered_variable_is_omitted_when_unset(self, local_host):
        environ = {k: v for k, v in os.environ.items() if k != "NEURON_RT_TEST_PASSTHROUGH"}
        with patch.dict(host_management.os.environ, environ, clear=True):
            with patch.object(host_management, "ENV_PASSTHROUGH", {"NEURON_RT_TEST_PASSTHROUGH"}):
                env = self._build_env(local_host)
        assert "NEURON_RT_TEST_PASSTHROUGH" not in env

    def test_unregistered_variable_is_not_forwarded(self, local_host):
        with patch.dict(host_management.os.environ, {"NEURON_RT_TEST_PASSTHROUGH": "/some/path"}):
            with patch.object(host_management, "ENV_PASSTHROUGH", set()):
                env = self._build_env(local_host)
        assert "NEURON_RT_TEST_PASSTHROUGH" not in env
        assert env["NEURON_RT_VISIBLE_CORES"] == "0"


# =============================================================================
# SshHostGroup: a Host over a multi-node deployment's SshHost members
# =============================================================================

import threading  # noqa: E402


def _fake_member(alias: str, events: list, cores: int = 64, devices: list | None = None) -> MagicMock:
    """Autospec SshHost member that records the group-facing calls in the shared ``events`` list."""
    member = create_autospec(SshHost, instance=True)
    member.get_host_id.return_value = alias
    member.get_total_physical_cores.return_value = cores
    member.get_neuron_device_info.return_value = list(devices) if devices is not None else [f"{alias}-dev"]
    member._check_patience.side_effect = lambda *a, **k: events.append(("check", alias, a, k))
    member._join_queue.side_effect = lambda *a, **k: events.append(("join", alias, a, k))
    member._lock_signal_counts.return_value = (0, 0)
    member.patience_seconds = DEFAULT_PATIENCE_SECONDS
    return member


def _recording_prepare(alias: str, events: list, barrier: threading.Barrier | None = None, error=None):
    """Return a ``prepare_host`` side effect whose context manager records enter/exit and can fail on enter."""

    @contextlib.contextmanager
    def _prepare(*args, **kwargs):  # noqa: ARG001
        if barrier is not None:
            barrier.wait()
        if error is not None:
            events.append(("enter-failed", alias))
            raise error
        events.append(("enter", alias))
        try:
            yield f"/remote/{alias}"
        except BaseException as e:
            events.append(("exit-error", alias, e))
            raise
        finally:
            events.append(("exit", alias))

    return _prepare


class TestSshHostGroup:
    """SshHostGroup identity, two-phase soft-join, and concurrent prepare."""

    @staticmethod
    def _group(*members):
        return host_management.SshHostGroup(tuple(members))

    def test_rejects_empty_members(self) -> None:
        with pytest.raises(ValueError):
            self._group()

    def test_rejects_duplicate_members(self) -> None:
        events: list = []
        with pytest.raises(ValueError):
            self._group(_fake_member("a", events), _fake_member("a", events))

    def test_identity_cores_and_devices(self) -> None:
        events: list = []
        a = _fake_member("a", events, cores=64, devices=["a0", "a1"])
        b = _fake_member("b", events, cores=32, devices=["b0"])
        group = self._group(a, b)
        assert group.hosts == (a, b)
        assert group.get_host_id() == "a"
        assert group.get_total_physical_cores() == 96
        assert group.get_neuron_device_info() == ["a0", "a1", "b0"]

    def test_base_host_spans_itself(self) -> None:
        local = LocalHost.__new__(LocalHost)
        assert local.hosts == (local,)

    def test_soft_join_passes_ranks_per_member_and_hold(self) -> None:
        events: list = []
        collector = _cm_collector()
        group = self._group(_fake_member("a", events), _fake_member("b", events))
        group.soft_join_queue(collector, collective_ranks=8, lnc_config=1)
        checks = [e for e in events if e[0] == "check"]
        joins = [e for e in events if e[0] == "join"]
        assert all(args[1:] == (4, 1) for _, _, args, _ in checks)
        assert all(args[1:] == (4, 1) for _, _, args, _ in joins)
        # Every member joins with the same sampled hold cap, within [0.9, 1.1] x patience.
        caps = {kwargs.pop("hold_cap_seconds") for _, _, _, kwargs in joins}
        assert len(caps) == 1 and 0.9 * DEFAULT_PATIENCE_SECONDS <= caps.pop() <= 1.1 * DEFAULT_PATIENCE_SECONDS
        assert all(kwargs == {"hold": True} for _, _, _, kwargs in joins)

    def test_soft_join_without_patience_rotation_joins_every_member_unprobed(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a._check_patience.side_effect = QueuePatienceRotation("over patience")
        group = host_management.SshHostGroup((a, b), rotate_on_patience=False)
        group.soft_join_queue(_cm_collector(), collective_ranks=8, lnc_config=1)
        a._check_patience.assert_not_called()
        b._check_patience.assert_not_called()
        joins = [e for e in events if e[0] == "join"]
        assert [alias for _, alias, _, _ in joins] == ["a", "b"]
        assert all(args[1:] == (4, 1) and kwargs["hold"] is True for _, _, args, kwargs in joins)

    def test_soft_join_patience_error_is_logged_and_that_member_is_not_joined(self, caplog) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a._check_patience.side_effect = OSError("probe down")
        with caplog.at_level("WARNING"):
            self._group(a, b).soft_join_queue(_cm_collector(), collective_ranks=2, lnc_config=2)
        assert "[a] Best-effort patience check failed: probe down" in caplog.text
        assert [alias for kind, alias, *_ in events if kind == "join"] == ["b"]
        a._join_queue.assert_not_called()

    def test_soft_join_member_join_error_is_logged_and_others_join(self, caplog) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a._join_queue.side_effect = OSError("ssh down")
        with caplog.at_level("WARNING"):
            self._group(a, b).soft_join_queue(_cm_collector(), collective_ranks=2, lnc_config=2)
        assert "[a] Best-effort soft-join failed: ssh down" in caplog.text
        assert [alias for kind, alias, *_ in events if kind == "join"] == ["b"]
        a._join_queue.assert_called_once()

    def test_prepare_host_yields_paths_in_rank_order_and_exits_every_member(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a.prepare_host.side_effect = _recording_prepare("a", events)
        b.prepare_host.side_effect = _recording_prepare("b", events)
        with self._group(a, b).prepare_host("/local/dir", _cm_collector()) as paths:
            assert paths == ("/remote/a", "/remote/b")
            assert {e for e in events if e[0] == "exit"} == set()
        assert sorted(e for e in events if e[0] == "exit") == [("exit", "a"), ("exit", "b")]

    def test_prepare_host_enters_members_concurrently(self) -> None:
        events: list = []
        barrier = threading.Barrier(3, timeout=5)
        members = [_fake_member(alias, events) for alias in ("a", "b", "c")]
        for member in members:
            member.prepare_host.side_effect = _recording_prepare(member.get_host_id(), events, barrier=barrier)
        # Sequential entry would block on the barrier and time out with BrokenBarrierError.
        with self._group(*members).prepare_host("/local/dir", _cm_collector()) as paths:
            assert paths == ("/remote/a", "/remote/b", "/remote/c")

    def test_prepare_host_body_error_reaches_every_member_exit_and_propagates(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a.prepare_host.side_effect = _recording_prepare("a", events)
        b.prepare_host.side_effect = _recording_prepare("b", events)
        error = RuntimeError("body failed")
        with pytest.raises(RuntimeError) as excinfo:
            with self._group(a, b).prepare_host("/local/dir", _cm_collector()):
                raise error
        assert excinfo.value is error
        assert sorted((e[1], e[2]) for e in events if e[0] == "exit-error") == [("a", error), ("b", error)]
        assert sorted(e for e in events if e[0] == "exit") == [("exit", "a"), ("exit", "b")]

    def test_prepare_host_forwards_arguments(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        a.prepare_host.side_effect = _recording_prepare("a", events)
        collector = _cm_collector()
        with self._group(a).prepare_host("/local/dir", collector, skip_remote_cleanup=True, force_local_cleanup=False):
            pass
        a.prepare_host.assert_called_once_with("/local/dir", collector, True, False)

    def test_prepare_host_partial_failure_exits_entered_and_reraises_original(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        error = OSError("upload failed")
        a.prepare_host.side_effect = _recording_prepare("a", events)
        b.prepare_host.side_effect = _recording_prepare("b", events, error=error)
        with pytest.raises(OSError) as excinfo:
            with self._group(a, b).prepare_host("/local/dir", _cm_collector()):
                raise pytest.fail.Exception("body must not run when a member fails to prepare")
        assert excinfo.value is error
        assert excinfo.value.failed_host_id == "b"
        assert ("exit", "a") in events

    def test_prepare_host_reraises_first_failure_in_rank_order(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        first, second = OSError("a failed"), OSError("b failed")
        b_failed = threading.Event()

        @contextlib.contextmanager
        def a_fails_after_b(*args, **kwargs):  # noqa: ARG001
            # Completion order (b, then a) differs from rank order (a, then b).
            assert b_failed.wait(timeout=5)
            events.append(("enter-failed", "a"))
            raise first
            yield  # pragma: no cover

        @contextlib.contextmanager
        def b_fails(*args, **kwargs):  # noqa: ARG001
            events.append(("enter-failed", "b"))
            try:
                raise second
            finally:
                b_failed.set()
            yield  # pragma: no cover

        a.prepare_host.side_effect = a_fails_after_b
        b.prepare_host.side_effect = b_fails
        with pytest.raises(OSError) as excinfo:
            with self._group(a, b).prepare_host("/local/dir", _cm_collector()):
                pass
        assert [e[1] for e in events if e[0] == "enter-failed"] == ["b", "a"]
        assert excinfo.value is first
        assert excinfo.value.failed_host_id == "a"  # ty: ignore[unresolved-attribute]

    def test_prepare_host_logs_every_member_error_and_notes_secondary_ones(self, caplog) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        c = _fake_member("c", events)
        first, second = OSError("a failed"), OSError("c failed")
        a.prepare_host.side_effect = _recording_prepare("a", events, error=first)
        b.prepare_host.side_effect = _recording_prepare("b", events)
        c.prepare_host.side_effect = _recording_prepare("c", events, error=second)
        with caplog.at_level("WARNING"):
            with pytest.raises(OSError) as excinfo:
                with self._group(a, b, c).prepare_host("/local/dir", _cm_collector()):
                    pass
        assert excinfo.value is first
        assert any("[c]" in note and "c failed" in note for note in excinfo.value.__notes__)
        assert "[a]" in caplog.text and "a failed" in caplog.text
        assert "[c]" in caplog.text and "c failed" in caplog.text
        assert ("exit", "b") in events

    def test_prepare_host_interrupted_wait_still_exits_every_entered_member(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a.prepare_host.side_effect = _recording_prepare("a", events)
        b.prepare_host.side_effect = _recording_prepare("b", events)

        def _interrupted_wait(*args, **kwargs):  # noqa: ARG001
            raise KeyboardInterrupt

        with patch.object(host_management, "wait", side_effect=_interrupted_wait):
            with pytest.raises(KeyboardInterrupt):
                with self._group(a, b).prepare_host("/local/dir", _cm_collector()):
                    raise pytest.fail.Exception("body must not run after an interrupt")
        assert sorted(e for e in events if e[0] == "enter") == [("enter", "a"), ("enter", "b")]
        assert sorted(e for e in events if e[0] == "exit") == [("exit", "a"), ("exit", "b")]

    def test_execution_not_supported(self) -> None:
        events: list = []
        group = self._group(_fake_member("a", events), _fake_member("b", events))
        with pytest.raises(NotImplementedError):
            group.execute_command("cmd", "/local/dir", _cm_collector(), collective_ranks=2, lnc_config=2)
        with pytest.raises(NotImplementedError):
            group._get_debug_output_dir("/local/dir")

    def test_tag_failed_host_keeps_existing_attribution(self) -> None:
        error = OSError("boom")
        host_management.tag_failed_host(error, "inner")
        host_management.tag_failed_host(error, "outer")
        assert error.failed_host_id == "inner"  # ty: ignore[unresolved-attribute]

    def test_attempt_manager_rebuilds_when_cached_manager_has_another_collector(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path)
        stale, fresh = MagicMock(), MagicMock()
        stale.collector = _cm_collector()
        host._core_lock_manager = stale
        collector = _cm_collector()
        with patch.object(host, "_ensure_core_lock_manager", return_value=fresh) as ensure:
            assert host._attempt_manager(collector) is fresh
        ensure.assert_called_once_with(collector)

    def test_attempt_manager_reuses_cached_manager_for_same_collector(self, tmp_path) -> None:
        host = _make_ssh_host(tmp_path)
        collector = _cm_collector()
        cached = MagicMock()
        cached.collector = collector
        host._core_lock_manager = cached
        with patch.object(host, "_ensure_core_lock_manager") as ensure:
            assert host._attempt_manager(collector) is cached
        ensure.assert_not_called()


# =============================================================================
# SshHostGroup coordinated core allocation
# =============================================================================

from types import SimpleNamespace  # noqa: E402

from test.utils.core_lock_client import COMMIT_WINDOW  # noqa: E402
from test.utils.core_lock_manager import GroupCoreAllocation, LockVersionError  # noqa: E402
from test.utils.host_management import MAX_RETRYABLE_LOCK_ERRORS  # noqa: E402


def _queued(eta: int, position: int | None = 0) -> AllocationOutcome:
    return AllocationOutcome(status=AllocationStatus.QUEUED, position=position, worst_case_eta=eta)


# Queued with a region set aside and none of its cores locked: ready to commit.
_ASSEMBLED = AllocationOutcome(status=AllocationStatus.QUEUED, position=0, worst_case_eta=0, assembled=True)
_DRAINING = AllocationOutcome(status=AllocationStatus.DRAINING, position=0, worst_case_eta=0)


def _allocated(physical: list[int], should_reset: bool | None = False) -> AllocationOutcome:
    return AllocationOutcome(
        status=AllocationStatus.ALLOCATED,
        logical_cores=[core // 2 for core in physical[::2]],
        physical_cores=physical,
        should_reset_cores=should_reset,
    )


def _next_step(script: list):
    """Consume the next scripted step; the last step repeats forever. Callables are evaluated per poll."""
    step = script.pop(0) if len(script) > 1 else script[0]
    if callable(step):
        step = step()
    if isinstance(step, BaseException):
        raise step
    return step


def _alloc_member(alias: str, events: list, wait: list, commit: list) -> MagicMock:
    """Member whose ``ready=False``/``ready=True`` polls follow the ``wait``/``commit`` scripts."""
    member = _fake_member(alias, events)
    member.get_instance_type.return_value = "trn2.48xlarge"

    def _poll(collector, ranks, lnc_config, *, ready, hold=False, hold_cap_seconds=None):  # noqa: ARG001
        events.append(("poll", alias, ranks, lnc_config, ready, hold))
        return _next_step(commit if ready else wait)

    member._poll_queue.side_effect = _poll
    member._release_cores.side_effect = lambda collector, cores, **kwargs: events.append(("release", alias, cores))
    member._abandon_queue.side_effect = lambda: events.append(("abandon", alias))
    member._record_contention_metrics.side_effect = lambda: events.append(("contention", alias))
    return member


class _SleepAdvancedClock:
    """Deterministic clock whose ``sleep`` advances ``time`` and records the sleep in ``events``."""

    def __init__(self, events: list):
        self.now = 1_000.0
        self._events = events

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self._events.append(("sleep", seconds))
        self.now += seconds


@pytest.fixture
def group_timed_clock():
    """Patch the clock and jitter used by host_management; yields the shared event list and the clock."""
    events: list = []
    clock = _SleepAdvancedClock(events)
    with (
        patch.object(host_management, "time", SimpleNamespace(time=clock.time, sleep=clock.sleep)),
        patch.object(host_management, "random", SimpleNamespace(uniform=lambda low, high: 0.0)),  # noqa: ARG005
    ):
        yield events, clock


@pytest.fixture
def group_clock(group_timed_clock):
    """``group_timed_clock`` without the clock: yields the shared event list."""
    return group_timed_clock[0]


def _polls(events: list, *, ready: bool | None = None) -> list[tuple]:
    return [e for e in events if e[0] == "poll" and (ready is None or e[4] is ready)]


def _kinds(events: list, kind: str) -> list[tuple]:
    return [e for e in events if e[0] == kind]


def _non_retryable_lock_error() -> LockAcquisitionError:
    error = LockAcquisitionError("corrupt lock state")
    error.retryable = False
    return error


class TestSshHostGroupCoreAllocation:
    """Coordinated wait/commit core allocation across SshHostGroup members."""

    @staticmethod
    def _group(*members):
        return host_management.SshHostGroup(tuple(members))

    def test_waits_with_hold_and_commits_only_when_all_assembled_in_same_cycle(self, group_clock) -> None:
        events = group_clock
        # a is assembled in cycles 1 and 3, b in cycles 2 and 3: only cycle 3 has both.
        a = _alloc_member("a", events, wait=[_ASSEMBLED, _queued(5), _ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(30), _ASSEMBLED, _ASSEMBLED], commit=[_allocated([4, 5])])
        with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=8, lnc_config=2):
            pass
        waits = _polls(events, ready=False)
        assert all(e[2:] == (4, 2, False, True) for e in waits)
        # Members are polled concurrently, so each cycle's polls are compared as a set.
        assert [sorted(e[1] for e in waits[i : i + 2]) for i in range(0, len(waits), 2)] == [["a", "b"]] * 3
        first_commit = events.index(_polls(events, ready=True)[0])
        assert events.index(waits[-1]) < first_commit
        assert len(_kinds(events[:first_commit], "sleep")) == 2
        assert [(e[1], e[5]) for e in _polls(events, ready=True)] == [("a", True), ("b", True)]

    def test_queued_with_zero_eta_but_not_assembled_is_not_ready(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(0), _queued(0), _ASSEMBLED], commit=[_allocated([2, 3])])
        with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
            pass
        first_commit = events.index(_polls(events, ready=True)[0])
        assert len(_polls(events[:first_commit], ready=False)) == 6

    def test_success_yields_group_allocation_and_releases_each_member_after_body(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1], should_reset=False)])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([6, 7], should_reset=None)])
        collector = _cm_collector()
        with self._group(a, b).get_core_allocation(collector, collective_ranks=2, lnc_config=2) as allocation:
            assert isinstance(allocation, GroupCoreAllocation)
            assert isinstance(allocation, CoreAllocation)
            assert allocation.host_id == "a"
            assert allocation.logical_core_ids == [0]
            assert allocation.lnc_config == 2
            assert allocation.should_reset_cores is True
            assert [m.logical_core_ids for m in allocation.member_allocations] == [[0], [3]]
            assert [m.host_id for m in allocation.member_allocations] == ["a", "b"]
            assert _kinds(events, "release") == []
            assert _kinds(events, "contention") == [("contention", "a"), ("contention", "b")]
        assert _kinds(events, "release") == [("release", "a", [0, 1]), ("release", "b", [6, 7])]
        assert _kinds(events, "abandon") == []
        a._ensure_core_lock_manager.assert_called_once_with(collector)
        b._ensure_core_lock_manager.assert_called_once()
        collector.timer.assert_called_once_with(MetricName.CORE_ALLOCATION_TIME)
        collector.add_dimension.assert_called_once_with({MetricName.INSTANCE_TYPE: "trn2.48xlarge"})

    def test_releases_every_member_when_body_raises(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        with pytest.raises(RuntimeError, match="body failed"):
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise RuntimeError("body failed")
        assert _kinds(events, "release") == [("release", "a", [0, 1]), ("release", "b", [2, 3])]

    def test_draining_member_is_never_ready(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_DRAINING, _DRAINING, _ASSEMBLED], commit=[_allocated([2, 3])])
        with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
            pass
        first_commit = events.index(_polls(events, ready=True)[0])
        assert len(_polls(events[:first_commit], ready=False)) == 6

    def test_missed_commit_releases_committed_and_returns_to_wait(self, group_clock, caplog) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        # b's region is reclaimed between polls: its first commit poll finds it queued again.
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_ASSEMBLED, _allocated([2, 3])])
        with caplog.at_level("WARNING"):
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                releases_before_success = list(_kinds(events, "release"))
        assert releases_before_success == [("release", "a", [0, 1])]
        rollback = events.index(releases_before_success[0])
        first_commit = events.index(_polls(events, ready=True)[0])
        # One commit round with no re-poll window: one poll per member, no sleep.
        assert [e[1] for e in _polls(events[first_commit:rollback], ready=True)] == ["a", "b"]
        assert _kinds(events[first_commit:rollback], "sleep") == []
        after = events[rollback:]
        assert sorted(e[1] for e in _polls(after, ready=False)) == ["a", "b"]
        assert [e[1] for e in _polls(after, ready=True)] == ["a", "b"]
        assert _kinds(after[: after.index(_polls(after, ready=False)[0])], "sleep") != []
        assert _kinds(events, "abandon") == []
        assert "did not commit" in caplog.text

    def test_retryable_error_in_commit_round_releases_committed_then_succeeds(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[LockAcquisitionError("ssh blip"), _allocated([2, 3])])
        with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2) as allocation:
            releases_before_success = list(_kinds(events, "release"))
            assert [m.host_id for m in allocation.member_allocations] == ["a", "b"]
        assert releases_before_success == [("release", "a", [0, 1])]
        assert [e[1] for e in _polls(events, ready=True)] == ["a", "b", "a", "b"]
        assert _kinds(events, "abandon") == []

    def test_retryable_errors_in_commit_rounds_count_as_failed_cycles(self, group_clock) -> None:
        events = group_clock
        error = LockAcquisitionError("ssh blip")
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[error])
        with pytest.raises(OSError) as excinfo:
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert excinfo.value.__cause__ is error
        assert excinfo.value.failed_host_id == "b"
        assert _kinds(events, "release") == [("release", "a", [0, 1])] * MAX_RETRYABLE_LOCK_ERRORS
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    @pytest.mark.parametrize(
        "make_error",
        [
            pytest.param(_non_retryable_lock_error, id="non_retryable_lock_error"),
            pytest.param(lambda: RuntimeError("unexpected"), id="unexpected_error"),
        ],
    )
    def test_non_retryable_error_releases_committed_abandons_all_and_tags_member(self, group_clock, make_error) -> None:
        events = group_clock
        error = make_error()
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[error])
        with pytest.raises(type(error)) as excinfo:
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert excinfo.value is error
        assert excinfo.value.failed_host_id == "b"
        assert _kinds(events, "release") == [("release", "a", [0, 1])]
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    def test_ten_consecutive_retryable_error_cycles_raise_oserror(self, group_clock) -> None:
        events = group_clock
        error = LockVersionError(required_version=99, current_version=3)
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[error], commit=[_allocated([2, 3])])
        with pytest.raises(OSError) as excinfo:
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert excinfo.value.__cause__ is error
        assert excinfo.value.failed_host_id == "b"
        assert len(_polls(events, ready=False)) == 20
        assert _polls(events, ready=True) == []
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    def test_clean_cycle_resets_retryable_error_budget(self, group_clock) -> None:
        events = group_clock
        errors = [LockAcquisitionError("ssh blip") for _ in range(9)]
        b_wait = [*errors, _queued(30), *errors, _ASSEMBLED]
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=b_wait, commit=[_allocated([2, 3])])
        with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2) as allocation:
            assert len(allocation.member_allocations) == 2

    def test_timeout_is_tagged_with_member_not_ready_in_last_cycle(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(300)], commit=[_allocated([2, 3])])
        with pytest.raises(TimeoutException) as excinfo:
            with self._group(a, b).get_core_allocation(
                _cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=20
            ):
                raise pytest.fail.Exception("body must not run")
        assert excinfo.value.failed_host_id == "b"

    def test_timeout_after_commit_phase_is_tagged_with_first_uncommitted_member(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_ASSEMBLED])
        with pytest.raises(TimeoutException) as excinfo:
            with self._group(a, b).get_core_allocation(
                _cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=COMMIT_WINDOW
            ):
                raise pytest.fail.Exception("body must not run")
        assert excinfo.value.failed_host_id == "b"

    def test_timeout_before_any_cycle_is_untagged(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        with pytest.raises(TimeoutException) as excinfo:
            with self._group(a).get_core_allocation(
                _cm_collector(), collective_ranks=1, lnc_config=2, timeout_seconds=0
            ):
                raise pytest.fail.Exception("body must not run")
        assert getattr(excinfo.value, "failed_host_id", None) is None

    def test_setup_failure_on_later_member_abandons_every_member(self, group_clock) -> None:
        events = group_clock
        error = OSError("ssh down")
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        b._ensure_core_lock_manager.side_effect = error
        with pytest.raises(OSError) as excinfo:
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert excinfo.value is error
        assert excinfo.value.failed_host_id == "b"
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    def test_instance_type_failure_abandons_every_member(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        a.get_instance_type.side_effect = OSError("neuron-ls failed")
        with pytest.raises(OSError, match="neuron-ls failed"):
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    def test_abandon_failure_on_one_member_does_not_skip_others_or_mask_error(self, group_clock, caplog) -> None:
        events = group_clock
        error = LockAcquisitionError("corrupt lock state")
        error.retryable = False
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[error], commit=[_allocated([2, 3])])
        a._abandon_queue.side_effect = RuntimeError("dequeue exploded")
        with caplog.at_level("WARNING"):
            with pytest.raises(LockAcquisitionError) as excinfo:
                with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                    raise pytest.fail.Exception("body must not run")
        assert excinfo.value is error
        b._abandon_queue.assert_called_once_with()
        assert "dequeue exploded" in caplog.text

    def test_release_failure_on_one_committed_member_does_not_skip_others(self, group_clock, caplog) -> None:
        events = group_clock
        error = LockAcquisitionError("corrupt lock state")
        error.retryable = False
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        c = _alloc_member("c", events, wait=[_ASSEMBLED], commit=[error])
        a._release_cores.side_effect = RuntimeError("release exploded")
        with caplog.at_level("WARNING"):
            with pytest.raises(LockAcquisitionError) as excinfo:
                with self._group(a, b, c).get_core_allocation(_cm_collector(), collective_ranks=3, lnc_config=2):
                    raise pytest.fail.Exception("body must not run")
        assert excinfo.value is error
        assert _kinds(events, "release") == [("release", "b", [2, 3])]
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b"), ("abandon", "c")]
        assert "release exploded" in caplog.text

    def test_wait_cycle_logs_every_member_retryable_error(self, group_clock, caplog) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[LockAcquisitionError("a blip"), _ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[LockAcquisitionError("b blip"), _ASSEMBLED], commit=[_allocated([2, 3])])
        with caplog.at_level("WARNING"):
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                pass
        assert "a blip" in caplog.text
        assert "b blip" in caplog.text

    def test_allocated_without_cores_releases_committed_and_abandons_all(self, group_clock) -> None:
        events = group_clock
        no_cores = AllocationOutcome(status=AllocationStatus.ALLOCATED, logical_cores=None, physical_cores=None)
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[no_cores])
        with pytest.raises(TimeoutException, match="without a core assignment"):
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert _kinds(events, "release") == [("release", "a", [0, 1])]
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    @pytest.mark.parametrize("entry_point", ["soft_join_queue", "get_core_allocation"])
    def test_uneven_ranks_raise_before_any_member_call(self, group_clock, entry_point) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        group = self._group(a, b)

        def call() -> None:
            if entry_point == "soft_join_queue":
                group.soft_join_queue(_cm_collector(), collective_ranks=3, lnc_config=2)
            else:
                with group.get_core_allocation(_cm_collector(), collective_ranks=3, lnc_config=2):
                    raise pytest.fail.Exception("body must not run")

        with pytest.raises(ValueError, match="3 collective ranks cannot be split evenly across 2 hosts"):
            call()
        assert events == []
        for member in (a, b):
            member._ensure_core_lock_manager.assert_not_called()


class _FractionRng(random.Random):
    """A ``random.Random`` whose ``uniform`` returns the next fraction of each range, cycling."""

    def __init__(self, *fractions: float) -> None:
        super().__init__()
        self._fractions = list(fractions)
        self._next = 0

    def uniform(self, a: float, b: float) -> float:
        fraction = self._fractions[self._next % len(self._fractions)]
        self._next += 1
        return a + fraction * (b - a)


class TestSshHostGroupAttemptPolicy:
    """Per-attempt hold cap and the partial-readiness withdrawal of SshHostGroup."""

    @staticmethod
    def _group(*members, rng=None):
        return host_management.SshHostGroup(tuple(members), rng=rng)

    @staticmethod
    def _caps(*members) -> list:
        calls = [c for m in members for c in (*m._join_queue.call_args_list, *m._poll_queue.call_args_list)]
        return [c.kwargs.get("hold_cap_seconds") for c in calls]

    def test_sampled_bounds_keep_partial_wait_below_cap(self) -> None:
        for fraction in (0.0, 1.0):
            policy = host_management.sample_group_attempt_policy(180, _FractionRng(fraction))
            assert policy.hold_cap_seconds == (162 if fraction == 0.0 else 198)
            assert policy.partial_wait_seconds == (90.0 if fraction == 0.0 else 135.0)
            assert policy.retry_backoff_seconds == (POLL_PERIOD if fraction == 0.0 else 2 * POLL_PERIOD)

    def test_default_policy_is_independent_of_the_global_random_seed(self) -> None:
        events: list = []
        policies = []
        for _ in range(2):
            random.seed(42)
            group = self._group(_fake_member("a", events), _fake_member("b", events))
            policies.append(group._attempt_policy())
        assert policies[0] != policies[1]
        for policy in policies:
            assert 162 <= policy.hold_cap_seconds <= 198
            assert 90 <= policy.partial_wait_seconds <= 135

    def test_one_cap_per_attempt_across_join_wait_and_commit_then_resampled(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(5), _ASSEMBLED], commit=[_allocated([2, 3])])
        a.patience_seconds = 200
        group = self._group(a, b, rng=_FractionRng(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        collector = _cm_collector()
        group.soft_join_queue(collector, collective_ranks=2, lnc_config=2)
        with group.get_core_allocation(collector, collective_ranks=2, lnc_config=2):
            pass
        # The smallest positive member patience (b's 180s) is the base; every held call sent one cap.
        assert set(self._caps(a, b)) == {162}
        assert len(self._caps(a, b)) == 2 + len(_polls(events))
        # All members assembled: committed on the first attempt with no withdrawal.
        assert _kinds(events, "abandon") == []
        for member in (a, b):
            member._join_queue.reset_mock()
            member._poll_queue.reset_mock()
        with group.get_core_allocation(collector, collective_ranks=2, lnc_config=2):
            pass
        assert set(self._caps(a, b)) == {198}

    def test_non_positive_patience_falls_back_to_default_base(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        a.patience_seconds = b.patience_seconds = 0
        with self._group(a, b, rng=_FractionRng(1.0)).get_core_allocation(
            _cm_collector(), collective_ranks=2, lnc_config=2
        ):
            pass
        assert set(self._caps(a, b)) == {round(1.1 * DEFAULT_PATIENCE_SECONDS)}

    def test_partial_readiness_withdraws_every_member_then_backs_off_then_rotates(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(30, position=1)], commit=[_allocated([2, 3])])
        start = host_management.time.time()
        with pytest.raises(QueuePatienceRotation, match="stalled behind a predecessor for 90s") as excinfo:
            with self._group(a, b, rng=_FractionRng(0.0)).get_core_allocation(
                _cm_collector(), collective_ranks=2, lnc_config=2
            ):
                raise pytest.fail.Exception("body must not run")
        # Partial since the first cycle; withdrawn in the first cycle at least 90s later.
        assert host_management.time.time() - start == 90 + POLL_PERIOD
        abandons = [events.index(e) for e in _kinds(events, "abandon")]
        assert [events[i][1] for i in abandons] == ["a", "b"]
        # Both entries are dequeued before the backoff sleep, which is the last thing before the rotation.
        assert events[-1] == ("sleep", POLL_PERIOD) and max(abandons) < len(events) - 1
        assert _polls(events, ready=True) == []
        assert getattr(excinfo.value, "failed_host_id", None) is None

    def test_partial_wait_is_not_restarted_when_readiness_is_lost(self, group_clock) -> None:
        """Stalled at 0s, all unready from 5s to 120s (past the 90s budget), stalled again at 125s."""
        events = group_clock
        a_wait = [_ASSEMBLED, *[_queued(30)] * 24, _ASSEMBLED]
        a = _alloc_member("a", events, wait=a_wait, commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(30, position=1)], commit=[_allocated([2, 3])])
        start = host_management.time.time()
        with pytest.raises(QueuePatienceRotation):
            with self._group(a, b, rng=_FractionRng(0.0)).get_core_allocation(
                _cm_collector(), collective_ranks=2, lnc_config=2
            ):
                raise pytest.fail.Exception("body must not run")
        # All-unready cycles do not withdraw; the resumed stall uses the clock started at 0s.
        assert host_management.time.time() - start == 125 + POLL_PERIOD

    def test_stalled_member_back_at_queue_head_waits_on_an_old_stall_clock(self, group_clock) -> None:
        """B heads its queue, stalls behind a predecessor from 5s, then heads it again past the budget."""
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b_wait = [_queued(30), *[_queued(30, position=1)] * 4, *[_queued(150)] * 40, _ASSEMBLED]
        b = _alloc_member("b", events, wait=b_wait, commit=[_allocated([2, 3])])
        start = host_management.time.time()
        with self._group(a, b, rng=_FractionRng(1.0)).get_core_allocation(
            _cm_collector(), collective_ranks=2, lnc_config=2
        ):
            committed_at = host_management.time.time()
        # Position 0 again is not a new best, but a lagging member at the head is not stalled.
        assert committed_at - start == 225
        assert _kinds(events, "abandon") == []

    @pytest.mark.parametrize(
        ("a_wait", "b_wait"),
        [
            pytest.param([_queued(30)], [_queued(30)], id="all_unready"),
            pytest.param([_ASSEMBLED], [_DRAINING], id="member_draining"),
        ],
    )
    def test_no_partial_withdrawal_while_all_unready_or_draining(self, group_clock, a_wait, b_wait) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=a_wait, commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=b_wait, commit=[_allocated([2, 3])])
        # Past the largest partial wait (135s) and cap (198s): only the deadline ends the wait.
        with pytest.raises(TimeoutException, match="within 400s") as excinfo:
            with self._group(a, b, rng=_FractionRng(1.0)).get_core_allocation(
                _cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=400
            ):
                raise pytest.fail.Exception("body must not run")
        assert not isinstance(excinfo.value, QueuePatienceRotation)
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]

    @staticmethod
    def _slow_abandon(
        member, events: list, clock: _SleepAdvancedClock, seconds: float, error: Exception | None = None
    ) -> None:
        """Make ``member``'s abandon advance the shared fake clock by ``seconds`` (and optionally fail)."""

        def abandon() -> None:
            events.append(("abandon", member.get_host_id()))
            clock.now += seconds
            if error is not None:
                raise error

        member._abandon_queue.side_effect = abandon

    def _withdrawn(self, events: list, a, b, timeout_seconds: int = 9000) -> None:
        with pytest.raises(QueuePatienceRotation):
            with self._group(a, b, rng=_FractionRng(0.0)).get_core_allocation(
                _cm_collector(), collective_ranks=2, lnc_config=2, timeout_seconds=timeout_seconds
            ):
                raise pytest.fail.Exception("body must not run")

    def _stalled_pair(self, events: list):
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_queued(30, position=1)], commit=[_allocated([2, 3])])
        return a, b

    def test_backoff_never_extends_past_the_allocation_timeout(self, group_clock) -> None:
        events = group_clock
        a, b = self._stalled_pair(events)
        start = host_management.time.time()
        self._withdrawn(events, a, b, timeout_seconds=92)
        assert events[-1] == ("sleep", 2)
        assert host_management.time.time() - start == 92

    def test_backoff_is_the_budget_left_after_cleanup(self, group_timed_clock) -> None:
        events, clock = group_timed_clock
        a, b = self._stalled_pair(events)
        for member in (a, b):
            self._slow_abandon(member, events, clock, 1.0)
        start = host_management.time.time()
        # Withdrawal starts at 90s of a 94s budget; cleanup takes 2s, leaving 2s of the 5s backoff.
        self._withdrawn(events, a, b, timeout_seconds=94)
        assert events[-1] == ("sleep", 2)
        assert host_management.time.time() - start == 94

    def test_no_backoff_once_cleanup_passes_the_deadline(self, group_timed_clock) -> None:
        events, clock = group_timed_clock
        a, b = self._stalled_pair(events)
        for member in (a, b):
            self._slow_abandon(member, events, clock, 2.0)
        start = host_management.time.time()
        # Cleanup runs from 90s to 94s, past the 92s deadline: it completes, then no sleep is added.
        self._withdrawn(events, a, b, timeout_seconds=92)
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]
        assert events[-1] == ("sleep", 0)
        assert host_management.time.time() - start == 94

    def test_failed_withdrawal_cleanup_attempts_every_member_and_is_not_reported_as_done(
        self, group_timed_clock, caplog
    ) -> None:
        events, clock = group_timed_clock
        a, b = self._stalled_pair(events)
        self._slow_abandon(a, events, clock, 0.0, error=OSError("ssh dropped"))
        with caplog.at_level("INFO"):
            self._withdrawn(events, a, b)
        # The failing first member does not skip the second.
        assert _kinds(events, "abandon") == [("abandon", "a"), ("abandon", "b")]
        assert "[a] Abandoning the core queue entry failed: ssh dropped" in caplog.text
        assert "withdrawal incomplete on a" in caplog.text
        assert "withdrew every queue entry" not in caplog.text

    def test_lagging_member_at_queue_head_waits_without_withdrawing(self, group_clock) -> None:
        """B at the head of its queue waits for active work to release, past the largest partial wait."""
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[*[_queued(150)] * 30, _ASSEMBLED], commit=[_allocated([2, 3])])
        start = host_management.time.time()
        with self._group(a, b, rng=_FractionRng(1.0)).get_core_allocation(
            _cm_collector(), collective_ranks=2, lnc_config=2
        ):
            committed_at = host_management.time.time()
        assert committed_at - start == 150
        assert _kinds(events, "abandon") == []

    def test_queue_position_progress_restarts_the_stall_budget(self, group_clock) -> None:
        """B moves up within every 90s budget, so a 180s stall behind predecessors never withdraws."""
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        steps = [_queued(240 - 60 * step, position=3 - step) for step in range(3) for _ in range(12)]
        b = _alloc_member("b", events, wait=[*steps, _queued(30), _ASSEMBLED], commit=[_allocated([2, 3])])
        start = host_management.time.time()
        with self._group(a, b, rng=_FractionRng(0.0)).get_core_allocation(
            _cm_collector(), collective_ranks=2, lnc_config=2
        ):
            committed_at = host_management.time.time()
        assert committed_at - start == 185
        assert _kinds(events, "abandon") == []

    @pytest.mark.parametrize(
        ("a_wait", "b_wait"),
        [
            pytest.param([_ASSEMBLED], [_queued(30, position=1)], id="frozen"),
            pytest.param(
                [_ASSEMBLED],
                [*[_queued(30, position=1)] * 6, *[_queued(30, position=2)] * 6, _queued(30, position=1)],
                id="worsened_then_back_to_best",
            ),
            pytest.param(
                [*[_ASSEMBLED] * 6, *[_queued(30, position=1)] * 6],
                [*[_queued(30, position=1)] * 6, _ASSEMBLED],
                id="crossed_swap_after_cap_bump",
            ),
            pytest.param(
                [_ASSEMBLED],
                [*[_queued(30, position=1)] * 3, LockAcquisitionError("blip"), _queued(30, position=1)],
                id="retryable_error",
            ),
            pytest.param([_ASSEMBLED], [_queued(30, position=None)], id="unknown_position"),
        ],
    )
    def test_stall_budget_is_not_restarted_without_a_new_best_position(self, group_clock, a_wait, b_wait) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=a_wait, commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=b_wait, commit=[_allocated([2, 3])])
        start = host_management.time.time()
        self._withdrawn(events, a, b)
        assert host_management.time.time() - start == 90 + POLL_PERIOD
        assert _polls(events, ready=True) == []


class TestSshHostGroupIntegration(TestSshHostPollLoopIntegration):
    """No-hardware integration: an SshHostGroup of two real SshHosts, each running the
    real lock helper against its own locks.json, sharing one fake clock."""

    _ALIASES = ("host-a", "host-b")

    def _member_setup(
        self, tmp_path, alias, total_physical_cores=8, on_call=None, patience_seconds=DEFAULT_PATIENCE_SECONDS
    ):
        """Build a real SshHost with its own helper executor and locks.json under tmp_path/alias."""
        root = tmp_path / alias
        root.mkdir()
        helpers_file, lock_file, locks_json = self._paths(root)
        executor = _HelperExecutor(helpers_file, lock_file, locks_json, on_call=on_call)
        host = _make_ssh_host(
            root, total_physical_cores=total_physical_cores, patience_seconds=patience_seconds, ssh_alias=alias
        )
        return host, executor, locks_json

    @contextlib.contextmanager
    def _wired(self, clock, members):
        """Route each member's lock calls to its own helper and patch the shared clock."""
        with contextlib.ExitStack() as stack:
            for host, executor in members:
                stack.enter_context(patch.object(host, "_get_remote_executor", return_value=executor))
                stack.enter_context(patch.object(host, "get_instance_type", return_value="trn1"))
            stack.enter_context(patch("time.sleep", clock.sleep))
            stack.enter_context(patch("time.time", clock.time))
            yield

    def _locked(self, locks_json) -> dict:
        with open(locks_json) as f:
            return json.load(f).get("physical_neuron_cores_lock_timeout", {})

    def _live_locks(self, locks_json, now) -> dict:
        return {core: until for core, until in self._locked(locks_json).items() if until > now}

    def test_soft_join_then_lock_commits_both_members_together(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        for path in (locks_a, locks_b):
            self._write_locks(path)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            entry_ids = [[e["entry_id"] for e in self._read_queue(p)] for p in (locks_a, locks_b)]
            assert all(len(ids) == 1 for ids in entry_ids)

            with group.get_core_allocation(
                collector, collective_ranks=4, lnc_config=2, timeout_seconds=300
            ) as allocation:
                assert isinstance(allocation, GroupCoreAllocation)
                assert [m.host_id for m in allocation.member_allocations] == list(self._ALIASES)
                assert all(len(m.logical_core_ids) == 2 for m in allocation.member_allocations)
                # 2 logical cores at lnc=2 hold 4 physical cores on each member.
                for path in (locks_a, locks_b):
                    assert len(self._live_locks(path, clock.time())) == 4
                    assert self._read_queue(path) == []
                # The soft-joined entries are the ones that committed.
                for path, ids in zip((locks_a, locks_b), entry_ids, strict=True):
                    commits = [e for e in self._read_events(path) if e["event"] == "COMMIT"]
                    assert len(commits) == 1
                    assert [e["entry_id"] for e in self._read_events(path) if e["event"] == "ENQUEUE"] == ids

        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []
            assert self._live_locks(path, clock.time()) == {}

    def test_member_waits_in_queue_with_hold_until_sibling_is_ready(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Long past the readiness grace, but within patience.
        b_free_at = int(_REALISTIC_EPOCH) + 6 * COMMIT_WINDOW
        assert b_free_at - _REALISTIC_EPOCH < DEFAULT_PATIENCE_SECONDS
        snapshots: list[tuple[float, list, dict]] = []
        b_snapshots: list[tuple[float, dict]] = []

        def record_a(_call_n, path):
            snapshots.append((clock.time(), self._read_queue(path), self._locked(path)))

        def record_b(_call_n, path):
            # Read from B's own hook: wait-phase polls run concurrently, so A's hook could race B's write.
            b_snapshots.append((clock.time(), self._locked(path)))

        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a", on_call=record_a)
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b", on_call=record_b)
        self._write_locks(locks_a)
        self._write_locks(locks_b, locked_until=b_free_at)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            (entry_a,) = self._read_queue(locks_a)
            with group.get_core_allocation(
                collector, collective_ranks=4, lnc_config=2, timeout_seconds=300
            ) as allocation:
                assert clock.time() >= b_free_at
                assert len(allocation.member_allocations) == 2
                for path in (locks_a, locks_b):
                    assert len(self._live_locks(path, clock.time())) == 4

        waiting = [s for s in snapshots if s[0] < b_free_at and s[1]]
        # A was polled across many readiness grace windows while B was busy.
        assert waiting and waiting[-1][0] - waiting[0][0] > 2 * COMMIT_WINDOW
        assert [e for e in self._read_events(locks_a) if e["event"] == "BUMP"] == []
        for _t, queue_a, locked_a in waiting:
            assert [(e["entry_id"], e.get("hold", False)) for e in queue_a] == [(entry_a["entry_id"], True)]
            assert locked_a == {}
        b_waiting = [(t, locked_b) for t, locked_b in b_snapshots if t < b_free_at]
        # B was polled across the same span as A and stayed locked by its holder throughout.
        assert b_waiting and b_waiting[-1][0] - b_waiting[0][0] > 2 * COMMIT_WINDOW
        assert all(set(locked_b.values()) == {b_free_at} for _t, locked_b in b_waiting)
        commits_a = [e for e in self._read_events(locks_a) if e["event"] == "COMMIT"]
        commits_b = [e for e in self._read_events(locks_b) if e["event"] == "COMMIT"]
        assert len(commits_a) == len(commits_b) == 1
        assert commits_a[0]["ts"] >= b_free_at
        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []

    def test_rotation_on_one_member_leaves_no_entry_on_any_member(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        self._write_locks(locks_a)
        self._write_locks(locks_b, locked_until=int(_REALISTIC_EPOCH) + 100000)
        group = SshHostGroup((host_a, host_b))

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            with pytest.raises(QueuePatienceRotation, match="host-b"):
                group.soft_join_queue(_cm_collector(), collective_ranks=4, lnc_config=2)

        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []
            assert [e for e in self._read_events(path) if e["event"] == "ENQUEUE"] == []

    def test_timeout_abandons_both_queues(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        self._write_locks(locks_a)
        # Within patience, so the join succeeds, but past the allocation timeout.
        self._write_locks(locks_b, locked_until=int(_REALISTIC_EPOCH) + 120)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            assert [len(self._read_queue(p)) for p in (locks_a, locks_b)] == [1, 1]
            with pytest.raises(TimeoutException, match=r"\[host-a\] Unable to allocate 4 logical cores across 2"):
                with group.get_core_allocation(collector, collective_ranks=4, lnc_config=2, timeout_seconds=30):
                    raise pytest.fail.Exception("body must not run")

        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []
            # Never committed, as opposed to committed and then released.
            assert [e for e in self._read_events(path) if e["event"] == "COMMIT"] == []
        assert self._live_locks(locks_a, clock.time()) == {}

    def test_ssh_host_soft_join_is_unheld_and_held_join_persists_hold(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        solo, exec_solo, locks_solo = self._member_setup(tmp_path, "host-a")
        held, exec_held, locks_held = self._member_setup(tmp_path, "host-b")
        for path in (locks_solo, locks_held):
            self._write_locks(path)
        collector = _cm_collector()

        with self._wired(clock, [(solo, exec_solo), (held, exec_held)]):
            solo.soft_join_queue(collector, collective_ranks=2, lnc_config=2)
            held._join_queue(collector, 2, 2, hold=True)

        (solo_entry,) = self._read_queue(locks_solo)
        assert "hold" not in solo_entry
        assert "hold_cap" not in solo_entry
        (held_entry,) = self._read_queue(locks_held)
        assert held_entry["hold"] is True
        assert held_entry["hold_cap"] == held.patience_seconds

    def test_retry_loop_yields_group_that_locks_both_members(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        manager = HostManager(
            state_store=_initialized_store(str(store_dir), []),
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=False,
        )
        OwnerHostStateStore(str(store_dir))._update(
            lambda state: HostState(
                hosts=[
                    TestSizeBasedHostRouting._member("host-b", "L", 1, size=2),
                    TestSizeBasedHostRouting._member("host-a", "L", 0, size=2),
                ]
            )
        )
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a", total_physical_cores=128)
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b", total_physical_cores=128)
        for path in (locks_a, locks_b):
            self._write_locks(path, ncores=128)
        # HostManager._get_host serves cached connections, so the claim resolves to these hosts.
        manager.target_hosts.update({"host-a": host_a, "host-b": host_b})
        collector = _cm_collector()

        def depths() -> dict[str, int]:
            state = manager.state_store.read()
            assert state is not None
            return {rec.resolved.ssh_host: rec.work_queue_depth for rec in state.hosts}

        targets = []
        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            # 128 ranks at lnc=2 is 256 physical cores: above the multi-node threshold.
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=collector,
                    collective_ranks=128,
                    lnc_config=2,
                    backoff_seconds=0,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as target:
                        targets.append(target)
                        assert isinstance(target, SshHostGroup)
                        assert target.hosts == (host_a, host_b)
                        assert depths() == {"host-a": 256, "host-b": 256}
                        # prepare_host needs a real transport (unit-tested separately), so it is skipped.
                        target.soft_join_queue(collector, collective_ranks=128, lnc_config=2)
                        with target.get_core_allocation(
                            collector, collective_ranks=128, lnc_config=2, timeout_seconds=300
                        ) as allocation:
                            assert [m.host_id for m in allocation.member_allocations] == ["host-a", "host-b"]
                            for path in (locks_a, locks_b):
                                assert len(self._live_locks(path, clock.time())) == 128

        assert len(targets) == 1
        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []
            assert self._live_locks(path, clock.time()) == {}
        assert depths() == {"host-a": 0, "host-b": 0}

    def _retry_manager(self, tmp_path, records, hosts) -> HostManager:
        """A HostManager over ``records`` that serves the given real SshHosts from its connection cache."""
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        manager = HostManager(
            state_store=_initialized_store(str(store_dir), []),
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=False,
        )
        OwnerHostStateStore(str(store_dir))._update(lambda state: HostState(hosts=list(records)))
        manager.target_hosts.update({host.get_host_id(): host for host in hosts})
        return manager

    def _two_member_manager(self, tmp_path):
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a", total_physical_cores=128)
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b", total_physical_cores=128)
        for path in (locks_a, locks_b):
            self._write_locks(path, ncores=128)
        records = [
            TestSizeBasedHostRouting._member("host-a", "L", 0, size=2),
            TestSizeBasedHostRouting._member("host-b", "L", 1, size=2),
        ]
        manager = self._retry_manager(tmp_path, records, (host_a, host_b))
        return manager, [(host_a, exec_a), (host_b, exec_b)], (locks_a, locks_b)

    def _single_host_manager(self, tmp_path):
        host, executor, locks_json = self._member_setup(tmp_path, "host-a")
        self._write_locks(locks_json)
        record = HostRecord(resolved=ResolvedHost(ssh_host="host-a", host_type=Platforms.TRN2, num_physical_cores=8))
        manager = self._retry_manager(tmp_path, [record], (host,))
        return manager, host, executor, locks_json

    @staticmethod
    def _run_attempt(manager, collector, collective_ranks, body) -> None:
        """Run ``body(target)`` inside one retry-loop attempt; exceptions from ``body`` propagate."""
        with closing(
            manager.get_host_assignment_with_retry(
                platform_target=Platforms.TRN2,
                collector=collector,
                collective_ranks=collective_ranks,
                lnc_config=2,
                backoff_seconds=0,
            )
        ) as host_generator:
            for host_attempt in host_generator:
                with host_attempt as target:
                    body(target)

    def test_retry_loop_abandons_group_entries_when_the_attempt_fails(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        manager, members, paths = self._two_member_manager(tmp_path)
        collector = _cm_collector()

        def body(target):
            target.soft_join_queue(collector, collective_ranks=128, lnc_config=2)
            assert [len(self._read_queue(p)) for p in paths] == [1, 1]
            target.execute_command("cmd", "/remote", collector, 128, 2)

        with self._wired(clock, members), pytest.raises(NotImplementedError, match="multi-node"):
            self._run_attempt(manager, collector, 128, body)
        for path in paths:
            assert self._read_queue(path) == []

    def test_retry_loop_abandons_single_host_entry_when_the_attempt_fails(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        manager, host, executor, locks_json = self._single_host_manager(tmp_path)
        collector = _cm_collector()

        def body(target):
            target.soft_join_queue(collector, collective_ranks=2, lnc_config=2)
            assert len(self._read_queue(locks_json)) == 1
            raise RuntimeError("upload failed")

        with self._wired(clock, [(host, executor)]), pytest.raises(RuntimeError, match="upload failed"):
            self._run_attempt(manager, collector, 2, body)
        assert self._read_queue(locks_json) == []

    def test_retry_loop_abandon_after_successful_allocation_is_a_noop(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        manager, host, executor, locks_json = self._single_host_manager(tmp_path)
        collector = _cm_collector()
        real_dequeue = CoreLockManager.dequeue

        def body(target):
            target.soft_join_queue(collector, collective_ranks=2, lnc_config=2)
            with target.get_core_allocation(collector, collective_ranks=2, lnc_config=2, timeout_seconds=300):
                pass

        with (
            self._wired(clock, [(host, executor)]),
            patch.object(CoreLockManager, "dequeue", autospec=True, side_effect=real_dequeue) as dequeue,
        ):
            self._run_attempt(manager, collector, 2, body)
        dequeue.assert_not_called()
        assert self._read_queue(locks_json) == []
        assert [e for e in self._read_events(locks_json) if e["event"] == "COMMIT"] != []

    def test_retry_loop_abandon_failure_does_not_mask_the_attempt_error(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        manager, host, executor, _locks_json = self._single_host_manager(tmp_path)

        def body(_target):
            raise RuntimeError("upload failed")

        with (
            self._wired(clock, [(host, executor)]),
            patch.object(host, "abandon_queue", side_effect=ValueError("abandon broke")) as abandon,
            pytest.raises(RuntimeError, match="upload failed"),
        ):
            self._run_attempt(manager, _cm_collector(), 2, body)
        abandon.assert_called_once_with()

    @staticmethod
    @contextlib.contextmanager
    def _counting_teardown():
        """Count real ``record_contention_metrics`` and ``dequeue`` calls per manager host."""
        flushes: collections.Counter = collections.Counter()
        dequeues: collections.Counter = collections.Counter()
        real_flush = CoreLockManager.record_contention_metrics
        real_dequeue = CoreLockManager.dequeue

        def flush(manager):
            flushes[manager.host] += 1
            return real_flush(manager)

        def dequeue(manager):
            dequeues[manager.host] += 1
            return real_dequeue(manager)

        with (
            patch.object(CoreLockManager, "record_contention_metrics", autospec=True, side_effect=flush),
            patch.object(CoreLockManager, "dequeue", autospec=True, side_effect=dequeue),
        ):
            yield flushes, dequeues

    def test_single_host_timeout_then_abandon_flushes_metrics_once(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host, executor, locks_json = self._member_setup(tmp_path, "host-a")
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100000)
        collector = _cm_collector()

        with self._wired(clock, [(host, executor)]), self._counting_teardown() as (flushes, dequeues):
            with pytest.raises(TimeoutException, match="Unable to allocate"):
                with host.get_core_allocation(collector, collective_ranks=2, lnc_config=2, timeout_seconds=30):
                    raise pytest.fail.Exception("body must not run")
            host.abandon_queue()

        no_cores = [
            c for c in collector.record_metric.call_args_list if c.args[0] == MetricName.CORE_LOCK_NO_CORES_COUNT
        ]
        assert len(no_cores) == 1
        assert flushes == {"host-a": 1}
        assert dequeues == {"host-a": 1}
        assert host._core_lock_manager is None
        assert self._read_queue(locks_json) == []

    def test_group_acquire_failure_then_abandon_tears_down_each_member_once(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        self._write_locks(locks_a)
        self._write_locks(locks_b, locked_until=int(_REALISTIC_EPOCH) + 120)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()

        with (
            self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]),
            self._counting_teardown() as (flushes, dequeues),
        ):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            with pytest.raises(TimeoutException):
                with group.get_core_allocation(collector, collective_ranks=4, lnc_config=2, timeout_seconds=30):
                    raise pytest.fail.Exception("body must not run")
            group.abandon_queue()

        assert flushes == {"host-a": 1, "host-b": 1}
        assert dequeues == {"host-a": 1, "host-b": 1}
        assert host_a._core_lock_manager is None and host_b._core_lock_manager is None
        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []

    def test_patience_rotation_then_abandon_keeps_the_manager_and_sends_no_dequeue(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host, executor, locks_json = self._member_setup(tmp_path, "host-a")
        self._write_locks(locks_json, locked_until=int(_REALISTIC_EPOCH) + 100000)

        with self._wired(clock, [(host, executor)]), self._counting_teardown() as (flushes, dequeues):
            with pytest.raises(QueuePatienceRotation):
                host.soft_join_queue(_cm_collector(), collective_ranks=2, lnc_config=2)
            manager = host._core_lock_manager
            assert manager is not None
            host.abandon_queue()

        assert host._core_lock_manager is manager
        assert flushes == {}
        assert dequeues == {}

    def test_group_rotation_then_abandon_sends_no_dequeue(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        self._write_locks(locks_a)
        self._write_locks(locks_b, locked_until=int(_REALISTIC_EPOCH) + 100000)
        group = SshHostGroup((host_a, host_b))

        with (
            self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]),
            self._counting_teardown() as (flushes, dequeues),
        ):
            with pytest.raises(QueuePatienceRotation):
                group.soft_join_queue(_cm_collector(), collective_ranks=4, lnc_config=2)
            group.abandon_queue()

        assert flushes == {}
        assert dequeues == {}

    def test_member_release_failure_is_recorded_on_the_caller_collector(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        for path in (locks_a, locks_b):
            self._write_locks(path)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()
        real_release = CoreLockManager.release

        def release(manager, cores):
            if manager.host == "host-b":
                raise LockAcquisitionError("ssh blip")
            return real_release(manager, cores)

        with (
            self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]),
            patch.object(CoreLockManager, "release", autospec=True, side_effect=release),
        ):
            with group.get_core_allocation(collector, collective_ranks=4, lnc_config=2, timeout_seconds=300):
                pass
        failures = [
            c.args[1:]
            for c in collector.record_metric.call_args_list
            if c.args[0] == MetricName.CORE_LOCK_RELEASE_FAILED_COUNT
        ]
        assert failures == [(1,)]

    def _competitor(self, tmp_path, alias, locks_json) -> CoreLockManager:
        """A single-host caller on ``alias``'s helper files, outside the group."""
        helpers_file, lock_file, _ = self._paths(tmp_path / alias)
        return CoreLockManager(
            alias,
            total_physical_cores=8,
            collector=NoopMetricsCollector(),
            executor=_HelperExecutor(helpers_file, lock_file, locks_json),
            host_locking_version=3,
        )

    @pytest.mark.parametrize(
        "locked_cores",
        [
            pytest.param(tuple(range(8)), id="all_cores_locked"),
            # Free cores cover the request by count, but no aligned 4-core block is free.
            pytest.param((2, 3, 4, 5), id="lock_straddles_aligned_blocks"),
        ],
    )
    def test_group_beats_ready_backfill_on_busy_member_host(self, tmp_path, locked_cores) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # Past the readiness grace, within patience.
        b_free_at = int(_REALISTIC_EPOCH) + 6 * COMMIT_WINDOW
        assert b_free_at - _REALISTIC_EPOCH < DEFAULT_PATIENCE_SECONDS
        competitor: list[CoreLockManager] = []
        s1_polls: list[tuple[float, AllocationStatus]] = []

        def poll_s1(_call_n, _path):
            # S1 is ready and wants all 8 cores of host B; it polls just before each group call on B.
            if competitor and not any(status == AllocationStatus.ALLOCATED for _t, status in s1_polls):
                outcome = competitor[0].acquire(num_logical_cores=4, lnc_config=2, ready=True)
                s1_polls.append((clock.time(), outcome.status))

        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b", on_call=poll_s1)
        self._write_locks(locks_a)
        self._write_locks(locks_b)
        with open(locks_b) as f:
            state = json.load(f)
        # S0 holds these cores of B until b_free_at.
        state["physical_neuron_cores_lock_timeout"] = {str(core): b_free_at for core in locked_cores}
        with open(locks_b, "w") as f:
            json.dump(state, f)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            (entry_a,) = self._read_queue(locks_a)
            (entry_b,) = self._read_queue(locks_b)
            # S1 enqueues on B after the group member joined.
            competitor.append(self._competitor(tmp_path, "host-b", locks_b))
            with group.get_core_allocation(
                collector, collective_ranks=4, lnc_config=2, timeout_seconds=300
            ) as allocation:
                assert len(allocation.member_allocations) == 2
                for path in (locks_a, locks_b):
                    assert len(self._live_locks(path, clock.time())) == 4
                events_a_in_body = self._read_events(locks_a)
                events_b_in_body = self._read_events(locks_b)

        s1_id = competitor[0].entry_id
        # S1 polled ready=True both while S0 held B and after S0's lock expired, and never got cores.
        assert any(t < b_free_at for t, _s in s1_polls) and any(t >= b_free_at for t, _s in s1_polls)
        assert all(status == AllocationStatus.QUEUED for _t, status in s1_polls)

        def commits(events, entry_id=None):
            return [
                e
                for e in events
                if e["event"] == "COMMIT" and (entry_id is None or e["caller"].startswith(f"{entry_id}:"))
            ]

        # B's region went to the group member: the only COMMIT on B is the group's.
        assert commits(events_b_in_body) == commits(events_b_in_body, entry_b["entry_id"])
        # A did not commit while B's cores were still locked.
        assert [e["ts"] for e in commits(events_a_in_body) if e["ts"] < b_free_at] == []
        # Exactly one commit per member, in the same cycle.
        assert len(commits(events_a_in_body)) == len(commits(events_b_in_body)) == 1
        ((commit_a,), (commit_b,)) = (commits(events_a_in_body), commits(events_b_in_body))
        assert commit_a["caller"].startswith(f"{entry_a['entry_id']}:")
        assert abs(commit_a["ts"] - commit_b["ts"]) < POLL_PERIOD
        # No rollback: A held nothing it had to give back, and no group entry lost its place.
        assert [e for e in events_a_in_body if e["event"] == "RELEASE"] == []
        for events, entry in ((events_a_in_body, entry_a), (events_b_in_body, entry_b)):
            assert [e for e in events if e.get("entry_id") == entry["entry_id"] and e["event"] == "BUMP"] == []

        assert self._read_queue(locks_a) == []
        assert [e["entry_id"] for e in self._read_queue(locks_b)] == [s1_id]
        for path in (locks_a, locks_b):
            assert self._live_locks(path, clock.time()) == {}
            assert len(commits(self._read_events(path))) == 1
            assert len([e for e in self._read_events(path) if e["event"] == "RELEASE"]) == 1

    def test_hold_cap_lapse_bumps_member_then_group_still_completes(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        # A's patience is the base of the hold cap; B stays locked well past it, within B's patience.
        a_hold_cap = 2 * COMMIT_WINDOW
        b_free_at = int(_REALISTIC_EPOCH) + 6 * COMMIT_WINDOW
        assert b_free_at - _REALISTIC_EPOCH > 2 * a_hold_cap
        assert b_free_at - _REALISTIC_EPOCH < DEFAULT_PATIENCE_SECONDS
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a", patience_seconds=a_hold_cap)
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        self._write_locks(locks_a)
        self._write_locks(locks_b, locked_until=b_free_at)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()
        # Pin the cap at A's patience and disable the partial-readiness withdrawal, which would
        # otherwise end the attempt before the cap lapses, to isolate the helper's cap behaviour.
        policy = host_management.GroupAttemptPolicy(
            hold_cap_seconds=a_hold_cap, partial_wait_seconds=float("inf"), retry_backoff_seconds=0.0
        )

        with (
            self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]),
            patch.object(host_management, "sample_group_attempt_policy", return_value=policy),
        ):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            (entry_a,) = self._read_queue(locks_a)
            assert entry_a["hold_cap"] == a_hold_cap
            with group.get_core_allocation(
                collector, collective_ranks=4, lnc_config=2, timeout_seconds=300
            ) as allocation:
                assert len(allocation.member_allocations) == 2
                for path in (locks_a, locks_b):
                    assert len(self._live_locks(path, clock.time())) == 4

        events_a = self._read_events(locks_a)
        entry_events_a = [e for e in events_a if e.get("entry_id") == entry_a["entry_id"]]
        # A's assembled region was reclaimed and A bumped once its cap lapsed, while B was still locked.
        expired_ts = [e["ts"] for e in entry_events_a if e["event"] == "RESERVE_EXPIRED"]
        bump_ts = [e["ts"] for e in entry_events_a if e["event"] == "BUMP"]
        assert bump_ts and bump_ts == expired_ts
        assert _REALISTIC_EPOCH + a_hold_cap < bump_ts[0] < b_free_at
        # The group kept waiting on the same entry: one join, no commit until B freed.
        assert [e for e in events_a if e["event"] == "ENQUEUE"] == [entry_events_a[0]]
        commits_a = [e for e in events_a if e["event"] == "COMMIT"]
        commits_b = [e for e in self._read_events(locks_b) if e["event"] == "COMMIT"]
        assert len(commits_a) == len(commits_b) == 1
        assert commits_a[0]["caller"].startswith(f"{entry_a['entry_id']}:")
        assert commits_a[0]["ts"] >= b_free_at
        assert abs(commits_a[0]["ts"] - commits_b[0]["ts"]) < POLL_PERIOD
        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []
            assert self._live_locks(path, clock.time()) == {}

    def test_members_share_one_set_of_lock_metrics_and_keep_their_managers(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        for path in (locks_a, locks_b):
            self._write_locks(path)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            joined = (host_a._core_lock_manager, host_b._core_lock_manager)
            assert joined[0] is not None and joined[1] is not None
            assert joined[0].collector is collector
            secondary = joined[1].collector
            assert secondary is not collector
            assert isinstance(secondary, NoopMetricsCollector)
            assert secondary.test_name == collector.test_name
            with group.get_core_allocation(collector, collective_ranks=4, lnc_config=2, timeout_seconds=300):
                assert (host_a._core_lock_manager, host_b._core_lock_manager) == joined

        queue_waits = [
            c for c in collector.record_timer.call_args_list if c.args[0] == MetricName.CORE_LOCK_QUEUE_WAIT_TIME
        ]
        assert len(queue_waits) == 1
        # The secondary's lock-helper events still name the test.
        commits_b = [e for e in self._read_events(locks_b) if e["event"] == "COMMIT"]
        assert commits_b and commits_b[0]["caller"].endswith(f":{collector.test_name}")

    def test_missed_commit_rollback_keeps_entry_and_counts_once(self, tmp_path) -> None:
        clock = _FakeClock(start=_REALISTIC_EPOCH)
        host_a, exec_a, locks_a = self._member_setup(tmp_path, "host-a")
        host_b, exec_b, locks_b = self._member_setup(tmp_path, "host-b")
        for path in (locks_a, locks_b):
            self._write_locks(path)
        group = SshHostGroup((host_a, host_b))
        collector = _cm_collector()
        real_poll_a = host_a._poll_queue
        reclaimed: list[float] = []

        def poll_a(*args, ready, hold=False, **kwargs):
            outcome = real_poll_a(*args, ready=ready, hold=hold, **kwargs)
            if ready and outcome.status == AllocationStatus.ALLOCATED and not reclaimed:
                # B's region is taken between its wait and commit polls, so B misses this commit round.
                reclaimed.append(clock.time())
                with open(locks_b) as f:
                    state = json.load(f)
                until = int(clock.time()) + 2 * POLL_PERIOD
                state["physical_neuron_cores_lock_timeout"] = {str(core): until for core in range(8)}
                with open(locks_b, "w") as f:
                    json.dump(state, f)
            return outcome

        with self._wired(clock, [(host_a, exec_a), (host_b, exec_b)]):
            group.soft_join_queue(collector, collective_ranks=4, lnc_config=2)
            (entry_a,) = self._read_queue(locks_a)
            with patch.object(host_a, "_poll_queue", side_effect=poll_a):
                with group.get_core_allocation(
                    collector, collective_ranks=4, lnc_config=2, timeout_seconds=300
                ) as allocation:
                    assert reclaimed
                    assert len(allocation.member_allocations) == 2

        rollbacks = [
            c for c in collector.record_metric.call_args_list if c.args[0] == MetricName.GROUP_COMMIT_ROLLBACK_COUNT
        ]
        assert [c.args[1:] for c in rollbacks] == [(1.0, "Count")]
        commits_a = [e for e in self._read_events(locks_a) if e["event"] == "COMMIT"]
        assert len(commits_a) == 2
        # The rolled-back member re-joined under the same entry id.
        assert all(e["caller"].startswith(f"{entry_a['entry_id']}:") for e in commits_a)
        enqueues_a = [e for e in self._read_events(locks_a) if e["event"] == "ENQUEUE"]
        assert len(enqueues_a) >= 2
        assert {e["entry_id"] for e in enqueues_a} == {entry_a["entry_id"]}
        for path in (locks_a, locks_b):
            assert self._read_queue(path) == []
            assert self._live_locks(path, clock.time()) == {}


class TestSshHostGroupMemberCollectorsAndConcurrency:
    """Per-member collectors, rollback accounting and concurrent wait-phase polls."""

    @staticmethod
    def _group(*members):
        return host_management.SshHostGroup(tuple(members))

    @staticmethod
    def _collectors(mock_method) -> list:
        return [c.args[0] for c in mock_method.call_args_list]

    def test_primary_gets_caller_collector_and_others_one_stable_discarding_collector(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        c = _alloc_member("c", events, wait=[_ASSEMBLED], commit=[_allocated([4, 5])])
        collector = _cm_collector()
        group = self._group(a, b, c)
        group.soft_join_queue(collector, collective_ranks=3, lnc_config=2)
        with group.get_core_allocation(collector, collective_ranks=3, lnc_config=2):
            pass
        member_collectors = []
        for member in (b, c):
            used = {
                id(x)
                for method in (
                    member._check_patience,
                    member._join_queue,
                    member._ensure_core_lock_manager,
                    member._poll_queue,
                )
                for x in self._collectors(method)
            }
            assert len(used) == 1
            (member_collector,) = {
                x for method in (member._check_patience, member._poll_queue) for x in self._collectors(method)
            }
            assert member_collector is not collector
            assert isinstance(member_collector, NoopMetricsCollector)
            assert member_collector.test_name == collector.test_name
            member_collectors.append(member_collector)
        assert member_collectors[0] is not member_collectors[1]
        for method in (a._check_patience, a._join_queue, a._ensure_core_lock_manager, a._poll_queue, a._release_cores):
            assert all(x is collector for x in self._collectors(method))
        # Release failures are recorded on the caller's collector for every member.
        for member in (b, c):
            assert all(x is collector for x in self._collectors(member._release_cores))

    def test_rollback_releases_record_on_the_caller_collector(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        c = _alloc_member("c", events, wait=[_ASSEMBLED], commit=[_ASSEMBLED, _allocated([4, 5])])
        collector = _cm_collector()
        with self._group(a, b, c).get_core_allocation(collector, collective_ranks=3, lnc_config=2):
            pass
        assert [call.kwargs for call in b._release_cores.call_args_list] == [{"end_attempt": False}, {}]
        for member in (a, b, c):
            assert all(x is collector for x in self._collectors(member._release_cores))

    def test_records_member_max_lock_signal_counts_once_per_allocation(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_ASSEMBLED, _allocated([2, 3])])
        c = _alloc_member("c", events, wait=[_ASSEMBLED], commit=[_allocated([4, 5])])
        a._lock_signal_counts.return_value = (1, 0)
        b._lock_signal_counts.return_value = (3, 0)
        c._lock_signal_counts.side_effect = lambda: events.append(("counts", "c")) or (0, 2)
        collector = _cm_collector()
        with self._group(a, b, c).get_core_allocation(collector, collective_ranks=3, lnc_config=2):
            pass
        recorded = [
            call.args
            for call in collector.record_metric.call_args_list
            if call.args[0] in (MetricName.GROUP_MEMBER_MAX_BUMP_COUNT, MetricName.GROUP_MEMBER_MAX_REENQUEUE_COUNT)
        ]
        assert recorded == [
            (MetricName.GROUP_MEMBER_MAX_BUMP_COUNT, 3, "Count"),
            (MetricName.GROUP_MEMBER_MAX_REENQUEUE_COUNT, 2, "Count"),
        ]
        # Read while every member still holds its cores, before any end-of-attempt release.
        counts_at = events.index(("counts", "c"))
        final_releases = [i for i, e in enumerate(events) if e[0] == "release"][-3:]
        assert all(counts_at < i for i in final_releases)

    def test_failed_allocation_records_no_member_max_counts(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_non_retryable_lock_error()])
        collector = _cm_collector()
        with pytest.raises(LockAcquisitionError):
            with self._group(a, b).get_core_allocation(collector, collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        names = [c.args[0] for c in collector.record_metric.call_args_list]
        assert MetricName.GROUP_MEMBER_MAX_BUMP_COUNT not in names
        assert MetricName.GROUP_MEMBER_MAX_REENQUEUE_COUNT not in names

    def test_prepare_host_times_the_group_preparation_once(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a.prepare_host.side_effect = _recording_prepare("a", events)
        b.prepare_host.side_effect = _recording_prepare("b", events)
        collector = _cm_collector()
        with self._group(a, b).prepare_host("/local/dir", collector):
            pass
        timers = [c.args[0] for c in collector.timer.call_args_list]
        assert timers.count(MetricName.GROUP_PREPARE_TIME) == 1

    def test_new_caller_collector_gets_new_member_collectors(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_allocated([2, 3])])
        group = self._group(a, b)
        first, second = _cm_collector(), _cm_collector()
        second.test_name = "other-test"
        group.soft_join_queue(first, collective_ranks=2, lnc_config=2)
        group.soft_join_queue(second, collective_ranks=2, lnc_config=2)
        first_b, second_b = self._collectors(b._check_patience)
        assert first_b is not second_b
        assert first_b is not first and second_b is not second
        assert self._collectors(a._check_patience) == [first, second]
        assert second_b.test_name == "other-test"

    def test_prepare_host_passes_caller_collector_only_to_primary(self) -> None:
        events: list = []
        a = _fake_member("a", events)
        b = _fake_member("b", events)
        a.prepare_host.side_effect = _recording_prepare("a", events)
        b.prepare_host.side_effect = _recording_prepare("b", events)
        collector = _cm_collector()
        group = self._group(a, b)
        with group.prepare_host("/local/dir", collector):
            pass
        group.soft_join_queue(collector, collective_ranks=2, lnc_config=2)
        assert a.prepare_host.call_args.args[1] is collector
        b_collector = b.prepare_host.call_args.args[1]
        assert b_collector is not collector
        assert isinstance(b_collector, NoopMetricsCollector)
        assert b._check_patience.call_args.args[0] is b_collector

    def test_missed_commit_records_one_rollback_and_keeps_the_attempt(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[_ASSEMBLED, _ASSEMBLED, _allocated([2, 3])])
        collector = _cm_collector()
        with self._group(a, b).get_core_allocation(collector, collective_ranks=2, lnc_config=2):
            pass
        rollbacks = [
            c.args[1:]
            for c in collector.record_metric.call_args_list
            if c.args[0] == MetricName.GROUP_COMMIT_ROLLBACK_COUNT
        ]
        assert rollbacks == [(1.0, "Count")] * 2
        rollback_releases = a._release_cores.call_args_list[:-1]
        assert len(rollback_releases) == 2
        assert all(c.kwargs == {"end_attempt": False} for c in rollback_releases)
        # The release after the body ends the attempt.
        assert a._release_cores.call_args_list[-1].kwargs.get("end_attempt", True) is True
        assert b._release_cores.call_args.kwargs.get("end_attempt", True) is True

    def test_retryable_error_after_partial_commit_records_one_rollback(self, group_clock) -> None:
        events = group_clock
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[LockAcquisitionError("ssh blip"), _allocated([2, 3])])
        collector = _cm_collector()
        with self._group(a, b).get_core_allocation(collector, collective_ranks=2, lnc_config=2):
            pass
        rollbacks = [
            c for c in collector.record_metric.call_args_list if c.args[0] == MetricName.GROUP_COMMIT_ROLLBACK_COUNT
        ]
        assert len(rollbacks) == 1
        assert a._release_cores.call_args_list[0].kwargs == {"end_attempt": False}

    def test_abandon_after_partial_commit_ends_the_attempt(self, group_clock) -> None:
        events = group_clock
        error = LockAcquisitionError("corrupt lock state")
        error.retryable = False
        a = _alloc_member("a", events, wait=[_ASSEMBLED], commit=[_allocated([0, 1])])
        b = _alloc_member("b", events, wait=[_ASSEMBLED], commit=[error])
        with pytest.raises(LockAcquisitionError):
            with self._group(a, b).get_core_allocation(_cm_collector(), collective_ranks=2, lnc_config=2):
                raise pytest.fail.Exception("body must not run")
        assert a._release_cores.call_args.kwargs.get("end_attempt", True) is True

    def test_wait_phase_polls_members_concurrently(self, group_clock) -> None:
        events = group_clock
        members = [
            _alloc_member(alias, events, wait=[_ASSEMBLED], commit=[_allocated([2 * i, 2 * i + 1])])
            for i, alias in enumerate(("a", "b", "c"))
        ]
        barrier = threading.Barrier(len(members), timeout=5)
        for member in members:
            scripted = member._poll_queue.side_effect

            def _poll(*args, _scripted=scripted, **kwargs):
                if not kwargs["ready"]:
                    # Sequential polls would block here and break the barrier.
                    barrier.wait()
                return _scripted(*args, **kwargs)

            member._poll_queue.side_effect = _poll
        with self._group(*members).get_core_allocation(_cm_collector(), collective_ranks=3, lnc_config=2) as allocation:
            assert [m.host_id for m in allocation.member_allocations] == ["a", "b", "c"]
        assert [e[1] for e in _polls(events, ready=True)] == ["a", "b", "c"]


from typing import cast  # noqa: E402

from test.utils.exceptions import RemoteExecutionException  # noqa: E402


def _neuron_ls_stdout(num_devices: int = 8, ultraserver: dict | None = None) -> str:
    """neuron-ls JSON for ``num_devices`` 4-core lnc2 devices (8 physical cores each). With
    ``ultraserver`` it is the instance-level format carrying that deployment, else the legacy
    flat device list."""
    base = {k: v for k, v in json.loads(SAMPLE_NEURON_LS_OUTPUT)[0].items() if k != "instance_type"}
    devices = [{**base, "neuron_device": i, "nc_count": 4, "neuroncore_ids": [0, 1, 2, 3]} for i in range(num_devices)]
    if ultraserver is None:
        return json.dumps([{**device, "instance_type": "trn2.48xlarge"} for device in devices])
    return json.dumps(
        {"instance_info": {"instance_type": "trn2.48xlarge", "ultraserver_info": ultraserver}, "devices": devices}
    )


def _neuron_ls_host(tmp_path, alias: str, stdout: str | None = None, error: Exception | None = None) -> SshHost:
    """A real SshHost whose only remote call, the neuron-ls RPC, is faked."""
    host = _make_ssh_host(tmp_path, ssh_alias=alias)
    host._total_physical_cores = None  # computed from the faked neuron-ls report
    rpc = (
        MagicMock(side_effect=error)
        if error is not None
        else MagicMock(return_value=MagicMock(failed=False, stdout=stdout))
    )
    host.__run_with_retry__ = rpc
    return host


def _rpc(host: SshHost) -> MagicMock:
    """The faked neuron-ls RPC installed by ``_neuron_ls_host``."""
    return cast(MagicMock, host.__run_with_retry__)


def _store_rows(manager: HostManager) -> list[HostRecord]:
    state = manager.state_store.read()
    assert state is not None
    return state.hosts


def _deployment(rank: int, size: int = 4, deployment_id: str = "dep-1") -> dict:
    return {"ultraserver_id": deployment_id, "ultraserver_size": size, "ultraserver_instance_index": rank}


class TestSshHostNeuronLsReport:
    """One cached neuron-ls report serves devices, core count and probed identity alike."""

    def test_structured_report_is_fetched_once_and_carries_the_deployment(self, tmp_path):
        host = _neuron_ls_host(tmp_path, "h1", _neuron_ls_stdout(ultraserver=_deployment(rank=2)))
        assert len(host.get_neuron_device_info()) == 8
        identity = host.probe_identity(Platforms.TRN2)
        assert identity == ResolvedHost(
            ssh_host="h1",
            host_type=Platforms.TRN2,
            num_physical_cores=64,
            logical_host_id="dep-1",
            logical_host_rank=2,
            logical_host_size=4,
        )
        assert host.get_total_physical_cores() == 64
        assert len(host.get_neuron_device_info()) == 8
        _rpc(host).assert_called_once()

    def test_legacy_flat_report_probes_as_single_node(self, tmp_path):
        host = _neuron_ls_host(tmp_path, "h1", SAMPLE_NEURON_LS_OUTPUT)
        assert [d.neuron_device for d in host.get_neuron_device_info()] == [0]
        assert host.probe_identity(Platforms.TRN2) == ResolvedHost("h1", Platforms.TRN2, num_physical_cores=4)
        _rpc(host).assert_called_once()

    @pytest.mark.parametrize("ultraserver", [None, _deployment(rank=0)])
    def test_instance_type_is_reported_in_either_format(self, tmp_path, ultraserver):
        host = _neuron_ls_host(tmp_path, "h1", _neuron_ls_stdout(ultraserver=ultraserver))
        assert host.get_instance_type() == "trn2.48xlarge"

    def test_device_list_is_a_copy_of_the_cache(self, tmp_path):
        host = _neuron_ls_host(tmp_path, "h1", SAMPLE_NEURON_LS_OUTPUT)
        host.get_neuron_device_info().clear()
        assert len(host.get_neuron_device_info()) == 1

    def test_failed_rpc_is_not_cached(self, tmp_path):
        host = _neuron_ls_host(tmp_path, "h1")
        host.__run_with_retry__ = MagicMock(
            side_effect=[MagicMock(failed=True, stdout=""), MagicMock(failed=False, stdout=SAMPLE_NEURON_LS_OUTPUT)]
        )
        with pytest.raises(RemoteExecutionException):
            host.get_neuron_device_info()
        assert host.probe_identity(Platforms.TRN2).num_physical_cores == 4
        assert _rpc(host).call_count == 2

    def test_group_has_no_single_identity(self, tmp_path):
        group = SshHostGroup((_neuron_ls_host(tmp_path, "a"), _neuron_ls_host(tmp_path, "b")))
        with pytest.raises(NotImplementedError):
            group.probe_identity(Platforms.TRN2)


class TestPlainTargetProbe:
    """HostManager.probe_and_record_capacity on plain SSH targets persists what neuron-ls reports."""

    @staticmethod
    def _manager(tmp_path, rows: list[HostRecord], hosts: list[SshHost]) -> HostManager:
        store = HostStateStore(str(tmp_path / "state"))
        store._update(lambda s: HostState(hosts=list(rows)))
        manager = HostManager(
            state_store=store,
            neuron_installation_path="/opt/aws/neuron/bin",
            ssh_config_path="~/.ssh/config",
            testrun_uid="test-run",
            needs_local_host=False,
        )
        # Pre-built fake-SSH hosts so _get_host never opens a connection.
        manager.target_hosts.update({host.get_host_id(): host for host in hosts})
        return manager

    @staticmethod
    def _rows(*aliases: str) -> list[HostRecord]:
        return [HostRecord(resolved=ResolvedHost(alias, Platforms.TRN2)) for alias in aliases]

    def test_four_member_deployment_is_claimed_as_one_group(self, tmp_path):
        # Aliases listed out of rank order: membership and order come from neuron-ls, not names.
        ranks = {"node-c": 2, "node-a": 0, "node-d": 3, "node-b": 1}
        hosts = [
            _neuron_ls_host(tmp_path, alias, _neuron_ls_stdout(ultraserver=_deployment(rank)))
            for alias, rank in ranks.items()
        ]
        manager = self._manager(tmp_path, self._rows(*ranks), hosts)

        manager.probe_and_record_capacity(list(ranks))

        unit = manager._claim_unit(Platforms.TRN2, 256)
        assert isinstance(unit, HostGroup)
        assert unit.member_ids == ("node-a", "node-b", "node-c", "node-d")
        target = manager._get_target(unit)
        assert isinstance(target, SshHostGroup)
        assert [member.get_host_id() for member in target.hosts] == ["node-a", "node-b", "node-c", "node-d"]
        assert all(member is manager.target_hosts[member.get_host_id()] for member in target.hosts)
        for host in hosts:
            _rpc(host).assert_called_once()

    def test_failed_probe_keeps_known_state_and_others_are_recorded(self, tmp_path):
        known = HostRecord(
            resolved=ResolvedHost("bad", Platforms.TRN2, 64, "dep-1", 1, 2), work_queue_depth=3, available=True
        )
        rows = [known, *self._rows("good")]
        hosts = [
            _neuron_ls_host(tmp_path, "bad", error=OSError("unreachable")),
            _neuron_ls_host(tmp_path, "good", _neuron_ls_stdout(ultraserver=_deployment(0, size=2))),
        ]
        manager = self._manager(tmp_path, rows, hosts)

        manager.probe_and_record_capacity(["bad", "good"])

        by_alias = {rec.resolved.ssh_host: rec for rec in _store_rows(manager)}
        assert by_alias["bad"] == known
        assert by_alias["good"].resolved == ResolvedHost("good", Platforms.TRN2, 64, "dep-1", 0, 2)

    def test_legacy_report_keeps_a_known_deployment(self, tmp_path):
        known = HostRecord(resolved=ResolvedHost("h1", Platforms.TRN2, 32, "dep-1", 0, 2), work_queue_depth=5)
        manager = self._manager(tmp_path, [known], [_neuron_ls_host(tmp_path, "h1", _neuron_ls_stdout())])

        manager.probe_and_record_capacity(["h1"])

        rec = _store_rows(manager)[0]
        assert rec.resolved == ResolvedHost("h1", Platforms.TRN2, 64, "dep-1", 0, 2)
        assert (rec.available, rec.work_queue_depth) == (True, 5)

    def test_single_node_hosts_stay_single(self, tmp_path):
        hosts = [_neuron_ls_host(tmp_path, alias, _neuron_ls_stdout()) for alias in ("h1", "h2")]
        manager = self._manager(tmp_path, self._rows("h1", "h2"), hosts)

        manager.probe_and_record_capacity(["h1", "h2"])

        assert [rec.resolved for rec in _store_rows(manager)] == [
            ResolvedHost("h1", Platforms.TRN2, 64),
            ResolvedHost("h2", Platforms.TRN2, 64),
        ]

    def test_alias_without_a_row_is_neither_probed_nor_added(self, tmp_path):
        host = _neuron_ls_host(tmp_path, "h1", _neuron_ls_stdout())
        manager = self._manager(tmp_path, self._rows("h1"), [host])

        manager.probe_and_record_capacity(["h1", "ghost"])

        assert "ghost" not in manager.target_hosts
        assert [rec.resolved.ssh_host for rec in _store_rows(manager)] == ["h1"]
