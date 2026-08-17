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
"""
Core lock manager for neuron core allocation.

Provides orchestration of core locking operations for parallel test execution
on shared Neuron hardware, with metrics collection, version checking, and
contention tracking. Delegates low-level SSH lock operations to core_lock_client.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum

from . import core_lock_client as lock_client
from .metrics_collector import IMetricsCollector, MetricName
from .scripts.remote_lock_scripts import LockStatus

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """Manager-level outcome of an allocation attempt.

    Deliberately distinct from the wire ``remote_lock_scripts.LockStatus`` so the
    low-level protocol types do not leak above the manager boundary; callers
    branch only on this manager-level status.
    """

    ALLOCATED = "ALLOCATED"
    QUEUED = "QUEUED"
    DRAINING = "DRAINING"


@dataclass
class AllocationOutcome:
    """Manager-level result of ``CoreLockManager.acquire``.

    Mapped from the wire ``LockResult`` at the client boundary so callers branch
    on ``status`` without importing the remote protocol types.

    Attributes:
        status: Manager-level allocation status.
        logical_cores: Logical core IDs (set only when ALLOCATED).
        physical_cores: Physical core IDs locked (set only when ALLOCATED).
        position: 0-based FIFO queue position (set only when QUEUED/DRAINING).
        worst_case_eta: Worst-case wait estimate in relative seconds from now
            (QUEUED/DRAINING).
    """

    status: AllocationStatus
    logical_cores: list[int] | None = None
    physical_cores: list[int] | None = None
    position: int | None = None
    worst_case_eta: int | None = None


@dataclass
class CoreAllocation:
    """Represents an allocation of logical cores on a host.

    Logical cores are what the runtime sees via NEURON_RT_VISIBLE_CORES.
    The underlying locking mechanism uses physical cores to prevent conflicts
    between LNC1 and LNC2 tests running in parallel.

    Attributes:
        host_id: Identifier for the host where cores are allocated
        logical_core_ids: List of logical core IDs for NEURON_RT_VISIBLE_CORES
        lnc_config: LNC configuration (1 or 2) - physical cores per logical core
    """

    host_id: str
    logical_core_ids: list[int]
    lnc_config: int

    def get_core_list_str(self) -> str:
        """Get comma-separated logical core IDs for NEURON_RT_VISIBLE_CORES."""
        return ",".join(str(core_id) for core_id in self.logical_core_ids)


class LockVersionError(Exception):
    """
    Raised when the host requires a newer locking protocol version than the client supports.

    This is a retryable error - the caller can try a different host.
    """

    def __init__(self, required_version: int, current_version: int):
        self.required_version = required_version
        self.current_version = current_version
        self.retryable = True  # Can retry on a different host
        super().__init__(
            f"Host requires locking protocol v{required_version}, "
            f"but client only supports up to v{current_version}. "
            "Please update your client."
        )


class LockAcquisitionError(Exception):
    """
    Raised when lock acquisition fails unexpectedly.

    This is a retryable error - the caller can try again or try a different host.
    """

    def __init__(self, message: str):
        self.retryable = True  # Can retry on a different host
        super().__init__(message)


class InsufficientCoreCountError(LockAcquisitionError):
    """
    Raised when a host does not have enough physical cores to satisfy the request.

    This is a non-retryable error on the same host - the core count is a fixed property.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = False


def check_lock_version(host_locking_version: int) -> None:
    """
    Check if the host's locking protocol version is compatible with this client.

    Args:
        host_locking_version: The host's required minimum client locking version

    Raises:
        LockVersionError: If the host requires a newer locking protocol than this client supports
    """
    if host_locking_version > lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION:
        raise LockVersionError(
            required_version=host_locking_version,
            current_version=lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION,
        )


def calculate_total_needed_physical_cores(collectives_ranks: int, lnc_config: int) -> int:
    return collectives_ranks * lnc_config


class CoreLockManager:
    """
    Manages neuron core allocation using flock + JSON state file.

    Orchestrates core locking operations for parallel test execution on shared
    Neuron hardware. Adds metrics collection, version checking, and contention
    tracking on top of the low-level lock client.
    """

    def __init__(
        self,
        host: str,
        total_physical_cores: int,
        collector: IMetricsCollector,
        executor=None,
        host_locking_version: int | None = None,
    ):
        """
        Initialize the core lock manager.

        Args:
            host: Remote host identifier (for logging)
            total_physical_cores: Total number of physical neuron cores on the host
            collector: Metrics collector for the current test
            executor: Optional RemoteExecutor for low-latency remote calls
            host_locking_version: Pre-resolved locking version (skips remote query)
        """
        self.total_physical_cores = total_physical_cores
        self.host = host
        self._collector = collector
        self._executor = executor
        self._entry_id = uuid.uuid4().hex
        self._caller_id = f"{self._entry_id}:{getattr(collector, 'test_name', None) or 'unknown'}"
        self._host_locking_version = host_locking_version
        self._no_cores_count = 0
        # Wall-clock span accounting: stamp the first observation of each waiting
        # state and compute the span (terminal - first) at metric-emit time so the
        # inter-poll sleeps between acquire() calls are included, not just each
        # poll's RPC duration.
        self._first_contention_ts: float | None = None
        self._current_expiry: int | None = None
        # Queue-fairness observability state. Tracked manager-locally and emitted
        # at the point of measurement (the commit / contention-metrics call),
        # never propagated up the stack.
        self._enqueue_ts: float | None = None
        self._first_queue_position: int | None = None
        self._first_queue_eta: int | None = None
        self._last_position: int | None = None
        self._bump_count = 0
        self._bump_warned = False
        self._reenqueue_count = 0
        self._first_drain_ts: float | None = None
        # Guards the queue-fairness commit metrics so they are emitted exactly
        # once per attempt regardless of whether the attempt ends in an
        # ALLOCATED commit or an abandon flush.
        self._commit_metrics_recorded = False

    @property
    def entry_id(self) -> str:
        """Get the stable per-allocation-attempt UUID for this manager."""
        return self._entry_id

    @property
    def collector(self) -> IMetricsCollector:
        """Get the metrics collector that owns this manager's allocation attempt.

        Exposed read-only so the host layer can bind the cached manager's
        validity to its owning collector (the per-test/per-attempt owner)
        without reaching into private state.
        """
        return self._collector

    @property
    def host_locking_version(self) -> int:
        """Get the locking protocol version for this host."""
        if self._host_locking_version is None:
            raise LockAcquisitionError(f"[{self.host}] Lock manager not initialized — host_locking_version not set")
        return self._host_locking_version

    @staticmethod
    def _physical_to_logical_cores(physical_core_ids: list[int], lnc_config: int) -> list[int]:
        """
        Convert physical core IDs to logical core IDs based on LNC configuration.

        Physical cores are grouped by lnc_config to form logical cores.
        For LNC1: logical_core = physical_core (1:1 mapping)
        For LNC2: logical_core = physical_core // 2 (pairs of physical cores)

        Args:
            physical_core_ids: List of physical core IDs
            lnc_config: LNC configuration (1 or 2)

        Returns:
            List of unique logical core IDs

        Example:
            LNC2: [0, 1, 2, 3] -> [0, 1] (cores 0,1 map to logical 0; cores 2,3 map to logical 1)
            LNC1: [0, 1, 2, 3] -> [0, 1, 2, 3] (1:1 mapping)
        """
        if not physical_core_ids:
            return []

        first_physical = physical_core_ids[0]
        first_logical = first_physical // lnc_config
        num_logical = len(physical_core_ids) // lnc_config
        # Verify physical cores are contiguous and aligned
        expected = list(range(first_logical * lnc_config, (first_logical + num_logical) * lnc_config))
        assert physical_core_ids == expected, (
            f"Physical cores must be contiguous and aligned (lnc_config={lnc_config}): expected {expected}, got {physical_core_ids}"
        )
        return list(range(first_logical, first_logical + num_logical))

    def acquire(
        self,
        num_logical_cores: int,
        lnc_config: int,
        timeout_seconds: int = lock_client.DEFAULT_LOCK_TIMEOUT_SECONDS,
        ready: bool = True,
    ) -> AllocationOutcome:
        """Acquire logical cores via one FIFO-queue ``poll`` call.

        Performs a single ``lock_client.poll`` against the host's FIFO core-
        allocation queue and maps the wire ``LockResult`` to a manager-level
        ``AllocationOutcome`` at this boundary. Physical cores are
        locked to prevent conflicts between LNC1 and LNC2 tests; logical cores
        are for NEURON_RT_VISIBLE_CORES.

        Args:
            num_logical_cores: Number of logical cores to allocate
            lnc_config: LNC configuration (1 or 2) - physical cores per logical core
            timeout_seconds: How long the lock should be held before auto-expiring
            ready: Whether the caller is ready to commit. A ``ready=False`` poll
                enqueues/refreshes a queue slot but never grabs cores.

        Returns:
            AllocationOutcome with status:
            - ALLOCATED: cores granted (logical_cores + physical_cores set)
            - QUEUED: enqueued/waiting (position + worst_case_eta set)
            - DRAINING: host draining, grants withheld (position + eta set)

        Raises:
            LockAcquisitionError: If an unexpected error occurs during the poll
            InsufficientCoreCountError: If the host cannot satisfy the request
        """
        num_physical_cores = calculate_total_needed_physical_cores(num_logical_cores, lnc_config)
        assert num_physical_cores % lnc_config == 0, (
            f"num_physical_cores ({num_physical_cores}) must be a multiple of lnc_config ({lnc_config})"
        )

        if num_physical_cores > self.total_physical_cores:
            raise InsufficientCoreCountError(
                f"[{self.host}] Requested {num_logical_cores} logical cores (lnc{lnc_config} = "
                f"{num_physical_cores} physical) but host only has {self.total_physical_cores} physical cores"
            )

        attempt_start = time.time()

        try:
            with self._collector.timer(MetricName.CORE_LOCK_ACQUIRE_TIME):
                lock_result = lock_client.poll(
                    self._executor,
                    self.total_physical_cores,
                    num_physical_cores,
                    timeout_seconds,
                    self.host_locking_version,
                    self._entry_id,
                    ready,
                    caller_id=self._caller_id,
                )
        except Exception as e:
            raise LockAcquisitionError(str(e)) from e

        if lock_result.status == LockStatus.ALLOCATED:
            if lock_result.cores is None:
                raise LockAcquisitionError(f"[{self.host}] ALLOCATED status but no cores returned")
            allocated_physical_cores = lock_result.cores
            self._current_expiry = lock_result.expiry
            logical_cores = self._physical_to_logical_cores(allocated_physical_cores, lnc_config)
            self._record_commit_metrics()
            logging.info(
                f"[{self.host}] Acquired logical cores {logical_cores} "
                f"(physical: {allocated_physical_cores}, expires in {timeout_seconds}s)"
            )
            return AllocationOutcome(
                status=AllocationStatus.ALLOCATED,
                logical_cores=logical_cores,
                physical_cores=allocated_physical_cores,
            )

        if lock_result.status in (LockStatus.DRAINING, LockStatus.IN_QUEUE):
            draining = lock_result.status == LockStatus.DRAINING
            if draining:
                logging.info(f"[{self.host}] System is draining, staying in queue")
            self._no_cores_count += 1
            if self._first_contention_ts is None:
                self._first_contention_ts = attempt_start
            if draining and self._first_drain_ts is None:
                self._first_drain_ts = attempt_start
            self._note_enqueued(
                attempt_start,
                lock_result.position,
                lock_result.worst_case_eta,
                lock_result.bumped,
                lock_result.re_enqueued,
            )
            return AllocationOutcome(
                status=AllocationStatus.DRAINING if draining else AllocationStatus.QUEUED,
                position=lock_result.position,
                worst_case_eta=lock_result.worst_case_eta,
            )

        if lock_result.status == LockStatus.ERROR:
            raise LockAcquisitionError(f"[{self.host}] Lock helper error: {lock_result.message}")

        # Unknown status - treat as error
        raise LockAcquisitionError(f"[{self.host}] Unexpected lock status: {lock_result.status}")

    def probe(
        self,
        num_logical_cores: int,
        lnc_config: int,
        timeout_seconds: int = lock_client.DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> AllocationOutcome:
        """Read-only worst-case ETA peek for a not-yet-queued caller.

        Maps the wire ``LockResult`` to an ``AllocationOutcome``. Joins no queue
        and emits no event, so it touches none of the contention/position
        counters ``acquire`` maintains (metric-neutral).

        Args:
            num_logical_cores: Number of logical cores the caller would request
            lnc_config: LNC configuration (1 or 2) - physical cores per logical core
            timeout_seconds: Hold-window passed to the probe so the quoted ETA
                matches what a subsequent ``acquire`` would observe

        Returns:
            AllocationOutcome with status QUEUED (IN_QUEUE) or DRAINING, carrying
            ``worst_case_eta``.

        Raises:
            InsufficientCoreCountError: If the host cannot satisfy the request
            LockAcquisitionError: If an unexpected error occurs during the probe
        """
        num_physical_cores = calculate_total_needed_physical_cores(num_logical_cores, lnc_config)
        if num_physical_cores > self.total_physical_cores:
            raise InsufficientCoreCountError(
                f"[{self.host}] Requested {num_logical_cores} logical cores (lnc{lnc_config} = "
                f"{num_physical_cores} physical) but host only has {self.total_physical_cores} physical cores"
            )

        try:
            lock_result = lock_client.probe(
                self._executor,
                self.total_physical_cores,
                num_physical_cores,
                timeout_seconds,
                self.host_locking_version,
            )
        except Exception as e:
            raise LockAcquisitionError(str(e)) from e

        if lock_result.status in (LockStatus.DRAINING, LockStatus.IN_QUEUE):
            draining = lock_result.status == LockStatus.DRAINING
            return AllocationOutcome(
                status=AllocationStatus.DRAINING if draining else AllocationStatus.QUEUED,
                worst_case_eta=lock_result.worst_case_eta,
            )

        if lock_result.status == LockStatus.ERROR:
            raise LockAcquisitionError(f"[{self.host}] Lock helper error: {lock_result.message}")

        # Unknown status - treat as error
        raise LockAcquisitionError(f"[{self.host}] Unexpected lock status: {lock_result.status}")

    def dequeue(self) -> None:
        """Gracefully remove this manager's entry from the host FIFO queue.

        Used when abandoning a host (patience rotation or pre-acquire error).
        Removing a non-existent entry is a server-side no-op.
        """
        logging.info(f"[{self.host}] Dequeuing entry {self._entry_id}")
        lock_result = lock_client.dequeue(
            self._executor,
            self.total_physical_cores,
            self.host_locking_version,
            self._entry_id,
            caller_id=self._caller_id,
        )
        if lock_result.status != LockStatus.RELEASED:
            logging.warning(f"[{self.host}] Dequeue may have failed: {lock_result.status}, {lock_result.message}")

    def release(self, core_ids: list[int]) -> None:
        """
        Release previously acquired cores.

        Args:
            core_ids: List of physical core IDs to release
        """
        if not core_ids:
            return

        logging.info(f"[{self.host}] Releasing cores {core_ids}")

        lock_result = lock_client.release(
            self._executor,
            core_ids,
            self.host_locking_version,
            caller_id=self._caller_id,
            expected_expiry=self._current_expiry,
        )

        if lock_result.status == LockStatus.RELEASED:
            logging.info(f"[{self.host}] Released cores {core_ids}")
            self._current_expiry = None
        else:
            logging.warning(f"[{self.host}] Release may have failed: {lock_result.status}, {lock_result.message}")

    def drain(self, timeout_seconds: int) -> int:
        """
        Enable drain mode to block new acquisitions.

        Used for graceful shutdown - blocks new test acquisitions while
        allowing existing tests to complete.

        Args:
            timeout_seconds: How long drain should last before auto-expiring

        Returns:
            Max lock expiry epoch seconds (0 if no active locks)
        """
        logging.info(f"[{self.host}] Enabling drain mode for {timeout_seconds}s")

        lock_result = lock_client.drain(self._executor, timeout_seconds, self.host_locking_version)

        if lock_result.status == LockStatus.DRAINED:
            max_lock_expiry = lock_result.max_lock_expiry or 0
            logging.info(f"[{self.host}] Drain mode enabled, max_lock_expiry={max_lock_expiry}")
            return max_lock_expiry
        else:
            logging.warning(f"[{self.host}] Drain may have failed: {lock_result.status}, {lock_result.message}")
            return 0

    def disable_drain(self) -> None:
        """Disable drain mode to allow new test acquisitions."""
        logging.info(f"[{self.host}] Disabling drain mode")

        lock_result = lock_client.undrain(self._executor, self.host_locking_version)

        if lock_result.status == LockStatus.UNDRAINED:
            logging.info(f"[{self.host}] Drain disabled")
        else:
            logging.warning(f"[{self.host}] Undrain may have failed: {lock_result.status}, {lock_result.message}")

    def _note_enqueued(
        self,
        enqueue_ts: float,
        position: int | None,
        worst_case_eta: int | None,
        bumped: bool,
        re_enqueued: bool = False,
    ) -> None:
        """Record queue-fairness state for one IN_QUEUE/DRAINING poll.

        Stamps the first enqueue time for ``QueueWaitTime``, captures the
        first-IN_QUEUE position/ETA, and counts
        bumps. Bump detection is driven by the explicit ``bumped`` signal the
        server sets when reconcile moves THIS entry to the tail. This catches a
        sole-entry bump (the head bumps to an empty tail and stays at position 0,
        a 0->0 move a position delta cannot see) and avoids miscounting a
        prune->re-enqueue position jump as a bump. Three bumps for one entry
        signal a possible sole-entry bump livelock, so warn once at >=3.

        ``re_enqueued`` is a distinct, mutually-exclusive signal: the server sets
        it (and never ``bumped``) when THIS entry was pruned this cycle then
        re-appended at the tail -- a silent FIFO position loss tracked separately
        so the soft-join upload-grace cost is observable apart from bumps.

        ``position``/``worst_case_eta`` are still tracked for position metrics.
        """
        if self._enqueue_ts is None:
            self._enqueue_ts = enqueue_ts
        if bumped:
            self._bump_count += 1
            if self._bump_count >= 3 and not self._bump_warned:
                self._bump_warned = True
                logging.warning(
                    f"[{self.host}] entry {self._entry_id} bumped {self._bump_count} times; "
                    f"possible sole-entry bump livelock"
                )
        if re_enqueued:
            self._reenqueue_count += 1
        if position is None:
            return
        if self._first_queue_position is None:
            self._first_queue_position = position
            self._first_queue_eta = worst_case_eta
        self._last_position = position

    def _record_commit_metrics(self) -> None:
        """Emit queue-fairness metrics once per attempt.

        Called both at the ALLOCATED commit and on the abandon flush; the
        ``_commit_metrics_recorded`` guard makes it idempotent so the two paths
        (which are mutually exclusive within an attempt) never double-emit.

        QueueDepthAtEnqueue is intentionally omitted: the wire ``LockResult``
        carries only ``position``, not absolute queue depth, so
        ``QueuePositionAtEnqueue`` is recorded instead as a truthful lower bound
        on depth.
        """
        if self._commit_metrics_recorded:
            return
        self._commit_metrics_recorded = True
        if self._enqueue_ts is not None:
            actual_wait = time.time() - self._enqueue_ts
            self._collector.record_timer(MetricName.CORE_LOCK_QUEUE_WAIT_TIME, actual_wait)
            if self._first_queue_eta is not None:
                # How much longer the caller actually waited than the
                # worst-case ETA it was first quoted at enqueue. Clamped to
                # zero so it is never negative: a positive value captures
                # scheduling inefficiency (the optimistic ETA under-estimated).
                overrun = max(0.0, actual_wait - self._first_queue_eta)
                self._collector.record_timer(MetricName.CORE_LOCK_ETA_OVERRUN_TIME, overrun)
        if self._first_queue_position is not None:
            self._collector.record_metric(
                MetricName.CORE_LOCK_QUEUE_POSITION_AT_ENQUEUE, self._first_queue_position, "Count"
            )
        if self._first_queue_eta is not None:
            self._collector.record_metric(MetricName.CORE_LOCK_ETA_AT_ENQUEUE, self._first_queue_eta, "Seconds")
        self._collector.record_metric(MetricName.CORE_LOCK_BUMP_COUNT, self._bump_count, "Count")
        self._collector.record_metric(MetricName.CORE_LOCK_REENQUEUE_COUNT, self._reenqueue_count, "Count")

    def record_contention_metrics(self) -> None:
        """Flush this attempt's queue-fairness and contention metrics.

        Safe to call on either terminal path: the ALLOCATED success path and
        every abandon path (patience rotation, retryable give-up, wall-clock
        timeout). It first flushes the queue-fairness commit metrics (idempotent
        via ``_record_commit_metrics``, so the success path's earlier commit-time
        emission is not double-counted), then emits the per-attempt NO_CORES count
        and the contention / drain-wait spans measured as wall-clock from the
        first contended / draining poll to this terminal moment (inter-poll
        sleeps included).
        """
        self._record_commit_metrics()
        # Compute the wait spans as wall-clock from first observation to this
        # terminal moment, so the inter-poll sleeps that elapse between acquire()
        # calls in the host loop are included (not just each poll's RPC duration).
        terminal = time.time()
        contention_wait = terminal - self._first_contention_ts if self._first_contention_ts is not None else 0.0
        drain_wait = terminal - self._first_drain_ts if self._first_drain_ts is not None else 0.0
        self._collector.record_metric(MetricName.CORE_LOCK_NO_CORES_COUNT, self._no_cores_count, "Count")
        self._collector.record_timer(MetricName.CORE_LOCK_CONTENTION_WAIT_TIME, contention_wait)
        self._collector.record_timer(MetricName.CORE_LOCK_DRAIN_WAIT_TIME, drain_wait)
        if self._no_cores_count > 0:
            logging.info(f"[{self.host}] Contention: {self._no_cores_count} rejections, {contention_wait:.2f}s wait")
