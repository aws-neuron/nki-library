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
from dataclasses import dataclass

import fabric2

from . import core_lock_client as lock_client
from .metrics_collector import IMetricsCollector, MetricName
from .scripts.remote_lock_scripts import LockStatus

logger = logging.getLogger(__name__)


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


def check_lock_version(conn: fabric2.Connection) -> None:
    """
    Check if the host's locking protocol version is compatible with this client.

    Args:
        conn: Active fabric2 Connection to the remote host

    Raises:
        LockVersionError: If the host requires a newer locking protocol
    """
    required_version = lock_client.get_host_locking_version(conn)

    if required_version > lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION:
        raise LockVersionError(
            required_version=required_version,
            current_version=lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION,
        )


class CoreLockManager:
    """
    Manages neuron core allocation using flock + JSON state file.

    Orchestrates core locking operations for parallel test execution on shared
    Neuron hardware. Adds metrics collection, version checking, and contention
    tracking on top of the low-level lock client.
    """

    def __init__(
        self,
        connection: fabric2.Connection,
        total_physical_cores: int,
        collector: IMetricsCollector,
    ):
        """
        Initialize the core lock manager.

        Args:
            connection: Fabric SSH connection to the remote host
            total_physical_cores: Total number of physical neuron cores on the host
            collector: Metrics collector for recording timing metrics
        """
        self.connection = connection
        self.total_physical_cores = total_physical_cores
        self.host = connection.host
        self._collector = collector
        self._initialized = False
        self._version_checked = False
        self._host_locking_version: int | None = None
        # Contention tracking
        self._no_cores_count = 0
        self._contention_wait_time = 0.0

    def initialize(self) -> None:
        """
        Initialize the remote lock directory and files.

        This is called automatically before any lock operations, but can be
        called explicitly for pre-flight setup. It:
        - Creates the lock directory on the remote host
        - Deploys the lock helper script to the remote host
        """
        if self._initialized:
            return

        with self._collector.timer(MetricName.CORE_LOCK_INIT_TIME):
            logging.info(f"[{self.host}] Initializing core lock manager")
            lock_client.initialize(self.connection)

            with self._collector.timer(MetricName.CORE_LOCK_DEPLOY_TIME):
                lock_client.deploy_lock_helpers(self.connection)

            self._initialized = True

    @property
    def host_locking_version(self) -> int:
        """Get the locking protocol version for this host."""
        if self._host_locking_version is None:
            with self._collector.timer(MetricName.CORE_LOCK_VERSION_CHECK_TIME):
                self._host_locking_version = lock_client.get_host_locking_version(self.connection)
        return self._host_locking_version

    def _check_lock_version(self) -> None:
        """
        Check lock version compatibility.

        Raises:
            LockVersionError: If the host requires a newer locking protocol
        """
        if self._version_checked:
            return

        required_version = self.host_locking_version

        if required_version > lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION:
            raise LockVersionError(
                required_version=required_version,
                current_version=lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION,
            )

        logging.info(
            f"[{self.host}] Using locking protocol v{required_version} "
            f"(client supports up to v{lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION})"
        )
        self._version_checked = True

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
        assert (
            physical_core_ids == expected
        ), f"Physical cores must be contiguous and aligned (lnc_config={lnc_config}): expected {expected}, got {physical_core_ids}"
        return list(range(first_logical, first_logical + num_logical))

    def acquire(
        self,
        num_logical_cores: int,
        lnc_config: int,
        timeout_seconds: int = lock_client.DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> tuple[list[int], list[int]] | None:
        """
        Acquire logical cores by locking physical cores.

        Locking is done on physical cores to prevent conflicts between LNC1 and LNC2 tests.
        For example, LNC2 logical core 0 maps to physical cores [0,1], while LNC1 logical
        cores 0-1 also map to physical cores [0,1]. By locking physical cores, we ensure
        these tests don't run simultaneously on the same hardware.

        Args:
            num_logical_cores: Number of logical cores to allocate
            lnc_config: LNC configuration (1 or 2) - physical cores per logical core
            timeout_seconds: How long the lock should be held before auto-expiring

        Returns:
            Tuple of (logical_core_ids, physical_core_ids), or None if:
            - System is draining (graceful shutdown in progress)
            - Not enough contiguous cores available

            Physical cores are locked via flock+JSON; logical cores are for NEURON_RT_VISIBLE_CORES.

        Raises:
            LockAcquisitionError: If an unexpected error occurs during acquisition

        Example:
            # Acquire 2 logical cores with LNC2 (needs 4 physical cores)
            result = manager.acquire(num_logical_cores=2, lnc_config=2, timeout_seconds=60)
            # Returns e.g. ([0, 1], [0, 1, 2, 3]) or ([2, 3], [4, 5, 6, 7])
        """
        self.initialize()

        # Check lock version before attempting to acquire
        self._check_lock_version()

        num_physical_cores = num_logical_cores * lnc_config
        assert (
            num_physical_cores % lnc_config == 0
        ), f"num_physical_cores ({num_physical_cores}) must be a multiple of lnc_config ({lnc_config})"

        attempt_start = time.time()

        try:
            with self._collector.timer(MetricName.CORE_LOCK_ACQUIRE_TIME):
                lock_result = lock_client.acquire(
                    self.connection,
                    self.total_physical_cores,
                    num_physical_cores,
                    timeout_seconds,
                    self.host_locking_version,
                )
        except RuntimeError as e:
            raise LockAcquisitionError(str(e))

        if lock_result.status == LockStatus.DRAINING:
            logging.info(f"[{self.host}] System is draining, rejecting acquisition")
            self._contention_wait_time += time.time() - attempt_start
            return None

        if lock_result.status == LockStatus.NO_CORES:
            logging.info(f"[{self.host}] No contiguous cores available")
            self._no_cores_count += 1
            self._contention_wait_time += time.time() - attempt_start
            return None

        if lock_result.status == LockStatus.ALLOCATED:
            if lock_result.cores is None:
                raise LockAcquisitionError(f"[{self.host}] ALLOCATED status but no cores returned")
            allocated_physical_cores = lock_result.cores
            logical_cores = self._physical_to_logical_cores(allocated_physical_cores, lnc_config)
            logging.info(
                f"[{self.host}] Acquired logical cores {logical_cores} "
                f"(physical: {allocated_physical_cores}, expires in {timeout_seconds}s)"
            )
            return logical_cores, allocated_physical_cores

        if lock_result.status == LockStatus.ERROR:
            raise LockAcquisitionError(f"[{self.host}] Lock helper error: {lock_result.message}")

        # Unknown status - treat as error
        raise LockAcquisitionError(f"[{self.host}] Unexpected lock status: {lock_result.status}")

    def release(self, core_ids: list[int]) -> None:
        """
        Release previously acquired cores.

        Args:
            core_ids: List of physical core IDs to release
        """
        if not core_ids:
            return

        self.initialize()

        logging.info(f"[{self.host}] Releasing cores {core_ids}")

        lock_result = lock_client.release(self.connection, core_ids, self.host_locking_version)

        if lock_result.status == LockStatus.RELEASED:
            logging.info(f"[{self.host}] Released cores {core_ids}")
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
        self.initialize()

        logging.info(f"[{self.host}] Enabling drain mode for {timeout_seconds}s")

        lock_result = lock_client.drain(self.connection, timeout_seconds, self.host_locking_version)

        if lock_result.status == LockStatus.DRAINED:
            max_lock_expiry = lock_result.max_lock_expiry or 0
            logging.info(f"[{self.host}] Drain mode enabled, max_lock_expiry={max_lock_expiry}")
            return max_lock_expiry
        else:
            logging.warning(f"[{self.host}] Drain may have failed: {lock_result.status}, {lock_result.message}")
            return 0

    def disable_drain(self) -> None:
        """Disable drain mode to allow new test acquisitions."""
        self.initialize()

        logging.info(f"[{self.host}] Disabling drain mode")

        lock_result = lock_client.undrain(self.connection, self.host_locking_version)

        if lock_result.status == LockStatus.UNDRAINED:
            logging.info(f"[{self.host}] Drain disabled")
        else:
            logging.warning(f"[{self.host}] Undrain may have failed: {lock_result.status}, {lock_result.message}")

    def record_contention_metrics(self) -> None:
        """Record contention metrics to the collector.

        Call this after successfully acquiring cores to record how many
        NO_CORES rejections occurred and how long was spent waiting.
        """
        self._collector.record_metric(MetricName.CORE_LOCK_NO_CORES_COUNT, self._no_cores_count, "Count")
        self._collector.record_timer(MetricName.CORE_LOCK_CONTENTION_WAIT_TIME, self._contention_wait_time)
        if self._no_cores_count > 0:
            logging.info(
                f"[{self.host}] Contention: {self._no_cores_count} rejections, {self._contention_wait_time:.2f}s wait"
            )
