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
import contextlib
import functools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Generator, final

from filelock import FileLock
from invoke.exceptions import CommandTimedOut
from paramiko import SSHException
from typing_extensions import override

from . import core_lock_client as lock_client
from .common_dataclasses import (
    INF_ARTIFACT_DIR_NAME,
    NeuronDeviceInfo,
    Platforms,
    ResolvedHost,
    TargetHost,
    resolve_probe_worker_count,
)
from .core_lock_client import POLL_JITTER_MAX, POLL_PERIOD, REMOTE_LOCKS_JSON, STALE_THRESHOLD
from .core_lock_manager import (
    AllocationStatus,
    CoreAllocation,
    CoreLockManager,
    LockAcquisitionError,
    LockVersionError,
    calculate_total_needed_physical_cores,
    check_lock_version,
)
from .core_reset_strategy import CoreResetStrategy
from .exceptions import (
    FleetEmptyError,
    HostsBusyError,
    InferenceException,
    LocalExecutionException,
    NoNeuronDevicesException,
    QueuePatienceRotation,
    RemoteExecutionException,
    TimeoutException,
    UnimplementedException,
)
from .host_communication import (
    SSH_INFERENCE_TIMEOUT_SECONDS,
    LocalCommunication,
    ParamikoCommunication,
    SshCommunication,
)
from .host_state import HostRecord, HostStateStore
from .metrics_collector import NOOP_METRICS_COLLECTOR, IMetricsCollector, MetricName
from .remote_executor import RemoteExecutor
from .s3_utils import S3ArtifactUploadConfig
from .scripts.remote_lock_scripts import LockState, find_contiguous_cores

# Constructed once at module load and reused as the shared default; CoreResetStrategy
# with exclusive_run=False has no file-backed state (_state_path/_lock are None) and is
# never mutated in place, so sharing this instance across callers is safe.
_DEFAULT_CORE_RESET_STRATEGY = CoreResetStrategy(exclusive_run=False)

# Caller-side patience for FIFO core allocation.
# If a host's worst-case ETA exceeds this, the caller dequeues and rotates to
# another host rather than camping. Capped well under the per-test pytest
# timeout so the caller can cycle through a few hosts before the test is killed.
DEFAULT_PATIENCE_SECONDS = 3 * lock_client.DEFAULT_LOCK_TIMEOUT_SECONDS  # = 180

# Wall-clock deadline for acquiring a host + core allocation. A busy FIFO queue
# is transient (not a host failure), so the caller keeps rotating across hosts
# until this deadline elapses rather than giving up after a fixed attempt count.
# ~2.5h, capped under the overall suite budget.
DEFAULT_ACQUISITION_DEADLINE_SECONDS = 9000
# Backoff between host rotations when the whole fleet is momentarily busy, so a
# fully-busy fleet re-cycles at a bounded rate instead of hot-spinning.
HOST_ROTATION_BACKOFF_SECONDS = 1

# Short poll cadence for a near-front waiter. 1 s is the empirical FCFS hand-off
# floor (bounded by the SSH round-trip), so polling this fast near the front
# closes the per-hand-off idle gap without spending extra load for no gain.
# Must stay below POLL_PERIOD (FAST_POLL_PERIOD < POLL_PERIOD).
FAST_POLL_PERIOD = 1

# Post-lock commands (e.g. profile post-processing) run without a core lock and
# may legitimately take much longer than a locked capture; bound them at 15 min.
POST_LOCK_TIMEOUT_SECONDS = 900


def select_poll_base(position: int | None, *, fast: int = FAST_POLL_PERIOD, slow: int = POLL_PERIOD) -> int:
    """Return the poll-loop sleep base for a queued caller.

    A caller at or one slot from the front of the queue (``position <= 1``)
    polls at the short ``fast`` cadence so it commits its reserved block
    within ~1 s of a hand-off instead of waiting a full slow poll cycle; any
    deeper position (or an unknown position) uses the normal ``slow`` cadence.
    The ``fast``/``slow`` arguments are injectable so callers can pass a
    customized slow period.
    """
    if position is not None and position <= 1:
        return fast
    return slow


def should_rotate(worst_case_eta: int | None, draining: bool, patience_seconds: int) -> bool:
    """Pure predicate: should a queued caller abandon this host and rotate away?

    Returns True only for a non-draining caller whose worst-case ETA exceeds
    ``patience_seconds``. Negative patience disables rotation i.e. would always return negative guaidance.
    A draining host never triggers rotation (ETA is meaningless mid-drain, so the caller stays and keeps probing),
    and a missing ETA is treated as "stay". Kept pure/host-free for unit testing.
    """
    return (not draining) and worst_case_eta is not None and worst_case_eta > patience_seconds and patience_seconds > 0


@functools.lru_cache(maxsize=1)
def _run_neuron_ls(neuron_installation_path: str) -> list[NeuronDeviceInfo] | None:
    """Run neuron-ls --json-output and return parsed JSON, or None on failure."""
    try:
        neuron_ls_path = (
            os.path.join(neuron_installation_path, "neuron-ls")
            if not neuron_installation_path.endswith("neuron-ls")
            else neuron_installation_path
        )
        if not os.path.isfile(neuron_ls_path):
            return None
        result = subprocess.run(
            [neuron_ls_path, "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data:
                return [NeuronDeviceInfo.from_dict(device) for device in data]
        return None
    except Exception:
        return None


def detect_local_neuron_devices(neuron_installation_path: str) -> bool:
    """Check if local Neuron devices are available via neuron-ls.

    Returns False gracefully if neuron-ls is not installed, times out, or finds no devices.
    """
    return _run_neuron_ls(neuron_installation_path) is not None


def detect_local_platform(neuron_installation_path: str) -> Platforms | None:
    """Detect the local Neuron platform type from neuron-ls instance_type.

    Returns the Platforms enum value, or None if detection fails.
    """
    data = _run_neuron_ls(neuron_installation_path)
    if not data:
        return None

    instance_type = data[0].instance_type
    # instance_type is e.g. "trn2.48xlarge" or "trn3pds.48xlarge" — extract "trn" + digits
    match = re.match(r"(trn\d+)", instance_type)
    if not match:
        logging.warning(f"Unknown platform from instance_type '{instance_type}', cannot auto-detect")
        return None
    try:
        return Platforms(match.group(1))
    except ValueError:
        logging.warning(f"Unknown platform from instance_type '{instance_type}', cannot auto-detect")
        return None


class Host(ABC):
    def __init__(self, core_reset_strategy: CoreResetStrategy) -> None:
        self._reset_strategy = core_reset_strategy

    def execute_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
        collective_ranks: int,
        lnc_config: int,
        do_copy_artifacts: bool = False,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None = None,
        post_lock_command: str | None = None,
    ) -> str | None:
        run_exception = None
        stdout = ""

        with self.get_core_allocation(
            collective_ranks=collective_ranks, lnc_config=lnc_config, collector=collector
        ) as core_allocation:
            with collector.timer(MetricName.CORE_LOCK_HOLD_TIME):
                unique_port = self._get_unique_collectives_port(core_allocation.logical_core_ids[0])
                neuron_env = self._build_neuron_env(
                    core_allocation,
                    lnc_config,
                    unique_port,
                    self._get_debug_output_dir(target_directory),
                )

                neuron_env.update(self._setup_neuron_tags())
                try:
                    stdout = self._run_command(command, target_directory, neuron_env, collector)
                except Exception as e:
                    run_exception = e
            # Record the outcome while still holding the core lock: releasing first
            # would let another worker read stale LNC state in the release-to-teardown gap.
            self._reset_strategy.mark_test_complete(core_allocation, lnc_config, failed=run_exception is not None)

        if not run_exception and post_lock_command:
            if self._should_run_post_lock():
                post_lock_stdout = self._run_post_lock_command(post_lock_command, target_directory, collector)
                stdout = stdout + "\n" + post_lock_stdout
            else:
                logging.error("Skipping post-lock commands: host is draining")

        artifact_path = None
        if do_copy_artifacts:
            try:
                artifact_path = self._collect_artifacts(target_directory, stdout, get_list_of_files_to_copy, collector)
            except Exception:
                if run_exception:
                    logging.warning("Failed to collect artifacts after command failure", exc_info=True)
                else:
                    raise

        if run_exception:
            raise run_exception

        return artifact_path

    @abstractmethod
    def get_total_physical_cores(self) -> int:
        raise UnimplementedException()

    @abstractmethod
    def _get_debug_output_dir(self, target_directory: str) -> str:
        raise UnimplementedException()

    @abstractmethod
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        """Run the command on the host and return stdout."""
        raise UnimplementedException()

    @abstractmethod
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        """Download/copy artifacts and return the local artifact directory path."""
        raise UnimplementedException()

    @abstractmethod
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ) -> contextlib.AbstractContextManager[Any, Any]:
        raise UnimplementedException()

    @abstractmethod
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> contextlib.AbstractContextManager[CoreAllocation, Any]:
        """
        Allocate logical cores for execution by locking physical cores.

        Prefers aligned allocations for better packing, with randomization to reduce
        contention when multiple workers allocate simultaneously.

        Args:
            collective_ranks: Number of logical cores to allocate
            lnc_config: LNC configuration (1 or 2) - determines physical cores per logical core
            timeout_seconds: Maximum time to wait for core allocation
            poll_period_seconds: Time between allocation attempts

        Returns:
            Context manager yielding CoreAllocation with allocated logical core IDs
        """
        raise UnimplementedException()

    @abstractmethod
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        """Run a command that does NOT require Neuron hardware, after core lock release."""
        raise UnimplementedException()

    def _should_run_post_lock(self) -> bool:
        """Check if post-lock commands should run. Override to check drain state."""
        return True

    def soft_join_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
    ) -> None:
        """Reserve a FIFO core-queue slot before/while artifacts upload.

        Default no-op. Only SshHost participates in the remote FIFO queue;
        LocalHost uses the file-lock path and must NOT enqueue.
        """
        return None

    @abstractmethod
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        raise UnimplementedException()

    @abstractmethod
    def get_host_id(self) -> str:
        raise UnimplementedException()

    @staticmethod
    def _get_unique_collectives_port(core_id: int) -> int:
        """Generate unique port for collectives coordination based on first allocated core.

        This ensures parallel tests don't conflict on the same port.
        E.g., test on cores [8, 9] gets port 61242, test on cores [16, 17] gets port 61250.
        """
        return 61234 + core_id

    def _build_neuron_env(
        self,
        core_allocation: CoreAllocation,
        lnc_config: int,
        unique_port: int,
        debug_output_dir: str,
    ) -> dict[str, str]:
        """Build the common Neuron runtime environment variables for execution."""
        return {
            "NEURON_RT_ENABLE_OCP": "1",
            "NEURON_RT_ENABLE_OCP_SATURATION": "1",
            "NEURON_RT_VISIBLE_CORES": core_allocation.get_core_list_str(),
            "NEURON_LOGICAL_NC_CONFIG": str(lnc_config),
            "NEURON_RT_ROOT_COMM_ID": f"localhost:{unique_port}",
            "NEURON_RT_DEBUG_OUTPUT_DIR": debug_output_dir,
            # Avoid core reset between tests (overriding per-test default).
            "NEURON_RT_RESET_CORES": "1"
            if self._reset_strategy.should_reset_cores(core_allocation, lnc_config)
            else "0",
        }

    @staticmethod
    def _setup_neuron_tags() -> dict[str, str]:
        """Build the NEURON_TAG_* dict that ships to the remote workload:
        every NEURON_TAG_* in the local env, plus a fresh per-invocation
        REQUEST_ID. Future tags are picked up automatically.
        """
        tags = {k: v for k, v in os.environ.items() if k.startswith("NEURON_TAG_")}
        tags["NEURON_TAG_REQUEST_ID"] = str(uuid.uuid4())
        return tags

    def __lock__(self, lock_file_path: str, timeout_seconds: int):
        return FileLock(f"{lock_file_path}.lock", timeout=timeout_seconds * 1000)


@final
class LocalHost(Host):
    def __init__(
        self,
        local_neuron_installation_path: str,
        host_id: str,
        core_allocation_dir: str,
        core_reset_strategy: CoreResetStrategy,
    ):
        super().__init__(core_reset_strategy)
        self._comm = LocalCommunication(host_id=host_id)
        self.neuron_ls_path: str = os.path.join(local_neuron_installation_path, "neuron-ls")
        self.host_id: str = host_id
        self.core_allocation_dir: str = core_allocation_dir
        self._core_state_path: str = os.path.join(core_allocation_dir, "local_core_locks.json")

    @override
    def get_host_id(self) -> str:
        return self.host_id

    @staticmethod
    def _check_no_remote_locks() -> None:
        """Raise if this machine has active remote SSH-based core locks (locks.json)."""
        if not os.path.isfile(REMOTE_LOCKS_JSON):
            return
        try:
            with open(REMOTE_LOCKS_JSON, "r") as f:
                data = json.load(f)
            state = LockState.from_dict(data, default_version=2)
            active = state.get_locked_cores(int(time.time()))
            if active:
                raise RuntimeError(
                    f"Active remote core locks found in {REMOTE_LOCKS_JSON} (cores: {active}). "
                    + "This machine appears to be in use as a shared fleet host via SSH. "
                    + "Local testing is not supported on shared fleet instances to avoid core allocation conflicts."
                )
        except (json.JSONDecodeError, OSError):
            pass

    @override
    def get_total_physical_cores(self) -> int:
        devices = _run_neuron_ls(self.neuron_ls_path)
        if devices is None:
            raise RuntimeError("Unable to detect local number of physical cores!")

        device_lnc_count: int = sum(len(d.neuroncore_ids) * d.logical_neuroncore_config for d in devices)
        return device_lnc_count

    @override
    def _get_debug_output_dir(self, target_directory: str) -> str:
        return os.path.join(target_directory, "debug_output")

    @override
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        env = os.environ.copy()
        env.update(neuron_env)

        full_command = f"set -o pipefail; cd {target_directory} && {command}"
        logging.info(
            f"Executing local command: {full_command} "
            f"with NEURON_RT_VISIBLE_CORES={env.get('NEURON_RT_VISIBLE_CORES')} "
            f"NEURON_LOGICAL_NC_CONFIG={env.get('NEURON_LOGICAL_NC_CONFIG')}"
        )

        with collector.timer(MetricName.INFERENCE_TIME):
            result = subprocess.run(
                ["bash", "-c", full_command],
                env=env,
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            raise LocalExecutionException(
                f"Unable to execute {command} in {target_directory}",
                result,
            )
        return result.stdout

    @override
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        full_command = f"set -o pipefail; cd {target_directory} && {command}"
        logging.info(f"Executing local post-lock command: {full_command}")

        result = subprocess.run(
            ["bash", "-c", full_command],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise LocalExecutionException(
                f"Post-lock command failed in {target_directory}",
                result,
            )
        return result.stdout

    @override
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        local_download_location = os.path.join(target_directory, INF_ARTIFACT_DIR_NAME)
        os.makedirs(local_download_location, exist_ok=True)

        if get_list_of_files_to_copy:
            files_to_copy = get_list_of_files_to_copy(stdout)
            for f in files_to_copy:
                src = os.path.join(target_directory, f)
                dst = os.path.join(local_download_location, f)
                if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)

        return local_download_location

    @override
    @contextlib.contextmanager
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ):
        yield

    @override
    @contextlib.contextmanager
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> Generator[CoreAllocation, None, None]:
        # Guard against running LocalHost on a machine also used as a remote SshHost target.
        # SshHost uses locks.json for core locking — if it has active (non-expired) locks,
        # another user is running tests via SSH and our local locks won't coordinate with theirs.
        self._check_no_remote_locks()

        with collector.timer(MetricName.CORE_ALLOCATION_TIME):
            devices = self.get_neuron_device_info()
            collector.add_dimension({MetricName.INSTANCE_TYPE: devices[0].instance_type})
            total_physical_cores = self.get_total_physical_cores()

            # We lock at the physical core level to prevent conflicts between
            # LNC1 and LNC2 tests (same approach as SshHost/CoreLockManager).
            num_physical_needed = collective_ranks * lnc_config

            if num_physical_needed > total_physical_cores:
                raise NoNeuronDevicesException(
                    f"Requested {collective_ranks} logical cores (lnc{lnc_config} = {num_physical_needed} physical) "
                    f"but only {total_physical_cores} physical cores available on {self.host_id}"
                )

            os.makedirs(self.core_allocation_dir, exist_ok=True)
            lock = self.__lock__(self._core_state_path, timeout_seconds)
            allocated_physical: list[int] = []
            pid = os.getpid()

            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                with lock.acquire():
                    state = self._read_core_state(total_physical_cores)
                    self._purge_stale_owners(state)
                    all_physical = list(range(total_physical_cores))
                    available = [c for c in all_physical if c not in state["in_use"]]

                    # Find a contiguous, aligned block of physical cores
                    result = find_contiguous_cores(available, num_physical_needed, total_physical_cores)
                    if result:
                        allocated_physical = result
                        state["in_use"].extend(allocated_physical)
                        state.setdefault("owners", {})[str(pid)] = allocated_physical
                        self._write_core_state(state)
                        break

                logging.info(
                    f"Waiting for {num_physical_needed} physical cores (lnc{lnc_config}), "
                    f"{len(available)} available. Retrying in {poll_period_seconds}s..."
                )
                time.sleep(poll_period_seconds)
            else:
                raise TimeoutException(
                    f"Timed out waiting for {num_physical_needed} physical cores on {self.host_id} "
                    f"after {timeout_seconds}s"
                )

        # Convert physical core IDs to logical core IDs (same logic as CoreLockManager)
        logical_cores = CoreLockManager._physical_to_logical_cores(allocated_physical, lnc_config)

        try:
            yield CoreAllocation(host_id=self.host_id, logical_core_ids=logical_cores, lnc_config=lnc_config)
        finally:
            with lock.acquire():
                state = self._read_core_state(total_physical_cores)
                for core in allocated_physical:
                    if core in state["in_use"]:
                        state["in_use"].remove(core)
                state.get("owners", {}).pop(str(pid), None)
                self._write_core_state(state)

    @staticmethod
    def _purge_stale_owners(state: dict) -> None:
        """Remove core reservations from PIDs that no longer exist."""
        owners = state.get("owners", {})
        stale_pids = []
        for pid_str, _cores in owners.items():
            try:
                os.kill(int(pid_str), 0)
            except OSError:
                stale_pids.append(pid_str)
        for pid_str in stale_pids:
            stale_cores = owners.pop(pid_str)
            for core in stale_cores:
                if core in state["in_use"]:
                    state["in_use"].remove(core)
            logging.warning(f"Purged stale core locks from dead PID {pid_str}: cores {stale_cores}")

    def _read_core_state(self, total_physical_cores: int) -> dict:
        """Read or initialize the local core allocation state file."""
        if os.path.exists(self._core_state_path):
            with open(self._core_state_path, "r") as f:
                return json.load(f)
        return {"total_physical_cores": total_physical_cores, "in_use": []}

    def _write_core_state(self, state: dict) -> None:
        """Write the local core allocation state file."""
        with open(self._core_state_path, "w") as f:
            json.dump(state, f)

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        result = subprocess.run(
            [self.neuron_ls_path, "--json-output"],
            capture_output=True,
            text=True,
            timeout=SSH_INFERENCE_TIMEOUT_SECONDS,
        )
        data = json.loads(result.stdout)
        if not data:
            raise NoNeuronDevicesException("localhost")
        return [NeuronDeviceInfo.from_dict(device) for device in data]


@final
class SshHost(Host):
    def __init__(
        self,
        ssh_alias: str,
        test_base_path: str,
        remote_neuron_install_dir: str,
        ssh_config_path: str,
        s3_config: S3ArtifactUploadConfig,
        patience_seconds: int = DEFAULT_PATIENCE_SECONDS,
        remote_base_path: str = "/tmp/neuronx-cc/tests",
        core_reset_strategy: CoreResetStrategy = _DEFAULT_CORE_RESET_STRATEGY,
        exclusive: bool = False,
    ):
        super().__init__(core_reset_strategy)
        self.ssh_alias: str = ssh_alias
        self.s3_config = s3_config
        self.patience_seconds = patience_seconds
        self._ssh_config_path = ssh_config_path

        # SshCommunication is an experimental communication format; it is guarded
        # by the "exclusive host" feature flag until it will be streamlined for
        # all use cases.
        if exclusive:
            self._comm = SshCommunication(
                ssh_config_path=ssh_config_path,
                alias=ssh_alias,
                s3_config=s3_config,
            )
        else:
            self._comm = ParamikoCommunication(
                ssh_config_path,
                ssh_alias,
                s3_config=s3_config,
            )
        self.remote_base_path: str = remote_base_path
        self.remote_full_path: str | None = None

        self.lock_file_path: str = os.path.join(test_base_path, self.ssh_alias)
        self._total_physical_cores: int | None = None
        self._remote_executor = None
        self._host_locking_version: int | None = None
        # Cached during one allocation attempt so a manager created by
        # soft_join_queue during artifact upload is reused (same entry_id) at
        # commit time. Bound to the owning collector and cleared on teardown so
        # it never bleeds across tests/attempts (see _ensure_core_lock_manager).
        self._core_lock_manager: CoreLockManager | None = None
        self.neuron_ls_path: str = os.path.join(remote_neuron_install_dir, "neuron-ls")
        # Cached for the lifetime of this SshHost — device topology is assumed stable during a test run.
        self._cached_device_info: list[NeuronDeviceInfo] | None = None

    @override
    def get_host_id(self) -> str:
        return self.ssh_alias

    @override
    def get_total_physical_cores(self) -> int:
        if self._total_physical_cores is None:
            devices = self.get_neuron_device_info()
            self._total_physical_cores = sum(len(d.neuroncore_ids) * d.logical_neuroncore_config for d in devices)
        return self._total_physical_cores

    @override
    def _get_debug_output_dir(self, target_directory: str) -> str:
        return "$(pwd)/debug_output"

    def _get_remote_executor(self) -> RemoteExecutor:
        """Lazily create a RemoteExecutor for this host's connection."""
        if self._remote_executor is None:
            if isinstance(self._comm, SshCommunication):
                lock_comm = ParamikoCommunication(self._ssh_config_path, self.ssh_alias)
                self._remote_executor = RemoteExecutor(lock_comm)
            else:
                self._remote_executor = RemoteExecutor(self._comm)
        return self._remote_executor

    def _reconnect(self):
        """Reconnect SSH and rebuild the remote executor."""
        self._comm.reconnect()
        # Force lazy re-creation: the old executor holds the dead transport.
        self._remote_executor = None

    @override
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        env_var = " && ".join(f'export {k}="{v}"' for k, v in neuron_env.items())
        full_command = self.inside_venv(f"set -o pipefail; cd {self.remote_full_path} && {env_var} && {command}")
        logging.info(f"Executing remote command: {full_command}")

        result = self._comm.run(full_command, timeout=self._comm.WORKLOAD_COMMAND_TIMEOUT)

        if result.failed:
            # Log PATH on remote host to help debug missing tools issues
            path_result = self._comm.run("echo DIAGNOSTIC: PATH=$PATH", warn=True)
            logging.warning(
                f"Remote PATH on {self.ssh_alias}: {path_result.stdout.strip() if path_result.ok else 'FAILED TO GET PATH'}"
            )
            raise RemoteExecutionException(f"Unable to execute {command} in {self.remote_full_path}", result)

        return result.stdout

    @override
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        full_command = self.inside_venv(f"set -o pipefail; cd {self.remote_full_path} && {command}")
        logging.info(f"Executing remote post-lock command: {full_command}")

        result = self._comm.run(full_command, timeout=POST_LOCK_TIMEOUT_SECONDS)

        if result.failed:
            raise RemoteExecutionException(f"Post-lock command failed in {self.remote_full_path}", result)

        return result.stdout

    @override
    def _should_run_post_lock(self) -> bool:
        if self._host_locking_version is None:
            return True
        try:
            return not lock_client.is_draining(self._get_remote_executor(), self._host_locking_version)
        except Exception as e:
            logging.warning(f"[{self.ssh_alias}] Failed to check drain state: {e}")
            return True

    @override
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        local_download_location = os.path.join(target_directory, INF_ARTIFACT_DIR_NAME)
        # Clean up any existing infer_result directory from previous failed/retry attempts
        shutil.rmtree(local_download_location, ignore_errors=True)

        return self.__download_artifacts__(
            remote_path=self.remote_full_path,
            local_path=local_download_location,
            list_of_files_to_copy=(get_list_of_files_to_copy(stdout) if get_list_of_files_to_copy else None),
            collector=collector,
        )

    def __download_artifacts__(
        self,
        remote_path: str,
        local_path: str,
        collector: IMetricsCollector,
        list_of_files_to_copy: list[str] | None = None,
    ):
        with collector.timer(MetricName.FILE_TRANSFER_DOWNLOAD_TIME):
            return self._comm.directory(remote_path, collector).download(
                destination_dir_path=local_path,
                list_of_files=list_of_files_to_copy,
            )

    def __cleanup_remote_paths__(self, *remote_path_list: str, base_exception: Exception | None = None):
        exceptions: list[Exception] = [base_exception] if base_exception else []
        for remote_path in remote_path_list:
            try:
                self._comm.run(f"rm -rf {remote_path}")
            except Exception as e:
                exceptions.append(e)

        if len(exceptions) > 0:
            raise Exception(*exceptions)

    def __cleanup_local_paths__(self, *local_path_list: str, base_exception: Exception | None = None):
        exceptions: list[Exception] = [base_exception] if base_exception else []
        for local_path in local_path_list:
            command_result = subprocess.run(["rm", "-rf", local_path])
            if command_result.returncode != 0:
                exceptions.append(LocalExecutionException(f"Unable to delete {local_path}", command_result))

        if len(exceptions) > 0:
            raise Exception(*exceptions)

    @override
    @contextlib.contextmanager
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ):
        # Add PID to make remote path unique per process
        # allows multiple machines to run the same test on the same host
        pid = os.getpid()
        test_base_dir = f"{os.path.basename(target_directory)}_pid{pid}"
        remote_full_path = os.path.join(self.remote_base_path, test_base_dir)

        remote_dir = self._comm.directory(remote_full_path, collector)

        with collector.timer(MetricName.FILE_TRANSFER_UPLOAD_TIME):
            remote_dir.upload(target_directory, force_local_cleanup=force_local_cleanup)

        self.__install_prerequisites__(remote_path=remote_full_path)

        self.remote_full_path = remote_full_path

        try:
            yield remote_full_path
        finally:
            if not skip_remote_cleanup:
                remote_dir.cleanup()

    def __install_prerequisites__(self, remote_path: str):
        result = self._comm.run(f"python3 -m venv {remote_path}/.venv")
        if result.failed:
            raise RemoteExecutionException(
                f"Unable to initialize python virtual env at {remote_path}/.venv",
                result,
            )

    def inside_venv(self, command):
        assert self.remote_full_path
        return f"source {self.remote_full_path}/.venv/bin/activate && {command}"

    def __run_with_retry__(self, command: str, max_retries: int = 5, base_delay: float = 1.0, hide: bool = False):
        """
        Execute SSH command with exponential backoff retry logic to avoid SSH rate limiting.
        """
        import random

        for attempt in range(max_retries):
            try:
                logging.info(f"Executing remote command (attempt {attempt + 1}): {command}")
                result = self._comm.run(command, hide=hide)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if isinstance(e, (SSHException, OSError)):
                    logging.warning(f"SSH connection error (attempt {attempt + 1}): {e}. Reconnecting...")
                    try:
                        self._reconnect()
                    except Exception:
                        pass
                else:
                    logging.warning(f"Non-SSH error (attempt {attempt + 1}): {e}. Retrying...")
                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)

        raise Exception("Retry logic failed unexpectedly")

    def _ensure_core_lock_manager(self, collector: IMetricsCollector) -> CoreLockManager:
        """Create or return a CoreLockManager bound to the given collector.

        Resolves the host's physical-core count and locking version (creating
        infra_version.json with the default if missing), then returns a manager.
        The cached manager is reused ONLY while it belongs to the same
        ``collector`` (the per-test/per-attempt owner): a manager created by
        ``soft_join_queue`` during artifact upload is the SAME instance
        ``get_core_allocation`` later commits with, so the FIFO slot anchored at
        soft-join time is reused (same ``entry_id``). When the collector differs
        (a new test/attempt) the cache is rebuilt, which re-mints ``entry_id``,
        rebinds the collector, and resets all fairness counters — making
        cross-test metric bleed impossible.
        """
        if self._core_lock_manager is not None and self._core_lock_manager.collector is collector:
            return self._core_lock_manager

        self._total_physical_cores = self.get_total_physical_cores()
        if self._host_locking_version is None:
            executor = self._get_remote_executor()
            with collector.timer(MetricName.CORE_LOCK_INIT_TIME):
                self._host_locking_version = lock_client.initialize_and_deploy(executor)
            check_lock_version(self._host_locking_version)
        logging.info(f"[{self.ssh_alias}] Using locking protocol v{self._host_locking_version}")

        self._core_lock_manager = CoreLockManager(
            self.ssh_alias,
            self._total_physical_cores,
            collector,
            executor=self._get_remote_executor(),
            host_locking_version=self._host_locking_version,
        )
        return self._core_lock_manager

    @override
    def soft_join_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
    ) -> None:
        """Reserve a FIFO core-queue slot before/while artifacts upload.

        Synchronous (no background threads). First issue a read-only ``probe``
        that peeks the worst-case ETA for a not-yet-queued caller WITHOUT
        joining the queue. If that ETA exceeds patience (and the host is not
        draining) the host is rotated BEFORE any enqueue or upload by raising
        ``QueuePatienceRotation`` — no enqueue, no dequeue, no churn. Otherwise a
        single ``ready=False`` poll anchors this attempt's FIFO position so it
        does not pay the upload latency with its queue slot; the real commit
        happens later in ``get_core_allocation``, which reuses the SAME cached
        manager/``entry_id``.

        ``probe`` is the single ETA gate. ``should_rotate`` returns ``False``
        while draining (ETA is meaningless mid-drain), so a ``DRAINING`` probe
        does not short-circuit — it falls through to ``acquire(ready=False)`` and
        the normal stay-and-poll behavior.

        Best-effort: every ``probe``/``acquire`` RPC error is logged at warning
        and swallowed; ``get_core_allocation`` (the hard acquire) owns host
        lifetime — soft-join adds none. Only ``QueuePatienceRotation`` propagates
        so the surrounding host-assignment retry rotates. The cached manager is
        intentionally NOT dropped: a patience rotation does not mark the host
        failed, so the unused/enqueued entry is reused or transparently
        re-enqueued (by the ``poll`` verb) on the next attempt.
        """
        try:
            manager = self._ensure_core_lock_manager(collector)
            probe_outcome = manager.probe(collective_ranks, lnc_config)
            if probe_outcome.status in (AllocationStatus.QUEUED, AllocationStatus.DRAINING) and should_rotate(
                probe_outcome.worst_case_eta,
                draining=(probe_outcome.status == AllocationStatus.DRAINING),
                patience_seconds=self.patience_seconds,
            ):
                raise QueuePatienceRotation(
                    f"[{self.ssh_alias}] Queue ETA {probe_outcome.worst_case_eta}s exceeds "
                    f"patience {self.patience_seconds}s, rotating host"
                )
            outcome = manager.acquire(collective_ranks, lnc_config, ready=False)
            logging.info(
                f"[{self.ssh_alias}] Soft-joined core queue (ready=False): "
                f"status={outcome.status} position={outcome.position}"
            )
        except QueuePatienceRotation:
            raise
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[{self.ssh_alias}] Best-effort soft-join failed: {e}")

    @override
    @contextlib.contextmanager
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> Generator[CoreAllocation, None, None]:
        """
        Allocate logical cores for execution by locking physical cores.

        Physical cores are locked to prevent conflicts between LNC1 and LNC2 tests.
        Logical core IDs are returned for use with NEURON_RT_VISIBLE_CORES.
        """
        collector.add_dimension({MetricName.INSTANCE_TYPE: self.get_instance_type()})

        # Reuse the cached manager (and its entry_id) if soft_join_queue already
        # created one during artifact upload; otherwise build it now.
        core_lock_manager = self._ensure_core_lock_manager(collector)

        with collector.timer(MetricName.CORE_ALLOCATION_TIME):
            logging.info(
                f"[{self.ssh_alias}] Trying to acquire {collective_ranks} logical cores (lnc_config={lnc_config})"
            )

            # Poll until we get cores or timeout
            max_retryable_errors = 10
            # Liveness invariant: STALE_THRESHOLD must cover the poll loop's full
            # retry budget *including jitter* so a caller actively retrying
            # through transient SSH disruption is never pruned out from under
            # itself mid-budget. The real inter-refresh gap is POLL_PERIOD +
            # up to POLL_JITTER_MAX, so the budget is max_retryable_errors
            # refreshes at that spacing.
            assert STALE_THRESHOLD >= max_retryable_errors * (POLL_PERIOD + POLL_JITTER_MAX)
            consecutive_retryable_errors = 0
            start_time = time.time()
            outcome = None

            def _dequeue_best_effort() -> None:
                # Free the abandoned host's FIFO slot. Best-effort: a failed
                # dequeue is non-fatal (TTL prune is the crash safety net).
                try:
                    core_lock_manager.dequeue()
                except Exception as dq_err:  # noqa: BLE001
                    logging.warning(f"[{self.ssh_alias}] Best-effort dequeue failed: {dq_err}")

            def _abandon_attempt() -> None:
                # Terminal abandon teardown. Flush this attempt's queue-wait /
                # contention / drain-wait metrics BEFORE freeing the slot so the
                # starved attempts the observability was built to measure are not
                # invisible (success and abandon are mutually exclusive within an
                # attempt, so this never double-counts the success path's flush).
                core_lock_manager.record_contention_metrics()
                _dequeue_best_effort()

            while time.time() - start_time < timeout_seconds:
                try:
                    # Artifacts are uploaded before acquisition in the current
                    # ordering, so the caller is always ready to commit.
                    outcome = core_lock_manager.acquire(collective_ranks, lnc_config, ready=True)
                    consecutive_retryable_errors = 0
                    if outcome.status == AllocationStatus.ALLOCATED:
                        break
                    if outcome.status == AllocationStatus.DRAINING:
                        # Host draining: stay-and-probe. Keep polling
                        # the same host without counting this as a retryable error
                        # and without aborting. Drains are typically fleet-wide, so
                        # rotating off would just land on another draining host, so
                        # position is preserved organically as last_seen_ts keeps
                        # refreshing.
                        logging.info(
                            f"[{self.ssh_alias}] Host draining; continuing to poll (position={outcome.position})"
                        )
                    # QUEUED / DRAINING: no cores yet, keep polling.
                except (LockAcquisitionError, LockVersionError) as e:
                    logging.warning(f"[{self.ssh_alias}] Lock error (retryable={e.retryable}): {e}")
                    if not e.retryable:
                        raise
                    consecutive_retryable_errors += 1
                    # An error is not evidence the caller is near the front, so
                    # discard any stale near-front position from a prior
                    # successful poll; the cadence below must fall back to the
                    # SLOW base (select_poll_base(None) -> slow) instead of
                    # fast-spinning and burning the retry budget early.
                    outcome = None
                    if consecutive_retryable_errors >= max_retryable_errors:
                        _abandon_attempt()
                        raise OSError(
                            f"[{self.ssh_alias}] Lock acquisition failed after {max_retryable_errors} "
                            f"consecutive retryable errors. Last error: {e}"
                        ) from e
                base = select_poll_base(
                    outcome.position if outcome is not None else None,
                    slow=poll_period_seconds,
                )
                jitter = random.uniform(0, POLL_JITTER_MAX)
                time.sleep(base + jitter)

            if outcome is None or outcome.status != AllocationStatus.ALLOCATED:
                _abandon_attempt()
                raise TimeoutException(
                    f"[{self.ssh_alias}] Unable to allocate {collective_ranks} logical cores within {timeout_seconds}s"
                )

            logical_cores = outcome.logical_cores
            physical_cores = outcome.physical_cores
            core_lock_manager.record_contention_metrics()
            logging.info(f"[{self.ssh_alias}] Allocated logical cores {logical_cores} (physical: {physical_cores})")

        try:
            yield CoreAllocation(host_id=self.ssh_alias, logical_core_ids=logical_cores, lnc_config=lnc_config)
        finally:
            try:
                # best effort release attempt. we don't much care about the outcome because core reservation
                # is time bound at 60s, so worst case scenario we are going to waste a bit of time
                core_lock_manager.release(physical_cores)
            except Exception as e:
                logging.warning(
                    f"Unable to gracefully unlock cores, locks with autoexpire automatically after delay. Saw exception: {e}"
                )
                collector.record_metric(MetricName.CORE_LOCK_RELEASE_FAILED_COUNT, 1)
            # Drop the cached manager so the next attempt always starts fresh
            # (re-minted entry_id, rebound collector, reset counters) even if it
            # skips soft-join. The collector-identity guard in
            # _ensure_core_lock_manager already prevents cross-test reuse; this
            # is defense in depth and a clean per-attempt teardown hook.
            self._core_lock_manager = None

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        """Get Neuron device information from remote host with retry logic. Cached after first call."""
        if self._cached_device_info is not None:
            return list(self._cached_device_info)
        result = self.__run_with_retry__(
            f"NEURON_LOGICAL_NC_CONFIG={NeuronDeviceInfo.logical_neuroncore_config} {self.neuron_ls_path} --json-output",
            hide=True,
        )
        if result.failed:
            raise RemoteExecutionException(f"Unable to find neuron device on {self.ssh_alias}", result)
        logging.debug("neuron-ls output: %s", result.stdout)
        data = json.loads(result.stdout)
        if not data:
            raise NoNeuronDevicesException(self.ssh_alias)
        self._cached_device_info = [NeuronDeviceInfo.from_dict(device) for device in data]
        return self._cached_device_info

    def get_instance_type(self) -> str:
        return self.get_neuron_device_info()[0].instance_type


def _safe_probe_cores(host: Host) -> int:
    """Probe a host's physical-core count, returning 0 (ineligible) on any
    failure so a single unreachable host cannot abort fleet initialization."""
    try:
        return host.get_total_physical_cores()
    except Exception as e:
        logging.warning(f"Could not determine physical cores for a host; marking 0 (ineligible): {e}")
        return 0


@final
class HostManager:
    """Reader/distributor over an already-initialized host-state store.

    The store is the single source of truth for membership, host type, work-queue
    depth, and availability. HostManager only consumes it: it claims the least-busy
    available host, releases it, marks hosts unavailable, and hands out live host
    connections.
    """

    # Poll cadence while waiting for a recoverable host. In the normal case the wait ends
    # well before the deadline below: the background re-resolver monotonically drives the
    # pool toward either a claimable host (-> claim succeeds) or poison (-> fail fast), and
    # a short poll picks up whichever lands within ~one interval.
    _RECOVERABLE_CLAIM_POLL_SECONDS = 2
    # Hard ceiling on the wait, as a backstop for the case where the platform is never
    # poisoned due to some failure (eg. background process that brings hosts online or
    # poisons dies unexpectedly).
    _RECOVERABLE_CLAIM_DEADLINE_SECONDS = 1800  # 30 min

    def __init__(
        self,
        state_store: HostStateStore,
        neuron_installation_path: str,
        ssh_config_path: str,
        testrun_uid: str,
        needs_local_host: bool,
        s3_config: S3ArtifactUploadConfig | None = None,
        exclusive: bool = False,
        host_rotation_patience_seconds: int | None = DEFAULT_PATIENCE_SECONDS,
        hosts_recoverable: bool = False,
    ) -> None:
        self.testrun_uid = testrun_uid
        self.ssh_config_path = ssh_config_path
        self.s3_config = s3_config or S3ArtifactUploadConfig()
        self.exclusive = exclusive
        self.neuron_installation_path = neuron_installation_path
        self.hosts_recoverable = hosts_recoverable
        # A missing/None value (e.g. the unset CLI flag resolving to None) falls
        # back to the default rather than propagating None into should_rotate.
        if host_rotation_patience_seconds is None:
            host_rotation_patience_seconds = DEFAULT_PATIENCE_SECONDS
        self.host_rotation_patience_seconds = host_rotation_patience_seconds

        self.state_store = state_store
        self.base_host_info_path = state_store.base_dir

        # In-process cache of live Host objects keyed by alias — NOT persisted state.
        self.target_hosts: dict[str, Host] = {}

        if needs_local_host:
            self._setup_local_host()

    def _get_host(self, alias: str) -> Host:
        """Return the host connection for alias (built and cached on first use)."""
        host = self.target_hosts.get(alias)
        if host is None:
            host = SshHost(
                alias,
                test_base_path=self.base_host_info_path,
                remote_neuron_install_dir=self.neuron_installation_path,
                ssh_config_path=self.ssh_config_path,
                s3_config=self.s3_config,
                patience_seconds=self.host_rotation_patience_seconds,
                core_reset_strategy=self.__build_reset_strategy__(alias),
                exclusive=self.exclusive,
            )
            self.target_hosts[alias] = host
        return host

    def __build_reset_strategy__(self, ssh_alias: str) -> CoreResetStrategy:
        """Per-capture reset policy for a remote host.

        Exclusive sessions skip the per-capture reset except when core state
        demands one (see CoreResetStrategy); shared runs always reset. State is
        namespaced by testrun_uid + sanitized alias so xdist workers share it.
        """
        return CoreResetStrategy(
            exclusive_run=self.exclusive,
            testrun_uid=self.testrun_uid,
            ssh_alias=ssh_alias,
        )

    def mark_host_unavailable(self, host_id: str):
        """Mark a host unavailable in the store after a host-level failure (connection
        fault or exhausted core-lock acquisition); the row is kept (available=false)."""
        self.state_store.mark_unavailable(host_id)
        logging.warning(f"Host {host_id} marked unavailable")

    def get_failed_host_count(self) -> int:
        """Number of hosts currently marked unavailable — the deduplicated count
        of hosts that have failed this run (sourced from the store)."""
        return self.state_store.unavailable_count()

    def _eligible_host_aliases(self, platform_target: Platforms, num_of_physical_cores_needed: int) -> set[str]:
        """Aliases currently eligible to serve this request (same predicate as the store's
        claim). The retry loop compares its busy set against this to know when the whole
        eligible fleet has been tried-and-busy and it should re-rotate."""
        return self.state_store.eligible_host_aliases(platform_target, num_of_physical_cores_needed)

    def probe_and_record_capacity(self, aliases: list[str], max_probe_workers: int | None = None) -> None:
        """Probe the physical-core capacity of ``aliases`` concurrently and persist it onto
        their store rows, so capacity-based routing has real core counts to rank by."""
        if not aliases:
            return
        workers = resolve_probe_worker_count(len(aliases), max_probe_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {alias: executor.submit(_safe_probe_cores, self._get_host(alias)) for alias in aliases}
            cores_by_alias = {alias: future.result() for alias, future in futures.items()}
        self.state_store.set_physical_cores(cores_by_alias)

    LOCAL_HOST_ID = "localhost"

    def __get_host_assignment__(
        self,
        platform_target: Platforms,
        num_of_physical_cores_needed: int,
        collector: IMetricsCollector = NOOP_METRICS_COLLECTOR,
        exclude: set[str] | None = None,
    ) -> Host:
        # Claim a host atomically for a request needing num_of_physical_cores_needed cores.
        # The store does capacity-aware selection (skip too-small hosts, rank by
        # post-placement load ratio, deprioritize the ``exclude`` set of already-tried-busy
        # hosts) + work_queue_depth increment under one lock, so concurrent workers can't
        # both grab the same headroom. claim raises FleetEmptyError if the platform is
        # poisoned — applies equally to the recoverable and non-recoverable paths, so it's
        # handled by the store rather than duplicated here.
        # FIXME(work-queue-leak): a worker killed between claim and release_host() leaks its
        # reservation for the session, skewing balancing (not a correctness bug — core locks
        # still protect cores).
        claimed = self.state_store.claim_least_busy(platform_target, num_of_physical_cores_needed, exclude=exclude)
        if claimed is None:
            if self.hosts_recoverable:
                claimed = self._wait_for_recoverable_host(
                    platform_target, num_of_physical_cores_needed, collector, exclude=exclude
                )
            else:
                # Fail fast with a diagnostic distinguishing "no such platform" / "too
                # small" / "unavailable" so a sizing mismatch reads differently from an
                # exhausted fleet.
                reason = self.state_store.explain_unclaimable(platform_target, num_of_physical_cores_needed)
                raise FleetEmptyError(f"No available hosts for platform {platform_target.value} - {reason}")

        return self._get_host(claimed.resolved.ssh_host)

    def _setup_local_host(self):
        detected_platform = detect_local_platform(self.neuron_installation_path)
        if detected_platform:
            logging.info(f"Auto-detected local platform: {detected_platform.value}")
        else:
            raise ValueError("Unable to detect the platform on this (local) host. Ensure neuron-ls is available.")

        host = LocalHost(
            self.neuron_installation_path,
            self.LOCAL_HOST_ID,
            self.base_host_info_path,
            core_reset_strategy=CoreResetStrategy(exclusive_run=False),
        )
        resolved_host = ResolvedHost(self.LOCAL_HOST_ID, detected_platform, host.get_total_physical_cores())

        self.target_hosts[self.LOCAL_HOST_ID] = host
        self.state_store.initialize([resolved_host])

    def _wait_for_recoverable_host(
        self,
        platform_target: Platforms,
        num_of_physical_cores_needed: int,
        collector: IMetricsCollector = NOOP_METRICS_COLLECTOR,
        exclude: set[str] | None = None,
    ) -> HostRecord:
        """Poll until a host can be claimed or the fleet is poisoned. Only called when
        ``hosts_recoverable`` is set.

        Returns the claimed record as soon as one is available, or raises
        ``FleetEmptyError``.

        A ``_RECOVERABLE_CLAIM_DEADLINE_SECONDS`` ceiling backstops the one case the
        poison path can't terminate on its own: a dead background re-resolver that never
        reaches the poison threshold. It then raises ``TimeoutException`` (NOT
        ``FleetEmptyError``: the fleet was never declared empty).

        Records the wall-clock spent waiting so the time absorbed here is visible in
        metrics rather than hidden as unexplained wall-clock; the timer context records on
        both the success and the timeout (TimeoutException) paths (it records on __exit__ and
        does not suppress the exception)."""
        with collector.timer(MetricName.RECOVERABLE_HOST_WAIT_TIME) as timer:
            deadline = timer.start_time + self._RECOVERABLE_CLAIM_DEADLINE_SECONDS
            while time.time() < deadline:
                # claim_least_busy raises FleetEmptyError if the fleet is poisoned.
                claimed = self.state_store.claim_least_busy(
                    platform_target, num_of_physical_cores_needed, exclude=exclude
                )
                if claimed is not None:
                    return claimed
                time.sleep(self._RECOVERABLE_CLAIM_POLL_SECONDS)
            raise TimeoutException(
                f"No host for platform {platform_target.value} within "
                f"{self._RECOVERABLE_CLAIM_DEADLINE_SECONDS}s, and the pool was never poisoned - "
                "the background host re-resolver likely stopped advancing (e.g. its thread died)."
            )

    def release_host(self, host: Host, num_of_physical_cores_to_release: int):
        self.state_store.release(host.get_host_id(), num_of_physical_cores_to_release)

    def get_host_assignment_with_retry(
        self,
        platform_target: Platforms,
        collector: IMetricsCollector,
        collective_ranks: int,
        lnc_config: int,
        *,
        deadline_seconds: float = DEFAULT_ACQUISITION_DEADLINE_SECONDS,
        connection_failure_cap: int = 3,
        backoff_seconds: float = HOST_ROTATION_BACKOFF_SECONDS,
    ):
        """Execute a function with automatic retry on different hosts.

        A busy FIFO queue (``QueuePatienceRotation``) is transient: the host is NOT
        marked unavailable and stays eligible, the caller backs off briefly and keeps
        rotating across hosts until ``deadline_seconds`` elapses -- it never
        self-terminates merely because all matching hosts are currently busy.
        Genuine host-level failures (connection faults, or core-lock acquisition
        erroring out) mark the host unavailable (recoverable -- a later re-resolution
        can bring it back), and stop after ``connection_failure_cap`` of them.
        """
        # Track errors from each host attempt for better debugging
        host_errors: dict[str, str] = {}

        def format_host_errors() -> str:
            """Format all captured host errors for the exception message."""
            return "\n".join(f"  - {host}: {error}" for host, error in host_errors.items())

        def reraise_with_history(e: Exception) -> None:
            """Re-raise ``e`` as an InferenceException prefixed with the errors from prior
            host rotations, so a terminal failure shows the whole history. With no prior
            errors, re-raises ``e`` unchanged."""
            if host_errors:
                raise InferenceException(f"{e}\n\nErrors from previous host attempts:\n{format_host_errors()}") from e
            raise

        # in case code block that's yielded to by context_manager_wrapper does not directly return
        # make sure that we record successes and terminate retries
        success = False
        # Genuine connection failures for THIS allocation; bounded by connection_failure_cap.
        connection_failures = 0
        # Transient busy-queue rotations across hosts for THIS allocation (no failure).
        rotation_count = 0
        # Hosts whose FIFO queue was busy during THIS allocation. Excluded from the next
        # deterministic pick so selection rotates instead of re-picking the same host;
        # cleared once the whole eligible fleet has been tried so a sustained-busy fleet
        # keeps cycling until the deadline.
        busy_hosts: set[str] = set()

        def succeeded():
            nonlocal success
            success = True

        @contextlib.contextmanager
        def context_manager_wrapper(notify_success: Callable[[], None], execution_host: Host):
            nonlocal connection_failures, rotation_count
            try:
                yield execution_host
            except QueuePatienceRotation as e:
                # A busy FIFO queue is transient, NOT a host failure: do not mark the
                # host unavailable, do not count it toward the connection-failure cap, and
                # leave it eligible for re-selection. Record the rotation metric and
                # back off so a fully-busy fleet re-cycles at a bounded rate.
                host_id = execution_host.get_host_id() if execution_host else "unknown"
                rotation_count += 1
                busy_hosts.add(host_id)
                # Reset against the currently-ELIGIBLE hosts, not all target hosts:
                # ineligible hosts (wrong platform / too small) never enter busy_hosts,
                # so comparing against the full target set would keep this reset from
                # ever firing in a heterogeneous pool.
                eligible = self._eligible_host_aliases(platform_target, num_of_physical_cores_needed)
                busy_hosts.intersection_update(eligible)  # drop entries no longer eligible (e.g. later failed)
                if eligible and busy_hosts >= eligible:
                    # Every eligible host has been tried-and-busy; clear so deterministic
                    # selection re-rotates the fleet rather than sticking on one host.
                    busy_hosts.clear()
                logging.info(f"Host {host_id} queue busy (patience rotation #{rotation_count}); rotating: {e}")
                if collector:
                    collector.record_metric(
                        MetricName.CORE_LOCK_HOST_ROTATION_COUNT,
                        1.0,
                        "Count",
                    )
                time.sleep(backoff_seconds)
            except (OSError, TimeoutError, SSHException, TimeoutException, CommandTimedOut) as e:
                host_id = execution_host.get_host_id() if execution_host else "unknown"
                error_msg = f"{type(e).__name__}: {e}"
                host_errors[host_id] = error_msg

                connection_failures += 1
                logging.error(
                    f"Connection error on host {host_id}, failure {connection_failures}/{connection_failure_cap}: {e}"
                )
                self.mark_host_unavailable(host_id)
                busy_hosts.discard(host_id)

                if collector:
                    # Emit one datapoint of 1.0 per rotation (a delta) so that
                    # sum/avg aggregations over EMF datapoints reflect the true
                    # rotation count instead of over-counting the cumulative value.
                    collector.record_metric(
                        MetricName.CORE_LOCK_HOST_ROTATION_COUNT,
                        1.0,
                        "Count",
                    )
                    collector.record_metric(
                        MetricName.FAILED_HOSTS_COUNT,
                        float(self.get_failed_host_count()),
                        "Count",
                    )

                if connection_failures >= connection_failure_cap:
                    error_details = format_host_errors()
                    raise InferenceException(
                        f"Connection error after {connection_failures} attempts. "
                        + f"Hosts attempted: {', '.join(attempted_hosts)}\n"
                        + f"Errors from each host:\n{error_details}"
                    ) from e

                logging.warning("Retrying on a different host")
            else:
                notify_success()

        attempted_hosts = []

        num_of_physical_cores_needed: int = calculate_total_needed_physical_cores(
            collectives_ranks=collective_ranks, lnc_config=lnc_config
        )

        deadline = time.time() + deadline_seconds
        while not success and time.time() < deadline:
            execution_host = None

            try:
                with collector.timer(MetricName.HOST_LOCK_TIME):
                    execution_host = self.__get_host_assignment__(
                        platform_target,
                        num_of_physical_cores_needed=num_of_physical_cores_needed,
                        collector=collector,
                        exclude=busy_hosts,
                    )

                host_id = execution_host.get_host_id()
                attempted_hosts.append(host_id)

                yield context_manager_wrapper(succeeded, execution_host)

            except FleetEmptyError as e:
                reraise_with_history(e)
            except InferenceException:
                raise
            except Exception as e:
                reraise_with_history(e)
            finally:
                if execution_host:
                    self.release_host(execution_host, num_of_physical_cores_needed)

        if not success:
            # Distinct from "No available hosts" and "Connection error after N attempts":
            # the fleet was reachable but every matching host stayed busy past the deadline.
            raise InferenceException(
                f"Core allocation deadline ({deadline_seconds}s) exceeded after rotating across hosts; "
                f"all matching hosts remained busy. "
                f"Patience rotations: {rotation_count}. Hosts attempted: {', '.join(attempted_hosts)}"
            )

    @staticmethod
    def test_session(
        *,
        target_hosts: list[TargetHost],
        ssh_config_path: str,
        testrun_uid: str,
    ) -> str:
        """Probe candidate hosts and return the alias of the first reachable one.

        Used by the controller in exclusive-host mode to pick one host for the
        entire session. Raises ``HostsBusyError`` if no host responds.
        """
        errors: list[str] = []
        for target_host in target_hosts:
            alias = target_host.ssh_host
            comm = SshCommunication(ssh_config_path=ssh_config_path, alias=alias)
            if comm.is_active():
                try:
                    result = comm.run("true", timeout=10)
                    if result.ok:
                        logging.info(f"[exclusive] Reserved host {alias} for session {testrun_uid}")
                        return alias
                except Exception as e:
                    logging.info(f"[exclusive] Host {alias} unavailable: {e}")
                    errors.append(f"{alias}: {e}")
            else:
                logging.info(f"[exclusive] Host {alias} unavailable: not active")
                errors.append(f"{alias}: not active")
        raise HostsBusyError(
            f"Could not reserve any exclusive host from {len(target_hosts)} candidate(s):\n" + "\n".join(errors)
        )
