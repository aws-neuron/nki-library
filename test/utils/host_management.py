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
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from random import Random, SystemRandom
from typing import Any, Callable, Generator, final

from filelock import FileLock
from invoke.exceptions import CommandTimedOut
from paramiko import SSHException
from typing_extensions import override

from . import core_lock_client as lock_client
from .common_dataclasses import (
    INF_ARTIFACT_DIR_NAME,
    MAX_PATH_COMPONENT_LENGTH,
    NEURON_LS_JSON_ARGS,
    NeuronDeviceInfo,
    NeuronLsReport,
    Platforms,
    ResolvedHost,
    looks_like_json,
    neuron_ls_json_command,
    resolve_probe_worker_count,
)
from .core_lock_client import POLL_JITTER_MAX, POLL_PERIOD, REMOTE_LOCKS_JSON, STALE_THRESHOLD
from .core_lock_manager import (
    AllocationOutcome,
    AllocationStatus,
    CoreAllocation,
    CoreLockManager,
    GroupCoreAllocation,
    LockAcquisitionError,
    LockVersionError,
    calculate_total_needed_physical_cores,
    check_lock_version,
)
from .exceptions import (
    FleetEmptyError,
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
    ParamikoCommunication,
    SshCommunication,
)
from .host_state import HostGroup, HostStateStore, HostUnit
from .metrics_collector import NOOP_METRICS_COLLECTOR, IMetricsCollector, MetricName, NoopMetricsCollector
from .remote_executor import RemoteExecutor
from .s3_utils import S3ArtifactUploadConfig
from .scripts.remote_lock_scripts import LockState, find_contiguous_cores

# Consecutive retryable lock errors (or error cycles, for a group) tolerated before giving up on a host.
MAX_RETRYABLE_LOCK_ERRORS = 10

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

# Directories excluded from the upload archive to reduce transfer size.
UPLOAD_EXCLUDED_PATHS = ["artifacts", "metrics"]

# Names of environment variables to forward from the local environment to the remote
# workload when they are set. Test configurations that depend on an optional runtime
# component add the variable that selects it here, so the workload runs against the same
# component as the local process.
ENV_PASSTHROUGH: set[str] = set()


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


def _helper_rejected_argument(error: LockAcquisitionError) -> bool:
    """Whether a poll failed because the deployed lock helper does not accept an argument.

    An older helper raises ``TypeError: ... got an unexpected keyword argument`` for
    a keyword it does not know; the client only sees that text in the error message.
    """
    return "unexpected keyword argument" in str(error)


def should_rotate(worst_case_eta: int | None, draining: bool, patience_seconds: int) -> bool:
    """Pure predicate: should a queued caller abandon this host and rotate away?

    Returns True only for a non-draining caller whose worst-case ETA exceeds
    ``patience_seconds``. Negative patience disables rotation i.e. would always return negative guaidance.
    A draining host never triggers rotation (ETA is meaningless mid-drain, so the caller stays and keeps probing),
    and a missing ETA is treated as "stay". Kept pure/host-free for unit testing.
    """
    return (not draining) and worst_case_eta is not None and worst_case_eta > patience_seconds and patience_seconds > 0


def _run_local_neuron_ls_json(neuron_ls_path: str, timeout: float) -> "subprocess.CompletedProcess[str]":
    """Run local neuron-ls with the newest supported JSON flag, falling back to the legacy one when
    the installed neuron-ls rejects it. Returns the first JSON-producing run, or the last attempt so
    the caller can report the error.

    Acceptance tests the payload shape, not the exit code: an old neuron-ls prints ``unknown flag``
    to stdout, so a rejected attempt can otherwise masquerade as output.
    """
    newest, *older = NEURON_LS_JSON_ARGS
    result = subprocess.run([neuron_ls_path, *newest], capture_output=True, text=True, timeout=timeout)
    for args in older:
        if result.returncode == 0 and looks_like_json(result.stdout):
            break
        result = subprocess.run([neuron_ls_path, *args], capture_output=True, text=True, timeout=timeout)
    return result


@functools.lru_cache(maxsize=1)
def _run_neuron_ls(neuron_installation_path: str) -> list[NeuronDeviceInfo] | None:
    """Run neuron-ls with JSON output and return the parsed devices, or None on failure."""
    try:
        neuron_ls_path = (
            os.path.join(neuron_installation_path, "neuron-ls")
            if not neuron_installation_path.endswith("neuron-ls")
            else neuron_installation_path
        )
        if not os.path.isfile(neuron_ls_path):
            return None
        result = _run_local_neuron_ls_json(neuron_ls_path, timeout=10)
        if result.returncode == 0:
            devices = NeuronLsReport.from_stdout(result.stdout).devices
            if devices:
                return devices
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
        skip_core_reset: bool = False,
    ) -> str | None:
        run_exception = None
        stdout = ""

        with self.get_core_allocation(
            collective_ranks=collective_ranks, lnc_config=lnc_config, collector=collector
        ) as core_allocation:
            with collector.timer(MetricName.CORE_LOCK_HOLD_TIME):
                unique_port = self._get_unique_collectives_port(core_allocation.logical_core_ids[0])
                reset_cores = (not skip_core_reset) or core_allocation.should_reset_cores
                neuron_env = self._build_neuron_env(
                    core_allocation,
                    lnc_config,
                    unique_port,
                    self._get_debug_output_dir(target_directory),
                    reset_cores,
                )

                neuron_env.update(self._setup_neuron_tags())
                try:
                    stdout = self._run_command(command, target_directory, neuron_env, collector)
                except Exception as e:
                    run_exception = e

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

    def abandon_queue(self) -> None:
        """Dequeue any core-queue entry this attempt still holds (best-effort, idempotent).

        Default no-op for hosts that do not join the remote FIFO queue.
        """
        return None

    @abstractmethod
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        raise UnimplementedException()

    @abstractmethod
    def get_host_id(self) -> str:
        raise UnimplementedException()

    def probe_identity(self, platform: Platforms) -> ResolvedHost:
        """Probe this host's identity: its id, ``platform`` (the platform it was requested as)
        and physical-core capacity. The default reports a single-node host; a host whose
        tooling reports a multi-node deployment fills in the ``logical_host_*`` fields."""
        return ResolvedHost(
            ssh_host=self.get_host_id(),
            host_type=platform,
            num_physical_cores=self.get_total_physical_cores(),
        )

    @property
    def hosts(self) -> tuple["Host", ...]:
        """The physical hosts this execution target spans."""
        return (self,)

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
        reset_cores: bool,
    ) -> dict[str, str]:
        """Build the common Neuron runtime environment variables for execution."""
        env = {
            "NEURON_RT_ENABLE_OCP": "1",
            "NEURON_RT_ENABLE_OCP_SATURATION": "1",
            "NEURON_RT_VISIBLE_CORES": core_allocation.get_core_list_str(),
            "NEURON_LOGICAL_NC_CONFIG": str(lnc_config),
            "NEURON_RT_ROOT_COMM_ID": f"localhost:{unique_port}",
            "NEURON_RT_DEBUG_OUTPUT_DIR": debug_output_dir,
            # Avoid core reset between tests (overriding per-test default).
            "NEURON_RT_RESET_CORES": "1" if reset_cores else "0",
        }
        for name in sorted(ENV_PASSTHROUGH):
            value = os.environ.get(name)
            if value:
                env[name] = value
        return env

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
    ):
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
        result = _run_local_neuron_ls_json(self.neuron_ls_path, timeout=SSH_INFERENCE_TIMEOUT_SECONDS)
        devices = NeuronLsReport.from_stdout(result.stdout).devices
        if not devices:
            raise NoNeuronDevicesException("localhost")
        return devices


def unique_remote_dir_name(local_dir_basename: str) -> str:
    """``<basename>_<16 random hex>``: the remote work directory for one test run.

    The suffix makes the directory unique per client, not per process: ``os.getpid()``
    (the previous scheme) repeats across PID namespaces, so two sandboxed clients
    running the same test on the same host shared a directory and overwrote each
    other's NEFF (``NEFF/NTFF mismatch``) or deleted each other's inputs on cleanup.

    Raises ValueError, naming the test, when the suffixed name would exceed NAME_MAX.
    """
    # 16 hex characters = 64 random bits. Directories whose cleanup never ran pile up on
    # the hosts (thousands per host); a new name must not land on one of those either,
    # and 64 bits keeps that below 1e-15 per run. The local name is capped at 200
    # characters (feature_flag_helper.MAX_DIR_NAME_LENGTH), so 16 always fits; the
    # check below guards other callers.
    name = f"{local_dir_basename}_{uuid.uuid4().hex[:16]}"
    if len(name.encode()) > MAX_PATH_COMPONENT_LENGTH:
        raise ValueError(
            f"Test name is too long for the fleet host's filesystem: the remote work directory "
            f"'{local_dir_basename}' is {len(local_dir_basename.encode())} bytes and needs 17 more "
            f"for a unique suffix, but path components are limited to {MAX_PATH_COMPONENT_LENGTH} "
            f"bytes. Shorten the test's parametrization id."
        )
    return name


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
        transport: str = "paramiko",
    ):
        self.ssh_alias: str = ssh_alias
        self.s3_config = s3_config
        self.patience_seconds = patience_seconds
        self._ssh_config_path = ssh_config_path

        if transport == "ssh":
            self._comm = SshCommunication(
                ssh_config_path=ssh_config_path,
                alias=ssh_alias,
                s3_config=s3_config,
            )
        elif transport == "paramiko":
            self._comm = ParamikoCommunication(
                ssh_config_path,
                ssh_alias,
                s3_config=s3_config,
            )
        else:
            raise ValueError(f"Unknown transport '{transport}' (expected 'ssh' or 'paramiko')")
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
        # Holds the full neuron-ls report so its multi-node deployment is kept alongside the devices.
        self._cached_neuron_ls_report: NeuronLsReport | None = None

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
        # Unique per client (random suffix), not per process: see unique_remote_dir_name.
        test_base_dir = unique_remote_dir_name(os.path.basename(target_directory))
        remote_full_path = os.path.join(self.remote_base_path, test_base_dir)

        remote_dir = self._comm.directory(remote_full_path, collector)

        with collector.timer(MetricName.FILE_TRANSFER_UPLOAD_TIME):
            remote_dir.upload(
                target_directory,
                force_local_cleanup=force_local_cleanup,
                excluded_paths=UPLOAD_EXCLUDED_PATHS,
            )

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
            self._check_patience(collector, collective_ranks, lnc_config)
            self._join_queue(collector, collective_ranks, lnc_config)
        except QueuePatienceRotation:
            raise
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[{self.ssh_alias}] Best-effort soft-join failed: {e}")

    def _attempt_manager(self, collector: IMetricsCollector) -> CoreLockManager:
        """Return the manager cached by an earlier step of this attempt for ``collector``, else build one."""
        manager = self._core_lock_manager
        if manager is not None and manager.collector is collector:
            return manager
        return self._ensure_core_lock_manager(collector)

    def _check_patience(self, collector: IMetricsCollector, collective_ranks: int, lnc_config: int) -> None:
        """Probe the queue without joining; raise ``QueuePatienceRotation`` if the ETA exceeds patience."""
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

    def _join_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int,
        lnc_config: int,
        *,
        hold: bool = False,
        hold_cap_seconds: int | None = None,
    ) -> AllocationOutcome:
        """Anchor a FIFO slot with one ``ready=False`` poll and return its outcome."""
        outcome = self._poll_queue(
            collector, collective_ranks, lnc_config, ready=False, hold=hold, hold_cap_seconds=hold_cap_seconds
        )
        logging.info(
            f"[{self.ssh_alias}] Soft-joined core queue (ready=False): "
            f"status={outcome.status} position={outcome.position}"
        )
        return outcome

    def _poll_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int,
        lnc_config: int,
        *,
        ready: bool,
        hold: bool = False,
        hold_cap_seconds: int | None = None,
    ) -> AllocationOutcome:
        """Issue one ``acquire`` poll on this attempt's manager.

        A held poll sends ``hold_cap_seconds`` as the hold cap when given, else this
        host's patience, or ``DEFAULT_PATIENCE_SECONDS`` when patience is not
        positive (rotation disabled), so a held entry always carries a positive cap.
        An unheld poll sends no cap. If the deployed helper rejects the held poll's
        arguments, the helper is redeployed once and the poll retried; a second
        failure propagates.
        """
        if not hold:
            hold_cap_seconds = None
        elif hold_cap_seconds is None:
            hold_cap_seconds = self.patience_seconds if self.patience_seconds > 0 else DEFAULT_PATIENCE_SECONDS

        def poll() -> AllocationOutcome:
            return self._attempt_manager(collector).acquire(
                collective_ranks, lnc_config, ready=ready, hold=hold, hold_cap_seconds=hold_cap_seconds
            )

        try:
            return poll()
        except LockAcquisitionError as e:
            if not hold or not _helper_rejected_argument(e):
                raise
            logging.warning(f"[{self.ssh_alias}] Lock helper rejected a held poll ({e}); redeploying and retrying once")
            self._redeploy_lock_helper()
        return poll()

    def _redeploy_lock_helper(self) -> None:
        """Re-run the version-gated helper deploy and refresh the cached locking version."""
        self._host_locking_version = lock_client.initialize_and_deploy(self._get_remote_executor())
        check_lock_version(self._host_locking_version)

    def _abandon_queue(self) -> None:
        """End this attempt: flush contention metrics, best-effort dequeue, then drop the cached manager.

        No-op without a manager. Dropping the manager (as ``_release_cores`` does)
        makes any later abandon in the same attempt a no-op, so metrics are
        flushed and the entry dequeued at most once.
        """
        manager = self._core_lock_manager
        if manager is None:
            return
        try:
            # Flush this attempt's queue-wait / contention / drain-wait metrics BEFORE
            # freeing the slot so starved attempts are not invisible (success and
            # abandon are mutually exclusive within an attempt, so this never
            # double-counts the success path's flush).
            manager.record_contention_metrics()
            # Best-effort: a failed dequeue is non-fatal (TTL prune is the crash safety net).
            try:
                manager.dequeue()
            except Exception as dq_err:  # noqa: BLE001
                logging.warning(f"[{self.ssh_alias}] Best-effort dequeue failed: {dq_err}")
        finally:
            self._core_lock_manager = None

    @override
    def abandon_queue(self) -> None:
        """Dequeue any core-queue entry this attempt still holds (best-effort, idempotent).

        A no-op without a cached manager (released or already abandoned) and when
        the cached manager never enqueued: a patience rotation keeps that manager
        for reuse on a later attempt on this host and has no entry to dequeue.
        """
        try:
            manager = self._core_lock_manager
            if manager is None or not manager.has_enqueued:
                return
            self._abandon_queue()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[{self.ssh_alias}] Abandoning the core queue entry failed: {e}")

    def _lock_signal_counts(self) -> tuple[int, int]:
        """Return this attempt's ``(bump_count, reenqueue_count)``; ``(0, 0)`` without a manager."""
        manager = self._core_lock_manager
        if manager is None:
            return 0, 0
        return manager.bump_count, manager.reenqueue_count

    def _record_contention_metrics(self) -> None:
        """Flush this attempt's queue-wait / contention metrics; no-op without a manager."""
        if self._core_lock_manager is not None:
            self._core_lock_manager.record_contention_metrics()

    def _release_cores(
        self, collector: IMetricsCollector, physical_cores: list[int], *, end_attempt: bool = True
    ) -> None:
        """Best-effort release of ``physical_cores``; with ``end_attempt``, also drop the cached manager.

        Dropping the manager ends this attempt's queue identity: the next poll
        mints a new entry at the tail of the queue. With ``end_attempt=False`` the
        manager is kept, so the next ``ready=False`` poll re-enqueues the same
        entry id (at the tail) and the attempt's metrics are not emitted again.
        """
        manager = self._core_lock_manager
        if manager is not None:
            try:
                # best effort release attempt. we don't much care about the outcome because core reservation
                # is time bound at 60s, so worst case scenario we are going to waste a bit of time
                manager.release(physical_cores)
            except Exception as e:
                logging.warning(
                    f"Unable to gracefully unlock cores, locks with autoexpire automatically after delay. Saw exception: {e}"
                )
                collector.record_metric(MetricName.CORE_LOCK_RELEASE_FAILED_COUNT, 1)
        if not end_attempt:
            return
        # Drop the cached manager so the next attempt always starts fresh
        # (re-minted entry_id, rebound collector, reset counters) even if it
        # skips soft-join. The collector-identity guard in
        # _ensure_core_lock_manager already prevents cross-test reuse; this
        # is defense in depth and a clean per-attempt teardown hook.
        self._core_lock_manager = None

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
            max_retryable_errors = MAX_RETRYABLE_LOCK_ERRORS
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

            while time.time() - start_time < timeout_seconds:
                try:
                    # Artifacts are uploaded before acquisition in the current
                    # ordering, so the caller is always ready to commit.
                    outcome = self._poll_queue(collector, collective_ranks, lnc_config, ready=True)
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
                        self._abandon_queue()
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
                self._abandon_queue()
                raise TimeoutException(
                    f"[{self.ssh_alias}] Unable to allocate {collective_ranks} logical cores within {timeout_seconds}s"
                )

            logical_cores = outcome.logical_cores
            physical_cores = outcome.physical_cores
            if logical_cores is None or physical_cores is None:
                self._abandon_queue()
                raise TimeoutException(
                    f"[{self.ssh_alias}] Allocation reported success without a core assignment "
                    f"(logical={logical_cores}, physical={physical_cores})"
                )
            core_lock_manager.record_contention_metrics()
            logging.info(f"[{self.ssh_alias}] Allocated logical cores {logical_cores} (physical: {physical_cores})")

        try:
            yield CoreAllocation(
                host_id=self.ssh_alias,
                logical_core_ids=logical_cores,
                lnc_config=lnc_config,
                should_reset_cores=outcome.should_reset_cores if outcome.should_reset_cores is not None else True,
            )
        finally:
            self._release_cores(collector, physical_cores)

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        """Get Neuron device information from remote host with retry logic. Cached after first call."""
        return list(self._get_neuron_ls_report().devices)

    def _get_neuron_ls_report(self) -> NeuronLsReport:
        """Run neuron-ls once (with retry) and cache the parsed report, devices and deployment alike.
        Raises if the command fails or reports no devices; nothing is cached then."""
        if self._cached_neuron_ls_report is not None:
            return self._cached_neuron_ls_report
        result = self.__run_with_retry__(neuron_ls_json_command(self.neuron_ls_path), hide=True)
        if result.failed:
            raise RemoteExecutionException(f"Unable to find neuron device on {self.ssh_alias}", result)
        logging.debug("neuron-ls output: %s", result.stdout)
        report = NeuronLsReport.from_stdout(result.stdout)
        if not report.devices:
            raise NoNeuronDevicesException(self.ssh_alias)
        self._cached_neuron_ls_report = report
        return report

    @override
    def probe_identity(self, platform: Platforms) -> ResolvedHost:
        """The base identity plus the multi-node deployment from the same cached neuron-ls report
        (all ``None`` for a single-node host or a neuron-ls predating instance-level output)."""
        report = self._get_neuron_ls_report()
        return replace(
            super().probe_identity(platform),
            logical_host_id=report.logical_host_id,
            logical_host_rank=report.logical_host_rank,
            logical_host_size=report.logical_host_size,
        )

    def get_instance_type(self) -> str:
        return self.get_neuron_device_info()[0].instance_type


_MULTINODE_EXECUTION_UNSUPPORTED = "multi-node execution is not supported yet"


def tag_failed_host(error: BaseException, host_id: str) -> None:
    """Attribute ``error`` to ``host_id`` unless an inner call already attributed it."""
    if getattr(error, "failed_host_id", None) is None:
        error.failed_host_id = host_id  # ty: ignore[unresolved-attribute]


@dataclass(frozen=True)
class GroupAttemptPolicy:
    """Contention bounds for one group allocation attempt, shared by every member.

    ``hold_cap_seconds`` is sent with every held poll. ``partial_wait_seconds`` is the
    stall budget: how long the group waits, partially assembled, while a lagging member
    is queued behind predecessors and no lagging member reaches a new best queue
    position, before it attempts to withdraw every queue entry and retries.
    ``retry_backoff_seconds`` is the sleep between that withdrawal and the retry.
    """

    hold_cap_seconds: int
    partial_wait_seconds: float
    retry_backoff_seconds: float


def sample_group_attempt_policy(base_seconds: int, rng: Random) -> GroupAttemptPolicy:
    """Sample a policy from ``base_seconds`` (see ``SshHostGroup._attempt_policy``).

    - Hold cap in [0.9, 1.1] x base: equal caps on two groups whose held entries cross
      on two hosts lapse in the same poll and swap the crossing forever.
    - Partial wait (stall budget) in [0.5, 0.75] x base: numerically below the smallest
      cap. This does not guarantee a stuck group withdraws before its held regions are
      bumped: the helper starts a region's cap when it assembles, which can happen during
      the soft-join upload, while the stall clock starts later, at the first allocation
      poll that observes the stall. Either outcome is live: a bumped region rejoins the
      tail and the stall clock keeps running. The 0.25 x base spread (45s at the 180s
      default, 9 poll periods) makes two groups stuck since the same poll usually
      withdraw in different polls.
    - Backoff in [1, 2] poll periods: the peer that the withdrawal unblocks commits on
      its next poll, before the withdrawn group joins again.
    """
    return GroupAttemptPolicy(
        hold_cap_seconds=max(1, round(rng.uniform(0.9, 1.1) * base_seconds)),
        partial_wait_seconds=rng.uniform(0.5, 0.75) * base_seconds,
        retry_backoff_seconds=rng.uniform(POLL_PERIOD, 2 * POLL_PERIOD),
    )


@dataclass(frozen=True)
class _CycleReadiness:
    """Outcome of one wait cycle: every member assembled, only some assembled, or any member draining.

    ``positions`` holds each member's 0-based queue position in rank order (None when the
    poll reported none). ``lagging`` holds the rank indices of members not assembled.
    """

    all_ready: bool
    partial: bool
    draining: bool
    positions: tuple[int | None, ...]
    lagging: tuple[int, ...]

    @property
    def stalled_behind_predecessor(self) -> bool:
        """Partially assembled with a lagging member queued behind predecessors.

        A lagging member at the queue head waits for active work to release its cores,
        which is ordinary progress, not a crossed queue. A lagging member with an unknown
        position is conservatively counted as queued behind predecessors.
        """
        return self.partial and any(self.positions[index] != 0 for index in self.lagging)


class _MemberMetricsCollector(NoopMetricsCollector):
    """Discards metrics but carries the caller's test name, so lock-helper event logs still name the test."""

    def __init__(self, test_name: str):
        super().__init__()
        self.test_name = test_name


@final
class SshHostGroup(Host):
    """One execution target spanning the SshHost members of a multi-node deployment.

    Members are held in rank order and the primary (first) member identifies the
    group. Supports identity, capacity, device discovery, the soft-join queue
    reservation, concurrent preparation of every member and coordinated core
    allocation; command execution across members is not supported yet.

    Each member call receives a member collector: the primary gets the caller's
    collector and every other member a metrics-discarding one, so a group test
    records one set of per-host lock and upload metrics (the primary's) and only
    the primary's calls touch the caller's collector concurrently. Group-level
    signals are recorded on the caller's collector from the calling thread:
    ``GROUP_PREPARE_TIME``, ``GROUP_COMMIT_ROLLBACK_COUNT``, the member maxima
    ``GROUP_MEMBER_MAX_BUMP_COUNT`` / ``GROUP_MEMBER_MAX_REENQUEUE_COUNT``, and
    ``CORE_LOCK_RELEASE_FAILED_COUNT`` for every member's release.

    Each attempt (soft-join through release or abandon) samples one
    ``GroupAttemptPolicy`` from ``rng``, by default a ``SystemRandom`` so groups in
    differently seeded or identically seeded processes draw independent bounds.

    ``rotate_on_patience`` enables the soft-join patience gate. The caller enables it only
    while another eligible unit is untried: a group passes the gate only when every member
    is under patience in the same pass, which a single busy deployment may never reach.
    """

    def __init__(
        self,
        members: tuple[SshHost, ...],
        *,
        rng: Random | None = None,
        rotate_on_patience: bool = True,
    ):
        if not members:
            raise ValueError("SshHostGroup needs at least one member")
        member_ids = [member.get_host_id() for member in members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError(f"SshHostGroup members must be distinct, got {member_ids}")
        self._members = members
        # First member not ready (or not committed) in the latest lock cycle; blamed on timeout.
        self._lagging_member: SshHost | None = None
        # Member collectors for the caller collector they were built for; reused across soft-join,
        # prepare and allocation so each member's cached lock manager (bound to its collector) stays valid.
        self._collectors_owner: IMetricsCollector | None = None
        self._collectors: tuple[IMetricsCollector, ...] = ()
        self._rng: Random = rng if rng is not None else SystemRandom()
        self._rotate_on_patience = rotate_on_patience
        # Sampled on first use in an attempt; cleared when the attempt ends.
        self._policy: GroupAttemptPolicy | None = None

    def _attempt_policy(self) -> GroupAttemptPolicy:
        """Return this attempt's policy, sampling it from the smallest positive member patience on first use.

        With no positive patience the base is ``DEFAULT_PATIENCE_SECONDS``: disabling
        ETA-based rotation (patience <= 0) does not disable group liveness recovery, so
        the stall budget stays positive and a stalled group still withdraws.
        """
        if self._policy is None:
            patience = [member.patience_seconds for member in self._members if member.patience_seconds > 0]
            base = min(patience) if patience else DEFAULT_PATIENCE_SECONDS
            self._policy = sample_group_attempt_policy(base, self._rng)
        return self._policy

    def _member_collectors(self, collector: IMetricsCollector) -> tuple[IMetricsCollector, ...]:
        """Return the per-member collectors (rank order) for ``collector``, creating them on first use."""
        if self._collectors_owner is not collector:
            test_name = getattr(collector, "test_name", "") or ""
            self._collectors = (collector, *(_MemberMetricsCollector(test_name) for _ in self._members[1:]))
            self._collectors_owner = collector
        return self._collectors

    @property
    @override
    def hosts(self) -> tuple[SshHost, ...]:
        return self._members

    @override
    def get_host_id(self) -> str:
        return self._members[0].get_host_id()

    @override
    def get_total_physical_cores(self) -> int:
        return sum(member.get_total_physical_cores() for member in self._members)

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        return [device for member in self._members for device in member.get_neuron_device_info()]

    @override
    def probe_identity(self, platform: Platforms) -> ResolvedHost:
        """A group spans several store rows, so it has no single identity; probe each member."""
        raise NotImplementedError("Probe the members of an SshHostGroup individually")

    def _ranks_per_member(self, collective_ranks: int) -> int:
        if collective_ranks % len(self._members) != 0:
            raise ValueError(
                f"{collective_ranks} collective ranks cannot be split evenly across {len(self._members)} hosts"
            )
        return collective_ranks // len(self._members)

    @override
    def soft_join_queue(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
    ) -> None:
        """Reserve a FIFO slot on every member; with ``rotate_on_patience``, none if any member exceeds patience.

        With ``rotate_on_patience``, every member's patience is checked before any
        member joins, so a ``QueuePatienceRotation`` leaves no entry behind. Without
        it, every member joins without a probe; the hold cap and the stall budget
        bound what the held entries cost. Joined entries hold their position while
        the caller keeps polling them; an upload longer than the stale threshold can
        still drop them, and they rejoin at the tail on the next poll. Other errors
        are logged per member; a member whose patience check failed is not joined.
        Every join carries this attempt's hold cap.
        """
        ranks_per_member = self._ranks_per_member(collective_ranks)
        collectors = self._member_collectors(collector)
        joinable: list[tuple[SshHost, IMetricsCollector]] = []
        for member, member_collector in zip(self._members, collectors, strict=True):
            if not self._rotate_on_patience:
                joinable.append((member, member_collector))
                continue
            try:
                member._check_patience(member_collector, ranks_per_member, lnc_config)
            except QueuePatienceRotation:
                self._policy = None
                raise
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[{member.get_host_id()}] Best-effort patience check failed: {e}")
                logging.warning(f"[{member.get_host_id()}] Skipping soft-join after the failed patience check")
                continue
            joinable.append((member, member_collector))
        hold_cap_seconds = self._attempt_policy().hold_cap_seconds
        for member, member_collector in joinable:
            try:
                member._join_queue(
                    member_collector, ranks_per_member, lnc_config, hold=True, hold_cap_seconds=hold_cap_seconds
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[{member.get_host_id()}] Best-effort soft-join failed: {e}")

    @override
    def abandon_queue(self) -> None:
        """Dequeue every member's core-queue entry this attempt still holds (best-effort, idempotent).

        Applies ``SshHost.abandon_queue`` per member, so only members whose manager enqueued are abandoned.
        Ends the attempt, so the next one samples a new policy.
        """
        self._policy = None
        for member in self._members:
            member.abandon_queue()

    @override
    @contextlib.contextmanager
    def prepare_host(
        self,
        target_directory: str,
        collector: IMetricsCollector,
        skip_remote_cleanup: bool = False,
        force_local_cleanup: bool = False,
    ) -> Generator[tuple[str, ...], None, None]:
        """Prepare every member concurrently and yield their remote paths in rank order.

        If any member fails, the members that were prepared are cleaned up and the
        first failure in rank order is re-raised, tagged with that member's id;
        every failure is logged and the later ones are attached as notes. If
        waiting is interrupted, every member still finishes entering and the
        entered ones are cleaned up before the interrupt propagates. The concurrent
        preparation is timed as ``GROUP_PREPARE_TIME`` on ``collector``; only the
        primary records its own upload metrics.
        """
        managers = [
            member.prepare_host(target_directory, member_collector, skip_remote_cleanup, force_local_cleanup)
            for member, member_collector in zip(self._members, self._member_collectors(collector), strict=True)
        ]
        with contextlib.ExitStack() as stack:
            pool = ThreadPoolExecutor(max_workers=len(managers))
            futures = [pool.submit(manager.__enter__) for manager in managers]
            interrupt: BaseException | None = None
            try:
                with collector.timer(MetricName.GROUP_PREPARE_TIME):
                    wait(futures)
            except BaseException as e:
                interrupt = e
            finally:
                # Joins the workers, so every future is done below even after an interrupt.
                pool.shutdown(wait=True)
            paths: list[str] = []
            errors: list[tuple[SshHost, BaseException]] = []
            for member, manager, future in zip(self._members, managers, futures, strict=True):
                error = future.exception()
                if error is None:
                    stack.push(manager.__exit__)
                    paths.append(future.result())
                else:
                    logging.warning(f"[{member.get_host_id()}] Preparing host failed: {error}")
                    errors.append((member, error))
            # Leaving the stack exits the members that did enter.
            if interrupt is not None:
                raise interrupt
            if errors:
                first_member, first_error = errors[0]
                tag_failed_host(first_error, first_member.get_host_id())
                for member, error in errors[1:]:
                    first_error.add_note(f"[{member.get_host_id()}] also failed to prepare: {error!r}")
                raise first_error
            yield tuple(paths)

    @override
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
        skip_core_reset: bool = False,
    ) -> str | None:
        raise NotImplementedError(_MULTINODE_EXECUTION_UNSUPPORTED)

    @override
    def _run_command(
        self,
        command: str,
        target_directory: str,
        neuron_env: dict[str, str],
        collector: IMetricsCollector,
    ) -> str:
        raise NotImplementedError(_MULTINODE_EXECUTION_UNSUPPORTED)

    @override
    def _collect_artifacts(
        self,
        target_directory: str,
        stdout: str,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None,
        collector: IMetricsCollector,
    ) -> str:
        raise NotImplementedError(_MULTINODE_EXECUTION_UNSUPPORTED)

    @override
    def _run_post_lock_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
    ) -> str:
        raise NotImplementedError(_MULTINODE_EXECUTION_UNSUPPORTED)

    @override
    def _get_debug_output_dir(self, target_directory: str) -> str:
        raise NotImplementedError(_MULTINODE_EXECUTION_UNSUPPORTED)

    @override
    @contextlib.contextmanager
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = POLL_PERIOD,
    ) -> Generator[GroupCoreAllocation, None, None]:
        """Lock ``collective_ranks // len(members)`` logical cores on every member at once.

        Wait phase: every cycle polls each member ``ready=False, hold=True``,
        concurrently. A held entry keeps its FIFO position and has a region of cores
        set aside for it without locking them. A member is ready when its host is not
        draining and reports that region assembled (set aside with none of its cores
        locked).

        Commit phase: entered in the first cycle where every member is assembled.
        Each member is polled ``ready=True`` once, in rank order, which locks its
        assembled region, so every member's lock lifetime starts within one cycle.
        If any member does not commit (its region was reclaimed between polls), or a
        retryable error interrupts the round after a commit, the committed members'
        cores are released and the group waits again; they keep their entry ids and
        rejoin at the tail on the next wait-phase poll. Each such rollback records
        ``GROUP_COMMIT_ROLLBACK_COUNT`` on ``collector``. Once every member commits,
        the highest per-member bump and re-enqueue counts are recorded on
        ``collector``; other queue metrics come only from the primary.

        Stalled partial readiness: the stall clock starts at the first cycle where some
        members are assembled and a lagging (unassembled) member is queued behind
        predecessors (position > 0, or an unknown position). A lagging member already at
        its queue head waits for active work to release cores and starts no clock. The
        clock restarts only when a lagging member reaches a new best queue position for
        this attempt (strictly below every position it reported before, assembled or
        not), so it is not restarted by lost readiness, a worse or repeated position,
        retryable errors or a cap bump; an ETA that falls over time is not progress.
        Cycles that are not stalled (every member unready, a drain, or every lagging
        member at its queue head) keep the clock running but never withdraw, so a stall
        that resumes uses the time already elapsed. Once the clock has reached this
        attempt's ``partial_wait_seconds`` in a stalled cycle, the group releases
        committed cores and attempts to dequeue every member's entry
        (best effort: an entry whose dequeue fails can keep its reservation until the
        helper prunes it as stale), sleeps ``retry_backoff_seconds`` clipped to what is
        left of ``timeout_seconds`` after that cleanup, and raises
        ``QueuePatienceRotation`` so the retry loop retries without marking any host
        unavailable. Two groups whose held entries cross on two hosts each hold one host
        and wait behind the other's entry on the other; the first to withdraw lets the
        other commit. Waits made only of cycles with every member unready, any member
        draining, or every lagging member at its queue head end under ``timeout_seconds``
        only.

        Errors, including setup failures, abandon every member's entry after
        releasing committed cores. Lock errors are attributed to the member that
        raised them; retryable ones are tolerated for up to
        ``MAX_RETRYABLE_LOCK_ERRORS`` consecutive cycles. A timeout is attributed
        to the member that was not ready or did not commit in the last cycle.
        Every member's cores are released when the caller's block exits; a failed
        member release is counted on ``collector``.
        """
        ranks_per_member = self._ranks_per_member(collective_ranks)
        primary = self._members[0]
        collectors = self._member_collectors(collector)
        try:
            for member, member_collector in zip(self._members, collectors, strict=True):
                self._call_member(member, member._ensure_core_lock_manager, member_collector)
            instance_type = self._call_member(primary, primary.get_instance_type)
            collector.add_dimension({MetricName.INSTANCE_TYPE: instance_type})
        except BaseException:
            self._abandon_all()
            raise

        with collector.timer(MetricName.CORE_ALLOCATION_TIME):
            logging.info(
                f"[{self.get_host_id()}] Trying to acquire {ranks_per_member} logical cores on each of "
                f"{len(self._members)} hosts (lnc_config={lnc_config})"
            )
            outcomes = self._acquire_all(
                collectors, collective_ranks, ranks_per_member, lnc_config, timeout_seconds, poll_period_seconds
            )
            for member in self._members:
                member._record_contention_metrics()
        self._record_member_signal_maxima(collector)

        member_allocations = tuple(
            CoreAllocation(
                host_id=member.get_host_id(),
                logical_core_ids=list(outcome.logical_cores or []),
                lnc_config=lnc_config,
                should_reset_cores=outcome.should_reset_cores if outcome.should_reset_cores is not None else True,
            )
            for member, outcome in zip(self._members, outcomes, strict=True)
        )
        try:
            yield GroupCoreAllocation(
                host_id=primary.get_host_id(),
                logical_core_ids=member_allocations[0].logical_core_ids,
                lnc_config=lnc_config,
                should_reset_cores=any(allocation.should_reset_cores for allocation in member_allocations),
                member_allocations=member_allocations,
            )
        finally:
            # The caller's collector records release failures for every member; releases run sequentially.
            self._policy = None
            for member, outcome in zip(self._members, outcomes, strict=True):
                member._release_cores(collector, list(outcome.physical_cores or []))

    def _record_member_signal_maxima(self, collector: IMetricsCollector) -> None:
        """Record the highest bump and re-enqueue counts over members; read before any release drops a manager."""
        counts = [member._lock_signal_counts() for member in self._members]
        collector.record_metric(MetricName.GROUP_MEMBER_MAX_BUMP_COUNT, max(bump for bump, _ in counts), "Count")
        collector.record_metric(
            MetricName.GROUP_MEMBER_MAX_REENQUEUE_COUNT, max(reenqueue for _, reenqueue in counts), "Count"
        )

    @staticmethod
    def _call_member(member: SshHost, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``call`` and attribute any exception it raises to ``member``."""
        try:
            return call(*args, **kwargs)
        except Exception as e:
            tag_failed_host(e, member.get_host_id())
            raise

    def _acquire_all(
        self,
        collectors: tuple[IMetricsCollector, ...],
        collective_ranks: int,
        ranks_per_member: int,
        lnc_config: int,
        timeout_seconds: int,
        poll_period_seconds: int,
    ) -> list[AllocationOutcome]:
        """Run wait/commit cycles until every member holds cores; return their outcomes in rank order.

        ``collectors`` holds the per-member collectors in rank order; the first is the caller's.
        """
        max_retryable_errors = MAX_RETRYABLE_LOCK_ERRORS
        consecutive_error_cycles = 0
        committed: dict[int, AllocationOutcome] = {}
        self._lagging_member = None
        start_time = time.time()
        deadline = start_time + timeout_seconds
        policy = self._attempt_policy()
        # Start of the current stall (see the docstring); restarted only by a new best position.
        stalled_since: float | None = None
        best_positions: dict[int, int] = {}
        try:
            # One pool for the whole allocation: a wait can last hours at one cycle per
            # POLL_PERIOD, so threads are not re-created every cycle. Shut down (joining any
            # in-flight poll) before the handler below releases or abandons entries.
            pool = ThreadPoolExecutor(max_workers=len(self._members))
            try:
                while time.time() - start_time < timeout_seconds:
                    try:
                        readiness = self._wait_cycle(pool, collectors, ranks_per_member, lnc_config, policy)
                        if readiness.all_ready and self._commit_all(
                            collectors, ranks_per_member, lnc_config, committed, policy
                        ):
                            return [committed[index] for index in range(len(self._members))]
                        consecutive_error_cycles = 0
                        now = time.time()
                        if self._record_best_positions(best_positions, readiness):
                            stalled_since = None
                        if readiness.stalled_behind_predecessor and stalled_since is None:
                            stalled_since = now
                        # Withdraw only from a cycle that is itself stalled (which excludes draining);
                        # all-unready or head-of-queue cycles keep the clock but never withdraw.
                        if (
                            stalled_since is not None
                            and readiness.stalled_behind_predecessor
                            and now - stalled_since >= policy.partial_wait_seconds
                        ):
                            self._withdraw(collectors, committed, policy, deadline)
                    except (LockAcquisitionError, LockVersionError) as e:
                        logging.warning(
                            f"[{getattr(e, 'failed_host_id', None)}] Lock error (retryable={e.retryable}): {e}"
                        )
                        if not e.retryable:
                            raise
                        consecutive_error_cycles += 1
                        if consecutive_error_cycles >= max_retryable_errors:
                            error = OSError(
                                f"[{self.get_host_id()}] Group lock acquisition failed after {max_retryable_errors} "
                                f"consecutive cycles with retryable errors. Last error: {e}"
                            )
                            tag_failed_host(error, getattr(e, "failed_host_id", None) or self.get_host_id())
                            raise error from e
                    time.sleep(poll_period_seconds + random.uniform(0, POLL_JITTER_MAX))
            finally:
                pool.shutdown(wait=True)
            timeout = TimeoutException(
                f"[{self.get_host_id()}] Unable to allocate {collective_ranks} logical cores across "
                f"{len(self._members)} hosts within {timeout_seconds}s"
            )
            if self._lagging_member is not None:
                tag_failed_host(timeout, self._lagging_member.get_host_id())
            raise timeout
        except QueuePatienceRotation:
            # Raised by _withdraw, which already attempted to release and dequeue every member.
            raise
        except BaseException:
            self._release_committed(collectors, committed)
            self._abandon_all()
            raise

    @staticmethod
    def _record_best_positions(best_positions: dict[int, int], readiness: _CycleReadiness) -> bool:
        """Record each member's lowest known queue position; return whether a lagging member set a new best.

        Every known position counts toward a member's best, including cycles where it is
        assembled. A first observation is not an improvement, and an unknown position
        neither improves nor resets a best.
        """
        improved = False
        for index, position in enumerate(readiness.positions):
            if position is None:
                continue
            best = best_positions.get(index)
            if best is None or position < best:
                if best is not None and index in readiness.lagging:
                    improved = True
                best_positions[index] = position
        return improved

    def _withdraw(
        self,
        collectors: tuple[IMetricsCollector, ...],
        committed: dict[int, AllocationOutcome],
        policy: GroupAttemptPolicy,
        deadline: float,
    ) -> None:
        """Attempt to release and dequeue every member's entry, back off, then raise ``QueuePatienceRotation``.

        Cleanup is best effort and runs to completion; the backoff is what remains of
        ``deadline`` afterwards, so no sleep is added once cleanup has used the budget.
        """
        lagging = self._lagging_member.get_host_id() if self._lagging_member is not None else None
        self._release_committed(collectors, committed)
        failed = self._abandon_all()
        backoff = max(0.0, min(policy.retry_backoff_seconds, deadline - time.time()))
        if failed:
            logging.warning(
                f"[{self.get_host_id()}] Queue withdrawal incomplete on {', '.join(failed)}; an entry whose "
                "dequeue failed may keep its reservation until the lock helper prunes it as stale"
            )
        logging.info(
            f"[{self.get_host_id()}] Group stalled behind a queue predecessor for "
            f"{policy.partial_wait_seconds:.0f}s (first unready: {lagging}); attempted to withdraw every "
            f"queue entry (best effort), retrying in {backoff:.1f}s"
        )
        time.sleep(backoff)
        raise QueuePatienceRotation(
            f"[{self.get_host_id()}] Group stalled behind a predecessor for {policy.partial_wait_seconds:.0f}s "
            "while partially assembled; attempted to withdraw every queue entry to retry"
        )

    def _abandon_all(self) -> list[str]:
        """Abandon every member's queue entry and end the attempt; one member's failure does not skip the others.

        Returns the ids of members whose abandon raised. An empty list does not confirm
        removal: a member's dequeue is itself best effort.
        """
        self._policy = None
        failed: list[str] = []
        for member in self._members:
            try:
                member._abandon_queue()
            except Exception as e:  # noqa: BLE001
                failed.append(member.get_host_id())
                logging.warning(f"[{member.get_host_id()}] Abandoning the core queue entry failed: {e}")
        return failed

    def _wait_cycle(
        self,
        pool: ThreadPoolExecutor,
        collectors: tuple[IMetricsCollector, ...],
        ranks_per_member: int,
        lnc_config: int,
        policy: GroupAttemptPolicy,
    ) -> _CycleReadiness:
        """Poll every member ``ready=False, hold=True`` concurrently; return the cycle's readiness.

        Outcomes are evaluated in rank order once every poll finished, so each entry
        is refreshed even when another member fails. The first non-retryable error in
        rank order is raised; otherwise every retryable error is logged and the last
        one is re-raised. Records the first member that is not ready (None if all are).
        ``partial`` means some members are ready, some are not and none is draining.
        """
        # Liveness: with concurrent polls, the gap between two refreshes of any entry is
        # bounded by the slowest single member RPC plus POLL_PERIOD + POLL_JITTER_MAX, the
        # same budget the single-host loop asserts against STALE_THRESHOLD.
        futures = [
            pool.submit(
                self._call_member,
                member,
                member._poll_queue,
                member_collector,
                ranks_per_member,
                lnc_config,
                ready=False,
                hold=True,
                hold_cap_seconds=policy.hold_cap_seconds,
            )
            for member, member_collector in zip(self._members, collectors, strict=True)
        ]
        wait(futures)
        ready_count = 0
        draining = False
        positions: list[int | None] = []
        lagging: list[int] = []
        last_error: LockAcquisitionError | LockVersionError | None = None
        self._lagging_member = None
        for index, (member, future) in enumerate(zip(self._members, futures, strict=True)):
            error = future.exception()
            if error is not None:
                if not isinstance(error, (LockAcquisitionError, LockVersionError)) or not error.retryable:
                    raise error
                logging.warning(f"[{member.get_host_id()}] Retryable lock error while waiting: {error}")
                last_error = error
                positions.append(None)
                lagging.append(index)
                if self._lagging_member is None:
                    self._lagging_member = member
                continue
            outcome: AllocationOutcome = future.result()
            if outcome.status == AllocationStatus.DRAINING:
                draining = True
                logging.info(
                    f"[{member.get_host_id()}] Host draining; continuing to poll (position={outcome.position})"
                )
            ready = outcome.status == AllocationStatus.QUEUED and outcome.assembled
            ready_count += ready
            positions.append(outcome.position)
            if not ready:
                lagging.append(index)
                if self._lagging_member is None:
                    self._lagging_member = member
        if last_error is not None:
            raise last_error
        return _CycleReadiness(
            all_ready=ready_count == len(self._members),
            partial=0 < ready_count < len(self._members) and not draining,
            draining=draining,
            positions=tuple(positions),
            lagging=tuple(lagging),
        )

    def _commit_all(
        self,
        collectors: tuple[IMetricsCollector, ...],
        ranks_per_member: int,
        lnc_config: int,
        committed: dict[int, AllocationOutcome],
        policy: GroupAttemptPolicy,
    ) -> bool:
        """Poll every member ``ready=True`` once in rank order; return whether all committed.

        ``committed`` maps rank index to its ALLOCATED outcome and is updated in
        place so the caller can release it on error. If a member does not commit,
        or a retryable lock error interrupts the round after a commit, the round is
        rolled back (see ``_roll_back``); a missed commit records the first
        uncommitted member as lagging and returns False, an error is re-raised.
        """
        for index, (member, member_collector) in enumerate(zip(self._members, collectors, strict=True)):
            try:
                outcome = self._call_member(
                    member,
                    member._poll_queue,
                    member_collector,
                    ranks_per_member,
                    lnc_config,
                    ready=True,
                    hold=True,
                    hold_cap_seconds=policy.hold_cap_seconds,
                )
            except (LockAcquisitionError, LockVersionError) as e:
                if e.retryable and committed:
                    self._roll_back(collectors, committed)
                raise
            if outcome.status != AllocationStatus.ALLOCATED:
                logging.warning(
                    f"[{member.get_host_id()}] Assembled host did not commit (status={outcome.status}); "
                    f"releasing {len(committed)} committed hosts and waiting again"
                )
                self._lagging_member = member
                self._roll_back(collectors, committed)
                return False
            if outcome.logical_cores is None or outcome.physical_cores is None:
                error = TimeoutException(
                    f"[{member.get_host_id()}] Allocation reported success without a core assignment "
                    f"(logical={outcome.logical_cores}, physical={outcome.physical_cores})"
                )
                tag_failed_host(error, member.get_host_id())
                raise error
            committed[index] = outcome
        return True

    def _roll_back(self, collectors: tuple[IMetricsCollector, ...], committed: dict[int, AllocationOutcome]) -> None:
        """Count one rollback on the caller's collector and release committed cores, keeping each queue entry."""
        collectors[0].record_metric(MetricName.GROUP_COMMIT_ROLLBACK_COUNT, 1.0, "Count")
        self._release_committed(collectors, committed, end_attempt=False)

    def _release_committed(
        self,
        collectors: tuple[IMetricsCollector, ...],
        committed: dict[int, AllocationOutcome],
        *,
        end_attempt: bool = True,
    ) -> None:
        """Release every committed member's cores and clear ``committed``; one failure does not skip the others.

        Release failures are counted on the caller's collector (``collectors[0]``) for every member.
        """
        for index, outcome in sorted(committed.items()):
            member = self._members[index]
            try:
                member._release_cores(collectors[0], list(outcome.physical_cores or []), end_attempt=end_attempt)
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[{member.get_host_id()}] Releasing committed cores failed: {e}")
        committed.clear()


def _safe_probe_identity(host: Host, platform: Platforms) -> ResolvedHost | None:
    """Probe a host's identity, returning None on any failure so a single unreachable host
    cannot abort fleet initialization; its store row is then left as it was."""
    try:
        return host.probe_identity(platform)
    except Exception as e:
        logging.warning(f"Could not probe host {host.get_host_id()}; keeping its known state: {e}")
        return None


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
        transport: str = "paramiko",
        host_rotation_patience_seconds: int | None = DEFAULT_PATIENCE_SECONDS,
        hosts_recoverable: bool = False,
    ) -> None:
        self.testrun_uid = testrun_uid
        self.ssh_config_path = ssh_config_path
        self.s3_config = s3_config or S3ArtifactUploadConfig()
        self.transport = transport
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
                transport=self.transport,
            )
            self.target_hosts[alias] = host
        return host

    def _get_target(self, unit: HostUnit, *, rotate_on_patience: bool = True) -> Host:
        """Return the execution target for a claimed unit: an SshHostGroup of the members (in
        rank order) for a multi-node deployment, otherwise the unit's single host.
        ``rotate_on_patience`` applies to a group only (see ``SshHostGroup``)."""
        if not isinstance(unit, HostGroup):
            return self._get_host(unit.host_id)
        members = []
        for alias in unit.member_ids:
            member = self._get_host(alias)
            if not isinstance(member, SshHost):
                raise TypeError(f"Multi-node member {alias} is a {type(member).__name__}, not an SshHost")
            members.append(member)
        return SshHostGroup(members=tuple(members), rotate_on_patience=rotate_on_patience)

    def mark_host_unavailable(self, host_id: str):
        """Mark a host unavailable in the store after a host-level failure (connection
        fault or exhausted core-lock acquisition); the row is kept (available=false)."""
        self.state_store.mark_unavailable(host_id)
        logging.warning(f"Host {host_id} marked unavailable")

    def get_failed_host_count(self) -> int:
        """Number of hosts currently marked unavailable — the deduplicated count
        of hosts that have failed this run (sourced from the store)."""
        return self.state_store.unavailable_count()

    def _eligible_host_aliases(
        self, platform_target: Platforms, num_of_physical_cores_needed: int, *, lnc_config: int = 1
    ) -> set[str]:
        """One id per unit currently eligible to serve this request (the unit's primary alias;
        same partition and predicate as the store's claim). The retry loop compares its busy set
        against this to know when every eligible unit has been tried-and-busy and it should
        re-rotate."""
        return self.state_store.eligible_host_aliases(
            platform_target, num_of_physical_cores_needed, lnc_config=lnc_config
        )

    def probe_and_record_capacity(self, aliases: list[str], max_probe_workers: int | None = None) -> None:
        """Probe ``aliases`` concurrently and persist each successful probe's physical-core
        capacity and multi-node deployment (as neuron-ls reports it) onto its existing store row,
        so capacity-based routing has real core counts to rank by and a deployment's members can
        be claimed together. A failed probe leaves its row unchanged; an alias with no row is
        skipped (its platform is unknown)."""
        state = self.state_store.read()
        platform_by_alias = {rec.resolved.ssh_host: rec.resolved.host_type for rec in state.hosts} if state else {}
        known = [alias for alias in aliases if alias in platform_by_alias]
        skipped = sorted(set(aliases) - set(known))
        if skipped:
            logging.warning(f"Not probing hosts with no store row: {skipped}")
        if not known:
            return
        workers = resolve_probe_worker_count(len(known), max_probe_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_safe_probe_identity, self._get_host(alias), platform_by_alias[alias])
                for alias in known
            ]
            probed = [identity for future in futures if (identity := future.result()) is not None]
        self.state_store.record_probed_hosts(probed)

    LOCAL_HOST_ID = "localhost"

    def __get_host_assignment__(
        self,
        platform_target: Platforms,
        num_of_physical_cores_needed: int,
        collector: IMetricsCollector = NOOP_METRICS_COLLECTOR,
        exclude: set[str] | None = None,
        *,
        lnc_config: int = 1,
    ) -> Host:
        return self._get_host(
            self._claim_unit(
                platform_target,
                num_of_physical_cores_needed,
                collector=collector,
                exclude=exclude,
                lnc_config=lnc_config,
            ).host_id
        )

    def _claim_unit(
        self,
        platform_target: Platforms,
        num_of_physical_cores_needed: int,
        collector: IMetricsCollector = NOOP_METRICS_COLLECTOR,
        exclude: set[str] | None = None,
        *,
        lnc_config: int = 1,
    ) -> HostUnit:
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
        claimed = self.state_store.claim_least_busy(
            platform_target, num_of_physical_cores_needed, exclude=exclude, lnc_config=lnc_config
        )
        if claimed is None:
            if self.hosts_recoverable:
                claimed = self._wait_for_recoverable_host(
                    platform_target, num_of_physical_cores_needed, collector, exclude=exclude, lnc_config=lnc_config
                )
            else:
                # Fail fast with a diagnostic distinguishing "no such platform" / "too
                # small" / "unavailable" so a sizing mismatch reads differently from an
                # exhausted fleet.
                reason = self.state_store.explain_unclaimable(
                    platform_target, num_of_physical_cores_needed, lnc_config=lnc_config
                )
                raise FleetEmptyError(f"No available hosts for platform {platform_target.value} - {reason}")

        return claimed

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
        *,
        lnc_config: int = 1,
    ) -> HostUnit:
        """Poll until a host can be claimed or the fleet is poisoned. Only called when
        ``hosts_recoverable`` is set.

        Returns the claimed unit as soon as one is available, or raises
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
                    platform_target, num_of_physical_cores_needed, exclude=exclude, lnc_config=lnc_config
                )
                if claimed is not None:
                    return claimed
                time.sleep(self._RECOVERABLE_CLAIM_POLL_SECONDS)
            reason = self.state_store.explain_unclaimable(
                platform_target, num_of_physical_cores_needed, lnc_config=lnc_config
            )
            raise TimeoutException(
                f"No host for platform {platform_target.value} within "
                f"{self._RECOVERABLE_CLAIM_DEADLINE_SECONDS}s, and the pool was never poisoned - "
                f"{reason}. Unless that placement is impossible, the background host re-resolver "
                "likely stopped advancing (e.g. its thread died)."
            )

    def release_host(
        self,
        host: Host,
        num_of_physical_cores_to_release: int,
        member_ids: tuple[str, ...] | None = None,
    ):
        self.state_store.release(host.get_host_id(), num_of_physical_cores_to_release, member_ids=member_ids)

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
                eligible = self._eligible_host_aliases(
                    platform_target, num_of_physical_cores_needed, lnc_config=lnc_config
                )
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
                target_id = execution_host.get_host_id() if execution_host else "unknown"
                # A multi-node target tags the failing member; mark that host, not the primary.
                failed_id = getattr(e, "failed_host_id", None) or target_id
                error_msg = f"{type(e).__name__}: {e}"
                host_errors[failed_id] = error_msg

                connection_failures += 1
                logging.error(
                    f"Connection error on host {failed_id}, failure {connection_failures}/{connection_failure_cap}: {e}"
                )
                self.mark_host_unavailable(failed_id)
                busy_hosts.discard(failed_id)
                busy_hosts.discard(target_id)

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
            claimed_unit: HostUnit | None = None

            try:
                with collector.timer(MetricName.HOST_LOCK_TIME):
                    claimed_unit = self._claim_unit(
                        platform_target,
                        num_of_physical_cores_needed=num_of_physical_cores_needed,
                        collector=collector,
                        exclude=busy_hosts,
                        lnc_config=lnc_config,
                    )
                    rotate_on_patience = True
                    if isinstance(claimed_unit, HostGroup):
                        # A group rotates on patience only while another eligible unit is untried in
                        # this pass; the last untried unit joins instead of re-cycling until the deadline.
                        eligible = self._eligible_host_aliases(
                            platform_target, num_of_physical_cores_needed, lnc_config=lnc_config
                        )
                        rotate_on_patience = bool(eligible - busy_hosts - {claimed_unit.host_id})
                    execution_host = self._get_target(claimed_unit, rotate_on_patience=rotate_on_patience)

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
                # Dequeue any entry the attempt left behind (e.g. soft-joined before a failed
                # upload or an unsupported execute); a no-op after the cores were released.
                if execution_host is not None:
                    try:
                        execution_host.abandon_queue()
                    except Exception as abandon_error:  # noqa: BLE001
                        logging.warning(f"Abandoning core queue entries failed: {abandon_error}")
                # Release by the claimed unit, not the host connection: building the
                # connection can fail after the claim succeeded.
                if claimed_unit is not None:
                    self.state_store.release(
                        claimed_unit.host_id,
                        num_of_physical_cores_needed,
                        member_ids=claimed_unit.member_ids,
                    )

        if not success:
            # Distinct from "No available hosts" and "Connection error after N attempts":
            # the fleet was reachable but every matching host stayed busy past the deadline.
            raise InferenceException(
                f"Core allocation deadline ({deadline_seconds}s) exceeded after rotating across hosts; "
                f"all matching hosts remained busy. "
                f"Patience rotations: {rotation_count}. Hosts attempted: {', '.join(attempted_hosts)}"
            )
