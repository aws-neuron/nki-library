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
import json
import logging
import os
import random
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, final

import fabric2
from attr import dataclass
from filelock import FileLock
from paramiko import SSHException
from typing_extensions import override

from .common_dataclasses import INF_ARTIFACT_DIR_NAME, NeuronDeviceInfo, Platforms, TargetHost
from .core_lock_manager import check_infra_locking_version
from .exceptions import (
    InferenceException,
    LocalExecutionException,
    NoNeuronDevicesException,
    RemoteExecutionException,
    TimeoutException,
    UnimplementedException,
)
from .metrics_collector import IMetricsCollector, MetricName
from .resources import RemoteDirectory
from .s3_utils import S3ArtifactUploadConfig


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


class Host(ABC):
    @abstractmethod
    def execute_command(
        self,
        command: str,
        target_directory: str,
        collector: IMetricsCollector,
        collective_ranks: int,
        lnc_config: int,
        do_copy_artifacts: bool = False,
        get_list_of_files_to_copy: Callable[[str], list[str]] | None = None,
    ) -> str | None:
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
        poll_period_seconds: int = 5,
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
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        raise UnimplementedException()

    @abstractmethod
    def get_host_id(self) -> str:
        raise UnimplementedException()

    def __lock__(self, lock_file_path: str, timeout_seconds: int):
        return FileLock(f"{lock_file_path}.lock", timeout=timeout_seconds * 1000)


@final
class LocalHost(Host):
    def __init__(self, local_neuron_installation_path: str, host_id: str):
        super().__init__()
        self.neuron_ls_path: str = os.path.join(local_neuron_installation_path, "neuron-ls")
        self.host_id: str = host_id

    @override
    def get_host_id(self) -> str:
        return self.host_id

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
    ) -> str | None:
        pass

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
        poll_period_seconds: int = 5,
    ) -> Generator[CoreAllocation, None, None]:
        # TODO: this needs to be fixed
        yield CoreAllocation(
            host_id=self.host_id, logical_core_ids=list(range(collective_ranks)), lnc_config=lnc_config
        )

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        result = subprocess.run([self.neuron_ls_path, "--json-output"])
        data = json.loads(result.stdout)
        if not data:
            raise NoNeuronDevicesException("localhost")
        return [NeuronDeviceInfo.from_dict(device) for device in data]


@final
class SshHost(Host):
    # Default timeout for core locks (in minutes)
    DEFAULT_CORE_LOCK_TIMEOUT_MINUTES = 1

    def __init__(
        self,
        ssh_alias: str,
        test_base_path: str,
        remote_neuron_install_dir: str,
        run_id: str,
        ssh_config_path: str,
        s3_config: S3ArtifactUploadConfig,
        remote_base_path: str = "/tmp/neuronx-cc/tests",
    ):
        super().__init__()
        self.ssh_alias: str = ssh_alias
        self.run_id: str = run_id
        self.s3_config = s3_config

        config_overrides = {"run": {"in_stream": False, "warn": True, "pty": True}}

        self.connection: fabric2.Connection = fabric2.Connection(
            host=ssh_alias,
            config=fabric2.Config(
                runtime_ssh_path=ssh_config_path,
                overrides=config_overrides,
            ),
        )
        self.remote_base_path: str = remote_base_path
        self.remote_full_path: str | None = None

        self.lock_file_path: str = os.path.join(test_base_path, self.ssh_alias)
        self.core_allocation_file: str = os.path.join(test_base_path, f"{self.ssh_alias}_core_allocation.json")
        self.remote_lock_dir: str = "/tmp/neuronx-cc/core_locks"
        self._lock_system_initialized: bool = False

        self.neuron_ls_path: str = os.path.join(remote_neuron_install_dir, "neuron-ls")

    @override
    def get_host_id(self) -> str:
        return self.ssh_alias

    @staticmethod
    def __get_unique_collectives_port__(core_id: int) -> int:
        """Generate unique port for collectives coordination based on first allocated core.

        This ensures parallel tests don't conflict on the same port.
        E.g., test on cores [8, 9] gets port 61242, test on cores [16, 17] gets port 61250.
        """
        return 61234 + core_id

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
    ) -> str | None:
        assert self.remote_full_path is not None, "You have to prepare host first!"

        logging.info(f"Attempting to run {command=} on {self.ssh_alias=}")

        # Common env var for every run
        common_env_var = "export NEURON_RT_ENABLE_OCP=1 && export NEURON_RT_ENABLE_OCP_SATURATION=1 && export NEURON_RT_DEBUG_OUTPUT_DIR=\"$(pwd)/debug_output\""

        with self.get_core_allocation(
            collective_ranks=collective_ranks, lnc_config=lnc_config, collector=collector
        ) as core_allocation:
            unique_port = self.__get_unique_collectives_port__(core_allocation.logical_core_ids[0])

            # This env var sets which core will be used for this command
            env_var = (
                f"{common_env_var}"
                f" && export NEURON_RT_VISIBLE_CORES={core_allocation.get_core_list_str()}"
                f" && export NEURON_LOGICAL_NC_CONFIG={lnc_config}"
                # NEURON_RT_ROOT_COMM_ID is needed for collectives coordination (when collective_ranks > 1)
                f" && export NEURON_RT_ROOT_COMM_ID=localhost:{unique_port}"
            )
            full_command = self.inside_venv(f"set -o pipefail; cd {self.remote_full_path} && {env_var} && {command}")
            logging.info(f"Executing remote command: {full_command}")

            result: fabric2.Result = self.connection.run(full_command)

            if result.failed:
                # Log PATH on remote host to help debug missing tools issues
                path_result = self.connection.run("echo DIAGNOSTIC: PATH=$PATH", warn=True)
                logging.warning(
                    f"Remote PATH on {self.ssh_alias}: {path_result.stdout.strip() if path_result.ok else 'FAILED TO GET PATH'}"
                )

                raise RemoteExecutionException(f"Unable to execute {command} in {self.remote_full_path}", result)

        if do_copy_artifacts:
            local_download_location = os.path.join(target_directory, INF_ARTIFACT_DIR_NAME)
            # Clean up any existing infer_result directory from previous failed/retry attempts
            shutil.rmtree(local_download_location, ignore_errors=True)

            return self.__download_artifacts__(
                remote_path=self.remote_full_path,
                local_path=local_download_location,
                list_of_files_to_copy=(get_list_of_files_to_copy(result.stdout) if get_list_of_files_to_copy else None),
                collector=collector,
            )
        else:
            return None

    def __download_artifacts__(
        self,
        remote_path: str,
        local_path: str,
        collector: IMetricsCollector,
        list_of_files_to_copy: list[str] | None = None,
    ):
        with collector.timer(MetricName.FILE_TRANSFER_TIME):
            return RemoteDirectory(remote_path, self.connection).download(
                destination_dir_path=local_path,
                s3_config=self.s3_config,
                list_of_files=list_of_files_to_copy,
                collector=collector,
            )

    def __cleanup_remote_paths__(self, *remote_path_list: str, base_exception: Exception | None = None):
        exceptions: list[Exception] = [base_exception] if base_exception else []
        for remote_path in remote_path_list:
            try:
                self.connection.run(f"rm -rf {remote_path}")
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
        max_retries: int = 3,
    ):
        # Add PID to make remote path unique per process
        # allows multiple machines to run the same test on the same host
        pid = os.getpid()
        test_base_dir = f"{os.path.basename(target_directory)}_pid{pid}"
        remote_full_path = os.path.join(self.remote_base_path, test_base_dir)

        remote_dir = RemoteDirectory(remote_full_path, self.connection)

        for attempt in range(max_retries):
            try:
                with collector.timer(MetricName.FILE_TRANSFER_TIME):
                    remote_dir.upload(
                        target_directory, collector, self.s3_config, force_local_cleanup=force_local_cleanup
                    )

                self.__install_prerequisites__(remote_path=remote_full_path)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = (2**attempt) + random.uniform(0, 1)
                logging.warning(
                    f"prepare_host failed on {self.ssh_alias} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                # Clean up partial upload before retry
                remote_dir.cleanup()
                time.sleep(delay)

        self.remote_full_path = remote_full_path

        try:
            yield remote_full_path
        finally:
            if not skip_remote_cleanup:
                remote_dir.cleanup()

    def __install_prerequisites__(self, remote_path: str):
        result: fabric2.Result = self.connection.run(f"python3 -m venv {remote_path}/.venv")
        if result.failed:
            raise RemoteExecutionException(
                f"Unable to initialize python virtual env at {remote_path}/.venv",
                result,
            )

    def inside_venv(self, command):
        assert self.remote_full_path
        return f"source {self.remote_full_path}/.venv/bin/activate && {command}"

    def __run_with_retry__(self, command: str, max_retries: int = 5, base_delay: float = 1.0) -> fabric2.Result:
        """
        Execute SSH command with exponential backoff retry logic to avoid SSH rate limiting.
        """
        import random

        for attempt in range(max_retries):
            try:
                logging.info(f"Executing remote command (attempt {attempt + 1}): {command}")
                result = self.connection.run(command)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)

        raise Exception("Retry logic failed unexpectedly")

    def __initialize_remote_lock_dir__(self):
        """
        Initialize remote core lock directory on remote host.
        Should only be called once per SshHost instance.
        """
        logging.info(f"Initializing remote lock directory {self.remote_lock_dir} on {self.ssh_alias}")
        # Create with global permissions so any user can use the shared lock directory
        # Also create flock file for coordinating concurrent lock operations
        result = self.connection.run(
            f"mkdir -p -m 777 {self.remote_lock_dir} && touch {self.remote_lock_dir}/atomic_lock && chmod 666 {self.remote_lock_dir}/atomic_lock 2>/dev/null || true",
            warn=True,
        )
        if result.failed:
            raise RemoteExecutionException(f"Failed to create remote lock directory {self.remote_lock_dir}", result)

        # Check infrastructure version compatibility before any locking operations
        check_infra_locking_version(self.connection)

        self.__cleanup_stale_remote_locks__()

    def __cleanup_stale_remote_locks__(self):
        """
        Clean up stale remote locks based on per-lock timeout.
        """
        logging.info(f"Cleaning up stale locks on {self.ssh_alias}")
        # Loops through all lock directories, extracts timeout from name, deletes if stale
        cleanup_script = f"""
        for lock_dir in {self.remote_lock_dir}/physical_lock_*_timeout_*m; do
            [ -d "$lock_dir" ] || continue
            timeout=$(echo "$lock_dir" | sed 's/.*_timeout_\\([0-9]*\\)m/\\1/')
            if [ -n "$timeout" ]; then
                find "$lock_dir" -maxdepth 0 -mmin +$timeout -delete 2>/dev/null || true
            fi
        done
        """
        self.connection.run(cleanup_script, warn=True)

    def __flock_run__(self, script: str) -> str | None:
        """Run script inside flock for atomicity. Returns stdout on success, None on failure."""
        result = self.connection.run(
            f"flock -w 30 {self.remote_lock_dir}/atomic_lock -c '{script}'",
            warn=True,
        )
        if result.ok:
            return result.stdout
        logging.warning(f"flock command failed on {self.ssh_alias}: exit={result.return_code}")
        return None

    def __try_lock_remote_physical_cores__(self, physical_core_ids: list[int]) -> list[int]:
        """
        Try to lock multiple physical cores in a single SSH command.
        Returns list of successfully locked physical core IDs.

        Timeout is encoded in directory name for coordination between processes.
        Locks are per physical core to prevent conflicts between LNC1 and LNC2 tests.
        """
        if not physical_core_ids:
            return []

        logging.info(f"Trying to lock physical cores {physical_core_ids} on {self.ssh_alias}")
        # Atomic lock acquisition via mkdir - prints core_id on success
        # Use 777 so any user can delete stale locks
        lock_cmds = " ; ".join(
            f"mkdir -m 777 {self.remote_lock_dir}/physical_lock_{core_id}_timeout_{self.DEFAULT_CORE_LOCK_TIMEOUT_MINUTES}m 2>/dev/null && echo {core_id}"
            for core_id in physical_core_ids
        )
        stdout = self.__flock_run__(lock_cmds)
        if stdout is None:
            return []
        locked = [int(x) for x in stdout.strip().split('\n') if x.strip().isdigit()]
        if locked:
            logging.info(f"Locked physical cores {locked} on {self.ssh_alias}")
        return locked

    def __unlock_remote_physical_cores__(self, physical_core_ids: list[int]):
        """Release remote locks for physical cores. Retries to ensure locks are removed."""
        if not physical_core_ids:
            return
        lock_dirs = " ".join(
            f"{self.remote_lock_dir}/physical_lock_{core_id}_timeout_{self.DEFAULT_CORE_LOCK_TIMEOUT_MINUTES}m"
            for core_id in physical_core_ids
        )
        logging.info(f"Releasing physical cores {physical_core_ids} on {self.ssh_alias}")
        for attempt in range(5):
            result = self.__flock_run__(f"rmdir {lock_dirs} 2>/dev/null || true")
            if result is not None:
                return
            delay = (2**attempt) + random.uniform(0, 1)
            logging.warning(
                f"Failed to release physical cores {physical_core_ids} on {self.ssh_alias} "
                f"(attempt {attempt + 1}/5). Retrying in {delay:.1f}s"
            )
            time.sleep(delay)
        logging.error(f"Failed to release physical cores {physical_core_ids} on {self.ssh_alias} after 5 attempts")

    def __initialize_core_allocation_file__(self):
        """
        Initialize the core allocation state file with run_id tracking.
        Creates new file or resets if run_id doesn't match (stale from previous run).
        """
        lock_file = FileLock(f"{self.core_allocation_file}.lock", timeout=10000)
        with lock_file.acquire():
            needs_reset = False

            if os.path.isfile(self.core_allocation_file):
                # File exists - check if it's from current run
                with open(self.core_allocation_file, "r") as f:
                    try:
                        existing_state = json.load(f)
                        if existing_state.get("run_id") != self.run_id:
                            needs_reset = True
                    except (json.JSONDecodeError, KeyError):
                        needs_reset = True
            else:
                needs_reset = True

            if needs_reset:
                # Get device info to build initial state
                devices = self.get_neuron_device_info()

                # Count total physical cores available across devices
                total_physical_cores = 0
                lnc_mode: int | None = None
                for device in devices:
                    if lnc_mode is None:
                        lnc_mode = device.logical_neuroncore_config
                    else:
                        assert (
                            lnc_mode == device.logical_neuroncore_config
                        ), f"Not all attached neuron devices have the same lnc config: {lnc_mode} vs {device.logical_neuroncore_config}"
                    # neuron-ls returns logical cores, multiply by lnc_mode to get physical
                    total_physical_cores += len(device.neuroncore_ids) * lnc_mode

                core_state = {
                    "physical_cores_in_use": 0,
                    "total_physical_cores": total_physical_cores,
                    "run_id": self.run_id,
                }

                with open(self.core_allocation_file, "w") as f:
                    json.dump(core_state, f)

    @staticmethod
    def __physical_to_logical_cores__(physical_core_ids: list[int], lnc_config: int) -> list[int]:
        """Convert contiguous physical core IDs to contiguous logical core IDs.

        Physical cores must be contiguous (NEURON_RT_VISIBLE_CORES requires it) and
        aligned to lnc_config boundary.

        Examples:
            LNC2 (lnc_config=2): physical [0,1,2,3] -> logical [0,1]
            LNC1 (lnc_config=1): physical [0,1,2] -> logical [0,1,2]
        """
        first_logical = physical_core_ids[0] // lnc_config
        num_logical = len(physical_core_ids) // lnc_config
        # Verify physical cores are contiguous and aligned
        expected = list(range(first_logical * lnc_config, (first_logical + num_logical) * lnc_config))
        assert (
            physical_core_ids == expected
        ), f"Physical cores must be contiguous and aligned (lnc_config={lnc_config}): expected {expected}, got {physical_core_ids}"
        return list(range(first_logical, first_logical + num_logical))

    def __allocate_cores__(self, num_logical_cores: int, lnc_config: int) -> tuple[list[int], list[int]]:
        """
        Allocate logical cores by locking physical cores.

        Locking is done on physical cores to prevent conflicts between LNC1 and LNC2 tests.
        For example, LNC2 logical core 0 maps to physical cores [0,1], while LNC1 logical
        cores 0-1 also map to physical cores [0,1]. By locking physical cores, we ensure
        these tests don't run simultaneously on the same hardware.

        Args:
            num_logical_cores: Number of logical cores to allocate
            lnc_config: LNC configuration (1 or 2) - physical cores per logical core

        Returns:
            Tuple of (logical_core_ids, physical_core_ids) or ([], []) if not enough available.
            Physical cores are locked via mkdir; logical cores are for NEURON_RT_VISIBLE_CORES.
        """
        num_physical_cores = num_logical_cores * lnc_config

        lock_file = FileLock(f"{self.core_allocation_file}.lock", timeout=10000)
        with lock_file.acquire():
            with open(self.core_allocation_file, "r+") as f:
                core_state = json.load(f)

                total_physical_cores = core_state["total_physical_cores"]
                available_physical = total_physical_cores - core_state["physical_cores_in_use"]
                if available_physical < num_physical_cores:
                    logging.info(f"Not enough cores available: need {num_physical_cores}, have {available_physical}")
                    return [], []  # Not enough cores

                allocated_physical_cores = self.__try_allocate_physical_cores__(
                    total_physical_cores, num_physical_cores, lnc_config
                )

                # Update local count only if we got enough cores
                if len(allocated_physical_cores) == num_physical_cores:
                    core_state["physical_cores_in_use"] += num_physical_cores

                    # Write back
                    f.seek(0)
                    f.truncate()
                    json.dump(core_state, f)

                    # Convert physical cores to logical cores for NEURON_RT_VISIBLE_CORES
                    logical_core_ids = self.__physical_to_logical_cores__(allocated_physical_cores, lnc_config)

                    return logical_core_ids, allocated_physical_cores
                else:
                    return [], []

    def __try_allocate_physical_cores__(
        self, total_physical_cores: int, num_physical_cores: int, lnc_config: int
    ) -> list[int]:
        """Try to lock contiguous physical cores via remote mkdir locks.

        Returns list of locked physical core IDs, or empty list if unable to lock.
        """
        # Collectives require start positions aligned to num_physical_cores.
        # Since num_physical_cores is always a multiple of lnc_config, aligning to
        # num_physical_cores automatically satisfies the lnc_config alignment requirement.
        # Example: 128 physical cores, 4 collective ranks with LNC2 (num_physical_cores=8)
        #   start_positions = [0, 8, 16, 24, ...] -> shuffled to e.g. [16, 0, 24, 8, ...]
        # Example: 128 physical cores, 4 collective ranks with LNC1 (num_physical_cores=4)
        #   start_positions = [0, 4, 8, 12, ...] -> shuffled to e.g. [8, 0, 12, 4, ...]
        assert (
            num_physical_cores % lnc_config == 0
        ), f"num_physical_cores ({num_physical_cores}) must be a multiple of lnc_config ({lnc_config})"
        num_start_positions = total_physical_cores - num_physical_cores + 1
        start_positions = list(range(0, num_start_positions, num_physical_cores))
        random.shuffle(start_positions)

        for start_pcore in start_positions:
            # Try to lock all physical cores in this contiguous range
            pcores_to_lock = list(range(start_pcore, start_pcore + num_physical_cores))
            locked = self.__try_lock_remote_physical_cores__(pcores_to_lock)

            if len(locked) == num_physical_cores:
                return locked
            else:
                # Failed to lock all cores, release what we got
                self.__unlock_remote_physical_cores__(locked)

        return []

    def __release_cores__(self, physical_core_ids: list[int]):
        """
        Release previously allocated physical cores in both local state and remote locks.
        """
        # Update local state FIRST so concurrent __allocate_cores__ callers see
        # the cores as available.  The remote locks are still held at this point,
        # so any concurrent attempt to mkdir the same cores will safely fail and
        # retry until the remote release below completes.
        lock_file = FileLock(f"{self.core_allocation_file}.lock", timeout=10000)
        with lock_file.acquire():
            with open(self.core_allocation_file, "r+") as f:
                core_state = json.load(f)

                # Decrement count (tracked in physical cores)
                core_state["physical_cores_in_use"] -= len(physical_core_ids)

                # Write back
                f.seek(0)
                f.truncate()
                json.dump(core_state, f)

        # Release remote physical core locks
        self.__unlock_remote_physical_cores__(physical_core_ids)

    @override
    @contextlib.contextmanager
    def get_core_allocation(
        self,
        collector: IMetricsCollector,
        collective_ranks: int = 1,
        lnc_config: int = 2,
        timeout_seconds: int = 9000,
        poll_period_seconds: int = 5,
    ) -> Generator[CoreAllocation, None, None]:
        """
        Allocate logical cores for execution by locking physical cores.

        Physical cores are locked to prevent conflicts between LNC1 and LNC2 tests.
        Logical core IDs are returned for use with NEURON_RT_VISIBLE_CORES.
        """
        with collector.timer(MetricName.CORE_ALLOCATION_TIME):
            # Initialize lock system if needed
            if not self._lock_system_initialized:
                self.__initialize_remote_lock_dir__()
                self.__initialize_core_allocation_file__()
                self._lock_system_initialized = True

            logging.info(
                f"Trying to lock {collective_ranks} logical cores (lnc_config={lnc_config}) on {self.ssh_alias}"
            )

            current_time = time.time()
            done_at_time = current_time + timeout_seconds
            allocated_logical_cores: list[int] = []
            allocated_physical_cores: list[int] = []
            last_cleanup_time = current_time

            # Cleanup interval matches minimum lock timeout
            cleanup_interval_seconds = self.DEFAULT_CORE_LOCK_TIMEOUT_MINUTES * 60

            while current_time <= done_at_time:
                allocated_logical_cores, allocated_physical_cores = self.__allocate_cores__(
                    collective_ranks, lnc_config
                )
                if allocated_logical_cores:
                    break

                if current_time - last_cleanup_time >= cleanup_interval_seconds:
                    # Cleanup stale locks
                    self.__cleanup_stale_remote_locks__()
                    last_cleanup_time = current_time

                jitter = random.uniform(0, 0.5)
                time.sleep(poll_period_seconds + jitter)
                current_time = time.time()

            if not allocated_logical_cores:
                raise TimeoutException(
                    f"Unable to allocate {collective_ranks} logical cores on {self.ssh_alias} within {timeout_seconds} seconds"
                )

            logging.info(f"Allocated logical cores {allocated_logical_cores} (physical: {allocated_physical_cores})")
        try:
            yield CoreAllocation(
                host_id=self.ssh_alias, logical_core_ids=allocated_logical_cores, lnc_config=lnc_config
            )
        finally:
            # Always release physical cores, even on exception
            self.__release_cores__(allocated_physical_cores)

    @override
    def get_neuron_device_info(self) -> list[NeuronDeviceInfo]:
        """Get Neuron device information from remote host with retry logic."""
        result: fabric2.Result = self.__run_with_retry__(
            f"NEURON_LOGICAL_NC_CONFIG={NeuronDeviceInfo.logical_neuroncore_config} {self.neuron_ls_path} --json-output",
        )
        if result.failed:
            raise RemoteExecutionException(f"Unable to find neuron device on {self.ssh_alias}", result)
        data = json.loads(result.stdout)
        if not data:
            raise NoNeuronDevicesException(self.ssh_alias)
        return [NeuronDeviceInfo.from_dict(device) for device in data]


@dataclass
class HostInfo:
    host_alias: str
    work_queue_depth: int
    run_id: str
    host_type: str

    def to_json(self):
        return {
            "host_alias": self.host_alias,
            "work_queue_depth": self.work_queue_depth,
            "run_id": self.run_id,
            "host_type": self.host_type,
        }

    @classmethod
    def from_json(cls, input: Any):
        return HostInfo(
            input["host_alias"],
            input["work_queue_depth"],
            input.get("run_id", ""),
            input.get("host_type", ""),
        )


@final
class HostManager:
    def __init__(
        self,
        base_host_info_path: str,
        target_hosts: list[TargetHost],
        neuron_installation_path: str,
        ssh_config_path: str,
        default_platform_target: Platforms,
        s3_config: S3ArtifactUploadConfig | None = None,
    ) -> None:
        self.run_id = str(os.getppid())
        self.ssh_config_path = ssh_config_path
        self.s3_config = s3_config or S3ArtifactUploadConfig()
        self.target_hosts, self.host_types = self.__derive_hosts__(
            target_hosts,
            neuron_installation_path=neuron_installation_path,
            base_host_info_path=base_host_info_path,
            default_platform_target=default_platform_target,
        )
        self.is_local: bool = len(target_hosts) < 1
        self.host_info_path: str = os.path.join(base_host_info_path, "host_stats.json")
        self.failed_hosts: set[str] = set()  # Track hosts that have timed out

    def __derive_hosts__(
        self,
        target_hosts: list[TargetHost],
        neuron_installation_path: str,
        base_host_info_path: str,
        default_platform_target: Platforms,
    ) -> tuple[dict[str, Host], dict[str, Platforms]]:
        if len(target_hosts) == 0:
            # TODO: Using default_platform_target for localhost is a workaround because we
            # cannot yet query the local host type. Replace with host type discovery when available.
            local_host_id = "localhost"
            return (
                {local_host_id: LocalHost(neuron_installation_path, local_host_id)},
                {local_host_id: default_platform_target},
            )
        else:
            hosts: dict[str, Host] = dict()
            host_types: dict[str, Platforms] = dict()

            for target_host in target_hosts:
                # Note: ssh_alias format must match fetch_shared_fleet_metadata.sh which has to derive the same alias from the JSON
                ssh_alias = target_host.ssh_host
                hosts[ssh_alias] = SshHost(
                    ssh_alias,
                    test_base_path=base_host_info_path,
                    remote_neuron_install_dir=neuron_installation_path,
                    run_id=self.run_id,
                    ssh_config_path=self.ssh_config_path,
                    s3_config=self.s3_config,
                )
                host_types[ssh_alias] = target_host.host_type

            return hosts, host_types

    def __construct_lock_file_name(self):
        return f"{self.host_info_path}.lock"

    def __create_lockfile__(self, timeout_seconds: int):
        return FileLock(self.__construct_lock_file_name(), timeout=timeout_seconds * 1000)

    @contextlib.contextmanager
    def __read_host_file__(self) -> Generator[list[HostInfo], list[HostInfo], None]:
        with self.__create_lockfile__(10).acquire():
            with open(self.host_info_path, "r+") as fp:
                j = json.load(fp)

                assert isinstance(j, list)

                hosts = [HostInfo.from_json(json_host_info) for json_host_info in j]

                yield hosts

                # Seek to beginning and truncate before writing
                _ = fp.seek(0)
                _ = fp.truncate()
                # Convert HostInfo objects to dictionaries for JSON serialization
                json.dump(
                    [h.to_json() for h in hosts],
                    fp,
                )

    def initialize_host_stats(self):
        # Each pytest worker tries to create this file, but only one needs to.
        # All others use the file created by the first to reach this point.
        # If file exists from a previous run (different run_id), reset all work_queue_depth to 0.
        with self.__create_lockfile__(10).acquire():
            if not os.path.isfile(self.host_info_path):
                with open(self.host_info_path, "w+") as fp:
                    hosts_info = [
                        HostInfo(
                            host_alias,
                            work_queue_depth=0,
                            run_id=self.run_id,
                            host_type=self.host_types.get(host_alias, Platforms.TRN2).value,
                        )
                        for host_alias in self.target_hosts.keys()
                    ]
                    # randomly shuffle hosts, so that different test suites don't always hammer the
                    # same hosts first
                    random.shuffle(hosts_info)
                    json.dump([h.to_json() for h in hosts_info], fp)
            else:
                # File exists - validate hosts and check run_id
                with open(self.host_info_path, "r+") as fp:
                    existing_host_infos = [HostInfo.from_json(h) for h in json.load(fp)]
                    existing_hosts = {h.host_alias for h in existing_host_infos}
                    current_hosts = set(self.target_hosts.keys())

                    # Check if run_id matches
                    needs_reset = False
                    if existing_host_infos and existing_host_infos[0].run_id != self.run_id:
                        needs_reset = True

                    # Reset if hosts changed or stale run_id
                    if existing_hosts != current_hosts or needs_reset:
                        _ = fp.seek(0)
                        _ = fp.truncate()
                        hosts_info = [
                            HostInfo(
                                host_alias,
                                work_queue_depth=0,
                                run_id=self.run_id,
                                host_type=self.host_types.get(host_alias, Platforms.TRN2).value,
                            )
                            for host_alias in current_hosts
                        ]
                        json.dump([h.to_json() for h in hosts_info], fp)

    def mark_host_as_failed(self, host_id: str):
        """Mark a host as failed to exclude it from future assignments."""
        self.failed_hosts.add(host_id)
        logging.warning(
            f"Host {host_id} marked as failed. Failed hosts: {len(self.failed_hosts)}/{len(self.target_hosts)}"
        )

    def get_failed_host_count(self) -> int:
        """Get the number of hosts that have failed during this run."""
        return len(self.failed_hosts)

    def __get_host_assignment__(self, platform_target: Platforms, timeout_seconds: int = 10) -> Host:
        generator = self.__read_host_file__()
        with generator as hosts:
            # Filter out failed hosts and hosts that don't match the platform target
            available_hosts = [
                h for h in hosts if h.host_alias not in self.failed_hosts and h.host_type == platform_target.value
            ]

            if not available_hosts:
                matching_hosts = [h for h in hosts if h.host_type == platform_target.value]
                if not matching_hosts:
                    raise Exception(f"No hosts available for platform {platform_target.value}")
                raise Exception(
                    f"No available hosts for platform {platform_target.value} - "
                    f"all {len(matching_hosts)} matching hosts have failed"
                )

            asc_host_infos = sorted(
                available_hosts,
                key=lambda host: host.work_queue_depth,
            )

            # Pick randomly from top 3 hosts with lowest queue depth
            top_n = min(3, len(asc_host_infos))
            selected_host_info = random.choice(asc_host_infos[:top_n])

            selected_host_info.work_queue_depth += 1
            host = self.target_hosts[selected_host_info.host_alias]

            return host

    def release_host(self, host: Host):
        host_id = host.get_host_id()

        with self.__read_host_file__() as hosts:
            for h in hosts:
                if h.host_alias == host_id:
                    h.work_queue_depth -= 1
                    break

    def get_host_assignment_with_retry(
        self,
        platform_target: Platforms,
        collector: IMetricsCollector,
        max_retries: int = 3,
    ):
        """Execute a function with automatic retry on different hosts if connection times out."""
        # Track errors from each host attempt for better debugging
        host_errors: dict[str, str] = {}

        def format_host_errors() -> str:
            """Format all captured host errors for the exception message."""
            return "\n".join(f"  - {host}: {error}" for host, error in host_errors.items())

        # in case code block that's yielded to by context_manager_wrapper does not directly return
        # make sure that we record successes and terminate retries
        success = False

        def succeeded():
            nonlocal success
            success = True

        @contextlib.contextmanager
        def context_manager_wrapper(notify_success: Callable[[], None], execution_host: Host, attempt: int):
            try:
                yield execution_host
            except (OSError, TimeoutError, SSHException) as e:
                host_id = execution_host.get_host_id() if execution_host else "unknown"
                error_msg = f"{type(e).__name__}: {e}"
                host_errors[host_id] = error_msg

                logging.error(f"Connection error on host {host_id}, attempt {attempt + 1}/{max_retries}: {e}")
                self.mark_host_as_failed(host_id)

                if collector:
                    collector.record_metric(
                        MetricName.FAILED_HOSTS_COUNT,
                        float(self.get_failed_host_count()),
                        "Count",
                    )

                if attempt == max_retries - 1:
                    error_details = format_host_errors()
                    raise InferenceException(
                        f"Connection error after {attempt + 1} attempts. "
                        f"Hosts attempted: {', '.join(attempted_hosts)}\n"
                        f"Errors from each host:\n{error_details}"
                    ) from e

                logging.warning(f"Retrying on different host (attempt {attempt + 2}/{max_retries})")
            else:
                notify_success()

        attempted_hosts = []

        for attempt in range(max_retries):
            if success:
                break

            execution_host = None

            try:
                if collector:
                    with collector.timer(MetricName.HOST_LOCK_TIME):
                        execution_host = self.__get_host_assignment__(platform_target)
                else:
                    execution_host = self.__get_host_assignment__(platform_target)

                host_id = execution_host.get_host_id()
                attempted_hosts.append(host_id)

                yield context_manager_wrapper(succeeded, execution_host, attempt)

            except Exception as e:
                # Report error with details from previous failures
                if host_errors and not isinstance(e, InferenceException):
                    error_details = format_host_errors()
                    raise InferenceException(f"{e}\n\nErrors from previous host attempts:\n{error_details}") from e
                raise
            finally:
                if execution_host:
                    self.release_host(execution_host)
