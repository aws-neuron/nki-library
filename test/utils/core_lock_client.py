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
Lock client for executing lock operations on remote hosts over SSH.

Provides flock-guarded execution of lock_helpers.py commands on remote hosts
via fabric2.Connection.
"""

from __future__ import annotations

import functools
import json
import logging
import uuid
from pathlib import Path

import fabric2

from .scripts.remote_lock_scripts import LockResult, LockStatus

logger = logging.getLogger(__name__)

# Remote paths for lock files
REMOTE_LOCK_DIR = "/tmp/neuronx-cc/core_locks"
REMOTE_VERSION_JSON = f"{REMOTE_LOCK_DIR}/infra_version.json"
REMOTE_LOCKS_JSON = f"{REMOTE_LOCK_DIR}/locks.json"
REMOTE_FLOCK_FILE = f"{REMOTE_LOCK_DIR}/atomic_lock"
REMOTE_LOCK_HELPERS = f"{REMOTE_LOCK_DIR}/lock_helpers.py"

# Default locking protocol version for hosts without version file
DEFAULT_LOCKING_PROTOCOL_VERSION = 2

# JSON key in version file
MIN_CLIENT_VERSION_KEY = "minClientLockingVersion"

# Default timeouts
DEFAULT_LOCK_TIMEOUT_SECONDS = 60
FLOCK_TIMEOUT_SECONDS = 30

# Path to the local helper script that gets deployed to remote hosts
_LOCAL_LOCK_HELPERS_PATH = Path(__file__).parent / "scripts" / "remote_lock_scripts.py"


@functools.lru_cache(maxsize=1)
def get_lock_helpers_content() -> str:
    """Load the lock helpers script content from the local filesystem."""
    try:
        return _LOCAL_LOCK_HELPERS_PATH.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Lock helpers script not found at {_LOCAL_LOCK_HELPERS_PATH}. "
            f"Ensure {_LOCAL_LOCK_HELPERS_PATH.name} is present in the same directory."
        )


# =============================================================================
# SSH operations
# =============================================================================


def initialize(conn: fabric2.Connection) -> None:
    """Create the lock directory and flock file on the remote host.

    Args:
        conn: Active fabric2 Connection to the remote host
    """
    init_script = f"""
        mkdir -p -m 777 {REMOTE_LOCK_DIR}
        touch {REMOTE_FLOCK_FILE}
        chmod 666 {REMOTE_FLOCK_FILE} 2>/dev/null || true
    """
    result = conn.run(init_script, warn=True, hide=True)
    if result.failed:
        raise RuntimeError(f"[{conn.host}] Failed to initialize lock directory: {result.stderr}")


def flock_execute(conn: fabric2.Connection, script: str) -> fabric2.Result:
    """Execute a script inside flock for atomicity.

    Uses the same flock file as both CoreLockManager and the host patcher,
    so all callers are mutually exclusive.

    Args:
        conn: Active fabric2 Connection to the remote host
        script: Bash script to execute while holding the lock

    Returns:
        fabric2.Result from the script execution
    """
    full_script = f"""flock -w {FLOCK_TIMEOUT_SECONDS} {REMOTE_FLOCK_FILE} bash << 'FLOCK_SCRIPT'
{script}
FLOCK_SCRIPT"""
    result = conn.run(full_script, warn=True, hide=True)

    if result.return_code == LockStatus.FLOCK_TIMEOUT.exit_code:
        logger.warning(f"[{conn.host}] flock timed out after {FLOCK_TIMEOUT_SECONDS}s")

    return result


def deploy_lock_helpers(conn: fabric2.Connection, script_content: str | None = None) -> None:
    """Deploy the lock helper script to remote host atomically under flock.

    Args:
        conn: Active fabric2 Connection to the remote host
        script_content: Script content to deploy. If None, reads from local filesystem.
    """
    if script_content is None:
        script_content = get_lock_helpers_content()

    deploy_script = f"""cat > {REMOTE_LOCK_HELPERS} << 'LOCK_HELPERS_EOF'
{script_content}
LOCK_HELPERS_EOF"""
    result = flock_execute(conn, deploy_script)
    if result.failed:
        raise RuntimeError(f"[{conn.host}] Failed to deploy lock helpers: {result.stderr}")
    logger.info(f"[{conn.host}] Deployed lock helpers to {REMOTE_LOCK_HELPERS}")


def run_lock_helper(conn: fabric2.Connection, command: str, *args: str | int) -> LockResult:
    """Run a lock helper CLI command atomically under flock.

    Calls the deployed lock_helpers.py CLI with --result-file, reads back
    the JSON result.

    Args:
        conn: Active fabric2 Connection to the remote host
        command: CLI subcommand (acquire, release, drain, undrain)
        *args: Arguments to pass to the subcommand

    Returns:
        LockResult parsed from the result file or exit code

    Raises:
        RuntimeError: If flock times out or result file cannot be read
    """
    result_file = f"{REMOTE_LOCK_DIR}/result_{uuid.uuid4().hex}.json"
    args_str = " ".join(str(a) for a in args)

    exec_result = flock_execute(
        conn,
        f"python3 {REMOTE_LOCK_HELPERS} {command} --result-file {result_file} {args_str}",
    )

    rc = exec_result.return_code
    status = LockStatus.from_exit_code(rc)

    if status == LockStatus.FLOCK_TIMEOUT:
        conn.run(f"rm -f {result_file}", warn=True, hide=True)
        raise RuntimeError(f"[{conn.host}] flock timeout for '{command}'")

    if status is None:
        conn.run(f"rm -f {result_file}", warn=True, hide=True)
        raise RuntimeError(f"[{conn.host}] Unknown exit code {rc} from '{command}'")

    # ALLOCATED and DRAINED need the result file for extra data (cores / max_lock_expiry)
    if status in (LockStatus.ALLOCATED, LockStatus.DRAINED):
        read_result = conn.run(f"cat {result_file} && rm -f {result_file}", warn=True, hide=True)
        if read_result.failed or not read_result.stdout.strip():
            conn.run(f"rm -f {result_file}", warn=True, hide=True)
            raise RuntimeError(f"[{conn.host}] Failed to read result for '{command}': {exec_result.stderr}")
        try:
            return LockResult.from_json(read_result.stdout.strip())
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"[{conn.host}] Failed to parse result for '{command}': {e}")

    # Other statuses (RELEASED, UNDRAINED, NO_CORES, DRAINING, ERROR): clean up and return
    conn.run(f"rm -f {result_file}", warn=True, hide=True)
    logger.info(f"[{conn.host}] Lock helper '{command}' status={status}")
    return LockResult(status=status)


# =============================================================================
# Version management
# =============================================================================


def get_host_locking_version(conn: fabric2.Connection) -> int:
    """Get the locking protocol version from the host, creating the file if missing.

    For hosts managed by host patcher, this reads the version set by the patcher.
    For hosts without a version file, creates one with the default version.

    Args:
        conn: Active fabric2 Connection to the remote host

    Returns:
        The locking protocol version (1 = mkdir, 2 = flock+JSON)
    """
    check_result = conn.run(f"test -f {REMOTE_VERSION_JSON}", warn=True, hide=True)

    if check_result.ok:
        result = conn.run(f"cat {REMOTE_VERSION_JSON}", warn=True, hide=True)
        if result.failed:
            raise RuntimeError(f"[{conn.host}] Failed to read {REMOTE_VERSION_JSON}: {result.stderr}")

        try:
            data = json.loads(result.stdout.strip())
            version = data.get(MIN_CLIENT_VERSION_KEY)
            if version is not None:
                return version
            logger.warning(f"[{conn.host}] Version file missing {MIN_CLIENT_VERSION_KEY} key, will recreate")
        except json.JSONDecodeError:
            logger.warning(f"[{conn.host}] Corrupted infra_version.json, will recreate")

    logger.info(f"[{conn.host}] Creating infra_version.json with version {DEFAULT_LOCKING_PROTOCOL_VERSION}")
    data = json.dumps({MIN_CLIENT_VERSION_KEY: DEFAULT_LOCKING_PROTOCOL_VERSION})
    write_result = conn.run(
        f"mkdir -p {REMOTE_LOCK_DIR} && echo '{data}' > {REMOTE_VERSION_JSON}", warn=True, hide=True
    )
    if write_result.failed:
        raise RuntimeError(f"[{conn.host}] Failed to create {REMOTE_VERSION_JSON}: {write_result.stderr}")

    return DEFAULT_LOCKING_PROTOCOL_VERSION


# =============================================================================
# Lock operations API
# =============================================================================


def acquire(
    conn: fabric2.Connection,
    total_physical_cores: int,
    num_physical_cores: int,
    timeout_seconds: int,
    version: int,
) -> LockResult:
    """Acquire physical cores on a remote host.

    Args:
        conn: Active fabric2 Connection to the remote host
        total_physical_cores: Total physical cores on the host
        num_physical_cores: Number of physical cores to acquire
        timeout_seconds: How long the lock should be held before auto-expiring
        version: Lock protocol version

    Returns:
        LockResult with ALLOCATED status and cores, or NO_CORES/DRAINING/ERROR
    """
    return run_lock_helper(
        conn, "acquire", REMOTE_LOCKS_JSON, total_physical_cores, num_physical_cores, timeout_seconds, version
    )


def release(conn: fabric2.Connection, core_ids: list[int], version: int) -> LockResult:
    """Release previously acquired physical cores.

    Args:
        conn: Active fabric2 Connection to the remote host
        core_ids: List of physical core IDs to release
        version: Lock protocol version

    Returns:
        LockResult with RELEASED status
    """
    return run_lock_helper(conn, "release", REMOTE_LOCKS_JSON, version, *core_ids)


def drain(conn: fabric2.Connection, timeout_seconds: int, version: int) -> LockResult:
    """Enable drain mode to block new test acquisitions.

    Args:
        conn: Active fabric2 Connection to the remote host
        timeout_seconds: How long drain should last after existing locks expire
        version: Lock protocol version

    Returns:
        LockResult with DRAINED status and max_lock_expiry
    """
    return run_lock_helper(conn, "drain", REMOTE_LOCKS_JSON, timeout_seconds, version)


def undrain(conn: fabric2.Connection, version: int) -> LockResult:
    """Disable drain mode to allow new test acquisitions.

    Args:
        conn: Active fabric2 Connection to the remote host
        version: Lock protocol version

    Returns:
        LockResult with UNDRAINED status
    """
    return run_lock_helper(conn, "undrain", REMOTE_LOCKS_JSON, version)
