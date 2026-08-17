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
from pathlib import Path
from typing import TYPE_CHECKING

# fabric2 is imported lazily (see _is_fabric_connection)
if TYPE_CHECKING:
    import fabric2

# Redundant aliases on the timing constants mark them as intentional re-exports
# so the client reads identical values without redefining them. The deployed
# helper remains the single source of truth for these timings.
from .scripts.remote_lock_scripts import (  # noqa: F401
    COMMIT_WINDOW as COMMIT_WINDOW,
)
from .scripts.remote_lock_scripts import (  # noqa: F401
    POLL_JITTER_MAX as POLL_JITTER_MAX,
)
from .scripts.remote_lock_scripts import (  # noqa: F401
    POLL_PERIOD as POLL_PERIOD,
)
from .scripts.remote_lock_scripts import (  # noqa: F401
    STALE_THRESHOLD as STALE_THRESHOLD,
)
from .scripts.remote_lock_scripts import (  # noqa: F401
    LockResult,
    LockStatus,
)

logger = logging.getLogger(__name__)


def _is_fabric_connection(executor: object) -> bool:
    """True if `executor` is a ``fabric2.Connection``.

    Imports fabric2 lazily so this module stays importable where fabric2 is
    absent.
    """
    # TODO: remove Connection support once host patcher uses RemoteExecutor
    try:
        import fabric2
    except ImportError:
        return False
    return isinstance(executor, fabric2.Connection)


# Remote paths for lock files
REMOTE_LOCK_DIR = "/tmp/neuronx-cc/core_locks"
REMOTE_VERSION_JSON = f"{REMOTE_LOCK_DIR}/infra_version.json"
REMOTE_LOCKS_JSON = f"{REMOTE_LOCK_DIR}/locks.json"
REMOTE_FLOCK_FILE = f"{REMOTE_LOCK_DIR}/atomic_lock"
REMOTE_LOCK_HELPERS = f"{REMOTE_LOCK_DIR}/lock_helpers.py"

# Default locking protocol version for hosts without version file
DEFAULT_LOCKING_PROTOCOL_VERSION = 5

# Build-time guard: this constant and the deployed v5 queue verbs must move
# together. Fail fast at import if the constant drifts from the protocol the
# verbs implement, rather than silently regressing hosts.
assert DEFAULT_LOCKING_PROTOCOL_VERSION == 5, "locking protocol constant must stay pinned to v5"

# JSON key in version file
MIN_CLIENT_VERSION_KEY = "minClientLockingVersion"

# Default timeouts
# Uniform hard-cap hold window: a lock is never renewed, so this
# doubles as the truthful worst-case release time. The on-core capture is wrapped
# in a shorter shell ``timeout`` so the cap never truncates a real in-progress
# capture.
DEFAULT_LOCK_TIMEOUT_SECONDS = 60

# Path to the local helper script that gets deployed to remote hosts
_LOCAL_LOCK_HELPERS_PATH = Path(__file__).parent / "scripts" / "remote_lock_scripts.py"


@functools.lru_cache(maxsize=1)
def get_lock_helpers_content() -> str:
    """Load the lock helpers script content from the local filesystem."""
    try:
        return _LOCAL_LOCK_HELPERS_PATH.read_text()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Lock helpers script not found at {_LOCAL_LOCK_HELPERS_PATH}. "
            f"Ensure {_LOCAL_LOCK_HELPERS_PATH.name} is present in the same directory."
        ) from e


# =============================================================================
# Remote functions — executed on the remote host via RemoteExecutor.call_function().
# Must be self-contained: all imports inside the body, no closures.
# =============================================================================


def _remote_initialize(
    lock_dir, lock_file, version_file, helpers_file, default_version_json, helpers_content, flock_timeout=30
):
    """Create lock directory, read/create version file, deploy helper script under flock."""
    import fcntl  # noqa: E401
    import json
    import os
    import random
    import re
    import time

    def _parse_script_version(text):
        # Best-effort parse of SCRIPT_VERSION from helper source; None if absent.
        m = re.search(r"SCRIPT_VERSION\s*=\s*(\d+)", text)
        return int(m.group(1)) if m else None

    os.makedirs(lock_dir, mode=0o777, exist_ok=True)
    open(lock_file, "a").close()
    try:
        os.chmod(lock_file, 0o666)
    except OSError:
        pass
    try:
        with open(version_file) as f:
            version_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        version_data = json.loads(default_version_json)
        with open(version_file, "w") as f:
            json.dump(version_data, f)
    deadline = time.monotonic() + flock_timeout
    with open(lock_file, "a") as lf:
        while True:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"flock timed out after {flock_timeout}s") from e
                time.sleep(0.1 + random.uniform(0, 0.025))
        try:
            # Version-gate the deploy under the held flock (read + decide + write
            # atomically, no new TOCTOU). Only a strictly-newer client may
            # overwrite; an equal-or-newer deployed helper is preserved so a
            # stale client cannot downgrade verbs out from under newer clients.
            client_version = _parse_script_version(helpers_content)
            deployed_version = None
            try:
                with open(helpers_file) as hf:
                    deployed_version = _parse_script_version(hf.read())
            except (FileNotFoundError, OSError):
                deployed_version = None
            should_write = deployed_version is None or (
                client_version is not None and client_version > deployed_version
            )
            if should_write:
                with open(helpers_file, "w") as hf:
                    hf.write(helpers_content)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return version_data


def _remote_lock_operation(lock_file, helpers_file, command, args, kwargs=None, flock_timeout=30):
    """Load lock_helpers module and call a function under flock."""
    import fcntl  # noqa: E401
    import importlib.util
    import random
    import sys
    import time

    spec = importlib.util.spec_from_file_location("lock_helpers", helpers_file)
    assert spec is not None and spec.loader is not None
    lh = importlib.util.module_from_spec(spec)
    sys.modules["lock_helpers"] = lh
    spec.loader.exec_module(lh)
    fn = getattr(lh, command)
    if kwargs is None:
        kwargs = {}
    deadline = time.monotonic() + flock_timeout
    with open(lock_file, "a") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"flock timed out after {flock_timeout}s") from e
                time.sleep(0.1 + random.uniform(0, 0.025))
        try:
            lr = fn(*args, **kwargs)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    result = {
        "status": lr.status.value,
        "cores": lr.cores,
        "message": lr.message,
        "expiry": lr.expiry,
        "max_lock_expiry": lr.max_lock_expiry,
        "position": lr.position,
        "worst_case_eta": lr.worst_case_eta,
        "bumped": lr.bumped,
        "re_enqueued": lr.re_enqueued,
        "should_reset_cores": lr.should_reset_cores,
    }
    return result


# =============================================================================
# SSH operations (used by v1 locking path)
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

    # Recreating the version file is a silent-regression risk: a wiped /tmp could
    # otherwise be rebuilt at an unexpected version. The downgrade race is accepted
    # as-is -- a concurrent recreate by another client is logged,
    # not prevented. Log at warning so the recreate is observable in test output.
    logger.warning(f"[{conn.host}] Recreating infra_version.json with version {DEFAULT_LOCKING_PROTOCOL_VERSION}")
    data = json.dumps({MIN_CLIENT_VERSION_KEY: DEFAULT_LOCKING_PROTOCOL_VERSION})
    write_result = conn.run(
        f"mkdir -p {REMOTE_LOCK_DIR} && echo '{data}' > {REMOTE_VERSION_JSON}", warn=True, hide=True
    )
    if write_result.failed:
        raise RuntimeError(f"[{conn.host}] Failed to create {REMOTE_VERSION_JSON}: {write_result.stderr}")

    return DEFAULT_LOCKING_PROTOCOL_VERSION


# =============================================================================
# RemoteExecutor operations
# =============================================================================


def initialize_and_deploy(executor) -> int:
    """Initialize lock dir, deploy helpers, and get version in a single remote call.

    Combines initialize(), deploy_lock_helpers(), and get_host_locking_version()
    into one RemoteExecutor.call_function() round-trip.

    Args:
        executor: RemoteExecutor instance

    Returns:
        Host locking protocol version
    """
    version_data = executor.call_function(
        _remote_initialize,
        lock_dir=REMOTE_LOCK_DIR,
        lock_file=REMOTE_FLOCK_FILE,
        version_file=REMOTE_VERSION_JSON,
        helpers_file=REMOTE_LOCK_HELPERS,
        default_version_json=json.dumps({MIN_CLIENT_VERSION_KEY: DEFAULT_LOCKING_PROTOCOL_VERSION}),
        helpers_content=get_lock_helpers_content(),
    )
    return version_data.get(MIN_CLIENT_VERSION_KEY, DEFAULT_LOCKING_PROTOCOL_VERSION)


def _run_lock_helper(executor, command: str, *args, **kwargs) -> LockResult:
    """Run a lock helper command via RemoteExecutor."""
    resp = executor.call_function(
        _remote_lock_operation,
        lock_file=REMOTE_FLOCK_FILE,
        helpers_file=REMOTE_LOCK_HELPERS,
        command=command,
        args=list(args),
        kwargs=kwargs if kwargs else None,
    )
    return LockResult(
        status=LockStatus[resp["status"]],
        cores=resp.get("cores"),
        message=resp.get("message"),
        expiry=resp.get("expiry"),
        max_lock_expiry=resp.get("max_lock_expiry"),
        position=resp.get("position"),
        worst_case_eta=resp.get("worst_case_eta"),
        bumped=resp.get("bumped", False),
        re_enqueued=resp.get("re_enqueued", False),
        should_reset_cores=resp.get("should_reset_cores", True),
    )


def release(
    executor, core_ids: list[int], version: int, caller_id: str | None = None, expected_expiry: int | None = None
) -> LockResult:
    """Release previously acquired physical cores."""
    return _run_lock_helper(
        executor,
        "release",
        REMOTE_LOCKS_JSON,
        core_ids,
        version,
        caller_id=caller_id,
        expected_expiry=expected_expiry,
    )


def poll(
    executor,
    total_physical_cores: int,
    num_physical_cores: int,
    timeout_seconds: int,
    version: int,
    entry_id: str,
    ready: bool,
    lnc_config: int,
    caller_id: str | None = None,
) -> LockResult:
    """Poll the FIFO core-allocation queue (fast-path / enqueue / commit)."""
    return _run_lock_helper(
        executor,
        "poll",
        REMOTE_LOCKS_JSON,
        total_physical_cores,
        num_physical_cores,
        timeout_seconds,
        version,
        entry_id,
        ready,
        lnc_config,
        caller_id=caller_id,
    )


def dequeue(
    executor,
    total_physical_cores: int,
    version: int,
    entry_id: str,
    caller_id: str | None = None,
) -> LockResult:
    """Gracefully remove an entry from the FIFO core-allocation queue."""
    return _run_lock_helper(
        executor,
        "dequeue",
        REMOTE_LOCKS_JSON,
        total_physical_cores,
        version,
        entry_id,
        caller_id=caller_id,
    )


def probe(
    executor,
    total_physical_cores: int,
    num_physical_cores: int,
    timeout_seconds: int,
    version: int,
) -> LockResult:
    """Read-only worst-case ETA peek (no enqueue / no mutation)."""
    return _run_lock_helper(
        executor,
        "probe",
        REMOTE_LOCKS_JSON,
        total_physical_cores,
        num_physical_cores,
        timeout_seconds,
        version,
    )


def _conn_run_lock_helper(conn, command, *args):
    """Fallback for host patcher: run lock helper via connection.run() when RemoteExecutor is unavailable."""
    import uuid

    result_file = f"{REMOTE_LOCK_DIR}/result_{uuid.uuid4().hex}.json"
    args_str = " ".join(str(a) for a in args)
    cmd = f"flock -w 30 {REMOTE_FLOCK_FILE} python3 {REMOTE_LOCK_HELPERS} {command} --result-file {result_file} {args_str}"
    exec_result = conn.run(cmd, warn=True, hide=True)
    rc = exec_result.return_code
    status = LockStatus.from_exit_code(rc)
    if status in (LockStatus.ALLOCATED, LockStatus.DRAINED, LockStatus.ERROR):
        read_result = conn.run(f"cat {result_file} && rm -f {result_file}", warn=True, hide=True)
        if read_result.ok and read_result.stdout.strip():
            return LockResult.from_json(read_result.stdout.strip())
    conn.run(f"rm -f {result_file}", warn=True, hide=True)
    return LockResult(status=status) if status else LockResult(status=LockStatus.ERROR, message=f"exit code {rc}")


def is_draining(executor, version: int) -> bool:
    """Check if the remote host is currently in drain mode.

    Args:
        executor: RemoteExecutor instance

    Returns:
        True if the host is draining, False otherwise
    """
    result = _run_lock_helper(executor, "is_host_draining", REMOTE_LOCKS_JSON, version)
    return result.status == LockStatus.DRAINING


def drain(executor, timeout_seconds: int, version: int) -> LockResult:
    """Enable drain mode. Accepts executor or fabric2.Connection (for host patcher compat)."""
    if _is_fabric_connection(executor):
        return _conn_run_lock_helper(executor, "drain", REMOTE_LOCKS_JSON, timeout_seconds, version)
    return _run_lock_helper(executor, "drain", REMOTE_LOCKS_JSON, timeout_seconds, version)


def undrain(executor, version: int) -> LockResult:
    """Disable drain mode. Accepts executor or fabric2.Connection (for host patcher compat)."""
    if _is_fabric_connection(executor):
        return _conn_run_lock_helper(executor, "undrain", REMOTE_LOCKS_JSON, version)
    return _run_lock_helper(executor, "undrain", REMOTE_LOCKS_JSON, version)


def initialize(executor) -> None:
    """Create lock dir, deploy helpers, read version. Accepts executor or Connection (host patcher compat)."""
    if _is_fabric_connection(executor):
        conn = executor
        conn.run(
            f"mkdir -p -m 777 {REMOTE_LOCK_DIR} && touch {REMOTE_FLOCK_FILE} && chmod 666 {REMOTE_FLOCK_FILE} 2>/dev/null || true",
            warn=True,
            hide=True,
        )
        script = get_lock_helpers_content()
        conn.run(
            f"flock -w 30 {REMOTE_FLOCK_FILE} bash -c 'cat > {REMOTE_LOCK_HELPERS}' << 'LOCK_HELPERS_EOF'\n{script}\nLOCK_HELPERS_EOF",
            warn=True,
            hide=True,
        )
        return
    initialize_and_deploy(executor)


def deploy_lock_helpers(conn: fabric2.Connection) -> None:
    # TODO: remove once host patcher uses RemoteExecutor (initialize() already deploys)
    pass
