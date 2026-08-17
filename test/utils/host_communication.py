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
"""Transport-agnostic host communication.

A ``HostCommunication`` is just the wire to a machine: ``run`` a command,
``directory`` for file I/O, ``open_channel`` for streaming, and enter/exit. All
workload logic (core locking, env setup, venv, neuron-ls, artifacts) lives on
``Host`` and is expressed in terms of ``run``/``directory`` so it is written once
regardless of backend. Three backends:

- ``LocalCommunication``    — runs locally via subprocess (no SSH, no transfer).
- ``ParamikoCommunication`` — fabric2/paramiko, one connection per worker. The
  default shared-fleet transport; behaviour identical to the pre-refactor code.
- ``SshCommunication``      — native ``ssh`` subprocess per command, stateless,
  for exclusive sessions.

The three are interchangeable behind the ``HostCommunication`` surface; the only
backend-specific knob ``Host`` needs is ``wrap_command`` (the remote venv-activate
prefix vs. local identity) and whether transfer is local (``is_local``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import fabric2
import paramiko
from attr import dataclass
from invoke.exceptions import CommandTimedOut

from .core_lock_client import DEFAULT_LOCK_TIMEOUT_SECONDS
from .exceptions import LocalExecutionException, TimeoutException

if TYPE_CHECKING:
    from .host_io import HostDirectory
    from .metrics_collector import IMetricsCollector
    from .s3_utils import S3ArtifactUploadConfig

# Default per-command budget for workload (inference) commands. Reuses the
# lock-lease constant so the SSH budget stays coupled to the declared core-hold
# window; callers with a larger per-command hold pass an explicit ``timeout=``.
SSH_INFERENCE_TIMEOUT_SECONDS = DEFAULT_LOCK_TIMEOUT_SECONDS  # = 60


class SshChannelOpenError(Exception):
    """Raised when a session channel cannot be opened on the connection."""


@dataclass(frozen=True)
class CommandResult:
    """Backend-neutral stand-in for ``fabric2.Result``.

    Exposes exactly the attributes the call sites read (``stdout``/``stderr``/
    ``return_code``/``ok``/``failed``) so a native-backend result is
    interchangeable with a ``fabric2.Result`` without callers type-checking.
    """

    command: str
    return_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.return_code == 0

    @property
    def failed(self) -> bool:
        return self.return_code != 0

    @property
    def exited(self) -> int:
        # fabric2.Result alias; kept for defensive parity.
        return self.return_code


class SshChannel(ABC):
    """A single exec session over a connection, for binary streaming.

    The method surface matches ``paramiko.Channel`` exactly so both backends can
    satisfy it with minimal adaptation. Used by RemoteExecutor (persistent
    JSON-RPC server) and the remote HostDirectory I/O (tar-pipe upload/download).
    """

    @abstractmethod
    def recv(self, n: int) -> bytes:
        """Return up to ``n`` bytes from the channel; ``b""`` at EOF."""
        raise NotImplementedError

    @abstractmethod
    def sendall(self, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown_write(self) -> None:
        """Signal EOF to the remote command's stdin."""
        raise NotImplementedError

    @abstractmethod
    def recv_exit_status(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class _ParamikoChannel(SshChannel):
    """Adapts a ``paramiko.Channel`` to the ``SshChannel`` interface."""

    def __init__(self, channel: paramiko.Channel) -> None:
        self._chan = channel

    def recv(self, n: int) -> bytes:
        return self._chan.recv(n)

    def sendall(self, data: bytes) -> None:
        self._chan.sendall(data)

    def shutdown_write(self) -> None:
        self._chan.shutdown_write()

    def recv_exit_status(self) -> int:
        return self._chan.recv_exit_status()

    def close(self) -> None:
        self._chan.close()


class HostCommunication(ABC):
    """Transport. Connection is established lazily on first use and reused;
    ``__enter__`` may warm it eagerly, ``__exit__`` releases it.

    Every ``run`` is bounded by a per-command timeout so a wedged host surfaces
    as a catchable ``TimeoutException`` instead of hanging the caller:

    - ``WORKLOAD_COMMAND_TIMEOUT`` — budget for workload commands, derived from
      the custom ETA/hold window the caller declared (default: the lock lease).
    - ``CONTROL_COMMAND_TIMEOUT`` — shorter budget for setup verbs (mkdir, rm,
      venv, neuron-ls, ...); defaults to half the workload budget.
    """

    #: Whether file I/O is on the same machine (no upload/download needed).
    is_local: bool = False

    #: Default timeout for short control commands (mkdir, rm, venv, neuron-ls).
    CONTROL_COMMAND_TIMEOUT: float = SSH_INFERENCE_TIMEOUT_SECONDS / 2

    #: Default timeout for workload commands (inference, long-running ops).
    WORKLOAD_COMMAND_TIMEOUT: float = SSH_INFERENCE_TIMEOUT_SECONDS

    def __init__(
        self,
        *,
        max_timeout_seconds: float = SSH_INFERENCE_TIMEOUT_SECONDS,
        control_timeout_seconds: float | None = None,
    ) -> None:
        self._max_timeout = max_timeout_seconds
        self._control_timeout = (
            control_timeout_seconds if control_timeout_seconds is not None else max_timeout_seconds / 2
        )
        # Update instance-level constants to match constructor overrides.
        self.WORKLOAD_COMMAND_TIMEOUT = self._max_timeout
        self.CONTROL_COMMAND_TIMEOUT = self._control_timeout

    def _effective_timeout(self, timeout: float | None) -> float:
        """Resolve the per-command budget: explicit > control default."""
        if timeout is not None:
            return timeout
        return self._control_timeout

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        hide: bool = False,
        warn: bool = True,
        timeout: float | None = None,
    ):
        """Run ``command`` to completion and return a Result-like object
        (``stdout``/``stderr``/``return_code``/``ok``/``failed``). ``warn=True``
        means a non-zero exit does NOT raise. The command is bounded by
        ``timeout``; defaults to ``CONTROL_COMMAND_TIMEOUT`` when not given.
        Exceeding it raises ``TimeoutException``."""
        raise NotImplementedError

    @abstractmethod
    def directory(self, path: str, collector: "IMetricsCollector") -> HostDirectory:
        """Return the HostDirectory for file I/O at ``path`` on this host."""
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        """Ensure the underlying connection is established (idempotent)."""
        raise NotImplementedError

    @abstractmethod
    def open_channel(self, command: str) -> SshChannel:
        """Open a fresh session channel with ``command`` already exec'd on it."""
        raise NotImplementedError

    @abstractmethod
    def is_active(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reconnect(self) -> None:
        """Drop and re-establish the connection (used by Host's retry path)."""
        raise NotImplementedError

    @abstractmethod
    def get_host_id(self) -> str:
        raise NotImplementedError

    def wrap_command(self, working_dir: str, command: str) -> str:
        """Wrap a command for execution in ``working_dir`` (e.g. venv activation).
        Default: identity (local). Remote backends prepend venv activation."""
        return command

    @abstractmethod
    def __enter__(self) -> HostCommunication:
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, *exc) -> None:
        raise NotImplementedError


class LocalCommunication(HostCommunication):
    """Runs commands locally via subprocess. No SSH, no remote transfer."""

    is_local = True

    def __init__(
        self,
        host_id: str = "localhost",
        *,
        max_timeout_seconds: float = SSH_INFERENCE_TIMEOUT_SECONDS,
        control_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(max_timeout_seconds=max_timeout_seconds, control_timeout_seconds=control_timeout_seconds)
        self._host_id = host_id

    def run(
        self,
        command: str,
        *,
        hide: bool = False,
        warn: bool = True,
        timeout: float | None = None,
    ) -> CommandResult:
        eff = self._effective_timeout(timeout)
        # bash (not the shell=True /bin/sh default) — the workload commands use
        # bash-isms (`set -o pipefail`, `source`).
        try:
            proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=eff)
        except subprocess.TimeoutExpired as e:
            raise TimeoutException(f"[{self._host_id}] command exceeded {eff}s and was aborted: {command[:120]}") from e
        result = CommandResult(command=command, return_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if not warn and result.failed:
            raise LocalExecutionException(f"[local] command failed ({proc.returncode}): {command}\n{proc.stderr}", proc)
        return result

    def directory(self, path: str, collector: "IMetricsCollector") -> HostDirectory:
        from .host_io import LocalHostIO  # noqa: PLC0415

        return LocalHostIO(path, collector)

    def open(self) -> None:
        return None

    def open_channel(self, command: str) -> SshChannel:
        raise SshChannelOpenError("LocalCommunication does not support channels")

    def is_active(self) -> bool:
        return True

    def reconnect(self) -> None:
        return None

    def get_host_id(self) -> str:
        return self._host_id

    def __enter__(self) -> LocalCommunication:
        return self

    def __exit__(self, *exc) -> None:
        return None


_SSH_CONNECT_TIMEOUT_SECONDS = 10


class ParamikoCommunication(HostCommunication):
    """fabric2/paramiko transport — one connection per worker (the default
    shared-fleet path). Behaviour-identical to the pre-refactor FabricSshClient."""

    # SSH keepalive interval. paramiko probes the peer every N seconds so a host
    # that wedges or is silently recycled mid-operation is detected (within ~2
    # intervals) and surfaces as an SSHException inside a blocked run(), instead
    # of hanging the caller indefinitely.
    SSH_KEEPALIVE_SECONDS = 15

    def __init__(
        self,
        ssh_config_path: str,
        ssh_alias: str,
        *,
        s3_config: "S3ArtifactUploadConfig | None" = None,
        max_timeout_seconds: float = SSH_INFERENCE_TIMEOUT_SECONDS,
        control_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(max_timeout_seconds=max_timeout_seconds, control_timeout_seconds=control_timeout_seconds)
        self._ssh_config_path = ssh_config_path
        self._ssh_alias = ssh_alias
        self._s3_config = s3_config
        self._connection: fabric2.Connection | None = None

    def _conn(self) -> fabric2.Connection:
        if self._connection is None:
            config_overrides = {"run": {"in_stream": False, "warn": True, "pty": True}}
            self._connection = fabric2.Connection(
                host=self._ssh_alias,
                connect_timeout=_SSH_CONNECT_TIMEOUT_SECONDS,
                config=fabric2.Config(runtime_ssh_path=self._ssh_config_path, overrides=config_overrides),
            )
        return self._connection

    def _apply_keepalive(self) -> None:
        """Open the connection if needed and enable SSH keepalive.

        Keepalive makes paramiko probe the peer every ``SSH_KEEPALIVE_SECONDS``,
        so a host that wedges or is silently recycled mid-operation surfaces as
        an SSHException inside a blocked ``run()`` (within ~2 intervals) instead
        of hanging the caller indefinitely.
        """
        conn = self._conn()
        if not conn.is_connected:
            conn.open()
        client = getattr(conn, "client", None)
        transport = client.get_transport() if client is not None else None
        if transport is not None:
            transport.set_keepalive(ParamikoCommunication.SSH_KEEPALIVE_SECONDS)

    def run(
        self,
        command: str,
        *,
        hide: bool = False,
        warn: bool = True,
        timeout: float | None = None,
    ) -> fabric2.Result:
        eff = self._effective_timeout(timeout)
        self._apply_keepalive()
        try:
            return self._conn().run(command, hide=hide, warn=warn, timeout=eff)
        except CommandTimedOut as e:
            # Translated so the host-rotation loop treats a wedged host as a
            # connection failure and rotates away rather than hanging the test.
            raise TimeoutException(
                f"[{self.get_host_id()}] remote command exceeded {eff}s and was aborted: {command[:120]}"
            ) from e

    def directory(self, path: str, collector: "IMetricsCollector") -> HostDirectory:
        from .host_io import RemoteParamikoIO, RemoteS3IO  # noqa: PLC0415

        if self._s3_config is not None and self._s3_config.is_enabled():
            return RemoteS3IO(self, path, self._s3_config, collector)
        return RemoteParamikoIO(self, path, collector)

    def open(self) -> None:
        self._conn().open()
        self._apply_keepalive()

    def open_channel(self, command: str) -> SshChannel:
        conn = self._conn()
        if not self.is_active():
            conn.open()
        transport = conn.transport
        try:
            channel = transport.open_session()
            channel.exec_command(command)
        except paramiko.SSHException as e:
            raise SshChannelOpenError(f"Failed to open channel for: {command}") from e
        return _ParamikoChannel(channel)

    def is_active(self) -> bool:
        transport = self._conn().transport
        return transport is not None and transport.is_active()

    def reconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection.open()
            self._apply_keepalive()

    def get_host_id(self) -> str:
        return self._ssh_alias

    def wrap_command(self, working_dir: str, command: str) -> str:
        return f"source {working_dir}/.venv/bin/activate && {command}"

    def __enter__(self) -> ParamikoCommunication:
        self._conn().open()
        return self

    def __exit__(self, *exc) -> None:
        if self._connection is not None:
            self._connection.close()


class SshCommunication(HostCommunication):
    """Native ``ssh`` transport — stateless subprocess per command for exclusive
    sessions. Retries transient SSH transport failures with exponential backoff."""

    _SSH_TRANSPORT_FAILURE = 255
    _MAX_TRANSIENT_RETRIES = 4
    _TRANSIENT_PATTERNS = (
        "kex_exchange_identification",
        "Session open refused",
        "Connection closed",
        "Connection refused",
        "Connection timed out",
        "Connection reset",
        "broken pipe",
    )
    _KEEPALIVE_OPTS = (
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
    )

    def __init__(
        self,
        *,
        ssh_config_path: str,
        alias: str,
        s3_config: "S3ArtifactUploadConfig | None" = None,
        max_timeout_seconds: float = SSH_INFERENCE_TIMEOUT_SECONDS,
        control_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(max_timeout_seconds=max_timeout_seconds, control_timeout_seconds=control_timeout_seconds)
        self._ssh_config_path = ssh_config_path
        self._alias = alias
        self._s3_config = s3_config
        # Per-process ControlPath so each xdist worker multiplexes over its own
        # SSH connection instead of serializing through a single shared socket.
        self._control_path = f"/tmp/nkilib-cm-{os.getpid()}-%C"

    def _ssh_argv(self, command: str) -> list[str]:
        return [
            "ssh",
            "-F",
            self._ssh_config_path,
            *self._KEEPALIVE_OPTS,
            "-o",
            f"ControlPath={self._control_path}",
            "-T",
            self._alias,
            command,
        ]

    @classmethod
    def _is_transient(cls, returncode: int, stderr: str) -> bool:
        if returncode != cls._SSH_TRANSPORT_FAILURE:
            return False
        return any(p.lower() in stderr.lower() for p in cls._TRANSIENT_PATTERNS)

    def run(
        self,
        command: str,
        *,
        hide: bool = False,
        warn: bool = True,
        timeout: float | None = None,
    ) -> CommandResult:
        eff = self._effective_timeout(timeout)
        for attempt in range(self._MAX_TRANSIENT_RETRIES):
            try:
                proc = subprocess.run(self._ssh_argv(command), capture_output=True, text=True, timeout=eff)
            except subprocess.TimeoutExpired as e:
                raise TimeoutException(
                    f"[{self._alias}] remote command exceeded {eff}s and was aborted: {command[:120]}"
                ) from e
            if not self._is_transient(proc.returncode, proc.stderr):
                break
            logging.warning(
                f"[{self._alias}] transient SSH failure (attempt {attempt + 1}): {proc.stderr.strip()[:120]}"
            )
            time.sleep(0.5 * (2**attempt))
        result = CommandResult(command=command, return_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if not warn and result.failed:
            raise LocalExecutionException(
                f"[{self._alias}] command failed ({proc.returncode}): {command}\n{proc.stderr}", proc
            )
        return result

    def directory(self, path: str, collector: "IMetricsCollector") -> HostDirectory:
        from .host_io import RemoteS3IO, ScpIO  # noqa: PLC0415

        if self._s3_config is not None and self._s3_config.is_enabled():
            return RemoteS3IO(self, path, self._s3_config, collector)
        return ScpIO(self._ssh_config_path, self._alias, path, collector)

    def open_channel(self, command: str) -> SshChannel:
        raise NotImplementedError("SshCommunication does not support channels")

    def open(self) -> None:
        return None

    def is_active(self) -> bool:
        try:
            proc = subprocess.run(self._ssh_argv("true"), capture_output=True, timeout=_SSH_CONNECT_TIMEOUT_SECONDS)
            return proc.returncode == 0
        except Exception:
            return False

    def reconnect(self) -> None:
        return None

    def get_host_id(self) -> str:
        return self._alias

    def wrap_command(self, working_dir: str, command: str) -> str:
        return f"source {working_dir}/.venv/bin/activate && {command}"

    def __enter__(self) -> SshCommunication:
        return self

    def __exit__(self, *exc) -> None:
        return None
