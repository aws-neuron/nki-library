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
"""Persistent remote Python process for low-latency command execution over SSH.

Opens a single SSH channel running a Python server process. Subsequent calls
exchange newline-delimited JSON, avoiding the ~1.4s overhead of opening a new
channel per command.

Usage::

    executor = RemoteExecutor(connection)
    executor.call("ping")                                    # -> "ok"
    executor.call("shell", command="whoami")                 # -> {"returncode": 0, "stdout": "..."}
    executor.call("exec_python", code="result = 1 + 1")     # -> 2
    executor.close()
"""

import inspect
import json
import logging
import secrets
import shlex
from types import FunctionType
from typing import TYPE_CHECKING, Any

from .host_communication import SshChannelOpenError

if TYPE_CHECKING:
    from .host_communication import HostCommunication

logger = logging.getLogger(__name__)


def _server_main():
    """JSON-RPC server loop. Runs on the remote host, reads stdin, writes stdout."""
    import json  # noqa: E401
    import os
    import subprocess
    import sys
    import traceback

    token = os.environ.get("_EXECUTOR_TOKEN", "")

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid = req.get("id")
        if req.get("token") != token:
            resp = {"id": rid, "error": "invalid token"}
            json.dump(resp, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
            continue
        method = req.get("method")
        params = req.get("params", {})
        try:
            if method == "ping":
                resp = {"id": rid, "result": "ok"}
            elif method == "shell":
                r = subprocess.run(params["command"], shell=True, capture_output=True, text=True)
                resp = {"id": rid, "result": {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}}
            elif method == "exec_python":
                env: dict[str, Any] = {}
                exec(params["code"], env)
                resp = {"id": rid, "result": env.get("result")}
            elif method == "shutdown":
                json.dump({"id": rid, "result": "ok"}, sys.stdout)
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            else:
                resp = {"id": rid, "error": "unknown method: " + str(method)}
        except Exception:
            resp = {"id": rid, "error": traceback.format_exc()}
        json.dump(resp, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()


class RemoteExecutorError(Exception):
    """Raised when the remote server returns an error or the channel fails."""


class RemoteExecutor:
    """Persistent remote Python process for low-latency command execution over SSH."""

    def __init__(self, client: "HostCommunication"):
        """Start a persistent Python server on the remote host.

        Args:
            client: A HostCommunication (must be open or openable).
        """
        self._client = client
        self._next_id = 0
        self._channel = None
        self._explicitly_closed = False
        self._start()

    def _start(self):
        if not self._client.is_active():
            self._client.open()
        self._token = secrets.token_hex(16)
        server_code = inspect.getsource(_server_main) + "\n_server_main()\n"
        server_cmd = f"_EXECUTOR_TOKEN={self._token} python3 -u -c {shlex.quote(server_code)}"
        try:
            channel = self._client.open_channel(server_cmd)
        except SshChannelOpenError as e:
            raise RemoteExecutorError(f"Failed to open remote executor channel: {e}") from e
        self._channel = channel
        self._recv_buf = b""
        # Verify the server is alive
        resp = self.call("ping")
        if resp != "ok":
            # Null the channel so the next call self-heals cleanly instead of
            # reusing this half-started (live socket, bad server) channel.
            self._channel = None
            raise RemoteExecutorError(f"Remote executor ping failed: {resp}")

    def _recv_line(self) -> str:
        """Read one newline-terminated line from the channel using recv()."""
        assert self._channel is not None
        while b"\n" not in self._recv_buf:
            data = self._channel.recv(65536)
            if not data:
                raise RemoteExecutorError("Remote executor channel closed unexpectedly")
            self._recv_buf += data
        line, self._recv_buf = self._recv_buf.split(b"\n", 1)
        return line.decode()

    def call(self, method: str, **params) -> Any:
        """Send a JSON-RPC request and return the result.

        Args:
            method: Handler name ("ping", "shell", "exec_python", "shutdown").
            **params: Handler-specific parameters.

        Returns:
            The "result" field from the server response.

        Raises:
            RemoteExecutorError: On protocol or server-side errors.
        """
        # Distinguish an intentional shutdown from a crashed channel. close() sets
        # _explicitly_closed; a transient I/O failure only nulls _channel. In the
        # latter case self-heal by rebuilding the channel before serving this request
        # (we never silently retry the in-flight request that hit the failure).
        if self._channel is None:
            if self._explicitly_closed:
                raise RemoteExecutorError("Executor is closed")
            self._start()
        channel = self._channel
        assert channel is not None, "a started executor holds an open channel"
        req_id = self._next_id
        self._next_id += 1
        request = {"id": req_id, "token": self._token, "method": method, "params": params}
        try:
            channel.sendall((json.dumps(request) + "\n").encode())
            response_line = self._recv_line()
        except RemoteExecutorError:
            self._channel = None
            raise
        except Exception as e:
            self._channel = None
            raise RemoteExecutorError(f"Channel I/O failed: {e}") from e
        resp = json.loads(response_line)
        if "error" in resp:
            raise RemoteExecutorError(f"Remote error: {resp['error']}")
        return resp.get("result")

    def call_function(self, func: FunctionType, **kwargs) -> Any:
        """Send a Python function to execute remotely and return its result.

        The function's source is shipped via ``inspect.getsource`` and exec'd on
        the remote, so it must not close over local state. It may ``import``
        modules available on the remote (e.g. a previously deployed helper).

        Args:
            func: A Python function. Its source is sent via inspect.getsource().
            **kwargs: Arguments passed to the function.

        Returns:
            The return value of the function.
        """
        source = inspect.getsource(func)
        code = f"{source}\nresult = {func.__name__}(**{kwargs!r})\n"
        return self.call("exec_python", code=code)

    def close(self):
        """Shut down the remote server and close the channel."""
        # Mark closed first and unconditionally: a crashed executor (channel
        # already nulled by a prior I/O failure) must still be treated as
        # explicitly closed so a later call() does not silently resurrect it.
        self._explicitly_closed = True
        if self._channel is None:
            return
        try:
            self.call("shutdown")
        except Exception:
            pass
        try:
            self._channel.close()
        except Exception:
            pass
        self._channel = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
