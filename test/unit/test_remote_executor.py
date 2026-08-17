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
"""Unit tests for RemoteExecutor."""

import json
from unittest.mock import MagicMock

import pytest

from test.utils.remote_executor import RemoteExecutor, RemoteExecutorError

# A mis-stubbed channel can make _recv_line() loop forever accumulating data;
# bound every test so a regression fails fast instead of exhausting memory.
pytestmark = pytest.mark.timeout(60)


class FakeChannel:
    """Simulates a paramiko channel running the JSON-RPC server."""

    def __init__(self):
        self._inbox = []  # requests sent by client
        self._outbox = b""  # responses to be read by client
        self._closed = False

    def exec_command(self, cmd):
        pass

    def sendall(self, data):
        line = data.decode().strip()
        req = json.loads(line)
        try:
            resp = self._handle(req)
        except Exception as e:
            resp = {"id": req.get("id"), "error": str(e)}
        self._outbox += (json.dumps(resp) + "\n").encode()

    def recv(self, bufsize):
        if not self._outbox:
            return b""
        chunk = self._outbox[:bufsize]
        self._outbox = self._outbox[bufsize:]
        return chunk

    def close(self):
        self._closed = True

    def _handle(self, req):
        rid = req["id"]
        method = req["method"]
        params = req.get("params", {})
        if method == "ping":
            return {"id": rid, "result": "ok"}
        if method == "shell":
            return {"id": rid, "result": {"returncode": 0, "stdout": "hello\n", "stderr": ""}}
        if method == "exec_python":
            env = {}
            exec(params["code"], env)
            return {"id": rid, "result": env.get("result")}
        if method == "shutdown":
            return {"id": rid, "result": "ok"}
        return {"id": rid, "error": f"unknown method: {method}"}


def _make_executor():
    """Create a RemoteExecutor with a fake channel (no SSH)."""
    client = MagicMock()
    channel = FakeChannel()
    client.is_active.return_value = True
    client.open_channel.return_value = channel
    return RemoteExecutor(client), channel


def _add(a, b):
    return a + b


def _get_pid():
    import os

    return os.getpid()


class TestRemoteExecutor:
    def test_ping(self):
        executor, _ = _make_executor()
        assert executor.call("ping") == "ok"
        executor.close()

    def test_shell(self):
        executor, _ = _make_executor()
        result = executor.call("shell", command="echo hello")
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]
        executor.close()

    def test_exec_python(self):
        executor, _ = _make_executor()
        result = executor.call("exec_python", code="result = 1 + 2")
        assert result == 3
        executor.close()

    def test_call_function(self):
        executor, _ = _make_executor()
        assert executor.call_function(_add, a=3, b=4) == 7
        executor.close()

    def test_call_function_with_imports(self):
        executor, _ = _make_executor()
        result = executor.call_function(_get_pid)
        assert isinstance(result, int)
        executor.close()

    def test_remote_error_raises(self):
        executor, _ = _make_executor()
        with pytest.raises(RemoteExecutorError, match="division by zero"):
            executor.call("exec_python", code="result = 1 / 0")
        # executor channel is dead after error — but we can still close
        executor.close()

    def test_close_is_idempotent(self):
        executor, _ = _make_executor()
        executor.close()
        executor.close()  # should not raise

    def test_call_after_close_raises(self):
        executor, _ = _make_executor()
        executor.close()
        with pytest.raises(RemoteExecutorError, match="closed"):
            executor.call("ping")

    def test_context_manager(self):
        executor, _ = _make_executor()
        with executor as ex:
            assert ex.call("ping") == "ok"
        # should be closed now
        with pytest.raises(RemoteExecutorError, match="closed"):
            executor.call("ping")

    def test_unknown_method_raises(self):
        executor, _ = _make_executor()
        with pytest.raises(RemoteExecutorError, match="unknown method"):
            executor.call("nonexistent")

    def test_call_reconnects_after_channel_crash(self):
        """A crashed channel (set to None by a prior I/O failure, NOT close()) is
        transparently re-established on the next call()."""
        client = MagicMock()
        client.is_active.return_value = True
        # First open_channel builds the initial channel (during __init__/_start);
        # the second is the fresh channel after the simulated crash.
        client.open_channel.side_effect = [FakeChannel(), FakeChannel()]
        executor = RemoteExecutor(client)
        assert executor.call("ping") == "ok"
        open_channel_calls_before = client.open_channel.call_count

        # Simulate a transient reset: prior call's I/O failure set _channel = None
        # WITHOUT an explicit close().
        executor._channel = None

        # Next call should self-heal by opening a fresh channel and serving the request.
        assert executor.call("ping") == "ok"
        assert client.open_channel.call_count > open_channel_calls_before

    def test_call_after_close_does_not_reconnect(self):
        """Regression: explicit close() keeps call() raising; no self-heal."""
        executor, _ = _make_executor()
        executor.close()
        with pytest.raises(RemoteExecutorError, match="closed"):
            executor.call("ping")

    def test_close_on_crashed_channel_marks_explicitly_closed(self):
        """A crashed executor (channel nulled by a prior I/O failure) that is then
        close()d must stay closed: a later call() raises and does NOT resurrect a
        session. Guards the contract that close() marks _explicitly_closed even when
        the channel was already gone."""
        client = MagicMock()
        client.is_active.return_value = True
        client.open_channel.side_effect = [FakeChannel(), FakeChannel()]
        executor = RemoteExecutor(client)
        open_channel_calls_after_init = client.open_channel.call_count

        # Simulate a transient reset (channel nulled), then dispose via close().
        executor._channel = None
        executor.close()

        with pytest.raises(RemoteExecutorError, match="closed"):
            executor.call("ping")
        # No fresh channel was opened -- the closed executor did not self-heal.
        assert client.open_channel.call_count == open_channel_calls_after_init

    def test_start_nulls_channel_when_ping_fails(self):
        """If the freshly opened channel's server answers ping with a non-\"ok\"
        payload, _start() must null the channel before raising so the NEXT call can
        re-heal cleanly rather than reusing a half-started (live socket, bad server)
        channel."""

        class _BadPingChannel(FakeChannel):
            def _handle(self, req):
                if req["method"] == "ping":
                    return {"id": req["id"], "result": "not-ok"}
                return super()._handle(req)

        client = MagicMock()
        client.is_active.return_value = True
        client.open_channel.return_value = _BadPingChannel()
        with pytest.raises(RemoteExecutorError, match="ping failed"):
            RemoteExecutor(client)
