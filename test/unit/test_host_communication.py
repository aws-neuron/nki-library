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
"""Unit tests for the transport-level P0 outage hardening.

Per-command timeouts + keepalive live inside the ``HostCommunication``
transports so a wedged host surfaces as a catchable ``TimeoutException`` that
feeds the existing host-rotation loop, instead of hanging the caller.
"""

from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest
from invoke.exceptions import CommandTimedOut

from test.utils.exceptions import TimeoutException
from test.utils.host_communication import (
    SSH_INFERENCE_TIMEOUT_SECONDS,
    CommandResult,
    HostCommunication,
    LocalCommunication,
    ParamikoCommunication,
    SshCommunication,
)


def _make_paramiko_comm(**kwargs) -> tuple[ParamikoCommunication, MagicMock]:
    """A ParamikoCommunication with a pre-injected MagicMock connection."""
    comm = ParamikoCommunication("/tmp/fake_ssh_config", "fake-host", **kwargs)
    conn = MagicMock()
    conn.is_connected = True
    comm._connection = conn
    return comm, conn


class TestTimeoutBudgets:
    """Effective-timeout resolution: explicit > control default."""

    def test_timeout_constants(self) -> None:
        from test.utils.core_lock_client import DEFAULT_LOCK_TIMEOUT_SECONDS

        # Inference budget reuses the lock-lease constant; control is half of it.
        assert SSH_INFERENCE_TIMEOUT_SECONDS == DEFAULT_LOCK_TIMEOUT_SECONDS == 60
        assert ParamikoCommunication.SSH_KEEPALIVE_SECONDS == 15
        assert HostCommunication.CONTROL_COMMAND_TIMEOUT == SSH_INFERENCE_TIMEOUT_SECONDS / 2
        assert HostCommunication.WORKLOAD_COMMAND_TIMEOUT == SSH_INFERENCE_TIMEOUT_SECONDS

    def test_default_run_uses_control_budget(self) -> None:
        comm, conn = _make_paramiko_comm()
        comm.run("echo hi")
        assert conn.run.call_args.kwargs["timeout"] == HostCommunication.CONTROL_COMMAND_TIMEOUT

    def test_workload_timeout_uses_max_budget(self) -> None:
        comm, conn = _make_paramiko_comm()
        comm.run("inference_cmd", timeout=comm.WORKLOAD_COMMAND_TIMEOUT)
        assert conn.run.call_args.kwargs["timeout"] == SSH_INFERENCE_TIMEOUT_SECONDS

    def test_explicit_timeout_overrides_budgets(self) -> None:
        comm, conn = _make_paramiko_comm()
        comm.run("long_capture", timeout=900)
        assert conn.run.call_args.kwargs["timeout"] == 900

    def test_run_forwards_hide_and_warn_kwargs(self) -> None:
        """hide/warn reach the underlying connection alongside the resolved timeout."""
        comm, conn = _make_paramiko_comm()
        comm.run("echo hi", hide=True, warn=False, timeout=12)
        conn.run.assert_called_once_with("echo hi", hide=True, warn=False, timeout=12)

    def test_constructor_max_budget_is_honored(self) -> None:
        comm, conn = _make_paramiko_comm(max_timeout_seconds=1200)
        comm.run("cmd", timeout=comm.WORKLOAD_COMMAND_TIMEOUT)
        assert conn.run.call_args.kwargs["timeout"] == 1200

    def test_constructor_control_budget_is_honored(self) -> None:
        comm, conn = _make_paramiko_comm(max_timeout_seconds=1200, control_timeout_seconds=45)
        comm.run("cmd")
        assert conn.run.call_args.kwargs["timeout"] == 45


class TestTimeoutTranslation:
    """A command exceeding its budget becomes TimeoutException so HostManager's
    rotation loop marks the host failed and rotates instead of hanging."""

    def test_paramiko_maps_command_timeout_to_timeout_exception(self) -> None:
        comm, conn = _make_paramiko_comm()
        conn.run.side_effect = CommandTimedOut(result=MagicMock(), timeout=30)
        with pytest.raises(TimeoutException):
            comm.run("sleep 999")

    def test_local_maps_command_timeout_to_timeout_exception(self) -> None:
        comm = LocalCommunication()
        with pytest.raises(TimeoutException):
            comm.run("sleep 5", timeout=0.2)

    def test_local_run_within_budget_succeeds(self) -> None:
        result = LocalCommunication().run("echo hi")
        assert isinstance(result, CommandResult)
        assert result.ok
        assert result.stdout.strip() == "hi"


class TestKeepalive:
    """Keepalive self-contained in the transports: paramiko set_keepalive(15)."""

    def test_apply_keepalive_sets_interval_on_transport(self) -> None:
        comm, conn = _make_paramiko_comm()
        transport = MagicMock()
        conn.client.get_transport.return_value = transport
        comm._apply_keepalive()
        transport.set_keepalive.assert_called_once_with(ParamikoCommunication.SSH_KEEPALIVE_SECONDS)

    def test_run_applies_keepalive(self) -> None:
        comm, conn = _make_paramiko_comm()
        comm._apply_keepalive = MagicMock()
        comm.run("echo hi")
        comm._apply_keepalive.assert_called_once()

    def test_apply_keepalive_opens_closed_connection(self) -> None:
        comm, conn = _make_paramiko_comm()
        conn.is_connected = False
        comm._apply_keepalive()
        conn.open.assert_called_once()

    def test_apply_keepalive_tolerates_missing_transport(self) -> None:
        comm, conn = _make_paramiko_comm()
        conn.client.get_transport.return_value = None
        comm._apply_keepalive()  # must not raise


class TestSshCommunication:
    """Native-ssh transport: argv shape and command-timeout translation."""

    def test_ssh_argv_uses_config_and_per_process_control_path(self) -> None:
        comm = SshCommunication(ssh_config_path="/tmp/fake_ssh_config", alias="fake-host")
        argv = comm._ssh_argv("echo hi")
        assert argv[:3] == ["ssh", "-F", "/tmp/fake_ssh_config"]
        assert argv[-2:] == ["fake-host", "echo hi"]
        control_paths = [a for a in argv if a.startswith("ControlPath=")]
        assert control_paths == [f"ControlPath={comm._control_path}"]
        assert comm._control_path.endswith("-%C")

    def test_run_maps_command_timeout_to_timeout_exception(self) -> None:
        comm = SshCommunication(ssh_config_path="/tmp/fake_ssh_config", alias="fake-host")
        with patch(
            "test.utils.host_communication.subprocess.run",
            side_effect=TimeoutExpired(cmd="ssh", timeout=30),
        ):
            with pytest.raises(TimeoutException):
                comm.run("sleep 999", timeout=30)
