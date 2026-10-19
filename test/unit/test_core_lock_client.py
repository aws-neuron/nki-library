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
"""Unit + no-hardware integration tests for the poll/probe/dequeue client wrappers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from unittest.mock import MagicMock

from test.utils import core_lock_client
from test.utils.core_lock_client import dequeue, poll, probe
from test.utils.scripts import remote_lock_scripts
from test.utils.scripts.remote_lock_scripts import LockStatus

# Common request parameters used across the wrapper tests.
TOTAL_CORES = 16
NUM_PHYSICAL = 4
TIMEOUT = 60
VERSION = 3
LNC_CONFIG = 2


# =============================================================================
# Unit tests: MagicMock executor returning canned response dicts. Verify the
# client maps response -> LockResult (including position/worst_case_eta and the
# IN_QUEUE/DRAINING/ALLOCATED statuses).
# =============================================================================


class TestPollWrapperUnit:
    """Unit tests for poll() response -> LockResult mapping."""

    def test_poll_in_queue_carries_position_and_eta(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "IN_QUEUE",
            "position": 3,
            "worst_case_eta": 1234567890,
        }
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", False, LNC_CONFIG)
        assert result.status == LockStatus.IN_QUEUE
        assert result.position == 3
        assert result.worst_case_eta == 1234567890
        assert result.cores is None

    def test_poll_allocated_carries_cores(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "ALLOCATED",
            "cores": [0, 1, 2, 3],
            "expiry": 999,
        }
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", True, LNC_CONFIG)
        assert result.status == LockStatus.ALLOCATED
        assert result.cores == [0, 1, 2, 3]
        assert result.expiry == 999
        # Back-compatible: absent position/eta parse as None.
        assert result.position is None
        assert result.worst_case_eta is None

    def test_poll_draining_status(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "DRAINING",
            "position": 0,
            "worst_case_eta": 42,
        }
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", True, LNC_CONFIG)
        assert result.status == LockStatus.DRAINING
        assert result.position == 0
        assert result.worst_case_eta == 42

    def test_poll_maps_bumped_flag(self) -> None:
        """The additive bumped flag round-trips through the response -> LockResult mapping."""
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "IN_QUEUE",
            "position": 0,
            "worst_case_eta": 1,
            "bumped": True,
        }
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", False, LNC_CONFIG)
        assert result.bumped is True

    def test_poll_bumped_defaults_false_when_absent(self) -> None:
        """A response without bumped (older helper) maps to bumped=False."""
        executor = MagicMock()
        executor.call_function.return_value = {"status": "IN_QUEUE", "position": 0, "worst_case_eta": 1}
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", False, LNC_CONFIG)
        assert result.bumped is False

    def test_poll_maps_re_enqueued_flag(self) -> None:
        """The additive re_enqueued flag round-trips through the response -> LockResult mapping."""
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "IN_QUEUE",
            "position": 3,
            "worst_case_eta": 1,
            "re_enqueued": True,
        }
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", False, LNC_CONFIG)
        assert result.re_enqueued is True

    def test_poll_re_enqueued_defaults_false_when_absent(self) -> None:
        """A response without re_enqueued (older helper) maps to re_enqueued=False."""
        executor = MagicMock()
        executor.call_function.return_value = {"status": "IN_QUEUE", "position": 0, "worst_case_eta": 1}
        result = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", False, LNC_CONFIG)
        assert result.re_enqueued is False

    def test_poll_forwards_expected_args(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {"status": "IN_QUEUE", "position": 0, "worst_case_eta": 1}
        poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-9", True, LNC_CONFIG, caller_id="gw1:test")
        _, kwargs = executor.call_function.call_args
        assert kwargs["command"] == "poll"
        assert kwargs["args"] == [
            core_lock_client.REMOTE_LOCKS_JSON,
            TOTAL_CORES,
            NUM_PHYSICAL,
            TIMEOUT,
            VERSION,
            "entry-9",
            True,
            LNC_CONFIG,
        ]
        assert kwargs["kwargs"] == {"caller_id": "gw1:test"}


class TestDequeueWrapperUnit:
    """Unit tests for dequeue() response -> LockResult mapping."""

    def test_dequeue_released(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {"status": "RELEASED"}
        result = dequeue(executor, TOTAL_CORES, VERSION, "entry-1")
        assert result.status == LockStatus.RELEASED

    def test_dequeue_forwards_expected_args(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {"status": "RELEASED"}
        dequeue(executor, TOTAL_CORES, VERSION, "entry-2", caller_id="gw2:test")
        _, kwargs = executor.call_function.call_args
        assert kwargs["command"] == "dequeue"
        assert kwargs["args"] == [core_lock_client.REMOTE_LOCKS_JSON, TOTAL_CORES, VERSION, "entry-2"]
        assert kwargs["kwargs"] == {"caller_id": "gw2:test"}


class TestProbeWrapperUnit:
    """Unit tests for probe() response -> LockResult mapping."""

    def test_probe_in_queue_carries_eta(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "IN_QUEUE",
            "worst_case_eta": 1234567890,
        }
        result = probe(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION)
        assert result.status == LockStatus.IN_QUEUE
        assert result.worst_case_eta == 1234567890

    def test_probe_draining_status(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {
            "status": "DRAINING",
            "worst_case_eta": 42,
        }
        result = probe(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION)
        assert result.status == LockStatus.DRAINING
        assert result.worst_case_eta == 42

    def test_probe_forwards_expected_args(self) -> None:
        executor = MagicMock()
        executor.call_function.return_value = {"status": "IN_QUEUE", "worst_case_eta": 1}
        probe(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION)
        _, kwargs = executor.call_function.call_args
        assert kwargs["command"] == "probe"
        assert kwargs["args"] == [
            core_lock_client.REMOTE_LOCKS_JSON,
            TOTAL_CORES,
            NUM_PHYSICAL,
            TIMEOUT,
            VERSION,
        ]
        assert not kwargs["kwargs"]


# =============================================================================
# Integration test (no hardware): a fake executor whose call_function actually
# invokes the real remote helper verb against a temp locks.json, exercising the
# client -> remote-helper boundary (_remote_lock_operation round-trip) without
# any SSH. Proves the wire serialization round-trips position/status correctly.
# =============================================================================


class _FakeExecutor:
    """Fake RemoteExecutor that runs the deployed helper locally.

    Mirrors how RemoteExecutor.call_function is invoked by the client: the
    client passes ``_remote_lock_operation`` plus its kwargs. We redirect the
    remote file paths (helpers_file / locks_file) to local temp paths and run
    the *real* ``_remote_lock_operation`` so the helper module is genuinely
    loaded and the result dict is built exactly as on a remote host.
    """

    def __init__(self, helpers_file: str, lock_file: str, locks_json: str) -> None:
        self._helpers_file = helpers_file
        self._lock_file = lock_file
        self._locks_json = locks_json

    def call_function(self, func, **kwargs):
        kwargs = dict(kwargs)
        kwargs["helpers_file"] = self._helpers_file
        kwargs["lock_file"] = self._lock_file
        # args[0] is the remote locks.json path; redirect to the temp file.
        args = list(kwargs["args"])
        args[0] = self._locks_json
        kwargs["args"] = args
        return func(**kwargs)


def test_poll_round_trip_enqueue_then_allocate(tmp_path) -> None:
    """Client poll(ready=False) enqueues; a later poll(ready=True) -> ALLOCATED.

    Uses the real helper file on disk loaded by _remote_lock_operation, proving
    the client<->remote-helper wire round-trip works without SSH.
    """
    helpers_file = str(remote_lock_scripts.__file__)
    lock_file = str(tmp_path / "atomic_lock")
    locks_json = str(tmp_path / "locks.json")
    executor = _FakeExecutor(helpers_file, lock_file, locks_json)

    # First poll: not ready -> should enqueue and report IN_QUEUE at head.
    r1 = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", False, LNC_CONFIG)
    assert r1.status == LockStatus.IN_QUEUE
    assert r1.position == 0
    assert r1.worst_case_eta is not None

    # Second poll: ready -> head with an open window commits and gets cores.
    r2 = poll(executor, TOTAL_CORES, NUM_PHYSICAL, TIMEOUT, VERSION, "entry-1", True, LNC_CONFIG)
    assert r2.status == LockStatus.ALLOCATED
    assert r2.cores is not None
    assert len(r2.cores) == NUM_PHYSICAL


# =============================================================================
# Protocol v3 cutover: drift guard + recreate-warning tests.
# The two version sites (DEFAULT_LOCKING_PROTOCOL_VERSION and the default the
# remote initializer would write) must move together and cannot diverge.
# =============================================================================


class TestProtocolVersionDriftGuard:
    """Build-time guard: both version sites are coupled."""

    def test_remote_initialize_default_derives_from_constant(self) -> None:
        """The version initialize_and_deploy would write must equal the constant.

        Capture the default_version_json passed into _remote_initialize and assert
        it decodes to DEFAULT_LOCKING_PROTOCOL_VERSION, so the two sites cannot drift.
        """
        captured: dict = {}

        def fake_call_function(fn, **kwargs):
            captured.update(kwargs)
            # Mimic _remote_initialize returning the parsed default when no file exists.
            return json.loads(kwargs["default_version_json"])

        executor = MagicMock()
        executor.call_function.side_effect = fake_call_function

        returned = core_lock_client.initialize_and_deploy(executor)

        written = json.loads(captured["default_version_json"])
        assert written[core_lock_client.MIN_CLIENT_VERSION_KEY] == (core_lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION)
        assert returned == core_lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION


class TestGetHostLockingVersionRecreateWarns:
    """The silent-regression recreate path must be observable (logging.warning)."""

    def test_recreate_when_file_missing_warns_and_returns_constant(self, caplog) -> None:
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # First call: test -f (file missing). Second: mkdir && echo (write).
        test_result = MagicMock()
        test_result.ok = False
        write_result = MagicMock()
        write_result.failed = False
        mock_conn.run.side_effect = [test_result, write_result]

        with caplog.at_level(logging.WARNING, logger=core_lock_client.logger.name):
            version = core_lock_client.get_host_locking_version(mock_conn)

        assert version == core_lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION
        assert any(
            "Recreating infra_version.json" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records
        )

    def test_recreate_when_key_missing_warns_and_returns_constant(self, caplog) -> None:
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # File exists but lacks the version key -> recreate with the default.
        test_result = MagicMock()
        test_result.ok = True
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = json.dumps({"someOtherKey": 123})
        write_result = MagicMock()
        write_result.failed = False
        mock_conn.run.side_effect = [test_result, cat_result, write_result]

        with caplog.at_level(logging.WARNING, logger=core_lock_client.logger.name):
            version = core_lock_client.get_host_locking_version(mock_conn)

        assert version == core_lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION
        assert any(
            "Recreating infra_version.json" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records
        )

    def test_recreate_when_json_corrupted_warns_and_returns_constant(self, caplog) -> None:
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # File exists but contains corrupted JSON -> recreate with the default.
        test_result = MagicMock()
        test_result.ok = True
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = "not valid json {{{"
        write_result = MagicMock()
        write_result.failed = False
        mock_conn.run.side_effect = [test_result, cat_result, write_result]

        with caplog.at_level(logging.WARNING, logger=core_lock_client.logger.name):
            version = core_lock_client.get_host_locking_version(mock_conn)

        assert version == core_lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION
        assert any(
            "Recreating infra_version.json" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records
        )


class TestRemoteInitializeDeployVersionGate:
    """Version-gated helper deploy: a stale client must not downgrade a newer
    deployed lock_helpers.py (finding F7)."""

    @staticmethod
    def _call(tmpdir, helpers_content, existing_helper=None):
        lock_dir = os.path.join(tmpdir, "lockdir")
        lock_file = os.path.join(lock_dir, "lock")
        version_file = os.path.join(lock_dir, "infra_version.json")
        helpers_file = os.path.join(lock_dir, "lock_helpers.py")
        os.makedirs(lock_dir, exist_ok=True)
        if existing_helper is not None:
            with open(helpers_file, "w") as f:
                f.write(existing_helper)
        default_version_json = json.dumps({core_lock_client.MIN_CLIENT_VERSION_KEY: 3})
        core_lock_client._remote_initialize(
            lock_dir,
            lock_file,
            version_file,
            helpers_file,
            default_version_json,
            helpers_content,
        )
        with open(helpers_file) as f:
            return f.read()

    def test_newer_deployed_helper_not_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deployed = "SCRIPT_VERSION = 99\n# deployed newer\n"
            client = "SCRIPT_VERSION = 2\n# stale client\n"
            result = self._call(tmp, client, existing_helper=deployed)
            assert result == deployed  # preserved, not downgraded

    def test_equal_deployed_helper_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deployed = "SCRIPT_VERSION = 5\n# deployed\n"
            client = "SCRIPT_VERSION = 5\n# client different body\n"
            result = self._call(tmp, client, existing_helper=deployed)
            assert result == deployed  # equal => preserved

    def test_lower_deployed_helper_upgraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deployed = "SCRIPT_VERSION = 1\n# old\n"
            client = "SCRIPT_VERSION = 7\n# new client\n"
            result = self._call(tmp, client, existing_helper=deployed)
            assert result == client  # strictly-newer client upgrades

    def test_absent_deployed_helper_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = "SCRIPT_VERSION = 2\n# first deploy\n"
            result = self._call(tmp, client, existing_helper=None)
            assert result == client  # first deploy

    def test_malformed_deployed_helper_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deployed = "this file has no version marker at all\n"
            client = "SCRIPT_VERSION = 2\n# client\n"
            result = self._call(tmp, client, existing_helper=deployed)
            assert result == client  # unparseable deployed => write


class TestScriptVersionBumpGuard:
    """Guard so future verb additions don't forget to bump SCRIPT_VERSION."""

    def test_script_version_was_bumped_past_one(self) -> None:
        assert remote_lock_scripts.SCRIPT_VERSION > 1

    def test_script_version_bumped_for_lnc_reset_tracking(self) -> None:
        # The LNC reset-tracking change (poll gains lnc_config, LockResult gains
        # should_reset_cores) redeploys the helper; the bump is the deploy gate.
        assert remote_lock_scripts.SCRIPT_VERSION == 5
        assert core_lock_client.DEFAULT_LOCKING_PROTOCOL_VERSION == 5


class TestRemoteInitializeStaleRedeployIntegration:
    """No-hardware integration: a stale client redeploy must preserve the real
    newer deployed helper so its queue verbs (poll) remain callable (F7)."""

    def test_stale_redeploy_keeps_poll_verb_present(self) -> None:
        import importlib.util
        import sys

        real_helper = core_lock_client._LOCAL_LOCK_HELPERS_PATH
        real_source = real_helper.read_text()
        # Simulate a deployed helper that is NEWER than this stale client.
        newer_deployed = real_source.replace(
            f"SCRIPT_VERSION = {remote_lock_scripts.SCRIPT_VERSION}", "SCRIPT_VERSION = 999", 1
        )
        stale_client = "SCRIPT_VERSION = 1\n# stale client, lacks queue verbs\n"

        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = os.path.join(tmp, "lockdir")
            lock_file = os.path.join(lock_dir, "lock")
            version_file = os.path.join(lock_dir, "infra_version.json")
            helpers_file = os.path.join(lock_dir, "lock_helpers.py")
            os.makedirs(lock_dir, exist_ok=True)
            with open(helpers_file, "w") as f:
                f.write(newer_deployed)

            core_lock_client._remote_initialize(
                lock_dir,
                lock_file,
                version_file,
                helpers_file,
                json.dumps({core_lock_client.MIN_CLIENT_VERSION_KEY: 3}),
                stale_client,
            )

            # The preserved deployed helper must still expose the poll verb.
            spec = importlib.util.spec_from_file_location("deployed_helper", helpers_file)
            assert spec is not None, "the preserved helper file must be loadable as a module"
            loader = spec.loader
            assert loader is not None, "a file-based module spec has a loader"
            module_name = spec.name
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                loader.exec_module(module)
                assert hasattr(module, "poll") and callable(module.poll)
                assert module.SCRIPT_VERSION == 999
            finally:
                sys.modules.pop(module_name, None)
