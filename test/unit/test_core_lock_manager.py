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
"""Unit tests for core_lock_manager module."""

import json
from test.utils.core_lock_client import (
    DEFAULT_LOCKING_PROTOCOL_VERSION,
)
from test.utils.core_lock_manager import (
    CoreLockManager,
    LockVersionError,
    check_lock_version,
)
from test.utils.metrics_collector import NoopMetricsCollector
from test.utils.scripts.remote_lock_scripts import LockResult, LockStatus
from unittest.mock import MagicMock

import pytest


class TestLockVersionError:
    """Tests for LockVersionError exception."""

    def test_error_message(self):
        """Test that error message contains version information."""
        error = LockVersionError(required_version=99, current_version=2)
        assert "99" in str(error)
        assert "2" in str(error)
        assert error.required_version == 99
        assert error.current_version == 2
        assert error.retryable is True


class TestCheckLockVersion:
    """Tests for check_lock_version function."""

    def _mock_conn_with_file(self, file_content: str) -> MagicMock:
        """Create mock connection where version file exists with given content."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # First call: test -f (file exists)
        test_result = MagicMock()
        test_result.ok = True

        # Second call: cat (returns content)
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = file_content

        mock_conn.run.side_effect = [test_result, cat_result]
        return mock_conn

    def _mock_conn_without_file(self) -> MagicMock:
        """Create mock connection where version file doesn't exist."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # First call: test -f (file doesn't exist)
        test_result = MagicMock()
        test_result.ok = False

        # Second call: mkdir && echo (create file)
        write_result = MagicMock()
        write_result.failed = False

        mock_conn.run.side_effect = [test_result, write_result]
        return mock_conn

    def test_version_check_passes_when_compatible(self):
        """Test that version check passes when client version is sufficient."""
        mock_conn = self._mock_conn_with_file(json.dumps({"minClientLockingVersion": 1}))
        # Should not raise
        check_lock_version(mock_conn)

    def test_version_check_fails_when_client_too_old(self):
        """Test that version check fails when client version is too old."""
        mock_conn = self._mock_conn_with_file(json.dumps({"minClientLockingVersion": 99}))

        with pytest.raises(LockVersionError) as exc_info:
            check_lock_version(mock_conn)

        assert exc_info.value.required_version == 99
        assert exc_info.value.current_version == DEFAULT_LOCKING_PROTOCOL_VERSION

    def test_version_check_handles_missing_file(self):
        """Test that version check creates file when missing."""
        mock_conn = self._mock_conn_without_file()
        # Should not raise - creates file with default version
        check_lock_version(mock_conn)

    def test_version_check_handles_missing_key(self):
        """Test that version check recreates file when key is missing."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # First call: test -f (file exists)
        test_result = MagicMock()
        test_result.ok = True

        # Second call: cat (returns content without key)
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = json.dumps({"someOtherKey": 123})

        # Third call: mkdir && echo (recreate file)
        write_result = MagicMock()
        write_result.failed = False

        mock_conn.run.side_effect = [test_result, cat_result, write_result]

        # Should not raise - recreates file
        check_lock_version(mock_conn)

    def test_version_check_handles_corrupted_json(self):
        """Test that version check recreates file when JSON is corrupted."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"

        # First call: test -f (file exists)
        test_result = MagicMock()
        test_result.ok = True

        # Second call: cat (returns corrupted content)
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = "not valid json {{{"

        # Third call: mkdir && echo (recreate file)
        write_result = MagicMock()
        write_result.failed = False

        mock_conn.run.side_effect = [test_result, cat_result, write_result]

        # Should not raise - recreates file
        check_lock_version(mock_conn)


class TestPhysicalToLogicalCores:
    """Tests for CoreLockManager._physical_to_logical_cores."""

    def test_lnc2_conversion(self):
        """Test LNC2: 2 physical cores = 1 logical core."""
        result = CoreLockManager._physical_to_logical_cores([0, 1, 2, 3], lnc_config=2)
        assert result == [0, 1]

    def test_lnc2_offset(self):
        """Test LNC2 with offset physical cores."""
        result = CoreLockManager._physical_to_logical_cores([4, 5, 6, 7], lnc_config=2)
        assert result == [2, 3]

    def test_lnc1_conversion(self):
        """Test LNC1: 1 physical core = 1 logical core."""
        result = CoreLockManager._physical_to_logical_cores([0, 1, 2], lnc_config=1)
        assert result == [0, 1, 2]

    def test_lnc1_offset(self):
        """Test LNC1 with offset physical cores."""
        result = CoreLockManager._physical_to_logical_cores([4, 5, 6, 7], lnc_config=1)
        assert result == [4, 5, 6, 7]

    def test_misaligned_raises(self):
        """Test that misaligned physical cores raise assertion."""
        with pytest.raises(AssertionError) as exc_info:
            CoreLockManager._physical_to_logical_cores([1, 2, 3, 4], lnc_config=2)
        assert "contiguous and aligned" in str(exc_info.value)


class TestDrain:
    """Tests for CoreLockManager.drain method."""

    def _make_manager(self, run_side_effects: list) -> CoreLockManager:
        """Create a CoreLockManager with mocked connection and noop collector."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_conn.run.side_effect = run_side_effects
        mgr = CoreLockManager(mock_conn, total_physical_cores=8, collector=NoopMetricsCollector())
        mgr._initialized = True  # Skip initialize
        mgr._host_locking_version = 2
        return mgr

    def test_drain_returns_max_lock_expiry(self):
        """drain() returns max_lock_expiry from result file."""
        result_json = LockResult(status=LockStatus.DRAINED, max_lock_expiry=1234567890).to_json()
        # flock_execute result (exit code 13 = DRAINED)
        flock_result = MagicMock()
        flock_result.return_code = LockStatus.DRAINED.exit_code
        # cat result file
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = result_json

        mgr = self._make_manager([flock_result, cat_result])
        max_expiry = mgr.drain(timeout_seconds=1800)
        assert max_expiry == 1234567890

    def test_drain_returns_zero_when_no_active_locks(self):
        """drain() returns 0 when no active locks."""
        result_json = LockResult(status=LockStatus.DRAINED, max_lock_expiry=0).to_json()
        flock_result = MagicMock()
        flock_result.return_code = LockStatus.DRAINED.exit_code
        cat_result = MagicMock()
        cat_result.failed = False
        cat_result.stdout = result_json

        mgr = self._make_manager([flock_result, cat_result])
        max_expiry = mgr.drain(timeout_seconds=1800)
        assert max_expiry == 0

    def test_drain_returns_zero_on_failure(self):
        """drain() returns 0 when helper reports error."""
        flock_result = MagicMock()
        flock_result.return_code = LockStatus.ERROR.exit_code
        # rm -f cleanup
        rm_result = MagicMock()
        rm_result.failed = False

        mgr = self._make_manager([flock_result, rm_result])
        max_expiry = mgr.drain(timeout_seconds=1800)
        assert max_expiry == 0


class TestDisableDrain:
    """Tests for CoreLockManager.disable_drain method."""

    def _make_manager(self, run_side_effects: list) -> CoreLockManager:
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_conn.run.side_effect = run_side_effects
        mgr = CoreLockManager(mock_conn, total_physical_cores=8, collector=NoopMetricsCollector())
        mgr._initialized = True
        mgr._host_locking_version = 2
        return mgr

    def test_disable_drain_succeeds(self):
        """disable_drain() calls undrain helper."""
        flock_result = MagicMock()
        flock_result.return_code = LockStatus.UNDRAINED.exit_code
        rm_result = MagicMock()
        rm_result.failed = False

        mgr = self._make_manager([flock_result, rm_result])
        mgr.disable_drain()  # Should not raise

    def test_disable_drain_warns_on_failure(self):
        """disable_drain() logs warning on failure."""
        flock_result = MagicMock()
        flock_result.return_code = LockStatus.ERROR.exit_code
        rm_result = MagicMock()
        rm_result.failed = False

        mgr = self._make_manager([flock_result, rm_result])
        mgr.disable_drain()  # Should not raise, just warn
