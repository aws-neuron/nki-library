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
from test.utils.core_lock_manager import (
    LOCKING_PROTOCOL_VERSION,
    InfraVersionError,
    check_infra_locking_version,
)
from unittest.mock import MagicMock

import pytest


class TestInfraVersionError:
    """Tests for InfraVersionError."""

    def test_error_message(self):
        """Test error message format."""
        error = InfraVersionError(required_version=3, current_version=1)
        message = str(error)
        assert "v3" in message
        assert "v1" in message
        assert "brazil ws use" in message
        assert "SHARED FLEET VERSION MISMATCH" in message
        assert "update your nkilib" in message.lower()

    def test_error_attributes(self):
        """Test error stores attributes correctly."""
        error = InfraVersionError(required_version=5, current_version=1)
        assert error.required_version == 5
        assert error.current_version == 1


class TestCheckInfraLockingVersion:
    """Tests for check_infra_locking_version function."""

    def test_version_check_passes_when_client_meets_requirement(self):
        """Test that version check passes when client version >= required."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.stdout = json.dumps({"minClientLockingVersion": 1})
        mock_conn.run.return_value = mock_result

        # Should not raise
        check_infra_locking_version(mock_conn)

    def test_version_check_fails_when_client_too_old(self):
        """Test that version check fails when infra requires newer version."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.stdout = json.dumps({"minClientLockingVersion": 99})
        mock_conn.run.return_value = mock_result

        with pytest.raises(InfraVersionError) as exc_info:
            check_infra_locking_version(mock_conn)

        assert exc_info.value.required_version == 99
        assert exc_info.value.current_version == LOCKING_PROTOCOL_VERSION

    def test_version_check_handles_missing_file(self):
        """Test that version check handles missing infra_version.json."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.stdout = "{}"  # Empty JSON when file doesn't exist
        mock_conn.run.return_value = mock_result

        # Should not raise - logs info and returns
        check_infra_locking_version(mock_conn)

    def test_version_check_handles_corrupted_json(self):
        """Test that version check handles corrupted infra_version.json."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.stdout = "not valid json"
        mock_conn.run.return_value = mock_result

        # Should not raise - logs warning and continues
        check_infra_locking_version(mock_conn)

    def test_version_check_handles_connection_error(self):
        """Test that version check handles connection errors gracefully."""
        mock_conn = MagicMock()
        mock_conn.host = "test-host"
        mock_conn.run.side_effect = Exception("Connection failed")

        # Should not raise - logs warning and continues
        check_infra_locking_version(mock_conn)
