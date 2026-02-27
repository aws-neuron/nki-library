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
"""Unit tests for host_management retry logic and error reporting."""

import json
import os
import tempfile
from contextlib import closing
from test.utils.common_dataclasses import Platforms, TargetHost
from test.utils.exceptions import InferenceException
from test.utils.host_management import HostManager
from unittest.mock import MagicMock, patch

import pytest
from paramiko import SSHException


class TestHostManagerRetryErrorReporting:
    """Test that host errors are captured and reported when all hosts fail."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_host_manager(self, temp_dir):
        """Create a HostManager with mocked hosts for testing."""
        target_hosts = [
            TargetHost(ssh_host="host1", host_type=Platforms.TRN2),
            TargetHost(ssh_host="host2", host_type=Platforms.TRN2),
        ]

        with patch("test.utils.host_management.SshHost"):
            manager = HostManager(
                base_host_info_path=temp_dir,
                target_hosts=target_hosts,
                neuron_installation_path="/opt/aws/neuron/bin",
                ssh_config_path="~/.ssh/config",
                default_platform_target=Platforms.TRN2,
            )

        # Initialize host stats file
        host_stats_path = os.path.join(temp_dir, "host_stats.json")
        with open(host_stats_path, "w") as f:
            json.dump(
                [
                    {"host_alias": "host1", "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"},
                    {"host_alias": "host2", "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"},
                ],
                f,
            )

        # Create mock hosts that return their host_id
        mock_host1 = MagicMock()
        mock_host1.get_host_id.return_value = "host1"
        mock_host2 = MagicMock()
        mock_host2.get_host_id.return_value = "host2"

        manager.target_hosts = {"host1": mock_host1, "host2": mock_host2}
        manager.host_types = {"host1": Platforms.TRN2, "host2": Platforms.TRN2}

        return manager

    def test_error_details_included_when_all_retries_exhausted(self, mock_host_manager):
        """Test that error details from each host are included in final exception."""
        mock_collector = MagicMock()

        with pytest.raises(InferenceException) as exc_info:
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    max_retries=2,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        # Simulate different errors on each host
                        host_id = host.get_host_id()
                        if host_id == "host1":
                            raise TimeoutError("Connection timed out after 30 seconds")
                        else:
                            raise SSHException("Authentication failed")

        # Verify error message contains details from both hosts
        error_message = str(exc_info.value)
        print("\n=== Test 1: All Retries Exhausted ===")
        print(error_message)
        print("=" * 50)
        assert "Connection error after 2 attempts" in error_message
        assert "host1" in error_message or "host2" in error_message
        assert "Errors from each host:" in error_message

    def test_error_details_included_when_no_hosts_available(self, mock_host_manager):
        """Test that previous errors are included when __get_host_assignment__ raises."""
        mock_collector = MagicMock()

        # First, manually fail host1 with a specific error
        with pytest.raises(InferenceException) as exc_info:
            with closing(
                mock_host_manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    max_retries=3,
                )
            ) as host_generator:
                attempt = 0
                for host_attempt in host_generator:
                    with host_attempt as host:
                        attempt += 1
                        host_id = host.get_host_id()
                        # Fail both hosts - after 2 failures, no hosts will be available
                        raise OSError(f"Network unreachable on {host_id}")

        # Verify error message contains the enriched details
        error_message = str(exc_info.value)
        print("\n=== Test 2: No Hosts Available ===")
        print(error_message)
        print("=" * 50)
        assert "Errors from" in error_message
        assert "OSError" in error_message or "Network unreachable" in error_message

    def test_single_host_failure_shows_specific_error(self, temp_dir):
        """single host fails, error details should be shown."""
        target_hosts = [
            TargetHost(ssh_host="single-host", host_type=Platforms.TRN2),
        ]

        with patch("test.utils.host_management.SshHost"):
            manager = HostManager(
                base_host_info_path=temp_dir,
                target_hosts=target_hosts,
                neuron_installation_path="/opt/aws/neuron/bin",
                ssh_config_path="~/.ssh/config",
                default_platform_target=Platforms.TRN2,
            )

        # Initialize host stats file with single host
        host_stats_path = os.path.join(temp_dir, "host_stats.json")
        with open(host_stats_path, "w") as f:
            json.dump(
                [{"host_alias": "single-host", "work_queue_depth": 0, "run_id": manager.run_id, "host_type": "trn2"}],
                f,
            )

        mock_host = MagicMock()
        mock_host.get_host_id.return_value = "single-host"
        manager.target_hosts = {"single-host": mock_host}
        manager.host_types = {"single-host": Platforms.TRN2}

        mock_collector = MagicMock()

        with pytest.raises(InferenceException) as exc_info:
            with closing(
                manager.get_host_assignment_with_retry(
                    platform_target=Platforms.TRN2,
                    collector=mock_collector,
                    max_retries=3,
                )
            ) as host_generator:
                for host_attempt in host_generator:
                    with host_attempt as host:
                        raise TimeoutError("SSH connection timed out during inference")

        # Verify the specific error is included
        error_message = str(exc_info.value)
        print("\n=== Test 3: Single Host Failure ===")
        print(error_message)
        print("=" * 50)
        assert "single-host" in error_message
        assert "TimeoutError" in error_message
        assert "SSH connection timed out during inference" in error_message
        assert "Errors from previous host attempts:" in error_message
