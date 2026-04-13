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
Unit tests for s3_utils module.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.credentials import EnvProvider
from botocore.exceptions import ClientError

from ..utils.s3_utils import (
    S3ArtifactUploadConfig,
    S3TransferDirection,
    _create_boto3_session_with_retry,
    _is_throttling_error,
    build_remote_s3_cli_command,
    generate_s3_key,
    get_s3_client_and_session,
    prefetch_and_cache_credentials,
)


class TestGenerateS3Key:
    """Test generate_s3_key function."""

    def test_generates_inputs_key(self):
        config = S3ArtifactUploadConfig(prefix="test_prefix")
        key = generate_s3_key(config, S3TransferDirection.INPUTS)
        assert "/inputs/" in key
        assert key.endswith(".tar")

    def test_generates_outputs_key(self):
        config = S3ArtifactUploadConfig(prefix="test_prefix")
        key = generate_s3_key(config, S3TransferDirection.OUTPUTS)
        assert "/outputs/" in key
        assert key.endswith(".tar")

    def test_keys_are_unique(self):
        config = S3ArtifactUploadConfig(prefix="test_prefix")
        key1 = generate_s3_key(config, S3TransferDirection.INPUTS)
        key2 = generate_s3_key(config, S3TransferDirection.INPUTS)
        assert key1 != key2


class TestBuildRemoteS3CliCommand:
    """Test build_remote_s3_cli_command function."""

    def test_inputs_direction_downloads_from_s3(self):
        creds = MagicMock()
        creds.access_key = "AKIATEST"
        creds.secret_key = "secret123"
        creds.token = "token456"

        cmd = build_remote_s3_cli_command(
            s3_bucket="my-bucket",
            s3_key="path/to/file.tar",
            remote_path="/tmp/file.tar",
            creds=creds,
            direction=S3TransferDirection.INPUTS,
        )

        # remote_path is quoted to handle special characters like < > ' in test names
        assert "aws s3 cp s3://my-bucket/path/to/file.tar '/tmp/file.tar'" in cmd
        assert "AWS_ACCESS_KEY_ID=AKIATEST" in cmd
        assert "AWS_SECRET_ACCESS_KEY=secret123" in cmd
        assert "AWS_SESSION_TOKEN=token456" in cmd

    def test_outputs_direction_uploads_to_s3(self):
        creds = MagicMock()
        creds.access_key = "AKIATEST"
        creds.secret_key = "secret123"
        creds.token = None

        cmd = build_remote_s3_cli_command(
            s3_bucket="my-bucket",
            s3_key="path/to/file.tar",
            remote_path="/tmp/file.tar",
            creds=creds,
            direction=S3TransferDirection.OUTPUTS,
        )

        # remote_path is quoted to handle special characters like < > ' in test names
        assert "aws s3 cp '/tmp/file.tar' s3://my-bucket/path/to/file.tar" in cmd
        assert "AWS_SESSION_TOKEN" not in cmd  # No token when None

    def test_handles_special_characters_in_path(self):
        """Test that paths with special characters like < > are properly quoted."""
        creds = MagicMock()
        creds.access_key = "AKIATEST"
        creds.secret_key = "secret123"
        creds.token = None

        # Path with < character that would break bash without quoting
        cmd = build_remote_s3_cli_command(
            s3_bucket="my-bucket",
            s3_key="path/to/file.tar",
            remote_path="/tmp/out-test_sweep_64-4096-<class 'numpy.float32'>/file.tar",
            creds=creds,
            direction=S3TransferDirection.INPUTS,
        )

        assert "'/tmp/out-test_sweep_64-4096-<class 'numpy.float32'>/file.tar'" in cmd


class TestIsThrottlingError:
    """Test _is_throttling_error function."""

    def test_detects_client_error_throttling(self):
        """ClientError with throttling error code should be detected."""
        error = ClientError(
            {"Error": {"Code": "Throttling", "Message": "Rate exceeded"}},
            "AssumeRole",
        )
        assert _is_throttling_error(error) is True

    def test_detects_client_error_too_many_requests(self):
        """ClientError with TooManyRequestsException should be detected."""
        error = ClientError(
            {"Error": {"Code": "TooManyRequestsException", "Message": "Too many requests"}},
            "GetCredentials",
        )
        assert _is_throttling_error(error) is True

    def test_client_error_non_throttling(self):
        """ClientError with non-throttling code should not be detected as throttling."""
        error = ClientError({"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "AssumeRole")
        assert _is_throttling_error(error) is False

    def test_detects_throttling_exception(self):
        error = Exception("ThrottlingException: Rate exceeded")
        assert _is_throttling_error(error) is True

    def test_detects_rate_exceeded(self):
        error = Exception("Rate exceeded for account")
        assert _is_throttling_error(error) is True

    def test_detects_rate_limit(self):
        error = Exception("API rate limit hit")
        assert _is_throttling_error(error) is True

    def test_detects_too_many_requests(self):
        error = Exception("Too many requests")
        assert _is_throttling_error(error) is True

    def test_case_insensitive(self):
        error = Exception("THROTTLING ERROR")
        assert _is_throttling_error(error) is True

    def test_non_throttling_error(self):
        error = Exception("Access denied")
        assert _is_throttling_error(error) is False

    def test_connection_error(self):
        error = Exception("Connection refused")
        assert _is_throttling_error(error) is False


class TestCreateBoto3SessionWithRetry:
    """Test _create_boto3_session_with_retry function."""

    @patch("test.utils.s3_utils.boto3.Session")
    def test_success_on_first_attempt(self, mock_session_class):
        mock_session = MagicMock()
        mock_creds = MagicMock()
        mock_session.get_credentials.return_value = mock_creds
        mock_session_class.return_value = mock_session

        result = _create_boto3_session_with_retry(profile=None)

        assert result == mock_session
        mock_session_class.assert_called_once()

    @patch("test.utils.s3_utils.time.sleep")
    @patch("test.utils.s3_utils.boto3.Session")
    def test_retries_on_throttling_error(self, mock_session_class, mock_sleep):
        mock_session = MagicMock()
        mock_creds = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        # Fail twice with throttling, then succeed
        mock_session_class.side_effect = [
            Exception("ThrottlingException: Rate exceeded"),
            Exception("Rate exceeded"),
            mock_session,
        ]

        result = _create_boto3_session_with_retry(profile=None, max_retries=3)

        assert result == mock_session
        assert mock_session_class.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("test.utils.s3_utils.time.sleep")
    @patch("test.utils.s3_utils.boto3.Session")
    def test_raises_after_max_retries(self, mock_session_class, mock_sleep):
        mock_session_class.side_effect = Exception("ThrottlingException: Rate exceeded")

        with pytest.raises(Exception, match="Rate exceeded"):
            _create_boto3_session_with_retry(profile=None, max_retries=2)

        assert mock_session_class.call_count == 3  # Initial + 2 retries

    @patch("test.utils.s3_utils.boto3.Session")
    def test_non_throttling_error_fails_immediately(self, mock_session_class):
        """Non-throttling errors should not be retried."""
        mock_session_class.side_effect = Exception("Access denied")

        with pytest.raises(Exception, match="Access denied"):
            _create_boto3_session_with_retry(profile=None, max_retries=5)

        # Should fail on first attempt, no retries
        assert mock_session_class.call_count == 1

    @patch("test.utils.s3_utils.boto3.Session")
    def test_uses_profile_when_provided(self, mock_session_class):
        mock_session = MagicMock()
        mock_creds = MagicMock()
        mock_session.get_credentials.return_value = mock_creds
        mock_session_class.return_value = mock_session

        _create_boto3_session_with_retry(profile="my-profile")

        mock_session_class.assert_called_once_with(profile_name="my-profile")


class TestPrefetchAndCacheCredentials:
    """Test prefetch_and_cache_credentials function."""

    def setup_method(self):
        """Save original env vars before each test."""
        self._original_env = {}
        for var in [EnvProvider.ACCESS_KEY, EnvProvider.SECRET_KEY, EnvProvider.TOKENS[1]]:
            self._original_env[var] = os.environ.get(var)
            os.environ.pop(var, None)

    def teardown_method(self):
        """Restore original env vars after each test."""
        for var, value in self._original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_sets_environment_variables(self, mock_create_session):
        mock_session = MagicMock()
        mock_creds = MagicMock()
        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIATEST"
        mock_frozen_creds.secret_key = "secret123"
        mock_frozen_creds.token = "token456"

        mock_creds.get_frozen_credentials.return_value = mock_frozen_creds
        mock_session.get_credentials.return_value = mock_creds
        mock_create_session.return_value = mock_session

        prefetch_and_cache_credentials(profile="test-profile")

        assert os.environ[EnvProvider.ACCESS_KEY] == "AKIATEST"
        assert os.environ[EnvProvider.SECRET_KEY] == "secret123"
        assert os.environ[EnvProvider.TOKENS[1]] == "token456"

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_handles_no_credentials(self, mock_create_session):
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        mock_create_session.return_value = mock_session

        # Should not raise
        prefetch_and_cache_credentials()

        # Should not set env vars
        assert EnvProvider.ACCESS_KEY not in os.environ

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_handles_exception_gracefully(self, mock_create_session):
        mock_create_session.side_effect = Exception("Connection failed")

        # Should not raise, just log warning
        prefetch_and_cache_credentials()

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_handles_no_session_token(self, mock_create_session):
        """Test that missing session token (for non-STS credentials) is handled."""
        mock_session = MagicMock()
        mock_creds = MagicMock()
        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIATEST"
        mock_frozen_creds.secret_key = "secret123"
        mock_frozen_creds.token = None  # No session token

        mock_creds.get_frozen_credentials.return_value = mock_frozen_creds
        mock_session.get_credentials.return_value = mock_creds
        mock_create_session.return_value = mock_session

        prefetch_and_cache_credentials()

        assert os.environ[EnvProvider.ACCESS_KEY] == "AKIATEST"
        assert os.environ[EnvProvider.SECRET_KEY] == "secret123"
        assert EnvProvider.TOKENS[1] not in os.environ


class TestGetS3ClientAndSession:
    """Test get_s3_client_and_session function."""

    def setup_method(self):
        """Clear LRU cache and save original env vars before each test."""
        get_s3_client_and_session.cache_clear()
        self._original_env = {}
        for var in [EnvProvider.ACCESS_KEY, EnvProvider.SECRET_KEY, EnvProvider.TOKENS[1]]:
            self._original_env[var] = os.environ.get(var)
            os.environ.pop(var, None)

    def teardown_method(self):
        """Cleanup after each test."""
        get_s3_client_and_session.cache_clear()
        for var, value in self._original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_uses_retry_logic_for_credentials(self, mock_create_session):
        """Test that get_s3_client_and_session uses retry logic for credential fetching."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_create_session.return_value = mock_session

        client, session = get_s3_client_and_session(profile="my-profile")

        mock_create_session.assert_called_once_with("my-profile")
        assert session == mock_session

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_returns_s3_client_and_session(self, mock_create_session):
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_create_session.return_value = mock_session

        client, session = get_s3_client_and_session()

        assert client == mock_client
        assert session == mock_session
        # Verify S3 client is created
        mock_session.client.assert_called_once_with("s3")

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_caches_result(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        # Call twice
        result1 = get_s3_client_and_session()
        result2 = get_s3_client_and_session()

        # Should only create session once
        mock_create_session.assert_called_once()
        assert result1 == result2
