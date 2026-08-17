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
from unittest.mock import ANY, MagicMock, patch

import pytest
from botocore.credentials import EnvProvider
from botocore.exceptions import ClientError

from ..utils.s3_utils import (
    S3ArtifactUploadConfig,
    S3TransferDirection,
    _create_boto3_session_with_retry,
    _get_boto_client_cached,
    _get_boto_session_cached,
    _is_throttling_error,
    build_remote_s3_cli_command,
    generate_s3_key,
    get_boto_client,
    get_boto_session,
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


class _BotoCacheTestBase:
    """Shared setup/teardown: clear the per-process session/client caches and isolate
    AWS_* env vars so cache state never leaks across tests."""

    def setup_method(self):
        _get_boto_session_cached.cache_clear()
        _get_boto_client_cached.cache_clear()
        self._original_env = {}
        for var in [EnvProvider.ACCESS_KEY, EnvProvider.SECRET_KEY, EnvProvider.TOKENS[1]]:
            self._original_env[var] = os.environ.get(var)
            os.environ.pop(var, None)

    def teardown_method(self):
        _get_boto_session_cached.cache_clear()
        _get_boto_client_cached.cache_clear()
        for var, value in self._original_env.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)


class TestGetBotoSessionAndClient(_BotoCacheTestBase):
    """Test the shared session/client helpers used across services and regions."""

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_session_uses_retry_logic_and_caches_per_profile(self, mock_create_session):
        mock_create_session.side_effect = lambda profile=None: MagicMock()

        a1 = get_boto_session("p1")
        a2 = get_boto_session("p1")
        b = get_boto_session("p2")

        assert a1 is a2  # same profile -> one session (one credential fetch)
        assert a1 is not b  # distinct profile -> distinct session
        assert mock_create_session.call_count == 2
        mock_create_session.assert_any_call("p1")

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_client_is_built_with_region_and_retry_config(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        get_boto_client("ec2", region="eu-west-1", profile="p1")

        _, kwargs = mock_session.client.call_args
        assert mock_session.client.call_args.args[0] == "ec2"
        assert kwargs["region_name"] == "eu-west-1"
        assert kwargs["config"].retries["max_attempts"] == 5

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_client_caches_per_service_region_profile_sharing_one_session(self, mock_create_session):
        mock_session = MagicMock()
        mock_session.client.side_effect = lambda *a, **k: MagicMock()  # fresh client per underlying call
        mock_create_session.return_value = mock_session

        a1 = get_boto_client("ec2", region="us-west-2", profile="p1")
        a2 = get_boto_client("ec2", region="us-west-2", profile="p1")
        b = get_boto_client("ec2", region="eu-west-1", profile="p1")
        c = get_boto_client("autoscaling", region="us-west-2", profile="p1")

        assert a1 is a2  # same (service, region, profile) -> cached
        assert len({id(a1), id(b), id(c)}) == 3  # distinct keys -> distinct clients
        assert mock_session.client.call_count == 3  # a2 served from cache
        mock_create_session.assert_called_once()  # all clients share one session

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_session_call_forms_canonicalize_to_one_entry(self, mock_create_session):
        # The private-cached-core split closes the lru_cache literal-call-shape gap: omitting
        # profile, passing it positionally, and passing it by keyword are the same logical
        # session and must share ONE cache entry (one credential fetch), not three.
        mock_create_session.side_effect = lambda profile=None: MagicMock()

        omitted = get_boto_session()
        positional = get_boto_session(None)
        keyword = get_boto_session(profile=None)

        assert omitted is positional is keyword
        assert mock_create_session.call_count == 1

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_client_omitted_region_and_explicit_none_canonicalize(self, mock_create_session):
        # Same gap for the client: omitting region vs passing region=None are one logical
        # client -> one cache entry, not two.
        mock_session = MagicMock()
        mock_session.client.side_effect = lambda *a, **k: MagicMock()
        mock_create_session.return_value = mock_session

        omitted = get_boto_client("ec2", profile="p1")
        explicit_none = get_boto_client("ec2", region=None, profile="p1")

        assert omitted is explicit_none
        assert mock_session.client.call_count == 1


class TestGetS3ClientAndSession(_BotoCacheTestBase):
    """Test get_s3_client_and_session — now a thin wrapper over the shared helpers."""

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_uses_retry_logic_for_credentials(self, mock_create_session):
        mock_create_session.return_value = MagicMock()

        _, session = get_s3_client_and_session(profile="my-profile")

        mock_create_session.assert_called_once_with("my-profile")
        assert session is mock_create_session.return_value

    @patch("test.utils.s3_utils._create_boto3_session_with_retry")
    def test_returns_s3_client_and_session(self, mock_create_session):
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_create_session.return_value = mock_session

        client, session = get_s3_client_and_session()

        assert client is mock_client
        assert session is mock_session
        # The wrapper builds exactly one client, for the s3 service (region/config
        # wiring is get_boto_client's contract — covered in TestGetBotoSessionAndClient).
        mock_session.client.assert_called_once_with("s3", region_name=None, config=ANY)

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
