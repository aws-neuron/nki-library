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
"""Unit tests for artifact_manager module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from ..utils.artifact_manager import (
    S3ArtifactUploadConfig,
    S3UploadResult,
    UploadOutcome,
    should_include_file_in_upload,
    should_upload_artifacts,
    upload_test_artifacts_to_s3,
    validate_s3_credentials,
)


def test_should_upload_artifacts():
    """Test should_upload_artifacts() decision logic with both enum and string values."""
    # Empty config should never upload
    assert should_upload_artifacts("passed", "") == False
    assert should_upload_artifacts("failed", "") == False
    assert should_upload_artifacts("skipped", "") == False

    # "all" should always upload
    assert should_upload_artifacts("passed", "all") == True
    assert should_upload_artifacts("failed", "all") == True
    assert should_upload_artifacts("skipped", "all") == True
    assert should_upload_artifacts("passed", UploadOutcome.ALL) == True
    assert should_upload_artifacts("failed", UploadOutcome.ALL) == True
    assert should_upload_artifacts("skipped", UploadOutcome.ALL) == True

    # "success" should only upload on passed
    assert should_upload_artifacts("passed", "success") == True
    assert should_upload_artifacts("failed", "success") == False
    assert should_upload_artifacts("skipped", "success") == False
    assert should_upload_artifacts("passed", UploadOutcome.SUCCESS) == True

    # "fail" should only upload on failed
    assert should_upload_artifacts("passed", "fail") == False
    assert should_upload_artifacts("failed", "fail") == True
    assert should_upload_artifacts("skipped", "fail") == False
    assert should_upload_artifacts("failed", UploadOutcome.FAIL) == True


def test_should_include_file_in_upload():
    """Test should_include_file_in_upload() file filtering logic."""
    # Files that should be included
    assert should_include_file_in_upload("test.neff") == True
    assert should_include_file_in_upload("test.log") == True
    assert should_include_file_in_upload("test.txt") == True
    assert should_include_file_in_upload("test.csv") == True
    assert should_include_file_in_upload("/path/to/test.neff") == True
    assert should_include_file_in_upload("/path/to/test.log") == True
    assert should_include_file_in_upload("/path/to/test.txt") == True
    assert should_include_file_in_upload("/path/to/test.csv") == True

    # Case insensitive
    assert should_include_file_in_upload("TEST.NEFF") == True
    assert should_include_file_in_upload("TEST.LOG") == True
    assert should_include_file_in_upload("TEST.TXT") == True
    assert should_include_file_in_upload("TEST.CSV") == True

    # Files that should be excluded
    assert should_include_file_in_upload("test.ntff") == False
    assert should_include_file_in_upload("metrics.json") == False
    assert should_include_file_in_upload("ntff.json") == False
    assert should_include_file_in_upload("k_out") == False
    assert should_include_file_in_upload("v_out") == False
    assert should_include_file_in_upload("test_out") == False
    assert should_include_file_in_upload("test.py") == False
    assert should_include_file_in_upload("test.c") == False
    assert should_include_file_in_upload("test") == False  # No extension
    assert should_include_file_in_upload("/path/to/test.ntff") == False


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_upload_nonexistent_directory(mock_get_s3_client_and_session, caplog):
    """Test handling of nonexistent directory."""
    mock_get_s3_client_and_session.return_value = (MagicMock(), MagicMock())
    s3_config = S3ArtifactUploadConfig(bucket="test-bucket", prefix="prefix")
    result = upload_test_artifacts_to_s3(
        test_dir="/nonexistent/path",
        s3_config=s3_config,
        test_name="test_name",
    )

    assert result == S3UploadResult(uploaded_count=0, skipped_count=0, error_count=0, s3_url=None)
    # Should log warning but not crash
    assert "Test directory does not exist" in caplog.text
    # Should not attempt any S3 operations
    mock_get_s3_client_and_session.assert_not_called()


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_upload_successful(mock_get_s3_client_and_session, caplog):
    """Test successful upload of files with prefix."""
    mock_s3_client = MagicMock()
    mock_session = MagicMock()
    mock_get_s3_client_and_session.return_value = (mock_s3_client, mock_session)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create included files (.neff, .csv, .log, .txt)
        for filename in ["test.neff", "test.log", "output.txt", "data.TXT", "metrics.csv"]:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write("test content")

        # Create excluded files
        for filename in ["test.ntff", "metrics.json", "k_out"]:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write("test content")

        s3_config = S3ArtifactUploadConfig(bucket="test-bucket", prefix="prefix")
        # Upload artifacts
        result = upload_test_artifacts_to_s3(
            test_dir=tmpdir,
            s3_config=s3_config,
            test_name="test_name",
        )

    # Verify only included files were uploaded (5 included, 3 excluded)
    # included: .neff, .csv, .log, .txt files
    # excluded: .ntff, .json, k_out
    assert mock_s3_client.upload_file.call_count == 5

    # Verify return value
    assert result.uploaded_count == 5
    assert result.skipped_count == 3
    assert result.error_count == 0
    assert result.s3_url is not None
    assert "s3.console.aws.amazon.com" in result.s3_url
    assert "test-bucket" in result.s3_url

    # Verify each upload call has correct parameters
    for call in mock_s3_client.upload_file.call_args_list:
        args = call[0]
        local_file, bucket, s3_key = args

        # Check bucket name is correct
        assert bucket == "test-bucket"

        # Check S3 key has correct prefix structure: prefix/test_name/uuid/filename
        # The path now includes a unique UUID to prevent clobbering
        assert s3_key.startswith("prefix/test_name/") and "/" in s3_key[len("prefix/test_name/") :]

        # Check filename is one of the included files
        filename = os.path.basename(local_file)
        assert filename.lower().endswith((".neff", ".csv", ".log", ".txt"))

    # Verify summary log
    assert "5 uploaded" in caplog.text
    assert "3 skipped" in caplog.text
    assert "0 errors" in caplog.text


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_upload_with_subdirectories(mock_get_s3_client_and_session, caplog):
    """Test upload preserves subdirectory structure."""
    mock_s3_client = MagicMock()
    mock_session = MagicMock()
    mock_get_s3_client_and_session.return_value = (mock_s3_client, mock_session)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectory structure
        subdir = os.path.join(tmpdir, "subdir", "nested")
        os.makedirs(subdir)

        # Create file in subdirectory
        filepath = os.path.join(subdir, "test.log")
        with open(filepath, "w") as f:
            f.write("test content")

        s3_config = S3ArtifactUploadConfig(bucket="test-bucket", prefix="")
        # Upload artifacts
        upload_test_artifacts_to_s3(
            test_dir=tmpdir,
            s3_config=s3_config,
            test_name="test_name",
        )

    # Verify subdirectory structure is preserved in S3 key
    assert mock_s3_client.upload_file.call_count == 1
    args = mock_s3_client.upload_file.call_args[0]
    s3_key = args[2]
    # Path now includes unique UUID: test_name/uuid/subdir/nested/test.log
    assert s3_key.startswith("test_name/") and s3_key.endswith("/subdir/nested/test.log")


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_upload_failure_logging(mock_get_s3_client_and_session, caplog):
    """Test error logging when upload fails."""
    mock_s3_client = MagicMock()
    mock_s3_client.upload_file.side_effect = ClientError(
        error_response={"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}},
        operation_name="PutObject",
    )
    mock_session = MagicMock()
    mock_get_s3_client_and_session.return_value = (mock_s3_client, mock_session)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.neff")
        with open(filepath, "w") as f:
            f.write("test content")

        s3_config = S3ArtifactUploadConfig(bucket="test-bucket", prefix="")
        # Upload artifacts
        upload_test_artifacts_to_s3(
            test_dir=tmpdir,
            s3_config=s3_config,
            test_name="test_name",
        )

    # Verify error was logged
    assert "Failed to upload" in caplog.text
    assert "test.neff" in caplog.text
    # Verify summary shows error
    assert "1 errors" in caplog.text


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_upload_s3_bucket_without_prefix(mock_get_s3_client_and_session):
    """Test S3 bucket without prefix."""
    mock_s3_client = MagicMock()
    mock_session = MagicMock()
    mock_get_s3_client_and_session.return_value = (mock_s3_client, mock_session)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.log")
        with open(filepath, "w") as f:
            f.write("test content")

        s3_config = S3ArtifactUploadConfig(bucket="test-bucket", prefix="")
        upload_test_artifacts_to_s3(
            test_dir=tmpdir,
            s3_config=s3_config,
            test_name="test_name",
        )

    # Verify S3 key doesn't have extra slashes
    args = mock_s3_client.upload_file.call_args[0]
    s3_key = args[2]
    # Path now includes unique UUID: test_name/uuid/test.log
    assert s3_key.startswith("test_name/") and s3_key.endswith("/test.log") and not s3_key.startswith("/")


def test_validate_s3_credentials_empty_bucket():
    """Test validation fails for empty bucket name."""
    s3_config = S3ArtifactUploadConfig(bucket=None)
    with pytest.raises(ValueError, match="S3 bucket name cannot be empty"):
        validate_s3_credentials(s3_config)


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_validate_s3_credentials_success(mock_get_s3_client_and_session):
    """Test successful credential validation."""
    mock_s3_client = MagicMock()
    mock_session = MagicMock()
    mock_get_s3_client_and_session.return_value = (mock_s3_client, mock_session)

    # Validate with default credentials
    s3_config = S3ArtifactUploadConfig(bucket="my-bucket")
    validate_s3_credentials(s3_config)
    mock_s3_client.head_bucket.assert_called_once_with(Bucket="my-bucket")
    mock_get_s3_client_and_session.assert_called_once_with(None)

    # Validate with profile
    mock_s3_client.reset_mock()
    mock_get_s3_client_and_session.reset_mock()
    s3_config = S3ArtifactUploadConfig(bucket="my-bucket", profile="my-profile")
    validate_s3_credentials(s3_config)
    mock_s3_client.head_bucket.assert_called_once_with(Bucket="my-bucket")
    mock_get_s3_client_and_session.assert_called_once_with("my-profile")


@patch("test.utils.artifact_manager.get_s3_client_and_session")
def test_validate_s3_credentials_with_profile(mock_get_s3_client_and_session):
    """Test credential validation uses profile when provided."""
    mock_s3_client = MagicMock()
    mock_session = MagicMock()
    mock_get_s3_client_and_session.return_value = (mock_s3_client, mock_session)

    s3_config = S3ArtifactUploadConfig(bucket="my-bucket", profile="my-aws-profile")
    validate_s3_credentials(s3_config)

    # Verify profile was passed to get_s3_client_and_session
    mock_get_s3_client_and_session.assert_called_once_with("my-aws-profile")
    mock_s3_client.head_bucket.assert_called_once_with(Bucket="my-bucket")
