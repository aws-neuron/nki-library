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
Test artifact management utilities.

Handles artifact lifecycle operations including:
- Uploading artifacts to S3 based on test outcomes
- File filtering to control what gets uploaded
"""

import logging
import os
import uuid
from dataclasses import dataclass
from enum import Enum
from test.utils.s3_utils import S3ArtifactUploadConfig, get_s3_client_and_session
from urllib.parse import quote

from botocore.exceptions import BotoCoreError, ClientError


class UploadOutcome(str, Enum):
    """Valid values for upload_test_outcomes configuration."""

    ALL = "all"  # Upload on all outcomes
    SUCCESS = "success"  # Upload only on passed tests
    FAIL = "fail"  # Upload only on failed tests

    def __str__(self):
        """Return the enum value as string for backwards compatibility."""
        return self.value


# File extensions to include in S3 upload (all others excluded)
INCLUDE_EXTENSIONS = [
    ".neff",  # Compiled neuron executable files
    ".csv",  # Metrics and CSV data files
    ".log",  # Log files
    ".txt",  # Text files
]


@dataclass
class S3UploadResult:
    """Result of an S3 upload operation."""

    uploaded_count: int
    skipped_count: int
    error_count: int
    s3_url: str | None  # Console URL to the S3 location, None if no files uploaded


def should_upload_artifacts(test_outcome: str, upload_test_outcomes: str) -> bool:
    """
    Determine if artifacts should be uploaded based on test outcome and upload configuration.

    Args:
        test_outcome: "passed", "failed", "skipped", etc.
        upload_test_outcomes: UploadOutcome value or compatible string

    Returns:
        True if artifacts should be uploaded, False otherwise
    """
    if not upload_test_outcomes:
        return False

    upload_config = str(upload_test_outcomes)  # Handle both enum and string
    if upload_config == UploadOutcome.ALL.value:
        return True
    elif upload_config == UploadOutcome.SUCCESS.value:
        return test_outcome == "passed"
    elif upload_config == UploadOutcome.FAIL.value:
        return test_outcome == "failed"
    return False


def should_include_file_in_upload(file_path: str) -> bool:
    """
    Determine if a file should be included in S3 upload.

    Only files with extensions in INCLUDE_EXTENSIONS are uploaded.
    All other files are excluded by default.

    Args:
        file_path: Path to the file

    Returns:
        True if file should be uploaded, False otherwise
    """
    for ext in INCLUDE_EXTENSIONS:
        if file_path.lower().endswith(ext):
            return True
    return False


def validate_s3_credentials(s3_config: S3ArtifactUploadConfig):
    """
    Validate S3 bucket access and credentials.

    Let boto3 exceptions propagate naturally for clearer error messages.

    Args:
        s3_config: S3 configuration containing bucket, prefix, and profile

    Returns:
        None

    Raises:
        ValueError: If bucket name is invalid
        botocore.exceptions.*: boto3 exceptions for credential/access issues
    """
    if not s3_config.bucket:
        raise ValueError("S3 bucket name cannot be empty")

    # Validate credentials and bucket access (let boto3 exceptions propagate)
    s3_client, _ = get_s3_client_and_session(s3_config.profile)
    s3_client.head_bucket(Bucket=s3_config.bucket)
    logging.info(f"S3 authentication verified for bucket: {s3_config.bucket}")


def upload_test_artifacts_to_s3(test_dir: str, s3_config: S3ArtifactUploadConfig, test_name: str) -> S3UploadResult:
    """
    Upload filtered test artifacts to S3.

    Args:
        test_dir: Local directory containing test artifacts
        s3_config: S3 configuration containing bucket, prefix, and profile
        test_name: Name of the test for S3 path organization

    Returns:
        S3UploadResult containing upload statistics and console URL
    """
    if not os.path.exists(test_dir):
        logging.warning(f"Test directory does not exist, skipping S3 upload: {test_dir}")
        return S3UploadResult(uploaded_count=0, skipped_count=0, error_count=0, s3_url=None)

    s3_client, _ = get_s3_client_and_session(s3_config.profile)

    # Build S3 path with unique identifier to prevent parallel uploads from clobbering
    # Format: prefix/test_name/uuid/
    upload_uuid = str(uuid.uuid4())
    prefix = s3_config.prefix or ""
    s3_base = f"{prefix}/{test_name}/{upload_uuid}" if prefix else f"{test_name}/{upload_uuid}"
    logging.info(f"S3 upload: Uploading artifacts to s3://{s3_config.bucket}/{s3_base}/")

    uploaded_count = 0
    skipped_count = 0
    error_count = 0

    # Upload each file that passes the filter
    for root, _, files in os.walk(test_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if should_include_file_in_upload(file_path):
                relative_path = os.path.relpath(file_path, test_dir)
                s3_key = f"{s3_base}/{relative_path}"
                try:
                    s3_client.upload_file(file_path, s3_config.bucket, s3_key)
                    logging.info(f"Uploaded {relative_path} to s3://{s3_config.bucket}/{s3_key}")
                    uploaded_count += 1
                except (BotoCoreError, ClientError) as e:
                    logging.error(f"Failed to upload {relative_path} to S3: {e}")
                    error_count += 1
            else:
                skipped_count += 1

    logging.info(f"S3 upload: Completed - {uploaded_count} uploaded, {skipped_count} skipped, {error_count} errors")

    # Build S3 console URL for easy access
    s3_url = None
    if uploaded_count > 0:
        # URL-encode the prefix to handle special characters like brackets in test names
        encoded_prefix = quote(f"{s3_base}/", safe="")
        s3_url = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_config.bucket}?prefix={encoded_prefix}"

    return S3UploadResult(
        uploaded_count=uploaded_count, skipped_count=skipped_count, error_count=error_count, s3_url=s3_url
    )
