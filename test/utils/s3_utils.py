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
Shared S3 utilities for test infrastructure.
"""

import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.credentials import EnvProvider
from botocore.exceptions import ClientError

# Retry configuration for credential fetching
# The credential vendor rate-limits GetIAMRole (~3 TPS), so we back off when hitting limits
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds


@dataclass(frozen=True)
class S3Uri:
    """Parsed S3 URI with bucket and key prefix."""

    bucket: str
    prefix: str

    @property
    def uri(self) -> str:
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix}"
        return f"s3://{self.bucket}"


def parse_s3_uri(uri: str) -> S3Uri:
    """Parse an S3 URI into bucket and prefix components.

    Args:
        uri: S3 URI (e.g. 's3://my-bucket/path/to/prefix/')

    Returns:
        S3Uri with bucket and prefix (trailing slashes stripped from prefix).

    Raises:
        ValueError: If the URI scheme is not 's3'.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    return S3Uri(bucket=bucket, prefix=prefix)


class S3TransferDirection(Enum):
    """Direction of S3 file transfer relative to local dev machine."""

    INPUTS = "inputs"  # Local -> S3 -> Remote (upload to remote)
    OUTPUTS = "outputs"  # Remote -> S3 -> Local (download from remote)


@dataclass
class S3ArtifactUploadConfig:
    """Configuration for S3-based artifact uploads."""

    bucket: str | None = None
    prefix: str | None = None
    profile: str | None = None
    # Region of ``bucket``. Explicitly carried so the remote ``aws s3 cp`` signs
    # for and connects to the bucket's real region rather than resolving the
    # host's own region. Critical in EC2 Local Zones (e.g. us-east-2-sbn-1),
    # whose region string has no S3 endpoint, and it also removes the
    # wrong-region -> redirect round trip on every cross-region transfer.
    # If None, resolved on demand from the bucket via ``get_bucket_region``.
    region: str | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_enabled(self) -> bool:
        """Check if S3 upload is enabled (bucket must be specified)."""
        return self.bucket is not None

    def get_session_prefix(self) -> str:
        """
        Get the full S3 prefix including session ID for load distribution.

        Returns prefix like: artifacts_tmp/550e8400-e29b-41d4-a716-446655440000
        This distributes S3 requests across partitions to avoid rate limiting.
        """
        base_prefix = self.prefix or ""
        return f"{base_prefix}/{self.session_id}" if base_prefix else self.session_id


def generate_s3_key(config: S3ArtifactUploadConfig, direction: S3TransferDirection) -> str:
    """
    Generate a unique S3 key for file transfer.

    Args:
        config: S3 configuration with bucket, prefix, and session info
        direction: S3TransferDirection.INPUTS or S3TransferDirection.OUTPUTS

    Returns:
        S3 key like: artifacts_tmp/{session_id}/inputs/{uuid}.tar
    """
    transfer_uuid = str(uuid.uuid4())
    return f"{config.get_session_prefix()}/{direction.value}/{transfer_uuid}.tar"


def build_remote_s3_cli_command(
    s3_bucket: str,
    s3_key: str,
    remote_path: str,
    creds,
    direction: S3TransferDirection,
    region: str | None = None,
) -> str:
    """
    Build AWS CLI command with embedded credentials for remote execution.

    The command direction is determined by the transfer direction:
    - INPUTS: Remote downloads from S3 (S3 -> remote)
    - OUTPUTS: Remote uploads to S3 (remote -> S3)

    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 object key
        remote_path: File path on remote machine
        creds: Frozen credentials from boto3 session
        direction: S3TransferDirection indicating the overall transfer flow
        region: Region of the bucket. When provided, passed as ``--region`` so
            the CLI signs for and connects to the bucket's region instead of
            resolving the remote host's own region (which has no S3 endpoint in
            a Local Zone and forces a wrong-region redirect elsewhere).

    Returns:
        AWS CLI command string with embedded credentials
    """
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    env_vars = (
        f"AWS_ACCESS_KEY_ID={creds.access_key} "
        f"AWS_SECRET_ACCESS_KEY={creds.secret_key} "
        f"{f'AWS_SESSION_TOKEN={creds.token} ' if creds.token else ''}"
    )
    region_arg = f" --region {region}" if region else ""
    # Quote remote_path to handle special characters like < > ' in test names
    if direction == S3TransferDirection.INPUTS:
        # Remote downloads from S3
        return f"{env_vars}aws s3 cp {s3_uri} '{remote_path}'{region_arg}"
    else:
        # Remote uploads to S3
        return f"{env_vars}aws s3 cp '{remote_path}' {s3_uri}{region_arg}"


@lru_cache(maxsize=None)
def get_bucket_region(bucket: str, profile: str | None = None) -> str:
    """Resolve the AWS region a bucket lives in (cached per bucket/profile).

    Uses ``head_bucket`` and reads the ``x-amz-bucket-region`` response header,
    which is returned even when the caller's client is pointed at a different
    region (boto3 transparently follows the redirect). Preferred over
    ``get_bucket_location``, which returns ``None`` for us-east-1 and requires
    the ``s3:GetBucketLocation`` permission.

    The lookup runs against the local (Hydra worker) boto3 session, whose region
    already matches the bucket, so it does not incur the redirect it exists to
    prevent on the remote host.
    """
    s3_client = get_boto_client("s3", profile=profile)
    response = s3_client.head_bucket(Bucket=bucket)
    region = response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("x-amz-bucket-region")
    if not region:
        raise ValueError(f"Could not determine region for bucket {bucket!r} from head_bucket response")
    return region


def _is_throttling_error(exception: Exception) -> bool:
    """
    Check if an exception is a throttling/rate limit error.

    Uses boto3's recommended error handling pattern for ClientError,
    with fallback to string matching for other exception types.

    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
    """
    # Known throttling error codes from AWS services
    throttling_error_codes = {
        "Throttling",
        "ThrottlingException",
        "RequestLimitExceeded",
        "ProvisionedThroughputExceededException",
        "TooManyRequestsException",
        "RequestThrottled",
        "BandwidthLimitExceeded",
        "SlowDown",  # S3-specific throttling
    }

    # Check ClientError using boto3's recommended pattern
    if isinstance(exception, ClientError):
        error_code = exception.response.get("Error", {}).get("Code", "")
        if error_code in throttling_error_codes:
            return True

    # Fallback: string matching for non-ClientError exceptions (e.g., from credential providers)
    # This handles credential-vendor errors which may not be wrapped in ClientError
    error_str = str(exception).lower()
    string_indicators = ["throttl", "rate exceeded", "rate limit", "too many requests"]
    return any(indicator in error_str for indicator in string_indicators)


def _create_boto3_session_with_retry(
    profile: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> boto3.Session:
    """
    Create a boto3 session with retry logic for throttling errors during credential fetching.

    Note: boto3's built-in retry (Config retries) only applies to API calls made by clients,
    not to the credential fetching process itself. The throttling we're handling here occurs
    when the credential vendor fetches credentials, which happens before we have a boto3 client.

    Uses exponential backoff with jitter to handle credential-vendor rate limiting: the
    vendor rate-limits GetIAMRole (~3 TPS), which can be exceeded when multiple
    processes/containers fetch credentials simultaneously.

    Args:
        profile: Optional AWS profile name for authentication.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        boto3.Session with valid credentials.

    Raises:
        Exception: If all retry attempts fail.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            # Force credential resolution to detect throttling errors early
            creds = session.get_credentials()
            if creds is not None:
                # Trigger actual credential fetch (may involve a credential-vendor call)
                creds.get_frozen_credentials()
            return session
        except Exception as e:
            if attempt == max_retries:
                logging.error(f"Failed to create boto3 session after {max_retries + 1} attempts: {e}")
                raise

            # Only retry throttling errors - other errors should fail fast
            if not _is_throttling_error(e):
                logging.error(f"Credential fetch failed with non-throttling error: {e}")
                raise

            # Exponential backoff with jitter for throttling errors
            last_exception = e
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
            logging.warning(
                f"Credential fetch throttled (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.2f}s: {e}"
            )
            time.sleep(delay)

    # Should only reach here after exhausting retries on throttling errors
    raise last_exception if last_exception else RuntimeError("Failed to create boto3 session")


def prefetch_and_cache_credentials(profile: str | None = None) -> None:
    """
    Pre-fetch AWS credentials and cache them in environment variables.

    This should be called once from the main pytest process before xdist workers spawn.
    Workers will then use these pre-fetched credentials instead of calling the credential
    vendor, avoiding rate limiting when multiple workers start simultaneously.

    Args:
        profile: Optional AWS profile name for authentication.
    """
    logging.info(f"Pre-fetching AWS credentials for profile: {profile or 'default'}")
    try:
        session = _create_boto3_session_with_retry(profile)
        creds = session.get_credentials()
        if creds is None:
            logging.warning("No AWS credentials found, skipping pre-fetch")
            return

        frozen_creds = creds.get_frozen_credentials()
        os.environ[EnvProvider.ACCESS_KEY] = frozen_creds.access_key
        os.environ[EnvProvider.SECRET_KEY] = frozen_creds.secret_key
        if frozen_creds.token:
            os.environ[EnvProvider.TOKENS[1]] = (
                frozen_creds.token
            )  # EnvProvider.TOKENS = ['AWS_SECURITY_TOKEN', 'AWS_SESSION_TOKEN']
        logging.info("Successfully pre-fetched and cached AWS credentials")
    except Exception as e:
        logging.warning(f"Failed to pre-fetch AWS credentials: {e}. Workers will fetch their own credentials.")


# Standard client-side retry for AWS API calls. Built fresh per client: botocore
# normalizes a Config's ``retries`` dict IN PLACE when constructing a client (e.g.
# ``max_attempts`` -> ``total_max_attempts``), so a shared module-global Config would be
# mutated out from under later callers (and tests asserting on it).
def _boto_client_retry_config() -> Config:
    return Config(retries={"max_attempts": 5, "mode": "standard"})


@lru_cache(maxsize=None)
def _get_boto_session_cached(profile: str | None) -> boto3.Session:
    """Helper to ensure canonical form for caching"""
    return _create_boto3_session_with_retry(profile)


def get_boto_session(profile: str | None = None) -> boto3.Session:
    """Get or create a boto3 Session (cached per profile).

    Uses retry logic for credential fetching to handle credential-vendor rate limiting.
    Cached so every caller in a process shares one session — and thus one
    credential fetch — per profile.

    boto3 picks up AWS_* env vars automatically if they were pre-fetched (see
    prefetch_and_cache_credentials).
    """
    return _get_boto_session_cached(profile)


@lru_cache(maxsize=None)
def _get_boto_client_cached(service: str, region: str | None, profile: str | None) -> BaseClient:
    """Helper to ensure canonical form for caching"""
    session = get_boto_session(profile)
    return session.client(service, region_name=region, config=_boto_client_retry_config())


def get_boto_client(service: str, *, region: str | None = None, profile: str | None = None) -> BaseClient:
    """Get or create a boto3 client for ``service`` (cached per service/region/profile).

    Backed by the shared per-profile session, so clients across services and regions
    reuse one session (one credential fetch). Configured with standard API retry.

    Args:
        service: AWS service name (e.g. "s3", "ec2", "autoscaling").
        region: Region for the client; None uses the session/default region.
        profile: Optional AWS profile name for authentication.
    """
    return _get_boto_client_cached(service, region, profile)


def get_s3_client_and_session(profile: str | None = None):
    """
    Get or create boto3 S3 client and session (cached).

    Thin wrapper over the shared ``get_boto_client`` / ``get_boto_session`` helpers,
    returning both because callers need each for different things:
    - Client for S3 operations (upload, download, head_bucket, etc.)
    - Session for extracting credentials (needed for remote AWS CLI commands)

    Args:
        profile: Optional AWS profile name for authentication.
                 If None, uses default AWS credentials (including env vars).

    Returns:
        Tuple of (s3_client, session)

    Example:
        # Default credentials
        s3_client, session = get_s3_client_and_session()

        # With profile
        s3_client, session = get_s3_client_and_session(profile="my-aws-profile")

        # Extract credentials for remote commands
        creds = session.get_credentials().get_frozen_credentials()
    """
    return get_boto_client("s3", profile=profile), get_boto_session(profile)
