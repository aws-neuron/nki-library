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
Infrastructure version checking for shared fleet locking compatibility.

This module provides version checking to ensure clients are compatible with
the locking protocol used by the shared fleet infrastructure. The version
check runs before any locking operations to fail fast with a clear error
message if the client needs to be updated.

Usage:
    from test.utils.core_lock_manager import check_infra_locking_version

    # In conftest.py or before acquiring locks:
    check_infra_locking_version(ssh_connection)
"""

import json
import logging

import fabric2

logger = logging.getLogger(__name__)

# Locking protocol version - bump this when making incompatible changes
# Infra will set minClientLockingVersion, and clients must have at least that version
LOCKING_PROTOCOL_VERSION = 1

# JSON key for minimum client version in infra_version.json
MIN_CLIENT_VERSION_KEY = "minClientLockingVersion"

# Remote path for infrastructure version file
REMOTE_LOCK_DIR = "/tmp/neuronx-cc/core_locks"
REMOTE_INFRA_VERSION_JSON = f"{REMOTE_LOCK_DIR}/infra_version.json"


class InfraVersionError(Exception):
    """
    Raised when the infrastructure requires a newer locking protocol version.

    This error indicates that the shared fleet infrastructure has been updated
    to use a newer locking protocol that this client doesn't support. The user
    needs to update their nkilib installation.
    """

    def __init__(self, required_version: int, current_version: int):
        self.required_version = required_version
        self.current_version = current_version
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"\n"
            f"{'=' * 60}\n"
            f"SHARED FLEET VERSION MISMATCH\n"
            f"{'=' * 60}\n"
            f"Infrastructure requires locking protocol v{self.required_version}, "
            f"but this client only supports v{self.current_version}.\n"
            f"\n"
            f"Please update your nkilib installation:\n"
            f"  brazil ws use -p KaenaNeuronKernelLibrary\n"
            f"  brazil ws sync --md\n"
            f"{'=' * 60}"
        )


def check_infra_locking_version(conn: fabric2.Connection) -> None:
    """
    Check that the client supports the infrastructure's locking protocol version.

    This function reads the infra_version.json file from the remote host and
    verifies that the client's LOCKING_PROTOCOL_VERSION meets the minimum
    required by the infrastructure.

    Args:
        conn: Active fabric2 Connection to the remote host

    Raises:
        InfraVersionError: If the infrastructure requires a newer locking protocol
            version than this client supports
    """
    try:
        result = conn.run(
            f"cat {REMOTE_INFRA_VERSION_JSON} 2>/dev/null || echo '{{}}'",
            hide=True,
            warn=True,
        )

        if result.ok and result.stdout.strip():
            try:
                infra_version = json.loads(result.stdout.strip())

                if MIN_CLIENT_VERSION_KEY not in infra_version:
                    # File doesn't exist or is empty
                    logger.info(f"[{conn.host}] No infra_version.json found")
                    return

                required_version = infra_version[MIN_CLIENT_VERSION_KEY]

                if required_version > LOCKING_PROTOCOL_VERSION:
                    raise InfraVersionError(
                        required_version=required_version,
                        current_version=LOCKING_PROTOCOL_VERSION,
                    )

                logger.info(
                    f"[{conn.host}] Infra version check passed "
                    f"(required: v{required_version}, client: v{LOCKING_PROTOCOL_VERSION})"
                )
            except json.JSONDecodeError:
                logger.warning(f"[{conn.host}] Corrupted infra_version.json, skipping version check")
    except InfraVersionError:
        raise  # Re-raise version errors - these should fail the test
    except Exception as e:
        # Don't fail on connection errors - let the actual locking code handle that
        logger.warning(f"[{conn.host}] Could not check infra version: {e}")
