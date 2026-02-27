#!/usr/bin/env bash
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

# Removes legacy lock artifacts from remote hosts.
#
# Legacy locks (core_*_timeout_*m, legacy_lock_all_timeout_*m, atomic_lock)
# were used by the old two-phase locking protocol. They are no longer created
# by the current code and can be safely cleaned up.
#
# Usage:
#   ./cleanup_legacy_locks.sh <host1> [host2 ...]
#   ./cleanup_legacy_locks.sh --ssh-config ~/.ssh/config <host1> [host2 ...]

set -euo pipefail

LOCK_DIR="/tmp/neuronx-cc/core_locks"
SSH_CONFIG=""

usage() {
    echo "Usage: $0 [--ssh-config <path>] <host1> [host2 ...]"
    echo
    echo "Removes legacy lock artifacts from remote hosts:"
    echo "  - core_*_timeout_*m directories"
    echo "  - legacy_lock_all_timeout_*m directories"
    echo "  - atomic_lock file"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ssh-config)
            SSH_CONFIG="-F $2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    usage
fi

for host in "$@"; do
    echo "Cleaning legacy locks on ${host}..."
    # shellcheck disable=SC2086
    ssh ${SSH_CONFIG} "${host}" bash -c "'
        rm -f ${LOCK_DIR}/atomic_lock 2>/dev/null
        rmdir ${LOCK_DIR}/core_*_timeout_*m 2>/dev/null
        rmdir ${LOCK_DIR}/legacy_lock_all_timeout_*m 2>/dev/null
        echo \"  Done.\"
    '" || echo "  WARNING: failed to connect to ${host}"
done
