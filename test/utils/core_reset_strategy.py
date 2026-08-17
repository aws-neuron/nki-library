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
"""Per-capture NeuronCore reset decision for the test harness.

The capture's own ``nrt_init`` performs the reset when told to
(``NEURON_RT_RESET_CORES=1``); ``CoreResetStrategy`` only decides that env var.
"""

from __future__ import annotations

import glob
import json
import os
import re

from filelock import FileLock

from .core_lock_manager import CoreAllocation


def _allocation_physical_cores(logical_core_ids: list[int], lnc_config: int) -> list[int]:
    """Physical cores backing the given logical cores.

    Inverse of ``CoreLockManager._physical_to_logical_cores``: logical core L at
    ``lnc_config`` maps to physical cores ``[L*lnc + k for k in range(lnc)]``.
    """
    return [logical * lnc_config + k for logical in logical_core_ids for k in range(lnc_config)]


class CoreResetStrategy:
    """Single decision point for NEURON_RT_RESET_CORES.

    Shared/local runs always reset (the runtime default). Exclusive runs skip
    the per-capture reset unless core state demands one: first use of a core,
    a prior failed test on it, or an LNC-config transition (nrt_init at a
    different LNC than a core's last init wedges the TPB).

    State is a FileLock'd JSON file shared across xdist worker processes:
    ``{"<phys_core>": {"lnc": int, "failed": bool}}``.
    """

    def __init__(self, exclusive_run: bool, testrun_uid: str | None = None, ssh_alias: str | None = None) -> None:
        if exclusive_run and (not testrun_uid or not ssh_alias):
            raise ValueError("exclusive_run requires testrun_uid and ssh_alias")
        self._exclusive_run = exclusive_run
        self._state_path = self.build_state_path(testrun_uid, ssh_alias) if exclusive_run else None
        self._lock = FileLock(f"{self._state_path}.lock", timeout=30) if self._state_path else None

    def _read_state(self) -> dict:
        if self._state_path and os.path.exists(self._state_path):
            with open(self._state_path, "r") as f:
                return json.load(f)
        return {}

    def _write_state(self, state: dict) -> None:
        with open(self._state_path, "w") as f:
            json.dump(state, f)

    def should_reset_cores(self, core_allocation: CoreAllocation, lnc_config: int) -> bool:
        """Whether this capture must reset its cores (pure read — never mutates state)."""
        if not self._exclusive_run:
            return True  # shared/local: always reset (the runtime default)
        with self._lock:
            state = self._read_state()
            for core in _allocation_physical_cores(core_allocation.logical_core_ids, lnc_config):
                record = state.get(str(core))
                if record is None:  # first run on this core
                    return True
                if record["failed"]:  # prior failed test left it dirty
                    return True
                if record["lnc"] != lnc_config:  # LNC transition
                    # LNC transition requires reset: nrt_init at LNC2 on cores last initialized
                    # at LNC1 wedges the TPBs beyond userspace recovery (hardware-verified);
                    # the reverse direction happens to work but is not contractual, so any
                    # change forces a reset.
                    return True
        return False

    def mark_test_complete(self, core_allocation: CoreAllocation, lnc_config: int, failed: bool) -> None:
        """Record the capture outcome for each physical core (the only mutator)."""
        if not self._exclusive_run:
            return
        with self._lock:
            state = self._read_state()
            for core in _allocation_physical_cores(core_allocation.logical_core_ids, lnc_config):
                state[str(core)] = {"lnc": lnc_config, "failed": failed}
            self._write_state(state)

    @staticmethod
    def build_state_path(testrun_uid: str, ssh_alias: str) -> str:
        """Construct the state file path for a given session + host."""
        sanitized = re.sub(r"[^a-z0-9]", "_", ssh_alias.lower())
        return f"/tmp/nkilib_excl_{testrun_uid}_{sanitized}_reset_state.json"

    @staticmethod
    def cleanup_session_locks(testrun_uid: str) -> None:
        """Remove all reset-state/lock files for a session (controller only, post-workers)."""
        for path in glob.glob(f"/tmp/nkilib_excl_{testrun_uid}_*"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
