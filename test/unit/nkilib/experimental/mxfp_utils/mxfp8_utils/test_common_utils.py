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

"""Thread isolation of the mxfp8 active SbufManager.

Regression test for the LNC2 simulator race: the NKI simulator runs the kernel body
in one thread per LNC core, so the active-SBM slot must be per-thread. When it was a
shared module global, one core's ``clear_active_sbm()`` could null the manager while
another core was mid-kernel, giving a non-deterministic ``'NoneType' object has no
attribute 'alloc_stack'`` failure under simulation only.
"""

import threading

from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils import _common_utils


def test_active_sbm_is_thread_local():
    """Two threads hold independent active SbufManagers concurrently.

    With a shared global this either trips the "already set" assert or makes both
    threads observe the same manager; with per-thread state each thread owns its own.
    """
    both_set = threading.Barrier(2, timeout=30)
    seen: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def worker(name: str) -> None:
        try:
            # A freshly spawned thread starts with no active manager.
            assert _common_utils.get_active_sbm() is None, "fresh thread should have no SBM"  # noqa: S101
            _common_utils.create_and_set_active_sbm()
            both_set.wait()  # both threads now hold their own manager at the same time
            seen[name] = _common_utils.get_active_sbm()
            _common_utils.clear_active_sbm()
            # Clearing this thread's manager must not be observable as anything but None here.
            assert _common_utils.get_active_sbm() is None, "clear should reset this thread's SBM"  # noqa: S101
        except BaseException as exc:  # noqa: BLE001 - surface worker failure in the main thread
            errors[name] = exc

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("core0", "core1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"worker threads raised: {errors}"  # noqa: S101
    assert seen["core0"] is not None and seen["core1"] is not None  # noqa: S101
    assert seen["core0"] is not seen["core1"], "threads must not share one active-SBM slot"  # noqa: S101
