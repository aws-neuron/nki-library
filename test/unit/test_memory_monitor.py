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
"""Unit tests for ProcessTreeMemoryMonitor."""

import ctypes
import os
import time

import pytest

from ..utils.memory_monitor import (
    MemoryLimitExceeded,
    ProcessTreeMemoryMonitor,
    get_process_tree_memory_bytes,
    memory_metric_available,
    release_process_memory,
)


def test_monitor_tracks_peak():
    """Monitor should return a non-zero peak for the current process."""
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    monitor.start()
    time.sleep(0.3)
    snapshot = monitor.stop()
    assert snapshot.peak_bytes > 0
    assert snapshot.peak_mb > 0


def test_monitor_captures_baseline():
    """Monitor should record a baseline at start and compute delta."""
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    monitor.start()
    time.sleep(0.3)
    snapshot = monitor.stop()
    assert snapshot.baseline_bytes > 0
    assert snapshot.peak_bytes >= snapshot.baseline_bytes
    assert snapshot.delta_bytes == snapshot.peak_bytes - snapshot.baseline_bytes
    assert snapshot.delta_mb >= 0


def test_memory_limit_exceeded_raises():
    """Monitor with a limit below current usage should raise MemoryLimitExceeded."""
    # Use a 10 MB limit and allocate ~50 MB to guarantee the delta exceeds it
    # even when run after 670+ other tests with a noisy baseline.
    limit = 10 * 1024 * 1024
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1, memory_limit_bytes=limit)
    monitor.start()
    with pytest.raises(MemoryLimitExceeded, match="exceeded --memory-limit"):
        # Allocate ~50 MB so the delta clearly exceeds the 10 MB limit.
        # bytearray is backed by a single C malloc (mmap for >128KB),
        # so new pages are faulted in and resident memory grows immediately.
        _buf = bytearray(50 * 1024 * 1024)
        time.sleep(2)


def test_memory_limit_checks_delta():
    """Memory limit should be checked against delta, not absolute usage.

    A limit larger than the process baseline but set to 1 byte below the peak should
    not trigger because the delta (new allocations) during sleep is ~0.
    """
    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    monitor.start()
    time.sleep(0.2)
    baseline_snapshot = monitor.stop()

    # Set limit well above any realistic delta but well below absolute usage.
    # If limit were checked against absolute usage, this would always fire.
    limit = baseline_snapshot.peak_bytes - 1
    if limit <= 0:
        pytest.skip("Process memory too low to test delta vs absolute distinction")

    monitor = ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1, memory_limit_bytes=limit)
    monitor.start()
    # Sleep without allocating significant memory — delta should stay near 0
    time.sleep(0.5)
    snapshot = monitor.stop()
    # Should not have raised — delta is small even though absolute usage > limit
    assert snapshot.delta_mb < snapshot.peak_mb


def test_release_process_memory_runs_without_error():
    """release_process_memory should not raise on Linux."""
    release_process_memory()


def test_memory_metric_available_on_linux():
    """The per-process memory metric is expected to be available on the supported platform."""
    assert memory_metric_available() is True


def test_start_fails_loudly_when_metric_unavailable(monkeypatch):
    """If the memory metric is unavailable, start() must raise rather than under-report.

    A memory guard that cannot measure memory should refuse to run, not quietly
    disable the --memory-limit check.
    """
    import test.utils.memory_monitor as mm

    monkeypatch.setattr(mm, "memory_metric_available", lambda: False)
    monitor = mm.ProcessTreeMemoryMonitor(os.getpid(), interval_seconds=0.1)
    with pytest.raises(RuntimeError, match="smaps_rollup"):
        monitor.start()


def test_tree_memory_not_doubled_by_forked_child():
    """Tree memory must not double-count copy-on-write pages of a freshly forked child.

    Reproduces the fork/exec window the compiler backend goes through: a child that
    has forked but not yet replaced its address space shares all of the parent's pages
    copy-on-write. Summing plain RSS across the tree would count those shared pages
    twice; the metric used here (PSS) must not. The tree total during the fork window
    should stay close to the parent-alone total, not ~2x it. This is the regression
    guard for the phantom multi-GB spikes that flakily tripped --memory-limit.
    """
    # Allocate ~200 MB of private, resident memory in the parent.
    size = 200 * 1024 * 1024
    buf = (ctypes.c_char * size)()
    ctypes.memset(buf, 1, size)

    parent_alone = get_process_tree_memory_bytes(os.getpid())

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: do NOT write the inherited pages; just wait, then exit
        os.close(w)
        try:
            os.read(r, 1)
        finally:
            os._exit(0)

    try:
        os.close(r)
        time.sleep(0.3)  # let the child settle in the copy-on-write sharing state
        tree_during_fork = get_process_tree_memory_bytes(os.getpid())
        # The metric splits shared pages, so the tree total stays near parent-alone.
        # Allow generous headroom for the child's own interpreter pages; the key
        # assertion is that it is NOT ~2x (which summed RSS would produce).
        assert tree_during_fork < parent_alone + 100 * 1024 * 1024
    finally:
        os.write(w, b"x")
        os.close(w)
        os.waitpid(pid, 0)
