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
"""CPU-time based test timeout plugin for pytest.

Adds a CPU-time watchdog that hooks into pytest-timeout's existing machinery.
Two layers, one signal:
  - CPU watchdog: polls RUSAGE_SELF + RUSAGE_CHILDREN every 2 s
  - Wall-clock ITIMER: pytest-timeout's existing ITIMER_REAL

Both layers share a single SIGALRM handler.  Whichever fires first triggers
pytest-timeout's own ``timeout_sigalrm()`` for stack dumps and test failure.
"""

from __future__ import annotations

import logging
import os
import resource
import signal
import threading
import time
from typing import Callable, Optional, Protocol, runtime_checkable

import pytest
import pytest_timeout

from .feature_flag_helper import get_feature_flag

logger: logging.Logger = logging.getLogger(__name__)


@runtime_checkable
class _CancellableItem(Protocol):
    """A collected test whose timeout has been armed, exposing the disarm hook.

    The hook is a plain attribute on the item by convention of the wall-clock
    timeout plugin, which both sets and later looks it up by that name, so the
    CPU-time layer has to extend it in place rather than keep its own copy.
    """

    cancel_timeout: Callable[[], None]


def _get_total_cpu() -> float:
    """Return CPU time of current process + all waited-for children.

    Sums user + system time from both ``RUSAGE_SELF`` and ``RUSAGE_CHILDREN``.
    """
    r_self: resource.struct_rusage = resource.getrusage(resource.RUSAGE_SELF)
    r_children: resource.struct_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r_self.ru_utime + r_self.ru_stime + r_children.ru_utime + r_children.ru_stime


class CPUTimeWatchdog(threading.Thread):
    """Polls total CPU time; triggers timeout when budget exceeded."""

    def __init__(
        self,
        max_cpu: float,
        start_cpu: float,
        trigger: Callable[[], None],
        poll_interval: float = 2.0,
    ) -> None:
        super().__init__(daemon=True)
        self.max_cpu: float = max_cpu
        self.start_cpu: float = start_cpu
        self.trigger: Callable[[], None] = trigger
        self.poll_interval: float = poll_interval
        self._stop_event: threading.Event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            elapsed_cpu: float = _get_total_cpu() - self.start_cpu
            if elapsed_cpu > self.max_cpu:
                logger.debug(
                    "watchdog: cpu budget exceeded  elapsed=%.3fs > limit=%.1fs",
                    elapsed_cpu,
                    self.max_cpu,
                )
                self.trigger()
                break

    def stop(self) -> None:
        self._stop_event.set()


def _get_cpu_timeout(item: pytest.Item) -> float | None:
    """Resolve CPU timeout: marker > CLI > ``None``."""
    marker: pytest.Mark | None = item.get_closest_marker("cpu_timeout")
    if marker is not None:
        return float(marker.args[0])

    cpu_timeout_seconds: float | None = get_feature_flag(item.config, "cpu_timeout", None)
    return cpu_timeout_seconds


def prepare_timeout_watchdog(item: pytest.Item, settings: pytest_timeout.Settings) -> Optional[bool]:
    """Hook into pytest-timeout's timer setup to add CPU-time watchdog."""
    cpu_limit: Optional[float] = _get_cpu_timeout(item)
    if cpu_limit is None:
        return None  # Default pytest-timeout, unchanged

    # Let pytest-timeout set up wall-clock ITIMER + cancel function
    pytest_timeout.pytest_timeout_set_timer(item, settings)

    # Resolve effective method (same logic pytest-timeout uses)
    timeout_method: str = settings.method
    if timeout_method == "signal" and threading.current_thread() is not threading.main_thread():
        timeout_method = "thread"

    start_cpu: float = _get_total_cpu()
    start_wall: float = time.monotonic()
    logger.debug(
        "%s: cpu-timeout armed  cpu_limit=%.1fs  wall_limit=%.1fs  start_cpu=%.3fs",
        item.nodeid,
        cpu_limit,
        settings.timeout,
        start_cpu,
    )

    if timeout_method == "signal":
        cpu_settings: pytest_timeout.Settings = settings._replace(timeout=cpu_limit)
        fired: threading.Event = threading.Event()

        def handler(_signum: int, _frame: object) -> None:
            __tracebackhide__ = True
            if fired.is_set():
                return
            fired.set()

            elapsed_cpu: float = _get_total_cpu() - start_cpu
            if elapsed_cpu > cpu_limit:
                pytest_timeout.timeout_sigalrm(item, cpu_settings)
            else:
                pytest_timeout.timeout_sigalrm(item, settings)

        # Overwrite pytest-timeout's handler; ITIMER_REAL keeps ticking
        _ = signal.signal(signal.SIGALRM, handler)

        def trigger() -> None:
            os.kill(os.getpid(), signal.SIGALRM)
    else:

        def trigger() -> None:
            pytest_timeout.timeout_timer(item, settings)

    watchdog: CPUTimeWatchdog = CPUTimeWatchdog(cpu_limit, start_cpu, trigger)

    # Extend pytest-timeout's cancel to also stop watchdog
    assert isinstance(item, _CancellableItem), "the wall-clock timer must be armed before the CPU watchdog"
    original_cancel: Callable[[], None] = item.cancel_timeout

    def cancel() -> None:
        watchdog.stop()
        original_cancel()  # disarms ITIMER_REAL + restores SIG_DFL
        watchdog.join(timeout=5)
        elapsed_cpu: float = _get_total_cpu() - start_cpu
        elapsed_wall: float = time.monotonic() - start_wall
        logger.debug(
            "%s: cpu-timeout disarmed  cpu=%.3fs/%.1fs  wall=%.3fs/%.1fs",
            item.nodeid,
            elapsed_cpu,
            cpu_limit,
            elapsed_wall,
            settings.timeout,
        )

    item.cancel_timeout = cancel
    watchdog.start()

    return True
