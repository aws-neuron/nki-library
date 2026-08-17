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
"""Tests for cpu_timeout module - CPU-time based test timeout plugin."""

from __future__ import annotations

import resource
import signal
import threading
import time
from types import FrameType
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

import pytest
import pytest_timeout

from test.utils.cpu_timeout import (
    CPUTimeWatchdog,
    _get_cpu_timeout,
    _get_total_cpu,
    prepare_timeout_watchdog,
)

Settings = pytest_timeout.Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _burn_cpu(seconds: float) -> None:
    """Consume roughly *seconds* of CPU time via busy-wait."""
    end: float = time.process_time() + seconds
    while time.process_time() < end:
        pass


def _make_mock_item(
    marker_value: Optional[float] = None,
    cli_value: Optional[float] = None,
) -> MagicMock:
    """Build a minimal mock pytest.Item for _get_cpu_timeout."""
    item: MagicMock = MagicMock(spec=pytest.Item)
    if marker_value is not None:
        marker: MagicMock = MagicMock()
        marker.args = (marker_value,)
        item.get_closest_marker = MagicMock(return_value=marker)
    else:
        item.get_closest_marker = MagicMock(return_value=None)
    item.config.getoption = MagicMock(return_value=cli_value)
    return item


def _installed_sigalrm_handler() -> Callable[[int, FrameType | None], Any]:
    """Return the installed SIGALRM handler, rejecting the SIG_DFL/SIG_IGN sentinels."""
    handler = signal.getsignal(signal.SIGALRM)
    assert handler is not None and not isinstance(handler, int), "no SIGALRM handler is installed"
    return handler


# ===================================================================
# _get_total_cpu
# ===================================================================


class TestGetTotalCpu:
    def test_returns_positive_float(self) -> None:
        result: float = _get_total_cpu()
        assert isinstance(result, float)
        assert result > 0.0

    def test_sums_self_and_children_components(self) -> None:
        """Verify all four rusage components are summed."""
        fake_self: MagicMock = MagicMock(ru_utime=1.0, ru_stime=2.0)
        fake_children: MagicMock = MagicMock(ru_utime=3.0, ru_stime=4.0)
        with patch(
            "test.utils.cpu_timeout.resource.getrusage",
            side_effect=[fake_self, fake_children],
        ) as mock_getrusage:
            result: float = _get_total_cpu()
        assert result == 10.0
        mock_getrusage.assert_any_call(resource.RUSAGE_SELF)
        mock_getrusage.assert_any_call(resource.RUSAGE_CHILDREN)

    def test_increases_after_cpu_work(self) -> None:
        before: float = _get_total_cpu()
        _burn_cpu(0.05)
        after: float = _get_total_cpu()
        assert after > before


# ===================================================================
# CPUTimeWatchdog
# ===================================================================


class TestCPUTimeWatchdog:
    def test_does_not_trigger_under_budget(self) -> None:
        triggered: threading.Event = threading.Event()
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=9999.0,
            start_cpu=_get_total_cpu(),
            trigger=triggered.set,
            poll_interval=0.05,
        )
        watchdog.start()
        time.sleep(0.15)
        watchdog.stop()
        watchdog.join(timeout=2)
        assert not triggered.is_set()

    def test_triggers_when_budget_exceeded(self) -> None:
        triggered: threading.Event = threading.Event()
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=0.0,
            start_cpu=_get_total_cpu(),
            trigger=triggered.set,
            poll_interval=0.05,
        )
        watchdog.start()
        _burn_cpu(0.01)
        assert triggered.wait(timeout=3), "Watchdog should have triggered"
        watchdog.stop()
        watchdog.join(timeout=2)

    def test_stop_prevents_trigger(self) -> None:
        triggered: threading.Event = threading.Event()
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=0.0,
            start_cpu=_get_total_cpu() + 99999,  # delta always negative
            trigger=triggered.set,
            poll_interval=0.05,
        )
        watchdog.start()
        watchdog.stop()
        watchdog.join(timeout=2)
        time.sleep(0.1)
        assert not triggered.is_set()

    def test_stops_cleanly(self) -> None:
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=9999.0,
            start_cpu=_get_total_cpu(),
            trigger=lambda: None,
            poll_interval=0.05,
        )
        watchdog.start()
        watchdog.stop()
        watchdog.join(timeout=2)
        assert not watchdog.is_alive()

    def test_is_daemon_thread(self) -> None:
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=1.0,
            start_cpu=0.0,
            trigger=lambda: None,
        )
        assert watchdog.daemon is True

    def test_with_negative_max_cpu(self) -> None:
        """Negative budget triggers immediately since elapsed > -1 always."""
        triggered: threading.Event = threading.Event()
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=-1.0,
            start_cpu=_get_total_cpu(),
            trigger=triggered.set,
            poll_interval=0.05,
        )
        watchdog.start()
        _burn_cpu(0.01)
        assert triggered.wait(timeout=3)
        watchdog.stop()
        watchdog.join(timeout=2)

    def test_custom_poll_interval(self) -> None:
        """Fast poll_interval still detects budget breach."""
        triggered: threading.Event = threading.Event()
        watchdog: CPUTimeWatchdog = CPUTimeWatchdog(
            max_cpu=0.0,
            start_cpu=_get_total_cpu(),
            trigger=triggered.set,
            poll_interval=0.01,
        )
        watchdog.start()
        _burn_cpu(0.01)
        assert triggered.wait(timeout=2)
        watchdog.stop()
        watchdog.join(timeout=2)


# ===================================================================
# _get_cpu_timeout
# ===================================================================


class TestGetCpuTimeout:
    def test_returns_none_when_no_config(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=None, cli_value=None)
        assert _get_cpu_timeout(item) is None

    def test_marker_takes_precedence_over_cli(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=60.0, cli_value=120.0)
        assert _get_cpu_timeout(item) == 60.0

    def test_falls_back_to_cli(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=None, cli_value=120.0)
        assert _get_cpu_timeout(item) == 120.0

    def test_zero_marker_is_not_none(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=0, cli_value=None)
        result: Optional[float] = _get_cpu_timeout(item)
        assert result == 0.0
        assert result is not None

    def test_float_marker_value(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=45.5, cli_value=None)
        assert _get_cpu_timeout(item) == 45.5

    def test_integer_marker_coerced_to_float(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=120, cli_value=None)
        result: Optional[float] = _get_cpu_timeout(item)
        assert result is not None, "integer marker value must resolve to a timeout"
        assert isinstance(result, float)
        assert result == 120.0


# ===================================================================
# prepare_timeout_watchdog hook
# ===================================================================


class TestSetTimerHook:
    """Unit tests for the prepare_timeout_watchdog hook."""

    @staticmethod
    def _make_settings(
        timeout: float = 900.0,
        method: str = "signal",
    ) -> Settings:
        return Settings(
            timeout=timeout,
            method=method,
            func_only=True,
            disable_debugger_detection=False,
        )

    def test_returns_none_when_no_cpu_limit(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=None, cli_value=None)
        settings: Settings = self._make_settings()
        result: Optional[bool] = prepare_timeout_watchdog(item, settings)
        assert result is None

    def test_returns_true_when_cpu_limit_set(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=120.0)
        settings: Settings = self._make_settings()
        item.cancel_timeout = MagicMock()
        with patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"):
            result: Optional[bool] = prepare_timeout_watchdog(item, settings)
        assert result is True
        item.cancel_timeout()

    def test_delegates_to_pytest_timeout_set_timer(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=120.0)
        settings: Settings = self._make_settings()
        item.cancel_timeout = MagicMock()
        with patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer") as mock_pt:
            prepare_timeout_watchdog(item, settings)
        mock_pt.assert_called_once_with(item, settings)
        item.cancel_timeout()

    def test_cancel_stops_watchdog_and_calls_original(self) -> None:
        item: MagicMock = _make_mock_item(marker_value=9999.0)
        settings: Settings = self._make_settings()
        original_cancel: MagicMock = MagicMock()
        item.cancel_timeout = original_cancel
        with patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"):
            prepare_timeout_watchdog(item, settings)
        wrapped_cancel = item.cancel_timeout
        wrapped_cancel()
        original_cancel.assert_called_once()

    def test_handler_double_fire_prevention(self) -> None:
        """SIGALRM handler fires once; second call is a no-op."""
        item: MagicMock = _make_mock_item(marker_value=120.0)
        settings: Settings = self._make_settings()
        item.cancel_timeout = MagicMock()
        call_count: int = 0

        def counting_timeout_sigalrm(item: Any, settings: Any) -> None:
            nonlocal call_count
            call_count += 1

        with (
            patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"),
            patch(
                "test.utils.cpu_timeout.pytest_timeout.timeout_sigalrm",
                side_effect=counting_timeout_sigalrm,
            ),
        ):
            prepare_timeout_watchdog(item, settings)
            handler = _installed_sigalrm_handler()
            handler(signal.SIGALRM, None)
            handler(signal.SIGALRM, None)

        assert call_count == 1
        item.cancel_timeout()

    def test_handler_routes_cpu_breach_to_cpu_settings(self) -> None:
        """When CPU exceeds limit, timeout_sigalrm gets cpu_settings."""
        item: MagicMock = _make_mock_item(marker_value=0.001)
        settings: Settings = self._make_settings(timeout=900.0)
        item.cancel_timeout = MagicMock()
        captured: list[Settings] = []

        with (
            patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"),
            patch(
                "test.utils.cpu_timeout.pytest_timeout.timeout_sigalrm",
                side_effect=lambda _i, s: captured.append(s),
            ),
            # start_cpu=0.0, then handler reads 999.0 -> elapsed=999 > 0.001
            patch("test.utils.cpu_timeout._get_total_cpu", side_effect=[0.0, 999.0]),
        ):
            prepare_timeout_watchdog(item, settings)
            handler = _installed_sigalrm_handler()
            handler(signal.SIGALRM, None)

        assert len(captured) == 1
        assert captured[0].timeout == 0.001
        item.cancel_timeout()

    def test_handler_routes_wall_breach_to_original_settings(self) -> None:
        """When CPU is under limit (wall-clock fires), original settings used."""
        item: MagicMock = _make_mock_item(marker_value=9999.0)
        settings: Settings = self._make_settings(timeout=900.0)
        item.cancel_timeout = MagicMock()
        captured: list[Settings] = []

        with (
            patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"),
            patch(
                "test.utils.cpu_timeout.pytest_timeout.timeout_sigalrm",
                side_effect=lambda _i, s: captured.append(s),
            ),
            # start_cpu=100.0, handler reads 100.0 -> elapsed=0 < 9999
            patch("test.utils.cpu_timeout._get_total_cpu", side_effect=[100.0, 100.0]),
        ):
            prepare_timeout_watchdog(item, settings)
            handler = _installed_sigalrm_handler()
            handler(signal.SIGALRM, None)

        assert len(captured) == 1
        assert captured[0].timeout == 900.0
        item.cancel_timeout()

    def test_thread_method_fallback(self) -> None:
        """Non-main thread -> trigger uses timeout_timer, not os.kill."""
        item: MagicMock = _make_mock_item(marker_value=9999.0)
        settings: Settings = self._make_settings(method="signal")
        item.cancel_timeout = MagicMock()
        with (
            patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"),
            patch(
                "test.utils.cpu_timeout.threading.current_thread",
                return_value=MagicMock(),
            ),
            patch(
                "test.utils.cpu_timeout.threading.main_thread",
                return_value=MagicMock(),
            ),
        ):
            prepare_timeout_watchdog(item, settings)
        item.cancel_timeout()

    def test_watchdog_joins_on_cancel(self) -> None:
        """cancel_timeout waits for watchdog thread to finish."""
        item: MagicMock = _make_mock_item(marker_value=9999.0)
        settings: Settings = self._make_settings()
        item.cancel_timeout = MagicMock()
        with patch("test.utils.cpu_timeout.pytest_timeout.pytest_timeout_set_timer"):
            prepare_timeout_watchdog(item, settings)
        cancel_fn = item.cancel_timeout
        threads_before: int = threading.active_count()
        cancel_fn()
        time.sleep(0.1)
        threads_after: int = threading.active_count()
        assert threads_after <= threads_before
