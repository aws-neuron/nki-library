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
"""Unit tests for pytest_plugin session trace-mode resolution.

Regression coverage for a trace-mode reporting mismatch. resolve_session_trace_mode is the
single source of truth for both the executed trace mode (session_trace_mode fixture) and the
one reported to SQS/dashboards (SessionContext), so any remote host pool must resolve to
CompileAndInfer or hardware runs get mislabeled compile_only and undercounted:
  * ``--target-host`` (static host list) and ``--target-host-file`` (``--shared-fleet``'s
    static-file form), and
  * a plugin-provisioned fleet, which carries NO CLI flag -- it is signalled only by the
    HostProvisioningResult the provisioning plug-in returns to the controller (stashed on
    config; workers rebuild it from workerinput). This is the production pipeline path, so a
    drift here mislabels real pipeline hardware runs.
resolve_session_trace_mode projects remote-vs-local off resolve_host_provisioning_mode, which
reads that result from config, so these tests set it there (see _config).
"""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from _pytest.stash import Stash

from test.utils.common_dataclasses import HostProvisioningResult, TraceMode
from test.utils.pytest_plugin import (
    KEEP_FAST_PROPERTY,
    apply_host_provisioning_result,
    get_pytest_mark_names,
    record_keep_fast,
    resolve_session_trace_mode,
)
from test.utils.simulation_setup import apply_xfail_simulation
from test.utils.simulation_xfails import SIMULATION_XFAIL_NODEIDS, SIMULATION_XFAIL_REASON


def _flags(**overrides):
    """Return a get_feature_flag side effect that resolves flag values by key."""

    def _side_effect(config, key, *args, **kwargs):
        return overrides.get(key)

    return _side_effect


def _config(*, plugin_provisioned=False):
    """A config with a real (empty) config.stash — resolve_session_trace_mode /
    resolve_host_provisioning_mode read and cache there — and a provisioning plug-in's result
    decomposed onto it as the controller does (apply_host_provisioning_result). Non-plug-in
    leaves the mode to lazy CLI derivation; plugin_provisioned=True pins it to
    PLUGIN_PROVISIONED (the pipeline/fleet path)."""
    config = SimpleNamespace(stash=Stash())
    apply_host_provisioning_result(config, HostProvisioningResult(plugin_provisioned=plugin_provisioned))
    return config


class _MarkerNode:
    """Stands in for a collected test: reports a preset list of resolved markers."""

    def __init__(self, markers: list[pytest.Mark]) -> None:
        self._markers = markers

    def iter_markers(self, name: str | None = None) -> Iterator[pytest.Mark]:
        return (marker for marker in self._markers if name is None or marker.name == name)


class _CollectedItem:
    """Stands in for a collected test: resolves and records markers like a pytest Item."""

    def __init__(self, markers: list[pytest.Mark], nodeid: str = "") -> None:
        self._markers = list(markers)
        self.nodeid = nodeid
        self.added: list[pytest.Mark] = []
        self.user_properties: list[tuple[str, object]] = []

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        for marker in reversed(self._markers):
            if marker.name == name:
                return marker
        return None

    def add_marker(self, marker) -> None:
        mark = getattr(marker, "mark", marker)
        self._markers.append(mark)
        self.added.append(mark)


class TestApplyXfailSimulation:
    """apply_xfail_simulation translates the xfail_simulation marker into a
    non-strict xfail, and only for tagged tests (it is called in simulation mode only)."""

    def test_marks_tagged_item_with_its_reason(self):
        item = _CollectedItem([pytest.mark.xfail_simulation(reason="sim numerical gap").mark])
        apply_xfail_simulation([item])
        xfails = [m for m in item.added if m.name == "xfail"]
        assert len(xfails) == 1
        assert xfails[0].kwargs["reason"] == "sim numerical gap"
        assert xfails[0].kwargs["strict"] is False

    def test_defaults_reason_when_marker_has_none(self):
        item = _CollectedItem([pytest.mark.xfail_simulation.mark])
        apply_xfail_simulation([item])
        xfails = [m for m in item.added if m.name == "xfail"]
        assert len(xfails) == 1 and xfails[0].kwargs["reason"]

    def test_ignores_untagged_item(self):
        item = _CollectedItem([pytest.mark.fast.mark], nodeid="test/x.py::test_ok")
        apply_xfail_simulation([item])
        assert item.added == []

    def test_marks_item_listed_in_nodeid_registry(self):
        listed = next(iter(SIMULATION_XFAIL_NODEIDS))
        item = _CollectedItem([], nodeid=listed)
        apply_xfail_simulation([item])
        xfails = [m for m in item.added if m.name == "xfail"]
        assert len(xfails) == 1
        assert xfails[0].kwargs["reason"] == SIMULATION_XFAIL_REASON
        assert xfails[0].kwargs["strict"] is False

    def test_inline_marker_takes_precedence_over_registry(self):
        listed = next(iter(SIMULATION_XFAIL_NODEIDS))
        item = _CollectedItem([pytest.mark.xfail_simulation(reason="inline reason").mark], nodeid=listed)
        apply_xfail_simulation([item])
        xfails = [m for m in item.added if m.name == "xfail"]
        assert len(xfails) == 1 and xfails[0].kwargs["reason"] == "inline reason"


class TestRecordKeepFast:
    """record_keep_fast carries a keep_fast test's reason to the junit report and
    rejects a keep_fast marker without one."""

    def test_records_reason_as_property(self):
        item = _CollectedItem([pytest.mark.fast.mark, pytest.mark.keep_fast(reason="production config").mark])
        record_keep_fast([item])
        assert item.user_properties == [(KEEP_FAST_PROPERTY, "production config")]

    def test_ignores_unmarked_item(self):
        item = _CollectedItem([pytest.mark.fast.mark], nodeid="test/x.py::test_ok")
        record_keep_fast([item])
        assert item.user_properties == []

    @pytest.mark.parametrize("mark", [pytest.mark.keep_fast, pytest.mark.keep_fast(reason="")])
    def test_rejects_marker_without_reason(self, mark):
        item = _CollectedItem([getattr(mark, "mark", mark)], nodeid="test/x.py::test_kept")
        with pytest.raises(pytest.UsageError, match="test/x.py::test_kept: keep_fast needs a reason"):
            record_keep_fast([item])


def test_get_pytest_mark_names_returns_sorted_unique_names():
    node = _MarkerNode(
        [
            pytest.mark.slow.mark,
            pytest.mark.platforms(exclude=["trn1"]).mark,
            pytest.mark.fast.mark,
            pytest.mark.slow("duplicate").mark,
        ]
    )

    assert get_pytest_mark_names(node) == ["fast", "platforms", "slow"]


class TestResolveSessionTraceMode:
    """resolve_session_trace_mode is the single source of truth for both the
    execution path (session_trace_mode fixture) and session-level reporting."""

    def test_target_host_file_implies_compile_and_infer(self):
        """--shared-fleet sets target_host_file; with no --test-mode it must be
        CompileAndInfer, else the reported trace mode is mislabeled compile_only."""
        with patch(
            "test.utils.pytest_plugin.get_feature_flag",
            side_effect=_flags(target_host_file="/tmp/shared_fleet.json"),
        ):
            assert resolve_session_trace_mode(_config()) == TraceMode.CompileAndInfer

    def test_target_host_implies_compile_and_infer(self):
        """A single --target-host run also resolves to CompileAndInfer."""
        with patch(
            "test.utils.pytest_plugin.get_feature_flag",
            side_effect=_flags(target_host="host-1"),
        ):
            assert resolve_session_trace_mode(_config()) == TraceMode.CompileAndInfer

    def test_plugin_provisioned_implies_compile_and_infer(self):
        """A plugin-provisioned fleet carries no CLI flag -- only the provisioning result. With
        no --test-mode it must still resolve to CompileAndInfer (this is the pipeline path), else
        real pipeline hardware runs are reported as compile_only. Regression guard: the earlier
        fixture-only handling of this case drifted from resolve_session_trace_mode, which
        reported compile_only for exactly this run."""
        with patch("test.utils.pytest_plugin.get_feature_flag", side_effect=_flags()):
            assert resolve_session_trace_mode(_config(plugin_provisioned=True)) == TraceMode.CompileAndInfer

    def test_explicit_test_mode_takes_precedence(self):
        """An explicit --test-mode wins over the host flags."""
        with patch(
            "test.utils.pytest_plugin.get_feature_flag",
            side_effect=_flags(test_mode="compile-only", target_host_file="/tmp/f.json"),
        ):
            assert resolve_session_trace_mode(_config()) == TraceMode.create("compile-only")

    def test_explicit_test_mode_takes_precedence_over_plugin_provisioned(self):
        """--test-mode wins even over the plugin-provisioned result."""
        with patch(
            "test.utils.pytest_plugin.get_feature_flag",
            side_effect=_flags(test_mode="simulation"),
        ):
            assert resolve_session_trace_mode(_config(plugin_provisioned=True)) == TraceMode.create("simulation")

    def test_defaults_to_compile_only_without_hosts_or_devices(self):
        """No host flags, no plugin signal, and no local Neuron devices -> CompileOnly."""
        with (
            patch("test.utils.pytest_plugin.get_feature_flag", side_effect=_flags()),
            patch("test.utils.pytest_plugin.detect_local_neuron_devices", return_value=False),
        ):
            assert resolve_session_trace_mode(_config()) == TraceMode.CompileOnly

    def test_single_compile_separation_not_requested_by_default(self):
        """With no single compile separation request, trace mode resolution is unaffected."""
        with (
            patch("test.utils.pytest_plugin.get_feature_flag", side_effect=_flags()),
            patch("test.utils.pytest_plugin._single_compile_separation_requested", return_value=False),
            patch("test.utils.pytest_plugin.detect_local_neuron_devices", return_value=False),
        ):
            assert resolve_session_trace_mode(_config()) == TraceMode.CompileOnly
