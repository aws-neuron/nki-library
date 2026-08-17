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

from types import SimpleNamespace
from unittest.mock import patch

from _pytest.stash import Stash

from test.utils.common_dataclasses import HostProvisioningResult, TraceMode
from test.utils.pytest_plugin import apply_host_provisioning_result, get_pytest_mark_names, resolve_session_trace_mode


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


def test_get_pytest_mark_names_returns_sorted_unique_names():
    node = SimpleNamespace(
        iter_markers=lambda: iter(
            [
                SimpleNamespace(name="slow", args=(), kwargs={}),
                SimpleNamespace(name="platforms", args=(), kwargs={"exclude": ["trn1"]}),
                SimpleNamespace(name="fast", args=(), kwargs={}),
                SimpleNamespace(name="slow", args=("duplicate",), kwargs={}),
            ]
        )
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
