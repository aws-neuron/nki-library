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
"""Unit tests for the central run-mode classification: resolve_host_provisioning_mode and
the boolean projections (is_local_run / is_remote_bootstrap_run / is_recoverable_run) that
everything else derives from.

Classification inputs are the public CLI signals plus the HostProvisioningResult an internal
provisioning plug-in returns to the controller (plugin_provisioned folds into the mode as
PLUGIN_PROVISIONED; recoverable is read by is_recoverable_run) — a plug-in provisions a
remote pool without any public flag. These tests drive both the flags and the result object
(the controller stashes it on config; here we set it directly).
"""

from unittest.mock import MagicMock

import pytest
from _pytest.stash import Stash

from test.utils.common_dataclasses import HostProvisioningMode, HostProvisioningResult
from test.utils.pytest_plugin import (
    HOST_PROVISIONING_MODE_KEY,
    apply_host_provisioning_result,
    is_local_run,
    is_recoverable_run,
    is_remote_bootstrap_run,
    make_host_manager,
    resolve_host_provisioning_mode,
)


def _config(*, plugin_provisioned=False, recoverable=False, **flags):
    """A fake pytest Config whose getoption returns the given run-mode flags (others None),
    with a real (empty) config.stash — the run-mode helpers read/cache their state there — and
    a provisioning plug-in's result decomposed onto it exactly as the controller does (via
    apply_host_provisioning_result). A fresh object per call keeps stash state from leaking."""
    cfg = MagicMock()
    cfg.getoption.side_effect = lambda key, *a, **k: flags.get(key)
    cfg.stash = Stash()
    apply_host_provisioning_result(
        cfg, HostProvisioningResult(plugin_provisioned=plugin_provisioned, recoverable=recoverable)
    )
    return cfg


class TestResolveHostProvisioningMode:
    """Classification from the public CLI signals plus the plug-in's provisioning result."""

    def test_local_when_no_signals(self):
        assert resolve_host_provisioning_mode(_config()) is HostProvisioningMode.LOCAL

    def test_static_hosts_from_target_host(self):
        cfg = _config(target_host=["1.2.3.4"])
        assert resolve_host_provisioning_mode(cfg) is HostProvisioningMode.STATIC_HOSTS

    def test_static_file_from_target_host_file(self):
        cfg = _config(target_host_file="/path/hosts.json")
        assert resolve_host_provisioning_mode(cfg) is HostProvisioningMode.STATIC_FILE

    def test_plugin_provisioned_from_result(self):
        # A provisioning plug-in carries no public flag; its returned result
        # (plugin_provisioned=True) classifies the run as PLUGIN_PROVISIONED.
        assert resolve_host_provisioning_mode(_config(plugin_provisioned=True)) is (
            HostProvisioningMode.PLUGIN_PROVISIONED
        )

    def test_precedence_file_over_hosts(self):
        # file beats host list
        cfg = _config(target_host_file="/f.json", target_host=["1.2.3.4"])
        assert resolve_host_provisioning_mode(cfg) is HostProvisioningMode.STATIC_FILE

    def test_precedence_plugin_remote_over_static_flags(self):
        # A plug-in-provisioned pool wins over any static CLI flag (controller layering:
        # the plug-in claims the run before static hosts are considered).
        cfg = _config(plugin_provisioned=True, target_host_file="/f.json", target_host=["1.2.3.4"])
        assert resolve_host_provisioning_mode(cfg) is HostProvisioningMode.PLUGIN_PROVISIONED

    def test_result_is_cached_on_config(self):
        cfg = _config(target_host_file="/f.json")
        first = resolve_host_provisioning_mode(cfg)
        assert cfg.stash[HOST_PROVISIONING_MODE_KEY] is first
        # getoption is consulted on the first resolve; the cache serves subsequent calls.
        calls_after_first = cfg.getoption.call_count
        resolve_host_provisioning_mode(cfg)
        assert cfg.getoption.call_count == calls_after_first


class TestProjections:
    """The booleans are pure projections of the mode — classification lives entirely in
    resolve_host_provisioning_mode, including the plug-in's provisioning result."""

    def test_is_local_run(self):
        assert is_local_run(_config()) is True
        assert is_local_run(_config(target_host=["1.2.3.4"])) is False
        assert is_local_run(_config(target_host_file="/f.json")) is False

    def test_plugin_provisioned_is_not_local_and_is_bootstrap(self):
        # PLUGIN_PROVISIONED is neither local nor a bare-flag run: it is a controller
        # bootstrap run.
        assert is_local_run(_config(plugin_provisioned=True)) is False
        assert is_remote_bootstrap_run(_config(plugin_provisioned=True)) is True

    def test_is_remote_bootstrap_run_excludes_local_and_target_host(self):
        # The deliberate distinction: --target-host is NOT a bootstrap run.
        assert is_remote_bootstrap_run(_config()) is False
        assert is_remote_bootstrap_run(_config(target_host=["1.2.3.4"])) is False
        assert is_remote_bootstrap_run(_config(target_host_file="/f.json")) is True


class TestIsRecoverableRun:
    """Recoverable requires BOTH the result's recoverable flag AND a plug-in-provisioned
    pool — only that pool runs the background re-resolver a wait depends on."""

    def test_recoverable_when_result_recoverable_and_plugin_provisioned(self):
        # The pipeline path: the plug-in returns both flags together.
        assert is_recoverable_run(_config(plugin_provisioned=True, recoverable=True)) is True

    def test_not_recoverable_without_signal(self):
        assert is_recoverable_run(_config()) is False
        assert is_recoverable_run(_config(target_host_file="/f.json")) is False

    def test_remote_but_not_recoverable(self):
        # A remote pool with no background re-resolver (one-shot static bootstrap) is not recoverable.
        assert is_recoverable_run(_config(plugin_provisioned=True)) is False

    def test_recoverable_signal_ignored_without_plugin_provisioned(self):
        # SMELL #1 guard: a stray recoverable flag on a local or static run must NOT make a
        # failed claim wait for a re-resolver that isn't running — those fail fast. (Can't
        # arise in production: only the plug-in sets recoverable, and only when it provisions.)
        assert is_recoverable_run(_config(recoverable=True)) is False  # LOCAL
        assert is_recoverable_run(_config(recoverable=True, target_host=["1.2.3.4"])) is False  # STATIC_HOSTS
        assert is_recoverable_run(_config(recoverable=True, target_host_file="/f.json")) is False  # STATIC_FILE

    def test_make_host_manager_threads_recoverable_from_result(self, tmp_path):
        assert (
            make_host_manager(
                _config(plugin_provisioned=True, recoverable=True, output_directory=str(tmp_path))
            ).hosts_recoverable
            is True
        )
        assert (
            make_host_manager(
                _config(plugin_provisioned=True, recoverable=False, output_directory=str(tmp_path))
            ).hosts_recoverable
            is False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
