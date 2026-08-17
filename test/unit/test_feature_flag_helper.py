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
import os
from unittest.mock import MagicMock

from test.utils.feature_flag_helper import (
    NEURON_NEV_PREFIX,
    derive_pytest_test_id,
    env_flag_enabled,
    get_feature_flag,
    resolve_ssh_config_path,
)


class TestEnvFlagEnabled:
    """Boolean env parsing: only recognized truthy strings enable; "0"/"false"/unset do not
    (so setting a flag to "0" to disable it doesn't silently enable via bare truthiness)."""

    _KEY = "NEURON_TEST_FLAG_XYZ"

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv(self._KEY, raising=False)
        assert env_flag_enabled(self._KEY) is False

    def test_truthy_values_enable(self, monkeypatch):
        for val in ("1", "true", "TRUE", "yes", "on", " 1 "):
            monkeypatch.setenv(self._KEY, val)
            assert env_flag_enabled(self._KEY) is True, val

    def test_falsy_values_do_not_enable(self, monkeypatch):
        for val in ("0", "false", "no", "off", "", "  "):
            monkeypatch.setenv(self._KEY, val)
            assert env_flag_enabled(self._KEY) is False, val


class TestGetFeatureFlag:
    def test_returns_config_value_when_set(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = "config_value"

        result = get_feature_flag(mock_config, "test-key")

        assert result == "config_value"

    def test_returns_env_value_when_config_is_none(self, monkeypatch):
        mock_config = MagicMock()
        mock_config.getoption.return_value = None
        monkeypatch.setenv(f"{NEURON_NEV_PREFIX}TEST_KEY", "env_value")

        result = get_feature_flag(mock_config, "test-key")

        assert result == "env_value"

    def test_returns_default_when_no_config_or_env(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = None

        result = get_feature_flag(mock_config, "test-key", default_value="default")

        assert result == "default"

    def test_config_takes_precedence_over_env(self, monkeypatch):
        mock_config = MagicMock()
        mock_config.getoption.return_value = "config_value"
        monkeypatch.setenv(f"{NEURON_NEV_PREFIX}TEST_KEY", "env_value")

        result = get_feature_flag(mock_config, "test-key")

        assert result == "config_value"

    def test_resolves_env_reference_in_config_value(self, monkeypatch):
        mock_config = MagicMock()
        mock_config.getoption.return_value = "$MY_VAR"
        monkeypatch.setenv("MY_VAR", "resolved_value")

        result = get_feature_flag(mock_config, "test-key")

        assert result == "resolved_value"

    def test_converts_dashes_to_underscores_in_env_key(self, monkeypatch):
        mock_config = MagicMock()
        mock_config.getoption.return_value = None
        monkeypatch.setenv(f"{NEURON_NEV_PREFIX}MY_TEST_KEY", "env_value")

        result = get_feature_flag(mock_config, "my-test-key")

        assert result == "env_value"

    def test_zero_config_value_is_returned_not_treated_as_falsy(self):
        mock_config = MagicMock()
        mock_config.getoption.return_value = 0

        result = get_feature_flag(mock_config, "test-key", default_value="default")

        assert result == 0


class TestDerivePytestTestId:
    def test_sanitizes_parentheses_and_commas(self, monkeypatch):
        """Parentheses and commas cause issues in shell commands when passed through SSH."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_module.py::test_sweep[(3648,128)-(1024,512)] (call)")

        result = derive_pytest_test_id()

        assert result == "test_sweep_3648_128_1024_512"


class TestResolveSshConfigPath:
    """resolve_ssh_config_path is the single source of truth shared by make_host_manager
    (worker connections) and the fleet reachability probe — so they can't drift apart."""

    def test_defaults_to_user_ssh_config(self, monkeypatch):
        monkeypatch.delenv(f"{NEURON_NEV_PREFIX}SSH_CONFIG_PATH", raising=False)
        config = MagicMock()
        config.getoption.return_value = None  # no CLI flag
        assert resolve_ssh_config_path(config) == os.path.expanduser("~/.ssh/config")

    def test_uses_and_expands_configured_path(self):
        config = MagicMock()
        config.getoption.return_value = "~/custom/ssh_config"
        assert resolve_ssh_config_path(config) == os.path.expanduser("~/custom/ssh_config")
