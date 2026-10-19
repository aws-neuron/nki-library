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

"""Host-side tests for the pure-Python logging helpers.

These tests exercise the level-gating, prefix formatting, environment-variable
parsing, and get_logger priority-order branches. All logic is pure Python, so
the tests import the functions directly and run under the simulator without any
device kernel or marker.
"""

import nkilib_src.nkilib.core.utils.logging as logging_mod
import pytest
from nkilib_src.nkilib.core.utils.logging import (
    Logger,
    LogLevel,
    _init_from_env_py,
    get_logger,
)


@pytest.mark.fast
def test_from_string_valid_levels():
    assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
    assert LogLevel.from_string("INFO") == LogLevel.INFO
    assert LogLevel.from_string("WARN") == LogLevel.WARN
    assert LogLevel.from_string("ERROR") == LogLevel.ERROR
    assert LogLevel.from_string("OFF") == LogLevel.OFF


@pytest.mark.fast
def test_from_string_invalid_raises_key_error():
    with pytest.raises(KeyError):
        LogLevel.from_string("NOPE")


@pytest.mark.fast
def test_should_log_gating_at_each_level():
    # DEBUG logger: everything at/above DEBUG is enabled.
    debug_logger = Logger("x", LogLevel.DEBUG)
    assert debug_logger._should_log(LogLevel.DEBUG)
    assert debug_logger._should_log(LogLevel.INFO)
    assert debug_logger._should_log(LogLevel.WARN)
    assert debug_logger._should_log(LogLevel.ERROR)
    # is_enabled_for delegates to _should_log.
    assert debug_logger.is_enabled_for(LogLevel.DEBUG)

    # ERROR logger: only ERROR is enabled.
    error_logger = Logger("x", LogLevel.ERROR)
    assert not error_logger._should_log(LogLevel.DEBUG)
    assert not error_logger._should_log(LogLevel.INFO)
    assert not error_logger._should_log(LogLevel.WARN)
    assert error_logger._should_log(LogLevel.ERROR)

    # OFF logger: nothing is enabled.
    off_logger = Logger("x", LogLevel.OFF)
    assert not off_logger._should_log(LogLevel.ERROR)


@pytest.mark.fast
def test_log_methods_emit_when_enabled(capsys):
    """DEBUG-level logger takes the true arc of every level method (calls _print)."""
    logger = Logger("SBM", LogLevel.DEBUG)
    logger.debug("d")
    logger.info("i")
    logger.warn("w")
    logger.error("e")
    out = capsys.readouterr().out
    assert "[DEBUG] [SBM] d" in out
    assert "[INFO] [SBM] i" in out
    assert "[WARN] [SBM] w" in out
    assert "[ERROR] [SBM] e" in out


@pytest.mark.fast
def test_log_methods_suppressed_when_disabled(capsys):
    """OFF-level logger takes the false (fall-through) arc of every level method."""
    logger = Logger("SBM", LogLevel.OFF)
    logger.debug("d")
    logger.info("i")
    logger.warn("w")
    logger.error("e")
    out = capsys.readouterr().out
    assert out == ""


@pytest.mark.fast
def test_print_with_and_without_name(capsys):
    """Cover both arcs of the name_prefix ternary in _print."""
    named = Logger("SBM", LogLevel.DEBUG)
    named._print("hello", prefix="[INFO] ")
    unnamed = Logger("", LogLevel.DEBUG)
    unnamed._print("hello", prefix="[INFO] ")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[INFO] [SBM] hello"
    assert lines[1] == "[INFO] hello"


@pytest.mark.fast
def test_init_from_env_global_and_specific(monkeypatch):
    """Cover the global (L137) and per-logger (L144) true arcs of _init_from_env_py."""
    monkeypatch.setattr(
        logging_mod.os,
        "environ",
        {
            "NKILIB_LOG_LEVEL": "warn",  # global override, lowercase -> upper()
            "NKILIB_LOG_LEVEL_SBM": "debug",  # per-logger override
            "UNRELATED_VAR": "ignored",  # neither arc
        },
    )
    global_level, specific = _init_from_env_py()
    assert global_level == LogLevel.WARN
    assert specific == {"SBM": LogLevel.DEBUG}


@pytest.mark.fast
def test_init_from_env_invalid_values_warn(monkeypatch, capsys):
    """Invalid values hit the KeyError except branches for both global and per-logger."""
    monkeypatch.setattr(
        logging_mod.os,
        "environ",
        {
            "NKILIB_LOG_LEVEL": "bogus",
            "NKILIB_LOG_LEVEL_SBM": "alsobogus",
        },
    )
    global_level, specific = _init_from_env_py()
    assert global_level is None
    assert specific == {}
    err = capsys.readouterr().err
    assert "Invalid NKILIB_LOG_LEVEL='bogus'" in err
    assert "Invalid log level 'alsobogus' for SBM" in err


@pytest.mark.fast
def test_get_logger_specific_env_override(monkeypatch):
    """Per-logger env override wins (L180 true arc)."""
    monkeypatch.setattr(logging_mod, "_ENV_LOG_LEVELS", {"SBM": LogLevel.DEBUG})
    monkeypatch.setattr(logging_mod, "_ENV_GLOBAL_LOG_LEVEL", LogLevel.ERROR)
    logger = get_logger("SBM", level=LogLevel.WARN)
    assert logger.level == LogLevel.DEBUG


@pytest.mark.fast
def test_get_logger_global_env_override(monkeypatch):
    """Global env override wins when no per-logger entry (L180 false, L183 true)."""
    monkeypatch.setattr(logging_mod, "_ENV_LOG_LEVELS", {})
    monkeypatch.setattr(logging_mod, "_ENV_GLOBAL_LOG_LEVEL", LogLevel.ERROR)
    logger = get_logger("qkv_tkg", level=LogLevel.WARN)
    assert logger.level == LogLevel.ERROR


@pytest.mark.fast
def test_get_logger_falls_back_to_code_level(monkeypatch):
    """No env overrides -> code-level default (L180 false, L183 false)."""
    monkeypatch.setattr(logging_mod, "_ENV_LOG_LEVELS", {})
    monkeypatch.setattr(logging_mod, "_ENV_GLOBAL_LOG_LEVEL", None)
    logger = get_logger("qkv_tkg", level=LogLevel.DEBUG)
    assert logger.level == LogLevel.DEBUG
