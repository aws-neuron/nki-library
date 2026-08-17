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
import hashlib
import os
import re
from typing import Any, Optional

import pytest
from _pytest.config import Config

from .common_dataclasses import Platforms

NEURON_NEV_PREFIX = "NKILIB_"


# Strings that count as "true" for a boolean env var. Matches the idiom used elsewhere
# (e.g. NKILIB_USE_TORCH_REF), so "0"/"false"/unset all read as false — a var set to "0"
# to *disable* a behavior does not silently enable it via bare truthiness.
_TRUE_ENV_VALUES = ("1", "true", "yes", "on")


def env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is set to a recognized truthy value (see _TRUE_ENV_VALUES).
    Unset, empty, "0", and "false" are all false."""
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


# Maximum length for directory names (Linux filesystem limit is 255)
# We use 200 to leave room for the "out-" prefix and hash suffix
MAX_DIR_NAME_LENGTH = 200


def maybe_resolve_env_value(maybe_env_key: Any) -> str:
    if isinstance(maybe_env_key, str) and maybe_env_key.startswith("$"):
        return os.environ.get(maybe_env_key.replace("$", ""), maybe_env_key)
    else:
        return maybe_env_key


def split_tokens(raw: Optional[str]) -> list[str]:
    """Split a comma-separated string into stripped, non-empty tokens."""
    return [t for t in (s.strip() for s in (raw or "").split(",")) if t]


def get_feature_flag(
    pytest_config: Config,
    config_key: str,
    default_value: Optional[Any] = None,
) -> Any:
    # TODO: get_feature_flag relies on falsy pytest CLI default for fallback to environment variable/default_value
    config_value: Optional[Any] = pytest_config.getoption(config_key)
    constructed_env_key: str = f"{NEURON_NEV_PREFIX}{config_key.upper().replace('-', '_')}"
    env_value: Optional[Any] = os.environ.get(constructed_env_key)

    if config_value or isinstance(config_value, int):
        return maybe_resolve_env_value(config_value)
    elif env_value:
        result = maybe_resolve_env_value(env_value)
        if isinstance(config_value, list):
            result = result.split(" ")
        return result
    else:
        return default_value


def resolve_base_output_directory(config: Config) -> str:
    output_directory_option: str = get_feature_flag(config, "output_directory")

    absolute_directory_path = output_directory_option
    if not os.path.isabs(output_directory_option):
        absolute_directory_path = os.path.abspath(os.path.join(os.getcwd(), output_directory_option))

    return absolute_directory_path


def resolve_ssh_config_path(config: Config) -> str:
    """The ssh-config path for this run (``--ssh-config-path``, default ``~/.ssh/config``),
    user-expanded."""
    return os.path.expanduser(get_feature_flag(config, "ssh_config_path", "~/.ssh/config"))


def get_platform_targets(config: Config) -> list[Platforms]:
    """Return the list of target platforms from CLI (or default to [TRN2])."""
    valid_names = [p.value for p in Platforms]
    raw = get_feature_flag(config, "platform_target", None)
    if raw is None:
        return [Platforms.TRN2]
    platforms: list[Platforms] = []
    for name in split_tokens(raw):
        try:
            platforms.append(Platforms(name))
        except ValueError as e:
            raise pytest.UsageError(f"Unknown platform target '{name}'. Valid options: {', '.join(valid_names)}") from e
    if not platforms:
        raise pytest.UsageError(f"--platform-target resolved to an empty list. Valid options: {', '.join(valid_names)}")
    return platforms


def derive_pytest_test_id() -> str:
    # rsplit to handle spaces in test parameters like (1260, 128, 4, 4)
    # Only split once from right to remove " (call)" or " (teardown)" suffix
    test_id: str = os.environ["PYTEST_CURRENT_TEST"].split("::")[-1].rsplit(" ", 1)[0]
    # Sanitize characters that cause issues in shell commands and file paths
    # - Square brackets from pytest parametrize: [param1-param2]
    # - Parentheses, commas, spaces, hyphens, quotes, braces, colons from parameters
    test_id = re.sub(r'[\[\](), \-\'\"{}:]', '_', test_id)
    test_id = re.sub(r'_+', '_', test_id)  # Collapse multiple underscores
    test_id = test_id.strip('_')  # Remove leading/trailing underscores
    return test_id


def truncate_name_for_filesystem(name: str, max_length: int = MAX_DIR_NAME_LENGTH) -> str:
    """Truncate long file/directory names to avoid filesystem limits (255 chars).
    Append a hash of the full name to maintain uniqueness."""
    if len(name) <= max_length:
        return name
    name_hash = hashlib.md5(name.encode()).hexdigest()[:12]
    truncated = name[: max_length - 13]
    return f"{truncated}_{name_hash}"


def construct_test_output_directory_name():
    test_id: str = derive_pytest_test_id()
    dir_name = f"out-{test_id}"

    return truncate_name_for_filesystem(dir_name)
