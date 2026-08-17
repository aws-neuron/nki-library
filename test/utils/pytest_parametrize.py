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

"""Drop-in replacement for @pytest.mark.parametrize with keyword-prefixed test IDs."""

import pytest

from .coverage_parametrized_tests import MAX_PATH_COMPONENT_LENGTH, format_param_value


def pytest_parametrize(param_names, param_values, abbrevs=None, prefix=None, test_func_name=None):
    """Drop-in replacement for @pytest.mark.parametrize that auto-generates keyword-prefixed test IDs.

    Args:
        param_names: Comma-separated parameter names string (same as pytest.mark.parametrize).
        param_values: List of parameter tuples (same as pytest.mark.parametrize).
        abbrevs: Optional dict mapping full param names to short aliases.
            Example: {"tokens": "t", "hidden": "h"} → "t-4_h-3072" instead of "tokens-4_hidden-3072".
        prefix: Optional string prefix for test IDs (e.g., "manual" → "_manual_cfg__vnc-2_...").
        test_func_name: Optional test function name for full path component length
            validation. When provided, validates ``len(test_func_name) + 2 +
            len(test_id) <= 255``. When ``None``, only the test ID is checked.

    Returns:
        pytest.mark.parametrize decorator with auto-generated ids.

    Example::

        @pytest_parametrize("vnc, tokens, hidden", [(2, 4, 3072)], abbrevs={"tokens": "t", "hidden": "h"})
        def test_foo(self, vnc, tokens, hidden): ...
        # Test ID: test_foo[vnc-2_t-4_h-3072]
    """
    names = [n.strip() for n in param_names.split(",")]
    overhead = len(test_func_name) + 2 if test_func_name else 0

    def make_id(params):
        if hasattr(params, "values") and not isinstance(params, dict):
            values = params.values
        elif len(names) == 1:
            # Single-name parametrize passes bare scalars (not tuples); wrap so a
            # scalar (e.g. a dtype string) isn't iterated character-by-character.
            values = (params,)
        else:
            values = params
        parts = []
        for name, val in zip(names, values, strict=True):
            short = abbrevs.get(name, name) if abbrevs else name
            parts.append(f"{short}-{format_param_value(val)}")
        id_str = "_".join(parts)
        test_id = f"_{prefix}_{id_str}" if prefix else id_str
        full_len = overhead + len(test_id)
        assert full_len <= MAX_PATH_COMPONENT_LENGTH, (
            f"Test ID length {full_len} exceeds {MAX_PATH_COMPONENT_LENGTH}. "
            f"Use abbrevs to shorten parameter names. ID: {test_id}"
        )
        return test_id

    ids = [make_id(p) for p in param_values]
    return pytest.mark.parametrize(param_names, param_values, ids=ids)


def tag_params(tag, params):
    """Prepend a tag value to each param tuple, preserving pytest.param marks.

    Useful for adding a model/group identifier to parametrized test vectors.

    Example::

        SWITCH = [(4, 4096), (1, 1)]
        DEEP = [pytest.param(2, 4096, marks=pytest.mark.fast)]
        ALL = tag_params("switch", SWITCH) + tag_params("deep", DEEP)
        # → [("switch", 4, 4096), ("switch", 1, 1), pytest.param("deep", 2, 4096, marks=fast)]
    """
    return [pytest.param(tag, *p.values, marks=p.marks) if hasattr(p, "values") else (tag,) + p for p in params]
