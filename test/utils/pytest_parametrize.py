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


def _fmt_val(v):
    """Format a parameter value for test ID: bools→int, enums→.value, else str.

    Lists/tuples are joined with 'x' to avoid spaces and commas in test IDs
    (IDs may be used as folder names).
    """
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "value"):
        return v.value
    # numpy dtype types (e.g., np.float16) → short name like "float16"
    if isinstance(v, type) and hasattr(v, "__name__") and hasattr(v, "dtype"):
        return v.__name__
    if isinstance(v, (list, tuple)):
        return "x".join(str(_fmt_val(x)) for x in v)
    return v


def pytest_parametrize(param_names, param_values, abbrevs=None, prefix=None):
    """Drop-in replacement for @pytest.mark.parametrize that auto-generates keyword-prefixed test IDs.

    Args:
        param_names: Comma-separated parameter names string (same as pytest.mark.parametrize).
        param_values: List of parameter tuples (same as pytest.mark.parametrize).
        abbrevs: Optional dict mapping full param names to short aliases.
            Example: {"tokens": "t", "hidden": "h"} → "t-4_h-3072" instead of "tokens-4_hidden-3072".
        prefix: Optional string prefix for test IDs (e.g., "manual" → "_manual_cfg__vnc-2_...").

    Returns:
        pytest.mark.parametrize decorator with auto-generated ids.

    Example::

        @pytest_parametrize("vnc, tokens, hidden", [(2, 4, 3072)], abbrevs={"tokens": "t", "hidden": "h"})
        def test_foo(self, vnc, tokens, hidden): ...
        # Test ID: test_foo[vnc-2_t-4_h-3072]
    """
    names = [n.strip() for n in param_names.split(",")]

    def make_id(params):
        values = params.values if hasattr(params, "values") and not isinstance(params, dict) else params
        parts = []
        for name, val in zip(names, values):
            short = abbrevs.get(name, name) if abbrevs else name
            parts.append(f"{short}-{_fmt_val(val)}")
        id_str = "_".join(parts)
        return f"_{prefix}_{id_str}" if prefix else id_str

    ids = [make_id(p) for p in param_values]
    return pytest.mark.parametrize(param_names, param_values, ids=ids)
