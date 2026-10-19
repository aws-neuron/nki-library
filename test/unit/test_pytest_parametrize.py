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

"""Unit tests for pytest_parametrize utility."""

from enum import Enum

import pytest

from test.utils.common_dataclasses import MAX_PATH_COMPONENT_LENGTH
from test.utils.coverage_parametrized_tests import format_param_value
from test.utils.pytest_parametrize import pytest_parametrize, tag_params


class TestFormatParamValue:
    """Tests for format_param_value (shared formatter)."""

    def test_bool_to_int(self):
        assert format_param_value(True) == "1"
        assert format_param_value(False) == "0"

    def test_enum_to_value(self):
        class Color(Enum):
            RED = 1

        assert format_param_value(Color.RED) == "1"

    def test_passthrough(self):
        assert format_param_value(42) == "42"
        assert format_param_value("hello") == "hello"


class TestPytestParametrize:
    """Tests for pytest_parametrize."""

    def test_basic_ids(self):
        mark = pytest_parametrize("a, b", [(1, 2), (3, 4)])
        assert mark.args[1] == [(1, 2), (3, 4)]
        assert mark.kwargs["ids"] == ["a-1_b-2", "a-3_b-4"]

    def test_abbrevs(self):
        mark = pytest_parametrize("tokens, hidden", [(4, 3072)], abbrevs={"tokens": "t", "hidden": "h"})
        assert mark.kwargs["ids"] == ["t-4_h-3072"]

    def test_prefix(self):
        mark = pytest_parametrize("a, b", [(1, 2)], prefix="manual")
        assert mark.kwargs["ids"] == ["_manual_a-1_b-2"]

    def test_bool_and_enum_formatting(self):
        class Mode(Enum):
            FAST = 0

        mark = pytest_parametrize("flag, mode", [(True, Mode.FAST)])
        assert mark.kwargs["ids"] == ["flag-1_mode-0"]

    def test_partial_abbrevs(self):
        mark = pytest_parametrize("vnc, tokens", [(2, 4)], abbrevs={"tokens": "t"})
        assert mark.kwargs["ids"] == ["vnc-2_t-4"]

    def test_length_assertion(self):
        long_names = ", ".join(f"long_parameter_name_{i}" for i in range(20))
        long_vals = [tuple(100000000000 for _ in range(20))]
        with pytest.raises(AssertionError, match="abbrevs"):
            pytest_parametrize(long_names, long_vals)

    def test_length_ok_with_abbrevs(self):
        long_names = ", ".join(f"long_parameter_name_{i}" for i in range(20))
        long_vals = [tuple(1 for _ in range(20))]
        abbrevs = {f"long_parameter_name_{i}": f"p{i}" for i in range(20)}
        mark = pytest_parametrize(long_names, long_vals, abbrevs=abbrevs)
        assert len(mark.kwargs["ids"][0]) <= MAX_PATH_COMPONENT_LENGTH


class TestTagParams:
    """Tests for tag_params."""

    def test_plain_tuples(self):
        params = [(1, 2), (3, 4)]
        result = tag_params("model_a", params)
        assert result == [("model_a", 1, 2), ("model_a", 3, 4)]

    def test_pytest_param_preserves_marks(self):
        params = [pytest.param(1, 2, marks=pytest.mark.fast)]
        result = tag_params("sw", params)
        assert len(result) == 1
        assert result[0].values == ("sw", 1, 2)
        assert any(m.name == "fast" for m in result[0].marks)

    def test_mixed_tuples_and_pytest_param(self):
        params = [(1, 2), pytest.param(3, 4, marks=pytest.mark.slow)]
        result = tag_params("tag", params)
        assert result[0] == ("tag", 1, 2)
        assert result[1].values == ("tag", 3, 4)

    def test_empty_list(self):
        assert tag_params("x", []) == []
