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
from test.utils.pytest_parametrize import _fmt_val, pytest_parametrize


class TestFmtVal:
    """Tests for _fmt_val."""

    def test_bool_to_int(self):
        assert _fmt_val(True) == 1
        assert _fmt_val(False) == 0

    def test_enum_to_value(self):
        class Color(Enum):
            RED = 1

        assert _fmt_val(Color.RED) == 1

    def test_passthrough(self):
        assert _fmt_val(42) == 42
        assert _fmt_val("hello") == "hello"
        assert _fmt_val(None) is None


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
