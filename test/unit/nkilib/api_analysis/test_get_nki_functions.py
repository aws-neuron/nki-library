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

"""Tests for the call-graph helpers used by the API analysis reports."""

import ast

import pytest

from .get_nki_functions import _topological_sort


def _func_node(name: str) -> ast.FunctionDef:
    module = ast.parse(f"def {name}():\n    pass\n")
    node = module.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_topological_sort_orders_dependencies_first():
    a = ("a.py", "a")
    b = ("a.py", "b")
    func_info = {a: (1, _func_node("a")), b: (5, _func_node("b"))}
    # a depends on b, so b must be sorted before a.
    sorted_keys = _topological_sort(func_info, {a: {b}, b: set()})

    assert sorted_keys == [b, a]


def test_topological_sort_reports_cycle():
    a = ("a.py", "a")
    b = ("b.py", "b")
    func_info = {a: (1, _func_node("a")), b: (2, _func_node("b"))}

    with pytest.raises(ValueError) as excinfo:
        _topological_sort(func_info, {a: {b}, b: {a}})

    message = str(excinfo.value)
    assert message.startswith("Cycle detected: ")
    assert "a.py:a" in message
    assert "b.py:b" in message
    assert " -> " in message
