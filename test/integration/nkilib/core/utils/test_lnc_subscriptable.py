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

"""Host-side tests for the pure-Python LncSubscriptable dispatch wrapper.

LncSubscriptable wraps a kernel function and, at construction time, verifies the
function's signature and docstring stay in sync with a torch-reference Protocol
(_verify_protocol_sync). Because that check inspects the *caller's* source frame
for the assignment target and its docstring, the happy path can only be produced
by a module-level annotated assignment (see ``lnc_ok`` below). The frame/docstring
failure arcs are a developer-time consistency check that only fires at import time
against a real caller frame, so they are not exercised here (see the
``# pragma: no cover`` on those arcs in lnc_subscriptable.py). Everything is plain
Python and runs under the simulator with no device kernel or marker.
"""

from typing import Protocol

import pytest
from nkilib_src.nkilib.core.utils.lnc_subscriptable import (
    LncSubscriptable,
    _compare_docstrings,
    _extract_var_docstring,
    _find_assignment_target,
)


class MatchProto(Protocol):
    def __call__(self, a: int, b: float) -> int:
        """Compute the reference."""
        ...


def _match_impl(a: int, b: float) -> int:
    return a


# Module-level annotated assignment + matching docstring: this is the ONLY way to
# drive _verify_protocol_sync all the way through the frame/docstring stage
# successfully (var name found -> L102, var_doc found -> L106, no diff -> L110
# falls through). Runs at import time; must succeed.
lnc_ok: LncSubscriptable = LncSubscriptable(_match_impl, MatchProto)
"""Compute the reference."""


# --------------------------------------------------------------------------- #
# LncSubscriptable object behavior (happy path already constructed as lnc_ok)  #
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_construction_exposes_wrapped_signature():
    """Successful construction records __wrapped__ / __signature__ from the func."""
    assert lnc_ok.__wrapped__ is _match_impl
    assert list(lnc_ok.__signature__.parameters) == ["a", "b"]


@pytest.mark.fast
def test_getitem_sets_lnc_and_dispatches():
    """__getitem__ returns a wrapper that records the LNC value and calls the func."""
    result = lnc_ok[3](5, 2.0)
    assert result == 5  # _match_impl returns its first arg
    assert lnc_ok.lnc == 3  # wrapper recorded the subscripted LNC value


@pytest.mark.fast
def test_call_without_subscript_raises_typeerror():
    """Calling without subscripting raises a guidance TypeError."""
    with pytest.raises(TypeError, match="must be subscripted with LNC value"):
        lnc_ok(5, 2.0)


@pytest.mark.fast
def test_shard_id_getter_and_setter():
    """shard_id defaults to 0 and can be set."""
    assert lnc_ok.shard_id == 0
    lnc_ok.shard_id = 2
    assert lnc_ok.shard_id == 2
    lnc_ok.shard_id = 0  # restore for isolation


# --------------------------------------------------------------------------- #
# _verify_protocol_sync signature-mismatch failure paths (raise before frame)  #
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_param_count_mismatch_raises():
    """Function/Protocol parameter-count mismatch fails the count kernel_assert."""

    class P(Protocol):
        def __call__(self, a: int) -> int: ...

    def f(a: int, b: int) -> int:
        return a

    with pytest.raises(AssertionError, match="Parameter count mismatch"):
        LncSubscriptable(f, P)


@pytest.mark.fast
def test_param_name_mismatch_raises():
    """Matching counts but different parameter names fail inside the zip loop (L79 -> L80)."""

    class P(Protocol):
        def __call__(self, b: int) -> int: ...

    def f(a: int) -> int:
        return a

    with pytest.raises(AssertionError, match="Parameter name mismatch"):
        LncSubscriptable(f, P)


@pytest.mark.fast
def test_param_annotation_mismatch_raises():
    """Matching names but different annotations fail the annotation kernel_assert."""

    class P(Protocol):
        def __call__(self, a: str) -> int: ...

    def f(a: int) -> int:
        return a  # type: ignore[return-value]

    with pytest.raises(AssertionError, match="annotation mismatch"):
        LncSubscriptable(f, P)


@pytest.mark.fast
def test_return_annotation_mismatch_raises():
    """Matching params but different return annotation fails the return kernel_assert."""

    class P(Protocol):
        def __call__(self, a: int) -> str: ...

    def f(a: int) -> int:
        return a

    with pytest.raises(AssertionError, match="Return annotation mismatch"):
        LncSubscriptable(f, P)


# --------------------------------------------------------------------------- #
# _verify_protocol_sync frame/docstring failure paths                          #
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_missing_assignment_target_raises_runtimeerror():
    """A function-local construction has no module-level assignment target, so
    _find_assignment_target returns None -> RuntimeError (L100 -> L101)."""

    class P(Protocol):
        def __call__(self, a: int) -> int:
            """Compute the reference."""
            ...

    def f(a: int) -> int:
        return a

    with pytest.raises(RuntimeError, match="Could not find assignment target"):
        _local = LncSubscriptable(f, P)  # noqa: F841


# --------------------------------------------------------------------------- #
# _find_assignment_target: direct AST-walk branch coverage                     #
# --------------------------------------------------------------------------- #

_FIND_SRC = (
    "a = 1\n"  # 1: Assign, Name target
    "b: int = 2\n"  # 2: AnnAssign with value, Name target
    "c, d = 3, 4\n"  # 3: Assign, tuple target (targets[0] not a Name)
    "e: int\n"  # 4: AnnAssign with no value
    "import os\n"  # 5: neither Assign nor AnnAssign
    "o.p: int = 5\n"  # 6: AnnAssign with value, Attribute target (not a Name)
)


@pytest.mark.fast
def test_find_assignment_target_plain_assign():
    """Plain assignment whose line is in range with a Name target -> returns it."""
    assert _find_assignment_target(_FIND_SRC, 1) == "a"


@pytest.mark.fast
def test_find_assignment_target_annassign():
    """AnnAssign with value + Name target in range -> returns it (elif branch)."""
    assert _find_assignment_target(_FIND_SRC, 2) == "b"


@pytest.mark.fast
def test_find_assignment_target_none_for_non_name_targets():
    """Line 3 (tuple Assign target) and line 4 (valueless AnnAssign) yield no match;
    the loop exhausts and returns None."""
    assert _find_assignment_target(_FIND_SRC, 3) is None


@pytest.mark.fast
def test_find_assignment_target_annassign_non_name_target():
    """AnnAssign with value but an Attribute target (not a Name) -> None (L124 -> continue)."""
    assert _find_assignment_target(_FIND_SRC, 6) is None


# --------------------------------------------------------------------------- #
# _extract_var_docstring: direct AST-walk branch coverage                      #
# --------------------------------------------------------------------------- #

_DOC_SRC = (
    "v: int = 1\n"  # 1: AnnAssign Name v ...
    '"""doc for v"""\n'  # 2: ... followed by a string docstring
    "w: int = 2\n"  # 3: AnnAssign Name w (docstring is NOT the next node)
    "x = 3\n"  # 4: plain Assign (next node after w)
    "y: int = 4\n"  # 5: AnnAssign Name y
    "z: int = 6\n"  # 6: AnnAssign Name z ...
    "42\n"  # 7: ... followed by a non-string constant Expr
    "last: int = 9\n"  # 8: AnnAssign Name last as the final body node
)


@pytest.mark.fast
def test_extract_var_docstring_found():
    """Var declaration immediately followed by a string literal -> returns docstring."""
    assert _extract_var_docstring(_DOC_SRC, "v") == "doc for v"


@pytest.mark.fast
def test_extract_var_docstring_next_node_not_docstring():
    """Var whose following node is not a string Expr (nor is any var found) -> None.

    Walking for 'w' also passes non-AnnAssign nodes and a name-mismatched var,
    exercising the L133/L134/L137 false arcs and the L132 -> L140 exhaustion."""
    assert _extract_var_docstring(_DOC_SRC, "w") is None


@pytest.mark.fast
def test_extract_var_docstring_next_node_non_string_constant():
    """Var followed by a numeric constant Expr -> the isinstance(str) guard is False (L138)."""
    assert _extract_var_docstring(_DOC_SRC, "z") is None


@pytest.mark.fast
def test_extract_var_docstring_var_is_last_node():
    """Var declared as the final body node has no following node (L135 -> continue) -> None."""
    assert _extract_var_docstring(_DOC_SRC, "last") is None


# --------------------------------------------------------------------------- #
# _compare_docstrings                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_compare_docstrings_match_returns_none():
    """Equal (after dedent/strip) docstrings compare equal -> None (L149 false arc)."""
    assert _compare_docstrings("  hello\n", "hello") is None


@pytest.mark.fast
def test_compare_docstrings_mismatch_returns_diff():
    """Differing docstrings produce a unified diff string (L149 -> L150)."""
    diff = _compare_docstrings("hello", "world")
    assert diff is not None
    assert "hello" in diff and "world" in diff
