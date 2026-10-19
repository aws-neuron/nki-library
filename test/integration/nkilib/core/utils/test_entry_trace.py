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

"""Simulator-mode branch-coverage tests for core/utils/entry_trace.

Every helper in entry_trace is pure-Python trace-time introspection: it only
reads ``.shape`` / ``.dtype`` / ``__dict__`` off whatever object it is handed
and formats strings. Nothing needs an ``nl.ndarray`` or a device backend -- a
numpy array already satisfies ``_is_tensor`` (it has both ``.shape`` and
``.dtype``), so the tests build inputs host-side and call the functions
directly. They carry no trace_only / compile_only marker, so they run (and
count for coverage) under ``--test-mode=simulation``.
"""

import dataclasses

import numpy as np
import pytest
from nkilib_src.nkilib.core.utils import entry_trace
from nkilib_src.nkilib.core.utils.logging import LogLevel


@dataclasses.dataclass
class _Point:
    x: int
    y: int


class _Plain:
    """Object with a populated ``__dict__`` (assignment-order fields)."""

    def __init__(self):
        self.a = 1
        self.b = "two"


class _Empty:
    """Object whose ``__dict__`` exists but is empty."""


@pytest.fixture
def enabled(monkeypatch):
    """Force the module flag on; it is otherwise resolved from env at import."""
    monkeypatch.setattr(entry_trace, "_ENABLED", True)


# ---------------------------------------------------------------------------
# _dtype_name -- L102 isinstance(name, str)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_dtype_name_uses_numpy_name_attribute():
    # numpy dtype exposes a str .name -> L102 true arc (L103)
    assert entry_trace._dtype_name(np.dtype(np.float32)) == "float32"


@pytest.mark.fast
def test_dtype_name_falls_back_to_str():
    # plain object has no str .name -> L102 false arc (L104)
    assert entry_trace._dtype_name(5) == "5"


# ---------------------------------------------------------------------------
# _short_repr -- L120 len(text) > _MAX_REPR
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_short_repr_returns_full_when_short():
    # short repr -> L120 false arc (L122)
    assert entry_trace._short_repr("hi") == "'hi'"


@pytest.mark.fast
def test_short_repr_truncates_when_long():
    # repr longer than _MAX_REPR -> L120 true arc (L121), ellipsis-terminated
    text = entry_trace._short_repr("x" * 200)
    assert text.endswith("...")
    assert len(text) == entry_trace._MAX_REPR


@pytest.mark.fast
def test_short_repr_handles_unreprable_value():
    # repr() raising -> except arc (L117-118): degrade to a type-named placeholder
    class _Boom:
        def __repr__(self):
            raise RuntimeError("no repr")

    assert entry_trace._short_repr(_Boom()) == "<unreprable _Boom>"


# ---------------------------------------------------------------------------
# _shape_str -- L110-111 except arc when the shape is not tuple-able
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_shape_str_falls_back_when_not_iterable():
    # tuple(5) raises TypeError -> except arc (L110-111): str(shape)
    assert entry_trace._shape_str(5) == "5"


# ---------------------------------------------------------------------------
# _iter_fields -- L136 scalar/None, L139 enum, L142 dataclass, L145 __dict__
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("value", [None, True, 3, 3.5, "s", b"bytes"])
def test_iter_fields_scalars_are_opaque(value):
    # L136 true arc (L137) for None and every scalar type
    assert entry_trace._iter_fields(value) is None


@pytest.mark.fast
def test_iter_fields_enum_is_opaque():
    # L136 false -> L139 true arc (L140): an Enum stays one-lined
    assert entry_trace._iter_fields(LogLevel.INFO) is None


@pytest.mark.fast
def test_iter_fields_expands_dataclass():
    # L139 false -> L142 true arc (L143): declared field order
    assert entry_trace._iter_fields(_Point(1, 2)) == [("x", 1), ("y", 2)]


@pytest.mark.fast
def test_iter_fields_expands_object_dict():
    # L142 false -> L145 true arc (L146): assignment-order __dict__ items
    assert entry_trace._iter_fields(_Plain()) == [("a", 1), ("b", "two")]


@pytest.mark.fast
def test_iter_fields_empty_dict_is_opaque():
    # L145 false arc (L149): __dict__ present but empty -> opaque
    assert entry_trace._iter_fields(_Empty()) is None


@pytest.mark.fast
def test_iter_fields_swallows_introspection_errors():
    # Accessing __dict__ raising a non-AttributeError -> except arc (L147-148)
    class _Boom:
        def __getattribute__(self, name):
            if name == "__dict__":
                raise RuntimeError("boom")
            return object.__getattribute__(self, name)

    assert entry_trace._iter_fields(_Boom()) is None


# ---------------------------------------------------------------------------
# _render_value -- L161 tensor, L164 list/tuple, L165 structured element,
#                  L167 element loop, L171 opaque, L175 field loop
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_render_value_tensor():
    # L161 true arc (L162)
    lines = []
    entry_trace._render_value("t", np.zeros((2, 3), dtype=np.float32), lines, 0, 0)
    assert len(lines) == 1
    assert "(2, 3)" in lines[0] and "float32" in lines[0]


@pytest.mark.fast
def test_render_value_opaque_scalar():
    # L161 false -> L164 false (not a sequence) -> L171 true arc (L172)
    lines = []
    entry_trace._render_value("n", 7, lines, 0, 0)
    assert len(lines) == 1
    assert lines[0].startswith("n")
    assert lines[0].endswith("= 7")


@pytest.mark.fast
def test_render_value_list_of_scalars_is_one_lined():
    # L164 true -> L165 false arc (L170): no element is structured -> opaque one-liner
    lines = []
    entry_trace._render_value("xs", [1, 2, 3], lines, 0, 0)
    assert len(lines) == 1
    assert "[1, 2, 3]" in lines[0]


@pytest.mark.fast
def test_render_value_list_of_structured_expands_per_element():
    # L164 true -> L165 true arc (L166) -> element loop L167/L168 -> return L169
    lines = []
    entry_trace._render_value("pts", [_Point(1, 2), _Point(3, 4)], lines, 0, 0)
    header = lines[0]
    assert "list[2]" in header
    # each element expanded as a nested _Point with its fields
    body = "\n".join(lines)
    assert body.count("_Point:") == 2
    assert "x" in body and "y" in body


@pytest.mark.fast
def test_render_value_structured_object_expands_fields():
    # L171 false -> L174 header -> field loop L175/L176 (body + exhaust)
    lines = []
    entry_trace._render_value("p", _Point(1, 2), lines, 0, 0)
    assert lines[0].endswith("_Point:")
    body = "\n".join(lines)
    assert "x" in body and "= 1" in body
    assert "y" in body and "= 2" in body


# ---------------------------------------------------------------------------
# trace_kernel_entry -- L192 gate, L201/L202/L204 arg classification,
#                       L210/L212 tensors, L214/L216 flags, L218 unset
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_trace_kernel_entry_silent_when_disabled(monkeypatch, capsys):
    # L192 true arc (L193): default OFF -> no output
    monkeypatch.setattr(entry_trace, "_ENABLED", False)
    entry_trace.trace_kernel_entry("k", {"a": np.zeros((1, 2), np.float32)})
    assert capsys.readouterr().out == ""


@pytest.mark.fast
def test_trace_kernel_entry_full_signature(enabled, capsys):
    # L192 false -> body; one tensor, one flag, one None covers L202 all arcs,
    # L204 both arcs, L210/L212/L214/L216 true, L218 true.
    args = {
        "hidden": np.zeros((1, 8), dtype=np.float32),
        "num_heads": 8,
        "bias": None,
    }
    entry_trace.trace_kernel_entry("mla", args)
    out = capsys.readouterr().out
    assert "[nki-entry #" in out and "mla" in out
    assert "tensors:" in out and "hidden" in out and "(1, 8)" in out and "float32" in out
    assert "flags:" in out and "num_heads" in out and "8" in out
    assert "none: bias" in out


@pytest.mark.fast
def test_trace_kernel_entry_only_flags(enabled, capsys):
    # No tensors -> L210 false arc (L214); no None -> L218 false arc (L221)
    entry_trace.trace_kernel_entry("k", {"num_heads": 8})
    out = capsys.readouterr().out
    assert "flags:" in out and "num_heads" in out
    assert "tensors:" not in out
    assert "none:" not in out


@pytest.mark.fast
def test_trace_kernel_entry_only_tensor(enabled, capsys):
    # A tensor and nothing else -> L214 false arc (L218); no None -> L218 false (L221)
    entry_trace.trace_kernel_entry("k", {"hidden": np.zeros((2, 4), dtype=np.float32)})
    out = capsys.readouterr().out
    assert "tensors:" in out and "hidden" in out
    assert "flags:" not in out
    assert "none:" not in out


# ---------------------------------------------------------------------------
# trace_kernel_note -- L226 gate
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_trace_kernel_note_silent_when_disabled(monkeypatch, capsys):
    # L226 true arc (L227)
    monkeypatch.setattr(entry_trace, "_ENABLED", False)
    entry_trace.trace_kernel_note("k", "note")
    assert capsys.readouterr().out == ""


@pytest.mark.fast
def test_trace_kernel_note_prints_when_enabled(enabled, capsys):
    # L226 false -> L228
    entry_trace.trace_kernel_note("k", "lnc1 body selected")
    assert capsys.readouterr().out.strip() == "[nki-entry] k: lnc1 body selected"
