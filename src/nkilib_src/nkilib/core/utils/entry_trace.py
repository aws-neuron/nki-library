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

"""Trace-time entry logging for megakernels and their sub-kernels.

A kernel opts in by starting its body with::

    trace_kernel_entry("mla_decode_qkv", locals())

which prints the bound call signature -- tensor shapes/dtypes, flag values, and
which optionals arrived as ``None``. The call sits inside the body, so inlined
sub-kernels print too and ``#N`` orders the whole composition tree.

A config object (dataclass, or anything with a ``__dict__`` such as an SBM) is
expanded one field per line rather than one-lined, recursively, so nested
objects and lists of them are readable too::

    [nki-entry #1] mla_decode_qkv
      tensors:
        hidden_states                  (1, 7168)                bfloat16
        k_cache                        (512, 1, 16, 576)        bfloat16
      flags:
        num_heads                      = 8
        sbm                            = BufferManager:
          lower_bound                  = 0
          upper_bound                  = 131072
          use_auto_alloc               = False
          logger                       = Logger:
            name                       = 'SBM'
            level                      = <LogLevel.INFO: 1>
          scopes                       = list[1]:
            [0]                        = Scope:
              starting_addr            = 0
              num_sections             = 1
              name                     = 'qkv'
          heap                         = []
          ...                                 # remaining fields elided here
      none: input_norm_w

Default is OFF and silent. Enable with ``NKILIB_TRACE_KERNEL_ENTRY=1``,
``NKILIB_LOG_LEVEL_kernel_entry=INFO``, or a library-wide
``NKILIB_LOG_LEVEL=INFO`` (that override outranks this module's code-level
default); ``NKILIB_LOG_LEVEL_kernel_entry=OFF`` opts out of just these dumps.

Only reads ``.shape`` / ``.dtype`` and prints, so instrumenting a kernel does
not change what it computes.

TRACER FRONTEND ONLY -- AND THAT IS THE KERNEL WRITER'S RESPONSIBILITY. Adding a
trace call is opting that kernel out of parser compilation: ParserFrontend
statically rejects both this module's body and ``locals()`` at the call site,
and nothing here checks or enforces it. Instrument a kernel only if you know
nobody compiles it with ``--nki-compilation-mode parser``, and note that the
whole @nki.jit closure inherits the restriction -- a megakernel that inlines an
instrumented sub-kernel is tracer-only too. No runtime guard can help, since the
rejection is by AST reachability rather than execution. This is why the shared
``attention_block_tkg`` / ``transformer_tkg`` kernels are not instrumented.
"""

import dataclasses as _dataclasses
import os as _os
from enum import Enum as _Enum

from .logging import LogLevel, get_logger

# Resolved at import: the parser rejects an os.environ read inside a kernel body.
_DEDICATED_ENV_VAR = "NKILIB_TRACE_KERNEL_ENTRY"
_LOGGER_NAME = "kernel_entry"

# level=OFF is the lowest-priority input, so NKILIB_LOG_LEVEL=INFO still enables.
_logger = get_logger(_LOGGER_NAME, level=LogLevel.OFF)

_ENABLED = _os.environ.get(_DEDICATED_ENV_VAR, "0") not in ("0", "", "false", "False") or _logger.is_enabled_for(
    LogLevel.INFO
)

# Global call counter, so nested inline sub-kernel entries read in call order.
_SEQ = [0]

# Column widths for the aligned per-argument lines.
_NAME_W = 30
_SHAPE_W = 24
# Truncation width for an opaque value's repr; structured values expand instead.
_MAX_REPR = 72
# Recursion cap for nested structured values (a BufferManager holds a Logger).
_MAX_FIELD_DEPTH = 3


def _dtype_name(dtype) -> str:
    """Short readable dtype name for an NKI / numpy dtype object."""
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    return str(dtype)


def _shape_str(shape) -> str:
    try:
        return str(tuple(shape))
    except Exception:  # noqa: BLE001 - trace-time introspection must never break a compile
        return str(shape)


def _short_repr(value) -> str:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001
        return f"<unreprable {type(value).__name__}>"
    text = " ".join(text.split())
    if len(text) > _MAX_REPR:
        return text[: _MAX_REPR - 3] + "..."
    return text


def _is_tensor(value) -> bool:
    """True for anything carrying tensor metadata, i.e. both .shape and .dtype."""
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _iter_fields(value):
    """Field ``(name, value)`` pairs of a structured value, or None if opaque.

    Covers dataclasses (declared field order) and objects with a non-empty
    ``__dict__`` (assignment order); scalars, strings and enums stay opaque.
    """
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return None
    # An enum reads better as its repr than as an expanded name/value pair.
    if isinstance(value, _Enum):
        return None
    try:
        if _dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [(f.name, getattr(value, f.name, "<unset>")) for f in _dataclasses.fields(value)]
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict) and state:
            return list(state.items())
    except Exception:  # noqa: BLE001 - trace-time introspection must never break a compile
        return None
    return None


def _render_value(name, value, lines, indent, depth):
    """Append ``name = value`` to ``lines``, expanding structured values per field.

    An opaque value goes on one line, truncated at ``_MAX_REPR``; a structured one
    recurses to ``_MAX_FIELD_DEPTH``. A sequence expands per element only if some
    element is itself structured.
    """
    pad = " " * indent
    width = max(_NAME_W - indent + 4, 8)
    if _is_tensor(value):
        lines.append(f"{pad}{name:<{width}} {_shape_str(value.shape)} {_dtype_name(value.dtype)}")
        return
    if isinstance(value, (list, tuple)) and depth < _MAX_FIELD_DEPTH:
        if any(_iter_fields(v) is not None or _is_tensor(v) for v in value):
            lines.append(f"{pad}{name:<{width}} = {type(value).__name__}[{len(value)}]:")
            for i, v in enumerate(value):
                _render_value(f"[{i}]", v, lines, indent + 2, depth + 1)
            return
    fields = _iter_fields(value) if depth < _MAX_FIELD_DEPTH else None
    if not fields:
        lines.append(f"{pad}{name:<{width}} = {_short_repr(value)}")
        return
    lines.append(f"{pad}{name:<{width}} = {type(value).__name__}:")
    for fname, fvalue in fields:
        _render_value(fname, fvalue, lines, indent + 2, depth + 1)


def trace_kernel_entry(kernel_name: str, args: dict) -> None:
    """Print the bound call signature of ``kernel_name``; no-op unless enabled.

    Call as the first statement of a kernel body, passing ``locals()`` -- at entry
    that is exactly the bound parameters, so no argument list is kept by hand.

    Calling this makes the kernel TRACER-ONLY, and the caller owns that: it is
    never checked, so a parser-compiled caller fails to compile.

    Args:
        kernel_name: name to tag the entry with (the kernel's own name).
        args: parameter name -> bound value, normally ``locals()``.
    """
    if not _ENABLED:
        return

    _SEQ[0] += 1
    lines = [f"[nki-entry #{_SEQ[0]}] {kernel_name}"]

    tensors = []
    flags = []
    unset = []
    for name, value in args.items():
        if value is None:
            unset.append(name)
        elif _is_tensor(value):
            tensors.append((name, _shape_str(value.shape), _dtype_name(value.dtype)))
        else:
            # keep the raw value: _render_value decides one-line vs expanded.
            flags.append((name, value))

    if tensors:
        lines.append("  tensors:")
        for name, shape, dtype in tensors:
            lines.append(f"    {name:<{_NAME_W}} {shape:<{_SHAPE_W}} {dtype}")
    if flags:
        lines.append("  flags:")
        for name, value in flags:
            _render_value(name, value, lines, 4, 0)
    if unset:
        lines.append(f"  none: {', '.join(unset)}")

    print("\n".join(lines))


def trace_kernel_note(kernel_name: str, note: str) -> None:
    """Print a one-line note, e.g. which specialized body a dispatcher selected."""
    if not _ENABLED:
        return
    print(f"[nki-entry] {kernel_name}: {note}")
