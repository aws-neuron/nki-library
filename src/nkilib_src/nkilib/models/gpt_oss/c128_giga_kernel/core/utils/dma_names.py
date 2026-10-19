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

"""Stable instruction names for DMA-order-constraint pinning.

A DMA-order constraint file (``--dma-order-constraint-file``) identifies each op by its BIR
instruction name. An op given no ``name=`` gets a compiler-assigned *positional* name (``I-<n>``,
n = its index in the instruction stream), so adding or removing any earlier DMA renumbers
everything after it: a tuned schedule then still "matches" but binds to different instructions.
Passing ``name=`` makes the compiler emit ``U-<name>`` instead, which survives unrelated codegen
changes.

Names must be unique within one NEFF, and subkernels are inlined once per decoder layer, so a
call site's own loop indices are not sufficient on their own. The pattern is:

    set_dma_name_scope(f"L{layer_idx}_")            # caller, once per layer
    nisa.dma_copy(dst=..., src=..., name=dma_name(f"wout_w_h{h}"))   # call site

:func:`dma_name` returns ``None`` until a scope is set, and ``name=None`` is the default for every
``nisa`` op. Naming is therefore inert for any kernel that does not opt in by setting a scope,
even though the call sites live in shared code.
"""

from .kernel_assert import kernel_assert

_scope: str | None = None
_issued: set[str] = set()


def set_dma_name_scope(scope: str | None) -> None:
    """Begin naming DMA ops under ``scope``, or pass ``None`` to stop naming them.

    Call once per inlined instance (per decoder layer) with a scope unique to that instance.
    Duplicate detection resets on every call: names carry the scope as a prefix, so uniqueness
    within a scope is enough to make them unique across the whole NEFF.
    """
    global _scope
    _scope = scope
    _issued.clear()


def get_dma_name_scope() -> str | None:
    """Return the active scope, or ``None`` when DMA naming is off."""
    return _scope


def dma_name(base: str) -> str | None:
    """Qualify ``base`` with the active scope, or return ``None`` when naming is off.

    Raises when a scope+base pair repeats: two ops sharing a name is exactly the silent
    mis-binding that naming exists to prevent, so it fails the trace instead.
    """
    if _scope is None:
        return None
    name = f"{_scope}{base}"
    kernel_assert(
        name not in _issued,
        f"duplicate DMA instruction name {name!r}: the call site needs another index to tell its "
        f"instances apart, otherwise a DMA-order constraint on this name binds to an arbitrary one",
    )
    _issued.add(name)
    return name
