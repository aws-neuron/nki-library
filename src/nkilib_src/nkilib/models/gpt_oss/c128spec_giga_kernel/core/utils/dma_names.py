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

_scope: str | None = None
# name -> how many times it has been requested in the current scope. A dict rather than a set
# because repeats are disambiguated by index instead of rejected; see dma_name().
_issued: dict[str, int] = {}


def set_dma_name_scope(scope: str | None) -> None:
    """Begin naming DMA ops under ``scope``, or pass ``None`` to stop naming them.

    Call once per inlined instance (per decoder layer) with a scope unique to that instance.
    The duplicate-index table resets on every call: names carry the scope as a prefix, so
    uniqueness within a scope is enough to make them unique across the whole NEFF.
    """
    global _scope
    _scope = scope
    _issued.clear()


def get_dma_name_scope() -> str | None:
    """Return the active scope, or ``None`` when DMA naming is off."""
    return _scope


def dma_name(base: str) -> str | None:
    """Qualify ``base`` with the active scope, or return ``None`` when naming is off.

    A repeated scope+base is **disambiguated by index**, not rejected: the FIRST instance keeps its
    unsuffixed name and the Nth gets ``<name>_d<N>``.

    Why this fork diverges from the upstream assert. Upstream fails the trace on a repeat, on the
    grounds that "a DMA-order constraint on this name binds to an arbitrary one". That is the right
    call for a name that is genuinely ambiguous, but it makes DMA-order scheduling **structurally
    impossible for any T_ep > 128 configuration**, which includes every arm of this kernel:
    ``attention_tkg``'s v-index deinterleave name encodes only ``(num_groups, N)`` while its memo
    key also carries ``id(v_idx_src)``, ``s_prior``, ``tile_offset`` and ``tile_s_prior``, so
    geometry-equal-but-distinct entries collide. Confirmed not spec-specific: the non-spec kernel
    hits the same collision at ``T_SHARD=4``.

    Indexing removes the ambiguity instead of refusing to proceed. Trace order is deterministic for
    a fixed config, so every op gets a unique STABLE name, which is all a constraint file needs. The
    first-instance name is deliberately left unsuffixed so any constraint file that already
    references it stays bound -- the alternative (adding a discriminator at the call site) would
    rename every instance and silently unbind existing schedules while still reporting zero
    mismatches. A schedule derived under a config with fewer duplicates simply under-covers one with
    more, which is a visible, expected condition measured by the OrderDMA match count.

    ``dma_names.py`` is vendored per giga-kernel fork, so this change cannot affect the c128 fork or
    the constraint files bound against it.
    """
    if _scope is None:
        return None
    name = f"{_scope}{base}"
    if name not in _issued:
        _issued[name] = 1
        return name
    n = _issued[name] + 1
    # Skip indices already taken by a literal call site that happened to spell the suffixed form.
    while f"{name}_d{n}" in _issued:
        n += 1
    _issued[name] = n
    uniq = f"{name}_d{n}"
    _issued[uniq] = 1
    return uniq


# Non-DMA instructions take the same optional ``name=``; the c512 line names compute ops through this alias so the
# same scope and duplicate handling apply (CR-302598650).
inst_name = dma_name
