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

"""Pin the vendored c128 gigakernel packed kernel against the live one.

``models/gpt_oss/c128_giga_kernel`` is a hand-maintained vendored snapshot, not an
import of ``core``. The packed kernel was copied verbatim and the whole port is
reviewable only because the copy stays equal to live apart from a small, known and
fully specified delta. Nothing else in the package enforces that, so editing the live
kernel without mirroring it would diverge the copy silently.

The delta is NOT cosmetic. The ISA-enum spelling is, but the per-layer DMA-name
qualification and the ``sbm`` threading that carries it change what the kernel emits:
unqualified names collide across inlined decoder layers and fail at trace time. So
"known delta" here means specified and reviewed, never harmless.

This test removes the need to trust that. It NORMALISES the vendored file by the
known divergences and requires the result to equal live byte for byte, so a count
of differences cannot pass by swapping one divergence for another.

Sibling in spirit: ``test_packed_fa_tile_size_matches_builder_assumption`` in
test/integration/nkilib/experimental/gpt_oss/test_attention_block_tkg_packed_gptoss.py,
which pins the other cross-tree coupling. This one is a unit test so it runs in the
default build gate rather than only when someone selects the integration suite.
"""

import ast
import difflib
import re
from pathlib import Path

import nkilib_src

_NKILIB = Path(nkilib_src.__file__).parent / "nkilib"
_LIVE = _NKILIB / "core" / "attention" / "attention_tkg_seq_packed.py"
_VENDORED = _NKILIB / "models" / "gpt_oss" / "c128_giga_kernel" / "core" / "attention" / "attention_tkg_seq_packed.py"

# The vendored copy imports the three ISA enums directly and uses the bare names;
# live reaches them through the `nisa` module alias. Functionally identical.
_VENDOR_ONLY_IMPORT = "from nki.isa import dge_mode, dma_engine, oob_mode\n"
_BARE_ENUMS = ("dge_mode", "dma_engine", "oob_mode")

# The spec lives in core/attention, so the vendored docstring must name the path
# rather than say "in this directory". Renamed upstream to
# attention_tkg_sequence_packing_design_spec.md, and both sides track the new name.
_VENDORED_DOC = "    See nkilib/core/attention/attention_tkg_sequence_packing_design_spec.md for the\n"
_LIVE_DOC = "    See attention_tkg_sequence_packing_design_spec.md in this directory for the\n"

# Update these two deliberately, together with the normalisation above, if the
# vendored delta is ever meant to change. Moved 15->17 lines and 8->9 hunks when the
# packed kernel was re-vendored from post-CR-303397311-rev-2 live: that revision added
# one more bare-enum call site. Parity itself still holds byte for byte, so this is the
# delta tracking live rather than a widened tolerance absorbing a real divergence. Moved 17->20
# lines and 9->11 hunks when the packed leaf gained per-layer DMA-name qualifiers: the 29 label
# changes normalise away BY RULE and contribute nothing, so the +3 is exactly the two-line ``sbm``
# threading enumerated above, counted as -/+ pairs.
_EXPECTED_CHANGED_SOURCE_LINES = 20
_EXPECTED_HUNKS = 11


# The fork inlines this leaf once per decoder layer, so its explicit DMA instruction names
# must be per-layer unique; live is not inlined and does not need that. The fork therefore
# qualifies them with ``sbm.get_name_prefix()``. That divergence is expressed as a RULE applied
# to BOTH sides rather than as a list of exempt lines: a line list would grow with every new
# DMA and would let one divergence be swapped for another unnoticed, which is the same weakness
# a count has. Applied symmetrically because live already qualifies five of these labels itself,
# so stripping only the fork would break the comparison it is meant to make.
_DMA_NAME_PREFIX_CALL = "{sbm.get_name_prefix()}"

# Threading the SBUF manager into ``_load_mask`` so it can reach the qualifier above. Two lines,
# enumerated rather than normalised on purpose: a rule broad enough to erase a signature change
# would erase real ones too, and this one wants a reviewer's eye every time it moves.
# Anchored on surrounding context, not on the bare line: ``sbm: SbufManager,`` appears many times
# in this file legitimately, so removing it unanchored would erase real signatures.
_STRUCTURAL_SUBSTITUTIONS = (
    (
        "    fa_ctx: FATileContext,\n    sbm: SbufManager,\n):",
        "    fa_ctx: FATileContext,\n):",
    ),
    (
        "    _load_mask(mask, TC, bufs, fa_ctx, sbm)\n",
        "    _load_mask(mask, TC, bufs, fa_ctx)\n",
    ),
)


def _undo_structural_divergence(text: str) -> str:
    """Undo the two-line ``sbm`` threading, each anchored so it cannot over-match."""
    for vendored_form, live_form in _STRUCTURAL_SUBSTITUTIONS:
        assert text.count(vendored_form) == 1, (
            f"expected exactly one occurrence of the vendored form {vendored_form!r}; "
            "the anchor has drifted and this normalisation is no longer safe"
        )
        text = text.replace(vendored_form, live_form, 1)
    return text


def _strip_dma_name_prefix(text: str) -> str:
    """Remove the per-layer DMA-name qualifier from ``name=f"..."`` labels.

    Two steps, because removing the qualifier can leave the ``f`` prefix with nothing to
    interpolate: a label that was ``name="x"`` upstream becomes ``name=f"{...}x"`` in the fork,
    and stripping the qualifier alone would leave ``name=f"x"``, which is not what upstream
    writes. Labels that interpolate something else keep their ``f`` legitimately, so the second
    step only fires when no brace remains.
    """
    text = text.replace(f'name=f"{_DMA_NAME_PREFIX_CALL}', 'name=f"')
    text = re.sub(r'name=f"([^"{}]*)"', r'name="\1"', text)
    # A label FORWARDED through a helper reads `name=f"{sbm.get_name_prefix()}{name}"` in the fork
    # and `name=name` upstream, so once the qualifier is gone a lone interpolation collapses back to
    # the bare expression. Anchored on the parameter actually forwarded rather than any \w+: a
    # broader pattern would also absorb `name=f"{other}"` -> `name=other` for any identifier, which
    # is a divergence class this delta does not claim to cover.
    return re.sub(r'name=f"\{name\}"', "name=name", text)


def _normalise_vendored(text: str) -> str:
    """Rewrite the vendored source into the form live is expected to have."""
    text = text.replace(_VENDOR_ONLY_IMPORT, "", 1)
    text = text.replace(_VENDORED_DOC, _LIVE_DOC, 1)
    text = _strip_dma_name_prefix(text)
    text = _undo_structural_divergence(text)
    return "".join(_prefix_bare_enums(line) for line in text.splitlines(keepends=True))


def _prefix_bare_enums(line: str) -> str:
    """Qualify bare ISA-enum uses in CODE, leaving comment text alone.

    Both trees write the enums unqualified in prose (e.g. "# ... uses oob_mode.skip
    to skip these entries"), so rewriting comments invents divergences that are not
    there -- which is exactly what this normaliser did on its first outing.
    """
    code, sep, comment = line.partition("#")
    for enum in _BARE_ENUMS:
        # Only bare uses: never re-prefix an already-qualified `nisa.<enum>`, and
        # never touch a longer identifier that merely ends in the same characters.
        code = re.sub(rf"(?<![\w.]){enum}\.", f"nisa.{enum}.", code)
    return code + sep + comment


def test_vendored_packed_kernel_matches_live_modulo_known_delta():
    """The vendored packed kernel must be live plus only the known, specified delta."""
    live = _strip_dma_name_prefix(_LIVE.read_text())
    vendored = _VENDORED.read_text()
    normalised = _normalise_vendored(vendored)

    if normalised != live:
        diff = "".join(
            difflib.unified_diff(
                normalised.splitlines(keepends=True),
                live.splitlines(keepends=True),
                fromfile="vendored (normalised)",
                tofile="live",
                n=2,
            )
        )
        raise AssertionError(
            "The vendored c128 gigakernel packed kernel has diverged from live beyond the "
            "known and fully specified delta.\n\n"
            f"  live:     {_LIVE}\n"
            f"  vendored: {_VENDORED}\n\n"
            "If you edited the live kernel, mirror the change into the vendored copy. If the "
            "vendored delta is genuinely meant to grow, update _normalise_vendored and the two "
            "_EXPECTED_* constants in this file in the same change, so the new delta is reviewed "
            "rather than absorbed.\n\n"
            f"{diff}"
        )


def test_every_named_dma_in_the_vendored_leaf_is_layer_qualified():
    """Every named DMA in the vendored packed leaf must carry the per-layer qualifier.

    THE PARITY TEST ABOVE CANNOT DO THIS JOB, and that is why this one exists. It strips the
    qualifier from BOTH trees before comparing, so a vendored label that never had one normalises
    to exactly what live writes and the comparison passes. It is a guard against DRIFT, not
    against INCOMPLETENESS, and citing it for the latter was a mistake.

    This check follows the label through helper FORWARDING, which is what a `name=` census misses:
    two labels survived every earlier count because they were passed positionally into
    ``_load_and_broadcast_pos_ids``, which forwards them as ``name=name``. A scan for ``name=``
    string literals cannot see those, so it reported complete while two labels were bare.

    A bare identifier is exempt only when it is the enclosing function's forwarded parameter, whose
    call sites arm 2 checks instead. Any other identifier -- a local alias this walk cannot resolve
    -- is REPORTED rather than skipped, so incompleteness fails loudly instead of passing silently.

    The two arms are complementary, and arm 2's forwarder map is EMPTY at this tip by construction
    rather than by breakage: ``_load_and_broadcast_pos_ids`` qualifies at its own emit site, so its
    ``name=`` is an f-string and arm 1 checks it directly. Revert that helper to forwarding bare and
    the map fills, arm 1 exempts the parameter, and arm 2 reports both call sites. Said explicitly
    because an empty map otherwise reads like an inert check.

    Why it matters: the mega loop inlines this leaf once per decoder layer and ``dma_names.py``
    requires NEFF-global uniqueness, so an unqualified label collides across layers and raises
    ``duplicate op name`` at trace time. The repo records that exact failure on ``sink_load_bt0``.
    """
    tree = ast.parse(_VENDORED.read_text())

    def is_qualified(node) -> bool:
        """True when the expression interpolates the per-layer qualifier."""
        return isinstance(node, ast.JoinedStr) and _DMA_NAME_PREFIX_CALL.strip("{}") in ast.unparse(node)

    # Functions that forward a parameter straight into a named DMA: label -> param name.
    forwarders: dict[str, str] = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        params = {a.arg for a in fn.args.args}
        for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
            if ast.unparse(call.func) != "nisa.dma_copy":
                continue
            for kw in call.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Name) and kw.value.id in params:
                    forwarders[fn.name] = kw.value.id

    unqualified: list[str] = []

    # 1. Direct named DMAs must be qualified in place. A bare identifier is exempt ONLY when it is
    #    the enclosing function's forwarded parameter, which arm 2 checks at every call site
    #    instead; any other identifier is a local alias whose value this walk cannot see, so it is
    #    reported rather than skipped. Exempting every ast.Name would let `label = "foo"` followed
    #    by `name=label` pass silently -- the same blind spot as the `name=` census, one level in.
    def visit(node, forwarded: str | None):
        """Walk, tracking the NEAREST enclosing function so each call is judged once."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                visit(child, forwarders.get(child.name))
                continue
            if isinstance(child, ast.Call) and ast.unparse(child.func) == "nisa.dma_copy":
                for kw in child.keywords:
                    if kw.arg != "name" or is_qualified(kw.value):
                        continue
                    if isinstance(kw.value, ast.Name) and kw.value.id == forwarded:
                        continue
                    unqualified.append(f"line {kw.value.lineno}: nisa.dma_copy(name={ast.unparse(kw.value)})")
            visit(child, forwarded)

    visit(tree, None)

    # 2. Every call INTO a forwarder must pass a qualified label. This is the arm that catches a
    #    label handed in positionally, which no `name=` scan can see.
    for call in (c for c in ast.walk(tree) if isinstance(c, ast.Call)):
        fname = ast.unparse(call.func)
        if fname not in forwarders:
            continue
        param = forwarders[fname]
        passed = next((kw.value for kw in call.keywords if kw.arg == param), None)
        if passed is None and call.args:
            passed = call.args[-1]  # the forwarded label is the trailing positional
        if passed is not None and not is_qualified(passed):
            unqualified.append(f"line {passed.lineno}: {fname}(..., {ast.unparse(passed)})")

    assert not unqualified, (
        "these DMA labels in the vendored packed leaf are not per-layer qualified, so they collide "
        "across the mega loop's per-layer inlining and raise `duplicate op name` at trace time:\n  "
        + "\n  ".join(unqualified)
    )


def test_vendored_delta_is_the_expected_shape():
    """Pin the delta's size and composition, so it cannot drift silently either.

    The parity test above is the real guard. This one fails if the delta changes
    shape while still normalising cleanly -- for example if someone adds a second
    vendored-only import that the normaliser happens to strip.

    Note what the composition assert below can and cannot do: the count assert runs FIRST, so the
    per-line classification is only ever reached on a tree whose changed-line count already equals
    _EXPECTED_CHANGED_SOURCE_LINES. Its domain is therefore count-PRESERVING divergences -- exactly
    the "one divergence swapped for another of equal size" case this delta exists to catch -- and it
    is structurally incapable of classifying anything that moves the count. That is by design, but
    "the composition assert validates every divergence" is the wrong reading.
    """
    live = _strip_dma_name_prefix(_LIVE.read_text()).splitlines(keepends=True)
    vendored = _strip_dma_name_prefix(_VENDORED.read_text()).splitlines(keepends=True)

    changed = [ln for ln in difflib.ndiff(vendored, live) if ln[:1] in ("-", "+")]
    hunks = sum(1 for ln in difflib.unified_diff(vendored, live, n=0) if ln.startswith("@@"))

    assert len(changed) == _EXPECTED_CHANGED_SOURCE_LINES, (
        f"expected {_EXPECTED_CHANGED_SOURCE_LINES} changed source lines between the vendored "
        f"and live packed kernels, found {len(changed)}:\n" + "".join(changed)
    )
    assert hunks == _EXPECTED_HUNKS, f"expected {_EXPECTED_HUNKS} diff hunks, found {hunks}"

    # The delta must be exactly: one vendored-only import, one docstring path, and
    # `nisa.`-prefix differences. Anything else is a real divergence.
    assert _VENDOR_ONLY_IMPORT in vendored, "the vendored-only nki.isa import is gone; update this test"
    for ln in changed:
        body = ln[2:]
        if body in (_VENDOR_ONLY_IMPORT, _VENDORED_DOC, _LIVE_DOC):
            continue
        # Derived from the same constant the normaliser uses, so the allowed structural lines
        # cannot drift apart from the substitution that erases them.
        structural_lines = {
            line.strip()
            for pair in _STRUCTURAL_SUBSTITUTIONS
            for form in pair
            for line in form.splitlines()
            if line.strip()
        }
        if body.strip() in structural_lines:
            continue
        assert any(e in body for e in _BARE_ENUMS) or _DMA_NAME_PREFIX_CALL in body, (
            "unexpected divergence: not an ISA-enum prefix, a per-layer DMA-name qualifier, or one "
            f"of the enumerated structural lines:\n{ln}"
        )
