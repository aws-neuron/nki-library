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

"""Tests for the chunked (wide-vocab) top-k kernel.

THE GATE THAT MATTERS: value<->index PAIRING
--------------------------------------------
This kernel's characteristic failure is a wrong chunk offset -- stage 1 returns indices
local to a chunk, and the kernel maps them to global vocab ids with ``c*(V/P)``. Get that
wrong and:

  * the VALUES are still exactly the true global top-k (the reduce ran on real values), so
    a value-set comparison passes;
  * the indices are still inside ``[0, vocab)``, so a range check passes.

Only ``inp[i, idx[i, j]] == val[i, j]`` catches it. ``test_offset_gate_can_fail`` proves
that, by running the numpy model of this kernel with the offset deliberately dropped and
asserting the value-set and range checks stay silent while the pairing check fires. Every
hardware case below carries the pairing check for that reason.

TIES ARE FORGIVEN ON INDICES, NOT ON VALUES
-------------------------------------------
bf16 keeps 8 mantissa bits, so a 201,088-wide row has many exact ties and WHICH index wins
a tied slot is not determined. Values are therefore compared as a sorted multiset and
indices only through the pairing check plus a distinctness check -- never against the
oracle's index set directly.
"""

import os
from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
import torch
from nkilib_src.nkilib.experimental.topk.chunked_topk import (
    _HW_SORT_AVAILABLE,
    TAP_NAMES,
    _chunk_legal,
    _slab_sizes_match,
    _validate_chunked_topk,
    chunked_topk,
    create_chunked_topk_config,
    pick_chunks,
)
from nkilib_src.nkilib.experimental.topk.chunked_topk_torch import (
    MAPPING_BUGS,
    chunked_topk_ref_chunkwise,
    chunked_topk_torch_ref,
)
from typing_extensions import override

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    InferenceArgs,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# GPT-OSS-120B at LNC=1: vocab 201,088 over 128 ranks -> per-rank shard 1,571, and the
# full vocab is what this kernel reduces on ONE core.
_GPTOSS_VOCAB = 201088
_TOPK_N_MAX = 65536
_PARTS_PER_GROUP = 16
_GROUPS_PER_TILE = 8


def _np_dtype(dtype):
    return ml_dtypes.bfloat16 if dtype == nl.bfloat16 else dtype


# nl dtypes type as plain strings, and the config factories annotate inp_dtype as
# np.dtype (the convention across the topk kernels). Parametrised callers pass opaque
# variables, so only LITERAL call sites trip the type checker in the release build;
# route them through this Any-typed constant.
_BF16: Any = nl.bfloat16


# ======================================================================================
# DEBUG TAPS (NKILIB_TOPK_TAPS=1): run every chunked hardware case with the kernel's six
# diagnostic taps appended and walk the index chain IN THE VALIDATOR, so a wrong index is
# localised to one link in the same 2-minute run that detects it. The chain needs no
# oracle -- consecutive taps are related by closed forms -- and mirrors podbundle/taps.py,
# which documents the method. The two ENDS of the chain stay with the ordinary gates.
# ======================================================================================
_TAPS_ON = os.environ.get("NKILIB_TOPK_TAPS") == "1"


def _tap_shapes(config) -> dict[str, tuple]:
    """Shapes of the six taps, in the kernel's append order (TAP_NAMES)."""
    bxs, p, k = config.BxS, config.chunks, config.k
    k_pad = -(-k // _PARTS_PER_GROUP) * _PARTS_PER_GROUP
    return {
        "s1_pos": (bxs * p, k),
        "s1_gid": (bxs * p, k),
        "s1_late": (bxs * p, k_pad),
        "cand_i": (bxs, p * k),
        "r_pos": (bxs, k),
        "gathered": (bxs, k),
        "s1_val": (bxs * p, k),
    }


def _torch_ref_maybe_taps(inp: "torch.Tensor", config) -> dict:
    """chunked_topk_torch_ref plus ZERO goldens for the taps when they are enabled.

    The framework requires torch_ref keys == output-descriptor keys; the tap validators
    never read their golden (the chain is checked against the taps' own neighbours), so
    zeros are the honest placeholder.
    """
    ref = chunked_topk_torch_ref(inp, config)
    if getattr(config, "debug_taps", False):
        for name, shape in _tap_shapes(config).items():
            ref[name] = torch.zeros(shape, dtype=torch.float32)
    return ref


def _walk_tap_chain(seen: dict, vocab: int, p: int, k: int, inp=None) -> tuple[str, str] | None:
    """First broken link of the index chain, or None. Ported from podbundle/taps.py.

    L1 s1_pos==s1_late (either first-read convention), L2 the snake->vocab remap (either
    offset convention -- which one holds identifies the stage-1 path), L3 concat+offset,
    L4 the lookup, L5 the narrow+store. Exact comparisons throughout. With ``inp`` (skipped
    for non-orderable input), the VALUE side is checked too: L6a pairs every intermediate
    candidate (value, id) against the caller's own input row, and L6b requires each chunk's
    candidate values to be exactly the top-k multiset of that chunk.
    """
    width = vocab // p
    n_cols = -(-width // _PARTS_PER_GROUP)
    red_w = p * k
    ids = seen["topk_indices"].astype(np.int64)
    bxs = ids.shape[0]
    s1_pos = seen["s1_pos"].reshape(bxs * p, k)
    s1_gid = seen["s1_gid"].astype(np.int64).reshape(bxs * p, k)
    late_full = seen["s1_late"].reshape(bxs * p, -1)
    cand_i = seen["cand_i"].reshape(bxs, red_w)
    r_pos = seen["r_pos"].reshape(bxs, k)
    gathered = seen["gathered"].reshape(bxs, k)

    # L1: two reads of one address must agree; which first-read tap matches identifies the
    # stage-1 path (raw snake positions vs already-global ids).
    #
    # The raw de-snake buffer's layout is flat(p, c) = p*k_cols + c with column c valid only
    # for p < min(16, k - 16c). When k % 16 == 0 (or k fits one column) that flat order IS
    # the compacted phase-2 order and the comparison is exact and elementwise. For other k
    # the k valid slots are SCATTERED among k_pad columns -- slicing [:, :k] reads unwritten
    # garbage columns and a naive elementwise gate deterministically reports a fake L1 race
    # on a correct kernel -- so the gate compares the sorted VALID multiset instead, which
    # still catches a real store/reload race (a stale read changes the multiset).
    if k % _PARTS_PER_GROUP == 0 or k <= _PARTS_PER_GROUP:
        late = late_full[:, :k]
        if not (np.array_equal(s1_pos, late) or np.array_equal(s1_gid.astype(np.float32), late)):
            n = int(min((s1_pos != late).sum(), (s1_gid.astype(np.float32) != late).sum()))
            return "L1 desnake store/reload", (
                f"{n} slots differ from both first-read taps; zeros pos/gid/late "
                f"{int((s1_pos == 0).sum())}/{int((s1_gid == 0).sum())}/{int((late == 0).sum())}"
            )
    else:
        k_cols_pad = -(-k // _PARTS_PER_GROUP)
        valid = np.zeros(late_full.shape[1], dtype=bool)
        for c in range(k_cols_pad):
            for pp in range(min(_PARTS_PER_GROUP, k - c * _PARTS_PER_GROUP)):
                valid[pp * k_cols_pad + c] = True
        assert int(valid.sum()) == k, (int(valid.sum()), k)
        lv = np.sort(late_full[:, valid], axis=1)
        if not (
            np.array_equal(np.sort(s1_pos, axis=1), lv)
            or np.array_equal(np.sort(s1_gid.astype(np.float32), axis=1), lv)
        ):
            return "L1 desnake store/reload", (
                f"valid-slot multiset differs from both first-read taps at k={k} "
                f"(k % 16 != 0: scattered de-snake layout, multiset form of the gate)"
            )

    # L2: the remap, exact, under either offset convention.
    s = s1_pos.astype(np.int64)
    base = np.minimum((s % _PARTS_PER_GROUP) * n_cols + (s // _PARTS_PER_GROUP), width - 1)
    chunk_of_row = (np.arange(bxs * p) % p).reshape(-1, 1)
    # The bypass path folds the offset in BEFORE its clamp, and clamps to the full vocab.
    want_global = np.minimum(
        (chunk_of_row * _PARTS_PER_GROUP + (s % _PARTS_PER_GROUP)) * n_cols + (s // _PARTS_PER_GROUP), vocab - 1
    )
    if np.array_equal(s1_gid, base):
        offsets_in_stage1 = False
    elif np.array_equal(s1_gid, want_global):
        offsets_in_stage1 = True
    else:
        n = int(min((s1_gid != base).sum(), (s1_gid != want_global).sum()))
        return "L2 snake->vocab remap", (
            f"{n} ids match neither offset convention, {int((s1_gid == 0).sum())} exactly 0"
        )

    # L3: stage-1 row j*P + c -> columns [c*k, (c+1)*k) of row j, plus c*width if stage 1
    # did not already fold it in.
    want = np.empty((bxs, red_w), dtype=np.int64)
    for j in range(bxs):
        for c in range(p):
            off = 0 if offsets_in_stage1 else c * width
            want[j, c * k : (c + 1) * k] = s1_gid[j * p + c] + off
    if not np.array_equal(cand_i.astype(np.int64), want):
        n = int((cand_i.astype(np.int64) != want).sum())
        return "L3 concat + chunk offset", f"{n} candidates != s1_gid{'' if offsets_in_stage1 else ' + c*width'}"

    # L4: a pure lookup. On failure, report enough to see the MECHANISM: which output slots,
    # which gather window owns each (the gather is windowed by 512), and whether the wrong
    # value is a stale zero, another window's result for the same local offset, or noise.
    pos = np.clip(r_pos.astype(np.int64), 0, red_w - 1)
    want4 = np.take_along_axis(cand_i, pos, axis=1)
    if not np.array_equal(gathered, want4):
        bad = gathered != want4
        n = int(bad.sum())
        rows_bad, cols_bad = np.nonzero(bad)
        lines = []
        for r, c in list(zip(rows_bad, cols_bad, strict=True))[:12]:
            p_rc = int(pos[r, c])
            win = p_rc // 512
            g = float(gathered[r, c])
            # Does the wrong value match what ANOTHER window would return for this slot
            # (same local offset within its 512-wide window, or the clamped edge)?
            aliases = [
                w
                for w in range(-(-red_w // 512))
                if w != win and float(cand_i[r, min(w * 512 + min(max(p_rc - w * 512, 0), 511), red_w - 1)]) == g
            ]
            lines.append(f"[{r},{c}] pos={p_rc} win={win} got={g} want={float(want4[r, c])} alias_wins={aliases}")
        return (
            "L4 windowed gather",
            f"{n} slots != cand_i[r_pos]; slot zeros {int((gathered[bad] == 0).sum())}; " + "; ".join(lines),
        )

    # L5.
    if not np.array_equal(ids, gathered.astype(np.int64)):
        n = int((ids != gathered.astype(np.int64)).sum())
        return "L5 narrow + store", f"{n} returned ids != gathered; id zeros {int((ids == 0).sum())}"

    # L6: the VALUE side of stage 1, against the input itself. Exact comparisons are valid
    # because the values pass through unchanged as bf16 and both sides are widened the same way.
    if inp is not None and "s1_val" in seen:
        s1_val = seen["s1_val"].reshape(bxs * p, k)
        gid_global = s1_gid if offsets_in_stage1 else s1_gid + chunk_of_row * width
        looked_up = np.take_along_axis(np.asarray(inp, dtype=np.float32)[np.arange(bxs * p) // p], gid_global, axis=1)
        if not np.array_equal(s1_val, looked_up):
            n = int((s1_val != looked_up).sum())
            return "L6a stage-1 value<->id pairing", f"{n} candidate slots where inp[gid] != value"
        for r in range(bxs * p):
            j, c = r // p, r % p
            chunk = np.asarray(inp, dtype=np.float32)[j, c * width : (c + 1) * width]
            want = np.sort(chunk)[-k:]
            if not np.array_equal(np.sort(s1_val[r]), want):
                return "L6b stage-1 value multiset", f"stage-1 row {r} (token {j}, chunk {c}) is not the chunk's top-k"
    return None


def _make_input(bxs: int, vocab: int, chunks: int, mode: str, dtype, seed: int = 1234) -> npt.NDArray[Any]:
    """Deterministic inputs. Modes marked (structure) target the chunk geometry itself."""
    rng = np.random.default_rng(seed)
    if mode == "allequal":
        # Every element identical -> the whole row is one tie group and nothing about
        # which index wins is determined. Exactly where a tie-handling bug yields a
        # plausible-but-wrong answer.
        x = np.full((bxs, vocab), 0.125, dtype=np.float32)
    elif mode == "allzero":
        x = np.zeros((bxs, vocab), dtype=np.float32)
    elif mode == "bigmag":
        x = (rng.standard_normal((bxs, vocab)) * 30.0).astype(np.float32)
    elif mode == "tinymag":
        x = (rng.standard_normal((bxs, vocab)) * 1e-6).astype(np.float32)
    elif mode == "chunk_edges":
        # (structure) Winners placed at the FIRST and LAST element of every chunk -- the
        # two positions an off-by-one in c*(V/P) moves onto the wrong side of a boundary.
        x = (rng.standard_normal((bxs, vocab)) * 0.1).astype(np.float32)
        width = vocab // chunks
        for c in range(chunks):
            x[:, c * width] = 50.0 + c
            x[:, (c + 1) * width - 1] = 40.0 + c
    elif mode == "last_chunk_hot":
        # (structure) The entire top-k lives in the HIGHEST chunk, so every returned index
        # needs the largest offset. A dropped offset is maximally wrong here.
        x = (rng.standard_normal((bxs, vocab)) * 0.1).astype(np.float32)
        width = vocab // chunks
        x[:, (chunks - 1) * width :] += 100.0
    elif mode == "uninit":
        # Arbitrary 16-bit patterns read as bf16: ~0.4% NaN plus both infinities and
        # saturated finites -- what an UNINITIALISED buffer holds. Non-orderable values
        # break the max8 / nc_match_replace8 / nc_n_gather chain, so an unmatchable value
        # leaves its reported position undefined and that position is used as a gather
        # index. There is no well-defined "k largest" here, so only the bounds, distinctness
        # and pairing gates apply.
        raw = np.random.default_rng(7).integers(0, 1 << 16, size=(bxs, vocab), dtype=np.uint16)
        return raw.view(ml_dtypes.bfloat16).astype(_np_dtype(dtype))
    elif mode == "unique_per_chunk":
        # (structure) Exactly ONE winner per chunk, at a non-boundary offset, with values
        # 2^(c+1) so every one is exactly representable in bf16 and no two can tie. That
        # makes the top-P slots fully DETERMINED, which lets the test assert the returned
        # global index exactly rather than only via the pairing check. Background is 0.
        x = np.zeros((bxs, vocab), dtype=np.float32)
        width = vocab // chunks
        for c in range(chunks):
            x[:, c * width + _winner_offset(c, width)] = float(2 ** (c + 1))
    else:  # "randn"
        x = (rng.standard_normal((bxs, vocab)) * 0.1).astype(np.float32)
    return x.astype(_np_dtype(dtype))


def _winner_offset(chunk: int, width: int) -> int:
    """Non-boundary, chunk-dependent position for the ``unique_per_chunk`` winner."""
    return (chunk * 37 + 5) % width


def _expected_unique_prefix(vocab: int, chunks: int) -> npt.NDArray[Any]:
    """Global ids of the ``unique_per_chunk`` winners, in descending-value order.

    Values are 2^(c+1), so descending value means descending chunk index.
    """
    width = vocab // chunks
    return np.array([c * width + _winner_offset(c, width) for c in range(chunks - 1, -1, -1)], dtype=np.int64)


# ======================================================================================
# THE GATES. Module-level and shared on purpose: the hardware validators below call these,
# and so does TestChunkedTopkMappingGates, which feeds them deliberately mis-mapped output
# from the numpy model. So proving a gate has teeth proves it for the code that runs on
# hardware, not for a parallel copy of it. Each returns (passed, detail-on-failure).
# ======================================================================================


def gate_value_set(got: npt.NDArray[Any], golden: npt.NDArray[Any]) -> tuple[bool, str]:
    """The k largest VALUES, as a multiset. Internal order is forgiven here."""
    if np.allclose(np.sort(got, axis=-1), np.sort(golden, axis=-1), rtol=1e-3, atol=1e-5):
        return True, ""
    row = int(np.argmax(np.abs(np.sort(got, -1) - np.sort(golden, -1)).max(-1)))
    return False, (
        f"value set differs. Worst row {row}: got={np.sort(got[row])[-6:].tolist()} "
        f"golden={np.sort(golden[row])[-6:].tolist()}"
    )


def gate_descending(got: npt.NDArray[Any]) -> tuple[bool, str]:
    """Descending order, which the sampler's top-p cumsum depends on.

    Equal neighbours pass explicitly: ``np.diff`` of two equal ``-inf`` slots is NaN
    (``inf - inf``), and ``-inf`` values are in-contract (GEMM overflow saturation), so the
    plain diff test would fail a CORRECT sorted row. Same patch as the megakernel suite's
    order check.
    """
    ok = (np.diff(got, axis=-1) <= 1e-5) | (got[..., 1:] == got[..., :-1])
    if bool(ok.all()):
        return True, ""
    r = int(np.argmax((~ok).sum(-1)))
    return False, f"row {r} not descending: {got[r][:8].tolist()}"


def gate_index_range(idx: npt.NDArray[Any], vocab: int, width: int) -> tuple[bool, str]:
    """Indices inside [0, vocab). An over-large chunk offset lands here."""
    bad = (idx < 0) | (idx >= vocab)
    if not bad.any():
        return True, ""
    return False, (
        f"{int(np.count_nonzero(bad))} indices outside [0, {vocab}), max={int(idx.max())}. "
        f"An offset one chunk too large (a multiple of {width}) lands here."
    )


def gate_distinct(idx: npt.NDArray[Any], k: int) -> tuple[bool, str]:
    """No repeated vocab id: the chunks are DISJOINT, so a repeat means offsets collapsed."""
    for r in range(idx.shape[0]):
        if len(np.unique(idx[r])) != k:
            dup = [int(x) for x in np.unique(idx[r]) if int((idx[r] == x).sum()) > 1][:6]
            return False, f"row {r} repeats vocab ids {dup}; chunk offsets collapsed"
    return True, ""


def gate_pairing(inp: npt.NDArray[Any], idx: npt.NDArray[Any], vals: npt.NDArray[Any], width: int) -> tuple[bool, str]:
    """inp[i, idx[i, j]] == val[i, j]. THE gate -- see the module docstring."""
    gathered = np.take_along_axis(inp, idx.astype(np.int64), axis=-1)
    bad = ~np.isclose(gathered, vals, rtol=1e-3, atol=1e-5)
    if not bad.any():
        return True, ""
    row = int(np.argmax(np.count_nonzero(bad, axis=-1)))
    cols = np.flatnonzero(bad[row])[:8].tolist()
    return False, (
        f"{int(np.count_nonzero(bad))} slots where inp[i, idx] != val. Worst row {row}, "
        f"slots {cols}: idx={idx[row, cols].tolist()} "
        f"chunk={[int(i) // width for i in idx[row, cols]]} "
        f"val={vals[row, cols].tolist()} inp@idx={gathered[row, cols].tolist()}"
    )


def gate_exact_prefix(idx: npt.NDArray[Any], expected: npt.NDArray[Any], width: int) -> tuple[bool, str]:
    """The leading indices, for inputs where the top slots are fully determined."""
    n = len(expected)
    for r in range(idx.shape[0]):
        if not np.array_equal(idx[r, :n].astype(np.int64), expected):
            return False, (
                f"row {r} leading {n} indices {idx[r, :n].tolist()} != {expected.tolist()}; "
                f"a wrong chunk offset shows up as a wrong multiple of {width}"
            )
    return True, ""


def run_all_gates(inp, idx, vals, golden, vocab, width, k):
    """Every gate that applies without knowing the input distribution. Returns {name: ok}."""
    in_range = gate_index_range(idx, vocab, width)[0]
    return {
        "value_set": gate_value_set(vals, golden)[0],
        "index_range": in_range,
        "distinct": gate_distinct(idx, k)[0],
        # Pairing needs in-range indices to gather at all; report it as failed when the
        # range gate already fired rather than raising IndexError.
        "pairing": gate_pairing(inp, idx, vals, width)[0] if in_range else False,
    }


class TestChunkedTopkGeometry:
    """Hardware-free gates on the chunking arithmetic. These run in seconds."""

    @pytest.mark.parametrize(
        "bxs,expected_p",
        [
            (1, 8),  # gbs=128: P=4 would leave 4 of 8 tile rows as padding
            (4, 4),  # gbs=512: both fill a tile, so the smaller reduce wins
            (8, 4),  # gbs=1024
            (16, 4),
        ],
    )
    def test_pick_chunks_production(self, bxs: int, expected_p: int):
        """P is the smallest legal count that fills a whole 128-partition tile."""
        assert pick_chunks(bxs, _GPTOSS_VOCAB, 256) == expected_p

    def test_pick_chunks_no_chunking_when_it_fits(self):
        """A vocab already inside the instruction bound must not be split."""
        assert pick_chunks(1, 3168, 256) == 1
        assert pick_chunks(32, _TOPK_N_MAX - 16, 256) == 1

    @pytest.mark.parametrize("bxs", [1, 4, 8, 16])
    def test_chunk_geometry_invariants(self, bxs: int):
        """Every constraint the two nisa.topk calls impose, at the picked P."""
        k = 256
        p = pick_chunks(bxs, _GPTOSS_VOCAB, k)
        width = _GPTOSS_VOCAB // p
        assert _GPTOSS_VOCAB % p == 0, "chunks must divide the vocab exactly"
        # Stage 1 faces the chunk width.
        assert 8 <= width < _TOPK_N_MAX, f"chunk width {width} outside the instruction bound"
        assert k <= width
        # % 16 == 0 keeps gpsimd_topk on its coalesced prefetch path (has_pad False).
        assert width % _PARTS_PER_GROUP == 0, f"chunk width {width} loses the coalesced load"
        # Filling whole tiles keeps it on fast_dma_safe.
        assert (bxs * p) % _GROUPS_PER_TILE == 0, f"stage-1 rows {bxs * p} leave a partial tile"
        # The reduce faces P*k, a second independent bound.
        assert 8 <= p * k < _TOPK_N_MAX

    def test_stage1_cost_is_independent_of_p(self):
        """The closed form behind CHOOSING P: ceil(rows/8)*ceil(width/16) == R*V/128.

        If this ever stops holding, the rationale for picking the smallest tile-filling P
        is gone and the choice must be re-derived rather than trusted.
        """
        for bxs in (1, 4, 8):
            costs = {}
            for p in (4, 8, 16, 32):
                width = _GPTOSS_VOCAB // p
                rows = bxs * p
                if rows % _GROUPS_PER_TILE or width >= _TOPK_N_MAX:
                    continue
                costs[p] = (rows // _GROUPS_PER_TILE) * -(-width // _PARTS_PER_GROUP)
            # Widths that are not multiples of 16 round n_cols up, so allow a 1% drift.
            base = bxs * _GPTOSS_VOCAB // 128
            for p, c in costs.items():
                assert abs(c - base) / base < 0.01, f"P={p} stage-1 cost {c} != {base}"

    @pytest.mark.parametrize("chunks", [4, 8])
    def test_reshape_identities_are_the_chunking(self, chunks: int):
        """Both reshapes the kernel relies on are pure views, so chunking costs nothing.

        (a) inp.reshape(BxS*P, V/P) row j*P+c IS chunk c of token j.
        (b) stage1_out.reshape(BxS, P*k) puts chunk c at columns [c*k, (c+1)*k).
        """
        bxs, vocab, k = 3, 96, 5
        inp = np.arange(bxs * vocab).reshape(bxs, vocab)
        width = vocab // chunks
        view = inp.reshape(bxs * chunks, width)
        for j in range(bxs):
            for c in range(chunks):
                assert np.array_equal(view[j * chunks + c], inp[j, c * width : (c + 1) * width])
        s1 = np.arange(bxs * chunks * k).reshape(bxs * chunks, k)
        cand = s1.reshape(bxs, chunks * k)
        for j in range(bxs):
            for c in range(chunks):
                assert np.array_equal(cand[j, c * k : (c + 1) * k], s1[j * chunks + c])

    @pytest.mark.parametrize("mode", ["randn", "chunk_edges", "last_chunk_hot", "allequal"])
    def test_chunkwise_equals_global_topk(self, mode: str):
        """The algorithm itself: P disjoint chunk top-k, reduced, == one global top-k."""
        bxs, vocab, k, chunks = 4, 2048, 16, 8
        x = _make_input(bxs, vocab, chunks, mode, np.float32).astype(np.float32)
        v, i = chunked_topk_ref_chunkwise(x, k, chunks)
        order = np.argsort(-x, axis=-1, kind="stable")[:, :k]
        gv = np.take_along_axis(x, order, axis=-1)
        assert np.allclose(np.sort(v, axis=-1), np.sort(gv, axis=-1)), "value set differs from global top-k"
        assert np.allclose(np.take_along_axis(x, i, axis=-1), v), "value<->index pairing broken"


class TestChunkedTopkMappingGates:
    """NEGATIVE CONTROLS: every index-mapping bug must be caught by at least one gate.

    A gate that cannot fail is not a gate. These feed deliberately mis-mapped output --
    the numpy model with a named bug injected -- through the SAME gate functions the
    hardware validators call, so proving teeth here proves it for what runs on hardware.

    The reason the suite needs them: in every index-only bug below the returned VALUES stay
    exactly the true global top-k, so a value comparison passes and most keep indices in
    range too. Only the pairing gate is sensitive to all of them. Hardware-free by nature
    (they mutate a model, not the kernel); the gates they validate are then applied to the
    real kernel on trn3/trn2 by TestChunkedTopkKernel.
    """

    _BXS, _VOCAB, _K, _CHUNKS = 4, 2048, 16, 8
    #: Bugs that corrupt ONLY the indices, leaving the value set correct.
    _INDEX_ONLY = ("no_offset", "off_by_one", "reversed", "scrambled")

    def _fixture(self):
        rng = np.random.default_rng(99)
        x = (rng.standard_normal((self._BXS, self._VOCAB)) * 0.1).astype(np.float32)
        order = np.argsort(-x, axis=-1, kind="stable")[:, : self._K]
        return x, np.take_along_axis(x, order, axis=-1), self._VOCAB // self._CHUNKS

    def _gates(self, bug):
        x, golden, width = self._fixture()
        vals, idx = chunked_topk_ref_chunkwise(x, self._K, self._CHUNKS, bug=bug)
        return run_all_gates(x, idx, vals, golden, self._VOCAB, width, self._K)

    def test_correct_mapping_passes_every_gate(self):
        """The mirror the negative controls need: correct code must be silent everywhere.

        Without this, a gate that rejected everything would still satisfy the tests below.
        """
        res = self._gates("none")
        assert all(res.values()), f"correct mapping must pass every gate, got {res}"

    @pytest.mark.parametrize("bug", MAPPING_BUGS)
    def test_pairing_gate_catches_every_mapping_bug(self, bug):
        """The load-bearing assertion of the whole suite."""
        res = self._gates(bug)
        assert not res["pairing"], f"bug '{bug}' slipped past the pairing gate: {res}"

    @pytest.mark.parametrize("bug", _INDEX_ONLY)
    def test_value_set_gate_is_blind_to_index_only_bugs(self, bug):
        """Why pairing is load-bearing rather than a nice-to-have.

        These bugs return the CORRECT k largest values with lying indices, so a value-set
        comparison -- the obvious thing to test -- sees nothing wrong. A suite built on it
        would ship a kernel that returns wrong token ids.
        """
        assert self._gates(bug)["value_set"], f"expected '{bug}' to leave the value set intact"

    def test_wrong_row_also_moves_values(self):
        """The one bug that is NOT index-only, recorded so the split above stays honest."""
        assert not self._gates("wrong_row")["value_set"]

    def test_descending_gate_tolerates_equal_neg_inf_neighbours(self):
        """np.diff of two adjacent -inf slots is NaN (inf - inf), and NaN <= tol is False,
        so the plain diff test failed a CORRECT sorted row whose top-k holds >= 2 -inf --
        an in-contract shape (GEMM overflow saturation; the tail suite's inf_topk mode
        produces exactly it). The explicit equal-neighbour patch must pass it while a
        genuinely ascending pair, -inf runs included, still fails."""
        ninf = float("-inf")
        ok, _ = gate_descending(np.array([[2.0, 1.0, ninf, ninf]], dtype=np.float32))
        assert ok, "correct descending row with an -inf tie must pass"
        ok, _ = gate_descending(np.array([[1.0, 2.0, ninf, ninf]], dtype=np.float32))
        assert not ok, "ascending finite pair must still fail"
        ok, _ = gate_descending(np.array([[ninf, ninf, 2.0, 1.0]], dtype=np.float32))
        assert not ok, "finite values after -inf must still fail"

    def test_off_by_one_chunk_escapes_the_vocab(self):
        """An offset one chunk too large indexes past the vocab, so the range gate sees it."""
        assert not self._gates("off_by_one")["index_range"]


@pytest_test_metadata(name="Chunked TopK")
@pytest_marks(["topk", "gpsimd", "chunked"])
@final
class TestChunkedTopkKernel:
    """Hardware gates. LNC=1 only -- see the LNC note in chunked_topk's docstring."""

    @staticmethod
    def output_tensors(kernel_input):
        """Output shapes/dtypes, derived from the kernel input the framework will pass.

        Order matters: the framework matches outputs POSITIONALLY, and the kernel appends
        the taps AFTER its two products, so the tap entries must come last and in TAP_NAMES
        order.
        """
        config = kernel_input["config"]
        # The row count comes from the CONFIG, not the physical input tensor: on the
        # interleaved cases the input is the scrambled [BxS*(vocab/w), w] receive layout
        # (or a slab LIST of it), whose leading dim is NOT the output row count.
        rows = config.BxS
        out = {
            "topk_values": np.zeros((rows, config.k), dtype=_np_dtype(config.inp_dtype)),
            "topk_indices": np.zeros((rows, config.k), dtype=np.uint32),
        }
        if getattr(config, "debug_taps", False):
            shapes = _tap_shapes(config)
            for name in TAP_NAMES:
                out[name] = np.zeros(shapes[name], dtype=np.float32)
        return out

    def run_case(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bxs: int,
        vocab: int,
        k: int,
        chunks: int,
        dtype,
        sorted: bool = True,
        mode: str = "randn",
        expect_prefix: npt.NDArray[Any] | None = None,
        index_bounds_only: bool = False,
        seed: int = 1234,
        force_taps: bool = False,
        determinism: bool = False,
        skip_nan_fold: bool = False,
        interleaved_peer_width: int = 0,
    ):
        """One chunked_topk case through UnitTestFramework at LNC=1.

        ``expect_prefix`` asserts the leading returned indices EXACTLY (used with the
        ``unique_per_chunk`` mode, where the top slots are fully determined).
        ``index_bounds_only`` drops the value-set and exactness checks, for inputs like NaN
        where "the k largest" has no well-defined answer; the range, distinctness and
        pairing gates still run.
        """

        # chunked_topk is hardware-sort ONLY: a sorted request without the capability is
        # refused at trace time (the software sort was deleted), so those cases SKIP loudly
        # here rather than fail. They regain coverage the moment the version set carries the
        # patched nki + neuronx-cc >= 2.0.280935.
        if sorted and not _HW_SORT_AVAILABLE:
            pytest.skip("hardware sort unavailable on this toolchain; chunked_topk refuses sorted=True")

        # Taps only exist on the chunked path (the p == 1 short-circuit asserts against them),
        # so probe the chunk count the config will pick before deciding to enable them.
        taps = (_TAPS_ON or force_taps) and (chunks if chunks else pick_chunks(bxs, vocab, k)) > 1

        def input_generator(test_config, input_tensor_def=None):
            config = create_chunked_topk_config(
                inp_shape=(bxs, vocab),
                inp_dtype=dtype,
                k=k,
                chunks=chunks,
                sorted=sorted,
                debug_taps=taps,
                skip_nan_fold=skip_nan_fold,
                interleaved_peer_width=interleaved_peer_width,
            )
            arr = _make_input(bxs, vocab, config.chunks, mode, dtype, seed)
            if interleaved_peer_width:
                # Scramble the LOGICAL [bxs, vocab] into the all-to-all RECEIVE layout the
                # kernel is told to read: row p*bxs + j holds token j's columns
                # [p*w, (p+1)*w). The oracle below stays on the logical array, so every gate
                # checks that the kernel undid the interleave exactly.
                w = interleaved_peer_width
                arr = np.ascontiguousarray(
                    arr.reshape(bxs, vocab // w, w).transpose(1, 0, 2).reshape(bxs * (vocab // w), w)
                )
            return {"inp": arr, "config": config}

        inputs = input_generator(test_config=None)
        inp_f32 = np.asarray(inputs["inp"], dtype=np.float32)
        if interleaved_peer_width:
            w = interleaved_peer_width
            inp_f32 = np.ascontiguousarray(inp_f32.reshape(vocab // w, bxs, w).transpose(1, 0, 2).reshape(bxs, vocab))
        n_chunks = inputs["config"].chunks

        def comparator(golden_dict, output_tensors):
            golden_values = np.asarray(golden_dict["topk_values"], dtype=np.float32)
            width = vocab // n_chunks
            # Shared between the two validators: pairing needs both tensors, so it runs from
            # whichever fills this last.
            seen: dict[str, npt.NDArray[Any]] = {}

            def report(v: CustomValidator, name: str, result: tuple[bool, str]) -> bool:
                if not result[0]:
                    v._print_with_log(f"FAIL {name}: {result[1]}")
                return result[0]

            def pairing_when_both_present(v: CustomValidator) -> bool:
                if "topk_values" not in seen or "topk_indices" not in seen:
                    return True  # the other validator has not run yet; it will re-check
                idx = seen["topk_indices"]
                if not gate_index_range(idx, vocab, width)[0]:
                    return True  # the range gate already reported this
                return report(v, "pairing", gate_pairing(inp_f32, idx, seen["topk_values"], width))

            def chain_when_complete(v: CustomValidator) -> bool:
                """The tap chain walk, from whichever validator fills `seen` last."""
                if not taps or not ({"topk_indices", *TAP_NAMES} <= seen.keys()):
                    return True
                broke = _walk_tap_chain(seen, vocab, n_chunks, k, inp=None if index_bounds_only else inp_f32)
                if broke is not None:
                    v._print_with_log(f"FAIL tap chain at {broke[0]}: {broke[1]}")
                    return False
                v._print_with_log("tap chain: every link consistent")
                return True

            class ValuesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    got = np.asarray(
                        np.frombuffer(inference_output, dtype=_np_dtype(dtype)).reshape(bxs, k),
                        dtype=np.float32,
                    )
                    seen["topk_values"] = got
                    if index_bounds_only:
                        # No well-defined "k largest" (e.g. NaN input): only pairing holds.
                        return pairing_when_both_present(self)
                    ok = report(self, "value_set", gate_value_set(got, golden_values))
                    if ok and sorted:
                        ok = report(self, "descending", gate_descending(got))
                    return ok and pairing_when_both_present(self)

            class IndicesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    idx = np.frombuffer(inference_output, dtype=np.uint32).reshape(bxs, k)
                    seen["topk_indices"] = idx
                    ok = report(self, "index_range", gate_index_range(idx, vocab, width))
                    ok = report(self, "distinct", gate_distinct(idx, k)) and ok
                    if expect_prefix is not None:
                        ok = report(self, "exact_index", gate_exact_prefix(idx, expect_prefix, width)) and ok
                    return ok and pairing_when_both_present(self) and chain_when_complete(self)

            validators = {
                "topk_values": CustomValidatorWithOutputTensorData(
                    validator=ValuesValidator, output_ndarray=output_tensors["topk_values"]
                ),
                "topk_indices": CustomValidatorWithOutputTensorData(
                    validator=IndicesValidator, output_ndarray=output_tensors["topk_indices"]
                ),
            }
            if taps:
                # One capturing validator per tap. Each returns the chain verdict once all
                # seven arrays are present, so the walk runs exactly once -- from whichever
                # validator the framework happens to call last -- and its verdict fails the
                # test rather than only logging.
                def _tap_validator(name: str):
                    class TapValidator(CustomValidator):
                        @override
                        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                            seen[name] = np.frombuffer(inference_output, dtype=np.float32).copy()
                            return chain_when_complete(self)

                    return TapValidator

                for name in TAP_NAMES:
                    validators[name] = CustomValidatorWithOutputTensorData(
                        validator=_tap_validator(name), output_ndarray=output_tensors[name]
                    )
            return validators

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=chunked_topk,
            torch_ref=torch_ref_wrapper(_torch_ref_maybe_taps),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            rtol=1e-3,
            atol=1e-5,
            custom_comparator=comparator,
            # Run-to-run bit-exactness doubles as a RACE DETECTOR: every ordering defect this
            # kernel has had was schedule-dependent, and a survivor would show up as two
            # executions of one NEFF disagreeing.
            inference_args=InferenceArgs(enable_determinism_check=True) if determinism else None,
        )

    # ---- cheap plumbing rungs: chunks forced so the chunked path runs on a small vocab --
    @pytest.mark.parametrize("mode", ["randn", "chunk_edges", "last_chunk_hot", "allequal", "allzero"])
    @pytest.mark.parametrize("bxs,vocab,k,chunks", [(8, 2048, 16, 4), (8, 4096, 32, 8)])
    def test_chunked_topk_small(self, test_manager, platform_target, bxs, vocab, k, chunks, mode):
        """Proves the two-stage machinery and the offset arithmetic cheaply.

        ``chunks`` is forced: these vocabs fit one nisa.topk call, so auto-selection would
        short-circuit to P=1 and never exercise the chunked path at all.
        """
        self.run_case(test_manager, platform_target, bxs, vocab, k, chunks, nl.bfloat16, mode=mode)

    # ---- ACCEPTANCE: the GPT-OSS production shapes, P auto-selected ----------------------
    @pytest.mark.parametrize("mode", ["randn", "chunk_edges", "last_chunk_hot"])
    @pytest.mark.parametrize("bxs", [1, 4, 8])
    def test_chunked_topk_gptoss(self, test_manager, platform_target, bxs, mode):
        """Full 201,088 vocab on one core -- the shape a collective-free tail would use.

        bxs = 1 / 4 / 8 are the per-rank row counts at gbs = 128 / 512 / 1024 with 128
        ranks. chunks=0 exercises pick_chunks (P=8 at bxs=1, P=4 above).
        """
        self.run_case(test_manager, platform_target, bxs, _GPTOSS_VOCAB, 256, 0, nl.bfloat16, mode=mode)

    @pytest.mark.parametrize("mode", ["randn", "last_chunk_hot"])
    def test_chunked_topk_p16_bypass(self, test_manager, platform_target, mode):
        """P = 16 on the stage-1 bypass path -- the chunk-base offset regression.

        pick_chunks MUST return 16 here (P = 8 gives width 98,304 >= the single-call
        ceiling), and the bypass preconditions all hold (k % 16 == 0, s1_rows % 8 == 0,
        width % 16 == 0), so stage 1 runs more chunks than one 128-partition tile holds.
        A hoisted chunk-base offset tile built for tile 0 and reused verbatim gave every
        row past tile 0 a global id short by 8 * chunk_width -- in range, values exactly
        right, silently wrong ids (found by review simulation; nothing else in the suite
        reaches P > 8). last_chunk_hot puts every winner in the highest chunk, which is
        the far end of the misaddressed region.
        """
        assert pick_chunks(1, 786432, 256) == 16
        self.run_case(test_manager, platform_target, 1, 786432, 256, 0, nl.bfloat16, mode=mode)

    # ---- The all-to-all receive layout, folded into stage 1's loads -----------------------
    @pytest.mark.parametrize(
        "bxs,vocab,k,chunks,w",
        [
            # The single-hop production shape: peer width V_s = 1571 divides the stage-1
            # snake row width 3,142 -- the exact geometry the tail folds its de-interleave
            # into. And a small legal shape that runs in simulation.
            (8, _GPTOSS_VOCAB, 256, 0, 1571),
            (8, 8192, 32, 4, 64),
        ],
    )
    def test_chunked_topk_interleaved_src(self, test_manager, platform_target, bxs, vocab, k, chunks, w):
        """``interleaved_peer_width``: the receive layout read in place, gates unchanged.

        The kernel input is the logical array SCRAMBLED into the all-to-all receive layout;
        the oracle and every gate (value multiset, pairing against the logical row, index
        range/distinctness) run on the logical layout -- so a wrong pattern that still loads
        plausible values cannot pass. Results must equal the contiguous load bitwise; only
        the DMA descriptors differ.
        """
        self.run_case(
            test_manager,
            platform_target,
            bxs,
            vocab,
            k,
            chunks,
            nl.bfloat16,
            interleaved_peer_width=w,
        )

    # ---- The finite-input contract: stage-1 NaN fold skipped ------------------------------
    @pytest.mark.parametrize(
        "bxs,vocab,k,chunks",
        [
            # The production shape (desnake-only bypass path) and a k % 16 != 0 shape that
            # forces the general _inline_topk stage 1, so BOTH fold sites' skips are gated.
            (4, _GPTOSS_VOCAB, 256, 0),
            (8, 2048, 24, 4),
        ],
    )
    def test_chunked_topk_skip_nan_fold(self, test_manager, platform_target, bxs, vocab, k, chunks):
        """``skip_nan_fold=True`` on finite input: same gates, same reference, fold deleted.

        The flag is a caller CONTRACT that the input has no NaN (the GPT-OSS tail's logits,
        for example, are its own GEMM output). On such input the fold is the identity, so
        every gate must hold exactly as with the fold present. NaN-input behavior of the
        fold itself keeps its coverage in the ``uninit`` bitpattern cases, which run with
        the default ``skip_nan_fold=False``.
        """
        self.run_case(test_manager, platform_target, bxs, vocab, k, chunks, nl.bfloat16, skip_nan_fold=True)

    # ---- INTERMEDIATE-TENSOR chain, as a DEFAULT test rather than an env-gated mode -------
    @pytest.mark.parametrize("bxs", [1, 8])
    def test_chunked_topk_taps_chain(self, test_manager, platform_target, bxs):
        """Every intermediate tensor of the index AND value chains, validated on hardware.

        Runs the production shapes with ``debug_taps=True`` unconditionally, so the L1..L6b
        chain walk (see ``_walk_tap_chain``) executes on every hardware run of this suite --
        not only when NKILIB_TOPK_TAPS=1 opts the whole suite in. The taps deliberately run a
        DIFFERENT schedule than the plain acceptance cases: both this kernel's real ordering
        defects were invisible in the default schedule and caught by the taps one.
        """
        self.run_case(test_manager, platform_target, bxs, _GPTOSS_VOCAB, 256, 0, nl.bfloat16, force_taps=True)

    # ---- run-to-run determinism: N executions of one NEFF must agree bit-exactly ----------
    @pytest.mark.parametrize("bxs", [1, 8])
    def test_chunked_topk_run_to_run_determinism(self, test_manager, platform_target, bxs):
        """The same NEFF, executed repeatedly, must return identical bytes every time.

        A surviving ordering defect is schedule- and traffic-dependent, so it shows up as two
        executions disagreeing long before it shows up in a single-run gate. This is the
        cheapest standing race detector the harness offers.
        """
        self.run_case(test_manager, platform_target, bxs, _GPTOSS_VOCAB, 256, 0, nl.bfloat16, determinism=True)

    @pytest.mark.fast
    def test_chunked_topk_p1_delegates(self, test_manager, platform_target):
        """A vocab inside the instruction bound must behave exactly like gpsimd_topk."""
        self.run_case(test_manager, platform_target, 8, 3168, 256, 1, nl.bfloat16)

    # ---- EXACT index gate: winners are uniquely valued, so top slots are determined -----
    @pytest.mark.parametrize(
        "bxs,vocab,k,chunks",
        [
            pytest.param(8, 2048, 16, 8, marks=pytest.mark.fast),
            (1, _GPTOSS_VOCAB, 256, 0),
            (8, _GPTOSS_VOCAB, 256, 0),
        ],
    )
    def test_chunked_topk_exact_index(self, test_manager, platform_target, bxs, vocab, k, chunks):
        """Assert the leading indices EXACTLY, not just via the pairing check.

        ``unique_per_chunk`` puts one winner per chunk at value 2^(c+1) -- distinct and
        exactly representable in bf16 -- so the top-P slots have one correct answer. This is
        the sharpest available gate on the ``c*(V/P)`` offset: a wrong multiple shows up
        directly, with no tie ambiguity to hide behind.
        """
        n_chunks = chunks or pick_chunks(bxs, vocab, k)
        self.run_case(
            test_manager,
            platform_target,
            bxs,
            vocab,
            k,
            chunks,
            nl.bfloat16,
            mode="unique_per_chunk",
            expect_prefix=_expected_unique_prefix(vocab, n_chunks),
        )

    # ---- distinctness at scale: many independent reduce tiles per run -------------------
    @pytest.mark.parametrize("seed", [11, 22, 33, 44])
    def test_chunked_topk_distinct_seed_sweep(self, test_manager, platform_target, seed):
        """bxs=128 at the production vocab: 16 reduce tiles per run, 4 seeds -> 64 blocks.

        Every acceptance case above runs ONE draw of ONE geometry, which gives almost no power
        against a defect that shows on a small FRACTION of reduce tiles -- and the tail's
        duplicate-id bug was exactly that shape (2 of 128 row blocks). This trades one cheap
        NEFF for 64 independent draws of the distinctness and pairing gates. It is deliberately
        the largest bxs that still auto-selects the production P=4 / 1,024-wide reduce.
        """
        self.run_case(test_manager, platform_target, 128, _GPTOSS_VOCAB, 256, 0, nl.bfloat16, seed=seed)

    # ---- k edge cases, including k % 16 != 0 which drives the compaction path -----------
    @pytest.mark.parametrize(
        "vocab,k,chunks",
        [
            (2048, 1, 8),  # k == 1: reduce width P*k == 8, the minimum legal
            (2048, 7, 8),  # k % 16 != 0 -> k_pad > k, exercises the unsorted compaction
            (2048, 256, 8),  # k == chunk width
            (64, 8, 8),  # chunk width == 8, the instruction minimum
        ],
    )
    def test_chunked_topk_k_edges(self, test_manager, platform_target, vocab, k, chunks):
        """Boundaries of k and of the chunk width, at the edges of the two topk bounds.

        ``k % 16 != 0`` matters here beyond the usual: stage 1 always runs unsorted, and the
        unsorted path compacts the valid de-snaked columns rather than sorting them, so a
        k that leaves padding slots exercises code the sorted path never reaches.
        """
        self.run_case(test_manager, platform_target, 8, vocab, k, chunks, nl.bfloat16)

    # ---- reduce row padding: BxS not a multiple of the 8-row tile -----------------------
    @pytest.mark.parametrize("bxs", [3, 9])
    def test_chunked_topk_partial_reduce_tile(self, test_manager, platform_target, bxs):
        """BxS % 8 != 0 forces the reduce onto the padded path (rows -> next multiple of 8).

        The pad rows carry the most-negative finite bf16 so they can never win a slot; this
        gates that they do not, and that only the real rows are returned.
        """
        self.run_case(test_manager, platform_target, bxs, 2048, 16, 8, nl.bfloat16)

    # ---- non-orderable input: bounds + pairing only -------------------------------------
    def test_chunked_topk_uninit(self, test_manager, platform_target):
        """NaN / inf / saturated patterns. No well-defined top-k, so no value-set check.

        What must still hold is what a consumer relies on: every index in range, no repeats,
        and the index at slot j pointing at the value at slot j.
        """
        self.run_case(
            test_manager,
            platform_target,
            8,
            2048,
            16,
            8,
            nl.bfloat16,
            mode="uninit",
            index_bounds_only=True,
        )


class TestChunkedTopkValidation:
    """Hardware-free: the guards reject shapes the two nisa.topk calls cannot serve."""

    def test_pick_chunks_raises_when_nothing_legal(self):
        """A prime vocab above the ceiling has no power-of-two P dividing it."""
        with pytest.raises(ValueError, match="no legal chunk count"):
            pick_chunks(1, 65537, 8)  # 65537 is prime and >= the bound

    @pytest.mark.parametrize(
        "vocab,k,chunks,expected",
        [
            # vocab % P != 0 -- the chunks would not tile the vocab.
            (2049, 256, 4, "must be divisible by chunks"),
            # Non-power-of-two P: divides the vocab and passes every width bound, but the
            # stage-1 bypass derives chunk offsets with a bitwise-AND mask that is a valid
            # modulus ONLY for power-of-two P -- chunks=3 returned exactly-right values with
            # silently wrong global ids before this guard existed.
            (48000, 16, 3, "must be a power of two"),
            # V/P == 100,544, past the instruction's own n < 65536 bound.
            (_GPTOSS_VOCAB, 256, 2, "chunk width V/P"),
            # V/P == 65,528: below the RAW bound, but the instruction receives the 16-PADDED
            # snake size 16*ceil(width/16) == 65,536 -- the widths 65,521..65,535 all fall in
            # this hole, at chunks == 1 (the delegate) and any P alike.
            (65528, 256, 1, "padded to a whole 16-column snake"),
            (131056, 256, 2, "padded to a whole 16-column snake"),
            # V/P == 16 but k == 256: stage 1 cannot return more than a chunk holds.
            (2048, 256, 128, "must be <= chunk width"),
            # P*k == 65,536 -- far past the FUSED-REDUCE bound (P*k <= 16,384: the one-hot
            # position lookup lays P*k across at most 128 chunks of 128 partitions).
            (131072, 1024, 64, r"reduce width P\*k"),
            # P*k == 32,768: passes the instruction's own n < 65,536 bound that used to be
            # the only check here, but the fused reduce would slice 256 partitions of
            # 128-partition tiles at trace time -- validator-legal was trace-illegal.
            (_GPTOSS_VOCAB, 4096, 8, "one-hot position lookup"),
            # P*k == 4, below the instruction's n >= 8 minimum.
            (2048, 1, 4, r"reduce width P\*k"),
        ],
    )
    def test_config_guards_reject_illegal_geometry(self, vocab, k, chunks, expected):
        """Each guard fires, and its message names the bound it protects.

        Both nisa.topk bounds apply TWICE -- once to the chunk width that stage 1 faces and
        once to P*k that the reduce faces -- and the two move in opposite directions as P
        changes. The cases below pin one violation each so a future change to pick_chunks
        cannot quietly widen the accepted envelope.
        """
        # match= is a REGEX, so metacharacters in the expected text must be escaped.
        # sorted=False: these pin the GEOMETRY asserts, and sorted=True would first hit the
        # hardware-sort availability refusal on a toolchain without the capability.
        cfg = create_chunked_topk_config(inp_shape=(8, vocab), inp_dtype=_BF16, k=k, chunks=chunks, sorted=False)
        with pytest.raises(AssertionError, match=expected):
            _validate_chunked_topk(cfg)

    @pytest.mark.parametrize(
        "bxs,vocab,k,expected_p",
        [
            # k=4096 at the GPT-OSS vocab: P=8 has P*k = 32,768 past the fused-reduce cap, so
            # the picker must fall to P=4 (16,384, exactly at the cap) instead of returning a
            # P the validator refuses.
            (1, _GPTOSS_VOCAB, 4096, 4),
            # Widths in the padded-snake hole: 65,528 does NOT fit one call (16-padded is
            # 65,536), so chunk; 131,056 / 2 = 65,528 is the same hole one level up.
            (8, 65528, 256, 2),
            (8, 131056, 256, 4),
            # The boundary that DOES fit one call.
            (8, 65520, 256, 1),
            # Production picks are unchanged.
            (1, _GPTOSS_VOCAB, 256, 8),
            (4, _GPTOSS_VOCAB, 256, 4),
        ],
    )
    def test_pick_chunks_uses_the_validator_envelope(self, bxs, vocab, k, expected_p):
        """pick_chunks ranks only P values the validator accepts (they share _chunk_legal).

        It used to carry the OLD envelope -- raw width, P*k < 65,536, and a
        'vocab < 65,536 -> 1' short-circuit -- so the default auto-pick could return a P the
        tightened validator then refused at trace, with another P legal.
        """
        assert pick_chunks(bxs, vocab, k) == expected_p

    @pytest.mark.parametrize("bxs", [1, 4, 8, 32])
    @pytest.mark.parametrize("k", [8, 16, 256, 4096])
    @pytest.mark.parametrize("vocab", [65520, 65528, 65536, 131056, _GPTOSS_VOCAB, 8 * 8191, 1 << 20])
    def test_pick_chunks_agrees_with_validator(self, bxs, vocab, k):
        """Whatever pick_chunks returns must trace: the validator accepts it, or pick_chunks
        itself refused. The two are the same predicate by construction; this pins it."""
        try:
            p = pick_chunks(bxs, vocab, k)
        except ValueError:
            return
        assert _chunk_legal(vocab // p, p, k)
        _validate_chunked_topk(
            create_chunked_topk_config(inp_shape=(bxs, vocab), inp_dtype=_BF16, k=k, chunks=p, sorted=False)
        )

    def test_slab_sizes_are_checked_per_slab(self):
        """A slab LIST is flattened per tensor (no reshape to fail), so sizes are checked
        explicitly: slabs whose sizes merely SUM to the total must be refused."""
        assert _slab_sizes_match([(4, 1024), (4, 1024)], 4 * 1024)
        assert not _slab_sizes_match([(3, 1024), (5, 1024)], 4 * 1024)
        assert not _slab_sizes_match([(4, 1024), (4, 1023)], 4 * 1024)

    def test_sorted_refused_without_hardware_sort(self, monkeypatch):
        """The hardware sort is the ONLY sort: sorted=True without the capability must be
        refused at trace time, never silently served by a fallback (the software sort was
        deleted). Both directions are pinned so the gate can neither rot into a no-op nor
        reject a capable toolchain."""
        import nkilib_src.nkilib.experimental.topk.chunked_topk as ct

        cfg = create_chunked_topk_config(inp_shape=(8, 2048), inp_dtype=_BF16, k=16, chunks=8, sorted=True)
        monkeypatch.setattr(ct, "_HW_SORT_AVAILABLE", False)
        with pytest.raises(AssertionError, match="software sort was removed"):
            _validate_chunked_topk(cfg)
        monkeypatch.setattr(ct, "_HW_SORT_AVAILABLE", True)
        _validate_chunked_topk(cfg)
        # And the ISA's own sorted-mode cap is enforced rather than falling back.
        big = create_chunked_topk_config(inp_shape=(8, 65024), inp_dtype=_BF16, k=16256, chunks=2, sorted=True)
        with pytest.raises(AssertionError, match="capped at"):
            _validate_chunked_topk(big)

    @pytest.mark.parametrize(
        "vocab,k,chunks",
        [
            (2048, 4, 2),
            (64, 8, 8),
            (_GPTOSS_VOCAB, 256, 8),
            # BOUNDARY legals, one per new guard: the largest width whose 16-padded snake
            # still fits (65,520), and P*k exactly at the fused-reduce cap (128*128).
            (65520, 256, 1),
            (131072, 256, 64),
        ],
    )
    def test_config_guards_accept_legal_geometry(self, vocab, k, chunks):
        """The mirror of the above: legal geometry must pass silently.

        Without this, a guard that rejected everything would still pass the negative cases.
        """
        cfg = create_chunked_topk_config(inp_shape=(8, vocab), inp_dtype=_BF16, k=k, chunks=chunks, sorted=False)
        _validate_chunked_topk(cfg)
