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

"""Tests for the single-collective GPT-OSS sampling tail.

WHAT THE GATES ARE FOR
----------------------
Three of this kernel's possible failures return plausible output:

* A wrong collective group ORDER still returns the correct top-k VALUES -- the value multiset
  over a row does not depend on which peer contributed which column -- but every returned
  vocab id is then wrong. Caught only by checking ``full_row[id] == value``.
* A wrong row-ownership assumption returns a full, well-formed token per row, every one in
  the wrong batch position. Caught by ``test_row_ownership_is_identity_and_tiles`` plus the
  per-rank golden being generated for the rows that rank actually owns.
* Passing the whole ``[gbs, 3]`` sampling_params instead of this rank's slice makes every
  rank apply rows ``0..R-1``'s temperature and top-p. The kernel rejects the wrong shape with
  a ``kernel_assert``, and -- because a shape check cannot catch a rank that slices the right
  SIZE from the wrong OFFSET -- every hardware case below uses PER-ROW-VARYING params, so a
  misindexed rank gets a different answer rather than an identical one.

Ties are forgiven on indices, never on values: bf16 has 8 mantissa bits, so a 201,088-wide
row has many exact ties and which index wins a tied slot is not determined.
"""

import os
import pickle
import tempfile
from typing import Any, final

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.gpt_oss_tail.tail_chunked_megakernel import (
    compose_hops,
    gpt_oss_tail_chunked_megakernel,
    owned_row_block,
    owned_rows,
)
from nkilib_src.nkilib.experimental.gpt_oss_tail.tail_chunked_megakernel_torch import (
    _DETERMINISTIC_DRAW,
    gpt_oss_tail_chunked_ref,
    rank_row_block,
    rmsnorm_tail,
    sample_rows,
)
from nkilib_src.nkilib.experimental.topk.chunked_topk import _HW_SORT_AVAILABLE, pick_chunks
from typing_extensions import override

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    InferenceArgs,
    Platforms,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.torch_ref_invalidation import ref_dependency_hash
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework
from test.utils.unit_test_framework import torch_ref_wrapper

# GPT-OSS-120B at LNC=1 is VTP = 128 ranks, vocab 201,088, so a per-rank shard of 1,571.
_OUTPUT_KEYS = ["logits", "topk_values", "topk_indices", "tokens"]

_SHARD_CACHE: dict = {}
_REF_CACHE: dict = {}
# The oracle knobs every hardware case runs with. ONE definition, used by the per-rank input
# generator AND the parent-side cache warm, so the two can never key different goldens.
_EPS = 1e-5
_ZIP_PAD = True
# Bump when _inputs' or _cached_shards' generation recipe changes meaning for an EXISTING
# mode: those helpers live in this test module, outside the reference module's import graph,
# so ref_dependency_hash cannot see their edits (see _cached_ref).
# rev 2: pad zeroing moved from the contiguous tail to the interleaved lanes.
# rev 3: topk_cut carries NON-INTEGER top_k on half the rows (the kernel-side floor's pin).
_INPUT_RECIPE_REV = 3


def _np_bf16(x):
    return np.asarray(x).astype(ml_dtypes.bfloat16)


def _cached_shards(vtp: int, v_s: int, hidden: int, mode: str = "randn"):
    """Per-rank lm_head shards, TRANSPOSED to [H, V_s] and C-contiguous.

    ``ascontiguousarray`` is load-bearing rather than cosmetic: a bare ``.T`` view keeps the
    [V_s, H] strides, and the DMA descriptor is built from strides, so a view silently
    restores the strided weight read the pre-transposed contract exists to remove -- while
    still producing correct numbers, so only a profile would reveal it.

    Cached because the collective harness needs shards in several places and, at VTP=128,
    building them is ~2.5 GB of fp32.

    Two adversarial WEIGHT modes (both keep shard row h=0 small and per-rank-distinct,
    because ``_ref`` fingerprints ranks by ``my_w[0, :8]``):

    * ``inf_topk``: all but the first 16 GLOBAL vocab columns hold a large NEGATIVE
      constant, so with a positive hidden every "bad" logit saturates the fp32 GEMM
      accumulation to -inf -- the in-contract overflow case -- while 16 columns stay
      finite. The top-k then contains both finite values and -inf.
    * ``subsentinel``: every weight is ~-12 with small noise, so every logit lands near
      -5000 -- BELOW the sampler's finite -3000 top-k mask sentinel.
    """
    key = (vtp, v_s, hidden, mode)
    if key not in _SHARD_CACHE:
        # bfloat16, matching the checkpoint and the kernel's DMA dtype. Also halves the dumped
        # per-rank weight from 19.30 MB to 9.65 MB, which at VTP=128 is 2.47 GB -> 1.23 GB.
        shards = [
            np.ascontiguousarray(
                (np.random.default_rng(9000 + r).standard_normal((v_s, hidden)) / np.sqrt(hidden)).astype(np.float32).T
            ).astype(ml_dtypes.bfloat16)
            for r in range(vtp)
        ]
        if mode in ("inf_topk", "inf_topk_deep"):
            for r in range(vtp):
                w = shards[r].astype(np.float32)
                fp = w[0:1, :].copy()  # keep the per-rank fingerprint row finite/distinct
                for j in range(v_s):
                    if r * v_s + j >= 16:
                        w[:, j] = -3.0e38
                w[0:1, :] = fp
                shards[r] = np.ascontiguousarray(w).astype(ml_dtypes.bfloat16)
        elif mode == "subsentinel":
            for r in range(vtp):
                rng = np.random.default_rng(9500 + r)
                w = (-12.0 + 0.01 * rng.standard_normal((hidden, v_s))).astype(np.float32)
                w[0:1, :] = shards[r][0:1, :].astype(np.float32)  # fingerprint row
                shards[r] = np.ascontiguousarray(w).astype(ml_dtypes.bfloat16)
        _SHARD_CACHE[key] = shards
    return _SHARD_CACHE[key]


def _inputs(gbs: int, hidden: int, hidden_actual: int, k: int, with_sampler: bool, mode: str = "randn"):
    """Deterministic inputs. Hidden states and gamma are REPLICATED across ranks."""
    rng = np.random.default_rng(1234)
    if mode == "allzero":
        # Every logit becomes 0 -> fully tied rows, and the variance is 0 so the norm's rsqrt
        # runs on eps alone. Drives the degenerate end of the norm and the top-k at once.
        x = np.zeros((gbs, hidden), dtype=np.float32)
    elif mode == "bigmag":
        x = (rng.standard_normal((gbs, hidden)) * 30.0).astype(np.float32)
    elif mode == "tinymag":
        x = (rng.standard_normal((gbs, hidden)) * 1e-6).astype(np.float32)
    elif mode in ("inf_topk", "inf_topk_deep", "subsentinel"):
        # POSITIVE hidden, so every (hidden, weight-column) product has the weight's sign and
        # the adversarial weight modes control the logits' sign deterministically -- inf_topk
        # needs its saturating columns to go to -inf on EVERY row, not a per-row coin flip.
        x = (0.05 + np.abs(rng.standard_normal((gbs, hidden))) * 0.1).astype(np.float32)
    else:
        x = (rng.standard_normal((gbs, hidden)) * 0.1).astype(np.float32)
    if mode in ("bf16grid", "fp16grid"):
        # Round the hidden onto the 16-bit grid so a 16-bit ``hidden_states`` input and this
        # fp32 oracle see the SAME numbers: the kernel's on-chip widen of a 16-bit value is
        # exact, so with grid-aligned inputs the two dtypes must agree bit for bit and the
        # gates stay exact rather than needing a rounding tolerance.
        grid = ml_dtypes.bfloat16 if mode == "bf16grid" else np.float16
        x = x.astype(grid).astype(np.float32)
    if hidden_actual < hidden:
        # Zero the INTERLEAVED pad lanes -- the layout every hardware case declares with
        # zero_interleaved_pad=True (4 groups of H/4; each group's last (H - H_u)/4 lanes are
        # pad). Zeroing the CONTIGUOUS tail instead violated that contract on every test: the
        # true pad lanes carried nonzero randn, so the sum(x^2)/H_u variance premise was never
        # exercised as the mxfp4 checkpoint presents it (kernel and oracle agreed on the same
        # contract-violating input, so the gates couldn't see it).
        assert hidden % 4 == 0 and hidden_actual % 4 == 0, (hidden, hidden_actual)
        x.reshape(gbs, 4, hidden // 4)[:, :, hidden_actual // 4 :] = 0.0
    gamma = (1.0 + 0.02 * rng.standard_normal((1, hidden))).astype(np.float32)
    sp = None
    if with_sampler:
        # PER-ROW-VARYING on purpose. Uniform params would hide a rank that sampled with the
        # wrong rows' settings, which is exactly the contract this kernel tightens.
        temp = (0.50 + 1.00 * rng.random(gbs)).astype(np.float32)  # temperature in [0.5, 1.5)
        if mode == "greedy":
            # GREEDY COVERAGE. Every other mode leaves temperature in [0.5, 1.5), i.e. strictly
            # above _SAMPLING_EPS, so the sampler's greedy branch -- the argmax path -- was never
            # executed by any test. Zeroing HALF the rows exercises both branches in one case, and
            # keeps them per-row so a kernel that applied one row's decision to another fails.
            temp[::2] = 0.0
        if mode == "draw_sweep":
            # A few greedy rows AMONG the swept draws prove greedy rows ignore u entirely.
            temp[::16] = 0.0
        top_k_col = np.full(gbs, k, dtype=np.float32)
        if mode == "inf_topk":
            # The cut must land INSIDE the 16 finite candidates: the regression this mode
            # exists for is a finite kth threshold selected from a value tile that ALSO
            # holds -inf slots (the masked-sum gather NaN-poisoned exactly there).
            top_k_col = rng.integers(1, 17, size=gbs).astype(np.float32)
        if mode == "inf_topk_deep":
            # The cut lands ON an -inf slot (17..32 of a 32-wide top-k with 16 finite
            # candidates): the selected threshold is itself -inf. Pins the range_select
            # select's one observable edge (threshold FP32_MIN instead of -inf, so -inf
            # values take the finite sentinel) -- tokens must still match the reference,
            # since both forms give those slots probability exactly 0.
            top_k_col = rng.integers(17, 33, size=gbs).astype(np.float32)
        if mode == "subsentinel":
            # Per-row cuts below k make the -3000 sentinel appear, which every top-k value
            # in this mode sits BELOW -- the greedy-slot-0 contract regime. Greedy rows
            # interleaved with sampled ones cover both branches in it.
            top_k_col = rng.integers(1, k + 1, size=gbs).astype(np.float32)
            temp[::2] = 0.0
        if mode == "topk_cut":
            # THE CUT IN ITS ACTIVE REGIME. Every other mode sets top_k == k, which makes the
            # kernel's threshold the row minimum and the below-threshold predicate ALL-FALSE:
            # the kth-select and the sentinel masking execute but never bite, so an off-by-one
            # in either would ship undetected. Per-row cuts down to top_k = 1 make them bite --
            # and the masked tail is exactly what builds the cdf plateaus the draw lands on.
            top_k_col = rng.integers(1, k + 1, size=gbs).astype(np.float32)
            # NON-INTEGER top_k on half the rows. top_k is fp32 RUNTIME data and the model
            # floors it; the kernel's threshold selection is an exact fp32 equality against an
            # integer ramp, so before the kernel-side floor a top_k of x.5 matched no slot and
            # silently produced thresh = 0.0 (masking every negative logit). These rows fail
            # the token gate on that kernel and pin the floor. (+0.5 at top_k == k exercises
            # the min(top_k, k) clamp path with a fractional input, flooring to k - 1 exactly.)
            top_k_col[1::2] += 0.5
        cols = [
            top_k_col,
            (0.80 + 0.19 * rng.random(gbs)).astype(np.float32),  # top_p in [0.80, 0.99)
            temp,
        ]
        if mode in ("draw_sweep", "topk_cut"):
            # RUNTIME DRAW: the 4-column sampling_params form, where u is per-row DATA rather
            # than the compiled-in 0.5 -- sweeping it needs no rebuild. Rows 0 and 1 pin the
            # count-below-threshold boundaries (u = 0 must select position 0; u -> 1 must stay
            # inside the kept set); the rest spans (0, 1).
            draw = (0.002 + 0.996 * rng.random(gbs)).astype(np.float32)
            draw[0] = 0.0
            if gbs > 1:
                draw[1] = 0.999
            cols.append(draw)
        sp = np.stack(cols, axis=1)
    return x, gamma, sp


def _cached_ref(vtp, gbs, hidden, hidden_actual, v_s, k, with_sampler, mode, eps=_EPS, zip_pad=_ZIP_PAD):
    """Memoize the oracle, IN PROCESS and ON DISK, as FULL-BATCH rows.

    An in-process dict alone is not enough: the collective harness generates the golden by
    FORKING one process per rank, so each fork would recompute the whole reference. At
    VTP=128 that is 128 x (a gbs*H*V matmul plus 2.5 GB of shards resident), which is
    indistinguishable from a hung test.

    The disk cache is what makes the forks share one computation. Callers should also warm it
    from the PARENT before ``run_test`` so no fork ever has to build the shards -- see
    ``_run_cross_rank``. Written atomically (temp + rename) because forks can race here.

    The key carries NO hop decomposition or batch-group count: every per-row output of the
    oracle is a pure function of that row (no rank-seeded draw), and hops/groups only decide
    which rows a rank OWNS. So the golden is stored once per shape/mode as [gbs, k] rows in
    global batch order and partitioned per rank at lookup with ``owned_rows`` (see
    ``_rank_view``). Keying on hops recomputed the same 1024x3072x201088 matmul up to six
    times per shape.
    """
    # eps and zip_pad (zero_interleaved_pad) are real oracle inputs, so they are IN the key:
    # leaving them out while the wrapper hardcoded 1e-5/True meant any future case exercising
    # the kernel's actual knobs would validate against (and cache-collide with) a wrong golden.
    key = (vtp, gbs, hidden, hidden_actual, v_s, k, with_sampler, mode, eps, zip_pad)
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    # The on-disk name carries a dependency hash of the ORACLE CODE (the reference module's
    # transitive import graph) on top of the shape/mode key: /tmp pickles outlive the pytest
    # session and any code edit, so a parameter-only key served STALE goldens after oracle or
    # input-recipe changes (and cross-contaminated workspaces sharing one /tmp). A None hash
    # means "unsafe to cache" -- fail closed and recompute. NOTE: _inputs lives in THIS test
    # module, outside the reference module's import graph, so bump _INPUT_RECIPE_REV whenever
    # the input generation recipe changes meaning for an existing mode.
    ref_hash = ref_dependency_hash(gpt_oss_tail_chunked_ref)
    cache_dir = os.environ.get("TAIL_REF_CACHE_DIR", tempfile.gettempdir())
    path = os.path.join(
        cache_dir,
        "tailchunkedreffull_" + "_".join(str(x) for x in key) + f"_r{_INPUT_RECIPE_REV}_{ref_hash}.pkl",
    )
    if ref_hash is None:
        path = None
    if path is not None and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                _REF_CACHE[key] = pickle.load(fh)
            return _REF_CACHE[key]
        except (EOFError, pickle.UnpicklingError):
            pass  # a partially-written file from a racing fork; recompute below

    x, gamma, sp = _inputs(gbs, hidden, hidden_actual, k, with_sampler, mode)
    out = gpt_oss_tail_chunked_ref(
        x,
        gamma,
        _cached_shards(vtp, v_s, hidden, mode),
        k,
        eps=eps,
        hidden_actual=hidden_actual,
        zero_interleaved_pad=zip_pad,
        sampling_params=sp,
    )
    # Keep only the small per-row results, in GLOBAL batch order: with the single ungrouped
    # hop rank r owns rows [r*R, (r+1)*R), so concatenating ranks in order IS batch order.
    # The assembled [R, V] rows are not retained; the index<->value pairing gate uses the
    # oracle's own index->value map instead, which needs no full row and cannot be fooled by
    # a wrong id (a wrong id is not in the map).
    slim = {
        "topk_values": np.concatenate([out["topk_values"][r] for r in range(vtp)], axis=0),
        "topk_indices": np.concatenate([out["topk_indices"][r] for r in range(vtp)], axis=0),
        "vocab": out["vocab"],
        "rows_per_rank": out["rows_per_rank"],
    }
    if with_sampler:
        slim["tokens"] = np.concatenate([out["tokens"][r] for r in range(vtp)], axis=0)
    _REF_CACHE[key] = slim
    if path is not None:
        try:
            fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(slim, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)  # atomic: concurrent forks never see a partial file
        except OSError:
            pass  # the cache is an optimisation, not a requirement
    return slim


def _rank_view(full: dict, rank: int, vtp: int, gbs: int, hops=None, batch_groups: int = 1) -> dict:
    """This rank's rows of a full-batch golden, IN RECEIVE ORDER, under the ownership map in
    force (contiguous for one ungrouped hop, mixed-radix for multi-hop, strided for the
    batch-grouped exchange). The rows a rank must produce are the only thing hops and groups
    change, so this is where they enter -- and a kernel with the wrong composition disagrees
    here on rows rather than agreeing by luck."""
    rows = np.asarray(owned_rows(rank, vtp, gbs, hops, batch_groups))
    view = {"topk_values": full["topk_values"][rows], "topk_indices": full["topk_indices"][rows]}
    if "tokens" in full:
        view["tokens"] = full["tokens"][rows]
    return view


class TestTailChunkedOwnership:
    """Hardware-free gates on the sharding arithmetic itself."""

    @pytest.mark.fast
    @pytest.mark.parametrize("vtp", [8, 16, 128])
    @pytest.mark.parametrize("gbs_mult", [1, 4, 8])
    def test_single_hop_row_ownership_is_identity_and_tiles(self, vtp: int, gbs_mult: int):
        """With ONE hop, ownership must tile the batch exactly once AND be the identity.

        Being the identity is the whole reason the collective is routed over a rank-ordered
        group: it is what lets the caller's final all_gather land in batch order with no
        de-permutation. If a future change makes the map a shuffle, the caller silently puts
        every token in the wrong row, so it is asserted rather than assumed.
        """
        gbs = vtp * gbs_mult
        seen = np.zeros(gbs, dtype=np.int64)
        starts = []
        for r in range(vtp):
            r0, nr = rank_row_block(r, vtp, gbs)
            assert nr == gbs // vtp
            seen[r0 : r0 + nr] += 1
            starts.append(r0)
        assert (seen == 1).all(), "row ownership does not tile the batch exactly once"
        assert starts == sorted(starts) and starts[0] == 0, "ownership is not the identity map"
        assert starts == [r * (gbs // vtp) for r in range(vtp)]

    def test_rank_row_block_rejects_indivisible_batch(self):
        with pytest.raises(ValueError, match="divisible"):
            rank_row_block(0, 128, 200)

    def test_rank_row_block_rejects_hops_that_do_not_multiply_to_vtp(self):
        """A typo'd decomposition must raise, not silently address the wrong rows."""
        with pytest.raises(ValueError, match="multiply to"):
            rank_row_block(0, 128, 1024, hops=(8, 4))

    @pytest.mark.parametrize(
        "vtp,gbs,hops",
        [
            (128, 128, (8, 4, 4)),
            (128, 512, (8, 4, 4)),
            (128, 1024, (8, 4, 4)),
            (128, 1024, (8, 8, 2)),
            (128, 128, (8, 16)),
            (16, 16, (4, 4)),
            (8, 8, (2, 2, 2)),
        ],
    )
    def test_multi_hop_preserves_global_vocab_ids_and_row_ownership(self, vtp: int, gbs: int, hops: tuple):
        """Simulate the hop dataflow in numpy and check the TWO properties the kernel relies on.

        trn3 cannot execute an all-to-all mesh wider than 8, so production has to decompose
        into hops -- and neither of the two things that makes safe is checkable on the shared
        fleet, whose trn3_a0 hosts reject more than one mesh per NEFF outright. So they are
        pinned here, hardware-free, against the SAME ``compose_hops``/``rank_row_block`` the
        kernel imports:

        1. ``column index == global vocab id`` still holds after every hop, which is what lets
           ``chunked_topk`` return global ids with no correction term anywhere.
        2. ``rank_row_block`` agrees with where the rows actually land, and the ownership map
           still tiles the batch exactly once.

        A hop mesh listed in the wrong order, or an off-by-one in the mixed-radix sum, breaks
        exactly one of these while leaving the other intact, so both are asserted.
        """
        meshes = compose_hops(vtp, hops)
        for h, mesh in enumerate(meshes):
            flat = sorted(r for g in mesh for r in g)
            assert flat == list(range(vtp)), f"hop {h} mesh does not cover each rank exactly once"

        # state[r][row, col] = (vocab shard that column came from, batch row it came from)
        state = {}
        for r in range(vtp):
            a = np.empty((gbs, 1, 2), dtype=np.int64)
            a[..., 0] = r  # rank r starts holding vocab shard r ...
            a[..., 1] = np.arange(gbs)[:, None]  # ... for every batch row
            state[r] = a
        for mesh in meshes:
            new = {}
            for group in mesh:
                g = len(group)
                out_rows = state[group[0]].shape[0] // g
                # all_to_all splits on rows and the de-interleave concatenates on columns, in
                # ASCENDING PEER ORDER -- the ordering the vocab-id property depends on.
                for m, r in enumerate(group):
                    new[r] = np.concatenate(
                        [state[group[p]][m * out_rows : (m + 1) * out_rows] for p in range(g)], axis=1
                    )
            state = new

        rows_per_rank = gbs // vtp
        seen = np.zeros(gbs, dtype=np.int64)
        for r in range(vtp):
            a = state[r]
            assert a.shape == (rows_per_rank, vtp, 2), f"rank {r} ended with shape {a.shape}"
            # Property 1: column block p holds vocab shard p, on every row.
            assert (a[..., 0] == np.arange(vtp)[None, :]).all(), (
                f"rank {r}: column order is not global vocab order after hops {hops}; "
                f"block sources = {a[0, :, 0].tolist()[:8]}"
            )
            # Property 2: the rows present are exactly the ones rank_row_block claims.
            r0, nr = rank_row_block(r, vtp, gbs, hops)
            assert nr == rows_per_rank
            assert (a[:, 0, 1] == np.arange(r0, r0 + nr)).all(), (
                f"rank {r}: holds rows {a[:, 0, 1].tolist()[:4]} but rank_row_block says {r0}"
            )
            seen[r0 : r0 + nr] += 1
        assert (seen == 1).all(), "multi-hop ownership does not tile the batch exactly once"

    @pytest.mark.fast
    def test_owned_rows_grouped_keeps_hop_validation_and_matches_formula(self):
        """owned_rows(G > 1) is DERIVED from owned_row_block, so a mis-specified hop is
        refused on the grouped path exactly as on the ungrouped one (it used to return a
        plausible strided set for hops=(8,) at vtp=128), and the result equals the closed
        form row(g, j) = g*gbs/G + rank*(R/G) + j for every rank, G in {1, 2, 4}."""
        with pytest.raises(ValueError, match="multiply to"):
            owned_rows(5, 128, 1024, hops=(8,), batch_groups=2)
        with pytest.raises(ValueError, match="single-hop"):
            owned_rows(0, 128, 1024, hops=(8, 16), batch_groups=2)
        for vtp in (8, 32, 128):
            for g_ in (1, 2, 4):
                gbs = vtp * 8
                seen = np.zeros(gbs, dtype=np.int64)
                for r in range(vtp):
                    got = owned_rows(r, vtp, gbs, None, g_)
                    s_, rpc = gbs // g_, (gbs // vtp) // g_
                    want = [g * s_ + r * rpc + j for g in range(g_) for j in range(rpc)]
                    assert got == want, (vtp, g_, r)
                    seen[got] += 1
                    if g_ == 1:
                        start, rows = owned_row_block(r, vtp, gbs)
                        assert got == list(range(start, start + rows))
                assert (seen == 1).all(), "grouped ownership must tile the batch exactly once"

    @pytest.mark.fast
    def test_oracle_ownership_only_permutes_rows(self):
        """Hops and batch groups change WHICH rows a rank owns and nothing else.

        Pins two things at once: (1) the oracle's ``tokens_full`` is assembled with the
        ownership map in force (it used the contiguous block for G > 1 and silently
        row-permuted half the tokens); (2) the golden cache's premise -- the full-batch rows
        from the ungrouped single hop, partitioned by ``owned_rows``, equal a direct oracle
        run under multi-hop or grouped ownership, row for row.
        """
        vtp, gbs, hidden, v_s, k = 8, 64, 64, 64, 8
        rng = np.random.default_rng(7)
        x = (rng.standard_normal((gbs, hidden)) * 0.1).astype(np.float32)
        gamma = (1.0 + 0.02 * rng.standard_normal((1, hidden))).astype(np.float32)
        shards = [(rng.standard_normal((hidden, v_s)) * 0.05).astype(np.float32) for _ in range(vtp)]
        sp = np.stack(
            [
                rng.integers(1, k + 1, size=gbs).astype(np.float32),
                (0.80 + 0.19 * rng.random(gbs)).astype(np.float32),
                (0.5 + rng.random(gbs)).astype(np.float32),
            ],
            axis=1,
        )
        base = gpt_oss_tail_chunked_ref(x, gamma, shards, k, sampling_params=sp)
        full = {
            key: np.concatenate([base[key][r] for r in range(vtp)], axis=0)
            for key in ("topk_values", "topk_indices", "tokens")
        }
        for hops, groups in ((None, 2), (None, 4), ((2, 2, 2), 1), ((4, 2), 1)):
            other = gpt_oss_tail_chunked_ref(x, gamma, shards, k, sampling_params=sp, hops=hops, batch_groups=groups)
            assert np.array_equal(other["tokens_full"], base["tokens_full"]), (hops, groups)
            for r in range(vtp):
                rows = owned_rows(r, vtp, gbs, hops, groups)
                assert np.array_equal(other["tokens_full"][rows], other["tokens"][r]), (hops, groups, r)
                view = _rank_view(full, r, vtp, gbs, hops, groups)
                for key in ("topk_values", "topk_indices", "tokens"):
                    assert np.array_equal(view[key], other[key][r]), (hops, groups, r, key)

    @pytest.mark.fast
    def test_norm_uses_the_interleaved_padding_convention(self):
        """The mxfp4 convention is not interchangeable with the contiguous one.

        Both zero the same NUMBER of lanes, so a shape check cannot tell them apart; they
        zero DIFFERENT lanes and give different answers. This pins which one is implemented.
        """
        h, h_u = 32, 24
        x = np.ones((2, h), dtype=np.float32)
        x[:, h_u:] = 0.0
        gamma = np.ones((1, h), dtype=np.float32)
        inter = rmsnorm_tail(x, gamma, 1e-5, h_u, zero_interleaved_pad=True)
        contig = rmsnorm_tail(x, gamma, 1e-5, h_u, zero_interleaved_pad=False)
        gw, keep = h // 4, h_u // 4
        for g in range(4):
            assert np.allclose(inter[:, g * gw + keep : (g + 1) * gw], 0.0), "interleaved lanes not zeroed"
        assert not np.allclose(inter, contig), "the two conventions must differ, or the test proves nothing"


@pytest.mark.skipif(
    not _HW_SORT_AVAILABLE,
    reason="hardware sort unavailable on this toolchain; the tail's chunked_topk refuses sorted=True",
)
@pytest_test_metadata(name="GPT-OSS Tail Chunked")
@pytest_marks(["gpt_oss_tail", "topk", "gpsimd", "collectives"])
@final
class TestTailChunkedMegakernel:
    """Hardware gates. LNC=1 only -- the kernel asserts n_prgs == 1."""

    @staticmethod
    def _out_desc(gbs, v_s, rows_per_rank, k):
        return {
            "logits": np.zeros((gbs, v_s), dtype=np.float32),
            "topk_values": np.zeros((rows_per_rank, k), dtype=ml_dtypes.bfloat16),
            "topk_indices": np.zeros((rows_per_rank, k), dtype=np.uint32),
            "tokens": np.zeros((rows_per_rank, 1), dtype=np.uint32),
        }

    def _run_cross_rank(
        self,
        test_manager,
        platform_target,
        vtp,
        gbs,
        hidden,
        hidden_actual,
        v_s,
        k,
        mode="randn",
        hops=None,
        deferred_rms=True,
        skip_nan_fold=True,
        hidden_dtype=None,
        a2a_batch_groups=1,
    ):
        """One cross-rank case at LNC=1, all VTP ranks.

        ``hops`` selects the collective form: None is the single mesh over all VTP ranks, and a
        tuple is the hop decomposition that trn3 requires above 8 ranks. Both must produce
        identical top-k values and GLOBAL vocab ids -- only row ownership differs -- so running
        the same geometry both ways is what proves the hop composition preserves the index map.
        """
        rows_per_rank = gbs // vtp
        groups = (tuple(range(vtp)),)
        meshes = None if hops is None else compose_hops(vtp, hops)
        collective_kwarg = {"replica_groups": groups} if hops is None else {"hop_meshes": meshes}
        # Profiling runs opt in: the fence makes per-hop durations mean transfer rather than
        # dispatch skew. It is numerically a no-op, so leaving it on here keeps the configuration
        # that gets profiled also the one that is gated.
        if os.environ.get("NKILIB_TAIL_FENCE") == "1":
            collective_kwarg["dummy_collectives"] = True
        # The fp32 logits are a gating output, not a product: dropping them removes an 805 KB HBM
        # write per rank. Correctness runs keep them (they are what isolates the norm + GEMM from
        # the top-k); profiling runs set NKILIB_TAIL_NO_LOGITS=1 so the measurement reflects what a
        # tokens-only deployment would actually execute.
        emit_logits = os.environ.get("NKILIB_TAIL_NO_LOGITS") != "1"
        collective_kwarg["emit_logits"] = emit_logits
        # NKILIB_TOPK_TAPS=1 appends chunked_topk's six diagnostic taps (see DEBUG TAPS there).
        # For --test-mode compile-only NEFFs run BY HAND on a 128-rank pod, which is where the
        # defects they diagnose live: the on-device profiler's capture flow saves every
        # output as output_<n>.2.npy, and because the taps are appended LAST the existing output
        # positions -- which this harness matches to validators POSITIONALLY -- do not move.
        # Taps only exist where the CHUNKING chain exists: a vocab small enough that
        # pick_chunks returns 1 delegates to the single-call path, whose config
        # kernel_asserts on debug_taps -- so gate on the same predicate the chunked_topk
        # suite uses instead of crashing every small-vocab rung of a taps-enabled run.
        if os.environ.get("NKILIB_TOPK_TAPS") == "1" and pick_chunks(gbs // vtp, vtp * v_s, k) > 1:
            collective_kwarg["debug_taps"] = True
        # Deferred rms and the stage-1 NaN-fold skip default ON in the kernel; the explicit
        # passthroughs are what let test_tail_chunked_deferred_rms_off and
        # test_tail_chunked_nan_fold_on pin their fallback paths on hardware.
        collective_kwarg["deferred_rms"] = deferred_rms
        collective_kwarg["skip_nan_fold"] = skip_nan_fold
        collective_kwarg["a2a_batch_groups"] = a2a_batch_groups
        out_keys = list(_OUTPUT_KEYS) if emit_logits else [key for key in _OUTPUT_KEYS if key != "logits"]
        x, gamma, sp = _inputs(gbs, hidden, hidden_actual, k, True, mode)
        shards = _cached_shards(vtp, v_s, hidden, mode)
        # Warm the oracle cache HERE, in the parent, so the per-rank forks never have to hold
        # the shards or redo the gbs*H*V matmul. Without this the VTP=128 cases look hung.
        golden_all = _cached_ref(vtp, gbs, hidden, hidden_actual, v_s, k, True, mode, _EPS, _ZIP_PAD)

        def per_rank_input_generator(rank_id: int):
            own = np.asarray(owned_rows(rank_id, vtp, gbs, hops, a2a_batch_groups))
            return {
                # hidden_dtype hands the kernel a 16-bit hidden; with a grid-aligned mode the
                # cast is exact, so the fp32 oracle stays valid unchanged.
                "hidden_states": x if hidden_dtype is None else x.astype(hidden_dtype),
                "gamma": gamma,
                "lm_head_weight_t": shards[rank_id],
                "eps": _EPS,
                "hidden_actual": hidden_actual,
                "zero_interleaved_pad": _ZIP_PAD,
                "k": k,
                # THIS RANK'S rows only, per the OWNERSHIP MAP IN FORCE (contiguous at
                # a2a_batch_groups == 1, strided otherwise) -- the kernel's contract, and
                # what stops a rank from sampling with another row's temperature.
                "sampling_params": sp[own, :],
                **collective_kwarg,
            }

        def _ref(
            hidden_states,
            gamma,
            lm_head_weight_t,
            eps=1e-5,
            hidden_actual=None,
            zero_interleaved_pad=True,
            k=256,
            sampling_params=None,
            replica_groups=None,
            hop_meshes=None,
            dummy_collectives=False,
            emit_logits=True,
            debug_taps=False,
            deferred_rms=True,
            skip_nan_fold=True,
            a2a_batch_groups=1,
        ):
            # validate_torch_ref_signature compares NAMED PARAMETER SETS, so every kernel
            # parameter must be named here even when unused. debug_taps only APPENDS outputs, so
            # the reference has nothing to add for it -- but it still has to be named.
            # deferred_rms is a pure realization choice (same math, same tolerance band), so the
            # ONE reference serves both settings. skip_nan_fold is the identity on the finite
            # logits every mode here produces, so it too has nothing to model.
            # eps and zero_interleaved_pad, in contrast, ARE modeled: they flow into the cached
            # oracle (and its memo key) and the per-rank logits below, so a case exercising
            # either knob validates against the right golden instead of a hardcoded-1e-5/True one.
            del sampling_params, replica_groups, hop_meshes, dummy_collectives
            del deferred_rms, skip_nan_fold
            del debug_taps
            cached = _cached_ref(vtp, gbs, hidden, hidden_actual, v_s, k, True, mode, eps, zero_interleaved_pad)
            # The reference is called once per rank but is NOT told which rank it is, so the
            # weight shard is the only per-rank signal. Fingerprint row 0 of the transposed
            # tile rather than comparing full shards, which would be O(world) x 19 MB.
            my_w = np.asarray(
                lm_head_weight_t.detach().to("cpu").float().numpy()
                if hasattr(lm_head_weight_t, "detach")
                else lm_head_weight_t,
                dtype=np.float32,
            )
            probe = my_w[0, :8]
            my_rank = next(
                (r for r in range(vtp) if np.array_equal(probe, np.asarray(shards[r][0, :8], dtype=np.float32))), 0
            )
            # logits is cheap to produce per rank: [gbs, H] @ [H, V_s], no full vocab needed.
            # Both operands bf16, matching the kernel's bf16 lm_head matmul. Widen a 16-bit
            # hidden first (exact) -- the kernel does the same on chip.
            hs = np.asarray(
                hidden_states.detach().to("cpu").float().numpy() if hasattr(hidden_states, "detach") else hidden_states,
                dtype=np.float32,
            )
            normed = rmsnorm_tail(hs, gamma, eps, hidden_actual, zero_interleaved_pad)
            # Partition the full-batch golden by the ownership map in force for THIS case.
            mine = _rank_view(cached, my_rank, vtp, gbs, hops, a2a_batch_groups)
            ref = {
                "topk_values": _np_bf16(mine["topk_values"]),
                "topk_indices": mine["topk_indices"].astype(np.uint32),
                "tokens": mine["tokens"].astype(np.uint32),
            }
            if emit_logits:
                ref["logits"] = _np_bf16(normed).astype(np.float32) @ _np_bf16(my_w).astype(np.float32)
            return ref

        vocab = golden_all["vocab"]

        def comparator(rank_id: int, golden: dict):
            g_val = np.asarray(golden["topk_values"], dtype=np.float32).reshape(rows_per_rank, k)
            g_idx = np.asarray(golden["topk_indices"]).reshape(rows_per_rank, k).astype(np.int64)
            # Present only when the kernel emits logits: under NKILIB_TAIL_NO_LOGITS=1 _ref
            # deliberately omits the key (and no LogitsValidator is built below), so an
            # unconditional dereference here would KeyError before any gate ran.
            g_logits = np.asarray(golden["logits"], dtype=np.float32).reshape(gbs, v_s) if emit_logits else None
            # index -> value, straight from the oracle. This is what makes the pairing gate
            # possible without recomputing the [R, V] row in every fork: a wrong vocab id is
            # simply absent from the map.
            id2val = [dict(zip(g_idx[j].tolist(), g_val[j].tolist(), strict=True)) for j in range(rows_per_rank)]
            own_rows = np.asarray(owned_rows(rank_id, vtp, gbs, hops, a2a_batch_groups))
            sp_rows = np.asarray(sp[own_rows, :], dtype=np.float32)
            # tinymag's entire dynamic range sits near 3e-4 (the norm's rsqrt runs on eps
            # alone), which is far BELOW the fixed absolute tolerances -- with atol=1e-2 the
            # value/logits/pairing gates were vacuously true for the very magnitudes the mode
            # exists to stress (an all-zero output passed). Scale the absolute tolerances to
            # the golden's own magnitude there; every other mode has O(1)+ magnitudes where
            # the fixed defaults already bite, so their gates are unchanged.
            tol_scale = max(float(np.abs(g_val).max()), 1e-30) if mode == "tinymag" else 1.0
            seen: dict[str, npt.NDArray[Any]] = {}

            def maybe_check_tokens(v: CustomValidator) -> bool:
                """Run the sampler gate as soon as values, indices AND tokens are decoded.

                Validator order is not guaranteed, so every one of the three calls this and
                whichever completes the set does the work. Without that, the gate would
                silently no-op whenever tokens happened to be validated first.
                """
                if not {"values", "indices", "tokens"} <= seen.keys():
                    return True
                kv, ki, got = seen["values"], seen["indices"], seen["tokens"]
                # Reference sampler on the KERNEL'S OWN top-k: deterministic given
                # (values, ids, params), so this checks the sampler exactly while staying
                # immune to which of several tied top-k sets the kernel returned.
                expect, ref_cdf = sample_rows(kv, ki.astype(np.uint32), sp_rows, return_cdf=True)
                expect = expect.astype(np.int64)
                draws = (
                    sp_rows[:, 3] if sp_rows.shape[1] > 3 else np.full(rows_per_rank, _DETERMINISTIC_DRAW, np.float32)
                )
                bad = []
                for j in range(rows_per_rank):
                    if got[j, 0] == expect[j, 0]:
                        continue
                    # A different id is legal only if it carries the SAME value. On fully
                    # degenerate input (every logit equal) the top-p boundary sits exactly on
                    # a cdf step, so the on-chip Hillis-Steele scan and numpy's cumsum can
                    # land one slot apart and pick a different, equally-valued candidate. A
                    # genuinely wrong pick has a different value and still fails here.
                    sg = np.flatnonzero(ki[j] == got[j, 0])
                    se = np.flatnonzero(ki[j] == expect[j, 0])
                    if sg.size and se.size and np.isclose(kv[j, sg[0]], kv[j, se[0]], rtol=1e-2, atol=1e-2 * tol_scale):
                        continue
                    # DRAW-BOUNDARY forgiveness, and ONLY that. The kernel's sampler and this
                    # numpy reference cannot agree to the last ulp: the on-chip exp and
                    # reciprocal are table-interpolated activations (~1e-3 relative vs libm),
                    # and the position compare is evaluated in multiplied form
                    # (fcum < draw * last) where the reference divides -- so every cdf step
                    # carries an O(1e-3) relative band in which the two sides may count one
                    # entry apart (measured: an adjacent-slot flip at |cdf - draw| = 2.4e-3).
                    # Forgive a mismatch ONLY when the picks are ADJACENT slots and the cdf
                    # step between them sits within that band of the draw. A real sampler
                    # defect (wrong mask, off-by-one counting) shows up as non-adjacent picks,
                    # many rows at once, or a boundary distance far outside the band -- all of
                    # which still fail.
                    bdelta = None
                    if sg.size and se.size and abs(int(sg[0]) - int(se[0])) == 1:
                        boundary = float(ref_cdf[j, min(int(sg[0]), int(se[0]))])
                        bdelta = boundary - float(draws[j])
                        if abs(bdelta) <= 5e-3 * max(1.0, abs(float(draws[j]))):
                            continue
                    bad.append(
                        (
                            j,
                            int(got[j, 0]),
                            int(expect[j, 0]),
                            int(sg[0]) if sg.size else -1,
                            int(se[0]) if se.size else -1,
                            bdelta,
                        )
                    )
                if bad:
                    v._print_with_log(
                        f"FAIL tokens rank {rank_id}: {len(bad)} rows where the sampler disagrees with the "
                        f"reference run on the kernel's OWN top-k, first (row, got, expected, got_slot, exp_slot, cdf-draw) = {bad[:6]}"
                    )
                return not bad

            def maybe_check_pairing(v: CustomValidator) -> bool:
                """Run the pairing gate as soon as BOTH values and indices are decoded.

                Same deferral pattern as maybe_check_tokens, and for the same reason:
                validator order is not guaranteed, and the checks below need the KERNEL's
                values tile. Running the pairing inline in IndicesValidator with
                ``seen.get("values")`` silently skipped every exemption whenever indices
                happened to validate first -- an order-dependent flake on correct kernels.

                PAIRING, as an ID-SET property with a boundary-tie allowance. Any correct
                top-k for a row must (a) contain EVERY oracle id whose value sits strictly
                above the k-th value (nothing can legally displace those), (b) contain
                nothing else except ids whose reported value TIES the k-th value (the only
                place a legal tie-break can swap an outside id in), and (c) pair each id the
                oracle also has with the oracle's value FOR THAT ID. All three are checked
                per row, ties or not.

                This deliberately replaced an earlier per-slot POSITIONAL tie exemption:
                at production width (vocab 201,088, k=256, bf16) nearly every row has some
                tied values, so that form forgave slot-by-slot -- and a fully PERMUTED vocab
                (mis-ordered replica group, wrong hop composition) has positionally identical
                values at every slot, passing vacuously on exactly the bug this gate exists
                to catch. The set form fails a permutation twice over: the kernel's ids are
                absent from the oracle's top-k at above-boundary values (b), and the oracle's
                above-boundary ids are missing from the kernel's set (a). Fully tied input
                (all-zero logits, any k of V ids correct) still passes: every value ties the
                boundary, so (a) requires nothing and (b) forgives everything.
                """
                if not {"values", "indices"} <= seen.keys():
                    return True
                kv, got = seen["values"], seen["indices"]
                for j in range(rows_per_rank):
                    boundary = float(g_val[j].min())

                    def ties_boundary(x: float, b: float = boundary) -> bool:
                        return bool(np.isclose(x, b, rtol=1e-2, atol=1e-2 * tol_scale))

                    got_ids = {int(i) for i in got[j].tolist()}
                    # (b) an id the oracle does NOT have is legal only at the boundary value
                    foreign = [
                        (slot, int(i))
                        for slot, i in enumerate(got[j].tolist())
                        if int(i) not in id2val[j] and not ties_boundary(float(kv[j, slot]))
                    ]
                    # (c) an id the oracle DOES have must carry the oracle's value for it
                    mispaired = [
                        (slot, int(i))
                        for slot, i in enumerate(got[j].tolist())
                        if int(i) in id2val[j]
                        and not np.isclose(float(kv[j, slot]), id2val[j][int(i)], rtol=1e-2, atol=1e-2 * tol_scale)
                    ]
                    # (a) every oracle id strictly above the boundary band must be returned
                    absent = [
                        int(i) for i, val in id2val[j].items() if not ties_boundary(val) and int(i) not in got_ids
                    ]
                    if foreign or mispaired or absent:
                        v._print_with_log(
                            f"FAIL pairing rank {rank_id} row {j}: {len(foreign)} returned ids are "
                            f"outside the oracle's top-{k} at above-boundary values (first "
                            f"{foreign[:4]}), {len(mispaired)} ids carry a value that is not the "
                            f"oracle's value for that id (first {mispaired[:4]}), {len(absent)} "
                            f"oracle ids strictly above the boundary are missing (first "
                            f"{absent[:4]}). A wrong peer or hop ordering in the collective lands "
                            f"exactly here, values still positionally correct."
                        )
                        return False
                return True

            class LogitsValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    # Only ever instantiated when emit_logits (see the validators dict), so
                    # the golden is present; the assert narrows the Optional for the checker.
                    assert g_logits is not None
                    got = np.frombuffer(inference_output, dtype=np.float32).reshape(gbs, v_s)
                    ok = np.allclose(got, g_logits, rtol=2e-2, atol=2e-2 * tol_scale)
                    if not ok:
                        d = np.abs(got - g_logits)
                        j = int(np.unravel_index(np.argmax(d), d.shape)[0])
                        self._print_with_log(
                            f"FAIL logits rank {rank_id}: max abs diff {float(d.max()):.4g} at row {j}; "
                            f"the norm or the lm_head GEMM disagrees with the oracle"
                        )
                    return ok

            class ValuesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    got = np.asarray(
                        np.frombuffer(inference_output, dtype=ml_dtypes.bfloat16).reshape(rows_per_rank, k),
                        dtype=np.float32,
                    )
                    seen["values"] = got
                    ok = np.allclose(np.sort(got, -1), np.sort(g_val, -1), rtol=1e-2, atol=1e-2 * tol_scale)
                    if not ok:
                        self._print_with_log(
                            f"FAIL topk_values rank {rank_id}: sorted value set differs. "
                            f"got_top6={np.sort(got[0])[-6:].tolist()} golden={np.sort(g_val[0])[-6:].tolist()}"
                        )
                    # Equal neighbours pass explicitly: np.diff of two -inf slots is NaN
                    # (inf - inf), and -inf values are in-contract (GEMM overflow saturation).
                    d = np.diff(got, axis=-1)
                    if ok and not bool(((d <= 1e-2 * tol_scale) | (got[:, 1:] == got[:, :-1])).all()):
                        self._print_with_log(f"FAIL order rank {rank_id}: not descending: {got[0][:8].tolist()}")
                        ok = False
                    return ok and maybe_check_pairing(self) and maybe_check_tokens(self)

            class IndicesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    got = np.frombuffer(inference_output, dtype=np.uint32).reshape(rows_per_rank, k).astype(np.int64)
                    seen["indices"] = got
                    ok = True
                    if not ((got >= 0) & (got < vocab)).all():
                        self._print_with_log(f"FAIL index range rank {rank_id}: max={int(got.max())} vs vocab={vocab}")
                        return False
                    for j in range(rows_per_rank):
                        if len(np.unique(got[j])) != k:
                            self._print_with_log(f"FAIL distinct rank {rank_id} row {j}: repeated vocab ids")
                            ok = False
                            break
                    return ok and maybe_check_pairing(self) and maybe_check_tokens(self)

            class TokensValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    seen["tokens"] = (
                        np.frombuffer(inference_output, dtype=np.uint32).reshape(rows_per_rank, 1).astype(np.int64)
                    )
                    return maybe_check_tokens(self)

            desc = self._out_desc(gbs, v_s, rows_per_rank, k)
            # ORDER MATTERS, not just the keys: the framework zips this mapping against the
            # kernel's RETURN TUPLE positionally, so `logits` has to come first exactly when the
            # kernel emits it. Appending it last instead fed the fp32 logits buffer to the values
            # validator, which showed up as "cannot reshape array of size 16384 into shape (8,32)".
            validators = {}
            if emit_logits:
                validators["logits"] = CustomValidatorWithOutputTensorData(LogitsValidator, desc["logits"])
            validators["topk_values"] = CustomValidatorWithOutputTensorData(ValuesValidator, desc["topk_values"])
            validators["topk_indices"] = CustomValidatorWithOutputTensorData(IndicesValidator, desc["topk_indices"])
            validators["tokens"] = CustomValidatorWithOutputTensorData(TokensValidator, desc["tokens"])
            return validators

        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gpt_oss_tail_chunked_megakernel,
            torch_ref=torch_ref_wrapper(_ref),
            per_rank_input_generator=per_rank_input_generator,
            collective_ranks=vtp,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            output_keys=out_keys,
            rtol=1e-2,
            atol=1e-2,
            inference_args=InferenceArgs(collective_ranks=vtp),
            custom_comparator=comparator,
        )

    @pytest.mark.skip_simulation  # class-B nccl collectives have no CPU-sim backend
    @pytest.mark.parametrize(
        "vtp,gbs,hidden,hidden_actual,v_s,k",
        [
            # Plumbing rungs: prove the collective + chunked top-k wiring cheaply.
            (8, 8, 512, 480, 512, 8),
            (16, 16, 512, 480, 512, 8),
            (8, 64, 512, 480, 128, 32),
            # MULTI-TILE: gbs > 128 means the norm+lm_head loop runs more than one 128-row tile.
            # Every other rung here has gbs <= 128 and so exercises exactly one tile, which
            # leaves the tile loop itself -- and the weight re-read across tiles -- untested.
            (8, 512, 512, 480, 128, 32),
            # PER-PEER PAYLOAD: the rungs above ship only v_s*2 = 128-256 B to each peer, and a
            # collective verifier has rejected chunks that small before. This rung keeps VTP=16
            # but uses the PRODUCTION shard width, so a failure that is really about payload size
            # is told apart from one about the kernel. Needs 16 cores -> trn3_a0 48xlarge.
            (16, 16, 3072, 2880, 1571, 256),
            # The production width via the SINGLE-hop form, xfailed: one mesh over 128 contiguous
            # ranks is 16 devices x 8 cores, and all_to_all has no mesh subtype for that mixed
            # device layout -- the capability is absent, on every host. Kept in the matrix so the
            # limitation is visible next to the form that does work. Real acceptance at these
            # three batch sizes is test_tail_chunked_cross_rank_multihop with hops=(8, 16).
            *[
                pytest.param(
                    128,
                    gbs_,
                    3072,
                    2880,
                    1571,
                    256,
                    marks=pytest.mark.xfail(
                        reason=(
                            "single-hop all_to_all over 128 contiguous ranks: mixed device layout, "
                            "unsupported mesh operation. Use hop_meshes=(8, 16)."
                        ),
                        strict=False,
                    ),
                )
                for gbs_ in (128, 512, 1024)
            ],
        ],
    )
    def test_tail_chunked_cross_rank(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        vtp: int,
        gbs: int,
        hidden: int,
        hidden_actual: int,
        v_s: int,
        k: int,
    ):
        """The tail on real hardware, all VTP ranks, with per-row-varying sampling params.

        Each rank gets its OWN lm_head shard (never a replicated copy), so a wrong peer
        ordering in the collective, a wrong row block, or a sampling_params misindex all fail
        here rather than silently agreeing.
        """
        self._run_cross_rank(test_manager, platform_target, vtp, gbs, hidden, hidden_actual, v_s, k)

    @pytest.mark.skip_simulation
    @pytest.mark.skipif(
        os.environ.get("NKILIB_RESERVED_TRN3") != "1",
        reason=(
            "multi-hop needs a host that allows more than ONE replica-group mesh per NEFF, and "
            "the shared fleet does not: both trn3 (3xlarge) and trn3_a0 (48xlarge) fail at LOAD "
            "with 'Failed to build and load collectives resources' at any rank count. A reserved "
            "16-device trn3 host does allow it, and is also the only host with the 128 physical "
            "cores the production rungs need. Set NKILIB_RESERVED_TRN3=1 to run there. The hop "
            "ARITHMETIC is covered hardware-free by "
            "TestTailChunkedOwnership::test_multi_hop_preserves_global_vocab_ids_and_row_ownership."
        ),
    )
    @pytest.mark.parametrize(
        "vtp,gbs,hidden,hidden_actual,v_s,k,hops",
        [
            # ACCEPTANCE, the form that actually EXECUTES at the production width: trn3 rejects
            # an all-to-all mesh wider than 8 (one device's worth of LNC=1 cores), and 128 needs
            # at least three such hops because two cap out at 64.
            # TWO hops, the minimum for VTP=128: hop 0 device-local, hop 1 one core per device
            # across all 16. Same decomposition the existing sharded tail uses.
            (128, 128, 3072, 2880, 1571, 256, (8, 16)),
            (128, 512, 3072, 2880, 1571, 256, (8, 16)),
            # gbs=1024 is the largest production size and the one that caught gpsimd_topk's
            # de-snake store/reload race: it used to return REPEATED vocab ids on 2 of 128 row
            # blocks with the top-k values exactly right, because under dge_mode.none nothing
            # ordered the phase-1 de-snake store against the phase-2 reload, and values and
            # indices ride in separate private_hbm buffers so the index one could lose alone.
            # Fixed in gpsimd_topk (desnake_dge is now unconditionally compiler-selected); all
            # four gates now pass on all 128 ranks at every size below. KEEP this case: nothing
            # cheaper reproduces it -- the same reduce geometry is clean over 544 standalone
            # blocks and over 128 blocks of the full dataflow at 32 ranks, because exposure comes
            # from the collective traffic in flight when the top-k starts.
            (128, 1024, 3072, 2880, 1571, 256, (8, 16)),
            # Three hops: legal too (every hop is one-core-per-device after hop 0) but strictly
            # more collectives for the same result. Kept to show the cost of the extra hop.
            (128, 1024, 3072, 2880, 1571, 256, (8, 4, 4)),
            # (8,8,2) sends rank r to a DIFFERENT row block than (8,16) and (8,4,4) do, which
            # happen to agree. Keeping a decomposition that disagrees is what makes a per-rank
            # failure attributable: follow-the-rows means data, stay-on-the-rank means core.
            (128, 1024, 3072, 2880, 1571, 256, (8, 8, 2)),
            # Cheap rungs, so a hop-composition regression is caught without the 128-core host.
            (16, 16, 256, 240, 64, 8, (4, 4)),
            (8, 8, 256, 240, 64, 8, (2, 2, 2)),
        ],
    )
    def test_tail_chunked_cross_rank_multihop(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        vtp: int,
        gbs: int,
        hidden: int,
        hidden_actual: int,
        v_s: int,
        k: int,
        hops: tuple,
    ):
        """Same geometry as the single-hop acceptance cases, but via the hop decomposition.

        This is the configuration that runs in production, because a single 128-participant
        all-to-all is rejected by the trn3 runtime. The gates are identical to the single-hop
        case -- same values, same GLOBAL vocab ids -- and only row ownership differs, so if the
        hop composition permuted the vocab the pairing gate fails with the values still right.

        HOST NOTE: needs a host that allows more than one replica-group mesh per NEFF. The
        shared fleet's trn3_a0 does not (it fails at LOAD, at any rank count), so these cases
        are expected to fail there for host reasons; a reserved 16-device trn3 host does allow
        it. See WHICH COLLECTIVE ACTUALLY RUNS in the kernel's module docstring.
        """
        self._run_cross_rank(test_manager, platform_target, vtp, gbs, hidden, hidden_actual, v_s, k, hops=hops)

    @pytest.mark.skip_simulation
    @pytest.mark.parametrize("mode", ["allzero", "bigmag", "tinymag", "greedy", "draw_sweep", "topk_cut"])
    def test_tail_chunked_adversarial_inputs(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        mode: str,
    ):
        """Degenerate and extreme magnitudes at a cheap VTP.

        ``allzero`` makes every logit identical (fully tied rows) and zeroes the variance so
        the norm's rsqrt runs on eps alone; the magnitude modes stress the rsqrt and the bf16
        dynamic range instead. ``greedy`` sets temperature to 0 on half the rows, which is the
        ONLY coverage of the sampler's argmax branch -- every other case has temperature well
        above the greedy epsilon, so that branch used to be dead code in testing.

        ``draw_sweep`` and ``topk_cut`` run the 4-COLUMN sampling_params form, where the
        inverse-transform draw u is a runtime per-row input rather than the compiled-in 0.5.
        ``draw_sweep`` sweeps u across (0, 1) plus both edges (with a few greedy rows to prove
        they ignore u); ``topk_cut`` varies top_k per row down to 1 -- the only coverage where
        the top-k threshold mask actually MASKS (every other mode's predicate is all-false) --
        with u swept on top, so draws land on the cdf plateaus that masking creates.
        """
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, mode=mode)

    @pytest.mark.skip_simulation
    def test_tail_chunked_deferred_rms_off(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """The legacy in-chain rms form behind ``deferred_rms=False``, gated on hardware.

        ``deferred_rms=True`` is the default and every other case exercises it, so this is
        what keeps the SAFETY FALLBACK from rotting: the flag must keep producing the
        pre-deferred realization -- ``(x*rms)*gamma`` feeding the GEMM -- and keep passing
        the same four gates against the same reference. If the fallback ever breaks, a
        production rollback of deferred_rms would fail exactly when it is needed most.
        """
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, deferred_rms=False)

    @pytest.mark.skip_simulation
    def test_tail_chunked_nan_fold_on(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """The stage-1 NaN fold behind ``skip_nan_fold=False``, gated on hardware.

        ``skip_nan_fold=True`` is the default and every other case exercises the fold-free
        path, so this is what keeps the fold from rotting: on finite input the two settings
        must be BITWISE identical (the fold's effect on finite data is the identity), and
        both must keep passing the same four gates against the same reference. NaN-INPUT
        coverage of the fold itself lives in the chunked_topk suite's bitpattern mode -- this
        kernel's logits cannot be NaN, which is the whole basis of the default.
        """
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, skip_nan_fold=False)

    @pytest.mark.skip_simulation
    @pytest.mark.parametrize("hdtype", [ml_dtypes.bfloat16, np.float16], ids=["bf16", "fp16"])
    def test_tail_chunked_hidden_16bit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        hdtype,
    ):
        """16-bit ``hidden_states``, gated on hardware, both half formats.

        The kernel widens a 16-bit hidden to fp32 through an explicit conversion copy right
        after the (half-sized) load, which is exact -- so on grid-aligned inputs it must
        agree with the fp32 oracle to the same gates as the fp32 input path. Both formats
        matter because real producers differ: the fused GPT-OSS caller hands bf16 off its
        seam all_reduce and fp16 off its gather. The explicit copy is load-bearing: feeding
        16-bit data STRAIGHT into the square/gamma chains computes at input precision
        (measured as real top-k/token diffs on hardware, 2026-08-31), which is exactly the
        defect this case would catch if the widen ever regressed to direct consumption.
        """
        mode = "bf16grid" if hdtype == ml_dtypes.bfloat16 else "fp16grid"
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, mode=mode, hidden_dtype=hdtype)

    @pytest.mark.skip_simulation
    def test_tail_chunked_inf_topk(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """+/-inf INSIDE the top-k values, finite kth threshold -- the sampler inf-safety pin.

        The inf_topk weight mode saturates every logit outside 16 designated vocab columns
        to -inf in the GEMM's fp32 accumulation (in-contract per skip_nan_fold), so each
        row's top-k holds finite values AND -inf slots, and the per-row top_k cut (1..16)
        selects a FINITE threshold from that mixed tile. The fused masked-sum gather
        NaN-poisoned exactly here -- ``0.0 * (-inf) = NaN`` in the non-selected slots made
        the reduced threshold NaN, ``vals < NaN`` all-false, and sub-threshold candidates
        kept their real values and got sampled -- so this case fails without the inf-safe
        predicated-copy select and pins it against regression.
        """
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, mode="inf_topk")

    @pytest.mark.skip_simulation
    def test_tail_chunked_inf_topk_deep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """The per-row cut lands ON an -inf slot: the selected threshold is itself -inf.

        Same saturating weights as inf_topk, but top_k in 17..32 of a 32-wide top-k with 16
        finite candidates. The one-instruction ``range_select`` threshold reads FP32_MIN for
        unselected slots, so here it returns FP32_MIN where the reference has -inf; the
        -inf values below it take the finite sentinel instead of staying -inf, and both
        give those slots probability exactly 0. This pins that the tokens still agree.
        """
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, mode="inf_topk_deep")

    @pytest.mark.skip_simulation
    def test_tail_chunked_subsentinel(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """Every top-k value BELOW the finite -3000 mask sentinel -- the greedy contract pin.

        The subsentinel weight mode lands every logit near -4400, so per-row top_k cuts
        make the sentinel numerically OUTRANK every kept value. Greedy rows (half, per-row
        interleaved) must still return slot 0 -- the highest KEPT candidate -- exactly as
        the reference sampler's pinned contract does; an argmax over the masked array would
        return a token the top_k cut explicitly excluded. Sampled rows in this regime
        follow the finite-sentinel math verbatim on both sides, so the token gate stays
        exact there too.
        """
        self._run_cross_rank(test_manager, platform_target, 8, 64, 512, 480, 128, 32, mode="subsentinel")

    @pytest.mark.skip_simulation
    @pytest.mark.parametrize(
        "vtp,gbs,hidden,hidden_actual,v_s,k,groups",
        [
            # THE cheapest geometry where a2a_batch_groups > 1 is LEGAL. The grouped exchange
            # requires the receive-layout fold, whose peer-width divisibility works out to
            # vtp >= 16 * P (stage-1 snake row width vtp*v_s/(16P) must be a multiple of v_s),
            # so it is impossible at vtp 8 and needs P >= 2 (vocab >= 65,536) for the fold to
            # exist at all: vtp=32, v_s=2048 -> vocab 65,536, P=2, rows_per_rank=8. G=2 then
            # satisfies whole-tile groups (128 rows each) and G | rows_per_rank. Every gate
            # runs under the STRIDED owned_rows map -- a kernel that grouped the exchange but
            # kept the contiguous ownership would fail values/pairing/tokens here, not agree
            # by luck.
            (32, 256, 256, 240, 2048, 16, 2),
            # The production arms (single-hop 128-rank form; same capability caveat as the
            # single-hop 128 rungs of test_tail_chunked_cross_rank, hence the same soft xfail).
            # G=2 and G=4 at gbs 512 and 1024. G=4 at 512 is the STRADDLING case: a stage-1
            # sort tile spans two receive slabs ((rows_per_rank/G)*P = 4 rows, half an 8-row
            # tile). Each 16-partition group's load selects its slab per token, so it must
            # be exactly as correct as the aligned rungs -- the hardware proof that alignment
            # is a performance preference, not a legality rule. No cheaper straddling rung
            # exists: groups must be whole 128-row lm_head tiles (gbs/G >= 128) and the fold
            # needs vtp >= 16*P, which together force vtp = 128 for any misaligned ratio.
            *[
                pytest.param(
                    128,
                    gbs_,
                    3072,
                    2880,
                    1571,
                    256,
                    g_,
                    marks=pytest.mark.xfail(
                        reason=(
                            "single-hop all_to_all over 128 contiguous ranks: mixed device layout, "
                            "unsupported mesh operation on runtimes without the enable."
                        ),
                        strict=False,
                    ),
                )
                for gbs_, g_ in ((512, 2), (512, 4), (1024, 2), (1024, 4))
            ],
        ],
    )
    def test_tail_chunked_grouped_a2a(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        vtp: int,
        gbs: int,
        hidden: int,
        hidden_actual: int,
        v_s: int,
        k: int,
        groups: int,
    ):
        """The batch-grouped exchange (a2a_batch_groups=G) under every gate, G-aware.

        G splits the single all-to-all into G full-mesh exchanges over gbs/G whole lm_head
        tiles each, so exchange g fires under the rest of the GEMM -- and CHANGES THE
        OWNERSHIP CONTRACT: rank r owns the strided ``owned_rows`` set, not a contiguous
        block. The harness slices sampling_params, builds the golden and gates values, ids
        and tokens with that map, so this is the test that pins the contract the fused
        caller must honour (measured -35/-55..-67 us tail window at gbs 512/1024 in the
        fused GPT-OSS decode, all gates passing on 128 ranks).
        """
        self._run_cross_rank(
            test_manager, platform_target, vtp, gbs, hidden, hidden_actual, v_s, k, a2a_batch_groups=groups
        )

    @pytest.mark.skip_simulation
    # 64 and 128 are xfailed, not removed: keeping them in the matrix is what documents the
    # limitation. This sweep uses the SINGLE-hop form, so its replica group is all vtp ranks
    # CONTIGUOUSLY. At 64 and 128 that group spans several devices with several cores each, and
    # all_to_all has no mesh subtype for that layout -- the capability simply is not there. A
    # group must be entirely within one device (<= 8 cores at LNC=1) or hold exactly one core per
    # device (<= 16), so 32 is the largest contiguous group that can work and the production path
    # is multi-hop instead. See test_tail_chunked_cross_rank_multihop for 128 ranks.
    @pytest.mark.parametrize(
        "vtp",
        [
            8,
            16,
            32,
            pytest.param(
                64,
                marks=pytest.mark.xfail(
                    reason=(
                        "all_to_all cannot span 64 contiguous ranks: that is 8 devices x 8 cores, "
                        "a MIXED device layout with no supporting mesh subtype. trn3_a0 aborts in "
                        "encd_mesh_memcopy; reserved trn3 rejects it as op_type 7 unsupported. Use "
                        "the multi-hop form."
                    ),
                    strict=False,
                ),
            ),
            pytest.param(
                128,
                marks=pytest.mark.xfail(
                    reason=(
                        "all_to_all cannot span 128 contiguous ranks: 16 devices x 8 cores, a MIXED "
                        "device layout with no supporting mesh subtype. This is the production rank "
                        "count and exactly why the kernel takes hop_meshes -- (8, 16) is legal and "
                        "is validated by test_tail_chunked_cross_rank_multihop."
                    ),
                    strict=False,
                ),
            ),
        ],
    )
    def test_tail_chunked_rank_scaling(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        vtp: int,
    ):
        """Same small shape at increasing rank counts, to isolate the COLLECTIVE from the math.

        The all-to-all is the one part of this kernel whose cost and correctness depend on the
        participant count, so it gets its own sweep at a cheap vocab. A failure that appears
        only past some rank count is a runtime/mesh limit rather than a kernel bug, and this
        is what tells the two apart.
        """
        # v_s=512 -> 1024 B per peer. At v_s=64 the per-peer chunk is 128 B and trn3_a0's
        # collective refuses it above 8 ranks, which made this sweep look like a rank-count limit
        # when it was a payload-size one. The point of the sweep is the participant count, so the
        # payload has to be big enough not to be the variable.
        self._run_cross_rank(test_manager, platform_target, vtp, vtp, 512, 480, 512, 8)
