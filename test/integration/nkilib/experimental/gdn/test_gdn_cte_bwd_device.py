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
"""Device-kernel tests for `gdn_cte_bwd`, scored against the autograd oracle.

Driven through the SAME harness as the forward (`test_gdn_cte.py`): `UnitTestFramework`
compiles `gdn_cte_bwd` and runs it on the selected platform, and `_torch_ref` supplies the
golden. One test body therefore covers compile-only, simulation and hardware, and the
`--test-mode` flag decides which -- the tests do not call `nki.simulate` themselves.

Compute dtype follows `query.dtype`, so the kernel is exercised in both fp32 and bf16, with
the per-dtype tolerance the forward uses. q/k/v carry that dtype; g_log, beta, d_out and the
state stay fp32 on both sides, as in the kernel.
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.gdn.gdn_cte_bwd import gdn_cte_bwd
from nkilib_src.nkilib.experimental.gdn.gdn_cte_bwd_torch import gdn_cte_bwd_torch_ref
from nkilib_src.nkilib.experimental.gdn.gdn_cte_utils import CHUNK, build_mask_pack

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Per-token |g| magnitude for the ordinary shapes. The model's real decay is far larger; see
# `test_cases_large_decay`.
MILD_DECAY_SCALE = 0.05

DEFAULT_GROUP_SIZE = 8

# The forward's tolerance (2e-2) is the bf16 floor and holds for every gradient; fp32 is tighter.
_DTYPE_TOL = {nl.bfloat16: 2e-2, nl.float32: 1e-3}


def _generate_inputs(
    batch,
    n_k_heads,
    n_v_heads,
    seqlen,
    key_head_dim,
    value_head_dim,
    dtype,
    with_state,
    group_size=DEFAULT_GROUP_SIZE,
    decay_scale=MILD_DECAY_SCALE,
) -> dict:
    """Kernel inputs in the forward's distribution (q/k L2-normed, g<0, beta in (0,1), v*0.5).

    `decay_scale` sets the per-token |g| magnitude. q/k/v are cast to `dtype`; everything else
    stays fp32, matching the kernel's own split.
    """
    rng = np.random.RandomState(42)

    def l2_normed(shape):
        arr = rng.randn(*shape).astype(np.float32)
        return arr / np.sqrt((arr * arr).sum(axis=-1, keepdims=True) + 1e-6)

    inputs = {
        "query": dt.static_cast(l2_normed((batch, n_k_heads, seqlen, key_head_dim)), dtype),
        "key": dt.static_cast(l2_normed((batch, n_k_heads, seqlen, key_head_dim)), dtype),
        # `value` is the conv+silu output on the real path, so O(1) magnitude.
        "value": dt.static_cast(rng.randn(batch, n_v_heads, seqlen, value_head_dim).astype(np.float32) * 0.5, dtype),
        "g_log": -np.abs(rng.randn(batch, n_v_heads, seqlen).astype(np.float32)) * decay_scale,
        "beta": (1.0 / (1.0 + np.exp(-rng.randn(batch, n_v_heads, seqlen).astype(np.float32)))).astype(np.float32),
        "d_out": rng.randn(batch, n_v_heads, seqlen, value_head_dim).astype(np.float32),
        "mask_pack": build_mask_pack(),
        "scale": float(1.0 / np.sqrt(np.float32(key_head_dim))),
        "group_size": group_size,
    }
    if with_state:
        # Read-only here: unlike the forward's in/out state, the backward does not alias its
        # state inputs, so neither needs the ".must_alias_input" key.
        inputs["recurrent_state"] = rng.randn(batch, n_v_heads, key_head_dim, value_head_dim).astype(np.float32)
        inputs["d_state"] = rng.randn(batch, n_v_heads, key_head_dim, value_head_dim).astype(np.float32)
    return inputs


@torch_ref_wrapper
def _torch_ref(
    query,
    key,
    value,
    g_log,
    beta,
    d_out,
    mask_pack=None,
    scale=1.0,
    group_size=DEFAULT_GROUP_SIZE,
    recurrent_state=None,
    d_state=None,
):
    grads = gdn_cte_bwd_torch_ref(
        query,
        key,
        value,
        g_log,
        beta,
        d_out,
        mask_pack=mask_pack,
        scale=scale,
        group_size=group_size,
        recurrent_state=recurrent_state,
        d_state=d_state,
    )
    # Keys and ORDER must match the kernel's return tuple -- six on every path, including
    # d_recurrent_state when the state was omitted (dL/dS0 of the implicit zero state).
    return {
        "d_query": grads.d_query,
        "d_key": grads.d_key,
        "d_value": grads.d_value,
        "d_g_log": grads.d_g_log,
        "d_beta": grads.d_beta,
        "d_recurrent_state": grads.d_recurrent_state,
    }


@pytest_test_metadata(name="GDN CTE BWD")
@pytest_marks(["attention", "gdn", "gdn_cte", "gdn_cte_bwd"])
class TestGdnCteBwd:
    gdn_cte_bwd_params = "batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, dtype"
    _ABBREVS = {
        "batch": "b",
        "n_k_heads": "nkh",
        "n_v_heads": "nvh",
        "seqlen": "seq",
        "key_head_dim": "dk",
        "value_head_dim": "dv",
        "dtype": "dt",
        "with_state": "st",
        "decay_scale": "gs",
        "group_size": "g",
        "logical_nc_config": "lnc",
    }

    # Small shapes, both dtypes. 256 is two chunks, the cheapest case that still exercises the
    # reverse scan across a chunk boundary; 128 is ONE chunk, where the scan has no boundary to
    # cross and `group_layout` clamps the group to a single member.
    test_cases_basic = [
        (1, 1, 1, 256, 16, 16, nl.float32),
        (1, 1, 1, 256, 16, 16, nl.bfloat16),
        (1, 1, 1, 128, 16, 16, nl.bfloat16),
    ]

    # Batch, Dk != Dv, GQA head sharing, and full 128 head dims -- the shapes the kernel's
    # tiling and its dQ/dK group accumulation actually differ on.
    test_cases_large = [
        (2, 1, 1, 384, 16, 24, nl.bfloat16),  # batch, Dk != Dv, three chunks
        (1, 2, 4, 256, 16, 16, nl.bfloat16),  # GQA: one key head shared by two value heads
        (1, 1, 1, 256, 128, 128, nl.float32),  # full head dims
        (1, 1, 1, 256, 128, 128, nl.bfloat16),
        (1, 2, 4, 256, 128, 128, nl.bfloat16),  # GQA at full head dims
    ]

    # LONG CONTEXT, past what the state stash used to allow. 8192 is 64 chunks; the stash budget
    # admits up to 49152 at these head dims, and 32768 and 49152 were both validated by hand on
    # trn2 against the oracle. 8192 is the largest that is cheap enough to keep in the suite.
    test_cases_long_context = [
        (1, 1, 1, 8192, 128, 128, nl.bfloat16),
    ]

    # DEEP GQA groups. `test_cases_large` tops out at a head ratio of 2, which never exercises the
    # dQ/dK head sum's middle case -- a value head that is neither the first nor the last of its
    # group, i.e. the only one that both reads and writes the fp32 accumulator.
    test_cases_gqa_depth = [
        (1, 1, 4, 256, 64, 64, nl.bfloat16),
        (1, 1, 8, 256, 64, 64, nl.bfloat16),
    ]

    # Large per-token decay. The model's g = -exp(A_log)*softplus(a+dt_bias) reaches |g| ~ 10 at
    # init, so G_s = exp(cumsum g) UNDERFLOWS to 0 inside a 128-chunk. Every case above runs at
    # |g| ~ 0.05 and so never reaches that regime -- which is exactly where a gate gradient
    # carrying a 1/G_s intermediate turns the underflow into inf*0 = nan.
    test_cases_large_decay = [
        (1, 1, 1, 256, 16, 16, nl.float32),  # dense
        (1, 2, 4, 256, 16, 16, nl.float32),  # GQA: the dQ/dK group accumulation also sees it
    ]

    # THE RESIDUAL-GROUP PATH, which no shape above reaches: every chunk count above divides
    # evenly by its group width, so `group_layout`'s residual is 0 in all of them. These are the
    # (seqlen, group_size) pairs where it does not divide, and the backward is the side that needs
    # them most -- it walks the groups in REVERSE, so the residual group is the FIRST one its scan
    # visits rather than the last, and the state gradient has to cross that boundary intact.
    # Mirrors the forward's own residual cases (`test_gdn_cte.py`, 2176/2304/8320) at a cheaper
    # head dim, including a PRIME chunk count, which is the case a divisor-lowering group scheme
    # collapses to no grouping at all.
    test_cases_grouping = [
        (1152, 8),  # 9 chunks at the default: 1 full group of 8 + residual 1
        (768, 4),  # 6 chunks: 1 full group of 4 + residual 2
        (2176, 16),  # 17 chunks (prime) at MAX_GROUP_SIZE: 1 full group of 16 + residual 1
        (768, 1),  # group_size=1: no grouping at all, the reference emission order
    ]

    # HIGH BATCH, matching the forward's own batch-32 cases (`test_gdn_cte.py`). Batch multiplies
    # the SPMD lane count, so this is where the lane loop runs deep: 32 lanes over the default
    # 2-program grid is 16 per program, against the 1-3 lanes everything else here uses. 128 is one
    # chunk per lane, the cheapest way to get there; 2176 is 17 chunks, a PRIME count that leaves a
    # residual group, so it is the batch-and-residual combination together -- measured at ~26 s in
    # the simulator against ~1.6 s for the single-chunk case, hence `slow_simulation`, which still
    # leaves it running under compile-only and on hardware. The forward's third batch-32 shape
    # (2304, residual 2) is omitted: `test_cases_grouping` already covers a residual of 2.
    test_cases_high_batch = [
        (32, 1, 1, 128, 16, 16, nl.bfloat16),
        pytest.param(32, 1, 1, 2176, 16, 16, nl.bfloat16, marks=pytest.mark.slow_simulation),
    ]

    # SPMD sharding. Lanes here are (batch x KEY head), not the forward's (batch x value head),
    # because every dQ/dK slice must be owned by ONE program -- so this is the backward's own
    # split, not an inherited one, and it needs its own cases. lnc=1 is in the list because a trn2
    # launch is a 2-program grid by DEFAULT, so without it the single-program path is covered
    # nowhere in this file.
    test_cases_sharding = [
        (1, 1, 1, 1),  # one core: the unsharded baseline
        (2, 1, 1, 2),  # 2 lanes on 2 cores: even split
        (3, 1, 1, 2),  # 3 lanes on 2 cores: core 0 takes 2, core 1 takes 1 (the extra_lanes path)
        (1, 1, 1, 2),  # ONE lane on 2 cores: core 1 gets no work at all
        (1, 2, 4, 2),  # GQA: each core owns a whole key head and both of its value heads
    ]

    def _run_test(
        self,
        test_manager,
        platform_target,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
        with_state=True,
        decay_scale=MILD_DECAY_SCALE,
        group_size=DEFAULT_GROUP_SIZE,
        logical_nc_config=None,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_inputs(
                batch,
                n_k_heads,
                n_v_heads,
                seqlen,
                key_head_dim,
                value_head_dim,
                dtype,
                with_state,
                group_size=group_size,
                decay_scale=decay_scale,
            )

        def output_tensors(kernel_input):
            compute_dtype = kernel_input["query"].dtype
            # ORDER must match the kernel's return tuple. dQ/dK/dV carry the compute dtype; the
            # gate/beta gradients and the state gradient are fp32 in the kernel regardless.
            return {
                "d_query": np.zeros((batch, n_k_heads, seqlen, key_head_dim), dtype=compute_dtype),
                "d_key": np.zeros((batch, n_k_heads, seqlen, key_head_dim), dtype=compute_dtype),
                "d_value": np.zeros((batch, n_v_heads, seqlen, value_head_dim), dtype=compute_dtype),
                "d_g_log": np.zeros((batch, n_v_heads, seqlen), dtype=np.float32),
                "d_beta": np.zeros((batch, n_v_heads, seqlen), dtype=np.float32),
                "d_recurrent_state": np.zeros((batch, n_v_heads, key_head_dim, value_head_dim), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_cte_bwd,
            torch_ref=_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        tol = _DTYPE_TOL[dtype]
        framework.run_test(
            test_config=None,
            compiler_args=(
                CompilerArgs(platform_target=platform_target)
                if logical_nc_config is None
                else CompilerArgs(platform_target=platform_target, logical_nc_config=logical_nc_config)
            ),
            rtol=tol,
            atol=tol,
        )

    @pytest.mark.fast
    @pytest_parametrize(gdn_cte_bwd_params, test_cases_basic, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_basic(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
    ):
        self._run_test(
            test_manager, platform_target, batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, dtype
        )

    @pytest_parametrize(gdn_cte_bwd_params, test_cases_large, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_large(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
    ):
        self._run_test(
            test_manager, platform_target, batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, dtype
        )

    @pytest.mark.fast
    @pytest_parametrize("dtype", [nl.float32, nl.bfloat16], abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_omitted_state(self, test_manager: Orchestrator, platform_target: Platforms, dtype):
        """Omitting `recurrent_state`/`d_state` must start the scan from zero on both carries.

        A DIFFERENT TRACE, not different data: with the arguments absent the kernel emits no
        state load and no d_state load, memsetting both carries instead. It still returns all
        six gradients, and dL/dS0 -- the scan's carry out of chunk 0, here the gradient of the
        implicit zero state -- is validated with the rest. Prefill callers take this path.
        """
        self._run_test(test_manager, platform_target, 1, 1, 1, 256, 16, 16, dtype, with_state=False)

    @pytest.mark.fast
    @pytest_parametrize(
        gdn_cte_bwd_params + ", decay_scale",
        [case + (decay_scale,) for case in test_cases_large_decay for decay_scale in (5.0, 12.0)],
        abbrevs=_ABBREVS,
    )
    def test_gdn_cte_bwd_stable_under_large_decay(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
        decay_scale,
    ):
        """The KERNEL's gradients stay finite and match the oracle when exp(cumsum g) underflows.

        The device counterpart of `test_gdn_cte_bwd.py::test_backward_stable_under_large_decay`,
        which pins this same property on the chunked torch reference. Both halves need their own
        test: the stabilization -- routing the ratio gradient through the bounded d_ratio*ratio
        in log-space, and forming ratio = exp(log_g_last - log_g) instead of the quotient
        G_last/G_s -- was applied to the kernel and to the reference as SEPARATE edits, so a
        reference-only test leaves the kernel's half unguarded. Negative control: reverting the
        kernel's half alone makes d_g_log non-finite here (which fails validation) at
        decay_scale >= 5, while every other case in this file still passes.

        The mild-decay regression guard is `test_gdn_cte_bwd_basic`, which shares these shapes.
        """
        self._run_test(
            test_manager,
            platform_target,
            batch,
            n_k_heads,
            n_v_heads,
            seqlen,
            key_head_dim,
            value_head_dim,
            dtype,
            decay_scale=decay_scale,
        )

    @pytest_parametrize(gdn_cte_bwd_params, test_cases_long_context, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_long_context(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
    ):
        """Many chunks at full head dims, where the per-chunk state stash dominates SBUF.

        The stash is the kernel's only seqlen-sized SBUF tile, so this is the axis that runs out
        first: `MAX_STATE_STASH_BYTES_PER_PARTITION` is what admits or rejects a length, and 64
        chunks here is an order of magnitude past the 2-3 chunks the rest of the file uses. Pairs
        with `test_seqlen_over_state_stash_budget_rejected`, which pins the other side of the
        budget -- together they show the cap admits real lengths rather than rejecting everything.

        Not marked `fast`: it is the most expensive case in the file, and the head dims themselves
        are already covered at 2 chunks by `test_gdn_cte_bwd_large`.
        """
        self._run_test(
            test_manager,
            platform_target,
            batch,
            n_k_heads,
            n_v_heads,
            seqlen,
            key_head_dim,
            value_head_dim,
            dtype,
        )

    @pytest.mark.fast
    @pytest_parametrize(gdn_cte_bwd_params, test_cases_gqa_depth, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_deep_gqa_group(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
    ):
        """Head ratios of 4 and 8: the dQ/dK head sum's read-modify-write middle case.

        The head-repeat adjoint sums `n_v_heads // n_k_heads` partial dQ/dK into one shared (q, k)
        slice, through an fp32 scratch so the partials are not rounded to the compute dtype before
        the add. At the ratio of 2 the rest of the file uses, every value head is either the first
        (writes the scratch) or the last (reads it and rounds to the output) -- these ratios are
        what reach the middle heads that do both.

        NOTE this is coverage, not a precision guard: the accumulator's benefit is real but small
        next to the bf16 floor (measured in the simulator at seqlen 256, Dk=Dv=64, dQ max abs error
        vs the oracle -- 4.8e-03 vs 6.5e-03 at ratio 4, and 6.9e-03 vs 1.2e-02 at ratio 8, against
        a 2e-2 tolerance), and the validator's combined atol+rtol metric is looser still, so no
        tolerance here separates the two forms stably enough to rely on.
        """
        self._run_test(
            test_manager,
            platform_target,
            batch,
            n_k_heads,
            n_v_heads,
            seqlen,
            key_head_dim,
            value_head_dim,
            dtype,
        )

    @pytest_parametrize(gdn_cte_bwd_params, test_cases_high_batch, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_high_batch(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        key_head_dim,
        value_head_dim,
        dtype,
    ):
        """Batch 32, where the SPMD lane loop runs 16 lanes deep per program.

        Everything else in this file runs 1-3 lanes, so the per-lane state stash, the dQ/dK writes
        and the lane-to-program arithmetic are only ever exercised shallow. Pairs with
        `test_gdn_cte_bwd_lane_sharding`, which pins the split arithmetic itself at small lane
        counts: these two shapes are the depth, those five are the edge cases. Not marked `fast` --
        the head dims and chunk counts here are already covered at batch 1.
        """
        self._run_test(
            test_manager, platform_target, batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, dtype
        )

    @pytest.mark.fast
    @pytest_parametrize("seqlen, group_size", test_cases_grouping, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_group_residual(
        self, test_manager: Orchestrator, platform_target: Platforms, seqlen, group_size
    ):
        """Chunk counts that do NOT divide by the group width, plus group_size=1.

        `group_layout` splits the chunks into full groups of `group_width` plus ONE residual group
        of the remainder, and every other case in this file has a remainder of 0 -- so without
        these the residual group is never emitted, in either pass. The backward is the harder side:
        it walks the groups in reverse, which makes the residual group the first one the state-
        gradient scan visits, and the carry has to cross that boundary as it does any other.
        group_size=1 is the ungrouped reference, and 16 is MAX_GROUP_SIZE.
        """
        self._run_test(test_manager, platform_target, 1, 1, 1, seqlen, 16, 16, nl.bfloat16, group_size=group_size)

    @pytest.mark.fast
    @pytest_parametrize("batch, n_k_heads, n_v_heads, logical_nc_config", test_cases_sharding, abbrevs=_ABBREVS)
    def test_gdn_cte_bwd_lane_sharding(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        logical_nc_config,
    ):
        """The SPMD split is over (batch x KEY head) lanes, and the result must not change.

        Sharding on the KEY head rather than the value head is what makes the dQ/dK accumulation
        safe: a shared (q, k) slice is read-modify-written across a group's value heads, so the
        whole group must live on one program. These cases pin the split arithmetic -- an even
        division, an odd lane count where the first cores take one extra, a core with no work at
        all, and a GQA lane whose value heads must travel with their key head.
        """
        self._run_test(
            test_manager,
            platform_target,
            batch,
            n_k_heads,
            n_v_heads,
            256,
            16,
            16,
            nl.bfloat16,
            logical_nc_config=logical_nc_config,
        )

    @pytest.mark.fast
    def test_group_size_is_bit_identical(self):
        """Grouping is a SCHEDULING transform: no gradient may depend on `group_size`.

        The kernel documents the grouped emission as bit-identical to group_size=1, and that is a
        stronger claim than "within tolerance" -- the reverse scan visits chunks in the same
        strictly-descending order at any group size, so the arithmetic is not merely close but the
        same operations in the same order. Checked by running one set of inputs at group_size=1 and
        at 3 (one group spanning all three chunks) and requiring EXACT equality on all six
        gradients. A tolerance-based comparison against the oracle, as every other test here uses,
        would not catch a reordering that changes accumulation.
        """
        import nki

        inputs = _generate_inputs(1, 1, 1, 384, 16, 16, nl.float32, with_state=True)
        outputs = []
        for group_size in (1, 3):
            outputs.append(
                nki.simulate(gdn_cte_bwd)(
                    inputs["query"],
                    inputs["key"],
                    inputs["value"],
                    inputs["g_log"],
                    inputs["beta"],
                    inputs["d_out"],
                    inputs["mask_pack"],
                    scale=inputs["scale"],
                    group_size=group_size,
                    recurrent_state=inputs["recurrent_state"],
                    d_state=inputs["d_state"],
                )
            )
        names = ("d_query", "d_key", "d_value", "d_g_log", "d_beta", "d_recurrent_state")
        for name, ungrouped, grouped in zip(names, outputs[0], outputs[1], strict=True):
            np.testing.assert_array_equal(
                ungrouped, grouped, err_msg=f"{name} differs between group_size=1 and group_size=3"
            )

    @pytest.mark.fast
    def test_group_size_out_of_range_rejected(self):
        """`group_size` outside [1, MAX_GROUP_SIZE] must fail at trace time.

        The bound is an SBUF/PSUM one: a group holds about seven CHUNKxCHUNK tiles per member live
        across the whole level loop, so an over-wide group spills rather than failing cleanly.
        Traced via the simulator, since the kernel_assert fires before any compute.
        """
        import nki
        from nkilib_src.nkilib.experimental.gdn.gdn_cte_bwd import MAX_GROUP_SIZE

        inputs = _generate_inputs(1, 1, 1, 256, 16, 16, nl.float32, with_state=False)
        with pytest.raises(AssertionError, match="group_size must be in"):
            nki.simulate(gdn_cte_bwd)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                inputs["d_out"],
                inputs["mask_pack"],
                scale=inputs["scale"],
                group_size=MAX_GROUP_SIZE + 1,
            )

    @pytest.mark.fast
    def test_bf16_cotangent_rejected(self):
        """A bf16 `d_out` must fail at trace time, not on a dtype-mismatched DMA in the scan.

        d_out is consumed in fp32 and feeds the fp32 gate and beta gradients, so unlike q/k/v it
        does NOT follow the compute dtype -- and bf16 is the natural dtype for an upstream gradient
        in a bf16 training step, which makes this the easiest mistake to make at the call site.
        Traced via the simulator, since the kernel_assert fires before any compute.
        """
        import nki

        inputs = _generate_inputs(1, 1, 1, 256, 16, 16, nl.bfloat16, with_state=False)
        bad_d_out = dt.static_cast(inputs["d_out"], nl.bfloat16)
        with pytest.raises(AssertionError, match="d_out must be float32"):
            nki.simulate(gdn_cte_bwd)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                bad_d_out,
                inputs["mask_pack"],
                scale=inputs["scale"],
            )

    @pytest.mark.fast
    def test_mask_pack_wrong_plane_size_rejected(self):
        """A pack with the right plane COUNT but the wrong plane size must fail at trace time.

        The count alone is what the kernel checked, so a (N_MASKS, 64, 64) pack -- the shape a
        64-chunk algorithm hands it -- passed validation and then produced wrong masks with no
        error anywhere. The forward has always checked both dimensions. Traced via the simulator,
        since the kernel_assert fires before any compute.
        """
        import nki

        inputs = _generate_inputs(1, 1, 1, 256, 16, 16, nl.float32, with_state=False)
        bad_pack = np.zeros((inputs["mask_pack"].shape[0], 64, 64), dtype=np.float32)
        with pytest.raises(AssertionError, match="mask_pack planes must be"):
            nki.simulate(gdn_cte_bwd)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                inputs["d_out"],
                bad_pack,
                scale=inputs["scale"],
            )

    @pytest.mark.fast
    def test_seqlen_over_state_stash_budget_rejected(self):
        """The forward state pass's stash is LINEAR in S, and it bounds the sequence length.

        `states_sbuf` is one (Dk, n_chunks*Dv) fp32 tile -- 4*n_chunks*Dv bytes on EVERY
        partition -- and on trn2 it fills the whole 212,984 B/partition SBUF at S=53248, Dv=128.
        The budget caps S a little under that measured wall, so an over-long sequence is rejected
        up front with the arithmetic rather than bottoming out in an allocation failure that names
        no cause. Traced via the simulator, since the kernel_assert fires before any compute.
        """
        import nki
        from nkilib_src.nkilib.experimental.gdn.gdn_cte_bwd import MAX_STATE_STASH_BYTES_PER_PARTITION

        over_budget = (MAX_STATE_STASH_BYTES_PER_PARTITION // (4 * 128) + 1) * CHUNK
        inputs = _generate_inputs(1, 1, 1, over_budget, 128, 128, nl.float32, with_state=False)
        with pytest.raises(AssertionError, match="state stash"):
            nki.simulate(gdn_cte_bwd)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                inputs["d_out"],
                inputs["mask_pack"],
                scale=inputs["scale"],
            )
