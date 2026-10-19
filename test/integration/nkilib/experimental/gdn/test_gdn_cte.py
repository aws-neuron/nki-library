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
"""Tests for the gdn_cte prefill kernel and BOTH of its algorithms.

`gdn_cte` is the only prefill entry point -- its two algorithms are traced bodies it
inlines -- so this is the only prefill test file and everything here goes through `gdn_cte`.
"""

from typing import Any

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.gdn.gdn_cte import gdn_cte
from nkilib_src.nkilib.experimental.gdn.gdn_cte_torch import gdn_cte_torch_ref
from nkilib_src.nkilib.experimental.gdn.gdn_cte_utils import (
    ALGORITHM_NEUMANN,
    ALGORITHM_SCHUR,
    CHUNK,
    build_mask_pack,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

HEAD_DIM = 128
QUERY_SCALE = float(1.0 / np.sqrt(np.float32(HEAD_DIM)))

# INPUT KEY FOR THE MUTABLE IN/OUT STATE. ".must_alias_input" is the harness convention for
# an input a kernel aliases to an output.
STATE_INPUT_KEY = "recurrent_state.must_alias_input"


def supports_state(algorithm: str) -> bool:
    """Whether `algorithm` accepts a supplied `recurrent_state`.

    Args:
        algorithm (str): The dispatcher's algorithm selector.

    Returns:
        bool: True for the schur kernel; False for neumann, which always starts from a
            zero state and allocates its own state output, so the dispatcher rejects a
            supplied one rather than silently ignoring it.
    """
    return algorithm != ALGORITHM_NEUMANN


def requires_equal_heads(algorithm: str) -> bool:
    """Whether `algorithm` needs n_k_heads == n_v_heads (no GQA head sharing).

    Args:
        algorithm (str): The dispatcher's algorithm selector.

    Returns:
        bool: True for neumann -- its flat [B*Hv, S, D] layout gives every value head its
            own q/k rows, so the test duplicates the key/query heads on the host.
    """
    return algorithm == ALGORITHM_NEUMANN


def _generate_inputs(
    batch: int,
    n_k_heads: int,
    n_v_heads: int,
    seqlen: int,
    algorithm: str,
    dtype: Any,
    with_state: bool = True,
    key_head_dim: int = HEAD_DIM,
    value_head_dim: int = HEAD_DIM,
) -> dict:
    """Production-shaped operands for either prefill algorithm (see module docstring)."""
    rng = np.random.RandomState(42)

    def l2_normed(shape):
        arr = rng.randn(*shape).astype(np.float32)
        return arr / np.sqrt((arr * arr).sum(axis=-1, keepdims=True) + 1e-6)

    # HEADS ARE DUPLICATED FOR NEUMANN, on the host, before the call: it takes a flat
    # [B, S, D] layout with no head sharing, so the dispatcher requires Hk == Hv (see
    # `gdn_cte._dispatch_neumann`). Repeating the key/query heads lets the same GQA shape run
    # on both algorithms -- schur shares one key head across `n_v_heads // n_k_heads` value
    # heads internally, neumann is handed those values already expanded.
    query_heads = l2_normed((batch, n_k_heads, seqlen, key_head_dim))
    key_heads = l2_normed((batch, n_k_heads, seqlen, key_head_dim))
    if requires_equal_heads(algorithm) and n_k_heads != n_v_heads:
        repeats = n_v_heads // n_k_heads
        query_heads = np.repeat(query_heads, repeats, axis=1)
        key_heads = np.repeat(key_heads, repeats, axis=1)

    inputs = {
        "query": dt.static_cast(query_heads, dtype),
        "key": dt.static_cast(key_heads, dtype),
        # `value` is the conv+silu output on the real path, so O(1) magnitude.
        "value": dt.static_cast(rng.randn(batch, n_v_heads, seqlen, value_head_dim).astype(np.float32) * 0.5, dtype),
        "g_log": -np.abs(rng.randn(batch, n_v_heads, seqlen).astype(np.float32)) * 0.05,
        "beta": 1.0 / (1.0 + np.exp(-rng.randn(batch, n_v_heads, seqlen).astype(np.float32))),
        "scale": float(1.0 / np.sqrt(np.float32(key_head_dim))),
        # Trace-time constant, so each algorithm compiles to its own NEFF.
        "algorithm": algorithm,
    }
    if algorithm != ALGORITHM_NEUMANN:
        inputs["mask_pack"] = build_mask_pack()
    if with_state and supports_state(algorithm):
        # fp32 and ALIASED to output 1: the kernel updates this buffer in place. Zeros for
        # a fresh prefill.
        inputs[STATE_INPUT_KEY] = np.zeros((batch, n_v_heads, key_head_dim, value_head_dim), dtype=np.float32)
    return inputs


@torch_ref_wrapper
def _torch_ref(
    query,
    key,
    value,
    g_log,
    beta,
    scale=1.0,
    algorithm=ALGORITHM_SCHUR,
    mask_pack=None,
    group_size=4,
    recurrent_state=None,
):
    out, state_out = gdn_cte_torch_ref(
        query,
        key,
        value,
        g_log,
        beta,
        mask_pack=mask_pack,
        scale=scale,
        algorithm=algorithm,
        group_size=group_size,
        recurrent_state=recurrent_state,
    )
    return {"out": out, "state_out": state_out}


@pytest_test_metadata(name="GDN CTE")
@pytest_marks(["attention", "gdn", "gdn_cte"])
class TestGdnCte:
    gdn_cte_params = "batch, n_k_heads, n_v_heads, seqlen, algorithm, dtype"
    _ABBREVS = {
        "batch": "b",
        "n_k_heads": "nkh",
        "n_v_heads": "nvh",
        "seqlen": "seq",
        "algorithm": "algo",
        "dtype": "dt",
    }
    _ALGORITHMS = [ALGORITHM_SCHUR, ALGORITHM_NEUMANN]

    # Small shapes, both FEWER CHUNKS THAN THE DEFAULT GROUP SIZE of 8, so each runs as one
    # short group: 128 is a single chunk (the ungrouped path) and 640 is five.
    test_cases_basic = [
        (1, 1, 1, 128, ALGORITHM_SCHUR, nl.bfloat16),
        (1, 1, 1, 640, ALGORITHM_SCHUR, nl.bfloat16),
    ]

    # Batched and long shapes, both algorithms. 8192 and 12288 both split exactly into
    # groups of the default 8 (8 and 12 of them). 128 and 768 are shorter than one group
    # and so run as a single short group of 1 and 6.
    test_cases_large = (
        [(32, 1, 1, 128, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
        + [(1, 2, 4, 768, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
        + [(1, 1, 1, 8192, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
        + [(1, 2, 4, 12288, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
        # THE RESIDUAL-GROUP PATH, which nothing above reaches: at the default group size of
        # 8, these are the only shapes here whose chunk count does not divide evenly. 2176 is
        # 17 chunks (2 full groups + 1), 2304 is 18 (2 + 2), and 8320 is 65 (8 + 1) at a
        # production-ish length. 17 is prime, which is exactly the case the old
        # divisor-lowering scheme collapsed to no grouping at all.
        + [(32, 1, 1, 2176, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
        + [(32, 1, 1, 2304, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
        + [(1, 1, 1, 8320, algorithm, nl.bfloat16) for algorithm in _ALGORITHMS]
    )

    # Multiples of 64 that are NOT multiples of 128: only the neumann kernel accepts
    # these, and its partial-final-group path is what makes 64 (a single chunk, i.e. a
    # group of one) work at all.
    test_cases_neumann_only = [
        (1, 1, 1, 64, ALGORITHM_NEUMANN, nl.bfloat16),
    ]

    def _run_test(
        self,
        test_manager,
        platform_target,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        algorithm,
        dtype,
        with_state=True,
        atol=2e-2,
        rtol=2e-2,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        logical_nc_config=None,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_inputs(
                batch,
                n_k_heads,
                n_v_heads,
                seqlen,
                algorithm,
                dtype,
                with_state,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, n_v_heads, seqlen, value_head_dim), dtype=kernel_input["query"].dtype),
                "state_out": np.zeros((batch, n_v_heads, key_head_dim, value_head_dim), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_cte,
            torch_ref=_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=(
                CompilerArgs(platform_target=platform_target)
                if logical_nc_config is None
                else CompilerArgs(platform_target=platform_target, logical_nc_config=logical_nc_config)
            ),
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.fast
    @pytest_parametrize(gdn_cte_params, test_cases_basic, abbrevs=_ABBREVS)
    def test_gdn_cte_basic(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        algorithm,
        dtype,
    ):
        self._run_test(test_manager, platform_target, batch, n_k_heads, n_v_heads, seqlen, algorithm, dtype)

    @pytest_parametrize(gdn_cte_params, test_cases_large, abbrevs=_ABBREVS)
    def test_gdn_cte_large(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        algorithm,
        dtype,
    ):
        self._run_test(test_manager, platform_target, batch, n_k_heads, n_v_heads, seqlen, algorithm, dtype)

    @pytest.mark.fast
    @pytest_parametrize(gdn_cte_params, test_cases_neumann_only, abbrevs=_ABBREVS)
    def test_gdn_cte_neumann_sub_chunk_multiple(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_k_heads,
        n_v_heads,
        seqlen,
        algorithm,
        dtype,
    ):
        self._run_test(test_manager, platform_target, batch, n_k_heads, n_v_heads, seqlen, algorithm, dtype)

    @pytest.mark.fast
    @pytest_parametrize(
        "batch, n_v_heads, logical_nc_config",
        [
            (4, 1, 2),  # ONE head, four batches: only batch sharding can split this
            (2, 3, 2),  # 6 lanes on 2 cores: an even split that crosses batch boundaries
            (3, 1, 2),  # 3 lanes on 2 cores: core 0 takes 2, core 1 takes 1
        ],
        abbrevs={"batch": "b", "n_v_heads": "nvh", "logical_nc_config": "lnc"},
    )
    def test_gdn_cte_batch_sharding(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        n_v_heads,
        logical_nc_config,
    ):
        """The SPMD split is over (batch x value head) LANES, not the head axis alone.

        These cases are the ones head-only sharding could not split: a single head with
        several batches, and lane counts that divide unevenly or straddle a batch
        boundary.
        """
        self._run_test(
            test_manager,
            platform_target,
            batch,
            1,
            n_v_heads,
            256,
            ALGORITHM_SCHUR,
            nl.bfloat16,
            logical_nc_config=logical_nc_config,
        )

    @pytest.mark.fast
    @pytest_parametrize("algorithm", [ALGORITHM_SCHUR], abbrevs={"algorithm": "algo"})
    def test_gdn_cte_omitted_state(self, test_manager: Orchestrator, platform_target: Platforms, algorithm):
        """Omitting `recurrent_state` must start from zero and allocate the output.

        A DIFFERENT TRACE, not different data: with the argument absent the kernel emits
        no state load at all, memsets an SBUF tile instead, and allocates output 1 rather
        than aliasing an input. Prefill callers use this path, so it needs its own case --
        the main matrix supplies the state and therefore never traces it.
        """
        self._run_test(
            test_manager,
            platform_target,
            1,
            1,
            1,
            256,
            algorithm,
            nl.bfloat16,
            with_state=False,
        )

    @pytest.mark.fast
    @pytest_parametrize(
        "n_chunks, group_size, expected",
        [
            (96, 4, (4, 24, 0)),  # the production length: an exact split
            (11, 4, (4, 2, 3)),  # 8 chunks in full groups, then a residual of 3
            (5, 4, (4, 1, 1)),
            (17, 4, (4, 4, 1)),  # prime: the case the old lowering collapsed to group 1
            (97, 4, (4, 24, 1)),
            (3, 4, (3, 1, 0)),  # fewer chunks than the group: one short group
            (7, 1, (1, 7, 0)),  # group 1 is the ungrouped control
        ],
        abbrevs={"n_chunks": "n", "group_size": "g", "expected": "exp"},
    )
    def test_group_layout(self, n_chunks, group_size, expected):
        """The requested group size must be HONORED, with the remainder in one last group."""
        from nkilib_src.nkilib.experimental.gdn.gdn_cte_utils import group_layout

        layout = group_layout(n_chunks, group_size)
        assert layout == expected, f"group_layout({n_chunks}, {group_size}) = {layout}, expected {expected}"
        group, n_full_groups, residual = layout
        assert n_full_groups * group + residual == n_chunks, (
            f"layout {layout} does not cover {n_chunks} chunks -- the kernel would drop or re-run chunks"
        )
        assert residual < group, f"residual {residual} should have been another full group"

    @pytest.mark.fast
    def test_group_size_out_of_range_rejected(self):
        """A `group_size` above MAX_GROUP_SIZE must fail at trace time.

        The bound is SBUF/PSUM pressure -- a group holds that many (CHUNK, CHUNK) tiles live
        at once -- so an unbounded request would spill rather than run slowly. Traced via the
        simulator; the kernel_assert fires before any compute.
        """
        import nki
        from nkilib_src.nkilib.experimental.gdn.gdn_cte_schur import MAX_GROUP_SIZE

        inputs = _generate_inputs(1, 1, 1, 256, ALGORITHM_SCHUR, nl.bfloat16, with_state=False)
        with pytest.raises(AssertionError, match="group_size must be in"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                mask_pack=inputs["mask_pack"],
                scale=QUERY_SCALE,
                algorithm=ALGORITHM_SCHUR,
                group_size=MAX_GROUP_SIZE + 1,
            )

    @pytest.mark.fast
    @pytest_parametrize("seqlen", [100], abbrevs={"seqlen": "seq"})
    def test_schur_rejects_sub_chunk_multiple_seqlen(self, seqlen):
        """`algorithm="schur"` must reject any S that is not a multiple of CHUNK=128.

        Callers are expected to PAD to a multiple of 128, and this is the guard that makes
        that a contract rather than a hope. It has to be an assert: `n_chunks = S // CHUNK`
        floors, so a ragged S would process only the whole chunks and leave the trailing
        output rows unwritten -- garbage HBM, no error.

        Traced via the simulator, since the kernel_assert fires before any compute.
        """
        import nki

        inputs = _generate_inputs(1, 1, 1, seqlen, ALGORITHM_SCHUR, nl.bfloat16, with_state=False)
        with pytest.raises(AssertionError, match=f"seqlen must be a positive multiple of CHUNK={CHUNK}"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                mask_pack=inputs["mask_pack"],
                scale=QUERY_SCALE,
                algorithm=ALGORITHM_SCHUR,
            )

    @pytest.mark.fast
    @pytest_parametrize(
        "n_k_heads, n_v_heads, logical_nc_config",
        [
            (1, 4, 1),  # one core, four heads: the unsharded baseline
            (1, 4, 2),  # even split, 2 heads each
            (2, 4, 2),  # even split under GQA: each core owns whole value heads
            (1, 3, 2),  # ODD count: core 0 takes 2, core 1 takes 1
            (1, 1, 2),  # fewer heads than cores: core 1 gets no work at all
        ],
        abbrevs={"n_k_heads": "nkh", "n_v_heads": "nvh", "logical_nc_config": "lnc"},
    )
    def test_gdn_cte_head_sharding(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_k_heads,
        n_v_heads,
        logical_nc_config,
    ):
        """`gdn_cte_schur` splits value heads across the SPMD grid; the result must not change.

        Heads carry no state between them, so sharding is pure work division and every case
        here is checked against the same CPU reference as the single-core runs. The split does
        not require an even head count: the first `n_v_heads % n_shards` cores take one extra,
        so an odd count runs n+1 and n, and a core with no heads emits no work. `lnc=1` is in
        the list because a trn2 launch is a 2-program grid by DEFAULT -- without it, the
        single-core program would not be covered anywhere in this file.
        """
        self._run_test(
            test_manager,
            platform_target,
            1,
            n_k_heads,
            n_v_heads,
            256,
            ALGORITHM_SCHUR,
            nl.bfloat16,
            logical_nc_config=logical_nc_config,
        )

    @pytest.mark.fast
    def test_head_dim_above_bound_rejected(self):
        """A head dim past MAX_HEAD_DIM=128 must fail at trace time, not in allocation.

        128 is the partition bound: without the assert the failure surfaces as a backend
        message about a memset partition dimension, far from the caller's mistake.
        """
        import nki
        from nkilib_src.nkilib.experimental.gdn.gdn_cte_schur import MAX_HEAD_DIM

        inputs = _generate_inputs(
            1,
            1,
            1,
            128,
            ALGORITHM_SCHUR,
            nl.bfloat16,
            with_state=False,
            key_head_dim=MAX_HEAD_DIM + 32,
            value_head_dim=MAX_HEAD_DIM + 32,
        )
        with pytest.raises(AssertionError, match="head dims must be at most"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                mask_pack=inputs["mask_pack"],
                scale=inputs["scale"],
            )

    @pytest.mark.fast
    def test_mismatched_query_key_head_dim_rejected(self):
        """query and key must share a head dim -- they contract in k^T k and q^T k."""
        import nki

        inputs = _generate_inputs(1, 1, 1, 128, ALGORITHM_SCHUR, nl.bfloat16, with_state=False)
        narrow_key = np.ascontiguousarray(inputs["key"][..., :64])
        with pytest.raises(AssertionError, match="query and key must have the same head dim"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                narrow_key,
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                mask_pack=inputs["mask_pack"],
                scale=inputs["scale"],
            )

    @pytest.mark.fast
    def test_neumann_rejects_unequal_head_dims(self):
        """`algorithm="neumann"` with Dv != Dk must fail at trace time.

        Its flat [B, S, D] signature carries ONE head dim, so a differing Dv is read at the
        wrong stride: measured, Dk < Dv returns a Dk-wide output and Dk > Dv fails to
        compile. The schur path accepts both, which is why this is a per-algorithm guard.
        """
        import nki

        inputs = _generate_inputs(
            1,
            1,
            1,
            128,
            ALGORITHM_NEUMANN,
            nl.bfloat16,
            with_state=False,
            key_head_dim=128,
            value_head_dim=64,
        )
        with pytest.raises(AssertionError, match="requires key_head_dim == value_head_dim"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                scale=inputs["scale"],
                algorithm=ALGORITHM_NEUMANN,
            )

    @pytest.mark.fast
    def test_schur_rejects_missing_mask_pack(self):
        """`mask_pack` is optional in the signature but REQUIRED by the schur algorithm.

        It is optional so the neumann path can omit it (that path builds its constants on
        device, and an unused pack would be a 720 KB NEFF input). The schur path cannot
        synthesize the masks, so omitting it there must fail at trace time rather than
        producing a kernel that reads an absent operand.
        """
        import nki

        inputs = _generate_inputs(1, 1, 1, 128, ALGORITHM_SCHUR, nl.bfloat16, with_state=False)
        with pytest.raises(AssertionError, match="requires `mask_pack`"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                scale=inputs["scale"],
            )

    @pytest.mark.fast
    def test_neumann_rejects_gqa(self):
        """`algorithm="neumann"` under GQA must fail at trace time, not silently mis-share.

        The neumann kernel's flat [B*Hv, S, D] layout gives every value head its own q/k
        rows, so Hk < Hv has no meaning there: the reshape would pair value head 1 with
        value head 0's key rows if the dispatcher just went ahead. Traced (not run) via the
        simulator, since the kernel_assert fires before any compute.
        """
        import nki

        inputs = _generate_inputs(1, 1, 2, 64, ALGORITHM_SCHUR, nl.bfloat16, with_state=False)
        with pytest.raises(AssertionError, match="requires n_k_heads == n_v_heads"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                mask_pack=inputs["mask_pack"],
                scale=QUERY_SCALE,
                algorithm=ALGORITHM_NEUMANN,
            )

    @pytest.mark.fast
    def test_neumann_rejects_supplied_state(self):
        """`algorithm="neumann"` with a `recurrent_state` must fail, not ignore it.

        That kernel always starts from a memset-zero state and allocates its own state
        output, so it can neither read an incoming state nor alias the caller's buffer.
        Accepting the argument and dropping it would be a silent wrong answer for a
        continuation caller -- the one case where the initial state is not zeros.
        """
        import nki

        inputs = _generate_inputs(1, 1, 1, 64, ALGORITHM_SCHUR, nl.bfloat16)
        with pytest.raises(AssertionError, match="does not accept a `recurrent_state`"):
            nki.simulate(gdn_cte)(
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["g_log"],
                inputs["beta"],
                mask_pack=inputs["mask_pack"],
                scale=QUERY_SCALE,
                algorithm=ALGORITHM_NEUMANN,
                recurrent_state=inputs[STATE_INPUT_KEY],
            )
