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
"""Integration tests for the DeepSeek-V4 CSA decode kernels.

The headline test is ``test_score_topk_gather_fused``, which grades the fused
megakernel end to end. That is the only way to grade it -- the indexer score, the
GpSimd top-k and the O(k) attention all happen inside one launch, so nothing in
between is observable -- and it is also what transitively covers the snake
reformat, the cross-core barrier and the ``sendrecv`` softmax merge, none of which
has an observable output of its own.

Three kernels are covered compile-only rather than numerically, and deliberately:
``nisa_topk_snake_kernel``, ``nki_indexer_score_topk_kernel`` and
``nki_indexer_score_topk_2core`` all return SELECTED POSITIONS. The kernels emit
their winners as an unordered set into row 0 of an 8-row buffer, leaving rows 1-7
undefined, and ``nisa.topk`` gives no ordering guarantee within the set -- so an
elementwise comparison against ``torch.topk`` would report differences that are not
errors. Their numerics are covered through the fused kernel, whose softmax over
gathered positions is permutation-invariant and therefore insensitive to exactly
the freedom that makes a direct comparison meaningless.

These kernels use ``priority=`` DMA class-of-service hints, which exist only on
NeuronCore-v4, so every test here is trn3-only.
"""

from typing import Any, final

import ml_dtypes
import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_decode_attention import (
    nisa_topk_snake_kernel,
    nki_decode_gather_ok_kernel,
    nki_indexer_qproj_gemv,
    nki_indexer_score_2core,
    nki_indexer_score_kernel,
    nki_indexer_score_topk_2core,
    nki_indexer_score_topk_gather_2core,
    nki_indexer_score_topk_kernel,
    nki_qkv_rms_rope_kernel,
)
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_decode_attention_torch import (
    nki_decode_gather_ok_torch_ref,
    nki_indexer_qproj_gemv_torch_ref,
    nki_indexer_score_2core_torch_ref,
    nki_indexer_score_topk_gather_2core_torch_ref,
    nki_indexer_score_torch_ref,
    nki_qkv_rms_rope_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# The `priority=` DMA hints these kernels carry are NeuronCore-v4 only.
pytestmark = pytest.mark.platforms(exclude=list(set(Platforms) - {Platforms.TRN3, Platforms.TRN3_A0}))

_INDEX_HEAD_DIM = 128
_WINDOW = 128
_S_Q = 128

_BF16 = ml_dtypes.bfloat16
_WARMUP_RUNS = 2


def _rng(seed: int = 42) -> np.random.Generator:
    """A seeded generator, so every test case is reproducible."""
    return np.random.default_rng(seed)


def _bf16(x: np.ndarray) -> np.ndarray:
    return dt.static_cast(x.astype(np.float32), nl.bfloat16)


def _f16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)


def _indexer_inputs(t_c: int, n_index_heads: int, k: int | None = None, seed: int = 42) -> dict[str, Any]:
    """Indexer scoring operands. With ``k`` given, the top-k boundary is UNAMBIGUOUS.

    Why that matters: the kernels select with ``nisa.topk`` and the reference with
    ``torch.topk``. If the k-th and (k+1)-th scores are merely *close*, the two can
    legitimately pick different positions, and a test built on smoothly-varying
    scores fails on a difference that is not an error -- which is exactly what a
    ``linspace`` of per-position gains produced here at ``T_c = 8192``, where 8192
    gains inside one bounded range sit far closer together than a bf16 ulp.

    So when ``k`` is supplied the scores are built as TWO WIDELY SEPARATED CLUSTERS
    of DISTINCT values: exactly ``k`` positions land in a high band and the rest in a
    band ~50x below it, all sharing one direction so the score is strictly
    proportional to a per-position gain. The 50x gap puts the k-th boundary far
    beyond anything rounding can cross, so the top-k is precisely the high cluster.

    Distinct *within* each band is the other half of the requirement, and it is not
    optional. Giving a whole band one shared value instead makes ``k`` positions score
    *exactly* the same, and on that degenerate tied distribution which ``k`` of the
    equal scores come back is not pinned down -- so the comparison against
    ``torch.topk`` fails on the tie, not on the kernel. Real indexer scores are a
    projection of activations and are never tied like that, so a tied input tests a
    regime the kernel never sees.

    Order within the high cluster is left arbitrary on purpose: all ``k`` of them are
    selected regardless, and the softmax over gathered positions is
    permutation-invariant. The high positions are shuffled, so a kernel that quietly
    returned "the first k" still fails.

    ``kv_t`` is the INDEXER cache and feeds scoring only; the attention gathers from
    a separate ``compress_kv`` whose rows stay distinct, so sharing the scoring
    direction costs the attention no coverage.

    With ``k`` omitted (the score-only tests, which have no selection step) each
    position gets its own gain instead, which exercises a wider score range.
    """
    rng = _rng(seed)
    q = rng.standard_normal((_INDEX_HEAD_DIM, n_index_heads)).astype(np.float32) * 0.5

    if k is None:
        gains = np.linspace(0.2, 1.8, t_c, dtype=np.float32)
        rng.shuffle(gains)
        kv = rng.standard_normal((_INDEX_HEAD_DIM, t_c)).astype(np.float32) * 0.3 * gains
    else:
        gains = rng.uniform(0.01, 0.02, t_c).astype(np.float32)
        high = rng.permutation(t_c)[:k]
        gains[high] = rng.uniform(1.0, 2.0, k).astype(np.float32)
        base = rng.standard_normal((_INDEX_HEAD_DIM, 1)).astype(np.float32) * 0.3
        kv = base * gains

    # q_T_all[d, h * S_q + s] == q[s, h, d]; all S_q rows are the same query.
    q_T_all = np.repeat(q, _S_Q, axis=1)
    weights = np.tile(rng.uniform(0.5, 1.5, (1, n_index_heads)).astype(np.float32), (_S_Q, 1))
    return {"q_T_all": _bf16(q_T_all), "kv_t": _bf16(kv), "weights": weights}


@final
@pytest_test_metadata(name="DeepSeek-V4 CSA Decode")
@pytest_marks(["attention", "deepseek_v4_csa"])
class TestCsaDecodeAttention:
    """Decode-side CSA kernels: projection tail, indexer, top-k, and O(k) attention."""

    # ---------------- fused RMS + RoPE projection tail ----------------

    _RMS_ROPE_PARAMS = "n_heads, head_dim, rope_head_dim"
    _RMS_ROPE_CASES = [
        (32, 512, 64),  # the production per-rank shape
        (8, 128, 32),
    ]
    _RMS_ROPE_ABBREVS = {"n_heads": "h", "head_dim": "d", "rope_head_dim": "rd"}

    @pytest.mark.fast
    @pytest_parametrize(_RMS_ROPE_PARAMS, _RMS_ROPE_CASES, abbrevs=_RMS_ROPE_ABBREVS)
    def test_qkv_rms_rope(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
    ):
        """The q heads and the kv row normalized and rotated on one packed tile.

        The kernel unifies the two projection tails with a per-partition gain of
        ``1.0`` on the q rows, relying on ``x * 1.0 == x`` being exact in fp32. The
        reference runs the two paths separately, so an inexact gain would show up
        here as a q-row mismatch.
        """
        rng = _rng()
        half_rope = rope_head_dim // 2
        eps = 1e-6

        def input_generator(test_config):
            return {
                "q_in": _bf16(rng.standard_normal((n_heads, head_dim)) * 0.5),
                "kv_in": _bf16(rng.standard_normal((1, head_dim)) * 0.5),
                # A gain near 1 with real spread: an all-ones gain would not
                # distinguish the kv row's learnable scaling from the q rows'.
                "weight_in": rng.uniform(0.8, 1.2, (1, head_dim)).astype(np.float32),
                "cos_in": np.cos(rng.uniform(0, 2 * np.pi, (1, half_rope))).astype(np.float32),
                "sin_in": np.sin(rng.uniform(0, 2 * np.pi, (1, half_rope))).astype(np.float32),
                "eps_val": eps,
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_heads + 1, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_qkv_rms_rope_kernel,
            torch_ref=torch_ref_wrapper(nki_qkv_rms_rope_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=1e-2,
            rtol=3e-2,
        )

    # ---------------- indexer q-projection GEMV ----------------

    _QPROJ_PARAMS = "q_lora_rank, n_index_heads, lnc"
    _QPROJ_CASES = [
        (1536, 64, 2),  # production: 12 k-tiles x 64 n-tiles, sharded over both cores
        (1536, 64, 1),  # the same weight on one core -- must agree with the sharded run
        (256, 8, 2),
    ]
    _QPROJ_ABBREVS = {"q_lora_rank": "r", "n_index_heads": "h", "lnc": "lnc"}

    @pytest.mark.fast
    @pytest_parametrize(_QPROJ_PARAMS, _QPROJ_CASES, abbrevs=_QPROJ_ABBREVS)
    def test_indexer_qproj_gemv(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        q_lora_rank: int,
        n_index_heads: int,
        lnc: int,
    ):
        """The indexer query projection as a hand-tiled GEMV, returned transposed.

        Run at both ``lnc=1`` and ``lnc=2``. The two-core launch shards the N axis,
        with each core writing a disjoint half of a NAMED ``shared_hbm`` output; the
        one-core launch writes all of it. Both are graded against the same
        reference, so a half that never landed fails rather than passing as
        plausible numbers.
        """
        rng = _rng()
        n_ktiles = q_lora_rank // 128
        n = n_index_heads * _INDEX_HEAD_DIM

        def input_generator(test_config):
            # wT[t, kk, n] == wq_b.weight[n, t * 128 + kk], the host-side pre-tiling.
            weight = rng.standard_normal((n, q_lora_rank)).astype(np.float32) / np.sqrt(q_lora_rank)
            w_t = weight.reshape(n, n_ktiles, 128).transpose(1, 2, 0)
            return {
                "wT": _bf16(np.ascontiguousarray(w_t)),
                "qr_in": _bf16(rng.standard_normal((1, q_lora_rank)) * 0.5),
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((128, n // 128), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_qproj_gemv,
            torch_ref=torch_ref_wrapper(nki_indexer_qproj_gemv_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=2e-2,
            rtol=5e-2,
        )

    # ---------------- indexer scoring ----------------

    _SCORE_PARAMS = "t_c, n_index_heads"
    _SCORE_CASES = [(1024, 64), (512, 8)]
    _SCORE_ABBREVS = {"t_c": "tc", "n_index_heads": "h"}

    @pytest.mark.fast
    @pytest_parametrize(_SCORE_PARAMS, _SCORE_CASES, abbrevs=_SCORE_ABBREVS)
    def test_indexer_score(self, test_manager: Orchestrator, platform_target: Platforms, t_c: int, n_index_heads: int):
        """Raw indexer scores plus the causal bias, one core, ``[S_q, T_c]`` out.

        The relu is applied per head BEFORE the per-head weight, so a head with a
        negative dot product contributes nothing rather than contributing
        negatively. That ordering is what makes every real score non-negative, which
        the top-k padding downstream depends on.
        """

        def input_generator(test_config):
            inputs = _indexer_inputs(t_c, n_index_heads)
            # Zero bias: decode has no causal masking to apply within the cache, and
            # a zero table is what the block passes.
            inputs["causal_bias"] = np.zeros((_S_Q, t_c), dtype=np.float32)
            return inputs

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((_S_Q, t_c), dtype=np.float32)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_score_kernel,
            torch_ref=torch_ref_wrapper(nki_indexer_score_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=5e-2,
            rtol=5e-2,
        )

    @pytest.mark.fast
    @pytest_parametrize(_SCORE_PARAMS, [(1024, 64)], abbrevs=_SCORE_ABBREVS)
    def test_indexer_score_2core(
        self, test_manager: Orchestrator, platform_target: Platforms, t_c: int, n_index_heads: int
    ):
        """Two-core scoring of disjoint ``T_c`` halves into one shared score row.

        This is the test for the shared-buffer hand-off: both cores write disjoint
        halves of a NAMED ``shared_hbm`` row, and if that allocation were anonymous
        each core would get a private copy and the returned row would carry core 0's
        half with core 1's left as zeros. The reference scores the whole range, so
        that failure shows up as a mismatch over the second half rather than as
        plausible output.
        """

        def input_generator(test_config):
            return _indexer_inputs(t_c, n_index_heads)

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((1, t_c), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_score_2core,
            torch_ref=torch_ref_wrapper(nki_indexer_score_2core_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=5e-2,
            rtol=5e-2,
        )

    # ---------------- O(k) gathered attention ----------------

    _GATHER_PARAMS = "n_heads, head_dim, rope_head_dim, t_c, k"
    _GATHER_CASES = [
        (32, 512, 64, 1024, 256),  # production head/dim shape, 2 gather chunks
        (32, 512, 64, 2048, 512),  # 4 chunks and a full-width score group
        (8, 256, 64, 512, 128),  # one chunk, and a head count well under 128
    ]
    _GATHER_ABBREVS = {"n_heads": "h", "head_dim": "d", "rope_head_dim": "rd", "t_c": "tc", "k": "k"}

    @pytest.mark.fast
    @pytest_parametrize(_GATHER_PARAMS, _GATHER_CASES, abbrevs=_GATHER_ABBREVS)
    def test_decode_gather_attention(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        t_c: int,
        k: int,
    ):
        """O(k) attention on caller-supplied indices, with the output de-RoPE fused in.

        The selected positions are SHUFFLED rather than a contiguous or monotone
        run: the indirect gather reads them in whatever order the index row holds,
        so a monotone list would hide an off-by-one in the gather's address
        arithmetic. It also passes the same position more than once nowhere, so each
        gathered row is distinguishable.
        """
        rng = _rng()
        half_rope = rope_head_dim // 2
        s_len = 1

        def input_generator(test_config):
            idx = rng.permutation(t_c)[:k].astype(np.uint32).reshape(k, s_len)
            return {
                "topk_indices_T": idx,
                # Already scaled by softmax_scale, as the block hands it over.
                "all_q_T": _f16(rng.standard_normal((head_dim, n_heads * s_len)) * (head_dim**-0.5)),
                "win_K_T": _f16(rng.standard_normal((head_dim, _WINDOW)) * 0.3),
                "win_V": _f16(rng.standard_normal((_WINDOW, head_dim)) * 0.3),
                "compress_kv": _bf16(rng.standard_normal((t_c, head_dim)) * 0.3),
                "attn_sink_in": rng.standard_normal((1, n_heads)).astype(np.float32) * 0.5,
                "derope_cos": np.cos(rng.uniform(0, 2 * np.pi, (1, half_rope))).astype(np.float32),
                "derope_sin": np.sin(rng.uniform(0, 2 * np.pi, (1, half_rope))).astype(np.float32),
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_heads * s_len, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_decode_gather_ok_kernel,
            torch_ref=torch_ref_wrapper(nki_decode_gather_ok_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=2e-2,
            rtol=5e-2,
        )

    # ---------------- the fused megakernel ----------------

    _FUSED_PARAMS = "n_heads, head_dim, rope_head_dim, n_index_heads, t_c, k, n_val"
    _FUSED_CASES = [
        # k divides 2 * COMP_CHUNK, so the kernel takes the K-SPLIT path: the
        # gathered positions are halved across the cores and the softmax is
        # recombined with two sendrecv exchanges.
        (32, 512, 64, 64, 1024, 256, 2048),
        # k does NOT divide 256, so the K-split is rejected and T_c <= 4096 selects
        # the HEAD-SPLIT path instead -- the two cores take disjoint head blocks
        # after a second core_barrier publishes the winners.
        (32, 512, 64, 64, 1024, 384, 2048),
        # The production 32K-context shape. k still divides 2 * COMP_CHUNK so this is
        # also the K-split path, but at 8x the cache: the score stage runs 8 chunks per
        # core and the top-k runs at its full pinned width with no padding tail.
        (32, 512, 64, 64, 8192, 1024, 8192),
        # 8K context, but the top-k still runs at the pinned n_val=8192 -- so three
        # QUARTERS of the score row is sentinel padding. This is what the block
        # actually traces at seq_len=8192, and it is the case where the padding
        # dominates: a top-k that mishandles a mostly-padded row is wrong here and
        # right at both of the shapes above (one has no padding, the other half).
        (32, 512, 64, 64, 2048, 1024, 8192),
    ]
    _FUSED_ABBREVS = {
        "n_heads": "h",
        "head_dim": "d",
        "rope_head_dim": "rd",
        "n_index_heads": "ih",
        "t_c": "tc",
        "k": "k",
        "n_val": "n",
    }

    @pytest.mark.fast
    @pytest_parametrize(_FUSED_PARAMS, _FUSED_CASES[:2], abbrevs=_FUSED_ABBREVS)
    def test_score_topk_gather_fused(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        n_index_heads: int,
        t_c: int,
        k: int,
        n_val: int,
    ):
        """Indexer score, GpSimd top-k and O(k) attention in ONE launch, graded end to end.

        Nothing between the three stages is observable, so this single comparison is
        what covers the snake reformat, the cross-core barrier, the ``nisa.topk``
        call and the gather that reads its winners back. The parametrization picks
        which cross-core strategy the kernel compiles to -- K-split or head-split --
        via ``k`` and ``T_c``, both trace-time constants.
        """
        self._run_fused(test_manager, platform_target, n_heads, head_dim, rope_head_dim, n_index_heads, t_c, k, n_val)

    @pytest_parametrize(_FUSED_PARAMS, _FUSED_CASES[2:], abbrevs=_FUSED_ABBREVS)
    def test_score_topk_gather_fused_large(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        n_index_heads: int,
        t_c: int,
        k: int,
        n_val: int,
    ):
        """The fused kernel at the production 32K-context cache size."""
        self._run_fused(test_manager, platform_target, n_heads, head_dim, rope_head_dim, n_index_heads, t_c, k, n_val)

    def _run_fused(self, test_manager, platform_target, n_heads, head_dim, rope_head_dim, n_index_heads, t_c, k, n_val):
        rng = _rng()
        half_rope = rope_head_dim // 2
        s_len = 1

        def input_generator(test_config):
            inputs = _indexer_inputs(t_c, n_index_heads, k=k)
            inputs.update(
                {
                    "k_val": k,
                    "n_val": n_val,
                    "all_q_T": _f16(rng.standard_normal((head_dim, n_heads * s_len)) * (head_dim**-0.5)),
                    "win_K_T": _f16(rng.standard_normal((head_dim, _WINDOW)) * 0.3),
                    "win_V": _f16(rng.standard_normal((_WINDOW, head_dim)) * 0.3),
                    "compress_kv": _bf16(rng.standard_normal((t_c, head_dim)) * 0.3),
                    "attn_sink_in": rng.standard_normal((1, n_heads)).astype(np.float32) * 0.5,
                    "derope_cos": np.cos(rng.uniform(0, 2 * np.pi, (1, half_rope))).astype(np.float32),
                    "derope_sin": np.sin(rng.uniform(0, 2 * np.pi, (1, half_rope))).astype(np.float32),
                }
            )
            return inputs

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_heads * s_len, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_score_topk_gather_2core,
            torch_ref=torch_ref_wrapper(nki_indexer_score_topk_gather_2core_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=2e-2,
            rtol=5e-2,
        )

    # ---------------- compile-only coverage of the index-returning kernels ----------------

    @pytest.mark.fast
    def test_topk_snake_compiles(self, test_manager: Orchestrator, platform_target: Platforms):
        """``nisa_topk_snake_kernel`` traces and compiles on the snake-encoded layout.

        Compile-only: the kernel returns the top-k values AND their positions, and
        ``nisa.topk`` orders neither, so an elementwise comparison against
        ``torch.topk`` would flag orderings that are equally correct.
        """
        rng = _rng()
        n_val, k_val = 2048, 64
        src_x = n_val // 16
        kernel_input = {
            "in_tensor": _bf16(rng.standard_normal((128, src_x))),
            "k_val": k_val,
            "n_val": n_val,
        }
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nisa_topk_snake_kernel,
            kernel_input_generator=lambda _: kernel_input,
            trace_only=True,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
        )

    @pytest.mark.fast
    def test_score_topk_single_core_compiles(self, test_manager: Orchestrator, platform_target: Platforms):
        """``nki_indexer_score_topk_kernel`` traces and compiles (fallback path).

        Compile-only for the same reason as above: the returned winners are an
        unordered set written into row 0 of an 8-row buffer whose other rows are
        never written.
        """
        kernel_input = dict(_indexer_inputs(2048, 64, k=128), k_val=128)
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_score_topk_kernel,
            kernel_input_generator=lambda _: kernel_input,
            trace_only=True,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
        )

    @pytest.mark.fast
    def test_score_topk_2core_compiles(self, test_manager: Orchestrator, platform_target: Platforms):
        """``nki_indexer_score_topk_2core`` traces and compiles on the ``[2]`` grid.

        Compile-only for the unordered-set reason; its scoring stage is graded
        numerically by ``test_indexer_score_2core`` and its top-k by the fused
        kernel, both of which share this code verbatim.
        """
        kernel_input = dict(_indexer_inputs(1024, 64, k=128), k_val=128, n_val=2048)
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_score_topk_2core,
            kernel_input_generator=lambda _: kernel_input,
            trace_only=True,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        )
