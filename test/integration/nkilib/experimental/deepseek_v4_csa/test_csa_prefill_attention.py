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
"""Integration tests for the DeepSeek-V4 CSA prefill kernels.

All five prefill kernels are graded numerically. Two things about the inputs are
load-bearing rather than incidental:

* The selection masks are built with CAUSAL MASKING ALREADY BAKED IN, as the
  indexer produces them: a compressed position is selectable only once every raw
  token it covers is at or before the query. ``nki_gather_csa_attn_kernel``
  truncates its compressed loop at a per-tile compile-time causal bound and relies
  on everything past that bound being ``-1e9``, while the reference attends over all
  ``T_c`` columns -- so a bound that was ever too TIGHT would show up as a mismatch.
* The window bias tables come from the library's own
  ``precompute_win_bias_parts``, so the kernel and the reference agree on which
  window position each row may attend and where its attention sink sits.

These kernels use ``priority=`` DMA class-of-service hints, which exist only on
NeuronCore-v4, so every test here is trn3-only.
"""

from typing import Any, final

import ml_dtypes
import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_common import precompute_win_bias_parts
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_prefill_attention import (
    nki_compressor_core_kernel,
    nki_fused_csa_attn_kernel,
    nki_gather_csa_attn_kernel,
    nki_indexer_score_mask_kernel,
    nki_prefill_sparse_attn_kernel,
    nki_prefill_topk_kernel,
    nki_rms_rope_kernel,
)
from nkilib_src.nkilib.experimental.deepseek_v4_csa.csa_prefill_attention_torch import (
    nki_compressor_core_torch_ref,
    nki_fused_csa_attn_torch_ref,
    nki_gather_csa_attn_torch_ref,
    nki_indexer_score_mask_torch_ref,
    nki_prefill_sparse_attn_torch_ref,
    nki_rms_rope_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

pytestmark = pytest.mark.platforms(exclude=list(set(Platforms) - {Platforms.TRN3, Platforms.TRN3_A0}))

_INDEX_HEAD_DIM = 128
_WINDOW = 128
_WIN_SIZE = 2 * _WINDOW
_NEG_INF = -1e9
_BF16 = ml_dtypes.bfloat16

# Every device test runs the kernel twice and grades the SECOND execution
# (`--save-nth-output` tracks `num_runs`), i.e. one warmup then one measured run.
#
# This is load-bearing, not boilerplate: the warmup makes the tests measure steady
# state, which is what they are for. It does mean they do NOT cover
# first-execution behaviour -- grading run 0 is a separate exercise, and dropping
# `inference_args` to save a run changes what these tests assert.
_WARMUP_RUNS = 2


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _bf16(x: np.ndarray) -> np.ndarray:
    return dt.static_cast(x.astype(np.float32), nl.bfloat16)


def _f16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)


def _hadamard(n: int) -> np.ndarray:
    """Orthonormal ``[n, n]`` Sylvester Hadamard matrix, matching the model's own."""
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h * n**-0.5


def _window_bias(s_len: int) -> tuple[np.ndarray, np.ndarray]:
    """The library's own ``[S, 2W]`` window bias tables, as numpy fp32.

    Taking them from the library rather than rebuilding them here is deliberate: the
    kernel and the reference must agree on which window column each row may attend
    and which one carries the attention sink, and rebuilding the rule twice is how
    that agreement quietly breaks.
    """
    base, sink = precompute_win_bias_parts(s_len, _WINDOW)
    return base.numpy().astype(np.float32), sink.numpy().astype(np.float32)


def _causal_selection_bias(s_len: int, t_c: int, ratio: int, split_pos: int, density: float) -> np.ndarray:
    """A ``0 / -1e9`` compressed-selection mask with causal masking baked in.

    Compressed position ``t`` pools raw tokens ``[t * ratio, (t + 1) * ratio)``, so a
    query at global position ``p`` may attend it only once ``(t + 1) * ratio - 1 <= p``.
    Among the positions that pass that test, a random ``density`` fraction is marked
    selected, mimicking the indexer's top-k. Everything else is ``-1e9``, which is
    what makes the additive mask both the selection and the causal predicate.
    """
    rng = _rng(7)
    positions = split_pos + np.arange(s_len)
    covered_by = (np.arange(t_c) + 1) * ratio - 1
    causal = covered_by[None, :] <= positions[:, None]
    chosen = rng.random((s_len, t_c)) < density
    return np.where(causal & chosen, 0.0, _NEG_INF).astype(np.float32)


@final
@pytest_test_metadata(name="DeepSeek-V4 CSA Prefill")
@pytest_marks(["attention", "deepseek_v4_csa"])
class TestCsaPrefillAttention:
    """Prefill-side CSA kernels: projection tail, compressor, indexer mask, attention."""

    # ---------------- fused RMS + RoPE projection tail ----------------

    _RMS_ROPE_PARAMS = "s_rows, head_dim, rope_head_dim, do_rms, inverse, with_gain"
    _RMS_ROPE_CASES = [
        # kv-path: learnable RMSNorm gain then forward rotation.
        (256, 512, 64, 1, 0, True),
        # q-path: per-head RMS with no gain, forward rotation.
        (256, 512, 64, 1, 0, False),
        # output de-RoPE: rotation only, inverted, no norm.
        (256, 512, 64, 0, 1, False),
        (128, 128, 32, 1, 0, True),
    ]
    _RMS_ROPE_ABBREVS = {
        "s_rows": "s",
        "head_dim": "d",
        "rope_head_dim": "rd",
        "do_rms": "rms",
        "inverse": "inv",
        "with_gain": "gain",
    }

    @pytest.mark.fast
    @pytest_parametrize(_RMS_ROPE_PARAMS, _RMS_ROPE_CASES, abbrevs=_RMS_ROPE_ABBREVS)
    def test_rms_rope(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        s_rows: int,
        head_dim: int,
        rope_head_dim: int,
        do_rms: int,
        inverse: int,
        with_gain: bool,
    ):
        """One kernel covering all three of prefill's projection tails.

        ``do_rms``, ``inverse`` and whether ``gain_in`` is supplied are trace-time
        constants, so each combination compiles to its own specialization -- which is
        why they are test parameters rather than separate tests.
        """
        rng = _rng()
        half_rope = rope_head_dim // 2

        def input_generator(test_config):
            angles = rng.uniform(0, 2 * np.pi, (s_rows, half_rope))
            return {
                "x_in": _bf16(rng.standard_normal((s_rows, head_dim)) * 0.5),
                "cos_in": np.cos(angles).astype(np.float32),
                "sin_in": np.sin(angles).astype(np.float32),
                "gain_in": rng.uniform(0.8, 1.2, (1, head_dim)).astype(np.float32) if with_gain else None,
                "eps_val": 1e-6,
                "do_rms": do_rms,
                "inverse": inverse,
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((s_rows, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_rms_rope_kernel,
            torch_ref=torch_ref_wrapper(nki_rms_rope_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=1e-2,
            rtol=3e-2,
        )

    # ---------------- compressor ----------------

    _COMPRESSOR_PARAMS = "t_c, head_dim, rope_head_dim, compress_ratio, lnc, rotate"
    _COMPRESSOR_CASES = [
        (256, 512, 64, 4, 1, 0),
        (256, 512, 64, 4, 2, 0),
        (128, 256, 64, 4, 1, 0),
        # rotate=1 is the INDEXER's compressor: head_dim 128, result rotated by an
        # orthonormal Hadamard. Both lnc values, because the rotation is per position
        # tile and must not depend on which core owns the tile.
        (256, 128, 64, 4, 1, 1),
        (256, 128, 64, 4, 2, 1),
    ]
    _COMPRESSOR_ABBREVS = {
        "t_c": "tc",
        "head_dim": "d",
        "rope_head_dim": "rd",
        "compress_ratio": "r",
        "lnc": "lnc",
        "rotate": "rot",
    }

    @pytest.mark.fast
    @pytest_parametrize(_COMPRESSOR_PARAMS, _COMPRESSOR_CASES, abbrevs=_COMPRESSOR_ABBREVS)
    def test_compressor_core(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        t_c: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        lnc: int,
        rotate: int,
    ):
        """Gated pooling over the overlapped slots, then RMSNorm, then RoPE.

        The gate softmax runs over the ``2 * compress_ratio`` slots INDEPENDENTLY PER
        CHANNEL, so the gate scores are drawn with real spread across both the slot
        and the channel axis -- a per-position-only gate would let a kernel that
        collapsed the channel axis still pass.

        Both reductions are per-position, so nothing is reduced across cores; running
        at ``lnc=2`` checks that the position tiles really are partitioned rather than
        duplicated.
        """
        rng = _rng()
        ratio2 = 2 * compress_ratio
        half_rope = rope_head_dim // 2

        def input_generator(test_config):
            angles = rng.uniform(0, 2 * np.pi, (t_c, half_rope))
            # cos/sin arrive with each pair's angle duplicated across its two channels.
            cos_rep = np.repeat(np.cos(angles), 2, axis=1).astype(np.float32)
            sin_rep = np.repeat(np.sin(angles), 2, axis=1).astype(np.float32)
            inputs = {
                # kv8/score8 are BF16: the block hands the kernel its bf16 projection
                # output directly and the kernel widens on load, so ape (the fp32 gate
                # bias) is added inside the kernel rather than by the caller.
                "kv8": _bf16(rng.standard_normal((t_c, ratio2, head_dim)) * 0.5),
                "score8": _bf16(rng.standard_normal((t_c, ratio2, head_dim)) * 1.5),
                "norm_weight": rng.uniform(0.8, 1.2, (1, head_dim)).astype(np.float32),
                "cos_rep": cos_rep,
                "sin_rep": sin_rep,
                "eps": 1e-6,
                "ape": (rng.standard_normal((ratio2, head_dim)) * 0.3).astype(np.float32),
            }
            if rotate:
                inputs["hadamard"] = _bf16(_hadamard(head_dim))
            return inputs

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((t_c, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_compressor_core_kernel,
            torch_ref=torch_ref_wrapper(nki_compressor_core_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=1e-2,
            rtol=3e-2,
        )

    # ---------------- indexer scoring + bisection selection mask ----------------

    _MASK_PARAMS = "s_q, t_c, n_index_heads, k, lnc"
    _MASK_CASES = [
        (256, 1024, 64, 256, 1),
        (256, 1024, 64, 256, 2),
        (128, 512, 8, 64, 1),
    ]
    _MASK_ABBREVS = {"s_q": "sq", "t_c": "tc", "n_index_heads": "h", "k": "k", "lnc": "lnc"}

    @pytest.mark.fast
    @pytest_parametrize(_MASK_PARAMS, _MASK_CASES, abbrevs=_MASK_ABBREVS)
    def test_indexer_score_mask(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        s_q: int,
        t_c: int,
        n_index_heads: int,
        k: int,
        lnc: int,
    ):
        """Indexer scoring plus the bisection threshold that becomes the selection mask.

        The threshold comes from a fixed 9 rounds of bisection, so it does not
        necessarily admit exactly ``k`` positions -- the reference runs the same
        bisection rather than ``torch.topk``, which is what makes the comparison a
        test of the kernel instead of a test of the algorithm's approximation.

        Each query row gets its own causal bias, so the rows have genuinely different
        valid ranges and therefore different thresholds; a kernel that computed one
        threshold for the whole tile would fail.
        """
        rng = _rng()

        def input_generator(test_config):
            q = rng.standard_normal((_INDEX_HEAD_DIM, n_index_heads, s_q)).astype(np.float32) * 0.5
            q_T_all = q.transpose(0, 1, 2).reshape(_INDEX_HEAD_DIM, n_index_heads * s_q)

            # Two widely separated score clusters, k positions high and the rest at
            # 1/20th scale, sharing one direction so scores WITHIN a cluster are
            # exactly equal. The mask is a step function of the score, so smoothly
            # varying scores make the threshold land between near-identical values and
            # a single bf16 ulp of disagreement flips a position -- a full 1e9 mask
            # error rather than a small numeric one. Equal-within-cluster scores make
            # `score >= threshold` give the same answer in kernel and reference
            # wherever the threshold lands, including when the causal mask leaves a row
            # with fewer than k selectable positions.
            gains = np.full(t_c, 0.05, dtype=np.float32)
            gains[rng.permutation(t_c)[:k]] = 1.0
            kv = (rng.standard_normal((_INDEX_HEAD_DIM, 1)).astype(np.float32) * 0.3) * gains

            # Causal bias: query row s may attend compressed position t only once
            # every token t pools sits at or before it. Rows therefore differ.
            covered_by = (np.arange(t_c) + 1) * 4 - 1
            causal = covered_by[None, :] <= np.arange(s_q)[:, None] * 4
            causal_bias = np.where(causal, 0.0, _NEG_INF).astype(np.float32)

            return {
                "q_T_all": _bf16(np.ascontiguousarray(q_T_all)),
                "kv_t": _bf16(kv),
                "weights": rng.uniform(0.5, 1.5, (s_q, n_index_heads)).astype(np.float32),
                "causal_bias": causal_bias,
                "k": k,
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((s_q, t_c), dtype=np.float32)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_indexer_score_mask_kernel,
            torch_ref=torch_ref_wrapper(nki_indexer_score_mask_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            # The mask is exactly 0 or -1e9, so any threshold disagreement is a huge
            # absolute difference -- these tolerances admit no wrong bit, they only
            # allow the -1e9 magnitude itself to compare equal.
            atol=1e-3,
            rtol=1e-3,
        )

    # ---------------- mask-predicated sparse attention ----------------

    _ATTN_PARAMS = "s_len, t_c, n_heads, head_dim, lnc"
    _ATTN_CASES = [
        (256, 512, 16, 512, 1),
        (256, 512, 32, 256, 2),
    ]
    _ATTN_ABBREVS = {"s_len": "s", "t_c": "tc", "n_heads": "h", "head_dim": "d", "lnc": "lnc"}

    @pytest.mark.fast
    @pytest_parametrize(_ATTN_PARAMS, _ATTN_CASES, abbrevs=_ATTN_ABBREVS)
    def test_fused_csa_attention(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        s_len: int,
        t_c: int,
        n_heads: int,
        head_dim: int,
        lnc: int,
    ):
        """Sparse attention over ``[window | compressed]`` with one global-max softmax.

        The window and the compressed positions share a single normalization, so the
        two contributions are directly comparable and no online rescaling is needed.
        Masking is additive before ``exp``, which is what makes an unselected position
        contribute exactly zero to both the denominator and the value sum.
        """
        rng = _rng()
        split = s_len + _WINDOW
        base, sink = _window_bias(s_len)

        def input_generator(test_config):
            total = split + t_c
            return {
                "compress_sel": _bf16(_causal_selection_bias(s_len, t_c, 4, 0, 0.3)),
                # Already scaled by softmax_scale, as the core hands it over.
                "all_q_T": _f16(rng.standard_normal((head_dim, n_heads * s_len)) * (head_dim**-0.5)),
                "all_K_T": _f16(rng.standard_normal((head_dim, total)) * 0.3),
                "all_V": _bf16(rng.standard_normal((total, head_dim)) * 0.3),
                "win_bias_base_in": base,
                "win_bias_sink_in": sink,
                "attn_sink_in": (rng.standard_normal((1, n_heads)) * 0.5).astype(np.float32),
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_heads * s_len, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_fused_csa_attn_kernel,
            torch_ref=torch_ref_wrapper(nki_fused_csa_attn_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=2e-2,
            rtol=5e-2,
        )

    # ---------------- causal-bound sparse attention ----------------

    _GATHER_PARAMS = "s_len, t_c, n_heads, head_dim, split_pos, ratio, lnc"
    _GATHER_CASES = [
        (256, 512, 32, 512, 1024, 4, 1),
        (256, 512, 32, 256, 1024, 4, 2),
        # split_pos = 0 makes the causal bound at its tightest, so the first query
        # tile reaches only the leading compressed chunk.
        (256, 512, 32, 256, 0, 4, 1),
        # A/B shape vs the sparse kernel above: matched s_len / t_c / n_heads /
        # head_dim, split_pos chosen so the causal bound spans ALL of t_c.
        (128, 2048, 128, 512, 8064, 4, 1),
        # Establish the dense kernel's slope in t_c by MEASUREMENT rather than assuming
        # linearity, at both the sequence-parallel head count (128) and the current
        # head-parallel one (32). split_pos is set so the causal bound spans all of t_c.
        (128, 4096, 128, 512, 16256, 4, 1),
        (128, 8192, 128, 512, 32640, 4, 1),
        (128, 2048, 32, 512, 8064, 4, 1),
        (128, 4096, 32, 512, 16256, 4, 1),
        (128, 8192, 32, 512, 32640, 4, 1),
        # lnc=2 so the dense A/B baseline uses BOTH LNC cores, matching the sparse
        # kernel's [2] grid. Comparing a 2-core sparse kernel against a 1-core dense
        # one would overstate the win.
        (128, 2048, 128, 512, 8064, 4, 2),
        (128, 4096, 128, 512, 16256, 4, 2),
        (128, 8192, 128, 512, 32640, 4, 2),
    ]
    _GATHER_ABBREVS = {
        "s_len": "s",
        "t_c": "tc",
        "n_heads": "h",
        "head_dim": "d",
        "split_pos": "sp",
        "ratio": "r",
        "lnc": "lnc",
    }

    @pytest.mark.fast
    @pytest_parametrize(_GATHER_PARAMS, _GATHER_CASES, abbrevs=_GATHER_ABBREVS)
    def test_gather_csa_attention(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        s_len: int,
        t_c: int,
        n_heads: int,
        head_dim: int,
        split_pos: int,
        ratio: int,
        lnc: int,
    ):
        """The same attention with the compressed loop capped at a compile-time causal bound.

        The reference deliberately attends over ALL ``T_c`` compressed columns while
        the kernel stops at its per-tile bound. That only agrees because every
        column past the bound is ``-1e9`` in the selection mask and so contributes
        exactly zero -- so if the bound were ever too tight, dropping a column that
        mattered, the two would disagree. ``split_pos=0`` is included because it makes
        the bound tightest, where an off-by-one is most likely.
        """
        rng = _rng()
        base, sink = _window_bias(s_len)

        def input_generator(test_config):
            return {
                "topk_sel_bias": _bf16(_causal_selection_bias(s_len, t_c, ratio, split_pos, 0.3)),
                "all_q_T": _f16(rng.standard_normal((head_dim, n_heads * s_len)) * (head_dim**-0.5)),
                "all_K_T_win": _f16(rng.standard_normal((head_dim, s_len + _WINDOW)) * 0.3),
                "all_V_win": _bf16(rng.standard_normal((s_len + _WINDOW, head_dim)) * 0.3),
                "compress_kv_T": _f16(rng.standard_normal((head_dim, t_c)) * 0.3),
                "compress_kv": _bf16(rng.standard_normal((t_c, head_dim)) * 0.3),
                "win_bias_base_in": base,
                "win_bias_sink_in": sink,
                "attn_sink_in": (rng.standard_normal((1, n_heads)) * 0.5).astype(np.float32),
                "split_pos": split_pos,
                "ratio": ratio,
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_heads * s_len, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_gather_csa_attn_kernel,
            torch_ref=torch_ref_wrapper(nki_gather_csa_attn_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=2e-2,
            rtol=5e-2,
        )

    # ---------------- TRUE sparse prefill (per-query indirect gather) ----------------

    _SPARSE_PARAMS = "s_len, t_c, k_val, n_heads, head_dim, q_base"
    _SPARSE_CASES = [
        (8, 2048, 1024, 128, 512, 4096),
        (8, 8192, 1024, 128, 512, 16384),
        (16, 8192, 1024, 128, 512, 16384),
        # A/B shapes vs the dense kernel below: one query tile, all 128 heads.
        # t_c is varied at fixed k to show the cost is FLAT in context length.
        (128, 2048, 1024, 128, 512, 8192),
        (128, 8192, 1024, 128, 512, 8192),
        # 256 is the tile size the block actually launches with.
        (256, 4096, 1024, 128, 512, 8192),
        (256, 8192, 1024, 128, 512, 8192),
    ]
    _SPARSE_ABBREVS = {
        "s_len": "s",
        "t_c": "tc",
        "k_val": "k",
        "n_heads": "h",
        "head_dim": "d",
        "q_base": "qb",
    }

    @pytest.mark.fast
    @pytest_parametrize(_SPARSE_PARAMS, _SPARSE_CASES, abbrevs=_SPARSE_ABBREVS)
    def test_prefill_sparse_attention(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        s_len: int,
        t_c: int,
        k_val: int,
        n_heads: int,
        head_dim: int,
        q_base: int,
    ):
        """O(k) sparse prefill: gather each query's selected rows instead of masking all T_c.

        Graded against a reference that gathers the same rows, so a wrong index, a
        wrong window slice or a mis-shared softmax denominator all show up as a
        numeric failure. ``t_c`` is varied at fixed ``k`` because the kernel's work
        must be independent of context length -- only the gather addresses change.
        """
        rng = _rng()

        def input_generator(test_config):
            idx = np.stack([rng.permutation(t_c)[:k_val] for _ in range(s_len)], axis=1).astype(np.uint32)
            return {
                "topk_idx_T": idx,
                "all_q": _f16(rng.standard_normal((s_len * n_heads, head_dim)) * (head_dim**-0.5)),
                "all_K_T_win": _f16(rng.standard_normal((head_dim, s_len + _WINDOW)) * 0.3),
                "all_V_win": _f16(rng.standard_normal((s_len + _WINDOW, head_dim)) * 0.3),
                "compress_kv": _f16(rng.standard_normal((t_c, head_dim)) * 0.3),
                "attn_sink_in": (rng.standard_normal((1, n_heads)) * 0.5).astype(np.float32),
            }

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            return {"output_0": np.zeros((n_heads * s_len, head_dim), dtype=_BF16)}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_prefill_sparse_attn_kernel,
            torch_ref=torch_ref_wrapper(nki_prefill_sparse_attn_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            inference_args=InferenceArgs(num_runs=_WARMUP_RUNS),
            atol=2e-2,
            rtol=5e-2,
        )

    # ---------------- fused on-chip per-query top-k ----------------

    _TOPK_PARAMS = "s_q, t_c, k_val, n_val"
    _TOPK_CASES = [
        (256, 4096, 1024, 8192),
        (256, 8192, 1024, 8192),
        (2048, 8192, 1024, 8192),
    ]
    _TOPK_ABBREVS = {"s_q": "sq", "t_c": "tc", "k_val": "k", "n_val": "n"}

    @pytest.mark.fast
    @pytest_parametrize(_TOPK_PARAMS, _TOPK_CASES, abbrevs=_TOPK_ABBREVS)
    def test_prefill_topk(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        s_q: int,
        t_c: int,
        k_val: int,
        n_val: int,
    ):
        """Per-query top-k positions for the sparse prefill's gather.

        ``trace_only``, because the kernel returns the k winners as an UNORDERED SET
        (nisa.topk emits each snake partition's winners in ascending POSITION order,
        not by value), so there is no elementwise oracle: a correct result is a
        permutation of ``torch.topk``'s indices, and in bf16 -- where the k-th and
        (k+1)-th scores are frequently exact ties -- not even the same set. What this
        guards is that every shape the block dispatches still compiles and allocates;
        the numeric contract on the indices is graded end-to-end by
        ``test_csa_block``'s prefill cases, which fail if a selected position is wrong.
        """
        rng = _rng()

        def input_generator(test_config):
            # Indexer-shaped scores: non-negative after the relu, with a -1e9 causal tail.
            sc = np.abs(rng.standard_normal((s_q, t_c))).astype(np.float32) * 0.5
            frontier = np.minimum((np.arange(s_q) + k_val * 4 + 1) // 4, t_c)
            sc[np.arange(t_c)[None, :] >= frontier[:, None]] = -1e9
            return {"scores": sc.astype(_BF16), "k_val": k_val, "n_val": n_val}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki_prefill_topk_kernel,
            kernel_input_generator=input_generator,
            trace_only=True,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        )
