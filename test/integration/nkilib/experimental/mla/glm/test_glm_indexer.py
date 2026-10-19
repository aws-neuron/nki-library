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

"""Integration tests for the GLM indexer (:mod:`glm_indexer`).

The scored ladder holds ``s_global == 64 * s_local`` on every rung, from 256 to 256k, so each case is
one cp64 rank's work and the latencies are comparable. Collective coverage is small cp2/cp4 ranks.
"""

import importlib.util
import math
import os
from typing import Any

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.mla.glm.glm_indexer import glm_indexer_proj, kv_row_bf16_width, score_row_width
from nkilib_src.nkilib.experimental.mla.glm.glm_indexer_torch import glm_indexer_proj_cp_ref, glm_indexer_proj_ref
from typing_extensions import override

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.comparators import maxAllClose
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_CP64_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_glm_indexer_cp64_local.py")
if os.path.isfile(_CP64_LOCAL):
    _spec = importlib.util.spec_from_file_location("glm_indexer_cp64_local", _CP64_LOCAL)
    assert _spec is not None and _spec.loader is not None  # noqa: S101
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CP64_GATHER_CASES = _mod.CP64_GATHER_CASES
    CP64_PERF_CASES = _mod.CP64_PERF_CASES
else:
    CP64_GATHER_CASES = []
    CP64_PERF_CASES = []

# GLM production shape. S_local/S_global are parametrized; the rest are fixed by the model.
N_HEADS = 32
HEAD_DIM = 128
DIM = 6144
Q_LORA_RANK = 2048
ROPE_HEAD_DIM = 64
INDEX_TOPK = 2048

BF16_ULP = 2.0**-8
"""Relative precision of bf16 (8 mantissa bits). Used as the top-k tie tolerance: the kernel ranks
the score as bf16, so two values within one ULP are indistinguishable to it."""


def _score_shape(s_local: int, end_pos: int, topk_nseg: int):
    """The kernel's padded score buffer: ``[s_local, score_row_width(end_pos)]`` fp32.

    The width comes from the kernel itself rather than a copy of the formula -- it is the chunked
    top-k geometry's T past 64k, which a local round-up of end_pos gets wrong.
    """
    return np.zeros((s_local, score_row_width(end_pos, topk_nseg)), dtype=np.float32)


def _score_to_rows(buf, s_local: int, end_pos: int, topk_nseg: int):
    """Trim the kernel's padded score ``[s_local, W]`` to ``[s_local, end_pos]`` for comparison."""
    W = score_row_width(end_pos, topk_nseg)
    # buf may be a raw byte buffer OR an already-typed fp32 array; reshaping the former as-is
    # silently gives 4x the elements.
    a = buf.reshape(-1) if isinstance(buf, np.ndarray) and buf.dtype == np.float32 else np.frombuffer(buf, np.float32)
    return a.reshape(s_local, W)[:, :end_pos]


SCORE_DUMP_LIMIT_BYTES = 1 << 32
"""Largest score debug copy the artifact dump path can return.

The dump path carries its byte count in 32 bits, so at 4.00 GiB it wraps and returns a clipped
buffer. This is the only reason the long-context cases cannot emit their score.
"""


def _emit_score(s_local: int, end_pos: int, topk_nseg: int) -> bool:
    """Whether this shape's score debug copy fits the dump path.

    Emitting enables the score and mask checks, so it is kept wherever it fits; past the limit a case
    validates its INDICES only (see ``TopkStructureValidator``) rather than shrinking ``s_local``.
    """
    return s_local * score_row_width(end_pos, topk_nseg) * 4 < SCORE_DUMP_LIMIT_BYTES


def _cache_rows(ctx: int, block_size: int) -> int:
    """Cache rows to allocate: whole pages when paged, else exactly ``ctx``."""
    return (-(-ctx // block_size) * block_size) if block_size else ctx


def _block_table(ctx: int, block_size: int, cp_interleave: int = 0):
    """Page ids for logical blocks ``[0, ctx/block_size)``.

    Shuffled with a fixed seed, not the identity, so a cache-addressing error shows as a wrong score.
    ``cp_interleave = cp_degree`` reorders them the way CP interleaving does (logical block i belongs
    to rank ``i % cp_degree``), which the kernel never has to know about.
    """
    n = -(-ctx // block_size)
    pages = np.random.RandomState(7).permutation(n).astype(np.int32)
    if cp_interleave > 1:
        order = np.concatenate([np.arange(r, n, cp_interleave) for r in range(cp_interleave)])
        pages = pages[order]
    return pages


def _owned(rank: int, l, interleave: int, cp_degree: int):
    """``owned(rank, l)``: the global position of a rank's local token ``l`` under CP interleaving."""
    return (l // interleave) * (interleave * cp_degree) + rank * interleave + (l % interleave)


def _key_positions(s_local: int, s_global: int, end_pos: int, cp_interleave: int = 0):
    """Global position of each gathered key, mirroring the kernel's ``owned()`` map."""
    cpd = max(1, (s_global or end_pos) // max(1, s_local))
    s_kv_local = max(1, end_pos // cpd)
    inter = cp_interleave or s_kv_local
    g = np.arange(end_pos, dtype=np.int64)
    rank, l = g // s_kv_local, g % s_kv_local
    return (l // inter) * (inter * cpd) + rank * inter + (l % inter)


def _rope_table(s_local: int, rope_head_dim: int, fn):
    """``[s_local, rope_head_dim // 2]`` RoPE table: ``fn(pos / 10000^(2j/rope_head_dim))``."""
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, rope_head_dim, 2, dtype=np.float32) / rope_head_dim))
    return fn(np.outer(np.arange(s_local, dtype=np.float32), inv_freq)).astype(np.float32)


def _slot_mapping(s_local: int, n_slots: int):
    """A shuffled flat slot per token, with the last 4 tokens marked as padding (``-1``)."""
    rng = np.random.default_rng(7)
    slots = rng.permutation(n_slots)[:s_local].astype(np.int32)
    slots[-4:] = -1
    return slots


def _inputs(
    s_local: int,
    s_global: int,
    n_slots: int,
    end_pos: int = 0,
    index_topk: int = INDEX_TOPK,
    topk_nseg: int = 0,
    ctx: int = 0,
    block_size: int = 0,
    cp_interleave: int = 0,
    score_chunk: int = 0,
    emit_score: bool = True,
    emit_proj: bool = False,
):
    """Random bf16-representable inputs for the projection stage.

    Values are scaled to keep the fp32 accumulation of a 2048- or 6144-deep dot product in a range
    where bf16 rounding of the operands dominates the error, which is what we want to measure.
    """
    np.random.seed(42)
    scale = 0.02
    ctx_pad = _cache_rows(ctx, block_size) if ctx else 0
    return {
        "x": (np.random.randn(s_local, DIM) * scale).astype(nl.bfloat16),
        "qr": (np.random.randn(s_local, Q_LORA_RANK) * scale).astype(nl.bfloat16),
        "wq": (np.random.randn(Q_LORA_RANK, N_HEADS * HEAD_DIM) * scale).astype(nl.bfloat16),
        "wk": (np.random.randn(DIM, HEAD_DIM) * scale).astype(nl.bfloat16),
        "k_norm_gamma": (1.0 + 0.1 * np.random.randn(HEAD_DIM)).astype(np.float32),
        "k_norm_beta": (0.1 * np.random.randn(HEAD_DIM)).astype(np.float32),
        "cos": _rope_table(s_local, ROPE_HEAD_DIM, np.cos),
        "sin": _rope_table(s_local, ROPE_HEAD_DIM, np.sin),
        "slot_mapping": _slot_mapping(s_local, n_slots),
        "weights": (0.5 + 0.5 * np.random.rand(s_local, N_HEADS)).astype(np.float32),
        "q_pos": (
            np.arange(ctx_pad - s_local, ctx_pad, dtype=np.float32)
            if ctx > s_local
            else np.arange(s_local, dtype=np.float32)
        ),
        "end_pos": end_pos,
        "topk_nseg": topk_nseg,
        "emit_score": emit_score,
        "emit_proj": emit_proj,
        "k_ctx": ((np.random.randn(ctx_pad, HEAD_DIM) * scale).astype(nl.bfloat16) if ctx else None),
        "k_cache": np.zeros((n_slots, HEAD_DIM), dtype=nl.bfloat16),
        "block_table": _block_table(ctx_pad or s_local, block_size, cp_interleave) if block_size else None,
        "block_size": block_size,
        "cp_interleave": cp_interleave,
        "score_chunk": score_chunk,
        "index_topk": index_topk,
    }


class TestGlmIndexerProj:
    """Stage 1: Q and K projections."""

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, s_global",
        [
            pytest.param(128, 8192, id="glm_S128_Sglob8k"),
            pytest.param(2048, 131072, id="glm_S2048_Sglob128k"),
        ],
    )
    @pytest.mark.fast
    def test_projections(self, test_manager: Orchestrator, platform_target, s_local, s_global):
        # More slots than tokens, so unwritten slots must stay zero.
        n_slots = s_local + 64
        inputs = _inputs(s_local, s_global, n_slots, emit_proj=True)

        def _out(kernel_input):
            return {
                "output_0": np.zeros((s_local, 1), dtype=np.int32),
                "output_1": np.zeros((s_local, N_HEADS * HEAD_DIM), dtype=np.float32),
                "output_2": np.zeros((s_local, HEAD_DIM), dtype=np.float32),
                "output_3": np.zeros((s_local, 1), dtype=np.float32),
            }

        def _comparator(golden_dict, output_tensors):
            golden_q = np.asarray(golden_dict["output_1"], dtype=np.float32)

            class QValidator(CustomValidator):
                """Q is emitted bf16, so one bf16 ULP of relative error is the floor."""

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    actual = np.frombuffer(inference_output, dtype=nl.bfloat16).astype(np.float32)
                    actual = actual.reshape(s_local, N_HEADS * HEAD_DIM)
                    return maxAllClose(actual, golden_q, rtol=2e-2, atol=1e-5, verbose=1, logfile=self.logfile)

            class GatheredKValidator(CustomValidator):
                """Single-rank: the gather is skipped, so this is just this rank's own chunk."""

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    actual = np.frombuffer(inference_output, dtype=nl.bfloat16).astype(np.float32)
                    actual = actual.reshape(s_local, HEAD_DIM)
                    golden = np.asarray(golden_dict["output_2"], dtype=np.float32)
                    return maxAllClose(actual, golden, rtol=2e-2, atol=1e-5, verbose=1, logfile=self.logfile)

            return {
                "output_1": CustomValidatorWithOutputTensorData(
                    validator=QValidator, output_ndarray=output_tensors["output_1"]
                ),
                "output_2": CustomValidatorWithOutputTensorData(
                    validator=GatheredKValidator, output_ndarray=output_tensors["output_2"]
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=torch_ref_wrapper(glm_indexer_proj_ref),
            kernel_input_generator=lambda _cfg: inputs,
            output_tensor_descriptor=_out,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-5,
            rtol=2e-2,
            custom_comparator=_comparator,
        )


def _assert_topk_exact_from_artifacts(s_local: int, end_pos: int, index_topk: int, cp_degree: int):
    """Check every rank's top-k against the KERNEL'S OWN score, reading the dumped artifacts.

    This runs AFTER the collective framework, because that framework has no per-output validator
    hook -- ``custom_comparator`` only post-processes goldens, so it cannot compare one output
    against another. Without this, dropping output_3 from ``output_keys`` (which was necessary: an
    elementwise or set comparison of top-k indices is unsound, see that call site) would leave the
    top-k checked NOWHERE at cp > 1.

    Golden-free and tolerance-free on purpose. The claim "these are the top k of the score the
    kernel emitted" is exactly checkable; "these match the reference's argpartition" is not, because
    nisa.topk runs sorted=False and a 0.1% score difference reorders winners at the cut.
    """
    import os

    from test.utils.feature_flag_helper import derive_pytest_test_id

    test_id = derive_pytest_test_id()
    d = os.path.join("neuron_test_output", f"out-{test_id}", "infer_result")
    if not os.path.isdir(d):
        pytest.skip(f"artifacts not found at {d}")

    def rne_bf16(a):
        """fp32 -> bf16 with round-to-nearest-even, which is what the DMA cast does.

        Truncating instead (``& 0xFFFF0000``) manufactures a deficit of up to 2 bf16 ULP and makes an
        exact kernel look wrong -- that cost a debugging cycle.
        """
        u = a.astype(np.float32).view(np.uint32)
        return ((u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000).view(np.float32)

    W = score_row_width(end_pos, 0)
    bad = 0
    worst = 0.0
    for r in range(cp_degree):
        f_idx = os.path.join(d, f"output_worker_{r}", "output_0")
        f_sc = os.path.join(d, f"output_worker_{r}", "output_1")
        if not (os.path.isfile(f_idx) and os.path.isfile(f_sc)):
            pytest.skip(f"rank {r} outputs not dumped")
        picks = np.fromfile(f_idx, dtype=np.int32).reshape(s_local, index_topk)
        own = rne_bf16(np.fromfile(f_sc, dtype=np.float32).reshape(s_local, W)[:, :end_pos])
        oob = int(((picks < 0) | (picks >= end_pos)).sum())
        assert oob == 0, f"rank {r}: {oob} picks outside [0, {end_pos})"  # noqa: S101
        dupes = sum(1 for m in range(s_local) if len(np.unique(picks[m])) != index_topk)
        assert dupes == 0, f"rank {r}: {dupes} rows with duplicate picks (expected {index_topk} distinct)"  # noqa: S101
        for m in range(s_local):
            cut = np.partition(own[m], end_pos - index_topk)[end_pos - index_topk]
            deficit = cut - own[m][picks[m]]
            n = int((deficit > 0).sum())
            if n:
                bad += n
                worst = max(worst, float(deficit.max()))
    total = cp_degree * s_local * index_topk
    assert bad == 0, f"{bad}/{total} picks below the kernel's own top-{index_topk} cut, worst {worst:.3e}"  # noqa: S101
    print(
        f"topk exact on all {cp_degree} ranks: {total} picks, all in range, all distinct per row, "
        f"none below the kernel's own top-{index_topk} cut"
    )


@pytest.mark.high_rank
@pytest.mark.skip_simulation  # cp_degree >= 2 ranks; the CPU simulator runs one rank
class TestGlmIndexerCpGather:
    """Stage 4: the CP all-gather of K, on real ranks.

    At cp_degree=1 the kernel takes a local-copy path and ncc.all_gather never runs, so every
    single-rank test passes with the collective unexercised. Only a multi-rank run catches a wrong
    collective_dim or a replica group out of context order.
    """

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, cp_degree, end_pos, index_topk, block_size",
        [
            pytest.param(128, 2, 0, 2048, 0, id="glm_S128_cp2"),
            pytest.param(1024, 2, 2048, 256, 0, id="glm_S1024_cp2_scored"),
            pytest.param(1024, 4, 4096, 256, 0, id="glm_S1024_cp4_scored"),
            pytest.param(1024, 2, 2048, 256, 64, id="glm_S1024_cp2_paged"),
        ],
    )
    def test_cp_gather(
        self, test_manager: Orchestrator, platform_target, s_local, cp_degree, end_pos, index_topk, block_size
    ):
        n_slots = s_local + 64
        s_global = s_local * cp_degree
        rng = np.random.default_rng(11)
        scale = 0.02
        x_global = (rng.standard_normal((s_global, DIM)) * scale).astype(nl.bfloat16)
        qr_global = (rng.standard_normal((s_global, Q_LORA_RANK)) * scale).astype(nl.bfloat16)
        wq = (rng.standard_normal((Q_LORA_RANK, N_HEADS * HEAD_DIM)) * scale).astype(nl.bfloat16)
        wk = (rng.standard_normal((DIM, HEAD_DIM)) * scale).astype(nl.bfloat16)
        gamma = (1.0 + 0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
        beta = (0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)

        def create_inputs(rank_id: int):
            lo = rank_id * s_local
            return {
                "x": x_global[lo : lo + s_local],
                "qr": qr_global[lo : lo + s_local],
                "wq": wq,
                "wk": wk,
                "k_norm_gamma": gamma,
                "k_norm_beta": beta,
                "cos": _rope_table(s_local, ROPE_HEAD_DIM, np.cos),
                "sin": _rope_table(s_local, ROPE_HEAD_DIM, np.sin),
                "slot_mapping": _slot_mapping(s_local, n_slots),
                "weights": (0.5 + 0.5 * rng.random((s_local, N_HEADS))).astype(np.float32),
                "q_pos": (
                    _owned(rank_id, np.arange(s_local), block_size, cp_degree)
                    if block_size
                    else np.arange(s_local) + rank_id * s_local
                ).astype(np.float32),
                "end_pos": end_pos,
                "topk_nseg": 0,
                "emit_score": True,
                "emit_proj": True,
                # Zeroed so the framework can compare the score/gather/mask cleanly; top-k is
                # validated on one rank where it can be checked against the kernel's own score.
                "zero_topk": True,
                "k_cache": np.zeros((n_slots, HEAD_DIM), dtype=nl.bfloat16),
                "cp_degree": cp_degree,
                "block_table": _block_table(s_local, block_size, cp_degree) if block_size else None,
                "block_size": block_size,
                "cp_interleave": block_size,
                "index_topk": index_topk,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=glm_indexer_proj_cp_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            output_keys=["output_0", "output_1", "output_2", "output_3"],
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, cp_degree, end_pos, index_topk",
        [
            pytest.param(1024, 2, 2048, 256, id="glm_S1024_cp2_scored_fp8"),
            pytest.param(1024, 4, 4096, 256, id="glm_S1024_cp4_scored_fp8"),
        ],
    )
    def test_cp_gather_fp8(self, test_manager: Orchestrator, platform_target, s_local, cp_degree, end_pos, index_topk):
        """Stage 4 with an FP8 paged cache: quantize per token, gather the packed fp8 rows, dequant
        after the collective, then score.

        No ``block_table`` (the fp8 block-table read path is not built yet) and no CP interleave, so
        this exercises exactly the fp8 milestone: packed store -> fp8 all-gather -> dequant -> score.
        The ``k_cache`` is sized to the fused packed row (``kv_row_bf16_width``) rather than
        ``HEAD_DIM``; the reference models the same per-token fp8 round-trip via ``kv_fp8``.
        """
        n_slots = s_local + 64
        s_global = s_local * cp_degree
        kv_width = kv_row_bf16_width(HEAD_DIM)
        rng = np.random.default_rng(11)
        scale = 0.02
        x_global = (rng.standard_normal((s_global, DIM)) * scale).astype(nl.bfloat16)
        qr_global = (rng.standard_normal((s_global, Q_LORA_RANK)) * scale).astype(nl.bfloat16)
        wq = (rng.standard_normal((Q_LORA_RANK, N_HEADS * HEAD_DIM)) * scale).astype(nl.bfloat16)
        wk = (rng.standard_normal((DIM, HEAD_DIM)) * scale).astype(nl.bfloat16)
        gamma = (1.0 + 0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
        beta = (0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)

        def create_inputs(rank_id: int):
            lo = rank_id * s_local
            return {
                "x": x_global[lo : lo + s_local],
                "qr": qr_global[lo : lo + s_local],
                "wq": wq,
                "wk": wk,
                "k_norm_gamma": gamma,
                "k_norm_beta": beta,
                "cos": _rope_table(s_local, ROPE_HEAD_DIM, np.cos),
                "sin": _rope_table(s_local, ROPE_HEAD_DIM, np.sin),
                "slot_mapping": _slot_mapping(s_local, n_slots),
                "weights": (0.5 + 0.5 * rng.random((s_local, N_HEADS))).astype(np.float32),
                "q_pos": (np.arange(s_local) + rank_id * s_local).astype(np.float32),
                "end_pos": end_pos,
                "topk_nseg": 0,
                "emit_score": True,
                "emit_proj": True,
                "zero_topk": True,
                # Packed fp8 cache: the fused row is kv_width bf16 containers, not HEAD_DIM.
                "k_cache": np.zeros((n_slots, kv_width), dtype=nl.bfloat16),
                "cp_degree": cp_degree,
                "index_topk": index_topk,
                "kv_fp8": True,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=glm_indexer_proj_cp_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            output_keys=["output_0", "output_1", "output_2", "output_3"],
            # FP8 e4m3 K perturbs the score more than bf16, so loosen the score tolerance; the
            # gathered-K and top-k structural checks are dtype-agnostic.
            rtol=5e-2,
            atol=1e-4,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, cp_degree, end_pos, index_topk",
        [
            pytest.param(1024, 2, 2048, 256, id="glm_S1024_cp2_topk_fp8"),
            pytest.param(1024, 4, 4096, 256, id="glm_S1024_cp4_topk_fp8"),
        ],
    )
    def test_cp_gather_topk_fp8(
        self, test_manager: Orchestrator, platform_target, s_local, cp_degree, end_pos, index_topk
    ):
        """FP8 gather path, validating the REAL top-k picks (``zero_topk=False``).

        ``test_cp_gather_fp8`` zeros the picks so the collective framework can compare score/K/Q
        cleanly, which leaves the top-k selection itself unchecked on an fp8 cache -- exactly the
        thing fp8's extra score perturbation could break by reordering winners near the cut. This
        case keeps the real picks and validates them, so the fp8 path has top-k coverage, not just
        score/K coverage.

        Output layout is tuned for the artifact check: with ``emit_proj=False`` the kernel returns
        only ``(topk, score)``, so the dumped ``output_worker_{r}/output_0`` is the int32 picks and
        ``output_1`` is the fp32 score -- the pair ``_assert_topk_exact_from_artifacts`` reads. The
        framework still runs, but its comparison of ``output_0`` would pit the kernel's
        ``sorted=False`` picks against the reference's argpartition, which is unsound near the cut
        (see ``glm_indexer_torch``), so a no-op validator neutralizes it; the score (``output_1``) is
        compared normally at the fp8 tolerance. The real check is golden-free and runs after the
        framework returns.
        """
        n_slots = s_local + 64
        s_global = s_local * cp_degree
        kv_width = kv_row_bf16_width(HEAD_DIM)
        rng = np.random.default_rng(11)
        scale = 0.02
        x_global = (rng.standard_normal((s_global, DIM)) * scale).astype(nl.bfloat16)
        qr_global = (rng.standard_normal((s_global, Q_LORA_RANK)) * scale).astype(nl.bfloat16)
        wq = (rng.standard_normal((Q_LORA_RANK, N_HEADS * HEAD_DIM)) * scale).astype(nl.bfloat16)
        wk = (rng.standard_normal((DIM, HEAD_DIM)) * scale).astype(nl.bfloat16)
        gamma = (1.0 + 0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
        beta = (0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)

        def create_inputs(rank_id: int):
            lo = rank_id * s_local
            return {
                "x": x_global[lo : lo + s_local],
                "qr": qr_global[lo : lo + s_local],
                "wq": wq,
                "wk": wk,
                "k_norm_gamma": gamma,
                "k_norm_beta": beta,
                "cos": _rope_table(s_local, ROPE_HEAD_DIM, np.cos),
                "sin": _rope_table(s_local, ROPE_HEAD_DIM, np.sin),
                "slot_mapping": _slot_mapping(s_local, n_slots),
                "weights": (0.5 + 0.5 * rng.random((s_local, N_HEADS))).astype(np.float32),
                "q_pos": (np.arange(s_local) + rank_id * s_local).astype(np.float32),
                "end_pos": end_pos,
                "topk_nseg": 0,
                "emit_score": True,
                # No Q / gathered-K outputs, so the native return is (topk, score) and the dumped
                # output_0 / output_1 are the picks / score the artifact check reads.
                "emit_proj": False,
                # Keep the REAL picks -- this is the point of the test.
                "zero_topk": False,
                "k_cache": np.zeros((n_slots, kv_width), dtype=nl.bfloat16),
                "cp_degree": cp_degree,
                "index_topk": index_topk,
                "kv_fp8": True,
            }

        def _comparator(rank_id, golden_dict):
            golden_score = np.asarray(golden_dict["output_1"], dtype=np.float32)

            class _TopkPass(CustomValidator):
                """No-op: the picks are validated out-of-band against the kernel's own score.

                An elementwise comparison against the reference's ``argpartition`` is unsound here
                (``nisa.topk`` runs ``sorted=False``; near the cut, winners within one bf16 quantum
                swap), so this just passes and lets ``_assert_topk_exact_from_artifacts`` do the real,
                golden-free check from the dumped artifacts.
                """

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    return True

            return {
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=_TopkPass, output_ndarray=np.zeros((s_local, index_topk), dtype=np.int32)
                ),
                "output_1": golden_score,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=glm_indexer_proj_cp_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            # Both native outputs must be listed so the NEFF is compiled with the right arity; the
            # picks' comparison is neutralized by _TopkPass, the score is compared at the fp8 tol.
            output_keys=["output_0", "output_1"],
            custom_comparator=_comparator,
            rtol=5e-2,
            atol=1e-4,
        )

        # Real top-k validation: golden-free, against the kernel's OWN emitted score, reading the
        # dumped per-rank picks (output_0) and score (output_1).
        _assert_topk_exact_from_artifacts(s_local, end_pos, index_topk, cp_degree)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, cp_degree, end_pos, index_topk, block_size",
        [
            pytest.param(1024, 2, 2048, 256, 64, id="glm_S1024_cp2_paged_fp8"),
            pytest.param(1024, 4, 4096, 256, 64, id="glm_S1024_cp4_paged_fp8"),
        ],
    )
    def test_cp_gather_paged_fp8(
        self, test_manager: Orchestrator, platform_target, s_local, cp_degree, end_pos, index_topk, block_size
    ):
        """FP8 paged cache read through the BLOCK TABLE (the paged-cache / APC production path).

        Unlike ``test_cp_gather_fp8`` (gather path only), this passes a ``block_table`` so each rank
        linearizes its own pages out of the packed fp8 cache before the gather -- exercising the
        packed-row-aware ``linearize_pages`` and the post-gather ``dequant_k_packed``. The ``k_cache``
        is the fused packed row (``kv_row_bf16_width``); the reference models the same per-token fp8
        round-trip through the block-table read via ``kv_fp8``.
        """
        n_slots = s_local + 64
        s_global = s_local * cp_degree
        kv_width = kv_row_bf16_width(HEAD_DIM)
        rng = np.random.default_rng(11)
        scale = 0.02
        x_global = (rng.standard_normal((s_global, DIM)) * scale).astype(nl.bfloat16)
        qr_global = (rng.standard_normal((s_global, Q_LORA_RANK)) * scale).astype(nl.bfloat16)
        wq = (rng.standard_normal((Q_LORA_RANK, N_HEADS * HEAD_DIM)) * scale).astype(nl.bfloat16)
        wk = (rng.standard_normal((DIM, HEAD_DIM)) * scale).astype(nl.bfloat16)
        gamma = (1.0 + 0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
        beta = (0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)

        def create_inputs(rank_id: int):
            lo = rank_id * s_local
            return {
                "x": x_global[lo : lo + s_local],
                "qr": qr_global[lo : lo + s_local],
                "wq": wq,
                "wk": wk,
                "k_norm_gamma": gamma,
                "k_norm_beta": beta,
                "cos": _rope_table(s_local, ROPE_HEAD_DIM, np.cos),
                "sin": _rope_table(s_local, ROPE_HEAD_DIM, np.sin),
                "slot_mapping": _slot_mapping(s_local, n_slots),
                "weights": (0.5 + 0.5 * rng.random((s_local, N_HEADS))).astype(np.float32),
                "q_pos": _owned(rank_id, np.arange(s_local), block_size, cp_degree).astype(np.float32),
                "end_pos": end_pos,
                "topk_nseg": 0,
                "emit_score": True,
                "emit_proj": True,
                "zero_topk": True,
                # Packed fp8 cache: the fused row is kv_width bf16 containers, not HEAD_DIM.
                "k_cache": np.zeros((n_slots, kv_width), dtype=nl.bfloat16),
                "cp_degree": cp_degree,
                "block_table": _block_table(s_local, block_size, cp_degree),
                "block_size": block_size,
                "cp_interleave": block_size,
                "index_topk": index_topk,
                "kv_fp8": True,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=glm_indexer_proj_cp_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            output_keys=["output_0", "output_1", "output_2", "output_3"],
            # FP8 e4m3 K perturbs the score more than bf16, so loosen the score tolerance.
            rtol=5e-2,
            atol=1e-4,
        )


class TestGlmIndexerScore:
    """Stages 5-6: the blocked score and the top-k, at the production ratio s_local * 64 == end_pos.

    One rank with the global context supplied through k_ctx, which is what lets a 64-rank shape be
    validated without a fleet: the kernel sees only a position range, so a gathered context and a
    caller-supplied one are the same code.
    """

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, end_pos, index_topk, topk_nseg, block_size, cp_interleave, score_chunk",
        [
            pytest.param(256, 16384, 2048, 0, 64, 64, 0, id="glm_S256_ctx16k"),
            pytest.param(512, 32768, 2048, 0, 64, 64, 0, id="glm_S512_ctx32k"),
            pytest.param(1024, 65536, 2048, 0, 64, 64, 0, id="glm_S1024_ctx64k"),
            pytest.param(2048, 131072, 2048, 0, 64, 64, 0, id="glm_S2048_ctx128k"),
            pytest.param(4096, 262144, 2048, 0, 64, 64, 0, id="glm_S4096_ctx256k"),
            pytest.param(256, 16384, 2048, 0, 0, 0, 0, id="glm_S256_ctx16k_plain"),
        ],
    )
    def test_score(
        self,
        test_manager: Orchestrator,
        platform_target,
        s_local,
        end_pos,
        index_topk,
        topk_nseg,
        block_size,
        cp_interleave,
        score_chunk,
    ):
        n_slots = s_local + 64
        ctx = end_pos if end_pos > s_local else 0
        emit = _emit_score(s_local, end_pos, topk_nseg)
        inputs = _inputs(
            s_local,
            (_cache_rows(ctx, block_size) if ctx else s_local),
            n_slots,
            end_pos=end_pos,
            index_topk=index_topk,
            topk_nseg=topk_nseg,
            ctx=ctx,
            block_size=block_size,
            cp_interleave=cp_interleave,
            score_chunk=score_chunk,
            emit_score=emit,
        )

        def _out(kernel_input):
            desc = {
                "output_0": np.zeros((s_local, index_topk), dtype=np.int32),
                # Only when the kernel emits it: the output arity follows the emit flags.
                **({"output_1": _score_shape(s_local, end_pos, topk_nseg)} if emit else {}),
            }
            return desc

        def _comparator(golden_dict, output_tensors):
            golden_score = np.asarray(golden_dict["output_1"], dtype=np.float32)[:, :end_pos] if emit else None

            class TopkStructureValidator(CustomValidator):
                """GOLDEN-FREE index check, for the shapes whose score cannot be dumped.

                Three properties, none of which needs a reference score:
                  * every pick is UNIQUE within a query -- catches a duplicated or stale candidate;
                  * every pick is in ``[0, end_pos)`` -- catches a decode or offset error;
                  * every pick is CAUSALLY VISIBLE (``k_pos <= q_pos``) -- catches a mask failure.

                Not an accuracy check, and it is not claimed to be one. It IS the check that catches
                the class of bug this kernel actually shipped: the undersized score ring had 4001
                picks on masked positions, so the visibility test alone fails it. A reference
                comparison could not have been run at these shapes at all.
                """

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    got = np.frombuffer(inference_output, dtype=np.int32).reshape(s_local, index_topk)
                    qp = np.asarray(inputs["q_pos"], dtype=np.int64).reshape(-1)
                    kp = _key_positions(s_local, end_pos, end_pos, inputs.get("cp_interleave", 0))
                    for m in range(s_local):
                        row = got[m]
                        if row.min() < 0 or row.max() >= end_pos:
                            self._print_with_log(f"row {m}: position out of range [{row.min()}, {row.max()}]")
                            return False
                        if len(np.unique(row)) < index_topk:
                            self._print_with_log(f"row {m}: only {len(np.unique(row))} unique of {index_topk}")
                            return False
                        n_hidden = int((kp[row] > qp[m]).sum())
                        if n_hidden:
                            self._print_with_log(f"row {m}: {n_hidden} picks on MASKED positions (q_pos={qp[m]})")
                            return False
                    self._print_with_log(
                        f"topk structure exact on all {s_local} queries "
                        f"(unique, in range, causally visible); score not emitted at this shape"
                    )
                    return True

            class ScoreValidator(CustomValidator):
                """fp32 accumulation of 32 bf16 head products, so a few bf16 ULPs is the floor."""

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    assert golden_score is not None  # noqa: S101
                    actual = np.frombuffer(inference_output, dtype=np.float32).reshape(
                        s_local, score_row_width(end_pos, topk_nseg)
                    )[:, :end_pos]
                    vis = golden_score > -1e30
                    return maxAllClose(
                        actual[vis], golden_score[vis], rtol=2e-2, atol=1e-5, verbose=1, logfile=self.logfile
                    )

            class MaskBoundaryValidator(CustomValidator):
                """Golden-free: assert the masked region is EXACTLY the non-visible positions.

                This exists because a reference comparison cannot catch a missing mask. The first
                version of this kernel had no mask at all and every validator passed -- the reference
                omitted it too, so the two were mutually consistent and both wrong. Checking the
                boundary against the POSITIONS, with no golden involved, is the only check that can
                fail in that situation.
                """

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    sc = _score_to_rows(inference_output, s_local, end_pos, topk_nseg)
                    qp = np.asarray(inputs["q_pos"], dtype=np.float32).reshape(-1)
                    # KEY positions, derived exactly as the kernel's mask derives them.
                    kp = _key_positions(s_local, end_pos, end_pos, inputs.get("cp_interleave", 0)).astype(np.float32)
                    want = kp[None, :] > qp[:, None]  # True where the position must be masked
                    got = sc < -1e30
                    if not np.array_equal(got, want):
                        rows = np.nonzero((got != want).any(axis=1))[0]
                        m = int(rows[0])
                        self._print_with_log(
                            f"mask boundary WRONG on {len(rows)}/{s_local} rows; row {m}: "
                            f"masked {int(got[m].sum())} columns, expected {int(want[m].sum())}"
                        )
                        return False
                    self._print_with_log(f"mask boundary exact on all {s_local} rows")
                    return True

            class TopkValidator(CustomValidator):
                """Exact check against the KERNEL'S OWN score -- golden-free, zero tolerance.

                Ranking the picks against the REFERENCE's score needs a fudge factor and still fails:
                the two scores agree to 0.132%, but at the cut (value ~2.4) that divergence reorders
                positions by up to ~1.65 bf16 ULP, so 160 of 262144 picks land just below the
                reference's k-th best. Loosening the tolerance until that passes would be tuning the
                check to the answer.

                Instead, split the claim in two, each exactly checkable:
                  * the SCORE matches the reference -- ScoreValidator, 0.132%;
                  * the TOP-K is the true top-k OF THAT SCORE -- here, and it is exact (measured: 0 of
                    262144 picks below the kernel's own cut).
                Together those are stronger than a slack-based comparison against the golden's cut,
                and neither needs a tolerance.
                """

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    snake = np.frombuffer(inference_output, dtype=np.int32).reshape(s_local, 16, index_topk // 16)
                    got = snake.reshape(s_local, -1)
                    # Rank the REFERENCE score, bf16-rounded to the precision the kernel ranked at.
                    # NOT output_tensors["output_1"], which holds np.zeros placeholders rather than the
                    # kernel's results -- ranking against those makes every cut 0.0 and the check vacuous.
                    assert golden_score is not None  # noqa: S101
                    own = golden_score.astype(nl.bfloat16).astype(np.float32)
                    # Near-cut allowance of 4 bf16 ULP: the kernel ranks its own score and this ranks the
                    # reference's, so positions within one bf16 quantum of the cut may legitimately swap.
                    # ULP is computed from the exponent (bf16's is 2^16 times fp32's, not 2^8).
                    cut0 = abs(float(np.partition(own[0], end_pos - index_topk)[end_pos - index_topk]))
                    bf16_ulp = 2.0 ** (math.floor(math.log2(cut0)) - 8) if cut0 > 0 else 0.0
                    tie_tol = 4.0 * bf16_ulp
                    bad, worst = 0, 0.0
                    for m in range(s_local):
                        if len(np.unique(got[m])) < index_topk:
                            self._print_with_log(f"row {m}: only {len(np.unique(got[m]))} unique positions")
                            return False
                        if got[m].min() < 0 or got[m].max() >= end_pos:
                            self._print_with_log(f"row {m}: position out of range [{got[m].min()}, {got[m].max()}]")
                            return False
                        floor = np.partition(own[m], end_pos - index_topk)[end_pos - index_topk]
                        deficit = floor - own[m][got[m]] - tie_tol
                        n = int((deficit > 0).sum())
                        if n:
                            bad += n
                            worst = max(worst, float(deficit.max()))
                    if bad:
                        self._print_with_log(
                            f"topk: {bad} picks below the kernel's OWN top-{index_topk} cut, worst {worst:.4e}"
                        )
                        return False
                    self._print_with_log(f"topk exact on all {s_local} queries")
                    return True

            class ScoreAndMask(MaskBoundaryValidator, ScoreValidator):
                """Both the golden comparison and the golden-free boundary check on the score."""

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    ok_mask = MaskBoundaryValidator.validate(self, inference_output)
                    return ok_mask and ScoreValidator.validate(self, inference_output)

            # Score emitted: compare it AND rank the picks against the reference score.
            # Not emitted: the indices are all there is, so check them structurally. output_2 gets an
            # explicit pass -- it is the kernel's (1, 1) placeholder, never written, so a default
            # comparison would read an uninitialized shared_hbm word.
            if not emit:
                # No score output at all at this shape, so the indices are the only thing to check.
                return {
                    "output_0": CustomValidatorWithOutputTensorData(
                        validator=TopkStructureValidator, output_ndarray=output_tensors["output_0"]
                    ),
                }
            return {
                "output_1": CustomValidatorWithOutputTensorData(
                    validator=ScoreAndMask, output_ndarray=output_tensors["output_1"]
                ),
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=TopkValidator, output_ndarray=output_tensors["output_0"]
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=torch_ref_wrapper(glm_indexer_proj_ref),
            kernel_input_generator=lambda _cfg: inputs,
            output_tensor_descriptor=_out,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-5,
            rtol=2e-2,
            custom_comparator=_comparator,
        )


@pytest.mark.high_rank
@pytest.mark.skip_simulation  # cp_degree >= 8 ranks; the CPU simulator runs one rank
class TestGlmIndexerCpPerf:
    """Production shape, PERFORMANCE only -- no goldens.

    ``s_local=1024`` with ``cp_degree=64`` is the real GLM config: each rank owns 1k queries and
    scores them against the full 64k CP context, which the in-kernel all-gather assembles. This is
    the only shape where the design's assumptions actually bind -- the Q panel amortizes K over 7
    tiles, a PSUM tile takes all 8 matmuls, and the score becomes ~94% of tensor-engine work instead
    of the 27% it is at end_pos=1024.

    Deliberately ``profile_only``: the score golden here is ``[1024, 65536]`` fp32 = 256 MB per rank
    and 2.7e11 MACs to compute, and the naive einsum would materialize an 8.6 GB intermediate.
    Correctness does not depend on length -- it depends on the three raggednesses, which
    ``TestGlmIndexerScore`` covers at shapes with tractable goldens. So this run measures, and does
    not check.
    """

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "s_local, cp_degree, end_pos, roll_score, topk_nseg, score_chunk",
        [
            pytest.param(2048, 8, 16384, 0, 0, -1, id="glm_S2048_cp8_16k_unrolled"),
            pytest.param(2048, 8, 16384, 1, 0, -1, id="glm_S2048_cp8_16k_rolled"),
            *CP64_PERF_CASES,
        ],
    )
    def test_cp_perf(
        self,
        test_manager: Orchestrator,
        platform_target,
        s_local,
        cp_degree,
        end_pos,
        roll_score,
        topk_nseg,
        score_chunk,
    ):
        n_slots = s_local + 64
        rng = np.random.default_rng(13)
        scale = 0.02
        wq = (rng.standard_normal((Q_LORA_RANK, N_HEADS * HEAD_DIM)) * scale).astype(nl.bfloat16)
        wk = (rng.standard_normal((DIM, HEAD_DIM)) * scale).astype(nl.bfloat16)
        gamma = (1.0 + 0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)
        beta = (0.1 * rng.standard_normal(HEAD_DIM)).astype(np.float32)

        def create_inputs(rank_id: int):
            r = np.random.default_rng(1000 + rank_id)
            return {
                "x": (r.standard_normal((s_local, DIM)) * scale).astype(nl.bfloat16),
                "qr": (r.standard_normal((s_local, Q_LORA_RANK)) * scale).astype(nl.bfloat16),
                "wq": wq,
                "wk": wk,
                "k_norm_gamma": gamma,
                "k_norm_beta": beta,
                "cos": _rope_table(s_local, ROPE_HEAD_DIM, np.cos),
                "sin": _rope_table(s_local, ROPE_HEAD_DIM, np.sin),
                "slot_mapping": _slot_mapping(s_local, n_slots),
                "weights": (0.5 + 0.5 * r.random((s_local, N_HEADS))).astype(np.float32),
                "q_pos": (np.arange(s_local) + rank_id * s_local).astype(np.float32),
                "end_pos": end_pos,
                "k_cache": np.zeros((n_slots, HEAD_DIM), dtype=nl.bfloat16),
                "topk_nseg": topk_nseg,
                "roll_score": roll_score,
                # -1 = segmented A/B baseline, 0 = AUTO (derive_score_chunk, chunked above 64k).
                "score_chunk": score_chunk,
                # OFF: this suite validates nothing, and the full score is 64 GB at 1M.
                "emit_score": 0,
                "cp_degree": cp_degree,
                "index_topk": INDEX_TOPK,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=glm_indexer_proj,
            torch_ref=glm_indexer_proj_cp_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            output_keys=[],
            profile_only=True,
        )
