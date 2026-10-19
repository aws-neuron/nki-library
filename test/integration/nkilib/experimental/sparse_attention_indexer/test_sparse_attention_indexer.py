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

"""Integration tests for the Sparse Attention Indexer entry kernels.

Both BF16 and MX entries return ``(index_score, topk_relayout)``. The 16-fold
``topk_relayout`` tensor is what the new ``num_active_channels=128,
batch_size=8`` topk API will consume; until that API is wired, these tests
validate the relayout buffer directly against a numpy golden built by a
reshape of the reference score row.
"""

import inspect
from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.mla.deepseek.mla_common_cte import MlaPrecision, RopeLayout
from nkilib_src.nkilib.experimental.sparse_attention_indexer import (
    sparse_attention_indexer_mx_bf16score,
)
from nkilib_src.nkilib.experimental.sparse_attention_indexer.sparse_attention_indexer_mx_bf16score_torch import (
    sparse_attention_indexer_mx_bf16score_torch_ref,
)
from nkilib_src.nkilib.experimental.sparse_attention_indexer.sparse_attention_indexer_utils import (
    NUM_TOPK_BATCHES,
    P_MAX,
    TOPK_QUERIES_PER_BATCH,
    TOPK_QUERY_CHUNKS,
    round_up_to_chunks,
)
from typing_extensions import override

from test.integration.nkilib.experimental.sparse_attention_indexer.test_sparse_attention_indexer_common import (
    _dequantize_weight_for_torch_ref,
    expand_compact_mx_scale_to_native,
    generate_inputs,
    generate_inputs_bf16,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.comparators import maxAllClose
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _topk_output_tensors(kernel_input):
    """Output descriptors for the bf16-score entry (hardware-topk output).

    output_0: index_score [M, end_pos] f32.
    output_1: topk_idx [num_S_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk]
        uint32 — snake-encoded top-k position indices.
    """
    M = kernel_input["x"].shape[0]
    S = M // kernel_input["batch_size"]
    end_pos = kernel_input["start_pos"] + S
    index_topk = kernel_input["index_topk"]
    num_s_tiles_per_batch = (S + P_MAX - 1) // P_MAX
    num_s_tiles_total = kernel_input["batch_size"] * num_s_tiles_per_batch
    return {
        "output_0": np.zeros((M, end_pos), dtype=np.float32),
        "output_1": np.zeros((num_s_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk), dtype=np.uint32),
    }


def _topk_idx_golden(score_2d, end_pos, index_topk):
    """Numpy golden for the hardware-topk output (snake-encoded indices).

    Mirrors ``topk_over_score``: per S-tile, ``NUM_TOPK_BATCHES`` calls of 8
    queries each; nisa.topk ranks each query's ``end_pos`` positions (bf16) and
    emits the top-``index_topk`` ascending. Output ``idx`` snake-encoded:
    ``out[t, 16g + j%16, j//16]`` = position of query ``8t+g``'s j-th largest.

    score_2d: [S, end_pos] f32 (zero-padded to P_MAX partitions upstream).
    Returns: [NUM_TOPK_BATCHES, P_MAX, index_topk] uint32.
    """
    S = score_2d.shape[0]
    k = index_topk
    # Match the kernel's bf16 ranking precision.
    score_bf16 = np.zeros((P_MAX, end_pos), dtype=np.float32)
    score_bf16[:S, :end_pos] = score_2d[:S, :end_pos].astype(nl.bfloat16).astype(np.float32)

    out = np.zeros((NUM_TOPK_BATCHES, P_MAX, k), dtype=np.uint32)
    for t in range(NUM_TOPK_BATCHES):
        for g in range(TOPK_QUERIES_PER_BATCH):
            q = t * TOPK_QUERIES_PER_BATCH + g
            flat = score_bf16[q, :end_pos]
            idx_desc = np.argsort(-flat, kind="stable")[:k]
            idx_asc = idx_desc[::-1]  # nisa.topk returns ascending
            for j in range(k):
                p = g * TOPK_QUERY_CHUNKS + j % TOPK_QUERY_CHUNKS
                c = j // TOPK_QUERY_CHUNKS
                out[t, p, c] = idx_asc[j]
    return out


def _torch_ref_bf16score(
    x,
    wq_b,
    wk,
    k_norm_gamma,
    k_norm_beta,
    weights_proj,
    cos,
    sin,
    k_cache,
    mask,
    n_heads,
    head_dim,
    rope_head_dim,
    index_topk,
    start_pos,
    use_hadamard=False,
    batch_size=1,
    wq_b_scale=None,
    wk_scale=None,
    k_scale_cache=None,
    x_non_mx=None,
    x_mx_data=None,
    x_mx_scale=None,
    qr_qtz_hbm=None,
    qr_scale_hbm=None,
    phase="all",
    k_seq_out_hbm=None,
    end_pos_arg=None,
    emit_flat_topk=False,
    emit_tiled_topk=False,
    compact_scales=False,
    precision=MlaPrecision.MX,
    rope_layout=RopeLayout.HALF_SPLIT,
    qr_bf16_hbm=None,
):
    """Reference for the bf16-score variant. Computes index_score with no
    Hadamard and bf16 score matmul (cast Q/K to bf16 then matmul); MX
    projection noise still applied.

    Signature matches the kernel entry ``sparse_attention_indexer_mx_bf16score``:
    the MX entry consumes the pre-quantized qr latent (qr_qtz_hbm/qr_scale_hbm)
    from an upstream QKV kernel, not raw qr. The ref dequantizes it back to
    ``[M, q_lora_rank]`` so the Q projection replays the exact MX-rounded qr the
    kernel matmuls (dropping the raw-qr golden it used before).
    """
    # Same MX weight dequant as the MX path (Q/K/W projections are still MX).
    assert wq_b_scale is not None and wk_scale is not None, "the MX reference requires MX weight scales"
    wq_b_scale_np = wq_b_scale.numpy() if hasattr(wq_b_scale, "numpy") else wq_b_scale
    wk_scale_np = wk_scale.numpy() if hasattr(wk_scale, "numpy") else wk_scale
    wq_b_scale_np = wq_b_scale_np.astype(np.uint8)
    wk_scale_np = wk_scale_np.astype(np.uint8)
    if compact_scales:
        wq_b_scale_np = expand_compact_mx_scale_to_native(wq_b_scale_np)
        wk_scale_np = expand_compact_mx_scale_to_native(wk_scale_np)
    q_lora_rank = wq_b.shape[0] * 4
    wq_b_arg = _dequantize_weight_for_torch_ref(wq_b, wq_b_scale_np, q_lora_rank, n_heads * head_dim)
    dim = wk.shape[0] * 4
    wk_dim_first = _dequantize_weight_for_torch_ref(wk, wk_scale_np, dim, head_dim)
    wk_arg = wk_dim_first.T

    # The src ref reconstructs the MX-rounded qr from qr_qtz_hbm/qr_scale_hbm
    # itself, so pass them straight through (matching the kernel signature).
    ref = sparse_attention_indexer_mx_bf16score_torch_ref(
        x,
        wq_b_arg,
        wk_arg,
        k_norm_gamma,
        k_norm_beta,
        weights_proj,
        cos,
        sin,
        k_cache,
        mask,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_topk=index_topk,
        start_pos=start_pos,
        use_hadamard=False,
        batch_size=batch_size,
        qr_qtz_hbm=qr_qtz_hbm,
        qr_scale_hbm=qr_scale_hbm,
        phase=phase,
        precision=precision,
        rope_layout=rope_layout,
        qr_bf16_hbm=qr_bf16_hbm,
    )
    if phase == "kproj":
        M_k = x.shape[0]
        S_k = M_k // batch_size
        num_s_tiles_total = batch_size * ((S_k + P_MAX - 1) // P_MAX)
        return {
            "k_seq_out_hbm": ref["output_0"],
            "output_1": np.zeros((num_s_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk), dtype=np.uint32),
        }
    index_score = ref["output_0"]
    if hasattr(index_score, "numpy"):
        index_score_np = index_score.numpy().astype(np.float32)
    else:
        index_score_np = np.asarray(index_score, dtype=np.float32)

    M = x.shape[0]
    S = M // batch_size
    end_pos = start_pos + S
    num_s_tiles_per_batch = (S + P_MAX - 1) // P_MAX

    topk_idx_out = np.zeros(
        (batch_size * num_s_tiles_per_batch, NUM_TOPK_BATCHES, P_MAX, index_topk),
        dtype=np.uint32,
    )
    for b in range(batch_size):
        for s_idx in range(num_s_tiles_per_batch):
            s_start = s_idx * P_MAX
            s_end = min(s_start + P_MAX, S)
            tile_size = s_end - s_start
            row_start = b * S + s_start
            score_tile = np.zeros((P_MAX, end_pos), dtype=np.float32)
            score_tile[:tile_size, :end_pos] = index_score_np[row_start : row_start + tile_size, :end_pos]
            global_idx = b * num_s_tiles_per_batch + s_idx
            topk_idx_out[global_idx] = _topk_idx_golden(score_tile, end_pos, index_topk)

    return {"output_0": index_score, "output_1": topk_idx_out}


def _torch_ref_bf16_proj(
    x,
    wq_b,
    wk,
    k_norm_gamma,
    k_norm_beta,
    weights_proj,
    cos,
    sin,
    k_cache,
    mask,
    n_heads,
    head_dim,
    rope_head_dim,
    index_topk,
    start_pos,
    use_hadamard=False,
    batch_size=1,
    wq_b_scale=None,
    wk_scale=None,
    k_scale_cache=None,
    x_non_mx=None,
    x_mx_data=None,
    x_mx_scale=None,
    qr_qtz_hbm=None,
    qr_scale_hbm=None,
    phase="all",
    k_seq_out_hbm=None,
    end_pos_arg=None,
    emit_flat_topk=False,
    emit_tiled_topk=False,
    compact_scales=False,
    precision=MlaPrecision.BF16,
    rope_layout=RopeLayout.HALF_SPLIT,
    qr_bf16_hbm=None,
):
    """Reference wrapper for the BF16-projection path.

    Signature matches the kernel entry. Unlike :func:`_torch_ref_bf16score` there is no MX
    weight dequant to undo — the weights arrive as bf16 already. The one fixup is ``wk``: the
    kernel takes it ``[dim, head_dim]`` (contraction on rows, as ``nc_matmul`` needs) while the
    shared ref computes ``x @ wk.T`` and so wants ``[head_dim, dim]``.
    """
    wk_arg = wk.T if wk is not None else None
    ref = sparse_attention_indexer_mx_bf16score_torch_ref(
        x,
        wq_b,
        wk_arg,
        k_norm_gamma,
        k_norm_beta,
        weights_proj,
        cos,
        sin,
        k_cache,
        mask,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        index_topk=index_topk,
        start_pos=start_pos,
        use_hadamard=False,
        batch_size=batch_size,
        phase=phase,
        precision=precision,
        rope_layout=rope_layout,
        qr_bf16_hbm=qr_bf16_hbm,
    )
    index_score = ref["output_0"]
    index_score_np = (
        index_score.numpy().astype(np.float32) if hasattr(index_score, "numpy") else np.asarray(index_score, np.float32)
    )

    M = x.shape[0]
    S = M // batch_size
    end_pos = start_pos + S
    num_s_tiles_per_batch = (S + P_MAX - 1) // P_MAX
    topk_idx_out = np.zeros((batch_size * num_s_tiles_per_batch, NUM_TOPK_BATCHES, P_MAX, index_topk), dtype=np.uint32)
    for b in range(batch_size):
        for s_idx in range(num_s_tiles_per_batch):
            s_start = s_idx * P_MAX
            tile_size = min(s_start + P_MAX, S) - s_start
            row_start = b * S + s_start
            score_tile = np.zeros((P_MAX, end_pos), dtype=np.float32)
            score_tile[:tile_size, :end_pos] = index_score_np[row_start : row_start + tile_size, :end_pos]
            topk_idx_out[b * num_s_tiles_per_batch + s_idx] = _topk_idx_golden(score_tile, end_pos, index_topk)
    return {"output_0": index_score, "output_1": topk_idx_out}


# fmt: off
FAST_PARAM_NAMES = "n_heads, head_dim, dim, q_lora_rank, rope_head_dim, seqlen, batch, start_pos, use_hadamard, index_topk"

# MX projections + BF16 score matmul + hardware topk.
# dim must be a multiple of 512 with num_dim_tiles >= 8 (W-proj fast path),
# so the configs below all use the V3 dim=7168.
BF16SCORE_TEST_PARAMS = [
    pytest.param(128, 128, 7168, 1536, 64, 1024, 1, 0, False, 64,
                 id="bf16score_v3_long"),
    pytest.param(64, 128, 7168, 1536, 64, 1024, 1, 0, False, 64,
                 id="bf16score_v3_h64"),
    pytest.param(128, 128, 7168, 1536, 64, 128, 1, 1920, False, 128,
                 id="bf16score_ds_slocal128_skv2k"),
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 1920, False, 2048,
                 id="bf16score_ds_real_h64_topk2k"),
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 8064, False, 2048,
                 id="bf16score_ds_real_h64_skv8k"),
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 16256, False, 2048,
                 id="bf16score_ds_real_h64_skv16k"),
    pytest.param(64, 128, 7168, 1536, 64, 64, 1, 4032, False, 2048,
                 id="bf16score_ws64_s64_wproj_repro"),
]

# BF16 Q/K projections (precision=BF16).
BF16_PROJ_PARAM_NAMES = FAST_PARAM_NAMES + ", rope_layout"
BF16_PROJ_TEST_PARAMS = [
    # ---- DeepSeek-V3.2 indexer dims in bf16 (HALF_SPLIT). Different target model, so these
    # carry DS values (64 heads, dim 7168, q_lora 1536), not GLM's. ----
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 0, False, 128, RopeLayout.HALF_SPLIT,
                 id="bf16proj_ds_h64_S128"),
    pytest.param(128, 128, 7168, 1536, 64, 256, 1, 0, False, 128, RopeLayout.HALF_SPLIT,
                 id="bf16proj_ds_h128_S256"),
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 1920, False, 2048, RopeLayout.HALF_SPLIT,
                 id="bf16proj_ds_h64_skv2k_topk2k"),

    pytest.param(32, 128, 6144, 2048, 64, 64, 1, 4032, False, 2048, RopeLayout.INTERLEAVED,
                 id="glm_S64_Sglob4k"),
    pytest.param(32, 128, 6144, 2048, 64, 128, 1, 8064, False, 2048, RopeLayout.INTERLEAVED,
                 id="glm_S128_Sglob8k"),
    pytest.param(32, 128, 6144, 2048, 64, 256, 1, 16128, False, 2048, RopeLayout.INTERLEAVED,
                 id="glm_S256_Sglob16k"),
    pytest.param(32, 128, 6144, 2048, 64, 512, 1, 32256, False, 2048, RopeLayout.INTERLEAVED,
                 id="glm_S512_Sglob32k"),
    pytest.param(32, 128, 6144, 2048, 64, 128, 1, 8064, False, 2048, RopeLayout.HALF_SPLIT,
                 id="glm_S128_halfsplit_guard"),
]
# fmt: on


@pytest_test_metadata(name="SparseAttentionIndexer")
@pytest_marks(["sparse_attention_indexer"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
class TestSparseAttentionIndexer:
    """MX-projection + BF16-score entry kernel with hardware topk."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "sai_entry",
        [
            # Single consolidated entry: the score+topk phase is always seq-sharded
            # (the performant flow). The former separate _seqshard entry was folded in.
            pytest.param(sparse_attention_indexer_mx_bf16score, id="seqshard"),
        ],
    )
    @pytest.mark.parametrize(FAST_PARAM_NAMES, BF16SCORE_TEST_PARAMS)
    def test_indexer_bf16score(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
        sai_entry,
    ):
        """BF16-score variant: MX projections, bf16 score matmul, no Hadamard.

        Parametrized over the base kernel and the seq-sharded score/topk variant
        (``sai_entry``); both must produce identical topk (the seq-shard only
        changes which LNC core computes each query, not the math)."""
        self._run_bf16score_case(
            test_manager,
            platform_target,
            n_heads,
            head_dim,
            dim,
            q_lora_rank,
            rope_head_dim,
            seqlen,
            batch,
            start_pos,
            use_hadamard,
            index_topk,
            sai_entry,
        )

    # Compact block-128 weight scales (compact_scales=True): the DeepSeek checkpoint layout,
    # broadcast to native MX granularity in-kernel.
    @pytest.mark.parametrize(
        "sai_entry",
        [pytest.param(sparse_attention_indexer_mx_bf16score, id="seqshard")],
    )
    @pytest.mark.parametrize(
        FAST_PARAM_NAMES,
        [
            pytest.param(128, 128, 7168, 1536, 64, 1024, 1, 0, False, 64, id="compact_v3_long"),
            pytest.param(64, 128, 7168, 1536, 64, 128, 1, 1920, False, 2048, id="compact_ds_real_h64_topk2k"),
            pytest.param(64, 128, 7168, 1536, 64, 128, 1, 8064, False, 2048, id="compact_ds_real_h64_skv8k"),
            pytest.param(128, 128, 7168, 1536, 64, 128, 1, 1920, False, 128, id="compact_h128_slocal128_skv2k"),
            pytest.param(64, 128, 7168, 1536, 64, 64, 1, 4032, False, 2048, id="compact_ws64_s64"),
            pytest.param(64, 128, 7168, 1536, 64, 1024, 1, 0, False, 64, id="compact_v3_h64"),
        ],
    )
    def test_indexer_bf16score_compact_scales(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
        sai_entry,
    ):
        """Weight scales supplied in the compact block-128 layout instead of native block-32."""
        self._run_bf16score_case(
            test_manager,
            platform_target,
            n_heads,
            head_dim,
            dim,
            q_lora_rank,
            rope_head_dim,
            seqlen,
            batch,
            start_pos,
            use_hadamard,
            index_topk,
            sai_entry,
            compact_scales=True,
        )

    @pytest.mark.parametrize(BF16_PROJ_PARAM_NAMES, BF16_PROJ_TEST_PARAMS)
    def test_indexer_bf16_proj(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
        rope_layout,
    ):
        """BF16 Q/K projections (precision=BF16) with a selectable RoPE pairing.

        Covers the GLM-MoE-DSA indexer config (index_n_heads=32, dim=6144, q_lora_rank=2048,
        INTERLEAVED RoPE) alongside the DeepSeek-V3.2 dims in bf16.
        """
        self._run_bf16score_case(
            test_manager,
            platform_target,
            n_heads,
            head_dim,
            dim,
            q_lora_rank,
            rope_head_dim,
            seqlen,
            batch,
            start_pos,
            use_hadamard,
            index_topk,
            sparse_attention_indexer_mx_bf16score,
            bf16_proj=True,
            rope_layout=rope_layout,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(BF16_PROJ_PARAM_NAMES, [BF16_PROJ_TEST_PARAMS[0], BF16_PROJ_TEST_PARAMS[3]])
    def test_indexer_bf16_proj_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
        rope_layout,
    ):
        """Fast lane: one DeepSeek-dims half-split case + one GLM interleaved case."""
        self._run_bf16score_case(
            test_manager,
            platform_target,
            n_heads,
            head_dim,
            dim,
            q_lora_rank,
            rope_head_dim,
            seqlen,
            batch,
            start_pos,
            use_hadamard,
            index_topk,
            sparse_attention_indexer_mx_bf16score,
            bf16_proj=True,
            rope_layout=rope_layout,
        )

    def _run_bf16score_case(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
        sai_entry,
        compact_scales=False,
        bf16_proj=False,
        rope_layout=RopeLayout.HALF_SPLIT,
    ):
        """Shared body for the bf16-score cases (MX or BF16 projections, native or compact scales)."""
        # Local aliases: kernel/generate_inputs use S / batch_size naming.
        S = seqlen
        batch_size = batch

        if bf16_proj:
            inputs = generate_inputs_bf16(
                n_heads,
                head_dim,
                dim,
                q_lora_rank,
                rope_head_dim,
                S,
                batch_size,
                start_pos,
                index_topk,
                rope_layout,
            )
            torch_ref = _torch_ref_bf16_proj
        else:
            inputs = generate_inputs(
                n_heads,
                head_dim,
                dim,
                q_lora_rank,
                rope_head_dim,
                S,
                batch_size,
                start_pos,
                use_hadamard,
                index_topk,
                bf16_score_kv_layout=True,
                compact_scales=compact_scales,
            )
            inputs = {k: v for k, v in inputs.items() if k != "qr"}
            torch_ref = _torch_ref_bf16score

        def topk_comparator(golden_dict, output_tensors):
            golden_score = golden_dict["output_0"]
            if hasattr(golden_score, "numpy"):
                golden_score = golden_score.numpy()
            golden_score = np.asarray(golden_score, dtype=np.float32)
            M = golden_score.shape[0]
            end_pos = golden_score.shape[1]
            S = M // batch_size
            num_s_tiles_per_batch = (S + P_MAX - 1) // P_MAX
            k = index_topk

            class IndexScoreValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    actual = np.frombuffer(inference_output, dtype=np.float32).reshape(M, end_pos)
                    return maxAllClose(
                        actual, golden_score, rtol=5e-2, atol=1e-5, equal_nan_inf=True, verbose=1, logfile=self.logfile
                    )

            class TopkIdxValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    hw = np.frombuffer(inference_output, dtype=np.uint32).reshape(
                        batch_size * num_s_tiles_per_batch, NUM_TOPK_BATCHES, P_MAX, k
                    )

                    T_pad = round_up_to_chunks(end_pos)
                    src_x = T_pad // TOPK_QUERY_CHUNKS
                    hw_vals = np.full((M, k), -np.inf, dtype=np.float32)
                    ref_vals = np.full((M, k), -np.inf, dtype=np.float32)
                    for b in range(batch_size):
                        for s_idx in range(num_s_tiles_per_batch):
                            g_idx = b * num_s_tiles_per_batch + s_idx
                            s_start = s_idx * P_MAX
                            s_end = min(s_start + P_MAX, S)
                            for t in range(NUM_TOPK_BATCHES):
                                for grp in range(TOPK_QUERIES_PER_BATCH):
                                    ql = t * TOPK_QUERIES_PER_BATCH + grp
                                    if s_start + ql >= s_end:
                                        continue
                                    row = b * S + s_start + ql
                                    for j in range(k):
                                        p = grp * TOPK_QUERY_CHUNKS + j % TOPK_QUERY_CHUNKS
                                        c = j // TOPK_QUERY_CHUNKS
                                        snake_i = int(hw[g_idx, t, p, c])
                                        pos = (snake_i % TOPK_QUERY_CHUNKS) * src_x + (snake_i // TOPK_QUERY_CHUNKS)
                                        if pos < end_pos:
                                            hw_vals[row, j] = golden_score[row, pos]
                    # golden ref values: top-k of bf16-cast score per query
                    sb = golden_score.astype(nl.bfloat16).astype(np.float32)
                    for row in range(M):
                        topv = np.sort(sb[row])[::-1][:k]
                        ref_vals[row, : len(topv)] = topv
                    self._print_with_log("Results for topk_idx (gathered values):")
                    return maxAllClose(
                        np.sort(hw_vals, axis=-1),
                        np.sort(ref_vals, axis=-1),
                        rtol=5e-2,
                        atol=1e-5,
                        verbose=1,
                        logfile=self.logfile,
                    )

            return {
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=IndexScoreValidator,
                    output_ndarray=output_tensors["output_0"],
                ),
                "output_1": CustomValidatorWithOutputTensorData(
                    validator=TopkIdxValidator,
                    output_ndarray=output_tensors["output_1"],
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=sai_entry,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=lambda _cfg: inputs,
            output_tensor_descriptor=_topk_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-5,
            rtol=5e-2,
            custom_comparator=topk_comparator,
        )

    # CP phased split: this rank's K-projection half. phase="kproj" runs only
    # wk + LayerNorm + RoPE and writes the SEQ-MAJOR [M, head_dim] K the CP parent
    # all-gathers before feeding phase="score". Validates the K hand-off buffer
    # (the score-over-gathered-K half shares code with the phase="all" path the
    # start_pos cases above already cover). Real DeepSeek-V3.2 indexer dims.
    @pytest.mark.parametrize(
        FAST_PARAM_NAMES,
        [
            pytest.param(64, 128, 7168, 1536, 64, 128, 1, 0, False, 2048, id="kproj_ds_real_h64_S128"),
            pytest.param(64, 128, 7168, 1536, 64, 256, 1, 0, False, 2048, id="kproj_ds_real_h64_S256"),
        ],
    )
    def test_indexer_bf16score_kproj(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
    ):
        """CP phase="kproj": validate the seq-major projected-K output (wk + LayerNorm + RoPE)."""
        S = seqlen
        batch_size = batch
        M = batch_size * S

        inputs = generate_inputs(
            n_heads,
            head_dim,
            dim,
            q_lora_rank,
            rope_head_dim,
            S,
            batch_size,
            start_pos,
            use_hadamard,
            index_topk,
            bf16_score_kv_layout=True,
        )
        inputs = {k: v for k, v in inputs.items() if k != "qr"}
        inputs["phase"] = "kproj"
        inputs["k_seq_out_hbm.must_alias_input"] = np.zeros((M, head_dim), dtype=nl.bfloat16)

        num_s_tiles_total = batch_size * ((S + P_MAX - 1) // P_MAX)

        def _kproj_output_tensors(kernel_input):
            # phase="kproj" returns (k_seq_out_hbm, topk_out).
            return {
                "k_seq_out_hbm": np.zeros((M, head_dim), dtype=nl.bfloat16),
                "output_1": np.zeros((num_s_tiles_total, NUM_TOPK_BATCHES, P_MAX, index_topk), dtype=np.uint32),
            }

        def _kproj_comparator(golden_dict, output_tensors):
            golden_k = np.asarray(
                golden_dict["k_seq_out_hbm"].numpy()
                if hasattr(golden_dict["k_seq_out_hbm"], "numpy")
                else golden_dict["k_seq_out_hbm"],
                dtype=np.float32,
            ).reshape(M, head_dim)

            class KSeqValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    actual = np.frombuffer(inference_output, dtype=nl.bfloat16).astype(np.float32).reshape(M, head_dim)
                    return maxAllClose(actual, golden_k, rtol=5e-2, atol=1e-2, verbose=1, logfile=self.logfile)

            class AlwaysPass(CustomValidator):
                # output_1 is the kernel's uncomputed topk buffer in the kproj phase -- not validated.
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    return True

            return {
                "k_seq_out_hbm": CustomValidatorWithOutputTensorData(
                    validator=KSeqValidator, output_ndarray=output_tensors["k_seq_out_hbm"]
                ),
                "output_1": CustomValidatorWithOutputTensorData(
                    validator=AlwaysPass, output_ndarray=output_tensors["output_1"]
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=sparse_attention_indexer_mx_bf16score,
            torch_ref=torch_ref_wrapper(_torch_ref_bf16score),
            kernel_input_generator=lambda _cfg: inputs,
            output_tensor_descriptor=_kproj_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-2,
            rtol=5e-2,
            custom_comparator=_kproj_comparator,
        )


@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestIndexerChunkedTopk:
    """Numerical validation of the TWO-STAGE (chunked) top-k path.

    Reached whenever ``T >= 65536``, where ``nisa.topk`` cannot rank the context in one call.
    S is kept at 128 (a single s-tile) purely so the numpy golden stays cheap: the reference
    still argsorts the full 65536-wide score row per query, which is the expensive part.

    Validation compares GATHERED VALUES, not indices: a two-stage top-k may legitimately pick a
    different member of a tied group than a single-pass reference, so the invariant that holds
    is "the selected positions carry the top-k values", which is what this checks.
    """

    @pytest.mark.parametrize(
        BF16_PROJ_PARAM_NAMES,
        [
            # Target config, the two S_local values whose S_global crosses nisa.topk's ceiling:
            # S_local 1024 -> S_global 65536, S_local 2048 -> S_global 131072. index_topk 2048.
            pytest.param(
                32,
                128,
                6144,
                2048,
                64,
                1024,
                1,
                64512,
                False,
                2048,
                RopeLayout.INTERLEAVED,
                id="glm_S1024_Sglob64k",
            ),
            pytest.param(
                32,
                128,
                6144,
                2048,
                64,
                2048,
                1,
                129024,
                False,
                2048,
                RopeLayout.INTERLEAVED,
                id="glm_S2048_Sglob128k",
            ),
        ],
    )
    def test_chunked_topk_values(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_heads,
        head_dim,
        dim,
        q_lora_rank,
        rope_head_dim,
        seqlen,
        batch,
        start_pos,
        use_hadamard,
        index_topk,
        rope_layout,
    ):
        S, batch_size = seqlen, batch
        M = batch_size * S
        end_pos = start_pos + S
        inputs = generate_inputs_bf16(
            n_heads,
            head_dim,
            dim,
            q_lora_rank,
            rope_head_dim,
            S,
            batch_size,
            start_pos,
            index_topk,
            rope_layout,
        )
        inputs["emit_flat_topk"] = True  # chunked path emits decoded positions

        def _ref(*args, **kwargs):
            r = _torch_ref_bf16_proj(*args, **kwargs)
            return {"output_0": r["output_0"], "output_1": np.zeros((M, index_topk), dtype=np.int32)}

        # The framework validates the ref's signature against the kernel's, so _ref has to carry
        # the real ref's signature rather than (*args, **kwargs).
        _ref.__signature__ = inspect.signature(  # ty: ignore[unresolved-attribute]
            sparse_attention_indexer_mx_bf16score_torch_ref
        )

        def _out(kernel_input):
            return {
                "output_0": np.zeros((M, end_pos), dtype=np.float32),
                "output_1": np.zeros((M, index_topk), dtype=np.int32),
            }

        def _comparator(golden_dict, output_tensors):
            gs = golden_dict["output_0"]
            gs = gs.numpy() if hasattr(gs, "numpy") else gs
            gs = np.asarray(gs, dtype=np.float32)
            # bf16 is the ranking precision the kernel uses.
            gs_bf16 = gs.astype(nl.bfloat16).astype(np.float32)

            class ScorePass(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    actual = np.frombuffer(inference_output, dtype=np.float32).reshape(M, end_pos)
                    return maxAllClose(
                        actual, gs, rtol=5e-2, atol=1e-5, equal_nan_inf=True, verbose=1, logfile=self.logfile
                    )

            class FlatTopkValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    pos = np.frombuffer(inference_output, dtype=np.int32).reshape(M, index_topk)
                    if pos.min() < 0 or pos.max() >= end_pos:
                        self._print_with_log(
                            f"positions out of range: min={pos.min()} max={pos.max()} end_pos={end_pos}"
                        )
                        return False
                    got = np.take_along_axis(gs_bf16, pos.astype(np.int64), axis=1)
                    want = np.sort(gs_bf16, axis=1)[:, ::-1][:, :index_topk]
                    self._print_with_log("Results for chunked topk (gathered values):")
                    return maxAllClose(
                        np.sort(got, axis=1),
                        np.sort(want, axis=1),
                        rtol=5e-2,
                        atol=1e-5,
                        verbose=1,
                        logfile=self.logfile,
                    )

            return {
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=ScorePass, output_ndarray=output_tensors["output_0"]
                ),
                "output_1": CustomValidatorWithOutputTensorData(
                    validator=FlatTopkValidator, output_ndarray=output_tensors["output_1"]
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=sparse_attention_indexer_mx_bf16score,
            torch_ref=torch_ref_wrapper(_ref),
            kernel_input_generator=lambda _cfg: inputs,
            output_tensor_descriptor=_out,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-5,
            rtol=5e-2,
            custom_comparator=_comparator,
        )
