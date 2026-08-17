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

from typing import final

import nki.language as nl
import numpy as np
import pytest
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
    generate_inputs,
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
    wq_b_scale_np = wq_b_scale.numpy() if hasattr(wq_b_scale, "numpy") else wq_b_scale
    wk_scale_np = wk_scale.numpy() if hasattr(wk_scale, "numpy") else wk_scale
    wq_b_scale_np = wq_b_scale_np.astype(np.uint8)
    wk_scale_np = wk_scale_np.astype(np.uint8)
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
    )
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
    # DeepSeek cp64-style shard: this rank scores S_local=128 queries against a
    # gathered context, real indexer dims (128 heads, index_topk=128).
    # start_pos=1920 -> end_pos=2048. This profiles the fused snake->flat indexer
    # at a single-tile context with the real 128-head / topk=128 dims.
    pytest.param(128, 128, 7168, 1536, 64, 128, 1, 1920, False, 128,
                 id="bf16score_ds_slocal128_skv2k"),
    # REAL DeepSeek-V3.2 indexer dims (config_671B_v3.2.json): index_n_heads=64,
    # index_head_dim=128, index_topk=2048. Same S_local=128 / end_pos=2048 CP-shard
    # profiling point as above but with the true 64-head / topk=2048 op mix.
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 1920, False, 2048,
                 id="bf16score_ds_real_h64_topk2k"),
    # 8k gathered-K repro (real cp64 score phase: S_local=128 vs full S_kv=8192).
    # end_pos=8192 stresses the PSUM/SBUF tiling that the full-model indexer hit
    # (NCC_IGCA088). start_pos=8064 -> end_pos=8192.
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 8064, False, 2048,
                 id="bf16score_ds_real_h64_skv8k"),
    # 16k repro: end_pos=16384 forces the score PSUM to 16 banks WITHOUT the
    # PSUM_MAX_W tiling cap (must-fail NCC_IGCA088), confirming the cap keeps it
    # within the 8 hardware banks. start_pos=16256 -> end_pos=16384.
    pytest.param(64, 128, 7168, 1536, 64, 128, 1, 16256, False, 2048,
                 id="bf16score_ds_real_h64_skv16k"),
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
        # Local aliases: kernel/generate_inputs use S / batch_size naming.
        S = seqlen
        batch_size = batch

        def input_generator(test_config):
            return generate_inputs(
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

        inputs = input_generator(None)
        # Drop the ref-only raw `qr`: the MX kernel consumes the pre-quantized
        # qr_qtz_hbm/qr_scale_hbm generate_inputs produces (`qr` is not a kernel
        # param), and the ref reconstructs the MX-rounded qr from those. Leaving
        # `qr` in would trip validate_input_keys (extra key vs kernel signature).
        inputs = {k: v for k, v in inputs.items() if k != "qr"}

        def topk_comparator(golden_dict, output_tensors):
            # output_0: index_score [M, end_pos] f32 — strict compare via the
            # framework's default (we still hand it the golden array).
            # output_1: topk_idx [num_S, NUM_TOPK_BATCHES, P_MAX, index_topk]
            #   uint32 — validate by gathering index_score at the selected
            #   positions and comparing sorted top-k score VALUES per query.
            #   This is tie-robust: equal scores at different positions (e.g.
            #   the -inf causal-mask padding) compare equal regardless of which
            #   index the hardware vs numpy picked.
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
                def validate(self, actual_raw_output):
                    actual = np.frombuffer(actual_raw_output, dtype=np.float32).reshape(M, end_pos)
                    return maxAllClose(
                        actual, golden_score, rtol=5e-2, atol=1e-5, equal_nan_inf=True, verbose=1, logfile=self.logfile
                    )

            class TopkIdxValidator(CustomValidator):
                @override
                def validate(self, actual_raw_output):
                    hw = np.frombuffer(actual_raw_output, dtype=np.uint32).reshape(
                        batch_size * num_s_tiles_per_batch, NUM_TOPK_BATCHES, P_MAX, k
                    )
                    # Gather the score each selected index points to, per query,
                    # then compare sorted values vs the golden's top-k scores.
                    # The kernel loads a CONTIGUOUS-chunk snake, so topk's snake
                    # index i maps to actual position: (i%16)*src_x + (i//16).
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
            torch_ref=torch_ref_wrapper(_torch_ref_bf16score),
            kernel_input_generator=lambda _cfg: inputs,
            output_tensor_descriptor=_topk_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=Platforms.TRN3_A0),
            atol=1e-5,
            rtol=5e-2,
            custom_comparator=topk_comparator,
        )
