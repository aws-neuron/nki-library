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

"""Integration tests for the standalone sparse MLA latent + RoPE attention kernel
(``mla_sparse_attention_cte_kernel``), kernel A of the split DeepSeek-V3.2 sparse-MLA forward.
"""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mla.deepseek.mla_sparse_attention_cte import (
    mla_sparse_attention_cte_kernel,
)
from nkilib_src.nkilib.experimental.mla.deepseek.mla_sparse_attention_cte_torch import (
    mla_sparse_attention_cte_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _build_topk_causal_future(batch, seqlen, K, S_kv, q_pos_offset=0):
    """Topk exactly as the real DSA indexer emits it -- the sole sparse-flow topk builder.

    For a query at GLOBAL position q_global the topk holds all valid causal keys [0, q_global]
    then, when there are fewer than K of them, pads to K with FUTURE positions (idx > q_global)
    drawn from the indexer's -inf tie region. The sparse kernel causal-re-masks those fillers so
    the query attends only its true causal prefix; if it did not, it would attend future keys and
    diverge from the reference. A query with >= K valid keys gets no fillers (mask is a no-op), so
    a single sweep of shapes/offsets covers both the filler and the fully-valid paths."""
    topk = np.zeros((batch, seqlen, K), dtype=np.int32)
    for b in range(batch):
        for s in range(seqlen):
            g = min(q_pos_offset + s, S_kv - 1)
            valid = list(range(g + 1))  # causal keys [0, g]
            if len(valid) >= K:
                sel = valid[-K:]  # enough valid keys: no fillers, mask is a no-op
            else:
                future = list(range(g + 1, S_kv))[: K - len(valid)]  # future/filler positions
                while len(future) < K - len(valid):  # tiny S_kv: repeat a future position
                    future.append(S_kv - 1)
                sel = valid + future
            topk[b, s, :] = np.asarray(sel[:K], dtype=np.int32)
    return topk


def _generate_inputs(batch, seqlen, S_kv, hidden, K, L, R, dense=False, q_pos_offset=0):
    np.random.seed(42)
    inputs = {
        "q_lift_hbm": np.random.randn(batch, seqlen, hidden, L).astype(ml_dtypes.bfloat16),
        "q_pe_hbm": np.random.randn(batch, seqlen, hidden, R).astype(ml_dtypes.bfloat16),
        "c_kv_hbm": np.random.randn(batch, S_kv, L).astype(ml_dtypes.bfloat16),
        "k_pe_hbm": np.random.randn(batch, S_kv, R).astype(ml_dtypes.bfloat16),
        "softmax_scale": float(1.0 / np.sqrt(L + R)),
    }
    if dense:
        # Dense (no-indexer): attend all S_kv keys, causal-masked in-kernel. topk_indices is
        # not needed at all -- omit it entirely (the kernel arg is optional / defaults None).
        inputs["dense"] = True
        inputs["q_pos_offset"] = q_pos_offset
    else:
        # Sparse: the causal re-mask is the default flow now, so the topk always carries the
        # indexer's future fillers and the kernel always gets the GLOBAL query offset.
        inputs["topk_indices_hbm"] = _build_topk_causal_future(batch, seqlen, K, S_kv, q_pos_offset)
        inputs["q_pos_offset"] = q_pos_offset
    return inputs


def _output_tensors(kernel_input):
    batch, seqlen, hidden, L = kernel_input["q_lift_hbm"].shape
    return {"out_attn": np.zeros((batch, seqlen, hidden * L), dtype=ml_dtypes.bfloat16)}


# fmt: off
PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R"
SPARSE_PARAM_NAMES = PARAM_NAMES + ", q_pos_offset"
# Sparse params: the causal re-mask is the DEFAULT sparse flow now, so every case runs through
# the future-filler topk (see _build_topk_causal_future) at a GLOBAL query offset q_pos_offset
# (= cp_rank * seqlen). Cases where q_global < K exercise the re-mask (fillers dropped); cases
# where a shard's later queries reach q_global >= K also cover the fully-valid no-op path. This
# one list carries the full shape sweep (K / heads / seqlen / S_kv) plus CP-rank offsets.
SPARSE_PARAMS = [
    # ---- baseline small (early queries get fillers, later ones fully valid) ----
    pytest.param(1, 256, 512,  16,  256,  512, 64, 0, id="baseline_S256_K256"),
    # ---- real DeepSeek-V3.2 dims: seqlen seq-shard 128, S_kv=8192, 128 heads, topk=2048 ----
    # (id carries "8192" so the slow full-shape case auto-skips in simulation, runs on hardware)
    pytest.param(1, 128, 8192, 128, 2048, 512, 64, 0, id="deepseek_S128_Skv8192"),

    # ---- vary K (topk): smallest legal (K=128) up to large; K%128==0 ----
    pytest.param(1, 128, 2048, 16,  128,  512, 64, 0, id="K128_smallest"),
    pytest.param(1, 128, 2048, 16,  512,  512, 64, 0, id="K512"),
    pytest.param(1, 128, 4096, 16,  1024, 512, 64, 0, id="K1024"),

    # vary hidden (heads): 32, 64 (full 128 is the deepseek case above). hidden>=16 required (q_lift
    # dma_transpose output step hidden*bf16 must be 32B-aligned; hidden<16 fails NCC_IBIR155 at compile).
    pytest.param(1, 128, 1024, 32,  256,  512, 64, 0, id="H32"),
    pytest.param(1, 256, 2048, 64,  512,  512, 64, 0, id="H64"),

    # ---- vary seqlen (per-core query count): mixes filler queries (g < K) with fully-valid ones ----
    pytest.param(1, 512,  1024, 16, 256,  512, 64, 0, id="S512_mixed"),
    pytest.param(1, 1024, 2048, 16, 256,  512, 64, 0, id="S1024_mixed"),

    # ---- larger S_kv (cache) with modest topk (sparse selection over long cache) ----
    pytest.param(1, 128, 4096, 32,  512,  512, 64, 0, id="Skv4096"),

    # ---- SMALL sequence lengths at the full DeepSeek config (S_kv=8192, 128 heads, topk=2048) ----
    pytest.param(1, 2,  8192, 128, 2048, 512, 64, 0, id="S2_deepseek"),
    pytest.param(1, 16, 8192, 128, 2048, 512, 64, 0, id="S16_deepseek"),
    pytest.param(1, 32, 8192, 128, 2048, 512, 64, 0, id="S32_deepseek"),

    # ---- CP rank > 0: GLOBAL query offset q_pos_offset = cp_rank * seqlen (mask offset != 0) ----
    pytest.param(1, 16,  2048, 32,  512,  512, 64, 100, id="cprank_allfiller_S16"),
    pytest.param(1, 128, 8192, 128, 2048, 512, 64, 256, id="cprank_deepseek_S128_Skv8192"),
    # NOTE: L is always 512 for DeepSeek-V3.2 (kv_lora_rank=512); L!=512 is out of scope
    # (the MM2 4-pack output permute + torch ref assume the single-latent-tile layout).
]

# Dense (no-indexer) params: the S <= 2048 case where the indexer is skipped and every query
# attends ALL S_kv keys with an in-kernel causal mask. Under CP=64 the per-rank query count is
# S/64 (so at most ~32 real queries), while S_kv is the full all-gathered sequence (a multiple
# of 128). We therefore vary seqlen (queries/rank) with S_kv fixed at 128; K is a dummy (unused).
# The last row is the production profiling point: heads=128 (real DeepSeek), seqlen=128. LNC=2.
DENSE_PARAMS = [
    pytest.param(1, 2, 128, 128, 16, 512, 64, id="dense_S2"),
    pytest.param(1, 16, 128, 128, 16, 512, 64, id="dense_S16"),
    pytest.param(1, 32, 128, 128, 16, 512, 64, id="dense_S32"),
    pytest.param(1, 128, 128, 128, 16, 512, 64, id="dense_S128_h128_prod"),
]

# Dense + GLOBAL query position (q_pos_offset).
DENSE_QGLOBAL_PARAMS = [
    pytest.param(1, 32, 128, 128, 16, 512, 64, 32, id="qglobal_rank1_S32"),
    pytest.param(1, 32, 128, 128, 16, 512, 64, 96, id="qglobal_rank3_S32_full_prefix"),
    pytest.param(1, 2, 128, 128, 16, 512, 64, 126, id="qglobal_tail_S2"),
    pytest.param(1, 16, 384, 128, 16, 512, 64, 200, id="qglobal_midchunk_S16"),
    pytest.param(1, 16, 384, 128, 16, 512, 64, 256, id="qglobal_chunkedge_S16"),
    pytest.param(1, 128, 1024, 128, 16, 512, 64, 128,  id="qglobal_deepseek_S128_Skv1024"),
    pytest.param(1, 128, 2048, 128, 16, 512, 64, 1920, id="qglobal_deepseek_S128_Skv2048"),
]
# fmt: on


@pytest_test_metadata(name="SparseMlaLatentAttnCteStandalone")
@pytest_marks(["attention", "mla_sparse_attention_cte_kernel"])
class TestSparseMlaLatentAttnCte:
    """Standalone sparse MLA latent + RoPE attention (kernel A)."""

    def _run(self, test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, dense=False, q_pos_offset=0):
        def input_generator(test_config):
            return _generate_inputs(batch, seqlen, S_kv, hidden, K, L, R, dense=dense, q_pos_offset=q_pos_offset)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_sparse_attention_cte_kernel,
            torch_ref=torch_ref_wrapper(mla_sparse_attention_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            atol=5e-2,
            rtol=5e-2,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(SPARSE_PARAM_NAMES, SPARSE_PARAMS[:1])
    def test_sparse_mla_latent_attn_cte_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset
    ):
        self._run(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset=q_pos_offset)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(SPARSE_PARAM_NAMES, SPARSE_PARAMS)
    def test_sparse_mla_latent_attn_cte(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset
    ):
        """Sparse latent + RoPE attention. The topk always carries the indexer's future fillers
        (idx > q_global), so this verifies the DEFAULT sparse flow causal-re-masks them and each
        query attends only its true causal prefix."""
        self._run(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset=q_pos_offset)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PARAMS)
    def test_dense_mla_latent_attn_cte(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R
    ):
        self._run(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, dense=True)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES + ", q_pos_offset", DENSE_QGLOBAL_PARAMS)
    def test_dense_mla_latent_attn_cte_q_global(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        q_pos_offset,
    ):
        """Dense causal mask at a nonzero GLOBAL query position (CP rank > 0)."""
        self._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=True,
            q_pos_offset=q_pos_offset,
        )
