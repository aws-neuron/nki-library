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
from nkilib_src.nkilib.experimental.mla.deepseek.mla_common_cte import MlaPrecision
from nkilib_src.nkilib.experimental.mla.deepseek.mla_sparse_attention_cte import (
    mla_sparse_attention_cte_kernel,
)
from nkilib_src.nkilib.experimental.mla.deepseek.mla_sparse_attention_cte_torch import (
    mla_sparse_attention_cte_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, ModelTestType, Platforms, prepare_model_parametrize
from test.utils.model_test_configs import no_model_configs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

try:
    from test.integration.nkilib.experimental.mla.test_mla_sparse_attention_cte_model_config import (
        MODEL_PARAM_NAMES,
        mla_sparse_attention_cte_model_configs,
    )
except ImportError:
    MODEL_PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset, dense, precision"
    mla_sparse_attention_cte_model_configs = no_model_configs()


def _build_topk_causal_future(batch, seqlen, K, S_kv, q_pos_offset=0):
    """Topk exactly as the real DSA indexer emits it -- the sole sparse-flow topk builder.

    For a query at GLOBAL position q_global the topk holds all valid causal keys [0, q_global]
    then, when there are fewer than K of them, pads to K with FUTURE positions (idx > q_global)
    drawn from the indexer's -inf tie region. The sparse kernel causal-re-masks those fillers so
    the query attends only its true causal prefix; if it did not, it would attend future keys and
    diverge from the reference. A query with >= K valid keys gets no fillers (mask is a no-op), so
    a single sweep of shapes/offsets covers both the filler and the fully-valid paths.

    Built with numpy rather than Python lists: the list version materialized range(g+1) and
    range(g+1, S_kv) per query, which is O(seqlen * S_kv) and takes minutes once S_kv reaches
    128k. Output is identical."""
    topk = np.zeros((batch, seqlen, K), dtype=np.int32)
    for b in range(batch):
        for s in range(seqlen):
            g = min(q_pos_offset + s, S_kv - 1)
            n_valid = g + 1
            if n_valid >= K:
                sel = np.arange(n_valid - K, n_valid, dtype=np.int32)  # enough valid keys: no fillers
            else:
                n_fut = K - n_valid
                fut = np.arange(g + 1, min(g + 1 + n_fut, S_kv), dtype=np.int32)
                if fut.size < n_fut:  # tiny S_kv: repeat the last future position
                    fut = np.concatenate([fut, np.full(n_fut - fut.size, S_kv - 1, dtype=np.int32)])
                sel = np.concatenate([np.arange(n_valid, dtype=np.int32), fut])
            topk[b, s, :] = sel
    return topk


def _generate_inputs(
    batch,
    seqlen,
    S_kv,
    hidden,
    K,
    L,
    R,
    dense=False,
    q_pos_offset=0,
    precision=MlaPrecision.MX,
):
    np.random.seed(42)
    inputs = {
        "q_lift_hbm": np.random.randn(batch, seqlen, hidden, L).astype(ml_dtypes.bfloat16),
        "q_pe_hbm": np.random.randn(batch, seqlen, hidden, R).astype(ml_dtypes.bfloat16),
        "c_kv_hbm": np.random.randn(batch, S_kv, L).astype(ml_dtypes.bfloat16),
        "k_pe_hbm": np.random.randn(batch, S_kv, R).astype(ml_dtypes.bfloat16),
        # Arbitrary but shared with the ref, so validation is scale-invariant.
        "softmax_scale": float(1.0 / np.sqrt(L + R)),
        "precision": precision,
    }
    if dense:
        # Dense (no-indexer): attend all S_kv keys, causal-masked in-kernel. topk_indices is
        # not needed at all -- omit it entirely (the kernel arg is optional / defaults None).
        inputs["dense"] = True
    else:
        # Sparse: the causal re-mask is the default flow now, so the topk always carries the
        # indexer's future fillers and the kernel always gets the GLOBAL query offset.
        inputs["topk_indices_hbm"] = _build_topk_causal_future(batch, seqlen, K, S_kv, q_pos_offset or 0)
    # The GLOBAL query offset is a runtime [1, 1] TENSOR (not a kernel constant) so one traced
    # graph serves every CP rank. q_pos_offset None omits it, exercising the kernel's non-CP
    # default (offset 0).
    if q_pos_offset is not None:
        inputs["q_pos_offset_hbm"] = np.array([[q_pos_offset]], dtype=np.int32)
    return inputs


def _output_tensors(kernel_input):
    batch, seqlen, hidden, L = kernel_input["q_lift_hbm"].shape
    return {"out_attn": np.zeros((batch, seqlen, hidden * L), dtype=ml_dtypes.bfloat16)}


# fmt: off
PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R"
SPARSE_PARAM_NAMES = PARAM_NAMES + ", q_pos_offset"

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
]

BF16_SPARSE_PARAMS = [
    # ---- GLM-MoE-DSA: 64 heads, L=512, R=64, topk 2048 ----
    pytest.param(1, 128, 8192, 64, 2048, 512, 64, 0, id="bf16_glm_S128_Skv8192"),
    pytest.param(1, 256, 4096, 64, 2048, 512, 64, 0, id="bf16_glm_S256_Skv4096"),
    pytest.param(1, 128, 2048, 64, 2048, 512, 64, 0, id="bf16_glm_S128_Skv2048"),
    # GLM under CP (global query offset) at the deployment topk.
    pytest.param(1, 128, 8192, 64, 2048, 512, 64, 256, id="bf16_glm_cprank_S128_Skv8192"),
    # GLM TP shards: 32 heads (TP=2) and 16 heads (TP=4, the max H % 16 allows).
    pytest.param(1, 256, 2048, 32, 512, 512, 64, 0, id="bf16_glm_H32_tp2"),
    pytest.param(1, 256, 2048, 16, 512, 512, 64, 0, id="bf16_glm_H16_tp4"),
    # ---- S_local sweep at the full GLM config (2..2048), so every kernel in the stack is
    # covered at the same per-rank query counts. S_kv is held at 8192 here; the long gathered
    # cache is covered by the 128k rows below.
    pytest.param(1, 2,   8192, 64, 2048, 512, 64, 0, id="bf16_glm_S2"),
    pytest.param(1, 16,  8192, 64, 2048, 512, 64, 0, id="bf16_glm_S16"),
    pytest.param(1, 32,  8192, 64, 2048, 512, 64, 0, id="bf16_glm_S32"),
    pytest.param(1, 64,  8192, 64, 2048, 512, 64, 0, id="bf16_glm_S64"),
    pytest.param(1, 512, 8192, 64, 2048, 512, 64, 0, id="bf16_glm_S512"),
    pytest.param(1, 1024, 8192, 64, 2048, 512, 64, 0, id="bf16_glm_S1024"),
    # ---- DeepSeek dims with the bf16 layout, to isolate the layout from the shapes ----
    pytest.param(1, 128, 8192, 128, 2048, 512, 64, 0, id="bf16_ds_S128_Skv8192"),

    # ---- S_global = 128k, the CP deployment target. Only reachable with the DMA-indirect gather
    pytest.param(1, 1024, 131072, 64, 2048, 512, 64, 0,      id="bf16_glm_S1024_Skv128k"),
    pytest.param(1, 2048, 131072, 64, 2048, 512, 64, 0,      id="bf16_glm_S2048_Skv128k"),
    pytest.param(1, 2048, 131072, 64, 2048, 512, 64, 129024, id="bf16_glm_S2048_Skv128k_lastrank"),
    pytest.param(1, 4096,   262144, 64, 2048, 512, 64, 0,    id="bf16_glm_S4096_Skv256k"),
    pytest.param(1, 8192,   524288, 64, 2048, 512, 64, 0,    id="bf16_glm_S8192_Skv512k"),
    pytest.param(1, 16384, 1048576, 64, 2048, 512, 64, 0,    id="bf16_glm_S16384_Skv1M"),
]

BF16_DENSE_PARAMS = [
    # Dense (S <= index_topk, indexer skipped) at GLM head count. S_local <= 32 is the range
    # production always takes dense -- the indexer is never invoked there -- so the whole
    # small end of the S_local sweep is covered here rather than in the indexer tests.
    pytest.param(1, 2,   128, 64, 16, 512, 64, id="bf16_dense_glm_S2"),
    pytest.param(1, 16,  128, 64, 16, 512, 64, id="bf16_dense_glm_S16"),
    pytest.param(1, 32,  128, 64, 16, 512, 64, id="bf16_dense_glm_S32"),
    pytest.param(1, 128, 128, 64, 16, 512, 64, id="bf16_dense_glm_S128"),
]


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


@pytest_test_metadata(name="SparseMlaLatentAttnCteStandalone", tags=["model"])
@pytest_marks(["attention", "mla_sparse_attention_cte_kernel"])
class TestSparseMlaLatentAttnCte:
    """Standalone sparse MLA latent + RoPE attention (kernel A).

    ``tags=["model"]`` is what enrols this file as its own step in the model-regression
    lane, which is the only lane that emits results under the model run type the
    kernel-to-model dashboard reads. It sits here rather than on the model-driven class
    below because one metadata annotation per file is enforced.
    """

    def _run(
        self,
        test_manager,
        platform_target,
        batch,
        seqlen,
        S_kv,
        hidden,
        K,
        L,
        R,
        dense=False,
        q_pos_offset=0,
        precision=MlaPrecision.MX,
    ):
        def input_generator(test_config):
            return _generate_inputs(
                batch,
                seqlen,
                S_kv,
                hidden,
                K,
                L,
                R,
                dense=dense,
                q_pos_offset=q_pos_offset,
                precision=precision,
            )

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
    @pytest.mark.fast
    @pytest.mark.parametrize(SPARSE_PARAM_NAMES, BF16_SPARSE_PARAMS[:1])
    def test_sparse_mla_latent_attn_cte_bf16_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset
    ):
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
            q_pos_offset=q_pos_offset,
            precision=MlaPrecision.BF16,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(SPARSE_PARAM_NAMES, BF16_SPARSE_PARAMS)
    def test_sparse_mla_latent_attn_cte_bf16(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset
    ):
        """Sparse flow emitting NATURAL latent columns for the bf16 o_proj, at GLM-MoE-DSA dims."""
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
            q_pos_offset=q_pos_offset,
            precision=MlaPrecision.BF16,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, BF16_DENSE_PARAMS)
    def test_dense_mla_latent_attn_cte_bf16(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R
    ):
        """Dense (no-indexer) flow with the bf16 natural-order latent layout."""
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
            precision=MlaPrecision.BF16,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, DENSE_PARAMS)
    def test_dense_mla_latent_attn_cte(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R
    ):
        """Dense causal mask with q_pos_offset OMITTED (non-CP): covers the kernel's offset-0
        default, i.e. the path where no offset tensor is passed at all."""
        self._run(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, dense=True, q_pos_offset=None)

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


@pytest_marks(["attention", "cte", "model", "mla_sparse_attention_cte_kernel"])
class TestSparseMlaLatentAttnCteModel:
    """Model-driven tests for the sparse MLA latent attention CTE kernel, organized by tier.

    Parametrized from configuration/test_vector_metadata/, which is also what feeds the
    kernel-to-model benchmark dashboard, so a config registered there is a config that runs.

    Carries no @pytest_test_metadata of its own: one annotation per file is enforced, and the
    kernel is already named by the class above.
    """

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = prepare_model_parametrize(
        {
            ModelTestType.OPTIMAL: mla_sparse_attention_cte_model_configs.get(ModelTestType.OPTIMAL, []),
        }
    )

    def _run_model_test(
        self, test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset, dense, precision
    ):
        # The config table carries the precision by NAME so it stays free of kernel imports;
        # resolve it here, where the kernel is already a dependency.
        TestSparseMlaLatentAttnCte()._run(
            test_manager,
            platform_target,
            batch,
            seqlen,
            S_kv,
            hidden,
            K,
            L,
            R,
            dense=dense,
            q_pos_offset=q_pos_offset,
            precision=MlaPrecision[precision],
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.optimal
    @pytest.mark.parametrize(MODEL_PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
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
        dense,
        precision,
    ):
        """OPTIMAL: the model configs the benchmark dashboard tracks, DeepSeek-V3.2 and GLM."""
        self._run_model_test(
            test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R, q_pos_offset, dense, precision
        )
