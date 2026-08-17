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


def _build_topk(batch, seqlen, K, S_kv):
    topk = np.zeros((batch, seqlen, K), dtype=np.int32)
    for b in range(batch):
        for s in range(seqlen):
            causal_hi = min(s, S_kv - 1)
            sel = list(range(causal_hi + 1))[::2]
            if len(sel) > K:
                sel = sel[-K:]
            pad = [sel[0]] * (K - len(sel))
            topk[b, s, :] = np.asarray(sorted(pad + sel)[:K], dtype=np.int32)
    return topk


def _generate_inputs(batch, seqlen, S_kv, hidden, K, L, R):
    np.random.seed(42)
    return {
        "q_lift_hbm": np.random.randn(batch, seqlen, hidden, L).astype(ml_dtypes.bfloat16),
        "q_pe_hbm": np.random.randn(batch, seqlen, hidden, R).astype(ml_dtypes.bfloat16),
        "c_kv_hbm": np.random.randn(batch, S_kv, L).astype(ml_dtypes.bfloat16),
        "k_pe_hbm": np.random.randn(batch, S_kv, R).astype(ml_dtypes.bfloat16),
        "topk_indices_hbm": _build_topk(batch, seqlen, K, S_kv),
        "softmax_scale": float(1.0 / np.sqrt(L + R)),
    }


def _output_tensors(kernel_input):
    batch, seqlen, hidden, L = kernel_input["q_lift_hbm"].shape
    return {"out_attn": np.zeros((batch, seqlen, hidden * L), dtype=ml_dtypes.bfloat16)}


# fmt: off
PARAM_NAMES = "batch, seqlen, S_kv, hidden, K, L, R"
TEST_PARAMS = [
    # ---- baseline small ----
    pytest.param(1, 256, 512,  16,  256,  512, 64, id="1_256_512_16_256_512_64"),
    # ---- real DeepSeek-V3.2 dims: seqlen seq-shard 128, S_kv=8192, 128 heads, topk=2048 ----
    pytest.param(1, 128, 8192, 128, 2048, 512, 64, id="1_128_8192_128_2048_512_64_deepseek"),

    # ---- vary K (topk): smallest legal (K=128) up to large; K%128==0 ----
    pytest.param(1, 128, 2048, 16,  128,  512, 64, id="K128_smallest"),
    pytest.param(1, 128, 2048, 16,  512,  512, 64, id="K512"),
    pytest.param(1, 128, 4096, 16,  1024, 512, 64, id="K1024"),

    # vary hidden (heads): 32, 64 (full 128 is the deepseek case above). hidden>=16 required (q_lift
    # dma_transpose output step hidden*bf16 must be 32B-aligned; hidden<16 fails NCC_IBIR155 at compile).
    pytest.param(1, 128, 1024, 32,  256,  512, 64, id="H32"),
    pytest.param(1, 256, 2048, 64,  512,  512, 64, id="H64"),

    # ---- vary seqlen (per-core query count): larger sequence shards ----
    pytest.param(1, 512,  1024, 16, 256,  512, 64, id="S512"),
    pytest.param(1, 1024, 2048, 16, 256,  512, 64, id="S1024"),

    # ---- larger S_kv (cache) with modest topk (sparse selection over long cache) ----
    pytest.param(1, 128, 4096, 32,  512,  512, 64, id="Skv4096_sparse"),
    # NOTE: L is always 512 for DeepSeek-V3.2 (kv_lora_rank=512); L!=512 is out of scope
    # (the MM2 4-pack output permute + torch ref assume the single-latent-tile layout).
]
# fmt: on


@pytest_test_metadata(name="SparseMlaLatentAttnCteStandalone")
@pytest_marks(["attention", "mla_sparse_attention_cte_kernel"])
class TestSparseMlaLatentAttnCte:
    """Standalone sparse MLA latent + RoPE attention (kernel A)."""

    def _run(self, test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R):
        def input_generator(test_config):
            return _generate_inputs(batch, seqlen, S_kv, hidden, K, L, R)

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
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS[:1])
    def test_sparse_mla_latent_attn_cte_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R
    ):
        self._run(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_sparse_mla_latent_attn_cte(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, S_kv, hidden, K, L, R
    ):
        self._run(test_manager, platform_target, batch, seqlen, S_kv, hidden, K, L, R)
