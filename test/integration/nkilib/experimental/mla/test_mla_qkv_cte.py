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

"""Integration tests for the absorbed-latent MLA QKV CTE kernel."""

import functools
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mla.deepseek.mla_qkv_cte import mla_qkv_cte_kernel
from nkilib_src.nkilib.experimental.mla.deepseek.mla_qkv_cte_torch import mla_qkv_cte_torch_ref

from test.integration.nkilib.core.moe.moe_cte.test_utils import build_prequantized_hidden_concat
from test.integration.nkilib.core.qkv.test_qkv_common import build_qkv_mla_input
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@functools.wraps(mla_qkv_cte_torch_ref)
def _qkv_ref_no_qr(*args, **kwargs):
    """The shared ref also returns ``qr`` (for the fused indexer golden). The
    standalone kernel emits only q_lift/q_pe/c_kv/k_pe, so drop the extra key.
    functools.wraps preserves the signature the test framework validates."""
    out = mla_qkv_cte_torch_ref(*args, **kwargs)
    out.pop("qr", None)
    return out


def _build_wuk_bf16(n_heads, qk_nope_head_dim, kv_lora_rank):
    """Build the absorption weight ``W_uk`` (the K_b half of kv_b_proj) as a
    plain bf16 array for nope-contraction.

    W_uk has logical shape [nope, n_heads * kv_lora]; contraction = nope, head
    ``h`` owns columns [h*kv_lora, (h+1)*kv_lora). Values are small (randn*0.1)
    to keep the bf16 absorption in a benign numerical range.
    """
    out_dim = n_heads * kv_lora_rank
    wuk = (np.random.randn(qk_nope_head_dim, out_dim) * 0.1).astype(nl.bfloat16)
    return wuk


@pytest_test_metadata(name="MLA QKV CTE", tags=["model"])
@pytest_marks(["qkv", "cte", "mx", "mla"])
@final
class TestMlaQkvCteKernel:
    # fmt: off
    qkv_mla_absorbed_test_params = (
        "vnc_degree, batch, seqlen, hidden, n_heads, qk_lora_rank, "
        "kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, norm_eps"
    )
    qkv_mla_absorbed_test_perms = [
        # Small H (2048) avoids the simulator's large-shape auto-skip (H=7168 is
        # in the skip list) so these cases run numerically in sim. Full 128 heads.
        pytest.param(2, 1, 256, 2048, 128, 1536, 512, 64, 128, 1e-6, marks=pytest.mark.fast),
        [2, 1, 512, 2048, 128, 1536, 512, 64, 128, 1e-6],
        [2, 1, 128, 2048, 128, 1536, 512, 64, 128, 1e-6],
        [1, 1, 128, 2048, 16, 1536, 512, 64, 128, 1e-6],
        [2, 1, 128, 7168, 128, 1536, 512, 64, 128, 1e-6],
        [2, 1, 256, 7168, 128, 1536, 512, 64, 128, 1e-6],
        [2, 1, 512, 7168, 128, 1536, 512, 64, 128, 1e-6],
        [2, 1, 256, 7168, 64, 1536, 512, 64, 128, 1e-6],
        [2, 1, 512, 7168, 32, 1536, 512, 64, 128, 1e-6],
        [2, 1, 4096, 7168, 4, 1536, 512, 64, 128, 1e-6],
        [2, 1, 8192, 7168, 2, 1536, 512, 64, 128, 1e-6],
    ]
    # fmt: on

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(qkv_mla_absorbed_test_params, qkv_mla_absorbed_test_perms)
    def test_mla_qkv_cte_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        hidden,
        n_heads,
        qk_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        norm_eps,
    ):
        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
            additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
        )

        def input_generator(test_config):
            # Reuse the v32 builder for the shared first/second Q projections,
            # norm gammas and RoPE caches; drop the un-absorbed wkv_b path and
            # add the absorption weight W_uk.
            v32 = build_qkv_mla_input(
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden,
                qk_lora_rank=qk_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                n_heads=n_heads,
                variant="v32",
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=qk_nope_head_dim,
            )
            np.random.seed(42)
            wuk_hbm = _build_wuk_bf16(n_heads, qk_nope_head_dim, kv_lora_rank)
            # PACKED MX input (rmsnorm_mx_prefill pack_scales=True), built by the proven MoE
            # helper; the torch ref decodes the same concat so both see identical activations.
            n_H512 = hidden // (128 * 4)
            concat, _ = build_prequantized_hidden_concat(batch * seqlen, n_H512, hidden)
            return {
                "x_hbm_mx": np.asarray(concat).reshape(batch, seqlen, -1),
                "wqkv_a_hbm": v32["wqkv_a_hbm"],
                "wqkv_a_scale_hbm": v32["wqkv_a_scale_hbm"],
                "wq_b_hbm": v32["wq_b_hbm"],
                "wq_b_scale_hbm": v32["wq_b_scale_hbm"],
                "q_norm_gamma_hbm": v32["q_norm_gamma_hbm"],
                "kv_norm_gamma_hbm": v32["kv_norm_gamma_hbm"],
                "wuk_hbm": wuk_hbm,
                "cos_cache_hbm": v32["cos_cache_hbm"],
                "sin_cache_hbm": v32["sin_cache_hbm"],
                "n_heads": n_heads,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
                "kv_lora_rank": kv_lora_rank,
                "qk_lora_rank": qk_lora_rank,
                "norm_eps": norm_eps,
            }

        def output_tensor_descriptor(kernel_input):
            return {
                "q_lift": np.zeros((batch, seqlen, n_heads, kv_lora_rank), dtype=nl.bfloat16),
                "q_pe": np.zeros((batch, seqlen, n_heads, qk_rope_head_dim), dtype=nl.bfloat16),
                "c_kv": np.zeros((batch, seqlen, kv_lora_rank), dtype=nl.bfloat16),
                "k_pe": np.zeros((batch, seqlen, qk_rope_head_dim), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_qkv_cte_kernel,
            torch_ref=torch_ref_wrapper(_qkv_ref_no_qr),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
            check_unused_params=True,
        )
        # q_lift passes through two MX matmuls (stage1 + Q stage2) then a bf16
        # absorption matmul; the bf16 final stage adds little error beyond the
        # upstream MX rounding, so the standard MX rtol covers it.
        framework.run_test(test_config=None, compiler_args=compiler_args, rtol=8e-2, atol=1e-2)
