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

"""Integration tests for the standalone MX V-up + MX o_proj kernel
(``mla_vupmx_oproj_cte_kernel``), kernel batch of the split DeepSeek-V3.2 sparse-MLA
forward. Input is the latent attention output out_attn[batch, seqlen, hidden*L] from kernel A.
"""

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mla.deepseek.mla_common_cte import MlaPrecision
from nkilib_src.nkilib.experimental.mla.deepseek.mla_vup_oproj_cte import (
    mla_vupmx_oproj_cte_kernel,
)
from nkilib_src.nkilib.experimental.mla.deepseek.mla_vup_oproj_cte_torch import (
    mla_vupmx_oproj_cte_torch_ref,
)

from test.integration.nkilib.core.qkv.test_qkv_common import _reduce_mx_scale_to_compact_block128
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _ds128_to_native32(compact_scale, in_dim, out_dim):
    """Broadcast a compact block-128 scale [in//128, ceil(out/128)] to the NATIVE block-32
    MX layout [in//32, out] the kernel consumes with compact_scales=False (lossless: one
    128x128 block = 4 identical 32-K sub-blocks x 128 identical N cols)."""
    s = compact_scale.cpu().numpy() if hasattr(compact_scale, "cpu") else np.asarray(compact_scale)
    full = np.repeat(np.repeat(s, 4, axis=0), 128, axis=1)
    return np.ascontiguousarray(full[: in_dim // 32, :out_dim]).astype(np.uint8)


def _generate_inputs(batch, seqlen, hidden, L, D_V, HID, compact_scales=True):
    np.random.seed(42)
    # Latent attention output (as if produced by kernel A); scaled to a benign range.
    out_attn = (np.random.randn(batch, seqlen, hidden * L).astype(np.float32) * 0.3).astype(ml_dtypes.bfloat16)
    _, wuv_qtz, wuv_scale = generate_stabilized_mx_data(nl.float8_e4m3fn_x4, (hidden * L // 4, D_V * 4), val_range=3)
    wuv_scale_compact = _reduce_mx_scale_to_compact_block128(wuv_scale, hidden * L, D_V)
    Hdv = hidden * D_V
    _, wo_qtz, wo_scale = generate_stabilized_mx_data(nl.float8_e4m3fn_x4, (Hdv // 4, HID * 4), val_range=5)
    wo_scale_compact = _reduce_mx_scale_to_compact_block128(wo_scale, Hdv, HID)
    if compact_scales:
        wuv_scale_out, wo_scale_out = wuv_scale_compact, wo_scale_compact
    else:
        # Native block-32 A/B: broadcast the compact scales to [K//32, N] offline (lossless).
        wuv_scale_out = _ds128_to_native32(wuv_scale_compact, hidden * L, D_V)
        wo_scale_out = _ds128_to_native32(wo_scale_compact, Hdv, HID)
    return {
        "out_attn_hbm": out_attn,
        "wuv_qtz_hbm": wuv_qtz,
        "wuv_scale_hbm": wuv_scale_out,
        "wo_qtz_hbm": wo_qtz,
        "wo_scale_hbm": wo_scale_out,
        "compact_scales": compact_scales,
    }


def _output_tensors(kernel_input):
    batch, seqlen, _HL = kernel_input["out_attn_hbm"].shape
    HID = kernel_input["wo_qtz_hbm"].shape[1]
    return {"out": np.zeros((batch, seqlen, HID), dtype=ml_dtypes.bfloat16)}


# fmt: off
# L must be 512 (one MX 512-tile per head). D_V = value head dim (== 128); HID = o_proj
# out (% 512). hidden must be a multiple of 4 (>= 4) and <= 128.
PARAM_NAMES = "batch, seqlen, hidden, L, D_V, HID, compact_scales"
TEST_PARAMS = [
    # ---- baseline small ----
    pytest.param(1, 256, 16,  512, 128, 1024, True, id="1_256_16_512_128_1024"),
    # ---- real DeepSeek-V3.2 dims: seqlen seq-shard 128, 128 heads, HID=7168 ----
    pytest.param(1, 128, 128, 512, 128, 7168, True, id="1_128_128_512_128_7168_deepseek"),

    # ---- vary hidden (heads/rank): 4 (kernel min, Hdv=512=one MX tile), 32, 64 ----
    pytest.param(1, 256, 4,   512, 128, 1024, True, id="H4_min"),
    pytest.param(1, 256, 32,  512, 128, 2048, True, id="H32"),
    pytest.param(1, 128, 64,  512, 128, 7168, True, id="H64"),

    # ---- SMALL sequence lengths at the full DeepSeek config (128 heads, HID=7168) ----
    pytest.param(1, 2,  128, 512, 128, 7168, True, id="S2_deepseek"),
    pytest.param(1, 16, 128, 512, 128, 7168, True, id="S16_deepseek"),
    pytest.param(1, 32, 128, 512, 128, 7168, True, id="S32_deepseek"),

    # ---- vary HID (o_proj out width, % 512): small, mid, non-power-of-2-of-1024 ----
    pytest.param(1, 256, 16,  512, 128, 512,  True, id="HID512"),
    pytest.param(1, 256, 16,  512, 128, 1536, True, id="HID1536"),
    pytest.param(1, 256, 16,  512, 128, 4096, True, id="HID4096"),

    # ---- multi-s-batch (seqlen/n_prgs > _OPROJ_S_TILE=128) exercising the s-batch streaming loop ----
    pytest.param(1, 512,  4, 512, 128, 1024, True, id="S512_multibatch_H4"),
    pytest.param(1, 1024, 4, 512, 128, 2048, True, id="S1024_multibatch_H4"),

    # ---- head-sharded long-seq (TIER-1 full-W_o residency: small Hdv, big seqlen) ----
    pytest.param(1, 4096, 4, 512, 128, 7168, True, id="headshard_tp32_cp2_4096_4_512_128_7168"),
    pytest.param(1, 8192, 4, 512, 128, 7168, True, id="headshard_tp32_cp1_8192_4_512_128_7168"),

    # ---- NATIVE block-32 scales (compact_scales=False) ----
    pytest.param(1, 256, 16,  512, 128, 1024, False, id="1_256_16_512_128_1024_native32"),
    pytest.param(1, 128, 128, 512, 128, 7168, False, id="1_128_128_512_128_7168_deepseek_native32"),
]
# fmt: on


def _generate_inputs_bf16(batch, seqlen, hidden, L, D_V, HID):
    """BF16 (``MlaPrecision.BF16``) inputs: plain bf16, natural column order, no scales.

    Weights are scaled by ``1 / sqrt(K)`` so each matmul output stays roughly unit variance —
    bf16 has ~3 decimal digits of mantissa, so letting the two chained projections drift by
    orders of magnitude would measure dynamic range rather than the kernel.
    """
    np.random.seed(42)
    Hdv = hidden * D_V
    return {
        # Natural latent column order (bf16 kernel A writes no 4-pack permutation).
        "out_attn_hbm": (np.random.randn(batch, seqlen, hidden * L) * 0.3).astype(ml_dtypes.bfloat16),
        "wuv_qtz_hbm": (np.random.randn(hidden * L, D_V) * L**-0.5).astype(ml_dtypes.bfloat16),
        "wuv_scale_hbm": None,  # unused in bf16
        "wo_qtz_hbm": (np.random.randn(Hdv, HID) * Hdv**-0.5).astype(ml_dtypes.bfloat16),
        "wo_scale_hbm": None,  # unused in bf16
        "compact_scales": True,  # ignored in bf16
        "precision": MlaPrecision.BF16,
        "kv_lora_rank": L,
    }


def _output_tensors_bf16(kernel_input):
    batch, seqlen, _HL = kernel_input["out_attn_hbm"].shape
    HID = kernel_input["wo_qtz_hbm"].shape[1]
    return {"out": np.zeros((batch, seqlen, HID), dtype=ml_dtypes.bfloat16)}


# fmt: off
PARAM_NAMES_BF16 = "batch, seqlen, hidden, L, D_V, HID"
TEST_PARAMS_BF16 = [
    # ---- DeepSeek-V3.2 dims in bf16 (D_V=128): parity with the MX cases above ----
    pytest.param(1, 256, 16,  512, 128, 1024, id="ds_1_256_16_1024"),
    pytest.param(1, 128, 128, 512, 128, 7168, id="ds_1_128_128_7168_full"),
    pytest.param(1, 256, 4,   512, 128, 1024, id="ds_H4"),
    pytest.param(1, 128, 64,  512, 128, 7168, id="ds_H64"),

    # ---- GLM-MoE-DSA (D_V=256, HID=6144) ----
    pytest.param(1, 256, 8,   512, 256, 6144, id="glm_tp8"),
    pytest.param(1, 128, 16,  512, 256, 6144, id="glm_tp4"),
    pytest.param(1, 256, 32,  512, 256, 6144, id="glm_tp2"),
    pytest.param(1, 128, 64,  512, 256, 6144, id="glm_full_64h"),
    pytest.param(1, 256, 64,  512, 256, 6144, id="glm_full_64h_s256"),
    # GLM prefill-length reference config (multi-s-batch: seqlen/2 cores > 128).
    pytest.param(1, 2048, 8,  512, 256, 6144, id="glm_s2048_tp8"),
    pytest.param(1, 4096, 4,  512, 256, 6144, id="glm_s4096_tp16"),

    # ---- odd head counts: bf16 drops the MX `hidden % 4 == 0` 4-pack constraint ----
    pytest.param(1, 256, 2,   512, 256, 6144, id="glm_H2"),
    pytest.param(1, 256, 1,   512, 256, 6144, id="glm_H1"),

    # ---- S_local sweep at the full GLM config (2..2048), matched across all four kernels in
    # the stack. seqlen/2 is the per-core count, so 512/1024/2048 exercise the multi-s-batch
    # streaming loop (s_per_core > _OPROJ_S_TILE=128).
    pytest.param(1, 2,    64, 512, 256, 6144, id="glm_S2"),
    pytest.param(1, 16,   64, 512, 256, 6144, id="glm_S16"),
    pytest.param(1, 32,   64, 512, 256, 6144, id="glm_S32"),
    pytest.param(1, 64,   64, 512, 256, 6144, id="glm_S64"),
    pytest.param(1, 512,  64, 512, 256, 6144, id="glm_S512"),
    pytest.param(1, 1024, 64, 512, 256, 6144, id="glm_S1024"),
    pytest.param(1, 2048, 64, 512, 256, 6144, id="glm_S2048_64h"),

    # ---- vary HID ----
    pytest.param(1, 256, 8, 512, 256, 512,  id="glm_HID512"),
    pytest.param(1, 256, 8, 512, 256, 1536, id="glm_HID1536"),

    # ---- long-context per-rank shards (head-sharded long-seq, H=4) ----
    # NOTE: `seqlen` here is the per-rank shard, not the global context. With the tests'
    # x64 sharding convention (S_local * 64), seqlen=8192 corresponds to a 512k global
    # context. With H=4 (Hdv small), out_attn [1, S, H*L] and out [1, S, HID] stay within
    # HBM at these S. These drive num_s_batches = S/128 up to 128, stressing the
    # multi-s-batch streaming loop far past the existing 4096/8192 long-seq cases.
    #   256k global context / x64 -> S_local = 4096
    pytest.param(1, 4096,  4, 512, 256, 6144, id="glm_256k_ctx_s4096"),
    #   512k global context / x64 -> S_local = 8192
    pytest.param(1, 8192,  4, 512, 256, 6144, id="glm_512k_ctx_s8192"),
    #   deeper s-batch loop (S_local = 16384)
    pytest.param(1, 16384, 4, 512, 256, 6144, id="glm_long_ctx_s16384"),
]
# One DeepSeek-dims case + one GLM case (exercising D_V=256) for the fast lane.
TEST_PARAMS_BF16_FAST = [TEST_PARAMS_BF16[0], TEST_PARAMS_BF16[4]]
# fmt: on


@pytest_test_metadata(name="SparseMlaVupMxOprojKernelCte")
@pytest_marks(["attention", "sparse_mla_vupmx_oproj_cte"])
class TestSparseMlaVupMxOprojKernelCte:
    """Standalone MX V-up + MX o_proj (kernel batch)."""

    def _run(self, test_manager, platform_target, batch, seqlen, hidden, L, D_V, HID, compact_scales=True):
        def input_generator(test_config):
            return _generate_inputs(batch, seqlen, hidden, L, D_V, HID, compact_scales=compact_scales)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_vupmx_oproj_cte_kernel,
            torch_ref=torch_ref_wrapper(mla_vupmx_oproj_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
                additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
            ),
            # rtol matches the sibling MX V-up + o_proj test
            # (test_qkv_mla_absorbed_attn_vupmx_oproj_cte): the two-stage MX-fp8 matmul
            # (activation quantize + dequant weights) runs ~5-10% relative error, which the
            # smallest head count (hidden=4, fewest reduction terms) sits right at. 5% was too tight.
            atol=5e-2,
            rtol=1e-1,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS[:1])
    def test_mla_vupmx_oproj_cte_kernel_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, hidden, L, D_V, HID, compact_scales
    ):
        self._run(test_manager, platform_target, batch, seqlen, hidden, L, D_V, HID, compact_scales=compact_scales)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_mla_vupmx_oproj_cte_kernel(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, hidden, L, D_V, HID, compact_scales
    ):
        self._run(test_manager, platform_target, batch, seqlen, hidden, L, D_V, HID, compact_scales=compact_scales)

    # ---------------------------------------------------------------- BF16 ----
    def _run_bf16(self, test_manager, platform_target, batch, seqlen, hidden, L, D_V, HID):
        def input_generator(test_config):
            return _generate_inputs_bf16(batch, seqlen, hidden, L, D_V, HID)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mla_vupmx_oproj_cte_kernel,
            torch_ref=torch_ref_wrapper(mla_vupmx_oproj_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors_bf16,
        )
        """
        Much tighter than the MX path's rtol=1e-1: bf16 has ~8 mantissa bits against MX fp8's
        ~3 plus a block scale, and the tensor engine accumulates in fp32, so the only error is
        bf16 rounding of the two matmuls' operands.
        """
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            atol=1e-2,
            rtol=4e-2,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES_BF16, TEST_PARAMS_BF16_FAST)
    def test_mla_vup_oproj_cte_bf16_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, hidden, L, D_V, HID
    ):
        self._run_bf16(test_manager, platform_target, batch, seqlen, hidden, L, D_V, HID)

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES_BF16, TEST_PARAMS_BF16)
    def test_mla_vup_oproj_cte_bf16(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, seqlen, hidden, L, D_V, HID
    ):
        self._run_bf16(test_manager, platform_target, batch, seqlen, hidden, L, D_V, HID)
