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
