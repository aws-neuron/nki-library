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

from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import build_moe_tkg, get_expert_affinity_dtype
from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_wrapper import moe_tkg_sbuf_io_wrapper
from test.utils.common_dataclasses import TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import (
    UnitTestFramework,
    torch_ref_wrapper,
)

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe import moe_tkg
from nkilib_src.nkilib.core.moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
)

try:
    from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_model_config import moe_tkg_model_configs
except ImportError:
    moe_tkg_model_configs = []

moe_tkg_ref = torch_ref_wrapper(moe_tkg_torch_ref)


def _run_moe_tkg_test(
    test_manager: Orchestrator,
    vnc: int,
    build_kwargs,
    rtol: float = 2e-2,
    platform_target: Platforms = Platforms.TRN2,
    is_negative: bool = False,
):
    """Common test runner for moe_tkg kernel tests."""
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=moe_tkg,
        torch_ref=moe_tkg_ref,
        kernel_input_generator=lambda _: build_moe_tkg(**build_kwargs),
        output_tensor_descriptor=lambda ki: {"out": np.zeros(ki["hidden_input"].shape, dtype=ki["output_dtype"])},
    )
    compiler_args = CompilerArgs(logical_nc_config=vnc, platform_target=platform_target)
    framework.run_test(
        test_config=None,
        compiler_args=compiler_args,
        rtol=rtol,
        atol=1e-5,
        is_negative_test=is_negative,
        inference_args=TKG_INFERENCE_ARGS,
    )


# =============================================================================
# MoE TKG Tests
# =============================================================================

# fmt: off
MOE_TKG_PARAM_NAMES = \
    "vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode,               all_expert, q_dtype,    q_type,                dtype,      clamp, bias"
MOE_TKG_TEST_PARAMS = [
    # === All experts (14 tests) ===
    # Basic (no bias)
    (2, 4,  32768, 256,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    (2, 32, 3072,  512,  1,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    # With bias
    (2, 4,  3072,  64,   4,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 4,  3072,  192,  4,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 32, 3072,  768,  8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 32, 3072,  768,  128, None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 32, 3072,  1536, 4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 32, 3072,  384,  8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),   # gptoss_120b
    (2, 2,  4096,  192,  8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),  # qwen3_235b
    (2, 2,  5120,  128,  128, None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),  # llama4_maverick
    (2, 2,  5120,  128,  16,  None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),  # llama4_scout
    (2, 4,  3072,  384,  4,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    # All experts with T > 128
    (2, 128,  256,  128,  2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    (2, 256,  256,  128,  2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    (2, 512,  256,  128,  2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    (2, 1024, 256,  128,  2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    # All experts with T > 128 and T not divisible by 128
    (2, 300,  256,  128,  2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    # llama4_scout with T=256: hidden=5120, moe_intermediate=8192, num_local_experts=16, top_k=1, tp=64, ep=1
    (2, 256,  5120, 128,  16,  None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),  # 8192/64=128, 16/1=16
    # Model configs with T=512 for perf comparison
    # gptoss_120b T=512
    (2, 512,  3072, 384,  8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    # qwen3_235b_a22b T=512
    (2, 512,  4096, 192,  8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    # llama4_maverick T=512
    (2, 512,  5120, 128,  128, None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    # llama4_scout T=512
    (2, 512,  5120, 128,  16,  None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),
    # Stress tests: T=1024 with TP=1 equivalent configs (larger intermediate sizes)
    # gptoss_120b T=1024, TP=1 equivalent: hidden=3072, moe_intermediate=3072, num_local_experts=128, ep=16 -> i=3072, e=8
    (2, 1024, 3072, 3072, 8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
    # qwen3_235b_a22b T=1024, TP=1 equivalent: hidden=4096, moe_intermediate=1536, num_local_experts=128, ep=16 -> i=1536, e=8
    (2, 1024, 4096, 1536, 8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),

    # Negative tests
    (2, 4,  384,   128,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),  # H=384
    (2, 4,  378,   128,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  False),  # H=378

    # === Selective experts (17 tests) ===
    # Basic (no bias)
    (2, 2,  512,   128,  2,   1,    ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  False),
    # With bias
    (2, 2,  512,   64,   2,   1,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 4,  512,   128,  2,   1,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    # Odd token counts (shard_on_T)
    (2, 3,  512,   128,  2,   1,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 5,  512,   128,  4,   2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 7,  512,   128,  4,   2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    # Large expert counts (E=128)
    (2, 4,  3072,  192,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 1,  3072,  384,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 4,  3072,  384,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 16, 3072,  512,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 32, 3072,  1024, 128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    (2, 32, 3072,  1536, 128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),
    # Model configs
    (2, 32, 3072,  384,  8,   4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  True),   # gptoss_120b
    (2, 2,  4096,  192,  8,   8,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  False),  # qwen3_235b
    (2, 2,  5120,  128,  128, 1,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  False),  # llama4_maverick
    (2, 2,  5120,  128,  16,  1,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  False),  # llama4_scout
    (2, 4,  384,   128,  4,   2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, None,            QuantizationType.NONE, nl.float16, True,  False),

    # === FP8 ROW Quantization (6 tests) ===
    # All experts
    (2, 4,  512,   64,   2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3, QuantizationType.ROW,  nl.float16, True,  True),
    (2, 4,  1024,  128,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3, QuantizationType.ROW,  nl.float16, True,  True),
    (2, 32, 3072,  192,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3, QuantizationType.ROW,  nl.float16, True,  True),
    # Selective experts
    (2, 4,  512,   64,   4,   2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3, QuantizationType.ROW,  nl.float16, True,  True),
    (2, 4,  3072,  128,  8,   4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3, QuantizationType.ROW,  nl.float16, True,  True),
    (2, 32, 3072,  192,  8,   4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3, QuantizationType.ROW,  nl.float16, True,  True),
]
# fmt: on


# =============================================================================
# MX Quantization Tests
# =============================================================================

# fmt: off
MOE_TKG_MX_PARAMS = [
    # vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode,               all_expert, q_dtype,        q_type,              dtype,       clamp, bias
    # === MXFP4 All experts (19 tests) ===
    (2, 32,   3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 1536, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 768,  1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 192,  1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 96,   1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 360,  1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 180,  1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 90,   1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 64,   3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 256,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 512,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 1024, 3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 2048, 3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 640,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 32,   3072, 3072, 2,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 32,   3072, 3072, 4,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 32,   3072, 3072, 8,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 32,   3072, 3072, 16,  None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 128,  4096, 3072, 4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    # === MXFP4 Selective experts (5 tests) ===
    (2, 1,    3072, 192,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 4,    3072, 192,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 4,    3072, 384,  128, 4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 1,    512,  64,   128, 2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 2,    3072, 1536, 128, 8,    ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    # === MXFP8 (2 tests) ===
    (2, 128,  4096, 3072, 4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 2,    3072, 1536, 128, 8,    ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
]
# fmt: on


# =============================================================================
# MX + Dynamic All-Expert Tests
# =============================================================================

# fmt: off
MOE_TKG_DYNAMIC_PARAM_NAMES = (
    "vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode, "
    "all_expert, q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size"
)
MOE_TKG_DYNAMIC_PARAMS = [
    # vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode,               all_expert, q_dtype,              q_type,              dtype,       clamp, bias, routed_token_ratio, block_size
    # MXFP4 large T, E=128, K=4 — average skew (T*K/E)
    (2, 128,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 16),
    (2, 256,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 32),
    (2, 512,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64),
    (2, 1024, 3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128),
    (2, 2048, 3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 256),
    # Worst case skew (all tokens routed)
    (2, 128,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 16),
    (2, 256,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 32),
    (2, 512,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 64),
    (2, 1024, 3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 128),
    (2, 2048, 3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 256),
    # Block size sweep (functionality)
    (2, 32,   3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 8),
    (2, 96,   3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 24),
    (2, 192,  3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 96),
    # MXFP8 large T, E=128, K=8 — average skew
    (2, 256,  4096, 3072, 1, None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True, nl.float8_e4m3fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2, 64),
    # MXFP8 worst case skew
    (2, 256,  4096, 3072, 1, None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True, nl.float8_e4m3fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0, 64),
]
# fmt: on


# =============================================================================
# SBUF I/O Tests
# =============================================================================

# fmt: off
MOE_TKG_SBUF_IO_PARAM_NAMES = \
    "vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode,          all_expert, dtype,       clamp, bias"
MOE_TKG_SBUF_IO_PARAMS = [
    # All-expert SBUF IO
    (2, 4,   3072, 128, 2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.bfloat16, True,  True),
    (2, 4,   3072, 128, 2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.bfloat16, False, False),
    (2, 4,   3072, 128, 2,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.bfloat16, True,  True),
    # All-expert SBUF IO with T > 128 (T=512, real model configs)
    (2, 512, 3072, 384, 8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float16,  True,  True),   # gptoss_120b
    (2, 512, 4096, 192, 8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float16,  True,  False),  # qwen3_235b
    (2, 512, 5120, 128, 16,  None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float16,  True,  False),  # llama4_scout
    (2, 512, 5120, 128, 128, None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float16,  True,  False),  # llama4_maverick
    # T not divisible by pmax (partial last tile)
    (2, 300, 3072, 128, 2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float16,  True,  False),

    # Selective expert SBUF IO
    (2, 4,   3072, 128, 4,   2,    ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False, nl.bfloat16, False, False),
    (2, 4,   3072, 128, 4,   2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.bfloat16, True,  True),

]
# fmt: on


# =============================================================================
# IO Dtype Tests (TRN3)
# =============================================================================

# fmt: off
MOE_TKG_IO_DTYPE_PARAMS = [
    # vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode,           all_expert, q_dtype,       q_type,              in_dtype,   out_dtype,  clamp, bias
    (2, 32, 3072, 3072, 1, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.float16, nl.bfloat16, True, True),
]
# fmt: on

MOE_TKG_IO_DTYPE_PARAM_NAMES = (
    "vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode, "
    "all_expert, q_dtype, q_type, in_dtype, out_dtype, clamp, bias"
)


# =============================================================================
# Sweep Tests
# =============================================================================


def _po2(lo, hi):
    """Generate power-of-2 sequence from lo to hi."""
    v, r = lo, []
    while v <= hi:
        r.append(v)
        v *= 2
    return r


# 4 standard feature configs: (act_fn, scale_mode, all_expert, dtype, clamp, bias, top_k)
_ALL_BASIC = (ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, True, nl.float16, True, False, None)
_ALL_FULL = (ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, True, nl.float16, True, True, None)
_SEL_BASIC = (ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, False, 1)
_SEL_FULL = (ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, True, 4)
_STD_CONFIGS = [_ALL_BASIC, _ALL_FULL, _SEL_BASIC, _SEL_FULL]


def _make_sweep_params(*, T, H, I, E, configs=_STD_CONFIGS, vnc=2):
    """Generate (vnc, T, H, I, E, top_k, act_fn, scale_mode, all_expert, dtype, clamp, bias) tuples.

    Args: T, H, I, E are lists of values; shorter lists are broadcast to match the longest.
    """
    max_len = max(len(T), len(H), len(I), len(E))

    def _expand(lst):
        rep = max_len // len(lst)
        e = []
        for v in lst:
            e.extend([v] * rep)
        while len(e) < max_len:
            e.append(lst[-1])
        return e

    dim_tuples = list(zip(_expand(T), _expand(H), _expand(I), _expand(E)))
    params = []
    for t, h, i, e in dim_tuples:
        for act_fn, scale_mode, all_expert, dtype, clamp, bias, top_k in configs:
            if top_k is not None and top_k > e:
                continue
            params.append((vnc, t, h, i, e, top_k, act_fn, scale_mode, all_expert, dtype, clamp, bias))
    return params


# fmt: off
MOE_TKG_SWEEP_PARAMS = (
    # tokens_sweep
    _make_sweep_params(T=_po2(1, 128), H=[3072], I=[256], E=[4])
    # hidden_sweep
    + _make_sweep_params(T=[4], H=_po2(256, 32768), I=[256], E=[4])
    # intermediate_sweep
    + _make_sweep_params(T=[4], H=[3072], I=_po2(128, 1024), E=[4])
    # intermediate_non_mult_128
    + _make_sweep_params(T=[4], H=[3072], I=list(range(64, 961, 64)), E=[4])
    # expert_sweep
    + _make_sweep_params(T=[4], H=[3072], I=[256], E=_po2(2, 128))
    # combined_sweep
    + _make_sweep_params(T=_po2(2, 32), H=_po2(512, 8192), I=_po2(128, 512), E=_po2(2, 32))
    # large_hidden
    + _make_sweep_params(T=[4], H=_po2(16384, 32768), I=[256], E=[4])
    # large_expert
    + _make_sweep_params(T=_po2(2, 16), H=[3072], I=[128], E=_po2(64, 128))
    # tokens_hidden
    + _make_sweep_params(T=_po2(1, 64), H=_po2(1024, 8192), I=[256], E=[8])
    # expert_intermediate
    + _make_sweep_params(T=[8], H=[3072], I=_po2(128, 512), E=_po2(2, 32))
    # all_expert dtype/scale/actfn sweep
    + _make_sweep_params(T=_po2(1, 128), H=[3072], I=[256], E=[4], configs=[
            (ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, True, nl.float16, True, False, None),
            (ActFnType.SiLU, ExpertAffinityScaleMode.NO_SCALE, True, nl.float16, True, False, None),
            (ActFnType.GELU, ExpertAffinityScaleMode.POST_SCALE, True, nl.float16, True, False, None),
            (ActFnType.GELU_Tanh_Approx, ExpertAffinityScaleMode.POST_SCALE, True, nl.float16, True, False, None),
            (ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True, nl.float16, True, False, None),
            (ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, True, nl.bfloat16, True, False, None),
    ])
    # selective_expert dtype/scale/actfn sweep
    + _make_sweep_params(T=_po2(1, 128), H=[3072], I=[256], E=[4], configs=[
            (ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, False, 1),
            (ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.bfloat16, True, False, 1),
            (ActFnType.Swish, ExpertAffinityScaleMode.NO_SCALE, False, nl.float16, True, False, 1),
            (ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, False, 1),
            (ActFnType.GELU, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, False, 1),
            (ActFnType.GELU_Tanh_Approx, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, False, 1),
    ])
    # topk sweep
    + _make_sweep_params(T=[4], H=[3072], I=[256], E=[16], configs=[
            (ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float16, True, False, k)
            for k in [1, 2, 4, 8]
    ])
)
MOE_TKG_SWEEP_PARAMS = list(dict.fromkeys(MOE_TKG_SWEEP_PARAMS))  # deduplicate overlapping sweeps
# fmt: on

MOE_TKG_SWEEP_PARAM_NAMES = (
    "vnc, tokens, hidden, intermediate, expert, top_k, act_fn, scale_mode, all_expert, dtype, clamp, bias"
)


# =============================================================================
# Model Config Tests (xfail - coverage tests)
# =============================================================================

MOE_TKG_MODEL_PARAMS = [tuple(cfg) for cfg in moe_tkg_model_configs]


@pytest_test_metadata(
    name="MoE TKG",
    pytest_marks=["moe", "tkg"],
)
class TestMoeTkgKernel:
    @pytest.mark.fast
    @pytest.mark.parametrize(MOE_TKG_PARAM_NAMES, MOE_TKG_TEST_PARAMS)
    def test_moe_tkg_bfloat16(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        platform_target,
    ):
        _run_moe_tkg_test(
            test_manager,
            vnc,
            build_kwargs=dict(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                quant_dtype=_resolve_dtype(q_dtype),
                quant_type=q_type,
                in_dtype=_resolve_dtype(dtype),
                out_dtype=_resolve_dtype(dtype),
                bias=bias,
                clamp=clamp,
            ),
            is_negative=_is_negative_test(vnc, hidden, all_expert, tokens),
            platform_target=platform_target,
        )

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(MOE_TKG_PARAM_NAMES, MOE_TKG_MX_PARAMS)
    def test_moe_tkg_mx(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        platform_target,
    ):
        _run_moe_tkg_test(
            test_manager,
            vnc,
            build_kwargs=dict(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                quant_dtype=_resolve_dtype(q_dtype),
                quant_type=q_type,
                in_dtype=_resolve_dtype(dtype),
                out_dtype=_resolve_dtype(dtype),
                bias=bias,
                clamp=clamp,
            ),
            rtol=5e-2,
            platform_target=platform_target,
        )

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(MOE_TKG_DYNAMIC_PARAM_NAMES, MOE_TKG_DYNAMIC_PARAMS)
    def test_moe_tkg_dynamic(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        routed_token_ratio: float,
        block_size: int,
        platform_target,
    ):
        _run_moe_tkg_test(
            test_manager,
            vnc,
            build_kwargs=dict(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                quant_dtype=_resolve_dtype(q_dtype),
                quant_type=q_type,
                in_dtype=_resolve_dtype(dtype),
                out_dtype=_resolve_dtype(dtype),
                bias=bias,
                clamp=clamp,
                is_all_expert_dynamic=True,
                routed_token_ratio=routed_token_ratio,
                block_size=block_size,
            ),
            rtol=5e-2,
            platform_target=platform_target,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(MOE_TKG_SBUF_IO_PARAM_NAMES, MOE_TKG_SBUF_IO_PARAMS)
    def test_moe_tkg_sbuf_io(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        dtype,
        clamp: bool,
        bias: bool,
        platform_target,
    ):
        def input_generator(test_config):
            return build_moe_tkg(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                in_dtype=_resolve_dtype(dtype),
                out_dtype=_resolve_dtype(dtype),
                bias=bias,
                clamp=clamp,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((tokens, hidden), dtype=dtype)}

        # SBUF wrapper ref with matching signature
        def sbuf_wrapper_ref(
            hidden_input,
            gate_up_weights,
            down_weights,
            expert_affinities,
            expert_index,
            is_all_expert,
            rank_id=None,
            gate_up_weights_bias=None,
            down_weights_bias=None,
            expert_affinities_scaling_mode=None,
            activation_fn=None,
            gate_clamp_upper_limit=None,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=None,
            up_clamp_lower_limit=None,
            mask_unselected_experts=False,
        ):
            return moe_tkg_ref(
                hidden_input=hidden_input,
                expert_gate_up_weights=gate_up_weights,
                expert_down_weights=down_weights,
                expert_affinities=expert_affinities,
                expert_index=expert_index,
                is_all_expert=is_all_expert,
                rank_id=rank_id,
                expert_gate_up_bias=gate_up_weights_bias,
                expert_down_bias=down_weights_bias,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                activation_fn=activation_fn,
                gate_clamp_upper_limit=gate_clamp_upper_limit,
                up_clamp_upper_limit=up_clamp_upper_limit,
                up_clamp_lower_limit=up_clamp_lower_limit,
            )

        # Transform kernel_input to wrapper_input format
        def transform_input(kernel_input):
            return {
                "hidden_input": kernel_input["hidden_input"],
                "gate_up_weights": kernel_input["expert_gate_up_weights"],
                "down_weights": kernel_input["expert_down_weights"],
                "expert_affinities": kernel_input["expert_affinities"],
                "expert_index": kernel_input["expert_index"],
                "is_all_expert": all_expert,
                "rank_id": kernel_input.get("rank_id"),
                "gate_up_weights_bias": kernel_input.get("expert_gate_up_bias"),
                "down_weights_bias": kernel_input.get("expert_down_bias"),
                "expert_affinities_scaling_mode": scale_mode,
                "activation_fn": act_fn,
                "gate_clamp_upper_limit": 7.0 if clamp else None,
                "gate_clamp_lower_limit": None,
                "up_clamp_upper_limit": 8.0 if clamp else None,
                "up_clamp_lower_limit": -6.0 if clamp else None,
                "mask_unselected_experts": False,
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_tkg_sbuf_io_wrapper,
            torch_ref=torch_ref_wrapper(sbuf_wrapper_ref),
            kernel_input_generator=lambda _: transform_input(input_generator(None)),
            output_tensor_descriptor=output_tensors,
        )

        compiler_args = CompilerArgs(logical_nc_config=vnc, platform_target=platform_target)
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2,
            atol=1e-5,
            inference_args=TKG_INFERENCE_ARGS,
        )

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(MOE_TKG_IO_DTYPE_PARAM_NAMES, MOE_TKG_IO_DTYPE_PARAMS)
    def test_moe_tkg_mx_io_dtype(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        q_dtype,
        q_type: QuantizationType,
        in_dtype,
        out_dtype,
        clamp: bool,
        bias: bool,
        platform_target,
    ):
        _run_moe_tkg_test(
            test_manager,
            vnc,
            build_kwargs=dict(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                quant_dtype=_resolve_dtype(q_dtype),
                quant_type=q_type,
                in_dtype=_resolve_dtype(in_dtype),
                out_dtype=_resolve_dtype(out_dtype),
                bias=bias,
                clamp=clamp,
            ),
            rtol=5e-2,
            platform_target=platform_target,
        )

    @pytest.mark.parametrize(MOE_TKG_SWEEP_PARAM_NAMES, MOE_TKG_SWEEP_PARAMS)
    def test_moe_tkg_sweep(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        dtype,
        clamp: bool,
        bias: bool,
        platform_target,
    ):
        # xfail selective_expert I=960 topk=4: failing determinism check
        if not all_expert and intermediate == 960 and top_k == 4:
            pytest.xfail("failing determinism check")
        _run_moe_tkg_test(
            test_manager,
            vnc,
            build_kwargs=dict(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                in_dtype=_resolve_dtype(dtype),
                out_dtype=_resolve_dtype(dtype),
                bias=bias,
                clamp=clamp,
            ),
            rtol=2e-2 if dtype != nl.bfloat16 else 3e-2,
            is_negative=_is_negative_test(vnc, hidden, all_expert, tokens),
            platform_target=platform_target,
        )

    @pytest.mark.xfail(strict=False, reason="Model coverage test")
    @pytest.mark.parametrize(MOE_TKG_PARAM_NAMES, MOE_TKG_MODEL_PARAMS)
    def test_moe_tkg_model(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        top_k,
        act_fn: ActFnType,
        scale_mode: ExpertAffinityScaleMode,
        all_expert: bool,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        platform_target,
    ):
        is_negative = _is_negative_test(vnc, hidden, all_expert, tokens)
        assert not is_negative, "Model configs must never be marked as negative test cases"
        _run_moe_tkg_test(
            test_manager,
            vnc,
            build_kwargs=dict(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=scale_mode,
                is_all_expert=all_expert,
                expert_affinities_dtype=get_expert_affinity_dtype(all_expert),
                quant_dtype=_resolve_dtype(q_dtype),
                quant_type=q_type,
                in_dtype=_resolve_dtype(dtype),
                out_dtype=_resolve_dtype(dtype),
                bias=bias,
                clamp=clamp,
            ),
            platform_target=platform_target,
        )


def _resolve_dtype(d):
    """Convert string dtype back to nki dtype (pytest parametrize serializes custom dtypes)."""
    if isinstance(d, str):
        return getattr(nl, d, np.dtype(d))
    return d


def _is_negative_test(vnc: int, hidden: int, is_all_expert: bool, tokens: int) -> bool:
    """Check if test should be marked as negative (expected to fail compilation)."""
    # Hidden for each core must be divisible by 128
    if hidden // vnc % 128 != 0 and (is_all_expert or tokens == 1):
        return True
    # H1 must be evenly divisible by num_shards (vnc_degree)
    H1 = hidden // 128
    if H1 % vnc != 0 and (is_all_expert or tokens == 1):
        return True
    return False
