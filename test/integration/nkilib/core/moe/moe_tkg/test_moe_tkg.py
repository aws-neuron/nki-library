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

import functools
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe import moe_tkg
from nkilib_src.nkilib.core.moe.moe_tkg.all_expert_mx_utils import BF16_PER_INT32
from nkilib_src.nkilib.core.moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    DtypeMode,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
    QuantizationType,
)

from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import build_moe_tkg, get_expert_affinity_dtype
from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_wrapper import moe_tkg_sbuf_io_wrapper
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.common_dataclasses import MODEL_TEST_TYPE, TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import (
    UnitTestFramework,
    torch_ref_wrapper,
)

try:
    from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_model_config import moe_tkg_model_configs
except ImportError:
    moe_tkg_model_configs = []


def _fp8_input_dtype_converter(value):
    """Convert fp8/bf16 numpy arrays to torch tensors, preserving dtype."""
    import torch

    dtype_str = str(value.dtype)
    # Dtype converters: preserve fp8 and bf16 dtypes through the torch ref
    _NP_TO_TORCH_FP8 = {
        'float8_e4m3': torch.float8_e4m3fn,  # not supported in torch, use fp8_e4m3fn instead for torch ref
        'float8_e4m3fn': torch.float8_e4m3fn,
        'float8_e5m2': torch.float8_e5m2,
    }
    torch_dtype = _NP_TO_TORCH_FP8.get(dtype_str)
    if torch_dtype is not None:
        return torch.from_numpy(value.astype(np.float32)).to(torch_dtype)
    if 'bfloat16' in dtype_str:
        return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
    return None


# Torch references
moe_tkg_ref = torch_ref_wrapper(moe_tkg_torch_ref)
moe_tkg_ref_fp8_inp = torch_ref_wrapper(moe_tkg_torch_ref, input_dtype_converter=_fp8_input_dtype_converter)


def _resolve_dtype(d):
    """Convert string dtype back to nki dtype (pytest parametrize serializes custom dtypes)."""
    if isinstance(d, str):
        return getattr(nl, d, np.dtype(d))
    return d


class MoeTkgDtypeModeConfig(TypedDict):
    """Shapes, quantization and fusion settings shared by every dtype_mode canary case."""

    vnc: int
    tokens: int
    hidden: int
    intermediate: int
    expert: int
    top_k: int
    act_fn: ActFnType
    scale_mode: ExpertAffinityScaleMode
    clamp: bool
    bias: bool
    q_dtype: str
    dtype: str


class MoeTkgBuildKwargs(TypedDict):
    """Keyword arguments forwarded to the MoE TKG input builder.

    Keys marked as not required are only supplied by the test variants that
    exercise the corresponding feature.
    """

    tokens: int
    hidden: int
    intermediate: int
    expert: int
    top_k: int | None
    act_fn: ActFnType
    expert_affinities_scaling_mode: ExpertAffinityScaleMode
    is_all_expert: bool
    expert_affinities_dtype: str
    in_dtype: str | np.dtype
    out_dtype: str | np.dtype
    bias: bool
    clamp: bool
    quant_dtype: NotRequired[str | np.dtype]
    quant_type: NotRequired[QuantizationType]
    is_all_expert_dynamic: NotRequired[bool]
    routed_token_ratio: NotRequired[float]
    block_size: NotRequired[int | None]
    all_to_all_v_strategy: NotRequired[MoEAllToAllVStrategy]
    dtype_mode: NotRequired[DtypeMode]


def _run_moe_tkg_test(
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
    clamp: bool,
    bias: bool,
    platform_target: Platforms = Platforms.TRN2,
    rtol: float = 2e-2,
    is_negative: bool = False,
    # optional params — present in some test variants
    q_dtype=None,
    q_type=None,
    dtype=None,
    in_dtype=None,
    out_dtype=None,
    is_all_expert_dynamic: bool = False,
    routed_token_ratio: float = 1.0,
    block_size: int | None = None,
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED,
    torch_ref: Callable[..., Any] | None = None,
    dtype_mode=None,
    **_ignored,
):
    """Common test runner for moe_tkg kernel tests."""
    resolved_in = _resolve_dtype(in_dtype if in_dtype is not None else dtype)
    resolved_out = _resolve_dtype(out_dtype if out_dtype is not None else dtype)
    build_kw: MoeTkgBuildKwargs = {
        'tokens': tokens,
        'hidden': hidden,
        'intermediate': intermediate,
        'expert': expert,
        'top_k': top_k,
        'act_fn': act_fn,
        'expert_affinities_scaling_mode': scale_mode,
        'is_all_expert': all_expert,
        'expert_affinities_dtype': get_expert_affinity_dtype(all_expert),
        'in_dtype': resolved_in,
        'out_dtype': resolved_out,
        'bias': bias,
        'clamp': clamp,
    }
    if q_dtype is not None:
        build_kw["quant_dtype"] = _resolve_dtype(q_dtype)
    if q_type is not None:
        build_kw["quant_type"] = q_type
    if is_all_expert_dynamic:
        build_kw["is_all_expert_dynamic"] = True
        build_kw["routed_token_ratio"] = routed_token_ratio
        build_kw["block_size"] = block_size
        build_kw["all_to_all_v_strategy"] = all_to_all_v_strategy
    if dtype_mode is not None:
        build_kw["dtype_mode"] = dtype_mode

    # Non-A2Av I/O shapes match
    if all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:

        def output_tensor_descriptor(ki):
            return {"out": np.zeros(ki["hidden_input"].shape, dtype=ki["output_dtype"])}

    # A2Av I/O have different shapes
    # Input: [T, H + H/4 + 2 * E_L + 4]fp8
    # Output: [T, H + 2]bf16
    else:

        def output_tensor_descriptor(ki):
            return {
                "out": np.zeros(
                    (ki["hidden_input"].shape[0], ki["expert_down_weights"].shape[-1] + BF16_PER_INT32),
                    dtype=ki["output_dtype"],
                )
            }

    # Wrap the torch ref to pre-resolve DtypeMode.AUTO using platform_target.
    # The torch ref runs on CPU and can't query hardware; without this the
    # ref would always clip at 240 regardless of target.
    base_torch_ref = torch_ref if torch_ref is not None else moe_tkg_ref

    @functools.wraps(base_torch_ref)
    def _torch_ref_with_resolved_dtype_mode(**kwargs):
        kernel_dtype_mode = kwargs.get("dtype_mode")
        if kernel_dtype_mode is not None:
            kwargs["dtype_mode"] = resolve_dtype_mode_for_torch_ref(kernel_dtype_mode, platform_target)
        return base_torch_ref(**kwargs)

    test_torch_ref = _torch_ref_with_resolved_dtype_mode

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=moe_tkg,
        torch_ref=test_torch_ref,
        kernel_input_generator=lambda _: build_moe_tkg(**build_kw),
        output_tensor_descriptor=output_tensor_descriptor,
    )
    compiler_args = CompilerArgs(
        logical_nc_config=vnc,
        platform_target=platform_target,
    )
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

# Abbreviation mappings for keyword-prefixed test IDs
_ABBREVS = {
    "vnc": "vnc",
    "tokens": "t",
    "hidden": "h",
    "intermediate": "i",
    "expert": "e",
    "top_k": "k",
    "act_fn": "act",
    "scale_mode": "scaling",
    "all_expert": "all_expert",
    "q_dtype": "quant_dtype",
    "q_type": "quant_type",
    "dtype": "dtype",
    "clamp": "clamp",
    "bias": "bias",
    "routed_token_ratio": "rr",
    "block_size": "bs",
    "in_dtype": "in_dtype",
    "out_dtype": "out_dtype",
    "a2av_strategy": "a2av",
}

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
    (2, 8,  640,   160,  8,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  None,            QuantizationType.NONE, nl.float16, True,  True),
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

    # === FP8 STATIC Quantization (6 tests) ===
    # All experts
    (2, 4,  512,   64,   2,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3, QuantizationType.STATIC, nl.float16, True,  False),
    (2, 4,  1024,  128,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3, QuantizationType.STATIC, nl.float16, True,  True),
    (2, 32, 3072,  192,  4,   None, ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,  nl.float8_e4m3, QuantizationType.STATIC, nl.float16, True,  False),
    # Selective experts
    (2, 4,  512,   64,   4,   2,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3, QuantizationType.STATIC, nl.float16, True,  False),
    (2, 4,  3072,  128,  8,   4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3, QuantizationType.STATIC, nl.float16, True,  True),
    (2, 32, 3072,  192,  8,   4,    ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False, nl.float8_e4m3, QuantizationType.STATIC, nl.float16, True,  False),
]
# fmt: on

# (vnc, tokens, hidden, intermediate, expert) keys for fast tests
_FAST_BFLOAT16_KEYS = frozenset(
    {
        (2, 4, 378, 128, 4),
        (2, 32, 3072, 192, 4),
        (2, 300, 256, 128, 2),
        (2, 4, 512, 64, 4),
        (2, 2, 5120, 128, 16),
        (2, 32, 3072, 512, 1),
        (2, 4, 384, 128, 4),
        (2, 4, 512, 64, 2),
        (2, 1024, 256, 128, 2),
        (2, 256, 5120, 128, 16),
    }
)

MOE_TKG_ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[:5]) in _FAST_BFLOAT16_KEYS else pytest.param(*c)
    for c in MOE_TKG_TEST_PARAMS
]

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
    # T-tiling configs (all-expert only): tile_limit=256 for H=3072/vnc=2, so T>256 triggers tiling
    (2, 384,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 320,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 768,  3072, 3072, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
    (2, 384,  3072, 1536, 1,   None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,  nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True),
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

# (vnc, tokens, hidden, intermediate, expert) keys for MX fast tests
_FAST_MX_KEYS = frozenset(
    {
        (2, 128, 3072, 96, 1),
        (2, 1, 512, 64, 128),
        (2, 128, 3072, 768, 1),
        (2, 1, 3072, 192, 128),
        (2, 2048, 3072, 3072, 1),
        (2, 128, 3072, 180, 1),
        (2, 4, 3072, 192, 128),
        (2, 32, 3072, 3072, 2),
    }
)

MOE_TKG_MX_ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[:5]) in _FAST_MX_KEYS else pytest.param(*c)
    for c in MOE_TKG_MX_PARAMS
]

# fmt: off
_DYNAMIC_DEFAULTS = {
    "top_k": None,
    "scale_mode": ExpertAffinityScaleMode.POST_SCALE,
    "all_expert": True,
}

MOE_TKG_DYNAMISM_PARAM_NAMES = (
    "vnc, tokens, hidden, intermediate, expert, act_fn, "
    "q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size"
)
MOE_TKG_DYNAMISM_PARAMS = [
    # vnc, tokens, hidden, intermediate, expert, act_fn, q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size
    # LNC=1 (single core) DLoC config
    (1, 512,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128),
    # MXFP4 large T, E=128, K=4 — average skew (T*K/E)
    (2, 128,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 16),
    (2, 256,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 32),
    (2, 512,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64),
    (2, 1024, 3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128),
    (2, 2048, 3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 256),
    # Worst case skew (all tokens routed)
    (2, 128,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      16),
    (2, 256,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      32),
    (2, 512,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64),
    (2, 1024, 3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128),
    (2, 2048, 3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256),
    (2, 128,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      16),
    (2, 128,  3072, 3072, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      16),
    (2, 128,  3072, 3072, 8,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      16),
    # Block size sweep (functionality)
    (2, 32,   3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      8),
    (2, 96,   3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      24),
    (2, 192,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      96),
    # E_L>1
    (2, 256,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 32),
    (2, 256,  3072, 3072, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      32),
    (2, 256,  3072, 3072, 8,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 32),
    (2, 256,  3072, 3072, 10, ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      32),
    # MXFP8 large T, E=128, K=8 — average skew
    (2, 256,  4096, 3072, 1,  ActFnType.SiLU,  nl.float8_e4m3fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64),
    # MXFP8 worst case skew
    (2, 256,  4096, 3072, 1,  ActFnType.SiLU,  nl.float8_e4m3fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64),
    # Non-MX (bfloat16) dynamic
    (2, 128,  3072, 384,  1,  ActFnType.SiLU,  None,                QuantizationType.NONE, nl.float16, True, True,  1.0,      32),
    (2, 256,  3072, 384,  1,  ActFnType.SiLU,  None,                QuantizationType.NONE, nl.float16, True, True,  3.125e-2, 64),
    (2, 512,  3072, 384,  1,  ActFnType.SiLU,  None,                QuantizationType.NONE, nl.float16, True, False, 1.0,      128),
]

MOE_TKG_A2AV_PARAM_NAMES = (
    "vnc, tokens, hidden, intermediate, expert, act_fn, "
    "q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size, a2av_strategy"
)
MOE_TKG_A2AV_PARAMS = [
    # vnc, tokens, hidden, intermediate, expert, act_fn, q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size, a2av_strategy
    # MXFP4 large T, E=128, K=4 — average skew (T*K/E)
    (2, 8,    3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 4,   MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 128,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 16,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 256,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 32,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 2048, 3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 3072, 3072, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 3072, 3072, 8,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # Worst case skew (all tokens routed)
    (2, 8,    3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      4,   MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 128,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      16,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 256,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      32,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 2048, 3072, 3072, 1,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 128,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      16,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 3072, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 3072, 3072, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 3072, 1536, 8,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # Generality configs, average and worst case skew
    # H=4K, I/TP=4K, K=4, E_L=2
    (2, 512,  4096, 4096, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  4096, 4096, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # H=5K, I/TP=5K, K=8, E_L=2
    (2, 256,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  32,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 320,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 384,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 448,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 640,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 768,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 896,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1280, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1536, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1792, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 2048, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 2560, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 3072, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 3584, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 4096, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 256,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      32,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 320,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 384,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 448,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 640,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 768,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 896,  5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1280, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1536, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1792, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 2048, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 2560, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 3072, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 3584, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 4096, 5120, 5120, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      256, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # H=5K, I/TP=2.5K, K=8, E_L=4
    # No fused unpermute
    (2, 512,  5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 1024, 5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # Fused unpermute
    (2, 512,  4096, 4096, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 512,  5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 1024, 5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 6.25e-2,  128, MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 512,  4096, 4096, 2,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 512,  5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 1024, 5120, 2560, 4,  ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      128, MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    # E_L not divisible by 4
    (2, 512,  3072, 1536, 10, ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 1536, 10, ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 1536, 10, ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 3.125e-2, 64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 512,  3072, 1536, 10, ActFnType.Swish, nl.float4_e2m1fn_x4, QuantizationType.MX, nl.bfloat16, True, True, 1.0,      64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
]
# fmt: on
# =============================================================================

# fmt: off
MOE_TKG_A2AV_NON_MX_PARAM_NAMES = (
    "vnc, tokens, hidden, intermediate, expert, act_fn, "
    "q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size, a2av_strategy"
)
MOE_TKG_A2AV_NON_MX_PARAMS = [
    # vnc, tokens, hidden, intermediate, expert, act_fn, q_dtype, q_type, dtype, clamp, bias, routed_token_ratio, block_size, a2av_strategy
    # Non-MX (bf16/fp16) DLoC A2AV — PRESERVE_ROW_ORDER
    (2, 128,  3072, 384,  1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      32,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 256,  3072, 384,  1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  3.125e-2, 64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 512,  3072, 384,  1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, False, 1.0,      128, MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # Non-MX (bf16/fp16) DLoC A2AV — PACK_OUTPUT_ROWS
    (2, 128,  3072, 384,  1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      32,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 256,  3072, 384,  1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  3.125e-2, 64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 512,  3072, 384,  1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, False, 1.0,      128, MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    # More aggressive dimension test
    (2, 8,    8192, 2048, 1,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      8,   MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    # E_L > 1 test
    (2, 64,  3072, 384,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,   1e-1, 64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 64,  3072, 384,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,   1e-1, 64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # E2E-matching dims: T=16, H=8192, E=8, partial routing
    (2, 16,  8192, 2048, 8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,   1.25e-1, 16,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 16,  8192, 2048, 8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,   1.25e-1, 16,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 256,  3072, 384,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  3.125e-2, 64,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 256,  3072, 384,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  3.125e-2, 64,  MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 8,    8192, 2048, 8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      8,   MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 8,    8192, 2048, 8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      8,   MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    # Small dimension config: H=640 (hidden/TP), I=160 (intermediate/TP), E_L=8
    (2, 8,    640,  160,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      8,   MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
    (2, 8,    640,  160,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.0,      8,   MoEAllToAllVStrategy.PRESERVE_ROW_ORDER),
    (2, 16,   640,  160,  8,  ActFnType.SiLU,  None, QuantizationType.NONE, nl.float16, True, True,  1.25e-1, 16,  MoEAllToAllVStrategy.PACK_OUTPUT_ROWS),
]
# fmt: on
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

# Compile-time-weighted minimum set for SBUF IO. See nki-fast-test-minimum-set skill.
_FAST_SBUF_IO_KEYS = frozenset(
    {
        (2, 4, 3072, 128, 2),
        (2, 4, 3072, 128, 4),
        (2, 300, 3072, 128, 2),
    }
)

MOE_TKG_SBUF_IO_ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[:5]) in _FAST_SBUF_IO_KEYS else pytest.param(*c)
    for c in MOE_TKG_SBUF_IO_PARAMS
]


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

    dim_tuples = list(zip(_expand(T), _expand(H), _expand(I), _expand(E), strict=True))
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


@pytest_test_metadata(name="MoE TKG")
@pytest_marks(["moe", "tkg"])
class TestMoeTkgKernel:
    @pytest_parametrize(MOE_TKG_PARAM_NAMES, MOE_TKG_ALL_PARAMS, abbrevs=_ABBREVS)
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
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs["is_negative"] = _is_negative_test(vnc, hidden, all_expert, tokens)
        _run_moe_tkg_test(**kwargs)

    @pytest_marks(["mx"])
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(MOE_TKG_PARAM_NAMES, MOE_TKG_MX_ALL_PARAMS, abbrevs=_ABBREVS)
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
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs["rtol"] = 5e-2
        _run_moe_tkg_test(**kwargs)

    @pytest_marks(["mx"])
    @pytest.mark.platforms(exclude=[Platforms.TRN1])
    @pytest_parametrize(MOE_TKG_DYNAMISM_PARAM_NAMES, MOE_TKG_DYNAMISM_PARAMS, abbrevs=_ABBREVS)
    def test_moe_tkg_dynamic(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        act_fn: ActFnType,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        routed_token_ratio: float,
        block_size: int,
        platform_target,
    ):
        # MX dtypes require TRN3+
        if q_dtype in (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4) and platform_target in (Platforms.TRN2,):
            pytest.skip("MX weights require TRN3+")

        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs.update(_DYNAMIC_DEFAULTS)
        kwargs["is_all_expert_dynamic"] = True
        # # We see a tiny number of elements just outside 5% rtol when using very large T, due to high sample size of bf16 error
        kwargs["rtol"] = 5e-2  # if tokens <= 1024 else 5.1e-2
        _run_moe_tkg_test(**kwargs)

    @pytest_marks(["mx"])
    @pytest.mark.tier0
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(MOE_TKG_A2AV_PARAM_NAMES, MOE_TKG_A2AV_PARAMS, abbrevs=_ABBREVS)
    def test_moe_tkg_a2av(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        act_fn: ActFnType,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        routed_token_ratio: float,
        block_size: int,
        a2av_strategy: MoEAllToAllVStrategy,
        platform_target,
    ):
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs.update(_DYNAMIC_DEFAULTS)
        kwargs["is_all_expert_dynamic"] = True
        # At very large T a few elements land just outside 5% rtol (high bf16 sample count);
        # the alt-EMAX golden default tips the maximal-load config (t=2048, rr=1.0) to 5.25%.
        # Relax large-T only; small-T stays at the tight 5%.
        kwargs["rtol"] = 5e-2 if tokens <= 1024 else 6e-2
        kwargs["all_to_all_v_strategy"] = a2av_strategy
        kwargs["torch_ref"] = moe_tkg_ref_fp8_inp
        _run_moe_tkg_test(**kwargs)

    @pytest.mark.platforms(exclude=[Platforms.TRN1])
    @pytest_parametrize(MOE_TKG_A2AV_NON_MX_PARAM_NAMES, MOE_TKG_A2AV_NON_MX_PARAMS, abbrevs=_ABBREVS)
    def test_moe_tkg_a2av_bfloat16(
        self,
        test_manager: Orchestrator,
        vnc: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        act_fn: ActFnType,
        q_dtype,
        q_type: QuantizationType,
        dtype,
        clamp: bool,
        bias: bool,
        routed_token_ratio: float,
        block_size: int,
        a2av_strategy: MoEAllToAllVStrategy,
        platform_target,
    ):
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs.update(_DYNAMIC_DEFAULTS)
        kwargs["is_all_expert_dynamic"] = True
        kwargs["rtol"] = 5e-2
        kwargs["all_to_all_v_strategy"] = a2av_strategy
        _run_moe_tkg_test(**kwargs)

    @pytest_parametrize(MOE_TKG_SBUF_IO_PARAM_NAMES, MOE_TKG_SBUF_IO_ALL_PARAMS, abbrevs=_ABBREVS)
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

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(MOE_TKG_IO_DTYPE_PARAM_NAMES, MOE_TKG_IO_DTYPE_PARAMS, abbrevs=_ABBREVS)
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
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs["rtol"] = 5e-2
        _run_moe_tkg_test(**kwargs)

    @pytest_parametrize(MOE_TKG_SWEEP_PARAM_NAMES, MOE_TKG_SWEEP_PARAMS, abbrevs=_ABBREVS)
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
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs["rtol"] = 2e-2 if dtype != nl.bfloat16 else 3e-2
        kwargs["is_negative"] = _is_negative_test(vnc, hidden, all_expert, tokens)
        _run_moe_tkg_test(**kwargs)

    @pytest.mark.xfail(strict=False, reason="Model coverage test")
    @pytest_parametrize(MOE_TKG_PARAM_NAMES, MOE_TKG_MODEL_PARAMS, abbrevs=_ABBREVS, prefix=MODEL_TEST_TYPE)
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
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        _run_moe_tkg_test(**kwargs)

    # ------------------------------------------------------------------
    # Opt-in FP8 E4M3 canary (dtype_mode).
    #
    # This is a transient test: the flag exists only until every MoE MLP
    # caller migrates to OCP float8_e4m3fn. Remove this test together with
    # the ``dtype_mode`` kwarg on ``moe_tkg()`` once the flag is deleted.
    # ------------------------------------------------------------------
    # Params mirror an existing passing config from ``MOE_TKG_TEST_PARAMS``
    # (row 287: fp8 ROW, all-expert + selective × STATIC, tokens=4) so any
    # failure here is specific to the DtypeMode migration, not to the shape.
    _MOE_TKG_BY_DTYPE_MODE_CONFIG: MoeTkgDtypeModeConfig = {
        'vnc': 2,
        'tokens': 4,
        'hidden': 512,
        'intermediate': 64,
        'expert': 2,
        'top_k': 2,
        'act_fn': ActFnType.SiLU,
        'scale_mode': ExpertAffinityScaleMode.POST_SCALE,
        'clamp': True,
        'bias': True,
        'q_dtype': nl.float8_e4m3,
        'dtype': nl.bfloat16,
    }

    @pytest.mark.fast
    @pytest.mark.parametrize("dtype_mode", [DtypeMode.NON_OCP, DtypeMode.OCP, DtypeMode.AUTO])
    @pytest.mark.parametrize("q_type", [QuantizationType.STATIC, QuantizationType.ROW])
    @pytest.mark.parametrize("all_expert", [False, True])
    def test_moe_tkg_by_dtype_mode(
        self,
        test_manager: Orchestrator,
        q_type: QuantizationType,
        all_expert: bool,
        platform_target,
        dtype_mode: DtypeMode,
    ):
        """Canary exercising every DtypeMode branch on the MoE TKG path.

        Parametrized over STATIC/ROW quant and selective/all-expert dispatch
        so every migrated tile-dtype resolver path is covered.

        NON_OCP → ``nl.float8_e4m3`` (240), any platform.
        OCP     → ``nl.float8_e4m3fn`` (448), TRN3 only.
        AUTO    → ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3`` elsewhere.
        """
        if dtype_mode == DtypeMode.OCP and not platform_target.is_trn3():
            pytest.skip("dtype_mode=DtypeMode.OCP only exercises the OCP path on TRN3")
        _run_moe_tkg_test(
            test_manager=test_manager,
            all_expert=all_expert,
            q_type=q_type,
            platform_target=platform_target,
            dtype_mode=dtype_mode,
            **self._MOE_TKG_BY_DTYPE_MODE_CONFIG,
        )


def _is_negative_test(vnc: int, hidden: int, is_all_expert: bool, tokens: int) -> bool:
    """Check if test should be marked as negative (expected to fail compilation)."""
    # Hidden must be divisible by 128
    if hidden % 128 != 0:
        return True
    # H1 must be evenly divisible by num_shards (vnc_degree),
    # except for BF16 all-expert which supports unbalanced H-sharding
    H1 = hidden // 128
    if H1 % vnc != 0 and not is_all_expert and tokens == 1:
        return True
    return False
