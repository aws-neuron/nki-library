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

"""Test suite for MoE BWMM MX CTE kernels using UnitTestFramework."""

import random
from collections.abc import Callable
from typing import Any, final

from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import map_skip_mode
from test.integration.nkilib.core.moe.moe_cte.test_utils import (
    block128_to_native_down,
    block128_to_native_gate_up,
    build_moe_bwmm_mx_cte,
    build_moe_bwmm_mx_cte_from_model_test_config,
    dequant_prequantized_hidden_concat,
    gather_from_packed_down,
    gather_from_packed_gate_up,
    n_packed_buffers_for,
    order_kernel_input,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.model_test_configs import no_model_configs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

try:
    from test.integration.nkilib.core.moe.moe_cte.test_moe_bwmm_mx_cte_model_config import (
        moe_bwmm_mx_cte_model_configs,
    )
except ImportError:
    moe_bwmm_mx_cte_model_configs = no_model_configs()

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx import bwmm_shard_on_block_mx
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx_torch import bwmm_shard_on_block_mx_torch_ref
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I_mx import (
    blockwise_mm_shard_intermediate_mx,
    blockwise_mm_shard_intermediate_mx_hybrid,
)
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I_mx_torch import (
    blockwise_mm_shard_intermediate_mx_hybrid_torch_ref,
    blockwise_mm_shard_intermediate_mx_torch_ref,
)
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from typing_extensions import override

# fmt: off
SHARD_ON_BLOCK_PARAMS = "vnc_degree, hidden, tokens, intermediate, expert, block_size, top_k, ep_degree, act_fn, expert_affinities_scaling_mode, dtype, weight_dtype, skip_mode, bias, is_dynamic, gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower, use_uint_weights, unpacked_weights, n_static_blocks, use_packed_scales, quant_type"

MOE_BWMM_MX_CTE_MODEL_PARAMS = "variant, vnc_degree, hidden, tokens, intermediate, expert, block_size, act_fn, expert_affinities_scaling_mode, dtype, weight_dtype, skip_mode, bias, is_dynamic, gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower, use_uint_weights, skewness_pct, global_top_k, ep_degree"

# ============================================================================
# Test Parameters - Shard-on-Block (unit tests)
# ============================================================================

# Full-only entries: excluded from fast suite (memory >3000 MB)
_SHARD_BLOCK_FULL_ONLY = [
    # MXFP4 test cases (blk=256)
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, True, QuantizationType.MX],
    [2, 3072, 8192, 1536, 64, 256, 4, 2, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, False, None, True, QuantizationType.MX],
    [2, 3072, 2048, 3072, 16, 256, 4, 2, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, True, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 1536, 32, 256, 4, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 1536, 32, 256, 4, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 2048, 3072, 16, 256, 4, 2, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # Scale Packing
    [2, 3072, 256, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, True, QuantizationType.MX],
    # MXFP4 test cases (blk=512)
    [2, 3072, 10240, 384, 128, 512, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 512, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 512, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 512, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # MXFP8 e4m3fn test cases
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # MXFP8 e5m2 test cases
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 10240, 384, 128, 256, 4, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # uint weights test cases (simulates NxD behavior: uint16 for MXFP4, uint32 for MXFP8)
    [2, 3072, 1024, 384, 128, 256, 2, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 384, 128, 256, 2, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 384, 128, 256, 2, 1, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    [2, 3072, 128, 3072, 2, 256, 2, 64, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    # No Bias and No Clipping
    [2, 3072, 1024, 384, 8, 128, 4, 16, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, False, True, None, None, None, None, False, False, None, False, QuantizationType.MX],
    # weight skipping
    [2, 3072, 1024, 384, 16, 256, 2, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 384, 16, 256, 2, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # STATIC_MX large-config
    [2, 4096, 10240, 1536, 16, 256, 2, 8, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 3, False, True, None, None, None, None, False, True, None, False, QuantizationType.STATIC_MX],
    # No Bias with partial Clipping (demoted from fast by min-set workflow)
    [2, 3072, 1024, 384, 8, 128, 4, 16, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, False, True, 7.0, None, None, None, False, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 384, 8, 128, 4, 16, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, False, False, None, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # STATIC_MX with smaller block (demoted from fast by min-set workflow)
    [2, 4096, 4096, 1536, 2, 128, 2, 64, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 3, False, True, None, None, None, None, False, True, None, False, QuantizationType.STATIC_MX],
]


_SHARD_BLOCK_FAST_RAW = [
    # n_static_blocks test cases
    [2, 4096, 4096, 1536, 2, 256, 2, 64, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 3, False, True, None, None, None, None, False, False, 2, False, QuantizationType.MX],
    [2, 4096, 4096, 1024, 4, 256, 4, 32, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, 2, False, QuantizationType.MX],
    [2, 3072, 4096, 384, 2, 256, 2, 64, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, 2, False, QuantizationType.MX],
    # STATIC_MX coverage (MXFP8 e4m3fn weights, unpacked fp8 carriers, no clamp)
    [2, 4096, 4096, 1536, 2, 256, 2, 64, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 3, False, True, None, None, None, None, False, True, None, False, QuantizationType.STATIC_MX],
    [2, 4096, 10240, 768, 4, 256, 2, 32, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 3, False, True, None, None, None, None, False, True, None, False, QuantizationType.STATIC_MX],
    [2, 4096, 4096, 384, 16, 256, 2, 8, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, True, None, False, QuantizationType.STATIC_MX],
]

SHARD_ON_BLOCK_UNIT_PERMS = [
    pytest.param(*c, marks=pytest.mark.fast) for c in _SHARD_BLOCK_FAST_RAW
] + _SHARD_BLOCK_FULL_ONLY

# ============================================================================
# Test Parameters - Shard-on-I-MX (unit tests)
# ============================================================================

SHARD_ON_I_UNIT_PERMS = [
    [2, 3072, 1024, 2048, 8, 256, 4, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 2048, 8, 256, 4, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 7168, 10240, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 7168, 10240, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 7168, 1024, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 7168, 1024, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 7168, 10240, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 7168, 10240, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    # Alternative dtype weights test cases (simulates NxD behavior: uint16 for MXFP4, uint32 for MXFP8)
    [2, 3072, 1024, 2048, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 2048, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 2048, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, True, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 2048, 8, 256, 4, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, False, True, 7.0, None, 7.0, -7.0, False, False, None, False, QuantizationType.MX],
    [2, 3072, 1024, 2048, 8, 256, 4, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, False, True, 7.0, None, None, None, False, False, None, False, QuantizationType.MX],
    # n_static_blocks test case
    [2, 4096, 4096, 2048, 2, 512, 2, 64, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, 2, False, QuantizationType.MX],
    [2, 4096, 4096, 2048, 2, 256, 2, 64, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False, False, 2, False, QuantizationType.MX],
]

# Shard-on-I block-128 (DeepSeek ue8m0) scale configs. I_TP must be a multiple of 1024.
# Params: vnc, hidden, tokens, intermediate, expert, block_size, top_k, ep_degree, act_fn,
#         eas_mode, dtype, weight_dtype, skip_mode, bias, is_dynamic, gcu, gcl, ucu, ucl
# NOTE: intentionally NOT marked @pytest.mark.fast — like SHARD_ON_I_UNIT_PERMS, shard-on-I MX
# compiles are memory-heavy and the fast dry-run suite already runs near the 3000 MB per-worker
# limit. Adding these to the fast suite tips borderline co-scheduled tests over. These run in the
# full shared-fleet suite instead (validated on trn3_a0 with compile-and-infer).
SHARD_ON_I_BLOCK128_PERMS = [
    [2, 3072, 1024, 2048, 8, 256, 4, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 1024, 2048, 8, 256, 4, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 7168, 1024, 1024, 8, 256, 8, 16, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0],
]
SHARD_ON_I_BLOCK128_PARAMS = "vnc_degree, hidden, tokens, intermediate, expert, block_size, top_k, ep_degree, act_fn, expert_affinities_scaling_mode, dtype, weight_dtype, skip_mode, bias, is_dynamic, gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower"
# fmt: on

_ABBREVS = {
    "vnc_degree": "vnc",
    "hidden": "hid",
    "tokens": "tok",
    "intermediate": "inter",
    "expert": "exp",
    "block_size": "blk",
    "top_k": "k",
    "ep_degree": "ep",
    "ep_rank": "eprk",
    "act_fn": "act",
    "expert_affinities_scaling_mode": "eas_mode",
    "dtype": "dt",
    "weight_dtype": "wdt",
    "skip_mode": "sk",
    "bias": "bi",
    "is_dynamic": "dyn",
    "gate_clamp_upper": "gcu",
    "gate_clamp_lower": "gcl",
    "up_clamp_upper": "ucu",
    "up_clamp_lower": "ucl",
    "use_uint_weights": "uint",
    "unpacked_weights": "unp",
    "use_packed_scales": "pack",
    "pack_affinities": "paff",
    "packed_affinities_dtype": "adt",
}

# Per-target maps from packed _x4 weight dtype to the framework's carrier dtype.
#   "uint":         NxD torch/xla simulation (uint16 for MXFP4, uint32 for MXFP8).
#   "unpacked_fp8": STATIC_MX path that hands the kernel scalar fp8 carriers; the
#                   kernel's convert_to_mxfp_dtype reinterpret_casts back to _x4.
_MX_WEIGHT_VIEW_TARGETS = {
    "uint": {
        nl.float4_e2m1fn_x4: np.uint16,
        nl.float8_e4m3fn_x4: np.uint32,
        nl.float8_e5m2_x4: np.uint32,
    },
    "unpacked_fp8": {
        nl.float8_e4m3fn_x4: nl.float8_e4m3fn,
        nl.float8_e5m2_x4: nl.float8_e5m2,
    },
}


def convert_mx_weights_view(kernel_input: dict, weight_dtype: Any, target: str) -> dict:
    """Reinterpret-view MX weights as the framework's carrier dtype.

    ``target`` selects the dispatch table in ``_MX_WEIGHT_VIEW_TARGETS``. The view
    aliases the underlying buffer; the kernel's ``convert_to_mxfp_dtype`` reverses
    it at entry so the same kernel works regardless of carrier.
    """
    table = _MX_WEIGHT_VIEW_TARGETS[target]
    carrier = table.get(weight_dtype)
    assert carrier is not None, f"target={target!r} unsupported for weight_dtype={weight_dtype}"
    if target == "unpacked_fp8":
        carrier = dt.finfo(carrier).dtype
    result = kernel_input.copy()
    for key in ('gate_up_proj_weight', 'down_proj_weight'):
        if key in result and result[key] is not None:
            result[key] = result[key].view(carrier)
    return result


def filter_moe_bwmm_mx_shard_block_combinations(
    vnc_degree=None,
    hidden=None,
    intermediate=None,
    block_size=None,
    expert_affinities_scaling_mode=None,
    skip_mode=None,
    **kwargs,
):
    """Filter invalid parameter combinations for MoE BWMM MX CTE kernel (shard-on-block).

    Checks constraints from kernel_assert statements in bwmm_shard_on_block_mx.py:
    - vnc_degree == 2 (num_shards)
    - block_size % 128 == 0
    - 512 <= hidden <= 8192 and hidden % 512 == 0
    - intermediate % 16 == 0 and MX tiling rules
    - expert_affinities_scaling_mode == POST_SCALE
    - skip_mode == 3 requires intermediate (I_TP) <= 512
    """
    if vnc_degree is not None and vnc_degree != 2:
        return FilterResult.INVALID
    if (
        expert_affinities_scaling_mode is not None
        and expert_affinities_scaling_mode != ExpertAffinityScaleMode.POST_SCALE
    ):
        return FilterResult.INVALID
    # STATIC_MX uses the dummy-127 per-block scale path; the packed per-block scale
    # layout is unused there, so this combo is rejected by a kernel_assert.
    if kwargs.get("use_packed_scales") and kwargs.get("quant_type") == QuantizationType.STATIC_MX:
        return FilterResult.INVALID
    if hidden is not None and (not (512 <= hidden <= 8192) or hidden % 512 != 0):
        return FilterResult.INVALID
    if block_size is not None and block_size % 128 != 0:
        return FilterResult.INVALID
    if intermediate is not None:
        if intermediate % 16 != 0:
            return FilterResult.INVALID
        # Kernel supports multiples of 512 (whole I-tiles), partial last I-tile with
        # alignment % (_q_width * _q_height) = 32, and small I with I % 32 == 0.
        if not (
            intermediate % 512 == 0
            or (intermediate > 512 and intermediate % 32 == 0)
            or (intermediate < 512 and intermediate % 32 == 0)
        ):
            return FilterResult.INVALID

    # TODO: Add a proper SBUF budget check in the kernel instead of hard-coding this combo.
    if hidden == 3584 and intermediate == 1536 and block_size == 512:
        return FilterResult.REDUNDANT

    return FilterResult.VALID


def filter_moe_bwmm_mx_shard_I_combinations(vnc_degree=None, **kwargs):
    """Filter invalid parameter combinations for MoE BWMM MX shard-on-I kernel.

    Only vnc_degree == 2 is enforced at runtime (kernel_assert in hybrid function).
    """
    if vnc_degree is not None and vnc_degree != 2:
        return FilterResult.INVALID
    return FilterResult.VALID


def _gen_block_inputs(
    vnc_degree: int,
    hidden: int,
    tokens: int,
    intermediate: int,
    expert: int,
    block_size: int,
    act_fn: ActFnType,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    dtype,
    weight_dtype,
    skip_mode: int,
    bias: bool,
    is_dynamic: bool,
    gate_clamp_upper: float | None,
    gate_clamp_lower: float | None,
    up_clamp_upper: float | None,
    up_clamp_lower: float | None,
    use_uint_weights: bool = False,
    unpacked_weights: bool = False,
    top_k: int | None = None,
    n_static_blocks: int | None = None,
    skewness_pct: float | None = None,
    global_top_k: int | None = None,
    ep_degree: int = 1,
    ep_rank: int = 0,
    use_packed_scales: bool = False,
    quantization_type: QuantizationType = QuantizationType.MX,
    use_prequant_hidden: bool = False,
    pack_affinities_into_hidden: bool = False,
    packed_affinities_dtype=nl.bfloat16,
) -> dict:
    assert skewness_pct is not None or top_k is not None, "Either skewness_pct or top_k must be provided"
    assert not (use_uint_weights and unpacked_weights), "use_uint_weights and unpacked_weights are mutually exclusive"
    if skewness_pct is not None:
        assert global_top_k is not None, "global_top_k is required when skewness_pct is provided"
        ki = build_moe_bwmm_mx_cte_from_model_test_config(
            H=hidden,
            T=tokens,
            E=expert,
            B=block_size,
            I_TP=intermediate,
            skewness_pct=skewness_pct,
            global_top_k=global_top_k,
            ep_degree=ep_degree,
            dtype=dtype,
            weight_dtype=weight_dtype,
            skip_mode=skip_mode,
            bias=bias,
            activation_function=act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            is_dynamic=is_dynamic,
            vnc_degree=vnc_degree,
            gate_clamp_upper_limit=gate_clamp_upper,
            gate_clamp_lower_limit=gate_clamp_lower,
            up_clamp_upper_limit=up_clamp_upper,
            up_clamp_lower_limit=up_clamp_lower,
            use_packed_scales=use_packed_scales,
            quantization_type=quantization_type,
        )
    else:
        assert top_k is not None, "top_k is required when skewness_pct is not provided"
        ki = build_moe_bwmm_mx_cte(
            H=hidden,
            T=tokens,
            E=expert,
            B=block_size,
            TOPK=top_k,
            I_TP=intermediate,
            dtype=dtype,
            weight_dtype=weight_dtype,
            skip_mode=skip_mode,
            bias=bias,
            activation_function=act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            is_dynamic=is_dynamic,
            vnc_degree=vnc_degree,
            n_dynamic_blocks=-1,
            gate_clamp_upper_limit=gate_clamp_upper,
            gate_clamp_lower_limit=gate_clamp_lower,
            up_clamp_upper_limit=up_clamp_upper,
            up_clamp_lower_limit=up_clamp_lower,
            n_static_blocks=n_static_blocks,
            use_packed_scales=use_packed_scales,
            quantization_type=quantization_type,
            use_prequant_hidden=use_prequant_hidden,
            pack_affinities_into_hidden=pack_affinities_into_hidden,
            packed_affinities_dtype=packed_affinities_dtype,
            ep_degree=ep_degree,
            ep_rank=ep_rank,
        )
    ordered = order_kernel_input(ki, variant='shard_on_block_mx')
    # Forward routing shape to the kernel for its best-case static-block estimate.
    ordered['top_k'] = global_top_k if skewness_pct is not None else top_k
    ordered['ep_degree'] = ep_degree
    if use_uint_weights:
        ordered = convert_mx_weights_view(ordered, weight_dtype, target="uint")
    elif unpacked_weights:
        ordered = convert_mx_weights_view(ordered, weight_dtype, target="unpacked_fp8")
    return ordered


def _mx_torch_ref_wrapper(torch_ref_func) -> Callable[..., Any]:
    """Custom torch_ref_wrapper that handles uint16 MX weights (NxD simulation).

    When ``use_packed_scales=True`` is in kwargs (forwarded by the test input
    dict), gather both scale tensors back to the standard 16-partition layout
    before invoking the reference. The reference is layout-agnostic of packing
    and never sees the kernel-only flag.
    """
    base_wrapper = torch_ref_wrapper(torch_ref_func)
    import functools

    @functools.wraps(torch_ref_func)
    def wrapped(**kwargs):
        # Pre-quantized hidden: the kernel receives a concat [T, H + scale_region (+ affinity tail)]
        # tensor, but the reference math wants fp32 [T, H]. Dequantize the concat back to fp32 so kernel
        # and golden compare the same numbers. The concat arrives as fp8 (unpacked affinities) or viewed
        # as the affinity dtype (bf16/fp32) when affinities are packed into the row; reinterpret to fp8
        # first so the width/offset math is in fp8 columns. Detect prequant from the fp8-width vs H+scale.
        hidden = kwargs.get('hidden_states')
        gup_w_for_h = kwargs.get('gate_up_proj_weight')
        if isinstance(hidden, np.ndarray) and gup_w_for_h is not None:
            # gate_up_proj_weight: (E, 128, 2, n_H512_tile, I) → H = 128 * n_H512_tile * 4
            _n_H512 = gup_w_for_h.shape[3]
            _H = gup_w_for_h.shape[1] * _n_H512 * 4
            _scale_region = n_packed_buffers_for(_n_H512) * 128  # packed scale region width
            _affin_dtype = hidden.dtype  # bf16/fp32 when packed; fp8 when unpacked
            hidden_fp8 = hidden.view(nl.float8_e4m3fn)  # no-op if already fp8; fp8-column view otherwise
            # Prequant detection. Two prequant layouts reach this wrapper:
            #   - unpacked: hidden arrives as an fp8 concat (affinities via the standalone tensor).
            #   - packed:   affinities are fused into the row, so expert_affinities_masked is None.
            # A plain online [T, H] bf16 input (real affinities tensor, non-fp8 dtype) must NEVER be
            # treated as prequant: its fp8-view width (2H) can exceed H+scale_region and would wrongly
            # trip a width-only check, corrupting the golden by dequantizing a non-prequant input.
            _is_prequant = 'float8' in str(hidden.dtype) or kwargs.get('expert_affinities_masked') is None
            if _is_prequant:
                # Reconstruct the standalone affinities the reference needs from the packed tail (the
                # kernel now gets expert_affinities_masked=None on the packed path). Tail starts at fp8
                # column H+scale_region, is E elements of _affin_dtype wide.
                if kwargs.get('expert_affinities_masked') is None:
                    _aff_off = _H + _scale_region
                    _E = gup_w_for_h.shape[0]
                    _row = hidden.view(_affin_dtype)  # [T, row_region // aff_as_fp8]
                    _aff_as_fp8 = _row.itemsize  # fp8 columns per affinity element (bf16 -> 2, fp32 -> 4)
                    # EP: read this rank's slot in the global tail, not the first E cols (the pre-fix bug).
                    # The kernel now takes ep_rank; the local expert base is ep_rank * E_local (== _E).
                    _ep_rank = int(np.asarray(kwargs.get('ep_rank', 0)).reshape(-1)[0])
                    _aff_col = _aff_off // _aff_as_fp8 + _ep_rank * _E
                    _aff = _row[:, _aff_col : _aff_col + _E].astype(np.float32)  # [T, E]
                    kwargs['expert_affinities_masked'] = _aff.reshape(-1, 1)
                kwargs['hidden_states'] = dequant_prequantized_hidden_concat(hidden_fp8, _H).astype(np.float32)

        # View framework-carrier weights (uint16/uint32 NxD or unpacked fp8 STATIC_MX)
        # back to the packed _x4 dtype before the base wrapper processes them.
        wdt = kwargs.get('weight_dtype')
        unpacked_fp8_dtypes = {
            np.dtype(dt.finfo(unp).dtype) for unp in _MX_WEIGHT_VIEW_TARGETS["unpacked_fp8"].values()
        }
        carrier_dtypes = {np.dtype(np.uint16), np.dtype(np.uint32)} | unpacked_fp8_dtypes
        for key in ('gate_up_proj_weight', 'down_proj_weight'):
            arr = kwargs.get(key)
            if isinstance(arr, np.ndarray) and arr.dtype in carrier_dtypes and wdt is not None:
                kwargs[key] = arr.view(np.dtype(wdt))
        # Strip the kernel-only flag and gather packed scales back to the standard
        # layout for the reference.
        if kwargs.pop('use_packed_scales', False):
            gup_w = kwargs.get('gate_up_proj_weight')
            assert gup_w is not None, "need gate_up_proj_weight to determine n_H512_tile"
            # gate_up_proj_weight shape: (E, _pmax, 2, n_H512_tile, I)
            n_H512_tile = gup_w.shape[3]
            kwargs['gate_up_proj_scale'] = gather_from_packed_gate_up(kwargs['gate_up_proj_scale'], n_H512_tile)
            dwn_w = kwargs.get('down_proj_weight')
            assert dwn_w is not None, "need down_proj_weight to determine n_I512_tile / p_scale"
            # down_proj_weight shape: (E, p_I, n_total_I512_tile, H);
            # standard scale shape: (E, p_I // _q_height, n_total_I512_tile, H)
            n_I512_tile = dwn_w.shape[2]
            p_I = dwn_w.shape[1]
            p_scale = p_I // 8  # _q_height
            kwargs['down_proj_scale'] = gather_from_packed_down(kwargs['down_proj_scale'], n_I512_tile, p_scale=p_scale)
        # Strip the kernel-only block-128 flag and expand the compact block-128
        # scales back to the native (coarse) layout the reference consumes — the
        # SAME values the kernel materializes, so kernel and ref agree exactly.
        if kwargs.pop('use_block128_scales', False):
            gup_w = kwargs.get('gate_up_proj_weight')
            assert gup_w is not None, "need gate_up_proj_weight to determine n_H512_tile / I"
            # gate_up_proj_weight shape: (E, _pmax, 2, n_H512_tile, I)
            n_H512_tile = gup_w.shape[3]
            I = gup_w.shape[4]
            kwargs['gate_up_proj_scale'] = block128_to_native_gate_up(kwargs['gate_up_proj_scale'], n_H512_tile, I)
            dwn_w = kwargs.get('down_proj_weight')
            assert dwn_w is not None, "need down_proj_weight to determine n_I512_tile / p_scale / H"
            # down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
            n_I512_tile = dwn_w.shape[2]
            p_I = dwn_w.shape[1]
            H = dwn_w.shape[3]
            p_scale = p_I // 8  # _q_height
            kwargs['down_proj_scale'] = block128_to_native_down(
                kwargs['down_proj_scale'], n_I512_tile, H, p_scale=p_scale
            )
        return base_wrapper(**kwargs)

    return wrapped


def _gen_I_inputs(
    vnc_degree: int,
    hidden: int,
    tokens: int,
    intermediate: int,
    expert: int,
    block_size: int,
    act_fn: ActFnType,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    dtype,
    weight_dtype,
    skip_mode: int,
    bias: bool,
    is_dynamic: bool,
    gate_clamp_upper: float | None,
    gate_clamp_lower: float | None,
    up_clamp_upper: float | None,
    up_clamp_lower: float | None,
    top_k: int | None = None,
    n_static_blocks: int | None = None,
    skewness_pct: float | None = None,
    global_top_k: int | None = None,
    ep_degree: int | None = None,
    use_block128_scales: bool = False,
) -> dict:
    assert skewness_pct is not None or top_k is not None, "Either skewness_pct or top_k must be provided"
    if skewness_pct is not None:
        assert global_top_k is not None and ep_degree is not None, (
            "global_top_k and ep_degree are required when skewness_pct is provided"
        )
        ki = build_moe_bwmm_mx_cte_from_model_test_config(
            H=hidden,
            T=tokens,
            E=expert,
            B=block_size,
            I_TP=intermediate,
            skewness_pct=skewness_pct,
            global_top_k=global_top_k,
            ep_degree=ep_degree,
            dtype=dtype,
            weight_dtype=weight_dtype,
            skip_mode=skip_mode,
            bias=bias,
            activation_function=act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            is_dynamic=is_dynamic,
            vnc_degree=vnc_degree,
            gate_clamp_upper_limit=gate_clamp_upper,
            gate_clamp_lower_limit=gate_clamp_lower,
            up_clamp_upper_limit=up_clamp_upper,
            up_clamp_lower_limit=up_clamp_lower,
            is_shard_on_I=True,
        )
    else:
        assert top_k is not None, "top_k is required when skewness_pct is not provided"
        ki = build_moe_bwmm_mx_cte(
            H=hidden,
            T=tokens,
            E=expert,
            B=block_size,
            TOPK=top_k,
            I_TP=intermediate,
            dtype=dtype,
            weight_dtype=weight_dtype,
            skip_mode=skip_mode,
            bias=bias,
            activation_function=act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            vnc_degree=vnc_degree,
            is_dynamic=is_dynamic,
            gate_clamp_upper_limit=gate_clamp_upper,
            gate_clamp_lower_limit=gate_clamp_lower,
            up_clamp_upper_limit=up_clamp_upper,
            up_clamp_lower_limit=up_clamp_lower,
            is_shard_on_I=True,
            n_static_blocks=n_static_blocks,
            use_block128_scales=use_block128_scales,
        )
    variant = 'shard_on_I_mx_hybrid' if is_dynamic else 'shard_on_I_mx'
    # Shard-on-I kernel doesn't accept STATIC_MX kwargs (no STATIC_MX path yet);
    # build_moe_bwmm_mx_cte always emits these, so strip them here.
    for k in ('quantization_type', 'gate_up_in_scale', 'down_in_scale'):
        ki.pop(k, None)
    return order_kernel_input(ki, variant=variant)


def _block_output(
    ki: dict, tokens: int, hidden: int, dtype, skip_mode: int, is_accumulating: bool, vnc_degree: int
) -> dict:
    dma_skip = map_skip_mode(skip_mode)
    out_T = tokens if dma_skip.skip_token else tokens + 1
    numpy_dtype = dt.finfo(dtype).dtype
    if is_accumulating:
        return {"output": np.zeros((vnc_degree, out_T, hidden), dtype=numpy_dtype)}
    return {"output": np.zeros((out_T, hidden), dtype=numpy_dtype)}


def _shard0_comparator(tokens, hidden, vnc_degree, dtype, rtol, atol):
    """Return a custom_comparator that compares only output[0, :tokens, :hidden].

    After reduce_outputs, output[0] has the correct reduced result but output[1]
    contains garbage (not zeroed); this ignores shard 1. The golden comes from the
    framework (custom_comparator receives it), so the torch-ref cache can serve it
    instead of the validator recomputing it.
    """
    from test.utils.comparators import maxAllClose

    numpy_dtype = dt.finfo(dtype).dtype

    def comparator(golden_dict, output_tensors):
        golden = golden_dict["output"]
        if hasattr(golden, "numpy"):
            golden = golden.numpy()
        golden = golden.astype(np.float32)

        class Shard0Validator(CustomValidator):
            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                actual = (
                    np.frombuffer(inference_output, dtype=numpy_dtype)
                    .reshape(vnc_degree, -1, hidden)
                    .astype(np.float32)
                )
                # Only compare shard 0, real tokens
                return maxAllClose(
                    actual[0, :tokens, :hidden],
                    golden[0, :tokens, :hidden],
                    rtol=rtol,
                    atol=atol,
                    verbose=1,
                    logfile=self.logfile,
                )

        return {
            "output": CustomValidatorWithOutputTensorData(
                validator=Shard0Validator,
                output_ndarray=output_tensors["output"],
            )
        }

    return comparator


def _I_output(ki: dict, tokens: int, hidden: int, dtype, skip_mode: int) -> dict:
    dma_skip = map_skip_mode(skip_mode)
    out_T = tokens if dma_skip.skip_token else tokens + 1
    return {"output": np.zeros((out_T, hidden), dtype=dtype)}


@pytest_test_metadata(name="MoE BWMM MX CTE", tags=["model"])
@pytest_marks(["moe", "cte", "mx"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMoeBwmmMxShardBlockKernel:
    """Test class for MoE BWMM MX CTE kernel (shard-on-block)."""

    @pytest_parametrize(SHARD_ON_BLOCK_PARAMS, SHARD_ON_BLOCK_UNIT_PERMS, abbrevs=_ABBREVS)
    def test_moe_bwmm_mx_shard_block_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        ep_degree: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
        use_uint_weights: bool,
        unpacked_weights: bool,
        n_static_blocks: int,
        use_packed_scales: bool,
        quant_type: QuantizationType,
    ):
        """Unit test for MoE BWMM MX CTE kernel with manual test vectors."""

        def input_gen(tc):
            return _gen_block_inputs(
                vnc_degree=vnc_degree,
                hidden=hidden,
                tokens=tokens,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                ep_degree=ep_degree,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                gate_clamp_upper=gate_clamp_upper,
                gate_clamp_lower=gate_clamp_lower,
                up_clamp_upper=up_clamp_upper,
                up_clamp_lower=up_clamp_lower,
                use_uint_weights=use_uint_weights,
                unpacked_weights=unpacked_weights,
                n_static_blocks=n_static_blocks,
                use_packed_scales=use_packed_scales,
                quantization_type=quant_type,
            )

        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )

        def out_desc(ki):
            return _block_output(ki, tokens, hidden, dtype, skip_mode, top_k != 1, vnc_degree)

        is_accumulating = top_k != 1
        custom_comparator = None
        if is_accumulating:
            custom_comparator = _shard0_comparator(tokens, hidden, vnc_degree, dtype, rtol=5e-2, atol=1e-5)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=bwmm_shard_on_block_mx,
            torch_ref=_mx_torch_ref_wrapper(bwmm_shard_on_block_mx_torch_ref),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=out_desc,
        ).run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-5,
            custom_comparator=custom_comparator,
        )

    # fmt: off
    # Pre-quantized fp8 hidden states (real MX): hidden_states arrives as a concatenated
    # [T, H+H/4] fp8 tensor (fp8 data | uint8 MX scales). Real MX weights only.
    # use_packed_scales packs the WEIGHT scales (orthogonal to the hidden-state path).
    # pack_affin (last field): when True the dense expert affinities are folded into the hidden concat
    # Fast configs are kept SMALL: the prequant path generates the fp8 concat plus a full fp32
    # reference hidden on top of the MX weight tensors, so host input-gen memory (gated at 3000 MB in
    # the suite) scales with E * H * I. Keep E/H/I modest here; the larger shapes live in the
    # full-only list below.
    _PREQUANT_HIDDEN_PERMS = [
        # vnc, hidden, tokens, inter, exp, blk, k, act, eas, dtype, wdt, sk, bias, dyn, gcu, gcl, ucu, ucl, pack, pack_affin, affin_dt
        [2, 512, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        # Odd E=17 -> E*2 not a multiple of 4: exercises the row pad-to-4 (bf16 tail).
        [2, 512, 1024, 384, 17, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        # skip_weight only (skip_mode=1): no affinity holes -> memset is a no-op path.
        [2, 512, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        # no skip (skip_mode=0).
        [2, 512, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        # FP32 packed affinities (NOAUX_TC layout: E*4 fp8 cols). skip both (dynamic) + no skip.
        [2, 512, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, True, True, nl.float32],
        [2, 512, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, True, True, nl.float32],
        # Odd E=17 with fp32 tail (E*4 is always a multiple of 4, so no row pad).
        [2, 512, 1024, 384, 17, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, True, True, nl.float32],
        # Unpacked-affinity prequant: hidden+scale fp8 concat, affinities passed as the standalone
        # tensor (expert_affinities_masked != None) -> exercises the non-packed fp8-hidden path.
        [2, 512, 1024, 384, 16, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, True, False, nl.bfloat16],
    ]

    # EP + packed affinities: global-width tail (E*ep_degree), local block_expert offset by
    # local_expert_start_idx = ep_rank*E. 
    _EP_BASE = [2, 3072, 8192, 1536, 64, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16]
    _PREQUANT_HIDDEN_EP_PERMS = [
        # (ep_degree, ep_rank): begin/middle/end shard within each degree.
        _EP_BASE + [2, 0],   # ep2, begin
        _EP_BASE + [2, 1],   # ep2, end
        _EP_BASE + [4, 0],   # ep4, begin
        _EP_BASE + [4, 2],   # ep4, middle
        _EP_BASE + [4, 3],   # ep4, end
        _EP_BASE + [16, 0],  # ep16, begin
        _EP_BASE + [16, 8],  # ep16, middle
        _EP_BASE + [16, 15],  # ep16, end
    ]

    # Full-suite prequant coverage: the MX shapes from _SHARD_BLOCK_FULL_ONLY run through the
    # pre-quantized fp8 hidden path with affinities packed into the row (pack_affin=True). STATIC_MX
    # rows are excluded (prequant fp8-hidden only supports QuantizationType.MX); the uint/unpacked
    # weight-carrier flags don't apply here so those rows become plain MX shapes. Each row keeps its
    # source use_packed_scales (the WEIGHT-scale packing, orthogonal to the hidden path).
    _PREQUANT_HIDDEN_FULL_PERMS = [
        # vnc, hidden, tokens, inter, exp, blk, k, act, eas, dtype, wdt, sk, bias, dyn, gcu, gcl, ucu, ucl, pack, pack_affin, affin_dt
        # MXFP4 (blk=256)
        [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False, True, nl.bfloat16],
        [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        [2, 3072, 8192, 1536, 64, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, True, True, nl.bfloat16],
        [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0, False, True, nl.bfloat16],
        [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False, True, nl.bfloat16],
        # FP32 packed affinities (NOAUX_TC layout) on a full-suite shape.
        [2, 3072, 10240, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, True, nl.float32],
        [2, 3072, 1024, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 3, True, True, 7.0, None, 7.0, -7.0, False, True, nl.bfloat16],
        # DeepSeek 3.2: EP32 TP2, H=7168, I_TP=1024, E_local=8, blk=256, top_k=8, MXFP8.
        [2, 7168, 1024, 1024, 8, 256, 8, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, False, True, None, None, None, None, True, True, nl.bfloat16],
        [2, 7168, 4096, 1024, 8, 256, 8, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 3, False, True, None, None, None, None, True, True, nl.bfloat16],
        [2, 7168, 8192, 1024, 8, 256, 8, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, False, True, None, None, None, None, True, True, nl.bfloat16],
    ]
    # fmt: on

    @pytest_parametrize(
        "vnc_degree, hidden, tokens, intermediate, expert, block_size, top_k, act_fn, "
        "expert_affinities_scaling_mode, dtype, weight_dtype, skip_mode, bias, is_dynamic, "
        "gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower, use_packed_scales, pack_affinities, "
        "packed_affinities_dtype, ep_degree, ep_rank",
        # Non-EP rows predate the ep columns -> default (ep_degree=1, ep_rank=0). EP rows carry both and
        [pytest.param(*c, 1, 0, marks=pytest.mark.fast) for c in _PREQUANT_HIDDEN_PERMS]
        + _PREQUANT_HIDDEN_EP_PERMS
        + [c + [1, 0] for c in _PREQUANT_HIDDEN_FULL_PERMS],
        abbrevs=_ABBREVS,
    )
    def test_moe_bwmm_mx_shard_block_prequant_hidden(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
        use_packed_scales: bool,
        pack_affinities: bool,
        packed_affinities_dtype: Any,
        ep_degree: int,
        ep_rank: int,
    ):
        """Unit test for the pre-quantized fp8 hidden-state (real MX) gate-up path."""

        def input_gen(tc):
            return _gen_block_inputs(
                vnc_degree=vnc_degree,
                hidden=hidden,
                tokens=tokens,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                gate_clamp_upper=gate_clamp_upper,
                gate_clamp_lower=gate_clamp_lower,
                up_clamp_upper=up_clamp_upper,
                up_clamp_lower=up_clamp_lower,
                use_packed_scales=use_packed_scales,
                quantization_type=QuantizationType.MX,
                use_prequant_hidden=True,
                pack_affinities_into_hidden=pack_affinities,
                packed_affinities_dtype=packed_affinities_dtype,
                ep_degree=ep_degree,
                ep_rank=ep_rank,
            )

        compiler_args = CompilerArgs(
            logical_nc_config=vnc_degree,
            platform_target=platform_target,
        )

        def out_desc(ki):
            return _block_output(ki, tokens, hidden, dtype, skip_mode, top_k != 1, vnc_degree)

        is_accumulating = top_k != 1
        custom_comparator = None
        if is_accumulating:
            custom_comparator = _shard0_comparator(tokens, hidden, vnc_degree, dtype, rtol=5e-2, atol=1e-5)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=bwmm_shard_on_block_mx,
            torch_ref=_mx_torch_ref_wrapper(bwmm_shard_on_block_mx_torch_ref),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=out_desc,
        ).run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-5,
            custom_comparator=custom_comparator,
        )

    @pytest.mark.coverage_parametrize(
        vnc_degree=BoundedRange([2], boundary_values=[]),
        # change back to 6144 upper bound after accuracy issue is resolved. NKI-1582
        hidden=BoundedRange([1536, 3072] + random.sample(range(1024, 4096, 512), 2), boundary_values=[]),
        tokens=BoundedRange([1024, 10240, 32768] + random.sample([2048, 4096, 8192], 2), boundary_values=[]),
        intermediate=BoundedRange([384, 512, 768, 1536], boundary_values=[100, 15]),
        expert=BoundedRange([8, 16, 32, 128], boundary_values=[]),
        block_size=BoundedRange([128, 256, 512], boundary_values=[64]),
        top_k=BoundedRange([2, 3, 4, 5], boundary_values=[1]),
        act_fn=BoundedRange([ActFnType.Swish], boundary_values=[]),
        expert_affinities_scaling_mode=BoundedRange([ExpertAffinityScaleMode.POST_SCALE], boundary_values=[]),
        dtype=BoundedRange([nl.bfloat16], boundary_values=[]),
        weight_dtype=BoundedRange([nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4], boundary_values=[]),
        skip_mode=BoundedRange([0, 1, 3], boundary_values=[]),
        bias=BoundedRange([True], boundary_values=[]),
        is_dynamic=BoundedRange([False, True], boundary_values=[]),
        use_packed_scales=BoundedRange([False, True], boundary_values=[]),
        quant_type=BoundedRange([QuantizationType.MX, QuantizationType.STATIC_MX], boundary_values=[]),
        filter=filter_moe_bwmm_mx_shard_block_combinations,
        coverage="pairs",
        abbrev=_ABBREVS,
    )
    def test_moe_bwmm_mx_shard_block_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        use_packed_scales: bool,
        quant_type: QuantizationType,
        is_negative_test_case: bool,
    ):
        """Sweep test for MoE BWMM MX CTE kernel using coverage_parametrize."""

        def input_gen(tc):
            return _gen_block_inputs(
                vnc_degree=vnc_degree,
                hidden=hidden,
                tokens=tokens,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                ep_degree=128 // expert,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                gate_clamp_upper=7.0,
                gate_clamp_lower=None,
                up_clamp_upper=7.0,
                up_clamp_lower=-7.0,
                use_packed_scales=use_packed_scales,
                quantization_type=quant_type,
            )

        def out_desc(ki):
            return _block_output(ki, tokens, hidden, dtype, skip_mode, top_k != 1, vnc_degree)

        is_accumulating = top_k != 1
        custom_comparator = None
        if is_accumulating and not is_negative_test_case:
            custom_comparator = _shard0_comparator(tokens, hidden, vnc_degree, dtype, rtol=5e-2, atol=1e-5)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=bwmm_shard_on_block_mx,
            torch_ref=_mx_torch_ref_wrapper(bwmm_shard_on_block_mx_torch_ref),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=out_desc,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=vnc_degree,
                platform_target=platform_target,
            ),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
            custom_comparator=custom_comparator,
        )


@pytest_marks(["moe", "cte", "mx", "shard_on_I"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMoeBwmmMxShardIKernel:
    """Test class for MoE BWMM MX kernel with intermediate dimension sharding."""

    @pytest.mark.parametrize(SHARD_ON_BLOCK_PARAMS, SHARD_ON_I_UNIT_PERMS)
    def test_moe_bwmm_mx_shard_I_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        ep_degree: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
        use_uint_weights: bool,
        unpacked_weights: bool,
        n_static_blocks: int,
        use_packed_scales: bool,
        quant_type: QuantizationType,
    ):
        """Unit test for MoE BWMM MX shard-on-I kernel with manual test vectors."""
        if quant_type != QuantizationType.MX:
            pytest.skip("Shard-on-I kernel only supports MX quantization (no STATIC_MX path yet).")
        if vnc_degree != 2:
            pytest.skip("Shard-on-I kernel requires exactly 2 shards.")
        if use_packed_scales:
            pytest.skip("Shard-on-I kernel does not yet support scale packing.")
        kf = blockwise_mm_shard_intermediate_mx_hybrid if is_dynamic else blockwise_mm_shard_intermediate_mx
        tr = (
            blockwise_mm_shard_intermediate_mx_hybrid_torch_ref
            if is_dynamic
            else blockwise_mm_shard_intermediate_mx_torch_ref
        )

        def input_gen(tc):
            return _gen_I_inputs(
                vnc_degree=vnc_degree,
                hidden=hidden,
                tokens=tokens,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                gate_clamp_upper=gate_clamp_upper,
                gate_clamp_lower=gate_clamp_lower,
                up_clamp_upper=up_clamp_upper,
                up_clamp_lower=up_clamp_lower,
                n_static_blocks=n_static_blocks,
            )

        def out_desc(ki):
            return _I_output(ki, tokens, hidden, dtype, skip_mode)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kf,
            torch_ref=_mx_torch_ref_wrapper(tr),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=out_desc,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
        )

    @pytest.mark.parametrize(SHARD_ON_I_BLOCK128_PARAMS, SHARD_ON_I_BLOCK128_PERMS)
    def test_moe_bwmm_mx_shard_I_block128_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        ep_degree: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
    ):
        """Unit test for shard-on-I kernel consuming DeepSeek block-128 weight scales."""
        if vnc_degree != 2:
            pytest.skip("Shard-on-I kernel requires exactly 2 shards.")
        kf = blockwise_mm_shard_intermediate_mx_hybrid if is_dynamic else blockwise_mm_shard_intermediate_mx
        tr = (
            blockwise_mm_shard_intermediate_mx_hybrid_torch_ref
            if is_dynamic
            else blockwise_mm_shard_intermediate_mx_torch_ref
        )

        def input_gen(tc):
            return _gen_I_inputs(
                vnc_degree=vnc_degree,
                hidden=hidden,
                tokens=tokens,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                gate_clamp_upper=gate_clamp_upper,
                gate_clamp_lower=gate_clamp_lower,
                up_clamp_upper=up_clamp_upper,
                up_clamp_lower=up_clamp_lower,
                use_block128_scales=True,
            )

        def out_desc(ki):
            return _I_output(ki, tokens, hidden, dtype, skip_mode)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kf,
            torch_ref=_mx_torch_ref_wrapper(tr),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=out_desc,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
        )

    @pytest.mark.coverage_parametrize(
        vnc_degree=BoundedRange([2], boundary_values=[]),
        hidden=BoundedRange([3072, 7168] + random.sample(range(1024, 8192, 512), 2), boundary_values=[]),
        tokens=BoundedRange([1024, 10240] + random.sample([2048, 4096, 8192], 2), boundary_values=[]),
        intermediate=BoundedRange([1024, 2048], boundary_values=[]),
        # disabling all 128 experts test for pipeline to flow KTK-102
        # expert=BoundedRange([8, 32, 128], boundary_values=[]),
        expert=BoundedRange([8, 32], boundary_values=[]),
        block_size=BoundedRange([256], boundary_values=[]),
        top_k=BoundedRange([1, 2, 4, 8], boundary_values=[]),
        act_fn=BoundedRange([ActFnType.Swish], boundary_values=[]),
        expert_affinities_scaling_mode=BoundedRange([ExpertAffinityScaleMode.POST_SCALE], boundary_values=[]),
        dtype=BoundedRange([nl.bfloat16], boundary_values=[]),
        weight_dtype=BoundedRange([nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4], boundary_values=[]),
        skip_mode=BoundedRange([0, 1], boundary_values=[]),
        bias=BoundedRange([True], boundary_values=[]),
        is_dynamic=BoundedRange([False, True], boundary_values=[]),
        filter=filter_moe_bwmm_mx_shard_I_combinations,
        coverage="pairs",
        abbrev=_ABBREVS,
    )
    def test_moe_bwmm_mx_shard_I_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        is_negative_test_case: bool,
    ):
        """Sweep test for MoE BWMM MX shard-on-I kernel using coverage_parametrize."""
        kf = blockwise_mm_shard_intermediate_mx_hybrid if is_dynamic else blockwise_mm_shard_intermediate_mx
        tr = (
            blockwise_mm_shard_intermediate_mx_hybrid_torch_ref
            if is_dynamic
            else blockwise_mm_shard_intermediate_mx_torch_ref
        )

        def input_gen(tc):
            return _gen_I_inputs(
                vnc_degree=vnc_degree,
                hidden=hidden,
                tokens=tokens,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                gate_clamp_upper=7.0,
                gate_clamp_lower=None,
                up_clamp_upper=7.0,
                up_clamp_lower=-7.0,
            )

        def out_desc(ki):
            return _I_output(ki, tokens, hidden, dtype, skip_mode)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kf,
            torch_ref=_mx_torch_ref_wrapper(tr),
            kernel_input_generator=input_gen,
            output_tensor_descriptor=out_desc,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )


@pytest_marks(["moe", "cte", "mx", "model"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMoeBwmmMxCteModel:
    """Model-driven tests for MoE BWMM MX CTE kernels, organized by tier."""

    _MODEL_PARAMS = MOE_BWMM_MX_CTE_MODEL_PARAMS

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize({ModelTestType.OPTIMAL: moe_bwmm_mx_cte_model_configs.get(ModelTestType.OPTIMAL, [])})
        if moe_bwmm_mx_cte_model_configs
        else ([], [])
    )

    _GENERALITY_PARAMS, _GENERALITY_IDS = (
        prepare_model_parametrize(
            {ModelTestType.GENERALITY: moe_bwmm_mx_cte_model_configs.get(ModelTestType.GENERALITY, [])}
        )
        if moe_bwmm_mx_cte_model_configs
        else ([], [])
    )

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        variant: str,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
        use_uint_weights: bool,
        skewness_pct: float,
        global_top_k: int,
        ep_degree: int,
    ):
        """Common model test logic dispatching to shard-on-block or shard-on-I."""
        if not platform_target.is_trn3():
            pytest.skip("MX quantization is only supported on TRN3.")

        if variant == "shard_on_I":
            kf = blockwise_mm_shard_intermediate_mx_hybrid if is_dynamic else blockwise_mm_shard_intermediate_mx
            tr = (
                blockwise_mm_shard_intermediate_mx_hybrid_torch_ref
                if is_dynamic
                else blockwise_mm_shard_intermediate_mx_torch_ref
            )

            def input_gen(tc):
                return _gen_I_inputs(
                    vnc_degree=vnc_degree,
                    hidden=hidden,
                    tokens=tokens,
                    intermediate=intermediate,
                    expert=expert,
                    block_size=block_size,
                    act_fn=act_fn,
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    dtype=dtype,
                    weight_dtype=weight_dtype,
                    skip_mode=skip_mode,
                    bias=bias,
                    is_dynamic=is_dynamic,
                    gate_clamp_upper=gate_clamp_upper,
                    gate_clamp_lower=gate_clamp_lower,
                    up_clamp_upper=up_clamp_upper,
                    up_clamp_lower=up_clamp_lower,
                    skewness_pct=skewness_pct,
                    global_top_k=global_top_k,
                    ep_degree=ep_degree,
                )

            def out_desc(ki):
                return _I_output(ki, tokens, hidden, dtype, skip_mode)

            UnitTestFramework(
                test_manager=test_manager,
                kernel_entry=kf,
                torch_ref=_mx_torch_ref_wrapper(tr),
                kernel_input_generator=input_gen,
                output_tensor_descriptor=out_desc,
            ).run_test(
                test_config=None,
                compiler_args=CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target),
                rtol=5e-2,
                atol=1e-5,
            )
        else:

            def input_gen(tc):
                return _gen_block_inputs(
                    vnc_degree=vnc_degree,
                    hidden=hidden,
                    tokens=tokens,
                    intermediate=intermediate,
                    expert=expert,
                    block_size=block_size,
                    act_fn=act_fn,
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    dtype=dtype,
                    weight_dtype=weight_dtype,
                    skip_mode=skip_mode,
                    bias=bias,
                    is_dynamic=is_dynamic,
                    gate_clamp_upper=gate_clamp_upper,
                    gate_clamp_lower=gate_clamp_lower,
                    up_clamp_upper=up_clamp_upper,
                    up_clamp_lower=up_clamp_lower,
                    use_uint_weights=use_uint_weights,
                    skewness_pct=skewness_pct,
                    global_top_k=global_top_k,
                    ep_degree=ep_degree,
                )

            def out_desc(ki):
                return _block_output(ki, tokens, hidden, dtype, skip_mode, min(expert, global_top_k) > 1, vnc_degree)

            is_accumulating = min(expert, global_top_k) > 1
            custom_comparator = None
            if is_accumulating:
                custom_comparator = _shard0_comparator(tokens, hidden, vnc_degree, dtype, rtol=5e-2, atol=1e-5)

            UnitTestFramework(
                test_manager=test_manager,
                kernel_entry=bwmm_shard_on_block_mx,
                torch_ref=_mx_torch_ref_wrapper(bwmm_shard_on_block_mx_torch_ref),
                kernel_input_generator=input_gen,
                output_tensor_descriptor=out_desc,
            ).run_test(
                test_config=None,
                compiler_args=CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target),
                rtol=5e-2,
                atol=1e-5,
                custom_comparator=custom_comparator,
            )

    @pytest.mark.optimal
    @pytest.mark.parametrize(_MODEL_PARAMS, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        variant: str,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
        use_uint_weights: bool,
        skewness_pct: float,
        global_top_k: int,
        ep_degree: int,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        self._run_model_test(**{k: v for k, v in locals().items() if k != "self"})

    @pytest.mark.generality
    @pytest.mark.parametrize(_MODEL_PARAMS, _GENERALITY_PARAMS, ids=_GENERALITY_IDS)
    def test_generality(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        variant: str,
        vnc_degree: int,
        hidden: int,
        tokens: int,
        intermediate: int,
        expert: int,
        block_size: int,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        dtype: Any,
        weight_dtype: Any,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float | None,
        gate_clamp_lower: float | None,
        up_clamp_upper: float | None,
        up_clamp_lower: float | None,
        use_uint_weights: bool,
        skewness_pct: float,
        global_top_k: int,
        ep_degree: int,
    ):
        """GENERALITY: Broader model coverage configs."""
        self._run_model_test(**{k: v for k, v in locals().items() if k != "self"})
