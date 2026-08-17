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

"""Integration tests for MoE CTE blockwise matrix multiplication kernels using UnitTestFramework."""

from typing import final

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    BWMMFunc,
    generate_moe_cte_inputs,
    moe_cte_kernel_wrapper,
    moe_cte_output_tensors,
    moe_cte_torch_wrapper,
)

try:
    from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_model_config import (
        moe_cte_model_configs,
    )
except ImportError:
    moe_cte_model_configs = {}
from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
# Parameter names for pytest.mark.parametrize
BWMM_LNC2_PARAM_NAMES = \
    "bwmm_func,                            hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, bias,  training, quantize, act_fn,          expert_affinities_scaling_mode,     gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, block_sharding_strategy, is_block_quant, is_per_tensor"

# All test cases
BWMM_LNC2_TEST_CASES = [
    # HEAD-only: additional intermediate sizes
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    4864,   1024,   8,      512,        4,     1216,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    4864,   1024,   8,      512,        4,     1216,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2880,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2880,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     720,          nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     720,          nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    # multiply_on_I with bias test
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish,  ExpertAffinityScaleMode.PRE_SCALE, None,          None,          None,        None,        True, None, False, False),
    # SquaredReLU activation
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       1024,   512,    2,      256,        2,     512,          nl.bfloat16, 0,   True,  False,    None,            ActFnType.SquaredReLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       1024,   512,    2,      256,        2,     512,          nl.bfloat16, 0,   True,  False,    None,            ActFnType.SquaredReLU,  ExpertAffinityScaleMode.PRE_SCALE, None,          None,          None,        None,        True, None, False, False),
    # Incoming branch cases
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    False, False,    None,     ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    True,  False,    None,     ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          7,             None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          7,           None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        7,           False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          7,             None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          7,           None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        7,           False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False, None, False, False),
    # Qwen3-235B-A22B FP8 scenario: hits fused gate+up FP8 path in compute_gate_and_up_projections_shard_on_intermediate
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    4096,   2048,   2,      512,        8,     1536,         nl.bfloat16, 0,   False, False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    # Original SHARD_ON_BLOCK kernel tests
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     1536,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None, False, False],
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     1536,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None, False, False],
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     3072,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None, False, False],
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     3072,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None, False, False],
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   10240,  128,    256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    # E < TOPK test vectors
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       4096,   10240,  4,      512,        8,     768,          nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    4096,   10240,  4,      512,        8,     768,          nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536, 8192,  2,      4096,       2,     6144,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 2048, 2048,  2,      1024,       2,     8192,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 4096, 4096,  2,      2048,       8,     1536,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536, 8192,  2,      4096,       2,     6144,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 2048, 2048,  2,      1024,       2,     8192,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 4096, 4096,  2,      2048,       8,     1536,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, True, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     1536,         nl.bfloat16, 0,   False, False,    nl.float8_e4m3,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, True, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   False, False,    nl.float8_e4m3,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, True),
]

# Test cases with expert=64
BWMM_LNC2_SLOW_CASES = [pytest.param(*case, marks=pytest.mark.slow_simulation) for case in [
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False, None, False, False),
]]
# fmt: on


# (hidden, tokens, expert, block_size, top_k, intermediate) keys for fast tests.
_FAST_LNC2_KEYS = frozenset(
    {
        (3072, 1024, 8, 512, 4, 192),  # SHARD_ON_BLOCK skip=3 / skip=1
        (3072, 1024, 8, 512, 4, 720),  # SHARD_ON_INTERMEDIATE_HW
        (2048, 2048, 2, 1024, 2, 8192),  # SHARD_ON_INTERMEDIATE_DROPPING
        (3072, 1024, 8, 512, 4, 768),  # SHARD_ON_INTERMEDIATE_HW (non-clamped + clamped variants)
        (3072, 1024, 8, 512, 4, 1536),  # SHARD_ON_BLOCK
        (3072, 1024, 8, 512, 4, 2048),  # SHARD_ON_INTERMEDIATE (skip=0 and skip=1)
        (4096, 4096, 2, 2048, 8, 1536),  # SHARD_ON_INTERMEDIATE_DROPPING
        (4864, 1024, 8, 512, 4, 1216),  # SHARD_ON_INTERMEDIATE_HW
    }
)


@pytest_test_metadata(name="MoE BWMM BF16 CTE", tags=["model"])
@pytest_marks(["moe", "blockwise_mm", "lnc2"])
@final
class TestMoeBlockwiseMatMulLnc2:
    """Tests for LNC2 blockwise matmul, across different sharding axis (Batch, Hidden, Intermediate).

    skip modes:
    - 0: SkipMode(False, False)
    - 1: SkipMode(True, False)  - skip token
    - 2: SkipMode(False, True)  - skip weight
    - 3: SkipMode(True, True)   - skip both
    """

    ALL_PARAMS = [
        pytest.param(*c, marks=pytest.mark.fast) if tuple(c[1:7]) in _FAST_LNC2_KEYS else pytest.param(*c)
        for c in BWMM_LNC2_TEST_CASES
    ] + BWMM_LNC2_SLOW_CASES

    @pytest.mark.parametrize(BWMM_LNC2_PARAM_NAMES, ALL_PARAMS)
    def test_moe_blockwise_mm_kernel_lnc2(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bwmm_func: BWMMFunc,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        dtype,
        skip: int,
        bias: bool,
        training: bool,
        quantize,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        gate_cl_upper,
        gate_cl_lower,
        up_cl_upper,
        up_cl_lower,
        expert_affinity_multiply_on_I: bool,
        block_sharding_strategy: str,
        is_block_quant: bool,
        is_per_tensor: bool,
        platform_target: Platforms,
        accumulation_dtype=None,
    ):
        lnc_degree = 2

        def input_generator(test_config):
            return generate_moe_cte_inputs(
                bwmm_func_enum=bwmm_func,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                skip=skip,
                bias=bias,
                training=training,
                quantize=quantize,
                activation_function=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                gate_clamp_upper=gate_cl_upper,
                gate_clamp_lower=gate_cl_lower,
                up_clamp_upper=up_cl_upper,
                up_clamp_lower=up_cl_lower,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
                block_sharding_strategy=block_sharding_strategy,
                is_block_quant=is_block_quant,
                is_per_tensor=is_per_tensor,
                accumulation_dtype=accumulation_dtype,
            )

        is_dropping = bwmm_func == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING

        def output_tensors(kernel_input):
            return moe_cte_output_tensors(
                kernel_input=kernel_input,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                bwmm_func_enum=bwmm_func,
                training=training or is_dropping,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
                accumulation_dtype=accumulation_dtype,
            )

        rtol, atol = (5e-2, 1e-5) if quantize else (2e-2, 1e-5)
        compiler_args = CompilerArgs(
            logical_nc_config=lnc_degree,
            platform_target=platform_target,
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_cte_kernel_wrapper,
            torch_ref=torch_ref_wrapper(moe_cte_torch_wrapper),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            check_unused_params=True,
            collector=collector,
        )

        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=rtol,
            atol=atol,
        )

    FP32_ACCUM_CASES = [
        pytest.param(*c, marks=pytest.mark.fast)
        for c in [
            (
                BWMMFunc.SHARD_ON_INTERMEDIATE,
                3072,
                1024,
                8,
                512,
                4,
                2048,
                nl.bfloat16,
                0,
                False,
                True,
                None,
                ActFnType.Swish,
                ExpertAffinityScaleMode.POST_SCALE,
                None,
                None,
                None,
                None,
                False,
                None,
                False,
                False,
            ),
            (
                BWMMFunc.SHARD_ON_INTERMEDIATE,
                3072,
                1024,
                8,
                512,
                4,
                2048,
                nl.bfloat16,
                0,
                False,
                True,
                None,
                ActFnType.Swish,
                ExpertAffinityScaleMode.POST_SCALE,
                None,
                None,
                None,
                None,
                True,
                None,
                False,
                False,
            ),
            (
                BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING,
                2048,
                2048,
                2,
                1024,
                2,
                8192,
                nl.bfloat16,
                0,
                False,
                True,
                None,
                ActFnType.SiLU,
                ExpertAffinityScaleMode.POST_SCALE,
                None,
                None,
                None,
                None,
                False,
                None,
                False,
                False,
            ),
        ]
    ]

    @pytest.mark.parametrize(BWMM_LNC2_PARAM_NAMES, FP32_ACCUM_CASES)
    def test_moe_blockwise_mm_kernel_lnc2_fp32_accum(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        bwmm_func: BWMMFunc,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        dtype,
        skip: int,
        bias: bool,
        training: bool,
        quantize,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        gate_cl_upper,
        gate_cl_lower,
        up_cl_upper,
        up_cl_lower,
        expert_affinity_multiply_on_I: bool,
        block_sharding_strategy: str,
        is_block_quant: bool,
        is_per_tensor: bool,
        platform_target: Platforms,
    ):
        self.test_moe_blockwise_mm_kernel_lnc2(
            test_manager,
            collector,
            bwmm_func,
            hidden,
            tokens,
            expert,
            block_size,
            top_k,
            intermediate,
            dtype,
            skip,
            bias,
            training,
            quantize,
            act_fn,
            expert_affinities_scaling_mode,
            gate_cl_upper,
            gate_cl_lower,
            up_cl_upper,
            up_cl_lower,
            expert_affinity_multiply_on_I,
            block_sharding_strategy,
            is_block_quant,
            is_per_tensor,
            platform_target,
            accumulation_dtype=nl.float32,
        )


# fmt: off
MOE_CTE_MODEL_PARAMS = (
    "bwmm_func, hidden, tokens, expert, block_size, intermediate, dtype, skip, bias, "
    "training, quantize, act_fn, expert_affinities_scaling_mode, gate_cl_upper, gate_cl_lower, "
    "up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, skewness_pct, global_top_k, ep_degree"
)
# fmt: on


@pytest_marks(["moe", "cte", "model", "mx"])
@final
class TestMoeCteModel:
    """Model-driven tests for MoE CTE kernels, organized by tier."""

    _MODEL_PARAMS = MOE_CTE_MODEL_PARAMS
    _MODEL_PARAMS = MOE_CTE_MODEL_PARAMS

    _GENERALITY_PARAMS, _GENERALITY_IDS = (
        prepare_model_parametrize({ModelTestType.GENERALITY: moe_cte_model_configs.get(ModelTestType.GENERALITY, [])})
        if moe_cte_model_configs
        else ([], [])
    )

    def _run_model_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bwmm_func: BWMMFunc,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        intermediate: int,
        dtype,
        skip: int,
        bias: bool,
        training: bool,
        quantize,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        gate_cl_upper,
        gate_cl_lower,
        up_cl_upper,
        up_cl_lower,
        expert_affinity_multiply_on_I: bool,
        skewness_pct: float,
        global_top_k: int,
        ep_degree: int,
    ):
        lnc_degree = 2

        def input_generator(test_config):
            return generate_moe_cte_inputs(
                bwmm_func_enum=bwmm_func,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=global_top_k,
                dtype=dtype,
                skip=skip,
                bias=bias,
                training=training,
                quantize=quantize,
                activation_function=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                gate_clamp_upper=gate_cl_upper,
                gate_clamp_lower=gate_cl_lower,
                up_clamp_upper=up_cl_upper,
                up_clamp_lower=up_cl_lower,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
            )

        def output_tensors(kernel_input):
            return moe_cte_output_tensors(
                kernel_input=kernel_input,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=global_top_k,
                dtype=dtype,
                bwmm_func_enum=bwmm_func,
                training=training,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
            )

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_cte_kernel_wrapper,
            torch_ref=torch_ref_wrapper(moe_cte_torch_wrapper),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            check_unused_params=True,
            collector=collector,
        ).run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2,
            atol=1e-5,
            metadata={
                "config_name": "test_moe_cte",
                "key": {
                    "fn": bwmm_func,
                    "hid": hidden,
                    "tok": tokens,
                    "exp": expert,
                    "bs": block_size,
                    "int": intermediate,
                },
            },
        )

    @pytest.mark.generality
    @pytest.mark.parametrize(_MODEL_PARAMS, _GENERALITY_PARAMS, ids=_GENERALITY_IDS)
    def test_generality(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        bwmm_func: BWMMFunc,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        intermediate: int,
        dtype,
        skip: int,
        bias: bool,
        training: bool,
        quantize,
        act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        gate_cl_upper,
        gate_cl_lower,
        up_cl_upper,
        up_cl_lower,
        expert_affinity_multiply_on_I: bool,
        skewness_pct: float,
        global_top_k: int,
        ep_degree: int,
    ):
        """GENERALITY: Broader model coverage configs."""
        self._run_model_test(**{k: v for k, v in locals().items() if k != "self"})


SKIP_GATE_PROJ_PARAMS = [
    pytest.param(
        ActFnType.SquaredReLU,
        ExpertAffinityScaleMode.POST_SCALE,
        512,
        1024,
        512,
        2,
        256,
        2,
        id="SquaredReLU-POST_SCALE",
    ),
    pytest.param(
        ActFnType.SquaredReLU,
        ExpertAffinityScaleMode.PRE_SCALE,
        512,
        1024,
        512,
        2,
        256,
        2,
        id="SquaredReLU-PRE_SCALE",
    ),
    pytest.param(
        ActFnType.SiLU,
        ExpertAffinityScaleMode.POST_SCALE,
        512,
        1024,
        512,
        2,
        256,
        2,
        id="SiLU-POST_SCALE",
    ),
    pytest.param(
        ActFnType.SquaredReLU,
        ExpertAffinityScaleMode.POST_SCALE,
        2048,
        2048,
        1024,
        64,
        256,
        8,
        id="SquaredReLU-POST_SCALE-e64-k8",
    ),
]


class TestMoeBwmmSkipGateProj:
    """Test skip_gate_proj=True for non-gated MLP in BWMM shard-on-I."""

    @pytest.mark.parametrize(
        "act_fn, scaling_mode, tokens, hidden, intermediate, expert, block_size, top_k",
        SKIP_GATE_PROJ_PARAMS,
    )
    def test_bwmm_shard_I_skip_gate_proj(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        act_fn: ActFnType,
        scaling_mode: ExpertAffinityScaleMode,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        platform_target: Platforms,
    ):
        def input_generator(test_config):
            return generate_moe_cte_inputs(
                bwmm_func_enum=BWMMFunc.SHARD_ON_INTERMEDIATE,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=nl.bfloat16,
                skip=0,
                bias=False,
                training=False,
                quantize=None,
                activation_function=act_fn,
                expert_affinities_scaling_mode=scaling_mode,
                expert_affinity_multiply_on_I=True,
                skip_gate_proj=True,
            )

        def output_tensors(kernel_input):
            return moe_cte_output_tensors(
                kernel_input=kernel_input,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=nl.bfloat16,
                bwmm_func_enum=BWMMFunc.SHARD_ON_INTERMEDIATE,
                training=False,
                expert_affinity_multiply_on_I=True,
                lnc_degree=2,
            )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_cte_kernel_wrapper,
            torch_ref=torch_ref_wrapper(moe_cte_torch_wrapper),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            check_unused_params=True,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )
