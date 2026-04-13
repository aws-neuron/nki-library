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

from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    BWMMFunc,
    generate_moe_cte_inputs,
    moe_cte_kernel_wrapper,
    moe_cte_output_tensors,
    moe_cte_torch_wrapper,
)
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_model_config import (
    moe_cte_model_configs,
)
from test.utils.common_dataclasses import MODEL_TEST_TYPE, CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

# fmt: off
# Parameter names for pytest.mark.parametrize
BWMM_LNC2_PARAM_NAMES = \
    "bwmm_func,                            hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, bias,  training, quantize, act_fn,          expert_affinities_scaling_mode,     gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I"

# All test cases
BWMM_LNC2_TEST_CASES = [
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    False, False,    None,     ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    True,  False,    None,     ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          7,             None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          7,           None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        7,           False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          7,             None,        None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          7,           None,        False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        7,           False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    # (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     1536,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     1536,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     3072,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
    [BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     3072,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
    (BWMMFunc.SHARD_ON_BLOCK,              3072,   10240,  128,    256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   False, True,     None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    # E < TOPK test vectors
    (BWMMFunc.SHARD_ON_INTERMEDIATE,       4096,   10240,  4,      512,        8,     768,          nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    4096,   10240,  4,      512,        8,     768,          nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536, 8192,  2,      4096,       2,     6144,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 2048, 2048,  2,      1024,       2,     8192,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 4096, 4096,  2,      2048,       8,     1536,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536, 8192,  2,      4096,       2,     6144,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 2048, 2048,  2,      1024,       2,     8192,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
    (BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 4096, 4096,  2,      2048,       8,     1536,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False),
]
# fmt: on


@pytest_test_metadata(
    name="MoE Blockwise MatMul LNC2",
    pytest_marks=["moe", "blockwise_mm", "lnc2"],
)
@final
class TestMoeBlockwiseMatMulLnc2:
    """Tests for LNC2 blockwise matmul, across different sharding axis (Batch, Hidden, Intermediate).

    skip modes:
    - 0: SkipMode(False, False)
    - 1: SkipMode(True, False)  - skip token
    - 2: SkipMode(False, True)  - skip weight
    - 3: SkipMode(True, True)   - skip both
    """

    ALL_PARAMS = BWMM_LNC2_TEST_CASES + moe_cte_model_configs

    ALL_PARAM_IDS = [None] * len(BWMM_LNC2_TEST_CASES) + [
        f"{MODEL_TEST_TYPE}_" + "-".join(str(p.value) if hasattr(p, 'value') else str(p) for p in params)
        for params in moe_cte_model_configs
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize(BWMM_LNC2_PARAM_NAMES, ALL_PARAMS, ids=ALL_PARAM_IDS)
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
        platform_target: Platforms,
        request,
    ):
        lnc_degree = 2

        is_model_config = MODEL_TEST_TYPE in request.node.callspec.id
        metadata_key = (
            {
                "fn": bwmm_func,
                "hid": hidden,
                "tok": tokens,
                "exp": expert,
                "bs": block_size,
                "k": top_k,
                "int": intermediate,
            }
            if is_model_config
            else None
        )

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
            )

        rtol, atol = (5e-2, 1e-5) if quantize else (2e-2, 1e-5)

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=platform_target)

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
            metadata={
                "config_name": "test_moe_cte",
                "key": metadata_key,
            }
            if is_model_config
            else None,
        )
