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

import math
from test.integration.nkilib.core.mlp.test_mlp_common import gen_moe_mx_weights
from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import (
    _get_clamp_limits,
    _pmax,
    _q_height,
    _q_width,
    golden_all_expert_moe_tkg,
    golden_selective_expert_moe_tkg,
    is_dtype_low_precision,
    is_dtype_mx,
)
from test.integration.nkilib.core.moe_block.test_moe_block_tkg_model_config import (
    moe_block_tkg_model_configs,
)
from test.integration.nkilib.core.router_topk.router_topk_torch import router_topk_torch_ref
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.test_kernel_common import rms_norm as golden_rms_norm
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorRangeConfig,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import Any, Callable, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe_block.moe_block_tkg import moe_block_tkg as moe_block_tkg_kernel
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType

# Dimension names for test config (short string values to avoid long test names)
MOE_BLOCK_TKG_CONFIG = "moe_block_tkg_config"
LNC_DIM_NAME = "ln"
IS_ALL_EXPERT_DIM_NAME = "ae"
BATCH_DIM_NAME = "ba"
SEQLEN_DIM_NAME = "sq"
HIDDEN_DIM_NAME = "hi"
HIDDEN_ACTUAL_DIM_NAME = "ha"
INTERMEDIATE_DIM_NAME = "im"
NUM_GLOBAL_EXPERTS_DIM_NAME = "ge"
NUM_LOCAL_EXPERTS_DIM_NAME = "le"
TOP_K_DIM_NAME = "tk"
ROUTER_FN_DIM_NAME = "rf"
HIDDEN_ACT_FN_DIM_NAME = "af"
EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME = "sm"
MOE_WEIGHT_DTYPE_DIM_NAME = "wd"
INPUT_DTYPE_DIM_NAME = "id"
HAS_SHARED_EXPERT_DIM_NAME = "se"
HAS_BIAS_DIM_NAME = "bi"
HAS_CLAMP_DIM_NAME = "cl"
ROUTER_ACT_FIRST_DIM_NAME = "ra"
NORM_TOPK_PROB_DIM_NAME = "np"
SKIP_ROUTER_LOGITS_DIM_NAME = "sr"
ROUTER_MM_DTYPE_DIM_NAME = "rd"


def create_moe_block_tkg_test_config(test_vectors, test_type: str = "manual"):
    """
    Utility function to create complete RangeTestConfig from list of MoE Block TKG config vectors.
    """
    test_cases = []
    for test_params in test_vectors:
        (
            lnc,
            is_all_expert,
            batch,
            seqlen,
            hidden,
            hidden_actual,
            intermediate,
            num_global_experts,
            num_local_experts,
            top_k,
            router_fn,
            hidden_act_fn,
            expert_affinities_scaling_mode,
            moe_weight_dtype,
            input_dtype,
            has_shared_expert,
            has_bias,
            has_clamp,
            router_act_first,
            norm_topk_prob,
            skip_router_logits,
            router_mm_dtype,
        ) = test_params

        test_case = {
            MOE_BLOCK_TKG_CONFIG: {
                LNC_DIM_NAME: lnc,
                IS_ALL_EXPERT_DIM_NAME: int(is_all_expert),
                BATCH_DIM_NAME: batch,
                SEQLEN_DIM_NAME: seqlen,
                HIDDEN_DIM_NAME: hidden,
                HIDDEN_ACTUAL_DIM_NAME: hidden_actual,
                INTERMEDIATE_DIM_NAME: intermediate,
                NUM_GLOBAL_EXPERTS_DIM_NAME: num_global_experts,
                NUM_LOCAL_EXPERTS_DIM_NAME: num_local_experts,
                TOP_K_DIM_NAME: top_k,
                ROUTER_FN_DIM_NAME: router_fn,
                HIDDEN_ACT_FN_DIM_NAME: hidden_act_fn,
                EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode.value,
                MOE_WEIGHT_DTYPE_DIM_NAME: moe_weight_dtype,
                INPUT_DTYPE_DIM_NAME: input_dtype,
                HAS_SHARED_EXPERT_DIM_NAME: int(has_shared_expert),
                HAS_BIAS_DIM_NAME: int(has_bias),
                HAS_CLAMP_DIM_NAME: int(has_clamp),
                ROUTER_ACT_FIRST_DIM_NAME: int(router_act_first),
                NORM_TOPK_PROB_DIM_NAME: int(norm_topk_prob),
                SKIP_ROUTER_LOGITS_DIM_NAME: int(skip_router_logits),
                ROUTER_MM_DTYPE_DIM_NAME: router_mm_dtype,
            }
        }
        test_cases.append(test_case)

    generators = [RangeManualGeneratorStrategy(test_cases=test_cases, test_type=test_type)]

    return RangeTestConfig(
        additional_params={},
        global_tensor_configs=TensorRangeConfig(
            tensor_configs={},
            monotonic_step_size=1,
            custom_generators=generators,
        ),
    )


# fmt: off
moe_block_tkg_manual_params = [
    # lnc, is_all_expert,   batch,  seqlen,     hidden,     hidden_actual,  intermediate,   num_global_experts, num_local_experts, top_k,       router_fn,                  hidden_act_fn,      expert_affinities_scaling_mode,         moe_weight_dtype,       input_dtype,    has_shared_expert,  has_bias,   has_clamp,  router_act_first,   norm_topk_prob,     skip_router_logits,     router_mm_dtype
    # Selective-load tests (num_global_experts == num_local_experts)
    # GPT-OSS 120B
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      4,          3072,       2880,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      1,          3072,       None,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      4,          3072,       2880,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      4,          3072,       2880,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      1,          3072,       None,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      4,          3072,       2880,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      5,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          4,      1,          3072,       2880,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          8,      1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          16,     1,          3072,       2880,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          19,     1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          32,     1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    # Qwen3 235B
    [2,     False,          1,      1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     False,          1,      1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    # All-expert BF16 tests (num_local_experts can be smaller than num_global_experts)
    # GPT OSS 120B
    [2,     True,           19,     1,          3072,       None,           128,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           128,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           384,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           768,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           768,            128,                4,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           1536,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           1536,           128,                2,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     1,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    # Qwen3 235B
    [2,     True,           16,     1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     True,           16,     1,          4096,       None,           384,            128,                8,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    # All-expert MXFP4 tests (T must be divisible by 4)
    # GPT-OSS 120B
    [2,     True,           32,     4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           64,     5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           128,    5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           128,    4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           256,    3,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           512,    2,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
]
# fmt: on


def moe_block_tkg_combined_config():
    """Create combined config with manual and model test cases."""
    manual_config = create_moe_block_tkg_test_config(moe_block_tkg_manual_params, test_type="manual")
    model_config = create_moe_block_tkg_test_config(moe_block_tkg_model_configs, test_type=MODEL_TEST_TYPE)
    # Combine by extending generators
    manual_config.global_tensor_configs.custom_generators.extend(model_config.global_tensor_configs.custom_generators)
    return manual_config


def build_moe_block_tkg_input(
    batch,
    seqlen,
    hidden,
    intermediate,
    num_global_experts,
    num_local_experts,
    top_k,
    router_fn,
    hidden_act_fn,
    expert_affinities_scaling_mode,
    router_act_first,
    norm_topk_prob,
    is_all_expert,
    hidden_actual,
    has_shared_expert,
    has_bias,
    has_clamp,
    skip_router_logits,
    moe_weight_dtype,
    input_dtype,
    router_mm_dtype,
    tensor_generator: Callable = None,
):
    """Build input tensors for moe_block_tkg kernel tests"""

    # Validate expert counts
    if not is_all_expert:
        assert num_global_experts == num_local_experts, (
            f"Selective-load mode requires num_global_experts == num_local_experts, "
            f"got {num_global_experts} vs {num_local_experts}"
        )

    is_mx_weight = is_dtype_mx(moe_weight_dtype)

    if is_mx_weight:
        mx_weights = gen_moe_mx_weights(hidden, intermediate, num_local_experts, moe_weight_dtype)

    if tensor_generator is None:
        np.random.seed(42)
        rng = np.random.default_rng(42)

        def default_tensor_generator(inp):
            if inp.name in ('inp', 'router_w', 'gamma'):
                return dt.static_cast(rng.uniform(low=-0.1, high=0.1, size=inp.shape), inp.dtype)
            elif inp.name == 'expert_gate_up_weights' and is_mx_weight:
                return mx_weights.gate_up_w_qtz
            elif inp.name == 'expert_down_weights' and is_mx_weight:
                return mx_weights.down_w_qtz
            elif inp.name == 'expert_gate_up_weights_scale' and is_mx_weight:
                return mx_weights.gate_up_w_scale
            elif inp.name == 'expert_down_weights_scale' and is_mx_weight:
                return mx_weights.down_w_scale
            elif inp.name in ('expert_gate_up_bias', 'expert_down_bias'):
                return rng.normal(size=inp.shape).astype(inp.dtype)
            else:
                mean = 0.0
                std = 1.0
                return (rng.normal(size=inp.shape) * std + mean).astype(inp.dtype)

        tensor_generator = default_tensor_generator

        num_tokens = batch * seqlen

    # Create a simple object to hold shape, dtype, and name for the generator
    class TensorTemplate:
        def __init__(self, shape, dtype, name):
            self.shape = shape
            self.dtype = dtype
            self.name = name

    hidden_dtype = input_dtype
    weight_dtype = moe_weight_dtype
    scale_dtype = np.uint8 if is_mx_weight else np.float32

    kernel_input_args = {}
    kernel_input_args["inp"] = tensor_generator(TensorTemplate((batch, seqlen, hidden), hidden_dtype, "inp"))
    kernel_input_args["gamma"] = tensor_generator(TensorTemplate((1, hidden), hidden_dtype, "gamma"))

    # Prepare the router weights (uses global experts)
    kernel_input_args["router_weights"] = tensor_generator(
        TensorTemplate((hidden, num_global_experts), router_mm_dtype, "router_weights")
    )

    # Prepare the expert weights (uses local experts)
    if is_mx_weight:
        intermediate_p = (intermediate // 4) if intermediate < 512 else _pmax  # for mxfp4
        n_H512_tile = hidden // (_pmax * _q_width)
        n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))

    expert_gate_up_w_shape = (
        (num_local_experts, _pmax, 2, n_H512_tile, intermediate)
        if is_mx_weight
        else (num_local_experts, hidden, 2, intermediate)
    )
    expert_down_w_shape = (
        (num_local_experts, intermediate_p, n_I512_tile, hidden)
        if is_mx_weight
        else (num_local_experts, intermediate, hidden)
    )
    kernel_input_args["expert_gate_up_weights"] = tensor_generator(
        TensorTemplate(expert_gate_up_w_shape, weight_dtype, "expert_gate_up_weights")
    )
    kernel_input_args["expert_down_weights"] = tensor_generator(
        TensorTemplate(expert_down_w_shape, weight_dtype, "expert_down_weights")
    )

    # Prepare the expert scale (uses local experts)
    expert_gate_up_w_scale_shape = (
        (num_local_experts, _pmax // _q_height, 2, n_H512_tile, intermediate)
        if is_mx_weight
        else (num_local_experts, 2, intermediate)
    )
    expert_down_w_scale_shape = (
        (num_local_experts, intermediate_p // _q_height, n_I512_tile, hidden)
        if is_mx_weight
        else (num_local_experts, hidden)
    )
    kernel_input_args["expert_gate_up_weights_scale"] = (
        tensor_generator(TensorTemplate(expert_gate_up_w_scale_shape, scale_dtype, "expert_gate_up_weights_scale"))
        if is_mx_weight
        else None
    )
    kernel_input_args["expert_down_weights_scale"] = (
        tensor_generator(TensorTemplate(expert_down_w_scale_shape, scale_dtype, "expert_down_weights_scale"))
        if is_mx_weight
        else None
    )

    # Prepare bias tensors (router uses global experts, expert uses local experts)
    kernel_input_args["router_bias"] = (
        tensor_generator(TensorTemplate((1, num_global_experts), router_mm_dtype, "router_bias")) if has_bias else None
    )
    expert_gate_up_bias_shape = (
        (num_local_experts, intermediate_p, 2, n_I512_tile, _q_width)
        if is_mx_weight
        else (num_local_experts, 2, intermediate)
    )
    kernel_input_args["expert_gate_up_bias"] = (
        tensor_generator(TensorTemplate(expert_gate_up_bias_shape, hidden_dtype, "expert_gate_up_bias"))
        if has_bias
        else None
    )
    kernel_input_args["expert_down_bias"] = (
        tensor_generator(TensorTemplate((num_local_experts, hidden), hidden_dtype, "expert_down_bias"))
        if has_bias
        else None
    )

    kernel_input_args["top_k"] = top_k
    kernel_input_args["router_act_fn"] = router_fn
    kernel_input_args["router_pre_norm"] = router_act_first
    kernel_input_args["norm_topk_prob"] = norm_topk_prob

    kernel_input_args["expert_affinities_scaling_mode"] = expert_affinities_scaling_mode
    kernel_input_args["hidden_act_fn"] = hidden_act_fn

    clamp_limit = _get_clamp_limits(has_clamp)
    kernel_input_args["gate_clamp_upper_limit"] = clamp_limit[0]
    kernel_input_args["gate_clamp_lower_limit"] = clamp_limit[1]
    kernel_input_args["up_clamp_upper_limit"] = clamp_limit[2]
    kernel_input_args["up_clamp_lower_limit"] = clamp_limit[3]

    kernel_input_args["router_mm_dtype"] = router_mm_dtype
    kernel_input_args["hidden_actual"] = hidden_actual
    kernel_input_args["skip_router_logits"] = skip_router_logits
    kernel_input_args["is_all_expert"] = is_all_expert

    # rank_id for all-expert mode (rank 0 means all experts are local)
    if is_all_expert:
        kernel_input_args["rank_id"] = np.array([[0]], dtype=np.uint32)
    else:
        kernel_input_args["rank_id"] = None

    return kernel_input_args


def golden_moe_block_tkg(
    kernel_input_args: dict[str, Any],
):
    golden_kernel_input_args = kernel_input_args.copy()
    # Step 0: process input
    hidden = golden_kernel_input_args["inp"]
    dtype = hidden.dtype
    B, S, H = hidden.shape
    T = B * S

    # Step 1: compute RMSNorm
    gamma = golden_kernel_input_args["gamma"]
    hidden_actual = golden_kernel_input_args["hidden_actual"]
    rmsnorm_out = golden_rms_norm(hidden, gamma, hidden_actual=hidden_actual)
    rmsnorm_out = rmsnorm_out.reshape((T, H))

    # Step 2: compute router top-k
    router_w = golden_kernel_input_args["router_weights"]
    _, num_global_experts = router_w.shape
    router_has_bias = "router_bias" in golden_kernel_input_args
    router_w_bias = golden_kernel_input_args["router_bias"] if router_has_bias else None
    k = golden_kernel_input_args["top_k"]
    router_outputs = router_topk_torch_ref(
        x=rmsnorm_out,
        w=router_w,
        w_bias=router_w_bias,
        router_logits=np.zeros((T, num_global_experts), dtype=dtype),
        expert_affinities=np.zeros((T, num_global_experts), dtype=dtype),
        expert_index=np.zeros((T, k), dtype=np.uint32),
        act_fn=golden_kernel_input_args["router_act_fn"],
        k=k,
        x_hbm_layout=1,
        x_sb_layout=0,
        router_pre_norm=golden_kernel_input_args["router_pre_norm"],
        norm_topk_prob=golden_kernel_input_args["norm_topk_prob"],
    )

    # Step 3: compute expert MLP
    #         To reuse the existing MoE golden functions, we need to rename/reassign the inputs
    golden_kernel_input_args["hidden_input"] = rmsnorm_out
    golden_kernel_input_args["expert_affinities"] = router_outputs["expert_affinities"]
    golden_kernel_input_args["expert_index"] = router_outputs["expert_index"]

    is_all_expert = golden_kernel_input_args.get("is_all_expert", False)
    expert_has_bias = golden_kernel_input_args.get("expert_gate_up_bias", None) is not None

    if is_all_expert:
        expert_mlp_outputs = golden_all_expert_moe_tkg(
            inp_np=golden_kernel_input_args,
            expert_affinities_scaling_mode=golden_kernel_input_args["expert_affinities_scaling_mode"],
            dtype=dtype,
            bias=expert_has_bias,
            clamp=golden_kernel_input_args["gate_clamp_upper_limit"] is not None,
            act_fn_type=golden_kernel_input_args["hidden_act_fn"],
            rank_id=0,
            mask_unselected_experts=golden_kernel_input_args["router_pre_norm"],
        )
    else:
        expert_mlp_outputs = golden_selective_expert_moe_tkg(
            inp_np=golden_kernel_input_args,
            expert_affinities_scaling_mode=golden_kernel_input_args["expert_affinities_scaling_mode"],
            dtype=dtype,
            bias=expert_has_bias,
            clamp=golden_kernel_input_args["gate_clamp_upper_limit"] is not None,
            act_fn_type=golden_kernel_input_args["hidden_act_fn"],
        )

    # Gather results
    result_dict = {
        "out": expert_mlp_outputs["out"],
    }

    if not golden_kernel_input_args["skip_router_logits"]:
        result_dict["router_logits"] = router_outputs["router_logits"]

    return result_dict


@pytest_test_metadata(
    name="MoE Block TKG",
    pytest_marks=["moe", "tkg"],
    tags=["model", "trn2", "trn3"],  # Run on TRN2 (full execution) and TRN3 (compile-only)
)
@final
class TestMoEBlockTkgKernel:
    def run_moe_block_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        batch,
        seqlen,
        hidden,
        hidden_actual,
        intermediate,
        num_global_experts,
        num_local_experts,
        top_k,
        router_fn,
        hidden_act_fn,
        expert_affinities_scaling_mode,
        moe_weight_dtype,
        input_dtype,
        has_shared_expert,
        has_bias,
        has_clamp,
        router_act_first,
        norm_topk_prob,
        skip_router_logits,
        router_mm_dtype,
        is_all_expert,
    ):
        kernel_input = build_moe_block_tkg_input(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            intermediate=intermediate,
            num_global_experts=num_global_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
            router_fn=router_fn,
            hidden_act_fn=hidden_act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            router_act_first=router_act_first,
            norm_topk_prob=norm_topk_prob,
            is_all_expert=is_all_expert,
            hidden_actual=hidden_actual,
            has_shared_expert=has_shared_expert,
            has_bias=has_bias,
            has_clamp=has_clamp,
            skip_router_logits=skip_router_logits,
            moe_weight_dtype=moe_weight_dtype,
            input_dtype=input_dtype,
            router_mm_dtype=router_mm_dtype,
        )

        def create_lazy_golden():
            return golden_moe_block_tkg(kernel_input_args=kernel_input)

        # Build output placeholder dict - output shape is [tokens, hidden] (padded dimension)
        tokens = batch * seqlen
        # input_dtype is an NKI dtype (e.g., nl.float16) which is compatible with numpy
        output_placeholder = {"out": np.zeros((tokens, hidden), dtype=input_dtype)}
        if not skip_router_logits:
            output_placeholder["router_logits"] = np.zeros((tokens, num_global_experts), dtype=input_dtype)

        test_manager.execute(
            KernelArgs(
                kernel_func=moe_block_tkg_kernel,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=5e-2 if is_dtype_low_precision(moe_weight_dtype) else 1e-2,
                    absolute_accuracy=1e-5,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    @pytest.mark.fast
    @range_test_config(moe_block_tkg_combined_config())
    def test_moe_block_kernel(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        platform_target: Platforms,
        request,
    ):
        """Test MoE Block TKG kernel with manual and model configurations."""
        cfg = range_test_options.tensors[MOE_BLOCK_TKG_CONFIG]

        moe_tkg_metadata_list = load_model_configs("test_moe_block")

        # Apply xfail for model configs and add metadata dimensions
        if range_test_options.test_type == MODEL_TEST_TYPE:
            request.node.add_marker(pytest.mark.xfail(strict=False, reason="Model coverage test"))
            test_metadata_key = {
                "ln": cfg[LNC_DIM_NAME],
                "ae": bool(cfg[IS_ALL_EXPERT_DIM_NAME]),
                "ba": cfg[BATCH_DIM_NAME],
                "sq": cfg[SEQLEN_DIM_NAME],
                "hi": cfg[HIDDEN_DIM_NAME],
                "im": cfg[INTERMEDIATE_DIM_NAME],
                "ge": cfg[NUM_GLOBAL_EXPERTS_DIM_NAME],
                "le": cfg[NUM_LOCAL_EXPERTS_DIM_NAME],
            }
            collector.match_and_add_metadata_dimensions(test_metadata_key, moe_tkg_metadata_list)

        lnc = cfg[LNC_DIM_NAME]
        moe_weight_dtype = cfg[MOE_WEIGHT_DTYPE_DIM_NAME]

        if is_dtype_mx(moe_weight_dtype) and platform_target is not Platforms.TRN3:
            pytest.skip("MX is only supported on TRN3.")

        is_all_expert = bool(cfg[IS_ALL_EXPERT_DIM_NAME])
        intermediate = cfg[INTERMEDIATE_DIM_NAME]

        # xfail specific configurations failing determinism check
        batch = cfg[BATCH_DIM_NAME]
        seqlen = cfg[SEQLEN_DIM_NAME]
        if not is_all_expert and intermediate == 192 and (batch * seqlen) == 1 and not is_dtype_mx(moe_weight_dtype):
            pytest.xfail("failing determinism check")

        compiler_args = CompilerArgs(logical_nc_config=lnc, platform_target=platform_target)
        self.run_moe_block_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            batch=cfg[BATCH_DIM_NAME],
            seqlen=cfg[SEQLEN_DIM_NAME],
            hidden=cfg[HIDDEN_DIM_NAME],
            hidden_actual=cfg[HIDDEN_ACTUAL_DIM_NAME],
            intermediate=intermediate,
            num_global_experts=cfg[NUM_GLOBAL_EXPERTS_DIM_NAME],
            num_local_experts=cfg[NUM_LOCAL_EXPERTS_DIM_NAME],
            top_k=cfg[TOP_K_DIM_NAME],
            router_fn=cfg[ROUTER_FN_DIM_NAME],
            hidden_act_fn=cfg[HIDDEN_ACT_FN_DIM_NAME],
            expert_affinities_scaling_mode=ExpertAffinityScaleMode(cfg[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]),
            moe_weight_dtype=moe_weight_dtype,
            input_dtype=cfg[INPUT_DTYPE_DIM_NAME],
            has_shared_expert=bool(cfg[HAS_SHARED_EXPERT_DIM_NAME]),
            has_bias=bool(cfg[HAS_BIAS_DIM_NAME]),
            has_clamp=bool(cfg[HAS_CLAMP_DIM_NAME]),
            router_act_first=bool(cfg[ROUTER_ACT_FIRST_DIM_NAME]),
            norm_topk_prob=bool(cfg[NORM_TOPK_PROB_DIM_NAME]),
            skip_router_logits=bool(cfg[SKIP_ROUTER_LOGITS_DIM_NAME]),
            router_mm_dtype=cfg[ROUTER_MM_DTYPE_DIM_NAME],
            is_all_expert=is_all_expert,
        )
