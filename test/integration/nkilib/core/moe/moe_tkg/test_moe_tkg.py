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

try:
    from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_model_config import (
        moe_tkg_model_configs,
    )
except ImportError:
    moe_tkg_model_configs = []

from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import (
    build_moe_tkg,
    golden_all_expert_moe_tkg,
    golden_selective_expert_moe_tkg,
)
from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_wrapper import (
    moe_tkg_sbuf_io_wrapper,
)
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
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeMonotonicGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe import moe_tkg
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
)

MOE_TKG_CONFIG = "moe_tkg_config"
VNC_DEGREE_DIM_NAME = "vnc"
TOKENS_DIM_NAME = "tokens"
HIDDEN_DIM_NAME = "h"
INTERMEDIATE_DIM_NAME = "i"
EXPERT_DIM_NAME = "expert"
TOP_K_DIM_NAME = "top_k"
ACT_FN_DIM_NAME = "act_fn"
EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME = (
    "scaling_mode"  # expert affinity scaling mode, short name to avoid filename length limit
)
QUANT_DTYPE_DIM_NAME = "quant_dtype"
QUANT_TYPE_DIM_NAME = "quant_type"
DTYPE_DIM_NAME = "dtype"
IN_DTYPE_DIM_NAME = "in_dtype"
OUT_DTYPE_DIM_NAME = "out_dtype"
CLAMP_DIM_NAME = "clamp"
BIAS_DIM_NAME = "bias"
IS_ALL_EXPERT_DIM_NAME = "is_all_expert"
DUMMY_STEP_SIZE = 1  # Placeholder value for range configs

# Configuration-based testing
ALL_EXPERT_BASIC_CONFIG = {
    "in_dtype": nl.float16,
    "act_fn": ActFnType.SiLU,
    "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
    "is_all_expert": True,
    "clamp": True,
    "bias": False,
}

ALL_EXPERT_FULL_FEATURE_CONFIG = {
    "in_dtype": nl.float16,
    "act_fn": ActFnType.SiLU,
    "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
    "is_all_expert": True,
    "clamp": True,
    "bias": True,
}

SELECTIVE_EXPERT_BASIC_CONFIG = {
    "in_dtype": nl.float16,
    "act_fn": ActFnType.Swish,
    "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
    "is_all_expert": False,
    "clamp": True,
    "bias": False,
}

SELECTIVE_EXPERT_FULL_FEATURE_CONFIG = {
    "in_dtype": nl.float16,
    "act_fn": ActFnType.Swish,
    "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
    "is_all_expert": False,
    "clamp": True,
    "bias": True,
}

# fmt: off
moe_tkg_sbuf_io_params = [
# vnc_degree, tokens, hidden, intermediate, expert, top_k, act_fn,          expert_affinities_scaling_mode,     is_all_expert, dtype,      clamp, bias
    [2,        4,     3072,   128,          2,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          nl.bfloat16, False, False],
    [2,        4,     3072,   128,          2,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.bfloat16, True,  True],
    [2,        4,     3072,   128,          4,      2,     ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False,         nl.bfloat16, False, False],
    [2,        4,     3072,   128,          4,      2,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.bfloat16, True,  True],
]
moe_tkg_kernel_bfloat16_params = [
# vnc_degree, tokens, hidden, intermediate, expert, top_k, act_fn,          expert_affinities_scaling_mode,     is_all_expert, quant_dtype, quant_type,            dtype,       clamp, bias
    # All experts (top_k is None)
    # Large hidden test case for SBUF allocation
    [2,        4,     32768,  256,          4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True, False],
    [2,        32,    3072,   512,          1,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   True,          None,        QuantizationType.NONE, nl.float16,  True,  False],
    [2,        4,     3072,   64,           4,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        4,     3072,   192,          4,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        32,    3072,   768,          8,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        32,    3072,   768,          128,    None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        32,    3072,   1536,         4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  True],
    # gptoss_120b: hidden=3072, moe_intermediate=3072, num_local_experts=128, top_k=4, tp=8, ep=16
    [2,        32,    3072,   384,          8,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  True],  # 3072/8=384, 128/16=8
    # qwen3_235b_a22b: hidden=4096, moe_intermediate=1536, num_local_experts=128, top_k=8, tp=8, ep=16
    [2,        2,     4096,   192,          8,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  False],  # 1536/8=192, 128/16=8
    # llama4_maverick: hidden=5120, moe_intermediate=8192, num_local_experts=128, top_k=1, tp=64, ep=1
    [2,        2,     5120,   128,          128,    None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  False],  # 8192/64=128, 128/1=128
    # llama4_scout: hidden=5120, moe_intermediate=8192, num_local_experts=16, top_k=1, tp=64, ep=1
    [2,        2,     5120,   128,          16,     None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True,  False],  # 8192/64=128, 16/1=16
    # Negative test: H=384 -> H1=3 cannot be evenly divided by 2 cores
    [2,        4,     384,    128,          4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True, False],
    # Negative test: H=378 not divisible by 128
    [2,        4,     378,    128,          4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          None,        QuantizationType.NONE, nl.float16,  True, False],

    [2,        4,     3072,   384,          4,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,         None,        QuantizationType.NONE, nl.float16,  True,  True],

    # Selective experts (top_k is not None)
    [2,        2,     512,    128,          2,      1,     ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  False],
    [2,        2,     512,    64,           2,      1,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        4,     512,    128,          2,      1,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    # Selective experts with odd T (shard_on_T enabled with uneven distribution)
    [2,        3,     512,    128,          2,      1,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        5,     512,    128,          4,      2,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        7,     512,    128,          4,      2,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        4,     3072,   192,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        1,     3072,   384,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        4,     3072,   384,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        16,    3072,   512,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        32,    3072,   1024,         128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    [2,        32,    3072,   1536,         128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    # gptoss_120b
    [2,        32,    3072,   384,          8,      4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  True],
    # qwen3_235b_a22b
    [2,        2,     4096,   192,          8,      8,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  False],
    # llama4_maverick
    [2,        2,     5120,   128,          128,    1,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  False],
    # llama4_scout
    [2,        2,     5120,   128,          16,     1,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True,  False],
    [2,        4,     384,    128,          4,      2,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         None,        QuantizationType.NONE, nl.float16,  True, False],
]

moe_tkg_kernel_mxfp4_quant_params = [
# vnc_degree, tokens, hidden, intermediate, expert, top_k, act_fn,          expert_affinities_scaling_mode,     is_all_expert, quant_dtype,          quant_type,           dtype,        clamp, bias
    # All experts
    [2,        32,    3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        64,    3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        128,   3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        256,   3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        512,   3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        1024,  3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        640,   3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        32,    3072,   3072,         2,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        32,    3072,   3072,         4,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        32,    3072,   3072,         8,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        32,    3072,   3072,         16,     None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        128,   4096,   3072,         4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    # Selective experts
    [2,        1,     3072,   192,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        4,     3072,   192,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        4,     3072,   384,          128,    4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        1,     512,    64,           128,    2,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    [2,        2,     3072,   1536,         128,    8,     ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False,         nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
]

moe_tkg_kernel_mxfp8_quant_params = [
# vnc_degree, tokens, hidden, intermediate, expert, top_k, act_fn,          expert_affinities_scaling_mode,     is_all_expert, quant_dtype,          quant_type,           dtype,        clamp, bias
    # All experts
    [2,        128,   4096,   3072,         4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          nl.float8_e4m3fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
    # Selective experts
    [2,        2,     3072,   1536,         128,    8,     ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, False,         nl.float8_e4m3fn_x4,  QuantizationType.MX,  nl.bfloat16,  True,  True],
]

moe_tkg_kernel_fp8_quant_params = [
# vnc_degree, tokens, hidden, intermediate, expert, top_k, act_fn,          expert_affinities_scaling_mode,     is_all_expert, quant_dtype,     quant_type,            dtype,       clamp, bias
    # All experts
    [2,        4,     512,    64,           2,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          nl.float8_e4m3,  QuantizationType.ROW,  nl.float16,  True,  True],
    [2,        4,     1024,   128,          4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          nl.float8_e4m3,  QuantizationType.ROW,  nl.float16,  True,  True],
    [2,        32,    3072,   192,          4,      None,  ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, True,          nl.float8_e4m3,  QuantizationType.ROW,  nl.float16,  True,  True],
    # Selective experts
    [2,        4,     512,    64,           4,      2,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float8_e4m3,  QuantizationType.ROW,  nl.float16,  True,  True],
    [2,        4,     3072,   128,          8,      4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float8_e4m3,  QuantizationType.ROW,  nl.float16,  True,  True],
    [2,        32,    3072,   192,          8,      4,     ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, False,         nl.float8_e4m3,  QuantizationType.ROW,  nl.float16,  True,  True],
]

# Separate IO dtype tests - tests with different in_dtype and out_dtype combinations
moe_tkg_io_dtype_params = [
# vnc_degree, tokens, hidden, intermediate, expert, top_k, act_fn,          expert_affinities_scaling_mode,     is_all_expert, quant_dtype,         quant_type,            in_dtype,    out_dtype,   clamp,  bias
    # All experts - different IO dtypes
    [2,        32,    3072,   3072,         1,      None,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, True,          nl.float4_e2m1fn_x4,  QuantizationType.MX,  nl.float16,  nl.bfloat16,  True,  True],
]
# fmt: on
def create_moe_tkg_test_config(test_vectors, test_type: str = "manual"):
    """
    Utility function to create complete RangeTestConfig from list of MoE TKG config vectors.

    Args:
        test_vectors: List of test config vectors
        test_type: Test type label for test names

    Returns:
        Complete RangeTestConfig ready to use with @range_test_config decorator
    """
    test_cases = []
    for test_vector in test_vectors:
        for test_params in test_vector:
            (
                vnc_degree,
                tokens,
                hidden,
                intermediate,
                expert,
                top_k,
                act_fn,
                expert_affinities_scaling_mode,
                is_all_expert,
                quant_dtype,
                quant_type,
                dtype,
                clamp,
                bias,
            ) = test_params

            test_case = {
                MOE_TKG_CONFIG: {
                    VNC_DEGREE_DIM_NAME: vnc_degree,
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    TOP_K_DIM_NAME: top_k,
                    ACT_FN_DIM_NAME: act_fn,
                    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode.value,
                    IS_ALL_EXPERT_DIM_NAME: int(is_all_expert),
                    QUANT_DTYPE_DIM_NAME: quant_dtype,
                    QUANT_TYPE_DIM_NAME: quant_type,
                    IN_DTYPE_DIM_NAME: dtype,
                    OUT_DTYPE_DIM_NAME: dtype,
                    CLAMP_DIM_NAME: int(clamp),
                    BIAS_DIM_NAME: int(bias),
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


# Load model metadata for matching test configs to model names
moe_tkg_metadata_list = load_model_configs("test_moe_tkg")


def moe_tkg_combined_config():
    """Create combined config with manual and model test cases."""

    manual_config = create_moe_tkg_test_config(
        [
            moe_tkg_kernel_bfloat16_params,
            moe_tkg_kernel_mxfp4_quant_params,
            moe_tkg_kernel_mxfp8_quant_params,
            moe_tkg_kernel_fp8_quant_params,
        ],
        test_type="manual",
    )
    model_config = create_moe_tkg_test_config([moe_tkg_model_configs], test_type=MODEL_TEST_TYPE)
    # Combine by extending generators
    manual_config.global_tensor_configs.custom_generators.extend(model_config.global_tensor_configs.custom_generators)
    return manual_config


@pytest_test_metadata(
    name="MoE TKG",
    pytest_marks=["moe", "tkg"],
    tags=["model"],
)
@final
class TestMoeTkgKernels:
    def run_moe_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree,
        quant_dtype,
        quant_type,
        in_dtype,
        out_dtype,
        expert_affinities_scaling_mode,
        is_all_expert,
        clamp,
        bias,
        collector: IMetricsCollector,
    ):
        # test_options is shared across multiple configs.
        # Model configs should never be marked as negative test cases
        is_model_config = test_options.test_type == MODEL_TEST_TYPE
        # Initialize `is_negative_test_case` to False (or bypass for model configs)
        is_negative_test_case = False

        mlp_config = test_options.tensors[MOE_TKG_CONFIG]

        tokens = mlp_config[TOKENS_DIM_NAME]
        hidden = mlp_config[HIDDEN_DIM_NAME]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]
        expert = mlp_config[EXPERT_DIM_NAME]
        top_k = mlp_config[TOP_K_DIM_NAME]
        act_fn = mlp_config[ACT_FN_DIM_NAME]
        is_mx_quant = quant_type is QuantizationType.MX

        # Hidden for each core must be divisible by 128
        if hidden // lnc_degree % 128 != 0 and (is_all_expert or tokens == 1) and not is_model_config:
            is_negative_test_case = True

        # H1 must be evenly divisible by num_shards (vnc_degree)
        H1 = hidden // 128
        if H1 % lnc_degree != 0 and (is_all_expert or tokens == 1) and not is_model_config:
            is_negative_test_case = True

        with assert_negative_test_case(is_negative_test_case):
            kernel_input = build_moe_tkg(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=act_fn,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                is_all_expert=is_all_expert,
                quant_dtype=quant_dtype,
                quant_type=quant_type,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                bias=bias,
                clamp=clamp,
            )

            # Create lazy golden generator with closure capturing all needed variables
            def create_lazy_golden():
                # FIXME: all_expert mx_quant need to have its own golden function
                if is_all_expert and not is_mx_quant:
                    return golden_all_expert_moe_tkg(
                        inp_np=kernel_input,
                        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                        dtype=out_dtype,
                        quant_dtype=quant_dtype,
                        quant_type=quant_type,
                        bias=bias,
                        clamp=clamp,
                        act_fn_type=act_fn,
                        rank_id=0,
                        mask_unselected_experts=False,
                    )
                else:
                    return golden_selective_expert_moe_tkg(
                        inp_np=kernel_input,
                        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                        dtype=out_dtype,
                        quant_dtype=quant_dtype,
                        quant_type=quant_type,
                        bias=bias,
                        clamp=clamp,
                        act_fn_type=act_fn,
                    )

            # Output shape is (tokens, hidden)
            output_placeholder = {"out": np.zeros((tokens, hidden), dtype=out_dtype)}

            validation_args = ValidationArgs(
                golden_output=LazyGoldenGenerator(
                    lazy_golden_generator=create_lazy_golden,
                    output_ndarray=output_placeholder,
                ),
                relative_accuracy=5e-2 if is_mx_quant else 1e-2,
                absolute_accuracy=1e-5,
            )

            from nkilib_src.nkilib.core.moe import moe_tkg

            test_manager.execute(
                KernelArgs(
                    kernel_func=moe_tkg,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=validation_args,
                    inference_args=TKG_INFERENCE_ARGS,
                )
            )

    @staticmethod
    def moe_tkg_tokens_sweep_config():
        # Sweep T from 1 to 128 (8 values) x 4 configs = 32 tests
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=128, power_of=2, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=256, max=256, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=4, max=4, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_hidden_sweep_config():
        # Sweep H from 256 to 32K (8 values) x 4 configs = 32 tests
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=256, max=32768, power_of=2, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=256, max=256, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=4, max=4, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_intermediate_sweep_config():
        # Sweep I from 128 to 1024 (4 values) x 4 configs = 16 tests
        # Note: 64 is covered by moe_tkg_intermediate_non_multiple_128_sweep_config
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=128, max=1024, power_of=2, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=4, max=4, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_intermediate_non_multiple_128_sweep_config():
        # Sweep I non-multiple of 128: 64, 192, 320, 448, 576, 704, 832, 960 (8 values) x 4 configs = 32 tests
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=64, max=960, multiple_of=64, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=4, max=4, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_expert_sweep_config():
        # Sweep E from 2 to 128 (7 values) x 4 configs = 28 tests
        # Note: E=1 is not a valid MoE workload (it's dense), and local_gather requires src_buffer_size > 1
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=256, max=256, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=2, max=128, power_of=2, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_combined_sweep_config():
        # Combined sweep with multiple dimensions varying
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=2, max=32, power_of=2, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=512, max=8192, power_of=2, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=128, max=512, power_of=2, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=2, max=32, power_of=2, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_large_hidden_sweep_config():
        # Large hidden sweep: H=16384 and 32768 only (smaller values covered by moe_tkg_hidden_sweep_config)
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=16384, max=32768, power_of=2, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=256, max=256, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=4, max=4, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_large_expert_sweep_config():
        # Large expert sweep: E=64 and 128 with various T (smaller E values covered by moe_tkg_expert_sweep_config)
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=2, max=16, power_of=2, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=128, max=128, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=64, max=128, power_of=2, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_tokens_hidden_sweep_config():
        # T x H sweep: T from 1-64, H from 1024-8192
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=1, max=64, power_of=2, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=1024, max=8192, power_of=2, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=256, max=256, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=8, max=8, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_expert_intermediate_sweep_config():
        # E x I sweep: E from 2-32, I from 128-512
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=8, max=8, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=128, max=512, power_of=2, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=2, max=32, power_of=2, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    @staticmethod
    def moe_tkg_topk_sweep_config():
        # Sweep top_k from 1 to 8 for selective expert (E=16 to ensure top_k <= E)
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={
                    MOE_TKG_CONFIG: TensorConfig(
                        [
                            DimensionRangeConfig(min=4, max=4, name=TOKENS_DIM_NAME),
                            DimensionRangeConfig(min=3072, max=3072, name=HIDDEN_DIM_NAME),
                            DimensionRangeConfig(min=256, max=256, name=INTERMEDIATE_DIM_NAME),
                            DimensionRangeConfig(min=16, max=16, name=EXPERT_DIM_NAME),
                        ]
                    ),
                },
                monotonic_step_size=DUMMY_STEP_SIZE,
                custom_generators=[RangeMonotonicGeneratorStrategy(step_size=DUMMY_STEP_SIZE)],
            ),
        )

    # Unit Tests Entry Point
    @pytest.mark.fast
    @range_test_config(moe_tkg_combined_config())
    def test_moe_tkg_unit(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        platform_target: Platforms,
        request,
    ):
        # Get the tensor config
        mlp_config = range_test_options.tensors[MOE_TKG_CONFIG]

        # Apply xfail for model configs and add metadata dimensions
        if range_test_options.test_type == MODEL_TEST_TYPE:
            request.node.add_marker(pytest.mark.xfail(strict=False, reason="Model coverage test"))
            test_metadata_key = {
                "tokens": mlp_config[TOKENS_DIM_NAME],
                "h": mlp_config[HIDDEN_DIM_NAME],
                "i": mlp_config[INTERMEDIATE_DIM_NAME],
                "expert": mlp_config[EXPERT_DIM_NAME],
            }
            collector.match_and_add_metadata_dimensions(test_metadata_key, moe_tkg_metadata_list)

        lnc_count = mlp_config[VNC_DEGREE_DIM_NAME]
        quant_type = mlp_config[QUANT_TYPE_DIM_NAME]

        if quant_type is QuantizationType.MX and platform_target is not Platforms.TRN3:
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=platform_target)
        self.run_moe_tkg_test(
            test_manager=test_manager,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            compiler_args=compiler_args,
            quant_dtype=mlp_config[QUANT_DTYPE_DIM_NAME],
            quant_type=mlp_config[QUANT_TYPE_DIM_NAME],
            in_dtype=mlp_config[IN_DTYPE_DIM_NAME],
            out_dtype=mlp_config[OUT_DTYPE_DIM_NAME],
            expert_affinities_scaling_mode=ExpertAffinityScaleMode(mlp_config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]),
            is_all_expert=bool(mlp_config[IS_ALL_EXPERT_DIM_NAME]),
            clamp=bool(mlp_config[CLAMP_DIM_NAME]),
            bias=bool(mlp_config[BIAS_DIM_NAME]),
            collector=collector,
        )

    def _run_sweep_test(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        config,
        top_k,
    ):
        mlp_config = range_test_options.tensors[MOE_TKG_CONFIG]
        tokens = mlp_config[TOKENS_DIM_NAME]
        hidden = mlp_config[HIDDEN_DIM_NAME]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]
        expert = mlp_config[EXPERT_DIM_NAME]
        in_dtype = config["in_dtype"]

        if top_k is not None and top_k > expert:
            pytest.skip(f"top_k ({top_k}) > expert ({expert})")

        compiler_args = CompilerArgs()
        is_negative_test_case = hidden // compiler_args.logical_nc_config % 128 != 0

        with assert_negative_test_case(is_negative_test_case):
            kernel_input = build_moe_tkg(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                top_k=top_k,
                act_fn=config["act_fn"],
                expert_affinities_scaling_mode=config["expert_affinities_scaling_mode"],
                is_all_expert=config["is_all_expert"],
                in_dtype=in_dtype,
                out_dtype=in_dtype,
                bias=config["bias"],
                clamp=config["clamp"],
            )

            def create_lazy_golden():
                if top_k is None:
                    return golden_all_expert_moe_tkg(
                        inp_np=kernel_input,
                        expert_affinities_scaling_mode=config["expert_affinities_scaling_mode"],
                        dtype=in_dtype,
                        bias=config["bias"],
                        clamp=config["clamp"],
                        act_fn_type=config["act_fn"],
                        rank_id=0,
                        mask_unselected_experts=False,
                    )
                else:
                    return golden_selective_expert_moe_tkg(
                        inp_np=kernel_input,
                        expert_affinities_scaling_mode=config["expert_affinities_scaling_mode"],
                        dtype=in_dtype,
                        bias=config["bias"],
                        clamp=config["clamp"],
                        act_fn_type=config["act_fn"],
                    )

            from nkilib_src.nkilib.core.moe import moe_tkg

            test_manager.execute(
                KernelArgs(
                    kernel_func=moe_tkg,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=create_lazy_golden,
                            output_ndarray={"out": np.zeros((tokens, hidden), dtype=in_dtype)},
                        ),
                        relative_accuracy=1e-2,
                        absolute_accuracy=1e-5,
                    ),
                    inference_args=TKG_INFERENCE_ARGS,
                )
            )

    @range_test_config(moe_tkg_tokens_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_tokens_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_hidden_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_hidden_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_intermediate_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_intermediate_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_intermediate_non_multiple_128_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_intermediate_non_multiple_128_sweep(
        self, test_manager, range_test_options, collector, config_name, config, top_k
    ):
        # xfail specific configurations failing determinism check
        mlp_config = range_test_options.tensors[MOE_TKG_CONFIG]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]
        if config_name == "selective_expert_full_features_topk4" and intermediate == 960:
            pytest.xfail("failing determinism check")
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_expert_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_expert_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_combined_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_combined_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_large_hidden_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_large_hidden_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_large_expert_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_large_expert_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_tokens_hidden_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_tokens_hidden_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_expert_intermediate_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("all_expert_basic", ALL_EXPERT_BASIC_CONFIG, None),
            ("all_expert_full_features", ALL_EXPERT_FULL_FEATURE_CONFIG, None),
            ("selective_expert_basic_topk1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_expert_full_features_topk4", SELECTIVE_EXPERT_FULL_FEATURE_CONFIG, 4),
        ],
    )
    def test_moe_tkg_expert_intermediate_sweep(
        self, test_manager, range_test_options, collector, config_name, config, top_k
    ):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    # Sweep tests for dtype, scale_mode, and act_fn
    @range_test_config(moe_tkg_tokens_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            # in_dtype sweep: float16
            ("dtype_float16", {**ALL_EXPERT_BASIC_CONFIG, "in_dtype": nl.float16}, None),
            # scale_mode sweep: NO_SCALE, POST_SCALE
            (
                "scale_no_scale",
                {**ALL_EXPERT_BASIC_CONFIG, "expert_affinities_scaling_mode": ExpertAffinityScaleMode.NO_SCALE},
                None,
            ),
            (
                "scale_post_scale",
                {**ALL_EXPERT_BASIC_CONFIG, "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE},
                None,
            ),
            # act_fn sweep: SiLU, GELU, GELU_Tanh_Approx, Swish
            ("act_silu", {**ALL_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.SiLU}, None),
            ("act_gelu", {**ALL_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.GELU}, None),
            ("act_gelu_tanh", {**ALL_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.GELU_Tanh_Approx}, None),
            ("act_swish", {**ALL_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.Swish}, None),
        ],
    )
    def test_moe_tkg_dtype_scale_actfn_sweep(
        self, test_manager, range_test_options, collector, config_name, config, top_k
    ):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_tokens_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            # bf16 has lower precision, use 2% tolerance
            ("dtype_bfloat16", {**ALL_EXPERT_BASIC_CONFIG, "in_dtype": nl.bfloat16}, None),
        ],
    )
    def test_moe_tkg_dtype_bfloat16_sweep(
        self, test_manager, range_test_options, collector, config_name, config, top_k
    ):
        mlp_config = range_test_options.tensors[MOE_TKG_CONFIG]
        tokens = mlp_config[TOKENS_DIM_NAME]
        hidden = mlp_config[HIDDEN_DIM_NAME]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]
        expert = mlp_config[EXPERT_DIM_NAME]
        in_dtype = config["in_dtype"]

        compiler_args = CompilerArgs()
        kernel_input = build_moe_tkg(
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            expert=expert,
            top_k=top_k,
            act_fn=config["act_fn"],
            expert_affinities_scaling_mode=config["expert_affinities_scaling_mode"],
            is_all_expert=config["is_all_expert"],
            in_dtype=in_dtype,
            out_dtype=in_dtype,
            bias=config["bias"],
            clamp=config["clamp"],
        )

        def create_lazy_golden():
            return golden_all_expert_moe_tkg(
                inp_np=kernel_input,
                expert_affinities_scaling_mode=config["expert_affinities_scaling_mode"],
                dtype=in_dtype,
                bias=config["bias"],
                clamp=config["clamp"],
                act_fn_type=config["act_fn"],
                rank_id=0,
                mask_unselected_experts=False,
            )

        from nkilib_src.nkilib.core.moe import moe_tkg

        test_manager.execute(
            KernelArgs(
                kernel_func=moe_tkg,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray={"out": np.zeros((tokens, hidden), dtype=in_dtype)},
                    ),
                    relative_accuracy=2e-2,
                    absolute_accuracy=1e-5,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    @range_test_config(moe_tkg_tokens_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            # in_dtype sweep for selective expert
            ("selective_dtype_float16", {**SELECTIVE_EXPERT_BASIC_CONFIG, "in_dtype": nl.float16}, 1),
            ("selective_dtype_bfloat16", {**SELECTIVE_EXPERT_BASIC_CONFIG, "in_dtype": nl.bfloat16}, 1),
            # scale_mode sweep for selective expert
            (
                "selective_scale_no_scale",
                {**SELECTIVE_EXPERT_BASIC_CONFIG, "expert_affinities_scaling_mode": ExpertAffinityScaleMode.NO_SCALE},
                1,
            ),
            (
                "selective_scale_post_scale",
                {**SELECTIVE_EXPERT_BASIC_CONFIG, "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE},
                1,
            ),
            # act_fn sweep for selective expert
            ("selective_act_silu", {**SELECTIVE_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.SiLU}, 1),
            ("selective_act_gelu", {**SELECTIVE_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.GELU}, 1),
            ("selective_act_gelu_tanh", {**SELECTIVE_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.GELU_Tanh_Approx}, 1),
            ("selective_act_swish", {**SELECTIVE_EXPERT_BASIC_CONFIG, "act_fn": ActFnType.Swish}, 1),
        ],
    )
    def test_moe_tkg_selective_dtype_scale_actfn_sweep(
        self, test_manager, range_test_options, collector, config_name, config, top_k
    ):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @range_test_config(moe_tkg_topk_sweep_config())
    @pytest.mark.parametrize(
        "config_name, config, top_k",
        [
            ("selective_topk_1", SELECTIVE_EXPERT_BASIC_CONFIG, 1),
            ("selective_topk_2", SELECTIVE_EXPERT_BASIC_CONFIG, 2),
            ("selective_topk_4", SELECTIVE_EXPERT_BASIC_CONFIG, 4),
            ("selective_topk_8", SELECTIVE_EXPERT_BASIC_CONFIG, 8),
        ],
    )
    def test_moe_tkg_topk_sweep(self, test_manager, range_test_options, collector, config_name, config, top_k):
        self._run_sweep_test(test_manager, range_test_options, config, top_k)

    @staticmethod
    def moe_tkg_sbuf_io_config():
        test_cases = []
        for test_params in moe_tkg_sbuf_io_params:
            (
                vnc_degree,
                tokens,
                hidden,
                intermediate,
                expert,
                top_k,
                act_fn,
                expert_affinities_scaling_mode,
                is_all_expert,
                dtype,
                clamp,
                bias,
            ) = test_params
            test_case = {
                MOE_TKG_CONFIG: {
                    VNC_DEGREE_DIM_NAME: vnc_degree,
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    TOP_K_DIM_NAME: top_k,
                    ACT_FN_DIM_NAME: act_fn,
                    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode.value,
                    IS_ALL_EXPERT_DIM_NAME: int(is_all_expert),
                    IN_DTYPE_DIM_NAME: dtype,
                    CLAMP_DIM_NAME: int(clamp),
                    BIAS_DIM_NAME: int(bias),
                },
            }
            test_cases.append(test_case)
        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={},
                monotonic_step_size=1,
                custom_generators=[
                    RangeManualGeneratorStrategy(test_cases=test_cases),
                ],
            ),
        )

    @pytest.mark.fast
    @range_test_config(moe_tkg_sbuf_io_config())
    def test_moe_tkg_sbuf_io(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        """Test TkG MoE kernel with SBUF input."""
        mlp_config = range_test_options.tensors[MOE_TKG_CONFIG]
        lnc_count = mlp_config[VNC_DEGREE_DIM_NAME]
        in_dtype = mlp_config[IN_DTYPE_DIM_NAME]
        tokens = mlp_config[TOKENS_DIM_NAME]
        hidden = mlp_config[HIDDEN_DIM_NAME]
        intermediate = mlp_config[INTERMEDIATE_DIM_NAME]
        expert = mlp_config[EXPERT_DIM_NAME]
        act_fn = mlp_config[ACT_FN_DIM_NAME]
        expert_affinities_scaling_mode = ExpertAffinityScaleMode(mlp_config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME])
        is_all_expert = bool(mlp_config[IS_ALL_EXPERT_DIM_NAME])
        bias = bool(mlp_config[BIAS_DIM_NAME])
        clamp = bool(mlp_config[CLAMP_DIM_NAME])

        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=platform_target)

        kernel_input = build_moe_tkg(
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            expert=expert,
            top_k=mlp_config[TOP_K_DIM_NAME],
            act_fn=act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            is_all_expert=is_all_expert,
            in_dtype=in_dtype,
            bias=bias,
            clamp=clamp,
        )

        wrapper_input = {
            "hidden_input": kernel_input["hidden_input"],
            "gate_up_weights": kernel_input["expert_gate_up_weights"],
            "down_weights": kernel_input["expert_down_weights"],
            "expert_affinities": kernel_input["expert_affinities"],
            "expert_index": kernel_input["expert_index"],
            "is_all_expert": is_all_expert,
            "rank_id": kernel_input["rank_id"],
            "gate_up_weights_bias": kernel_input["expert_gate_up_bias"],
            "down_weights_bias": kernel_input["expert_down_bias"],
            "expert_affinities_scaling_mode": expert_affinities_scaling_mode,
            "activation_fn": act_fn,
            "gate_clamp_upper_limit": float(7.0) if clamp else None,
            "gate_clamp_lower_limit": None,
            "up_clamp_upper_limit": float(8.0) if clamp else None,
            "up_clamp_lower_limit": float(-6.0) if clamp else None,
            "mask_unselected_experts": kernel_input["mask_unselected_experts"],
        }

        def create_lazy_golden():
            if is_all_expert:
                return golden_all_expert_moe_tkg(
                    inp_np=kernel_input,
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    dtype=in_dtype,
                    bias=bias,
                    clamp=clamp,
                    act_fn_type=act_fn,
                    rank_id=0,
                    mask_unselected_experts=False,
                )
            else:
                return golden_selective_expert_moe_tkg(
                    inp_np=kernel_input,
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    dtype=in_dtype,
                    bias=bias,
                    clamp=clamp,
                    act_fn_type=act_fn,
                )

        test_manager.execute(
            KernelArgs(
                kernel_func=moe_tkg_sbuf_io_wrapper,
                compiler_input=compiler_args,
                kernel_input=wrapper_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray={"out": np.zeros((tokens, hidden), dtype=in_dtype)},
                    ),
                    absolute_accuracy=1e-5,
                    relative_accuracy=1e-2,
                ),
                inference_args=TKG_INFERENCE_ARGS,
            )
        )

    @staticmethod
    def moe_tkg_io_dtype_config():
        """Config for testing different in_dtype and out_dtype combinations."""
        test_cases = []
        for test_params in moe_tkg_io_dtype_params:
            (
                vnc_degree,
                tokens,
                hidden,
                intermediate,
                expert,
                top_k,
                act_fn,
                expert_affinities_scaling_mode,
                is_all_expert,
                quant_dtype,
                quant_type,
                in_dtype,
                out_dtype,
                clamp,
                bias,
            ) = test_params

            test_case = {
                MOE_TKG_CONFIG: {
                    VNC_DEGREE_DIM_NAME: vnc_degree,
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    TOP_K_DIM_NAME: top_k,
                    ACT_FN_DIM_NAME: act_fn,
                    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode.value,
                    IS_ALL_EXPERT_DIM_NAME: int(is_all_expert),
                    QUANT_DTYPE_DIM_NAME: quant_dtype,
                    QUANT_TYPE_DIM_NAME: quant_type,
                    IN_DTYPE_DIM_NAME: in_dtype,
                    OUT_DTYPE_DIM_NAME: out_dtype,
                    CLAMP_DIM_NAME: int(clamp),
                    BIAS_DIM_NAME: int(bias),
                },
            }
            test_cases.append(test_case)

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={},
                monotonic_step_size=1,
                custom_generators=[
                    RangeManualGeneratorStrategy(test_cases=test_cases),
                ],
            ),
        )

    @pytest.mark.fast
    @range_test_config(moe_tkg_io_dtype_config())
    def test_moe_tkg_io_dtype(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        """Test MoE TKG kernel with different input and output dtypes (MX only, TRN3 only)."""
        mlp_config = range_test_options.tensors[MOE_TKG_CONFIG]

        lnc_count = mlp_config[VNC_DEGREE_DIM_NAME]
        quant_type = mlp_config[QUANT_TYPE_DIM_NAME]

        # MX quantization is only supported on TRN3
        if quant_type is QuantizationType.MX and platform_target is not Platforms.TRN3:
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=platform_target)

        self.run_moe_tkg_test(
            test_manager=test_manager,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            compiler_args=compiler_args,
            quant_dtype=mlp_config[QUANT_DTYPE_DIM_NAME],
            quant_type=mlp_config[QUANT_TYPE_DIM_NAME],
            in_dtype=mlp_config[IN_DTYPE_DIM_NAME],
            out_dtype=mlp_config[OUT_DTYPE_DIM_NAME],
            expert_affinities_scaling_mode=ExpertAffinityScaleMode(mlp_config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]),
            is_all_expert=bool(mlp_config[IS_ALL_EXPERT_DIM_NAME]),
            clamp=bool(mlp_config[CLAMP_DIM_NAME]),
            bias=bool(mlp_config[BIAS_DIM_NAME]),
            collector=collector,
        )
