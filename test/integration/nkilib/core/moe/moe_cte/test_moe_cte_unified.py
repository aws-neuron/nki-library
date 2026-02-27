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

"""Unified tests for moe_cte entry point covering all implementations."""

import math
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte import (
    ACT_FN_DIM_NAME,
    BIAS_DIM_NAME,
    BLOCK_SIZE_DIM_NAME,
    BWMM_CONFIG,
    BWMM_FUNC_DIM_NAME,
    DTYPE_DIM_NAME,
    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME,
    EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME,
    EXPERT_DIM_NAME,
    GATE_CLAMP_LOWER_DIM_NAME,
    GATE_CLAMP_UPPER_DIM_NAME,
    HIDDEN_DIM_NAME,
    INTERMEDIATE_DIM_NAME,
    QUANTIZE_DIM_NAME,
    SKIP_DIM_NAME,
    TOKENS_DIM_NAME,
    TOP_K_DIM_NAME,
    TRAINING_DIM_NAME,
    UP_CLAMP_LOWER_DIM_NAME,
    UP_CLAMP_UPPER_DIM_NAME,
    VNC_DEGREE_DIM_NAME,
    BWMMFunc,
    dtype2dtype_range,
    generate_token_position_to_id_and_experts,
    generate_token_position_to_id_and_experts_dropping,
    get_n_blocks,
    golden_bwmm,
    map_skip_mode,
    quantize_strategy2scale_shapes,
)
from test.integration.nkilib.core.moe.moe_cte.test_utils import (
    build_moe_bwmm_mx_cte,
    golden_moe_bwmm_mx_cte,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.mx_utils import is_mx_quantize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe.moe_cte import (
    MoECTEImplementation,
    MoECTESpec,
    QuantizationConfig,
    ShardOnBlockConfig,
    ShardOnIConfig,
    moe_cte,
)
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

# Additional dimension constants for MX (MXFP4/MXFP8) tests
WEIGHT_DTYPE_DIM_NAME = "wdt"
IS_DYNAMIC_DIM_NAME = "dyn"
IMPL_DIM_NAME = "impl"


def bwmm_func_to_implementation(bwmm_func: BWMMFunc) -> MoECTEImplementation:
    """Map BWMMFunc enum to MoECTEImplementation."""
    mapping = {
        BWMMFunc.SHARD_ON_BLOCK: MoECTEImplementation.shard_on_block,
        BWMMFunc.SHARD_ON_INTERMEDIATE: MoECTEImplementation.shard_on_i,
        BWMMFunc.SHARD_ON_INTERMEDIATE_HW: MoECTEImplementation.shard_on_i_hybrid,
        BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING: MoECTEImplementation.shard_on_i_dropping,
    }
    if bwmm_func not in mapping:
        raise ValueError(f"No MoECTEImplementation mapping for {bwmm_func}")
    return mapping[bwmm_func]


def build_moe_cte_unified_inputs(
    impl: MoECTEImplementation,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    top_k: int,
    dtype,
    dma_skip: SkipMode,
    bias: bool,
    quantize,
    quantize_strategy: int,
    vnc_degree: int,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    activation_function: ActFnType,
    spec: MoECTESpec,
    weight_dtype=None,
    is_dynamic: bool = False,
    gate_clamp_lower_limit=None,
    gate_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
):
    """Build input tensors for unified moe_cte kernel."""
    # Check if MX quantization (MXFP4/MXFP8)
    if is_mx_quantize(weight_dtype):
        skip_mode = (1 if dma_skip.skip_token else 0) + (2 if dma_skip.skip_weight else 0)
        inputs = build_moe_bwmm_mx_cte(
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
            activation_function=activation_function,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            is_dynamic=is_dynamic,
            vnc_degree=vnc_degree,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )
        inputs['spec'] = spec
        return inputs

    # Non-MX path
    is_dropping = impl == MoECTEImplementation.shard_on_i_dropping
    is_block_parallel = impl == MoECTEImplementation.shard_on_block
    n_block_per_iter = 1

    if is_dropping:
        N = expert
        token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts_dropping(
            T=tokens, E_Local=expert, B=block_size, rtype=1, topK=top_k
        )
        expert_masks = np.ones([tokens, expert], dtype=dtype)
        conditions = None
    else:
        N = get_n_blocks(
            tokens, top_k, expert, block_size, n_block_per_iter=vnc_degree if is_block_parallel else n_block_per_iter
        )
        expert_masks, token_position_to_id, block_to_expert, conditions = generate_token_position_to_id_and_experts(
            T=tokens,
            TOPK=top_k,
            E=expert,
            B=block_size,
            dma_skip=dma_skip,
            N=N,
            vnc_degree=vnc_degree,
            n_block_per_iter=n_block_per_iter,
            is_block_parallel=is_block_parallel,
            quantize=quantize,
        )

    np.random.seed(0)

    if dma_skip.skip_token:
        expert_affinities_masked = (np.random.random_sample([tokens, expert]) * expert_masks).astype(dtype)
        hidden_states = np.random.random_sample([tokens, hidden]).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([tokens + 1, expert]).astype(dtype)
        expert_affinities_masked[:tokens] = expert_affinities_masked[:tokens] * expert_masks
        expert_affinities_masked[tokens] = 0
        hidden_states = np.random.random_sample([tokens + 1, hidden]).astype(dtype)
        hidden_states[tokens, ...] = 0

    gate_and_up_proj_bias = None
    down_proj_bias = None
    if bias:
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[expert, hidden]).astype(dtype)
        gate_and_up_proj_bias = np.random.uniform(-1, 1, size=[expert, 2, intermediate]).astype(dtype)

    gate_up_proj_scale = None
    down_proj_scale = None
    I_TP_padded = math.ceil(intermediate / 16) * 16

    if quantize:
        quantize_dt = nl.int8 if quantize == 1 else quantize
        dtype_min, dtype_max = dtype2dtype_range[quantize_dt]
        down_proj_weight = np.random.randint(dtype_min, dtype_max, size=[expert, I_TP_padded, hidden]).astype(
            quantize_dt
        )
        gate_up_proj_weight = np.random.randint(dtype_min, dtype_max, size=[expert, hidden, 2, intermediate]).astype(
            quantize_dt
        )
        gup_scale_shape, down_scale_shape = quantize_strategy2scale_shapes(
            quantize_strategy, expert, intermediate, hidden
        )
        scale_dtype = dtype if quantize_strategy == 2 else nl.float32
        gate_up_proj_scale = np.random.uniform(-0.1, 0.1, size=gup_scale_shape).astype(scale_dtype)
        down_proj_scale = np.random.uniform(-0.1, 0.1, size=down_scale_shape).astype(scale_dtype)
    else:
        down_proj_weight = np.random.uniform(-0.1, 0.1, size=[expert, I_TP_padded, hidden]).astype(dtype)
        gate_up_proj_weight = np.random.uniform(-0.1, 0.1, size=[expert, hidden, 2, intermediate]).astype(dtype)

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
        "gate_up_proj_weight": gate_up_proj_weight,
        "down_proj_weight": down_proj_weight,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert,
        "block_size": block_size,
        "spec": spec,
        "activation_function": activation_function,
        "skip_dma": dma_skip,
        "compute_dtype": dtype,
        "is_tensor_update_accumulating": top_k != 1,
        "expert_affinities_scaling_mode": expert_affinities_scaling_mode,
    }

    if is_dynamic and conditions is not None:
        inputs["conditions"] = conditions
    if bias:
        inputs["gate_and_up_proj_bias"] = gate_and_up_proj_bias
        inputs["down_proj_bias"] = down_proj_bias
    if quantize:
        inputs["quantization_config"] = QuantizationConfig(
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
        )
    if gate_clamp_lower_limit is not None:
        inputs["gate_clamp_lower_limit"] = gate_clamp_lower_limit
    if gate_clamp_upper_limit is not None:
        inputs["gate_clamp_upper_limit"] = gate_clamp_upper_limit
    if up_clamp_lower_limit is not None:
        inputs["up_clamp_lower_limit"] = up_clamp_lower_limit
    if up_clamp_upper_limit is not None:
        inputs["up_clamp_upper_limit"] = up_clamp_upper_limit
    # breakpoint()
    return inputs


# fmt: off
# =============================================================================
# UNIFIED TEST PARAMETERS
# =============================================================================
# Format: [bwmm_func, hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, bias, training, quantize, act_fn, expert_affinities_scaling_mode, gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I]
# For MX (MXFP4/MXFP8): [bwmm_func, hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, bias, training, quantize, act_fn, expert_affinities_scaling_mode, gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, weight_dtype, is_dynamic, vnc]
#
# Fields:
#   bwmm_func: BWMMFunc enum (non-MX) or None (MX)
#   hidden: Hidden dimension (H)
#   tokens: Number of tokens (T)
#   expert: Number of experts (E)
#   block_size: Block size (B)
#   top_k: TopK value
#   intermediate: Intermediate dimension (I_TP)
#   dtype: Compute dtype (e.g., nl.bfloat16)
#   skip: Skip mode (0=none, 1=skip_token, 2=skip_weight, 3=both)
#   bias: Whether to use bias
#   training: Training mode (checkpoint activations)
#   quantize: Quantization dtype (None, nl.int8, nl.float8_e4m3)
#   act_fn: Activation function type
#   expert_affinities_scaling_mode: Affinity scaling mode
#   gate_cl_upper/lower: Gate clamp limits
#   up_cl_upper/lower: Up projection clamp limits
#   expert_affinity_multiply_on_I: Whether to multiply affinity on I dimension
#   weight_dtype: Weight dtype for MX (MXFP4/MXFP8 only)
#   is_dynamic: Dynamic blocks mode (MX only)
#   vnc: LNC degree (MX only, typically 2)

moe_cte_test_params = [
# =============================================================================
# NON-MX TESTS
# =============================================================================
# SHARD_ON_INTERMEDIATE_HW tests
[BWMMFunc.SHARD_ON_INTERMEDIATE_HW, 3072, 1024, 8, 512, 4, 2048, nl.bfloat16, 0, False, False, None, ActFnType.SiLU, ExpertAffinityScaleMode.NO_SCALE, None, None, None, None, False],
# SHARD_ON_INTERMEDIATE tests
[BWMMFunc.SHARD_ON_INTERMEDIATE, 3072, 1024, 8, 512, 4, 2048, nl.bfloat16, 0, True, False, None, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, None, None, None, None, False],
# SHARD_ON_BLOCK tests
[BWMMFunc.SHARD_ON_BLOCK, 3072, 1024, 8, 512, 4, 384, nl.bfloat16, 1, True, False, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7, None, 8, -9, False],
# Dropping kernel tests
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536, 8192, 2, 4096, 2, 6144, nl.bfloat16, 0, False, True, None, ActFnType.SiLU, ExpertAffinityScaleMode.POST_SCALE, None, None, None, None, True],
]

# =============================================================================
# MX (MXFP4/MXFP8) SHARD-ON-BLOCK TEST PARAMETERS
# =============================================================================
# Same format as non-MX: [bwmm_func, hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, bias, training, quantize, act_fn, expert_affinities_scaling_mode, gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, weight_dtype, is_dynamic, vnc]
moe_cte_mxfp4_block_params = [
[None, 3072, 1024, 128, 256, 4, 384, nl.bfloat16, 1, True, False, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0, None, 7.0, -7.0, False, nl.float4_e2m1fn_x4, False, 2],

]

# =============================================================================
# MX (MXFP4/MXFP8) SHARD-ON-INTERMEDIATE TEST PARAMETERS
# =============================================================================
# Same format as non-MX: [bwmm_func, hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, bias, training, quantize, act_fn, expert_affinities_scaling_mode, gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, weight_dtype, is_dynamic, vnc]
moe_cte_mxfp4_shard_I_params = [
# [None, 7168, 1024, 8, 256, 8, 1024, nl.bfloat16, 1, True, False, None, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0, None, 7.0, -7.0, False, nl.float4_e2m1fn_x4, True, 2],
]
# fmt: on


@pytest_test_metadata(
    name="MoE CTE Unified Entry Point",
    pytest_marks=["moe", "cte", "unified"],
)
@final
class TestMoeCTEUnified:
    """Unified tests for moe_cte() entry point covering all implementations."""

    def run_unified_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        collector: IMetricsCollector,
    ):
        config = test_options.tensors[BWMM_CONFIG]

        # Extract common parameters
        tokens = config[TOKENS_DIM_NAME]
        hidden = config[HIDDEN_DIM_NAME]
        intermediate = config[INTERMEDIATE_DIM_NAME]
        expert = config[EXPERT_DIM_NAME]
        block_size = config[BLOCK_SIZE_DIM_NAME]
        top_k = config[TOP_K_DIM_NAME]
        act_fn = config[ACT_FN_DIM_NAME]
        expert_affinities_scaling_mode = config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]
        dtype = config[DTYPE_DIM_NAME]
        skip = config[SKIP_DIM_NAME]
        bias = config[BIAS_DIM_NAME]
        gate_clamp_upper = config[GATE_CLAMP_UPPER_DIM_NAME]
        gate_clamp_lower = config[GATE_CLAMP_LOWER_DIM_NAME]
        up_clamp_upper = config[UP_CLAMP_UPPER_DIM_NAME]
        up_clamp_lower = config[UP_CLAMP_LOWER_DIM_NAME]

        # Get implementation type
        impl = config[IMPL_DIM_NAME]
        weight_dtype = config.get(WEIGHT_DTYPE_DIM_NAME)
        training = config.get(TRAINING_DIM_NAME, False)
        quantize = config.get(QUANTIZE_DIM_NAME)
        expert_affinity_multiply_on_I = config.get(EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME, False)

        is_mx = is_mx_quantize(weight_dtype)

        # Determine is_dynamic based on test type
        if is_mx:
            is_dynamic = config.get(IS_DYNAMIC_DIM_NAME, False)
        else:
            # For non-MX, get is_dynamic from BWMMFunc
            bwmm_func_enum = config.get(BWMM_FUNC_DIM_NAME)
            _, _, is_dynamic = bwmm_func_enum.get_bwmm_func()

        is_block_parallel = impl in [MoECTEImplementation.shard_on_block, MoECTEImplementation.shard_on_block_mx]
        is_dropping = impl == MoECTEImplementation.shard_on_i_dropping

        dma_skip = map_skip_mode(skip)
        quantize_strategy = 6 if quantize else 0

        # Build spec with implementation-specific config
        if impl == MoECTEImplementation.shard_on_block:
            spec = MoECTESpec(
                implementation=impl,
                shard_on_block=ShardOnBlockConfig(),
            )
        elif impl in [MoECTEImplementation.shard_on_i, MoECTEImplementation.shard_on_i_dropping]:
            spec = MoECTESpec(
                implementation=impl,
                shard_on_I=ShardOnIConfig(
                    checkpoint_activation=training,
                    expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                ),
            )
        elif impl == MoECTEImplementation.shard_on_i_hybrid:
            spec = MoECTESpec(
                implementation=impl,
                shard_on_I=ShardOnIConfig(),
            )
        elif impl == MoECTEImplementation.shard_on_block_mx:
            spec = MoECTESpec(
                implementation=impl,
                shard_on_block=ShardOnBlockConfig(),
            )
        else:
            spec = MoECTESpec(implementation=impl)

        with assert_negative_test_case(False):
            kernel_input = build_moe_cte_unified_inputs(
                impl=impl,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias=bias,
                quantize=quantize,
                quantize_strategy=quantize_strategy,
                vnc_degree=lnc_degree,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                activation_function=act_fn,
                spec=spec,
                weight_dtype=weight_dtype,
                is_dynamic=is_dynamic,
                gate_clamp_lower_limit=gate_clamp_lower,
                gate_clamp_upper_limit=gate_clamp_upper,
                up_clamp_lower_limit=up_clamp_lower,
                up_clamp_upper_limit=up_clamp_upper,
            )

            # Create golden generator
            if is_mx:

                def create_lazy_golden():
                    return golden_moe_bwmm_mx_cte(
                        kernel_input=kernel_input,
                        dtype=dtype,
                        lnc_degree=lnc_degree,
                    )
            else:

                def create_lazy_golden():
                    return golden_bwmm(
                        inp_np=kernel_input,
                        lnc_degree=lnc_degree,
                        tokens=tokens,
                        hidden=hidden,
                        intermediate=intermediate,
                        expert=expert,
                        block_size=block_size,
                        dtype=dtype,
                        dma_skip=dma_skip,
                        bias=bias,
                        quantize=quantize,
                        quantize_strategy=quantize_strategy,
                        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                        activation_function=act_fn,
                        checkpoint_activation=training or is_dropping,
                        is_block_parallel=is_block_parallel,
                        top_k=top_k,
                        gate_clamp_lower_limit=gate_clamp_lower,
                        gate_clamp_upper_limit=gate_clamp_upper,
                        up_clamp_lower_limit=up_clamp_lower,
                        up_clamp_upper_limit=up_clamp_upper,
                        is_shard_block=(impl == MoECTEImplementation.shard_on_block),
                        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                        is_dropping=is_dropping,
                    )

            # Determine output shape
            T_out = tokens if dma_skip.skip_token else tokens + 1
            separate_outputs = is_block_parallel and top_k > 1
            is_accumulating = top_k != 1

            if is_mx:
                # MX shard-on-block with accumulation needs lnc_degree dimension
                if impl == MoECTEImplementation.shard_on_block_mx and is_accumulating:
                    output_shape = (lnc_degree, T_out, hidden)
                else:
                    output_shape = (T_out, hidden)
            elif separate_outputs:
                if impl == MoECTEImplementation.shard_on_block:
                    output_shape = (T_out, lnc_degree, hidden)
                else:
                    output_shape = (lnc_degree, T_out, hidden)
            else:
                output_shape = (T_out, hidden)

            # Build output placeholder
            if is_mx:
                output_placeholder = {"output": np.zeros(output_shape, dtype=dtype)}
            elif training or is_dropping:
                if is_dropping:
                    N = expert
                else:
                    N = get_n_blocks(
                        tokens, top_k, expert, block_size, n_block_per_iter=lnc_degree if is_block_parallel else 1
                    )
                if expert_affinity_multiply_on_I:
                    output_placeholder = {
                        "output": np.zeros(output_shape, dtype=dtype),
                        "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
                    }
                else:
                    output_placeholder = {
                        "output": np.zeros(output_shape, dtype=dtype),
                        "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
                        "down_activations": np.zeros((N, block_size, hidden), dtype=dtype),
                    }
            else:
                output_placeholder = {"output": np.zeros(output_shape, dtype=dtype)}

            # Set tolerances
            if is_mx or quantize:
                rtol, atol = 5e-2, 1e-5
            else:
                rtol, atol = 2e-2, 1e-5

            validation_args = ValidationArgs(
                golden_output=LazyGoldenGenerator(
                    lazy_golden_generator=create_lazy_golden,
                    output_ndarray=output_placeholder,
                ),
                relative_accuracy=rtol,
                absolute_accuracy=atol,
            )

            # Remove internal data before passing to kernel
            kernel_input_for_kernel = {k: v for k, v in kernel_input.items() if k != '_internal'}

            test_manager.execute(
                KernelArgs(
                    kernel_func=moe_cte,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input_for_kernel,
                    validation_args=validation_args,
                )
            )

    @staticmethod
    def unified_config():
        test_cases = []

        # Add non-MX test cases
        for params in moe_cte_test_params:
            (
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
            ) = params

            impl = bwmm_func_to_implementation(bwmm_func)
            test_cases.append(
                {
                    BWMM_CONFIG: {
                        IMPL_DIM_NAME: impl,
                        BWMM_FUNC_DIM_NAME: bwmm_func,
                        VNC_DEGREE_DIM_NAME: 2,
                        TOKENS_DIM_NAME: tokens,
                        HIDDEN_DIM_NAME: hidden,
                        INTERMEDIATE_DIM_NAME: intermediate,
                        EXPERT_DIM_NAME: expert,
                        BLOCK_SIZE_DIM_NAME: block_size,
                        TOP_K_DIM_NAME: top_k,
                        ACT_FN_DIM_NAME: act_fn,
                        EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode,
                        DTYPE_DIM_NAME: dtype,
                        SKIP_DIM_NAME: skip,
                        BIAS_DIM_NAME: bias,
                        TRAINING_DIM_NAME: training,
                        QUANTIZE_DIM_NAME: quantize,
                        GATE_CLAMP_UPPER_DIM_NAME: gate_cl_upper,
                        GATE_CLAMP_LOWER_DIM_NAME: gate_cl_lower,
                        UP_CLAMP_UPPER_DIM_NAME: up_cl_upper,
                        UP_CLAMP_LOWER_DIM_NAME: up_cl_lower,
                        EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME: expert_affinity_multiply_on_I,
                        WEIGHT_DTYPE_DIM_NAME: None,
                        IS_DYNAMIC_DIM_NAME: False,
                    },
                }
            )

        # Add MX shard-on-block test cases
        for params in moe_cte_mxfp4_block_params:
            (
                _,
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
                weight_dtype,
                is_dynamic,
                vnc,
            ) = params

            impl = MoECTEImplementation.shard_on_block_mx
            test_cases.append(
                {
                    BWMM_CONFIG: {
                        IMPL_DIM_NAME: impl,
                        BWMM_FUNC_DIM_NAME: None,
                        VNC_DEGREE_DIM_NAME: vnc,
                        TOKENS_DIM_NAME: tokens,
                        HIDDEN_DIM_NAME: hidden,
                        INTERMEDIATE_DIM_NAME: intermediate,
                        EXPERT_DIM_NAME: expert,
                        BLOCK_SIZE_DIM_NAME: block_size,
                        TOP_K_DIM_NAME: top_k,
                        ACT_FN_DIM_NAME: act_fn,
                        EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode,
                        DTYPE_DIM_NAME: dtype,
                        WEIGHT_DTYPE_DIM_NAME: weight_dtype,
                        SKIP_DIM_NAME: skip,
                        BIAS_DIM_NAME: bias,
                        IS_DYNAMIC_DIM_NAME: is_dynamic,
                        GATE_CLAMP_UPPER_DIM_NAME: gate_cl_upper,
                        GATE_CLAMP_LOWER_DIM_NAME: gate_cl_lower,
                        UP_CLAMP_UPPER_DIM_NAME: up_cl_upper,
                        UP_CLAMP_LOWER_DIM_NAME: up_cl_lower,
                        TRAINING_DIM_NAME: training,
                        QUANTIZE_DIM_NAME: quantize,
                        EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME: expert_affinity_multiply_on_I,
                    },
                }
            )

        # Add MX shard-on-I test cases
        for params in moe_cte_mxfp4_shard_I_params:
            (
                _,
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
                weight_dtype,
                is_dynamic,
                vnc,
            ) = params

            impl = MoECTEImplementation.shard_on_i_mx_hybrid if is_dynamic else MoECTEImplementation.shard_on_i_mx
            test_cases.append(
                {
                    BWMM_CONFIG: {
                        IMPL_DIM_NAME: impl,
                        BWMM_FUNC_DIM_NAME: None,
                        VNC_DEGREE_DIM_NAME: vnc,
                        TOKENS_DIM_NAME: tokens,
                        HIDDEN_DIM_NAME: hidden,
                        INTERMEDIATE_DIM_NAME: intermediate,
                        EXPERT_DIM_NAME: expert,
                        BLOCK_SIZE_DIM_NAME: block_size,
                        TOP_K_DIM_NAME: top_k,
                        ACT_FN_DIM_NAME: act_fn,
                        EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode,
                        DTYPE_DIM_NAME: dtype,
                        WEIGHT_DTYPE_DIM_NAME: weight_dtype,
                        SKIP_DIM_NAME: skip,
                        BIAS_DIM_NAME: bias,
                        IS_DYNAMIC_DIM_NAME: is_dynamic,
                        GATE_CLAMP_UPPER_DIM_NAME: gate_cl_upper,
                        GATE_CLAMP_LOWER_DIM_NAME: gate_cl_lower,
                        UP_CLAMP_UPPER_DIM_NAME: up_cl_upper,
                        UP_CLAMP_LOWER_DIM_NAME: up_cl_lower,
                        TRAINING_DIM_NAME: training,
                        QUANTIZE_DIM_NAME: quantize,
                        EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME: expert_affinity_multiply_on_I,
                    },
                }
            )

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
    @range_test_config(unified_config())
    def test_moe_cte_unified(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        platform_target: Platforms,
    ):
        """Unified test for moe_cte() entry point."""
        config = range_test_options.tensors[BWMM_CONFIG]
        lnc_count = config[VNC_DEGREE_DIM_NAME]
        impl = config[IMPL_DIM_NAME]
        weight_dtype = config.get(WEIGHT_DTYPE_DIM_NAME)

        # Skip MX tests on non-TRN3 platforms
        is_mx = is_mx_quantize(weight_dtype)
        if is_mx and platform_target is not Platforms.TRN3:
            pytest.skip("MX (MXFP4/MXFP8) is only supported on TRN3.")

        # Set platform target based on test type
        target = Platforms.TRN3 if is_mx else platform_target
        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=target)

        self.run_unified_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            collector=collector,
        )
