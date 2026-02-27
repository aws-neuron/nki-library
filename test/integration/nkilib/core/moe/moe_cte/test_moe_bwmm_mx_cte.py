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


from test.integration.nkilib.core.moe.moe_cte.test_moe_cte import (
    map_skip_mode,
)
from test.integration.nkilib.core.moe.moe_cte.test_utils import (
    build_moe_bwmm_mx_cte,
    golden_moe_bwmm_mx_cte,
    order_kernel_input,
)
from test.integration.nkilib.utils.dtype_helper import dt
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, Platforms, ValidationArgs
from test.utils.metrics_collector import MetricsCollector
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
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)

# Configuration key names
MOE_BWMM_MX_CONFIG = "cfg"
VNC_DEGREE_DIM_NAME = "vnc"
TOKENS_DIM_NAME = "tok"
HIDDEN_DIM_NAME = "hid"
INTERMEDIATE_DIM_NAME = "int"
EXPERT_DIM_NAME = "exp"
BLOCK_SIZE_DIM_NAME = "bs"
TOP_K_DIM_NAME = "k"
ACT_FN_DIM_NAME = "act"
EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME = "easm"
DTYPE_DIM_NAME = "dt"
WEIGHT_DTYPE_DIM_NAME = "wdt"
SKIP_MODE_DIM_NAME = "sk"
BIAS_DIM_NAME = "bi"
IS_DYNAMIC_DIM_NAME = "dyn"
GATE_CLAMP_UPPER_DIM_NAME = "gcu"
GATE_CLAMP_LOWER_DIM_NAME = "gcl"
UP_CLAMP_UPPER_DIM_NAME = "ucu"
UP_CLAMP_LOWER_DIM_NAME = "ucl"

# fmt: off

moe_bwmm_mx_cte_kernel_params = [
    # vnc, H, T, I_TP, E, B, Top_K, Act_Fn, Exp_Affn_Scaling_Mode, dtype, weight_dtype, skip_mode, bias, dynamic_blocks, gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower
    # MXFP4 test cases
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0],
    [2, 3072, 1024, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0],

    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0],
    
    # MXFP8 e4m3fn test cases
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 0, True, True, 7.0, None, 7.0, -7.0],
    
    # MXFP8 e5m2 test cases
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 0, True, True, 7.0, None, 7.0, -7.0],
]

# ═══════════════════════════════════════════════════════════════════════════════
# SHARD-ON-INTERMEDIATE TEST PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

moe_bwmm_mx_shard_I_kernel_params = [
    # vnc, H, T, I_TP, E, B, Top_K, Act_Fn, Exp_Affn_Scaling_Mode, dtype, weight_dtype, skip_mode, bias, dynamic_blocks, gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower
    
    # Basic tests with different block sizes
    [2, 3072, 1024, 2048, 8, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 3072, 1024, 2048, 8, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0],
    [2, 7168, 1024, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 7168, 1024, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0],
]

# fmt: on
@pytest_test_metadata(
    name="MoE BWMM MX CTE",
    pytest_marks=["moe", "cte", "mx"],
)
@final
class TestMoeBwmmMxCteKernel:
    """Test class for MoE BWMM MX CTE kernel."""

    def run_moe_bwmm_mx_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        dtype,
        weight_dtype,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        bias: bool,
        is_dynamic: bool,
        collector: MetricsCollector,
    ):
        """
        Run a single MoE BWMM MX CTE test case.

        Args:
            test_manager: Orchestrator instance for test execution
            compiler_args: Compiler configuration
            test_options: Test case parameters
            lnc_degree: LNC sharding degree
            dtype: Data type for activations
            weight_dtype: Data type for weights (e.g., nl.float4_e2m1fn_x4)
            expert_affinities_scaling_mode: Expert affinity scaling mode
            bias: Whether to include bias tensors
            is_dynamic: Whether to use dynamic loop
            collector: Metrics collector
        """
        is_negative_test_case = False

        config = test_options.tensors[MOE_BWMM_MX_CONFIG]

        tokens = config[TOKENS_DIM_NAME]
        hidden = config[HIDDEN_DIM_NAME]
        intermediate = config[INTERMEDIATE_DIM_NAME]
        expert = config[EXPERT_DIM_NAME]
        block_size = config[BLOCK_SIZE_DIM_NAME]
        top_k = config[TOP_K_DIM_NAME]
        act_fn = config[ACT_FN_DIM_NAME]
        skip_mode = config[SKIP_MODE_DIM_NAME]
        gate_clamp_upper = config.get(GATE_CLAMP_UPPER_DIM_NAME)
        gate_clamp_lower = config.get(GATE_CLAMP_LOWER_DIM_NAME)
        up_clamp_upper = config.get(UP_CLAMP_UPPER_DIM_NAME)
        up_clamp_lower = config.get(UP_CLAMP_LOWER_DIM_NAME)

        # Check for negative test case: hidden for each core must be divisible by 128
        if hidden // lnc_degree % 128 != 0:
            is_negative_test_case = True

        with assert_negative_test_case(is_negative_test_case):
            # Build kernel inputs
            kernel_input = build_moe_bwmm_mx_cte(
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
                vnc_degree=lnc_degree,
                n_dynamic_blocks=55,
                gate_clamp_upper_limit=gate_clamp_upper,
                gate_clamp_lower_limit=gate_clamp_lower,
                up_clamp_upper_limit=up_clamp_upper,
                up_clamp_lower_limit=up_clamp_lower,
            )

            # Create lazy golden generator to defer computation until needed
            def create_lazy_golden():
                return golden_moe_bwmm_mx_cte(
                    kernel_input=kernel_input,
                    dtype=dtype,
                    lnc_degree=lnc_degree,
                )

            # Create output placeholder with correct shape and dtype based on skip mode and accumulation
            numpy_dtype = dt.finfo(dtype).dtype
            dma_skip = map_skip_mode(skip_mode)
            is_accumulating = top_k != 1

            # Output shape depends on separate_outputs (when accumulating) and skip_token
            if is_accumulating:
                out_T = tokens if dma_skip.skip_token else tokens + 1
                output_placeholder = {"output": np.zeros((lnc_degree, out_T, hidden), dtype=numpy_dtype)}
            else:
                out_T = tokens if dma_skip.skip_token else tokens + 1
                output_placeholder = {"output": np.zeros((out_T, hidden), dtype=numpy_dtype)}

            # Create validation args with tolerances
            validation_args = ValidationArgs(
                golden_output=LazyGoldenGenerator(
                    output_ndarray=output_placeholder,
                    lazy_golden_generator=create_lazy_golden,
                ),
                relative_accuracy=5e-2,
                absolute_accuracy=1e-5,
            )

            # Import kernel function
            from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx import bwmm_shard_on_block_mx

            # Remove _internal data before passing to kernel
            kernel_input_for_kernel = order_kernel_input(kernel_input, variant='shard_on_block_mx')

            # Execute test
            test_manager.execute(
                KernelArgs(
                    kernel_func=bwmm_shard_on_block_mx,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input_for_kernel,
                    validation_args=validation_args,
                )
            )

    @staticmethod
    def moe_bwmm_mx_cte_unit_config():
        """
        Generate unit test configuration for MoE BWMM MX CTE kernel.

        Returns:
            RangeTestConfig with test cases matching old framework parameters
        """
        test_cases = []

        for test_params in moe_bwmm_mx_cte_kernel_params:
            # Parameter order: [vnc_degree, H, T, I_TP, E, B, top_k, act_fn, expert_affinities_scaling_mode,
            #                   dtype, weight_dtype, skip_mode, bias, is_dynamic, gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower]
            (
                vnc_degree,
                hidden,
                tokens,
                intermediate,
                expert,
                block_size,
                top_k,
                act_fn,
                expert_affinities_scaling_mode,
                dtype,
                weight_dtype,
                skip_mode,
                bias,
                is_dynamic,
                gate_clamp_upper,
                gate_clamp_lower,
                up_clamp_upper,
                up_clamp_lower,
            ) = test_params

            test_case = {
                MOE_BWMM_MX_CONFIG: {
                    VNC_DEGREE_DIM_NAME: vnc_degree,
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    BLOCK_SIZE_DIM_NAME: block_size,
                    TOP_K_DIM_NAME: top_k,
                    ACT_FN_DIM_NAME: act_fn,
                    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode.value,
                    DTYPE_DIM_NAME: dtype,
                    WEIGHT_DTYPE_DIM_NAME: weight_dtype,
                    SKIP_MODE_DIM_NAME: skip_mode,
                    BIAS_DIM_NAME: int(bias),
                    IS_DYNAMIC_DIM_NAME: int(is_dynamic),
                    GATE_CLAMP_UPPER_DIM_NAME: gate_clamp_upper,
                    GATE_CLAMP_LOWER_DIM_NAME: gate_clamp_lower,
                    UP_CLAMP_UPPER_DIM_NAME: up_clamp_upper,
                    UP_CLAMP_LOWER_DIM_NAME: up_clamp_lower,
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
    @range_test_config(moe_bwmm_mx_cte_unit_config())
    def test_moe_bwmm_mx_cte_unit(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """
        Unit test entry point for MoE BWMM MX CTE kernel.

        This test validates the blockwise matrix multiplication kernel with
        MX quantization against golden outputs computed in NumPy.
        """
        config = range_test_options.tensors[MOE_BWMM_MX_CONFIG]

        if platform_target is not Platforms.TRN3:
            pytest.skip("MX quantization is only supported on TRN3.")

        lnc_count = config[VNC_DEGREE_DIM_NAME]
        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=platform_target)

        self.run_moe_bwmm_mx_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            dtype=config[DTYPE_DIM_NAME],
            weight_dtype=config[WEIGHT_DTYPE_DIM_NAME],
            expert_affinities_scaling_mode=ExpertAffinityScaleMode(config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]),
            bias=bool(config[BIAS_DIM_NAME]),
            is_dynamic=bool(config[IS_DYNAMIC_DIM_NAME]),
            collector=collector,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEW TEST CLASS: SHARD-ON-INTERMEDIATE MXFP4
# ═══════════════════════════════════════════════════════════════════════════════


@pytest_test_metadata(
    name="MoE BWMM MX Shard-on-I",
    pytest_marks=["moe", "cte", "mx", "shard_on_I"],
)
@final
class TestMoeBwmmMxShardIKernel:
    """
    Test class for MoE BWMM MXFP4/MXFP8 kernel with intermediate dimension sharding.

    This kernel shards the intermediate dimension (I_TP) across cores and uses
    send/recv for cross-core reduction.
    """

    def run_moe_bwmm_mx_shard_I_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        dtype,
        weight_dtype,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        bias: bool,
        is_dynamic,
        collector: MetricsCollector,
    ):
        """
        Run a single MoE BWMM MXFP4 shard-on-I test case.

        Args:
            test_manager: Orchestrator instance for test execution
            compiler_args: Compiler configuration
            test_options: Test case parameters
            lnc_degree: LNC sharding degree (must be 2)
            dtype: Data type for activations
            weight_dtype: Data type for weights
            expert_affinities_scaling_mode: Expert affinity scaling mode
            bias: Whether to include bias tensors
            collector: Metrics collector
        """
        is_negative_test_case = False

        config = test_options.tensors[MOE_BWMM_MX_CONFIG]

        tokens = config[TOKENS_DIM_NAME]
        hidden = config[HIDDEN_DIM_NAME]
        intermediate = config[INTERMEDIATE_DIM_NAME]
        expert = config[EXPERT_DIM_NAME]
        block_size = config[BLOCK_SIZE_DIM_NAME]
        top_k = config[TOP_K_DIM_NAME]
        act_fn = config[ACT_FN_DIM_NAME]
        skip_mode = config[SKIP_MODE_DIM_NAME]
        gate_clamp_upper = config.get(GATE_CLAMP_UPPER_DIM_NAME)
        gate_clamp_lower = config.get(GATE_CLAMP_LOWER_DIM_NAME)
        up_clamp_upper = config.get(UP_CLAMP_UPPER_DIM_NAME)
        up_clamp_lower = config.get(UP_CLAMP_LOWER_DIM_NAME)

        # Validation checks for shard-on-I kernel
        # H must be multiple of 512
        if hidden % 512 != 0:
            is_negative_test_case = True
        # I_TP must be multiple of 512
        if intermediate % 512 != 0:
            is_negative_test_case = True
        # Block size must be multiple of 256
        if block_size % 256 != 0:
            is_negative_test_case = True
        # Must use 2 shards
        if lnc_degree != 2:
            is_negative_test_case = True

        with assert_negative_test_case(is_negative_test_case):
            # Build kernel inputs - reuse the helper with shard_on_I flag
            kernel_input = build_moe_bwmm_mx_cte(
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
                vnc_degree=lnc_degree,
                is_dynamic=is_dynamic,
                gate_clamp_upper_limit=gate_clamp_upper,
                gate_clamp_lower_limit=gate_clamp_lower,
                up_clamp_upper_limit=up_clamp_upper,
                up_clamp_lower_limit=up_clamp_lower,
                is_shard_on_I=True,
            )

            # Create lazy golden generator to defer computation until needed
            def create_lazy_golden():
                return golden_moe_bwmm_mx_cte(
                    kernel_input=kernel_input,
                    dtype=dtype,
                    lnc_degree=lnc_degree,
                    is_shard_on_I=True,
                )

            # # Compute golden output
            dma_skip = map_skip_mode(skip_mode)
            out_T = tokens if dma_skip.skip_token else tokens + 1
            output_placeholder = {"output": np.zeros((out_T, hidden), dtype=dtype)}
            # Create validation args with tolerances
            validation_args = ValidationArgs(
                golden_output=LazyGoldenGenerator(
                    output_ndarray=output_placeholder,
                    lazy_golden_generator=create_lazy_golden,
                ),
                relative_accuracy=5e-2,
                absolute_accuracy=1e-5,
            )

            # Import kernel function
            if is_dynamic:
                from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I_mx import (
                    blockwise_mm_shard_intermediate_mx_hybrid,
                )

                kernel_func = blockwise_mm_shard_intermediate_mx_hybrid
            else:
                from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I_mx import blockwise_mm_shard_intermediate_mx

                kernel_func = blockwise_mm_shard_intermediate_mx
            # Remove internal data and reorder to match kernel signature
            variant = 'shard_on_I_mx_hybrid' if is_dynamic else 'shard_on_I_mx'
            kernel_input_for_kernel = order_kernel_input(kernel_input, variant=variant)

            # Execute test
            test_manager.execute(
                KernelArgs(
                    kernel_func=kernel_func,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input_for_kernel,
                    validation_args=validation_args,
                )
            )

    @staticmethod
    def moe_bwmm_mx_shard_I_unit_config():
        """
        Generate unit test configuration for MoE BWMM MXFP4/MXFP8 shard-on-I kernel.

        Returns:
            RangeTestConfig with test cases for shard-on-I variant
        """
        test_cases = []

        for test_params in moe_bwmm_mx_shard_I_kernel_params:
            # Parameter order: [vnc_degree, H, T, I_TP, E, B, top_k, act_fn, expert_affinities_scaling_mode,
            #                   dtype, weight_dtype, skip_mode, bias, gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower]
            (
                vnc_degree,
                hidden,
                tokens,
                intermediate,
                expert,
                block_size,
                top_k,
                act_fn,
                expert_affinities_scaling_mode,
                dtype,
                weight_dtype,
                skip_mode,
                bias,
                is_dynamic,
                gate_clamp_upper,
                gate_clamp_lower,
                up_clamp_upper,
                up_clamp_lower,
            ) = test_params

            test_case = {
                MOE_BWMM_MX_CONFIG: {
                    VNC_DEGREE_DIM_NAME: vnc_degree,
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    BLOCK_SIZE_DIM_NAME: block_size,
                    TOP_K_DIM_NAME: top_k,
                    ACT_FN_DIM_NAME: act_fn,
                    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode.value,
                    DTYPE_DIM_NAME: dtype,
                    WEIGHT_DTYPE_DIM_NAME: weight_dtype,
                    SKIP_MODE_DIM_NAME: skip_mode,
                    BIAS_DIM_NAME: int(bias),
                    IS_DYNAMIC_DIM_NAME: int(is_dynamic),
                    GATE_CLAMP_UPPER_DIM_NAME: gate_clamp_upper,
                    GATE_CLAMP_LOWER_DIM_NAME: gate_clamp_lower,
                    UP_CLAMP_UPPER_DIM_NAME: up_clamp_upper,
                    UP_CLAMP_LOWER_DIM_NAME: up_clamp_lower,
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

    @range_test_config(moe_bwmm_mx_shard_I_unit_config())
    def test_moe_bwmm_mx_shard_I_unit(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
        platform_target: Platforms,
    ):
        """
        Unit test entry point for MoE BWMM MXFP4/MXFP8 shard-on-I kernel.

        This test validates the blockwise matrix multiplication kernel with
        MXFP4/MXFP8 quantization and intermediate dimension sharding against
        golden outputs computed in NumPy.
        """
        config = range_test_options.tensors[MOE_BWMM_MX_CONFIG]

        if platform_target is not Platforms.TRN3:
            pytest.skip("MXFP4/MXFP8 is only supported on TRN3.")

        lnc_count = config[VNC_DEGREE_DIM_NAME]

        # Shard-on-I requires exactly 2 shards
        if lnc_count != 2:
            pytest.skip("Shard-on-I kernel requires exactly 2 shards.")

        compiler_args = CompilerArgs(logical_nc_config=lnc_count, platform_target=platform_target)

        self.run_moe_bwmm_mx_shard_I_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            dtype=config[DTYPE_DIM_NAME],
            weight_dtype=config[WEIGHT_DTYPE_DIM_NAME],
            expert_affinities_scaling_mode=ExpertAffinityScaleMode(config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]),
            bias=bool(config[BIAS_DIM_NAME]),
            is_dynamic=config[IS_DYNAMIC_DIM_NAME],
            collector=collector,
        )
