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

"""Test suite for MoE BWMM MX CTE kernels using native pytest and coverage_parametrize."""

import random
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte import map_skip_mode
from test.integration.nkilib.core.moe.moe_cte.test_utils import (
    build_moe_bwmm_mx_cte,
    golden_moe_bwmm_mx_cte,
    order_kernel_input,
)
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, Platforms, ValidationArgs
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult, assert_negative_test_case
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

try:
    from test.integration.nkilib.core.moe.moe_cte.test_moe_bwmm_mx_cte_model_config import (
        moe_bwmm_mx_cte_model_configs,
    )
except ImportError:
    moe_bwmm_mx_cte_model_configs = []

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx import bwmm_shard_on_block_mx
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I_mx import (
    blockwise_mm_shard_intermediate_mx,
    blockwise_mm_shard_intermediate_mx_hybrid,
)
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

# fmt: off

# ============================================================================
# Test Parameters - Shard-on-Block (unit tests)
# ============================================================================

moe_bwmm_mx_cte_unit_params = "vnc_degree, hidden, tokens, intermediate, expert, block_size, top_k, act_fn, expert_affinities_scaling_mode, dtype, weight_dtype, skip_mode, bias, is_dynamic, gate_clamp_upper, gate_clamp_lower, up_clamp_upper, up_clamp_lower, use_uint_weights"

moe_bwmm_mx_cte_unit_perms = [
    # MXFP4 test cases
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 1024, 1536, 32, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 512, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False],
    # MXFP8 e4m3fn test cases
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 0, True, True, 7.0, None, 7.0, -7.0, False],
    # MXFP8 e5m2 test cases
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 10240, 384, 128, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 0, True, True, 7.0, None, 7.0, -7.0, False],
    # Alternative dtype weights test cases (simulates NxD behavior: uint16 for MXFP4, uint32 for MXFP8)
    [2, 3072, 1024, 384, 128, 256, 2, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, True],
    [2, 3072, 1024, 384, 128, 256, 2, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, True],
    [2, 3072, 1024, 384, 128, 256, 2, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, True],
]

# ============================================================================
# Test Parameters - Shard-on-I-MX (unit tests)
# ============================================================================

moe_bwmm_mx_shard_I_unit_perms = [
    [2, 3072, 1024, 2048, 8, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 3072, 1024, 2048, 8, 256, 4, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float4_e2m1fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 7168, 1024, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 7168, 1024, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, False],
    [2, 7168, 10240, 1024, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, False],      
    # Alternative dtype weights test cases (simulates NxD behavior: uint16 for MXFP4, uint32 for MXFP8)
    [2, 3072, 1024, 2048, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e4m3fn_x4, 1, True, True, 7.0, None, 7.0, -7.0, True],
    [2, 3072, 1024, 2048, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, False, 7.0, None, 7.0, -7.0, True],
    [2, 3072, 1024, 2048, 8, 256, 8, ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, nl.bfloat16, nl.float8_e5m2_x4, 1, True, True, 7.0, None, 7.0, -7.0, True],      
 
]

# fmt: on


# ============================================================================
# uint16 Weight Wrapper (simulates NxD behavior)
# ============================================================================


# Mapping from MXFP weight dtype to the alternative dtype used by NxD (torch/xla)
_MXFP_TO_ALTERNATIVE_DTYPE = {
    nl.float4_e2m1fn_x4: np.uint16,
    nl.float8_e4m3fn_x4: np.uint32,
    nl.float8_e5m2_x4: np.uint32,
}


def convert_mx_weights_to_uint_dtype(kernel_input: dict, weight_dtype) -> dict:
    """Convert MX weights to alternative dtype (uint16/uint32) to simulate NxD behavior."""
    alt_dtype = _MXFP_TO_ALTERNATIVE_DTYPE.get(weight_dtype)
    assert alt_dtype is not None, f"No alternative dtype mapping for {weight_dtype}"
    result = kernel_input.copy()
    for key in ['gate_up_proj_weight', 'down_proj_weight']:
        if key in result and result[key] is not None:
            result[key] = result[key].view(alt_dtype)
    return result


# ============================================================================
# Filter Functions
# ============================================================================


def filter_moe_bwmm_mx_cte_combinations(
    vnc_degree=None,
    hidden=None,
    intermediate=None,
    block_size=None,
    expert_affinities_scaling_mode=None,
    **kwargs,
):
    """
    Filter invalid parameter combinations for MoE BWMM MX CTE kernel (shard-on-block).

    Checks constraints from kernel_assert statements in bwmm_shard_on_block_mx.py:
    - vnc_degree == 2 (num_shards)
    - block_size % 128 == 0
    - 512 <= hidden <= 8192 and hidden % 512 == 0
    - intermediate % 16 == 0 and MX tiling rules
    - expert_affinities_scaling_mode == POST_SCALE
    """
    if vnc_degree is not None and vnc_degree != 2:
        return FilterResult.INVALID

    if (
        expert_affinities_scaling_mode is not None
        and expert_affinities_scaling_mode != ExpertAffinityScaleMode.POST_SCALE
    ):
        return FilterResult.INVALID

    if hidden is not None:
        if not (512 <= hidden <= 8192) or hidden % 512 != 0:
            return FilterResult.INVALID

    if block_size is not None and block_size % 128 != 0:
        return FilterResult.INVALID

    if intermediate is not None:
        if intermediate % 16 != 0:
            return FilterResult.INVALID
        if not (intermediate % 512 == 0 or (intermediate < 512 and intermediate % 32 == 0)):
            return FilterResult.INVALID

    return FilterResult.VALID


def filter_moe_bwmm_mx_shard_I_combinations(
    vnc_degree=None,
    **kwargs,
):
    """
    Filter invalid parameter combinations for MoE BWMM MX shard-on-I kernel.

    Only vnc_degree == 2 is enforced at runtime (kernel_assert in hybrid function).
    check_kernel_compatibility_mx exists but is not called by the kernel.
    """
    if vnc_degree is not None and vnc_degree != 2:
        return FilterResult.INVALID

    return FilterResult.VALID


# ============================================================================
# Test Class: Shard-on-Block
# ============================================================================


@pytest_test_metadata(
    name="MoE BWMM MX CTE",
    pytest_marks=["moe", "cte", "mx"],
)
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMoeBwmmMxCteKernel:
    """Test class for MoE BWMM MX CTE kernel (shard-on-block)."""

    def run_moe_bwmm_mx_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        lnc_degree: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        act_fn: ActFnType,
        skip_mode: int,
        gate_clamp_upper: float,
        gate_clamp_lower: float,
        up_clamp_upper: float,
        up_clamp_lower: float,
        dtype,
        weight_dtype,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        bias: bool,
        is_dynamic: bool,
        collector: MetricsCollector,
        use_uint_weights: bool = False,
    ):
        """Run a single MoE BWMM MX CTE test case."""
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

        def create_lazy_golden():
            return golden_moe_bwmm_mx_cte(
                kernel_input=kernel_input,
                dtype=dtype,
                lnc_degree=lnc_degree,
            )

        numpy_dtype = dt.finfo(dtype).dtype
        dma_skip = map_skip_mode(skip_mode)
        is_accumulating = top_k != 1
        out_T = tokens if dma_skip.skip_token else tokens + 1

        if is_accumulating:
            output_placeholder = {"output": np.zeros((lnc_degree, out_T, hidden), dtype=numpy_dtype)}
        else:
            output_placeholder = {"output": np.zeros((out_T, hidden), dtype=numpy_dtype)}

        validation_args = ValidationArgs(
            golden_output=LazyGoldenGenerator(
                output_ndarray=output_placeholder,
                lazy_golden_generator=create_lazy_golden,
            ),
            relative_accuracy=5e-2,
            absolute_accuracy=1e-5,
        )

        kernel_input_for_kernel = order_kernel_input(kernel_input, variant='shard_on_block_mx')

        if use_uint_weights:
            kernel_input_for_kernel = convert_mx_weights_to_uint_dtype(kernel_input_for_kernel, weight_dtype)

        test_manager.execute(
            KernelArgs(
                kernel_func=bwmm_shard_on_block_mx,
                compiler_input=compiler_args,
                kernel_input=kernel_input_for_kernel,
                validation_args=validation_args,
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(moe_bwmm_mx_cte_unit_params, moe_bwmm_mx_cte_unit_perms)
    def test_moe_bwmm_mx_cte_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
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
        dtype,
        weight_dtype,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float,
        gate_clamp_lower: float,
        up_clamp_upper: float,
        up_clamp_lower: float,
        use_uint_weights: bool,
    ):
        """Unit test for MoE BWMM MX CTE kernel with manual test vectors."""

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        self.run_moe_bwmm_mx_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            lnc_degree=vnc_degree,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            act_fn=act_fn,
            skip_mode=skip_mode,
            gate_clamp_upper=gate_clamp_upper,
            gate_clamp_lower=gate_clamp_lower,
            up_clamp_upper=up_clamp_upper,
            up_clamp_lower=up_clamp_lower,
            dtype=dtype,
            weight_dtype=weight_dtype,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            bias=bias,
            is_dynamic=is_dynamic,
            collector=collector,
            use_uint_weights=use_uint_weights,
        )

    @pytest.mark.coverage_parametrize(
        vnc_degree=BoundedRange([2], boundary_values=[]),
        hidden=BoundedRange([1536, 3072] + random.sample(range(1024, 6144, 512), 2), boundary_values=[]),
        tokens=BoundedRange([1024, 10240, 32768] + random.sample([2048, 4096, 8192], 2), boundary_values=[]),
        intermediate=BoundedRange([384, 512, 768, 1536], boundary_values=[100, 15]),
        expert=BoundedRange([8, 16, 32, 128], boundary_values=[]),
        block_size=BoundedRange([128, 256, 512], boundary_values=[64]),
        top_k=BoundedRange([2, 3, 4, 5], boundary_values=[1]),
        act_fn=BoundedRange([ActFnType.Swish], boundary_values=[]),
        expert_affinities_scaling_mode=BoundedRange([ExpertAffinityScaleMode.POST_SCALE], boundary_values=[]),
        dtype=BoundedRange([nl.bfloat16], boundary_values=[]),
        weight_dtype=BoundedRange([nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4], boundary_values=[]),
        skip_mode=BoundedRange([0, 1], boundary_values=[]),
        bias=BoundedRange([True], boundary_values=[]),
        is_dynamic=BoundedRange([False, True], boundary_values=[]),
        filter=filter_moe_bwmm_mx_cte_combinations,
        coverage="pairs",
    )
    def test_moe_bwmm_mx_cte_sweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
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
        dtype,
        weight_dtype,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        is_negative_test_case: bool,
    ):
        """Sweep test for MoE BWMM MX CTE kernel using coverage_parametrize."""

        with assert_negative_test_case(is_negative_test_case):
            compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

            self.run_moe_bwmm_mx_cte_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                lnc_degree=vnc_degree,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                act_fn=act_fn,
                skip_mode=skip_mode,
                gate_clamp_upper=7.0,
                gate_clamp_lower=None,
                up_clamp_upper=7.0,
                up_clamp_lower=-7.0,
                dtype=dtype,
                weight_dtype=weight_dtype,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                collector=collector,
            )

    ####################################################################################################################
    # MoE BWMM MX CTE Model Config Tests
    ####################################################################################################################

    @pytest.mark.parametrize(
        moe_bwmm_mx_cte_unit_params,
        moe_bwmm_mx_cte_model_configs,
    )
    def test_moe_bwmm_mx_cte_model(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
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
        dtype,
        weight_dtype,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float,
        gate_clamp_lower: float,
        up_clamp_upper: float,
        up_clamp_lower: float,
        use_uint_weights: bool,
    ):
        """Model config test for MoE BWMM MX CTE kernel."""
        if platform_target is not Platforms.TRN3:
            pytest.skip("MX quantization is only supported on TRN3.")

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        self.run_moe_bwmm_mx_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            lnc_degree=vnc_degree,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            act_fn=act_fn,
            skip_mode=skip_mode,
            gate_clamp_upper=gate_clamp_upper,
            gate_clamp_lower=gate_clamp_lower,
            up_clamp_upper=up_clamp_upper,
            up_clamp_lower=up_clamp_lower,
            dtype=dtype,
            weight_dtype=weight_dtype,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            bias=bias,
            is_dynamic=is_dynamic,
            collector=collector,
            use_uint_weights=use_uint_weights,
        )


# ============================================================================
# Test Class: Shard-on-I-MX
# ============================================================================


@pytest_test_metadata(
    name="MoE BWMM MX Shard-on-I",
    pytest_marks=["moe", "cte", "mx", "shard_on_I"],
)
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMoeBwmmMxShardIKernel:
    """Test class for MoE BWMM MX kernel with intermediate dimension sharding."""

    def run_moe_bwmm_mx_shard_I_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        lnc_degree: int,
        tokens: int,
        hidden: int,
        intermediate: int,
        expert: int,
        block_size: int,
        top_k: int,
        act_fn: ActFnType,
        skip_mode: int,
        gate_clamp_upper: float,
        gate_clamp_lower: float,
        up_clamp_upper: float,
        up_clamp_lower: float,
        dtype,
        weight_dtype,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        bias: bool,
        is_dynamic: bool,
        collector: MetricsCollector,
    ):
        """Run a single MoE BWMM MX shard-on-I test case."""
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

        def create_lazy_golden():
            return golden_moe_bwmm_mx_cte(
                kernel_input=kernel_input,
                dtype=dtype,
                lnc_degree=lnc_degree,
                is_shard_on_I=True,
            )

        dma_skip = map_skip_mode(skip_mode)
        out_T = tokens if dma_skip.skip_token else tokens + 1
        output_placeholder = {"output": np.zeros((out_T, hidden), dtype=dtype)}

        validation_args = ValidationArgs(
            golden_output=LazyGoldenGenerator(
                output_ndarray=output_placeholder,
                lazy_golden_generator=create_lazy_golden,
            ),
            relative_accuracy=5e-2,
            absolute_accuracy=1e-5,
        )

        if is_dynamic:
            kernel_func = blockwise_mm_shard_intermediate_mx_hybrid
        else:
            kernel_func = blockwise_mm_shard_intermediate_mx

        variant = 'shard_on_I_mx_hybrid' if is_dynamic else 'shard_on_I_mx'
        kernel_input_for_kernel = order_kernel_input(kernel_input, variant=variant)

        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_func,
                compiler_input=compiler_args,
                kernel_input=kernel_input_for_kernel,
                validation_args=validation_args,
            )
        )

    @pytest.mark.parametrize(moe_bwmm_mx_cte_unit_params, moe_bwmm_mx_shard_I_unit_perms)
    def test_moe_bwmm_mx_shard_I_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
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
        dtype,
        weight_dtype,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        gate_clamp_upper: float,
        gate_clamp_lower: float,
        up_clamp_upper: float,
        up_clamp_lower: float,
        use_uint_weights: bool,
    ):
        """Unit test for MoE BWMM MX shard-on-I kernel with manual test vectors."""

        if vnc_degree != 2:
            pytest.skip("Shard-on-I kernel requires exactly 2 shards.")

        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

        self.run_moe_bwmm_mx_shard_I_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            lnc_degree=vnc_degree,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            act_fn=act_fn,
            skip_mode=skip_mode,
            gate_clamp_upper=gate_clamp_upper,
            gate_clamp_lower=gate_clamp_lower,
            up_clamp_upper=up_clamp_upper,
            up_clamp_lower=up_clamp_lower,
            dtype=dtype,
            weight_dtype=weight_dtype,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            bias=bias,
            is_dynamic=is_dynamic,
            collector=collector,
        )

    @pytest.mark.coverage_parametrize(
        vnc_degree=BoundedRange([2], boundary_values=[]),
        hidden=BoundedRange([3072, 7168] + random.sample(range(1024, 8192, 512), 2), boundary_values=[]),
        tokens=BoundedRange([1024, 10240] + random.sample([2048, 4096, 8192], 2), boundary_values=[]),
        intermediate=BoundedRange([1024, 2048], boundary_values=[]),
        expert=BoundedRange([8, 32, 128], boundary_values=[]),
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
    )
    def test_moe_bwmm_mx_shard_I_sweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
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
        dtype,
        weight_dtype,
        skip_mode: int,
        bias: bool,
        is_dynamic: bool,
        is_negative_test_case: bool,
    ):
        """Sweep test for MoE BWMM MX shard-on-I kernel using coverage_parametrize."""

        with assert_negative_test_case(is_negative_test_case):
            compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)

            self.run_moe_bwmm_mx_shard_I_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                lnc_degree=vnc_degree,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                act_fn=act_fn,
                skip_mode=skip_mode,
                gate_clamp_upper=7.0,
                gate_clamp_lower=None,
                up_clamp_upper=7.0,
                up_clamp_lower=-7.0,
                dtype=dtype,
                weight_dtype=weight_dtype,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                bias=bias,
                is_dynamic=is_dynamic,
                collector=collector,
            )
