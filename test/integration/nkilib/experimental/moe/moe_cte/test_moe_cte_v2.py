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

"""Tests for experimental bwmm_shard_on_block_v2 and bwmm_shard_on_block_hybrid kernels."""

from typing import final

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    BWMMFunc,
    _shard_on_block_output_validator,
    generate_moe_cte_inputs,
    moe_cte_kernel_wrapper,
    moe_cte_output_tensors,
    moe_cte_torch_wrapper,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidatorWithOutputTensorData,
    Platforms,
    ValidationArgs,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
PARAM_NAMES = \
    "bwmm_func,                            hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, bias,  training, quantize, act_fn,          expert_affinities_scaling_mode,     gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, block_sharding_strategy"

SHARD_ON_BLOCK_V2_CASES = [
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     256,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, "HI_LO"),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     256,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, "PING_PONG"),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     256,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, "HI_LO_NO"),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     256,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,           3072,   10240,   8,      512,        1,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      256,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      256,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
    [BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      256,        4,     1536,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None],
    [BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     1536,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None],
    [BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      256,        4,     3072,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None],
    [BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,   8,      512,        4,     3072,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False, None],
    (BWMMFunc.SHARD_ON_BLOCK_V2,              3072,   10240,  128,    256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
]

# SHARD_ON_BLOCK_HW (hybrid/DLoC) - disabled pending follow-up
# SHARD_ON_BLOCK_HW_CASES = [
#     (BWMMFunc.SHARD_ON_BLOCK_HW,           3072,   10240,   8,      512,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
#     (BWMMFunc.SHARD_ON_BLOCK_HW,           3072,   10240,   8,      512,        4,     192,          nl.bfloat16, 3,   False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
#     (BWMMFunc.SHARD_ON_BLOCK_HW,           3072,   10240,   8,      256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
#     (BWMMFunc.SHARD_ON_BLOCK_HW,           3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False, None),
# ]

_FULL_ONLY_KEYS = {
    (3072, 10240, 8, 256, 4, 384),
    (3072, 10240, 8, 256, 4, 1536),
    (3072, 10240, 8, 256, 4, 3072),
    (3072, 10240, 8, 512, 1, 192),
    (3072, 10240, 8, 512, 4, 192),
    (3072, 10240, 8, 512, 4, 256),
    (3072, 10240, 8, 512, 4, 1536),
    (3072, 10240, 8, 512, 4, 3072),
    (3072, 10240, 128, 256, 4, 192),
}

ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[1:7]) not in _FULL_ONLY_KEYS else c
    for c in SHARD_ON_BLOCK_V2_CASES
]
# fmt: on


@final
@pytest_marks(["moe", "blockwise_mm", "lnc2", "v2"])
@pytest_test_metadata(name="MoE BWMM V2 CTE", tags=["experimental"])
class TestMoeBwmmV2:
    """Tests for experimental bwmm_shard_on_block_v2 kernel (manual alloc + DLoC)."""

    @pytest.mark.parametrize(PARAM_NAMES, ALL_PARAMS)
    def test_moe_bwmm_v2(
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
        platform_target: Platforms,
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
                dtype=dtype,
                bwmm_func_enum=bwmm_func,
                training=training,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
            )

        rtol, atol = 2e-2, 1e-5
        compiler_args = CompilerArgs(
            logical_nc_config=lnc_degree, platform_target=platform_target, dump_after_lowering=True
        )

        # Custom validator for (T, 2, H+E) output shape
        is_shard_block_accumulating = bwmm_func in (BWMMFunc.SHARD_ON_BLOCK_V2, BWMMFunc.SHARD_ON_BLOCK_HW) and (
            top_k > 1
        )
        custom_validation = None
        if is_shard_block_accumulating:
            kernel_input = input_generator(None)
            dma_skip = kernel_input["skip_dma"]
            T_out = tokens if dma_skip.skip_token else tokens + 1

            from inspect import signature

            torch_ref = torch_ref_wrapper(moe_cte_torch_wrapper)
            ref_sig = signature(torch_ref)
            ref_input = {k: v for k, v in kernel_input.items() if k in ref_sig.parameters}

            validator_cls = _shard_on_block_output_validator(
                torch_ref_fn=torch_ref,
                ref_input=ref_input,
                T_out=T_out,
                tokens=tokens,
                hidden=hidden,
                expert=expert,
                lnc_degree=lnc_degree,
                dtype=dtype,
                rtol=rtol,
                atol=atol,
            )
            output_placeholder = output_tensors(kernel_input)
            custom_validation = ValidationArgs(
                golden_output={
                    k: CustomValidatorWithOutputTensorData(
                        validator=validator_cls,
                        output_ndarray=v,
                    )
                    if k == "output"
                    else v
                    for k, v in output_placeholder.items()
                },
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
            custom_validation_args=custom_validation,
        )
