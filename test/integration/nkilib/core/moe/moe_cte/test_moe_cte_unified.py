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

"""Unified tests for moe_cte entry point using UnitTestFramework."""

from typing import final

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.moe.moe_cte import (
    MoECTEImplementation,
    moe_cte,
)
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_torch import moe_cte_torch_ref
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType

from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    _shard_on_block_mx_output_validator,
    _shard_on_block_output_validator,
    generate_moe_cte_unified_inputs,
    moe_cte_unified_output_tensors,
)
from test.utils.common_dataclasses import CompilerArgs, CustomValidatorWithOutputTensorData, Platforms, ValidationArgs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.mx_utils import is_mx_quantize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
# Parameter names for pytest.mark.parametrize
UNIFIED_PARAM_NAMES = \
    "impl,                                              hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, bias,  training, quantize,        act_fn,          expert_affinities_scaling_mode,     gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I, weight_dtype,           is_dynamic, non_overlapping_shards, use_packed_scales, ep_degree, quant_type"

# =============================================================================
# NON-MX TEST CASES
# =============================================================================
NON_MX_TEST_CASES = [
    # SHARD_ON_INTERMEDIATE_HW tests
    pytest.param(MoECTEImplementation.shard_on_i_hybrid,            3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False,                         None,                   False, False, False, 1, QuantizationType.NONE, marks=pytest.mark.fast),
    # SHARD_ON_INTERMEDIATE tests
    pytest.param(MoECTEImplementation.shard_on_i,                   3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,    True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False,                         None,                   False, False, False, 1, QuantizationType.NONE, marks=pytest.mark.fast),
    # SHARD_ON_BLOCK tests (skip=3: skip_token=True, skip_weight=True)
    pytest.param(MoECTEImplementation.shard_on_block,               3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 3,    True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False,                         None,                   False, True,  False, 1, QuantizationType.NONE, marks=pytest.mark.fast),
    pytest.param(MoECTEImplementation.shard_on_block,               3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 3,    True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,          False,                         None,                   False, False, False, 1, QuantizationType.NONE, marks=pytest.mark.fast),
    # Dropping kernel tests
    pytest.param(MoECTEImplementation.shard_on_i_dropping,          1536,   8192,   2,      4096,       2,     6144,         nl.bfloat16, 0,    False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True,                          None,                   False, False, False, 1, QuantizationType.NONE, marks=pytest.mark.fast),
]

# =============================================================================
# MX (MXFP4/MXFP8) SHARD-ON-BLOCK TEST CASES
# Covers both weight dtypes (MXFP4/MXFP8) crossed with both scale layouts
# (standard/packed).
# =============================================================================
MX_BLOCK_TEST_CASES = [
    # MXFP4 + standard scales
    (MoECTEImplementation.shard_on_block_mx,            3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,    False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float4_e2m1fn_x4,    False, False, False, 1, QuantizationType.MX),
    # MXFP4 + packed scales
    pytest.param(MoECTEImplementation.shard_on_block_mx,            3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,    False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float4_e2m1fn_x4,    False, False, True,  1, QuantizationType.MX, marks=pytest.mark.fast),
    # MXFP8 (e4m3) + standard scales
    (MoECTEImplementation.shard_on_block_mx,            3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,    False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float8_e4m3fn_x4,    False, False, False, 1, QuantizationType.MX),
    # MXFP8 (e4m3) + packed scales
    (MoECTEImplementation.shard_on_block_mx,            3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,    False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float8_e4m3fn_x4,    False, False, True,  1, QuantizationType.MX),
    # STATIC_MX (MXFP8 e4m3), dynamic loop
    pytest.param(MoECTEImplementation.shard_on_block_mx,            3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,    False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False,                         nl.float8_e4m3fn_x4,    True,  False, False, 8, QuantizationType.STATIC_MX, marks=pytest.mark.fast),
]

# =============================================================================
# MX (MXFP4/MXFP8) SHARD-ON-INTERMEDIATE TEST CASES
# =============================================================================
MX_SHARD_I_TEST_CASES = [
    # (MoECTEImplementation.shard_on_i_mx_hybrid,       7168,   1024,   8,      256,        8,     1024,         nl.bfloat16, 1,    True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7.0,           None,          7.0,         -7.0,        False,                         nl.float4_e2m1fn_x4,    True, False, False, 16, QuantizationType.MX),
]

ALL_TEST_CASES = NON_MX_TEST_CASES + MX_BLOCK_TEST_CASES + MX_SHARD_I_TEST_CASES
# fmt: on


@pytest_test_metadata(name="MoE CTE Unified Entry Point")
@pytest_marks(["moe", "cte", "unified"])
@final
class TestMoeCTEUnified:
    """Unified tests for moe_cte() entry point covering all implementations.

    skip modes:
    - 0: SkipMode(False, False)
    - 1: SkipMode(True, False)  - skip token
    - 2: SkipMode(False, True)  - skip weight
    - 3: SkipMode(True, True)   - skip both
    """

    @pytest.mark.parametrize(UNIFIED_PARAM_NAMES, ALL_TEST_CASES)
    def test_moe_cte_unified(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        impl: MoECTEImplementation,
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
        weight_dtype,
        is_dynamic: bool,
        non_overlapping_shards: bool,
        use_packed_scales: bool,
        ep_degree: int,
        quant_type: QuantizationType,
        platform_target: Platforms,
        request,
    ):
        lnc_degree = 2

        # Skip MX tests on non-TRN3 platforms
        is_mx = is_mx_quantize(weight_dtype)
        if is_mx and not platform_target.is_trn3():
            pytest.skip("MX (MXFP4/MXFP8) is only supported on TRN3.")

        target = platform_target

        def input_generator(test_config):
            return generate_moe_cte_unified_inputs(
                impl=impl,
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
                weight_dtype=weight_dtype,
                is_dynamic=is_dynamic,
                lnc_degree=lnc_degree,
                non_overlapping_shards=non_overlapping_shards,
                use_packed_scales=use_packed_scales,
                ep_degree=ep_degree,
                quantization_type=quant_type,
            )

        def output_tensors(kernel_input):
            return moe_cte_unified_output_tensors(
                kernel_input=kernel_input,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                impl=impl,
                training=training,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                lnc_degree=lnc_degree,
            )

        rtol, atol = (5e-2, 1e-5) if (is_mx or quantize) else (2e-2, 1e-5)

        compiler_args = CompilerArgs(logical_nc_config=lnc_degree, platform_target=target)

        # Block-sharding kernels emit a 2-shard output where shard 1 is scratch;
        # need a custom validator that compares only shard 0 against the ref.
        validator_cls = None
        if impl == MoECTEImplementation.shard_on_block_mx and top_k > 1:
            validator_cls = _shard_on_block_mx_output_validator(
                kernel_input=input_generator(None),
                tokens=tokens,
                hidden=hidden,
                lnc_degree=lnc_degree,
                rtol=rtol,
                atol=atol,
            )
        elif impl == MoECTEImplementation.shard_on_block and top_k > 1 and non_overlapping_shards:
            from inspect import signature

            kernel_input = input_generator(None)
            dma_skip = kernel_input["skip_dma"]
            T_out = tokens if dma_skip.skip_token else tokens + 1
            torch_ref = torch_ref_wrapper(moe_cte_torch_ref)
            ref_input = {k: v for k, v in kernel_input.items() if k in signature(torch_ref).parameters}

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

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_cte,
            torch_ref=torch_ref_wrapper(moe_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )

        if validator_cls is not None:
            output_placeholder = output_tensors(input_generator(None))
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
            framework.run_test(
                test_config=None,
                compiler_args=compiler_args,
                rtol=rtol,
                atol=atol,
                custom_validation_args=custom_validation,
            )
        else:
            framework.run_test(
                test_config=None,
                compiler_args=compiler_args,
                rtol=rtol,
                atol=atol,
            )
