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

import functools
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType
from nkilib_src.nkilib.experimental.moe_block.mx_moe_block_tkg_wrapper import mx_moe_block_tkg_wrapper
from nkilib_src.nkilib.experimental.moe_block.mx_moe_block_tkg_wrapper_torch import mx_moe_block_tkg_wrapper_torch_ref

from test.integration.nkilib.core.moe_block.test_moe_block_tkg import generate_inputs
from test.integration.nkilib.utils.test_kernel_common import (
    is_dtype_low_precision,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    Platforms,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Mapping from MX x4 dtype to the NKI unsigned integer dtype with matching bit-width.
_MX_TO_UINT_DTYPE = {
    nl.float4_e2m1fn_x4: nl.uint16,
    nl.float8_e4m3fn_x4: nl.uint32,
    nl.float8_e5m2_x4: nl.uint32,
}


def convert_mx_weights_to_uint(kernel_input: dict, moe_weight_dtype) -> dict:
    """Convert MX weights to unsigned integer dtype to simulate NxD behavior.

    NxD passes MX weights as raw unsigned integer tensors (uint16 for mxfp4_x4,
    uint32 for mxfp8_x4) that need to be reinterpreted inside the kernel.
    """
    uint_dtype = _MX_TO_UINT_DTYPE[moe_weight_dtype]
    result = kernel_input.copy()
    for key in ['expert_gate_up_weights', 'expert_down_weights']:
        if key in result and result[key] is not None:
            result[key] = result[key].view(uint_dtype)
    return result


# fmt: off
# Abbreviation mapping for keyword-prefixed test IDs (must match PARAM_NAMES order)
_PARAM_ABBREVS = \
    "ln,  ae,              ba,     sq,         hi,         ha,             im,             ge,                 le,                tk,          rf,                         af,                 sm,                                     wd,                     id,             se,                 bi,         cl,         ra,                 np,                 sr,                     rd"
PARAM_NAMES = \
    "lnc, is_all_expert,   batch,  seqlen,     hidden,     hidden_actual,  intermediate,   num_global_experts, num_local_experts, top_k,       router_fn,                  hidden_act_fn,      expert_affinities_scaling_mode,         moe_weight_dtype,       input_dtype,    has_shared_expert,  has_bias,   has_clamp,  router_act_first,   norm_topk_prob,     skip_router_logits,     router_mm_dtype"

MANUAL_PARAMS = [
    # mxfp4 selective-load (GPT-OSS 120B)
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    # mxfp8 selective-load (GPT-OSS 120B)
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    # mxfp8 all-expert (GPT-OSS 120B)
    [2,     True,           32,     4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
]
# fmt: on


def _format_val(v):
    """Format a parameter value for test ID: enums→value, bools→int, else str."""
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "value"):
        return v.value
    return v


def _make_id(params):
    """Generate a keyword-prefixed test ID string from a parameter list."""
    return "_".join(f"{k.strip()}-{_format_val(v)}" for k, v in zip(_PARAM_ABBREVS.split(","), params, strict=True))


MANUAL_PARAM_IDS = [_make_id(p) for p in MANUAL_PARAMS]


def _make_mx_torch_ref(kernel_input):
    """Create a torch ref that swaps uint weights back to original MX weights for golden computation."""
    wrapped = torch_ref_wrapper(mx_moe_block_tkg_wrapper_torch_ref)

    @functools.wraps(wrapped)
    def mx_torch_ref(**kwargs):
        kwargs['expert_gate_up_weights'] = kernel_input['expert_gate_up_weights']
        kwargs['expert_down_weights'] = kernel_input['expert_down_weights']
        return wrapped(**kwargs)

    return mx_torch_ref


@pytest.mark.xfail(
    strict=False,
    reason="Intermittent failures on MX MoE Block TKG Wrapper; xfailed until root-caused.",
)
@pytest_test_metadata(name="MX MoE Block TKG Wrapper")
@pytest_marks(["moe", "block", "tkg", "mx"])
@final
class TestMxMoEBlockTkgWrapper:
    def _run_test(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        lnc: int,
        is_all_expert: bool,
        batch: int,
        seqlen: int,
        hidden: int,
        hidden_actual: int | None,
        intermediate: int,
        num_global_experts: int,
        num_local_experts: int,
        top_k: int,
        router_fn: RouterActFnType,
        hidden_act_fn: ActFnType,
        expert_affinities_scaling_mode: ExpertAffinityScaleMode,
        moe_weight_dtype,
        input_dtype,
        has_shared_expert: bool,
        has_bias: bool,
        has_clamp: bool,
        router_act_first: bool,
        norm_topk_prob: bool,
        skip_router_logits: bool,
        router_mm_dtype,
        platform_target: Platforms,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        kernel_input = generate_inputs(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            hidden_actual=hidden_actual,
            intermediate=intermediate,
            num_global_experts=num_global_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
            router_fn=router_fn,
            hidden_act_fn=hidden_act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            moe_weight_dtype=moe_weight_dtype,
            input_dtype=input_dtype,
            has_bias=has_bias,
            has_clamp=has_clamp,
            router_act_first=router_act_first,
            norm_topk_prob=norm_topk_prob,
            skip_router_logits=skip_router_logits,
            router_mm_dtype=router_mm_dtype,
            is_all_expert=is_all_expert,
        )
        tokens = batch * seqlen

        # Convert MX weights to uint to simulate NxD behavior.
        # Keep original kernel_input for the torch ref golden computation.
        kernel_input_for_nki = convert_mx_weights_to_uint(kernel_input, moe_weight_dtype)

        def input_generator(test_config, input_tensor_def=None):
            return kernel_input_for_nki

        def output_tensors(ki):
            out = {"out": np.zeros((tokens, hidden), dtype=input_dtype)}
            if not skip_router_logits:
                out["router_logits"] = np.zeros((tokens, num_global_experts), dtype=input_dtype)
            return out

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mx_moe_block_tkg_wrapper,
            torch_ref=_make_mx_torch_ref(kernel_input),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=5e-2 if is_dtype_low_precision(moe_weight_dtype) else 1e-2,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, MANUAL_PARAMS, ids=MANUAL_PARAM_IDS)
    def test_mx_moe_block_kernel_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
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
        platform_target: Platforms,
    ):
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_test(**kwargs)
