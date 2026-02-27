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
    _q_width,
)
from test.integration.nkilib.core.moe_block.test_moe_block_tkg_model_config import (
    moe_block_tkg_model_configs,
)
from test.integration.nkilib.utils.test_kernel_common import (
    is_dtype_low_precision,
    is_dtype_mx,
)
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    CompilerArgs,
    Platforms,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe_block.moe_block_tkg import moe_block_tkg as moe_block_tkg_kernel
from nkilib_src.nkilib.core.moe_block.moe_block_tkg_torch import moe_block_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType


def generate_inputs(
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
    has_bias: bool,
    has_clamp: bool,
    router_act_first: bool,
    norm_topk_prob: bool,
    skip_router_logits: bool,
    router_mm_dtype,
    is_all_expert: bool,
) -> dict:
    """Build input tensors for moe_block_tkg kernel tests. Returns dict of numpy arrays."""
    is_mx_weight = is_dtype_mx(moe_weight_dtype)

    if is_mx_weight:
        mx_weights = gen_moe_mx_weights(hidden, intermediate, num_local_experts, moe_weight_dtype)

    np.random.seed(42)
    rng = np.random.default_rng(42)

    hidden_dtype = input_dtype
    weight_dtype = moe_weight_dtype

    inputs = {}
    inputs["inp"] = dt.static_cast(rng.uniform(low=-0.1, high=0.1, size=(batch, seqlen, hidden)), hidden_dtype)
    inputs["gamma"] = dt.static_cast(rng.uniform(low=-0.1, high=0.1, size=(1, hidden)), hidden_dtype)
    inputs["router_weights"] = dt.static_cast(
        rng.uniform(low=-0.1, high=0.1, size=(hidden, num_global_experts)), router_mm_dtype
    )

    # Expert weights
    if is_mx_weight:
        intermediate_p = math.ceil(intermediate / 4 / 8) * 8 if intermediate < 512 else _pmax
        n_H512_tile = hidden // (_pmax * _q_width)
        n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))
        inputs["expert_gate_up_weights"] = mx_weights.gate_up_w_qtz
        inputs["expert_down_weights"] = mx_weights.down_w_qtz
        inputs["expert_gate_up_weights_scale"] = mx_weights.gate_up_w_scale
        inputs["expert_down_weights_scale"] = mx_weights.down_w_scale
    else:
        inputs["expert_gate_up_weights"] = rng.normal(size=(num_local_experts, hidden, 2, intermediate)).astype(
            weight_dtype
        )
        inputs["expert_down_weights"] = rng.normal(size=(num_local_experts, intermediate, hidden)).astype(weight_dtype)
        inputs["expert_gate_up_weights_scale"] = None
        inputs["expert_down_weights_scale"] = None

    # Bias
    inputs["router_bias"] = (
        dt.static_cast(rng.uniform(low=-0.1, high=0.1, size=(1, num_global_experts)), router_mm_dtype)
        if has_bias
        else None
    )
    if has_bias:
        if is_mx_weight:
            expert_gate_up_bias_shape = (num_local_experts, intermediate_p, 2, n_I512_tile, _q_width)
        else:
            expert_gate_up_bias_shape = (num_local_experts, 2, intermediate)
        inputs["expert_gate_up_bias"] = rng.normal(size=expert_gate_up_bias_shape).astype(hidden_dtype)
        inputs["expert_down_bias"] = rng.normal(size=(num_local_experts, hidden)).astype(hidden_dtype)
    else:
        inputs["expert_gate_up_bias"] = None
        inputs["expert_down_bias"] = None

    # Scalar / enum params
    inputs["top_k"] = top_k
    inputs["router_act_fn"] = router_fn
    inputs["router_pre_norm"] = router_act_first
    inputs["norm_topk_prob"] = norm_topk_prob
    inputs["expert_affinities_scaling_mode"] = expert_affinities_scaling_mode
    inputs["hidden_act_fn"] = hidden_act_fn

    clamp_limit = _get_clamp_limits(has_clamp)
    inputs["gate_clamp_upper_limit"] = clamp_limit[0]
    inputs["gate_clamp_lower_limit"] = clamp_limit[1]
    inputs["up_clamp_upper_limit"] = clamp_limit[2]
    inputs["up_clamp_lower_limit"] = clamp_limit[3]

    inputs["router_mm_dtype"] = router_mm_dtype
    inputs["hidden_actual"] = hidden_actual
    inputs["skip_router_logits"] = skip_router_logits
    inputs["is_all_expert"] = is_all_expert

    # rank_id for all-expert mode
    if is_all_expert:
        num_ranks = num_global_experts // num_local_experts
        rank_id_val = np.random.RandomState(42).randint(0, num_ranks)
        inputs["rank_id"] = np.array([[rank_id_val]], dtype=np.uint32)
    else:
        inputs["rank_id"] = None

    return inputs


# fmt: off
# Abbreviation mapping for keyword-prefixed test IDs (must match PARAM_NAMES order)
_PARAM_ABBREVS = \
    "ln,  ae,              ba,     sq,         hi,         ha,             im,             ge,                 le,                tk,          rf,                         af,                 sm,                                     wd,                     id,             se,                 bi,         cl,         ra,                 np,                 sr,                     rd"
PARAM_NAMES = \
    "lnc, is_all_expert,   batch,  seqlen,     hidden,     hidden_actual,  intermediate,   num_global_experts, num_local_experts, top_k,       router_fn,                  hidden_act_fn,      expert_affinities_scaling_mode,         moe_weight_dtype,       input_dtype,    has_shared_expert,  has_bias,   has_clamp,  router_act_first,   norm_topk_prob,     skip_router_logits,     router_mm_dtype"

MANUAL_PARAMS = [
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
    [2,     False,          38,     1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     False,          120,     1,         3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
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
    [2,     True,           16,     1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     True,           16,     1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     True,           16,     1,          4096,       None,           384,            128,                8,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    # All-expert MXFP4 tests (T must be divisible by 4)
    # GPT-OSS 120B
    [2,     True,           32,     4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           32,     5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           64,     4,          3072,       None,           3072,           128,                2,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           64,     5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           128,    4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           128,    5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           256,    3,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
    [2,     True,           512,    2,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              False,                  nl.float16],
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
    return "_".join(f"{k.strip()}-{_format_val(v)}" for k, v in zip(_PARAM_ABBREVS.split(","), params))


MANUAL_PARAM_IDS = [_make_id(p) for p in MANUAL_PARAMS]
MODEL_PARAM_IDS = [f"{MODEL_TEST_TYPE}_{_make_id(p)}" for p in moe_block_tkg_model_configs]


@pytest_test_metadata(
    name="MoE Block TKG",
    pytest_marks=["moe", "block", "tkg"],
    tags=["model"],
)
@final
class TestMoEBlockTkgKernel:
    def _run_moe_block_test(
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
        metadata: dict | None = None,
    ):
        if is_dtype_mx(moe_weight_dtype) and platform_target is not Platforms.TRN3:
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

        def input_generator(test_config, input_tensor_def=None):
            return kernel_input

        def output_tensors(kernel_input):
            out = {"out": np.zeros((tokens, hidden), dtype=input_dtype)}
            if not skip_router_logits:
                out["router_logits"] = np.zeros((tokens, num_global_experts), dtype=input_dtype)
            return out

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_block_tkg_kernel,
            torch_ref=torch_ref_wrapper(moe_block_tkg_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=5e-2 if is_dtype_low_precision(moe_weight_dtype) else 1e-2,
            atol=1e-5,
            metadata=metadata,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, MANUAL_PARAMS, ids=MANUAL_PARAM_IDS)
    def test_moe_block_kernel_unit(
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
        if not is_all_expert and intermediate == 192 and (batch * seqlen) == 1 and not is_dtype_mx(moe_weight_dtype):
            pytest.xfail("failing determinism check")

        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_moe_block_test(**kwargs)

    @pytest.mark.parametrize(PARAM_NAMES, moe_block_tkg_model_configs, ids=MODEL_PARAM_IDS)
    def test_moe_block_kernel_model(
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
        metadata_key = {
            "ln": lnc,
            "ae": bool(is_all_expert),
            "ba": batch,
            "sq": seqlen,
            "hi": hidden,
            "im": intermediate,
            "ge": num_global_experts,
            "le": num_local_experts,
        }
        self._run_moe_block_test(**kwargs, metadata={"config_name": "test_moe_block", "key": metadata_key})
