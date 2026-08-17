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
import math
from typing import final

import neuron_dtypes as dt
import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe_block.moe_block_tkg import moe_block_tkg as moe_block_tkg_kernel
from nkilib_src.nkilib.core.moe_block.moe_block_tkg_torch import moe_block_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEBlockIOLayout,
    QuantizationType,
    RouterActFnType,
)

from test.integration.nkilib.core.mlp.test_mlp_common import gen_moe_mx_weights
from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import (
    _get_clamp_limits,
    _pmax,
    _q_width,
)

try:
    from test.integration.nkilib.core.moe_block.test_moe_block_tkg_model_config import (
        moe_block_tkg_model_configs,
    )
except ImportError:
    moe_block_tkg_model_configs = {}

from test.integration.nkilib.utils.test_kernel_common import (
    is_dtype_low_precision,
    is_dtype_mx,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Mapping from MX x4 dtype to the NKI unsigned integer dtype with matching bit-width.
# mxfp4_x4: 4 bits × 4 = 16 bits → uint16
# mxfp8_x4: 8 bits × 4 = 32 bits → uint32
# NKI dtypes (nl.uint16/nl.uint32) must be used instead of numpy dtypes (np.uint16/np.uint32)
# because the NKI compiler rejects numpy dtypes during kernel specialization.
_MX_TO_UINT_DTYPE = {
    nl.float4_e2m1fn_x4: nl.uint16,
    nl.float8_e4m3fn_x4: nl.uint32,
    nl.float8_e5m2_x4: nl.uint32,
}

# Reverse mapping: NKI unsigned integer dtype → MX x4 dtype for reinterpret_cast
_UINT_TO_MX_DTYPE = {
    nl.uint16: nl.float4_e2m1fn_x4,
    nl.uint32: nl.float8_e4m3fn_x4,
}


def get_dynamic_all_expert_params(is_all_expert, moe_weight_dtype, T, num_local_experts):
    """Compute dynamic all-expert params from test config.

    Returns (is_all_expert_dynamic, block_size).
    Enables dynamism only for all-expert MX mode with E_L=1, T > 512, T divisible by 256 for now.
    """
    if not (is_all_expert and is_dtype_mx(moe_weight_dtype) and T > 512 and T % 256 == 0 and num_local_experts == 1):
        return False, None

    if T % 128 == 0:
        block_size = 128
    elif T % 64 == 0:
        block_size = 64
    else:
        return False, None

    return True, block_size


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


@nki.jit
def mx_moe_block_tkg_wrapper(
    inp,
    gamma,
    router_weights,
    expert_gate_up_weights,
    expert_down_weights,
    shared_expert_gate_w=None,
    shared_expert_up_w=None,
    shared_expert_down_w=None,
    expert_gate_up_weights_scale=None,
    expert_down_weights_scale=None,
    router_bias=None,
    expert_gate_up_bias=None,
    expert_down_bias=None,
    shared_expert_gate_bias=None,
    shared_expert_up_bias=None,
    shared_expert_down_bias=None,
    eps=1e-6,
    top_k=1,
    router_act_fn=None,
    router_pre_norm=True,
    norm_topk_prob=False,
    expert_affinities_scaling_mode=None,
    hidden_act_fn=None,
    hidden_act_scale_factor=None,
    hidden_act_bias=None,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    router_mm_dtype=None,
    hidden_actual=None,
    skip_router_logits=False,
    is_all_expert=False,
    rank_id=None,
    residual=None,
    expert_gate_up_input_scale=None,
    expert_down_input_scale=None,
    is_all_expert_dynamic=False,
    block_size=None,
    inp_layout=None,
    outp_layout=None,
):
    """Wrapper that bitcasts unsigned integer weights to MX x4 dtype before calling the kernel.

    Simulates how NxD passes MX weights — as raw uint16/uint32 tensors that need to be
    reinterpreted as float4_e2m1fn_x4 or float8_e4m3fn_x4 dtype.
    """
    mx_dtype = _UINT_TO_MX_DTYPE[expert_gate_up_weights.dtype]
    gate_up_view = expert_gate_up_weights.view(mx_dtype)
    down_view = expert_down_weights.view(mx_dtype)

    return moe_block_tkg_kernel(
        inp=inp,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=gate_up_view,
        expert_down_weights=down_view,
        shared_expert_gate_w=shared_expert_gate_w,
        shared_expert_up_w=shared_expert_up_w,
        shared_expert_down_w=shared_expert_down_w,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        shared_expert_gate_bias=shared_expert_gate_bias,
        shared_expert_up_bias=shared_expert_up_bias,
        shared_expert_down_bias=shared_expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=router_act_fn,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        hidden_act_fn=hidden_act_fn,
        hidden_act_scale_factor=hidden_act_scale_factor,
        hidden_act_bias=hidden_act_bias,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=router_mm_dtype,
        hidden_actual=hidden_actual,
        skip_router_logits=skip_router_logits,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        residual=residual,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
        is_all_expert_dynamic=is_all_expert_dynamic,
        block_size=block_size,
    )


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
    is_static_mx: bool = False,
    is_static: bool = False,
    is_row_quant: bool = False,
    inp_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
    outp_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
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
    # Router weights: for large-H float16 configs, use separate RNG with small w_range
    # to avoid sigmoid saturation / near-ties that cause router_topk fp16 matmul precision
    # to flip top-K selection (NKILIB-729). Other configs keep original behavior.
    if not has_bias and router_mm_dtype == nl.float16 and hidden >= 7168:  # sigmoid saturates at this H scale
        router_rng = np.random.default_rng(180)  # seed maximizes min-gap (45e-6) across E_L=8/16/32
        w_range = 0.1  # keeps logits ~1.3, sigmoid in linear range
        inputs["router_weights"] = dt.static_cast(
            router_rng.uniform(low=-w_range, high=w_range, size=(hidden, num_global_experts)), router_mm_dtype
        )
    else:
        w_range = 1.0 if (not has_bias and router_mm_dtype == nl.float16) else 0.1
        inputs["router_weights"] = dt.static_cast(
            rng.uniform(low=-w_range, high=w_range, size=(hidden, num_global_experts)), router_mm_dtype
        )

    # Expert weights
    if is_mx_weight:
        intermediate_p = math.ceil(intermediate / 4 / 8) * 8 if intermediate < 512 else _pmax
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

    # STATIC_MX and STATIC share the same scale generation:
    # Per-expert weight dequant scales + per-tensor activation dequant scales.
    # STATIC_MX: used for post-matmul rescaling with dummy MX block scales (127).
    # STATIC: input scales used for detection only (TRN2 does BF16 × FP8 matmul, no activation quantization).
    if is_static_mx or is_static:
        inputs["expert_gate_up_weights_scale"] = rng.uniform(0.001, 0.01, size=(num_local_experts, 2, 1)).astype(
            np.float32
        )
        inputs["expert_down_weights_scale"] = rng.uniform(0.001, 0.01, size=(num_local_experts, 1)).astype(np.float32)
        inputs["expert_gate_up_input_scale"] = rng.uniform(0.001, 0.01, size=(num_local_experts, 1)).astype(np.float32)
        inputs["expert_down_input_scale"] = rng.uniform(0.01, 0.1, size=(num_local_experts, 1)).astype(np.float32)
    elif is_row_quant:
        # Per-row weight dequant scales (pre-shuffled to match MX output layout)
        # gate/up: [E_L, 2, n_I512*4], down: [E_L, H//128]
        n_I512 = math.ceil(intermediate / (_pmax * _q_width))
        gate_up_scale_cols = n_I512 * _q_width
        down_scale_cols = hidden // _pmax
        inputs["expert_gate_up_weights_scale"] = rng.uniform(
            0.001, 0.01, size=(num_local_experts, 2, gate_up_scale_cols)
        ).astype(np.float32)
        inputs["expert_down_weights_scale"] = rng.uniform(
            0.001, 0.01, size=(num_local_experts, down_scale_cols)
        ).astype(np.float32)
        # No input scales for ROW_MX (computed dynamically at runtime)
        inputs["expert_gate_up_input_scale"] = None
        inputs["expert_down_input_scale"] = None
    else:
        inputs["expert_gate_up_input_scale"] = None
        inputs["expert_down_input_scale"] = None

    # Bias
    # Add per-expert offset to router_bias to reduce bf16 ties in top-K selection (CR-265459436).
    router_bias_tiebreak = np.linspace(0, 1.0, num_global_experts).reshape(1, num_global_experts)
    inputs["router_bias"] = (
        dt.static_cast(
            rng.uniform(low=-0.1, high=0.1, size=(1, num_global_experts)) + router_bias_tiebreak, router_mm_dtype
        )
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
        # Use rank_id=0 for large-H float16 configs to avoid router top-k tie-breaking (NKILIB-729).
        # Other configs keep random rank_id for broader coverage.
        if not has_bias and router_mm_dtype == nl.float16 and hidden >= 7168:
            rank_id_val = 0
        else:
            # Hardcoded to rank 62 for now: the index-monotonic router bias funnels tokens to
            # high-index experts, so a high rank sees the most traffic (nonzero output). 62 sits one
            # below the max valid rank (num_ranks-1) for headroom off the exact gather boundary.
            # Clamped to the valid range so configs with fewer ranks stay in bounds.
            # Was: rank_id_val = np.random.RandomState(42).randint(0, num_ranks)
            rank_id_val = min(62, num_ranks - 1)
        inputs["rank_id"] = np.array([[rank_id_val]], dtype=np.uint32)
    else:
        inputs["rank_id"] = None

    if inp_layout != MoEBlockIOLayout.B_S_H:
        inputs["inp_layout"] = inp_layout
    if outp_layout != MoEBlockIOLayout.B_S_H:
        inputs["outp_layout"] = outp_layout

    # Convert input to transposed layout [H0, n_prgs, H1_shard, BxS]
    if inp_layout == MoEBlockIOLayout._128_Nprgs_Hfree_T:
        n_prgs = 2  # LNC=2
        H0 = 128
        H1_shard = hidden // (H0 * n_prgs)
        BxS = batch * seqlen
        flat = inputs["inp"].reshape(BxS, hidden)
        inputs["inp"] = flat.reshape(BxS, n_prgs, H0, H1_shard).transpose(2, 1, 3, 0).copy()

    return inputs


# fmt: off
# Abbreviation mapping for keyword-prefixed test IDs (must match PARAM_NAMES order)
_PARAM_ABBREVS = \
    "ln,  ae,              ba,     sq,         hi,         ha,             im,             ge,                 le,                tk,          rf,                         af,                 sm,                                     wd,                     id,             se,                 bi,         cl,         ra,                 np,                 sr,                     rd,             qt"
PARAM_NAMES = \
    "lnc, is_all_expert,   batch,  seqlen,     hidden,     hidden_actual,  intermediate,   num_global_experts, num_local_experts, top_k,       router_fn,                  hidden_act_fn,      expert_affinities_scaling_mode,         moe_weight_dtype,       input_dtype,    has_shared_expert,  has_bias,   has_clamp,  router_act_first,   norm_topk_prob,     skip_router_logits,     router_mm_dtype,    quant_type"

MANUAL_PARAMS = [
    # Selective-load tests (num_global_experts == num_local_experts)
    # GPT-OSS 120B
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    pytest.param(2,     False,          1,      5,          3072,       2880,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX, marks=pytest.mark.fast),
    [2,     False,          1,      1,          3072,       None,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     False,          1,      4,          3072,       2880,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          1,      4,          3072,       2880,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          1,      1,          3072,       None,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          1,      1,          3072,       None,           576,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          1,      4,          3072,       2880,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    pytest.param(2,     False,          1,      5,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE, marks=pytest.mark.fast),
    [2,     False,          4,      1,          3072,       2880,           192,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          8,      1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          16,     1,          3072,       2880,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          19,     1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    pytest.param(2,     False,          32,     1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE, marks=pytest.mark.fast),
    [2,     False,          38,     1,          3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     False,          120,     1,         3072,       None,           128,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    # Qwen3 235B
    [2,     False,          1,      1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               True,                   nl.bfloat16, QuantizationType.NONE],
    [2,     False,          1,      1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               True,                   nl.bfloat16, QuantizationType.NONE],
    # All-expert BF16 tests (num_local_experts can be smaller than num_global_experts)
    # Minimal test for BF16 with H=640 (odd multiple of 128, unbalanced H-sharding across LNC-2)
    [2,     True,           16,     1,           640,       None,           640,            8,                  1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    # GPT OSS 120B
    [2,     True,           19,     1,          3072,       None,           128,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     True,           32,     1,          3072,       None,           128,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     True,           32,     1,          3072,       None,           384,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     True,           32,     1,          3072,       None,           768,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     True,           32,     1,          3072,       None,           768,            128,                4,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     True,           32,     1,          3072,       None,           1536,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    [2,     True,           32,     1,          3072,       None,           1536,           128,                2,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE],
    pytest.param(2,     True,           32,     1,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float16,             nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.NONE, marks=pytest.mark.fast),
    # Qwen3 235B
    [2,     True,           16,     1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               True,                   nl.bfloat16, QuantizationType.NONE],
    [2,     True,           16,     1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,            nl.bfloat16,    False,              False,      False,      True,               True,               True,                   nl.bfloat16, QuantizationType.NONE],
    [2,     True,           16,     1,          4096,       None,           384,            128,                8,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               True,                   nl.bfloat16, QuantizationType.NONE],
    # All-expert MXFP4 tests
    # LNC=1 (single core) configs — verified for all-expert MX and selective-load MX paths
    [1,     True,           512,    1,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [1,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    # GPT-OSS 120B
    [2,     True,           32,     4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    pytest.param(2,     True,           32,     4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX, marks=pytest.mark.fast),
    pytest.param(2,     True,           32,     5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX, marks=pytest.mark.fast),
    [2,     True,           64,     4,          3072,       None,           3072,           128,                2,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           64,     5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           128,    4,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                  nl.float16, QuantizationType.MX],
    [2,     True,           128,    5,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           256,    3,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    pytest.param(2,     True,           512,    2,          3072,       None,           3072,           128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX, marks=pytest.mark.fast),
    # E/EP>1, T<128 tests
    [2,     True,           4,      1,          3072,       2880,           3072,           128,                16,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           8,      1,          3072,       2880,           3072,           128,                8,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           16,     1,          3072,       2880,           3072,           128,                4,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           32,     1,          3072,       2880,           3072,           128,                2,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    # small I
    [2,     True,           32,      1,          3072,       2880,           384,           128,                128,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           64,      1,          3072,       2880,           192,           128,                128,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    [2,     True,           128,      1,          3072,       2880,           96,           128,                128,                 4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float4_e2m1fn_x4,    nl.float16,     False,              True,       True,       False,              False,              True,                   nl.float16, QuantizationType.MX],
    # DeepSeek-equivalent large-T configs (H=7168) — triggers T-dimension tiling (NKILIB-729)
    # Uses float16 (not bfloat16) for input/router to reduce router top-k tie-breaking mismatches
    [2,     True,           4,       1,          7168,       None,           2048,           256,                2,                  8,          RouterActFnType.SIGMOID,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              False,      False,      True,               True,               True,                   nl.float16, QuantizationType.MX],
    [2,     True,           4,       1,          7168,       None,           2048,           256,                4,                  8,          RouterActFnType.SIGMOID,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              False,      False,      True,               True,               True,                   nl.float16, QuantizationType.MX],
    [2,     True,           2048,    1,          7168,       None,           2048,           256,                2,                  8,          RouterActFnType.SIGMOID,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              False,      False,      True,               True,               True,                   nl.float16, QuantizationType.MX],
    [2,     True,           2048,    1,          7168,       None,           2048,           256,                4,                  8,          RouterActFnType.SIGMOID,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              False,      False,      True,               True,               True,                   nl.float16, QuantizationType.MX],
    pytest.param(2,     True,           512,     4,          7168,       None,           2048,           256,                2,                  8,          RouterActFnType.SIGMOID,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              False,      False,      True,               True,               True,                   nl.float16, QuantizationType.MX, marks=pytest.mark.fast),
    # NKILIB-729: E_L=8 ticket config (E_L=16/32 blocked on router_topk fp16 precision)
    [2,     True,           2048,    1,          7168,       None,           1024,           256,                8,                  8,          RouterActFnType.SIGMOID,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3fn_x4,    nl.float16,     False,              False,      False,      True,               True,               True,                   nl.float16, QuantizationType.MX],
]

# STATIC_MX test configs: selective-load and all-expert combined
# Uses MX-packed weights with float32 dequant scales + software static FP8 quantization
# fmt: off
_MOE_MX_ALL_EXPERT_XFAIL = pytest.mark.xfail(
    reason="all-expert MX MLP incorrect output on trn3_a0; pre-existing kernel bug, fix pending",
    strict=False,
)

STATIC_MX_PARAM_NAMES = "lnc, is_all_expert, batch, seqlen, hidden, hidden_actual, intermediate, num_global_experts, num_local_experts, top_k, router_fn, hidden_act_fn, expert_affinities_scaling_mode, input_dtype, has_bias, has_clamp, router_act_first, norm_topk_prob, skip_router_logits, router_mm_dtype"
STATIC_MX_PARAMS = [
    # LNC=1 (single core) tiled STATIC_MX, T=128: exercises the shard_on_tokens
    # gate at the tiled _router_topk call (must be n_prgs>1, else LNC=1 bakes an
    # invalid single-core SendRecv -> NCC_IBTN243). is_all_expert=False path.
    [1,     False,          128,    1,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    # Qwen3 235B selective-load (is_all_expert=False)
    [2,     False,          1,      1,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     False,          8,      1,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     False,          1,      4,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    # Qwen3 235B all-expert (is_all_expert=True)
    [2,     True,           1,      32,         4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    [2,     True,           4,      1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    pytest.param(2,     True,           256,    1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32, marks=_MOE_MX_ALL_EXPERT_XFAIL),
    [2,     True,           4,      1,          4096,       None,           1536,           128,                1,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    [2,     True,           256,    1,          4096,       None,           1536,           128,                1,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    [2,     True,           4,      4,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    pytest.param(2,     True,           4,      4,          4096,       None,           1536,           128,                1,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32, marks=pytest.mark.fast),
    # Qwen3 235B all-expert (T not divisible by 4)
    [2,     True,           1,      1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    [2,     True,           2,      1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    [2,     True,           3,      1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    # other all-expert, T not divisible by 4 cases
    [2,     True,           63,      1,         4096,       None,           384,            128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    # Large T all-expert: T=2048 triggers moe_block T-tiling (HBM [T,H] fed into kernel) and kernel SHARD_T.
    # skip_router_logits=True because the tiled path does not return router logits.
    [2,     True,           2048,    1,         4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               True,                   nl.float32],
    # Large T all-expert with E_L=8 (has_bias=True needed as router tie-breaker)
    pytest.param(2,     True,           2048,    1,         4096,       None,           1536,           128,                8,                  8,          RouterActFnType.SOFTMAX,     ActFnType.SiLU,    ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    True,       False,      True,               True,               True,                   nl.float32, marks=_MOE_MX_ALL_EXPERT_XFAIL),
]
# fmt: on
STATIC_MX_PARAM_IDS = [f"static_mx_{i}" for i in range(len(STATIC_MX_PARAMS))]

# STATIC FP8 test configs: per-tensor weight dequant on TRN2 (BF16 × FP8 matmul, no activation quantization)
# fmt: off
STATIC_PARAM_NAMES = "lnc, is_all_expert, batch, seqlen, hidden, hidden_actual, intermediate, num_global_experts, num_local_experts, top_k, router_fn, hidden_act_fn, expert_affinities_scaling_mode, moe_weight_dtype, input_dtype, has_shared_expert, has_bias, has_clamp, router_act_first, norm_topk_prob, skip_router_logits, router_mm_dtype"
STATIC_PARAMS = [
    # Selective-load
    [2,     False,          1,      1,          3072,       None,           384,            128,                128,                4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.float16,     False,              False,      True,       False,              False,              False,                  nl.float16],
    [2,     False,          1,      1,          4096,       None,           384,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
    # All-expert
    [2,     True,           4,      1,          3072,       None,           384,            128,                1,                  4,          RouterActFnType.SOFTMAX,    ActFnType.Swish,    ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.float16,     False,              False,      True,       False,              False,              False,                  nl.float16],
    [2,     True,           16,     1,          4096,       None,           384,            128,                8,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.float8_e4m3,         nl.bfloat16,    False,              False,      False,      True,               True,               False,                  nl.bfloat16],
]
# fmt: on
STATIC_PARAM_IDS = [f"static_{i}" for i in range(len(STATIC_PARAMS))]


def _exclusion_key(params):
    """(is_all_expert, batch, seqlen, hidden, intermediate, moe_weight_dtype) — uniquely identifies test configs."""
    return (params[1], params[2], params[3], params[4], params[6], params[13])


# (is_all_expert, batch, seqlen, hidden, intermediate, moe_weight_dtype) keys for full-only tests (excluded from fast suite)
_FULL_ONLY_STATIC_KEYS = frozenset(
    {
        (False, 1, 1, 3072, 384, nl.float8_e4m3),
        (False, 1, 1, 4096, 384, nl.float8_e4m3),
    }
)

# ROW_MX test configs: all-expert and selective-load (per-token dynamic FP8 quantization)
# Uses MX-packed weights with per-row float32 dequant scales + per-token dynamic FP8 quantization
# fmt: off
ROW_MX_PARAM_NAMES = "lnc, is_all_expert, batch, seqlen, hidden, hidden_actual, intermediate, num_global_experts, num_local_experts, top_k, router_fn, hidden_act_fn, expert_affinities_scaling_mode, input_dtype, has_bias, has_clamp, router_act_first, norm_topk_prob, skip_router_logits, router_mm_dtype"
ROW_MX_PARAMS = [
    # Qwen3 235B selective-load (is_all_expert=False)
    [2,     False,          1,      1,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     False,          8,      1,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    [2,     False,          1,      4,          4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.bfloat16],
    # Qwen3 235B all-expert (is_all_expert=True)
    [2,     True,           1,      32,         4096,       None,           192,            128,                128,                8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    pytest.param(2,     True,           4,      1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32, marks=_MOE_MX_ALL_EXPERT_XFAIL),
    pytest.param(2,     True,           4,      1,          4096,       None,           1536,           128,                1,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32, marks=pytest.mark.fast),
    # Bias + clamp coverage
    pytest.param(2,     True,           4,      1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    True,       True,       True,               True,               False,                  nl.float32, marks=_MOE_MX_ALL_EXPERT_XFAIL),
    # Disabled: router topk tie-breaking on degenerate test inputs leads to numerical mismatch
    # [2,     True,           4,      4,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    # [2,     True,           4,      4,          4096,       None,           1536,           128,                1,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    # [2,     True,           256,    1,          4096,       None,           1536,           128,                2,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
    # [2,     True,           256,    1,          4096,       None,           1536,           128,                1,                  8,          RouterActFnType.SOFTMAX,    ActFnType.SiLU,     ExpertAffinityScaleMode.POST_SCALE,     nl.bfloat16,    False,      False,      True,               True,               False,                  nl.float32],
]
# fmt: on
ROW_MX_PARAM_IDS = [f"row_mx_{i}" for i in range(len(ROW_MX_PARAMS))]


def _format_val(v):
    """Format a parameter value for test ID: enums→value, bools→int, else str."""
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "value"):
        return v.value
    return v


def _make_id(params):
    """Generate a keyword-prefixed test ID string from a parameter list."""
    if hasattr(params, "values") and hasattr(params, "marks"):
        params = params.values
    return "_".join(f"{k.strip()}-{_format_val(v)}" for k, v in zip(_PARAM_ABBREVS.split(","), params, strict=True))


MANUAL_PARAM_IDS = [_make_id(p) for p in MANUAL_PARAMS]

# (is_all_expert, batch, seqlen, hidden, intermediate, moe_weight_dtype) keys for full-only tests (excluded from fast suite)
_FULL_ONLY_MANUAL_KEYS = frozenset(
    {
        (False, 1, 1, 3072, 384, nl.float16),
        (False, 1, 4, 3072, 384, nl.float16),
        (False, 1, 1, 3072, 576, nl.float16),
        (False, 120, 1, 3072, 128, nl.float16),
        (False, 1, 1, 4096, 384, nl.bfloat16),
        (False, 1, 1, 4096, 384, nl.float8_e4m3),
        (True, 16, 1, 4096, 384, nl.bfloat16),
        # Moved out of fast: high memory (1.4 GB+) with 128 experts, covered by other fast configs
        (False, 4, 1, 3072, 192, nl.float16),
        (False, 1, 4, 3072, 192, nl.float16),
        (False, 1, 1, 3072, 192, nl.float16),
        # Moved out of fast: high duration (>2min)
        (True, 2048, 1, 7168, 1024, nl.float8_e4m3fn_x4),
        (True, 2048, 1, 7168, 2048, nl.float8_e4m3fn_x4),
        (True, 32, 1, 3072, 384, nl.float4_e2m1fn_x4),
        (True, 64, 1, 3072, 192, nl.float4_e2m1fn_x4),
        (True, 128, 1, 3072, 96, nl.float4_e2m1fn_x4),
        (False, 38, 1, 3072, 128, nl.float16),
    }
)


@pytest_test_metadata(name="MoE Block TKG Model", tags=["model"])
@pytest_marks(["moe", "block", "tkg", "mx"])
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
        is_static_mx: bool = False,
        is_static: bool = False,
        is_row_quant: bool = False,
        inp_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
        outp_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
    ):
        if is_dtype_mx(moe_weight_dtype) and not platform_target.is_trn3():
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
            is_static_mx=is_static_mx,
            is_static=is_static,
            is_row_quant=is_row_quant,
            inp_layout=inp_layout,
            outp_layout=outp_layout,
        )
        tokens = batch * seqlen

        # For MX dtypes, convert weights to uint and use the wrapper to simulate NxD behavior.
        # Keep original kernel_input (with MX-typed weights) for the torch ref, and create
        # a separate copy with uint-typed weights for the NKI kernel.
        is_mx = is_dtype_mx(moe_weight_dtype)
        if is_mx:
            kernel_input_for_nki = convert_mx_weights_to_uint(kernel_input, moe_weight_dtype)
            kernel_func = mx_moe_block_tkg_wrapper
        else:
            kernel_input_for_nki = kernel_input
            kernel_func = moe_block_tkg_kernel

        # Auto-detect and enable dynamic all-expert mode
        # Exclude STATIC_MX and ROW_MX paths which don't support dynamism
        is_dynamic, dyn_block_size = (
            get_dynamic_all_expert_params(
                is_all_expert,
                moe_weight_dtype,
                tokens,
                num_local_experts,
            )
            if not (is_static_mx or is_row_quant)
            else (False, None)
        )
        if is_dynamic:
            kernel_input_for_nki["is_all_expert_dynamic"] = True
            kernel_input_for_nki["block_size"] = dyn_block_size

        def input_generator(test_config, input_tensor_def=None):
            return kernel_input_for_nki

        def output_tensors(ki):
            if outp_layout == MoEBlockIOLayout._128_Nprgs_Hfree_T:
                H0 = 128
                n_prgs = 2
                H1_shard = hidden // (H0 * n_prgs)
                out = {"out": np.zeros((H0, n_prgs, H1_shard, tokens), dtype=input_dtype)}
            else:
                out = {"out": np.zeros((tokens, hidden), dtype=input_dtype)}
            if not skip_router_logits:
                out["router_logits"] = np.zeros((tokens, num_global_experts), dtype=input_dtype)
            return out

        # Wrap torch ref to use original MX-typed weights for golden computation
        if is_mx:
            original_torch_ref = torch_ref_wrapper(moe_block_tkg_torch_ref)

            @functools.wraps(original_torch_ref)
            def mx_torch_ref(**kwargs):
                kwargs['expert_gate_up_weights'] = kernel_input['expert_gate_up_weights']
                kwargs['expert_down_weights'] = kernel_input['expert_down_weights']
                return original_torch_ref(**kwargs)

            torch_ref_fn = mx_torch_ref
        else:
            torch_ref_fn = torch_ref_wrapper(moe_block_tkg_torch_ref)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref_fn,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
            collector=collector,
        )
        # bf16/fp16 tolerance: 1.1% to account for K-way expert accumulation rounding
        # in bf16 vs f32 torch ref (see NKILIB-977).
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
            ),
            rtol=7e-2
            if (is_static_mx or is_row_quant)
            else (5e-2 if (is_dtype_low_precision(moe_weight_dtype) or is_dynamic) else 1.1e-2),
            atol=1e-5,
            metadata=metadata,
        )

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
        quant_type,
        platform_target: Platforms,
    ):
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs.pop("quant_type", None)
        self._run_moe_block_test(**kwargs)

    @pytest.mark.parametrize(STATIC_MX_PARAM_NAMES, STATIC_MX_PARAMS, ids=STATIC_MX_PARAM_IDS)
    def test_moe_block_static_mx(
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
        input_dtype,
        has_bias,
        has_clamp,
        router_act_first,
        norm_topk_prob,
        skip_router_logits,
        router_mm_dtype,
        platform_target: Platforms,
    ):
        if not platform_target.is_trn3():
            pytest.skip("STATIC_MX uses MX matmul engine, only supported on TRN3.")

        self._run_moe_block_test(
            test_manager=test_manager,
            collector=collector,
            lnc=2,
            is_all_expert=is_all_expert,
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
            moe_weight_dtype=nl.float8_e4m3fn_x4,
            input_dtype=input_dtype,
            has_shared_expert=False,
            has_bias=has_bias,
            has_clamp=has_clamp,
            router_act_first=router_act_first,
            norm_topk_prob=norm_topk_prob,
            skip_router_logits=skip_router_logits,
            router_mm_dtype=router_mm_dtype,
            platform_target=platform_target,
            is_static_mx=True,
        )

    @pytest.mark.parametrize(STATIC_PARAM_NAMES, STATIC_PARAMS, ids=STATIC_PARAM_IDS)
    @pytest.mark.platforms(exclude=[Platforms.TRN1])
    def test_moe_block_static(
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
        if platform_target.is_trn3():
            pytest.skip("STATIC uses BF16 × FP8 matmul, only supported on TRN2.")
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_moe_block_test(**kwargs, is_static=True)

    @pytest.mark.parametrize(ROW_MX_PARAM_NAMES, ROW_MX_PARAMS, ids=ROW_MX_PARAM_IDS)
    def test_moe_block_row_mx(
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
        input_dtype,
        has_bias,
        has_clamp,
        router_act_first,
        norm_topk_prob,
        skip_router_logits,
        router_mm_dtype,
        platform_target: Platforms,
    ):
        if not platform_target.is_trn3():
            pytest.skip("ROW_MX uses MX matmul engine, only supported on TRN3.")

        self._run_moe_block_test(
            test_manager=test_manager,
            collector=collector,
            lnc=2,
            is_all_expert=is_all_expert,
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
            moe_weight_dtype=nl.float8_e4m3fn_x4,
            input_dtype=input_dtype,
            has_shared_expert=False,
            has_bias=has_bias,
            has_clamp=has_clamp,
            router_act_first=router_act_first,
            norm_topk_prob=norm_topk_prob,
            skip_router_logits=skip_router_logits,
            router_mm_dtype=router_mm_dtype,
            platform_target=platform_target,
            is_row_quant=True,
        )

    # fmt: off
    @pytest.mark.parametrize(
        "is_all_expert, batch, hidden, intermediate, num_local_experts, top_k, inp_layout, outp_layout, moe_weight_dtype, is_static",
        [
            # Selective bf16 tin+tout
            (False, 1, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            # All-expert bf16 tin+tout — batch sweep
            (True, 2, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            (True, 4, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            (True, 16, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            (True, 64, 4096, 1536, 2, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            (True, 128, 4096, 1536, 2, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            # Directional tests
            (True, 16, 4096, 384, 128, 8, MoEBlockIOLayout.B_S_H, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
            (False, 1, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout.B_S_H, nl.bfloat16, False),
            # Quant tests
            (True, 16, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.float8_e4m3, False),
            (False, 1, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.float8_e4m3, False),
            (True, 16, 4096, 384, 128, 8, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.float8_e4m3, True),
            # Selective transposed with shard_on_h_disabled — cheap config (8 experts) for fast suite
            pytest.param(False, 1, 3072, 384, 8, 4, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False, marks=pytest.mark.fast),
            # Selective transposed with 128 experts — full suite only (high memory: 2.3 GB)
            (False, 1, 3072, 384, 128, 4, MoEBlockIOLayout._128_Nprgs_Hfree_T, MoEBlockIOLayout._128_Nprgs_Hfree_T, nl.bfloat16, False),
        ],
    )
    # fmt: on
    def test_moe_block_transposed_io(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        is_all_expert,
        batch,
        hidden,
        intermediate,
        num_local_experts,
        top_k,
        inp_layout,
        outp_layout,
        moe_weight_dtype,
        is_static,
    ):
        self._run_moe_block_test(
            test_manager=test_manager,
            collector=collector,
            lnc=2,
            is_all_expert=is_all_expert,
            batch=batch,
            seqlen=1,
            hidden=hidden,
            hidden_actual=None,
            intermediate=intermediate,
            # Selective mode: num_global_experts must equal num_local_experts because topK
            # indices index directly into the weight tensor [num_local_experts, ...].
            # All-expert mode: num_local_experts < num_global_experts is valid (EP sharding).
            num_global_experts=num_local_experts if not is_all_expert else 128,
            num_local_experts=num_local_experts,
            top_k=top_k,
            router_fn=RouterActFnType.SOFTMAX,
            hidden_act_fn=ActFnType.SiLU,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            moe_weight_dtype=moe_weight_dtype,
            input_dtype=nl.bfloat16,
            has_shared_expert=False,
            has_bias=False,
            has_clamp=False,
            router_act_first=True,
            norm_topk_prob=True,
            skip_router_logits=False,
            router_mm_dtype=nl.bfloat16,
            platform_target=platform_target,
            inp_layout=inp_layout,
            outp_layout=outp_layout,
            is_static=is_static,
        )


# MODEL TESTING ENTRY POINT
@pytest_marks(["moe", "block", "tkg", "mx", "model"])
@final
class TestMoEBlockTkgModel:
    """Model regression tests for MoE Block TKG kernel.

    Separate test methods per tier for cleaner pytest discovery:
    - test_tier0: Critical model configs (high priority)
    - test_optimal: Optimal performance configs
    - test_generality: Generality/coverage configs
    """

    # Tier params resolved at class definition time (lazy loading would require conditional imports)
    _TIER0_PARAMS, _TIER0_IDS = (
        prepare_model_parametrize(
            {ModelTestType.TIER0: moe_block_tkg_model_configs.get(ModelTestType.TIER0, [])}, id_formatter=_make_id
        )
        if moe_block_tkg_model_configs
        else ([], [])
    )
    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize(
            {ModelTestType.OPTIMAL: moe_block_tkg_model_configs.get(ModelTestType.OPTIMAL, [])}, id_formatter=_make_id
        )
        if moe_block_tkg_model_configs
        else ([], [])
    )
    _GENERALITY_PARAMS, _GENERALITY_IDS = (
        prepare_model_parametrize(
            {ModelTestType.GENERALITY: moe_block_tkg_model_configs.get(ModelTestType.GENERALITY, [])},
            id_formatter=_make_id,
        )
        if moe_block_tkg_model_configs
        else ([], [])
    )

    def _run_model_test(self, **kwargs):
        """Delegate to TestMoEBlockTkgKernel._run_moe_block_test with metadata."""
        metadata_key = {
            "ln": kwargs.get("lnc"),
            "ae": bool(kwargs.get("is_all_expert")),
            "ba": kwargs.get("batch"),
            "sq": kwargs.get("seqlen"),
            "hi": kwargs.get("hidden"),
            "im": kwargs.get("intermediate"),
            "ge": kwargs.get("num_global_experts"),
            "le": kwargs.get("num_local_experts"),
            "wd": str(kwargs.get("moe_weight_dtype")),
        }
        metadata = {"config_name": "test_moe_block_tkg", "key": metadata_key}
        qt = kwargs.pop("quant_type", QuantizationType.NONE)
        kwargs["is_static"] = qt == QuantizationType.STATIC
        kwargs["is_static_mx"] = qt == QuantizationType.STATIC_MX
        kwargs["is_row_quant"] = qt == QuantizationType.ROW_MX
        TestMoEBlockTkgKernel()._run_moe_block_test(**kwargs, metadata=metadata)

    @pytest.mark.tier0
    @pytest.mark.parametrize(PARAM_NAMES, _TIER0_PARAMS, ids=_TIER0_IDS)
    def test_tier0(
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
        quant_type,
        platform_target: Platforms,
    ):
        """TIER0: Critical model configs - highest priority for model validation."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)

    @pytest.mark.optimal
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(PARAM_NAMES, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
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
        quant_type,
        platform_target: Platforms,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)

    @pytest.mark.generality
    @pytest.mark.parametrize(PARAM_NAMES, _GENERALITY_PARAMS, ids=_GENERALITY_IDS)
    def test_generality(
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
        quant_type,
        platform_target: Platforms,
    ):
        """GENERALITY: Broad coverage model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)
