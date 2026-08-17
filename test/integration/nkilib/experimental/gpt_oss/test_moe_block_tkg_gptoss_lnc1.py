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

"""Standalone LNC=1 MoE-block unit test for the GPT-OSS-120B MXFP4 decode shard.

This is the *MoE half* of the GPT-OSS MXFP4 decode layer, exercised on its own at
the single-rank / single-NeuronCore (LNC=1) configuration we target for trn3.3xl
bring-up. It reuses the production ``moe_block_tkg`` kernel (via the test-local
``mx_moe_block_tkg_wrapper`` bitcast shim, exactly as the core MoE tests do) and
its input builders; the only thing that makes it "standalone" is the pinned
GPT-OSS shape + SwiGLU/router wiring below.

LNC=1 support comes from the cherry-picked kernel enablement (commit
"Enable lnc=1 for moe_tkg, moe_block_tkg, and attention_block_tkg"): the
``n_prgs==2`` assert is relaxed for MX weight paths and the cross-core
shard/reduce steps self-disable at ``n_prgs==1``. All-expert MXFP4 is the GPT-OSS
decode path.

GPT-OSS-120B MoE (from the vendored golden config):
  * hidden H = 2880, padded to 3072 (the kernel needs H % 512 == 0; 2880 fails,
    3072 passes). ``hidden_actual=2880`` so RMSNorm normalizes over the true width.
  * intermediate I = 2880, padded to 3072 for the same reason.
  * 128 experts total, top_k = 4, router softmax (applied *after* top-k).
  * SwiGLU: gate clamped to <= 7.0, up clamped to [-6, 8] which is
    ``clamp(up, -7, 7) + 1`` once the GPT-OSS "+1" is folded into the up bias
    (the kernel adds bias before clamp), and activation ``x*sigmoid(1.702*x)``
    (``ActFnType.Swish`` bakes alpha=1.702).
  * MXFP4 experts: ``float4_e2m1fn_x4`` weights + uint8 block-of-32 E8M0 scales,
    passed to the kernel as raw uint16 (NxD convention) via the wrapper.

Expert sharding / bring-up ladder (all at T=128, LNC=1):
  * 1-rank  : num_global=128, num_local=128  (all experts on the one device),
              rank_id=0. This is the single-rank bring-up case — everything the
              router can pick is resident, so the golden matches exactly.
  * 8-rank  : num_global=128, num_local=16   (1/8th of the experts per rank).
  * 128-rank: num_global=128, num_local=1    (one expert per rank; the eventual
              target, needs a high-rank machine).
Only the single-rank case runs here (per the "low-rank LNC=1 only" scope); the
8/128-rank rungs are recorded as skipped params so the ladder is explicit.

The test diffs the kernel against ``moe_block_tkg_torch_ref`` (RMSNorm -> router
top-k -> per-expert SwiGLU MLP) via ``UnitTestFramework``. With no Neuron device
the run is compile/trace-only; on ``trn3pds-pdx10-1`` (TRN3) it executes and
numerically validates at the MXFP4 low-precision tolerance.
"""

import functools

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe_block.moe_block_tkg_torch import moe_block_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEBlockIOLayout,
    RouterActFnType,
)

# Reuse the maintained MoE input builders + the test-local MX bitcast wrapper.
from test.integration.nkilib.core.moe_block.test_moe_block_tkg import (
    convert_mx_weights_to_uint,
    generate_inputs,
    mx_moe_block_tkg_wrapper,
)
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# ── GPT-OSS-120B MoE constants (padded for the LNC=1 MXFP4 kernel) ───────────
_T = 128  # decode tokens (batch * seqlen)
_H = 3072  # hidden, padded from 2880 to a multiple of 512
_H_ACTUAL = 2880  # true GPT-OSS-120B hidden width (RMSNorm divisor)
_I = 3072  # intermediate, padded from 2880
_NUM_GLOBAL_EXPERTS = 128
_TOP_K = 4
_EPS = 1e-5
# SwiGLU: gate <= 7.0, up in [-6, 8] == clamp(up, -7, 7) + 1 (bias-before-clamp),
# activation Swish (x*sigmoid(1.702*x)). These are exactly _get_clamp_limits(True).
_GATE_CLAMP_UPPER = 7.0
_UP_CLAMP_UPPER = 8.0
_UP_CLAMP_LOWER = -6.0
_MOE_WEIGHT_DTYPE = nl.float4_e2m1fn_x4  # MXFP4
_INPUT_DTYPE = nl.float16
_ROUTER_MM_DTYPE = nl.float16

# Expert-parallel bring-up ladder: (num_local_experts, run_this_rung?).
#   1-rank => all 128 experts local; 8-rank => 16; 128-rank => 1.
# Only the single-rank rung runs under the low-rank LNC=1 scope.
_EP_LADDER = [
    pytest.param(128, id="rank1_local128"),
    pytest.param(16, id="rank8_local16", marks=pytest.mark.skip(reason="8-rank: needs multi-rank machine")),
    pytest.param(1, id="rank128_local1", marks=pytest.mark.skip(reason="128-rank: needs high-rank machine")),
]


def _bake_up_bias_plus_one(gate_up_bias):
    """Fold the GPT-OSS SwiGLU "+1" into the up half of the fused gate/up bias.

    The kernel adds the provided bias *before* the up clamp, so a stored up bias
    of ``b + 1`` combined with the up clamp ``[-6, 8]`` reproduces GPT-OSS's
    ``clamp(up, -7, 7) + 1``. The MXFP4 fused-bias layout is
    ``[E_L, intermediate_p, 2, n_I512, q_width]`` with axis-2 index 1 == "up".
    Returns the same array (mutated in place); safe because the kernel and the
    torch ref both consume this identical tensor, so the +1 stays self-consistent.
    """
    if gate_up_bias is None:
        return gate_up_bias
    # axis 2 is the gate(0)/up(1) selector in the fused MX bias layout.
    gate_up_bias[:, :, 1, ...] += 1.0
    return gate_up_bias


def _build_gptoss_moe_inputs(num_local_experts):
    """Build the GPT-OSS MXFP4 MoE kernel inputs for the given EP shard.

    Reuses the core MoE ``generate_inputs`` (pure-MX, all-expert), then applies
    the GPT-OSS-specific "+1 up bias" fold that the generic builder omits.
    Returns ``(kernel_input_mx_typed, kernel_input_for_nki)`` where the first
    keeps the MX-typed weights (for the torch golden) and the second has the
    uint16-viewed weights (for the NKI kernel), mirroring ``_run_moe_block_test``.
    """
    kernel_input = generate_inputs(
        batch=_T,
        seqlen=1,
        hidden=_H,
        hidden_actual=_H_ACTUAL,
        intermediate=_I,
        num_global_experts=_NUM_GLOBAL_EXPERTS,
        num_local_experts=num_local_experts,
        top_k=_TOP_K,
        router_fn=RouterActFnType.SOFTMAX,
        hidden_act_fn=ActFnType.Swish,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        moe_weight_dtype=_MOE_WEIGHT_DTYPE,
        input_dtype=_INPUT_DTYPE,
        has_bias=True,
        has_clamp=True,  # -> gate<=7, up in [-6, 8]
        router_act_first=False,  # softmax AFTER top-k (GPT-OSS)
        norm_topk_prob=False,
        skip_router_logits=True,
        router_mm_dtype=_ROUTER_MM_DTYPE,
        is_all_expert=True,  # 1-per-rank / all-expert decode path
        is_static_mx=False,  # pure MX (block scales only), not STATIC_MX
        is_row_quant=False,
        inp_layout=MoEBlockIOLayout.B_S_H,
        outp_layout=MoEBlockIOLayout.B_S_H,
    )
    # GPT-OSS "+1" on the up branch (generate_inputs uses plain rng.normal bias).
    _bake_up_bias_plus_one(kernel_input["expert_gate_up_bias"])

    # NKI sees uint16-viewed MX weights (NxD convention); golden keeps MX dtype.
    kernel_input_for_nki = convert_mx_weights_to_uint(kernel_input, _MOE_WEIGHT_DTYPE)
    return kernel_input, kernel_input_for_nki


# Not marked @pytest.mark.fast: the rank1/local128 MXFP4 MoE config fails compile-only
# validation (NCC_INKI016) in the release precommit lane. Runs on shared fleet instead.
@pytest.mark.parametrize("num_local_experts", _EP_LADDER)
def test_moe_block_tkg_gptoss_lnc1(
    test_manager: Orchestrator,
    platform_target: Platforms,
    num_local_experts: int,
):
    """Trace/compile (and, on TRN3 hardware, numerically validate) the GPT-OSS
    MXFP4 MoE decode block at LNC=1 against the torch reference."""
    if not platform_target.is_trn3():
        pytest.skip("MXFP4 (float4_e2m1fn_x4) is only supported on TRN3.")

    kernel_input, kernel_input_for_nki = _build_gptoss_moe_inputs(num_local_experts)

    def input_generator(test_config, input_tensor_def=None):
        return kernel_input_for_nki

    def output_tensors(ki):
        # skip_router_logits=True -> only the hidden-state output.
        return {"out": np.zeros((_T, _H), dtype=_INPUT_DTYPE)}

    # Torch golden runs on the MX-typed weights (not the uint16 view).
    original_torch_ref = torch_ref_wrapper(moe_block_tkg_torch_ref)

    @functools.wraps(original_torch_ref)
    def mx_torch_ref(**kwargs):
        kwargs["expert_gate_up_weights"] = kernel_input["expert_gate_up_weights"]
        kwargs["expert_down_weights"] = kernel_input["expert_down_weights"]
        return original_torch_ref(**kwargs)

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=mx_moe_block_tkg_wrapper,
        torch_ref=mx_torch_ref,
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(
            logical_nc_config=1,  # <-- LNC=1 (single NeuronCore)
            platform_target=platform_target,
            additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
        ),
        rtol=5e-2,  # MXFP4 low-precision tolerance (matches core MoE MX tests)
        atol=1e-5,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
