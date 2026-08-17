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

"""MX MoE Block TKG wrapper that bitcasts unsigned integer weights to MX x4 dtype.

Simulates how NxD passes MX weights — as raw uint16/uint32 tensors that need to be
reinterpreted as float4_e2m1fn_x4 or float8_e4m3fn_x4 dtype before calling the kernel.
"""

import nki
import nki.language as nl

from ...core.moe_block.moe_block_tkg import moe_block_tkg as moe_block_tkg_kernel

# Reverse mapping: NKI unsigned integer dtype → MX x4 dtype for reinterpret_cast
# mxfp4_x4: 4 bits × 4 = 16 bits → uint16
# mxfp8_x4: 8 bits × 4 = 32 bits → uint32
_UINT_TO_MX_DTYPE = {
    nl.uint16: nl.float4_e2m1fn_x4,
    nl.uint32: nl.float8_e4m3fn_x4,
}


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
    )
