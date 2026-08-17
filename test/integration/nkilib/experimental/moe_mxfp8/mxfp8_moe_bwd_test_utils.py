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

"""Test-only input generation and routing-table helpers for the MXFP8 MoE backward tests.

The pure-PyTorch reference implementation of the kernel itself lives in
``nkilib.experimental.moe_mxfp8.bwd.blockwise_mm_backward_mxfp8_torch``
(co-located with the kernel per coding guidelines). This module hosts
test-fixture utilities that don't belong in the production source tree:
random input generation, routing-table construction, and the forward-pass
golden used solely to seed the activation checkpoint consumed by the
backward kernel under test.
"""

import hashlib
import math

import neuron_dtypes as ndtype
import nki.language as nl
import numpy as np
from neuronxcc.nki._private.test import mx_util
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import _get_mx_max_exp
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)

from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.matmul_mxfp8.utils import (
    resize_scales_compact_to_oversized_2d,
)
from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    _generate_fwd_golden,
    generate_token_position_to_id_and_experts,
)
from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    Q_TILE_K,
    generate_golden_packed_scales,
)

# ============================================================================
# Routing table generation
# ============================================================================


def _get_n_blocks(T, top_k, E, B):
    N = math.ceil((T * top_k - (E - 1)) / B) + E - 1
    return N


# ============================================================================
# Pre-quantization utilities
# ============================================================================


def _prequantize_2d(tensor_2d, use_scale_packing):
    """Pre-quantize a single 2D [K, F] tensor to MXFP8 x4 format.

    Steps:
      1. Swizzle [K, F] → [K/4, F*4]
      2. Quantize → (x4_data [K/4, F], compact_scales)
      3. Format scales based on use_scale_packing

    Returns (data, scales).
    """
    K, F = tensor_2d.shape
    swizzled = matmul_utils.swizzle_tensor(tensor_2d)
    fp8_x4_dtype = ndtype.float8_e4m3fn_x4
    data, compact_scales = mx_util.quantize_mx_golden(swizzled, fp8_x4_dtype, custom_mx_max_exp=_get_mx_max_exp)
    if use_scale_packing:
        scales = generate_golden_packed_scales(compact_scales, K, F, Q_TILE_K)
    else:
        scales = resize_scales_compact_to_oversized_2d(compact_scales)
    return data, scales


def prequantize_moe_weights(gate_up_proj_weight, down_proj_weight, use_scale_packing):
    """Pre-quantize MoE weights per-expert for the backward kernel.

    gate_up_proj_weight: [E, H, 2, I_TP] → per-expert [2*I_TP, H] → x4 [2*I_TP/4, H]
    down_proj_weight: [E, I_TP, H] → per-expert [H, I_TP] → x4 [H/4, I_TP]

    Returns:
        (gate_up_data, gate_up_scales, down_data, down_scales)
        gate_up_data: [E, 2*I_TP//4, H]
        gate_up_scales: [E, scales_K, scales_F]
        down_data: [E, H//4, I_TP]
        down_scales: [E, scales_K, scales_F]
    """
    E, H, _, I_TP = gate_up_proj_weight.shape

    gate_up_data_list = []
    gate_up_scales_list = []
    down_data_list = []
    down_scales_list = []

    for e in range(E):
        # gate_up: [H, 2, I_TP] → reshape to [H, 2*I_TP] → transpose to [2*I_TP, H]
        gate_up_2d = gate_up_proj_weight[e].reshape(H, 2 * I_TP).T.astype(np.float32)
        gu_data, gu_scales = _prequantize_2d(gate_up_2d, use_scale_packing)
        gate_up_data_list.append(gu_data)
        gate_up_scales_list.append(gu_scales)

        # down: [I_TP, H] → transpose to [H, I_TP]
        down_2d = down_proj_weight[e].T.astype(np.float32)
        d_data, d_scales = _prequantize_2d(down_2d, use_scale_packing)
        down_data_list.append(d_data)
        down_scales_list.append(d_scales)

    gate_up_data = np.stack(gate_up_data_list, axis=0)
    gate_up_scales = np.stack(gate_up_scales_list, axis=0)
    down_data = np.stack(down_data_list, axis=0)
    down_scales = np.stack(down_scales_list, axis=0)

    return gate_up_data, gate_up_scales, down_data, down_scales


# ============================================================================
# Input generation
# ============================================================================


def build_mxfp8_moe_bwd_inputs(
    tokens,
    hidden,
    intermediate,
    expert,
    block_size,
    top_k,
    dtype=nl.bfloat16,
    run_with_lnc2=True,
    blocking_params=None,
    spill_reload=False,
    use_scale_packing=False,
    bias=False,
    clamp_limits=None,
    prequantize_weights=False,
):
    T, H, I_TP, E, B = tokens, hidden, intermediate, expert, block_size
    N = _get_n_blocks(T, top_k, E, B)

    dma_skip = SkipMode(True, False)
    token_experts, token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts(
        T, top_k, E, B, dma_skip, N
    )

    param_string = f"{T}_{top_k}_{B}_{E}_{I_TP}_{H}_mxfp8"
    seed = int(hashlib.sha256(param_string.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    down_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, I_TP, H]).astype(dtype)
    gate_and_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, H, 2, I_TP]).astype(dtype)

    expert_affinities_masked = (np.random.random_sample([T, E]) * token_experts).astype(dtype)
    hidden_states = np.random.random_sample([T, H]).astype(dtype)
    grad_output = np.random.uniform(-1.0, 1.0, size=[T, H]).astype(dtype)

    gate_up_proj_bias = None
    down_proj_bias = None
    if bias:
        gate_up_proj_bias = np.random.uniform(-0.1, 0.1, size=[E, 2, I_TP]).astype(dtype)
        down_proj_bias = np.random.uniform(-0.1, 0.1, size=[E, H]).astype(dtype)

    fwd_clamp_limits = clamp_limits if clamp_limits is not None else ClampLimits()
    _, gate_up_proj_act_checkpoint_T, _ = _generate_fwd_golden(
        expert_affinities=expert_affinities_masked,
        down_proj_weights=down_proj_weights,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_and_up_proj_weights=gate_and_up_proj_weights,
        hidden_states=hidden_states,
        T=T,
        H=H,
        B=B,
        N=N,
        E=E,
        I_TP=I_TP,
        dtype=dtype,
        dma_skip=dma_skip,
        activation_function=ActFnType.SiLU,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj_bias=down_proj_bias,
        clamp_limits=fwd_clamp_limits,
    )

    if prequantize_weights:
        gate_up_data, gate_up_scales, down_data, down_scales = prequantize_moe_weights(
            gate_and_up_proj_weights, down_proj_weights, use_scale_packing
        )
    else:
        gate_up_data = gate_and_up_proj_weights
        gate_up_scales = None
        down_data = down_proj_weights
        down_scales = None

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
        "gate_up_proj_weight": gate_up_data,
        "down_proj_weight": down_data,
        "gate_up_proj_act_checkpoint_T": gate_up_proj_act_checkpoint_T,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert.reshape(-1, 1),
        "output_hidden_states_grad": grad_output,
        "block_size": block_size,
        "run_with_lnc2": run_with_lnc2,
        "affinity_option": AffinityOption.AFFINITY_ON_I,
        "shard_option": ShardOption.SHARD_ON_FREE,
        "activation_type": ActFnType.SiLU,
        "is_tensor_update_accumulating": top_k != 1,
        "spill_reload": spill_reload,
        "use_scale_packing": use_scale_packing,
        "skip_dma": SkipMode(True, False),
        "bias": bias,
        "clamp_limits": clamp_limits,
    }

    if gate_up_scales is not None:
        inputs["gate_up_weight_scales"] = gate_up_scales
    if down_scales is not None:
        inputs["down_weight_scales"] = down_scales

    if blocking_params is not None:
        inputs["phase1_config"] = blocking_params.phase1
        inputs["phase2_config"] = blocking_params.phase2
        inputs["phase3_config"] = blocking_params.phase3
        inputs["phase4_config"] = blocking_params.phase4

    if prequantize_weights:
        return inputs, gate_and_up_proj_weights, down_proj_weights

    return inputs
