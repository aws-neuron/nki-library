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

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.bwd.config import MXFP8MOEBwdConfig
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch import (
    _dynamic_checkpoint_extent,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    _generate_fwd_golden,
    generate_token_position_to_id_and_experts,
)
from test.utils.mx_utils import quantize_mx_golden

# ============================================================================
# Routing table generation
# ============================================================================


def _get_n_blocks(T, top_k, E, B):
    N = math.ceil((T * top_k - (E - 1)) / B) + E - 1
    return N


# ============================================================================
# Pre-quantization utilities
# ============================================================================


def _prequantize_2d(tensor_2d, quant_scheme=QuantScheme.WRAPX):
    """Pre-quantize a single 2D [K, F] tensor to MXFP8 x4 format.

    Steps:
      1. Swizzle [K, F] → [K/4, F*4]
      2. Quantize → (x4_data [K/4, F], compact_scales)
      3. Lay the compact scales out for HBM (dense [K // 32, F])

    Returns (data, scales).
    """
    K, F = tensor_2d.shape
    swizzle = matmul_utils.swizzle_tensor_1x32 if quant_scheme == QuantScheme._1x32 else matmul_utils.swizzle_tensor
    swizzled = swizzle(tensor_2d)
    fp8_x4_dtype = nl.float8_e4m3fn_x4
    data, compact_scales = quantize_mx_golden(swizzled, fp8_x4_dtype)
    return data, matmul_utils.to_hbm_scales(compact_scales)


def prequantize_moe_weights(
    gate_up_proj_weight,
    down_proj_weight,
    quant_scheme=QuantScheme.WRAPX,
):
    """Pre-quantize MoE weights per-expert for the backward kernel.

    gate_up_proj_weight: [E, H, 2, I_TP] → per-expert [2*I_TP, H] → x4 [2*I_TP/4, H]
    down_proj_weight: [E, I_TP, H] → per-expert [H, I_TP] → x4 [H/4, I_TP]

    Returns:
        (gate_up_data, gate_up_scales, down_data, down_scales)
        gate_up_data: [E, 2*I_TP//4, H]
        gate_up_scales: [E, 2*I_TP//32, H]
        down_data: [E, H//4, I_TP]
        down_scales: [E, H//32, I_TP]
    """
    E, H, _, I_TP = gate_up_proj_weight.shape

    gate_up_data_list = []
    gate_up_scales_list = []
    down_data_list = []
    down_scales_list = []

    for e in range(E):
        # gate_up: [H, 2, I_TP] → reshape to [H, 2*I_TP] → transpose to [2*I_TP, H]
        gate_up_2d = gate_up_proj_weight[e].reshape(H, 2 * I_TP).T.astype(np.float32)
        gu_data, gu_scales = _prequantize_2d(gate_up_2d, quant_scheme)
        gate_up_data_list.append(gu_data)
        gate_up_scales_list.append(gu_scales)

        # down: [I_TP, H] → transpose to [H, I_TP]
        down_2d = down_proj_weight[e].T.astype(np.float32)
        d_data, d_scales = _prequantize_2d(down_2d, quant_scheme)
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
    moe_bwd_config=None,
    spill_reload=False,
    use_scale_packing=False,
    bias=False,
    clamp_limits=None,
    prequantize_weights=False,
    single_expert_dense=False,
    fast_dma_transpose=False,
    swizzle_modes=None,
    pe_transpose_only=False,
    quant_scheme=QuantScheme.WRAPX,
    num_valid_tokens=None,
    num_static_block=None,
    gate_up_weight_orientation=None,
    down_weight_orientation=None,
):
    T, H, I_TP, E, B = tokens, hidden, intermediate, expert, block_size
    base = moe_bwd_config if moe_bwd_config is not None else MXFP8MOEBwdConfig()
    if gate_up_weight_orientation is None:
        gate_up_weight_orientation = (
            base.gate_up_weight_td.orientation if base.gate_up_weight_td is not None else TensorOrientation.F_BY_K
        )
    if down_weight_orientation is None:
        down_weight_orientation = (
            base.down_weight_td.orientation if base.down_weight_td is not None else TensorOrientation.F_BY_K
        )
    if single_expert_dense:
        assert E == 1, f"single_expert_dense requires expert=1, got {E}"
        assert top_k == 1, f"single_expert_dense requires top_k=1, got {top_k}"
        assert T % B == 0, f"single_expert_dense requires tokens divisible by block_size, got {T} and {B}"
    N = _get_n_blocks(T, top_k, E, B)

    skip_dma = SkipMode(True, False)
    # The routed tables are always generated: even on the dense path they seed the
    # forward golden that produces the activation checkpoint. For E=1 / top_k=1 /
    # T == N*B they are the identity token map with every block on expert 0, i.e.
    # exactly the contiguous order the dense kernel assumes.
    token_experts, token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts(
        T, top_k, E, B, skip_dma, N
    )

    param_string = f"{T}_{top_k}_{B}_{E}_{I_TP}_{H}_mxfp8"
    seed = int(hashlib.sha256(param_string.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    down_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, I_TP, H]).astype(dtype)
    gate_and_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, H, 2, I_TP]).astype(dtype)

    expert_affinities_masked = (np.random.random_sample([T, E]) * token_experts).astype(dtype)
    valid_tokens = None
    if num_valid_tokens is not None:
        assert single_expert_dense, "num_valid_tokens is only supported on the single_expert_dense path"
        assert num_valid_tokens > 0, f"num_valid_tokens must be positive, got {num_valid_tokens}"
        valid_tokens = np.array([num_valid_tokens], dtype=np.int32)
        # With zero affinity a padding token contributes nothing to the weight gradients
        # or to d_hidden, so the reference matches the kernel on the blocks it skips.
        # d_affinity carries no such factor, which is why it is compared masked.
        expert_affinities_masked[min(num_valid_tokens, T) :] = 0
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
        dma_skip=skip_dma,
        activation_function=ActFnType.SiLU,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj_bias=down_proj_bias,
        clamp_limits=fwd_clamp_limits,
    )
    if num_valid_tokens is not None:
        produced_blocks, shortened_rows = _dynamic_checkpoint_extent(
            T,
            B,
            valid_tokens,
            num_static_block,
        )
        if shortened_rows is not None:
            gate_up_proj_act_checkpoint_T[produced_blocks - 1, :, :, shortened_rows:] = 0

    # The shared BF16 forward golden emits [N, 2, I_TP, B]; the MXFP8 forward
    # and backward checkpoint contract is token-major [N, 2, B, I_TP].
    gate_up_proj_act_checkpoint_T = np.ascontiguousarray(gate_up_proj_act_checkpoint_T.transpose(0, 1, 3, 2))

    if prequantize_weights:
        assert gate_up_weight_orientation == TensorOrientation.F_BY_K
        assert down_weight_orientation == TensorOrientation.F_BY_K
        gate_up_data, gate_up_scales, down_data, down_scales = prequantize_moe_weights(
            gate_and_up_proj_weights,
            down_proj_weights,
            quant_scheme,
        )
    else:
        gate_up_data = (
            np.ascontiguousarray(gate_and_up_proj_weights.transpose(0, 3, 2, 1))
            if gate_up_weight_orientation == TensorOrientation.K_BY_F
            else gate_and_up_proj_weights
        )
        gate_up_scales = None
        down_data = (
            np.ascontiguousarray(down_proj_weights.transpose(0, 2, 1))
            if down_weight_orientation == TensorOrientation.K_BY_F
            else down_proj_weights
        )
        down_scales = None

    # Collapse every non-tensor knob into the fused backward config. Phase configs and
    # transpose modes come from ``base``; swizzle modes come from ``swizzle_modes``.
    # spill/reload and scale packing are per-phase settings, not config-level ones.
    for phase_config in (base.phase1_config, base.phase2_config, base.phase3_config, base.phase4_config):
        phase_config.spill_reload = spill_reload
        phase_config.enable_scale_packing = use_scale_packing
    config = MXFP8MOEBwdConfig(
        affinity_option=AffinityOption.AFFINITY_ON_I,
        shard_option=ShardOption.SHARD_ON_FREE,
        activation_type=ActFnType.SiLU,
        accumulate_hidden_states_grad=top_k != 1,
        execution_mode=(ExecutionMode.DENSE_MOE if single_expert_dense else ExecutionMode.ROUTED_MOE),
        num_static_block=base.num_static_block if num_static_block is None else num_static_block,
        fast_dma_transpose=fast_dma_transpose,
        pe_transpose_only=pe_transpose_only,
        skip_dma=skip_dma,
        bias=bias,
        clamp_limits=clamp_limits if clamp_limits is not None else ClampLimits(),
        phase1_config=base.phase1_config,
        phase2_config=base.phase2_config,
        phase3_config=base.phase3_config,
        phase4_config=base.phase4_config,
        phase3_transpose_mode=base.phase3_transpose_mode,
        phase4_transpose_mode=base.phase4_transpose_mode,
        gate_up_weight_td=(
            TensorDescriptor(orientation=TensorOrientation.K_BY_F)
            if gate_up_weight_orientation == TensorOrientation.K_BY_F
            else None
        ),
        down_weight_td=(
            TensorDescriptor(orientation=TensorOrientation.K_BY_F)
            if down_weight_orientation == TensorOrientation.K_BY_F
            else None
        ),
        **(swizzle_modes or {}),
    )

    # Routing tensors as the kernel wants them: single-entry dummies on the dense
    # path (which never reads them), full-size tables otherwise. Same contract as
    # build_mxfp8_moe_fwd_inputs, so one pair feeds a chained fwd+bwd call.
    if single_expert_dense:
        kernel_token_position_to_id = np.zeros((1,), dtype=np.int32)
        kernel_block_to_expert = np.zeros((1, 1), dtype=np.int32)
    else:
        kernel_token_position_to_id = token_position_to_id
        kernel_block_to_expert = block_to_expert.reshape(-1, 1)

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
        "gate_up_proj_weight": gate_up_data,
        "down_proj_weight": down_data,
        "gate_up_proj_act_checkpoint_T": gate_up_proj_act_checkpoint_T,
        "token_position_to_id": kernel_token_position_to_id,
        "block_to_expert": kernel_block_to_expert,
        "output_hidden_states_grad": grad_output,
        "block_size": block_size,
        "config": config,
    }

    if gate_up_scales is not None:
        inputs["gate_up_weight_scales"] = gate_up_scales
    if down_scales is not None:
        inputs["down_weight_scales"] = down_scales
    if valid_tokens is not None:
        inputs["num_valid_tokens"] = valid_tokens

    if prequantize_weights:
        return inputs, gate_and_up_proj_weights, down_proj_weights

    return inputs
