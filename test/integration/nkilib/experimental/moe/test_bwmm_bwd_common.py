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

"""Test utilities for blockwise MM backward tests."""

import hashlib
import math

import numpy as np
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ShardOption,
    SkipMode,
)

from test.integration.nkilib.utils.test_kernel_common import gelu_apprx_sigmoid, silu


def map_skip_mode(skip_mode: int) -> SkipMode:
    if skip_mode == 0:
        return SkipMode(False, False)
    elif skip_mode == 1:
        return SkipMode(True, False)
    elif skip_mode == 2:
        return SkipMode(False, True)
    elif skip_mode == 3:
        return SkipMode(True, True)
    else:
        raise ValueError("Invalid skip_mode")


def get_n_blocks(T, TOPK, E, B, n_block_per_iter=1):
    N = math.ceil((T * TOPK - (E - 1)) / B) + E - 1
    N = n_block_per_iter * math.ceil(N / n_block_per_iter)
    return N


def get_router_with_controlled_distribution(T: int, TOPK: int, E: int, alpha=None):
    router = np.zeros((T, TOPK))
    np.random.seed(0)
    if alpha is None:
        for i in range(T):
            router[i] = np.random.choice(range(E), (TOPK), replace=False)
    elif alpha == -1:
        p_of_e = np.random.dirichlet(np.ones(E) * 0.001)
        for i in range(0, T):
            router[i] = np.random.choice(range(E), (TOPK), replace=False, p=p_of_e)
    else:
        for i in range(E):
            router[i] = np.arange(i, (i + TOPK) % T)
        p_of_e = np.random.dirichlet(np.ones(E) * alpha)
        for i in range(E, T):
            router[i] = np.random.choice(range(E), (TOPK), replace=False, p=p_of_e)
    return router


def generate_token_position_to_id_and_experts(T, TOPK, E, B, dma_skip: SkipMode, N: int):
    router = get_router_with_controlled_distribution(T=T, TOPK=TOPK, E=E, alpha=None)
    one_hot = np.arange(E)
    token_experts = np.zeros((T, E))
    for i in range(TOPK):
        token_experts += np.expand_dims(router[:, i], 1) == np.expand_dims(one_hot, 0)

    blocks_per_expert = np.ceil(token_experts.sum(0) / B).astype(np.int32)
    n_padding_block = N - np.sum(blocks_per_expert)
    blocks_per_expert[E - 1] += n_padding_block

    cumulative_blocks_per_expert = np.cumsum(blocks_per_expert)
    block_to_expert = np.arange(E).repeat(blocks_per_expert).astype(np.int32)

    token_position_by_id_and_expert = np.cumsum(token_experts, axis=0)
    expert_block_offsets = cumulative_blocks_per_expert * B
    token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
    token_position_by_id_and_expert = np.where(token_experts, token_position_by_id_and_expert, 0).astype(np.int32)

    if dma_skip.skip_token:
        token_position_to_id = np.full((int(N * B + 1),), -1)
    else:
        token_position_to_id = np.full((int(N * B + 1),), T)
    tokens_ids = np.arange(T)
    token_position_to_id[token_position_by_id_and_expert] = np.expand_dims(tokens_ids, 1)
    token_position_to_id = token_position_to_id[1:]
    token_position_to_id = token_position_to_id.astype(np.int32)

    return token_experts, token_position_to_id, block_to_expert


def _generate_fwd_golden(
    expert_affinities,
    down_proj_weights,
    token_position_to_id,
    block_to_expert,
    gate_and_up_proj_weights,
    hidden_states,
    T,
    H,
    B,
    N,
    E,
    I_TP,
    dtype,
    dma_skip,
    activation_function,
    gate_up_proj_bias,
    down_proj_bias,
    clamp_limits,
    skip_gate_proj: bool = False,
):
    output_np = np.zeros([T + 1, H]).astype(dtype)
    token_position_to_id = token_position_to_id.reshape(N, B)
    gate_up_activations_T = np.zeros([N, 2, I_TP, B]).astype(dtype)
    down_activations = np.zeros([N, B, H]).astype(dtype)

    E_local = gate_and_up_proj_weights.shape[0]
    gate_and_up_proj_weights = gate_and_up_proj_weights.reshape(E_local, H, 2 * I_TP).astype(np.float32)
    down_proj_weights = down_proj_weights.astype(np.float32)

    for b in range(N):
        local_token_position_to_id = token_position_to_id[b, :]
        if dma_skip.skip_token:
            zeros_hidden = np.zeros((1, H)).astype(dtype)
            hidden_states = np.concatenate([hidden_states, zeros_hidden], axis=0)
            zeros_exaf = np.zeros((1, E)).astype(dtype)
            expert_affinities = np.concatenate([expert_affinities, zeros_exaf], axis=0)

        local_hidden_states = hidden_states[local_token_position_to_id[:], :].astype(np.float32)
        expert_idx = block_to_expert[b]
        local_expert_affinities = expert_affinities[local_token_position_to_id, expert_idx].reshape(-1, 1).astype(dtype)

        gate_up_weights = gate_and_up_proj_weights[expert_idx]
        down_weights = down_proj_weights[expert_idx, :, :]

        gate_up_activation = np.matmul(local_hidden_states, gate_up_weights).reshape(B, 2, I_TP)
        gate_activation = gate_up_activation[:, 0, :]
        up_activation = gate_up_activation[:, 1, :]

        if gate_up_proj_bias is not None:
            gate_activation += gate_up_proj_bias[expert_idx, 0, :]
            up_activation += gate_up_proj_bias[expert_idx, 1, :]

        if (
            clamp_limits.non_linear_clamp_lower_limit is not None
            or clamp_limits.non_linear_clamp_upper_limit is not None
        ):
            np.clip(
                gate_activation,
                a_min=clamp_limits.non_linear_clamp_lower_limit,
                a_max=clamp_limits.non_linear_clamp_upper_limit,
                out=gate_activation,
            )
        if clamp_limits.linear_clamp_lower_limit is not None or clamp_limits.linear_clamp_upper_limit is not None:
            np.clip(
                up_activation,
                a_min=clamp_limits.linear_clamp_lower_limit,
                a_max=clamp_limits.linear_clamp_upper_limit,
                out=up_activation,
            )

        gate_up_activations_T[b] = gate_up_activation.transpose(1, 2, 0)

        if skip_gate_proj:
            if activation_function == ActFnType.SiLU:
                multiply_1 = silu(up_activation)
            elif activation_function == ActFnType.Swish:
                multiply_1 = gelu_apprx_sigmoid(up_activation)
            elif activation_function == ActFnType.SquaredReLU:
                multiply_1 = np.maximum(up_activation, 0) ** 2
        else:
            if activation_function == ActFnType.SiLU:
                act_res = silu(gate_activation)
            elif activation_function == ActFnType.Swish:
                act_res = gelu_apprx_sigmoid(gate_activation)
            elif activation_function == ActFnType.SquaredReLU:
                act_res = np.maximum(gate_activation, 0) ** 2
            multiply_1 = act_res * up_activation
        down_activation = np.matmul(multiply_1, down_weights)

        if down_proj_bias is not None:
            down_activation += down_proj_bias[expert_idx]

        down_activations[b] = down_activation
        scale = down_activation * local_expert_affinities
        output_np[local_token_position_to_id[:], :] += scale.astype(output_np.dtype)

    out_return = output_np[:T, :] if dma_skip.skip_token else output_np
    return out_return, gate_up_activations_T, down_activations


def build_bwmm_bwd_inputs(
    tokens,
    hidden,
    intermediate,
    expert,
    block_size,
    top_k,
    dtype,
    dma_skip,
    bias_flag,
    clamp_limits,
    activation_type,
    affinity_option=AffinityOption.AFFINITY_ON_H,
    blocking_params=None,
    shard_option=ShardOption.SHARD_ON_FREE,
    skip_gate_proj: bool = False,
):
    """Build kernel inputs and return (inputs_dict, gate_up_proj_bias, down_proj_bias)."""
    N = get_n_blocks(tokens, top_k, expert, block_size)
    expert_masks, token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts(
        tokens, top_k, expert, block_size, dma_skip, N
    )

    param_string = f"{tokens}_{top_k}_{block_size}_{expert}_{intermediate}_{hidden}_{dma_skip.skip_token}"
    seed = int(hashlib.sha256(param_string.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    down_proj_weights = np.random.uniform(-0.1, 0.1, size=[expert, intermediate, hidden]).astype(dtype)
    gate_and_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[expert, hidden, 2, intermediate]).astype(dtype)

    gate_up_proj_bias = None
    down_proj_bias = None
    if bias_flag:
        down_proj_bias = np.random.uniform(-0.1, 0.1, size=[expert, hidden]).astype(dtype)
        gate_up_proj_bias = np.random.uniform(-0.1, 0.1, size=[expert, 2, intermediate]).astype(dtype)

    if dma_skip.skip_token:
        expert_affinities_masked = (np.random.random_sample([tokens, expert]) * expert_masks).astype(dtype)
        hidden_states = np.random.random_sample([tokens, hidden]).astype(dtype)
        grad_output = np.random.uniform(-1.0, 1.0, size=[tokens, hidden]).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([tokens + 1, expert]).astype(dtype)
        expert_affinities_masked[:tokens] = expert_affinities_masked[:tokens] * expert_masks
        expert_affinities_masked[tokens] = 0
        hidden_states = np.random.random_sample([tokens + 1, hidden]).astype(dtype)
        grad_output = np.random.uniform(-1.0, 1.0, size=[tokens + 1, hidden]).astype(dtype)
        hidden_states[tokens, ...] = 0
        grad_output[tokens, ...] = 0

    _, gate_up_proj_act_checkpoint_T, down_proj_act_checkpoint = _generate_fwd_golden(
        expert_affinities_masked,
        down_proj_weights,
        token_position_to_id,
        block_to_expert,
        gate_and_up_proj_weights,
        hidden_states,
        tokens,
        hidden,
        block_size,
        N,
        expert,
        intermediate,
        dtype,
        dma_skip,
        activation_type,
        gate_up_proj_bias,
        down_proj_bias,
        clamp_limits,
        skip_gate_proj=skip_gate_proj,
    )

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
        "gate_up_proj_weight": gate_and_up_proj_weights,
        "down_proj_weight": down_proj_weights,
        "gate_up_proj_act_checkpoint_T": gate_up_proj_act_checkpoint_T,
        "down_proj_act_checkpoint": down_proj_act_checkpoint,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert,
        "output_hidden_states_grad": grad_output,
        "block_size": block_size,
        "skip_dma": dma_skip,
        "compute_dtype": dtype,
        "is_tensor_update_accumulating": top_k != 1,
        "clamp_limits": clamp_limits,
        "bias": bias_flag,
        "activation_type": activation_type,
        "affinity_option": affinity_option,
        "blocking_params": blocking_params,
        "shard_option": shard_option,
        "skip_gate_proj": skip_gate_proj,
    }

    if affinity_option == AffinityOption.AFFINITY_ON_I:
        inputs["down_proj_act_checkpoint"] = None

    return inputs, gate_up_proj_bias, down_proj_bias
