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
from test.integration.nkilib.utils.test_kernel_common import gelu_apprx_sigmoid, gelu_apprx_sigmoid_dx, silu

import numpy as np
import torch
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    KernelTypeOption,
    ShardOption,
    SkipMode,
)
from scipy.special import expit


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

        if activation_function == ActFnType.SiLU:
            act_res = silu(gate_activation)
        elif activation_function == ActFnType.Swish:
            act_res = gelu_apprx_sigmoid(gate_activation)

        multiply_1 = act_res * up_activation
        down_activation = np.matmul(multiply_1, down_weights)

        if down_proj_bias is not None:
            down_activation += down_proj_bias[expert_idx]

        down_activations[b] = down_activation
        scale = down_activation * local_expert_affinities
        output_np[local_token_position_to_id[:], :] += scale.astype(output_np.dtype)

    out_return = output_np[:T, :] if dma_skip.skip_token else output_np
    return out_return, gate_up_activations_T, down_activations


def _generate_bwd_golden(
    grad_output,
    hidden_states,
    expert_affinities_masked,
    block_to_token_indices,
    block_to_expert,
    gate_up_weight,
    down_weight,
    gate_up_activations_T,
    down_activations,
    N,
    B,
    dma_skip,
    activation_function,
    clamp_limits,
    gate_up_bias,
    down_bias,
):
    E, I, H = down_weight.shape
    T, _ = hidden_states.shape

    if dma_skip.skip_token:
        zeros_hidden = np.zeros((1, H)).astype(hidden_states.dtype)
        hidden_states = np.concatenate([hidden_states, zeros_hidden], axis=0)
        zeros_exaf = np.zeros((1, E)).astype(expert_affinities_masked.dtype)
        expert_affinities_masked = np.concatenate([expert_affinities_masked, zeros_exaf], axis=0)
        zeros_gradout = np.zeros((1, H)).astype(grad_output.dtype)
        grad_output = np.concatenate([grad_output, zeros_gradout], axis=0)

    hidden_states_grad = np.zeros_like(hidden_states)
    affinities_grad = np.zeros_like(expert_affinities_masked)
    down_weight_grad = np.zeros_like(down_weight)
    down_bias_grad = np.zeros_like(down_bias) if down_bias is not None else None
    gate_up_weight_grad = np.zeros_like(gate_up_weight)
    gate_up_bias_grad = np.zeros_like(gate_up_bias) if gate_up_bias is not None else None
    block_to_token_indices = block_to_token_indices.reshape(N, B)

    for block_idx in range(N):
        token_position_to_id = block_to_token_indices[block_idx]
        block_expert_idx = block_to_expert[block_idx]
        block_hidden_states = hidden_states[token_position_to_id]
        block_grad = grad_output[token_position_to_id]
        down_activation = down_activations[block_idx]

        mul = block_grad.astype(np.float32) * down_activation.astype(np.float32)
        affinities_grad[token_position_to_id, block_expert_idx] = np.sum(mul, axis=1)
        down_out_grad = block_grad * expert_affinities_masked[token_position_to_id, block_expert_idx][:, np.newaxis]

        gate_up_activation_T = gate_up_activations_T[block_idx]
        gate_activation_T, up_activation_T = np.split(gate_up_activation_T, 2, axis=0)
        gate_activation, up_activation = gate_activation_T.squeeze(0).T, up_activation_T.squeeze(0).T

        if activation_function == ActFnType.SiLU:
            silu_activation = silu(gate_activation)
        elif activation_function == ActFnType.Swish:
            silu_activation = gelu_apprx_sigmoid(gate_activation)

        first_dot_activation = silu_activation * up_activation
        block_down_weight_grad = first_dot_activation.T @ down_out_grad

        if down_bias_grad is not None:
            block_down_bias_grad = np.sum(down_out_grad.astype(np.float32), axis=0)
            down_bias_grad[block_expert_idx] += block_down_bias_grad

        down_weight_grad[block_expert_idx] += block_down_weight_grad
        first_dot_grad = down_out_grad @ down_weight[block_expert_idx].T
        silu_grad = first_dot_grad * up_activation

        if (
            clamp_limits.non_linear_clamp_lower_limit is not None
            or clamp_limits.non_linear_clamp_upper_limit is not None
        ):
            mask_lower = (
                gate_activation > clamp_limits.non_linear_clamp_lower_limit
                if clamp_limits.non_linear_clamp_lower_limit is not None
                else np.ones_like(gate_activation, dtype=bool)
            )
            mask_upper = (
                gate_activation < clamp_limits.non_linear_clamp_upper_limit
                if clamp_limits.non_linear_clamp_upper_limit is not None
                else np.ones_like(gate_activation, dtype=bool)
            )
            gate_activation_clamp_grad = (mask_lower & mask_upper).astype(np.float32)
        else:
            gate_activation_clamp_grad = np.ones_like(gate_activation)

        if clamp_limits.linear_clamp_lower_limit is not None or clamp_limits.linear_clamp_upper_limit is not None:
            mask_lower = (
                up_activation > clamp_limits.linear_clamp_lower_limit
                if clamp_limits.linear_clamp_lower_limit is not None
                else np.ones_like(up_activation, dtype=bool)
            )
            mask_upper = (
                up_activation < clamp_limits.linear_clamp_upper_limit
                if clamp_limits.linear_clamp_upper_limit is not None
                else np.ones_like(up_activation, dtype=bool)
            )
            up_activation_clamp_grad = (mask_lower & mask_upper).astype(np.float32)
        else:
            up_activation_clamp_grad = np.ones_like(up_activation)

        if activation_function == ActFnType.SiLU:
            gate_output_grad = (
                silu_grad
                * expit(gate_activation)
                * (1 + gate_activation * (1 - expit(gate_activation)))
                * gate_activation_clamp_grad
            )
        elif activation_function == ActFnType.Swish:
            gate_output_grad = silu_grad * gelu_apprx_sigmoid_dx(gate_activation) * gate_activation_clamp_grad

        up_output_grad = first_dot_grad * silu_activation * up_activation_clamp_grad
        gate_up_out_grad = np.concatenate([gate_output_grad, up_output_grad], axis=-1)

        block_gate_up_grad = block_hidden_states.T @ gate_up_out_grad
        if gate_up_bias_grad is not None:
            block_gate_up_bias_grad = np.sum(gate_up_out_grad.astype(np.float32), axis=0).reshape(2, I)
            gate_up_bias_grad[block_expert_idx] += block_gate_up_bias_grad

        gate_up_weight_grad[block_expert_idx] += block_gate_up_grad.reshape(H, 2, I)
        block_hidden_grad = gate_up_out_grad @ gate_up_weight[block_expert_idx].reshape(H, 2 * I).T
        hidden_states_grad[token_position_to_id] += block_hidden_grad

    if dma_skip.skip_token:
        return (
            hidden_states_grad[:T, :],
            affinities_grad[:T, :],
            gate_up_weight_grad,
            down_weight_grad,
            gate_up_bias_grad,
            down_bias_grad,
        )
    else:
        return (
            hidden_states_grad,
            affinities_grad,
            gate_up_weight_grad,
            down_weight_grad,
            gate_up_bias_grad,
            down_bias_grad,
        )


def build_bwmm_bwd_inputs(
    tokens, hidden, intermediate, expert, block_size, top_k, dtype, dma_skip, bias_flag, clamp_limits, activation_type
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
    }

    return inputs, gate_up_proj_bias, down_proj_bias


def blockwise_mm_bwd_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_proj_act_checkpoint_T: torch.Tensor,
    down_proj_act_checkpoint: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    output_hidden_states_grad: torch.Tensor,
    block_size: int,
    skip_dma: SkipMode = None,
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    shard_option: ShardOption = ShardOption.SHARD_ON_HIDDEN,
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H,
    kernel_type_option: KernelTypeOption = KernelTypeOption.DROPLESS,
    clamp_limits: ClampLimits = None,
    bias: bool = False,
    activation_type: ActFnType = ActFnType.SiLU,
    block_tile_size: int = None,
) -> dict:
    """Torch reference for blockwise_mm_bwd. Converts to numpy, runs golden, returns dict of torch tensors."""
    if skip_dma is None:
        skip_dma = SkipMode(False, False)
    if clamp_limits is None:
        clamp_limits = ClampLimits()

    # Convert torch tensors to numpy
    hidden_np = hidden_states.numpy()
    expert_aff_np = expert_affinities_masked.numpy()
    gate_up_w_np = gate_up_proj_weight.numpy()
    down_w_np = down_proj_weight.numpy()
    gate_up_act_np = gate_up_proj_act_checkpoint_T.numpy()
    down_act_np = down_proj_act_checkpoint.numpy()
    tok_pos_np = token_position_to_id.numpy()
    blk_exp_np = block_to_expert.numpy()
    grad_out_np = output_hidden_states_grad.numpy()

    E = down_w_np.shape[0]
    expert_aff_2d = expert_aff_np.reshape(-1, E)
    N = tok_pos_np.shape[0] // block_size

    # For bias gradient computation, create dummy bias arrays of correct shape
    # (bwd golden only uses bias shape for allocating grad arrays, not bias values)
    gate_up_bias = None
    down_bias = None
    if bias:
        _, H_dim, _, I_dim = gate_up_w_np.shape
        gate_up_bias = np.zeros((E, 2, I_dim), dtype=gate_up_w_np.dtype)
        down_bias = np.zeros((E, H_dim), dtype=down_w_np.dtype)

    (hidden_grad, aff_grad, gate_up_w_grad, down_w_grad, gate_up_bias_grad, down_bias_grad) = _generate_bwd_golden(
        grad_output=grad_out_np,
        hidden_states=hidden_np,
        expert_affinities_masked=expert_aff_2d,
        block_to_token_indices=tok_pos_np,
        block_to_expert=blk_exp_np,
        gate_up_weight=gate_up_w_np,
        down_weight=down_w_np,
        gate_up_activations_T=gate_up_act_np,
        down_activations=down_act_np,
        N=N,
        B=block_size,
        dma_skip=skip_dma,
        activation_function=activation_type,
        clamp_limits=clamp_limits,
        gate_up_bias=gate_up_bias,
        down_bias=down_bias,
    )

    result = {
        "hidden_states_grad": torch.from_numpy(hidden_grad),
        "expert_affinities_masked_grad": torch.from_numpy(aff_grad.reshape(-1, 1)),
        "gate_up_proj_weight_grad": torch.from_numpy(gate_up_w_grad),
        "down_proj_weight_grad": torch.from_numpy(down_w_grad),
    }

    if bias:
        result["gate_and_up_proj_bias_grad"] = torch.from_numpy(gate_up_bias_grad)
        result["down_proj_bias_grad"] = torch.from_numpy(down_bias_grad)

    return result
