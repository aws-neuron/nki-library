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
"""PyTorch reference implementation for blockwise_mm_bwd kernel."""

import numpy as np
import torch
from scipy.special import expit

from .moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    KernelTypeOption,
    ShardOption,
    SkipMode,
)

# ─── Activation helpers (inlined from test utilities) ───────────────────────


def _silu(x: np.ndarray):
    """silu(x) = x * sigmoid(x)."""
    return x * expit(x)


def _gelu_apprx_sigmoid(x: np.ndarray):
    """Approximate GELU using sigmoid: x * sigmoid(1.702 * x)."""
    return x * (1 / (1 + np.exp(-1.702 * x)))


def _gelu_apprx_sigmoid_dx(x: np.ndarray):
    """Numerically stable derivative of _gelu_apprx_sigmoid."""
    X = 1.702 * x
    S0 = expit(X)
    S1 = expit(-X)
    return S0 * (1.0 + X * S1)


# ─── Backward golden computation ───────────────────────────────────────────


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
    affinity_option=AffinityOption.AFFINITY_ON_H,
    skip_gate_proj: bool = False,
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

    is_affinity_i = affinity_option == AffinityOption.AFFINITY_ON_I

    for block_idx in range(N):
        token_position_to_id = block_to_token_indices[block_idx]
        block_expert_idx = block_to_expert[block_idx]
        block_hidden_states = hidden_states[token_position_to_id]
        block_grad = grad_output[token_position_to_id]

        gate_up_activation_T = gate_up_activations_T[block_idx]
        gate_activation_T, up_activation_T = np.split(gate_up_activation_T, 2, axis=0)
        gate_activation, up_activation = gate_activation_T.squeeze(0).T, up_activation_T.squeeze(0).T

        if skip_gate_proj:
            if activation_function == ActFnType.SiLU:
                first_dot_activation = _silu(up_activation)
            elif activation_function == ActFnType.Swish:
                first_dot_activation = _gelu_apprx_sigmoid(up_activation)
            elif activation_function == ActFnType.SquaredReLU:
                first_dot_activation = np.maximum(up_activation, 0) ** 2
        else:
            if activation_function == ActFnType.SiLU:
                silu_activation = _silu(gate_activation)
            elif activation_function == ActFnType.Swish:
                silu_activation = _gelu_apprx_sigmoid(gate_activation)
            elif activation_function == ActFnType.SquaredReLU:
                silu_activation = np.maximum(gate_activation, 0) ** 2
            first_dot_activation = silu_activation * up_activation
        ea = expert_affinities_masked[token_position_to_id, block_expert_idx][:, np.newaxis]

        if is_affinity_i:
            down_out_grad = block_grad
            scaled_first_dot_activation = first_dot_activation * ea
            block_down_weight_grad = scaled_first_dot_activation.T @ down_out_grad
        else:
            down_activation = down_activations[block_idx]
            mul = block_grad.astype(np.float32) * down_activation.astype(np.float32)
            affinities_grad[token_position_to_id, block_expert_idx] = np.sum(mul, axis=1)
            down_out_grad = block_grad * ea
            block_down_weight_grad = first_dot_activation.T @ down_out_grad

        if down_bias_grad is not None:
            bias_grad_src = block_grad * ea if is_affinity_i else down_out_grad
            block_down_bias_grad = np.sum(bias_grad_src.astype(np.float32), axis=0)
            down_bias_grad[block_expert_idx] += block_down_bias_grad

        down_weight_grad[block_expert_idx] += block_down_weight_grad
        first_dot_grad = down_out_grad @ down_weight[block_expert_idx].T

        if is_affinity_i:
            affinities_grad[token_position_to_id, block_expert_idx] = np.sum(
                first_dot_grad.astype(np.float32) * first_dot_activation.astype(np.float32), axis=1
            )
            first_dot_grad = first_dot_grad * ea

        if skip_gate_proj:
            if activation_function == ActFnType.SiLU:
                up_output_grad = first_dot_grad * (
                    expit(up_activation) * (1 + up_activation * (1 - expit(up_activation)))
                )
            elif activation_function == ActFnType.Swish:
                up_output_grad = first_dot_grad * _gelu_apprx_sigmoid_dx(up_activation)
            elif activation_function == ActFnType.SquaredReLU:
                up_output_grad = first_dot_grad * 2 * np.maximum(up_activation, 0)
            gate_output_grad = np.zeros_like(up_output_grad)
            gate_up_out_grad = np.concatenate([gate_output_grad, up_output_grad], axis=-1)
        else:
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
                gate_output_grad = silu_grad * _gelu_apprx_sigmoid_dx(gate_activation) * gate_activation_clamp_grad
            elif activation_function == ActFnType.SquaredReLU:
                gate_output_grad = silu_grad * 2 * np.maximum(gate_activation, 0) * gate_activation_clamp_grad

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


# ─── Dispatch-ready torch_ref ──────────────────────────────────────────────


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
    skip_grad_initialization: bool = False,
    shard_option: ShardOption = ShardOption.SHARD_ON_FREE,
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H,
    kernel_type_option: KernelTypeOption = KernelTypeOption.DROPLESS,
    clamp_limits: ClampLimits = None,
    bias: bool = False,
    activation_type: ActFnType = ActFnType.SiLU,
    block_tile_size: int = None,
    blocking_params=None,
    hidden_states_grad_out=None,
    expert_affinities_masked_grad_out=None,
    gate_up_proj_weight_grad_out=None,
    down_proj_weight_grad_out=None,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
) -> dict:
    """Torch reference for blockwise_mm_bwd.

    Converts torch tensors to numpy, runs the backward golden computation,
    and returns gradient tensors as a dict.
    """
    if skip_dma is None:
        skip_dma = SkipMode(False, False)
    if clamp_limits is None:
        clamp_limits = ClampLimits()

    # Parameters consumed only by NKI kernel scheduling/allocation logic
    _ = compute_dtype, is_tensor_update_accumulating, skip_grad_initialization
    _ = shard_option, kernel_type_option, block_tile_size, blocking_params
    _ = hidden_states_grad_out, expert_affinities_masked_grad_out
    _ = gate_up_proj_weight_grad_out, down_proj_weight_grad_out, accumulation_dtype

    # Convert torch tensors to numpy
    hidden_np = hidden_states.numpy()
    expert_aff_np = expert_affinities_masked.numpy()
    gate_up_w_np = gate_up_proj_weight.numpy()
    down_w_np = down_proj_weight.numpy()
    gate_up_act_np = gate_up_proj_act_checkpoint_T.numpy()
    down_act_np = down_proj_act_checkpoint.numpy() if down_proj_act_checkpoint is not None else None
    tok_pos_np = token_position_to_id.numpy()
    blk_exp_np = block_to_expert.numpy()
    grad_out_np = output_hidden_states_grad.numpy()

    E = down_w_np.shape[0]
    expert_aff_2d = expert_aff_np.reshape(-1, E)
    N = tok_pos_np.shape[0] // block_size

    # For bias gradient computation, create dummy bias arrays of correct shape
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
        affinity_option=affinity_option,
        skip_gate_proj=skip_gate_proj,
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
