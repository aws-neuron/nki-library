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

import hashlib
import math
from test.integration.nkilib.utils.test_kernel_common import gelu_apprx_sigmoid, gelu_apprx_sigmoid_dx, silu
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.moe.bwd.blockwise_mm_backward import (
    blockwise_mm_bwd,
)
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import ActFnType, ClampLimits, SkipMode
from scipy.special import expit

bfloat16 = nl.bfloat16
float32 = np.float32

# Dimension name constants
BWMM_BWD_CONFIG = "cfg"
HIDDEN_DIM_NAME = "hid"
TOKENS_DIM_NAME = "tok"
EXPERT_DIM_NAME = "exp"
BLOCK_SIZE_DIM_NAME = "bs"
TOP_K_DIM_NAME = "k"
INTERMEDIATE_DIM_NAME = "int"
DTYPE_DIM_NAME = "dt"
SKIP_DIM_NAME = "sk"
CLAMP_LIMITS_DIM_NAME = "cl"
BIAS_DIM_NAME = "bi"
ACTIVATION_TYPE_DIM_NAME = "act"


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


def get_router_with_controlled_distribution(T: int, TOPK: int, E: int, alpha: np.float32 = None):
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


def generate_blockwise_numpy_golden_fwd(
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
    dma_skip: SkipMode,
    activation_function=ActFnType.SiLU,
    gate_up_proj_bias=None,
    down_proj_bias=None,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
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

        if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
            np.clip(gate_activation, a_min=gate_clamp_lower_limit, a_max=gate_clamp_upper_limit, out=gate_activation)
        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            np.clip(up_activation, a_min=up_clamp_lower_limit, a_max=up_clamp_upper_limit, out=up_activation)

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


def generate_blockwise_bwd_numpy_golden(
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
    activation_function=ActFnType.SiLU,
    clamp_limits: ClampLimits = ClampLimits(),
    gate_up_bias=None,
    down_bias=None,
):
    E, I, H = down_weight.shape
    T, H = hidden_states.shape

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
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    top_k: int,
    dtype,
    dma_skip: SkipMode,
    bias: bool,
    clamp_limits: ClampLimits,
    activation_type: ActFnType,
):
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
    if bias:
        down_proj_bias = np.random.uniform(-0.1, 0.1, size=[expert, hidden]).astype(dtype)
        gate_up_proj_bias = np.random.uniform(-0.1, 0.1, size=[expert, 2, intermediate]).astype(dtype)

    if dma_skip.skip_token:
        expert_affinities_masked = np.random.random_sample([tokens, expert]).astype(dtype)
        expert_affinities_masked = (expert_affinities_masked * expert_masks).astype(dtype)
        hidden_states = np.random.random_sample([tokens, hidden]).astype(dtype)
        grad_output = np.random.uniform(-1.0, 1.0, size=[tokens, hidden]).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([tokens + 1, expert]).astype(dtype)
        expert_affinities_masked[:tokens] = expert_affinities_masked[:tokens] * expert_masks
        expert_affinities_masked[tokens] = 0
        hidden_states = np.random.random_sample([tokens + 1, hidden]).astype(dtype)
        grad_output = np.random.uniform(-1.0, 1.0, size=[tokens + 1, hidden]).astype(dtype)
        expert_masks = np.vstack([expert_masks, np.zeros([1, expert])])
        hidden_states[tokens, ...] = 0
        grad_output[tokens, ...] = 0

    _, gate_up_proj_act_checkpoint_T, down_proj_act_checkpoint = generate_blockwise_numpy_golden_fwd(
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
        activation_function=activation_type,
        gate_clamp_upper_limit=clamp_limits.non_linear_clamp_upper_limit,
        gate_clamp_lower_limit=clamp_limits.non_linear_clamp_lower_limit,
        up_clamp_lower_limit=clamp_limits.linear_clamp_lower_limit,
        up_clamp_upper_limit=clamp_limits.linear_clamp_upper_limit,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj_bias=down_proj_bias,
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
        "bias": bias,
        "activation_type": activation_type,
    }

    return inputs, gate_up_proj_bias, down_proj_bias


def golden_bwmm_bwd(
    inp_np: dict,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    dtype,
    dma_skip: SkipMode,
    bias: bool,
    clamp_limits: ClampLimits,
    activation_type: ActFnType,
    top_k: int,
    gate_up_proj_bias=None,
    down_proj_bias=None,
):
    N = get_n_blocks(tokens, top_k, expert, block_size)

    expert_affinities_masked = inp_np["expert_affinities_masked"].reshape(-1, expert)
    down_proj_weights = inp_np["down_proj_weight"]
    token_position_to_id = inp_np["token_position_to_id"]
    block_to_expert = inp_np["block_to_expert"]
    gate_up_proj_weights = inp_np["gate_up_proj_weight"]
    hidden_states = inp_np["hidden_states"]
    grad_output = inp_np["output_hidden_states_grad"]
    gate_up_proj_act_checkpoint_T = inp_np["gate_up_proj_act_checkpoint_T"]
    down_proj_act_checkpoint = inp_np["down_proj_act_checkpoint"]

    hidden_states_grad, affinities_grad, gate_up_weight_grad, down_weight_grad, gate_up_bias_grad, down_bias_grad = (
        generate_blockwise_bwd_numpy_golden(
            grad_output,
            hidden_states,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            gate_up_proj_weights,
            down_proj_weights,
            gate_up_proj_act_checkpoint_T,
            down_proj_act_checkpoint,
            N,
            block_size,
            dma_skip,
            activation_function=activation_type,
            clamp_limits=clamp_limits,
            gate_up_bias=gate_up_proj_bias,
            down_bias=down_proj_bias,
        )
    )

    result = {
        "hidden_states_grad": hidden_states_grad,
        "expert_affinities_masked_grad": affinities_grad.reshape(-1, 1),
        "gate_up_proj_weight_grad": gate_up_weight_grad,
        "down_proj_weight_grad": down_weight_grad,
    }

    if bias:
        result["gate_and_up_proj_bias_grad"] = gate_up_bias_grad
        result["down_proj_bias_grad"] = down_bias_grad

    return result


# fmt: off
moe_blockwise_mm_bwd_shardH_affinityH_dropless_lnc2_kernel_test_params = [
# H, T, E, B, TOPK, I_TP, dtype, skip, clamp_limits, bias, activation_type

[5120, 8192, 16, 512, 1, 256, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],
[5120, 8192, 16, 256, 4, 1024, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],
[5120, 8192, 128, 256, 1, 128, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],

[6144, 4096, 16, 512, 4, 1024, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],
[6144, 4096, 16, 512, 4, 128, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],
[6144, 4096, 1, 512, 1, 128, bfloat16, 0, ClampLimits(7, -7, 7, -7), False, ActFnType.SiLU],

[2880, 4096, 2, 512, 2, 2880, bfloat16, 0, ClampLimits(7, -7, 7, -7), True, ActFnType.Swish],
[2880, 4096, 2, 512, 2, 2880, bfloat16, 1, ClampLimits(7, -7, 7, -7), True, ActFnType.Swish],
[2880, 4096, 2, 256, 2, 2880, bfloat16, 1, ClampLimits(7, -7, 7, -7), True, ActFnType.Swish],
[2880, 4096, 2, 1024, 2, 2880, bfloat16, 0, ClampLimits(7, -7, 7, -7), True, ActFnType.Swish],
[2880, 4096, 32, 1024, 4, 2880, bfloat16, 1, ClampLimits(7, -7, 7, -7), True, ActFnType.Swish],

[4096, 4096, 2, 512, 2, 384, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],
[4096, 4096, 4, 512, 2, 384, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU],

]
# fmt: on


@final
class TestMoeBlockwiseMatMulBwdShardHAffinityHDroplessLnc2:
    """
    Tests for LNC2 blockwise matmul backward pass with shard on hidden and affinity on hidden.
    """

    def run_bwmm_bwd_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        collector: IMetricsCollector,
    ):
        is_negative_test_case = False

        bwmm_config = test_options.tensors[BWMM_BWD_CONFIG]

        tokens = bwmm_config[TOKENS_DIM_NAME]
        hidden = bwmm_config[HIDDEN_DIM_NAME]
        intermediate = bwmm_config[INTERMEDIATE_DIM_NAME]
        expert = bwmm_config[EXPERT_DIM_NAME]
        block_size = bwmm_config[BLOCK_SIZE_DIM_NAME]
        top_k = bwmm_config[TOP_K_DIM_NAME]
        dtype = bwmm_config[DTYPE_DIM_NAME]
        skip = bwmm_config[SKIP_DIM_NAME]
        clamp_limits = bwmm_config[CLAMP_LIMITS_DIM_NAME]
        bias = bwmm_config[BIAS_DIM_NAME]
        activation_type = bwmm_config[ACTIVATION_TYPE_DIM_NAME]

        dma_skip = map_skip_mode(skip)
        N = get_n_blocks(tokens, top_k, expert, block_size)

        with assert_negative_test_case(is_negative_test_case):
            kernel_input, gate_up_proj_bias, down_proj_bias = build_bwmm_bwd_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias=bias,
                clamp_limits=clamp_limits,
                activation_type=activation_type,
            )

            def create_lazy_golden():
                return golden_bwmm_bwd(
                    inp_np=kernel_input,
                    tokens=tokens,
                    hidden=hidden,
                    intermediate=intermediate,
                    expert=expert,
                    block_size=block_size,
                    dtype=dtype,
                    dma_skip=dma_skip,
                    bias=bias,
                    clamp_limits=clamp_limits,
                    activation_type=activation_type,
                    top_k=top_k,
                    gate_up_proj_bias=gate_up_proj_bias,
                    down_proj_bias=down_proj_bias,
                )

            T_out = tokens if dma_skip.skip_token else tokens + 1
            output_placeholder = {
                "hidden_states_grad": np.zeros((T_out, hidden), dtype=dtype),
                "expert_affinities_masked_grad": np.zeros((T_out * expert, 1), dtype=dtype),
                "gate_up_proj_weight_grad": np.zeros((expert, hidden, 2, intermediate), dtype=dtype),
                "down_proj_weight_grad": np.zeros((expert, intermediate, hidden), dtype=dtype),
            }

            if bias:
                output_placeholder["gate_and_up_proj_bias_grad"] = np.zeros((expert, 2, intermediate), dtype=dtype)
                output_placeholder["down_proj_bias_grad"] = np.zeros((expert, hidden), dtype=dtype)

            rtol, atol = 2e-2, 1e-5

            validation_args = ValidationArgs(
                golden_output=LazyGoldenGenerator(
                    lazy_golden_generator=create_lazy_golden,
                    output_ndarray=output_placeholder,
                ),
                relative_accuracy=rtol,
                absolute_accuracy=atol,
            )

            test_manager.execute(
                KernelArgs(
                    kernel_func=blockwise_mm_bwd,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=validation_args,
                )
            )

    @staticmethod
    def bwmm_bwd_lnc2_config():
        test_cases = []

        for test_params in moe_blockwise_mm_bwd_shardH_affinityH_dropless_lnc2_kernel_test_params:
            (
                hidden,
                tokens,
                expert,
                block_size,
                top_k,
                intermediate,
                dtype,
                skip,
                clamp_limits,
                bias,
                activation_type,
            ) = test_params

            test_case = {
                BWMM_BWD_CONFIG: {
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    BLOCK_SIZE_DIM_NAME: block_size,
                    TOP_K_DIM_NAME: top_k,
                    DTYPE_DIM_NAME: dtype,
                    SKIP_DIM_NAME: skip,
                    CLAMP_LIMITS_DIM_NAME: clamp_limits,
                    BIAS_DIM_NAME: bias,
                    ACTIVATION_TYPE_DIM_NAME: activation_type,
                },
            }
            test_cases.append(test_case)

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=TensorRangeConfig(
                tensor_configs={},
                monotonic_step_size=1,
                custom_generators=[
                    RangeManualGeneratorStrategy(test_cases=test_cases),
                ],
            ),
        )

    @pytest.mark.fast
    @range_test_config(bwmm_bwd_lnc2_config())
    def test_moe_blockwise_mm_bwd_shardH_affinityH_dropless_lnc2(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        lnc_count = 2
        compiler_args = CompilerArgs(logical_nc_config=lnc_count, enable_birsim=False)

        self.run_bwmm_bwd_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            collector=collector,
        )
