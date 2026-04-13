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

"""Shared utilities for MoE CTE blockwise matrix multiplication tests."""

import math
from enum import Enum
from test.utils.mx_utils import is_mx_quantize
from typing import Callable, Tuple

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block import bwmm_shard_on_block
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    SkipMode,
    blockwise_mm_baseline_shard_intermediate,
    blockwise_mm_baseline_shard_intermediate_hybrid,
    blockwise_mm_shard_intermediate_dropping,
)
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_utils import BlockShardStrategy
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from ....utils.test_kernel_common import act_fn_type2func

# Constants
_pmax = 128
_q_height = 8
_q_width = 4

# Dimension name constants
BWMM_CONFIG = "cfg"
BWMM_FUNC_DIM_NAME = "fn"
VNC_DEGREE_DIM_NAME = "vnc"
TOKENS_DIM_NAME = "tok"
HIDDEN_DIM_NAME = "hid"
INTERMEDIATE_DIM_NAME = "int"
EXPERT_DIM_NAME = "exp"
BLOCK_SIZE_DIM_NAME = "bs"
TOP_K_DIM_NAME = "k"
ACT_FN_DIM_NAME = "act"
EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME = "easm"
DTYPE_DIM_NAME = "dt"
SKIP_DIM_NAME = "sk"
BIAS_DIM_NAME = "bi"
TRAINING_DIM_NAME = "tr"
QUANTIZE_DIM_NAME = "q"
GATE_CLAMP_UPPER_DIM_NAME = "gcu"
GATE_CLAMP_LOWER_DIM_NAME = "gcl"
UP_CLAMP_UPPER_DIM_NAME = "ucu"
UP_CLAMP_LOWER_DIM_NAME = "ucl"
EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME = "eamoi"

dtype2dtype_range = {nl.int8: (-127, 127), nl.float8_e4m3: (-240.0, 240.0)}


class BWMMFunc(Enum):
    SHARD_ON_BLOCK = 1
    SHARD_ON_HIDDEN = 2
    SHARD_ON_INTERMEDIATE = 3
    SHARD_ON_INTERMEDIATE_AFFINITY_ON_INTERMEDIATE = 4
    SHARD_ON_INTERMEDIATE_HW = 5
    SHARD_ON_INTERMEDIATE_DROPPING = 6

    MXFP4_SHARD_ON_BLOCK = 10
    MXFP4_SHARD_ON_BLOCK_DYNAMIC = 11

    BASELINE = 100
    BASELINE_ALLOCATED = 200

    def get_bwmm_func(self) -> Tuple[Callable, bool, bool]:
        """Return the kernel function, whether the kernel is nki.jit, whether it is dynamic."""
        if self == BWMMFunc.SHARD_ON_INTERMEDIATE:
            return blockwise_mm_baseline_shard_intermediate, True, False
        elif self == BWMMFunc.SHARD_ON_INTERMEDIATE_HW:
            return blockwise_mm_baseline_shard_intermediate_hybrid, True, True
        elif self == BWMMFunc.SHARD_ON_BLOCK:
            return bwmm_shard_on_block, True, False
        elif self == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING:
            return blockwise_mm_shard_intermediate_dropping, True, False
        else:
            assert False, "invalid BWMM kernel name"


def get_n_blocks(T, TOPK, E, B, n_block_per_iter=1):
    N = math.ceil((T * TOPK - (E - 1)) / B) + E - 1
    N = n_block_per_iter * math.ceil(N / n_block_per_iter)
    return N


def get_block_size_dropping(seq_len, batch_size, top_k, capacity_factor, num_experts):
    return seq_len * batch_size * top_k * capacity_factor // num_experts


def calculate_local_t_dropping(seq_len, batch_size, top_k, capacity_factor, expert_parallelism):
    return seq_len * batch_size * top_k * capacity_factor // expert_parallelism


def generate_token_position_to_id_and_experts_dropping(T, E_Local, B, rtype, topK):
    """Generate token position to ID mapping for dropping kernel where N = E_Local."""
    token_position_to_id = np.full((int(E_Local * B),), T, dtype=np.int32)
    token_usage_count = np.zeros(T, dtype=np.int32)

    for e in range(E_Local):
        expert_start_idx = e * B
        needed_tokens = B if rtype == 1 else int(B * np.random.rand())
        available_tokens = np.where(token_usage_count < topK)[0]

        if len(available_tokens) >= needed_tokens:
            selected_tokens = np.random.choice(available_tokens, needed_tokens, replace=False)
        else:
            selected_tokens = available_tokens.copy()
            needed_tokens = len(selected_tokens)

        token_usage_count[selected_tokens] += 1

        if rtype == 1:
            token_indices = selected_tokens.astype(np.int32)
        else:
            padded_block_size = B - needed_tokens
            token_indices = np.concatenate(
                [selected_tokens.astype(np.int32), np.full((padded_block_size,), T, dtype=np.int32)]
            )

        token_position_to_id[expert_start_idx : expert_start_idx + B] = token_indices

    block_to_expert = np.arange(E_Local).astype(np.int32)
    return token_position_to_id, block_to_expert


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


def quantize_strategy2scale_shapes(quantize_strategy, E, I_TP, H):
    if quantize_strategy == 0:
        assert False, "no scales for no quant"
    elif quantize_strategy == 1:
        return [1, 1, 2 * 1], [1, 1, 1]
    elif quantize_strategy == 2:
        return [1, 1, 2 * I_TP], [1, 1, H]
    elif quantize_strategy == 3:
        assert False, "per block quantize not supported yet"
    elif quantize_strategy == 4:
        return [E, 1, 2 * 1], [E, 1, 1]
    elif quantize_strategy == 5:
        return [E, 1, 2 * I_TP], [E, 1, H]
    elif quantize_strategy == 6:
        return [E, 1, 2 * I_TP], [E, 1, H]
    else:
        raise ValueError("Unrecognized quantize strategy")


def get_router_with_controlled_distribution(T: int, TOPK: int, E: int, alpha: np.float32 = None):
    """Generate a uniform or controlled probability distribution over E experts for tokens."""
    actual_k = min(E, TOPK)

    router = np.zeros((T, actual_k))
    np.random.seed(0)
    if alpha is None:
        if E < TOPK:
            for i in range(T):
                router[i] = np.random.choice(range(E), (E), replace=False)
        else:
            for i in range(T):
                router[i] = np.random.choice(range(E), (TOPK), replace=False)
    elif alpha == -1:
        p_of_e = np.random.dirichlet(np.ones(E) * 0.001)
        for i in range(0, T):
            router[i] = np.random.choice(range(E), (actual_k), replace=False, p=p_of_e)
    else:
        for i in range(E):
            router[i] = np.arange(i, (i + actual_k) % T)
        p_of_e = np.random.dirichlet(np.ones(E) * alpha)
        for i in range(E, T):
            router[i] = np.random.choice(range(E), (actual_k), replace=False, p=p_of_e)
    return router


def generate_token_position_to_id_and_experts(
    T: int,
    TOPK: int,
    E: int,
    B: int,
    dma_skip: SkipMode,
    N: int,
    use_split_padding: bool = False,
    n_block_per_iter: int = 1,
    vnc_degree: int = 1,
    alpha: np.float32 = None,
    is_block_parallel: bool = False,
    quantize=None,
):
    if n_block_per_iter > 1:
        assert vnc_degree == 1

    router = get_router_with_controlled_distribution(T=T, TOPK=TOPK, E=E, alpha=alpha)
    one_hot = np.arange(E)
    token_experts = np.zeros((T, E))
    actual_k = min(E, TOPK)
    for i in range(actual_k):
        token_experts += np.expand_dims(router[:, i], 1) == np.expand_dims(one_hot, 0)

    blocks_per_expert = np.ceil(token_experts.sum(0) / B).astype(np.int32)
    n_padding_block = N - np.sum(blocks_per_expert)

    if use_split_padding:
        blocks_per_expert[(E - 1) // 2] += n_padding_block // 2
        blocks_per_expert[E - 1] += n_padding_block - n_padding_block // 2
    else:
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

    if not is_block_parallel:
        n_condition = n_block_per_iter * math.ceil(N / n_block_per_iter)
        conditions = np.ones((n_condition + 1,), dtype=np.int32)
        conditions[-(n_padding_block + 1) :] = 0
    else:
        if is_mx_quantize(quantize):
            assert n_block_per_iter == 1, f"BWMM MXFP4 shard on block only support n_block_per_iter=1"
            conditions = np.ones((N + 2,), dtype=np.int32)
            conditions[-(n_padding_block + 2) :] = 0
        else:
            conditions = np.ones((math.ceil(N / vnc_degree) + 1,), dtype=np.int32)
            conditions[-(math.ceil(n_padding_block / vnc_degree) + 1) :] = 0

    return token_experts, token_position_to_id, block_to_expert, conditions


def generate_blockwise_numpy_golden(
    lnc_degree: int,
    expert_affinities,
    down_proj_weights,
    token_position_to_id,
    block_to_expert,
    gate_and_up_proj_weights,
    hidden_states,
    T: int,
    H: int,
    B: int,
    N: int,
    E: int,
    I_TP: int,
    dtype,
    dma_skip: SkipMode,
    quantize=False,
    quantize_strategy: int = 5,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    activation_function=ActFnType.SiLU,
    gate_up_proj_bias=None,
    down_proj_bias=None,
    gate_up_proj_scale=np.empty([]),
    down_proj_scale=np.empty([]),
    checkpoint_activation: bool = False,
    separate_outputs: bool = False,
    conditions=None,
    n_block_per_iter: int = 1,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    DBG_KERNEL: bool = False,
    is_shard_block: bool = False,
):
    DBG_golden_tensors = {}

    if is_shard_block:
        output_shape = [T + 1, lnc_degree, H] if separate_outputs else [T + 1, H]
    else:
        output_shape = [lnc_degree, T + 1, H] if separate_outputs else [T + 1, H]

    output_np = np.zeros(output_shape).astype(dtype)
    token_position_to_id = token_position_to_id.reshape(N, B)

    if checkpoint_activation:
        gate_up_activations_T = np.zeros([N, 2, I_TP, B]).astype(dtype)
        down_activations = np.zeros([N, B, H]).astype(dtype)

    if dma_skip.skip_weight:
        is_weight_same_as_prev = np.zeros((N))
        is_weight_same_as_prev[1:] = block_to_expert[1:] == block_to_expert[:-1]
        is_weight_same_as_prev = is_weight_same_as_prev.astype(np.uint8)

    gate_up_weights = None
    down_weights = None

    if quantize and quantize_strategy == 5:
        down_proj_scale = np.transpose(down_proj_scale.reshape((E, 128, H // 128)), (0, 2, 1)).reshape((E, 1, H))

    E_local = gate_and_up_proj_weights.shape[0]
    gate_and_up_proj_weights = gate_and_up_proj_weights.reshape(E_local, H, 2 * I_TP).astype(np.float32)
    down_proj_weights = down_proj_weights.astype(np.float32)

    for b in range(N):
        if conditions is not None and conditions[b] == 0:
            break

        local_token_position_to_id = token_position_to_id[b, :]

        if dma_skip.skip_token:
            zeros_hidden = np.zeros((1, H)).astype(dtype)
            hidden_states = np.concatenate([hidden_states, zeros_hidden], axis=0)
            zeros_exaf = np.zeros((1, E)).astype(dtype)
            expert_affinities = np.concatenate([expert_affinities, zeros_exaf], axis=0)

        local_hidden_states = hidden_states[local_token_position_to_id[:], :].astype(np.float32)

        if DBG_KERNEL and b == 0:
            DBG_golden_tensors["dbg_hidden_states"] = (
                local_hidden_states.reshape(
                    -1,
                    32 * _q_width,
                    H // _pmax // _q_width,
                    _pmax,
                )
                .transpose(3, 2, 0, 1)
                .astype(dtype)
            )

        expert_idx = block_to_expert[b]
        local_expert_affinities = expert_affinities[local_token_position_to_id, expert_idx].reshape(-1, 1).astype(dtype)

        if expert_affinities_scaling_mode in [
            ExpertAffinityScaleMode.PRE_SCALE,
            ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        ]:
            local_hidden_states = local_expert_affinities * local_hidden_states

        if dma_skip.skip_weight:
            expert_idx = E if is_weight_same_as_prev[b] else expert_idx

        if expert_idx < E:
            gate_up_weights = gate_and_up_proj_weights[expert_idx]
            down_weights = down_proj_weights[expert_idx, :, :]

            if quantize and gate_up_proj_scale.shape[0] == E:
                gup_scale = gate_up_proj_scale[expert_idx]
                down_scale = down_proj_scale[expert_idx]

            if gate_up_proj_bias is not None:
                gate_up_bias = gate_up_proj_bias[expert_idx]

            if down_proj_bias is not None:
                down_bias = down_proj_bias[expert_idx]

        gate_up_activation = np.matmul(local_hidden_states, gate_up_weights).reshape(B, 2, I_TP)
        gate_activation = gate_up_activation[:, 0, :]
        up_activation = gate_up_activation[:, 1, :]

        if quantize and gate_up_proj_scale.shape[0] == 1:
            gate_activation *= gate_up_proj_scale.squeeze()[:I_TP]
            up_activation *= gate_up_proj_scale.squeeze()[I_TP:]
        elif quantize and gate_up_proj_scale.shape[0] == E:
            gate_activation *= gup_scale[0, :I_TP]
            up_activation *= gup_scale[0, I_TP:]

        if gate_up_proj_bias is not None:
            gate_activation += gate_up_bias[0, :]
            up_activation += gate_up_bias[1, :]

        if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
            np.clip(gate_activation, a_min=gate_clamp_lower_limit, a_max=gate_clamp_upper_limit, out=gate_activation)

        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            np.clip(up_activation, a_min=up_clamp_lower_limit, a_max=up_clamp_upper_limit, out=up_activation)

        if checkpoint_activation:
            gate_up_activations_T[b] = gate_up_activation.transpose(1, 2, 0)

        act_fn_method = act_fn_type2func[activation_function]
        act_res = act_fn_method(gate_activation)
        multiply_1 = act_res * up_activation
        down_activation = np.matmul(multiply_1, down_weights)

        if quantize and gate_up_proj_scale.shape[0] == 1:
            down_activation = down_activation * down_proj_scale.squeeze()
        elif quantize and gate_up_proj_scale.shape[0] == E:
            down_activation = down_activation * down_scale[0, :]

        if down_proj_bias is not None:
            down_activation += down_bias

        if checkpoint_activation:
            down_activations[b] = down_activation

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            scale = down_activation * local_expert_affinities
        else:
            scale = down_activation

        if separate_outputs:
            if is_shard_block:
                output_np[local_token_position_to_id[:], 0, :] += scale.astype(output_np.dtype)
            else:
                output_np[0, local_token_position_to_id[:], :] += scale.astype(output_np.dtype)
        else:
            output_np[local_token_position_to_id[:], :] += scale.astype(output_np.dtype)

    if separate_outputs:
        if is_shard_block:
            out_return = output_np[:T, :, :] if dma_skip.skip_token else output_np
        else:
            out_return = output_np[:, :T, :] if dma_skip.skip_token else output_np
    else:
        out_return = output_np[:T, :] if dma_skip.skip_token else output_np

    if checkpoint_activation:
        return out_return, gate_up_activations_T, down_activations

    return out_return, DBG_golden_tensors


# NOTE: golden_bwmm is still used by test_moe_cte_unified.py (legacy unified test path)
def golden_bwmm(
    inp_np: dict,
    lnc_degree: int,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    dtype,
    dma_skip: SkipMode,
    bias: bool,
    quantize,
    quantize_strategy: int,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    activation_function: ActFnType,
    checkpoint_activation: bool,
    is_block_parallel: bool,
    top_k: int,
    n_block_per_iter: int = 1,
    gate_clamp_lower_limit=None,
    gate_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    is_shard_block: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    is_dropping: bool = False,
):
    """Generate golden output for blockwise matmul kernel."""
    if is_dropping:
        N = expert
    else:
        N = get_n_blocks(
            tokens,
            top_k,
            expert,
            block_size,
            n_block_per_iter=lnc_degree if is_block_parallel else n_block_per_iter,
        )
    conditions = inp_np.get("conditions", None)
    expert_affinities_masked = inp_np["expert_affinities_masked"].reshape(-1, expert)
    down_proj_weights = inp_np["down_proj_weight"]
    token_position_to_id = inp_np["token_position_to_id"]
    block_to_expert = inp_np["block_to_expert"]
    gate_up_proj_weights = inp_np["gate_up_proj_weight"]
    hidden_states = inp_np["hidden_states"]

    gate_up_proj_bias = inp_np.get("gate_and_up_proj_bias", None)
    down_proj_bias = inp_np.get("down_proj_bias", None)
    gate_up_proj_scale = inp_np.get("gate_up_proj_scale", np.empty([]))
    down_proj_scale = inp_np.get("down_proj_scale", np.empty([]))

    separate_outputs = is_block_parallel and top_k > 1

    result = generate_blockwise_numpy_golden(
        lnc_degree=lnc_degree,
        expert_affinities=expert_affinities_masked,
        down_proj_weights=down_proj_weights,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_and_up_proj_weights=gate_up_proj_weights,
        hidden_states=hidden_states,
        T=tokens,
        H=hidden,
        B=block_size,
        N=N,
        E=expert,
        I_TP=intermediate,
        dtype=dtype,
        dma_skip=dma_skip,
        quantize=quantize,
        quantize_strategy=quantize_strategy,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_function=activation_function,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        checkpoint_activation=checkpoint_activation,
        separate_outputs=separate_outputs,
        conditions=conditions,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        is_shard_block=is_shard_block,
    )

    if checkpoint_activation:
        output, gate_up_activations_T, down_activations = result
        return_dict = {
            "output": output,
            "gate_up_activations_T": gate_up_activations_T,
        }
        if not expert_affinity_multiply_on_I:
            return_dict["down_activations"] = down_activations
        return return_dict
    else:
        output, dbg_tensors = result
        golden = {"output": output}
        golden.update(dbg_tensors)
        return golden


def build_bwmm_inputs(
    bwmm_func_enum: BWMMFunc,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    top_k: int,
    dtype,
    dma_skip: SkipMode,
    bias: bool,
    quantize,
    quantize_strategy: int,
    vnc_degree: int,
    is_block_parallel: bool,
    is_dynamic: bool,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    activation_function: ActFnType,
    n_block_per_iter: int = 1,
    checkpoint_activation: bool = False,
    gate_clamp_lower_limit=None,
    gate_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    alpha: np.float32 = None,
    expert_affinity_multiply_on_I: bool = False,
    is_dropping: bool = False,
    rtype: int = 1,
):
    """Build input tensors and parameters for blockwise matmul kernel."""
    is_dropping = bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING

    if is_dropping:
        N = expert
        token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts_dropping(
            T=tokens, E_Local=expert, B=block_size, rtype=rtype, topK=top_k
        )
        expert_masks = np.ones([tokens, expert], dtype=dtype)
        conditions = None
    else:
        N = get_n_blocks(
            tokens,
            top_k,
            expert,
            block_size,
            n_block_per_iter=vnc_degree if is_block_parallel else n_block_per_iter,
        )
        (
            expert_masks,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = generate_token_position_to_id_and_experts(
            T=tokens,
            TOPK=top_k,
            E=expert,
            B=block_size,
            dma_skip=dma_skip,
            N=N,
            use_split_padding=False,
            vnc_degree=vnc_degree,
            n_block_per_iter=n_block_per_iter,
            alpha=alpha,
            is_block_parallel=is_block_parallel,
            quantize=quantize,
        )

    np.random.seed(0)

    if dma_skip.skip_token:
        expert_affinities_masked = np.random.random_sample([tokens, expert]).astype(dtype)
        expert_affinities_masked = expert_affinities_masked * expert_masks
        hidden_states = np.random.random_sample([tokens, hidden]).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([tokens + 1, expert]).astype(dtype)
        expert_affinities_masked[:tokens] = expert_affinities_masked[:tokens] * expert_masks
        expert_affinities_masked[tokens] = 0
        hidden_states = np.random.random_sample([tokens + 1, hidden]).astype(dtype)
        expert_masks = np.vstack([expert_masks, np.zeros([1, expert])])
        hidden_states[tokens, ...] = 0

    expert_affinities_masked = expert_affinities_masked.astype(dtype)

    gate_up_proj_bias = None
    down_proj_bias = None
    if bias:
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[expert, hidden]).astype(dtype)
        gate_up_proj_bias = np.random.uniform(-1, 1, size=[expert, 2, intermediate]).astype(dtype)

    gate_up_proj_scale = None
    down_proj_scale = None
    I_TP_padded = math.ceil(intermediate / 16) * 16

    if quantize:
        quantize_dt = nl.int8 if quantize == 1 else quantize
        dtype_min, dtype_max = dtype2dtype_range[quantize_dt]
        down_proj_weights_padded = np.random.randint(dtype_min, dtype_max, size=[expert, I_TP_padded, hidden]).astype(
            quantize_dt
        )
        gate_up_proj_weights = np.random.randint(dtype_min, dtype_max, size=[expert, hidden, 2, intermediate]).astype(
            quantize_dt
        )
        gup_scale_shape, down_scale_shape = quantize_strategy2scale_shapes(
            quantize_strategy, expert, intermediate, hidden
        )
        scale_dtype = dtype if quantize_strategy == 2 else nl.float32
        gate_up_proj_scale = np.random.uniform(-0.1, 0.1, size=gup_scale_shape).astype(scale_dtype)
        down_proj_scale = np.random.uniform(-0.1, 0.1, size=down_scale_shape).astype(scale_dtype)
    else:
        down_proj_weights_padded = np.random.uniform(-0.1, 0.1, size=[expert, I_TP_padded, hidden]).astype(dtype)
        gate_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[expert, hidden, 2, intermediate]).astype(dtype)

    inputs = {}
    if is_dynamic:
        inputs["conditions"] = conditions

    inputs.update(
        {
            "hidden_states": hidden_states,
            "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
            "gate_up_proj_weight": gate_up_proj_weights,
            "down_proj_weight": down_proj_weights_padded,
            "token_position_to_id": token_position_to_id,
            "block_to_expert": block_to_expert,
            "block_size": block_size,
            "skip_dma": dma_skip,
            "compute_dtype": dtype,
            "is_tensor_update_accumulating": top_k != 1,
            "expert_affinities_scaling_mode": expert_affinities_scaling_mode,
            "activation_function": activation_function,
        }
    )

    if bias:
        inputs["gate_and_up_proj_bias"] = gate_up_proj_bias
        inputs["down_proj_bias"] = down_proj_bias

    if quantize:
        inputs["gate_up_proj_scale"] = gate_up_proj_scale
        inputs["down_proj_scale"] = down_proj_scale

    if gate_clamp_lower_limit is not None:
        inputs["gate_clamp_lower_limit"] = gate_clamp_lower_limit
    if gate_clamp_upper_limit is not None:
        inputs["gate_clamp_upper_limit"] = gate_clamp_upper_limit
    if up_clamp_lower_limit is not None:
        inputs["up_clamp_lower_limit"] = up_clamp_lower_limit
    if up_clamp_upper_limit is not None:
        inputs["up_clamp_upper_limit"] = up_clamp_upper_limit

    if bwmm_func_enum in [BWMMFunc.SHARD_ON_INTERMEDIATE, BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING]:
        inputs["checkpoint_activation"] = checkpoint_activation
        inputs["expert_affinity_multiply_on_I"] = expert_affinity_multiply_on_I

    if bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING:
        inputs.pop("checkpoint_activation", None)

    return inputs


def _create_bwmm_test_config(test_params_list, test_type="manual"):
    """Create RangeTestConfig from parameter list. Used by legacy tests."""
    from test.utils.ranged_test_harness import (
        RangeManualGeneratorStrategy,
        RangeTestConfig,
        TensorRangeConfig,
    )

    test_cases = []
    for test_params in test_params_list:
        (
            bwmm_func,
            hidden,
            tokens,
            expert,
            block_size,
            top_k,
            intermediate,
            dtype,
            skip,
            bias,
            training,
            quantize,
            act_fn,
            expert_affinities_scaling_mode,
            gate_cl_upper,
            gate_cl_lower,
            up_cl_upper,
            up_cl_lower,
            expert_affinity_multiply_on_I,
        ) = test_params

        test_cases.append(
            {
                BWMM_CONFIG: {
                    BWMM_FUNC_DIM_NAME: bwmm_func,
                    VNC_DEGREE_DIM_NAME: 2,
                    TOKENS_DIM_NAME: tokens,
                    HIDDEN_DIM_NAME: hidden,
                    INTERMEDIATE_DIM_NAME: intermediate,
                    EXPERT_DIM_NAME: expert,
                    BLOCK_SIZE_DIM_NAME: block_size,
                    TOP_K_DIM_NAME: top_k,
                    ACT_FN_DIM_NAME: act_fn,
                    EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME: expert_affinities_scaling_mode,
                    DTYPE_DIM_NAME: dtype,
                    SKIP_DIM_NAME: skip,
                    BIAS_DIM_NAME: bias,
                    TRAINING_DIM_NAME: training,
                    QUANTIZE_DIM_NAME: quantize,
                    GATE_CLAMP_UPPER_DIM_NAME: gate_cl_upper,
                    GATE_CLAMP_LOWER_DIM_NAME: gate_cl_lower,
                    UP_CLAMP_UPPER_DIM_NAME: up_cl_upper,
                    UP_CLAMP_LOWER_DIM_NAME: up_cl_lower,
                    EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME: expert_affinity_multiply_on_I,
                },
            }
        )

    return RangeTestConfig(
        additional_params={},
        global_tensor_configs=TensorRangeConfig(
            tensor_configs={},
            monotonic_step_size=1,
            custom_generators=[RangeManualGeneratorStrategy(test_cases=test_cases, test_type=test_type)],
        ),
    )


# =============================================================================
# UnitTestFramework wrappers
# =============================================================================


def moe_cte_kernel_wrapper(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    block_size: int,
    bwmm_func: BWMMFunc,
    lnc_degree: int = 2,
    conditions=None,
    gate_and_up_proj_bias=None,
    down_proj_bias=None,
    gate_up_proj_scale=None,
    down_proj_scale=None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    n_block_per_iter: int = 1,
    block_sharding_strategy=None,
    num_static_block=None,
    gate_up_activations_T=None,
    down_activations=None,
    top_k: int = 1,
):
    """Wrapper that dispatches to the correct kernel based on bwmm_func."""
    # lnc_degree and top_k are test-framework parameters not forwarded to kernels
    _ = lnc_degree, top_k

    if bwmm_func == BWMMFunc.SHARD_ON_BLOCK:
        return bwmm_shard_on_block(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            n_block_per_iter=n_block_per_iter,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            block_sharding_strategy=block_sharding_strategy
            if block_sharding_strategy is not None
            else BlockShardStrategy.PING_PONG,
        )
    elif bwmm_func == BWMMFunc.SHARD_ON_INTERMEDIATE:
        return blockwise_mm_baseline_shard_intermediate(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            checkpoint_activation=checkpoint_activation,
            expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        )
    elif bwmm_func == BWMMFunc.SHARD_ON_INTERMEDIATE_HW:
        return blockwise_mm_baseline_shard_intermediate_hybrid(
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            num_static_block=num_static_block,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            gate_up_activations_T=gate_up_activations_T,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
        )
    elif bwmm_func == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING:
        return blockwise_mm_shard_intermediate_dropping(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
        )
    else:
        assert False, f"Unsupported bwmm_func: {bwmm_func}"


def moe_cte_torch_wrapper(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    block_size: int,
    bwmm_func: BWMMFunc,
    lnc_degree: int = 2,
    conditions=None,
    gate_and_up_proj_bias=None,
    down_proj_bias=None,
    gate_up_proj_scale=None,
    down_proj_scale=None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    n_block_per_iter: int = 1,
    block_sharding_strategy=None,
    num_static_block=None,
    gate_up_activations_T=None,
    down_activations=None,
    top_k: int = 1,
) -> dict:
    """Torch reference wrapper matching moe_cte_kernel_wrapper signature."""
    from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_torch import moe_cte_torch_ref

    return moe_cte_torch_ref(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=bwmm_func,
        lnc_degree=lnc_degree,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        n_block_per_iter=n_block_per_iter,
        block_sharding_strategy=block_sharding_strategy,
        num_static_block=num_static_block,
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        top_k=top_k,
    )


def generate_moe_cte_inputs(
    bwmm_func_enum: BWMMFunc,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    top_k: int,
    dtype,
    skip: int,
    bias: bool,
    training: bool,
    quantize,
    activation_function: ActFnType,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    gate_clamp_upper=None,
    gate_clamp_lower=None,
    up_clamp_upper=None,
    up_clamp_lower=None,
    expert_affinity_multiply_on_I: bool = False,
    lnc_degree: int = 2,
):
    """Generate inputs dict compatible with UnitTestFramework (keys match wrapper signature)."""
    _, is_nkijit, is_dynamic = bwmm_func_enum.get_bwmm_func()
    is_block_parallel = bwmm_func_enum == BWMMFunc.SHARD_ON_BLOCK
    is_dropping = bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING

    dma_skip = map_skip_mode(skip)
    quantize_strategy = 6 if quantize else 0

    raw = build_bwmm_inputs(
        bwmm_func_enum=bwmm_func_enum,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        expert=expert,
        block_size=block_size,
        top_k=top_k,
        dtype=dtype,
        dma_skip=dma_skip,
        bias=bias,
        quantize=quantize,
        quantize_strategy=quantize_strategy,
        vnc_degree=lnc_degree,
        n_block_per_iter=1,
        is_block_parallel=is_block_parallel,
        is_dynamic=is_dynamic,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_function=activation_function,
        checkpoint_activation=training,
        gate_clamp_lower_limit=gate_clamp_lower,
        gate_clamp_upper_limit=gate_clamp_upper,
        up_clamp_lower_limit=up_clamp_lower,
        up_clamp_upper_limit=up_clamp_upper,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
    )

    # Remap to wrapper signature
    inputs = {
        "hidden_states": raw["hidden_states"],
        "expert_affinities_masked": raw["expert_affinities_masked"],
        "gate_up_proj_weight": raw["gate_up_proj_weight"],
        "down_proj_weight": raw["down_proj_weight"],
        "token_position_to_id": raw["token_position_to_id"],
        "block_to_expert": raw["block_to_expert"],
        "block_size": raw["block_size"],
        "bwmm_func": bwmm_func_enum,
        "lnc_degree": lnc_degree,
        "activation_function": raw["activation_function"],
        "skip_dma": raw["skip_dma"],
        "compute_dtype": raw["compute_dtype"],
        "is_tensor_update_accumulating": raw["is_tensor_update_accumulating"],
        "expert_affinities_scaling_mode": raw["expert_affinities_scaling_mode"],
        "top_k": top_k,
    }

    if "conditions" in raw:
        inputs["conditions"] = raw["conditions"]
    if "gate_and_up_proj_bias" in raw:
        inputs["gate_and_up_proj_bias"] = raw["gate_and_up_proj_bias"]
    if "down_proj_bias" in raw:
        inputs["down_proj_bias"] = raw["down_proj_bias"]
    if "gate_up_proj_scale" in raw:
        inputs["gate_up_proj_scale"] = raw["gate_up_proj_scale"]
    if "down_proj_scale" in raw:
        inputs["down_proj_scale"] = raw["down_proj_scale"]
    if "gate_clamp_upper_limit" in raw:
        inputs["gate_clamp_upper_limit"] = raw["gate_clamp_upper_limit"]
    if "gate_clamp_lower_limit" in raw:
        inputs["gate_clamp_lower_limit"] = raw["gate_clamp_lower_limit"]
    if "up_clamp_upper_limit" in raw:
        inputs["up_clamp_upper_limit"] = raw["up_clamp_upper_limit"]
    if "up_clamp_lower_limit" in raw:
        inputs["up_clamp_lower_limit"] = raw["up_clamp_lower_limit"]
    if "checkpoint_activation" in raw:
        inputs["checkpoint_activation"] = raw["checkpoint_activation"]
    if "expert_affinity_multiply_on_I" in raw:
        inputs["expert_affinity_multiply_on_I"] = raw["expert_affinity_multiply_on_I"]

    return inputs


def moe_cte_output_tensors(
    kernel_input: dict,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    top_k: int,
    dtype,
    bwmm_func_enum: BWMMFunc,
    training: bool,
    expert_affinity_multiply_on_I: bool,
    lnc_degree: int = 2,
):
    """Generate output tensor placeholders for UnitTestFramework."""
    is_block_parallel = bwmm_func_enum == BWMMFunc.SHARD_ON_BLOCK
    is_dropping = bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING
    is_shard_block = bwmm_func_enum == BWMMFunc.SHARD_ON_BLOCK

    dma_skip = kernel_input["skip_dma"]
    T_out = tokens if dma_skip.skip_token else tokens + 1
    separate_outputs = is_block_parallel and top_k > 1

    if separate_outputs:
        if is_shard_block:
            output_shape = (T_out, lnc_degree, hidden)
        else:
            output_shape = (lnc_degree, T_out, hidden)
    else:
        output_shape = (T_out, hidden)

    if training or is_dropping:
        if is_dropping:
            N = expert
        else:
            N = get_n_blocks(
                tokens,
                top_k,
                expert,
                block_size,
                n_block_per_iter=lnc_degree if is_block_parallel else 1,
            )
        if expert_affinity_multiply_on_I:
            return {
                "output": np.zeros(output_shape, dtype=dtype),
                "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
            }
        else:
            return {
                "output": np.zeros(output_shape, dtype=dtype),
                "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
                "down_activations": np.zeros((N, block_size, hidden), dtype=dtype),
            }
    else:
        return {"output": np.zeros(output_shape, dtype=dtype)}
