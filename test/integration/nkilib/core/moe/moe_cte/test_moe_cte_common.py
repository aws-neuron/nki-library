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
from typing import Optional

import ml_dtypes
import nki.language as nl
import numpy as np

# BWMMFunc and _mx_internal_data live in src/ so torch_refs can import them
# without depending on test/. Re-exported here for existing test callers.
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_func import BWMMFunc  # noqa: F401
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block import bwmm_shard_on_block
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    SkipMode,
    blockwise_mm_baseline_shard_intermediate,
    blockwise_mm_baseline_shard_intermediate_hybrid,
    blockwise_mm_shard_intermediate_dropping,
)
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_utils import (
    BlockShardStrategy,
)
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from nkilib_src.nkilib.experimental.moe.moe_cte.bwmm_shard_on_block_v2 import (
    bwmm_shard_on_block as bwmm_shard_on_block_v2,
)
from nkilib_src.nkilib.experimental.moe.moe_cte.bwmm_shard_on_block_v2 import (
    bwmm_shard_on_block_hybrid,
)
from typing_extensions import override

from test.utils.common_dataclasses import CustomValidator
from test.utils.mx_utils import is_mx_quantize

# Constants
_pmax = 128
_q_height = 8
_q_width = 4

# Constructed once at module load and reused as the shared default for `skip_dma`
# parameters below; SkipMode instances are not mutated by any callee in this module.
_DEFAULT_SKIP_MODE = SkipMode(False, False)

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
        raise AssertionError("no scales for no quant")
    elif quantize_strategy == 1:
        return [1, 1, 2 * 1], [1, 1, 1]
    elif quantize_strategy == 2:
        return [1, 1, 2 * I_TP], [1, 1, H]
    elif quantize_strategy == 3:
        raise AssertionError("per block quantize not supported yet")
    elif quantize_strategy == 4:
        return [E, 1, 2 * 1], [E, 1, 1]
    elif quantize_strategy == 5:
        return [E, 1, 2 * I_TP], [E, 1, H]
    elif quantize_strategy == 6:
        return [E, 1, 2 * I_TP], [E, 1, H]
    else:
        raise ValueError("Unrecognized quantize strategy")


def get_router_with_controlled_distribution(
    T: int, TOPK: int, E: int, alpha: np.float32 = None, non_overlapping_shards: bool = False
):
    """Generate a uniform or controlled probability distribution over E experts for tokens."""
    actual_k = min(E, TOPK)

    router = np.zeros((T, actual_k))
    np.random.seed(0)
    if non_overlapping_shards:
        # Force each token's experts to be in the same half: 0..E/2-1 or E/2..E-1
        half_E = E // 2
        for i in range(T):
            if i < T // 2:
                router[i] = np.random.choice(range(half_E), actual_k, replace=False)
            else:
                router[i] = np.random.choice(range(half_E, E), actual_k, replace=False)
    elif alpha is None:
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
    non_overlapping_shards: bool = False,
):
    if n_block_per_iter > 1:
        assert vnc_degree == 1

    router = get_router_with_controlled_distribution(
        T=T, TOPK=TOPK, E=E, alpha=alpha, non_overlapping_shards=non_overlapping_shards
    )
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
            assert n_block_per_iter == 1, "BWMM MXFP4 shard on block only support n_block_per_iter=1"
            conditions = np.ones((N + 2,), dtype=np.int32)
            conditions[-(n_padding_block + 2) :] = 0
        else:
            # Per-shard conditions derived from block_to_expert, concatenated:
            # [shard0 (n_bps entries) | 0 | shard1 (n_bps entries) | 0]
            # Each shard reads from offset shard_id * (n_bps + 1)
            # The 0 after each shard's entries guarantees loop termination
            n_bps = math.ceil(N / vnc_degree)
            conditions = np.zeros((vnc_degree * (n_bps + 1),), dtype=np.int32)
            for s in range(vnc_degree):
                for i in range(n_bps):
                    global_idx = s * n_bps + i
                    if global_idx < N and block_to_expert[global_idx] < E:
                        conditions[s * (n_bps + 1) + i] = 1

    return token_experts, token_position_to_id, block_to_expert, conditions


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
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    non_overlapping_shards: bool = False,
    is_block_quant: bool = False,
    is_per_tensor: bool = False,
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
            non_overlapping_shards=non_overlapping_shards,
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
        _bias_h = hidden // down_bias_tp_degree if down_bias_tp_degree is not None else hidden
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[expert, _bias_h]).astype(dtype)
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
        # Generate per-token hidden scales for static quantization (only for SHARD_ON_INTERMEDIATE)
        if bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE:
            T_scale = tokens if dma_skip.skip_token else tokens + 1
            inputs["gate_up_hidden_scale"] = np.random.uniform(0.01, 0.1, size=[T_scale, 1]).astype(np.float32)
            inputs["down_hidden_scale"] = np.random.uniform(0.01, 0.1, size=[T_scale, 1]).astype(np.float32)
        if is_block_quant:
            # Override scales with 256x256 block quant shapes, pre-broadcasted with TILE_SIZE dim
            BQ = 256
            H_blocks = hidden // BQ
            I_blocks = intermediate // BQ
            inputs["gate_up_proj_scale"] = np.repeat(
                np.random.uniform(0.01, 0.1, size=[expert, H_blocks, 2, I_blocks, 1]).astype(np.float32), 128, axis=4
            )
            inputs["down_proj_scale"] = np.repeat(
                np.random.uniform(0.01, 0.1, size=[expert, I_TP_padded // BQ, H_blocks, 1]).astype(np.float32),
                128,
                axis=3,
            )
            inputs["is_block_quant"] = True
        if is_per_tensor:
            # Per-tensor: [E, 2, 1] for gate/up, [E, 1] for down
            inputs["gate_up_proj_scale"] = np.random.uniform(0.01, 0.1, size=[expert, 2, 1]).astype(np.float32)
            inputs["down_proj_scale"] = np.random.uniform(0.01, 0.1, size=[expert, 1]).astype(np.float32)
            # Per-tensor activation: [E, 2, 1] for gate/up hidden, [E, 1] for down hidden
            inputs["gate_up_hidden_scale"] = np.random.uniform(0.01, 0.1, size=[expert, 2, 1]).astype(np.float32)
            inputs["down_hidden_scale"] = np.random.uniform(0.01, 0.1, size=[expert, 1]).astype(np.float32)
            inputs["is_per_tensor"] = True

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
    gate_up_hidden_scale=None,
    down_hidden_scale=None,
    is_block_quant=False,
    is_per_tensor=False,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = _DEFAULT_SKIP_MODE,
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
    down_bias_tp_degree=None,
    down_bias_tp_rank=None,
    non_overlapping_shards=False,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
):
    """Wrapper that dispatches to the correct kernel based on bwmm_func."""
    # lnc_degree and top_k are test-framework parameters not forwarded to kernels
    _ = lnc_degree, top_k, is_block_quant, is_per_tensor

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
    elif bwmm_func == BWMMFunc.SHARD_ON_BLOCK_V2:
        return bwmm_shard_on_block_v2(
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
            else BlockShardStrategy.HI_LO,
            down_bias_tp_degree=down_bias_tp_degree,
            down_bias_tp_rank=down_bias_tp_rank,
            non_overlapping_shards=non_overlapping_shards,
        )
    elif bwmm_func == BWMMFunc.SHARD_ON_BLOCK_HW:
        return bwmm_shard_on_block_hybrid(
            conditions=conditions,
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
            else BlockShardStrategy.HI_LO,
            down_bias_tp_degree=down_bias_tp_degree,
            down_bias_tp_rank=down_bias_tp_rank,
            non_overlapping_shards=non_overlapping_shards,
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
            gate_up_hidden_scale=gate_up_hidden_scale,
            down_hidden_scale=down_hidden_scale,
            is_block_quant=is_block_quant,
            is_per_tensor=is_per_tensor,
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
            accumulation_dtype=accumulation_dtype,
            skip_gate_proj=skip_gate_proj,
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
            accumulation_dtype=accumulation_dtype,
        )
    else:
        assert False, f"Unsupported bwmm_func: {bwmm_func}"  # noqa: B011  # NKI kernels forbid raise; this wrapper is traced as a kernel


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
    gate_up_hidden_scale=None,
    down_hidden_scale=None,
    is_block_quant=False,
    is_per_tensor=False,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = _DEFAULT_SKIP_MODE,
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
    down_bias_tp_degree=None,
    down_bias_tp_rank=None,
    non_overlapping_shards: bool = False,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
) -> dict:
    """Torch reference wrapper matching moe_cte_kernel_wrapper signature."""
    _ = accumulation_dtype
    from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_torch import _moe_cte_torch_ref_impl

    return _moe_cte_torch_ref_impl(
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
        gate_up_hidden_scale=gate_up_hidden_scale,
        down_hidden_scale=down_hidden_scale,
        is_block_quant=is_block_quant,
        is_per_tensor=is_per_tensor,
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
        skip_gate_proj=skip_gate_proj,
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
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    block_sharding_strategy: Optional[str] = None,
    is_block_quant: bool = False,
    is_per_tensor: bool = False,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
):
    """Generate inputs dict compatible with UnitTestFramework (keys match wrapper signature)."""
    _, is_nkijit, is_dynamic = bwmm_func_enum.get_bwmm_func()
    is_block_parallel = bwmm_func_enum in (
        BWMMFunc.SHARD_ON_BLOCK,
        BWMMFunc.SHARD_ON_BLOCK_V2,
        BWMMFunc.SHARD_ON_BLOCK_HW,
    )

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
        down_bias_tp_degree=down_bias_tp_degree,
        down_bias_tp_rank=down_bias_tp_rank,
        is_block_quant=is_block_quant,
        is_per_tensor=is_per_tensor,
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
    if "gate_up_hidden_scale" in raw:
        inputs["gate_up_hidden_scale"] = raw["gate_up_hidden_scale"]
    if "down_hidden_scale" in raw:
        inputs["down_hidden_scale"] = raw["down_hidden_scale"]
    if "is_block_quant" in raw:
        inputs["is_block_quant"] = raw["is_block_quant"]
    if "is_per_tensor" in raw:
        inputs["is_per_tensor"] = raw["is_per_tensor"]
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

    if accumulation_dtype is not None and bwmm_func_enum in (
        BWMMFunc.SHARD_ON_INTERMEDIATE,
        BWMMFunc.SHARD_ON_INTERMEDIATE_HW,
        BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING,
    ):
        inputs["accumulation_dtype"] = accumulation_dtype

    if down_bias_tp_degree is not None:
        inputs["down_bias_tp_degree"] = down_bias_tp_degree
        inputs["down_bias_tp_rank"] = down_bias_tp_rank

    if skip_gate_proj:
        inputs["skip_gate_proj"] = True

    if block_sharding_strategy is not None:
        if block_sharding_strategy == "HI_LO_NO":
            inputs["block_sharding_strategy"] = BlockShardStrategy.HI_LO
            inputs["non_overlapping_shards"] = True
        else:
            inputs["block_sharding_strategy"] = BlockShardStrategy[block_sharding_strategy]

    return inputs


def _shard_on_block_output_validator(
    torch_ref_fn,
    ref_input,
    T_out,
    tokens,
    hidden,
    expert,
    lnc_degree,
    dtype,
    rtol=2e-2,
    atol=1e-5,
):
    """Create a CustomValidator that compares only [:T, 0, :H] of the (T_out, 2, H+E) output."""
    from test.utils.comparators import maxAllClose

    class ShardOnBlockOutputValidator(CustomValidator):
        @override
        def validate(self, actual_raw_output):
            actual = (
                np.frombuffer(actual_raw_output, dtype=ml_dtypes.bfloat16)
                .reshape(T_out, 2, hidden + expert)
                .astype(np.float32)
            )

            golden_dict = torch_ref_fn(**ref_input)
            golden = golden_dict["output"]
            if hasattr(golden, "numpy"):
                golden = golden.numpy()
            golden = golden.astype(np.float32)
            if golden.ndim == 3:
                golden = golden[:, 0, :]  # Extract shard 0 from (T, lnc, H)
            golden = golden.reshape(T_out, hidden)

            # Compare only real tokens [:T] in slot 0, H columns
            actual_h = actual[:tokens, 0, :hidden]
            golden_h = golden[:tokens, :hidden]

            passed = maxAllClose(actual_h, golden_h, rtol=rtol, atol=atol, verbose=1, logfile=self.logfile)

            if not passed and self.logfile is not None:
                import os

                golden_dir = os.path.dirname(self.logfile.name) if hasattr(self.logfile, 'name') else '.'
                golden_path = os.path.join(golden_dir, "golden-output.bin")
                golden.tofile(golden_path)
                self._print_with_log(f"Golden output saved to: {golden_path}")
                actual_path = os.path.join(golden_dir, "actual-output-f32.bin")
                actual.tofile(actual_path)
                self._print_with_log(f"Actual output (f32) saved to: {actual_path}")

            return passed

    return ShardOnBlockOutputValidator


def _shard_on_block_mx_output_validator(
    kernel_input,
    tokens,
    hidden,
    lnc_degree,
    rtol=5e-2,
    atol=1e-5,
):
    """Validator for the MX shard-on-block output layout [lnc, T_out, H].

    Calls bwmm_shard_on_block_mx_torch_ref directly (matching the standalone
    direct test) — its layout matches the kernel output, unlike the
    _moe_cte_torch_ref_impl which emits [T+1, lnc, H]. Compares shard 0 only;
    shard 1 is kernel scratch.
    """
    from inspect import signature

    from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx_torch import bwmm_shard_on_block_mx_torch_ref

    from test.integration.nkilib.core.moe.moe_cte.test_utils import (
        gather_from_packed_down,
        gather_from_packed_gate_up,
    )
    from test.utils.comparators import maxAllClose

    # Filter kernel_input to the ref's signature: kernel_input is keyed against
    # moe_cte() (uses `spec` etc.) which is a superset of the direct ref.
    ref_kwargs = {k: v for k, v in kernel_input.items() if k in signature(bwmm_shard_on_block_mx_torch_ref).parameters}

    # MX-only knobs live on spec.shard_on_block (not on moe_cte()'s top-level signature).
    spec = kernel_input.get("spec")
    if spec and spec.shard_on_block:
        ref_kwargs["weight_dtype"] = spec.shard_on_block.weight_dtype
        # quantization_type selects the STATIC_MX golden path in the torch ref; it
        # rides on the config, not in kernel_input, so forward it explicitly.
        ref_kwargs["quantization_type"] = spec.shard_on_block.quantization_type

    # The ref expects standard scale layout. Mirror _mx_torch_ref_wrapper in
    # test_moe_bwmm_mx_cte.py: gather packed scales back before invoking.
    use_packed = bool(spec and spec.shard_on_block and spec.shard_on_block.use_packed_scales)
    if use_packed:
        # gate_up_proj_weight: (E, _pmax, 2, n_H512_tile, I)
        n_H512_tile = ref_kwargs["gate_up_proj_weight"].shape[3]
        ref_kwargs["gate_up_proj_scale"] = gather_from_packed_gate_up(ref_kwargs["gate_up_proj_scale"], n_H512_tile)
        # down_proj_weight: (E, p_I, n_total_I512_tile, H); p_scale = p_I / _q_height
        dwn_w = ref_kwargs["down_proj_weight"]
        ref_kwargs["down_proj_scale"] = gather_from_packed_down(
            ref_kwargs["down_proj_scale"], dwn_w.shape[2], p_scale=dwn_w.shape[1] // 8
        )

    class ShardOnBlockMxOutputValidator(CustomValidator):
        @override
        def validate(self, actual_raw_output):
            actual = (
                np.frombuffer(actual_raw_output, dtype=ml_dtypes.bfloat16)
                .reshape(lnc_degree, -1, hidden)
                .astype(np.float32)
            )
            golden = bwmm_shard_on_block_mx_torch_ref(**ref_kwargs)["output"]
            if hasattr(golden, "numpy"):
                golden = golden.numpy()
            golden = golden.astype(np.float32)
            return maxAllClose(
                actual[0, :tokens, :hidden],
                golden[0, :tokens, :hidden],
                rtol=rtol,
                atol=atol,
                verbose=1,
                logfile=self.logfile,
            )

    return ShardOnBlockMxOutputValidator


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
    accumulation_dtype=None,
):
    """Generate output tensor placeholders for UnitTestFramework."""
    is_block_parallel = bwmm_func_enum in (
        BWMMFunc.SHARD_ON_BLOCK,
        BWMMFunc.SHARD_ON_BLOCK_V2,
        BWMMFunc.SHARD_ON_BLOCK_HW,
    )
    is_dropping = bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING
    is_shard_block = bwmm_func_enum in (BWMMFunc.SHARD_ON_BLOCK, BWMMFunc.SHARD_ON_BLOCK_V2, BWMMFunc.SHARD_ON_BLOCK_HW)

    dma_skip = kernel_input["skip_dma"]
    T_out = tokens if dma_skip.skip_token else tokens + 1
    separate_outputs = is_block_parallel and top_k > 1

    if separate_outputs:
        if is_shard_block:
            output_shape = (T_out, 2, hidden + expert)
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


# =============================================================================
# Unified moe_cte() entry point wrappers for UnitTestFramework
# =============================================================================

# Module-level storage for MX _internal data lives in src/ so torch_refs can
# share state with test input generators without test/ depending on src/. The
# re-export keeps existing test callers working.
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_func import (  # noqa: E402, F401
    store_mx_internal,
)

# =============================================================================


def generate_moe_cte_unified_inputs(
    impl,
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
    activation_function,
    expert_affinities_scaling_mode,
    gate_clamp_upper=None,
    gate_clamp_lower=None,
    up_clamp_upper=None,
    up_clamp_lower=None,
    expert_affinity_multiply_on_I: bool = False,
    weight_dtype=None,
    is_dynamic: bool = False,
    lnc_degree: int = 2,
    non_overlapping_shards: bool = False,
    use_packed_scales: bool = False,
    ep_degree: int = 1,
    quantization_type: QuantizationType = QuantizationType.MX,
):
    """Generate inputs dict for unified moe_cte() entry point, compatible with UnitTestFramework.

    Keys match moe_cte() signature: uses spec and quantization_config instead of
    raw bwmm_func/gate_up_proj_scale/down_proj_scale.

    Args:
        impl: MoECTEImplementation enum value
        (other args match test parameter columns)

    Returns:
        dict with keys matching moe_cte() parameter names
    """
    from nkilib_src.nkilib.core.moe.moe_cte import (
        MoECTEImplementation,
        MoECTESpec,
        QuantizationConfig,
        ShardOnBlockConfig,
        ShardOnIConfig,
    )

    from test.integration.nkilib.core.moe.moe_cte.test_utils import build_moe_bwmm_mx_cte
    from test.utils.mx_utils import is_mx_quantize

    dma_skip = map_skip_mode(skip)

    # Build spec
    if impl in (MoECTEImplementation.shard_on_block, MoECTEImplementation.shard_on_block_mx):
        _strategy = BlockShardStrategy.HI_LO if non_overlapping_shards else BlockShardStrategy.PING_PONG
        spec = MoECTESpec(
            implementation=impl,
            shard_on_block=ShardOnBlockConfig(
                non_overlapping_shards=non_overlapping_shards,
                block_sharding_strategy=_strategy,
                use_packed_scales=use_packed_scales,
                weight_dtype=weight_dtype,
                top_k=top_k,
                ep_degree=ep_degree,
                quantization_type=quantization_type,
            ),
            shard_on_I=None,
        )
    elif impl in (MoECTEImplementation.shard_on_i, MoECTEImplementation.shard_on_i_dropping):
        spec = MoECTESpec(
            implementation=impl,
            shard_on_block=None,
            shard_on_I=ShardOnIConfig(
                checkpoint_activation=training,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
            ),
        )
    elif impl == MoECTEImplementation.shard_on_i_hybrid:
        spec = MoECTESpec(implementation=impl, shard_on_block=None, shard_on_I=ShardOnIConfig())
    elif impl in (MoECTEImplementation.shard_on_i_mx, MoECTEImplementation.shard_on_i_mx_hybrid):
        spec = MoECTESpec(implementation=impl, shard_on_block=None, shard_on_I=None)
    else:
        spec = MoECTESpec(implementation=impl, shard_on_block=None, shard_on_I=None)

    # MX path
    if is_mx_quantize(weight_dtype):
        skip_mode = (1 if dma_skip.skip_token else 0) + (2 if dma_skip.skip_weight else 0)
        is_shard_on_I = impl in (
            MoECTEImplementation.shard_on_i_mx,
            MoECTEImplementation.shard_on_i_mx_hybrid,
        )
        raw = build_moe_bwmm_mx_cte(
            H=hidden,
            T=tokens,
            E=expert,
            B=block_size,
            TOPK=top_k,
            I_TP=intermediate,
            dtype=dtype,
            weight_dtype=weight_dtype,
            skip_mode=skip_mode,
            bias=bias,
            activation_function=activation_function,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            is_dynamic=is_dynamic,
            vnc_degree=lnc_degree,
            gate_clamp_upper_limit=gate_clamp_upper,
            gate_clamp_lower_limit=gate_clamp_lower,
            up_clamp_upper_limit=up_clamp_upper,
            up_clamp_lower_limit=up_clamp_lower,
            is_shard_on_I=is_shard_on_I,
            use_packed_scales=use_packed_scales,
            quantization_type=quantization_type,
        )
        # Remap to moe_cte() signature - pass scales directly (not in QuantizationConfig)
        # because NKI tracer can't handle tensors wrapped in dataclass
        inputs = {
            "hidden_states": raw["hidden_states"],
            "expert_affinities_masked": raw["expert_affinities_masked"],
            "gate_up_proj_weight": raw["gate_up_proj_weight"],
            "down_proj_weight": raw["down_proj_weight"],
            "token_position_to_id": raw["token_position_to_id"],
            "block_to_expert": raw["block_to_expert"],
            "block_size": raw["block_size"],
            "spec": spec,
            "activation_function": raw["activation_function"],
            "skip_dma": raw["skip_dma"],
            "compute_dtype": raw.get("compute_dtype", dtype),
            "is_tensor_update_accumulating": raw.get("is_tensor_update_accumulating", top_k != 1),
            "expert_affinities_scaling_mode": raw["expert_affinities_scaling_mode"],
        }
        if raw.get("gate_up_proj_scale") is not None:
            inputs["gate_up_proj_scale"] = raw["gate_up_proj_scale"]
        if raw.get("down_proj_scale") is not None:
            inputs["down_proj_scale"] = raw["down_proj_scale"]
        # STATIC_MX per-expert input scales (quantization_type is part of spec.shard_on_block).
        # The per-expert weight scales reuse gate_up_proj_scale / down_proj_scale above.
        for _scale_key in (
            "gate_up_in_scale",
            "down_in_scale",
        ):
            if raw.get(_scale_key) is not None:
                inputs[_scale_key] = raw[_scale_key]
        if "conditions" in raw:
            inputs["conditions"] = raw["conditions"]
        if "gate_and_up_proj_bias" in raw:
            inputs["gate_and_up_proj_bias"] = raw["gate_and_up_proj_bias"]
        if "down_proj_bias" in raw:
            inputs["down_proj_bias"] = raw["down_proj_bias"]
        for clamp_key in (
            "gate_clamp_upper_limit",
            "gate_clamp_lower_limit",
            "up_clamp_upper_limit",
            "up_clamp_lower_limit",
        ):
            if clamp_key in raw:
                inputs[clamp_key] = raw[clamp_key]
        # Store _internal in module-level dict for MX golden computation.
        # store_mx_internal ties the entry's lifetime to the spec object so it
        # is evicted at test teardown (prevents an unbounded per-MX-test leak).
        if "_internal" in raw:
            store_mx_internal(inputs["spec"], raw["_internal"])
        return inputs

    # Non-MX path: reuse build_bwmm_inputs
    # Map impl back to BWMMFunc for build_bwmm_inputs
    from nkilib_src.nkilib.core.moe.moe_cte import MoECTEImplementation

    impl_to_bwmm = {
        MoECTEImplementation.shard_on_block: BWMMFunc.SHARD_ON_BLOCK,
        MoECTEImplementation.shard_on_i: BWMMFunc.SHARD_ON_INTERMEDIATE,
        MoECTEImplementation.shard_on_i_hybrid: BWMMFunc.SHARD_ON_INTERMEDIATE_HW,
        MoECTEImplementation.shard_on_i_dropping: BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING,
    }
    bwmm_func_enum = impl_to_bwmm[impl]
    _, _, is_dyn = bwmm_func_enum.get_bwmm_func()
    is_block_parallel = bwmm_func_enum in (
        BWMMFunc.SHARD_ON_BLOCK,
        BWMMFunc.SHARD_ON_BLOCK_V2,
        BWMMFunc.SHARD_ON_BLOCK_HW,
    )
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
        is_dynamic=is_dyn,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_function=activation_function,
        checkpoint_activation=training,
        gate_clamp_lower_limit=gate_clamp_lower,
        gate_clamp_upper_limit=gate_clamp_upper,
        up_clamp_lower_limit=up_clamp_lower,
        up_clamp_upper_limit=up_clamp_upper,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        non_overlapping_shards=non_overlapping_shards,
    )

    # Remap to moe_cte() signature
    inputs = {
        "hidden_states": raw["hidden_states"],
        "expert_affinities_masked": raw["expert_affinities_masked"],
        "gate_up_proj_weight": raw["gate_up_proj_weight"],
        "down_proj_weight": raw["down_proj_weight"],
        "token_position_to_id": raw["token_position_to_id"],
        "block_to_expert": raw["block_to_expert"],
        "block_size": raw["block_size"],
        "spec": spec,
        "activation_function": raw["activation_function"],
        "skip_dma": raw["skip_dma"],
        "compute_dtype": raw["compute_dtype"],
        "is_tensor_update_accumulating": raw["is_tensor_update_accumulating"],
        "expert_affinities_scaling_mode": raw["expert_affinities_scaling_mode"],
    }
    if "conditions" in raw:
        inputs["conditions"] = raw["conditions"]
    if "gate_and_up_proj_bias" in raw:
        inputs["gate_and_up_proj_bias"] = raw["gate_and_up_proj_bias"]
    if "down_proj_bias" in raw:
        inputs["down_proj_bias"] = raw["down_proj_bias"]
    if raw.get("gate_up_proj_scale") is not None or raw.get("down_proj_scale") is not None:
        inputs["quantization_config"] = QuantizationConfig(
            gate_up_proj_scale=raw.get("gate_up_proj_scale"),
            down_proj_scale=raw.get("down_proj_scale"),
        )
    for clamp_key in (
        "gate_clamp_upper_limit",
        "gate_clamp_lower_limit",
        "up_clamp_upper_limit",
        "up_clamp_lower_limit",
    ):
        if clamp_key in raw:
            inputs[clamp_key] = raw[clamp_key]
    return inputs


def moe_cte_unified_output_tensors(
    kernel_input: dict,
    tokens: int,
    hidden: int,
    intermediate: int,
    expert: int,
    block_size: int,
    top_k: int,
    dtype,
    impl,
    training: bool,
    expert_affinity_multiply_on_I: bool,
    lnc_degree: int = 2,
):
    """Generate output tensor placeholders for unified moe_cte() UnitTestFramework tests.

    Args:
        kernel_input: dict from generate_moe_cte_unified_inputs
        (other args match test parameter columns)

    Returns:
        dict of numpy zero arrays matching expected output shapes
    """

    from nkilib_src.nkilib.core.moe.moe_cte import MoECTEImplementation

    is_block_parallel = impl in (
        MoECTEImplementation.shard_on_block,
        MoECTEImplementation.shard_on_block_mx,
    )
    is_dropping = impl == MoECTEImplementation.shard_on_i_dropping
    is_shard_block = impl == MoECTEImplementation.shard_on_block
    is_mx = impl in (
        MoECTEImplementation.shard_on_block_mx,
        MoECTEImplementation.shard_on_i_mx,
        MoECTEImplementation.shard_on_i_mx_hybrid,
    )

    dma_skip = kernel_input["skip_dma"]
    T_out = tokens if dma_skip.skip_token else tokens + 1
    separate_outputs = is_block_parallel and top_k > 1
    is_accumulating = top_k != 1

    if is_mx:
        if impl == MoECTEImplementation.shard_on_block_mx and is_accumulating:
            output_shape = (lnc_degree, T_out, hidden)
        else:
            output_shape = (T_out, hidden)
    elif separate_outputs:
        if is_shard_block:
            # When non_overlapping_shards is set, v2 kernel produces (T, 2, H+E)
            _non_overlap = kernel_input.get("non_overlapping_shards", False)
            if _non_overlap:
                output_shape = (T_out, 2, hidden + expert)
            else:
                output_shape = (T_out, lnc_degree, hidden)
        else:
            output_shape = (lnc_degree, T_out, hidden)
    else:
        output_shape = (T_out, hidden)

    if is_mx:
        return {"output": np.zeros(output_shape, dtype=dtype)}

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
        result = {
            "output": np.zeros(output_shape, dtype=dtype),
            "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
        }
        if not expert_affinity_multiply_on_I:
            result["down_activations"] = np.zeros((N, block_size, hidden), dtype=dtype)
        return result

    return {"output": np.zeros(output_shape, dtype=dtype)}
