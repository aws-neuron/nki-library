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

# LEGACY SWEEP TEST FRAMEWORK - Uses @range_test_config / RangeTestHarness
# New tests should use @pytest.mark.coverage_parametrize instead

import math
from enum import Enum

try:
    from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_model_config import (
        moe_cte_model_configs,
    )
except ImportError:
    moe_cte_model_configs = []

from functools import lru_cache
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.mx_utils import is_mx_quantize
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
from typing import Callable, Tuple, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_block import bwmm_shard_on_block
from nkilib_src.nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    # ActFnType,
    # ExpertAffinityScaleMode,
    SkipMode,
    blockwise_mm_baseline_shard_intermediate,
    blockwise_mm_baseline_shard_intermediate_hybrid,
    blockwise_mm_shard_intermediate_dropping,
)
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

from ....utils.test_kernel_common import act_fn_type2func

# Constants
_pmax = 128
_q_height = 8
_q_width = 4

# Dimension name constants
# BWMM_CONFIG = "bwmm_config"
# BWMM_FUNC_DIM_NAME = "bwmm_func"
# VNC_DEGREE_DIM_NAME = "vnc"
# TOKENS_DIM_NAME = "tokens"
# HIDDEN_DIM_NAME = "hidden"
# INTERMEDIATE_DIM_NAME = "intermediate"
# EXPERT_DIM_NAME = "expert"
# BLOCK_SIZE_DIM_NAME = "block_size"
# TOP_K_DIM_NAME = "top_k"
# ACT_FN_DIM_NAME = "act_fn"
# EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME = "expert_affinities_scaling_mode"
# DTYPE_DIM_NAME = "dtype"
# SKIP_DIM_NAME = "skip"
# BIAS_DIM_NAME = "bias"
# TRAINING_DIM_NAME = "training"
# QUANTIZE_DIM_NAME = "quantize"
# GATE_CLAMP_UPPER_DIM_NAME = "gate_clamp_upper"
# GATE_CLAMP_LOWER_DIM_NAME = "gate_clamp_lower"
# UP_CLAMP_UPPER_DIM_NAME = "up_clamp_upper"
# UP_CLAMP_LOWER_DIM_NAME = "up_clamp_lower"
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
EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME = 'eamoi'

dtype2dtype_range = {nl.int8: (-127, 127), nl.float8_e4m3: (-240.0, 240.0)}


class BWMMFunc(Enum):
    SHARD_ON_BLOCK = 1
    SHARD_ON_HIDDEN = 2
    SHARD_ON_INTERMEDIATE = 3
    SHARD_ON_INTERMEDIATE_AFFINITY_ON_INTERMEDIATE = 4
    SHARD_ON_INTERMEDIATE_HW = 5  # hybrid while
    SHARD_ON_INTERMEDIATE_DROPPING = 6  # dropping kernel

    MXFP4_SHARD_ON_BLOCK = 10
    MXFP4_SHARD_ON_BLOCK_DYNAMIC = 11

    BASELINE = 100
    BASELINE_ALLOCATED = 200

    def get_bwmm_func(self) -> Tuple[Callable, bool, bool]:
        """
        return the kernel function, whether the kernel is nki.jit, whether it is dynamic
        """
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
    # round to multiple of n_block_per_iter for simplicity
    N = n_block_per_iter * math.ceil(N / n_block_per_iter)
    return N


def get_block_size_dropping(seq_len, batch_size, top_k, capacity_factor, num_experts):
    """Calculate block size for dropping kernel: B = Seq * BS * TOPK * cf // E"""
    return seq_len * batch_size * top_k * capacity_factor // num_experts


def calculate_local_t_dropping(seq_len, batch_size, top_k, capacity_factor, expert_parallelism):
    """Calculate local T for dropping kernel: T = Seq * BS * TOPK * cf // EP"""
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
        """
    when strategy is 5, down_scale is transpose as follows:
    down_proj_scale = np.transpose(down_proj_scale.reshape((E, 128, H//128)), (0, 2, 1)).reshape((E, 1, H))
    """
        return [E, 1, 2 * I_TP], [E, 1, H]
    elif quantize_strategy == 6:
        """
    when strategy is 6, down_scale is not transposed (still E, 1, H)
    """
        return [E, 1, 2 * I_TP], [E, 1, H]
    else:
        raise ValueError("Unrecognized quantize strategy")


def get_router_with_controlled_distribution(T: int, TOPK: int, E: int, alpha: np.float32 = None):
    """
    Generate a uniform or controlled probability distribution over E experts for tokens.

    Args:
    alpha (float32): expert distribution sparsity parameter (smaller = sparser, so less padding blocks);
        when alpha is None, a uniform random distribution will be generated, close to the maximum padding blocks.
        General recommended range is (0.01, 10), some examples data:

    for T=16384, B=256, E=128, N=79, TOPK=1
        alpha               0.01  0.0259  0.035  0.0459  0.061  0.131  0.019 0.0309 0.0498 0.0548 0.802  1.42  3.04 7.897 9.556  None
        padding_blocks       1      5      7      10      15     21     25    31      35    40      46    50    55   60     62    63

    for T=16384, B=256, E=128, N=255, TOPK=2
        alpha               0.01 0.0259  0.035   0.0459  0.061  0.131  0.019 0.0309 0.0498 0.0548 0.802  1.42  3.04 7.897 9.556  None
        padding_blocks       3     7       10     14      17     27     27     36     42     44     46     58    63   71    69    67

    for T=16384, B=256, E=16, N=79, TOPK=1, the recommended alpha range is: (0.03, 5)
        alpha               0.01 0.0259  0.035   0.0459  0.061  0.131  0.019 0.0309 0.0498 0.0548 0.802  1.42  3.04 7.897 9.556  None
        padding_blocks       0     0       0       1      1      2      5      6     7       5     6     5     9     6     7     8
    """

    router = np.zeros((T, TOPK))
    np.random.seed(0)  # set random seed is required for while loop test stability
    if alpha == None:
        for i in range(T):
            # uniform random distribution over experts as the default config
            router[i] = np.random.choice(range(E), (TOPK), replace=False)
    elif alpha == -1:
        # Dirichlet distribution probabilities of experts by controlled sparsity (sums to 1).
        p_of_e = np.random.dirichlet(np.ones(E) * 0.001)
        for i in range(0, T):
            # Generate a sparse probability distribution over E experts.
            router[i] = np.random.choice(range(E), (TOPK), replace=False, p=p_of_e)
    else:
        for i in range(E):
            # Make sure each expert has at least 1 token routed to
            router[i] = np.arange(i, (i + TOPK) % T)
        # Dirichlet distribution probabilities of experts by controlled sparsity (sums to 1).
        p_of_e = np.random.dirichlet(np.ones(E) * alpha)
        for i in range(E, T):
            # Generate a sparse probability distribution over E experts.
            router[i] = np.random.choice(range(E), (TOPK), replace=False, p=p_of_e)
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
    """
    use_split_padding: insert padding blocks half and half to middle and end of the expert,
    otherwise only insert to the end
    """
    if n_block_per_iter > 1:
        assert vnc_degree == 1

    router = get_router_with_controlled_distribution(T=T, TOPK=TOPK, E=E, alpha=alpha)
    one_hot = np.arange(E)
    token_experts = np.zeros((T, E))
    for i in range(TOPK):
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
            conditions[-(n_padding_block + 1) :] = 0
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
            np.clip(
                gate_activation,
                a_min=gate_clamp_lower_limit,
                a_max=gate_clamp_upper_limit,
                out=gate_activation,
            )

        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            np.clip(
                up_activation,
                a_min=up_clamp_lower_limit,
                a_max=up_clamp_upper_limit,
                out=up_activation,
            )

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
        # For dropping kernel: N = expert (number of blocks = number of experts)
        N = expert
        token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts_dropping(
            T=tokens, E_Local=expert, B=block_size, rtype=rtype, topK=top_k
        )
        # For dropping, expert_masks is all ones since tokens are directly assigned
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

    # Generate hidden states and expert affinities
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

    # Generate bias tensors
    gate_up_proj_bias = None
    down_proj_bias = None
    if bias:
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[expert, hidden]).astype(dtype)
        gate_up_proj_bias = np.random.uniform(-1, 1, size=[expert, 2, intermediate]).astype(dtype)

    # Generate weight and scale tensors
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
        down_proj_weights = down_proj_weights_padded[:, :intermediate, :]
    else:
        down_proj_weights_padded = np.random.uniform(-0.1, 0.1, size=[expert, I_TP_padded, hidden]).astype(dtype)
        gate_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[expert, hidden, 2, intermediate]).astype(dtype)
        down_proj_weights = down_proj_weights_padded[:, :intermediate, :]

    # Build input dictionary with tensor inputs (matching kernel parameter names)
    # Add conditions tensor for dynamic kernels
    inputs = {}
    if is_dynamic:
        inputs["conditions"] = conditions

    inputs.update(
        {
            # Tensor inputs for kernel
            "hidden_states": hidden_states,
            "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
            "gate_up_proj_weight": gate_up_proj_weights,
            "down_proj_weight": down_proj_weights_padded,
            "token_position_to_id": token_position_to_id,
            "block_to_expert": block_to_expert,
            # Non-tensor kernel parameters
            "block_size": block_size,
            "skip_dma": dma_skip,
            "compute_dtype": dtype,
            "is_tensor_update_accumulating": top_k != 1,
            "expert_affinities_scaling_mode": expert_affinities_scaling_mode,
            "activation_function": activation_function,
        }
    )

    # Add bias tensors
    if bias:
        inputs["gate_and_up_proj_bias"] = gate_up_proj_bias
        inputs["down_proj_bias"] = down_proj_bias

    # Add quantization scale tensors
    if quantize:
        inputs["gate_up_proj_scale"] = gate_up_proj_scale
        inputs["down_proj_scale"] = down_proj_scale

    # Add clamp parameters
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

    # Dropping kernel always uses checkpoint_activation=True implicitly (returns activations)
    if bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING:
        inputs.pop("checkpoint_activation", None)  # Dropping kernel doesn't have this param

    return inputs


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
        N = expert  # For dropping kernel, N = E
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


# fmt: off
moe_blockwise_mm_lnc2_kernel_test_params = [
# # bwmm_func,                           hidden, tokens, expert, block_size, top_k, intermediate, dtype,      skip, bias, training, quantize,        act_fn,          expert_affinities_scaling_mode,    gate_cl_upper, gate_cl_lower, up_cl_upper, up_cl_lower, expert_affinity_multiply_on_I
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 0,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# # test clamping fp8
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   False, False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          7,             None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          7,           None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        7,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 0,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
# # Skip token DMA skipping
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.NO_SCALE,   None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  64,     512,        8,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    7168,   10240,  1,      512,        1,     2048,         nl.bfloat16, 1,   False, False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.PRE_SCALE,  None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# # test clamping + fp8
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          7,             None,        None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          7,           None,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        7,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   1024,   8,      512,        4,     768,          nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
# # 10K approximation config
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    nl.float8_e4m3,  ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
# [BWMMFunc.SHARD_ON_INTERMEDIATE_HW,    3072,   10240,  8,      512,        4,     1536,         nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          7,           -7,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      512,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     384,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 1,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   1024,   8,      256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
[BWMMFunc.SHARD_ON_BLOCK,              3072,   10240,   128,      256,        4,     192,          nl.bfloat16, 3,   True,  False,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, 7,             None,          8,           -9,        False],
# Test for expert affinity on I dimension
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
# Test for training, expert affinity on H dimension
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 0,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     2048,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   64,     512,        8,     3072,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     2048,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE,       3072,   1024,   8,      512,        4,     3072,         nl.bfloat16, 1,   False,  True,    None,            ActFnType.Swish, ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
# Dropping kernel tests
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536,   8192,   2,      4096,       2,     6144,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 2048,   2048,   2,      1024,       2,     8192,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 4096,   4096,   2,      2048,       8,     1536,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        True],
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 1536,   8192,   2,      4096,       2,     6144,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 2048,   2048,   2,      1024,       2,     8192,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
[BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING, 4096,   4096,   2,      2048,       8,     1536,         nl.bfloat16, 0,   False, True,     None,            ActFnType.SiLU,  ExpertAffinityScaleMode.POST_SCALE, None,          None,          None,        None,        False],
]
# fmt: on


def _create_bwmm_test_config(test_params_list, test_type="manual"):
    """Create RangeTestConfig from parameter list."""
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


@lru_cache(maxsize=1)
def _get_moe_cte_metadata():
    return load_model_configs("test_moe_cte")


@pytest_test_metadata(
    name="MoE Blockwise MatMul LNC2",
    pytest_marks=["moe", "blockwise_mm", "lnc2"],
)
@final
class TestMoeBlockwiseMatMulLnc2:
    """
    Tests for LNC2 blockwise matmul, across different sharding axis (Batch, Hidden, Intermediate)

    skip modes:
    - 0: SkipMode(False, False)
    - 1: SkipMode(True, False)  - skip token
    - 2: SkipMode(False, True)  - skip weight
    - 3: SkipMode(True, True)   - skip both
    """

    def run_bwmm_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        collector: IMetricsCollector,
    ):
        is_negative_test_case = False

        bwmm_config = test_options.tensors[BWMM_CONFIG]

        bwmm_func_enum: BWMMFunc = bwmm_config[BWMM_FUNC_DIM_NAME]
        tokens = bwmm_config[TOKENS_DIM_NAME]
        hidden = bwmm_config[HIDDEN_DIM_NAME]
        intermediate = bwmm_config[INTERMEDIATE_DIM_NAME]
        expert = bwmm_config[EXPERT_DIM_NAME]
        block_size = bwmm_config[BLOCK_SIZE_DIM_NAME]
        top_k = bwmm_config[TOP_K_DIM_NAME]
        act_fn = bwmm_config[ACT_FN_DIM_NAME]
        expert_affinities_scaling_mode = bwmm_config[EXPERT_AFFINITIES_SCALING_MODE_DIM_NAME]
        dtype = bwmm_config[DTYPE_DIM_NAME]
        skip = bwmm_config[SKIP_DIM_NAME]
        bias = bwmm_config[BIAS_DIM_NAME]
        training = bwmm_config[TRAINING_DIM_NAME]
        quantize = bwmm_config[QUANTIZE_DIM_NAME]
        gate_clamp_upper = bwmm_config[GATE_CLAMP_UPPER_DIM_NAME]
        gate_clamp_lower = bwmm_config[GATE_CLAMP_LOWER_DIM_NAME]
        up_clamp_upper = bwmm_config[UP_CLAMP_UPPER_DIM_NAME]
        up_clamp_lower = bwmm_config[UP_CLAMP_LOWER_DIM_NAME]
        expert_affinity_multiply_on_I = bwmm_config[EXPERT_AFFINITY_MULTIPLY_ON_I_DIM_NAME]

        blockwise_mm_func, is_nkijit, is_dynamic = bwmm_func_enum.get_bwmm_func()
        is_block_parallel = blockwise_mm_func in (bwmm_shard_on_block,)
        is_shard_block = bwmm_func_enum == BWMMFunc.SHARD_ON_BLOCK
        is_dropping = bwmm_func_enum == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING

        dma_skip = map_skip_mode(skip)
        quantize_strategy = 6 if quantize else 0

        with assert_negative_test_case(is_negative_test_case):
            kernel_input = build_bwmm_inputs(
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
                activation_function=act_fn,
                checkpoint_activation=training,
                gate_clamp_lower_limit=gate_clamp_lower,
                gate_clamp_upper_limit=gate_clamp_upper,
                up_clamp_lower_limit=up_clamp_lower,
                up_clamp_upper_limit=up_clamp_upper,
                expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
            )

            # breakpoint()
            def create_lazy_golden():
                return golden_bwmm(
                    inp_np=kernel_input,
                    lnc_degree=lnc_degree,
                    tokens=tokens,
                    hidden=hidden,
                    intermediate=intermediate,
                    expert=expert,
                    block_size=block_size,
                    dtype=dtype,
                    dma_skip=dma_skip,
                    bias=bias,
                    quantize=quantize,
                    quantize_strategy=quantize_strategy,
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    activation_function=act_fn,
                    checkpoint_activation=training or is_dropping,  # Dropping always returns activations
                    is_block_parallel=is_block_parallel,
                    top_k=top_k,
                    gate_clamp_lower_limit=gate_clamp_lower,
                    gate_clamp_upper_limit=gate_clamp_upper,
                    up_clamp_lower_limit=up_clamp_lower,
                    up_clamp_upper_limit=up_clamp_upper,
                    is_shard_block=is_shard_block,
                    expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
                    is_dropping=is_dropping,
                )

            # Determine output shape
            separate_outputs = is_block_parallel and top_k > 1
            T_out = tokens if dma_skip.skip_token else tokens + 1
            if separate_outputs:
                if is_shard_block:
                    output_shape = (T_out, lnc_degree, hidden)
                else:
                    output_shape = (lnc_degree, T_out, hidden)
            else:
                output_shape = (T_out, hidden)

            # For dropping kernel, N = expert; otherwise use get_n_blocks
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

            if training or is_dropping:  # Dropping kernel always returns activations
                if expert_affinity_multiply_on_I:
                    output_placeholder = {
                        "output": np.zeros(output_shape, dtype=dtype),
                        "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
                    }
                else:
                    output_placeholder = {
                        "output": np.zeros(output_shape, dtype=dtype),
                        "gate_up_activations_T": np.zeros((N, 2, intermediate, block_size), dtype=dtype),
                        "down_activations": np.zeros((N, block_size, hidden), dtype=dtype),
                    }
            else:
                output_placeholder = {"output": np.zeros(output_shape, dtype=dtype)}

            # Set tolerance based on dtype and quantization
            if quantize:
                rtol, atol = 5e-2, 1e-5
            else:
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
                    kernel_func=blockwise_mm_func,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=validation_args,
                )
            )

    @staticmethod
    def bwmm_lnc2_config():
        """Create combined config with manual and model test cases."""
        manual_config = _create_bwmm_test_config(moe_blockwise_mm_lnc2_kernel_test_params, test_type="manual")
        model_config = _create_bwmm_test_config(moe_cte_model_configs, test_type=MODEL_TEST_TYPE)
        manual_config.global_tensor_configs.custom_generators.extend(
            model_config.global_tensor_configs.custom_generators
        )
        return manual_config

    @pytest.mark.fast
    @range_test_config(bwmm_lnc2_config())
    def test_moe_blockwise_mm_kernel_lnc2(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
        request,
    ):
        bwmm_config = range_test_options.tensors[BWMM_CONFIG]

        # Add metadata dimensions for model configs
        if range_test_options.test_type == MODEL_TEST_TYPE:
            moe_cte_metadata_list = _get_moe_cte_metadata()
            # Pass entire config dict - all params used for metadata matching
            collector.match_and_add_metadata_dimensions(bwmm_config, moe_cte_metadata_list)

        lnc_count = bwmm_config[VNC_DEGREE_DIM_NAME]

        compiler_args = CompilerArgs(logical_nc_config=lnc_count)

        self.run_bwmm_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            collector=collector,
        )
