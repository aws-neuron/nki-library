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

"""
Tests for blockwise_mm_baseline_shard_hidden kernel (H-shard MoE).
"""

import math
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    RangeManualGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorRangeConfig,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ExpertAffinityScaleMode
from nkilib_src.nkilib.experimental.moe.forward.bwmm_shard_on_H import (
    SkipMode,
    blockwise_mm_baseline_shard_hidden,
)

from ...utils.test_kernel_common import silu

# Dimension name constants
BWMM_H_CONFIG = "cfg"
TOKENS_DIM = "tok"
HIDDEN_DIM = "hid"
INTERMEDIATE_DIM = "int"
EXPERT_DIM = "exp"
BLOCK_SIZE_DIM = "bs"
TOP_K_DIM = "k"
SCALING_MODE_DIM = "sm"
DTYPE_DIM = "dt"
SKIP_DIM = "sk"
TRAINING_DIM = "tr"


def get_n_blocks(T, TOPK, E, B):
    """Calculate number of blocks needed."""
    N = math.ceil((T * TOPK - (E - 1)) / B) + E - 1
    return N


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


def generate_token_position_to_id_and_experts(T, TOPK, E, B, N, dma_skip):
    """Generate token position to ID mapping and block to expert mapping."""
    np.random.seed(0)

    router = np.zeros((T, TOPK), dtype=np.int32)
    for i in range(T):
        router[i] = np.random.choice(range(E), TOPK, replace=False)

    one_hot = np.arange(E)
    token_experts = np.zeros((T, E))
    for i in range(TOPK):
        token_experts += np.expand_dims(router[:, i], 1) == np.expand_dims(one_hot, 0)

    blocks_per_expert = np.ceil(token_experts.sum(0) / B).astype(np.int32)
    n_padding_block = N - np.sum(blocks_per_expert)
    blocks_per_expert[E - 1] += n_padding_block

    block_to_expert = np.arange(E).repeat(blocks_per_expert).astype(np.int32)

    cumulative_blocks_per_expert = np.cumsum(blocks_per_expert)
    token_position_by_id_and_expert = np.cumsum(token_experts, axis=0)
    expert_block_offsets = cumulative_blocks_per_expert * B
    token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
    token_position_by_id_and_expert = np.where(token_experts, token_position_by_id_and_expert, 0).astype(np.int32)

    if dma_skip.skip_token:
        token_position_to_id = np.full((int(N * B + 1),), -1, dtype=np.int32)
    else:
        token_position_to_id = np.full((int(N * B + 1),), T, dtype=np.int32)

    tokens_ids = np.arange(T)
    token_position_to_id[token_position_by_id_and_expert] = np.expand_dims(tokens_ids, 1)
    token_position_to_id = token_position_to_id[1:].astype(np.int32)

    return token_experts, token_position_to_id, block_to_expert


def generate_numpy_golden(
    hidden_states,
    expert_affinities,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    T,
    H,
    B,
    N,
    E,
    I_TP,
    dtype,
    dma_skip,
    expert_affinities_scaling_mode,
    checkpoint_activation=False,
):
    """Generate numpy golden output for validation."""
    T_out = T if dma_skip.skip_token else T + 1
    output_np = np.zeros((T_out, H), dtype=dtype)
    token_position_to_id_reshaped = token_position_to_id.reshape(N, B)

    gate_up_weights = gate_up_proj_weight.reshape(E, H, 2 * I_TP).astype(np.float32)
    down_weights = down_proj_weight.astype(np.float32)

    if checkpoint_activation:
        gate_up_activations_T = np.zeros([N, 2, I_TP, B], dtype=dtype)
        down_activations = np.zeros([N, B, H], dtype=dtype)

    for b in range(N):
        local_token_ids = token_position_to_id_reshaped[b, :]
        local_hidden = hidden_states[local_token_ids, :].astype(np.float32)

        expert_idx = block_to_expert[b]
        local_affinities = expert_affinities[local_token_ids, expert_idx].reshape(-1, 1).astype(dtype)

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
            local_hidden = local_affinities * local_hidden

        # Gate and up projections
        gate_up = np.matmul(local_hidden, gate_up_weights[expert_idx]).reshape(B, 2, I_TP)
        gate = gate_up[:, 0, :]
        up = gate_up[:, 1, :]

        if checkpoint_activation:
            # Store gate_up_activations_T: [N, 2, I_TP, B] - transposed from [B, 2, I_TP]
            gate_up_activations_T[b] = gate_up.transpose(1, 2, 0)

        # Activation and intermediate
        intermediate = silu(gate) * up

        # Down projection
        down = np.matmul(intermediate, down_weights[expert_idx])

        if checkpoint_activation:
            # Store down_activations: [N, B, H] - before affinity scaling
            down_activations[b] = down.astype(dtype)

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            down = down * local_affinities

        output_np[local_token_ids, :] += down.astype(dtype)

    if checkpoint_activation:
        return output_np, gate_up_activations_T, down_activations

    return output_np


def build_bwmm_shard_h_inputs(
    tokens, hidden, intermediate, expert, block_size, top_k, dtype, dma_skip, scaling_mode, checkpoint_activation=False
):
    """Build input tensors for the H-shard kernel."""
    T, H, I_TP, E, B = tokens, hidden, intermediate, expert, block_size
    N = get_n_blocks(T, top_k, E, B)
    I_TP_padded = math.ceil(I_TP / 16) * 16

    expert_masks, token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts(
        T, top_k, E, B, N, dma_skip
    )

    np.random.seed(0)
    if dma_skip.skip_token:
        hidden_states = np.random.random_sample([T, H]).astype(dtype)
        expert_affinities = (np.random.rand(T, E) * expert_masks).astype(dtype)
    else:
        hidden_states = np.random.random_sample([T + 1, H]).astype(dtype)
        expert_affinities = np.random.random_sample([T + 1, E]).astype(dtype)
        expert_affinities[:T] = expert_affinities[:T] * expert_masks
        expert_affinities[T] = 0
        hidden_states[T, ...] = 0

    expert_affinities_masked = expert_affinities.reshape(-1, 1).astype(dtype)
    gate_up_proj_weight = np.random.uniform(-0.1, 0.1, size=[E, H, 2, I_TP]).astype(dtype)
    down_proj_weight = np.random.uniform(-0.1, 0.1, size=[E, I_TP_padded, H]).astype(dtype)

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked,
        "gate_up_proj_weight": gate_up_proj_weight,
        "down_proj_weight": down_proj_weight,
        "block_size": B,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert.reshape(-1, 1),
        "skip_dma": dma_skip,
        "compute_dtype": dtype,
        "is_tensor_update_accumulating": top_k != 1,
        "expert_affinities_scaling_mode": scaling_mode,
    }

    if checkpoint_activation:
        # Add checkpoint tensors: gate_up_activations_T [N, 2, I_TP, B], down_activations [N, B, H]
        inputs["gate_up_activations_T"] = np.zeros([N, 2, I_TP, B], dtype=dtype)
        inputs["down_activations"] = np.zeros([N, B, H], dtype=dtype)

    return inputs, expert_affinities, N


def golden_bwmm_shard_h(
    inp_np,
    tokens,
    hidden,
    intermediate,
    expert,
    block_size,
    top_k,
    dtype,
    dma_skip,
    scaling_mode,
    checkpoint_activation=False,
):
    """Generate golden output for H-shard kernel."""
    N = get_n_blocks(tokens, top_k, expert, block_size)
    expert_affinities = inp_np["expert_affinities_masked"].reshape(-1, expert)
    down_proj_weights = inp_np["down_proj_weight"][:, :intermediate, :]
    token_position_to_id = inp_np["token_position_to_id"]
    block_to_expert = inp_np["block_to_expert"].flatten()
    gate_up_proj_weights = inp_np["gate_up_proj_weight"]
    hidden_states = inp_np["hidden_states"]

    result = generate_numpy_golden(
        hidden_states=hidden_states,
        expert_affinities=expert_affinities,
        gate_up_proj_weight=gate_up_proj_weights,
        down_proj_weight=down_proj_weights,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        T=tokens,
        H=hidden,
        B=block_size,
        N=N,
        E=expert,
        I_TP=intermediate,
        dtype=dtype,
        dma_skip=dma_skip,
        expert_affinities_scaling_mode=scaling_mode,
        checkpoint_activation=checkpoint_activation,
    )

    if checkpoint_activation:
        output, gate_up_activations_T, down_activations = result
        return {
            "output": output,
            "gate_up_activations_T": gate_up_activations_T,
            "down_activations": down_activations,
        }

    return {"output": result}


# fmt: off
moe_bwmm_shard_h_test_params = [
# hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, scaling_mode,                          checkpoint_activation
# Inference tests
# deepseek
[7168,    1024,   256,    512,        8,     32,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
[7168,    1024,   256,    256,        8,     32,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
[7168,    1024,   256,    256,        8,     32,           nl.bfloat16, 2,    ExpertAffinityScaleMode.POST_SCALE,     False],
# H = 6144
[6144,    4096,   16,     512,        4,     336,          nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
[6144,    4096,   16,     512,        4,     336,          nl.bfloat16, 1,    ExpertAffinityScaleMode.POST_SCALE,     False],
[6144,    4096,   16,     512,        4,     336,          nl.bfloat16, 2,    ExpertAffinityScaleMode.POST_SCALE,     False],
[6144,    4096,   16,     1024,       4,     336,          nl.bfloat16, 1,    ExpertAffinityScaleMode.POST_SCALE,     False],
# K = 1
[6144,    4096,   16,     512,        1,     336,          nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
[5120,    8192,   16,     256,        1,     128,          nl.bfloat16, 2,    ExpertAffinityScaleMode.POST_SCALE,     False],
# float32
[6144,    4096,   16,     512,        4,     336,          nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     False],
# Llama 4 - OLD TP64
[5120,    8192,   16,     256,        1,     128,          nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False],
[5120,    8192,   16,     256,        1,     128,          nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False],
# Llama 4 - OLD TP16
[5120,    8192,   16,     256,        1,     512,          nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False],
[5120,    8192,   16,     256,        1,     512,          nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False],
# Llama 4 - NEW
[5120,    8192,   128,    128,        2,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False],
[5120,    8192,   128,    128,        2,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
[5120,    8192,   128,    128,        2,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False],
[5120,    8192,   128,    128,        4,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False],
[5120,    8192,   128,    128,        4,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False],
[5120,    8192,   128,    128,        4,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
# odd hidden distribution
[1536,    4096,   234,    128,        7,     288,          nl.bfloat16, 1,    ExpertAffinityScaleMode.POST_SCALE,     False],
# inference with large intermediate
[6144,    4096,   16,     512,        4,     1024,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False],
[6144,    4096,   16,     512,        4,     1024,         nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     False],
# Training tests
[5120,    8192,   128,    256,        4,     1024,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True],
[5120,    8192,   16,     256,        4,     1024,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True],
[5120,    8192,   16,     256,        4,     1024,         nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     True],
[5120,    8192,   128,    256,        1,     128,          nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True],
[5120,    8192,   128,    256,        1,     128,          nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     True],
# GPT-OSS configs
[2880,   8192,   128,     512,        4,     2880,         nl.bfloat16, 0, ExpertAffinityScaleMode.POST_SCALE,        True],
[2880,   8192,   128,     512,        4,     2880,         nl.bfloat16, 0, ExpertAffinityScaleMode.POST_SCALE,        False]

]
# fmt: on

CHECKPOINT_ACTIVATION_DIM = "ca"


@pytest_test_metadata(
    name="MoE Blockwise MatMul H-Shard LNC2",
    pytest_marks=["moe", "blockwise_mm", "lnc2", "h_shard"],
)
@final
class TestMoeBlockwiseMatMulShardH:
    """Tests for H-shard blockwise matmul kernel."""

    def run_bwmm_h_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree: int,
        collector: IMetricsCollector,
    ):
        cfg = test_options.tensors[BWMM_H_CONFIG]

        tokens = cfg[TOKENS_DIM]
        hidden = cfg[HIDDEN_DIM]
        intermediate = cfg[INTERMEDIATE_DIM]
        expert = cfg[EXPERT_DIM]
        block_size = cfg[BLOCK_SIZE_DIM]
        top_k = cfg[TOP_K_DIM]
        dtype = cfg[DTYPE_DIM]
        skip = cfg[SKIP_DIM]
        scaling_mode = cfg[SCALING_MODE_DIM]
        checkpoint_activation = cfg[CHECKPOINT_ACTIVATION_DIM]

        dma_skip = map_skip_mode(skip)

        kernel_input, _, N = build_bwmm_shard_h_inputs(
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            expert=expert,
            block_size=block_size,
            top_k=top_k,
            dtype=dtype,
            dma_skip=dma_skip,
            scaling_mode=scaling_mode,
            checkpoint_activation=checkpoint_activation,
        )

        def create_lazy_golden():
            result = golden_bwmm_shard_h(
                inp_np=kernel_input,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                scaling_mode=scaling_mode,
                checkpoint_activation=checkpoint_activation,
            )
            # When checkpoint_activation=True, result is a dict; only validate output
            # (checkpoint tensors are input-output and can't be in output_placeholder)
            if checkpoint_activation:
                return {"output": result["output"]}
            return result

        T_out = tokens if dma_skip.skip_token else tokens + 1
        output_placeholder = {"output": np.zeros((T_out, hidden), dtype=dtype)}

        validation_args = ValidationArgs(
            golden_output=LazyGoldenGenerator(
                lazy_golden_generator=create_lazy_golden,
                output_ndarray=output_placeholder,
            ),
            relative_accuracy=2e-2,
            absolute_accuracy=1e-5,
        )

        test_manager.execute(
            KernelArgs(
                kernel_func=blockwise_mm_baseline_shard_hidden,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=validation_args,
            )
        )

    @staticmethod
    def bwmm_h_config():
        test_cases = []

        for test_params in moe_bwmm_shard_h_test_params:
            (
                hidden,
                tokens,
                expert,
                block_size,
                top_k,
                intermediate,
                dtype,
                skip,
                scaling_mode,
                checkpoint_activation,
            ) = test_params

            test_case = {
                BWMM_H_CONFIG: {
                    TOKENS_DIM: tokens,
                    HIDDEN_DIM: hidden,
                    INTERMEDIATE_DIM: intermediate,
                    EXPERT_DIM: expert,
                    BLOCK_SIZE_DIM: block_size,
                    TOP_K_DIM: top_k,
                    DTYPE_DIM: dtype,
                    SKIP_DIM: skip,
                    SCALING_MODE_DIM: scaling_mode,
                    CHECKPOINT_ACTIVATION_DIM: checkpoint_activation,
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
    @range_test_config(bwmm_h_config())
    def test_moe_blockwise_mm_shard_h_lnc2(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        lnc_count = 2  # H-shard uses LNC2
        compiler_args = CompilerArgs(logical_nc_config=lnc_count)

        self.run_bwmm_h_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            test_options=range_test_options,
            lnc_degree=lnc_count,
            collector=collector,
        )
