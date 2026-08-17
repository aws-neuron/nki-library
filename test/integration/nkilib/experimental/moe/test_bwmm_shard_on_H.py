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
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ExpertAffinityScaleMode
from nkilib_src.nkilib.experimental.moe.forward.bwmm_shard_on_H import (
    SkipMode,
    blockwise_mm_baseline_shard_hidden,
)
from nkilib_src.nkilib.experimental.moe.forward.bwmm_shard_on_H_torch import (
    blockwise_mm_baseline_shard_hidden_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def get_n_blocks(T, TOPK, E, B):
    """Calculate number of blocks needed."""
    return math.ceil((T * TOPK - (E - 1)) / B) + E - 1


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


def build_bwmm_shard_h_inputs(
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
    activation_dtype=None,
    accum_dtype=None,
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

    # Optional mixed-precision knobs (default to kernel defaults when unset).
    if activation_dtype is not None:
        inputs["activation_dtype"] = activation_dtype
    if accum_dtype is not None:
        inputs["accum_dtype"] = accum_dtype

    if checkpoint_activation:
        inputs["gate_up_activations_T.must_alias_input"] = np.zeros([N, 2, I_TP, B], dtype=dtype)
        inputs["down_activations"] = np.zeros([N, B, H], dtype=dtype)

    return inputs


# fmt: off
PARAM_NAMES = \
    "hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, scaling_mode, checkpoint_activation"
TEST_PARAMS = [
# hidden, tokens, expert, block_size, top_k, intermediate, dtype,       skip, scaling_mode,                          checkpoint_activation
# Inference tests
# deepseek
(7168,    1024,   256,    512,        8,     32,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
(7168,    1024,   256,    256,        8,     32,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
(7168,    1024,   256,    256,        8,     32,           nl.bfloat16, 2,    ExpertAffinityScaleMode.POST_SCALE,     False),
# H = 6144
(6144,    4096,   16,     512,        4,     336,          nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
(6144,    4096,   16,     512,        4,     336,          nl.bfloat16, 1,    ExpertAffinityScaleMode.POST_SCALE,     False),
(6144,    4096,   16,     512,        4,     336,          nl.bfloat16, 2,    ExpertAffinityScaleMode.POST_SCALE,     False),
(6144,    4096,   16,     1024,       4,     336,          nl.bfloat16, 1,    ExpertAffinityScaleMode.POST_SCALE,     False),
# K = 1
(6144,    4096,   16,     512,        1,     336,          nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
pytest.param(5120,    8192,   16,     256,        1,     128,          nl.bfloat16, 2,    ExpertAffinityScaleMode.POST_SCALE,     False, marks=pytest.mark.fast),
# float32
(6144,    4096,   16,     512,        4,     336,          nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     False),
# Llama 4 - OLD TP64
(5120,    8192,   16,     256,        1,     128,          nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False),
pytest.param(5120,    8192,   16,     256,        1,     128,          nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False, marks=pytest.mark.fast),
# Llama 4 - OLD TP16
(5120,    8192,   16,     256,        1,     512,          nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False),
(5120,    8192,   16,     256,        1,     512,          nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False),
# Llama 4 - NEW
(5120,    8192,   128,    128,        2,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False),
(5120,    8192,   128,    128,        2,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
(5120,    8192,   128,    128,        2,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False),
(5120,    8192,   128,    128,        4,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.PRE_SCALE,      False),
(5120,    8192,   128,    128,        4,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.NO_SCALE,       False),
(5120,    8192,   128,    128,        4,     64,           nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
# odd hidden distribution
(1536,    4096,   234,    128,        7,     288,          nl.bfloat16, 1,    ExpertAffinityScaleMode.POST_SCALE,     False),
# inference with large intermediate
(6144,    4096,   16,     512,        4,     1024,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
(6144,    4096,   16,     512,        4,     1024,         nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     False),
# Training tests
(5120,    8192,   128,    256,        4,     1024,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True),
(5120,    8192,   16,     256,        4,     1024,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True),
(5120,    8192,   16,     256,        4,     1024,         nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     True),
(5120,    8192,   128,    256,        1,     128,          nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True),
(5120,    8192,   128,    256,        1,     128,          nl.float32,  0,    ExpertAffinityScaleMode.POST_SCALE,     True),
# GPT-OSS configs
(2880,    8192,   128,    512,        4,     2880,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     True),
(2880,    8192,   128,    512,        4,     2880,         nl.bfloat16, 0,    ExpertAffinityScaleMode.POST_SCALE,     False),
]
# fmt: on

_ABBREVS = {
    "hidden": "hid",
    "tokens": "tok",
    "expert": "exp",
    "block_size": "bs",
    "top_k": "k",
    "intermediate": "int",
    "dtype": "dt",
    "activation_dtype": "act_dt",
    "accum_dtype": "accum_dt",
    "skip": "sk",
    "scaling_mode": "sm",
    "checkpoint_activation": "ca",
}


# fmt: off
# Mixed-precision combinations: compute_dtype x activation_dtype x accum_dtype.
# Covers the bf16-io + fp32 cross-expert accumulation opt-in path (exercises the HBM->HBM down-cast).
PARAM_NAMES_COMBO = \
    "hidden, tokens, expert, block_size, top_k, intermediate, dtype, activation_dtype, accum_dtype, scaling_mode"
DTYPE_COMBO_PARAMS = [
# bf16 io: default (fp32 SwiGLU activation, bf16 accumulation == baseline)
(5120, 8192, 16, 256, 1, 512, nl.bfloat16, nl.float32,  nl.bfloat16, ExpertAffinityScaleMode.POST_SCALE),
# bf16 io + fp32 cross-expert/cross-block accumulation (opt-in; exercises fp32->bf16 HBM down-cast)
(5120, 8192, 16, 256, 1, 512, nl.bfloat16, nl.float32,  nl.float32,  ExpertAffinityScaleMode.POST_SCALE),
# bf16 io + bf16 SwiGLU activation (legacy all-bf16, no precision promotion)
(5120, 8192, 16, 256, 1, 512, nl.bfloat16, nl.bfloat16, nl.bfloat16, ExpertAffinityScaleMode.POST_SCALE),
# fp32 io (full precision)
(6144, 4096, 16, 512, 4, 336, nl.float32,  nl.float32,  nl.float32,  ExpertAffinityScaleMode.POST_SCALE),
]
# fmt: on


@pytest_test_metadata(name="MoE Blockwise MatMul H-Shard LNC2")
@pytest_marks(["moe", "blockwise_mm", "lnc2", "h_shard"])
@final
class TestMoeBlockwiseMatMulShardH:
    """Tests for H-shard blockwise matmul kernel."""

    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_shard_h_lnc2(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        dtype,
        skip: int,
        scaling_mode: ExpertAffinityScaleMode,
        checkpoint_activation: bool,
    ):
        dma_skip = map_skip_mode(skip)

        def input_generator(test_config):
            return build_bwmm_shard_h_inputs(
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

        def output_tensors(kernel_input):
            T_out = tokens if dma_skip.skip_token else tokens + 1
            result = {"output": np.zeros((T_out, hidden), dtype=dtype)}
            if checkpoint_activation:
                N = get_n_blocks(tokens, top_k, expert, block_size)
                result["gate_up_activations_T"] = np.zeros((N, 2, intermediate, block_size), dtype=dtype)
            return result

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_baseline_shard_hidden,
            torch_ref=torch_ref_wrapper(blockwise_mm_baseline_shard_hidden_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
                # Skipping address_rotation_sb is a temporary workaround as we switch to latest nki, remove once KTK-151 resolved
                additional_cmd_args=[],
            ),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest_parametrize(PARAM_NAMES_COMBO, DTYPE_COMBO_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_shard_h_dtype_combos_lnc2(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        hidden: int,
        tokens: int,
        expert: int,
        block_size: int,
        top_k: int,
        intermediate: int,
        dtype,
        activation_dtype,
        accum_dtype,
        scaling_mode: ExpertAffinityScaleMode,
    ):
        """compute_dtype x activation_dtype x accum_dtype coverage (incl. bf16-io + fp32 accum)."""
        dma_skip = map_skip_mode(0)

        def input_generator(test_config):
            return build_bwmm_shard_h_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                scaling_mode=scaling_mode,
                checkpoint_activation=False,
                activation_dtype=activation_dtype,
                accum_dtype=accum_dtype,
            )

        def output_tensors(kernel_input):
            # accum_dtype output is cast back to the io dtype, so output is always `dtype`.
            T_out = tokens + 1
            return {"output": np.zeros((T_out, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_baseline_shard_hidden,
            torch_ref=torch_ref_wrapper(blockwise_mm_baseline_shard_hidden_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
            ),
            rtol=2e-2,
            atol=1e-5,
        )
