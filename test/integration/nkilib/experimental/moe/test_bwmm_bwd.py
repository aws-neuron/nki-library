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

"""Integration tests for the blockwise MM backward kernel."""

from typing import final

import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.logging import get_logger
from nkilib_src.nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd
from nkilib_src.nkilib.experimental.moe.bwd.blockwise_mm_backward_torch import blockwise_mm_bwd_torch_ref
from nkilib_src.nkilib.experimental.moe.bwd.bwmm_bwd_dropless import (
    MAX_AVAILABLE_SBUF_SIZE,
    _compute_hidden_states_grad,
    _load_block_expert,
    _load_token_indices,
)
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    DownWeightGradBlocking,
    GateUpOutputGradBlocking,
    GateUpWeightGradBlocking,
    HiddenGradBlocking,
    MOEBwdDroplessBlockingParams,
    ShardOption,
    SkipMode,
)

from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    build_bwmm_bwd_inputs,
    map_skip_mode,
)
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

bfloat16 = nl.bfloat16
float32 = nl.float32

# fmt: off
AFFINITY_H = AffinityOption.AFFINITY_ON_H
AFFINITY_I = AffinityOption.AFFINITY_ON_I
SHARD_FREE = ShardOption.SHARD_ON_FREE
SHARD_H = ShardOption.SHARD_ON_HIDDEN
DEFAULT_BP = MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(),
                                          down_weight_grad=DownWeightGradBlocking(),
                                          hidden_grad=HiddenGradBlocking(),
                                          gate_up_weight_grad=GateUpWeightGradBlocking())

PARAM_NAMES = \
    "hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, clamp_limits, bias_flag, activation_type, affinity_option, blocking_params, shard_option, preallocate_grad_out"
TEST_PARAMS = [
# H,    T,    E,   B,   TOPK, I_TP, dtype,    skip, clamp_limits,                        bias,  activation_type,  affinity, blocking_params,                                                                                                                                                                                          shard_option, preallocate_grad_out
[5120,  8192, 16,  512, 1,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[5120,  8192, 16,  256, 4,    1024, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[5120,  8192, 128, 256, 1,    128,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

[6144,  4096, 16,  512, 4,    1024, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[6144,  4096, 16,  512, 4,    128,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[6144,  4096, 1,   512, 1,    128,  bfloat16, 0,    ClampLimits(7, -7, 7, -7),            False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

[2880,  4096, 2,   512, 2,    2880, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[2880,  4096, 2,   512, 2,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[2880,  4096, 2,   256, 2,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 2,   512, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 4,   512, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 4,   128, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

[4096,  4096, 4,   128, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 4,   256, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

[4096,  4096, 4,   128, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 4,   256, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

[2880,  4096, 2,   128, 2,    720, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

# SquaredReLU activation backward
[2048,  512,  2,   256, 2,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SquaredReLU, AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[2048,  512,  2,   256, 1,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SquaredReLU, AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

[5120,  4096, 4,   128, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
[5120,  4096, 4,   256, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],

# Qwen3-235B single-expert-dense (H=4096, E=1, TOPK=1): TP1 I_TP=1536, TP2 I_TP=768
[4096,  4096, 1,   2048, 1,   1536, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 1,   4096, 1,   1536, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 1,   2048, 1,   768,  bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 1,   4096, 1,   768,  bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],

# Affinity I test cases
[4096,  4096, 4,   128, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 4,   256, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],

[4096,  4096, 4,   128, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[4096,  4096, 4,   256, 2,    1536,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],

[5120,  4096, 4,   128, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[5120,  4096, 4,   256, 1,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),   False,  ActFnType.SiLU,  AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],

[2880,  4096, 2,  128, 2,    2880, bfloat16, 0,    ClampLimits(7, -7, 7, -7),  True, ActFnType.Swish,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[2048,  4096, 2, 128, 2,    768,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[2048,  4096, 2, 128, 2,    192,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[5120,  4096, 2,  128, 2,    2048, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[2048,  4096, 2,  128, 2,    1408, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[2048,  4096, 2,  128, 2,    352,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],

[2048,  512, 2, 512, 2,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=16, block_b=4, block_i=2),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=4, block_i=2),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=16, block_b=4, block_i=2),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=4, block_i=2)), SHARD_FREE, False],

# Shard H Test
[2048,  512,  2,  512, 2,    352,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=4, block_i=3)), SHARD_H, False],

[2048,  16384, 64, 512, 6,    352,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=16, block_b=4, block_i=3),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=4, block_i=3)), SHARD_H, False],

[3072,  8192, 32,   1024, 6,    720, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_I, MOEBwdDroplessBlockingParams(gate_up_output_grad=GateUpOutputGradBlocking(block_h=8, block_b=4, block_i=2),
                                                                                                                                                           down_weight_grad=DownWeightGradBlocking(block_h=16, block_b=8, block_i=6),
                                                                                                                                                           hidden_grad=HiddenGradBlocking(block_h=2, block_b=8, block_i=6),
                                                                                                                                                   gate_up_weight_grad=GateUpWeightGradBlocking(block_h=16, block_b=8, block_i=6)), SHARD_H, False],

# Pre-allocated grad output test cases
[4096,  4096, 2, 512, 2, 384, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU, AFFINITY_H, DEFAULT_BP, SHARD_FREE, True],
[2880,  4096, 2, 512, 2, 2880, bfloat16, 1, ClampLimits(7, -7, 7, -7), True, ActFnType.Swish, AFFINITY_H, DEFAULT_BP, SHARD_FREE, True],
# fp32 io + fp32 grad accumulation (exercises get_buffer_degree(fp32) -> reduced weight-grad degrees)
[2048,  512,  2,   512, 2,    256,  float32, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[2048,  4096, 2,   128, 2,    768,  float32, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU,   AFFINITY_I, DEFAULT_BP, SHARD_FREE, False],
[2880,  4096, 2,   128, 2,    720,  float32, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish,  AFFINITY_H, DEFAULT_BP, SHARD_FREE, False],
]
# fmt: on

# (hidden, tokens, expert, block_size, top_k, intermediate) keys for full-only tests (excluded from fast suite)
_FULL_ONLY_KEYS = {
    (5120, 8192, 16, 256, 4, 1024),
    (5120, 8192, 128, 256, 1, 128),
    (6144, 4096, 16, 512, 4, 1024),
    (2880, 4096, 2, 128, 2, 2880),
    (4096, 4096, 4, 128, 2, 1536),
    (4096, 4096, 4, 256, 2, 1536),
    (4096, 4096, 4, 128, 2, 384),
    (2880, 4096, 2, 256, 2, 2880),
    (2880, 4096, 2, 512, 2, 2880),
    (2880, 4096, 2, 128, 2, 720),
    (6144, 4096, 16, 512, 4, 128),
    (5120, 4096, 4, 256, 1, 2048),
    (2048, 4096, 2, 128, 2, 1408),
    (5120, 4096, 2, 128, 2, 2048),
    (5120, 4096, 4, 128, 1, 2048),
    (5120, 8192, 16, 512, 1, 256),
    (4096, 4096, 4, 256, 2, 384),
    (2048, 4096, 2, 128, 2, 768),
    (2048, 16384, 64, 512, 6, 256),
    (6144, 4096, 1, 512, 1, 128),
    (4096, 4096, 2, 512, 2, 384),
    (4096, 4096, 4, 512, 2, 384),
    (2048, 4096, 2, 128, 2, 192),
    (2048, 4096, 2, 128, 2, 352),
    (2048, 16384, 64, 512, 6, 352),
    (2048, 512, 2, 512, 2, 352),
    (3072, 8192, 32, 1024, 6, 720),
}

ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[:6]) not in _FULL_ONLY_KEYS else c for c in TEST_PARAMS
]

# fmt: off
# Mixed precision: bf16 io/weights + fp32 grad-out buffers (the nki_moe.py production case).
# preallocate_grad_out=True so the fp32 grad buffers are caller-provided via must_alias, which
# drives grad_accum_dtype=fp32 in the kernel (matmul operands stay bf16, accumulators go fp32).
# Covers AFFINITY_ON_I, AFFINITY_ON_H (incl. fp32 bias-grad), and SHARD_ON_HIDDEN.
MIXED_GRAD_PARAMS = [
[2048, 4096, 2, 128, 2, 768, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU, AFFINITY_I, DEFAULT_BP, SHARD_FREE, True],
[2880, 4096, 2, 128, 2, 720, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU, AFFINITY_H, DEFAULT_BP, SHARD_FREE, True],
[2048, 4096, 2, 128, 2, 768, bfloat16, 0, ClampLimits(None, None, None, None), False, ActFnType.SiLU, AFFINITY_I, DEFAULT_BP, SHARD_H,    True],
[2880, 4096, 2, 128, 2, 720, bfloat16, 0, ClampLimits(7, -7, 7, -7),           True,  ActFnType.Swish, AFFINITY_H, DEFAULT_BP, SHARD_FREE, True],
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
    "skip": "sk",
    "clamp_limits": "cl",
    "bias_flag": "bi",
    "activation_type": "act",
    "affinity_option": "aff",
    "blocking_params": "bp",
    "shard_option": "sh",
    "preallocate_grad_out": "pgo",
}


@pytest_test_metadata(name="MoE Blockwise MatMul BWD")
@pytest_marks(["moe", "blockwise_mm_bwd", "lnc2"])
@final
class TestMoeBlockwiseMatMulBwdShardHDroplessLnc2:
    """Tests for LNC2 blockwise matmul backward pass."""

    @pytest_parametrize(PARAM_NAMES, ALL_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_bwd_dropless_lnc2(
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
        clamp_limits: ClampLimits,
        bias_flag: bool,
        activation_type: ActFnType,
        affinity_option: AffinityOption,
        blocking_params,
        shard_option: ShardOption,
        preallocate_grad_out: bool,
    ):
        dma_skip = map_skip_mode(skip)

        def input_generator(test_config):
            inputs, _, _ = build_bwmm_bwd_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias_flag=bias_flag,
                clamp_limits=clamp_limits,
                activation_type=activation_type,
                affinity_option=affinity_option,
                blocking_params=blocking_params,
                shard_option=shard_option,
            )
            if preallocate_grad_out:
                T_out = tokens if dma_skip.skip_token else tokens + 1
                inputs["hidden_states_grad_out.must_alias_input"] = np.zeros((T_out, hidden), dtype=dtype)
                inputs["expert_affinities_masked_grad_out.must_alias_input"] = np.zeros(
                    (T_out * expert, 1), dtype=dtype
                )
                inputs["gate_up_proj_weight_grad_out.must_alias_input"] = np.zeros(
                    (expert, hidden, 2, intermediate), dtype=dtype
                )
                inputs["down_proj_weight_grad_out.must_alias_input"] = np.zeros(
                    (expert, intermediate, hidden), dtype=dtype
                )
            return inputs

        def output_tensors(kernel_input):
            T_out = tokens if dma_skip.skip_token else tokens + 1
            result = {
                "hidden_states_grad": np.zeros((T_out, hidden), dtype=dtype),
                "expert_affinities_masked_grad": np.zeros((T_out * expert, 1), dtype=dtype),
                "gate_up_proj_weight_grad": np.zeros((expert, hidden, 2, intermediate), dtype=dtype),
                "down_proj_weight_grad": np.zeros((expert, intermediate, hidden), dtype=dtype),
            }
            if bias_flag:
                result["gate_and_up_proj_bias_grad"] = np.zeros((expert, 2, intermediate), dtype=dtype)
                result["down_proj_bias_grad"] = np.zeros((expert, hidden), dtype=dtype)
            return result

        lnc_count = 2
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_bwd,
            torch_ref=torch_ref_wrapper(blockwise_mm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc_count,
                enable_birsim=False,
                platform_target=platform_target,
                dump_after_lowering=False,
            ),
            inference_args=InferenceArgs(),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest_parametrize(PARAM_NAMES, MIXED_GRAD_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_bwd_dropless_mixed_grad_lnc2(
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
        clamp_limits: ClampLimits,
        bias_flag: bool,
        activation_type: ActFnType,
        affinity_option: AffinityOption,
        blocking_params,
        shard_option: ShardOption,
        preallocate_grad_out: bool,
    ):
        """Mixed precision: bf16 io/weights, fp32 grad-out buffers (matches nki_moe.py production)."""
        grad_dtype = nl.float32
        dma_skip = map_skip_mode(skip)

        def input_generator(test_config):
            inputs, _, _ = build_bwmm_bwd_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias_flag=bias_flag,
                clamp_limits=clamp_limits,
                activation_type=activation_type,
                affinity_option=affinity_option,
                blocking_params=blocking_params,
                shard_option=shard_option,
            )
            T_out = tokens if dma_skip.skip_token else tokens + 1
            inputs["hidden_states_grad_out.must_alias_input"] = np.zeros((T_out, hidden), dtype=grad_dtype)
            inputs["expert_affinities_masked_grad_out.must_alias_input"] = np.zeros(
                (T_out * expert, 1), dtype=grad_dtype
            )
            inputs["gate_up_proj_weight_grad_out.must_alias_input"] = np.zeros(
                (expert, hidden, 2, intermediate), dtype=grad_dtype
            )
            inputs["down_proj_weight_grad_out.must_alias_input"] = np.zeros(
                (expert, intermediate, hidden), dtype=grad_dtype
            )
            return inputs

        def output_tensors(kernel_input):
            T_out = tokens if dma_skip.skip_token else tokens + 1
            result = {
                "hidden_states_grad": np.zeros((T_out, hidden), dtype=grad_dtype),
                "expert_affinities_masked_grad": np.zeros((T_out * expert, 1), dtype=grad_dtype),
                "gate_up_proj_weight_grad": np.zeros((expert, hidden, 2, intermediate), dtype=grad_dtype),
                "down_proj_weight_grad": np.zeros((expert, intermediate, hidden), dtype=grad_dtype),
            }
            if bias_flag:
                # Bias grads follow their paired weight-grad dtype (fp32 here), so they accumulate in fp32.
                result["gate_and_up_proj_bias_grad"] = np.zeros((expert, 2, intermediate), dtype=grad_dtype)
                result["down_proj_bias_grad"] = np.zeros((expert, hidden), dtype=grad_dtype)
            return result

        lnc_count = 2
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_bwd,
            torch_ref=torch_ref_wrapper(blockwise_mm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc_count,
                enable_birsim=False,
                platform_target=platform_target,
                dump_after_lowering=False,
            ),
            inference_args=InferenceArgs(),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest_parametrize(PARAM_NAMES, MIXED_GRAD_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_bwd_dropless_accum_dtype_lnc2(
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
        clamp_limits: ClampLimits,
        bias_flag: bool,
        activation_type: ActFnType,
        affinity_option: AffinityOption,
        blocking_params,
        shard_option: ShardOption,
        preallocate_grad_out: bool,
    ):
        """Row-2: bf16 io AND bf16 grad-out buffers + accumulation_dtype=float32. The kernel accumulates
        into an internal fp32 scratch and downcasts to the bf16 grad buffers on return (the param-driven
        fp32-accumulation path, with bf16 grads kept out)."""
        grad_dtype = dtype  # bf16 grad-out buffers (customer keeps bf16)
        dma_skip = map_skip_mode(skip)

        def input_generator(test_config):
            inputs, _, _ = build_bwmm_bwd_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias_flag=bias_flag,
                clamp_limits=clamp_limits,
                activation_type=activation_type,
                affinity_option=affinity_option,
                blocking_params=blocking_params,
                shard_option=shard_option,
            )
            # Opt-in fp32 accumulation with bf16 grad buffers -> internal fp32 scratch + downcast.
            inputs["accumulation_dtype"] = nl.float32
            T_out = tokens if dma_skip.skip_token else tokens + 1
            inputs["hidden_states_grad_out.must_alias_input"] = np.zeros((T_out, hidden), dtype=grad_dtype)
            inputs["expert_affinities_masked_grad_out.must_alias_input"] = np.zeros(
                (T_out * expert, 1), dtype=grad_dtype
            )
            inputs["gate_up_proj_weight_grad_out.must_alias_input"] = np.zeros(
                (expert, hidden, 2, intermediate), dtype=grad_dtype
            )
            inputs["down_proj_weight_grad_out.must_alias_input"] = np.zeros(
                (expert, intermediate, hidden), dtype=grad_dtype
            )
            return inputs

        def output_tensors(kernel_input):
            T_out = tokens if dma_skip.skip_token else tokens + 1
            result = {
                "hidden_states_grad": np.zeros((T_out, hidden), dtype=grad_dtype),
                "expert_affinities_masked_grad": np.zeros((T_out * expert, 1), dtype=grad_dtype),
                "gate_up_proj_weight_grad": np.zeros((expert, hidden, 2, intermediate), dtype=grad_dtype),
                "down_proj_weight_grad": np.zeros((expert, intermediate, hidden), dtype=grad_dtype),
            }
            if bias_flag:
                result["gate_and_up_proj_bias_grad"] = np.zeros((expert, 2, intermediate), dtype=grad_dtype)
                result["down_proj_bias_grad"] = np.zeros((expert, hidden), dtype=grad_dtype)
            return result

        lnc_count = 2
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_bwd,
            torch_ref=torch_ref_wrapper(blockwise_mm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc_count,
                enable_birsim=False,
                platform_target=platform_target,
                dump_after_lowering=False,
            ),
            inference_args=InferenceArgs(),
            rtol=2e-2,
            atol=1e-5,
        )


@nki.jit
def _skip_gate_hidden_grad_reproducer(
    gate_up_output_grad,
    gate_up_weight,
    token_position_to_id,
    block_to_expert,
    skip_gate_proj,
):
    block_size = gate_up_output_grad.shape[0]
    hidden = gate_up_weight.shape[1]
    tile_size = nl.tile_size.gemm_stationary_fmax
    num_tiles = (block_size + tile_size - 1) // tile_size

    hidden_grad = nl.ndarray(
        (block_size, hidden),
        dtype=gate_up_output_grad.dtype,
        buffer=nl.shared_hbm,
    )
    sbm = SbufManager(0, MAX_AVAILABLE_SBUF_SIZE, logger=get_logger("skip_gate_repro"))
    sbm.open_scope(name="skip_gate_repro")
    expert_idx = _load_block_expert(block_to_expert, 0, sbm)
    token_indices = _load_token_indices(
        token_position_to_id,
        block_idx=0,
        B=block_size,
        NUM_TILES=num_tiles,
        sbm=sbm,
    )

    _compute_hidden_states_grad(
        gate_up_proj_output_grad_hbm=gate_up_output_grad,
        gate_up_proj_weight=gate_up_weight,
        hidden_states_grad=hidden_grad,
        block_token_pos_to_id_full=token_indices,
        shard_id=0,
        num_shards=1,
        expert_idx=expert_idx,
        skip_dma=SkipMode(False, False),
        compute_dtype=gate_up_output_grad.dtype,
        is_tensor_update_accumulating=False,
        block_idx=0,
        sbm=sbm,
        skip_gate_proj=skip_gate_proj,
    )
    sbm.close_scope()
    return hidden_grad


def _build_skip_gate_read_inputs(block_size: int, hidden: int, intermediate: int):
    rng = np.random.default_rng(42)
    gate_up_output_grad = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(block_size, 2, intermediate),
    ).astype(nl.bfloat16)
    gate_up_output_grad[:, 0, :] = np.nan
    gate_up_weight = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(1, hidden, 2, intermediate),
    ).astype(nl.bfloat16)
    token_position_to_id = np.arange(block_size, dtype=np.int32)
    block_to_expert = np.zeros((1, 1), dtype=np.int32)
    return gate_up_output_grad, gate_up_weight, token_position_to_id, block_to_expert


def _simulate_skip_gate_hidden_grad(inputs, skip_gate_proj: bool):
    result = nki.simulate(_skip_gate_hidden_grad_reproducer)[1](
        gate_up_output_grad=inputs[0],
        gate_up_weight=inputs[1],
        token_position_to_id=inputs[2],
        block_to_expert=inputs[3],
        skip_gate_proj=skip_gate_proj,
    )
    return np.asarray(result)


def _skip_gate_hidden_grad_reference(gate_up_output_grad, gate_up_weight):
    up_grad = gate_up_output_grad[:, 1, :].astype(np.float32)
    up_weight = gate_up_weight[0, :, 1, :].astype(np.float32)
    return (up_grad @ up_weight.T).astype(nl.bfloat16)


def _skip_gate_hidden_grad_torch_ref(
    gate_up_output_grad,
    gate_up_weight,
    token_position_to_id,
    block_to_expert,
    skip_gate_proj,
):
    del token_position_to_id, block_to_expert
    assert skip_gate_proj
    up_grad = gate_up_output_grad[:, 1, :].to(torch.float32)
    up_weight = gate_up_weight[0, :, 1, :].to(torch.float32)
    return (up_grad @ up_weight.T).to(gate_up_output_grad.dtype)


SKIP_GATE_BWD_PARAMS = [
    pytest.param(2048, 512, 2, 256, 2, 256, bfloat16, ActFnType.SquaredReLU, id="SquaredReLU-skip_gate"),
    pytest.param(2048, 512, 2, 256, 1, 256, bfloat16, ActFnType.SiLU, id="SiLU-skip_gate"),
    pytest.param(2048, 512, 2, 256, 2, 256, bfloat16, ActFnType.Swish, id="Swish-skip_gate"),
    pytest.param(2048, 2048, 64, 256, 8, 1024, bfloat16, ActFnType.SquaredReLU, id="SquaredReLU-skip_gate-e64-k8"),
]


@pytest_marks(["moe", "bwd", "skip_gate_proj"])
class TestMoeBwdSkipGateProj:
    """Test backward kernel with skip_gate_proj=True."""

    def test_skip_gate_hidden_grad_does_not_read_gate_slot(self):
        inputs = _build_skip_gate_read_inputs(block_size=256, hidden=2048, intermediate=1024)
        reference = _skip_gate_hidden_grad_reference(inputs[0], inputs[1])

        old_output = _simulate_skip_gate_hidden_grad(inputs, skip_gate_proj=False)
        fixed_output = _simulate_skip_gate_hidden_grad(inputs, skip_gate_proj=True)

        assert not np.isfinite(old_output).all(), "old read behavior did not propagate the poisoned gate slot"
        assert np.isfinite(fixed_output).all(), "skip-gate path propagated a nonfinite value"
        np.testing.assert_allclose(
            fixed_output.astype(np.float32),
            reference.astype(np.float32),
            rtol=2e-2,
            atol=3e-4,
        )

    @pytest.mark.fast
    def test_skip_gate_hidden_grad_matches_up_only_reference(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        def input_generator(_):
            inputs = _build_skip_gate_read_inputs(block_size=256, hidden=2048, intermediate=1024)
            return {
                "gate_up_output_grad": inputs[0],
                "gate_up_weight": inputs[1],
                "token_position_to_id": inputs[2],
                "block_to_expert": inputs[3],
                "skip_gate_proj": True,
            }

        def output_tensors(_):
            return {"out": np.zeros((256, 2048), dtype=nl.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_skip_gate_hidden_grad_reproducer,
            torch_ref=torch_ref_wrapper(_skip_gate_hidden_grad_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=1,
                enable_birsim=False,
                platform_target=platform_target,
                dump_after_lowering=False,
            ),
            rtol=2e-2,
            atol=3e-4,
        )

    @pytest.mark.parametrize(
        "hidden, tokens, expert, block_size, top_k, intermediate, dtype, activation_type", SKIP_GATE_BWD_PARAMS
    )
    def test_moe_bwd_dropless_skip_gate_proj(
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
        activation_type: ActFnType,
    ):
        dma_skip = SkipMode(False, False)

        def input_generator(test_config):
            inputs, _, _ = build_bwmm_bwd_inputs(
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                expert=expert,
                block_size=block_size,
                top_k=top_k,
                dtype=dtype,
                dma_skip=dma_skip,
                bias_flag=False,
                clamp_limits=ClampLimits(None, None, None, None),
                activation_type=activation_type,
                affinity_option=AffinityOption.AFFINITY_ON_H,
                blocking_params=DEFAULT_BP,
                shard_option=ShardOption.SHARD_ON_FREE,
                skip_gate_proj=True,
            )
            return inputs

        def output_tensors(kernel_input):
            T_out = tokens + 1
            return {
                "hidden_states_grad": np.zeros((T_out, hidden), dtype=dtype),
                "expert_affinities_masked_grad": np.zeros((T_out * expert, 1), dtype=dtype),
                "gate_up_proj_weight_grad": np.zeros((expert, hidden, 2, intermediate), dtype=dtype),
                "down_proj_weight_grad": np.zeros((expert, intermediate, hidden), dtype=dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=blockwise_mm_bwd,
            torch_ref=torch_ref_wrapper(blockwise_mm_bwd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=2, enable_birsim=False, platform_target=platform_target, dump_after_lowering=False
            ),
            inference_args=InferenceArgs(),
            rtol=2e-2,
            atol=1e-5,
        )
