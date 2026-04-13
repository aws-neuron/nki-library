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

from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    blockwise_mm_bwd_torch_ref,
    build_bwmm_bwd_inputs,
    map_skip_mode,
)
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import ActFnType, ClampLimits

bfloat16 = nl.bfloat16

# fmt: off
PARAM_NAMES = \
    "hidden, tokens, expert, block_size, top_k, intermediate, dtype, skip, clamp_limits, bias_flag, activation_type"
TEST_PARAMS = [
# H,    T,    E,   B,   TOPK, I_TP, dtype,    skip, clamp_limits,                        bias,  activation_type
[5120,  8192, 16,  512, 1,    256,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],
[5120,  8192, 16,  256, 4,    1024, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],
[5120,  8192, 128, 256, 1,    128,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],

[6144,  4096, 16,  512, 4,    1024, bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],
[6144,  4096, 16,  512, 4,    128,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],
[6144,  4096, 1,   512, 1,    128,  bfloat16, 0,    ClampLimits(7, -7, 7, -7),            False, ActFnType.SiLU],

[2880,  4096, 2,   512, 2,    2880, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish],
[2880,  4096, 2,   512, 2,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish],
[2880,  4096, 2,   256, 2,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish],
[2880,  4096, 2,   1024,2,    2880, bfloat16, 0,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish],
[2880,  4096, 32,  1024,4,    2880, bfloat16, 1,    ClampLimits(7, -7, 7, -7),            True,  ActFnType.Swish],

[4096,  4096, 2,   512, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],
[4096,  4096, 4,   512, 2,    384,  bfloat16, 0,    ClampLimits(None, None, None, None),  False, ActFnType.SiLU],
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
}


@pytest_test_metadata(
    name="MoE Blockwise MatMul BWD ShardH AffinityH Dropless LNC2",
    pytest_marks=["moe", "blockwise_mm_bwd", "lnc2"],
)
@final
class TestMoeBlockwiseMatMulBwdShardHAffinityHDroplessLnc2:
    """Tests for LNC2 blockwise matmul backward pass with shard on hidden and affinity on hidden."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS, abbrevs=_ABBREVS)
    def test_moe_blockwise_mm_bwd_shardH_affinityH_dropless_lnc2(
        self,
        test_manager: Orchestrator,
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
            compiler_args=CompilerArgs(logical_nc_config=lnc_count, enable_birsim=False),
            inference_args=InferenceArgs(num_runs=2, profile_all_runs=True),
            rtol=2e-2,
            atol=1e-5,
        )
