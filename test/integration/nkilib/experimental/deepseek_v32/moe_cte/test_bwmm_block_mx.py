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

"""Integration test for the DeepSeek-V3.2 blockwise-MX MoE context-encoding kernel."""

from typing import final

import nki.language as nl
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from nkilib_src.nkilib.experimental.deepseek_v32.moe_cte import bwmm_block_mx, bwmm_block_mx_torch_ref

from test.integration.nkilib.core.moe.moe_cte.test_moe_bwmm_mx_cte import (
    _block_output,
    _gen_block_inputs,
    _mx_torch_ref_wrapper,
)
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework

# DeepSeek-V3.2 production dims at EP8 x TP16: hidden_size, experts per rank, intermediate per
# rank, and n_activated_experts.
HIDDEN = 7168
E_LOCAL = 32
I_PER_RANK = 128
TOP_K = 8

# This kernel asserts a single logical NeuronCore.
_LOGICAL_NC_CONFIG = 1

# Prefill token count: one 4096-token context chunk, which is the context-encoding granularity,
# not the decode batch. Blockwise MoE cost scales with tokens x block_size, so this is the
# dimension prefill performance work sweeps.
_TOKENS = 4096
_BLOCK_SIZE = 256


@final
class TestBwmmBlockMx:
    """Blockwise-MX MoE CTE kernel at the DeepSeek-V3.2 production expert and hidden dims."""

    @pytest.mark.parametrize("case", ["DS32_EP8TP16_PROD_CONFIG"])
    def test_bwmm_block_mx(self, test_manager: Orchestrator, platform_target: Platforms, case: str):
        """Blockwise MXFP8 expert MLP at hidden 7168, 32 local experts, top-k 8, single core."""

        dtype = nl.bfloat16
        weight_dtype = nl.float8_e4m3fn_x4
        # map_skip_mode(1) resolves to SkipMode(False, False) in the shared suite.
        skip_mode = 1

        def input_gen(test_config):
            return _gen_block_inputs(
                vnc_degree=_LOGICAL_NC_CONFIG,
                hidden=HIDDEN,
                tokens=_TOKENS,
                intermediate=I_PER_RANK,
                expert=E_LOCAL,
                block_size=_BLOCK_SIZE,
                top_k=TOP_K,
                ep_degree=1,
                act_fn=ActFnType.SiLU,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                dtype=dtype,
                weight_dtype=weight_dtype,
                skip_mode=skip_mode,
                bias=False,
                is_dynamic=True,
                gate_clamp_upper=None,
                gate_clamp_lower=None,
                up_clamp_upper=None,
                up_clamp_lower=None,
                use_uint_weights=False,
                unpacked_weights=False,
                n_static_blocks=None,
                use_packed_scales=False,
                quantization_type=QuantizationType.MX,
            )

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=bwmm_block_mx,
            torch_ref=_mx_torch_ref_wrapper(bwmm_block_mx_torch_ref),
            kernel_input_generator=input_gen,
            # This kernel emits [tokens, hidden]; the shared two-core kernel emits
            # [2, tokens, hidden] pre-reduce, so the descriptor is built at vnc_degree 1.
            output_tensor_descriptor=lambda kernel_input: _block_output(
                kernel_input, _TOKENS, HIDDEN, dtype, skip_mode, TOP_K != 1, _LOGICAL_NC_CONFIG
            ),
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=_LOGICAL_NC_CONFIG, platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
        )
