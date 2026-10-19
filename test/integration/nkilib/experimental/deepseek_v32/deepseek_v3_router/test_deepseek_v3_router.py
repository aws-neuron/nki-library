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

"""Integration tests for the DeepSeek-V3.2 group-limited noaux_tc router kernel."""

from typing import Any

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.deepseek_v32.deepseek_v3_router import (
    RouterOutputs,
    deepseek_v3_router,
    deepseek_v3_router_torch_ref,
    get_deepseek_v3_router_shape,
)

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# DeepSeek-V3.2 routing constants (config.n_routed_experts and the group-limited widths).
HIDDEN = 7168
N_EXPERTS = 256
N_GROUP = 8
TOPK_GROUP = 4
TOP_K = 8
ROUTED_SCALING_FACTOR = 2.5

PARAM_NAMES = "n_tokens, outputs, dtype"
TEST_PARAMS = [
    pytest.param(1, RouterOutputs.DENSE, np.float32, id="ds32_router_t1_dense_f32", marks=pytest.mark.fast),
    pytest.param(8, RouterOutputs.TOPK, ml_dtypes.bfloat16, id="ds32_router_t8_topk_bf16", marks=pytest.mark.fast),
    pytest.param(
        16,
        RouterOutputs.DENSE_AND_TOPK,
        ml_dtypes.bfloat16,
        id="ds32_router_t16_dense_and_topk_bf16",
        marks=pytest.mark.fast,
    ),
    pytest.param(1, RouterOutputs.TOPK, ml_dtypes.bfloat16, id="ds32_router_t1_topk_bf16"),
    pytest.param(64, RouterOutputs.DENSE, ml_dtypes.bfloat16, id="ds32_router_t64_dense_bf16"),
    pytest.param(16, RouterOutputs.DENSE, ml_dtypes.bfloat16, id="DS32_ROUTER_PROD_CONFIG_B16"),
]


class TestDeepseekV3Router:
    """Test executor for DeepSeek-V3.2 router tests, covering all three output modes."""

    @pytest.mark.parametrize("lnc", [1])
    @pytest.mark.parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_deepseek_v3_router(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        n_tokens: int,
        outputs: RouterOutputs,
        dtype: type,
        lnc: int,
    ):
        def input_generator(test_config):
            return self._generate_inputs(n_tokens, outputs, dtype)

        def output_tensors(kernel_input: dict[str, Any]) -> dict[str, Any]:
            shapes = get_deepseek_v3_router_shape(
                kernel_input["normed"],
                kernel_input["router_weight"],
                kernel_input["e_score_correction_bias"],
                kernel_input["n_group"],
                kernel_input["topk_group"],
                kernel_input["top_k"],
            )
            dense = np.zeros((shapes.n_tokens, shapes.n_experts), dtype=np.float32)
            weight = np.zeros((shapes.n_tokens, shapes.top_k), dtype=np.float32)
            index = np.zeros((shapes.n_tokens, shapes.top_k), dtype=np.int32)
            if outputs == RouterOutputs.DENSE:
                return {"output_0": dense}
            if outputs == RouterOutputs.TOPK:
                return {"output_0": weight, "output_1": index}
            return {"output_0": dense, "output_1": weight, "output_2": index}

        framework = UnitTestFramework(
            test_manager=test_manager,
            torch_ref=torch_ref_wrapper(deepseek_v3_router_torch_ref),
            kernel_entry=deepseek_v3_router,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            atol=1e-4,
            rtol=1e-3,
        )

    @staticmethod
    def _generate_inputs(n_tokens: int, outputs: RouterOutputs, dtype: type) -> dict[str, Any]:
        """Wrapper around input generator for TestDeepseekV3Router."""

        np.random.seed(42)

        def random_tensor(
            name: str, *shape, tensor_dtype: type = dtype, mean: float = 0.0, std: float = 0.2
        ) -> npt.NDArray[Any]:
            """Generates a random tensor for the provided name, shape, dtype and optionally mean and std."""
            generator = gaussian_tensor_generator(mean, std)
            return generator(shape, tensor_dtype, name)

        return {
            "normed": random_tensor("normed", n_tokens, HIDDEN),
            # fp32 regardless of the input dtype: the logits feed a tie-sensitive discrete
            # selection, so the kernel keeps this operand fp32 to match the reference.
            "router_weight": random_tensor("router_weight", N_EXPERTS, HIDDEN, tensor_dtype=np.float32),
            "e_score_correction_bias": random_tensor("e_score_correction_bias", 1, N_EXPERTS, tensor_dtype=np.float32),
            "n_group": N_GROUP,
            "topk_group": TOPK_GROUP,
            "top_k": TOP_K,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "outputs": outputs,
            "output_in_sbuf": False,
        }
