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

"""Unit test for rmsnorm_router_topk kernel."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import RouterActFnType
from nkilib_src.nkilib.experimental.moe_block.rmsnorm_router_topk_a2av import rmsnorm_router_topk_a2av
from nkilib_src.nkilib.experimental.moe_block.rmsnorm_router_topk_a2av_torch import rmsnorm_router_topk_a2av_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _torch_ref(hidden_states, gamma, router_weights, router_bias, eps, top_k, router_act_fn):
    """Torch reference for rmsnorm_router_topk."""
    import torch

    B, S, H = hidden_states.shape
    T = B * S
    E = router_weights.shape[1]

    # RMSNorm
    x = hidden_states.reshape(T, H).float()
    rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + eps)
    norm_output = (x / rms).to(hidden_states.dtype)

    # Router matmul + activation + top-K
    logits = norm_output.float() @ router_weights.float()  # [T, E]
    if router_bias is not None:
        logits = logits + router_bias.float()

    if router_act_fn == RouterActFnType.SIGMOID:
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=-1)

    # Top-K
    topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
    expert_index = topk_idx.to(torch.int32)

    # Masked affinities
    expert_affinities = torch.zeros(T, E, dtype=torch.bfloat16)
    for t in range(T):
        for k in range(top_k):
            expert_affinities[t, expert_index[t, k]] = probs[t, expert_index[t, k]].to(torch.bfloat16)

    return {
        "norm_output": norm_output.numpy(),
        "expert_index": expert_index.numpy(),
        "expert_affinities": expert_affinities.numpy(),
    }


# fmt: off
PARAMS = "batch, seqlen, hidden, num_experts, top_k, router_act_fn"
TEST_CASES = [
    (1, 1,  3072, 128, 4, RouterActFnType.SOFTMAX),
    (1, 8,  3072, 128, 4, RouterActFnType.SOFTMAX),
    (1, 8,  8192, 2, 2, RouterActFnType.SOFTMAX),
    (1, 1,  8192, 2, 2, RouterActFnType.SOFTMAX),
    (1, 8,  8192, 2, 2, RouterActFnType.SOFTMAX),
]
# fmt: on


@pytest_test_metadata(name="RMSNorm Router TopK A2AV")
@pytest_marks(["rmsnorm", "router_topk", "moe"])
class TestRmsNormRouterTopkA2av:
    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1])
    @pytest.mark.parametrize(PARAMS, TEST_CASES)
    def test_rmsnorm_router_topk_a2av(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        seqlen,
        hidden,
        num_experts,
        top_k,
        router_act_fn,
    ):
        np.random.seed(42)

        def input_generator(_):
            return {
                "hidden_states": np.random.randn(batch, seqlen, hidden).astype(np.float16) * 0.1,
                "gamma": (np.ones((1, hidden)) + np.random.randn(1, hidden) * 0.01).astype(np.float16),
                "router_weights": np.random.randn(hidden, num_experts).astype(np.float16) * 0.01,
                "router_bias": None,
                "eps": 1e-6,
                "top_k": top_k,
                "router_act_fn": router_act_fn,
            }

        def output_descriptor(ki):
            T = ki["hidden_states"].shape[0] * ki["hidden_states"].shape[1]
            H = ki["hidden_states"].shape[2]
            E = ki["router_weights"].shape[1]
            K = ki["top_k"]
            return {
                "norm_output": np.zeros((T, H), dtype=ki["hidden_states"].dtype),
                "expert_index": np.zeros((T, K), dtype=np.int32),
                "expert_affinities": np.zeros((T, E), dtype=nl.bfloat16),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_router_topk_a2av,
            torch_ref=torch_ref_wrapper(rmsnorm_router_topk_a2av_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            rtol=5e-2,
            atol=1e-3,
        )
