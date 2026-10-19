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

"""Integration tests for Kimi-K3's flat noaux_tc router."""

import ml_dtypes
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.kimi_k3.router_topk import (
    kimi_k3_router_topk,
    kimi_k3_router_topk_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_inputs(tokens, hidden, expert, top_k, routed_scaling_factor=1.0):
    np.random.seed(42)
    rng = np.random.default_rng(42)
    return {
        "hidden_states": (rng.standard_normal((tokens, hidden)) * 0.2).astype(ml_dtypes.bfloat16),
        "gate_weight": (rng.standard_normal((expert, hidden)) / np.sqrt(hidden)).astype(ml_dtypes.bfloat16),
        "e_score_correction_bias": (rng.standard_normal((expert,)) * 0.05).astype(np.float32),
        "top_k": top_k,
        "routed_scaling_factor": routed_scaling_factor,
    }


def _output_tensors(kernel_input):
    tokens = kernel_input["hidden_states"].shape[0]
    top_k = kernel_input["top_k"]
    return {
        "topk_weight": np.zeros((tokens, top_k), dtype=np.float32),
        "topk_idx": np.zeros((tokens, top_k), dtype=np.int32),
    }


# fmt: off
_ROUTER_CASES = [
    pytest.param(4,   128,  24,  3,  marks=pytest.mark.fast, id="component_T4_H128_E24_k3"),
    pytest.param(4,   128, 257, 16,  marks=pytest.mark.fast, id="tail_T4_H128_E257_k16"),
    pytest.param(4,  7168, 896, 16, marks=pytest.mark.fast, id="decode_T4_H7168_E896_k16"),
    pytest.param(128, 7168, 896, 16, id="prefill_T128_H7168_E896_k16"),
]
# fmt: on


@pytest.mark.parametrize("tokens,hidden,expert,top_k", _ROUTER_CASES)
def test_kimi_k3_router_topk(
    test_manager: Orchestrator,
    platform_target: Platforms,
    tokens,
    hidden,
    expert,
    top_k,
):
    """Match fp32 noaux_tc selection and pre-bias normalized weights."""
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kimi_k3_router_topk,
        torch_ref=torch_ref_wrapper(kimi_k3_router_topk_torch_ref),
        kernel_input_generator=lambda _: _generate_inputs(tokens, hidden, expert, top_k),
        output_tensor_descriptor=_output_tensors,
        check_unused_params=True,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.fast
def test_kimi_k3_router_topk_tie_break(
    test_manager: Orchestrator,
    platform_target: Platforms,
):
    """Exact selection-score ties choose ascending expert ids."""
    tokens, hidden, expert, top_k = 2, 128, 24, 16

    def input_generator(_):
        return {
            "hidden_states": np.zeros((tokens, hidden), dtype=ml_dtypes.bfloat16),
            "gate_weight": np.zeros((expert, hidden), dtype=ml_dtypes.bfloat16),
            "e_score_correction_bias": np.zeros((expert,), dtype=np.float32),
            "top_k": top_k,
            "routed_scaling_factor": 1.0,
        }

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kimi_k3_router_topk,
        torch_ref=torch_ref_wrapper(kimi_k3_router_topk_torch_ref),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=_output_tensors,
        check_unused_params=True,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.fast
def test_kimi_k3_router_topk_selection_bias_is_not_weight_bias(
    test_manager: Orchestrator,
    platform_target: Platforms,
):
    """Correction bias changes selection but not the gathered score values."""
    tokens, hidden, expert, top_k = 2, 128, 24, 2

    def input_generator(_):
        selection_bias = np.zeros((expert,), dtype=np.float32)
        selection_bias[20] = 2.0
        selection_bias[7] = 1.0
        return {
            "hidden_states": np.zeros((tokens, hidden), dtype=ml_dtypes.bfloat16),
            "gate_weight": np.zeros((expert, hidden), dtype=ml_dtypes.bfloat16),
            "e_score_correction_bias": selection_bias,
            "top_k": top_k,
            "routed_scaling_factor": 1.0,
        }

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kimi_k3_router_topk,
        torch_ref=torch_ref_wrapper(kimi_k3_router_topk_torch_ref),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=_output_tensors,
        check_unused_params=True,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
        atol=0.0,
        rtol=0.0,
    )


def test_kimi_k3_router_topk_reference_mutation_response():
    """Every mathematical input changes the compact reference result."""
    torch.manual_seed(42)
    tokens, hidden, expert, top_k = 4, 128, 24, 3
    hidden_states = torch.randn(tokens, hidden, dtype=torch.float32).to(torch.bfloat16)
    gate_weight = (torch.randn(expert, hidden, dtype=torch.float32) / np.sqrt(hidden)).to(torch.bfloat16)
    bias = torch.linspace(-0.2, 0.2, expert, dtype=torch.float32)

    baseline = kimi_k3_router_topk_torch_ref(hidden_states, gate_weight, bias, top_k)

    hidden_mut = hidden_states.clone()
    hidden_mut[0, 0] += 1
    from_hidden = kimi_k3_router_topk_torch_ref(hidden_mut, gate_weight, bias, top_k)
    assert not torch.equal(from_hidden["topk_weight"], baseline["topk_weight"])

    weight_mut = gate_weight.clone()
    selected_expert = int(baseline["topk_idx"][0, 0])
    weight_mut[selected_expert, :] += 1
    from_weight = kimi_k3_router_topk_torch_ref(hidden_states, weight_mut, bias, top_k)
    assert not torch.equal(from_weight["topk_weight"], baseline["topk_weight"])

    bias_mut = bias.clone()
    bias_mut[0] += 10
    from_bias = kimi_k3_router_topk_torch_ref(hidden_states, gate_weight, bias_mut, top_k)
    assert not torch.equal(from_bias["topk_idx"], baseline["topk_idx"])

    from_scale = kimi_k3_router_topk_torch_ref(
        hidden_states,
        gate_weight,
        bias,
        top_k,
        routed_scaling_factor=0.5,
    )
    torch.testing.assert_close(from_scale["topk_weight"], baseline["topk_weight"] * 0.5)

    from_k = kimi_k3_router_topk_torch_ref(hidden_states, gate_weight, bias, top_k + 1)
    assert from_k["topk_idx"].shape == (tokens, top_k + 1)
