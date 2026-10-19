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

"""CPU-only semantic tests for Kimi-K3 routed-expert decode."""

import pytest
import torch
from nkilib_src.nkilib.experimental.kimi_k3.latent_moe_mx_prefill import latent_moe_dense_torch_ref

_BETA = 4.0
_LINEAR_BETA = 25.0


def _make_dense_inputs(tokens: int):
    torch.manual_seed(42)
    hidden = 8
    intermediate = 6
    experts = 3
    return {
        "latent": (torch.randn(tokens, hidden) * 0.75).to(torch.bfloat16),
        "local_affinities": torch.softmax(torch.randn(tokens, experts), dim=-1),
        "gate_weights": (torch.randn(experts, intermediate, hidden) * 0.5).to(torch.bfloat16),
        "up_weights": (torch.randn(experts, intermediate, hidden) * 0.5).to(torch.bfloat16),
        "down_weights": (torch.randn(experts, hidden, intermediate) * 0.5).to(torch.bfloat16),
        "beta": _BETA,
        "linear_beta": _LINEAR_BETA,
    }


def _explicit_situ_reference(inputs):
    latent = inputs["latent"].to(torch.bfloat16)
    gate = torch.einsum("td,eid->tei", latent, inputs["gate_weights"].to(torch.bfloat16))
    up = torch.einsum("td,eid->tei", latent, inputs["up_weights"].to(torch.bfloat16))
    gate_f32 = gate.float()
    up_f32 = up.float()
    gate_term = inputs["beta"] * torch.tanh(gate_f32 / inputs["beta"]) * torch.sigmoid(gate_f32)
    up_term = inputs["linear_beta"] * torch.tanh(up_f32 / inputs["linear_beta"])
    activation = (gate_term * up_term).to(torch.bfloat16)
    expert_out = torch.einsum("tei,edi->ted", activation, inputs["down_weights"].to(torch.bfloat16))
    return (expert_out.float() * inputs["local_affinities"].float().unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 4])
def test_exact_two_sided_situ_for_decode_token_counts(tokens):
    inputs = _make_dense_inputs(tokens)

    actual = latent_moe_dense_torch_ref(**inputs)

    assert torch.equal(actual, _explicit_situ_reference(inputs))


def test_exact_situ_rejects_hard_clamped_silu():
    latent = torch.tensor([[1.0]], dtype=torch.bfloat16)
    affinities = torch.ones((1, 1), dtype=torch.float32)
    gate_weights = torch.tensor([[[12.0]]], dtype=torch.bfloat16)
    up_weights = torch.tensor([[[40.0]]], dtype=torch.bfloat16)
    down_weights = torch.ones((1, 1, 1), dtype=torch.bfloat16)

    exact = latent_moe_dense_torch_ref(
        latent=latent,
        local_affinities=affinities,
        gate_weights=gate_weights,
        up_weights=up_weights,
        down_weights=down_weights,
        beta=_BETA,
        linear_beta=_LINEAR_BETA,
    )

    gate = torch.einsum("td,eid->tei", latent, gate_weights).float()
    up = torch.einsum("td,eid->tei", latent, up_weights).float()
    hard_gate = torch.clamp(gate, min=-_BETA, max=_BETA) * torch.sigmoid(gate)
    hard_up = torch.clamp(up, min=-_LINEAR_BETA, max=_LINEAR_BETA)
    hard_clamped = (
        torch.einsum(
            "tei,edi->ted",
            (hard_gate * hard_up).to(torch.bfloat16),
            down_weights,
        )
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    assert torch.equal(
        exact,
        _explicit_situ_reference(
            {
                "latent": latent,
                "local_affinities": affinities,
                "gate_weights": gate_weights,
                "up_weights": up_weights,
                "down_weights": down_weights,
                "beta": _BETA,
                "linear_beta": _LINEAR_BETA,
            }
        ),
    )
    assert not torch.equal(exact, hard_clamped)


@pytest.mark.parametrize(
    ("parameter", "mutated_value"),
    [
        ("latent", None),
        ("local_affinities", None),
        ("gate_weights", None),
        ("up_weights", None),
        ("down_weights", None),
        ("beta", 2.0),
        ("linear_beta", 8.0),
    ],
)
def test_each_decode_input_changes_t1_output(parameter, mutated_value):
    baseline_inputs = _make_dense_inputs(tokens=1)
    baseline = latent_moe_dense_torch_ref(**baseline_inputs)
    mutated_inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in baseline_inputs.items()
    }
    if mutated_value is None:
        mutated_inputs[parameter] += 0.25
    else:
        mutated_inputs[parameter] = mutated_value

    mutated = latent_moe_dense_torch_ref(**mutated_inputs)

    assert not torch.equal(mutated, baseline)
