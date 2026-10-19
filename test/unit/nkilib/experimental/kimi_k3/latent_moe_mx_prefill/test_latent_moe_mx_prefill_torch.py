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

"""CPU-only component tests for the latent-MoE prefill reference."""

import pytest
import torch
from nkilib_src.nkilib.experimental.kimi_k3.latent_moe_mx_prefill import latent_moe_dense_torch_ref

T = 4
D = 8
I = 6
E = 3
_BETA = 4.0
_LINEAR_BETA = 25.0


def _make_inputs():
    torch.manual_seed(42)
    return {
        "latent": (torch.randn(T, D) * 0.75).to(torch.bfloat16),
        "local_affinities": torch.tensor(
            [
                [0.0, 0.75, 0.25],
                [1.0, 0.0, 0.0],
                [0.0, 0.4, 0.6],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "gate_weights": (torch.randn(E, I, D) * 0.5).to(torch.bfloat16),
        "up_weights": (torch.randn(E, I, D) * 0.5).to(torch.bfloat16),
        "down_weights": (torch.randn(E, D, I) * 0.5).to(torch.bfloat16),
        "beta": _BETA,
        "linear_beta": _LINEAR_BETA,
    }


def _explicit_reference(inputs):
    latent = inputs["latent"].to(torch.bfloat16)
    gate_weights = inputs["gate_weights"].to(torch.bfloat16)
    up_weights = inputs["up_weights"].to(torch.bfloat16)
    down_weights = inputs["down_weights"].to(torch.bfloat16)
    affinities = inputs["local_affinities"].float()
    beta = inputs["beta"]
    linear_beta = inputs["linear_beta"]

    gate = torch.einsum("td,eid->tei", latent, gate_weights)
    up = torch.einsum("td,eid->tei", latent, up_weights)
    gate_f32 = gate.float()
    up_f32 = up.float()
    situ_a = beta * torch.tanh(gate_f32 / beta) * torch.sigmoid(gate_f32)
    up_clipped = linear_beta * torch.tanh(up_f32 / linear_beta)
    activation = (situ_a * up_clipped).to(torch.bfloat16)
    expert_out = torch.einsum("tei,edi->ted", activation, down_weights)
    return torch.sum(expert_out.float() * affinities[:, :, None], dim=1).to(torch.bfloat16)


def _pre_scale_reference(inputs):
    gate = torch.einsum("td,eid->tei", inputs["latent"], inputs["gate_weights"])
    up = torch.einsum("td,eid->tei", inputs["latent"], inputs["up_weights"])
    situ_a = inputs["beta"] * torch.tanh(gate.float() / inputs["beta"]) * torch.sigmoid(gate.float())
    up_clipped = inputs["linear_beta"] * torch.tanh(up.float() / inputs["linear_beta"])
    activation = (situ_a * up_clipped).to(torch.bfloat16)
    scaled_activation = (activation.float() * inputs["local_affinities"].float().unsqueeze(-1)).to(torch.bfloat16)
    expert_out = torch.einsum("tei,edi->ted", scaled_activation, inputs["down_weights"])
    return expert_out.float().sum(dim=1).to(torch.bfloat16)


def test_exact_situ_component_and_all_remote_row():
    inputs = _make_inputs()

    actual = latent_moe_dense_torch_ref(**inputs)
    expected = _explicit_reference(inputs)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[3]).item() == 0


@pytest.mark.parametrize(
    "parameter",
    [
        "latent",
        "local_affinities",
        "gate_weights",
        "up_weights",
        "down_weights",
        "beta",
        "linear_beta",
    ],
)
def test_each_logical_input_changes_the_output(parameter):
    baseline_inputs = _make_inputs()
    baseline = latent_moe_dense_torch_ref(**baseline_inputs)
    mutated = _make_inputs()

    if parameter == "latent":
        mutated[parameter][1, 0] += 1.0
    elif parameter == "local_affinities":
        mutated[parameter][1, 0] += 0.5
    elif parameter == "gate_weights":
        mutated[parameter][0, 0, 0] += 1.0
    elif parameter == "up_weights":
        mutated[parameter][0, 0, 0] += 1.0
    elif parameter == "down_weights":
        mutated[parameter][0, 0, 0] += 1.0
    elif parameter == "beta":
        mutated[parameter] = 2.0
    else:
        mutated[parameter] = 8.0

    changed = latent_moe_dense_torch_ref(**mutated)
    assert not torch.equal(changed, baseline)


def test_affinity_is_applied_after_down_projection():
    inputs = _make_inputs()

    post_scaled = latent_moe_dense_torch_ref(**inputs)
    pre_scaled = _pre_scale_reference(inputs)

    assert torch.equal(post_scaled, _explicit_reference(inputs))
    assert not torch.equal(post_scaled, pre_scaled)


def test_expert_accumulation_is_fp32():
    inputs = _make_inputs()
    inputs["gate_weights"][1] = inputs["gate_weights"][0]
    inputs["gate_weights"][2] = inputs["gate_weights"][0]
    inputs["up_weights"][1] = inputs["up_weights"][0]
    inputs["up_weights"][2] = inputs["up_weights"][0]
    inputs["down_weights"][1] = inputs["down_weights"][0]
    inputs["down_weights"][2] = -inputs["down_weights"][0]
    inputs["local_affinities"][:] = torch.tensor([1.0, 1.0 / 256.0, 1.0])

    actual = latent_moe_dense_torch_ref(**inputs)

    single_expert_inputs = dict(inputs)
    single_expert_inputs["local_affinities"] = torch.zeros_like(inputs["local_affinities"])
    single_expert_inputs["local_affinities"][:, 0] = 1.0
    expert_out = latent_moe_dense_torch_ref(**single_expert_inputs)
    expected_fp32 = (expert_out.float() + expert_out.float() / 256.0 - expert_out.float()).to(torch.bfloat16)

    bf16_accumulation = torch.zeros_like(expert_out)
    for contribution in (expert_out, expert_out.float() / 256.0, -expert_out):
        bf16_accumulation = (bf16_accumulation.float() + contribution.float()).to(torch.bfloat16)

    assert torch.equal(actual, expected_fp32)
    assert torch.count_nonzero(actual).item() > 0
    assert not torch.equal(actual, bf16_accumulation)
