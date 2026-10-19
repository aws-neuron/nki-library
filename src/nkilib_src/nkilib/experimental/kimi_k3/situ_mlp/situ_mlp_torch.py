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

"""PyTorch oracle for the Kimi-K3 exact two-sided SiTU MLP."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "dtype") and str(value.dtype) == "bfloat16":
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(torch.bfloat16)
    return torch.as_tensor(value)


def kimi_k3_situ_mlp_torch_ref(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    beta: float,
    linear_beta: float,
):
    """Return the BF16 Kimi-K3 MLP output ``{"output": [T, H]}``."""
    hidden_states = _as_tensor(hidden_states)
    gate_proj_weight = _as_tensor(gate_proj_weight)
    up_proj_weight = _as_tensor(up_proj_weight)
    down_proj_weight = _as_tensor(down_proj_weight)

    if hidden_states.ndim != 2:
        raise ValueError(f"Expected hidden_states [T, H], got {tuple(hidden_states.shape)}")
    token_count, hidden_size = hidden_states.shape
    if not 1 <= token_count <= 128:
        raise ValueError(f"Expected 1 <= T <= 128, got T={token_count}")
    if gate_proj_weight.ndim != 2:
        raise ValueError(f"Expected gate_proj_weight [I, H], got {tuple(gate_proj_weight.shape)}")
    intermediate_size = gate_proj_weight.shape[0]
    expected_gate_up_shape = (intermediate_size, hidden_size)
    if tuple(gate_proj_weight.shape) != expected_gate_up_shape:
        raise ValueError(f"Expected gate_proj_weight {expected_gate_up_shape}, got {tuple(gate_proj_weight.shape)}")
    if tuple(up_proj_weight.shape) != expected_gate_up_shape:
        raise ValueError(f"Expected up_proj_weight {expected_gate_up_shape}, got {tuple(up_proj_weight.shape)}")
    expected_down_shape = (hidden_size, intermediate_size)
    if tuple(down_proj_weight.shape) != expected_down_shape:
        raise ValueError(f"Expected down_proj_weight {expected_down_shape}, got {tuple(down_proj_weight.shape)}")
    for name, value in (("beta", beta), ("linear_beta", linear_beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a real number, got {value!r}")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")

    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
    gate_f32 = gate.to(torch.float32)
    up_f32 = up.to(torch.float32)
    gate_term = float(beta) * torch.tanh(gate_f32 / float(beta)) * torch.sigmoid(gate_f32)
    up_term = float(linear_beta) * torch.tanh(up_f32 / float(linear_beta))
    activation = (gate_term * up_term).to(hidden_states.dtype)
    output = F.linear(activation, down_proj_weight)
    return {"output": output.to(hidden_states.dtype)}
