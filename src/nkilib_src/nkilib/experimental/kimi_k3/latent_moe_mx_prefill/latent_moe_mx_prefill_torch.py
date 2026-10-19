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

"""Independent PyTorch reference for Kimi-K3 :func:`latent_moe_mx_prefill`."""

import numpy as np
import torch

_PMAX = 128
_Q_WIDTH = 4
_Q_HEIGHT = 8
_MX_GROUP_SIZE = _Q_WIDTH * _Q_HEIGHT
_MX_CONTRACT_DIM = _PMAX * _Q_WIDTH

_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def latent_moe_mx_prefill_torch_ref(
    latent,
    local_affinities,
    expert_gate_up_weights,
    expert_down_weights,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> dict[str, torch.Tensor]:
    """Dequantize the serving layout and evaluate Kimi-K3's dense BF16 oracle."""
    latent_t = _to_torch(latent).to(torch.bfloat16)
    affinities_t = _to_torch(local_affinities).float()
    gate_up = _dequantize_gate_up_serving_layout(
        expert_gate_up_weights,
        expert_gate_up_weights_scale,
        out_dtype=torch.bfloat16,
    )
    down = _dequantize_down_serving_layout(
        expert_down_weights,
        expert_down_weights_scale,
        out_dtype=torch.bfloat16,
    )
    return {
        "out": latent_moe_dense_torch_ref(
            latent=latent_t,
            local_affinities=affinities_t,
            gate_weights=gate_up[:, 0],
            up_weights=gate_up[:, 1],
            down_weights=down,
            beta=beta,
            linear_beta=linear_beta,
            compute_dtype=torch.bfloat16,
        )
    }


def latent_moe_dense_torch_ref(
    latent: torch.Tensor,
    local_affinities: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Evaluate the exact dense model oracle, including BF16 matmul boundaries."""
    x = latent.to(compute_dtype)
    gate = torch.einsum("td,eid->tei", x, gate_weights.to(compute_dtype))
    up = torch.einsum("td,eid->tei", x, up_weights.to(compute_dtype))
    gate_f32 = gate.float()
    up_f32 = up.float()
    situ_a = beta * torch.tanh(gate_f32 / beta) * torch.sigmoid(gate_f32)
    up_clipped = linear_beta * torch.tanh(up_f32 / linear_beta)
    activation = (situ_a * up_clipped).to(compute_dtype)
    expert_out = torch.einsum("tei,edi->ted", activation, down_weights.to(compute_dtype))
    weighted = expert_out.float() * local_affinities.float().unsqueeze(-1)
    return weighted.sum(dim=1).to(compute_dtype)


def _dequantize_gate_up_serving_layout(
    packed,
    scale,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert ``[E,128,2,D/512,I]`` serving bytes to dense ``[E,2,I,D]``."""
    packed_np = np.asarray(packed)
    scale_t = _to_torch(scale).to(torch.uint8)
    if packed_np.ndim != 5:
        raise ValueError(f"Expected rank-5 gate/up weights, got {packed_np.shape}")

    experts, partitions, branches, n_d512, intermediate = packed_np.shape
    if partitions != _PMAX or branches != 2 or intermediate % _MX_CONTRACT_DIM != 0:
        raise ValueError(f"Unexpected gate/up serving shape {packed_np.shape}")
    expected_scale = (experts, _PMAX // _Q_HEIGHT, branches, n_d512, intermediate)
    if tuple(scale_t.shape) != expected_scale:
        raise ValueError(f"Expected gate/up scale {expected_scale}, got {tuple(scale_t.shape)}")

    n_i512 = intermediate // _MX_CONTRACT_DIM
    hidden = n_d512 * _MX_CONTRACT_DIM
    dense = torch.empty((experts, branches, intermediate, hidden), dtype=out_dtype)
    for expert_idx in range(experts):
        values = _unpack_float4_elements(packed_np[expert_idx]).reshape(
            _PMAX // _Q_HEIGHT,
            _Q_HEIGHT,
            branches,
            n_d512,
            n_i512,
            _Q_WIDTH,
            _PMAX // _Q_HEIGHT,
            _Q_HEIGHT,
            _Q_WIDTH,
        )
        factors = torch.exp2(scale_t[expert_idx].float() - 127.0).reshape(
            _PMAX // _Q_HEIGHT,
            branches,
            n_d512,
            n_i512,
            _Q_WIDTH,
            _PMAX // _Q_HEIGHT,
            _Q_HEIGHT,
        )
        values = values * factors[:, None, ..., None]
        dense[expert_idx] = values.permute(2, 4, 6, 7, 5, 3, 0, 1, 8).reshape(
            branches,
            intermediate,
            hidden,
        )
    return dense


def _dequantize_down_serving_layout(
    packed,
    scale,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert ``[E,128,I/512,D]`` serving bytes to dense ``[E,D,I]``."""
    packed_np = np.asarray(packed)
    scale_t = _to_torch(scale).to(torch.uint8)
    if packed_np.ndim != 4:
        raise ValueError(f"Expected rank-4 down weights, got {packed_np.shape}")

    experts, partitions, n_i512, hidden = packed_np.shape
    if partitions != _PMAX:
        raise ValueError(f"Unexpected down serving shape {packed_np.shape}")
    expected_scale = (experts, _PMAX // _Q_HEIGHT, n_i512, hidden)
    if tuple(scale_t.shape) != expected_scale:
        raise ValueError(f"Expected down scale {expected_scale}, got {tuple(scale_t.shape)}")

    intermediate = n_i512 * _MX_CONTRACT_DIM
    dense = torch.empty((experts, hidden, intermediate), dtype=out_dtype)
    for expert_idx in range(experts):
        values = _unpack_float4_elements(packed_np[expert_idx]).reshape(
            _PMAX // _Q_HEIGHT,
            _Q_HEIGHT,
            n_i512,
            hidden,
            _Q_WIDTH,
        )
        factors = torch.exp2(scale_t[expert_idx].float() - 127.0)
        values = values * factors[:, None, :, :, None]
        dense[expert_idx] = values.permute(3, 2, 0, 1, 4).reshape(hidden, intermediate)
    return dense


def _unpack_float4_elements(packed) -> torch.Tensor:
    """Decode every packed E2M1 nibble while preserving all tensor axes."""
    packed_np = np.asarray(packed)
    raw = torch.from_numpy(packed_np.view(np.uint16).astype(np.int32))
    nibbles = torch.stack(
        (
            raw & 0xF,
            (raw >> 4) & 0xF,
            (raw >> 8) & 0xF,
            (raw >> 12) & 0xF,
        ),
        dim=-1,
    )
    return _FP4_E2M1_LUT[nibbles]


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if "bfloat16" in str(array.dtype):
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(array)
