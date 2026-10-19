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

"""PyTorch references for Kimi-K3 :func:`moe_tkg_decode`."""

import math

import nki.language as nl
import numpy as np
import torch

from ....core.utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
)
from ..latent_moe_mx_prefill import latent_moe_mx_prefill_torch_ref

_PMAX = 128
_MX_TILE = 512
_Q_WIDTH = 4
_Q_HEIGHT = 8


def moe_tkg_decode_torch_ref(
    latent,
    local_affinities,
    expert_gate_up_weights,
    expert_down_weights,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> dict[str, torch.Tensor]:
    """Match the kernel's hardware-MX and BF16 rounding boundaries."""
    latent_t = _to_torch(latent).to(torch.bfloat16)
    affinities_t = _to_torch(local_affinities).float()
    gate_up_scale_t = _to_torch(expert_gate_up_weights_scale).to(torch.uint8)
    down_scale_t = _to_torch(expert_down_weights_scale).to(torch.uint8)
    gate_up_weights_np = np.asarray(expert_gate_up_weights)
    down_weights_np = np.asarray(expert_down_weights)

    tokens, hidden = latent_t.shape
    experts = gate_up_weights_np.shape[0]
    intermediate = gate_up_weights_np.shape[-1]
    n_hidden_tiles = hidden // _MX_TILE
    padded_tokens = math.ceil(tokens / _Q_WIDTH) * _Q_WIDTH

    hidden_swizzled = torch.zeros(
        (_PMAX, n_hidden_tiles, padded_tokens, _Q_WIDTH),
        dtype=torch.float32,
    )
    hidden_swizzled[:, :, :tokens, :] = (
        latent_t.reshape(tokens, n_hidden_tiles, _PMAX, _Q_WIDTH).permute(2, 1, 0, 3).float()
    )
    hidden_quant_np, hidden_scale_np = quantize_to_mx(
        hidden_swizzled.reshape(_PMAX, -1).numpy(),
        nl.float8_e4m3fn_x4,
    )
    hidden_quant_np = hidden_quant_np.reshape(_PMAX, n_hidden_tiles, padded_tokens)
    hidden_scale_t = torch.from_numpy(hidden_scale_np.reshape(_PMAX // _Q_HEIGHT, n_hidden_tiles, padded_tokens))

    output = torch.zeros((padded_tokens, hidden), dtype=torch.float32)
    for expert_idx in range(experts):
        gate = _gate_up_projection_torch(
            hidden_qtz=hidden_quant_np,
            hidden_scale=hidden_scale_t,
            weight_qtz=gate_up_weights_np[expert_idx, :, 0, :, :],
            weight_scale=gate_up_scale_t[expert_idx, :, 0, :, :],
            hidden=hidden,
            intermediate=intermediate,
            tokens=padded_tokens,
        )
        up = _gate_up_projection_torch(
            hidden_qtz=hidden_quant_np,
            hidden_scale=hidden_scale_t,
            weight_qtz=gate_up_weights_np[expert_idx, :, 1, :, :],
            weight_scale=gate_up_scale_t[expert_idx, :, 1, :, :],
            hidden=hidden,
            intermediate=intermediate,
            tokens=padded_tokens,
        )

        gate_f32 = gate.float()
        up_f32 = up.float()
        gate_term = beta * torch.tanh(gate_f32 / beta) * torch.sigmoid(gate_f32)
        up_term = linear_beta * torch.tanh(up_f32 / linear_beta)
        activation = (gate_term * up_term).to(torch.bfloat16)

        expert_out = _down_projection_torch(
            activation=activation,
            weight_qtz=down_weights_np[expert_idx],
            weight_scale=down_scale_t[expert_idx],
            hidden=hidden,
            intermediate=intermediate,
            tokens=padded_tokens,
        )
        affinity = torch.zeros((padded_tokens, 1), dtype=torch.float32)
        affinity[:tokens] = affinities_t[:, expert_idx : expert_idx + 1]
        output += expert_out.float() * affinity

    return {"out": output[:tokens].to(torch.bfloat16)}


def _gate_up_projection_torch(
    hidden_qtz: np.ndarray,
    hidden_scale: torch.Tensor,
    weight_qtz: np.ndarray,
    weight_scale: torch.Tensor,
    hidden: int,
    intermediate: int,
    tokens: int,
) -> torch.Tensor:
    """Mirror one BF16 PSUM accumulation per hidden 512 tile."""
    n_hidden_tiles = hidden // _MX_TILE
    n_intermediate_tiles = intermediate // _MX_TILE
    hidden_unpacked = unpack_float8_e4m3fn_x4(hidden_qtz.transpose(1, 0, 2).reshape(hidden // _Q_WIDTH, tokens))
    weight_unpacked = unpack_float4_x4(weight_qtz.transpose(1, 0, 2).reshape(hidden // _Q_WIDTH, intermediate))
    hidden_scales = hidden_scale.permute(1, 0, 2)
    weight_scales = weight_scale.permute(1, 0, 2)

    result = torch.zeros((intermediate, tokens), dtype=torch.float32)
    for tile_h in range(n_hidden_tiles):
        packed_rows = slice(tile_h * _PMAX, (tile_h + 1) * _PMAX)
        tile_result = mx_matmul(
            weight_unpacked[packed_rows],
            hidden_unpacked[packed_rows],
            weight_scales[tile_h],
            hidden_scales[tile_h],
        )
        result = (result + tile_result).to(torch.bfloat16).float()

    return result.reshape(n_intermediate_tiles, _Q_WIDTH, _PMAX, tokens).permute(2, 0, 3, 1).to(torch.bfloat16)


def _down_projection_torch(
    activation: torch.Tensor,
    weight_qtz: np.ndarray,
    weight_scale: torch.Tensor,
    hidden: int,
    intermediate: int,
    tokens: int,
) -> torch.Tensor:
    """Mirror activation quantization and one BF16 PSUM update per I512 tile."""
    n_intermediate_tiles = intermediate // _MX_TILE
    activation_flat = activation.float().permute(1, 0, 2, 3).reshape(-1, tokens * _Q_WIDTH).numpy()
    activation_qtz, activation_scale = quantize_to_mx(activation_flat, nl.float8_e4m3fn_x4)
    activation_unpacked = unpack_float8_e4m3fn_x4(activation_qtz)
    weight_unpacked = unpack_float4_x4(weight_qtz.transpose(1, 0, 2).reshape(-1, hidden))
    activation_scales = torch.from_numpy(activation_scale.astype(np.float32)).reshape(
        n_intermediate_tiles,
        _PMAX // _Q_HEIGHT,
        tokens,
    )
    weight_scales = weight_scale.permute(1, 0, 2)

    result = torch.zeros((tokens, hidden), dtype=torch.float32)
    for tile_i in range(n_intermediate_tiles):
        packed_rows = slice(tile_i * _PMAX, (tile_i + 1) * _PMAX)
        tile_result = mx_matmul(
            activation_unpacked[packed_rows],
            weight_unpacked[packed_rows],
            activation_scales[tile_i],
            weight_scales[tile_i],
        )
        result = (result + tile_result).to(torch.bfloat16).float()
    return result.to(torch.bfloat16)


def moe_tkg_decode_dense_torch_ref(
    latent,
    local_affinities,
    expert_gate_up_weights,
    expert_down_weights,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> dict[str, torch.Tensor]:
    """Evaluate the exact dense serving-weight oracle without activation MX quantization."""
    return latent_moe_mx_prefill_torch_ref(
        latent=latent,
        local_affinities=local_affinities,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        beta=beta,
        linear_beta=linear_beta,
    )


def _to_torch(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if "bfloat16" in str(array.dtype):
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(array)
