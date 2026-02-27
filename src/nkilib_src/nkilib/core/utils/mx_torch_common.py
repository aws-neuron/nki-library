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

"""Generic MX (Microscaling) quantization primitives for PyTorch reference implementations.

Provides block-scaled MX math operations (float4/float8 unpacking, MX quantization,
MX matmul) used by multiple kernel torch references (MLP, MoE, QKV, etc.).

These are stateless utilities with no kernel-specific dependencies.

Note: x4 packed numpy arrays are the only numpy inputs — they arrive as numpy
because torch_ref_wrapper cannot convert them. The unpack functions convert them
to torch immediately. All other inputs are torch tensors.
"""

import numpy as np
import torch

# float4_e2m1fn: S(1) E(2) M(1), bias=1, finite-only (no inf/nan)
# All 16 possible 4-bit values mapped to their float32 equivalents.
# Index = raw 4-bit pattern, Value = decoded float32.
#
# Positive values (sign=0, indices 0-7):
#   0b0000 -> +0.0    (zero)
#   0b0001 -> +0.5    (subnormal: 0.5 * 2^(1-bias) = 0.5)
#   0b0010 -> +1.0    (exp=1: 1.0 * 2^(1-1) = 1.0)
#   0b0011 -> +1.5    (exp=1: 1.5 * 2^(1-1) = 1.5)
#   0b0100 -> +2.0    (exp=2: 1.0 * 2^(2-1) = 2.0)
#   0b0101 -> +3.0    (exp=2: 1.5 * 2^(2-1) = 3.0)
#   0b0110 -> +4.0    (exp=3: 1.0 * 2^(3-1) = 4.0)
#   0b0111 -> +6.0    (exp=3: 1.5 * 2^(3-1) = 6.0)
# Negative values (sign=1, indices 8-15): same magnitudes, negated.
_FP4_LUT = torch.tensor(
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


def unpack_float4_x4(packed_np):
    """Unpack float4_e2m1fn_x4 numpy array [P, F] -> torch float32 [P, F*4]."""
    raw = torch.from_numpy(packed_np.view(np.uint16).astype(np.int32))
    nibbles = torch.stack(
        [
            raw & 0xF,
            (raw >> 4) & 0xF,
            (raw >> 8) & 0xF,
            (raw >> 12) & 0xF,
        ],
        dim=-1,
    )
    return _FP4_LUT[nibbles.reshape(raw.shape[:-1] + (raw.shape[-1] * 4,))]


def unpack_float8_e5m2_x4(packed_np):
    """Unpack float8_e5m2_x4 numpy array [P, F] -> torch float32 [P, F*4].

    float8_e5m2: S(1) E(5) M(2), bias=15.
    """
    raw = torch.from_numpy(packed_np.view(np.uint32).astype(np.int64))
    raw_bytes = torch.stack(
        [
            raw & 0xFF,
            (raw >> 8) & 0xFF,
            (raw >> 16) & 0xFF,
            (raw >> 24) & 0xFF,
        ],
        dim=-1,
    )
    flat = raw_bytes.reshape(raw.shape[:-1] + (raw.shape[-1] * 4,))
    sign = (flat >> 7) & 1
    exp = (flat >> 2) & 0x1F
    man = flat & 0x3
    subnormal = (man.float() / 4.0) * (2.0**-14)
    normal = (1.0 + man.float() / 4.0) * torch.pow(2.0, exp.float() - 15.0)
    val = torch.where(exp == 0, subnormal, normal)
    return torch.where(sign == 1, -val, val)


def unpack_float8_e4m3fn_x4(packed_np):
    """Unpack float8_e4m3fn_x4 numpy array [P, F] -> torch float32 [P, F*4].

    float8_e4m3fn: S(1) E(4) M(3), bias=7, no inf, max=448.
    """
    raw = torch.from_numpy(packed_np.view(np.uint32).astype(np.int64))
    raw_bytes = torch.stack(
        [
            raw & 0xFF,
            (raw >> 8) & 0xFF,
            (raw >> 16) & 0xFF,
            (raw >> 24) & 0xFF,
        ],
        dim=-1,
    )
    flat = raw_bytes.reshape(raw.shape[:-1] + (raw.shape[-1] * 4,))
    sign = (flat >> 7) & 1
    exp = (flat >> 3) & 0xF
    man = flat & 0x7
    subnormal = (man.float() / 8.0) * (2.0**-6)
    normal = (1.0 + man.float() / 8.0) * torch.pow(2.0, exp.float() - 7.0)
    val = torch.where(exp == 0, subnormal, normal)
    return torch.where(sign == 1, -val, val)


def mx_matmul(stationary, moving, stationary_scale, moving_scale):
    """Hardware-accurate MX block-scaled matmul. All inputs are torch tensors.

    Reshapes to [P, F//4, 4], applies block scale per 8x1x4 block,
    then einsum("kiq,kjq->ij").
    """
    moving = moving.reshape(moving.shape[0], moving.shape[1] // 4, 4)
    MP, MF0, _ = moving.shape

    stationary = stationary.reshape(stationary.shape[0], stationary.shape[1] // 4, 4)
    SP, SF0, _ = stationary.shape

    ms = torch.pow(2.0, moving_scale[:, :MF0] - 127.0)
    ms = ms.unsqueeze(2).repeat(1, 1, 4).unsqueeze(1).repeat(1, 8, 1, 1).reshape(-1, MF0, 4)[:MP]
    moving = moving * ms

    ss = torch.pow(2.0, stationary_scale[:, :SF0] - 127.0)
    ss = ss.unsqueeze(2).repeat(1, 1, 4).unsqueeze(1).repeat(1, 8, 1, 1).reshape(-1, SF0, 4)[:SP]
    stationary = stationary * ss

    return torch.einsum("kiq,kjq->ij", stationary, moving)


def quantize_to_mx_fp8(data_f32):
    """Quantize float32 numpy [P, F] to mxfp8 x4 format. P%8==0, F%4==0 required.

    Returns: (x4_packed numpy, scale uint8 numpy [P//8, F//4])
    """
    P, F = data_f32.shape
    if P % 8 != 0 or F % 4 != 0:
        raise ValueError(f"Shape ({P}, {F}) must be divisible by (8, 4) for MX block quantization")

    max_exp_fp8 = 8
    max_val = 448.0

    exp_field = ((data_f32.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
    block_max_exp = exp_field.reshape(P // 8, 8, F // 4, 4).max(axis=(1, 3))
    scale_uint8 = (block_max_exp.astype(np.int32) - max_exp_fp8).clip(0, 255).astype(np.uint8)

    scale_factors = np.power(2.0, block_max_exp.astype(np.float64) - max_exp_fp8 - 127)
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)
    clipped = np.clip(
        data_f32.astype(np.float64) / np.where(scale_expanded == 0, 1.0, scale_expanded), -max_val, max_val
    )

    from neuronxcc.starfish.support.dtype import float8_e4m3fn_x4, static_cast

    return static_cast(clipped.astype(np.float32), float8_e4m3fn_x4), scale_uint8
