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

import neuron_dtypes as dt
import nki.language as nl
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


def quantize_to_mx(data, out_x4_dtype):
    """Quantize numpy [P, F] to MX x4 format. P%8==0, F%4==0 required.

    Accepts any numeric dtype (bf16, fp16, fp32, etc.) and casts to float32
    internally. Uses numpy because dt.static_cast (the only available packing
    function for MX x4 dtypes) operates on numpy arrays.

    Supports nl.float8_e4m3fn_x4, nl.float8_e5m2_x4, and nl.float4_e2m1fn_x4.

    Args:
        data (np.ndarray): Input array of shape [P, F] in any numeric dtype.
        out_x4_dtype: Target MX x4 dtype (nl.dtype).

    Returns:
        (x4_packed numpy [P, F//4], scale uint8 numpy [P//8, F//4])

    Notes:
        Scale is returned in dense logical layout [P//8, F//4] (one row per
        8-row block), NOT in hardware quadrant layout [P, F//4].
        Use quantize_mx_golden to get scale in hardware quadrant layout.
    """
    # Cast to float32 if needed (exponent extraction requires IEEE 754 float32)
    data_f32 = dt.static_cast(data, np.float32) if data.dtype != np.float32 else data

    # Resolve dtype: numpy custom dtypes (.dtype from array) are numpy.dtype objects,
    # not nl.* objects. Convert via str(dtype) to match against known names.
    _STR_TO_NL_DTYPE = {
        'float8_e5m2_x4': nl.float8_e5m2_x4,
        'float8_e4m3fn_x4': nl.float8_e4m3fn_x4,
        'float4_e2m1fn_x4': nl.float4_e2m1fn_x4,
    }
    dtype_str = str(out_x4_dtype)
    if dtype_str in _STR_TO_NL_DTYPE:
        out_x4_dtype = _STR_TO_NL_DTYPE[dtype_str]

    # max exponent and max representable value per MX dtype
    # float8_e5m2:   max_exp=15, max_val=57344
    # float8_e4m3fn: max_exp=8,  max_val=448
    # float4_e2m1fn: max_exp=2,  max_val=6
    _MX_DTYPE_PARAMS = {
        nl.float8_e5m2_x4: (15, 57344.0),
        nl.float8_e4m3fn_x4: (8, 448.0),
        nl.float4_e2m1fn_x4: (2, 6.0),
    }

    if out_x4_dtype not in _MX_DTYPE_PARAMS:
        raise ValueError(f"Unsupported out_x4_dtype: {out_x4_dtype}. Must be one of {list(_MX_DTYPE_PARAMS.keys())}")

    max_exp, max_val = _MX_DTYPE_PARAMS[out_x4_dtype]

    P, F = data_f32.shape
    if P % 8 != 0 or F % 4 != 0:
        raise ValueError(f"Shape ({P}, {F}) must be divisible by (8, 4) for MX block quantization")

    exp_field = ((data_f32.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
    block_max_exp = exp_field.reshape(P // 8, 8, F // 4, 4).max(axis=(1, 3))
    scale_uint8 = (block_max_exp.astype(np.int32) - max_exp).clip(0, 255).astype(np.uint8)

    scale_factors = np.power(2.0, block_max_exp.astype(np.float64) - max_exp - 127)
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)
    clipped = np.clip(
        data_f32.astype(np.float64) / np.where(scale_expanded == 0, 1.0, scale_expanded), -max_val, max_val
    )

    return dt.static_cast(clipped.astype(np.float32), out_x4_dtype), scale_uint8


def quantize_mx_golden(src_hbm, out_data_hbm, out_scale_hbm):
    """Golden reference matching nisa.quantize_mx hardware behavior.

    Calls quantize_to_mx (which returns dense scale [P//8, F//4]) and
    converts the scale to hardware quadrant layout [P, F//4] so it can
    be compared against nisa.quantize_mx output.

    Scale layout conversion:

        quantize_to_mx          →  hardware layout
        [P//8, F//4] dense         [P, F//4] sparse

        ┌────────┐                ┌────────┐
        │ s0     │                │ s0     │  ← partition 0
        │ s1     │                │ s1     │  ← partition 1
        │ s2     │  ──convert──►  │ s2     │  ← partition 2
        │ s3     │                │ s3     │  ← partition 3
        │ s4     │                │ 0      │  ← partitions 4-31 (zero padding)
        │ ...    │                │ ...    │
        └────────┘                │ s4     │  ← partition 32 (next quadrant)
                                  │ s5     │
                                  │ ...    │
                                  └────────┘

    Args:
        src_hbm: Input tensor (torch or numpy), any numeric dtype.
        out_data_hbm: Output data tensor (dtype determines MX format).
        out_scale_hbm: Output scale tensor (uint8).

    Returns:
        dict with 'out_data_hbm' (packed x4 numpy) and 'out_scale_hbm' (uint8 numpy [P, F//4]).
    """
    src_np = src_hbm.numpy() if hasattr(src_hbm, 'numpy') else src_hbm
    out_x4_dtype = out_data_hbm.dtype

    if out_x4_dtype == nl.float4_e2m1fn_x4:
        raise ValueError(
            "float4_e2m1fn_x4 is not supported by nisa.quantize_mx. Use float8_e4m3fn_x4 or float8_e5m2_x4 instead."
        )

    packed, scale_block = quantize_to_mx(src_np, out_x4_dtype)

    # Convert [P//8, F//4] block scale to hardware layout [P, F//4]:
    # 4 valid scale rows per 32-partition quadrant, rest zero.
    P = src_np.shape[0]
    _QUADRANT_SIZE = 32
    _SCALE_PER_QUADRANT = 4
    num_quadrants = P // _QUADRANT_SIZE
    scale_hw = np.zeros((P, scale_block.shape[1]), dtype=np.uint8)
    for quadrant_idx in range(num_quadrants):
        block_start = quadrant_idx * _SCALE_PER_QUADRANT
        hw_start = quadrant_idx * _QUADRANT_SIZE
        scale_hw[hw_start : hw_start + _SCALE_PER_QUADRANT, :] = scale_block[
            block_start : block_start + _SCALE_PER_QUADRANT, :
        ]

    return {
        "out_data_hbm": packed,
        "out_scale_hbm": scale_hw,
    }
