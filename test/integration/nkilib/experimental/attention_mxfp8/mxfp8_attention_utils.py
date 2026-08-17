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

"""Utilities for MXFP8 attention TKG test harness.

Provides the swizzle+quantize+pack layout transformation that converts BF16 KV
cache blocks into the packed MXFP8 format consumed by the kernel.

MXFP8 Block KV Layout
----------------------
Each block stores block_len tokens of d_head dimensions. For the kernel, the data
is swizzled into the interleaved layout required by ``nisa.quantize_mx`` / ``nisa.nc_matmul_mx``,
then quantized. Scales are packed into the same tensor as the data.

K cache swizzle (contraction along d_head=128):
    [block_len, d_head] → swizzle → [d_head//4, block_len*4] BF16
    → quantize_mx → mx_data [32P, block_len] x4  +  mx_scale [4P, block_len] uint8
    → pack scales as FP32 view → [32P, block_len + 32] FP32

V cache swizzle (contraction along block_len=128):
    [d_head, block_len] → swizzle → [block_len//4, d_head*4] BF16
    → quantize_mx → mx_data [32P, d_head] x4  +  mx_scale [4P, d_head] uint8
    → pack scales as FP32 view → [32P, d_head + 32] FP32
"""

from typing import Tuple

import nki.language as nl
import numpy as np
from neuronxcc.starfish.support.dtype import static_cast  # noqa: F401  (re-exported for tests)

from test.utils.mx_utils import quantize_mx_golden

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_LEN = 128
D_HEAD = 128
X4_PACK_FACTOR = 4
SBUF_QUADRANT_SIZE = 32


# ---------------------------------------------------------------------------
# Swizzle + Quantize
# ---------------------------------------------------------------------------


def swizzle_for_mx(tensor_2d: np.ndarray) -> np.ndarray:
    """Swizzle a [T, H] BF16/FP32 tensor into the interleaved layout for quantize_mx.

    The transformation groups 4 elements that are H/4 apart along H and packs them
    adjacent on the free dimension, producing shape [H//4, T*4].

    Args:
        tensor_2d: Input array of shape [T, H] in float32.

    Returns:
        Swizzled array of shape [H//4, T*4] in float32.
    """
    T, H = tensor_2d.shape
    assert H % X4_PACK_FACTOR == 0, f"H={H} must be divisible by {X4_PACK_FACTOR}"
    P_out = H // X4_PACK_FACTOR
    return tensor_2d.reshape(T, P_out, X4_PACK_FACTOR).transpose(1, 0, 2).reshape(P_out, T * X4_PACK_FACTOR)


def swizzle_quantize_block(
    block_bf16: np.ndarray,
    contraction_dim: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Swizzle and MX-quantize a single [block_len, d_head] BF16 block.

    Args:
        block_bf16: Shape [block_len, d_head] float32 data for one cache block.
        contraction_dim: 'd_head' for K cache, 'block_len' for V cache.
            Determines which axis becomes the partition (contraction) dimension.

    Returns:
        mx_data: Quantized data, shape [32, F] in x4 dtype (stored as float32 view).
        mx_scale: Scale tensor, shape [4, F] in uint8.
        Where F = block_len for K cache, F = d_head for V cache.
    """
    assert block_bf16.shape == (BLOCK_LEN, D_HEAD)

    if contraction_dim == "d_head":
        # K: contraction along d_head → swizzle [block_len, d_head]
        to_swizzle = block_bf16  # [T=block_len, H=d_head]
    elif contraction_dim == "block_len":
        # V: contraction along block_len → transpose first so block_len is H
        to_swizzle = block_bf16.T  # [T=d_head, H=block_len]
    else:
        raise ValueError(f"contraction_dim must be 'd_head' or 'block_len', got {contraction_dim}")

    swizzled = swizzle_for_mx(to_swizzle.astype(np.float32))
    # swizzled shape: [32, T*4] where T is the non-contraction dim
    mx_data, mx_scale = quantize_mx_golden(swizzled, nl.float8_e4m3fn_x4)
    return mx_data, mx_scale


def pack_mx_block(mx_data: np.ndarray, mx_scale: np.ndarray) -> np.ndarray:
    """Pack MX data and scales into a single [32P, F+32] FP32 tensor.

    The mx_scale [4P, F] uint8 is viewed as FP32 [4P, F//4], zero-padded to
    [32P, F//4], then concatenated to mx_data (viewed as FP32) along the free dim.

    Args:
        mx_data: [32, F] in x4 dtype (from quantize_mx_golden).
        mx_scale: [4, F] in uint8.

    Returns:
        Packed tensor [32, F + F//4] in FP32 view, where the last F//4 columns
        are the zero-padded scales.
    """
    P_data, F_data = mx_data.shape  # 32, F (in x4 elements)
    P_scale, F_scale = mx_scale.shape  # 4, F (in uint8)
    assert P_data == SBUF_QUADRANT_SIZE, f"Expected 32 partitions, got {P_data}"
    assert F_data == F_scale, f"Data F={F_data} != Scale F={F_scale}"

    # View mx_data as float32 (x4 types are 4 bytes each, same as float32)
    data_fp32 = mx_data.view(np.float32)  # [32, F]

    # View scales as FP32: group 4 uint8 into 1 float32
    assert F_scale % X4_PACK_FACTOR == 0, f"Scale F={F_scale} must be divisible by 4"
    scale_fp32 = mx_scale.view(np.float32)  # [4, F//4]

    # Zero-pad scale to [32, F//4]
    scale_padded = np.zeros((P_data, scale_fp32.shape[1]), dtype=np.float32)
    scale_padded[:P_scale] = scale_fp32

    # Concatenate along free dim
    return np.concatenate([data_fp32, scale_padded], axis=1)


def quantize_kv_cache_block(
    block_bf16: np.ndarray,
    contraction_dim: str,
) -> np.ndarray:
    """Swizzle, quantize, and pack a single KV cache block.

    Args:
        block_bf16: [block_len, d_head] float32 data.
        contraction_dim: 'd_head' for K, 'block_len' for V.

    Returns:
        Packed [32, F + F//4] FP32 tensor (data + zero-padded scales).
    """
    mx_data, mx_scale = swizzle_quantize_block(block_bf16, contraction_dim)
    return pack_mx_block(mx_data, mx_scale)
