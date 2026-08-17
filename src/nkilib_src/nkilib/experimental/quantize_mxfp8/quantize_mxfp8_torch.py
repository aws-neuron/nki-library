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

"""PyTorch/numpy reference implementation for quantize_block_mxfp8_kernel.

Computes the expected quantized FP8 data and float8_e8m0fnu scales in the same
interleaved/packed layout the kernel produces, using only src-tree
utilities (no neuronxcc test internals).
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np

from ..mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    get_remainder_partition_offset,
    get_scale_output_shape,
    get_scale_packing_info,
)

Q_TILE_K = 128

# Alternative emax values used by the kernel (--enable-mx-alternative-emax).
# Standard MX emax is 15/8; alternative is 14/7.
_ALT_MAX_EXP = {
    nl.float8_e5m2_x4: 14,
    nl.float8_e4m3fn_x4: 7,
}

# Clipping max values are the standard FP8 max representable values
# (NOT affected by alternative emax — only scale computation changes).
_FP8_MAX_VAL = {
    nl.float8_e5m2_x4: 57344.0,
    nl.float8_e4m3fn_x4: 448.0,
}

_FP8_DTYPE_MAP = {
    "float8_e5m2": nl.float8_e5m2_x4,
    "float8_e4m3fn": nl.float8_e4m3fn_x4,
}


def quantize_block_mxfp8_torch_ref(
    src_tensor: np.ndarray,
    return_fp8_dtype: str,
    run_with_lnc2: bool = False,
    enable_scale_packing: bool = True,
) -> dict[str, np.ndarray]:
    """Golden reference for quantize_block_mxfp8_kernel.

    Computes the expected quantized data and scales using MX block
    quantization with alternative emax, then arranges them in the same
    interleaved/packed layout the kernel produces.

    Returns:
        dict with "quantized_data_hbm" (FP8 ndarray) and "quantized_scales_hbm" (float8_e8m0fnu ndarray).
    """
    F, K = src_tensor.shape
    non_x4_dtype = _get_non_x4_dtype(return_fp8_dtype)
    x4_dtype = _get_x4_dtype(return_fp8_dtype)

    # Interleave input, quantize, reshape to kernel output layout
    interleaved = _interleave_tensor(src_tensor.T)  # (K, F) -> (K//4, F*4)
    golden_data_x4, golden_scales = _quantize_mx_alt_emax(interleaved, x4_dtype)
    golden_data = golden_data_x4.view(non_x4_dtype).reshape(K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR)

    return {
        "quantized_data_hbm": golden_data,
        "quantized_scales_hbm": _pack_scales(golden_scales, K, F, enable_scale_packing),
    }


"""Private helpers for quantization reference implementation."""


def _get_x4_dtype(return_fp8_dtype: str):
    return _FP8_DTYPE_MAP[return_fp8_dtype]


def _get_non_x4_dtype(return_fp8_dtype: str):
    return getattr(dt, return_fp8_dtype)


def _quantize_mx_alt_emax(data: np.ndarray, x4_dtype) -> tuple[np.ndarray, np.ndarray]:
    """MX block quantization with alternative emax values.

    Same algorithm as quantize_to_mx in mx_torch_common.py but uses the
    alternative max-exponent values (14/7 instead of 15/8) that the kernel
    uses when compiled with --enable-mx-alternative-emax.

    Args:
        data: float32 array [P, F], P%8==0, F%4==0.
        x4_dtype: Target nl x4 dtype.

    Returns:
        (x4_packed [P, F//4], scale_uint8 [P//8, F//4])
    """
    data_f32 = dt.static_cast(data, np.float32) if data.dtype != np.float32 else data
    max_exp = _ALT_MAX_EXP[x4_dtype]
    max_val = _FP8_MAX_VAL[x4_dtype]

    P, F = data_f32.shape
    exp_field = ((data_f32.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
    block_max_exp = exp_field.reshape(P // 8, 8, F // 4, 4).max(axis=(1, 3))
    # Scale as int (matches neuronxcc: uint8 - int -> int64, no clip)
    scale_int = block_max_exp.astype(np.uint8).astype(np.int64) - max_exp

    scale_factors = np.power(2.0, (scale_int - 127).astype(np.float64))
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)
    clipped = np.clip(data_f32.astype(np.float64) / scale_expanded, -max_val, max_val)

    return dt.static_cast(clipped.astype(np.float32), x4_dtype), scale_int.astype(np.uint8)


def _interleave_chunk(
    src_tensor: np.ndarray, dst_tensor: np.ndarray, src_p_start: int, dst_p_start: int, chunk_size: int, F: int
) -> None:
    """Interleave a single chunk of chunk_size P-elements into dst_tensor."""
    sub_tile_p = chunk_size // INTERLEAVE_FACTOR
    for sub_tile_idx in range(INTERLEAVE_FACTOR):
        for p_idx in range(sub_tile_p):
            src_p = src_p_start + sub_tile_idx * sub_tile_p + p_idx
            for f_idx in range(F):
                dst_tensor[dst_p_start + p_idx, f_idx * INTERLEAVE_FACTOR + sub_tile_idx] = src_tensor[src_p, f_idx]


def _interleave_tensor(src: np.ndarray, TILE_P: int = 512) -> np.ndarray:
    """Interleave rows within each tile: [P, F] -> [P//4, F*4].

    Processes full TILE_P tiles, then handles remainder as 256-tile + 128-tile,
    matching the kernel's tiling strategy.
    """
    P, F = src.shape
    dst = np.zeros((P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR), dtype=np.float32)

    NUM_FULL_TILES = P // TILE_P
    SUB_TILE_P = TILE_P // INTERLEAVE_FACTOR

    # Process full tiles
    for tile_idx in range(NUM_FULL_TILES):
        _interleave_chunk(
            src, dst, src_p_start=tile_idx * TILE_P, dst_p_start=tile_idx * SUB_TILE_P, chunk_size=TILE_P, F=F
        )

    # Process remainder as 256-tile then 128-tile
    remainder = P % TILE_P
    src_offset = NUM_FULL_TILES * TILE_P
    dst_offset = NUM_FULL_TILES * SUB_TILE_P

    if remainder >= 256:
        _interleave_chunk(src, dst, src_p_start=src_offset, dst_p_start=dst_offset, chunk_size=256, F=F)
        src_offset += 256
        dst_offset += 256 // INTERLEAVE_FACTOR
        remainder -= 256

    if remainder >= 128:
        _interleave_chunk(src, dst, src_p_start=src_offset, dst_p_start=dst_offset, chunk_size=128, F=F)

    return dst.astype(src.dtype)


def _pack_scales(golden_scales: np.ndarray, K: int, F: int, enable_scale_packing: bool) -> np.ndarray:
    """Arrange dense scales [K//4//8, F] into the kernel's output layout.

    Returns scales as float8_e8m0fnu matching the kernel output dtype.
    """
    # View source uint8 scales as float8_e8m0fnu so assignments into the e8m0
    # output buffer preserve the binary representation.
    golden_scales_e8m0 = golden_scales.view(nl.float8_e8m0fnu)

    if not enable_scale_packing:
        out = np.zeros((K // 4, F), dtype=nl.float8_e8m0fnu)
        for row_idx in range(golden_scales_e8m0.shape[0]):
            out[(row_idx // 4) * 32 + (row_idx % 4), :] = golden_scales_e8m0[row_idx, :]
        return out

    L_TILE_K = 512
    NUM_TILES_IN_K = K // L_TILE_K
    REMAINDER_K = K % L_TILE_K
    HAS_REMAINDER_256 = REMAINDER_K >= 256
    HAS_REMAINDER_128 = REMAINDER_K % 256 >= 128

    golden_offset = 0
    tile_idx = 0

    def _pack_tile(tile_k_size, golden_offset, tile_idx, out, k_idx_within_tile=0):
        """Pack scales for a single tile into the packed output."""
        scaling_group_idx, _, slot_partition_offset = get_scale_packing_info(tile_idx, True)
        remainder_partition_offset = get_remainder_partition_offset(k_idx_within_tile, Q_TILE_K)
        scale_p_start = scaling_group_idx * Q_TILE_K
        num_rows = (tile_k_size // INTERLEAVE_FACTOR) // 8
        tile_scales = golden_scales_e8m0[golden_offset : golden_offset + num_rows, :]
        for row_idx in range(num_rows):
            packed_row = (row_idx // 4) * 32 + (row_idx % 4)
            out[scale_p_start + slot_partition_offset + packed_row + remainder_partition_offset, :] = tile_scales[
                row_idx, :
            ]
        return golden_offset + num_rows

    packed_scale_P, _ = get_scale_output_shape(K, F, Q_TILE_K, enable_scale_packing=True)
    out = np.zeros((packed_scale_P, F), dtype=nl.float8_e8m0fnu)

    # Process full 512-tiles
    for i in range(NUM_TILES_IN_K):
        golden_offset = _pack_tile(L_TILE_K, golden_offset, tile_idx, out)
        tile_idx += 1

    # Process remainder tiles - both 256 and 128 share the same tile_idx
    if HAS_REMAINDER_256:
        golden_offset = _pack_tile(256, golden_offset, tile_idx, out)

    if HAS_REMAINDER_128:
        k_idx_within_tile = 256 if HAS_REMAINDER_256 else 0
        golden_offset = _pack_tile(128, golden_offset, tile_idx, out, k_idx_within_tile)

    # Handle small remainders (< 128 logical elements) that still have scale data
    remaining = REMAINDER_K - (256 if HAS_REMAINDER_256 else 0) - (128 if HAS_REMAINDER_128 else 0)
    if remaining > 0 and remaining % INTERLEAVE_FACTOR == 0:
        k_idx_within_tile = REMAINDER_K - remaining
        golden_offset = _pack_tile(remaining, golden_offset, tile_idx, out, k_idx_within_tile)

    return out
