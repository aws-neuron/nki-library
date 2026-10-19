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

"""Utility functions and constants for MXFP8 quantization operations."""

import nki
import nki.language as nl

from ...matmul_mxfp8.matmul_mxfp8_constants import QuantScheme

INTERLEAVE_FACTOR = 4
MX_PARTITION_SIZE = 32
TILE_SIZE_GEMM_MOVING_MAX = 8192
MAX_TILES_PER_SCALE_PACKING_GROUP = 4  # Number of SBUF tiles packed into one scaling group
SCALE_PACKING_STRIDE = 4  # Partition offset between slots (0, 4, 8, 12)
# Quantize
L_TILE_K = 512
L_TILE_F = 512
Q_TILE_K = 128  # K dimension size for scale tiles (Q_TILE_K from quantize kernel)
MIN_K_FOR_QUANTIZATION = 128
MIN_F_FOR_QUANTIZATION = 8
MIN_F_FOR_LNC2 = 512

# Quantization-scheme names as they appear in kernel ABIs (a string, like return_fp8_dtype).
QUANT_SCHEME_WRAPX = QuantScheme.WRAPX.value
QUANT_SCHEME_1X32 = QuantScheme._1x32.value
QUANT_SCHEME_MAP = {QUANT_SCHEME_WRAPX: QuantScheme.WRAPX, QUANT_SCHEME_1X32: QuantScheme._1x32}

# K tile sizes used to decompose a sub-L_TILE_K remainder, largest first.
REMAINDER_TILE_SIZES = (256, 128)


def decompose_k_remainder(remainder_k: int) -> list:
    """
    Split a sub-L_TILE_K K remainder into load tiles, largest first.

    Any remainder aligned to MX_PARTITION_SIZE decomposes into at most one 256 tile, one
    128 tile, and one final sub-128 tile of 32/64/96 -- the binary expansion of
    ``remainder_k // MX_PARTITION_SIZE``. The decomposition defines the wrapX data layout of
    the remainder, so the quantize kernel, the matmul loaders and the goldens must share it.

    Args:
        remainder_k (int): K elements left after the full L_TILE_K tiles; a multiple of
            MX_PARTITION_SIZE and < L_TILE_K.

    Returns:
        list: Tile sizes in K, in processing order.
    """
    tile_sizes = []
    k_left = remainder_k
    for tile_size in REMAINDER_TILE_SIZES:
        if k_left >= tile_size:
            tile_sizes.append(tile_size)
            k_left -= tile_size
    if k_left > 0:
        tile_sizes.append(k_left)
    return tile_sizes


def get_fp8_dtype_x4(dtype_str: nki.dtype):
    """
    Map dtype string to NKI x4 dtype

    Args:
        dtype_str (str): String representation of FP8 dtype

    Returns:
        NKI x4 dtype: Corresponding x4 format dtype (nl.float8_e5m2_x4 or nl.float8_e4m3fn_x4)
    """
    # Fallback to original dictionary lookup for exact matches
    dtype_map = {
        "float8_e5m2": nl.float8_e5m2_x4,
        "nl.float8_e5m2_x4": nl.float8_e5m2_x4,
        "float8_e4m3fn": nl.float8_e4m3fn_x4,
        "nl.float8_e4m3fn_x4": nl.float8_e4m3fn_x4,
    }
    return dtype_map.get(dtype_str, nl.float8_e5m2_x4)


def get_fp8_dtype(dtype_str: str):
    """
    Map dtype string to NKI non-x4 dtype (OCP-compliant)

    Args:
        dtype_str (str): String representation of dtype ("float8_e5m2" or "float8_e4m3fn").

    Returns:
        nki.dtype: Corresponding NKI dtype object or string constant (float8_e5m2 or "float8_e4m3fn").
    """
    dtype_map = {
        "float8_e5m2": nl.float8_e5m2,
        "float8_e4m3fn": "float8_e4m3fn",
    }
    return dtype_map.get(dtype_str, nl.float8_e5m2)


def get_scale_packing_info(tile_k_index: int, enable_scale_packing: bool = True) -> tuple[int, int, int]:
    """
    Calculate scaling group and slot offset for a given tile.

    Args:
        tile_k_index (int): Index of the K tile
        enable_scale_packing (bool): Whether scale packing is enabled

    Returns:
        tuple[int, int, int]: (scaling_group_index, slot_within_group, partition_offset)

    Notes:
        Partition offsets: 0, 4, 8, 12 for slots 0, 1, 2, 3
    """
    if not enable_scale_packing:
        return 0, 0, 0

    scaling_group_index = tile_k_index // MAX_TILES_PER_SCALE_PACKING_GROUP
    slot_within_group = tile_k_index % MAX_TILES_PER_SCALE_PACKING_GROUP
    partition_offset = slot_within_group * SCALE_PACKING_STRIDE  # 0, 4, 8, 12

    return scaling_group_index, slot_within_group, partition_offset
