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

INTERLEAVE_FACTOR = 4
MX_PARTITION_SIZE = 32
TILE_SIZE_GEMM_MOVING_MAX = 8192
MAX_SCALE_PACKING_P_SIZE = 2048  # Max P dimension elements per scaling group (was MAX_P_PER_SCALING_GROUP)
MAX_TILES_PER_SCALE_PACKING_GROUP = 4  # Number of tiles packed into one scaling group
SCALE_PACKING_STRIDE = 4  # Partition offset between slots (0, 4, 8, 12)
# Quantize
L_TILE_K = 512
L_TILE_F = 512
Q_TILE_K = 128  # K dimension size for scale tiles (Q_TILE_K from quantize kernel)
MIN_K_FOR_QUANTIZATION = 128
MIN_F_FOR_QUANTIZATION = 8
MIN_F_FOR_LNC2 = 512


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


def are_scales_packed(data_shape: tuple, scales_shape: tuple, data_is_is_swizzled: bool = True) -> bool:
    """
    Determine if scales are packed by comparing data and scales shapes.

    When scales are packed, the K dimension of scales is smaller than data.
    Unpacked: scales_K == data_K
    Packed: scales_K < data_K (scales_K = ceil(data_K * 4 / MAX_SCALE_PACKING_P_SIZE) * SCALE_TILE_K)

    Args:
        data_shape (tuple): Shape of quantized data tensor (K_physical, F)
        scales_shape (tuple): Shape of scales tensor (K_scales, F)

    Returns:
        bool: True if scales are packed, False otherwise
    """
    data_K, data_F = data_shape
    scales_K = scales_shape[0]
    data_K_logical = data_K * INTERLEAVE_FACTOR if data_is_is_swizzled else data_K
    packed_scales_K, _ = calculate_packed_scale_shape(data_K_logical, data_F, Q_TILE_K)

    return scales_K == packed_scales_K and packed_scales_K != data_K


def scale_packing_not_possible(data_shape: tuple, scales_shape: tuple, data_is_is_swizzled: bool = True) -> bool:
    data_K, data_F = data_shape
    scales_K = scales_shape[0]
    data_K_logical = data_K * INTERLEAVE_FACTOR if data_is_is_swizzled else data_K
    packed_scales_K, _ = calculate_packed_scale_shape(data_K_logical, data_F, Q_TILE_K)

    return packed_scales_K == data_K


def get_scale_output_shape(
    K: int, F: int, Q_TILE_K: int = Q_TILE_K, enable_scale_packing: bool = True
) -> tuple[int, int]:
    """
    Get the expected output shape for scales based on packing mode.

    Args:
        K: K dimension size
        F: F dimension size
        enable_scale_packing: Whether scale packing is enabled

    Returns:
        tuple: (scale_P, scale_F) dimensions
    """
    if enable_scale_packing:
        return calculate_packed_scale_shape(K, F, Q_TILE_K)
    else:
        return (K // INTERLEAVE_FACTOR, F)


def calculate_packed_scale_shape(K: int, F: int, Q_TILE_K: int = Q_TILE_K) -> tuple[int, int]:
    """
    Calculate the packed scale tensor shape.

    Args:
        K (int): K dimension size
        F (int): F dimension size

    Returns:
        tuple[int, int]: (packed_scale_P, F) dimensions

    Notes:
        Rule: Input [P, F] → Scales=[ceil(P/2048) * 128, F]
    """
    num_scale_groups = (
        K + MAX_SCALE_PACKING_P_SIZE - 1
    ) // MAX_SCALE_PACKING_P_SIZE  # ceil(K / MAX_P_PER_SCALING_GROUP)
    packed_scale_P = num_scale_groups * Q_TILE_K
    return packed_scale_P, F


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


def num_scale_packing_groups_spanned(global_k_idx_start, global_k_idx_end):
    scale_packing_group_start, _, _ = get_scale_packing_info(global_k_idx_start)
    scale_packing_group_end, _, _ = get_scale_packing_info(global_k_idx_end)

    return scale_packing_group_end - scale_packing_group_start + 1


def get_remainder_partition_offset(k_idx_within_tile, Q_TILE_K):
    return (k_idx_within_tile // Q_TILE_K) * MX_PARTITION_SIZE
