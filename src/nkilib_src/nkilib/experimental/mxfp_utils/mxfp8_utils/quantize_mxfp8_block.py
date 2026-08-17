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

"""Block-level MXFP8 quantization operations for tensor tiles."""

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from .common_utils import get_active_sbm
from .mxfp8_wrappers import quantize_mx_x4_wrapper
from .quantize_mxfp8_utils import INTERLEAVE_FACTOR, get_fp8_dtype_x4, get_scale_packing_info


def _quantize_tile(
    src_tensor: nl.ndarray,
    quantized_data: nl.ndarray,
    quantized_scales: nl.ndarray,
    k_index: int,
    f_start: int,
    f_size: int,
    TILE_K: int,
    scale_packing_partition_offset: int,
    enable_scale_packing: bool,
    remainder_partition_offset: int = 0,
) -> None:
    """
    Quantize a single tile by slicing into input tensors and calling quantize_mx_x4_wrapper.

    Args:
        src_tensor (nl.ndarray): Source tensor in SBUF, shape [TILE_K, NUM_TILES_K, F]
        quantized_data (nl.ndarray): Output quantized data buffer
        quantized_scales (nl.ndarray): Output quantization scales buffer
        k_index (int): Tile index in K dimension
        f_start (int): Starting offset in F dimension
        f_size (int): Size of tile in F dimension
        TILE_K (int): Tile size in K dimension
        scale_packing_partition_offset (int): Where in k dimension to start indexing for scale packing
        enable_scale_packing (bool): Whether scale packing is enabled
        remainder_partition_offset (int): Offset for remainder tiles to align data and scales to same quadrant

    Returns:
        None
    """
    f_start_out = f_start // INTERLEAVE_FACTOR
    f_size_out = f_size // INTERLEAVE_FACTOR

    src_ap = src_tensor[nl.ds(0, TILE_K), nl.ds(k_index, 1), nl.ds(f_start, f_size)]

    # Both data and scales must start at the same quadrant offset for quantize_mx
    data_ap = quantized_data[
        nl.ds(remainder_partition_offset, TILE_K), nl.ds(k_index, 1), nl.ds(f_start_out, f_size_out)
    ]

    local_tile_idx = k_index
    if enable_scale_packing:
        local_tile_idx, _, scale_packing_partition_offset = get_scale_packing_info(k_index, True)
    local_partition_offset = remainder_partition_offset + scale_packing_partition_offset

    scale_ap = quantized_scales[
        nl.ds(local_partition_offset, TILE_K - scale_packing_partition_offset),
        nl.ds(local_tile_idx, 1),
        nl.ds(f_start_out, f_size_out),
    ]

    # Quantize using wrapper with BIR AP views directly
    quantize_mx_x4_wrapper(data_ap, scale_ap, src_ap, P=TILE_K, F=f_size)


def quantize_mxfp8_block(
    src_tensor: nl.ndarray,
    tile_shape: tuple[int, int],
    k_first: bool,
    return_fp8_dtype: str,
    quantized_scales: nl.ndarray = None,
    slot_partition_offset: int = 0,
    enable_scale_packing: bool = False,
    remainder_partition_offset: int = 0,
) -> tuple[nl.ndarray, nl.ndarray]:
    """
    Quantize BF16 tensor to MXFP8 format.

    WARNING: This function returns quantized scales where the scale values are correct,
    but the padded values are random garbage, not zeros.
    TODO: add follow up CR so that this function can take in a flag to use 0 padding
    by doing memset, which will have worse performance.

    Args:
        src_tensor (nl.ndarray): BF16 tensor in SBUF, shape [TILE_K, NUM_TILES_K, F]
        tile_shape (tuple[int, int]): Tile shape [TILE_K, TILE_F] for quantization
        k_first (bool): If True, process K dimension first; else process F dimension first
        return_fp8_dtype (str): FP8 dtype string ("float8_e5m2", "float8_e4m3fn", "float4_e2m1fn")
        quantized_scales (nl.ndarray): Optional pre-allocated scales buffer (default: None)
        slot_partition_offset (int): Partition offset for scale packing (default: 0)
        enable_scale_packing (bool): Whether scale packing is enabled (default: False)

    Returns:
        tuple[nl.ndarray, nl.ndarray]: (quantized_data, quantized_scales)
    """

    sbm = get_active_sbm()

    # Handle pre-quantized input
    if isinstance(src_tensor, tuple):
        return src_tensor[0], src_tensor[1]

    # Validate inputs
    kernel_assert(len(src_tensor.shape) == 3, "src_tensor must be 3D")
    kernel_assert(len(tile_shape) == 2, "tile_shape must be 2D")

    src_TILE_K, NUM_TILES_K, F = src_tensor.shape
    TILE_K, TILE_F = tile_shape

    kernel_assert(TILE_K <= nl.tile_size.pmax, f"TILE_K cannot exceed {nl.tile_size.pmax}")
    kernel_assert(src_TILE_K == TILE_K, "Tile K mismatch")
    kernel_assert(F % INTERLEAVE_FACTOR == 0, "F must be divisible by INTERLEAVE_FACTOR")
    kernel_assert(TILE_F % INTERLEAVE_FACTOR == 0, "TILE_F must be divisible by INTERLEAVE_FACTOR")
    kernel_assert(F >= TILE_F, "F must be >= TILE_F")

    return_fp8_x4_dtype = get_fp8_dtype_x4(return_fp8_dtype)

    # Allocate output buffers - size must accommodate remainder_partition_offset for quadrant alignment
    data_k_size = remainder_partition_offset + TILE_K
    quantized_data = sbm.alloc_stack(
        shape=(data_k_size, NUM_TILES_K, F // INTERLEAVE_FACTOR), dtype=return_fp8_x4_dtype, buffer=nl.sbuf
    )

    if quantized_scales == None:
        quantized_scales = sbm.alloc_stack(
            shape=(TILE_K, NUM_TILES_K, F // INTERLEAVE_FACTOR), dtype=nl.float8_e8m0fnu, buffer=nl.sbuf
        )

    NUM_TILES_F = F // TILE_F
    REMAINDER_F = F % TILE_F

    # Process tiles based on loop order
    if k_first:
        # F outer, K inner
        for f_index in range(NUM_TILES_F):
            for k_index in range(NUM_TILES_K):
                _quantize_tile(
                    src_tensor=src_tensor,
                    quantized_data=quantized_data,
                    quantized_scales=quantized_scales,
                    k_index=k_index,
                    f_start=f_index * TILE_F,
                    f_size=TILE_F,
                    TILE_K=TILE_K,
                    scale_packing_partition_offset=slot_partition_offset,
                    enable_scale_packing=enable_scale_packing,
                    remainder_partition_offset=remainder_partition_offset,
                )

        # Handle remainder
        if REMAINDER_F >= INTERLEAVE_FACTOR:
            for k_index in range(NUM_TILES_K):
                _quantize_tile(
                    src_tensor=src_tensor,
                    quantized_data=quantized_data,
                    quantized_scales=quantized_scales,
                    k_index=k_index,
                    f_start=NUM_TILES_F * TILE_F,
                    f_size=REMAINDER_F,
                    TILE_K=TILE_K,
                    scale_packing_partition_offset=slot_partition_offset,
                    enable_scale_packing=enable_scale_packing,
                    remainder_partition_offset=remainder_partition_offset,
                )
    else:
        # K outer, F inner
        for k_index in range(NUM_TILES_K):
            for f_index in range(NUM_TILES_F):
                _quantize_tile(
                    src_tensor=src_tensor,
                    quantized_data=quantized_data,
                    quantized_scales=quantized_scales,
                    k_index=k_index,
                    f_start=f_index * TILE_F,
                    f_size=TILE_F,
                    TILE_K=TILE_K,
                    scale_packing_partition_offset=slot_partition_offset,
                    enable_scale_packing=enable_scale_packing,
                    remainder_partition_offset=remainder_partition_offset,
                )

            # Handle remainder
            if REMAINDER_F >= INTERLEAVE_FACTOR:
                _quantize_tile(
                    src_tensor=src_tensor,
                    quantized_data=quantized_data,
                    quantized_scales=quantized_scales,
                    k_index=k_index,
                    f_start=NUM_TILES_F * TILE_F,
                    f_size=REMAINDER_F,
                    TILE_K=TILE_K,
                    scale_packing_partition_offset=slot_partition_offset,
                    enable_scale_packing=enable_scale_packing,
                    remainder_partition_offset=remainder_partition_offset,
                )

    return quantized_data, quantized_scales
