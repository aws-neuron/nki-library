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

"""Kernel for quantizing BF16 tensors to MXFP8 format with block-wise quantization."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ..mxfp_utils.mxfp8_utils import quantize_mxfp8_block, quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor, TileLocation
from ..mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from ..mxfp_utils.mxfp8_utils.load_apis import load_tile


def should_store_packed_scales(tile_k_idx: int, total_num_tiles: int) -> bool:
    """
    Determine if packed scales should be stored for the current tile.

    Args:
        tile_k_idx (int): Current tile index in K dimension (across all tiles including remainder)
        total_num_tiles (int): Total number of tiles including remainder tiles

    Returns:
        bool: True if scales should be stored, False otherwise
    """
    return (
        tile_k_idx + 1
    ) % quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP == 0 or tile_k_idx + 1 == total_num_tiles


def _process_tile(
    src_td,
    tile_k_size,
    current_tile_f,
    k_offset,
    tile_f_idx,
    L_TILE_F,
    tile_k_idx,
    total_num_tiles,
    INTERLEAVE_FACTOR,
    fp8_dtype,
    return_fp8_dtype,
    quantized_data_hbm,
    quantized_scales_hbm,
    enable_scale_packing,
    quantized_scales=None,
    k_idx_within_tile=0,
):
    """
    Load, quantize, and store a single tile of any size (512, 256, or 128 K-elements).

    Args:
        src_td: Source TensorDescriptor
        tile_k_size: K dimension size for this tile (512, 256, or 128)
        current_tile_f: Current F tile size
        k_offset: Offset in K dimension in the source tensor
        tile_f_idx: Absolute F tile index
        L_TILE_F: F tile size constant
        tile_k_idx: Absolute tile index across all K tiles (for scale packing group calculation)
        total_num_tiles: Total number of K tiles including remainder (for should_store check)
        INTERLEAVE_FACTOR: Interleave factor (4)
        fp8_dtype: FP8 dtype for output
        return_fp8_dtype: FP8 dtype string
        quantized_data_hbm: Output data tensor on HBM
        quantized_scales_hbm: Output scales tensor on HBM
        enable_scale_packing: Whether scale packing is enabled
        quantized_scales: Pre-allocated scales buffer for packing (passed in when continuing a packing group)
        k_idx_within_tile: Offset within the packing group for remainder tiles (e.g., 256 for the 128-tile after a 256-tile)

    Returns:
        quantized_scales: The scales buffer (for passing to next tile in same packing group)
    """

    sbm = get_active_sbm()

    tile_k_interleaved = tile_k_size // INTERLEAVE_FACTOR

    scaling_group_idx, slot_idx, slot_partition_offset = quantize_mxfp8_utils.get_scale_packing_info(
        tile_k_idx, enable_scale_packing
    )

    if enable_scale_packing and slot_idx == 0 and k_idx_within_tile == 0:
        quantized_scales = sbm.alloc_stack(
            shape=(quantize_mxfp8_utils.Q_TILE_K, 1, current_tile_f),
            dtype=nl.float8_e8m0fnu,
            buffer=nl.sbuf,
        )

    # TODO call load and quantize API when its merged
    # Allocate destination SBUF tensor and load
    src_tensor_sbuf = sbm.alloc_stack(
        shape=(tile_k_interleaved, 1, 1, current_tile_f * INTERLEAVE_FACTOR),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )

    load_loc = TileLocation(
        tensor=src_td,
        tile_k=tile_k_size,
        tile_f=current_tile_f,
        k_offset=k_offset,
        f_offset=tile_f_idx * L_TILE_F,
    )
    store_loc = TileLocation(
        tensor=TensorDescriptor(data=src_tensor_sbuf, is_swizzled=True, is_f_by_k=False),
        tile_k=tile_k_size,
        tile_f=current_tile_f,
    )

    load_tile(load_loc, data_store_loc=store_loc)
    # Quantize
    if enable_scale_packing:
        remainder_partition_offset = quantize_mxfp8_utils.get_remainder_partition_offset(
            k_idx_within_tile, quantize_mxfp8_utils.Q_TILE_K
        )
        quantized_data_x4, _ = quantize_mxfp8_block.quantize_mxfp8_block(
            src_tensor_sbuf.reshape((tile_k_interleaved, 1, current_tile_f * INTERLEAVE_FACTOR)),
            (tile_k_interleaved, current_tile_f),
            True,
            return_fp8_dtype,
            quantized_scales,
            slot_partition_offset=slot_partition_offset,
            remainder_partition_offset=remainder_partition_offset,
        )
    else:
        remainder_partition_offset = 0
        quantized_data_x4, quantized_scales = quantize_mxfp8_block.quantize_mxfp8_block(
            src_tensor_sbuf.reshape((tile_k_interleaved, 1, current_tile_f * INTERLEAVE_FACTOR)),
            (tile_k_interleaved, current_tile_f),
            True,
            return_fp8_dtype,
        )

    # Store quantized data - read from remainder_partition_offset since that's where quantize_mx wrote it
    # Offset is flat element offset: partition * elements_per_partition (F dimension is F // INTERLEAVE_FACTOR)
    output_k_start = k_offset // INTERLEAVE_FACTOR
    output_f_start_data = tile_f_idx * L_TILE_F * INTERLEAVE_FACTOR
    nisa.dma_copy(
        dst=quantized_data_hbm[
            nl.ds(output_k_start, tile_k_interleaved),
            nl.ds(output_f_start_data, current_tile_f * INTERLEAVE_FACTOR),
        ],
        src=quantized_data_x4.ap(
            pattern=[
                [current_tile_f * INTERLEAVE_FACTOR, tile_k_interleaved],
                [1, current_tile_f * INTERLEAVE_FACTOR],
            ],
            dtype=fp8_dtype,
            offset=remainder_partition_offset * current_tile_f * INTERLEAVE_FACTOR,
        ),
    )

    # Store scales
    if not enable_scale_packing or should_store_packed_scales(tile_k_idx, total_num_tiles):
        output_f_start_scales = tile_f_idx * L_TILE_F

        if enable_scale_packing:
            output_k_start_scales = scaling_group_idx * quantize_mxfp8_utils.Q_TILE_K
        else:
            output_k_start_scales = output_k_start

        out_k_size = quantized_scales.shape[0]
        nisa.dma_copy(
            dst=quantized_scales_hbm[
                nl.ds(output_k_start_scales, out_k_size),
                nl.ds(output_f_start_scales, current_tile_f),
            ],
            src=quantized_scales[:, 0, :],
        )

    return quantized_scales


@nki.jit
@with_active_sbm
def quantize_block_mxfp8_kernel(
    src_tensor: nl.ndarray,
    return_fp8_dtype: str,
    run_with_lnc2: bool = False,
    enable_scale_packing: bool = True,
) -> tuple[nl.ndarray, nl.ndarray]:
    """
    Kernel for quantizing BF16 tensor to MXFP8 format with block-wise quantization.

    This kernel performs block-wise quantization of BF16 tensors to MXFP8 format,
    producing both quantized data and scaling factors. Supports LNC2 parallelization
    for improved performance on large tensors.
    WARNING!!! This kernel returns quantized scales where the scale values are correct,
    but the padded values are random garbage, not zeros.
    TODO: add follow up CR so that this function can take in a flag to use 0 padding
    by doing memset, which will have worse performance.

    Dimensions:
        F: Feature dimension (rows in input tensor)
        K: K dimension (columns in input tensor)

    Args:
        src_tensor (nl.ndarray): [F, K], Input tensor in BF16 format on HBM
        return_fp8_dtype (str): FP8 dtype string like "float8_e4m3fn" or "float8_e5m2"
        run_with_lnc2 (bool): Enable LNC2 parallelization along F dimension (default: False)
        enable_scale_packing (bool): Enable scale packing optimization (default: True)

    Returns:
        quantized_scales_hbm (nl.ndarray): [K // 4, F], Scales in float8_e8m0fnu format on HBM
        quantized_data_hbm (nl.ndarray): [K // 4, F * INTERLEAVE_FACTOR], Quantized data in FP8 format on HBM

    Notes:
        - K dimension must be divisible by 512 for mxfp8 quantization
        - F dimension must be divisible by 8 for quantization
        - LNC2 splits work along F dimension; supports uneven splits
        - TODO: Add support for splitting along K dimension for LNC2
        - TODO: .view() is not supported for x4 to non-x4 conversion, so BIR ap() is used instead

    Pseudocode:
        # Allocate output tensors
        quantized_data_hbm = allocate([K // 4, F * INTERLEAVE_FACTOR], fp8_dtype)
        quantized_scales_hbm = allocate([K // 4, F], float8_e8m0fnu)

        # Generate vector offsets for DMA gather transpose
        vector_offsets = generate_vector_offsets(K, L_TILE_K, L_TILE_F)

        # Process tiles
        for tile_k_idx in range(NUM_TILES_IN_K):
            for tile_f_idx in range(NUM_TILES_IN_F):
                # Load and interleave tile using DMA gather transpose
                src_tile = dma_transpose_load(src_tensor, tile_k_idx, tile_f_idx, vector_offsets)

                # Quantize the tile
                quantized_data, scales = quantize_mxfp8_block(src_tile, tile_size, return_fp8_dtype)

                # Store quantized data and scales
                store(quantized_data_hbm, quantized_data, tile_k_idx, tile_f_idx)
                store(quantized_scales_hbm, scales, tile_k_idx, tile_f_idx)

        return quantized_scales_hbm, quantized_data_hbm
    """

    if get_active_sbm() is None:
        create_and_set_active_sbm()
        sbm = get_active_sbm()
        sbm.open_scope(name="QUANTIZE KERNEL")

    sbm = get_active_sbm()

    F, K = src_tensor.shape

    L_TILE_K = quantize_mxfp8_utils.L_TILE_K
    L_TILE_F = quantize_mxfp8_utils.L_TILE_F

    kernel_assert(
        K % quantize_mxfp8_utils.MIN_K_FOR_QUANTIZATION == 0,
        f"K dimension must be divisible by {quantize_mxfp8_utils.MIN_K_FOR_QUANTIZATION} for mxfp8 quantization, got {K=}",
    )
    kernel_assert(
        F % quantize_mxfp8_utils.MIN_F_FOR_QUANTIZATION == 0,
        f"F dim must be divisible by {quantize_mxfp8_utils.MIN_F_FOR_QUANTIZATION} for quantization, got {F=}",
    )

    fp8_dtype = quantize_mxfp8_utils.get_fp8_dtype(return_fp8_dtype)
    INTERLEAVE_FACTOR = quantize_mxfp8_utils.INTERLEAVE_FACTOR
    quantized_data_hbm = nl.ndarray(
        (K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR),
        dtype=fp8_dtype,
        buffer=nl.shared_hbm,
    )

    K_scales, F_scales = quantize_mxfp8_utils.get_scale_output_shape(K, F, enable_scale_packing=enable_scale_packing)
    quantized_scales_hbm = nl.ndarray((K_scales, F_scales), dtype=nl.float8_e8m0fnu, buffer=nl.shared_hbm)

    NUM_TILES_IN_F_TOTAL = (F + L_TILE_F - 1) // L_TILE_F

    if run_with_lnc2:
        LNC_ID = nl.program_id(axis=0)
        kernel_assert(
            F >= quantize_mxfp8_utils.MIN_F_FOR_LNC2,
            f"LNC2 requires F >= {quantize_mxfp8_utils.MIN_F_FOR_LNC2}, but got {F=}",
        )
        tiles_for_lnc0 = (NUM_TILES_IN_F_TOTAL + 1) // 2
        tiles_for_lnc1 = NUM_TILES_IN_F_TOTAL - tiles_for_lnc0
        if LNC_ID == 0:
            tile_f_offset = 0
            NUM_TILES_IN_F = tiles_for_lnc0
        else:
            tile_f_offset = tiles_for_lnc0
            NUM_TILES_IN_F = tiles_for_lnc1
    else:
        tile_f_offset = 0
        NUM_TILES_IN_F = NUM_TILES_IN_F_TOTAL

    src_td = TensorDescriptor(data=src_tensor, is_swizzled=False, is_f_by_k=True)
    src_td.set_vector_offset_patterns(L_TILE_K, L_TILE_F)

    NUM_TILES_IN_K = K // L_TILE_K
    REMAINDER_K = K % L_TILE_K
    HAS_REMAINDER_256 = REMAINDER_K >= 256
    HAS_REMAINDER_128 = REMAINDER_K % 256 >= 128
    # Remainder tiles (256 and/or 128) share the same tile_k_idx, so count as 1 logical tile
    HAS_ANY_REMAINDER = HAS_REMAINDER_256 or HAS_REMAINDER_128
    TOTAL_NUM_TILES = NUM_TILES_IN_K + (1 if HAS_ANY_REMAINDER else 0)

    for tile_f_local_idx in nl.affine_range(NUM_TILES_IN_F):
        tile_f_idx = tile_f_local_idx + tile_f_offset
        current_tile_f = min(L_TILE_F, F - tile_f_idx * L_TILE_F)

        quantized_scales = None
        for tile_k_idx in range(NUM_TILES_IN_K):
            quantized_scales = _process_tile(
                src_td,
                L_TILE_K,
                current_tile_f,
                k_offset=tile_k_idx * L_TILE_K,
                tile_f_idx=tile_f_idx,
                L_TILE_F=L_TILE_F,
                tile_k_idx=tile_k_idx,
                total_num_tiles=TOTAL_NUM_TILES,
                INTERLEAVE_FACTOR=INTERLEAVE_FACTOR,
                fp8_dtype=fp8_dtype,
                return_fp8_dtype=return_fp8_dtype,
                quantized_data_hbm=quantized_data_hbm,
                quantized_scales_hbm=quantized_scales_hbm,
                enable_scale_packing=enable_scale_packing,
                quantized_scales=quantized_scales,
            )

        # Handle remainder K (K not divisible by L_TILE_K=512)
        """
        Both 256-tile and 128-tile remainders share the same tile_k_idx (NUM_TILES_IN_K)
        since they together form one logical remainder block for scale packing.
        """
        remainder_k_offset = NUM_TILES_IN_K * L_TILE_K

        if HAS_REMAINDER_256:
            quantized_scales = _process_tile(
                src_td,
                256,
                current_tile_f,
                k_offset=remainder_k_offset,
                tile_f_idx=tile_f_idx,
                L_TILE_F=L_TILE_F,
                tile_k_idx=NUM_TILES_IN_K,
                total_num_tiles=TOTAL_NUM_TILES,
                INTERLEAVE_FACTOR=INTERLEAVE_FACTOR,
                fp8_dtype=fp8_dtype,
                return_fp8_dtype=return_fp8_dtype,
                quantized_data_hbm=quantized_data_hbm,
                quantized_scales_hbm=quantized_scales_hbm,
                enable_scale_packing=enable_scale_packing,
                quantized_scales=quantized_scales,
            )
            remainder_k_offset += 256

        if HAS_REMAINDER_128:
            k_idx_within_tile = 256 if HAS_REMAINDER_256 else 0
            quantized_scales = _process_tile(
                src_td,
                128,
                current_tile_f,
                k_offset=remainder_k_offset,
                tile_f_idx=tile_f_idx,
                L_TILE_F=L_TILE_F,
                tile_k_idx=NUM_TILES_IN_K,
                total_num_tiles=TOTAL_NUM_TILES,
                INTERLEAVE_FACTOR=INTERLEAVE_FACTOR,
                fp8_dtype=fp8_dtype,
                return_fp8_dtype=return_fp8_dtype,
                quantized_data_hbm=quantized_data_hbm,
                quantized_scales_hbm=quantized_scales_hbm,
                enable_scale_packing=enable_scale_packing,
                quantized_scales=quantized_scales,
                k_idx_within_tile=k_idx_within_tile,
            )

    return quantized_scales_hbm, quantized_data_hbm
