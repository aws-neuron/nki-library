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

"""Generic MXFP8 matmul API that factors out the block-processing logic from the monolithic
matmul_mxfp8 kernel. Callers control the outer M/N/K loops and can keep results in SBUF
for fused element-wise operations, or write directly to HBM."""

from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ..mxfp_utils.mxfp8_utils import load_block, quantize_mxfp8_block, quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils.common_dataclasses import BlockDescriptor, TensorDescriptor
from ..mxfp_utils.mxfp8_utils.common_utils import get_active_sbm
from . import matmul_mxfp8_blocks
from .matmul_mxfp8_config import MatmulMxfp8KernelConfig


def spill_block(
    quantized_data_hbm: nl.ndarray,
    quantized_scales_hbm: nl.ndarray,
    quantized_data_sbuf: nl.ndarray,
    quantized_scales_sbuf: nl.ndarray,
    block_idx_k: int,
    block_idx_f: int,
    BLOCK_K_PHYSICAL: int,
    BLOCK_F: int,
    enable_scale_packing: bool = False,
    NUM_TILES_IN_TENSOR_K: int = 0,
) -> None:
    """
    Spill a quantized block from SBUF to HBM scratch buffer.

    The SBUF data is always in x4 format (output of quantize_mxfp8_block).
    spill_format is 'x4', data is written directly.

    Dimensions:
        TILE_K: Tile size in K dimension.
        NUM_TILES_K: Number of K tiles in the block.
        F_x4: F dimension in x4 packed elements.

    Args:
        quantized_data_hbm (nl.ndarray): Quantized data in HBM (x4 format).
        quantized_scales_hbm (nl.ndarray): Quantized scales in HBM.
        quantized_data_sbuf (nl.ndarray): Quantized data in SBUF (x4 format),
            shape (TILE_K, NUM_TILES_K, F_x4).
        quantized_scales_sbuf (nl.ndarray): Scales in SBUF,
            shape (TILE_K, NUM_TILES_K, F_x4).
        block_idx_k (int): K block index (determines row position in spill buffer).
        block_idx_f (int): M or N block index (determines column position).
        BLOCK_K_PHYSICAL (int): Physical block size in K dimension.
        BLOCK_F (int): Block size in F dimension (M_LOGICAL for lhs, N_LOGICAL for rhs).
        enable_scale_packing (bool): Whether scale packing is enabled (default: False).
        NUM_TILES_IN_TENSOR_K (int): Number of tiles in K dimension for the entire tensor.

    Returns:
        None

    Notes:
        - TODO: Spill/reload should depend on tensor sizes; skip if entire tensor fits in SBUF
        - TODO: Skip spilling blocks that will not be reloaded (e.g., last block still in use)
        - TODO: Coalesce data and scales to improve spill/reload efficiency
        - TODO: Update scale spill/reload to only transfer valid data
            (4/32 partitions are valid, rest is junk)
        - TODO: Investigate whether spilling a single block underutilizes DMA bandwidth
        - TODO: Load tile shape calculations assume pre-swizzled BF16 inputs;
            update when this assumption changes
        - TODO: Avoid HBM bank conflicts when both cores load the same tensor
        - TODO: Update spill/reload logic once quantized data is written as
            [TILE_K, NUM_TILES_K, F] to reduce number of loads

    Pseudocode:
        for tile_k in range(NUM_TILES_K):
            DMA copy data tile from SBUF to HBM at (k_start + tile_k*TILE_K, f_start)
            if enable_scale_packing:
                if first tile in scaling group:
                    DMA copy entire scaling group from SBUF to HBM
            else:
                DMA copy scales from SBUF to HBM
    """

    TILE_K = quantized_data_sbuf.shape[0]
    NUM_TILES_K = quantized_data_sbuf.shape[1]
    F_x4 = quantized_data_sbuf.shape[2]  # F in x4 elements

    # Calculate HBM destination region
    k_start = block_idx_k * BLOCK_K_PHYSICAL
    f_start_scales = block_idx_f * BLOCK_F

    # BIR AP parameters for reading from 3D SBUF tensor
    step_p = NUM_TILES_K * F_x4

    # Iterate over K tiles within the block
    for tile_k in range(NUM_TILES_K):
        k_hbm = k_start + tile_k * TILE_K
        sbuf_offset = tile_k * F_x4

        # x4 spill: direct copy, same dtype
        nisa.dma_copy(
            dst=quantized_data_hbm[nl.ds(k_hbm, TILE_K), nl.ds(f_start_scales, F_x4)],
            src=quantized_data_sbuf.ap(pattern=[[step_p, TILE_K], [1, F_x4]], offset=sbuf_offset),
        )

        if enable_scale_packing:
            global_tile_k_index = block_idx_k * NUM_TILES_K + tile_k

            global_scaling_group_idx, global_slot_idx, global_partition_offset = (
                quantize_mxfp8_utils.get_scale_packing_info(global_tile_k_index)
            )

            tile_scaling_group_idx, tile_slot_idx, tile_partition_offset = quantize_mxfp8_utils.get_scale_packing_info(
                tile_k
            )

            NUM_BLOCK_SCALING_GROUPS = quantized_scales_sbuf.shape[1]
            step_p_sbuf = NUM_BLOCK_SCALING_GROUPS * F_x4

            f_start_scales = block_idx_f * F_x4

            """
            Copy this tile's scale data at each MX_PARTITION_SIZE-spaced offset.
            Each copy transfers MAX_TILES_PER_SCALE_PACKING_GROUP (4) rows.
            Stop when the offset would exceed TILE_K.
            """
            for scale_offset_idx in range(TILE_K // quantize_mxfp8_utils.MX_PARTITION_SIZE):
                partition = global_partition_offset + scale_offset_idx * quantize_mxfp8_utils.MX_PARTITION_SIZE
                if partition + quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP > TILE_K:
                    break

                sbuf_partition = tile_partition_offset + scale_offset_idx * quantize_mxfp8_utils.MX_PARTITION_SIZE
                hbm_offset = global_scaling_group_idx * TILE_K + partition
                sbuf_offset = tile_scaling_group_idx * F_x4 + sbuf_partition * step_p_sbuf

                nisa.dma_copy(
                    dst=quantized_scales_hbm[
                        nl.ds(hbm_offset, quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP),
                        nl.ds(f_start_scales, F_x4),
                    ],
                    src=quantized_scales_sbuf.ap(
                        pattern=[
                            [step_p_sbuf, quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP],
                            [1, F_x4],
                        ],
                        offset=sbuf_offset,
                    ),
                )

        else:
            # Scales are always the same format (float8_e8m0fnu, one per x4 group)
            nisa.dma_copy(
                dst=quantized_scales_hbm[nl.ds(k_hbm, TILE_K), nl.ds(f_start_scales, F_x4)],
                src=quantized_scales_sbuf.ap(pattern=[[step_p, TILE_K], [1, F_x4]], offset=sbuf_offset),
            )


def _compute_spill_reload_tile_shapes(
    lhs_td: TensorDescriptor,
    rhs_td: TensorDescriptor,
    lhs_matmul_tile_shape_physical: tuple,
    rhs_matmul_tile_shape_physical: tuple,
    lhs_load_tile_shape: tuple,
    rhs_load_tile_shape: tuple,
) -> Tuple[tuple, tuple]:
    """Compute load tile shapes for spill-reloaded (quantized x4) data."""
    lhsq_load_tile_shape = (
        lhs_matmul_tile_shape_physical
        if not lhs_td.is_swizzled
        else (lhs_load_tile_shape[0], lhs_load_tile_shape[1] // quantize_mxfp8_utils.INTERLEAVE_FACTOR)
    )
    rhsq_load_tile_shape = (
        rhs_matmul_tile_shape_physical
        if not rhs_td.is_swizzled
        else (rhs_load_tile_shape[0], rhs_load_tile_shape[1] // quantize_mxfp8_utils.INTERLEAVE_FACTOR)
    )
    return lhsq_load_tile_shape, rhsq_load_tile_shape


def _store_output_block_to_hbm(
    output_sbuf: nl.ndarray,
    output_tensor_hbm: nl.ndarray,
    block_idx_m: int,
    block_idx_n: int,
    bd: BlockDescriptor,
    LHS_MATMUL_TILE_M: int,
    M_LOGICAL: int,
    N_LOGICAL: int,
) -> None:
    """DMA copy one (m, n) block from the 3D SBUF accumulator to HBM.

    Note: If SBUF dtype is FP32 and output_dtype is BF16, the DMA engine
    will automatically cast FP32 to BF16 during the copy.
    """
    for tile_idx_m in range(bd.TILES_IN_BLOCK_M):
        output_idx_m = block_idx_m * bd.BLOCK_M_LOGICAL + tile_idx_m * LHS_MATMUL_TILE_M
        out_idx_n = block_idx_n * bd.BLOCK_N_LOGICAL

        size_m = min(LHS_MATMUL_TILE_M, M_LOGICAL - output_idx_m)
        size_n = min(bd.BLOCK_N_LOGICAL, N_LOGICAL - out_idx_n)

        if size_m > 0 and size_n > 0:
            sbuf_step_p = bd.TILES_IN_BLOCK_M * bd.BLOCK_N_LOGICAL
            sbuf_offset = tile_idx_m * bd.BLOCK_N_LOGICAL
            nisa.dma_copy(
                dst=output_tensor_hbm[nl.ds(output_idx_m, size_m), nl.ds(out_idx_n, size_n)],
                src=output_sbuf.ap(pattern=[[sbuf_step_p, size_m], [1, size_n]], offset=sbuf_offset),
            )


def extract_from_sbuf_td(sbuf_td: TensorDescriptor) -> tuple:
    """Extract (loaded, data_loaded, scales_loaded) from a pre-loaded sbuf_td."""
    if sbuf_td.is_quantized:
        return None, sbuf_td.data, sbuf_td.scales
    return sbuf_td.data, None, None


def _quantize_and_spill_operand(
    loaded: nl.ndarray,
    quantize_tile_shape: tuple,
    float8_dtype: str,
    use_scale_packing: bool,
    TILES_IN_BLOCK_K: int,
    spill_reload: bool,
    spill_td: TensorDescriptor,
    idx_k: int,
    idx_f: int,
    BLOCK_K_LOGICAL: int,
    BLOCK_F_LOGICAL: int,
    NUM_TILES_IN_K: int,
    sbuf_td: TensorDescriptor,
    sbuf_has_data: bool,
) -> tuple:
    """Quantize a loaded operand and optionally spill to HBM.

    Returns:
        (data_loaded, scales_loaded) after quantization.
    """
    sbm = get_active_sbm()

    if use_scale_packing:
        scales_loaded = sbm.alloc_stack(
            shape=(
                quantize_tile_shape[0],
                (TILES_IN_BLOCK_K + quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP - 1)
                // quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP,
                loaded.shape[2] // quantize_mxfp8_utils.INTERLEAVE_FACTOR,
            ),
            dtype=nl.float8_e8m0fnu,
            buffer=nl.sbuf,
        )
        data_loaded, _ = quantize_mxfp8_block.quantize_mxfp8_block(
            loaded,
            quantize_tile_shape,
            True,
            float8_dtype,
            scales_loaded,
            slot_partition_offset=0,
            enable_scale_packing=use_scale_packing,
        )
    else:
        data_loaded, scales_loaded = quantize_mxfp8_block.quantize_mxfp8_block(
            loaded, quantize_tile_shape, True, float8_dtype
        )

    if spill_reload:
        spill_block(
            spill_td.data,
            spill_td.scales,
            data_loaded,
            scales_loaded,
            idx_k,
            idx_f,
            BLOCK_K_LOGICAL // quantize_mxfp8_utils.INTERLEAVE_FACTOR,
            BLOCK_F_LOGICAL,
            use_scale_packing,
            NUM_TILES_IN_K,
        )

    if sbuf_td != None and not sbuf_has_data:
        sbuf_td.data = data_loaded
        sbuf_td.scales = scales_loaded
        sbuf_td.is_quantized = True

    return data_loaded, scales_loaded


def generic_matmul_mxfp8_api(
    lhs_hbm_td: TensorDescriptor,
    rhs_hbm_td: TensorDescriptor,
    # --- Config (preferred) --- #
    config: MatmulMxfp8KernelConfig = None,
    # --- Legacy flat params (used if config is None) --- #
    bd: BlockDescriptor = None,
    # --- Output --- #
    output_td: TensorDescriptor = None,
    output_dtype=nl.float32,
    # --- Tile Loading Config --- #
    TILES_IN_LOAD_M: int = None,
    TILES_IN_LOAD_N: int = None,
    # --- Block Location --- #
    block_idx_m: Optional[Tuple[int, int]] = None,
    block_idx_n: Optional[Tuple[int, int]] = None,
    block_idx_k: Optional[Tuple[int, int]] = None,
    # --- Tile shapes --- #
    lhs_matmul_tile_shape_physical: tuple = None,
    rhs_matmul_tile_shape_physical: tuple = None,
    lhs_load_tile_shape: tuple = None,
    rhs_load_tile_shape: tuple = None,
    lhs_quantize_tile_shape: tuple = None,
    rhs_quantize_tile_shape: tuple = None,
    # --- Flags --- #
    tile_loop_order: str = 'mnk',
    float8_dtype: str = "float8_e4m3fn",
    use_scale_packing: bool = False,
    # --- Spill Reload --- #
    spill_reload: bool = False,
    lhsq_td: Optional[TensorDescriptor] = None,
    rhsq_td: Optional[TensorDescriptor] = None,
    # --- Pre-loaded SBUF tensors (skip loading if provided) --- #
    lhs_sbuf_td: Optional[TensorDescriptor] = None,
    rhs_sbuf_td: Optional[TensorDescriptor] = None,
    # --- LNC2 offset --- #
    rhs_n_offset: int = 0,
    lhs_m_offset: int = 0,
    # --- Accumulation control --- #
    initialize_accumulator: bool = True,
) -> None:
    """
    Generic MXFP8 matmul API that processes a configurable range of M/N/K blocks.

    This function encapsulates the load → quantize → spill → matmul pipeline and
    gives the caller control over:
      - Which block ranges to process (block_idx_m/n/k)
      - Where to store the result (SBUF via output_td for fusion, or HBM)
      - Whether to skip loading via pre-loaded SBUF tensors (lhs_sbuf_td / rhs_sbuf_td)

    Dimensions:
        M: Number of rows in LHS matrix (output rows)
        K: Contraction dimension
        N: Number of columns in RHS matrix (output columns)

    Args:
        lhs_hbm_td: LHS tensor descriptor in HBM (BF16 or prequantized MXFP8).
        rhs_hbm_td: RHS tensor descriptor in HBM (BF16 or prequantized MXFP8).
        bd: BlockDescriptor defining tile/block geometry.
        output_td: Destination tensor descriptor. If output_td.data is in nl.sbuf,
            the result stays in SBUF for the caller to use. If in HBM, the API
            DMA-copies the result out after K accumulation.
        output_dtype: Data type for the SBUF accumulator (default: nl.float32).
        TILES_IN_LOAD_M: Tiles loaded per DMA in M. None = TILES_IN_BLOCK_M.
        TILES_IN_LOAD_N: Tiles loaded per DMA in N. None = TILES_IN_BLOCK_N.
        block_idx_m: (start, end) range of M blocks. None = all.
        block_idx_n: (start, end) range of N blocks. None = all.
        block_idx_k: (start, end) range of K blocks. None = all.
        lhs_matmul_tile_shape_physical: (TILE_K, TILE_M) physical tile for LHS matmul.
        rhs_matmul_tile_shape_physical: (TILE_K, TILE_N) physical tile for RHS matmul.
        lhs_load_tile_shape: Load tile shape for LHS.
        rhs_load_tile_shape: Load tile shape for RHS.
        lhs_quantize_tile_shape: Quantize tile shape for LHS.
        rhs_quantize_tile_shape: Quantize tile shape for RHS.
        tile_loop_order: Tile loop order within matmul_mxfp8_blocks ('mnk', 'nmk', 'mkn').
        float8_dtype: FP8 dtype for quantization.
        use_scale_packing: Enable scale packing.
        spill_reload: Spill quantized blocks to HBM for reuse.
        lhsq_td: Pre-allocated spill/reload buffer for LHS. Required if spill_reload=True.
        rhsq_td: Pre-allocated spill/reload buffer for RHS. Required if spill_reload=True.
        lhs_sbuf_td: If provided, skip LHS loading — use this pre-loaded SBUF data
            (already swizzled and quantized).
        rhs_sbuf_td: If provided, skip RHS loading — use this pre-loaded SBUF data
            (already swizzled and quantized).
        rhs_n_offset: Additional N offset for RHS (used for LNC2 N-sharding).
        lhs_m_offset: Additional M offset for LHS (used for LNC2 M-sharding).
        initialize_accumulator: If True (default), the first K-block zeroes the SBUF
            accumulator before writing. If False, all K-blocks accumulate (+=) into
            the existing SBUF contents. Set to False when multiple API calls must
            sum their results into the same SBUF output (e.g. d_gate @ W_gate + d_up @ W_up).

    Returns:
        None. Results are written to output_td.data (SBUF or HBM depending on buffer type).

    Pseudocode:
        for idx_m in range(m_start, m_end):
            for idx_n in range(n_start, n_end):
                allocate output_sbuf
                for idx_k in range(k_start, k_end):
                    load lhs and rhs from HBM (or use pre-loaded SBUF)
                    quantize lhs/rhs if BF16
                    optionally spill quantized blocks to HBM
                    matmul_mxfp8_blocks(output_sbuf, lhs, rhs, accumulate=(idx_k > k_start))
                if output is HBM: DMA copy output_sbuf to HBM
    """

    sbm = get_active_sbm()

    # Resolve params: prefer config if provided, fall back to flat params
    if config is not None:
        bd = bd or config.bd
        output_dtype = output_dtype if output_dtype != nl.float32 else (config.output_dtype or nl.float32)
        TILES_IN_LOAD_M = TILES_IN_LOAD_M or config.TILES_IN_LOAD_M
        TILES_IN_LOAD_N = TILES_IN_LOAD_N or config.TILES_IN_LOAD_N
        lhs_matmul_tile_shape_physical = lhs_matmul_tile_shape_physical or config.lhs_matmul_tile_shape_physical
        rhs_matmul_tile_shape_physical = rhs_matmul_tile_shape_physical or config.rhs_matmul_tile_shape_physical
        lhs_load_tile_shape = lhs_load_tile_shape or config.lhs_load_tile_shape
        rhs_load_tile_shape = rhs_load_tile_shape or config.rhs_load_tile_shape
        lhs_quantize_tile_shape = lhs_quantize_tile_shape or config.lhs_quantize_tile_shape
        rhs_quantize_tile_shape = rhs_quantize_tile_shape or config.rhs_quantize_tile_shape
        tile_loop_order = tile_loop_order or config.tile_loop_order
        float8_dtype = float8_dtype or config.float8_dtype
        use_scale_packing = use_scale_packing or config.enable_scale_packing
        spill_reload = spill_reload or config.spill_reload

    kernel_assert(lhs_matmul_tile_shape_physical != None, "lhs_matmul_tile_shape_physical is required")

    MATMUL_TILE_K, LHS_MATMUL_TILE_M = lhs_matmul_tile_shape_physical

    # Derive dimensions from tensor descriptors
    K_PHYSICAL_LHS, _ = lhs_hbm_td.physical_shape
    if not lhs_hbm_td.is_swizzled:
        K_PHYSICAL_LHS //= quantize_mxfp8_utils.INTERLEAVE_FACTOR
    NUM_TILES_IN_K = div_ceil(K_PHYSICAL_LHS, MATMUL_TILE_K)

    _, M_LOGICAL = lhs_hbm_td.sharded_logical_shape
    _, N_LOGICAL = rhs_hbm_td.sharded_logical_shape
    K_LOGICAL = lhs_hbm_td.logical_shape[0]

    # Default load tile counts
    if TILES_IN_LOAD_M == None:
        TILES_IN_LOAD_M = bd.TILES_IN_BLOCK_M
    if TILES_IN_LOAD_N == None:
        TILES_IN_LOAD_N = bd.TILES_IN_BLOCK_N

    # Compute block counts
    BLOCKS_IN_M = div_ceil(M_LOGICAL, bd.BLOCK_M_LOGICAL)
    BLOCKS_IN_N = div_ceil(N_LOGICAL, bd.BLOCK_N_LOGICAL)
    BLOCKS_IN_K = div_ceil(K_LOGICAL, bd.BLOCK_K_LOGICAL)

    # Resolve block ranges
    m_start, m_end = block_idx_m if block_idx_m != None else (0, BLOCKS_IN_M)
    n_start, n_end = block_idx_n if block_idx_n != None else (0, BLOCKS_IN_N)
    k_start, k_end = block_idx_k if block_idx_k != None else (0, BLOCKS_IN_K)

    lhsq_load_tile_shape, rhsq_load_tile_shape = _compute_spill_reload_tile_shapes(
        lhs_hbm_td,
        rhs_hbm_td,
        lhs_matmul_tile_shape_physical,
        rhs_matmul_tile_shape_physical,
        lhs_load_tile_shape,
        rhs_load_tile_shape,
    )

    # Determine if output is HBM (need to store) or SBUF (caller manages)
    output_is_hbm = output_td.data.buffer != nl.sbuf

    # Use FP32 accumulator in SBUF when multiple K blocks and BF16 output
    # to preserve precision during cross-K-block accumulation.
    # The cast to BF16 happens during the final DMA store to HBM.
    # TODO: If we allow for PSUM Accumulation across multiple K blocks, then update this logic
    num_k_blocks = k_end - k_start
    sbuf_dtype = nl.float32 if (output_dtype == nl.bfloat16 and num_k_blocks > 1) else output_dtype

    for idx_m in range(m_start, m_end):
        for idx_n in range(n_start, n_end):
            # Allocate SBUF accumulator
            if output_is_hbm:
                output_sbuf = sbm.alloc_stack(
                    shape=(LHS_MATMUL_TILE_M, bd.TILES_IN_BLOCK_M, bd.BLOCK_N_LOGICAL),
                    dtype=sbuf_dtype,
                    buffer=nl.sbuf,
                )
            else:
                # Caller provided SBUF output — use it directly
                output_sbuf = output_td.data

            for idx_k in range(k_start, k_end):
                # --- Determine effective TDs (original vs spill-reload) ---
                lhs_load_from_hbm = idx_n == 0 or (not spill_reload) or lhs_hbm_td.is_quantized
                rhs_load_from_hbm = idx_m == 0 or (not spill_reload) or rhs_hbm_td.is_quantized

                eff_lhs_td = lhs_hbm_td if lhs_load_from_hbm else lhsq_td
                eff_rhs_td = rhs_hbm_td if rhs_load_from_hbm else rhsq_td

                spill_reload_bd = bd.get_spill_reload_bd(
                    lhs_hbm_td.is_swizzled,
                    rhs_hbm_td.is_swizzled,
                    lhs_load_from_hbm,
                    rhs_load_from_hbm,
                )

                # --- Load (skip if sbuf_td provided and non-empty) ---
                lhs_sbuf_has_data = lhs_sbuf_td != None and lhs_sbuf_td.data != None
                rhs_sbuf_has_data = rhs_sbuf_td != None and rhs_sbuf_td.data != None
                load_lhs_td = None if lhs_sbuf_has_data else eff_lhs_td
                load_rhs_td = None if rhs_sbuf_has_data else eff_rhs_td

                (
                    lhs_loaded,
                    rhs_loaded,
                    lhs_data_loaded,
                    lhs_scales_loaded,
                    rhs_data_loaded,
                    rhs_scales_loaded,
                ) = (None, None, None, None, None, None)

                # if at least one operand needs to be loaded, load
                if load_lhs_td != None or load_rhs_td != None:
                    (
                        lhs_loaded,
                        rhs_loaded,
                        lhs_data_loaded,
                        lhs_scales_loaded,
                        rhs_data_loaded,
                        rhs_scales_loaded,
                    ) = load_block.load_lhs_and_rhs(
                        lhs_td=load_lhs_td,
                        rhs_td=load_rhs_td,
                        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
                        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
                        lhs_load_tile_shape=lhs_load_tile_shape if lhs_load_from_hbm else lhsq_load_tile_shape,
                        rhs_load_tile_shape=rhs_load_tile_shape if rhs_load_from_hbm else rhsq_load_tile_shape,
                        block_idx=(idx_m, idx_k, idx_n),
                        bd=spill_reload_bd,
                        rhs_n_offset=rhs_n_offset if rhs_load_from_hbm else 0,
                        lhs_m_offset=lhs_m_offset if lhs_load_from_hbm else 0,
                    )

                if lhs_sbuf_has_data:
                    lhs_loaded, lhs_data_loaded, lhs_scales_loaded = extract_from_sbuf_td(lhs_sbuf_td)
                if rhs_sbuf_has_data:
                    rhs_loaded, rhs_data_loaded, rhs_scales_loaded = extract_from_sbuf_td(rhs_sbuf_td)

                # --- Quantize LHS if needed ---
                if lhs_loaded != None:
                    lhs_data_loaded, lhs_scales_loaded = _quantize_and_spill_operand(
                        loaded=lhs_loaded,
                        quantize_tile_shape=lhs_quantize_tile_shape,
                        float8_dtype=float8_dtype,
                        use_scale_packing=use_scale_packing,
                        TILES_IN_BLOCK_K=bd.TILES_IN_BLOCK_K,
                        spill_reload=spill_reload,
                        spill_td=lhsq_td,
                        idx_k=idx_k,
                        idx_f=idx_m,
                        BLOCK_K_LOGICAL=bd.BLOCK_K_LOGICAL,
                        BLOCK_F_LOGICAL=bd.BLOCK_M_LOGICAL,
                        NUM_TILES_IN_K=NUM_TILES_IN_K,
                        sbuf_td=lhs_sbuf_td,
                        sbuf_has_data=lhs_sbuf_has_data,
                    )

                # --- Quantize RHS if needed ---
                if rhs_loaded != None:
                    rhs_data_loaded, rhs_scales_loaded = _quantize_and_spill_operand(
                        loaded=rhs_loaded,
                        quantize_tile_shape=rhs_quantize_tile_shape,
                        float8_dtype=float8_dtype,
                        use_scale_packing=use_scale_packing,
                        TILES_IN_BLOCK_K=bd.TILES_IN_BLOCK_K,
                        spill_reload=spill_reload,
                        spill_td=rhsq_td,
                        idx_k=idx_k,
                        idx_f=idx_n,
                        BLOCK_K_LOGICAL=bd.BLOCK_K_LOGICAL,
                        BLOCK_F_LOGICAL=bd.BLOCK_N_LOGICAL,
                        NUM_TILES_IN_K=NUM_TILES_IN_K,
                        sbuf_td=rhs_sbuf_td,
                        sbuf_has_data=rhs_sbuf_has_data,
                    )

                # --- Tiled matmul ---
                accumulate_output = (not initialize_accumulator) or idx_k > k_start

                m_block_size = min(bd.BLOCK_M_LOGICAL, M_LOGICAL - idx_m * bd.BLOCK_M_LOGICAL)
                n_block_size = min(bd.BLOCK_N_LOGICAL, N_LOGICAL - idx_n * bd.BLOCK_N_LOGICAL)

                global_k_tile_start = idx_k * bd.TILES_IN_BLOCK_K

                matmul_mxfp8_blocks.matmul_mxfp8_blocks(
                    output_tensor=output_sbuf,
                    lhs_td=TensorDescriptor(
                        data=lhs_data_loaded,
                        scales=lhs_scales_loaded,
                        scales_are_packed=eff_lhs_td.scales_are_packed,
                    ),
                    LHS_tile_shape=lhs_matmul_tile_shape_physical,
                    rhs_td=TensorDescriptor(
                        data=rhs_data_loaded,
                        scales=rhs_scales_loaded,
                        scales_are_packed=eff_rhs_td.scales_are_packed,
                    ),
                    RHS_tile_shape=rhs_matmul_tile_shape_physical,
                    loop_order=tile_loop_order,
                    accumulate_output=accumulate_output,
                    m_log_end=m_block_size,
                    n_log_end=n_block_size,
                    enable_scale_packing=use_scale_packing,
                    global_block_tile_k_idx=global_k_tile_start,
                    psum_dtype=output_dtype if bd.TILES_IN_BLOCK_K == 1 and output_dtype != nl.float32 else nl.float32,
                    enable_psum_copy_in=config.enable_psum_copy_in if config else True,
                    inputs_were_prequantized=eff_lhs_td.is_quantized or eff_rhs_td.is_quantized,
                )

            # --- Store to HBM if output is HBM ---
            if output_is_hbm:
                _store_output_block_to_hbm(
                    output_sbuf=output_sbuf,
                    output_tensor_hbm=output_td.data,
                    block_idx_m=idx_m,
                    block_idx_n=idx_n,
                    bd=bd,
                    LHS_MATMUL_TILE_M=LHS_MATMUL_TILE_M,
                    M_LOGICAL=M_LOGICAL,
                    N_LOGICAL=N_LOGICAL,
                )
