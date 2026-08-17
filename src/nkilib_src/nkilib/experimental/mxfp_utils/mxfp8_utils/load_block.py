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

"""Helper functions for loading LHS and RHS tensor blocks from HBM to SBUF for MXFP8 matrix multiplication."""

from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ...mlp_mxfp8.common_utils import TILE_M as TILE_128
from . import quantize_mxfp8_utils
from .common_dataclasses import BlockDescriptor, TensorDescriptor, TileLocation
from .common_utils import get_active_sbm
from .load_apis import load_tile


def _load_single_tensor(
    load_td: TensorDescriptor,
    store_td: TensorDescriptor,
    k_global: int,
    f_global: int,
    f_sbuf_start: int,
    tile_idx_k: int,
    LOAD_TILE_K: int,
    LOAD_TILE_F: int,
    BLOCK_F: int,
    f_offset: int = 0,
    global_tile_k_index: int = 0,
) -> None:
    """
    Load a single tensor tile from HBM to SBUF using load_tile API.

    Args:
        load_td: HBM source tensor descriptor.
        store_td: SBUF destination tensor descriptor.
        k_global: Global K dimension offset in HBM.
        f_global: Global F dimension offset in HBM.
        f_sbuf_start: Starting F offset within the current tile group in SBUF.
        tile_idx_k: Tile index in K dimension within the current block.
        LOAD_TILE_K: Tile size in K dimension.
        LOAD_TILE_F: Tile size in F dimension.
        BLOCK_F: Block size in F dimension.
        f_offset: Additional offset for F dimension (used in LNC2 to access correct half of tensor).
        global_tile_k_index: Global tile index in K for scale packing.
    """
    sbm = get_active_sbm()

    if load_td == None:
        return

    if load_td.is_unswizzled_bf16:
        # Unswizzled BF16: use DGT or PE swizzle path (dispatched by load_tile)
        # Detect remainder: if remaining K from k_global is less than LOAD_TILE_K
        f_effective = load_td.effective_f_dim if load_td.effective_f_dim is not None else load_td.logical_shape[1]
        if f_global + f_offset >= f_effective:
            return

        if load_td.uses_pe_swizzle and load_td.indirect_dma_vector_offset is not None:
            indirect_start_col = (f_global + f_offset) // TILE_128
            indirect_num_cols = LOAD_TILE_F // TILE_128
            indirect_vector_offset = load_td.indirect_dma_vector_offset[
                :, indirect_start_col : indirect_start_col + indirect_num_cols
            ]
        else:
            indirect_vector_offset = None

        # logical_shape[0] is the logical K dimension regardless of the physical layout
        # (F-by-K [F, K] vs K-by-F [K, F]); TensorDescriptor encapsulates that mapping.
        remaining_k = load_td.logical_shape[0] - k_global
        is_remainder = remaining_k < LOAD_TILE_K
        INTERLEAVE_FACTOR = quantize_mxfp8_utils.INTERLEAVE_FACTOR

        # guard against loading 0-sized tiles
        if remaining_k <= 0:
            return

        if not is_remainder:
            # Full tile load
            load_loc = TileLocation(
                tensor=load_td,
                tile_k=LOAD_TILE_K,
                tile_f=LOAD_TILE_F,
                k_offset=k_global,
                f_offset=f_global + f_offset,
                vector_offset=indirect_vector_offset,
            )
            data_store_loc = TileLocation(
                tensor=store_td,
                tile_k=LOAD_TILE_K,
                tile_f=LOAD_TILE_F,
                k_offset=tile_idx_k,
                f_offset=f_sbuf_start,
            )
            load_tile(load_loc, data_store_loc=data_store_loc)
        elif load_td.fast_dma_transpose:
            # Fast DMA gathers the whole (32-aligned) remainder in one op — no 256/128
            # decomposition. SBUF is pre-zeroed so any masked K contributes zero.
            load_loc = TileLocation(
                tensor=load_td,
                tile_k=remaining_k,
                tile_f=LOAD_TILE_F,
                k_offset=k_global,
                f_offset=f_global + f_offset,
            )
            data_store_loc = TileLocation(
                tensor=store_td,
                tile_k=remaining_k,
                tile_f=LOAD_TILE_F,
                k_offset=tile_idx_k,
                f_offset=f_sbuf_start,
            )
            load_tile(load_loc, data_store_loc=data_store_loc)
        else:
            # Remainder tile: decompose into 256 and/or 128 sub-tiles
            k_offset = k_global
            p_offset = 0

            if remaining_k >= 256:
                load_loc = TileLocation(
                    tensor=load_td,
                    tile_k=256,
                    tile_f=LOAD_TILE_F,
                    k_offset=k_offset,
                    f_offset=f_global + f_offset,
                    vector_offset=indirect_vector_offset,
                )
                data_store_loc = TileLocation(
                    tensor=store_td,
                    tile_k=256,
                    tile_f=LOAD_TILE_F,
                    k_offset=tile_idx_k,
                    f_offset=f_sbuf_start,
                )
                load_tile(load_loc, data_store_loc=data_store_loc)
                k_offset += 256
                remaining_k -= 256
                p_offset = 256 // INTERLEAVE_FACTOR

            if remaining_k >= 128:
                load_loc = TileLocation(
                    tensor=load_td,
                    tile_k=128,
                    tile_f=LOAD_TILE_F,
                    k_offset=k_offset,
                    f_offset=f_global + f_offset,
                    vector_offset=indirect_vector_offset,
                )
                # TODO: Remove if else, just call load_tile with p_offset, no need for extra tensor copy. Need compiler / ucode bug fix
                if p_offset > 0:
                    # Workaround: p_offset in dma_transpose not writing correctly.
                    # Load into a temp SBUF, then tensor_copy into the correct partition offset.
                    physical_f = LOAD_TILE_F * INTERLEAVE_FACTOR
                    tmp_sbuf = sbm.alloc_stack(shape=(32, 1, 1, physical_f), dtype=nl.bfloat16, buffer=nl.sbuf)
                    tmp_store_td = TensorDescriptor(data=tmp_sbuf, is_swizzled=True)
                    tmp_store_loc = TileLocation(
                        tensor=tmp_store_td,
                        tile_k=128,
                        tile_f=LOAD_TILE_F,
                        k_offset=0,
                        f_offset=0,
                    )
                    load_tile(load_loc, data_store_loc=tmp_store_loc)
                    # Copy from temp into the actual store at the correct partition offset
                    nisa.tensor_copy(
                        dst=store_td.data[
                            nl.ds(p_offset, 32), tile_idx_k, :, nl.ds(f_sbuf_start * INTERLEAVE_FACTOR, physical_f)
                        ],
                        src=tmp_sbuf[:, 0, :, :],
                    )
                else:
                    data_store_loc = TileLocation(
                        tensor=store_td,
                        tile_k=128,
                        tile_f=LOAD_TILE_F,
                        k_offset=tile_idx_k,
                        f_offset=f_sbuf_start,
                    )
                    load_tile(load_loc, data_store_loc=data_store_loc)
        return

    # Swizzled/quantized path
    # Map 3D SBUF layout (LOAD_TILE_K, LOAD_TILES_IN_K, BLOCK_F) to 2D f_offset
    f_sbuf_offset = tile_idx_k * BLOCK_F + f_sbuf_start

    # Apply f_offset to HBM f_offset for all paths
    f_hbm = f_global + f_offset

    # Build HBM source TileLocation
    load_loc = TileLocation(
        tensor=load_td,
        tile_k=LOAD_TILE_K,
        tile_f=LOAD_TILE_F,
        k_offset=k_global,
        f_offset=f_hbm,
    )

    # Build SBUF data store TileLocation
    data_store_loc = TileLocation(
        tensor=store_td,
        tile_k=LOAD_TILE_K,
        tile_f=LOAD_TILE_F,
        k_offset=0,
        f_offset=f_sbuf_offset,
    )

    if load_td.is_quantized and store_td.scales_are_packed:
        # Load data via load_tile (data only)
        load_tile(load_loc, data_store_loc=data_store_loc)

        """
        Determine per-group stride from the HBM scales tensor.
        Original input scales use Q_TILE_K (128) per group.
        Spill buffer scales use LOAD_TILE_K (physical tile K) per group.
        """
        data_K = load_td.effective_f_dim if load_td.effective_f_dim is not None else load_td.data.shape[0]
        num_tiles = div_ceil(data_K, LOAD_TILE_K)
        num_groups = (
            num_tiles + quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP - 1
        ) // quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP
        expected_q_tile_k_size = num_groups * quantize_mxfp8_utils.Q_TILE_K
        scales_K = (
            load_td.effective_f_dim_scales if load_td.effective_f_dim_scales is not None else load_td.scales.shape[0]
        )
        scale_group_stride = quantize_mxfp8_utils.Q_TILE_K if scales_K == expected_q_tile_k_size else LOAD_TILE_K

        # Handle packed scales: only load at scaling group boundaries
        scaling_group_idx, slot_idx, _ = quantize_mxfp8_utils.get_scale_packing_info(global_tile_k_index)
        if slot_idx == 0 or tile_idx_k == 0:
            block_scaling_group_idx, _, _ = quantize_mxfp8_utils.get_scale_packing_info(
                global_tile_k_index - tile_idx_k
            )
            scale_k_sbuf_start = scaling_group_idx - block_scaling_group_idx
            scale_k_hbm_start = scaling_group_idx * scale_group_stride

            # Build HBM source TileLocation for packed scales
            scale_load_loc = TileLocation(
                tensor=load_td,
                tile_k=LOAD_TILE_K,
                tile_f=LOAD_TILE_F,
                k_offset=scale_k_hbm_start,
                f_offset=f_hbm,
            )

            # Build SBUF destination TileLocation for packed scales
            # Flatten 3D index [0, scale_k_sbuf_start, f_sbuf_start] into f_offset
            scales_last_dim = store_td.scales.shape[-1]
            scale_store_loc = TileLocation(
                tensor=store_td,
                tile_k=LOAD_TILE_K,
                tile_f=LOAD_TILE_F,
                k_offset=0,
                f_offset=scale_k_sbuf_start * scales_last_dim + f_sbuf_start,
            )

            load_tile(scale_load_loc, scale_store_loc=scale_store_loc, load_scales_only=True)
    elif load_td.is_quantized:
        # Build SBUF scale store TileLocation
        scale_store_loc = TileLocation(
            tensor=store_td,
            tile_k=LOAD_TILE_K,
            tile_f=LOAD_TILE_F,
            k_offset=0,
            f_offset=f_sbuf_offset,
        )
        load_tile(load_loc, data_store_loc=data_store_loc, scale_store_loc=scale_store_loc)
    else:
        # bf16 swizzled: data_store_loc covers the full tensor
        load_tile(load_loc, data_store_loc=data_store_loc)


def _allocate_sbuf(
    td: TensorDescriptor,
    LOAD_TILE_K: int,
    LOAD_TILES_IN_K: int,
    F: int,
    block_k_idx: int = 0,
) -> Tuple[Optional[nl.ndarray], Optional[nl.ndarray], Optional[nl.ndarray]]:
    """
    Allocate SBUF arrays for tensor data.

    Args:
        td (TensorDescriptor): Tensor descriptor with data, scales, and metadata flags
        LOAD_TILE_K (int): Tile size in K dimension (logical for unswizzled, physical for swizzled)
        LOAD_TILES_IN_K (int): Number of tiles in K dimension
        F (int): Free dimension size (logical for unswizzled, physical for swizzled)
        block_k_idx (int): Block K index for scale packing, default 0

    Returns:
        Tuple[Optional[nl.ndarray], Optional[nl.ndarray], Optional[nl.ndarray]]: (sbuf, data_sbuf, scales_sbuf)
            where non-quantized returns (sbuf, None, None) and quantized returns (None, data_sbuf, scales_sbuf)
    """

    sbm = get_active_sbm()

    INTERLEAVE_FACTOR = quantize_mxfp8_utils.INTERLEAVE_FACTOR

    if td.is_quantized:
        fp8_dtype_x4 = td.data.dtype if td.is_x4 else quantize_mxfp8_utils.get_fp8_dtype_x4(str(td.data.dtype))
        data_sbuf = sbm.alloc_stack(shape=(LOAD_TILE_K, LOAD_TILES_IN_K, F), dtype=fp8_dtype_x4, buffer=nl.sbuf)

        """
        When scales are packed, the scales SBUF is 1/4 the size in K dimension
        because 4 tiles share one scaling group.
        """
        if td.scales_are_packed:
            """
            Calculate the number of scaling groups needed for LOAD_TILES_IN_K tiles.
            Each scaling group covers 4 tiles.
            """
            num_scale_groups = quantize_mxfp8_utils.num_scale_packing_groups_spanned(
                block_k_idx, block_k_idx + LOAD_TILES_IN_K
            )
            scales_sbuf = sbm.alloc_stack(
                shape=(LOAD_TILE_K, num_scale_groups, F), dtype=td.scales.dtype, buffer=nl.sbuf
            )
        else:
            scales_sbuf = sbm.alloc_stack(
                shape=(LOAD_TILE_K, LOAD_TILES_IN_K, F), dtype=td.scales.dtype, buffer=nl.sbuf
            )
        return None, data_sbuf, scales_sbuf
    else:
        if td.is_swizzled:
            sbuf = sbm.alloc_stack(shape=(LOAD_TILE_K, LOAD_TILES_IN_K, F), dtype=td.data.dtype, buffer=nl.sbuf)
        else:
            physical_k = LOAD_TILE_K // INTERLEAVE_FACTOR
            physical_f = F * INTERLEAVE_FACTOR
            sbuf = sbm.alloc_stack(
                shape=(physical_k, LOAD_TILES_IN_K, 1, physical_f), dtype=td.data.dtype, buffer=nl.sbuf
            )
        return sbuf, None, None


def _zero_sbuf(
    sbuf: Optional[nl.ndarray],
    data_sbuf: Optional[nl.ndarray],
    scales_sbuf: Optional[nl.ndarray],
    is_quantized: bool,
    needs_masking: bool,
) -> None:
    """
    Zero out SBUF buffers when dimensions are not evenly divisible by block sizes.

    Args:
        sbuf (Optional[nl.ndarray]): SBUF buffer for non-quantized tensor.
        data_sbuf (Optional[nl.ndarray]): SBUF buffer for quantized data.
        scales_sbuf (Optional[nl.ndarray]): SBUF buffer for quantized scales.
        is_quantized (bool): Whether the tensor is quantized.
        needs_masking (bool): Whether dimensions require masking.

    Returns:
        None
    """
    if not needs_masking:
        return
    if is_quantized:
        nisa.memset(data_sbuf, value=0)
        nisa.memset(scales_sbuf, value=0)
    else:
        nisa.memset(sbuf, value=0)


def load_lhs_and_rhs(
    lhs_td: Optional[TensorDescriptor] = None,
    rhs_td: Optional[TensorDescriptor] = None,
    TILES_IN_LOAD_M: Optional[int] = None,
    TILES_IN_LOAD_N: Optional[int] = None,
    lhs_load_tile_shape: Optional[Tuple[int, int]] = None,
    rhs_load_tile_shape: Optional[Tuple[int, int]] = None,
    block_idx: Tuple[int, int, int] = (0, 0, 0),
    bd: 'BlockDescriptor' = None,
    loop_order: str = "mnk",
    rhs_n_offset: int = 0,
    lhs_m_offset: int = 0,
) -> Tuple[
    Optional[nl.ndarray],
    Optional[nl.ndarray],
    Optional[nl.ndarray],
    Optional[nl.ndarray],
    Optional[nl.ndarray],
    Optional[nl.ndarray],
]:
    """
    Load LHS and RHS data blocks from HBM to SBUF.

    Loads tensor blocks from HBM into SBUF memory for matrix multiplication operations.
    Supports both quantized (data, scales) and non-quantized tensors, with optional
    masking for non-divisible tensor shapes.

    Args:
        lhs_td (Optional[TensorDescriptor]): LHS tensor descriptor (or None).
        rhs_td (Optional[TensorDescriptor]): RHS tensor descriptor (or None).
        TILES_IN_LOAD_M (Optional[int]): Number of matmul tiles to load at once in M dimension.
        TILES_IN_LOAD_N (Optional[int]): Number of matmul tiles to load at once in N dimension.
        lhs_load_tile_shape (Optional[Tuple[int, int]]): (TILE_K, TILE_M) for LHS matmul operations.
        rhs_load_tile_shape (Optional[Tuple[int, int]]): (TILE_K, TILE_N) for RHS matmul operations.
        block_idx (Tuple[int, int, int]): Block indices (m, k, n).
        bd (BlockDescriptor): Block descriptor with physical block sizes.
        loop_order (str): Loop nesting order.
        rhs_n_offset (int): Additional offset for RHS N dimension (default: 0)
        lhs_m_offset (int): Additional offset for LHS M dimension (default: 0)

    Returns:
        Tuple of (lhs_sbuf, rhs_sbuf, lhs_data_sbuf, lhs_scales_sbuf,
            rhs_data_sbuf, rhs_scales_sbuf).
    """
    # Validate loop order
    if (
        loop_order != "mnk"
        and loop_order != "mkn"
        and loop_order != "nmk"
        and loop_order != "nkm"
        and loop_order != "kmn"
        and loop_order != "knm"
    ):
        loop_order = "mnk"

    # Early return if both are None
    if lhs_td == None and rhs_td == None:
        return None, None, None, None, None, None

    # Unpack parameters
    block_idx_m, block_idx_k, block_idx_n = block_idx
    BLOCK_M = bd.BLOCK_M_PHYSICAL
    BLOCK_K_LHS = bd.BLOCK_K_PHYSICAL_LHS
    BLOCK_K_RHS = bd.BLOCK_K_PHYSICAL_RHS
    BLOCK_N = bd.BLOCK_N_PHYSICAL
    LHS_MATMUL_TILE_K, LHS_MATMUL_TILE_M = lhs_load_tile_shape
    RHS_MATMUL_TILE_K, RHS_MATMUL_TILE_N = rhs_load_tile_shape

    # Calculate load tile sizes
    LHS_LOAD_TILE_K = LHS_MATMUL_TILE_K
    LHS_LOAD_TILE_M = TILES_IN_LOAD_M * LHS_MATMUL_TILE_M
    RHS_LOAD_TILE_K = RHS_MATMUL_TILE_K
    RHS_LOAD_TILE_N = TILES_IN_LOAD_N * RHS_MATMUL_TILE_N

    # For unswizzled inputs, load tile K is logical but block K is physical
    # Convert to same units for validation
    INTERLEAVE_FACTOR = quantize_mxfp8_utils.INTERLEAVE_FACTOR
    lhs_load_k_for_check = (
        LHS_LOAD_TILE_K // INTERLEAVE_FACTOR if (lhs_td != None and not lhs_td.is_swizzled) else LHS_LOAD_TILE_K
    )
    rhs_load_k_for_check = (
        RHS_LOAD_TILE_K // INTERLEAVE_FACTOR if (rhs_td != None and not rhs_td.is_swizzled) else RHS_LOAD_TILE_K
    )

    kernel_assert(
        lhs_load_k_for_check <= BLOCK_K_LHS and LHS_LOAD_TILE_M <= BLOCK_M and RHS_LOAD_TILE_N <= BLOCK_N,
        f"Load tile sizes must fit within block sizes: "
        f"lhs_load_k_for_check={lhs_load_k_for_check} <= BLOCK_K_LHS={BLOCK_K_LHS}, LHS_LOAD_TILE_M={LHS_LOAD_TILE_M} <= BLOCK_M={BLOCK_M}, "
        f"RHS_LOAD_TILE_N={RHS_LOAD_TILE_N} <= BLOCK_N={BLOCK_N}",
    )

    # Calculate offsets (use respective K physical for each tensor)
    block_k_offset_lhs = block_idx_k * BLOCK_K_LHS
    block_k_offset_rhs = block_idx_k * BLOCK_K_RHS
    block_m_offset = block_idx_m * BLOCK_M
    block_n_offset = block_idx_n * BLOCK_N

    # Calculate tile counts (use respective K physical for each tensor)
    LOAD_TILES_IN_K_LHS = BLOCK_K_LHS // LHS_LOAD_TILE_K if lhs_td != None else 1
    LOAD_TILES_IN_K_RHS = BLOCK_K_RHS // RHS_LOAD_TILE_K if rhs_td != None else 1
    LOAD_TILES_IN_M = BLOCK_M // LHS_LOAD_TILE_M if lhs_td != None else 1
    LOAD_TILES_IN_N = BLOCK_N // RHS_LOAD_TILE_N if rhs_td != None else 1

    # Set defaults when only one tensor provided
    if lhs_td == None:
        LOAD_TILES_IN_K_LHS = LOAD_TILES_IN_K_RHS
    if rhs_td == None:
        LOAD_TILES_IN_K_RHS = LOAD_TILES_IN_K_LHS

    # Allocate SBUF buffers
    lhs_sbuf, lhs_data_sbuf, lhs_scales_sbuf = (None, None, None)
    rhs_sbuf, rhs_data_sbuf, rhs_scales_sbuf = (None, None, None)

    # Determine loop ranges
    LOAD_TILES_IN_M_RANGE = LOAD_TILES_IN_M if lhs_td != None else 1
    LOAD_TILES_IN_N_RANGE = LOAD_TILES_IN_N if rhs_td != None else 1
    LOAD_TILES_IN_K_RANGE = LOAD_TILES_IN_K_LHS if lhs_td != None else LOAD_TILES_IN_K_RHS

    # Compute masking locally: needed when physical dimensions aren't divisible by block sizes,
    # or when effective_f_dim causes partial loading in stacked-expert tensors.
    # For swizzled/quantized TDs: effective_f_dim is per-expert K → check vs BLOCK_K.
    # For unswizzled BF16 TDs: effective_f_dim is per-expert F → check vs BLOCK_M/N.
    # TODO: Simplify the below to reuse the effective shapes for F and K
    lhs_eff = lhs_td.effective_f_dim if (lhs_td != None and lhs_td.effective_f_dim is not None) else None
    rhs_eff = rhs_td.effective_f_dim if (rhs_td != None and rhs_td.effective_f_dim is not None) else None
    needs_masking = (
        (BLOCK_K_LHS > 0 and (lhs_td != None and lhs_td.physical_shape[0] % BLOCK_K_LHS != 0))
        or (BLOCK_K_RHS > 0 and (rhs_td != None and rhs_td.physical_shape[0] % BLOCK_K_RHS != 0))
        or (lhs_td != None and lhs_td.physical_shape[1] % BLOCK_M != 0)
        or (rhs_td != None and rhs_td.physical_shape[1] % BLOCK_N != 0)
        or (lhs_eff is not None and lhs_td.is_swizzled and BLOCK_K_LHS > 0 and lhs_eff % BLOCK_K_LHS != 0)
        or (rhs_eff is not None and rhs_td.is_swizzled and BLOCK_K_RHS > 0 and rhs_eff % BLOCK_K_RHS != 0)
        or (lhs_eff is not None and not lhs_td.is_swizzled and BLOCK_M > 0 and lhs_eff % BLOCK_M != 0)
        or (rhs_eff is not None and not rhs_td.is_swizzled and BLOCK_N > 0 and rhs_eff % BLOCK_N != 0)
    )
    if lhs_td != None:
        lhs_sbuf, lhs_data_sbuf, lhs_scales_sbuf = _allocate_sbuf(
            lhs_td,
            LHS_LOAD_TILE_K,
            LOAD_TILES_IN_K_LHS,
            BLOCK_M,
            block_idx_k * LOAD_TILES_IN_K_RANGE,
        )
        _zero_sbuf(lhs_sbuf, lhs_data_sbuf, lhs_scales_sbuf, lhs_td.is_quantized, needs_masking)

        if lhs_td.is_unswizzled_bf16 and not lhs_td.uses_pe_swizzle and not lhs_td.fast_dma_transpose:
            lhs_td.set_vector_offset_patterns(LHS_LOAD_TILE_K, LHS_LOAD_TILE_M)

        # Build SBUF TD for load destination
        lhs_sbuf_td = None
        if not lhs_td.is_quantized:
            lhs_sbuf_td = TensorDescriptor(data=lhs_sbuf, is_swizzled=True)
        else:
            lhs_sbuf_td = TensorDescriptor(
                data=lhs_data_sbuf,
                scales=lhs_scales_sbuf,
                is_swizzled=True,
                is_x4=lhs_td.is_x4,
                scales_are_packed=lhs_td.scales_are_packed,
            )

    if rhs_td != None:
        rhs_sbuf, rhs_data_sbuf, rhs_scales_sbuf = _allocate_sbuf(
            rhs_td,
            RHS_LOAD_TILE_K,
            LOAD_TILES_IN_K_RHS,
            BLOCK_N,
            block_idx_k * LOAD_TILES_IN_K_RANGE,
        )
        _zero_sbuf(rhs_sbuf, rhs_data_sbuf, rhs_scales_sbuf, rhs_td.is_quantized, needs_masking)

        if rhs_td.is_unswizzled_bf16 and not rhs_td.uses_pe_swizzle and not rhs_td.fast_dma_transpose:
            rhs_td.set_vector_offset_patterns(RHS_LOAD_TILE_K, RHS_LOAD_TILE_N)

        # Build SBUF TD for load destination
        rhs_sbuf_td = None
        if not rhs_td.is_quantized:
            rhs_sbuf_td = TensorDescriptor(data=rhs_sbuf, is_swizzled=True)
        else:
            rhs_sbuf_td = TensorDescriptor(
                data=rhs_data_sbuf,
                scales=rhs_scales_sbuf,
                is_swizzled=True,
                is_x4=rhs_td.is_x4,
                scales_are_packed=rhs_td.scales_are_packed,
            )

    """
    Load tiles using simplified loop (mnk order - most common).
    For unswizzled DGT path, k_global must be in logical units (tile_k * LOAD_TILE_K).
    For swizzled path, k_global is in physical units (tile_k * load_k_for_check).
    """
    lhs_k_stride = LHS_LOAD_TILE_K if (lhs_td != None and not lhs_td.is_swizzled) else lhs_load_k_for_check
    rhs_k_stride = RHS_LOAD_TILE_K if (rhs_td != None and not rhs_td.is_swizzled) else rhs_load_k_for_check

    """
    When LHS and RHS have different K tile counts (e.g., one swizzled, one unswizzled),
    we iterate over the max and load each tensor when its K boundary aligns.
    Example: LHS has 8 tiles, RHS has 2 tiles -> load LHS at 0,1,2,3,4,5,6,7 and RHS at 0,4
    """
    if LOAD_TILES_IN_K_LHS == LOAD_TILES_IN_K_RHS:
        max_tiles_in_k = LOAD_TILES_IN_K_LHS
        min_tiles_per_max_tile = 1
        lhs_is_max_tiles = True
    elif LOAD_TILES_IN_K_LHS > LOAD_TILES_IN_K_RHS:
        max_tiles_in_k = LOAD_TILES_IN_K_LHS
        kernel_assert(
            LOAD_TILES_IN_K_LHS % LOAD_TILES_IN_K_RHS == 0,
            f"LOAD_TILES_IN_K_LHS must be divisible by LOAD_TILES_IN_K_RHS, got {LOAD_TILES_IN_K_LHS} and {LOAD_TILES_IN_K_RHS}",
        )
        min_tiles_per_max_tile = LOAD_TILES_IN_K_LHS // LOAD_TILES_IN_K_RHS
        lhs_is_max_tiles = True
    elif LOAD_TILES_IN_K_RHS > LOAD_TILES_IN_K_LHS:
        max_tiles_in_k = LOAD_TILES_IN_K_RHS
        kernel_assert(
            LOAD_TILES_IN_K_RHS % LOAD_TILES_IN_K_LHS == 0,
            f"LOAD_TILES_IN_K_RHS must be divisible by LOAD_TILES_IN_K_LHS, got {LOAD_TILES_IN_K_RHS} and {LOAD_TILES_IN_K_LHS}",
        )
        min_tiles_per_max_tile = LOAD_TILES_IN_K_RHS // LOAD_TILES_IN_K_LHS
        lhs_is_max_tiles = False

    for tile_m in range(LOAD_TILES_IN_M_RANGE):
        for tile_n in range(LOAD_TILES_IN_N_RANGE):
            for tile_k in range(max_tiles_in_k):
                # LHS is invariant in tile_n; load (and swizzle) once on tile_n == 0.
                if lhs_td != None and tile_n == 0:
                    if lhs_is_max_tiles or (tile_k % min_tiles_per_max_tile == 0):
                        lhs_tile_k = tile_k if lhs_is_max_tiles else tile_k // min_tiles_per_max_tile
                        global_tile_k_idx = block_idx_k * LOAD_TILES_IN_K_LHS + lhs_tile_k
                        k_global = block_k_offset_lhs + lhs_tile_k * lhs_k_stride
                        m_global = block_m_offset + tile_m * LHS_LOAD_TILE_M
                        m_sbuf_start = tile_m * LHS_LOAD_TILE_M

                        _load_single_tensor(
                            lhs_td,
                            lhs_sbuf_td,
                            k_global,
                            m_global,
                            m_sbuf_start,
                            lhs_tile_k,
                            LHS_LOAD_TILE_K,
                            LHS_LOAD_TILE_M,
                            BLOCK_M,
                            lhs_m_offset,
                            global_tile_k_idx,
                        )

                # RHS is invariant in tile_m; load (and swizzle) once on tile_m == 0.
                if rhs_td != None and tile_m == 0:
                    if (not lhs_is_max_tiles) or (tile_k % min_tiles_per_max_tile == 0):
                        rhs_tile_k = tile_k if not lhs_is_max_tiles else tile_k // min_tiles_per_max_tile
                        global_tile_k_idx = block_idx_k * LOAD_TILES_IN_K_RHS + rhs_tile_k
                        k_global = block_k_offset_rhs + rhs_tile_k * rhs_k_stride
                        n_global = block_n_offset + tile_n * RHS_LOAD_TILE_N
                        n_sbuf_start = tile_n * RHS_LOAD_TILE_N

                        _load_single_tensor(
                            rhs_td,
                            rhs_sbuf_td,
                            k_global,
                            n_global,
                            n_sbuf_start,
                            rhs_tile_k,
                            RHS_LOAD_TILE_K,
                            RHS_LOAD_TILE_N,
                            BLOCK_N,
                            rhs_n_offset,
                            global_tile_k_idx,
                        )

    """
    Reshape 4D unswizzled SBUF to 3D and normalize K tile count.
    Unswizzled DGT output: (physical_k, tiles_loaded, 1, physical_f).
    Need to reshape to (MATMUL_TILE_K, NUM_K_TILES, BLOCK_F) to match matmul expectations
    where NUM_K_TILES = BLOCK_K_PHYSICAL / MATMUL_TILE_K.
    """
    if lhs_sbuf != None and len(lhs_sbuf.shape) == 4:
        lhs_sbuf = lhs_sbuf.reshape((lhs_sbuf.shape[0], lhs_sbuf.shape[1], lhs_sbuf.shape[3]))

    if rhs_sbuf != None and len(rhs_sbuf.shape) == 4:
        rhs_sbuf = rhs_sbuf.reshape((rhs_sbuf.shape[0], rhs_sbuf.shape[1], rhs_sbuf.shape[3]))

    return lhs_sbuf, rhs_sbuf, lhs_data_sbuf, lhs_scales_sbuf, rhs_data_sbuf, rhs_scales_sbuf
