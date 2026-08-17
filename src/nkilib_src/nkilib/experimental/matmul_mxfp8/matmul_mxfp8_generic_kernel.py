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

"""Implements MXFP8 matrix multiplication with configurable tiling, supporting both pre-quantized and BF16 inputs.
The kernel handles physical/logical dimension mapping, block-level accumulation, and non-divisible shape masking."""

import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ..mxfp_utils.mxfp8_utils import quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils.common_dataclasses import BlockDescriptor, QuantScheme, TensorDescriptor
from ..mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from ..mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import get_fp8_dtype_x4
from .matmul_mxfp8_config import MatmulMxfp8KernelConfig, auto_generate_default, resolve_lnc2_sharding, validate_shapes
from .matmul_mxfp8_constants import PRECISION_BFLOAT16, PRECISION_FP32, PRECISION_MXFP8, PRECISION_MXFP8_X4
from .matmul_mxfp8_generic_api import generic_matmul_mxfp8_api


def _auto_generate_config(
    lhs_td: TensorDescriptor,
    rhs_td: TensorDescriptor,
    TILES_IN_BLOCK_M: int = None,
    TILES_IN_BLOCK_N: int = None,
    TILES_IN_BLOCK_K: int = None,
    TILES_IN_LOAD_M: int = None,
    TILES_IN_LOAD_N: int = None,
    lhs_matmul_tile_shape_logical: tuple = None,
    rhs_matmul_tile_shape_logical: tuple = None,
) -> tuple:
    """
    Auto-generate missing configuration parameters for matmul operation.

    Takes input tensor descriptors and any user-specified parameters, generates optimal
    defaults for any unspecified parameters based on tensor dimensions.
    If all parameters are already specified, returns them unchanged.

    Args:
            lhs_td: Left-hand side TensorDescriptor
            rhs_td: Right-hand side TensorDescriptor
            TILES_IN_BLOCK_M: Number of matmul tiles per block in M dimension (or None)
            TILES_IN_BLOCK_N: Number of matmul tiles per block in N dimension (or None)
            TILES_IN_BLOCK_K: Number of matmul tiles per block in K dimension (or None)
            TILES_IN_LOAD_M: Number of tiles to load at once in M dimension (or None)
            TILES_IN_LOAD_N: Number of tiles to load at once in N dimension (or None)
            lhs_matmul_tile_shape_logical: LHS tile shape (TILE_K, TILE_M) in logical space (or None)
            rhs_matmul_tile_shape_logical: RHS tile shape (TILE_K, TILE_N) in logical space (or None)

    Returns:
            Tuple of (TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K,
                                TILES_IN_LOAD_M, TILES_IN_LOAD_N,
                                lhs_matmul_tile_shape_logical, rhs_matmul_tile_shape_logical)

    Pseudocode:
        if all params specified: return unchanged
        derive K, M, N from tensor descriptors
        set tile shapes to (512, 128) and (512, 512) if dimensions allow
        set TILES_IN_BLOCK_K = 4, TILES_IN_BLOCK_N = 2, TILES_IN_BLOCK_M = 16 if dimensions allow
        set TILES_IN_LOAD_M = TILES_IN_BLOCK_M, TILES_IN_LOAD_N = TILES_IN_BLOCK_N
        return all parameters
    """
    # If all parameters are already specified, return them unchanged
    if None not in [
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M,
        TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical,
    ]:
        return (
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
            TILES_IN_BLOCK_K,
            TILES_IN_LOAD_M,
            TILES_IN_LOAD_N,
            lhs_matmul_tile_shape_logical,
            rhs_matmul_tile_shape_logical,
        )

    # Get dimensions from tensor descriptors (sharded for LNC2)
    K_logical_lhs, M_logical = lhs_td.sharded_logical_shape
    K_logical_rhs, N_logical = rhs_td.sharded_logical_shape
    kernel_assert(K_logical_lhs == K_logical_rhs, f"K dimension mismatch: {K_logical_lhs} vs {K_logical_rhs}")
    K_logical = K_logical_lhs

    # Set lhs_matmul_tile_shape_logical = (512, 128) if tensor size allows, else tensor size
    if lhs_matmul_tile_shape_logical == None:
        tile_k = 512 if K_logical >= 512 else K_logical
        tile_m = 128 if M_logical >= 128 else M_logical
        lhs_matmul_tile_shape_logical = (tile_k, tile_m)

    # Set rhs_matmul_tile_shape_logical = (512, 512) if tensor size allows, else tensor size
    if rhs_matmul_tile_shape_logical == None:
        tile_k = 512 if K_logical >= 512 else K_logical
        tile_n = 512 if N_logical >= 512 else N_logical
        rhs_matmul_tile_shape_logical = (tile_k, tile_n)

    # Extract tile dimensions for subsequent calculations
    LHS_TILE_K, LHS_TILE_M = lhs_matmul_tile_shape_logical
    RHS_TILE_K, RHS_TILE_N = rhs_matmul_tile_shape_logical

    # Set TILES_IN_BLOCK_K = 4 if tensor size allows, else K // tile_k (full tensor in K)
    if TILES_IN_BLOCK_K == None:
        max_tiles_k = K_logical // LHS_TILE_K
        TILES_IN_BLOCK_K = 4 if max_tiles_k >= 4 else max(1, max_tiles_k)

    # Set TILES_IN_BLOCK_N = 4 if tensor size allows, else N // tile_n
    if TILES_IN_BLOCK_N == None:
        max_tiles_n = N_logical // RHS_TILE_N
        TILES_IN_BLOCK_N = 2 if max_tiles_n >= 2 else max(1, max_tiles_n)

    # Set TILES_IN_BLOCK_M = 16 if tensor size allows, else M // tile_m
    if TILES_IN_BLOCK_M == None:
        max_tiles_m = M_logical // LHS_TILE_M
        TILES_IN_BLOCK_M = 16 if max_tiles_m >= 16 else max(1, max_tiles_m)

    # Set TILES_IN_LOAD_M = TILES_IN_BLOCK_M
    if TILES_IN_LOAD_M == None:
        TILES_IN_LOAD_M = TILES_IN_BLOCK_M

    # Set TILES_IN_LOAD_N = TILES_IN_BLOCK_N
    if TILES_IN_LOAD_N == None:
        TILES_IN_LOAD_N = TILES_IN_BLOCK_N

    return (
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M,
        TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical,
    )


def _validate_and_calculate_shapes(
    lhs_td: TensorDescriptor,
    rhs_td: TensorDescriptor,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_N: int,
    TILES_IN_BLOCK_K: int,
    TILES_IN_LOAD_M: int,
    TILES_IN_LOAD_N: int,
    lhs_matmul_tile_shape_logical: tuple,
    rhs_matmul_tile_shape_logical: tuple,
    use_scale_packing: bool = False,
) -> dict:
    """
    Validates inputs and calculates all shape parameters. See Key Concepts documentation at the top of the file.

    Args:
            lhs_td: Left-hand side TensorDescriptor
            rhs_td: Right-hand side TensorDescriptor
            TILES_IN_BLOCK_M (int): Number of matmul tiles per block in M dimension
            TILES_IN_BLOCK_N (int): Number of matmul tiles per block in N dimension
            TILES_IN_BLOCK_K (int): Number of matmul tiles per block in K dimension
            TILES_IN_LOAD_M (int): Number of tiles to load at once in M dimension
            TILES_IN_LOAD_N (int): Number of tiles to load at once in N dimension
            lhs_matmul_tile_shape_logical (tuple): LHS tile shape (TILE_K, TILE_M) in logical space
            rhs_matmul_tile_shape_logical (tuple): RHS tile shape (TILE_K, TILE_N) in logical space
            use_scale_packing (bool): If True and inputs are pre-quantized, assert that scales are packed

    Returns:
            Dictionary containing all calculated shapes and dimensions
    """
    # Extract tile shapes
    LHS_MATMUL_TILE_K_LOGICAL, LHS_MATMUL_TILE_M_LOGICAL = lhs_matmul_tile_shape_logical
    RHS_MATMUL_TILE_K_LOGICAL, RHS_MATMUL_TILE_N_LOGICAL = rhs_matmul_tile_shape_logical
    kernel_assert(
        LHS_MATMUL_TILE_K_LOGICAL == RHS_MATMUL_TILE_K_LOGICAL,
        f"LHS and RHS matmul tile K dimensions must match in logical space, got {LHS_MATMUL_TILE_K_LOGICAL=}, {RHS_MATMUL_TILE_K_LOGICAL=}",
    )

    MATMUL_TILE_K_PHYSICAL = LHS_MATMUL_TILE_K_LOGICAL // quantize_mxfp8_utils.INTERLEAVE_FACTOR
    LHS_MATMUL_TILE_M_PHYSICAL = LHS_MATMUL_TILE_M_LOGICAL
    RHS_MATMUL_TILE_N_PHYSICAL = RHS_MATMUL_TILE_N_LOGICAL
    lhs_matmul_tile_shape_physical = (MATMUL_TILE_K_PHYSICAL, LHS_MATMUL_TILE_M_PHYSICAL)
    rhs_matmul_tile_shape_physical = (MATMUL_TILE_K_PHYSICAL, RHS_MATMUL_TILE_N_PHYSICAL)

    if lhs_td.is_quantized:
        lhs_load_tile_shape = lhs_matmul_tile_shape_physical
        # Detect if LHS scales are packed by comparing data and scales shapes
        lhs_td.scales_are_packed = quantize_mxfp8_utils.are_scales_packed(lhs_td.data.shape, lhs_td.scales.shape)
        # If use_scale_packing is True, assert that pre-quantized LHS scales are packed
        kernel_assert(
            quantize_mxfp8_utils.scale_packing_not_possible(lhs_td.data.shape, lhs_td.scales.shape)
            or use_scale_packing == lhs_td.scales_are_packed,
            f"use_scale_packing={use_scale_packing} but lhs_scales_are_packed={lhs_td.scales_are_packed}. "
            f"LHS data shape: {lhs_td.data.shape}, LHS scales shape: {lhs_td.scales.shape}. ",
        )
    else:
        if lhs_td.is_swizzled:
            lhs_load_tile_shape = (
                lhs_matmul_tile_shape_physical[0],
                lhs_matmul_tile_shape_physical[1] * 4,
            )
        else:
            # Unswizzled: load tile is logical [K, M], DGT will transpose
            lhs_load_tile_shape = (LHS_MATMUL_TILE_K_LOGICAL, LHS_MATMUL_TILE_M_LOGICAL)

    if rhs_td.is_quantized:
        rhs_load_tile_shape = rhs_matmul_tile_shape_physical
        # Detect if RHS scales are packed by comparing data and scales shapes
        rhs_td.scales_are_packed = quantize_mxfp8_utils.are_scales_packed(rhs_td.data.shape, rhs_td.scales.shape)
        # If use_scale_packing is True, assert that pre-quantized RHS scales are packed
        kernel_assert(
            quantize_mxfp8_utils.scale_packing_not_possible(rhs_td.data.shape, rhs_td.scales.shape)
            or use_scale_packing == rhs_td.scales_are_packed,
            f"use_scale_packing={use_scale_packing} but rhs_scales_are_packed={rhs_td.scales_are_packed}. "
            f"RHS data shape: {rhs_td.data.shape}, RHS scales shape: {rhs_td.scales.shape}. ",
        )
    else:
        if rhs_td.is_swizzled:
            rhs_load_tile_shape = (
                rhs_matmul_tile_shape_physical[0],
                rhs_matmul_tile_shape_physical[1] * 4,
            )
        else:
            # Unswizzled: load tile is logical [K, N], DGT will transpose
            rhs_load_tile_shape = (RHS_MATMUL_TILE_K_LOGICAL, RHS_MATMUL_TILE_N_LOGICAL)

    lhs_quantize_tile_shape = (lhs_matmul_tile_shape_physical[0], lhs_matmul_tile_shape_physical[1] * 4)
    rhs_quantize_tile_shape = (rhs_matmul_tile_shape_physical[0], rhs_matmul_tile_shape_physical[1] * 4)

    """
    Pre-quantized packed scales require MATMUL_TILE_K_PHYSICAL == Q_TILE_K because
    the packed format uses Q_TILE_K-sized tile indexing that doesn't align with
    smaller physical tile boundaries in the matmul instruction.
    """
    if use_scale_packing and MATMUL_TILE_K_PHYSICAL < quantize_mxfp8_utils.Q_TILE_K:
        kernel_assert(
            not (lhs_td.is_quantized and lhs_td.scales_are_packed),
            f"Pre-quantized LHS with packed scales requires tile_k >= {quantize_mxfp8_utils.Q_TILE_K * quantize_mxfp8_utils.INTERLEAVE_FACTOR} "
            f"(MATMUL_TILE_K_PHYSICAL >= Q_TILE_K={quantize_mxfp8_utils.Q_TILE_K}), "
            f"got MATMUL_TILE_K_PHYSICAL={MATMUL_TILE_K_PHYSICAL}",
        )
        kernel_assert(
            not (rhs_td.is_quantized and rhs_td.scales_are_packed),
            f"Pre-quantized RHS with packed scales requires tile_k >= {quantize_mxfp8_utils.Q_TILE_K * quantize_mxfp8_utils.INTERLEAVE_FACTOR} "
            f"(MATMUL_TILE_K_PHYSICAL >= Q_TILE_K={quantize_mxfp8_utils.Q_TILE_K}), "
            f"got MATMUL_TILE_K_PHYSICAL={MATMUL_TILE_K_PHYSICAL}",
        )

    # Validate that the load tiles are valid
    kernel_assert(
        TILES_IN_LOAD_M <= TILES_IN_BLOCK_M,
        f"TILES_IN_LOAD_M ({TILES_IN_LOAD_M}) must be <= TILES_IN_BLOCK_M ({TILES_IN_BLOCK_M})",
    )
    kernel_assert(
        TILES_IN_LOAD_N <= TILES_IN_BLOCK_N,
        f"TILES_IN_LOAD_N ({TILES_IN_LOAD_N}) must be <= TILES_IN_BLOCK_N ({TILES_IN_BLOCK_N})",
    )
    kernel_assert(
        TILES_IN_BLOCK_M % TILES_IN_LOAD_M == 0,
        f"TILES_IN_BLOCK_M ({TILES_IN_BLOCK_M}) must be a positive multiple of TILES_IN_LOAD_M ({TILES_IN_LOAD_M})",
    )
    kernel_assert(
        TILES_IN_BLOCK_N % TILES_IN_LOAD_N == 0,
        f"TILES_IN_BLOCK_N ({TILES_IN_BLOCK_N}) must be a positive multiple of TILES_IN_LOAD_N ({TILES_IN_LOAD_N})",
    )

    # Calculate block dimensions from tile counts
    bd = BlockDescriptor(
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
    )

    # Get tensor dimensions from tensor descriptors
    K_PHYSICAL_LHS, M_PHYSICAL = lhs_td.physical_shape
    K_PHYSICAL_RHS, N_PHYSICAL = rhs_td.physical_shape

    K_LOGICAL_LHS, M_LOGICAL = lhs_td.logical_shape
    K_LOGICAL_RHS, N_LOGICAL = rhs_td.logical_shape
    kernel_assert(K_LOGICAL_LHS == K_LOGICAL_RHS, f"K dimension mismatch: {K_LOGICAL_LHS} vs {K_LOGICAL_RHS}")
    K_LOGICAL = K_LOGICAL_LHS

    # Validate that the Logical dimensions are divisible
    # Blocks are divisible by tiles (required for hardware)
    kernel_assert(
        bd.BLOCK_M_LOGICAL % LHS_MATMUL_TILE_M_LOGICAL == 0,
        f"BLOCK_M_LOGICAL ({bd.BLOCK_M_LOGICAL}) must be divisible by LHS_MATMUL_TILE_M_LOGICAL ({LHS_MATMUL_TILE_M_LOGICAL})",
    )
    kernel_assert(
        bd.BLOCK_K_LOGICAL % LHS_MATMUL_TILE_K_LOGICAL == 0,
        f"BLOCK_K_LOGICAL ({bd.BLOCK_K_LOGICAL}) must be divisible by LHS_MATMUL_TILE_K_LOGICAL ({LHS_MATMUL_TILE_K_LOGICAL})",
    )
    kernel_assert(
        bd.BLOCK_N_LOGICAL % RHS_MATMUL_TILE_N_LOGICAL == 0,
        f"BLOCK_N_LOGICAL ({bd.BLOCK_N_LOGICAL}) must be divisible by RHS_MATMUL_TILE_N_LOGICAL ({RHS_MATMUL_TILE_N_LOGICAL})",
    )

    # Validate that the Physical dimensions are correct
    # Tile dimensions are valid for hardware
    kernel_assert(
        MATMUL_TILE_K_PHYSICAL <= nl.tile_size.pmax
        and MATMUL_TILE_K_PHYSICAL % quantize_mxfp8_utils.MX_PARTITION_SIZE == 0,
        f"MATMUL_TILE_K_PHYSICAL ({MATMUL_TILE_K_PHYSICAL}) must be <= pmax ({nl.tile_size.pmax}) and divisible by quantize_mxfp8_utils.MX_PARTITION_SIZE ({quantize_mxfp8_utils.MX_PARTITION_SIZE})",
    )
    kernel_assert(
        LHS_MATMUL_TILE_M_PHYSICAL <= nl.tile_size.gemm_stationary_fmax
        and LHS_MATMUL_TILE_M_PHYSICAL % quantize_mxfp8_utils.MX_PARTITION_SIZE == 0,
        f"LHS_MATMUL_TILE_M_PHYSICAL ({LHS_MATMUL_TILE_M_PHYSICAL}) must be <= gemm_stationary_fmax ({nl.tile_size.gemm_stationary_fmax}) and divisible by quantize_mxfp8_utils.MX_PARTITION_SIZE ({quantize_mxfp8_utils.MX_PARTITION_SIZE})",
    )
    kernel_assert(
        RHS_MATMUL_TILE_N_PHYSICAL <= quantize_mxfp8_utils.TILE_SIZE_GEMM_MOVING_MAX
        and RHS_MATMUL_TILE_N_PHYSICAL % quantize_mxfp8_utils.MX_PARTITION_SIZE == 0,
        f"RHS_MATMUL_TILE_N_PHYSICAL ({RHS_MATMUL_TILE_N_PHYSICAL}) must be <= quantize_mxfp8_utils.TILE_SIZE_GEMM_MOVING_MAX ({quantize_mxfp8_utils.TILE_SIZE_GEMM_MOVING_MAX}) and divisible by quantize_mxfp8_utils.MX_PARTITION_SIZE ({quantize_mxfp8_utils.MX_PARTITION_SIZE})",
    )
    # Blocks are divisible by tiles (required for hardware)
    kernel_assert(
        bd.BLOCK_M_PHYSICAL % LHS_MATMUL_TILE_M_PHYSICAL == 0,
        f"BLOCK_M_PHYSICAL ({bd.BLOCK_M_PHYSICAL}) must be divisible by LHS_MATMUL_TILE_M_PHYSICAL ({LHS_MATMUL_TILE_M_PHYSICAL})",
    )
    kernel_assert(
        bd.BLOCK_K_PHYSICAL_LHS % MATMUL_TILE_K_PHYSICAL == 0,
        f"BLOCK_K_PHYSICAL_LHS ({bd.BLOCK_K_PHYSICAL_LHS}) must be divisible by MATMUL_TILE_K_PHYSICAL ({MATMUL_TILE_K_PHYSICAL})",
    )
    kernel_assert(
        bd.BLOCK_K_PHYSICAL_RHS % MATMUL_TILE_K_PHYSICAL == 0,
        f"BLOCK_K_PHYSICAL_RHS ({bd.BLOCK_K_PHYSICAL_RHS}) must be divisible by MATMUL_TILE_K_PHYSICAL ({MATMUL_TILE_K_PHYSICAL})",
    )
    kernel_assert(
        bd.BLOCK_N_PHYSICAL % RHS_MATMUL_TILE_N_PHYSICAL == 0,
        f"BLOCK_N_PHYSICAL ({bd.BLOCK_N_PHYSICAL}) must be divisible by RHS_MATMUL_TILE_N_PHYSICAL ({RHS_MATMUL_TILE_N_PHYSICAL})",
    )
    # Block sizes must not exceed tensor dimensions
    kernel_assert(
        2 * M_PHYSICAL >= bd.BLOCK_M_PHYSICAL or M_PHYSICAL < LHS_MATMUL_TILE_M_PHYSICAL,
        f"M_PHYSICAL ({M_PHYSICAL}) must be >= BLOCK_M_PHYSICAL ({bd.BLOCK_M_PHYSICAL}), or M_PHYSICAL ({M_PHYSICAL}) must be < LHS_MATMUL_TILE_M_PHYSICAL ({LHS_MATMUL_TILE_M_PHYSICAL})",
    )
    kernel_assert(
        2 * M_LOGICAL >= bd.BLOCK_M_LOGICAL or M_LOGICAL < LHS_MATMUL_TILE_M_LOGICAL,
        f"M_LOGICAL ({M_LOGICAL}) must be >= BLOCK_M_LOGICAL ({bd.BLOCK_M_LOGICAL}), or M_LOGICAL ({M_LOGICAL}) must be < LHS_MATMUL_TILE_M_LOGICAL ({LHS_MATMUL_TILE_M_LOGICAL})",
    )
    kernel_assert(
        2 * N_PHYSICAL >= bd.BLOCK_N_PHYSICAL or N_PHYSICAL < RHS_MATMUL_TILE_N_PHYSICAL,
        f"N_PHYSICAL ({N_PHYSICAL}) must be >= BLOCK_N_PHYSICAL ({bd.BLOCK_N_PHYSICAL}), or N_PHYSICAL ({N_PHYSICAL}) must be < RHS_MATMUL_TILE_N_PHYSICAL ({RHS_MATMUL_TILE_N_PHYSICAL})",
    )
    kernel_assert(
        2 * N_LOGICAL >= bd.BLOCK_N_LOGICAL or N_LOGICAL < RHS_MATMUL_TILE_N_LOGICAL,
        f"N_LOGICAL ({N_LOGICAL}) must be >= BLOCK_N_LOGICAL ({bd.BLOCK_N_LOGICAL}), or N_LOGICAL ({N_LOGICAL}) must be < RHS_MATMUL_TILE_N_LOGICAL ({RHS_MATMUL_TILE_N_LOGICAL})",
    )
    kernel_assert(
        2 * K_LOGICAL >= bd.BLOCK_K_LOGICAL or K_LOGICAL < LHS_MATMUL_TILE_K_LOGICAL,
        f"K_LOGICAL ({K_LOGICAL}) must be >= BLOCK_K_LOGICAL ({bd.BLOCK_K_LOGICAL}), or K_LOGICAL ({K_LOGICAL}) must be < LHS_MATMUL_TILE_K_LOGICAL ({LHS_MATMUL_TILE_K_LOGICAL})",
    )

    # With masking support, tensor dimensions can be non-divisible by block sizes.
    # Masking prevents out-of-bounds access. Use ceiling division to calculate block counts.
    BLOCKS_IN_M = div_ceil(M_LOGICAL, bd.BLOCK_M_LOGICAL)
    BLOCKS_IN_N = div_ceil(N_LOGICAL, bd.BLOCK_N_LOGICAL)
    BLOCKS_IN_K = div_ceil(K_LOGICAL, bd.BLOCK_K_LOGICAL)
    if not lhs_td.is_swizzled or not rhs_td.is_swizzled:
        # DGT requires K divisible by 128; the fast DMA transpose path relaxes this to
        # MX_PARTITION_SIZE (32) since it gathers any 32-aligned K in a single op.
        unswizzled_fast_dma = (not lhs_td.is_swizzled and lhs_td.fast_dma_transpose) or (
            not rhs_td.is_swizzled and rhs_td.fast_dma_transpose
        )
        k_alignment = quantize_mxfp8_utils.MX_PARTITION_SIZE if unswizzled_fast_dma else 128
        kernel_assert(K_LOGICAL % k_alignment == 0, f"K must be divisible by {k_alignment} for DGT, got {K_LOGICAL}")

    return {
        'lhs_matmul_tile_shape_physical': lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_shape_physical': rhs_matmul_tile_shape_physical,
        'lhs_load_tile_shape': lhs_load_tile_shape,
        'rhs_load_tile_shape': rhs_load_tile_shape,
        'lhs_quantize_tile_shape': lhs_quantize_tile_shape,
        'rhs_quantize_tile_shape': rhs_quantize_tile_shape,
        'bd': bd,
        'M_LOGICAL': M_LOGICAL,
        'N_LOGICAL': N_LOGICAL,
        'K_LOGICAL': K_LOGICAL,
        'M_PHYSICAL': M_PHYSICAL,
        'K_PHYSICAL_LHS': K_PHYSICAL_LHS,
        'K_PHYSICAL_RHS': K_PHYSICAL_RHS,
        'N_PHYSICAL': N_PHYSICAL,
        'BLOCKS_IN_M': BLOCKS_IN_M,
        'BLOCKS_IN_N': BLOCKS_IN_N,
        'BLOCKS_IN_K': BLOCKS_IN_K,
    }


@with_active_sbm
def matmul_mxfp8(
    lhs,
    rhs,
    TILES_IN_BLOCK_M: int = None,
    TILES_IN_BLOCK_N: int = None,
    TILES_IN_BLOCK_K: int = None,
    TILES_IN_LOAD_M: int = None,
    TILES_IN_LOAD_N: int = None,
    lhs_matmul_tile_shape_logical: tuple = None,
    rhs_matmul_tile_shape_logical: tuple = None,
    block_loop_order: str = 'mnk',
    tile_loop_order: str = 'mnk',
    float8_dtype: str = "float8_e5m2",
    output_dtype=nl.bfloat16,
    run_with_lnc2: bool = True,
    lnc_2_shard_rhs=None,
    lhs_scales=None,
    rhs_scales=None,
    use_scale_packing: bool = False,
    spill_reload: bool = False,
    lhs_is_swizzled: bool = True,
    rhs_is_swizzled: bool = True,
    load_with_PE_swizzle: bool = False,
    lhs_is_f_by_k: bool = True,
    rhs_is_f_by_k: bool = True,
    fast_dma_transpose: bool = False,
    enable_psum_copy_in=None,
    quant_scheme: str = "wrapX",
) -> nl.ndarray:
    """
    Performs matrix multiplication with MXFP8 quantization.

    This kernel implements efficient matrix multiplication using MXFP8 quantization format,
    supporting both pre-quantized inputs and automatic quantization from BF16. The kernel
    uses hardware-optimized tiling and supports LNC2 parallelization for improved throughput.

    Dimensions:
        M: Number of rows in left-hand side matrix (output rows)
        K: Contraction dimension (columns in LHS, rows in RHS)
        N: Number of columns in right-hand side matrix (output columns)

    Args:
        lhs: Left-hand side matrix, either BF16 tensor or tuple (data, scales) for pre-quantized MXFP8
        rhs: Right-hand side matrix, either BF16 tensor or tuple (data, scales) for pre-quantized MXFP8
        TILES_IN_BLOCK_M (int): Number of matmul tiles per block in M dimension (auto-generated if None)
        TILES_IN_BLOCK_N (int): Number of matmul tiles per block in N dimension (auto-generated if None)
        TILES_IN_BLOCK_K (int): Number of matmul tiles per block in K dimension (auto-generated if None)
        TILES_IN_LOAD_M (int): Number of tiles to load at once in M dimension (auto-generated if None)
        TILES_IN_LOAD_N (int): Number of tiles to load at once in N dimension (auto-generated if None)
        lhs_matmul_tile_shape_logical: LHS tile shape (TILE_K, TILE_M) in logical space (auto-generated if None)
        rhs_matmul_tile_shape_logical: RHS tile shape (TILE_K, TILE_N) in logical space (auto-generated if None)
        block_loop_order (str): Block processing order, default 'mnk'
        tile_loop_order (str): Tile processing order within blocks, default 'mnk'
        float8_dtype (str): FP8 dtype for quantization, default "float8_e5m2"
        output_dtype: Output data type, default nl.float32
        run_with_lnc2 (bool): Enable LNC2 parallelization across 2 cores, default True
        lnc_2_shard_rhs (bool): When run_with_lnc2=True, shard on N dimension (RHS) if True,
            or shard on M dimension (LHS) if False. Default True.
        lhs_scales: Optional pre-computed scales for LHS
        rhs_scales: Optional pre-computed scales for RHS
        use_scale_packing (bool): If True and inputs are pre-quantized,
            assert that scales are packed, default False
        spill_reload (bool): If True, each quantized block will be written to HBM
            and on every subsequent load, this spilled block will be reloaded.
        lhs_is_swizzled (bool): Whether LHS BF16 tensor is pre-swizzled [K/4, M*4],
            default True. If False, expects [M, K] layout.
        rhs_is_swizzled (bool): Whether RHS BF16 tensor is pre-swizzled [K/4, N*4],
            default True. If False, expects [N, K] layout.
        fast_dma_transpose (bool): When True, use a direct 4D access pattern on the
            source tensor for DMA gather-transpose instead of flattening + vector offsets.
            Avoids allocating vector_offset_pattern buffers in SBUF. Only applies to
            unswizzled inputs. Default False.

    Returns:
        nl.ndarray: Result of matrix multiplication [M, N] in HBM with specified output_dtype

    Notes:
        - Supports non-divisible tensor shapes using dynamic slicing (nl.ds)
        - Auto-generates optimal tiling parameters when not specified
        - LNC2 mode requires at least 2 blocks in N dimension
        - Pre-quantized inputs must be in MXFP8 format (data, scales) tuple
        - When use_scale_packing=True, pre-quantized inputs must have packed scales
        - TODO: Specify intended usage range for optimal performance

        Physical vs Logical Dimensions:
            - Logical: Theoretical tensor dimensions [M, K] @ [K, N] for the matmul operation
            - Physical: Hardware storage format (depends on quantization and swizzling)
                * Pre-swizzled: [K//4, M*4] or [K//4, N*4]
                * Quantized: [K//4, M] or [K//4, N]

        Tiles (smallest processing unit):
            - Matmul Tile: Hardware matmul operation shape
                * LHS: [128, 128] physical, [512, 128] logical
                * RHS: [128, 512] physical, [512, 512] logical
            - Load Tile: Data loaded per matmul tile (varies by quantization state)
            - Quantize Tile: Input shape for quantization to produce one matmul tile

        Blocks (collection of tiles):
            - Group of tiles processed together
            - Must fit in SBUF (including load, quantize, and output buffers)
            - Accumulates results across K dimension before storing to HBM

        Non-Divisible Shape Handling:
            - Uses ceiling division for block counts
            - Applies nl.ds (dynamic slice) for boundary handling at load and store operations

    Example::

        import nki.language as nl

        # Basic usage with BF16 inputs
        lhs = nl.ndarray((512, 1024), dtype=nl.bfloat16, buffer=nl.hbm)
        rhs = nl.ndarray((512, 2048), dtype=nl.bfloat16, buffer=nl.hbm)

        result = matmul_mxfp8(
            lhs=lhs,
            rhs=rhs,
            TILES_IN_BLOCK_M=2,
            TILES_IN_BLOCK_N=2,
            TILES_IN_BLOCK_K=1,
            TILES_IN_LOAD_M=1,
            TILES_IN_LOAD_N=1,
            lhs_matmul_tile_shape_logical=(512, 128),
            rhs_matmul_tile_shape_logical=(512, 512),
        )

        # Usage with pre-quantized inputs (tuple of data and scales)
        lhs_quantized = (lhs_data, lhs_scales)
        result = matmul_mxfp8(lhs=lhs_quantized, rhs=rhs, ...)

    Pseudocode:
        # Setup: Auto-generate config and validate shapes
        config = auto_generate_config(lhs, rhs, ...)
        shapes = validate_and_calculate_shapes(...)

        # LNC2 sharding (if enabled)
        if run_with_lnc2:
            shard_id = program_id(axis=0)
            rhs_sharded = rhs[:, shard_id * N//2 : (shard_id+1) * N//2]

        # Block-level matmul with K-accumulation
        for block_m in range(BLOCKS_IN_M):
            for block_n in range(BLOCKS_IN_N):
                output_block = zeros()
                for block_k in range(BLOCKS_IN_K):
                    lhs_block = load_and_quantize(lhs, block_m, block_k)
                    rhs_block = load_and_quantize(rhs, block_k, block_n)
                    output_block += matmul_mxfp8_blocks(lhs_block, rhs_block)
                store(output_block, block_m, block_n)
    """

    # Construct TensorDescriptors from raw inputs
    if get_active_sbm() is None:
        create_and_set_active_sbm()

    sbm = get_active_sbm()
    sbm.open_scope(name="MXFP8 Matmul")

    kernel_assert(
        lhs_is_f_by_k or not lhs_is_swizzled,
        "K-by-F layout (lhs_is_f_by_k=False) is not supported for pre-swizzled inputs.",
    )
    kernel_assert(
        rhs_is_f_by_k or not rhs_is_swizzled,
        "K-by-F layout (rhs_is_f_by_k=False) is not supported for pre-swizzled inputs.",
    )

    # Resolve quant_scheme string to enum
    _quant_scheme_map = {"wrapX": QuantScheme.WRAPX, "1x32": QuantScheme._1x32}
    kernel_assert(
        quant_scheme in _quant_scheme_map, f"Invalid quant_scheme '{quant_scheme}', must be 'wrapX' or '1x32'."
    )
    resolved_quant_scheme = _quant_scheme_map[quant_scheme]

    # NOTE: the K-by-F F-dimension requirement (F % 512 for non-swizzled BF16) is enforced
    # centrally in validate_shapes() so every kernel using the generic API gets it.
    lhs_td = TensorDescriptor(
        data=lhs,
        scales=lhs_scales,
        is_swizzled=lhs_is_swizzled,
        is_col_parallel_sharded=False,
        load_with_PE_swizzle=load_with_PE_swizzle if not lhs_is_swizzled else False,
        is_f_by_k=None if lhs_is_f_by_k else False,
        fast_dma_transpose=fast_dma_transpose if not lhs_is_swizzled else False,
        quant_scheme=resolved_quant_scheme,
    )
    rhs_td = TensorDescriptor(
        data=rhs,
        scales=rhs_scales,
        is_swizzled=rhs_is_swizzled,
        is_col_parallel_sharded=False,
        load_with_PE_swizzle=load_with_PE_swizzle if not rhs_is_swizzled else False,
        is_f_by_k=None if rhs_is_f_by_k else False,
        fast_dma_transpose=fast_dma_transpose if not rhs_is_swizzled else False,
        quant_scheme=resolved_quant_scheme,
    )

    run_with_lnc2, lnc_2_shard_rhs = resolve_lnc2_sharding(
        lhs_td.logical_shape[1], rhs_td.logical_shape[1], run_with_lnc2, lnc_2_shard_rhs
    )
    shard_rhs = run_with_lnc2 and lnc_2_shard_rhs
    shard_lhs = run_with_lnc2 and not lnc_2_shard_rhs
    if shard_lhs:
        lhs_td.shard_col_parallel()
    elif shard_rhs:
        rhs_td.shard_col_parallel()

    # Build MatmulMxfp8KernelConfig and auto-generate missing fields
    K_logical_lhs, M_logical = lhs_td.sharded_logical_shape
    _, N_logical = rhs_td.sharded_logical_shape
    lhs_precision = (
        PRECISION_BFLOAT16 if not lhs_td.is_quantized else (PRECISION_MXFP8_X4 if lhs_td.is_x4 else PRECISION_MXFP8)
    )
    rhs_precision = (
        PRECISION_BFLOAT16 if not rhs_td.is_quantized else (PRECISION_MXFP8_X4 if rhs_td.is_x4 else PRECISION_MXFP8)
    )
    output_precision = PRECISION_FP32 if output_dtype == nl.float32 else PRECISION_BFLOAT16

    config = MatmulMxfp8KernelConfig(
        M=M_logical,
        K=K_logical_lhs,
        N=N_logical,
        tile_m=lhs_matmul_tile_shape_logical[1] if lhs_matmul_tile_shape_logical else None,
        tile_k=lhs_matmul_tile_shape_logical[0] if lhs_matmul_tile_shape_logical else None,
        tile_n=rhs_matmul_tile_shape_logical[1] if rhs_matmul_tile_shape_logical else None,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        enable_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
        lnc_2_shard_rhs=lnc_2_shard_rhs,
        lhs_is_swizzled=lhs_is_swizzled,
        rhs_is_swizzled=rhs_is_swizzled,
        load_with_PE_swizzle=load_with_PE_swizzle,
        quant_scheme=quant_scheme,
    )
    auto_generate_default(
        config, lhs_precision, rhs_precision, output_precision, enable_psum_copy_in=enable_psum_copy_in
    )

    # Validate and calculate all shapes
    validate_shapes(config, lhs_td, rhs_td)

    output_tensor_hbm = nl.ndarray(
        (lhs_td.logical_shape[1], rhs_td.logical_shape[1]),
        dtype=output_dtype,
        buffer=nl.shared_hbm,
    )
    kernel_assert(
        not run_with_lnc2 or not lnc_2_shard_rhs or config.BLOCKS_IN_N >= 2,
        (
            f"LNC2 N-sharding requires at least 2 blocks in N dimension, but got {config.BLOCKS_IN_N}. "
            f"Either increase N dimension, decrease block size, or use run_with_lnc2=False."
        ),
    )
    kernel_assert(
        not run_with_lnc2 or lnc_2_shard_rhs or config.BLOCKS_IN_M >= 2,
        (
            f"LNC2 M-sharding requires at least 2 blocks in M dimension, but got {config.BLOCKS_IN_M}. "
            f"Either increase M dimension, decrease block size, or use run_with_lnc2=False."
        ),
    )

    K_LOGICAL = lhs_td.logical_shape[0]
    # DGT loading requires K divisible by 128, except the fast DMA transpose path, which
    # gathers any K aligned to MX_PARTITION_SIZE (32) in a single op (see load_tile_dgt).
    rhs_k_alignment = quantize_mxfp8_utils.MX_PARTITION_SIZE if rhs_td.fast_dma_transpose else 128
    lhs_k_alignment = quantize_mxfp8_utils.MX_PARTITION_SIZE if lhs_td.fast_dma_transpose else 128
    kernel_assert(
        rhs_td.is_quantized or rhs_td.is_swizzled or K_LOGICAL % rhs_k_alignment == 0,
        f"If kernel is not pre-quantized or pre-swizzled it K dim must be divisible by {rhs_k_alignment} for DGT loading",
    )

    kernel_assert(
        lhs_td.is_quantized or lhs_td.is_swizzled or K_LOGICAL % lhs_k_alignment == 0,
        f"If kernel is not pre-quantized or pre-swizzled it K dim must be divisible by {lhs_k_alignment} for DGT loading",
    )

    N_LOGICAL_SHARDED = rhs_td.sharded_logical_shape[1]
    N_PHYSICAL_SHARDED = rhs_td.sharded_physical_shape[1]
    M_LOGICAL_SHARDED = lhs_td.sharded_logical_shape[1]
    M_PHYSICAL_SHARDED = lhs_td.sharded_physical_shape[1]

    rhs_n_offset = 0
    lhs_m_offset = 0
    if run_with_lnc2:
        LNC_ID = nl.program_id(axis=0)
        if lnc_2_shard_rhs:
            # Shard on N dimension (RHS)
            out_col_idx_start = LNC_ID * N_LOGICAL_SHARDED
            out_col_idx_end = out_col_idx_start + N_LOGICAL_SHARDED
            output_tensor_hbm_sharded = output_tensor_hbm[:, out_col_idx_start:out_col_idx_end]
            rhs_n_offset = LNC_ID * N_PHYSICAL_SHARDED
            BLOCKS_IN_N_sharded = div_ceil(config.BLOCKS_IN_N, 2)
            BLOCKS_IN_M_sharded = config.BLOCKS_IN_M
        else:
            # Shard on M dimension (LHS)
            out_row_idx_start = LNC_ID * M_LOGICAL_SHARDED
            out_row_idx_end = out_row_idx_start + M_LOGICAL_SHARDED
            output_tensor_hbm_sharded = output_tensor_hbm[out_row_idx_start:out_row_idx_end, :]
            lhs_m_offset = LNC_ID * M_PHYSICAL_SHARDED
            BLOCKS_IN_M_sharded = div_ceil(config.BLOCKS_IN_M, 2)
            BLOCKS_IN_N_sharded = config.BLOCKS_IN_N
    else:
        output_tensor_hbm_sharded = output_tensor_hbm
        BLOCKS_IN_N_sharded = config.BLOCKS_IN_N
        BLOCKS_IN_M_sharded = config.BLOCKS_IN_M

    # Allocate HBM for the quantized LHS and RHS.
    # Defaults to using X4 for spill/reload as this is an internal utility.
    lhsq_td = None
    rhsq_td = None

    fp8_x4_dtype = get_fp8_dtype_x4(float8_dtype)

    # TODO: WHEN QUANTIZATION IS DONE IN LNC2 SHARDED MANNER, THIS WILL NEED UPDATE
    # Private HBM is needed because each core is writing to the same locations
    data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

    TILE_K = config.lhs_load_tile_shape[0]

    BLOCK_K_SIZE = TILE_K * config.TILES_IN_BLOCK_K

    NUM_TILES_K = div_ceil(lhs_td.physical_shape[0], TILE_K)

    NUM_SCALE_GROUPS = (
        NUM_TILES_K + quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP - 1
    ) // quantize_mxfp8_utils.MAX_TILES_PER_SCALE_PACKING_GROUP

    bd = config.bd

    if spill_reload and not lhs_td.is_quantized:
        lhs_qdata_hbm = nl.ndarray(
            (config.BLOCKS_IN_K * BLOCK_K_SIZE, BLOCKS_IN_M_sharded * bd.BLOCK_M_LOGICAL),
            dtype=fp8_x4_dtype,
            buffer=data_buffer,
        )

        if use_scale_packing:
            lhs_scale_hbm = nl.ndarray(
                (TILE_K * NUM_SCALE_GROUPS, BLOCKS_IN_M_sharded * bd.BLOCK_M_LOGICAL),
                dtype=nl.float8_e8m0fnu,
                buffer=data_buffer,
            )

        else:
            lhs_scale_hbm = nl.ndarray(
                (config.BLOCKS_IN_K * BLOCK_K_SIZE, BLOCKS_IN_M_sharded * bd.BLOCK_M_LOGICAL),
                dtype=nl.float8_e8m0fnu,
                buffer=data_buffer,
            )

        lhsq_td = TensorDescriptor(
            data=lhs_qdata_hbm,
            scales=lhs_scale_hbm,
            is_swizzled=True,
            is_x4=True,
            scales_are_packed=use_scale_packing,
        )

    if spill_reload and not rhs_td.is_quantized:
        rhs_qdata_hbm = nl.ndarray(
            (config.BLOCKS_IN_K * BLOCK_K_SIZE, BLOCKS_IN_N_sharded * bd.BLOCK_N_LOGICAL),
            dtype=fp8_x4_dtype,
            buffer=data_buffer,
        )

        if use_scale_packing:
            rhs_scale_hbm = nl.ndarray(
                (TILE_K * NUM_SCALE_GROUPS, BLOCKS_IN_N_sharded * bd.BLOCK_N_LOGICAL),
                dtype=nl.float8_e8m0fnu,
                buffer=data_buffer,
            )

        else:
            rhs_scale_hbm = nl.ndarray(
                (config.BLOCKS_IN_K * BLOCK_K_SIZE, BLOCKS_IN_N_sharded * bd.BLOCK_N_LOGICAL),
                dtype=nl.float8_e8m0fnu,
                buffer=data_buffer,
            )

        rhsq_td = TensorDescriptor(
            data=rhs_qdata_hbm,
            scales=rhs_scale_hbm,
            is_swizzled=True,
            is_x4=True,
            scales_are_packed=use_scale_packing,
        )

    # This code currently only supports MNK loop order over blocks
    if block_loop_order == 'mnk':
        output_td = TensorDescriptor(data=output_tensor_hbm_sharded)

        generic_matmul_mxfp8_api(
            lhs_hbm_td=lhs_td,
            rhs_hbm_td=rhs_td,
            config=config,
            output_td=output_td,
            output_dtype=output_dtype,
            block_idx_m=(0, BLOCKS_IN_M_sharded),
            block_idx_n=(0, BLOCKS_IN_N_sharded),
            block_idx_k=(0, config.BLOCKS_IN_K),
            tile_loop_order=tile_loop_order,
            float8_dtype=float8_dtype,
            use_scale_packing=use_scale_packing,
            spill_reload=spill_reload,
            lhsq_td=lhsq_td,
            rhsq_td=rhsq_td,
            rhs_n_offset=rhs_n_offset,
            lhs_m_offset=lhs_m_offset,
        )

    sbm.close_scope()

    return output_tensor_hbm
