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

"""Shared tile constants and helpers for MXFP8 MLP forward and backward kernels."""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ..mxfp_utils.mxfp8_utils.common_dataclasses import BlockDescriptor, TensorDescriptor
from ..mxfp_utils.mxfp8_utils.common_utils import get_active_sbm
from ..mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    get_fp8_dtype_x4,
    get_scale_output_shape,
)

# Base tile sizes (shared across fwd and bwd configs)
TILE_M = 128
TILE_N = 512
L_TILE_K = 512
DGT_MIN_K = 128

# Maximum number of M tiles to load at once in a single DMA transfer
# TODO: Replace with auto-tuned value once perf optimization changes are in
MAX_TILES_IN_LOAD_M = 8

# Number of LNC (Logical Neuron Core) cores for LNC2 sharding
NUM_LNC2_CORES = 2

# Number of fused projections in the gate_up weight matrix (gate + up)
NUM_GATE_UP_PROJECTIONS = 2

# Physical tile K after x4 interleaving (L_TILE_K / INTERLEAVE_FACTOR)
MATMUL_TILE_K_PHYSICAL = L_TILE_K // INTERLEAVE_FACTOR

# Tile shape constants for generic_matmul_mxfp8_api
LHS_MATMUL_TILE_PHYSICAL = (MATMUL_TILE_K_PHYSICAL, TILE_M)
RHS_MATMUL_TILE_PHYSICAL = (MATMUL_TILE_K_PHYSICAL, TILE_N)
LHS_LOAD_TILE = (L_TILE_K, TILE_M)
RHS_LOAD_TILE = (L_TILE_K, TILE_N)
LHS_QUANTIZE_TILE = (MATMUL_TILE_K_PHYSICAL, TILE_M * INTERLEAVE_FACTOR)
RHS_QUANTIZE_TILE = (MATMUL_TILE_K_PHYSICAL, TILE_N * INTERLEAVE_FACTOR)


def build_tile_sizes(tile_m: int = TILE_M, l_tile_k: int = L_TILE_K, tile_n: int = TILE_N) -> dict:
    """Build a tiles dict from explicit tile sizes (no dimension-adaptive capping).

    Companion to :func:`get_tile_sizes`, which derives the tile sizes from the
    tensor dims and caps ``tile_n`` at ``TILE_N``. This variant takes the tile
    sizes directly (e.g. from an autotune-cache config) so a caller can use a
    tuned ``tile_n`` that exceeds ``TILE_N`` (a moving-dim tile spanning multiple
    PSUM banks) while producing the same derived shapes ``get_tile_sizes`` does.

    Args:
        tile_m: Stationary (M) tile size.
        l_tile_k: Logical contraction (K) tile size; physical K is
            ``l_tile_k // INTERLEAVE_FACTOR``.
        tile_n: Moving (N) tile size.

    Returns:
        Dict with keys: tile_m, tile_n, l_tile_k, matmul_tile_k_physical,
        lhs_matmul_tile_physical, rhs_matmul_tile_physical,
        lhs_load_tile, rhs_load_tile, lhs_quantize_tile, rhs_quantize_tile.
    """
    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR

    return {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': (matmul_tile_k_physical, tile_m),
        'rhs_matmul_tile_physical': (matmul_tile_k_physical, tile_n),
        'lhs_load_tile': (l_tile_k, tile_m),
        'rhs_load_tile': (l_tile_k, tile_n),
        'lhs_quantize_tile': (matmul_tile_k_physical, tile_m * INTERLEAVE_FACTOR),
        'rhs_quantize_tile': (matmul_tile_k_physical, tile_n * INTERLEAVE_FACTOR),
    }


def get_tile_sizes(K, M, N):
    """Return tile sizes adapted to tensor dimensions, matching matmul kernel auto-generation.

    Args:
        K: Contraction dimension (hidden size for gate/up, intermediate size for down).
        M: Stationary dimension (sequence length).
        N: Moving dimension (intermediate size for gate/up, hidden size for down).

    Returns:
        Dict with keys: tile_m, tile_n, l_tile_k, matmul_tile_k_physical,
        lhs_matmul_tile_physical, rhs_matmul_tile_physical,
        lhs_load_tile, rhs_load_tile, lhs_quantize_tile, rhs_quantize_tile.
    """
    tile_m = TILE_M if M >= TILE_M else M
    tile_n = TILE_N if N >= TILE_N else N
    l_tile_k = L_TILE_K if K >= L_TILE_K else K
    return build_tile_sizes(tile_m=tile_m, l_tile_k=l_tile_k, tile_n=tile_n)


def _build_matmul_params(
    tiles_in_block_m: int,
    tiles_in_block_n: int,
    tiles_in_block_k: int,
    lhs_load_tile_shape: tuple = None,
    rhs_load_tile_shape: tuple = None,
    tiles: dict = None,
) -> BlockDescriptor:
    """Build BlockDescriptor for generic_matmul_mxfp8_api."""
    if tiles == None:
        l_tile_k, tile_m, tile_n = L_TILE_K, TILE_M, TILE_N
        lhs_load_default, rhs_load_default = LHS_LOAD_TILE, RHS_LOAD_TILE
    else:
        l_tile_k = tiles['l_tile_k']
        tile_m = tiles['tile_m']
        tile_n = tiles['tile_n']
        lhs_load_default = tiles['lhs_load_tile']
        rhs_load_default = tiles['rhs_load_tile']
    return BlockDescriptor(
        TILES_IN_BLOCK_M=tiles_in_block_m,
        TILES_IN_BLOCK_N=tiles_in_block_n,
        TILES_IN_BLOCK_K=tiles_in_block_k,
        lhs_matmul_tile_shape_logical=(l_tile_k, tile_m),
        rhs_matmul_tile_shape_logical=(l_tile_k, tile_n),
        lhs_load_tile_shape=lhs_load_tile_shape or lhs_load_default,
        rhs_load_tile_shape=rhs_load_tile_shape or rhs_load_default,
    )


def _compute_load_tile_shape(td: TensorDescriptor, tiles: dict, tile_f: int) -> tuple:
    """Compute load tile shape based on TensorDescriptor state.

    Mirrors the logic in matmul_mxfp8_generic_kernel._validate_and_calculate_shapes:
      - Pre-quantized: load tile = matmul tile physical (already x4)
      - Pre-swizzled BF16: load tile = (K_physical, F * 4)
      - Unswizzled BF16: load tile = (l_tile_k, F) logical, DGT transposes
    """
    if td.is_quantized:
        return (tiles['matmul_tile_k_physical'], tile_f)
    elif td.is_swizzled:
        return (tiles['matmul_tile_k_physical'], tile_f * INTERLEAVE_FACTOR)
    else:
        return (tiles['l_tile_k'], tile_f)


def _allocate_spill_buffer(
    num_k_blocks: int,
    num_f_blocks: int,
    block_f_logical: int,
    tiles_in_block_k: int,
    use_scale_packing: bool,
    data_buffer,
) -> TensorDescriptor:
    """Allocate and zero-initialize an HBM spill/reload buffer for one operand.

    Args:
        num_k_blocks: Number of K-blocks in the full tensor.
        num_f_blocks: Number of blocks in the F dimension (M for LHS, N for RHS).
        block_f_logical: Logical block size in F dimension (BLOCK_M_LOGICAL or BLOCK_N_LOGICAL).
        tiles_in_block_k: Number of K-tiles per block.
        use_scale_packing: Whether to use packed scale layout.
        data_buffer: HBM buffer type (nl.hbm or nl.private_hbm).

    Returns:
        TensorDescriptor wrapping the allocated data and scale buffers.
    """
    fp8_x4_dtype = get_fp8_dtype_x4("float8_e4m3fn")
    k_dim = num_k_blocks * MATMUL_TILE_K_PHYSICAL * tiles_in_block_k
    f_dim = num_f_blocks * block_f_logical
    logical_k = k_dim * INTERLEAVE_FACTOR
    scale_shape = get_scale_output_shape(logical_k, f_dim, MATMUL_TILE_K_PHYSICAL, use_scale_packing)

    return TensorDescriptor(
        data=nl.ndarray((k_dim, f_dim), dtype=fp8_x4_dtype, buffer=data_buffer),
        scales=nl.ndarray(scale_shape, dtype=nl.float8_e8m0fnu, buffer=data_buffer),
        is_swizzled=True,
        is_x4=True,
        scales_are_packed=use_scale_packing,
    )


def _store_unswizzled_sbuf_block_to_hbm(
    output_sbuf: nl.ndarray,
    dst_hbm: nl.ndarray,
    row_base: int,
    col_base: int,
    tiles_in_block_m: int,
    block_m: int,
    block_n: int,
    lhs_matmul_tile_m: int,
    block_idx_m: int,
    m_logical: int,
    n_logical: int,
) -> None:
    """DMA copy one unswizzled SBUF accumulator block to HBM with a row offset.

    This helper is only used for storing unswizzled intermediate data (e.g. bf16
    accumulator results) from SBUF to HBM. It should NOT be used for pre-swizzled
    or pre-quantized tensors, which have different physical layouts.

    Args:
        output_sbuf (nl.ndarray): SBUF accumulator with tiled layout.
        dst_hbm (nl.ndarray): Destination HBM tensor.
        row_base (int): Row offset into dst_hbm (e.g. s_base or h_base for LNC sharding).
        col_base (int): Column offset into dst_hbm.
        tiles_in_block_m (int): Number of M-tiles in the block.
        block_m (int): Total M-dimension size of the block (tiles_in_block_m * TILE_M).
        block_n (int): Total N-dimension size of the block (tiles_in_block_n * TILE_N).
        lhs_matmul_tile_m (int): M-dimension size of a single matmul tile.
        block_idx_m (int): Block index along the M dimension.
        m_logical (int): Logical M dimension. Clamps the store height to avoid
            writing past the logical boundary.
        n_logical (int): Logical N dimension. Clamps the store width to avoid
            writing past the logical boundary.

    Returns:
        None.

    Notes:
        TODO: Reuse this helper in the generic matmul API (matmul_mxfp8_generic_api.py).
    """
    actual_n = min(block_n, n_logical - col_base)
    sbuf_step_p = tiles_in_block_m * block_n
    for tile_idx_m in range(tiles_in_block_m):
        output_idx_m = row_base + block_idx_m * block_m + tile_idx_m * lhs_matmul_tile_m
        sbuf_offset = tile_idx_m * block_n
        actual_m = min(lhs_matmul_tile_m, m_logical - (block_idx_m * block_m + tile_idx_m * lhs_matmul_tile_m))
        if actual_m > 0 and actual_n > 0:
            nisa.dma_copy(
                dst=dst_hbm[output_idx_m : output_idx_m + actual_m, col_base : col_base + actual_n],
                src=output_sbuf.ap(
                    pattern=[[sbuf_step_p, actual_m], [1, actual_n]],
                    offset=sbuf_offset,
                ),
            )


def apply_gradient_clamp(gradient, activation, upper_limit, lower_limit, dtype):
    """Zero out gradient elements where the forward activation is outside [lower, upper].

    No-op if both limits are None. Operates entirely in SBUF.
    Follows the same pattern as the MOE kernel (bwmm_bwd_dropless.py).

    Args:
        gradient: SBUF tensor to clamp in-place (e.g. d_gate or d_up).
        activation: SBUF tensor to check against limits (forward checkpoint).
        upper_limit: float or None — zero gradient where activation >= upper.
        lower_limit: float or None — zero gradient where activation <= lower.
        dtype: compute dtype (e.g. nl.bfloat16).
    """
    if upper_limit is None and lower_limit is None:
        return

    sbm = get_active_sbm()
    shape = gradient.shape

    mask1 = sbm.alloc_stack(shape=shape, dtype=dtype, buffer=nl.sbuf)
    mask2 = sbm.alloc_stack(shape=shape, dtype=dtype, buffer=nl.sbuf)
    nisa.memset(mask1, value=1.0)
    nisa.memset(mask2, value=1.0)

    if upper_limit is not None:
        nisa.tensor_scalar(dst=mask1, data=activation, op0=nl.less, operand0=upper_limit)

    if lower_limit is not None:
        nisa.tensor_scalar(dst=mask2, data=activation, op0=nl.greater, operand0=lower_limit)

    nisa.tensor_tensor(dst=mask1, data1=mask1, data2=mask2, op=nl.logical_and)
    nisa.tensor_tensor(dst=gradient, op=nl.multiply, data1=gradient, data2=mask1)


def apply_activation_clamp(tensor, upper_limit, lower_limit):
    """Clamp activation values in-place using tensor_scalar min/max.

    Mirrors the MoE forward kernel (bwmm_shard_on_I.py) clamping pattern.
    No-op if both limits are None.

    Args:
        tensor: SBUF tensor to clamp in-place.
        upper_limit: float or None — clamp values above this.
        lower_limit: float or None — clamp values below this.
    """
    if upper_limit is None and lower_limit is None:
        return

    if upper_limit is not None and lower_limit is not None:
        nisa.tensor_scalar(
            data=tensor, op0=nl.minimum, operand0=upper_limit, op1=nl.maximum, operand1=lower_limit, dst=tensor
        )
    elif upper_limit is not None:
        nisa.tensor_scalar(data=tensor, op0=nl.minimum, operand0=upper_limit, dst=tensor)
    else:
        nisa.tensor_scalar(data=tensor, op0=nl.maximum, operand0=lower_limit, dst=tensor)


# DMA transpose tile dimensions: load (FREE_DIM, PAR_DIM) from HBM, produce (PAR_DIM, FREE_DIM) in SBUF
_TRANSPOSE_PAR_DIM = 128
_TRANSPOSE_FREE_DIM = 512


def hbm_dma_transpose(
    src_hbm, dst_hbm, M=None, N=None, src_row_offset=0, src_col_offset=0, dst_row_offset=0, dst_col_offset=0
):
    """Transpose a sub-region of src_hbm into a sub-region of dst_hbm.

    Reads src_hbm[src_row_offset : src_row_offset+M, src_col_offset : src_col_offset+N]
    and writes the transpose to
    dst_hbm[dst_row_offset : dst_row_offset+N, dst_col_offset : dst_col_offset+M].

    Tiles into (_TRANSPOSE_FREE_DIM, _TRANSPOSE_PAR_DIM) = (512, 128) chunks,
    transposes each via nisa.dma_transpose into SBUF, then copies out.

    TODO: Currently, the function doesn't assume src_hbm and dst_hbm are logical
    transposes of each other i.e. src_hbm.shape = dst_hbm.T.shape. Assess whether
    this is necessary.

    Args:
        src_hbm: Source HBM tensor (full, unsliced).
        dst_hbm: Destination HBM tensor (full, unsliced).
        M: Number of rows to transpose from src. Defaults to src rows - src_row_offset.
        N: Number of cols to transpose from src. Defaults to src cols - src_col_offset.
        src_row_offset: Starting row in src.
        src_col_offset: Starting col in src.
        dst_row_offset: Starting row in dst for output.
        dst_col_offset: Starting col in dst for output.
    """
    src_full_rows, src_full_cols = src_hbm.shape
    dst_full_rows, dst_full_cols = dst_hbm.shape
    sbm = get_active_sbm()

    if M is None:
        M = src_full_rows - src_row_offset
    if N is None:
        N = src_full_cols - src_col_offset

    kernel_assert(M > 0, f"M ({M}) must be positive")
    kernel_assert(N > 0, f"N ({N}) must be positive")
    kernel_assert(
        src_row_offset + M <= src_full_rows,
        f"src_row_offset ({src_row_offset}) + M ({M}) exceeds src rows ({src_full_rows})",
    )
    kernel_assert(
        src_col_offset + N <= src_full_cols,
        f"src_col_offset ({src_col_offset}) + N ({N}) exceeds src cols ({src_full_cols})",
    )
    kernel_assert(
        dst_row_offset + N <= dst_full_rows,
        f"dst_row_offset ({dst_row_offset}) + N ({N}) exceeds dst rows ({dst_full_rows})",
    )
    kernel_assert(
        dst_col_offset + M <= dst_full_cols,
        f"dst_col_offset ({dst_col_offset}) + M ({M}) exceeds dst cols ({dst_full_cols})",
    )

    num_row_tiles = div_ceil(M, _TRANSPOSE_FREE_DIM)
    num_col_tiles = div_ceil(N, _TRANSPOSE_PAR_DIM)

    for row_idx in nl.affine_range(num_row_tiles):
        row_start = row_idx * _TRANSPOSE_FREE_DIM
        row_size = min(_TRANSPOSE_FREE_DIM, M - row_start)

        for col_idx in nl.affine_range(num_col_tiles):
            col_start = col_idx * _TRANSPOSE_PAR_DIM
            col_size = min(_TRANSPOSE_PAR_DIM, N - col_start)

            sbuf_tile = sbm.alloc_stack(shape=(col_size, row_size), dtype=src_hbm.dtype, buffer=nl.sbuf)

            src_r = src_row_offset + row_start
            src_c = src_col_offset + col_start
            src_ap_offset = src_r * src_full_cols + src_c
            nisa.dma_transpose(
                dst=sbuf_tile,
                src=src_hbm.ap(
                    pattern=[[src_full_cols, row_size], [1, col_size]],
                    offset=src_ap_offset,
                ),
            )

            dst_r = dst_row_offset + col_start
            dst_c = dst_col_offset + row_start
            dst_ap_offset = dst_r * dst_full_cols + dst_c
            nisa.dma_copy(
                dst=dst_hbm.ap(
                    pattern=[[dst_full_cols, col_size], [1, row_size]],
                    offset=dst_ap_offset,
                ),
                src=sbuf_tile,
            )
