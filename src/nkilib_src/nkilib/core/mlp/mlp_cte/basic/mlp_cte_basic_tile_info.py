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

"""MLP CTE basic tiling information for non-MX quantization types."""

from dataclasses import dataclass

import nki.language as nl

from ....utils.allocator import sizeinbytes
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, get_ceil_aligned_size
from ....utils.tile_info import TiledDimInfo
from ...mlp_parameters import (
    MLPParameters,
    mlpp_has_layer_normalization,
    mlpp_has_rms_normalization,
    mlpp_has_up_projection_bias,
)
from ..mlp_cte_constants import BN_AGGR_ELEMENTS_PER_TILE, BN_STATS_ELEMENTS_PER_TILE, MLPCTEConstants
from ..mlp_cte_sharding import DimShard, ShardedDim, is_sharded_dim_bxs

_xpose_hidden_dim_tile_size = 512
_layer_norm_hidden_dim_tile_size = 512
_xpose_intermediate_dim_tile_size = 512
_down_proj_intermediate_dim_tile_size = 128

# Max FMAX-wide slices one weight DMA covers along the contiguous HBM dim (longer run per descriptor);
# also bounded at runtime by the PSUM banks the BxS subtiles leave free.
_max_weight_dma_slice_count = 4


####################################
# Weight DMA Tiling Info Functions #
####################################


def calc_weight_dma_slice_count(bxs_dim_tile: TiledDimInfo) -> int:
    """Number of FMAX-wide slices one weight DMA is allowed to cover.

    The matmuls consume the loaded tile one FMAX-wide slice at a time, and every (BxS subtile, slice)
    pair accumulates into its own PSUM bank, so the BxS subtiles of a BxS tile and the slices of a
    weight DMA share the PSUM banks.
    """
    return min(
        _max_weight_dma_slice_count,
        max(1, NUM_HW_PSUM_BANKS // bxs_dim_tile.subtile_dim_info.tile_count),
    )


def build_src_proj_weights_int_dma_tile(intermediate_size: int, bxs_dim_tile: TiledDimInfo) -> TiledDimInfo:
    """Tiling of I for one gate/up weight DMA, subtiled by the FMAX-wide slices the matmuls consume.

    The gate and up weights are contiguous along I in HBM, so the I extent of the load is what sets
    how much contiguous HBM one descriptor covers. H stays tiled at the partition dim size.
    """
    FMAX = nl.tile_size.gemm_moving_fmax
    dma_tile_size = min(
        FMAX * calc_weight_dma_slice_count(bxs_dim_tile),
        get_ceil_aligned_size(intermediate_size, FMAX),
    )
    return TiledDimInfo.build_with_subtiling(intermediate_size, dma_tile_size, FMAX)


def build_down_proj_weights_hidden_dma_tile(hidden_size: int, bxs_dim_tile: TiledDimInfo) -> TiledDimInfo:
    """Tiling of H for one down weight DMA, subtiled by the FMAX-wide slices the matmuls consume.

    The down weights are contiguous along H in HBM. I is contracted over, so it has to stay tiled at
    the partition dim size and the load remains 2D.
    """
    FMAX = nl.tile_size.gemm_moving_fmax
    dma_tile_size = min(
        FMAX * calc_weight_dma_slice_count(bxs_dim_tile),
        get_ceil_aligned_size(hidden_size, FMAX),
    )
    return TiledDimInfo.build_with_subtiling(hidden_size, dma_tile_size, FMAX)


def get_src_proj_weights_ring_buffer_count(
    constants: MLPCTEConstants,
    src_proj_weights_int_dma_tile: TiledDimInfo,
) -> int:
    """Planning (footprint) count of the gate/up weight ring, kept footprint-neutral (scaled down as
    the buffers widen). Used only to size the BxS (T) tile via calc_up_proj_footprint, so widening
    or deepening the ring never inflates the T-sizing model and shrinks T. The real allocation depth
    is get_src_proj_weights_alloc_ring_buffer_count, filled from leftover SBUF."""
    return max(
        constants.min_weights_ring_buffer_count,
        constants.src_proj_weights_max_buffer_count // src_proj_weights_int_dma_tile.subtile_dim_info.tile_count,
    )


def get_src_proj_weights_alloc_ring_buffer_count(
    constants: MLPCTEConstants,
    src_proj_weights_int_dma_tile: TiledDimInfo,
) -> int:
    """Allocation-target depth of the gate/up weight ring. Prefetch depth is decoupled from DMA
    width so the matmuls overlap several in-flight transfers instead of stalling on each wide one.
    The allocator clamps this to the free SBUF left after the T-sized activations are placed
    (allocate_src_projection_weights), so it spends only leftover SBUF and never shrinks T."""
    return constants.src_proj_weights_max_buffer_count


def get_down_proj_weights_ring_buffer_count(
    constants: MLPCTEConstants,
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
) -> int:
    """Planning (footprint) count of the down weight ring, kept footprint-neutral. Used only for T
    sizing via calc_down_proj_footprint. Real allocation depth is
    get_down_proj_weights_alloc_ring_buffer_count."""
    return max(
        constants.min_weights_ring_buffer_count,
        constants.down_proj_weights_buffer_count // down_proj_weights_hidden_dma_tile.subtile_dim_info.tile_count,
    )


def get_down_proj_weights_alloc_ring_buffer_count(
    constants: MLPCTEConstants,
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
) -> int:
    """Allocation-target depth of the down weight ring. Depth decoupled from DMA width; clamped to
    free SBUF by allocate_down_projection_weights so it spends only leftover SBUF."""
    return constants.down_proj_weights_buffer_count


def get_down_proj_weights_row_scales_ring_buffer_count(
    constants: MLPCTEConstants,
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
) -> int:
    """Buffer count of the down weight row scales ring, scaled down by how wide each buffer got."""
    scales_buffer_count = max(
        constants.min_weights_scales_ring_buffer_count,
        constants.down_proj_weights_scales_buffer_count
        // down_proj_weights_hidden_dma_tile.subtile_dim_info.tile_count,
    )
    return min(scales_buffer_count, down_proj_weights_hidden_dma_tile.tile_count)


#############################################
# Weights SBUF Tensor Info Getter Functions #
#############################################


def get_gate_up_weights_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    src_proj_hidden_dim_tile: TiledDimInfo,
    src_proj_weights_int_dma_tile: TiledDimInfo,
) -> tuple:
    I_DMA_TILE_SIZE = src_proj_weights_int_dma_tile.tile_size
    if mlp_params.quant_params.is_quant():
        return (
            src_proj_hidden_dim_tile.tile_size,
            2 * I_DMA_TILE_SIZE,
        ), constants.src_proj_quant_data_type
    else:
        return (
            src_proj_hidden_dim_tile.tile_size,
            I_DMA_TILE_SIZE,
        ), constants.compute_data_type


def get_down_weights_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
    down_proj_intermediate_dim_tile: TiledDimInfo,
) -> tuple:
    if mlp_params.quant_params.is_quant():
        return (
            down_proj_intermediate_dim_tile.tile_size,
            2 * down_proj_weights_hidden_dma_tile.tile_size,
        ), constants.down_proj_quant_data_type
    else:
        return (
            down_proj_intermediate_dim_tile.tile_size,
            down_proj_weights_hidden_dma_tile.tile_size,
        ), constants.compute_data_type


def get_down_weights_row_scales_tensor_info(
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
) -> tuple:
    return (nl.tile_size.pmax, down_proj_weights_hidden_dma_tile.tile_size), nl.float32


#################################################
# Activations SBUF Tensor Info Getter Functions #
#################################################


def get_hidden_tile_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
) -> tuple:
    tensor_shape = (
        bxs_dim_tile.subtile_dim_info.tile_size,
        mlp_params.hidden_size,
    )
    return tensor_shape, constants.hidden_tile_data_type


def get_intermediate_tile_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
) -> tuple:
    I_512 = get_ceil_aligned_size(mlp_params.intermediate_size, nl.tile_size.gemm_moving_fmax)
    tensor_shape = (
        bxs_dim_tile.subtile_dim_info.tile_size,
        I_512,
    )
    return tensor_shape, constants.compute_data_type


def get_output_tile_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
) -> tuple:
    tensor_shape = (
        bxs_dim_tile.subtile_dim_info.tile_size,
        mlp_params.hidden_size,
    )
    return tensor_shape, constants.compute_data_type


##########################################
# SBUF Utilization Calculation Functions #
##########################################


def footprint_B(shape, dtype) -> int:
    """Calculate the size of a tensor along the free dimension in bytes.

    Args:
        shape: Shape of the tensor
        dtype: Dtype of the tensor

    Returns:
        Footprint of the tensor in bytes
    """
    footprint = sizeinbytes(dtype)
    for dim_size in shape[1:]:
        footprint *= dim_size
    return footprint


def calc_normalization_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
    layer_norm_hidden_dim_tile: TiledDimInfo,
) -> int:
    """Footprint while normalizing the hidden tile: the live activation tiles plus norm scratch space."""
    footprint = 0
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count

    # hidden tensor tile
    hiddens_shape, hiddens_dtype = get_hidden_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(hiddens_shape, hiddens_dtype) * BXS_SUBTILE_COUNT

    # intermediate tensor tile
    intermediates_shape, intermediates_dtype = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(intermediates_shape, intermediates_dtype) * BXS_SUBTILE_COUNT

    # norm scratch space
    if mlpp_has_rms_normalization(mlp_params):
        footprint += footprint_B((BXS_SUBTILE_SIZE, mlp_params.hidden_size), constants.activation_data_type)
        footprint += footprint_B((BXS_SUBTILE_SIZE, 1), constants.activation_data_type) * BXS_SUBTILE_COUNT
    elif mlpp_has_layer_normalization(mlp_params):
        footprint += footprint_B(
            (BXS_SUBTILE_SIZE, layer_norm_hidden_dim_tile.tile_count * BN_STATS_ELEMENTS_PER_TILE),
            constants.activation_data_type,
        )
        footprint += (
            footprint_B((BXS_SUBTILE_SIZE, BN_AGGR_ELEMENTS_PER_TILE), constants.activation_data_type)
            * BXS_SUBTILE_COUNT
        )

    return footprint


def calc_up_proj_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
    src_proj_hidden_dim_tile: TiledDimInfo,
    src_proj_weights_int_dma_tile: TiledDimInfo,
    include_src_proj_weights: bool = True,
) -> int:
    footprint = 0

    # weights buffers (accounted for separately when they are resident for the whole compute)
    if include_src_proj_weights:
        weights_shape, weights_dtype = get_gate_up_weights_tensor_info(
            mlp_params,
            constants,
            src_proj_hidden_dim_tile,
            src_proj_weights_int_dma_tile,
        )
        footprint += footprint_B(weights_shape, weights_dtype) * get_src_proj_weights_ring_buffer_count(
            constants,
            src_proj_weights_int_dma_tile,
        )

    # hidden tensor tile
    hiddens_shape, hiddens_dtype = get_hidden_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(hiddens_shape, hiddens_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    # intermediate tensor tile
    intermediates_shape, intermediates_dtype = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(intermediates_shape, intermediates_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    # up projection bias buffers
    if mlpp_has_up_projection_bias(mlp_params):
        footprint += footprint_B(intermediates_shape, intermediates_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    return footprint


def calc_quantization_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
) -> int:
    footprint = 0

    # src proj result
    src_proj_res_shape, src_proj_res_dtype = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(src_proj_res_shape, src_proj_res_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    # intermediate tensor
    src_proj_res_shape, _ = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += (
        footprint_B(src_proj_res_shape, constants.down_proj_quant_data_type) * bxs_dim_tile.subtile_dim_info.tile_count
    )

    # row quant reduction result
    if mlp_params.quant_params.is_quant_row():
        red_res_shape, red_res_dtype = get_intermediate_tile_tensor_info(
            mlp_params,
            constants,
            bxs_dim_tile,
        )
        footprint += footprint_B(red_res_shape, red_res_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    # output tensor tile
    outputs_shape, outputs_dtype = get_output_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(outputs_shape, outputs_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    return footprint


def calc_down_proj_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
    down_proj_intermediate_dim_tile: TiledDimInfo,
) -> int:
    footprint = 0

    # weights buffers
    weights_shape, weights_dtype = get_down_weights_tensor_info(
        mlp_params,
        constants,
        down_proj_weights_hidden_dma_tile,
        down_proj_intermediate_dim_tile,
    )
    footprint += footprint_B(weights_shape, weights_dtype) * get_down_proj_weights_ring_buffer_count(
        constants,
        down_proj_weights_hidden_dma_tile,
    )

    # intermediate tensor tile (with possible quantized dtype)
    intermediates_shape, _ = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(intermediates_shape, weights_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    # output tensor tile
    outputs_shape, outputs_dtype = get_output_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(outputs_shape, outputs_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    return footprint


def calc_sendrecv_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
) -> int:
    footprint = 0

    # output tensor tile
    outputs_shape, outputs_dtype = get_output_tile_tensor_info(
        mlp_params,
        constants,
        bxs_dim_tile,
    )
    footprint += footprint_B(outputs_shape, outputs_dtype) * bxs_dim_tile.subtile_dim_info.tile_count

    # other core sendrecv buffer
    if constants.sharded_dim == ShardedDim.INTERMEDIATE:
        footprint += footprint_B(outputs_shape, outputs_dtype) * bxs_dim_tile.subtile_dim_info.tile_count // 2

    return footprint


def calc_peak_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    bxs_dim_tile: TiledDimInfo,
    src_proj_hidden_dim_tile: TiledDimInfo,
    src_proj_weights_int_dma_tile: TiledDimInfo,
    down_proj_weights_hidden_dma_tile: TiledDimInfo,
    down_proj_intermediate_dim_tile: TiledDimInfo,
    include_src_proj_weights: bool = True,
) -> int:
    """SBUF footprint at the most space-intensive spots in the compute for a given BxS tile size:
    1. Up proj
    2. Quantization
    3. Down proj
    4. Post-down sendrecv (if I-sharded)
    """
    footprint = calc_up_proj_footprint(
        mlp_params,
        constants,
        bxs_dim_tile,
        src_proj_hidden_dim_tile,
        src_proj_weights_int_dma_tile,
        include_src_proj_weights,
    )

    if mlp_params.quant_params.is_quant():
        footprint = max(
            footprint,
            calc_quantization_footprint(
                mlp_params,
                constants,
                bxs_dim_tile,
            ),
        )

    footprint = max(
        footprint,
        calc_down_proj_footprint(
            mlp_params,
            constants,
            bxs_dim_tile,
            down_proj_weights_hidden_dma_tile,
            down_proj_intermediate_dim_tile,
        ),
    )

    footprint = max(
        footprint,
        calc_sendrecv_footprint(
            mlp_params,
            constants,
            bxs_dim_tile,
        ),
    )

    return footprint


def calc_sbuf_bound_bxs_tile_size(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    sbuf_free_space: int,
    bxs_dim_size: int,
    bxs_subtile_size: int,
    intermediate_size: int,
    src_proj_hidden_dim_tile: TiledDimInfo,
    down_proj_intermediate_dim_tile: TiledDimInfo,
) -> int:
    """
    We want to maximize the BxS tile size because for each BxS tile, we have to load each weight tensor
    in its entirety into SBUF. Therefore, in most cases, higher BxS tile size means higher arithmetic
    intensity. There are two limitations on the BxS tile size:
        - The PSUM Limit: This is a constant 1024. Say we load a FMAX-sized (512) tile along I of the
                          up weight tensor. Then, we multiply it by no more than 8 BxS tiles of size
                          128 before we run out of PSUM space.
        - The SBUF Limit: As BxS tile size increases, so does the memory footprint of our activation
                          tensors. This limit must be calculated dynamically.
    Therefore, this function uses a slightly simplified model of SBUF utilization during the compute and
    uses it to find the maximum BxS tile that is within both constraints.
    """
    BXS_TILE_SIZE_PSUM_LIMIT = 1024

    # set the initial tile size back one step because we do our update at the top of the loop
    cur_bxs_tile_size = bxs_subtile_size // 2
    within_sbuf_bounds = True
    while within_sbuf_bounds and cur_bxs_tile_size // 2 < bxs_dim_size:
        # Create a candidate bxs_dim_tile object with the current tile size and use it to
        # probe the SBUF utilization at the most space-intensive spots in the compute
        cur_bxs_tile_size *= 2
        bxs_dim_tile = TiledDimInfo.build_with_subtiling(bxs_dim_size, cur_bxs_tile_size, bxs_subtile_size)

        peak_footprint = calc_peak_footprint(
            mlp_params,
            constants,
            bxs_dim_tile,
            src_proj_hidden_dim_tile,
            build_src_proj_weights_int_dma_tile(intermediate_size, bxs_dim_tile),
            build_down_proj_weights_hidden_dma_tile(mlp_params.hidden_size, bxs_dim_tile),
            down_proj_intermediate_dim_tile,
        )
        within_sbuf_bounds = sbuf_free_space - peak_footprint > 0

    cur_bxs_tile_size = max(cur_bxs_tile_size // 2, bxs_subtile_size)
    cur_bxs_tile_size = min(cur_bxs_tile_size, BXS_TILE_SIZE_PSUM_LIMIT)
    return cur_bxs_tile_size


#############################
# TileInfo Object Utilities #
#############################


@dataclass
class MLPCTEBasicTileInfo(nl.NKIObject):
    bxs_dim_tile: TiledDimInfo
    layer_norm_hidden_dim_tile: TiledDimInfo
    xpose_hidden_dim_tile: TiledDimInfo
    src_proj_hidden_dim_tile: TiledDimInfo
    src_proj_intermediate_dim_tile: TiledDimInfo
    xpose_intermediate_dim_tile: TiledDimInfo
    down_proj_intermediate_dim_tile: TiledDimInfo
    # I extent of one gate/up weight DMA, subtiled by the I tiles the matmuls consume
    src_proj_weights_int_dma_tile: TiledDimInfo
    # H extent of one down weight DMA, subtiled by the H tiles the matmuls consume
    down_proj_weights_hidden_dma_tile: TiledDimInfo


def build_mlp_cte_basic_tile_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    sbuf_free_space: int,
    sharded_dim: ShardedDim,
    dim_shard: DimShard = None,
) -> MLPCTEBasicTileInfo:
    bxs_dim_size = (
        dim_shard.dim_size if is_sharded_dim_bxs(sharded_dim) else mlp_params.batch_size * mlp_params.sequence_len
    )
    intermediate_size = dim_shard.dim_size if sharded_dim == ShardedDim.INTERMEDIATE else mlp_params.intermediate_size

    layer_norm_hidden_dim_tile = TiledDimInfo.build(mlp_params.hidden_size, _layer_norm_hidden_dim_tile_size)
    xpose_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size, _xpose_hidden_dim_tile_size, nl.tile_size.pmax
    )
    src_proj_hidden_dim_tile = TiledDimInfo.build(mlp_params.hidden_size, nl.tile_size.pmax)
    src_proj_intermediate_dim_tile = TiledDimInfo.build(intermediate_size, nl.tile_size.gemm_moving_fmax)
    xpose_intermediate_dim_tile = TiledDimInfo.build_with_subtiling(
        intermediate_size, _xpose_intermediate_dim_tile_size, nl.tile_size.pmax
    )
    down_proj_intermediate_dim_tile = TiledDimInfo.build(intermediate_size, _down_proj_intermediate_dim_tile_size)

    bxs_dim_subtile_size = nl.tile_size.pmax
    bxs_dim_tile_size = calc_sbuf_bound_bxs_tile_size(
        mlp_params,
        constants,
        sbuf_free_space,
        bxs_dim_size,
        bxs_dim_subtile_size,
        intermediate_size,
        src_proj_hidden_dim_tile,
        down_proj_intermediate_dim_tile,
    )
    bxs_dim_tile = TiledDimInfo.build_with_subtiling(bxs_dim_size, bxs_dim_tile_size, bxs_dim_subtile_size)
    src_proj_weights_int_dma_tile = build_src_proj_weights_int_dma_tile(intermediate_size, bxs_dim_tile)
    down_proj_weights_hidden_dma_tile = build_down_proj_weights_hidden_dma_tile(mlp_params.hidden_size, bxs_dim_tile)

    return MLPCTEBasicTileInfo(
        bxs_dim_tile=bxs_dim_tile,
        layer_norm_hidden_dim_tile=layer_norm_hidden_dim_tile,
        xpose_hidden_dim_tile=xpose_hidden_dim_tile,
        src_proj_hidden_dim_tile=src_proj_hidden_dim_tile,
        src_proj_intermediate_dim_tile=src_proj_intermediate_dim_tile,
        xpose_intermediate_dim_tile=xpose_intermediate_dim_tile,
        down_proj_intermediate_dim_tile=down_proj_intermediate_dim_tile,
        src_proj_weights_int_dma_tile=src_proj_weights_int_dma_tile,
        down_proj_weights_hidden_dma_tile=down_proj_weights_hidden_dma_tile,
    )
