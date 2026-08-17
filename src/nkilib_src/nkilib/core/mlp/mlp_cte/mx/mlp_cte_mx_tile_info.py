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

"""MLP CTE MX tiling information for MX, STATIC_MX, and ROW_MX quantization types."""

import math
from dataclasses import dataclass

import nki.language as nl
from nki.language import NKIObject

from ....utils.allocator import sizeinbytes
from ....utils.tile_info import TiledDimInfo
from ...mlp_parameters import MLPParameters
from ..mlp_cte_constants import MLPCTEConstants
from ..mlp_cte_sharding import DimShard, ShardedDim, is_sharded_dim_bxs

_mx_q_width = 4
_down_proj_hidden_dim_tile_size = 1024


#############################################
# Weights SBUF Tensor Info Getter Functions #
#############################################


def get_gate_up_weights_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    src_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
) -> tuple:
    tensor_shape = (
        src_proj_hidden_dim_tile.subtile_dim_info.tile_count,  # 128
        intermediate_dim_tile.subtile_dim_info.tile_size,  # 4
        intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
        src_proj_hidden_dim_tile.subtile_dim_info.tile_size,  # 4
    )
    return tensor_shape, constants.src_proj_quant_data_type


def get_gate_up_scales_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    src_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
) -> tuple:
    if mlp_params.quant_params.is_quant_row_mx():
        return (
            intermediate_dim_tile.subtile_dim_info.tile_count,
            intermediate_dim_tile.tile_count,
            intermediate_dim_tile.subtile_dim_info.tile_size,
        ), nl.float32
    elif mlp_params.quant_params.is_quant_mx():
        NUM_SLOTS = 4
        return (
            src_proj_hidden_dim_tile.subtile_dim_info.tile_count,
            math.ceil(src_proj_hidden_dim_tile.tile_count / NUM_SLOTS),
            mlp_params.intermediate_size,
        ), nl.uint8


def get_down_weights_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
) -> tuple:
    tensor_shape = (
        intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
        down_proj_hidden_dim_tile.tile_size,  # 1024
        intermediate_dim_tile.subtile_dim_info.tile_size,  # 4
    )
    return tensor_shape, constants.down_proj_quant_data_type


def get_down_scales_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
) -> tuple:
    if mlp_params.quant_params.is_quant_row_mx():
        return (nl.tile_size.pmax, down_proj_hidden_dim_tile.tile_size), nl.float32
    else:
        return (
            intermediate_dim_tile.subtile_dim_info.tile_count,
            intermediate_dim_tile.tile_count,
            down_proj_hidden_dim_tile.tile_size,
        ), nl.uint8


#################################################
# Activations SBUF Tensor Info Getter Functions #
#################################################


def get_hidden_tile_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    src_proj_hidden_dim_tile: TiledDimInfo,
):
    tensor_shape = (
        nl.tile_size.pmax,  # 128_H
        src_proj_hidden_dim_tile.tile_count,  # H/512
        2 * nl.tile_size.pmax,  # 256_T
        src_proj_hidden_dim_tile.subtile_dim_info.tile_size,  # 4
    )
    return tensor_shape, constants.hidden_tile_data_type


def get_intermediate_tile_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_bxs_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
):
    tensor_shape = (
        intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
        intermediate_dim_tile.tile_count,  # I/512
        down_proj_bxs_dim_tile.subtile_dim_info.tile_size,  # 128
        intermediate_dim_tile.subtile_dim_info.tile_size,  # 4
    )
    return tensor_shape, constants.compute_data_type


def get_output_tile_tensor_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_bxs_dim_tile: TiledDimInfo,
) -> tuple:
    tensor_shape = (
        down_proj_bxs_dim_tile.subtile_dim_info.tile_size,
        mlp_params.hidden_size,
    )
    return tensor_shape, constants.compute_data_type


##########################################
# SBUF Utilization Calculation Functions #
##########################################


def footprint_B(shape, dtype):
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


def calc_up_proj_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    src_proj_bxs_dim_tile: TiledDimInfo,
    down_proj_bxs_dim_tile: TiledDimInfo,
    src_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
) -> int:
    footprint = 0

    # weights buffers
    weights_shape, weights_dtype = get_gate_up_weights_tensor_info(
        mlp_params,
        constants,
        src_proj_hidden_dim_tile,
        intermediate_dim_tile,
    )
    footprint += footprint_B(weights_shape, weights_dtype) * constants.src_proj_weights_max_buffer_count

    # src proj scales buffers
    if mlp_params.quant_params.is_quant_row_mx() or mlp_params.quant_params.is_quant_mx():
        scales_shape, scales_dtype = get_gate_up_scales_tensor_info(
            mlp_params,
            constants,
            src_proj_hidden_dim_tile,
            intermediate_dim_tile,
        )
        num_scales_buffers = 1 if mlp_params.skip_gate_proj else 2
        footprint += footprint_B(scales_shape, scales_dtype) * num_scales_buffers

    # hidden tensor tile
    hiddens_shape, hiddens_dtype = get_hidden_tile_tensor_info(
        mlp_params,
        constants,
        src_proj_hidden_dim_tile,
    )
    footprint += footprint_B(hiddens_shape, hiddens_dtype) * src_proj_bxs_dim_tile.subtile_dim_info.tile_count

    # intermediate tensor tile
    intermediates_shape, intermediates_dtype = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        down_proj_bxs_dim_tile,
        intermediate_dim_tile,
    )
    footprint += (
        footprint_B(intermediates_shape, intermediates_dtype) * down_proj_bxs_dim_tile.subtile_dim_info.tile_count
    )

    return footprint


def calc_down_proj_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_bxs_dim_tile: TiledDimInfo,
    src_proj_hidden_dim_tile: TiledDimInfo,
    down_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
):
    footprint = 0

    # weights buffers
    weights_shape, weights_dtype = get_down_weights_tensor_info(
        mlp_params,
        constants,
        down_proj_hidden_dim_tile,
        intermediate_dim_tile,
    )
    footprint += footprint_B(weights_shape, weights_dtype) * constants.down_proj_weights_buffer_count

    # scales buffers
    if mlp_params.quant_params.is_quant_row_mx() or mlp_params.quant_params.is_quant_mx():
        scales_shape, scales_dtype = get_gate_up_scales_tensor_info(
            mlp_params,
            constants,
            src_proj_hidden_dim_tile,
            intermediate_dim_tile,
        )
        num_scales_buffers = 1 if mlp_params.skip_gate_proj else 2
        footprint += footprint_B(scales_shape, scales_dtype) * num_scales_buffers
        scales_shape, scales_dtype = get_down_scales_tensor_info(
            mlp_params,
            constants,
            down_proj_hidden_dim_tile,
            intermediate_dim_tile,
        )
        footprint += footprint_B(scales_shape, scales_dtype) * constants.down_proj_weights_scales_buffer_count

    # intermediate tensor tile (with quantized dtype)
    intermediates_shape, _ = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        down_proj_bxs_dim_tile,
        intermediate_dim_tile,
    )
    footprint += footprint_B(intermediates_shape, weights_dtype) * down_proj_bxs_dim_tile.subtile_dim_info.tile_count

    # output tensor tile
    outputs_shape, outputs_dtype = get_output_tile_tensor_info(
        mlp_params,
        constants,
        down_proj_bxs_dim_tile,
    )
    footprint += footprint_B(outputs_shape, outputs_dtype) * down_proj_bxs_dim_tile.subtile_dim_info.tile_count

    return footprint


def calc_sendrecv_footprint(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    down_proj_bxs_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
):
    footprint = 0

    # intermediate tensor tile (with quantized dtype)
    intermediates_shape, _ = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        down_proj_bxs_dim_tile,
        intermediate_dim_tile,
    )
    footprint += (
        footprint_B(intermediates_shape, constants.down_proj_quant_data_type)
        * down_proj_bxs_dim_tile.subtile_dim_info.tile_count
    )

    # output tensor tile
    outputs_shape, outputs_dtype = get_output_tile_tensor_info(
        mlp_params,
        constants,
        down_proj_bxs_dim_tile,
    )
    footprint += footprint_B(outputs_shape, outputs_dtype) * down_proj_bxs_dim_tile.subtile_dim_info.tile_count

    # other core sendrecv buffer
    if constants.sharded_dim == ShardedDim.INTERMEDIATE:
        footprint += footprint_B(outputs_shape, outputs_dtype) * down_proj_bxs_dim_tile.subtile_dim_info.tile_count // 2

    return footprint


def calc_sbuf_bound_bxs_tile_size(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    sbuf_free_space: int,
    bxs_dim_size: int,
    src_proj_bxs_subtile_size: int,
    src_proj_hidden_dim_tile: TiledDimInfo,
    intermediate_dim_tile: TiledDimInfo,
    down_proj_hidden_dim_tile: TiledDimInfo,
) -> int:
    BXS_TILE_SIZE_PSUM_LIMIT = 1024

    # set the initial tile size back one step because we do our update at the top of the loop
    cur_bxs_tile_size = src_proj_bxs_subtile_size // 2
    within_sbuf_bounds = True
    while within_sbuf_bounds and cur_bxs_tile_size // 2 < bxs_dim_size:
        # Create candidate src and down proj bxs_dim_tile object with the current tile size and
        # use it to probe the SBUF utilization at the three most space-intensive spots in the compute:
        #   1. Up proj
        #   2. Down proj
        #   3. Post-down sendrecv (if I-sharded)
        cur_bxs_tile_size *= 2
        src_proj_bxs_dim_tile = TiledDimInfo.build_with_subtiling(
            bxs_dim_size, cur_bxs_tile_size, src_proj_bxs_subtile_size
        )
        down_proj_bxs_dim_tile = TiledDimInfo.build_with_subtiling(
            bxs_dim_size, cur_bxs_tile_size, src_proj_bxs_subtile_size // 2
        )

        up_proj_footprint = calc_up_proj_footprint(
            mlp_params,
            constants,
            src_proj_bxs_dim_tile,
            down_proj_bxs_dim_tile,
            src_proj_hidden_dim_tile,
            intermediate_dim_tile,
        )
        within_sbuf_bounds = within_sbuf_bounds and sbuf_free_space - up_proj_footprint > 0

        down_proj_footprint = calc_down_proj_footprint(
            mlp_params,
            constants,
            down_proj_bxs_dim_tile,
            src_proj_hidden_dim_tile,
            down_proj_hidden_dim_tile,
            intermediate_dim_tile,
        )
        within_sbuf_bounds = within_sbuf_bounds and sbuf_free_space - down_proj_footprint > 0

        sendrecv_footprint = calc_sendrecv_footprint(
            mlp_params,
            constants,
            down_proj_bxs_dim_tile,
            intermediate_dim_tile,
        )
        within_sbuf_bounds = within_sbuf_bounds and sbuf_free_space - sendrecv_footprint > 0

    cur_bxs_tile_size = max(cur_bxs_tile_size // 2, src_proj_bxs_subtile_size)
    cur_bxs_tile_size = min(cur_bxs_tile_size, BXS_TILE_SIZE_PSUM_LIMIT)
    return cur_bxs_tile_size
    # return 512


#############################
# TileInfo Object Utilities #
#############################


@dataclass
class MLPCTEMXTileInfo(NKIObject):  # dim_size / tile_size / subtile_size
    src_proj_bxs_dim_tile: TiledDimInfo  # BxS / s / 256  (tile size s is variable)
    src_proj_hidden_dim_tile: TiledDimInfo  # H / 512 / 4
    intermediate_dim_tile: TiledDimInfo  # I / 512 / 4
    down_proj_bxs_dim_tile: TiledDimInfo  # BxS / s / 128
    down_proj_hidden_dim_tile: TiledDimInfo  # H / 1024 / 512


def build_mlp_cte_mx_tile_info(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    sbuf_free_space: int,
    sharded_dim: ShardedDim,
    dim_shard: DimShard = None,
) -> MLPCTEMXTileInfo:
    bxs_dim_size = (
        dim_shard.dim_size if is_sharded_dim_bxs(sharded_dim) else mlp_params.batch_size * mlp_params.sequence_len
    )
    intermediate_size = dim_shard.dim_size if sharded_dim == ShardedDim.INTERMEDIATE else mlp_params.intermediate_size

    src_proj_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size, nl.tile_size.pmax * _mx_q_width, _mx_q_width
    )
    intermediate_dim_tile = TiledDimInfo.build_with_subtiling(
        intermediate_size, nl.tile_size.pmax * _mx_q_width, _mx_q_width
    )
    down_proj_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size,
        _down_proj_hidden_dim_tile_size,
        nl.tile_size.gemm_moving_fmax,
    )

    # Gate/Up projection uses a wider BxS subtile (2*pmax); pad tile size to fit at least one full subtile
    src_proj_bxs_subtile_size = 2 * nl.tile_size.pmax
    bxs_dim_tile_size = calc_sbuf_bound_bxs_tile_size(
        mlp_params,
        constants,
        sbuf_free_space,
        bxs_dim_size,
        src_proj_bxs_subtile_size,
        src_proj_hidden_dim_tile,
        intermediate_dim_tile,
        down_proj_hidden_dim_tile,
    )
    src_proj_bxs_dim_tile = TiledDimInfo.build_with_subtiling(
        bxs_dim_size, bxs_dim_tile_size, src_proj_bxs_subtile_size
    )
    down_proj_bxs_dim_tile = TiledDimInfo.build_with_subtiling(
        bxs_dim_size, bxs_dim_tile_size, src_proj_bxs_subtile_size // 2
    )

    return MLPCTEMXTileInfo(
        src_proj_bxs_dim_tile=src_proj_bxs_dim_tile,
        src_proj_hidden_dim_tile=src_proj_hidden_dim_tile,
        intermediate_dim_tile=intermediate_dim_tile,
        down_proj_bxs_dim_tile=down_proj_bxs_dim_tile,
        down_proj_hidden_dim_tile=down_proj_hidden_dim_tile,
    )
