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

"""MLP CTE MX quantization functions for STATIC_MX and ROW_MX quantization types."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.kernel_helpers import get_max_positive_value_for_dtype
from ....utils.tiled_range import TiledRange
from ...mlp_parameters import MLPParameters
from ..mlp_cte_constants import MLPCTEConstants
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo


def perform_intermediate_quantization(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    src_proj_res_sbuf_list: list[nl.NkiTensor],
    quantized_output_sbuf_list: list[nl.NkiTensor],
    dequant_scales_output_sbuf_list: Optional[nl.NkiTensor],
    static_input_quant_scale_sbuf: Optional[nl.NkiTensor],
    sbm: SbufManager,
):
    if mlp_params.quant_params.is_logical_quant_static():
        _perform_intermediate_static_quantization(
            mlp_params,
            tile_info,
            constants,
            bxs_tile_idx,
            src_proj_res_sbuf_list,
            quantized_output_sbuf_list,
            static_input_quant_scale_sbuf,
        )
    elif mlp_params.quant_params.is_quant_row_mx() or mlp_params.quant_params.is_quant_mx():
        _perform_intermediate_mx_quantization(
            mlp_params,
            tile_info,
            constants,
            bxs_tile_idx,
            src_proj_res_sbuf_list,
            quantized_output_sbuf_list,
            dequant_scales_output_sbuf_list,
        )


def _perform_intermediate_static_quantization(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    src_proj_res_sbuf_list: list[nl.NkiTensor],
    quantized_output_sbuf_list: list[nl.NkiTensor],
    static_input_quant_scale_sbuf: Optional[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    max_pos_val = get_max_positive_value_for_dtype(constants.down_proj_quant_data_type)

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    buffer_shape = (
        int_dim_tile.subtile_dim_info.tile_count,  # 128
        int_dim_tile.tile_count,  # I/512
        BXS_SUBTILE_SIZE * int_dim_tile.subtile_dim_info.tile_size,  # 128 * 4
    )

    for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
        src_proj_res_sbuf_view = src_proj_res_sbuf_list[bxs_subtile.index].reshape(buffer_shape)
        quantized_output_sbuf_view = quantized_output_sbuf_list[bxs_subtile.index].reshape(buffer_shape)

        for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):
            p_size = len(TiledRange(int_tile, int_dim_tile.subtile_dim_info.tile_size))
            f_size = bxs_subtile.size * int_dim_tile.subtile_dim_info.tile_size

            nisa.activation(
                dst=src_proj_res_sbuf_view[:p_size, int_tile.index, :f_size],
                op=nl.copy,
                data=src_proj_res_sbuf_view[:p_size, int_tile.index, :f_size],
                scale=static_input_quant_scale_sbuf[:p_size, 0:1],
                bias=bias_vector[:p_size, 0:1],
            )
            nisa.tensor_scalar(
                dst=quantized_output_sbuf_view[:p_size, int_tile.index, :f_size],
                data=src_proj_res_sbuf_view[:p_size, int_tile.index, :f_size],
                op0=nl.minimum,
                operand0=max_pos_val,
                op1=nl.maximum,
                operand1=-max_pos_val,
            )


def _perform_intermediate_mx_quantization(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    src_proj_res_sbuf_list: list[nl.NkiTensor],
    quantized_output_sbuf_list: list[nl.NkiTensor],
    dequant_scales_output_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 128
    INT_TILE_COUNT = int_dim_tile.tile_count  # I/512
    INT_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128
    INT_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
        for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):
            int_subtiles = TiledRange(int_tile, INT_SUBTILE_SIZE)
            bf16_pattern = [
                [INT_TILE_COUNT * BXS_SUBTILE_SIZE * INT_SUBTILE_SIZE, len(int_subtiles)],
                [INT_SUBTILE_SIZE, bxs_subtile.size],
                [1, INT_SUBTILE_SIZE],
            ]
            fp8x4_pattern = [
                [INT_TILE_COUNT * BXS_SUBTILE_SIZE, len(int_subtiles)],
                [1, bxs_subtile.size],
            ]
            nisa.quantize_mx(
                dst=quantized_output_sbuf_list[bxs_subtile.index].ap(
                    fp8x4_pattern, dtype=nl.float8_e4m3fn_x4, offset=int_tile.index * BXS_SUBTILE_SIZE
                ),
                src=src_proj_res_sbuf_list[bxs_subtile.index].ap(
                    bf16_pattern, offset=int_tile.index * BXS_SUBTILE_SIZE * INT_SUBTILE_SIZE
                ),
                dst_scale=dequant_scales_output_sbuf_list[bxs_subtile.index][
                    : len(int_subtiles), int_tile.index, : bxs_subtile.size
                ],
            )
