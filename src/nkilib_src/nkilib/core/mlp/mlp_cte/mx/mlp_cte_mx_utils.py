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

"""MLP CTE MX utility functions for MX quantization types."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.kernel_helpers import get_nl_act_fn_from_type
from ....utils.tiled_range import TiledRange
from ...mlp_parameters import MLPParameters, mlpp_input_has_mx_block_scale
from ..mlp_cte_constants import MLPCTEConstants
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo


def perform_elementwise_multiply(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    gate_tile_sbuf_list: list[nl.NkiTensor],
    up_tile_data_list: list[nl.NkiTensor],
    up_weight_row_scales_sbuf: Optional[nl.NkiTensor],
    up_static_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[nl.NkiTensor],
    output_tile_sbuf_list: list[nl.NkiTensor],
    up_data_is_psum: bool = False,
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    MX_INT_SUBTILE_SIZE = tile_info.intermediate_dim_tile.subtile_dim_info.tile_size  # 4
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 128
    MX_BXS_SUBTILE_SIZE = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    for bxs_subtile in TiledRange(current_bxs_tile.size, BXS_SUBTILE_SIZE):
        for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):
            int_subtiles = TiledRange(int_tile.size, MX_INT_SUBTILE_SIZE)

            dst_pattern = [
                [int_dim_tile.tile_count * BXS_SUBTILE_SIZE * MX_INT_SUBTILE_SIZE, len(int_subtiles)],
                [MX_INT_SUBTILE_SIZE, bxs_subtile.size],
                [1, MX_INT_SUBTILE_SIZE],
            ]
            dst_offset = int_tile.index * BXS_SUBTILE_SIZE * MX_INT_SUBTILE_SIZE
            gate_data = gate_tile_sbuf_list[bxs_subtile.index].ap(pattern=dst_pattern, offset=dst_offset)

            if up_data_is_psum:
                psum_bank = (bxs_subtile.index // 2) * int_dim_tile.tile_count + int_tile.index
                # Read with a stride size of 4 so that 4 adjacent elements of I lie adjacent to each other in the
                # projection result, ready to be contracted together during down projection.
                up_data = up_tile_data_list[psum_bank].ap(
                    pattern=[
                        [MX_BXS_SUBTILE_SIZE * MX_INT_SUBTILE_SIZE, len(int_subtiles)],
                        [1, bxs_subtile.size],
                        [MX_BXS_SUBTILE_SIZE, MX_INT_SUBTILE_SIZE],
                    ],
                    offset=(bxs_subtile.index % 2) * BXS_SUBTILE_SIZE,
                )
            else:
                up_data = up_tile_data_list[bxs_subtile.index].ap(pattern=dst_pattern, offset=dst_offset)
            dst_tile = output_tile_sbuf_list[bxs_subtile.index].ap(pattern=dst_pattern, offset=dst_offset)

            nisa.tensor_tensor(dst=dst_tile, data1=gate_data, data2=up_data, op=nl.multiply)
            if mlp_params.quant_params.is_quant_mx() and not mlpp_input_has_mx_block_scale(mlp_params):
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=dst_tile,
                    data2=hidden_scales_sbuf_list[bxs_subtile.index // 2].ap(
                        [[MX_BXS_SUBTILE_SIZE, len(int_subtiles)], [1, bxs_subtile.size], [0, MX_INT_SUBTILE_SIZE]],
                        offset=(bxs_subtile.index % 2) * BXS_SUBTILE_SIZE,
                    ),
                )
            elif mlp_params.quant_params.is_quant_static_mx():
                nisa.activation(
                    dst=dst_tile,
                    op=nl.copy,
                    data=dst_tile,
                    scale=up_static_scales_sbuf[: len(int_subtiles), 0:1],
                    bias=bias_vector[: len(int_subtiles), 0:1],
                )
            elif mlp_params.quant_params.is_quant_row_mx():
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=dst_tile,
                    data2=hidden_scales_sbuf_list[bxs_subtile.index // 2].ap(
                        [[MX_BXS_SUBTILE_SIZE, len(int_subtiles)], [1, bxs_subtile.size], [0, MX_INT_SUBTILE_SIZE]],
                        offset=(bxs_subtile.index % 2) * BXS_SUBTILE_SIZE,
                    ),
                )
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=dst_tile,
                    data2=up_weight_row_scales_sbuf.ap(
                        [
                            [int_dim_tile.tile_count * MX_INT_SUBTILE_SIZE, len(int_subtiles)],
                            [0, bxs_subtile.size],
                            [1, MX_INT_SUBTILE_SIZE],
                        ],
                        offset=int_tile.index * MX_INT_SUBTILE_SIZE,
                    ),
                )


def apply_source_projection_activation(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    proj_data_list: list[nl.NkiTensor],
    src_weight_row_scales_sbuf: Optional[nl.NkiTensor],
    src_static_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[nl.NkiTensor],
    act_fn_res_sbuf_list: list[nl.NkiTensor],
    data_is_psum: bool = False,
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    MX_INT_SUBTILE_SIZE = tile_info.intermediate_dim_tile.subtile_dim_info.tile_size  # 4
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 128
    MX_BXS_SUBTILE_SIZE = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    for bxs_subtile in TiledRange(current_bxs_tile.size, BXS_SUBTILE_SIZE):
        for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):
            int_subtiles = TiledRange(int_tile.size, MX_INT_SUBTILE_SIZE)

            dst_pattern = [
                [int_dim_tile.tile_count * BXS_SUBTILE_SIZE * MX_INT_SUBTILE_SIZE, len(int_subtiles)],
                [MX_INT_SUBTILE_SIZE, bxs_subtile.size],
                [1, MX_INT_SUBTILE_SIZE],
            ]
            dst_offset = int_tile.index * BXS_SUBTILE_SIZE * MX_INT_SUBTILE_SIZE

            if data_is_psum:
                psum_bank = (bxs_subtile.index // 2) * int_dim_tile.tile_count + int_tile.index
                # Read with a stride size of 4 so that 4 adjacent elements of I lie adjacent to each other in the
                # projection result, ready to be contracted together during down projection.
                proj_data = proj_data_list[psum_bank].ap(
                    pattern=[
                        [MX_BXS_SUBTILE_SIZE * MX_INT_SUBTILE_SIZE, len(int_subtiles)],
                        [1, bxs_subtile.size],
                        [MX_BXS_SUBTILE_SIZE, MX_INT_SUBTILE_SIZE],
                    ],
                    offset=(bxs_subtile.index % 2) * BXS_SUBTILE_SIZE,
                )
            else:
                proj_data = proj_data_list[bxs_subtile.index].ap(pattern=dst_pattern, offset=dst_offset)
            dst_tile = act_fn_res_sbuf_list[bxs_subtile.index].ap(pattern=dst_pattern, offset=dst_offset)
            if mlp_params.quant_params.is_quant_mx() and not mlpp_input_has_mx_block_scale(mlp_params):
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=proj_data,
                    data2=hidden_scales_sbuf_list[bxs_subtile.index // 2].ap(
                        [[MX_BXS_SUBTILE_SIZE, len(int_subtiles)], [1, bxs_subtile.size], [0, MX_INT_SUBTILE_SIZE]],
                        offset=(bxs_subtile.index % 2) * BXS_SUBTILE_SIZE,
                    ),
                )
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=dst_tile,
                    bias=bias_vector[: len(int_subtiles), 0:1],
                )
            elif mlp_params.quant_params.is_quant_mx() and mlpp_input_has_mx_block_scale(mlp_params):
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=proj_data,
                    bias=bias_vector[: len(int_subtiles), 0:1],
                )
            elif mlp_params.quant_params.is_quant_static_mx():
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=proj_data,
                    scale=src_static_scales_sbuf[: len(int_subtiles), 0:1],
                    bias=bias_vector[: len(int_subtiles), 0:1],
                )
            elif mlp_params.quant_params.is_quant_row_mx():
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=proj_data,
                    data2=hidden_scales_sbuf_list[bxs_subtile.index // 2].ap(
                        [[MX_BXS_SUBTILE_SIZE, len(int_subtiles)], [1, bxs_subtile.size], [0, MX_INT_SUBTILE_SIZE]],
                        offset=(bxs_subtile.index % 2) * BXS_SUBTILE_SIZE,
                    ),
                )
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=dst_tile,
                    data2=src_weight_row_scales_sbuf.ap(
                        [
                            [int_dim_tile.tile_count * MX_INT_SUBTILE_SIZE, len(int_subtiles)],
                            [0, bxs_subtile.size],
                            [1, MX_INT_SUBTILE_SIZE],
                        ],
                        offset=int_tile.index * MX_INT_SUBTILE_SIZE,
                    ),
                )
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=dst_tile,
                    bias=bias_vector[: len(int_subtiles), 0:1],
                )
