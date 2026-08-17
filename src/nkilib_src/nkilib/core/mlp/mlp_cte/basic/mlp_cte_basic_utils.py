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

"""MLP CTE basic utility functions for non-MX quantization types."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.kernel_helpers import get_nl_act_fn_from_type
from ....utils.tiled_range import TiledRange
from ...mlp_parameters import MLPParameters
from ..mlp_cte_constants import MLPCTEConstants
from .mlp_cte_basic_tile_info import MLPCTEBasicTileInfo


def apply_source_projection_bias(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    proj_psum_list: list[nl.NkiTensor],
    bias_tensor_sbuf: nl.NkiTensor,
    bias_res_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count

    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        for intermediate_tile_idx in range(int_dim_tile.tile_count):
            p_bxs_size = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
            f_int_size = int_dim_tile.get_tile_bound(intermediate_tile_idx)

            if p_bxs_size <= 0 or f_int_size <= 0:
                continue

            psum_bank = bxs_subtile_idx * int_dim_tile.tile_count + intermediate_tile_idx
            nisa.tensor_tensor(
                dst=bias_res_sbuf_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size],
                data1=proj_psum_list[psum_bank][:p_bxs_size, :f_int_size],
                data2=bias_tensor_sbuf[
                    :p_bxs_size,
                    int_dim_tile.get_tile_indices(intermediate_tile_idx, f_int_size),
                ],
                op=nl.add,
            )


def perform_elementwise_multiply(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
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
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    for bxs_subtile in TiledRange(current_bxs_tile.size, BXS_SUBTILE_SIZE):
        for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):
            gate_data = gate_tile_sbuf_list[bxs_subtile.index][: bxs_subtile.size, int_tile.index, : int_tile.size]

            if up_data_is_psum:
                psum_bank = bxs_subtile.index * int_dim_tile.tile_count + int_tile.index
                up_data = up_tile_data_list[psum_bank][: bxs_subtile.size, : int_tile.size]
            else:
                up_data = up_tile_data_list[bxs_subtile.index][: bxs_subtile.size, int_tile.index, : int_tile.size]
            dst_tile = output_tile_sbuf_list[bxs_subtile.index][: bxs_subtile.size, int_tile.index, : int_tile.size]

            nisa.tensor_tensor(dst=dst_tile, data1=gate_data, data2=up_data, op=nl.multiply)

            if mlp_params.quant_params.is_quant_static():
                nisa.activation(
                    dst=dst_tile,
                    op=nl.copy,
                    data=dst_tile,
                    scale=up_static_scales_sbuf[: bxs_subtile.size, 0:1],
                    bias=bias_vector[: bxs_subtile.size, 0:1],
                )
            elif mlp_params.quant_params.is_quant_row():
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=dst_tile,
                    data2=up_weight_row_scales_sbuf[
                        : bxs_subtile.size, nl.ds(int_tile.index * int_dim_tile.tile_size, int_tile.size)
                    ],
                )
                nisa.activation(
                    dst=dst_tile,
                    op=nl.copy,
                    data=dst_tile,
                    scale=hidden_scales_sbuf_list[bxs_subtile.index][: bxs_subtile.size, 0:1],
                    bias=bias_vector[: bxs_subtile.size, 0:1],
                )


def apply_source_projection_activation(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    proj_data_list: list[nl.NkiTensor],
    src_weight_row_scales_sbuf: Optional[nl.NkiTensor],
    src_static_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[nl.NkiTensor],
    act_fn_res_sbuf_list: list[nl.NkiTensor],
    data_is_psum: bool = False,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    for bxs_subtile in TiledRange(current_bxs_tile.size, BXS_SUBTILE_SIZE):
        for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):
            if data_is_psum:
                psum_bank = bxs_subtile.index * int_dim_tile.tile_count + int_tile.index
                proj_data = proj_data_list[psum_bank][: bxs_subtile.size, : int_tile.size]
            else:
                proj_data = proj_data_list[bxs_subtile.index][: bxs_subtile.size, int_tile.index, : int_tile.size]
            dst_tile = act_fn_res_sbuf_list[bxs_subtile.index][: bxs_subtile.size, int_tile.index, : int_tile.size]

            if mlp_params.quant_params.is_quant_static():
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=proj_data,
                    scale=src_static_scales_sbuf[: bxs_subtile.size, 0:1],
                    bias=bias_vector[: bxs_subtile.size, 0:1],
                )
            elif mlp_params.quant_params.is_quant_row():
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=proj_data,
                    data2=src_weight_row_scales_sbuf[
                        : bxs_subtile.size, nl.ds(int_tile.index * int_dim_tile.tile_size, int_tile.size)
                    ],
                )
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=dst_tile,
                    scale=hidden_scales_sbuf_list[bxs_subtile.index][: bxs_subtile.size, 0:1],
                    bias=bias_vector[: bxs_subtile.size, 0:1],
                )
            elif not mlp_params.quant_params.is_quant():
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=proj_data,
                    bias=bias_vector[: bxs_subtile.size, 0:1],
                )
