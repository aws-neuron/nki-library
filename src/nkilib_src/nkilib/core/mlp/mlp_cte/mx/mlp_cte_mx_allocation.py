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

"""MLP CTE MX allocation functions for MX, STATIC_MX, and ROW_MX quantization types."""

import math

import nki.language as nl

from ....utils.allocator import SbufManager, sizeinbytes
from ...mlp_parameters import MLPParameters, mlpp_input_has_mx_block_scale, mlpp_input_has_packed_scale
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo


def allocate_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf_list: list,
    hidden_tile_scales_sbuf_list: list,
    sbm: SbufManager,
):
    heap_alloc = sbm.alloc_heap if sbm else nl.NkiTensor
    stack_alloc = sbm.alloc_stack if sbm else nl.NkiTensor

    for bxs_subtile_idx in range(tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_count):
        hidden_tensor = heap_alloc(
            (
                nl.tile_size.pmax,  # 128_H
                tile_info.src_proj_hidden_dim_tile.tile_count,  # H/512
                2 * nl.tile_size.pmax,  # 256_T
                tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_size,  # 4
            ),
            dtype=constants.hidden_tile_data_type,
            buffer=nl.sbuf,
            align=32,  # xbar transpose requires 32B alignment
            name=indices.get_tensor_name("hidden_tensor", f"subbxs{bxs_subtile_idx}"),
        )
        hidden_tile_sbuf_list.append(hidden_tensor)
        if mlpp_input_has_mx_block_scale(mlp_params):
            n_packed = math.ceil(tile_info.src_proj_hidden_dim_tile.tile_count / 4)
            hidden_scale_tensor = stack_alloc(
                (nl.tile_size.pmax, n_packed, tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size),
                dtype=nl.uint8,
                buffer=nl.sbuf,
                align=32,
                name=indices.get_tensor_name('hidden_mx_block_scale_tensor', f"subbxs{bxs_subtile_idx}"),
            )
            hidden_tile_scales_sbuf_list.append(hidden_scale_tensor)
        elif mlpp_input_has_packed_scale(mlp_params):
            hidden_scale_tensor = stack_alloc(
                (nl.tile_size.pmax, tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size),
                dtype=nl.float32,
                buffer=nl.sbuf,
                align=32,
                name=indices.get_tensor_name('hidden_scale_tensor', f"subbxs{bxs_subtile_idx}"),
            )
            hidden_tile_scales_sbuf_list.append(hidden_scale_tensor)


def allocate_intermediate_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    name: str,
    dtype,
    intermediate_tensor_sbuf_list: list,
    sbm: SbufManager,
):
    stack_alloc = sbm.alloc_stack if sbm else nl.NkiTensor

    intermediate_tensor_shape = (
        tile_info.intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
        tile_info.intermediate_dim_tile.tile_count,  # I/512
        tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_size,  # 128
        tile_info.intermediate_dim_tile.subtile_dim_info.tile_size,  # 4
    )

    for bxs_subtile_idx in range(tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_count):
        intermediate_tensor = stack_alloc(
            intermediate_tensor_shape,
            dtype=dtype,
            name=indices.get_tensor_name(name, f'subbxs{bxs_subtile_idx}'),
        )
        intermediate_tensor_sbuf_list.append(intermediate_tensor)


def allocate_src_projection_weights(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    src_proj_weights_sbuf_list: list,
    sbm: SbufManager,
):
    heap_alloc = sbm.alloc_heap if sbm else nl.NkiTensor

    weights_tensor_shape = (
        tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_count,  # 128
        tile_info.intermediate_dim_tile.tile_count,  # I/512
        tile_info.intermediate_dim_tile.subtile_dim_info.tile_size,  # 4
        tile_info.intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
        tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_size,  # 4
    )
    weights_tensor_dtype = constants.src_proj_quant_data_type
    if sbm != None:
        weights_tensor_size = sizeinbytes(weights_tensor_dtype)
        for dim_size in weights_tensor_shape[1:]:
            weights_tensor_size *= dim_size
        actual_src_proj_weights_buffer_count = min(
            sbm.get_free_space() // weights_tensor_size,
            constants.src_proj_weights_max_buffer_count,
        )
    else:
        actual_src_proj_weights_buffer_count = constants.src_proj_weights_max_buffer_count
    for weight_buffer_idx in range(actual_src_proj_weights_buffer_count):
        weights_tensor = heap_alloc(
            weights_tensor_shape,
            dtype=weights_tensor_dtype,
            buffer=nl.sbuf,
            name=indices.get_tensor_name("weights_tensor", f"buf{weight_buffer_idx}"),
        )
        src_proj_weights_sbuf_list.append(weights_tensor)


def allocate_down_projection_weights(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    down_proj_weights_sbuf: list,
    sbm: SbufManager,
):
    stack_alloc = sbm.alloc_stack if sbm else nl.NkiTensor

    buffer_shape = (
        tile_info.intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
        tile_info.down_proj_hidden_dim_tile.tile_size,  # 1024
        tile_info.intermediate_dim_tile.subtile_dim_info.tile_size,  # 4
    )

    buffer_dtype = constants.down_proj_quant_data_type

    for weight_buffer_idx in range(constants.down_proj_weights_buffer_count):
        down_proj_weights_tensor = stack_alloc(
            buffer_shape,
            buffer_dtype,
            name=indices.get_tensor_name("down_proj_weights_sbuf", f"buffer{weight_buffer_idx}"),
        )
        down_proj_weights_sbuf.append(down_proj_weights_tensor)
