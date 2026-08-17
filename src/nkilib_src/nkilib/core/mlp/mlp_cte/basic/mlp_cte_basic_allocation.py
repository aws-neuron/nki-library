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

"""MLP CTE basic allocation functions for non-MX quantization types."""

from typing import Callable

import nki.language as nl

from ....utils.allocator import SbufManager, sizeinbytes
from ...mlp_parameters import MLPParameters, mlpp_has_dma_xpose, mlpp_input_has_packed_scale
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from .mlp_cte_basic_tile_info import (
    MLPCTEBasicTileInfo,
    get_down_weights_tensor_info,
    get_gate_up_weights_tensor_info,
    get_hidden_tile_tensor_info,
    get_intermediate_tile_tensor_info,
    get_output_tile_tensor_info,
)


def allocate_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf_list: list,
    hidden_tile_scales_sbuf_list: list,
    alloc_tile: Callable,
    alloc_scale: Callable,
):
    hidden_tile_shape, hidden_tile_dtype = get_hidden_tile_tensor_info(
        mlp_params,
        constants,
        tile_info.bxs_dim_tile,
    )

    for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
        hidden_tensor = alloc_tile(
            hidden_tile_shape,
            dtype=hidden_tile_dtype,
            buffer=nl.sbuf,
            align=(32 if mlpp_has_dma_xpose(mlp_params) else None),  # DMA transpose requires 32B align
            name=indices.get_tensor_name("hidden_tensor", f"subbxs{bxs_subtile_idx}"),
        )
        hidden_tile_sbuf_list.append(hidden_tensor)
        if mlpp_input_has_packed_scale(mlp_params):
            hidden_scale_tensor = alloc_scale(
                (tile_info.bxs_dim_tile.subtile_dim_info.tile_size, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
                name=indices.get_tensor_name('hidden_scale_tensor', f'subbxs{bxs_subtile_idx}'),
            )
            hidden_tile_scales_sbuf_list.append(hidden_scale_tensor)


def allocate_intermediate_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    name: str,
    dtype,
    intermediate_tensor_sbuf_list: list,
    alloc: Callable,
):
    intermediate_tensor_shape, _ = get_intermediate_tile_tensor_info(
        mlp_params,
        constants,
        tile_info.bxs_dim_tile,
    )

    for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
        intermediate_tensor = alloc(
            intermediate_tensor_shape,
            dtype=dtype,
            name=indices.get_tensor_name(name, f'subbxs{bxs_subtile_idx}'),
        )
        intermediate_tensor_sbuf_list.append(intermediate_tensor)


def allocate_output_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list,
    alloc: Callable,
):
    output_tile_shape, output_tile_dtype = get_output_tile_tensor_info(
        mlp_params,
        constants,
        tile_info.bxs_dim_tile,
    )

    for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
        output_tile_sbuf = alloc(
            output_tile_shape,
            dtype=output_tile_dtype,
            name=indices.get_tensor_name('output_tensor', f'subbxs{bxs_subtile_idx}'),
        )
        output_tile_sbuf_list.append(output_tile_sbuf)


def allocate_src_projection_weights(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    src_proj_weights_sbuf_list: list,
    alloc: Callable,
    sbm: SbufManager,
):
    weights_tensor_shape, weights_tensor_dtype = get_gate_up_weights_tensor_info(
        mlp_params,
        constants,
        tile_info.src_proj_hidden_dim_tile,
        tile_info.src_proj_intermediate_dim_tile,
    )
    weights_tensor_size = sizeinbytes(weights_tensor_dtype)
    for dim_size in weights_tensor_shape[1:]:
        weights_tensor_size *= dim_size
    actual_src_proj_weights_buffer_count = min(
        sbm.get_free_space() // weights_tensor_size,
        constants.src_proj_weights_max_buffer_count,
    )
    for weight_buffer_idx in range(actual_src_proj_weights_buffer_count):
        weights_tensor = alloc(
            weights_tensor_shape,
            dtype=weights_tensor_dtype,
            buffer=nl.sbuf,
            name=indices.get_tensor_name("weights_tensor", f"buf{weight_buffer_idx}"),
        )
        src_proj_weights_sbuf_list.append(weights_tensor)


def allocate_down_projection_weights(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    down_proj_weights_sbuf: list,
    alloc: Callable,
):
    weights_tensor_shape, weights_tensor_dtype = get_down_weights_tensor_info(
        mlp_params,
        constants,
        tile_info.down_proj_hidden_dim_tile,
        tile_info.down_proj_intermediate_dim_tile,
    )

    for weight_buffer_idx in range(constants.down_proj_weights_buffer_count):
        down_proj_weights_tensor = alloc(
            weights_tensor_shape,
            dtype=weights_tensor_dtype,
            name=indices.get_tensor_name("down_proj_weights_sbuf", f"buffer{weight_buffer_idx}"),
        )
        down_proj_weights_sbuf.append(down_proj_weights_tensor)
