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

"""
MLP CTE Tensor I/O Module

Handles tensor loading and storing operations for MLP CTE kernels including hidden tensor
tiles, fused operations, and cross-partition vector loading.

"""

from math import prod
from typing import Callable, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ..mlp_parameters import MLPParameters, mlpp_has_fused_add, mlpp_input_has_packed_scale, mlpp_store_fused_add
from .mlp_cte_constants import MLPCTEConstants
from .mlp_cte_sharding import ShardedDim
from .mlp_cte_tile_info import MlpBxsIndices, MLPCTETileInfo
from .mlp_cte_utils import calc_vec_crossload_free_dim_len


# This method ensures I/O tensors where the first 2 dimensions are batch and sequence length have the correct shape
# based on the sharding strategy
def _reshape_io_tensor(constants: MLPCTEConstants, tensor: nl.ndarray) -> nl.ndarray:
    # No need to change the shape if we aren't sharding on B X S
    if constants.sharded_dim != ShardedDim.BATCH_X_SEQUENCE_LENGTH:
        return tensor

    # Set the batch to 1 for the 1st dimension and multiply B X S for the 2nd dimension and leave the
    # rest of the dimensions the same
    # Workaround: Build tuple manually without using tuple expansion
    shape_list = [1, tensor.shape[0] * tensor.shape[1]]
    for i in range(2, len(tensor.shape)):
        shape_list.append(tensor.shape[i])
    new_shape = tuple(shape_list)
    return tensor.reshape(new_shape)


# This method loads an entire tile into SBUF as subtiles.
# The hidden (input) tensor is tiled along the S dimension.
def load_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.ndarray],
):
    # Alias this to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile

    # Ensure we have the shapes we need
    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)

    # This is the offset into the original tensor and the total size from the tensor that we are computing
    tensor_bxs_offset = constants.get_bxs_offset()
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(indices.bxs_tile_idx, bxs_subtile_idx)
        if p_bxs_size > 0:
            bxs_offset = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx) + tensor_bxs_offset
            hidden_tensor_offset = (
                indices.batch_idx * hidden_tensor_hbm_view.shape[1] * hidden_tensor_hbm_view.shape[2]
                + (bxs_offset) * hidden_tensor_hbm_view.shape[2]
            )
            nisa.dma_copy(
                dst=output_tile_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 0 : mlp_params.hidden_size],
                src=hidden_tensor_hbm_view.ap(
                    [[mlp_params.hidden_size, p_bxs_size], [1, mlp_params.hidden_size]],
                    offset=hidden_tensor_offset,
                ),
            )


# The hidden (input) tensor is tiled along the S dimension while fusing it with the given fuse input
# tensor. This method loads and entire tile into SBUF as subtiles.


def load_fused_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.ndarray],
):
    # Alias this to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile

    # Ensure we have the shapes we need
    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)
    fused_add_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.fused_add_params.fused_add_tensor)

    # This is the offset into the original tensor and the total size from the tensor that we are computing
    tensor_bxs_offset = constants.get_bxs_offset()
    H = mlp_params.hidden_size

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_offset = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx) + tensor_bxs_offset
        if p_bxs_size > 0:
            nisa.dma_compute(
                output_tile_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 0:H],
                [
                    hidden_tensor_hbm_view.ap([[H, p_bxs_size], [1, H]], offset=bxs_offset * H),
                    fused_add_tensor_hbm_view.ap([[H, p_bxs_size], [1, H]], offset=bxs_offset * H),
                ],
                [1.0, 1.0],
                nl.add,
            )


#
# The hidden (input) tensor is tiled along the S dimension and stored in SBUF using subtiles.
# This method stores an entire tile into HBM.


def store_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf: list[nl.ndarray],
    output_tensor_hbm: nl.ndarray,
):
    # Alias this to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size

    output_tensor_hbm_view = _reshape_io_tensor(constants, output_tensor_hbm)
    # This is the offset into the output tensor and the total size from the tensor that we are computing
    tensor_bxs_offset = constants.get_bxs_offset()
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        bxs_subtile_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_subtile_start
        if bxs_subtile_rest > 0:
            p_bxs_size = min(bxs_subtile_rest, BXS_SUBTILE_SIZE)
            f_h_size = mlp_params.hidden_size
            bxs_offset = bxs_subtile_start + tensor_bxs_offset
            output_offset = indices.batch_idx * output_tensor_hbm.shape[2] * output_tensor_hbm.shape[1] + (
                (bxs_offset) * output_tensor_hbm.shape[2]
            )
            hidden_tile_sbuf_view = hidden_tile_sbuf[bxs_subtile_idx].reshape(
                (BXS_SUBTILE_SIZE, mlp_params.hidden_size)
            )
            nisa.dma_copy(
                dst=output_tensor_hbm_view.ap([[f_h_size, p_bxs_size], [1, f_h_size]], offset=output_offset),
                src=hidden_tile_sbuf_view[0:p_bxs_size, 0:f_h_size],
            )


#
# The hidden (input) tensor is tiled along the S dimension and stored in SBUF using subtiles.
# This method stores half of the entire tile into HBM.
# The split is on the hidden dimension. So this method effectively stores a shape of size [S, H/2].
# We do this because for small sequence lengths.
# We shard on the intermediate dimension. Each core ends up with a [S, H] result but those results
# have to be added to get the final output.
# So each core does half the adding and writes half of the output.  Hence the need for this method.


def store_half_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf: list[nl.ndarray],
    output_tensor_hbm: nl.ndarray,
):
    bxs_dim_tile = tile_info.bxs_dim_tile

    output_tensor_hbm_view = _reshape_io_tensor(constants, output_tensor_hbm)
    half_hidden_size = mlp_params.hidden_size // 2
    hidden_offset = indices.program_id * half_hidden_size
    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    H = output_tensor_hbm_view.shape[2]

    # Note: hidden_tile_sbuf is now a Python list (migrated from nl.par_dim block dimensions)
    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_offset = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        output_offset = indices.batch_idx * H * output_tensor_hbm_view.shape[1] + bxs_offset * H + hidden_offset
        nisa.dma_copy(
            # This can't be replaced with slicing because of batch dimension (no nested slicing support)
            dst=output_tensor_hbm_view.ap([[H, p_bxs_size], [1, half_hidden_size]], offset=output_offset),
            src=hidden_tile_sbuf[bxs_subtile_idx][0:p_bxs_size, hidden_offset : hidden_offset + half_hidden_size],
        )


def load_hidden_tensor_tile_opt_fused_add(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.ndarray],
    output_tile_scales_sbuf_list: Optional[nl.ndarray],
    output_stored_add_tensor_hbm: Optional[nl.ndarray],
):
    if mlpp_has_fused_add(mlp_params):
        # Load the hidden tensor tile with the fused add applied
        load_fused_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list)
        if mlpp_store_fused_add(mlp_params):
            # Store the resulting fused add hidden tensor
            store_hidden_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                output_tile_sbuf_list,
                output_stored_add_tensor_hbm,
            )
    else:  # No fused add
        # Load the hidden tensor tile
        load_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list)
        if mlpp_input_has_packed_scale(mlp_params):
            load_packed_hidden_scales(mlp_params, tile_info, constants, indices, output_tile_scales_sbuf_list)


#
# Load bias vector and broadcast it so it can be used for tensor/tensor adds


def load_bias_vector(bias_tensor_hbm: nl.ndarray, data_type: nki.dtype, allocator: Callable) -> nl.ndarray:
    shuffle_group_size = 32  # This is dictated by the hardware
    num_broadcasts = nl.tile_size.pmax // shuffle_group_size
    # Ensure the partition dimension is the full size so we can broadcast the bias vector
    bias_vector_len = bias_tensor_hbm.shape[1]
    bias_tensor_sbuf = allocator((nl.tile_size.pmax, bias_vector_len), dtype=data_type)

    # Load [1, bias_vector_len]
    kernel_assert(
        bias_tensor_hbm.shape[0] == 1,
        "Internal error: Bias vector first dimension should be of length 1",
    )

    shuffle_mask = [0] * shuffle_group_size
    # Do multiple 32-partition broadcasts to get the bias vector into all partitions
    nisa.dma_copy(
        dst=bias_tensor_sbuf[0:1, 0:bias_vector_len],
        src=bias_tensor_hbm[0:1, 0:bias_vector_len],
    )
    for b in range(num_broadcasts):
        nisa.nc_stream_shuffle(
            src=bias_tensor_sbuf[0:1, 0:bias_vector_len],
            dst=bias_tensor_sbuf[b * shuffle_group_size : (b + 1) * shuffle_group_size, 0:bias_vector_len],
            shuffle_mask=shuffle_mask,
        )

    return bias_tensor_sbuf


def load_packed_hidden_scales(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_scales_sbuf_list: list[nl.ndarray],
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    DTYPE_SIZE_RATIO = 4

    tensor_bxs_offset = constants.get_bxs_offset()

    hidden_size_fp32 = (mlp_params.hidden_size // DTYPE_SIZE_RATIO) + 1
    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(indices.bxs_tile_idx, bxs_subtile_idx)
        if p_bxs_size > 0:
            bxs_offset = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx) + tensor_bxs_offset
            nisa.dma_copy(
                dst=output_tile_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, :1],
                src=hidden_tensor_hbm_view.ap(
                    dtype=nl.float32,
                    pattern=[[hidden_size_fp32, p_bxs_size], [1, 1]],
                    offset=(bxs_offset * hidden_size_fp32) + (hidden_size_fp32 - 1),
                ),
            )


def load_source_projection_row_scales(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    src_proj_scales_hbm: nl.ndarray,
    allocator: Callable,
) -> nl.ndarray:
    src_proj_scales_sbuf = allocator((nl.tile_size.pmax, mlp_params.intermediate_size), dtype=nl.float32)
    nisa.dma_copy(
        dst=src_proj_scales_sbuf[nl.ds(0, nl.tile_size.pmax), nl.ds(0, mlp_params.intermediate_size)],
        src=src_proj_scales_hbm[
            nl.ds(0, nl.tile_size.pmax), nl.ds(constants.get_intermediate_offset(), mlp_params.intermediate_size)
        ],
    )
    return src_proj_scales_sbuf


def load_static_input_scales(static_quant_scale_hbm: nl.ndarray, allocator: Callable) -> nl.ndarray:
    src_proj_scales_sbuf = allocator((nl.tile_size.pmax, 1), dtype=nl.float32)
    nisa.dma_copy(
        dst=src_proj_scales_sbuf[0 : nl.tile_size.pmax, 0:1],
        src=static_quant_scale_hbm[0 : nl.tile_size.pmax, 0:1],
    )
    return src_proj_scales_sbuf


def load_and_multiply_static_weight_scales(
    constants: MLPCTEConstants,
    static_weight_scale_hbm: nl.ndarray,
    static_input_scale_sbuf: nl.ndarray,
    allocator: Callable,
) -> nl.ndarray:
    static_weight_scale_sbuf = allocator((nl.tile_size.pmax, 1), dtype=nl.float32)
    nisa.dma_copy(
        dst=static_weight_scale_sbuf[0 : nl.tile_size.pmax, 0:1],
        src=static_weight_scale_hbm[0 : nl.tile_size.pmax, 0:1],
    )
    nisa.activation(
        dst=static_weight_scale_sbuf[0 : nl.tile_size.pmax, 0:1],
        op=nl.copy,
        data=static_weight_scale_sbuf[0 : nl.tile_size.pmax, 0:1],
        bias=constants.bxs_dim_subtile_zero_bias_vector_sbuf[0 : nl.tile_size.pmax, 0:1],
        scale=static_input_scale_sbuf[0 : nl.tile_size.pmax, 0:1],
    )
    return static_weight_scale_sbuf


#
# Loads a vector such that:
#   partition 0, free 0 = element 0
#   partition 1, free 0 = element 1
#   partition 2, free 0 = element 2
#   ...
#   partition pmax, free 0 = element pmax - 1
#   partition    0, free 1 = element pmax
#   ...


def load_vector_across_partitions(
    vector_tensor_hbm: nl.ndarray,
    data_type: nki.dtype,
    allocator: Callable,
    tensor_name: str,
) -> nl.ndarray:
    vec_len, elements_per_partition = calc_vec_crossload_free_dim_len(vector_tensor_hbm)
    vector_tensor_hbm_view = vector_tensor_hbm.reshape((vec_len, 1))
    output_tensor_sbuf = allocator(
        (nl.tile_size.pmax, elements_per_partition),
        dtype=data_type,
        name=tensor_name,
    )

    p_size = nl.tile_size.pmax
    for p_element_idx in range(elements_per_partition):
        safe_p_size = min(p_size, vec_len - p_element_idx * p_size)
        nisa.dma_copy(
            src=vector_tensor_hbm_view[nl.ds(p_element_idx * p_size, safe_p_size), 0:1],
            dst=output_tensor_sbuf[0:safe_p_size, p_element_idx],
        )
    return output_tensor_sbuf
