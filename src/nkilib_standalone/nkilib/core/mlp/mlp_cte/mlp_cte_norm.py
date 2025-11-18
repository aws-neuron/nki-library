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
MLP CTE Normalization Module

Implements RMS and Layer normalization functions for MLP CTE kernels with detailed
pseudo-code documentation and tile-based processing.

"""

import numpy as np
from typing import Optional, Callable

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.allocator import SbufManager
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_rms_normalization,
    mlpp_has_layer_normalization,
)
from .mlp_cte_constants import (
    MLPCTEConstants,
    BN_STATS_ELEMENTS_PER_TILE,
    BN_AGGR_ELEMENTS_PER_TILE,
)
from .mlp_cte_tile_info import MLPCTETileInfo
from .mlp_cte_tile_info import MlpBxsIndices


#
# Apply RMS normalization to a tile
def apply_rms_norm_to_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    allocate: Callable,
):
    """
    Apply RMS normalization to source tensor tile.

    Pseudo code:
    for each sequence in batch:
        # Step 1: Compute sum of squares across hidden dimension
        sum_of_squares = sum(x_i^2 for x_i in hidden_features)

        # Step 2: Compute RMS normalization factor
        rms_factor = rsqrt(sum_of_squares / hidden_size + epsilon)

        # Step 3: Apply normalization
        output = input * rms_factor
    """

    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    eps_vector = constants.epsilon_bias_vector_sbuf
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size

    kernel_assert(
        len(source_tile_sbuf_list) == BXS_SUBTILE_COUNT,
        f"source_tile_sbuf_list (len = {len(source_tile_sbuf_list)} must have {BXS_SUBTILE_COUNT} tensors",
    )
    kernel_assert(
        len(output_tile_sbuf_list) == BXS_SUBTILE_COUNT,
        f"output_tile_sbuf_list (len = {len(output_tile_sbuf_list)} must have {BXS_SUBTILE_COUNT} tensors",
    )

    square_result_sbuf = allocate(
        (
            nl.par_dim(BXS_SUBTILE_SIZE),
            mlp_params.hidden_size,
        ),
        dtype=constants.activation_data_type,
        name=indices.get_tensor_name("square_result_sbuf"),
    )

    square_accum_result_sbuf_list = []
    for i in range(BXS_SUBTILE_COUNT):
        square_accum_result_tensor = allocate(
            (
                nl.par_dim(BXS_SUBTILE_SIZE),
                1,
            ),
            dtype=constants.activation_data_type,
            name=indices.get_tensor_name("square_accum_result_sbuf", f"subtile{i}"),
        )
        square_accum_result_sbuf_list.append(square_accum_result_tensor)

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    # Loop over all the subtiles in current B x S dimension tile
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        kernel_assert(
            source_tile_sbuf_list[bxs_subtile_idx].shape[1] == mlp_params.hidden_size,
            f"source_tile_sbuf_list[{bxs_subtile_idx}].shape[1] must be "
            f"{mlp_params.hidden_size} but got {source_tile_sbuf_list[bxs_subtile_idx].shape[1]}",
        )
        kernel_assert(
            output_tile_sbuf_list[bxs_subtile_idx].shape[1] == mlp_params.hidden_size,
            f"output_tile_sbuf_list[{bxs_subtile_idx}].shape[1] must be "
            f"{mlp_params.hidden_size} but got {output_tile_sbuf_list[bxs_subtile_idx].shape[1]}",
        )
        source_tile_sbuf = source_tile_sbuf_list[bxs_subtile_idx]
        output_tile_sbuf = output_tile_sbuf_list[bxs_subtile_idx]
        square_accum_result = square_accum_result_sbuf_list[bxs_subtile_idx]

        bxs_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_start
        if bxs_subtile_rest > 0:
            p_bxs_size, f_hidden_size = (
                min(bxs_subtile_rest, BXS_SUBTILE_SIZE),
                mlp_params.hidden_size,
            )
            p_sqa_size, f_sqa_size = p_bxs_size, 1
            p_bias_size, f_bias_size = (
                min(bxs_subtile_rest, bias_vector.shape[0]),
                bias_vector.shape[1],
            )
            p_eps_size, f_eps_size = (
                min(bxs_subtile_rest, eps_vector.shape[0]),
                eps_vector.shape[1],
            )

            nisa.activation_reduce(
                dst=square_result_sbuf[0:p_bxs_size, 0:f_hidden_size],
                op=nl.square,
                data=source_tile_sbuf[0:p_bxs_size, 0:f_hidden_size],
                reduce_op=nl.add,
                reduce_res=square_accum_result[0:p_sqa_size, 0:f_sqa_size],
                bias=bias_vector[0:p_bias_size, 0:f_bias_size],
            )

            nisa.activation(
                dst=square_accum_result[0:p_sqa_size, 0:f_sqa_size],
                op=nl.rsqrt,
                data=square_accum_result[0:p_sqa_size, 0:f_sqa_size],
                bias=eps_vector[0:p_eps_size, 0:f_eps_size],
                scale=float(1.0 / mlp_params.hidden_size),
            )

            nisa.activation(
                dst=output_tile_sbuf[0:p_bxs_size, 0:f_hidden_size],
                op=nl.copy,
                data=source_tile_sbuf[0:p_bxs_size, 0:f_hidden_size],
                scale=square_accum_result[0:p_sqa_size, 0:f_sqa_size],
                bias=bias_vector[0:p_bias_size, 0:f_bias_size],
            )


#
# Apply Layer normalization to a tile
def apply_layer_norm_to_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    allocate: Callable,
):
    """
    Apply Layer normalization to source tensor tile.

    Pseudo code:
    for each sequence in batch:
        # Step 1: Compute statistics across hidden dimension
        mean = sum(x_i for x_i in hidden_features) / hidden_size
        variance = sum((x_i - mean)^2 for x_i in hidden_features) / hidden_size

        # Step 2: Compute normalization factor
        std_inv = rsqrt(variance + epsilon)

        # Step 3: Apply normalization
        output = (input - mean) * std_inv
    """

    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.layer_norm_hidden_dim_tile
    eps_vector = constants.epsilon_bias_vector_sbuf
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count

    kernel_assert(
        len(source_tile_sbuf_list) == BXS_SUBTILE_COUNT,
        f"source_tile_sbuf_list (len = {len(source_tile_sbuf_list)} must have {BXS_SUBTILE_COUNT} tensors",
    )
    kernel_assert(
        len(output_tile_sbuf_list) == BXS_SUBTILE_COUNT,
        f"output_tile_sbuf_list (len = {len(output_tile_sbuf_list)} must have {BXS_SUBTILE_COUNT} tensors",
    )

    bn_stats_result_sbuf = allocate(
        (BXS_SUBTILE_SIZE, hidden_dim_tile.tile_count * BN_STATS_ELEMENTS_PER_TILE),
        dtype=constants.activation_data_type,
        name=indices.get_tensor_name("bn_stats_result_sbuf"),
    )

    bn_aggr_result_sbuf_list = []
    for i in range(BXS_SUBTILE_COUNT):
        bn_aggr_result_tensor = allocate(
            (
                nl.par_dim(BXS_SUBTILE_SIZE),
                BN_AGGR_ELEMENTS_PER_TILE,
            ),
            dtype=constants.activation_data_type,
            name=indices.get_tensor_name("bn_aggr_result_sbuf", f"subtile{i}"),
        )
        bn_aggr_result_sbuf_list.append(bn_aggr_result_tensor)

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    # Loop over all the subtiles in current B x S dimension tile
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        source_tile_sbuf = source_tile_sbuf_list[bxs_subtile_idx]
        output_tile_sbuf = output_tile_sbuf_list[bxs_subtile_idx]
        kernel_assert(
            source_tile_sbuf.shape[1] == mlp_params.hidden_size,
            f"source_tile_sbuf_list[{bxs_subtile_idx}].shape[1] must be "
            f"{mlp_params.hidden_size} instead of {source_tile_sbuf.shape[1]}",
        )
        kernel_assert(
            output_tile_sbuf.shape[1] == mlp_params.hidden_size,
            f"output_tile_sbuf_list[{bxs_subtile_idx}].shape[1] must be "
            f"{mlp_params.hidden_size} instead of {output_tile_sbuf.shape[1]}",
        )

        bxs_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_start
        if bxs_subtile_rest > 0:
            # Loop over hidden dimension tiles and calculate the bn stats for all tiles
            for hidden_tile_idx in range(hidden_dim_tile.tile_count):
                p_bxs_size, f_hidden_size = (
                    min(bxs_subtile_rest, BXS_SUBTILE_SIZE),
                    hidden_dim_tile.get_tile_bound(hidden_tile_idx),
                )
                f_bns_src_size = BN_STATS_ELEMENTS_PER_TILE * hidden_dim_tile.tile_count
                f_bns_dst_size = BN_STATS_ELEMENTS_PER_TILE
                f_bna_size = BN_AGGR_ELEMENTS_PER_TILE
                f_std_size = 1
                p_eps_size, f_eps_size = (
                    min(bxs_subtile_rest, eps_vector.shape[0]),
                    eps_vector.shape[1],
                )
                f_dst_size = mlp_params.hidden_size

                nisa.bn_stats(
                    dst=bn_stats_result_sbuf[
                        0:p_bxs_size,
                        nl.ds(hidden_tile_idx * BN_STATS_ELEMENTS_PER_TILE, f_bns_dst_size),
                    ],
                    data=source_tile_sbuf[
                        :p_bxs_size,
                        nl.ds(hidden_tile_idx * hidden_dim_tile.tile_size, f_hidden_size),
                    ],
                )

            nisa.bn_aggr(
                dst=bn_aggr_result_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 0:f_bna_size],
                data=bn_stats_result_sbuf[0:p_bxs_size, 0:f_bns_src_size],
            )

            nisa.activation(
                dst=bn_aggr_result_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 1 : 1 + f_std_size],
                op=nl.rsqrt,
                data=bn_aggr_result_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 1 : 1 + f_std_size],
                bias=eps_vector[0:p_eps_size, 0:f_eps_size],
            )

            nisa.tensor_scalar(
                dst=output_tile_sbuf[0:p_bxs_size, 0:f_dst_size],
                data=source_tile_sbuf[0:p_bxs_size, 0:f_dst_size],
                op0=nl.subtract,
                op1=nl.multiply,
                operand0=bn_aggr_result_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 0:f_std_size],
                operand1=bn_aggr_result_sbuf_list[bxs_subtile_idx][0:p_bxs_size, 1 : 1 + f_std_size],
                engine=nisa.vector_engine,
            )


#
# Apply normalization to a tile if it is required
def apply_normalization_if_necessary(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    allocate: Callable,
):
    # Apply normalization if necessary
    if mlpp_has_rms_normalization(mlp_params):
        apply_rms_norm_to_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices,
            source_tile_sbuf_list,
            output_tile_sbuf_list,
            allocate,
        )
    elif mlpp_has_layer_normalization(mlp_params):
        apply_layer_norm_to_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices,
            source_tile_sbuf_list,
            output_tile_sbuf_list,
            allocate,
        )


def cleanup_heap_for_normalization(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    sbm: Optional[SbufManager],
):
    if sbm != None and mlpp_has_rms_normalization(mlp_params):
        sbm.pop_heap()  # square_result_sbuf
        for _ in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
            sbm.pop_heap()  # square_accum_result_sbuf_list
