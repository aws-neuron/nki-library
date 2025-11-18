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
MLP CTE Projection Module

Implements matrix multiplication operations for MLP projections including up, gate, and down
projections with detailed pseudo-code documentation and optimized processing order.

"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_helpers import get_ceil_quotient, PSUM_BANK_SIZE
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_gate_projection,
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
)
from .mlp_cte_constants import MLPCTEConstants
from .mlp_cte_tile_info import MLPCTETileInfo
from .mlp_cte_utils import (
    apply_source_projection_activation,
    apply_source_projection_bias,
    perform_elementwise_multiply,
)
from .mlp_cte_tile_info import MlpBxsIndices
from .mlp_cte_sharding import ShardedDim


def perform_down_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    weights_tensor_hbm: nl.ndarray,
    weights_sbuf_list: list[nl.ndarray],
    bias_tensor_sbuf: Optional[nl.ndarray],
    source_scales_sbuf: Optional[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    sbm: SbufManager,
):
    """Multiply source [BxS, I] by weights [I, H] to produce [BxS, H].

    Performs down projection matrix multiplication with optional bias application.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        indices: Batch×sequence indices for tensor naming
        source_tile_sbuf_list: Source tensors in SBUF
        weights_tensor_hbm: Weight tensor in HBM
        weights_sbuf_list: Weight buffers in SBUF
        bias_tensor_sbuf: Optional bias tensor in SBUF
        source_scales_sbuf: Optional source scales in SBUF
        output_tile_sbuf_list: Output tensors in SBUF
        sbm: SBUF memory manager

    Returns:
        None

    Intended Usage:
        Called to perform down projection in MLP forward pass
    """
    apply_bias = bias_tensor_sbuf != None
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.down_proj_hidden_dim_tile
    int_dim_tile = tile_info.down_proj_intermediate_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size
    I_SHARD_OFFSET = constants.get_intermediate_offset()

    alloc_stack = sbm.alloc_stack if sbm else nl.ndarray

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    hidden_tiles = TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size)
    int_tiles = TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size)

    for hidden_tile in hidden_tiles:
        # Use this for PSUM allocation to reflect the index as a bank number with the proper total banks
        proj_results_psum_list = []
        for bank in range(constants.required_down_proj_psum_bank_count):
            psum_tensor = nl.ndarray(
                (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name("down_psum_tensor", f"hidden{hidden_tile.index}__bank{bank}"),
                # lazy_initialization=True,
            )
            proj_results_psum_list.append(psum_tensor)

        for int_tile in int_tiles:
            # Load the weights for the current H tile
            weights_buffer_idx = (
                hidden_tile.index * len(int_tiles) + int_tile.index
            ) % constants.down_proj_weights_buffer_count

            nisa.dma_copy(
                src=weights_tensor_hbm[
                    nl.ds(I_SHARD_OFFSET + int_tile.start_offset, int_tile.size),
                    nl.ds(hidden_tile.start_offset, hidden_tile.size),
                ],
                dst=weights_sbuf_list[weights_buffer_idx][: int_tile.size, : hidden_tile.size],
            )

            hidden_subtiles = TiledRange(hidden_tile, H_SUBTILE_SIZE)
            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
                for hidden_subtile in hidden_subtiles:
                    # When calculating the PSUM bank to use, we have to be sure not to include the
                    # intermediate tile index into the calculation.
                    # The intermediate dimension is the contraction (accumulation) dimension so we need to
                    # make sure the bank number is
                    # invariant relative to that index so all the data that needs to accumulate together does.
                    psum_bank = (
                        hidden_tile.index * BXS_SUBTILE_COUNT * len(hidden_subtiles)
                        + bxs_subtile.index * len(hidden_subtiles)
                        + hidden_subtile.index
                    ) % constants.required_down_proj_psum_bank_count

                    source_tile_sbuf_view = source_tile_sbuf_list[bxs_subtile.index].reshape(
                        (
                            BXS_SUBTILE_SIZE,
                            tile_info.src_proj_intermediate_dim_tile.tile_count
                            * tile_info.src_proj_intermediate_dim_tile.tile_size,
                        )
                    )

                    weights_sbuf_view = weights_sbuf_list[weights_buffer_idx].reshape(
                        (
                            int_dim_tile.tile_size,
                            H_SUBTILE_COUNT,
                            H_SUBTILE_SIZE,
                        )
                    )

                    # For down projection:
                    # - source_tile_sbuf has shape (BXS_SUBTILE_SIZE, intermediate_size)
                    # - weights have shape (intermediate_size, hidden_size)
                    # - Result is (bxs_size, hidden_size)

                    # Perform matmul with accumulation
                    # For down projection: stationary is [bxs, intermediate], moving is [intermediate, hidden]
                    # The stationary tensor needs proper indexing
                    nisa.nc_matmul(
                        dst=proj_results_psum_list[psum_bank][0 : bxs_subtile.size, 0 : hidden_subtile.size],
                        stationary=source_tile_sbuf_view.ap(
                            [
                                [source_tile_sbuf_view.shape[1], int_tile.size],
                                [1, bxs_subtile.size],
                            ],
                            offset=int_tile.start_offset,
                        ),
                        moving=weights_sbuf_view[
                            0 : int_tile.size,
                            hidden_subtile.index,
                            0 : hidden_subtile.size,
                        ],
                    )

                    # Copy each completed portion to the output after it is done accumulating across the I dimension
                    if int_tile.index == (int_dim_tile.tile_count - 1):
                        if apply_bias:
                            d2_tile = bias_tensor_sbuf[
                                : bxs_subtile.size,
                                nl.ds(hidden_subtile.start_offset, hidden_subtile.size),
                            ]

                            nisa.tensor_tensor(
                                dst=output_tile_sbuf_list[bxs_subtile.index][
                                    : bxs_subtile.size,
                                    nl.ds(hidden_subtile.start_offset, hidden_subtile.size),
                                ],
                                data1=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_subtile.size],
                                data2=d2_tile,
                                op=nl.add,
                            )
                        else:
                            nisa.tensor_copy(
                                output_tile_sbuf_list[bxs_subtile.index][
                                    : bxs_subtile.size,
                                    nl.ds(hidden_subtile.start_offset, hidden_subtile.size),
                                ],
                                src=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_subtile.size],
                                engine=nisa.vector_engine,
                            )

    # Perform local sendrecv if necessary to get the results from the other core
    if constants.sharded_dim == ShardedDim.INTERMEDIATE:
        bxs_dim_tile = tile_info.bxs_dim_tile
        PIPE_ID_INT_SHARD_COLLECT_RESULTS = 1
        hidden_size_per_core = mlp_params.hidden_size // constants.total_programs
        other_core_program_id = 1 - indices.program_id

        other_core_result_tensor_sbuf_list = []
        for i in range(BXS_SUBTILE_COUNT):
            tensor = alloc_stack(
                (BXS_SUBTILE_SIZE, hidden_size_per_core),
                dtype=constants.compute_data_type,
                buffer=nl.sbuf,
                name=indices.get_tensor_name("other_core_result_tensor_sbuf", f"subbxs{i}"),
            )
            other_core_result_tensor_sbuf_list.append(tensor)

        for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
            nisa.sendrecv(
                send_to_rank=other_core_program_id,
                recv_from_rank=other_core_program_id,
                src=output_tile_sbuf_list[bxs_subtile.index][
                    : bxs_subtile.size,
                    nl.ds(
                        hidden_size_per_core * other_core_program_id,
                        hidden_size_per_core,
                    ),
                ],
                dst=other_core_result_tensor_sbuf_list[bxs_subtile.index][: bxs_subtile.size, :hidden_size_per_core],
                pipe_id=PIPE_ID_INT_SHARD_COLLECT_RESULTS,
            )
            nisa.tensor_tensor(
                dst=output_tile_sbuf_list[bxs_subtile.index][
                    : bxs_subtile.size,
                    nl.ds(
                        (hidden_size_per_core * indices.program_id),
                        hidden_size_per_core,
                    ),
                ],
                data1=output_tile_sbuf_list[bxs_subtile.index][
                    : bxs_subtile.size,
                    nl.ds(hidden_size_per_core * indices.program_id, hidden_size_per_core),
                ],
                data2=other_core_result_tensor_sbuf_list[bxs_subtile.index][: bxs_subtile.size, :hidden_size_per_core],
                op=nl.add,
            )


def perform_gate_projection_if_necessary(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf: list[nl.ndarray],
    weights_sbuf_list: list[nl.ndarray],
    bias_tensor_sbuf: Optional[nl.ndarray],
    gate_scales_sbuf: Optional[nl.ndarray],
    hidden_scales_sbuf: Optional[nl.ndarray],
    proj_results_sbuf: list[nl.ndarray],
    sbm: SbufManager,
):
    """Conditionally perform gate projection [BxS, H] -> [BxS, I] with activation.

    Performs gate projection if enabled in MLP parameters, with optional bias and activation.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        indices: Batch×sequence indices for tensor naming
        source_tile_sbuf: Source tensors in SBUF
        weights_sbuf_list: Weight buffers in SBUF
        bias_tensor_sbuf: Optional bias tensor in SBUF
        gate_scales_sbuf: Optional gate scales in SBUF
        hidden_scales_sbuf: Optional hidden scales in SBUF
        proj_results_sbuf: Output projection results in SBUF
        sbm: SBUF memory manager

    Returns:
        None

    Intended Usage:
        Called to perform gate projection in gated MLP architectures
    """
    if mlpp_has_gate_projection(mlp_params):
        # Perform gate projection
        gate_proj_psum_list = []
        for bank in range(constants.required_src_proj_psum_bank_count):
            gate_proj_psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("gate_proj_psum", f"bank{bank}"),
                )
            )

        project_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            source_tile_sbuf,
            mlp_params.gate_proj_weights_tensor,
            weights_sbuf_list,
            gate_proj_psum_list,
        )
        if mlpp_has_gate_projection_bias(mlp_params):
            apply_source_projection_bias(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                gate_proj_psum_list,
                bias_tensor_sbuf,
                proj_results_sbuf,
            )
            apply_source_projection_activation(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                proj_results_sbuf,
                None,
                None,
                proj_results_sbuf,
                data_is_psum=False,
            )
        else:  # No gate projection bias
            # Apply activation function while copying the result back to SBUF
            apply_source_projection_activation(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                gate_proj_psum_list,
                gate_scales_sbuf,
                hidden_scales_sbuf,
                proj_results_sbuf,
                data_is_psum=True,
            )


def perform_up_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf: list[nl.ndarray],
    weights_sbuf_list: list[nl.ndarray],
    bias_tensor_sbuf: Optional[nl.ndarray],
    up_scales_sbuf: Optional[nl.ndarray],
    hidden_scales_sbuf: Optional[nl.ndarray],
    proj_results_sbuf: list[nl.ndarray],
    sbm: SbufManager,
):
    """Perform up projection [BxS, H] -> [BxS, I] and elementwise multiply with gate results.

    Performs up projection with optional bias, then combines with gate results if available.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        indices: Batch×sequence indices for tensor naming
        source_tile_sbuf: Source tensors in SBUF
        weights_sbuf_list: Weight buffers in SBUF
        bias_tensor_sbuf: Optional bias tensor in SBUF
        up_scales_sbuf: Optional up projection scales in SBUF
        hidden_scales_sbuf: Optional hidden scales in SBUF
        proj_results_sbuf: Output projection results in SBUF
        sbm: SBUF memory manager

    Returns:
        None

    Intended Usage:
        Called to perform up projection in gated MLP architectures
    """
    alloc_stack = sbm.alloc_stack if sbm else nl.ndarray
    if mlpp_has_gate_projection(mlp_params):
        # Create space in PSUM for up projection results
        up_proj_psum_list = []
        for bank in range(constants.required_src_proj_psum_bank_count):
            up_proj_psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("up_proj_psum", f"bank{bank}"),
                )
            )

        # Perform up projection
        project_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            source_tile_sbuf,
            mlp_params.up_proj_weights_tensor,
            weights_sbuf_list,
            up_proj_psum_list,
        )

        if mlpp_has_up_projection_bias(mlp_params):
            # We need another tensor to hold the result in SBUF of the bias application
            up_proj_res_sbuf_list = []
            for i in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                up_proj_tensor = alloc_stack(
                    (
                        tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                        tile_info.src_proj_intermediate_dim_tile.tile_count,
                        tile_info.src_proj_intermediate_dim_tile.tile_size,
                    ),
                    dtype=constants.compute_data_type,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("up_proj_res_sbuf", f"subbxs{i}"),
                )
                up_proj_res_sbuf_list.append(up_proj_tensor)

            apply_source_projection_bias(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                up_proj_psum_list,
                bias_tensor_sbuf,
                up_proj_res_sbuf_list,
            )
            # Perform the elementwise multiply between the up and gate projection results
            perform_elementwise_multiply(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                proj_results_sbuf,
                up_proj_res_sbuf_list,
                up_scales_sbuf,
                hidden_scales_sbuf,
                proj_results_sbuf,
                up_data_is_psum=False,
            )

        else:  # No up projection bias
            # Perform the elementwise multiply between the up and gate projection results
            perform_elementwise_multiply(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                proj_results_sbuf,
                up_proj_psum_list,
                up_scales_sbuf,
                hidden_scales_sbuf,
                proj_results_sbuf,
                up_data_is_psum=True,
            )
    else:  # Skip gate projection
        # Create space in PSUM for up projection results
        up_proj_psum_list = []
        for bank in range(constants.required_src_proj_psum_bank_count):
            up_proj_psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("up_proj_psum", f"bank{bank}"),
                )
            )

        # Perform up projection
        project_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            source_tile_sbuf,
            mlp_params.up_proj_weights_tensor,
            weights_sbuf_list,
            up_proj_psum_list,
        )

        if mlpp_has_up_projection_bias(mlp_params):
            apply_source_projection_bias(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                up_proj_psum_list,
                bias_tensor_sbuf,
                proj_results_sbuf,
            )
            apply_source_projection_activation(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                proj_results_sbuf,
                None,
                None,
                proj_results_sbuf,
                data_is_psum=False,
            )
        else:  # No up projection bias
            apply_source_projection_activation(
                mlp_params,
                tile_info,
                constants,
                indices.bxs_tile_idx,
                up_proj_psum_list,
                up_scales_sbuf,
                hidden_scales_sbuf,
                proj_results_sbuf,
                data_is_psum=True,
            )


# Perform source projection (up or gate) on a hidden tensor tile
def project_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    source_tile_sbuf_list: list[nl.ndarray],
    weights_tensor_hbm: nl.ndarray,
    weights_sbuf_list: list[nl.ndarray],
    proj_results_psum_list: list[nl.ndarray],
):
    """Multiply source tile [BxS, H] by weights [H, I] to produce [BxS, I].

    Performs source projection matrix multiplication with optimized tiling strategy.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        bxs_tile_idx: Current batch×sequence tile index
        source_tile_sbuf_list: Source tensors in SBUF
        weights_tensor_hbm: Weight tensor in HBM
        weights_sbuf_list: Weight buffer in SBUF
        proj_results_psum_list: Output projection results in PSUM

    Returns:
        None

    Intended Usage:
        Called to perform source projections (up/gate) in MLP forward pass
    """

    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size
    I = weights_tensor_hbm.shape[-1]
    I_SHARD_SIZE = int_dim_tile.tiled_dim_size
    I_SHARD_OFFSET = constants.get_intermediate_offset()

    # Create TiledRange for dimensions
    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    int_tiles = TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size)
    for hidden_tile in TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size):
        # Do the strided load of the weights for the current H tile
        weights_buffer_idx = hidden_tile.index % constants.src_proj_weights_buffer_count
        hidden_subtiles = TiledRange(hidden_tile, H_SUBTILE_SIZE)

        # Strided load pattern
        nisa.dma_copy(
            dst=weights_sbuf_list[weights_buffer_idx].ap(
                pattern=[
                    [I_SHARD_SIZE * 8, 128],
                    [I_SHARD_SIZE, len(hidden_subtiles)],
                    [1, I_SHARD_SIZE],
                ],
                offset=0,
            ),
            src=weights_tensor_hbm.ap(
                pattern=[[I, 128], [I * 128, len(hidden_subtiles)], [1, I_SHARD_SIZE]],
                offset=hidden_tile.index * 8 * I * 128 + I_SHARD_OFFSET,
            ),
        )

        # Do matmuls and accumulate results (H is the contraction dimension) along the BxS and I dimensions
        for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
            for hidden_subtile in hidden_subtiles:
                for int_tile in int_tiles:
                    psum_bank = bxs_subtile.index * len(int_tiles) + int_tile.index

                    st_tile = source_tile_sbuf_list[bxs_subtile.index][
                        0 : hidden_subtile.size,
                        nl.ds(hidden_subtile.start_offset, bxs_subtile.size),
                    ]

                    weights_slice_3d = weights_sbuf_list[weights_buffer_idx][
                        : hidden_subtile.size,
                        hidden_subtile.index,
                        nl.ds(int_tile.start_offset, int_tile.size),
                    ]

                    nisa.nc_matmul(
                        dst=proj_results_psum_list[psum_bank][: bxs_subtile.size, : int_tile.size],
                        stationary=st_tile,
                        moving=weights_slice_3d,
                    )
