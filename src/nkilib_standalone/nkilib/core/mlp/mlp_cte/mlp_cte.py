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
MLP CTE Main Module

Main implementation of MLP Context Encoding (CTE) kernel with SPMD support, automatic sharding,
and complete MLP pipeline including normalization, projections, and activation functions.

"""

from typing import Callable

import nki.language as nl
import nki.isa as nisa

from ...utils.allocator import SbufManager
from ...utils.logging import Logger
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_program_sharding_info, is_launched_as_spmd
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
    mlpp_has_down_projection_bias,
    mlpp_has_normalization_weights,
    mlpp_has_normalization_bias,
    mlpp_has_normalization,
)
from .mlp_cte_constants import (
    MLPCTEConstants,
    build_mlp_cte_constants,
    cleanup_mlp_cte_constants,
    MAX_AVAILABLE_SBUF_SIZE,
)

from .mlp_cte_tile_info import (
    MLPCTETileInfo,
    MlpBxsIndices,
    build_mlp_cte_tile_info,
)
from .mlp_cte_utils import is_launch_grid_valid_for_mlp

from .mlp_cte_sharding import (
    ShardedDim,
    DimShard,
    calculate_sharding,
)
from .mlp_cte_tensor_io import (
    load_hidden_tensor_tile_opt_fused_add,
    load_bias_vector,
    load_vector_across_partitions,
    store_hidden_tensor_tile,
    store_half_hidden_tensor_tile,
)
from .mlp_cte_norm import (
    apply_normalization_if_necessary,
    cleanup_heap_for_normalization,
)
from .mlp_cte_transpose import (
    transpose_source_tensor_tile,
    transpose_intermediate_tensor_tile,
)
from .mlp_cte_projection import (
    perform_gate_projection_if_necessary,
    perform_up_projection,
    perform_down_projection,
)


def mlp_cte_invoke_kernel(
    mlp_params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
):
    """NKI kernel implementing MLP Context Encoding (CTE) computation with SPMD support.

    Performs the MLP computation optimized for context encoding workloads with large batch
    and sequence dimensions. Supports both single-core and multi-core SPMD execution with
    automatic sharding across different dimensions.

    Args:
        mlp_params: Complete MLP configuration parameters
        output_tensor_hbm: Pre-allocated HBM output tensor for MLP results
        output_stored_add_tensor_hbm: Pre-allocated HBM tensor for fused addition results

    Returns:
        None

    Intended Usage:
        This kernel is optimized for:
        - Batch sizes (B) between 1 and 64 for optimal memory utilization
        - Sequence lengths (S) > 1024 to amortize kernel launch overhead
        - Hidden dimensions (H) that are multiples of 128 for efficient vectorization
        - Intermediate dimensions (I) that are multiples of 256 for optimal tensor core usage
        - Use when sequence length is large (context encoding scenarios)
    """
    kernel_assert(
        is_launch_grid_valid_for_mlp(),
        "Launch grid is not valid. MLP CTE only supports sharding on 1 dimension.",
    )

    top_level_interleave_degree = 1 if mlp_params.hidden_size >= 8192 else 2
    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=MAX_AVAILABLE_SBUF_SIZE,
        logger=Logger("mlp_cte"),
    )
    sbm.open_scope(interleave_degree=top_level_interleave_degree)

    heap_alloc = sbm.alloc_heap

    if is_launched_as_spmd():
        _, total_programs, program_id = get_program_sharding_info()

        sharding_info = calculate_sharding(mlp_params)
        for i in range(len(sharding_info.shards)):
            shard = sharding_info.shards[i]
            _execute_on_shard(
                shard.shard_mlp_params,
                sharding_info.sharded_dim,
                shard,
                total_programs,
                program_id,
                i,
                sbm,
                heap_alloc,
                output_tensor_hbm,
                output_stored_add_tensor_hbm,
            )

    else:  # No SPMD
        dim_shard = DimShard(
            dim_offset=0,
            dim_size=mlp_params.batch_size * mlp_params.sequence_len,
            shard_mlp_params=mlp_params,
        )
        _execute_on_shard(
            mlp_params,
            ShardedDim.BATCH_X_SEQUENCE_LENGTH,
            dim_shard,
            1,
            0,
            0,
            sbm,
            heap_alloc,
            output_tensor_hbm,
            output_stored_add_tensor_hbm,
        )

    if sbm != None:
        sbm.close_scope()


def _execute_on_shard(
    shard_mlp_params: MLPParameters,
    sharded_dim: ShardedDim,
    dim_shard: DimShard,
    total_programs: int,
    program_id: int,
    shard_idx: int,
    sbm: SbufManager,
    heap_alloc: Callable,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
):
    """Execute MLP computation on a single shard.

    Builds tile info and constants for the shard, then executes the computation.

    Args:
        shard_mlp_params: MLP parameters for this shard
        sharded_dim: Dimension being sharded
        dim_shard: Shard configuration
        total_programs: Total number of programs
        program_id: Current program ID
        shard_idx: Current shard index
        sbm: SBUF memory manager
        heap_alloc: Memory allocator function
        output_tensor_hbm: Output tensor in HBM
        output_stored_add_tensor_hbm: Stored add tensor in HBM

    Returns:
        None

    Intended Usage:
        Called internally for each shard in SPMD execution
    """
    tile_info = build_mlp_cte_tile_info(shard_mlp_params, sharded_dim, dim_shard)
    constants = build_mlp_cte_constants(
        shard_mlp_params,
        tile_info,
        sharded_dim,
        total_programs,
        sbm,
        shard_idx,
        program_id,
        dim_shard,
        heap_alloc,
    )
    _mlp_cte_single_shard(
        program_id,
        shard_idx,
        shard_mlp_params,
        tile_info,
        constants,
        output_tensor_hbm,
        output_stored_add_tensor_hbm,
        sbm,
    )
    cleanup_mlp_cte_constants(sbm)


def _mlp_cte_single_shard(
    program_id: int,
    shard_idx: int,
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
    sbm: SbufManager,
):
    """Execute MLP computation on a single shard with detailed tile processing.

    Processes all tiles within a shard, handling normalization, projections, and output storage.

    Args:
        program_id: Current program ID
        shard_idx: Current shard index
        mlp_params: MLP configuration parameters
        tile_info: Tiling information
        constants: MLP CTE constants
        output_tensor_hbm: Output tensor in HBM
        output_stored_add_tensor_hbm: Stored add tensor in HBM
        sbm: SBUF memory manager

    Returns:
        None

    Intended Usage:
        Called internally to process individual shards with full MLP pipeline
    """
    # The handling of the outer loop depend on the sharding strategy.  For sequence length
    # and intermediate sharding, the outer loop range is the batch size. But if we shard on
    # batch X sequence length, the outer loop range is 1 since the batch is folded into the
    # next inner loop.
    if constants.sharded_dim == ShardedDim.BATCH_X_SEQUENCE_LENGTH:
        batch_range = 1
    else:
        batch_range = mlp_params.batch_size

    # Load any bias vectors that are necessary
    heap_alloc = sbm.alloc_heap if sbm else nl.ndarray
    stack_alloc = sbm.alloc_stack if sbm else nl.ndarray
    gate_proj_bias_tensor_sbuf = (
        load_bias_vector(
            mlp_params.bias_params.gate_proj_bias_tensor,
            constants.compute_data_type,
            heap_alloc,
        )
        if mlpp_has_gate_projection_bias(mlp_params)
        else None
    )
    up_proj_bias_tensor_sbuf = (
        load_bias_vector(
            mlp_params.bias_params.up_proj_bias_tensor,
            constants.compute_data_type,
            heap_alloc,
        )
        if mlpp_has_up_projection_bias(mlp_params)
        else None
    )
    down_proj_bias_tensor_sbuf = (
        load_bias_vector(
            mlp_params.bias_params.down_proj_bias_tensor,
            constants.compute_data_type,
            heap_alloc,
        )
        if mlpp_has_down_projection_bias(mlp_params)
        else None
    )

    # Load normalization weights and bias if necessary
    norm_weights_tensor_sbuf = (
        load_vector_across_partitions(
            mlp_params.norm_params.normalization_weights_tensor,
            constants.norm_weights_bias_data_type,
            heap_alloc,
            tensor_name=f"norm_weights_tensor__shard{shard_idx}__prog{program_id}",
        )
        if mlpp_has_normalization_weights(mlp_params)
        else None
    )
    norm_bias_tensor_sbuf = (
        load_vector_across_partitions(
            mlp_params.norm_params.normalization_bias_tensor,
            constants.norm_weights_bias_data_type,
            heap_alloc,
            tensor_name=f"norm_bias_tensor__shard{shard_idx}__prog{program_id}",
        )
        if mlpp_has_normalization_bias(mlp_params)
        else None
    )

    up_proj_scales_sbuf = None
    gate_proj_scales_sbuf = None

    # Loop over the entire batch
    for batch_idx in range(batch_range):
        # Loop over all the tiles in the B x S dimension
        for bxs_tile_idx in range(tile_info.bxs_dim_tile.tile_count):
            # Create indices for standardized naming
            indices = MlpBxsIndices(program_id, shard_idx, batch_idx, bxs_tile_idx)

            # Create sbuf array for input tensor data for the current hidden tensor tile
            hidden_tile_sbuf_list = []

            for i in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                hidden_tensor = stack_alloc(
                    (
                        tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                        mlp_params.hidden_size,
                    ),
                    dtype=constants.compute_data_type,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("hidden_tensor", f"subbxs{i}"),
                )
                hidden_tile_sbuf_list.append(hidden_tensor)

            hidden_tile_scales_sbuf = None

            # Create space for source projection storage in SBUF
            src_proj_res_sbuf_list = []
            for i in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                src_proj_tensor = stack_alloc(
                    (
                        tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                        tile_info.src_proj_intermediate_dim_tile.tile_count,
                        tile_info.src_proj_intermediate_dim_tile.tile_size,
                    ),
                    dtype=constants.compute_data_type,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("src_proj_res_sbuf", f"subbxs{i}"),
                )
                src_proj_res_sbuf_list.append(src_proj_tensor)

            # Load the hidden tensor tile with optional fused add and optional storage of fused add result
            load_hidden_tensor_tile_opt_fused_add(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                hidden_tile_scales_sbuf,
                output_stored_add_tensor_hbm,
            )
            # Apply normalization if it has been specified in the kernel parameters
            apply_normalization_if_necessary(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                hidden_tile_sbuf_list,
                heap_alloc,
            )

            # Mark that norm buffers have been allocated (if normalization is enabled)
            if mlpp_has_normalization(mlp_params):
                cleanup_heap_for_normalization(mlp_params, tile_info, sbm)

            # Declare our SBUF tensor for the source projection weights.
            # Create "multiple buffers" using the 2nd dimension to facilitate overlapped loads.
            src_proj_weights_sbuf_list = []
            for i in range(constants.src_proj_weights_buffer_count):
                weights_tensor = heap_alloc(
                    (
                        tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_size,
                        tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_count,
                        mlp_params.intermediate_size,
                    ),
                    dtype=constants.compute_data_type,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("weights_tensor", f"subbxs{i}"),
                )
                src_proj_weights_sbuf_list.append(weights_tensor)

            # Perform transpose on the source tensor to prepare for up/gate projection matmuls
            # Apply norm weights (gamma) and norm bias on hidden normalized hidden tensor
            transpose_source_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                norm_weights_tensor_sbuf,
                norm_bias_tensor_sbuf,
                hidden_tile_sbuf_list,
                sbm,
            )

            # Perform gate projection if it has been specified in the kernel parameters
            perform_gate_projection_if_necessary(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                src_proj_weights_sbuf_list,
                gate_proj_bias_tensor_sbuf,
                gate_proj_scales_sbuf,
                hidden_tile_scales_sbuf,
                src_proj_res_sbuf_list,
                sbm,
            )

            # Perform up projection
            perform_up_projection(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                src_proj_weights_sbuf_list,
                up_proj_bias_tensor_sbuf,
                up_proj_scales_sbuf,
                hidden_tile_scales_sbuf,
                src_proj_res_sbuf_list,
                sbm,
            )
            intermediate_dequant_scales_sbuf = None
            intermediate_tensor_list = src_proj_res_sbuf_list

            if sbm != None:
                for i in range(constants.src_proj_weights_buffer_count):
                    sbm.pop_heap()

            # Declare our SBUF tensor for the down projection weights.
            # Create "multiple buffers" using the 2nd dimension to facilitate overlapped loads.
            down_proj_weights_sbuf = []
            for i in range(constants.down_proj_weights_buffer_count):
                down_proj_weights_tensor = stack_alloc(
                    (
                        tile_info.down_proj_intermediate_dim_tile.tile_size,
                        tile_info.down_proj_hidden_dim_tile.tile_size,
                    ),
                    dtype=constants.compute_data_type,
                    name=indices.get_tensor_name("down_proj_weights_sbuf", f"subbxs{i}"),
                )
                down_proj_weights_sbuf.append(down_proj_weights_tensor)

            # Perform transpose on the intermediate tensor to prepare for down projection matmuls
            transpose_intermediate_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                intermediate_tensor_list,
                intermediate_tensor_list,
                sbm,
            )

            output_tile_sbuf_list = hidden_tile_sbuf_list
            perform_down_projection(
                mlp_params,
                tile_info,
                constants,
                indices,
                intermediate_tensor_list,
                mlp_params.down_proj_weights_tensor,
                down_proj_weights_sbuf,
                down_proj_bias_tensor_sbuf,
                intermediate_dequant_scales_sbuf,
                output_tile_sbuf_list,
                sbm,
            )

            # Store output into HBM
            if constants.sharded_dim == ShardedDim.INTERMEDIATE:
                store_half_hidden_tensor_tile(
                    mlp_params,
                    tile_info,
                    constants,
                    indices,
                    output_tile_sbuf_list,
                    output_tensor_hbm,
                )
            else:
                store_hidden_tensor_tile(
                    mlp_params,
                    tile_info,
                    constants,
                    indices,
                    output_tile_sbuf_list,
                    output_tensor_hbm,
                )

            if sbm != None:
                sbm.increment_section()
