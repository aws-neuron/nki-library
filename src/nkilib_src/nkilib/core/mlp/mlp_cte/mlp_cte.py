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

"""MLP CTE kernel implementing context encoding computation with SPMD support and automatic sharding."""

from typing import Callable

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_program_sharding_info, is_launched_as_spmd
from ...utils.logging import get_logger
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_down_projection_bias,
    mlpp_has_gate_projection_bias,
    mlpp_has_normalization,
    mlpp_has_normalization_bias,
    mlpp_has_normalization_weights,
    mlpp_has_quantized_input,
    mlpp_has_quantized_weights,
    mlpp_has_up_projection_bias,
    mlpp_input_has_packed_scale,
)
from .mlp_cte_constants import (
    MAX_AVAILABLE_SBUF_SIZE,
    MLPCTEConstants,
    build_mlp_cte_constants,
    cleanup_mlp_cte_constants,
)
from .mlp_cte_norm import (
    apply_normalization_if_necessary,
    cleanup_heap_for_normalization,
)
from .mlp_cte_projection import (
    perform_down_projection,
    perform_gate_projection_if_necessary,
    perform_up_projection,
)
from .mlp_cte_quantization import (
    invert_static_scales,
    perform_intermediate_quantization,
)
from .mlp_cte_sharding import (
    DimShard,
    ShardedDim,
    calculate_sharding,
)
from .mlp_cte_tensor_io import (
    load_and_multiply_static_weight_scales,
    load_bias_vector,
    load_hidden_tensor_tile_opt_fused_add,
    load_source_projection_row_scales,
    load_static_input_scales,
    load_vector_across_partitions,
    store_half_hidden_tensor_tile,
    store_hidden_tensor_tile,
)
from .mlp_cte_tile_info import (
    MlpBxsIndices,
    MLPCTETileInfo,
    build_mlp_cte_tile_info,
)
from .mlp_cte_transpose import (
    transpose_intermediate_tensor_tile,
    transpose_source_tensor_tile,
)
from .mlp_cte_utils import is_launch_grid_valid_for_mlp


def mlp_cte(
    mlp_params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
):
    """
    MLP Context Encoding (CTE) kernel with SPMD support and automatic sharding.

    Performs MLP computation optimized for context encoding workloads with large batch
    and sequence dimensions. Supports both single-core and multi-core SPMD execution.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        I: Intermediate dimension size

    Args:
        mlp_params (MLPParameters): Complete MLP configuration parameters including
            hidden_tensor, weight tensors, normalization params, and quantization params
        output_tensor_hbm (nl.ndarray): [B, S, H], Pre-allocated HBM output tensor for MLP results
        output_stored_add_tensor_hbm (nl.ndarray): [B, S, H], Pre-allocated HBM tensor for
            fused addition results (optional, can be None)

    Returns:
        None: Results are written directly to output_tensor_hbm

    Notes:
        - Optimized for batch sizes B between 1 and 64
        - Best performance with sequence lengths S > 1024
        - Hidden dimensions H must be multiples of 128
        - Intermediate dimensions I should be multiples of 256
        - Supports RMS and Layer normalization fusion
        - Supports FP8 quantization (static and row-wise)

    Pseudocode:
        # Step 1: Calculate sharding strategy based on tensor dimensions
        sharding_info = calculate_sharding(mlp_params)

        # Step 2: For each shard assigned to this program
        for shard in sharding_info.shards:
            # Step 2a: Build tile info and constants
            tile_info = build_tile_info(shard)
            constants = build_constants(shard)

            # Step 2b: Execute MLP pipeline on shard
            for batch in range(batch_size):
                for bxs_tile in range(num_tiles):
                    # Load hidden tensor with optional fused add
                    hidden = load_hidden_tile(mlp_params.hidden_tensor)

                    # Apply normalization if configured
                    if has_normalization:
                        hidden = normalize(hidden)

                    # Transpose for projection matmuls
                    hidden_T = transpose(hidden)

                    # Gate projection: hidden @ gate_weights -> gate_out
                    if has_gate_projection:
                        gate_out = matmul(hidden_T, gate_weights)
                        gate_out = activation(gate_out)

                    # Up projection: hidden @ up_weights -> up_out
                    up_out = matmul(hidden_T, up_weights)

                    # Elementwise multiply gate and up results
                    if has_gate_projection:
                        intermediate = gate_out * up_out
                    else:
                        intermediate = activation(up_out)

                    # Quantize intermediate if configured
                    if has_quantization:
                        intermediate = quantize(intermediate)

                    # Down projection: intermediate @ down_weights -> output
                    output = matmul(intermediate, down_weights)

                    # Store output to HBM
                    store_output(output, output_tensor_hbm)
    """
    kernel_assert(
        is_launch_grid_valid_for_mlp(),
        "Launch grid is not valid. MLP CTE only supports sharding on 1 dimension.",
    )

    top_level_interleave_degree = 1 if mlp_params.hidden_size >= 8192 else 2
    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=MAX_AVAILABLE_SBUF_SIZE,
        logger=get_logger("mlp_cte"),
    )
    sbm.open_scope(interleave_degree=top_level_interleave_degree)

    heap_alloc = sbm.alloc_heap

    if is_launched_as_spmd():
        _, total_programs, program_id = get_program_sharding_info()

        sharding_info = calculate_sharding(mlp_params)
        for shard_idx in range(len(sharding_info.shards)):
            shard = sharding_info.shards[shard_idx]
            _execute_on_shard(
                shard.shard_mlp_params,
                sharding_info.sharded_dim,
                shard,
                total_programs,
                program_id,
                shard_idx,
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
    """
    Execute MLP computation on a single shard.

    Builds tile info and constants for the shard, then executes the full MLP pipeline.

    Args:
        shard_mlp_params (MLPParameters): MLP parameters for this shard
        sharded_dim (ShardedDim): Dimension being sharded (BATCH_X_SEQUENCE_LENGTH or INTERMEDIATE)
        dim_shard (DimShard): Shard configuration with offset and size
        total_programs (int): Total number of programs in SPMD execution
        program_id (int): Current program ID (0 to total_programs-1)
        shard_idx (int): Current shard index within this program
        sbm (SbufManager): SBUF memory manager for allocation
        heap_alloc (Callable): Memory allocator function for heap allocations
        output_tensor_hbm (nl.ndarray): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.ndarray): [B, S, H], Stored add tensor in HBM

    Returns:
        None: Results written to output tensors

    Notes:
        - Called internally for each shard in SPMD execution
        - Handles cleanup of allocated constants after execution
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
    """
    Execute MLP computation on a single shard with detailed tile processing.

    Processes all tiles within a shard, handling normalization, projections, and output storage.

    Args:
        program_id (int): Current program ID (0 to total_programs-1)
        shard_idx (int): Current shard index within this program
        mlp_params (MLPParameters): MLP configuration parameters
        tile_info (MLPCTETileInfo): Tiling information for all dimensions
        constants (MLPCTEConstants): MLP CTE constants including data types and buffer counts
        output_tensor_hbm (nl.ndarray): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.ndarray): [B, S, H], Stored add tensor in HBM
        sbm (SbufManager): SBUF memory manager for allocation

    Returns:
        None: Results written to output tensors

    Notes:
        - Called internally to process individual shards with full MLP pipeline
        - Handles bias loading, normalization, projections, and quantization
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

    # Load up and gate projection scales for row quantization if necessary
    gate_proj_row_weight_scales_sbuf = (
        load_source_projection_row_scales(
            mlp_params,
            constants,
            mlp_params.quant_params.gate_w_scale,
            heap_alloc,
        )
        if mlp_params.quant_params.is_quant_row()
        else None
    )
    up_proj_row_weight_scales_sbuf = (
        load_source_projection_row_scales(
            mlp_params,
            constants,
            mlp_params.quant_params.up_w_scale,
            heap_alloc,
        )
        if mlp_params.quant_params.is_quant_row()
        else None
    )
    # Load all static quantization scales. Multiply weight scales by input scales
    if mlp_params.quant_params.is_quant_static():
        gate_up_proj_static_input_scales_sbuf = load_static_input_scales(
            mlp_params.quant_params.gate_up_in_scale,
            heap_alloc,
        )
        down_proj_static_input_scales_sbuf = load_static_input_scales(
            mlp_params.quant_params.down_in_scale,
            heap_alloc,
        )
        gate_proj_static_weight_scales_sbuf = load_and_multiply_static_weight_scales(
            constants,
            mlp_params.quant_params.gate_w_scale,
            gate_up_proj_static_input_scales_sbuf,
            heap_alloc,
        )
        up_proj_static_weight_scales_sbuf = load_and_multiply_static_weight_scales(
            constants,
            mlp_params.quant_params.up_w_scale,
            gate_up_proj_static_input_scales_sbuf,
            heap_alloc,
        )
        down_proj_static_weight_scales_sbuf = load_and_multiply_static_weight_scales(
            constants,
            mlp_params.quant_params.down_w_scale,
            down_proj_static_input_scales_sbuf,
            heap_alloc,
        )
        invert_static_scales(down_proj_static_input_scales_sbuf)
    else:
        gate_up_proj_static_input_scales_sbuf = None
        down_proj_static_input_scales_sbuf = None
        gate_proj_static_weight_scales_sbuf = None
        up_proj_static_weight_scales_sbuf = None
        down_proj_static_weight_scales_sbuf = None

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

    # Loop over the entire batch
    for batch_idx in range(batch_range):
        # Loop over all the tiles in the B x S dimension
        for bxs_tile_idx in range(tile_info.bxs_dim_tile.tile_count):
            # Create indices for standardized naming
            indices = MlpBxsIndices(program_id, shard_idx, batch_idx, bxs_tile_idx)

            # Create sbuf array for input tensor data for the current hidden tensor tile
            hidden_tile_sbuf_list = []
            hidden_tile_scales_sbuf_list = []

            for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                hidden_tensor = heap_alloc(
                    (
                        tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                        mlp_params.hidden_size,
                    ),
                    dtype=constants.hidden_tile_data_type,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("hidden_tensor", f"subbxs{bxs_subtile_idx}"),
                )
                hidden_tile_sbuf_list.append(hidden_tensor)
                if mlpp_input_has_packed_scale(mlp_params):
                    hidden_scale_tensor = stack_alloc(
                        (tile_info.bxs_dim_tile.subtile_dim_info.tile_size, 1),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                        name=indices.get_tensor_name('hidden_scale_tensor', f'subbxs{bxs_subtile_idx}'),
                    )
                    hidden_tile_scales_sbuf_list.append(hidden_scale_tensor)

            # Create space for source projection storage in SBUF
            src_proj_res_sbuf_list = []
            for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                src_proj_tensor = stack_alloc(
                    (
                        tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                        tile_info.src_proj_intermediate_dim_tile.tile_count,
                        tile_info.src_proj_intermediate_dim_tile.tile_size,
                    ),
                    dtype=constants.compute_data_type,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("src_proj_res_sbuf", f"subbxs{bxs_subtile_idx}"),
                )
                src_proj_res_sbuf_list.append(src_proj_tensor)

            # Load the hidden tensor tile with optional fused add and optional storage of fused add result
            load_hidden_tensor_tile_opt_fused_add(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                hidden_tile_scales_sbuf_list,
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
            for weight_buffer_idx in range(constants.src_proj_weights_buffer_count):
                weights_tensor = heap_alloc(
                    (
                        tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_size,
                        tile_info.src_proj_hidden_dim_tile.subtile_dim_info.tile_count,
                        mlp_params.intermediate_size,
                    ),
                    dtype=(
                        constants.src_proj_quant_data_type
                        if mlpp_has_quantized_weights(mlp_params)
                        else constants.compute_data_type
                    ),
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name("weights_tensor", f"buf{weight_buffer_idx}"),
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
                gate_proj_row_weight_scales_sbuf,
                gate_proj_static_weight_scales_sbuf,
                hidden_tile_scales_sbuf_list,
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
                up_proj_row_weight_scales_sbuf,
                up_proj_static_weight_scales_sbuf,
                hidden_tile_scales_sbuf_list,
                src_proj_res_sbuf_list,
                sbm,
            )

            if sbm != None:
                for weight_buffer_idx in range(constants.src_proj_weights_buffer_count):
                    sbm.pop_heap()  # src_proj_weights_sbuf
                if mlpp_has_quantized_input(mlp_params):
                    for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                        sbm.pop_heap()  # hidden_tile_sbuf

            # Perform quantization on the intermediate tensor if necessary
            if mlpp_has_quantized_weights(mlp_params):
                intermediate_dequant_scales_sbuf_list = []
                intermediate_tensor_sbuf_list = []
                for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                    if mlp_params.quant_params.is_quant_row():
                        intermediate_dequant_scales_sbuf = stack_alloc(
                            (tile_info.bxs_dim_tile.subtile_dim_info.tile_size, 1),
                            dtype=nl.float32,
                            name=indices.get_tensor_name('intermediate_scale_tensor', f'subbxs{bxs_subtile_idx}'),
                        )
                        intermediate_dequant_scales_sbuf_list.append(intermediate_dequant_scales_sbuf)
                    intermediate_tensor = stack_alloc(
                        (
                            tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                            tile_info.src_proj_intermediate_dim_tile.tile_count,
                            tile_info.src_proj_intermediate_dim_tile.tile_size,
                        ),
                        dtype=constants.down_proj_quant_data_type,
                        name=indices.get_tensor_name('intermediate_tensor', f'subbxs{bxs_subtile_idx}'),
                    )
                    intermediate_tensor_sbuf_list.append(intermediate_tensor)
                perform_intermediate_quantization(
                    mlp_params,
                    tile_info,
                    constants,
                    bxs_tile_idx,
                    src_proj_res_sbuf_list,
                    intermediate_tensor_sbuf_list,
                    intermediate_dequant_scales_sbuf_list,
                    down_proj_static_input_scales_sbuf,
                    sbm,
                )
            else:
                intermediate_dequant_scales_sbuf_list = None
                intermediate_tensor_sbuf_list = src_proj_res_sbuf_list

            # Declare our SBUF tensor for the down projection weights.
            # Create "multiple buffers" using the 2nd dimension to facilitate overlapped loads.
            down_proj_weights_sbuf = []
            for weight_buffer_idx in range(constants.down_proj_weights_buffer_count):
                down_proj_weights_tensor = stack_alloc(
                    (
                        tile_info.down_proj_intermediate_dim_tile.tile_size,
                        tile_info.down_proj_hidden_dim_tile.tile_size
                        * (2 if mlpp_has_quantized_weights(mlp_params) else 1),
                    ),
                    dtype=(
                        constants.down_proj_quant_data_type
                        if mlpp_has_quantized_weights(mlp_params)
                        else constants.compute_data_type
                    ),
                    name=indices.get_tensor_name("down_proj_weights_sbuf", f"buf{weight_buffer_idx}"),
                )
                down_proj_weights_sbuf.append(down_proj_weights_tensor)

            # Perform transpose on the intermediate tensor to prepare for down projection matmuls
            transpose_intermediate_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                intermediate_tensor_sbuf_list,
                intermediate_tensor_sbuf_list,
                sbm,
            )

            if mlpp_has_quantized_input(mlp_params):
                output_tile_sbuf_list = []
                for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                    output_tile_sbuf = stack_alloc(
                        (
                            tile_info.bxs_dim_tile.subtile_dim_info.tile_size,
                            mlp_params.hidden_size,
                        ),
                        dtype=constants.compute_data_type,
                        name=indices.get_tensor_name('output_tensor', f'subbxs{bxs_subtile_idx}'),
                    )
                    output_tile_sbuf_list.append(output_tile_sbuf)
            else:
                output_tile_sbuf_list = hidden_tile_sbuf_list

            perform_down_projection(
                mlp_params,
                tile_info,
                constants,
                indices,
                intermediate_tensor_sbuf_list,
                mlp_params.down_proj_weights_tensor,
                down_proj_weights_sbuf,
                down_proj_bias_tensor_sbuf,
                down_proj_static_weight_scales_sbuf,
                intermediate_dequant_scales_sbuf_list,
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

            if sbm != None and not mlpp_has_quantized_input(mlp_params):
                for bxs_subtile_idx in range(tile_info.bxs_dim_tile.subtile_dim_info.tile_count):
                    sbm.pop_heap()  # output_tile_sbuf aka hidden_tile_sbuf

            if sbm != None:
                sbm.increment_section()

    if sbm != None:
        if mlpp_has_normalization_bias(mlp_params):
            sbm.pop_heap()  # norm_bias_tensor_sbuf
        if mlpp_has_normalization_weights(mlp_params):
            sbm.pop_heap()  # norm_weights_tensor_sbuf
        if mlp_params.quant_params.is_quant_row():
            sbm.pop_heap()  # gate_proj_row_weight_scales_sbuf
            sbm.pop_heap()  # up_proj_row_weight_scales_sbuf
        elif mlp_params.quant_params.is_quant_static():
            sbm.pop_heap()  # gate_up_proj_static_input_scales_sbuf
            sbm.pop_heap()  # down_proj_static_input_scales_sbuf
            sbm.pop_heap()  # gate_proj_static_weight_scales_sbuf
            sbm.pop_heap()  # up_proj_static_weight_scales_sbuf
            sbm.pop_heap()  # down_proj_static_weight_scales_sbuf
        if mlpp_has_down_projection_bias(mlp_params):
            sbm.pop_heap()  # down_proj_bias_tensor_sbuf
        if mlpp_has_up_projection_bias(mlp_params):
            sbm.pop_heap()  # up_proj_bias_tensor_sbuf
        if mlpp_has_gate_projection_bias(mlp_params):
            sbm.pop_heap()  # gate_proj_bias_tensor_sbuf
