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

"""MLP CTE MX kernel implementation for MX, STATIC_MX, and ROW_MX quantization types."""

import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.common_types import GateUpDim
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import get_program_sharding_info, is_launched_as_spmd
from ....utils.logging import get_logger
from ...mlp_parameters import (
    MLPParameters,
    mlpp_has_dma_xpose,
)
from ..mlp_cte_constants import (
    MAX_AVAILABLE_SBUF_SIZE,
    MlpBxsIndices,
    MLPCTEConstants,
    build_mlp_cte_constants,
    cleanup_mlp_cte_constants,
)
from ..mlp_cte_sharding import (
    DimShard,
    ShardedDim,
    calculate_sharding,
    is_launch_grid_valid_for_mlp,
)
from .mlp_cte_mx_allocation import (
    allocate_down_projection_scales,
    allocate_down_projection_weights,
    allocate_hidden_tensor_tile,
    allocate_intermediate_tensor_tile,
    allocate_output_tensor_tile,
    allocate_src_projection_scales,
    allocate_src_projection_weights,
)
from .mlp_cte_mx_projection import (
    perform_down_projection,
    perform_gate_up_projection,
    sync_down_proj_results_across_int_dim,
)
from .mlp_cte_mx_quantization import perform_intermediate_quantization
from .mlp_cte_mx_tensor_io import (
    load_hidden_tensor_tile_and_scales,
    load_source_projection_weight_scales,
    prepare_static_scales,
    store_half_hidden_tensor_tile,
    store_hidden_tensor_tile,
)
from .mlp_cte_mx_tile_info import (
    MLPCTEMXTileInfo,
    build_mlp_cte_mx_tile_info,
)
from .mlp_cte_mx_transpose import transpose_source_tensor_tile


def mlp_cte_mx(
    mlp_params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
):
    """
    MLP CTE kernel for MX quantization types (MX, STATIC_MX, ROW_MX).
    """
    kernel_assert(
        is_launch_grid_valid_for_mlp(),
        "Launch grid is not valid. MLP CTE only supports sharding on 1 dimension.",
    )

    top_level_interleave_degree = 1
    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=MAX_AVAILABLE_SBUF_SIZE,
        logger=get_logger("mlp_cte"),
    )
    sbm.open_scope(interleave_degree=top_level_interleave_degree, name='mlp_cte_mx')

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
            output_tensor_hbm,
            output_stored_add_tensor_hbm,
        )

    sbm.close_scope()  # mlp_cte_mx


def _execute_on_shard(
    shard_mlp_params: MLPParameters,
    sharded_dim: ShardedDim,
    dim_shard: DimShard,
    total_programs: int,
    program_id: int,
    shard_idx: int,
    sbm: SbufManager,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
):
    constants = build_mlp_cte_constants(
        shard_mlp_params,
        sharded_dim,
        total_programs,
        sbm,
        shard_idx,
        program_id,
        dim_shard,
    )
    tile_info = build_mlp_cte_mx_tile_info(shard_mlp_params, constants, sbm.get_free_space(), sharded_dim, dim_shard)
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
    cleanup_mlp_cte_constants(shard_mlp_params, sbm)


def _mlp_cte_single_shard(
    program_id: int,
    shard_idx: int,
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
    sbm: SbufManager,
):
    if constants.sharded_dim == ShardedDim.BATCH_X_SEQUENCE_LENGTH:
        batch_range = 1
    else:
        batch_range = mlp_params.batch_size

    gate_proj_weight_scales_sbuf = None
    up_proj_weight_scales_sbuf = None
    preload_w_scales = mlp_params.quant_params.is_logical_quant_row() or mlp_params.quant_params.is_quant_mx()
    if preload_w_scales:
        if not mlp_params.skip_gate_proj:
            gate_proj_weight_scales_sbuf = allocate_src_projection_scales(
                mlp_params,
                tile_info,
                constants,
                f"gate_proj_weight_scales__shard{shard_idx}__prog{program_id}",
                sbm.alloc_heap,
            )
            load_source_projection_weight_scales(
                mlp_params,
                tile_info,
                constants,
                mlp_params.quant_params.gate_w_scale,
                gate_proj_weight_scales_sbuf,
            )
        up_proj_weight_scales_sbuf = allocate_src_projection_scales(
            mlp_params,
            tile_info,
            constants,
            f"up_proj_weight_scales__shard{shard_idx}__prog{program_id}",
            sbm.alloc_heap,
        )
        load_source_projection_weight_scales(
            mlp_params,
            tile_info,
            constants,
            mlp_params.quant_params.up_w_scale,
            up_proj_weight_scales_sbuf,
        )

    gate_up_proj_static_input_scales_sbuf = None
    down_proj_static_input_scales_sbuf = None
    gate_proj_static_weight_scales_sbuf = None
    up_proj_static_weight_scales_sbuf = None
    down_proj_static_weight_scales_sbuf = None
    if mlp_params.quant_params.is_logical_quant_static():
        gate_up_proj_static_input_scales_sbuf = sbm.alloc_heap(
            (nl.tile_size.pmax, 1),
            dtype=nl.float32,
            name=f'gate_up_proj_static_input_scales__shard{shard_idx}__prog{program_id}',
        )
        down_proj_static_input_scales_sbuf = sbm.alloc_heap(
            (nl.tile_size.pmax, 1),
            dtype=nl.float32,
            name=f'down_proj_static_input_scales__shard{shard_idx}__prog{program_id}',
        )
        if not mlp_params.skip_gate_proj:
            gate_proj_static_weight_scales_sbuf = sbm.alloc_heap(
                (nl.tile_size.pmax, 1),
                dtype=nl.float32,
                name=f'gate_proj_static_weight_scales__shard{shard_idx}__prog{program_id}',
            )
        up_proj_static_weight_scales_sbuf = sbm.alloc_heap(
            (nl.tile_size.pmax, 1),
            dtype=nl.float32,
            name=f'up_proj_static_weight_scales__shard{shard_idx}__prog{program_id}',
        )
        down_proj_static_weight_scales_sbuf = sbm.alloc_heap(
            (nl.tile_size.pmax, 1),
            dtype=nl.float32,
            name=f'down_proj_static_weight_scales__shard{shard_idx}__prog{program_id}',
        )
        prepare_static_scales(
            mlp_params,
            constants,
            gate_up_proj_static_input_scales_sbuf,
            down_proj_static_input_scales_sbuf,
            gate_proj_static_weight_scales_sbuf,
            up_proj_static_weight_scales_sbuf,
            down_proj_static_weight_scales_sbuf,
        )

    for batch_idx in range(batch_range):
        for bxs_tile_idx in range(tile_info.down_proj_bxs_dim_tile.tile_count):
            indices = MlpBxsIndices(program_id, shard_idx, batch_idx, bxs_tile_idx)

            sbm.open_scope(name='intermediate_alive')

            src_proj_res_sbuf_list = []
            allocate_intermediate_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                "src_proj_res_sbuf",
                constants.compute_data_type,
                src_proj_res_sbuf_list,
                sbm.alloc_stack,
            )

            sbm.open_scope(name='gate_up_projection')

            hidden_tile_sbuf_list = []
            hidden_tile_scales_sbuf_list = []
            allocate_hidden_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                hidden_tile_scales_sbuf_list,
                sbm.alloc_stack,
                sbm.alloc_stack,
            )

            load_hidden_tensor_tile_and_scales(
                mlp_params,
                tile_info,
                constants,
                indices,
                hidden_tile_sbuf_list,
                hidden_tile_scales_sbuf_list,
                sbm,
            )

            src_proj_weights_sbuf_list = []
            allocate_src_projection_weights(
                mlp_params,
                tile_info,
                constants,
                indices,
                src_proj_weights_sbuf_list,
                sbm.alloc_stack,
                sbm,
            )

            if not mlpp_has_dma_xpose(mlp_params):
                transpose_source_tensor_tile(
                    mlp_params,
                    tile_info,
                    constants,
                    indices,
                    hidden_tile_sbuf_list,
                    hidden_tile_sbuf_list,
                    sbm,
                )

            if not mlp_params.skip_gate_proj:
                perform_gate_up_projection(
                    mlp_params,
                    tile_info,
                    constants,
                    indices,
                    GateUpDim.GATE,
                    hidden_tile_sbuf_list,
                    src_proj_weights_sbuf_list,
                    gate_proj_static_weight_scales_sbuf,
                    gate_proj_weight_scales_sbuf,
                    hidden_tile_scales_sbuf_list,
                    src_proj_res_sbuf_list,
                    sbm,
                )

            perform_gate_up_projection(
                mlp_params,
                tile_info,
                constants,
                indices,
                GateUpDim.UP,
                hidden_tile_sbuf_list,
                src_proj_weights_sbuf_list,
                up_proj_static_weight_scales_sbuf,
                up_proj_weight_scales_sbuf,
                hidden_tile_scales_sbuf_list,
                src_proj_res_sbuf_list,
                sbm,
            )

            sbm.close_scope()  # gate_up_projection (free hidden_tile_scales_sbuf)

            output_tile_sbuf_list = []
            allocate_output_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                output_tile_sbuf_list,
                sbm.alloc_heap,
            )

            intermediate_dequant_scales_sbuf_list = []
            for bxs_subtile_idx in range(tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_count):
                if mlp_params.quant_params.is_quant_mx() or mlp_params.quant_params.is_quant_row_mx():
                    intermediate_dequant_scales_sbuf = sbm.alloc_heap(
                        (
                            tile_info.intermediate_dim_tile.subtile_dim_info.tile_count,  # 128
                            tile_info.intermediate_dim_tile.tile_count,  # I / 512
                            tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_size,  # 128
                        ),
                        dtype=nl.uint8,
                        name=indices.get_tensor_name('intermediate_scale_tensor', f'subbxs{bxs_subtile_idx}'),
                    )
                    intermediate_dequant_scales_sbuf_list.append(intermediate_dequant_scales_sbuf)

            intermediate_tensor_sbuf_list = []
            allocate_intermediate_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                'intermediate_tensor',
                constants.down_proj_quant_data_type,
                intermediate_tensor_sbuf_list,
                sbm.alloc_heap,
            )

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

            sbm.close_scope()  # intermediate_alive

            down_proj_weights_sbuf_list = []
            allocate_down_projection_weights(
                mlp_params,
                tile_info,
                constants,
                indices,
                down_proj_weights_sbuf_list,
                sbm.alloc_heap,
            )

            down_proj_scales_sbuf_list = []
            if preload_w_scales:
                allocate_down_projection_scales(
                    mlp_params,
                    tile_info,
                    constants,
                    indices,
                    down_proj_scales_sbuf_list,
                    sbm.alloc_heap,
                )

            perform_down_projection(
                mlp_params,
                tile_info,
                constants,
                indices,
                intermediate_tensor_sbuf_list,
                mlp_params.down_proj_weights_tensor,
                down_proj_weights_sbuf_list,
                None,
                down_proj_static_weight_scales_sbuf,
                intermediate_dequant_scales_sbuf_list,
                down_proj_scales_sbuf_list,
                output_tile_sbuf_list,
                sbm,
            )

            if preload_w_scales:
                for scale_buffer_idx in range(len(down_proj_scales_sbuf_list)):
                    sbm.pop_heap()
            for weight_buffer_idx in range(len(down_proj_weights_sbuf_list)):
                sbm.pop_heap()
            for bxs_subtile_idx in range(tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_count):
                sbm.pop_heap()  # intermediate_tensor_sbuf
            if mlp_params.quant_params.is_quant_mx() or mlp_params.quant_params.is_quant_row_mx():
                for bxs_subtile_idx in range(tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_count):
                    sbm.pop_heap()  # intermediate_dequant_scales_sbuf

            if constants.sharded_dim == ShardedDim.INTERMEDIATE:
                sync_down_proj_results_across_int_dim(
                    mlp_params,
                    tile_info,
                    constants,
                    indices,
                    output_tile_sbuf_list,
                    sbm,
                )
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

            for bxs_subtile_idx in range(len(output_tile_sbuf_list)):
                sbm.pop_heap()  # output_tile_sbuf

            sbm.increment_section()  # mlp_cte_mx

    if preload_w_scales:
        sbm.pop_heap()  # gate_proj_weight_scales_sbuf
        sbm.pop_heap()  # up_proj_weight_scales_sbuf
    elif mlp_params.quant_params.is_logical_quant_static():
        sbm.pop_heap()  # gate_up_proj_static_input_scales_sbuf
        sbm.pop_heap()  # down_proj_static_input_scales_sbuf
        sbm.pop_heap()  # gate_proj_static_weight_scales_sbuf
        sbm.pop_heap()  # up_proj_static_weight_scales_sbuf
        sbm.pop_heap()  # down_proj_static_weight_scales_sbuf
