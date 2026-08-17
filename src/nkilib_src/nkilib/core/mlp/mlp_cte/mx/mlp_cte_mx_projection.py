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

"""MLP CTE MX projection operations for MX, STATIC_MX, and ROW_MX quantization types."""

import math
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE
from ....utils.tiled_range import TiledRange
from ...mlp_parameters import MLPParameters, mlpp_input_has_mx_block_scale
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from ..mlp_cte_sharding import ShardedDim
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo
from .mlp_cte_mx_utils import (
    apply_source_projection_activation,
    perform_elementwise_multiply,
)


def perform_down_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_tensor_hbm: nl.NkiTensor,
    weights_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    source_dequant_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    kernel_assert(bias_tensor_sbuf == None, "Down projection bias is not supported with MX quantization")
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.down_proj_hidden_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_TILE_SIZE = hidden_dim_tile.tile_size  # 1024
    I_SHARD_OFFSET = constants.get_intermediate_offset()
    I_TILE_SIZE = int_dim_tile.tile_size  # 512
    I_TILE_COUNT = int_dim_tile.tile_count  # I/512
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128
    FULL_I_TILE_COUNT = weights_tensor_hbm.shape[1]  # fp8[128_I, I/512, H, 4]

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    hidden_tiles = TiledRange(mlp_params.hidden_size, H_TILE_SIZE)
    int_tiles = TiledRange(mlp_params.intermediate_size, I_TILE_SIZE)

    alloc_stack = sbm.alloc_stack if sbm else nl.NkiTensor

    if mlp_params.quant_params.is_quant_row_mx():
        weight_row_scales_sbuf_list = []
        for scales_buffer_idx in range(
            min(constants.down_proj_weights_scales_buffer_count, hidden_dim_tile.tile_count)
        ):
            weight_row_scales_sbuf = alloc_stack(
                (nl.tile_size.pmax, H_TILE_SIZE),
                dtype=nl.float32,
                name=indices.get_tensor_name('down_weight_scale', f'buffer{scales_buffer_idx}'),
            )
            weight_row_scales_sbuf_list.append(weight_row_scales_sbuf)
    elif mlp_params.quant_params.is_quant_mx():
        weight_mx_scales_sbuf_list = []
        for scales_buffer_idx in range(
            min(constants.down_proj_weights_scales_buffer_count, hidden_dim_tile.tile_count)
        ):
            weight_mx_scales_sbuf = alloc_stack(
                (I_SUBTILE_COUNT, I_TILE_COUNT, H_TILE_SIZE),
                dtype=nl.uint8,
                name=indices.get_tensor_name('down_weight_scale', f'buffer{scales_buffer_idx}'),
            )
            weight_mx_scales_sbuf_list.append(weight_mx_scales_sbuf)

    for hidden_tile in hidden_tiles:  # 1024 in H
        proj_results_psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            psum_tensor = nl.ndarray(
                (nl.tile_size.pmax, constants.psum_fmax),
                dtype=constants.psum_accumulation_data_type,
                buffer=nl.psum,
                address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name('down_psum_tensor', f'hidden{hidden_tile.index}__bank{bank}'),
            )
            proj_results_psum_list.append(psum_tensor)

        scale_buffer_idx = hidden_tile.index % constants.down_proj_weights_scales_buffer_count
        if mlp_params.quant_params.is_quant_row_mx():
            nisa.dma_copy(
                src=mlp_params.quant_params.down_w_scale[
                    : nl.tile_size.pmax, nl.ds(hidden_tile.start_offset, hidden_tile.size)
                ],
                dst=weight_row_scales_sbuf_list[scale_buffer_idx][: nl.tile_size.pmax, : hidden_tile.size],
            )
        elif mlp_params.quant_params.is_quant_mx():
            QUADRANT_SIZE = 32
            PARTITIONS_PER_SLOT = 4
            I_TILE_OFFSET = I_SHARD_OFFSET // I_TILE_SIZE
            for quadrant_idx in range(math.ceil(I_SUBTILE_COUNT / QUADRANT_SIZE)):
                nisa.dma_copy(
                    src=mlp_params.quant_params.down_w_scale[
                        nl.ds(quadrant_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                        nl.ds(I_TILE_OFFSET, I_TILE_COUNT),
                        nl.ds(hidden_tile.start_offset, hidden_tile.size),
                    ],
                    dst=weight_mx_scales_sbuf_list[scale_buffer_idx][
                        nl.ds(quadrant_idx * QUADRANT_SIZE, PARTITIONS_PER_SLOT), :I_TILE_COUNT, : hidden_tile.size
                    ],
                )

        for int_tile in int_tiles:  # 512 in I
            weights_buffer_idx = (
                hidden_tile.index * len(int_tiles) + int_tile.index
            ) % constants.down_proj_weights_buffer_count

            weights_sbuf_view = weights_sbuf_list[weights_buffer_idx].reshape(
                (I_SUBTILE_COUNT, H_TILE_SIZE, I_SUBTILE_SIZE),
            )
            int_subtiles = TiledRange(int_tile, I_SUBTILE_SIZE)

            nisa.dma_copy(
                src=weights_tensor_hbm.ap(
                    pattern=[
                        [FULL_I_TILE_COUNT * mlp_params.hidden_size * I_SUBTILE_SIZE, len(int_subtiles)],
                        [I_SUBTILE_SIZE, hidden_tile.size],
                        [1, I_SUBTILE_SIZE],
                    ],
                    offset=(I_SHARD_OFFSET * mlp_params.hidden_size // I_SUBTILE_COUNT)
                    + (int_tile.index * mlp_params.hidden_size * I_SUBTILE_SIZE)
                    + (hidden_tile.index * H_TILE_SIZE * I_SUBTILE_SIZE),
                    dtype=constants.down_proj_quant_data_type,
                ),
                dst=weights_sbuf_view[: len(int_subtiles), : hidden_tile.size, :I_SUBTILE_SIZE],
            )

            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 128 in BxS
                psum_bank = bxs_subtile.index  # this will at most use 4 banks

                if mlp_params.quant_params.is_quant_row_mx() or mlp_params.quant_params.is_quant_mx():
                    stationary_scale = source_dequant_scales_sbuf_list[bxs_subtile.index][
                        : len(int_subtiles), int_tile.index, : bxs_subtile.size
                    ]
                else:
                    stationary_scale = constants.mx_stationary_neutral_scale_sbuf[
                        : len(int_subtiles), : bxs_subtile.size
                    ]

                if mlp_params.quant_params.is_quant_mx():
                    moving_scale = weight_mx_scales_sbuf_list[scale_buffer_idx][
                        : len(int_subtiles), int_tile.index, : hidden_tile.size
                    ]
                else:
                    moving_scale = constants.mx_moving_neutral_scale_sbuf[: len(int_subtiles), : hidden_tile.size]

                nisa.nc_matmul_mx(
                    dst=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                    stationary=source_tile_sbuf_list[bxs_subtile.index].ap(
                        pattern=[
                            [I_TILE_COUNT * BXS_SUBTILE_SIZE, len(int_subtiles)],
                            [1, bxs_subtile.size],
                        ],
                        offset=(int_tile.index * BXS_SUBTILE_SIZE),
                        dtype=nl.float8_e4m3fn_x4,
                    ),
                    moving=weights_sbuf_list[weights_buffer_idx].ap(
                        pattern=[
                            [H_TILE_SIZE, len(int_subtiles)],
                            [1, hidden_tile.size],
                        ],
                        offset=0,
                        dtype=nl.float8_e4m3fn_x4,
                    ),
                    stationary_scale=stationary_scale,
                    moving_scale=moving_scale,
                )

                # Copy each completed portion to the output after it is done accumulating across the I dimension
                if int_tile.index == I_TILE_COUNT - 1:
                    output_tile = output_tile_sbuf_list[bxs_subtile.index][
                        : bxs_subtile.size,
                        nl.ds(hidden_tile.start_offset, hidden_tile.size),
                    ]
                    if mlp_params.quant_params.is_quant_static_mx():
                        nisa.activation(
                            dst=output_tile,
                            op=nl.copy,
                            data=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            scale=static_scales_sbuf[: bxs_subtile.size, 0:1],
                            bias=constants.bxs_dim_subtile_zero_bias_vector_sbuf[: bxs_subtile.size, 0:1],
                        )
                    elif mlp_params.quant_params.is_quant_row_mx():
                        nisa.tensor_tensor(
                            dst=output_tile,
                            data1=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            data2=weight_row_scales_sbuf_list[scale_buffer_idx][: bxs_subtile.size, : hidden_tile.size],
                            op=nl.multiply,
                        )
                    elif mlp_params.quant_params.is_quant_mx():
                        nisa.tensor_copy(
                            dst=output_tile,
                            src=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                        )
    if constants.sharded_dim == ShardedDim.INTERMEDIATE:
        _sync_down_proj_results_across_int_dim(
            mlp_params,
            tile_info,
            constants,
            indices,
            output_tile_sbuf_list,
            sbm,
        )


def _sync_down_proj_results_across_int_dim(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    PIPE_ID_INT_SHARD_COLLECT_RESULTS = 1
    hidden_size_per_core = mlp_params.hidden_size // constants.total_programs
    other_core_program_id = 1 - indices.program_id

    alloc_stack = sbm.alloc_stack if sbm else nl.NkiTensor

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    other_core_result_tensor_sbuf_list = []
    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        tensor = alloc_stack(
            (bxs_dim_tile.subtile_dim_info.tile_size, hidden_size_per_core),
            dtype=constants.compute_data_type,
            buffer=nl.sbuf,
            name=indices.get_tensor_name("other_core_result_tensor_sbuf", f"subbxs{bxs_subtile_idx}"),
        )
        other_core_result_tensor_sbuf_list.append(tensor)

    for bxs_subtile in TiledRange(current_bxs_tile, bxs_dim_tile.subtile_dim_info.tile_size):
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
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf: list[nl.NkiTensor],
    weights_sbuf_list: list[nl.NkiTensor],
    gate_weight_scales_sbuf: Optional[nl.NkiTensor],
    gate_static_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    proj_results_sbuf: list[nl.NkiTensor],
    sbm: SbufManager,
):
    if not mlp_params.skip_gate_proj:
        gate_proj_psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            gate_proj_psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, constants.psum_fmax),
                    dtype=constants.psum_accumulation_data_type,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("gate_proj_psum", f"bank{bank}"),
                )
            )

        _project_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            source_tile_sbuf,
            mlp_params.gate_proj_weights_tensor,
            weights_sbuf_list,
            gate_weight_scales_sbuf,
            gate_proj_psum_list,
            hidden_scales_sbuf_list,
        )
        apply_source_projection_activation(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            gate_proj_psum_list,
            gate_weight_scales_sbuf,
            gate_static_scales_sbuf,
            hidden_scales_sbuf_list,
            proj_results_sbuf,
            data_is_psum=True,
        )


def perform_up_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf: list[nl.NkiTensor],
    weights_sbuf_list: list[nl.NkiTensor],
    up_weight_scales_sbuf: Optional[nl.NkiTensor],
    up_static_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    proj_results_sbuf: list[nl.NkiTensor],
    sbm: SbufManager,
):
    if not mlp_params.skip_gate_proj:
        up_proj_psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            up_proj_psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, constants.psum_fmax),
                    dtype=constants.psum_accumulation_data_type,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("up_proj_psum", f"bank{bank}"),
                )
            )

        _project_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            source_tile_sbuf,
            mlp_params.up_proj_weights_tensor,
            weights_sbuf_list,
            up_weight_scales_sbuf,
            up_proj_psum_list,
            hidden_scales_sbuf_list,
        )

        perform_elementwise_multiply(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            proj_results_sbuf,
            up_proj_psum_list,
            up_weight_scales_sbuf,
            up_static_scales_sbuf,
            hidden_scales_sbuf_list,
            proj_results_sbuf,
            up_data_is_psum=True,
        )
    else:  # Skip gate projection
        up_proj_psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            up_proj_psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, constants.psum_fmax),
                    dtype=constants.psum_accumulation_data_type,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("up_proj_psum", f"bank{bank}"),
                )
            )

        _project_source_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            source_tile_sbuf,
            mlp_params.up_proj_weights_tensor,
            weights_sbuf_list,
            up_weight_scales_sbuf,
            up_proj_psum_list,
            hidden_scales_sbuf_list,
        )

        apply_source_projection_activation(
            mlp_params,
            tile_info,
            constants,
            indices.bxs_tile_idx,
            up_proj_psum_list,
            up_weight_scales_sbuf,
            up_static_scales_sbuf,
            hidden_scales_sbuf_list,
            proj_results_sbuf,
            data_is_psum=True,
        )


def _project_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_tensor_hbm: nl.NkiTensor,
    weights_sbuf_list: list[nl.NkiTensor],
    weight_scales_sbuf: Optional[nl.NkiTensor],
    proj_results_psum_list: list[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]] = None,
):
    bxs_dim_tile = tile_info.src_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 256
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size  # 4
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count  # 128
    I_TILE_COUNT = int_dim_tile.tile_count  # I/512
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128
    SLOTS_PER_QUADRANT = 4
    PARTITIONS_PER_SLOT = 4
    # weights_tensor_hbm shape is [128_H, H/512, I/512, 4_I, 128_I, 4_H]
    I = weights_tensor_hbm.shape[2] * weights_tensor_hbm.shape[3] * weights_tensor_hbm.shape[4]
    I_SHARD_OFFSET = constants.get_intermediate_offset()

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[bxs_tile_idx]

    _has_mx_block_scale = mlpp_input_has_mx_block_scale(mlp_params)

    for hidden_tile in TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size):  # 512 in H
        weights_buffer_idx = hidden_tile.index % len(weights_sbuf_list)
        hidden_subtiles = TiledRange(hidden_tile, H_SUBTILE_SIZE)

        weights_sbuf_view = weights_sbuf_list[weights_buffer_idx].reshape(
            (
                H_SUBTILE_COUNT,  # 128
                I_TILE_COUNT * I_SUBTILE_SIZE * I_SUBTILE_COUNT * H_SUBTILE_SIZE,  # I/512 * 4 * 128 * 4
            )
        )
        weights_hbm_view = weights_tensor_hbm.reshape(
            (
                H_SUBTILE_COUNT,  # 128
                hidden_dim_tile.tile_count,  # H / 512
                I * H_SUBTILE_SIZE,  # I/512 * 4 * 128 * 4
            )
        )
        nisa.dma_copy(
            dst=weights_sbuf_view[
                : len(hidden_subtiles), : I_TILE_COUNT * I_SUBTILE_SIZE * I_SUBTILE_COUNT * H_SUBTILE_SIZE
            ],
            src=weights_hbm_view[
                : len(hidden_subtiles),
                hidden_tile.index,
                nl.ds(
                    I_SHARD_OFFSET * H_SUBTILE_SIZE, I_TILE_COUNT * I_SUBTILE_SIZE * I_SUBTILE_COUNT * H_SUBTILE_SIZE
                ),
            ],
        )
        for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 256 in BxS
            if _has_mx_block_scale:
                _packed_buf_idx = hidden_tile.index // SLOTS_PER_QUADRANT
                _slot_part_off = (hidden_tile.index % SLOTS_PER_QUADRANT) * PARTITIONS_PER_SLOT
                moving_scale = hidden_scales_sbuf_list[bxs_subtile.index][
                    _slot_part_off:,
                    _packed_buf_idx,
                    : bxs_subtile.size,
                ]
            else:
                moving_scale = constants.mx_moving_neutral_scale_sbuf[: len(hidden_subtiles), : bxs_subtile.size]

            for int_tile in TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size):  # 512 in I
                psum_bank = bxs_subtile.index * int_dim_tile.tile_count + int_tile.index
                for int_row_tile in TiledRange(int_tile, I_SUBTILE_COUNT):  # 128 in 512
                    stationary_scale = (
                        weight_scales_sbuf[
                            PARTITIONS_PER_SLOT * (hidden_tile.index % SLOTS_PER_QUADRANT) : len(hidden_subtiles),
                            hidden_tile.index // SLOTS_PER_QUADRANT,
                            nl.ds(int_row_tile.start_offset, int_row_tile.size),
                        ]
                        if mlp_params.quant_params.is_quant_mx()
                        else constants.mx_stationary_neutral_scale_sbuf[: len(hidden_subtiles), : int_row_tile.size]
                    )
                    nisa.nc_matmul_mx(
                        dst=proj_results_psum_list[psum_bank].ap(
                            pattern=[
                                [BXS_SUBTILE_SIZE * I_SUBTILE_SIZE, int_row_tile.size],
                                [1, bxs_subtile.size],
                            ],
                            offset=(int_row_tile.index * BXS_SUBTILE_SIZE),
                        ),
                        stationary=weights_sbuf_list[weights_buffer_idx].ap(
                            pattern=[
                                [I_TILE_COUNT * I_SUBTILE_SIZE * I_SUBTILE_COUNT, len(hidden_subtiles)],
                                [1, int_row_tile.size],
                            ],
                            offset=(int_tile.index * I_SUBTILE_SIZE * I_SUBTILE_COUNT)
                            + (int_row_tile.index * I_SUBTILE_COUNT),
                            dtype=nl.float8_e4m3fn_x4,
                        ),
                        moving=source_tile_sbuf_list[bxs_subtile.index].ap(
                            pattern=[
                                [hidden_dim_tile.tile_count * BXS_SUBTILE_SIZE, len(hidden_subtiles)],
                                [1, bxs_subtile.size],
                            ],
                            offset=(hidden_tile.index * BXS_SUBTILE_SIZE),
                            dtype=nl.float8_e4m3fn_x4,
                        ),
                        stationary_scale=stationary_scale,
                        moving_scale=moving_scale,
                    )
