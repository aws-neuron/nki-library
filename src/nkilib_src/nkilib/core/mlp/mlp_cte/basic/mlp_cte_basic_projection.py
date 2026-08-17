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

"""MLP CTE basic projection operations for non-MX quantization types."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.common_types import GateUpDim
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE, get_nl_act_fn_from_type
from ....utils.tiled_range import TiledRange, TiledRangeIterator
from ...mlp_parameters import MLPParameters
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from .mlp_cte_basic_allocation import allocate_intermediate_tensor_tile
from .mlp_cte_basic_tile_info import MLPCTEBasicTileInfo


def perform_down_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
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
    """
    Perform down projection [BxS, I] -> [BxS, H] with optional dequantization.

    Routes to standard or quantized implementation based on quantization type.

    Args:
        mlp_params (MLPParameters): MLP configuration parameters.
        tile_info (MLPCTEBasicTileInfo): Tiling information for the computation.
        constants (MLPCTEConstants): MLP CTE constants configuration.
        indices (MlpBxsIndices): Batch x sequence indices for tensor naming.
        source_tile_sbuf_list (list[nl.NkiTensor]): Transposed intermediate tensors in SBUF.
        weights_tensor_hbm (nl.NkiTensor): Down projection weight matrix on HBM.
        weights_sbuf_list (list[nl.NkiTensor]): Weight buffers in SBUF.
        bias_tensor_sbuf (Optional[nl.NkiTensor]): Optional bias tensor in SBUF.
        static_scales_sbuf (Optional[nl.NkiTensor]): Optional static dequant scales in SBUF.
        source_dequant_scales_sbuf_list (Optional[list[nl.NkiTensor]]): Optional per-row dequant scales.
        output_tile_sbuf_list (list[nl.NkiTensor]): Output buffers in SBUF.
        sbm (SbufManager): SBUF memory manager.
    """
    if mlp_params.quant_params.is_quant_row() or mlp_params.quant_params.is_quant_static():
        _perform_doublerow_down_projection(
            mlp_params,
            tile_info,
            constants,
            indices,
            source_tile_sbuf_list,
            weights_tensor_hbm,
            weights_sbuf_list,
            bias_tensor_sbuf,
            static_scales_sbuf,
            source_dequant_scales_sbuf_list,
            output_tile_sbuf_list,
            sbm,
        )
    else:
        _perform_standard_down_projection(
            mlp_params,
            tile_info,
            constants,
            indices,
            source_tile_sbuf_list,
            weights_tensor_hbm,
            weights_sbuf_list,
            bias_tensor_sbuf,
            output_tile_sbuf_list,
            sbm,
        )


def _perform_standard_down_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_tensor_hbm: nl.NkiTensor,
    weights_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform non-quantized down projection matmul: intermediate[BxS, I] @ weights[I, H] + bias -> output[BxS, H].

    Iterates over H tiles (outer) and I tiles (inner), accumulating partial products in PSUM.
    Evicts final result with optional bias addition after all I tiles are processed.
    """
    apply_bias = bias_tensor_sbuf != None
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.down_proj_hidden_dim_tile
    int_dim_tile = tile_info.down_proj_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_SHARD_OFFSET = constants.get_intermediate_offset()

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    hidden_tiles = TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size)
    int_tiles = TiledRange(mlp_params.intermediate_size, int_dim_tile.tile_size)

    for hidden_tile in hidden_tiles:
        proj_results_psum_list = []
        for psum_idx in range(NUM_HW_PSUM_BANKS):
            psum_tensor = nl.ndarray(
                (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, psum_idx * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name("down_psum_tensor", f"hidden{hidden_tile.index}__bank{psum_idx}"),
            )
            proj_results_psum_list.append(psum_tensor)

        for int_tile in int_tiles:
            weights_buffer_idx = (
                hidden_tile.index * len(int_tiles) + int_tile.index
            ) % constants.down_proj_weights_buffer_count

            nisa.dma_copy(
                src=weights_tensor_hbm[
                    nl.ds(I_SHARD_OFFSET + int_tile.start_offset, int_tile.size),
                    nl.ds(hidden_tile.start_offset, hidden_tile.size),
                ],
                dst=weights_sbuf_list[weights_buffer_idx][: int_tile.size, : hidden_tile.size],
                dge_mode=nisa.dge_mode.hwdge,
            )

            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
                psum_bank = bxs_subtile.index

                source_tile_sbuf_view = source_tile_sbuf_list[bxs_subtile.index]

                nisa.nc_matmul(
                    dst=proj_results_psum_list[psum_bank][0 : bxs_subtile.size, 0 : hidden_tile.size],
                    stationary=source_tile_sbuf_view.ap(
                        [
                            [source_tile_sbuf_view.shape[1], int_tile.size],
                            [1, bxs_subtile.size],
                        ],
                        offset=int_tile.start_offset,
                    ),
                    moving=weights_sbuf_list[weights_buffer_idx][
                        0 : int_tile.size,
                        0 : hidden_tile.size,
                    ],
                )

                if int_tile.index == (int_dim_tile.tile_count - 1):
                    if apply_bias:
                        d2_tile = bias_tensor_sbuf[
                            : bxs_subtile.size,
                            nl.ds(hidden_tile.start_offset, hidden_tile.size),
                        ]

                        nisa.tensor_tensor(
                            dst=output_tile_sbuf_list[bxs_subtile.index][
                                : bxs_subtile.size,
                                nl.ds(hidden_tile.start_offset, hidden_tile.size),
                            ],
                            data1=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            data2=d2_tile,
                            op=nl.add,
                        )
                    else:
                        nisa.tensor_copy(
                            output_tile_sbuf_list[bxs_subtile.index][
                                : bxs_subtile.size,
                                nl.ds(hidden_tile.start_offset, hidden_tile.size),
                            ],
                            src=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            engine=nisa.vector_engine,
                        )


def _perform_doublerow_down_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_tensor_hbm: nl.NkiTensor,
    weights_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    source_row_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform quantized (ROW/STATIC) down projection with double-row matmul optimization.

    Uses fp8 inputs with double-row perf mode where possible, applying dequantization
    scales (weight row scales and/or static scales) after accumulation.
    """
    kernel_assert(bias_tensor_sbuf == None, "Down projection bias is not supported with quantization")
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.down_proj_hidden_dim_tile
    int_dim_tile = tile_info.down_proj_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_TILE_SIZE = hidden_dim_tile.tile_size
    I_TILE_SIZE = int_dim_tile.tile_size
    I_SHARD_OFFSET = constants.get_intermediate_offset()
    src_proj_int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    ROUNDED_INT_DIM = src_proj_int_dim_tile.tile_count * src_proj_int_dim_tile.tile_size

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    hidden_tiles = TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size)
    int_doublerow_tile_size = 2 * int_dim_tile.tile_size
    int_doublerow_tiles = TiledRange(mlp_params.intermediate_size, int_doublerow_tile_size)

    if mlp_params.quant_params.is_quant_row():
        weight_row_scales_sbuf_list = []
        for scales_buffer_idx in range(
            min(constants.down_proj_weights_scales_buffer_count, hidden_dim_tile.tile_count)
        ):
            weight_row_scales_sbuf = sbm.alloc_stack(
                (nl.tile_size.pmax, H_TILE_SIZE),
                dtype=nl.float32,
                name=indices.get_tensor_name('down_weight_scale', f'buffer{scales_buffer_idx}'),
            )
            weight_row_scales_sbuf_list.append(weight_row_scales_sbuf)

    for hidden_tile in hidden_tiles:
        proj_results_psum_list = []
        for psum_idx in range(NUM_HW_PSUM_BANKS):
            psum_tensor = nl.ndarray(
                (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, psum_idx * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name('down_psum_tensor', f'hidden{hidden_tile.index}__bank{psum_idx}'),
            )
            proj_results_psum_list.append(psum_tensor)

        if mlp_params.quant_params.is_quant_row():
            scale_buffer_idx = hidden_tile.index % constants.down_proj_weights_scales_buffer_count
            nisa.dma_copy(
                src=mlp_params.quant_params.down_w_scale[
                    nl.ds(0, nl.tile_size.pmax), nl.ds(hidden_tile.start_offset, hidden_tile.size)
                ],
                dst=weight_row_scales_sbuf_list[scale_buffer_idx][: nl.tile_size.pmax, : hidden_tile.size],
                dge_mode=nisa.dge_mode.hwdge,
            )

        for int_doublerow_tile in int_doublerow_tiles:
            perform_doublerow_matmul = int_doublerow_tile.size > I_TILE_SIZE
            single_int_tile_size = int_doublerow_tile.size // 2

            weights_buffer_idx = (
                hidden_tile.index * len(int_doublerow_tiles) + int_doublerow_tile.index
            ) % constants.down_proj_weights_buffer_count

            weights_sbuf_view = weights_sbuf_list[weights_buffer_idx].reshape((int_dim_tile.tile_size, 2, H_TILE_SIZE))

            in_load_pattern = (
                [
                    [mlp_params.hidden_size, single_int_tile_size],
                    [single_int_tile_size * mlp_params.hidden_size, 2],
                    [1, hidden_tile.size],
                ]
                if perform_doublerow_matmul
                else [
                    [mlp_params.hidden_size, int_doublerow_tile.size],
                    [int_doublerow_tile.size * mlp_params.hidden_size, 1],
                    [1, hidden_tile.size],
                ]
            )
            in_load_offset = (
                int_doublerow_tile.index * int_doublerow_tile_size + I_SHARD_OFFSET
            ) * mlp_params.hidden_size + hidden_tile.index * H_TILE_SIZE

            out_load_pattern = (
                [[2 * H_TILE_SIZE, single_int_tile_size], [H_TILE_SIZE, 2], [1, hidden_tile.size]]
                if perform_doublerow_matmul
                else [[2 * H_TILE_SIZE, int_doublerow_tile.size], [H_TILE_SIZE, 1], [1, hidden_tile.size]]
            )

            nisa.dma_copy(
                src=weights_tensor_hbm.ap(
                    pattern=in_load_pattern, offset=in_load_offset, dtype=constants.down_proj_quant_data_type
                ),
                dst=weights_sbuf_view.ap(pattern=out_load_pattern, offset=0, dtype=constants.down_proj_quant_data_type),
            )

            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
                psum_bank = bxs_subtile.index

                dst_tile = proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size]

                source_tile_sbuf_view = source_tile_sbuf_list[bxs_subtile.index].reshape(
                    (
                        BXS_SUBTILE_SIZE,
                        ROUNDED_INT_DIM,
                    )
                )

                st_pattern = (
                    [[ROUNDED_INT_DIM, single_int_tile_size], [BXS_SUBTILE_SIZE, 2], [1, bxs_subtile.size]]
                    if perform_doublerow_matmul
                    else [[ROUNDED_INT_DIM, int_doublerow_tile.size], [1, bxs_subtile.size]]
                )
                st_offset = int_doublerow_tile.index * 2 * BXS_SUBTILE_SIZE
                intermediate_mm_in = source_tile_sbuf_view.ap(pattern=st_pattern, offset=st_offset)

                mv_pattern = (
                    [[2 * H_TILE_SIZE, single_int_tile_size], [H_TILE_SIZE, 2], [1, hidden_tile.size]]
                    if perform_doublerow_matmul
                    else [[2 * H_TILE_SIZE, int_doublerow_tile.size], [1, hidden_tile.size]]
                )
                weights_mm_in = weights_sbuf_view.ap(pattern=mv_pattern, offset=0)

                nisa.nc_matmul(
                    dst=dst_tile,
                    stationary=intermediate_mm_in,
                    moving=weights_mm_in,
                    perf_mode=('double_row' if perform_doublerow_matmul else 'none'),
                )

                if int_doublerow_tile.index == len(int_doublerow_tiles) - 1:
                    output_tile = output_tile_sbuf_list[bxs_subtile.index][
                        : bxs_subtile.size,
                        nl.ds(hidden_tile.start_offset, hidden_tile.size),
                    ]
                    if mlp_params.quant_params.is_quant_row():
                        nisa.tensor_tensor(
                            dst=output_tile,
                            data1=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            data2=weight_row_scales_sbuf_list[scale_buffer_idx][
                                : bxs_subtile.size,
                                : hidden_tile.size,
                            ],
                            op=nl.multiply,
                        )
                        nisa.activation(
                            dst=output_tile,
                            op=nl.copy,
                            data=output_tile,
                            scale=source_row_scales_sbuf_list[bxs_subtile.index][: bxs_subtile.size, 0:1],
                            bias=constants.bxs_dim_subtile_zero_bias_vector_sbuf[: bxs_subtile.size, 0:1],
                        )
                    elif mlp_params.quant_params.is_quant_static():
                        nisa.tensor_scalar(
                            dst=output_tile,
                            data=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            op0=nl.multiply,
                            operand0=static_scales_sbuf[: bxs_subtile.size, 0:1],
                            engine=nisa.vector_engine if bxs_subtile.index % 2 == 0 else nisa.scalar_engine,
                        )
                    else:
                        kernel_assert(False, "Unrecognized quantization type")


def sync_down_proj_results_across_int_dim(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Synchronize partial down projection results across cores when sharding on the intermediate dimension.

    Each core computes a partial sum over its I-shard. This function uses send/receive to
    exchange and accumulate partial results between cores to produce the final output.
    """
    bxs_dim_tile = tile_info.bxs_dim_tile
    PIPE_ID_INT_SHARD_COLLECT_RESULTS = 1
    hidden_size_per_core = mlp_params.hidden_size // constants.total_programs
    other_core_program_id = 1 - indices.program_id

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    other_core_result_tensor_sbuf_list = []
    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        tensor = sbm.alloc_heap(
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

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        sbm.pop_heap()


def perform_gate_up_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    gate_or_up: GateUpDim,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_sbuf_list: list[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_row_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    proj_results_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform a single gate or up source projection [BxS, H] -> [BxS, I] with activation and elementwise multiply.

    Routes to standard (non-quantized) or double-row (quantized fp8) implementation. For the gate
    pass, applies activation and stores in proj_results_sbuf_list. For the up pass, multiplies
    with the previously stored gate result to produce the gated intermediate.
    """
    if mlp_params.quant_params.is_quant():
        _perform_doublerow_gate_up_projection(
            mlp_params,
            tile_info,
            constants,
            indices,
            gate_or_up,
            source_tile_sbuf_list,
            weights_sbuf_list,
            bias_sbuf,
            static_scales_sbuf,
            weight_row_scales_sbuf,
            hidden_scales_sbuf_list,
            proj_results_sbuf_list,
            sbm,
        )
    else:
        _perform_standard_gate_up_projection(
            mlp_params,
            tile_info,
            constants,
            indices,
            gate_or_up,
            source_tile_sbuf_list,
            weights_sbuf_list,
            bias_sbuf,
            proj_results_sbuf_list,
            sbm,
        )


def _perform_standard_gate_up_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    gate_or_up: GateUpDim,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_sbuf_list: list[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    proj_results_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform non-quantized gate or up projection: hidden[BxS, H] @ weights[H, I] -> result[BxS, I].

    Iterates over I tiles (outer) and H tiles (inner), accumulating partial products in PSUM.
    After the last H tile, evicts the result with activation (gate) or elementwise multiply (up).
    """
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_TILE_SIZE = int_dim_tile.tile_size
    I_SHARD_OFFSET = constants.get_intermediate_offset()

    weights_tensor_hbm = (
        mlp_params.gate_proj_weights_tensor if gate_or_up == GateUpDim.GATE else mlp_params.up_proj_weights_tensor
    )

    requires_bias_output_buffer = bias_sbuf is not None and gate_or_up == GateUpDim.UP and not mlp_params.skip_gate_proj
    if requires_bias_output_buffer:
        sbm.open_scope('up_projection')
        bias_output_buffer_sbuf_list = []
        allocate_intermediate_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices,
            'up_proj_bias_output',
            constants.compute_data_type,
            bias_output_buffer_sbuf_list,
            sbm.alloc_stack,
        )
    else:
        bias_output_buffer_sbuf_list = None

    # Create TiledRange for dimensions
    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    hidden_tiles = TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size)
    for int_tile in TiledRange(mlp_params.intermediate_size, I_TILE_SIZE):  # 512 in I
        psum_list = []
        for psum_idx in range(NUM_HW_PSUM_BANKS):
            psum_name_prefix = f"{'gate' if gate_or_up == GateUpDim.GATE else 'up'}_proj_psum"
            psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, constants.psum_fmax),
                    dtype=constants.psum_accumulation_data_type,
                    buffer=nl.psum,
                    address=(0, psum_idx * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name(psum_name_prefix, f"int{int_tile.index}__bank{psum_idx}"),
                )
            )

        for hidden_tile in hidden_tiles:  # 128 in H
            weights_buffer_idx = (int_tile.index * len(hidden_tiles) + hidden_tile.index) % len(weights_sbuf_list)
            nisa.dma_copy(
                dst=weights_sbuf_list[weights_buffer_idx][: hidden_tile.size, : int_tile.size],
                src=weights_tensor_hbm[
                    nl.ds(hidden_tile.start_offset, hidden_tile.size),
                    nl.ds(I_SHARD_OFFSET + int_tile.start_offset, int_tile.size),
                ],
                dge_mode=nisa.dge_mode.hwdge,
            )
            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 128 in BxS
                psum_bank_idx = bxs_subtile.index
                nisa.nc_matmul(
                    dst=psum_list[psum_bank_idx][: bxs_subtile.size, : int_tile.size],
                    stationary=source_tile_sbuf_list[bxs_subtile.index][
                        : hidden_tile.size,
                        nl.ds(hidden_tile.start_offset, bxs_subtile.size),
                    ],
                    moving=weights_sbuf_list[weights_buffer_idx][
                        : hidden_tile.size,
                        : int_tile.size,
                    ],
                )

                if hidden_tile.index == len(hidden_tiles) - 1:
                    if gate_or_up == GateUpDim.GATE or (gate_or_up == GateUpDim.UP and mlp_params.skip_gate_proj):
                        _evict_gate_res_tile(
                            mlp_params,
                            constants,
                            psum_list[psum_bank_idx],
                            proj_results_sbuf_list,
                            bxs_subtile,
                            int_tile,
                            bias_sbuf,
                            None,  # static_scales_sbuf
                            None,  # weight_row_scales_sbuf
                            None,  # hidden_scales_sbuf_list
                        )
                    else:
                        _evict_up_res_tile(
                            mlp_params,
                            constants,
                            psum_list[psum_bank_idx],
                            proj_results_sbuf_list,
                            bxs_subtile,
                            int_tile,
                            bias_sbuf,
                            bias_output_buffer_sbuf_list,
                            None,  # static_scales_sbuf
                            None,  # weight_row_scales_sbuf
                            None,  # hidden_scales_sbuf_list
                        )
    if requires_bias_output_buffer:
        sbm.close_scope()  # up_projection


def _perform_doublerow_gate_up_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    gate_or_up: GateUpDim,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_sbuf_list: list[nl.NkiTensor],
    bias_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_row_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    proj_results_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform quantized (ROW/STATIC) gate or up projection with double-row matmul optimization.

    Uses fp8 hidden and weight inputs with double-row perf mode (2 H tiles per matmul).
    Iterates over I tiles (outer) and H double-row tiles (inner), accumulating in PSUM.
    Applies dequantization scales and activation/multiply during eviction.
    """
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_TILE_SIZE = hidden_dim_tile.tile_size
    I_TILE_SIZE = int_dim_tile.tile_size
    I_SHARD_OFFSET = constants.get_intermediate_offset()

    weights_tensor_hbm = (
        mlp_params.gate_proj_weights_tensor if gate_or_up == GateUpDim.GATE else mlp_params.up_proj_weights_tensor
    )
    I = weights_tensor_hbm.shape[-1]

    requires_bias_output_buffer = bias_sbuf is not None and gate_or_up == GateUpDim.UP and not mlp_params.skip_gate_proj
    if requires_bias_output_buffer:
        sbm.open_scope('up_projection')
        bias_output_buffer_sbuf_list = []
        allocate_intermediate_tensor_tile(
            mlp_params,
            tile_info,
            constants,
            indices,
            'up_proj_bias_output',
            constants.compute_data_type,
            bias_output_buffer_sbuf_list,
            sbm.alloc_stack,
        )
    else:
        bias_output_buffer_sbuf_list = None

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    hidden_doublerow_tiles = TiledRange(mlp_params.hidden_size, 2 * H_TILE_SIZE)

    for int_tile in TiledRange(mlp_params.intermediate_size, I_TILE_SIZE):  # 512 in I
        psum_list = []
        for psum_idx in range(NUM_HW_PSUM_BANKS):
            psum_name_prefix = f"{'gate' if gate_or_up == GateUpDim.GATE else 'up'}_proj_psum"
            psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, constants.psum_fmax),
                    dtype=constants.psum_accumulation_data_type,
                    buffer=nl.psum,
                    address=(0, psum_idx * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name(psum_name_prefix, f"int{int_tile.index}__bank{psum_idx}"),
                )
            )

        for hidden_doublerow_tile in hidden_doublerow_tiles:  # 256 in H (double-row)
            perform_doublerow_matmul = hidden_doublerow_tile.size == 2 * H_TILE_SIZE
            weights_buffer_idx = (int_tile.index * len(hidden_doublerow_tiles) + hidden_doublerow_tile.index) % len(
                weights_sbuf_list
            )

            # Load 2 H tiles × I_TILE_SIZE into buffer (128, 2*I_TILE_SIZE)
            # Buffer layout: [h_inner, row_idx * I_TILE_SIZE + i] for row_idx in {0,1}
            in_load_pattern = (
                [[I, H_TILE_SIZE], [I * H_TILE_SIZE, 2], [1, int_tile.size]]
                if perform_doublerow_matmul
                else [[I, H_TILE_SIZE], [1, int_tile.size]]
            )
            in_load_offset = hidden_doublerow_tile.start_offset * I + I_SHARD_OFFSET + int_tile.start_offset

            out_load_pattern = (
                [[2 * I_TILE_SIZE, H_TILE_SIZE], [I_TILE_SIZE, 2], [1, int_tile.size]]
                if perform_doublerow_matmul
                else [[2 * I_TILE_SIZE, H_TILE_SIZE], [1, int_tile.size]]
            )

            nisa.dma_copy(
                src=weights_tensor_hbm.ap(
                    pattern=in_load_pattern,
                    offset=in_load_offset,
                    dtype=constants.src_proj_quant_data_type,
                ),
                dst=weights_sbuf_list[weights_buffer_idx].ap(
                    pattern=out_load_pattern,
                    offset=0,
                    dtype=constants.src_proj_quant_data_type,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )

            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 128 in BxS
                psum_bank_idx = bxs_subtile.index

                st_pattern = (
                    [[mlp_params.hidden_size, H_TILE_SIZE], [BXS_SUBTILE_SIZE, 2], [1, bxs_subtile.size]]
                    if perform_doublerow_matmul
                    else [[mlp_params.hidden_size, H_TILE_SIZE], [1, bxs_subtile.size]]
                )
                st_offset = hidden_doublerow_tile.index * 2 * BXS_SUBTILE_SIZE
                hidden_mm_in = source_tile_sbuf_list[bxs_subtile.index].ap(pattern=st_pattern, offset=st_offset)

                mv_pattern = (
                    [[2 * I_TILE_SIZE, H_TILE_SIZE], [I_TILE_SIZE, 2], [1, int_tile.size]]
                    if perform_doublerow_matmul
                    else [[2 * I_TILE_SIZE, H_TILE_SIZE], [1, int_tile.size]]
                )
                weights_mm_in = weights_sbuf_list[weights_buffer_idx].ap(pattern=mv_pattern, offset=0)

                nisa.nc_matmul(
                    dst=psum_list[psum_bank_idx][: bxs_subtile.size, : int_tile.size],
                    stationary=hidden_mm_in,
                    moving=weights_mm_in,
                    perf_mode=('double_row' if perform_doublerow_matmul else 'none'),
                )

                if hidden_doublerow_tile.index == len(hidden_doublerow_tiles) - 1:
                    if gate_or_up == GateUpDim.GATE or (gate_or_up == GateUpDim.UP and mlp_params.skip_gate_proj):
                        _evict_gate_res_tile(
                            mlp_params,
                            constants,
                            psum_list[psum_bank_idx],
                            proj_results_sbuf_list,
                            bxs_subtile,
                            int_tile,
                            bias_sbuf,
                            static_scales_sbuf,
                            weight_row_scales_sbuf,
                            hidden_scales_sbuf_list,
                        )
                    else:
                        _evict_up_res_tile(
                            mlp_params,
                            constants,
                            psum_list[psum_bank_idx],
                            proj_results_sbuf_list,
                            bxs_subtile,
                            int_tile,
                            bias_sbuf,
                            bias_output_buffer_sbuf_list,
                            static_scales_sbuf,
                            weight_row_scales_sbuf,
                            hidden_scales_sbuf_list,
                        )
    if requires_bias_output_buffer:
        sbm.close_scope()  # up_projection


def _evict_gate_res_tile(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    psum_bank: nl.NkiTensor,
    proj_results_sbuf_list: list[nl.NkiTensor],
    bxs_subtile: TiledRangeIterator,
    int_tile: TiledRangeIterator,
    bias_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_row_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
):
    """
    Evict gate projection PSUM result to SBUF with dequantization and activation.

    Applies: dst = activation(dequant(psum) + bias). For non-quantized: activation(psum + bias).
    For STATIC quant: activation(psum * static_scale). For ROW quant: activation(psum * row_scale * hidden_scale).
    """
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    dst_tile = proj_results_sbuf_list[bxs_subtile.index][
        : bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)
    ]

    # Apply bias if necessary and evict to SBUF
    # Signal to downstream operations whether to read the projection result from SBUF or PSUM
    if bias_sbuf is not None:
        nisa.tensor_tensor(
            dst=dst_tile,
            op=nl.add,
            data1=psum_bank[: bxs_subtile.size, : int_tile.size],
            data2=bias_sbuf[: bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)],
        )
        proj_res = dst_tile
    else:
        proj_res = psum_bank[: bxs_subtile.size, : int_tile.size]

    if mlp_params.quant_params.is_quant_static():
        nisa.activation(
            dst=dst_tile,
            op=get_nl_act_fn_from_type(mlp_params.activation_fn),
            data=proj_res,
            scale=static_scales_sbuf[: bxs_subtile.size, 0:1],
            bias=bias_vector[: bxs_subtile.size, 0:1],
        )
    elif mlp_params.quant_params.is_quant_row():
        nisa.tensor_tensor(
            dst=dst_tile,
            op=nl.multiply,
            data1=proj_res,
            data2=weight_row_scales_sbuf[: bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)],
        )
        nisa.activation(
            dst=dst_tile,
            op=get_nl_act_fn_from_type(mlp_params.activation_fn),
            data=dst_tile,
            scale=hidden_scales_sbuf_list[bxs_subtile.index][: bxs_subtile.size, 0:1],
            bias=bias_vector[: bxs_subtile.size, 0:1],
        )
    else:  # no quant
        nisa.activation(
            dst=dst_tile,
            op=get_nl_act_fn_from_type(mlp_params.activation_fn),
            data=proj_res,
            bias=bias_vector[: bxs_subtile.size, 0:1],
        )


def _evict_up_res_tile(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    psum_bank: nl.NkiTensor,
    proj_results_sbuf_list: list[nl.NkiTensor],
    bxs_subtile: TiledRangeIterator,
    int_tile: TiledRangeIterator,
    bias_sbuf: Optional[nl.NkiTensor],
    bias_output_buffer_sbuf_list: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_row_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
):
    """
    Evict up projection PSUM result to SBUF with dequantization and elementwise multiply.

    When skip_gate_proj is True, applies activation directly (same as gate evict).
    Otherwise, multiplies the gate result (already in proj_results_sbuf_list) by the dequantized
    up PSUM: dst = gate_activated * dequant(up_psum).
    """
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    dst_tile = proj_results_sbuf_list[bxs_subtile.index][
        : bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)
    ]

    # Apply bias if necessary and evict to SBUF
    # Signal to downstream operations whether to read the projection result from SBUF or PSUM
    if bias_sbuf is not None:
        bias_buffer_tile = bias_output_buffer_sbuf_list[bxs_subtile.index][
            : bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)
        ]
        nisa.tensor_tensor(
            dst=bias_buffer_tile,
            op=nl.add,
            data1=psum_bank[: bxs_subtile.size, : int_tile.size],
            data2=bias_sbuf[: bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)],
        )
        proj_res = bias_buffer_tile
    else:
        proj_res = psum_bank[: bxs_subtile.size, : int_tile.size]

    # elementwise multiply gate result
    nisa.tensor_tensor(
        dst=dst_tile,
        data1=dst_tile,
        data2=proj_res,
        op=nl.multiply,
    )

    # dequantize
    if mlp_params.quant_params.is_quant_static():
        nisa.activation(
            dst=dst_tile,
            op=nl.copy,
            data=dst_tile,
            scale=static_scales_sbuf[: bxs_subtile.size, 0:1],
            bias=bias_vector[: bxs_subtile.size, 0:1],
        )
    elif mlp_params.quant_params.is_quant_row():
        nisa.tensor_tensor(
            dst=dst_tile,
            op=nl.multiply,
            data1=dst_tile,
            data2=weight_row_scales_sbuf[: bxs_subtile.size, nl.ds(int_tile.start_offset, int_tile.size)],
        )
        nisa.activation(
            dst=dst_tile,
            op=nl.copy,
            data=dst_tile,
            scale=hidden_scales_sbuf_list[bxs_subtile.index][: bxs_subtile.size, 0:1],
            bias=bias_vector[: bxs_subtile.size, 0:1],
        )
