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

"""MLP CTE MX tensor I/O operations for MX, STATIC_MX, and ROW_MX quantization types."""

import math
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.tiled_range import TiledRange
from ...mlp_parameters import (
    MLPParameters,
    mlpp_has_dma_xpose,
    mlpp_input_has_mx_block_scale,
    mlpp_input_has_packed_scale,
)
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from ..mlp_cte_sharding import ShardedDim
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo


def _reshape_io_tensor(constants: MLPCTEConstants, tensor: nl.NkiTensor) -> nl.NkiTensor:
    if constants.sharded_dim != ShardedDim.BATCH_X_SEQUENCE_LENGTH:
        return tensor
    shape_list = [1, tensor.shape[0] * tensor.shape[1]]
    for i in range(2, len(tensor.shape)):
        shape_list.append(tensor.shape[i])
    new_shape = tuple(shape_list)
    return tensor.reshape(new_shape)


def load_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 128
    H_TILE_SIZE = hidden_dim_tile.tile_size  # 512
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size  # 4
    BXS_BUFFER_COUNT = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size // nl.tile_size.pmax  # 2

    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)
    hidden_size_hbm = hidden_tensor_hbm_view.shape[-1]

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]
    tensor_bxs_offset = constants.get_bxs_offset()

    for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 128 in BxS
        bxs_256_subtile_idx = bxs_subtile.index // 2
        bxs_buffer_idx = bxs_subtile.index % 2
        output_tile_sbuf_view = output_tile_sbuf_list[bxs_256_subtile_idx].reshape(
            (
                nl.tile_size.pmax,  # 128_T
                hidden_dim_tile.tile_count,  # H/512
                BXS_BUFFER_COUNT,  # 2_T
                H_SUBTILE_SIZE,  # 4_H
                nl.tile_size.pmax,  # 128_H
            )
        )
        for h_tile in TiledRange(mlp_params.hidden_size, H_TILE_SIZE):  # 512 in H
            h_subtile_count = h_tile.size // H_SUBTILE_SIZE
            nisa.dma_copy(
                src=hidden_tensor_hbm_view.ap(
                    pattern=[
                        [hidden_size_hbm, bxs_subtile.size],
                        [h_subtile_count, H_SUBTILE_SIZE],
                        [1, h_subtile_count],
                    ],
                    offset=((tensor_bxs_offset + bxs_subtile.start_offset) * hidden_size_hbm + h_tile.start_offset),
                ),
                dst=output_tile_sbuf_view[
                    : bxs_subtile.size,
                    h_tile.index,
                    bxs_buffer_idx,
                    :H_SUBTILE_SIZE,
                    :h_subtile_count,
                ],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync if h_tile.index % 2 == 0 else nisa.engine.scalar,
            )


def load_and_transpose_hidden_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    H_TILE_COUNT = hidden_dim_tile.tile_count  # H/512
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size  # 4
    BXS_SUBTILE_SIZE = 2 * bxs_dim_tile.subtile_dim_info.tile_size  # 256
    FP32_FP8_SIZE_RATIO = 4

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]
    tensor_bxs_offset = constants.get_bxs_offset()

    hidden_size_hbm = mlp_params.hidden_tensor.shape[-1]

    for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 256 in BXS
        for hidden_tile in TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size):  # 512 in H
            hidden_subtiles = TiledRange(hidden_tile, H_SUBTILE_SIZE)

            src_pattern = [
                [hidden_size_hbm // FP32_FP8_SIZE_RATIO, bxs_subtile.size],
                [1, 1],
                [1, 1],
                [1, len(hidden_subtiles)],
            ]
            src_offset = (
                (tensor_bxs_offset + bxs_subtile.start_offset) * hidden_size_hbm + hidden_tile.start_offset
            ) // FP32_FP8_SIZE_RATIO

            bxs_x4_subtile_size_fp32 = BXS_SUBTILE_SIZE * H_SUBTILE_SIZE // FP32_FP8_SIZE_RATIO
            dst_pattern = [
                [H_TILE_COUNT * bxs_x4_subtile_size_fp32, len(hidden_subtiles)],
                [1, 1],
                [1, 1],
                [1, bxs_subtile.size],
            ]
            dst_offset = hidden_tile.index * bxs_x4_subtile_size_fp32

            nisa.dma_transpose(
                src=mlp_params.hidden_tensor.ap(src_pattern, dtype=nl.float32, offset=src_offset),
                dst=output_tile_sbuf_list[bxs_subtile.index].ap(dst_pattern, dtype=nl.float32, offset=dst_offset),
            )


def load_packed_hidden_scales(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_scales_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.src_proj_bxs_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 256
    FP32_FP8_SIZE_RATIO = 4
    SHUFFLE_GROUP_SIZE = 32
    SHUFFLE_MASK = [0] * SHUFFLE_GROUP_SIZE
    num_broadcasts = nl.tile_size.pmax // SHUFFLE_GROUP_SIZE

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]
    tensor_bxs_offset = constants.get_bxs_offset()

    hidden_size_hbm_fp32 = mlp_params.hidden_tensor.shape[-1] // FP32_FP8_SIZE_RATIO
    hidden_size_fp32 = mlp_params.hidden_size // FP32_FP8_SIZE_RATIO
    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)

    for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 256 in BxS tile
        nisa.dma_transpose(
            dst=output_tile_scales_sbuf_list[bxs_subtile.index].ap(
                [[BXS_SUBTILE_SIZE, 1], [1, 1], [1, 1], [1, bxs_subtile.size]]
            ),
            src=hidden_tensor_hbm_view.ap(
                pattern=[
                    [hidden_size_hbm_fp32, bxs_subtile.size],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
                dtype=nl.float32,
                offset=(tensor_bxs_offset + bxs_subtile.start_offset) * hidden_size_hbm_fp32 + hidden_size_fp32,
            ),
        )
        for broadcast_idx in range(num_broadcasts):
            nisa.nc_stream_shuffle(
                src=output_tile_scales_sbuf_list[bxs_subtile.index][0:1, : bxs_subtile.size],
                dst=output_tile_scales_sbuf_list[bxs_subtile.index][
                    broadcast_idx * SHUFFLE_GROUP_SIZE : (broadcast_idx + 1) * SHUFFLE_GROUP_SIZE, : bxs_subtile.size
                ],
                shuffle_mask=SHUFFLE_MASK,
            )


def load_mx_block_hidden_scales(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_scales_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """Load MX per-block scales from the packed hidden tensor (rmsnorm_mx_prefill format).

    The hidden tensor layout is [B, S, H + n_packed*128] in fp8. The scale region starts at
    offset H and contains n_packed contiguous 128-wide blocks (each holding 4 H512 tiles in
    quadrant-fold packing). Transposes [BxS, 128_H] into [128_H, n_packed, BxS] in SBUF.

    Uses DMA copy (flat) to load scale bytes into a staging SBUF (128 tokens at a time due to
    SBUF partition limit), then PE transpose (nc_transpose with fp8_e5m2 view) through PSUM to
    the final scale buffer. dma_transpose requires 2-byte dtype so cannot handle uint8 directly;
    nc_transpose supports fp8_e5m2 (1-byte) with a stride-of-2 output constraint.
    """
    bxs_dim_tile = tile_info.src_proj_bxs_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 256
    TILE_H = nl.tile_size.pmax  # 128
    PMAX = nl.tile_size.pmax  # 128 — max partition dimension for SBUF
    _FP8_TP_OUT_STEP = 2  # PE fp8 transpose output interleave factor
    _UINT8_TP_VIEW_DTYPE = nl.float8_e5m2  # Same byte width as uint8

    n_H512 = mlp_params.hidden_size // 512
    n_packed = math.ceil(n_H512 / 4)

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]
    tensor_bxs_offset = constants.get_bxs_offset()

    hidden_size_hbm = mlp_params.hidden_tensor.shape[-1]
    scale_region_size = n_packed * TILE_H  # Total scale bytes per token row

    # Staging SBUF: partition dim is capped at 128 (PMAX), so we process 128 tokens at a time.
    stack_alloc = sbm.alloc_stack if sbm else nl.NkiTensor
    staging_sbuf = stack_alloc(
        (PMAX, scale_region_size),
        dtype=_UINT8_TP_VIEW_DTYPE,
        buffer=nl.sbuf,
        name=indices.get_tensor_name('mx_scale_staging', ''),
    )

    # PSUM buffer for PE transpose: [128_H, 1, 128_BxS, 2_interleave]
    scale_psum = nl.ndarray(
        (TILE_H, 1, PMAX, _FP8_TP_OUT_STEP),
        dtype=_UINT8_TP_VIEW_DTYPE,
        buffer=nl.psum,
    )

    for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):
        # Process 128 tokens at a time (SBUF partition limit)
        for chunk in TiledRange(bxs_subtile.size, PMAX):
            chunk_bxs_offset = bxs_subtile.start_offset + chunk.start_offset

            # Step 1: DMA copy scale bytes from HBM → staging SBUF (flat, no transpose)
            nisa.dma_copy(
                src=mlp_params.hidden_tensor.ap(
                    pattern=[
                        [hidden_size_hbm, chunk.size],
                        [1, scale_region_size],
                    ],
                    dtype=_UINT8_TP_VIEW_DTYPE,
                    offset=(tensor_bxs_offset + chunk_bxs_offset) * hidden_size_hbm + mlp_params.hidden_size,
                ),
                dst=staging_sbuf[: chunk.size, :scale_region_size],
            )

            # Step 2: PE transpose each pack block [chunk_size, 128] → [128, chunk_size] via PSUM
            for pack_idx in range(n_packed):
                nisa.nc_transpose(
                    data=staging_sbuf.ap(
                        pattern=[[scale_region_size, chunk.size], [1, TILE_H]],
                        offset=pack_idx * TILE_H,
                        dtype=_UINT8_TP_VIEW_DTYPE,
                    ),
                    dst=scale_psum[:, 0, : chunk.size, 0],
                )
                # Step 3: Copy from PSUM to final scale buffer at the correct BxS offset
                nisa.tensor_copy(
                    src=scale_psum[:, 0, : chunk.size, 0],
                    dst=output_tile_scales_sbuf_list[bxs_subtile.index].ap(
                        pattern=[[n_packed * BXS_SUBTILE_SIZE, TILE_H], [1, chunk.size]],
                        offset=pack_idx * BXS_SUBTILE_SIZE + chunk.start_offset,
                        dtype=_UINT8_TP_VIEW_DTYPE,
                    ),
                )


def store_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf: list[nl.NkiTensor],
    output_tensor_hbm: nl.NkiTensor,
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size

    output_tensor_hbm_view = _reshape_io_tensor(constants, output_tensor_hbm)
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


def store_half_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf: list[nl.NkiTensor],
    output_tensor_hbm: nl.NkiTensor,
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile

    output_tensor_hbm_view = _reshape_io_tensor(constants, output_tensor_hbm)
    half_hidden_size = mlp_params.hidden_size // 2
    hidden_offset = indices.program_id * half_hidden_size
    H = output_tensor_hbm_view.shape[2]

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(indices.bxs_tile_idx, bxs_subtile_idx)
        if p_bxs_size > 0:
            bxs_offset = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
            output_offset = indices.batch_idx * H * output_tensor_hbm_view.shape[1] + bxs_offset * H + hidden_offset
            nisa.dma_copy(
                dst=output_tensor_hbm_view.ap([[H, p_bxs_size], [1, half_hidden_size]], offset=output_offset),
                src=hidden_tile_sbuf[bxs_subtile_idx][0:p_bxs_size, hidden_offset : hidden_offset + half_hidden_size],
            )


def load_hidden_tensor_tile_and_scales(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
    output_tile_scales_sbuf_list: Optional[nl.NkiTensor],
    sbm: SbufManager = None,
):
    if mlpp_has_dma_xpose(mlp_params):
        load_and_transpose_hidden_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list)
    else:
        load_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list)
    if mlpp_input_has_mx_block_scale(mlp_params):
        load_mx_block_hidden_scales(mlp_params, tile_info, constants, indices, output_tile_scales_sbuf_list, sbm)
    elif mlpp_input_has_packed_scale(mlp_params):
        load_packed_hidden_scales(mlp_params, tile_info, constants, indices, output_tile_scales_sbuf_list)


def load_source_projection_weight_scales(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    src_proj_scales_hbm: nl.NkiTensor,
    src_proj_scales_sbuf: nl.NkiTensor,
) -> nl.NkiTensor:
    if mlp_params.quant_params.is_quant_row_mx():
        int_dim_tile = tile_info.intermediate_dim_tile
        INT_TILE_COUNT = int_dim_tile.tile_count  # I / 512
        INT_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
        INT_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128

        nisa.dma_copy(
            dst=src_proj_scales_sbuf[:INT_SUBTILE_COUNT, :INT_TILE_COUNT, :INT_SUBTILE_SIZE],
            src=src_proj_scales_hbm[
                :INT_SUBTILE_COUNT,
                nl.ds(
                    math.ceil(constants.get_intermediate_offset() / int_dim_tile.tile_size),
                    INT_TILE_COUNT,
                ),
                :INT_SUBTILE_SIZE,
            ],
        )
    elif mlp_params.quant_params.is_quant_mx():
        hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
        int_dim_tile = tile_info.intermediate_dim_tile
        H_TILE_COUNT = hidden_dim_tile.tile_count  # H / 512
        H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count  # 128
        I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
        I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128
        QUADRANT_SIZE = 32
        PARTITIONS_PER_SLOT = 4  # how many partitions to load per quadrant
        NUM_SLOTS = 4
        I_SHARD_OFFSET = constants.get_intermediate_offset()

        if mlp_params.quant_params.use_folded_mx_scales:
            # Scales are pre-folded into the physical SBUF layout [128, ceil((H/512)/4), I].
            # A single DMA fills the whole buffer; only the I dimension is sliced for sharding.
            nisa.dma_copy(
                dst=src_proj_scales_sbuf[
                    :H_SUBTILE_COUNT,
                    :,
                    : mlp_params.intermediate_size,
                ],
                src=src_proj_scales_hbm[
                    :H_SUBTILE_COUNT,
                    :,
                    nl.ds(I_SHARD_OFFSET, mlp_params.intermediate_size),
                ],
            )
        else:
            # Scale HBM shape: [16, H/512, I/512, 4, 128] — already in physical I order
            src_proj_scales_hbm_view = src_proj_scales_hbm.reshape(
                (
                    src_proj_scales_hbm.shape[0],
                    H_TILE_COUNT,
                    src_proj_scales_hbm.shape[2] * I_SUBTILE_SIZE * I_SUBTILE_COUNT,
                )
            )
            for quadrant_idx in range(math.ceil(H_SUBTILE_COUNT / QUADRANT_SIZE)):
                for h_tile in TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size):
                    slot_idx = h_tile.index % NUM_SLOTS
                    nisa.dma_copy(
                        dst=src_proj_scales_sbuf[
                            nl.ds(quadrant_idx * QUADRANT_SIZE + slot_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                            h_tile.index // NUM_SLOTS,
                            : mlp_params.intermediate_size,
                        ],
                        src=src_proj_scales_hbm_view[
                            nl.ds(quadrant_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                            h_tile.index,
                            nl.ds(I_SHARD_OFFSET, mlp_params.intermediate_size),
                        ],
                    )
    return src_proj_scales_sbuf


def prepare_static_scales(
    mlp_params: MLPParameters,
    constants: MLPCTEConstants,
    gate_up_proj_static_input_scales_sbuf: nl.NkiTensor,
    down_proj_static_input_scales_sbuf: nl.NkiTensor,
    gate_proj_static_weight_scales_sbuf: nl.NkiTensor,
    up_proj_static_weight_scales_sbuf: nl.NkiTensor,
    down_proj_static_weight_scales_sbuf: nl.NkiTensor,
):
    nisa.dma_copy(
        dst=gate_up_proj_static_input_scales_sbuf[0 : nl.tile_size.pmax, 0:1],
        src=mlp_params.quant_params.gate_up_in_scale[0 : nl.tile_size.pmax, 0:1],
    )
    nisa.dma_copy(
        dst=down_proj_static_input_scales_sbuf[0 : nl.tile_size.pmax, 0:1],
        src=mlp_params.quant_params.down_in_scale[0 : nl.tile_size.pmax, 0:1],
    )
    if not mlp_params.skip_gate_proj:
        _load_and_multiply_static_weight_scales(
            constants,
            mlp_params.quant_params.gate_w_scale,
            gate_up_proj_static_input_scales_sbuf,
            gate_proj_static_weight_scales_sbuf,
        )
    _load_and_multiply_static_weight_scales(
        constants,
        mlp_params.quant_params.up_w_scale,
        gate_up_proj_static_input_scales_sbuf,
        up_proj_static_weight_scales_sbuf,
    )
    _load_and_multiply_static_weight_scales(
        constants,
        mlp_params.quant_params.down_w_scale,
        down_proj_static_input_scales_sbuf,
        down_proj_static_weight_scales_sbuf,
    )
    nisa.reciprocal(
        dst=down_proj_static_input_scales_sbuf[0 : nl.tile_size.pmax, 0:1],
        data=down_proj_static_input_scales_sbuf[0 : nl.tile_size.pmax, 0:1],
    )


def _load_and_multiply_static_weight_scales(
    constants: MLPCTEConstants,
    static_weight_scale_hbm: nl.NkiTensor,
    static_input_scale_sbuf: nl.NkiTensor,
    static_weight_scale_sbuf: nl.NkiTensor,
) -> nl.NkiTensor:
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
