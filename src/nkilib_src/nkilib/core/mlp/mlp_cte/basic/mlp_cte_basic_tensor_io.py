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

"""MLP CTE basic tensor I/O operations for non-MX quantization types."""

from typing import Callable, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import get_ceil_quotient
from ...mlp_parameters import (
    MLPParameters,
    mlpp_has_fused_add,
    mlpp_input_has_packed_scale,
    mlpp_store_fused_add,
)
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from ..mlp_cte_sharding import ShardedDim
from .mlp_cte_basic_tile_info import MLPCTEBasicTileInfo


def _calc_vec_crossload_free_dim_len(vector_tensor_hbm: nl.NkiTensor) -> tuple[int, int]:
    kernel_assert(
        (len(vector_tensor_hbm.shape) == 2) and (vector_tensor_hbm.shape[0] == 1 or vector_tensor_hbm.shape[1] == 1),
        f"Unexpected HBM tensor shape of {vector_tensor_hbm.shape}. Expected a vector with shape [1, X] or [X, 1].",
    )
    vec_len = max(vector_tensor_hbm.shape[0], vector_tensor_hbm.shape[1])
    return (vec_len, get_ceil_quotient(vec_len, nl.tile_size.pmax))


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
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.bxs_dim_tile

    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)

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
                    [[hidden_tensor_hbm_view.shape[2], p_bxs_size], [1, mlp_params.hidden_size]],
                    offset=hidden_tensor_offset,
                ),
            )


def load_fused_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
):
    bxs_dim_tile = tile_info.bxs_dim_tile

    hidden_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.hidden_tensor)
    fused_add_tensor_hbm_view = _reshape_io_tensor(constants, mlp_params.fused_add_params.fused_add_tensor)

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
                scales=[1.0, 1.0],
                reduce_op=nl.add,
            )


def store_hidden_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf: list[nl.NkiTensor],
    output_tensor_hbm: nl.NkiTensor,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
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
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    hidden_tile_sbuf: list[nl.NkiTensor],
    output_tensor_hbm: nl.NkiTensor,
):
    bxs_dim_tile = tile_info.bxs_dim_tile

    output_tensor_hbm_view = _reshape_io_tensor(constants, output_tensor_hbm)
    half_hidden_size = mlp_params.hidden_size // 2
    hidden_offset = indices.program_id * half_hidden_size
    tensor_bxs_size = constants.get_bxs_size(mlp_params)
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


def load_hidden_tensor_tile_opt_fused_add(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
    output_tile_scales_sbuf_list: Optional[nl.NkiTensor],
    output_stored_add_tensor_hbm: Optional[nl.NkiTensor],
):
    if mlpp_has_fused_add(mlp_params):
        load_fused_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list)
        if mlpp_store_fused_add(mlp_params):
            store_hidden_tensor_tile(
                mlp_params,
                tile_info,
                constants,
                indices,
                output_tile_sbuf_list,
                output_stored_add_tensor_hbm,
            )
    else:
        load_hidden_tensor_tile(mlp_params, tile_info, constants, indices, output_tile_sbuf_list)
        if mlpp_input_has_packed_scale(mlp_params):
            load_packed_hidden_scales(mlp_params, tile_info, constants, indices, output_tile_scales_sbuf_list)


def load_bias_vector(bias_tensor_hbm: nl.NkiTensor, data_type: nki.dtype, allocator: Callable) -> nl.NkiTensor:
    shuffle_group_size = 32
    num_broadcasts = nl.tile_size.pmax // shuffle_group_size
    bias_vector_len = bias_tensor_hbm.shape[1]
    bias_tensor_sbuf = allocator((nl.tile_size.pmax, bias_vector_len), dtype=data_type)

    kernel_assert(
        bias_tensor_hbm.shape[0] == 1,
        "Internal error: Bias vector first dimension should be of length 1",
    )

    shuffle_mask = [0] * shuffle_group_size
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
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_scales_sbuf_list: list[nl.NkiTensor],
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
                    offset=(indices.batch_idx * hidden_tensor_hbm_view.shape[1] * hidden_size_fp32)
                    + (bxs_offset * hidden_size_fp32)
                    + (hidden_size_fp32 - 1),
                ),
            )


def load_source_projection_row_scales(
    mlp_params: MLPParameters,
    tile_info: MLPCTEBasicTileInfo,
    constants: MLPCTEConstants,
    src_proj_scales_hbm: nl.NkiTensor,
    sbm: SbufManager,
    tensor_name: str,
) -> nl.NkiTensor:
    alloc_heap = sbm.alloc_heap if sbm else nl.NkiTensor

    src_proj_scales_sbuf = alloc_heap(
        (nl.tile_size.pmax, mlp_params.intermediate_size),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=tensor_name,
    )
    nisa.dma_copy(
        dst=src_proj_scales_sbuf[nl.ds(0, nl.tile_size.pmax), nl.ds(0, mlp_params.intermediate_size)],
        src=src_proj_scales_hbm[
            nl.ds(0, nl.tile_size.pmax), nl.ds(constants.get_intermediate_offset(), mlp_params.intermediate_size)
        ],
    )
    return src_proj_scales_sbuf


def load_vector_across_partitions(
    vector_tensor_hbm: nl.NkiTensor,
    data_type: nki.dtype,
    allocator: Callable,
    tensor_name: str,
) -> nl.NkiTensor:
    vec_len, elements_per_partition = _calc_vec_crossload_free_dim_len(vector_tensor_hbm)
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
