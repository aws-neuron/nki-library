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
MLP CTE Transpose Module

Implements tensor transpose operations for MLP CTE kernels including source and
intermediate tensor transposition with optional scale and bias application.

"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import PSUM_BANK_SIZE, get_ceil_quotient
from ...utils.tile_info import TiledDimInfo
from ..mlp_parameters import MLPParameters
from .mlp_cte_constants import MLPCTEConstants
from .mlp_cte_tile_info import MlpBxsIndices, MLPCTETileInfo

#
# Transpose the source tensor tile in SBUF


def transpose_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.ndarray],
    scale_sbuf: Optional[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    sbm: Optional[SbufManager] = None,
):
    apply_scale = scale_sbuf != None
    apply_bias = bias_sbuf != None

    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.xpose_hidden_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size

    # We will tile PSUM to make the indexing cleaner
    psum_tile_info = TiledDimInfo.build(nl.tile_size.psum_fmax, H_SUBTILE_SIZE)

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    if apply_scale or apply_bias:
        # For the calculation of the scale and bias indices, we are assuming that
        # the hidden subtile size == scale_sbuf.shape[0] == pmax
        if apply_scale:
            kernel_assert(
                (scale_sbuf.shape[0] == nl.tile_size.pmax) and (H_SUBTILE_SIZE == nl.tile_size.pmax),
                "Scale tile must equal the hidden dimension subtile size and they must equal PMAX",
            )
        if apply_bias:
            kernel_assert(
                (bias_sbuf.shape[0] == nl.tile_size.pmax) and (H_SUBTILE_SIZE == nl.tile_size.pmax),
                "Bias tile must equal the hidden dimension subtile size and they must equal PMAX",
            )

    res_psum_list = []
    for bank in range(constants.required_src_xpose_psum_bank_count):
        res_psum_list.append(
            nl.ndarray(
                (
                    H_SUBTILE_SIZE,
                    psum_tile_info.tile_count,
                    psum_tile_info.tile_size,
                ),
                dtype=constants.xpose_data_type,
                buffer=nl.psum,
                address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name("src_transpose_res_psum", f"bank{bank}"),
            )
        )

    # Loop over all the subtiles in current B x S dimension tile
    # Usig continue here to avoid deep nested code here
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        # Calculate mask condition for this subtile using TiledDimInfo methods
        bxs_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_start

        # Only process if there are valid elements in the BxS dimension
        if bxs_subtile_rest <= 0:
            continue

        # Loop over the hidden dimension tiles
        for hidden_tile_idx in range(hidden_dim_tile.tile_count):
            # Calculate hidden dimension mask condition
            hidden_tile_rest = mlp_params.hidden_size - (hidden_tile_idx * hidden_dim_tile.tile_size)
            if hidden_tile_rest > 0:
                psum_bank = (
                    bxs_subtile_idx * hidden_dim_tile.tile_count + hidden_tile_idx
                ) % constants.required_src_xpose_psum_bank_count

                _perform_hidden_transpose(
                    mlp_params,
                    tile_info,
                    constants,
                    tensor_bxs_size,
                    indices.bxs_tile_idx,
                    bxs_subtile_idx,
                    hidden_tile_idx,
                    source_tile_sbuf_list[bxs_subtile_idx],
                    res_psum_list[psum_bank],
                    psum_tile_info,
                )

                _apply_scale_bias_if_necessary(
                    apply_scale,
                    apply_bias,
                    tile_info,
                    tensor_bxs_size,
                    indices.bxs_tile_idx,
                    bxs_subtile_idx,
                    hidden_tile_idx,
                    hidden_tile_rest,
                    res_psum_list[psum_bank],
                    output_tile_sbuf_list[bxs_subtile_idx],
                    scale_sbuf,
                    bias_sbuf,
                )


#
# Transpose the intermediate tensor tile in SBUF
def transpose_intermediate_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    int_tile_sbuf_list: list[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    sbm: SbufManager,
):
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    # We will tile PSUM to make the indexing cleaner
    psum_tile_info = TiledDimInfo.build(nl.tile_size.psum_fmax, I_SUBTILE_SIZE)

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    res_psum_list = []
    for bank in range(constants.required_int_xpose_psum_bank_count):
        psum_tensor = nl.ndarray(
            (
                I_SUBTILE_SIZE,
                psum_tile_info.tile_count,
                psum_tile_info.tile_size,
            ),
            dtype=constants.xpose_data_type,
            buffer=nl.psum,
            address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
            name=indices.get_tensor_name("int_transpose_res_psum", f"bank{bank}"),
        )
        res_psum_list.append(psum_tensor)

    # Loop over all the subtiles in current B x S dimension tile
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        # Calculate mask condition for this subtile using TiledDimInfo methods
        bxs_start = bxs_dim_tile.get_subtile_start(indices.bxs_tile_idx, bxs_subtile_idx)
        bxs_subtile_rest = tensor_bxs_size - bxs_start

        # Only process if there are valid elements in the BxS dimension
        if bxs_subtile_rest > 0:
            # Loop over the intermediate dimension tiles
            for int_tile_idx in range(int_dim_tile.tile_count):
                # Calculate intermediate dimension mask condition
                int_tile_rest = mlp_params.intermediate_size - (int_tile_idx * int_dim_tile.tile_size)

                if int_tile_rest > 0:
                    psum_bank = (
                        bxs_subtile_idx * int_dim_tile.tile_count + int_tile_idx
                    ) % constants.required_int_xpose_psum_bank_count

                    _perform_intermediate_transpose(
                        tile_info,
                        constants,
                        int_tile_idx,
                        int_tile_rest,
                        bxs_subtile_rest,
                        int_tile_sbuf_list[bxs_subtile_idx],
                        res_psum_list[psum_bank],
                    )

                    _copy_intermediate_transpose_result(
                        tile_info,
                        tensor_bxs_size,
                        indices.bxs_tile_idx,
                        bxs_subtile_idx,
                        int_tile_idx,
                        int_tile_rest,
                        res_psum_list[psum_bank],
                        output_tile_sbuf_list[bxs_subtile_idx],
                    )


def _perform_hidden_transpose(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    tensor_bxs_size: int,
    bxs_tile_idx: int,
    bxs_subtile_idx: int,
    hidden_tile_idx: int,
    source_tile_sbuf: nl.ndarray,
    res_psum_tensor: nl.ndarray,
    psum_tile_info,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.xpose_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size

    hidden_tile_rest = mlp_params.hidden_size - (hidden_tile_idx * hidden_dim_tile.tile_size)

    for hidden_subtile_idx in range(H_SUBTILE_COUNT):
        hidden_subtile_tile_rest = hidden_tile_rest - (hidden_subtile_idx * H_SUBTILE_SIZE)
        if hidden_subtile_tile_rest > 0:
            bxs_subtile_bound = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
            hidden_subtile_bound = min(hidden_subtile_tile_rest, H_SUBTILE_SIZE)

            st_tile = source_tile_sbuf[
                0:bxs_subtile_bound,
                hidden_dim_tile.get_subtile_indices(hidden_tile_idx, hidden_subtile_idx, hidden_subtile_bound),
            ]
            nisa.nc_matmul(
                dst=res_psum_tensor.ap(
                    [
                        [
                            psum_tile_info.tile_count * psum_tile_info.tile_size,
                            BXS_SUBTILE_SIZE,
                        ],
                        [1, 1],
                        [1, H_SUBTILE_SIZE],
                    ],
                    offset=hidden_subtile_idx * H_SUBTILE_SIZE,
                ),
                stationary=st_tile,
                moving=constants.identity_tensor_sbuf[:bxs_subtile_bound, :hidden_subtile_bound],
                is_moving_zero=True,
                is_transpose=constants.use_pe_xpose_flag,
            )


def _apply_scale_bias_if_necessary(
    apply_scale: bool,
    apply_bias: bool,
    tile_info: MLPCTETileInfo,
    tensor_bxs_size: int,
    bxs_tile_idx: int,
    bxs_subtile_idx: int,
    hidden_tile_idx: int,
    hidden_tile_rest: int,
    res_psum_tensor: nl.ndarray,
    output_tile_sbuf: nl.ndarray,
    scale_sbuf: Optional[nl.ndarray],
    bias_sbuf: Optional[nl.ndarray],
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    hidden_dim_tile = tile_info.xpose_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size

    if apply_scale or apply_bias:
        op0 = nl.multiply if apply_scale else nl.add
        operand0 = scale_sbuf if apply_scale else bias_sbuf
        op1 = nl.add if apply_scale and apply_bias else None
        operand1 = bias_sbuf if apply_scale and apply_bias else None

        for hidden_subtile_idx in range(H_SUBTILE_COUNT):
            hidden_subtile_tile_rest = hidden_tile_rest - (hidden_subtile_idx * H_SUBTILE_SIZE)
            if hidden_subtile_tile_rest > 0:
                hidden_subtile_bound = min(hidden_subtile_tile_rest, H_SUBTILE_SIZE)
                bxs_subtile_bound = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)

                nisa.tensor_scalar(
                    dst=output_tile_sbuf[
                        :hidden_subtile_bound,
                        hidden_dim_tile.get_subtile_indices(hidden_tile_idx, hidden_subtile_idx, bxs_subtile_bound),
                    ],
                    data=res_psum_tensor[:hidden_subtile_bound, hidden_subtile_idx, :bxs_subtile_bound],
                    op0=op0,
                    operand0=operand0[
                        : operand0.shape[0],
                        nl.ds(hidden_tile_idx * H_SUBTILE_COUNT + hidden_subtile_idx, 1),
                    ],
                    op1=op1,
                    operand1=(
                        operand1[
                            : operand1.shape[0],
                            nl.ds(hidden_tile_idx * H_SUBTILE_COUNT + hidden_subtile_idx, 1),
                        ]
                        if operand1 != None
                        else None
                    ),
                    engine=nisa.vector_engine,
                )
    else:
        hidden_subtile_bound = min(hidden_tile_rest, hidden_dim_tile.tile_size)
        res_psum_view = res_psum_tensor.reshape((nl.tile_size.pmax, nl.tile_size.psum_fmax))

        nisa.tensor_copy(
            dst=output_tile_sbuf.ap(
                [
                    [output_tile_sbuf.shape[1], BXS_SUBTILE_SIZE],
                    [1, hidden_subtile_bound],
                ],
                offset=hidden_tile_idx * hidden_dim_tile.tile_size,
            ),
            src=res_psum_view[:BXS_SUBTILE_SIZE, :hidden_subtile_bound],
            engine=nisa.vector_engine,
        )


def _perform_intermediate_transpose(
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    int_tile_idx: int,
    int_tile_rest: int,
    bxs_subtile_rest: int,
    int_tile_sbuf: nl.ndarray,
    res_psum_tensor: nl.ndarray,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    for int_subtile_idx in range(I_SUBTILE_COUNT):
        int_subtile_rest = int_tile_rest - (int_subtile_idx * I_SUBTILE_SIZE)

        if int_subtile_rest > 0:
            bxs_subtile_bound = min(bxs_subtile_rest, BXS_SUBTILE_SIZE)
            int_subtile_bound = min(int_subtile_rest, I_SUBTILE_SIZE)
            int_tile_sbuf_view = int_tile_sbuf.reshape(
                (
                    int_tile_sbuf.shape[0],
                    int_dim_tile.tile_count,
                    I_SUBTILE_COUNT,
                    I_SUBTILE_SIZE,
                )
            )
            nisa.nc_matmul(
                dst=res_psum_tensor[:int_subtile_bound, int_subtile_idx, :BXS_SUBTILE_SIZE],
                stationary=int_tile_sbuf_view[
                    :bxs_subtile_bound,
                    int_tile_idx,
                    int_subtile_idx,
                    :int_subtile_bound,
                ],
                moving=constants.identity_tensor_sbuf[:bxs_subtile_bound, :I_SUBTILE_SIZE],
                is_moving_zero=True,
                is_transpose=constants.use_pe_xpose_flag,
            )


def _copy_intermediate_transpose_result(
    tile_info: MLPCTETileInfo,
    tensor_bxs_size: int,
    bxs_tile_idx: int,
    bxs_subtile_idx: int,
    int_tile_idx: int,
    int_tile_rest: int,
    res_psum_tensor: nl.ndarray,
    output_tile_sbuf: nl.ndarray,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    actual_int_tile = min(int_tile_rest, int_dim_tile.tile_size)
    actual_int_tiles = get_ceil_quotient(actual_int_tile, I_SUBTILE_SIZE)
    bxs_subtile_bound = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
    int_subtile_bound = min(int_tile_rest, I_SUBTILE_SIZE)

    res_psum_view = res_psum_tensor.reshape((BXS_SUBTILE_SIZE, I_SUBTILE_COUNT * I_SUBTILE_SIZE))

    nisa.tensor_copy(
        dst=output_tile_sbuf.ap(
            [
                [
                    int_dim_tile.tile_count * I_SUBTILE_COUNT * I_SUBTILE_SIZE,
                    int_subtile_bound,
                ],
                [BXS_SUBTILE_SIZE, actual_int_tiles],
                [1, bxs_subtile_bound],
            ],
            offset=int_tile_idx * I_SUBTILE_COUNT * I_SUBTILE_SIZE,
        ),
        src=res_psum_view.ap(
            [
                [nl.tile_size.psum_fmax, int_subtile_bound],
                [BXS_SUBTILE_SIZE, actual_int_tiles],
                [1, bxs_subtile_bound],
            ],
            offset=0,
        ),
        engine=nisa.vector_engine,
    )
