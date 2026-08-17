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

"""MLP CTE MX transpose operations for MX quantization types."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE
from ....utils.tiled_range import TiledRange
from ...mlp_parameters import MLPParameters, mlpp_has_quantized_input
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo


def transpose_source_tensor_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.NkiTensor],
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: Optional[SbufManager] = None,
):
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 128
    MX_BXS_SUBTILE_SIZE = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256
    H_TILE_SIZE = hidden_dim_tile.tile_size  # 512
    H_SUBTILE_COUNT = hidden_dim_tile.subtile_dim_info.tile_count  # 128
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size  # 4
    BXS_BUFFER_COUNT = MX_BXS_SUBTILE_SIZE // BXS_SUBTILE_SIZE  # 2

    # 1-byte dtype PE transpose requires a step size of 2
    psum_step_size = 2 if mlpp_has_quantized_input(mlp_params) else 1

    bxs_tiles = TiledRange(constants.get_bxs_size(mlp_params), bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    # calculate how many elements of H will fit in PSUM at once
    # each psum bank will be [P(128_H), 2_T, 4_H, 128_T] which contains 512 elements of H
    max_h_elements_in_psum = NUM_HW_PSUM_BANKS * (PSUM_BANK_SIZE // BXS_BUFFER_COUNT // 2)
    max_h_tiles_in_psum = max_h_elements_in_psum // H_TILE_SIZE

    for h_psum_tile in TiledRange(mlp_params.hidden_size, max_h_elements_in_psum):  # 4096 in H
        # One h_psum_tile is large enough to fill PSUM. Create our PSUM buffers at this point.
        res_psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            res_psum_list.append(
                nl.ndarray(
                    (
                        nl.tile_size.pmax,  # 128_H
                        BXS_BUFFER_COUNT,  # 2_T
                        H_SUBTILE_SIZE,  # 4_H
                        nl.tile_size.pmax,  # 128_T
                        psum_step_size,
                    ),
                    dtype=source_tile_sbuf_list[0].dtype,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name("src_transpose_res_psum", f"itr{h_psum_tile.index}_bank{bank}"),
                )
            )
        for mx_bxs_subtile in TiledRange(current_bxs_tile, MX_BXS_SUBTILE_SIZE):  # 256 in BxS
            source_tile_sbuf_view = source_tile_sbuf_list[mx_bxs_subtile.index].reshape(
                (
                    nl.tile_size.pmax,  # 128_T
                    hidden_dim_tile.tile_count,  # H/512
                    BXS_BUFFER_COUNT,  # 2_T
                    H_SUBTILE_SIZE,  # 4_H
                    nl.tile_size.pmax,  # 128_H
                )
            )
            output_tile_sbuf_view = output_tile_sbuf_list[mx_bxs_subtile.index].reshape(
                (
                    nl.tile_size.pmax,  # 128_H
                    hidden_dim_tile.tile_count,  # H/512
                    BXS_BUFFER_COUNT,  # 2_T
                    BXS_SUBTILE_SIZE,  # 128_T
                    H_SUBTILE_SIZE,  # 4_H
                )
            )
            for h_tile in TiledRange(h_psum_tile, H_TILE_SIZE):  # 512 in 4096_H
                # h_tile index < 8; can be used directly as the PSUM buffer index
                bxs_subtiles = TiledRange(mx_bxs_subtile, BXS_SUBTILE_SIZE)
                for bxs_subtile in bxs_subtiles:  # 128 in 256_T
                    for h_row_tile in TiledRange(h_tile, H_SUBTILE_COUNT):  # 128 in 512_H
                        nisa.nc_transpose(
                            dst=res_psum_list[h_tile.index][
                                : h_row_tile.size,
                                bxs_subtile.index,
                                h_row_tile.index,
                                : bxs_subtile.size,
                                0,
                            ],
                            data=source_tile_sbuf_view[
                                : bxs_subtile.size,
                                h_psum_tile.index * max_h_tiles_in_psum + h_tile.index,
                                bxs_subtile.index,
                                h_row_tile.index,
                                : h_row_tile.size,
                            ],
                        )
                    # Evict half of the PSUM buffer (everything after the 2_T dim)
                    # [128_H, 2_T, 4_H, 128_T] -> [128_H, 2_T, 128_T, 4_H]
                    nisa.tensor_copy(
                        src=res_psum_list[h_tile.index].ap(
                            [
                                [
                                    BXS_BUFFER_COUNT * H_SUBTILE_SIZE * nl.tile_size.pmax * psum_step_size,
                                    h_row_tile.size,
                                ],
                                [psum_step_size, bxs_subtile.size],
                                [nl.tile_size.pmax * psum_step_size, H_SUBTILE_SIZE],
                            ],
                            offset=(bxs_subtile.index * H_SUBTILE_SIZE * nl.tile_size.pmax * psum_step_size),
                        ),
                        dst=output_tile_sbuf_view[
                            : h_row_tile.size,
                            h_psum_tile.index * max_h_tiles_in_psum + h_tile.index,
                            bxs_subtile.index,
                            : bxs_subtile.size,
                            :H_SUBTILE_SIZE,
                        ],
                    )
