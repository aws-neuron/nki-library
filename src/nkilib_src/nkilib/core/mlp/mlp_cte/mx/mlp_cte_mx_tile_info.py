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

"""MLP CTE MX tiling information for MX, STATIC_MX, and ROW_MX quantization types."""

from dataclasses import dataclass

import nki.language as nl
from nki.language import NKIObject

from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, get_ceil_aligned_size
from ....utils.tile_info import TiledDimInfo
from ...mlp_parameters import MLPParameters
from ..mlp_cte_sharding import DimShard, ShardedDim, is_sharded_dim_bxs

_mx_q_width = 4
_down_proj_hidden_dim_tile_size = 1024


@dataclass
class MLPCTEMXTileInfo(NKIObject):  # dim_size / tile_size / subtile_size
    src_proj_bxs_dim_tile: TiledDimInfo  # BxS / s / 256  (tile size s is variable)
    src_proj_hidden_dim_tile: TiledDimInfo  # H / 512 / 4
    intermediate_dim_tile: TiledDimInfo  # I / 512 / 4
    down_proj_bxs_dim_tile: TiledDimInfo  # BxS / s / 128
    down_proj_hidden_dim_tile: TiledDimInfo  # H / 1024 / 512


def calc_batch_seqlen_dim_tile_size(
    mlp_params: MLPParameters,
    bxs_dim_size: int,
    bxs_dim_subtile_size: int,
    mx_int_dim_tile_count: int,
) -> int:
    bxs_dim_max_subtiles = NUM_HW_PSUM_BANKS // mx_int_dim_tile_count
    # In the MX setting, during src projection, we use a wider tile size of 2*pmax.
    bxs_dim_max_subtiles *= 2
    tile_size = bxs_dim_subtile_size * bxs_dim_max_subtiles
    if bxs_dim_size == 768 and mlp_params.hidden_size == 8192:
        tile_size = min(tile_size, 384)
    else:
        aligned_bxs_dim = get_ceil_aligned_size(bxs_dim_size, nl.tile_size.pmax)
        tile_size = min(tile_size, aligned_bxs_dim)

    mx_min_tile = 2 * nl.tile_size.pmax
    kernel_assert(
        bxs_dim_subtile_size * bxs_dim_max_subtiles >= mx_min_tile,
        f"Static MX quant requires PSUM capacity for at least {mx_min_tile} BxS elements, "
        f"but intermediate_size={mlp_params.intermediate_size} is too large",
    )

    kernel_assert(
        (tile_size % nl.tile_size.pmax) == 0,
        "Internal error: The batch size/sequence length nominal tile size should always be "
        "a multiple of the partition dimension size.",
    )
    return min(tile_size, 512)


def build_mlp_cte_mx_tile_info(
    mlp_params: MLPParameters,
    sharded_dim: ShardedDim,
    dim_shard: DimShard = None,
) -> MLPCTEMXTileInfo:
    bxs_dim_size = (
        dim_shard.dim_size if is_sharded_dim_bxs(sharded_dim) else mlp_params.batch_size * mlp_params.sequence_len
    )
    intermediate_size = dim_shard.dim_size if sharded_dim == ShardedDim.INTERMEDIATE else mlp_params.intermediate_size

    src_proj_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size, nl.tile_size.pmax * _mx_q_width, _mx_q_width
    )
    intermediate_dim_tile = TiledDimInfo.build_with_subtiling(
        intermediate_size, nl.tile_size.pmax * _mx_q_width, _mx_q_width
    )
    down_proj_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size,
        _down_proj_hidden_dim_tile_size,
        nl.tile_size.gemm_moving_fmax,
    )

    bxs_dim_subtile_size = nl.tile_size.pmax
    bxs_dim_tile_size = calc_batch_seqlen_dim_tile_size(
        mlp_params,
        bxs_dim_size,
        bxs_dim_subtile_size,
        intermediate_dim_tile.tile_count,
    )
    down_proj_bxs_dim_tile = TiledDimInfo.build_with_subtiling(bxs_dim_size, bxs_dim_tile_size, bxs_dim_subtile_size)

    # Gate/Up projection uses a wider BxS subtile (2*pmax); pad tile size to fit at least one full subtile
    src_proj_bxs_subtile_size = 2 * bxs_dim_subtile_size
    src_proj_bxs_tile_size = max(bxs_dim_tile_size, src_proj_bxs_subtile_size)
    src_proj_bxs_dim_tile = TiledDimInfo.build_with_subtiling(
        bxs_dim_size, src_proj_bxs_tile_size, src_proj_bxs_subtile_size
    )

    return MLPCTEMXTileInfo(
        src_proj_bxs_dim_tile=src_proj_bxs_dim_tile,
        src_proj_hidden_dim_tile=src_proj_hidden_dim_tile,
        intermediate_dim_tile=intermediate_dim_tile,
        down_proj_bxs_dim_tile=down_proj_bxs_dim_tile,
        down_proj_hidden_dim_tile=down_proj_hidden_dim_tile,
    )
