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

"""MLP CTE basic tiling information for non-MX quantization types."""

from dataclasses import dataclass

import nki.language as nl
from nki.language import NKIObject

from ....utils.kernel_assert import kernel_assert
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, get_ceil_aligned_size
from ....utils.tile_info import TiledDimInfo
from ...mlp_parameters import MLPParameters
from ..mlp_cte_sharding import DimShard, ShardedDim, is_sharded_dim_bxs

_xpose_hidden_dim_tile_size = 512
_layer_norm_hidden_dim_tile_size = 512
_src_proj_hidden_dim_tile_size = 1024
_src_proj_intermediate_dim_tile_size = 512
_xpose_intermediate_dim_tile_size = 512
_down_proj_hidden_dim_tile_size = 1024
_down_proj_intermediate_dim_tile_size = 128


@dataclass
class MLPCTEBasicTileInfo(NKIObject):
    bxs_dim_tile: TiledDimInfo
    layer_norm_hidden_dim_tile: TiledDimInfo
    xpose_hidden_dim_tile: TiledDimInfo
    src_proj_hidden_dim_tile: TiledDimInfo
    src_proj_intermediate_dim_tile: TiledDimInfo
    xpose_intermediate_dim_tile: TiledDimInfo
    down_proj_hidden_dim_tile: TiledDimInfo
    down_proj_intermediate_dim_tile: TiledDimInfo


def calc_batch_seqlen_dim_tile_size(
    mlp_params: MLPParameters,
    bxs_dim_size: int,
    bxs_dim_subtile_size: int,
    src_proj_int_dim_tile_count: int,
) -> int:
    bxs_dim_max_subtiles = NUM_HW_PSUM_BANKS // src_proj_int_dim_tile_count
    tile_size = bxs_dim_subtile_size * bxs_dim_max_subtiles
    if bxs_dim_size == 768 and mlp_params.hidden_size == 8192:
        tile_size = min(tile_size, 384)
    else:
        aligned_bxs_dim = get_ceil_aligned_size(bxs_dim_size, nl.tile_size.pmax)
        tile_size = min(tile_size, aligned_bxs_dim)

    kernel_assert(
        (tile_size % nl.tile_size.pmax) == 0,
        "Internal error: The batch size/sequence length nominal tile size should always be "
        "a multiple of the partition dimension size.",
    )
    return min(tile_size, 512)


def build_mlp_cte_basic_tile_info(
    mlp_params: MLPParameters,
    sharded_dim: ShardedDim,
    dim_shard: DimShard = None,
) -> MLPCTEBasicTileInfo:
    bxs_dim_size = (
        dim_shard.dim_size if is_sharded_dim_bxs(sharded_dim) else mlp_params.batch_size * mlp_params.sequence_len
    )
    intermediate_size = dim_shard.dim_size if sharded_dim == ShardedDim.INTERMEDIATE else mlp_params.intermediate_size

    layer_norm_hidden_dim_tile = TiledDimInfo.build(mlp_params.hidden_size, _layer_norm_hidden_dim_tile_size)
    xpose_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size, _xpose_hidden_dim_tile_size, nl.tile_size.pmax
    )
    src_proj_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size, _src_proj_hidden_dim_tile_size, nl.tile_size.pmax
    )
    src_proj_intermediate_dim_tile = TiledDimInfo.build(intermediate_size, _src_proj_intermediate_dim_tile_size)
    xpose_intermediate_dim_tile = TiledDimInfo.build_with_subtiling(
        intermediate_size, _xpose_intermediate_dim_tile_size, nl.tile_size.pmax
    )
    down_proj_hidden_dim_tile = TiledDimInfo.build_with_subtiling(
        mlp_params.hidden_size,
        _down_proj_hidden_dim_tile_size,
        nl.tile_size.gemm_moving_fmax,
    )
    down_proj_intermediate_dim_tile = TiledDimInfo.build(intermediate_size, _down_proj_intermediate_dim_tile_size)

    bxs_dim_subtile_size = nl.tile_size.pmax
    bxs_dim_tile_size = calc_batch_seqlen_dim_tile_size(
        mlp_params,
        bxs_dim_size,
        bxs_dim_subtile_size,
        src_proj_intermediate_dim_tile.tile_count,
    )
    bxs_dim_tile = TiledDimInfo.build_with_subtiling(bxs_dim_size, bxs_dim_tile_size, bxs_dim_subtile_size)

    return MLPCTEBasicTileInfo(
        bxs_dim_tile=bxs_dim_tile,
        layer_norm_hidden_dim_tile=layer_norm_hidden_dim_tile,
        xpose_hidden_dim_tile=xpose_hidden_dim_tile,
        src_proj_hidden_dim_tile=src_proj_hidden_dim_tile,
        src_proj_intermediate_dim_tile=src_proj_intermediate_dim_tile,
        xpose_intermediate_dim_tile=xpose_intermediate_dim_tile,
        down_proj_hidden_dim_tile=down_proj_hidden_dim_tile,
        down_proj_intermediate_dim_tile=down_proj_intermediate_dim_tile,
    )
