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
MLP CTE Tile Info Module

Defines tiling information and index management for MLP CTE kernels including
batch×sequence indices and tensor naming utilities.

"""

from dataclasses import dataclass
from typing import Optional

import nki.language as nl
from nki.language import NKIObject

from ...utils.kernel_helpers import get_ceil_aligned_size, NUM_HW_PSUM_BANKS
from ...utils.kernel_assert import kernel_assert
from ...utils.tile_info import TiledDimInfo
from ..mlp_parameters import MLPParameters
from .mlp_cte_sharding import ShardedDim, DimShard, is_sharded_dim_bxs

#
# Local constants
_xpose_hidden_dim_tile_size = 512
_layer_norm_hidden_dim_tile_size = 512
_src_proj_hidden_dim_tile_size = 1024
_src_proj_intermediate_dim_tile_size = 512
_xpose_intermediate_dim_tile_size = 512
_down_proj_hidden_dim_tile_size = 1024
_down_proj_intermediate_dim_tile_size = 128


#
# ***************
# MLP Indices
# ***************
#


@dataclass(frozen=True)
class MlpBxsIndices(NKIObject):
    program_id: int
    shard_idx: int
    batch_idx: int
    bxs_tile_idx: int

    def get_tensor_name(self, object_name: str, suffix: Optional[str] = None) -> str:
        """Generate standardized tensor names with optional suffix.

        Creates unique tensor names incorporating shard, program, batch, and tile indices
        to ensure no naming conflicts in multi-core execution.

        Args:
            object_name: Base name for the tensor object
            suffix: Optional suffix to append to the name

        Returns:
            Standardized tensor name string

        Intended Usage:
            Called when allocating tensors to ensure unique naming across all execution contexts
        """
        base_name = (
            f"{object_name}__shard{self.shard_idx}__prog{self.program_id}__"
            f"batch{self.batch_idx}__bxs{self.bxs_tile_idx}"
        )
        if suffix:
            return f"{base_name}__{suffix}"
        return base_name


#
# Calculate the tile size for the batch/sequence dimension
def calc_batch_seqlen_dim_tile_size(
    mlp_params: MLPParameters,
    bxs_dim_size: int,
    bxs_dim_subtile_size: int,
    src_proj_int_dim_tile_count: int,
) -> int:
    """Calculate the tile size for the batch/sequence dimension.

    Computes optimal tile size for batch×sequence dimension based on PSUM bank constraints
    and model-specific optimizations.

    Args:
        mlp_params: MLP configuration parameters
        bxs_dim_size: Size of batch×sequence dimension
        bxs_dim_subtile_size: Subtile size for batch×sequence dimension
        src_proj_int_dim_tile_count: Number of tiles in source projection intermediate dimension

    Returns:
        Calculated tile size for batch×sequence dimension

    Intended Usage:
        Called during tile info construction to determine optimal tiling strategy
    """
    # TODO: Implement tile size calculation properly
    # We have to ensure that the tile size not exceed the number of PSUM banks needed during up/gate
    # projection.  It is related to the structure of the inner loops.

    bxs_dim_max_subtiles = NUM_HW_PSUM_BANKS // src_proj_int_dim_tile_count
    # This is the max tile size we can choose
    tile_size = bxs_dim_subtile_size * bxs_dim_max_subtiles
    # Special tiling optimization for LLaMA3 70B (heuristic) is to use a tile size of 384 if we can
    if bxs_dim_size == 768 and mlp_params.hidden_size == 8192:
        tile_size = min(tile_size, 384)
    else:
        aligned_bxs_dim = get_ceil_aligned_size(bxs_dim_size, nl.tile_size.pmax)
        tile_size = min(tile_size, aligned_bxs_dim)

    # There are assumptions in the code that this is true
    kernel_assert(
        (tile_size % nl.tile_size.pmax) == 0,
        "Internal error: The batch size/sequence length nominal tile size should always be "
        "a multiple of the partition dimension size.",
    )
    # TODO: Revisit this.  For now, we limit the max tile size to 512 since that is what has been tested.
    return min(tile_size, 512)


#
#
# Primary tuple that holds all tile information needed for the kernel
#
@dataclass
class MLPCTETileInfo(NKIObject):
    # Tile information for the batch/sequence dimension
    bxs_dim_tile: TiledDimInfo
    # Tile information for the hidden dimension for layer norm
    layer_norm_hidden_dim_tile: TiledDimInfo
    # Tile information for the hidden dimension for transposes
    xpose_hidden_dim_tile: TiledDimInfo
    # Tile information for the hidden dimension for up/gate projection
    src_proj_hidden_dim_tile: TiledDimInfo
    # Tile information for the intermediate dimension for up/gate projection
    src_proj_intermediate_dim_tile: TiledDimInfo
    # Tile information for the intermediate dimension for transposes
    xpose_intermediate_dim_tile: TiledDimInfo
    # Tile information for the hidden dimension for down projection
    down_proj_hidden_dim_tile: TiledDimInfo
    # Tile information for the intermediate dimension for down projection
    down_proj_intermediate_dim_tile: TiledDimInfo


def build_mlp_cte_tile_info(
    mlp_params: MLPParameters,
    sharded_dim: ShardedDim,
    dim_shard: DimShard = None,
) -> MLPCTETileInfo:
    """Build MLP CTE tile information configuration.

    Creates tiling information for all dimensions used in MLP CTE computation including
    batch×sequence, hidden, and intermediate dimensions with appropriate subtiling.

    Args:
        mlp_params: MLP configuration parameters
        sharded_dim: Dimension being sharded across cores
        dim_shard: Shard information for the current dimension

    Returns:
        MLPCTETileInfo object with all tiling configurations

    Intended Usage:
        Called once per shard to initialize tiling strategy for the computation
    """
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

    return MLPCTETileInfo(
        bxs_dim_tile=bxs_dim_tile,
        layer_norm_hidden_dim_tile=layer_norm_hidden_dim_tile,
        xpose_hidden_dim_tile=xpose_hidden_dim_tile,
        src_proj_hidden_dim_tile=src_proj_hidden_dim_tile,
        src_proj_intermediate_dim_tile=src_proj_intermediate_dim_tile,
        xpose_intermediate_dim_tile=xpose_intermediate_dim_tile,
        down_proj_hidden_dim_tile=down_proj_hidden_dim_tile,
        down_proj_intermediate_dim_tile=down_proj_intermediate_dim_tile,
    )
