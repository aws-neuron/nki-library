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

"""MLP CTE sharding strategies for distributing computation across multiple cores."""

from dataclasses import dataclass
from enum import Enum, auto

import nki
import nki.language as nl
from nki.language import NKIObject

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_program_sharding_info
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_projection_bias,
    override_inter_size,
    override_seq_len,
)


# Calculate the type of sharding we should do given the MLP parameters
# NOTE: Sharding on just batch * sequence length is supported through the code but is never activated.
#       But we leave the infrastructure in place in case we need it in the future.
# Returns a tuple (sharded_dim, shards) instead of ShardInfo to work around KLIR tracing limitations
# @throws AssertionError: If no valid tile size can be found that divides the bxs evenly, or
#                         if the bxs is not divisible by a power of 2 >= 256
def calculate_sharding(mlp_params: MLPParameters):
    """Calculate optimal sharding strategy for MLP computation.

    Determines whether to shard on intermediate dimension or batch×sequence dimension
    based on tensor sizes and hardware constraints.

    Args:
        mlp_params: MLP configuration parameters

    Returns:
        ShardInfo object containing sharding configuration

    Intended Usage:
        Called once per program to determine sharding strategy for the computation
    """
    _, num_shard_workers, program_id = get_program_sharding_info()
    # In order to shard on the intermediate dimension, there must be 2 total workers and the batch * sequence must be
    # less than or equal to 256.  We pick this because that means each core will have a batch sequence of 128 or less
    # to process.  This is right at the threshold of the size of the partition dimension.  These small numbers cause
    # inefficient use of compute because of that.
    # Additionally, we don't currently support any projection bias (up/down/gate) when sharding on I.  So just don't
    # shard on I if any of those are active.
    bxs = mlp_params.batch_size * mlp_params.sequence_len
    shard_on_inter = (num_shard_workers == 2) and (bxs <= 256) and not mlpp_has_projection_bias(mlp_params)
    # Put some heuristics on shard on I that are conservative. There are clearly cases where the performance of sharding
    # on S is better for small sizes of S.  Those cases are where both I and H are small.
    if mlp_params.hidden_size < 7168 or mlp_params.intermediate_size < 1024:
        shard_on_inter = False

    if shard_on_inter:
        sharded_dim = ShardedDim.INTERMEDIATE
        shards = _calculate_intermediate_sharding(mlp_params, program_id, num_shard_workers)
    else:
        sharded_dim = ShardedDim.BATCH_X_SEQUENCE_LENGTH
        shards = _calculate_bxs_sharding(mlp_params, program_id, num_shard_workers)

    return ShardInfo(sharded_dim, shards)


def _get_bxs_shard_size(
    bxs: int,
    num_shards: int,
    default_tile_size: int = 512,
    default_tiling_factor: int = 16,
) -> int:
    """
    This function calculates an appropriate shard size based on the batch * sequence length (bxs)
    against number of shards.
    For bxs size <= 16K, it uses a default tile size. For larger bxs, it attempts to find
    a tile size that enables efficient fusion with the cc_pipeline and lnc degrees.

    @param bxs: The dimension of batch * sequence length to be tiled
    @param num_shards: Number of shards for parallel processing
    @param default_tile_size: Default tile size to use for bxs <= 16K (default: 512)
    @param default_tiling_factor: Default tiling factor when num_shards <= 2 (default: 16)

    @return: The calculated bxs tile size
    """

    # Choose bxs tile size for CTE cases
    bxs_tile_size = None
    tile_size_candidates = [
        32 * 1024,
        16 * 1024,
        8 * 1024,
        4 * 1024,
        2 * 1024,
        1024,
        512,
        256,
    ]
    if num_shards > 2:
        # If num_shards > 2 then the user is passing additional tiling axis (e.g., cc_pipeline) in
        # the SPMD grid, and we can just tile accordingly, i.e., make num_tile == num_shards.
        return bxs // num_shards

    if bxs <= 16 * 1024:
        bxs_tile_size = default_tile_size
        if bxs <= (num_shards * bxs_tile_size):
            bxs_tile_size = bxs // num_shards
    else:
        ignore_divisibility = bxs % tile_size_candidates[-1] != 0
        for c_idx in range(len(tile_size_candidates)):
            candidate = tile_size_candidates[c_idx]
            if (bxs / candidate >= default_tiling_factor) and (ignore_divisibility or bxs % candidate == 0):
                bxs_tile_size = candidate
                break

    return bxs_tile_size


#
# Program helpers


class ShardedDim(Enum):
    NONE = auto()
    SEQUENCE_LENGTH = auto()
    INTERMEDIATE = auto()
    BATCH_X_SEQUENCE_LENGTH = auto()


def is_sharded_dim_bxs_or_s(sharded_dim: ShardedDim) -> bool:
    """Check if sharded dimension is batch×sequence or sequence length."""
    return sharded_dim == ShardedDim.SEQUENCE_LENGTH or sharded_dim == ShardedDim.BATCH_X_SEQUENCE_LENGTH


def is_sharded_dim_bxs(sharded_dim: ShardedDim) -> bool:
    """Check if sharded dimension is batch×sequence length."""
    return sharded_dim == ShardedDim.BATCH_X_SEQUENCE_LENGTH


"""
    An instance of DimShard represents a shard on a specific dimension.  The members specific the shard region
    along with the associated kernels parameters.  shard_mlp_params is typically provided in this structure
    such that the sharded dimension size is specified as the size of the shard.  We do this with shard_mlp_params
    so that the most of the rest of the code can be written in a way that is agnostic to shard size, original tensor
    size, and whether or not sharding is even used.

    For example, assume we are sharding on the S dimension, the S dimension size is 1024, and our shard size is 512.
    Two instances of DimShard would represent this.  Instance #1 would have a dim_offset of 0, a dim_size of 512,
    and the S dimension size in shard_mlp_params would be set to 512.  Instance #2 would have a dim_offset of 512 (i.e.
    offset into the tensor by 512), a dim_size of 512, and the S dimension size in shard_mlp_params would be set to 512.
"""


@dataclass
class DimShard(NKIObject):
    dim_offset: int  # Offset into the dimension being sharded on
    dim_size: int  # The size of the shard
    shard_mlp_params: MLPParameters  # Kernel params with an updated sharded dimension based on the shard size


@dataclass
class ShardInfo(NKIObject):
    sharded_dim: ShardedDim  # The dimension being sharded
    shards: list  # The shards that the worker will process - list[DimShard]


#
# Based on the kernel parameters and the current SPMD program, build and return a list of shards
# on the intermediate dimension that the kernel needs to process.
def _calculate_intermediate_sharding(
    mlp_params: MLPParameters,
    program_id: int,
    num_shard_workers: int,
) -> list[DimShard]:
    """Calculate sharding configuration for intermediate dimension."""
    inter_shard_size = mlp_params.intermediate_size // num_shard_workers
    inter_offset = program_id * inter_shard_size
    shard_mlp_params = override_inter_size(mlp_params, inter_shard_size)

    return [
        DimShard(
            dim_offset=inter_offset,
            dim_size=inter_shard_size,
            shard_mlp_params=shard_mlp_params,
        )
    ]


#
# Based on the kernel parameters and the current SPMD program, build and return a list of shards
# on the combined sequence length and batch dimensions that the kernel needs to process.
def _calculate_bxs_sharding(
    mlp_params: MLPParameters,
    program_id: int,
    num_shard_workers: int,
) -> list[DimShard]:
    """Calculate sharding configuration for batch×sequence dimension."""
    # Get the tile size based on the sequence length but then we have to multiply by the batch size because
    # we are sharding across batch size * sequence length (bxs)
    bxs = mlp_params.batch_size * mlp_params.sequence_len
    kernel_assert(
        bxs % num_shard_workers == 0,
        f"Batch times sequence length {bxs} must be a multiple of the number of shards {num_shard_workers}",
    )
    bxs_shard_size = _get_bxs_shard_size(bxs, num_shard_workers)
    bxs_size_per_worker = bxs // num_shard_workers
    bxs_tiles = TiledRange(bxs_size_per_worker, bxs_shard_size)

    shard_list = []
    for bxs_tile in bxs_tiles:
        shard_list.append(DimShard(bxs_size_per_worker * program_id + bxs_tile.start_offset, bxs_tile.size, mlp_params))
    return shard_list
