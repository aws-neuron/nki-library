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
MLP CTE Constants Module

Defines constants and configuration parameters for MLP CTE kernels including data types,
buffer counts, PSUM bank requirements, and sharding information.

"""

from dataclasses import dataclass
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
from nki.language import NKIObject

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import (
    NUM_HW_PSUM_BANKS,
)
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_down_projection_bias,
    mlpp_has_gate_projection,
    mlpp_has_gate_projection_bias,
    mlpp_has_layer_normalization,
    mlpp_has_normalization_bias,
    mlpp_has_normalization_weights,
    mlpp_has_rms_normalization,
    mlpp_has_up_projection_bias,
)
from .mlp_cte_sharding import DimShard, ShardedDim, is_sharded_dim_bxs
from .mlp_cte_tile_info import MLPCTETileInfo

#
# Public constants

BN_STATS_ELEMENTS_PER_TILE = 6
BN_AGGR_ELEMENTS_PER_TILE = 2

# Total available SBUF - DynamicDMAScratchLoc - EvalAccelReservedLoc - identity tensor size for transpose
MAX_AVAILABLE_SBUF_SIZE = 224 * 1024 - 16384 - 8 - 256


#
#
# Primary tuple that holds miscellaneous constants required by the kernel
#
@dataclass
class MLPCTEConstants(NKIObject):
    identity_tensor_sbuf: nl.ndarray
    # Compute data type used for matmuls, etc.
    compute_data_type: nki.dtype
    # Data type used for activations
    activation_data_type: nki.dtype
    # Data type used for normalization weights and biases
    norm_weights_bias_data_type: nki.dtype
    # Bias vector of zeros used for activation functions
    bxs_dim_subtile_zero_bias_vector_sbuf: nl.ndarray
    # Bias vector of the epsilon value used for activation functions
    epsilon_bias_vector_sbuf: nl.ndarray
    # Number of PSUM banks required for various matmuls
    required_src_xpose_psum_bank_count: int
    required_int_xpose_psum_bank_count: int
    required_src_proj_psum_bank_count: int
    required_down_proj_psum_bank_count: int
    # Constants for weights buffering to facilitate overlapped loads
    src_proj_weights_buffer_count: int
    down_proj_weights_buffer_count: int
    # True if PE transpose should be used
    use_pe_xpose_flag: bool
    # The data type to be used for writing to PSUM for transposes
    xpose_data_type: nki.dtype
    # The sharded dimension
    sharded_dim: ShardedDim
    # The size along the sharded dimension we are processing
    shard_size: Optional[int]
    # The global offset into the sharded dimension where our processing starts
    shard_offset: Optional[int]
    # Total number of programs
    total_programs: int

    def get_bxs_offset(self) -> int:
        return self.shard_offset if is_sharded_dim_bxs(self.sharded_dim) else 0

    def get_bxs_size(self, mlp_params: MLPParameters) -> int:
        return (
            self.shard_size if is_sharded_dim_bxs(self.sharded_dim) else mlp_params.batch_size * mlp_params.sequence_len
        )

    def get_intermediate_offset(self) -> int:
        return self.shard_offset if self.sharded_dim == ShardedDim.INTERMEDIATE else 0


def _get_compute_data_type(mlp_params: MLPParameters) -> nki.dtype:
    """Determine compute data type based on input tensor type.

    Selects appropriate compute data type for internal calculations.

    Args:
        mlp_params: MLP configuration parameters

    Returns:
        Compute data type for internal operations

    Intended Usage:
        Called internally to determine optimal data type for computations
    """
    io_type = mlp_params.hidden_tensor.dtype
    if io_type == nl.float32:
        compute_data_type = nl.bfloat16
    else:
        compute_data_type = io_type

    return compute_data_type


def build_mlp_cte_constants(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    sharded_dim: ShardedDim,
    total_programs: int,
    sbm: SbufManager,
    shard_idx: int,
    program_id: int,
    dim_shard: DimShard = None,
    allocator=None,
) -> MLPCTEConstants:
    """Build MLP CTE constants configuration.

    Creates and initializes all constants and configuration parameters needed for MLP CTE kernel execution.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        sharded_dim: Dimension being sharded across cores
        total_programs: Total number of programs in SPMD execution
        sbm: SBUF memory manager
        shard_idx: Index of current shard
        program_id: ID of current program
        dim_shard: Shard information for the current dimension
        allocator: Memory allocator function

    Returns:
        MLPCTEConstants object with all configuration parameters

    Intended Usage:
        Called once per shard to initialize kernel constants and configuration
    """
    identity_tensor = builtin.lang.shared_constant(builtin.lang.identity(nl.tile_size.pmax, nl.float32))
    identity_tensor_sbuf = allocator(
        (nl.tile_size.pmax, nl.tile_size.pmax),
        dtype=nl.bfloat16,
        # , buffer=alloc_info.identity_alloc.get_allocator()
    )
    nisa.dma_copy(dst=identity_tensor_sbuf, src=identity_tensor)

    # Empirical number of weights buffering to facilitate overlapped DMA loads with following matmul
    src_proj_weights_buffer_count = 4
    down_proj_weights_buffer_count = 8

    # Data types
    compute_data_type = _get_compute_data_type(mlp_params)
    activation_data_type = nl.float32
    norm_weights_bias_data_type = nl.float32

    alloc_heap = nl.ndarray if sbm == None else sbm.alloc_heap
    # We need a zero bias vector for activations to work around a runtime issue when no bias vector
    # is supplied to the activation method
    bias_vector_sbuf = alloc_heap(
        (tile_info.bxs_dim_tile.subtile_dim_info.tile_size, 1),
        activation_data_type,
        buffer=nl.sbuf,
        name=f"bias_vector_sbuf__shard{shard_idx}__prog{program_id}",
    )
    nisa.memset(bias_vector_sbuf, value=0.0)

    # We need an epsilon bias vector for certain math operations in the kernel
    epsilon_bias_vector_sbuf = alloc_heap(
        (nl.tile_size.pmax, 1),
        compute_data_type,
        buffer=nl.sbuf,
        name=f"epsilon_bias_vector_sbuf__shard{shard_idx}__prog{program_id}",
    )
    nisa.memset(
        epsilon_bias_vector_sbuf,
        value=mlp_params.eps,
    )

    # PSUM bank count requirements
    required_src_xpose_psum_bank_count = min(
        NUM_HW_PSUM_BANKS,
        tile_info.bxs_dim_tile.subtile_dim_info.tile_count * tile_info.xpose_hidden_dim_tile.tile_count,
    )
    required_int_xpose_psum_bank_count = min(
        NUM_HW_PSUM_BANKS,
        tile_info.bxs_dim_tile.subtile_dim_info.tile_count * tile_info.xpose_intermediate_dim_tile.tile_count,
    )
    required_src_proj_psum_bank_count = min(
        NUM_HW_PSUM_BANKS,
        tile_info.bxs_dim_tile.subtile_dim_info.tile_count * tile_info.src_proj_intermediate_dim_tile.tile_count,
    )
    # We want to use all the banks for down projection
    required_down_proj_psum_bank_count = NUM_HW_PSUM_BANKS
    # Set up PE transpose info
    use_pe_xpose_flag = True
    xpose_data_type = compute_data_type

    return MLPCTEConstants(
        identity_tensor_sbuf=identity_tensor_sbuf,
        compute_data_type=compute_data_type,
        activation_data_type=activation_data_type,
        norm_weights_bias_data_type=norm_weights_bias_data_type,
        bxs_dim_subtile_zero_bias_vector_sbuf=bias_vector_sbuf,
        epsilon_bias_vector_sbuf=epsilon_bias_vector_sbuf,
        required_src_xpose_psum_bank_count=required_src_xpose_psum_bank_count,
        required_int_xpose_psum_bank_count=required_int_xpose_psum_bank_count,
        required_src_proj_psum_bank_count=required_src_proj_psum_bank_count,
        required_down_proj_psum_bank_count=required_down_proj_psum_bank_count,
        src_proj_weights_buffer_count=src_proj_weights_buffer_count,
        down_proj_weights_buffer_count=down_proj_weights_buffer_count,
        use_pe_xpose_flag=use_pe_xpose_flag,
        xpose_data_type=xpose_data_type,
        sharded_dim=sharded_dim,
        shard_size=dim_shard.dim_size if sharded_dim != ShardedDim.NONE else None,
        shard_offset=dim_shard.dim_offset if sharded_dim != ShardedDim.NONE else None,
        total_programs=total_programs,
    )


def cleanup_mlp_cte_constants(sbm: SbufManager):
    """Clean up MLP CTE constants and free allocated memory.

    Frees memory allocated for bias vectors and other constants.

    Args:
        sbm: SBUF memory manager

    Returns:
        None

    Intended Usage:
        Called at end of shard execution to clean up allocated memory
    """
    if sbm != None:
        sbm.pop_heap()  # epsilon_bias_vector_sbuf
        sbm.pop_heap()  # bias_vector_sbuf
