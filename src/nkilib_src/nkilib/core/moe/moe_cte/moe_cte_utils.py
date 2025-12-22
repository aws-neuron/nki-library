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
Shared utility functions and configuration classes for MoE CTE kernels.

This module provides reusable components for Mixture of Experts (MoE) kernel implementations,
ensuring consistency and reducing code duplication across different sharding strategies.

Components:
    Configuration Classes:
        - SkipMode: Controls DMA skipping behavior for tokens and weights
        - BlockShardStrategy: Defines block distribution strategies (HI_LO, PING_PONG)
        - InputTensors: Container for all input tensor references
        - Configs: Comprehensive kernel execution configuration

    Memory Utilities:
        - stream_shuffle_broadcast: Broadcasts data across partition dimension
        - load_block_expert: Loads expert index for current block

    Computation Utilities:
        - compute_intermediate_states: Computes gated activation with expert affinity
        - calculate_expert_affinities: Computes expert routing weights

    Helper Functions:
        - div_ceil: Integer ceiling division
        - compatible_dtype: Returns compatible dtype for NC version

Usage:
    These utilities are shared across bwmm_shard_on_I.py and bwmm_shard_on_block.py
    to ensure consistent behavior and reduce maintenance overhead.
"""

import math
from enum import Enum
from typing import Any, List, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import sendrecv
from nki.isa.constants import oob_mode
from nki.language import NKIObject

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import reduce

TILE_SIZE = 128
PSUM_SIZE = 512
N_PSUM_BANKS = 8
DVE_CHANNELS_PER_BANK = 32
TOTAL_PSUM_SIZE = PSUM_SIZE * N_PSUM_BANKS
SB_QUADRANT_SIZE = 32


class SkipMode(NKIObject):
    """
    Controls DMA skipping behavior for memory optimization.

    Attributes:
        skip_token (bool): Skip DMA operations for out-of-bounds tokens (default: False)
        skip_weight (bool): Skip DMA operations for weight loading (default: False)

    Usage:
        skip_mode = SkipMode(skip_token=True, skip_weight=False)
    """

    skip_token: bool = False
    skip_weight: bool = False

    def __init__(self, skip_token: bool = False, skip_weight: bool = False):
        self.skip_token = skip_token
        self.skip_weight = skip_weight


class BlockShardStrategy(Enum):
    """
    Block distribution strategies for multi-core execution.

    Strategies:
        HI_LO: Assigns first half of blocks to shard 0, second half to shard 1
        PING_PONG: Alternates blocks between shards (0, 1, 0, 1, ...)
    """

    HI_LO = 0
    PING_PONG = 1


class InputTensors(NKIObject):
    """
    Container for all input tensor references.

    Groups all input tensors for MoE kernel execution to simplify function signatures
    and ensure consistent tensor passing across kernel implementations.

    Attributes:
        token_position_to_id: Token-to-block mapping tensor
        block_to_expert: Block-to-expert assignment tensor
        hidden_states: Input token embeddings
        gate_up_proj_weight: Gate and up projection weights
        gate_and_up_proj_bias: Optional bias for gate/up projections
        down_proj_bias: Optional bias for down projection
        down_proj_weight: Down projection weights
        expert_affinities_masked: Expert routing weights
        gate_up_proj_scale: Optional FP8 dequantization scales for gate/up
        down_proj_scale: Optional FP8 dequantization scales for down projection
    """

    token_position_to_id: Any
    block_to_expert: Any
    hidden_states: Any
    gate_up_proj_weight: Any
    gate_and_up_proj_bias: Any
    down_proj_bias: Any
    down_proj_weight: Any
    expert_affinities_masked: Any
    gate_up_proj_scale: Any
    down_proj_scale: Any


class Configs(NKIObject):
    """
    Comprehensive kernel execution configuration.

    Encapsulates all configuration parameters for MoE kernel execution including
    data types, optimization flags, quantization settings, and execution modes.

    Configuration Hierarchy:
        - InputTensors: Holds references to all input tensors
        - Configs: Holds execution parameters and flags (this class)
        - SkipMode: Controls DMA skipping behavior (subset of Configs)

    Attributes:
        scaling_mode: Expert affinity application mode (PRE_SCALE, POST_SCALE, PRE_SCALE_DELAYED)
        skip_dma: DMA skipping configuration
        compute_dtype: Data type for internal computations
        weight_dtype: Data type for weight tensors
        io_dtype: Data type for input/output tensors
        is_tensor_update_accumulating: Enable accumulation for TopK > 1
        use_dynamic_while: Use dynamic loop control for variable-length sequences
        linear_bias: Enable bias addition in projections
        activation_function: Activation function type (SiLU, GELU, etc.)
        is_quant: Enable FP8 quantization support
        fuse_gate_and_up_load: Fuse gate and up weight loading
        gate_clamp_upper_limit: Upper clamp limit for gate projections
        gate_clamp_lower_limit: Lower clamp limit for gate projections
        up_clamp_upper_limit: Upper clamp limit for up projections
        up_clamp_lower_limit: Lower clamp limit for up projections
        checkpoint_activation (bool): Enable activation checkpointing for gradient computation (default: False)
        expert_affinity_multiply_on_I (bool): Controls where expert affinity scaling is applied.
            - True: Apply affinity scaling on intermediate states (I) after activation
            - False: Apply affinity scaling on H (hidden states) after down projection (default)

    Usage:
        cfg = Configs(
            compute_dtype=nl.bfloat16,
            is_quant=True,
            linear_bias=True,
            activation_function=ActFnType.SiLU,
            scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_dma=SkipMode(skip_token=False, skip_weight=False),
            is_tensor_update_accumulating=True,
            use_dynamic_while=False,
            fuse_gate_and_up_load=True,
        )
    """

    scaling_mode: ExpertAffinityScaleMode
    skip_dma: SkipMode
    compute_dtype: Any
    weight_dtype: Any
    io_dtype: Any
    is_tensor_update_accumulating: bool
    use_dynamic_while: bool
    linear_bias: bool
    activation_function: ActFnType
    is_quant: bool
    fuse_gate_and_up_load: bool
    gate_clamp_upper_limit: Optional[float]
    gate_clamp_lower_limit: Optional[float]
    up_clamp_upper_limit: Optional[float]
    up_clamp_lower_limit: Optional[float]
    checkpoint_activation: bool = False
    expert_affinity_multiply_on_I: bool = False


def div_ceil(n, d):
    return (n + d - 1) // d


def compatible_dtype(compute_type):
    return compute_type if nisa.get_nc_version() >= nisa.nc_version.gen3 else nl.float32


def stream_shuffle_broadcast(src, dst):
    """
    Broadcasts the first partition of src onto the partition dim of dst.

    This is exactly the same as the one in neuronxcc.nki._pre_prod_kernels.stream_shuffle_broadcast.
    The reason we must put it here is because we must remove the jit decorator.
    All inputs and outputs to this function are assumed to be in sbuf.
    This requires 2D src and dst, and the final dim of src matching the final dim of dst.

    Args:
        src: 2D input tensor in SBUF.
        dst: 2D output tensor in SBUF.

    Returns:
        None: Broadcasts src to dst in-place.

    Pseudocode:
        dst_npar = dst.shape[0]
        dst_free = dst.shape[1]
        shuffle_mask = [0] * 32
        for bank_idx in range(ceil(dst_npar / 32)):
            cur_npar = min(32, dst_npar - bank_idx * 32)
            nc_stream_shuffle(src[0:1, 0:dst_free], dst[bank_idx*32:bank_idx*32+cur_npar, 0:dst_free], shuffle_mask)
    """
    dst_npar = dst.shape[0]
    dst_free = dst.shape[1]

    shuffle_mask = [0] * 32
    for bank_idx in range((dst_npar + 31) // 32):
        cur_npar = min(32, dst_npar - bank_idx * 32)

        nisa.nc_stream_shuffle(
            src=src[0:1, 0:dst_free],
            dst=dst[bank_idx * 32 : bank_idx * 32 + cur_npar, 0:dst_free],
            shuffle_mask=shuffle_mask,
        )


def load_block_expert(block_to_expert, block_idx):
    """
    Load expert ID assigned to the current block.

    Retrieves the expert index for the specified block from the block-to-expert
    mapping tensor, handling both static and dynamic block indices.

    Args:
        block_to_expert (nl.ndarray): Mapping tensor of shape [N, 1] where N is
            number of blocks, containing expert indices.
        block_idx (int or nl.ndarray): Block index to load, either static integer
            or dynamic tensor value.

    Returns:
        block_expert (nl.ndarray): Expert ID tensor of shape [1, 1] in SBUF.

    Notes:
        - Handles both static (int) and dynamic (tensor) block indices
        - Uses scalar_offset for dynamic indices via temporary tensor
        - Result stored in SBUF for efficient access
        - Required for expert-specific weight and bias loading

    Pseudocode:
        block_expert = allocate tensor [1, 1] in SBUF
        if block_idx is int:
            dma_copy block_to_expert[block_idx] to block_expert
        else:
            dma_copy block_to_expert[block_idx] to block_expert using scalar_offset
        return block_expert
    """
    block_expert = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)

    if isinstance(block_idx, int):
        nisa.dma_copy(dst=block_expert[0, 0], src=block_to_expert.ap(pattern=[[1, 1], [1, 1]], offset=block_idx))
    else:
        nisa.dma_copy(
            dst=block_expert[0, 0],
            src=block_to_expert.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=block_idx, indirect_dim=0),
        )
    return block_expert


def load_token_indices(token_position_to_id, block_idx, B, NUM_TILES):
    """
    Load and transpose token indices for the current block.

    Loads token position IDs for the current block and transposes them for
    efficient partition-dimension access.

    Args:
        token_position_to_id (nl.ndarray): Token position mapping of shape [N*B].
        block_idx (int): Current block index.
        B (int): Block size (number of tokens per block).
        NUM_TILES (int): Number of tiles (B // TILE_SIZE).

    Returns:
        result (nl.ndarray): Transposed token indices of shape [TILE_SIZE, NUM_TILES] in SBUF.

    Notes:
        - Uses dma_transpose for efficient layout transformation
        - Requires 4D tensor with [1,1] padding in middle dimensions
        - Result has tokens distributed across partition dimension
        - Enables efficient vector DGE for token gathering

    Pseudocode:
        result = allocate [TILE_SIZE, NUM_TILES] in SBUF
        offset = block_idx * B
        dma_transpose token_position_to_id[offset:offset+B] to result
        return result
    """
    result = nl.ndarray((TILE_SIZE, NUM_TILES), dtype=nl.int32, buffer=nl.sbuf)
    offset = block_idx * B
    nisa.dma_transpose(
        dst=result.ap(pattern=[[NUM_TILES, TILE_SIZE], [1, 1], [1, 1], [1, NUM_TILES]]),
        src=token_position_to_id.ap(pattern=[[TILE_SIZE, NUM_TILES], [1, 1], [1, 1], [1, TILE_SIZE]], offset=offset),
    )
    return result


def load_token_indices_dynamic_block(
    token_position_to_id, block_idx, B, NUM_TILES, skip_dma: SkipMode = SkipMode(False, False)
):
    """
    Load token indices for dynamic block with runtime block index.

    Loads token position IDs when block_idx is a dynamic tensor value rather
    than a static integer, using scalar_offset for indirect addressing.

    Args:
        token_position_to_id (nl.ndarray): Token position mapping tensor.
        block_idx (nl.ndarray): Dynamic block index tensor.
        B (int): Block size (number of tokens per block).
        NUM_TILES (int): Number of tiles (B // TILE_SIZE).
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        local_token_indices (nl.ndarray): Token indices of shape [TILE_SIZE, NUM_TILES] in SBUF.

    Notes:
        - Handles dynamic block_idx by copying to temp tensor for scalar_offset
        - Reshapes token_position_to_id to [total_size//B, B] for indexing
        - Validates total_size is divisible by B
        - Memsets to zero when skip_dma.skip_token is True
        - Transposes result for partition-dimension access

    Pseudocode:
        local_token_indices = allocate [TILE_SIZE, NUM_TILES] in SBUF
        total_size = product of token_position_to_id.shape
        validate total_size % B == 0
        reshaped = reshape token_position_to_id to [total_size//B, B]
        block_idx_copy = copy block_idx to SBUF
        for idx in range(NUM_TILES):
            if skip_dma.skip_token:
                memset local_token_indices[:, idx] to 0
            dma_copy reshaped[block_idx_copy, idx*TILE_SIZE:(idx+1)*TILE_SIZE] to local_token_indices[:, idx]
        return local_token_indices
    """
    local_token_indices = nl.ndarray((TILE_SIZE, NUM_TILES), dtype=token_position_to_id.dtype, buffer=nl.sbuf)
    total_size = reduce(op='mul', input=token_position_to_id.shape, initial_value=1)
    kernel_assert(total_size % B == 0, "token_position_to_id shape must be divisible by B")
    reshaped_token_position_to_id = token_position_to_id.reshape((total_size // B, B))

    block_idx_copy = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(block_idx_copy, block_idx)

    for idx in range(0, NUM_TILES):
        if skip_dma.skip_token:
            nisa.memset(local_token_indices[0:TILE_SIZE, idx], value=0)

        """
        Load contiguous TILE_SIZE elements from row block_idx.
        
        src pattern [[B, 1], [1, TILE_SIZE]] accesses elements [offset, offset+1, ..., offset+TILE_SIZE-1] within row.
        scalar_offset selects which row (block_idx * B added to base).
        """
        nisa.dma_copy(
            dst=local_token_indices.ap(pattern=[[NUM_TILES, TILE_SIZE], [1, 1]], offset=idx),
            src=reshaped_token_position_to_id.ap(
                pattern=[[1, TILE_SIZE], [1, 1]], offset=TILE_SIZE * idx, scalar_offset=block_idx_copy, indirect_dim=0
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
    return local_token_indices


def compute_intermediate_states(
    gate_and_up_proj_states,
    B,
    I_TP,
    dtype,
    activation_function=ActFnType.SiLU,
    expert_affinity_T_broadcasted=None,
    gup_scale=None,
    expert_affinity_multiply_on_I=False,
):
    """
    Compute intermediate states with activation and gating.

    Applies activation function to gate projection and performs element-wise multiplication
    with up projection to produce intermediate states. Optionally applies expert affinity
    scaling and FP8 dequantization.

    Memory Usage:
        - SBUF: intermediate_states_lst[i_tile] = (TILE_SIZE, B) per tile
        - SBUF: tmp_lst[i_tile] = (TILE_SIZE, B) per tile
        - Total SBUF: ~2 * GUP_N_TILES * TILE_SIZE * B * sizeof(dtype)
        - Example: For I_TP=2048, B=256, dtype=bf16: ~2 * 16 * 128 * 256 * 2 = 2MB

    Args:
        gate_and_up_proj_states (list): Nested list [2][N_PSUM_TILE][GUP_N_TILES] of projection results,
                                        each element is (TILE_SIZE, free_size) in PSUM
        B (int): Block size (number of tokens per block)
        I_TP (int): Intermediate size per shard
        dtype: Data type for intermediate computations
        activation_function (ActFnType): Activation function (SiLU, Swish, etc.)
        expert_affinity_T_broadcasted (nl.ndarray, optional): Broadcasted expert affinities (TILE_SIZE, B)
        gup_scale (list, optional): FP8 dequantization scales [GUP_N_TILES][2] for gate/up projections
        expert_affinity_multiply_on_I (bool): Controls where expert affinity scaling is applied.
            - True: Apply affinity scaling on intermediate states (I) after activation
            - False: Apply affinity scaling on H (hidden states) after down projection (default)

    Returns:
        intermediate_states_lst (list): List of tensors [GUP_N_TILES], each of shape (TILE_SIZE, B)

    Pseudocode:
        N_PSUM_TILE = ceil(B / PSUM_SIZE)
        GUP_N_TILES = ceil(I_TP / TILE_SIZE)
        intermediate_states_lst = allocate list of [GUP_N_TILES] tensors
        tmp_lst = allocate list of [GUP_N_TILES] tensors
        for i_tile_idx in range(GUP_N_TILES):
            for b_psum_idx in range(N_PSUM_TILE):
                if expert_affinity_T_broadcasted:
                    gate_and_up_proj_states *= expert_affinity_T_broadcasted
                if gup_scale:
                    gate_and_up_proj_states *= gup_scale
                if activation_function == SiLU:
                    tmp = silu(gate_proj)
                    intermediate = tmp * up_proj
                elif activation_function == Swish:
                    tmp = gelu_apprx_sigmoid(gate_proj)
                    intermediate = tmp * up_proj
        return intermediate_states_lst

    Example:
        # Compute intermediate states with SiLU activation
        intermediate = compute_intermediate_states(
            gate_and_up_proj_states=gup_states,
            B=256,
            I_TP=2048,
            dtype=nl.bfloat16,
            activation_function=ActFnType.SiLU,
            expert_affinity_T_broadcasted=affinity_tensor,
            gup_scale=None,
        )
    """
    N_PSUM_TILE = div_ceil(B, PSUM_SIZE)
    GUP_N_TILES = div_ceil(I_TP, TILE_SIZE)

    intermediate_states_lst = []
    tmp_lst = []
    for i_tile_idx in range(GUP_N_TILES):
        intermediate_states_lst.append(nl.ndarray((TILE_SIZE, B), dtype=dtype, buffer=nl.sbuf))
        tmp_lst.append(nl.ndarray((TILE_SIZE, B), dtype=dtype, buffer=nl.sbuf))

    free_size = gate_and_up_proj_states[0][0][0].shape[-1]

    for i_tile_idx in range(GUP_N_TILES):
        num_tile = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
        for b_psum_idx in range(N_PSUM_TILE):
            start_idx = b_psum_idx * PSUM_SIZE
            end_idx = start_idx + free_size

            if expert_affinity_T_broadcasted != None and not expert_affinity_multiply_on_I:
                for gate_or_up in nl.sequential_range(2):
                    if gup_scale != None:
                        nisa.scalar_tensor_tensor(
                            data=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                            op0=nl.multiply,
                            operand0=gup_scale[i_tile_idx][gate_or_up],
                            op1=nl.multiply,
                            operand1=expert_affinity_T_broadcasted[0:num_tile, 0:free_size],
                            dtype=dtype,
                            dst=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        )
                    else:
                        nisa.tensor_tensor(
                            data1=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                            data2=expert_affinity_T_broadcasted[0:num_tile, 0:free_size],
                            op=nl.multiply,
                            dst=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        )

            elif gup_scale != None:
                for gate_or_up in range(2):
                    nisa.tensor_scalar(
                        data=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        op0=nl.multiply,
                        operand0=gup_scale[i_tile_idx][gate_or_up],
                        dst=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                    )

            if activation_function == ActFnType.SiLU:
                nisa.activation(
                    op=nl.silu,
                    data=gate_and_up_proj_states[0][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                    scale=1.0,
                    dst=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                )
                nisa.tensor_tensor(
                    dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    data1=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    op=nl.multiply,
                    data2=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                )

            elif activation_function == ActFnType.Swish:
                nisa.activation(
                    op=nl.gelu_apprx_sigmoid,
                    data=gate_and_up_proj_states[0][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                    scale=1.0,
                    dst=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                )

                nisa.tensor_tensor(
                    dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    data1=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    data2=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                    op=nl.multiply,
                )

            if expert_affinity_T_broadcasted != None and expert_affinity_multiply_on_I:
                nisa.tensor_tensor(
                    dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    data1=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    op=nl.multiply,
                    data2=expert_affinity_T_broadcasted[0:num_tile, start_idx:end_idx],
                )

    return intermediate_states_lst


def calculate_expert_affinities(
    expert_affinities_masked,
    token_indices,
    block_expert,
    E,
    NUM_TILES,
    dtype,
    skip_dma: SkipMode = SkipMode(False, False),
    token_indices_offset=0,
):
    """
    Calculate expert affinities for the current block.

    Loads and computes expert affinity scores for tokens in the current block
    using indirect addressing based on token indices and expert ID.

    Args:
        expert_affinities_masked: Expert affinities tensor of shape [(T+1)*E, 1].
        token_indices: Token indices for current block of shape [TILE_SIZE, NUM_TILES].
        block_expert: Expert ID for current block of shape [1, 1].
        E (int): Number of experts.
        NUM_TILES (int): Number of tiles for the block when TILE_SIZE is used.
        dtype: Data type for affinity values.
        skip_dma (SkipMode): DMA skip configuration.
        token_indices_offset (int): Offset for block tiling, used when blocks == experts
            resulting in block sizes > 1024.

    Returns:
        expert_affinity_f32 (List[nl.ndarray]): List of expert affinity tensors in SBUF,
            one per tile, each of shape [TILE_SIZE, 1] in float32.

    Notes:
        - Uses pointer arithmetic: addr = token_indices * E + block_expert
        - Broadcasts block_expert to all partitions via stream_shuffle_broadcast
        - Performs indirect load from expert_affinities_masked
        - Skips DMA for invalid tokens when skip_dma.skip_token is True
        - Returns list of tensors for per-tile processing

    Pseudocode:
        v_expert = broadcast block_expert to [TILE_SIZE, 1]
        expert_affinity_f32 = allocate list of NUM_TILES tensors
        for n in range(NUM_TILES):
            addr = token_indices[:, n] * E
            addr_fin = addr + v_expert
            if skip_dma.skip_token:
                addr_fin = max(addr_fin, -1)
            expert_affinity_f32[n] = load expert_affinities_masked[addr_fin]
        return expert_affinity_f32
    """
    v_expert = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=block_expert, dst=v_expert)

    expert_affinity_f32 = []
    for n in range(NUM_TILES):
        expert_affinity_f32.append(nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf))

    for n in range(NUM_TILES):
        # Use pointer arithmetic to index into expert affinities
        # addr = token_indices * E
        addr = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=addr,
            data=token_indices[0:TILE_SIZE, token_indices_offset + n],
            op0=nl.multiply,
            operand0=E,
        )

        # Cast so that we can workaround TensorScalarAddr check
        v_expert_f32 = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(v_expert_f32, v_expert)

        # addr_fin = addr + v_expert_f32
        addr_fin = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=addr_fin,
            data1=addr,
            op=nl.add,
            data2=v_expert_f32,
        )

        # Handle DMA skipping cases
        if skip_dma.skip_token:
            nisa.tensor_scalar(
                dst=addr_fin,
                data=addr_fin,
                op0=nl.maximum,
                operand0=-1,
            )

        expert_affinity_dtype = nl.ndarray((TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
        if skip_dma.skip_token:
            nisa.memset(expert_affinity_dtype[0:TILE_SIZE, 0], value=0)

        # Create destination buffer in sbuf for DMA
        expert_affinity_loaded = nl.ndarray((TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
        # expert_affinities_masked shape: ((T+1) * E, 1)
        # Use vector AP with addr_fin as vector_offset
        if skip_dma.skip_token:
            nisa.memset(expert_affinity_loaded, value=0)

        nisa.dma_copy(
            dst=expert_affinity_loaded,
            src=expert_affinities_masked.ap(
                pattern=[[1, TILE_SIZE], [1, 1]], offset=0, vector_offset=addr_fin, indirect_dim=0
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )

        # Cast to float32 to be compatible with tensorscalarptr
        nisa.tensor_copy(expert_affinity_f32[n][0:TILE_SIZE, 0], expert_affinity_loaded[0:TILE_SIZE, 0])

    return expert_affinity_f32


def reduce_outputs(
    output: nl.ndarray,
    zeros: nl.ndarray,
    num_tiles: int,
    reduce_tile_size: int,
    offset: int,
    dim_hidden: int,
):
    """
    Synchronize across axis=0 in output by performing reduce and store.

    Performs element-wise reduction across the first dimension of a 3D output tensor
    by adding corresponding tiles from output[0] and output[1], storing results in
    output[0], and zeroing output[1].

    Args:
        output (nl.ndarray): Output tensor of shape [2, T, H] where T is sequence
            length and H is hidden dimension.
        zeros (nl.ndarray): Zero tensor of shape [reduce_tile_size, H] used for
            zeroing operations.
        num_tiles (int): Number of tiles to process in the reduction loop.
        reduce_tile_size (int): Size of each tile along the partition dimension.
        offset (int): Starting tile offset for read/write operations.
        dim_hidden (int): Hidden dimension size (H).

    Returns:
        None: Modifies output tensor in-place.

    Notes:
        - Output[0] contains the reduced sum after execution
        - Output[1] is zeroed after reduction
        - Uses DMA operations for efficient memory access

    Pseudocode:
        T = output.shape[1]
        for tile_idx in range(num_tiles):
            start_idx = (tile_idx + offset) * reduce_tile_size
            output[0, start_idx:start_idx+reduce_tile_size, :] = output[0, start_idx:start_idx+reduce_tile_size, :] + output[1, start_idx:start_idx+reduce_tile_size, :]
            output[1, start_idx:start_idx+reduce_tile_size, :] = zeros
    """
    T = output.shape[1]

    for tile_idx in range(num_tiles):
        start_idx = (tile_idx + offset) * reduce_tile_size

        # Reduce output[0] + output[1] and store directly to output[0]
        nisa.dma_compute(
            dst=output.ap(pattern=[[dim_hidden, reduce_tile_size], [1, dim_hidden]], offset=start_idx * dim_hidden),
            srcs=[
                output.ap(pattern=[[dim_hidden, reduce_tile_size], [1, dim_hidden]], offset=start_idx * dim_hidden),
                output.ap(
                    pattern=[[dim_hidden, reduce_tile_size], [1, dim_hidden]],
                    offset=T * dim_hidden + start_idx * dim_hidden,
                ),
            ],
            scales=[1.0, 1.0],
            reduce_op=nl.add,
        )

        # Zero out output[1]
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[dim_hidden, reduce_tile_size], [1, dim_hidden]],
                offset=T * dim_hidden + start_idx * dim_hidden,
            ),
            src=zeros.ap(pattern=[[dim_hidden, reduce_tile_size], [1, dim_hidden]], offset=0),
        )


def output_initialization(output, dims):
    """
    Zero initialize output buffer for accumulation mode.

    Initializes output tensor to zeros, required for topK > 1 scenarios where
    multiple expert contributions are accumulated per token.

    Args:
        output (nl.ndarray): Output tensor in HBM, either [T, H] or [num_shards, T, H].
        dims: Dimension configuration object containing T, H, shard_id, num_shards, and TILESIZE.

    Returns:
        None: Modifies output tensor in-place.

    Notes:
        - Required for is_tensor_update_accumulating mode
        - Handles both single-shard [T, H] and multi-shard [num_shards, T, H] layouts
        - Processes in tiles of dims.TILESIZE for memory efficiency
        - Uses DMA copy of zero buffer for initialization

    Pseudocode:
        T = dims.T
        H = dims.H
        for tile_idx in range(ceil(T / TILESIZE)):
            num_p = min(TILESIZE, T - tile_idx * TILESIZE)
            zeros = allocate [TILESIZE, H] in SBUF
            memset zeros to 0
            if dims.num_shards > 1:
                offset = shard_id * T * H + tile_idx * TILESIZE * H
                dma_copy zeros to output[offset:offset+num_p*H]
            else:
                dma_copy zeros to output[tile_idx*TILESIZE:tile_idx*TILESIZE+num_p, :]
    """
    T = dims.T
    H = dims.H

    for tile_idx in range(div_ceil(T, TILE_SIZE)):
        num_p = min(TILE_SIZE, T - tile_idx * TILE_SIZE)
        zeros = nl.ndarray((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros, value=0)

        if dims.num_shards > 1:
            nisa.dma_copy(
                dst=output.ap(pattern=[[H, num_p], [1, H]], offset=dims.shard_id * T * H + tile_idx * TILE_SIZE * H),
                src=zeros.ap(pattern=[[H, num_p], [1, H]], offset=0),
            )
        else:
            nisa.dma_copy(
                dst=output[tile_idx * TILE_SIZE : tile_idx * TILE_SIZE + num_p, 0:H],
                src=zeros.ap(pattern=[[H, num_p], [1, H]], offset=0),
            )
