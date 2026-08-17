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

from enum import Enum
from typing import Any, Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode
from nki.language import NKIObject

from ...utils.allocator import SbufManager
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import _sbm_alloc, reduce

TILE_SIZE = 128
PSUM_SIZE = 512
N_PSUM_BANKS = 8
DVE_CHANNELS_PER_BANK = 32
TOTAL_PSUM_SIZE = PSUM_SIZE * N_PSUM_BANKS
SBUF_QUADRANT_SIZE = 32


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


class WeightQuantMode(Enum):
    """Weight quantization mode for FP8 MoE kernels.

    Scale shapes per mode (gate_up / down):
        NONE:        No scales
        PER_CHANNEL: [E, 1, 2*I_TP] / [E, 1, H]
        PER_TENSOR:  [E, 2, 1] / [E, 1]
        BLOCK_256:   [E, H//256, 2, I_TP//256, 128] / [E, I_TP//256, H//256, 128]
    """

    NONE = 0
    PER_CHANNEL = 1
    PER_TENSOR = 2
    BLOCK_256 = 3


class ActivationQuantMode(Enum):
    """Activation (hidden/intermediate) quantization mode.

    Scale shapes per mode (gate_up_hidden / down_hidden):
        NONE:       No scales
        PER_TOKEN:  [T+1, 1] / [T+1, 1]
        PER_TENSOR: [E, 2, 1] / [E, 1]
    """

    NONE = 0
    PER_TOKEN = 1
    PER_TENSOR = 2


class ScaleFormat(Enum):
    """How scales are stored in HBM.

    RAW:           Scalar values as-is
    PRE_BROADCAST: Scalars pre-broadcasted to (TILE_SIZE, ...) on partition dim
    """

    RAW = 0
    PRE_BROADCAST = 1


class QuantConfig(NKIObject):
    """Quantization configuration for MoE shard-on-I kernel.

    Attributes:
        weight_mode: Weight quantization granularity
        activation_mode: Activation quantization granularity
        scale_format: How scales are stored in HBM
    """

    weight_mode: WeightQuantMode = WeightQuantMode.NONE
    activation_mode: ActivationQuantMode = ActivationQuantMode.NONE
    scale_format: ScaleFormat = ScaleFormat.RAW

    @property
    def is_quant(self):
        return self.weight_mode != WeightQuantMode.NONE

    @property
    def is_block_quant(self):
        return self.weight_mode == WeightQuantMode.BLOCK_256

    @property
    def is_per_tensor(self):
        return self.weight_mode == WeightQuantMode.PER_TENSOR

    @property
    def has_activation_scale(self):
        return self.activation_mode != ActivationQuantMode.NONE


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
    gate_up_hidden_scale: Any = None
    down_hidden_scale: Any = None
    # Pre-reshaped tensors (populated at kernel entry point where direct params are available)
    gate_up_proj_scale_flat: Any = None  # reshaped for block quant DMA
    gate_up_proj_scale_E2I: Any = None  # reshaped for per-channel DMA


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
    skip_gate_proj: bool = False
    # Accumulator/checkpoint precision; None = io_dtype (default). Set fp32 to reduce bf16 error.
    accumulation_dtype: Any = None
    # Flat quant fields. Kept flat (not on a nested QuantConfig) because the NKI
    # parser frontend cannot read attributes off a nested NKIObject inside a kernel.
    quant_activation_mode: ActivationQuantMode = ActivationQuantMode.NONE
    quant_is_per_tensor: bool = False
    is_block_quant: bool = False


def div_ceil(n, d):
    return (n + d - 1) // d


def compatible_dtype(compute_type):
    return compute_type if nisa.get_nc_version() >= nisa.nc_version.gen3 else nl.float32


def stream_shuffle_broadcast(src, dst):
    """
    Broadcasts the first partition of src onto the partition dim of dst.

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


def load_block_expert(block_to_expert, block_idx, sbm: Optional[SbufManager] = None, name="block_expert"):
    """
    Load expert ID assigned to the current block.

    Retrieves the expert index for the specified block from the block-to-expert
    mapping tensor, handling both static and dynamic block indices.

    Args:
        block_to_expert (nl.NkiTensor): Mapping tensor of shape [N, 1] where N is
            number of blocks, containing expert indices.
        block_idx (int or nl.NkiTensor): Block index to load, either static integer
            or dynamic tensor value.
        sbm (SbufManager): Optional SbufManager for SBUF allocation.
        name (str): Name for the allocated tensor (must be unique within scope).

    Returns:
        block_expert (nl.NkiTensor): Expert ID tensor of shape [1, 1] in SBUF.

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
    block_expert = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name=name if sbm else None, align=SBUF_QUADRANT_SIZE)

    if isinstance(block_idx, int):
        nisa.dma_copy(dst=block_expert[0, 0], src=block_to_expert.ap(pattern=[[1, 1], [1, 1]], offset=block_idx))
    else:
        nisa.dma_copy(
            dst=block_expert[0, 0],
            src=block_to_expert.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=block_idx, indirect_dim=0),
        )
    return block_expert


def load_token_indices(token_position_to_id, block_idx, B, NUM_TILES, sbm=None):
    """
    Load and transpose token indices for the current block.

    Loads token position IDs for the current block and transposes them for
    efficient partition-dimension access.

    Args:
        token_position_to_id (nl.NkiTensor): Token position mapping of shape [N*B].
        block_idx (int): Current block index.
        B (int): Block size (number of tokens per block).
        NUM_TILES (int): Number of tiles (B // TILE_SIZE).
        sbm (SbufManager): Optional SbufManager for SBUF allocation.

    Returns:
        result (nl.NkiTensor): Transposed token indices of shape [TILE_SIZE, NUM_TILES] in SBUF.

    Notes:
        - Runs on HWDGE: static offset, matching src/dst dtype, no runtime index compute
        - Result has tokens distributed across partition dimension
        - Enables efficient vector DGE for token gathering

    Pseudocode:
        result = allocate [TILE_SIZE, NUM_TILES] in SBUF
        offset = block_idx * B
        dma_copy transpose of token_position_to_id[offset:offset+B] (NUM_TILES, TILE_SIZE)
            into result (TILE_SIZE, NUM_TILES) on HWDGE
        return result
    """
    result = _sbm_alloc(
        sbm,
        (TILE_SIZE, NUM_TILES),
        dtype=nl.int32,
        name=f"token_indices_{block_idx}" if sbm else None,
        align=SBUF_QUADRANT_SIZE,
    )
    offset = block_idx * B
    """
    Transpose token_position_to_id[offset:offset+B] (NUM_TILES, TILE_SIZE) -> result
    (TILE_SIZE, NUM_TILES) with a strided dma_copy on HWDGE. 
        result[p, t] = token_position_to_id[offset + t*TILE_SIZE + p]
        - src dim0 [1, TILE_SIZE]: TILE_SIZE contiguous elements -> partition dim
        - src dim1 [TILE_SIZE, NUM_TILES]: NUM_TILES tiles, stride TILE_SIZE -> free dim
    """
    nisa.dma_copy(
        dst=result.ap(pattern=[[NUM_TILES, TILE_SIZE], [1, NUM_TILES]]),
        src=token_position_to_id.ap(pattern=[[1, TILE_SIZE], [TILE_SIZE, NUM_TILES]], offset=offset),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return result


def load_token_indices_dynamic_block(
    token_position_to_id, block_idx, B, NUM_TILES, skip_dma: SkipMode = SkipMode(False, False), sbm=None
):
    """
    Load token indices for dynamic block with runtime block index.

    Loads token position IDs when block_idx is a dynamic tensor value rather
    than a static integer, using scalar_offset for indirect addressing.

    Args:
        token_position_to_id (nl.NkiTensor): Token position mapping tensor.
        block_idx (nl.NkiTensor): Dynamic block index tensor.
        B (int): Block size (number of tokens per block).
        NUM_TILES (int): Number of tiles (B // TILE_SIZE).
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        local_token_indices (nl.NkiTensor): Token indices of shape [TILE_SIZE, NUM_TILES] in SBUF.

    Notes:
        - Handles dynamic block_idx by copying to temp tensor for scalar_offset
        - Reshapes token_position_to_id to [total_size//B, B] for indexing
        - Validates total_size is divisible by B
        - Memsets to zero when skip_dma.skip_token == True
        - Transposes result for partition-dimension access

    Pseudocode:
        local_token_indices = allocate [TILE_SIZE, NUM_TILES] in SBUF
        total_size = product of token_position_to_id.shape
        validate total_size % B == 0
        reshaped = reshape token_position_to_id to [total_size//B, B]
        if skip_dma.skip_token:
            memset local_token_indices to 0
        dma_copy transpose of reshaped[block_idx] (NUM_TILES, TILE_SIZE) into
            local_token_indices (TILE_SIZE, NUM_TILES) in a single DMA
        return local_token_indices
    """
    local_token_indices = _sbm_alloc(
        sbm,
        (TILE_SIZE, NUM_TILES),
        dtype=token_position_to_id.dtype,
        name="local_token_indices" if sbm else None,
        align=SBUF_QUADRANT_SIZE,
    )  # (128, n_B128_tiles)
    total_size = reduce(op='mul', input=token_position_to_id.shape, initial_value=1)  # Blocks * Block_Size
    kernel_assert(total_size % B == 0, "token_position_to_id shape must be divisible by B")
    reshaped_token_position_to_id = token_position_to_id.reshape((total_size // B, B))  # (Blocks, Block_Size)

    if skip_dma.skip_token:
        nisa.memset(local_token_indices[0:TILE_SIZE, 0:NUM_TILES], value=0)

    """
    Transpose all B = TILE_SIZE * NUM_TILES elements of row block_idx in a single DMA.

    This is the transpose reshaped_token_position_to_id[block_idx] (NUM_TILES, TILE_SIZE)
    -> local_token_indices (TILE_SIZE, NUM_TILES)
       Mapping is identical: dst[p, t] = reshaped[block_idx, t*TILE_SIZE + p].
        - src dim0 [1, TILE_SIZE]: TILE_SIZE contiguous elements -> partition dim
        - src dim1 [TILE_SIZE, NUM_TILES]: NUM_TILES tiles, stride TILE_SIZE
        - scalar_offset with indirect_dim=0: adds block_idx * B
    """
    nisa.dma_copy(
        dst=local_token_indices.ap(pattern=[[NUM_TILES, TILE_SIZE], [1, NUM_TILES]]),
        src=reshaped_token_position_to_id.ap(
            pattern=[[1, TILE_SIZE], [TILE_SIZE, NUM_TILES]], offset=0, scalar_offset=block_idx, indirect_dim=0
        ),
        oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        dge_mode=nisa.dge_mode.hwdge,
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
    sbm: Optional[SbufManager] = None,
    intermediate_states_lst: Optional[list] = None,
    skip_gate_proj: bool = False,
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
        expert_affinity_T_broadcasted (nl.NkiTensor, optional): Broadcasted expert affinities (TILE_SIZE, B)
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

    if intermediate_states_lst is None:
        intermediate_states_lst = []
        for i_tile_idx in range(GUP_N_TILES):
            if sbm is not None:
                intermediate_states_lst.append(
                    sbm.alloc_stack((TILE_SIZE, B), dtype=dtype, name=f"inter_state_i{i_tile_idx}")
                )
            else:
                intermediate_states_lst.append(nl.ndarray((TILE_SIZE, B), dtype=dtype, buffer=nl.sbuf))

    tmp_lst = []
    for i_tile_idx in range(GUP_N_TILES):
        if sbm is not None:
            tmp_lst.append(sbm.alloc_stack((TILE_SIZE, B), dtype=dtype, name=f"inter_tmp_i{i_tile_idx}"))
        else:
            tmp_lst.append(nl.ndarray((TILE_SIZE, B), dtype=dtype, buffer=nl.sbuf))

    free_size = gate_and_up_proj_states[0][0][0].shape[-1]

    for i_tile_idx in range(GUP_N_TILES):
        num_tile = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
        for b_psum_idx in range(N_PSUM_TILE):
            start_idx = b_psum_idx * PSUM_SIZE
            end_idx = start_idx + free_size

            if expert_affinity_T_broadcasted != None and not expert_affinity_multiply_on_I:
                for gate_or_up in range(2):
                    if skip_gate_proj and gate_or_up == 0:
                        continue
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
                    if skip_gate_proj and gate_or_up == 0:
                        continue
                    nisa.tensor_scalar(
                        data=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        op0=nl.multiply,
                        operand0=gup_scale[i_tile_idx][gate_or_up],
                        dst=gate_and_up_proj_states[gate_or_up][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                    )

            if skip_gate_proj:
                if activation_function == ActFnType.SquaredReLU:
                    nisa.activation(
                        op=nl.relu,
                        data=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        scale=1.0,
                        dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    )
                    nisa.activation(
                        op=nl.square,
                        data=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                        dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    )
                elif activation_function == ActFnType.SiLU:
                    nisa.activation(
                        op=nl.silu,
                        data=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        scale=1.0,
                        dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    )
                elif activation_function == ActFnType.Swish:
                    nisa.activation(
                        op=nl.gelu_apprx_sigmoid,
                        data=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                        scale=1.0,
                        dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    )
            elif activation_function == ActFnType.SquaredReLU:
                nisa.activation(
                    op=nl.relu,
                    data=gate_and_up_proj_states[0][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                    scale=1.0,
                    dst=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                )
                nisa.activation(
                    op=nl.square,
                    data=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    dst=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                )
                nisa.tensor_tensor(
                    dst=intermediate_states_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    data1=tmp_lst[i_tile_idx][0:num_tile, start_idx:end_idx],
                    op=nl.multiply,
                    data2=gate_and_up_proj_states[1][b_psum_idx][i_tile_idx][0:num_tile, 0:free_size],
                )

            elif activation_function == ActFnType.SiLU:
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
    sbm: Optional[SbufManager] = None,
    cast_to_f32=False,
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
        expert_affinity_f32 (List[nl.NkiTensor]): List of expert affinity tensors in SBUF,
            one per tile, each of shape [TILE_SIZE, 1] in float32.

    Notes:
        - Uses pointer arithmetic: addr = token_indices * E + block_expert
        - Broadcasts block_expert to all partitions via stream_shuffle_broadcast
        - Performs indirect load from expert_affinities_masked
        - Skips DMA for invalid tokens when skip_dma.skip_token == True
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
    v_expert = _sbm_alloc(
        sbm, (TILE_SIZE, 1), dtype=nl.int32, name="aff_v_expert" if sbm else None, align=SBUF_QUADRANT_SIZE
    )
    stream_shuffle_broadcast(block_expert, v_expert)
    expert_affinity_f32_lst = []

    # Vectorized address computation for all tiles at once
    # addr_all[i, j] = token_indices[i, offset + j] * E
    addr_all = _sbm_alloc(
        sbm, (TILE_SIZE, NUM_TILES), dtype=nl.int32, name="aff_addr_all" if sbm else None, align=SBUF_QUADRANT_SIZE
    )
    nisa.tensor_scalar(
        dst=addr_all,
        op0=nl.multiply,
        data=token_indices[0:TILE_SIZE, token_indices_offset : token_indices_offset + NUM_TILES],
        operand0=E,
    )

    # addr_fin_all[i, j] = addr_all[i, j] + v_expert[i, 0] (v_expert broadcasts across free dim)
    addr_fin_all = _sbm_alloc(
        sbm, (TILE_SIZE, NUM_TILES), dtype=nl.int32, name="aff_addr_fin_all" if sbm else None, align=SBUF_QUADRANT_SIZE
    )
    nisa.tensor_tensor(
        dst=addr_fin_all, data1=addr_all, op=nl.add, data2=v_expert.ap(pattern=[[1, TILE_SIZE], [0, NUM_TILES]])
    )

    if skip_dma.skip_token:
        nisa.tensor_scalar(dst=addr_fin_all, data=addr_fin_all, op0=nl.maximum, operand0=-1)

    """
    Load expert affinities for each tile.

    Cannot vectorize DMA loads because indirect addressing requires per-tile
    [TILE_SIZE, 1] address tensors — each tile has different row indices per partition lane.
    """
    expert_affinity_f32_lst = []
    num_cols = expert_affinities_masked.shape[1]
    output_dtype = nl.float32 if cast_to_f32 else dtype
    expert_affinity_buf = _sbm_alloc(
        sbm,
        (TILE_SIZE, NUM_TILES),
        dtype=output_dtype,
        name="aff_expert_buf" if sbm else None,
        align=SBUF_QUADRANT_SIZE,
    )
    if skip_dma.skip_token:
        nisa.memset(value=0, dst=expert_affinity_buf)
    for tile_idx in range(NUM_TILES):
        nisa.dma_copy(
            dst=expert_affinity_buf[0:TILE_SIZE, tile_idx : tile_idx + 1],
            src=expert_affinities_masked.ap(
                pattern=[[num_cols, TILE_SIZE], [1, 1]],
                offset=0,
                vector_offset=addr_fin_all.ap(pattern=[[NUM_TILES, TILE_SIZE], [1, 1]], offset=tile_idx),
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
        expert_affinity_f32_lst.append(expert_affinity_buf[0:TILE_SIZE, tile_idx : tile_idx + 1])

    return expert_affinity_f32_lst


def reduce_outputs(
    output: nl.NkiTensor,
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
        output (nl.NkiTensor): Output tensor of shape [2, T, H] where T is sequence
            length and H is hidden dimension.
        zeros (nl.NkiTensor): Zero tensor of shape [reduce_tile_size, H] used for
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


def output_initialization(output, dims, sbm=None, zeros=None):
    """
    Zero initialize output buffer for accumulation mode.

    Args:
        output (nl.NkiTensor): Output tensor in HBM, either [T, H] or [num_shards, T, H].
        dims: Dimension configuration object containing T, H, shard_id, num_shards, and TILESIZE.
        sbm: Optional SBUF manager.
        zeros: Optional pre-allocated zero buffer of shape [TILE_SIZE, H]. If None, allocates on stack.
    """
    T = dims.T
    H = dims.H

    if zeros == None:
        if H % 2 == 0 or H % 4 == 0:
            zeros = _sbm_alloc(sbm, (TILE_SIZE, H), dtype=nl.bfloat16, name=f"init_zeros", align=SBUF_QUADRANT_SIZE)
            zeros_fp32 = zeros.view(nl.float32)
            nisa.memset(zeros_fp32, value=0.0)
        else:
            zeros = _sbm_alloc(sbm, (TILE_SIZE, H), dtype=nl.bfloat16, name=f"init_zeros", align=SBUF_QUADRANT_SIZE)
            nisa.memset(zeros, value=0.0)

    for tile_idx in nl.affine_range(div_ceil(T, TILE_SIZE)):
        if sbm != None:
            sbm.open_scope()
        num_p = min(TILE_SIZE, T - tile_idx * TILE_SIZE)

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
        if sbm != None:
            sbm.close_scope()
