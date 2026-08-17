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

"""This file contains high-performance MoE kernels implementing blockwise matrix multiplication with sharding on the intermediate dimension."""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import sendrecv
from nki.isa.constants import oob_mode
from nki.language import NKIObject

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from .moe_cte_utils import (
    DVE_CHANNELS_PER_BANK,
    PSUM_SIZE,
    TILE_SIZE,
    ActivationQuantMode,
    Configs,
    InputTensors,
    SkipMode,
    calculate_expert_affinities,
    compatible_dtype,
    compute_intermediate_states,
    div_ceil,
    load_block_expert,
    stream_shuffle_broadcast,
)

SB_QUADRANT_SIZE = 32
MAX_BLOCK_TILE_SIZE = 1024
FUSE_GATE_WEIGHT_SIZE = 4096 * 1024
P_MAX = nl.tile_size.pmax
F_MAX = nl.tile_size.psum_fmax
# Block quant tile size: scales are organized as 256x256 blocks along (H, I_TP).
BLOCK_QUANT_SIZE = 256


class OutputTensors(NKIObject):
    """Container for kernel output tensors."""

    output: Any
    gate_up_activations_T: Any
    down_activations: Any


class DebugTensors(NKIObject):
    """Container for debug tensors."""

    hidden_states: Any


class DimensionSizes(NKIObject):
    """Container for tensor dimension sizes and derived tiling parameters."""

    B: int
    H: int
    T: int
    E: int
    N: int
    I_TP: int
    I_TP_sharded: int
    GUP_N_TILES: int

    def derive_all_dims(self):
        self.NUM_SHARDS = nl.num_programs(axes=0)
        self.NUM_B_TILES = self.B // TILE_SIZE

        self.NUM_B_BLOCK_TILES = self.B // MAX_BLOCK_TILE_SIZE
        self.NUM_TILES_IN_B_BLOCK_TILE = MAX_BLOCK_TILE_SIZE // TILE_SIZE

        self.NUM_B_TILES_SHARDED = self.NUM_B_TILES // self.NUM_SHARDS
        self.NUM_STATIC_BLOCKS = (
            (self.N - self.E) if (self.N - self.E) % self.NUM_SHARDS == 0 else (self.N - self.E - 1)
        )
        self.NUM_DYNAMIC_BLOCKS = self.N - self.NUM_STATIC_BLOCKS
        self.I_TP_sharded = self.I_TP // self.NUM_SHARDS
        self.GUP_N_TILES = div_ceil(self.I_TP_sharded, TILE_SIZE)


@nki.jit
def blockwise_mm_baseline_shard_intermediate(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    block_size,
    token_position_to_id,
    block_to_expert,
    gate_and_up_proj_bias=None,
    down_proj_bias=None,
    # quantize scales. because we are using strategy 5,
    # gate_up_proj_scale shape: [E, 1, 2 * I_TP],
    # down_proj_scale shape: [E, 1, H]
    gate_up_proj_scale=None,
    down_proj_scale=None,
    gate_up_hidden_scale=None,
    down_hidden_scale=None,
    # Quant mode flags. Flat booleans (not a nested QuantConfig) because the NKI
    # parser frontend cannot read attributes off a nested NKIObject inside a kernel.
    is_block_quant: bool = False,
    is_per_tensor: bool = False,
    # Meta parameters
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.PRE_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    checkpoint_activation=False,
    expert_affinity_multiply_on_I=False,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
):
    """
    Blockwise matrix multiplication kernel for Mixture of Experts (MoE) with intermediate dimension sharding.

    Implements MoE layer at block granularity as an alternative to token dropping. Tokens are pre-assigned
    to blocks via token_position_to_id. Shards gate/up projection over I_TP dimension and block accumulation
    over B dimension. Processes all blocks sequentially without distinguishing padded vs non-padded blocks.

    Intended Usage:
        - Block size B: 256-1024 tokens (must be multiple of 256)
        - Total tokens T: Up to 32K tokens per call
        - Hidden dimension H: 512-8192 (optimal: 2048-4096)
        - Intermediate dimension I_TP: 2048-16384 (optimal: 8192)
        - Number of experts E: 8-64 (optimal: 8-16)
        - Use this variant when all blocks are active (no padding) or padding is minimal
        - For sequences with significant padding, use hybrid variant instead

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens (after linearizing across the batch dimension)
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I_TP: Intermediate size / tp degree

    Args:
        hidden_states (nl.tensor): [T+1, H], Input hidden states on HBM. T+1 because padding token position is set to T.
        expert_affinities_masked (nl.tensor): [(T+1) * E, 1], Expert affinities for each token on HBM.
        gate_up_proj_weight (nl.tensor): [E, H, 2, I_TP], Concatenated gate and up projection weights on HBM.
        down_proj_weight (nl.tensor): [E, I_TP, H], Down projection weights on HBM.
        block_size (int): Number of tokens per block.
        token_position_to_id (nl.tensor): [N * B], Block index of corresponding tokens on HBM. Includes padding tokens (N * B >= T). Padding token id is set to T.
        block_to_expert (nl.tensor): [N, 1], Expert indices of corresponding blocks on HBM.
        gate_and_up_proj_bias (nl.tensor, optional): [E, 2, I_TP], Gate and up projection bias. For Swiglu, up_bias = up_bias + 1.
        down_proj_bias (nl.tensor, optional): [E, H], Down projection bias.
        gate_up_proj_scale (nl.tensor, optional): [E, 1, 2 * I_TP], Quantization scale for gate/up projection (fp8 dequantization).
        down_proj_scale (nl.tensor, optional): [E, 1, H], Quantization scale for down projection (fp8 dequantization).
        gate_up_hidden_scale (nl.tensor, optional): [T+1, 1] or [T, 1], Per-token dequantization scale for hidden states
            in gate/up projection. Only used when is_quant=True. Combined with weight scale post-matmul.
        down_hidden_scale (nl.tensor, optional): [T+1, 1] or [T, 1], Per-token dequantization scale for intermediate states
            in down projection. Only used when is_quant=True. Combined with weight scale post-matmul.
        activation_function (ActFnType): Activation function for MLP block (default: SiLU).
        skip_dma (SkipMode): DMA skip mode configuration (default: SkipMode()).
        compute_dtype: Compute data type (default: nl.bfloat16).
        is_tensor_update_accumulating (bool): Whether to accumulate results over multiple blocks (default: True).
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Post or pre scaling mode (default: PRE_SCALE).
        gate_clamp_upper_limit (float, optional): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (float, optional): Lower clamp limit for gate projection.
        up_clamp_upper_limit (float, optional): Upper clamp limit for up projection.
        up_clamp_lower_limit (float, optional): Lower clamp limit for up projection.
        checkpoint_activation (bool): Enable activation checkpointing for gradient computation (default: False).
            When True, gate/up activations and optionally down activations are saved for backward pass.
        expert_affinity_multiply_on_I (bool): Controls where expert affinity scaling is applied (default: False).
            When True, apply affinity scaling on intermediate states (I) after activation.
            When False, apply affinity scaling on H (hidden states) after down projection.

    Returns:
        When checkpoint_activation=False:
            output (nl.tensor): [T+1, H], Output hidden states on HBM.
        When checkpoint_activation=True and expert_affinity_multiply_on_I=True:
            Tuple of (output, gate_up_activations_T):
                - output (nl.tensor): [T+1, H], Output hidden states on HBM.
                - gate_up_activations_T (nl.tensor): [N, 2, I_TP, B], Gate and up projection activations (transposed).
        When checkpoint_activation=True and expert_affinity_multiply_on_I=False:
            Tuple of (output, gate_up_activations_T, down_activations):
                - output (nl.tensor): [T+1, H], Output hidden states on HBM.
                - gate_up_activations_T (nl.tensor): [N, 2, I_TP, B], Gate and up projection activations (transposed).
                - down_activations (nl.tensor): [N, B, H], Down projection activations before affinity scaling.

    Notes:
        - All input/output tensors must have the same floating point dtype
        - token_position_to_id and block_to_expert must be nl.int32 tensors
        - Block size B must be multiple of 256
        - Hidden dimension H must be between 512 and 8192, and multiple of PSUM_SIZE (512)
        - I_TP must be divisible by 16
        - DMA weight skipping not yet supported

    Pseudocode:
        # Initialize output tensor
        output = zeros([T+1, H])

        # Process each block sequentially
        for block_idx in range(N):
            # Step 1: Load token indices and expert assignment for this block
            token_indices = load_token_indices(token_position_to_id, block_idx)
            block_expert = load_block_expert(block_to_expert, block_idx)

            # Step 2: Gather hidden states for tokens in this block
            block_hidden_states = gather_hidden_states(hidden_states, token_indices)  # [B, H]
            block_hidden_states_T = transpose(block_hidden_states)  # [H, B]

            # Step 3: Compute gate and up projections (sharded over I_TP dimension)
            # Each shard computes I_TP/num_shards of the intermediate dimension
            gate_proj = matmul(block_hidden_states_T, gate_weight[block_expert, :, 0, shard_slice])  # [B, I_TP_shard]
            up_proj = matmul(block_hidden_states_T, up_weight[block_expert, :, 1, shard_slice])  # [B, I_TP_shard]

            if linear_bias:
                gate_proj += gate_bias[block_expert, shard_slice]
                up_proj += up_bias[block_expert, shard_slice]

            if clamp_limits:
                gate_proj = clamp(gate_proj, gate_lower, gate_upper)
                up_proj = clamp(up_proj, up_lower, up_upper)

            # Step 4: Apply expert affinity scaling (PRE_SCALE mode)
            if scaling_mode == PRE_SCALE:
                expert_affinity = expert_affinities_masked[token_indices, block_expert]
                gate_proj *= expert_affinity
                up_proj *= expert_affinity

            # Step 5: Compute intermediate states with activation
            intermediate = activation(gate_proj) * up_proj  # [B, I_TP_shard], element-wise

            # Step 6: Compute down projection (sharded over I_TP dimension)
            block_output_shard = matmul(intermediate, down_weight[block_expert, shard_slice, :])  # [B, H]

            # Step 7: All-reduce across shards to get full output
            block_output = all_reduce_sum(block_output_shard)  # [B, H]

            if linear_bias:
                block_output += down_bias[block_expert]

            # Step 8: Apply expert affinity scaling (POST_SCALE mode)
            if scaling_mode == POST_SCALE:
                expert_affinity = expert_affinities_masked[token_indices, block_expert]
                block_output *= expert_affinity

            # Step 9: Accumulate to output (for top-K > 1)
            if is_tensor_update_accumulating:
                output[token_indices] += block_output
            else:
                output[token_indices] = block_output

            # Synchronize across shards
            barrier()

        return output
    """
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
    # Infer Config from input shape
    T, H = hidden_states.shape
    B = block_size
    E, _I_TP, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B
    SHARD_ID = nl.program_id(axis=0)
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=_I_TP)
    dims.derive_all_dims()

    inps = InputTensors(
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale
        if gate_up_proj_scale is not None
        else nl.ndarray((dims.E, 2, dims.I_TP), dtype=nl.float32, buffer=nl.shared_hbm),
        down_proj_scale=down_proj_scale
        if down_proj_scale is not None
        else nl.ndarray((dims.E, 1, dims.H), dtype=nl.float32, buffer=nl.shared_hbm),
        gate_up_hidden_scale=gate_up_hidden_scale
        if gate_up_hidden_scale is not None
        else nl.ndarray((dims.E, 2, 1), dtype=nl.float32, buffer=nl.shared_hbm),
        down_hidden_scale=down_hidden_scale
        if down_hidden_scale is not None
        else nl.ndarray((dims.E, 1), dtype=nl.float32, buffer=nl.shared_hbm),
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
    )

    # Pre-reshape scale tensors at entry point (direct params work with .reshape(); inps.X doesn't).
    # Per-channel uses E2I; block-256 uses the flat layout. Per-tensor needs no reshape.
    if gate_up_proj_scale is not None and not is_per_tensor and not is_block_quant:
        inps.gate_up_proj_scale_E2I = gate_up_proj_scale.reshape((dims.E, 2, dims.I_TP))
    if gate_up_proj_scale is not None and is_block_quant:
        H_blocks = dims.H // BLOCK_QUANT_SIZE
        I_blocks_total = dims.I_TP // BLOCK_QUANT_SIZE
        inps.gate_up_proj_scale_flat = gate_up_proj_scale.reshape((dims.E, H_blocks * 2 * I_blocks_total * TILE_SIZE))

    # Derive quant flags from direct kernel params. The NKI parser frontend can
    # eliminate dead branches when guards are `param != None` on direct args; it
    # cannot do so for attribute access on a nested NKIObject.
    _is_quant = gate_up_proj_scale != None and down_proj_scale != None
    if is_per_tensor:
        _quant_activation_mode = ActivationQuantMode.PER_TENSOR
    elif gate_up_hidden_scale != None or down_hidden_scale != None:
        _quant_activation_mode = ActivationQuantMode.PER_TOKEN
    else:
        _quant_activation_mode = ActivationQuantMode.NONE

    configs = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=(gate_and_up_proj_bias != None and down_proj_bias != None),
        activation_function=activation_function,
        is_quant=_is_quant,
        fuse_gate_and_up_load=(dims.H * dims.I_TP_sharded <= FUSE_GATE_WEIGHT_SIZE) and not skip_gate_proj,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        skip_gate_proj=skip_gate_proj,
        quant_activation_mode=_quant_activation_mode,
        quant_is_per_tensor=is_per_tensor,
        is_block_quant=is_block_quant,
        accumulation_dtype=accumulation_dtype if accumulation_dtype is not None else hidden_states.dtype,
    )
    check_blockwise_mm_shard_I_kernel_compatibility(dims, configs)

    # Cross-expert accumulator dtype (defaults to I/O dtype; fp32 is cast back before return).
    _acc_dtype = configs.accumulation_dtype
    _acc_upcast = _acc_dtype != hidden_states.dtype
    output = nl.ndarray(hidden_states.shape, dtype=_acc_dtype, buffer=nl.shared_hbm)
    gate_up_activations_T, down_activations = None, None
    if checkpoint_activation:
        gate_up_activations_T = nl.ndarray((N, 2, _I_TP, B), dtype=gate_up_proj_weight.dtype, buffer=nl.shared_hbm)
        if not expert_affinity_multiply_on_I:
            down_activations = nl.ndarray((N, B, H), dtype=down_proj_weight.dtype, buffer=nl.shared_hbm)

    outs = OutputTensors(
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        output=output,
    )

    if is_tensor_update_accumulating:
        output_initialization(output)
    for block_idx in nl.sequential_range(N):
        compute_one_block(block_idx, dims, inps, outs, configs, SHARD_ID)
        if dims.NUM_SHARDS == 1:
            nisa.core_barrier(output, (0))
        elif dims.NUM_SHARDS == 2:
            nisa.core_barrier(output, (0, 1))
        else:
            kernel_assert(False, "unsupported NUM_SHARDS")

    if _acc_upcast:
        # Cast the fp32 accumulator back to the I/O dtype via a single HBM->HBM DMA.
        output_ret = nl.ndarray(output.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=output_ret, src=output)
    else:
        output_ret = output

    if checkpoint_activation:
        if expert_affinity_multiply_on_I:
            return output_ret, gate_up_activations_T
        else:
            return output_ret, gate_up_activations_T, down_activations
    else:
        return output_ret


@nki.jit
def blockwise_mm_baseline_shard_intermediate_hybrid(
    conditions,
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    block_size: int,
    token_position_to_id,
    block_to_expert,
    num_static_block: Optional[int] = None,
    gate_and_up_proj_bias=None,
    down_proj_bias=None,
    # because we are using strategy 5,
    # gate_up_proj_scale shape: [E, 1, 2 * I_TP],
    # down_proj_scale shape: [E, 1, H]
    gate_up_proj_scale=None,
    down_proj_scale=None,
    gate_up_activations_T=None,
    down_activations=None,
    # meta parameters
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    accumulation_dtype=None,
):
    """
    Blockwise matrix multiplication kernel for MoE with hybrid static/dynamic loop control.

    Implements MoE layer at block granularity with intermediate dimension sharding. Uses hybrid approach:
    static loop for non-padded blocks, dynamic loop for padded blocks. Utilizes Trn hardware dynamic loop
    capabilities for efficient handling of variable-length sequences.

    Intended Usage:
        - Block size B: 256-1024 tokens (must be multiple of 256)
        - Total tokens T: Up to 32K tokens per call
        - Hidden dimension H: 512-8192 (optimal: 2048-4096)
        - Intermediate dimension I_TP: 2048-16384 (optimal: 8192)
        - Number of experts E: 8-64 (optimal: 8-16)
        - Use this variant when sequences have variable lengths with significant padding (>10% padded blocks)
        - Provides better performance than baseline when N_padded_blocks > E
        - Requires TRN2 hardware (NUM_SHARDS == 2)
        - Set num_static_block to known non-padded count for optimal performance

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens (after linearizing across the batch dimension)
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I_TP: Intermediate size / tp degree

    Args:
        conditions (nl.tensor): [N+1], Indicates whether block is padded (0) or non-padded (1). Last entry must be 0 to guarantee loop termination.
        hidden_states (nl.tensor): [T+1, H], Input hidden states on HBM. T+1 because padding token position is set to T.
        expert_affinities_masked (nl.tensor): [(T+1) * E, 1], Expert affinities for each token on HBM.
        gate_up_proj_weight (nl.tensor): [E, H, 2, I_TP], Concatenated gate and up projection weights on HBM.
        down_proj_weight (nl.tensor): [E, I_TP, H], Down projection weights on HBM.
        block_size (int): Number of tokens per block.
        token_position_to_id (nl.tensor): [N * B], Block index of corresponding tokens on HBM. Includes padding tokens (N * B >= T). Padding token id is set to T.
        block_to_expert (nl.tensor): [N, 1], Expert indices of corresponding blocks on HBM.
        num_static_block (int, optional): Number of non-padded blocks if known.
        gate_and_up_proj_bias (nl.tensor, optional): [E, 2, I_TP], Gate and up projection bias. For Swiglu, up_bias = up_bias + 1.
        down_proj_bias (nl.tensor, optional): [E, H], Down projection bias.
        gate_up_proj_scale (nl.tensor, optional): [E, 1, 2 * I_TP], Quantization scale for gate/up projection (fp8 dequantization).
        down_proj_scale (nl.tensor, optional): [E, 1, H], Quantization scale for down projection (fp8 dequantization).
        gate_up_activations_T (nl.tensor, optional): Currently not supported. Set to None.
        down_activations (nl.tensor, optional): Currently not supported. Set to None.
        activation_function (ActFnType): Activation function for MLP block (default: SiLU).
        skip_dma (SkipMode): DMA skip mode configuration (default: SkipMode()).
        compute_dtype: Compute data type (default: nl.bfloat16).
        is_tensor_update_accumulating (bool): Whether to accumulate results over multiple blocks (default: True).
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Post or pre scaling mode (default: POST_SCALE).
        gate_clamp_upper_limit (float, optional): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (float, optional): Lower clamp limit for gate projection.
        up_clamp_upper_limit (float, optional): Upper clamp limit for up projection.
        up_clamp_lower_limit (float, optional): Lower clamp limit for up projection.

    Returns:
        output (nl.tensor): [T+1, H], Output hidden states on HBM.

    Notes:
        - All input/output tensors must have the same floating point dtype
        - token_position_to_id and block_to_expert must be nl.int32 tensors
        - Block size B must be multiple of 256
        - Hidden dimension H must be between 512 and 8192, and multiple of PSUM_SIZE (512)
        - I_TP must be divisible by 16
        - DMA weight skipping not yet supported
        - Only works on TRN2 (requires NUM_SHARDS == 2)

    Pseudocode:
        # Initialize output tensor
        output = zeros([T+1, H])

        # Determine number of static (non-padded) blocks
        if num_static_block is provided:
            NUM_STATIC_BLOCKS = num_static_block
        else:
            NUM_STATIC_BLOCKS = N - E  # Assume last E blocks may be padded

        # Phase 1: Static loop over non-padded blocks (predictable iteration count)
        for block_idx in range(NUM_STATIC_BLOCKS):
            # Same processing as baseline kernel
            token_indices = load_token_indices(token_position_to_id, block_idx)
            block_expert = load_block_expert(block_to_expert, block_idx)
            block_hidden_states = gather_hidden_states(hidden_states, token_indices)
            block_hidden_states_T = transpose(block_hidden_states)

            # Gate/up projections (sharded over I_TP)
            gate_proj = matmul(block_hidden_states_T, gate_weight[block_expert, :, 0, shard_slice])
            up_proj = matmul(block_hidden_states_T, up_weight[block_expert, :, 1, shard_slice])

            if linear_bias:
                gate_proj += gate_bias[block_expert, shard_slice]
                up_proj += up_bias[block_expert, shard_slice]

            if clamp_limits:
                gate_proj = clamp(gate_proj, gate_lower, gate_upper)
                up_proj = clamp(up_proj, up_lower, up_upper)

            if scaling_mode == PRE_SCALE:
                expert_affinity = expert_affinities_masked[token_indices, block_expert]
                gate_proj *= expert_affinity
                up_proj *= expert_affinity

            intermediate = activation(gate_proj) * up_proj
            block_output_shard = matmul(intermediate, down_weight[block_expert, shard_slice, :])
            block_output = all_reduce_sum(block_output_shard)

            if linear_bias:
                block_output += down_bias[block_expert]

            if scaling_mode == POST_SCALE:
                expert_affinity = expert_affinities_masked[token_indices, block_expert]
                block_output *= expert_affinity

            if is_tensor_update_accumulating:
                output[token_indices] += block_output
            else:
                output[token_indices] = block_output

            barrier()

        # Phase 2: Dynamic loop over potentially padded blocks (data-dependent iteration)
        # Use hardware dynamic loop with condition tensor to skip padded blocks
        block_idx = NUM_STATIC_BLOCKS
        total_active_blocks = sum(conditions)  # Count non-padded blocks remaining

        while block_idx < N and conditions[block_idx] == 1:
            # Same block processing as above, but with dynamic indexing
            token_indices = load_token_indices_dynamic(token_position_to_id, block_idx)
            block_expert = load_block_expert_dynamic(block_to_expert, block_idx)

            # Process block (same steps as static loop)
            block_hidden_states = gather_hidden_states(hidden_states, token_indices)
            # ... (same gate/up/down projection logic)

            if is_tensor_update_accumulating:
                output[token_indices] += block_output
            else:
                output[token_indices] = block_output

            barrier()
            block_idx += 1

        return output
    """
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
    # Infer Config from input shape
    T, H = hidden_states.shape
    B = block_size
    E, _I_TP, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B
    SHARD_ID = nl.program_id(axis=0)
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=_I_TP)
    dims.derive_all_dims()

    inps = InputTensors(
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
    )

    configs = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=(gate_and_up_proj_bias != None and down_proj_bias != None),
        activation_function=activation_function,
        is_quant=(gate_up_proj_scale != None and down_proj_scale != None),
        fuse_gate_and_up_load=(dims.H * dims.I_TP_sharded <= FUSE_GATE_WEIGHT_SIZE),
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        # Hybrid kernel only supports per-channel weight quant (no per-tensor / block / activation rescale).
        quant_activation_mode=ActivationQuantMode.NONE,
        quant_is_per_tensor=False,
        is_block_quant=False,
        accumulation_dtype=accumulation_dtype if accumulation_dtype is not None else hidden_states.dtype,
    )

    check_blockwise_mm_shard_I_kernel_compatibility(dims, configs)
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    outs = OutputTensors(
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        output=output,
    )
    if is_tensor_update_accumulating:
        output_initialization(output)
    if num_static_block != None:
        NUM_STATIC_BLOCKS = num_static_block
        # kernel_assert(num_static_block <= N, f"num_static_block must be less than or equal to N")
        if num_static_block < T // B:
            print("num_static_block is less than T//B, this may lead to performance degradation")
    else:
        NUM_STATIC_BLOCKS = N - E
    kernel_assert(dims.NUM_SHARDS == 2, "shard-on-I with dynamic control flow only work on TRN2")
    # step 1: static loop over the non-padded blocks
    for block_idx in nl.sequential_range(NUM_STATIC_BLOCKS):
        compute_one_block(block_idx, dims, inps, outs, configs, SHARD_ID)
        nisa.core_barrier(output, (0, 1))
    # step 2: dynamic loop over the padded blocks
    configs.use_dynamic_while = True
    dynamic_start_index = NUM_STATIC_BLOCKS
    cond_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    conditions_sbuf = nl.ndarray((1, conditions.shape[0]), buffer=nl.sbuf, dtype=nl.int32)

    nisa.dma_copy(dst=conditions_sbuf, src=conditions.reshape((1, conditions.shape[0])))
    nisa.tensor_reduce(dst=cond_sbuf, data=conditions_sbuf, op=nl.add, axis=1)
    cond_reg = nisa.register_alloc()
    nisa.register_load(cond_reg, cond_sbuf)
    block_idx = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.memset(block_idx, value=NUM_STATIC_BLOCKS)

    def _dynamic_block_body(i):
        compute_one_block(block_idx, dims, inps, outs, configs, SHARD_ID)
        nisa.core_barrier(output, (0, 1))
        nisa.tensor_scalar(dst=block_idx, data=block_idx, op0=nl.add, operand0=1)
        nisa.core_barrier(block_idx, (0, 1))

    nl.fori_loop(NUM_STATIC_BLOCKS, cond_reg, _dynamic_block_body)
    return output


def check_blockwise_mm_shard_I_kernel_compatibility(dims: DimensionSizes, configs: Configs):
    """
    Validate kernel input dimensions and configuration compatibility.

    Args:
        dims: DimensionSizes object with tensor dimensions.
        configs: Configs object with kernel configuration.

    Raises:
        AssertionError: If any compatibility check fails.
    """
    kernel_assert(dims.B % 256 == 0, f"Blocksize must be a multiple of 256")
    kernel_assert(512 <= dims.H <= 8192, f"Hidden dims must be between 512 and 8192, found {dims.H}")
    kernel_assert(dims.H % TILE_SIZE == 0, f"Hidden dim size must be multiples of {TILE_SIZE}, found {dims.H} ")
    kernel_assert(dims.I_TP % 16 == 0, f"down_proj_weight I must be divisible by 16, found {dims.I_TP} . Please pad it")
    kernel_assert(
        dims.I_TP % dims.NUM_SHARDS == 0,
        f"I_TP ({dims.I_TP}) must be divisible by NUM_SHARDS ({dims.NUM_SHARDS}) for intermediate sharding",
    )
    kernel_assert(
        configs.skip_dma.skip_weight == False, "DMA weight skipping is not yet supported by the BWMM shard on I kernel"
    )
    if configs.is_block_quant:
        # Block quant size must be 256x256 to be compatible with double_row matmul (256 H contraction)
        kernel_assert(dims.H % 256 == 0, f"Block quant requires H divisible by 256, found {dims.H}")
        kernel_assert(dims.I_TP % 256 == 0, f"Block quant requires I_TP divisible by 256, found {dims.I_TP}")


def output_initialization(output, shard_id=None):
    """
    Zero initialize buffer at `output`. Required for accumulation (top K > 1).

    Args:
        output: External memory tensor to initialize.
        shard_id: Optional shard ID for multi-shard initialization.

    Returns:
        None: Initializes output buffer in-place with zeros.
    """
    if shard_id == None:
        T, H = output.shape
    else:
        (
            _,
            T,
            H,
        ) = output.shape

    for tile_idx in range(div_ceil(T, TILE_SIZE)):
        zeros = nl.ndarray((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros, value=0.0)

        if shard_id != None:
            num_elements = min(TILE_SIZE, T - tile_idx * TILE_SIZE)
            nisa.dma_copy(
                src=zeros[0:num_elements, 0:H], dst=output[shard_id, nl.ds(tile_idx * TILE_SIZE, num_elements), 0:H]
            )
        else:
            num_elements = min(TILE_SIZE, T - tile_idx * TILE_SIZE)
            nisa.dma_copy(src=zeros[0:num_elements, 0:H], dst=output[nl.ds(tile_idx * TILE_SIZE, num_elements), 0:H])


def load_token_indices(token_position_to_id, block_idx, dims: DimensionSizes):
    """
    Load and transpose token indices for the current block.

    Args:
        token_position_to_id: Token position to ID mapping tensor.
        block_idx: Current block index.
        dims: DimensionSizes object containing B and NUM_B_TILES.

    Returns:
        token_position_to_id_sbuf: Tensor of shape (TILE_SIZE, NUM_B_TILES) containing transposed token indices.
    """
    token_position_to_id_sbuf = nl.ndarray((TILE_SIZE, dims.NUM_B_TILES), dtype=nl.int32, buffer=nl.sbuf)

    token_pos_to_id_fp32 = nl.ndarray((1, TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)

    for b_tile_idx in range(dims.NUM_B_TILES):
        offset = block_idx * dims.B + TILE_SIZE * b_tile_idx
        """
        This instruction is causing an xbar transpose and having offset as b_tile_idx is throwing compilation errors 
        since it expects 32B offsets.
        """
        nisa.dma_copy(
            token_pos_to_id_fp32.ap(pattern=[[TILE_SIZE, 1], [1, TILE_SIZE]]),
            token_position_to_id.ap(pattern=[[TILE_SIZE, 1], [1, TILE_SIZE]], offset=offset),
        )

        transposed_token_pos_to_id_fp32 = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=transposed_token_pos_to_id_fp32, data=token_pos_to_id_fp32)

        nisa.tensor_copy(token_position_to_id_sbuf[:, b_tile_idx], transposed_token_pos_to_id_fp32)

    return token_position_to_id_sbuf


def load_dynamic_block_expert(block_to_expert, block_idx):
    """Load expert index for the current block.

    Args:
        block_to_expert: Tensor mapping blocks to expert indices.
        block_idx: Current block index (dynamic).

    Returns:
        expert_idx_tensor: Tensor of shape (1, 1) containing the expert index.
    """
    expert_idx_tensor = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(expert_idx_tensor.ap([[1, 1], [1, 1]]), block_to_expert.ap([[1, 1], [1, 1]], scalar_offset=block_idx))
    return expert_idx_tensor


def create_block_hidden_states(H, NUM_TILES, dtype):
    """
    Create list of tensors for block hidden states.

    Args:
        H: Hidden dimension size.
        NUM_TILES: Number of tiles.
        dtype: Data type for tensors.

    Returns:
        block_hidden_states: List of tensors, each of shape (TILE_SIZE, H).
    """
    block_hidden_states = []
    for tile_idx in range(NUM_TILES):
        tile = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
        block_hidden_states.append(tile)

    return block_hidden_states


def load_hidden_states(
    hidden_states,
    block_hidden_states,
    token_indices,
    NUM_TILES,
    dtype,
    skip_dma: SkipMode = SkipMode(),
    token_indices_offset=0,
):
    """
    Load hidden states for tokens in the current block.

    Args:
        hidden_states: Hidden States of shape (T, H).
        block_hidden_states: An sbuf tensor which is loaded with hidden states for the current block.
        token_indices: Token Indices for the current block.
        NUM_TILES: Number of Tiles for the given block when TILE_SIZE is used.
        dtype: Data Type.
        skip_dma: Skip DMA.
        token_indices_offset: An Offset used when block tiling is done on the current block.
            This is used for the dropping kernel where number of blocks is number of experts resulting in block sizes > 1024.

    Returns:
        None: Loads block_hidden_states with the hiddens states of all the tokens belonging to the current block.
    """
    T, H = hidden_states.shape

    for tile_idx in range(NUM_TILES):
        if skip_dma.skip_token:
            nisa.memset(dst=block_hidden_states[tile_idx], value=0)
        tmp_vector_index = nl.ndarray((TILE_SIZE, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_vector_index, src=token_indices[0:TILE_SIZE, token_indices_offset + tile_idx])
        nisa.dma_copy(
            src=hidden_states.ap([[H, TILE_SIZE], [1, H]], offset=0, vector_offset=tmp_vector_index, indirect_dim=0),
            dst=block_hidden_states[tile_idx][0:TILE_SIZE, 0:H],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


def transpose_hidden_states_allocated(block_hidden_states, H, B, compute_dtype, is_quant=False):
    """
    Transpose block hidden states from B x H to H x B.

    Args:
        block_hidden_states: List of tensors, each of shape (TILE_SIZE, H).
        H: Hidden dimension size.
        B: Block size.
        compute_dtype: Compute data type.
        is_quant: If True, use a 4D contiguous layout per h_outer (needed for double_row .ap() views
            in the quant path). If False, use the mainline list-of-list-of-3D-tiles layout — the 4D
            layout triggers backend NCC_IGCA108 (loop-carried-dependency) errors for non-power-of-2 H
            (e.g. H=4864) when the consumer accesses tiles via direct 4D indexing.

    Returns:
        block_hidden_states_T:
            - is_quant=True: list[h_outer_tripcount] of 4D tensors of shape
              (TILE_SIZE, h_inner_tripcount, block_psum_tiles, free_size).
            - is_quant=False: nested list[h_outer_tripcount][h_inner_tripcount] of 3D tensors of shape
              (TILE_SIZE, block_psum_tiles, free_size).
    """
    h_outer_tripcount = div_ceil(H, PSUM_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    linearized_tripcount = div_ceil(H, TILE_SIZE)
    block_psum_tiles = div_ceil(B, PSUM_SIZE)
    free_size = min(PSUM_SIZE, B)
    block_hidden_states_T = []
    for h_outer_idx in range(h_outer_tripcount):
        if is_quant:
            # Single contiguous 4D tile so the consumer can build double_row views via .ap().
            tile = nl.ndarray(
                (TILE_SIZE, h_inner_tripcount, block_psum_tiles, free_size), dtype=compute_dtype, buffer=nl.sbuf
            )
            block_hidden_states_T.append(tile)
        else:
            inner_list = []
            for _ in range(h_inner_tripcount):
                inner_tile = nl.ndarray((TILE_SIZE, block_psum_tiles, free_size), dtype=compute_dtype, buffer=nl.sbuf)
                inner_list.append(inner_tile)
            block_hidden_states_T.append(inner_list)

    block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)
    identity_sbuf = nl.shared_identity_matrix(TILE_SIZE, dtype=compute_dtype)

    for psum_tile_idx in range(block_psum_tiles):
        for h_outer_idx in range(h_outer_tripcount):
            for h_inner_idx in range(h_inner_tripcount):
                i_lin = h_outer_idx * h_inner_tripcount + h_inner_idx
                if i_lin >= linearized_tripcount:
                    continue
                trans_f_offset = TILE_SIZE * h_inner_idx + PSUM_SIZE * h_outer_idx
                psum_dtype = (
                    block_hidden_states[0].dtype if nisa.get_nc_version() >= nisa.nc_version.gen3 else nl.float32
                )
                tmp_res = nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=psum_dtype, buffer=nl.psum)
                for b_tile_idx in range(block_free_tiles):
                    offset = TILE_SIZE * b_tile_idx
                    nisa.nc_matmul(
                        stationary=block_hidden_states[block_free_tiles * psum_tile_idx + b_tile_idx][
                            0:TILE_SIZE, trans_f_offset : trans_f_offset + TILE_SIZE
                        ],
                        moving=identity_sbuf[0:TILE_SIZE, 0:TILE_SIZE],
                        dst=tmp_res[0:TILE_SIZE, offset : offset + TILE_SIZE],
                        is_transpose=True,
                    )

                if is_quant:
                    dst = block_hidden_states_T[h_outer_idx][0:TILE_SIZE, h_inner_idx, psum_tile_idx, 0:free_size]
                else:
                    dst = block_hidden_states_T[h_outer_idx][h_inner_idx][0:TILE_SIZE, psum_tile_idx, 0:free_size]
                nisa.tensor_copy(
                    src=tmp_res[0:TILE_SIZE, 0:free_size],
                    dst=dst,
                )

    return block_hidden_states_T


def load_block_hidden_scale(hidden_scale_hbm, token_indices, NUM_B_TILES, skip_dma):
    """
    Load per-token hidden scales for the current block, gathered by token_indices.

    Args:
        hidden_scale_hbm: [T+1, 1] or [T, 1], Per-token scale tensor in HBM.
        token_indices: (TILE_SIZE, NUM_B_TILES), Token indices for the current block.
        NUM_B_TILES: Number of B tiles.
        skip_dma: Skip DMA configuration.

    Returns:
        block_hidden_scale: List of (TILE_SIZE, 1) tensors, one per B tile.
    """
    block_hidden_scale = []
    for tile_idx in range(NUM_B_TILES):
        scale_tile = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        if skip_dma.skip_token:
            nisa.memset(dst=scale_tile, value=0.0)
        tmp_vector_index = nl.ndarray((TILE_SIZE, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_vector_index, src=token_indices[0:TILE_SIZE, tile_idx])
        nisa.dma_copy(
            src=hidden_scale_hbm.ap([[1, TILE_SIZE], [1, 1]], offset=0, vector_offset=tmp_vector_index, indirect_dim=0),
            dst=scale_tile[0:TILE_SIZE, 0:1],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
        block_hidden_scale.append(scale_tile)
    return block_hidden_scale


def compute_one_block(block_idx, dims: DimensionSizes, inps: InputTensors, outs: OutputTensors, cfg: Configs, shard_id):
    """
    Compute MoE forward pass for a single block.

    Processes one block of tokens through gate/up projections, activation,
    down projection, and accumulates results to output.

    Args:
        block_idx: Current block index.
        dims: DimensionSizes object with tensor dimensions.
        inps: InputTensors object with input tensors.
        outs: OutputTensors object with output tensors.
        cfg: Configs object with kernel configuration.
        shard_id: Current shard ID for intermediate dimension sharding.
    """
    if cfg.use_dynamic_while:
        token_indices = load_token_indices_dynamic_block(
            inps.token_position_to_id, block_idx, dims, skip_dma=cfg.skip_dma
        )
        block_expert = load_dynamic_block_expert(inps.block_to_expert, block_idx)
    else:
        token_indices = load_token_indices(inps.token_position_to_id, block_idx, dims)
        block_expert = load_block_expert(inps.block_to_expert, block_idx)

    block_hidden_states = create_block_hidden_states(dims.H, dims.NUM_B_TILES, cfg.compute_dtype)

    # hidden states are unsharded tensors
    load_hidden_states(
        inps.hidden_states, block_hidden_states, token_indices, dims.NUM_B_TILES, cfg.compute_dtype, cfg.skip_dma
    )

    # Use flattened quant config fields for compiler frontend compatibility
    quant_activation_mode = cfg.quant_activation_mode
    quant_is_per_tensor = cfg.quant_is_per_tensor
    down_hidden_scale_hbm = inps.down_hidden_scale

    # prepare gate/up dequantization scale
    if cfg.is_quant:
        if cfg.is_block_quant:
            # Block quant scales loaded in batch below, before projection call
            gup_scale = None  # not used for block quant
        elif quant_is_per_tensor:
            # Per-tensor quant: scale shape [E, 2, 1] — one scalar per expert per projection
            gup_scale_per_tensor = []
            for gate_or_up in range(2):
                if cfg.skip_gate_proj and gate_or_up == 0:
                    gup_scale_per_tensor.append(None)
                    continue
                s = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=s[0:1, 0:1],
                    src=inps.gate_up_proj_scale.reshape((dims.E, 2)).ap(
                        pattern=[[2, 1], [1, 1]],
                        offset=gate_or_up,
                        scalar_offset=block_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.error,
                )
                stream_shuffle_broadcast(s[0:1, 0:1], s)
                gup_scale_per_tensor.append(s)
            # Reuse same scale for all i_tiles — same code path as per-channel
            gup_scale = []
            for _ in range(dims.GUP_N_TILES):
                gup_scale.append([gup_scale_per_tensor[0], gup_scale_per_tensor[1]])
            gup_block_scale = None
        else:
            # Per-channel quant: scale shape [E, 1, 2*I_TP]
            gup_scale = []
            for _ in range(dims.GUP_N_TILES):
                tmp = []
                for _ in range(2):
                    tmp.append(nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf))
                gup_scale.append(tmp)

            gate_up_proj_scale_reshaped = inps.gate_up_proj_scale.reshape((dims.E, 2, dims.I_TP))

            for gup_tile_idx in range(dims.GUP_N_TILES):
                for gate_or_up in range(2):
                    if cfg.skip_gate_proj and gate_or_up == 0:
                        continue
                    elem_offset = TILE_SIZE * gup_tile_idx + shard_id * dims.I_TP_sharded
                    num_elems = min(TILE_SIZE, dims.I_TP - elem_offset)

                    nisa.dma_copy(
                        dst=gup_scale[gup_tile_idx][gate_or_up][0:num_elems, :],
                        src=gate_up_proj_scale_reshaped.ap(
                            pattern=[[1, num_elems], [1, 1]],
                            offset=gate_or_up * dims.I_TP + elem_offset,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.error,
                    )
            gup_block_scale = None

    # Load and apply hidden/activation scales for static quantization
    gate_up_hs = None
    down_hs = None
    if cfg.is_quant and quant_activation_mode == ActivationQuantMode.PER_TOKEN and inps.gate_up_hidden_scale != None:
        gate_up_hs_per_tile = load_block_hidden_scale(
            inps.gate_up_hidden_scale, token_indices, dims.NUM_B_TILES, cfg.skip_dma
        )
        free_size = min(PSUM_SIZE, dims.B)
        N_PSUM_TILE = div_ceil(dims.B, PSUM_SIZE)
        block_free_tiles = free_size // TILE_SIZE
        gate_up_hs = []
        for psum_tile_idx in range(N_PSUM_TILE):
            hs_row = nl.ndarray((1, free_size), dtype=nl.float32, buffer=nl.psum)
            for bt in range(block_free_tiles):
                b_tile_idx = psum_tile_idx * block_free_tiles + bt
                num_f = min(TILE_SIZE, dims.B - b_tile_idx * TILE_SIZE)
                nisa.nc_transpose(
                    data=gate_up_hs_per_tile[b_tile_idx][0:num_f, 0:1],
                    dst=hs_row[0:1, bt * TILE_SIZE : bt * TILE_SIZE + num_f],
                )
            hs_row_sbuf = nl.ndarray((1, free_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hs_row_sbuf, src=hs_row)
            hs_broadcast = nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.sbuf)
            stream_shuffle_broadcast(hs_row_sbuf, hs_broadcast)
            gate_up_hs.append(hs_broadcast)
    elif cfg.is_quant and quant_activation_mode == ActivationQuantMode.PER_TENSOR and inps.gate_up_hidden_scale != None:
        # Per-tensor activation: [E, 2, 1] — pre-combine with weight scale
        # Multiply each gup_scale entry by the activation scale
        # Note: for per-tensor, all i_tiles share the same tensor, so only multiply once
        for gate_or_up in range(2):
            if cfg.skip_gate_proj and gate_or_up == 0:
                continue
            act_s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=act_s[0:1, 0:1],
                src=inps.gate_up_hidden_scale.reshape((dims.E, 2)).ap(
                    pattern=[[2, 1], [1, 1]],
                    offset=gate_or_up,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
            )
            act_s_broadcast = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
            stream_shuffle_broadcast(act_s[0:1, 0:1], act_s_broadcast)
            # Only multiply once — all gup_scale[gup_tile_idx][gate_or_up] share the same tensor
            nisa.tensor_tensor(
                data1=gup_scale[0][gate_or_up],
                data2=act_s_broadcast,
                op=nl.multiply,
                dst=gup_scale[0][gate_or_up],
            )
        # gate_up_hs stays None — activation scale folded into weight scale

    if cfg.is_quant and quant_activation_mode == ActivationQuantMode.PER_TOKEN and down_hidden_scale_hbm != None:
        down_hs = load_block_hidden_scale(inps.down_hidden_scale, token_indices, dims.NUM_B_TILES, cfg.skip_dma)
    elif cfg.is_quant and quant_activation_mode == ActivationQuantMode.PER_TENSOR and down_hidden_scale_hbm != None:
        # Per-tensor down activation: folded into down_scale inside compute_down_proj
        # Keep down_hs as None — the down proj function handles it internally
        pass

    # 4D layout is only needed for block-quant's double_row .ap() views.
    # Per-channel and non-quant use the mainline 3D-list-of-list layout.
    block_hidden_states_T = transpose_hidden_states_allocated(
        block_hidden_states, dims.H, dims.B, cfg.compute_dtype, is_quant=cfg.is_block_quant
    )

    # Cast transposed hidden states to FP8 only for block-quant, which still uses double_row.
    # Per-channel quant falls back to mainline-style bf16-upcast + standard matmul (see below)
    # because the double_row .ap() pattern produces a partition/free-product mismatch that the
    # MLIR tracer rejects (and the parser frontend silently miscompiles for per-channel scales).
    if cfg.is_quant and cfg.is_block_quant:
        h_outer_tc = div_ceil(dims.H, PSUM_SIZE)
        h_inner_tc = PSUM_SIZE // TILE_SIZE
        bpt = div_ceil(dims.B, PSUM_SIZE)
        fs = min(PSUM_SIZE, dims.B)
        block_hidden_states_T_fp8 = []
        for h_outer_idx in range(h_outer_tc):
            tile_fp8 = nl.ndarray((TILE_SIZE, h_inner_tc, bpt, fs), dtype=cfg.weight_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=tile_fp8[0:TILE_SIZE, 0:h_inner_tc, 0:bpt, 0:fs],
                src=block_hidden_states_T[h_outer_idx][0:TILE_SIZE, 0:h_inner_tc, 0:bpt, 0:fs],
            )
            block_hidden_states_T_fp8.append(tile_fp8)
        block_hidden_states_T_matmul = block_hidden_states_T_fp8
    else:
        block_hidden_states_T_matmul = block_hidden_states_T

    # Load current expert's block quant scales in one batch
    gup_block_scale = None
    if cfg.is_quant and cfg.is_block_quant:
        H_blocks = dims.H // BLOCK_QUANT_SIZE
        I_blocks_total = dims.I_TP // BLOCK_QUANT_SIZE
        I_blocks_sharded = dims.I_TP_sharded // BLOCK_QUANT_SIZE
        i_block_offset = shard_id * I_blocks_sharded
        num_gup_block_scales = H_blocks * 2 * I_blocks_sharded
        gup_block_scale_buf = nl.ndarray((TILE_SIZE, num_gup_block_scales), dtype=nl.float32, buffer=nl.sbuf)
        gup_scale_reshaped = inps.gate_up_proj_scale.reshape((dims.E, H_blocks * 2 * I_blocks_total * TILE_SIZE))
        for h_blk in range(H_blocks):
            for gate_or_up_idx in range(2):
                flat_dst = (h_blk * 2 + gate_or_up_idx) * I_blocks_sharded
                src_offset = (h_blk * 2 + gate_or_up_idx) * I_blocks_total * TILE_SIZE + i_block_offset * TILE_SIZE
                nisa.dma_copy(
                    dst=gup_block_scale_buf[0:TILE_SIZE, flat_dst : flat_dst + I_blocks_sharded],
                    src=gup_scale_reshaped.ap(
                        pattern=[[1, TILE_SIZE], [TILE_SIZE, I_blocks_sharded]],
                        offset=src_offset,
                        scalar_offset=block_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.error,
                )
        gup_block_scale = gup_block_scale_buf

    gate_and_up_proj_states = compute_gate_and_up_projections_shard_on_intermediate(
        inps,
        block_expert,
        block_hidden_states_T_matmul,
        shard_id,
        dims,
        cfg,
        gup_scale=gup_scale if (cfg.is_quant and not cfg.is_block_quant) else None,
        gate_up_activations_T=outs.gate_up_activations_T,
        block_idx=block_idx,
        gate_up_hidden_scale=gate_up_hs,
        gup_block_scale=gup_block_scale if (cfg.is_quant and cfg.is_block_quant) else None,
    )
    if cfg.scaling_mode == ExpertAffinityScaleMode.PRE_SCALE or cfg.expert_affinity_multiply_on_I:
        expert_affinity_T_broadcasted = calculate_expert_affinity_T(inps, dims, cfg, block_expert, token_indices)
    else:
        expert_affinity_T_broadcasted = None

    intermediate_states = compute_intermediate_states(
        gate_and_up_proj_states=gate_and_up_proj_states,
        B=dims.B,
        I_TP=dims.I_TP_sharded,
        dtype=cfg.compute_dtype,
        activation_function=cfg.activation_function,
        expert_affinity_T_broadcasted=expert_affinity_T_broadcasted,
        gup_scale=None,
        expert_affinity_multiply_on_I=cfg.expert_affinity_multiply_on_I,
        skip_gate_proj=cfg.skip_gate_proj,
    )

    expert_affinity = None
    need_affinity_for_post_scale = (
        cfg.scaling_mode == ExpertAffinityScaleMode.POST_SCALE and not cfg.expert_affinity_multiply_on_I
    )
    need_affinity_for_bias = cfg.expert_affinity_multiply_on_I and cfg.linear_bias
    if need_affinity_for_post_scale or need_affinity_for_bias:
        expert_affinity = calculate_expert_affinities(
            inps.expert_affinities_masked,
            token_indices,
            block_expert,
            dims.E,
            dims.NUM_B_TILES,
            cfg.compute_dtype,
            cfg.skip_dma,
            cast_to_f32=False if cfg.is_tensor_update_accumulating else True,
        )

    if cfg.is_tensor_update_accumulating:
        # Match accumulator precision when upcasting, else original compute dtype (None=io).
        _ad = cfg.accumulation_dtype if cfg.accumulation_dtype is not None else cfg.io_dtype
        if _ad != cfg.io_dtype:
            _block_old_dtype = _ad
        else:
            _block_old_dtype = cfg.compute_dtype
        block_old = load_old_block(outs.output, token_indices, dims.NUM_B_TILES, _block_old_dtype, cfg.skip_dma)
    else:
        block_old = None

    block_new = compute_down_proj_shard_on_intermediate(
        intermediate_states,
        inps,
        dims,
        cfg,
        block_expert,
        expert_affinity,
        block_old,
        shard_id,
        allocate=True,
        outs=outs,
        block_idx=block_idx,
        down_hidden_scale=down_hs,
    )
    store_block_output_shard_over_block_size(outs.output, block_new, token_indices, dims, shard_id, cfg.skip_dma)


def compute_gate_and_up_projections_shard_on_intermediate(
    inps: InputTensors,
    block_expert,
    block_hidden_states_T,
    shard_id,
    dims: DimensionSizes,
    cfg: Configs,
    gup_scale=None,
    gate_up_activations_T=None,
    allocate=False,
    block_idx=None,
    activation_block_write_offset=0,
    gate_up_hidden_scale=None,
    gup_block_scale=None,
):
    """Compute gate and up projections.

    Args:
        inps: InputTensors object.
        block_expert: Expert index tensor.
        block_hidden_states_T: Transposed hidden states.
        shard_id: Current shard ID.
        dims: DimensionSizes object.
        cfg: Configs object.
        gup_scale: Optional weight quantization scale.
        gate_up_activations_T: Optional output tensor for activations.
        allocate: Whether to allocate new tensors.
        block_idx: Optional block index.
        activation_block_write_offset: Offset for writing activations.
        gate_up_hidden_scale: Optional list of (TILE_SIZE, free_size) broadcasted per-token hidden scales, one per psum_tile.

    Returns:
        gate_and_up_proj_res_sbuf_lst: Nested list of shape [2][N_PSUM_TILE][GUP_N_TILES],
                                        where each element is a tensor of shape (TILE_SIZE, free_size).
    """

    N_PSUM_TILE = div_ceil(dims.B, PSUM_SIZE)
    GUP_N_TILES = div_ceil(dims.I_TP_sharded, TILE_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    block_psum_tiles = N_PSUM_TILE

    # block_hidden_states_T is a list of 4D tiles (block-quant only, for double_row .ap()
    # views) or a list-of-list of 3D tiles (non-quant and per-channel quant; mainline-style).
    if cfg.is_block_quant:
        free_size = block_hidden_states_T[0].shape[-1]
    else:
        free_size = block_hidden_states_T[0][0].shape[-1]
    h_outer_tripcount = div_ceil(dims.H, PSUM_SIZE)
    linearized_h_tripcount = div_ceil(dims.H, TILE_SIZE)

    gate_and_up_proj_res_sbuf_lst = []
    for gate_or_up in range(2):
        psum_lst = []
        for psum_tile_idx in range(N_PSUM_TILE):
            i_lst = []
            for i_tile_idx in range(GUP_N_TILES):
                i_lst.append(nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.sbuf))
            psum_lst.append(i_lst)
        gate_and_up_proj_res_sbuf_lst.append(psum_lst)

    if cfg.linear_bias:
        gate_up_bias = nl.ndarray((2, dims.I_TP_sharded), dtype=cfg.compute_dtype, buffer=nl.sbuf)
        gate_up_bias_T = nl.ndarray((TILE_SIZE, 2 * GUP_N_TILES), dtype=cfg.compute_dtype, buffer=nl.sbuf)

        _, _, I_TP_total = inps.gate_and_up_proj_bias.shape

        for gate_or_up in range(2):
            if cfg.skip_gate_proj and gate_or_up == 0:
                continue
            # Calculate offset for this gate_or_up position and shard
            offset = gate_or_up * I_TP_total + shard_id * dims.I_TP_sharded

            nisa.dma_copy(
                dst=gate_up_bias[gate_or_up : gate_or_up + 1, 0 : dims.I_TP_sharded],
                src=inps.gate_and_up_proj_bias.ap(
                    pattern=[
                        [I_TP_total, 1],  # Dim1: stride=I_TP_total, count=1
                        [1, dims.I_TP_sharded],  # Dim2: stride=1, count=I_TP_sharded
                    ],
                    offset=offset,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
            )

        tmp_psum = nl.ndarray((TILE_SIZE, 2 * GUP_N_TILES), dtype=gate_up_bias.dtype, buffer=nl.psum)
        for gup_tile_idx in range(GUP_N_TILES):
            actual_f_size = min(TILE_SIZE, dims.I_TP_sharded - gup_tile_idx * TILE_SIZE)
            nisa.nc_transpose(
                data=gate_up_bias[0:2, gup_tile_idx * TILE_SIZE : gup_tile_idx * TILE_SIZE + actual_f_size],
                dst=tmp_psum[0:actual_f_size, gup_tile_idx * 2 : (gup_tile_idx + 1) * 2],
            )

        nisa.tensor_copy(
            dst=gate_up_bias_T[0:TILE_SIZE, 0 : 2 * GUP_N_TILES], src=tmp_psum[0:TILE_SIZE, 0 : 2 * GUP_N_TILES]
        )

    if cfg.fuse_gate_and_up_load:
        gup_weights = load_gate_up_proj_weights_shard_intermediate(
            0, inps.gate_up_proj_weight, block_expert, cfg, dims.NUM_SHARDS, shard_id, load_dst=None
        )

    for gate_or_up in range(2):
        if cfg.skip_gate_proj and gate_or_up == 0:
            continue
        if not cfg.fuse_gate_and_up_load:
            gup_weights = load_gate_up_proj_weights_shard_intermediate(
                gate_or_up, inps.gate_up_proj_weight, block_expert, cfg, dims.NUM_SHARDS, shard_id, None
            )

        gate_or_up_psum_lst = []
        for psum_tile_idx in range(N_PSUM_TILE):
            i_lst = []
            for i_tile_idx in range(GUP_N_TILES):
                i_lst.append(nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum))
            gate_or_up_psum_lst.append(i_lst)

        for i_tile_idx in range(GUP_N_TILES):
            i_start = TILE_SIZE * i_tile_idx
            num_i_tile = min(TILE_SIZE, dims.I_TP_sharded - TILE_SIZE * i_tile_idx)

            for h_outer_idx in range(h_outer_tripcount):
                if gup_block_scale != None:
                    # Block quant with 256x256 blocks: double_row matmul → scale → accumulate in sbuf
                    # Each double_row pair (256 H) = one block along H, shares one scale
                    i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE  # 2
                    for h_inner_idx in range(0, h_inner_tripcount, 2):
                        h_lin = h_outer_idx * h_inner_tripcount + h_inner_idx
                        h_block = h_lin // i_tiles_per_block
                        i_block = i_tile_idx // i_tiles_per_block
                        for b_psum_idx in range(N_PSUM_TILE):
                            tmp_psum = nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum)
                            # Double-row FP8 matmul — .ap() on contiguous weight and hidden_T buffers
                            N_WEIGHTS = gup_weights[h_outer_idx].shape[2]
                            I_TP_per_shard = gup_weights[h_outer_idx].shape[3]
                            w_h_stride = N_WEIGHTS * I_TP_per_shard
                            w_total_free = h_inner_tripcount * w_h_stride
                            if cfg.fuse_gate_and_up_load:
                                w_offset = h_inner_idx * w_h_stride + gate_or_up * I_TP_per_shard + i_start
                            else:
                                w_offset = h_inner_idx * w_h_stride + i_start
                            dr_weight = gup_weights[h_outer_idx].ap(
                                pattern=[[w_total_free, TILE_SIZE], [w_h_stride, 2], [1, num_i_tile]],
                                offset=w_offset,
                            )
                            dr_moving = block_hidden_states_T[h_outer_idx].ap(
                                pattern=[
                                    [h_inner_tripcount * block_psum_tiles * free_size, TILE_SIZE],
                                    [block_psum_tiles * free_size, 2],
                                    [1, free_size],
                                ],
                                offset=h_inner_idx * block_psum_tiles * free_size + b_psum_idx * free_size,
                            )
                            nisa.nc_matmul(
                                dst=tmp_psum[0:TILE_SIZE, 0:free_size],
                                stationary=dr_weight,
                                moving=dr_moving,
                                perf_mode="double_row",
                            )
                            # Scale by block scalar and accumulate
                            i_block = i_tile_idx // (BLOCK_QUANT_SIZE // TILE_SIZE)
                            I_blocks_sharded = dims.I_TP_sharded // BLOCK_QUANT_SIZE
                            flat_scale_idx = (h_block * 2 + gate_or_up) * I_blocks_sharded + i_block
                            if h_outer_idx == 0 and h_inner_idx == 0:
                                nisa.tensor_scalar(
                                    data=tmp_psum[0:TILE_SIZE, 0:free_size],
                                    op0=nl.multiply,
                                    operand0=gup_block_scale[0:TILE_SIZE, flat_scale_idx : flat_scale_idx + 1],
                                    dst=gate_and_up_proj_res_sbuf_lst[gate_or_up][b_psum_idx][i_tile_idx][
                                        0:TILE_SIZE, 0:free_size
                                    ],
                                )
                            else:
                                nisa.scalar_tensor_tensor(
                                    data=tmp_psum[0:TILE_SIZE, 0:free_size],
                                    op0=nl.multiply,
                                    operand0=gup_block_scale[0:TILE_SIZE, flat_scale_idx : flat_scale_idx + 1],
                                    op1=nl.add,
                                    operand1=gate_and_up_proj_res_sbuf_lst[gate_or_up][b_psum_idx][i_tile_idx][
                                        0:TILE_SIZE, 0:free_size
                                    ],
                                    dst=gate_and_up_proj_res_sbuf_lst[gate_or_up][b_psum_idx][i_tile_idx][
                                        0:TILE_SIZE, 0:free_size
                                    ],
                                )
                else:
                    # Per-channel quant or non-quant: standard nc_matmul + psum auto-accumulation.
                    # block_hidden_states_T is list[h_outer][h_inner] of 3D tiles for both paths.
                    # For per-channel FP8 quant we upcast the FP8 weights to bf16 first (matching mainline);
                    # the alternate FP8xFP8 + perf_mode="double_row" path produces an MLIR
                    # partition/free-product mismatch that silently miscompiles in the parser frontend.
                    for h_inner_idx in range(h_inner_tripcount):
                        # Skip H tiles past the actual tensor extent (when H is not a multiple of PSUM_SIZE).
                        if h_outer_idx * h_inner_tripcount + h_inner_idx >= linearized_h_tripcount:
                            continue
                        for b_psum_idx in range(N_PSUM_TILE):
                            if cfg.is_quant:
                                # Mainline-style: upcast FP8 weights to bf16 and do bf16xbf16 matmul.
                                # Per mainline 21106510, the upcast dst must be 2D (matches the 2D source after
                                # scalar gate_or_up indexing); the 3D (TILE_SIZE, 1, num_i_tile) form caused
                                # all-zero output on some configs.
                                if cfg.fuse_gate_and_up_load:
                                    gup_w_upcasted = nl.ndarray(
                                        (TILE_SIZE, num_i_tile), dtype=nl.bfloat16, buffer=nl.sbuf
                                    )
                                    nisa.tensor_copy(
                                        dst=gup_w_upcasted,
                                        src=gup_weights[h_outer_idx][
                                            0:TILE_SIZE, h_inner_idx, gate_or_up, i_start : i_start + num_i_tile
                                        ],
                                    )
                                else:
                                    gup_w_upcasted = nl.ndarray(
                                        (TILE_SIZE, num_i_tile), dtype=nl.bfloat16, buffer=nl.sbuf
                                    )
                                    nisa.tensor_copy(
                                        dst=gup_w_upcasted,
                                        src=gup_weights[h_outer_idx][
                                            0:TILE_SIZE, h_inner_idx, 0, i_start : i_start + num_i_tile
                                        ],
                                    )
                                nisa.nc_matmul(
                                    dst=gate_or_up_psum_lst[b_psum_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                                    stationary=gup_w_upcasted,
                                    moving=block_hidden_states_T[h_outer_idx][h_inner_idx][
                                        0:TILE_SIZE, b_psum_idx, 0:free_size
                                    ],
                                )
                            else:
                                if cfg.fuse_gate_and_up_load:
                                    nisa.nc_matmul(
                                        dst=gate_or_up_psum_lst[b_psum_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                                        stationary=gup_weights[h_outer_idx][
                                            0:TILE_SIZE,
                                            h_inner_idx,
                                            gate_or_up,
                                            i_start : i_start + num_i_tile,
                                        ],
                                        moving=block_hidden_states_T[h_outer_idx][h_inner_idx][
                                            0:TILE_SIZE, b_psum_idx, 0:free_size
                                        ],
                                    )
                                else:
                                    nisa.nc_matmul(
                                        dst=gate_or_up_psum_lst[b_psum_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                                        stationary=gup_weights[h_outer_idx][
                                            0:TILE_SIZE,
                                            h_inner_idx,
                                            0,
                                            i_start : i_start + num_i_tile,
                                        ],
                                        moving=block_hidden_states_T[h_outer_idx][h_inner_idx][
                                            0:TILE_SIZE, b_psum_idx, 0:free_size
                                        ],
                                    )

        for psum_tile_idx in range(N_PSUM_TILE):
            for i_tile_idx in range(GUP_N_TILES):
                # Step 1: Apply weight scale (move from psum to sbuf)
                # For block quant, result is already scaled and in sbuf — skip
                if gup_block_scale != None:
                    pass  # already in sbuf with block scale applied
                elif gup_scale != None:
                    nisa.tensor_scalar(
                        data=gate_or_up_psum_lst[psum_tile_idx][i_tile_idx][0:TILE_SIZE, 0:free_size],
                        op0=nl.multiply,
                        operand0=gup_scale[i_tile_idx][gate_or_up],
                        dst=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:TILE_SIZE, 0:free_size
                        ],
                    )
                else:
                    num_i_tile = min(TILE_SIZE, dims.I_TP_sharded - TILE_SIZE * i_tile_idx)
                    nisa.tensor_copy(
                        dst=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:num_i_tile, 0:free_size
                        ],
                        src=gate_or_up_psum_lst[psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                    )

                # Step 2: Apply per-token hidden scale (post-dequant, before bias)
                if gate_up_hidden_scale != None:
                    nisa.tensor_tensor(
                        data1=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:TILE_SIZE, 0:free_size
                        ],
                        data2=gate_up_hidden_scale[psum_tile_idx][0:TILE_SIZE, 0:free_size],
                        op=nl.multiply,
                        dst=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:TILE_SIZE, 0:free_size
                        ],
                    )

                # Step 3: Apply bias
                if cfg.linear_bias:
                    nisa.tensor_tensor(
                        data1=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:TILE_SIZE, 0:free_size
                        ],
                        data2=gate_up_bias_T.ap(
                            pattern=[[gate_up_bias_T.shape[1], TILE_SIZE], [0, free_size]],
                            offset=i_tile_idx * 2 + gate_or_up,
                        ),
                        op=nl.add,
                        dst=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:TILE_SIZE, 0:free_size
                        ],
                    )

    # Clipping section
    if (
        cfg.gate_clamp_upper_limit != None
        or cfg.gate_clamp_lower_limit != None
        or cfg.up_clamp_lower_limit != None
        or cfg.up_clamp_upper_limit != None
    ):
        for psum_tile_idx in range(N_PSUM_TILE):
            for i_tile_idx in range(GUP_N_TILES):
                num_i_tile = min(TILE_SIZE, dims.I_TP_sharded - TILE_SIZE * i_tile_idx)
                if not cfg.skip_gate_proj and cfg.gate_clamp_lower_limit != None and cfg.gate_clamp_upper_limit != None:
                    nisa.tensor_scalar(
                        data=gate_and_up_proj_res_sbuf_lst[0][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                        op0=nl.minimum,
                        operand0=cfg.gate_clamp_upper_limit,
                        op1=nl.maximum,
                        operand1=cfg.gate_clamp_lower_limit,
                        dst=gate_and_up_proj_res_sbuf_lst[0][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                    )
                elif not cfg.skip_gate_proj:
                    if cfg.gate_clamp_upper_limit != None:
                        nisa.tensor_scalar(
                            data=gate_and_up_proj_res_sbuf_lst[0][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                            op0=nl.minimum,
                            operand0=cfg.gate_clamp_upper_limit,
                            dst=gate_and_up_proj_res_sbuf_lst[0][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                        )
                    if cfg.gate_clamp_lower_limit != None:
                        nisa.tensor_scalar(
                            data=gate_and_up_proj_res_sbuf_lst[0][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                            op0=nl.maximum,
                            operand0=cfg.gate_clamp_lower_limit,
                            dst=gate_and_up_proj_res_sbuf_lst[0][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                        )

                if cfg.up_clamp_upper_limit != None and cfg.up_clamp_lower_limit != None:
                    nisa.tensor_scalar(
                        data=gate_and_up_proj_res_sbuf_lst[1][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                        op0=nl.minimum,
                        operand0=cfg.up_clamp_upper_limit,
                        op1=nl.maximum,
                        operand1=cfg.up_clamp_lower_limit,
                        dst=gate_and_up_proj_res_sbuf_lst[1][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                    )
                else:
                    if cfg.up_clamp_upper_limit != None:
                        nisa.tensor_scalar(
                            data=gate_and_up_proj_res_sbuf_lst[1][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                            op0=nl.minimum,
                            operand0=cfg.up_clamp_upper_limit,
                            dst=gate_and_up_proj_res_sbuf_lst[1][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                        )
                    if cfg.up_clamp_lower_limit != None:
                        nisa.tensor_scalar(
                            data=gate_and_up_proj_res_sbuf_lst[1][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                            op0=nl.maximum,
                            operand0=cfg.up_clamp_lower_limit,
                            dst=gate_and_up_proj_res_sbuf_lst[1][psum_tile_idx][i_tile_idx][0:num_i_tile, 0:free_size],
                        )

    if gate_up_activations_T != None:
        # gate_up_activations_T shape: (N, 2, I_TP_total, B)
        activation_I_TP = gate_up_activations_T.shape[2]
        activation_B = gate_up_activations_T.shape[3]

        for psum_tile_idx in range(N_PSUM_TILE):
            for i_tile_idx in range(GUP_N_TILES):
                num_i_tile = min(TILE_SIZE, dims.I_TP_sharded - TILE_SIZE * i_tile_idx)

                for gate_or_up in range(2):
                    if cfg.skip_gate_proj and gate_or_up == 0:
                        continue
                    offset = (
                        block_idx * (2 * activation_I_TP * activation_B)
                        + gate_or_up * (activation_I_TP * activation_B)
                        + (dims.I_TP_sharded * shard_id + i_tile_idx * TILE_SIZE) * activation_B
                        + activation_block_write_offset
                        + psum_tile_idx * free_size
                    )

                    nisa.dma_copy(
                        dst=gate_up_activations_T.ap(
                            pattern=[[activation_B, num_i_tile], [1, free_size]], offset=offset
                        ),
                        src=gate_and_up_proj_res_sbuf_lst[gate_or_up][psum_tile_idx][i_tile_idx][
                            0:num_i_tile, 0:free_size
                        ],
                    )

    return gate_and_up_proj_res_sbuf_lst


def load_token_indices_dynamic_block(
    token_position_to_id, block_idx, dims: DimensionSizes, skip_dma: SkipMode = SkipMode()
):
    """
    Load token indices for the current block (dynamic version).

    Args:
        token_position_to_id: Token position to ID mapping tensor.
        block_idx: Current block index (dynamic).
        dims: DimensionSizes object containing B and NUM_B_TILES.
        skip_dma: Skip DMA mode.

    Returns:
        local_token_indices: Tensor of shape (TILE_SIZE, NUM_B_TILES) containing token indices.
    """
    local_token_indices = nl.ndarray((TILE_SIZE, dims.NUM_B_TILES), dtype=token_position_to_id.dtype, buffer=nl.sbuf)
    total_size = int(token_position_to_id.shape[0])
    reshaped_token_position_to_id = token_position_to_id.reshape((total_size // dims.B, dims.B))

    for b_tile_idx in range(dims.NUM_B_TILES):
        nisa.dma_copy(
            dst=local_token_indices[0:TILE_SIZE, b_tile_idx],
            src=reshaped_token_position_to_id.ap(
                pattern=[[1, TILE_SIZE], [1, 1]], offset=TILE_SIZE * b_tile_idx, scalar_offset=block_idx, indirect_dim=0
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )

    return local_token_indices


def load_gate_up_proj_weights_shard_intermediate(
    gate_or_up, gate_up_proj_weight, block_expert, cfg: Configs, num_shards, shard_id, load_dst=None
):
    """
    Return nested list structure equivalent to [h_outer_tripcount, h_inner_tripcount, TILE_SIZE, N_WEIGHTS, I_TP_sharded]
    """

    if cfg.fuse_gate_and_up_load:
        kernel_assert(gate_or_up == 0, "gate_or_up must be 0 when fuse_gate_and_up_load is True")
        N_WEIGHTS = 2
    else:
        N_WEIGHTS = 1

    _, H, _, _I_TP = gate_up_proj_weight.shape
    I_TP_per_shard = _I_TP // num_shards
    I_TP_offset = I_TP_per_shard * shard_id
    h_outer_tripcount = div_ceil(H, PSUM_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE

    if load_dst == None:
        load_dst = []
        for h_i in range(h_outer_tripcount):
            # Store all h_inner tiles contiguously for .ap() double_row views
            load_dst.append(
                nl.ndarray(
                    (TILE_SIZE, h_inner_tripcount, N_WEIGHTS, I_TP_per_shard), dtype=cfg.weight_dtype, buffer=nl.sbuf
                )
            )

    for h_i in range(h_outer_tripcount):
        for h_j in range(h_inner_tripcount):
            load_p_offset = PSUM_SIZE * h_i + TILE_SIZE * h_j
            if load_p_offset >= H:
                nisa.memset(dst=load_dst[h_i][0:TILE_SIZE, h_j, 0:N_WEIGHTS, 0:I_TP_per_shard], value=0.0)
                continue
            num_p_elems = min(TILE_SIZE, H - load_p_offset)

            if num_p_elems < TILE_SIZE:
                nisa.memset(dst=load_dst[h_i][0:TILE_SIZE, h_j, 0:N_WEIGHTS, 0:I_TP_per_shard], value=0.0)

            if cfg.fuse_gate_and_up_load:
                offset = load_p_offset * (N_WEIGHTS * _I_TP) + I_TP_offset
                nisa.dma_copy(
                    dst=load_dst[h_i][0:num_p_elems, h_j, 0:N_WEIGHTS, 0:I_TP_per_shard],
                    src=gate_up_proj_weight.ap(
                        pattern=[[N_WEIGHTS * _I_TP, num_p_elems], [_I_TP, N_WEIGHTS], [1, I_TP_per_shard]],
                        offset=offset,
                        scalar_offset=block_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if cfg.skip_dma.skip_weight else oob_mode.error,
                )
            else:
                offset = load_p_offset * (2 * _I_TP) + gate_or_up * _I_TP + I_TP_offset
                nisa.dma_copy(
                    dst=load_dst[h_i][0:num_p_elems, h_j, 0:N_WEIGHTS, 0:I_TP_per_shard],
                    src=gate_up_proj_weight.ap(
                        pattern=[[2 * _I_TP, num_p_elems], [_I_TP, N_WEIGHTS], [1, I_TP_per_shard]],
                        offset=offset,
                        scalar_offset=block_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if cfg.skip_dma.skip_weight else oob_mode.error,
                )

    return load_dst


def calculate_expert_affinity_T(
    inps: InputTensors, dims: DimensionSizes, cfg: Configs, block_expert, token_indices, token_indices_offset=0
):
    """Calculate expert affinity transposed and broadcasted.

    Args:
        inps: InputTensors object.
        dims: DimensionSizes object.
        cfg: Configs object.
        block_expert: Expert index tensor.
        token_indices: Token indices tensor.
        token_indices_offset: Offset for token indices.

    Returns:
        expert_affinity_T_broadcasted: Tensor of shape (TILE_SIZE, dims.B) containing broadcasted expert affinities.
    """
    compatible_psum_dtype = compatible_dtype(cfg.compute_dtype)
    expert_affinity_T_broadcasted = nl.ndarray((TILE_SIZE, dims.B), dtype=compatible_psum_dtype, buffer=nl.sbuf)

    v_expert = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    shuffle_mask = [0] * 32
    for channel_bank_idx in range(4):
        nisa.nc_stream_shuffle(
            dst=v_expert[
                DVE_CHANNELS_PER_BANK * channel_bank_idx : DVE_CHANNELS_PER_BANK * (channel_bank_idx + 1), 0:1
            ],
            src=block_expert.ap(
                pattern=[
                    [1, 1],
                    [1, 1],
                ],
                offset=0,
            ),
            shuffle_mask=shuffle_mask,
        )

    expert_affinity_T = nl.ndarray((1, dims.B), dtype=compatible_psum_dtype, buffer=nl.psum)

    expert_affinity_lst = []
    for b_tile_idx in range(dims.NUM_B_TILES):
        expert_affinity_lst.append(nl.ndarray((TILE_SIZE, 1), dtype=cfg.compute_dtype, buffer=nl.sbuf))

    for b_tile_idx in range(dims.NUM_B_TILES):
        addr = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=addr,
            data=token_indices[0:TILE_SIZE, token_indices_offset + b_tile_idx],
            op0=nl.multiply,
            operand0=dims.E,
        )
        addr_fin = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=addr_fin, data1=addr, op=nl.add, data2=v_expert)

        if cfg.skip_dma.skip_token:
            nisa.tensor_scalar(dst=addr_fin, data=addr_fin, op0=nl.maximum, operand0=-1)

        if cfg.skip_dma.skip_token:
            nisa.memset(dst=expert_affinity_lst[b_tile_idx].ap(pattern=[[1, TILE_SIZE], [1, 1]], offset=0), value=0)

        """
        Access: expert_affinities_masked[addr_fin[i], 0] for i in 0:TILE_SIZE
        This is vector indirect access where addr_fin contains the row indices.
        Pattern: [[num_cols, TILE_SIZE], [1, 1]]
        vector_offset: addr_fin (reshaped to match)
        indirect_dim: 0
        """

        num_cols = inps.expert_affinities_masked.shape[1]
        addr_fin_reshaped = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=addr_fin_reshaped, src=addr_fin[0:TILE_SIZE, 0:1])

        nisa.dma_copy(
            dst=expert_affinity_lst[b_tile_idx][0:TILE_SIZE, 0:1],
            src=inps.expert_affinities_masked.ap(
                pattern=[[num_cols, TILE_SIZE], [1, 1]], offset=0, vector_offset=addr_fin_reshaped, indirect_dim=0
            ),
            oob_mode=oob_mode.skip if cfg.skip_dma.skip_token else oob_mode.error,
        )

        num_f = min(TILE_SIZE, dims.B - (b_tile_idx * TILE_SIZE))

        nisa.nc_transpose(
            data=expert_affinity_lst[b_tile_idx][0:num_f, 0:1],
            dst=expert_affinity_T[0:1, b_tile_idx * TILE_SIZE : b_tile_idx * TILE_SIZE + num_f],
        )

    # broadcast
    for channel_bank_idx in range(4):
        nisa.nc_stream_shuffle(
            dst=expert_affinity_T_broadcasted[
                channel_bank_idx * DVE_CHANNELS_PER_BANK : (channel_bank_idx + 1) * DVE_CHANNELS_PER_BANK, 0 : dims.B
            ],
            src=expert_affinity_T.ap(pattern=[[dims.B, 1], [1, dims.B]], offset=0),
            shuffle_mask=shuffle_mask,
        )

    return expert_affinity_T_broadcasted


def load_old_block(
    output, token_indices, NUM_TILES, dtype, skip_dma: SkipMode = SkipMode(), shard_id=None, token_indices_offset=0
):
    """Loads the partially computed output hidden states for the current block's token indices.

    Args:
        output: Output tensor.
        token_indices: Token indices tensor.
        NUM_TILES: Number of tiles.
        dtype: Data type.
        skip_dma: Skip DMA mode.
        shard_id: Shard ID.
        token_indices_offset: Offset for token indices.

    Returns:
        block_old_lst: List of tensors, each of shape (TILE_SIZE, H).
    """
    H = output.shape[-1]

    block_old_lst = []
    for tile_idx in range(NUM_TILES):
        block_old_lst.append(nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf))

    for tile_idx in range(NUM_TILES):
        if skip_dma.skip_token:
            nisa.memset(value=0, dst=block_old_lst[tile_idx][0:TILE_SIZE, 0:H])

        block_token_mapping = nl.ndarray((TILE_SIZE, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=block_token_mapping,
            src=token_indices[0:TILE_SIZE, token_indices_offset + tile_idx : token_indices_offset + tile_idx + 1],
        )

        if shard_id != None:
            """
            output shape: (num_shards, num_tokens, H)
            Pattern: [[H, TILE_SIZE], [1, H]]
            - First dimension (indirect): TILE_SIZE iterations with stride H (row stride)
            - Second dimension: H iterations with stride 1 (within row)
            Offset: shard_id * num_tokens * H (to access the correct shard)
            vector_offset: block_token_mapping (shape TILE_SIZE, 1)
            indirect_dim: 0 (we're indirecting on the first dimension of the pattern)
            """
            num_tokens = output.shape[1]

            nisa.dma_copy(
                dst=block_old_lst[tile_idx][0:TILE_SIZE, 0:H],
                src=output.ap(
                    pattern=[[H, TILE_SIZE], [1, H]],
                    offset=shard_id * num_tokens * H,
                    vector_offset=block_token_mapping,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
        else:
            """
            output shape: (num_tokens, H)
            Pattern: [[H, TILE_SIZE], [1, H]]
            Offset: 0
            vector_offset: block_token_mapping
            indirect_dim: 0
            """

            nisa.dma_copy(
                dst=block_old_lst[tile_idx][0:TILE_SIZE, 0:H],
                src=output.ap(
                    pattern=[[H, TILE_SIZE], [1, H]], offset=0, vector_offset=block_token_mapping, indirect_dim=0
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

    return block_old_lst


def compute_down_proj_shard_on_intermediate(
    intermediate_states,
    inps: InputTensors,
    dims: DimensionSizes,
    cfg: Configs,
    block_expert,
    expert_affinity,
    block_old,
    shard_id,
    allocate=False,
    down_scale=None,
    outs: OutputTensors = None,
    block_idx=0,
    block_tile_index=0,
    down_hidden_scale=None,
):
    """Compute the new block output with down projection and expert affinity adjustment.

    Args:
        intermediate_states: List of intermediate state tensors.
        inps: InputTensors object.
        dims: DimensionSizes object.
        cfg: Configs object.
        block_expert: Expert index tensor.
        expert_affinity: Expert affinity list.
        block_old: Previously computed block output.
        shard_id: Current shard ID.
        allocate: Whether to allocate new tensors.
        down_scale: Optional quantization scale.
        outs: OutputTensors object.
        block_idx: Block index.
        block_tile_index: Block tile index.
        down_hidden_scale: Optional list of (TILE_SIZE, 1) per-token scales for down projection post-dequant.

    Returns:
        block_new_lnc_recv_sbuf_lst: List of tensors, each of shape (TILE_SIZE, dims.H).
    """

    # Partials/bias precision follow the accumulator (None = io_dtype = default, no upcast).
    _acc_dtype = cfg.accumulation_dtype if cfg.accumulation_dtype is not None else cfg.io_dtype
    if _acc_dtype != cfg.io_dtype:
        _acc_upcast = True
        _bn_dtype = _acc_dtype
        _bias_dtype = _acc_dtype
    else:
        _acc_upcast = False
        _bn_dtype = cfg.io_dtype
        _bias_dtype = cfg.compute_dtype

    if cfg.linear_bias:
        down_bias = nl.ndarray((1, dims.H), dtype=_bias_dtype, buffer=nl.sbuf)

        nisa.dma_copy(
            dst=down_bias[0:1, 0 : dims.H],
            src=inps.down_proj_bias.ap(
                pattern=[
                    [dims.H, 1],  # Dim 0: stride=dims.H, count=1 row
                    [1, dims.H],  # Dim 1: stride=1, count=dims.H (contiguous)
                ],
                offset=0,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.error,
        )
        down_bias_broadcasted = nl.ndarray((TILE_SIZE, dims.H), dtype=_bias_dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(down_bias, down_bias_broadcasted)

    block_new_lst = []
    for b_tile_idx in range(dims.NUM_B_TILES):
        block_new_lst.append(nl.ndarray((TILE_SIZE, dims.H), dtype=_bn_dtype, buffer=nl.sbuf))

    block_new_lnc_recv_sbuf_lst = []
    for b_tile_idx in range(dims.NUM_B_TILES_SHARDED):
        block_new_lnc_recv_sbuf_lst.append(nl.ndarray((TILE_SIZE, dims.H), dtype=_bn_dtype, buffer=nl.sbuf))

    GUP_N_TILES = div_ceil(dims.I_TP_sharded, TILE_SIZE)
    H_tile_size = min(1024, dims.H)
    h_i_upper = div_ceil(dims.H, H_tile_size)

    dp_load_dst_lst = []
    for i_tile_idx in range(GUP_N_TILES):
        dp_load_dst_lst.append(nl.ndarray((TILE_SIZE, H_tile_size), dtype=inps.down_proj_weight.dtype, buffer=nl.sbuf))

    # Use flattened quant config fields for compiler frontend compatibility
    quant_is_per_tensor = cfg.quant_is_per_tensor
    quant_activation_mode = cfg.quant_activation_mode

    for H_tile1024_idx in nl.sequential_range(h_i_upper):
        actual_H_tile_size = min(H_tile_size, dims.H - H_tile_size * H_tile1024_idx)
        num_h_tiles = div_ceil(actual_H_tile_size, PSUM_SIZE)
        if cfg.is_quant:
            if cfg.is_block_quant:
                # Block quant: scale shape [E, I_TP//256, H//256, TILE_SIZE] — pre-broadcasted
                I_blocks = dims.I_TP_sharded // BLOCK_QUANT_SIZE
                I_blocks_total = dims.I_TP // BLOCK_QUANT_SIZE
                h_block_offset = H_tile1024_idx * H_tile_size // BLOCK_QUANT_SIZE
                num_h_256_blocks = div_ceil(actual_H_tile_size, BLOCK_QUANT_SIZE)
                num_dp_scales = I_blocks * num_h_256_blocks
                dp_block_scale = nl.ndarray((TILE_SIZE, num_dp_scales), dtype=nl.float32, buffer=nl.sbuf)
                dp_scale_reshaped = inps.down_proj_scale.reshape(
                    (dims.E, I_blocks_total * (dims.H // BLOCK_QUANT_SIZE) * TILE_SIZE)
                )
                i_block_offset = shard_id * I_blocks
                for i_blk in range(I_blocks):
                    for h_blk in range(num_h_256_blocks):
                        flat_idx = i_blk * num_h_256_blocks + h_blk
                        src_offset = (
                            (i_block_offset + i_blk) * (dims.H // BLOCK_QUANT_SIZE) + h_block_offset + h_blk
                        ) * TILE_SIZE
                        nisa.dma_copy(
                            dst=dp_block_scale[0:TILE_SIZE, flat_idx : flat_idx + 1],
                            src=dp_scale_reshaped.ap(
                                pattern=[[1, TILE_SIZE], [TILE_SIZE, 1]],
                                offset=src_offset,
                                scalar_offset=block_expert,
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.error,
                        )
                down_scale = None
                down_scale_pt = None
            elif quant_is_per_tensor:
                # Per-tensor: pre-combine weight + activation scales into (TILE_SIZE, 1)
                # Applied during psum→sbuf copy via nisa.activation, no 3D tensor needed
                pt_w = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=pt_w[0:1, 0:1],
                    src=inps.down_proj_scale.ap(
                        pattern=[[1, 1], [1, 1]],
                        offset=0,
                        scalar_offset=block_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.error,
                )
                down_scale_pt = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
                stream_shuffle_broadcast(pt_w, down_scale_pt)
                if down_hidden_scale != None:
                    # Per-token activation: fold per-token scale into weight scale
                    # down_hidden_scale is per-token so we can't fold here — handle per B_tile below
                    pass
                elif quant_activation_mode == ActivationQuantMode.PER_TENSOR and inps.down_hidden_scale != None:
                    # Per-tensor activation: load and fold into combined scale
                    pt_act = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=pt_act[0:1, 0:1],
                        src=inps.down_hidden_scale.ap(
                            pattern=[[1, 1], [1, 1]],
                            offset=0,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.error,
                    )
                    pt_act_p = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    stream_shuffle_broadcast(pt_act, pt_act_p)
                    nisa.tensor_tensor(
                        data1=down_scale_pt,
                        data2=pt_act_p,
                        op=nl.multiply,
                        dst=down_scale_pt,
                    )
                down_scale = None
                dp_block_scale = None
            else:
                # Per-channel: [E, 1, H]
                down_scale = nl.ndarray((TILE_SIZE, num_h_tiles, PSUM_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                down_scale_pt = None
                for h_tile_idx in range(num_h_tiles):
                    num_psum_elems = min(PSUM_SIZE, dims.H - (PSUM_SIZE * h_tile_idx + H_tile_size * H_tile1024_idx))

                    elem_offset = PSUM_SIZE * h_tile_idx + H_tile_size * H_tile1024_idx

                    nisa.dma_copy(
                        dst=down_scale[0:1, h_tile_idx, 0:num_psum_elems],
                        src=inps.down_proj_scale.ap(
                            pattern=[[dims.H, 1], [dims.H, 1], [1, num_psum_elems]],
                            offset=elem_offset,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                    )
                    for channel_bank_idx in range(4):
                        nisa.nc_stream_shuffle(
                            src=down_scale[0:1, h_tile_idx, :],
                            dst=down_scale[
                                nl.ds(DVE_CHANNELS_PER_BANK * channel_bank_idx, DVE_CHANNELS_PER_BANK), h_tile_idx, :
                            ],
                            shuffle_mask=[0] * DVE_CHANNELS_PER_BANK,
                        )
                dp_block_scale = None

        dp_weights = load_down_proj_weight_shard_intermediate_H_tile(
            inps.down_proj_weight,
            block_expert,
            cfg,
            dims.NUM_SHARDS,
            shard_id,
            H_tile_size,
            H_tile1024_idx,
            dp_load_dst=dp_load_dst_lst,
        )

        down_proj_lst = []
        for h_idx in range(num_h_tiles):
            down_proj_h_lst = []
            for b_idx in range(dims.NUM_B_TILES):
                down_proj_h_lst.append(nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=nl.float32, buffer=nl.psum))
            down_proj_lst.append(down_proj_h_lst)

        for B_tile_idx in range(dims.NUM_B_TILES):
            for h_j in range(num_h_tiles):
                num_h_elems = min(PSUM_SIZE, dims.H - (H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j))
                if cfg.is_quant and cfg.is_block_quant:
                    # Block quant 256x256: accumulate 2 i_tiles per block in psum, scale, sbuf accumulate
                    i_tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE  # 2
                    num_h_256 = div_ceil(num_h_elems, BLOCK_QUANT_SIZE)
                    for i_block in range(GUP_N_TILES // i_tiles_per_block):
                        # Accumulate 2 i_tiles into psum
                        tmp_dp_psum = nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=nl.float32, buffer=nl.psum)
                        for i_sub in range(i_tiles_per_block):
                            i_i = i_block * i_tiles_per_block + i_sub
                            num_i_elems = min(TILE_SIZE, dims.I_TP_sharded - TILE_SIZE * i_i)
                            nisa.nc_matmul(
                                dst=tmp_dp_psum[0:TILE_SIZE, 0:num_h_elems],
                                stationary=intermediate_states[i_i][
                                    0:num_i_elems, TILE_SIZE * B_tile_idx : TILE_SIZE * B_tile_idx + TILE_SIZE
                                ],
                                moving=dp_weights[i_i].ap(
                                    pattern=[[H_tile_size, num_i_elems], [1, num_h_elems]], offset=PSUM_SIZE * h_j
                                ),
                            )
                        # Scale each 256-wide H chunk and accumulate into sbuf
                        for h_256_idx in range(num_h_256):
                            h_256_local = h_256_idx * BLOCK_QUANT_SIZE
                            h_256_count = min(BLOCK_QUANT_SIZE, num_h_elems - h_256_local)
                            h_256_within_H1024 = h_j * (PSUM_SIZE // BLOCK_QUANT_SIZE) + h_256_idx
                            flat_dp_idx = i_block * num_h_256_blocks + h_256_within_H1024
                            dst_h_start = H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j + h_256_local
                            if i_block == 0:
                                nisa.tensor_scalar(
                                    data=tmp_dp_psum[0:TILE_SIZE, h_256_local : h_256_local + h_256_count],
                                    op0=nl.multiply,
                                    operand0=dp_block_scale[0:TILE_SIZE, flat_dp_idx : flat_dp_idx + 1],
                                    dst=block_new_lst[B_tile_idx][0:TILE_SIZE, dst_h_start : dst_h_start + h_256_count],
                                )
                            else:
                                nisa.scalar_tensor_tensor(
                                    data=tmp_dp_psum[0:TILE_SIZE, h_256_local : h_256_local + h_256_count],
                                    op0=nl.multiply,
                                    operand0=dp_block_scale[0:TILE_SIZE, flat_dp_idx : flat_dp_idx + 1],
                                    op1=nl.add,
                                    operand1=block_new_lst[B_tile_idx][
                                        0:TILE_SIZE, dst_h_start : dst_h_start + h_256_count
                                    ],
                                    dst=block_new_lst[B_tile_idx][0:TILE_SIZE, dst_h_start : dst_h_start + h_256_count],
                                )
                    # Apply hidden scale if present (on partition dim, broadcast across H)
                    if down_hidden_scale != None:
                        dst_h_start_full = H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j
                        nisa.tensor_scalar(
                            data=block_new_lst[B_tile_idx][
                                0:TILE_SIZE, dst_h_start_full : dst_h_start_full + num_h_elems
                            ],
                            op0=nl.multiply,
                            operand0=down_hidden_scale[B_tile_idx][0:TILE_SIZE, 0:1],
                            dst=block_new_lst[B_tile_idx][
                                0:TILE_SIZE, dst_h_start_full : dst_h_start_full + num_h_elems
                            ],
                        )
                else:
                    # Per-channel quant or non-quant: auto-accumulate in psum across I tiles
                    for i_i in range(GUP_N_TILES):
                        num_i_elems = min(TILE_SIZE, dims.I_TP_sharded - TILE_SIZE * i_i)
                        if cfg.is_quant:
                            dp_weights_upcasted = nl.ndarray(
                                (num_i_elems, num_h_elems), dtype=nl.bfloat16, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(
                                dst=dp_weights_upcasted,
                                src=dp_weights[i_i].ap(
                                    pattern=[[H_tile_size, num_i_elems], [1, num_h_elems]], offset=PSUM_SIZE * h_j
                                ),
                            )
                            nisa.nc_matmul(
                                dst=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                                stationary=intermediate_states[i_i][
                                    0:num_i_elems, TILE_SIZE * B_tile_idx : TILE_SIZE * B_tile_idx + TILE_SIZE
                                ],
                                moving=dp_weights_upcasted,
                            )
                        else:
                            nisa.nc_matmul(
                                dst=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                                stationary=intermediate_states[i_i][
                                    0:num_i_elems, TILE_SIZE * B_tile_idx : TILE_SIZE * B_tile_idx + TILE_SIZE
                                ],
                                moving=dp_weights[i_i].ap(
                                    pattern=[[H_tile_size, num_i_elems], [1, num_h_elems]], offset=PSUM_SIZE * h_j
                                ),
                            )

                    if cfg.is_quant:
                        if quant_is_per_tensor:
                            # Per-tensor: apply combined scale during psum→sbuf copy
                            if down_hidden_scale != None:
                                # Per-token activation with per-tensor weight: combine both (TILE_SIZE,1) scales
                                combined_pt = nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
                                nisa.tensor_tensor(
                                    data1=down_scale_pt,
                                    data2=down_hidden_scale[B_tile_idx][0:TILE_SIZE, 0:1],
                                    op=nl.multiply,
                                    dst=combined_pt,
                                )
                                nisa.tensor_scalar(
                                    data=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                                    op0=nl.multiply,
                                    operand0=combined_pt,
                                    dst=block_new_lst[B_tile_idx][
                                        0:TILE_SIZE,
                                        H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j : H_tile_size * H_tile1024_idx
                                        + PSUM_SIZE * h_j
                                        + num_h_elems,
                                    ],
                                )
                            else:
                                nisa.tensor_scalar(
                                    data=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                                    op0=nl.multiply,
                                    operand0=down_scale_pt,
                                    dst=block_new_lst[B_tile_idx][
                                        0:TILE_SIZE,
                                        H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j : H_tile_size * H_tile1024_idx
                                        + PSUM_SIZE * h_j
                                        + num_h_elems,
                                    ],
                                )
                        elif down_hidden_scale != None:
                            nisa.scalar_tensor_tensor(
                                data=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                                op0=nl.multiply,
                                operand0=down_hidden_scale[B_tile_idx][0:TILE_SIZE, 0:1],
                                op1=nl.multiply,
                                operand1=down_scale[0:TILE_SIZE, h_j, 0:num_h_elems],
                                dst=block_new_lst[B_tile_idx][
                                    0:TILE_SIZE,
                                    H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j : H_tile_size * H_tile1024_idx
                                    + PSUM_SIZE * h_j
                                    + num_h_elems,
                                ],
                            )
                        else:
                            nisa.tensor_tensor(
                                dst=block_new_lst[B_tile_idx][
                                    0:TILE_SIZE,
                                    H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j : H_tile_size * H_tile1024_idx
                                    + PSUM_SIZE * h_j
                                    + num_h_elems,
                                ],
                                data1=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                                data2=down_scale[0:TILE_SIZE, h_j, 0:num_h_elems],
                                op=nl.multiply,
                            )
                    else:
                        nisa.tensor_copy(
                            src=down_proj_lst[h_j][B_tile_idx][0:TILE_SIZE, 0:num_h_elems],
                            engine=nisa.scalar_engine,
                            dst=block_new_lst[B_tile_idx][
                                0:TILE_SIZE,
                                H_tile_size * H_tile1024_idx + PSUM_SIZE * h_j : H_tile_size * H_tile1024_idx
                                + PSUM_SIZE * h_j
                                + num_h_elems,
                            ],
                        )

    N_B_TILES_OFFSET = dims.NUM_B_TILES_SHARDED * shard_id

    for b_shard_tile_idx in range(dims.NUM_B_TILES_SHARDED):
        if _acc_upcast:
            # Send the peer partial directly in the accumulator dtype and reduce in
            # accumulator precision (avoids a separate io-dtype downcast before the exchange).
            block_new_acc_recv = nl.ndarray((TILE_SIZE, dims.H), dtype=_acc_dtype, buffer=nl.sbuf)
            sendrecv(
                src=block_new_lst[b_shard_tile_idx + dims.NUM_B_TILES_SHARDED * (1 - shard_id)][
                    0:TILE_SIZE, 0 : dims.H
                ],
                dst=block_new_acc_recv[0:TILE_SIZE, 0 : dims.H],
                send_to_rank=(1 - shard_id),
                recv_from_rank=(1 - shard_id),
                pipe_id=0,
            )

            # reduce in accumulator precision
            nisa.tensor_tensor(
                data1=block_new_lst[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0 : dims.H],
                data2=block_new_acc_recv[0:TILE_SIZE, 0 : dims.H],
                op=nl.add,
                dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
            )
        else:
            # Default path: exchange and reduce in io dtype.
            sendrecv(
                src=block_new_lst[b_shard_tile_idx + dims.NUM_B_TILES_SHARDED * (1 - shard_id)][
                    0:TILE_SIZE, 0 : dims.H
                ],
                dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                send_to_rank=(1 - shard_id),
                recv_from_rank=(1 - shard_id),
                pipe_id=0,
            )

            nisa.tensor_tensor(
                data1=block_new_lst[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0 : dims.H],
                data2=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                op=nl.add,
                dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
            )
        if outs and outs.down_activations and not cfg.expert_affinity_multiply_on_I:
            """
            outs.down_activations shape: (N_blocks, total_tokens, H)
            Access: [block_idx, token_offset + b_shard_tile_idx*TILE_SIZE + i, j]
            where i in 0:TILE_SIZE, j in 0:dims.H
            Pattern: [[dims.H, TILE_SIZE], [1, dims.H]]
            Offset: block_idx * (total_tokens * dims.H) + token_offset * dims.H
            """

            total_tokens = outs.down_activations.shape[1]
            token_offset = (
                (block_tile_index * MAX_BLOCK_TILE_SIZE)
                + (N_B_TILES_OFFSET * TILE_SIZE)
                + (b_shard_tile_idx * TILE_SIZE)
            )

            offset = block_idx * (total_tokens * dims.H) + token_offset * dims.H

            nisa.dma_copy(
                dst=outs.down_activations.ap(pattern=[[dims.H, TILE_SIZE], [1, dims.H]], offset=offset),
                src=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                oob_mode=oob_mode.skip if cfg.skip_dma.skip_token else oob_mode.error,
            )

    if cfg.expert_affinity_multiply_on_I:
        for b_shard_tile_idx in range(dims.NUM_B_TILES_SHARDED):
            if cfg.linear_bias and expert_affinity != None:
                # bias must be scaled by affinity: b_down * affinity + block_new
                nisa.scalar_tensor_tensor(
                    data=down_bias_broadcasted[0:TILE_SIZE, 0 : dims.H],
                    op0=nl.multiply,
                    operand0=expert_affinity[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0],
                    op1=nl.add,
                    operand1=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                )
            elif cfg.linear_bias:
                nisa.tensor_tensor(
                    data1=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                    data2=down_bias_broadcasted[0:TILE_SIZE, 0 : dims.H],
                    op=nl.add,
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                )
            if block_old != None:
                nisa.tensor_tensor(
                    data1=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                    data2=block_old[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0 : dims.H],
                    op=nl.add,
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                )
            else:
                return block_new_lnc_recv_sbuf_lst
    else:
        for b_shard_tile_idx in range(dims.NUM_B_TILES_SHARDED):
            if cfg.linear_bias:
                nisa.tensor_tensor(
                    data1=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                    data2=down_bias_broadcasted[0:TILE_SIZE, 0 : dims.H],
                    op=nl.add,
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                )

            if block_old != None:
                if expert_affinity != None:
                    nisa.scalar_tensor_tensor(
                        data=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                        op0=nl.multiply,
                        operand0=expert_affinity[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0],
                        op1=nl.add,
                        operand1=block_old[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0 : dims.H],
                        dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                    )
                else:
                    nisa.tensor_tensor(
                        data1=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                        data2=block_old[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0 : dims.H],
                        op=nl.add,
                        dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                    )
            else:
                if expert_affinity != None:
                    nisa.tensor_scalar(
                        dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                        data=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
                        operand0=expert_affinity[b_shard_tile_idx + N_B_TILES_OFFSET][0:TILE_SIZE, 0],
                        op0=nl.multiply,
                    )
                else:
                    return block_new_lnc_recv_sbuf_lst

    return block_new_lnc_recv_sbuf_lst


def load_down_proj_weight_shard_intermediate_H_tile(
    down_proj_weight, block_expert, cfg: Configs, num_shards, shard_id, H_tile_size, H_tile_idx, dp_load_dst=None
):
    """
    Instead of loading the full [I, H] weight matrix, we load a smaller vertical slice of size [I, H_tile_size]
    Returns a list of tensors instead of a multi-dimensional tensor

    Args:
        down_proj_weight: Down projection weight tensor.
        block_expert: Expert index tensor.
        cfg: Configs object.
        num_shards: Number of shards.
        shard_id: Current shard ID.
        H_tile_size: Size of H tile.
        H_tile_idx: H tile index.
        dp_load_dst: Optional pre-allocated destination tensors.

    Returns:
        dp_load_dst: List of tensors, each of shape (TILE_SIZE, H_tile_size).
    """
    _, _I_TP, _H = down_proj_weight.shape
    I_TP_sharded = _I_TP // num_shards
    I_TP_offset = I_TP_sharded * shard_id
    GUP_N_TILES = div_ceil(I_TP_sharded, TILE_SIZE)

    if dp_load_dst != None:
        dp_load_dst = []
        for i_tile_idx in range(GUP_N_TILES):
            dp_load_dst.append(nl.ndarray((TILE_SIZE, H_tile_size), dtype=cfg.weight_dtype, buffer=nl.sbuf))

    for i_tile_idx in range(GUP_N_TILES):
        num_p = min(TILE_SIZE, _I_TP - (TILE_SIZE * i_tile_idx + I_TP_offset))
        num_f = min(H_tile_size, _H - H_tile_size * H_tile_idx)

        p_offset = TILE_SIZE * i_tile_idx + I_TP_offset
        f_offset = H_tile_size * H_tile_idx

        offset = p_offset * _H + f_offset

        nisa.dma_copy(
            dst=dp_load_dst[i_tile_idx][0:num_p, 0:num_f],
            src=down_proj_weight.ap(pattern=[[_H, num_p], [1, num_f]], offset=offset, scalar_offset=block_expert),
            oob_mode=oob_mode.skip if cfg.skip_dma.skip_weight else oob_mode.error,
        )

    return dp_load_dst


def store_block_output_shard_over_block_size(
    output,
    block_new,
    token_indices,
    dims: DimensionSizes,
    shard_id,
    skip_dma: SkipMode = SkipMode(),
    token_indices_offset=0,
):
    """
    Store the computed block output in the output tensor.

    Assume the full output block is of the shape (B, H), then
    block_new is of the shape (B/2, H).
    Note: block_new is now expected to be a Python list of tensors.

    Args:
        output: Output tensor.
        block_new: List of new block tensors.
        token_indices: Token indices tensor.
        dims: DimensionSizes object.
        shard_id: Current shard ID.
        skip_dma: Skip DMA mode.
        token_indices_offset: Offset for token indices.

    Returns:
        None: Stores results to output tensor in-place.
    """
    N_B_TILES_OFFSET = dims.NUM_B_TILES_SHARDED * shard_id

    for b_shard_tile_idx in range(dims.NUM_B_TILES_SHARDED):
        token_mapping = nl.ndarray((TILE_SIZE, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=token_mapping,
            src=token_indices[
                0:TILE_SIZE,
                token_indices_offset + b_shard_tile_idx + N_B_TILES_OFFSET : token_indices_offset
                + b_shard_tile_idx
                + N_B_TILES_OFFSET
                + 1,
            ],
        )

        if len(output.shape) == 3:
            num_tokens = output.shape[1]
            shard_offset = shard_id * num_tokens * dims.H
        else:
            shard_offset = 0

        nisa.dma_copy(
            dst=output.ap(
                pattern=[
                    [dims.H, TILE_SIZE],
                    [1, dims.H],
                ],
                offset=shard_offset,
                vector_offset=token_mapping,
                indirect_dim=0,
            ),
            src=block_new[b_shard_tile_idx][0:TILE_SIZE, 0 : dims.H],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


def compute_one_block_dropping(
    block_idx, dims: DimensionSizes, inps: InputTensors, outs: OutputTensors, cfg: Configs, shard_id
):
    """Compute one block for the dropping kernel with block tiling approach.

    Used for training with large block sizes (B >= 1024). Tiles the block dimension
    into MAX_BLOCK_TILE_SIZE chunks for efficient processing.
    """
    token_indices = load_token_indices(inps.token_position_to_id, block_idx, dims)

    ORIGINAL_B = dims.B
    ORIGINAL_NUM_B_TILES = dims.NUM_B_TILES
    ORIGINAL_NUM_B_TILES_SHARDED = dims.NUM_B_TILES_SHARDED

    dims.B = MAX_BLOCK_TILE_SIZE
    dims.NUM_B_TILES = dims.NUM_TILES_IN_B_BLOCK_TILE
    dims.NUM_B_TILES_SHARDED = dims.NUM_B_TILES // dims.NUM_SHARDS

    for block_tile_index in nl.sequential_range(dims.NUM_B_BLOCK_TILES):
        block_expert = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(block_expert, value=block_idx)

        block_hidden_states = create_block_hidden_states(dims.H, dims.NUM_B_TILES, cfg.compute_dtype)
        token_indices_offset = block_tile_index * MAX_BLOCK_TILE_SIZE // TILE_SIZE

        load_hidden_states(
            inps.hidden_states,
            block_hidden_states,
            token_indices,
            dims.NUM_B_TILES,
            cfg.compute_dtype,
            cfg.skip_dma,
            token_indices_offset=token_indices_offset,
        )
        block_hidden_states_T = transpose_hidden_states_allocated(
            block_hidden_states, dims.H, MAX_BLOCK_TILE_SIZE, cfg.compute_dtype, is_quant=cfg.is_block_quant
        )

        activation_block_write_offset = block_tile_index * MAX_BLOCK_TILE_SIZE
        gate_and_up_proj_states = compute_gate_and_up_projections_shard_on_intermediate(
            inps,
            block_expert,
            block_hidden_states_T,
            shard_id,
            dims,
            cfg,
            gup_scale=None,
            gate_up_activations_T=outs.gate_up_activations_T,
            block_idx=block_idx,
            activation_block_write_offset=activation_block_write_offset,
        )

        if cfg.expert_affinity_multiply_on_I:
            expert_affinity_T_broadcasted = calculate_expert_affinity_T(
                inps, dims, cfg, block_expert, token_indices, token_indices_offset=token_indices_offset
            )
        else:
            expert_affinity_T_broadcasted = None

        intermediate_states = compute_intermediate_states(
            gate_and_up_proj_states=gate_and_up_proj_states,
            B=dims.B,
            I_TP=dims.I_TP_sharded,
            dtype=cfg.compute_dtype,
            activation_function=cfg.activation_function,
            expert_affinity_T_broadcasted=expert_affinity_T_broadcasted,
            gup_scale=None,
            expert_affinity_multiply_on_I=cfg.expert_affinity_multiply_on_I,
        )

        if cfg.is_tensor_update_accumulating:
            block_old = load_old_block(
                outs.output,
                token_indices,
                dims.NUM_B_TILES,
                cfg.compute_dtype,
                cfg.skip_dma,
                token_indices_offset=token_indices_offset,
            )
        else:
            block_old = None

        if cfg.expert_affinity_multiply_on_I:
            expert_affinity = None
        else:
            expert_affinity = calculate_expert_affinities(
                inps.expert_affinities_masked,
                token_indices,
                block_expert,
                dims.E,
                dims.NUM_B_TILES,
                cfg.compute_dtype,
                cfg.skip_dma,
                token_indices_offset=token_indices_offset,
            )

        block_new = compute_down_proj_shard_on_intermediate(
            intermediate_states,
            inps,
            dims,
            cfg,
            block_expert,
            expert_affinity,
            block_old,
            shard_id,
            outs=outs,
            allocate=True,
            block_idx=block_idx,
            block_tile_index=block_tile_index,
        )

        store_block_output_shard_over_block_size(
            outs.output,
            block_new,
            token_indices,
            dims,
            shard_id,
            cfg.skip_dma,
            token_indices_offset=token_indices_offset,
        )

    dims.B = ORIGINAL_B
    dims.NUM_B_TILES = ORIGINAL_NUM_B_TILES
    dims.NUM_B_TILES_SHARDED = ORIGINAL_NUM_B_TILES_SHARDED


@nki.jit
def blockwise_mm_shard_intermediate_dropping(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    block_size,
    token_position_to_id,
    block_to_expert,
    gate_and_up_proj_bias=None,
    down_proj_bias=None,
    gate_up_proj_scale=None,
    down_proj_scale=None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.PRE_SCALE,
    expert_affinity_multiply_on_I: bool = False,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    accumulation_dtype=None,
):
    """
    Blockwise matrix multiplication kernel for MoE dropping layer with block tiling.

    Implements MoE dropping layer where each block is assigned to one expert. Uses block tiling
    approach for large block sizes (B >= 1024) by processing MAX_BLOCK_TILE_SIZE chunks.

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens
        B: Number of tokens per block
        N: Total number of blocks (equals number of experts E for dropping)
        E: Number of experts
        I_TP: Intermediate size / tp degree

    Args:
        hidden_states (nl.tensor): [T+1, H], Input hidden states. T+1 for padding token at index T.
        expert_affinities_masked (nl.tensor): [(T+1) * E, 1], Expert affinities per token.
        gate_up_proj_weight (nl.tensor): [E, H, 2, I_TP], Gate and up projection weights.
        down_proj_weight (nl.tensor): [E, I_TP, H], Down projection weights.
        block_size (int): Tokens per block.
        token_position_to_id (nl.tensor): [N * B], Token to block index mapping.
        block_to_expert (nl.tensor): [N, 1], Block to expert mapping (unused for dropping, block_idx=expert_idx).
        gate_and_up_proj_bias (nl.tensor, optional): [E, 2, I_TP], Projection bias.
        down_proj_bias (nl.tensor, optional): [E, H], Down projection bias.
        gate_up_proj_scale (nl.tensor, optional): [E, 1, 2 * I_TP], FP8 dequant scale.
        down_proj_scale (nl.tensor, optional): [E, 1, H], FP8 dequant scale.
        activation_function (ActFnType): Activation function (default: SiLU).
        skip_dma (SkipMode): DMA skip configuration.
        compute_dtype: Compute data type (default: bfloat16).
        is_tensor_update_accumulating (bool): Accumulate results over blocks (default: True).
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Scaling mode (default: PRE_SCALE).
        expert_affinity_multiply_on_I (bool): Apply affinity on intermediate dim (default: False).
        gate_clamp_upper_limit (float, optional): Gate projection upper clamp.
        gate_clamp_lower_limit (float, optional): Gate projection lower clamp.
        up_clamp_lower_limit (float, optional): Up projection lower clamp.
        up_clamp_upper_limit (float, optional): Up projection upper clamp.

    Returns:
        When expert_affinity_multiply_on_I=True:
            Tuple of (output, gate_up_activations_T)
        When expert_affinity_multiply_on_I=False:
            Tuple of (output, gate_up_activations_T, down_activations)
    """
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

    T, H = hidden_states.shape
    B = block_size
    E, _I_TP, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B
    SHARD_ID = nl.program_id(axis=0)

    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=_I_TP)
    dims.derive_all_dims()

    inps = InputTensors(
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
    )

    configs = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        is_quant=(gate_up_proj_scale is not None and down_proj_scale is not None),
        fuse_gate_and_up_load=(dims.H * dims.I_TP_sharded <= FUSE_GATE_WEIGHT_SIZE),
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        accumulation_dtype=accumulation_dtype if accumulation_dtype is not None else hidden_states.dtype,
    )

    check_blockwise_mm_shard_I_kernel_compatibility(dims, configs)

    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    gate_up_activations_T = nl.ndarray((N, 2, _I_TP, B), dtype=gate_up_proj_weight.dtype, buffer=nl.shared_hbm)

    down_activations = None
    if not configs.expert_affinity_multiply_on_I:
        down_activations = nl.ndarray((N, B, H), dtype=down_proj_weight.dtype, buffer=nl.shared_hbm)

    outs = OutputTensors(
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        output=output,
    )

    output_initialization(output)

    for block_idx in nl.sequential_range(N):
        compute_one_block_dropping(block_idx, dims, inps, outs, configs, SHARD_ID)
        if dims.NUM_SHARDS == 1:
            nisa.core_barrier(output, (0))
            nisa.core_barrier(gate_up_activations_T, (0))
            if not configs.expert_affinity_multiply_on_I:
                nisa.core_barrier(down_activations, (0))
        elif dims.NUM_SHARDS == 2:
            nisa.core_barrier(output, (0, 1))
            nisa.core_barrier(gate_up_activations_T, (0, 1))
            if not configs.expert_affinity_multiply_on_I:
                nisa.core_barrier(down_activations, (0, 1))
        else:
            kernel_assert(False, "Only 1 or 2 shards are supported")

    if configs.expert_affinity_multiply_on_I:
        return output, gate_up_activations_T

    return output, gate_up_activations_T, down_activations
