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

"""MXFP Blockwise Matrix Multiplication kernel for MoE with intermediate dimension sharding.

This kernel implements MoE MLP at block granularity using MXFP4/MXFP8 or MXFP8 quantized weights and
FP8 quantized activations, sharding the intermediate dimension (I) across 2 cores for
efficient computation on TRN3.

Pipeline Overview:
    1. Load hidden states with H-dimension folding for efficient vector DGE
    2. Transpose layout for quantization alignment
    3. Quantize activations to FP8 (float8_e4m3fn)
    4. Gate projection: hidden @ gate_weight (MXFP matmul)
    5. Up projection: hidden @ up_weight (MXFP matmul)
    6. Apply activation: SiLU(gate) * up
    7. Down projection: intermediate @ down_weight (MXFP matmul)
    8. Cross-core reduction via sendrecv
    9. Apply expert affinity and accumulate output

Key Features:
    - Supports both static loops (predictable blocks) and dynamic loops (padded blocks)
    - Uses indirect DGE for token gathering and expert weight selection
    - Implements cross-core reduction for I-dimension sharding
    - Supports both MXFP4 and MXFP8 weight quantization
"""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import sendrecv
from nki.isa.constants import dge_mode, oob_mode

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_nl_act_fn_from_type, get_program_sharding_info
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...utils.tensor_view import TensorView
from .bwmm_shard_on_I import OutputTensors
from .moe_cte_mx_utils import (
    SBUF_QUADRANT_SIZE,
    BWMMMXConfigs,
    BWMMMXDimensionSizes,
    InputTensors,
    ProjConfig,
    SharedBuffers,
    _generate_expert_index_vector,
    _pmax,
    _psum_fmax,
    _q_height,
    _q_width,
    compute_hidden_index_vector,
    convert_to_mxfp_dtype,
    load_and_quantize_hidden_states,
    load_hidden_states_mx,
    quantize_block_hidden_state_T,
    sbuf_layout_adapter,
)
from .moe_cte_utils import (
    SkipMode,
    calculate_expert_affinities,
    div_ceil,
    load_block_expert,
    load_token_indices,
    load_token_indices_dynamic_block,
)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Use PE-based transpose instead of DMA transpose (more reliable)
USE_DMA_TRANSPOSE = False

# Maximum block size supported
MAX_BLOCK_SIZE = 1024
# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


@nki.jit(mode="trace", platform_target="trn3")
def blockwise_mm_shard_intermediate_mx(
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    gate_and_up_proj_bias: Optional[nl.ndarray] = None,
    down_proj_bias: Optional[nl.ndarray] = None,
    gate_up_proj_scale: nl.ndarray = None,
    down_proj_scale: nl.ndarray = None,
    block_size: int = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=nl.bfloat16,
    weight_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.PRE_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
):
    """
    MXFP Blockwise matrix multiplication kernel for MoE with intermediate dimension sharding.

    Implements MoE layer at block granularity with intermediate dimension sharding using MXFP4/MXFP8
    or MXFP8 quantized weights. Processes tokens in blocks, with each block assigned to a single expert.
    The intermediate dimension (I) is sharded across 2 cores for TRN3.
    Supports both MXFP4 and MXFP8 weight quantization.

    Intended Usage:
        - Block size B: 256-1024 tokens (must be multiple of 256)
        - Hidden dimension H: 512-8192 (must be multiple of 512)
        - Intermediate dimension I: 2048-16384 (must be divisible by 2 * 512)
        - Number of experts E: 8-64
        - Requires TRN3 hardware (NUM_SHARDS == 2)

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens
        B: Number of tokens per block
        N: Total number of blocks (N = len(token_position_to_id) // B)
        E: Number of experts
        I: Intermediate size (before TP sharding)

    Args:
        hidden_states (nl.ndarray): [T+1, H], Input hidden states on HBM.
        expert_affinities_masked (nl.ndarray): [(T+1)*E, 1], Expert affinities per token.
        gate_up_proj_weight (nl.ndarray): [E, 128, 2, n_H512_tile, I], MXFP4/MXFP8 gate/up weights.
        down_proj_weight (nl.ndarray): [E, p_I, n_I512_tile, H], MXFP4/MXFP8 down projection weights.
        token_position_to_id (nl.ndarray): [N*B], Token to block position mapping.
        block_to_expert (nl.ndarray): [N, 1], Expert assignment per block.
        gate_and_up_proj_bias (nl.ndarray, optional): [E, 128, 2, n_I512_tile, 4], Projection bias.
        down_proj_bias (nl.ndarray, optional): [E, H], Down projection bias.
        gate_up_proj_scale (nl.ndarray): [E, 16, 2, n_H512_tile, I], uint8 dequant scales.
        down_proj_scale (nl.ndarray): [E, 16, n_I512_tile, H], uint8 dequant scales.
        block_size (int): Number of tokens per block.
        activation_function (ActFnType): Activation function (default: SiLU).
        skip_dma (SkipMode): DMA skip configuration for debugging.
        compute_dtype: Compute data type (default: bfloat16).
        is_tensor_update_accumulating (bool): Accumulate for TopK > 1 (default: True).
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Affinity scaling mode.
        gate_clamp_upper_limit (float, optional): Upper clamp for gate projection.
        gate_clamp_lower_limit (float, optional): Lower clamp for gate projection.
        up_clamp_upper_limit (float, optional): Upper clamp for up projection.
        up_clamp_lower_limit (float, optional): Lower clamp for up projection.

    Returns:
        output (nl.ndarray): [T+1, H], Output hidden states on HBM.

    Notes:
        - All input/output tensors must have the same floating point dtype
        - Block size B must be multiple of 256
        - Hidden dimension H must be multiple of 512
        - Requires TRN3 with 2 shards for I-dimension sharding

    Pseudocode:
        output = zeros([T+1, H])
        if is_tensor_update_accumulating:
            initialize output to zeros

        # Prefetch first block's hidden states
        load_and_transpose_hidden_states(block_idx=0)
        quantize_hidden_states()

        for block_idx in range(N):
            # Load expert weights for current block
            expert_idx = block_to_expert[block_idx]
            gate_weight, up_weight = load_gup_weights(expert_idx)
            down_weight = load_down_weights(expert_idx)

            # Prefetch next block (overlapped with compute)
            if block_idx < N-1:
                load_and_transpose_hidden_states(block_idx+1)

            # Gate and Up projections (MXFP4/MXFP8 matmul)
            gate_out = hidden @ gate_weight + gate_bias
            up_out = hidden @ up_weight + up_bias

            # Quantize next block's hidden states
            if block_idx < N-1:
                quantize_hidden_states()

            # Activation and intermediate state
            intermediate = activation(gate_out) * up_out

            # Down projection (MXFP4/MXFP8 matmul)
            block_new = intermediate @ down_weight + down_bias

            # Cross-core reduction (sendrecv)
            block_new = reduce_across_shards(block_new)

            # Apply expert affinity and accumulate
            if is_tensor_update_accumulating:
                block_old = load_output(token_indices)
                output[token_indices] = block_new * affinity + block_old
            else:
                output[token_indices] = block_new * affinity

            barrier()

        return output
    """

    # Handle scaling mode conversion
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

    T, H = hidden_states.shape
    B = block_size
    E, _, _, _, I = gate_up_proj_weight.shape
    _, num_shards, SHARD_ID = get_program_sharding_info()

    N = token_position_to_id.shape[0] // B
    dims = BWMMMXDimensionSizes(T=T, H=H, B=B, E=E, N=N, I=I, cond_vec_len=None)

    prj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=dims.B,
        force_lnc1=False,
        n_prgs=num_shards,
        prg_id=SHARD_ID,
        sharding_config="I",
    )

    # Convert weights to MXFP dtype (torch/xla passes weights as alternative dtypes)
    gate_up_proj_weight, target_dtype = convert_to_mxfp_dtype(gate_up_proj_weight, weight_dtype)
    down_proj_weight, _ = convert_to_mxfp_dtype(down_proj_weight, target_dtype)

    # Allocate reusable index vectors
    p_gup_idx_vector = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf, name="p_idx_vector")
    nisa.memset(dst=p_gup_idx_vector, value=-1.0)

    p_down_idx_vector = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf, name="p_down_idx_vector")
    nisa.memset(dst=p_down_idx_vector, value=-1.0)

    gup_scales_sb = nl.ndarray((_pmax, 2, prj_cfg.n_H512_tile, dims.I_lnc_sharded), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(gup_scales_sb, value=0.0)

    activation_bias = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(activation_bias, value=0.0)
    # Create input tensor container
    inps = InputTensors(
        hidden_states=hidden_states.reshape((T, _q_width, prj_cfg.n_H512_tile, _pmax)),
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_weight=down_proj_weight,
        down_proj_scale=down_proj_scale,
        down_proj_bias=down_proj_bias,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        p_gup_idx_vector=p_gup_idx_vector,
        p_down_idx_vector=p_down_idx_vector,
        gup_scales_sb=gup_scales_sb,
        activation_bias=activation_bias,
    )

    # Create configuration
    configs = BWMMMXConfigs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        n_static_blocks=N,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        qtz_dtype=nl.float8_e4m3fn_x4,
    )

    # Allocate output tensors
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    outs = OutputTensors(
        output=output,
    )

    block_hidden_states = None
    if not USE_DMA_TRANSPOSE:
        # if we use PE transpose, we need to load to block_hidden_states first, then transpose
        block_hidden_states = nl.ndarray(
            (_pmax, dims.B // 32, prj_cfg.n_H512_tile, 16 * 8), dtype=configs.compute_dtype, buffer=nl.sbuf
        )
        # zero memset in case of token skipping
        if skip_dma.skip_token:
            nisa.memset(block_hidden_states[:, :, :, :], value=0)

    token_4_H_indices_on_p = nl.ndarray((_pmax, dims.B // 32), dtype=nl.int32, buffer=nl.sbuf)

    block_hidden_states_T = nl.ndarray(
        (_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32 * 4), dtype=configs.compute_dtype, buffer=nl.sbuf
    )
    # zero memset in case of token skipping
    if skip_dma.skip_token and USE_DMA_TRANSPOSE:
        nisa.memset(block_hidden_states_T[:, :, :, :], value=0)

    hidden_qtz_sb = nl.ndarray((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32), dtype=configs.qtz_dtype)
    hidden_scale_sb = nl.ndarray((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32), dtype=nl.uint8)

    block_old = nl.ndarray((_pmax, dims.n_B128_tiles, dims.H), dtype=block_hidden_states_T.dtype, buffer=nl.sbuf)
    if skip_dma.skip_token:
        for b128_tile_idx in range(dims.n_B128_tiles):
            nisa.memset(block_old[0:_pmax, b128_tile_idx, 0:H], value=0)

    down_weight_qtz = nl.ndarray(
        (_pmax, prj_cfg.n_total_I512_tile_lnc_sharded, prj_cfg.H), dtype=inps.down_proj_weight.dtype, buffer=nl.sbuf
    )
    # Memset weight if input weight HBM does not pad on par dim
    if dims.p_I != _pmax:
        nisa.memset(down_weight_qtz[:, prj_cfg.n_total_I512_tile_lnc_sharded - 1, :], value=0)

    # Allocate shared buffers
    buffers = SharedBuffers(
        block_hidden_states=block_hidden_states,
        block_hidden_states_T=block_hidden_states_T,
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scale_sb=hidden_scale_sb,
        block_old=block_old,
        down_weight_qtz=down_weight_qtz,
        token_4_H_indices_on_p=token_4_H_indices_on_p,
    )

    # Initialize output for accumulation
    if is_tensor_update_accumulating:
        output_initialization(output)

    if USE_DMA_TRANSPOSE:
        load_hidden_states_mx(
            inps,
            dims,
            configs.skip_dma,
            block_idx=0,
            block_hidden_states_T=buffers.block_hidden_states_T,
            use_dma_transpose=True,
        )
    else:
        compute_hidden_index_vector(inps, buffers, 0, dims, configs.skip_dma, False)
        load_hidden_states_mx(
            inps,
            dims,
            configs.skip_dma,
            token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
            block_hidden_states=buffers.block_hidden_states,
            use_dma_transpose=False,
        )
        sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

    buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
    buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # Main computation loop
    for block_idx in nl.sequential_range(N):
        compute_one_block_mx(
            block_idx=block_idx,
            next_block_idx=block_idx + 1 if block_idx != N - 1 else None,
            dims=dims,
            inps=inps,
            outs=outs,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            buffers=buffers,
            shard_id=SHARD_ID,
        )

        # Synchronize across shards (only needed for multi-shard)
        if dims.num_shards == 2:
            nisa.core_barrier(output, (0, 1))
        elif dims.num_shards > 2:
            kernel_assert(False, "Only 1 or 2 shards supported")

    # Return results
    return output


@nki.jit(mode="trace", platform_target="trn3")
def blockwise_mm_shard_intermediate_mx_hybrid(
    conditions: nl.ndarray,
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    gate_and_up_proj_bias: Optional[nl.ndarray] = None,
    down_proj_bias: Optional[nl.ndarray] = None,
    gate_up_proj_scale: nl.ndarray = None,
    down_proj_scale: nl.ndarray = None,
    block_size: int = None,
    num_static_block: Optional[int] = None,
    # Meta parameters
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype=nl.bfloat16,
    weight_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
):
    """
    MXFP4/MXFP8 Blockwise matrix multiplication kernel for MoE with hybrid static/dynamic loop control.

    Implements MoE layer at block granularity with intermediate dimension sharding using MXFP4/MXFP8
    quantized weights. Uses hybrid approach: static loop for non-padded blocks, dynamic loop
    for padded blocks. Utilizes TRN3 hardware dynamic loop capabilities for efficient handling
    of variable-length sequences.

    Intended Usage:
        - Block size B: 256-1024 tokens (must be multiple of 256)
        - Total tokens T: Up to 32K tokens per call
        - Hidden dimension H: 512-8192 (optimal: 2048-4096)
        - Intermediate dimension I: 2048-16384 (optimal: 8192)
        - Number of experts E: 8-64 (optimal: 8-16)
        - Use this variant when sequences have variable lengths with significant padding
        - Requires TRN3 hardware (NUM_SHARDS == 2)
        - Set num_static_block to known non-padded count for optimal performance

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I: Intermediate size (before TP sharding)

    Args:
        conditions (nl.ndarray): [N+1], Indicates whether block is padded (0) or non-padded (1).
            Last entry must be 0 to guarantee loop termination.
        hidden_states (nl.ndarray): [T+1, H], Input hidden states on HBM.
        expert_affinities_masked (nl.ndarray): [(T+1)*E, 1], Expert affinities per token.
        gate_up_proj_weight (nl.ndarray): [E, 128, 2, n_H512_tile, I], MXFP4/MXFP8 gate/up weights.
        down_proj_weight (nl.ndarray): [E, p_I, n_I512_tile, H], MXFP4/MXFP8 down projection weights.
        token_position_to_id (nl.ndarray): [N*B], Token to block position mapping.
        block_to_expert (nl.ndarray): [N, 1], Expert assignment per block.
        gate_and_up_proj_bias (nl.ndarray, optional): [E, 128, 2, n_I512_tile, 4], Projection bias.
        down_proj_bias (nl.ndarray, optional): [E, H], Down projection bias.
        gate_up_proj_scale (nl.ndarray): [E, 16, 2, n_H512_tile, I], uint8 dequant scales.
        down_proj_scale (nl.ndarray): [E, 16, n_I512_tile, H], uint8 dequant scales.
        block_size (int): Number of tokens per block.
        num_static_block (int, optional): Number of non-padded blocks if known.
        activation_function (ActFnType): Activation function (default: SiLU).
        skip_dma (SkipMode): DMA skip configuration.
        compute_dtype: Compute data type (default: bfloat16).
        is_tensor_update_accumulating (bool): Accumulate for TopK > 1 (default: True).
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Affinity scaling mode.
        gate_clamp_upper_limit (float, optional): Upper clamp for gate projection.
        gate_clamp_lower_limit (float, optional): Lower clamp for gate projection.
        up_clamp_upper_limit (float, optional): Upper clamp for up projection.
        up_clamp_lower_limit (float, optional): Lower clamp for up projection.

    Returns:
        output (nl.ndarray): [T+1, H], Output hidden states on HBM.

    Notes:
        - All input/output tensors must have the same floating point dtype
        - Block size B must be multiple of 256
        - Hidden dimension H must be between 512 and 8192, and multiple of 512
        - Only works on TRN3 (requires NUM_SHARDS == 2)

    Pseudocode:
        # Initialize output tensor
        output = zeros([T+1, H])

        # Determine number of static (non-padded) blocks
        if num_static_block is provided:
            NUM_STATIC_BLOCKS = num_static_block
        else:
            NUM_STATIC_BLOCKS = N - E

        # Phase 1: Static loop over non-padded blocks (predictable iteration count)
        for block_idx in range(NUM_STATIC_BLOCKS):
            # Prefetch next block's hidden states while computing current block
            compute_one_block_mx(block_idx, next_block_idx=block_idx+1)
            barrier()

        # Phase 2: Dynamic loop over potentially padded blocks (data-dependent iteration)
        block_idx = NUM_STATIC_BLOCKS
        total_active_blocks = sum(conditions)

        while block_idx < N and conditions[block_idx] == 1:
            compute_one_block_mx(block_idx, is_dynamic=True)
            barrier()
            block_idx += 1

        return output
    """

    # Handle scaling mode conversion
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

    T, H = hidden_states.shape
    B = block_size
    E, _, _, _, I = gate_up_proj_weight.shape
    _, num_shards, SHARD_ID = get_program_sharding_info()

    N = token_position_to_id.shape[0] // B
    dims = BWMMMXDimensionSizes(
        T=T, H=H, B=B, E=E, N=N, I=I, cond_vec_len=conditions.shape[0] if conditions is not None else 0
    )

    prj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=dims.B,
        force_lnc1=False,
        n_prgs=num_shards,
        prg_id=SHARD_ID,
        sharding_config="I",
    )

    # Determine number of static blocks
    if num_static_block is not None:
        NUM_STATIC_BLOCKS = num_static_block
        if num_static_block < T // B:
            print("num_static_block is less than T//B, this may lead to performance degradation")
    else:
        NUM_STATIC_BLOCKS = (N - E) if (N - E) % num_shards == 0 else (N - E - 1)

    # Validate sharding requirements
    kernel_assert(
        dims.num_shards == 2, "MXFP4/MXFP8 shard-on-I with dynamic control flow only works on TRN3 with 2 shards"
    )

    # Convert weights to MXFP dtype (torch/xla passes weights as alternative dtypes)
    gate_up_proj_weight, target_dtype = convert_to_mxfp_dtype(gate_up_proj_weight, weight_dtype)
    down_proj_weight, _ = convert_to_mxfp_dtype(down_proj_weight, target_dtype)

    print(f"_pmax: {_pmax}")
    # Allocate reusable index vectors
    p_gup_idx_vector = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf, name="p_idx_vector")
    nisa.memset(dst=p_gup_idx_vector, value=-1.0)

    p_down_idx_vector = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf, name="p_down_idx_vector")
    nisa.memset(dst=p_down_idx_vector, value=-1.0)

    gup_scales_sb = nl.ndarray((_pmax, 2, prj_cfg.n_H512_tile, dims.I_lnc_sharded), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(gup_scales_sb, value=0.0)

    activation_bias = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(activation_bias, value=0.0)

    # Create input tensor container
    inps = InputTensors(
        hidden_states=hidden_states.reshape((T, _q_width, prj_cfg.n_H512_tile, _pmax)),
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_weight=down_proj_weight,
        down_proj_scale=down_proj_scale,
        down_proj_bias=down_proj_bias,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        p_gup_idx_vector=p_gup_idx_vector,
        p_down_idx_vector=p_down_idx_vector,
        gup_scales_sb=gup_scales_sb,
        activation_bias=activation_bias,
    )

    # Create configuration - initially for static blocks
    configs = BWMMMXConfigs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,  # Will be set to True for dynamic phase
        n_static_blocks=NUM_STATIC_BLOCKS,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        qtz_dtype=nl.float8_e4m3fn_x4,
    )

    # Allocate output tensors
    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    outs = OutputTensors(
        output=output,
    )

    # Allocate shared buffers
    block_hidden_states = None
    if not USE_DMA_TRANSPOSE:
        block_hidden_states = nl.ndarray(
            (_pmax, dims.B // 32, prj_cfg.n_H512_tile, 16 * 8), dtype=configs.compute_dtype, buffer=nl.sbuf
        )
        if skip_dma.skip_token:
            nisa.memset(block_hidden_states[:, :, :, :], value=0)

    token_4_H_indices_on_p = nl.ndarray((_pmax, dims.B // 32), dtype=nl.int32, buffer=nl.sbuf)

    block_hidden_states_T = nl.ndarray(
        (_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32 * 4), dtype=configs.compute_dtype, buffer=nl.sbuf
    )
    if skip_dma.skip_token and USE_DMA_TRANSPOSE:
        nisa.memset(block_hidden_states_T[:, :, :, :], value=0)

    hidden_qtz_sb = nl.ndarray((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32), dtype=configs.qtz_dtype)
    hidden_scale_sb = nl.ndarray((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32), dtype=nl.uint8)

    block_old = nl.ndarray((_pmax, dims.n_B128_tiles, dims.H), dtype=block_hidden_states_T.dtype, buffer=nl.sbuf)
    if skip_dma.skip_token:
        for b_tile_idx in range(dims.n_B128_tiles):
            nisa.memset(block_old[0:_pmax, b_tile_idx, 0:H], value=0)

    down_weight_qtz = nl.ndarray(
        (_pmax, prj_cfg.n_total_I512_tile_lnc_sharded, prj_cfg.H), dtype=inps.down_proj_weight.dtype, buffer=nl.sbuf
    )
    if dims.p_I != _pmax:
        nisa.memset(down_weight_qtz[:, prj_cfg.n_total_I512_tile_lnc_sharded - 1, :], value=0)

    buffers = SharedBuffers(
        block_hidden_states=block_hidden_states,
        block_hidden_states_T=block_hidden_states_T,
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scale_sb=hidden_scale_sb,
        block_old=block_old,
        down_weight_qtz=down_weight_qtz,
        token_4_H_indices_on_p=token_4_H_indices_on_p,
    )

    # Initialize output for accumulation
    if is_tensor_update_accumulating:
        output_initialization(output)

    # ═══════════════════════════════════════════════════════════════════════════════
    # PHASE 1: STATIC LOOP OVER NON-PADDED BLOCKS
    # ═══════════════════════════════════════════════════════════════════════════════

    # Prefetch first block's hidden states
    if NUM_STATIC_BLOCKS > 0:
        if USE_DMA_TRANSPOSE:
            load_hidden_states_mx(
                inps,
                dims,
                configs.skip_dma,
                block_idx=0,
                block_hidden_states_T=buffers.block_hidden_states_T,
                use_dma_transpose=True,
            )
        else:
            compute_hidden_index_vector(inps, buffers, 0, dims, configs.skip_dma, False)
            load_hidden_states_mx(
                inps,
                dims,
                configs.skip_dma,
                token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                block_hidden_states=buffers.block_hidden_states,
                use_dma_transpose=False,
            )
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # Static loop over non-padded blocks
    for block_idx in nl.sequential_range(NUM_STATIC_BLOCKS):
        # Determine next block for prefetching
        next_block_idx = block_idx + 1 if block_idx < NUM_STATIC_BLOCKS - 1 else None

        compute_one_block_mx(
            block_idx=block_idx,
            next_block_idx=next_block_idx,
            dims=dims,
            inps=inps,
            outs=outs,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            buffers=buffers,
            shard_id=SHARD_ID,
            is_dynamic=False,
        )

        # Synchronize across shards
        nisa.core_barrier(output, (0, 1))

    # ═══════════════════════════════════════════════════════════════════════════════
    # PHASE 2: DYNAMIC LOOP OVER POTENTIALLY PADDED BLOCKS
    # ═══════════════════════════════════════════════════════════════════════════════

    # Switch to dynamic mode (uses nl.dynamic_range for data-dependent iteration)
    configs.use_dynamic_while = True

    # Load conditions and compute total number of active blocks remaining
    cond_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    conditions_sbuf = nl.ndarray((1, conditions.shape[0]), buffer=nl.sbuf, dtype=nl.int32)

    nisa.dma_copy(dst=conditions_sbuf, src=conditions.reshape((1, conditions.shape[0])))
    nisa.tensor_reduce(dst=cond_sbuf, data=conditions_sbuf, op=nl.add, axis=1)

    cond_reg = nisa.register_alloc()
    nisa.register_load(cond_reg, cond_sbuf)

    # Block index for dynamic loop (stored in SBUF for dynamic indexing)
    block_idx_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.memset(block_idx_sbuf, value=NUM_STATIC_BLOCKS)
    first_block_idx = NUM_STATIC_BLOCKS
    # Prefetch first dynamic block's hidden states if there are any
    load_and_quantize_hidden_states(
        inps, first_block_idx, buffers, dims, configs, prj_cfg, use_dma_transpose=USE_DMA_TRANSPOSE
    )
    # Dynamic loop over remaining blocks
    for _ in nl.dynamic_range(NUM_STATIC_BLOCKS, cond_reg):
        next_block_idx_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        tmp_fp32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=tmp_fp32, data=block_idx_sbuf, op0=nl.add, operand0=1.0, op1=nl.minimum, operand1=float(N - 1)
        )
        nisa.tensor_copy(dst=next_block_idx_sbuf, src=tmp_fp32, engine=nisa.vector_engine)
        compute_one_block_mx(
            block_idx=block_idx_sbuf,
            next_block_idx=next_block_idx_sbuf,  # No prefetching in dynamic loop - handled inside
            dims=dims,
            inps=inps,
            outs=outs,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            buffers=buffers,
            shard_id=SHARD_ID,
            is_dynamic=True,
        )

        # Synchronize across shards
        nisa.core_barrier(output, (0, 1))

        # Increment block index
        nisa.tensor_scalar(dst=block_idx_sbuf, data=block_idx_sbuf, op0=nl.add, operand0=1)
        nisa.core_barrier(block_idx_sbuf, (0, 1))

    # Final DMA to ensure output is flushed
    nisa.dma_copy(output, output)

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def check_kernel_compatibility_mx(dims: BWMMMXDimensionSizes, cfg: BWMMMXConfigs):
    """Validate kernel configuration for MXFP4/MXFP8 execution."""

    kernel_assert(dims.B % 256 == 0, f"Block size must be multiple of 256, got {dims.B}")

    kernel_assert(512 <= dims.H <= 8192, f"Hidden dimension must be in [512, 8192], got {dims.H}")
    kernel_assert(dims.H % 512 == 0, f"Hidden dimension must be multiple of 512 for MXFP4/MXFP8, got {dims.H}")

    kernel_assert(dims.I_TP % 1024 == 0, f"I_TP must be divisible by 1024, got {dims.I_TP}")
    kernel_assert(
        dims.I_TP % dims.NUM_SHARDS == 0, f"I_TP must be divisible by NUM_SHARDS, got {dims.I_TP} / {dims.NUM_SHARDS}"
    )

    kernel_assert(dims.NUM_SHARDS == 2, f"MXFP4/MXFP8 shard-on-I requires exactly 2 shards, got {dims.NUM_SHARDS}")

    kernel_assert(
        dims.NUM_B_TILES % dims.NUM_SHARDS == 0, f"NUM_B_TILES must be divisible by NUM_SHARDS, got {dims.NUM_B_TILES}"
    )

    kernel_assert(not cfg.skip_dma.skip_weight, "DMA weight skipping not supported for MXFP4/MXFP8 kernel")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BLOCK COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_one_block_mx(
    block_idx: int,
    next_block_idx: int,
    buffers: SharedBuffers,
    dims: BWMMMXDimensionSizes,
    inps: InputTensors,
    outs: OutputTensors,
    kernel_cfg: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    shard_id: Any,
    is_dynamic: bool = False,
):
    """
    Process one block through the complete MXFP4/MXFP8 MoE MLP pipeline.

    This function orchestrates all computation stages for a single block of tokens,
    implementing the full MoE MLP forward pass with MXFP4/MXFP8 quantized weights.

    Pipeline Stages:
        1. Load token indices and expert assignment for current block
        2. Prefetch next block's hidden states (overlapped with compute)
        3. Quantize current block's hidden states to FP8
        4. Load MXFP4/MXFP8 weights and scales for assigned expert
        5. Gate projection: hidden @ gate_weight + bias
        6. Up projection: hidden @ up_weight + bias
        7. Apply activation: SiLU(gate) * up
        8. Down projection: intermediate @ down_weight + bias
        9. Cross-core reduction via sendrecv
        10. Apply expert affinity scaling and accumulate to output

    Args:
        block_idx (int): Current block index being processed.
        next_block_idx (int): Next block index for prefetching (None if last block).
        buffers (SharedBuffers): Shared SBUF buffers for intermediate results.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        inps (InputTensors): Input tensors (weights, hidden states, etc.).
        outs (OutputTensors): Output tensor container.
        kernel_cfg (BWMMMXConfigs): Kernel configuration flags.
        prj_cfg (ProjConfig): Projection configuration with tiling info.
        shard_id (Any): Current shard ID for cross-core operations.
        is_dynamic (bool): True if processing dynamic (padded) blocks.

    Returns:
        None: Results are written to outs.output via DMA.

    Notes:
        - Prefetching is disabled for dynamic blocks (next_block_idx handling differs)
        - Cross-core reduction uses sendrecv for I-dimension sharding
        - Expert affinity is applied after reduction for POST_SCALE mode
    """
    # Step 1: Load token indices and expert assignment
    if kernel_cfg.use_dynamic_while:
        token_indices = load_token_indices_dynamic_block(
            inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles, skip_dma=kernel_cfg.skip_dma
        )
        block_expert = load_block_expert(inps.block_to_expert, block_idx)
    else:
        token_indices = load_token_indices(inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles)
        block_expert = load_block_expert(inps.block_to_expert, block_idx)

    # Step 2: Prefetch next block's hidden state indices (overlapped with compute)
    if next_block_idx is not None:
        compute_hidden_index_vector(
            inps, buffers, next_block_idx, dims, kernel_cfg.skip_dma, is_block_idx_dynamic=is_dynamic
        )

    # Step 3: Quantize current block's hidden states to FP8
    # For static blocks, quantize here; for dynamic blocks, quantize after fetching
    if not is_dynamic:
        quantize_block_hidden_state_T(buffers, prj_cfg, dims)

    buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
    buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))

    # Step 4: Load MXFP4/MXFP8 weights and scales for assigned expert
    gate_and_up_weights, gate_and_up_scales, gup_bias, gup_token_indices_on_p, gup_n_quadrants_needed = (
        load_gup_weights_scales_shard_on_intermediate_mx(
            inps, block_expert, dims, prj_cfg=prj_cfg, skip_dma=kernel_cfg.skip_dma
        )
    )

    down_scale_sb, down_bias_sb = load_down_proj_weights_shard_on_intermediate_mx(
        inps,
        block_expert,
        buffers.down_weight_qtz,
        dims,
        prj_cfg,
        kernel_cfg.skip_dma,
        gup_token_indices_on_p=gup_token_indices_on_p,
        gup_n_quadrants_needed=gup_n_quadrants_needed,
    )
    down_bias_broadcasted = nl.ndarray((_pmax, dims.H), dtype=down_bias_sb.dtype, buffer=nl.sbuf)
    flatten_free_dim = prj_cfg.n_total_I512_tile_lnc_sharded * dims.B * _q_width

    if is_dynamic:
        token_indices_2D = load_token_indices_dynamic_block(
            inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles, skip_dma=kernel_cfg.skip_dma
        )
    else:
        token_indices_2D = load_token_indices(inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles)

    kernel_assert(
        token_indices_2D.shape == (_pmax, dims.n_B128_tiles),
        f"Expect token_indices_2D to have shape ({_pmax}, {dims.n_B128_tiles}), got {token_indices_2D.shape}",
    )
    if kernel_cfg.scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
        kernel_assert(False, "PRE_SCALE mode not implemented")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: GATE PROJECTION - hidden @ gate_weight + bias
    # ═══════════════════════════════════════════════════════════════════════════
    gup_weights_reshaped = gate_and_up_weights.reshape((_pmax, 2 * prj_cfg.n_H512_tile, dims.I_lnc_sharded))
    gup_scales_reshaped = gate_and_up_scales.reshape((_pmax, 2 * prj_cfg.n_H512_tile, dims.I_lnc_sharded))
    gup_bias_reshaped = gup_bias.reshape((_pmax, dims.num_shards * prj_cfg.n_total_I512_tile_lnc_sharded, _q_width))

    # Use TensorView to slice bias without tensor_copy
    gup_bias_view = TensorView(gup_bias_reshaped)
    gate_bias_view = gup_bias_view.slice(dim=1, start=0, end=prj_cfg.n_total_I512_tile_lnc_sharded)

    gate_proj_out_sb = gate_up_projection_mx_tp_shard_I(
        hidden_qtz_sb=buffers.hidden_qtz_sb[:, :, :],
        hidden_scale_sb=buffers.hidden_scale_sb[:, :, :],
        weight_qtz=gup_weights_reshaped[:, 0 : prj_cfg.n_H512_tile, :],
        weight_scale=gup_scales_reshaped[:, 0 : prj_cfg.n_H512_tile, :],
        bias_sb=gate_bias_view,
        cfg=prj_cfg,
    )

    gate_proj_out_sb = gate_proj_out_sb.reshape((_pmax, flatten_free_dim))

    # Apply optional clamping to gate projection output
    apply_clamp(
        gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        kernel_cfg.gate_clamp_upper_limit,
        kernel_cfg.gate_clamp_lower_limit,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 6: UP PROJECTION - hidden @ up_weight + bias
    # ═══════════════════════════════════════════════════════════════════════════
    # Use TensorView to slice bias without tensor_copy
    up_bias_view = gup_bias_view.slice(
        dim=1, start=prj_cfg.n_total_I512_tile_lnc_sharded, end=2 * prj_cfg.n_total_I512_tile_lnc_sharded
    )

    up_proj_out_sb = gate_up_projection_mx_tp_shard_I(
        hidden_qtz_sb=buffers.hidden_qtz_sb,
        hidden_scale_sb=buffers.hidden_scale_sb,
        weight_qtz=gup_weights_reshaped[:, prj_cfg.n_H512_tile : 2 * prj_cfg.n_H512_tile, :],
        weight_scale=gup_scales_reshaped[:, prj_cfg.n_H512_tile : 2 * prj_cfg.n_H512_tile, :],
        bias_sb=up_bias_view,
        cfg=prj_cfg,
    )

    up_proj_out_sb = up_proj_out_sb.reshape((_pmax, flatten_free_dim))
    # clipping up
    apply_clamp(
        up_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        kernel_cfg.up_clamp_upper_limit,
        kernel_cfg.up_clamp_lower_limit,
    )
    if next_block_idx is not None:
        """
        LOAD, TRANSPOSE, AND QUANTIZE BLOCK HIDDEN STATES
        """
        if USE_DMA_TRANSPOSE:
            load_hidden_states_mx(
                inps,
                dims,
                kernel_cfg.skip_dma,
                block_idx=next_block_idx,
                block_hidden_states_T=buffers.block_hidden_states_T,
                use_dma_transpose=True,
            )
            # TODO: Consider using PE broadcast instead of stream_shuffle_broadcast for better performance
            stream_shuffle_broadcast(src=down_bias_sb, dst=down_bias_broadcasted)
        else:
            load_hidden_states_mx(
                inps,
                dims,
                kernel_cfg.skip_dma,
                token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                block_hidden_states=buffers.block_hidden_states,
                use_dma_transpose=False,
            )
            # TODO: Consider using PE broadcast instead of stream_shuffle_broadcast for better performance
            stream_shuffle_broadcast(src=down_bias_sb, dst=down_bias_broadcasted)
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

        if is_dynamic:
            # quantize after fetching for dynamic blocks
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)

    else:
        # TODO: Consider using PE broadcast instead of stream_shuffle_broadcast for better performance
        stream_shuffle_broadcast(src=down_bias_sb, dst=down_bias_broadcasted)
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # activation
    """
    when activation function is silu, 
    intermediate = silu(gate_proj) * up_proj

    when activation function is swiglu, 
    intermediate = swiglu(gate_proj) * (up_proj + 1)
    Note that we expect up_proj_bias already contains +1 
    (ie the framework should give the kernel up_bias + 1 instead of just bias)
    """
    nisa.activation(
        dst=gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        op=get_nl_act_fn_from_type(kernel_cfg.activation_function),
        data=gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        scale=1.0,
        bias=inps.activation_bias,
    )

    # intermediate state
    intermediate_state_sb = nl.ndarray((_pmax, flatten_free_dim), dtype=nl.bfloat16, buffer=nl.sbuf)

    nisa.tensor_tensor(
        intermediate_state_sb[:_pmax, :flatten_free_dim],
        gate_proj_out_sb[:_pmax, :flatten_free_dim],
        up_proj_out_sb[:_pmax, :flatten_free_dim],
        op=nl.multiply,
    )
    intermediate_state_sb = intermediate_state_sb.reshape(
        (_pmax, prj_cfg.n_total_I512_tile_lnc_sharded, dims.B, _q_width)
    )

    """
    DOWN PROJECTION
    """

    block_new = down_projection_mx_shard_I(
        inter_sb=intermediate_state_sb,
        weight=buffers.down_weight_qtz,
        weight_scale=down_scale_sb,
        bias_sb=down_bias_broadcasted,
        cfg=prj_cfg,
    )

    if kernel_cfg.scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
        expert_affinity = calculate_expert_affinities(
            inps.expert_affinities_masked,
            token_indices,
            block_expert,
            dims.E,
            dims.n_B128_tiles,
            nl.float32,
            kernel_cfg.skip_dma,
        )
    if kernel_cfg.is_tensor_update_accumulating:
        block_old = load_old_block(
            outs.output, token_indices, dims.n_B128_tiles, kernel_cfg.compute_dtype, kernel_cfg.skip_dma
        )
    else:
        block_old = None
    block_new_lnc_recv_sbuf_lst = accumulation_after_down_proj(
        block_new, block_old, expert_affinity, dims, kernel_cfg, shard_id
    )
    store_block_output_shard_over_block_size(
        outs.output, block_new_lnc_recv_sbuf_lst, token_indices, dims, shard_id, kernel_cfg.skip_dma
    )


def load_gup_weights_scales_shard_on_intermediate_mx(
    inps: InputTensors, block_expert: nl.ndarray, dims: BWMMMXDimensionSizes, prj_cfg: ProjConfig, skip_dma: SkipMode
):
    """
    Load gate and up projection weights, scales, and biases for current expert.

    Loads MXFP4/MXFP8 quantized weights, uint8 scales, and biases for both gate and up
    projections from HBM to SBUF for the expert assigned to the current block.

    Args:
        inps (InputTensors): Input tensors containing gate_up_proj_weight of shape
            [E, 128, 2, n_H512_tile, I], gate_up_proj_scale, gate_and_up_proj_bias,
            and buffers for scales and index vectors.
        block_expert (nl.ndarray): Expert index for current block, shape [1, 1].
        dims (BWMMMXDimensionSizes): Dimension configuration with I, H.
        prj_cfg (ProjConfig): Projection configuration with n_H512_tile, I.
        skip_dma (SkipMode): DMA skip configuration for weight loading.

    Returns:
        tuple: (gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb)
            - gup_weights_qtz_sb (nl.ndarray): Quantized weights [128, 2, n_H512_tile, I]
            - gup_scales_sb (nl.ndarray): Dequantization scales [128, 2, n_H512_tile, I]
            - gup_bias_sb (nl.ndarray): Bias values [128, 2, n_total_I512_tile, 128]

    Notes:
        - Uses indirect DGE with block_expert for expert selection
        - Constructs partition index vector for scale loading with proper offsets
        - Pads bias to 512 when I < 512 for alignment
        - Scales are loaded with zero-padding for out-of-bounds partitions
        - Gate and up projections share weight buffer (dimension 1 has size 2)

    Pseudocode:
        gup_weights_qtz_sb = allocate [128, 2, n_H512_tile, I] in SBUF
        dma_copy gate_up_proj_weight[block_expert, :, :, :, :] to gup_weights_qtz_sb

        gup_scale_view = reshape gate_up_proj_scale to [E*16, 2, n_H512_tile, I]
        construct p_gup_idx_vector: [block_expert*16+0, block_expert*16+1, ..., block_expert*16+15, -1, -1, ...]
        dma_copy gup_scale_view[p_gup_idx_vector, :, :, :] to gup_scales_sb

        gup_bias_sb = allocate [128, 2, n_total_I512_tile, 128] in SBUF
        if I < 512:
            memset gup_bias_sb to 0
            dma_copy gate_and_up_proj_bias[block_expert, :I//4, :, :, :] to gup_bias_sb[:I//4, :, :, :]
        else:
            dma_copy gate_and_up_proj_bias[block_expert, :, :, :, :] to gup_bias_sb

        return gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb
    """
    gup_weights_qtz_sb = nl.ndarray(
        (_pmax, 2, prj_cfg.n_H512_tile, dims.I_lnc_sharded), dtype=inps.gate_up_proj_weight.dtype, buffer=nl.sbuf
    )
    # gate_up_proj_weight shape: (E, 128, 2, n_H512_tile, I)
    # We want to load expert[block_expert] -> shape (128, 2, n_H512_tile, I)
    # select expert -> (128, 2, n_H512_tile, I)
    # slice I for LNC sharding -> (128, 2, n_H512_tile, I_lnc_sharded)
    gup_weight_view = inps.gate_up_proj_weight.select(dim=0, index=block_expert).slice(
        dim=3, start=dims.I_lnc_sharded * dims.shard_id, end=dims.I_lnc_sharded * (dims.shard_id + 1)
    )
    nisa.dma_copy(
        dst=gup_weights_qtz_sb,
        src=gup_weight_view.get_view(),
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    """
    GATE UP SCALES
    """
    # Alloc and load weight scale, which needs zero padding in sbuf

    scale_shape = inps.gate_up_proj_scale.shape

    # fold E * 16 together
    gup_scale_view = inps.gate_up_proj_scale.reshape(
        (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
    )

    """
    Construct a vector DGE index to index into E*16
        if block_expert == 0, we want something like this (tranposed to the P dimension)
        [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]  
    
        if block_expert == 3, we want something like this
        [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]  
        i.e, basically the same as above, with offset 16*3 = 48
    """

    gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
    token_indices_on_p = _generate_expert_index_vector(
        expert_index=block_expert,
        dst_idx_vector=inps.p_gup_idx_vector,
        scale_factor=scale_shape[1],
        n_quadrants_needed=gup_n_quadrants_needed,
        n_remaining_partition=0,
    )
    # gup_scale_view shape: (E*16, 2, n_H512_tile, I) - use FULL source tensor dimensions for strides
    # The source tensor has full n_H512_tile, we only load n_H512_tile elements
    full_n_H512_tile_scale = scale_shape[3]  # Get actual n_H512_tile from source tensor (before reshape)
    stride_dim0 = 2 * full_n_H512_tile_scale * prj_cfg.I
    nisa.dma_copy(
        src=gup_scale_view.ap(
            pattern=[
                [stride_dim0, _pmax],  # stride for dim 0 (uses full n_H512_tile)
                [full_n_H512_tile_scale * prj_cfg.I, 2],  # stride for gate/up dim (uses full n_H512_tile)
                [prj_cfg.I, prj_cfg.n_H512_tile],  # stride for H512 tile (only load sharded count)
                [1, prj_cfg.I // 2],  # stride for I dim
            ],
            offset=(prj_cfg.I // 2) * dims.shard_id,
            vector_offset=token_indices_on_p.ap(
                [[1, _pmax], [1, 1]],
                offset=0,
            ),
            indirect_dim=0,
        ),
        dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile, : prj_cfg.I // 2],
        oob_mode=oob_mode.skip,
    )

    """
    GATE UP BIAS
    """
    ## TODO: handle edge case when prj_cfg.n_I512_tile_lnc_sharded is odd

    gup_bias_sb = nl.ndarray(
        (_pmax, 2, prj_cfg.n_I512_tile_lnc_sharded, _q_width), dtype=inps.gate_and_up_proj_bias.dtype, buffer=nl.sbuf
    )

    if dims.I < _pmax * _q_width:  # when I<512, gate/up bias HBM is not padded so pad it here
        nisa.memset(dst=gup_bias_sb[:, :, 0, :], value=0.0)
        # gate_and_up_proj_bias shape: (E, I_par_dim, 2, n_total_I512_tile, _q_width) where I_par_dim = I//4
        I_par_dim = dims.I // 4
        bias_stride_dim0 = 2 * prj_cfg.n_total_I512_tile * _q_width  # stride for I_par_dim
        bias_stride_dim1 = prj_cfg.n_total_I512_tile * _q_width  # stride for gate/up (2)
        bias_stride_dim2 = _q_width  # stride for n_total_I512_tile
        static_offset = prj_cfg.n_I512_tile_lnc_sharded * bias_stride_dim2 * dims.shard_id
        nisa.dma_copy(
            dst=gup_bias_sb[:I_par_dim, :, :, :],
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim0, I_par_dim],
                    [bias_stride_dim1, 2],
                    [bias_stride_dim2, prj_cfg.n_I512_tile_lnc_sharded],
                    [1, _q_width],
                ],
                offset=static_offset,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )
    else:
        # gate_and_up_proj_bias shape: (E, _pmax, 2, n_total_I512_tile, _q_width)
        # Strides: dim1=2*n_total_I512_tile*_q_width, dim2=n_total_I512_tile*_q_width, dim3=_q_width, dim4=1
        bias_stride_dim1 = 2 * prj_cfg.n_total_I512_tile * _q_width
        bias_stride_dim2 = prj_cfg.n_total_I512_tile * _q_width
        static_offset = _q_width * prj_cfg.n_I512_tile_lnc_sharded * dims.shard_id
        nisa.dma_copy(
            dst=gup_bias_sb,
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim1, _pmax],
                    [bias_stride_dim2, 2],
                    [_q_width, prj_cfg.n_I512_tile_lnc_sharded],
                    [1, _q_width],
                ],
                offset=static_offset,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )

    return gup_weights_qtz_sb, inps.gup_scales_sb, gup_bias_sb, token_indices_on_p, gup_n_quadrants_needed


def load_down_proj_weights_shard_on_intermediate_mx(
    inps: InputTensors,
    block_expert: nl.ndarray,
    dst_weight: nl.ndarray,
    dims: BWMMMXDimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
    gup_token_indices_on_p: nl.ndarray = None,
    gup_n_quadrants_needed: int = None,
):
    """
    Load down projection weights, scales, and biases for current expert.

    Loads MXFP4/MXFP8 quantized weights and uint8 scales for down projection from HBM
    to SBUF, constructing partition index vectors for proper expert selection.

    Args:
        inps (InputTensors): Input tensors containing down_proj_weight [E, p_I, n_total_I512_tile, H],
            down_proj_scale, down_proj_bias, and index vector buffer.
        block_expert (nl.ndarray): Expert index for current block, shape [1, 1].
        dst_weight (nl.ndarray): Destination buffer for weights in SBUF.
        dims (BWMMMXDimensionSizes): Dimension configuration with I, H, p_I.
        prj_cfg (ProjConfig): Projection configuration with sharding info.
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        tuple: (down_weight_hbm, down_scale_sb, down_bias_sb)
            - down_weight_hbm: Reference to weight tensor in HBM
            - down_scale_sb: Scales in SBUF [128, n_total_I512_tile, H]
            - down_bias_sb: Bias in SBUF [1, H]

    Notes:
        - Loads only sharded portion of H dimension per program
        - Constructs partition index with quadrant-based addressing
        - Handles remainder partitions when p_I not divisible by 32
        - Zero-pads scales when p_I < 128

    Pseudocode:
        dma_copy down_proj_weight[block_expert, :, :, :] to dst_weight

        down_scale_sb = allocate [128, n_total_I512_tile, H] in SBUF
        if p_I != 128:
            memset down_scale_sb[:, -1, :] to 0

        down_scale_view = reshape down_proj_scale to [E*16, n_total_I512_tile, H]
        construct p_down_idx_vector: [block_expert*16+0, ..., block_expert*16+15, -1, ...]
        dma_copy down_scale_view[p_down_idx_vector, :, :] to down_scale_sb

        down_bias_sb = allocate [1, H] in SBUF
        dma_copy down_proj_bias[block_expert, :] to down_bias_sb

        return down_scale_sb, down_bias_sb
    """
    """
    DOWN WEIGHTS
    """

    # down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    # Load directly into dst_weight with scalar AP
    # scalar_offset=block_expert with indirect_dim=0 means access starts at block_expert * (p_I * n_total_I512_tile * H)

    # down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    # select expert -> (p_I, n_total_I512_tile, H)
    # slice I512 tiles for LNC sharding -> (p_I, n_I512_tile_lnc_sharded, H)
    down_weight_view = inps.down_proj_weight.select(dim=0, index=block_expert).slice(
        dim=1,
        start=prj_cfg.n_I512_tile_lnc_sharded * dims.shard_id,
        end=prj_cfg.n_I512_tile_lnc_sharded * (dims.shard_id + 1),
    )
    nisa.dma_copy(
        src=down_weight_view.get_view(),
        dst=dst_weight[: dims.p_I, :, :],
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    """
    DOWN SCALES
    """
    scale_shape = inps.down_proj_scale.shape

    # Alloc and load weight scale, which needs zero padding in sbuf
    down_scale_sb = nl.ndarray(
        (_pmax, prj_cfg.n_I512_tile_lnc_sharded, prj_cfg.H), dtype=inps.down_proj_scale.dtype, buffer=nl.sbuf
    )  # original nl.uint8
    # Memset weight scale if input weight scale HBM does not pad on par dim
    if dims.p_I != _pmax:
        nisa.memset(dst=down_scale_sb[:, prj_cfg.n_I512_tile_lnc_sharded - 1, :], value=0)
    kernel_assert(
        down_scale_sb.shape == (128, prj_cfg.n_I512_tile_lnc_sharded, prj_cfg.H), f"Got {down_scale_sb.shape}"
    )

    down_scale_view = inps.down_proj_scale.reshape((scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3]))
    """
    Construct a vector DGE index to index into E*16
    if block_expert == 0, we want something like this (tranposed to the P dimension)
    [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]  

    if block_expert == 3, we want something like this
    [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]  
    i.e, basically the same as above, with offset 16*3 = 48
    """

    down_n_quadrants_needed, n_remaining_partition = divmod(dims.p_I, SBUF_QUADRANT_SIZE)
    n_remaining_partition = n_remaining_partition // _q_height

    # Reuse gup token indices if quadrants match, otherwise regenerate
    if gup_n_quadrants_needed is not None and gup_n_quadrants_needed == down_n_quadrants_needed:
        token_indices_on_p = gup_token_indices_on_p
    else:
        token_indices_on_p = _generate_expert_index_vector(
            expert_index=block_expert,
            dst_idx_vector=inps.p_down_idx_vector,
            scale_factor=scale_shape[1],
            n_quadrants_needed=down_n_quadrants_needed,
            n_remaining_partition=n_remaining_partition,
        )
    # down_scale_view shape: (E*16, n_total_I512_tile, H)
    # accumulated shape to right of dim 0: n_total_I512_tile * H
    down_scale_stride_dim0 = prj_cfg.n_total_I512_tile * dims.H
    static_offset = prj_cfg.n_I512_tile_lnc_sharded * dims.H * dims.shard_id
    # Use AP for dst to match src pattern and avoid issues with dynamic blocks
    nisa.dma_copy(
        src=down_scale_view.ap(
            pattern=[[down_scale_stride_dim0, _pmax], [dims.H, prj_cfg.n_I512_tile_lnc_sharded], [1, prj_cfg.H]],
            offset=static_offset,
            vector_offset=token_indices_on_p.ap(
                [[1, _pmax], [1, 1]],
                offset=0,
            ),
            indirect_dim=0,
        ),
        dst=down_scale_sb[:128, : prj_cfg.n_I512_tile_lnc_sharded, : prj_cfg.H],
        oob_mode=oob_mode.skip,
    )

    # load bias
    # down_proj_bias shape: (E, H)
    down_bias_sb = nl.ndarray((1, dims.H), dtype=inps.down_proj_bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        src=inps.down_proj_bias.ap(
            pattern=[[dims.H, 1], [1, dims.H]], offset=0, scalar_offset=block_expert, indirect_dim=0
        ),
        dst=down_bias_sb,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    return down_scale_sb, down_bias_sb


def accumulation_after_down_proj(block_new_lst, block_old, expert_affinity, dims, cfg, shard_id):
    N_B_TILES_OFFSET = dims.n_B128_tiles_sharded * shard_id
    block_new_lnc_recv_sbuf_lst = []
    for b_tile_idx in range(dims.n_B128_tiles_sharded):
        block_new_lnc_recv_sbuf_lst.append(nl.ndarray((_pmax, 1, dims.H), dtype=cfg.io_dtype, buffer=nl.sbuf))
    for b_shard_tile_idx in range(dims.n_B128_tiles_sharded):
        sendrecv(
            src=block_new_lst[0:_pmax, b_shard_tile_idx + dims.n_B128_tiles_sharded * (1 - shard_id), 0 : dims.H],
            dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
            send_to_rank=(1 - shard_id),
            recv_from_rank=(1 - shard_id),
            pipe_id=0,
        )

        nisa.tensor_tensor(
            data1=block_new_lst[0:_pmax, b_shard_tile_idx + N_B_TILES_OFFSET, 0 : dims.H],
            data2=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
            op=nl.add,
            dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
        )

    for b_shard_tile_idx in range(dims.n_B128_tiles_sharded):
        if block_old != None:
            if expert_affinity != None:
                nisa.scalar_tensor_tensor(
                    data=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
                    op0=nl.multiply,
                    operand0=expert_affinity[b_shard_tile_idx + N_B_TILES_OFFSET][0:_pmax, 0],
                    op1=nl.add,
                    operand1=block_old[b_shard_tile_idx + N_B_TILES_OFFSET][0:_pmax, 0, 0 : dims.H],
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
                )
            else:
                nisa.tensor_tensor(
                    data1=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
                    data2=block_old[b_shard_tile_idx + N_B_TILES_OFFSET][0:_pmax, 0, 0 : dims.H],
                    op=nl.add,
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
                )
        else:
            if expert_affinity != None:
                nisa.tensor_scalar(
                    dst=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
                    data=block_new_lnc_recv_sbuf_lst[b_shard_tile_idx][0:_pmax, 0, 0 : dims.H],
                    operand0=expert_affinity[b_shard_tile_idx + N_B_TILES_OFFSET][0:_pmax, 0],
                    op0=nl.multiply,
                )
            else:
                return block_new_lnc_recv_sbuf_lst

    return block_new_lnc_recv_sbuf_lst


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
        block_old_lst: List of tensors, each of shape (_pmax, H).
    """
    H = output.shape[-1]

    block_old_lst = []
    for tile_idx in range(NUM_TILES):
        block_old_lst.append(nl.ndarray((_pmax, 1, H), dtype=dtype, buffer=nl.sbuf))
    for tile_idx in range(NUM_TILES):
        if skip_dma.skip_token:
            nisa.memset(value=0, dst=block_old_lst[tile_idx][0:_pmax, :, 0:H])

        block_token_mapping = nl.ndarray((_pmax, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=block_token_mapping,
            src=token_indices[0:_pmax, token_indices_offset + tile_idx : token_indices_offset + tile_idx + 1],
        )

        if shard_id != None:
            """
            output shape: (num_shards, num_tokens, H)
            Pattern: [[H, _pmax], [1, H]]
            - First dimension (indirect): _pmax iterations with stride H (row stride)
            - Second dimension: H iterations with stride 1 (within row)
            Offset: shard_id * num_tokens * H (to access the correct shard)
            vector_offset: block_token_mapping (shape _pmax, 1)
            indirect_dim: 0 (we're indirecting on the first dimension of the pattern)
            """
            num_tokens = output.shape[1]

            nisa.dma_copy(
                dst=block_old_lst[tile_idx][0:_pmax, 0, 0:H],
                src=output.ap(
                    pattern=[[H, _pmax], [1, H]],
                    offset=shard_id * num_tokens * H,
                    vector_offset=block_token_mapping,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
        else:
            """
            output shape: (num_tokens, H)
            Pattern: [[H, _pmax], [1, H]]
            Offset: 0
            vector_offset: block_token_mapping
            indirect_dim: 0
            """

            nisa.dma_copy(
                dst=block_old_lst[tile_idx].reshape((_pmax, H))[0:_pmax, 0:H],
                src=output.ap(
                    pattern=[[H, _pmax], [1, H]], offset=0, vector_offset=block_token_mapping, indirect_dim=0
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

    return block_old_lst


def store_block_output_shard_over_block_size(
    output,
    block_new,
    token_indices,
    dims: BWMMMXDimensionSizes,
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
    N_B_TILES_OFFSET = dims.n_B128_tiles_sharded * shard_id

    for b_shard_tile_idx in range(dims.n_B128_tiles_sharded):
        token_mapping = nl.ndarray((_pmax, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=token_mapping,
            src=token_indices[
                0:_pmax,
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
                    [dims.H, _pmax],
                    [1, dims.H],
                ],
                offset=shard_offset,
                vector_offset=token_mapping,
                indirect_dim=0,
            ),
            src=block_new[b_shard_tile_idx].reshape((_pmax, dims.H))[0:_pmax, 0 : dims.H],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


def gate_up_projection_mx_tp_shard_I(
    hidden_qtz_sb: nl.ndarray,
    hidden_scale_sb: nl.ndarray,
    weight_qtz: nl.ndarray,
    weight_scale: nl.ndarray,
    bias_sb: Optional[TensorView],
    cfg: ProjConfig,
) -> nl.ndarray:
    """
    Performs the Gate/Up projection with I-dimension sharding. This is the TP version of the projection, i.e. the output will be in transposed
    for down projection. Math (Neuron matmul):
        hidden (moving) [H, BxS] @ weight (stationary) [H, I] → [I, BxS].

    Further, the output will be in SBUF with swizzle layout for subsequent quantization, thus the output layout will
    be: [sb_p, I // sb_p // _q_width, BxS, _q_width].

    NOTE: In the shapes below, H has a tile size of 512 because it's the contraction size of mx_matmul (_pmax * _q_width).

    :param hidden_qtz_sb: mxfp8_x4[_pmax, n_H512_tile, BxS] @ SB. Dim H is shuffled on _pmax.
    :param hidden_scale_sb: uint8[_pmax, n_H512_tile, BxS] @ SB. Dim H is shuffled on _pmax. NOTE: pdim has holes
    :param weight_qtz:
        - mxfp4_x4/mxfp8_x4[_pmax, n_H512_tile, I] @ SB, or
        - mxfp4_x4/mxfp8_x4[_pmax, n_H512_tile, I] @ HBM.
    :param weight_scale:
        - uint8[_pmax, n_H512_tile, I] @ SB, or
        - uint8[_pmax // _q_height, n_H512_tile, I] @ HBM.
    :param bias_sb [OPTIONAL]: TensorView of bf16[_pmax, n_I512_tile, _q_width] @ SB.
    :return: bf16[_pmax, ceil(I / 512), BxS, _q_width] @ SB.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H0, H1, H1, I, BxS = cfg.H0, cfg.H1, cfg.H1, cfg.I_lnc_sharded, cfg.BxS

    BxS_tile_sz = min(BxS, _psum_fmax * 2 // _q_width)  # double psum elts because out is in bf16
    n_BxS_tile = div_ceil(BxS, BxS_tile_sz)

    # Either load weight_qtz from HBM to sbuf or directly use it if it is already in SBUF
    weight_qtz_sb = None
    if weight_qtz.buffer == nl.sbuf:
        kernel_assert(
            weight_qtz.shape == (_pmax, cfg.n_H512_tile, I),
            f"Expect weight_qtz in SBUF to be in shape ({H0}, {cfg.n_H512_tile}, {I}), got {weight_qtz.shape}",
        )
        weight_qtz_sb = weight_qtz
    else:
        kernel_assert(
            weight_qtz.shape == (_pmax, cfg.n_H512_tile, I),
            f"Expect weight_qtz in HBM to be in shape (128, {cfg.n_H512_tile}, {I}), got {weight_qtz.shape}",
        )
        # Load weight into [H0, cfg.n_H512_tile, I] NOTE: this is pre-quantized and each elt is mxfp_x4 (packed H)
        weight_qtz_sb = nl.ndarray((H0, cfg.n_H512_tile, I), dtype=weight_qtz.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=weight_qtz_sb,
            src=weight_qtz[:, 0 : cfg.n_H512_tile, :],
        )

    if cfg.dbg_hidden:
        return hidden_qtz_sb, hidden_scale_sb

    weight_scale_sb = None
    if weight_scale.buffer == nl.sbuf:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_H512_tile, I),
            f"Expect weight_scale in SBUF to have the shape of (128, {cfg.n_H512_tile}, {I}), got {weight_scale.shape}",
        )
        weight_scale_sb = weight_scale
    else:
        # Load weight scale into [H0, n_H512_tile, I] NOTE: we have 1 scale per 8(p)x4(f) tile, but still span across full pdim with gaps
        kernel_assert(
            weight_scale.shape == (_pmax // _q_height, cfg.n_H512_tile, I),
            f"Expect weight_scale in SBUF to have the shape of (16, {cfg.n_H512_tile}, {I}), got {weight_scale.shape}",
        )
        weight_scale_sb = nl.ndarray(weight_qtz_sb.shape, dtype=nl.uint8, buffer=nl.sbuf)
        # Load 4 partitions of scales for every quadrant
        n_quadrants_needed = H0 // SBUF_QUADRANT_SIZE
        for quad_idx in range(n_quadrants_needed):
            nisa.dma_copy(
                src=weight_scale[
                    quad_idx * 4 : (quad_idx + 1) * 4,
                    0 : cfg.n_H512_tile,
                    :,
                ],
                dst=weight_scale_sb[quad_idx * SBUF_QUADRANT_SIZE : quad_idx * SBUF_QUADRANT_SIZE + 4, :, :],
            )

    if cfg.dbg_weight:
        return weight_qtz_sb, weight_scale_sb

    out_sb = nl.ndarray((_pmax, cfg.n_total_I512_tile_lnc_sharded, BxS, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)

    # Loop over BxS tiles (each of size 256)
    for bxs_tile_idx in range(n_BxS_tile):
        # For the last iter, we may have less than BxS_tile_sz to work with
        cur_BxS_tile_offset = bxs_tile_idx * BxS_tile_sz
        cur_BxS_tile_sz = min(BxS_tile_sz, BxS - cur_BxS_tile_offset)

        # Allocate and init output psum and sbuf. Note that there are cfg.n_total_I512_tile instances of out_psum
        out_psum_lst = []
        for i_tile_idx in range(cfg.n_total_I512_tile_lnc_sharded):
            out_psum_lst.append(nl.ndarray((_pmax, _q_width, cur_BxS_tile_sz), dtype=nl.bfloat16, buffer=nl.psum))

        # Matmul compute, tiles on H, then I, then _q_width (4)
        for h_tile_idx in range(cfg.n_H512_tile):
            for i_tile_idx in range(cfg.n_total_I512_tile_lnc_sharded):
                cur_I512_tile_sz = min(512, I - i_tile_idx * 512)

                # Iterate _q_width number of I128 tiles, each uses 1/4 elts of an I512 tile (which may not be 512 for the last tile)
                for i_mm_tile_idx in range(_q_width):
                    cur_I128_tile_sz = cur_I512_tile_sz // 4
                    weight_I_offset = i_tile_idx * 512 + i_mm_tile_idx * cur_I128_tile_sz
                    nisa.nc_matmul_mx(
                        dst=out_psum_lst[i_tile_idx][:cur_I128_tile_sz, i_mm_tile_idx, :cur_BxS_tile_sz],
                        stationary=weight_qtz_sb[:, h_tile_idx, weight_I_offset : weight_I_offset + cur_I128_tile_sz],
                        moving=hidden_qtz_sb[
                            :, h_tile_idx, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz
                        ],
                        stationary_scale=weight_scale_sb[
                            :, h_tile_idx, weight_I_offset : weight_I_offset + cur_I128_tile_sz
                        ],
                        moving_scale=hidden_scale_sb[
                            :, h_tile_idx, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz
                        ],
                    )

        # Copy out psum to output sbuf NOTE: final tile may not use all partitions
        for i_tile_idx in range(cfg.n_total_I512_tile_lnc_sharded):
            # Last tile of psum may have less partitions to copy
            cur_I_pdim_sz = min(_pmax, I // 4 - i_tile_idx * _pmax)

            """
            Copy output while adding bias if needed.
            
            out_sb shape: [_pmax, cfg.n_total_I512_tile, BxS, _q_width]
            out_psum shape: [_pmax, _q_width, BxS_tile_sz] (for each item in out_psum_lst)
            """
            if bias_sb is not None:
                """
                Use TensorView to slice and broadcast bias.
                
                For combined gate+up: bias_t_shared_base_offset // _q_width gives starting tile index
                For already-sliced: bias_t_shared_base_offset is 0
                """
                i_I512_offset = (cfg.bias_t_shared_base_offset // _q_width) + i_tile_idx

                bias_tile_view = bias_sb.slice(dim=0, start=0, end=cur_I_pdim_sz)
                bias_tile_view = bias_tile_view.slice(dim=1, start=i_I512_offset, end=i_I512_offset + 1)
                bias_tile_view = bias_tile_view.broadcast(dim=1, size=cur_BxS_tile_sz)

                nisa.tensor_tensor(
                    dst=out_sb[
                        :cur_I_pdim_sz, i_tile_idx, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                    ],
                    data1=out_psum_lst[i_tile_idx].ap(
                        [[_q_width * cur_BxS_tile_sz, cur_I_pdim_sz], [1, cur_BxS_tile_sz], [cur_BxS_tile_sz, _q_width]]
                    ),  # strided read
                    data2=bias_tile_view.get_view(),
                    op=nl.add,
                )
            else:
                nisa.tensor_copy(
                    dst=out_sb[
                        :cur_I_pdim_sz, i_tile_idx, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                    ],
                    src=out_psum_lst[i_tile_idx].ap(
                        [[_q_width * cur_BxS_tile_sz, cur_I_pdim_sz], [1, cur_BxS_tile_sz], [cur_BxS_tile_sz, _q_width]]
                    ),  # strided read
                )

    return out_sb


def down_projection_mx_shard_I(
    inter_sb: nl.ndarray, weight: nl.ndarray, weight_scale: nl.ndarray, bias_sb: nl.ndarray, cfg: ProjConfig
) -> nl.ndarray:
    """
    Perform down projection with MXFP4/MXFP8 quantization.

    Computes weight @ intermediate + bias using MXFP4/MXFP8/MXFP8 quantized weights,
    producing final MLP output. This version supports larger BxS values (CTE workloads)
    by tiling the BxS dimension.

    Args:
        inter_sb (nl.ndarray): Intermediate activations of shape [128, n_I512_tile, BxS, 4]
            in SBUF with I dimension shuffled on 128 partitions, bf16 type.
        weight (nl.ndarray): Quantized weights of shape [128, ceil(I/512), H] in HBM,
            mxfp4_x4/mxfp8_x4 type, zero-padded.
        weight_scale (nl.ndarray): Weight scales of shape [128//8, ceil(I/512), H] in HBM,
            uint8 type, zero-padded.
        bias_sb (nl.ndarray): Optional bias of shape [1, H] in SBUF, bf16 type.
        cfg (ProjConfig): Projection configuration with H, I, BxS, sharding info.

    Returns:
        output (nl.ndarray): Down projection result of shape [128, ceil(BxS/128), H] in SBUF,
            bf16 type. Note: end of last tile contains garbage when BxS % 128 != 0.

    Notes:
        - Math: weight [I, H] @ inter_sb [I, BxS] → [BxS, H]
        - Quantizes intermediate activations online to MXFP4/MXFP8
        - Uses nc_matmul_mx for MXFP4/MXFP8 matrix multiplication
        - Tiles BxS dimension in chunks of 128
        - Bias added only by program 0 when using LNC sharding
        - Supports optional partition offset for TKG scenarios
    """
    H, H0, H1, H1, I, BxS = cfg.H, cfg.H0, cfg.H1, cfg.H1, cfg.I, cfg.BxS

    n_BxS_tile = div_ceil(BxS, _pmax)
    BxS_tile_sz = _pmax

    # Prep inputs
    inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale = _down_proj_prep_inter_and_weights(
        inter_sb, weight, weight_scale, cfg
    )

    # Prepare bias broadcast if needed
    bias_broadcasted = None
    if bias_sb is not None:
        if bias_sb.shape[0] == 1:
            bias_broadcasted = nl.ndarray((BxS_tile_sz, H), dtype=bias_sb.dtype, buffer=nl.sbuf)
            stream_shuffle_broadcast(src=bias_sb, dst=bias_broadcasted)
        else:
            bias_broadcasted = bias_sb

    # Allocate output buffer
    if cfg.out_p_offset != 0:
        out_sb = nl.ndarray((_pmax, n_BxS_tile, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        out_sb_p_start = cfg.out_p_offset
        out_sb_p_end = cfg.out_p_offset + BxS
    else:
        out_sb = nl.ndarray((BxS_tile_sz, n_BxS_tile, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        out_sb_p_start = 0
        out_sb_p_end = BxS_tile_sz

    H__pmax = 1024 if H % 1024 == 0 else 512
    n_H_tile_sharded = H // H__pmax

    for h_tile_idx in nl.affine_range(n_H_tile_sharded):
        for bxs_tile_idx in nl.affine_range(n_BxS_tile):
            BxS_offset = bxs_tile_idx * BxS_tile_sz
            curr_BxS = min(BxS_tile_sz, BxS - BxS_offset)

            psum_bank = nl.ndarray((curr_BxS, H__pmax), dtype=nl.bfloat16, buffer=nl.psum)

            for i_tile_idx in nl.affine_range(cfg.n_total_I512_tile_lnc_sharded):
                H_offset = h_tile_idx * H__pmax
                nisa.nc_matmul_mx(
                    dst=psum_bank,
                    stationary=inter_qtz[:, i_tile_idx, BxS_offset : BxS_offset + curr_BxS],
                    moving=weight_qtz[:, i_tile_idx, H_offset : H_offset + H__pmax],
                    stationary_scale=inter_qtz_scale[:, i_tile_idx, BxS_offset : BxS_offset + curr_BxS],
                    moving_scale=weight_qtz_scale[:, i_tile_idx, H_offset : H_offset + H__pmax],
                )

            H_out_start = h_tile_idx * H__pmax

            if cfg.out_p_offset == 0:
                if bias_sb is not None:
                    nisa.tensor_tensor(
                        dst=out_sb[:curr_BxS, bxs_tile_idx, H_out_start : H_out_start + H__pmax],
                        data1=psum_bank,
                        data2=bias_broadcasted[:curr_BxS, H_offset : H_offset + H__pmax],
                        op=nl.add,
                    )
                else:
                    engine = nisa.scalar_engine if bxs_tile_idx % 2 == 0 else nisa.vector_engine
                    nisa.tensor_copy(
                        dst=out_sb[:curr_BxS, bxs_tile_idx, H_out_start : H_out_start + H__pmax],
                        src=psum_bank,
                        engine=engine,
                    )
            else:
                if bias_sb is not None:
                    nisa.tensor_tensor(
                        dst=out_sb[
                            out_sb_p_start : out_sb_p_start + curr_BxS,
                            bxs_tile_idx,
                            H_out_start : H_out_start + H__pmax,
                        ],
                        data1=psum_bank,
                        data2=bias_broadcasted[:curr_BxS, H_offset : H_offset + H__pmax],
                        op=nl.add,
                    )
                else:
                    engine = nisa.scalar_engine if bxs_tile_idx % 2 == 0 else nisa.vector_engine
                    nisa.tensor_copy(
                        dst=out_sb[
                            out_sb_p_start : out_sb_p_start + curr_BxS,
                            bxs_tile_idx,
                            H_out_start : H_out_start + H__pmax,
                        ],
                        src=psum_bank,
                        engine=engine,
                    )

    return out_sb


def apply_clamp(tensor, upper_limit, lower_limit):
    """Apply optional upper and/or lower clamping to a tensor."""
    p_dim, f_dim = tensor.shape

    if upper_limit is not None and lower_limit is not None:
        nisa.tensor_scalar(
            dst=tensor[0:p_dim, 0:f_dim],
            data=tensor[0:p_dim, 0:f_dim],
            op0=nl.minimum,
            operand0=upper_limit,
            op1=nl.maximum,
            operand1=lower_limit,
        )
    elif upper_limit is not None:
        nisa.tensor_scalar(
            dst=tensor[0:p_dim, 0:f_dim],
            data=tensor[0:p_dim, 0:f_dim],
            op0=nl.minimum,
            operand0=upper_limit,
        )
    elif lower_limit is not None:
        nisa.tensor_scalar(
            dst=tensor[0:p_dim, 0:f_dim],
            data=tensor[0:p_dim, 0:f_dim],
            op0=nl.maximum,
            operand0=lower_limit,
        )


def _down_proj_prep_inter_and_weights(
    inter_sb: nl.ndarray, weight: nl.ndarray, weight_scale: nl.ndarray, cfg: ProjConfig
) -> tuple[nl.ndarray, nl.ndarray, nl.ndarray, nl.ndarray]:
    """
    Prep intermediate and weights for down projection:
        - for intermediate, reshape and quantize (and reshape back);
        - for weight, load from HBM into SBUF.

    :param inter_sb: bf16[_pmax, n_I512_tile, BxS, 4] @ SB. Dim I is shuffled on 128.
    :param weight: mxfp4_x4/mxfp8_x4[_pmax, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :param weight_scale: mxfp4_x4/mxfp8_x4[_pmax // _q_height, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :return:
        1. (inter_qtz)        mxfp8_x4[_pmax, cfg.n_total_I512_tile, BxS]
        2. (inter_qtz_scale)  uint8[_pmax, cfg.n_total_I512_tile, BxS]
        3. (weight_qtz)       mxfp4_x4/mxfp8_x4[_pmax, cfg.n_total_I512_tile, H]
        4. (weight_qtz_scale) uint8[_pmax, cfg.n_total_I512_tile, H]
    """
    H, I, BxS = cfg.H, cfg.I, cfg.BxS
    p_I = _pmax if I > 512 else I // 4  # we do not pad I if I<512 to save HBM

    # Quantize inter_sb into mxfp4_x4/mxfp8_x4[_pmax, ceil(I/512), BxS] @ SB
    # NOTE: when I%512 != 0, the final I512 tile of inter_sb will contain garbage. Also nc_matmul_mx requires 32/64/128
    # partitions input so all 128 partitions are used (which includes garbage). However, we memset the last tile
    # of weight_qtz and weight_qtz_scale such that the garbage does not matter.
    inter_sb = inter_sb.reshape((_pmax, cfg.n_total_I512_tile_lnc_sharded * BxS * 4))  # flatten to 2D
    inter_qtz = nl.ndarray((_pmax, cfg.n_total_I512_tile_lnc_sharded * BxS), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    inter_qtz_scale = nl.ndarray(inter_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=inter_qtz, src=inter_sb, dst_scale=inter_qtz_scale)
    inter_qtz = inter_qtz.reshape((_pmax, cfg.n_total_I512_tile_lnc_sharded, BxS))
    inter_qtz_scale = inter_qtz_scale.reshape(inter_qtz.shape)

    if cfg.dbg_hidden:
        return inter_qtz, inter_qtz_scale, None, None  # DEBUG

    weight_qtz = None
    if weight.buffer == nl.sbuf:
        weight_qtz = weight
    else:
        # Load weight into [I0, ceil(I/512), H] NOTE: this is pre-quantized and each elt is mxfp_x4 (packed I)
        weight_qtz = nl.ndarray(
            (_pmax, cfg.n_total_I512_tile_lnc_sharded, H),
            dtype=weight.dtype,
            buffer=nl.sbuf,
            name='down_w_qtz_sb',
        )
        # Memset weight if input weight HBM does not pad on par dim
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz[:, cfg.n_total_I512_tile_lnc_sharded - 1, :], value=0.0)

        kernel_assert(weight.shape == (p_I, cfg.n_total_I512_tile_lnc_sharded, H), "Incorrect weight shape")
        nisa.dma_copy(src=weight[:, :, 0:H], dst=weight_qtz[:p_I, :, :], dge_mode=2)

    # Check if weight scale is already in SBUF or needs to be loaded from HBM
    weight_qtz_scale = None
    if weight_scale.buffer == nl.sbuf:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_total_I512_tile_lnc_sharded, H),
            f"Expect weight_scale in SBUF to have the shape of ({_pmax}, {cfg.n_total_I512_tile_lnc_sharded}, {H}), got {weight_scale.shape}",
        )
        weight_qtz_scale = weight_scale
    else:
        # Load weight scale into [I0, ceil(I/512), H] NOTE: we have 1 scale per 8(p)x4(f) tile, but still span across full pdim with gaps
        weight_qtz_scale = nl.ndarray(weight_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf, name="down_w_scale_sb")
        # Memset weight scale if input weight scale HBM does not pad on par dim
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz_scale[:, cfg.n_total_I512_tile_lnc_sharded - 1, :], value=0)

        # Load 4 partitions of scales for every quadrant
        n_quadrants_needed = _pmax // SBUF_QUADRANT_SIZE
        for quad_idx in range(n_quadrants_needed):
            kernel_assert(
                weight_scale.shape == (p_I // _q_height, cfg.n_total_I512_tile_lnc_sharded, H), "Incorrect weight shape"
            )
            # Scalar DGE needs AP to access either exactly 1 partitions or multiple of 16 partitions
            for partition_idx in range(4):
                if quad_idx * 4 + partition_idx < p_I // _q_height:
                    nisa.dma_copy(
                        src=weight_scale[quad_idx * 4 + partition_idx : quad_idx * 4 + partition_idx + 1, :, 0:H],
                        dst=weight_qtz_scale[
                            quad_idx * SBUF_QUADRANT_SIZE + partition_idx : quad_idx * SBUF_QUADRANT_SIZE
                            + partition_idx
                            + 1,
                            :,
                            :,
                        ],
                        dge_mode=2,
                    )

    return inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale


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

    for tile_idx in range(div_ceil(T, _pmax)):
        zeros = nl.ndarray((_pmax, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros, value=0.0)

        if shard_id != None:
            num_elements = min(_pmax, T - tile_idx * _pmax)
            nisa.dma_copy(
                src=zeros[0:num_elements, 0:H], dst=output[shard_id, nl.ds(tile_idx * _pmax, num_elements), 0:H]
            )
        else:
            num_elements = min(_pmax, T - tile_idx * _pmax)
            nisa.dma_copy(src=zeros[0:num_elements, 0:H], dst=output[nl.ds(tile_idx * _pmax, num_elements), 0:H])
