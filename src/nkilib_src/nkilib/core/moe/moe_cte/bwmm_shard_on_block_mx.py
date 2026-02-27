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
This kernel implements blockwise matrix multiplication for Mixture of Experts (MoE) layers using MXFP4 or MXFP8 quantization with block-level sharding. The implementation shards gate/up projections over the intermediate dimension and block accumulation over the batch dimension, processing all blocks without distinguishing between padded and non-padded blocks.
"""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ...mlp.mlp_tkg.down_projection_mx_shard_H import down_projection_mx_shard_H
from ...mlp.mlp_tkg.gate_up_projection_mx_shard_H import gate_up_projection_mx_tp_shard_H
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_nl_act_fn_from_type
from ...utils.logging import get_logger
from ...utils.tensor_view import TensorView
from .bwmm_shard_on_I import DebugTensors, OutputTensors
from .moe_cte_mx_utils import (
    SBUF_QUADRANT_SIZE,
    BWMMMXConfigs,
    BWMMMXDimensionSizes,
    InputTensors,
    ProjConfig,
    SharedBuffers,
    _generate_expert_index_vector,
    _pmax,
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
    PSUM_SIZE,
    SkipMode,
    calculate_expert_affinities,
    div_ceil,
    load_block_expert,
    load_token_indices,
    load_token_indices_dynamic_block,
    output_initialization,
    reduce_outputs,
)

DBG_KERNEL = False
USE_DMA_TRANSPOSE = False

logger = get_logger("bwmm_shard_on_block_mx")


@nki.jit(platform_target="trn3")
def bwmm_shard_on_block_mx(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    # dynamic-loop variables
    conditions: nl.ndarray = None,
    gate_and_up_proj_bias: nl.ndarray = None,
    down_proj_bias: nl.ndarray = None,
    # quantize scales
    gate_up_proj_scale: nl.ndarray = None,
    down_proj_scale: nl.ndarray = None,
    # Non-tensor args
    block_size=None,
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = 55,
    gate_up_activations_T=None,
    down_activations=None,
    # Meta parameters
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=nl.bfloat16,
    weight_dtype: Any = None,  # Target dtype for weight conversion (e.g., nl.float8_e4m3fn_x4, nl.float8_e5m2_x4)
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
):
    """
    Blockwise MXFP MoE kernel, decorated. Use as standalone kernel.

    The blockwise matrix multiplication (matmul) kernel implements a Mixture of Experts (MoE)
    layer at a block granularity, offering an alternative to token dropping approaches.
    This method assumes that tokens have already been assigned to blocks, as specified
    by the user through the token_position_to_id parameter. This kernel shards the gate/up projection
    over the I dimension, and shards the block accumulation over the B dimension.
    This kernel loops over all blocks, without considering they are padded or non-padded blocks.
    Supports both MXFP4 and MXFP8 weight quantization.

    Intended Usage:
        - Block size B: 128-1024 tokens
        - Total tokens T: 32k
        - Hidden dimension H: 512-8192
        - Intermediate dimension I_TP: 384-3072
        - Number of experts E: 8-128

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens (after linearizing across the batch dimension)
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I: Intermediate size / tp degree

    Args:
        hidden_states (nl.ndarray): Tensor of input hidden states on HBM of size (T+1, H). The reason it is T+1 is because padding token position is set to T.
                                        TODO: with skip_dma, id will be set to -1, so this shape can be (T, H). Similarly for expert_affinities_masked, output
        expert_affinities_masked (nl.ndarray): Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1).
                                        TODO: cannot refactor to (T+1, E) as we currently don't support dynamic slice on both axis.
        gate_up_proj_weight (nl.ndarray): Tensor of concatenated gate and up projection weights on HBM (E, H, 2, I).
                                          Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
        down_proj_weight (nl.ndarray): Tensor of down projection weights on HBM (E, I, H).
                                       Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
        block_size (int): Number of tokens per block
        token_position_to_id (nl.ndarray): Tensor of block index of the corresponding tokens on HBM (N * B,)
                                          Note that we include tokens included for padding purposes and N * B >= T.
                                          For padding token, id is set to T. TODO: with skip_dma, id will be set to -1.
        block_to_expert (nl.ndarray): Tensor of expert indices of corresponding blocks on HBM (N, 1)

        num_static_block (int): Optional. Number of non-padded blocks if known (default: -1).
        n_dynamic_blocks (int): Number of blocks to process with dynamic loop when n_static_blocks
            is not specified (default: 55, empirically tuned for GPT-OSS).
        gate_and_up_proj_bias: nl.ndarray = None, Optional. A tensor of shape [E, 2, I].
                              Note that if activation function is Swiglu, we expect up_bias = up_bias + 1
        down_proj_bias: nl.ndarray = None. Optional argument. A tensor of shape [E, H]

        # Arguments for quantization scales
        gate_up_proj_scale: nl.ndarray = None. A tensor of shape [E, 1, 2 * I]
        down_proj_scale: nl.ndarray = None. A tensor of shape [E, 1, H]

        # Unsupported output tensors. Please set to None.
        gate_up_activations_T: nl.ndarray = None. Currently not supported.
        down_activations: nl.ndarray = None. Currently not supported

        # meta parameters
        activation_function: one of the Enum in nkilib.core.utils.common_types.ActFnType.
                              Indicate what activation function to use in the MLP block
        skip_dma: SkipMode = SkipMode(False, False),
        compute_dtype=nl.bfloat16,
        weight_dtype: Target dtype for weight conversion when weights are passed as uint/int/float types.
                     For MXFP4: nl.float4_e2m1fn_x4
                     For MXFP8: nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4
                     If None, auto-detects (defaults to e4m3fn for MXFP8)
        is_tensor_update_accumulating: bool. Indicate whether we need to accumulate the results over multiple blocks
        expert_affinities_scaling_mode: one of the Enum in nkilib.core.utils.common_types.ExpertAffinityScaleMode.
                                        Indicate if the kernel is doing post or pre scaling.
        n_block_per_iter: int. Currently unsupported

        #parameters for clipping the MLP projections
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None

        skip_dma (bool): Whether to skip DMA operations (default: False)

    Returns:
        output (nl.ndarray): Tensor of output hidden states on HBM of size (T+1, H).

    Notes:
        - All input/output tensors must have the same floating point dtype
        - token_position_to_id and block_to_expert must be np.int32 tensors

    Pseudocode:
        if expert_affinities_scaling_mode == PRE_SCALE_DELAYED:
            expert_affinities_scaling_mode = PRE_SCALE

        T, H = hidden_states.shape
        B = block_size
        E, _, _, _, I = gate_up_proj_weight.shape
        N = token_position_to_id.shape[0] // B
        dims = BWMMMXDimensionSizes(T, H, B, E, N, I, cond_vec_len)
        prj_cfg = ProjConfig(H, I, B, force_lnc1=True, n_prgs=1, prg_id=0)
        configs = BWMMMXConfigs(...)

        allocate reused buffers: p_gup_idx_vector, p_down_idx_vector, gup_scales_sb, activation_bias
        inps = InputTensors(...)

        check_kernel_compatibility(dims, configs)

        if is_tensor_update_accumulating:
            output = allocate [2, T, H] in HBM
            output_initialization(output, dims)
        else:
            output = allocate [T, H] in HBM

        allocate shared buffers: block_hidden_states, block_hidden_states_T, hidden_qtz_sb, hidden_scale_sb
        allocate down_weight_qtz, block_old, cond, index

        if use_dynamic_while:
            n_dynamic_blocks = N - n_static_blocks (padded to even)
            n_static_blocks = N - n_dynamic_blocks
            process_static_blocks(n_static_blocks)
            process_dynamic_blocks(n_dynamic_blocks)
        else:
            process_static_blocks(N)

        if num_shards == 2:
            core_barrier(output)

        if is_tensor_update_accumulating and num_shards > 1:
            reduce_outputs(output)

        return output
    """

    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

    T, H = hidden_states.shape
    B = block_size
    E, _, _, _, I = gate_up_proj_weight.shape
    cond_vec_len = conditions.shape[0] if conditions is not None else 0

    N = token_position_to_id.shape[0] // B
    dims = BWMMMXDimensionSizes(T=T, H=H, B=B, E=E, N=N, I=I, cond_vec_len=cond_vec_len)

    prj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=dims.B,
        force_lnc1=True,
        n_prgs=1,
        prg_id=0,
        use_stream_shuffle_broadcast=False,
        sharding_config="H",
    )

    # Convert weights to MXFP dtype (torch/xla passes weights as alternative dtypes)
    gate_up_proj_weight, target_dtype = convert_to_mxfp_dtype(gate_up_proj_weight, weight_dtype)
    down_proj_weight, _ = convert_to_mxfp_dtype(down_proj_weight, target_dtype)

    # reused buffers
    p_gup_idx_vector = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf, name="p_idx_vector")
    nisa.memset(dst=p_gup_idx_vector, value=-1.0)

    p_down_idx_vector = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf, name="p_down_idx_vector")
    nisa.memset(dst=p_down_idx_vector, value=-1.0)

    gup_scales_sb = nl.ndarray((_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(gup_scales_sb, value=0.0)

    activation_bias = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(activation_bias, value=0.0)

    inps = InputTensors(
        hidden_states=hidden_states.reshape((T, _q_width, prj_cfg.n_H512_tile, _pmax)),
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        p_gup_idx_vector=p_gup_idx_vector,
        p_down_idx_vector=p_down_idx_vector,
        gup_scales_sb=gup_scales_sb,
        activation_bias=activation_bias,
        conditions=conditions,
    )

    configs = BWMMMXConfigs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=conditions is not None,
        n_static_blocks=n_static_blocks,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        qtz_dtype=nl.float8_e4m3fn_x4,
    )

    check_kernel_compatibility(dims, configs)

    if is_tensor_update_accumulating:
        output = nl.ndarray((2, dims.T, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
        output_initialization(output, dims)
    else:
        output = nl.ndarray((dims.T, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    outs = OutputTensors(
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        output=output,
    )
    dbg_tensors = None
    if DBG_KERNEL:
        dbg_hidden_states = nl.ndarray(
            (_pmax, dims.H // 512, dims.B // 32, 32 * 4),
            dtype=hidden_states.dtype,
            buffer=nl.shared_hbm,
            name='dbg_hidden_states',
        )
        flatten_free_dim_dbg = prj_cfg.n_total_I512_tile * dims.B * _q_width
        dbg_gate_proj = nl.ndarray(
            (_pmax, flatten_free_dim_dbg), dtype=hidden_states.dtype, buffer=nl.shared_hbm, name='dbg_gate_proj'
        )
        dbg_up_proj = nl.ndarray(
            (_pmax, flatten_free_dim_dbg), dtype=hidden_states.dtype, buffer=nl.shared_hbm, name='dbg_up_proj'
        )
        dbg_down_proj = nl.ndarray(
            (_pmax, dims.n_B128_tiles, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm, name='dbg_down_proj'
        )
        dbg_tensors = DebugTensors(
            hidden_states=dbg_hidden_states,
            gate_proj=dbg_gate_proj,
            down_proj=dbg_down_proj,
            up_proj=dbg_up_proj,
        )
    """
    Allocate buffers for prefetching and current block processing.
    
    hidden_sbuf_expected_shape defines the expected layout for hidden states:
    (32 * 4, dims.B // 32, mx4_prj_cfg.n_H512_tile, 16 * 8)
    """

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
        for n in range(dims.n_B128_tiles):
            nisa.memset(block_old[0:_pmax, n, 0:H], value=0)

    down_weight_qtz = nl.ndarray(
        (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded), dtype=inps.down_proj_weight.dtype, buffer=nl.sbuf
    )
    # Memset weight if input weight HBM does not pad on par dim
    if dims.p_I != _pmax:
        nisa.memset(down_weight_qtz[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

    # init counters
    # in shard-on-block we can move independently
    cond = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32) if configs.use_dynamic_while else None
    index = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32) if configs.use_dynamic_while else None

    buffers = SharedBuffers(
        block_hidden_states=block_hidden_states,
        block_hidden_states_T=block_hidden_states_T,
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scale_sb=hidden_scale_sb,
        block_old=block_old,
        down_weight_qtz=down_weight_qtz,
        # down_scale_sb=down_scale_sb,
        cond=cond,
        index=index,
        token_4_H_indices_on_p=token_4_H_indices_on_p,
    )

    """
    END OF PREPARING DIMS, CONFIGS, SHARED_BUFFERS

    MAIN COMPUTATION STARTS
    """
    if configs.use_dynamic_while:
        if configs.n_static_blocks > 0:
            kernel_assert(
                configs.n_static_blocks < dims.N,
                f"Cannot have more static blocks than total number of blocks. Got ({configs.n_static_blocks}) > N = {dims.N}",
            )
            n_dynamic_blocks = dims.N - configs.n_static_blocks
            n_dynamic_blocks = n_dynamic_blocks + 1 if n_dynamic_blocks % 2 == 1 else n_dynamic_blocks
            n_static_blocks = dims.N - n_dynamic_blocks
        else:
            # If invalid n_dynamic_blocks is passed, auto-calculate best case combination
            if conditions is not None and (n_dynamic_blocks < 0 or n_dynamic_blocks > dims.N):
                n_static_blocks = dims.T // dims.B  # real blocks (best case scenario)
                n_dynamic_blocks_local = dims.N - n_static_blocks
                logger.info(
                    f"n_dynamic_blocks={n_dynamic_blocks} out of range, auto-computing from T={dims.T}, B={dims.B}: "
                    f"{n_static_blocks} static, {n_dynamic_blocks_local} dynamic"
                )
            else:
                n_dynamic_blocks_local = n_dynamic_blocks
                n_static_blocks = dims.N - n_dynamic_blocks_local

        nisa.dma_copy(
            dst=buffers.cond.ap(pattern=[[1, 1], [1, 1]]),
            src=inps.conditions.ap(pattern=[[1, 1], [1, 1]], offset=n_static_blocks),
        )

        nisa.memset(dst=buffers.index[0, 0], value=n_static_blocks)

        logger.info(f"Processing {n_static_blocks} static blocks, {n_dynamic_blocks_local} dynamic blocks")
        process_static_blocks(
            dims=dims,
            configs=configs,
            prj_cfg=prj_cfg,
            inps=inps,
            outs=outs,
            dbg_tensors=dbg_tensors,
            buffers=buffers,
            num_static_blocks=n_static_blocks,
        )
        if n_dynamic_blocks_local > 0:
            process_dynamic_blocks(
                dims=dims,
                configs=configs,
                prj_cfg=prj_cfg,
                inps=inps,
                outs=outs,
                dbg_tensors=dbg_tensors,
                buffers=buffers,
                num_static_blocks=n_static_blocks,
                num_dynamic_blocks=n_dynamic_blocks_local,
            )

    else:
        """
        STATIC LOOP OVER ALL BLOCKS
        """
        process_static_blocks(
            dims=dims,
            configs=configs,
            prj_cfg=prj_cfg,
            inps=inps,
            outs=outs,
            dbg_tensors=dbg_tensors,
            buffers=buffers,
            num_static_blocks=dims.N,
        )

    """
    Final collective to produce the final result
    """

    if dims.num_shards == 2:
        nisa.core_barrier(output, (0, 1))

    if is_tensor_update_accumulating and dims.num_shards > 1:
        kernel_assert(dims.num_shards == 2, "only support reducing data from 2 shards")
        reduce_tile_size = _pmax
        if skip_dma.skip_token:
            reduce_tiles = div_ceil(T, _pmax)
        else:
            reduce_tiles = div_ceil(T - 1, _pmax)

        nc0_tiles = reduce_tiles // dims.num_shards
        nc1_tiles = reduce_tiles - nc0_tiles
        zeros_dummy = nl.ndarray(shape=(reduce_tile_size, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros_dummy, value=0.0)
        if dims.num_shards == 2:
            nisa.core_barrier(output, (0, 1))

        if dims.shard_id == 0:
            reduce_outputs(output, zeros_dummy, nc0_tiles, reduce_tile_size, 0, H)

        if dims.shard_id == 1:
            reduce_outputs(output, zeros_dummy, nc1_tiles, reduce_tile_size, nc0_tiles, H)

    if DBG_KERNEL:
        return output, dbg_hidden_states, dbg_gate_proj, dbg_down_proj, dbg_up_proj
    return output


def load_prev_block(output, token_indices, block_old, NUM_TILES, dtype, shard_id, skip_dma: SkipMode):
    """
    Load previous block outputs for accumulation in tensor update mode.

    Retrieves existing output values for tokens in the current block to enable
    accumulation across multiple expert evaluations (topK > 1).

    Args:
        output (nl.ndarray): Output tensor of shape [num_shards, T, H] containing
            accumulated results from previous blocks.
        token_indices (nl.ndarray): Token indices for current block of shape [P_MAX, NUM_TILES].
        block_old (nl.ndarray): Buffer to store loaded values of shape [P_MAX, NUM_TILES, H].
        NUM_TILES (int): Number of tiles in the block (B // 128).
        dtype: Data type for loading.
        shard_id (int): Current shard identifier (0 or 1).
        skip_dma (SkipMode): DMA skip configuration for handling invalid tokens.

    Returns:
        block_old (nl.ndarray): Loaded previous output values for the block.

    Notes:
        - Uses indirect addressing via token_indices for gather operation
        - Skips DMA for invalid tokens when skip_dma.skip_token is True
        - Required for topK > 1 scenarios where multiple experts contribute to same token
        - Reshapes output tensor for efficient access pattern

    Pseudocode:
        H = output.shape[-1]
        T = output.shape[-2]
        num_shards = output.shape[0]
        shard_offset = shard_id * T * H
        output_reshaped = reshape output to [num_shards * T, 1, H]

        for n in range(NUM_TILES):
            block_token_mapping = token_indices[:, n]
            dma_copy output_reshaped[block_token_mapping + shard_offset, :, :] to block_old[:, n, :]

        return block_old
    """
    H = output.shape[-1]
    T = output.shape[-2]
    num_shards = output.shape[0]
    shard_offset = shard_id * T * H

    # Reshape output to (num_shards * T, 1, H) for proper AP pattern
    output_reshaped = output.reshape((num_shards * T, 1, H))

    for n in range(NUM_TILES):
        block_token_mapping = token_indices.ap(
            [[NUM_TILES, _pmax], [1, 1]],
            offset=n,
        )
        nisa.dma_copy(
            dst=block_old[:_pmax, n, :H],
            src=output_reshaped.ap(
                pattern=[[H, _pmax], [1, 1], [1, H]],
                offset=shard_offset,
                vector_offset=block_token_mapping,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
    return block_old


def check_kernel_compatibility(dims: BWMMMXDimensionSizes, configs: BWMMMXConfigs):
    """
    Validate kernel configuration and dimension compatibility.

    Performs comprehensive validation of kernel parameters to ensure they meet
    hardware constraints and implementation requirements before execution.

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration containing B, H, I, N,
            num_shards, and cond_vec_len.
        configs (BWMMMXConfigs): Kernel configuration containing is_tensor_update_accumulating
            and use_dynamic_while flags.

    Returns:
        None: Raises assertion errors if validation fails.

    Notes:
        - Block size (B) must be multiple of 128 for efficient tiling
        - Hidden dimension (H) must be in range [512, 8192] and divisible by PSUM_SIZE (512)
        - Intermediate dimension (I) must be divisible by 16 for quantization alignment
        - Currently only supports 2-shard execution
        - Dynamic loop requires condition vector of length N+2
        - Only supports topK > 1 (tensor update accumulating mode)

    Pseudocode:
        assert B % 128 == 0
        assert 512 <= H <= 8192
        assert H % PSUM_SIZE == 0
        assert I % 16 == 0
        assert is_tensor_update_accumulating == True
        assert num_shards == 2
        if use_dynamic_while:
            assert cond_vec_len == N + 2
    """
    kernel_assert(dims.B % 128 == 0, f"Blocksize must be a multiple of 128")
    kernel_assert(512 <= dims.H <= 8192, f"Hidden dims must be between 512 and 8192, found {dims.H}")
    kernel_assert(dims.H % PSUM_SIZE == 0, f"Hidden dim size must be multiples of {PSUM_SIZE}, found {dims.H} ")

    kernel_assert(dims.I % 16 == 0, f"down_proj_weight I must be divisible by 16, found {dims.I} . Please pad it")
    kernel_assert(configs.is_tensor_update_accumulating, "Only support topK > 1 at the moment.")

    kernel_assert(dims.num_shards == 2, f"The kernel only support sharding on exactly 2 cores, got {dims.num_shards}")

    if configs.use_dynamic_while:
        kernel_assert(
            dims.cond_vec_len == dims.N + 2,
            f"condition vector must have exactly N+2 elements, got {dims.cond_vec_len} != N + 2 ({dims.N} + 2)",
        )


def load_gup_weights_scales_mx(
    inps: InputTensors,
    block_expert: nl.ndarray,
    dims: BWMMMXDimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
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
        prj_cfg (ProjConfig): Projection configuration with n_H512_tile_sharded, I.
        skip_dma (SkipMode): DMA skip configuration for weight loading.

    Returns:
        tuple: (gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb)
            - gup_weights_qtz_sb (nl.ndarray): Quantized weights [128, 2, n_H512_tile_sharded, I]
            - gup_scales_sb (nl.ndarray): Dequantization scales [128, 2, n_H512_tile_sharded, I]
            - gup_bias_sb (nl.ndarray): Bias values [128, 2, n_total_I512_tile, 128]

    Notes:
        - Uses indirect DGE with block_expert for expert selection
        - Generates index vectors on-the-fly for scale loading
        - Pads bias to 512 when I < 512 for alignment
        - Scales are loaded with zero-padding for out-of-bounds partitions
        - Gate and up projections share weight buffer (dimension 1 has size 2)

    Pseudocode:
        gup_weights_qtz_sb = allocate [128, 2, n_H512_tile_sharded, I] in SBUF
        dma_copy gate_up_proj_weight[block_expert, :, :, :, :] to gup_weights_qtz_sb

        gup_scale_view = reshape gate_up_proj_scale to [E*16, 2, n_H512_tile, I]
        token_indices_on_p = generate_expert_index_vector(block_expert)
        dma_copy gup_scale_view[token_indices_on_p, :, :, :] to gup_scales_sb

        gup_bias_sb = allocate [128, 2, n_total_I512_tile, 128] in SBUF
        if I < 512:
            memset gup_bias_sb to 0
            dma_copy gate_and_up_proj_bias[block_expert, :I//4, :, :, :] to gup_bias_sb[:I//4, :, :, :]
        else:
            dma_copy gate_and_up_proj_bias[block_expert, :, :, :, :] to gup_bias_sb

        return gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb
    """
    gup_weights_qtz_sb = nl.ndarray(
        (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I), dtype=inps.gate_up_proj_weight.dtype, buffer=nl.sbuf
    )
    # gate_up_proj_weight shape: (E, 128, 2, n_H512_tile, I)
    # We want to load expert[block_expert] -> shape (128, 2, n_H512_tile_sharded, I)
    full_n_H512_tile = inps.gate_up_proj_weight.shape[3]  # Get actual n_H512_tile from source tensor
    nisa.dma_copy(
        dst=gup_weights_qtz_sb,
        src=inps.gate_up_proj_weight.ap(
            pattern=[
                [2 * full_n_H512_tile * dims.I, _pmax],  # stride for partition dim (uses full n_H512_tile)
                [full_n_H512_tile * dims.I, 2],  # stride for gate/up dim (uses full n_H512_tile)
                [dims.I, prj_cfg.n_H512_tile_sharded],  # stride for H512 tile (only load sharded count)
                [1, dims.I],  # stride for I dim
            ],
            offset=0,
            scalar_offset=block_expert,
            indirect_dim=0,
        ),
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
    # The source tensor has full n_H512_tile, we only load n_H512_tile_sharded elements
    full_n_H512_tile_scale = scale_shape[3]  # Get actual n_H512_tile from source tensor (before reshape)
    stride_dim0 = 2 * full_n_H512_tile_scale * prj_cfg.I
    nisa.dma_copy(
        src=gup_scale_view.ap(
            pattern=[
                [stride_dim0, _pmax],  # stride for dim 0 (uses full n_H512_tile)
                [full_n_H512_tile_scale * prj_cfg.I, 2],  # stride for gate/up dim (uses full n_H512_tile)
                [prj_cfg.I, prj_cfg.n_H512_tile_sharded],  # stride for H512 tile (only load sharded count)
                [1, prj_cfg.I],  # stride for I dim
            ],
            offset=0,
            vector_offset=token_indices_on_p.ap(
                [[1, _pmax], [1, 1]],
                offset=0,
            ),
            indirect_dim=0,
        ),
        dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
        oob_mode=oob_mode.skip,
    )

    """
    GATE UP BIAS
    """
    gup_bias_sb = nl.ndarray(
        (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width), dtype=inps.gate_and_up_proj_bias.dtype, buffer=nl.sbuf
    )

    if dims.I < _pmax * _q_width:  # when I<512, gate/up bias HBM is not padded so pad it here
        nisa.memset(dst=gup_bias_sb[:, :, 0, :], value=0.0)
        # gate_and_up_proj_bias shape: (E, I_par_dim, 2, n_total_I512_tile, _q_width) where I_par_dim = I//4
        I_par_dim = dims.I // 4
        bias_stride_dim0 = 2 * prj_cfg.n_total_I512_tile * _q_width  # stride for I_par_dim
        bias_stride_dim1 = prj_cfg.n_total_I512_tile * _q_width  # stride for gate/up (2)
        bias_stride_dim2 = _q_width  # stride for n_total_I512_tile
        nisa.dma_copy(
            dst=gup_bias_sb[:I_par_dim, :, :, :],
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim0, I_par_dim],
                    [bias_stride_dim1, 2],
                    [bias_stride_dim2, prj_cfg.n_total_I512_tile],
                    [1, _q_width],
                ],
                offset=0,
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
        nisa.dma_copy(
            dst=gup_bias_sb,
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim1, _pmax],
                    [bias_stride_dim2, 2],
                    [_q_width, prj_cfg.n_total_I512_tile],
                    [1, _q_width],
                ],
                offset=0,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )

    return gup_weights_qtz_sb, inps.gup_scales_sb, gup_bias_sb, token_indices_on_p, gup_n_quadrants_needed


def load_down_proj_weights_mx(
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
            - down_scale_sb: Scales in SBUF [128, n_total_I512_tile, H_sharded]
            - down_bias_sb: Bias in SBUF [1, H]

    Notes:
        - Loads only sharded portion of H dimension per program
        - Constructs partition index with quadrant-based addressing
        - Handles remainder partitions when p_I not divisible by 32
        - Zero-pads scales when p_I < 128

    Pseudocode:
        dma_copy down_proj_weight[block_expert, :, :, :] to dst_weight

        down_scale_sb = allocate [128, n_total_I512_tile, H_sharded] in SBUF
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
    Load down projection weights from HBM to SBUF.
    
    down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    Load directly into dst_weight with scalar AP.
    scalar_offset=block_expert with indirect_dim=0 means access starts at 
    block_expert * (p_I * n_total_I512_tile * H)
    """

    stride_p_I = prj_cfg.n_total_I512_tile * dims.H
    nisa.dma_copy(
        src=inps.down_proj_weight.ap(
            pattern=[[stride_p_I, dims.p_I], [dims.H, prj_cfg.n_total_I512_tile], [1, dims.H]],
            offset=prj_cfg.prg_id * prj_cfg.H_sharded,
            scalar_offset=block_expert,
            indirect_dim=0,
        ),
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
        (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded), dtype=inps.down_proj_scale.dtype, buffer=nl.sbuf
    )  # original nl.uint8
    # Memset weight scale if input weight scale HBM does not pad on par dim
    if dims.p_I != _pmax:
        nisa.memset(dst=down_scale_sb[:, prj_cfg.n_total_I512_tile - 1, :], value=0)
    kernel_assert(
        down_scale_sb.shape == (128, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded), f"Got {down_scale_sb.shape}"
    )

    """
    Construct a vector DGE index to index into E*16
        if block_expert == 0, we want something like this (tranposed to the P dimension)
        [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]  
    
        if block_expert == 3, we want something like this
        [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]  
        i.e, basically the same as above, with offset 16*3 = 48
    """
    down_scale_view = inps.down_proj_scale.reshape((scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3]))

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

    # # Use AP for dst to match src pattern and avoid issues with dynamic blocks
    nisa.dma_copy(
        src=down_scale_view.ap(
            pattern=[[down_scale_stride_dim0, _pmax], [dims.H, prj_cfg.n_total_I512_tile], [1, prj_cfg.H_sharded]],
            offset=prj_cfg.prg_id * prj_cfg.H_sharded,
            vector_offset=token_indices_on_p.ap(
                [[1, _pmax], [1, 1]],
                offset=0,
            ),
            indirect_dim=0,
        ),
        dst=down_scale_sb[:128, : prj_cfg.n_total_I512_tile, : prj_cfg.H_sharded],
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


def compute_one_block(
    block_idx: int,
    next_block_idx: int,
    buffers: SharedBuffers,
    dims: BWMMMXDimensionSizes,
    inps: InputTensors,
    outs: OutputTensors,
    dbg_tensors: DebugTensors,
    kernel_cfg: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    shard_id: Any,
    is_dummy: bool = False,
    is_dynamic: bool = False,
    is_first_block: bool = False,
):
    """
    Process one block through complete MoE MLP pipeline.

    Executes gate projection, up projection, activation, and down projection
    for a single block with MXFP4 quantization and expert routing.

    Args:
        block_idx (int): Current block index.
        next_block_idx (int): Next block index for prefetching (None if last).
        buffers (SharedBuffers): Shared computation buffers.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        dbg_tensors (DebugTensors): Debug tensors (if enabled).
        kernel_cfg (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        shard_id (Any): Current shard identifier.
        is_dummy (bool): Whether this is a dummy block (for load balancing).
        is_dynamic (bool): Whether from dynamic loop.

    Returns:
        None: Writes results to outs.output.

    Notes:
        - Loads expert weights and scales
        - Prefetches next block hidden states if next_block_idx provided
        - Applies gate/up projections with optional clamping
        - Applies activation function (SiLU or Swish)
        - Computes down projection
        - Scales by expert affinity and accumulates
        - Dummy blocks have zero affinity for load balancing

    Pseudocode:
        block_expert = load_block_expert(block_to_expert, block_idx)

        if next_block_idx is not None:
            compute_hidden_index_vector(inps, buffers, next_block_idx, dims, skip_dma, is_dynamic)

        if not is_dynamic:
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)

        reshape buffers.hidden_qtz_sb and hidden_scale_sb

        gate_and_up_weights, gate_and_up_scales, gup_bias = load_gup_weights_scales_mx4(inps, block_expert, dims, prj_cfg, skip_dma)
        down_scale_sb, down_bias_sb = load_down_proj_weights_mx4(inps, block_expert, buffers.down_weight_qtz, dims, prj_cfg, skip_dma)

        token_indices_2D = load_token_indices(token_position_to_id, block_idx, B, n_B128_tiles)
        expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices_2D, block_expert, E, B//128, compute_dtype, skip_dma)
        block_old = load_prev_block(output, token_indices_2D, block_old, B//128, compute_dtype, shard_id, skip_dma)

        gate_proj_out = gate_up_proj_mxfp4_tp(hidden_qtz_sb, hidden_scale_sb, gate_weights, gate_scales, gate_bias, cfg)
        gate_proj_out = clamp(gate_proj_out, gate_clamp_lower_limit, gate_clamp_upper_limit)

        up_proj_out = gate_up_proj_mxfp4_tp(hidden_qtz_sb, hidden_scale_sb, up_weights, up_scales, up_bias, cfg)
        up_proj_out = clamp(up_proj_out, up_clamp_lower_limit, up_clamp_upper_limit)

        if next_block_idx is not None:
            load_and_quantize_hidden_states(
                inps, next_block_idx, buffers, dims, kernel_cfg, prj_cfg, is_dynamic, USE_DMA_TRANSPOSE
            )

        if activation_function == SiLU:
            gate_proj_out = silu(gate_proj_out)
        elif activation_function == Swish:
            gate_proj_out = gelu_apprx_sigmoid(gate_proj_out)

        intermediate_state = gate_proj_out * up_proj_out
        block_new = down_proj_mxfp4(intermediate_state, down_weight, down_scale, down_bias, cfg)

        for n in range(B // 128):
            if is_dummy:
                expert_affinity[n] = 0
            block_new[:, n, :] *= expert_affinity[n]
            block_new[:, n, :] += block_old[:, n, :]
            dma_copy block_new[:, n, :] to output[shard_id, token_indices_2D[:, n], :]
    """
    block_expert = load_block_expert(inps.block_to_expert, block_idx)

    # Store debug hidden states BEFORE prefetching next block (which overwrites block_hidden_states_T)
    if DBG_KERNEL and block_idx == 0:
        nisa.dma_copy(dst=dbg_tensors.hidden_states[:128, :, :, :], src=buffers.block_hidden_states_T[:128, :, :, :])

    if next_block_idx is not None:
        compute_hidden_index_vector(
            inps, buffers, next_block_idx, dims, kernel_cfg.skip_dma, is_block_idx_dynamic=is_dynamic
        )
    # quantize prefetched data. Note that online quantize can only quantize to fp8
    # only quantize here if it is a static block. for dynamic block we quantize immediately after fetching
    if not is_dynamic:
        quantize_block_hidden_state_T(buffers, prj_cfg, dims)

    buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
    buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))

    gate_and_up_weights, gate_and_up_scales, gup_bias, gup_token_indices_on_p, gup_n_quadrants_needed = (
        load_gup_weights_scales_mx(inps, block_expert, dims, prj_cfg=prj_cfg, skip_dma=kernel_cfg.skip_dma)
    )

    down_scale_sb, down_bias_sb = load_down_proj_weights_mx(
        inps,
        block_expert,
        buffers.down_weight_qtz,
        dims,
        prj_cfg,
        kernel_cfg.skip_dma,
        gup_token_indices_on_p,
        gup_n_quadrants_needed,
    )
    # TensorView handles broadcast in down_proj_mxfp4
    flatten_free_dim = prj_cfg.n_total_I512_tile * dims.B * _q_width

    if is_dynamic:
        token_indices_2D = load_token_indices_dynamic_block(
            inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles, skip_dma=kernel_cfg.skip_dma
        )
    else:
        token_indices_2D = load_token_indices(inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles)

    kernel_assert(
        token_indices_2D.shape == (128, dims.n_B128_tiles),
        f"Expect token_indices_2D to have shape (128, {dims.n_B128_tiles}), got {token_indices_2D.shape}",
    )

    # Skip load for first block (output is already zero-initialized)
    if not is_first_block:
        block_old = load_prev_block(
            outs.output,
            token_indices_2D,
            buffers.block_old,
            dims.B // 128,
            kernel_cfg.compute_dtype,
            shard_id,
            kernel_cfg.skip_dma,
        )

    expert_affinity = calculate_expert_affinities(
        inps.expert_affinities_masked,
        token_indices_2D,
        block_expert,
        dims.E,
        dims.B // 128,
        nl.float32,
        kernel_cfg.skip_dma,
    )

    """
    GATE PROJECTION
    """
    gup_weights_reshaped = gate_and_up_weights.reshape((_pmax, 2 * prj_cfg.n_H512_tile_sharded, dims.I))
    gup_scales_reshaped = gate_and_up_scales.reshape((_pmax, 2 * prj_cfg.n_H512_tile_sharded, dims.I))
    gup_bias_reshaped = gup_bias.reshape((_pmax, 2 * prj_cfg.n_total_I512_tile, _q_width))

    # Use TensorView to slice bias without tensor_copy
    gup_bias_view = TensorView(gup_bias_reshaped)
    gate_bias_view = gup_bias_view.slice(dim=1, start=0, end=prj_cfg.n_total_I512_tile)

    gate_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=TensorView(buffers.hidden_qtz_sb),
        hidden_scale_sb=TensorView(buffers.hidden_scale_sb),
        weight_qtz=TensorView(gup_weights_reshaped).slice(1, 0, prj_cfg.n_H512_tile_sharded),
        weight_scale=TensorView(gup_scales_reshaped).slice(1, 0, prj_cfg.n_H512_tile_sharded),
        bias_sb=gate_bias_view,
        cfg=prj_cfg,
    )

    gate_proj_out_sb = gate_proj_out_sb.reshape((_pmax, flatten_free_dim))
    # clipping gate
    nisa.tensor_scalar(
        gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        op0=nl.minimum if kernel_cfg.gate_clamp_upper_limit is not None else None,
        operand0=kernel_cfg.gate_clamp_upper_limit,
        op1=nl.maximum if kernel_cfg.gate_clamp_lower_limit is not None else None,
        operand1=kernel_cfg.gate_clamp_lower_limit,
    )

    # Debug: Store gate projection output (before activation)
    if DBG_KERNEL and block_idx == 0:
        nisa.dma_copy(
            dst=dbg_tensors.gate_proj[:_pmax, :flatten_free_dim], src=gate_proj_out_sb[:_pmax, :flatten_free_dim]
        )

    """
    UP PROJECTION
    """
    # Use TensorView to slice bias without tensor_copy
    up_bias_view = gup_bias_view.slice(dim=1, start=prj_cfg.n_total_I512_tile, end=2 * prj_cfg.n_total_I512_tile)

    up_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=TensorView(buffers.hidden_qtz_sb),
        hidden_scale_sb=TensorView(buffers.hidden_scale_sb),
        weight_qtz=TensorView(gup_weights_reshaped).slice(
            1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
        ),
        weight_scale=TensorView(gup_scales_reshaped).slice(
            1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
        ),
        bias_sb=up_bias_view,
        cfg=prj_cfg,
    )

    up_proj_out_sb = up_proj_out_sb.reshape((_pmax, flatten_free_dim))
    # clipping up
    nisa.tensor_scalar(
        up_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        up_proj_out_sb[0:_pmax, 0:flatten_free_dim],
        op0=nl.minimum if kernel_cfg.up_clamp_upper_limit is not None else None,
        operand0=kernel_cfg.up_clamp_upper_limit,
        op1=nl.maximum if kernel_cfg.up_clamp_lower_limit is not None else None,
        operand1=kernel_cfg.up_clamp_lower_limit,
    )

    # Debug: Store up projection output (after clipping)
    if DBG_KERNEL and block_idx == 0:
        nisa.dma_copy(dst=dbg_tensors.up_proj[:_pmax, :flatten_free_dim], src=up_proj_out_sb[:_pmax, :flatten_free_dim])

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
        else:
            load_hidden_states_mx(
                inps,
                dims,
                kernel_cfg.skip_dma,
                token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                block_hidden_states=buffers.block_hidden_states,
                use_dma_transpose=False,
            )
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

        if is_dynamic:
            # quantize after fetching for dynamic blocks
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)

    else:
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
    intermediate_state_sb = intermediate_state_sb.reshape((_pmax, prj_cfg.n_total_I512_tile, dims.B, _q_width))

    """
    DOWN PROJECTION
    """

    block_new = down_projection_mx_shard_H(
        inter_sb=intermediate_state_sb,
        weight=buffers.down_weight_qtz,
        weight_scale=down_scale_sb,
        bias_sb=down_bias_sb,
        cfg=prj_cfg,
    )

    # Debug: Store down projection output (before expert scaling and accumulation)
    if DBG_KERNEL and block_idx == 0:
        for n in range(dims.B // 128):
            nisa.dma_copy(dst=dbg_tensors.down_proj[:_pmax, n, : dims.H], src=block_new[:_pmax, n, : dims.H])

    for n in range(dims.B // 128):
        if is_dummy:
            zeros = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(zeros, value=0.0)
            nisa.tensor_copy(dst=expert_affinity[n][:, :], src=zeros, engine=nisa.vector_engine)

        nisa.tensor_scalar(
            dst=block_new[0:_pmax, n, 0 : dims.H],
            data=block_new[0:_pmax, n, 0 : dims.H],
            op0=nl.multiply,
            operand0=expert_affinity[n][0:_pmax, 0:1],
        )

        # accumulate
        if not is_first_block:
            nisa.tensor_tensor(
                dst=block_new[0:_pmax, n, 0 : dims.H],
                data1=block_new[0:_pmax, n, 0 : dims.H],
                op=nl.add,
                data2=block_old[0:_pmax, n, 0 : dims.H],
            )

        """
        Scatter write block results to output tensor.
        
        output shape: (num_shards, T, H)
        block_new shape: (_pmax, n_B128_tiles, H)
        Each partition writes H elements to output[shard_id, token_idx, :]
        Use AP for dst with vector_offset for indirect scatter, direct slice for src
        """
        T = outs.output.shape[-2]
        shard_offset = shard_id * T * dims.H

        # (128, 2)
        block_token_mapping = token_indices_2D.ap(
            [[dims.n_B128_tiles, _pmax], [1, 1]],
            offset=n,
        )

        num_shards = outs.output.shape[0]

        nisa.dma_copy(
            dst=outs.output.reshape((num_shards * T, 1, dims.H)).ap(
                pattern=[[dims.H, _pmax], [1, 1], [1, dims.H]],
                offset=shard_offset,
                vector_offset=block_token_mapping,  # (128, 1) -> T
                indirect_dim=0,
            ),
            src=block_new[0:_pmax, n, 0 : dims.H],
            oob_mode=oob_mode.skip if kernel_cfg.skip_dma.skip_token else oob_mode.error,  # Set to True in current test
        )


def process_static_blocks(
    dims: BWMMMXDimensionSizes,
    configs: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    inps: InputTensors,
    outs: OutputTensors,
    dbg_tensors: DebugTensors,
    buffers: SharedBuffers,
    num_static_blocks: int,
):
    """
    Process static (non-padded) blocks with prefetching optimization.

    Iterates through known non-padded blocks with double-buffering to overlap
    computation and data loading.

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration.
        configs (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        dbg_tensors (DebugTensors): Debug tensors.
        buffers (SharedBuffers): Shared buffers.
        num_static_blocks (int): Number of static blocks to process.

    Returns:
        None: Processes blocks and writes to outs.output.

    Notes:
        - Distributes blocks across shards evenly
        - Prefetches next block while processing current
        - Handles odd/even block counts differently
        - Last block has no prefetch
        - Shard 0 processes dummy block when N is odd

    Pseudocode:
        n_blocks_per_shard = num_static_blocks // num_shards
        r_block = num_static_blocks % num_shards
        first_block_idx = n_blocks_per_shard * shard_id

        load_and_quantize_hidden_states(inps, first_block_idx, buffers, dims, configs, prj_cfg)

        if num_static_blocks % num_shards == 0:
            for per_shard_block_idx in range(n_blocks_per_shard - 1):
                block_idx = per_shard_block_idx + n_blocks_per_shard * shard_id
                compute_one_block(block_idx, block_idx+1, buffers, dims, inps, outs, dbg_tensors, configs, prj_cfg, shard_id)
            last_block_idx = n_blocks_per_shard * shard_id + n_blocks_per_shard - 1
            compute_one_block(last_block_idx, None, buffers, dims, inps, outs, dbg_tensors, configs, prj_cfg, shard_id)
        else:
            for per_shard_block_idx in range(n_blocks_per_shard):
                block_idx = per_shard_block_idx + n_blocks_per_shard * shard_id
                compute_one_block(block_idx, block_idx+1, buffers, dims, inps, outs, dbg_tensors, configs, prj_cfg, shard_id)
            is_dummy = (shard_id == 0)
            remainder_block_idx = num_static_blocks - 1
            compute_one_block(remainder_block_idx, None, buffers, dims, inps, outs, dbg_tensors, configs, prj_cfg, shard_id, is_dummy)
    """
    n_blocks_per_shard, r_block = divmod(num_static_blocks, dims.num_shards)
    # prefetch the first block of each core
    first_block_idx = n_blocks_per_shard * dims.shard_id

    if USE_DMA_TRANSPOSE:
        load_hidden_states_mx(
            inps,
            dims,
            configs.skip_dma,
            block_idx=first_block_idx,
            block_hidden_states_T=buffers.block_hidden_states_T,
            use_dma_transpose=True,
        )
    else:
        compute_hidden_index_vector(inps, buffers, first_block_idx, dims, configs.skip_dma, False)
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
    # NOTE: we do not quantize here because we will do it in the beginning of each static block

    # 2 different code paths to handle N odd and N even to explicitly handle prefetching
    if num_static_blocks % dims.num_shards == 0:
        """
        N is even
        In this case, in each core we can only do prefetch in the first n_blocks_per_shard - 1 blocks
        """
        kernel_assert(r_block == 0, "Expected r_block to be 0 for even number of static blocks")
        for per_shard_block_idx in nl.sequential_range(n_blocks_per_shard - 1):
            block_idx = per_shard_block_idx + n_blocks_per_shard * dims.shard_id
            compute_one_block(
                block_idx,
                block_idx + 1,
                buffers,
                dims,
                inps,
                outs,
                dbg_tensors,
                kernel_cfg=configs,
                prj_cfg=prj_cfg,
                shard_id=dims.shard_id,
                is_first_block=(per_shard_block_idx == 0),
            )

        last_block_idx = n_blocks_per_shard * dims.shard_id + n_blocks_per_shard - 1
        compute_one_block(
            last_block_idx,
            None,
            buffers,
            dims,
            inps,
            outs,
            dbg_tensors,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            shard_id=dims.shard_id,
        )

    else:
        """
        N is odd
        In this case, each core can do prefetch the first n_blocks_per_shard
        """
        kernel_assert(r_block == 1, "Expected r_block to be 1 for odd number of static blocks")
        for per_shard_block_idx in nl.sequential_range(n_blocks_per_shard):
            block_idx = per_shard_block_idx + n_blocks_per_shard * dims.shard_id
            compute_one_block(
                block_idx,
                block_idx + 1,
                buffers,
                dims,
                inps,
                outs,
                dbg_tensors,
                kernel_cfg=configs,
                prj_cfg=prj_cfg,
                shard_id=dims.shard_id,
                is_first_block=(per_shard_block_idx == 0),
            )

        # one last remaining block
        # core 1 should have the data for this block prefetched in the previous loop
        # core 0 should process a dummy block. The way we signal the dummy is we memset the expert affinity to be 0
        is_dummy = dims.shard_id == 0
        remainder_block_idx = num_static_blocks - 1
        compute_one_block(
            remainder_block_idx,
            None,
            buffers,
            dims,
            inps,
            outs,
            dbg_tensors,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            shard_id=dims.shard_id,
            is_dummy=is_dummy,
        )


def process_dynamic_blocks(
    dims: BWMMMXDimensionSizes,
    configs: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    inps: InputTensors,
    outs: OutputTensors,
    dbg_tensors: DebugTensors,
    buffers: SharedBuffers,
    num_static_blocks: int,
    num_dynamic_blocks: int,
):
    """
    Process dynamic (potentially padded) blocks using condition vector.

    Iterates through blocks using runtime condition checks to skip padded blocks,
    processing two blocks per iteration (ping-pong between shards).

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration.
        configs (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        inps (InputTensors): Input tensors with conditions vector.
        outs (OutputTensors): Output tensors.
        dbg_tensors (DebugTensors): Debug tensors.
        buffers (SharedBuffers): Shared buffers with cond and index.
        num_static_blocks (int): Starting block index (after static blocks).
        num_dynamic_blocks (int): Number of dynamic blocks to process.

    Returns:
        None: Processes blocks and writes to outs.output.

    Notes:
        - Uses while loop with runtime condition checking
        - Processes blocks in tandem (2 at a time across shards)
        - Prefetches next block based on min(block_idx+2, N-1)
        - Quantizes immediately after fetching for dynamic blocks
        - Condition vector has length N+2
        - Terminates when condition becomes false

    Pseudocode:
        assert num_static_blocks + num_dynamic_blocks == N

        first_block_idx = num_static_blocks + shard_id
        load_and_quantize_hidden_states(inps, first_block_idx, buffers, dims, configs, prj_cfg)

        reg = register_alloc()
        register_load(reg, buffers.cond)

        while reg:
            tandem_block_idx = buffers.index
            dyn_block_idx = tandem_block_idx + shard_id
            dyn_next_block_idx = min(dyn_block_idx + 2, N - 1)

            compute_one_block(dyn_block_idx, dyn_next_block_idx, buffers, dims, inps, outs, dbg_tensors, configs, prj_cfg, shard_id, is_dynamic=True)

            tandem_next_block_idx = tandem_block_idx + 2
            cond_next = load conditions[tandem_next_block_idx]

            buffers.index = tandem_next_block_idx
            buffers.cond = cond_next
            register_load(reg, buffers.cond)
    """
    kernel_assert(
        num_static_blocks + num_dynamic_blocks == dims.N,
        f"num_static_blocks + num_dynamic_blocks must equal N, got {num_static_blocks} + {num_dynamic_blocks}!= {dims.N} ",
    )

    logger.info(f"Start looping over dynamic blocks {num_static_blocks} to {dims.cond_vec_len} - 1")

    logger.info("Prefetch first block for each core")
    first_block_idx = num_static_blocks + dims.shard_id
    load_and_quantize_hidden_states(
        inps, first_block_idx, buffers, dims, configs, prj_cfg, use_dma_transpose=USE_DMA_TRANSPOSE
    )

    reg = nisa.register_alloc()
    nisa.register_load(reg, buffers.cond)
    while reg:
        """
        Iterate 2 blocks at a time in ping-pong fashion.
        
        Example: if dynamic blocks start at block 15:
        - tandem_block_idx: 15, 17, 19, 21, ...
        - block_idx on core 0: 15, 17, 19, ...
        - block_idx on core 1: 16, 18, 20, ...
        """
        # Compute dyn_block_idx = buffers.index + shard_id directly
        dyn_block_idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dyn_block_idx, buffers.index, op0=nl.add, operand0=dims.shard_id)

        # Compute dyn_next_block_idx = min(dyn_block_idx + 2, N-1) directly
        dyn_next_block_idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dyn_next_block_idx, dyn_block_idx, op0=nl.add, operand0=2, op1=nl.minimum, operand1=dims.N - 1
        )
        compute_one_block(
            dyn_block_idx,
            dyn_next_block_idx,
            buffers,
            dims,
            inps,
            outs,
            dbg_tensors,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            shard_id=dims.shard_id,
            is_dynamic=True,
            is_first_block=False,
        )

        # Compute tandem_next_block_idx = buffers.index + 2 directly
        nisa.tensor_scalar(buffers.index, buffers.index, op0=nl.add, operand0=2)

        # cond_next = nl.ndarray((1, 1), dtype=inps.conditions.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=buffers.cond,
            src=inps.conditions.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=buffers.index, indirect_dim=0),
        )

        # Reload register for next iteration
        nisa.register_load(reg, buffers.cond)
