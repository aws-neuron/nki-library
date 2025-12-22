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
This kernel implements blockwise matrix multiplication for Mixture of Experts (MoE) layers using MXFP4 quantization with block-level sharding. The implementation shards gate/up projections over the intermediate dimension and block accumulation over the batch dimension, processing all blocks without distinguishing between padded and non-padded blocks.
"""

from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ...mlp.mlp_tkg.down_projection_mx_shard_H import down_projection_mx_shard_H
from ...mlp.mlp_tkg.gate_up_mx_shard_H import gate_up_projection_mx_tp_shard_H
from ...mlp.mlp_tkg.projection_mx_constants import (
    ProjConfig,
    _pmax,
    _q_height,
    _q_width,
)
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_nl_act_fn_from_type, get_program_sharding_info, reduce
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .bwmm_shard_on_I import DebugTensors, OutputTensors
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


@dataclass
class InputTensors(nl.NKIObject):
    token_position_to_id: nl.ndarray
    block_to_expert: nl.ndarray
    hidden_states: nl.ndarray
    gate_up_proj_weight: nl.ndarray
    gate_and_up_proj_bias: nl.ndarray
    down_proj_bias: nl.ndarray
    down_proj_weight: nl.ndarray
    expert_affinities_masked: nl.ndarray
    gate_up_proj_scale: nl.ndarray
    down_proj_scale: nl.ndarray
    p_gup_idx_vector: nl.ndarray = None
    p_down_idx_vector: nl.ndarray = None
    gup_scales_sb: nl.ndarray = None
    activation_bias: nl.ndarray = None
    conditions: nl.ndarray = None


@dataclass
class SharedBuffers(nl.NKIObject):
    block_hidden_states: nl.ndarray
    block_hidden_states_T: nl.ndarray
    token_4_H_indices_on_p: nl.ndarray

    hidden_qtz_sb: nl.ndarray
    hidden_scale_sb: nl.ndarray
    down_weight_qtz: nl.ndarray
    block_old: nl.ndarray
    # for dynamic loop
    cond: nl.ndarray
    index: nl.ndarray

    down_scale_sb: nl.ndarray = None


@dataclass
class BWMMMXFP4DimensionSizes(nl.NKIObject):
    B: int
    H: int
    T: int
    E: int
    N: int
    I: int
    cond_vec_len: int
    TILESIZE: int = 512

    def __post_init__(self):
        _, num_shards, shard_id = get_program_sharding_info()
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.MODULO_FACTOR = self.B // self.TILESIZE
        self.n_B128_tiles = div_ceil(self.B, _pmax)
        self.hidden_sbuf_expected_shape = (32 * 4, self.B // 32, div_ceil(self.H, 512), 16 * 8)
        self.p_I = _pmax if self.I > 512 else self.I // 4


@dataclass
class BWMMMXFP4Configs(nl.NKIObject):
    scaling_mode: ExpertAffinityScaleMode
    skip_dma: SkipMode
    compute_dtype: Any
    weight_dtype: Any
    io_dtype: Any
    is_tensor_update_accumulating: bool
    use_dynamic_while: bool
    n_static_blocks: int
    linear_bias: bool
    activation_function: ActFnType
    fuse_gate_and_up_load: bool
    gate_clamp_upper_limit: Optional[float]
    gate_clamp_lower_limit: Optional[float]
    up_clamp_upper_limit: Optional[float]
    up_clamp_lower_limit: Optional[float]
    qtz_dtype: Any


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
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
):
    """
    Blockwise MXFP4 MoE kernel, decorated. Use as standalone kernel.

    The blockwise matrix multiplication (matmul) kernel implements a Mixture of Experts (MoE)
    layer at a block granularity, offering an alternative to token dropping approaches.
    This method assumes that tokens have already been assigned to blocks, as specified
    by the user through the token_position_to_id parameter. This kernel shards the gate/up projection
    over the I dimension, and shards the block accumulation over the B dimension.
    This kernel loops over all blocks, without considering they are padded or non-padded blocks.

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
        gate_up_proj_weight (nl.ndarray): Tensor of concatenated gate and up projection weights on HBM (E, H, 2, I)
        down_proj_weight (nl.ndarray): Tensor of down projection weights on HBM (E, I, H)
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

        # Arguments for fp8 dequantization
        gate_up_proj_scale: nl.ndarray = None. A tensor of shape [E, 1, 2 * I]
        down_proj_scale: nl.ndarray = None. A tensor of shape [E, 1, H]

        # Unsupported output tensors. Please set to None.
        gate_up_activations_T: nl.ndarray = None. Currently not supported.
        down_activations: nl.ndarray = None. Currently not supported

        # meta parameters
        activation_function: one of the Enum in neuronxcc.nki._pre_prod_kernels.ActFnType.
                              Indicate what activation function to use in the MLP block
        skip_dma: SkipMode = SkipMode(False, False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating: bool. Indicate whether we need to accumulate the results over multiple blocks
        expert_affinities_scaling_mode: one of the Enum in neuronxcc.nki._pre_prod_kernels.ExpertAffinityScaleMode.
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
        dims = BWMMMXFP4DimensionSizes(T, H, B, E, N, I, cond_vec_len)
        prj_cfg = ProjConfig(H, I, B, force_lnc1=True, n_prgs=1, prg_id=0)
        configs = BWMMMXFP4Configs(...)

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
    dims = BWMMMXFP4DimensionSizes(T=T, H=H, B=B, E=E, N=N, I=I, cond_vec_len=cond_vec_len)

    prj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=dims.B,
        force_lnc1=True,
        n_prgs=1,
        prg_id=0,
    )

    # torch/xla doesnt support passing mxfp4 tensors to kernel
    alternative_input_mxfp4_dtypes = [nl.uint16, nl.int16, nl.float16]
    if gate_up_proj_weight.dtype in alternative_input_mxfp4_dtypes:
        gate_up_proj_weight = gate_up_proj_weight.view(nl.float4_e2m1fn_x4)
    if down_proj_weight.dtype in alternative_input_mxfp4_dtypes:
        down_proj_weight = down_proj_weight.view(nl.float4_e2m1fn_x4)

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

    configs = BWMMMXFP4Configs(
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
    # Allocate buffers for prefetching
    # hidden_sbuf_expected_shape = (32 * 4, dims.B // 32, mx4_prj_cfg.n_H512_tile, 16 * 8)

    # Current block buffers
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
            n_dynamic_blocks_local = n_dynamic_blocks
            n_static_blocks = dims.N - n_dynamic_blocks_local

        nisa.dma_copy(
            dst=buffers.cond.ap(pattern=[[1, 1], [1, 1]]),
            src=inps.conditions.ap(pattern=[[1, 1], [1, 1]], offset=n_static_blocks),
        )

        nisa.memset(dst=buffers.index[0, 0], value=n_static_blocks)

        print(f"Processing {n_static_blocks} static blocks, {n_dynamic_blocks_local} dynamic blocks")
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


def sbuf_layout_adapter(
    src: nl.ndarray,
    dst: nl.ndarray,
    dims: BWMMMXFP4DimensionSizes,
    use_dma_tp: bool = False,
):
    """
    Transpose tensor layout in SBUF to swap outermost and innermost dimensions.

    Performs layout transformation from [32_T * 4_H (P), T/32, H/512, 16_H * 8_H]
    to [16_H * 8_H(P), H/512, T/32, 32_T * 4_H] using either DMA transpose or
    nc_transpose.

    Args:
        src (nl.ndarray): Source 4D tensor in SBUF of shape [32_T * 4_H (P), T/32, H/512, 16_H * 8_H].
        dst (nl.ndarray): Destination 4D tensor in SBUF of shape [16_H * 8_H(P), H/512, T/32, 32_T * 4_H].
        dims (BWMMMXFP4DimensionSizes): Dimension configuration containing B and H.
        use_dma_tp (bool): If True, use DMA transpose; if False, use nc_transpose method.

    Returns:
        None: Modifies dst tensor in-place.

    Notes:
        - Performs T/32 * H/512 transpose operations
        - DMA transpose method is faster but may have hardware limitations
        - nc_transpose method uses multi-buffering with N_PSUM_BANKS for efficient pipelining
        - Reshapes dst tensor internally for efficient memory access patterns

    Pseudocode:
        T_div_32 = B // 32
        H_div_512 = H // 512
        n_transposes_per_bank = min(8, T_div_32)
        n_PSUM_banks = T_div_32 // n_transposes_per_bank

        if use_dma_tp:
            tmp_sbuf = dma_transpose(src, axes=(3,1,2,0))
            for H_div_512_idx in range(H_div_512):
                for B_div_32_idx in range(T_div_32):
                    dst[:, H_div_512_idx, B_div_32_idx, :] = tmp_sbuf[:, B_div_32_idx, H_div_512_idx, :]
        else:
            dst = reshape dst to [128, H_div_512, n_PSUM_banks, n_transposes_per_bank*128]
            for H_div_512_idx in range(H_div_512):
                for bank in range(n_PSUM_banks):
                    tmp_res = allocate in PSUM
                    for idx in range(n_transposes_per_bank):
                        B_div_32_idx = bank * n_transposes_per_bank + idx
                        tmp_res[:, idx*128:(idx+1)*128] = nc_transpose(src[:, B_div_32_idx, H_div_512_idx, :])
                    dst[:, H_div_512_idx, bank, :] = tmp_res
            dst = reshape dst to [128, H_div_512, T_div_32, 128]
    """

    src_sbuf = src

    T_div_32 = dims.B // 32
    H_div_512 = dims.H // 512

    n_transposes_per_bank = min(8, T_div_32)

    n_PSUM_banks = T_div_32 // n_transposes_per_bank  # each bank can hold 8 BF16 128x128 transpose

    if use_dma_tp:
        tmp_sbuf = nl.ndarray((_pmax, T_div_32, H_div_512, _pmax), dtype=src_sbuf.dtype)
        nisa.dma_transpose(dst=tmp_sbuf[:, :, :, :], src=src_sbuf[:, :, :, :], axes=(3, 1, 2, 0))

        for H_div_512_idx in range(H_div_512):
            for B_div_32_idx in range(T_div_32):
                dst[0:_pmax, H_div_512_idx, B_div_32_idx, :_pmax] = tmp_sbuf[
                    0:_pmax, B_div_32_idx, H_div_512_idx, :_pmax
                ]
    else:
        dst = dst.reshape((_pmax, H_div_512, n_PSUM_banks, n_transposes_per_bank * 32 * 4))
        for H_div_512_idx in range(H_div_512):
            for bank in range(n_PSUM_banks):
                # each bank will store the results of 8 transpose
                tmp_res = nl.ndarray((_pmax, n_transposes_per_bank * _pmax), dtype=src.dtype, buffer=nl.psum)

                for idx in range(n_transposes_per_bank):
                    B_div_32_idx = bank * n_transposes_per_bank + idx
                    # transpose [32_T * 4_H, 16_H*8_H] -> [16_H*8_H, 32_T * 4_H]
                    nisa.nc_transpose(
                        dst=tmp_res[:_pmax, idx * _pmax : (idx + 1) * _pmax],
                        data=src_sbuf[0:_pmax, B_div_32_idx, H_div_512_idx, 0:_pmax],
                    )

                # evict the full bank
                nisa.tensor_copy(
                    dst[0:_pmax, H_div_512_idx, bank, 0 : n_transposes_per_bank * _pmax],
                    tmp_res[:_pmax, : n_transposes_per_bank * _pmax],
                    engine=nisa.scalar_engine,
                )

        dst = dst.reshape((_pmax, H_div_512, T_div_32, 32 * 4))


def load_old_block_bwmm_mxfp4(output, token_indices, block_old, NUM_TILES, dtype, shard_id, skip_dma: SkipMode):
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


def check_kernel_compatibility(dims: BWMMMXFP4DimensionSizes, configs: BWMMMXFP4Configs):
    """
    Validate kernel configuration and dimension compatibility.

    Performs comprehensive validation of kernel parameters to ensure they meet
    hardware constraints and implementation requirements before execution.

    Args:
        dims (BWMMMXFP4DimensionSizes): Dimension configuration containing B, H, I, N,
            num_shards, and cond_vec_len.
        configs (BWMMMXFP4Configs): Kernel configuration containing is_tensor_update_accumulating
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


def load_gup_weights_scales_mx4(
    inps: InputTensors, block_expert: nl.ndarray, dims: BWMMMXFP4DimensionSizes, prj_cfg: ProjConfig, skip_dma: SkipMode
):
    """
    Load gate and up projection weights, scales, and biases for current expert.

    Loads MXFP4 quantized weights, uint8 scales, and biases for both gate and up
    projections from HBM to SBUF for the expert assigned to the current block.

    Args:
        inps (InputTensors): Input tensors containing gate_up_proj_weight of shape
            [E, 128, 2, n_H512_tile, I], gate_up_proj_scale, gate_and_up_proj_bias,
            and buffers for scales and index vectors.
        block_expert (nl.ndarray): Expert index for current block, shape [1, 1].
        dims (BWMMMXFP4DimensionSizes): Dimension configuration with I, H.
        prj_cfg (ProjConfig): Projection configuration with n_H512_tile_sharded, I.
        skip_dma (SkipMode): DMA skip configuration for weight loading.

    Returns:
        tuple: (gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb)
            - gup_weights_qtz_sb (nl.ndarray): Quantized weights [128, 2, n_H512_tile_sharded, I]
            - gup_scales_sb (nl.ndarray): Dequantization scales [128, 2, n_H512_tile_sharded, I]
            - gup_bias_sb (nl.ndarray): Bias values [128, 2, n_total_I512_tile, 128]

    Notes:
        - Uses indirect DGE with block_expert for expert selection
        - Constructs partition index vector for scale loading with proper offsets
        - Pads bias to 512 when I < 512 for alignment
        - Scales are loaded with zero-padding for out-of-bounds partitions
        - Gate and up projections share weight buffer (dimension 1 has size 2)

    Pseudocode:
        gup_weights_qtz_sb = allocate [128, 2, n_H512_tile_sharded, I] in SBUF
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

    n_quadrants_needed = prj_cfg.H0 // 32
    for i_quad in nl.static_range(n_quadrants_needed):
        # arange_4P = nisa.iota(nl.arange(i_quad*4, (i_quad+1)*4)[:, None], dtype = nl.float32)
        arange_4P = nl.ndarray((4, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(arange_4P, [[1, 1]], offset=i_quad * 4, channel_multiplier=1)
        block_expert_broadcast = nl.ndarray((4, 1), dtype=block_expert.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(src=block_expert, dst=block_expert_broadcast)
        nisa.activation(
            dst=inps.p_gup_idx_vector[i_quad * 32 : i_quad * 32 + 4],
            data=block_expert_broadcast,
            op=nl.copy,
            scale=float(scale_shape[1]),
            bias=arange_4P,
        )

    token_indices_on_p = nl.ndarray(inps.p_gup_idx_vector.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(token_indices_on_p, inps.p_gup_idx_vector, engine=nisa.scalar_engine)
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

    return gup_weights_qtz_sb, inps.gup_scales_sb, gup_bias_sb


def load_down_proj_weights_mx4(
    inps: InputTensors,
    block_expert: nl.ndarray,
    dst_weight: nl.ndarray,
    dims: BWMMMXFP4DimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
):
    """
    Load down projection weights, scales, and biases for current expert.

    Loads MXFP4 quantized weights and uint8 scales for down projection from HBM
    to SBUF, constructing partition index vectors for proper expert selection.

    Args:
        inps (InputTensors): Input tensors containing down_proj_weight [E, p_I, n_total_I512_tile, H],
            down_proj_scale, down_proj_bias, and index vector buffer.
        block_expert (nl.ndarray): Expert index for current block, shape [1, 1].
        dst_weight (nl.ndarray): Destination buffer for weights in SBUF.
        dims (BWMMMXFP4DimensionSizes): Dimension configuration with I, H, p_I.
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
    DOWN WEIGHTS
    """

    # down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    # Load directly into dst_weight with scalar AP
    # scalar_offset=block_expert with indirect_dim=0 means access starts at block_expert * (p_I * n_total_I512_tile * H)

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

    down_scale_view = inps.down_proj_scale.reshape((scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3]))
    """
    Construct a vector DGE index to index into E*16
    if block_expert == 0, we want something like this (tranposed to the P dimension)
    [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]  

    if block_expert == 3, we want something like this
    [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]  
    i.e, basically the same as above, with offset 16*3 = 48
    """

    n_quadrants_needed, n_remaining_partition = divmod(dims.p_I, 32)
    n_remaining_partition = n_remaining_partition // _q_height
    """
    assume I = 384
    p_I should be 96
    p_I // q_height = 12
    n_quadrants_needed = 3

    assume I = 192
    p_I should be 48
    p_I // q_height = 6
    n_quandrants_needed = 2
    The second quadrant will only have 2 meaningful partition in it. 
    """

    for i_quad in nl.static_range(n_quadrants_needed):
        arange_4P = nl.ndarray((4, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(arange_4P, [[1, 1]], offset=i_quad * 4, channel_multiplier=1)
        block_expert_broadcast = nl.ndarray((4, 1), dtype=block_expert.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(src=block_expert, dst=block_expert_broadcast)
        nisa.activation(
            dst=inps.p_down_idx_vector[i_quad * 32 : i_quad * 32 + 4],
            data=block_expert_broadcast,
            op=nl.copy,
            scale=float(scale_shape[1]),
            bias=arange_4P,
        )

    # handle remaining partitions
    if n_remaining_partition != 0:
        i_quad = n_quadrants_needed
        arange_remainder = nl.ndarray((n_remaining_partition, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(arange_remainder, [[1, 1]], offset=i_quad * 4, channel_multiplier=1)
        block_expert_broadcast_rem = nl.ndarray((n_remaining_partition, 1), dtype=block_expert.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(src=block_expert, dst=block_expert_broadcast_rem)
        nisa.activation(
            dst=inps.p_down_idx_vector[i_quad * 32 : i_quad * 32 + n_remaining_partition],
            data=block_expert_broadcast_rem,
            op=nl.copy,
            scale=float(scale_shape[1]),
            bias=arange_remainder,
        )

    token_indices_on_p = nl.ndarray(inps.p_down_idx_vector.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(token_indices_on_p, inps.p_down_idx_vector, engine=nisa.scalar_engine)
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


def compute_hidden_index_vector(
    inps: InputTensors,
    buffers: SharedBuffers,
    block_idx,
    dims: BWMMMXFP4DimensionSizes,
    skip_dma: SkipMode,
    is_block_idx_dynamic: bool = False,
):
    """
    Compute token-to-hidden-state index mapping for indirect DGE loading.

    Transforms token indices into 4H-folded indices for efficient vector DGE
    loading of hidden states with H dimension folded 4 times onto partitions.

    Args:
        inps (InputTensors): Input tensors with token_position_to_id and hidden_states.
        buffers (SharedBuffers): Buffers including token_4_H_indices_on_p output buffer.
        block_idx (int): Current block index.
        dims (BWMMMXFP4DimensionSizes): Dimension configuration with B, H.
        skip_dma (SkipMode): DMA skip configuration.
        is_block_idx_dynamic (bool): Whether block index is from dynamic loop.

    Returns:
        None: Modifies buffers.token_4_H_indices_on_p in-place.

    Notes:
        - Multiplies token indices by 4 and adds [0,1,2,3] for H folding
        - Stores result in partition dimension for vector DGE
        - Handles both static and dynamic block indexing
        - Processes B/32 tiles sequentially

    Pseudocode:
        T = B
        T_div_32 = T // 32

        if is_block_idx_dynamic:
            total_size = product of token_position_to_id.shape
            reshaped = reshape token_position_to_id to [total_size//B, B]
            dma_copy reshaped[block_idx, :] to token_indices
        else:
            dma_copy token_position_to_id[block_idx*B:(block_idx+1)*B] to token_indices

        arange_4H = [0, 1, 2, 3]
        all_token_4_H_indices = token_indices * 4 + arange_4H
        all_token_4_H_indices = reshape to [1, T*4]

        for T_div_32_idx in range(T_div_32):
            token_4_H_indices = all_token_4_H_indices[:, T_div_32_idx*128:(T_div_32_idx+1)*128]
            token_4_H_indices_psum = nc_transpose(token_4_H_indices)
            buffers.token_4_H_indices_on_p[:, T_div_32_idx] = token_4_H_indices_psum
    """
    T = dims.B
    T_div_32 = T // 32

    token_position_to_id = inps.token_position_to_id
    # We will use a 128-partition indirect DGE to load 32 tokens at a time, with H dim folded 4 times onto 4 partitions.
    if is_block_idx_dynamic:
        total_size = reduce(op='mul', input=token_position_to_id.shape, initial_value=1)
        kernel_assert(total_size % dims.B == 0, "token_position_to_id shape must be divisible by B")
        token_indices = nl.ndarray((1, dims.B), buffer=nl.sbuf, dtype=nl.int32)
        reshaped = token_position_to_id.reshape((total_size // dims.B, dims.B))
        nisa.dma_copy(
            src=reshaped.ap(pattern=[[dims.B, 1], [1, dims.B]], offset=0, scalar_offset=block_idx, indirect_dim=0),
            dst=token_indices[:, : dims.B],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )

    else:
        token_indices = nl.ndarray((1, T), buffer=nl.sbuf, dtype=nl.int32)
        nisa.dma_copy(
            src=token_position_to_id.reshape((1, token_position_to_id.shape[0]))[
                :, block_idx * dims.B : dims.B * (block_idx + 1)
            ],
            dst=token_indices[:, :T],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
    kernel_assert(token_indices.shape == (1, T), f'token_indices.shape = {token_indices.shape}')

    # hidden_states has shape of [T, H]
    # We will view it as [T * 4, H // 4], and do a 128-partition vector DGE on the outermost dim of T * 4.
    # To do so, the loaded 32 token indices should be multiplied by 4, and each added with 0, 1, 2, 3.
    arange_4H = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(arange_4H, [[1, 4]], offset=0)

    all_token_4_H_indices = nl.ndarray((1, T, 4), dtype=nl.float32, buffer=nl.sbuf)

    # Using AP with step=0 for broadcast:
    # - token_indices (1, T) broadcast to (1, T, 4): step=0 on the 4 dim
    # - arange_4H (1, 4) broadcast to (1, T, 4): step=0 on the T dim
    # token_indices shape: (1, T), so partition step = T
    # arange_4H shape: (1, 4), so partition step = 4
    nisa.scalar_tensor_tensor(
        dst=all_token_4_H_indices,
        data=token_indices.ap(
            pattern=[[T, 1], [1, T], [0, 4]],  # step=0 broadcasts across the 4 dim
            offset=0,
        ),
        op0=nl.multiply,
        operand0=4.0,
        op1=nl.add,
        operand1=arange_4H.ap(
            pattern=[[4, 1], [0, T], [1, 4]],  # step=0 broadcasts across the T dim
            offset=0,
        ),
    )

    all_token_4_H_indices = all_token_4_H_indices.reshape((1, T * 4))

    for T_div_32_idx in range(T_div_32):
        token_4_H_indices = all_token_4_H_indices[:, T_div_32_idx * 128 : T_div_32_idx * 128 + 128]

        token_4_H_indices_psum = nl.ndarray(
            (token_4_H_indices.shape[1], token_4_H_indices.shape[0]), dtype=token_4_H_indices.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(dst=token_4_H_indices_psum, data=token_4_H_indices)

        nisa.tensor_copy(
            dst=buffers.token_4_H_indices_on_p[:, T_div_32_idx], src=token_4_H_indices_psum, engine=nisa.scalar_engine
        )


def load_hidden_states(
    inps: InputTensors,
    token_4_H_indices_on_p: nl.ndarray,
    block_hidden_states,
    dims: BWMMMXFP4DimensionSizes,
    skip_dma: SkipMode,
):
    """
    Load hidden states from HBM to SBUF using precomputed indices.

    Uses indirect DGE with token_4_H_indices_on_p to gather hidden states
    for tokens in current block.

    Args:
        inps (InputTensors): Input tensors with hidden_states [T, H].
        token_4_H_indices_on_p (nl.ndarray): Precomputed indices [128, B/32].
        block_hidden_states (nl.ndarray): Destination buffer [128, B/32, H/512, 128].
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        None: Modifies block_hidden_states in-place.

    Notes:
        - Reshapes hidden_states to [T*4, H/512, 128] for folded access
        - Processes B/32 tiles with H/512 subtiles
        - Skips invalid tokens when skip_dma.skip_token is True

    Pseudocode:
        B = dims.B
        H_div_512 = H // 512
        B_div_32 = B // 32
        hidden_states_view = reshape hidden_states to [T*4, H_div_512, 128]

        for B_div_32_idx in range(B_div_32):
            dma_copy hidden_states_view[token_4_H_indices_on_p[:, B_div_32_idx], :, :] to block_hidden_states[:, B_div_32_idx, :, :]
    """
    B = dims.B
    H_div_512 = dims.H // 512
    B_div_32 = B // 32

    hidden_states_view = inps.hidden_states.reshape((dims.T * _q_width, H_div_512, _pmax))

    if DBG_KERNEL:
        pass

    for B_div_32_idx in range(B_div_32):
        nisa.dma_copy(
            src=hidden_states_view.ap(
                pattern=[[H_div_512 * _pmax, 32 * 4], [1, 1], [_pmax, H_div_512], [1, 16 * 8]],
                offset=0,
                vector_offset=token_4_H_indices_on_p.ap(
                    [[B_div_32, _pmax], [1, 1]],
                    offset=B_div_32_idx,
                ),
                indirect_dim=0,
            ),
            dst=block_hidden_states.ap(
                pattern=[
                    [B_div_32 * H_div_512 * _pmax, 32 * 4],
                    [H_div_512 * _pmax, 1],
                    [_pmax, H_div_512],
                    [1, 16 * 8],
                ],
                offset=B_div_32_idx * H_div_512 * _pmax,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


def quantize_block_hidden_state_T(buffers: SharedBuffers, prj_cfg: ProjConfig, dims: BWMMMXFP4DimensionSizes):
    """
    Quantize transposed block hidden states to MXFP4 format.

    Performs online quantization of hidden states from BF16/FP32 to MXFP4
    with per-block scaling factors.

    Args:
        buffers (SharedBuffers): Buffers with block_hidden_states_T, hidden_qtz_sb, hidden_scale_sb.
        prj_cfg (ProjConfig): Projection configuration.
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.

    Returns:
        None: Modifies buffers.hidden_qtz_sb and buffers.hidden_scale_sb in-place.

    Notes:
        - Input shape: [128, n_H512_tile, B/32, 128]
        - Output quantized shape: [128, n_H512_tile, B/32, 32]
        - Output scale shape: [128, n_H512_tile, B/32, 32]
        - Uses quantize_mx for hardware-accelerated quantization

    Pseudocode:
        quantize_mx(
            src=block_hidden_states_T[:128, :n_H512_tile, :B//32, :128],
            dst=hidden_qtz_sb[:128, :n_H512_tile, :B//32, :32],
            dst_scale=hidden_scale_sb[:128, :n_H512_tile, :B//32, :32]
        )
    """
    nisa.quantize_mx(
        src=buffers.block_hidden_states_T[:_pmax, : prj_cfg.n_H512_tile, : (dims.B // 32), :128],
        dst=buffers.hidden_qtz_sb[:_pmax, : prj_cfg.n_H512_tile, : (dims.B // 32), :32],
        dst_scale=buffers.hidden_scale_sb[:_pmax, : prj_cfg.n_H512_tile, : (dims.B // 32), :32],
    )


def load_and_quantize_hidden_states(
    inps: InputTensors,
    block_idx,
    buffers: SharedBuffers,
    dims: BWMMMXFP4DimensionSizes,
    kernel_cfg: BWMMMXFP4Configs,
    prj_cfg: ProjConfig,
    is_block_idx_dynamic: bool = False,
):
    """
    Load, transpose, and quantize hidden states for current block.

    Orchestrates the complete pipeline: load from HBM, transpose layout,
    and quantize to MXFP4 format.

    Args:
        inps (InputTensors): Input tensors.
        block_idx (int): Current block index.
        buffers (SharedBuffers): Shared buffers for intermediate results.
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.
        kernel_cfg (BWMMMXFP4Configs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        is_block_idx_dynamic (bool): Whether using dynamic block indexing.

    Returns:
        None: Modifies buffers in-place with quantized hidden states.

    Notes:
        - Supports both DMA transpose and PE transpose methods
        - Reshapes buffers for quantization
        - Called once per block during processing

    Pseudocode:
        if USE_DMA_TRANSPOSE:
            load_hidden_states_with_dma_transpose(inps, block_idx, buffers.block_hidden_states_T, dims, skip_dma)
        else:
            compute_hidden_index_vector(inps, buffers, block_idx, dims, skip_dma, is_block_idx_dynamic)
            load_hidden_states(inps, buffers.token_4_H_indices_on_p, buffers.block_hidden_states, dims, skip_dma)
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        reshape buffers.hidden_qtz_sb to [128, n_H512_tile, B//32, 32]
        reshape buffers.hidden_scale_sb to [128, n_H512_tile, B//32, 32]
        quantize_block_hidden_state_T(buffers, prj_cfg, dims)
    """
    if USE_DMA_TRANSPOSE:
        load_hidden_states_with_dma_transpose(inps, block_idx, buffers.block_hidden_states_T, dims, kernel_cfg.skip_dma)
    else:
        compute_hidden_index_vector(inps, buffers, block_idx, dims, kernel_cfg.skip_dma, is_block_idx_dynamic)
        load_hidden_states(inps, buffers.token_4_H_indices_on_p, buffers.block_hidden_states, dims, kernel_cfg.skip_dma)
        sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

    buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
    buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # quantize. Note that online quantize can only quantize to fp8
    quantize_block_hidden_state_T(buffers, prj_cfg, dims)


def compute_one_block(
    block_idx: int,
    next_block_idx: int,
    buffers: SharedBuffers,
    dims: BWMMMXFP4DimensionSizes,
    inps: InputTensors,
    outs: OutputTensors,
    dbg_tensors: DebugTensors,
    kernel_cfg: BWMMMXFP4Configs,
    prj_cfg: ProjConfig,
    shard_id: Any,
    is_dummy: bool = False,
    is_dynamic: bool = False,
):
    """
    Process one block through complete MoE MLP pipeline.

    Executes gate projection, up projection, activation, and down projection
    for a single block with MXFP4 quantization and expert routing.

    Args:
        block_idx (int): Current block index.
        next_block_idx (int): Next block index for prefetching (None if last).
        buffers (SharedBuffers): Shared computation buffers.
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        dbg_tensors (DebugTensors): Debug tensors (if enabled).
        kernel_cfg (BWMMMXFP4Configs): Kernel configuration.
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
        block_old = load_old_block_bwmm_mxfp4(output, token_indices_2D, block_old, B//128, compute_dtype, shard_id, skip_dma)

        gate_proj_out = gate_up_projection_mx_tp_shard_H(hidden_qtz_sb, hidden_scale_sb, gate_weights, gate_scales, gate_bias, cfg)
        gate_proj_out = clamp(gate_proj_out, gate_clamp_lower_limit, gate_clamp_upper_limit)

        up_proj_out = gate_up_projection_mx_tp_shard_H(hidden_qtz_sb, hidden_scale_sb, up_weights, up_scales, up_bias, cfg)
        up_proj_out = clamp(up_proj_out, up_clamp_lower_limit, up_clamp_upper_limit)

        if next_block_idx is not None:
            load_and_quantize_hidden_states(inps, next_block_idx, buffers, dims, kernel_cfg, prj_cfg, is_dynamic)

        if activation_function == SiLU:
            gate_proj_out = silu(gate_proj_out)
        elif activation_function == Swish:
            gate_proj_out = gelu_apprx_sigmoid(gate_proj_out)

        intermediate_state = gate_proj_out * up_proj_out
        block_new = down_projection_mx_shard_H(intermediate_state, down_weight, down_scale, down_bias, cfg)

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

    gate_and_up_weights, gate_and_up_scales, gup_bias = load_gup_weights_scales_mx4(
        inps, block_expert, dims, prj_cfg=prj_cfg, skip_dma=kernel_cfg.skip_dma
    )

    down_scale_sb, down_bias_sb = load_down_proj_weights_mx4(
        inps, block_expert, buffers.down_weight_qtz, dims, prj_cfg, kernel_cfg.skip_dma
    )
    down_bias_broadcasted = nl.ndarray((_pmax, dims.H), dtype=down_bias_sb.dtype, buffer=nl.sbuf)
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

    expert_affinity = calculate_expert_affinities(
        inps.expert_affinities_masked,
        token_indices_2D,
        block_expert,
        dims.E,
        dims.B // 128,
        kernel_cfg.compute_dtype,
        kernel_cfg.skip_dma,
    )

    block_old = load_old_block_bwmm_mxfp4(
        outs.output,
        token_indices_2D,
        buffers.block_old,
        dims.B // 128,
        kernel_cfg.compute_dtype,
        shard_id,
        kernel_cfg.skip_dma,
    )
    """
    GATE PROJECTION
    """
    gup_weights_reshaped = gate_and_up_weights.reshape((_pmax, 2 * prj_cfg.n_H512_tile_sharded, dims.I))
    gup_scales_reshaped = gate_and_up_scales.reshape((_pmax, 2 * prj_cfg.n_H512_tile_sharded, dims.I))
    gup_bias_reshaped = gup_bias.reshape((_pmax, 2 * prj_cfg.n_total_I512_tile, _q_width))

    gate_bias_sb = nl.ndarray((_pmax, prj_cfg.n_total_I512_tile, _q_width), dtype=gup_bias.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=gate_bias_sb, src=gup_bias_reshaped[:, 0 : prj_cfg.n_total_I512_tile, :], engine=nisa.vector_engine
    )

    gate_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=buffers.hidden_qtz_sb[:, :, :],
        hidden_scale_sb=buffers.hidden_scale_sb[:, :, :],
        weight_qtz=gup_weights_reshaped[:, 0 : prj_cfg.n_H512_tile_sharded, :],
        weight_scale=gup_scales_reshaped[:, 0 : prj_cfg.n_H512_tile_sharded, :],
        bias_sb=gate_bias_sb,
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
    # can't do an ap on an already sliced tensor
    up_bias_sb = nl.ndarray((_pmax, prj_cfg.n_total_I512_tile, _q_width), dtype=gup_bias.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=up_bias_sb,
        src=gup_bias_reshaped[:, prj_cfg.n_total_I512_tile : 2 * prj_cfg.n_total_I512_tile, :],
        engine=nisa.vector_engine,
    )

    up_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=buffers.hidden_qtz_sb,
        hidden_scale_sb=buffers.hidden_scale_sb,
        weight_qtz=gup_weights_reshaped[:, prj_cfg.n_H512_tile_sharded : 2 * prj_cfg.n_H512_tile_sharded, :],
        weight_scale=gup_scales_reshaped[:, prj_cfg.n_H512_tile_sharded : 2 * prj_cfg.n_H512_tile_sharded, :],
        bias_sb=up_bias_sb,
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
            load_hidden_states_with_dma_transpose(
                inps, next_block_idx, buffers.block_hidden_states_T, dims, kernel_cfg.skip_dma
            )
            stream_shuffle_broadcast(src=down_bias_sb, dst=down_bias_broadcasted)
        else:
            load_hidden_states(
                inps, buffers.token_4_H_indices_on_p, buffers.block_hidden_states, dims, kernel_cfg.skip_dma
            )
            stream_shuffle_broadcast(src=down_bias_sb, dst=down_bias_broadcasted)
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

        if is_dynamic:
            # quantize after fetching for dynamic blocks
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)

    else:
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
    intermediate_state_sb = intermediate_state_sb.reshape((_pmax, prj_cfg.n_total_I512_tile, dims.B, _q_width))

    """
    DOWN PROJECTION
    """

    block_new = down_projection_mx_shard_H(
        inter_sb=intermediate_state_sb,
        weight=buffers.down_weight_qtz,
        weight_scale=down_scale_sb,
        bias_sb=down_bias_broadcasted,
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
        nisa.tensor_tensor(
            dst=block_new[0:_pmax, n, 0 : dims.H],
            data1=block_new[0:_pmax, n, 0 : dims.H],
            op=nl.add,
            data2=block_old[0:_pmax, n, 0 : dims.H],
        )

        # output shape: (num_shards, T, H)
        T = outs.output.shape[-2]
        shard_offset = shard_id * T * dims.H

        # block_new shape: (_pmax, n_B128_tiles, H)
        # Scatter write: each partition writes H elements to output[shard_id, token_idx, :]
        # Use AP for dst with vector_offset for indirect scatter, direct slice for src

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


def load_hidden_states_with_dma_transpose(
    inps: InputTensors, block_idx, block_hidden_states_T, dims: BWMMMXFP4DimensionSizes, skip_dma: SkipMode
):
    """
    Load and transpose hidden states using DMA transpose operation.

    Alternative to PE-based transpose, uses hardware DMA transpose for
    potentially better performance.

    Args:
        inps (InputTensors): Input tensors with hidden_states and token_position_to_id.
        block_idx (int): Current block index.
        block_hidden_states_T (nl.ndarray): Destination buffer for transposed states.
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        None: Modifies block_hidden_states_T in-place.

    Notes:
        - Uses nisa.dma_transpose with axes=(2,1,0)
        - Processes B/32 tiles with multi-buffering
        - May have hardware limitations compared to PE transpose
        - Directly produces transposed layout

    Pseudocode:
        B = dims.B
        H_div_512 = H // 512
        B_div_32 = B // 32

        token_indices = load token_position_to_id[block_idx*B:(block_idx+1)*B]
        arange_4H = [0, 1, 2, 3]
        all_token_4_H_indices = token_indices * 4 + arange_4H
        all_token_4_H_indices = reshape to [1, B*4]

        for B_div_32_idx in range(B_div_32):
            token_4_H_indices = all_token_4_H_indices[:, B_div_32_idx*128:(B_div_32_idx+1)*128]
            token_4_H_indices_on_p = transpose token_4_H_indices to partition dimension
            dma_transpose hidden_states[token_4_H_indices_on_p, :, :] to block_hidden_states_T[:, :, B_div_32_idx, :] with axes=(2,1,0)
    """
    B = dims.B
    H_div_512 = dims.H // 512
    B_div_32 = B // 32

    hidden_states = inps.hidden_states
    token_position_to_id = inps.token_position_to_id
    # We will use a 128-partition indirect DGE to load 32 tokens at a time, with H dim folded 4 times onto 4 partitions.
    token_indices = nl.ndarray((1, dims.B), dtype=token_position_to_id.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=token_indices,
        src=token_position_to_id.reshape((1, token_position_to_id.shape[0]))[
            :, block_idx * dims.B : dims.B * (block_idx + 1)
        ],
        oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
    )
    kernel_assert(token_indices.shape == (1, dims.B), f'token_indices.shape = {token_indices.shape}')
    # hidden_states has shape of [T, H]
    # We will view it as [T * 4, H // 4], and do a 128-partition vector DGE on the outermost dim of T * 4.
    # To do so, the loaded 32 token indices should be multiplied by 4, and each added with 0, 1, 2, 3.
    arange_4H = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(arange_4H, [[1, 4]], offset=0)

    all_token_4_H_indices = nl.ndarray((1, dims.B, 4), dtype=nl.float32, buffer=nl.sbuf)

    # Using AP with step=0 for broadcast:
    # - token_indices (1, B) broadcast to (1, B, 4): step=0 on the 4 dim
    # - arange_4H (1, 4) broadcast to (1, B, 4): step=0 on the B dim
    nisa.scalar_tensor_tensor(
        dst=all_token_4_H_indices,
        data=token_indices.ap(pattern=[[dims.B, 1], [1, dims.B], [0, 4]], offset=0),
        op0=nl.multiply,
        operand0=4.0,
        op1=nl.add,
        operand1=arange_4H.ap(pattern=[[4, 1], [0, dims.B], [1, 4]], offset=0),
    )

    all_token_4_H_indices = all_token_4_H_indices.reshape((1, dims.B * 4))
    kernel_assert(all_token_4_H_indices.shape == (1, dims.B * 4), f'got {all_token_4_H_indices.shape}')
    for B_div_32_idx in range(B_div_32):  # directives=[ncc.multi_buffer(8)]
        token_4_H_indices = all_token_4_H_indices[:, B_div_32_idx * 128, B_div_32_idx * 128 + 128]
        token_4_H_indices_psum = nl.ndarray(
            (token_4_H_indices.shape[1], token_4_H_indices.shape[0]), dtype=token_4_H_indices.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(dst=token_4_H_indices_psum, data=token_4_H_indices)

        token_4_H_indices_on_p = nl.ndarray(token_4_H_indices_psum.shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=token_4_H_indices_on_p, src=token_4_H_indices_psum, engine=nisa.scalar_engine)

        kernel_assert(
            token_4_H_indices_on_p.shape == (128, 1), f'token_4_H_indices.shape = {token_4_H_indices_on_p.shape}'
        )

        nisa.dma_transpose(
            dst=block_hidden_states_T[: (32 * 4), :H_div_512, B_div_32_idx, : (16 * 8)],
            src=hidden_states.reshape((dims.T * 4, H_div_512, 128))[
                token_4_H_indices_on_p[:, 0], :H_div_512, : (16 * 8)
            ],
            axes=(2, 1, 0),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


def process_static_blocks(
    dims: BWMMMXFP4DimensionSizes,
    configs: BWMMMXFP4Configs,
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
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.
        configs (BWMMMXFP4Configs): Kernel configuration.
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
        load_hidden_states_with_dma_transpose(
            inps, first_block_idx, buffers.block_hidden_states_T, dims, configs.skip_dma
        )
    else:
        compute_hidden_index_vector(inps, buffers, first_block_idx, dims, configs.skip_dma, False)
        load_hidden_states(inps, buffers.token_4_H_indices_on_p, buffers.block_hidden_states, dims, configs.skip_dma)
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
    dims: BWMMMXFP4DimensionSizes,
    configs: BWMMMXFP4Configs,
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
        dims (BWMMMXFP4DimensionSizes): Dimension configuration.
        configs (BWMMMXFP4Configs): Kernel configuration.
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

    print(f"Start looping over dynamic blocks {num_static_blocks} to {dims.cond_vec_len} - 1")

    print("Prefetch first block for each core")
    first_block_idx = num_static_blocks + dims.shard_id  # ping-pong
    load_and_quantize_hidden_states(inps, first_block_idx, buffers, dims, configs, prj_cfg)

    # New register-based dynamic while loop
    reg = nisa.register_alloc()
    nisa.register_load(reg, buffers.cond)
    while reg:
        # we are iterating 2 blocks at a time
        # let's say the dynamic blocks start at block 15
        # tandem_block_idx: 15 17 19 21 ...
        # block_idx:
        # - on core 0: 15 17 19 ...
        # - on core 1: 16 18 20 ...
        tandem_block_idx = nl.ndarray(buffers.index.shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(tandem_block_idx, buffers.index, engine=nisa.vector_engine)
        # dyn_block_idx = tandem_block_idx + dims.shard_id
        dyn_block_idx = nl.ndarray(tandem_block_idx.shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dyn_block_idx, tandem_block_idx, op0=nl.add, operand0=dims.shard_id)

        # on each core, next block_idx = min(block_idx + 2, N-1)
        tmp_fp32_val = nl.ndarray(dyn_block_idx.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(tmp_fp32_val, dyn_block_idx, op0=nl.add, operand0=2.0, op1=nl.minimum, operand1=dims.N - 1.0)
        dyn_next_block_idx = nl.ndarray(tmp_fp32_val.shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dyn_next_block_idx, tmp_fp32_val, engine=nisa.vector_engine)
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
        )

        # tandem_next_block_idx = nl.add(tandem_block_idx, 2)
        tandem_next_block_idx = nl.ndarray(tandem_block_idx.shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(tandem_next_block_idx, tandem_block_idx, op0=nl.add, operand0=2)

        cond_next = nl.ndarray((1, 1), dtype=inps.conditions.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=cond_next,
            src=inps.conditions.ap(
                pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=tandem_next_block_idx, indirect_dim=0
            ),
        )

        # update tandem_block_index
        nisa.tensor_copy(
            dst=buffers.index.ap(pattern=[[1, 1], [1, 1]], offset=0),
            src=tandem_next_block_idx.ap(pattern=[[1, 1], [1, 1]], offset=0),
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(
            dst=buffers.cond.ap(pattern=[[1, 1], [1, 1]], offset=0),
            src=cond_next.ap(pattern=[[1, 1], [1, 1]], offset=0),
            engine=nisa.vector_engine,
        )

        # Reload register for next iteration
        nisa.register_load(reg, buffers.cond)
