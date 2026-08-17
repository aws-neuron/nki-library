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

"""This file implements blockwise matrix multiplication for MoE layers with block-level sharding strategies. The kernel processes tokens through expert-specific projections using static and dynamic loop structures for optimal performance."""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import core_barrier
from nki.isa.constants import oob_mode
from nki.language import NKIObject

from ....core.moe.moe_cte.moe_cte_utils import (
    DVE_CHANNELS_PER_BANK,
    N_PSUM_BANKS,
    PSUM_SIZE,
    TILE_SIZE,
    TOTAL_PSUM_SIZE,
    BlockShardStrategy,
    Configs,
    InputTensors,
    SkipMode,
    calculate_expert_affinities,
    compute_intermediate_states,
    div_ceil,
    load_block_expert,
    stream_shuffle_broadcast,
)
from ....core.utils import common_types
from ....core.utils.allocator import SbufManager, sizeinbytes
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import get_program_sharding_info
from ....core.utils.logging import get_logger

BLOCK_PARALLEL_FACTOR = 1
FUSE_AFFINITY_INTO_OUTPUT = (
    False  # When False, use calculate_expert_affinities instead of fused output[:, shard_id, H:H+E]
)
GUP_LOAD_COALESCE_FACTOR = 2
GUP_PROJ_DIM = 2


def _linear_to_global_block_idx(linear_idx, shard_id, shard_strat, n_blocks_per_shard):
    """Convert linear iteration index to global block index based on sharding strategy."""
    if shard_strat == BlockShardStrategy.PING_PONG:
        return 2 * linear_idx + shard_id
    elif shard_strat == BlockShardStrategy.HI_LO:
        return linear_idx + shard_id * n_blocks_per_shard


class DimensionSizes(NKIObject):
    B: int  # Block size (tokens per block)
    H: int  # Hidden dimension size
    T: int  # Total number of input tokens
    E: int  # Number of experts
    N: int  # Total number of blocks
    I_TP: int  # Intermediate size divided by tensor parallelism degree

    def __post_init__(self):
        self.h_tile_count = div_ceil(self.H, PSUM_SIZE)
        self.h_subtile_count = PSUM_SIZE // TILE_SIZE
        self.h_subtile_count_gup = PSUM_SIZE // TILE_SIZE // GUP_LOAD_COALESCE_FACTOR
        self.gup_tile_count = div_ceil(self.I_TP, TILE_SIZE)
        self.n_psum_tile_count = div_ceil(self.B, PSUM_SIZE)


def _clip_gate_up_projections(gate_and_up_proj_states_lst_sbuf, cfg, dims, cur_gup_tile_count, free_size):
    """Apply clipping to gate and up projection results.

    Args:
        gate_and_up_proj_states_lst_sbuf: [2][n_psum_tile][gup_tile] SBUF tensors
        cfg: Kernel configuration with clamp limits
        dims: DimensionSizes
        cur_gup_tile_count: Number of I_TP tiles in current chunk
        free_size: Free dimension size
    """
    if not (
        cfg.gate_clamp_upper_limit is not None
        or cfg.gate_clamp_lower_limit is not None
        or cfg.up_clamp_lower_limit is not None
        or cfg.up_clamp_upper_limit is not None
    ):
        return

    for token_tile_idx in range(dims.n_psum_tile_count):
        for i_tile_idx in range(cur_gup_tile_count):
            if cfg.gate_clamp_lower_limit is not None and cfg.gate_clamp_upper_limit is not None:
                nisa.tensor_scalar(
                    dst=gate_and_up_proj_states_lst_sbuf[0][token_tile_idx][i_tile_idx][
                        0:TILE_SIZE, nl.ds(0, free_size)
                    ],
                    data=gate_and_up_proj_states_lst_sbuf[0][token_tile_idx][i_tile_idx][
                        0:TILE_SIZE, nl.ds(0, free_size)
                    ],
                    op0=nl.minimum,
                    operand0=cfg.gate_clamp_upper_limit,
                    op1=nl.maximum,
                    operand1=cfg.gate_clamp_lower_limit,
                )
            else:
                if cfg.gate_clamp_upper_limit is not None:
                    nisa.tensor_scalar(
                        dst=gate_and_up_proj_states_lst_sbuf[0][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        data=gate_and_up_proj_states_lst_sbuf[0][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        op0=nl.minimum,
                        operand0=cfg.gate_clamp_upper_limit,
                    )
                if cfg.gate_clamp_lower_limit is not None:
                    nisa.tensor_scalar(
                        dst=gate_and_up_proj_states_lst_sbuf[0][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        data=gate_and_up_proj_states_lst_sbuf[0][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        op0=nl.maximum,
                        operand0=cfg.gate_clamp_lower_limit,
                    )
            if cfg.up_clamp_upper_limit is not None and cfg.up_clamp_lower_limit is not None:
                nisa.tensor_scalar(
                    dst=gate_and_up_proj_states_lst_sbuf[1][token_tile_idx][i_tile_idx][
                        0:TILE_SIZE, nl.ds(0, free_size)
                    ],
                    data=gate_and_up_proj_states_lst_sbuf[1][token_tile_idx][i_tile_idx][
                        0:TILE_SIZE, nl.ds(0, free_size)
                    ],
                    op0=nl.minimum,
                    operand0=cfg.up_clamp_upper_limit,
                    op1=nl.maximum,
                    operand1=cfg.up_clamp_lower_limit,
                )
            else:
                if cfg.up_clamp_upper_limit is not None:
                    nisa.tensor_scalar(
                        dst=gate_and_up_proj_states_lst_sbuf[1][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        data=gate_and_up_proj_states_lst_sbuf[1][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        op0=nl.minimum,
                        operand0=cfg.up_clamp_upper_limit,
                    )
                if cfg.up_clamp_lower_limit is not None:
                    nisa.tensor_scalar(
                        dst=gate_and_up_proj_states_lst_sbuf[1][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        data=gate_and_up_proj_states_lst_sbuf[1][token_tile_idx][i_tile_idx][
                            0:TILE_SIZE, nl.ds(0, free_size)
                        ],
                        op0=nl.maximum,
                        operand0=cfg.up_clamp_lower_limit,
                    )


def _store_block_output(
    output, block_new_lst, token_indices, H, NUM_TILES, shard_id, is_tensor_update_accumulating, skip_dma, sbm=None
):
    """Store computed block output to HBM.

    Args:
        output: Output tensor in HBM
        block_new_lst: Computed output [NUM_TILES] each (TILE_SIZE, H)
        token_indices: Token index mapping (TILE_SIZE, NUM_TILES)
        H: Hidden dimension
        NUM_TILES: Number of token tiles
        shard_id: LNC2 shard ID
        is_tensor_update_accumulating: Whether to use accumulating store pattern
        skip_dma: DMA skip configuration
        sbm: Optional SbufManager
    """
    for token_tile_idx in range(NUM_TILES):
        if sbm is not None:
            token_idx = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32, name=f"store_tok_idx_t{token_tile_idx}")
        else:
            token_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)

        if is_tensor_update_accumulating:
            nisa.tensor_copy(
                dst=token_idx,
                src=token_indices[0:TILE_SIZE, token_tile_idx : token_tile_idx + 1],
                engine=nisa.scalar_engine,
            )
            _out_tv = output.slice(1, shard_id, shard_id + 1).squeeze_dim(1).slice(1, 0, H).vector_select(0, token_idx)
            nisa.dma_copy(
                dst=_out_tv,
                src=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
        else:
            nisa.tensor_copy(
                dst=token_idx,
                src=token_indices[0:TILE_SIZE, token_tile_idx : token_tile_idx + 1],
            )
            _out_tv = output.vector_select(0, token_idx)
            nisa.dma_copy(
                dst=_out_tv,
                src=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )


@nki.jit
def bwmm_shard_on_block(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    block_size: int,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    down_activations: Optional[nl.NkiTensor] = None,
    activation_function: common_types.ActFnType = common_types.ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: common_types.ExpertAffinityScaleMode = common_types.ExpertAffinityScaleMode.POST_SCALE,
    n_block_per_iter: int = 1,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    sbm: Optional[SbufManager] = None,
    num_static_block: Optional[int] = None,
    total_n_blocks: Optional[int] = None,
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    non_overlapping_shards: bool = False,
):
    """
    Blockwise matrix multiplication kernel for context-encoding MoE layers.

    This kernel implements blockwise matrix multiplication for mixture-of-experts (MoE) layers, processing tokens
    through expert-specific gate, up, and down projections. The computation combines static optimization benefits
    with dynamic early-exit capabilities by using a hybrid loop structure. Optimized for block-level sharding
    with PING_PONG strategy and supports FP8 quantization, multiple expert affinity scaling modes, and TopK > 1
    accumulation patterns. Optimized for block sizes 128-512 tokens, 8-64 experts, and sequence lengths up to 32K
    tokens. Best performance when I_TP >= 512 and batch size * sequence length <= 4096.

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size
        B: Block size (tokens per block)
        E: Number of experts
        N: Total number of blocks (T / B)
        I_TP: Intermediate size divided by tensor parallelism degree

    Args:
        hidden_states (nl.NkiTensor): [T, H], Input token embeddings in HBM
        expert_affinities_masked (nl.NkiTensor): [(T+1)*E, 1], Expert routing weights for token assignments in HBM
        gate_up_proj_weight (nl.NkiTensor): [E, H, 2, I_TP], Combined gate and up projection weights in HBM
        down_proj_weight (nl.NkiTensor): [E, I_TP, H], Down projection weights in HBM
        block_size (int): Number of tokens processed per block
        token_position_to_id (nl.NkiTensor): [N*B], Mapping from block positions to token IDs in HBM
        block_to_expert (nl.NkiTensor): [N, 1], Expert assignment for each block in HBM
        gate_and_up_proj_bias (nl.NkiTensor, optional): [E, 2, I_TP], Bias terms for gate/up projections in HBM
        down_proj_bias (nl.NkiTensor, optional): [E, 1, H], Bias terms for down projection in HBM
        gate_up_proj_scale (nl.NkiTensor, optional): [E, 1, 2*I_TP], Dequantization scales for gate/up weights in HBM
        down_proj_scale (nl.NkiTensor, optional): [E, 1, H], Dequantization scales for down weights in HBM
        down_activations (nl.NkiTensor, optional): [N, B, H], Storage for intermediate activations in HBM
        activation_function (ActFnType): Activation function type (SiLU, GELU, etc.)
        skip_dma (SkipMode): DMA skip configuration for memory optimization
        compute_dtype (nki.dtype): Data type for internal computations (default: bfloat16)
        is_tensor_update_accumulating (bool): Enable accumulation for TopK > 1 scenarios
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Expert affinity application mode
        n_block_per_iter (int): Number of blocks processed per iteration
        gate_clamp_upper_limit (float, optional): Upper clamp limit for gate projections
        gate_clamp_lower_limit (float, optional): Lower clamp limit for gate projections
        up_clamp_upper_limit (float, optional): Upper clamp limit for up projections
        up_clamp_lower_limit (float, optional): Lower clamp limit for up projections
        block_sharding_strategy (BlockShardStrategy): Block distribution strategy across cores
        sbm (SbufManager, optional): SBUF memory manager. If None, one is created internally.
        num_static_block (int, optional): Number of blocks for static loop. Defaults to N.
        total_n_blocks (int, optional): Total block count for shard partitioning. Defaults to num_static_block.
        down_bias_tp_degree (int, optional): TP degree for down projection bias sharding.
        down_bias_tp_rank (int, optional): TP rank for down projection bias sharding.
        non_overlapping_shards (bool): When True, shards write to the same output slot (slot 0)
            and skip zero-init and cross-shard reduce. Requires non-overlapping token routing
            across shards (e.g., HI_LO strategy with sequence-level sharding). Default: False.

    Returns:
        output (nl.NkiTensor): Expert-processed token representations in HBM. Shape depends on accumulation mode:
            - Single expert (is_tensor_update_accumulating=False): [T, H]
            - Multiple experts (is_tensor_update_accumulating=True): [T, 2, H] for cross-core accumulation

    Notes:
        - Currently only supports PING_PONG block sharding strategy
        - Static loop processes N-E blocks with compile-time optimizations
        - Dynamic loop handles remaining blocks with early-exit capability
        - Supports FP8 quantization with dequantization scales
        - Expert affinity scaling modes: PRE_SCALE, POST_SCALE, PRE_SCALE_DELAYED
        - Multi-shard execution requires num_shards == 2 for accumulation

    Pseudocode:
        # Initialize output tensor
        output = zeros(T, H)

        # Process blocks in parallel across shards
        for block_idx in shard_blocks:
            # Load expert weights for current block
            expert_id = block_to_expert[block_idx]
            gup_weights = load_weights(gate_up_proj_weight[expert_id])
            down_weights = load_weights(down_proj_weight[expert_id])

            # Load block tokens
            token_ids = token_position_to_id[block_idx * B : (block_idx + 1) * B]
            hidden = hidden_states[token_ids]  # [B, H]

            # Gate and Up projections
            gate_proj = hidden @ gup_weights[:, 0, :]  # [B, I_TP]
            up_proj = hidden @ gup_weights[:, 1, :]    # [B, I_TP]

            # Apply activation and element-wise multiply
            intermediate = activation_fn(gate_proj) * up_proj  # [B, I_TP]

            # Down projection
            block_output = intermediate @ down_weights  # [B, H]

            # Scale by expert affinity and accumulate
            affinities = expert_affinities_masked[token_ids, expert_id]
            output[token_ids] += block_output * affinities

        return output
    """
    kernel_assert(
        block_sharding_strategy == BlockShardStrategy.PING_PONG or block_sharding_strategy == BlockShardStrategy.HI_LO,
        "Only support PING_PONG and HI_LO sharding strategies",
    )

    if sbm is None:
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            use_auto_alloc=False,
            logger=get_logger("bwmm_shard_on_block"),
        )
    sbm.open_scope(name="bwmm_shard_on_block")

    # Infer configurations from the input shapes
    T, H = hidden_states.shape
    B = block_size
    E, I_TP, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B
    NUM_TILES = B // TILE_SIZE
    shard_strat = block_sharding_strategy

    weights_dtype = compute_dtype
    _, num_shards, shard_id = get_program_sharding_info()
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=I_TP)
    dims.__post_init__()

    cfg = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        is_quant=gate_up_proj_scale is not None and down_proj_scale is not None,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
    )

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

    NUM_STATIC_BLOCKS = num_static_block if num_static_block is not None else N
    _output_shard_id = 0 if non_overlapping_shards else shard_id
    if is_tensor_update_accumulating:
        output = nl.ndarray((dims.T, 2, dims.H + E), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
        # Initialize output to zero — needed for block_old accumulation
        bwmm_output_initialization(
            output,
            shard_id=_output_shard_id,
            sbm=sbm,
            expert_affinities_masked=expert_affinities_masked
            if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE
            else None,
            E=E,
            H=dims.H,
            skip_zero_init=non_overlapping_shards,
        )
    else:
        output = nl.ndarray((dims.T, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Placeholder for FP8
    gup_scale = None
    down_scale = None

    # Down bias TP sharding
    _down_bias_h_size = H // down_bias_tp_degree if down_bias_tp_degree is not None else H
    _down_bias_h_offset = down_bias_tp_rank * _down_bias_h_size if down_bias_tp_rank is not None else 0

    _partition_n = total_n_blocks if total_n_blocks is not None else NUM_STATIC_BLOCKS
    n_blocks_per_shard = div_ceil(_partition_n, num_shards)
    # Actual number of valid blocks for this shard under interleaved distribution
    n_blocks_this_shard = div_ceil(max(NUM_STATIC_BLOCKS - shard_id, 0), num_shards)
    n_shard_tile_count = div_ceil(n_blocks_per_shard, BLOCK_PARALLEL_FACTOR)
    _padded_shard_size = n_blocks_per_shard + BLOCK_PARALLEL_FACTOR
    all_block_expert_broadcasted_per_shard = sbm.alloc_stack(
        (1, _padded_shard_size), dtype=nl.int32, name="all_blk_expert_bc"
    )
    nisa.memset(dst=all_block_expert_broadcasted_per_shard, value=E)
    if shard_strat == BlockShardStrategy.PING_PONG:
        _expert_load_pattern = [[1, 1], [2, n_blocks_this_shard]]
        _expert_load_offset = shard_id
    else:  # HI_LO
        _expert_load_pattern = [[1, 1], [1, n_blocks_this_shard]]
        _expert_load_offset = shard_id * n_blocks_per_shard
    nisa.dma_copy(
        dst=all_block_expert_broadcasted_per_shard[0:1, 0:n_blocks_this_shard],
        src=block_to_expert.reshape((block_to_expert.shape[0], 1)).ap(
            pattern=_expert_load_pattern, offset=_expert_load_offset
        ),
    )
    all_block_expert_real = sbm.alloc_stack((1, _padded_shard_size), dtype=nl.int32, name="all_blk_expert_real")
    nisa.tensor_copy(dst=all_block_expert_real, src=all_block_expert_broadcasted_per_shard)

    if skip_dma.skip_weight:
        # Convert multi-dimensional ndarray to list of ndarrays
        gup_weights_load_dst_lst = []
        for h_tile_idx in range(dims.h_tile_count):
            inner_lst = []
            for h_subtile_idx in range(dims.h_subtile_count_gup):
                tmp = sbm.alloc_stack(
                    (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP),
                    dtype=weights_dtype,
                    name=f"gup_w_cache_h{h_tile_idx}_s{h_subtile_idx}",
                )
                nisa.memset(dst=tmp, value=0)
                inner_lst.append(tmp)
            gup_weights_load_dst_lst.append(inner_lst)

        down_weights_load_dst_lst = []
        for n_i in range(dims.gup_tile_count):
            down_weights_load_dst_lst.append(
                sbm.alloc_stack((TILE_SIZE, H), dtype=weights_dtype, name=f"down_w_cache_i{n_i}")
            )

        is_weight_same_as_prev_hbm = compute_same_weights_block_parallel_hbm(
            N,
            block_to_expert=block_to_expert,
            num_shards=num_shards,
            shard_id=shard_id,
            shard_strat=shard_strat,
            sbm=sbm,
        )

        sbm.open_scope(name="skip_weight_mask")
        on_false = sbm.alloc_stack((1, _padded_shard_size), dtype=nl.int32, name="skip_on_false")
        nisa.memset(dst=on_false, value=E)
        need_skip = sbm.alloc_stack((1, _padded_shard_size), dtype=nl.uint8, name="skip_need_mask")
        nisa.memset(dst=need_skip, value=0)
        nisa.dma_copy(
            dst=need_skip[0:1, 0:n_blocks_per_shard],
            src=is_weight_same_as_prev_hbm.reshape((1, n_blocks_per_shard)).ap(
                pattern=[[1, 1], [1, n_blocks_per_shard]]
            ),
        )
        nisa.tensor_copy_predicated(
            dst=all_block_expert_broadcasted_per_shard,
            src=on_false,
            predicate=need_skip,
        )
        sbm.close_scope()
    else:
        gup_weights_load_dst_lst = None
        down_weights_load_dst_lst = None

    block_hidden_states_lst = []
    for blk_par_idx in range(BLOCK_PARALLEL_FACTOR):
        inner_lst = []
        for tile_idx in range(NUM_TILES):
            tmp = sbm.alloc_stack((TILE_SIZE, H), dtype=compute_dtype, name=f"blk_hidden_b{blk_par_idx}_t{tile_idx}")
            inner_lst.append(tmp)
        block_hidden_states_lst.append(inner_lst)

    token_indices_lst = []
    for blk_par_idx in range(BLOCK_PARALLEL_FACTOR):
        token_indices_lst.append(
            sbm.alloc_stack((TILE_SIZE, NUM_TILES), dtype=nl.int32, name=f"tok_idx_b{blk_par_idx}")
        )

    # STATIC LOOP
    for outer_block_iter in range(n_shard_tile_count):
        sbm.open_scope(interleave_degree=1, name="outer_block")
        block_psum_tiles = div_ceil(B, PSUM_SIZE)
        free_size = min(PSUM_SIZE, B)
        _H_DIV_128 = H // TILE_SIZE
        block_hidden_states_T_flat = []
        for k_ in range(BLOCK_PARALLEL_FACTOR):
            block_hidden_states_T_flat.append(
                sbm.alloc_stack(
                    (TILE_SIZE, 1, _H_DIV_128, NUM_TILES * TILE_SIZE),
                    dtype=nl.bfloat16,
                    name=f"hidden_T_flat_b{k_}",
                    align=32,
                )
            )
        # parallel load and transpose input
        for inner_block_iter in range(BLOCK_PARALLEL_FACTOR):
            linear_idx = outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
            block_idx = _linear_to_global_block_idx(linear_idx, shard_id, shard_strat, n_blocks_per_shard)
            sbm.set_name_prefix(f"o{outer_block_iter}_load{inner_block_iter}_")

            if block_idx < N and linear_idx < n_blocks_this_shard:
                shared_block_idx = shard_strat2blk_idx(shard_strat, outer_block_iter, inner_block_iter)
                local_block_idx = shared_block_idx + shard_strat2new_blk_idx_offset(
                    shard_id, shard_strat, n_blocks_per_shard
                )

                offset = local_block_idx * B
                nisa.dma_copy(
                    dst=token_indices_lst[inner_block_iter].ap(pattern=[[NUM_TILES, TILE_SIZE], [1, NUM_TILES]]),
                    src=token_position_to_id.reshape((token_position_to_id.shape[0], 1)).ap(
                        pattern=[[1, TILE_SIZE], [TILE_SIZE, NUM_TILES]], offset=offset
                    ),
                )

                if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE:
                    v_expert = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32, name="prescale_v_expert")
                    block_expert = load_block_expert(block_to_expert, local_block_idx, sbm=sbm)
                    shuffle_mask = [0] * DVE_CHANNELS_PER_BANK
                    for channel_idx in range(4):
                        nisa.nc_stream_shuffle(
                            dst=v_expert[
                                DVE_CHANNELS_PER_BANK * channel_idx : DVE_CHANNELS_PER_BANK * (channel_idx + 1), 0:1
                            ],
                            src=block_expert[0:1, 0:1],
                            shuffle_mask=shuffle_mask,
                        )

                    expert_affinity_f32_lst = []
                    for tile_idx in range(NUM_TILES):
                        expert_affinity_f32_lst.append(
                            sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.float32, name=f"prescale_aff_f32_t{tile_idx}")
                        )

                    for token_tile_idx in range(NUM_TILES):
                        addr = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32, name=f"prescale_addr_t{token_tile_idx}")
                        nisa.tensor_scalar(
                            dst=addr,
                            data=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx],
                            op0=nl.multiply,
                            operand0=E,
                        )
                        addr_fin = sbm.alloc_stack(
                            (TILE_SIZE, 1), dtype=nl.int32, name=f"prescale_addr_fin_t{token_tile_idx}"
                        )
                        nisa.tensor_tensor(dst=addr_fin, data1=addr, data2=v_expert, op=nl.add)

                        if skip_dma.skip_token:
                            nisa.tensor_scalar(dst=addr_fin, data=addr_fin, op0=nl.minimum, operand0=-1)

                        expert_affinity_dtype = sbm.alloc_stack(
                            (TILE_SIZE, 1), dtype=compute_dtype, name=f"prescale_aff_dtype_t{token_tile_idx}"
                        )
                        if skip_dma.skip_token:
                            nisa.memset(value=0.0, dst=expert_affinity_dtype)

                        # nl.load with indirect indexing -> nisa.dma_copy with .ap()
                        num_cols = expert_affinities_masked.shape[1]
                        addr_fin_reshaped = sbm.alloc_stack(
                            (TILE_SIZE, 1), dtype=nl.int32, name=f"prescale_addr_reshape_t{token_tile_idx}"
                        )
                        nisa.tensor_copy(dst=addr_fin_reshaped, src=addr_fin[0:TILE_SIZE, 0:1])

                        nisa.dma_copy(
                            dst=expert_affinity_dtype[0:TILE_SIZE, 0:1],
                            src=expert_affinities_masked.vector_select(0, addr_fin_reshaped),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )

                        # Cast to float32
                        nisa.tensor_copy(
                            dst=expert_affinity_f32_lst[token_tile_idx][0:TILE_SIZE, 0:1],
                            src=expert_affinity_dtype[0:TILE_SIZE, 0:1],
                        )

                if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE:
                    for token_tile_idx in range(NUM_TILES):
                        block_token_mapping = (token_indices_lst[inner_block_iter]).slice(
                            1, token_tile_idx, token_tile_idx + 1
                        )
                        _hs_tv = hidden_states.vector_select(0, block_token_mapping)
                        nisa.dma_copy(
                            dst=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                            src=_hs_tv,
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )
                        nisa.tensor_scalar(
                            dst=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                            data=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                            op0=nl.multiply,
                            operand0=expert_affinity_f32_lst[token_tile_idx][0:TILE_SIZE, 0],
                            engine=nisa.vector_engine,
                        )

                block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)

                # DMA transpose: HBM → SBUF transposed (frees PE for matmul)
                _H_DIV_128 = H // TILE_SIZE
                for b_tile_idx in range(NUM_TILES):
                    _tok_on_p = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.uint32, name=f"tok_p_{b_tile_idx}")
                    # Can be removed
                    nisa.tensor_copy(
                        dst=_tok_on_p,
                        src=token_indices_lst[inner_block_iter][0:TILE_SIZE, b_tile_idx : b_tile_idx + 1],
                        engine=nisa.scalar_engine,
                    )
                    nisa.dma_transpose(
                        dst=block_hidden_states_T_flat[inner_block_iter][
                            :, :, :, 128 * b_tile_idx : 128 * b_tile_idx + 128
                        ],
                        src=hidden_states.ap(
                            # TODO: fill src AP pattern
                            # src shape: (T, H) = (T, 3072) in bf16
                            # Need to read 128 tokens (via vector_offset) × 24 h_subtiles × 128 elements
                            pattern=[[H, 128], [1, 1], [128, H // 128], [1, 128]],
                            offset=0,
                            vector_offset=_tok_on_p,
                            indirect_dim=0,
                        ),
                        axes=(3, 1, 2, 0),
                        dge_mode=nisa.dge_mode.swdge,
                        oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                    )

        # sequential load weights and compute
        for inner_block_iter in range(BLOCK_PARALLEL_FACTOR):
            linear_idx = outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
            block_idx = _linear_to_global_block_idx(linear_idx, shard_id, shard_strat, n_blocks_per_shard)
            sbm.set_name_prefix(f"o{outer_block_iter}_comp{inner_block_iter}_")
            if block_idx < N and linear_idx < n_blocks_this_shard:
                sbm.open_scope(name="inner_block")
                shared_block_idx = shard_strat2blk_idx(shard_strat, outer_block_iter, inner_block_iter)
                local_block_idx = shared_block_idx + shard_strat2new_blk_idx_offset(
                    shard_id, shard_strat, n_blocks_per_shard
                )
                block_expert = sbm.alloc_stack((1, 1), dtype=nl.int32, name="inner_block_expert")
                nisa.tensor_copy(
                    dst=block_expert,
                    src=all_block_expert_broadcasted_per_shard[0:1, linear_idx : linear_idx + 1],
                    engine=nisa.scalar_engine,
                )
                real_expert = all_block_expert_real.slice(1, linear_idx, linear_idx + 1)
                nisa.tensor_copy(
                    dst=real_expert,
                    src=all_block_expert_real[0:1, linear_idx : linear_idx + 1],
                    engine=nisa.scalar_engine,
                )

                # Determine if I_TP tiling is needed: check if full I_TP weights fit
                _gup_cost = (
                    dims.h_tile_count
                    * dims.h_subtile_count_gup
                    * GUP_LOAD_COALESCE_FACTOR
                    * GUP_PROJ_DIM
                    * I_TP
                    * sizeinbytes(gate_up_proj_weight.dtype)
                )
                _dp_cost = dims.gup_tile_count * H * sizeinbytes(down_proj_weight.dtype)
                _free = sbm.get_free_space()
                needs_itp_tiling = gup_weights_load_dst_lst is None and _gup_cost + _dp_cost > _free // 2
                _logger = get_logger("bwmm_shard_on_block")
                _logger.debug(
                    f"needs_itp_tiling={needs_itp_tiling} gup_cost={_gup_cost} dp_cost={_dp_cost} total={_gup_cost + _dp_cost} free={_free} half_free={_free // 2} gup_load_dst_is_none={gup_weights_load_dst_lst is None}"
                )

                gup_weights = (
                    load_gate_up_proj_weights(
                        gate_up_proj_weight,
                        block_expert,
                        weights_dtype,
                        skip_dma,
                        load_dst=gup_weights_load_dst_lst,
                        sbm=sbm,
                    )
                    if not needs_itp_tiling
                    else None
                )

                dp_weights = (
                    load_down_proj_weight(
                        down_proj_weight,
                        block_expert,
                        weights_dtype,
                        skip_dma,
                        load_dst=down_weights_load_dst_lst,
                        sbm=sbm,
                    )
                    if not needs_itp_tiling
                    else None
                )

                # load bias
                if cfg.linear_bias:
                    gate_up_bias_T = load_and_transpose_gup_bias(inps, dims, cfg, real_expert, skip_dma, sbm=sbm)

                if is_tensor_update_accumulating:
                    block_old = bwmm_load_old_block(
                        output,
                        token_indices_lst[inner_block_iter],
                        NUM_TILES,
                        compute_dtype,
                        skip_dma,
                        shard_id=_output_shard_id,
                        sbm=sbm,
                    )
                else:
                    block_old = None

                free_size = min(PSUM_SIZE, B)

                # Extract expert affinity
                if (
                    expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE
                    and is_tensor_update_accumulating
                    and FUSE_AFFINITY_INTO_OUTPUT
                ):
                    real_expert_u32 = sbm.alloc_stack((1, 1), dtype=nl.uint32, name="real_expert_u32")
                    nisa.tensor_scalar(dst=real_expert_u32, data=real_expert, op0=nl.add, operand0=H)
                    expert_affinity_f32 = []
                    for _tile_idx in range(NUM_TILES):
                        _aff_tile = sbm.alloc_stack(
                            (TILE_SIZE, 1), dtype=compute_dtype, name=f"aff_from_old_t{_tile_idx}", align=32
                        )
                        _aff_tv = (block_old[_tile_idx]).select(dim=1, index=real_expert_u32).expand_dim(1)
                        nisa.tensor_copy(dst=_aff_tile, src=_aff_tv, engine=nisa.scalar_engine)
                        expert_affinity_f32.append(_aff_tile)
                elif expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE:
                    expert_affinity_f32 = calculate_expert_affinities(
                        expert_affinities_masked,
                        token_indices_lst[inner_block_iter],
                        real_expert,
                        E,
                        NUM_TILES,
                        compute_dtype,
                        skip_dma,
                        sbm=sbm,
                        cast_to_f32=not is_tensor_update_accumulating,
                    )
                else:
                    expert_affinity_f32 = None
                down_activations = None

                down_bias_raw = None
                if cfg.linear_bias:
                    down_bias_broadcasted = None
                    down_bias_raw = load_down_bias_raw(
                        inps, dims, cfg, real_expert, skip_dma, sbm=sbm, bias_h_size=_down_bias_h_size
                    )
                else:
                    down_bias_broadcasted = None

                # PRE_SCALE_DELAYED affinity: compute once before chunk loop
                expert_affinity_T_broadcasted = None
                if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
                    expert_affinity_T_broadcasted = sbm.alloc_stack((TILE_SIZE, free_size), dtype=compute_dtype)
                    sbm.open_scope(name="delayed_affinity")
                    v_expert = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32)
                    delayed_block_expert = load_block_expert(block_to_expert, local_block_idx, sbm=sbm)
                    shuffle_mask = [0] * DVE_CHANNELS_PER_BANK
                    for channel_idx in range(4):
                        nisa.nc_stream_shuffle(
                            dst=v_expert[
                                DVE_CHANNELS_PER_BANK * channel_idx : DVE_CHANNELS_PER_BANK * (channel_idx + 1), 0:1
                            ],
                            src=delayed_block_expert[0:1, 0:1],
                            shuffle_mask=shuffle_mask,
                        )

                    expert_affinity_T = nl.ndarray((1, free_size), dtype=compute_dtype, buffer=nl.psum)

                    expert_affinity_lst = []
                    for tile_idx in range(NUM_TILES):
                        expert_affinity_lst.append(sbm.alloc_stack((TILE_SIZE, 1), dtype=compute_dtype))

                    for token_tile_idx in range(NUM_TILES):
                        addr = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32)
                        nisa.tensor_scalar(
                            dst=addr,
                            data=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx],
                            op0=nl.multiply,
                            operand0=E,
                        )

                        addr_fin = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32)
                        nisa.tensor_tensor(dst=addr_fin, data1=addr, data2=v_expert, op=nl.add)

                        if skip_dma.skip_token:
                            nisa.tensor_scalar(dst=addr_fin, data=addr_fin, op0=nl.maximum, operand0=-1)

                        if skip_dma.skip_token:
                            nisa.memset(value=0.0, dst=expert_affinity_lst[token_tile_idx])

                        num_cols = expert_affinities_masked.shape[1]
                        addr_fin_tv = addr_fin.slice(1, 0, 1)

                        nisa.dma_copy(
                            dst=expert_affinity_lst[token_tile_idx][0:TILE_SIZE, 0:1],
                            src=expert_affinities_masked.vector_select(0, addr_fin_tv),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )

                        num_f = min(TILE_SIZE, free_size - token_tile_idx * TILE_SIZE)
                        nisa.nc_transpose(
                            dst=expert_affinity_T[0:1, token_tile_idx * TILE_SIZE : token_tile_idx * TILE_SIZE + num_f],
                            data=expert_affinity_lst[token_tile_idx][0:num_f, 0:1],
                        )

                    shuffle_mask_broadcast = [0] * DVE_CHANNELS_PER_BANK
                    for channel_idx in range(4):
                        nisa.nc_stream_shuffle(
                            dst=expert_affinity_T_broadcasted[
                                channel_idx * DVE_CHANNELS_PER_BANK : (channel_idx + 1) * DVE_CHANNELS_PER_BANK,
                                0:free_size,
                            ],
                            src=expert_affinity_T,
                            shuffle_mask=shuffle_mask_broadcast,
                        )
                    sbm.close_scope()

                # Compute I_TILE now — all non-chunk buffers are allocated, so get_free_space is accurate
                I_TILE = I_TP
                if gup_weights is None and dp_weights is None:
                    # Reserve space for block_new_lst_pre which is allocated when I_CHUNK_COUNT > 1
                    block_new_reserve = NUM_TILES * H * sizeinbytes(output.dtype)
                    for candidate in range(I_TP, TILE_SIZE - 1, -TILE_SIZE):
                        if candidate % TILE_SIZE != 0:
                            continue
                        needs_chunking = candidate < I_TP
                        gup_tc = div_ceil(candidate, TILE_SIZE)
                        ring_factor = 2 if needs_chunking else 1
                        chunk_cost = (
                            GUP_PROJ_DIM * dims.n_psum_tile_count * gup_tc * free_size * 4
                            + gup_tc * B * sizeinbytes(compute_dtype) * 2  # intermediate + tmp_lst
                            + ring_factor
                            * GUP_LOAD_COALESCE_FACTOR
                            * GUP_PROJ_DIM
                            * candidate
                            * sizeinbytes(gate_up_proj_weight.dtype)
                            + gup_tc
                            * H
                            * sizeinbytes(
                                down_proj_weight.dtype
                            )  # gup_n_tile down weight buffers (loaded once, reused)
                            + (block_new_reserve if needs_chunking else 0)
                            + 4352  # alignment and misc overhead
                        )
                        if chunk_cost <= sbm.get_free_space():
                            I_TILE = candidate
                            _logger.debug(
                                f"I_TILE={candidate} chunk_cost={chunk_cost} free={sbm.get_free_space()} needs_chunking={needs_chunking}"
                            )
                            break
                        else:
                            _logger.debug(
                                f"I_TILE candidate={candidate} rejected: chunk_cost={chunk_cost} > free={sbm.get_free_space()}"
                            )
                I_TILE = min(I_TILE, I_TP)  # Force chunking with I_TILE=512 for debugging
                # Cap I_TILE to fit within PSUM banks (one PSUM per i_tile per projection)
                I_TILE = min(I_TILE, N_PSUM_BANKS * TILE_SIZE)
                I_CHUNK_COUNT = div_ceil(I_TP, I_TILE)

                # block_new_lst: pre-allocate only when chunking (I_CHUNK_COUNT > 1)
                # to avoid re-allocation per chunk. When I_CHUNK_COUNT == 1, let
                # compute_block_output allocate internally (original behavior).
                block_new_lst_pre = None
                if I_CHUNK_COUNT > 1:
                    block_new_lst_pre = []
                    for tile_idx in range(NUM_TILES):
                        block_new_lst_pre.append(sbm.alloc_stack((TILE_SIZE, H), dtype=output.dtype))

                for i_chunk_idx in range(I_CHUNK_COUNT):
                    i_chunk_offset = i_chunk_idx * I_TILE
                    cur_i_tile = min(I_TILE, I_TP - i_chunk_offset)
                    cur_gup_tile_count = div_ceil(cur_i_tile, TILE_SIZE)

                    sbm.open_scope(name=f"i_chunk_{i_chunk_idx}")
                    sbm.set_name_prefix(f"o{outer_block_iter}_comp{inner_block_iter}_c{i_chunk_idx}_")

                    # Allocate intermediate_states for this chunk
                    intermediate_states = []
                    for gup_tile_idx in range(cur_gup_tile_count):
                        intermediate_states.append(sbm.alloc_stack((TILE_SIZE, B), dtype=compute_dtype))

                    # Sub-scope for gate-up projection
                    sbm.open_scope(name="gate_up_proj")

                    # Allocate SBUF for all projections upfront (needed after scope)
                    gate_and_up_proj_states_lst_sbuf = []
                    for proj_idx in range(GUP_PROJ_DIM):
                        n_sbuf_lst = []
                        for psum_tile_idx in range(dims.n_psum_tile_count):
                            gup_sbuf_lst = []
                            for gup_tile_idx in range(cur_gup_tile_count):
                                gup_sbuf_lst.append(sbm.alloc_stack((TILE_SIZE, free_size), dtype=nl.bfloat16))
                            n_sbuf_lst.append(gup_sbuf_lst)
                        gate_and_up_proj_states_lst_sbuf.append(n_sbuf_lst)

                    # Process gate and up projections separately to halve PSUM usage
                    for projection_idx in range(GUP_PROJ_DIM):
                        # Allocate PSUMs for this projection only
                        proj_psum_lst = []
                        for psum_tile_idx in range(dims.n_psum_tile_count):
                            gup_psum_lst = []
                            for gup_tile_idx in range(cur_gup_tile_count):
                                gup_psum_lst.append(
                                    nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum)
                                )
                            proj_psum_lst.append(gup_psum_lst)

                        # Gate-up weight loading with interleave_degree for DMA/compute overlap
                        if gup_weights is None:
                            sbm.open_scope(interleave_degree=1, name="gup_w_interleave")

                        for h_tile_idx in range(dims.h_tile_count):
                            for h_subtile_idx in range(dims.h_subtile_count_gup):
                                if gup_weights is not None:
                                    cur_gup_w = gup_weights[h_tile_idx][h_subtile_idx]
                                else:
                                    cur_gup_w = sbm.alloc_stack(
                                        (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, cur_i_tile),
                                        dtype=gate_up_proj_weight.dtype,
                                    )
                                    h_offset = (
                                        PSUM_SIZE * h_tile_idx + GUP_LOAD_COALESCE_FACTOR * TILE_SIZE * h_subtile_idx
                                    )
                                    h_remaining = H - h_offset
                                    num_h = min(TILE_SIZE, div_ceil(h_remaining, GUP_LOAD_COALESCE_FACTOR))
                                    if num_h < TILE_SIZE and not skip_dma.skip_weight:
                                        nisa.memset(dst=cur_gup_w, value=0)
                                    offset = h_offset * (GUP_PROJ_DIM * I_TP) + i_chunk_offset
                                    flat_idx = h_subtile_idx + h_tile_idx * dims.h_subtile_count_gup
                                    nisa.dma_copy(
                                        dst=cur_gup_w[
                                            0:num_h, 0:GUP_LOAD_COALESCE_FACTOR, 0:GUP_PROJ_DIM, 0:cur_i_tile
                                        ],
                                        src=gate_up_proj_weight.ap(
                                            pattern=[
                                                [GUP_PROJ_DIM * I_TP, num_h],
                                                [TILE_SIZE * GUP_PROJ_DIM * I_TP, GUP_LOAD_COALESCE_FACTOR],
                                                [I_TP, GUP_PROJ_DIM],
                                                [1, cur_i_tile],
                                            ],
                                            offset=offset,
                                            scalar_offset=block_expert,
                                        ),
                                        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                                        dge_mode=nisa.dge_mode.hwdge if flat_idx % 2 == 0 else nisa.dge_mode.swdge,
                                    )
                                    sbm.increment_section()

                                for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                                    for i_tile_idx in range(cur_gup_tile_count):
                                        global_i_pos = i_chunk_offset + TILE_SIZE * i_tile_idx
                                        num_valid_k = min(TILE_SIZE, I_TP - global_i_pos)
                                        local_i_pos = TILE_SIZE * i_tile_idx

                                        for batch_tile_idx in range(dims.n_psum_tile_count):
                                            nisa.nc_matmul(
                                                dst=proj_psum_lst[batch_tile_idx][i_tile_idx][
                                                    0:num_valid_k, nl.ds(0, free_size)
                                                ],
                                                stationary=cur_gup_w[
                                                    nl.ds(0, TILE_SIZE),
                                                    op_idx,
                                                    projection_idx,
                                                    nl.ds(local_i_pos, num_valid_k),
                                                ],
                                                moving=block_hidden_states_T_flat[inner_block_iter][
                                                    :,
                                                    0,
                                                    h_tile_idx * dims.h_subtile_count
                                                    + h_subtile_idx * GUP_LOAD_COALESCE_FACTOR
                                                    + op_idx,
                                                    nl.ds(0, free_size),
                                                ],
                                            )

                        if gup_weights is None:
                            sbm.close_scope()  # Close gup_w_interleave scope

                        # Copy this projection's PSUMs to SBUF
                        for i_tile_idx in range(cur_gup_tile_count):
                            global_i_pos = i_chunk_offset + TILE_SIZE * i_tile_idx
                            num_valid_k = min(TILE_SIZE, I_TP - global_i_pos)
                            for batch_tile_idx in range(dims.n_psum_tile_count):
                                if cfg.linear_bias:
                                    global_bias_idx = (i_chunk_offset // TILE_SIZE) + i_tile_idx
                                    nisa.tensor_scalar(
                                        dst=gate_and_up_proj_states_lst_sbuf[projection_idx][batch_tile_idx][
                                            i_tile_idx
                                        ][0:num_valid_k, nl.ds(0, free_size)],
                                        data=proj_psum_lst[batch_tile_idx][i_tile_idx][
                                            0:num_valid_k, nl.ds(0, free_size)
                                        ],
                                        operand0=gate_up_bias_T.ap(
                                            pattern=[
                                                [2 * dims.gup_tile_count, num_valid_k],
                                                [1, 1],
                                            ],
                                            offset=global_bias_idx * 2 + projection_idx,
                                        ),
                                        op0=nl.add,
                                    )
                                else:
                                    nisa.tensor_copy(
                                        dst=gate_and_up_proj_states_lst_sbuf[projection_idx][batch_tile_idx][
                                            i_tile_idx
                                        ][0:num_valid_k, nl.ds(0, free_size)],
                                        src=proj_psum_lst[batch_tile_idx][i_tile_idx][
                                            0:num_valid_k, nl.ds(0, free_size)
                                        ],
                                        engine=nisa.scalar_engine,
                                    )

                    _clip_gate_up_projections(
                        gate_and_up_proj_states_lst_sbuf,
                        cfg,
                        dims,
                        cur_gup_tile_count,
                        free_size,
                    )

                    intermediate_states = compute_intermediate_states(
                        gate_and_up_proj_states_lst_sbuf,
                        B,
                        cur_i_tile,
                        compute_dtype,
                        activation_function=activation_function,
                        expert_affinity_T_broadcasted=expert_affinity_T_broadcasted,
                        gup_scale=gup_scale,
                        sbm=sbm,
                        intermediate_states_lst=intermediate_states,
                    )

                    sbm.close_scope()  # Free gup_sb, gup_bias, tmp buffers for this chunk

                    # Down projection for this chunk
                    # All chunks accumulate raw matmul results. Affinity/bias applied after loop.
                    if I_CHUNK_COUNT > 1:
                        # Multi-chunk: accumulate raw results without affinity/bias/block_old
                        chunk_block_old = None if i_chunk_idx == 0 else block_new_lst_pre
                        chunk_accumulating = i_chunk_idx > 0
                    else:
                        # Single chunk: original behavior
                        chunk_block_old = block_old
                        chunk_accumulating = is_tensor_update_accumulating

                    block_new_lst = compute_block_output(
                        intermediate_states,
                        dp_weights,
                        expert_affinity_f32 if I_CHUNK_COUNT == 1 else None,
                        chunk_block_old,
                        down_activations,
                        local_block_idx,
                        H,
                        cur_i_tile,
                        NUM_TILES,
                        output_dtype=output.dtype if I_CHUNK_COUNT == 1 else compute_dtype,
                        down_bias_broadcasted=None,
                        down_bias_raw=down_bias_raw if (I_CHUNK_COUNT == 1 or i_chunk_idx == 0) else None,
                        is_tensor_update_accumulating=chunk_accumulating,
                        down_scale=down_scale,
                        sbm=sbm,
                        down_proj_weight_hbm=down_proj_weight if dp_weights is None else None,
                        block_expert=block_expert if dp_weights is None else None,
                        skip_dma=skip_dma,
                        block_new_lst_pre=block_new_lst_pre,
                        i_tp_offset=i_chunk_offset,
                        down_bias_h_offset=_down_bias_h_offset,
                        down_bias_h_size=_down_bias_h_size if down_bias_tp_degree is not None else None,
                    )

                    sbm.close_scope()  # Close i_chunk scope

                # After all chunks: apply expert affinity and bias if multi-chunk
                if I_CHUNK_COUNT > 1:
                    h_i_upper = div_ceil(H, TOTAL_PSUM_SIZE)
                    for token_tile_idx in range(NUM_TILES):
                        for h_tile_idx in range(h_i_upper):
                            for h_subtile_idx in range(N_PSUM_BANKS):
                                psum_start = TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * h_subtile_idx
                                actual_psum_size = min(PSUM_SIZE, H - psum_start)
                                if actual_psum_size <= 0:
                                    continue
                                sl = nl.ds(psum_start, actual_psum_size)
                                if expert_affinity_f32 is not None:
                                    if is_tensor_update_accumulating:
                                        nisa.scalar_tensor_tensor(
                                            dst=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                            data=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                            op0=nl.multiply,
                                            operand0=expert_affinity_f32[token_tile_idx][0:TILE_SIZE, 0],
                                            op1=nl.add,
                                            operand1=block_old[token_tile_idx][0:TILE_SIZE, sl],
                                        )
                                    else:
                                        nisa.tensor_scalar(
                                            dst=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                            data=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                            operand0=expert_affinity_f32[token_tile_idx][0:TILE_SIZE, 0],
                                            op0=nl.multiply,
                                        )
                                else:
                                    if is_tensor_update_accumulating:
                                        nisa.tensor_tensor(
                                            dst=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                            data1=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                            data2=block_old[token_tile_idx][0:TILE_SIZE, sl],
                                            op=nl.add,
                                        )

                _store_block_output(
                    output,
                    block_new_lst,
                    token_indices_lst[inner_block_iter],
                    H,
                    NUM_TILES,
                    _output_shard_id,
                    is_tensor_update_accumulating,
                    skip_dma,
                    sbm=sbm,
                )
                sbm.close_scope()
        sbm.increment_section()
        sbm.close_scope()
    # END OF STATIC LOOP
    sbm.set_name_prefix("")

    # Final accumulation across shards (only when not sequence-sharded)
    if not non_overlapping_shards and is_tensor_update_accumulating and num_shards > 1:
        kernel_assert(num_shards == 2, "only support reducing data from 2 shards")
        reduce_tile_size = 128
        if skip_dma.skip_token:
            reduce_tiles = div_ceil(T, 128)
        else:
            reduce_tiles = div_ceil(T - 1, 128)

        nc0_tiles = reduce_tiles // num_shards
        nc1_tiles = reduce_tiles - nc0_tiles

        if num_shards == 2:
            core_barrier(output, (0, 1))

        if shard_id == 0:
            reduce_outputs(output, nc0_tiles, reduce_tile_size, 0, H, sbm=sbm)

        if shard_id == 1:
            reduce_outputs(output, nc1_tiles, reduce_tile_size, nc0_tiles, H, sbm=sbm)

    sbm.close_scope()

    return output


@nki.jit
def bwmm_shard_on_block_hybrid(
    conditions: nl.NkiTensor,
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    block_size: int,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    down_activations: Optional[nl.NkiTensor] = None,
    activation_function: common_types.ActFnType = common_types.ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: common_types.ExpertAffinityScaleMode = common_types.ExpertAffinityScaleMode.POST_SCALE,
    n_block_per_iter: int = 1,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    non_overlapping_shards: bool = False,
):
    """Hybrid static/dynamic shard-on-block kernel.

    Static loop processes N-E blocks with compile-time optimizations.
    Dynamic loop handles remaining E blocks with early-exit for padded blocks.

    Args:
        conditions (nl.NkiTensor): [ceil(N/num_shards)+1] per-shard condition vector.
            1=active, 0=padded. Last entry must be 0 for loop termination.
        All other args: same as bwmm_shard_on_block.
    """
    kernel_assert(
        block_sharding_strategy == BlockShardStrategy.PING_PONG or block_sharding_strategy == BlockShardStrategy.HI_LO,
        "Only support PING_PONG and HI_LO sharding strategies",
    )

    T, H = hidden_states.shape
    B = block_size
    E, I_TP, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B
    NUM_TILES = B // TILE_SIZE
    shard_strat = block_sharding_strategy

    weights_dtype = compute_dtype
    _, num_shards, shard_id = get_program_sharding_info()
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=I_TP)
    dims.__post_init__()

    cfg = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        is_quant=gate_up_proj_scale is not None and down_proj_scale is not None,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
    )

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

    NUM_STATIC_BLOCKS = N - E
    # Ensure static blocks is even for PING_PONG (each shard gets equal work)
    if NUM_STATIC_BLOCKS % num_shards != 0:
        NUM_STATIC_BLOCKS = NUM_STATIC_BLOCKS - (NUM_STATIC_BLOCKS % num_shards)

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("bwmm_shard_on_block"),
    )

    # Step 1: call static kernel for first NUM_STATIC_BLOCKS blocks
    output = bwmm_shard_on_block(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        down_activations=down_activations,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        n_block_per_iter=n_block_per_iter,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        block_sharding_strategy=block_sharding_strategy,
        sbm=sbm,
        num_static_block=NUM_STATIC_BLOCKS,
        total_n_blocks=N,
        down_bias_tp_degree=down_bias_tp_degree,
        down_bias_tp_rank=down_bias_tp_rank,
        non_overlapping_shards=non_overlapping_shards,
    )

    # Step 2: dynamic loop over remaining blocks
    # For HI_LO: each shard processes contiguous blocks. conditions = [shard0_conds..., shard1_conds..., 0]
    #   shard 0 reads conditions[0..n_blocks_per_shard], shard 1 reads conditions[n_blocks_per_shard..]
    #   local_block_idx = shard_id * n_blocks_per_shard + shard_local_idx
    # For PING_PONG: shards interleave. conditions = [cond_0, cond_1, ..., 0]
    #   shared_block_idx increments by 1, local_block_idx = shared * num_shards + shard_id

    n_blocks_per_shard = div_ceil(N, num_shards)
    n_static_shard_blocks = NUM_STATIC_BLOCKS // num_shards
    n_dynamic_per_shard = n_blocks_per_shard - n_static_shard_blocks

    # Pre-allocate SBUF buffers for dynamic loop
    free_size = min(PSUM_SIZE, B)
    gup_tile_count = div_ceil(I_TP, TILE_SIZE)
    _H_DIV_128 = H // TILE_SIZE

    dyn_token_indices = nl.ndarray((TILE_SIZE, NUM_TILES), buffer=nl.sbuf, dtype=nl.int32)
    dyn_hidden_T = nl.ndarray((TILE_SIZE, 1, _H_DIV_128, NUM_TILES * TILE_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)

    # Placeholder for FP8
    gup_scale = None
    down_scale = None

    # Down bias TP sharding
    _down_bias_h_size = H // down_bias_tp_degree if down_bias_tp_degree is not None else H
    _down_bias_h_offset = down_bias_tp_rank * _down_bias_h_size if down_bias_tp_rank is not None else 0

    # Count active dynamic blocks for BOTH shards locally (no cross-shard communication needed)
    # Both shards have access to the full conditions tensor
    shard0_cond_offset = 0 * (n_blocks_per_shard + 1) + n_static_shard_blocks
    shard1_cond_offset = 1 * (n_blocks_per_shard + 1) + n_static_shard_blocks

    shard0_dyn_conds = nl.ndarray((1, n_dynamic_per_shard), buffer=nl.sbuf, dtype=nl.int32)
    nisa.dma_copy(
        dst=shard0_dyn_conds,
        src=conditions.reshape((conditions.shape[0], 1)).ap(
            pattern=[[1, n_dynamic_per_shard], [1, 1]], offset=shard0_cond_offset
        ),
    )
    count0 = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.tensor_reduce(dst=count0, data=shard0_dyn_conds, op=nl.add, axis=1)

    shard1_dyn_conds = nl.ndarray((1, n_dynamic_per_shard), buffer=nl.sbuf, dtype=nl.int32)
    nisa.dma_copy(
        dst=shard1_dyn_conds,
        src=conditions.reshape((conditions.shape[0], 1)).ap(
            pattern=[[1, n_dynamic_per_shard], [1, 1]], offset=shard1_cond_offset
        ),
    )
    count1 = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.tensor_reduce(dst=count1, data=shard1_dyn_conds, op=nl.add, axis=1)

    # min_count for Phase 2, remaining for Phase 3
    min_count = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.tensor_tensor(dst=min_count, data1=count0, data2=count1, op=nl.minimum)

    max_count = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.tensor_tensor(dst=max_count, data1=count0, data2=count1, op=nl.maximum)
    remaining = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.tensor_tensor(dst=remaining, data1=max_count, data2=min_count, op=nl.subtract)

    # Determine active shard for Phase 3: shard with more blocks
    # For HI_LO, padding blocks are at the end (high block indices), so shard 0 always has >= shard 1
    # active_sid = 0 for HI_LO
    active_sid = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.memset(active_sid, value=0)

    # Phase 2: synchronized dynamic loop — both shards run min_count iterations
    shard_local_idx = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)

    # Single register for both phases
    dyn_reg = nisa.register_alloc()

    # Phase 2: synchronized — both shards run min_count iterations
    shard_local_idx = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.memset(shard_local_idx, value=n_static_shard_blocks)
    nisa.register_load(dyn_reg, min_count)

    # I_TP chunking parameters (shared by Phase 2 and Phase 3)
    DYN_I_TILE = min(I_TP, TILE_SIZE * 4)  # 512 max per chunk
    DYN_I_CHUNK_COUNT = div_ceil(I_TP, DYN_I_TILE)

    def _phase2_process_chunk(_phase2_i):
        local_block_idx = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
        if shard_strat == BlockShardStrategy.HI_LO:
            nisa.tensor_scalar(
                dst=local_block_idx, data=shard_local_idx, op0=nl.add, operand0=shard_id * n_blocks_per_shard
            )
        else:
            nisa.tensor_scalar(dst=local_block_idx, data=shard_local_idx, op0=nl.multiply, operand0=num_shards)
            nisa.tensor_scalar(dst=local_block_idx, data=local_block_idx, op0=nl.add, operand0=shard_id)

        # Load token indices
        total_tok = int(token_position_to_id.shape[0])
        reshaped_tok = token_position_to_id.reshape((total_tok // B, B))
        for b_tile_idx in range(NUM_TILES):
            nisa.dma_copy(
                dst=dyn_token_indices[0:TILE_SIZE, b_tile_idx],
                src=reshaped_tok.ap(
                    pattern=[[1, TILE_SIZE], [1, 1]],
                    offset=TILE_SIZE * b_tile_idx,
                    scalar_offset=local_block_idx,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

        # Load block expert
        block_expert = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
        nisa.dma_copy(
            dst=block_expert.ap([[1, 1], [1, 1]]),
            src=block_to_expert.ap([[1, 1], [1, 1]], scalar_offset=local_block_idx),
        )

        # Transpose hidden states
        for b_tile_idx in range(NUM_TILES):
            _tok_p = nl.ndarray((TILE_SIZE, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_tok_p, src=dyn_token_indices[0:TILE_SIZE, b_tile_idx : b_tile_idx + 1], engine=nisa.scalar_engine
            )
            nisa.dma_transpose(
                dst=dyn_hidden_T[:, :, :, TILE_SIZE * b_tile_idx : TILE_SIZE * (b_tile_idx + 1)],
                src=hidden_states.ap(
                    pattern=[[H, TILE_SIZE], [1, 1], [TILE_SIZE, _H_DIV_128], [1, TILE_SIZE]],
                    offset=0,
                    vector_offset=_tok_p,
                    indirect_dim=0,
                ),
                axes=(3, 1, 2, 0),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

        # Load bias (if applicable)
        if cfg.linear_bias:
            gate_up_bias_T = load_and_transpose_gup_bias(inps, dims, cfg, block_expert, skip_dma, sbm=None)

        # I_TP chunking for SBUF budget

        # Load block_old before chunk loop (needed for accumulation)
        block_old = (
            bwmm_load_old_block(output, dyn_token_indices, NUM_TILES, compute_dtype, skip_dma, shard_id=0)
            if is_tensor_update_accumulating
            else None
        )

        # Expert affinity
        expert_affinity_f32 = None
        if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE:
            if is_tensor_update_accumulating:
                real_expert_u32 = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=real_expert_u32, data=block_expert, op0=nl.add, operand0=H)
                expert_affinity_f32 = []
                for tile_idx in range(NUM_TILES):
                    _aff = nl.ndarray((TILE_SIZE, 1), dtype=compute_dtype, buffer=nl.sbuf)
                    _aff_tv = (block_old[tile_idx]).select(dim=1, index=real_expert_u32).expand_dim(1)
                    nisa.tensor_copy(dst=_aff, src=_aff_tv, engine=nisa.scalar_engine)
                    expert_affinity_f32.append(_aff)
            else:
                expert_affinity_f32 = calculate_expert_affinities(
                    expert_affinities_masked, dyn_token_indices, block_expert, E, NUM_TILES, compute_dtype, skip_dma
                )

        # Down bias
        down_bias_raw = None
        if cfg.linear_bias:
            down_bias_raw = load_down_bias_raw(
                inps, dims, cfg, block_expert, skip_dma, sbm=None, bias_h_size=_down_bias_h_size
            )

        # Pre-allocate block_new for multi-chunk accumulation
        block_new_lst_pre = None
        if DYN_I_CHUNK_COUNT > 1:
            block_new_lst_pre = []
            for tile_idx in range(NUM_TILES):
                block_new_lst_pre.append(nl.ndarray((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf))

        for i_chunk_idx in range(DYN_I_CHUNK_COUNT):
            i_chunk_offset = i_chunk_idx * DYN_I_TILE
            cur_i_tile = min(DYN_I_TILE, I_TP - i_chunk_offset)
            cur_gup_tile_count = div_ceil(cur_i_tile, TILE_SIZE)

            # Allocate intermediate states for this chunk
            chunk_intermediate = []
            for gt in range(cur_gup_tile_count):
                chunk_intermediate.append(nl.ndarray((TILE_SIZE, B), dtype=compute_dtype, buffer=nl.sbuf))

            # Gate-up PSUM/SBUF for this chunk only
            chunk_psum = []
            chunk_sbuf = []
            for proj_idx in range(GUP_PROJ_DIM):
                p_lst = []
                s_lst = []
                for pt in range(dims.n_psum_tile_count):
                    pp = []
                    ss = []
                    for gt in range(cur_gup_tile_count):
                        pp.append(nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum))
                        ss.append(nl.ndarray((TILE_SIZE, free_size), dtype=nl.bfloat16, buffer=nl.sbuf))
                    p_lst.append(pp)
                    s_lst.append(ss)
                chunk_psum.append(p_lst)
                chunk_sbuf.append(s_lst)

            # Gate-up matmul with on-demand weight loading
            for h_tile_idx in range(dims.h_tile_count):
                for h_subtile_idx in range(dims.h_subtile_count_gup):
                    cur_gup_w = nl.ndarray(
                        (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, cur_i_tile),
                        dtype=gate_up_proj_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    h_offset = PSUM_SIZE * h_tile_idx + GUP_LOAD_COALESCE_FACTOR * TILE_SIZE * h_subtile_idx
                    h_remaining = H - h_offset
                    num_h = min(TILE_SIZE, div_ceil(h_remaining, GUP_LOAD_COALESCE_FACTOR))
                    if num_h < TILE_SIZE and not skip_dma.skip_weight:
                        nisa.memset(dst=cur_gup_w, value=0)
                    nisa.dma_copy(
                        dst=cur_gup_w[0:num_h, 0:GUP_LOAD_COALESCE_FACTOR, 0:GUP_PROJ_DIM, 0:cur_i_tile],
                        src=gate_up_proj_weight.ap(
                            pattern=[
                                [GUP_PROJ_DIM * I_TP, num_h],
                                [TILE_SIZE * GUP_PROJ_DIM * I_TP, GUP_LOAD_COALESCE_FACTOR],
                                [I_TP, GUP_PROJ_DIM],
                                [1, cur_i_tile],
                            ],
                            offset=h_offset * (GUP_PROJ_DIM * I_TP) + i_chunk_offset,
                            scalar_offset=block_expert,
                        ),
                        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                        dge_mode=nisa.dge_mode.hwdge,
                    )

                    for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                        for i_tile_idx in range(cur_gup_tile_count):
                            num_valid_k = min(TILE_SIZE, cur_i_tile - TILE_SIZE * i_tile_idx)
                            for proj_idx in range(GUP_PROJ_DIM):
                                for batch_tile_idx in range(dims.n_psum_tile_count):
                                    nisa.nc_matmul(
                                        dst=chunk_psum[proj_idx][batch_tile_idx][i_tile_idx][
                                            0:num_valid_k, nl.ds(0, free_size)
                                        ],
                                        stationary=cur_gup_w[
                                            nl.ds(0, TILE_SIZE),
                                            op_idx,
                                            proj_idx,
                                            nl.ds(TILE_SIZE * i_tile_idx, num_valid_k),
                                        ],
                                        moving=dyn_hidden_T[
                                            :,
                                            0,
                                            h_tile_idx * dims.h_subtile_count
                                            + h_subtile_idx * GUP_LOAD_COALESCE_FACTOR
                                            + op_idx,
                                            nl.ds(0, free_size),
                                        ],
                                    )

            # Copy PSUM to SBUF and apply bias (first chunk only for multi-chunk)
            for i_tile_idx in range(cur_gup_tile_count):
                num_valid_k = min(TILE_SIZE, cur_i_tile - TILE_SIZE * i_tile_idx)
                for proj_idx in range(GUP_PROJ_DIM):
                    for batch_tile_idx in range(dims.n_psum_tile_count):
                        if cfg.linear_bias:
                            global_bias_idx = (i_chunk_offset // TILE_SIZE) + i_tile_idx
                            nisa.tensor_scalar(
                                dst=chunk_sbuf[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size)
                                ],
                                data=chunk_psum[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size)
                                ],
                                operand0=gate_up_bias_T.ap(
                                    pattern=[[2 * dims.gup_tile_count, num_valid_k], [1, 1]],
                                    offset=global_bias_idx * 2 + proj_idx,
                                ),
                                op0=nl.add,
                            )
                        else:
                            nisa.tensor_copy(
                                dst=chunk_sbuf[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size)
                                ],
                                src=chunk_psum[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size)
                                ],
                                engine=nisa.scalar_engine,
                            )

            _clip_gate_up_projections(chunk_sbuf, cfg, dims, cur_gup_tile_count, free_size)

            chunk_intermediate = compute_intermediate_states(
                chunk_sbuf,
                B,
                cur_i_tile,
                compute_dtype,
                activation_function=activation_function,
                intermediate_states_lst=chunk_intermediate,
            )

            # Down projection for this chunk
            if DYN_I_CHUNK_COUNT > 1:
                chunk_block_old = None if i_chunk_idx == 0 else block_new_lst_pre
                chunk_accumulating = i_chunk_idx > 0
            else:
                chunk_block_old = block_old
                chunk_accumulating = is_tensor_update_accumulating

            block_new_lst = compute_block_output(
                chunk_intermediate,
                None,
                expert_affinity_f32 if DYN_I_CHUNK_COUNT == 1 else None,
                chunk_block_old,
                down_activations,
                local_block_idx,
                H,
                cur_i_tile,
                NUM_TILES,
                output_dtype=output.dtype if DYN_I_CHUNK_COUNT == 1 else compute_dtype,
                is_tensor_update_accumulating=chunk_accumulating,
                down_bias_raw=down_bias_raw if (DYN_I_CHUNK_COUNT == 1 or i_chunk_idx == 0) else None,
                down_scale=down_scale,
                sbm=None,
                skip_dma=skip_dma,
                down_proj_weight_hbm=down_proj_weight,
                block_expert=block_expert,
                block_new_lst_pre=block_new_lst_pre,
                i_tp_offset=i_chunk_offset,
                down_bias_h_offset=_down_bias_h_offset,
                down_bias_h_size=_down_bias_h_size if down_bias_tp_degree is not None else None,
            )

        # Post-chunk: apply affinity and bias if multi-chunk
        if DYN_I_CHUNK_COUNT > 1:
            h_i_upper = div_ceil(H, TOTAL_PSUM_SIZE)
            for token_tile_idx in range(NUM_TILES):
                for h_tile_idx in range(h_i_upper):
                    for h_subtile_idx in range(N_PSUM_BANKS):
                        psum_start = TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * h_subtile_idx
                        actual_psum_size = min(PSUM_SIZE, H - psum_start)
                        if actual_psum_size <= 0:
                            continue
                        sl = nl.ds(psum_start, actual_psum_size)
                        if expert_affinity_f32 is not None:
                            if is_tensor_update_accumulating:
                                nisa.scalar_tensor_tensor(
                                    dst=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                    data=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                    op0=nl.multiply,
                                    operand0=expert_affinity_f32[token_tile_idx][0:TILE_SIZE, 0],
                                    op1=nl.add,
                                    operand1=block_old[token_tile_idx][0:TILE_SIZE, sl],
                                )
                            else:
                                nisa.tensor_scalar(
                                    dst=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                    data=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                    operand0=expert_affinity_f32[token_tile_idx][0:TILE_SIZE, 0],
                                    op0=nl.multiply,
                                )
                        else:
                            if is_tensor_update_accumulating:
                                nisa.tensor_tensor(
                                    dst=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                    data1=block_new_lst[token_tile_idx][0:TILE_SIZE, sl],
                                    data2=block_old[token_tile_idx][0:TILE_SIZE, sl],
                                    op=nl.add,
                                )

        _store_block_output(
            output, block_new_lst, dyn_token_indices, H, NUM_TILES, 0, is_tensor_update_accumulating, skip_dma
        )

        nisa.tensor_scalar(dst=shard_local_idx, data=shard_local_idx, op0=nl.add, operand0=1)
        core_barrier(output, (0, 1))

    nl.fori_loop(0, dyn_reg, _phase2_process_chunk)

    # Phase 3: Block-split for imbalanced workload
    # active_local = n_static_shard_blocks + min_count (where the active shard left off)
    active_local_idx = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    nisa.tensor_scalar(dst=active_local_idx, data=min_count, op0=nl.add, operand0=n_static_shard_blocks)

    NUM_TILES_HALF = NUM_TILES // 2
    B_HALF = B // 2
    free_size_half = min(PSUM_SIZE, B_HALF)
    token_tile_offset = shard_id * NUM_TILES_HALF

    nisa.register_load(dyn_reg, remaining)

    def _phase3_process_chunk(_phase3_i):
        # Compute global block index for the active shard's block
        phase3_global = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
        # For HI_LO with active_sid=0: phase3_global = active_local_idx
        nisa.tensor_copy(dst=phase3_global, src=active_local_idx, engine=nisa.scalar_engine)

        # Load ALL token indices for the full block
        total_tok = int(token_position_to_id.shape[0])
        reshaped_tok = token_position_to_id.reshape((total_tok // B, B))
        full_token_indices = nl.ndarray((TILE_SIZE, NUM_TILES), buffer=nl.sbuf, dtype=nl.int32)
        for b_tile_idx in range(NUM_TILES):
            nisa.dma_copy(
                dst=full_token_indices[0:TILE_SIZE, b_tile_idx],
                src=reshaped_tok.ap(
                    pattern=[[1, TILE_SIZE], [1, 1]],
                    offset=TILE_SIZE * b_tile_idx,
                    scalar_offset=phase3_global,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

        # Each shard takes its half of token indices
        half_token_indices = nl.ndarray((TILE_SIZE, NUM_TILES_HALF), buffer=nl.sbuf, dtype=nl.int32)
        for t in range(NUM_TILES_HALF):
            nisa.tensor_copy(
                dst=half_token_indices[0:TILE_SIZE, t : t + 1],
                src=full_token_indices[0:TILE_SIZE, (token_tile_offset + t) : (token_tile_offset + t + 1)],
                engine=nisa.scalar_engine,
            )

        # Load block expert
        block_expert = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
        nisa.dma_copy(
            dst=block_expert.ap([[1, 1], [1, 1]]),
            src=block_to_expert.ap([[1, 1], [1, 1]], scalar_offset=phase3_global),
        )

        # Transpose hidden states (half tokens only)
        half_hidden_T = nl.ndarray(
            (TILE_SIZE, 1, _H_DIV_128, NUM_TILES_HALF * TILE_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        for b_tile_idx in range(NUM_TILES_HALF):
            _tok_p = nl.ndarray((TILE_SIZE, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_tok_p, src=half_token_indices[0:TILE_SIZE, b_tile_idx : b_tile_idx + 1], engine=nisa.scalar_engine
            )
            nisa.dma_transpose(
                dst=half_hidden_T[:, :, :, TILE_SIZE * b_tile_idx : TILE_SIZE * (b_tile_idx + 1)],
                src=hidden_states.ap(
                    pattern=[[H, TILE_SIZE], [1, 1], [TILE_SIZE, _H_DIV_128], [1, TILE_SIZE]],
                    offset=0,
                    vector_offset=_tok_p,
                    indirect_dim=0,
                ),
                axes=(3, 1, 2, 0),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

        if cfg.linear_bias:
            gate_up_bias_T = load_and_transpose_gup_bias(inps, dims, cfg, block_expert, skip_dma, sbm=None)

        # I_TP chunking for Phase 3 (same as Phase 2)
        n_psum_tile_count_half = div_ceil(B_HALF, PSUM_SIZE)

        # Load block_old before chunk loop
        block_old_h = (
            bwmm_load_old_block(output, half_token_indices, NUM_TILES_HALF, compute_dtype, skip_dma, shard_id=0)
            if is_tensor_update_accumulating
            else None
        )

        expert_affinity_h = None
        if (
            expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE
            and is_tensor_update_accumulating
        ):
            real_expert_u32 = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=real_expert_u32, data=block_expert, op0=nl.add, operand0=H)
            expert_affinity_h = []
            for tile_idx in range(NUM_TILES_HALF):
                _aff = nl.ndarray((TILE_SIZE, 1), dtype=compute_dtype, buffer=nl.sbuf)
                _aff_tv = (block_old_h[tile_idx]).select(dim=1, index=real_expert_u32).expand_dim(1)
                nisa.tensor_copy(dst=_aff, src=_aff_tv, engine=nisa.scalar_engine)
                expert_affinity_h.append(_aff)

        down_bias_raw_h = None
        if cfg.linear_bias:
            down_bias_raw_h = load_down_bias_raw(
                inps, dims, cfg, block_expert, skip_dma, sbm=None, bias_h_size=_down_bias_h_size
            )

        # Pre-allocate block_new for multi-chunk
        block_new_lst_pre_h = None
        if DYN_I_CHUNK_COUNT > 1:
            block_new_lst_pre_h = []
            for tile_idx in range(NUM_TILES_HALF):
                block_new_lst_pre_h.append(nl.ndarray((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf))

        for i_chunk_idx in range(DYN_I_CHUNK_COUNT):
            i_chunk_offset = i_chunk_idx * DYN_I_TILE
            cur_i_tile = min(DYN_I_TILE, I_TP - i_chunk_offset)
            cur_gup_tile_count = div_ceil(cur_i_tile, TILE_SIZE)

            chunk_intermediate_h = []
            for gt in range(cur_gup_tile_count):
                chunk_intermediate_h.append(nl.ndarray((TILE_SIZE, B_HALF), dtype=compute_dtype, buffer=nl.sbuf))

            chunk_psum_h = []
            chunk_sbuf_h = []
            for proj_idx in range(GUP_PROJ_DIM):
                p_lst = []
                s_lst = []
                for pt in range(n_psum_tile_count_half):
                    pp = []
                    ss = []
                    for gt in range(cur_gup_tile_count):
                        pp.append(nl.ndarray((TILE_SIZE, free_size_half), dtype=nl.float32, buffer=nl.psum))
                        ss.append(nl.ndarray((TILE_SIZE, free_size_half), dtype=nl.bfloat16, buffer=nl.sbuf))
                    p_lst.append(pp)
                    s_lst.append(ss)
                chunk_psum_h.append(p_lst)
                chunk_sbuf_h.append(s_lst)

            for h_tile_idx in range(dims.h_tile_count):
                for h_subtile_idx in range(dims.h_subtile_count_gup):
                    cur_gup_w = nl.ndarray(
                        (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, cur_i_tile),
                        dtype=gate_up_proj_weight.dtype,
                        buffer=nl.sbuf,
                    )
                    h_offset = PSUM_SIZE * h_tile_idx + GUP_LOAD_COALESCE_FACTOR * TILE_SIZE * h_subtile_idx
                    h_remaining = H - h_offset
                    num_h = min(TILE_SIZE, div_ceil(h_remaining, GUP_LOAD_COALESCE_FACTOR))
                    if num_h < TILE_SIZE and not skip_dma.skip_weight:
                        nisa.memset(dst=cur_gup_w, value=0)
                    nisa.dma_copy(
                        dst=cur_gup_w[0:num_h, 0:GUP_LOAD_COALESCE_FACTOR, 0:GUP_PROJ_DIM, 0:cur_i_tile],
                        src=gate_up_proj_weight.ap(
                            pattern=[
                                [GUP_PROJ_DIM * I_TP, num_h],
                                [TILE_SIZE * GUP_PROJ_DIM * I_TP, GUP_LOAD_COALESCE_FACTOR],
                                [I_TP, GUP_PROJ_DIM],
                                [1, cur_i_tile],
                            ],
                            offset=h_offset * (GUP_PROJ_DIM * I_TP) + i_chunk_offset,
                            scalar_offset=block_expert,
                        ),
                        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                        dge_mode=nisa.dge_mode.hwdge,
                    )

                    for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                        for i_tile_idx in range(cur_gup_tile_count):
                            num_valid_k = min(TILE_SIZE, cur_i_tile - TILE_SIZE * i_tile_idx)
                            for proj_idx in range(GUP_PROJ_DIM):
                                for batch_tile_idx in range(n_psum_tile_count_half):
                                    nisa.nc_matmul(
                                        dst=chunk_psum_h[proj_idx][batch_tile_idx][i_tile_idx][
                                            0:num_valid_k, nl.ds(0, free_size_half)
                                        ],
                                        stationary=cur_gup_w[
                                            nl.ds(0, TILE_SIZE),
                                            op_idx,
                                            proj_idx,
                                            nl.ds(TILE_SIZE * i_tile_idx, num_valid_k),
                                        ],
                                        moving=half_hidden_T[
                                            :,
                                            0,
                                            h_tile_idx * dims.h_subtile_count
                                            + h_subtile_idx * GUP_LOAD_COALESCE_FACTOR
                                            + op_idx,
                                            nl.ds(0, free_size_half),
                                        ],
                                    )

            for i_tile_idx in range(cur_gup_tile_count):
                num_valid_k = min(TILE_SIZE, cur_i_tile - TILE_SIZE * i_tile_idx)
                for proj_idx in range(GUP_PROJ_DIM):
                    for batch_tile_idx in range(n_psum_tile_count_half):
                        if cfg.linear_bias:
                            global_bias_idx = (i_chunk_offset // TILE_SIZE) + i_tile_idx
                            nisa.tensor_scalar(
                                dst=chunk_sbuf_h[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size_half)
                                ],
                                data=chunk_psum_h[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size_half)
                                ],
                                operand0=gate_up_bias_T.ap(
                                    pattern=[[2 * dims.gup_tile_count, num_valid_k], [1, 1]],
                                    offset=global_bias_idx * 2 + proj_idx,
                                ),
                                op0=nl.add,
                            )
                        else:
                            nisa.tensor_copy(
                                dst=chunk_sbuf_h[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size_half)
                                ],
                                src=chunk_psum_h[proj_idx][batch_tile_idx][i_tile_idx][
                                    0:num_valid_k, nl.ds(0, free_size_half)
                                ],
                                engine=nisa.scalar_engine,
                            )

            _clip_gate_up_projections(chunk_sbuf_h, cfg, dims, cur_gup_tile_count, free_size_half)

            chunk_intermediate_h = compute_intermediate_states(
                chunk_sbuf_h,
                B_HALF,
                cur_i_tile,
                compute_dtype,
                activation_function=activation_function,
                intermediate_states_lst=chunk_intermediate_h,
            )

            if DYN_I_CHUNK_COUNT > 1:
                chunk_block_old_h = None if i_chunk_idx == 0 else block_new_lst_pre_h
                chunk_accumulating_h = i_chunk_idx > 0
            else:
                chunk_block_old_h = block_old_h
                chunk_accumulating_h = is_tensor_update_accumulating

            block_new_h = compute_block_output(
                chunk_intermediate_h,
                None,
                expert_affinity_h if DYN_I_CHUNK_COUNT == 1 else None,
                chunk_block_old_h,
                down_activations,
                phase3_global,
                H,
                cur_i_tile,
                NUM_TILES_HALF,
                output_dtype=output.dtype if DYN_I_CHUNK_COUNT == 1 else compute_dtype,
                is_tensor_update_accumulating=chunk_accumulating_h,
                down_bias_raw=down_bias_raw_h if (DYN_I_CHUNK_COUNT == 1 or i_chunk_idx == 0) else None,
                down_scale=down_scale,
                sbm=None,
                skip_dma=skip_dma,
                down_proj_weight_hbm=down_proj_weight,
                block_expert=block_expert,
                block_new_lst_pre=block_new_lst_pre_h,
                i_tp_offset=i_chunk_offset,
                down_bias_h_offset=_down_bias_h_offset,
                down_bias_h_size=_down_bias_h_size if down_bias_tp_degree is not None else None,
            )

        # Post-chunk for Phase 3
        if DYN_I_CHUNK_COUNT > 1:
            h_i_upper = div_ceil(H, TOTAL_PSUM_SIZE)
            for token_tile_idx in range(NUM_TILES_HALF):
                for h_tile_idx in range(h_i_upper):
                    for h_subtile_idx in range(N_PSUM_BANKS):
                        psum_start = TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * h_subtile_idx
                        actual_psum_size = min(PSUM_SIZE, H - psum_start)
                        if actual_psum_size <= 0:
                            continue
                        sl = nl.ds(psum_start, actual_psum_size)
                        if expert_affinity_h is not None:
                            if is_tensor_update_accumulating:
                                nisa.scalar_tensor_tensor(
                                    dst=block_new_h[token_tile_idx][0:TILE_SIZE, sl],
                                    data=block_new_h[token_tile_idx][0:TILE_SIZE, sl],
                                    op0=nl.multiply,
                                    operand0=expert_affinity_h[token_tile_idx][0:TILE_SIZE, 0],
                                    op1=nl.add,
                                    operand1=block_old_h[token_tile_idx][0:TILE_SIZE, sl],
                                )
                            else:
                                nisa.tensor_scalar(
                                    dst=block_new_h[token_tile_idx][0:TILE_SIZE, sl],
                                    data=block_new_h[token_tile_idx][0:TILE_SIZE, sl],
                                    operand0=expert_affinity_h[token_tile_idx][0:TILE_SIZE, 0],
                                    op0=nl.multiply,
                                )
                        else:
                            if is_tensor_update_accumulating:
                                nisa.tensor_tensor(
                                    dst=block_new_h[token_tile_idx][0:TILE_SIZE, sl],
                                    data1=block_new_h[token_tile_idx][0:TILE_SIZE, sl],
                                    data2=block_old_h[token_tile_idx][0:TILE_SIZE, sl],
                                    op=nl.add,
                                )

        _store_block_output(
            output, block_new_h, half_token_indices, H, NUM_TILES_HALF, 0, is_tensor_update_accumulating, skip_dma
        )

        nisa.tensor_scalar(dst=active_local_idx, data=active_local_idx, op0=nl.add, operand0=1)
        core_barrier(output, (0, 1))

    nl.fori_loop(0, dyn_reg, _phase3_process_chunk)

    core_barrier(output, (0, 1))
    return output


def compute_same_weights_block_parallel_hbm(
    N: int,
    block_to_expert: nl.NkiTensor,
    num_shards: int,
    shard_id: int,
    shard_strat: BlockShardStrategy,
    sbm: Optional[SbufManager] = None,
) -> nl.NkiTensor:
    """
    Compute weight reuse mask for block-parallel execution.

    Determines which blocks can reuse previously loaded weights by comparing
    expert indices between consecutive blocks in the sharding pattern.

    Args:
        N (int): Total number of blocks
        block_to_expert (nl.NkiTensor): Expert assignment for each block
        num_shards (int): Number of shards for parallel execution
        shard_id (int): Current shard identifier
        shard_strat (BlockShardStrategy): Block distribution strategy

    Returns:
        nl.NkiTensor: Boolean mask indicating weight reuse opportunities
    """
    kernel_assert(
        shard_strat == BlockShardStrategy.PING_PONG or shard_strat == BlockShardStrategy.HI_LO,
        "only support PING_PONG and HI_LO",
    )
    if sbm is not None:
        sbm.open_scope(name="compute_same_weights")
    n_blocks_per_shard = div_ceil(N, num_shards)
    n_tile_minus_1 = div_ceil(n_blocks_per_shard - 1, TILE_SIZE)
    tile_size = min(TILE_SIZE, n_blocks_per_shard - 1)

    # Strategy-dependent parameters
    if shard_strat == BlockShardStrategy.PING_PONG:
        stride = 2
        base_offset = shard_id
        off_one_offset = 2
    else:  # HI_LO
        stride = 1
        base_offset = shard_id * n_blocks_per_shard
        off_one_offset = 1

    # max_linear_index is the number of valid comparisons
    if shard_strat == BlockShardStrategy.PING_PONG:
        max_linear_index = (N - shard_id - off_one_offset) // 2
    else:  # HI_LO
        max_linear_index = max(0, min(n_blocks_per_shard, N - shard_id * n_blocks_per_shard) - 1)
    # Cap to ensure HBM store index j+1 stays within [0, n_blocks_per_shard)
    max_linear_index = min(max_linear_index, n_blocks_per_shard - 1)
    num_full_rows = min(tile_size, max(0, max_linear_index // n_tile_minus_1))
    num_acc_p_1 = num_full_rows
    num_acc_f_1 = n_tile_minus_1 if num_full_rows > 0 else 0

    if num_full_rows < tile_size and max_linear_index > 0:
        num_partial_f = max_linear_index % n_tile_minus_1
        num_acc_p_2 = 1 if num_partial_f > 0 else 0
        num_acc_f_2 = num_partial_f
    else:
        num_acc_p_2 = 0
        num_acc_f_2 = 0

    # ----------------------------------------------------------------
    # Allocate SBUF tensors
    # ----------------------------------------------------------------
    # Full-row tensors (partition dim always starts at 0)
    _expert_shape = block_to_expert.ap(
        pattern=[[stride * n_tile_minus_1, tile_size], [stride, n_tile_minus_1]],
        offset=base_offset,
    ).shape
    if sbm is not None:
        all_expert_indices = sbm.alloc_stack(_expert_shape, dtype=nl.int32, name="same_w_expert_idx")
        all_expert_indices_off_one = sbm.alloc_stack(
            (tile_size, n_tile_minus_1), dtype=nl.int32, name="same_w_expert_idx_off1"
        )
        is_weight_same_as_prev = sbm.alloc_stack((tile_size, n_tile_minus_1), dtype=nl.uint8, name="same_w_mask")
        partial_expert_indices = sbm.alloc_stack((1, n_tile_minus_1), dtype=nl.int32, name="same_w_partial_idx")
        partial_expert_indices_off_one = sbm.alloc_stack(
            (1, n_tile_minus_1), dtype=nl.int32, name="same_w_partial_idx_off1"
        )
        partial_is_weight_same = sbm.alloc_stack((1, n_tile_minus_1), dtype=nl.uint8, name="same_w_partial_mask")
    else:
        all_expert_indices = nl.ndarray(_expert_shape, dtype=nl.int32, buffer=nl.sbuf)
        all_expert_indices_off_one = nl.ndarray((tile_size, n_tile_minus_1), dtype=nl.int32, buffer=nl.sbuf)
        is_weight_same_as_prev = nl.ndarray((tile_size, n_tile_minus_1), dtype=nl.uint8, buffer=nl.sbuf)
        partial_expert_indices = nl.ndarray((1, n_tile_minus_1), dtype=nl.int32, buffer=nl.sbuf)
        partial_expert_indices_off_one = nl.ndarray((1, n_tile_minus_1), dtype=nl.int32, buffer=nl.sbuf)
        partial_is_weight_same = nl.ndarray((1, n_tile_minus_1), dtype=nl.uint8, buffer=nl.sbuf)

    # ----------------------------------------------------------------
    # Load expert indices for current blocks
    # ----------------------------------------------------------------
    # Pattern 1 - Full rows:
    #   Loads block_to_expert[shard_id + 2*n_tile_minus_1*p + 2*f]
    #   for p in [0, num_acc_p_1), f in [0, num_acc_f_1)
    if num_acc_p_1 > 0:
        nisa.dma_copy(
            dst=all_expert_indices[0:num_acc_p_1, 0:num_acc_f_1],
            src=block_to_expert.ap(
                pattern=[
                    [stride * n_tile_minus_1, num_acc_p_1],
                    [stride, num_acc_f_1],
                ],
                offset=base_offset,
            ),
        )

    # Pattern 2 - Partial row into separate 1-row tensor:
    #   Loads block_to_expert[shard_id + 2*n_tile_minus_1*num_full_rows + 2*f]
    #   for f in [0, num_acc_f_2)
    if num_acc_p_2 > 0:
        partial_row_offset = base_offset + stride * n_tile_minus_1 * num_full_rows
        nisa.dma_copy(
            dst=partial_expert_indices[0:1, 0:num_acc_f_2],
            src=block_to_expert.ap(
                pattern=[
                    [stride * n_tile_minus_1, 1],
                    [stride, num_acc_f_2],
                ],
                offset=partial_row_offset,
            ),
        )

    # ----------------------------------------------------------------
    # Load expert indices for next blocks (offset by off_one_offset=2)
    # ----------------------------------------------------------------
    # Pattern 1 - Full rows:
    #   Loads block_to_expert[shard_id + 2 + 2*n_tile_minus_1*p + 2*f]
    #   for p in [0, num_acc_p_1), f in [0, num_acc_f_1)
    if num_acc_p_1 > 0:
        nisa.dma_copy(
            dst=all_expert_indices_off_one[0:num_acc_p_1, 0:num_acc_f_1],
            src=block_to_expert.ap(
                pattern=[
                    [stride * n_tile_minus_1, num_acc_p_1],
                    [stride, num_acc_f_1],
                ],
                offset=base_offset + off_one_offset,
            ),
        )

    # Pattern 2 - Partial row into separate 1-row tensor:
    #   Loads block_to_expert[shard_id + 2 + 2*n_tile_minus_1*num_full_rows + 2*f]
    #   for f in [0, num_acc_f_2)
    if num_acc_p_2 > 0:
        partial_row_offset_off_one = base_offset + off_one_offset + stride * n_tile_minus_1 * num_full_rows
        nisa.dma_copy(
            dst=partial_expert_indices_off_one[0:1, 0:num_acc_f_2],
            src=block_to_expert.ap(
                pattern=[
                    [stride * n_tile_minus_1, 1],
                    [stride, num_acc_f_2],
                ],
                offset=partial_row_offset_off_one,
            ),
        )

    # ----------------------------------------------------------------
    # Compare: is expert[block j] == expert[block j+1]?
    # ----------------------------------------------------------------
    # Full rows
    if num_acc_p_1 > 0:
        nisa.tensor_tensor(
            data1=all_expert_indices[0:num_acc_p_1, 0:num_acc_f_1],
            data2=all_expert_indices_off_one[0:num_acc_p_1, 0:num_acc_f_1],
            op=nl.equal,
            dst=is_weight_same_as_prev[0:num_acc_p_1, 0:num_acc_f_1],
        )

    # Partial row — operates on separate 1-row tensors (partition dim = 0)
    if num_acc_p_2 > 0:
        nisa.tensor_tensor(
            data1=partial_expert_indices[0:1, 0:num_acc_f_2],
            data2=partial_expert_indices_off_one[0:1, 0:num_acc_f_2],
            op=nl.equal,
            dst=partial_is_weight_same[0:1, 0:num_acc_f_2],
        )

    # Workaround alignment issue for the equal op
    free_size = 4 - n_tile_minus_1 % 4
    if sbm is not None:
        zero_index = sbm.alloc_stack((1, free_size), dtype=nl.uint8, name="same_w_zero_idx")
    else:
        zero_index = nl.ndarray((1, free_size), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=zero_index, value=0)

    # ----------------------------------------------------------------
    # Store to HBM
    # ----------------------------------------------------------------
    # HBM layout: is_weight_same_as_prev_hbm[b] for b in [0, n_blocks_per_shard)
    #   b=0: always 0 (first block cannot reuse)
    #   b=j+1: result of comparing block j with block j+1
    #
    # SBUF 2D layout: linear index j = n_tile_minus_1 * p + f
    #   => HBM index = j + 1 = n_tile_minus_1 * p + f + 1
    #
    # We reuse the SAME counts from the compute phase so we only
    # store exactly what was computed — no separate HBM row counting.
    # ----------------------------------------------------------------
    is_weight_same_as_prev_hbm = nl.ndarray((n_blocks_per_shard,), dtype=nl.uint8, buffer=nl.private_hbm)

    # Store index 0 = 0 (first block can never reuse weights)
    nisa.dma_copy(dst=is_weight_same_as_prev_hbm[0:1], src=zero_index[0:1, 0:1])

    # Store full rows:
    #   HBM[n_tile_minus_1 * p + f + 1] = is_weight_same_as_prev[p, f]
    #   for p in [0, num_acc_p_1), f in [0, num_acc_f_1)
    if num_acc_p_1 > 0:
        nisa.dma_copy(
            dst=is_weight_same_as_prev_hbm.ap(
                pattern=[
                    [n_tile_minus_1, num_acc_p_1],
                    [1, num_acc_f_1],
                ],
                offset=1,
            ),
            src=is_weight_same_as_prev[0:num_acc_p_1, 0:num_acc_f_1],
        )

    # Store partial row:
    #   HBM[n_tile_minus_1 * num_full_rows + f + 1] = partial_is_weight_same[0, f]
    #   for f in [0, num_acc_f_2)
    if num_acc_p_2 > 0:
        partial_row_offset_hbm = 1 + n_tile_minus_1 * num_full_rows
        nisa.dma_copy(
            dst=is_weight_same_as_prev_hbm.ap(
                pattern=[
                    [n_tile_minus_1, 1],
                    [1, num_acc_f_2],
                ],
                offset=partial_row_offset_hbm,
            ),
            src=partial_is_weight_same[0:1, 0:num_acc_f_2],
        )

    if sbm is not None:
        sbm.close_scope()
    return is_weight_same_as_prev_hbm


def load_down_proj_weight(
    down_proj_weight: nl.NkiTensor,
    block_expert: nl.NkiTensor,
    compute_dtype,
    skip_dma: SkipMode = SkipMode(),
    load_dst: Optional[list] = None,
    sbm: Optional[SbufManager] = None,
) -> list:
    """
    Load down projection weights.

    Args:
        down_proj_weight: Weight tensor with shape [E, I_TP, H]
        block_expert: Expert index tensor with shape (1, 1) in SBUF
        compute_dtype: Compute data type
        skip_dma: DMA skip configuration
        load_dst: Optional pre-allocated destination list

    Returns:
        List of weight tensors [gup_n_tile] each with shape (TILE_SIZE, H)

    Notes:
        - Assumes I_TP is divisible by 16 for vector operations
        - Partial tiles are zero-padded
        - Uses scalar_offset for dynamic expert indexing
    """
    kernel_assert(len(down_proj_weight.shape) == 3, "Unsupported down_proj_weight layout, should be [E, I_TP, H]")
    _, I_TP, H = down_proj_weight.shape
    kernel_assert(
        I_TP % 16 == 0, "Vector DGE expects the partition dimension to be either 1 or a multiple of 16. Please pad it."
    )

    gup_n_tile = div_ceil(I_TP, TILE_SIZE)

    if load_dst is None:
        load_dst = []
        for alloc_idx in range(gup_n_tile):
            if sbm is not None:
                load_dst.append(
                    sbm.alloc_stack((TILE_SIZE, H), dtype=down_proj_weight.dtype, name=f"down_w_load_i{alloc_idx}")
                )
            else:
                load_dst.append(nl.ndarray((TILE_SIZE, H), dtype=down_proj_weight.dtype, buffer=nl.sbuf))

    if down_proj_weight.dtype != compute_dtype:
        dp_weights = []
        for alloc_idx in range(gup_n_tile):
            if sbm is not None:
                dp_weights.append(
                    sbm.alloc_stack((TILE_SIZE, H), dtype=compute_dtype, name=f"down_w_cast_i{alloc_idx}")
                )
            else:
                dp_weights.append(nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf))

    for i_tile_idx in range(gup_n_tile):
        i_start = TILE_SIZE * i_tile_idx
        num_i = min(TILE_SIZE, I_TP - i_start)

        # Use .ap() with scalar_offset for dynamic expert indexing
        # down_proj_weight shape: [E, I_TP, H]
        # Strides: [I_TP*H, H, 1]
        # Access: [block_expert[0,0], i_start:i_start+num_i, 0:H]
        # Pattern: [[H, num_i], [1, H]] for (I_TP, H) dimensions
        # Static offset: i_start * H
        # scalar_offset: block_expert (compiler multiplies by I_TP*H)

        offset = i_start * H

        nisa.dma_copy(
            dst=load_dst[i_tile_idx][0:num_i, 0:H],
            src=down_proj_weight.ap(pattern=[[H, num_i], [1, H]], offset=offset, scalar_offset=block_expert),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=nisa.dge_mode.hwdge,
        )

        # Type conversion if needed
        if down_proj_weight.dtype != compute_dtype:
            # Initialize with zeros for masked-out elements
            if num_i < TILE_SIZE:
                nisa.memset(dst=dp_weights[i_tile_idx], value=0)

            nisa.tensor_copy(dst=dp_weights[i_tile_idx][0:num_i, 0:H], src=load_dst[i_tile_idx][0:num_i, 0:H])

    return load_dst if down_proj_weight.dtype == compute_dtype else dp_weights


def load_gate_up_proj_weights(
    gate_up_proj_weight: nl.NkiTensor,
    block_expert: nl.NkiTensor,
    compute_dtype,
    skip_dma: SkipMode = SkipMode(),
    load_dst: Optional[list] = None,
    sbm: Optional[SbufManager] = None,
) -> list:
    """
    Load gate and up projection weights.

    Args:
        gate_up_proj_weight: Weight tensor with shape [E, H, 2, I_TP]
        block_expert: Expert index tensor with shape (1, 1) in SBUF
        compute_dtype: Compute data type
        skip_dma: DMA skip configuration
        load_dst: Optional pre-allocated destination list

    Returns:
        Nested list [h_outer][h_inner] of weight tensors each with shape (TILE_SIZE, 2, I_TP)

    Notes:
        - Gate and up projections are interleaved in dimension 2
        - Partial tiles are zero-padded
        - Uses scalar_offset for dynamic expert indexing
    """
    kernel_assert(
        len(gate_up_proj_weight.shape) == 4, "Unsupported gate_up_proj_weight layout, should be [E, H, 2, I_TP]"
    )

    _, H, _, I_TP = gate_up_proj_weight.shape
    h_tile_count = div_ceil(H, PSUM_SIZE)
    h_subtile_count_gup = PSUM_SIZE // TILE_SIZE // GUP_LOAD_COALESCE_FACTOR

    if load_dst is None:
        load_dst = []
        for h_tile_idx in range(h_tile_count):
            h_j_lst = []
            for h_subtile_idx in range(h_subtile_count_gup):
                if sbm is not None:
                    h_j_lst.append(
                        sbm.alloc_stack(
                            (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP),
                            dtype=gate_up_proj_weight.dtype,
                            name=f"gup_w_load_h{h_tile_idx}_s{h_subtile_idx}",
                        )
                    )
                else:
                    h_j_lst.append(
                        nl.ndarray(
                            (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP),
                            dtype=gate_up_proj_weight.dtype,
                            buffer=nl.sbuf,
                        )
                    )
            load_dst.append(h_j_lst)

    if gate_up_proj_weight.dtype != compute_dtype:
        gup_weights = []
        for h_tile_idx in range(h_tile_count):
            h_j_lst = []
            for h_subtile_idx in range(h_subtile_count_gup):
                if sbm is not None:
                    h_j_lst.append(
                        sbm.alloc_stack(
                            (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP),
                            dtype=compute_dtype,
                            name=f"gup_w_cast_h{h_tile_idx}_s{h_subtile_idx}",
                        )
                    )
                else:
                    h_j_lst.append(
                        nl.ndarray(
                            (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP),
                            dtype=compute_dtype,
                            buffer=nl.sbuf,
                        )
                    )
            gup_weights.append(h_j_lst)

    for h_tile_idx in range(h_tile_count):
        for h_subtile_idx in range(h_subtile_count_gup):
            h_offset = PSUM_SIZE * h_tile_idx + GUP_LOAD_COALESCE_FACTOR * TILE_SIZE * h_subtile_idx
            h_remaining = H - h_offset
            num_h = min(TILE_SIZE, div_ceil(h_remaining, GUP_LOAD_COALESCE_FACTOR))

            # Initialize with zeros for partial tiles
            if num_h < TILE_SIZE:
                if not skip_dma.skip_weight:
                    nisa.memset(dst=load_dst[h_tile_idx][h_subtile_idx], value=0)

            # Use .ap() with scalar_offset for dynamic expert indexing
            # gate_up_proj_weight shape: [E, H, 2, I_TP]
            # Linearized strides: [H*2*I_TP, 2*I_TP, I_TP, 1]
            # Access: [block_expert[0,0], h_offset+p, g, i] for p∈[0,num_h), g∈[0,2), i∈[0,I_TP)
            # Pattern: [[2*I_TP, num_h], [I_TP, 2], [1, I_TP]]
            # Static offset: h_offset * 2 * I_TP
            # scalar_offset: block_expert (compiler multiplies by H*2*I_TP)

            offset = h_offset * (GUP_PROJ_DIM * I_TP)

            nisa.dma_copy(
                dst=load_dst[h_tile_idx][h_subtile_idx][0:num_h, 0:GUP_LOAD_COALESCE_FACTOR, 0:GUP_PROJ_DIM, 0:I_TP],
                src=gate_up_proj_weight.ap(
                    pattern=[
                        [GUP_PROJ_DIM * I_TP, num_h],
                        [TILE_SIZE * GUP_PROJ_DIM * I_TP, GUP_LOAD_COALESCE_FACTOR],
                        [I_TP, GUP_PROJ_DIM],
                        [1, I_TP],
                    ],
                    offset=offset,
                    scalar_offset=block_expert,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=nisa.dge_mode.hwdge,
            )

            # Type conversion if needed
            if gate_up_proj_weight.dtype != compute_dtype:
                # Initialize with zeros for partial tiles
                if num_h < TILE_SIZE:
                    if not skip_dma.skip_weight:
                        nisa.memset(dst=gup_weights[h_tile_idx][h_subtile_idx], value=0)

                nisa.tensor_copy(
                    dst=gup_weights[h_tile_idx][h_subtile_idx][
                        0:num_h, 0:GUP_LOAD_COALESCE_FACTOR, 0:GUP_PROJ_DIM, 0:I_TP
                    ],
                    src=load_dst[h_tile_idx][h_subtile_idx][
                        0:num_h, 0:GUP_LOAD_COALESCE_FACTOR, 0:GUP_PROJ_DIM, 0:I_TP
                    ],
                )

    return load_dst if gate_up_proj_weight.dtype == compute_dtype else gup_weights


def compute_block_output(
    intermediate_states,
    dp_weights,
    expert_affinity,
    block_old,
    down_activations,
    block_idx,
    H,
    I_TP,
    NUM_TILES,
    output_dtype,
    is_tensor_update_accumulating,
    down_bias_broadcasted=None,
    down_bias_raw=None,
    down_scale=None,
    sbm: Optional[SbufManager] = None,
    down_proj_weight_hbm=None,
    block_expert=None,
    skip_dma: SkipMode = SkipMode(),
    block_new_lst_pre=None,
    i_tp_offset=0,
    down_bias_h_offset: int = 0,
    down_bias_h_size: Optional[int] = None,
):
    """
    Compute block output with down projection and expert affinity scaling.

    Performs down projection (intermediate @ down_weights) and applies expert
    affinity scaling with optional bias addition and accumulation.

    Args:
        intermediate_states (list): Intermediate activation states [gup_tile_count][TILE_SIZE, B]
        dp_weights (list): Down projection weights [gup_tile_count][TILE_SIZE, H]
        expert_affinity (list, optional): Expert affinities [NUM_TILES][TILE_SIZE, 1]
        block_old (list, optional): Previous block outputs for accumulation [NUM_TILES][TILE_SIZE, H]
        down_activations (nl.NkiTensor, optional): Storage for intermediate activations
        block_idx (int): Current block index
        H (int): Hidden dimension size
        I_TP (int): Intermediate dimension size
        NUM_TILES (int): Number of tiles per block
        output_dtype (nki.dtype): Output data type
        compute_dtype (nki.dtype): Computation data type
        is_tensor_update_accumulating (bool): Enable accumulation mode
        down_bias_broadcasted (nl.NkiTensor, optional): Broadcasted bias [TILE_SIZE, H]
        allocate (bool): Unused parameter
        down_scale (nl.NkiTensor, optional): Dequantization scales

    Returns:
        list: Block output tensors [NUM_TILES][TILE_SIZE, H]

    Notes:
        - Supports FP8 dequantization with down_scale
        - Accumulation mode for TopK > 1 scenarios
        - Optional bias addition before affinity scaling
    """
    if block_new_lst_pre is not None:
        block_new_lst = block_new_lst_pre
    else:
        block_new_lst = []
        for tile_idx in range(NUM_TILES):
            if sbm is not None:
                block_new_lst.append(sbm.alloc_stack((TILE_SIZE, H), dtype=output_dtype, name=f"block_new_t{tile_idx}"))
            else:
                block_new_lst.append(nl.ndarray((TILE_SIZE, H), dtype=output_dtype, buffer=nl.sbuf))
    gup_n_tile = div_ceil(I_TP, TILE_SIZE)
    h_i_upper = div_ceil(H, TOTAL_PSUM_SIZE)
    H_NUM_PSUM_TILES = div_ceil(H, PSUM_SIZE)
    if down_bias_broadcasted is not None:
        kernel_assert(
            len(down_bias_broadcasted.shape) == 2,
            f"Expected down_bias_broadcasted to have shape [{TILE_SIZE}, {H}], got {down_bias_broadcasted.shape}",
        )
        kernel_assert(
            down_bias_broadcasted.shape[0] == TILE_SIZE,
            f"Expected down_bias_broadcasted to have shape [{TILE_SIZE}, {H}], got {down_bias_broadcasted.shape}",
        )
        kernel_assert(
            down_bias_broadcasted.shape[1] == H,
            f"Expected down_bias_broadcasted to have shape [{TILE_SIZE}, {H}], got {down_bias_broadcasted.shape}",
        )
    # Pre-allocate down weight buffers for on-demand loading (one per i_tile, loaded once, reused across h_subtiles)
    dp_weights_ondemand = None
    if dp_weights is None and down_proj_weight_hbm is not None:
        dp_weights_ondemand = []
        for alloc_idx in range(gup_n_tile):
            if sbm is not None:
                dp_weights_ondemand.append(sbm.alloc_stack((TILE_SIZE, H), dtype=down_proj_weight_hbm.dtype))
            else:
                dp_weights_ondemand.append(nl.ndarray((TILE_SIZE, H), dtype=down_proj_weight_hbm.dtype, buffer=nl.sbuf))

    _bias_ones_buf = None
    if down_bias_raw is not None:
        if sbm is not None:
            _bias_ones_buf = sbm.alloc_stack((1, TILE_SIZE), dtype=nl.bfloat16, name="bias_ones")
        else:
            _bias_ones_buf = nl.ndarray((1, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=_bias_ones_buf, value=1.0)

    for token_tile_idx in range(NUM_TILES):
        for h_tile_idx in range(h_i_upper):
            # Load down weights ONCE per (token_tile, h_tile) — reused across all h_subtiles
            if dp_weights is None and dp_weights_ondemand is not None:
                for i_tile_idx in range(gup_n_tile):
                    i_global_start = i_tp_offset + TILE_SIZE * i_tile_idx
                    num_i = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
                    nisa.dma_copy(
                        dst=dp_weights_ondemand[i_tile_idx][0:num_i, 0:H],
                        src=down_proj_weight_hbm.ap(
                            pattern=[[H, num_i], [1, H]],
                            offset=i_global_start * H,
                            scalar_offset=block_expert,
                        ),
                        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                        dge_mode=nisa.dge_mode.hwdge,
                    )

            down_proj_psum_lst = []
            for _ in range(N_PSUM_BANKS):
                tmp = nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=nl.float32, buffer=nl.psum)
                down_proj_psum_lst.append(tmp)

            if down_bias_raw is not None:
                _bias_h_size = down_bias_h_size if down_bias_h_size is not None else H
                _bias_h_offset = down_bias_h_offset
                for _bs_idx in range(N_PSUM_BANKS):
                    _bs_start = TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * _bs_idx
                    _bs_size = min(PSUM_SIZE, H - _bs_start)
                    _bs_end = _bs_start + _bs_size
                    # Check if PSUM tile [_bs_start, _bs_end) overlaps bias [_bias_h_offset, _bias_h_offset+_bias_h_size)
                    if _bs_size > 0 and _bs_start < _bias_h_offset + _bias_h_size and _bs_end > _bias_h_offset:
                        _overlap_start = max(_bs_start, _bias_h_offset)
                        _overlap_end = min(_bs_end, _bias_h_offset + _bias_h_size)
                        _psum_local_offset = _overlap_start - _bs_start
                        _bias_local_offset = _overlap_start - _bias_h_offset
                        _overlap_size = _overlap_end - _overlap_start
                        nisa.nc_matmul(
                            dst=down_proj_psum_lst[_bs_idx][0:TILE_SIZE, nl.ds(_psum_local_offset, _overlap_size)],
                            stationary=_bias_ones_buf[:, :TILE_SIZE],
                            moving=down_bias_raw[:, nl.ds(_bias_local_offset, _overlap_size)],
                            is_stationary_onezero=True,
                        )

            for h_subtile_idx in range(N_PSUM_BANKS):
                for i_tile_idx in range(gup_n_tile):
                    cur_dp_w = dp_weights[i_tile_idx] if dp_weights is not None else dp_weights_ondemand[i_tile_idx]

                    # Mask 1: H dimension - compute actual psum size
                    psum_start = TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * h_subtile_idx
                    actual_psum_size = min(PSUM_SIZE, H - psum_start)

                    # Mask 2: K dimension (rows) - compute valid rows
                    num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
                    if actual_psum_size > 0:
                        nisa.nc_matmul(
                            stationary=intermediate_states[i_tile_idx][
                                0:num_valid_k, nl.ds(TILE_SIZE * token_tile_idx, TILE_SIZE)
                            ],
                            moving=cur_dp_w[nl.ds(0, num_valid_k), nl.ds(psum_start, actual_psum_size)],
                            dst=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                        )

                if expert_affinity is not None:
                    if is_tensor_update_accumulating:
                        if down_scale is not None:
                            idx = (N_PSUM_BANKS * h_tile_idx) + h_subtile_idx
                            if idx < H_NUM_PSUM_TILES:
                                nisa.tensor_tensor(
                                    dst=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data1=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data2=down_scale[0:TILE_SIZE, idx, 0:PSUM_SIZE],
                                    op=nl.multiply,
                                )
                        if actual_psum_size > 0:
                            nisa.scalar_tensor_tensor(
                                dst=block_new_lst[token_tile_idx][0:TILE_SIZE, nl.ds(psum_start, actual_psum_size)],
                                data=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                op0=nl.multiply,
                                operand0=expert_affinity[token_tile_idx][0:TILE_SIZE, 0],
                                op1=nl.add,
                                operand1=block_old[token_tile_idx][0:TILE_SIZE, nl.ds(psum_start, actual_psum_size)],
                            )
                    else:
                        if down_scale is not None:
                            idx = (N_PSUM_BANKS * h_tile_idx) + h_subtile_idx
                            if idx < H_NUM_PSUM_TILES:
                                nisa.tensor_tensor(
                                    dst=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data1=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data2=down_scale[0:TILE_SIZE, idx, 0:PSUM_SIZE],
                                    op=nl.multiply,
                                )
                        if actual_psum_size > 0:
                            nisa.tensor_scalar(
                                dst=block_new_lst[token_tile_idx][0:TILE_SIZE, nl.ds(psum_start, actual_psum_size)],
                                data=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                operand0=expert_affinity[token_tile_idx][0:TILE_SIZE, 0],
                                op0=nl.multiply,
                            )
                else:
                    if is_tensor_update_accumulating:
                        if down_scale is not None:
                            idx = (N_PSUM_BANKS * h_tile_idx) + h_subtile_idx
                            if idx < H_NUM_PSUM_TILES:
                                nisa.tensor_tensor(
                                    dst=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data1=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data2=down_scale[0:TILE_SIZE, idx, 0:PSUM_SIZE],
                                    op=nl.multiply,
                                )
                        if actual_psum_size > 0:
                            nisa.tensor_tensor(
                                dst=block_new_lst[token_tile_idx][0:TILE_SIZE, nl.ds(psum_start, actual_psum_size)],
                                data1=block_old[token_tile_idx][0:TILE_SIZE, nl.ds(psum_start, actual_psum_size)],
                                data2=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                op=nl.add,
                            )
                    else:
                        if down_scale is not None:
                            idx = (N_PSUM_BANKS * h_tile_idx) + h_subtile_idx
                            if idx < H_NUM_PSUM_TILES:
                                nisa.tensor_tensor(
                                    dst=block_new_lst[token_tile_idx][
                                        0:TILE_SIZE,
                                        nl.ds(TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * h_subtile_idx, PSUM_SIZE),
                                    ],
                                    data1=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, 0:PSUM_SIZE],
                                    data2=down_scale[0:TILE_SIZE, idx, 0:PSUM_SIZE],
                                    op=nl.multiply,
                                )
                        else:
                            if actual_psum_size > 0:
                                nisa.tensor_copy(
                                    dst=block_new_lst[token_tile_idx][0:TILE_SIZE, nl.ds(psum_start, actual_psum_size)],
                                    src=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                )

                # checkpoint activations
                if down_activations is not None:
                    if actual_psum_size > 0:
                        output_block_start = token_tile_idx * TILE_SIZE
                        output_hidden_start = (N_PSUM_BANKS * h_tile_idx + h_subtile_idx) * PSUM_SIZE

                        nisa.dma_copy(
                            dst=down_activations[
                                block_idx,
                                nl.ds(output_block_start, TILE_SIZE),
                                nl.ds(output_hidden_start, actual_psum_size),
                            ],
                            src=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                        )

    return block_new_lst


def reduce_outputs(
    output: nl.NkiTensor,
    num_tiles: int,
    reduce_tile_size: int,
    offset: int,
    dim_hidden: int,
    sbm: Optional[SbufManager] = None,
):
    """Synchronize across axis=0 in output by performing FMA reduce and store.

    Args:
        output (nl.NkiTensor): Output tensor, size [T, 2, H]
        num_tiles (int): Number of tiles (iterations)
        reduce_tile_size (int): Size of tile size on partition dimension
        offset (int): Output read/write offset on row
        dim_hidden (int): Hidden dimension
        sbm (Optional[SbufManager]): Optional SBUF manager for allocation.
    """
    if sbm is not None:
        sbm.open_scope(name="reduce_outputs")
    if sbm is not None:
        input_reduce_tile = sbm.alloc_stack(
            (reduce_tile_size, 1, dim_hidden),
            dtype=output.dtype,
            name="reduce_tile",
        )
    else:
        input_reduce_tile = nl.ndarray(
            (reduce_tile_size, 1, dim_hidden),
            dtype=output.dtype,
            buffer=nl.sbuf,
        )
    for tile_idx in range(num_tiles):
        tile_start = (tile_idx + offset) * reduce_tile_size
        nisa.dma_compute(
            srcs=[
                output[nl.ds(tile_start, reduce_tile_size), 0:1, nl.ds(0, dim_hidden)],
                output[nl.ds(tile_start, reduce_tile_size), 1:2, nl.ds(0, dim_hidden)],
            ],
            dst=input_reduce_tile,
            scales=[1.0, 1.0],
            reduce_op=nl.add,
        )
        nisa.dma_copy(dst=output[nl.ds(tile_start, reduce_tile_size), 0:1, nl.ds(0, dim_hidden)], src=input_reduce_tile)
    if sbm is not None:
        sbm.close_scope()


def load_and_transpose_gup_bias(
    inps: InputTensors, dims: DimensionSizes, cfg: Configs, block_expert, skip_dma, sbm: Optional[SbufManager] = None
):
    """
    Load and transpose gate/up projection bias for current expert.

    Loads bias from HBM with shape [2, I_TP] and transposes to [TILE_SIZE, 2*gup_tile_count]
    for efficient broadcasting during projection computation.

    Args:
        inps (InputTensors): Input tensor container
        dims (DimensionSizes): Dimension configuration
        cfg (Configs): Kernel configuration
        block_expert (nl.NkiTensor): Expert index for current block [1, 1]
        skip_dma: DMA skip configuration
        sbm (Optional[SbufManager]): Optional SBUF manager for allocation.

    Returns:
        nl.NkiTensor: Transposed bias tensor [TILE_SIZE, 2*gup_tile_count] in SBUF
    """
    if sbm is not None:
        gate_up_bias = sbm.alloc_stack((2, dims.I_TP), dtype=cfg.compute_dtype, name="gup_bias_load")
        gate_up_bias_T = sbm.alloc_stack((TILE_SIZE, 2 * dims.gup_tile_count), dtype=nl.float32, name="gup_bias_T")
    else:
        gate_up_bias = nl.ndarray((2, dims.I_TP), dtype=cfg.compute_dtype)
        gate_up_bias_T = nl.ndarray((TILE_SIZE, 2 * dims.gup_tile_count), dtype=nl.float32)

    nisa.dma_copy(
        dst=gate_up_bias[nl.ds(0, 2), nl.ds(0, dims.I_TP)],
        src=inps.gate_and_up_proj_bias.ap(
            pattern=[[dims.I_TP, 2], [1, dims.I_TP]], offset=0, scalar_offset=block_expert, indirect_dim=0
        ),
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=nisa.dge_mode.hwdge,
    )

    # transpose
    tmp_psum = nl.ndarray((TILE_SIZE, 2 * dims.gup_tile_count), dtype=gate_up_bias.dtype, buffer=nl.psum)

    for i_tile_idx in range(dims.gup_tile_count):
        actual_f_size = min(TILE_SIZE, dims.I_TP - i_tile_idx * TILE_SIZE)
        nisa.nc_transpose(
            dst=tmp_psum[nl.ds(0, actual_f_size), nl.ds(i_tile_idx * 2, 2)],
            data=gate_up_bias[0:2, nl.ds(i_tile_idx * TILE_SIZE, actual_f_size)],
        )

    nisa.tensor_copy(
        dst=gate_up_bias_T[0:TILE_SIZE, nl.ds(0, 2 * dims.gup_tile_count)],
        src=tmp_psum[0:TILE_SIZE, nl.ds(0, 2 * dims.gup_tile_count)],
    )

    return gate_up_bias_T


def shard_strat2blk_idx(
    shard_strat: BlockShardStrategy,
    outer_block_iter: int,
    inner_block_iter: int,
) -> int:
    """
    Convert shard strategy indices to global block index.

    Args:
        shard_strat (BlockShardStrategy): Sharding strategy (HI_LO or PING_PONG)
        outer_block_iter (int): Outer block iteration index
        inner_block_iter (int): Inner block iteration index (0 to BLOCK_PARALLEL_FACTOR-1)

    Returns:
        int: Global block index
    """
    if shard_strat == BlockShardStrategy.HI_LO:
        return outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
    elif shard_strat == BlockShardStrategy.PING_PONG:
        return 2 * (outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter)


def shard_strat2new_blk_idx_offset(
    shard_id: int,
    shard_strat: BlockShardStrategy,
    n_blocks_per_shard: int,
) -> int:
    """
    Calculate block index offset based on shard ID and strategy.

    Args:
        shard_id (int): Current shard identifier (0 or 1)
        shard_strat (BlockShardStrategy): Sharding strategy
        n_blocks_per_shard (int): Number of blocks per shard

    Returns:
        int: Block index offset for the current shard
    """
    if shard_strat == BlockShardStrategy.HI_LO:
        return shard_id * n_blocks_per_shard
    elif shard_strat == BlockShardStrategy.PING_PONG:
        return shard_id


def load_and_broadcast_down_bias(
    inps: InputTensors,
    dims: DimensionSizes,
    cfg: Configs,
    block_expert,
    skip_dma,
    sbm: Optional[SbufManager] = None,
    use_pe_broadcast: bool = False,
):
    """
    Load and broadcast down projection bias for the current block.

    Loads bias from HBM and broadcasts it from [1, H] to [128, H] for element-wise operations.

    Args:
        inps (InputTensors): Input tensor container
        dims (DimensionSizes): Dimension configuration
        cfg (Configs): Kernel configuration
        block_expert (nl.NkiTensor): Expert index for current block
        skip_dma: DMA skip configuration
        sbm (Optional[SbufManager]): Optional SBUF manager for allocation.
        use_pe_broadcast (bool): Use PE matmul broadcast instead of DVE StreamShuffle.

    Returns:
        nl.NkiTensor: Broadcasted bias tensor with shape [128, H]
    """
    if sbm is not None:
        down_bias = sbm.alloc_stack((1, dims.H), dtype=cfg.compute_dtype, name="down_bias_load")
    else:
        down_bias = nl.ndarray((1, dims.H), dtype=cfg.compute_dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=down_bias[0:1, nl.ds(0, dims.H)],
        src=inps.down_proj_bias.ap(
            pattern=[[dims.H, 1], [1, dims.H]], offset=0, scalar_offset=block_expert, indirect_dim=0
        ),
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=nisa.dge_mode.hwdge,
    )

    if sbm is not None:
        down_bias_broadcasted = sbm.alloc_stack((TILE_SIZE, dims.H), dtype=cfg.compute_dtype, name="down_bias_bc")
    else:
        down_bias_broadcasted = nl.ndarray((TILE_SIZE, dims.H), dtype=cfg.compute_dtype, buffer=nl.sbuf)

    if use_pe_broadcast:
        ones_sb = nl.ndarray((1, TILE_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_sb, value=1.0)
        for bias_tile_idx in nl.affine_range(div_ceil(dims.H, PSUM_SIZE)):
            bias_h_offset = bias_tile_idx * PSUM_SIZE
            bias_h_size = min(PSUM_SIZE, dims.H - bias_h_offset)
            bias_psum = nl.ndarray((TILE_SIZE, bias_h_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=bias_psum,
                stationary=ones_sb[:, :TILE_SIZE],
                moving=down_bias[:, nl.ds(bias_h_offset, bias_h_size)],
                is_stationary_onezero=True,
            )
            nisa.tensor_copy(
                dst=down_bias_broadcasted[:, nl.ds(bias_h_offset, bias_h_size)],
                src=bias_psum,
            )
    else:
        stream_shuffle_broadcast(down_bias, down_bias_broadcasted)

    return down_bias_broadcasted


def load_down_bias_raw(inps, dims, cfg, block_expert, skip_dma, sbm=None, bias_h_size=None):
    """Load raw (1, bias_h_size) down bias without broadcasting. bias_h_size defaults to H."""
    _h = bias_h_size if bias_h_size is not None else dims.H
    down_bias = (
        sbm.alloc_stack((1, _h), dtype=cfg.compute_dtype, name="down_bias_raw")
        if sbm is not None
        else nl.ndarray((1, _h), dtype=cfg.compute_dtype, buffer=nl.sbuf)
    )
    nisa.dma_copy(
        dst=down_bias[0:1, nl.ds(0, _h)],
        src=inps.down_proj_bias.ap(pattern=[[_h, 1], [1, _h]], offset=0, scalar_offset=block_expert, indirect_dim=0),
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=nisa.dge_mode.hwdge,
    )
    return down_bias


def bwmm_output_initialization(
    output,
    shard_id=None,
    sbm: Optional[SbufManager] = None,
    expert_affinities_masked=None,
    E=0,
    H=0,
    skip_zero_init=False,
):
    """Zero initialize buffer at `output` and optionally copy expert affinities.

    When expert_affinities_masked is provided, copies affinities into output[:, shard_id, H:H+E]
    during the same tile loop, fusing the zero init and affinity copy.

    Args:
        output: External memory, shape (T, H) or (T, 2, H+E).
        shard_id: Optionally provide shard ID.
        sbm (Optional[SbufManager]): Optional SBUF manager for allocation.
        expert_affinities_masked: Expert affinities [(T+1)*E, 1], or None.
        E (int): Number of experts.
        H (int): Hidden dimension (excluding affinity columns).
        skip_zero_init (bool): Skip zero initialization of output[:, shard_id, :H].
            Used with non_overlapping_shards where zero-init is unnecessary.
    """
    if shard_id == None:
        T, H_full = output.shape
    else:
        T, _, H_full = output.shape
    if sbm is not None:
        sbm.open_scope(name="output_init")
    if sbm is not None:
        zeros = sbm.alloc_stack((TILE_SIZE, H), dtype=output.dtype, name="init_zeros")
    else:
        zeros = nl.ndarray((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf)
    nisa.memset(zeros, value=0.0)

    if expert_affinities_masked is not None:
        _T_plus_1 = expert_affinities_masked.shape[0] // E
        _aff_2d = expert_affinities_masked.reshape((_T_plus_1, E))

    for tile_idx in range(div_ceil(T, TILE_SIZE)):
        num_elements = min(TILE_SIZE, T - tile_idx * TILE_SIZE)
        if shard_id is not None:
            if num_elements != 1:
                if not skip_zero_init:
                    nisa.dma_copy(
                        src=zeros[0:num_elements, 0:H],
                        dst=output[nl.ds(tile_idx * TILE_SIZE, num_elements), shard_id, 0:H],
                    )
                # Copy expert affinities into columns H:H+E
                if _aff_2d is not None and FUSE_AFFINITY_INTO_OUTPUT:
                    nisa.dma_copy(
                        dst=output[nl.ds(tile_idx * TILE_SIZE, num_elements), shard_id, nl.ds(H, E)],
                        src=_aff_2d[nl.ds(tile_idx * TILE_SIZE, num_elements), 0:E],
                    )
        else:
            nisa.dma_copy(
                src=zeros[0:num_elements, 0:H_full],
                dst=output[nl.ds(tile_idx * TILE_SIZE, num_elements), 0:H_full],
            )
    if sbm is not None:
        sbm.close_scope()


def bwmm_load_old_block(
    output,
    token_indices,
    NUM_TILES,
    dtype,
    skip_dma: SkipMode = SkipMode(),
    shard_id=None,
    token_indices_offset=0,
    sbm: Optional[SbufManager] = None,
):
    """Loads the partially computed output hidden states for the current block's token indices."""
    H = output.shape[-1]

    block_old_lst = []
    for alloc_idx in range(NUM_TILES):
        if sbm is not None:
            block_old_lst.append(sbm.alloc_stack((TILE_SIZE, H), dtype=dtype, name=f"block_old_t{alloc_idx}"))
        else:
            block_old_lst.append(nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf))

    # Pre-zero a template buffer on DVE once, then copy to each tile via scalar engine
    if skip_dma.skip_token:
        nisa.memset(value=0, dst=block_old_lst[0])
        for token_tile_idx in range(1, NUM_TILES):
            nisa.tensor_copy(dst=block_old_lst[token_tile_idx], src=block_old_lst[0], engine=nisa.scalar_engine)

    for token_tile_idx in range(NUM_TILES):
        block_token_mapping = token_indices.slice(
            1, token_indices_offset + token_tile_idx, token_indices_offset + token_tile_idx + 1
        )

        if shard_id != None and len(output.shape) > 2:
            nisa.dma_copy(
                dst=block_old_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                src=output.slice(1, shard_id, shard_id + 1).squeeze_dim(1).vector_select(0, block_token_mapping),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
        else:
            nisa.dma_copy(
                dst=block_old_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                src=output.vector_select(0, block_token_mapping),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

    return block_old_lst
