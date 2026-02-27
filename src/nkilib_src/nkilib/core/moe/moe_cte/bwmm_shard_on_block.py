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

from ...utils import common_types
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_program_sharding_info
from .moe_cte_utils import (
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

BLOCK_PARALLEL_FACTOR = 3
GUP_LOAD_COALESCE_FACTOR = 2
GUP_PROJ_DIM = 2


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


@nki.jit(mode='trace')
def bwmm_shard_on_block(
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    block_size: int,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    gate_and_up_proj_bias: Optional[nl.ndarray] = None,
    down_proj_bias: Optional[nl.ndarray] = None,
    gate_up_proj_scale: Optional[nl.ndarray] = None,
    down_proj_scale: Optional[nl.ndarray] = None,
    down_activations: Optional[nl.ndarray] = None,
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
        hidden_states (nl.ndarray): [T, H], Input token embeddings in HBM
        expert_affinities_masked (nl.ndarray): [(T+1)*E, 1], Expert routing weights for token assignments in HBM
        gate_up_proj_weight (nl.ndarray): [E, H, 2, I_TP], Combined gate and up projection weights in HBM
        down_proj_weight (nl.ndarray): [E, I_TP, H], Down projection weights in HBM
        block_size (int): Number of tokens processed per block
        token_position_to_id (nl.ndarray): [N*B], Mapping from block positions to token IDs in HBM
        block_to_expert (nl.ndarray): [N, 1], Expert assignment for each block in HBM
        gate_and_up_proj_bias (nl.ndarray, optional): [E, 2, I_TP], Bias terms for gate/up projections in HBM
        down_proj_bias (nl.ndarray, optional): [E, 1, H], Bias terms for down projection in HBM
        gate_up_proj_scale (nl.ndarray, optional): [E, 1, 2*I_TP], Dequantization scales for gate/up weights in HBM
        down_proj_scale (nl.ndarray, optional): [E, 1, H], Dequantization scales for down weights in HBM
        down_activations (nl.ndarray, optional): [N, B, H], Storage for intermediate activations in HBM
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

    Returns:
        output (nl.ndarray): Expert-processed token representations in HBM. Shape depends on accumulation mode:
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
        block_sharding_strategy == BlockShardStrategy.PING_PONG, "Currently only support PING-PONG sharding strategy"
    )
    kernel_assert(
        block_sharding_strategy == BlockShardStrategy.PING_PONG, "Currently only support PING-PONG sharding strategy"
    )

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

    NUM_STATIC_BLOCKS = N
    if is_tensor_update_accumulating:
        output = nl.ndarray((dims.T, 2, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
        bwmm_output_initialization(output, shard_id=shard_id)
    else:
        output = nl.ndarray((dims.T, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Placeholder for FP8
    gup_scale = None
    down_scale = None
    n_blocks_per_shard = div_ceil(NUM_STATIC_BLOCKS, num_shards)
    # Actual number of valid blocks for this shard under interleaved distribution
    n_blocks_this_shard = div_ceil(max(NUM_STATIC_BLOCKS - shard_id, 0), num_shards)
    n_shard_tile_count = div_ceil(n_blocks_per_shard, BLOCK_PARALLEL_FACTOR)
    all_block_expert_broadcasted_per_shard = nl.ndarray((1, n_blocks_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=all_block_expert_broadcasted_per_shard, value=E)
    nisa.dma_copy(
        dst=all_block_expert_broadcasted_per_shard[0:1, 0:n_blocks_this_shard],
        src=block_to_expert.reshape((block_to_expert.shape[0], 1)).ap(
            pattern=[[1, 1], [2, n_blocks_this_shard]], offset=shard_id
        ),
    )
    all_block_expert_real = nl.ndarray((1, n_blocks_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=all_block_expert_real, src=all_block_expert_broadcasted_per_shard)

    if skip_dma.skip_weight:
        # Convert multi-dimensional ndarray with nl.par_dim to list of ndarrays
        gup_weights_load_dst_lst = []
        for h_tile_idx in range(dims.h_tile_count):
            inner_lst = []
            for h_subtile_idx in range(dims.h_subtile_count_gup):
                tmp = nl.ndarray(
                    (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP), dtype=weights_dtype, buffer=nl.sbuf
                )
                nisa.memset(dst=tmp, value=0)
                inner_lst.append(tmp)
            gup_weights_load_dst_lst.append(inner_lst)

        down_weights_load_dst_lst = []
        for n_i in range(dims.gup_tile_count):
            down_weights_load_dst_lst.append(nl.ndarray((TILE_SIZE, H), dtype=weights_dtype, buffer=nl.sbuf))

        is_weight_same_as_prev_hbm = compute_same_weights_block_parallel_hbm(
            N, block_to_expert=block_to_expert, num_shards=num_shards, shard_id=shard_id, shard_strat=shard_strat
        )

        on_false = nl.ndarray((1, n_blocks_per_shard), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=on_false, value=E)
        need_skip = nl.ndarray((1, n_blocks_per_shard), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=need_skip,
            src=is_weight_same_as_prev_hbm.reshape((1, n_blocks_per_shard)).ap(
                pattern=[[1, 1], [1, n_blocks_per_shard]]
            ),
        )
        nisa.tensor_copy_predicated(
            dst=all_block_expert_broadcasted_per_shard,
            src=on_false,
            predicate=need_skip,
        )
    else:
        gup_weights_load_dst_lst = None
        down_weights_load_dst_lst = None

    block_hidden_states_lst = []
    for _ in range(BLOCK_PARALLEL_FACTOR):
        inner_lst = []
        for _ in range(NUM_TILES):
            tmp = nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.memset(dst=tmp, value=0)
            inner_lst.append(tmp)
        block_hidden_states_lst.append(inner_lst)

    token_indices_lst = []
    for _ in range(BLOCK_PARALLEL_FACTOR):
        token_indices_lst.append(nl.ndarray((TILE_SIZE, NUM_TILES), dtype=nl.int32, buffer=nl.sbuf))

    # STATIC LOOP
    for outer_block_iter in range(n_shard_tile_count):
        block_psum_tiles = div_ceil(B, PSUM_SIZE)
        free_size = min(PSUM_SIZE, B)
        block_hidden_states_T_lst = []
        for k_ in range(BLOCK_PARALLEL_FACTOR):
            outer_lst = []
            for h_tile_idx in range(dims.h_tile_count):
                inner_lst = []
                for h_subtile_idx in range(dims.h_subtile_count):
                    tmp = nl.ndarray((TILE_SIZE, block_psum_tiles, free_size), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.memset(value=0, dst=tmp)
                    inner_lst.append(tmp)

                outer_lst.append(inner_lst)
            block_hidden_states_T_lst.append(outer_lst)
        # parallel load and transpose input
        for inner_block_iter in range(BLOCK_PARALLEL_FACTOR):
            linear_idx = outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
            block_idx = 2 * linear_idx + shard_id

            if block_idx < N:
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
                    v_expert = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    block_expert = load_block_expert(block_to_expert, local_block_idx)
                    shuffle_mask = [0] * DVE_CHANNELS_PER_BANK
                    for channel_idx in range(4):
                        nisa.nc_stream_shuffle(
                            dst=v_expert[
                                DVE_CHANNELS_PER_BANK * channel_idx : DVE_CHANNELS_PER_BANK * (channel_idx + 1), 0:1
                            ],
                            src=block_expert.ap(pattern=[[1, 1], [1, 1]], offset=0),
                            shuffle_mask=shuffle_mask,
                        )

                    expert_affinity_f32_lst = []
                    for _ in range(NUM_TILES):
                        expert_affinity_f32_lst.append(nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf))

                    for token_tile_idx in range(NUM_TILES):
                        addr = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_scalar(
                            dst=addr,
                            data=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx],
                            op0=nl.multiply,
                            operand0=E,
                        )
                        addr_fin = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=addr_fin, data1=addr, data2=v_expert, op=nl.add)

                        if skip_dma.skip_token:
                            nisa.tensor_scalar(dst=addr_fin, data=addr_fin, op0=nl.minimum, operand0=-1)

                        expert_affinity_dtype = nl.ndarray((TILE_SIZE, 1), dtype=compute_dtype, buffer=nl.sbuf)
                        if skip_dma.skip_token:
                            nisa.memset(value=0.0, dst=expert_affinity_dtype)

                        # nl.load with indirect indexing -> nisa.dma_copy with .ap()
                        num_cols = expert_affinities_masked.shape[1]
                        addr_fin_reshaped = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=addr_fin_reshaped, src=addr_fin[0:TILE_SIZE, 0:1])

                        nisa.dma_copy(
                            dst=expert_affinity_dtype[0:TILE_SIZE, 0:1],
                            src=expert_affinities_masked.ap(
                                pattern=[[num_cols, TILE_SIZE], [1, 1]],
                                offset=0,
                                vector_offset=addr_fin_reshaped,
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )

                        # Cast to float32
                        nisa.tensor_copy(
                            dst=expert_affinity_f32_lst[token_tile_idx][0:TILE_SIZE, 0:1],
                            src=expert_affinity_dtype[0:TILE_SIZE, 0:1],
                        )

                for token_tile_idx in range(NUM_TILES):
                    block_token_mapping = token_indices_lst[inner_block_iter].ap(
                        pattern=[[NUM_TILES, TILE_SIZE], [1, 1]],
                        offset=token_tile_idx,
                    )
                    nisa.dma_copy(
                        dst=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                        src=hidden_states.ap(
                            pattern=[[H, TILE_SIZE], [1, H]],
                            offset=0,
                            vector_offset=block_token_mapping,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                    )

                    if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE:
                        nisa.tensor_scalar(
                            dst=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                            data=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                            op0=nl.multiply,
                            operand0=expert_affinity_f32_lst[token_tile_idx][0:TILE_SIZE, 0],
                            engine=nisa.vector_engine,
                        )

                block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)

                for token_tile_idx in range(block_psum_tiles):
                    # ═══════════════════════════════════════════════════════════════════════
                    # DEFINE 8 PSUM BANK BUFFERS: 2 sets × (N_PSUM_BANKS // 2) h_subtiles
                    # ═══════════════════════════════════════════════════════════════════════
                    num_subtiles_per_set = N_PSUM_BANKS // 2
                    tmp_psum_set_0 = []
                    tmp_psum_set_1 = []
                    for _ in range(num_subtiles_per_set):
                        tmp_psum_set_0.append(nl.ndarray((TILE_SIZE, free_size), dtype=compute_dtype, buffer=nl.psum))
                        tmp_psum_set_1.append(nl.ndarray((TILE_SIZE, free_size), dtype=compute_dtype, buffer=nl.psum))

                    # ═══════════════════════════════════════════════════════════════════════
                    # PROCESS H_TILES IN PAIRS (fill 8 banks, then copy 8 banks)
                    # ═══════════════════════════════════════════════════════════════════════
                    num_pairs = (dims.h_tile_count + 1) // 2

                    for h_tile_pair in range(num_pairs):
                        h_tile_0 = h_tile_pair * 2  # Even h_tile → Set 0
                        h_tile_1 = h_tile_pair * 2 + 1  # Odd h_tile → Set 1

                        # ───────────────────────────────────────────────────────────────────
                        # PHASE 1: FILL ALL 8 PSUM BANKS (all transposes first)
                        # ───────────────────────────────────────────────────────────────────

                        # Fill Set 0 with h_tile_0
                        if h_tile_0 < dims.h_tile_count:
                            for batch_tile_idx in range(block_free_tiles):
                                input_tile_idx = block_free_tiles * token_tile_idx + batch_tile_idx
                                for h_subtile_idx in range(num_subtiles_per_set):
                                    nisa.nc_transpose(
                                        dst=tmp_psum_set_0[h_subtile_idx][
                                            0:TILE_SIZE, batch_tile_idx * TILE_SIZE : (batch_tile_idx + 1) * TILE_SIZE
                                        ],
                                        data=block_hidden_states_lst[inner_block_iter][input_tile_idx][
                                            0:TILE_SIZE,
                                            nl.ds(TILE_SIZE * h_subtile_idx + PSUM_SIZE * h_tile_0, TILE_SIZE),
                                        ],
                                    )

                        # Fill Set 1 with h_tile_1
                        if h_tile_1 < dims.h_tile_count:
                            for batch_tile_idx in range(block_free_tiles):
                                input_tile_idx = block_free_tiles * token_tile_idx + batch_tile_idx
                                for h_subtile_idx in range(num_subtiles_per_set):
                                    nisa.nc_transpose(
                                        dst=tmp_psum_set_1[h_subtile_idx][
                                            0:TILE_SIZE, batch_tile_idx * TILE_SIZE : (batch_tile_idx + 1) * TILE_SIZE
                                        ],
                                        data=block_hidden_states_lst[inner_block_iter][input_tile_idx][
                                            0:TILE_SIZE,
                                            nl.ds(TILE_SIZE * h_subtile_idx + PSUM_SIZE * h_tile_1, TILE_SIZE),
                                        ],
                                    )

                        # ───────────────────────────────────────────────────────────────────
                        # PHASE 2: COPY ALL 8 PSUM BANKS (close results copied together!)
                        # ───────────────────────────────────────────────────────────────────

                        # Copy Set 0 (h_tile_0, all h_subtiles together)
                        if h_tile_0 < dims.h_tile_count:
                            for h_subtile_idx in range(num_subtiles_per_set):
                                nisa.tensor_copy(
                                    dst=block_hidden_states_T_lst[inner_block_iter][h_tile_0][h_subtile_idx][
                                        0:TILE_SIZE, token_tile_idx, nl.ds(0, free_size)
                                    ],
                                    src=tmp_psum_set_0[h_subtile_idx],
                                    engine=nisa.scalar_engine,
                                )

                        # Copy Set 1 (h_tile_1, all h_subtiles together)
                        if h_tile_1 < dims.h_tile_count:
                            for h_subtile_idx in range(num_subtiles_per_set):
                                nisa.tensor_copy(
                                    dst=block_hidden_states_T_lst[inner_block_iter][h_tile_1][h_subtile_idx][
                                        0:TILE_SIZE, token_tile_idx, nl.ds(0, free_size)
                                    ],
                                    src=tmp_psum_set_1[h_subtile_idx],
                                    engine=nisa.scalar_engine,
                                )

        # sequential load weights and compute
        for inner_block_iter in range(BLOCK_PARALLEL_FACTOR):
            linear_idx = outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
            block_idx = 2 * linear_idx + shard_id
            if block_idx < N:
                shared_block_idx = shard_strat2blk_idx(shard_strat, outer_block_iter, inner_block_iter)
                local_block_idx = shared_block_idx + shard_strat2new_blk_idx_offset(
                    shard_id, shard_strat, n_blocks_per_shard
                )
                block_expert = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=block_expert, src=all_block_expert_broadcasted_per_shard[0:1, linear_idx : linear_idx + 1]
                )
                real_expert = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=real_expert, src=all_block_expert_real[0:1, linear_idx : linear_idx + 1])

                gup_weights = load_gate_up_proj_weights(
                    gate_up_proj_weight, block_expert, weights_dtype, skip_dma, load_dst=gup_weights_load_dst_lst
                )

                dp_weights = load_down_proj_weight(
                    down_proj_weight, block_expert, weights_dtype, skip_dma, load_dst=down_weights_load_dst_lst
                )

                # load bias
                if cfg.linear_bias:
                    gate_up_bias_T = load_and_transpose_gup_bias(inps, dims, cfg, real_expert, skip_dma)

                if block_idx != shard_id:
                    block_old = (
                        bwmm_load_old_block(
                            output,
                            token_indices_lst[inner_block_iter],
                            NUM_TILES,
                            compute_dtype,
                            skip_dma,
                            shard_id=shard_id,
                        )
                        if is_tensor_update_accumulating
                        else None
                    )
                else:
                    block_old = []
                    for alloc_idx in range(NUM_TILES):
                        tmp = nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.memset(tmp, value=0)
                        block_old.append(tmp)

                free_size = block_hidden_states_T_lst[0][0][0].shape[-1]
                gate_and_up_proj_states_lst_psum = []
                gate_and_up_proj_states_lst_sbuf = []
                for _ in range(GUP_PROJ_DIM):
                    n_psum_lst = []
                    n_sbuf_lst = []
                    for _ in range(dims.n_psum_tile_count):
                        gup_psum_lst = []
                        gup_sbuf_lst = []
                        for _ in range(dims.gup_tile_count):
                            gup_psum_lst.append(nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum))
                            gup_sbuf_lst.append(nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.sbuf))
                        n_psum_lst.append(gup_psum_lst)
                        n_sbuf_lst.append(gup_sbuf_lst)
                    gate_and_up_proj_states_lst_psum.append(n_psum_lst)
                    gate_and_up_proj_states_lst_sbuf.append(n_sbuf_lst)

                for h_tile_idx in range(dims.h_tile_count):
                    for h_subtile_idx in range(dims.h_subtile_count_gup):
                        for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                            is_last_accumulation = (
                                h_tile_idx == dims.h_tile_count - 1
                                and h_subtile_idx == dims.h_subtile_count_gup - 1
                                and op_idx == 1
                            )

                            if not is_last_accumulation:
                                # ───────────────────────────────────────────────────────────────
                                # NORMAL PATH: Just matmuls, no copies yet
                                # ───────────────────────────────────────────────────────────────
                                for i_tile_idx in range(dims.gup_tile_count):
                                    num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)

                                    for projection_idx in range(GUP_PROJ_DIM):
                                        for batch_tile_idx in range(dims.n_psum_tile_count):
                                            nisa.nc_matmul(
                                                dst=gate_and_up_proj_states_lst_psum[projection_idx][batch_tile_idx][
                                                    i_tile_idx
                                                ][0:num_valid_k, nl.ds(0, free_size)],
                                                stationary=gup_weights[h_tile_idx][h_subtile_idx][
                                                    nl.ds(0, TILE_SIZE),
                                                    op_idx,
                                                    projection_idx,
                                                    nl.ds(TILE_SIZE * i_tile_idx, num_valid_k),
                                                ],
                                                moving=block_hidden_states_T_lst[inner_block_iter][h_tile_idx][
                                                    h_subtile_idx * GUP_LOAD_COALESCE_FACTOR + op_idx
                                                ][0:TILE_SIZE, batch_tile_idx, nl.ds(0, free_size)],
                                            )
                            else:
                                # ───────────────────────────────────────────────────────────────
                                # LAST ACCUMULATION: Pipeline matmul with copy of previous tile
                                # ───────────────────────────────────────────────────────────────
                                for i_tile_idx in range(dims.gup_tile_count + 1):
                                    # COPY: Previous i_tile (accumulation complete after last matmul)
                                    if i_tile_idx > 0:
                                        prev_tile = i_tile_idx - 1
                                        num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * prev_tile)
                                        for projection_idx in range(2):
                                            for batch_tile_idx in range(dims.n_psum_tile_count):
                                                nisa.tensor_copy(
                                                    dst=gate_and_up_proj_states_lst_sbuf[projection_idx][
                                                        batch_tile_idx
                                                    ][prev_tile][0:num_valid_k, nl.ds(0, free_size)],
                                                    src=gate_and_up_proj_states_lst_psum[projection_idx][
                                                        batch_tile_idx
                                                    ][prev_tile][0:num_valid_k, nl.ds(0, free_size)],
                                                    engine=nisa.scalar_engine,
                                                )
                                                # add bias
                                                if cfg.linear_bias:
                                                    nisa.tensor_tensor(
                                                        dst=gate_and_up_proj_states_lst_sbuf[projection_idx][
                                                            batch_tile_idx
                                                        ][prev_tile][0:TILE_SIZE, nl.ds(0, free_size)],
                                                        data1=gate_and_up_proj_states_lst_sbuf[projection_idx][
                                                            batch_tile_idx
                                                        ][prev_tile][0:TILE_SIZE, nl.ds(0, free_size)],
                                                        data2=gate_up_bias_T.ap(
                                                            pattern=[
                                                                [2 * dims.gup_tile_count, TILE_SIZE],
                                                                [0, free_size],
                                                            ],
                                                            offset=prev_tile * 2 + projection_idx,
                                                        ),
                                                        op=nl.add,
                                                    )

                                    # MATMUL: Current i_tile
                                    if i_tile_idx < dims.gup_tile_count:
                                        num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)

                                        for projection_idx in range(2):
                                            for batch_tile_idx in range(dims.n_psum_tile_count):
                                                nisa.nc_matmul(
                                                    dst=gate_and_up_proj_states_lst_psum[projection_idx][
                                                        batch_tile_idx
                                                    ][i_tile_idx][0:num_valid_k, nl.ds(0, free_size)],
                                                    stationary=gup_weights[h_tile_idx][h_subtile_idx][
                                                        nl.ds(0, TILE_SIZE),
                                                        op_idx,
                                                        projection_idx,
                                                        nl.ds(TILE_SIZE * i_tile_idx, num_valid_k),
                                                    ],
                                                    moving=block_hidden_states_T_lst[inner_block_iter][h_tile_idx][
                                                        h_subtile_idx * 2 + op_idx
                                                    ][0:TILE_SIZE, batch_tile_idx, nl.ds(0, free_size)],
                                                )

                # Clipping the projections
                if (
                    cfg.gate_clamp_upper_limit is not None
                    or cfg.gate_clamp_lower_limit is not None
                    or cfg.up_clamp_lower_limit is not None
                    or cfg.up_clamp_upper_limit is not None
                ):
                    for token_tile_idx in range(dims.n_psum_tile_count):
                        for i_tile_idx in range(dims.gup_tile_count):
                            # Clipping GATE projection
                            # have both lower and upper limit for gate projection
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

                            # Clipping UP projection
                            # have both lower and upper limit for up projection
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

                expert_affinity_T_broadcasted = (
                    nl.ndarray((TILE_SIZE, free_size), dtype=compute_dtype, buffer=nl.sbuf)
                    if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE_DELAYED
                    else None
                )

                # for prescale, load affinities into [B/128, 128, 1] tensor, transpose each [128, 1] tile to [1, 128] B/128 times
                # and then broadcast this [1, B] result into [128, B]. Finally, do a tensor-tensor mul after each gate/up mm
                if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
                    v_expert = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    block_expert = load_block_expert(block_to_expert, local_block_idx)
                    shuffle_mask = [0] * DVE_CHANNELS_PER_BANK
                    for channel_idx in range(4):
                        nisa.nc_stream_shuffle(
                            dst=v_expert[
                                DVE_CHANNELS_PER_BANK * channel_idx : DVE_CHANNELS_PER_BANK * (channel_idx + 1), 0:1
                            ],
                            src=block_expert.ap(pattern=[[1, 1], [1, 1]], offset=0),
                            shuffle_mask=shuffle_mask,
                        )

                    expert_affinity_T = nl.ndarray((1, free_size), dtype=compute_dtype, buffer=nl.psum)

                    expert_affinity_lst = []
                    for _ in range(NUM_TILES):
                        expert_affinity_lst.append(nl.ndarray((TILE_SIZE, 1), dtype=compute_dtype, buffer=nl.sbuf))

                    for token_tile_idx in range(NUM_TILES):
                        addr = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_scalar(
                            dst=addr,
                            data=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx],
                            op0=nl.multiply,
                            operand0=E,
                        )

                        addr_fin = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=addr_fin, data1=addr, data2=v_expert, op=nl.add)

                        if skip_dma.skip_token:
                            nisa.tensor_scalar(dst=addr_fin, data=addr_fin, op0=nl.maximum, operand0=-1)

                        if skip_dma.skip_token:
                            nisa.memset(value=0.0, dst=expert_affinity_lst[token_tile_idx])

                        # nl.load with indirect indexing -> nisa.dma_copy with .ap()
                        num_cols = expert_affinities_masked.shape[1]
                        addr_fin_reshaped = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=addr_fin_reshaped, src=addr_fin[0:TILE_SIZE, 0:1])

                        nisa.dma_copy(
                            dst=expert_affinity_lst[token_tile_idx][0:TILE_SIZE, 0:1],
                            src=expert_affinities_masked.ap(
                                pattern=[[num_cols, TILE_SIZE], [1, 1]],
                                offset=0,
                                vector_offset=addr_fin_reshaped,
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )

                        # Calculate actual num_f for this tile
                        num_f = min(TILE_SIZE, free_size - token_tile_idx * TILE_SIZE)
                        nisa.nc_transpose(
                            dst=expert_affinity_T[0:1, token_tile_idx * TILE_SIZE : token_tile_idx * TILE_SIZE + num_f],
                            data=expert_affinity_lst[token_tile_idx][0:num_f, 0:1],
                            name=f"waqar_{token_tile_idx}",
                        )

                    # broadcast
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

                intermediate_states = compute_intermediate_states(
                    gate_and_up_proj_states_lst_sbuf,
                    B,
                    I_TP,
                    compute_dtype,
                    activation_function=activation_function,
                    expert_affinity_T_broadcasted=expert_affinity_T_broadcasted,
                    gup_scale=gup_scale,
                )

                expert_affinity_f32 = (
                    calculate_expert_affinities(
                        expert_affinities_masked,
                        token_indices_lst[inner_block_iter],
                        real_expert,
                        E,
                        NUM_TILES,
                        compute_dtype,
                        skip_dma,
                    )
                    if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE
                    else None
                )
                down_activations = None

                if cfg.linear_bias:
                    down_bias_broadcasted = load_and_broadcast_down_bias(inps, dims, cfg, real_expert, skip_dma)
                else:
                    down_bias_broadcasted = None
                block_new_lst = compute_block_output(
                    intermediate_states,
                    dp_weights,
                    expert_affinity_f32,
                    block_old,
                    down_activations,
                    local_block_idx,
                    H,
                    I_TP,
                    NUM_TILES,
                    output_dtype=output.dtype,
                    down_bias_broadcasted=down_bias_broadcasted,
                    is_tensor_update_accumulating=is_tensor_update_accumulating,
                    down_scale=down_scale,
                )

                # Placeholder for TopK > 1 case, need to add full implementation
                if is_tensor_update_accumulating:
                    for token_tile_idx in range(NUM_TILES):
                        token_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_copy(
                            dst=token_idx,
                            src=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx : token_tile_idx + 1],
                            engine=nisa.scalar_engine,
                        )

                        nisa.dma_copy(
                            dst=output.ap(
                                pattern=[[2 * H, TILE_SIZE], [1, H]],
                                offset=shard_id * H,
                                vector_offset=token_idx,
                                indirect_dim=0,
                            ),
                            src=block_new_lst[token_tile_idx].ap(pattern=[[H, TILE_SIZE], [1, H]], offset=0),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )
                else:
                    for token_tile_idx in range(NUM_TILES):
                        token_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_copy(
                            dst=token_idx,
                            src=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx : token_tile_idx + 1],
                        )

                        nisa.dma_copy(
                            dst=output.ap(
                                pattern=[[H, TILE_SIZE], [1, H]], offset=0, vector_offset=token_idx, indirect_dim=0
                            ),
                            src=block_new_lst[token_tile_idx].ap(pattern=[[H, TILE_SIZE], [1, H]], offset=0),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )
    # END OF STATIC LOOP

    # final accumulation
    if is_tensor_update_accumulating and num_shards > 1:
        kernel_assert(num_shards == 2, "only support reducing data from 2 shards")
        reduce_tile_size = 128
        if skip_dma.skip_token:
            reduce_tiles = div_ceil(T, 128)
        else:
            reduce_tiles = div_ceil(T - 1, 128)

        nc0_tiles = reduce_tiles // num_shards
        nc1_tiles = reduce_tiles - nc0_tiles
        zeros_dummy = nl.ndarray((reduce_tile_size, 1, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zeros_dummy, value=0.0)
        if num_shards == 1:
            core_barrier(output, (0))
        elif num_shards == 2:
            core_barrier(output, (0, 1))

        if shard_id == 0:
            reduce_outputs(output, zeros_dummy, nc0_tiles, reduce_tile_size, 0, H)

        if shard_id == 1:
            reduce_outputs(output, zeros_dummy, nc1_tiles, reduce_tile_size, nc0_tiles, H)

    return output


def compute_same_weights_block_parallel_hbm(
    N: int,
    block_to_expert: nl.ndarray,
    num_shards: int,
    shard_id: int,
    shard_strat: BlockShardStrategy,
) -> nl.ndarray:
    """
    Compute weight reuse mask for block-parallel execution.

    Determines which blocks can reuse previously loaded weights by comparing
    expert indices between consecutive blocks in the sharding pattern.

    Args:
        N (int): Total number of blocks
        block_to_expert (nl.ndarray): Expert assignment for each block
        num_shards (int): Number of shards for parallel execution
        shard_id (int): Current shard identifier
        shard_strat (BlockShardStrategy): Block distribution strategy

    Returns:
        nl.ndarray: Boolean mask indicating weight reuse opportunities
    """
    kernel_assert(shard_strat == BlockShardStrategy.PING_PONG, "only support PING_PONG for right now")
    n_blocks_per_shard = div_ceil(N, num_shards)
    n_tile_minus_1 = div_ceil(n_blocks_per_shard - 1, TILE_SIZE)
    tile_size = min(TILE_SIZE, n_blocks_per_shard - 1)

    off_one_offset = 2

    # max_linear_index is the number of valid comparisons:
    #   compare block j with block j+1, where j = 0 .. max_linear_index-1
    #   Input bound: shard_id + 2*(j+1) < N  =>  j < (N - shard_id - 2) / 2
    #   Output bound: j+1 < n_blocks_per_shard  =>  j < n_blocks_per_shard - 1
    max_linear_index = (N - shard_id - off_one_offset) // 2
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
    all_expert_indices = nl.ndarray(
        block_to_expert.ap(
            pattern=[[2 * n_tile_minus_1, tile_size], [2, n_tile_minus_1]],
            offset=shard_id,
        ).shape,
        dtype=nl.int32,
        buffer=nl.sbuf,
    )
    all_expert_indices_off_one = nl.ndarray((tile_size, n_tile_minus_1), dtype=nl.int32, buffer=nl.sbuf)
    is_weight_same_as_prev = nl.ndarray((tile_size, n_tile_minus_1), dtype=nl.uint8, buffer=nl.sbuf)

    # Separate 1-row tensors for partial rows so partition dim always starts at 0
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
                    [2 * n_tile_minus_1, num_acc_p_1],
                    [2, num_acc_f_1],
                ],
                offset=shard_id,
            ),
        )

    # Pattern 2 - Partial row into separate 1-row tensor:
    #   Loads block_to_expert[shard_id + 2*n_tile_minus_1*num_full_rows + 2*f]
    #   for f in [0, num_acc_f_2)
    if num_acc_p_2 > 0:
        partial_row_offset = shard_id + 2 * n_tile_minus_1 * num_full_rows
        nisa.dma_copy(
            dst=partial_expert_indices[0:1, 0:num_acc_f_2],
            src=block_to_expert.ap(
                pattern=[
                    [2 * n_tile_minus_1, 1],
                    [2, num_acc_f_2],
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
                    [2 * n_tile_minus_1, num_acc_p_1],
                    [2, num_acc_f_1],
                ],
                offset=shard_id + off_one_offset,
            ),
        )

    # Pattern 2 - Partial row into separate 1-row tensor:
    #   Loads block_to_expert[shard_id + 2 + 2*n_tile_minus_1*num_full_rows + 2*f]
    #   for f in [0, num_acc_f_2)
    if num_acc_p_2 > 0:
        partial_row_offset_off_one = shard_id + off_one_offset + 2 * n_tile_minus_1 * num_full_rows
        nisa.dma_copy(
            dst=partial_expert_indices_off_one[0:1, 0:num_acc_f_2],
            src=block_to_expert.ap(
                pattern=[
                    [2 * n_tile_minus_1, 1],
                    [2, num_acc_f_2],
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

    return is_weight_same_as_prev_hbm


def load_down_proj_weight(
    down_proj_weight: nl.ndarray,
    block_expert: nl.ndarray,
    compute_dtype,
    skip_dma: SkipMode = SkipMode(),
    load_dst: Optional[list] = None,
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
            load_dst.append(nl.ndarray((TILE_SIZE, H), dtype=down_proj_weight.dtype, buffer=nl.sbuf))

    if down_proj_weight.dtype != compute_dtype:
        dp_weights = []
        for alloc_idx in range(gup_n_tile):
            dp_weights.append(nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf))

    for i_tile_idx in range(gup_n_tile):
        i_start = TILE_SIZE * i_tile_idx
        num_i = min(TILE_SIZE, I_TP - i_start)

        # Initialize with zeros for masked-out elements
        if num_i < TILE_SIZE:
            if not skip_dma.skip_weight:
                nisa.memset(dst=load_dst[i_tile_idx], value=0)

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
        )

        # Type conversion if needed
        if down_proj_weight.dtype != compute_dtype:
            # Initialize with zeros for masked-out elements
            if num_i < TILE_SIZE:
                nisa.memset(dst=dp_weights[i_tile_idx], value=0)

            nisa.tensor_copy(dst=dp_weights[i_tile_idx][0:num_i, 0:H], src=load_dst[i_tile_idx][0:num_i, 0:H])

    return load_dst if down_proj_weight.dtype == compute_dtype else dp_weights


def load_gate_up_proj_weights(
    gate_up_proj_weight: nl.ndarray,
    block_expert: nl.ndarray,
    compute_dtype,
    skip_dma: SkipMode = SkipMode(),
    load_dst: Optional[list] = None,
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
                h_j_lst.append(
                    nl.ndarray(
                        (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP), dtype=compute_dtype, buffer=nl.sbuf
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
    down_scale=None,
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
        down_activations (nl.ndarray, optional): Storage for intermediate activations
        block_idx (int): Current block index
        H (int): Hidden dimension size
        I_TP (int): Intermediate dimension size
        NUM_TILES (int): Number of tiles per block
        output_dtype (nki.dtype): Output data type
        compute_dtype (nki.dtype): Computation data type
        is_tensor_update_accumulating (bool): Enable accumulation mode
        down_bias_broadcasted (nl.ndarray, optional): Broadcasted bias [TILE_SIZE, H]
        allocate (bool): Unused parameter
        down_scale (nl.ndarray, optional): Dequantization scales

    Returns:
        list: Block output tensors [NUM_TILES][TILE_SIZE, H]

    Notes:
        - Supports FP8 dequantization with down_scale
        - Accumulation mode for TopK > 1 scenarios
        - Optional bias addition before affinity scaling
    """
    block_new_lst = []
    for _ in range(NUM_TILES):
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
    for token_tile_idx in range(NUM_TILES):
        for h_tile_idx in range(h_i_upper):
            buffer_type = nl.psum
            down_proj_psum_lst = []
            for _ in range(N_PSUM_BANKS):
                tmp = nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=nl.float32, buffer=buffer_type)
                down_proj_psum_lst.append(tmp)
            for h_subtile_idx in range(N_PSUM_BANKS):
                for i_tile_idx in range(gup_n_tile):
                    # Mask 1: H dimension - compute actual psum size
                    psum_start = TOTAL_PSUM_SIZE * h_tile_idx + PSUM_SIZE * h_subtile_idx
                    actual_psum_size = min(PSUM_SIZE, H - psum_start)

                    # Mask 2: K dimension (rows) - compute valid rows
                    # Condition: -row - TILE_SIZE*i_tile_idx + I_TP - 1 >= 0  =>  row < I_TP - TILE_SIZE*i_tile_idx
                    num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
                    if actual_psum_size > 0:
                        nisa.nc_matmul(
                            stationary=intermediate_states[i_tile_idx][
                                0:num_valid_k, nl.ds(TILE_SIZE * token_tile_idx, TILE_SIZE)
                            ],
                            moving=dp_weights[i_tile_idx][nl.ds(0, num_valid_k), nl.ds(psum_start, actual_psum_size)],
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
                        # down bias
                        if actual_psum_size > 0:
                            if down_bias_broadcasted is not None:
                                bias_idx = (N_PSUM_BANKS * h_tile_idx) + h_subtile_idx
                                nisa.tensor_tensor(
                                    dst=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                    data1=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                    data2=down_bias_broadcasted[
                                        0:TILE_SIZE, nl.ds(bias_idx * PSUM_SIZE, actual_psum_size)
                                    ],
                                    op=nl.add,
                                )

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
                        # down bias
                        if actual_psum_size > 0:
                            if down_bias_broadcasted is not None:
                                bias_idx = (N_PSUM_BANKS * h_tile_idx) + h_subtile_idx
                                nisa.tensor_tensor(
                                    dst=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                    data1=down_proj_psum_lst[h_subtile_idx][0:TILE_SIZE, nl.ds(0, actual_psum_size)],
                                    data2=down_bias_broadcasted[
                                        0:TILE_SIZE, nl.ds(bias_idx * PSUM_SIZE, actual_psum_size)
                                    ],
                                    op=nl.add,
                                )
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
    output: nl.ndarray, zeros: nl.ndarray, num_tiles: int, reduce_tile_size: int, offset: int, dim_hidden: int
):
    """Synchronize across axis=0 in output by performing FMA reduce and store.

    Args:
        output (nl.ndarray): Output tensor, size [2, T, H]
        zeros (nl.ndarray): Zero tensor, size [reduce_tile_size, H]
        num_tiles (int): Number of tiles (iterations)
        reduce_tile_size (int): Size of tile size on partition dimension
        offset (int): Output read/write offset on row
        dim_hidden (int): Hidden dimension
    """
    for tile_idx in range(num_tiles):
        tile_start = (tile_idx + offset) * reduce_tile_size
        input_reduce_tile = nl.ndarray(
            (reduce_tile_size, 1, dim_hidden),
            dtype=output.dtype,
            buffer=nl.sbuf,
        )
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
        nisa.dma_copy(
            dst=output[nl.ds(tile_start, reduce_tile_size), 1:2, nl.ds(0, dim_hidden)],
            src=zeros,
        )


def load_and_transpose_gup_bias(inps: InputTensors, dims: DimensionSizes, cfg: Configs, block_expert, skip_dma):
    """
    Load and transpose gate/up projection bias for current expert.

    Loads bias from HBM with shape [2, I_TP] and transposes to [TILE_SIZE, 2*gup_tile_count]
    for efficient broadcasting during projection computation.

    Args:
        inps (InputTensors): Input tensor container
        dims (DimensionSizes): Dimension configuration
        cfg (Configs): Kernel configuration
        block_expert (nl.ndarray): Expert index for current block [1, 1]

    Returns:
        nl.ndarray: Transposed bias tensor [TILE_SIZE, 2*gup_tile_count] in SBUF
    """
    gate_up_bias = nl.ndarray((2, dims.I_TP), dtype=cfg.compute_dtype)
    gate_up_bias_T = nl.ndarray((TILE_SIZE, 2 * dims.gup_tile_count), dtype=cfg.compute_dtype)

    nisa.dma_copy(
        dst=gate_up_bias[nl.ds(0, 2), nl.ds(0, dims.I_TP)],
        src=inps.gate_and_up_proj_bias.ap(
            pattern=[[dims.I_TP, 2], [1, dims.I_TP]], offset=0, scalar_offset=block_expert, indirect_dim=0
        ),
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
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


def load_and_broadcast_down_bias(inps: InputTensors, dims: DimensionSizes, cfg: Configs, block_expert, skip_dma):
    """
    Load and broadcast down projection bias for the current block.

    Loads bias from HBM and broadcasts it from [1, H] to [128, H] for element-wise operations.

    Args:
        inps (InputTensors): Input tensor container
        dims (DimensionSizes): Dimension configuration
        cfg (Configs): Kernel configuration
        block_expert (nl.ndarray): Expert index for current block

    Returns:
        nl.ndarray: Broadcasted bias tensor with shape [128, H]
    """
    down_bias = nl.ndarray((1, dims.H), dtype=cfg.compute_dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=down_bias[0:1, nl.ds(0, dims.H)],
        src=inps.down_proj_bias.ap(
            pattern=[[dims.H, 1], [1, dims.H]], offset=0, scalar_offset=block_expert, indirect_dim=0
        ),
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
    )

    down_bias_broadcasted = nl.ndarray((TILE_SIZE, dims.H), dtype=cfg.compute_dtype, buffer=nl.sbuf)
    stream_shuffle_broadcast(down_bias, down_bias_broadcasted)
    return down_bias_broadcasted


def bwmm_output_initialization(output, shard_id=None):
    """Zero initialize buffer at `output`. Required for accumulation (top K > 1)

    Args:
        output (_type_): External memory
        shard_id (_type_, optional): Optionally provide shard ID. Defaults to None.
    """
    if shard_id == None:
        T, H = output.shape
    else:
        T, _, H = output.shape
    for tile_idx in range(div_ceil(T, TILE_SIZE)):
        zeros = nl.ndarray((TILE_SIZE, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros, value=0.0)

        if shard_id != None:
            num_elements = min(TILE_SIZE, T - tile_idx * TILE_SIZE)
            if num_elements != 1:
                nisa.dma_copy(
                    src=zeros[0:num_elements, 0:H], dst=output[nl.ds(tile_idx * TILE_SIZE, num_elements), shard_id, 0:H]
                )
        else:
            num_elements = min(TILE_SIZE, T - tile_idx * TILE_SIZE)
            nisa.dma_copy(src=zeros[0:num_elements, 0:H], dst=output[nl.ds(tile_idx * TILE_SIZE, num_elements), 0:H])


def bwmm_load_old_block(
    output, token_indices, NUM_TILES, dtype, skip_dma: SkipMode = SkipMode(), shard_id=None, token_indices_offset=0
):
    """Loads the partially computed output hidden states for the current block's token indices."""
    H = output.shape[-1]

    block_old_lst = []
    for alloc_idx in range(NUM_TILES):
        block_old_lst.append(nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf))

    for token_tile_idx in range(NUM_TILES):
        if skip_dma.skip_token:
            nisa.memset(value=0, dst=block_old_lst[token_tile_idx][0:TILE_SIZE, 0:H])

        block_token_mapping = nl.ndarray((TILE_SIZE, 1), dtype=token_indices.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=block_token_mapping,
            src=token_indices[
                0:TILE_SIZE, token_indices_offset + token_tile_idx : token_indices_offset + token_tile_idx + 1
            ],
        )

        if shard_id != None:
            # output shape: (num_shards, num_tokens, H)
            num_tokens = output.shape[0]

            # Pattern: [[H, TILE_SIZE], [1, H]]
            # - First dimension (indirect): TILE_SIZE iterations with stride H (row stride)
            # - Second dimension: H iterations with stride 1 (within row)
            # Offset: shard_id * num_tokens * H (to access the correct shard)
            # vector_offset: block_token_mapping (shape TILE_SIZE, 1)
            # indirect_dim: 0 (we're indirecting on the first dimension of the pattern)

            nisa.dma_copy(
                dst=block_old_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                src=output.ap(
                    pattern=[[2 * H, TILE_SIZE], [1, H]],
                    offset=shard_id * H,
                    vector_offset=block_token_mapping,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
        else:
            # output shape: (num_tokens, H)
            # Pattern: [[H, TILE_SIZE], [1, H]]
            # Offset: 0
            # vector_offset: block_token_mapping
            # indirect_dim: 0

            nisa.dma_copy(
                dst=block_old_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                src=output.ap(
                    pattern=[[H, TILE_SIZE], [1, H]], offset=0, vector_offset=block_token_mapping, indirect_dim=0
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

    return block_old_lst
