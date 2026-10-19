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

"""Backward pass kernel for blockwise matrix multiplication in dropless Mixture of Experts."""

import nki  # noqa: F401 - Required by NKI coding guidelines
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ....core.utils.allocator import SbufManager, align_to, sizeinbytes
from ....core.utils.kernel_helpers import (
    PSUM_BANK_SIZE,
    apply_activation,
    apply_activation_dx,
    div_ceil,
    get_program_sharding_info,
)
from ....core.utils.logging import LogLevel, get_logger
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ....core.utils.tiled_range import TiledRange
from .moe_bwd_parameters import AffinityOption, ClampLimits, MOEBwdParameters, ShardOption

# Total SBUF available minus reserved regions (DMA header, metadata, alignment overhead)
MAX_AVAILABLE_SBUF_SIZE = 224 * 1024 - 16384 - 8 - 520


def _load_block_expert(block_to_expert, block_idx, sbm, dst=None):
    """
    Load expert index for the current block from HBM into SBUF.

    Args:
        block_to_expert (nl.ndarray): [N, 1], Expert indices for each block on HBM.
        block_idx (int): Current block index.
        sbm (SbufManager): SBUF memory manager.
        dst (nl.ndarray, optional): Pre-allocated [1, 1] buffer. If None, allocates from sbm.

    Returns:
        nl.ndarray: [1, 1], Expert index tensor in SBUF for indirect addressing.
    """
    if dst is None:
        expert_idx_tensor = sbm.alloc_stack(
            (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"expert_idx_{block_idx}", align=32
        )
    else:
        expert_idx_tensor = dst
    nisa.dma_copy(expert_idx_tensor.ap([[1, 1], [1, 1]]), block_to_expert.ap([[1, 1], [1, 1]], offset=block_idx))
    return expert_idx_tensor


def _indirect_vector_dma_transpose(
    dst,
    src_reshaped,
    weight_idx_vec,
    valid_lane_count,
    h_128_start,
    num_h_128_tiles,
    dst_offset,
    dst_num_h_tiles,
    dst_h_block_size,
    vec_offset_stride=1,
    vec_offset_offset=0,
):
    """Load and transpose a weight tile using indirect vector dma_transpose.

    Replaces the pattern: dma_copy(HWDGE) + nc_transpose + tensor_copy with a single
    dma_transpose using vector_offset for expert-level indirect addressing.

    The source tensor must be pre-reshaped to [E*lane_dim, tile_count, TILE_SIZE] where
    lane_dim is the dimension indexed by expert_idx (e.g., I_TP for down_proj_weight).

    The caller must pre-compute weight_idx_vec = uint32(expert_idx * lane_dim + i_tp_offset + iota).
    Precomputing all index vectors before the loop avoids serial tensor_scalar → dma_transpose
    dependency chains that stall the DMA transpose engine.

    For multi-column index tensors (e.g., [TILE_SIZE, NUM_TILES] token index buffers),
    use vec_offset_stride and vec_offset_offset to select the correct column.

    Args:
        dst: Destination SBUF tensor [TILE_SIZE, num_h_tiles, block_size], must be 32-byte aligned.
        src_reshaped: Source HBM tensor reshaped to [E*lane_dim, tile_count, TILE_SIZE].
        weight_idx_vec: [TILE_SIZE, N] int32/uint32, pre-computed index vector(s).
        valid_lane_count: Number of valid lanes (for remainder tiles, OOB indices are skipped).
        h_128_start: Starting 128-element tile index in the H dimension.
        num_h_128_tiles: Number of 128-element H tiles to load.
        dst_offset: Offset into the free dimension of dst for placement.
        dst_num_h_tiles: Total number of H tiles in dst's second dimension (for stride calc).
        dst_h_block_size: Total H block size in dst's third dimension (for stride calc).
        vec_offset_stride (int): Partition-dim stride in weight_idx_vec AP pattern. Default 1
            for [TILE_SIZE, 1] vectors; use NUM_B_TILES for [TILE_SIZE, NUM_B_TILES] buffers.
        vec_offset_offset (int): Flat offset into weight_idx_vec AP pattern. Default 0;
            use b_tile_idx to select a column from multi-column buffers.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    _, src_tile_count, src_tile_size = src_reshaped.shape

    nisa.dma_transpose(
        dst=dst.ap(
            pattern=[
                [dst_num_h_tiles * dst_h_block_size, TILE_SIZE],
                [dst_h_block_size, num_h_128_tiles],
                [1, valid_lane_count],
            ],
            offset=dst_offset,
        ),
        src=src_reshaped.ap(
            pattern=[
                [src_tile_count * src_tile_size, valid_lane_count],
                [src_tile_size, num_h_128_tiles],
                [1, src_tile_size],
            ],
            offset=h_128_start * src_tile_size,
            vector_offset=weight_idx_vec.ap(
                [[vec_offset_stride, valid_lane_count], [1, 1]],
                offset=vec_offset_offset,
            ),
            indirect_dim=0,
        ),
        axes=(2, 1, 0),
        oob_mode=oob_mode.skip,
    )


def _initialize_gradient_outputs_shard(
    hidden_states_grad,
    gate_up_proj_weight_grad,
    down_proj_weight_grad,
    num_shards,
    shard_id,
    down_proj_bias_grad,
    gate_and_up_proj_bias_grad,
    expert_affinities_masked_grad,
    sbm,
):
    """
    Initialize all gradient output tensors to zero with LNC sharding.

    Zeros out the gradient tensors for hidden states, gate/up projection weights,
    down projection weights, biases, and expert affinities. Each shard initializes
    only its portion of the sharded dimensions.

    Args:
        hidden_states_grad (nl.ndarray): [T, H], Hidden states gradient tensor.
        gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Gate/up weight gradients.
        down_proj_weight_grad (nl.ndarray): [E, I_TP, H], Down projection weight gradients.
        num_shards (int): Number of LNC shards.
        shard_id (int): Current shard ID.
        down_proj_bias_grad (nl.ndarray, optional): [E, H], Down projection bias gradients.
        gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP], Gate/up bias gradients.
        expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Expert affinity gradients.
        sbm (SbufManager): SBUF memory manager.

    Returns:
        None: Gradient tensors are zeroed in-place.

    Notes:
        - H dimension is sharded across cores for hidden_states_grad and down_proj_weight_grad.
        - I dimension is sharded for gate_and_up_proj_bias_grad.
        - Requires H % num_shards == 0.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    T, H = hidden_states_grad.shape
    E, I_TP, _ = down_proj_weight_grad.shape
    T_TILE_SIZE = TILE_SIZE
    I_TILE_SIZE = TILE_SIZE
    NUM_T_TILES = div_ceil(T, T_TILE_SIZE)
    NUM_I_TILES = div_ceil(I_TP, I_TILE_SIZE)

    # shard over H
    H_PER_SHARD = H // num_shards
    H_TILE_SIZE = TILE_SIZE
    NUM_H_TILES = div_ceil(H_PER_SHARD, H_TILE_SIZE)
    GATE_OR_UP_WEIGHT_COUNT = 2

    # For GUP Bias
    I_PER_SHARD = I_TP // num_shards
    I_SHARD_OFFSET = I_PER_SHARD * shard_id

    if E == 1:
        shard_E = 1
        e_offset = 0
    else:
        shard_E = E // num_shards
        e_offset = shard_E * shard_id

    sbm.open_scope(name="Gradient Initialization")
    H_SHARD_OFFSET = H_PER_SHARD * shard_id
    zeros = sbm.alloc_stack(
        (T_TILE_SIZE, 1),
        dtype=expert_affinities_masked_grad.dtype,
        name="expert_affinities_masked_grad_zeros",
        align=32,
    )
    nisa.memset(zeros, value=0.0)
    for expert_idx in range(shard_E):
        for t_tile_idx in range(NUM_T_TILES):
            valid_t = min(T_TILE_SIZE, T - t_tile_idx * T_TILE_SIZE)
            nisa.dma_copy(
                dst=expert_affinities_masked_grad[
                    nl.ds((e_offset + expert_idx) * T + (t_tile_idx * T_TILE_SIZE), valid_t), :
                ],
                src=zeros[0:valid_t, :],
            )

    zeros = sbm.alloc_stack(
        (H_TILE_SIZE, GATE_OR_UP_WEIGHT_COUNT, I_TP),
        dtype=gate_up_proj_weight_grad.dtype,
        name="gate_up_proj_weight_grad_zeros",
        align=32,
    )
    nisa.memset(zeros, value=0.0)
    for e_idx in range(E):
        for h_tile_idx in range(NUM_H_TILES):
            num_p = min(H_TILE_SIZE, H_PER_SHARD - (h_tile_idx * H_TILE_SIZE))

            gate_up_proj_weight_grad_AP = gate_up_proj_weight_grad.ap(
                pattern=[[GATE_OR_UP_WEIGHT_COUNT * I_TP, num_p], [I_TP, GATE_OR_UP_WEIGHT_COUNT], [1, I_TP]],
                offset=(e_idx * H * GATE_OR_UP_WEIGHT_COUNT * I_TP)
                + (H_SHARD_OFFSET + h_tile_idx * H_TILE_SIZE) * GATE_OR_UP_WEIGHT_COUNT * I_TP,
            )
            nisa.dma_copy(dst=gate_up_proj_weight_grad_AP, src=zeros[0:num_p, 0:GATE_OR_UP_WEIGHT_COUNT, 0:I_TP])

    zeros = sbm.alloc_stack((I_TILE_SIZE, H_PER_SHARD), dtype=down_proj_weight_grad.dtype, align=32)
    nisa.memset(zeros, value=0.0)
    for e_idx in range(E):
        for i_tile_idx in range(NUM_I_TILES):
            valid_i = min(I_TILE_SIZE, I_TP - I_TILE_SIZE * i_tile_idx)

            down_proj_weight_grad_AP = down_proj_weight_grad.ap(
                pattern=[[H, valid_i], [1, H_PER_SHARD]],
                offset=(e_idx * I_TP * H) + (i_tile_idx * I_TILE_SIZE * H) + H_SHARD_OFFSET,
            )
            nisa.dma_copy(down_proj_weight_grad_AP, zeros[0:valid_i, 0:H_PER_SHARD])

    # Initialize blockwise bwd outputs (gradients)
    zeros = sbm.alloc_stack(
        (T_TILE_SIZE, H_PER_SHARD), dtype=hidden_states_grad.dtype, name="hidden_states_grad_zeros", align=32
    )
    nisa.memset(zeros, value=0.0)
    for t_tile_idx in range(NUM_T_TILES):
        # Calculate element count for mask replacement
        valid_t = min(T_TILE_SIZE, T - t_tile_idx * T_TILE_SIZE)

        nisa.dma_copy(
            dst=hidden_states_grad[nl.ds(t_tile_idx * T_TILE_SIZE, valid_t), nl.ds(H_SHARD_OFFSET, H_PER_SHARD)],
            src=zeros[0:valid_t, 0:H_PER_SHARD],
        )

    if down_proj_bias_grad:
        zeros = sbm.alloc_stack(
            (H_TILE_SIZE, 1), dtype=down_proj_bias_grad.dtype, name="down_proj_bias_grad_zeros", align=32
        )
        nisa.memset(zeros, value=0.0)
        for e_idx in range(E):
            for h_tile_idx in range(NUM_H_TILES):
                valid_h = min(TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)
                nisa.dma_copy(
                    dst=down_proj_bias_grad.ap(
                        pattern=[[1, valid_h], [1, 1]],
                        offset=(e_idx * H) + H_SHARD_OFFSET + (h_tile_idx * H_TILE_SIZE),
                    ),
                    src=zeros[0:valid_h, 0:1],
                )

    if gate_and_up_proj_bias_grad:
        NUM_I_TILES = div_ceil(I_PER_SHARD, I_TILE_SIZE)
        zeros = sbm.alloc_stack(
            (I_TILE_SIZE, 1),
            dtype=gate_and_up_proj_bias_grad.dtype,
            name="gate_and_up_proj_bias_grad_zeros",
            align=32,
        )
        nisa.memset(zeros, value=0.0)
        for e_idx in range(E):
            for gate_or_up in range(GATE_OR_UP_WEIGHT_COUNT):
                for i_tile_idx in range(NUM_I_TILES):
                    valid_i = min(I_TILE_SIZE, I_PER_SHARD - i_tile_idx * I_TILE_SIZE)
                    nisa.dma_copy(
                        dst=gate_and_up_proj_bias_grad.ap(
                            pattern=[[1, valid_i], [1, 1]],
                            offset=(e_idx * GATE_OR_UP_WEIGHT_COUNT * I_TP)
                            + (gate_or_up * I_TP)
                            + I_SHARD_OFFSET
                            + i_tile_idx * I_TILE_SIZE,
                        ),
                        src=zeros[0:valid_i, 0:1],
                    )

    sbm.close_scope()


def _generate_dynamic_offsets(
    block_token_pos_to_id,
    expert_idx_tensor,
    token_indices_offset,
    addr,
    b_tile_idx,
    skip_dma,
    E,
):
    """
    Generate dynamic offsets for indirect DMA addressing of expert affinities.

    Computes token_indices_offset = block_token_pos_to_id * E + expert_idx for
    indirect addressing into the flattened expert affinity tensor.

    Args:
        block_token_pos_to_id (nl.ndarray): [B_TILE_SIZE, NUM_TILES], Token position to ID mapping.
        expert_idx_tensor (nl.ndarray): [B_TILE_SIZE, 1], Broadcast expert index.
        token_indices_offset (nl.ndarray): [B_TILE_SIZE, 1], Output offset tensor.
        addr (nl.ndarray): [B_TILE_SIZE, 1], Temporary address buffer.
        b_tile_idx (int): Current batch tile index.
        skip_dma (SkipMode): Controls OOB handling.
        E (int): Number of experts.

    Returns:
        nl.ndarray: [B_TILE_SIZE, 1], Computed token indices offset for indirect addressing.
    """
    nisa.tensor_scalar(dst=addr, data=block_token_pos_to_id[:, b_tile_idx], op0=nl.multiply, operand0=E)
    nisa.tensor_tensor(dst=token_indices_offset, data1=addr, op=nl.add, data2=expert_idx_tensor)

    if skip_dma.skip_token:
        nisa.tensor_scalar(dst=token_indices_offset, data=token_indices_offset, op0=nl.maximum, operand0=-1)

    return token_indices_offset


def _load_expert_affinities(
    expert_affinities_masked,
    block_token_pos_to_id_full,
    expert_idx_tensor,
    b_tile_idx,
    skip_dma,
    E,
    sbm,
):
    """Load expert affinities for a B tile using indirect addressing.

    Args:
        expert_affinities_masked (nl.ndarray): [T * E, 1], Expert affinities.
        block_token_pos_to_id_full (nl.ndarray): [B_TILE_SIZE, NUM_TILES], Token position mapping.
        expert_idx_tensor (nl.ndarray): [B_TILE_SIZE, 1], Broadcast expert index.
        b_tile_idx (int): Current batch tile index (global).
        skip_dma (SkipMode): Controls OOB handling.
        E (int): Number of experts.
        sbm (SbufManager): SBUF memory manager.

    Returns:
        tuple: (expert_affinity_tile, ea_token_indices_offset)
            - expert_affinity_tile: [B_TILE_SIZE, 1] in SBUF
            - ea_token_indices_offset: [B_TILE_SIZE, 1] address offsets
    """
    B_TILE_SIZE = block_token_pos_to_id_full.shape[0]
    token_indices_offset = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, align=32)
    addr = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, align=32)
    ea_token_indices_offset = _generate_dynamic_offsets(
        block_token_pos_to_id_full,
        expert_idx_tensor,
        token_indices_offset,
        addr,
        b_tile_idx,
        skip_dma,
        E,
    )

    expert_affinity_tile = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32, align=32)
    if skip_dma.skip_token:
        nisa.memset(expert_affinity_tile, value=0.0)
    nisa.dma_copy(
        dst=expert_affinity_tile,
        src=expert_affinities_masked.ap(
            pattern=[[expert_affinities_masked.shape[1], B_TILE_SIZE], [1, 1]],
            offset=0,
            vector_offset=ea_token_indices_offset,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
    )
    return expert_affinity_tile, ea_token_indices_offset


def _compute_down_proj_bias_grad(
    down_proj_output_grad_hbm,
    down_proj_bias_grad,
    expert_idx,
    B_DIM,
    H_DIM,
    num_shards,
    shard_id,
    dtype,
    sbm,
    block_token_pos_to_id_full=None,
    skip_dma=None,
    block_idx=None,
    iota_vec=None,
    expert_idx_broadcast=None,
    manage_scope=True,
    expert_affinities_masked=None,
    E=None,
    accumulation_dtype=None,
):
    """
    Compute down projection bias gradient with H-dimension sharding.

    Reduces down_proj_output_grad [B, H] over B dimension to get bias grad [1, H].
    Each shard computes and writes H/num_shards of the gradient.

    Uses dma_compute for atomic read-modify-write accumulation.
    Block 0 overwrites (bias_grad is zero-initialized); subsequent blocks
    use scatter RMW via dma_compute.

    Args:
        down_proj_output_grad_hbm (nl.ndarray): [B, H] or [T, H], Gradient tensor in HBM.
        down_proj_bias_grad (nl.ndarray): [E, H], Output bias gradient in HBM.
        expert_idx (nl.ndarray): [1, 1], Expert index tensor for indirect addressing.
        B_DIM (int): Batch/block size.
        H_DIM (int): Hidden dimension.
        num_shards (int): Number of LNC shards.
        shard_id (int): Current shard ID.
        dtype: Data type.
        sbm (SbufManager): SBUF memory manager.
        block_token_pos_to_id_full (nl.ndarray, optional): Token indices for indirect addressing (Affinity I only).
        skip_dma (SkipMode, optional): OOB handling mode (Affinity I only).
        block_idx (int): Current block index.
        iota_vec (nl.ndarray, optional): Precomputed [0, 1, ..., TILE_SIZE-1].
        expert_idx_broadcast (nl.ndarray, optional): Precomputed (TILE_SIZE, N) broadcast of expert indices.
        expert_affinities_masked (nl.ndarray, optional): [(T+1) * E, 1], Per-token expert affinities. When
            provided (AFFINITY_ON_I), used to affinity-weight the down-bias gradient before reducing over tokens.
        E (int, optional): Number of experts, used to index expert_affinities_masked (AFFINITY_ON_I only).

    Returns:
        None: Bias gradient accumulated in-place into down_proj_bias_grad.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax

    H_PER_SHARD = H_DIM // num_shards
    H_SHARD_OFFSET = H_PER_SHARD * shard_id
    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    NUM_B_TILES = div_ceil(B_DIM, B_TILE_SIZE)
    H_TILE_SIZE = min(TILE_SIZE, H_PER_SHARD)
    NUM_H_TILES = div_ceil(H_PER_SHARD, H_TILE_SIZE)
    is_indirect = block_token_pos_to_id_full != None

    if manage_scope:
        sbm.open_scope(name="down_proj_bias_grad")

    # Allocate buffers
    grad_tiles = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES, B_TILE_SIZE), dtype=dtype, buffer=nl.sbuf, align=32)
    reduced = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES), dtype=nl.float32, buffer=nl.sbuf, align=32)
    # Bias-grad accumulator precision: opt-in fp32 via accumulation_dtype (default = dtype, baseline).
    bias_accum_dtype = accumulation_dtype if accumulation_dtype is not None else dtype
    bias_grad_accum = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES), dtype=bias_accum_dtype, buffer=nl.sbuf, align=32)
    nisa.memset(bias_grad_accum, value=0.0)

    if is_indirect:
        # Cast token indices from int32 to uint32 for indirect dma_transpose
        token_indices_uint32 = sbm.alloc_stack((TILE_SIZE, NUM_B_TILES), dtype=nl.uint32, align=32)
        nisa.tensor_copy(dst=token_indices_uint32, src=block_token_pos_to_id_full)
        NUM_FULL_H_TILES = H_PER_SHARD // TILE_SIZE
        H_REMAINDER = H_PER_SHARD % TILE_SIZE

    for b_tile_idx in range(NUM_B_TILES):
        if is_indirect:
            if skip_dma.skip_token:
                nisa.memset(grad_tiles, value=0)
            vec_ap = token_indices_uint32.ap(
                [[token_indices_uint32.shape[-1], B_TILE_SIZE], [1, 1]],
                offset=b_tile_idx,
            )
            # Full aligned tiles in one dma_transpose
            if NUM_FULL_H_TILES > 0:
                nisa.dma_transpose(
                    dst=grad_tiles.ap(
                        pattern=[
                            [NUM_H_TILES * B_TILE_SIZE, TILE_SIZE],
                            [B_TILE_SIZE, NUM_FULL_H_TILES],
                            [1, B_TILE_SIZE],
                        ],
                        offset=0,
                    ),
                    src=down_proj_output_grad_hbm.ap(
                        pattern=[
                            [H_DIM, B_TILE_SIZE],
                            [TILE_SIZE, NUM_FULL_H_TILES],
                            [1, TILE_SIZE],
                        ],
                        offset=H_SHARD_OFFSET,
                        vector_offset=vec_ap,
                        indirect_dim=0,
                    ),
                    axes=(2, 1, 0),
                    oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                )
            # Remainder tile
            if H_REMAINDER > 0:
                nisa.dma_transpose(
                    dst=grad_tiles.ap(
                        pattern=[[NUM_H_TILES * B_TILE_SIZE, H_REMAINDER], [1, 1], [1, 1], [1, B_TILE_SIZE]],
                        offset=NUM_FULL_H_TILES * B_TILE_SIZE,
                    ),
                    src=down_proj_output_grad_hbm.ap(
                        pattern=[[H_DIM, B_TILE_SIZE], [1, 1], [1, 1], [1, H_REMAINDER]],
                        offset=H_SHARD_OFFSET + NUM_FULL_H_TILES * TILE_SIZE,
                        vector_offset=vec_ap,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                )
        else:
            # Direct: single dma_transpose for all H tiles from contiguous HBM
            for h_tile_idx in range(NUM_H_TILES):
                h_offset = H_SHARD_OFFSET + h_tile_idx * H_TILE_SIZE
                valid_h = min(H_TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)
                nisa.dma_transpose(
                    dst=grad_tiles.ap(
                        pattern=[[NUM_H_TILES * B_TILE_SIZE, valid_h], [1, 1], [1, 1], [1, B_TILE_SIZE]],
                        offset=(h_tile_idx * B_TILE_SIZE),
                    ),
                    src=down_proj_output_grad_hbm.ap(
                        pattern=[[H_DIM, B_TILE_SIZE], [1, 1], [1, 1], [1, valid_h]],
                        offset=(b_tile_idx * B_TILE_SIZE) * H_DIM + h_offset,
                    ),
                )

        if is_indirect and expert_affinities_masked != None:
            """
            Affinity-weight the down-bias gradient on the AFFINITY_ON_I path.

            The forward scales the down-projection output (including the bias) by the per-token
            expert affinity (output += affinity * down_bias), so the down-bias gradient must be
            affinity-weighted before reducing over tokens. The gate/up and AFFINITY_ON_H paths
            already receive an affinity-weighted gradient.
            """
            expert_idx_tensor = expert_idx_broadcast[0:B_TILE_SIZE, block_idx : block_idx + 1]
            aff_addr = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, align=32)
            aff_off = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, align=32)
            _generate_dynamic_offsets(
                block_token_pos_to_id_full, expert_idx_tensor, aff_off, aff_addr, b_tile_idx, skip_dma, E
            )
            aff_col = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=dtype, align=32)
            if skip_dma.skip_token:
                nisa.memset(aff_col, value=0.0)
            nisa.dma_copy(
                dst=aff_col,
                src=expert_affinities_masked.ap(
                    pattern=[[expert_affinities_masked.shape[1], B_TILE_SIZE], [1, 1]],
                    offset=0,
                    vector_offset=aff_off,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
            aff_t_psum = nl.ndarray((1, B_TILE_SIZE), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(data=aff_col[0:B_TILE_SIZE, 0:1], dst=aff_t_psum[0:1, 0:B_TILE_SIZE])
            aff_row = sbm.alloc_stack((1, B_TILE_SIZE), dtype=dtype, align=32)
            nisa.tensor_copy(dst=aff_row, src=aff_t_psum)
            aff_bc = sbm.alloc_stack((H_TILE_SIZE, B_TILE_SIZE), dtype=dtype, align=32)
            stream_shuffle_broadcast(aff_row, aff_bc)
            for h_tile_idx in range(NUM_H_TILES):
                valid_h = min(H_TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)
                nisa.tensor_tensor(
                    dst=grad_tiles[0:valid_h, h_tile_idx, :],
                    data1=grad_tiles[0:valid_h, h_tile_idx, :],
                    data2=aff_bc[0:valid_h, :],
                    op=nl.multiply,
                )

        # Reduce over B dimension and accumulate for each H tile
        for h_tile_idx in range(NUM_H_TILES):
            valid_h = min(H_TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)
            nisa.tensor_reduce(
                dst=reduced[0:valid_h, h_tile_idx], op=nl.add, data=grad_tiles[0:valid_h, h_tile_idx, :], axis=1
            )
            nisa.tensor_tensor(
                dst=bias_grad_accum[0:valid_h, h_tile_idx],
                op=nl.add,
                data1=bias_grad_accum[0:valid_h, h_tile_idx],
                data2=reduced[0:valid_h, h_tile_idx],
            )

    # --- Scatter RMW address vector construction (hoisted outside h_tile loop) ---
    # dma_compute cannot use HWDGE's scalar_offset for indirect expert indexing,
    # so we flatten bias_grad to 1-D and build a per-element address vector that
    # encodes the expert's starting row.
    #
    # For expert e, the flat index of hidden element i within tile at h_offset is:
    #   vec_off[i] + h_offset  =  (e * H_DIM + i) + h_offset
    #
    # Construction:
    #   expert_base     = expert_idx * H_DIM              (scalar)
    #   expert_base_vec = broadcast to [H_TILE_SIZE, 1]   (all lanes = expert_base)
    #   iota_vec        = [0, 1, 2, ..., H_TILE_SIZE-1]   (sequential lane offsets)
    #   vec_off         = expert_base_vec + iota_vec       (per-lane flat address)
    if block_idx > 0:
        flat_size = down_proj_bias_grad.shape[0] * H_DIM
        bias_grad_flat = down_proj_bias_grad.reshape((flat_size, 1))  # [E*H, 1]
        # vec_off[i] = expert_idx * H_DIM + i
        expert_base_vec = sbm.alloc_stack((H_TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf, align=32)
        nisa.tensor_scalar(
            dst=expert_base_vec,
            data=expert_idx_broadcast[0:H_TILE_SIZE, block_idx : block_idx + 1],
            op0=nl.multiply,
            operand0=H_DIM,
        )
        vec_off = sbm.alloc_stack((H_TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf, align=32)
        nisa.tensor_tensor(dst=vec_off, op=nl.add, data1=expert_base_vec, data2=iota_vec[0:H_TILE_SIZE, 0:1])

    for h_tile_idx in range(NUM_H_TILES):
        h_offset = H_SHARD_OFFSET + h_tile_idx * H_TILE_SIZE
        valid_h = min(H_TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)

        if block_idx == 0:
            # First block: overwrite (bias_grad is zero-initialized)
            nisa.dma_copy(
                dst=down_proj_bias_grad.ap(
                    pattern=[[1, valid_h], [1, 1]],
                    offset=h_offset,
                    scalar_offset=expert_idx,
                    indirect_dim=0,
                ),
                src=bias_grad_accum[0:valid_h, h_tile_idx],
                # HWDGE requires src/dst dtypes to match; use SWDGE only when the fp32
                # accumulator must cast to a narrower HBM grad dtype on store.
                dge_mode=dge_mode.hwdge if bias_accum_dtype == down_proj_bias_grad.dtype else dge_mode.swdge,
            )
        else:
            # Subsequent blocks: atomic read-modify-write via dma_compute.
            # Effective flat address per lane i: h_offset + vec_off[i]
            #   = h_offset + expert_idx * H_DIM + i
            # dma_compute reads dst (existing HBM value) and src (new partial),
            # then writes dst = 1.0 * dst + 1.0 * src (i.e. accumulate).
            dst_ap = bias_grad_flat.ap(
                pattern=[[1, valid_h], [1, 1]],
                offset=h_offset,
                vector_offset=vec_off[0:valid_h, 0:1],
                indirect_dim=0,
            )
            src_ap = bias_grad_accum.ap(
                pattern=[[NUM_H_TILES, valid_h], [1, 1]],
                offset=h_tile_idx,
            )
            nisa.dma_compute(dst=dst_ap, srcs=[dst_ap, src_ap], scales=[1.0, 1.0], reduce_op=nl.add)

    if manage_scope:
        sbm.close_scope()


def _compute_gate_up_proj_bias_grad(
    gate_up_proj_output_grad_hbm,
    gate_and_up_proj_bias_grad,
    expert_idx,
    B,
    I_TP,
    shard_id,
    dtype,
    sbm,
    block_idx=None,
    iota_vec=None,
    expert_idx_broadcast=None,
    manage_scope=True,
    accumulation_dtype=None,
):
    """
    Compute gate and up projection bias gradient.

    Reduces gate_up_proj_output_grad [B, 2, I_TP] over B dimension.
    With LNC2, shard_id determines gate (0) or up (1).

    Uses dma_compute for atomic read-modify-write accumulation.
    Block 0 overwrites (bias_grad is zero-initialized); subsequent blocks
    use scatter RMW via dma_compute.

    Args:
        gate_up_proj_output_grad_hbm (nl.ndarray): [B, 2, I_TP], Gradient tensor in HBM.
        gate_and_up_proj_bias_grad (nl.ndarray): [E, 2, I_TP], Output bias gradient in HBM.
        expert_idx (nl.ndarray): [1, 1], Expert index tensor for indirect addressing.
        B (int): Batch/block size.
        I_TP (int): Intermediate dimension.
        shard_id (int): Current shard ID (0=gate, 1=up).
        dtype: Data type.
        sbm (SbufManager): SBUF memory manager.
        block_idx (int): Current block index.

    Returns:
        None: Bias gradient accumulated in-place into gate_and_up_proj_bias_grad.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    GATE_UP_WEIGHT_COUNT = 2

    B_TILE_SIZE = min(TILE_SIZE, B)
    NUM_B_TILES = div_ceil(B, B_TILE_SIZE)
    I_TILE_SIZE = min(TILE_SIZE, I_TP)
    NUM_I_TILES = div_ceil(I_TP, I_TILE_SIZE)

    if manage_scope:
        sbm.open_scope(name="gate_up_proj_bias_grad")

    # Allocate buffers - transposed layout [I, NUM_I_TILES, B]
    grad_tiles = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES, B_TILE_SIZE), dtype=dtype, buffer=nl.sbuf, align=32)
    reduced = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES), dtype=nl.float32, buffer=nl.sbuf, align=32)
    # Bias-grad accumulator precision: opt-in fp32 via accumulation_dtype (default = dtype, baseline).
    bias_accum_dtype = accumulation_dtype if accumulation_dtype is not None else dtype
    bias_grad_accum = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES), dtype=bias_accum_dtype, buffer=nl.sbuf, align=32)
    nisa.memset(bias_grad_accum, value=0.0)

    for b_tile_idx in range(NUM_B_TILES):
        b_offset = b_tile_idx * B_TILE_SIZE
        valid_b = min(B_TILE_SIZE, B - b_offset)

        for i_tile_idx in range(NUM_I_TILES):
            i_offset = i_tile_idx * I_TILE_SIZE
            valid_i = min(I_TILE_SIZE, I_TP - i_offset)

            # Load grad tile from HBM [B, 2, I_TP] with transpose to [I, B]
            nisa.dma_transpose(
                dst=grad_tiles.ap(
                    pattern=[[NUM_I_TILES * B_TILE_SIZE, valid_i], [1, 1], [1, 1], [1, valid_b]],
                    offset=(i_tile_idx * B_TILE_SIZE),
                ),
                src=gate_up_proj_output_grad_hbm.ap(
                    pattern=[[GATE_UP_WEIGHT_COUNT * I_TP, valid_b], [1, 1], [1, 1], [1, valid_i]],
                    offset=(b_offset) * GATE_UP_WEIGHT_COUNT * I_TP + shard_id * I_TP + i_offset,
                ),
            )

            # Reduce over B dimension
            nisa.tensor_reduce(
                dst=reduced[0:valid_i, i_tile_idx], op=nl.add, data=grad_tiles[0:valid_i, i_tile_idx, 0:valid_b], axis=1
            )

            # Accumulate
            nisa.tensor_tensor(
                dst=bias_grad_accum[0:valid_i, i_tile_idx],
                op=nl.add,
                data1=bias_grad_accum[0:valid_i, i_tile_idx],
                data2=reduced[0:valid_i, i_tile_idx],
            )

    # --- Scatter RMW address vector construction (hoisted outside i_tile loop) ---
    # Same approach as _compute_down_proj_bias_grad: flatten [E, 2, I_TP] to 1-D
    # and build per-lane addresses since dma_compute can't use HWDGE scalar_offset.
    #
    # For expert e, the flat index of element i within tile at (shard_id * I_TP + i_offset) is:
    #   vec_off[i] + shard_id * I_TP + i_offset  =  (e * 2 * I_TP + i) + shard_id * I_TP + i_offset
    #
    # Construction:
    #   expert_base     = expert_idx * (2 * I_TP)         (scalar)
    #   expert_base_vec = broadcast to [I_TILE_SIZE, 1]   (all lanes = expert_base)
    #   iota_vec        = [0, 1, 2, ..., I_TILE_SIZE-1]   (sequential lane offsets)
    #   vec_off         = expert_base_vec + iota_vec       (per-lane flat address)
    if block_idx > 0:
        flat_size = gate_and_up_proj_bias_grad.shape[0] * GATE_UP_WEIGHT_COUNT * I_TP
        bias_grad_flat = gate_and_up_proj_bias_grad.reshape((flat_size, 1))  # [E*2*I_TP, 1]
        # vec_off[i] = expert_idx * (2 * I_TP) + i
        expert_base_vec = sbm.alloc_stack((I_TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf, align=32)
        nisa.tensor_scalar(
            dst=expert_base_vec,
            data=expert_idx_broadcast[0:I_TILE_SIZE, block_idx : block_idx + 1],
            op0=nl.multiply,
            operand0=GATE_UP_WEIGHT_COUNT * I_TP,
        )
        vec_off = sbm.alloc_stack((I_TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf, align=32)
        nisa.tensor_tensor(dst=vec_off, op=nl.add, data1=expert_base_vec, data2=iota_vec[0:I_TILE_SIZE, 0:1])

    for i_tile_idx in range(NUM_I_TILES):
        i_offset = i_tile_idx * I_TILE_SIZE
        valid_i = min(I_TILE_SIZE, I_TP - i_offset)

        if block_idx == 0:
            # First block: overwrite (bias_grad is zero-initialized)
            nisa.dma_copy(
                dst=gate_and_up_proj_bias_grad.ap(
                    pattern=[[1, valid_i], [1, 1]],
                    offset=shard_id * I_TP + i_offset,
                    scalar_offset=expert_idx,
                    indirect_dim=0,
                ),
                src=bias_grad_accum[0:valid_i, i_tile_idx],
                # HWDGE requires src/dst dtypes to match; use SWDGE only when the fp32
                # accumulator must cast to a narrower HBM grad dtype on store.
                dge_mode=dge_mode.hwdge if bias_accum_dtype == gate_and_up_proj_bias_grad.dtype else dge_mode.swdge,
            )
        else:
            # Subsequent blocks: atomic read-modify-write via dma_compute.
            # Effective flat address per lane i: shard_id * I_TP + i_offset + vec_off[i]
            #   = shard_id * I_TP + i_offset + expert_idx * 2 * I_TP + i
            dst_ap = bias_grad_flat.ap(
                pattern=[[1, valid_i], [1, 1]],
                offset=shard_id * I_TP + i_offset,
                vector_offset=vec_off[0:valid_i, 0:1],
                indirect_dim=0,
            )
            src_ap = bias_grad_accum.ap(
                pattern=[[NUM_I_TILES, valid_i], [1, 1]],
                offset=i_tile_idx,
            )
            nisa.dma_compute(dst=dst_ap, srcs=[dst_ap, src_ap], scales=[1.0, 1.0], reduce_op=nl.add)

    if manage_scope:
        sbm.close_scope()


def _compute_down_projection_output_grad(
    output_hidden_states_grad,
    block_token_pos_to_id_full,
    expert_affinities_masked,
    expert_affinities_masked_grad,
    down_proj_act_checkpoint,
    block_idx,
    skip_dma,
    expert_idx,
    E,
    dtype,
    num_shards,
    shard_id,
    sbm,
    BLOCK_H=2,
    expert_idx_broadcast=None,
    manage_scope=True,
    BUFFER_DEGREE=4,
):
    """
    Compute down projection output gradient and expert affinity gradient.

    Computes the gradient of the down projection output by multiplying the upstream
    gradient with expert affinities. Also computes the expert affinity gradient by
    reducing the element-wise product of upstream gradient and checkpointed activations.

    Sharding: Each core processes half the B tiles, reads full H, no send/recv needed.

    Args:
        output_hidden_states_grad (nl.ndarray): [T, H], Upstream gradient from output.
        block_token_pos_to_id_full (nl.ndarray): [B_TILE_SIZE, NUM_TILES], Token position mapping.
        expert_affinities_masked (nl.ndarray): [T * E, 1], Expert affinities.
        expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Output expert affinity gradient.
        down_proj_act_checkpoint (nl.ndarray): [N, B, H], Checkpointed down projection activations.
        block_idx (int): Current block index.
        skip_dma (SkipMode): Controls OOB handling for DMA operations.
        expert_idx (nl.ndarray): [1, 1], Expert index for indirect addressing.
        E (int): Number of experts.
        dtype: Computation data type.
        num_shards (int): Number of LNC shards.
        shard_id (int): Current shard ID.
        sbm (SbufManager): SBUF memory manager.
        BLOCK_H (int): Number of H tiles per block (default: 2).

    Returns:
        nl.ndarray: [B, H], Down projection output gradient in shared HBM.

    Notes:
        - down_proj_output_grad = output_hidden_states_grad * expert_affinity
        - expert_affinity_grad = sum(output_hidden_states_grad * down_proj_checkpoint, axis=H)
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    _, B_DIM, H_DIM = down_proj_act_checkpoint.shape

    B_TILE_SIZE, NUM_B_TILES = block_token_pos_to_id_full.shape

    if NUM_B_TILES >= num_shards:
        NUM_B_TILE_SHARD = NUM_B_TILES // num_shards
        NUM_B_TILE_SHARD_OFFSET = NUM_B_TILE_SHARD * shard_id
    else:
        # Not enough B tiles to shard — both shards process all tiles (duplicate work, correct result)
        NUM_B_TILE_SHARD = NUM_B_TILES
        NUM_B_TILE_SHARD_OFFSET = 0
    H_TILE_SIZE = TILE_SIZE
    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM)
    NUM_H_BLOCKS = div_ceil(H_DIM, H_BLOCK_SIZE)

    down_proj_output_grad_hbm = nl.ndarray(
        (B_DIM, H_DIM),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name=f"down_proj_output_grad_hbm_shared_block_{block_idx}",
    )
    # Broadcast expert_idx to tile size
    if manage_scope:
        sbm.open_scope(name="Down Projection Output Grad")

    expert_idx_tensor = expert_idx_broadcast[0:B_TILE_SIZE, block_idx : block_idx + 1]

    expert_affinity_tile = []
    expert_affinityfp32_tile = []
    ea_grad_accum = []
    ea_grad_reduced = []
    token_indices_offset = []
    addr = []

    for n_buffer in range(BUFFER_DEGREE):
        expert_affinity_tile.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=expert_affinities_masked.dtype, align=32))
        expert_affinityfp32_tile.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32, align=32))
        ea_grad_accum.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32, align=32))
        ea_grad_reduced.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=dtype, align=32))
        token_indices_offset.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, align=32))
        addr.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, align=32))

    sbm.open_scope(name="Down Projection Output Grad Buffer", interleave_degree=BUFFER_DEGREE)

    for b_tile_idx in range(NUM_B_TILE_SHARD):
        ea_buffer_idx = b_tile_idx % BUFFER_DEGREE
        global_tile_idx = NUM_B_TILE_SHARD_OFFSET + b_tile_idx
        ea_token_indices_offset = _generate_dynamic_offsets(
            block_token_pos_to_id_full,
            expert_idx_tensor,
            token_indices_offset[ea_buffer_idx],
            addr[ea_buffer_idx],
            global_tile_idx,
            skip_dma,
            E,
        )

        if skip_dma.skip_token:
            nisa.memset(expert_affinity_tile[ea_buffer_idx], value=0.0)
        nisa.memset(ea_grad_accum[ea_buffer_idx], value=0.0)

        global_b_offset = global_tile_idx * B_TILE_SIZE

        # Load expert affinity once per B tile
        nisa.dma_copy(
            dst=expert_affinity_tile[ea_buffer_idx],
            src=expert_affinities_masked.ap(
                pattern=[[expert_affinities_masked.shape[1], B_TILE_SIZE], [1, 1]],
                offset=0,
                vector_offset=ea_token_indices_offset,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
        nisa.tensor_copy(expert_affinityfp32_tile[ea_buffer_idx], expert_affinity_tile[ea_buffer_idx])

        for h_block_idx in range(NUM_H_BLOCKS):
            h_offset = h_block_idx * H_BLOCK_SIZE
            valid_h = min(H_BLOCK_SIZE, H_DIM - h_offset)

            block_hidden_grad_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"block_hidden_grad_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
                align=32,
            )
            block_down_proj_ckpt_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"block_down_proj_ckpt_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
                align=32,
            )
            block_multiply_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"block_multiply_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
                align=32,
            )
            down_proj_output_grad_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"down_proj_output_grad_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
                align=32,
            )
            ea_grad_local = sbm.alloc_stack(
                (B_TILE_SIZE, 1),
                dtype=nl.float32,
                name=f"ea_grad_local_{block_idx}_{global_tile_idx}_{h_block_idx}",
                align=32,
            )

            if skip_dma.skip_token:
                nisa.memset(block_hidden_grad_tile, value=0)

            nisa.dma_copy(
                dst=block_hidden_grad_tile,
                src=output_hidden_states_grad.ap(
                    pattern=[[H_DIM, B_TILE_SIZE], [1, valid_h]],
                    offset=h_offset,
                    vector_offset=block_token_pos_to_id_full.ap(
                        pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]],
                        offset=global_tile_idx,
                    ),
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

            nisa.dma_copy(
                dst=block_down_proj_ckpt_tile,
                src=down_proj_act_checkpoint.ap(
                    pattern=[[H_DIM, B_TILE_SIZE], [1, valid_h]],
                    offset=(block_idx * B_DIM * H_DIM) + (global_b_offset * H_DIM) + h_offset,
                ),
                dge_mode=dge_mode.hwdge,
            )

            nisa.tensor_tensor(
                dst=block_multiply_tile,
                data1=block_hidden_grad_tile,
                data2=block_down_proj_ckpt_tile,
                op=nl.multiply,
            )

            # Reduce across H for EA grad
            nisa.tensor_reduce(dst=ea_grad_local, op=nl.add, data=block_multiply_tile, axis=1, negate=False)
            nisa.tensor_tensor(
                dst=ea_grad_accum[ea_buffer_idx],
                op=nl.add,
                data1=ea_grad_accum[ea_buffer_idx],
                data2=ea_grad_local,
            )

            # Compute down_proj output grad
            nisa.tensor_scalar(
                dst=down_proj_output_grad_tile,
                data=block_hidden_grad_tile,
                op0=nl.multiply,
                operand0=expert_affinityfp32_tile[ea_buffer_idx],
            )

            # Store to HBM
            nisa.dma_copy(
                dst=down_proj_output_grad_hbm[nl.ds(global_b_offset, B_TILE_SIZE), nl.ds(h_offset, valid_h)],
                src=down_proj_output_grad_tile,
                dge_mode=dge_mode.hwdge,
            )
            sbm.increment_section()

        # Store EA grad (no send/recv needed - we computed full H)
        nisa.tensor_copy(
            dst=ea_grad_reduced[ea_buffer_idx], src=ea_grad_accum[ea_buffer_idx], engine=nisa.scalar_engine
        )
        nisa.dma_copy(
            dst=expert_affinities_masked_grad.ap(
                pattern=[[1, B_TILE_SIZE], [1, 1]],
                offset=0,
                vector_offset=ea_token_indices_offset,
                indirect_dim=0,
            ),
            src=ea_grad_reduced[ea_buffer_idx],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )

    sbm.close_scope()
    if manage_scope:
        sbm.close_scope()

    nisa.core_barrier(down_proj_output_grad_hbm, (0, 1))
    return down_proj_output_grad_hbm


def _load_gate_up_projection_checkpoint(
    gate_up_proj_act_checkpoint_T,
    block_gate_or_up_projection,
    NUM_B_TILES,
    I_TP_BLOCK_SIZE,
    B_TILE_SIZE,
    B_DIM,
    block_idx,
    GATE_UP_WEIGHT_COUNT,
    I_TP_DIM,
    gate_or_up,
    i_tp_global_offset,
    b_block_idx,
    B_BLOCK_SIZE,
    valid_i_block,
):
    """
    Load and transpose gate or up projection checkpoint from HBM to SBUF.

    Loads the checkpointed gate or up activation for the current block and transposes
    it from [B, I] layout to [I, B] layout for subsequent matmul operations.

    Args:
        gate_up_proj_act_checkpoint_T (nl.ndarray): [N, 2, I_TP, B], Checkpointed activations.
        block_gate_or_up_projection (nl.ndarray): [B_TILE, NUM_B_TILES, I_TP_BLOCK], Output buffer.
        NUM_B_TILES (int): Number of batch tiles.
        I_TP_BLOCK_SIZE (int): I dimension block size.
        B_TILE_SIZE (int): Batch tile size.
        B_DIM (int): Total batch dimension.
        block_idx (int): Current block index.
        GATE_UP_WEIGHT_COUNT (int): Number of projections (2 for gate and up).
        I_TP_DIM (int): Total intermediate dimension.
        gate_or_up (int): 0 for gate, 1 for up projection.
        i_tp_global_offset (int): Global offset in I dimension.
        b_block_idx (int): Current batch block index.
        B_BLOCK_SIZE (int): Batch block size.
        valid_i_block (int): Valid I dimension size for this block.

    Returns:
        None: Data is loaded into block_gate_or_up_projection in-place.
    """

    for b_tile_idx in range(NUM_B_TILES):
        nisa.dma_transpose(
            dst=block_gate_or_up_projection.ap(
                pattern=[[NUM_B_TILES * I_TP_BLOCK_SIZE, B_TILE_SIZE], [1, 1], [1, 1], [1, valid_i_block]],
                offset=b_tile_idx * I_TP_BLOCK_SIZE,
            ),
            src=gate_up_proj_act_checkpoint_T.ap(
                pattern=[[B_DIM, valid_i_block], [1, 1], [1, 1], [1, B_TILE_SIZE]],
                offset=(block_idx * GATE_UP_WEIGHT_COUNT * I_TP_DIM * B_DIM)
                + (gate_or_up * I_TP_DIM * B_DIM)
                + (i_tp_global_offset) * B_DIM
                + (b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE),
            ),
        )


def _compute_gate_up_projection_output_grad(
    down_proj_output_grad_hbm,
    down_projection_weight,
    gate_up_proj_act_checkpoint_T,
    shard_id,
    num_shards,
    block_idx,
    expert_idx,
    compute_dtype,
    activation_type,
    sbm,
    clamp_limits: ClampLimits = ClampLimits(),
    skip_gate_proj: bool = False,
    grad_accum_dtype=None,
    BLOCK_H=2,
    BLOCK_B=2,
    BLOCK_I_TP=2,
    affinity_option=AffinityOption.AFFINITY_ON_H,
    block_token_pos_to_id_full=None,
    expert_affinities_masked=None,
    expert_affinities_masked_grad=None,
    E=None,
    skip_dma=None,
    expert_idx_broadcast=None,
    shard_option=ShardOption.SHARD_ON_FREE,
    manage_scope=True,
    BUFFER_DEGREE=3,
):
    """
    Compute gate and up projection output gradients with I-dimension sharding.

    Performs backward pass through the activation function and computes gradients
    for gate and up projections. Shards on I dimension to avoid send/recv after matmul.

    For Affinity I, additionally:
    - Scales gate_up_mult by EA before storing (for weight grad)
    - Computes EA gradient by reducing upstream_grad * gate_up_mult over I dimension
    - Scales matmul result by EA before activation backward

    Args:
        down_proj_output_grad_hbm (nl.ndarray): [B, H], Down projection output gradient.
        down_projection_weight (nl.ndarray): [E, I_TP, H], Down projection weights.
        gate_up_proj_act_checkpoint_T (nl.ndarray): [N, 2, I_TP, B], Checkpointed activations.
        shard_id (int): Current shard ID.
        num_shards (int): Number of LNC shards.
        block_idx (int): Current block index.
        expert_idx (nl.ndarray): [1, 1], Expert index for indirect addressing.
        compute_dtype: Computation data type.
        nki_activation_fwd_op: Forward activation function (e.g., nl.silu).
        nki_activation_bwd_op: Backward activation function (e.g., nl.silu_dx).
        sbm (SbufManager): SBUF memory manager.
        clamp_limits (ClampLimits): Gradient clamping limits.
        BLOCK_H (int): H dimension blocking factor.
        BLOCK_B (int): B dimension blocking factor.
        BLOCK_I_TP (int): I dimension blocking factor.
        affinity_option (AffinityOption): Affinity scaling dimension.
        block_token_pos_to_id_full (nl.ndarray): [B_TILE_SIZE, NUM_TILES], Token position mapping (Affinity I).
        expert_affinities_masked (nl.ndarray): [T * E, 1], Expert affinities (Affinity I).
        expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Output EA gradient (Affinity I).
        E (int): Number of experts (Affinity I).
        skip_dma (SkipMode): Controls OOB handling (Affinity I).

    Returns:
        tuple: (gate_up_proj_output_grad_hbm, gate_up_multiply_output_hbm)
            - gate_up_proj_output_grad_hbm: [B, 2, I_TP], Gate/up projection gradients.
            - gate_up_multiply_output_hbm: [B, I_TP], Gate * up product for weight grad
              (EA-scaled for Affinity I).
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    GATE_UP_WEIGHT_COUNT = 2

    is_affinity_i = affinity_option == AffinityOption.AFFINITY_ON_I
    is_shard_on_h = shard_option == ShardOption.SHARD_ON_HIDDEN

    # For Affinity I, down_proj_output_grad_hbm is output_hidden_states_grad[T, H].
    # B_DIM must be the block size, not the token dimension.
    if is_affinity_i:
        B_DIM = gate_up_proj_act_checkpoint_T.shape[-1]
    else:
        B_DIM, _ = down_proj_output_grad_hbm.shape
    E, I_TP_DIM, H_DIM = down_projection_weight.shape

    if is_shard_on_h:
        # Shard on H dimension — each shard processes full I_TP but H_DIM/num_shards of contraction dim
        H_DIM_SHARDED = H_DIM // num_shards
        H_SHARD_OFFSET = H_DIM_SHARDED * shard_id
        I_TP_DIM_SHARDED = I_TP_DIM
        I_TP_SHARD_OFFSET = 0
    else:
        # Shard on Free dimension — each shard processes I_TP_DIM/num_shards
        H_DIM_SHARDED = H_DIM
        H_SHARD_OFFSET = 0
        I_TP_DIM_SHARDED = I_TP_DIM // num_shards
        I_TP_SHARD_OFFSET = I_TP_DIM_SHARDED * shard_id

    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    H_TILE_SIZE = min(TILE_SIZE, H_DIM_SHARDED)
    I_TP_TILE_SIZE = min(PSUM_SIZE, I_TP_DIM_SHARDED)

    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM_SHARDED)
    I_TP_BLOCK_SIZE = min(BLOCK_I_TP * I_TP_TILE_SIZE, I_TP_DIM_SHARDED)
    B_BLOCK_SIZE = min(BLOCK_B * B_TILE_SIZE, B_DIM)

    NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
    NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
    NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

    NUM_I_TP_BLOCKS = div_ceil(I_TP_DIM_SHARDED, I_TP_BLOCK_SIZE)
    NUM_H_BLOCKS = div_ceil(H_DIM_SHARDED, H_BLOCK_SIZE)
    NUM_B_BLOCKS = div_ceil(B_DIM, B_BLOCK_SIZE)
    NUM_I_TP_INNER_TILES = div_ceil(I_TP_TILE_SIZE, TILE_SIZE)

    NUM_B_TILES_TOTAL = div_ceil(B_DIM, B_TILE_SIZE)

    gate_up_proj_output_grad_hbm = nl.ndarray(
        (B_DIM, GATE_UP_WEIGHT_COUNT, I_TP_DIM),
        dtype=compute_dtype,
        buffer=nl.shared_hbm,
        name=f"gate_up_proj_output_grad_hbm_shared_block_{block_idx}",
    )

    gate_up_multipy_output_hbm = nl.ndarray(
        (B_DIM, I_TP_DIM),
        dtype=compute_dtype,
        buffer=nl.shared_hbm,
        name=f"gate_up_multipy_output_block_{block_idx}",
    )

    # Affinity I: no longer need gathered buffer - gather is done inline in weight grad

    if manage_scope:
        sbm.open_scope(name=f"Gate Up Output Grad")

    # Adaptive accumulator precision: SwiGLU-bwd internal tiles follow the grad-output
    # HBM dtype. bf16 caller (nkilib default) -> bf16 (== baseline, fits SBUF); fp32
    # caller (e.g. nki_moe.py fp32 grad buffers) -> fp32 (full-precision SwiGLU bwd).
    if grad_accum_dtype is None:
        grad_accum_dtype = compute_dtype

    up_activation_grad = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
        dtype=compute_dtype,  # grad output of up — stays bf16 (cast on final multiply)
        name=f"up_act_grad_{block_idx}",
        align=32,
    )

    # Compute silu_dx(gate_activation) — fp32 (SwiGLU bwd internal compute)
    silu_dx_gate = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
        dtype=grad_accum_dtype,
        name=f"silu_dx_gate_{block_idx}",
        align=32,
    )
    silu_gate_grad = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
        dtype=grad_accum_dtype,
        name=f"silu_gate_grad_{block_idx}",
        align=32,
    )

    # Gradient of gate_activation: d_gate = d_silu_gate * silu_dx(gate)
    gate_activation_grad = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
        dtype=compute_dtype,  # grad output of gate — stays bf16 (cast on final multiply)
        name=f"gate_act_grad_{block_idx}",
        align=32,
    )

    # Allocate clamping buffers for non-linear activation gradient clamping
    if clamp_limits.non_linear_clamp_upper_limit != None or clamp_limits.non_linear_clamp_lower_limit != None:
        clamp_mask1 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        clamp_mask2 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        clamp_mask = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)

    # Allocate clamping buffers for linear activation gradient clamping
    if clamp_limits.linear_clamp_upper_limit != None or clamp_limits.linear_clamp_lower_limit != None:
        linear_clamp_mask1 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        linear_clamp_mask2 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        linear_clamp_mask = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)

    gate_silu_activation = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
        dtype=grad_accum_dtype,  # silu(gate) fwd recompute — follows grad-out dtype
        name=f"gate_silu_act_{block_idx}",
        align=32,
    )

    block_gate_up_mult = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
        dtype=compute_dtype,  # swiglu fwd value, stored to bf16 HBM — keep bf16
        name=f"gate_up_mult_{block_idx}",
        align=32,
    )

    partial_block_gate_up_mult_grad = []
    transpose_psum_idx = 0
    matmul_psum_idx = 0
    gate_activation = []
    up_activation = []

    down_proj_weight_temp = []
    for n_buffer in range(BUFFER_DEGREE):
        down_proj_weight_temp.append(
            sbm.alloc_stack(
                (TILE_SIZE, H_BLOCK_SIZE),
                dtype=compute_dtype,
                align=32,
            )
        )
        partial_block_gate_up_mult_grad.append(
            sbm.alloc_stack(
                (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
                dtype=grad_accum_dtype,  # grad input of swiglu block — follows grad-out dtype
                align=32,
            )
        )
        gate_activation.append(
            sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        )
        up_activation.append(
            sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        )

    # Affinity I: allocate buffered grad_load_temp for indirect DMA loads
    grad_load_temp = []
    recv_buf_list = []
    ea_grad_reduced_list = []
    if is_affinity_i:
        for n_buffer in range(BUFFER_DEGREE):
            grad_load_temp.append(sbm.alloc_stack((B_TILE_SIZE, H_BLOCK_SIZE), dtype=compute_dtype, align=32))

        for b_tile in range(NUM_B_TILES_TOTAL):
            recv_buf_list.append(
                sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32, name=f"ea_recv_{block_idx}_{b_tile}", align=32)
            )
            ea_grad_reduced_list.append(
                sbm.alloc_stack((B_TILE_SIZE, 1), dtype=compute_dtype, name=f"ea_gr_{block_idx}_{b_tile}", align=32)
            )

    # Shard-on-H: allocate recv buffer for sendrecv reduce-scatter
    if is_shard_on_h:
        b_tiles_per_core = max(1, NUM_B_TILES // num_shards)
        core_b_start = b_tiles_per_core * shard_id if NUM_B_TILES >= num_shards else 0
        peer_b_start = b_tiles_per_core * (1 - shard_id) if NUM_B_TILES >= num_shards else 0
        core_b_end = core_b_start + b_tiles_per_core
        sendrecv_recv_buf = sbm.alloc_stack(
            (B_TILE_SIZE, b_tiles_per_core, I_TP_BLOCK_SIZE), dtype=grad_accum_dtype, align=32
        )

    # Affinity I: allocate EA grad accumulators and broadcast expert_idx
    if is_affinity_i:
        ea_expert_idx_tensor = expert_idx_broadcast[0:B_TILE_SIZE, block_idx : block_idx + 1]
        ea_grad_accum_list = []
        for b_tile in range(NUM_B_TILES_TOTAL):
            acc = sbm.alloc_stack(
                (B_TILE_SIZE, 1), dtype=nl.float32, name=f"ea_grad_acc_{block_idx}_{b_tile}", align=32
            )
            nisa.memset(acc, value=0.0)
            ea_grad_accum_list.append(acc)
        ea_product_temp = sbm.alloc_stack(
            (B_TILE_SIZE, I_TP_BLOCK_SIZE),
            dtype=nl.float32,
            name=f"ea_product_{block_idx}",
            align=32,
        )
        ea_reduce_temp = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32, name=f"ea_reduce_{block_idx}", align=32)
        scaled_gate_up_mult_tile = sbm.alloc_stack(
            (B_TILE_SIZE, I_TP_BLOCK_SIZE),
            dtype=compute_dtype,
            name=f"scaled_gup_{block_idx}",
            align=32,
        )

        # Pre-load all EA tiles and offsets for the entire block.
        # B_DIM <= 1024 so at most 8 tiles fit in SBUF.
        # ea_tiles_all: single (B_TILE_SIZE, NUM_B_TILES_TOTAL) tensor, sliced per tile.
        # ea_offsets_all: list of (B_TILE_SIZE, 1) tensors (vector_offset requires standalone tensors).
        ea_tiles_all = sbm.alloc_stack(
            (B_TILE_SIZE, NUM_B_TILES_TOTAL), dtype=nl.float32, name=f"ea_tiles_all_{block_idx}", align=32
        )
        ea_offsets_all = []
        for b_tile in range(NUM_B_TILES_TOTAL):
            token_off = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, name=f"ea_off_{block_idx}_{b_tile}", align=32)
            addr_tmp = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, name=f"ea_addr_{block_idx}_{b_tile}", align=32)
            _generate_dynamic_offsets(
                block_token_pos_to_id_full,
                ea_expert_idx_tensor,
                token_off,
                addr_tmp,
                b_tile,
                skip_dma,
                E,
            )
            ea_offsets_all.append(token_off)

            ea_dst = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32, name=f"ea_load_{block_idx}_{b_tile}", align=32)
            if skip_dma.skip_token:
                nisa.memset(ea_dst, value=0.0)
            nisa.dma_copy(
                dst=ea_dst,
                src=expert_affinities_masked.ap(
                    pattern=[[expert_affinities_masked.shape[1], B_TILE_SIZE], [1, 1]],
                    offset=0,
                    vector_offset=token_off,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
            nisa.tensor_copy(dst=ea_tiles_all[:, b_tile], src=ea_dst)

    sbm.open_scope(interleave_degree=BUFFER_DEGREE, name=f"Gate Up Output Grad {block_idx}")

    for i_tp_block_idx in range(NUM_I_TP_BLOCKS):
        # Global I offset includes shard offset
        i_tp_global_offset = I_TP_SHARD_OFFSET + i_tp_block_idx * I_TP_BLOCK_SIZE
        valid_i_block = min(I_TP_BLOCK_SIZE, I_TP_DIM_SHARDED - i_tp_block_idx * I_TP_BLOCK_SIZE)

        for b_block_idx in range(NUM_B_BLOCKS):
            gup_buffer_idx = (i_tp_block_idx * NUM_B_BLOCKS + b_block_idx) % BUFFER_DEGREE
            # Perform Gup Forward and Save Intermediate States
            if skip_gate_proj:
                _load_gate_up_projection_checkpoint(
                    gate_up_proj_act_checkpoint_T,
                    up_activation[gup_buffer_idx],
                    NUM_B_TILES,
                    I_TP_BLOCK_SIZE,
                    B_TILE_SIZE,
                    B_DIM,
                    block_idx,
                    GATE_UP_WEIGHT_COUNT,
                    I_TP_DIM,
                    1,
                    i_tp_global_offset,
                    b_block_idx,
                    B_BLOCK_SIZE,
                    valid_i_block,
                )
                apply_activation(
                    dst=block_gate_up_mult[:, :, 0:valid_i_block],
                    data=up_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                    act_fn=activation_type,
                )
            else:
                for gate_or_up in range(GATE_UP_WEIGHT_COUNT):
                    if gate_or_up == 0:
                        _load_gate_up_projection_checkpoint(
                            gate_up_proj_act_checkpoint_T,
                            gate_activation[gup_buffer_idx],
                            NUM_B_TILES,
                            I_TP_BLOCK_SIZE,
                            B_TILE_SIZE,
                            B_DIM,
                            block_idx,
                            GATE_UP_WEIGHT_COUNT,
                            I_TP_DIM,
                            gate_or_up,
                            i_tp_global_offset,
                            b_block_idx,
                            B_BLOCK_SIZE,
                            valid_i_block,
                        )
                        apply_activation(
                            dst=gate_silu_activation[:, :, 0:valid_i_block],
                            data=gate_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                            act_fn=activation_type,
                        )
                    else:
                        _load_gate_up_projection_checkpoint(
                            gate_up_proj_act_checkpoint_T,
                            up_activation[gup_buffer_idx],
                            NUM_B_TILES,
                            I_TP_BLOCK_SIZE,
                            B_TILE_SIZE,
                            B_DIM,
                            block_idx,
                            GATE_UP_WEIGHT_COUNT,
                            I_TP_DIM,
                            gate_or_up,
                            i_tp_global_offset,
                            b_block_idx,
                            B_BLOCK_SIZE,
                            valid_i_block,
                        )
                        nisa.tensor_tensor(
                            dst=block_gate_up_mult[:, :, 0:valid_i_block],
                            op=nl.multiply,
                            data1=gate_silu_activation[:, :, 0:valid_i_block],
                            data2=up_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                        )

            # Store block_gate_up_mult to gate_up_multipy_output_hbm
            for b_tile_idx in range(NUM_B_TILES):
                b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                if is_affinity_i:
                    # Affinity I: scale gate_up_mult by EA before storing
                    global_b_tile_idx = b_offset // B_TILE_SIZE
                    nisa.tensor_scalar(
                        dst=scaled_gate_up_mult_tile[0:B_TILE_SIZE, 0:valid_i_block],
                        data=block_gate_up_mult[:, b_tile_idx, 0:valid_i_block],
                        op0=nl.multiply,
                        operand0=ea_tiles_all[:, global_b_tile_idx],
                    )
                    nisa.dma_copy(
                        dst=gate_up_multipy_output_hbm.ap(
                            pattern=[[I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                            offset=b_offset * I_TP_DIM + i_tp_global_offset,
                        ),
                        src=scaled_gate_up_mult_tile[0:B_TILE_SIZE, 0:valid_i_block],
                        dge_mode=dge_mode.hwdge,
                    )
                else:
                    nisa.dma_copy(
                        dst=gate_up_multipy_output_hbm.ap(
                            pattern=[[I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                            offset=b_offset * I_TP_DIM + i_tp_global_offset,
                        ),
                        src=block_gate_up_mult[:, b_tile_idx, 0:valid_i_block],
                        dge_mode=dge_mode.hwdge,
                    )

            result_buffer_idx = (i_tp_block_idx * NUM_B_BLOCKS + b_block_idx) % BUFFER_DEGREE

            for h_block_idx in range(NUM_H_BLOCKS):
                h_block_start = h_block_idx * H_BLOCK_SIZE
                valid_h_block = min(H_BLOCK_SIZE, H_DIM_SHARDED - h_block_start)
                # Global H offset for DMA addressing into full [E, I_TP, H] tensors
                h_block_start_global = H_SHARD_OFFSET + h_block_start

                down_projection_weight_transposed = sbm.alloc_stack(
                    (H_TILE_SIZE, NUM_H_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, buffer=nl.sbuf, align=32
                )
                down_proj_output_grad_transposed = sbm.alloc_stack(
                    (H_TILE_SIZE, NUM_H_TILES, B_BLOCK_SIZE),
                    dtype=compute_dtype,
                    buffer=nl.sbuf,
                    align=32,  # DMA transpose requires 32-byte alignment
                )

                # Load and Transpose DP Weights using HWDGE dma_copy + nc_transpose
                for i_tile_idx in range(NUM_I_TP_TILES):
                    i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                    if i_tile_start >= valid_i_block:
                        break
                    valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)

                    for i_inner_idx in range(NUM_I_TP_INNER_TILES):
                        i_inner_start = i_inner_idx * TILE_SIZE
                        if i_inner_start >= valid_i_tile:
                            break
                        # Use global I offset for weight loading
                        i_tp_offset = i_tp_global_offset + i_tile_start + i_inner_start
                        valid_i_inner = min(TILE_SIZE, valid_i_tile - i_inner_start)
                        buffer_idx = i_inner_idx % BUFFER_DEGREE
                        nisa.dma_copy(
                            dst=down_proj_weight_temp[buffer_idx][0:valid_i_inner, 0:valid_h_block],
                            src=down_projection_weight.ap(  # E, I_TP, H
                                pattern=[[H_DIM, valid_i_inner], [1, valid_h_block]],
                                offset=i_tp_offset * H_DIM + h_block_start_global,
                                scalar_offset=expert_idx,
                                indirect_dim=0,
                            ),
                            dge_mode=dge_mode.hwdge,
                        )
                        for h_tile_idx in range(NUM_H_TILES):
                            h_tile_start = h_tile_idx * H_TILE_SIZE
                            if h_tile_start >= valid_h_block:
                                break
                            valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)
                            transpose_psum_idx += 1
                            transpose_psum_idx = transpose_psum_idx % nl.tile_size.psum_num_banks

                            down_proj_weight_transposed_psum = nl.ndarray(
                                (H_TILE_SIZE, TILE_SIZE),
                                dtype=compute_dtype,
                                buffer=nl.psum,
                                address=(0, transpose_psum_idx * PSUM_BANK_SIZE),
                            )
                            nisa.nc_transpose(
                                dst=down_proj_weight_transposed_psum[0:valid_h_tile, 0:valid_i_inner],
                                data=down_proj_weight_temp[buffer_idx][
                                    0:valid_i_inner, nl.ds(h_tile_start, valid_h_tile)
                                ],
                            )
                            i_dst_offset = i_tile_start + i_inner_start
                            nisa.tensor_copy(
                                dst=down_projection_weight_transposed[
                                    0:valid_h_tile, h_tile_idx, nl.ds(i_dst_offset, valid_i_inner)
                                ],
                                src=down_proj_weight_transposed_psum[0:valid_h_tile, 0:valid_i_inner],
                            )

                # Load and Transpose down_proj_output_grad
                if is_affinity_i:
                    # Affinity I: load from output_hidden_states_grad[T, H] via indirect addressing
                    # and transpose per b_tile
                    for b_tile_idx in range(NUM_B_TILES):
                        b_tile_start = b_tile_idx * B_TILE_SIZE
                        global_b_tile_idx = (b_block_idx * B_BLOCK_SIZE + b_tile_start) // B_TILE_SIZE
                        buf_idx = b_tile_idx % BUFFER_DEGREE

                        if skip_dma.skip_token:
                            nisa.memset(grad_load_temp[buf_idx], value=0)

                        nisa.dma_copy(
                            dst=grad_load_temp[buf_idx][0:B_TILE_SIZE, 0:valid_h_block],
                            src=down_proj_output_grad_hbm.ap(
                                pattern=[[H_DIM, B_TILE_SIZE], [1, valid_h_block]],
                                offset=h_block_start_global,
                                vector_offset=block_token_pos_to_id_full.ap(
                                    pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]],
                                    offset=global_b_tile_idx,
                                ),
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )

                        # Transpose [B_TILE, H_TILE] -> [H_TILE, B_TILE] into down_proj_output_grad_transposed
                        for h_tile_idx in range(NUM_H_TILES):
                            h_tile_start = h_tile_idx * H_TILE_SIZE
                            if h_tile_start >= valid_h_block:
                                break
                            valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)
                            transpose_psum_idx += 1
                            transpose_psum_idx = transpose_psum_idx % nl.tile_size.psum_num_banks
                            grad_transpose_psum = nl.ndarray(
                                (H_TILE_SIZE, B_TILE_SIZE),
                                dtype=compute_dtype,
                                buffer=nl.psum,
                                address=(0, transpose_psum_idx * PSUM_BANK_SIZE),
                            )
                            nisa.nc_transpose(
                                dst=grad_transpose_psum[0:valid_h_tile, 0:B_TILE_SIZE],
                                data=grad_load_temp[buf_idx][0:B_TILE_SIZE, nl.ds(h_tile_start, valid_h_tile)],
                            )
                            nisa.tensor_copy(
                                dst=down_proj_output_grad_transposed[
                                    0:valid_h_tile, h_tile_idx, nl.ds(b_tile_start, B_TILE_SIZE)
                                ],
                                src=grad_transpose_psum[0:valid_h_tile, 0:B_TILE_SIZE],
                            )
                else:
                    # Affinity H: load from block-local down_proj_output_grad_hbm[B, H]
                    for h_tile_idx in range(NUM_H_TILES):
                        h_tile_start = h_tile_idx * H_TILE_SIZE
                        if h_tile_start >= valid_h_block:
                            break
                        valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)
                        nisa.dma_transpose(
                            dst=down_proj_output_grad_transposed.ap(
                                pattern=[[NUM_H_TILES * B_BLOCK_SIZE, valid_h_tile], [1, 1], [1, 1], [1, B_BLOCK_SIZE]],
                                offset=(h_tile_idx * B_BLOCK_SIZE),
                            ),
                            src=down_proj_output_grad_hbm.ap(
                                pattern=[[H_DIM, B_BLOCK_SIZE], [1, 1], [1, 1], [1, valid_h_tile]],
                                offset=(b_block_idx * B_BLOCK_SIZE) * H_DIM + h_block_start_global + h_tile_start,
                            ),
                        )

                # Matmul [B, H] @ [H, I] - now using full H (no sharding)
                for b_tile_idx in range(NUM_B_TILES):
                    b_tile_start = b_tile_idx * B_TILE_SIZE

                    for i_tile_idx in range(NUM_I_TP_TILES):
                        i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                        if i_tile_start >= valid_i_block:
                            break
                        valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)

                        # Psum address after transpose buffers (use separate bank range)
                        matmul_psum_idx += 1
                        matmul_psum_idx = matmul_psum_idx % nl.tile_size.psum_num_banks
                        result_psum = nl.ndarray(
                            (B_TILE_SIZE, I_TP_TILE_SIZE),
                            dtype=nl.float32,
                            buffer=nl.psum,
                            address=(0, matmul_psum_idx * PSUM_BANK_SIZE),
                        )
                        for h_tile_idx in range(NUM_H_TILES):
                            h_tile_start_inner = h_tile_idx * H_TILE_SIZE
                            if h_tile_start_inner >= valid_h_block:
                                break
                            valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start_inner)
                            nisa.nc_matmul(
                                dst=result_psum[:, 0:valid_i_tile],
                                stationary=down_proj_output_grad_transposed[
                                    0:valid_h_tile, h_tile_idx, nl.ds(b_tile_start, B_TILE_SIZE)
                                ],
                                moving=down_projection_weight_transposed[
                                    0:valid_h_tile, h_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                ],
                            )

                        if h_block_idx == 0:
                            nisa.tensor_copy(
                                src=result_psum[:, 0:valid_i_tile],
                                dst=partial_block_gate_up_mult_grad[result_buffer_idx][
                                    :, b_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                ],
                            )
                        else:
                            nisa.tensor_tensor(
                                data1=result_psum[:, 0:valid_i_tile],
                                data2=partial_block_gate_up_mult_grad[result_buffer_idx][
                                    :, b_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                ],
                                dst=partial_block_gate_up_mult_grad[result_buffer_idx][
                                    :, b_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                ],
                                op=nl.add,
                            )

                sbm.increment_section()

            # Shard-on-H: reduce-scatter on B tiles — each core reduces only its half
            if is_shard_on_h:
                nisa.sendrecv(
                    src=partial_block_gate_up_mult_grad[result_buffer_idx][
                        :, peer_b_start : peer_b_start + b_tiles_per_core, 0:valid_i_block
                    ],
                    dst=sendrecv_recv_buf[:, 0:b_tiles_per_core, 0:valid_i_block],
                    send_to_rank=(1 - shard_id),
                    recv_from_rank=(1 - shard_id),
                    pipe_id=1,
                )
                nisa.tensor_tensor(
                    dst=partial_block_gate_up_mult_grad[result_buffer_idx][:, core_b_start:core_b_end, 0:valid_i_block],
                    op=nl.add,
                    data1=partial_block_gate_up_mult_grad[result_buffer_idx][
                        :, core_b_start:core_b_end, 0:valid_i_block
                    ],
                    data2=sendrecv_recv_buf[:, 0:b_tiles_per_core, 0:valid_i_block],
                )

            # Affinity I: EA grad accumulation and matmul result scaling
            if is_affinity_i:
                ea_b_start = core_b_start if is_shard_on_h else 0
                ea_b_count = b_tiles_per_core if is_shard_on_h else NUM_B_TILES
                for b_tile_idx in range(ea_b_start, ea_b_start + ea_b_count):
                    b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                    global_b_tile_idx = b_offset // B_TILE_SIZE

                    # EA grad: product = upstream_grad * gate_up_mult, reduce over I
                    nisa.tensor_tensor(
                        dst=ea_product_temp[0:B_TILE_SIZE, 0:valid_i_block],
                        data1=partial_block_gate_up_mult_grad[result_buffer_idx][:, b_tile_idx, 0:valid_i_block],
                        data2=block_gate_up_mult[:, b_tile_idx, 0:valid_i_block],
                        op=nl.multiply,
                    )
                    nisa.tensor_reduce(
                        dst=ea_reduce_temp,
                        op=nl.add,
                        data=ea_product_temp[0:B_TILE_SIZE, 0:valid_i_block],
                        axis=1,
                    )
                    nisa.tensor_tensor(
                        dst=ea_grad_accum_list[global_b_tile_idx],
                        op=nl.add,
                        data1=ea_grad_accum_list[global_b_tile_idx],
                        data2=ea_reduce_temp,
                    )

                    # Scale matmul result by EA before activation backward
                    nisa.tensor_scalar(
                        dst=partial_block_gate_up_mult_grad[result_buffer_idx][:, b_tile_idx, 0:valid_i_block],
                        data=partial_block_gate_up_mult_grad[result_buffer_idx][:, b_tile_idx, 0:valid_i_block],
                        op0=nl.multiply,
                        operand0=ea_tiles_all[:, global_b_tile_idx],
                    )

            # Activation backward — shard-on-H operates on this core's B tile slice only
            act_bwd_b = slice(core_b_start, core_b_end) if is_shard_on_h else slice(0, NUM_B_TILES)

            if skip_gate_proj:
                apply_activation_dx(
                    dst=up_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    data=up_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                    act_fn=activation_type,
                )
                nisa.tensor_tensor(
                    dst=up_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    op=nl.multiply,
                    data1=up_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    data2=partial_block_gate_up_mult_grad[result_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                )
            else:
                nisa.tensor_tensor(
                    dst=up_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    op=nl.multiply,
                    data1=partial_block_gate_up_mult_grad[result_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                    data2=gate_silu_activation[:, act_bwd_b, 0:valid_i_block],
                )

            # Apply gradient clamping for linear activation (up) backward pass
            if clamp_limits.linear_clamp_upper_limit != None or clamp_limits.linear_clamp_lower_limit != None:
                nisa.memset(linear_clamp_mask1[:, act_bwd_b, :], value=1.0)
                nisa.memset(linear_clamp_mask2[:, act_bwd_b, :], value=1.0)

                if clamp_limits.linear_clamp_upper_limit != None:
                    nisa.tensor_scalar(
                        dst=linear_clamp_mask1[:, act_bwd_b, 0:valid_i_block],
                        data=up_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                        op0=nl.less,
                        operand0=clamp_limits.linear_clamp_upper_limit,
                    )

                if clamp_limits.linear_clamp_lower_limit != None:
                    nisa.tensor_scalar(
                        dst=linear_clamp_mask2[:, act_bwd_b, 0:valid_i_block],
                        data=up_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                        op0=nl.greater,
                        operand0=clamp_limits.linear_clamp_lower_limit,
                    )

                nisa.tensor_tensor(
                    dst=linear_clamp_mask[:, act_bwd_b, 0:valid_i_block],
                    data1=linear_clamp_mask1[:, act_bwd_b, 0:valid_i_block],
                    data2=linear_clamp_mask2[:, act_bwd_b, 0:valid_i_block],
                    op=nl.logical_and,
                )

                nisa.tensor_tensor(
                    dst=up_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    op=nl.multiply,
                    data1=up_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    data2=linear_clamp_mask[:, act_bwd_b, 0:valid_i_block],
                )

            nisa.tensor_tensor(
                dst=silu_gate_grad[:, act_bwd_b, 0:valid_i_block],
                op=nl.multiply,
                data1=partial_block_gate_up_mult_grad[result_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                data2=up_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
            )

            apply_activation_dx(
                dst=silu_dx_gate[:, act_bwd_b, 0:valid_i_block],
                data=gate_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                act_fn=activation_type,
            )

            nisa.tensor_tensor(
                dst=gate_activation_grad[:, act_bwd_b, 0:valid_i_block],
                op=nl.multiply,
                data1=silu_gate_grad[:, act_bwd_b, 0:valid_i_block],
                data2=silu_dx_gate[:, act_bwd_b, 0:valid_i_block],
            )

            # Apply gradient clamping for non-linear activation backward pass
            if clamp_limits.non_linear_clamp_upper_limit != None or clamp_limits.non_linear_clamp_lower_limit != None:
                nisa.memset(clamp_mask1[:, act_bwd_b, :], value=1.0)
                nisa.memset(clamp_mask2[:, act_bwd_b, :], value=1.0)

                if clamp_limits.non_linear_clamp_upper_limit != None:
                    nisa.tensor_scalar(
                        dst=clamp_mask1[:, act_bwd_b, 0:valid_i_block],
                        data=gate_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                        op0=nl.less,
                        operand0=clamp_limits.non_linear_clamp_upper_limit,
                    )

                if clamp_limits.non_linear_clamp_lower_limit != None:
                    nisa.tensor_scalar(
                        dst=clamp_mask2[:, act_bwd_b, 0:valid_i_block],
                        data=gate_activation[gup_buffer_idx][:, act_bwd_b, 0:valid_i_block],
                        op0=nl.greater,
                        operand0=clamp_limits.non_linear_clamp_lower_limit,
                    )

                nisa.tensor_tensor(
                    dst=clamp_mask[:, act_bwd_b, 0:valid_i_block],
                    data1=clamp_mask1[:, act_bwd_b, 0:valid_i_block],
                    data2=clamp_mask2[:, act_bwd_b, 0:valid_i_block],
                    op=nl.logical_and,
                )

                nisa.tensor_tensor(
                    dst=gate_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    op=nl.multiply,
                    data1=gate_activation_grad[:, act_bwd_b, 0:valid_i_block],
                    data2=clamp_mask[:, act_bwd_b, 0:valid_i_block],
                )

            # Write gate/up gradients to HBM — each core stores its B tile range
            store_b_start = core_b_start if is_shard_on_h else 0
            store_b_end = core_b_end if is_shard_on_h else NUM_B_TILES
            for b_tile_idx in range(store_b_start, store_b_end):
                b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                if not skip_gate_proj:
                    nisa.dma_copy(
                        dst=gate_up_proj_output_grad_hbm.ap(
                            pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                            offset=b_offset * GATE_UP_WEIGHT_COUNT * I_TP_DIM + 0 * I_TP_DIM + i_tp_global_offset,
                        ),
                        src=gate_activation_grad[:, b_tile_idx, 0:valid_i_block],
                        dge_mode=dge_mode.hwdge,
                    )
                nisa.dma_copy(
                    dst=gate_up_proj_output_grad_hbm.ap(
                        pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                        offset=b_offset * GATE_UP_WEIGHT_COUNT * I_TP_DIM + 1 * I_TP_DIM + i_tp_global_offset,
                    ),
                    src=up_activation_grad[:, b_tile_idx, 0:valid_i_block],
                    dge_mode=dge_mode.hwdge,
                )

    sbm.close_scope()

    # Affinity I: sendrecv to combine EA grad across LNC shards, then store
    # Each core handles half the b_tiles to avoid WAW hazard and reduce redundant work.
    if is_affinity_i:
        if is_shard_on_h:
            # Shard-on-H: each core already accumulated over full I_TP, no cross-core reduction needed.
            # The accumulation loop assigns per-block local B tile ranges to each core,
            # so the global tiles each core owns are interleaved across B blocks
            # (not contiguous). Iterate over the same pattern used during accumulation.
            for b_block_idx in range(NUM_B_BLOCKS):
                for b_tile_local in range(core_b_start, core_b_end):
                    tile_idx = b_block_idx * NUM_B_TILES + b_tile_local
                    if tile_idx < NUM_B_TILES_TOTAL:
                        nisa.tensor_copy(
                            dst=ea_grad_reduced_list[tile_idx],
                            src=ea_grad_accum_list[tile_idx],
                            engine=nisa.scalar_engine,
                        )
                        nisa.dma_copy(
                            dst=expert_affinities_masked_grad.ap(
                                pattern=[[1, B_TILE_SIZE], [1, 1]],
                                offset=0,
                                vector_offset=ea_offsets_all[tile_idx],
                                indirect_dim=0,
                            ),
                            src=ea_grad_reduced_list[tile_idx],
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )
        else:
            # Shard-on-free (I): each core has partial I sum, sendrecv to combine.
            if NUM_B_TILES_TOTAL >= num_shards:
                tiles_per_shard = NUM_B_TILES_TOTAL // num_shards
                shard_tile_start = tiles_per_shard * shard_id
            else:
                tiles_per_shard = NUM_B_TILES_TOTAL
                shard_tile_start = 0
            for b_tile in range(NUM_B_TILES_TOTAL):
                nisa.sendrecv(
                    src=ea_grad_accum_list[b_tile],
                    dst=recv_buf_list[b_tile],
                    send_to_rank=(1 - shard_id),
                    recv_from_rank=(1 - shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(
                    dst=ea_grad_accum_list[b_tile],
                    op=nl.add,
                    data1=ea_grad_accum_list[b_tile],
                    data2=recv_buf_list[b_tile],
                )
            for b_tile in range(tiles_per_shard):
                tile_idx = shard_tile_start + b_tile
                nisa.tensor_copy(
                    dst=ea_grad_reduced_list[tile_idx],
                    src=ea_grad_accum_list[tile_idx],
                    engine=nisa.scalar_engine,
                )
                nisa.dma_copy(
                    dst=expert_affinities_masked_grad.ap(
                        pattern=[[1, B_TILE_SIZE], [1, 1]],
                        offset=0,
                        vector_offset=ea_offsets_all[tile_idx],
                        indirect_dim=0,
                    ),
                    src=ea_grad_reduced_list[tile_idx],
                    oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                )

    if manage_scope:
        sbm.close_scope()
    nisa.core_barrier(gate_up_proj_output_grad_hbm, (0, 1))
    nisa.core_barrier(gate_up_multipy_output_hbm, (0, 1))

    return gate_up_proj_output_grad_hbm, gate_up_multipy_output_hbm


def _compute_down_projection_weight_grad(
    gate_up_multipy_output_hbm,
    down_proj_output_grad_hbm,
    down_projection_weight_grad,
    shard_id,
    num_shards,
    expert_idx,
    compute_dtype,
    block_idx,
    sbm,
    BLOCK_H=1,
    BLOCK_B=2,
    BLOCK_I_TP=2,
    block_token_pos_to_id_full=None,
    skip_dma=None,
    iota_vec=None,
    manage_scope=True,
    BUFFER_DEGREE=3,
):
    """
    Compute down projection weight gradient with H-dimension sharding.

    Performs matmul of gate_up_multiply_output^T @ down_proj_output_grad to compute
    the weight gradient. Each shard processes H/num_shards of the hidden dimension.

    Args:
        gate_up_multipy_output_hbm (nl.ndarray): [B, I_TP], Gate * up product.
        down_proj_output_grad_hbm (nl.ndarray): [B, H], Down projection output gradient.
        down_projection_weight_grad (nl.ndarray): [E, I_TP, H], Output weight gradient.
        shard_id (int): Current shard ID.
        num_shards (int): Number of LNC shards.
        expert_idx (nl.ndarray): [1, 1], Expert index for indirect addressing.
        compute_dtype: Computation data type.
        block_idx (int): Current block index.
        sbm (SbufManager): SBUF memory manager.
        BLOCK_H (int): H dimension blocking factor.
        BLOCK_B (int): B dimension blocking factor.
        BLOCK_I_TP (int): I dimension blocking factor.
        block_token_pos_to_id_full (nl.ndarray, optional): Token position mapping for indirect addressing.
        skip_dma (SkipMode, optional): Controls OOB handling for DMA operations.
        iota_vec (nl.ndarray, optional): Precomputed [0, 1, 2, ..., TILE_SIZE-1] vector (unused, kept for interface consistency).

    Returns:
        None: Weight gradient is accumulated in-place.

    Notes:
        - weight_grad[expert, :, H_shard] += gate_up_mult^T @ down_proj_grad
        - Each shard writes to its H/num_shards portion.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    B_DIM, _ = gate_up_multipy_output_hbm.shape
    E, I_TP_DIM, H_DIM = down_projection_weight_grad.shape
    # Convert to Python int to ensure compile-time constants

    H_DIM_SHARDED = H_DIM // num_shards
    H_SHARD_OFFSET = H_DIM_SHARDED * shard_id

    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    H_TILE_SIZE = min(PSUM_SIZE, H_DIM_SHARDED)
    I_TP_TILE_SIZE = TILE_SIZE

    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM_SHARDED)
    I_TP_BLOCK_SIZE = min(BLOCK_I_TP * I_TP_TILE_SIZE, I_TP_DIM)
    B_BLOCK_SIZE = min(BLOCK_B * B_TILE_SIZE, B_DIM)

    NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
    NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
    NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

    NUM_I_TP_BLOCKS = div_ceil(I_TP_DIM, I_TP_BLOCK_SIZE)
    NUM_H_BLOCKS = div_ceil(H_DIM_SHARDED, H_BLOCK_SIZE)
    NUM_B_BLOCKS = div_ceil(B_DIM, B_BLOCK_SIZE)

    matmul_psum_idx = 0
    if manage_scope:
        sbm.open_scope(name=f"Down Projection Weight Grad")

    result_tiles = []
    existing_weight_grad = []

    for n_buffer in range(BUFFER_DEGREE):
        # Weight-grad accumulators follow grad-output HBM dtype (adaptive): bf16 -> baseline
        # (fits SBUF); fp32 caller (nki_moe.py) -> fp32 full-precision cross-expert accum.
        result_tiles.append(
            sbm.alloc_stack(
                (I_TP_TILE_SIZE, NUM_I_TP_TILES, H_BLOCK_SIZE),
                dtype=down_projection_weight_grad.dtype,
                align=32,
            )
        )
        existing_weight_grad.append(
            sbm.alloc_stack(
                (I_TP_TILE_SIZE, NUM_I_TP_TILES, H_BLOCK_SIZE),
                dtype=down_projection_weight_grad.dtype,
                align=32,
            )
        )

    sbm.open_scope(name=f"Down Projection Weight Grad Double buffer", interleave_degree=BUFFER_DEGREE)

    for h_block_idx in range(NUM_H_BLOCKS):
        h_block_start = h_block_idx * H_BLOCK_SIZE
        valid_h_block = min(H_BLOCK_SIZE, H_DIM_SHARDED - h_block_start)

        for i_block_idx in range(NUM_I_TP_BLOCKS):
            i_block_start = i_block_idx * I_TP_BLOCK_SIZE
            valid_i_block = min(I_TP_BLOCK_SIZE, I_TP_DIM - i_block_start)

            result_tile_idx = (h_block_idx * NUM_I_TP_BLOCKS + i_block_idx) % BUFFER_DEGREE
            for b_block_idx in range(NUM_B_BLOCKS):
                lhs_tiles = sbm.alloc_stack(
                    (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
                    dtype=compute_dtype,
                    name=f"down_wgrad_lhs_blk{block_idx}_h{h_block_idx}_i{i_block_idx}_b{b_block_idx}",
                    align=32,
                )
                rhs_tiles = sbm.alloc_stack(
                    (B_TILE_SIZE, NUM_B_TILES, H_BLOCK_SIZE),
                    dtype=compute_dtype,
                    name=f"down_wgrad_rhs_blk{block_idx}_h{h_block_idx}_i{i_block_idx}_b{b_block_idx}",
                    align=32,
                )

                for b_tile_idx in range(NUM_B_TILES):
                    b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                    i_offset = i_block_start
                    h_offset = H_SHARD_OFFSET + h_block_start

                    nisa.dma_copy(
                        dst=lhs_tiles[:, b_tile_idx, 0:valid_i_block],
                        src=gate_up_multipy_output_hbm.ap(
                            pattern=[[I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                            offset=b_offset * I_TP_DIM + i_offset,
                        ),
                    )

                    if block_token_pos_to_id_full != None:
                        # Inline gather: indirect addressing into ungathered tensor
                        global_b_tile_idx = b_offset // B_TILE_SIZE
                        if skip_dma.skip_token:
                            nisa.memset(rhs_tiles[:, b_tile_idx, 0:valid_h_block], value=0)
                        nisa.dma_copy(
                            dst=rhs_tiles[:, b_tile_idx, 0:valid_h_block],
                            src=down_proj_output_grad_hbm.ap(
                                pattern=[[H_DIM, B_TILE_SIZE], [1, valid_h_block]],
                                offset=h_offset,
                                vector_offset=block_token_pos_to_id_full.ap(
                                    pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]],
                                    offset=global_b_tile_idx,
                                ),
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )
                    else:
                        nisa.dma_copy(
                            dst=rhs_tiles[:, b_tile_idx, 0:valid_h_block],
                            src=down_proj_output_grad_hbm.ap(
                                pattern=[[H_DIM, B_TILE_SIZE], [1, valid_h_block]],
                                offset=b_offset * H_DIM + h_offset,
                            ),
                        )
                for h_tile_idx in range(NUM_H_TILES):
                    h_tile_start = h_tile_idx * H_TILE_SIZE
                    if h_tile_start >= valid_h_block:
                        break
                    valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)

                    for i_tile_idx in range(NUM_I_TP_TILES):
                        i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                        if i_tile_start >= valid_i_block:
                            break

                        matmul_psum_idx += 1
                        matmul_psum_idx = matmul_psum_idx % nl.tile_size.psum_num_banks

                        valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)
                        result_psum = nl.ndarray(
                            (I_TP_TILE_SIZE, H_TILE_SIZE),
                            buffer=nl.psum,
                            dtype=nl.float32,
                            address=(0, matmul_psum_idx * PSUM_BANK_SIZE),
                        )
                        for b_tile_idx in range(NUM_B_TILES):
                            nisa.nc_matmul(
                                dst=result_psum[0:valid_i_tile, 0:valid_h_tile],
                                stationary=lhs_tiles[:, b_tile_idx, nl.ds(i_tile_start, valid_i_tile)],
                                moving=rhs_tiles[:, b_tile_idx, nl.ds(h_tile_start, valid_h_tile)],
                            )

                        if b_block_idx == 0:
                            nisa.tensor_copy(
                                src=result_psum[0:valid_i_tile, 0:valid_h_tile],
                                dst=result_tiles[result_tile_idx][
                                    0:valid_i_tile, i_tile_idx, nl.ds(h_tile_start, valid_h_tile)
                                ],
                                engine=nisa.scalar_engine,
                            )
                        else:
                            nisa.tensor_tensor(
                                data1=result_psum[0:valid_i_tile, 0:valid_h_tile],
                                data2=result_tiles[result_tile_idx][
                                    0:valid_i_tile, i_tile_idx, nl.ds(h_tile_start, valid_h_tile)
                                ],
                                dst=result_tiles[result_tile_idx][
                                    0:valid_i_tile, i_tile_idx, nl.ds(h_tile_start, valid_h_tile)
                                ],
                                op=nl.add,
                            )

                sbm.increment_section()

            # Write Down Projection weight - load all I_TP tiles in a single DMA
            num_full_i_tp_tiles = valid_i_block // I_TP_TILE_SIZE
            partial_i_tile = valid_i_block % I_TP_TILE_SIZE

            # Bulk load/accumulate/store for all full tiles
            if num_full_i_tp_tiles > 0:
                nisa.dma_copy(
                    dst=existing_weight_grad[result_tile_idx][:, 0:num_full_i_tp_tiles, 0:valid_h_block],
                    src=down_projection_weight_grad.ap(
                        pattern=[
                            [H_DIM, I_TP_TILE_SIZE],
                            [H_DIM * I_TP_TILE_SIZE, num_full_i_tp_tiles],
                            [1, valid_h_block],
                        ],
                        offset=i_block_start * H_DIM + H_SHARD_OFFSET + h_block_start,
                        scalar_offset=expert_idx,
                        indirect_dim=0,
                    ),
                    dge_mode=dge_mode.hwdge,
                )
                nisa.tensor_tensor(
                    dst=result_tiles[result_tile_idx][:, 0:num_full_i_tp_tiles, 0:valid_h_block],
                    op=nl.add,
                    data1=existing_weight_grad[result_tile_idx][:, 0:num_full_i_tp_tiles, 0:valid_h_block],
                    data2=result_tiles[result_tile_idx][:, 0:num_full_i_tp_tiles, 0:valid_h_block],
                )
                nisa.dma_copy(
                    dst=down_projection_weight_grad.ap(
                        pattern=[
                            [H_DIM, I_TP_TILE_SIZE],
                            [H_DIM * I_TP_TILE_SIZE, num_full_i_tp_tiles],
                            [1, valid_h_block],
                        ],
                        offset=i_block_start * H_DIM + H_SHARD_OFFSET + h_block_start,
                        scalar_offset=expert_idx,
                        indirect_dim=0,
                    ),
                    src=result_tiles[result_tile_idx][:, 0:num_full_i_tp_tiles, 0:valid_h_block],
                    dge_mode=dge_mode.hwdge,
                )

            # Handle partial last tile
            if partial_i_tile > 0:
                last_tile_idx = num_full_i_tp_tiles
                i_offset = i_block_start + last_tile_idx * I_TP_TILE_SIZE
                nisa.dma_copy(
                    dst=existing_weight_grad[result_tile_idx][0:partial_i_tile, last_tile_idx, 0:valid_h_block],
                    src=down_projection_weight_grad.ap(
                        pattern=[[H_DIM, partial_i_tile], [1, 1], [1, valid_h_block]],
                        offset=i_offset * H_DIM + H_SHARD_OFFSET + h_block_start,
                        scalar_offset=expert_idx,
                        indirect_dim=0,
                    ),
                    dge_mode=dge_mode.hwdge,
                )
                nisa.tensor_tensor(
                    dst=result_tiles[result_tile_idx][0:partial_i_tile, last_tile_idx, 0:valid_h_block],
                    op=nl.add,
                    data1=existing_weight_grad[result_tile_idx][0:partial_i_tile, last_tile_idx, 0:valid_h_block],
                    data2=result_tiles[result_tile_idx][0:partial_i_tile, last_tile_idx, 0:valid_h_block],
                )
                nisa.dma_copy(
                    dst=down_projection_weight_grad.ap(
                        pattern=[[H_DIM, partial_i_tile], [1, 1], [1, valid_h_block]],
                        offset=i_offset * H_DIM + H_SHARD_OFFSET + h_block_start,
                        scalar_offset=expert_idx,
                        indirect_dim=0,
                    ),
                    src=result_tiles[result_tile_idx][0:partial_i_tile, last_tile_idx, 0:valid_h_block],
                    dge_mode=dge_mode.hwdge,
                )

    sbm.close_scope()
    if manage_scope:
        sbm.close_scope()


def _compute_hidden_states_grad(
    gate_up_proj_output_grad_hbm,
    gate_up_proj_weight,
    hidden_states_grad,
    block_token_pos_to_id_full,
    shard_id,
    num_shards,
    expert_idx,
    skip_dma,
    compute_dtype,
    is_tensor_update_accumulating,
    block_idx,
    sbm,
    BLOCK_H=1,
    BLOCK_B=2,
    BLOCK_I_TP=2,
    manage_scope=True,
    BUFFER_DEGREE=3,
    skip_gate_proj=False,
):
    """
    Compute hidden states gradient with H-dimension sharding.

    Performs matmul of gate_up_proj_output_grad @ gate_up_proj_weight to compute
    the hidden states gradient. Each shard processes H/num_shards of the hidden dimension.

    Args:
        gate_up_proj_output_grad_hbm (nl.ndarray): [B, 2, I_TP], Gate/up projection gradients.
        gate_up_proj_weight (nl.ndarray): [E, H, 2, I_TP], Gate/up projection weights.
        hidden_states_grad (nl.ndarray): [T, H], Output hidden states gradient.
        block_token_pos_to_id_full (nl.ndarray): [B_TILE_SIZE, NUM_TILES], Token position mapping.
        shard_id (int): Current shard ID.
        num_shards (int): Number of LNC shards.
        expert_idx (nl.ndarray): [1, 1], Expert index for indirect addressing.
        skip_dma (SkipMode): Controls OOB handling for DMA operations.
        compute_dtype: Computation data type.
        is_tensor_update_accumulating (bool): Whether to accumulate into existing gradients.
        block_idx (int): Current block index.
        sbm (SbufManager): SBUF memory manager.
        BLOCK_H (int): H dimension blocking factor.
        BLOCK_B (int): B dimension blocking factor.
        BLOCK_I_TP (int): I dimension blocking factor.
        skip_gate_proj (bool): Whether to omit the gate-projection contribution.

    Returns:
        None: Hidden states gradient is written/accumulated in-place.

    Notes:
        - hidden_grad[:, H_shard] = sum(gate_up_grad @ gate_up_weight, axis=gate_or_up)
        - Uses indirect addressing via block_token_pos_to_id_full for scatter writes.
    """

    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    B_DIM, GATE_UP_WEIGHT_COUNT, I_TP_DIM = gate_up_proj_output_grad_hbm.shape
    T, H_DIM = hidden_states_grad.shape
    H_DIM_SHARDED = H_DIM // num_shards
    H_SHARD_OFFSET = H_DIM_SHARDED * shard_id

    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    H_TILE_SIZE = min(PSUM_SIZE, H_DIM_SHARDED)
    I_TP_TILE_SIZE = min(TILE_SIZE, I_TP_DIM)

    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM_SHARDED)
    I_TP_BLOCK_SIZE = min(BLOCK_I_TP * I_TP_TILE_SIZE, I_TP_DIM)
    B_BLOCK_SIZE = min(BLOCK_B * B_TILE_SIZE, B_DIM)

    NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
    NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
    NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

    NUM_I_TP_BLOCKS = div_ceil(I_TP_DIM, I_TP_BLOCK_SIZE)
    NUM_H_BLOCKS = div_ceil(H_DIM_SHARDED, H_BLOCK_SIZE)
    NUM_B_BLOCKS = div_ceil(B_DIM, B_BLOCK_SIZE)

    NUM_H_INNER_TILES = div_ceil(H_TILE_SIZE, TILE_SIZE)
    if manage_scope:
        sbm.open_scope(name=f"Hidden States Grad")

    matmul_psum_idx = 0
    transpose_psum_idx = 0
    result_tiles = []
    existing_hidden_grad = []
    rhs_temp = []
    for n_buffer in range(BUFFER_DEGREE):
        rhs_temp.append(
            sbm.alloc_stack(
                (TILE_SIZE, NUM_H_INNER_TILES, I_TP_BLOCK_SIZE),
                dtype=compute_dtype,
                align=32,
            )
        )
        # Cross-expert RMW accumulator follows the grad-output HBM dtype (adaptive):
        # bf16 caller -> bf16 (== baseline, fits SBUF); fp32 caller (nki_moe.py) -> fp32
        # so the RMW load+add stays dtype-matched with HBM.
        result_tiles.append(
            sbm.alloc_stack(
                (B_TILE_SIZE, NUM_B_TILES, H_BLOCK_SIZE),
                dtype=hidden_states_grad.dtype,
                align=32,
            )
        )

        if is_tensor_update_accumulating:
            existing_hidden_grad.append(
                sbm.alloc_stack(
                    (B_TILE_SIZE, NUM_B_TILES, H_BLOCK_SIZE),
                    dtype=hidden_states_grad.dtype,
                    align=32,
                )
            )

    sbm.open_scope(interleave_degree=BUFFER_DEGREE, name=f"hidden_grad Buffer")

    for b_block_idx in range(NUM_B_BLOCKS):
        for h_block_idx in range(NUM_H_BLOCKS):
            # Compute valid H size for this block
            h_block_start = h_block_idx * H_BLOCK_SIZE
            valid_h_block = min(H_BLOCK_SIZE, H_DIM_SHARDED - h_block_start)

            result_buffer_idx = (b_block_idx * NUM_H_BLOCKS + h_block_idx) % BUFFER_DEGREE

            for gate_or_up in range(GATE_UP_WEIGHT_COUNT):
                for i_block_idx in range(NUM_I_TP_BLOCKS):
                    psum_buf_idx = (gate_or_up * NUM_I_TP_BLOCKS + i_block_idx) % BUFFER_DEGREE

                    # Compute valid I size for this block
                    i_block_start = i_block_idx * I_TP_BLOCK_SIZE
                    valid_i_block = min(I_TP_BLOCK_SIZE, I_TP_DIM - i_block_start)

                    lhs_tiles = sbm.alloc_stack(
                        (I_TP_TILE_SIZE, NUM_I_TP_TILES, B_BLOCK_SIZE),
                        dtype=compute_dtype,
                        align=32,
                        name=f"hidden_grad_lhs_blk{block_idx}_b{b_block_idx}_h{h_block_idx}_g{gate_or_up}_i{i_block_idx}",
                    )
                    rhs_tiles = sbm.alloc_stack(
                        (I_TP_TILE_SIZE, NUM_I_TP_TILES, H_BLOCK_SIZE),
                        dtype=compute_dtype,
                        name=f"hidden_grad_rhs_blk{block_idx}_b{b_block_idx}_h{h_block_idx}_g{gate_or_up}_i{i_block_idx}",
                        align=32,
                    )

                    # Load lhs_tiles from gate_up_proj_output_grad_hbm [B, 2, I_TP] with dma_transpose
                    if skip_gate_proj and gate_or_up == 0:
                        nisa.memset(lhs_tiles[:, :, :], value=0.0)
                    for i_tile_idx in range(NUM_I_TP_TILES):
                        if skip_gate_proj and gate_or_up == 0:
                            break
                        i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                        if i_tile_start >= valid_i_block:
                            break
                        valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)
                        i_offset = i_block_start + i_tile_start
                        nisa.dma_transpose(
                            dst=lhs_tiles.ap(
                                pattern=[
                                    [NUM_I_TP_TILES * B_BLOCK_SIZE, valid_i_tile],
                                    [1, 1],
                                    [1, 1],
                                    [1, B_BLOCK_SIZE],
                                ],
                                offset=i_tile_idx * B_BLOCK_SIZE,
                            ),
                            src=gate_up_proj_output_grad_hbm.ap(
                                pattern=[
                                    [GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_BLOCK_SIZE],
                                    [1, 1],
                                    [1, 1],
                                    [1, valid_i_tile],
                                ],
                                offset=(b_block_idx * B_BLOCK_SIZE) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                                + gate_or_up * I_TP_DIM
                                + i_offset,
                            ),
                        )

                    # Load rhs_tiles from gate_up_proj_weight [E, H, 2, I_TP] with bulk DMA + transpose
                    for h_tile in TiledRange(valid_h_block, H_TILE_SIZE):
                        rhs_temp_buffer_idx = h_tile.index % BUFFER_DEGREE
                        num_full_h_inner_tiles = h_tile.size // TILE_SIZE
                        partial_h_inner = h_tile.size % TILE_SIZE
                        h_abs_offset = h_block_start + h_tile.start_offset

                        # Bulk DMA: load all full TILE_SIZE inner tiles in a single instruction
                        if num_full_h_inner_tiles > 0:
                            nisa.dma_copy(
                                dst=rhs_temp[rhs_temp_buffer_idx][:, 0:num_full_h_inner_tiles, 0:valid_i_block],
                                src=gate_up_proj_weight.ap(
                                    pattern=[
                                        [GATE_UP_WEIGHT_COUNT * I_TP_DIM, TILE_SIZE],
                                        [GATE_UP_WEIGHT_COUNT * I_TP_DIM * TILE_SIZE, num_full_h_inner_tiles],
                                        [1, valid_i_block],
                                    ],
                                    offset=(H_SHARD_OFFSET + h_abs_offset) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                                    + gate_or_up * I_TP_DIM
                                    + i_block_start,
                                    scalar_offset=expert_idx,
                                    indirect_dim=0,
                                ),
                                dge_mode=dge_mode.hwdge,
                            )

                        # Handle partial last inner tile
                        if partial_h_inner > 0:
                            nisa.dma_copy(
                                dst=rhs_temp[rhs_temp_buffer_idx][
                                    0:partial_h_inner, num_full_h_inner_tiles, 0:valid_i_block
                                ],
                                src=gate_up_proj_weight.ap(
                                    pattern=[
                                        [GATE_UP_WEIGHT_COUNT * I_TP_DIM, partial_h_inner],
                                        [1, 1],
                                        [1, valid_i_block],
                                    ],
                                    offset=(H_SHARD_OFFSET + h_abs_offset + num_full_h_inner_tiles * TILE_SIZE)
                                    * GATE_UP_WEIGHT_COUNT
                                    * I_TP_DIM
                                    + gate_or_up * I_TP_DIM
                                    + i_block_start,
                                    scalar_offset=expert_idx,
                                    indirect_dim=0,
                                ),
                                dge_mode=dge_mode.hwdge,
                            )

                        # Transpose each inner tile from rhs_temp into rhs_tiles
                        for h_inner in TiledRange(h_tile, TILE_SIZE):
                            for i_tile_idx in range(NUM_I_TP_TILES):
                                i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                                if i_tile_start >= valid_i_block:
                                    break
                                valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)
                                transpose_psum_idx += 1
                                transpose_psum_idx = transpose_psum_idx % nl.tile_size.psum_num_banks
                                rhs_transposed_psum = nl.ndarray(
                                    (I_TP_TILE_SIZE, TILE_SIZE),
                                    dtype=compute_dtype,
                                    buffer=nl.psum,
                                    address=(0, transpose_psum_idx * PSUM_BANK_SIZE),
                                )
                                nisa.nc_transpose(
                                    dst=rhs_transposed_psum[0:valid_i_tile, 0 : h_inner.size],
                                    data=rhs_temp[rhs_temp_buffer_idx][
                                        0 : h_inner.size, h_inner.index, nl.ds(i_tile_start, valid_i_tile)
                                    ],
                                )
                                nisa.tensor_copy(
                                    dst=rhs_tiles[
                                        0:valid_i_tile, i_tile_idx, nl.ds(h_inner.start_offset, h_inner.size)
                                    ],
                                    src=rhs_transposed_psum[0:valid_i_tile, 0 : h_inner.size],
                                    engine=nisa.scalar_engine,
                                )

                    # Matmul: [B, I_TP] @ [I_TP, H] -> [B, H]
                    for b_tile_idx in range(NUM_B_TILES):
                        b_tile_start = b_tile_idx * B_TILE_SIZE
                        for h_tile_idx in range(NUM_H_TILES):
                            h_tile_start = h_tile_idx * H_TILE_SIZE
                            if h_tile_start >= valid_h_block:
                                break
                            valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)
                            matmul_psum_idx += 1
                            matmul_psum_idx = matmul_psum_idx % nl.tile_size.psum_num_banks
                            result_psum = nl.ndarray(
                                (B_TILE_SIZE, H_TILE_SIZE),
                                buffer=nl.psum,
                                dtype=nl.float32,
                                address=(0, matmul_psum_idx * PSUM_BANK_SIZE),
                            )
                            for i_tile_idx in range(NUM_I_TP_TILES):
                                i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                                if i_tile_start >= valid_i_block:
                                    break
                                valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)
                                nisa.nc_matmul(
                                    dst=result_psum[:, 0:valid_h_tile],
                                    stationary=lhs_tiles[0:valid_i_tile, i_tile_idx, nl.ds(b_tile_start, B_TILE_SIZE)],
                                    moving=rhs_tiles[0:valid_i_tile, i_tile_idx, nl.ds(h_tile_start, valid_h_tile)],
                                )
                            if i_block_idx == 0 and (gate_or_up == 0 or (skip_gate_proj and gate_or_up == 1)):
                                nisa.tensor_copy(
                                    dst=result_tiles[result_buffer_idx][
                                        :, b_tile_idx, nl.ds(h_tile_start, valid_h_tile)
                                    ],
                                    src=result_psum[:, 0:valid_h_tile],
                                )
                            else:
                                nisa.tensor_tensor(
                                    dst=result_tiles[result_buffer_idx][
                                        :, b_tile_idx, nl.ds(h_tile_start, valid_h_tile)
                                    ],
                                    op=nl.add,
                                    data1=result_tiles[result_buffer_idx][
                                        :, b_tile_idx, nl.ds(h_tile_start, valid_h_tile)
                                    ],
                                    data2=result_psum[:, 0:valid_h_tile],
                                )

                    sbm.increment_section()

            # Write hidden states grad
            for b_tile_idx in range(NUM_B_TILES):
                global_tile_idx = b_block_idx * NUM_B_TILES + b_tile_idx
                h_offset = h_block_idx * H_BLOCK_SIZE

                if is_tensor_update_accumulating:
                    if skip_dma.skip_token:
                        nisa.memset(existing_hidden_grad[result_buffer_idx], value=0.0)

                    nisa.dma_copy(
                        dst=existing_hidden_grad[result_buffer_idx][:, b_tile_idx, 0:valid_h_block],
                        src=hidden_states_grad.ap(
                            pattern=[[H_DIM, B_TILE_SIZE], [1, 1], [1, valid_h_block]],
                            offset=H_SHARD_OFFSET + h_offset,
                            vector_offset=block_token_pos_to_id_full.ap(
                                pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]],
                                offset=global_tile_idx,
                            ),
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                    )

                    nisa.tensor_tensor(
                        dst=result_tiles[result_buffer_idx][:, b_tile_idx, 0:valid_h_block],
                        op=nl.add,
                        data1=existing_hidden_grad[result_buffer_idx][:, b_tile_idx, 0:valid_h_block],
                        data2=result_tiles[result_buffer_idx][:, b_tile_idx, 0:valid_h_block],
                    )

                nisa.dma_copy(
                    dst=hidden_states_grad.ap(
                        pattern=[[H_DIM, B_TILE_SIZE], [1, 1], [1, valid_h_block]],
                        offset=H_SHARD_OFFSET + h_offset,
                        vector_offset=block_token_pos_to_id_full.ap(
                            pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]],
                            offset=global_tile_idx,
                        ),
                        indirect_dim=0,
                    ),
                    src=result_tiles[result_buffer_idx][:, b_tile_idx, 0:valid_h_block],
                    oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                )

    sbm.close_scope()
    if manage_scope:
        sbm.close_scope()


def _compute_gate_up_projection_weight_grad(
    gate_up_proj_output_grad_hbm,
    hidden_states,
    gate_up_proj_weight_grad,
    block_token_pos_to_id_full,
    shard_id,
    num_shards,
    expert_idx,
    skip_dma,
    compute_dtype,
    block_idx,
    sbm,
    BLOCK_H=1,
    BLOCK_B=2,
    BLOCK_I_TP=1,
    manage_scope=True,
    BUFFER_DEGREE=3,
    skip_gate_proj=False,
):
    """
    Compute gate and up projection weight gradient with H-dimension sharding.

    Performs matmul of hidden_states^T @ gate_up_proj_output_grad to compute
    the weight gradient. Each shard processes H/num_shards of the hidden dimension.

    Args:
        gate_up_proj_output_grad_hbm (nl.ndarray): [B, 2, I_TP], Gate/up projection gradients.
        hidden_states (nl.ndarray): [T, H], Input hidden states.
        gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Output weight gradient.
        block_token_pos_to_id_full (nl.ndarray): [B_TILE_SIZE, NUM_TILES], Token position mapping.
        shard_id (int): Current shard ID.
        num_shards (int): Number of LNC shards.
        expert_idx (nl.ndarray): [1, 1], Expert index for indirect addressing.
        skip_dma (SkipMode): Controls OOB handling for DMA operations.
        compute_dtype: Computation data type.
        block_idx (int): Current block index.
        sbm (SbufManager): SBUF memory manager.
        BLOCK_H (int): H dimension blocking factor.
        BLOCK_B (int): B dimension blocking factor.
        BLOCK_I_TP (int): I dimension blocking factor.
        skip_gate_proj (bool): Whether to omit the gate-projection contribution.

    Returns:
        None: Weight gradient is accumulated in-place.

    Notes:
        - weight_grad[expert, H_shard, gate_or_up, :] += hidden_states^T @ gate_up_grad
        - Each shard writes to its H/num_shards portion.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    B_DIM, GATE_UP_WEIGHT_COUNT, I_TP_DIM = gate_up_proj_output_grad_hbm.shape
    _, H_DIM = hidden_states.shape
    H_DIM_SHARDED = H_DIM // num_shards
    H_SHARD_OFFSET = H_DIM_SHARDED * shard_id

    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    H_TILE_SIZE = min(TILE_SIZE, H_DIM_SHARDED)
    I_TP_TILE_SIZE = min(PSUM_SIZE, I_TP_DIM)

    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM_SHARDED)
    I_TP_BLOCK_SIZE = min(BLOCK_I_TP * I_TP_TILE_SIZE, I_TP_DIM)
    B_BLOCK_SIZE = min(BLOCK_B * B_TILE_SIZE, B_DIM)

    NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
    NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
    NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

    NUM_I_TP_BLOCKS = div_ceil(I_TP_DIM, I_TP_BLOCK_SIZE)
    NUM_H_BLOCKS = div_ceil(H_DIM_SHARDED, H_BLOCK_SIZE)
    NUM_B_BLOCKS = div_ceil(B_DIM, B_BLOCK_SIZE)

    if manage_scope:
        sbm.open_scope(name=f"Gate Up Weight Grad")

    weight_grad_accum = []
    existing_weight_grad = []
    matmul_psum_idx = 0

    for n_buffer in range(BUFFER_DEGREE):
        # Weight-grad accumulators follow grad-output HBM dtype (adaptive): bf16 -> baseline
        # (fits SBUF); fp32 caller (nki_moe.py) -> fp32 full-precision cross-expert accum.
        weight_grad_accum.append(
            sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES, I_TP_BLOCK_SIZE), dtype=gate_up_proj_weight_grad.dtype, align=32)
        )
        existing_weight_grad.append(
            sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES, I_TP_BLOCK_SIZE), dtype=gate_up_proj_weight_grad.dtype, align=32)
        )
    sbm.open_scope(name=f"Gate Up Weight Grad Double Buffer", interleave_degree=BUFFER_DEGREE)
    # Process one (h_tile, i_tile) at a time to minimize memory
    for h_block_idx in range(NUM_H_BLOCKS):
        h_block_start = h_block_idx * H_BLOCK_SIZE
        valid_h_block = min(H_BLOCK_SIZE, H_DIM_SHARDED - h_block_start)

        for gate_or_up in range(GATE_UP_WEIGHT_COUNT):
            for i_block_idx in range(NUM_I_TP_BLOCKS):
                i_block_start = i_block_idx * I_TP_BLOCK_SIZE
                valid_i_block = min(I_TP_BLOCK_SIZE, I_TP_DIM - i_block_start)

                # Small accumulator: allocate OUTSIDE inner scope so it survives close_scope()

                result_buffer_idx = (
                    h_block_idx * GATE_UP_WEIGHT_COUNT + gate_or_up * NUM_I_TP_BLOCKS + i_block_idx
                ) % BUFFER_DEGREE
                # Wrap entire iteration in a scope so memory is freed
                for b_block_idx in range(NUM_B_BLOCKS):
                    gate_up_proj_output_grad = sbm.alloc_stack(
                        (B_TILE_SIZE, NUM_B_TILES, valid_i_block),
                        dtype=compute_dtype,
                        name=f"gup_wgrad_ograd_blk{block_idx}_h{h_block_idx}_g{gate_or_up}_i{i_block_idx}_b{b_block_idx}",
                        align=32,
                    )
                    block_hidden_states = sbm.alloc_stack(
                        (B_TILE_SIZE, NUM_B_TILES, valid_h_block),
                        dtype=compute_dtype,
                        name=f"gup_wgrad_hidden_blk{block_idx}_h{h_block_idx}_g{gate_or_up}_i{i_block_idx}_b{b_block_idx}",
                        align=32,
                    )
                    if skip_dma.skip_token:
                        nisa.memset(block_hidden_states[:, :, 0:valid_h_block], value=0.0)

                    if skip_gate_proj and gate_or_up == 0:
                        nisa.memset(gate_up_proj_output_grad[:, :, 0:valid_i_block], value=0.0)

                    for b_tile_idx in range(NUM_B_TILES):
                        # Load gate_up_proj_output_grad for this tile
                        global_tile_idx = b_block_idx * NUM_B_TILES + b_tile_idx
                        b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                        if not (skip_gate_proj and gate_or_up == 0):
                            nisa.dma_copy(
                                dst=gate_up_proj_output_grad[:, b_tile_idx, 0:valid_i_block],
                                src=gate_up_proj_output_grad_hbm.ap(
                                    pattern=[
                                        [GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_TILE_SIZE],
                                        [1, 1],
                                        [1, valid_i_block],
                                    ],
                                    offset=b_offset * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                                    + gate_or_up * I_TP_DIM
                                    + i_block_start,
                                ),
                            )

                        nisa.dma_copy(
                            dst=block_hidden_states[:, b_tile_idx, 0:valid_h_block],
                            src=hidden_states.ap(
                                pattern=[[H_DIM, B_TILE_SIZE], [1, 1], [1, valid_h_block]],
                                offset=H_SHARD_OFFSET + h_block_start,
                                vector_offset=block_token_pos_to_id_full.ap(
                                    pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]],
                                    offset=global_tile_idx,
                                ),
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                        )

                    for i_tile_idx in range(NUM_I_TP_TILES):
                        i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                        if i_tile_start >= valid_i_block:
                            break
                        valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)

                        for h_tile_idx in range(NUM_H_TILES):
                            h_tile_start = h_tile_idx * H_TILE_SIZE
                            if h_tile_start >= valid_h_block:
                                break
                            valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)
                            matmul_psum_idx += 1
                            matmul_psum_idx = matmul_psum_idx % nl.tile_size.psum_num_banks

                            result_psum = nl.ndarray(
                                (H_TILE_SIZE, I_TP_TILE_SIZE),
                                buffer=nl.psum,
                                dtype=nl.float32,
                                address=(0, matmul_psum_idx * PSUM_BANK_SIZE),
                            )
                            for b_tile_idx in range(NUM_B_TILES):
                                nisa.nc_matmul(
                                    dst=result_psum[0:valid_h_tile, 0:valid_i_tile],
                                    stationary=block_hidden_states[:, b_tile_idx, nl.ds(h_tile_start, valid_h_tile)],
                                    moving=gate_up_proj_output_grad[:, b_tile_idx, nl.ds(i_tile_start, valid_i_tile)],
                                )

                            if b_block_idx == 0:
                                nisa.tensor_copy(
                                    dst=weight_grad_accum[result_buffer_idx][
                                        0:valid_h_tile, h_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                    ],
                                    src=result_psum[0:valid_h_tile, 0:valid_i_tile],
                                    engine=nisa.scalar_engine,
                                )
                            else:
                                nisa.tensor_tensor(
                                    dst=weight_grad_accum[result_buffer_idx][
                                        0:valid_h_tile, h_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                    ],
                                    op=nl.add,
                                    data1=weight_grad_accum[result_buffer_idx][
                                        0:valid_h_tile, h_tile_idx, nl.ds(i_tile_start, valid_i_tile)
                                    ],
                                    data2=result_psum[0:valid_h_tile, 0:valid_i_tile],
                                )

                    sbm.increment_section()

                # Accumulate Gradients - load all H tiles in a single DMA
                num_full_h_tiles = valid_h_block // H_TILE_SIZE
                partial_h_tile = valid_h_block % H_TILE_SIZE

                # Bulk load/accumulate/store for all full tiles
                if num_full_h_tiles > 0:
                    nisa.dma_copy(
                        dst=existing_weight_grad[result_buffer_idx][:, 0:num_full_h_tiles, 0:valid_i_block],
                        src=gate_up_proj_weight_grad.ap(
                            pattern=[
                                [GATE_UP_WEIGHT_COUNT * I_TP_DIM, H_TILE_SIZE],
                                [GATE_UP_WEIGHT_COUNT * I_TP_DIM * H_TILE_SIZE, num_full_h_tiles],
                                [1, valid_i_block],
                            ],
                            offset=(H_SHARD_OFFSET + h_block_start) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                            + gate_or_up * I_TP_DIM
                            + i_block_start,
                            scalar_offset=expert_idx,
                            indirect_dim=0,
                        ),
                        dge_mode=dge_mode.hwdge,
                    )
                    nisa.tensor_tensor(
                        dst=existing_weight_grad[result_buffer_idx][:, 0:num_full_h_tiles, 0:valid_i_block],
                        op=nl.add,
                        data1=existing_weight_grad[result_buffer_idx][:, 0:num_full_h_tiles, 0:valid_i_block],
                        data2=weight_grad_accum[result_buffer_idx][:, 0:num_full_h_tiles, 0:valid_i_block],
                    )
                    nisa.dma_copy(
                        dst=gate_up_proj_weight_grad.ap(
                            pattern=[
                                [GATE_UP_WEIGHT_COUNT * I_TP_DIM, H_TILE_SIZE],
                                [GATE_UP_WEIGHT_COUNT * I_TP_DIM * H_TILE_SIZE, num_full_h_tiles],
                                [1, valid_i_block],
                            ],
                            offset=(H_SHARD_OFFSET + h_block_start) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                            + gate_or_up * I_TP_DIM
                            + i_block_start,
                            scalar_offset=expert_idx,
                            indirect_dim=0,
                        ),
                        src=existing_weight_grad[result_buffer_idx][:, 0:num_full_h_tiles, 0:valid_i_block],
                        dge_mode=dge_mode.hwdge,
                    )

                # Handle partial last tile
                if partial_h_tile > 0:
                    last_tile_idx = num_full_h_tiles
                    h_tile_start = last_tile_idx * H_TILE_SIZE
                    nisa.dma_copy(
                        dst=existing_weight_grad[result_buffer_idx][0:partial_h_tile, last_tile_idx, 0:valid_i_block],
                        src=gate_up_proj_weight_grad.ap(
                            pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, partial_h_tile], [1, 1], [1, valid_i_block]],
                            offset=(H_SHARD_OFFSET + h_block_start + h_tile_start) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                            + gate_or_up * I_TP_DIM
                            + i_block_start,
                            scalar_offset=expert_idx,
                            indirect_dim=0,
                        ),
                        dge_mode=dge_mode.hwdge,
                    )
                    nisa.tensor_tensor(
                        dst=existing_weight_grad[result_buffer_idx][0:partial_h_tile, last_tile_idx, 0:valid_i_block],
                        op=nl.add,
                        data1=existing_weight_grad[result_buffer_idx][0:partial_h_tile, last_tile_idx, 0:valid_i_block],
                        data2=weight_grad_accum[result_buffer_idx][0:partial_h_tile, last_tile_idx, 0:valid_i_block],
                    )
                    nisa.dma_copy(
                        dst=gate_up_proj_weight_grad.ap(
                            pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, partial_h_tile], [1, 1], [1, valid_i_block]],
                            offset=(H_SHARD_OFFSET + h_block_start + h_tile_start) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                            + gate_or_up * I_TP_DIM
                            + i_block_start,
                            scalar_offset=expert_idx,
                            indirect_dim=0,
                        ),
                        src=existing_weight_grad[result_buffer_idx][0:partial_h_tile, last_tile_idx, 0:valid_i_block],
                        dge_mode=dge_mode.hwdge,
                    )

    sbm.close_scope()
    if manage_scope:
        sbm.close_scope()


def _load_token_indices(token_position_to_id, block_idx, B, NUM_TILES, sbm, dst=None, temporary_buffers=None):
    """
    Load and transpose token indices for the current block.

    Loads token position to ID mapping from HBM and transposes it for efficient
    indirect addressing in subsequent DMA operations.

    Args:
        token_position_to_id (nl.ndarray): [N * B], Token position to block index mapping.
        block_idx (int): Current block index.
        B (int): Block size (number of tokens per block).
        NUM_TILES (int): Number of tiles in the block.
        sbm (SbufManager): SBUF memory manager.
        dst (nl.ndarray, optional): Pre-allocated [TILE_SIZE, NUM_TILES] buffer. If None, allocates from sbm.
        temporary_buffers (list, optional): Pre-allocated list of NUM_TILES [1, TILE_SIZE] fp32 buffers.
            If None, allocates from sbm. Pre-allocating avoids address overlap with subsequent stages
            which would otherwise cause anti-dependencies (WAR) that serialize the prefetch.

    Returns:
        nl.ndarray: [TILE_SIZE, NUM_TILES], Transposed token indices in SBUF.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    if dst is None:
        token_position_to_id_sbuf = sbm.alloc_stack(
            (TILE_SIZE, NUM_TILES),
            dtype=nl.int32,
            name=f"token_position_to_id_sbuf_{block_idx}",
            align=32,
        )
    else:
        token_position_to_id_sbuf = dst

    sbm.open_scope(name="_load_token_indices", interleave_degree=2)
    transpose_psum_idx = 0
    for tile_idx in range(NUM_TILES):
        if temporary_buffers is None:
            token_pos_to_id_fp32 = sbm.alloc_stack(
                (1, TILE_SIZE),
                dtype=nl.float32,
                name=f"token_pos_to_id_fp32_{block_idx}_{tile_idx}",
                align=32,
            )
        else:
            token_pos_to_id_fp32 = temporary_buffers[tile_idx]
        offset = block_idx * B + TILE_SIZE * tile_idx

        nisa.dma_copy(
            token_pos_to_id_fp32.ap(pattern=[[TILE_SIZE, 1], [1, TILE_SIZE]]),
            token_position_to_id.ap(pattern=[[TILE_SIZE, 1], [1, TILE_SIZE]], offset=offset),
        )
        transpose_psum_idx += 1

        transpose_psum_idx = transpose_psum_idx % nl.tile_size.psum_num_banks
        transposed_token_pos_to_id_fp32 = nl.ndarray(
            (TILE_SIZE, 1),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, transpose_psum_idx * PSUM_BANK_SIZE),
            name=f"transposed_token_pos_to_id_fp32_{block_idx}_{tile_idx}",
        )
        nisa.nc_transpose(dst=transposed_token_pos_to_id_fp32, data=token_pos_to_id_fp32)
        nisa.tensor_copy(token_position_to_id_sbuf[:, tile_idx], transposed_token_pos_to_id_fp32)

        sbm.increment_section()

    sbm.close_scope()

    return token_position_to_id_sbuf


def _compute_function_sbuf_budgets(params, num_shards):
    """
    Compute estimated peak SBUF usage for each compute function.

    Returns list of 5 ints (bytes) in function execution order:
      [0] down_proj_output_grad (AFFINITY_ON_H) or gate_up_output_grad (AFFINITY_ON_I)
      [1] gate_up_output_grad (AFFINITY_ON_H only, 0 for AFFINITY_ON_I)
      [2] hidden_states_grad
      [3] gate_up_weight_grad
      [4] down_projection_weight_grad

    Args:
        params (MOEBwdParameters): Kernel parameters.
        num_shards (int): Number of LNC shards.

    Returns:
        list[int]: Per-function peak SBUF bytes.
    """
    bp = params.blocking_params
    B = params.block_size
    H = params.H
    I_TP = params.I_TP
    dtype = params.compute_dtype
    # Accumulator tiles in the bwd kernel follow the grad-output HBM dtype (fp32 when the caller
    # preallocates fp32 grad buffers); thread it so scope-merge budgeting sizes them correctly.
    hidden_grad_dtype = params.hidden_states_grad.dtype
    gate_up_wgrad_dtype = params.gate_up_proj_weight_grad.dtype
    down_wgrad_dtype = params.down_proj_weight_grad.dtype
    ao = params.affinity_option
    so = params.shard_option

    budgets = [0, 0, 0, 0, 0]
    if ao == AffinityOption.AFFINITY_ON_I:
        budgets[0] = bp.gate_up_output_grad.estimate_sbuf_usage(
            B, H, I_TP, num_shards, dtype, shard_option=so, affinity_option=ao, grad_accum_dtype=hidden_grad_dtype
        )
    else:
        budgets[0] = bp.down_proj_output_grad.estimate_sbuf_usage(
            B, H, I_TP, num_shards, dtype, shard_option=so, affinity_option=ao
        )
        budgets[1] = bp.gate_up_output_grad.estimate_sbuf_usage(
            B, H, I_TP, num_shards, dtype, shard_option=so, affinity_option=ao, grad_accum_dtype=hidden_grad_dtype
        )

    budgets[2] = bp.hidden_grad.estimate_sbuf_usage(
        B,
        H,
        I_TP,
        num_shards,
        dtype,
        is_tensor_update_accumulating=params.is_tensor_update_accumulating,
        grad_accum_dtype=hidden_grad_dtype,
    )
    budgets[3] = bp.gate_up_weight_grad.estimate_sbuf_usage(
        B, H, I_TP, num_shards, dtype, grad_accum_dtype=gate_up_wgrad_dtype
    )
    budgets[4] = bp.down_weight_grad.estimate_sbuf_usage(
        B, H, I_TP, num_shards, dtype, grad_accum_dtype=down_wgrad_dtype
    )

    return budgets


def _group_functions_by_sbuf_budget(budgets, available_sbuf):
    """
    Greedily group consecutive functions whose combined SBUF fits in available space.

    Returns per-function flags as 5 lists:
      - manage_scope: True if function manages its own scope, False if in merged group
      - open_merged_before: True if a merged scope should open before this function
      - close_merged_after: True if a merged scope should close after this function
      - increment_after: True if increment_section() should be called after this function
      - group_size_at_open: interleave_degree for the merged scope (only meaningful when open_merged_before is True)

    Args:
        budgets (list[int]): Per-function peak SBUF bytes (length 5).
        available_sbuf (int): Total available SBUF bytes.

    Returns:
        tuple: (manage_scope, open_merged_before, close_merged_after, increment_after, group_size_at_open)
            Each is a list of 5 booleans/ints.
    """
    # Build groups: track start index, end index (exclusive), and active count per group
    group_starts = [0, 0, 0, 0, 0]
    group_counts = [0, 0, 0, 0, 0]

    current_start = -1
    current_end = -1
    current_count = 0
    running_total = 0

    for func_idx in range(5):
        usage = budgets[func_idx]
        if usage == 0:
            continue
        if current_start == -1 or running_total + usage > available_sbuf:
            # Start a new group — finalize previous first
            if current_start != -1:
                for g_idx in range(current_start, current_end + 1):
                    if budgets[g_idx] > 0:
                        group_starts[g_idx] = current_start
                        group_counts[g_idx] = current_count
            current_start = func_idx
            current_end = func_idx
            current_count = 1
            running_total = usage
        else:
            current_end = func_idx
            current_count = current_count + 1
            running_total = running_total + usage

    # Finalize last group
    if current_start != -1:
        for g_idx in range(current_start, current_end + 1):
            if budgets[g_idx] > 0:
                group_starts[g_idx] = current_start
                group_counts[g_idx] = current_count

    # Derive per-function flags
    manage_scope = [True, True, True, True, True]
    open_merged_before = [False, False, False, False, False]
    close_merged_after = [False, False, False, False, False]
    increment_after = [False, False, False, False, False]
    group_size_at_open = [1, 1, 1, 1, 1]

    for func_idx in range(5):
        if budgets[func_idx] == 0:
            continue
        count = group_counts[func_idx]
        start = group_starts[func_idx]
        if count <= 1:
            manage_scope[func_idx] = True
        else:
            manage_scope[func_idx] = False
            if func_idx == start:
                open_merged_before[func_idx] = True
                group_size_at_open[func_idx] = count
            # Find if this is the last active function in the group
            is_last = True
            for check_idx in range(func_idx + 1, 5):
                if budgets[check_idx] > 0 and group_starts[check_idx] == start:
                    is_last = False
                    break
            if is_last:
                close_merged_after[func_idx] = True
            else:
                increment_after[func_idx] = True

    return manage_scope, open_merged_before, close_merged_after, increment_after, group_size_at_open


def blockwise_mm_bwd_dropless(params: "MOEBwdParameters"):
    """
    Backward pass for blockwise matrix multiplication in dropless Mixture of Experts.

    Computes gradients for hidden states, gate/up projection weights, down projection weights,
    expert affinities, and optional biases. Uses LNC2 sharding to distribute computation across
    cores. Processes tokens in blocks assigned to specific experts.

    Dimensions:
        T: Total number of input tokens (after linearizing across batch dimension)
        H: Hidden dimension size
        I_TP: Intermediate size / tensor parallel degree
        E: Number of experts
        B: Number of tokens per block (block_size)
        N: Total number of blocks (T / B)

    Args:
        params (MOEBwdParameters): All input tensors, output gradients, and configuration.

    Returns:
        None: Gradients are written in-place to the provided gradient tensors.

    Notes:
        - Uses LNC2 sharding: H dimension sharded for hidden_states_grad, down_proj_weight_grad;
          I dimension sharded for gate_up_proj_output_grad computation.
        - block_size must be one of: 128, 256, 512, 1024.
        - All gradient tensors are initialized to zero before accumulation.

    Pseudocode:
        _initialize_gradient_outputs_shard(...)

        # Bulk-load all expert indices into SBUF
        expert_idx_bufs = alloc (1, N) int32
        dma_copy(expert_idx_bufs, block_to_expert)

        # Double-buffer token indices
        token_indices_bufs = [alloc (TILE_SIZE, NUM_B_TILES) int32,
                              alloc (TILE_SIZE, NUM_B_TILES) int32]
        _load_token_indices(token_position_to_id, 0, dst=token_indices_bufs[0])

        for block_idx in range(N):
            expert_idx = expert_idx_bufs[0, block_idx]
            block_token_pos_to_id = token_indices_bufs[block_idx % 2]

            # Step 1: Compute down projection output gradient and expert affinity gradient
            down_proj_output_grad = _compute_down_projection_output_grad(...)

            # Prefetch next block's token indices (overlaps with stages 2-5)
            if block_idx < N - 1:
                nxt = (block_idx + 1) % 2
                _load_token_indices(token_position_to_id, block_idx + 1, dst=token_indices_bufs[nxt])

            # Step 2: Compute gate/up projection output gradient (backward through activation)
            gate_up_proj_output_grad = _compute_gate_up_projection_output_grad(...)

            # Step 3: Compute down projection weight gradient
            _compute_down_projection_weight_grad(...)

            # Step 4: Compute hidden states gradient
            _compute_hidden_states_grad(...)

            # Step 5: Compute gate/up projection weight gradient
            _compute_gate_up_projection_weight_grad(...)
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    # Validate parameters
    params.validate()

    # Extract from params for convenience
    B = params.block_size
    E = params.E
    N = params.N
    NUM_B_TILES = div_ceil(B, TILE_SIZE)

    # Get activation function type for forward/backward dispatch
    activation_type = params.get_activation_ops()

    _, num_shards, shard_id = get_program_sharding_info()
    params.validate_sharding(num_shards)

    sbm = SbufManager(0, MAX_AVAILABLE_SBUF_SIZE, logger=get_logger("SBM"))
    if not params.skip_grad_initialization:
        _initialize_gradient_outputs_shard(
            hidden_states_grad=params.hidden_states_grad,
            gate_up_proj_weight_grad=params.gate_up_proj_weight_grad,
            down_proj_weight_grad=params.down_proj_weight_grad,
            num_shards=num_shards,
            shard_id=shard_id,
            down_proj_bias_grad=params.down_proj_bias_grad,
            gate_and_up_proj_bias_grad=params.gate_and_up_proj_bias_grad,
            expert_affinities_masked_grad=params.expert_affinities_masked_grad,
            sbm=sbm,
        )

    # Get blocking params
    bp = params.blocking_params

    # Grad accumulators follow their grad-output HBM dtype (bf16 default == baseline; fp32 when the
    # caller preallocates fp32 grad buffers). Hoisted into named locals (mirrors __estimate_sbuf__)
    # so the per-tile dtype reads below stay readable.
    hidden_grad_dtype = params.hidden_states_grad.dtype
    gate_up_wgrad_dtype = params.gate_up_proj_weight_grad.dtype
    down_wgrad_dtype = params.down_proj_weight_grad.dtype

    # Open scope for buffers that persist across all block iterations
    sbm.open_scope(name="Block loop prefetch")

    # Bulk-load all expert indices to avoid DMA queue ordering issues
    # between prefetch (qSPIO0) and HWDGE bias grad writes (qSPDynamicHW).
    expert_idx_bufs = sbm.alloc_stack((1, N), dtype=nl.int32, buffer=nl.sbuf, name="expert_idx_bufs", align=32)
    block_to_expert_2d = params.block_to_expert.reshape((1, N))
    nisa.dma_copy(expert_idx_bufs[0, 0:N], block_to_expert_2d[0, 0:N])

    # Double-buffer token indices for prefetching
    token_indices_bufs = [
        sbm.alloc_stack((TILE_SIZE, NUM_B_TILES), dtype=nl.int32, name="token_indices_buf_0", align=32),
        sbm.alloc_stack((TILE_SIZE, NUM_B_TILES), dtype=nl.int32, name="token_indices_buf_1", align=32),
    ]

    # Precompute iota vector [0, 1, 2, ..., TILE_SIZE-1] once for reuse by all
    # dma_compute vec_off constructions (bias grad + weight grad).
    iota_vec = sbm.alloc_stack((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf, name="iota_vec", align=32)
    nisa.iota(dst=iota_vec, pattern=[[0, 1]], offset=0, channel_multiplier=1)

    # Broadcast all expert indices from (1, N) to (TILE_SIZE, N) in one shot.
    # This eliminates per-block stream_shuffle_broadcast calls in bias grad functions.
    expert_idx_broadcast = sbm.alloc_stack(
        (TILE_SIZE, N), dtype=nl.int32, buffer=nl.sbuf, name="expert_idx_broadcast", align=32
    )
    stream_shuffle_broadcast(src=expert_idx_bufs, dst=expert_idx_broadcast)

    # Prefetch block 0 token indices before the loop
    _load_token_indices(params.token_position_to_id, 0, B, NUM_B_TILES, sbm, dst=token_indices_bufs[0])

    # Compute SBUF budgets and group functions for scope merging.
    # For memory-bound configs with small shapes, consecutive functions can share
    # a single SBUF scope so the compiler can overlap DMA across function boundaries.
    sbuf_budgets = _compute_function_sbuf_budgets(params, num_shards)
    persistent_overhead = sbm.get_used_space()
    # Subtract per-block overhead: prefetch_temporary_buffers (NUM_B_TILES × 512 B)
    # are allocated inside each block scope before the compute functions run.
    per_block_overhead = NUM_B_TILES * align_to(TILE_SIZE * sizeinbytes(nl.float32), 32)
    available_sbuf = MAX_AVAILABLE_SBUF_SIZE - persistent_overhead - per_block_overhead
    ms_flags, open_before, close_after, inc_after, grp_sizes = _group_functions_by_sbuf_budget(
        sbuf_budgets, available_sbuf
    )

    # Log SBUF scope merging decisions (print directly — NKI compiler may not
    # execute logger.debug() calls outside of NKIObject methods)
    if sbm.logger._should_log(LogLevel.DEBUG):
        sbm.logger._print(
            f"SBUF scope merging: available={available_sbuf} B, persistent_overhead={persistent_overhead} B, "
            f"budgets=[{sbuf_budgets[0]}, {sbuf_budgets[1]}, {sbuf_budgets[2]}, {sbuf_budgets[3]}, {sbuf_budgets[4]}], "
            f"manage_scope=[{ms_flags[0]}, {ms_flags[1]}, {ms_flags[2]}, {ms_flags[3]}, {ms_flags[4]}], "
            f"open=[{open_before[0]}, {open_before[1]}, {open_before[2]}, {open_before[3]}, {open_before[4]}], "
            f"close=[{close_after[0]}, {close_after[1]}, {close_after[2]}, {close_after[3]}, {close_after[4]}], "
            f"inc=[{inc_after[0]}, {inc_after[1]}, {inc_after[2]}, {inc_after[3]}, {inc_after[4]}]",
            prefix="[DEBUG] ",
        )

    for block_idx in range(N):
        sbm.open_scope(name=f"Block {block_idx}")
        expert_idx = expert_idx_bufs[0, block_idx]
        cur = block_idx % 2
        block_token_pos_to_id_full = token_indices_bufs[cur]

        # Pre-allocate temp buffers for next block's token-indices prefetch so its
        # addresses don't overlap with later stages' tensors (avoids WAR anti-deps).
        prefetch_temporary_buffers = []
        for t in range(NUM_B_TILES):
            prefetch_temporary_buffers.append(
                sbm.alloc_stack((1, TILE_SIZE), dtype=nl.float32, name=f"prefetch_tmp_{block_idx}_{t}", align=32)
            )

        # --- Function 0: down_proj_output_grad (AFFINITY_ON_H) or gate_up_output_grad (AFFINITY_ON_I) ---
        if open_before[0]:
            sbm.open_scope(name="merged_0", interleave_degree=grp_sizes[0])

        if params.affinity_option == AffinityOption.AFFINITY_ON_I:
            down_proj_output_grad_hbm = params.output_hidden_states_grad
            gate_up_proj_output_grad_hbm, gate_up_multipy_output_hbm = _compute_gate_up_projection_output_grad(
                down_proj_output_grad_hbm=params.output_hidden_states_grad,
                down_projection_weight=params.down_proj_weight,
                gate_up_proj_act_checkpoint_T=params.gate_up_proj_act_checkpoint_T,
                shard_id=shard_id,
                num_shards=num_shards,
                block_idx=block_idx,
                expert_idx=expert_idx,
                compute_dtype=params.compute_dtype,
                grad_accum_dtype=hidden_grad_dtype,
                activation_type=activation_type,
                skip_gate_proj=params.skip_gate_proj,
                sbm=sbm,
                clamp_limits=params.clamp_limits,
                BLOCK_H=bp.gate_up_output_grad.block_h,
                BLOCK_B=bp.gate_up_output_grad.block_b,
                BLOCK_I_TP=bp.gate_up_output_grad.block_i,
                affinity_option=params.affinity_option,
                block_token_pos_to_id_full=block_token_pos_to_id_full,
                expert_affinities_masked=params.expert_affinities_masked,
                expert_affinities_masked_grad=params.expert_affinities_masked_grad,
                E=E,
                skip_dma=params.skip_dma,
                expert_idx_broadcast=expert_idx_broadcast,
                shard_option=params.shard_option,
                manage_scope=ms_flags[0],
                BUFFER_DEGREE=bp.gate_up_output_grad.buffer_degree,
            )
        else:
            down_proj_output_grad_hbm = _compute_down_projection_output_grad(
                params.output_hidden_states_grad,
                block_token_pos_to_id_full,
                params.expert_affinities_masked,
                params.expert_affinities_masked_grad,
                params.down_proj_act_checkpoint,
                block_idx,
                params.skip_dma,
                expert_idx,
                E,
                params.compute_dtype,
                num_shards,
                shard_id,
                sbm,
                BLOCK_H=bp.down_proj_output_grad.block_h,
                expert_idx_broadcast=expert_idx_broadcast,
                manage_scope=ms_flags[0],
                BUFFER_DEGREE=bp.down_proj_output_grad.buffer_degree,
            )

        if inc_after[0]:
            sbm.increment_section()
        if close_after[0]:
            sbm.close_scope()

        # --- Function 1: gate_up_output_grad (AFFINITY_ON_H only) ---
        if params.affinity_option != AffinityOption.AFFINITY_ON_I:
            if open_before[1]:
                sbm.open_scope(name="merged_1", interleave_degree=grp_sizes[1])

            gate_up_proj_output_grad_hbm, gate_up_multipy_output_hbm = _compute_gate_up_projection_output_grad(
                down_proj_output_grad_hbm=down_proj_output_grad_hbm,
                down_projection_weight=params.down_proj_weight,
                gate_up_proj_act_checkpoint_T=params.gate_up_proj_act_checkpoint_T,
                shard_id=shard_id,
                num_shards=num_shards,
                block_idx=block_idx,
                expert_idx=expert_idx,
                compute_dtype=params.compute_dtype,
                grad_accum_dtype=hidden_grad_dtype,
                activation_type=activation_type,
                skip_gate_proj=params.skip_gate_proj,
                sbm=sbm,
                clamp_limits=params.clamp_limits,
                BLOCK_H=bp.gate_up_output_grad.block_h,
                BLOCK_B=bp.gate_up_output_grad.block_b,
                BLOCK_I_TP=bp.gate_up_output_grad.block_i,
                expert_idx_broadcast=expert_idx_broadcast,
                shard_option=params.shard_option,
                manage_scope=ms_flags[1],
                BUFFER_DEGREE=bp.gate_up_output_grad.buffer_degree,
            )

            if inc_after[1]:
                sbm.increment_section()
            if close_after[1]:
                sbm.close_scope()

        # Prefetch next block's token_indices (overlaps with remaining functions)
        if block_idx < N - 1:
            nxt = (block_idx + 1) % 2
            _load_token_indices(
                params.token_position_to_id,
                block_idx + 1,
                B,
                NUM_B_TILES,
                sbm,
                dst=token_indices_bufs[nxt],
                temporary_buffers=prefetch_temporary_buffers,
            )

        # --- Function 2: hidden_states_grad ---
        if open_before[2]:
            sbm.open_scope(name="merged_2", interleave_degree=grp_sizes[2])

        _compute_hidden_states_grad(
            gate_up_proj_output_grad_hbm,
            params.gate_up_proj_weight,
            params.hidden_states_grad,
            block_token_pos_to_id_full,
            shard_id,
            num_shards,
            expert_idx,
            params.skip_dma,
            params.compute_dtype,
            params.is_tensor_update_accumulating,
            block_idx,
            sbm=sbm,
            BLOCK_H=bp.hidden_grad.get_block_h(hidden_grad_dtype),
            BLOCK_B=bp.hidden_grad.block_b,
            BLOCK_I_TP=bp.hidden_grad.block_i,
            manage_scope=ms_flags[2],
            BUFFER_DEGREE=bp.hidden_grad.get_buffer_degree(hidden_grad_dtype),
            skip_gate_proj=params.skip_gate_proj,
        )

        if inc_after[2]:
            sbm.increment_section()
        if close_after[2]:
            sbm.close_scope()

        # --- Function 3: gate_up_weight_grad ---
        if open_before[3]:
            sbm.open_scope(name="merged_3", interleave_degree=grp_sizes[3])

        _compute_gate_up_projection_weight_grad(
            gate_up_proj_output_grad_hbm,
            params.hidden_states,
            params.gate_up_proj_weight_grad,
            block_token_pos_to_id_full,
            shard_id,
            num_shards,
            expert_idx,
            params.skip_dma,
            params.compute_dtype,
            block_idx,
            sbm=sbm,
            BLOCK_H=bp.gate_up_weight_grad.block_h,
            BLOCK_B=bp.gate_up_weight_grad.block_b,
            BLOCK_I_TP=bp.gate_up_weight_grad.block_i,
            manage_scope=ms_flags[3],
            BUFFER_DEGREE=bp.gate_up_weight_grad.get_buffer_degree(gate_up_wgrad_dtype),
            skip_gate_proj=params.skip_gate_proj,
        )

        if inc_after[3]:
            sbm.increment_section()
        if close_after[3]:
            sbm.close_scope()

        # --- Function 4: down_projection_weight_grad ---
        if open_before[4]:
            sbm.open_scope(name="merged_4", interleave_degree=grp_sizes[4])

        _compute_down_projection_weight_grad(
            gate_up_multipy_output_hbm,
            down_proj_output_grad_hbm,
            params.down_proj_weight_grad,
            shard_id,
            num_shards,
            expert_idx,
            params.compute_dtype,
            block_idx,
            sbm=sbm,
            BLOCK_H=bp.down_weight_grad.get_block_h(down_wgrad_dtype),
            BLOCK_B=bp.down_weight_grad.block_b,
            BLOCK_I_TP=bp.down_weight_grad.block_i,
            block_token_pos_to_id_full=block_token_pos_to_id_full
            if params.affinity_option == AffinityOption.AFFINITY_ON_I
            else None,
            skip_dma=params.skip_dma if params.affinity_option == AffinityOption.AFFINITY_ON_I else None,
            iota_vec=iota_vec,
            manage_scope=ms_flags[4],
            BUFFER_DEGREE=bp.down_weight_grad.get_buffer_degree(down_wgrad_dtype),
        )

        if inc_after[4]:
            sbm.increment_section()
        if close_after[4]:
            sbm.close_scope()

        # Compute bias gradients (outside merged groups)
        if params.down_proj_bias_grad != None:
            _compute_down_proj_bias_grad(
                down_proj_output_grad_hbm=down_proj_output_grad_hbm,
                down_proj_bias_grad=params.down_proj_bias_grad,
                expert_idx=expert_idx,
                B_DIM=B,
                H_DIM=params.H,
                num_shards=num_shards,
                shard_id=shard_id,
                dtype=params.compute_dtype,
                sbm=sbm,
                block_token_pos_to_id_full=block_token_pos_to_id_full
                if params.affinity_option == AffinityOption.AFFINITY_ON_I
                else None,
                skip_dma=params.skip_dma if params.affinity_option == AffinityOption.AFFINITY_ON_I else None,
                block_idx=block_idx,
                iota_vec=iota_vec,
                expert_idx_broadcast=expert_idx_broadcast,
                expert_affinities_masked=params.expert_affinities_masked,
                E=E,
                accumulation_dtype=params.accumulation_dtype,
            )

        if params.gate_and_up_proj_bias_grad != None:
            _compute_gate_up_proj_bias_grad(
                gate_up_proj_output_grad_hbm=gate_up_proj_output_grad_hbm,
                gate_and_up_proj_bias_grad=params.gate_and_up_proj_bias_grad,
                expert_idx=expert_idx,
                B=B,
                I_TP=params.I_TP,
                shard_id=shard_id,
                dtype=params.compute_dtype,
                sbm=sbm,
                block_idx=block_idx,
                iota_vec=iota_vec,
                expert_idx_broadcast=expert_idx_broadcast,
                accumulation_dtype=params.accumulation_dtype,
            )

        sbm.close_scope()
