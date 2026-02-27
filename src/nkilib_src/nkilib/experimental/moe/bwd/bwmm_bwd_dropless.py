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

from ....core.utils.allocator import SbufManager
from ....core.utils.kernel_helpers import (
    NUM_HW_PSUM_BANKS,
    PSUM_BANK_SIZE,
    div_ceil,
    get_program_sharding_info,
)
from ....core.utils.logging import get_logger
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .moe_bwd_parameters import ClampLimits, MOEBwdParameters

MAX_AVAILABLE_SBUF_SIZE = 224 * 1024 - 16384 - 8 - 520


def _load_block_expert(block_to_expert, block_idx, sbm):
    """
    Load expert index for the current block from HBM into SBUF.

    Args:
        block_to_expert (nl.ndarray): [N, 1], Expert indices for each block on HBM.
        block_idx (int): Current block index.
        sbm (SbufManager): SBUF memory manager.

    Returns:
        nl.ndarray: [1, 1], Expert index tensor in SBUF for indirect addressing.
    """
    expert_idx_tensor = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"expert_idx_{block_idx}")
    nisa.dma_copy(expert_idx_tensor.ap([[1, 1], [1, 1]]), block_to_expert.ap([[1, 1], [1, 1]], offset=block_idx))
    return expert_idx_tensor


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
    zeros = sbm.alloc_stack(
        (T_TILE_SIZE, 1), dtype=expert_affinities_masked_grad.dtype, name="expert_affinities_masked_grad_zeros"
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

    H_SHARD_OFFSET = H_PER_SHARD * shard_id
    zeros = sbm.alloc_stack(
        (H_TILE_SIZE, GATE_OR_UP_WEIGHT_COUNT, I_TP),
        dtype=gate_up_proj_weight_grad.dtype,
        name="gate_up_proj_weight_grad_zeros",
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

    zeros = sbm.alloc_stack((I_TILE_SIZE, H_PER_SHARD), dtype=down_proj_weight_grad.dtype)
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
    zeros = sbm.alloc_stack((T_TILE_SIZE, H_PER_SHARD), dtype=hidden_states_grad.dtype, name="hidden_states_grad_zeros")
    nisa.memset(zeros, value=0.0)
    for t_tile_idx in range(NUM_T_TILES):
        # Calculate element count for mask replacement
        valid_t = min(T_TILE_SIZE, T - t_tile_idx * T_TILE_SIZE)

        nisa.dma_copy(
            dst=hidden_states_grad[nl.ds(t_tile_idx * T_TILE_SIZE, valid_t), nl.ds(H_SHARD_OFFSET, H_PER_SHARD)],
            src=zeros[0:valid_t, 0:H_PER_SHARD],
        )

    if down_proj_bias_grad:
        zeros = sbm.alloc_stack((H_TILE_SIZE, 1), dtype=down_proj_bias_grad.dtype, name="down_proj_bias_grad_zeros")
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
            (I_TILE_SIZE, 1), dtype=gate_and_up_proj_bias_grad.dtype, name="gate_and_up_proj_bias_grad_zeros"
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
        B_TILE_SIZE (int): Tile size for batch dimension.
        b_tile_idx (int): Current batch tile index.
        skip_dma (SkipMode): Controls OOB handling.
        E (int): Number of experts.
        sbm (SbufManager): SBUF memory manager.
        call_id (str): Identifier for debugging.

    Returns:
        nl.ndarray: [B_TILE_SIZE, 1], Computed token indices offset for indirect addressing.
    """
    nisa.tensor_scalar(dst=addr, data=block_token_pos_to_id[:, b_tile_idx], op0=nl.multiply, operand0=E)
    nisa.tensor_tensor(dst=token_indices_offset, data1=addr, op=nl.add, data2=expert_idx_tensor)

    if skip_dma.skip_token:
        nisa.tensor_scalar(dst=token_indices_offset, data=token_indices_offset, op0=nl.maximum, operand0=-1)

    return token_indices_offset


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
):
    """
    Compute down projection bias gradient with H-dimension sharding.

    Reduces down_proj_output_grad [B, H] over B dimension to get bias grad [1, H].
    Each shard computes and writes H/num_shards of the gradient.

    Args:
      down_proj_output_grad_hbm: Gradient tensor [B, H] in HBM
      down_proj_bias_grad: Output bias gradient [E, H] in HBM
      expert_idx: Expert index tensor for indirect addressing
      B: Batch/block size
      H: Hidden dimension
      num_shards: Number of LNC shards
      shard_id: Current shard ID
      dtype: Data type
      sbm: SBUF memory manager
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax

    H_PER_SHARD = H_DIM // num_shards
    H_SHARD_OFFSET = H_PER_SHARD * shard_id
    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    NUM_B_TILES = div_ceil(B_DIM, B_TILE_SIZE)
    H_TILE_SIZE = min(TILE_SIZE, H_PER_SHARD)
    NUM_H_TILES = div_ceil(H_PER_SHARD, H_TILE_SIZE)

    sbm.open_scope(name="down_proj_bias_grad")

    # Allocate buffers
    grad_tiles = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES, B_TILE_SIZE), dtype=dtype, buffer=nl.sbuf, align=32)
    reduced = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES), dtype=nl.float32, buffer=nl.sbuf)
    bias_grad_accum = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(bias_grad_accum, value=0.0)
    existing_bias = sbm.alloc_stack((H_TILE_SIZE, NUM_H_TILES), dtype=dtype, buffer=nl.sbuf)

    for b_tile_idx in range(NUM_B_TILES):
        b_offset = b_tile_idx * B_TILE_SIZE

        for h_tile_idx in range(NUM_H_TILES):
            h_offset = H_SHARD_OFFSET + h_tile_idx * H_TILE_SIZE
            valid_h = min(H_TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)

            # Load grad tile from HBM [B, H]
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

            # Reduce over B dimension
            nisa.tensor_reduce(
                dst=reduced[0:valid_h, h_tile_idx], op=nl.add, data=grad_tiles[0:valid_h, h_tile_idx, :], axis=1
            )

            # Accumulate
            nisa.tensor_tensor(
                dst=bias_grad_accum[0:valid_h, h_tile_idx],
                op=nl.add,
                data1=bias_grad_accum[0:valid_h, h_tile_idx],
                data2=reduced[0:valid_h, h_tile_idx],
            )

    for h_tile_idx in range(NUM_H_TILES):
        h_offset = H_SHARD_OFFSET + h_tile_idx * H_TILE_SIZE
        valid_h = min(H_TILE_SIZE, H_PER_SHARD - h_tile_idx * H_TILE_SIZE)
        # Load existing bias grad from HBM
        nisa.dma_copy(
            dst=existing_bias[0:valid_h, h_tile_idx],
            src=down_proj_bias_grad.ap(
                pattern=[[1, valid_h], [1, 1]],
                offset=h_offset,
                scalar_offset=expert_idx,
                indirect_dim=0,
            ),
            dge_mode=dge_mode.hwdge,
        )

        nisa.tensor_tensor(
            dst=bias_grad_accum[0:valid_h, h_tile_idx],
            op=nl.add,
            data1=bias_grad_accum[0:valid_h, h_tile_idx],
            data2=existing_bias[0:valid_h, h_tile_idx],
        )

        # Store back to HBM
        nisa.dma_copy(
            dst=down_proj_bias_grad.ap(
                pattern=[[1, valid_h], [1, 1]],
                offset=h_offset,
                scalar_offset=expert_idx,
                indirect_dim=0,
            ),
            src=bias_grad_accum[0:valid_h, h_tile_idx],
            dge_mode=dge_mode.hwdge,
        )

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
):
    """
    Compute gate and up projection bias gradient.

    Reduces gate_up_proj_output_grad [B, 2, I_TP] over B dimension.
    With LNC2, shard_id determines gate (0) or up (1).

    Args:
      gate_up_proj_output_grad_hbm: Gradient tensor [B, 2, I_TP] in HBM
      gate_and_up_proj_bias_grad: Output bias gradient [E, 2, I_TP] in HBM
      expert_idx: Expert index tensor for indirect addressing
      B: Batch/block size
      I_TP: Intermediate dimension
      shard_id: Current shard ID (0=gate, 1=up)
      dtype: Data type
      sbm: SBUF memory manager
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    GATE_UP_WEIGHT_COUNT = 2

    B_TILE_SIZE = min(TILE_SIZE, B)
    NUM_B_TILES = div_ceil(B, B_TILE_SIZE)
    I_TILE_SIZE = min(TILE_SIZE, I_TP)
    NUM_I_TILES = div_ceil(I_TP, I_TILE_SIZE)

    sbm.open_scope(name="gate_up_proj_bias_grad")

    # Allocate buffers - transposed layout [I, NUM_I_TILES, B]
    grad_tiles = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES, B_TILE_SIZE), dtype=dtype, buffer=nl.sbuf, align=32)
    reduced = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES), dtype=nl.float32, buffer=nl.sbuf)
    bias_grad_accum = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(bias_grad_accum, value=0.0)
    existing_bias = sbm.alloc_stack((I_TILE_SIZE, NUM_I_TILES), dtype=dtype, buffer=nl.sbuf)

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

    for i_tile_idx in range(NUM_I_TILES):
        i_offset = i_tile_idx * I_TILE_SIZE
        valid_i = min(I_TILE_SIZE, I_TP - i_offset)

        # Load existing bias grad from HBM [E, 2, I_TP]
        nisa.dma_copy(
            dst=existing_bias[0:valid_i, i_tile_idx],
            src=gate_and_up_proj_bias_grad.ap(
                pattern=[[1, valid_i], [1, 1]],
                offset=shard_id * I_TP + i_offset,
                scalar_offset=expert_idx,
                indirect_dim=0,
            ),
            dge_mode=dge_mode.hwdge,
        )

        # Add to existing
        nisa.tensor_tensor(
            dst=bias_grad_accum[0:valid_i, i_tile_idx],
            op=nl.add,
            data1=bias_grad_accum[0:valid_i, i_tile_idx],
            data2=existing_bias[0:valid_i, i_tile_idx],
        )

        # Store back to HBM
        nisa.dma_copy(
            dst=gate_and_up_proj_bias_grad.ap(
                pattern=[[1, valid_i], [1, 1]],
                offset=shard_id * I_TP + i_offset,
                scalar_offset=expert_idx,
                indirect_dim=0,
            ),
            src=bias_grad_accum[0:valid_i, i_tile_idx],
            dge_mode=dge_mode.hwdge,
        )

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
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    _, B_DIM, H_DIM = down_proj_act_checkpoint.shape

    B_TILE_SIZE, NUM_B_TILES = block_token_pos_to_id_full.shape

    NUM_B_TILE_SHARD = NUM_B_TILES // num_shards
    NUM_B_TILE_SHARD_OFFSET = NUM_B_TILE_SHARD * shard_id
    H_TILE_SIZE = TILE_SIZE
    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM)
    NUM_H_BLOCKS = div_ceil(H_DIM, H_BLOCK_SIZE)

    down_proj_output_grad_hbm = nl.ndarray(
        (B_DIM, H_DIM), dtype=dtype, buffer=nl.shared_hbm, name=f"down_proj_output_grad_hbm_shared_block_{block_idx}"
    )
    # Broadcast expert_idx to tile size
    sbm.open_scope(name="Down Projection Outptu Grad")

    expert_idx_tensor = sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32, name=f"expert_idx_tensor_{block_idx}")
    stream_shuffle_broadcast(expert_idx, expert_idx_tensor)
    BUFFER_DEGREE = 4

    expert_affinity_tile = []
    expert_affinityfp32_tile = []
    ea_grad_accum = []
    ea_grad_reduced = []

    token_indices_offset = []

    addr = []

    for n_buffer in range(BUFFER_DEGREE):
        expert_affinity_tile.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=expert_affinities_masked.dtype))
        expert_affinityfp32_tile.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32))
        ea_grad_accum.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.float32))
        ea_grad_reduced.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=dtype))
        token_indices_offset.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32))
        addr.append(sbm.alloc_stack((B_TILE_SIZE, 1), dtype=nl.int32))

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
            )
            block_down_proj_ckpt_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"block_down_proj_ckpt_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
            )
            block_multiply_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"block_multiply_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
            )
            down_proj_output_grad_tile = sbm.alloc_stack(
                (B_TILE_SIZE, valid_h),
                dtype=dtype,
                name=f"down_proj_output_grad_tile_{block_idx}_{global_tile_idx}_{h_block_idx}",
            )
            ea_grad_local = sbm.alloc_stack(
                (B_TILE_SIZE, 1), dtype=nl.float32, name=f"ea_grad_local_{block_idx}_{global_tile_idx}_{h_block_idx}"
            )

            if skip_dma.skip_token:
                nisa.memset(block_hidden_grad_tile, value=0)

            nisa.dma_copy(
                dst=block_hidden_grad_tile,
                src=output_hidden_states_grad.ap(
                    pattern=[[H_DIM, B_TILE_SIZE], [1, valid_h]],
                    offset=h_offset,
                    vector_offset=block_token_pos_to_id_full.ap(
                        pattern=[[block_token_pos_to_id_full.shape[-1], B_TILE_SIZE], [1, 1]], offset=global_tile_idx
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
                dst=ea_grad_accum[ea_buffer_idx], op=nl.add, data1=ea_grad_accum[ea_buffer_idx], data2=ea_grad_local
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

    for b_tile_idx in range(NUM_B_TILES):  # Add Mask to this Read
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
    nki_activation_fwd_op,
    nki_activation_bwd_op,
    sbm,
    clamp_limits: ClampLimits = ClampLimits(),
    BLOCK_H=2,
    BLOCK_B=2,
    BLOCK_I_TP=2,
):
    """
    Compute gate and up projection output gradients with I-dimension sharding.

    Performs backward pass through the activation function and computes gradients
    for gate and up projections. Shards on I dimension to avoid send/recv after matmul.

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
        down_proj_bias_grad (nl.ndarray, optional): [E, H], Down projection bias gradient.
        sbm (SbufManager): SBUF memory manager.
        clamp_limits (ClampLimits): Gradient clamping limits.
        BLOCK_H (int): H dimension blocking factor.
        BLOCK_B (int): B dimension blocking factor.
        BLOCK_I_TP (int): I dimension blocking factor.

    Returns:
        tuple: (gate_up_proj_output_grad_hbm, gate_up_multiply_output_hbm)
            - gate_up_proj_output_grad_hbm: [B, 2, I_TP], Gate/up projection gradients.
            - gate_up_multiply_output_hbm: [B, I_TP], Gate * up product for weight grad.

    Notes:
        - gate_grad = d_silu_gate * silu_dx(gate_activation)
        - up_grad = d_output * silu(gate_activation)
        - Optionally applies gradient clamping based on clamp_limits.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    GATE_UP_WEIGHT_COUNT = 2

    B_DIM, _ = down_proj_output_grad_hbm.shape
    E, I_TP_DIM, H_DIM = down_projection_weight.shape

    # Shard on I dimension instead of H - each shard processes I_TP_DIM/num_shards
    I_TP_DIM_SHARDED = I_TP_DIM // num_shards
    I_TP_SHARD_OFFSET = I_TP_DIM_SHARDED * shard_id

    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    H_TILE_SIZE = min(TILE_SIZE, H_DIM)
    I_TP_TILE_SIZE = min(PSUM_SIZE, I_TP_DIM_SHARDED)

    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM)
    I_TP_BLOCK_SIZE = min(BLOCK_I_TP * I_TP_TILE_SIZE, I_TP_DIM_SHARDED)
    B_BLOCK_SIZE = min(BLOCK_B * B_TILE_SIZE, B_DIM)

    NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)
    NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)
    NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

    NUM_I_TP_BLOCKS = div_ceil(I_TP_DIM_SHARDED, I_TP_BLOCK_SIZE)
    NUM_H_BLOCKS = div_ceil(H_DIM, H_BLOCK_SIZE)
    NUM_B_BLOCKS = div_ceil(B_DIM, B_BLOCK_SIZE)
    NUM_I_TP_INNER_TILES = div_ceil(I_TP_TILE_SIZE, TILE_SIZE)

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

    sbm.open_scope(name=f"Gate Up Output Grad")

    up_activation_grad = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, name=f"up_act_grad_{block_idx}"
    )

    # Compute silu_dx(gate_activation)
    silu_dx_gate = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, name=f"silu_dx_gate_{block_idx}"
    )
    silu_gate_grad = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, name=f"silu_gate_grad_{block_idx}"
    )

    # Gradient of gate_activation: d_gate = d_silu_gate * silu_dx(gate)
    gate_activation_grad = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, name=f"gate_act_grad_{block_idx}"
    )

    # Allocate clamping buffers for non-linear activation gradient clamping
    if clamp_limits.non_linear_clamp_upper_limit is not None or clamp_limits.non_linear_clamp_lower_limit is not None:
        clamp_mask1 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype)
        clamp_mask2 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype)
        clamp_mask = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype)

    # Allocate clamping buffers for linear activation gradient clamping
    if clamp_limits.linear_clamp_upper_limit is not None or clamp_limits.linear_clamp_lower_limit is not None:
        linear_clamp_mask1 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype)
        linear_clamp_mask2 = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype)
        linear_clamp_mask = sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype)

    gate_silu_activation = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, name=f"gate_silu_act_{block_idx}"
    )

    block_gate_up_mult = sbm.alloc_stack(
        (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, name=f"gate_up_mult_{block_idx}"
    )

    BUFFER_DEGREE = 3
    # Temp Allocation Outside the stack because DMA Transpose is currently not supported on scalar offsets
    down_proj_weight_temp = []
    partial_block_gate_up_mult_grad = []
    transpose_psum_idx = 0
    matmul_psum_idx = 0
    gate_activation = []
    up_activation = []

    for n_buffer in range(BUFFER_DEGREE):
        down_proj_weight_temp.append(
            sbm.alloc_stack(
                (TILE_SIZE, H_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )
        partial_block_gate_up_mult_grad.append(
            sbm.alloc_stack(
                (B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )
        gate_activation.append(
            sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        )
        up_activation.append(
            sbm.alloc_stack((B_TILE_SIZE, NUM_B_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, align=32)
        )

    sbm.open_scope(interleave_degree=BUFFER_DEGREE, name=f"Gate Up Output Grad {block_idx}")

    for i_tp_block_idx in range(NUM_I_TP_BLOCKS):
        # Global I offset includes shard offset
        i_tp_global_offset = I_TP_SHARD_OFFSET + i_tp_block_idx * I_TP_BLOCK_SIZE
        valid_i_block = min(I_TP_BLOCK_SIZE, I_TP_DIM_SHARDED - i_tp_block_idx * I_TP_BLOCK_SIZE)

        for b_block_idx in range(NUM_B_BLOCKS):
            gup_buffer_idx = (i_tp_block_idx * NUM_B_BLOCKS + b_block_idx) % BUFFER_DEGREE
            # Perform Gup Forward and Save Intermediate States
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
                    nisa.activation(
                        dst=gate_silu_activation[:, :, 0:valid_i_block],
                        op=nki_activation_fwd_op,
                        data=gate_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                        bias=None,
                        scale=1.0,
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
                valid_h_block = min(H_BLOCK_SIZE, H_DIM - h_block_start)

                down_projection_weight_transposed = sbm.alloc_stack(
                    (H_TILE_SIZE, NUM_H_TILES, I_TP_BLOCK_SIZE), dtype=compute_dtype, buffer=nl.sbuf
                )
                down_proj_output_grad_transposed = sbm.alloc_stack(
                    (H_TILE_SIZE, NUM_H_TILES, B_BLOCK_SIZE),
                    dtype=compute_dtype,
                    buffer=nl.sbuf,
                    align=32,  # DMA transpose requires 32-byte alignment
                )

                # Load and Transpose DP Weights - now using full H (no sharding)
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
                                offset=i_tp_offset * H_DIM + h_block_start,
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
                            transpose_psum_idx = transpose_psum_idx % NUM_HW_PSUM_BANKS

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

                # Load and Transpose down_proj_output_grad - full H
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
                            offset=(b_block_idx * B_BLOCK_SIZE) * H_DIM + h_block_start + h_tile_start,
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
                        matmul_psum_idx = matmul_psum_idx % NUM_HW_PSUM_BANKS
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

            nisa.tensor_tensor(
                dst=up_activation_grad[:, :, 0:valid_i_block],
                op=nl.multiply,
                data1=partial_block_gate_up_mult_grad[result_buffer_idx][:, :, 0:valid_i_block],
                data2=gate_silu_activation[:, :, 0:valid_i_block],
            )

            # Apply gradient clamping for linear activation (up) backward pass
            if clamp_limits.linear_clamp_upper_limit is not None or clamp_limits.linear_clamp_lower_limit is not None:
                nisa.memset(linear_clamp_mask1, value=1.0)
                nisa.memset(linear_clamp_mask2, value=1.0)

                if clamp_limits.linear_clamp_upper_limit is not None:
                    nisa.tensor_scalar(
                        dst=linear_clamp_mask1[:, :, 0:valid_i_block],
                        data=up_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                        op0=nl.less,
                        operand0=clamp_limits.linear_clamp_upper_limit,
                    )

                if clamp_limits.linear_clamp_lower_limit is not None:
                    nisa.tensor_scalar(
                        dst=linear_clamp_mask2[:, :, 0:valid_i_block],
                        data=up_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                        op0=nl.greater,
                        operand0=clamp_limits.linear_clamp_lower_limit,
                    )

                nisa.tensor_tensor(
                    dst=linear_clamp_mask[:, :, 0:valid_i_block],
                    data1=linear_clamp_mask1[:, :, 0:valid_i_block],
                    data2=linear_clamp_mask2[:, :, 0:valid_i_block],
                    op=nl.logical_and,
                )

                nisa.tensor_tensor(
                    dst=up_activation_grad[:, :, 0:valid_i_block],
                    op=nl.multiply,
                    data1=up_activation_grad[:, :, 0:valid_i_block],
                    data2=linear_clamp_mask[:, :, 0:valid_i_block],
                )

            nisa.tensor_tensor(
                dst=silu_gate_grad[:, :, 0:valid_i_block],
                op=nl.multiply,
                data1=partial_block_gate_up_mult_grad[result_buffer_idx][:, :, 0:valid_i_block],
                data2=up_activation[gup_buffer_idx][:, :, 0:valid_i_block],
            )

            nisa.activation(
                dst=silu_dx_gate[:, :, 0:valid_i_block],
                op=nki_activation_bwd_op,
                data=gate_activation[gup_buffer_idx][:, :, 0:valid_i_block],
            )

            nisa.tensor_tensor(
                dst=gate_activation_grad[:, :, 0:valid_i_block],
                op=nl.multiply,
                data1=silu_gate_grad[:, :, 0:valid_i_block],
                data2=silu_dx_gate[:, :, 0:valid_i_block],
            )

            # Apply gradient clamping for non-linear activation backward pass
            if (
                clamp_limits.non_linear_clamp_upper_limit is not None
                or clamp_limits.non_linear_clamp_lower_limit is not None
            ):
                nisa.memset(clamp_mask1, value=1.0)
                nisa.memset(clamp_mask2, value=1.0)

                if clamp_limits.non_linear_clamp_upper_limit is not None:
                    nisa.tensor_scalar(
                        dst=clamp_mask1[:, :, 0:valid_i_block],
                        data=gate_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                        op0=nl.less,
                        operand0=clamp_limits.non_linear_clamp_upper_limit,
                    )

                if clamp_limits.non_linear_clamp_lower_limit is not None:
                    nisa.tensor_scalar(
                        dst=clamp_mask2[:, :, 0:valid_i_block],
                        data=gate_activation[gup_buffer_idx][:, :, 0:valid_i_block],
                        op0=nl.greater,
                        operand0=clamp_limits.non_linear_clamp_lower_limit,
                    )

                nisa.tensor_tensor(
                    dst=clamp_mask[:, :, 0:valid_i_block],
                    data1=clamp_mask1[:, :, 0:valid_i_block],
                    data2=clamp_mask2[:, :, 0:valid_i_block],
                    op=nl.logical_and,
                )

                nisa.tensor_tensor(
                    dst=gate_activation_grad[:, :, 0:valid_i_block],
                    op=nl.multiply,
                    data1=gate_activation_grad[:, :, 0:valid_i_block],
                    data2=clamp_mask[:, :, 0:valid_i_block],
                )

            """
            Write gradients to gate_up_proj_output_grad_hbm.
            Shape: (B_DIM, GATE_UP_WEIGHT_COUNT, I_TP_DIM)
            Use global I offset directly - no additional sharding offset needed.
            """
            for b_tile_idx in range(NUM_B_TILES):
                b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                # Write gate gradient (index 0)
                nisa.dma_copy(
                    dst=gate_up_proj_output_grad_hbm.ap(
                        pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                        offset=b_offset * GATE_UP_WEIGHT_COUNT * I_TP_DIM + 0 * I_TP_DIM + i_tp_global_offset,
                    ),
                    src=gate_activation_grad[:, b_tile_idx, 0:valid_i_block],
                    dge_mode=dge_mode.hwdge,
                )
                # Write up gradient (index 1)
                nisa.dma_copy(
                    dst=gate_up_proj_output_grad_hbm.ap(
                        pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_TILE_SIZE], [1, valid_i_block]],
                        offset=b_offset * GATE_UP_WEIGHT_COUNT * I_TP_DIM + 1 * I_TP_DIM + i_tp_global_offset,
                    ),
                    src=up_activation_grad[:, b_tile_idx, 0:valid_i_block],
                    dge_mode=dge_mode.hwdge,
                )

    sbm.close_scope()
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

    Returns:
        None: Weight gradient is accumulated in-place.

    Notes:
        - weight_grad[expert, :, H_shard] += gate_up_mult^T @ down_proj_grad
        - Each shard writes to its H/num_shards portion.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    B_DIM, _ = down_proj_output_grad_hbm.shape
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

    BUFFER_DEGREE = 3
    matmul_psum_idx = 0
    sbm.open_scope(name=f"Down Projection Weight Grad")

    result_tiles = []
    existing_weight_grad = []

    for n_buffer in range(BUFFER_DEGREE):
        result_tiles.append(
            sbm.alloc_stack(
                (I_TP_TILE_SIZE, NUM_I_TP_TILES, H_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )
        existing_weight_grad.append(
            sbm.alloc_stack(
                (I_TP_TILE_SIZE, NUM_I_TP_TILES, H_BLOCK_SIZE),
                dtype=compute_dtype,
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
                )
                rhs_tiles = sbm.alloc_stack(
                    (B_TILE_SIZE, NUM_B_TILES, H_BLOCK_SIZE),
                    dtype=compute_dtype,
                    name=f"down_wgrad_rhs_blk{block_idx}_h{h_block_idx}_i{i_block_idx}_b{b_block_idx}",
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
                        matmul_psum_idx = matmul_psum_idx % NUM_HW_PSUM_BANKS

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

            # Write Down Projection weight
            for i_tile_idx in range(NUM_I_TP_TILES):
                i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                if i_tile_start >= valid_i_block:
                    break
                valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)

                i_offset = i_block_start + i_tile_start

                # Load existing gradient from HBM
                nisa.dma_copy(
                    dst=existing_weight_grad[result_tile_idx][0:valid_i_tile, i_tile_idx, 0:valid_h_block],
                    src=down_projection_weight_grad.ap(
                        pattern=[[H_DIM, valid_i_tile], [1, 1], [1, valid_h_block]],
                        offset=i_offset * H_DIM + H_SHARD_OFFSET + h_block_start,
                        scalar_offset=expert_idx,
                        indirect_dim=0,
                    ),
                    dge_mode=dge_mode.hwdge,
                )

                # Accumulate
                nisa.tensor_tensor(
                    dst=result_tiles[result_tile_idx][0:valid_i_tile, i_tile_idx, 0:valid_h_block],
                    op=nl.add,
                    data1=existing_weight_grad[result_tile_idx][0:valid_i_tile, i_tile_idx, 0:valid_h_block],
                    data2=result_tiles[result_tile_idx][0:valid_i_tile, i_tile_idx, 0:valid_h_block],
                )

                # Write back
                nisa.dma_copy(
                    dst=down_projection_weight_grad.ap(
                        pattern=[[H_DIM, valid_i_tile], [1, 1], [1, valid_h_block]],
                        offset=i_offset * H_DIM + H_SHARD_OFFSET + h_block_start,
                        scalar_offset=expert_idx,
                        indirect_dim=0,
                    ),
                    src=result_tiles[result_tile_idx][0:valid_i_tile, i_tile_idx, 0:valid_h_block],
                    dge_mode=dge_mode.hwdge,
                )

    sbm.close_scope()
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
        gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP], Bias gradients.
        BLOCK_H (int): H dimension blocking factor.
        BLOCK_B (int): B dimension blocking factor.
        BLOCK_I_TP (int): I dimension blocking factor.

    Returns:
        None: Hidden states gradient is written/accumulated in-place.

    Notes:
        - hidden_grad[:, H_shard] = sum(gate_up_grad @ gate_up_weight, axis=gate_or_up)
        - Optionally accumulates gate/up bias gradients by reducing over B dimension.
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

    BUFFER_DEGREE = 3
    NUM_H_INNER_TILES = div_ceil(H_TILE_SIZE, TILE_SIZE)
    sbm.open_scope(name=f"Hidden States Grad")

    matmul_psum_idx = 0
    transpose_psum_idx = 0
    result_tiles = []
    existing_hidden_grad = []
    rhs_temp = []
    for n_buffer in range(BUFFER_DEGREE):
        rhs_temp.append(
            sbm.alloc_stack(
                (TILE_SIZE, I_TP_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )
        result_tiles.append(
            sbm.alloc_stack(
                (B_TILE_SIZE, NUM_B_TILES, H_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )

        if is_tensor_update_accumulating:
            existing_hidden_grad.append(
                sbm.alloc_stack(
                    (B_TILE_SIZE, NUM_B_TILES, H_BLOCK_SIZE),
                    dtype=compute_dtype,
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
                    )

                    # Load lhs_tiles from gate_up_proj_output_grad_hbm [B, 2, I_TP] with dma_transpose
                    for i_tile_idx in range(NUM_I_TP_TILES):
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

                    # Load rhs_tiles from gate_up_proj_weight [E, H, 2, I_TP] with transpose
                    for h_tile_idx in range(NUM_H_TILES):
                        h_tile_start = h_tile_idx * H_TILE_SIZE
                        if h_tile_start >= valid_h_block:
                            break
                        valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)

                        for h_inner_idx in range(NUM_H_INNER_TILES):
                            h_inner_start = h_inner_idx * TILE_SIZE
                            if h_inner_start >= valid_h_tile:
                                break
                            valid_h_inner = min(TILE_SIZE, valid_h_tile - h_inner_start)
                            h_offset = h_block_start + h_tile_start + h_inner_start
                            rhs_temp_buffer_idx = (h_tile_idx * NUM_H_INNER_TILES + h_inner_idx) % BUFFER_DEGREE

                            nisa.dma_copy(
                                dst=rhs_temp[rhs_temp_buffer_idx][0:valid_h_inner, 0:valid_i_block],
                                src=gate_up_proj_weight.ap(
                                    pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, valid_h_inner], [1, valid_i_block]],
                                    offset=(H_SHARD_OFFSET + h_offset) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                                    + gate_or_up * I_TP_DIM
                                    + i_block_start,
                                    scalar_offset=expert_idx,
                                    indirect_dim=0,
                                ),
                                dge_mode=dge_mode.hwdge,
                            )
                            for i_tile_idx in range(NUM_I_TP_TILES):
                                i_tile_start = i_tile_idx * I_TP_TILE_SIZE
                                if i_tile_start >= valid_i_block:
                                    break
                                valid_i_tile = min(I_TP_TILE_SIZE, valid_i_block - i_tile_start)
                                transpose_psum_idx += 1
                                transpose_psum_idx = transpose_psum_idx % NUM_HW_PSUM_BANKS
                                rhs_transposed_psum = nl.ndarray(
                                    (I_TP_TILE_SIZE, TILE_SIZE),
                                    dtype=compute_dtype,
                                    buffer=nl.psum,
                                    address=(0, transpose_psum_idx * PSUM_BANK_SIZE),
                                )
                                nisa.nc_transpose(
                                    dst=rhs_transposed_psum[0:valid_i_tile, 0:valid_h_inner],
                                    data=rhs_temp[rhs_temp_buffer_idx][
                                        0:valid_h_inner, nl.ds(i_tile_start, valid_i_tile)
                                    ],
                                )
                                h_dst_offset = h_tile_start + h_inner_start
                                nisa.tensor_copy(
                                    dst=rhs_tiles[0:valid_i_tile, i_tile_idx, nl.ds(h_dst_offset, valid_h_inner)],
                                    src=rhs_transposed_psum[0:valid_i_tile, 0:valid_h_inner],
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
                            matmul_psum_idx = matmul_psum_idx % NUM_HW_PSUM_BANKS
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
                            if i_block_idx == 0 and gate_or_up == 0:
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
    H_DIM_SHARDED = H_DIM // num_shards  # 1440
    H_SHARD_OFFSET = H_DIM_SHARDED * shard_id

    B_TILE_SIZE = min(TILE_SIZE, B_DIM)
    H_TILE_SIZE = min(TILE_SIZE, H_DIM_SHARDED)
    I_TP_TILE_SIZE = min(PSUM_SIZE, I_TP_DIM)  # 512

    H_BLOCK_SIZE = min(BLOCK_H * H_TILE_SIZE, H_DIM_SHARDED)  # 256
    I_TP_BLOCK_SIZE = min(BLOCK_I_TP * I_TP_TILE_SIZE, I_TP_DIM)  # 1024
    B_BLOCK_SIZE = min(BLOCK_B * B_TILE_SIZE, B_DIM)

    NUM_H_TILES = div_ceil(H_BLOCK_SIZE, H_TILE_SIZE)  # 2
    NUM_I_TP_TILES = div_ceil(I_TP_BLOCK_SIZE, I_TP_TILE_SIZE)  # 2
    NUM_B_TILES = div_ceil(B_BLOCK_SIZE, B_TILE_SIZE)

    NUM_I_TP_BLOCKS = div_ceil(I_TP_DIM, I_TP_BLOCK_SIZE)  # 3
    NUM_H_BLOCKS = div_ceil(H_DIM_SHARDED, H_BLOCK_SIZE)  # 6
    NUM_B_BLOCKS = div_ceil(B_DIM, B_BLOCK_SIZE)

    BUFFER_DEGREE = 3
    sbm.open_scope(name=f"Gate Up Weight Grad")

    weight_grad_accum = []
    existing_weight_grad = []
    matmul_psum_idx = 0

    for n_buffer in range(BUFFER_DEGREE):
        weight_grad_accum.append(
            sbm.alloc_stack(  # 128, 2, 1024
                (H_TILE_SIZE, NUM_H_TILES, I_TP_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )
        existing_weight_grad.append(
            sbm.alloc_stack(  # 128, 2, 1024
                (H_TILE_SIZE, NUM_H_TILES, I_TP_BLOCK_SIZE),
                dtype=compute_dtype,
            )
        )
    sbm.open_scope(name=f"Gate Up Weight Grad Double Buffer", interleave_degree=BUFFER_DEGREE)
    # Process one (h_tile, i_tile) at a time to minimize memory
    for h_block_idx in range(NUM_H_BLOCKS):
        h_block_start = h_block_idx * H_BLOCK_SIZE
        valid_h_block = min(H_BLOCK_SIZE, H_DIM_SHARDED - h_block_start)  # 256

        for gate_or_up in range(GATE_UP_WEIGHT_COUNT):
            for i_block_idx in range(NUM_I_TP_BLOCKS):
                i_block_start = i_block_idx * I_TP_BLOCK_SIZE
                valid_i_block = min(I_TP_BLOCK_SIZE, I_TP_DIM - i_block_start)  # 1024

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
                    )
                    block_hidden_states = sbm.alloc_stack(
                        (B_TILE_SIZE, NUM_B_TILES, valid_h_block),
                        dtype=compute_dtype,
                        name=f"gup_wgrad_hidden_blk{block_idx}_h{h_block_idx}_g{gate_or_up}_i{i_block_idx}_b{b_block_idx}",
                    )
                    if skip_dma.skip_token:
                        nisa.memset(block_hidden_states[:, :, 0:valid_h_block], value=0.0)

                    for b_tile_idx in range(NUM_B_TILES):
                        # Load gate_up_proj_output_grad for this tile
                        global_tile_idx = b_block_idx * NUM_B_TILES + b_tile_idx
                        b_offset = b_block_idx * B_BLOCK_SIZE + b_tile_idx * B_TILE_SIZE
                        nisa.dma_copy(
                            dst=gate_up_proj_output_grad[:, b_tile_idx, 0:valid_i_block],
                            src=gate_up_proj_output_grad_hbm.ap(
                                pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, B_TILE_SIZE], [1, 1], [1, valid_i_block]],
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
                            matmul_psum_idx = matmul_psum_idx % NUM_HW_PSUM_BANKS

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

                # Accumulate Gradients
                for h_tile_idx in range(NUM_H_TILES):
                    h_tile_start = h_tile_idx * H_TILE_SIZE  # 0
                    if h_tile_start >= valid_h_block:
                        break
                    valid_h_tile = min(H_TILE_SIZE, valid_h_block - h_tile_start)  # 128

                    nisa.dma_copy(
                        dst=existing_weight_grad[result_buffer_idx][0:valid_h_tile, h_tile_idx, 0:valid_i_block],
                        src=gate_up_proj_weight_grad.ap(
                            pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, valid_h_tile], [1, 1], [1, valid_i_block]],
                            offset=(H_SHARD_OFFSET + h_block_start + h_tile_start) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                            + gate_or_up * I_TP_DIM
                            + i_block_start,
                            scalar_offset=expert_idx,
                            indirect_dim=0,
                        ),
                        dge_mode=dge_mode.hwdge,
                    )

                    nisa.tensor_tensor(
                        dst=existing_weight_grad[result_buffer_idx][0:valid_h_tile, h_tile_idx, 0:valid_i_block],
                        op=nl.add,
                        data1=existing_weight_grad[result_buffer_idx][0:valid_h_tile, h_tile_idx, 0:valid_i_block],
                        data2=weight_grad_accum[result_buffer_idx][0:valid_h_tile, h_tile_idx, 0:valid_i_block],
                    )

                    nisa.dma_copy(
                        dst=gate_up_proj_weight_grad.ap(
                            pattern=[[GATE_UP_WEIGHT_COUNT * I_TP_DIM, valid_h_tile], [1, 1], [1, valid_i_block]],
                            offset=(H_SHARD_OFFSET + h_block_start + h_tile_start) * GATE_UP_WEIGHT_COUNT * I_TP_DIM
                            + gate_or_up * I_TP_DIM
                            + i_block_start,
                            scalar_offset=expert_idx,
                            indirect_dim=0,
                        ),
                        src=existing_weight_grad[result_buffer_idx][0:valid_h_tile, h_tile_idx, 0:valid_i_block],
                        dge_mode=dge_mode.hwdge,
                    )

    sbm.close_scope()
    sbm.close_scope()


def _load_token_indices(token_position_to_id, block_idx, B, NUM_TILES, sbm):
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

    Returns:
        nl.ndarray: [TILE_SIZE, NUM_TILES], Transposed token indices in SBUF.
    """
    TILE_SIZE = nl.tile_size.gemm_stationary_fmax
    PSUM_SIZE = nl.tile_size.gemm_moving_fmax
    token_position_to_id_sbuf = sbm.alloc_stack(
        (TILE_SIZE, NUM_TILES), dtype=nl.int32, name=f"token_position_to_id_sbuf_{block_idx}"
    )

    sbm.open_scope(name="_load_token_indices", interleave_degree=2)
    transpose_psum_idx = 0
    for tile_idx in range(NUM_TILES):
        token_pos_to_id_fp32 = sbm.alloc_stack(
            (1, TILE_SIZE), dtype=nl.float32, name=f"token_pos_to_id_fp32_{block_idx}_{tile_idx}"
        )
        offset = block_idx * B + TILE_SIZE * tile_idx

        nisa.dma_copy(
            token_pos_to_id_fp32.ap(pattern=[[TILE_SIZE, 1], [1, TILE_SIZE]]),
            token_position_to_id.ap(pattern=[[TILE_SIZE, 1], [1, TILE_SIZE]], offset=offset),
        )
        transpose_psum_idx += 1

        transpose_psum_idx = transpose_psum_idx % NUM_HW_PSUM_BANKS
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
        for block_idx in range(N):
            expert_idx = _load_block_expert(block_to_expert, block_idx)
            block_token_pos_to_id = _load_token_indices(token_position_to_id, block_idx)

            # Step 1: Compute down projection output gradient and expert affinity gradient
            down_proj_output_grad = _compute_down_projection_output_grad(...)

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

    # Get activation functions
    nki_activation_fwd_op, nki_activation_bwd_op = params.get_activation_ops()

    _, num_shards, shard_id = get_program_sharding_info()
    params.validate_sharding(num_shards)

    sbm = SbufManager(0, MAX_AVAILABLE_SBUF_SIZE, logger=get_logger("SBM"))
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

    for block_idx in range(N):
        sbm.open_scope(name=f"Block {block_idx}")
        expert_idx = _load_block_expert(params.block_to_expert, block_idx, sbm)
        # Load token indices once for the entire block
        block_token_pos_to_id_full = _load_token_indices(params.token_position_to_id, block_idx, B, NUM_B_TILES, sbm)

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
        )

        gate_up_proj_output_grad_hbm, gate_up_multipy_output_hbm = _compute_gate_up_projection_output_grad(
            down_proj_output_grad_hbm=down_proj_output_grad_hbm,
            down_projection_weight=params.down_proj_weight,
            gate_up_proj_act_checkpoint_T=params.gate_up_proj_act_checkpoint_T,
            shard_id=shard_id,
            num_shards=num_shards,
            block_idx=block_idx,
            expert_idx=expert_idx,
            compute_dtype=params.compute_dtype,
            nki_activation_fwd_op=nki_activation_fwd_op,
            nki_activation_bwd_op=nki_activation_bwd_op,
            sbm=sbm,
            clamp_limits=params.clamp_limits,
            BLOCK_H=bp.gate_up_output_grad.block_h,
            BLOCK_B=bp.gate_up_output_grad.block_b,
            BLOCK_I_TP=bp.gate_up_output_grad.block_i,
        )

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
            BLOCK_H=bp.down_weight_grad.block_h,
            BLOCK_B=bp.down_weight_grad.block_b,
            BLOCK_I_TP=bp.down_weight_grad.block_i,
        )

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
            BLOCK_H=bp.hidden_grad.block_h,
            BLOCK_B=bp.hidden_grad.block_b,
            BLOCK_I_TP=bp.hidden_grad.block_i,
        )

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
        )

        # Compute bias gradients
        if params.down_proj_bias_grad is not None:
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
            )

        if params.gate_and_up_proj_bias_grad is not None:
            _compute_gate_up_proj_bias_grad(
                gate_up_proj_output_grad_hbm=gate_up_proj_output_grad_hbm,
                gate_and_up_proj_bias_grad=params.gate_and_up_proj_bias_grad,
                expert_idx=expert_idx,
                B=B,
                I_TP=params.I_TP,
                shard_id=shard_id,
                dtype=params.compute_dtype,
                sbm=sbm,
            )

        sbm.close_scope()
