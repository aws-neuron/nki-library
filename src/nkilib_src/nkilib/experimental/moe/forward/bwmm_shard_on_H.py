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

"""Blockwise matrix multiplication kernel with hidden dimension (H) sharding for MoE."""

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import sendrecv
from nki.isa.constants import oob_mode

from ....core.moe.moe_cte.moe_cte_utils import (
    N_PSUM_BANKS,
    PSUM_SIZE,
    TILE_SIZE,
    TOTAL_PSUM_SIZE,
    Configs,
    SkipMode,
    calculate_expert_affinities,
    compute_intermediate_states,
    div_ceil,
    load_block_expert,
    load_token_indices,
)
from ....core.utils.common_types import ExpertAffinityScaleMode
from ....core.utils.kernel_assert import kernel_assert


@dataclass
class DimensionSizes(nl.NKIObject):
    """Dimension configuration for H-shard kernel."""

    T: int = 0
    H: int = 0
    B: int = 0
    E: int = 0
    N: int = 0
    I_TP: int = 0
    NUM_SHARDS: int = 0
    H_per_shard: int = 0
    NUM_TILES: int = 0

    def derive_all_dims(self):
        """Derive computed dimensions from base dimensions."""
        self.NUM_SHARDS = nl.num_programs(axes=0)
        self.H_per_shard = self.H // self.NUM_SHARDS
        self.NUM_TILES = self.B // TILE_SIZE


def output_initialization_shard(output, dims, shard_id):
    """Zero initialize output buffer for the current shard."""
    for tile_idx in range(div_ceil(dims.T, TILE_SIZE)):
        num_p = min(TILE_SIZE, dims.T - tile_idx * TILE_SIZE)
        zeros = nl.ndarray((TILE_SIZE, dims.H_per_shard), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(zeros, value=0)
        h_offset = dims.H_per_shard * shard_id
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[dims.H, num_p], [1, dims.H_per_shard]],
                offset=tile_idx * TILE_SIZE * dims.H + h_offset,
            ),
            src=zeros[0:num_p, 0 : dims.H_per_shard],
        )


def create_block_hidden_states(H_per_shard, NUM_TILES, dtype):
    """Create list of tensors for block hidden states."""
    block_hidden_states = []
    for _ in range(NUM_TILES):
        tile = nl.ndarray((TILE_SIZE, H_per_shard), dtype=dtype, buffer=nl.sbuf)
        block_hidden_states.append(tile)
    return block_hidden_states


def load_hidden_states_shard_with_scale(
    hidden_states, block_hidden_states, token_indices, expert_affinity, dims, cfg, shard_id
):
    """
    Load hidden states for the current block with optional pre-scaling.

    Loads hidden states from HBM using indirect indexing via token_indices,
    then optionally applies expert affinity scaling (PRE_SCALE mode).

    Args:
        hidden_states: [T+1, H], Input hidden states tensor on HBM.
        block_hidden_states: List of [TILE_SIZE, H_per_shard] SBUF tensors to fill.
        token_indices: [TILE_SIZE, NUM_TILES], Token indices for indirect load.
        expert_affinity: List of [TILE_SIZE, 1] affinity tensors, or None.
        dims: DimensionSizes configuration object.
        cfg: Configs object with skip_dma and other settings.
        shard_id: Current shard/core ID for H offset calculation.

    Returns:
        None. Modifies block_hidden_states in-place.
    """
    h_offset = dims.H_per_shard * shard_id
    for tile_idx in range(dims.NUM_TILES):
        if cfg.skip_dma.skip_token:
            nisa.memset(dst=block_hidden_states[tile_idx], value=0)
        tmp_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_idx, src=token_indices[0:TILE_SIZE, tile_idx])
        nisa.dma_copy(
            src=hidden_states.ap(
                pattern=[[dims.H, TILE_SIZE], [1, dims.H_per_shard]],
                offset=h_offset,
                vector_offset=tmp_idx,
                indirect_dim=0,
            ),
            dst=block_hidden_states[tile_idx][0:TILE_SIZE, 0 : dims.H_per_shard],
            oob_mode=oob_mode.skip if cfg.skip_dma.skip_token else oob_mode.error,
        )
        if expert_affinity != None:
            nisa.tensor_scalar(
                dst=block_hidden_states[tile_idx][0:TILE_SIZE, 0 : dims.H_per_shard],
                data=block_hidden_states[tile_idx][0:TILE_SIZE, 0 : dims.H_per_shard],
                op0=nl.multiply,
                operand0=expert_affinity[tile_idx][0:TILE_SIZE, 0],
            )


def transpose_hidden_states(block_hidden_states, dims, compute_dtype):
    """
    Transpose block hidden states from [B, H] to [H, B] layout.

    Performs tiled transpose using PSUM as intermediate buffer. Output is organized
    as a 2D list indexed by [h_outer][h_inner], where each element has shape
    [TILE_SIZE, block_psum_tiles, free_size].

    Args:
        block_hidden_states: List of [TILE_SIZE, H_per_shard] tensors in SBUF.
        dims: DimensionSizes configuration object.
        compute_dtype: Data type for transposed output tensors.

    Returns:
        List[List[nl.ndarray]]: 2D list of transposed tensors with shape
            [h_outer_tripcount][h_inner_tripcount], each element is
            [TILE_SIZE, block_psum_tiles, free_size].
    """
    h_outer_tripcount = div_ceil(dims.H_per_shard, PSUM_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    linearized_tripcount = div_ceil(dims.H_per_shard, TILE_SIZE)
    block_psum_tiles = div_ceil(dims.B, PSUM_SIZE)
    free_size = min(PSUM_SIZE, dims.B)
    block_free_tiles = min(PSUM_SIZE // TILE_SIZE, dims.B // TILE_SIZE)

    block_hidden_states_T = []
    for _ in range(h_outer_tripcount):
        inner_list = []
        for _ in range(h_inner_tripcount):
            tile = nl.ndarray((TILE_SIZE, block_psum_tiles, free_size), dtype=compute_dtype, buffer=nl.sbuf)
            inner_list.append(tile)
        block_hidden_states_T.append(inner_list)

    for psum_tile_idx in range(block_psum_tiles):
        for b_tile_idx in range(block_free_tiles):
            for h_outer_idx in range(h_outer_tripcount):
                for h_inner_idx in range(h_inner_tripcount):
                    offset = TILE_SIZE * b_tile_idx
                    trans_f_offset = TILE_SIZE * h_inner_idx + PSUM_SIZE * h_outer_idx
                    h_lin_idx = h_outer_idx * h_inner_tripcount + h_inner_idx
                    if h_lin_idx < linearized_tripcount:
                        num_h = min(TILE_SIZE, dims.H_per_shard - trans_f_offset)
                        tmp_psum = nl.ndarray((TILE_SIZE, TILE_SIZE), dtype=compute_dtype, buffer=nl.psum)
                        nisa.nc_transpose(
                            data=block_hidden_states[block_free_tiles * psum_tile_idx + b_tile_idx][
                                0:TILE_SIZE, trans_f_offset : trans_f_offset + num_h
                            ],
                            dst=tmp_psum[0:num_h, 0:TILE_SIZE],
                        )
                        nisa.tensor_copy(
                            dst=block_hidden_states_T[h_outer_idx][h_inner_idx][
                                0:num_h, psum_tile_idx, offset : offset + TILE_SIZE
                            ],
                            src=tmp_psum[0:num_h, 0:TILE_SIZE],
                        )
    return block_hidden_states_T


def load_gate_up_proj_weights_shard(gate_up_proj_weight, block_expert, cfg, dims, shard_id):
    """
    Load gate and up projection weights for the current shard.

    Loads the H_per_shard slice of gate/up weights for the current expert,
    organized as a 2D list for tiled matmul access.

    Args:
        gate_up_proj_weight: [E, H, 2, I_TP], Gate and up projection weights on HBM.
        block_expert: Scalar tensor containing current expert index.
        cfg: Configs object with weight_dtype and skip_dma settings.
        dims: DimensionSizes configuration object.
        shard_id: Current shard/core ID for H offset calculation.

    Returns:
        List[List[nl.ndarray]]: 2D list [h_outer][h_inner] of [TILE_SIZE, 2, I_TP]
            weight tensors in SBUF.
    """
    _, H, _, I_TP = gate_up_proj_weight.shape
    h_offset = dims.H_per_shard * shard_id
    h_outer_tripcount = div_ceil(dims.H_per_shard, PSUM_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    linearized_tripcount = div_ceil(dims.H_per_shard, TILE_SIZE)

    gup_weights = []
    for _ in range(h_outer_tripcount):
        inner_list = []
        for _ in range(h_inner_tripcount):
            tile = nl.ndarray((TILE_SIZE, 2, I_TP), dtype=cfg.weight_dtype, buffer=nl.sbuf)
            inner_list.append(tile)
        gup_weights.append(inner_list)

    for h_outer_idx in range(h_outer_tripcount):
        for h_inner_idx in range(h_inner_tripcount):
            load_p_offset = PSUM_SIZE * h_outer_idx + TILE_SIZE * h_inner_idx
            h_lin_idx = h_outer_idx * h_inner_tripcount + h_inner_idx
            if h_lin_idx < linearized_tripcount:
                num_p = min(TILE_SIZE, dims.H_per_shard - load_p_offset)
                offset = (h_offset + load_p_offset) * 2 * I_TP
                nisa.dma_copy(
                    dst=gup_weights[h_outer_idx][h_inner_idx][0:num_p, 0:2, 0:I_TP],
                    src=gate_up_proj_weight.ap(
                        pattern=[[2 * I_TP, num_p], [I_TP, 2], [1, I_TP]],
                        offset=offset,
                        scalar_offset=block_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if cfg.skip_dma.skip_weight else oob_mode.error,
                )
    return gup_weights


def load_down_proj_weight_shard(down_proj_weight, block_expert, cfg, dims, shard_id):
    """
    Load down projection weights for the current shard.

    Loads the H_per_shard slice of down projection weights for the current expert.

    Args:
        down_proj_weight: [E, I_TP, H], Down projection weights on HBM.
        block_expert: Scalar tensor containing current expert index.
        cfg: Configs object with weight_dtype and skip_dma settings.
        dims: DimensionSizes configuration object.
        shard_id: Current shard/core ID for H offset calculation.

    Returns:
        List[nl.ndarray]: List of [TILE_SIZE, H_per_shard] weight tensors in SBUF.
    """
    _, I_TP, H = down_proj_weight.shape
    gup_n_tile = div_ceil(I_TP, TILE_SIZE)
    h_offset = dims.H_per_shard * shard_id

    dp_weights = []
    for _ in range(gup_n_tile):
        tile = nl.ndarray((TILE_SIZE, dims.H_per_shard), dtype=cfg.weight_dtype, buffer=nl.sbuf)
        dp_weights.append(tile)

    for i_tile_idx in range(gup_n_tile):
        num_p = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
        offset = (TILE_SIZE * i_tile_idx) * H + h_offset
        nisa.dma_copy(
            dst=dp_weights[i_tile_idx][0:num_p, 0 : dims.H_per_shard],
            src=down_proj_weight.ap(
                pattern=[[H, num_p], [1, dims.H_per_shard]],
                offset=offset,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if cfg.skip_dma.skip_weight else oob_mode.error,
        )
    return dp_weights


def compute_gate_and_up_projections_shard(
    block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, dims, cfg, shard_id
):
    """
    Compute gate and up projections with LNC all-reduce.

    Performs matmul of transposed hidden states with gate/up weights, then uses
    sendrecv for LNC all-reduce to sum partial results across shards.

    Args:
        block_hidden_states_T: 2D list [h_outer][h_inner] of transposed hidden states.
        gup_weights: 2D list [h_outer][h_inner] of [TILE_SIZE, 2, I_TP] weight tensors.
        gate_up_activations_T: Optional [N, 2, I_TP, B] tensor for checkpointing.
        block_idx: Current block index for activation storage offset.
        dims: DimensionSizes configuration object.
        cfg: Configs object with compute settings.
        shard_id: Current shard/core ID for sendrecv partner calculation.

    Returns:
        List[List[List[nl.ndarray]]]: 3D list [gate_or_up][b_tile][i_tile] of
            [TILE_SIZE, free_size] tensors containing all-reduced projections.
    """
    N_PSUM_TILE = div_ceil(dims.B, PSUM_SIZE)
    gup_n_tile = div_ceil(dims.I_TP, TILE_SIZE)
    h_inner_tripcount = PSUM_SIZE // TILE_SIZE
    free_size = block_hidden_states_T[0][0].shape[-1]
    linearized_tripcount = div_ceil(dims.H_per_shard, TILE_SIZE)
    h_outer_tripcount = div_ceil(dims.H_per_shard, PSUM_SIZE)

    gate_and_up_proj_states = []
    for _ in range(2):
        outer_list = []
        for _ in range(N_PSUM_TILE):
            inner_list = []
            for _ in range(gup_n_tile):
                tile = nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.sbuf)
                inner_list.append(tile)
            outer_list.append(inner_list)
        gate_and_up_proj_states.append(outer_list)

    for gate_or_up in range(2):
        for b_tile_idx in range(N_PSUM_TILE):
            for i_tile_idx in range(gup_n_tile):
                num_i = min(TILE_SIZE, dims.I_TP - TILE_SIZE * i_tile_idx)
                psum_acc = nl.ndarray((num_i, free_size), dtype=nl.float32, buffer=nl.psum)
                for h_outer_idx in range(h_outer_tripcount):
                    for h_inner_idx in range(h_inner_tripcount):
                        h_lin_idx = h_outer_idx * h_inner_tripcount + h_inner_idx
                        if h_lin_idx < linearized_tripcount:
                            h_offset_in_shard = PSUM_SIZE * h_outer_idx + TILE_SIZE * h_inner_idx
                            num_h = min(TILE_SIZE, dims.H_per_shard - h_offset_in_shard)
                            nisa.nc_matmul(
                                dst=psum_acc[0:num_i, 0:free_size],
                                stationary=gup_weights[h_outer_idx][h_inner_idx][
                                    0:num_h, gate_or_up, i_tile_idx * TILE_SIZE : i_tile_idx * TILE_SIZE + num_i
                                ],
                                moving=block_hidden_states_T[h_outer_idx][h_inner_idx][
                                    0:num_h, b_tile_idx, 0:free_size
                                ],
                            )

                sbuf_local = nl.ndarray((num_i, free_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sbuf_local, src=psum_acc)

                # LNC all-reduce via sendrecv
                recv_buf = nl.ndarray((num_i, free_size), dtype=nl.float32, buffer=nl.sbuf)
                sendrecv(
                    src=sbuf_local[0:num_i, 0:free_size],
                    dst=recv_buf[0:num_i, 0:free_size],
                    send_to_rank=(1 - shard_id),
                    recv_from_rank=(1 - shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(
                    dst=gate_and_up_proj_states[gate_or_up][b_tile_idx][i_tile_idx][0:num_i, 0:free_size],
                    data1=sbuf_local,
                    op=nl.add,
                    data2=recv_buf,
                )

                if gate_up_activations_T != None:
                    offset = (
                        block_idx * 2 * dims.I_TP * dims.B
                        + gate_or_up * dims.I_TP * dims.B
                        + i_tile_idx * TILE_SIZE * dims.B
                        + b_tile_idx * free_size
                    )
                    nisa.dma_copy(
                        dst=gate_up_activations_T.ap(pattern=[[dims.B, num_i], [1, free_size]], offset=offset),
                        src=gate_and_up_proj_states[gate_or_up][b_tile_idx][i_tile_idx][0:num_i, 0:free_size],
                    )

    return gate_and_up_proj_states


def load_old_block_shard(output, token_indices, dims, cfg, shard_id):
    """Load the old block from the output tensor."""
    h_offset = dims.H_per_shard * shard_id
    block_old = []
    for _ in range(dims.NUM_TILES):
        tile = nl.ndarray((TILE_SIZE, dims.H_per_shard), dtype=cfg.compute_dtype, buffer=nl.sbuf)
        block_old.append(tile)

    for tile_idx in range(dims.NUM_TILES):
        if cfg.skip_dma.skip_token:
            nisa.memset(dst=block_old[tile_idx], value=0)
        tmp_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_idx, src=token_indices[0:TILE_SIZE, tile_idx])
        nisa.dma_copy(
            dst=block_old[tile_idx][0:TILE_SIZE, 0 : dims.H_per_shard],
            src=output.ap(
                pattern=[[dims.H, TILE_SIZE], [1, dims.H_per_shard]],
                offset=h_offset,
                vector_offset=tmp_idx,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if cfg.skip_dma.skip_token else oob_mode.error,
        )
    return block_old


def compute_block_output_shard(
    intermediate_states, dp_weights, expert_affinity, block_old, down_activations, block_idx, dims, cfg, shard_id
):
    """
    Compute the new block output with down projection and expert affinity adjustment.

    Performs down projection matmul, optionally applies POST_SCALE expert affinity,
    and accumulates with previous output if is_tensor_update_accumulating is True.

    Args:
        intermediate_states: List of [TILE_SIZE, B] tensors (SiLU(gate) * up result).
        dp_weights: List of [TILE_SIZE, H_per_shard] down projection weight tensors.
        expert_affinity: List of [TILE_SIZE, 1] affinity tensors, or None.
        block_old: List of [TILE_SIZE, H_per_shard] previous output tensors, or None.
        down_activations: Optional [N, B, H] tensor for checkpointing.
        block_idx: Current block index for activation storage offset.
        dims: DimensionSizes configuration object.
        cfg: Configs object with scaling_mode and other settings.
        shard_id: Current shard/core ID for H offset calculation.

    Returns:
        List[nl.ndarray]: List of [TILE_SIZE, H_per_shard] output tensors in SBUF.
    """
    gup_n_tile = div_ceil(dims.I_TP, TILE_SIZE)
    H_NUM_PSUM_TILES = div_ceil(dims.H_per_shard, PSUM_SIZE)
    h_outer_upper = div_ceil(dims.H_per_shard, TOTAL_PSUM_SIZE)

    block_new = []
    for _ in range(dims.NUM_TILES):
        tile = nl.ndarray((TILE_SIZE, dims.H_per_shard), dtype=cfg.io_dtype, buffer=nl.sbuf)
        block_new.append(tile)

    for tile_idx in range(dims.NUM_TILES):
        for h_outer_idx in range(h_outer_upper):
            down_proj = []
            for _ in range(N_PSUM_BANKS):
                tile = nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=nl.float32, buffer=nl.psum)
                down_proj.append(tile)
            for h_bank_idx in range(N_PSUM_BANKS):
                h_psum_idx = N_PSUM_BANKS * h_outer_idx + h_bank_idx
                if h_psum_idx < H_NUM_PSUM_TILES:
                    for i_tile_idx in range(gup_n_tile):
                        num_i = min(TILE_SIZE, dims.I_TP - TILE_SIZE * i_tile_idx)
                        num_h = min(
                            PSUM_SIZE, dims.H_per_shard - (TOTAL_PSUM_SIZE * h_outer_idx + PSUM_SIZE * h_bank_idx)
                        )
                        nisa.nc_matmul(
                            dst=down_proj[h_bank_idx][0:TILE_SIZE, 0:num_h],
                            stationary=intermediate_states[i_tile_idx][
                                0:num_i, TILE_SIZE * tile_idx : TILE_SIZE * tile_idx + TILE_SIZE
                            ],
                            moving=dp_weights[i_tile_idx][
                                0:num_i,
                                TOTAL_PSUM_SIZE * h_outer_idx + PSUM_SIZE * h_bank_idx : TOTAL_PSUM_SIZE * h_outer_idx
                                + PSUM_SIZE * h_bank_idx
                                + num_h,
                            ],
                        )

                    o_f_offset = TOTAL_PSUM_SIZE * h_outer_idx + PSUM_SIZE * h_bank_idx
                    num_h = min(PSUM_SIZE, dims.H_per_shard - o_f_offset)

                    if cfg.is_tensor_update_accumulating:
                        if cfg.scaling_mode == ExpertAffinityScaleMode.POST_SCALE and expert_affinity != None:
                            scaled = nl.ndarray((TILE_SIZE, PSUM_SIZE), dtype=cfg.compute_dtype, buffer=nl.sbuf)
                            nisa.tensor_scalar(
                                dst=scaled[0:TILE_SIZE, 0:num_h],
                                data=down_proj[h_bank_idx][0:TILE_SIZE, 0:num_h],
                                op0=nl.multiply,
                                operand0=expert_affinity[tile_idx][0:TILE_SIZE, 0],
                            )
                            nisa.tensor_tensor(
                                dst=block_new[tile_idx][0:TILE_SIZE, o_f_offset : o_f_offset + num_h],
                                data1=block_old[tile_idx][0:TILE_SIZE, o_f_offset : o_f_offset + num_h],
                                op=nl.add,
                                data2=scaled[0:TILE_SIZE, 0:num_h],
                            )
                        else:
                            nisa.tensor_tensor(
                                dst=block_new[tile_idx][0:TILE_SIZE, o_f_offset : o_f_offset + num_h],
                                data1=block_old[tile_idx][0:TILE_SIZE, o_f_offset : o_f_offset + num_h],
                                op=nl.add,
                                data2=down_proj[h_bank_idx][0:TILE_SIZE, 0:num_h],
                            )
                    elif cfg.scaling_mode == ExpertAffinityScaleMode.POST_SCALE and expert_affinity != None:
                        nisa.tensor_scalar(
                            dst=block_new[tile_idx][0:TILE_SIZE, o_f_offset : o_f_offset + num_h],
                            data=down_proj[h_bank_idx][0:TILE_SIZE, 0:num_h],
                            op0=nl.multiply,
                            operand0=expert_affinity[tile_idx][0:TILE_SIZE, 0],
                        )
                    else:
                        nisa.tensor_copy(
                            dst=block_new[tile_idx][0:TILE_SIZE, o_f_offset : o_f_offset + num_h],
                            src=down_proj[h_bank_idx][0:TILE_SIZE, 0:num_h],
                        )

                    if down_activations != None:
                        h_offset = dims.H_per_shard * shard_id
                        offset = block_idx * dims.B * dims.H + tile_idx * TILE_SIZE * dims.H + h_offset + o_f_offset
                        nisa.dma_copy(
                            dst=down_activations.ap(pattern=[[dims.H, TILE_SIZE], [1, num_h]], offset=offset),
                            src=down_proj[h_bank_idx][0:TILE_SIZE, 0:num_h],
                        )
    return block_new


def store_block_output_shard(output, block_new, token_indices, dims, shard_id, skip_dma):
    """Store the computed block output in the output tensor."""
    h_offset = dims.H_per_shard * shard_id
    for tile_idx in range(dims.NUM_TILES):
        tmp_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_idx, src=token_indices[0:TILE_SIZE, tile_idx])
        nisa.dma_copy(
            dst=output.ap(
                pattern=[[dims.H, TILE_SIZE], [1, dims.H_per_shard]],
                offset=h_offset,
                vector_offset=tmp_idx,
                indirect_dim=0,
            ),
            src=block_new[tile_idx][0:TILE_SIZE, 0 : dims.H_per_shard],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )


@nki.jit(mode="trace", platform_target="trn2")
def blockwise_mm_baseline_shard_hidden(
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    block_size: int,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    gate_up_activations_T: nl.ndarray = None,
    down_activations: nl.ndarray = None,
    skip_dma: SkipMode = SkipMode(),
    compute_dtype: nki.dtype = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
) -> nl.ndarray:
    """
    Blockwise matrix multiplication kernel with hidden dimension sharding for MoE.

    Implements MoE layer computation with sharding over the hidden dimension (H),
    distributing H across multiple cores for parallel processing. Uses LNC all-reduce
    for gate/up projection results.

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I_TP: Intermediate size / tp degree

    Args:
        hidden_states (nl.ndarray): [T+1, H], Input hidden states. T+1 for padding token at index T.
        expert_affinities_masked (nl.ndarray): [(T+1) * E, 1], Expert affinities per token.
        gate_up_proj_weight (nl.ndarray): [E, H, 2, I_TP], Gate and up projection weights.
        down_proj_weight (nl.ndarray): [E, I_TP, H], Down projection weights.
        block_size (int): Tokens per block.
        token_position_to_id (nl.ndarray): [N * B], Token to block index mapping.
        block_to_expert (nl.ndarray): [N, 1], Block to expert mapping.
        gate_up_activations_T (nl.ndarray): Optional [N, 2, I_TP, B] for activation checkpointing.
        down_activations (nl.ndarray): Optional [N, B, H] for activation checkpointing.
        skip_dma (SkipMode): DMA skip configuration.
        compute_dtype (nki.dtype): Compute data type (default: bfloat16).
        is_tensor_update_accumulating (bool): Accumulate results over blocks (default: True).
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Scaling mode (default: POST_SCALE).

    Returns:
        output (nl.ndarray): [T+1, H], Output hidden states.

    Notes:
        - Hidden dimension H must be divisible by NUM_SHARDS (2 for LNC2)
        - down_proj_weight I_TP dimension must be divisible by 16
        - Uses LNC all-reduce for gate/up projection partial sums

    Pseudocode:
        output = zeros([T+1, H])
        for block_idx in range(N):
            token_ids = token_position_to_id[block_idx * B : (block_idx + 1) * B]
            expert_idx = block_to_expert[block_idx]

            # Load hidden states for this shard's H partition
            block_hidden = hidden_states[token_ids, shard_id * H_per_shard : (shard_id + 1) * H_per_shard]
            if PRE_SCALE:
                block_hidden *= expert_affinities[token_ids, expert_idx]

            # Transpose: [B, H_per_shard] -> [H_per_shard, B]
            block_hidden_T = transpose(block_hidden)

            # Gate and up projections with LNC all-reduce
            gate = matmul(block_hidden_T, gate_proj_weight[expert_idx])  # partial sum
            up = matmul(block_hidden_T, up_proj_weight[expert_idx])      # partial sum
            gate = all_reduce(gate)  # LNC sendrecv
            up = all_reduce(up)      # LNC sendrecv

            # Intermediate: SiLU(gate) * up
            intermediate = silu(gate) * up

            # Down projection (no all-reduce needed, each shard writes its H partition)
            down = matmul(intermediate, down_proj_weight[expert_idx, :, shard_H_slice])

            if POST_SCALE:
                down *= expert_affinities[token_ids, expert_idx]

            output[token_ids, shard_H_slice] += down
    """
    T, H = hidden_states.shape
    B = block_size
    _, _, _, I_TP = gate_up_proj_weight.shape
    E, I_TP_padded, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B

    kernel_assert(I_TP_padded % 16 == 0, "down_proj_weight I_TP must be divisible by 16")

    shard_id = nl.program_id(axis=0)
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=I_TP)
    dims.derive_all_dims()

    kernel_assert(H % dims.NUM_SHARDS == 0, f"Hidden dim must be divisible by {dims.NUM_SHARDS}")

    cfg = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=False,
        activation_function=None,
        is_quant=False,
        fuse_gate_and_up_load=False,
    )

    output = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    output_initialization_shard(output, dims, shard_id)

    for block_idx in nl.sequential_range(N):  # sequential_range for sequential HBM block access
        token_indices = load_token_indices(token_position_to_id, block_idx, B, dims.NUM_TILES)
        block_expert = load_block_expert(block_to_expert, block_idx)

        expert_affinity = None
        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
            expert_affinity = calculate_expert_affinities(
                expert_affinities_masked, token_indices, block_expert, E, dims.NUM_TILES, nl.float32, skip_dma
            )

        block_hidden_states = create_block_hidden_states(dims.H_per_shard, dims.NUM_TILES, compute_dtype)
        load_hidden_states_shard_with_scale(
            hidden_states, block_hidden_states, token_indices, expert_affinity, dims, cfg, shard_id
        )
        block_hidden_states_T = transpose_hidden_states(block_hidden_states, dims, compute_dtype)

        gup_weights = load_gate_up_proj_weights_shard(gate_up_proj_weight, block_expert, cfg, dims, shard_id)

        gate_and_up_proj_states = compute_gate_and_up_projections_shard(
            block_hidden_states_T, gup_weights, gate_up_activations_T, block_idx, dims, cfg, shard_id
        )

        intermediate_states = compute_intermediate_states(gate_and_up_proj_states, B, I_TP, compute_dtype)

        if is_tensor_update_accumulating:
            block_old = load_old_block_shard(output, token_indices, dims, cfg, shard_id)
        else:
            block_old = None

        dp_weights = load_down_proj_weight_shard(down_proj_weight, block_expert, cfg, dims, shard_id)

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            expert_affinity = calculate_expert_affinities(
                expert_affinities_masked, token_indices, block_expert, E, dims.NUM_TILES, nl.float32, skip_dma
            )

        block_new = compute_block_output_shard(
            intermediate_states,
            dp_weights,
            expert_affinity,
            block_old,
            down_activations,
            block_idx,
            dims,
            cfg,
            shard_id,
        )

        store_block_output_shard(output, block_new, token_indices, dims, shard_id, skip_dma)

        if dims.NUM_SHARDS == 2:
            nisa.core_barrier(output, (0, 1))
        else:
            nisa.core_barrier(output, (0,))

    return output
