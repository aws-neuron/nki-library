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

"""Expert affinity masking utilities for all-expert MoE kernels."""

import os as _os

import nki.isa as nisa
import nki.language as nl
from nki.isa import engine
from nki.isa.constants import dge_mode

from ...utils.dma_names import dma_name, inst_name
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast

# Hardware constants
_PARTITION_ALIGNMENT = 32  # Partition alignment for tensor operations
_DGE_ALIGNMENT = 16  # DMA gather engine alignment requirement


_AFFINITY_DMA_ENGINE = _os.environ.get("VLLM_NEURON_MEGA_AFFINITY_DMA_ENGINE", "")


def mask_expert_affinities(
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    rank_id: nl.NkiTensor,
    E_L: int,
    T: int,
    K: int,
    io_dtype,
    mask_unselected_experts: bool,
    output_in_sbuf: bool,
    affinity_packed_expert_major: int = 0,
) -> nl.NkiTensor:
    """
    Mask expert affinities for all-expert MoE computation.

    affinity_packed_expert_major: 0 for the token-major [T, E] layout. Otherwise the number of tokens
    per rank (T_shard) of the expert-major layout (c512 CR-310882146 trick 5): rank r's T_shard rows
    hold its E x T_shard affinities expert-major (col = (e % EPR) * T_shard + token,
    row = r * T_shard + e // EPR, EPR = E // T_shard), so this rank's slice of expert e is T_shard
    contiguous values per rank: 128 descriptors instead of T single-element ones. Requires E_L == 1 and
    the HBM output path.

    Slices the global expert affinities to local experts based on rank_id, and optionally
    masks affinities by checking expert_index against each local expert (when mask_unselected_experts=True).

    Args:
        expert_affinities (nl.NkiTensor): [T, E], Global expert affinities in HBM.
        expert_index (nl.NkiTensor): [T, K], Top-K expert indices per token in SBUF.
        rank_id (nl.NkiTensor): [1, 1], Rank ID tensor in HBM, specifies which experts this rank processes.
        E_L (int): Number of local experts.
        T (int): Number of tokens.
        K (int): Top-K value.
        io_dtype: Data type for computation.
        mask_unselected_experts (bool): If True, apply masking based on expert_index.
        output_in_sbuf (bool): If True, output affinities in SBUF; otherwise in shared HBM.

    Returns:
        expert_affinities_masked (nl.NkiTensor): [T, E_L]@SBUF or [128_T, T/128, E_L]@SBUF or [T, E_L]@HBM

    Notes:
        - For T <= 128, returns 2D tensor [T, E_L]
        - For T > 128, returns 3D tiled tensor [T_par, n_T128_tiles, E_L]
        - Uses indirect DMA with scalar_offset for rank-based expert slicing
    """
    kernel_assert(
        output_in_sbuf or not mask_unselected_experts,
        f"mask_unselected_experts=True only supports output_in_sbuf=True, "
        f"but got {mask_unselected_experts=}, {output_in_sbuf=}",
    )

    # Load rank_id to SBUF
    rank_id_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(src=rank_id[0:1, 0:1], dst=rank_id_sbuf[0:1, 0:1], name=dma_name("moe_rank_id_load"))

    # Calculate expert offset: expert_offset = rank_id * E_L
    expert_offset_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=expert_offset_sbuf[0:1, 0:1],
        data=rank_id_sbuf[0:1, 0:1],
        op0=nl.multiply,
        operand0=E_L,
        name=inst_name("moe_expert_offset_mul"),
    )

    # Slice expert affinities based on rank_id, load to SBUF
    if output_in_sbuf:
        kernel_assert(
            not affinity_packed_expert_major, "the expert-major packed affinity slice needs the HBM output path"
        )
        expert_affinities_masked = _load_slice_affinities_sbuf(
            expert_affinities=expert_affinities,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            io_dtype=io_dtype,
        )
    # Slice expert affinities into HBM intermediate buffer
    elif affinity_packed_expert_major:
        kernel_assert(E_L == 1, "the expert-major packed affinity slice supports one local expert")
        expert_affinities_masked = _slice_affinities_hbm_expert_major(
            expert_affinities=expert_affinities,
            expert_offset_sbuf=expert_offset_sbuf,
            T=T,
            T_shard=affinity_packed_expert_major,
            io_dtype=io_dtype,
        )
    else:
        expert_affinities_masked = _slice_affinities_hbm(
            expert_affinities=expert_affinities,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            io_dtype=io_dtype,
        )

    # Apply masking based on expert_index when mask_unselected_experts=True
    if mask_unselected_experts:
        _apply_expert_index_mask(
            expert_affinities_masked=expert_affinities_masked,
            expert_index=expert_index,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            K=K,
            io_dtype=io_dtype,
        )

    return expert_affinities_masked


def _load_slice_affinities_sbuf(expert_affinities, expert_offset_sbuf, E_L, T, io_dtype):
    """Load and slice expert affinities from HBM to SBUF based on rank offset."""
    E = expert_affinities.shape[1]

    if T <= 128:
        # 2D output [T, E_L]
        expert_affinities_masked = nl.ndarray((T, E_L), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=expert_affinities.ap(
                pattern=[[E, T], [1, E_L]],
                offset=0,
                scalar_offset=expert_offset_sbuf,
                indirect_dim=1,
            ),
            dst=expert_affinities_masked[0:T, 0:E_L],
            dge_mode=(
                dge_mode.hwdge
                if (_AFFINITY_DMA_ENGINE and T % _DGE_ALIGNMENT == 0)
                else (dge_mode.unknown if T % _DGE_ALIGNMENT == 0 else dge_mode.swdge)
            ),
            # When the affinities come from a collective (pre-gather router), this DMA gets
            # hoisted to the top of the layer and then blocks on the CC semaphore. It runs on
            # GpSimd, which is also the engine that issues the KV gather-transposes, so the wait
            # occupies the engine on the critical path (~18 us measured). Pin it elsewhere.
            engine=getattr(nisa.engine, _AFFINITY_DMA_ENGINE) if _AFFINITY_DMA_ENGINE else nisa.engine.unknown,
            name=dma_name("moe_affinity_load"),
        )
    else:
        # 3D tiled output [T_par, n_T128_tiles, E_L] for T > 128
        T_par = nl.tile_size.pmax
        n_T128_tiles = div_ceil(T, T_par)
        last_tile_T = T % T_par
        has_partial_tile = last_tile_T != 0
        n_full_tiles = n_T128_tiles - 1 if has_partial_tile else n_T128_tiles

        expert_affinities_masked = nl.ndarray((T_par, n_T128_tiles, E_L), dtype=io_dtype, buffer=nl.sbuf)

        if n_full_tiles > 0:
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[E, T_par], [T_par * E, n_full_tiles], [1, E_L]],
                    offset=0,
                    scalar_offset=expert_offset_sbuf,
                    indirect_dim=1,
                ),
                dst=expert_affinities_masked[0:T_par, 0:n_full_tiles, 0:E_L],
                dge_mode=dge_mode.unknown,
                name=dma_name("moe_affinity_load_full"),
            )

        if has_partial_tile:
            partial_tile_offset = n_full_tiles * T_par * E
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[E, last_tile_T], [T_par * E, 1], [1, E_L]],
                    offset=partial_tile_offset,
                    scalar_offset=expert_offset_sbuf,
                    indirect_dim=1,
                ),
                dst=expert_affinities_masked[0:last_tile_T, n_full_tiles : n_full_tiles + 1, 0:E_L],
                dge_mode=dge_mode.unknown if last_tile_T % _DGE_ALIGNMENT == 0 else dge_mode.swdge,
                name=dma_name("moe_affinity_load_partial"),
            )

    return expert_affinities_masked


def _slice_affinities_hbm_expert_major(expert_affinities, expert_offset_sbuf, T, T_shard, io_dtype):
    """Rank slice of expert-major packed, EP-gathered affinities (see mask_expert_affinities).

    expert_affinities: [T, E] view of the gathered buffer. Rank r's rows r*T_shard .. r*T_shard+T_shard-1 hold its
    E x T_shard affinities expert-major: the block is the transposed [E, T_shard] tile, so expert e's T_shard values
    for rank r sit at element offset e * T_shard of block r. The slice is therefore one T_shard-element descriptor
    per rank; the element offset is formed in place in expert_offset_sbuf (= e, since E_L == 1).
    """
    E = expert_affinities.shape[1]
    n_ranks = T // T_shard
    epr = E // T_shard
    kernel_assert(
        T % T_shard == 0 and E % T_shard == 0 and epr & (epr - 1) == 0 and (E * T_shard) % _DGE_ALIGNMENT == 0,
        "packed affinity slice: bad shape",
    )
    off = expert_offset_sbuf
    nisa.tensor_scalar(dst=off, data=off, op0=nl.multiply, operand0=T_shard, name=inst_name("moe_aff_pk_off"))
    expert_affinities_masked = nl.ndarray((T, 1), dtype=io_dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        src=expert_affinities.ap(
            pattern=[[E * T_shard, n_ranks], [1, T_shard]],
            offset=0,
            scalar_offset=off,
            indirect_dim=1,
        ),
        dst=expert_affinities_masked.ap(pattern=[[T_shard, n_ranks], [1, T_shard]], offset=0),
        dge_mode=dge_mode.unknown,
        name=dma_name("moe_affinity_hbm_load"),
    )
    return expert_affinities_masked


def _slice_affinities_hbm(expert_affinities, expert_offset_sbuf, E_L, T, io_dtype):
    """Load and slice expert affinities to shared HBM with LNC sharding."""
    E = expert_affinities.shape[1]
    _, n_prgs, prg_id = get_verified_program_sharding_info()
    T_shard = T // 2 if n_prgs > 1 else T
    T_offset = T_shard * prg_id

    expert_affinities_masked = nl.ndarray((T, E_L), dtype=io_dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        src=expert_affinities.ap(
            pattern=[[E, T_shard], [1, E_L]],
            offset=T_offset * E,
            scalar_offset=expert_offset_sbuf,
            indirect_dim=1,
        ),
        dst=expert_affinities_masked[nl.ds(T_offset, T_shard), :],
        dge_mode=dge_mode.unknown if T % _DGE_ALIGNMENT == 0 else dge_mode.swdge,
        name=dma_name("moe_affinity_hbm_load"),
    )
    if n_prgs > 1:
        nisa.core_barrier(expert_affinities_masked, (0, 1), name=inst_name("moe_affinity_hbm_barrier"))

    return expert_affinities_masked


def _apply_expert_index_mask(
    expert_affinities_masked: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_offset_sbuf: nl.NkiTensor,
    E_L: int,
    T: int,
    K: int,
    io_dtype,
):
    """
    Apply masking to expert affinities based on expert_index.

    For each local expert, checks if it was selected in expert_index for each token.
    Zeros out affinities for experts not selected by each token.

    Args:
        expert_affinities_masked (nl.NkiTensor): [T, E_L], Affinities to mask in-place in SBUF.
        expert_index (nl.NkiTensor): [T, K], Top-K expert indices in SBUF.
        expert_offset_sbuf (nl.NkiTensor): [1, 1], Starting expert index for this rank in SBUF.
        E_L (int): Number of local experts.
        T (int): Number of tokens.
        K (int): Top-K value.
        io_dtype: Data type for computation.

    Notes:
        - Modifies expert_affinities_masked in-place
        - Iterates through each local expert and applies element-wise masking
    """
    T_32s = _PARTITION_ALIGNMENT * div_ceil(T, _PARTITION_ALIGNMENT)

    # Load expert_index to SBUF if it's in HBM
    if expert_index.buffer == nl.shared_hbm:
        expert_index_sbuf = nl.ndarray((T, K), dtype=expert_index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=expert_index[0:T, 0:K], dst=expert_index_sbuf[0:T, 0:K], name=dma_name("moe_expert_index_load")
        )
    else:
        expert_index_sbuf = expert_index

    # Broadcast expert_offset to [T_32s, K] for comparison
    expert_offset_broadcast = nl.ndarray((T_32s, 1), dtype=nl.uint32, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=expert_offset_sbuf[0:1, 0:1], dst=expert_offset_broadcast[0:T_32s, 0:1])

    expert_offset_f = nl.ndarray((T_32s, K), dtype=nl.float32, buffer=nl.sbuf)
    for k_idx in nl.affine_range(K):
        if k_idx % 2 == 0:
            nisa.tensor_copy(
                src=expert_offset_broadcast[0:T_32s, 0:1],
                dst=expert_offset_f[0:T_32s, k_idx : k_idx + 1],
                engine=engine.vector,
                name=inst_name(f"moe_expert_offset_bcast_copy_k{k_idx}"),
            )
        else:
            nisa.tensor_copy(
                src=expert_offset_broadcast[0:T_32s, 0:1],
                dst=expert_offset_f[0:T_32s, k_idx : k_idx + 1],
                engine=engine.scalar,
                name=inst_name(f"moe_expert_offset_bcast_copy_k{k_idx}"),
            )

    # For each local expert, mask affinities
    for expert_idx in nl.affine_range(E_L):
        # Check expert_index against current expert: [T, K] comparison
        expert_check = nl.ndarray((T, K), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=expert_check[0:T, 0:K],
            data1=expert_index_sbuf[0:T, 0:K],
            data2=expert_offset_f[0:T, 0:K],
            op=nl.equal,
            name=inst_name(f"moe_mask_expert_eq_e{expert_idx}"),
        )

        # Sum across K dimension to get match indicator [T, 1]
        expert_match = nl.ndarray((T, 1), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=expert_match,
            op=nl.add,
            data=expert_check,
            axis=1,
            name=inst_name(f"moe_mask_match_reduce_e{expert_idx}"),
        )

        # Multiply affinities by match indicator
        nisa.tensor_tensor(
            dst=expert_affinities_masked[0:T, expert_idx : expert_idx + 1],
            data1=expert_affinities_masked[0:T, expert_idx : expert_idx + 1],
            data2=expert_match[0:T, 0:1],
            op=nl.multiply,
            name=inst_name(f"moe_mask_affinity_mul_e{expert_idx}"),
        )

        # Increment expert_offset_f by 1 for next iteration
        nisa.tensor_scalar(
            dst=expert_offset_f[0:T, 0:K],
            data=expert_offset_f[0:T, 0:K],
            op0=nl.add,
            operand0=1,
            name=inst_name(f"moe_mask_offset_incr_e{expert_idx}"),
        )
