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

"""Utility functions for MoE token generation kernels including expert affinity gathering and broadcasting."""

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...utils.tiled_range import TiledRange
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes


def gather_expert_affinities(
    expert_affinities_sb: nl.NkiTensor,
    expert_idx: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
) -> nl.NkiTensor:
    """
    Gathers expert affinities based on expert indices using local_gather operation.

    This function collects expert affinities for each token based on the expert indices.
    It handles different token count scenarios and performs necessary transpositions
    and local gather operations to prepare affinities for broadcasting.

    Args:
        expert_affinities_sb (nl.NkiTensor): [_pmax, E], Tensor containing expert affinities in SBUF.
        expert_idx (nl.NkiTensor): [T, K], Expert indices for each token.
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object containing T, K, _pmax and other constants.
        sbm (SbufManager): SBUF memory manager for allocation.

    Returns:
        gathered_affinities (nl.NkiTensor): [_pmax, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
            Gathered affinities tensor.

    Notes:
        - Uses different strategies for T <= 16 vs T > 16 for optimization
        - PARTITIONS_PER_GPSIMD_CORE = 16 (partitions per GPSIMD core)
        - PARTITIONS_PER_QUADRANT = 32 (partitions per quadrant)
    """
    # Hardware-specific constants
    PARTITIONS_PER_GPSIMD_CORE = 16  # Number of partitions per GPSIMD core
    kernel_assert(dims.K <= PARTITIONS_PER_GPSIMD_CORE, f"top_k {dims.K} exceeds {PARTITIONS_PER_GPSIMD_CORE}")
    kernel_assert(dims.E > 1, f"E={dims.E} must be > 1 for MoE (local_gather requires src_buffer_size > 1)")

    if dims.T <= PARTITIONS_PER_GPSIMD_CORE:
        # Optimized path for small token counts (T <= 16)

        # Convert expert indices to uint16 for local_gather operation
        expert_idx_u16 = sbm.alloc_stack((dims._pmax, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.memset(dst=expert_idx_u16, value=0)
        nisa.tensor_copy(dst=expert_idx_u16[0 : dims.T, 0 : dims.K], src=expert_idx[0 : dims.T, 0 : dims.K])

        # Prepare index values for gathering
        index_values = sbm.alloc_stack((dims._pmax, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.memset(dst=index_values, value=0)
        expert_indices_trans = sbm.alloc_stack(
            (PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE),
            dtype=nl.uint16,
            buffer=nl.sbuf,
            name="expert_indices_trans",
        )
        nisa.nc_transpose(
            dst=expert_indices_trans,
            data=expert_idx_u16[0:PARTITIONS_PER_GPSIMD_CORE, 0:PARTITIONS_PER_GPSIMD_CORE],
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(
            dst=index_values[0:PARTITIONS_PER_GPSIMD_CORE, 0:PARTITIONS_PER_GPSIMD_CORE],
            src=expert_indices_trans,
        )

    else:
        # Path for larger token counts (T > 16)
        # Use DMA_copy to avoid partition alignment problems
        active_channels = (dims.T + PARTITIONS_PER_GPSIMD_CORE - 1) // PARTITIONS_PER_GPSIMD_CORE

        # Convert expert indices to uint16 for local_gather operation
        expert_idx_u16 = sbm.alloc_stack((128, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.memset(dst=expert_idx_u16, value=0)
        nisa.tensor_copy(dst=expert_idx_u16[0 : dims.T, 0 : dims.K], src=expert_idx[0 : dims.T, 0 : dims.K])

        # Fill out 16 partition layout requirement in blocks of 16 partitions up to 128 partitions total
        index_values = sbm.alloc_stack(
            (dims._pmax, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, align=32
        )
        nisa.memset(dst=index_values, value=0)
        for channel_idx in range(active_channels):
            # Use DMA Transpose for better performance with larger token counts
            nisa.dma_transpose(
                dst=index_values.ap(
                    pattern=[
                        [PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
                        [1, 1],
                        [1, 1],
                        [1, PARTITIONS_PER_GPSIMD_CORE],
                    ],
                    offset=channel_idx * PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE,
                ),
                src=expert_idx_u16.ap(
                    pattern=[
                        [PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
                        [1, 1],
                        [1, 1],
                        [1, PARTITIONS_PER_GPSIMD_CORE],
                    ],
                    offset=channel_idx * PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE,
                ),
                dge_mode=nisa.dge_mode.none,
            )

    # Perform local gather to collect affinities based on indices
    gathered_affinities_sb = sbm.alloc_stack(
        (dims._pmax, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE),
        dtype=expert_affinities_sb.dtype,
        buffer=nl.sbuf,
        name="gathered_affinities_sb",
    )
    ga_sb_fdim = PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE

    # num_valid_indices is hard-coded due to compiler limitation
    nisa.memset(dst=gathered_affinities_sb, value=0.0)
    nisa.local_gather(
        dst=gathered_affinities_sb.ap([[ga_sb_fdim, dims._pmax], [1, ga_sb_fdim]]),
        src_buffer=expert_affinities_sb,
        index=index_values[:, :],
        num_elem_per_idx=1,
        num_valid_indices=ga_sb_fdim,
    )

    return gathered_affinities_sb


def broadcast_token_affinity(
    dst: nl.NkiTensor,
    gathered_affinities_sb: nl.NkiTensor,
    token_index: int,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
) -> nl.NkiTensor:
    """
    Broadcasts expert affinities for a specific token across partitions.

    This function takes gathered affinities and broadcasts the affinities for a specific
    token across all partitions, ensuring proper alignment with hardware constraints.

    Args:
        dst (nl.NkiTensor): Destination tensor for broadcasted affinities.
        gathered_affinities_sb (nl.NkiTensor): [_pmax, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
            Gathered affinities tensor.
        token_index (int): Index of the current token being processed (i_t).
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object containing K, _pmax and other constants.
        sbm (SbufManager): SBUF memory manager for allocation.

    Returns:
        broadcasted_affinities (nl.NkiTensor): [_pmax, K], Broadcasted token affinities ready for computation.

    Notes:
        - PARTITIONS_PER_GPSIMD_CORE = 16 (partitions per GPSIMD core)
        - PARTITIONS_PER_QUADRANT = 32 (partitions per quadrant)
        - Uses stream shuffle for proper partition alignment
    """
    # Hardware-specific constants
    PARTITIONS_PER_GPSIMD_CORE = 16  # Number of partitions per GPSIMD core
    PARTITIONS_PER_QUADRANT = 32  # Number of partitions per quadrant

    # Calculate partition and quadrant positions for the current token
    current_partition_channel = token_index % PARTITIONS_PER_GPSIMD_CORE  # Active Partition Channel 0..15
    current_quadrant_group = token_index // PARTITIONS_PER_QUADRANT  # Partition groups of 32
    current_quadrant_channel = token_index % PARTITIONS_PER_QUADRANT  # Active Quadrant Channel 0..31

    # Select token affinities from gathered data [T, PARTITIONS_PER_GPSIMD_CORE]
    token_affinities = gathered_affinities_sb[:, current_partition_channel, :]

    # Create shuffle mask for partition alignment
    shuffle_mask = [current_quadrant_channel] * PARTITIONS_PER_QUADRANT

    # Perform stream shuffle to align token affinities with partitions
    token_affinities_partition_aligned = sbm.alloc_stack(
        (dims._pmax, PARTITIONS_PER_GPSIMD_CORE),
        dtype=gathered_affinities_sb.dtype,
        buffer=nl.sbuf,
        name="token_affinities_partition_aligned",
    )
    nisa.nc_stream_shuffle(src=token_affinities, dst=token_affinities_partition_aligned, shuffle_mask=shuffle_mask)

    # Select the appropriate quadrant group and broadcast across all partitions
    quadrant_start = current_quadrant_group * PARTITIONS_PER_QUADRANT
    quadrant_end = quadrant_start + 1
    stream_shuffle_broadcast(src=token_affinities_partition_aligned[quadrant_start:quadrant_end, : dims.K], dst=dst)


def reshape_scale_for_mlp(scale_tensor: nl.NkiTensor):
    """
    Reshapes scale tensor for MLP operations by expanding and broadcasting.

    Args:
        scale_tensor (NkiTensor): Scale tensor to reshape.

    Returns:
        NkiTensor: Reshaped scale tensor with expanded dimension 0 and broadcasted to size 128.

    Notes:
        - Expands dimension 0 and broadcasts to partition size (128)
    """
    return scale_tensor.expand_dim(dim=0).broadcast(dim=0, size=128)


def load_all_expert_affinities(expert_affinities, expert_affinities_in_sbuf, T_total, dims, allocator):
    """Load all-expert affinities into SBUF, tiling along T in pmax-sized chunks.

    Args:
        expert_affinities: HBM or SBUF tensor (T, E).
        expert_affinities_in_sbuf: Whether affinities are already in SBUF.
        T_total: Total number of tokens.
        dims: MLPTKGConstantsDimensionSizes.
        allocator: SBUF allocator callable.

    Returns:
        expert_affinities_sb: SBUF tensor — (T, E) if single chunk, (pmax, num_chunks, E) if tiled.
        aff_num_tiles: Number of pmax-sized chunks along T.
    """
    pmax = dims._pmax
    aff_num_tiles = div_ceil(T_total, pmax)
    if expert_affinities_in_sbuf:
        return expert_affinities, aff_num_tiles

    aff_tile_T = min(T_total, pmax)
    if aff_num_tiles == 1:
        expert_affinities_sb = allocator((T_total, dims.E), dtype=expert_affinities.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=expert_affinities_sb[:T_total, : dims.E], src=expert_affinities[:T_total, : dims.E])
    else:
        expert_affinities_sb = allocator(
            (aff_tile_T, aff_num_tiles, dims.E), dtype=expert_affinities.dtype, buffer=nl.sbuf
        )
        for t_tile in TiledRange(T_total, aff_tile_T):
            nisa.dma_copy(
                dst=expert_affinities_sb[: t_tile.size, t_tile.index, : dims.E],
                src=expert_affinities[nl.ds(t_tile.start_offset, t_tile.size), : dims.E],
            )
    return expert_affinities_sb, aff_num_tiles


def get_all_expert_tile_affinities(expert_affinities_sb, aff_num_tiles, t_offset, current_tile_T, dims, allocator):
    """Get all-expert affinities for a T-tile, gathering from pmax-sized chunks if needed.

    Args:
        expert_affinities_sb: SBUF tensor from load_all_expert_affinities.
        aff_num_tiles: Number of pmax-sized affinity chunks.
        t_offset: Start offset of the current T-tile.
        current_tile_T: Size of the current T-tile.
        dims: MLPTKGConstantsDimensionSizes.
        allocator: SBUF allocator callable.

    Returns:
        SBUF tensor with affinities for this T-tile.
    """
    pmax = dims._pmax
    if aff_num_tiles == 1:
        return expert_affinities_sb

    aff_start = t_offset // pmax
    aff_count = div_ceil(current_tile_T, pmax)
    if aff_count == 1:
        result = allocator((current_tile_T, dims.E), dtype=expert_affinities_sb.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=result[:current_tile_T, : dims.E], src=expert_affinities_sb[:current_tile_T, aff_start, : dims.E]
        )
    else:
        result = allocator((pmax, aff_count, dims.E), dtype=expert_affinities_sb.dtype, buffer=nl.sbuf)
        for ai in range(aff_count):
            chunk_size = min(pmax, current_tile_T - ai * pmax)
            nisa.tensor_copy(
                dst=result[:chunk_size, ai, : dims.E],
                src=expert_affinities_sb[:chunk_size, aff_start + ai, : dims.E],
            )
    return result


def broadcast_all_expert_affinity(
    expert_affinities_sb,
    aff_num_tiles,
    expertIdx,
    t_offset,
    current_tile_T,
    io_dtype,
    identity_sb,
    dims,
    allocator,
    use_auto_alloc,
):
    """Broadcast per-token all-expert affinity to (pmax, T) for POST_SCALE element-wise scaling.

    Processes in pmax-sized chunks to support T > pmax.

    Args:
        expert_affinities_sb: SBUF tensor from load_all_expert_affinities.
        aff_num_tiles: Number of pmax-sized affinity chunks.
        expertIdx: Current expert index.
        t_offset: Start offset of the current T-tile.
        current_tile_T: Size of the current T-tile.
        io_dtype: Data type for the broadcast tensor.
        identity_sb: Pre-loaded identity matrix (or None if T > pmax).
        dims: MLPTKGConstantsDimensionSizes.
        allocator: SBUF allocator callable.
        use_auto_alloc: Whether auto allocation is enabled.

    Returns:
        expert_affinities_broadcast: SBUF tensor (pmax, current_tile_T).
    """
    pmax = dims._pmax
    expert_affinities_broadcast = allocator((pmax, current_tile_T), dtype=io_dtype, buffer=nl.sbuf)
    aff_chunks = div_ceil(current_tile_T, pmax)
    for ac in range(aff_chunks):
        chunk_t_start = ac * pmax
        chunk_size = min(pmax, current_tile_T - chunk_t_start)
        chunk_identity = (
            identity_sb
            if (identity_sb is not None and aff_chunks == 1)
            else nl.shared_identity_matrix(chunk_size, dtype=io_dtype)
        )

        chunk_affinity_cast = allocator((chunk_size, 1), dtype=io_dtype, buffer=nl.sbuf)
        if aff_num_tiles == 1:
            nisa.activation(
                dst=chunk_affinity_cast[:chunk_size, :],
                op=nl.copy,
                data=expert_affinities_sb[nl.ds(chunk_t_start, chunk_size), nl.ds(expertIdx, 1)],
            )
        else:
            aff_chunk_idx = (t_offset + chunk_t_start) // pmax
            nisa.activation(
                dst=chunk_affinity_cast[:chunk_size, :],
                op=nl.copy,
                data=expert_affinities_sb[:chunk_size, aff_chunk_idx, nl.ds(expertIdx, 1)],
            )

        chunk_psum = nl.ndarray(
            (1, chunk_size), dtype=nl.float32, buffer=nl.psum, address=None if use_auto_alloc else (0, 0)
        )
        nisa.nc_matmul(
            dst=chunk_psum[:, :],
            stationary=chunk_affinity_cast,
            moving=chunk_identity[:chunk_size, :chunk_size],
        )
        nisa.tensor_copy(
            src=chunk_psum[:, :],
            dst=expert_affinities_broadcast[:1, nl.ds(chunk_t_start, chunk_size)],
        )
    for partition_group_idx in range(4):
        nisa.nc_stream_shuffle(
            dst=expert_affinities_broadcast[nl.ds(32 * partition_group_idx, 32), :current_tile_T],
            src=expert_affinities_broadcast[:1, :current_tile_T],
            shuffle_mask=[0] * 32,
        )
    return expert_affinities_broadcast
