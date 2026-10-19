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
"""Capacity-bounded all-to-all-v combine and unpermute for MoE training."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup
from nki.isa.constants import oob_mode

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from .a2av_train_utils import (
    _A2AV_METADATA_NUM_ROWS,
    _LNC2_CORE_IDS,
    _exclusive_cumsum_u32,
    _validate_metadata_extent,
    _validate_packed_indices_counts,
    _validate_trn3_a2av_group_size,
    _write_packed_a2av_v_metadata,
)

TILE_SIZE = 128


def _scatter_expert_rows(
    data: nl.NkiTensor,
    row_indices: nl.NkiTensor,
    send_buffer: nl.NkiTensor,
):
    """Scatter a full row permutation directly into the communication buffer."""
    rows, hidden = data.shape
    num_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    program_id = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    num_tiles = div_ceil(rows, TILE_SIZE)
    tiles_per_program = div_ceil(num_tiles, num_programs)
    first_tile = program_id * tiles_per_program
    indices_2d = row_indices.reshape((rows, 1))
    for tile_offset in nl.sequential_range(tiles_per_program):
        tile_index = min(first_tile + tile_offset, num_tiles - 1)
        if first_tile + tile_offset < num_tiles:
            start = tile_index * TILE_SIZE
            end = min(start + TILE_SIZE, rows)
            tile_rows = end - start
            indices_sb = nl.ndarray((TILE_SIZE, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.memset(dst=indices_sb, value=rows)
            nisa.dma_copy(dst=indices_sb[:tile_rows, :], src=indices_2d[start:end, :])
            data_sb = nl.ndarray((TILE_SIZE, hidden), dtype=data.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=data_sb[:tile_rows, :], src=data[start:end, :])
            nisa.dma_copy(
                dst=send_buffer.ap(
                    pattern=[[hidden, TILE_SIZE], [1, hidden]],
                    offset=0,
                    vector_offset=indices_sb,
                    indirect_dim=0,
                ),
                src=data_sb,
                oob_mode=oob_mode.skip,
            )


@nki.jit
def unpermute_a2av(
    output: nl.NkiTensor,
    packed_source_indices: nl.NkiTensor,
    send_counts: nl.NkiTensor,
    recv_counts: nl.NkiTensor,
    replica_group: ReplicaGroup,
    regroup_indices: nl.NkiTensor = None,
    num_local_experts: int = 1,
) -> nl.NkiTensor:
    """Return packed expert output to origin ranks and restore routed rows.

    Summary:
        Reverses packed dispatch for Trn3 LNC=2 by sending only active rows
        from ``[C, H]``, receiving exactly ``N`` routed rows with explicit
        displacements, and restoring routed-row order. Supports ``EP >= 8`` in
        complete four-rank device groups and count totals below ``2**24`` so
        float32 prefix scans remain exact.

    This reverses :func:`permute_a2av`. Dispatch receive counts and
    displacements become combine send metadata; dispatch send counts and
    displacements become explicit combine receive metadata. ``packed_source_indices`` must be a permutation of the routed-row IDs:
    indirect DMA stores are not atomic duplicate-safe scatter-add operations.
    The dispatch routing vector satisfies this invariant. The kernel requires
    an LNC=2 launch and a sequential
    replica-group rank list that contains all four ranks from each of at least
    two participating devices.

    Dimensions:
        N:  Number of original local routed rows.
        H:  Hidden dimension.
        EP: Number of expert-parallel ranks.
        C:  Static packed expert-output capacity; may be smaller than N when
            the local dispatch receive total fits C.

    Args:
        output (nl.NkiTensor): [C, H]@HBM packed expert output.
        packed_source_indices (nl.NkiTensor): [N]@HBM, int32 permutation
            mapping packed return rows to original routed-row indices, in the
            same destination-major order used for dispatch.
        send_counts (nl.NkiTensor): [1, EP]@HBM original dispatch send counts.
        recv_counts (nl.NkiTensor): [1, EP]@HBM original dispatch receive counts.
        replica_group (ReplicaGroup): EP replica group.
        regroup_indices (nl.NkiTensor): Required int32/uint32 [C] permutation
            for ``num_local_experts > 1``. Maps each expert-major row to its
            source-major send-buffer position. Every position must occur once.
        num_local_experts (int): Number of local experts, in [1, 128]. Values
            above one select expert-major input and internal row scatter.
            The default retains the source-major input contract.

    Returns:
        result (nl.NkiTensor): [N, H]@HBM in original routed-row order.

    Notes:
        Active routing entries must assign every routed-row index exactly once.
        For multiple local experts, the inverse indices returned by dispatch
        restore source-major order directly in the communication send buffer.
        The unused capacity tail must also be covered by the permutation.
        Every count and displacement multiplied by ``H`` must fit uint32
        because A2AV metadata is measured in elements.

    Pseudocode:
        Prefix-sum dispatch send and receive counts.
        Reverse dispatch metadata for combine A2AV.
        Scatter expert-major rows into the send buffer when NLE > 1.
        Send only the active prefix of packed expert output.
        Receive source-major returned rows into a packed N-row buffer.
        Scatter packed rows to their unique original routed-row indices.
        Return routed-row order.
    """
    C, H = output.shape
    N = packed_source_indices.shape[0]
    EP = send_counts.shape[1]
    dtype = output.dtype
    NUM_TILES = div_ceil(N, TILE_SIZE)

    _validate_packed_indices_counts(packed_source_indices, send_counts, N, EP)
    _validate_trn3_a2av_group_size(EP)
    kernel_assert(
        tuple(recv_counts.shape) == (1, EP),
        f"recv_counts must be (1, EP={EP}), got {tuple(recv_counts.shape)}",
    )
    kernel_assert(
        recv_counts.dtype in (nl.int32, nl.uint32),
        f"recv_counts must be int32/uint32, got {recv_counts.dtype}",
    )
    kernel_assert(C >= 1, f"output capacity must be >= 1, got C={C}")
    kernel_assert(
        1 <= num_local_experts <= TILE_SIZE,
        f"num_local_experts must be in [1, {TILE_SIZE}], got {num_local_experts}",
    )
    if num_local_experts > 1:
        kernel_assert(regroup_indices is not None, "regroup_indices is required when num_local_experts > 1")
        kernel_assert(regroup_indices.shape == (C,), f"regroup_indices must have shape ({C},)")
        kernel_assert(
            regroup_indices.dtype in (nl.int32, nl.uint32),
            "regroup_indices must be int32/uint32",
        )
    _validate_metadata_extent(C, H, "output")
    _validate_metadata_extent(N, H, "result")

    send_counts_sb, send_displs_sb = _exclusive_cumsum_u32(send_counts, EP)
    recv_counts_sb, recv_displs_sb = _exclusive_cumsum_u32(recv_counts, EP)

    metadata = nl.ndarray(
        (_A2AV_METADATA_NUM_ROWS, EP),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_combine_meta",
    )
    _write_packed_a2av_v_metadata(
        metadata,
        recv_counts_sb,
        recv_displs_sb,
        send_counts_sb,
        send_displs_sb,
        H,
        EP,
    )
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)

    output_hbm = nl.ndarray((C, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_combine_send")
    if num_local_experts > 1:
        _scatter_expert_rows(output, regroup_indices, output_hbm)
    else:
        nisa.dma_copy(dst=output_hbm, src=output)
    # Reverse receive counts sum to N; C bounds only the local send buffer.
    got_hbm = nl.ndarray((N, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_combine_recv")
    nisa.core_barrier(output_hbm, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)
    ncc.all_to_all_v(
        srcs=[output_hbm],
        dsts=[got_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata,
        recv_counts_known=True,
        has_rdispls=True,
    )
    nisa.core_barrier(got_hbm, _LNC2_CORE_IDS)

    n_prgs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    prg_id = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    result = nl.ndarray((N, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_combine_result")
    # The permutation gives each result row one writer across both cores.
    for tile_idx in nl.affine_range(NUM_TILES):
        if tile_idx % n_prgs == prg_id:
            start = tile_idx * TILE_SIZE
            end = min(start + TILE_SIZE, N)
            tile_rows = end - start
            scatter_indices = nl.ndarray((tile_rows, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=scatter_indices,
                src=packed_source_indices[start:end].reshape((tile_rows, 1)),
            )
            data_sb = nl.ndarray((tile_rows, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=data_sb, src=got_hbm[start:end, :])
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[N, tile_rows], [1, H]],
                    offset=0,
                    vector_offset=scatter_indices,
                    indirect_dim=0,
                ),
                src=data_sb,
            )
    nisa.core_barrier(result, _LNC2_CORE_IDS)
    return result
