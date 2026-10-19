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
"""Capacity-bounded all-to-all-v dispatch for MoE training."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup
from nki.isa.constants import oob_mode

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ._packed_metadata import _build_packed_regroup_indices
from .a2av_train_utils import (
    _A2AV_METADATA_NUM_ROWS,
    _LNC2_CORE_IDS,
    _exclusive_cumsum_u32,
    _start_count_exchange,
    _validate_metadata_extent,
    _validate_packed_indices_counts,
    _validate_trn3_a2av_group_size,
    _write_packed_a2av_v_metadata,
)

TILE_SIZE = 128
_FLOAT32_EXACT_INT_LIMIT = 1 << 24


def _load_expert_counts_and_rank_totals(
    expert_counts_hbm: nl.NkiTensor,
    num_peers: int,
    num_local_experts: int,
    hbm_name: str,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Load ``[peer, expert]`` counts and materialize ``[1, peer]`` totals."""
    expert_counts_sb = nl.ndarray(
        (num_peers, num_local_experts),
        dtype=nl.uint32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=expert_counts_sb, src=expert_counts_hbm)

    expert_counts_f = nl.ndarray(
        (num_peers, num_local_experts),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=expert_counts_f, src=expert_counts_sb)
    rank_counts_f = nl.ndarray(
        (num_peers, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_reduce(
        dst=rank_counts_f,
        data=expert_counts_f,
        op=nl.add,
        axis=1,
    )
    rank_counts_t_psum = nl.ndarray(
        (1, num_peers),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    nisa.nc_transpose(dst=rank_counts_t_psum, data=rank_counts_f)
    rank_counts_t_f = nl.ndarray(
        (1, num_peers),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=rank_counts_t_f, src=rank_counts_t_psum)
    rank_counts_sb = nl.ndarray(
        (1, num_peers),
        dtype=nl.uint32,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=rank_counts_sb, src=rank_counts_t_f)

    rank_counts_hbm = nl.ndarray(
        (1, num_peers),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name=hbm_name,
    )
    nisa.dma_copy(dst=rank_counts_hbm, src=rank_counts_sb)
    nisa.core_barrier(rank_counts_hbm, _LNC2_CORE_IDS)
    return expert_counts_sb, rank_counts_hbm


def _gather_expert_rows(
    source_buffer: nl.NkiTensor,
    row_indices: nl.NkiTensor,
    expert_buffer: nl.NkiTensor,
    source_ready=None,
    indices_ready=None,
) -> None:
    """Gather source-major receive rows into expert-major order."""
    rows, hidden = expert_buffer.shape
    source_rows = source_buffer.shape[0]
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
            nisa.memset(dst=indices_sb, value=source_rows)
            indices_read = nisa.dma_copy(dst=indices_sb[:tile_rows, :], src=indices_2d[start:end, :])
            if indices_ready is not None:
                indices_read.depends_on(indices_ready)
            data_sb = nl.ndarray((TILE_SIZE, hidden), dtype=source_buffer.dtype, buffer=nl.sbuf)
            payload_read = nisa.dma_copy(
                dst=data_sb,
                src=source_buffer.ap(
                    pattern=[[hidden, TILE_SIZE], [1, hidden]],
                    offset=0,
                    vector_offset=indices_sb,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            if source_ready is not None:
                payload_read.depends_on(source_ready)
            nisa.dma_copy(dst=expert_buffer[start:end, :], src=data_sb[:tile_rows, :])


@nki.jit
def permute_a2av(
    hidden_states: nl.NkiTensor,
    packed_source_indices: nl.NkiTensor,
    send_counts: nl.NkiTensor,
    recv_capacity: int,
    replica_group: ReplicaGroup,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Gather and exchange routed rows into a local-expert-major buffer.

    Summary:
        Dispatch ``N`` routed rows for Trn3 LNC=2 expert-parallel groups with
        ``8 <= EP <= 128`` and ``1 <= NLE <= 128``. One count exchange and one
        rank-level A2AV transfer all expert segments together. A local gather
        then places the received source-major rows in local-expert-major order.
        With ``NLE == 1``, the local gather reduces to a contiguous copy.

    Dimensions:
        N: Number of routed rows on the local rank.
        H: Hidden dimension.
        EP: Number of expert-parallel ranks.
        NLE: Number of local experts per rank.

    Args:
        hidden_states (nl.NkiTensor): ``[N, H]`` routed input rows.
        packed_source_indices (nl.NkiTensor): ``[N]`` int32 source-row
            permutation in destination-major order. Each destination segment
            must be grouped by local expert according to ``send_counts``.
        send_counts (nl.NkiTensor): ``[EP, NLE]`` int32/uint32 rows sent to
            each destination-local-expert pair. The full matrix must sum to
            ``N``.
        recv_capacity (int): Static row capacity of the expert-major output.
            The runtime receive total must fit this capacity, and the capacity
            must be below ``2**24`` so float32 prefix scans remain exact.
        replica_group (ReplicaGroup): EP replica group.

    Returns:
        recv_data (nl.NkiTensor): ``[recv_capacity, H]``. Active rows are
            local-expert-major, with sources contiguous inside each expert;
            unused tail rows are zero.
        metadata (nl.NkiTensor): ``[4, EP]`` uint32 logical rank-level send
            and receive counts/displacements, measured in elements.
        recv_counts (nl.NkiTensor): ``[EP, NLE]`` uint32 rows received from
            each source for each local expert.
        regroup_indices (nl.NkiTensor): int32 ``[recv_capacity]`` permutation
            mapping expert-major rows to source-major positions for ``NLE > 1``.
            For ``NLE == 1``, an unused int32 ``[1]`` zero placeholder.
        bwmm_counts (nl.NkiTensor): int32 ``[NLE]`` expert row totals with
            unused capacity assigned to the final local expert.

    Notes:
        ``send_counts[destination, local_expert]`` describes the contiguous
        expert segments in each destination segment of
        ``packed_source_indices``.
        ``N`` and ``recv_capacity`` must be below ``2**24`` so float32 prefix
        scans remain exact. The internal receive allocation is rounded to a
        full 128-row tile before copying into the exact returned capacity.

    Pseudocode:
        Exchange the ``[EP, NLE]`` count matrix once.
        Reduce expert counts to rank totals and pack rows by destination.
        A2AV all expert segments once using rank totals and displacements.
        Build inverse row indices and padded expert counts in the same launch.
        Gather the source-major receive rows into the expert-major output.
        Return expert-major rows, rank metadata, receive counts and layout metadata.
    """
    N, H = hidden_states.shape
    EP = send_counts.shape[0]
    NLE = send_counts.shape[1]
    dtype = hidden_states.dtype
    num_tiles = div_ceil(N, TILE_SIZE)
    num_programs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    program_id = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0

    kernel_assert(
        tuple(send_counts.shape) == (EP, NLE),
        f"send_counts must be [EP, NLE], got {tuple(send_counts.shape)}",
    )
    kernel_assert(
        send_counts.dtype in (nl.int32, nl.uint32),
        f"send_counts must be int32/uint32, got {send_counts.dtype}",
    )
    kernel_assert(
        1 <= NLE <= TILE_SIZE,
        f"NLE must be in [1, {TILE_SIZE}], got {NLE}",
    )
    kernel_assert(
        EP <= TILE_SIZE,
        f"EP must be <= {TILE_SIZE}, got {EP}",
    )
    _validate_trn3_a2av_group_size(EP)
    kernel_assert(
        1 <= recv_capacity < _FLOAT32_EXACT_INT_LIMIT,
        (f"recv_capacity must be in [1, 2**24), got {recv_capacity}"),
    )
    kernel_assert(
        N < _FLOAT32_EXACT_INT_LIMIT,
        f"N must be below 2**24, got {N}",
    )
    _validate_metadata_extent(N, H, "hidden_states")
    _validate_metadata_extent(recv_capacity, H, "recv_data")

    recv_expert_counts_hbm = _start_count_exchange(
        send_counts,
        EP,
        replica_group,
        counts_per_peer=NLE,
    )
    _, send_rank_counts_hbm = _load_expert_counts_and_rank_totals(
        send_counts,
        EP,
        NLE,
        hbm_name="packed_send_rank_counts",
    )
    _validate_packed_indices_counts(
        packed_source_indices,
        send_rank_counts_hbm,
        N,
        EP,
    )
    send_rank_counts_sb, send_rank_displs_sb = _exclusive_cumsum_u32(
        send_rank_counts_hbm,
        EP,
    )

    send_hbm = nl.ndarray(
        (N, H),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="packed_dispatch_send",
    )
    pack_stores = []
    for tile_idx in nl.affine_range(num_tiles):
        if tile_idx % num_programs == program_id:
            start = tile_idx * TILE_SIZE
            end = min(start + TILE_SIZE, N)
            tile_rows = end - start
            gather_indices = nl.ndarray(
                (tile_rows, 1),
                dtype=nl.uint32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=gather_indices,
                src=packed_source_indices[start:end].reshape((tile_rows, 1)),
            )
            tile_sb = nl.ndarray(
                (tile_rows, H),
                dtype=dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=tile_sb,
                src=hidden_states.ap(
                    pattern=[[N, tile_rows], [1, H]],
                    offset=0,
                    vector_offset=gather_indices,
                    indirect_dim=0,
                ),
            )
            pack_stores.append(nisa.dma_copy(dst=send_hbm[start:end, :], src=tile_sb))
    packed_ready = nisa.core_barrier(send_hbm, _LNC2_CORE_IDS)
    for store in pack_stores:
        packed_ready.depends_on(store)

    _, recv_rank_counts_hbm = _load_expert_counts_and_rank_totals(
        recv_expert_counts_hbm,
        EP,
        NLE,
        hbm_name="packed_recv_rank_counts",
    )
    recv_rank_counts_sb, recv_rank_displs_sb = _exclusive_cumsum_u32(
        recv_rank_counts_hbm,
        EP,
    )
    if NLE == 1:
        recv_total_rows_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=recv_total_rows_sb,
            data1=recv_rank_displs_sb[0:1, EP - 1 : EP],
            data2=recv_rank_counts_sb[0:1, EP - 1 : EP],
            op=nl.add,
        )
        recv_full_tile_rows_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=recv_full_tile_rows_sb,
            data=recv_total_rows_sb,
            op0=nl.right_shift,
            operand0=7,
        )
        nisa.tensor_scalar(
            dst=recv_full_tile_rows_sb,
            data=recv_full_tile_rows_sb,
            op0=nl.left_shift,
            operand0=7,
        )
        recv_tail_rows_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=recv_tail_rows_sb,
            data1=recv_total_rows_sb,
            data2=recv_full_tile_rows_sb,
            op=nl.subtract,
        )
        recv_total_rows_reg = nisa.register_alloc()
        nisa.register_load(recv_total_rows_reg, recv_total_rows_sb)
        recv_tail_rows_reg = nisa.register_alloc()
        nisa.register_load(recv_tail_rows_reg, recv_tail_rows_sb)
        recv_capacity_full_rows = recv_capacity // TILE_SIZE * TILE_SIZE
        recv_capacity_tail_rows = recv_capacity - recv_capacity_full_rows
        if recv_capacity_tail_rows:
            recv_in_capacity_tail_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=recv_in_capacity_tail_sb,
                data=recv_total_rows_sb,
                op0=nl.greater,
                operand0=recv_capacity_full_rows,
            )
            recv_in_capacity_tail_reg = nisa.register_alloc()
            nisa.register_load(recv_in_capacity_tail_reg, recv_in_capacity_tail_sb)

    metadata_hbm = nl.ndarray(
        (_A2AV_METADATA_NUM_ROWS, EP),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_dispatch_meta",
    )
    metadata_writes = _write_packed_a2av_v_metadata(
        metadata_hbm,
        send_rank_counts_sb,
        send_rank_displs_sb,
        recv_rank_counts_sb,
        recv_rank_displs_sb,
        H,
        EP,
    )

    recv_hbm_capacity = div_ceil(recv_capacity, TILE_SIZE) * TILE_SIZE
    recv_hbm = nl.ndarray(
        (recv_hbm_capacity, H),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="packed_dispatch_recv",
    )
    zero_tile = nl.ndarray(
        (TILE_SIZE, H),
        dtype=dtype,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=zero_tile, value=0)
    zero_stores = []
    if NLE > 1:
        for tile_idx in nl.affine_range(recv_hbm_capacity // TILE_SIZE):
            if tile_idx % num_programs == program_id:
                tile_start = tile_idx * TILE_SIZE
                zero_stores.append(
                    nisa.dma_copy(
                        dst=recv_hbm[tile_start : tile_start + TILE_SIZE, :],
                        src=zero_tile,
                    )
                )
    else:
        # Parent NLE=1 path: zero the returned capacity and only the partial receive tile.
        recv_data = nl.ndarray((recv_capacity, H), dtype=dtype, buffer=nl.shared_hbm)
        for tile_idx in nl.affine_range(div_ceil(recv_capacity, TILE_SIZE)):
            if tile_idx % num_programs == program_id:
                tile_start = tile_idx * TILE_SIZE
                tile_end = min(tile_start + TILE_SIZE, recv_capacity)
                nisa.dma_copy(
                    dst=recv_data[tile_start:tile_end, :],
                    src=zero_tile[0 : tile_end - tile_start, :],
                )
        nisa.core_barrier(recv_data, _LNC2_CORE_IDS)

        def _zero_recv_tail(_):
            nisa.dma_copy(
                dst=recv_hbm.ap(pattern=[[H, TILE_SIZE], [1, H]], scalar_offset=recv_full_tile_rows_sb, indirect_dim=0),
                src=zero_tile,
                oob_mode=oob_mode.skip,
                name="zero_recv_partial_tile",
            )

        nl.fori_loop(0, recv_tail_rows_reg, _zero_recv_tail, step=TILE_SIZE)

    send_ready = nisa.core_barrier(send_hbm, _LNC2_CORE_IDS)
    send_ready.depends_on(packed_ready)
    zero_ready = nisa.core_barrier(recv_hbm, _LNC2_CORE_IDS)
    for store in zero_stores:
        zero_ready.depends_on(store)
    metadata_ready = nisa.core_barrier(metadata_hbm, _LNC2_CORE_IDS)
    for write in metadata_writes:
        metadata_ready.depends_on(write)

    if NLE > 1:
        # Rendezvous after each rank's packed payload, receive zeros, and metadata.
        gate_src = nl.ndarray((2, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="dispatch_ready_gate_src")
        gate_dst = nl.ndarray((2, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="dispatch_ready_gate_dst")
        gate_value = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gate_value, value=1.0)
        gate_store = nisa.dma_copy(dst=gate_src[program_id : program_id + 1, :], src=gate_value)
        gate_ready = nisa.core_barrier(gate_src, _LNC2_CORE_IDS)
        gate_ready.depends_on(gate_store, send_ready, zero_ready, metadata_ready)
        gate = ncc.all_reduce(
            srcs=[gate_src],
            dsts=[gate_dst],
            replica_group=replica_group,
            op=nl.add,
            name="dispatch_ready_gate",
        )
        gate.depends_on(gate_ready)
        gate_complete = nisa.core_barrier(gate_dst, _LNC2_CORE_IDS)
        gate_complete.depends_on(gate)

    transfer = ncc.all_to_all_v(
        srcs=[send_hbm],
        dsts=[recv_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata_hbm,
        recv_counts_known=True,
        has_rdispls=True,
        name="packed_dispatch",
    )
    if NLE > 1:
        transfer.depends_on(send_ready, zero_ready, metadata_ready, gate_complete)
    else:
        transfer.depends_on(send_ready, zero_ready, metadata_ready)
    receive_ready = nisa.core_barrier(recv_hbm, _LNC2_CORE_IDS)
    receive_ready.depends_on(transfer)

    recv_expert_counts_out = nl.ndarray(
        (EP, NLE),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
    )
    if program_id == 0:
        nisa.dma_copy(
            dst=recv_expert_counts_out,
            src=recv_expert_counts_hbm,
        )
    if NLE > 1:
        regroup_indices, bwmm_counts = _build_packed_regroup_indices(
            recv_expert_counts_hbm,
            recv_capacity,
            num_programs,
            program_id,
        )
    else:
        regroup_indices_2d = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        bwmm_counts_2d = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        if program_id == 0:
            value = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(dst=value, value=0)
            nisa.dma_copy(dst=regroup_indices_2d, src=value)
            nisa.memset(dst=value, value=recv_capacity)
            nisa.dma_copy(dst=bwmm_counts_2d, src=value)
        regroup_indices = regroup_indices_2d.reshape((1,))
        bwmm_counts = bwmm_counts_2d.reshape((1,))
    # Each core must see the complete map before reading another core's rows.
    indices_ready = nisa.core_barrier(regroup_indices, _LNC2_CORE_IDS)
    if NLE > 1:
        recv_data = nl.ndarray(
            (recv_capacity, H),
            dtype=dtype,
            buffer=nl.shared_hbm,
        )
        _gather_expert_rows(recv_hbm, regroup_indices, recv_data, receive_ready, indices_ready)
    else:
        # Parent NLE=1 path: copy only the received prefix.
        copy_columns = H // num_programs if H % num_programs == 0 else H
        copy_column_offset = program_id * copy_columns if H % num_programs == 0 else 0

        def _copy_recv_tile(row_offset):
            nisa.dma_copy(
                dst=recv_data.ap(
                    pattern=[[H, TILE_SIZE], [1, copy_columns]],
                    offset=copy_column_offset,
                    scalar_offset=row_offset,
                    indirect_dim=0,
                ),
                src=recv_hbm.ap(
                    pattern=[[H, TILE_SIZE], [1, copy_columns]],
                    offset=copy_column_offset,
                    scalar_offset=row_offset,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
                name="copy_recv_tile",
            )

        nl.fori_loop(0, recv_total_rows_reg, _copy_recv_tile, step=TILE_SIZE)
        if recv_capacity_tail_rows:
            # A scalar OOB DMA skips the whole tile, so copy the final capacity fragment explicitly.

            def _copy_recv_capacity_tail(_):
                nisa.dma_copy(
                    dst=recv_data[recv_capacity_full_rows:recv_capacity, :],
                    src=recv_hbm[recv_capacity_full_rows:recv_capacity, :],
                    name="copy_recv_capacity_tail",
                )

            nl.fori_loop(0, recv_in_capacity_tail_reg, _copy_recv_capacity_tail)

    # Collective metadata must be internal, separate from the returned tensor.
    metadata_out = nl.ndarray(
        (_A2AV_METADATA_NUM_ROWS, EP),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
    )
    if program_id == 0:
        nisa.dma_copy(dst=metadata_out, src=metadata_hbm)
    nisa.core_barrier(recv_data, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata_out, _LNC2_CORE_IDS)
    nisa.core_barrier(recv_expert_counts_out, _LNC2_CORE_IDS)
    # Metadata has physical 2D allocations for CoreBarrier and public 1D views.
    nisa.core_barrier(bwmm_counts, _LNC2_CORE_IDS)
    return recv_data, metadata_out, recv_expert_counts_out, regroup_indices, bwmm_counts
