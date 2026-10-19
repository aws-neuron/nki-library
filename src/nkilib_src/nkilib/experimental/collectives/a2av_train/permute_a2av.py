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


@nki.jit
def permute_a2av(
    hidden_states: nl.NkiTensor,
    packed_source_indices: nl.NkiTensor,
    send_counts: nl.NkiTensor,
    recv_capacity: int,
    replica_group: ReplicaGroup,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Gather and exchange routed rows into one packed capacity buffer.

    Summary:
        Packs ``N`` local routed rows by destination rank, overlaps an
        ``EP``-element count exchange with local indirect DMA, and receives
        source-major rows into a static ``[C, H]`` buffer using explicit A2AV
        receive displacements. This experimental kernel is intended for Trn3
        LNC=2, ``EP >= 8`` in complete four-rank device groups, and row totals
        below ``2**24`` so the float32 prefix scan remains exact.

    ``send_counts`` must sum to ``N``. The kernel exchanges those counts as
    ``[EP, 1]``, computes source-ordered receive displacements, and uses explicit
    A2AV receive displacements. The caller must guarantee that the total rows
    received by every rank fit ``recv_capacity``. A2AV validates destination
    size against metadata at execution, but the kernel cannot resize the static
    output or recover from an undersized capacity. The kernel requires an
    LNC=2 launch and a sequential replica-group rank list that contains all
    four ranks from each of at least two participating devices.

    Dimensions:
        N:  number of local routed rows.
        H:  hidden dimension.
        EP: number of expert-parallel ranks.
        C:  static receive capacity (``recv_capacity``).

    Args:
        hidden_states (nl.NkiTensor): [N, H]@HBM routed input rows.
        packed_source_indices (nl.NkiTensor): [N]@HBM, int32 permutation of
            source rows in destination-major order. Every entry must be in
            [0, N), with peer segments defined by send_counts.
        send_counts (nl.NkiTensor): [1, EP]@HBM, int32/uint32 row counts.
        recv_capacity (int): Static number of rows in the packed receive output.
            For top-k routing with unique expert IDs and ``L`` local experts
            per destination rank, a dropless bound is
            ``global_token_count * min(top_k, L)``. This reduces to
            ``global_token_count`` when ``L == 1``. Duplicate expert IDs
            within one token invalidate this bound.
        replica_group (ReplicaGroup): EP replica group.

    Returns:
        recv_data (nl.NkiTensor): [C, H]@HBM. Source-major packed rows followed
            by deterministic zero padding.
        metadata (nl.NkiTensor): [4, EP]@HBM, uint32. Send counts/displacements
            and receive counts/displacements, all measured in elements.

    Notes:
        The API routes opaque rows and does not depend on a framework, model,
        or local-expert count. Consumers that require expert-major output must
        provide their own per-expert regrouping metadata.
        Every count and displacement multiplied by ``H`` must fit uint32
        because A2AV metadata is measured in elements.

    Pseudocode:
        Start all-to-all exchange of send_counts.
        Prefix-sum send_counts and pack hidden_states by destination rank.
        Prefix-sum the exchanged receive counts.
        Build explicit send and receive A2AV metadata.
        Zero the returned receive capacity.
        Run A2AV into tile-padded internal source-major receive intervals.
        Copy the runtime receive prefix in full 128-row transfers.
        Copy at most one zero-padded partial transfer, clipped to capacity.
        Return the packed rows and metadata.
    """
    N, H = hidden_states.shape
    EP = send_counts.shape[1]
    dtype = hidden_states.dtype
    NUM_TILES = div_ceil(N, TILE_SIZE)

    _validate_packed_indices_counts(packed_source_indices, send_counts, N, EP)
    _validate_trn3_a2av_group_size(EP)
    kernel_assert(
        recv_capacity >= 1,
        f"recv_capacity must be >= 1, got {recv_capacity}",
    )
    _validate_metadata_extent(N, H, "hidden_states")
    _validate_metadata_extent(recv_capacity, H, "recv_data")

    recv_counts_exchange_hbm = _start_count_exchange(send_counts, EP, replica_group)
    send_counts_sb, send_displs_sb = _exclusive_cumsum_u32(send_counts, EP)

    send_hbm = nl.ndarray((N, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_dispatch_send")
    n_prgs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    prg_id = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0

    # Each static tile gathers only live rows and writes a contiguous peer-packed tile.
    # The vector already encodes count-prefix displacements, including zero-count peers.
    for tile_idx in nl.affine_range(NUM_TILES):
        if tile_idx % n_prgs == prg_id:
            start = tile_idx * TILE_SIZE
            end = min(start + TILE_SIZE, N)
            tile_rows = end - start
            gather_indices = nl.ndarray((tile_rows, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=gather_indices,
                src=packed_source_indices[start:end].reshape((tile_rows, 1)),
            )
            tile_sb = nl.ndarray((tile_rows, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tile_sb,
                src=hidden_states.ap(
                    pattern=[[N, tile_rows], [1, H]],
                    offset=0,
                    vector_offset=gather_indices,
                    indirect_dim=0,
                ),
            )
            nisa.dma_copy(dst=send_hbm[start:end, :], src=tile_sb)
    nisa.core_barrier(send_hbm, _LNC2_CORE_IDS)

    # The receive-count DMA is the first packing dependency and requires no extra cross-core barrier.
    recv_counts_hbm = recv_counts_exchange_hbm.reshape((1, EP))
    recv_counts_sb, recv_displs_sb = _exclusive_cumsum_u32(recv_counts_hbm, EP)
    nisa.core_barrier(recv_counts_exchange_hbm, _LNC2_CORE_IDS)

    recv_total_rows_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=recv_total_rows_sb,
        data1=recv_displs_sb[0:1, EP - 1 : EP],
        data2=recv_counts_sb[0:1, EP - 1 : EP],
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

    metadata = nl.ndarray(
        (_A2AV_METADATA_NUM_ROWS, EP),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_dispatch_meta",
    )
    _write_packed_a2av_v_metadata(
        metadata,
        send_counts_sb,
        send_displs_sb,
        recv_counts_sb,
        recv_displs_sb,
        H,
        EP,
    )
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)

    # Keep the runtime partial receive tile in bounds without changing the returned capacity.
    recv_hbm_capacity = div_ceil(recv_capacity, TILE_SIZE) * TILE_SIZE
    recv_hbm = nl.ndarray(
        (recv_hbm_capacity, H),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="packed_dispatch_recv",
    )
    recv_data = nl.ndarray((recv_capacity, H), dtype=dtype, buffer=nl.shared_hbm)
    zero_tile = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(zero_tile, 0)
    for tile_idx in nl.affine_range(div_ceil(recv_capacity, TILE_SIZE)):
        if tile_idx % n_prgs == prg_id:
            tile_start = tile_idx * TILE_SIZE
            tile_end = min(tile_start + TILE_SIZE, recv_capacity)
            nisa.dma_copy(
                dst=recv_data[tile_start:tile_end, :],
                src=zero_tile[0 : tile_end - tile_start, :],
            )
    # Order both cores' zero initialization before either core copies received rows.
    nisa.core_barrier(recv_data, _LNC2_CORE_IDS)

    def _zero_recv_tail(_):
        nisa.dma_copy(
            dst=recv_hbm.ap(
                pattern=[[H, TILE_SIZE], [1, H]],
                scalar_offset=recv_full_tile_rows_sb,
                indirect_dim=0,
            ),
            src=zero_tile,
            oob_mode=oob_mode.skip,
            name="zero_recv_partial_tile",
        )

    nl.fori_loop(0, recv_tail_rows_reg, _zero_recv_tail, step=TILE_SIZE)
    nisa.core_barrier(recv_hbm, _LNC2_CORE_IDS)

    nisa.core_barrier(send_hbm, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)
    ncc.all_to_all_v(
        srcs=[send_hbm],
        dsts=[recv_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata,
        recv_counts_known=True,
        has_rdispls=True,
    )
    nisa.core_barrier(recv_hbm, _LNC2_CORE_IDS)

    metadata_out = nl.ndarray((_A2AV_METADATA_NUM_ROWS, EP), dtype=nl.uint32, buffer=nl.shared_hbm)

    # Keep counted-loop iterations equal across cores; partition each tile by columns.
    copy_columns = H // n_prgs if H % n_prgs == 0 else H
    copy_column_offset = prg_id * copy_columns if H % n_prgs == 0 else 0

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

    nl.fori_loop(
        0,
        recv_total_rows_reg,
        _copy_recv_tile,
        step=TILE_SIZE,
    )

    if recv_capacity_tail_rows:
        # A scalar OOB DMA skips the whole tile, so copy the final capacity fragment explicitly.

        def _copy_recv_capacity_tail(_):
            nisa.dma_copy(
                dst=recv_data[recv_capacity_full_rows:recv_capacity, :],
                src=recv_hbm[recv_capacity_full_rows:recv_capacity, :],
                name="copy_recv_capacity_tail",
            )

        nl.fori_loop(0, recv_in_capacity_tail_reg, _copy_recv_capacity_tail)

    nisa.dma_copy(dst=metadata_out, src=metadata)
    nisa.core_barrier(recv_data, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata_out, _LNC2_CORE_IDS)
    return recv_data, metadata_out
