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
"""Shared helpers for the a2av_train dispatch / combine kernels.

This module exposes module-private helpers used by ``permute_a2av`` and
``unpermute_a2av`` to build the inputs ``ncc.all_to_all_v`` expects:

- :func:`_start_count_exchange` exchanges one fixed-width count row per peer
  before dispatch packing so the compiler can overlap metadata communication
  with local DMA.
- :func:`_exclusive_cumsum_u32` computes per-peer row displacements in SBUF
  for collective metadata.
- :func:`_write_packed_a2av_v_metadata` writes explicit send and receive counts
  and displacements for capacity-bounded packed A2AV.
- :func:`_validate_packed_indices_counts` asserts the shared shape/dtype
  contract on ``packed_source_indices[N]`` and ``counts[1, EP]``.
  Called at entry from both ``permute_a2av`` and ``unpermute_a2av``.

All helpers are plain Python functions (not decorated with
``@nki.jit``) so they inline into the caller's tracing context.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ....core.utils.kernel_assert import kernel_assert

_UINT32_MAX = (1 << 32) - 1
_TRN3_LNC2_RANKS_PER_DEVICE = 4
_TRN3_A2AV_MIN_DEVICES = 2
_A2AV_METADATA_NUM_ROWS = 4
_A2AV_METADATA_ROW_HEIGHT = 1
_A2AV_SEND_COUNTS_ROW = 0
_A2AV_SEND_DISPLS_ROW = 1
_A2AV_RECV_COUNTS_ROW = 2
_A2AV_RECV_DISPLS_ROW = 3
_LNC2_CORE_IDS = (0, 1)


def _validate_trn3_a2av_group_size(EP: int) -> None:
    """Assert the rank-count portion of the Trn3 LNC=2 A2AV contract.

    Trn3 A2AV requires more than one participating device and complete
    four-rank membership for every LNC=2 device in a replica-group rank list.
    The caller must separately ensure that each rank list is sequential and
    aligned to device boundaries.

    Args:
        EP (int): Number of ranks in each A2AV replica-group rank list.

    Returns:
        None.
    """
    minimum_ranks = _TRN3_LNC2_RANKS_PER_DEVICE * _TRN3_A2AV_MIN_DEVICES
    kernel_assert(
        EP >= minimum_ranks,
        (
            "Trn3 LNC=2 A2AV requires more than one device, with "
            f"{_TRN3_LNC2_RANKS_PER_DEVICE} ranks per device; got EP={EP}, "
            f"minimum={minimum_ranks}"
        ),
    )
    kernel_assert(
        EP % _TRN3_LNC2_RANKS_PER_DEVICE == 0,
        (
            "Trn3 LNC=2 A2AV requires complete device membership, so EP must "
            f"be divisible by {_TRN3_LNC2_RANKS_PER_DEVICE}; got EP={EP}"
        ),
    )


def _validate_metadata_extent(
    row_capacity: int,
    hidden_size: int,
    tensor_name: str,
) -> None:
    """Assert that a row-major tensor extent fits A2AV uint32 metadata.

    Args:
        row_capacity (int): Static number of rows addressable by metadata.
        hidden_size (int): Elements per row.
        tensor_name (str): User-facing tensor name for the assertion message.

    Returns:
        None.
    """
    kernel_assert(
        row_capacity * hidden_size <= _UINT32_MAX,
        (
            f"{tensor_name} extent must fit uint32 A2AV element metadata, got "
            f"rows={row_capacity}, hidden_size={hidden_size}, "
            f"elements={row_capacity * hidden_size}, max={_UINT32_MAX}"
        ),
    )


def _exclusive_cumsum_u32(
    counts_hbm: nl.NkiTensor,
    EP: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Exclusive cumulative sum of ``counts_hbm`` in tokens.

    Produces the scatter-displacement table
    ``[0, counts[0], counts[0] + counts[1], ..., sum(counts[:EP - 1])]``,
    which is the per-destination row offset into the packed send buffer
    that ``ncc.all_to_all_v`` expects.

    Args:
        counts_hbm (nl.NkiTensor): [1, EP] int32/uint32 HBM tensor.
            Per-destination counts in tokens.
        EP (int): collective world size (static).

    Returns:
        counts_sb (nl.NkiTensor): [1, EP] uint32 SBUF. Raw counts (in tokens)
            loaded from ``counts_hbm``; returned so callers can reuse
            without a second DMA.
        sdispls_sb (nl.NkiTensor): [1, EP] uint32 SBUF. Exclusive cumulative
            sum (scatter displacements) in tokens.

    Notes:
        ``tensor_tensor_scan`` requires float32 operands, so counts are
        cast through float32 and back to uint32. This is safe for the
        count ranges used in practice (sums up to hundreds of thousands
        of tokens fit comfortably in fp32 mantissa).
    """
    counts_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=counts_sb, src=counts_hbm)

    counts_f = nl.ndarray((1, EP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(counts_f, counts_sb)

    sdispls_f = nl.ndarray((1, EP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(sdispls_f, 0.0)
    init_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(init_sb, 0.0)
    # ones_sb acts as a multiplicative identity for the scan:
    #   acc = acc * 1 + counts[i]   → exclusive cumulative sum.
    ones_sb = nl.ndarray((1, EP - 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_sb, 1.0)
    nisa.tensor_tensor_scan(
        initial=init_sb,
        data0=ones_sb,
        op0=nl.multiply,
        data1=counts_f[0, : EP - 1],
        op1=nl.add,
        dst=sdispls_f[0, 1:],
    )

    sdispls_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(sdispls_sb, sdispls_f)
    return counts_sb, sdispls_sb


def _start_count_exchange(
    send_counts_hbm: nl.NkiTensor,
    EP: int,
    replica_group: ReplicaGroup,
    counts_per_peer: int = 1,
) -> nl.NkiTensor:
    """Exchange one fixed-width count row per peer."""
    exchange_shape = (EP, counts_per_peer)
    counts_send_hbm = nl.ndarray(
        exchange_shape,
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_count_send",
    )
    counts_recv_hbm = nl.ndarray(
        exchange_shape,
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_count_recv",
    )
    nisa.dma_copy(
        dst=counts_send_hbm,
        src=send_counts_hbm.reshape(exchange_shape),
    )
    nisa.core_barrier(counts_send_hbm, (0, 1))
    ncc.all_to_all(
        srcs=[counts_send_hbm],
        dsts=[counts_recv_hbm],
        replica_group=replica_group,
        collective_dim=0,
    )
    nisa.core_barrier(counts_recv_hbm, (0, 1))
    return counts_recv_hbm


def _write_packed_a2av_v_metadata(
    metadata_hbm: nl.NkiTensor,
    send_counts_sb: nl.NkiTensor,
    send_displs_sb: nl.NkiTensor,
    recv_counts_sb: nl.NkiTensor,
    recv_displs_sb: nl.NkiTensor,
    H: int,
    EP: int,
) -> tuple:
    """Write explicit packed A2AV metadata in element units.

    Args:
        metadata_hbm (nl.NkiTensor): [4, EP] uint32 shared-HBM destination.
        send_counts_sb (nl.NkiTensor): [1, EP] uint32 send counts in rows.
        send_displs_sb (nl.NkiTensor): [1, EP] uint32 send displacements in rows.
        recv_counts_sb (nl.NkiTensor): [1, EP] uint32 receive counts in rows.
        recv_displs_sb (nl.NkiTensor): [1, EP] uint32 receive displacements in
            rows.
        H (int): Elements per routed row.
        EP (int): Collective world size.

    Returns:
        Instruction handles for the four metadata writes.
    """
    send_counts_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=send_counts_el,
        data=send_counts_sb,
        op0=nl.multiply,
        operand0=H,
    )
    send_displs_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=send_displs_el,
        data=send_displs_sb,
        op0=nl.multiply,
        operand0=H,
    )
    recv_counts_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=recv_counts_el,
        data=recv_counts_sb,
        op0=nl.multiply,
        operand0=H,
    )
    recv_displs_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=recv_displs_el,
        data=recv_displs_sb,
        op0=nl.multiply,
        operand0=H,
    )
    send_counts_write = nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_SEND_COUNTS_ROW : _A2AV_SEND_COUNTS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=send_counts_el,
    )
    send_displs_write = nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_SEND_DISPLS_ROW : _A2AV_SEND_DISPLS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=send_displs_el,
    )
    recv_counts_write = nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_RECV_COUNTS_ROW : _A2AV_RECV_COUNTS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=recv_counts_el,
    )
    recv_displs_write = nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_RECV_DISPLS_ROW : _A2AV_RECV_DISPLS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=recv_displs_el,
    )
    return send_counts_write, send_displs_write, recv_counts_write, recv_displs_write


def _validate_packed_indices_counts(
    packed_source_indices: nl.NkiTensor,
    counts: nl.NkiTensor,
    N: int,
    EP: int,
) -> None:
    """Validate packed routing shapes and dtypes without reading device values.

    The caller must supply a permutation of [0, N), partitioned by counts.
    """
    kernel_assert(N >= 1, "routed row count must be positive")
    kernel_assert(tuple(packed_source_indices.shape) == (N,), "packed_source_indices must have shape [N]")
    kernel_assert(packed_source_indices.dtype == nl.int32, "packed_source_indices must be int32")
    kernel_assert(tuple(counts.shape) == (1, EP), "send_counts must have shape [1, EP]")
    kernel_assert(counts.dtype in (nl.int32, nl.uint32), "send_counts must be int32/uint32")
