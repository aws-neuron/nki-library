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
"""All-to-all-v combine plus unpermute kernel for MoE training.

Mirror of ``permute_a2av`` that re-permutes fixed-stride expert output into a packed send buffer, exchanges via
``ncc.all_to_all_v``, then scatter-adds received rows to original token positions.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup
from nki.isa.constants import oob_mode

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from .a2av_train_utils import (
    _exclusive_cumsum_u32,
    _validate_a2av_indices_counts,
    _write_a2av_v_metadata,
)

TILE_SIZE = 128


@nki.jit
def unpermute_a2av(
    output: nl.ndarray,
    send_indices: nl.ndarray,
    recv_counts: nl.ndarray,
    replica_group: ReplicaGroup,
) -> nl.ndarray:
    """All-to-all-v combine and unpermute to original token order.

    Accepts fixed-stride expert output (source ``s`` at rows ``[s*T, s*T + recv_counts[s])``),
    re-permutes into a packed send buffer using ``cumsum(recv_counts)`` offsets, builds the
    ``(4, EP)`` metadata, exchanges via ``ncc.all_to_all_v``, then scatter-adds received rows
    to original token positions via ``send_indices``.

    TODO: Specify intended usage range (e.g., T per rank, H, EP degree, top-k).

    Dimensions:
        T:  number of original local tokens (SP sharded).
        H:  hidden dimension.
        EP: number of expert-parallel ranks.

    Args:
        output (nl.ndarray): [EP*T, H]@HBM. Expert output in fixed-stride
            layout: source ``s`` contributes at rows
            ``[s*T, s*T + recv_counts[s])``.
        send_indices (nl.ndarray): [T, EP]@HBM, int32. MoE routing table
            shared with dispatch: ``send_indices[t, d]`` is the original
            local-token row that source rank ``d`` produces the ``t``-th
            contribution for. On combine we scatter-add the received rows
            into those original positions. Entries with value = T are
            skipped (via ``oob_mode.skip``).
        recv_counts (nl.ndarray): [1, EP]@HBM, int32/uint32. Original dispatch
            ``recv_counts`` — used as combine's send-counts (metadata row 0).
        replica_group (ReplicaGroup): EP replica group.

    Returns:
        result (nl.ndarray): [T, H]@HBM. Output in original token order.

    Notes:
        - Supports top-k > 1 because the scatter accumulates across all EP
          contributions.
        - With LNC=2, EP is partitioned across cores and partial results are
          combined via ``nisa.sendrecv`` before core 0 writes the final output.

    Pseudocode:
        rdispls = exclusive_cumsum(recv_counts)                # [EP]
        send_hbm = zeros((EP*T, H))
        for src_rank_idx in range(EP):
            for tile_idx in range(div_ceil(T, TILE_SIZE)):
                tile_start = tile_idx * TILE_SIZE
                tile_end = min(tile_start + TILE_SIZE, T)
                tile = output[src_rank_idx*T + tile_start : src_rank_idx*T + tile_end]
                dst_rows = arange(tile_start, tile_end) + rdispls[src_rank_idx]
                send_hbm[dst_rows] = tile
        metadata = build_metadata(recv_counts, rdispls, H, EP)  # [4, EP]
        got_hbm = all_to_all_v(send_hbm, metadata, replica_group)

        partial = zeros((T, H))  # per core shard
        for dest_rank_idx in shard(range(EP), across_cores):
            base = dest_rank_idx * T
            for tile_idx in range(div_ceil(T, TILE_SIZE)):
                idx = send_indices[tile_start:tile_end, dest_rank_idx]   # with OOB skip
                partial[idx] += got_hbm[base + tile_start : base + tile_end]
        result = all_reduce_across_cores(partial)   # via sendrecv
        return result  # only core 0 writes result
    """
    _, H = output.shape
    T = send_indices.shape[0]
    EP = send_indices.shape[1]
    dtype = output.dtype
    NUM_TILES = div_ceil(T, TILE_SIZE)

    # Shape / dtype validation.
    _validate_a2av_indices_counts(send_indices, recv_counts, T, EP)
    kernel_assert(
        tuple(output.shape) == (EP * T, H),
        f"output must be (EP*T={EP * T}, H={H}), got {tuple(output.shape)}",
    )

    # Step 1: exclusive cumsum of recv_counts -> scatter displacements.
    rc_sb, sdispls_sb, sdispls_hbm = _exclusive_cumsum_u32(recv_counts, EP)

    # Step 2: unpermute fixed-stride `output` into packed `send_hbm`.
    send_hbm = nl.ndarray((EP * T, H), dtype=dtype, buffer=nl.shared_hbm, name="comb_send_pk")
    sdispls_hbm_1d = sdispls_hbm.reshape((EP,))

    for src_rank_idx in nl.sequential_range(EP):
        for tile_idx in nl.affine_range(NUM_TILES):
            ts = tile_idx * TILE_SIZE
            te = min(ts + TILE_SIZE, T)
            tsize = te - ts

            tile_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            # Slice src to the actual tile width (tsize). The final tile is shorter
            # than TILE_SIZE when T is not a multiple of TILE_SIZE (e.g. T=64 when
            # the per-device token count b_local*s_local < 128), so a fixed
            # +TILE_SIZE src read has more elements than dst=tile_sb[0:tsize] and
            # dma_copy asserts "src and dst have the same number of elements"
            # (src=TILE_SIZE*H, dst=tsize*H). Downstream only consumes tsize rows.
            nisa.dma_copy(
                dst=tile_sb[0:tsize, :],
                src=output[src_rank_idx * T + ts : src_rank_idx * T + ts + tsize, :],
            )

            disp_tile = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=disp_tile,
                src=sdispls_hbm_1d.ap(pattern=[[0, TILE_SIZE], [1, 1]], offset=src_rank_idx),
            )
            iota_sb = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=iota_sb, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            row_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(data=iota_sb, op0=nl.add, operand0=ts, dst=row_idx)
            nisa.tensor_tensor(row_idx, row_idx, disp_tile, op=nl.add)

            nisa.dma_copy(
                dst=send_hbm.ap(
                    pattern=[[EP * T, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=row_idx,
                    indirect_dim=0,
                ),
                src=tile_sb,
                oob_mode=oob_mode.skip,
            )

    # Step 3: build metadata.
    metadata = nl.ndarray((4, EP), dtype=nl.uint32, buffer=nl.shared_hbm, name="comb_meta")
    _write_a2av_v_metadata(metadata, rc_sb, sdispls_sb, H, EP)

    # Step 4: all-to-all-v. Runtime handles recv_counts and recv_displs.
    got_hbm = nl.ndarray((EP * T, H), dtype=dtype, buffer=nl.shared_hbm, name="comb_got_pk")
    ncc.all_to_all_v(
        srcs=[send_hbm],
        dsts=[got_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata,
        recv_counts_known=False,
        has_rdispls=False,
    )

    nisa.core_barrier(got_hbm, (0, 1))

    """
    Step 5: scatter-add to original token positions.

    With LNC=2, partition EP across cores, accumulate into HBM
    per LNC, then all-reduce across cores via sendrecv.
    """
    n_prgs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    prg_id = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0

    ep_per_shard_0 = div_ceil(EP, n_prgs)
    ep_per_shard_1 = EP // n_prgs
    ep_per_shard = ep_per_shard_0 if prg_id == 0 else ep_per_shard_1
    d_start = 0 if prg_id == 0 else ep_per_shard_0
    max_ep_per_shard = ep_per_shard_0

    partial = nl.ndarray((T, H), dtype=dtype, buffer=nl.private_hbm, name="comb_partial")

    zero_tile_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(zero_tile_sb, 0)
    for tile_idx in nl.affine_range(NUM_TILES):
        ts = tile_idx * TILE_SIZE
        te = min(ts + TILE_SIZE, T)
        nisa.dma_copy(dst=partial[ts:te, :], src=zero_tile_sb[0 : te - ts, :])

    for d_off in nl.sequential_range(max_ep_per_shard):
        dest_rank_idx = min(d_start + d_off, EP - 1)
        base = dest_rank_idx * T
        for tile_idx in nl.sequential_range(NUM_TILES):
            ts = tile_idx * TILE_SIZE
            te = min(ts + TILE_SIZE, T)
            tsize = te - ts

            idx_sb = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(idx_sb, T)
            if d_off < ep_per_shard:
                nisa.dma_copy(
                    dst=idx_sb[0:tsize, 0:1],
                    src=send_indices[ts:te, dest_rank_idx : dest_rank_idx + 1],
                )

            data_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            # Slice both dst and src to tsize: on the final partial tile
            # (T % TILE_SIZE != 0) a fixed +TILE_SIZE read of the last dest_rank's
            # region runs past got_hbm's end (base+ts+TILE_SIZE > EP*T -> OOB read
            # / dma_copy size assert). Zero the full buffer first so the tail rows
            # [tsize:TILE_SIZE] the downstream tensor_tensor still reads are defined
            # (they are OOB-skipped on the final scatter, so they never reach output).
            nisa.memset(data_sb, 0)
            nisa.dma_copy(dst=data_sb[0:tsize, :], src=got_hbm[base + ts : base + ts + tsize, :])

            cur_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cur_sb,
                src=partial.ap(
                    pattern=[[T, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=idx_sb,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.tensor_tensor(cur_sb, cur_sb, data_sb, op=nl.add)
            nisa.dma_copy(
                dst=partial.ap(
                    pattern=[[T, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=idx_sb,
                    indirect_dim=0,
                ),
                src=cur_sb,
                oob_mode=oob_mode.skip,
            )

    # All-reduce across cores via sendrecv; core 0 writes the final result.
    result = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm, name="comb_result")
    for tile_idx in nl.affine_range(NUM_TILES):
        ts = tile_idx * TILE_SIZE
        te = min(ts + TILE_SIZE, T)
        tsize = te - ts
        my_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=my_sb[0:tsize, :], src=partial[ts:te, :])
        if n_prgs > 1:
            other_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.sendrecv(
                src=my_sb,
                dst=other_sb,
                send_to_rank=(1 - prg_id),
                recv_from_rank=(1 - prg_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=my_sb, data1=my_sb, data2=other_sb, op=nl.add)
        if prg_id == 0:
            nisa.dma_copy(dst=result[ts:te, :], src=my_sb[0:tsize, :])

    return result
