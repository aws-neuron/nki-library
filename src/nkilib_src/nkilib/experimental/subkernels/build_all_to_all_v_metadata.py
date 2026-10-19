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

"""Build metadata buffer containing send_counts, send_displs, recv_counts, and recv_displs for all_to_all_v collective."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert


@nki.jit
def build_all_to_all_v_metadata(
    expert_index: nl.NkiTensor,
    replica_group_size: int,
    E: int,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
):
    """
    Build metadata buffer for all_to_all_v collective from MoE routing decisions.

    Computes per-rank send counts and displacements from expert assignments.

    Dimensions:
        T: Number of tokens.
        K: Top-K experts per token.
        E: Total number of global experts.

    Args:
        expert_index (nl.NkiTensor): [T, K] int32 HBM tensor indicating the K experts each token is routed to.
        replica_group_size (int): Size of replica group for all_to_all_v collective.
        E (int): Number of global experts.
        recv_counts_known (bool): Not currently supported; when True, metadata includes recv counts.
        has_rdispls (bool): Not currently supported; when True, metadata includes recv displacements.

    Returns:
        metadata_hbm (nl.NkiTensor): [n_rows, replica_group_size] uint32 HBM tensor.
            n_rows is 4 when has_rdispls=True, 3 otherwise.
            Row 0: send counts, Row 1: send displacements, Row 2: recv counts (zeros), Row 3 (optional): recv displacements (zeros).
    """

    # Step 1: Extract/validate shapes, allocate buffers
    # Shapes, validation
    T, K, tile_E, tile_replica_group, n_tiles, experts_per_dst_rank = _validate_and_extract_shapes(
        expert_index, replica_group_size, E, recv_counts_known, has_rdispls
    )
    expert_index_flattened_bc_sb = nl.ndarray((tile_E, T * K), dtype=nl.float32)
    send_counts_T_sb = nl.ndarray((tile_E, 1), dtype=nl.float32, buffer=nl.sbuf)
    send_counts_sb = nl.ndarray((1, replica_group_size), dtype=nl.uint32, buffer=nl.sbuf)
    send_displs_sb = nl.ndarray((1, replica_group_size), dtype=nl.uint32, buffer=nl.sbuf)
    init_sb = nl.ndarray((1, 1), dtype=nl.float32)
    ones_sb = nl.ndarray((1, replica_group_size - 1), dtype=nl.float32)
    # NOTE: we purposely leave row 2 full of zeros; recv_counts must be allocated but is not yet supported
    n_metadata_rows = 4 if has_rdispls else 3
    metadata_hbm = nl.ndarray((n_metadata_rows, replica_group_size), dtype=nl.uint32, buffer=nl.shared_hbm)

    # Step 2: Compute send_counts, tiling on E dim
    # Step 2.1: Load flattened expert index, broadcasting across partitions
    expert_index_flattened_hbm = expert_index.reshape((1, T * K))
    nisa.dma_copy(
        src=expert_index_flattened_hbm.ap(
            pattern=[[0, tile_E], [1, T * K]],
            offset=0,
        ),
        dst=expert_index_flattened_bc_sb,
    )

    # Step 2.2: Compute send counts for groups of E
    for expert_group_idx in nl.affine_range(n_tiles):
        # Step 2.2.1: Calculate number of routed tokens for each expert in the current group
        expert_offset = expert_group_idx * tile_E
        replica_group_offset = expert_offset // experts_per_dst_rank
        dummy_dst = nl.ndarray((tile_E, T * K), dtype=nl.float32)
        arange_tile_P = nl.ndarray((tile_E, 1), dtype=nl.float32)
        send_counts_tile_psum = nl.ndarray((1, tile_E), dtype=nl.float32, buffer=nl.psum)
        nisa.iota(
            pattern=[[0, 1]],
            offset=expert_offset,
            channel_multiplier=1,
            dst=arange_tile_P,
        )
        nisa.tensor_scalar_reduce(
            data=expert_index_flattened_bc_sb,
            op0=nl.equal,
            operand0=arange_tile_P,
            dst=dummy_dst,
            reduce_op=nl.add,
            reduce_res=send_counts_T_sb,
        )

        # Step 2.2.2: Transpose expert counts, reduce counts from [1, E] to [1, replica_group_size]
        nisa.nc_transpose(send_counts_tile_psum, send_counts_T_sb)
        rg_slice = nl.ds(replica_group_offset, tile_replica_group)
        if experts_per_dst_rank == 1:
            nisa.tensor_copy(send_counts_sb[0, rg_slice], send_counts_tile_psum)
        else:
            send_counts_tile_psum = send_counts_tile_psum.reshape((1, tile_replica_group, experts_per_dst_rank))
            nisa.tensor_reduce(
                data=send_counts_tile_psum,
                op=nl.add,
                axis=2,
                dst=send_counts_sb[0, rg_slice],
            )

    # Step 3: Compute send_displs using cumsum with offset 1 on send_counts
    nisa.memset(send_displs_sb[0, 0], 0.0)  # first val is always 0
    nisa.memset(init_sb, 0.0)
    nisa.memset(ones_sb, 1.0)
    nisa.tensor_tensor_scan(
        initial=init_sb,
        data0=ones_sb,
        op1=nl.add,
        data1=send_counts_sb[0, : replica_group_size - 1],
        op0=nl.multiply,
        dst=send_displs_sb[0, 1:],
    )

    # Step 4: Copy outputs into corresponding rows in output buffer
    nisa.dma_copy(metadata_hbm[0, :], send_counts_sb)
    nisa.dma_copy(metadata_hbm[1, :], send_displs_sb)

    return metadata_hbm


def _validate_and_extract_shapes(expert_index, replica_group_size, E, recv_counts_known, has_rdispls):
    """Extract and validate shapes from build_all_to_all_v_metadata inputs."""
    T, K = expert_index.shape
    pmax = nl.tile_size.pmax
    experts_per_dst_rank = E // replica_group_size
    kernel_assert(
        E % replica_group_size == 0, f"Expected E divisible by replica_group_size, got {E=}, {replica_group_size=}"
    )
    kernel_assert(E % pmax == 0 or E <= pmax, f"Expected E divisible by pmax when E > pmax, got {E=}, {pmax=}")
    kernel_assert(
        not recv_counts_known, f"Kernel does not yet support recv_counts_known=True, got {recv_counts_known=}"
    )
    kernel_assert(not has_rdispls, f"Kernel does not yet support has_rdispls=True, got {has_rdispls=}")
    tile_E = min(E, pmax)
    tile_replica_group = tile_E // experts_per_dst_rank
    n_tiles = E // tile_E
    return T, K, tile_E, tile_replica_group, n_tiles, experts_per_dst_rank
