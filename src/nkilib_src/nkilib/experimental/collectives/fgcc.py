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
"""Fine grained Gather Collective Compute (FGCC) kernel for TRN2."""

from typing import List, Optional, Tuple

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.kernel_assert import kernel_assert


@nki.jit
def allgather_compute_matmul(
    lhs: nl.ndarray,
    rhs: nl.ndarray,
    tp_degree: int,
    num_groups: int,
    force_hbm_cc: bool = False,
) -> nl.ndarray:
    """
    Fine grained all-gather and matrix multiplication (FGCC) kernel for TRN2.

    Performs all-gather on lhs across ranks, then computes matmul with
    column-sharded rhs. Uses ring-based collective permute for communication
    overlapped with compute. Supports both SBUF and HBM communication paths
    with automatic selection based on tensor sizes.

    TODO: Specify intended usage range (e.g., sequence length, batch size)

    Dimensions:
        m: Local rows per rank (before all-gather).
        M: Total rows after all-gather (m * tp_degree).
        K: Shared (contraction) dimension.
        N: Output column dimension (column-sharded per rank).

    Args:
        lhs (nl.ndarray): [m, K], Left-hand side tensor, row-sharded across ranks.
        rhs (nl.ndarray): [K, N], Right-hand side tensor, column-sharded per rank.
        tp_degree (int): Tensor parallelism degree (number of ranks). Must be even.
        num_groups (int): Number of replica groups for collective communication.
        force_hbm_cc (bool): If True, force HBM collective communication path
            even when SBUF path is feasible.

    Returns:
        result (nl.ndarray): [RANK_N, ...], Column-sharded result tensor in shared HBM.
            Shape depends on communication path (SBUF vs HBM).

    Notes:
        - tp_degree must be even.
        - lhs and rhs must have matching K dimension.
        - M must be divisible by (RANK_N * LNC_N * CHANNEL_N).
        - Platform target is TRN2 only.

    Pseudocode:
        TODO: Add pseudocode description
    """
    m, K = lhs.shape
    K_rhs, N = rhs.shape
    RANK_N = tp_degree
    M = m * RANK_N

    kernel_assert(K == K_rhs, f"lhs K dimension ({K}) must match rhs K dimension ({K_rhs})")
    kernel_assert(RANK_N % 2 == 0, f"tp_degree must be even, got {RANK_N}")

    dtype = lhs.dtype

    if dtype == nl.float32:
        DTYPE_SIZE = 4
    elif dtype == nl.bfloat16 or dtype == nl.float16:
        DTYPE_SIZE = 2
    else:
        DTYPE_SIZE = 4

    LNC_N = nl.num_programs(axes=0)

    if LNC_N > 1:
        lnc_id = nl.program_id(0)
    else:
        lnc_id = 0

    if LNC_N == 2:
        if tp_degree == 4:
            CHANNEL_N = 2
        elif tp_degree == 8:
            CHANNEL_N = 2
        elif tp_degree == 16:
            CHANNEL_N = 2
        elif tp_degree == 32:
            if nisa.get_nc_version() >= 3:
                CHANNEL_N = 2
            else:
                CHANNEL_N = 4
        elif tp_degree == 64:
            CHANNEL_N = 4
        elif tp_degree == 128:
            CHANNEL_N = 4
        else:
            CHANNEL_N = 1
    else:
        CHANNEL_N = 1

    kernel_assert(
        M % (RANK_N * LNC_N * CHANNEL_N) == 0,
        f"M ({M}) must be divisible by RANK_N * LNC_N * CHANNEL_N ({RANK_N * LNC_N * CHANNEL_N})",
    )

    replica_group = ReplicaGroup(_generate_replica_groups(tp_degree=tp_degree, num_groups=num_groups))

    TILE_M = min(M // RANK_N // LNC_N // CHANNEL_N, nl.tile_size.gemm_stationary_fmax)
    TILE_N = min(N, nl.tile_size.gemm_moving_fmax)
    TILE_K = min(K, nl.tile_size.pmax)

    TILES_IN_BLOCK_M = (M // RANK_N // LNC_N // CHANNEL_N + TILE_M - 1) // TILE_M
    TILES_IN_BLOCK_N = (N + TILE_N - 1) // TILE_N
    TILES_IN_BLOCK_K = (K + TILE_K - 1) // TILE_K

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    NUM_BLOCK_M = (M // RANK_N // LNC_N // CHANNEL_N + BLOCK_M - 1) // BLOCK_M
    NUM_BLOCK_N = (N + BLOCK_N - 1) // BLOCK_N
    NUM_BLOCK_K = (K + BLOCK_K - 1) // BLOCK_K

    SIZE_LHS = BLOCK_M * NUM_BLOCK_M * BLOCK_K * NUM_BLOCK_K * DTYPE_SIZE
    SIZE_RHS = BLOCK_K * NUM_BLOCK_K * BLOCK_N * NUM_BLOCK_N * DTYPE_SIZE

    # Maximum SBUF budget for RHS double-buffering (16 MiB)
    _MAX_RHS_SBUF_BYTES = 16 << 20
    # Total SBUF budget for double-buffered operands (24 MiB)
    _TOTAL_SBUF_BUDGET_BYTES = 24 << 20

    rhs_in_sbuf = False
    if 2 * SIZE_RHS <= _MAX_RHS_SBUF_BYTES:
        rhs_in_sbuf = True
    else:
        TILES_IN_BLOCK_K = min(TILES_IN_BLOCK_K, 16 // DTYPE_SIZE)
        TILES_IN_BLOCK_N = min(TILES_IN_BLOCK_N, _MAX_RHS_SBUF_BYTES // BLOCK_K // TILE_N // DTYPE_SIZE)
        BLOCK_K = TILE_K * TILES_IN_BLOCK_K
        BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    _TOTAL_SBUF_BUDGET_BYTES -= 2 * (BLOCK_K * BLOCK_N * DTYPE_SIZE)

    lhs_in_sbuf = False
    if not force_hbm_cc and 2 * SIZE_LHS <= _TOTAL_SBUF_BUDGET_BYTES:
        lhs_in_sbuf = True
    else:
        max_tiles = _TOTAL_SBUF_BUDGET_BYTES // BLOCK_K // TILE_M // DTYPE_SIZE
        TILES_IN_BLOCK_M = max(1, min(TILES_IN_BLOCK_M, max_tiles))

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    NUM_BLOCK_M = (M // RANK_N // LNC_N // CHANNEL_N + BLOCK_M - 1) // BLOCK_M

    if lhs_in_sbuf:
        result = nl.ndarray(
            (RANK_N, LNC_N, CHANNEL_N, M // RANK_N // LNC_N // CHANNEL_N, N),
            dtype=dtype,
            buffer=nl.shared_hbm,
            name="result_tensor_sbuf",
        )
        lhs = lhs.reshape((LNC_N, CHANNEL_N, M // RANK_N // LNC_N // CHANNEL_N, K))
    else:
        result = nl.ndarray(
            (RANK_N, CHANNEL_N, LNC_N, M // RANK_N // LNC_N // CHANNEL_N, N),
            dtype=dtype,
            buffer=nl.shared_hbm,
            name="result_tensor_hbm",
        )
        lhs = lhs.reshape((CHANNEL_N, LNC_N, M // RANK_N // LNC_N // CHANNEL_N, K))

    # RHS loading into SBUF
    if rhs_in_sbuf:
        rhs_sbuf = []
        for k_blk_idx in range(NUM_BLOCK_K):
            k_list = []
            for bk_tile_idx in range(TILES_IN_BLOCK_K):
                bk_list = []
                for n_blk_idx in range(NUM_BLOCK_N):
                    tile = nl.ndarray((TILE_K, BLOCK_N), dtype=dtype, buffer=nl.sbuf)
                    rhs_row = k_blk_idx * BLOCK_K + bk_tile_idx * TILE_K
                    rhs_col = n_blk_idx * BLOCK_N
                    nisa.dma_copy(
                        dst=tile,
                        src=rhs.ap(pattern=[[N, TILE_K], [1, BLOCK_N]], offset=rhs_row * N + rhs_col),
                    )
                    bk_list.append(tile)
                k_list.append(bk_list)
            rhs_sbuf.append(k_list)
        rhs_ = rhs_sbuf
    else:
        rhs_ = rhs

    local_M = M // RANK_N // LNC_N // CHANNEL_N

    if lhs_in_sbuf:
        # SBUF path: lhs tiles are transposed and stored in SBUF for ring communication
        buf0_ = []
        buf1_ = []
        for ch_idx in range(CHANNEL_N):
            ch_list0, ch_list1 = [], []
            for k_idx in range(NUM_BLOCK_K):
                k_list0, k_list1 = [], []
                for bk_idx in range(TILES_IN_BLOCK_K):
                    bk_list0, bk_list1 = [], []
                    for m_idx in range(NUM_BLOCK_M):
                        bk_list0.append(nl.ndarray((TILE_K, BLOCK_M), dtype=dtype, buffer=nl.sbuf))
                        bk_list1.append(nl.ndarray((TILE_K, BLOCK_M), dtype=dtype, buffer=nl.sbuf))
                    k_list0.append(bk_list0)
                    k_list1.append(bk_list1)
                ch_list0.append(k_list0)
                ch_list1.append(k_list1)
            buf0_.append(ch_list0)
            buf1_.append(ch_list1)

        # Transpose lhs tiles into buf0_
        for channel_idx in nl.affine_range(CHANNEL_N):
            for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                    for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                        row_start = m_blk_idx * BLOCK_M
                        col_start = (k_blk_idx * TILES_IN_BLOCK_K + bk_tile_idx) * TILE_K
                        src_offset = (
                            (lnc_id * CHANNEL_N * local_M * K) + (channel_idx * local_M * K) + row_start * K + col_start
                        )
                        nisa.dma_transpose(
                            dst=buf0_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx].ap(
                                pattern=[[BLOCK_M, TILE_K], [1, 1], [1, 1], [1, BLOCK_M]]
                            ),
                            src=lhs.ap(pattern=[[K, BLOCK_M], [1, 1], [1, 1], [1, TILE_K]], offset=src_offset),
                        )

        # Flatten buf0_ tiles into contiguous buf0 for collective permute
        flat_size = CHANNEL_N * NUM_BLOCK_K * TILES_IN_BLOCK_K * NUM_BLOCK_M * BLOCK_M
        buf0 = nl.ndarray((TILE_K, flat_size), dtype=dtype, buffer=nl.sbuf)
        buf1 = nl.ndarray((TILE_K, flat_size), dtype=dtype, buffer=nl.sbuf)

        for channel_idx in nl.affine_range(CHANNEL_N):
            for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                    for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                        flat_offset = (
                            (channel_idx * NUM_BLOCK_K + k_blk_idx) * TILES_IN_BLOCK_K + bk_tile_idx
                        ) * NUM_BLOCK_M * BLOCK_M + m_blk_idx * BLOCK_M
                        nisa.tensor_copy(
                            dst=buf0[:, flat_offset : flat_offset + BLOCK_M],
                            src=buf0_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx],
                        )

        result_slice_size = local_M * N

        # Step 0: compute matmul with local data
        for channel_idx in nl.affine_range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=0, channel_id=channel_idx, replica_group=replica_group
            )
            result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
            _compute_matmul_sbuf(
                buf0_,
                rhs_,
                result,
                numerator,
                rhs_in_sbuf,
                TILE_M,
                TILE_N,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_N,
                NUM_BLOCK_M,
                NUM_BLOCK_N,
                local_M,
                K,
                N,
                channel_idx,
                result_offset,
                CHANNEL_N,
                LNC_N,
                RANK_N,
                NUM_BLOCK_K,
            )

        # First collective permute: buf0 -> buf1
        _launch_collective_permutes_sbuf(
            buf0, buf1, replica_group, CHANNEL_N, NUM_BLOCK_K, TILES_IN_BLOCK_K, NUM_BLOCK_M, BLOCK_M
        )

        # Unpack buf1 into buf1_ tiles
        for channel_idx in nl.affine_range(CHANNEL_N):
            for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                    for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                        flat_offset = (
                            (channel_idx * NUM_BLOCK_K + k_blk_idx) * TILES_IN_BLOCK_K + bk_tile_idx
                        ) * NUM_BLOCK_M * BLOCK_M + m_blk_idx * BLOCK_M
                        nisa.tensor_copy(
                            dst=buf1_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx],
                            src=buf1[:, flat_offset : flat_offset + BLOCK_M],
                        )

        # Step 1: compute matmul with first permuted data
        for channel_idx in nl.affine_range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=1, channel_id=channel_idx, replica_group=replica_group
            )
            result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
            _compute_matmul_sbuf(
                buf1_,
                rhs_,
                result,
                numerator,
                rhs_in_sbuf,
                TILE_M,
                TILE_N,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_N,
                NUM_BLOCK_M,
                NUM_BLOCK_N,
                local_M,
                K,
                N,
                channel_idx,
                result_offset,
                CHANNEL_N,
                LNC_N,
                RANK_N,
                NUM_BLOCK_K,
            )

        # Ring steps 2..RANK_N-1: alternate between buf0/buf1
        for step_idx in nl.sequential_range(1, RANK_N // 2):
            # Pack buf1_ -> buf1 flat
            for channel_idx in nl.affine_range(CHANNEL_N):
                for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                    for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                        for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                            flat_offset = (
                                (channel_idx * NUM_BLOCK_K + k_blk_idx) * TILES_IN_BLOCK_K + bk_tile_idx
                            ) * NUM_BLOCK_M * BLOCK_M + m_blk_idx * BLOCK_M
                            nisa.tensor_copy(
                                dst=buf1[:, flat_offset : flat_offset + BLOCK_M],
                                src=buf1_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx],
                            )

            _launch_collective_permutes_sbuf(
                buf1, buf0, replica_group, CHANNEL_N, NUM_BLOCK_K, TILES_IN_BLOCK_K, NUM_BLOCK_M, BLOCK_M
            )

            # Unpack buf0 -> buf0_ tiles
            for channel_idx in nl.affine_range(CHANNEL_N):
                for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                    for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                        for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                            flat_offset = (
                                (channel_idx * NUM_BLOCK_K + k_blk_idx) * TILES_IN_BLOCK_K + bk_tile_idx
                            ) * NUM_BLOCK_M * BLOCK_M + m_blk_idx * BLOCK_M
                            nisa.tensor_copy(
                                dst=buf0_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx],
                                src=buf0[:, flat_offset : flat_offset + BLOCK_M],
                            )

            # Even step: compute with buf0_
            for channel_idx in nl.affine_range(CHANNEL_N):
                numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                    iteration_id=2 * step_idx,
                    channel_id=channel_idx,
                    replica_group=replica_group,
                )
                result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
                _compute_matmul_sbuf(
                    buf0_,
                    rhs_,
                    result,
                    numerator,
                    rhs_in_sbuf,
                    TILE_M,
                    TILE_N,
                    TILE_K,
                    TILES_IN_BLOCK_M,
                    TILES_IN_BLOCK_N,
                    TILES_IN_BLOCK_K,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BLOCK_M,
                    NUM_BLOCK_N,
                    local_M,
                    K,
                    N,
                    channel_idx,
                    result_offset,
                    CHANNEL_N,
                    LNC_N,
                    RANK_N,
                    NUM_BLOCK_K,
                )

            # Pack buf0_ -> buf0 flat
            for channel_idx in nl.affine_range(CHANNEL_N):
                for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                    for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                        for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                            flat_offset = (
                                (channel_idx * NUM_BLOCK_K + k_blk_idx) * TILES_IN_BLOCK_K + bk_tile_idx
                            ) * NUM_BLOCK_M * BLOCK_M + m_blk_idx * BLOCK_M
                            nisa.tensor_copy(
                                dst=buf0[:, flat_offset : flat_offset + BLOCK_M],
                                src=buf0_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx],
                            )

            _launch_collective_permutes_sbuf(
                buf0, buf1, replica_group, CHANNEL_N, NUM_BLOCK_K, TILES_IN_BLOCK_K, NUM_BLOCK_M, BLOCK_M
            )

            # Unpack buf1 -> buf1_ tiles
            for channel_idx in nl.affine_range(CHANNEL_N):
                for k_blk_idx in nl.affine_range(NUM_BLOCK_K):
                    for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                        for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                            flat_offset = (
                                (channel_idx * NUM_BLOCK_K + k_blk_idx) * TILES_IN_BLOCK_K + bk_tile_idx
                            ) * NUM_BLOCK_M * BLOCK_M + m_blk_idx * BLOCK_M
                            nisa.tensor_copy(
                                dst=buf1_[channel_idx][k_blk_idx][bk_tile_idx][m_blk_idx],
                                src=buf1[:, flat_offset : flat_offset + BLOCK_M],
                            )

            # Odd step: compute with buf1_
            for channel_idx in nl.affine_range(CHANNEL_N):
                numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                    iteration_id=2 * step_idx + 1,
                    channel_id=channel_idx,
                    replica_group=replica_group,
                )
                result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
                _compute_matmul_sbuf(
                    buf1_,
                    rhs_,
                    result,
                    numerator,
                    rhs_in_sbuf,
                    TILE_M,
                    TILE_N,
                    TILE_K,
                    TILES_IN_BLOCK_M,
                    TILES_IN_BLOCK_N,
                    TILES_IN_BLOCK_K,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BLOCK_M,
                    NUM_BLOCK_N,
                    local_M,
                    K,
                    N,
                    channel_idx,
                    result_offset,
                    CHANNEL_N,
                    LNC_N,
                    RANK_N,
                    NUM_BLOCK_K,
                )

    else:
        # HBM path: lhs tiles stay in HBM, collective permute operates on HBM buffers
        buf0 = nl.ndarray((M // RANK_N, K), dtype=dtype, buffer=nl.shared_hbm, name="buf0_tensor_hbm")
        buf1 = nl.ndarray((M // RANK_N, K), dtype=dtype, buffer=nl.shared_hbm, name="buf1_tensor_hbm")

        buf0_ = buf0.reshape((CHANNEL_N, LNC_N, M // RANK_N // LNC_N // CHANNEL_N, K))
        buf1_ = buf1.reshape((CHANNEL_N, LNC_N, M // RANK_N // LNC_N // CHANNEL_N, K))

        lhs_slice_size = local_M * K
        buf_slice_size = local_M * K
        result_slice_size = local_M * N

        # Copy local lhs into buf0
        for channel_idx in nl.affine_range(CHANNEL_N):
            lhs_offset = (channel_idx * LNC_N + lnc_id) * lhs_slice_size
            buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
            nisa.dma_copy(
                dst=buf0.ap(pattern=[[K, local_M], [1, K]], offset=buf_offset),
                src=lhs.ap(pattern=[[K, local_M], [1, K]], offset=lhs_offset),
            )

        # Step 0: compute matmul with local data
        for channel_idx in nl.affine_range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=0, channel_id=channel_idx, replica_group=replica_group
            )
            buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
            result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
            _compute_matmul_hbm(
                buf0,
                rhs_,
                result,
                numerator,
                rhs_in_sbuf,
                TILE_M,
                TILE_N,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_N,
                NUM_BLOCK_M,
                NUM_BLOCK_N,
                local_M,
                K,
                N,
                buf_offset,
                result_offset,
                CHANNEL_N,
                LNC_N,
                RANK_N,
            )

        # First collective permute: buf0 -> buf1
        _launch_collective_permutes_hbm(buf0, buf1, replica_group, CHANNEL_N, M, RANK_N, LNC_N, K)

        # Step 1: compute matmul with first permuted data
        for channel_idx in nl.affine_range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=1, channel_id=channel_idx, replica_group=replica_group
            )
            buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
            result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
            _compute_matmul_hbm(
                buf1,
                rhs_,
                result,
                numerator,
                rhs_in_sbuf,
                TILE_M,
                TILE_N,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_N,
                NUM_BLOCK_M,
                NUM_BLOCK_N,
                local_M,
                K,
                N,
                buf_offset,
                result_offset,
                CHANNEL_N,
                LNC_N,
                RANK_N,
            )

        # Ring steps 2..RANK_N-1: alternate between buf0/buf1
        for step_idx in nl.sequential_range(1, RANK_N // 2):
            _launch_collective_permutes_hbm(buf1, buf0, replica_group, CHANNEL_N, M, RANK_N, LNC_N, K)

            # Even step: compute with buf0
            for channel_idx in nl.affine_range(CHANNEL_N):
                numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                    iteration_id=2 * step_idx,
                    channel_id=channel_idx,
                    replica_group=replica_group,
                )
                buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
                result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
                _compute_matmul_hbm(
                    buf0,
                    rhs_,
                    result,
                    numerator,
                    rhs_in_sbuf,
                    TILE_M,
                    TILE_N,
                    TILE_K,
                    TILES_IN_BLOCK_M,
                    TILES_IN_BLOCK_N,
                    TILES_IN_BLOCK_K,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BLOCK_M,
                    NUM_BLOCK_N,
                    local_M,
                    K,
                    N,
                    buf_offset,
                    result_offset,
                    CHANNEL_N,
                    LNC_N,
                    RANK_N,
                )

            _launch_collective_permutes_hbm(buf0, buf1, replica_group, CHANNEL_N, M, RANK_N, LNC_N, K)

            # Odd step: compute with buf1
            for channel_idx in nl.affine_range(CHANNEL_N):
                numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                    iteration_id=2 * step_idx + 1,
                    channel_id=channel_idx,
                    replica_group=replica_group,
                )
                buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
                result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
                _compute_matmul_hbm(
                    buf1,
                    rhs_,
                    result,
                    numerator,
                    rhs_in_sbuf,
                    TILE_M,
                    TILE_N,
                    TILE_K,
                    TILES_IN_BLOCK_M,
                    TILES_IN_BLOCK_N,
                    TILES_IN_BLOCK_K,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BLOCK_M,
                    NUM_BLOCK_N,
                    local_M,
                    K,
                    N,
                    buf_offset,
                    result_offset,
                    CHANNEL_N,
                    LNC_N,
                    RANK_N,
                )

    return result


def _generate_replica_groups(tp_degree: int, num_groups: int) -> List[List[int]]:
    """
    Generate replica groups for collective operations.

    Args:
        tp_degree (int): Tensor parallelism degree.
        num_groups (int): Number of replica groups.

    Returns:
        List[List[int]]: List of replica groups, each containing replica IDs.
    """
    num_replicas = tp_degree * num_groups
    replicas_per_group = num_replicas // num_groups
    replica_groups = []
    for group_idx in range(num_groups):
        group = []
        for replica_idx in range(replicas_per_group):
            group.append(group_idx * replicas_per_group + replica_idx)
        replica_groups.append(tuple(group))
    return tuple(replica_groups)


def _matmul(
    lhs: nl.ndarray,
    lhs_in_sbuf: bool,
    rhs: nl.ndarray,
    rhs_in_sbuf: bool,
    res: nl.ndarray,
    res_in_sbuf: bool,
    TILES_IN_BLOCKS: Tuple[int, int, int] = (16, 2, 8),
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    lhs_offset: int = 0,
) -> None:
    """
    Tiled matrix multiplication helper for allgather_compute_matmul.

    Performs blocked matmul: res += lhs @ rhs, with configurable tile sizes and
    support for operands in either SBUF or HBM.

    Args:
        lhs (nl.ndarray): Left-hand side matrix, shape [M, K].
        lhs_in_sbuf (bool): Whether lhs is pre-loaded in SBUF.
        rhs (nl.ndarray): Right-hand side matrix, shape [K, N].
        rhs_in_sbuf (bool): Whether rhs is pre-loaded in SBUF.
        res (nl.ndarray): Result accumulation buffer, shape [M, N].
        res_in_sbuf (bool): Whether res is in SBUF.
        TILES_IN_BLOCKS (Tuple[int, int, int]): Tile counts per block for (M, N, K).
        M (Optional[int]): Total rows of lhs.
        K (Optional[int]): Shared dimension.
        N (Optional[int]): Total columns of rhs.
        lhs_offset (int): Byte offset into lhs for DMA reads.

    Returns:
        None: Results are accumulated in-place into res.

    Notes:
        - When operands are not in SBUF, DMA transfers are issued per tile block.
        - Result tiles are packed and written back to HBM when res_in_sbuf is False.
    """
    TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K = TILES_IN_BLOCKS

    TILE_M = min(M, nl.tile_size.gemm_stationary_fmax)
    TILE_N = min(N, nl.tile_size.gemm_moving_fmax)
    TILE_K = min(K, nl.tile_size.pmax)

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    for n_blk_idx in nl.affine_range(NUM_BLOCK_N):
        if res_in_sbuf:
            result_tiles = res
        else:
            result_tiles = []
            for m_idx in range(NUM_BLOCK_M):
                m_row = []
                for bm_idx in range(TILES_IN_BLOCK_M):
                    bm_row = []
                    for bn_idx in range(TILES_IN_BLOCK_N):
                        tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.memset(tile, 0.0)
                        bm_row.append(tile)
                    m_row.append(bm_row)
                result_tiles.append(m_row)

        for k_blk_idx in nl.sequential_range(NUM_BLOCK_K):
            if rhs_in_sbuf:
                rhs_tiles = rhs
            else:
                rhs_tiles = []
                for bk_rhs_idx in nl.affine_range(TILES_IN_BLOCK_K):
                    rhs_tile = nl.ndarray((TILE_K, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)
                    rhs_row_start = (TILES_IN_BLOCK_K * k_blk_idx + bk_rhs_idx) * TILE_K
                    rhs_col_start = BLOCK_N * n_blk_idx
                    nisa.dma_copy(
                        dst=rhs_tile,
                        src=rhs.ap(
                            pattern=[[N, TILE_K], [1, BLOCK_N]],
                            offset=rhs_row_start * N + rhs_col_start,
                        ),
                    )
                    rhs_tiles.append(rhs_tile)

            for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                if lhs_in_sbuf:
                    lhsT_tiles = lhs
                else:
                    lhsT_tiles = []
                    for bk_lhs_idx in nl.affine_range(TILES_IN_BLOCK_K):
                        lhsT_tile = nl.ndarray((TILE_K, BLOCK_M), dtype=lhs.dtype, buffer=nl.sbuf)
                        row_start = BLOCK_M * m_blk_idx
                        col_start = (TILES_IN_BLOCK_K * k_blk_idx + bk_lhs_idx) * TILE_K
                        src_offset = lhs_offset + row_start * K + col_start
                        nisa.dma_transpose(
                            dst=lhsT_tile.ap(pattern=[[BLOCK_M, TILE_K], [1, 1], [1, 1], [1, BLOCK_M]]),
                            src=lhs.ap(pattern=[[K, BLOCK_M], [1, 1], [1, 1], [1, TILE_K]], offset=src_offset),
                        )
                        lhsT_tiles.append(lhsT_tile)

                for bn_tile_idx in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm_tile_idx in nl.affine_range(TILES_IN_BLOCK_M):
                        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                        for bk_tile_idx in nl.affine_range(TILES_IN_BLOCK_K):
                            if lhs_in_sbuf:
                                stat = lhs[k_blk_idx][bk_tile_idx][m_blk_idx][
                                    0:TILE_K, bm_tile_idx * TILE_M : (bm_tile_idx + 1) * TILE_M
                                ]
                            else:
                                stat = lhsT_tiles[bk_tile_idx][
                                    0:TILE_K, bm_tile_idx * TILE_M : (bm_tile_idx + 1) * TILE_M
                                ]

                            if rhs_in_sbuf:
                                mov = rhs[k_blk_idx][bk_tile_idx][n_blk_idx][
                                    0:TILE_K, bn_tile_idx * TILE_N : (bn_tile_idx + 1) * TILE_N
                                ]
                            else:
                                mov = rhs_tiles[bk_tile_idx][
                                    0:TILE_K, bn_tile_idx * TILE_N : (bn_tile_idx + 1) * TILE_N
                                ]

                            nisa.nc_matmul(dst=res_psum, stationary=stat, moving=mov)

                        if res_in_sbuf:
                            res_sb = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.tensor_copy(dst=res_sb, src=res_psum)
                            nisa.tensor_tensor(
                                dst=res[m_blk_idx][bm_tile_idx][n_blk_idx][bn_tile_idx],
                                data1=res[m_blk_idx][bm_tile_idx][n_blk_idx][bn_tile_idx],
                                data2=res_sb,
                                op=nl.add,
                            )
                        else:
                            res_sb = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.tensor_copy(dst=res_sb, src=res_psum)
                            nisa.tensor_tensor(
                                dst=result_tiles[m_blk_idx][bm_tile_idx][bn_tile_idx],
                                data1=result_tiles[m_blk_idx][bm_tile_idx][bn_tile_idx],
                                data2=res_sb,
                                op=nl.add,
                            )

        if not res_in_sbuf:
            for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
                for bm_tile_idx in nl.affine_range(TILES_IN_BLOCK_M):
                    result_packed = nl.ndarray((TILE_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
                    for bn_tile_idx in nl.affine_range(TILES_IN_BLOCK_N):
                        nisa.tensor_copy(
                            dst=result_packed[0:TILE_M, bn_tile_idx * TILE_N : (bn_tile_idx + 1) * TILE_N],
                            src=result_tiles[m_blk_idx][bm_tile_idx][bn_tile_idx],
                        )
                    nisa.dma_copy(
                        dst=res[
                            (TILES_IN_BLOCK_M * m_blk_idx + bm_tile_idx) * TILE_M : (
                                TILES_IN_BLOCK_M * m_blk_idx + bm_tile_idx + 1
                            )
                            * TILE_M,
                            BLOCK_N * n_blk_idx : BLOCK_N * (n_blk_idx + 1),
                        ],
                        src=result_packed,
                    )


def _compute_matmul_hbm(
    lhs_buf: nl.ndarray,
    rhs: nl.ndarray,
    result_tensor: nl.ndarray,
    numerator: int,
    rhs_in_sbuf: bool,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_N: int,
    TILES_IN_BLOCK_K: int,
    BLOCK_M: int,
    BLOCK_N: int,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    local_M: int,
    K: int,
    N: int,
    lhs_offset: int,
    result_offset: int,
    CHANNEL_N: int,
    LNC_N: int,
    RANK_N: int,
) -> None:
    """
    Compute tiled matmul with lhs in HBM and store results to HBM.

    Allocates SBUF result tiles, performs blocked matmul via _matmul, then
    writes result tiles back to the result tensor in HBM using indirect DMA.

    Args:
        lhs_buf (nl.ndarray): Left-hand side buffer in HBM.
        rhs (nl.ndarray): Right-hand side tensor (HBM or SBUF).
        result_tensor (nl.ndarray): Output tensor in shared HBM.
        numerator (int): Rank ID for indirect DMA offset.
        rhs_in_sbuf (bool): Whether rhs is pre-loaded in SBUF.
        TILE_M (int): Tile size along M dimension.
        TILE_N (int): Tile size along N dimension.
        TILE_K (int): Tile size along K dimension.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        BLOCK_M (int): Block size along M dimension.
        BLOCK_N (int): Block size along N dimension.
        NUM_BLOCK_M (int): Number of blocks along M dimension.
        NUM_BLOCK_N (int): Number of blocks along N dimension.
        local_M (int): Local M dimension per rank/LNC/channel.
        K (int): Shared dimension size.
        N (int): Output column dimension size.
        lhs_offset (int): Offset into lhs_buf for DMA reads.
        result_offset (int): Offset into result_tensor for DMA writes.
        CHANNEL_N (int): Number of communication channels.
        LNC_N (int): Number of LNC programs.
        RANK_N (int): Number of ranks (tensor parallelism degree).

    Returns:
        None: Results are written to result_tensor via DMA.
    """
    res_gate = []
    for m_idx in range(NUM_BLOCK_M):
        m_row = []
        for bm_idx in range(TILES_IN_BLOCK_M):
            bm_row = []
            for n_idx in range(NUM_BLOCK_N):
                n_row = []
                for bn_idx in range(TILES_IN_BLOCK_N):
                    tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(tile, 0.0)
                    n_row.append(tile)
                bm_row.append(n_row)
            m_row.append(bm_row)
        res_gate.append(m_row)

    _matmul(
        lhs=lhs_buf,
        lhs_in_sbuf=False,
        rhs=rhs,
        rhs_in_sbuf=rhs_in_sbuf,
        res=res_gate,
        res_in_sbuf=True,
        TILES_IN_BLOCKS=(TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K),
        M=local_M,
        K=K,
        N=N,
        lhs_offset=lhs_offset,
    )

    RESULT_N = N

    for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
        for bm_tile_idx in nl.affine_range(TILES_IN_BLOCK_M):
            for n_blk_idx in nl.affine_range(NUM_BLOCK_N):
                for bn_tile_idx in nl.affine_range(TILES_IN_BLOCK_N):
                    row_offset = m_blk_idx * BLOCK_M + bm_tile_idx * TILE_M
                    col_offset = n_blk_idx * BLOCK_N + bn_tile_idx * TILE_N
                    inner_offset = result_offset + row_offset * RESULT_N + col_offset
                    src_tile = res_gate[m_blk_idx][bm_tile_idx][n_blk_idx][bn_tile_idx]
                    nisa.dma_copy(
                        dst=result_tensor.ap(
                            pattern=[[RESULT_N, TILE_M], [1, TILE_N]],
                            offset=inner_offset,
                            scalar_offset=numerator,
                            indirect_dim=0,
                        ),
                        src=src_tile,
                    )


def _launch_collective_permutes_hbm(
    buf_src: nl.ndarray,
    buf_dst: nl.ndarray,
    replica_group: ReplicaGroup,
    CHANNEL_N: int,
    M: int,
    RANK_N: int,
    LNC_N: int,
    K: int,
) -> None:
    """
    Launch collective permutes for HBM mode.

    Reshapes source and destination buffers by channel, then issues
    implicit collective permute across all channels.

    Args:
        buf_src (nl.ndarray): Source buffer in shared HBM.
        buf_dst (nl.ndarray): Destination buffer in shared HBM.
        replica_group (ReplicaGroup): Replica group for collective communication.
        CHANNEL_N (int): Number of communication channels.
        M (int): Total M dimension across all ranks.
        RANK_N (int): Number of ranks.
        LNC_N (int): Number of LNC programs.
        K (int): Shared dimension size.

    Returns:
        None: Data is permuted from buf_src to buf_dst.
    """
    srcs_by_channel = []
    dsts_by_channel = []
    channel_ids = []
    for channel_idx in range(CHANNEL_N):
        buf_src_reshaped = buf_src.reshape((CHANNEL_N, LNC_N, M // RANK_N // LNC_N // CHANNEL_N, K))
        buf_dst_reshaped = buf_dst.reshape((CHANNEL_N, LNC_N, M // RANK_N // LNC_N // CHANNEL_N, K))
        srcs_by_channel.append([buf_src_reshaped[channel_idx]])
        dsts_by_channel.append([buf_dst_reshaped[channel_idx]])
        channel_ids.append(channel_idx)
    ncc.collective_permute_implicit(
        srcs_by_channel=srcs_by_channel,
        dsts_by_channel=dsts_by_channel,
        replica_group=replica_group,
        channel_ids=channel_ids,
    )


def _launch_collective_permutes_sbuf(
    buf_src: nl.ndarray,
    buf_dst: nl.ndarray,
    replica_group: ReplicaGroup,
    CHANNEL_N: int,
    NUM_BLOCK_K: int,
    TILES_IN_BLOCK_K: int,
    NUM_BLOCK_M: int,
    BLOCK_M: int,
) -> None:
    """
    Launch collective permutes for SBUF mode.

    Slices flattened SBUF buffers by channel and issues implicit collective
    permute across all channels.

    Args:
        buf_src (nl.ndarray): Source buffer in SBUF (flattened).
        buf_dst (nl.ndarray): Destination buffer in SBUF (flattened).
        replica_group (ReplicaGroup): Replica group for collective communication.
        CHANNEL_N (int): Number of communication channels.
        NUM_BLOCK_K (int): Number of K blocks.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        NUM_BLOCK_M (int): Number of M blocks.
        BLOCK_M (int): Block size along M dimension.

    Returns:
        None: Data is permuted from buf_src to buf_dst.
    """
    channel_offset = NUM_BLOCK_K * TILES_IN_BLOCK_K * NUM_BLOCK_M * BLOCK_M
    srcs_by_channel = []
    dsts_by_channel = []
    channel_ids = []
    for channel_idx in range(CHANNEL_N):
        buf_src_slice = buf_src[:, channel_idx * channel_offset : (channel_idx + 1) * channel_offset]
        buf_dst_slice = buf_dst[:, channel_idx * channel_offset : (channel_idx + 1) * channel_offset]
        srcs_by_channel.append([buf_src_slice])
        dsts_by_channel.append([buf_dst_slice])
        channel_ids.append(channel_idx)
    ncc.collective_permute_implicit(
        srcs_by_channel=srcs_by_channel,
        dsts_by_channel=dsts_by_channel,
        replica_group=replica_group,
        channel_ids=channel_ids,
    )


def _compute_matmul_sbuf(
    lhs_buf: list,
    rhs: nl.ndarray,
    result_tensor: nl.ndarray,
    numerator: int,
    rhs_in_sbuf: bool,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_N: int,
    TILES_IN_BLOCK_K: int,
    BLOCK_M: int,
    BLOCK_N: int,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    local_M: int,
    K: int,
    N: int,
    channel: int,
    result_offset: int,
    CHANNEL_N: int,
    LNC_N: int,
    RANK_N: int,
    NUM_BLOCK_K: int,
) -> None:
    """
    Compute tiled matmul with lhs in SBUF and store results to HBM.

    Selects the lhs channel slice from SBUF, performs blocked matmul via _matmul,
    then writes result tiles back to the result tensor in HBM using indirect DMA.

    Args:
        lhs_buf (list): Nested list of lhs tiles in SBUF, indexed by [channel][k][bk][m].
        rhs (nl.ndarray): Right-hand side tensor (HBM or SBUF).
        result_tensor (nl.ndarray): Output tensor in shared HBM.
        numerator (int): Rank ID for indirect DMA offset.
        rhs_in_sbuf (bool): Whether rhs is pre-loaded in SBUF.
        TILE_M (int): Tile size along M dimension.
        TILE_N (int): Tile size along N dimension.
        TILE_K (int): Tile size along K dimension.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        BLOCK_M (int): Block size along M dimension.
        BLOCK_N (int): Block size along N dimension.
        NUM_BLOCK_M (int): Number of blocks along M dimension.
        NUM_BLOCK_N (int): Number of blocks along N dimension.
        local_M (int): Local M dimension per rank/LNC/channel.
        K (int): Shared dimension size.
        N (int): Output column dimension size.
        channel (int): Channel index to select from lhs_buf.
        result_offset (int): Offset into result_tensor for DMA writes.
        CHANNEL_N (int): Number of communication channels.
        LNC_N (int): Number of LNC programs.
        RANK_N (int): Number of ranks (tensor parallelism degree).
        NUM_BLOCK_K (int): Number of K blocks.

    Returns:
        None: Results are written to result_tensor via DMA.
    """
    single_channel_lhs = lhs_buf[channel]

    res_gate = []
    for m_idx in range(NUM_BLOCK_M):
        m_row = []
        for bm_idx in range(TILES_IN_BLOCK_M):
            bm_row = []
            for n_idx in range(NUM_BLOCK_N):
                n_row = []
                for bn_idx in range(TILES_IN_BLOCK_N):
                    tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(tile, 0.0)
                    n_row.append(tile)
                bm_row.append(n_row)
            m_row.append(bm_row)
        res_gate.append(m_row)

    _matmul(
        lhs=single_channel_lhs,
        lhs_in_sbuf=True,
        rhs=rhs,
        rhs_in_sbuf=rhs_in_sbuf,
        res=res_gate,
        res_in_sbuf=True,
        TILES_IN_BLOCKS=(TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K),
        M=local_M,
        K=K,
        N=N,
        lhs_offset=0,
    )

    RESULT_N = N

    for m_blk_idx in nl.affine_range(NUM_BLOCK_M):
        for bm_tile_idx in nl.affine_range(TILES_IN_BLOCK_M):
            for n_blk_idx in nl.affine_range(NUM_BLOCK_N):
                for bn_tile_idx in nl.affine_range(TILES_IN_BLOCK_N):
                    row_offset = m_blk_idx * BLOCK_M + bm_tile_idx * TILE_M
                    col_offset = n_blk_idx * BLOCK_N + bn_tile_idx * TILE_N
                    inner_offset = result_offset + row_offset * RESULT_N + col_offset
                    nisa.dma_copy(
                        dst=result_tensor.ap(
                            pattern=[[RESULT_N, TILE_M], [1, TILE_N]],
                            offset=inner_offset,
                            scalar_offset=numerator,
                            indirect_dim=0,
                        ),
                        src=res_gate[m_blk_idx][bm_tile_idx][n_blk_idx][bn_tile_idx],
                    )
