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
"""Fine-grained ring-based all-gather kernel for TRN2."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from . import fgcc


@nki.jit
def fine_grained_allgather(
    lhs: nl.ndarray,
    tp_degree: int,
    num_groups: int,
    force_hbm_cc: bool = False,
) -> nl.ndarray:
    """
    Fine-grained ring-based all-gather kernel for TRN2.

    Performs all-gather on lhs across ranks along the row dimension using
    ring-based collective permute with double buffering to overlap
    communication and data movement. Supports both SBUF and HBM
    communication paths with automatic selection based on tensor sizes.

    TODO: Specify intended usage range (e.g., sequence length, batch size)

    Dimensions:
        m: Local rows per rank (before all-gather).
        M: Total rows after all-gather (m * tp_degree).
        K: Column dimension (preserved across all-gather).

    Args:
        lhs (nl.ndarray): [m, K], Input tensor, row-sharded across ranks.
        tp_degree (int): Tensor parallelism degree (number of ranks). Must be even.
            Supported values: 4, 8, 16, 32, 64, 128.
        num_groups (int): Number of replica groups for collective communication.
        force_hbm_cc (bool): If True, force HBM collective communication path
            even when SBUF path is feasible.

    Returns:
        result (nl.ndarray): [RANK_N, ...], Fully gathered tensor in shared HBM.
            Shape depends on communication path (SBUF vs HBM).

    Notes:
        - tp_degree must be even.
        - M must be divisible by (RANK_N * LNC_N * CHANNEL_N).
        - Platform target is TRN2 only.

    Pseudocode:
        result = zeros(RANK_N, ..., local_M, K)
        buf0 = load(lhs)
        result[my_rank] = buf0  # iteration 0: local data
        collective_permute(buf0 -> buf1)
        result[received_rank] = buf1  # iteration 1
        for step in range(1, RANK_N // 2):
            collective_permute(buf1 -> buf0)
            result[received_rank] = buf0  # even iteration
            collective_permute(buf0 -> buf1)
            result[received_rank] = buf1  # odd iteration
    """
    m, K = lhs.shape
    RANK_N = tp_degree
    M = m * RANK_N

    kernel_assert(RANK_N % 2 == 0, f"tp_degree must be even, got {RANK_N}")

    dtype = lhs.dtype
    if dtype == nl.float32:
        DTYPE_SIZE = 4
    elif dtype == nl.bfloat16 or dtype == nl.float16:
        DTYPE_SIZE = 2
    else:
        DTYPE_SIZE = 4

    LNC_N = nl.num_programs(axes=0)
    lnc_id = nl.program_id(0) if LNC_N > 1 else 0

    if LNC_N == 2:
        if tp_degree == 4:
            CHANNEL_N = 2
        elif tp_degree == 8:
            CHANNEL_N = 2
        elif tp_degree == 16:
            CHANNEL_N = 2
        elif tp_degree == 32:
            CHANNEL_N = 2 if nisa.get_nc_version() >= 3 else 4
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
        f"M ({M}) must be divisible by RANK_N*LNC_N*CHANNEL_N ({RANK_N * LNC_N * CHANNEL_N})",
    )

    replica_group = ReplicaGroup(fgcc._generate_replica_groups(tp_degree=tp_degree, num_groups=num_groups))

    TILE_M = min(M // RANK_N // LNC_N // CHANNEL_N, nl.tile_size.gemm_stationary_fmax)
    TILE_K = min(K, nl.tile_size.pmax)

    TILES_IN_BLOCK_M = div_ceil(M // RANK_N // LNC_N // CHANNEL_N, TILE_M)
    TILES_IN_BLOCK_K = div_ceil(K, TILE_K)

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    NUM_BLOCK_M = div_ceil(M // RANK_N // LNC_N // CHANNEL_N, BLOCK_M)
    NUM_BLOCK_K = div_ceil(K, BLOCK_K)

    # Total SBUF budget for double-buffered operands (24 MiB)
    _TOTAL_SBUF_BUDGET_BYTES = 24 << 20

    # SBUF sizing: account for multi-channel LNC2 layout
    if LNC_N == 2 and CHANNEL_N > 1:
        sbuf_per_buffer = TILE_K * CHANNEL_N * NUM_BLOCK_K * TILES_IN_BLOCK_K * NUM_BLOCK_M * BLOCK_M * DTYPE_SIZE
    else:
        sbuf_per_buffer = M // RANK_N * K * DTYPE_SIZE

    lhs_in_sbuf = False
    if not force_hbm_cc and 2 * sbuf_per_buffer <= _TOTAL_SBUF_BUDGET_BYTES:
        lhs_in_sbuf = True
    else:
        if not force_hbm_cc:
            max_tiles = _TOTAL_SBUF_BUDGET_BYTES // BLOCK_K // TILE_M // DTYPE_SIZE
            TILES_IN_BLOCK_M = max(1, min(TILES_IN_BLOCK_M, max_tiles))

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    NUM_BLOCK_M = div_ceil(M // RANK_N // LNC_N // CHANNEL_N, BLOCK_M)
    local_M = M // RANK_N // LNC_N // CHANNEL_N

    if lhs_in_sbuf:
        result = nl.ndarray(
            (RANK_N, LNC_N, CHANNEL_N, local_M, K),
            dtype=dtype,
            buffer=nl.shared_hbm,
            name="result_sbuf",
        )
        lhs = lhs.reshape((LNC_N, CHANNEL_N, local_M, K))
    else:
        result = nl.ndarray(
            (RANK_N, CHANNEL_N, LNC_N, local_M, K),
            dtype=dtype,
            buffer=nl.shared_hbm,
            name="result_hbm",
        )
        lhs = lhs.reshape((CHANNEL_N, LNC_N, local_M, K))

    if lhs_in_sbuf:
        _run_sbuf_path(
            lhs,
            result,
            replica_group,
            lnc_id,
            dtype,
            TILE_M,
            TILE_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_K,
            BLOCK_M,
            BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_K,
            local_M,
            K,
            CHANNEL_N,
            LNC_N,
            RANK_N,
        )
    else:
        _run_hbm_path(
            lhs,
            result,
            replica_group,
            lnc_id,
            dtype,
            TILE_M,
            TILE_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_K,
            BLOCK_M,
            BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_K,
            local_M,
            K,
            M,
            CHANNEL_N,
            LNC_N,
            RANK_N,
        )

    return result


def _copy_gathered_data_hbm(
    buf: nl.ndarray,
    result_tensor: nl.ndarray,
    numerator: int,
    TILE_M: int,
    TILE_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_K: int,
    BLOCK_M: int,
    BLOCK_K: int,
    NUM_BLOCK_M: int,
    NUM_BLOCK_K: int,
    local_M: int,
    K: int,
    buf_offset: int,
    result_offset: int,
) -> None:
    """
    Tile-copy gathered data from HBM buffer to result tensor with dynamic rank indexing.

    Reads (TILE_M, TILE_K) tiles from the HBM buffer at buf_offset and writes them
    to the result tensor at result_offset, using scalar_offset for the dynamic rank
    dimension determined at runtime by collective_permute_implicit.

    Args:
        buf (nl.ndarray): Source HBM buffer containing gathered data.
        result_tensor (nl.ndarray): Destination HBM result tensor.
        numerator (int): Dynamic rank index from collective_permute_implicit.
        TILE_M (int): Tile size along M dimension.
        TILE_K (int): Tile size along K dimension.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        BLOCK_M (int): Block size along M dimension (TILE_M * TILES_IN_BLOCK_M).
        BLOCK_K (int): Block size along K dimension (TILE_K * TILES_IN_BLOCK_K).
        NUM_BLOCK_M (int): Number of blocks along M dimension.
        NUM_BLOCK_K (int): Number of blocks along K dimension.
        local_M (int): Local M dimension per rank per LNC per channel.
        K (int): Column dimension.
        buf_offset (int): Flat element offset into buf for the current channel/LNC slice.
        result_offset (int): Flat element offset into result_tensor for the current slice.

    Returns:
        None

    Notes:
        - Uses scalar_offset with indirect_dim=0 for runtime rank indexing on result_tensor.
    """
    for m_block_idx in range(NUM_BLOCK_M):
        for m_tile_idx in range(TILES_IN_BLOCK_M):
            for k_block_idx in range(NUM_BLOCK_K):
                for k_tile_idx in range(TILES_IN_BLOCK_K):
                    tile = nl.ndarray((TILE_M, TILE_K), dtype=buf.dtype, buffer=nl.sbuf)
                    row = m_block_idx * BLOCK_M + m_tile_idx * TILE_M
                    col = k_block_idx * BLOCK_K + k_tile_idx * TILE_K
                    src_off = buf_offset + row * K + col
                    nisa.dma_copy(
                        dst=tile,
                        src=buf.ap(pattern=[[K, TILE_M], [1, TILE_K]], offset=src_off),
                    )
                    dst_off = result_offset + row * K + col
                    nisa.dma_copy(
                        dst=result_tensor.ap(
                            pattern=[[K, TILE_M], [1, TILE_K]],
                            offset=dst_off,
                            scalar_offset=numerator,
                            indirect_dim=0,
                        ),
                        src=tile,
                    )


def _copy_gathered_data_sbuf(
    lhs_buf: list,
    result_tensor: nl.ndarray,
    numerator: int,
    channel_idx: int,
    TILE_M: int,
    TILE_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_K: int,
    BLOCK_M: int,
    BLOCK_K: int,
    NUM_BLOCK_M: int,
    NUM_BLOCK_K: int,
    local_M: int,
    K: int,
    result_offset: int,
) -> None:
    """
    Copy gathered data from SBUF transposed tiles to HBM result tensor.

    SBUF data is stored as nested list [channel][k][bk][m] of (TILE_K, BLOCK_M) tensors.
    Extracts (TILE_K, TILE_M) sub-tiles, transposes them to (TILE_M, TILE_K) via
    nc_transpose through PSUM, then writes to HBM result with scalar_offset for
    dynamic rank indexing.

    Args:
        lhs_buf (list): Nested SBUF tile list [channel][k_block][k_tile][m_block],
            each element is (TILE_K, BLOCK_M).
        result_tensor (nl.ndarray): Destination HBM result tensor.
        numerator (int): Dynamic rank index from collective_permute_implicit.
        channel_idx (int): Current channel index to select from lhs_buf.
        TILE_M (int): Tile size along M dimension.
        TILE_K (int): Tile size along K dimension.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        BLOCK_M (int): Block size along M dimension (TILE_M * TILES_IN_BLOCK_M).
        BLOCK_K (int): Block size along K dimension (TILE_K * TILES_IN_BLOCK_K).
        NUM_BLOCK_M (int): Number of blocks along M dimension.
        NUM_BLOCK_K (int): Number of blocks along K dimension.
        local_M (int): Local M dimension per rank per LNC per channel.
        K (int): Column dimension.
        result_offset (int): Flat element offset into result_tensor for the current slice.

    Returns:
        None

    Notes:
        - Uses nc_transpose via PSUM to transpose (TILE_K, TILE_M) -> (TILE_M, TILE_K).
        - Uses scalar_offset with indirect_dim=0 for runtime rank indexing on result_tensor.
    """
    single_channel_lhs = lhs_buf[channel_idx]

    for k_block_idx in range(NUM_BLOCK_K):
        for m_block_idx in range(NUM_BLOCK_M):
            for m_tile_idx in range(TILES_IN_BLOCK_M):
                for k_tile_idx in range(TILES_IN_BLOCK_K):
                    sub_tile = single_channel_lhs[k_block_idx][k_tile_idx][m_block_idx][
                        :, m_tile_idx * TILE_M : (m_tile_idx + 1) * TILE_M
                    ]
                    psum_tile = nl.ndarray(
                        (TILE_M, TILE_K),
                        dtype=result_tensor.dtype,
                        buffer=nl.psum,
                    )
                    nisa.nc_transpose(dst=psum_tile, data=sub_tile)
                    sbuf_tile = nl.ndarray(
                        (TILE_M, TILE_K),
                        dtype=result_tensor.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.tensor_copy(dst=sbuf_tile, src=psum_tile)

                    row = m_block_idx * BLOCK_M + m_tile_idx * TILE_M
                    col = k_block_idx * BLOCK_K + k_tile_idx * TILE_K
                    dst_off = result_offset + row * K + col
                    nisa.dma_copy(
                        dst=result_tensor.ap(
                            pattern=[[K, TILE_M], [1, TILE_K]],
                            offset=dst_off,
                            scalar_offset=numerator,
                            indirect_dim=0,
                        ),
                        src=sbuf_tile,
                    )


def _run_sbuf_path(
    lhs: nl.ndarray,
    result: nl.ndarray,
    replica_group: ReplicaGroup,
    lnc_id: int,
    dtype: nki.dtype,
    TILE_M: int,
    TILE_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_K: int,
    BLOCK_M: int,
    BLOCK_K: int,
    NUM_BLOCK_M: int,
    NUM_BLOCK_K: int,
    local_M: int,
    K: int,
    CHANNEL_N: int,
    LNC_N: int,
    RANK_N: int,
) -> None:
    """
    SBUF communication path for fine_grained_allgather.

    Transposes lhs tiles into SBUF as nested list [channel][k][bk][m] of (TILE_K, BLOCK_M)
    tensors, flattens to 2D for collective_permute_implicit, then copies gathered data
    back to HBM result via nc_transpose. Uses double buffering (buf0/buf1) to overlap
    communication with data movement across ring iterations.

    Args:
        lhs (nl.ndarray): [LNC_N, CHANNEL_N, local_M, K], Reshaped input tensor.
        result (nl.ndarray): [RANK_N, LNC_N, CHANNEL_N, local_M, K], Output HBM tensor.
        replica_group (ReplicaGroup): Collective communication replica group.
        lnc_id (int): Current LNC program ID.
        dtype (nki.dtype): Data type for buffer allocation.
        TILE_M (int): Tile size along M dimension.
        TILE_K (int): Tile size along K dimension.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        BLOCK_M (int): Block size along M (TILE_M * TILES_IN_BLOCK_M).
        BLOCK_K (int): Block size along K (TILE_K * TILES_IN_BLOCK_K).
        NUM_BLOCK_M (int): Number of blocks along M.
        NUM_BLOCK_K (int): Number of blocks along K.
        local_M (int): Local M dimension per rank per LNC per channel.
        K (int): Column dimension.
        CHANNEL_N (int): Number of communication channels.
        LNC_N (int): Number of LNC programs.
        RANK_N (int): Tensor parallelism degree.

    Returns:
        None: Result is written in-place to the result tensor.

    Notes:
        - Allocates nested SBUF buffers for double buffering.
        - Flattens nested buffers to 2D for collective_permute_implicit.
        - Ring iterations alternate between buf0 and buf1.
    """
    # Allocate nested SBUF tile buffers: [channel][k][bk][m] = (TILE_K, BLOCK_M)
    buf0_nested = []
    buf1_nested = []
    for channel_alloc_idx in range(CHANNEL_N):
        ch_list0, ch_list1 = [], []
        for k_alloc_idx in range(NUM_BLOCK_K):
            k_list0, k_list1 = [], []
            for bk_alloc_idx in range(TILES_IN_BLOCK_K):
                bk_list0, bk_list1 = [], []
                for m_alloc_idx in range(NUM_BLOCK_M):
                    bk_list0.append(nl.ndarray((TILE_K, BLOCK_M), dtype=dtype, buffer=nl.sbuf))
                    bk_list1.append(nl.ndarray((TILE_K, BLOCK_M), dtype=dtype, buffer=nl.sbuf))
                k_list0.append(bk_list0)
                k_list1.append(bk_list1)
            ch_list0.append(k_list0)
            ch_list1.append(k_list1)
        buf0_nested.append(ch_list0)
        buf1_nested.append(ch_list1)

    # Transpose lhs tiles into buf0_nested
    for channel_idx in range(CHANNEL_N):
        for k_block_idx in range(NUM_BLOCK_K):
            for k_tile_idx in range(TILES_IN_BLOCK_K):
                for m_block_idx in range(NUM_BLOCK_M):
                    row_start = m_block_idx * BLOCK_M
                    col_start = (k_block_idx * TILES_IN_BLOCK_K + k_tile_idx) * TILE_K
                    src_offset = (
                        lnc_id * CHANNEL_N * local_M * K + channel_idx * local_M * K + row_start * K + col_start
                    )
                    nisa.dma_transpose(
                        dst=buf0_nested[channel_idx][k_block_idx][k_tile_idx][m_block_idx].ap(
                            pattern=[[BLOCK_M, TILE_K], [1, 1], [1, 1], [1, BLOCK_M]],
                        ),
                        src=lhs.ap(
                            pattern=[[K, BLOCK_M], [1, 1], [1, 1], [1, TILE_K]],
                            offset=src_offset,
                        ),
                    )

    # Flatten to 2D for collective permute
    flat_size = CHANNEL_N * NUM_BLOCK_K * TILES_IN_BLOCK_K * NUM_BLOCK_M * BLOCK_M
    buf0_flat = nl.ndarray((TILE_K, flat_size), dtype=dtype, buffer=nl.sbuf)
    buf1_flat = nl.ndarray((TILE_K, flat_size), dtype=dtype, buffer=nl.sbuf)

    _pack_nested_to_flat(
        buf0_nested,
        buf0_flat,
        CHANNEL_N,
        NUM_BLOCK_K,
        TILES_IN_BLOCK_K,
        NUM_BLOCK_M,
        BLOCK_M,
    )

    result_slice_size = local_M * K

    # Iteration 0: process local data
    for channel_idx in range(CHANNEL_N):
        numerator = ncc.collective_permute_implicit_current_processing_rank_id(
            iteration_id=0,
            channel_id=channel_idx,
            replica_group=replica_group,
        )
        result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
        _copy_gathered_data_sbuf(
            buf0_nested,
            result,
            numerator,
            channel_idx,
            TILE_M,
            TILE_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_K,
            BLOCK_M,
            BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_K,
            local_M,
            K,
            result_offset,
        )

    # First collective permute: buf0 -> buf1
    fgcc._launch_collective_permutes_sbuf(
        buf0_flat,
        buf1_flat,
        replica_group,
        CHANNEL_N,
        NUM_BLOCK_K,
        TILES_IN_BLOCK_K,
        NUM_BLOCK_M,
        BLOCK_M,
    )
    _unpack_flat_to_nested(
        buf1_flat,
        buf1_nested,
        CHANNEL_N,
        NUM_BLOCK_K,
        TILES_IN_BLOCK_K,
        NUM_BLOCK_M,
        BLOCK_M,
    )

    # Iteration 1: first received data
    for channel_idx in range(CHANNEL_N):
        numerator = ncc.collective_permute_implicit_current_processing_rank_id(
            iteration_id=1,
            channel_id=channel_idx,
            replica_group=replica_group,
        )
        result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
        _copy_gathered_data_sbuf(
            buf1_nested,
            result,
            numerator,
            channel_idx,
            TILE_M,
            TILE_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_K,
            BLOCK_M,
            BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_K,
            local_M,
            K,
            result_offset,
        )

    # Ring iterations: alternate between buf0/buf1
    for step_idx in nl.sequential_range(1, RANK_N // 2):
        # Pack buf1 nested -> flat, permute buf1 -> buf0, unpack buf0 flat -> nested
        _pack_nested_to_flat(
            buf1_nested,
            buf1_flat,
            CHANNEL_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_K,
            NUM_BLOCK_M,
            BLOCK_M,
        )
        fgcc._launch_collective_permutes_sbuf(
            buf1_flat,
            buf0_flat,
            replica_group,
            CHANNEL_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_K,
            NUM_BLOCK_M,
            BLOCK_M,
        )
        _unpack_flat_to_nested(
            buf0_flat,
            buf0_nested,
            CHANNEL_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_K,
            NUM_BLOCK_M,
            BLOCK_M,
        )

        # Even step: copy from buf0
        for channel_idx in range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=2 * step_idx,
                channel_id=channel_idx,
                replica_group=replica_group,
            )
            result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
            _copy_gathered_data_sbuf(
                buf0_nested,
                result,
                numerator,
                channel_idx,
                TILE_M,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_K,
                NUM_BLOCK_M,
                NUM_BLOCK_K,
                local_M,
                K,
                result_offset,
            )

        # Pack buf0 nested -> flat, permute buf0 -> buf1, unpack buf1 flat -> nested
        _pack_nested_to_flat(
            buf0_nested,
            buf0_flat,
            CHANNEL_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_K,
            NUM_BLOCK_M,
            BLOCK_M,
        )
        fgcc._launch_collective_permutes_sbuf(
            buf0_flat,
            buf1_flat,
            replica_group,
            CHANNEL_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_K,
            NUM_BLOCK_M,
            BLOCK_M,
        )
        _unpack_flat_to_nested(
            buf1_flat,
            buf1_nested,
            CHANNEL_N,
            NUM_BLOCK_K,
            TILES_IN_BLOCK_K,
            NUM_BLOCK_M,
            BLOCK_M,
        )

        # Odd step: copy from buf1
        for channel_idx in range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=2 * step_idx + 1,
                channel_id=channel_idx,
                replica_group=replica_group,
            )
            result_offset = (lnc_id * CHANNEL_N + channel_idx) * result_slice_size
            _copy_gathered_data_sbuf(
                buf1_nested,
                result,
                numerator,
                channel_idx,
                TILE_M,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_K,
                NUM_BLOCK_M,
                NUM_BLOCK_K,
                local_M,
                K,
                result_offset,
            )


def _run_hbm_path(
    lhs: nl.ndarray,
    result: nl.ndarray,
    replica_group: ReplicaGroup,
    lnc_id: int,
    dtype: nki.dtype,
    TILE_M: int,
    TILE_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_K: int,
    BLOCK_M: int,
    BLOCK_K: int,
    NUM_BLOCK_M: int,
    NUM_BLOCK_K: int,
    local_M: int,
    K: int,
    M: int,
    CHANNEL_N: int,
    LNC_N: int,
    RANK_N: int,
) -> None:
    """
    HBM communication path for fine_grained_allgather.

    Allocates double HBM buffers (buf0, buf1) of shape (M // RANK_N, K).
    Copies local lhs into buf0, then runs ring iterations using
    collective_permute_implicit on HBM buffers. Each iteration tile-copies
    received data to the correct rank slot in the result tensor.

    Args:
        lhs (nl.ndarray): [CHANNEL_N, LNC_N, local_M, K], Reshaped input tensor.
        result (nl.ndarray): [RANK_N, CHANNEL_N, LNC_N, local_M, K], Output HBM tensor.
        replica_group (ReplicaGroup): Collective communication replica group.
        lnc_id (int): Current LNC program ID.
        dtype (nki.dtype): Data type for buffer allocation.
        TILE_M (int): Tile size along M dimension.
        TILE_K (int): Tile size along K dimension.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        BLOCK_M (int): Block size along M (TILE_M * TILES_IN_BLOCK_M).
        BLOCK_K (int): Block size along K (TILE_K * TILES_IN_BLOCK_K).
        NUM_BLOCK_M (int): Number of blocks along M.
        NUM_BLOCK_K (int): Number of blocks along K.
        local_M (int): Local M dimension per rank per LNC per channel.
        K (int): Column dimension.
        M (int): Total rows after all-gather (m * RANK_N).
        CHANNEL_N (int): Number of communication channels.
        LNC_N (int): Number of LNC programs.
        RANK_N (int): Tensor parallelism degree.

    Returns:
        None: Result is written in-place to the result tensor.

    Notes:
        - Allocates two shared HBM buffers for double buffering.
        - Ring iterations alternate between buf0 and buf1.
    """
    buf0 = nl.ndarray(
        (M // RANK_N, K),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="fg_buf0",
    )
    buf1 = nl.ndarray(
        (M // RANK_N, K),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="fg_buf1",
    )

    buf_slice_size = local_M * K
    result_slice_size = local_M * K

    # Load initial data from lhs into buf0
    for channel_idx in range(CHANNEL_N):
        offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
        nisa.dma_copy(
            dst=buf0.ap(pattern=[[K, local_M], [1, K]], offset=offset),
            src=lhs.ap(pattern=[[K, local_M], [1, K]], offset=offset),
        )

    # Iteration 0: process local data
    for channel_idx in range(CHANNEL_N):
        numerator = ncc.collective_permute_implicit_current_processing_rank_id(
            iteration_id=0,
            channel_id=channel_idx,
            replica_group=replica_group,
        )
        buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
        result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
        _copy_gathered_data_hbm(
            buf0,
            result,
            numerator,
            TILE_M,
            TILE_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_K,
            BLOCK_M,
            BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_K,
            local_M,
            K,
            buf_offset,
            result_offset,
        )

    # First collective permute: buf0 -> buf1
    fgcc._launch_collective_permutes_hbm(buf0, buf1, replica_group, CHANNEL_N, M, RANK_N, LNC_N, K)

    # Iteration 1: first received data
    for channel_idx in range(CHANNEL_N):
        numerator = ncc.collective_permute_implicit_current_processing_rank_id(
            iteration_id=1,
            channel_id=channel_idx,
            replica_group=replica_group,
        )
        buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
        result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
        _copy_gathered_data_hbm(
            buf1,
            result,
            numerator,
            TILE_M,
            TILE_K,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_K,
            BLOCK_M,
            BLOCK_K,
            NUM_BLOCK_M,
            NUM_BLOCK_K,
            local_M,
            K,
            buf_offset,
            result_offset,
        )

    # Ring iterations: alternate between buf0/buf1
    for step_idx in nl.sequential_range(1, RANK_N // 2):
        fgcc._launch_collective_permutes_hbm(buf1, buf0, replica_group, CHANNEL_N, M, RANK_N, LNC_N, K)

        # Even step: copy from buf0
        for channel_idx in range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=2 * step_idx,
                channel_id=channel_idx,
                replica_group=replica_group,
            )
            buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
            result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
            _copy_gathered_data_hbm(
                buf0,
                result,
                numerator,
                TILE_M,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_K,
                NUM_BLOCK_M,
                NUM_BLOCK_K,
                local_M,
                K,
                buf_offset,
                result_offset,
            )

        fgcc._launch_collective_permutes_hbm(buf0, buf1, replica_group, CHANNEL_N, M, RANK_N, LNC_N, K)

        # Odd step: copy from buf1
        for channel_idx in range(CHANNEL_N):
            numerator = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=2 * step_idx + 1,
                channel_id=channel_idx,
                replica_group=replica_group,
            )
            buf_offset = (channel_idx * LNC_N + lnc_id) * buf_slice_size
            result_offset = (channel_idx * LNC_N + lnc_id) * result_slice_size
            _copy_gathered_data_hbm(
                buf1,
                result,
                numerator,
                TILE_M,
                TILE_K,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_K,
                BLOCK_M,
                BLOCK_K,
                NUM_BLOCK_M,
                NUM_BLOCK_K,
                local_M,
                K,
                buf_offset,
                result_offset,
            )


def _pack_nested_to_flat(
    nested: list,
    flat: nl.ndarray,
    CHANNEL_N: int,
    NUM_BLOCK_K: int,
    TILES_IN_BLOCK_K: int,
    NUM_BLOCK_M: int,
    BLOCK_M: int,
) -> None:
    """
    Pack nested SBUF tile list into flat 2D buffer for collective permute.

    Copies each (TILE_K, BLOCK_M) tile from nested[channel][k][bk][m] into the
    corresponding slice of the flat (TILE_K, flat_size) buffer, where flat_size =
    CHANNEL_N * NUM_BLOCK_K * TILES_IN_BLOCK_K * NUM_BLOCK_M * BLOCK_M.

    Args:
        nested (list): Nested SBUF tile list [channel][k_block][k_tile][m_block],
            each element is (TILE_K, BLOCK_M).
        flat (nl.ndarray): [TILE_K, flat_size], Destination flat 2D buffer.
        CHANNEL_N (int): Number of communication channels.
        NUM_BLOCK_K (int): Number of blocks along K.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        NUM_BLOCK_M (int): Number of blocks along M.
        BLOCK_M (int): Block size along M.

    Returns:
        None: flat buffer is written in-place.
    """
    for channel_idx in range(CHANNEL_N):
        for k_block_idx in range(NUM_BLOCK_K):
            for k_tile_idx in range(TILES_IN_BLOCK_K):
                for m_block_idx in range(NUM_BLOCK_M):
                    flat_offset = (
                        (channel_idx * NUM_BLOCK_K + k_block_idx) * TILES_IN_BLOCK_K + k_tile_idx
                    ) * NUM_BLOCK_M * BLOCK_M + m_block_idx * BLOCK_M
                    nisa.tensor_copy(
                        dst=flat[:, flat_offset : flat_offset + BLOCK_M],
                        src=nested[channel_idx][k_block_idx][k_tile_idx][m_block_idx],
                    )


def _unpack_flat_to_nested(
    flat: nl.ndarray,
    nested: list,
    CHANNEL_N: int,
    NUM_BLOCK_K: int,
    TILES_IN_BLOCK_K: int,
    NUM_BLOCK_M: int,
    BLOCK_M: int,
) -> None:
    """
    Unpack flat 2D buffer into nested SBUF tile list after collective permute.

    Copies each (TILE_K, BLOCK_M) slice from the flat buffer back into the
    corresponding nested[channel][k][bk][m] tile.

    Args:
        flat (nl.ndarray): [TILE_K, flat_size], Source flat 2D buffer.
        nested (list): Nested SBUF tile list [channel][k_block][k_tile][m_block],
            each element is (TILE_K, BLOCK_M).
        CHANNEL_N (int): Number of communication channels.
        NUM_BLOCK_K (int): Number of blocks along K.
        TILES_IN_BLOCK_K (int): Number of K tiles per block.
        NUM_BLOCK_M (int): Number of blocks along M.
        BLOCK_M (int): Block size along M.

    Returns:
        None: nested buffer is written in-place.
    """
    for channel_idx in range(CHANNEL_N):
        for k_block_idx in range(NUM_BLOCK_K):
            for k_tile_idx in range(TILES_IN_BLOCK_K):
                for m_block_idx in range(NUM_BLOCK_M):
                    flat_offset = (
                        (channel_idx * NUM_BLOCK_K + k_block_idx) * TILES_IN_BLOCK_K + k_tile_idx
                    ) * NUM_BLOCK_M * BLOCK_M + m_block_idx * BLOCK_M
                    nisa.tensor_copy(
                        dst=nested[channel_idx][k_block_idx][k_tile_idx][m_block_idx],
                        src=flat[:, flat_offset : flat_offset + BLOCK_M],
                    )
