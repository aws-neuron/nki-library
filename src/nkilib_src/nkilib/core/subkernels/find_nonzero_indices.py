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

"""Find nonzero indices kernel using GpSimd nonzero_with_count ISA.

Single-chunk path eliminates nc_stream_shuffle entirely by DMA-ing odd core
results directly from partition q*32+16 and round-tripping counts through HBM.
Multi-chunk path uses double-buffered input loading to overlap DMA with compute.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import constants as nisa_constants

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil

# Constants for GpSimd nonzero_with_count ISA
_QUADRANT_SIZE = 32  # Size of each quadrant in partition dimension
_NUM_QUADRANTS = 4  # Number of quadrants (128 / 32)
_NUM_GPSIMD_CORES = 8  # Number of GpSimd cores that process in parallel
_GPSIMD_CORES_PER_QUADRANT = 2  # GpSimd cores per quadrant
_PARTITIONS_PER_GPSIMD = 16  # Partitions between each GpSimd core (0, 16, 32, ..., 112)
_SHUFFLE_IDENTITY = 255  # nc_stream_shuffle identity value (no shuffle for this partition)


@nki.jit
def find_nonzero_indices(
    input_tensor: nl.NkiTensor,
    col_start_id: nl.NkiTensor = None,
    n_cols: int = None,
    chunk_size: int = None,
    index_dtype: nki.dtype = nl.int32,
):
    """Find indices of nonzero elements along the T dimension.

    This kernel computes the indices of nonzero elements in an input tensor of shape [T, C].
    It finds indices along the T dimension for each column. The kernel is optimized
    for LNC2 sharding and uses the GpSimd nonzero_with_count ISA for efficient parallel
    processing of 8 columns at a time. Optimized for token counts up to 65536 and column
    counts up to 128.

    Dimensions:
        T: Sequence/token dimension (first dimension of input)
        C: Column dimension that used to calculate the non zero indices (second dimension of input)
        C_full: Full columns dimension from input tensor shape
        C_per_shard: Columns processed per LNC shard (C // NUM Shards)

    Args:
        input_tensor (nl.NkiTensor): [T, C], Input tensor on HBM. Nonzero elements are found
            along the T dimension for each column.
        col_start_id (nl.NkiTensor): [1], Optional HBM tensor containing the starting column
            index in the C dimension. If specified, only n_cols Columns starting from col_start_id are processed.
            If None, all C Columns are processed.
        n_cols (int): Number of columns (in C dimension) to process. Required when
            col_start_id is specified, ignored otherwise.
        chunk_size (int): Size of chunks for processing T dimension. If None, defaults to T.
            Must divide T evenly. Smaller chunk sizes reduce memory usage.
        index_dtype (nki.dtype): Data type for output indices tensor. Default is nl.int32.

    Returns:
        indices (nl.NkiTensor): [C, T] or [n_cols, T], Tensor containing nonzero indices.
            For each column c, the first N values are the T-indices of nonzero elements,
            followed by -1 padding values.
        nonzero_counts (nl.NkiTensor): [C] or [n_cols], Count of nonzero elements per column.

    Notes:
        - Requires LNC2 configuration (2 NeuronCores)
        - C must be divisible by 2 (for LNC2 sharding)
        - chunk_size must be divisible by 128 (partition size)
        - Uses GpSimd nonzero_with_count ISA which only operates on partitions [0, 16, 32, ..., 112]

    Pseudocode:
        for each column c in [0, C):
            count = 0
            for t in [0, T):
                if input_tensor[t, c] != 0:
                    indices[c, count] = t
                    count += 1
            # Pad remaining with -1
            for i in [count, T):
                indices[c, i] = -1
            nonzero_counts[c] = count
    """
    T_DIM, C_DIM = input_tensor.shape
    if col_start_id != None and n_cols != None:
        col_start_id_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=col_start_id_sbuf, src=col_start_id[0:1])
        C = n_cols
    else:
        col_start_id_sbuf = None
        C = C_DIM

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    C_per_shard = C // num_shards
    C_offset = C_per_shard * shard_id

    P_MAX = nl.tile_size.pmax
    T_TILE_SIZE = P_MAX
    C_TILE_SIZE = P_MAX

    if chunk_size == None:
        chunk_size = T_DIM
    kernel_assert(T_DIM % chunk_size == 0, f"T_DIM ({T_DIM}) must be divisible by chunk_size ({chunk_size})")
    CHUNK_T_TILES = chunk_size // T_TILE_SIZE
    NUM_CHUNKS = T_DIM // chunk_size

    indices = nl.ndarray((C, T_DIM), dtype=index_dtype, buffer=nl.shared_hbm)

    if NUM_CHUNKS > 1:
        sbuf_init = nl.ndarray((P_MAX, C_per_shard * T_DIM // P_MAX), dtype=index_dtype, buffer=nl.sbuf)
        nisa.memset(dst=sbuf_init, value=-1)
        reshaped_dst = indices.reshape((P_MAX * 2, C_per_shard * T_DIM // P_MAX))
        nisa.dma_copy(dst=reshaped_dst[P_MAX * shard_id : P_MAX * (shard_id + 1), :], src=sbuf_init)

    nonzero_counts = nl.ndarray((C,), dtype=nl.int32, buffer=nl.shared_hbm)
    nonzero_counts_local = nl.ndarray((1, C_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=nonzero_counts_local, value=0)

    n_column_rounds = div_ceil(C_per_shard, _NUM_GPSIMD_CORES)

    identity_sb = nl.shared_identity_matrix(P_MAX, dtype=nl.float32)

    TILES_PER_GROUP = min(8, CHUNK_T_TILES)
    NUM_GROUPS = div_ceil(CHUNK_T_TILES, TILES_PER_GROUP)

    for column_round_idx in range(n_column_rounds):
        n_columns_this_round = min(_NUM_GPSIMD_CORES, C_per_shard - _NUM_GPSIMD_CORES * column_round_idx)
        column_start_offset = column_round_idx * _NUM_GPSIMD_CORES + C_offset

        # Track cumulative offsets for writing indices
        offsets = nl.ndarray((1, _NUM_GPSIMD_CORES), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=offsets, value=0)

        if NUM_CHUNKS == 1:
            _process_chunk_single(
                input_tensor=input_tensor,
                col_start_id_sbuf=col_start_id_sbuf,
                indices=indices,
                nonzero_counts_local=nonzero_counts_local,
                identity_sb=identity_sb,
                column_round_idx=column_round_idx,
                column_start_offset=column_start_offset,
                n_columns_this_round=n_columns_this_round,
                C_DIM=C_DIM,
                C_offset=C_offset,
                C_per_shard=C_per_shard,
                chunk_size=chunk_size,
                CHUNK_T_TILES=CHUNK_T_TILES,
                T_DIM=T_DIM,
                TILES_PER_GROUP=TILES_PER_GROUP,
                NUM_GROUPS=NUM_GROUPS,
            )
        else:
            # === MULTI-CHUNK: double-buffered input + batched DMA store ===
            buf_a = nl.ndarray(
                (T_TILE_SIZE, CHUNK_T_TILES, _NUM_GPSIMD_CORES),
                dtype=input_tensor.dtype,
                buffer=nl.sbuf,
                name=f"input_sbuf_a_er-{column_round_idx}",
            )
            buf_b = nl.ndarray(
                (T_TILE_SIZE, CHUNK_T_TILES, _NUM_GPSIMD_CORES),
                dtype=input_tensor.dtype,
                buffer=nl.sbuf,
                name=f"input_sbuf_b_er-{column_round_idx}",
            )
            chunk_bufs = [buf_a, buf_b]

            _dma_load_chunk(
                input_tensor,
                col_start_id_sbuf,
                buf_a,
                column_start_offset,
                0 * chunk_size,
                n_columns_this_round,
                C_DIM,
                CHUNK_T_TILES,
            )
            if NUM_CHUNKS > 1:
                _dma_load_chunk(
                    input_tensor,
                    col_start_id_sbuf,
                    buf_b,
                    column_start_offset,
                    1 * chunk_size,
                    n_columns_this_round,
                    C_DIM,
                    CHUNK_T_TILES,
                )

            indices_sbuf = nl.ndarray(
                (C_TILE_SIZE, chunk_size + 1),
                dtype=nl.int32,
                buffer=nl.sbuf,
                name=f"indices_sbuf_er-{column_round_idx}",
            )

            for chunk_idx in range(NUM_CHUNKS):
                cur_input = chunk_bufs[chunk_idx % 2]

                input_gpsimd_aligned_sbuf = nl.ndarray(
                    (T_TILE_SIZE, CHUNK_T_TILES, C_TILE_SIZE),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                    name=f"aligned_sbuf_er-{column_round_idx}_ch-{chunk_idx}",
                )
                input_gpsimd_aligned_transposed_sbuf = nl.ndarray(
                    (C_TILE_SIZE, CHUNK_T_TILES, T_TILE_SIZE),
                    dtype=input_tensor.dtype,
                    buffer=nl.sbuf,
                    name=f"transposed_sbuf_er-{column_round_idx}_ch-{chunk_idx}",
                )

                if chunk_idx > 0:
                    for column_idx in range(n_columns_this_round):
                        nisa.tensor_copy(
                            dst=input_gpsimd_aligned_sbuf[:, :, column_idx * _PARTITIONS_PER_GPSIMD],
                            src=cur_input[:, :, column_idx],
                            engine=nisa.engine.scalar,
                        )

                    for t_tile_idx in range(CHUNK_T_TILES):
                        transposed_psum = nl.ndarray((C_TILE_SIZE, T_TILE_SIZE), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(
                            dst=transposed_psum,
                            stationary=input_gpsimd_aligned_sbuf[:, t_tile_idx, :],
                            moving=identity_sb[0:P_MAX, 0:P_MAX],
                            is_transpose=True,
                        )
                        nisa.tensor_copy(
                            dst=input_gpsimd_aligned_transposed_sbuf[:, t_tile_idx, :], src=transposed_psum
                        )

                    _store_direct(
                        indices_sbuf=indices_sbuf,
                        indices=indices,
                        offsets=offsets,
                        column_round_idx=column_round_idx,
                        chunk_idx=chunk_idx - 1,
                        n_columns_this_round=n_columns_this_round,
                        C_offset=C_offset,
                        chunk_size=chunk_size,
                        T_DIM=T_DIM,
                        name_prefix="even",
                        is_even=True,
                    )

                    quad_mask = [_PARTITIONS_PER_GPSIMD] + [_SHUFFLE_IDENTITY] * (_QUADRANT_SIZE - 1)
                    nisa.nc_stream_shuffle(dst=indices_sbuf, src=indices_sbuf, shuffle_mask=quad_mask)

                    _store_direct(
                        indices_sbuf=indices_sbuf,
                        indices=indices,
                        offsets=offsets,
                        column_round_idx=column_round_idx,
                        chunk_idx=chunk_idx - 1,
                        n_columns_this_round=n_columns_this_round,
                        C_offset=C_offset,
                        chunk_size=chunk_size,
                        T_DIM=T_DIM,
                        name_prefix="odd",
                        is_even=False,
                    )
                else:
                    for column_idx in range(n_columns_this_round):
                        nisa.tensor_copy(
                            dst=input_gpsimd_aligned_sbuf[:, :, column_idx * _PARTITIONS_PER_GPSIMD],
                            src=cur_input[:, :, column_idx],
                            engine=nisa.engine.scalar,
                        )

                    for t_tile_idx in range(CHUNK_T_TILES):
                        transposed_psum = nl.ndarray((C_TILE_SIZE, T_TILE_SIZE), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(
                            dst=transposed_psum,
                            stationary=input_gpsimd_aligned_sbuf[:, t_tile_idx, :],
                            moving=identity_sb[0:P_MAX, 0:P_MAX],
                            is_transpose=True,
                        )
                        nisa.tensor_copy(
                            dst=input_gpsimd_aligned_transposed_sbuf[:, t_tile_idx, :], src=transposed_psum
                        )

                if chunk_idx + 2 < NUM_CHUNKS:
                    _dma_load_chunk(
                        input_tensor,
                        col_start_id_sbuf,
                        chunk_bufs[chunk_idx % 2],
                        column_start_offset,
                        (chunk_idx + 2) * chunk_size,
                        n_columns_this_round,
                        C_DIM,
                        CHUNK_T_TILES,
                    )

                input_2d = input_gpsimd_aligned_transposed_sbuf.reshape((C_TILE_SIZE, chunk_size))

                nisa.nonzero_with_count(
                    dst=indices_sbuf, src=input_2d, index_offset=chunk_idx * chunk_size, padding_val=-1
                )

            # Store last chunk
            _store_direct(
                indices_sbuf=indices_sbuf,
                indices=indices,
                offsets=offsets,
                column_round_idx=column_round_idx,
                chunk_idx=NUM_CHUNKS - 1,
                n_columns_this_round=n_columns_this_round,
                C_offset=C_offset,
                chunk_size=chunk_size,
                T_DIM=T_DIM,
                name_prefix="even",
                is_even=True,
            )
            quad_mask = [_PARTITIONS_PER_GPSIMD] + [_SHUFFLE_IDENTITY] * (_QUADRANT_SIZE - 1)
            nisa.nc_stream_shuffle(dst=indices_sbuf, src=indices_sbuf, shuffle_mask=quad_mask)
            _store_direct(
                indices_sbuf=indices_sbuf,
                indices=indices,
                offsets=offsets,
                column_round_idx=column_round_idx,
                chunk_idx=NUM_CHUNKS - 1,
                n_columns_this_round=n_columns_this_round,
                C_offset=C_offset,
                chunk_size=chunk_size,
                T_DIM=T_DIM,
                name_prefix="odd",
                is_even=False,
            )

            nisa.tensor_copy(
                dst=nonzero_counts_local[
                    0:1,
                    column_round_idx * _NUM_GPSIMD_CORES : column_round_idx * _NUM_GPSIMD_CORES + n_columns_this_round,
                ],
                src=offsets[0:1, 0:n_columns_this_round],
            )

    nonzero_counts_reshape = nonzero_counts.reshape((1, C))
    nisa.dma_copy(dst=nonzero_counts_reshape[0:1, C_offset : C_offset + C_per_shard], src=nonzero_counts_local)
    return indices, nonzero_counts


def _store_direct(
    indices_sbuf,
    indices,
    offsets,
    column_round_idx,
    chunk_idx,
    n_columns_this_round,
    C_offset,
    chunk_size,
    T_DIM,
    name_prefix,
    is_even,
):
    """DMA store helper for multi-chunk path."""
    ot0 = nl.ndarray(
        (1, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_offset_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-0",
    )
    ot1 = nl.ndarray(
        (1, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_offset_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-1",
    )
    ot2 = nl.ndarray(
        (1, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_offset_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-2",
    )
    ot3 = nl.ndarray(
        (1, 1),
        dtype=nl.int32,
        buffer=nl.sbuf,
        name=f"{name_prefix}_offset_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-3",
    )
    offset_tiles = [ot0, ot1, ot2, ot3]

    for quadrant_idx in range(_NUM_QUADRANTS):
        column_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + (0 if is_even else 1)
        if column_idx >= n_columns_this_round:
            continue
        nisa.tensor_copy(dst=offset_tiles[quadrant_idx], src=offsets[0:1, column_idx : column_idx + 1])

    for quadrant_idx in range(_NUM_QUADRANTS):
        column_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + (0 if is_even else 1)
        if column_idx >= n_columns_this_round:
            continue
        out_col = C_offset + column_round_idx * _NUM_GPSIMD_CORES + column_idx
        nisa.dma_copy(
            dst=indices.ap(
                pattern=[[T_DIM, 1], [1, chunk_size]],
                offset=out_col * T_DIM,
                scalar_offset=offset_tiles[quadrant_idx],
                indirect_dim=1,
            ),
            src=indices_sbuf[quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1, 0:chunk_size],
        )

    for quadrant_idx in range(_NUM_QUADRANTS):
        column_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + (0 if is_even else 1)
        if column_idx >= n_columns_this_round:
            continue
        count_tile = nl.ndarray(
            (1, 1),
            dtype=nl.int32,
            buffer=nl.sbuf,
            name=f"{name_prefix}_count_tile_er-{column_round_idx}_ch-{chunk_idx}_qi-{quadrant_idx}",
        )
        nisa.tensor_copy(
            dst=count_tile,
            src=indices_sbuf[
                quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1, chunk_size : chunk_size + 1
            ],
        )
        nisa.tensor_tensor(
            dst=offsets[0:1, column_idx : column_idx + 1],
            data1=offsets[0:1, column_idx : column_idx + 1],
            data2=count_tile,
            op=nl.add,
        )


def _process_chunk_single(
    input_tensor,
    col_start_id_sbuf,
    indices,
    nonzero_counts_local,
    identity_sb,
    column_round_idx,
    column_start_offset,
    n_columns_this_round,
    C_DIM,
    C_offset,
    C_per_shard,
    chunk_size,
    CHUNK_T_TILES,
    T_DIM,
    TILES_PER_GROUP,
    NUM_GROUPS,
):
    """Single-chunk path: no shuffle needed, DMA odd results directly from partition q*32+16."""
    P_MAX = 128
    T_TILE_SIZE = P_MAX
    C_TILE_SIZE = P_MAX

    input_gpsimd_aligned_transposed_sbuf = nl.ndarray(
        (C_TILE_SIZE, CHUNK_T_TILES, T_TILE_SIZE), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    indices_sbuf = nl.ndarray((C_TILE_SIZE, chunk_size + 1), dtype=nl.int32, buffer=nl.sbuf)

    for group_idx in range(NUM_GROUPS):
        group_start = group_idx * TILES_PER_GROUP
        tiles_this_group = min(TILES_PER_GROUP, CHUNK_T_TILES - group_start)
        t_group_start = group_start * T_TILE_SIZE

        input_sbuf = nl.ndarray(
            (T_TILE_SIZE, tiles_this_group, _NUM_GPSIMD_CORES), dtype=input_tensor.dtype, buffer=nl.sbuf
        )
        input_gpsimd_aligned_sbuf = nl.ndarray(
            (T_TILE_SIZE, tiles_this_group, C_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf
        )

        if col_start_id_sbuf != None:
            nisa.dma_copy(
                dst=input_sbuf[:, 0:tiles_this_group, 0:n_columns_this_round],
                src=input_tensor.ap(
                    pattern=[[C_DIM, T_TILE_SIZE], [C_DIM * T_TILE_SIZE, tiles_this_group], [1, n_columns_this_round]],
                    offset=column_start_offset + (t_group_start * C_DIM),
                    scalar_offset=col_start_id_sbuf,
                    indirect_dim=1,
                ),
                dge_mode=nisa_constants.dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                dst=input_sbuf[:, 0:tiles_this_group, 0:n_columns_this_round],
                src=input_tensor[
                    t_group_start : t_group_start + tiles_this_group * T_TILE_SIZE,
                    column_start_offset : column_start_offset + n_columns_this_round,
                ]
                .reshape_dim(0, (tiles_this_group, T_TILE_SIZE))
                .permute((1, 0, 2)),
            )

        for column_idx in range(n_columns_this_round):
            nisa.tensor_copy(
                dst=input_gpsimd_aligned_sbuf[:, :, column_idx * _PARTITIONS_PER_GPSIMD],
                src=input_sbuf[:, :, column_idx],
                engine=nisa.engine.scalar,
            )

        for t_tile_local_idx in range(tiles_this_group):
            t_tile_global_idx = group_start + t_tile_local_idx
            transposed_psum = nl.ndarray((C_TILE_SIZE, T_TILE_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=transposed_psum,
                stationary=input_gpsimd_aligned_sbuf[:, t_tile_local_idx, :],
                moving=identity_sb[0:P_MAX, 0:P_MAX],
                is_transpose=True,
            )
            nisa.tensor_copy(dst=input_gpsimd_aligned_transposed_sbuf[:, t_tile_global_idx, :], src=transposed_psum)

    nisa.nonzero_with_count(dst=indices_sbuf, src=input_gpsimd_aligned_transposed_sbuf, index_offset=0, padding_val=-1)

    # Even core indices: DMA directly from partition q*32
    for quadrant_idx in range(_NUM_QUADRANTS):
        even_col_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT
        if even_col_idx < n_columns_this_round:
            out_col = C_offset + column_round_idx * _NUM_GPSIMD_CORES + even_col_idx
            nisa.dma_copy(
                dst=indices[out_col, 0:chunk_size],
                src=indices_sbuf[quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1, 0:chunk_size],
            )

    # Odd core indices: DMA directly from partition q*32+16 (no shuffle needed)
    for quadrant_idx in range(_NUM_QUADRANTS):
        odd_col_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + 1
        if odd_col_idx < n_columns_this_round:
            out_col = C_offset + column_round_idx * _NUM_GPSIMD_CORES + odd_col_idx
            nisa.dma_copy(
                dst=indices[out_col, 0:chunk_size],
                src=indices_sbuf[
                    quadrant_idx * _QUADRANT_SIZE + _PARTITIONS_PER_GPSIMD : quadrant_idx * _QUADRANT_SIZE
                    + _PARTITIONS_PER_GPSIMD
                    + 1,
                    0:chunk_size,
                ],
            )

    # Even counts: tensor_copy from partition q*32 (VE-readable)
    for quadrant_idx in range(_NUM_QUADRANTS):
        even_col_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT
        if even_col_idx < n_columns_this_round:
            count_tile = nl.ndarray(
                (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"even_count_er-{column_round_idx}_qi-{quadrant_idx}"
            )
            nisa.tensor_copy(
                dst=count_tile,
                src=indices_sbuf[
                    quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1, chunk_size : chunk_size + 1
                ],
            )
            nisa.tensor_copy(
                dst=nonzero_counts_local[
                    0:1,
                    column_round_idx * _NUM_GPSIMD_CORES + even_col_idx : column_round_idx * _NUM_GPSIMD_CORES
                    + even_col_idx
                    + 1,
                ],
                src=count_tile,
            )

    # Odd counts: round-trip through HBM (partition q*32+16 not VE-readable)
    count_scratch_hbm = nl.ndarray((1, _NUM_QUADRANTS), dtype=nl.int32, buffer=nl.private_hbm)

    for quadrant_idx in range(_NUM_QUADRANTS):
        odd_col_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + 1
        if odd_col_idx < n_columns_this_round:
            nisa.dma_copy(
                dst=count_scratch_hbm[0:1, quadrant_idx : quadrant_idx + 1],
                src=indices_sbuf[
                    quadrant_idx * _QUADRANT_SIZE + _PARTITIONS_PER_GPSIMD : quadrant_idx * _QUADRANT_SIZE
                    + _PARTITIONS_PER_GPSIMD
                    + 1,
                    chunk_size : chunk_size + 1,
                ],
            )

    for quadrant_idx in range(_NUM_QUADRANTS):
        odd_col_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + 1
        if odd_col_idx < n_columns_this_round:
            odd_count_tile = nl.ndarray(
                (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"odd_count_er-{column_round_idx}_qi-{quadrant_idx}"
            )
            nisa.dma_copy(
                dst=odd_count_tile,
                src=count_scratch_hbm[0:1, quadrant_idx : quadrant_idx + 1],
            )
            nisa.tensor_copy(
                dst=nonzero_counts_local[
                    0:1,
                    column_round_idx * _NUM_GPSIMD_CORES + odd_col_idx : column_round_idx * _NUM_GPSIMD_CORES
                    + odd_col_idx
                    + 1,
                ],
                src=odd_count_tile,
            )


def _dma_load_chunk(
    input_tensor,
    col_start_id_sbuf,
    dst,
    column_start_offset,
    t_chunk_start,
    n_columns_this_round,
    C_DIM,
    CHUNK_T_TILES,
):
    """DMA load helper for double-buffered multi-chunk input loading."""
    P_MAX = nl.tile_size.pmax
    if col_start_id_sbuf != None:
        nisa.dma_copy(
            dst=dst[:, 0:CHUNK_T_TILES, 0:n_columns_this_round],
            src=input_tensor.ap(
                pattern=[[C_DIM, P_MAX], [C_DIM * P_MAX, CHUNK_T_TILES], [1, n_columns_this_round]],
                offset=column_start_offset + (t_chunk_start * C_DIM),
                scalar_offset=col_start_id_sbuf,
                indirect_dim=1,
            ),
            dge_mode=nisa_constants.dge_mode.hwdge,
        )
    else:
        nisa.dma_copy(
            dst=dst[:, 0:CHUNK_T_TILES, 0:n_columns_this_round],
            src=input_tensor[
                t_chunk_start : t_chunk_start + CHUNK_T_TILES * P_MAX,
                column_start_offset : column_start_offset + n_columns_this_round,
            ]
            .reshape_dim(0, (CHUNK_T_TILES, P_MAX))
            .permute((1, 0, 2)),
        )
