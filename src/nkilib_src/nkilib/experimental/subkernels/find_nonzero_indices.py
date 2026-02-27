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

"""Find nonzero indices kernel using GpSimd nonzero_with_count ISA."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil

# Constants for GpSimd nonzero_with_count ISA
_QUADRANT_SIZE = 32  # Size of each quadrant in partition dimension
_NUM_QUADRANTS = 4  # Number of quadrants (128 / 32)
_NUM_GPSIMD_CORES = 8  # Number of GpSimd cores that process in parallel
_GPSIMD_CORES_PER_QUADRANT = 2  # GpSimd cores per quadrant
_PARTITIONS_PER_GPSIMD = 16  # Partitions between each GpSimd core (0, 16, 32, ..., 112)


@nki.jit
def find_nonzero_indices(
    input_tensor: nl.ndarray,
    row_start_id: nl.ndarray = None,
    n_rows: int = None,
    chunk_size: int = None,
    index_dtype: nki.dtype = nl.int32,
):
    """Find indices of nonzero elements along the T dimension.

    This kernel computes the indices of nonzero elements in an input tensor of shape [T, E].
    It finds indices along the T dimension for each column (expert). The kernel is optimized
    for LNC2 sharding and uses the GpSimd nonzero_with_count ISA for efficient parallel
    processing of 8 experts at a time. Optimized for token counts up to 65536 and expert
    counts up to 128.

    Dimensions:
        T: Sequence/token dimension (first dimension of input)
        E: Expert dimension (second dimension of input)
        E_full: Full expert dimension from input tensor shape
        E_per_shard: Experts processed per LNC shard (E // 2)

    Args:
        input_tensor (nl.ndarray): [T, E], Input tensor on HBM. Nonzero elements are found
            along the T dimension for each column.
        row_start_id (nl.ndarray): [1], Optional HBM tensor containing the starting column
            index in the E dimension. If specified, only n_rows columns starting from this
            index are processed. If None, all E columns are processed.
        n_rows (int): Number of columns (in E dimension) to process. Required when
            row_start_id is specified, ignored otherwise.
        chunk_size (int): Size of chunks for processing T dimension. If None, defaults to T.
            Must divide T evenly. Smaller chunk sizes reduce memory usage.
        index_dtype (nki.dtype): Data type for output indices tensor. Default is nl.int32.

    Returns:
        indices (nl.ndarray): [E, T] or [n_rows, T], Tensor containing nonzero indices.
            For each column e, the first N values are the T-indices of nonzero elements,
            followed by -1 padding values.
        nonzero_counts (nl.ndarray): [E] or [n_rows], Count of nonzero elements per column.

    Notes:
        - Performance is not fully optimized; expect ~2x regression vs Beta 1 implementation
        - Requires LNC2 configuration (2 NeuronCores)
        - E must be divisible by 2 (for LNC2 sharding)
        - T must be divisible by chunk_size
        - chunk_size must be divisible by 128 (partition size)
        - Uses GpSimd nonzero_with_count ISA which only operates on partitions [0, 16, 32, ..., 112]

    Pseudocode:
        for each expert e in [0, E):
            count = 0
            for t in [0, T):
                if input_tensor[t, e] != 0:
                    indices[e, count] = t
                    count += 1
            # Pad remaining with -1
            for i in [count, T):
                indices[e, i] = -1
            nonzero_counts[e] = count
    """
    T, E_full = input_tensor.shape

    # Handle row_start_id parameter for processing subset of columns
    if row_start_id is not None:
        row_start_id_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row_start_id_sbuf, src=row_start_id[0:1])
        E = n_rows
    else:
        row_start_id_sbuf = None
        E = E_full

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)

    if chunk_size is None:
        chunk_size = T

    E_per_shard = E // num_shards
    E_offset = E_per_shard * shard_id

    P_MAX = nl.tile_size.pmax  # 128

    # Allocate output tensors
    indices = nl.ndarray((E, T), dtype=index_dtype, buffer=nl.shared_hbm)

    # Initialize indices to -1 if processing in chunks
    if chunk_size < T:
        sbuf_init = nl.ndarray((P_MAX, E_per_shard * T // P_MAX), dtype=index_dtype, buffer=nl.sbuf)
        nisa.memset(dst=sbuf_init, value=-1)
        reshaped_dst = indices.reshape((P_MAX * 2, E_per_shard * T // P_MAX))
        nisa.dma_copy(dst=reshaped_dst[P_MAX * shard_id : P_MAX * (shard_id + 1), :], src=sbuf_init)

    nonzero_counts = nl.ndarray((E,), dtype=nl.int32, buffer=nl.shared_hbm)
    nonzero_counts_local = nl.ndarray((1, E_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=nonzero_counts_local, value=0)

    # Calculate iteration counts
    n_expert_rounds = div_ceil(E_per_shard, _NUM_GPSIMD_CORES)
    n_t_chunks = T // chunk_size
    n_tiles_per_chunk = chunk_size // P_MAX

    # Process experts in groups of 8 (one per GpSimd core)
    for expert_round_idx in nl.static_range(n_expert_rounds):
        n_experts_this_round = _NUM_GPSIMD_CORES
        if expert_round_idx == n_expert_rounds - 1:
            n_experts_this_round = E_per_shard - _NUM_GPSIMD_CORES * expert_round_idx

        # Track cumulative offsets for writing indices
        offsets = nl.ndarray((1, _NUM_GPSIMD_CORES), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=offsets, value=0)

        # Process T dimension in chunks
        for chunk_idx in nl.static_range(n_t_chunks):
            """
            Load input data and arrange for GpSimd processing.

            The nonzero_with_count ISA only operates on partitions [0, 16, 32, ..., 112].
            We load data for 8 experts and transpose so each expert's data is in the
            correct partition for parallel processing.
            """
            input_local_gpsimd_aligned = nl.ndarray((P_MAX, 1, chunk_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
            nisa.memset(dst=input_local_gpsimd_aligned, value=0)

            for tile_idx in nl.affine_range(n_tiles_per_chunk):
                t_start = chunk_idx * chunk_size + tile_idx * P_MAX
                e_base = expert_round_idx * _NUM_GPSIMD_CORES + E_offset

                input_local_te = nl.ndarray((P_MAX, _NUM_GPSIMD_CORES), dtype=input_tensor.dtype, buffer=nl.sbuf)
                if n_experts_this_round < _NUM_GPSIMD_CORES:
                    nisa.memset(dst=input_local_te[0:P_MAX, n_experts_this_round:_NUM_GPSIMD_CORES], value=0)

                if row_start_id_sbuf is not None:
                    # Use dynamic access pattern with scalar_offset for row_start_id
                    nisa.dma_copy(
                        dst=input_local_te[0:P_MAX, 0:n_experts_this_round],
                        src=input_tensor.ap(
                            pattern=[[E_full, P_MAX], [1, n_experts_this_round]],
                            offset=t_start * E_full + e_base,
                            scalar_offset=row_start_id_sbuf,
                            indirect_dim=1,
                        ),
                    )
                else:
                    e_start = e_base
                    nisa.dma_copy(
                        dst=input_local_te[0:P_MAX, 0:n_experts_this_round],
                        src=input_tensor[t_start : t_start + P_MAX, e_start : e_start + n_experts_this_round],
                    )

                # Scatter columns to partitions 0, 16, 32, ..., 112 for GpSimd processing
                input_local_aligned_te = nl.ndarray((P_MAX, P_MAX), dtype=input_tensor.dtype, buffer=nl.sbuf)
                for expert_idx in nl.affine_range(n_experts_this_round):
                    nisa.tensor_copy(
                        dst=input_local_aligned_te[
                            :, expert_idx * _PARTITIONS_PER_GPSIMD : expert_idx * _PARTITIONS_PER_GPSIMD + 1
                        ],
                        src=input_local_te[:, expert_idx : expert_idx + 1],
                    )

                # Transpose so expert data is in rows 0, 16, 32, ..., 112
                transposed_sbuf = nl.ndarray((P_MAX, P_MAX), dtype=input_tensor.dtype, buffer=nl.sbuf)
                nisa.dma_transpose(
                    dst=transposed_sbuf.ap(pattern=[[P_MAX, P_MAX], [1, 1], [1, 1], [1, P_MAX]]),
                    src=input_local_aligned_te.ap(pattern=[[P_MAX, P_MAX], [1, 1], [1, 1], [1, P_MAX]]),
                )

                nisa.tensor_copy(
                    dst=input_local_gpsimd_aligned[:, 0, tile_idx * P_MAX : (tile_idx + 1) * P_MAX],
                    src=transposed_sbuf,
                )

            # Run GpSimd nonzero_with_count ISA
            output_local = nl.ndarray((P_MAX, 1, chunk_size + 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.nonzero_with_count(
                dst=output_local,
                src=input_local_gpsimd_aligned,
                index_offset=chunk_idx * chunk_size,
                padding_val=-1,
            )

            # Write results for experts in partitions 0, 32, 64, 96 (even GPSIMD cores)
            for quadrant_idx in nl.affine_range(_NUM_QUADRANTS):
                expert_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT
                if expert_idx < n_experts_this_round:
                    offset_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=offset_tile, src=offsets[0:1, expert_idx : expert_idx + 1])

                    out_row = E_offset + expert_round_idx * _NUM_GPSIMD_CORES + expert_idx
                    src_data = nl.ndarray((1, chunk_size), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=src_data,
                        src=output_local[
                            quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1, 0, 0:chunk_size
                        ],
                    )
                    nisa.dma_copy(
                        dst=indices.ap(
                            pattern=[[T, 1], [1, chunk_size]],
                            offset=out_row * T,
                            scalar_offset=offset_tile,
                            indirect_dim=1,
                        ),
                        src=src_data,
                    )

                    count_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=count_tile,
                        src=output_local[
                            quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1,
                            0,
                            chunk_size : chunk_size + 1,
                        ],
                    )
                    nisa.tensor_tensor(
                        dst=offsets[0:1, expert_idx : expert_idx + 1],
                        data1=offsets[0:1, expert_idx : expert_idx + 1],
                        data2=count_tile,
                        op=nl.add,
                    )

            # Shuffle to move data from partitions 16, 48, 80, 112 to 0, 32, 64, 96
            quad_mask = [_PARTITIONS_PER_GPSIMD] + [255] * (_QUADRANT_SIZE - 1)
            nisa.nc_stream_shuffle(dst=output_local, src=output_local, shuffle_mask=quad_mask)

            # Write results for experts in partitions 16, 48, 80, 112 (odd GPSIMD cores)
            for quadrant_idx in nl.affine_range(_NUM_QUADRANTS):
                expert_idx = quadrant_idx * _GPSIMD_CORES_PER_QUADRANT + 1
                if expert_idx < n_experts_this_round:
                    offset_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=offset_tile, src=offsets[0:1, expert_idx : expert_idx + 1])

                    out_row = E_offset + expert_round_idx * _NUM_GPSIMD_CORES + expert_idx
                    src_data = nl.ndarray((1, chunk_size), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=src_data,
                        src=output_local[
                            quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1, 0, 0:chunk_size
                        ],
                    )
                    nisa.dma_copy(
                        dst=indices.ap(
                            pattern=[[T, 1], [1, chunk_size]],
                            offset=out_row * T,
                            scalar_offset=offset_tile,
                            indirect_dim=1,
                        ),
                        src=src_data,
                    )

                    count_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=count_tile,
                        src=output_local[
                            quadrant_idx * _QUADRANT_SIZE : quadrant_idx * _QUADRANT_SIZE + 1,
                            0,
                            chunk_size : chunk_size + 1,
                        ],
                    )
                    nisa.tensor_tensor(
                        dst=offsets[0:1, expert_idx : expert_idx + 1],
                        data1=offsets[0:1, expert_idx : expert_idx + 1],
                        data2=count_tile,
                        op=nl.add,
                    )

        # Copy final counts for this round of experts
        nisa.tensor_copy(
            dst=nonzero_counts_local[
                0:1, expert_round_idx * _NUM_GPSIMD_CORES : expert_round_idx * _NUM_GPSIMD_CORES + n_experts_this_round
            ],
            src=offsets[0:1, 0:n_experts_this_round],
        )

    # Write nonzero counts to HBM
    nonzero_counts_reshape = nonzero_counts.reshape((1, E))
    nisa.dma_copy(dst=nonzero_counts_reshape[0:1, E_offset : E_offset + E_per_shard], src=nonzero_counts_local)

    return indices, nonzero_counts
