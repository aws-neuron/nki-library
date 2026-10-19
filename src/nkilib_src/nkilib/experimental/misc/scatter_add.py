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

"""
Scatter-add operation using gather-accumulate-scatter pattern.

This kernel scatters values from src and adds them to input based on indices in index.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info

# Multiplier applied to `nl.tile_size.psum_fmax` to size the D tile. Using 4x
# F_MAX amortizes the per-DMA setup cost of indirect loads/stores over a larger
# contiguous feature block.
_D_TILE_FACTOR = 4


@nki.jit
def scatter_add(
    input: nl.NkiTensor, dim: int, index: nl.NkiTensor, src: nl.NkiTensor, unique_indices: bool = True
) -> nl.NkiTensor:
    """
    Scatter-add from src into input based on indices using gather-accumulate-scatter pattern.

    Equivalent to PyTorch's ``input.scatter_add(dim=0, index=index, src=src)``.
    Performs: input[index[i], j] += src[i, j] for all i, j.
    TODO: Specify intended usage range (e.g., typical K source-row count, D feature
    size, and N destination-row size for which this kernel is optimized).

    Dimensions:
        N: Number of rows in input tensor
        D: Feature dimension size
        K: Number of source rows / indices

    Args:
        input (nl.NkiTensor): [N, D], Destination tensor to accumulate into (modified in-place)
        dim (int): Dimension along which to scatter (must be 0)
        index (nl.NkiTensor): [K], 1D tensor of row indices into input
        src (nl.NkiTensor): [K, D], Source values to scatter-add
        unique_indices (bool): If True (default), assume destination indices are unique within
            each 128-row tile and use the plain tile-wide gather/scatter path. If False,
            duplicate rows within a tile are pre-combined on the Tensor Engine so repeated
            indices accumulate correctly, which embedding-gradient scatters need. Costs one
            extra 128x128 matmul per PSUM-wide column chunk.

    Returns:
        input (nl.NkiTensor): [N, D], The input tensor with scattered values added

    Notes:
        - Input and src tensors must be 2D
        - Index tensor must be 1D
        - dim must be 0
        - With unique_indices=True (default) indices within a 128-row tile must be unique;
          duplicates are silently dropped. Pass unique_indices=False when they may repeat.

    Pseudocode:
        for k_tile in tiles(K):
            idx_tile = load(index[k_tile])
            for d_tile in tiles(D):
                src_tile = load(src[k_tile, d_tile])
                existing_tile = indirect_load(input, row_indices=idx_tile, col_range=d_tile)
                result_tile = existing_tile + src_tile
                indirect_store(input, row_indices=idx_tile, col_range=d_tile, result_tile)
    """
    k_size, d_size = src.shape

    _, num_shards, shard_id = get_program_sharding_info()

    _validate_scatter_add_inputs(input, dim, index, src, num_shards)

    k_tile_size = nl.tile_size.pmax
    d_tile_size = nl.tile_size.psum_fmax * _D_TILE_FACTOR
    num_k_tiles = div_ceil(k_size, k_tile_size)

    # LNC Sharding: shard over dimension 1 to avoid write conflicts
    d_per_shard = d_size // num_shards
    d_offset = shard_id * d_per_shard
    num_d_tiles_per_shard = div_ceil(d_per_shard, d_tile_size)

    for k_tile_idx in nl.sequential_range(num_k_tiles):
        k_valid = min(k_tile_size, k_size - k_tile_idx * k_tile_size)

        # Load index tile
        idx_tile = nl.ndarray((k_valid, 1), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_tile,
            src=index.ap(pattern=[[1, k_valid], [1, 1]], offset=k_tile_idx * k_tile_size),
        )

        if not unique_indices:
            dup_matrix = _build_duplicate_matrix(index, idx_tile, k_tile_idx * k_tile_size, k_valid, src.dtype)

        for local_d_tile_idx in nl.affine_range(num_d_tiles_per_shard):
            d_tile_offset = d_offset + local_d_tile_idx * d_tile_size
            d_valid = min(d_tile_size, d_per_shard - local_d_tile_idx * d_tile_size)

            # Load source tile
            src_tile = nl.ndarray((k_valid, d_valid), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=src_tile,
                src=src.ap(
                    pattern=[[d_size, k_valid], [1, d_valid]],
                    offset=d_tile_offset + k_tile_idx * k_tile_size * d_size,
                ),
            )

            # Gather existing values from input at indirect indices
            existing_tile = nl.ndarray((k_valid, d_valid), dtype=input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=existing_tile,
                src=input.ap(
                    pattern=[[d_size, k_valid], [1, d_valid]],
                    offset=d_tile_offset,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
            )

            # Add in SBUF
            result_tile = nl.ndarray((k_valid, d_valid), dtype=input.dtype, buffer=nl.sbuf)
            if unique_indices:
                nisa.tensor_tensor(dst=result_tile, data1=existing_tile, data2=src_tile, op=nl.add)
            else:
                # Every row of a duplicate group gathered the same `existing` value, and after
                # this matmul holds the same group total, so the racing indirect writes below
                # all store the same correct result. PSUM caps the matmul free dimension, so
                # walk the column tile in psum_fmax chunks.
                for chunk_start in range(0, d_valid, nl.tile_size.psum_fmax):
                    chunk_valid = min(nl.tile_size.psum_fmax, d_valid - chunk_start)
                    chunk = nl.ds(chunk_start, chunk_valid)
                    group_sum = nl.ndarray((k_valid, chunk_valid), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=group_sum,
                        stationary=dup_matrix,
                        moving=src_tile[:, chunk],
                        is_stationary_onezero=True,
                    )
                    nisa.tensor_tensor(
                        dst=result_tile[:, chunk],
                        data1=existing_tile[:, chunk],
                        data2=group_sum,
                        op=nl.add,
                    )

            # Scatter write back
            nisa.dma_copy(
                dst=input.ap(
                    pattern=[[d_size, k_valid], [1, d_valid]],
                    offset=d_tile_offset,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=result_tile,
            )

    return input


def _build_duplicate_matrix(index, idx_tile, k_offset, k_valid, dtype):
    """Build the [k_valid, k_valid] index-equality matrix for one k tile.

    ``E[i, j]`` is 1.0 when ``index[i] == index[j]`` and 0.0 otherwise, over the ``k_valid``
    indices starting at ``k_offset``. As the stationary operand of an ``nc_matmul`` against the
    src tile it produces ``sum_j E[i, j] * src[j, :]``, the total of every src row sharing row
    i's destination. ``E`` is symmetric, so no transpose is needed.

    Args:
        index (nl.NkiTensor): [K], the full 1D index tensor in HBM.
        idx_tile (nl.NkiTensor): [k_valid, 1], this tile's indices already in SBUF.
        k_offset (int): Start of this tile within ``index``.
        k_valid (int): Number of indices in this tile (<= 128).
        dtype: Data type of ``E``; must match the src tile's dtype for ``nc_matmul``.

    Returns:
        nl.NkiTensor: [k_valid, k_valid] equality matrix in SBUF.
    """
    # tensor_scalar requires a float32 per-partition operand, so cast this tile's indices.
    idx_col = nl.ndarray((k_valid, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_col, src=idx_tile)

    # Re-read the indices with a zero partition stride so every partition holds the full index
    # row, which forms the free-dimension side of the comparison.
    idx_row = nl.ndarray((k_valid, k_valid), dtype=index.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_row, src=index.ap(pattern=[[0, k_valid], [1, k_valid]], offset=k_offset))

    dup_matrix = nl.ndarray((k_valid, k_valid), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dup_matrix, data=idx_row, op0=nl.equal, operand0=idx_col)
    return dup_matrix


def _validate_scatter_add_inputs(input, dim, index, src, num_shards):
    """Validate inputs for scatter_add operation."""
    kernel_assert(len(input.shape) == 2, f"scatter_add only supports 2D tensors, got input shape {input.shape}")
    kernel_assert(len(index.shape) == 1, f"scatter_add expects 1D index tensor, got index shape {index.shape}")
    kernel_assert(len(src.shape) == 2, f"scatter_add only supports 2D tensors, got src shape {src.shape}")
    kernel_assert(dim == 0, f"scatter_add currently only supports dim=0, got dim={dim}")

    _, d_size = input.shape
    k_size, d_src = src.shape

    kernel_assert(
        index.shape[0] == k_size,
        f"Index and src must have same dimension 0, got index {index.shape[0]}, src {k_size}",
    )
    kernel_assert(d_size == d_src, f"Dimension mismatch: input has {d_size}, src has {d_src}")
    kernel_assert(
        d_size % num_shards == 0,
        f"dimension 1 ({d_size}) must be divisible by num_shards ({num_shards}) for LNC sharding",
    )
