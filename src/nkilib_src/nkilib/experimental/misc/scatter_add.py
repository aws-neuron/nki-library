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
def scatter_add(input: nl.ndarray, dim: int, index: nl.ndarray, src: nl.ndarray, unique_indices: bool = True) -> nl.ndarray:
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
        input (nl.ndarray): [N, D], Destination tensor to accumulate into (modified in-place)
        dim (int): Dimension along which to scatter (must be 0)
        index (nl.ndarray): [K], 1D tensor of row indices into input
        src (nl.ndarray): [K, D], Source values to scatter-add
        unique_indices (bool): If True (default), assume destination indices are unique
            within each 128-row tile and use the fast tile-wide gather/scatter path.
            If False, process one index per tile so repeated indices accumulate
            correctly (slower; needed for embedding-gradient-style scatters).

    Returns:
        input (nl.ndarray): [N, D], The input tensor with scattered values added

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

    # When indices may repeat within a 128-row tile, a tile-wide gather/scatter is
    # last-write-wins and silently drops duplicate contributions. k_tile_size=1 makes
    # each index its own sequential tile, so nl.sequential_range accumulates dups correctly.
    k_tile_size = nl.tile_size.pmax if unique_indices else 1
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
            nisa.tensor_tensor(dst=result_tile, data1=existing_tile, data2=src_tile, op=nl.add)

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
