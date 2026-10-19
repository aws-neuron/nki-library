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
Gather operation using indirect load.

This kernel gathers rows from input based on indices in index.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info

# Multiplier applied to `nl.tile_size.psum_fmax` to size the D tile. Using 4x
# F_MAX amortizes the per-DMA setup cost of indirect loads over a larger
# contiguous feature block.
_D_TILE_FACTOR = 4


@nki.jit
def gather(input: nl.NkiTensor, dim: int, index: nl.NkiTensor) -> nl.NkiTensor:
    """
    Gather rows from input based on indices using indirect DMA load.

    Equivalent to PyTorch's ``input[index]`` for dim=0 with 2D input and 1D index.
    TODO: Specify intended usage range (e.g., typical K index count, D feature size,
    and N source-row size for which this kernel is optimized).

    Dimensions:
        N: Number of rows in input tensor
        D: Feature dimension size
        K: Number of indices (output rows)

    Args:
        input (nl.NkiTensor): [N, D], Source tensor to gather from
        dim (int): Dimension along which to gather (must be 0)
        index (nl.NkiTensor): [K], 1D tensor of row indices into input

    Returns:
        output (nl.NkiTensor): [K, D], Gathered result where output[i, :] = input[index[i], :]

    Notes:
        - Input tensor must be 2D
        - Index tensor must be 1D
        - dim must be 0

    Pseudocode:
        output = zeros(K, D)
        for k_tile in tiles(K):
            idx_tile = load(index[k_tile])
            for d_tile in tiles(D):
                data_tile = indirect_load(input, row_indices=idx_tile, col_range=d_tile)
                store(output[k_tile, d_tile], data_tile)
    """
    k_size = index.shape[0]
    d_size = input.shape[1]
    dtype = input.dtype

    _, num_shards, shard_id = get_program_sharding_info()

    _validate_gather_inputs(input, dim, index, num_shards)

    k_tile_size = nl.tile_size.pmax
    d_tile_size = nl.tile_size.psum_fmax * _D_TILE_FACTOR
    num_d_tiles = div_ceil(d_size, d_tile_size)

    """
    LNC Sharding Strategy: Shard over output dimension 0 (not dimension 1).
    Gathering full rows ensures DMA transfers are large enough to saturate bandwidth.
    Each core processes complete rows for its assigned row range, maximizing DMA efficiency.
    """
    k_per_shard = k_size // num_shards
    k_offset = shard_id * k_per_shard
    num_k_tiles_per_shard = div_ceil(k_per_shard, k_tile_size)

    output = nl.ndarray((k_size, d_size), dtype=dtype, buffer=nl.shared_hbm)

    for local_tile_idx in nl.affine_range(num_k_tiles_per_shard):
        tile_k_offset = k_offset + local_tile_idx * k_tile_size

        # Calculate valid elements for this k tile
        k_valid = min(k_tile_size, k_per_shard - local_tile_idx * k_tile_size)

        # Load index tile
        idx_tile = nl.ndarray((k_valid, 1), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_tile, src=index.ap(pattern=[[1, k_valid], [1, 1]], offset=tile_k_offset))

        for d_tile_idx in nl.affine_range(num_d_tiles):
            # Calculate valid elements for this d tile
            d_valid = min(d_tile_size, d_size - d_tile_idx * d_tile_size)

            # Load data tile using Vector Dynamic Access
            data_tile = nl.ndarray((k_valid, d_valid), dtype=dtype, buffer=nl.sbuf)

            nisa.dma_copy(
                dst=data_tile,
                src=input.ap(
                    pattern=[[d_size, k_valid], [1, d_valid]],
                    offset=d_tile_idx * d_tile_size,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
            )

            # Store output
            nisa.dma_copy(
                dst=output[nl.ds(tile_k_offset, k_valid), nl.ds(d_tile_idx * d_tile_size, d_valid)],
                src=data_tile,
            )

    return output


def _validate_gather_inputs(input, dim, index, num_shards):
    """Validate inputs for gather operation."""
    kernel_assert(len(input.shape) == 2, f"gather only supports 2D tensors, got input shape {input.shape}")
    kernel_assert(len(index.shape) == 1, f"gather expects 1D index tensor, got index shape {index.shape}")
    kernel_assert(dim == 0, f"gather currently only supports dim=0, got dim={dim}")

    k_size = index.shape[0]
    kernel_assert(
        k_size % num_shards == 0,
        f"index size ({k_size}) must be divisible by num_shards ({num_shards}) for LNC sharding",
    )
