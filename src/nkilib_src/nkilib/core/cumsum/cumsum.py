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

"""Cumulative sum kernel for NKI. Computes cumulative sum along the last dimension."""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, reduce
from ..utils.tiled_range import TiledRange

P_MAX = 128  # Partition dimension max (nl.tile_size.pmax returns non-int, compiler requires Python int for shapes)
F_TILE_SIZE = 2048


@nki.jit
def cumsum(x: nl.ndarray, axis: int = -1) -> nl.ndarray:
    """
    Compute cumulative sum along the last dimension.

    This kernel computes the cumulative sum of elements along the last dimension
    of the input tensor. Optimized for batch sizes up to 2048 and hidden dimension
    sizes up to 8192. Supports 3D inputs with sequence length up to 10.

    Dimensions:
        B: Batch size, up to 2048
        S: Sequence length for 3D inputs, up to 10
        H: Hidden dimension size, up to 8192
        outer_dim: Product of all dimensions except the last (B for 2D, B * S for 3D)

    Args:
        x (nl.ndarray): Input HBM tensor of shape [B, H] for 2D or [B, S, H] for 3D.
            dtype can be any NKI-supported type.
        axis (int): Axis along which to compute cumsum. Must be -1 or the last
            dimension index. Default: -1.

    Returns:
        nl.ndarray: Output HBM tensor with same shape and dtype as input, containing
            cumulative sums along the last dimension.

    Notes:
        - Only supports cumsum along the last dimension
        - Uses float32 accumulation internally for numerical stability
        - For very long hidden dimensions (>5K), expect ~1e-2 absolute error due to fp32 accumulation

    Pseudocode:
        # Reshape to 2D: (outer_dim, hidden)
        x_2d = x.reshape(outer_dim, hidden)
        y_2d = zeros_like(x_2d)

        for p_tile_idx in range(num_partition_tiles):
            init = 0  # carry from previous free tile
            for f_tile_idx in range(num_free_tiles):
                tile = load(x_2d[p_tile, f_tile])
                # cumsum: result[i] = result[i-1] + tile[i]
                result = tensor_tensor_scan(ones, tile, init, multiply, add)
                store(y_2d[p_tile, f_tile], result)
                init = result[:, -1]  # carry forward last column

        return y_2d.reshape(x.shape)
    """
    rank = len(x.shape)

    # Normalize axis to positive index
    if axis < 0:
        axis = rank + axis

    # Only support last dimension
    kernel_assert(axis == rank - 1, f"Only support cumsum over last dim, got {axis=}, {rank=}")

    # Reshape to 2D: (outer_dim, last_dim)
    x_shape = x.shape
    outer_dim = reduce(op='mul', input=list(x_shape[:-1]), initial_value=1) if rank > 1 else 1
    last_dim = x_shape[-1]
    shape_2d = (outer_dim, last_dim)

    x_2d = x.reshape(shape_2d)

    # Allocate output on HBM
    y = nl.ndarray(x_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    y_2d = y.reshape(shape_2d)

    num_f_tiles = div_ceil(last_dim, F_TILE_SIZE)

    # Process partition tiles
    for p_tile in TiledRange(outer_dim, P_MAX):
        # Initialize carry for this partition tile
        # Note: float32 used for numerical stability; tensor_tensor_scan auto-casts to float32 internally
        init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=init_sb, value=0.0)

        # Allocate ones tensor for scan (data0 * prev + data1)
        ones_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_sb, value=1.0)

        # Sequential loop over free dimension tiles (must be sequential for cumsum)
        for f_tile_idx in nl.sequential_range(num_f_tiles):
            f_start = f_tile_idx * F_TILE_SIZE
            f_end = min(f_start + F_TILE_SIZE, last_dim)
            f_size = f_end - f_start

            # Load input tile to SBUF
            data_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=data_sb[0 : p_tile.size, 0:f_size],
                src=x_2d[p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
            )

            # Compute cumsum using tensor_tensor_scan
            # result[i] = ones[i] * result[i-1] + data[i] = result[i-1] + data[i]
            result_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=result_sb[0 : p_tile.size, 0:f_size],
                data0=ones_sb[0 : p_tile.size, 0:f_size],
                data1=data_sb[0 : p_tile.size, 0:f_size],
                initial=init_sb[0 : p_tile.size, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

            # Store result to HBM
            nisa.dma_copy(
                dst=y_2d[p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
                src=result_sb[0 : p_tile.size, 0:f_size],
            )

            # Update init for next tile: carry forward last column
            if f_tile_idx + 1 < num_f_tiles:
                last_col_idx = f_size - 1
                nisa.tensor_copy(
                    dst=init_sb[0 : p_tile.size, 0:1], src=result_sb[0 : p_tile.size, last_col_idx : last_col_idx + 1]
                )

    return y
