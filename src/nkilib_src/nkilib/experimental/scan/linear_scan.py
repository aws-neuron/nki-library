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

"""Linear scan kernel for NKI. Computes first-order linear recurrence along the last dimension."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, reduce
from ...core.utils.tiled_range import TiledRange

P_MAX = 128  # Partition dimension max (nl.tile_size.pmax returns non-int, compiler requires Python int for shapes)
F_TILE_SIZE = 2048


@nki.jit
def linear_scan(
    decay: nl.ndarray,
    data: nl.ndarray,
    initial: nl.ndarray = None,
) -> tuple:
    """
    Compute first-order linear recurrence along the last dimension.

    This kernel computes result[t] = decay[t] * result[t-1] + data[t] along the
    last dimension of the input tensors using nisa.tensor_tensor_scan. Supports
    arbitrary batch dimensions which are collapsed internally.

    Dimensions:
        P: Partition dimension (second-to-last), tiled at P_MAX=128
        L: Free dimension (last), tiled at F_TILE_SIZE=2048
        outer_dim: Product of all dimensions except the last two

    Args:
        decay (nl.ndarray): Input HBM tensor of shape (..., P, L) containing
            multiplicative decay coefficients. dtype can be any NKI-supported type.
        data (nl.ndarray): Input HBM tensor of shape (..., P, L) containing
            additive input values. Must have same shape as decay.
        initial (nl.ndarray, optional): Initial state tensor of shape (..., P, 1).
            If None, initial state is zero. Default: None.

    Returns:
        tuple: (result, final_state)
            - result: HBM tensor with same shape as inputs, containing the scan output.
            - final_state: HBM tensor of shape (..., P, 1) containing the last state
              for each sequence, in float32.

    Notes:
        - Only supports scan along the last dimension
        - Uses float32 accumulation internally for numerical stability
        - decay and data must have identical shapes and rank >= 2
        - For long sequences (>2048), the scan is tiled with carry propagation

    Pseudocode:
        # Reshape to 3D: (outer_dim, P, L)
        decay_3d = decay.reshape(outer_dim, P, L)
        data_3d = data.reshape(outer_dim, P, L)
        result_3d = zeros_like(decay_3d)
        final_state_3d = zeros(outer_dim, P, 1)

        for i_outer in range(outer_dim):
            for p_tile in TiledRange(P, P_MAX):
                carry = initial[i_outer, p_tile, :] if initial else 0
                for f_tile_idx in range(num_f_tiles):
                    d = load(decay_3d[i_outer, p_tile, f_tile])
                    x = load(data_3d[i_outer, p_tile, f_tile])
                    result = tensor_tensor_scan(d, x, carry, multiply, add)
                    store(result_3d[i_outer, p_tile, f_tile], result)
                    carry = result[:, -1]
                final_state_3d[i_outer, p_tile, :] = carry

        return result.reshape(decay.shape), final_state.reshape(..., P, 1)
    """
    rank = len(decay.shape)
    kernel_assert(rank >= 2, f"Input tensors must have rank >= 2, got rank={rank}")
    kernel_assert(
        decay.shape == data.shape,
        f"decay and data must have the same shape, got decay={decay.shape}, data={data.shape}",
    )

    if initial is not None:
        expected_initial_shape = tuple(list(decay.shape[:-1]) + [1])
        kernel_assert(
            initial.shape == expected_initial_shape,
            f"initial must have shape (..., P, 1), expected {expected_initial_shape}, got {initial.shape}",
        )

    # Extract last two dims: P and L
    decay_shape = decay.shape
    P = decay_shape[-2]
    L = decay_shape[-1]

    # Collapse all dims except last two into outer_dim
    outer_dim = reduce(op='mul', input=list(decay_shape[:-2]), initial_value=1) if rank > 2 else 1
    shape_3d = (outer_dim, P, L)
    final_state_shape = tuple(list(decay_shape[:-1]) + [1])

    decay_3d = decay.reshape(shape_3d)
    data_3d = data.reshape(shape_3d)

    if initial is not None:
        initial_3d = initial.reshape((outer_dim, P, 1))

    # Allocate outputs on HBM
    result = nl.ndarray(decay_shape, dtype=decay.dtype, buffer=nl.hbm)
    result_3d = result.reshape(shape_3d)

    final_state = nl.ndarray(final_state_shape, dtype=nl.float32, buffer=nl.hbm)
    final_state_3d = final_state.reshape((outer_dim, P, 1))

    num_f_tiles = div_ceil(L, F_TILE_SIZE)

    for i_outer in nl.affine_range(outer_dim):
        for p_tile in TiledRange(P, P_MAX):
            # Initialize carry for this partition tile
            carry_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            if initial is not None:
                nisa.dma_copy(
                    dst=carry_sb[0 : p_tile.size, 0:1],
                    src=initial_3d[i_outer, p_tile.start_offset : p_tile.start_offset + p_tile.size, 0:1],
                )
            else:
                nisa.memset(dst=carry_sb, value=0.0)

            # Sequential loop over free dimension tiles (must be sequential for recurrence)
            for f_tile_idx in nl.sequential_range(num_f_tiles):
                f_start = f_tile_idx * F_TILE_SIZE
                f_end = min(f_start + F_TILE_SIZE, L)
                f_size = f_end - f_start

                # Load decay tile to SBUF
                decay_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=decay.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=decay_sb[0 : p_tile.size, 0:f_size],
                    src=decay_3d[i_outer, p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
                )

                # Load data tile to SBUF
                data_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=data.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=data_sb[0 : p_tile.size, 0:f_size],
                    src=data_3d[i_outer, p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
                )

                # Core scan: result[t] = decay[t] * result[t-1] + data[t]
                result_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=decay.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(
                    dst=result_sb[0 : p_tile.size, 0:f_size],
                    data0=decay_sb[0 : p_tile.size, 0:f_size],
                    data1=data_sb[0 : p_tile.size, 0:f_size],
                    initial=carry_sb[0 : p_tile.size, 0:1],
                    op0=nl.multiply,
                    op1=nl.add,
                )

                # Store result to HBM
                nisa.dma_copy(
                    dst=result_3d[i_outer, p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
                    src=result_sb[0 : p_tile.size, 0:f_size],
                )

                # Update carry for next tile: carry forward last column
                if f_tile_idx + 1 < num_f_tiles:
                    last_col_idx = f_size - 1
                    nisa.tensor_copy(
                        dst=carry_sb[0 : p_tile.size, 0:1],
                        src=result_sb[0 : p_tile.size, last_col_idx : last_col_idx + 1],
                    )

            # Store final state: last column of the last tile
            last_f_start = (num_f_tiles - 1) * F_TILE_SIZE
            last_f_size = L - last_f_start
            last_col_idx = last_f_size - 1
            final_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=final_sb[0 : p_tile.size, 0:1],
                src=result_sb[0 : p_tile.size, last_col_idx : last_col_idx + 1],
            )
            nisa.dma_copy(
                dst=final_state_3d[i_outer, p_tile.start_offset : p_tile.start_offset + p_tile.size, 0:1],
                src=final_sb[0 : p_tile.size, 0:1],
            )

    return result, final_state
