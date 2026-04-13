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

"""Dynamic-shape elementwise add kernel. Computes output = input_a + input_b with runtime-variable M-dimension."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

P_MAX = 128  # Partition dimension tile size (rows per M-tile)
H_TILE_SIZE = 512  # Hidden dimension tile size (columns per H-tile)


@nki.jit
def dynamic_elementwise_add(
    input_a: nl.ndarray,
    input_b: nl.ndarray,
    num_m_tiles: nl.ndarray,
) -> nl.ndarray:
    """
    Elementwise addition with dynamic partition dimension tiling.

    Computes output = input_a + input_b for 2D bf16 tensors where the number of
    M-dimension tiles to process is determined at runtime via num_m_tiles. Optimized
    for M dimensions up to 2048 and H dimensions up to 8192.

    Dimensions:
        M: Row dimension, tiled at P_MAX (128). Dynamic at runtime via num_m_tiles.
        H: Hidden/column dimension, tiled at H_TILE_SIZE (512). Static.
        num_m_tiles: Runtime trip count for the M-tile loop (int32 scalar).

    Args:
        input_a (nl.ndarray): [M, H], First input tensor, bf16, on HBM.
        input_b (nl.ndarray): [M, H], Second input tensor, bf16, on HBM. Must match input_a shape.
        num_m_tiles (nl.ndarray): [1, 1], int32 scalar tensor on HBM. Value = number of M-tiles
            to process (0 <= num_m_tiles <= M // P_MAX).

    Returns:
        result (nl.ndarray): [M, H], bf16 output tensor on HBM. Elements in the first
            (num_m_tiles * P_MAX) rows contain input_a + input_b; remaining rows are unmodified.

    Notes:
        - M must be divisible by P_MAX (128)
        - H must be divisible by H_TILE_SIZE (512)
        - input_a and input_b must have identical shapes

    Pseudocode:
        result = allocate([M, H], bf16, HBM)
        trip_count = load_register(num_m_tiles)
        m_offset = 0

        for _ in dynamic_range(trip_count):
            for h_tile_idx in affine_range(H // H_TILE_SIZE):
                h_start = h_tile_idx * H_TILE_SIZE
                a_tile = dma_load(input_a[m_offset:m_offset+P_MAX, h_start:h_start+H_TILE_SIZE])
                b_tile = dma_load(input_b[m_offset:m_offset+P_MAX, h_start:h_start+H_TILE_SIZE])
                out_tile = a_tile + b_tile
                dma_store(result[m_offset:m_offset+P_MAX, h_start:h_start+H_TILE_SIZE], out_tile)
            m_offset += P_MAX

        return result
    """
    m_static, hidden = input_a.shape

    # Input validation
    kernel_assert(
        input_a.shape == input_b.shape,
        f"input_a and input_b must have the same shape, got {input_a.shape} and {input_b.shape}",
    )
    kernel_assert(
        hidden % H_TILE_SIZE == 0,
        f"Hidden dimension H={hidden} must be divisible by H_TILE_SIZE={H_TILE_SIZE}",
    )
    kernel_assert(
        m_static % P_MAX == 0,
        f"M dimension M={m_static} must be divisible by P_MAX={P_MAX}",
    )

    H_TILE_COUNT = hidden // H_TILE_SIZE

    # Allocate output on HBM
    result = nl.ndarray(shape=(m_static, hidden), dtype=input_a.dtype, buffer=nl.shared_hbm)

    # Load num_m_tiles scalar into a hardware register for dynamic_range
    num_m_tiles_sbuf = nl.ndarray(shape=(1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    # M-tile offset tracker in SBUF (incremented by P_MAX each iteration)
    m_tile_start = nl.ndarray(shape=(1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=m_tile_start, value=0)

    for _ in nl.dynamic_range(0, num_m_tiles_reg, 1):
        for h_tile_idx in nl.affine_range(H_TILE_COUNT):
            h_start = h_tile_idx * H_TILE_SIZE

            # DMA load input_a tile from HBM to SBUF
            a_tile = nl.ndarray(shape=(P_MAX, H_TILE_SIZE), dtype=input_a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=a_tile[0:P_MAX, 0:H_TILE_SIZE],
                src=input_a.ap(
                    pattern=[[hidden, P_MAX], [1, H_TILE_SIZE]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # DMA load input_b tile from HBM to SBUF
            b_tile = nl.ndarray(shape=(P_MAX, H_TILE_SIZE), dtype=input_b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=b_tile[0:P_MAX, 0:H_TILE_SIZE],
                src=input_b.ap(
                    pattern=[[hidden, P_MAX], [1, H_TILE_SIZE]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # Elementwise add in SBUF
            out_tile = nl.ndarray(shape=(P_MAX, H_TILE_SIZE), dtype=input_a.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=out_tile[0:P_MAX, 0:H_TILE_SIZE],
                data1=a_tile[0:P_MAX, 0:H_TILE_SIZE],
                data2=b_tile[0:P_MAX, 0:H_TILE_SIZE],
                op=nl.add,
            )

            # DMA store result tile from SBUF to HBM
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[hidden, P_MAX], [1, H_TILE_SIZE]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=out_tile[0:P_MAX, 0:H_TILE_SIZE],
            )

        # Advance M-tile offset by P_MAX rows
        nisa.tensor_scalar(dst=m_tile_start, data=m_tile_start, op0=nl.add, operand0=P_MAX)

    # HBM fence: ensure all DMA writes complete before kernel returns
    nisa.dma_copy(dst=result, src=result)
    return result
