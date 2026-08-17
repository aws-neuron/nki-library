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
Streaming: .stream() for rotating-buffer input/output, with dtype conversion
on load and 2D block streams.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

# ============================================================================
# Kernels
# ============================================================================


@nki.jit
def stream_input_output_scale(src):
    """Input + output streaming: per-row streams scale each tile and
    write back via the output stream. One tile per rotating slot.

    Demonstrates ``src_tiles[i].stream()`` -- a single int on the
    parent advances the cursor to dim 1, and ``.stream()`` defaults
    to walking the cursor's dim (the surviving F-tile axis)."""
    TILE_P, TILE_F = 128, 128

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))

    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for i in range(src_tiles.shape[0]):
        # Stream i'th tile row with double buffering -- each buffer is one tile.
        in_stream = src_tiles[i].stream(buffer_count=2)
        out_stream = dst_tiles[i].stream(buffer_count=2)
        for k in nl.affine_range(src_tiles.shape[1]):
            in_tile = in_stream.load(k)  # tile: [128, 128]
            out_tile = out_stream[k]
            nisa.tensor_scalar(out_tile.data, in_tile.data, nl.multiply, 2.0)
            out_stream.store(k)

    return dst


@nki.jit
def stream_with_dtype_conversion(src):
    """Per-column tile sum with dtype conversion on the stream load."""
    TILE_P, TILE_F = 128, 128

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))

    dst = nl.ndarray((TILE_P, src_tiles.shape[1] * TILE_F), dtype=src.dtype, buffer=nl.shared_hbm)
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for j in range(src_tiles.shape[1]):
        # Stream j'th tile column with double buffering -- each buffer is one tile.
        col_stream = src_tiles[:, j].stream(buffer_count=2)

        acc = nl.ndarray((TILE_P, TILE_F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc, 0.0)
        for k in nl.affine_range(src_tiles.shape[0]):
            tile = col_stream.load(k, dtype=nl.bfloat16)  # tile: [128, 128]
            nisa.tensor_tensor(acc, acc, tile.data, op=nl.add)

        # store() handles the FP32 -> source dtype conversion at DMA time.
        dst_tiles[0, j].store(acc)

    return dst


@nki.jit
def stream_dim_walks_columns(src):
    """tiles.stream(dim=1): each rotating slot holds a whole tile column
    (K * TILE_P, TILE_F) instead of a single tile. F_TILES coalesced
    DMAs total -- one per column -- vs F_TILES * K_TILES per-tile DMAs.
    Useful when the column is the natural compute unit and the per-slot
    SBUF footprint is acceptable."""
    TILE_P, TILE_F = 128, 128

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))

    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    # Stream tile columns with double buffering -- each buffer is one
    # whole column of K tiles, coalesced into a single DMA.
    col_stream = src_tiles.stream(dim=1, buffer_count=2)

    for j in range(src_tiles.shape[1]):
        col = col_stream.load(j)  # col slot: [K * 128, 128]
        for k in range(src_tiles.shape[0]):
            tile = col[k]  # tile: [128, 128]
            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
        dst_tiles[:, j].store(col.data)

    return dst


@nki.jit
def stream_2d_blocks(src):
    """Block streaming: outer Python loop over block-rows, inner stream
    walks block-cols within each row. Each rotating buffer holds ONE
    block (block_size * tile_size elements). Per-block work walks the
    interior tile grid; the block is stored back via a single coalesced
    block-level DMA."""
    TILE_P, TILE_F = 128, 256
    BLOCK_P, BLOCK_F = 2, 2

    src_blocks = nt.blocks(src, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))

    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    dst_blocks = nt.blocks(dst, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))

    for bi in range(src_blocks.shape[0]):
        # Stream bi'th block-row with double buffering -- each buffer is one block.
        row_stream = src_blocks[bi].stream(buffer_count=2)
        for bj in range(src_blocks.shape[1]):
            block = row_stream.load(bj)  # block: [256, 512]
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    tile = block[ti, tj]  # tile: [128, 256]
                    nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
            dst_blocks[bi, bj].store(block.data)

    return dst


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


# ============================================================================
# Tests
# ============================================================================


def test_stream_input_output_scale():
    torch.manual_seed(42)
    src = torch.randn(512, 1024, dtype=torch.bfloat16)
    result = stream_input_output_scale(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("stream_input_output_scale: PASSED")


def test_stream_with_dtype_conversion():
    torch.manual_seed(42)
    src = torch.rand(512, 1024, dtype=torch.bfloat16)
    result = stream_with_dtype_conversion(to_device(src))
    expected = src.view(4, 128, 8, 128).sum(dim=0).reshape(128, 1024)
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-3, atol=1e-3)
    print("stream_with_dtype_conversion: PASSED")


def test_stream_dim_walks_columns():
    torch.manual_seed(42)
    src = torch.randn(512, 1024, dtype=torch.bfloat16)
    result = stream_dim_walks_columns(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("stream_dim_walks_columns: PASSED")


def test_stream_2d_blocks():
    torch.manual_seed(42)
    src = torch.randn(512, 1024, dtype=torch.bfloat16)
    result = stream_2d_blocks(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("stream_2d_blocks: PASSED")


def main():
    test_stream_input_output_scale()
    test_stream_with_dtype_conversion()
    test_stream_dim_walks_columns()
    test_stream_2d_blocks()


if __name__ == "__main__":
    main()
