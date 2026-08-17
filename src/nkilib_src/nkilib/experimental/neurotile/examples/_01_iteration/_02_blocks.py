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
Block iteration: nt.blocks() for coalesced load/store of a block of tiles
plus on-chip per-tile iteration.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

TILE_P, TILE_F = 128, 128
BLOCK_P, BLOCK_F = 2, 2
BLOCK_ROWS, BLOCK_COLS = 4, 2


# ============================================================================
# Basic block iteration
# ============================================================================


@nki.jit
def block_level_scale(src):
    """Single block: one coalesced DMA loads BLOCK_P x BLOCK_F tiles
    into SBUF, the inner loop walks the on-chip tile grid, then a
    single coalesced DMA stores the block back."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))
    dst_blocks = nt.blocks(dst, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))

    for bi in range(src_blocks.shape[0]):
        for bj in range(src_blocks.shape[1]):
            block = src_blocks[bi, bj].load()
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    nisa.tensor_scalar(
                        block[ti, tj].data,
                        block[ti, tj].data,
                        nl.multiply,
                        2.0,
                    )
            dst_blocks[bi, bj].store(block.data)

    return dst


@nki.jit
def load_block_row(src):
    """Coalesced block-row DMA: one DMA loads every block in row bi,
    then iterate the loaded view as block-then-tile so the block grid
    and tile grid stay distinct."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))
    dst_blocks = nt.blocks(dst, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))

    for bi in range(src_blocks.shape[0]):
        block_row = src_blocks[bi].load()
        for bj in range(block_row.shape[0]):
            block = block_row[bj]
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    nisa.tensor_scalar(
                        block[ti, tj].data,
                        block[ti, tj].data,
                        nl.multiply,
                        2.0,
                    )
        dst_blocks[bi].store(block_row.data)

    return dst


@nki.jit
def load_block_subgrid(src):
    """Range slicing [a:b, c:d] over the block grid: coalesced load of a
    rectangular sub-grid of blocks. Pass-through everything; overwrite the
    inner sub-grid scaled by 4x."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))
    dst_blocks = nt.blocks(dst, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))

    # Pass-through everything first.
    for bi in range(src_blocks.shape[0]):
        for bj in range(src_blocks.shape[1]):
            block = src_blocks[bi, bj].load()
            dst_blocks[bi, bj].store(block.data)

    # Overwrite the inner block sub-grid (block-rows 1..2, block-cols 0..1).
    sub = src_blocks[1:3, 0:2].load()
    for bi in range(sub.shape[0]):
        for bj in range(sub.shape[1]):
            block = sub[bi, bj]
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    nisa.tensor_scalar(
                        block[ti, tj].data,
                        block[ti, tj].data,
                        nl.multiply,
                        4.0,
                    )
    dst_blocks[1:3, 0:2].store(sub.data)

    return dst


# ============================================================================
# Block <-> tile conversions
# ============================================================================


@nki.jit
def promote_tile_to_block_view(src):
    """Start with a tile view, promote to a block view by passing it
    to nt.blocks(). The existing tile_size is inherited; only
    block_size= is needed for the promotion."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    # Tile-grid view: shape (8, 4) on a (1024, 512) tensor.
    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    # Promote: prepend block axis on the existing tile view.
    src_blocks = nt.blocks(src_tiles, block_size=(BLOCK_P, BLOCK_F))
    dst_blocks = nt.blocks(dst, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))

    for bi in range(src_blocks.shape[0]):
        for bj in range(src_blocks.shape[1]):
            block = src_blocks[bi, bj].load()
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    nisa.tensor_scalar(
                        block[ti, tj].data,
                        block[ti, tj].data,
                        nl.multiply,
                        2.0,
                    )
            dst_blocks[bi, bj].store(block.data)

    return dst


@nki.jit
def descend_block_to_tile_view(src):
    """Strip the block axis from a block view by passing it to
    nt.tiles() without a new tile_size=. The result iterates the
    underlying tile grid directly (no block grouping)."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    # Block-grid view: shape (4, 2) on a (1024, 512) tensor.
    src_blocks = nt.blocks(src, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))
    # Descend: drop the block axis, keep the existing tile_size.
    src_tiles = nt.tiles(src_blocks)
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()
            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
            dst_tiles[i, j].store(tile.data)

    return dst


@nki.jit
def retile_block_view(src):
    """Re-tile a block view by passing a new tile_size= to nt.tiles().
    The block grouping is dropped; the new tile_size narrows the F
    extent per tile (here from 128 to 64 columns)."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(TILE_P, TILE_F), block_size=(BLOCK_P, BLOCK_F))
    # Re-tile: shape changes from (4, 2) blocks to (8, 8) tiles of (128, 64).
    src_tiles = nt.tiles(src_blocks, tile_size=(TILE_P, 64))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, 64))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()  # tile: [128, 64]
            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
            dst_tiles[i, j].store(tile.data)

    return dst


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


def _make_src():
    torch.manual_seed(42)
    M = BLOCK_ROWS * BLOCK_P * TILE_P
    N = BLOCK_COLS * BLOCK_F * TILE_F
    return torch.randn(M, N, dtype=torch.bfloat16)


# ============================================================================
# Tests
# ============================================================================


def test_block_level_scale():
    src = _make_src()
    result = block_level_scale(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("block_level_scale: PASSED")


def test_load_block_row():
    src = _make_src()
    result = load_block_row(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("load_block_row: PASSED")


def test_load_block_subgrid():
    src = _make_src()
    result = load_block_subgrid(to_device(src))
    expected = src.clone()
    # Block-rows 1..2 cover tile-rows [2..6) -> elem rows [256..768).
    # Block-cols 0..1 cover the full N width.
    expected[2 * TILE_P : 6 * TILE_P, :] *= 4.0
    assert torch.allclose(to_cpu(result), expected, rtol=1e-3, atol=1e-3)
    print("load_block_subgrid: PASSED")


def test_promote_tile_to_block_view():
    src = _make_src()
    result = promote_tile_to_block_view(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("promote_tile_to_block_view: PASSED")


def test_descend_block_to_tile_view():
    src = _make_src()
    result = descend_block_to_tile_view(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("descend_block_to_tile_view: PASSED")


def test_retile_block_view():
    src = _make_src()
    result = retile_block_view(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("retile_block_view: PASSED")


def main():
    test_block_level_scale()
    test_load_block_row()
    test_load_block_subgrid()
    test_promote_tile_to_block_view()
    test_descend_block_to_tile_view()
    test_retile_block_view()


if __name__ == "__main__":
    main()
