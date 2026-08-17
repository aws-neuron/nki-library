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
Tile iteration: nt.tiles() over HBM with the full indexing surface.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

TILE_P, TILE_F = 128, 128


# ============================================================================
# Basic tile iteration
# ============================================================================


@nki.jit
def basic_tile_copy(src):
    """Per-tile [i, j] iteration: each step loads / stores one tile."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()
            dst_tiles[i, j].store(tile.data)

    return dst


@nki.jit
def direct_indexing(src):
    """Access a single tile by coordinate without iterating."""
    dst = nl.ndarray((TILE_P, TILE_F), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    # tile [1, 1] -> rows [128:256], cols [128:256]
    tile = src_tiles[1, 1].load()
    nisa.dma_copy(dst, tile.data)

    return dst


@nki.jit
def scale_kernel(src, factor=2.0):
    """Element-wise scale demonstrating per-tile compute on indexed iteration."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()
            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, factor)
            dst_tiles[i, j].store(tile.data)

    return dst


# ============================================================================
# Index variations
# ============================================================================


@nki.jit
def load_tile_row(src):
    """Coalesced row-of-tiles DMA: one DMA covers a full tile-row, then
    iterate the loaded SBUF view per-tile for compute."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for i in range(src_tiles.shape[0]):
        row = src_tiles[i].load()
        for j in range(row.shape[0]):
            nisa.tensor_scalar(row[j].data, row[j].data, nl.multiply, 2.0)
        dst_tiles[i].store(row.data)

    return dst


@nki.jit
def load_tile_column(src):
    """Coalesced column-of-tiles DMA."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    for j in range(src_tiles.shape[1]):
        col = src_tiles[:, j].load()
        for i in range(col.shape[0]):
            nisa.tensor_scalar(col[i].data, col[i].data, nl.multiply, 3.0)
        dst_tiles[:, j].store(col.data)

    return dst


@nki.jit
def load_subgrid(src):
    """Range slicing [a:b, c:d]: coalesced load of a rectangular sub-grid of tiles.
    Scales the inner 2x2 sub-grid; copies the rest unchanged."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    # Pass-through everything first.
    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()
            dst_tiles[i, j].store(tile.data)

    # Overwrite the inner 2x2 sub-grid (rows 1..2, cols 1..2) scaled by 4x.
    sub = src_tiles[1:3, 1:3].load()
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            nisa.tensor_scalar(sub[i, j].data, sub[i, j].data, nl.multiply, 4.0)
    dst_tiles[1:3, 1:3].store(sub.data)

    return dst


@nki.jit
def load_negative_index(src):
    """Negative index [-1] reaches the last tile-row."""
    dst = nl.ndarray((TILE_P, src.shape[1]), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    last_row = src_tiles[-1].load()
    for j in range(last_row.shape[0]):
        nisa.tensor_scalar(last_row[j].data, last_row[j].data, nl.multiply, 2.0)
    dst_tiles[0].store(last_row.data)

    return dst


@nki.jit
def load_strided_rows(src):
    """Stepped slice [::2]: walk every other tile-row. Output keeps only
    the even tile-rows packed contiguously."""
    M, N = src.shape
    out_rows = M // 2
    dst = nl.ndarray((out_rows, N), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))

    even = src_tiles[::2]
    for i in range(even.shape[0]):
        row = even[i].load()
        for j in range(row.shape[0]):
            nisa.tensor_scalar(row[j].data, row[j].data, nl.multiply, 2.0)
        dst_tiles[i].store(row.data)

    return dst


# ============================================================================
# Tile shape variations
# ============================================================================


@nki.jit
def batched_tile_iteration(src):
    """tile_size=(P, F) on a 3D source (B, M, N) -- one auto-padded
    batch dim at the front. Index batch first via ``view[b, i, j]``;
    .load() asserts batch is consumed."""
    B, M, N = src.shape
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))
    for b in range(B):
        for i in range(M // TILE_P):
            for j in range(N // TILE_F):
                tile = src_tiles[b, i, j].load()  # tile: [128, 128]
                nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
                dst_tiles[b, i, j].store(tile.data)

    return dst


@nki.jit
def higher_rank_tile(src):
    """tile_size=(P, F1, F2): the F axis stays higher-rank on SBUF, so
    sub-tile slicing addresses F1 and F2 independently. Here each F1
    slice is scaled by a different factor."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, 2, 32))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, 2, 32))
    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            for k in range(src_tiles.shape[2]):
                tile = src_tiles[i, j, k].load()  # tile: [128, 2, 32]
                # Per-F1-slice scaling -- sub-tile slice narrows
                # element_shape from (128, 2, 32) to (128, 1, 32).
                slice0 = tile[:, 0, :]  # [128, 1, 32]
                slice1 = tile[:, 1, :]  # [128, 1, 32]
                nisa.tensor_scalar(slice0.data, slice0.data, nl.multiply, 2.0)
                nisa.tensor_scalar(slice1.data, slice1.data, nl.multiply, 3.0)
                dst_tiles[i, j, k].store(tile.data)

    return dst


@nki.jit
def partition_tile(src):
    """tile_size=(P, 1) -- one partition column per tile. Useful when
    each tile spans the full P axis but only one F element (e.g.
    column-wise reductions)."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, 1))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, 1))
    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()  # tile: [128, 1]
            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
            dst_tiles[i, j].store(tile.data)

    return dst


@nki.jit
def column_tile(src):
    """tile_size=(1, F) -- one P-row per tile. Useful when each tile
    spans the full F axis but only one P element (e.g. row-wise
    scaling)."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(1, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(1, TILE_F))
    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()  # tile: [1, 128]
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
    return torch.randn(4 * TILE_P, 4 * TILE_F, dtype=torch.bfloat16)


# ============================================================================
# Tests
# ============================================================================


def test_basic_tile_copy():
    src = _make_src()
    result = basic_tile_copy(to_device(src))
    assert torch.allclose(to_cpu(result), src, rtol=1e-3, atol=1e-3)
    print("basic_tile_copy: PASSED")


def test_direct_indexing():
    src = _make_src()
    result = direct_indexing(to_device(src))
    expected = src[TILE_P : 2 * TILE_P, TILE_F : 2 * TILE_F]
    assert torch.allclose(to_cpu(result), expected, rtol=1e-3, atol=1e-3)
    print("direct_indexing: PASSED")


def test_load_tile_row():
    src = _make_src()
    result = load_tile_row(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("load_tile_row: PASSED")


def test_load_tile_column():
    src = _make_src()
    result = load_tile_column(to_device(src))
    assert torch.allclose(to_cpu(result), src * 3.0, rtol=1e-3, atol=1e-3)
    print("load_tile_column: PASSED")


def test_load_subgrid():
    src = _make_src()
    result = load_subgrid(to_device(src))
    expected = src.clone()
    expected[TILE_P : 3 * TILE_P, TILE_F : 3 * TILE_F] *= 4.0
    assert torch.allclose(to_cpu(result), expected, rtol=1e-3, atol=1e-3)
    print("load_subgrid: PASSED")


def test_load_negative_index():
    src = _make_src()
    result = load_negative_index(to_device(src))
    expected = src[-TILE_P:, :] * 2.0
    assert torch.allclose(to_cpu(result), expected, rtol=1e-3, atol=1e-3)
    print("load_negative_index: PASSED")


def test_load_strided_rows():
    src = _make_src()
    result = load_strided_rows(to_device(src))
    # Even tile-rows scaled: rows [0:TILE_P], [2*TILE_P:3*TILE_P].
    even = torch.cat([src[0:TILE_P, :], src[2 * TILE_P : 3 * TILE_P, :]], dim=0)
    expected = even * 2.0
    assert torch.allclose(to_cpu(result), expected, rtol=1e-3, atol=1e-3)
    print("load_strided_rows: PASSED")


def test_scale_kernel():
    src = _make_src()
    result = scale_kernel(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("scale_kernel: PASSED")


def test_batched_tile_iteration():
    torch.manual_seed(42)
    src = torch.randn(4, 256, 256, dtype=torch.bfloat16)
    result = batched_tile_iteration(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("batched_tile_iteration: PASSED")


def test_higher_rank_tile():
    torch.manual_seed(42)
    # Source shape (256, 4, 64) divides cleanly by tile_size=(128, 2, 32).
    src = torch.randn(256, 4, 64, dtype=torch.bfloat16)
    result = higher_rank_tile(to_device(src))
    expected = src.clone()
    # Per-F1-slice: even F1 indices scaled by 2.0, odd by 3.0.
    expected[:, ::2, :] *= 2.0
    expected[:, 1::2, :] *= 3.0
    assert torch.allclose(to_cpu(result), expected, rtol=1e-3, atol=1e-3)
    print("higher_rank_tile: PASSED")


def test_partition_tile():
    torch.manual_seed(42)
    src = torch.randn(256, 4, dtype=torch.bfloat16)
    result = partition_tile(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("partition_tile: PASSED")


def test_column_tile():
    torch.manual_seed(42)
    src = torch.randn(4, 256, dtype=torch.bfloat16)
    result = column_tile(to_device(src))
    assert torch.allclose(to_cpu(result), src * 2.0, rtol=1e-3, atol=1e-3)
    print("column_tile: PASSED")


def main():
    test_basic_tile_copy()
    test_direct_indexing()
    test_scale_kernel()
    test_load_tile_row()
    test_load_tile_column()
    test_load_subgrid()
    test_load_negative_index()
    test_load_strided_rows()
    test_batched_tile_iteration()
    test_higher_rank_tile()
    test_partition_tile()
    test_column_tile()


if __name__ == "__main__":
    main()
