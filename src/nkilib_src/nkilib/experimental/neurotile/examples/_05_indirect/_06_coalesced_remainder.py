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
Coalesced remainder handling: when the source extent isn't a multiple
of tile_size, the trailing tile is partial. view.is_remainder reports
when a tile/row/column straddles or sits on the partial trailing tile;
guard with oob_mode.skip + oob_value=0.0 on those iterations.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

# ============================================================================
# Kernels
# ============================================================================


@nki.jit
def row_hoist_f_remainder(src):
    """Row hoist on a tensor where F doesn't divide evenly. The trailing
    column tile is partial; row.is_remainder fires for the row that
    contains it."""
    # src: [300, 500] (F doesn't divide evenly by tile_F=128)
    #   ->  dst: [300, 500]
    P, F = src.shape
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))

    for i in range(tiles.shape[0]):
        row = tiles[i, :]  # row of tiles
        if row.is_remainder:
            data = row.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
        else:
            data = row.load()
        out_tiles[i, :].store(data.data, oob_mode=nisa.oob_mode.skip)

    return out


@nki.jit
def col_hoist_p_remainder(src):
    """Column hoist on a tensor where P doesn't divide evenly. Per-tile
    is_remainder check guards the trailing P-tile."""
    # src: [300, 512] (P doesn't divide evenly by tile_P=128)
    #   ->  dst: [300, 512]
    P, F = src.shape
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))

    for j in range(tiles.shape[1]):
        for i in range(tiles.shape[0]):
            tile_view = tiles[i, j]
            if tile_view.is_remainder:
                data = tile_view.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
            else:
                data = tile_view.load()
            out_tiles[i, j].store(data.data, oob_mode=nisa.oob_mode.skip)

    return out


@nki.jit
def range_slice_remainder(src):
    """Per-tile load/store on a boundary range slice."""
    # src: [300, 500]  ->  dst: [300, 500]   (only tile-rows 1..2 written)
    P, F = src.shape
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))

    for i in range(1, 3):
        for j in range(tiles.shape[1]):
            tile_view = tiles[i, j]
            if tile_view.is_remainder:
                data = tile_view.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
            else:
                data = tile_view.load()
            out_tiles[i, j].store(data.data, oob_mode=nisa.oob_mode.skip)

    return out


@nki.jit
def store_oob_mode(src):
    """Store with oob_mode.skip on a row hoist with F-remainder."""
    # src: [128, 500] (one tile-row, F-remainder)  ->  dst: [128, 500]
    P, F = src.shape
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))

    row = tiles[0, :]
    if row.is_remainder:
        data = row.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
    else:
        data = row.load()
    out_tiles[0, :].store(data.data, oob_mode=nisa.oob_mode.skip)
    return out


@nki.jit
def multi_range_interior(src):
    """Interior sub-grid (no remainder): is_remainder is False, so plain
    .load() / .store() take the fast path with no OOB overhead."""
    # src: [300, 500]  ->  dst: [300, 500]   (only interior tiles[0:2, 0:3])
    P, F = src.shape
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))

    sub = tiles[0:2, 0:3]
    if sub.is_remainder:
        data = sub.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
    else:
        data = sub.load()  # interior fast path
    out_tiles[0:2, 0:3].store(data.data)
    return out


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


def test_row_hoist_f_remainder():
    np.random.seed(42)
    src_np = np.random.randn(300, 500).astype(np.float32)
    src = torch.from_numpy(src_np).to(torch.bfloat16)
    result = row_hoist_f_remainder(to_device(src))
    expected = src.to(torch.float32).numpy()
    np.testing.assert_allclose(
        to_cpu(result).to(torch.float32).numpy()[:256, :],
        expected[:256, :],
        rtol=0,
        atol=0,
    )
    print("row_hoist_f_remainder: PASSED")


def test_col_hoist_p_remainder():
    np.random.seed(43)
    src_np = np.random.randn(300, 512).astype(np.float32)
    src = torch.from_numpy(src_np).to(torch.bfloat16)
    result = col_hoist_p_remainder(to_device(src))
    expected = src.to(torch.float32).numpy()
    np.testing.assert_allclose(
        to_cpu(result).to(torch.float32).numpy()[:256, :],
        expected[:256, :],
        rtol=0,
        atol=0,
    )
    print("col_hoist_p_remainder: PASSED")


def test_range_slice_remainder():
    np.random.seed(44)
    src_np = np.random.randn(300, 500).astype(np.float32)
    src = torch.from_numpy(src_np).to(torch.bfloat16)
    result = range_slice_remainder(to_device(src))
    expected = src.to(torch.float32).numpy()
    np.testing.assert_allclose(
        to_cpu(result).to(torch.float32).numpy()[128:256, :384],
        expected[128:256, :384],
        rtol=0,
        atol=0,
    )
    print("range_slice_remainder: PASSED")


def test_store_oob_mode():
    np.random.seed(46)
    src_np = np.random.randn(128, 500).astype(np.float32)
    src = torch.from_numpy(src_np).to(torch.bfloat16)
    result = store_oob_mode(to_device(src))
    expected = src.to(torch.float32).numpy()
    np.testing.assert_allclose(
        to_cpu(result).to(torch.float32).numpy()[:, :384],
        expected[:, :384],
        rtol=0,
        atol=0,
    )
    print("store_oob_mode: PASSED")


def test_multi_range_interior():
    np.random.seed(51)
    src_np = np.random.randn(300, 500).astype(np.float32)
    src = torch.from_numpy(src_np).to(torch.bfloat16)
    result = multi_range_interior(to_device(src))
    expected = src.to(torch.float32).numpy()
    np.testing.assert_allclose(
        to_cpu(result).to(torch.float32).numpy()[:256, :384],
        expected[:256, :384],
        rtol=0,
        atol=0,
    )
    print("multi_range_interior: PASSED")


def main():
    test_row_hoist_f_remainder()
    test_col_hoist_p_remainder()
    test_range_slice_remainder()
    test_store_oob_mode()
    test_multi_range_interior()


if __name__ == "__main__":
    main()
