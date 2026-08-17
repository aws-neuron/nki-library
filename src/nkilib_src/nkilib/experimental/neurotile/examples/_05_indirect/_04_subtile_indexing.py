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
Sub-tile indexing patterns: pre-load view composition (compose the view
before .load() to narrow what's loaded) and post-load extraction
(slice the SBUF tile after .load() for on-chip rearrangement).
"""

import nki
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

# ============================================================================
# Kernels -- pre-load view composition
# ============================================================================


@nki.jit
def static_view_select(data, select_idx):
    """Select a 2D slab from a 3D tensor via static int index."""
    # data: [D0, P, F],  select_idx: int  ->  output: [P, F]
    D0, P, F = data.shape
    out = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(P, F))
    out_iter = nt.tiles(out, tile_size=(P, F))

    sub_view = data_iter[select_idx]  # one slab
    tile = sub_view.load()  # tile: [P, F]
    out_iter[0, 0].store(tile.data)
    return out


@nki.jit
def chained_view_select(data, slab_idx, row_idx):
    """Two static selects: slab then row within slab."""
    # data: [D0, P, F],  slab_idx/row_idx: int  ->  output: [1, F]
    D0, P, F = data.shape
    out = nl.ndarray((1, F), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(P, F))
    out_iter = nt.tiles(out, tile_size=(1, F))

    slab_view = data_iter[slab_idx]
    row_view = slab_view[row_idx, :]  # narrow PART to one row
    tile = row_view.load()  # tile: [1, F]
    out_iter[0, 0].store(tile.data)
    return out


@nki.jit
def loop_view_select(data):
    """Loop over E slabs, select each via int index, store in reverse order."""
    # data: [E, P, F]  ->  output: [E, P, F]   (slab order reversed)
    E, P, F = data.shape
    out = nl.ndarray((E, P, F), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(P, F))
    out_iter = nt.tiles(out, tile_size=(P, F))

    for e in range(E):
        slab = data_iter[e].load()  # tile: [P, F]
        out_iter[E - 1 - e].store(slab.data)

    return out


# ============================================================================
# Kernels -- post-load extraction (sub-tile slicing on SBUF)
# ============================================================================


@nki.jit
def tile_row_extract(data, row_idx):
    """Load full tile, then extract one row via tile[row_idx]."""
    # data: [P, F],  row_idx: int  ->  output: [1, F]
    P, F = data.shape
    out = nl.ndarray((1, F), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(P, F))
    out_iter = nt.tiles(out, tile_size=(1, F))

    tile = data_iter[0, 0].load()  # tile: [P, F]
    sub = tile[row_idx]  # [1, F]
    out_iter[0, 0].store(sub.data)
    return out


@nki.jit
def tile_element_extract(data, row_idx, col_idx):
    """Extract a single element via tile[row, col]."""
    # data: [P, F],  row_idx/col_idx: int  ->  output: [1, 1]
    P, F = data.shape
    out = nl.ndarray((1, 1), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(P, F))
    out_iter = nt.tiles(out, tile_size=(1, 1))

    tile = data_iter[0, 0].load()
    sub = tile[row_idx, col_idx]  # [1, 1]
    out_iter[0, 0].store(sub.data)
    return out


@nki.jit
def tile_subblock_extract(data, row_start, row_count, col_start, col_count):
    """Extract a rectangular sub-block via tile.data[r0:r1, c0:c1]."""
    # data: [P, F]  ->  output: [row_count, col_count]
    P, F = data.shape
    out = nl.ndarray((row_count, col_count), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(P, F))
    out_iter = nt.tiles(out, tile_size=(row_count, col_count))

    tile = data_iter[0, 0].load()
    sub_data = tile.data[row_start : row_start + row_count, col_start : col_start + col_count]
    out_iter[0, 0].store(sub_data)
    return out


@nki.jit
def dynamic_select_row_extract(weights, expert_id_tensor, row_idx):
    """Dynamic expert select followed by a static row extract on the loaded tile."""
    # weights: [E, P, F],  expert_id_tensor: [1, 1],  row_idx: int  ->  output: [1, F]
    E, P, F = weights.shape
    out = nl.ndarray((1, F), dtype=weights.dtype, buffer=nl.shared_hbm)

    w_iter = nt.tiles(weights, tile_size=(P, F))
    eid_iter = nt.tiles(expert_id_tensor, tile_size=(1, 1))
    out_iter = nt.tiles(out, tile_size=(1, F))

    eid_tile = eid_iter[0, 0].load()
    expert = w_iter[eid_tile].load()  # tile: [P, F]
    sub = expert[row_idx]  # [1, F]
    out_iter[0, 0].store(sub.data)
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


def test_static_view_select():
    torch.manual_seed(200)
    D0, P, F = 4, 128, 64
    data = torch.randn(D0, P, F, dtype=torch.bfloat16)
    result = static_view_select(to_device(data), 2)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), data[2].to(torch.float32))
    print("static_view_select: PASSED")


def test_chained_view_select():
    torch.manual_seed(201)
    D0, P, F = 4, 128, 64
    data = torch.randn(D0, P, F, dtype=torch.bfloat16)
    result = chained_view_select(to_device(data), 1, 42)
    expected = data[1, 42, :].reshape(1, F)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("chained_view_select: PASSED")


def test_loop_view_select():
    torch.manual_seed(202)
    E, P, F = 4, 128, 64
    data = torch.randn(E, P, F, dtype=torch.bfloat16)
    result = loop_view_select(to_device(data))
    expected = data.flip(0)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("loop_view_select: PASSED")


def test_tile_row_extract():
    torch.manual_seed(203)
    P, F = 128, 64
    data = torch.randn(P, F, dtype=torch.bfloat16)
    result = tile_row_extract(to_device(data), 5)
    expected = data[5, :].reshape(1, F)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("tile_row_extract: PASSED")


def test_tile_element_extract():
    torch.manual_seed(205)
    P, F = 128, 64
    data = torch.randn(P, F, dtype=torch.bfloat16)
    result = tile_element_extract(to_device(data), 42, 17)
    expected = data[42, 17].reshape(1, 1)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("tile_element_extract: PASSED")


def test_tile_subblock_extract():
    torch.manual_seed(207)
    P, F = 128, 64
    data = torch.randn(P, F, dtype=torch.bfloat16)
    result = tile_subblock_extract(to_device(data), 8, 16, 4, 32)
    expected = data[8:24, 4:36]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("tile_subblock_extract: PASSED")


def test_dynamic_select_row_extract():
    torch.manual_seed(204)
    E, P, F = 8, 128, 64
    w = torch.randn(E, P, F, dtype=torch.bfloat16)
    eid = torch.tensor([[5]], dtype=torch.int32)
    result = dynamic_select_row_extract(to_device(w), to_device(eid), 42)
    expected = w[5, 42, :].reshape(1, F)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("dynamic_select_row_extract: PASSED")


def main():
    test_static_view_select()
    test_chained_view_select()
    test_loop_view_select()
    test_tile_row_extract()
    test_tile_element_extract()
    test_tile_subblock_extract()
    test_dynamic_select_row_extract()


if __name__ == "__main__":
    main()
