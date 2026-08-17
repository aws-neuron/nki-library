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
DMA transpose via ``view.load(transpose=True)``, the only neurotile transpose
path (HBM->SBUF). Transposing is the tile-coordinate swap: source tile [mi, ni]
loads transposed into output tile [ni, mi]. Covers per-tile, coalesced tile-row,
caller-provided dst=, streaming, and indirect gather (2-D + N-D transpose_axes=).
"""

import nki
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

# ============================================================================
# Static transpose
# ============================================================================


@nki.jit
def transpose_tiled(src):
    """Transpose-load each tile [mi, ni] into output tile [ni, mi]."""
    # src: [M, N]  ->  out: [N, M]
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    for mi in range(src_tiles.shape[0]):
        for ni in range(src_tiles.shape[1]):
            xposed = src_tiles[mi, ni].load(transpose=True)  # one DMA per tile; src tile [mi, ni], transposed
            out_tiles[ni, mi].store(xposed.data)  # swapped coords == the transpose
    return out


@nki.jit
def transpose_coalesced(src):
    """Coalesce a tile-row src_tiles[mi, :] into ONE transpose DMA; the result keeps its tile grid, so index each transposed chunk directly."""
    # src: [M, N] (M, N multiples of 128)  ->  out: [N, M]
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 128))  # (M/128, N/128) tile grid
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    for mi in range(src_tiles.shape[0]):
        packed = src_tiles[mi, :].load(
            transpose=True
        )  # ONE DMA: tile-row mi (all N/128 tiles) transposed; grid (1, N/128)
        for ni in range(packed.shape[1]):
            out_tiles[ni, mi].store(packed[0, ni].data)  # chunk ni == src(mi,ni).T -> out tile [ni, mi]
    return out


@nki.jit
def transpose_coalesced_single_store(src):
    """Coalesce the tile-row transpose into ONE load DMA AND store it in ONE DMA.

    The transpose already happened on load, so packed.data (128, N) holds chunk ni
    on free columns [ni*128:(ni+1)*128] -- but each chunk's home is a partition
    row-band of out. A plain store maps partition->row / free->col and cannot
    scatter free-chunks onto row-bands, so rearrange the destination column with
    the einops-style (ni p) q -> p ni q: the view then iterates (p, ni, q),
    matching packed.data's read order, and the whole column stores in ONE DMA."""
    # src: [M, N] (M, N multiples of 128)  ->  out: [N, M]
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    for mi in range(src_tiles.shape[0]):
        packed = src_tiles[mi, :].load(transpose=True)  # ONE DMA -> (128, N), grid (1, N/128)
        dst = out_tiles[:, mi].rearrange((("ni", "p"), "q"), ("p", "ni", "q"), {"p": 128})
        dst.store(packed.data)  # ONE DMA: free chunks scattered to their row-bands
    return out


# ============================================================================
# Static transpose into a caller-provided SBUF buffer (dst=).
# ============================================================================


@nki.jit
def transpose_into_dst(src):
    """Transpose into a caller-provided SBUF buffer via dst= (reused across the grid)."""
    # src: [M, N]  ->  out: [N, M]   (M, N multiples of 128 so every tile is 128x128)
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    dst = nl.ndarray((128, 128), dtype=src.dtype, buffer=nl.sbuf)  # caller-managed transpose buffer
    for mi in range(src_tiles.shape[0]):
        for ni in range(src_tiles.shape[1]):
            xposed = src_tiles[mi, ni].load(transpose=True, dst=dst)  # transpose into caller's buffer
            out_tiles[ni, mi].store(xposed.data)
    return out


@nki.jit
def transpose_into_dst_partial(src):
    """dst= sized to a non-128-square transposed shape (F<=128 -> dst is (F, P))."""
    # src: [P, F] (F <= 128)  ->  out: [F, P]
    p, f = src.shape
    out = nl.ndarray((f, p), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(p, f))  # single (P, F) tile
    out_tiles = nt.tiles(out, tile_size=(f, p))
    dst = nl.ndarray((f, p), dtype=src.dtype, buffer=nl.sbuf)  # caller buffer == transposed shape (F, P)
    xposed = src_tiles[0, 0].load(transpose=True, dst=dst)
    out_tiles[0, 0].store(xposed.data)
    return out


# ============================================================================
# Static transpose -- streaming. Each tile flows through a rotating buffer.
# ============================================================================


@nki.jit
def transpose_streamed(src):
    """Transpose-load each tile through a rotating double buffer so DMA overlaps compute."""
    # src: [M, N]  ->  out: [N, M]
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    for mi in range(src_tiles.shape[0]):
        row_stream = src_tiles[mi].stream(buffer_count=2)  # walks row mi's column tiles
        for ni in range(src_tiles.shape[1]):
            xposed = row_stream.load(ni, transpose=True)  # slot: transposed tile [mi, ni]
            out_tiles[ni, mi].store(xposed.data)
    return out


@nki.jit
def transpose_block_streamed(src):
    """Block-stream contiguous seq-tile blocks: one transpose DMA per block through a rotating slot."""
    # src: [M, H] (H <= 128, M a multiple of BT*128)  ->  out: [H, M]. A contiguous block
    # of BT seq-tiles transposes in ONE DMA; the rotating slot overlaps the next block's DMA
    # with the current block's stores. Generic over M (any number of blocks).
    m, h = src.shape
    out = nl.ndarray((h, m), dtype=src.dtype, buffer=nl.shared_hbm)

    bt = 2  # seq-tiles per block along M
    s_tiles = nt.tiles(src, tile_size=(128, h))  # (M/128, 1) seq-tile grid
    x_blocks = nt.blocks(s_tiles, block_size=(bt, 1))  # group bt contiguous seq-tiles per block
    x_stream = x_blocks.stream(buffer_count=2, transpose=True)  # slots sized to the transposed block
    out_tiles = nt.tiles(out, tile_size=(h, bt * 128))  # (1, M/(bt*128)) transposed-block grid

    for bi in range(x_blocks.shape[0]):
        slot = x_stream.load(bi, transpose=True)  # one DMA: whole block [bt*128, H] -> [H, bt*128]
        out_tiles[0, bi].store(slot.data)  # one DMA: transposed block -> its column span
    return out


@nki.jit
def transpose_block_streamed_sharded(src):
    """Tile-sharded 2-D block grid: iterate block-columns, stream block-rows, transpose each owned tile into its slot sub-tile."""
    # src: [M, N] (N a multiple of 128). Interleaved M-sharding (shard 0 of NSH) owns every
    # NSH-th M-tile, so a block's owned M-tiles are NOT contiguous in HBM and can't be one
    # coalesced transpose DMA. Two loops: outer over block-columns (bj), inner streams the
    # block-rows of that column (bi); each owned tile [ti] transposes into its slot sub-tile.
    m, n = src.shape
    n_m_tiles = m // 128
    nsh = 2  # shards; this kernel runs shard 0
    bt = 2  # owned M-tiles per block
    n_owned = n_m_tiles // nsh
    out = nl.ndarray((n, n_owned * 128), dtype=src.dtype, buffer=nl.shared_hbm)

    own_r = nt.interleaved_range(rank=0, num_shards=nsh, total=n_m_tiles)  # M-tiles 0, nsh, 2*nsh, ...
    s_tiles = nt.tiles(src, tile_size=(128, 128))[own_r, :]  # owned (M, N) tile grid
    x_blocks = nt.blocks(s_tiles, block_size=(bt, 1))  # group bt owned M-tiles per block
    out_tiles = nt.tiles(out, tile_size=(128, 128))  # (N/128, n_owned) transposed-tile grid

    for bj in range(x_blocks.shape[1]):  # iterate block-columns
        col = x_blocks[:, bj]
        col_stream = col.stream(buffer_count=2, transpose=True)  # slot grid-typed to owned tiles
        for bi in range(col_stream.count):  # stream block-rows of this column
            slot = col_stream[bi]
            blk_tiles = nt.tiles(col[bi])  # this block's owned M-tiles
            for ti in range(blk_tiles.shape[0]):
                blk_tiles[ti, 0].load(transpose=True, dst=slot[0, ti].data)  # one transpose DMA per owned tile
                out_tiles[bj, bi * bt + ti].store(slot[0, ti].data)  # transposed tile -> its block
    return out


# ============================================================================
# Indirect (gather) transpose -- gather rows via an index tile, then transpose.
# ============================================================================


@nki.jit
def gather_transpose(data, indices):
    """Gather rows of data[N, D] via indices[N, 1] and transpose to (D, N) in one pass."""
    # data: [N, D], indices: [N, 1]  ->  out: [D, N]
    n, d = data.shape
    out = nl.ndarray((d, n), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(n, d))
    idx_tile = nt.tiles(indices, tile_size=(n, 1))[0, 0].load()
    xposed = data_iter[idx_tile, 0].load(transpose=True)  # gather + transpose -> [D, N]
    out_tiles = nt.tiles(out, tile_size=(d, n))
    out_tiles[0, 0].store(xposed.data)
    return out


@nki.jit
def gather_transpose_3d(data, indices):
    """3-D gather-transpose: gather rows, axes=(2,1,0) -> (tile, n_tiles, rows)."""
    # data: [rows, n_tiles, tile], indices: [rows, 1]  ->  out: [tile, n_tiles, rows]
    rows, n_tiles, tile = data.shape
    out = nl.ndarray((tile, n_tiles, rows), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(rows, n_tiles, tile))
    idx_tile = nt.tiles(indices, tile_size=(rows, 1))[0, 0].load()
    xposed = data_iter[idx_tile, 0, 0].load(transpose=True, transpose_axes=(2, 1, 0))
    out_tiles = nt.tiles(out, tile_size=(tile, n_tiles, rows))
    out_tiles[0, 0, 0].store(xposed.data)
    return out


@nki.jit
def gather_transpose_4d(data, indices):
    """4-D gather-transpose: gather rows, axes=(3,1,2,0) -> (P, 1, f_tiles, rows)."""
    # data: [rows, f_tiles, P], indices: [rows, 1]  ->  out: [P, 1, f_tiles, rows]
    rows, f_tiles, p = data.shape
    out = nl.ndarray((p, 1, f_tiles, rows), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(rows, f_tiles, p))
    idx_tile = nt.tiles(indices, tile_size=(rows, 1))[0, 0].load()
    xposed = data_iter[idx_tile, 0, 0].load(transpose=True, transpose_axes=(3, 1, 2, 0))
    out_tiles = nt.tiles(out, tile_size=(p, 1, f_tiles, rows))
    out_tiles[0, 0, 0, 0].store(xposed.data)
    return out


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


def _check(result, expected):
    assert torch.allclose(to_cpu(result).to(torch.float32), expected.to(torch.float32), rtol=1e-2, atol=1e-2)


# ============================================================================
# Tests
# ============================================================================


def test_transpose_direct():
    # F <= 128: a single 128x128 tile (the one-tile case of the grid walk).
    torch.manual_seed(42)
    src = torch.randn(128, 64, dtype=torch.bfloat16)
    _check(transpose_tiled(to_device(src)), src.t().contiguous())
    print("transpose_direct: PASSED")


def test_transpose_multi_tile():
    # 2x4 tile grid: exercises the generic double loop over row and column tiles.
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    _check(transpose_tiled(to_device(src)), src.t().contiguous())
    print("transpose_multi_tile: PASSED")


def test_transpose_coalesced():
    # One transpose DMA per full-width (128, N) tile; N % 128 == 0.
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    _check(transpose_coalesced(to_device(src)), src.t().contiguous())
    print("transpose_coalesced: PASSED")


def test_transpose_remainder():
    # F % 128 != 0 (400 = 3*128 + 16): the trailing column tile is narrower.
    torch.manual_seed(42)
    src = torch.randn(128, 400, dtype=torch.bfloat16)
    _check(transpose_tiled(to_device(src)), src.t().contiguous())
    print("transpose_remainder: PASSED")


def test_transpose_coalesced_single_store():
    # One transpose DMA per tile-row AND one store DMA per row (rearrange scatter).
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    _check(transpose_coalesced_single_store(to_device(src)), src.t().contiguous())
    print("transpose_coalesced_single_store: PASSED")


def test_transpose_into_dst():
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    _check(transpose_into_dst(to_device(src)), src.t().contiguous())
    print("transpose_into_dst: PASSED")


def test_transpose_into_dst_partial():
    # dst= sized to a non-128-square transposed shape: (128, 100) -> dst (100, 128).
    torch.manual_seed(42)
    src = torch.randn(128, 100, dtype=torch.bfloat16)
    _check(transpose_into_dst_partial(to_device(src)), src.t().contiguous())
    print("transpose_into_dst_partial: PASSED")


def test_transpose_streamed():
    torch.manual_seed(7)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    _check(transpose_streamed(to_device(src)), src.t().contiguous())
    print("transpose_streamed: PASSED")


def test_transpose_block_streamed():
    # 6 seq-tiles grouped into contiguous blocks of 2; one transpose DMA per block.
    torch.manual_seed(7)
    src = torch.randn(128 * 6, 96, dtype=torch.bfloat16)
    _check(transpose_block_streamed(to_device(src)), src.t().contiguous())
    print("transpose_block_streamed: PASSED")


def test_transpose_block_streamed_sharded():
    # 8x4 tile grid; shard 0 of 2 owns even M-tiles. Iterate cols, stream owned rows.
    torch.manual_seed(7)
    src = torch.randn(128 * 8, 128 * 4, dtype=torch.bfloat16)
    owned = torch.cat([src[t * 128 : (t + 1) * 128, :] for t in range(0, 8, 2)], dim=0)
    _check(transpose_block_streamed_sharded(to_device(src)), owned.t().contiguous())
    print("transpose_block_streamed_sharded: PASSED")


def test_gather_transpose():
    torch.manual_seed(42)
    n, d = 16, 64
    data = torch.randn(n, d, dtype=torch.bfloat16)
    indices = torch.randperm(n).to(torch.uint32).reshape(n, 1)
    expected = data[indices.flatten().long(), :].t().contiguous()
    _check(gather_transpose(to_device(data), to_device(indices)), expected)
    print("gather_transpose: PASSED")


def test_gather_transpose_3d():
    torch.manual_seed(42)
    rows, n_tiles, tile = 16, 4, 64
    data = torch.randn(rows, n_tiles, tile, dtype=torch.bfloat16)
    indices = torch.randperm(rows).to(torch.uint32).reshape(rows, 1)
    expected = data[indices.flatten().long()].permute(2, 1, 0).contiguous()
    _check(gather_transpose_3d(to_device(data), to_device(indices)), expected)
    print("gather_transpose_3d: PASSED")


def test_gather_transpose_4d():
    torch.manual_seed(42)
    rows, f_tiles, p = 16, 8, 128
    data = torch.randn(rows, f_tiles, p, dtype=torch.bfloat16)
    indices = torch.randperm(rows).to(torch.uint32).reshape(rows, 1)
    gathered = data[indices.flatten().long()]
    expected = gathered.reshape(rows, 1, f_tiles, p).permute(3, 1, 2, 0).contiguous()
    _check(gather_transpose_4d(to_device(data), to_device(indices)), expected)
    print("gather_transpose_4d: PASSED")


def main():
    test_transpose_direct()
    test_transpose_multi_tile()
    test_transpose_coalesced()
    test_transpose_coalesced_single_store()
    test_transpose_remainder()
    test_transpose_into_dst()
    test_transpose_into_dst_partial()
    test_transpose_streamed()
    test_transpose_block_streamed()
    test_transpose_block_streamed_sharded()
    test_gather_transpose()
    test_gather_transpose_3d()
    test_gather_transpose_4d()


if __name__ == "__main__":
    main()
