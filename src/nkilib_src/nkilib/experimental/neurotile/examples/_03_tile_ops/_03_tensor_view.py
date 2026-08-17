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
Transform-then-tile: build reshape / permute chains BEFORE tiling by calling
the raw tensor's own native view ops (.flatten_dims / .reshape_dim / .permute /
.slice), then pass the transformed tensor to nt.tiles. Also shows direct
higher-rank tile loads.
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
def tensor_view_chain(src):
    """tensor_view + flatten_dims + reshape_dim + permute -> 3-D logical view,
    then tile and scale. Demonstrates chaining transforms before tiling."""
    # src: [B, S, H]  ->  dst: [128, B*S*H1]   where H = 128*H1
    B = src.shape[0]
    S = src.shape[1]
    H = src.shape[2]
    H0 = 128
    H1 = H // H0

    dst = nl.ndarray((H0, B * S * H1), dtype=src.dtype, buffer=nl.shared_hbm)

    # Transform the raw tensor with its own native view ops, then tile it.
    view = src.flatten_dims(0, 1)  # [B*S, H]
    view = view.reshape_dim(1, [H0, H1])  # [B*S, H0, H1]
    view = view.permute((1, 0, 2))  # [H0, B*S, H1]

    src_tiles = nt.tiles(view, tile_size=(H0, B * S, H1))
    tile = src_tiles[0, 0, 0].load()  # tile: [H0, B*S, H1]

    tile_2d = tile.data.reshape((H0, B * S * H1))
    nisa.tensor_scalar(tile_2d, tile_2d, nl.multiply, 2.0)

    dst_tiles = nt.tiles(dst, tile_size=(H0, B * S * H1))
    dst_tiles[0, 0].store(tile_2d)
    return dst


@nki.jit
def tensor_view_select(src):
    """Select a batch slab via single-int indexing on a 3D tile view."""
    # src: [4, 128, 16]  ->  dst: [128, 16]   (one batch slab, scaled)
    dst = nl.ndarray((128, 16), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 16))  # 3D source, 2D tile -> dim 0 = batch
    slab = src_tiles[0]  # batch 0 -- already at single-tile element-level
    tile = slab.load()  # tile: [128, 16]

    nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 3.0)

    dst_tiles = nt.tiles(dst, tile_size=(128, 16))
    dst_tiles[0, 0].store(tile.data)
    return dst


@nki.jit
def tensor_view_3d_direct(src):
    """Load a 3D tensor as a 3D tile, scale, store back."""
    # src: [128, 4, 32]  ->  dst: [128, 4, 32]
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 4, 32))
    dst_tiles = nt.tiles(dst, tile_size=(128, 4, 32))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            for k in range(src_tiles.shape[2]):
                tile = src_tiles[i, j, k].load()  # tile: [128, 4, 32]
                tile_2d = tile.data.reshape((128, 4 * 32))
                nisa.tensor_scalar(tile_2d, tile_2d, nl.multiply, 5.0)
                dst_tiles[i, j, k].store(tile_2d.reshape((128, 4, 32)))

    return dst


@nki.jit
def nd_tile_alloc_sbuf(src):
    """Load 3D tile, scale, dma_copy to flat 2D dst -- demonstrates that
    the SBUF physical layout is 2D regardless of the logical 3D shape."""
    # src: [128, 4, 32]  ->  dst: [128, 4*32]
    P, M, N = src.shape
    dst = nl.ndarray((P, M * N), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 4, 32))
    tile = src_tiles[0, 0, 0].load()  # tile: [128, 4, 32]
    nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 3.0)

    nisa.dma_copy(dst, tile.data)
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


def test_tensor_view_chain():
    torch.manual_seed(42)
    B, S, H = 1, 4, 256
    H0 = 128
    H1 = H // H0
    src = torch.randn(B, S, H, dtype=torch.bfloat16)
    result = tensor_view_chain(to_device(src))
    expected = (src.reshape(B * S, H0, H1).permute(1, 0, 2) * 2.0).reshape(H0, B * S * H1)
    assert torch.allclose(
        to_cpu(result).to(torch.float32),
        expected.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print("tensor_view_chain: PASSED")


def test_tensor_view_select():
    torch.manual_seed(42)
    src = torch.randn(4, 128, 16, dtype=torch.bfloat16)
    result = tensor_view_select(to_device(src))
    expected = src[0] * 3.0
    assert torch.allclose(
        to_cpu(result).to(torch.float32),
        expected.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print("tensor_view_select: PASSED")


def test_tensor_view_3d_direct():
    torch.manual_seed(42)
    src = torch.randn(128, 4, 32, dtype=torch.float32)
    result = tensor_view_3d_direct(to_device(src))
    expected = src * 5.0
    assert torch.allclose(to_cpu(result), expected, rtol=1e-5, atol=1e-5)
    print("tensor_view_3d_direct: PASSED")


def test_nd_tile_alloc_sbuf():
    torch.manual_seed(42)
    src = torch.randn(128, 4, 32, dtype=torch.float32)
    result = nd_tile_alloc_sbuf(to_device(src))
    expected = (src * 3.0).reshape(128, 128)
    assert torch.allclose(to_cpu(result), expected, rtol=1e-5, atol=1e-5)
    print("nd_tile_alloc_sbuf: PASSED")


def main():
    test_tensor_view_chain()
    test_tensor_view_select()
    test_tensor_view_3d_direct()
    test_nd_tile_alloc_sbuf()


if __name__ == "__main__":
    main()
