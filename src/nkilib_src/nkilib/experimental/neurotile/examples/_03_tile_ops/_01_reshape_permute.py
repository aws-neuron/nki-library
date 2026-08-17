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
View transforms: reshape_dim / permute / flatten_dims chains, plus
sub-tile slice and split. Demonstrates that transforms restructure the
view's logical shape (metadata only) and can be applied either to the
HBM view (before load) or to the SBUF tile (after load).
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
def norm_reshape_transpose(src_hbm):
    """[BxS, H] -> reshape_dim + flatten_dims + load(transpose=True) -> [H0, BxS*H1].
    Shows reshape chain on an HBM view before a transpose-load."""
    # src_hbm: [4, 1024]  ->  dst: [128, 32]
    BxS, H = 4, 1024
    H0, H1 = 128, 8

    dst = nl.ndarray((H0, BxS * H1), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(BxS, H))
    hbm_view = (
        src_tiles[0, 0]
        .reshape_dim(dim=1, shape=[H1, H0])  # [4, 8, 128]
        .flatten_dims(start_dim=0, end_dim=1)  # [32, 128]
    )
    tile = hbm_view.load(transpose=True)  # tile: [128, 32]
    nisa.dma_copy(dst, tile.data)
    return dst


@nki.jit
def reshape_permute_load(src_hbm):
    """Non-contiguous load via reshape_dim + permute. Produces a permuted
    layout in SBUF without any extra DMA passes."""
    # src_hbm: [4, 1024]  ->  dst: [128, 4, 8]
    dst = nl.ndarray((128, 4, 8), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(4, 1024))
    src_view = src_tiles[0, 0].reshape_dim(dim=1, shape=[8, 128]).permute(dims=[2, 0, 1])
    tile = src_view.load()  # tile: [128, 4, 8]

    dst_tiles = nt.tiles(dst, tile_size=(128, 4, 8))
    dst_tiles[0, 0, 0].store(tile.data)
    return dst


@nki.jit
def bsh_to_h0_bs_h1(src_hbm):
    """(B, S, H) -> (H0, BxS, H1) via reshape_dim + permute + flatten_dims chain."""
    # src_hbm: [4, 32, 128]  ->  dst: [8, 4*32*16]
    B, S, H = 4, 32, 128
    H0, H1 = 8, 16
    dst = nl.ndarray((H0, B * S * H1), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(B, S, H))
    transformed = (
        src_tiles[0, 0, 0]
        .reshape_dim(2, [H0, H1])  # (4, 32, 8, 16)
        .permute([2, 0, 1, 3])  # (8, 4, 32, 16)
        .flatten_dims(1, 2)  # (8, 128, 16)
    )
    tile = transformed.load()  # tile: [8, 128, 16]

    dst_tiles = nt.tiles(dst, tile_size=(H0, B * S * H1))
    dst_tiles[0, 0].store(tile.data.reshape((H0, B * S * H1)))
    return dst


@nki.jit
def attention_qk_layout(q_hbm):
    """Attention Q layout: (B, S, H) -> (head_dim, B*S) for the first head."""
    # q_hbm: [4, 32, 128]  ->  dst: [16, 128]
    B, S, H = 4, 32, 128
    num_heads, head_dim = 8, 16
    dst = nl.ndarray((head_dim, B * S), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    q_tiles = nt.tiles(q_hbm, tile_size=(B, S, H))
    q_reshaped = (
        q_tiles[0, 0, 0]
        .reshape_dim(2, [num_heads, head_dim])  # (4, 32, 8, 16)
        .permute([3, 0, 1, 2])  # (16, 4, 32, 8)
        .flatten_dims(1, 2)  # (16, 128, 8)
    )
    q_head0 = q_reshaped.slice(2, 0, 1)  # (16, 128, 1)
    tile = q_head0.load()  # tile: [16, 128, 1]

    dst_tiles = nt.tiles(dst, tile_size=(head_dim, B * S))
    dst_tiles[0, 0].store(tile.data)
    return dst


@nki.jit
def tile_reshape_elementwise(src):
    """Reshape a loaded tile's free dim 512 -> (4, 128); apply per-chunk scale."""
    # src: [128, 512]  ->  dst: [128, 512]
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 512))
    dst_tiles = nt.tiles(dst, tile_size=(128, 512))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()  # tile: [128, 512]
            reshaped = tile.reshape_dim(1, [4, 128])  # [128, 4, 128]
            for ci in range(4):
                chunk = reshaped[:, ci, :]  # [128, 1, 128]
                nisa.tensor_scalar(chunk.data, chunk.data, nl.multiply, 2.0)
            dst_tiles[i, j].store(tile.data)

    return dst


@nki.jit
def block_reshape_elementwise(src):
    """Reshape per-tile free dim inside a loaded block, applying a per-chunk scale."""
    # src: [256, 512]  ->  dst: [256, 512]
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(128, 512), block_size=(2, 1))
    dst_blocks = nt.blocks(dst, tile_size=(128, 512), block_size=(2, 1))

    for bi in range(src_blocks.shape[0]):
        for bj in range(src_blocks.shape[1]):
            block = src_blocks[bi, bj].load()  # block: [2*128, 512]
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    tile_view = block[ti, tj]  # [128, 512]
                    reshaped = tile_view.reshape_dim(1, [4, 128])
                    for ci in range(4):
                        chunk = reshaped[:, ci, :]
                        nisa.tensor_scalar(chunk.data, chunk.data, nl.multiply, 2.0)
            dst_blocks[bi, bj].store(block.data)

    return dst


@nki.jit
def block_reshape_chunked(src):
    """Split the loaded tile's free dim into 2 chunks; add a constant per chunk."""
    # src: [256, 512]  ->  dst: [256, 512]
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_blocks = nt.blocks(src, tile_size=(128, 512), block_size=(2, 1))
    dst_blocks = nt.blocks(dst, tile_size=(128, 512), block_size=(2, 1))

    for bi in range(src_blocks.shape[0]):
        for bj in range(src_blocks.shape[1]):
            block = src_blocks[bi, bj].load()
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    tile_view = block[ti, tj]
                    chunked = tile_view.split(1, 2)  # split free dim into 2 of 256
                    for ci in range(2):
                        chunk = chunked[:, ci, :]
                        nisa.tensor_scalar(chunk.data, chunk.data, nl.add, 1.0)
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


def test_norm_reshape_transpose():
    torch.manual_seed(42)
    src = torch.randn(4, 1024, dtype=torch.float32)
    result = norm_reshape_transpose(to_device(src))

    H0, H1, BxS = 128, 8, 4
    expected = torch.zeros((H0, BxS * H1), dtype=torch.float32)
    for h0 in range(H0):
        for p in range(BxS * H1):
            addr = p * 128 + h0
            expected[h0, p] = src[addr // 1024, addr % 1024]
    assert torch.allclose(to_cpu(result), expected, rtol=1e-5, atol=1e-5)
    print("norm_reshape_transpose: PASSED")


def test_reshape_permute_load():
    torch.manual_seed(42)
    src = torch.randn(4, 1024, dtype=torch.float32)
    result = reshape_permute_load(to_device(src))

    expected = torch.zeros((128, 4, 8), dtype=torch.float32)
    for h0 in range(128):
        for bxs in range(4):
            for h1 in range(8):
                addr = h0 + bxs * 1024 + h1 * 128
                expected[h0, bxs, h1] = src[addr // 1024, addr % 1024]
    assert torch.allclose(to_cpu(result), expected, rtol=1e-5, atol=1e-5)
    print("reshape_permute_load: PASSED")


def test_bsh_to_h0_bs_h1():
    torch.manual_seed(42)
    B, S, H = 4, 32, 128
    H0, H1 = 8, 16
    src = torch.randn(B, S, H, dtype=torch.float32)
    result = bsh_to_h0_bs_h1(to_device(src))
    result_3d = to_cpu(result).numpy().reshape(H0, B * S, H1)
    expected = src.numpy().reshape(B, S, H0, H1).transpose(2, 0, 1, 3).reshape(H0, B * S, H1)
    np.testing.assert_allclose(result_3d, expected, rtol=1e-5, atol=1e-5)
    print("bsh_to_h0_bs_h1: PASSED")


def test_attention_qk_layout():
    torch.manual_seed(42)
    B, S, H = 4, 32, 128
    num_heads, head_dim = 8, 16
    q = torch.randn(B, S, H, dtype=torch.float32)
    result = attention_qk_layout(to_device(q))

    q_np = q.numpy().reshape(B, S, num_heads, head_dim)
    q_perm = q_np.transpose(3, 0, 1, 2).reshape(head_dim, B * S, num_heads)
    expected = q_perm[:, :, 0]
    np.testing.assert_allclose(to_cpu(result).numpy(), expected, rtol=1e-5, atol=1e-5)
    print("attention_qk_layout: PASSED")


def test_tile_reshape_elementwise():
    torch.manual_seed(42)
    src = torch.randn(128, 512, dtype=torch.bfloat16)
    result = tile_reshape_elementwise(to_device(src))
    expected = src * 2.0
    assert torch.allclose(
        to_cpu(result).to(torch.float32),
        expected.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print("tile_reshape_elementwise: PASSED")


def test_block_reshape_elementwise():
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    result = block_reshape_elementwise(to_device(src))
    expected = src * 2.0
    assert torch.allclose(
        to_cpu(result).to(torch.float32),
        expected.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print("block_reshape_elementwise: PASSED")


def test_block_reshape_chunked():
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    result = block_reshape_chunked(to_device(src))
    expected = src + 1.0
    assert torch.allclose(
        to_cpu(result).to(torch.float32),
        expected.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print("block_reshape_chunked: PASSED")


def main():
    test_norm_reshape_transpose()
    test_reshape_permute_load()
    test_bsh_to_h0_bs_h1()
    test_attention_qk_layout()
    test_tile_reshape_elementwise()
    test_block_reshape_elementwise()
    test_block_reshape_chunked()


if __name__ == "__main__":
    main()
