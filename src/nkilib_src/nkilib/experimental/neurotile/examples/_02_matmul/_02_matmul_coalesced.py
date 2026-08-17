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
Blocked matmul with coalesced block load/store via nt.blocks(): each
block-load DMAs BLOCK_K x BLOCK_M (or BLOCK_K x BLOCK_N) tiles in a
single coalesced transfer; per-tile matmul accumulates into an
nt.alloc_tiles output block; one coalesced store per output block.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt


@nki.jit
def matmul_coalesced(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=4,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=4,
):
    """Blocked matmul: C = lhsT.T @ rhs with coalesced block load/store."""
    # lhsT: [K, M],  rhs: [K, N]  ->  C: [M, N]
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    K, M = lhsT.shape
    _, N = rhs.shape
    C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    lhsT_blocks = nt.blocks(
        lhsT,
        tile_size=(TILE_K, TILE_M),
        block_size=(TILES_IN_BLOCK_K, TILES_IN_BLOCK_M),
    )
    rhs_blocks = nt.blocks(
        rhs,
        tile_size=(TILE_K, TILE_N),
        block_size=(TILES_IN_BLOCK_K, TILES_IN_BLOCK_N),
    )
    out_blocks = nt.blocks(
        C,
        tile_size=(TILE_M, TILE_N),
        block_size=(M // TILE_M, TILES_IN_BLOCK_N),
    )

    for nb in range(rhs_blocks.shape[1]):
        # Pre-allocate one output block of partials.
        acc = nt.alloc_tiles(
            tile_size=(TILE_M, TILE_N),
            grid=(M // TILE_M, TILES_IN_BLOCK_N),
            buffer_type=nl.sbuf,
            dtype=lhsT.dtype,
        )
        nisa.memset(acc.data, 0.0)

        for kb in nl.sequential_range(rhs_blocks.shape[0]):
            rhs_block = rhs_blocks[kb, nb].load()  # block: [BLOCK_K*128, BLOCK_N*512]

            for mb in range(lhsT_blocks.shape[1]):
                lhsT_block = lhsT_blocks[kb, mb].load()  # block: [BLOCK_K*128, BLOCK_M*128]

                for bm in range(TILES_IN_BLOCK_M):
                    for bn in range(TILES_IN_BLOCK_N):
                        psum = nl.ndarray(
                            (TILE_M, TILE_N),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        for bk in range(TILES_IN_BLOCK_K):
                            nisa.nc_matmul(
                                dst=psum,
                                stationary=lhsT_block[bk, bm].data,
                                moving=rhs_block[bk, bn].data,
                            )
                        acc_tile = acc[mb * TILES_IN_BLOCK_M + bm, bn].data
                        nisa.tensor_tensor(
                            dst=acc_tile,
                            data1=acc_tile,
                            data2=psum,
                            op=nl.add,
                        )

        # Single coalesced store back to HBM.
        out_blocks[0, nb].store(acc.data)

    return C


@nki.jit
def matmul_coalesced_streamed(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=2,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=4,
):
    """Blocked matmul with K-streamed weights and remainder-tile support.

    Like ``matmul_coalesced`` but streams each N-block's K-blocks through
    rotating SBUF buffers (next K-block's DMA overlaps the current matmul), and
    handles M/N not a multiple of the block span -- a partial trailing block is
    sized from its clamped ``element_shape`` so loops and store touch only real
    data.
    """
    # lhsT: [K, M],  rhs: [K, N]  ->  C: [M, N]
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    K, M = lhsT.shape
    _, N = rhs.shape
    C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    lhsT_blocks = nt.blocks(lhsT, tile_size=(TILE_K, TILE_M), block_size=(TILES_IN_BLOCK_K, TILES_IN_BLOCK_M))
    rhs_blocks = nt.blocks(rhs, tile_size=(TILE_K, TILE_N), block_size=(TILES_IN_BLOCK_K, TILES_IN_BLOCK_N))
    out_blocks = nt.blocks(C, tile_size=(TILE_M, TILE_N), block_size=(TILES_IN_BLOCK_M, TILES_IN_BLOCK_N))

    for m_blk in range(lhsT_blocks.shape[1]):
        # Load this M-block's full K-column once, then descend to its tile grid.
        lhsT_tiles = nt.tiles(lhsT_blocks[:, m_blk].load())
        for n_blk in range(rhs_blocks.shape[1]):
            out_block = out_blocks[m_blk, n_blk]
            psums = nt.psum_pool(tile_size=(TILE_M, TILE_N), element_shape=out_block.element_shape)
            acc = nt.alloc_tiles(
                tile_size=(TILE_M, TILE_N),
                element_shape=out_block.element_shape,
                buffer_type=nl.sbuf,
                dtype=lhsT.dtype,
            )
            # element_shape= clamps a partial trailing block, so the output tile
            # grid is the real per-block count (fewer tiles / a partial last one).
            m_tiles, n_tiles = acc.shape

            # Stream this N-block's K-blocks through 2 rotating SBUF buffers so
            # the DMA of K-block k+1 overlaps the matmul on K-block k.
            rhs_k_stream = rhs_blocks[:, n_blk].stream(buffer_count=2)
            for k_blk in nl.affine_range(rhs_blocks.shape[0]):
                rhs_tiles = nt.tiles(rhs_k_stream.load(k_blk))
                for bm in range(m_tiles):
                    for bn in range(n_tiles):
                        for bk in range(rhs_tiles.shape[0]):
                            nisa.nc_matmul(
                                dst=psums[bm, bn].data,
                                stationary=lhsT_tiles[k_blk * TILES_IN_BLOCK_K + bk, bm].data,
                                moving=rhs_tiles[bk, bn].data,
                            )

            # Evict the full-K PSUM accumulation to SBUF, then store the whole
            # block back to HBM in one coalesced DMA.
            for bm in range(m_tiles):
                for bn in range(n_tiles):
                    nisa.tensor_copy(acc[bm, bn].data, psums[bm, bn].data)
            out_block.store(acc.data)

    return C


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


def test_matmul_coalesced():
    """Single block per dimension."""
    torch.manual_seed(42)
    M, K, N = 512, 512, 1024
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    result = matmul_coalesced(to_device(lhs.T.contiguous()), to_device(rhs))
    expected = lhs @ rhs
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-2, atol=1e-2)
    print("matmul_coalesced: PASSED")


def test_matmul_coalesced_multi_block():
    """Multiple blocks per dimension."""
    torch.manual_seed(42)
    M, K, N = 512, 512, 1024
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    result = matmul_coalesced(
        to_device(lhs.T.contiguous()),
        to_device(rhs),
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=1,
        TILES_IN_BLOCK_K=2,
    )
    expected = lhs @ rhs
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-2, atol=1e-2)
    print("matmul_coalesced_multi_block: PASSED")


def test_matmul_coalesced_streamed():
    """K-streamed variant, block-aligned M and N."""
    torch.manual_seed(42)
    M, K, N = 512, 1024, 2048  # 2 M-blocks, 2 N-blocks, multiple K-blocks
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    result = matmul_coalesced_streamed(to_device(lhs.T.contiguous()), to_device(rhs))
    expected = lhs @ rhs
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=2e-2, atol=2.5)
    print("matmul_coalesced_streamed: PASSED")


def test_matmul_coalesced_streamed_remainder():
    """Remainder tiles: M and N not multiples of the block span.

    M=384 (block span 256) and N=1792 (block span 1024) exercise the
    partial-block clamp on a partition and a free dim at once.
    """
    torch.manual_seed(42)
    M, K, N = 384, 1024, 1792
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    result = matmul_coalesced_streamed(
        to_device(lhs.T.contiguous()),
        to_device(rhs),
        TILES_IN_BLOCK_M=2,
        TILES_IN_BLOCK_N=2,
        TILES_IN_BLOCK_K=4,
    )
    expected = lhs @ rhs
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=2e-2, atol=2.5)
    print("matmul_coalesced_streamed_remainder: PASSED")


def main():
    test_matmul_coalesced()
    test_matmul_coalesced_multi_block()
    test_matmul_coalesced_streamed()
    test_matmul_coalesced_streamed_remainder()


if __name__ == "__main__":
    main()
