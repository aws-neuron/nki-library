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
Source-layout access patterns: nt.tiles(access_pattern=) for non-contiguous /
strided source layouts. The AP describes how to walk the source tensor;
tile_size= is orthogonal and specifies the per-iteration tile granularity.
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
def contiguous_access_pattern(src):
    """access_pattern= equivalent to default contiguous layout (strides
    derived from tile_size=). Same output, explicit AP."""
    M, N = src.shape
    dst = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)

    TILE_M, TILE_N = 128, 512
    src_tiles = nt.tiles(src, access_pattern=[[N, M], [1, N]], tile_size=(TILE_M, TILE_N))
    dst_tiles = nt.tiles(dst, access_pattern=[[N, M], [1, N]], tile_size=(TILE_M, TILE_N))

    for i in range(src_tiles.shape[0]):
        for j in range(src_tiles.shape[1]):
            tile = src_tiles[i, j].load()
            dst_tiles[i, j].store(tile.data)

    return dst


@nki.jit
def strided_access_pattern(src):
    """Strided AP -- partition stride = 2*N skips every other source row,
    so only the even rows are visible to the tile grid."""
    M, N = src.shape
    TILE_M = 64

    num_out_rows = M // 2
    dst = nl.ndarray((num_out_rows, N), dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(
        src,
        access_pattern=[[2 * N, M // 2], [1, N]],
        tile_size=(TILE_M, N),
    )

    for i in range(src_tiles.shape[0]):
        tile = src_tiles[i, 0].load()
        nisa.dma_copy(dst[nl.ds(i * TILE_M, TILE_M), 0:N], tile.data)

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


def test_contiguous_access_pattern():
    torch.manual_seed(42)
    src = torch.randn(256, 512, dtype=torch.bfloat16)
    result = contiguous_access_pattern(to_device(src))
    assert torch.allclose(to_cpu(result), src, rtol=1e-2, atol=1e-2)
    print("contiguous_access_pattern: PASSED")


def test_strided_access_pattern():
    torch.manual_seed(42)
    src = torch.randn(512, 512, dtype=torch.bfloat16)
    result = strided_access_pattern(to_device(src))
    expected = src[::2, :].float()
    assert torch.allclose(to_cpu(result).float(), expected, rtol=1e-2, atol=1e-2)
    print("strided_access_pattern: PASSED")


def main():
    test_contiguous_access_pattern()
    test_strided_access_pattern()


if __name__ == "__main__":
    main()
