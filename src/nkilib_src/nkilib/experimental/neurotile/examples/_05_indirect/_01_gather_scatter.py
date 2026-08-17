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
Indirect gather / scatter via vector-offset indexing. Pass an SBUF tile
of indices as the index key; the DMA hardware does the gather/scatter.
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
def gather_kernel(data, indices):
    """Gather K rows from data[N, D] using indices[K, 1].
    output[i] = data[indices[i, 0]]. Loops over both K-row and D-col tiles."""
    # data: [N, D],  indices: [K, 1]  ->  output: [K, D]
    N = data.shape[0]
    D = data.shape[1]
    K = indices.shape[0]
    T_K, T_D = 16, 32
    out = nl.ndarray((K, D), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, T_D))  # full N rows, T_D col-tiles
    idx_iter = nt.tiles(indices, tile_size=(T_K, 1))
    out_iter = nt.tiles(out, tile_size=(T_K, T_D))

    for i in range(idx_iter.shape[0]):  # K // T_K
        idx_tile = idx_iter[i, 0].load()  # tile: [T_K, 1]
        for j in range(out_iter.shape[1]):  # D // T_D
            gathered = data_iter[idx_tile, j].load()  # tile: [T_K, T_D]
            out_iter[i, j].store(gathered.data)

    return out


@nki.jit
def scatter_kernel(source, indices, out_rows):
    """Scatter source[K, D] to dst[out_rows, D] using indices[K, 1].
    output[indices[i, 0]] = source[i]. Loops over both K-row and D-col tiles."""
    # source: [K, D],  indices: [K, 1]  ->  output: [out_rows, D]
    K = source.shape[0]
    D = source.shape[1]
    T_K, T_D = 16, 32
    out = nl.ndarray((out_rows, D), dtype=source.dtype, buffer=nl.shared_hbm)

    # Zero-initialize output -- single coalesced store.
    zero_iter = nt.tiles(out, tile_size=(out_rows, D))
    zero_sbuf = nl.ndarray((out_rows, D), dtype=source.dtype, buffer=nl.sbuf)
    nisa.memset(zero_sbuf, 0.0)
    zero_iter[0, 0].store(zero_sbuf)

    src_iter = nt.tiles(source, tile_size=(T_K, T_D))
    idx_iter = nt.tiles(indices, tile_size=(T_K, 1))
    out_iter = nt.tiles(out, tile_size=(out_rows, T_D))

    for i in range(src_iter.shape[0]):  # K // T_K
        idx_tile = idx_iter[i, 0].load()  # tile: [T_K, 1]
        for j in range(src_iter.shape[1]):  # D // T_D
            src_tile = src_iter[i, j].load()  # tile: [T_K, T_D]
            out_iter[idx_tile, j].store(src_tile.data)  # vector_offset scatter

    return out


@nki.jit
def scalar_gather_dim0(data, row_idx_tensor):
    """Select a single row dynamically via scalar_offset on dim 0."""
    # data: [N, D],  row_idx_tensor: [1, 1]  ->  output: [1, D]
    N = data.shape[0]
    D = data.shape[1]
    out = nl.ndarray((1, D), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(1, D))
    idx_iter = nt.tiles(row_idx_tensor, tile_size=(1, 1))
    out_iter = nt.tiles(out, tile_size=(1, D))

    idx_tile = idx_iter[0, 0].load()
    selected = data_iter[idx_tile, 0].load()  # tile: [1, D]
    out_iter[0, 0].store(selected.data)
    return out


@nki.jit
def scalar_gather_dim1(data, col_idx_tensor):
    """Select a single column dynamically via scalar_offset on dim 1."""
    # data: [N, D],  col_idx_tensor: [1, 1]  ->  output: [N, 1]
    N = data.shape[0]
    D = data.shape[1]
    out = nl.ndarray((N, 1), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, 1))
    idx_iter = nt.tiles(col_idx_tensor, tile_size=(1, 1))
    out_iter = nt.tiles(out, tile_size=(N, 1))

    idx_tile = idx_iter[0, 0].load()
    selected = data_iter[0, idx_tile].load()  # tile: [N, 1]
    out_iter[0, 0].store(selected.data)
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


def test_gather():
    torch.manual_seed(42)
    N, D, K = 128, 256, 64  # K=64=4*T_K, D=256=8*T_D
    data = torch.randn(N, D, dtype=torch.bfloat16)
    indices = torch.randperm(N)[:K].to(torch.int32).reshape(K, 1)
    result = gather_kernel(to_device(data), to_device(indices))
    expected = data[indices.flatten(), :]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("gather: PASSED")


def test_scatter():
    torch.manual_seed(43)
    N, D, K = 128, 256, 64
    source = torch.randn(K, D, dtype=torch.bfloat16)
    indices = torch.randperm(N)[:K].to(torch.int32).reshape(K, 1)
    result = scatter_kernel(to_device(source), to_device(indices), N)
    expected = torch.zeros(N, D, dtype=torch.float32)
    for i in range(K):
        expected[indices[i, 0], :] = source[i, :].to(torch.float32)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected)
    print("scatter: PASSED")


def test_scalar_gather_dim0():
    torch.manual_seed(46)
    N, D = 128, 64
    data = torch.randn(N, D, dtype=torch.bfloat16)
    idx = torch.tensor([[42]], dtype=torch.int32)
    result = scalar_gather_dim0(to_device(data), to_device(idx))
    expected = data[42, :].reshape(1, D)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("scalar_gather_dim0: PASSED")


def test_scalar_gather_dim1():
    torch.manual_seed(47)
    N, D = 128, 64
    data = torch.randn(N, D, dtype=torch.bfloat16)
    idx = torch.tensor([[17]], dtype=torch.int32)
    result = scalar_gather_dim1(to_device(data), to_device(idx))
    expected = data[:, 17].reshape(N, 1)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("scalar_gather_dim1: PASSED")


def main():
    test_gather()
    test_scatter()
    test_scalar_gather_dim0()
    test_scalar_gather_dim1()


if __name__ == "__main__":
    main()
