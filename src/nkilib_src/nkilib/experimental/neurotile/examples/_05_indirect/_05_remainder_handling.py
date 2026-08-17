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
Remainder handling for indirect indexing: oob_mode.skip suppresses
out-of-bounds DMA faults; oob_value pre-fills the SBUF slot so OOB
positions land on a deterministic value. Pair with view.is_remainder
to apply OOB protection only when needed.
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
def indirect_gather_oob(data, indices):
    """Gather with oob_mode.skip -- OOB indices silently skip the DMA."""
    # data: [N, D],  indices: [K, 1]  ->  output: [K, D]
    N = data.shape[0]
    D = data.shape[1]
    K = indices.shape[0]
    out = nl.ndarray((K, D), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, D))
    idx_iter = nt.tiles(indices, tile_size=(K, 1))
    out_iter = nt.tiles(out, tile_size=(K, D))

    idx_tile = idx_iter[0, 0].load()
    gathered = data_iter[idx_tile, 0].load(oob_mode=nisa.oob_mode.skip)
    out_iter[0, 0].store(gathered.data)
    return out


@nki.jit
def indirect_gather_oob_value(data, indices):
    """Gather with oob_value=0.0 -- OOB rows pre-filled with zero."""
    # data: [N, D],  indices: [K, 1]  ->  output: [K, D]   (OOB rows are 0)
    N = data.shape[0]
    D = data.shape[1]
    K = indices.shape[0]
    out = nl.ndarray((K, D), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, D))
    idx_iter = nt.tiles(indices, tile_size=(K, 1))
    out_iter = nt.tiles(out, tile_size=(K, D))

    idx_tile = idx_iter[0, 0].load()
    gathered = data_iter[idx_tile, 0].load(
        oob_mode=nisa.oob_mode.skip,
        oob_value=0.0,
    )
    out_iter[0, 0].store(gathered.data)
    return out


@nki.jit
def indirect_scatter_oob(source, indices, out_rows):
    """Scatter with oob_mode.skip -- OOB writes silently dropped."""
    # source: [K, D],  indices: [K, 1]  ->  output: [out_rows, D]
    K = source.shape[0]
    D = source.shape[1]
    out = nl.ndarray((out_rows, D), dtype=source.dtype, buffer=nl.shared_hbm)

    # Zero-init output.
    zero_iter = nt.tiles(out, tile_size=(out_rows, D))
    zero_sbuf = nl.ndarray((out_rows, D), dtype=source.dtype, buffer=nl.sbuf)
    nisa.memset(zero_sbuf, 0.0)
    zero_iter[0, 0].store(zero_sbuf)

    src_iter = nt.tiles(source, tile_size=(K, D))
    src_tile = src_iter[0, 0].load()
    idx_iter = nt.tiles(indices, tile_size=(K, 1))
    idx_tile = idx_iter[0, 0].load()

    out_iter = nt.tiles(out, tile_size=(out_rows, D))
    out_iter[idx_tile, 0].store(src_tile.data, oob_mode=nisa.oob_mode.skip)
    return out


@nki.jit
def is_remainder_guard(data, indices):
    """Guard OOB protection on view.is_remainder: indirect views report
    is_remainder=True since the indices are unknown at compile time."""
    # data: [N, D],  indices: [K, 1]  ->  output: [K, D]
    N = data.shape[0]
    D = data.shape[1]
    K = indices.shape[0]
    out = nl.ndarray((K, D), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, D))
    idx_iter = nt.tiles(indices, tile_size=(K, 1))
    out_iter = nt.tiles(out, tile_size=(K, D))

    idx_tile = idx_iter[0, 0].load()
    indirect_view = data_iter[idx_tile, 0]

    if indirect_view.is_remainder:
        gathered = indirect_view.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
    else:
        gathered = indirect_view.load()

    out_iter[0, 0].store(gathered.data)
    return out


@nki.jit
def multi_tile_gather_oob(data, indices):
    """Multi-tile gather with per-tile is_remainder guard."""
    # data: [N, D],  indices: [K, 1]  ->  output: [K, D]
    N = data.shape[0]
    D = data.shape[1]
    K = indices.shape[0]
    T = 16
    out = nl.ndarray((K, D), dtype=data.dtype, buffer=nl.shared_hbm)

    data_iter = nt.tiles(data, tile_size=(N, D))
    idx_iter = nt.tiles(indices, tile_size=(T, 1))
    out_iter = nt.tiles(out, tile_size=(T, D))

    for i in range(idx_iter.shape[0]):
        idx_tile = idx_iter[i, 0].load()
        view = data_iter[idx_tile, 0]
        if view.is_remainder:
            gathered = view.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
        else:
            gathered = view.load()
        out_iter[i, 0].store(gathered.data)

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


def test_indirect_gather_oob():
    torch.manual_seed(42)
    N, D, K = 128, 64, 16
    data = torch.randn(N, D, dtype=torch.bfloat16)
    indices = torch.arange(K, dtype=torch.int32).reshape(K, 1)
    result = indirect_gather_oob(to_device(data), to_device(indices))
    expected = data[:K, :]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("indirect_gather_oob: PASSED")


def test_indirect_gather_oob_value():
    torch.manual_seed(43)
    N, D, K = 128, 64, 16
    data = torch.randn(N, D, dtype=torch.bfloat16)
    indices = torch.arange(0, K * 5, 5, dtype=torch.int32).reshape(K, 1)
    result = indirect_gather_oob_value(to_device(data), to_device(indices))
    expected = data[indices.flatten().long(), :]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("indirect_gather_oob_value: PASSED")


def test_indirect_scatter_oob():
    torch.manual_seed(44)
    N, D, K = 128, 64, 8
    source = torch.randn(K, D, dtype=torch.bfloat16)
    indices = torch.tensor([3, 0, 7, 2, 100, 50, 33, 120], dtype=torch.int32).reshape(K, 1)
    result = indirect_scatter_oob(to_device(source), to_device(indices), N)
    expected = torch.zeros(N, D, dtype=torch.float32)
    for i in range(K):
        expected[indices[i, 0], :] = source[i, :].to(torch.float32)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected)
    print("indirect_scatter_oob: PASSED")


def test_is_remainder_guard():
    torch.manual_seed(45)
    N, D, K = 128, 64, 16
    data = torch.randn(N, D, dtype=torch.bfloat16)
    indices = torch.arange(K, dtype=torch.int32).reshape(K, 1)
    result = is_remainder_guard(to_device(data), to_device(indices))
    expected = data[:K, :]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("is_remainder_guard: PASSED")


def test_multi_tile_gather_oob():
    torch.manual_seed(46)
    N, D, K = 128, 64, 64
    data = torch.randn(N, D, dtype=torch.bfloat16)
    indices = torch.randperm(N)[:K].to(torch.int32).reshape(K, 1)
    result = multi_tile_gather_oob(to_device(data), to_device(indices))
    expected = data[indices.flatten().long(), :]
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("multi_tile_gather_oob: PASSED")


def main():
    test_indirect_gather_oob()
    test_indirect_gather_oob_value()
    test_indirect_scatter_oob()
    test_is_remainder_guard()
    test_multi_tile_gather_oob()


if __name__ == "__main__":
    main()
