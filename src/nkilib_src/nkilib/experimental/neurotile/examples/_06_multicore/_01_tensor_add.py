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
Multi-core tensor addition with block and interleaved sharding patterns.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

TILE_P, TILE_F = 128, 512


# ============================================================================
# Kernels
# ============================================================================


@nki.jit
def tensor_add_single_core(a, b):
    """Element-wise tensor add (single core baseline)."""
    # a, b: [M, N]  ->  c: [M, N]
    M, N = a.shape
    c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    A_tiles = nt.tiles(a, tile_size=(TILE_P, TILE_F))
    B_tiles = nt.tiles(b, tile_size=(TILE_P, TILE_F))
    C_tiles = nt.tiles(c, tile_size=(TILE_P, TILE_F))

    for i in range(A_tiles.shape[0]):
        for j in range(A_tiles.shape[1]):
            a_tile = A_tiles[i, j].load()  # tile: [128, 512]
            b_tile = B_tiles[i, j].load()
            nisa.tensor_tensor(a_tile.data, a_tile.data, b_tile.data, op=nl.add)
            C_tiles[i, j].store(a_tile.data)

    return c


@nki.jit
def tensor_add_block_sharded(a, b):
    """Block-sharded tensor add -- each core owns a contiguous half of the
    M-tiles."""
    # a, b: [M, N]  ->  c: [M, N]   (M tiles split contiguously across cores)
    M, N = a.shape
    c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    # Slice helper -- built in the body (module-level slice constants don't trace).
    r = nt.block_range(
        rank=nl.program_id(0),
        num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, TILE_P),
    )

    A_tiles = nt.tiles(a, tile_size=(TILE_P, TILE_F))[r, :]
    B_tiles = nt.tiles(b, tile_size=(TILE_P, TILE_F))[r, :]
    C_tiles = nt.tiles(c, tile_size=(TILE_P, TILE_F))[r, :]

    # i, j are LOCAL indices on the sharded view -- the slice handles offsets.
    for i in range(A_tiles.shape[0]):
        for j in range(A_tiles.shape[1]):
            a_tile = A_tiles[i, j].load()
            b_tile = B_tiles[i, j].load()
            nisa.tensor_tensor(a_tile.data, a_tile.data, b_tile.data, op=nl.add)
            C_tiles[i, j].store(a_tile.data)

    return c


@nki.jit
def tensor_add_interleaved(a, b):
    """Interleaved (round-robin) tensor add -- each core owns every Nth tile."""
    # a, b: [M, N]  ->  c: [M, N]   (M tiles round-robin across cores)
    M, N = a.shape
    c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    r = nt.interleaved_range(
        rank=nl.program_id(0),
        num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, TILE_P),
    )

    A_tiles = nt.tiles(a, tile_size=(TILE_P, TILE_F))[r, :]
    B_tiles = nt.tiles(b, tile_size=(TILE_P, TILE_F))[r, :]
    C_tiles = nt.tiles(c, tile_size=(TILE_P, TILE_F))[r, :]

    for i in range(A_tiles.shape[0]):
        for j in range(A_tiles.shape[1]):
            a_tile = A_tiles[i, j].load()
            b_tile = B_tiles[i, j].load()
            nisa.tensor_tensor(a_tile.data, a_tile.data, b_tile.data, op=nl.add)
            C_tiles[i, j].store(a_tile.data)

    return c


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


def test_single_core():
    torch.manual_seed(42)
    M, N = 512, 1024
    a = torch.rand(M, N, dtype=torch.bfloat16)
    b = torch.rand(M, N, dtype=torch.bfloat16)
    result = tensor_add_single_core(to_device(a), to_device(b))
    assert torch.allclose(to_cpu(result), a + b, rtol=1e-3, atol=1e-4)
    print("tensor_add_single_core: PASSED")


def test_block_sharded_lnc2():
    torch.manual_seed(42)
    M, N = 512, 1024
    a = torch.rand(M, N, dtype=torch.bfloat16)
    b = torch.rand(M, N, dtype=torch.bfloat16)
    result = tensor_add_block_sharded[2](to_device(a), to_device(b))
    assert torch.allclose(to_cpu(result), a + b, rtol=1e-3, atol=1e-4)
    print("tensor_add_block_sharded (LNC=2): PASSED")


def test_interleaved_lnc2():
    torch.manual_seed(42)
    M, N = 512, 1024
    a = torch.rand(M, N, dtype=torch.bfloat16)
    b = torch.rand(M, N, dtype=torch.bfloat16)
    result = tensor_add_interleaved[2](to_device(a), to_device(b))
    assert torch.allclose(to_cpu(result), a + b, rtol=1e-3, atol=1e-4)
    print("tensor_add_interleaved (LNC=2): PASSED")


def main():
    test_single_core()
    test_block_sharded_lnc2()
    test_interleaved_lnc2()


if __name__ == "__main__":
    main()
