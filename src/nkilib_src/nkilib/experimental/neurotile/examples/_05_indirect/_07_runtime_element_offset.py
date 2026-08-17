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
Runtime indexing patterns for dynamic loop grids.

The first kernel keeps the counter in logical tile coordinates, so
NeuroTile scales the SBUF scalar index internally. The second keeps the
counter in source-element units and marks it with nt.element_offset(...)
so NeuroTile passes it through without hidden scaling.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect._07_runtime_element_offset_torch import (
    dynamic_elementwise_add_logical_tile_index_torch_ref,
    dynamic_elementwise_add_torch_ref,
)

P_TILE = 128
H_TILE = 512


@nki.jit
def dynamic_elementwise_add_logical_tile_index(input_a, input_b, num_m_tiles):
    """Add [M, H] inputs with a runtime logical M-tile index."""
    m_static, hidden = input_a.shape
    h_tile_count = hidden // H_TILE
    out = nl.ndarray((m_static, hidden), dtype=input_a.dtype, buffer=nl.shared_hbm)

    a_tiles = nt.tiles(input_a, tile_size=(P_TILE, H_TILE))
    b_tiles = nt.tiles(input_b, tile_size=(P_TILE, H_TILE))
    out_tiles = nt.tiles(out, tile_size=(P_TILE, H_TILE))

    trip_count_tile = nt.tiles(num_m_tiles, tile_size=(1, 1))[0, 0].load()
    trip_count_reg = nisa.register_alloc()
    nisa.register_load(trip_count_reg, trip_count_tile.data)

    m_tile_idx = nt.alloc_tiles(
        tile_size=(1, 1),
        grid=(1, 1),
        buffer_type=nl.sbuf,
        dtype=nl.int32,
    )
    nisa.memset(dst=m_tile_idx.data, value=0)

    def body(_):
        for h in nl.affine_range(h_tile_count):
            a_tile = a_tiles[m_tile_idx, h].load()
            b_tile = b_tiles[m_tile_idx, h].load()
            nisa.tensor_tensor(dst=a_tile.data, data1=a_tile.data, data2=b_tile.data, op=nl.add)
            out_tiles[m_tile_idx, h].store(a_tile.data)

        nisa.tensor_scalar(
            dst=m_tile_idx.data,
            data=m_tile_idx.data,
            op0=nl.add,
            operand0=1,
        )

    nl.fori_loop(0, trip_count_reg, body, step=1)
    return out


@nki.jit
def dynamic_elementwise_add(input_a, input_b, num_m_tiles):
    """Add [M, H] inputs with a reusable runtime M element offset."""
    m_static, hidden = input_a.shape
    h_tile_count = hidden // H_TILE
    out = nl.ndarray((m_static, hidden), dtype=input_a.dtype, buffer=nl.shared_hbm)

    a_tiles = nt.tiles(input_a, tile_size=(P_TILE, H_TILE))
    b_tiles = nt.tiles(input_b, tile_size=(P_TILE, H_TILE))
    out_tiles = nt.tiles(out, tile_size=(P_TILE, H_TILE))

    trip_count_tile = nt.tiles(num_m_tiles, tile_size=(1, 1))[0, 0].load()
    trip_count_reg = nisa.register_alloc()
    nisa.register_load(trip_count_reg, trip_count_tile.data)

    m_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=m_offset, value=0)

    def body(_):
        for h in nl.affine_range(h_tile_count):
            a_tile = a_tiles[nt.element_offset(m_offset), h].load()
            b_tile = b_tiles[nt.element_offset(m_offset), h].load()
            nisa.tensor_tensor(dst=a_tile.data, data1=a_tile.data, data2=b_tile.data, op=nl.add)
            out_tiles[nt.element_offset(m_offset), h].store(a_tile.data)

        nisa.tensor_scalar(
            dst=m_offset,
            data=m_offset,
            op0=nl.add,
            operand0=a_tiles.index_stride_elements[0],
        )

    nl.fori_loop(0, trip_count_reg, body, step=1)
    return out


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


def test_dynamic_elementwise_add():
    torch.manual_seed(70)
    m_dim, h_dim = 384, 1024
    input_a = torch.randn(m_dim, h_dim, dtype=torch.bfloat16)
    input_b = torch.randn(m_dim, h_dim, dtype=torch.bfloat16)
    num_m_tiles = torch.tensor([[m_dim // P_TILE]], dtype=torch.int32)

    result = dynamic_elementwise_add(to_device(input_a), to_device(input_b), to_device(num_m_tiles))
    expected = dynamic_elementwise_add_torch_ref(input_a, input_b, num_m_tiles)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("dynamic_elementwise_add runtime element offset: PASSED")


def test_dynamic_elementwise_add_logical_tile_index():
    torch.manual_seed(71)
    m_dim, h_dim = 384, 1024
    input_a = torch.randn(m_dim, h_dim, dtype=torch.bfloat16)
    input_b = torch.randn(m_dim, h_dim, dtype=torch.bfloat16)
    num_m_tiles = torch.tensor([[m_dim // P_TILE]], dtype=torch.int32)

    result = dynamic_elementwise_add_logical_tile_index(to_device(input_a), to_device(input_b), to_device(num_m_tiles))
    expected = dynamic_elementwise_add_logical_tile_index_torch_ref(input_a, input_b, num_m_tiles)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("dynamic_elementwise_add runtime logical tile index: PASSED")


def main():
    test_dynamic_elementwise_add()
    test_dynamic_elementwise_add_logical_tile_index()


if __name__ == "__main__":
    main()
