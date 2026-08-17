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
KV cache access with dynamic sequence offset: combine static batch
indexing with runtime sequence-position selection on 3D tensors.
A common MoE / autoregressive pattern.
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
def kv_cache_load(kv_cache, batch_id, seq_offset_tensor):
    """Load one KV vector: kv_cache[B, S, D] -> output[1, D] at
    [batch_id, seq_pos, :]."""
    # kv_cache: [B, S, D],  batch_id: int,  seq_offset_tensor: [1, 1]
    #   ->  output: [1, D]
    B = kv_cache.shape[0]
    S = kv_cache.shape[1]
    D = kv_cache.shape[2]
    out = nl.ndarray((1, D), dtype=kv_cache.dtype, buffer=nl.shared_hbm)

    # Tile shape (1, D) -- one seq position on P, D head features on F.
    kv_iter = nt.tiles(kv_cache, tile_size=(1, D))

    seq_iter = nt.tiles(seq_offset_tensor, tile_size=(1, 1))
    seq_tile = seq_iter[0, 0].load()

    kv_data = kv_iter[batch_id, seq_tile].load()  # tile: [1, D]

    out_iter = nt.tiles(out, tile_size=(1, D))
    out_iter[0, 0].store(kv_data.data)
    return out


@nki.jit
def kv_cache_load_raw_sbuf_index(kv_cache, batch_id, seq_offset_tensor):
    """Same as kv_cache_load, but the index is a raw nl.ndarray in SBUF.
    Demonstrates that view[..., raw_sbuf_ndarray, ...] is accepted directly
    -- no nt.tiles/.load() wrapping needed when the index is already in SBUF."""
    # kv_cache: [B, S, D],  batch_id: int,  seq_offset_tensor: [1, 1]
    #   ->  output: [1, D]
    B = kv_cache.shape[0]
    S = kv_cache.shape[1]
    D = kv_cache.shape[2]
    out = nl.ndarray((1, D), dtype=kv_cache.dtype, buffer=nl.shared_hbm)

    # Allocate SBUF scratch for the index, DMA it in via raw nisa.dma_copy.
    seq_sbuf = nl.ndarray((1, 1), dtype=seq_offset_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(seq_sbuf, seq_offset_tensor)

    kv_iter = nt.tiles(kv_cache, tile_size=(1, D))
    kv_data = kv_iter[batch_id, seq_sbuf].load()  # raw SBUF ndarray as index

    out_iter = nt.tiles(out, tile_size=(1, D))
    out_iter[0, 0].store(kv_data.data)
    return out


@nki.jit
def kv_cache_multi_pos(kv_cache, batch_indices, seq_positions):
    """Gather K KV vectors at (batch, seq) pairs in a SINGLE coalesced
    indirect DMA via vector_offset (the canonical nkilib pattern -- see
    KaenaNeuronKernelLibrary block_kv_cache loads).

    Trick: vector_offset is restricted to indirect_dim=0, so we reshape
    [B, S, D] -> [B*S, D] (raw NKI tensor reshape, not a NeuroTile view --
    that produces a new tensor handle whose dim 0 stride matches what the
    vector indices need) and build flat indices = batch[k]*S + seq[k]
    in SBUF before issuing the gather."""
    # kv_cache: [B, S, D],  batch_indices: [K, 1],  seq_positions: [K, 1]
    #   ->  output: [K, D]
    B = kv_cache.shape[0]
    S = kv_cache.shape[1]
    D = kv_cache.shape[2]
    K = batch_indices.shape[0]
    out = nl.ndarray((K, D), dtype=kv_cache.dtype, buffer=nl.shared_hbm)

    # Reshape the raw NKI tensor (NEW handle with dim 0 = B*S) so vector_offset
    # on indirect_dim=0 walks the flat (batch, seq) space.
    kv_flat = kv_cache.reshape((B * S, D))
    flat_iter = nt.tiles(kv_flat, tile_size=(1, D))

    # Load both index streams into SBUF.
    b_tile = nt.tiles(batch_indices, tile_size=(K, 1))[0, 0].load()  # [K, 1]
    s_tile = nt.tiles(seq_positions, tile_size=(K, 1))[0, 0].load()  # [K, 1]

    # Compute flat index = batch[k]*S + seq[k] in SBUF.
    flat_idx = nl.ndarray((K, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(flat_idx, b_tile.data, nl.multiply, S)  # batch * S
    nisa.tensor_tensor(flat_idx, flat_idx, s_tile.data, nl.add)  # + seq

    # Single vector_offset DMA gathers K rows.
    kv_data = flat_iter[flat_idx, 0].load()  # tile: [K, D]

    out_iter = nt.tiles(out, tile_size=(K, D))
    out_iter[0, 0].store(kv_data.data)
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


def test_single_position():
    torch.manual_seed(60)
    B, S, D = 4, 128, 64
    kv = torch.randn(B, S, D, dtype=torch.bfloat16)
    seq_t = torch.tensor([[42]], dtype=torch.int32)
    result = kv_cache_load(to_device(kv), 2, to_device(seq_t))
    expected = kv[2, 42, :].reshape(1, D)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("kv_cache_load: PASSED")


def test_raw_sbuf_index():
    torch.manual_seed(62)
    B, S, D = 4, 128, 64
    kv = torch.randn(B, S, D, dtype=torch.bfloat16)
    seq_t = torch.tensor([[77]], dtype=torch.int32)
    result = kv_cache_load_raw_sbuf_index(to_device(kv), 3, to_device(seq_t))
    expected = kv[3, 77, :].reshape(1, D)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected.to(torch.float32))
    print("kv_cache_load_raw_sbuf_index: PASSED")


def test_multi_position():
    torch.manual_seed(61)
    B, S, D = 8, 256, 64
    kv = torch.randn(B, S, D, dtype=torch.bfloat16)
    batch_indices = torch.tensor([[1], [5]], dtype=torch.int32)
    seq_positions = torch.tensor([[100], [150]], dtype=torch.int32)
    K = batch_indices.shape[0]
    result = kv_cache_multi_pos(to_device(kv), to_device(batch_indices), to_device(seq_positions))
    expected = torch.zeros(K, D, dtype=torch.float32)
    for k in range(K):
        expected[k, :] = kv[batch_indices[k, 0], seq_positions[k, 0], :].to(torch.float32)
    torch.testing.assert_close(to_cpu(result).to(torch.float32), expected)
    print("kv_cache_multi_pos: PASSED")


def main():
    test_single_position()
    test_raw_sbuf_index()
    test_multi_position()


if __name__ == "__main__":
    main()
