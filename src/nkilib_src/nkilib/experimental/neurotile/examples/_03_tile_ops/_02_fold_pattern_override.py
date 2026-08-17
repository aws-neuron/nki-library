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
Fold (.fold) and pattern_override: dimension merging plus custom HBM
access patterns. .fold(src_dim, into_dim) folds src_dim into into_dim
either as the partition (into_dim=0) or a free axis (into_dim>0); both
flavors round-trip via load + store. .load(pattern_override=, out_shape=)
lets the caller supply a hand-built HBM AP for one DMA.
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
def fold_into_partition(src_hbm):
    """Fold last dim into the partition dim: (32, 512, 4) -> (128, 512)."""
    # src_hbm: [32, 512, 4]  ->  dst: [128, 512]
    P, F, K = 32, 512, 4
    dst = nl.ndarray((P * K, F), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(P, F, K))
    folded = src_tiles[0, 0, 0].fold(2, 0)  # fold dim 2 -> dim 0
    tile = folded.load()  # tile: [128, 512]

    dst_tiles = nt.tiles(dst, tile_size=(P * K, F))
    dst_tiles[0, 0].store(tile.data)
    return dst


@nki.jit
def fold_free_dim(src_hbm):
    """Fold last dim into a free dim: (32, 128, 4) -> (32, 512)."""
    # src_hbm: [32, 128, 4]  ->  dst: [32, 512]
    P, F, K = 32, 128, 4
    dst = nl.ndarray((P, F * K), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(P, F, K))
    folded = src_tiles[0, 0, 0].fold(2, 1)  # fold dim 2 -> dim 1
    tile = folded.load()  # tile: [32, 512]

    dst_tiles = nt.tiles(dst, tile_size=(P, F * K))
    dst_tiles[0, 0].store(tile.data)
    return dst


@nki.jit
def fold_partition_roundtrip(src_hbm):
    """fold(2, 0) load -> scale -> fold(2, 0) store roundtrip."""
    # src_hbm: [32, 128, 4]  ->  dst: [32, 128, 4]
    P, F, K = 32, 128, 4
    dst = nl.ndarray((P, F, K), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(P, F, K))
    folded = src_tiles[0, 0, 0].fold(2, 0)
    tile = folded.load()  # tile: [128, 128]

    result = nl.ndarray(tile.data.shape, dtype=tile.data.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(result, tile.data, nl.multiply, 2.0)

    dst_tiles = nt.tiles(dst, tile_size=(P, F, K))
    dst_folded = dst_tiles[0, 0, 0].fold(2, 0)
    dst_folded.store(result)
    return dst


@nki.jit
def fold_free_dim_roundtrip(src_hbm):
    """fold(2, 1) load -> scale -> fold(2, 1) store roundtrip."""
    # src_hbm: [32, 128, 4]  ->  dst: [32, 128, 4]
    P, F, K = 32, 128, 4
    dst = nl.ndarray((P, F, K), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(P, F, K))
    folded = src_tiles[0, 0, 0].fold(2, 1)
    tile = folded.load()  # tile: [32, 512]

    result = nl.ndarray(tile.data.shape, dtype=tile.data.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(result, tile.data, nl.multiply, 2.0)

    dst_tiles = nt.tiles(dst, tile_size=(P, F, K))
    dst_folded = dst_tiles[0, 0, 0].fold(2, 1)
    dst_folded.store(result)
    return dst


@nki.jit
def fold_chain_4d(src_hbm):
    """Chain two folds: (4, 8, 64, 8) -> (4, 8, 512) -> (32, 512)."""
    # src_hbm: [4, 8, 64, 8]  ->  dst: [32, 512]
    dst = nl.ndarray((32, 512), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(4, 8, 64, 8))
    step1 = src_tiles[0, 0, 0, 0].fold(3, 2)  # (4, 8, 512)
    step2 = step1.fold(0, 1)  # (32, 512)
    tile = step2.load()  # tile: [32, 512]

    dst_tiles = nt.tiles(dst, tile_size=(32, 512))
    dst_tiles[0, 0].store(tile.data)
    return dst


@nki.jit
def pattern_override_load(src_hbm):
    """load(pattern_override=, out_shape=) for stride-2 column gather.
    (32, 1024) -> (32, 512), reading every other column."""
    # src_hbm: [32, 1024]  ->  dst: [32, 512]
    P, F_src = 32, 1024
    F_dst = F_src // 2
    dst = nl.ndarray((P, F_dst), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src_hbm, tile_size=(P, F_src))
    tile = src_tiles[0, 0].load(
        pattern_override=[[F_src, P], [2, F_dst]],
        out_shape=(P, F_dst),
    )  # tile: [32, 512]

    dst_tiles = nt.tiles(dst, tile_size=(P, F_dst))
    dst_tiles[0, 0].store(tile.data)
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


def test_fold_into_partition():
    torch.manual_seed(42)
    P, F, K = 32, 512, 4
    src = torch.randn(P, F, K, dtype=torch.float32)
    result = fold_into_partition(to_device(src))
    result_np = to_cpu(result).numpy()
    src_np = src.numpy()
    for k in range(K):
        np.testing.assert_allclose(
            result_np[k * P : (k + 1) * P, :],
            src_np[:, :, k],
            rtol=1e-5,
            atol=1e-5,
        )
    print("fold_into_partition: PASSED")


def test_fold_free_dim():
    torch.manual_seed(42)
    P, F, K = 32, 128, 4
    src = torch.randn(P, F, K, dtype=torch.float32)
    result = fold_free_dim(to_device(src))
    expected = src.numpy().reshape(P, F * K)
    np.testing.assert_allclose(to_cpu(result).numpy(), expected, rtol=1e-5, atol=1e-5)
    print("fold_free_dim: PASSED")


def test_fold_partition_roundtrip():
    torch.manual_seed(42)
    P, F, K = 32, 128, 4
    src = torch.randn(P, F, K, dtype=torch.float32)
    result = fold_partition_roundtrip(to_device(src))
    expected = src.numpy() * 2.0
    np.testing.assert_allclose(to_cpu(result).numpy(), expected, rtol=1e-5, atol=1e-5)
    print("fold_partition_roundtrip: PASSED")


def test_fold_free_dim_roundtrip():
    torch.manual_seed(42)
    P, F, K = 32, 128, 4
    src = torch.randn(P, F, K, dtype=torch.float32)
    result = fold_free_dim_roundtrip(to_device(src))
    expected = src.numpy() * 2.0
    np.testing.assert_allclose(to_cpu(result).numpy(), expected, rtol=1e-5, atol=1e-5)
    print("fold_free_dim_roundtrip: PASSED")


def test_fold_chain_4d():
    torch.manual_seed(42)
    src = torch.randn(4, 8, 64, 8, dtype=torch.float32)
    result = fold_chain_4d(to_device(src))
    assert to_cpu(result).shape == (32, 512)
    print("fold_chain_4d: PASSED")


def test_pattern_override_load():
    torch.manual_seed(42)
    P, F_src = 32, 1024
    src = torch.randn(P, F_src, dtype=torch.float32)
    result = pattern_override_load(to_device(src))
    expected = src.numpy()[:, ::2]
    np.testing.assert_allclose(to_cpu(result).numpy(), expected, rtol=1e-5, atol=1e-5)
    print("pattern_override_load: PASSED")


def main():
    test_fold_into_partition()
    test_fold_free_dim()
    test_fold_partition_roundtrip()
    test_fold_free_dim_roundtrip()
    test_fold_chain_4d()
    test_pattern_override_load()


if __name__ == "__main__":
    main()
