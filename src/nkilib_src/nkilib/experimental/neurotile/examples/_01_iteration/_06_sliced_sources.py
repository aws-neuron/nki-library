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
Sliced HBM sources: tile a slice directly. A slice (``src[a:b, c:d]``) shares
its parent's storage and carries the parent's physical strides + its own
offset, so ``nt.tiles(slice, ...)`` addresses it correctly with no extra
argument -- including a nonzero-offset window.
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
def window_scale(src):
    """Tile a sliced HBM window directly -- the slice carries the parent's
    stride and its own offset, so no root= is needed."""
    P, F = 128, 512
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    window = src[0:P, 0:F]
    src_tiles = nt.tiles(window, tile_size=(P, F))
    out_tiles = nt.tiles(out, tile_size=(P, F))

    tile = src_tiles[0, 0].load()
    nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
    out_tiles[0, 0].store(tile.data)

    return out


@nki.jit
def two_windows(src):
    """Two non-overlapping windows of the same parent. The second is a
    nonzero-offset slice (rows P:2P) -- it self-addresses, no root= needed."""
    P, F = 128, 512
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    top = src[0:P, 0:F]
    bottom = src[P : 2 * P, 0:F]  # offset P*F into the parent

    top_tiles = nt.tiles(top, tile_size=(P, F))
    bottom_tiles = nt.tiles(bottom, tile_size=(P, F))
    out_tiles = nt.tiles(out, tile_size=(P, F))

    top_tile = top_tiles[0, 0].load()
    out_tiles[0, 0].store(top_tile.data)

    bottom_tile = bottom_tiles[0, 0].load()
    nisa.tensor_scalar(bottom_tile.data, bottom_tile.data, nl.multiply, -1.0)
    out_tiles[1, 0].store(bottom_tile.data)

    return out


# ============================================================================
# Helpers
# ============================================================================


def to_device(np_arr):
    import torch_xla.core.xla_model as xm

    return torch.from_numpy(np.ascontiguousarray(np_arr)).to(xm.xla_device())


def to_numpy(t):
    return t.cpu().to(torch.float32).numpy()


# ============================================================================
# Tests
# ============================================================================


def test_window_scale():
    np.random.seed(42)
    data = np.random.randn(256, 512).astype(np.float32)
    result = window_scale(to_device(data))
    expected = data[0:128, 0:512] * 2.0
    np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)
    print("window_scale: PASSED")


def test_two_windows():
    np.random.seed(42)
    data = np.random.randn(256, 512).astype(np.float32)
    result = two_windows(to_device(data))
    expected = data.copy()
    expected[128:256, :] *= -1.0
    np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)
    print("two_windows: PASSED")


def main():
    test_window_scale()
    test_two_windows()


if __name__ == "__main__":
    main()
