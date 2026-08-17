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

"""CPU-only regression tests for batch dim indexing in _index_multi.

Covers:
- Batch dim skip-over: keys skip consumed batch dims to target tile dims
- Indirect dim preservation: indirect_dim stays in source tensor coordinates
- Min-2D guard: tiled views never drop below 2D
- Chained view select: slab[idx] then row[idx, :] (negative remaining bug)
- Unconsumed batch dim guard: .load()/.store()/.stream() assert n_batch_dims==0

On-device tests live in test/integration/.../test_batch_dim_indexing.py.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_strides
from nkilib_src.nkilib.experimental.neurotile.core.axis import IndirectKind, IndirectOffset
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout
from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks


def make_view_3d(shape, tile_size_2d):
    """Create a 3D HBM NDSlice with n_batch_dims=1."""
    ndim = len(shape)
    n_batch = ndim - len(tile_size_2d)
    padded = []
    for _d in range(n_batch):
        padded.append(1)
    for d in range(len(tile_size_2d)):
        padded.append(tile_size_2d[d])
    full_tile = tuple(padded)
    grid = Grid.from_shape(shape, full_tile, n_batch_dims=n_batch)
    strides = contiguous_strides(shape)
    layout = HBMLayout(MockTensor(shape), 0, strides, "float32", "shared_hbm")
    return NDSlice(grid, layout)


@pytest_marks(["neurotile"])
class TestBatchDimSkipOver:
    """Keys skip over consumed batch dims to reach tile dims."""

    @pytest.mark.fast
    def test_3d_int_batch_then_int_tile(self):
        v = make_view_3d((8, 128, 256), (128, 128))
        assert v.shape == (1, 2)
        child = v[0, 0]
        assert child._grid.remaining[0] == 128
        assert child._grid.remaining[1] == 256

    def test_3d_batch_then_ftile(self):
        v = make_view_3d((8, 128, 256), (128, 128))
        child = v[0, 0, 1]
        assert child._grid.remaining[0] == 128
        assert child._grid.remaining[1] == 128

    def test_3d_runtime_batch_then_int_tile(self):
        v = make_view_3d((8, 128, 256), (128, 128))

        class FakeRuntime:
            pass

        child = v[FakeRuntime(), 0]
        assert child._grid.remaining[0] == 128
        assert child._grid.remaining[1] == 256

    def test_2d_no_batch(self):
        grid = Grid.from_shape((256, 512), (128, 256))
        strides = contiguous_strides((256, 512))
        layout = HBMLayout(MockTensor((256, 512)), 0, strides, "float32", "shared_hbm")
        v = NDSlice(grid, layout)
        child = v[0, 1]
        assert child._grid.remaining == (128, 256)


@pytest_marks(["neurotile"])
class TestIndirectDimPreservation:
    """indirect_dim stays in source tensor coordinates after batch skip."""

    @pytest.mark.fast
    def test_scalar_indirect_after_batch(self):
        v = make_view_3d((4, 8, 64), (1, 64))
        child_b = v[0]
        assert child_b._layout.indirect is None

    def test_drop_dim_preserves_indirect(self):
        layout = HBMLayout(
            source=None,
            offset=0,
            strides=(512, 64, 1),
            dtype="float32",
            indirect=IndirectOffset(kind=IndirectKind.SCALAR, value="seq_pos", dim=1),
        )
        dropped = layout.drop_dim(0)
        assert dropped.indirect.dim == 1
        assert dropped.strides == (64, 1)


@pytest_marks(["neurotile"])
class TestMin2DGuard:
    """Tiled views never drop below 2D."""

    @pytest.mark.fast
    def test_1d_tile_on_2d_tensor(self):
        v = make_view_3d((8, 256), (128,))
        assert v.ndim == 2
        assert v._grid.tile_size == (8, 128)
        child = v[0, 0]
        assert child.ndim >= 1

    def test_1d_tile_on_3d_tensor(self):
        v = make_view_3d((4, 8, 64), (64,))
        assert v.ndim == 3
        assert v._grid.tile_size == (4, 8, 64)
        child = v[0, 0]
        assert child.ndim >= 2


@pytest_marks(["neurotile"])
class TestChainedViewSelect:
    """Chained indexing: slab select then row select."""

    @pytest.mark.fast
    def test_slab_then_row(self):
        v = make_view_3d((4, 4, 64), (4, 64))
        slab = v[1]
        assert slab._grid.remaining[0] == 4
        assert slab._grid.remaining[1] == 64
        row = slab[2, :]
        assert row._grid.remaining[0] == 1
        assert row._grid.remaining[1] == 64


@pytest_marks(["neurotile"])
class TestUnconsumedBatchDimGuard:
    """.load()/.store()/.stream() must assert when batch dims are unconsumed."""

    @pytest.mark.fast
    def test_load_raises_with_unconsumed_batch_dim(self):
        v = make_view_3d((8, 128, 256), (128, 128))
        assert v._grid.n_batch_dims == 1
        with pytest.raises(AssertionError, match="batch dims to be consumed"):
            v.load()

    def test_load_raises_with_two_unconsumed_batch_dims(self):
        shape = (2, 4, 128, 256)
        n_batch = 2
        padded = (1, 1, 128, 256)
        grid = Grid.from_shape(shape, padded, n_batch_dims=n_batch)
        strides = contiguous_strides(shape)
        layout = HBMLayout(MockTensor(shape), 0, strides, "float32", "shared_hbm")
        v = NDSlice(grid, layout)
        assert v._grid.n_batch_dims == 2
        with pytest.raises(AssertionError, match="batch dims to be consumed"):
            v.load()

    def test_consuming_batch_dim_drops_guard(self):
        v = make_view_3d((8, 128, 256), (128, 128))
        assert v._grid.n_batch_dims == 1
        slab = v[3]
        assert slab._grid.n_batch_dims == 0

    def test_store_raises_with_unconsumed_batch_dim(self):
        v = make_view_3d((8, 128, 256), (128, 128))
        assert v._grid.n_batch_dims == 1
        fake_data = object()
        with pytest.raises(AssertionError, match="batch dims to be consumed"):
            v.store(fake_data)
