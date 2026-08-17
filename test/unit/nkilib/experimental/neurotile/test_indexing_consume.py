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

"""CPU-only behavioral tests for the locked-in indexing model.

These check NDSlice attributes (shape, element_shape, ndim, cursor) after
various indexing patterns. No NKI tracer, no device. Uses internal-class
deep imports (Grid, NDSlice, HBMLayout) per the §3.5 exception.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_strides
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout
from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks


def _make_view(element_shape, tile_size_2d, block_size=None, n_batch=0):
    """Build an NDSlice over a mock HBM layout for attribute checks."""
    full_tile = tuple([1] * n_batch) + tuple(tile_size_2d)
    full_block = None
    if block_size is not None:
        full_block = tuple([1] * n_batch) + tuple(block_size)
    grid = Grid.from_shape(element_shape, full_tile, block_size=full_block, n_batch_dims=n_batch)
    strides = contiguous_strides(element_shape)
    layout = HBMLayout(
        source=MockTensor(element_shape), offset=0, strides=strides, dtype="float32", buffer_type="shared_hbm"
    )
    return NDSlice(grid, layout)


def _tile_view():
    return _make_view((512, 512), (128, 128))


def _block_view():
    return _make_view((1024, 512), (128, 128), block_size=(2, 2))


@pytest_marks(["neurotile"])
class TestTileGridShape:
    @pytest.mark.fast
    def test_full_grid(self):
        v = _tile_view()
        assert v.shape == (4, 4)
        assert v.element_shape == (512, 512)
        assert v.ndim == 2

    def test_int_dim0_only(self):
        v = _tile_view()
        r = v[0]
        assert r.shape == (4,)
        assert r.ndim == 2
        assert r.element_shape == (128, 512)

    def test_int_dim0_then_full_dim1(self):
        v = _tile_view()
        r = v[0, :]
        assert r.shape == (4,)
        assert r.ndim == 2
        assert r.element_shape == (128, 512)

    def test_full_dim0_int_dim1(self):
        v = _tile_view()
        r = v[:, 0]
        assert r.shape == (4, 1)
        assert r.element_shape == (512, 128)

    def test_int_int_single_tile(self):
        v = _tile_view()
        r = v[0, 0]
        assert r.element_shape == (128, 128)
        assert r.ndim == 2
        assert r.shape == ()

    def test_slice_slice(self):
        v = _tile_view()
        r = v[1:3, 1:3]
        assert r.shape == (2, 2)
        assert r.element_shape == (256, 256)

    def test_int_slice_mix(self):
        v = _tile_view()
        r = v[1, 1:3]
        assert r.shape == (2,)
        assert r.element_shape == (128, 256)

    def test_slice_int_mix(self):
        v = _tile_view()
        r = v[1:3, 1]
        assert r.shape == (2, 1)
        assert r.element_shape == (256, 128)

    def test_negative_int(self):
        v = _tile_view()
        r = v[-1, :]
        assert r.shape == (4,)
        assert r.element_shape == (128, 512)

    def test_stepped_slice(self):
        v = _tile_view()
        r = v[::2, :]
        assert r.shape == (2, 4)
        assert r.ndim == 2


@pytest_marks(["neurotile"])
class TestBlockGridShape:
    @pytest.mark.fast
    def test_full_grid(self):
        v = _block_view()
        assert v.shape == (4, 2)
        assert v.element_shape == (1024, 512)

    def test_int_dim0_consumes_block_level(self):
        v = _block_view()
        r = v[0]
        assert r.shape == (2,)
        assert r.element_shape == (256, 512)

    def test_int_dim0_full_dim1(self):
        v = _block_view()
        r = v[0, :]
        assert r.shape == (2,)
        assert r.element_shape == (256, 512)

    def test_drill_block_row_to_block_interior(self):
        v = _block_view()
        block_row = v[0, :]
        block = block_row[0]
        assert block.shape == (2, 2)
        assert block.element_shape == (256, 256)

    def test_int_int_block_interior(self):
        v = _block_view()
        r = v[0, 1]
        assert r.shape == (2, 2)
        assert r.element_shape == (256, 256)

    def test_block_subgrid_via_range_slices(self):
        v = _block_view()
        r = v[1:3, 0:2]
        assert r.shape == (2, 2)

    def test_too_many_keys_rejected(self):
        v = _block_view()
        with pytest.raises(AssertionError, match="too many keys"):
            _ = v[0, 0, 0, 0]


@pytest_marks(["neurotile"])
class TestBlockSizeAfterConsume:
    """Grid.block_size after consume: no sentinel 0 for consumed dims."""

    @pytest.mark.fast
    def test_no_zero_in_block_size_after_int_consume(self):
        v = _block_view()
        block_row = v[0, :]
        if block_row.block_size is not None:
            for d in range(len(block_row.block_size)):
                bs = block_row.block_size[d]
                assert bs is None or bs >= 1


@pytest_marks(["neurotile"])
class TestSingleIntDeflectPastConsumedDims:
    """Single-int keys on a child view skip past consumed leading dims."""

    @pytest.mark.fast
    def test_row_j_targets_dim_1(self):
        v = _tile_view()
        row = v[0, :]
        assert row._grid.cursor == 1
        tile = row[2]
        assert tile.element_shape == (128, 128)

    def test_block_row_bj_targets_dim_1(self):
        v = _block_view()
        block_row = v[0, :]
        assert block_row._grid.cursor == 1
        block = block_row[0]
        assert block.shape == (2, 2)
        assert block.element_shape == (256, 256)

    def test_no_deflect_on_batch_dim(self):
        v = _make_view((8, 256, 256), (128, 128), n_batch=1)
        assert v._grid.cursor == 1
        assert v._grid.n_batch_dims == 1
        slab = v[3]
        assert slab.ndim == 2
        assert slab._grid.n_batch_dims == 0
