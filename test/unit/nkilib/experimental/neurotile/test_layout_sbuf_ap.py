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

"""Tests for SBUFLayout AP pattern construction.

Locks the partition-stride invariant: level-0 stride of every emitted
AP pattern must equal the underlying SBUF ndarray's free-dim flat width
(``product(sbuf.shape[1:])``), regardless of how narrow the walked region
is. The compiler enforces this rule in
``nki/isa/validation.py::validate_contiguous_partition_access``; emitting a
narrower stride causes a compile-time assertion.

Pure Python: ``MockTensor`` stands in for ``nl.ndarray`` and returns a
``MockApView`` from ``.ap()`` so we can introspect the pattern the layout would
have submitted.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks


@pytest_marks(["neurotile"])
class TestPartitionRowStride:
    """``_partition_row_stride`` returns the underlying-buffer free width."""

    @pytest.mark.fast
    def test_2d_buffer(self):
        sbuf = MockTensor((128, 512))
        assert SBUFLayout._partition_row_stride(sbuf) == 512

    @pytest.mark.fast
    def test_3d_buffer(self):
        sbuf = MockTensor((128, 32, 4))
        assert SBUFLayout._partition_row_stride(sbuf) == 128

    @pytest.mark.fast
    def test_independent_of_walked_remaining(self):
        """Stride is a property of the ndarray, not the caller's walk."""
        wide = MockTensor((128, 1024))
        assert SBUFLayout._partition_row_stride(wide) == 1024


@pytest_marks(["neurotile"])
class TestBuildDefaultAp:
    """``_build_default_ap`` produces compiler-valid AP patterns.

    Level-0 always pins to the underlying ndarray's free width (read from
    ``_storage_shape``), satisfying the new compiler's
    ``partition_step == tensor.free_dim_size`` invariant.
    """

    @pytest.mark.fast
    def test_single_p_tile_full_walk(self):
        sbuf = MockTensor((128, 512))
        ap = SBUFLayout._build_default_ap(sbuf, remaining=(128, 512), tile_p=128, p_tiles=1)
        # [[partition_stride, tile_p], [1, remaining[1]]]
        # partition_stride = product((128, 512)[1:]) = 512
        assert ap.pattern == [[512, 128], [1, 512]]

    @pytest.mark.fast
    def test_single_p_tile_subtile_walk(self):
        """Sub-tile (1x1) walk on a wider buffer: level-0 still pins to underlying free width."""
        sbuf = MockTensor((128, 64))
        ap = SBUFLayout._build_default_ap(sbuf, remaining=(1, 1), tile_p=1, p_tiles=1)
        # [[partition_stride=64, tile_p=1], [1, remaining[1]=1]]
        assert ap.pattern == [[64, 1], [1, 1]]

    @pytest.mark.fast
    def test_single_p_tile_partial_f_walk(self):
        """Partial F walk over a wider buffer (the original bug case)."""
        sbuf = MockTensor((128, 512))
        ap = SBUFLayout._build_default_ap(sbuf, remaining=(128, 128), tile_p=128, p_tiles=1)
        # partition_stride = 512 (parent), inner walks remaining[1] = 128
        assert ap.pattern == [[512, 128], [1, 128]]

    @pytest.mark.fast
    def test_multi_p_tile_uses_allocated_free_width_as_partition_stride(self):
        sbuf = MockTensor((128, 1024))
        ap = SBUFLayout._build_default_ap(sbuf, remaining=(256, 512), tile_p=128, p_tiles=2)
        # [[partition_stride, tile_p], [1, total_f]]
        # partition_stride = product((128, 1024)[1:]) = 1024
        # total_f = f_extent((256, 512), 128) = 512 * 2 = 1024
        assert ap.pattern == [[1024, 128], [1, 1024]]


class _StubGrid:
    """Minimal grid: ``_ap_with_transform`` reads only ``.remaining``."""

    def __init__(self, remaining):
        self.remaining = remaining


@pytest_marks(["neurotile"])
class TestApWithTransform:
    """``_ap_with_transform`` (broadcast / non-contiguous AP) must pin level-0
    to the underlying buffer's physical row width -- the same invariant the
    default path holds. The transform's own strides (stride-0 broadcast,
    contiguous inner) drive levels 1+, but level 0 is the physical partition
    stride, NOT the logical tile width from contiguous_strides(element_shape).
    """

    def _layout(self, source, alloc_tile_size, transform_strides):
        return SBUFLayout(
            source,
            0,  # offset
            (1,) * len(alloc_tile_size),  # strides (unused by this path)
            alloc_tile_size,
            dtype="bfloat16",
            transform_strides=transform_strides,
        )

    @pytest.mark.fast
    def test_subtile_partition_stride_uses_storage_width(self):
        """The bug case: a (128,128) broadcast view backed by a (128,256)
        buffer must emit level-0 stride 256 (physical), not 128 (logical)."""
        # cos tile (seq=128, d=128) sliced from a 2-tile (128,256) allocation,
        # broadcast to (seq, G=1, d): ap_strides = (d, 0, 1) = (128, 0, 1).
        src = MockTensor((128, 128), storage_shape=(128, 256))
        layout = self._layout(src, alloc_tile_size=(128, 128), transform_strides=(128, 0, 1))
        view = layout._ap_with_transform(_StubGrid(remaining=(128, 1, 128)))
        # Level 0 pinned to physical 256; inner levels keep the transform strides.
        assert view.pattern == [[256, 128], [0, 1], [1, 128]]

    @pytest.mark.fast
    def test_broadcast_keeps_inner_stride_zero(self):
        """The stride-0 broadcast level (and contiguous inner) survive intact;
        only level 0 is overridden."""
        src = MockTensor((128, 128), storage_shape=(128, 256))
        layout = self._layout(src, alloc_tile_size=(128, 128), transform_strides=(128, 0, 1))
        view = layout._ap_with_transform(_StubGrid(remaining=(128, 4, 128)))
        assert view.pattern[1] == [0, 4]  # broadcast level preserved
        assert view.pattern[2] == [1, 128]  # contiguous d preserved

    @pytest.mark.fast
    def test_standalone_tile_partition_stride_equals_width(self):
        """Standalone tile (storage == shape): level-0 stride equals the tile
        width, so the override is a no-op vs the pre-fix behavior."""
        src = MockTensor((128, 128))  # storage_shape defaults to shape
        layout = self._layout(src, alloc_tile_size=(128, 128), transform_strides=(128, 0, 1))
        view = layout._ap_with_transform(_StubGrid(remaining=(128, 1, 128)))
        assert view.pattern == [[128, 128], [0, 1], [1, 128]]


def _indexed_tile_grid(element_shape, tile_size, p_index, f_index):
    """Grid state after ``view[p_index, f_index]`` int-descent (Grid-level).

    Reproduces the folded-partition branch of ``NDSlice.__getitem__``: on a
    folded partition dim the elements consumed are ``index * tile_p``, so a
    trailing partial P-tile narrows to its addressable extent.
    """
    g = Grid.from_shape(element_shape=element_shape, tile_size=tile_size)
    for dim, k in ((0, p_index), (1, f_index)):
        g = g.consume(dim)
        g = g.truncate_to_source(dim, k * tile_size[dim])
    g, _ = g.cleanup()
    return g


@pytest_marks(["neurotile"])
class TestApStandardPartitionClamp:
    """``_ap_standard`` clamps the emitted AP's level-0 partition count to the
    Grid's addressable P. A partial partition tile's ``.data`` then reports its
    true height (e.g. 44), symmetric with a partial free tile reporting its
    true width -- while the level-0 stride stays the physical row width.
    """

    def _layout(self, source_shape):
        src = MockTensor(source_shape, storage_shape=source_shape)
        return SBUFLayout(src, 0, (1, 1), (128, 128), dtype="float32")

    @pytest.mark.fast
    def test_single_subtile_partition_clamps_count(self):
        """element_shape (44,256): the lone P-tile is a 44-row sub-tile. AP
        level-0 count is 44 (addressable), stride is the physical 256 width."""
        g = _indexed_tile_grid((44, 256), (128, 128), 0, 0)
        ap = self._layout((128, 256))._ap_standard(g)
        assert ap.pattern == [[256, 44], [1, 128]]

    @pytest.mark.fast
    def test_full_partition_unchanged(self):
        """Regression guard for the hot path: a full 128-row P-tile still emits
        count 128 -- the clamp is a no-op when the partition extent is full."""
        g = _indexed_tile_grid((128, 256), (128, 128), 0, 0)
        ap = self._layout((128, 256))._ap_standard(g)
        assert ap.pattern == [[256, 128], [1, 128]]

    @pytest.mark.fast
    def test_trailing_partition_tile_clamps_count(self):
        """element_shape (300,256): the trailing P-tile [2,0] is 44 rows. AP
        level-0 count is 44; stride is the physical folded width (6*128=768)."""
        g = _indexed_tile_grid((300, 256), (128, 128), 2, 0)
        ap = self._layout((128, 768))._ap_standard(g)
        assert ap.pattern == [[768, 44], [1, 128]]
