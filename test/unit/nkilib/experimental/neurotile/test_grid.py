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

"""Grid tests on the typed-axis API.

Pure Python -- no NKI, no tracer, no device.

Covers:
  - Grid.from_shape construction (tile / block / batch / untiled).
  - Derived attributes (shape, remaining, is_remainder, tile_shape).
  - Six primitives (consume, narrow, split, split_peers, merge, reorder, broadcast).
  - Composed ops (cleanup, drop_dim, with_cursor, tile, with_block, strip_block).
  - Queries (is_tiled, is_blocked, is_sharded, tile_size_of, block_size_of).
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core.axis import AxisLabel
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

from test.utils.pytest_test_metadata import pytest_marks


def make_tiled_grid(element_shape, tile_size, block_size=None):
    """Build a Grid with optional batch padding (leading dims with tile=1)."""
    ndim = len(element_shape)
    n_batch_dims = ndim - len(tile_size)
    assert n_batch_dims >= 0

    padded_tile = [1] * n_batch_dims + list(tile_size)
    full_tile_size = tuple(padded_tile)

    full_block_size = None
    if block_size is not None:
        padded_block = [1] * n_batch_dims + list(block_size)
        full_block_size = tuple(padded_block)

    return Grid.from_shape(element_shape, full_tile_size, block_size=full_block_size, n_batch_dims=n_batch_dims)


def make_untiled_grid(element_shape):
    return Grid.from_shape(element_shape, None)


# ============================================================================
# Construction and shape
# ============================================================================


@pytest_marks(["neurotile"])
class TestGridConstruction:
    @pytest.mark.fast
    def test_tiles_shape(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.shape == (4, 4)
        assert g.ndim == 2
        assert g.remaining == (512, 2048)
        assert g.tile_size == (128, 512)

    def test_blocks_shape(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        assert g.shape == (2, 2)
        assert g.block_size == (2, 2)
        assert g.is_blocked() is True

    def test_batch_dim_shape(self):
        """Cursor defaults to n_batch_dims, so shape skips batch dims."""
        g = make_tiled_grid((8, 128, 64), (128, 64))
        # Batch is dim 0; cursor=1 skips it -> shape = (1, 1) for the inner tile dims.
        assert g.shape == (1, 1)
        assert g.n_batch_dims == 1
        # Outer count on the batch dim is its element_shape.
        assert g.outer_axis(0).count == 8

    def test_untiled_shape(self):
        g = make_untiled_grid((256, 512))
        assert g.shape == (256, 512)
        assert g.tile_size == (256, 512)  # one elem axis per dim, count=extent
        assert g.is_tiled() is False

    def test_remainder_detected(self):
        g = make_tiled_grid((300, 512), (128, 512))
        assert g.is_remainder is True

    def test_no_remainder(self):
        g = make_tiled_grid((256, 1024), (128, 512))
        assert g.is_remainder is False

    def test_tile_size_of(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.tile_size_of(0) == 128
        assert g.tile_size_of(1) == 512

    def test_block_size_of(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 4))
        assert g.block_size_of(0) == 2
        assert g.block_size_of(1) == 4

    def test_block_size_of_unblocked_returns_none(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.block_size_of(0) is None


# ============================================================================
# index_stride_elements -- public index stride metadata
# ============================================================================


@pytest_marks(["neurotile"])
class TestIndexStrideElements:
    @pytest.mark.fast
    def test_tile_view(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.index_stride_elements() == (128, 512)

    def test_block_view(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        assert g.index_stride_elements() == (256, 1024)

    def test_selected_block_is_tile_relative(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        selected = g.consume(0).consume(1).with_cursor_past_consumed((0, 1))
        assert selected.index_stride_elements() == (128, 512)

    def test_cursor_relative_after_consumed_dim(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        row = g.consume(0).with_cursor_past_consumed((0,))
        assert row.index_stride_elements() == (512,)


# ============================================================================
# Axis structure: outer-to-inner order, partition leaf labelling
# ============================================================================


@pytest_marks(["neurotile"])
class TestAxisLayout:
    @pytest.mark.fast
    def test_tile_dim_has_two_axes(self):
        """A tiled dim is (tile, leaf) -- 2 axes."""
        g = make_tiled_grid((512, 2048), (128, 512))
        assert len(g.axes_for(0)) == 2
        assert g.axes_for(0)[0].label == AxisLabel.TILE

    def test_block_dim_has_three_axes(self):
        """A blocked dim is (block, tile, leaf) -- 3 axes."""
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        assert len(g.axes_for(0)) == 3
        labels = [a.label for a in g.axes_for(0)]
        assert labels == [AxisLabel.BLOCK, AxisLabel.TILE, AxisLabel.PARTITION]

    def test_partition_leaf_on_first_non_batch_dim(self):
        """The partition axis is the leaf of the first non-batch tile dim."""
        g = make_tiled_grid((4, 128, 64), (128, 64))
        assert g.has_label(0, AxisLabel.ELEM)  # batch dim leaf is elem
        assert g.has_label(1, AxisLabel.PARTITION)  # P-dim leaf
        assert g.has_label(2, AxisLabel.ELEM)  # F-dim leaf

    def test_outer_axis_is_outermost_on_dim(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        assert g.outer_axis(0).label == AxisLabel.BLOCK
        assert g.outer_axis(0).count == 2


# ============================================================================
# consume -- removes outermost axis; cursor unchanged
# ============================================================================


@pytest_marks(["neurotile"])
class TestConsume:
    @pytest.mark.fast
    def test_consume_pops_outermost(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        g2 = g.consume(0)
        # block axis gone -> tile is now outer.
        assert g2.outer_axis(0).label == AxisLabel.TILE
        assert len(g2.axes_for(0)) == 2

    def test_consume_to_empty(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.consume(0).consume(0)
        assert g2.dim_consumed(0)
        assert g2.outer_axis(0) is None

    def test_consume_other_dim_unchanged(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.consume(0)
        assert len(g2.axes_for(1)) == 2


# ============================================================================
# narrow -- shrinks outermost axis count
# ============================================================================


@pytest_marks(["neurotile"])
class TestNarrow:
    @pytest.mark.fast
    def test_narrow_preserves_axes(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.narrow(0, 2)
        assert len(g2.axes_for(0)) == 2
        assert g2.outer_axis(0).count == 2
        assert g2.remaining[0] == 2 * 128  # 2 tiles * 128 elements

    def test_narrow_to_one(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.narrow(0, 1)
        assert g2.outer_axis(0).count == 1


# ============================================================================
# split -- factors outermost axis (tile-style: inner has the original step)
# ============================================================================


@pytest_marks(["neurotile"])
class TestSplit:
    @pytest.mark.fast
    def test_split_factors_count(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        # Outer tile axis count=4, step=128. Split by 2 (tile-style).
        g2 = g.split(0, factor=2, outer_label=AxisLabel.BLOCK, inner_label=AxisLabel.TILE)
        # outer.count = 4//2 = 2; outer.step = 128 * 2 = 256
        # inner.count = 2;        inner.step = 128
        outer = g2.axes_for(0)[0]
        inner = g2.axes_for(0)[1]
        assert (outer.count, outer.step) == (2, 256)
        assert (inner.count, inner.step) == (2, 128)


# ============================================================================
# split_peers -- interleaved sharding factoring
# ============================================================================


@pytest_marks(["neurotile"])
class TestSplitPeers:
    @pytest.mark.fast
    def test_split_peers_step(self):
        """peer.step == self.step; owned.step == self.step * num_peers."""
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.split_peers(0, num_peers=2, peer_label=AxisLabel.SHARD, owned_label=AxisLabel.TILE)
        peer = g2.axes_for(0)[0]
        owned = g2.axes_for(0)[1]
        assert (peer.count, peer.step) == (2, 128)
        assert (owned.count, owned.step) == (2, 256)


# ============================================================================
# merge -- inverse of split
# ============================================================================


@pytest_marks(["neurotile"])
class TestMerge:
    @pytest.mark.fast
    def test_merge_two_axes(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        # block + tile = 256-step outer covering 4 items.
        g2 = g.merge(0, n_axes=2, label=AxisLabel.TILE)
        # After merge: one outer axis count=4 step=128, then leaf.
        assert len(g2.axes_for(0)) == 2
        assert g2.outer_axis(0).count == 4
        assert g2.outer_axis(0).step == 128


# ============================================================================
# drop_dim
# ============================================================================


@pytest_marks(["neurotile"])
class TestDropDim:
    @pytest.mark.fast
    def test_drops_axes_and_shape(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.drop_dim(0)
        assert g2.ndim == 1
        assert g2.element_shape == (2048,)
        assert g2.tile_size == (512,)

    def test_decrements_n_batch_dims(self):
        g = make_tiled_grid((8, 128, 64), (128, 64))
        assert g.n_batch_dims == 1
        g2 = g.drop_dim(0)
        assert g2.n_batch_dims == 0


# ============================================================================
# with_cursor -- shifts the iteration cursor
# ============================================================================


@pytest_marks(["neurotile"])
class TestWithCursor:
    @pytest.mark.fast
    def test_advance_cursor(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.shape == (4, 4)
        g2 = g.with_cursor(1)
        assert g2.shape == (4,)

    def test_past_all_dims(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.with_cursor(2)
        assert g2.shape == ()


# ============================================================================
# cleanup -- drops consumed dims, returns (Grid, dropped_dims)
# ============================================================================


@pytest_marks(["neurotile"])
class TestCleanup:
    @pytest.mark.fast
    def test_no_consumed_dims(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2, dropped = g.cleanup()
        assert g2.ndim == 2
        assert dropped == ()

    def test_consumed_batch_dropped(self):
        g = make_tiled_grid((8, 512, 2048), (128, 512))
        g2 = g.consume(0)
        g3, dropped = g2.cleanup()
        assert g3.ndim == 2
        assert g3.element_shape == (512, 2048)
        assert dropped == (0,)

    def test_min_2d_preserved_when_tiled(self):
        """Tiled view with one consumed dim still keeps 2 dims."""
        g = make_tiled_grid((128, 512), (128, 512))
        g2 = g.consume(0).consume(0)  # dim 0 consumed; dim 1 still tiled.
        g3, dropped = g2.cleanup()
        assert g3.ndim == 2
        assert dropped == ()


# ============================================================================
# tile -- re-tile primitive
# ============================================================================


@pytest_marks(["neurotile"])
class TestTile:
    @pytest.mark.fast
    def test_basic_retile(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.tile((64, 512))
        assert g2.tile_size == (64, 512)


# ============================================================================
# strip_block -- removes block axes, leaving tile + leaf per dim
# ============================================================================


@pytest_marks(["neurotile"])
class TestStripBlock:
    @pytest.mark.fast
    def test_strip_removes_block_axes(self):
        g = make_tiled_grid((512, 2048), (128, 512), block_size=(2, 2))
        g2 = g.strip_block()
        assert g2.is_blocked() is False
        assert len(g2.axes_for(0)) == 2

    def test_strip_unblocked_is_noop(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.strip_block()
        assert g2.axes == g.axes


# ============================================================================
# is_sharded / structural shard recognition
# ============================================================================


@pytest_marks(["neurotile"])
class TestSharded:
    @pytest.mark.fast
    def test_unsharded_grid(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.is_sharded(0) is False
        assert g.is_sharded(1) is False

    def test_split_peers_marks_shard(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        g2 = g.split_peers(1, num_peers=2, peer_label=AxisLabel.SHARD, owned_label=AxisLabel.TILE)
        # The peer axis carries the SHARD label; consume it to leave the owned axis.
        g3 = g2.consume(1)
        assert g3.is_sharded(1) is False  # peer consumed
        assert g2.is_sharded(1) is True  # peer still present


# ============================================================================
# owned_extents -- partition-aware element walk
# ============================================================================


@pytest_marks(["neurotile"])
class TestOwnedExtents:
    @pytest.mark.fast
    def test_unsharded_full_span(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        assert g.owned_extents()[0] == 512
        assert g.owned_extents()[1] == 2048


# ============================================================================
# repr
# ============================================================================


@pytest_marks(["neurotile"])
class TestRepr:
    @pytest.mark.fast
    def test_repr_includes_shape(self):
        g = make_tiled_grid((512, 2048), (128, 512))
        r = repr(g)
        assert "Grid(" in r
        assert "shape=(4, 4)" in r
