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

"""Coverage for ``Grid.remainder_dims`` + partial-trailing-tile semantics.

Pins three behaviors:

1. ``Grid.remainder_dims`` is per-dim -- a remainder on dim 1 must not
   trigger ``_compute_remaining``'s clamped-leaf path on other dims.
2. ``truncate_to_source`` finds the elem-leaf (innermost ``step==1``
   axis) on the dim it clamps, whether that is the outermost axis
   (post-``consume``) or sits inside a TILE / BLOCK wrapper
   (post-``narrow``).
3. ``view[:, last_tile:last_tile+1]`` over a partial trailing tile
   reports the partial-aware ``element_shape`` -- this is what unblocks
   the FGCC MLP CTE remainder kernels.

The PSUM new-API tests live in ``test_psum_pool_api.py``.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core.axis import Axis, AxisLabel
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

from test.utils.pytest_test_metadata import pytest_marks

# ============================================================================
# remainder_dims: per-dim flag tracking
# ============================================================================


@pytest_marks(["neurotile"])
class TestRemainderDimsPerDim:
    """``remainder_dims`` is a tuple of dim indices; ``is_remainder`` is a
    convenience bool (``len(remainder_dims) > 0``)."""

    @pytest.mark.fast
    def test_constructor_default_empty(self):
        # No clamping: an exactly-tile-aligned grid surfaces no remainder.
        g = Grid.from_shape(element_shape=(128, 512), tile_size=(128, 128))
        assert g.remainder_dims == ()
        assert g.is_remainder is False

    def test_construction_derives_remainder_from_overshoot(self):
        # Source 1792 with tile_size 512 -> ceil 4 tiles, last partial.
        # Outer axis count*step (= 4*512=2048) overshoots element_shape (1792).
        g = Grid.from_shape(element_shape=(128, 1792), tile_size=(128, 512))
        assert 1 in g.remainder_dims
        assert g.is_remainder is True

    def test_explicit_remainder_dims_carries_through(self):
        # Direct constructor: ``remainder_dims=`` records dims clamped by
        # truncate_to_source even when overshoot detection would not fire.
        es = (128, 256)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=256, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
            remainder_dims=(1,),
        )
        assert 1 in g.remainder_dims

    def test_remainder_on_one_dim_does_not_pollute_other_dims(self):
        """Regression: a single-bool ``is_remainder`` made
        ``_compute_remaining`` mis-clamp every dim once any dim was a
        remainder."""
        # Dim 0 fully tiles (4096 / 128 = 32); dim 1 has remainder
        # (1792 / 512 = 3 + partial).
        g = Grid.from_shape(
            element_shape=(4096, 1792),
            tile_size=(128, 512),
            block_size=(32, 1),
        )
        # Dim 0 should report the full source extent.
        assert g.remaining[0] == 4096
        # Dim 1 reports the source extent (overshoot clamped to source).
        assert g.remaining[1] == 1792

    def test_dims_walking_past_source_detects_per_dim(self):
        g = Grid.from_shape(
            element_shape=(4096, 1792),
            tile_size=(128, 512),
            block_size=(32, 1),
        )
        derived = g._dims_walking_past_source()
        # Dim 0 fits exactly; only dim 1 walks past.
        assert derived == (1,)


# ============================================================================
# truncate_to_source: finds the innermost step==1 axis and clamps
# ============================================================================


@pytest_marks(["neurotile"])
class TestTruncateToSource:
    @pytest.mark.fast
    def test_clamps_outer_when_outer_is_elem_leaf(self):
        """Single-axis dim: outer IS the elem-leaf -- clamp via narrow."""
        # Single elem axis count=512, source extent=1024. 100 elements
        # consumed -> addressable=924; leaf 512 fits, no clamp. Bump
        # consumed to 700: addressable=324 < 512 -> clamps to 324.
        es = (128, 1024)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        clamped = g.truncate_to_source(dim=1, elements_consumed=700)
        elem = [ax for ax in clamped.axes if ax.dim == 1 and ax.step == 1][0]
        assert elem.count == 324
        assert 1 in clamped.remainder_dims

    def test_clamps_elem_leaf_inside_tile_block_wrapper(self):
        """Multi-axis dim: BLOCK -> TILE -> ELEM. Clamp the inner ELEM."""
        es = (128, 1792)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        # Slice on dim 1 narrowed BLOCK to count=1; ELEM still 512.
        ax_b = Axis(count=1, step=512, dim=1, label=AxisLabel.BLOCK)
        ax_t = Axis(count=1, step=512, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_b, ax_t, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        # 1536 elements consumed (slice start 3 * 512). Addressable = 256.
        clamped = g.truncate_to_source(dim=1, elements_consumed=1536)
        elem = [ax for ax in clamped.axes if ax.dim == 1 and ax.step == 1][0]
        assert elem.count == 256
        # BLOCK / TILE wrappers untouched.
        block = [ax for ax in clamped.axes if ax.label == AxisLabel.BLOCK][0]
        assert block.count == 1 and block.step == 512

    def test_no_op_when_addressable_fits(self):
        """When the leaf already fits the addressable remainder, no change."""
        es = (128, 1024)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=128, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        # Addressable=1024-100=924 >= leaf.count=128 -> no-op.
        clamped = g.truncate_to_source(dim=1, elements_consumed=100)
        assert clamped is g or clamped.axes == g.axes
        assert 1 not in clamped.remainder_dims

    def test_no_op_for_non_int_elements_consumed(self):
        """Runtime ``elements_consumed`` (CExpr / LoopVar) -> no-op."""
        es = (128, 256)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_e),
            cursor=0,
            n_batch_dims=0,
        )

        class FakeCExpr:
            pass

        clamped = g.truncate_to_source(dim=1, elements_consumed=FakeCExpr())
        assert clamped is g

    def test_no_op_when_no_elem_leaf(self):
        """Dim has only TILE / BLOCK axes, no step==1 axis."""
        es = (128, 256)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_b = Axis(count=4, step=64, dim=1, label=AxisLabel.BLOCK)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_b),
            cursor=0,
            n_batch_dims=0,
        )
        clamped = g.truncate_to_source(dim=1, elements_consumed=100)
        assert clamped is g

    @pytest.mark.fast
    def test_clamps_tile_count_for_partial_block(self):
        """Partial trailing block walks fewer whole tiles than its nominal count.

        A 2-tile M-block over a source of 384 rows (tile 128): block 1 starts
        at row 256, so only 384 - 256 = 128 rows remain -- one whole tile, not
        two. The remainder is a whole number of tiles (no partial last tile),
        so the leaf stays 128 and only the TILE-grid count clamps 2 -> 1, which
        keeps ``shape`` / ``tile_shape`` and tile iteration from indexing a tile
        the block does not hold.
        """
        es = (384, 1024)
        ax_t = Axis(count=2, step=128, dim=0, label=AxisLabel.TILE)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_e = Axis(count=1024, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(element_shape=es, axes=(ax_t, ax_p, ax_e), cursor=0, n_batch_dims=0)
        clamped = g.truncate_to_source(dim=0, elements_consumed=256)
        tile = [ax for ax in clamped.axes if ax.dim == 0 and ax.label == AxisLabel.TILE][0]
        assert tile.count == 1  # 1 whole tile reachable, not the nominal 2
        partition = [ax for ax in clamped.axes if ax.label == AxisLabel.PARTITION][0]
        assert partition.count == 128  # leaf unchanged: the lone tile is full
        assert 0 in clamped.remainder_dims

    @pytest.mark.fast
    def test_multi_tile_remainder_keeps_full_leaf(self):
        """A remainder spanning several tiles keeps the shared leaf full width.

        A 2-tile N-block over 1792 cols (tile 512): block 1 starts at col 1024,
        leaving 768 = one full 512 tile + a 256 partial. The trailing partial
        is resolved per tile at descent, so the TILE count stays 2 and the leaf
        stays 512 -- only a lone partial tile narrows the leaf.
        """
        es = (128, 1792)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_t = Axis(count=2, step=512, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=512, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(element_shape=es, axes=(ax_p, ax_t, ax_e), cursor=0, n_batch_dims=0)
        clamped = g.truncate_to_source(dim=1, elements_consumed=1024)
        tile = [ax for ax in clamped.axes if ax.dim == 1 and ax.label == AxisLabel.TILE][0]
        elem = [ax for ax in clamped.axes if ax.dim == 1 and ax.step == 1][0]
        assert tile.count == 2  # ceil(768 / 512) = 2 tiles reachable
        assert elem.count == 512  # leaf full: trailing partial handled per tile


# ============================================================================
# _compute_remaining: partial-trailing-tile formula on flagged dims only
# ============================================================================


@pytest_marks(["neurotile"])
class TestComputeRemainingPartialTile:
    @pytest.mark.fast
    def test_normal_walk_uses_count_times_step(self):
        # 4 tiles of 128 each -> walked 512.
        g = Grid.from_shape(element_shape=(128, 512), tile_size=(128, 128))
        assert g.remaining == (128, 512)

    def test_partial_trailing_tile_uses_clamped_leaf(self):
        # 1792 / 512 -> 3 full + 1 partial of 256.
        # _compute_remaining for dim 1 takes the partial-aware path:
        #   walked = (4-1)*512 + 512 = 2048; min(2048, 1792) = 1792.
        # Dim 0: 32 * 128 = 4096 == element_shape -> normal path.
        g = Grid.from_shape(
            element_shape=(4096, 1792),
            tile_size=(128, 512),
            block_size=(32, 1),
        )
        assert g.remaining[0] == 4096
        assert g.remaining[1] == 1792

    def test_post_truncate_partial_clamps_dim_remaining(self):
        """After ``truncate_to_source`` clamps the elem-leaf, dim's
        ``remaining`` reflects the clamped count."""
        # gate[:, 3:4] over (128, 1792) with tile_size=(128, 512):
        #   BLOCK count=1 step=512, TILE count=1 step=512, ELEM count=256
        #   (clamped from 512). remainder_dims=(1,) -> partial-aware path:
        #   walked = (1-1)*512 + 256 = 256.
        es = (128, 1792)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_b = Axis(count=1, step=512, dim=1, label=AxisLabel.BLOCK)
        ax_t = Axis(count=1, step=512, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=256, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_b, ax_t, ax_e),
            cursor=0,
            n_batch_dims=0,
            remainder_dims=(1,),
        )
        assert g.remaining == (128, 256)

    def test_sharded_view_takes_normal_path(self):
        """Sharded / interleaved views never enter ``remainder_dims``;
        their ``count * step`` walks past sharded peers but the source
        extent itself is uniform -- the partial-aware path must not
        engage."""
        # 4 owned interleaved tiles of 128 each, peer step=256:
        # outer count=4 step=256, leaf count=128.
        # Without remainder_dims, walked = 4*256 = 1024 (full strided walk).
        es = (128, 1024)
        ax_p = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        ax_owned = Axis(count=4, step=256, dim=1, label=AxisLabel.TILE)
        ax_e = Axis(count=128, step=1, dim=1, label=AxisLabel.ELEM)
        g = Grid(
            element_shape=es,
            axes=(ax_p, ax_owned, ax_e),
            cursor=0,
            n_batch_dims=0,
        )
        # remainder_dims must be empty for sharded views.
        assert g.remainder_dims == ()
        assert g.remaining[1] == 1024


# ============================================================================
# Multi-P-tile trailing partial: a partition extent that spans several P-tiles
# with a short last one (e.g. 300 = 128 + 128 + 44). The partition axis folds
# its tiles into the free axis, so the P-tile index -- not layout offset -- is
# what identifies the trailing partial. These pin the folded-P descent and its
# symmetry with the free-axis trailing partial.
# ============================================================================


def _descend_tile(element_shape, tile_size, p_index, f_index):
    """Reproduce ``NDSlice.__getitem__`` int-descent for ``view[p_index, f_index]``
    at the Grid level (offsets live on Layout; this asserts Grid state only).

    On both axes the elements consumed to reach tile ``k`` are ``k * tile_size``:
    on the partition axis the tiles fold into free so a layout-offset query
    returns 0, and ``index * tile_p`` is what finds the addressable remainder.
    """
    g = Grid.from_shape(element_shape=element_shape, tile_size=tile_size)
    for dim, k in ((0, p_index), (1, f_index)):
        g = g.consume(dim)
        g = g.truncate_to_source(dim, k * tile_size[dim])
    g, _ = g.cleanup()
    return g


@pytest_marks(["neurotile"])
class TestFoldedPartitionTrailing:
    """A trailing partial P-tile narrows to its addressable partition extent
    (300 = 128 + 128 + 44). The extent is the load-bearing fact -- a compute /
    DMA op sizes off it -- and is consistent across trace and compile paths.
    """

    @pytest.mark.fast
    def test_full_p_tiles_report_full_extent(self):
        for pi in (0, 1):
            g = _descend_tile((300, 256), (128, 128), pi, 0)
            assert g.remaining == (128, 128), f"P-tile [{pi},0] expected full (128,128)"

    @pytest.mark.fast
    def test_trailing_p_tile_narrows_to_addressable_extent(self):
        g = _descend_tile((300, 256), (128, 128), 2, 0)
        # 300 - 2*128 = 44 addressable partition rows on the trailing tile.
        assert g.remaining[0] == 44

    @pytest.mark.fast
    def test_folded_partition_uses_tile_index_not_layout_offset(self):
        """The P-fold makes a layout-offset query 0, so the descent must use
        ``index * tile_p`` to find the addressable remainder. A second index
        (2 vs 1) must move the addressable extent."""
        g1 = _descend_tile((300, 256), (128, 128), 1, 0)
        g2 = _descend_tile((300, 256), (128, 128), 2, 0)
        assert g1.remaining[0] == 128  # full middle tile
        assert g2.remaining[0] == 44  # short trailing tile


@pytest_marks(["neurotile"])
class TestPartitionFreeAxisSymmetry:
    """A trailing partial narrows its addressable extent the same way on the
    partition axis and the free axis. Only the physical representation differs
    (the partition axis is capped at 128 rows and folds into free)."""

    @pytest.mark.fast
    def test_partition_trailing_extent_matches_free_trailing(self):
        # P-partial: (300,256) trailing tile [2,0] -> 44 partition rows.
        p_tile = _descend_tile((300, 256), (128, 128), 2, 0)
        # F-partial: (128,300) trailing tile [0,2] -> 44 free columns.
        f_tile = _descend_tile((128, 300), (128, 128), 0, 2)

        # Both narrow their trailing extent to the same addressable 44.
        assert p_tile.remaining[0] == 44
        assert f_tile.remaining[1] == 44

    @pytest.mark.fast
    def test_free_trailing_flags_is_remainder(self):
        # The free axis reliably flags is_remainder on its trailing partial.
        f_tile = _descend_tile((128, 300), (128, 128), 0, 2)
        assert f_tile.is_remainder is True
        assert 1 in f_tile.remainder_dims

    @pytest.mark.xfail(
        reason="Known asymmetry: is_remainder on a folded-P trailing tile is not "
        "set on the trace path (it is on the compile path). The addressable extent "
        "is correct on both; only the flag diverges. Remove xfail once the folded-P "
        "is_remainder is consistent with the free axis.",
        strict=False,
    )
    @pytest.mark.fast
    def test_partition_trailing_flags_is_remainder(self):
        # Desired (matches the free axis and the compile path): the folded-P
        # trailing tile flags is_remainder. Currently False on the trace path.
        p_tile = _descend_tile((300, 256), (128, 128), 2, 0)
        assert p_tile.is_remainder is True
        assert 0 in p_tile.remainder_dims
