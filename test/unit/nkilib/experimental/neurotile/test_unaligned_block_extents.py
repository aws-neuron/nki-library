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

"""Extent contract for unaligned (non-tile-multiple) block grids.

Pins three facts, so none can regress silently:

1. A partial block, and each of its interior tiles, reports its own valid extent.
2. The emitted HBM access pattern stays inside the source tensor -- an
   overshooting AP is a silent OOB DMA, not a compile error.
3. ``Grid.remaining`` is the *walked* extent and may exceed the addressable
   region, which lives on the Layout. Pinned so nobody "fixes" the overshoot by
   making a DMA path trust ``remaining``.

Pure Python -- no NKI tracer, no simulator, no device.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core._helpers import contiguous_ap_pattern
from nkilib_src.nkilib.experimental.neurotile.core.ap_emitter import APEmitter
from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks, tiles
from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout
from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks

# (Sk, tile_f, tiles_per_blk) -> free-axis layouts, Sk on the free axis like fa2's K.
_FREE_AXIS_TILE = 512
_FREE_AXIS_BLK = 4
# Sk on the partition axis like fa2's V.
_PART_AXIS_TILE = 128
_PART_AXIS_BLK = 4


def _free_axis_blocks(Sk, D=128):
    """fa2's K layout: ``nt.blocks(k, tile_size=(D_TILE, sk_tile), block_size=(d_tiles, sk_tiles_per_blk))``."""
    src = MockTensor(D, Sk)
    d_tiles = D // 128
    return src, blocks(src, tile_size=(128, _FREE_AXIS_TILE), block_size=(d_tiles, _FREE_AXIS_BLK))


def _part_axis_blocks(Sk, D=128):
    """fa2's V layout: ``nt.blocks(v, tile_size=(SV_TILE, D), block_size=(sk_blk // SV_TILE, 1))``."""
    src = MockTensor(Sk, D)
    return src, blocks(src, tile_size=(_PART_AXIS_TILE, D), block_size=(_PART_AXIS_BLK, 1))


def _ap_max_offset(view):
    """Largest source element offset the view's HBM access pattern addresses."""
    grid, layout = view._grid, view._layout
    levels = APEmitter.emit(grid.axes, layout.strides, layout.addressable_extents(grid))
    span = 0
    for stride, count in levels:
        span += stride * (count - 1)
    return layout.offset + span


def _ap_element_count(view):
    """Number of elements the view's HBM access pattern transfers."""
    grid, layout = view._grid, view._layout
    levels = APEmitter.emit(grid.axes, layout.strides, layout.addressable_extents(grid))
    total = 1
    for _, count in levels:
        total = total * count
    return total


# ============================================================================
# 1. Partial blocks and their interior tiles report their own valid extent
# ============================================================================


@pytest_marks(["neurotile"])
class TestPartialBlockExtents:
    @pytest.mark.fast
    def test_free_axis_partial_tile_inside_partial_block(self):
        # Sk=2816 = 5 whole 512-tiles + 256. Blocks of 4 tiles -> block 0 full
        # (2048), block 1 holds 1 whole tile + 1 partial tile of 256.
        _, bv = _free_axis_blocks(2816)
        assert bv.shape == (1, 2)

        whole, partial = bv[0, 0], bv[0, 1]
        assert whole.element_shape == (128, 2048)
        assert partial.element_shape == (128, 768)  # 2816 - 2048

        assert whole.shape == (1, 4)
        assert partial.shape == (1, 2)
        # The last interior tile is the only partial one.
        assert partial[0, 0].element_shape == (128, 512)
        assert partial[0, 1].element_shape == (128, 256)

    @pytest.mark.fast
    def test_free_axis_whole_tiles_partial_block(self):
        # Sk=2560 = 5 whole tiles: the trailing block is short on *tile count*
        # but every tile in it is full.
        _, bv = _free_axis_blocks(2560)
        partial = bv[0, 1]
        assert partial.element_shape == (128, 512)
        assert partial.shape == (1, 1)
        assert partial[0, 0].element_shape == (128, 512)

    @pytest.mark.fast
    def test_partition_axis_partial_tile(self):
        # Sk=576 = 4 whole 128-tiles + 64, on the partition axis.
        _, bv = _part_axis_blocks(576)
        assert bv.shape == (2, 1)
        assert bv[0, 0].element_shape == (512, 128)
        assert bv[1, 0].element_shape == (64, 128)

    @pytest.mark.fast
    def test_aligned_block_grid_has_no_remainder(self):
        _, bv = _free_axis_blocks(4096)
        assert bv._grid.remainder_dims == ()
        assert bv.shape == (1, 2)
        for b in range(2):
            assert bv[0, b].element_shape == (128, 2048)


# ============================================================================
# 2. The emitted HBM access pattern stays in bounds
# ============================================================================


@pytest_marks(["neurotile"])
class TestPartialBlockAccessPatternInBounds:
    """An overshooting AP is a silent OOB DMA, so bound it explicitly."""

    @pytest.mark.fast
    @pytest.mark.parametrize("Sk", [2816, 2560, 3072, 4096])
    def test_free_axis_stream_steps_in_bounds(self, Sk):
        src, bv = _free_axis_blocks(Sk)
        n_elements = 128 * Sk
        stream = bv[0, :].stream(buffer_count=2)
        transferred = 0
        for i in range(stream.count):
            child = stream._child(i)
            assert _ap_max_offset(child) < n_elements, (
                "step " + str(i) + " of Sk=" + str(Sk) + " reads past the end of the source"
            )
            transferred += _ap_element_count(child)
        # Every element is visited exactly once: no gap, no double-read.
        assert transferred == n_elements

    @pytest.mark.fast
    @pytest.mark.parametrize("Sk", [576, 640, 1152, 512])
    def test_partition_axis_stream_steps_in_bounds(self, Sk):
        src, bv = _part_axis_blocks(Sk)
        n_elements = Sk * 128
        stream = bv[:, 0].stream(buffer_count=2)
        transferred = 0
        for i in range(stream.count):
            child = stream._child(i)
            assert _ap_max_offset(child) < n_elements
            transferred += _ap_element_count(child)
        assert transferred == n_elements

    @pytest.mark.fast
    def test_partial_block_ap_transfers_only_valid_elements(self):
        # The tail step must move 128 x 768, not 128 x 1024 (see remaining below).
        _, bv = _free_axis_blocks(2816)
        stream = bv[0, :].stream(buffer_count=2)
        assert _ap_element_count(stream._child(0)) == 128 * 2048
        assert _ap_element_count(stream._child(1)) == 128 * 768


# ============================================================================
# 3. remaining is the walked extent; the Layout owns the addressable bound
# ============================================================================


@pytest_marks(["neurotile"])
class TestWalkedVersusAddressableExtent:
    @pytest.mark.fast
    def test_remaining_overshoots_on_multi_tile_partial_block(self):
        """A ragged tail cannot be expressed as one leaf count, so the walk
        overshoots. Deliberate; the Layout clamps. See
        ``Grid._compute_remaining``."""
        _, bv = _free_axis_blocks(2816)
        child = bv[0, :].stream(buffer_count=2)._child(1)
        grid, layout = child._grid, child._layout

        # Walked extent: 2 tiles x 512.
        assert grid.remaining == (128, 1024)
        # Addressable: only 768 columns exist past the 2048-element offset.
        assert layout.addressable_extents(grid) == (128, 768)
        assert layout.sbuf_load_extents(grid) == (128, 768)
        # The public view attribute reports the valid extent, not the walk.
        assert child.element_shape == (128, 768)

    @pytest.mark.fast
    def test_remaining_agrees_when_the_tail_tiles_are_whole(self):
        """With no ragged tile the walk is exact, on either axis."""
        _, bv = _free_axis_blocks(2560)
        child = bv[0, :].stream(buffer_count=2)._child(1)
        assert child._grid.remaining == (128, 512)
        assert child._layout.addressable_extents(child._grid) == (128, 512)

        _, bv = _part_axis_blocks(640)
        child = bv[:, 0].stream(buffer_count=2)._child(1)
        assert child._grid.remaining == (128, 128)
        assert child._layout.addressable_extents(child._grid) == (128, 128)

    @pytest.mark.fast
    def test_sbuf_load_extents_never_exceed_addressable(self):
        """The invariant the DMA paths depend on, across a spread of shapes."""
        for Sk in (2048, 2560, 2816, 3072, 4096):
            _, bv = _free_axis_blocks(Sk)
            self._assert_extents_within_addressable(bv[0, :], Sk)
        for Sk in (512, 576, 640, 1152):
            _, bv = _part_axis_blocks(Sk)
            self._assert_extents_within_addressable(bv[:, 0], Sk)

    def _assert_extents_within_addressable(self, view, Sk):
        stream = view.stream(buffer_count=2)
        for i in range(stream.count):
            child = stream._child(i)
            grid, layout = child._grid, child._layout
            addressable = layout.addressable_extents(grid)
            sbuf = layout.sbuf_load_extents(grid)
            for d in range(grid.ndim):
                assert sbuf[d] <= addressable[d], (
                    "Sk=" + str(Sk) + " step " + str(i) + " dim " + str(d) + ": SBUF extent exceeds addressable"
                )


# ============================================================================
# 5. A region that no single access pattern can express is rejected
# ============================================================================


@pytest_marks(["neurotile"])
class TestRaggedTileRegionRejected:
    """A region spanning several tiles whose *last* tile is partial is not a
    rectangle, so one access pattern cannot describe it. On a free axis the
    merge pass makes it exact anyway; on the partition axis it cannot, and the
    pattern would round the tail tile up and read past the tensor. That case
    must fail at trace time -- silently emitting it is an out-of-bounds DMA.

    The check is a precondition asked only by the DMA paths, not part of
    ``ap`` itself, because ``NDSlice.__init__`` builds an ``ap`` for every view
    it constructs: describing such a region and descending into it is legal,
    only moving it in one DMA is not.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize("Sk", [704, 832, 1216])
    def test_partition_axis_ragged_multi_tile_block_is_rejected(self, Sk):
        # Trailing block = at least one full 128-row tile plus a partial one.
        _, bv = _part_axis_blocks(Sk)
        stream = bv[:, 0].stream(buffer_count=2)
        offenders = []
        for i in range(stream.count):
            child = stream._child(i)
            try:
                child._layout._assert_region_is_transferable(child._grid)
            except AssertionError as exc:
                offenders.append(i)
                assert "cannot be transferred as one DMA" in str(exc)
        assert offenders, "Sk=" + str(Sk) + ": expected the ragged trailing block to be rejected"

    @pytest.mark.fast
    @pytest.mark.parametrize("Sk", [512, 576, 640, 768, 896, 1152])
    def test_expressible_partition_axis_blocks_are_accepted(self, Sk):
        """Aligned, short-but-whole-tile, and lone-partial-tile blocks are all
        rectangles, so they must still go through."""
        _, bv = _part_axis_blocks(Sk)
        stream = bv[:, 0].stream(buffer_count=2)
        for i in range(stream.count):
            child = stream._child(i)
            child._layout._assert_region_is_transferable(child._grid)

    @pytest.mark.fast
    @pytest.mark.parametrize("Sk", [2560, 2816, 3072, 4096])
    def test_free_axis_ragged_tail_is_accepted(self, Sk):
        """The free axis merges the tile walk into the element walk and clamps
        it, so a ragged tail is expressible there and must not be rejected."""
        _, bv = _free_axis_blocks(Sk)
        stream = bv[0, :].stream(buffer_count=2)
        for i in range(stream.count):
            child = stream._child(i)
            child._layout._assert_region_is_transferable(child._grid)

    @pytest.mark.fast
    def test_ap_still_describes_the_region_it_refuses_to_transfer(self):
        """Constructing the pattern stays legal -- only the DMA is refused."""
        _, bv = _part_axis_blocks(704)
        child = bv[:, 0].stream(buffer_count=2)._child(1)
        child._layout.ap(child._grid)  # describing it is fine
        assert child.element_shape == (192, 128)
        # And its individual tiles are transferable, which is the documented fix.
        for t in range(child.shape[0]):
            tile = child[t, 0]
            tile._layout._assert_region_is_transferable(tile._grid)


# ============================================================================
# 6. slot_ap / matched_sbuf_ap: the shared load / stream-store pattern builder
# ============================================================================


@pytest_marks(["neurotile"])
class TestSlotAp:
    """``slot_ap`` covers the one case where a buffer's size is the library's
    choice, not the region's: a ``BlockStream`` slot, uniformly sized for a whole
    step while a trailing step may be partial.

    The *pattern* is asserted, not just the element count -- a same-count but
    differently-ordered walk would corrupt data with no size assert firing.
    """

    @pytest.mark.fast
    def test_element_count_matches_the_hbm_side(self):
        _, bv = _free_axis_blocks(2816)
        stream = bv[0, :].stream(buffer_count=2)
        for i in range(stream.count):
            child = stream._child(i)
            # A whole-step slot: 4 tiles of 512 columns on 128 partitions.
            ap = child._layout.matched_sbuf_ap(MockTensor(128, 2048), child._grid)
            sbuf_elements = 1
            for level in ap.pattern:
                sbuf_elements = sbuf_elements * level[1]
            assert sbuf_elements == _ap_element_count(child), (
                "step " + str(i) + ": SBUF and HBM element counts disagree"
            )

    @pytest.mark.fast
    def test_walks_only_the_valid_prefix_of_the_slot(self):
        _, bv = _free_axis_blocks(2816)
        child = bv[0, :].stream(buffer_count=2)._child(1)
        ap = child._layout.matched_sbuf_ap(MockTensor(128, 2048), child._grid)
        # Partition level walks the slot's full free width (the row stride);
        # the inner level covers only the 768 valid columns.
        assert ap.pattern == [[2048, 128], [1, 768]]

    @pytest.mark.fast
    def test_uses_the_whole_slot_on_an_exact_fit_step(self):
        _, bv = _free_axis_blocks(4096)
        child = bv[0, :].stream(buffer_count=2)._child(0)
        ap = child._layout.matched_sbuf_ap(MockTensor(128, 2048), child._grid)
        assert ap.pattern == [[2048, 128], [1, 2048]]

    @pytest.mark.fast
    def test_covers_a_step_short_on_tile_count(self):
        """A step can be short without being ``is_remainder``: every tile it
        owns is full, there are just fewer of them. It still needs the
        narrowing, so the slot walk cannot key off that flag."""
        _, bv = _part_axis_blocks(640)
        direct = bv[1, 0]
        assert direct.is_remainder is False
        assert direct.element_shape == (128, 128)
        ap = direct._layout.matched_sbuf_ap(MockTensor(128, 512), direct._grid)
        assert ap.pattern == [[512, 128], [1, 128]]

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "src_shape, tile, blk, sel",
        [
            # Partition-axis blocks: aligned, then short on tile count, then ragged.
            ((512, 128), (128, 128), (4, 1), (slice(None), 0)),
            ((1024, 128), (128, 128), (4, 1), (slice(None), 0)),
            ((640, 128), (128, 128), (4, 1), (slice(None), 0)),
            ((896, 128), (128, 128), (4, 1), (slice(None), 0)),
            ((576, 128), (128, 128), (4, 1), (slice(None), 0)),
            # Free-axis blocks, where the merge pass is in play.
            ((128, 4096), (128, 512), (1, 4), (0, slice(None))),
            ((128, 2560), (128, 512), (1, 4), (0, slice(None))),
            ((128, 2816), (128, 512), (1, 4), (0, slice(None))),
        ],
    )
    def test_reproduces_the_contiguous_walk_on_a_whole_step(self, src_shape, tile, blk, sel):
        """``BlockStream.store`` applies this on *every* step, not just partial
        ones. That is only safe if a whole step's pattern is unchanged, so pin
        the equivalence -- otherwise the unconditional call looks like a
        behaviour change and invites a conditional being reintroduced.
        """
        src = MockTensor(*src_shape)
        bv = blocks(src, tile_size=tile, block_size=blk)
        stream = bv[sel].stream(buffer_count=2)
        first = stream._child(0)
        slot = MockTensor(*first._layout.sbuf_shape_for(first._grid))
        contiguous = contiguous_ap_pattern(tuple(slot.shape))
        for i in range(stream.count):
            child = stream._child(i)
            matched = child._layout.matched_sbuf_ap(slot, child._grid).pattern
            whole_step = matched == contiguous
            region = 1
            for extent in child._layout.sbuf_load_extents(child._grid):
                region = region * extent
            slot_elements = 1
            for extent in slot.shape:
                slot_elements = slot_elements * extent
            # A step filling the whole slot must walk it exactly as the
            # contiguous pattern does; a short one must not.
            assert whole_step == (region == slot_elements), (
                "step "
                + str(i)
                + " of "
                + str(src_shape)
                + ": matched="
                + str(matched)
                + " contiguous="
                + str(contiguous)
                + " region="
                + str(region)
            )


# ============================================================================
# 4. Re-tiling an SBUF-backed view agrees with re-tiling in HBM
# ============================================================================


def _sbuf_tile_grid(num_tiles, tile_p=_PART_AXIS_TILE, D=128, rows=None):
    """An SBUF-resident tile grid of ``num_tiles`` tiles on the partition axis.

    Stands in for the view ``blocks(...)[i, 0].load()`` returns, without a tracer. Pass
    ``rows`` for a source whose last tile is partial.
    """
    sbuf_data = MockTensor(tile_p, num_tiles * D)
    element_shape = (num_tiles * tile_p if rows is None else rows, D)
    grid, layout = SBUFLayout.build_view(sbuf_data, element_shape, (tile_p, D), "float32", "sbuf")
    return NDSlice(grid, layout)


@pytest_marks(["neurotile"])
class TestRetileMatchesAcrossBuffers:
    """Grouping tiles into blocks reports the same extents in SBUF as in HBM.

    The two layouts address a partition dim differently, so this pins them to one answer:
    a group that covers only part of the source must report what it covers, in either.
    """

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "num_tiles, group, expect_shape, expect_rows",
        [
            (7, 4, (3, 1), 384),  # 2 groups: 4 tiles, then 3
            (6, 4, (2, 1), 256),  # 2 groups: 4 tiles, then 2
            (5, 2, (1, 1), 128),  # 3 groups: 2, 2, then 1
            (9, 4, (1, 1), 128),  # 3 groups: 4, 4, then 1
        ],
    )
    def test_trailing_group_reports_covered_extent(self, num_tiles, group, expect_shape, expect_rows):
        groups = blocks(_sbuf_tile_grid(num_tiles), block_size=(group, 1))
        last = groups[groups.shape[0] - 1, 0]
        assert last.shape == expect_shape
        assert last.element_shape == (expect_rows, 128)

    @pytest.mark.fast
    @pytest.mark.parametrize("num_tiles", [5, 6, 7, 9, 11])
    def test_sbuf_and_hbm_agree(self, num_tiles):
        # Same tile structure, one still in HBM and one already loaded.
        hbm = tiles(MockTensor(num_tiles * _PART_AXIS_TILE, 128), tile_size=(_PART_AXIS_TILE, 128))
        sbuf = _sbuf_tile_grid(num_tiles)
        hbm_groups = blocks(hbm, block_size=(_PART_AXIS_BLK, 1))
        sbuf_groups = blocks(sbuf, block_size=(_PART_AXIS_BLK, 1))
        assert sbuf_groups.shape == hbm_groups.shape
        for i in range(hbm_groups.shape[0]):
            assert sbuf_groups[i, 0].shape == hbm_groups[i, 0].shape
            assert sbuf_groups[i, 0].element_shape == hbm_groups[i, 0].element_shape

    @pytest.mark.fast
    def test_padded_tile_of_partial_group_is_rejected(self):
        # 7 tiles in groups of 4: the second group holds 3, so tile 3 does not exist.
        partial = blocks(_sbuf_tile_grid(7), block_size=(4, 1))[1, 0]
        assert partial.shape == (3, 1)
        with pytest.raises(AssertionError, match="out of range"):
            partial[3, 0]


# ============================================================================
# 5. A trailing partial tile survives one level of descent
# ============================================================================


@pytest_marks(["neurotile"])
class TestTrailingPartialSurvivesDescent:
    """A view over several tiles keeps its leaf full width and resolves the trailing partial
    per tile at descent, so descent must know how much source the view skipped.

    A slice or block group that measures its indices against the whole source reports its
    short trailing tile as full, which for a matmul means contracting over rows that were
    never written.
    """

    @pytest.mark.fast
    def test_slice_reports_covered_not_walked_extent(self):
        # 7 tiles over 832 rows: tiles 0-5 hold 128, tile 6 holds 64.
        buf = _sbuf_tile_grid(7, rows=832)
        assert buf.element_shape == (832, 128)
        assert buf[6, 0].element_shape == (64, 128)
        # Tiles 4..6 cover 128 + 128 + 64 = 320, not 3 * 128.
        assert buf[4:7, 0].element_shape == (320, 128)

    @pytest.mark.fast
    def test_partial_tile_is_visible_through_a_slice(self):
        buf = _sbuf_tile_grid(7, rows=832)
        assert buf[4:7, 0][2, 0].element_shape == (64, 128)

    @pytest.mark.fast
    def test_partial_tile_is_visible_through_a_block_group(self):
        groups = blocks(_sbuf_tile_grid(7, rows=832), block_size=(4, 1))
        assert groups.shape == (2, 1)
        last = groups[1, 0]
        assert last.shape == (3, 1)
        assert last.element_shape == (320, 128)
        rows = [last[t, 0].element_shape[0] for t in range(last.shape[0])]
        assert rows == [128, 128, 64]

    @pytest.mark.fast
    @pytest.mark.parametrize("rows, expect_last", [(832, 64), (896, 128), (768, 128), (900, 4)])
    def test_last_tile_rows_across_extents(self, rows, expect_last):
        buf = _sbuf_tile_grid((rows + 127) // 128, rows=rows)
        last = buf.shape[0] - 1
        assert buf[last, 0].element_shape[0] == expect_last

    @pytest.mark.fast
    def test_slice_of_an_oversized_buffer_cannot_be_retiled_past_its_end(self):
        """A slice bounds re-tiling of itself, so a uniform buffer can back a shorter region.

        7 filled tiles of a 16-tile buffer, handed back as ``buf[0:7, 0]``, must regroup into
        4 and 3 -- otherwise a consumer walks tiles the slice excluded.
        """
        buf = _sbuf_tile_grid(16)
        view = buf[0:7, 0]
        assert view.shape == (7, 1)
        groups = blocks(view, block_size=(4, 1))
        assert groups.shape == (2, 1)
        assert groups[0, 0].shape == (4, 1)
        assert groups[1, 0].shape == (3, 1)
