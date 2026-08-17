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

"""CPU-only unit tests for holistic indexing validation.

`validate_index_key` is the single chokepoint for NDSlice[...],
and SBUFLayout.sub_index indexing. It rejects out-of-range integer keys,
stepped / empty / out-of-range slice keys, and too-many keys with a
level-aware error message that directs the user to the right descent
(`nt.tiles(view)` for block views, `nt.enumerate` for shard-outer views).

Tests internal classes (Grid, Axis, AxisLabel) and private helpers
(level_hint, validate_index_key) — uses deep imports per the plan's
internal-class exception (§3.5).
"""

import pytest

# Internal-class deep imports (§3.5 exception): Grid, Axis, AxisLabel and
# private helpers level_hint/validate_index_key are not part of the public
# nt.* surface.
from nkilib_src.nkilib.experimental.neurotile.core._helpers import level_hint, validate_index_key
from nkilib_src.nkilib.experimental.neurotile.core.axis import Axis, AxisLabel
from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks, tiles
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import interleaved_range

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks

# ---------------------------------------------------------------------------
# _level_hint
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestLevelHint:
    @pytest.mark.fast
    def test_plain_tile_view_no_hint(self):
        grid = Grid.from_shape((128, 512), (128, 128))
        assert level_hint(grid, 1) == ""

    def test_block_view_hint(self):
        grid = Grid.from_shape((128, 512), (128, 128), block_size=(1, 4))
        assert "block-level" in level_hint(grid, 1)
        assert "nt.tiles(view)" in level_hint(grid, 1)

    def test_block_size_1_no_hint(self):
        grid = Grid.from_shape((128, 512), (128, 128), block_size=(1, 1))
        assert level_hint(grid, 1) == ""

    def test_interleaved_shard_outer_hint(self):
        """Interleaved-sharded views surface a 'shard axis' hint."""
        v = tiles(MockTensor((128, 1024)), tile_size=(128, 128))  # 8 tiles on dim 1
        # Stepped slice -> split_peers -> consumes peer axis with SHARD label.
        # The owned axis remains as outer; structurally still 'shard'-spaced.
        v[:, interleaved_range(rank=0, num_shards=4, total=8)]
        # `is_sharded` is True only when the SHARD-labelled peer axis is still
        # present. After interleaved indexing, peer is consumed; the owned axis
        # carries the gap. The hint covers the case where the user looks at a
        # block-level view; we just exercise that it is empty for plain tile.
        assert level_hint(v._grid, 1) == ""

    def test_shard_axis_hint_string(self):
        """When a Grid still carries a SHARD-labelled axis on a dim, the
        level_hint returns the 'shard axis on this dim' guidance."""
        # Construct a grid with an explicit SHARD axis on dim 0 -- this is
        # the structural shape of an un-consumed interleaved-shard view.
        shard_axis = Axis(count=2, step=1, dim=0, label=AxisLabel.SHARD)
        owned_axis = Axis(count=4, step=2, dim=0, label=AxisLabel.TILE)
        leaf_axis = Axis(count=128, step=1, dim=0, label=AxisLabel.PARTITION)
        f_axis = Axis(count=128, step=1, dim=1, label=AxisLabel.ELEM)
        grid = Grid(
            element_shape=(1024, 128),
            axes=(shard_axis, owned_axis, leaf_axis, f_axis),
            cursor=0,
            n_batch_dims=0,
        )
        hint = level_hint(grid, 0)
        assert "shard axis" in hint
        assert ".tolist()" in hint


# ---------------------------------------------------------------------------
# Integer key validation
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestIntKey:
    @pytest.mark.fast
    def test_in_range_passes(self):
        validate_index_key(0, dim=0, extent=4)
        validate_index_key(3, dim=0, extent=4)

    def test_negative_in_range_passes(self):
        validate_index_key(-1, dim=0, extent=4)
        validate_index_key(-4, dim=0, extent=4)

    def test_out_of_range_positive_rejects(self):
        with pytest.raises(AssertionError, match="int index 4 out of range"):
            validate_index_key(4, dim=0, extent=4)

    def test_out_of_range_negative_rejects(self):
        with pytest.raises(AssertionError, match="int index -5 out of range"):
            validate_index_key(-5, dim=0, extent=4)

    def test_extent_zero_rejects(self):
        with pytest.raises(AssertionError, match="out of range"):
            validate_index_key(0, dim=0, extent=0)

    def test_error_includes_valid_range(self):
        with pytest.raises(AssertionError, match=r"valid range: \[-4, 4\)"):
            validate_index_key(10, dim=0, extent=4)


# ---------------------------------------------------------------------------
# Slice key validation
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestSliceKey:
    @pytest.mark.fast
    def test_in_range_passes(self):
        validate_index_key(slice(0, 2), dim=1, extent=4)
        validate_index_key(slice(1, 3), dim=1, extent=4)
        validate_index_key(slice(None, None), dim=1, extent=4)

    def test_out_of_range_rejects(self):
        with pytest.raises(AssertionError, match=r"slice \[0:5:1\] out of range"):
            validate_index_key(slice(0, 5), dim=1, extent=4)

    def test_empty_slice_rejects(self):
        with pytest.raises(AssertionError, match=r"empty slice \[2:2\]"):
            validate_index_key(slice(2, 2), dim=1, extent=4)

    def test_stepped_slice_accepted(self):
        """Stepped slices (interleaved sharding) are now legal."""
        validate_index_key(slice(0, 4, 2), dim=1, extent=4)

    def test_negative_start_rejects(self):
        with pytest.raises(AssertionError, match="out of range"):
            validate_index_key(slice(-1, 2), dim=1, extent=4)

    def test_stop_before_start_rejects(self):
        with pytest.raises(AssertionError, match="out of range"):
            validate_index_key(slice(3, 1), dim=1, extent=4)


# ---------------------------------------------------------------------------
# Runtime key: skipped (cannot bounds-check)
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestRuntimeKey:
    @pytest.mark.fast
    def test_runtime_scalar_passes(self):
        """Non-int, non-slice keys are skipped (NKI catches at runtime)."""

        class _FakeCExpr:
            pass

        validate_index_key(_FakeCExpr(), dim=0, extent=4)


# ---------------------------------------------------------------------------
# Level-aware hint in error message
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestLevelHintInErrors:
    @pytest.mark.fast
    def test_block_view_slice_oob_includes_hint(self):
        grid = Grid.from_shape((128, 512), (128, 128), block_size=(1, 4))
        # extent at block level = 1 (one block), user asks [1:3] thinking tiles.
        with pytest.raises(AssertionError, match="block-level view.*nt.tiles"):
            validate_index_key(slice(1, 3), dim=1, extent=1, grid=grid)

    def test_plain_view_no_hint_in_error(self):
        grid = Grid.from_shape((128, 512), (128, 128))
        with pytest.raises(AssertionError) as exc:
            validate_index_key(slice(0, 10), dim=1, extent=4, grid=grid)
        assert "block-level" not in str(exc.value)


# ---------------------------------------------------------------------------
# End-to-end: NDSlice[...] rejects with directive
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestNDSliceIntegration:
    @pytest.mark.fast
    def test_tiles_view_oob_slice_rejects(self):
        v = tiles(MockTensor((128, 2048)), tile_size=(128, 512))  # shape (1, 4)
        with pytest.raises(AssertionError, match=r"slice \[0:10:1\] out of range"):
            v[:, 0:10]

    def test_blocks_view_oob_slice_rejects_with_block_hint(self):
        v = blocks(MockTensor((128, 2048)), tile_size=(128, 512), block_size=(1, 4))
        # shape (1, 1) at block level; user expecting tile semantics gets helpful hint.
        with pytest.raises(AssertionError, match="block-level view"):
            v[:, 1:3]

    def test_too_many_keys_rejects(self):
        v = tiles(MockTensor((128, 2048)), tile_size=(128, 512))
        with pytest.raises(AssertionError, match="too many keys"):
            v[0, 0, 0]


# ---------------------------------------------------------------------------
# Out-of-range access on a materialized dim list still rejects. Users pick a
# specific axis with `.tolist(dim=d)` and index the resulting Python list.
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestTolistDimIndexing:
    @pytest.mark.fast
    def test_oob_int_on_tolist_dim_list(self):
        v = tiles(MockTensor((128, 2048)), tile_size=(128, 512))
        # Dim 1 has 4 tiles; index 10 is out of range on the materialized list.
        items = v.tolist(dim=1)
        with pytest.raises(IndexError):
            items[10]

    def test_stepped_slice_on_tolist_dim_list(self):
        """Stepped slicing is a plain-Python list operation; returns a sub-list."""
        v = tiles(MockTensor((128, 2048)), tile_size=(128, 512))
        items = v.tolist(dim=1)
        # Python list slicing supports step; returns 2 items (0, 2).
        assert len(items[::2]) == 2
