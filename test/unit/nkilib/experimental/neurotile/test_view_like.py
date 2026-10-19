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

"""Contract for ``nt.view_like``: one buffer, an extent declared per use.

The extent comes from ``src`` and the memory from ``storage``, independently -- so a buffer
sized for the largest region can back a shorter one without that region's trailing partial
tile claiming to be full.

Pure Python -- no NKI tracer, no simulator, no device.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks, tiles, view_like

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks

_TILE_P = 128
_D = 128
_TILES_PER_BLK = 16


def _hbm_blocks(rows, tiles_per_blk=_TILES_PER_BLK):
    """A partition-axis block grid over HBM."""
    return blocks(MockTensor(rows, _D), tile_size=(_TILE_P, _D), block_size=(tiles_per_blk, 1))


def _slot(num_tiles=_TILES_PER_BLK):
    """Flat SBUF storage wide enough for ``num_tiles`` tiles of (128, D)."""
    return MockTensor(_TILE_P, num_tiles * _D)


def _tile_rows(view):
    """Partition rows each tile of ``view`` reports."""
    return [view[t, 0].element_shape[0] for t in range(view.shape[0])]


@pytest_marks(["neurotile"])
class TestViewLikeExtent:
    """The view takes its extent from ``src``, not from the storage backing it."""

    @pytest.mark.fast
    def test_full_block_matches_source(self):
        src = _hbm_blocks(2880)[0, 0]
        dst = view_like(src, _slot())
        assert dst.element_shape == src.element_shape == (2048, _D)
        assert dst.shape == src.shape == (16, 1)

    @pytest.mark.fast
    def test_short_block_on_a_full_size_slot(self):
        # 2880 rows in blocks of 2048: the trailing block covers 832 = 6 whole tiles + 64.
        src = _hbm_blocks(2880)[1, 0]
        dst = view_like(src, _slot())
        assert src.element_shape == (832, _D)
        # The extent follows src even though the storage could hold 2048 rows.
        assert dst.element_shape == (832, _D)
        assert dst.shape == (7, 1)

    @pytest.mark.fast
    def test_trailing_partial_tile_reports_its_own_rows(self):
        dst = view_like(_hbm_blocks(2880)[1, 0], _slot())
        assert _tile_rows(dst) == [128, 128, 128, 128, 128, 128, 64]

    @pytest.mark.fast
    def test_one_slot_serves_blocks_of_differing_extent(self):
        grid = _hbm_blocks(2880)
        slot = _slot()
        extents = [view_like(grid[b, 0], slot).element_shape[0] for b in range(grid.shape[0])]
        assert extents == [2048, 832]

    @pytest.mark.fast
    @pytest.mark.parametrize("rows, expect_tiles, expect_last", [(832, 7, 64), (2048, 16, 128), (129, 2, 1)])
    def test_extents_across_shapes(self, rows, expect_tiles, expect_last):
        src = tiles(MockTensor(rows, _D), tile_size=(_TILE_P, _D))
        dst = view_like(src, _slot())
        assert dst.shape == (expect_tiles, 1)
        assert dst.element_shape == (rows, _D)
        assert dst[expect_tiles - 1, 0].element_shape[0] == expect_last

    @pytest.mark.fast
    def test_regrouping_the_view_is_exact(self):
        # Re-tiling the view into smaller groups must not reach past the extent it declared.
        dst = view_like(_hbm_blocks(2880)[1, 0], _slot())
        groups = blocks(dst, block_size=(4, 1))
        assert groups.shape == (2, 1)
        assert groups[1, 0].shape == (3, 1)
        assert _tile_rows(groups[1, 0]) == [128, 128, 64]
