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

"""Tests for remainder block streaming geometry.

Pure Python -- no NKI, no tracer, no device required. Covers ``block_table`` and the block partition
``remainder_block_stream`` derives from it, which is what a kernel walks and narrows against.
"""

import pytest
from nkilib_src.nkilib.experimental.attention.flash_attention2_trn3 import block_table


def _partition(extent, tile, tiles_per_blk, align=None):
    """Mirror the split remainder_block_stream performs, returning its block table."""
    grain = tile if align is None else align
    remainder = extent % grain
    table = block_table(extent - remainder, tile, tiles_per_blk)
    if remainder > 0:
        table.append((1, remainder))
    return table


def _covered(table, tile):
    """Total elements the table accounts for, so a partition can be checked against the extent."""
    return sum((count - 1) * tile + last for count, last in table)


class TestBlockTable:
    @pytest.mark.parametrize(
        "extent, expected",
        [
            (2048, [(4, 512)]),  # exactly one whole block
            (4096, [(4, 512), (4, 512)]),  # two whole blocks
            (896, [(2, 384)]),  # two tiles in one block, second partial
            (512, [(1, 512)]),  # single whole tile
            (384, [(1, 384)]),  # single partial tile
            (2560, [(4, 512), (1, 512)]),  # whole block then a lone whole tile
            (0, []),  # nothing at all
        ],
    )
    def test_groups_tiles_the_way_blocks_does(self, extent, expected):
        assert block_table(extent, 512, 4) == expected

    @pytest.mark.parametrize("extent", [128, 512, 896, 1000, 2048, 2560, 2880, 3000, 43136, 43200])
    def test_table_covers_the_extent_exactly(self, extent):
        assert _covered(block_table(extent, 512, 4), 512) == extent

    def test_only_the_final_block_may_hold_a_partial_tile(self):
        table = block_table(2880, 512, 4)
        for entry in table[:-1]:
            assert entry[1] == 512, "a non-final block must end on a whole tile"
        assert table[-1] == (2, 320), "2880 is five whole 512 tiles plus 320"


class TestRemainderPartition:
    @pytest.mark.parametrize(
        "extent, expected",
        [
            (2048, [(4, 512)]),  # aligned: no remainder block
            (2880, [(4, 512), (2, 256), (1, 64)]),  # 2048 + 768 whole, then a 64-row tail
            (1000, [(2, 384), (1, 104)]),  # 896 is two tiles in one block, then 104
            (200, [(1, 128), (1, 72)]),  # one whole 128 tile, then 72
            (43200, [(4, 512)] * 21 + [(1, 128), (1, 64)]),  # the large target shape
        ],
    )
    def test_partition_at_the_tile_granule(self, extent, expected):
        assert _partition(extent, 512, 4, align=128) == expected

    @pytest.mark.parametrize("extent", [64, 100, 128, 200, 384, 1000, 2880, 3000, 43200])
    def test_partition_covers_the_extent_exactly(self, extent):
        assert _covered(_partition(extent, 512, 4, align=128), 512) == extent

    @pytest.mark.parametrize("extent", [64, 100, 200, 1000, 2880, 3000, 43200])
    def test_no_block_mixes_whole_and_partial_granules(self, extent):
        """The property the split exists to guarantee: no block is a non-rectangle on the partition axis."""
        for count, last in _partition(extent, 512, 4, align=128):
            assert count == 1 or last % 128 == 0, f"block ({count}, {last}) mixes whole and partial granules"

    @pytest.mark.parametrize("extent", [64, 100, 127])
    def test_sub_granule_extent_is_remainder_only(self, extent):
        """Below one granule there is no body to stream, so the table is the lone remainder block."""
        assert _partition(extent, 512, 4, align=128) == [(1, extent)]

    def test_align_defaults_to_the_streamed_tile(self):
        """Without an explicit align the cut is at the last whole tile, not the last whole granule."""
        assert _partition(1000, 512, 4) == [(1, 512), (1, 488)]
        assert _partition(1000, 512, 4, align=128) == [(2, 384), (1, 104)]
