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
Unit tests for TiledRange and TiledRangeIterator classes.
"""

import pytest
from nkilib_src.nkilib.core.utils.tiled_range import (
    TiledRange,
    TiledRangeIterator,
)


class TestTiledRangeIterator:
    """Tests for TiledRangeIterator class."""

    def test_basic_properties(self):
        """Test basic properties of TiledRangeIterator."""
        iterator = TiledRangeIterator(tile_size=128, tile_index=0, start_offset=0, end_offset=128)
        assert iterator.size == 128
        assert iterator.index == 0
        assert iterator.start_offset == 0
        assert iterator.end_offset == 128

    def test_different_values(self):
        """Test TiledRangeIterator with different values."""
        iterator = TiledRangeIterator(tile_size=64, tile_index=2, start_offset=256, end_offset=320)
        assert iterator.size == 64
        assert iterator.index == 2
        assert iterator.start_offset == 256
        assert iterator.end_offset == 320

    def test_repr(self):
        """Test string representation."""
        iterator = TiledRangeIterator(tile_size=128, tile_index=1, start_offset=128, end_offset=256)
        repr_str = repr(iterator)
        assert "TiledRangeIterator" in repr_str
        assert "size=128" in repr_str
        assert "index=1" in repr_str
        assert "start_offset=128" in repr_str
        assert "end_offset=256" in repr_str


class TestTiledRangeBasic:
    """Basic tests for TiledRange function."""

    def test_exact_division(self):
        """Test when size divides evenly by tile_size."""
        tiles = TiledRange(256, 128)
        assert len(tiles) == 2

        # Check first tile
        assert tiles[0].size == 128
        assert tiles[0].index == 0
        assert tiles[0].start_offset == 0
        assert tiles[0].end_offset == 128

        # Check second tile
        assert tiles[1].size == 128
        assert tiles[1].index == 1
        assert tiles[1].start_offset == 128
        assert tiles[1].end_offset == 256

    def test_non_exact_division(self):
        """Test when size does not divide evenly by tile_size (the 300, 128 example)."""
        tiles = TiledRange(300, 128)
        assert len(tiles) == 3

        # Check first tile
        assert tiles[0].size == 128
        assert tiles[0].index == 0
        assert tiles[0].start_offset == 0
        assert tiles[0].end_offset == 128

        # Check second tile
        assert tiles[1].size == 128
        assert tiles[1].index == 1
        assert tiles[1].start_offset == 128
        assert tiles[1].end_offset == 256

        # Check third tile (partial)
        assert tiles[2].size == 44
        assert tiles[2].index == 2
        assert tiles[2].start_offset == 256
        assert tiles[2].end_offset == 300

    def test_single_tile(self):
        """Test when size is smaller than tile_size."""
        tiles = TiledRange(50, 128)
        assert len(tiles) == 1
        assert tiles[0].size == 50
        assert tiles[0].index == 0
        assert tiles[0].start_offset == 0
        assert tiles[0].end_offset == 50

    def test_iteration(self):
        """Test iterating over TiledRange tuple."""
        tiles = TiledRange(300, 128)
        tiles_list = list(tiles)

        assert len(tiles_list) == 3
        assert tiles_list[0].size == 128
        assert tiles_list[1].size == 128
        assert tiles_list[2].size == 44

        # Check indices
        for i, tile in enumerate(tiles):
            assert tile.index == i

    def test_small_tiles(self):
        """Test with small tile sizes."""
        tiles = TiledRange(100, 10)
        assert len(tiles) == 10

        for i, tile in enumerate(tiles):
            assert tile.size == 10
            assert tile.index == i
            assert tile.start_offset == i * 10
            assert tile.end_offset == (i + 1) * 10


class TestTiledRangeNested:
    """Tests for nested TiledRange functionality."""

    def test_nested_exact_division(self):
        """Test nested tiling with exact division."""
        outer_tiles = TiledRange(256, 128)

        # Tile the first outer tile (offset 0)
        inner_tiles = TiledRange(outer_tiles[0], 64)
        assert len(inner_tiles) == 2
        assert inner_tiles[0].size == 64
        assert inner_tiles[0].index == 0
        assert inner_tiles[0].start_offset == 0  # 0 + 0
        assert inner_tiles[0].end_offset == 64  # 0 + 64
        assert inner_tiles[1].size == 64
        assert inner_tiles[1].index == 1
        assert inner_tiles[1].start_offset == 64  # 0 + 64
        assert inner_tiles[1].end_offset == 128  # 0 + 128

        # Tile the second outer tile (offset 128)
        inner_tiles2 = TiledRange(outer_tiles[1], 64)
        assert len(inner_tiles2) == 2
        assert inner_tiles2[0].size == 64
        assert inner_tiles2[0].index == 0
        assert inner_tiles2[0].start_offset == 128  # 128 + 0
        assert inner_tiles2[0].end_offset == 192  # 128 + 64
        assert inner_tiles2[1].size == 64
        assert inner_tiles2[1].index == 1
        assert inner_tiles2[1].start_offset == 192  # 128 + 64
        assert inner_tiles2[1].end_offset == 256  # 128 + 128

    def test_nested_non_exact_division(self):
        """Test nested tiling with non-exact division."""
        outer_tiles = TiledRange(300, 128)

        # Tile the last outer tile (size 44)
        inner_tiles = TiledRange(outer_tiles[2], 20)
        assert len(inner_tiles) == 3
        assert inner_tiles[0].size == 20
        assert inner_tiles[0].index == 0
        assert inner_tiles[0].start_offset == 256  # 256 + 0
        assert inner_tiles[0].end_offset == 276  # 256 + 20
        assert inner_tiles[1].size == 20
        assert inner_tiles[1].index == 1
        assert inner_tiles[1].start_offset == 276  # 256 + 20
        assert inner_tiles[1].end_offset == 296  # 256 + 40
        assert inner_tiles[2].size == 4
        assert inner_tiles[2].index == 2
        assert inner_tiles[2].start_offset == 296  # 256 + 40
        assert inner_tiles[2].end_offset == 300  # 256 + 44

    def test_nested_loop_pattern(self):
        """Test the nested loop pattern described in the requirements."""
        outer_tiles = TiledRange(300, 128)

        results = []
        for outer_tile in outer_tiles:
            inner_tiles = TiledRange(outer_tile, 64)
            for inner_tile in inner_tiles:
                results.append(
                    {
                        "outer_idx": outer_tile.index,
                        "outer_size": outer_tile.size,
                        "inner_idx": inner_tile.index,
                        "inner_size": inner_tile.size,
                        "inner_offset": inner_tile.start_offset,
                        "inner_end_offset": inner_tile.end_offset,
                    }
                )

        # First outer tile (128) should have 2 inner tiles (64, 64)
        assert results[0]["outer_idx"] == 0
        assert results[0]["inner_idx"] == 0
        assert results[0]["inner_size"] == 64
        assert results[0]["inner_offset"] == 0
        assert results[0]["inner_end_offset"] == 64
        assert results[1]["outer_idx"] == 0
        assert results[1]["inner_idx"] == 1
        assert results[1]["inner_size"] == 64
        assert results[1]["inner_offset"] == 64
        assert results[1]["inner_end_offset"] == 128

        # Second outer tile (128) should have 2 inner tiles (64, 64)
        assert results[2]["outer_idx"] == 1
        assert results[2]["inner_idx"] == 0
        assert results[2]["inner_size"] == 64
        assert results[2]["inner_offset"] == 128
        assert results[2]["inner_end_offset"] == 192
        assert results[3]["outer_idx"] == 1
        assert results[3]["inner_idx"] == 1
        assert results[3]["inner_size"] == 64
        assert results[3]["inner_offset"] == 192
        assert results[3]["inner_end_offset"] == 256

        # Third outer tile (44) should have 1 inner tile (44)
        assert results[4]["outer_idx"] == 2
        assert results[4]["inner_idx"] == 0
        assert results[4]["inner_size"] == 44
        assert results[4]["inner_offset"] == 256
        assert results[4]["inner_end_offset"] == 300

        assert len(results) == 5


class TestTiledRangeEdgeCases:
    """Tests for edge cases."""

    def test_size_equals_tile_size(self):
        """Test when size equals tile_size."""
        tiles = TiledRange(128, 128)
        assert len(tiles) == 1
        assert tiles[0].size == 128
        assert tiles[0].index == 0
        assert tiles[0].start_offset == 0
        assert tiles[0].end_offset == 128

    def test_size_one_more_than_tile_size(self):
        """Test when size is one more than tile_size."""
        tiles = TiledRange(129, 128)
        assert len(tiles) == 2
        assert tiles[0].size == 128
        assert tiles[0].end_offset == 128
        assert tiles[1].size == 1
        assert tiles[1].end_offset == 129

    def test_very_small_size(self):
        """Test with very small size."""
        tiles = TiledRange(1, 128)
        assert len(tiles) == 1
        assert tiles[0].size == 1
        assert tiles[0].end_offset == 1

    def test_large_number_of_tiles(self):
        """Test with a large number of tiles."""
        tiles = TiledRange(10000, 100)
        assert len(tiles) == 100
        assert tiles[0].size == 100
        assert tiles[99].size == 100
        assert tiles[99].start_offset == 9900
        assert tiles[99].end_offset == 10000

    def test_tile_size_one(self):
        """Test with tile_size of 1."""
        tiles = TiledRange(5, 1)
        assert len(tiles) == 5
        for i in range(5):
            assert tiles[i].size == 1
            assert tiles[i].index == i
            assert tiles[i].start_offset == i
            assert tiles[i].end_offset == i + 1


class TestTiledRangeTupleOperations:
    """Tests for tuple operations on TiledRange results."""

    def test_tuple_indexing(self):
        """Test indexing into TiledRange tuple."""
        tiles = TiledRange(300, 128)
        tile0 = tiles[0]
        tile1 = tiles[1]
        tile2 = tiles[2]

        assert tile0.size == 128
        assert tile1.size == 128
        assert tile2.size == 44

    def test_tuple_len(self):
        """Test len() function on TiledRange tuple."""
        assert len(TiledRange(300, 128)) == 3
        assert len(TiledRange(256, 128)) == 2
        assert len(TiledRange(100, 128)) == 1

    def test_tuple_is_tuple(self):
        """Test that TiledRange returns a tuple."""
        tiles = TiledRange(300, 128)
        assert isinstance(tiles, tuple)

    def test_tuple_unpacking(self):
        """Test tuple unpacking."""
        tile0, tile1 = TiledRange(256, 128)
        assert tile0.size == 128
        assert tile1.size == 128


class TestTiledRangeComprehensive:
    """Comprehensive integration tests."""

    def test_example_from_requirements(self):
        """Test the exact example from the requirements: TiledRange(300, 128)."""
        tiles = TiledRange(300, 128)

        sizes = [tile.size for tile in tiles]
        indices = [tile.index for tile in tiles]
        start_offsets = [tile.start_offset for tile in tiles]
        end_offsets = [tile.end_offset for tile in tiles]

        assert sizes == [128, 128, 44]
        assert indices == [0, 1, 2]
        assert start_offsets == [0, 128, 256]
        assert end_offsets == [128, 256, 300]

    def test_multiple_nesting_levels(self):
        """Test multiple levels of nesting."""
        level1_tiles = TiledRange(1000, 256)

        for l1_tile in level1_tiles:
            level2_tiles = TiledRange(l1_tile, 64)
            for l2_tile in level2_tiles:
                level3_tiles = TiledRange(l2_tile, 16)
                # Just verify it works without errors
                assert len(level3_tiles) > 0
                for l3_tile in level3_tiles:
                    assert l3_tile.size > 0
                    assert l3_tile.size <= 16

    def test_consistency_across_iterations(self):
        """Test that multiple iterations produce consistent results."""
        tiles = TiledRange(300, 128)

        first_pass = [(t.size, t.index, t.start_offset) for t in tiles]
        second_pass = [(t.size, t.index, t.start_offset) for t in tiles]

        assert first_pass == second_pass

    def test_various_sizes_and_tile_sizes(self):
        """Test various combinations of sizes and tile_sizes."""
        test_cases = [
            (100, 10, 10),  # Exact division
            (105, 10, 11),  # Non-exact division
            (1, 100, 1),  # Size < tile_size
            (1000, 1, 1000),  # tile_size = 1
            (512, 128, 4),  # Power of 2
            (1023, 128, 8),  # Almost power of 2
        ]

        for size, tile_size, expected_tiles in test_cases:
            tiles = TiledRange(size, tile_size)
            assert len(tiles) == expected_tiles, f"Failed for size={size}, tile_size={tile_size}"

            # Verify all tiles are accounted for
            total_covered = sum(tile.size for tile in tiles)
            assert total_covered == size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
