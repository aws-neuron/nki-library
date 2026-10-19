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

"""Host-side tests for the pure-Python TiledRange helper.

TiledRange is plain Python (ceil-division tiling with last-tile clipping), so
these tests call it directly and run under the simulator with no device kernel
or marker.
"""

import pytest
from nkilib_src.nkilib.core.utils.tiled_range import TiledRange, TiledRangeIterator


@pytest.mark.fast
def test_tiled_range_int_size_with_remainder():
    """Integer size arg (L85 false arc -> L90) plus a clipped last tile (L98 loop body)."""
    tiles = TiledRange(300, 128)
    assert len(tiles) == 3
    assert [(t.size, t.index, t.start_offset, t.end_offset) for t in tiles] == [
        (128, 0, 0, 128),
        (128, 1, 128, 256),
        (44, 2, 256, 300),  # last tile clipped to the remainder
    ]


@pytest.mark.fast
def test_tiled_range_exact_multiple_no_clip():
    """Evenly divisible size: every tile is full size (no last-tile clip)."""
    tiles = TiledRange(256, 128)
    assert len(tiles) == 2
    assert all(t.size == 128 for t in tiles)
    assert tiles[-1].end_offset == 256


@pytest.mark.fast
def test_tiled_range_nested_iterator_size():
    """Nested TiledRangeIterator size arg (L85 true arc -> L87) uses its size/offset."""
    outer = TiledRange(300, 128)
    # Third outer tile: size=44, start_offset=256.
    outer_tile = outer[2]
    assert isinstance(outer_tile, TiledRangeIterator)

    inner = TiledRange(outer_tile, 32)
    assert len(inner) == 2  # ceil(44 / 32)
    # start_offset must be absolute (base_offset 256 added).
    assert [(t.size, t.index, t.start_offset, t.end_offset) for t in inner] == [
        (32, 0, 256, 288),
        (12, 1, 288, 300),  # remainder tile, still absolute-offset
    ]


@pytest.mark.fast
def test_tiled_range_single_tile_larger_than_size():
    """tile_size >= size -> exactly one clipped tile; exercises the loop-then-return path."""
    tiles = TiledRange(50, 128)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.size, t.index, t.start_offset, t.end_offset) == (50, 0, 0, 50)


@pytest.mark.fast
def test_tiled_range_iterator_repr():
    it = TiledRangeIterator(44, 2, 256, 300)
    assert repr(it) == "TiledRangeIterator(size=44, index=2, start_offset=256, end_offset=300)"
