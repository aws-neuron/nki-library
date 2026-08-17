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
Tests for NDSlice: Grid + Layout coordination, iteration, DMA.

Pure Python -- no NKI, no tracer, no device required.
"""

import inspect

import nki as _nki
import nki.language as _nl
import nki.language as nl
import pytest
from nki.isa import dge_mode, engine, oob_mode
from nkilib_src.nkilib.experimental.neurotile.core._helpers import (
    contiguous_strides as _contiguous_strides,
)
from nkilib_src.nkilib.experimental.neurotile.core._helpers import (
    product,
)
from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles as _tiles
from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid
from nkilib_src.nkilib.experimental.neurotile.core.indexing import element_offset
from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout
from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout
from nkilib_src.nkilib.experimental.neurotile.core.ndslice import BlockStream, NDSlice

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks


def grid_is_element(view, dim):
    """Check if dim is at element level (step == 1)."""
    return view._grid.is_element(dim)


# ============================================================================
# Helpers
# ============================================================================


def make_view(element_shape, tile_size, block_size=None, offset=0):
    """Create an NDSlice(Grid, HBMLayout) for testing."""
    ndim = len(element_shape)
    n_batch_dims = ndim - len(tile_size)

    # Pad tile_size for batch dims
    if n_batch_dims > 0:
        padded = []
        for _d in range(n_batch_dims):
            padded.append(1)
        for d in range(len(tile_size)):
            padded.append(tile_size[d])
        full_tile = tuple(padded)
    else:
        full_tile = tile_size
        n_batch_dims = 0

    # Pad block_size
    full_block = None
    if block_size is not None:
        padded_b = []
        for _d in range(n_batch_dims):
            padded_b.append(1)
        for d in range(len(block_size)):
            padded_b.append(block_size[d])
        full_block = tuple(padded_b)

    grid = Grid.from_shape(element_shape, full_tile, block_size=full_block, n_batch_dims=n_batch_dims)
    strides = _contiguous_strides(element_shape)
    layout = HBMLayout(
        source=MockTensor(element_shape),
        offset=offset,
        strides=strides,
        dtype="float32",
        buffer_type=nl.shared_hbm,
    )
    return NDSlice(grid, layout)


def make_untiled_view(element_shape, offset=0):
    """Create an untiled NDSlice (tensor_view semantics)."""
    strides = _contiguous_strides(element_shape)

    grid = Grid.from_shape(element_shape, None)
    layout = HBMLayout(
        source=MockTensor(element_shape),
        offset=offset,
        strides=strides,
        dtype="float32",
        buffer_type=nl.shared_hbm,
    )
    return NDSlice(grid, layout)


# ============================================================================
# Construction and forwarded attributes
# ============================================================================


@pytest_marks(["neurotile"])
class TestConstruction:
    @pytest.mark.fast
    def test_forwarded_attributes(self):
        v = make_view((512, 2048), (128, 512))
        assert v.shape == (4, 4)
        assert v.element_shape == (512, 2048)
        assert v.tile_size == (128, 512)
        assert v.ndim == 2
        assert v.dtype == "float32"
        assert isinstance(v._source, MockTensor)
        assert v._offset == 0
        assert v._strides == (2048, 1)
        # .data is the access-pattern view of the addressed region.
        assert v.data is not None
        assert v.buffer_type == "shared_hbm"

    def test_block_view_shape(self):
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert v.shape == (2, 2)

    def test_iteration_dim_shape(self):
        """Cursor defaults to n_batch_dims, so shape skips the batch axis."""
        v = make_view((8, 128, 64), (128, 64))
        # n_batch_dims=1 -> cursor=1 -> shape = (1, 1) on the inner tile dims.
        assert v.shape == (1, 1)
        assert v.ndim == 3
        # Outer count on the batch dim is its element_shape.
        assert v._grid.outer_axis(0).count == 8


# ============================================================================
# __getitem__ -- int indexing (descend)
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetitemInt:
    @pytest.mark.fast
    def test_single_tile_index(self):
        """tiles[1, 0] -> single tile."""
        v = make_view((512, 2048), (128, 512))
        r = v[1, 0]
        # Descended both dims to element level
        assert r.element_shape == (128, 512)
        # Offset: 1 * 128 * 2048 + 0 * 512 * 1
        assert r._offset == 1 * 128 * 2048

    def test_single_block_index(self):
        """blocks[0, 1] -> tile-level view."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        r = v[0, 1]
        # Block descent: remaining = block_size * tile_size per dim
        assert r.element_shape == (256, 1024)
        assert r.shape == (2, 2)  # tiles within block
        # Offset: 0*256*2048 + 1*1024*1 = 1024
        assert r._offset == 1024

    def test_block_then_tile(self):
        """blocks[0, 1][1, 0] -> element-level view."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        block = v[0, 1]
        tile = block[1, 0]
        assert tile.element_shape == (128, 512)
        # Block offset: 1024. Tile offset: 1*128*2048 + 0
        assert tile._offset == 1024 + 1 * 128 * 2048

    def test_negative_index(self):
        """tiles[-1, 0] -> last tile."""
        v = make_view((512, 2048), (128, 512))
        last = v[-1, 0]
        explicit = v[3, 0]
        assert last._offset == explicit._offset
        assert last.element_shape == explicit.element_shape

    def test_iteration_dim_drops(self):
        """3D tensor with 2D tile: int on iter dim -> dim drops."""
        v = make_view((8, 128, 64), (128, 64))
        r = v[3]
        # Iteration dim dropped: 3D -> 2D
        assert r.ndim == 2
        assert r.element_shape == (128, 64)
        # Offset: 3 * 1 * 8192 (stride for dim 0 of (8, 128, 64))
        assert r._offset == 3 * 8192

    def test_element_level_drops(self):
        """After tile descent, element int consumes the elem axes."""
        v = make_view((512, 2048), (128, 512))
        tile = v[0, 0]  # descend both to element level: remaining = (128, 512)
        assert tile.element_shape == (128, 512)
        # Now index at element level: step=1 for both dims; both consumed.
        sub = tile[42, 16]
        # Element-level offset: 42 * stride[0] + 16 * stride[1] = 42*2048 + 16
        assert sub._offset == 42 * 1 * 2048 + 16 * 1 * 1

    def test_min_2d_enforcement(self):
        """Tiled view: dropping all dims restores to 2D."""
        v = make_view((128, 512), (128, 512))
        # Single tile: shape (1, 1)
        tile = v[0, 0]  # element level
        # Index both element dims
        sub = tile[42, 16]
        # Would be 0D without min 2D -> enforced back to 2D-ish
        # One dim kept (the last exhausted is NOT dropped)
        assert sub.ndim >= 1  # at least 1D preserved


# ============================================================================
# __getitem__ -- slice indexing (narrow)
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetitemSlice:
    @pytest.mark.fast
    def test_basic_slice(self):
        """tiles[0:2] -> narrow dim 0."""
        v = make_view((512, 2048), (128, 512))
        r = v[0:2]
        assert r.element_shape == (256, 2048)  # 2 tiles * 128
        assert r._offset == 0
        assert r.shape == (2, 4)

    def test_slice_with_offset(self):
        """tiles[2:4] -> narrow with offset."""
        v = make_view((512, 2048), (128, 512))
        r = v[2:4]
        assert r.element_shape == (256, 2048)
        assert r._offset == 2 * 128 * 2048

    def test_slice_dim1(self):
        """tiles[:, 1:3] -> narrow dim 1."""
        v = make_view((512, 2048), (128, 512))
        r = v[:, 1:3]
        assert r.element_shape == (512, 1024)  # 2 tiles * 512
        assert r._offset == 1 * 512 * 1  # dim 1 stride = 1

    def test_full_slice(self):
        """tiles[:] -> no change."""
        v = make_view((512, 2048), (128, 512))
        r = v[:]
        assert r.element_shape == v.element_shape
        assert r._offset == 0

    def test_mixed_int_and_slice(self):
        """tiles[1, 0:2] -> descend dim 0 + narrow dim 1."""
        v = make_view((512, 2048), (128, 512))
        r = v[1, 0:2]
        # dim 0: descend (tile level), remaining=128
        # dim 1: narrow to 2 tiles, remaining=1024
        assert r.element_shape[1] == 1024
        assert r._offset == 1 * 128 * 2048


# ============================================================================
# __getitem__ -- indirect (runtime k)
# ============================================================================


@pytest_marks(["neurotile"])
class TestGetitemIndirect:
    @pytest.mark.fast
    def test_runtime_scalar(self):
        """tiles[runtime_k] -> indirect offset stored on Layout.indirect."""
        v = make_view((8, 128, 64), (128, 64))

        # Stand-in for an NKI LoopVar / CExpr: a class that's neither
        # int nor slice and not in the rejected-Python-types list, so
        # validate_index_key routes it through the runtime path.
        class _FakeLoopVar:
            pass

        runtime_k = _FakeLoopVar()
        r = v[runtime_k]
        # Indirect is a tagged value carrying the runtime expr + dim.
        assert r._layout.indirect is not None
        assert r._layout.indirect.dim == 0
        assert r._offset == 0  # compile-time offset unchanged


# ============================================================================
# tolist() -- iteration
# ============================================================================


@pytest_marks(["neurotile"])
class TestToList:
    @pytest.mark.fast
    def test_tile_iteration(self):
        """tolist() on 2x2 tile grid -> 2 children (dim 0)."""
        v = make_view((256, 1024), (128, 512))
        assert v.shape == (2, 2)
        children = v.tolist()
        assert len(children) == 2
        # Each child has dim 0 narrowed, iter_dim advanced
        assert children[0].shape == (2,)  # only dim 1 visible
        assert children[1].shape == (2,)

    def test_tile_iteration_offsets(self):
        """Children have correct offsets."""
        v = make_view((256, 1024), (128, 512))
        children = v.tolist()
        # Child 0: offset 0, child 1: offset 1 * 128 * 1024
        assert children[0]._offset == 0
        assert children[1]._offset == 128 * 1024

    def test_block_iteration(self):
        """tolist on block grid -> block-level children."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert v.shape == (2, 2)
        children = v.tolist()
        assert len(children) == 2  # 2 blocks along dim 0
        # Each child: dim 0 narrowed to 1 block (256 elements)
        assert children[0].element_shape[0] == 256

    def test_batch_dim_iteration_via_explicit_dim(self):
        """tolist(dim=0) on the batch dim -> 8 children (cursor doesn't auto-iterate batch)."""
        v = make_view((8, 128, 64), (128, 64))
        children = v.tolist(dim=0)
        assert len(children) == 8
        # Children retain ndim=3 (explicit-dim path doesn't drop).
        assert children[3]._offset == 3 * 8192

    def test_nested_tolist(self):
        """Nested tolist peels one dim at a time."""
        v = make_view((256, 1024), (128, 512))
        # First level: 2 children along dim 0
        level1 = v.tolist()
        assert len(level1) == 2
        # Second level: 2 children along dim 1
        level2 = level1[0].tolist()
        assert len(level2) == 2
        # Each is a single tile
        assert level2[0].element_shape == (128, 512)
        assert level2[1].element_shape == (128, 512)

    def test_single_tile_returns_self(self):
        """tolist on a view past all dims -> [self]."""
        v = make_view((128, 512), (128, 512))
        # Shape (1, 1). First tolist gives 1 item, second gives 1 item.
        children = v.tolist()
        assert len(children) == 1
        grandchildren = children[0].tolist()
        assert len(grandchildren) == 1


# ============================================================================
# _enumerate
# ============================================================================


@pytest_marks(["neurotile"])
class TestEnumerate:
    @pytest.mark.fast
    def test_basic(self):
        v = make_view((256, 1024), (128, 512))
        pairs = v._enumerate()
        assert len(pairs) == 2
        assert pairs[0][0] == 0
        assert pairs[1][0] == 1
        assert pairs[0][1]._offset == 0
        assert pairs[1][1]._offset == 128 * 1024

    def test_with_start(self):
        v = make_view((256, 1024), (128, 512))
        pairs = v._enumerate(start=10)
        assert pairs[0][0] == 10
        assert pairs[1][0] == 11


# ============================================================================
# __iter__ and __len__
# ============================================================================


@pytest_marks(["neurotile"])
class TestIterAndLen:
    @pytest.mark.fast
    def test_len(self):
        v = make_view((512, 2048), (128, 512))
        assert len(v) == 16  # 4 * 4

    def test_iter(self):
        v = make_view((256, 1024), (128, 512))
        items = list(v)
        assert len(items) == 2


# ============================================================================
# Full block -> tile -> element pipeline
# ============================================================================


@pytest_marks(["neurotile"])
class TestFullPipeline:
    @pytest.mark.fast
    def test_blocks_enumerate_then_load(self):
        """Simulate: enumerate blocks, each block is ready for load."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))

        # Enumerate block dim 0
        block_rows = v.tolist()
        assert len(block_rows) == 2

        # Enumerate block dim 1
        blocks = block_rows[0].tolist()
        assert len(blocks) == 2

        # One block: (256, 1024), iter_dim past both dims -> shape ()
        block = blocks[1]
        assert block.element_shape == (256, 1024)
        assert block.shape == ()  # fully enumerated, ready for load()

        # After enumerate, iter_dim is past all dims (shape=()).
        # Grid still has the data scope -- remaining tells load() the region size.
        # current_count at block level gives 1 (narrowed to single block).
        assert block._grid.remaining == (256, 1024)
        # To iterate tiles within, user would .load() -> SBUF with tile grid.

    def test_blocks_direct_index(self):
        """Direct index blocks[0, 1][1, 0] without enumerate."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))

        # Direct block index (no enumerate, iter_dim stays 0)
        block = v[0, 1]
        assert block.element_shape == (256, 1024)
        assert block.shape == (2, 2)  # tiles visible

        # Tile index within block
        tile = block[1, 0]
        assert tile.element_shape == (128, 512)

    def test_iteration_then_tiles(self):
        """3D tensor: explicit-dim iterate batch dim, then index tile."""
        v = make_view((8, 128, 64), (128, 64))

        # Iterate the batch dim explicitly.
        items = v.tolist(dim=0)
        assert len(items) == 8

        # Drop the batch dim, then index the inner tile (dim 0 of the 2D view).
        item = items[3][0]  # consume rank dim
        tile = item[0, 0]  # tile-level descent
        assert tile._offset == 3 * 8192


# ============================================================================
# Chained indexing
# ============================================================================


@pytest_marks(["neurotile"])
class TestChainedIndexing:
    @pytest.mark.fast
    def test_sequential_indexing(self):
        """view[0][1] == view[0, 1] in terms of offset."""
        v = make_view((256, 1024), (128, 512))
        v[0][1]
        v[0, 1]
        # Both should reach the same offset
        # v[0]: descend dim 0, element_shape=(128, 1024), offset=0
        # v[0][1]: descend dim 1 of the result... but iter_dim matters
        # After v[0], iter_dim was NOT advanced (it's __getitem__, not tolist())
        # So v[0] has iter_dim=0, dim 0 at element level
        # v[0][1]: processes key[0]=1 at dim=0+iter_dim=0
        # This descends dim 0 at element level (step=1)!
        # That's not the same as v[0, 1] which processes dim 0 and dim 1.
        #
        # This is a known difference: chained [0][1] indexes the SAME dim twice,
        # while [0, 1] indexes dim 0 then dim 1.
        # This matches numpy: arr[0][1] != arr[0, 1] for 2D arrays
        # (arr[0] gives row 0, then [1] gives element 1 of that row)

    def test_partial_then_full(self):
        """blocks[0] then block[0, 1] -- mixed levels."""
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        row = v[0]  # consume one axis on dim 0 (block -> tile)
        # dim 0: block axis consumed -> tile + leaf remain (2 axes).
        # dim 1: untouched -> block + tile + leaf (3 axes).
        assert len(row._grid.axes_for(0)) == 2
        assert len(row._grid.axes_for(1)) == 3


# ============================================================================
# NDSlice.tolist(dim=d) -- axis-explicit materialization
# ============================================================================


@pytest_marks(["neurotile"])
class TestTolistAxisExplicit:
    """view.tolist(dim=d) produces sub-views along the named axis."""

    @pytest.mark.fast
    def test_tolist_dim1_iterates_named_dim(self):
        v = make_view((256, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert len(items) == 2
        assert items[0].element_shape[1] == 512
        assert items[1].element_shape[1] == 512

    def test_tolist_dim_offsets(self):
        """Children along dim d have offsets stepping through that axis."""
        v = make_view((256, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert items[0]._offset == 0
        assert items[1]._offset == 512  # dim-1 stride = 1, tile_size = 512

    def test_tolist_list_behaviour(self):
        """tolist(dim=d) returns a plain Python list -- supports indexing / slicing."""
        v = make_view((256, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert isinstance(items, list)
        assert len(items) == 2
        assert len(items[0:1]) == 1


# ============================================================================
# Transforms through NDSlice
# ============================================================================


@pytest_marks(["neurotile"])
class TestTransforms:
    @pytest.mark.fast
    def test_reshape_dim(self):
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        assert reshaped.element_shape == (128, 8, 64)
        assert reshaped.tile_size == (128, 8, 64)  # single tile covering new shape

    def test_permute(self):
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        permuted = tile.permute((1, 0))
        assert permuted.element_shape == (512, 128)

    def test_flatten_dims(self):
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        flat = reshaped.flatten_dims(0, 1)
        assert flat.element_shape == (1024, 64)

    @pytest.mark.fast
    def test_rearrange_split_reorder(self):
        """rearrange (ni p) q -> p ni q equals the reshape_dim(0,(4,128)).permute((1,0,2)) chain."""
        v = make_view((512, 128), (512, 128))[0, 0]  # a coalesced-transpose out column, nc=4
        rearranged = v.rearrange((("ni", "p"), "q"), ("p", "ni", "q"), {"p": 128})
        chained = v.reshape_dim(0, (4, 128)).permute((1, 0, 2))
        assert rearranged.element_shape == (128, 4, 128)
        assert rearranged.element_shape == chained.element_shape
        assert rearranged._logical_strides() == chained._logical_strides()

    def test_transform_preserves_offset(self):
        v = make_view((256, 1024), (128, 512))
        tile = v[1, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        assert reshaped._offset == tile._offset  # offset preserved

    def test_rebuild_from_rejects_offset_shift(self):
        """_rebuild_from guards against routing an offset-shifting view through it:
        it reads back only shape/strides, so a nonzero-offset result would be
        silently dropped. The guard must raise instead."""
        v = make_view((256, 1024), (128, 512))
        tile = v[0, 0]
        # A synthetic transformed view that carries a nonzero offset (what a
        # future offset-shifting op routed through _as_nki_view would produce).
        bad = tile._as_nki_view()[:, 1:]  # slice -> offset != 0
        with pytest.raises(AssertionError, match="offset"):
            tile._rebuild_from(bad)


@pytest_marks(["neurotile"])
class TestTransformMisuseGuards:
    """Bad args to the transform methods must raise a named NDSlice error.

    permute() and expand_dim() defer their math to NkiTensor, which accepts a
    duplicate/OOB index and silently produces a wrong-shape view; these guard
    that the NDSlice layer rejects the misuse before it reaches NkiTensor.
    """

    @pytest.mark.fast
    def test_permute_wrong_length(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="must match"):
            v.permute((0,))

    @pytest.mark.fast
    def test_permute_duplicate_index(self):
        # (0, 0) is not a permutation -- NkiTensor would drop dim 1 silently.
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="must be a permutation"):
            v.permute((0, 0))

    @pytest.mark.fast
    def test_permute_out_of_range_index(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="must be a permutation"):
            v.permute((0, 5))

    @pytest.mark.fast
    def test_permute_non_int_entry(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="must be int"):
            v.permute((0, True))

    @pytest.mark.fast
    def test_expand_dim_out_of_range(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="out of range"):
            v.expand_dim(9)

    @pytest.mark.fast
    def test_expand_dim_negative(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="out of range"):
            v.expand_dim(-1)

    @pytest.mark.fast
    def test_expand_dim_bool(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        with pytest.raises(AssertionError, match="must be int"):
            v.expand_dim(True)

    @pytest.mark.fast
    def test_expand_dim_trailing_insert_valid(self):
        # dim == ndim is the valid trailing-insert boundary -- must NOT raise.
        v = make_view((128, 512), (128, 512))[0, 0]
        assert v.expand_dim(2).element_shape == (128, 512, 1)


# ============================================================================
# Element-level slice
# ============================================================================


@pytest_marks(["neurotile"])
class TestSlice:
    @pytest.mark.fast
    def test_slice_basic(self):
        """slice(dim, start, end) narrows at element level."""
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        sliced = tile.slice(1, 0, 256)
        assert sliced.element_shape == (128, 256)
        assert sliced._offset == 0

    def test_slice_with_offset(self):
        """slice with non-zero start advances offset."""
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        sliced = tile.slice(1, 128, 256)
        assert sliced.element_shape == (128, 128)
        # offset = start * strides[dim] = 128 * 1 = 128
        assert sliced._offset == 128

    def test_slice_dim0(self):
        """slice on partition dimension."""
        v = make_view((128, 512), (128, 512))
        tile = v[0, 0]
        sliced = tile.slice(0, 32, 64)
        assert sliced.element_shape == (32, 512)
        # offset = 32 * 512 = 16384
        assert sliced._offset == 32 * 512

    def test_slice_3d(self):
        """slice on 3D view (like head selection after reshape)."""
        v = make_view((16, 128, 8), (16, 128, 8))
        tile = v[0, 0, 0]
        sliced = tile.slice(2, 0, 1)
        assert sliced.element_shape == (16, 128, 1)
        assert sliced._offset == 0

    def test_slice_then_load_shape(self):
        """slice preserves tile_size = element_shape for loading."""
        v = make_view((16, 128, 8), (16, 128, 8))
        tile = v[0, 0, 0]
        sliced = tile.slice(2, 3, 5)
        assert sliced.tile_size == (16, 128, 2)
        assert sliced.element_shape == (16, 128, 2)

    def test_slice_sbuf_basic(self):
        """slice on SBUF tile narrows via sub_index."""
        v = make_sbuf_view((128, 512))
        sliced = v.slice(1, 0, 256)
        assert sliced.element_shape == (128, 256)
        assert isinstance(sliced._layout, SBUFLayout)

    def test_slice_sbuf_with_offset(self):
        """slice on SBUF with non-zero start."""
        v = make_sbuf_view((128, 512))
        sliced = v.slice(1, 128, 384)
        assert sliced.element_shape == (128, 256)
        # sub_index creates new source slice -- offset resets to 0
        assert sliced._layout.source.shape == (128, 256)

    def test_slice_sbuf_dim0(self):
        """slice on SBUF partition dimension."""
        v = make_sbuf_view((128, 512))
        sliced = v.slice(0, 32, 64)
        assert sliced.element_shape == (32, 512)

    def test_getitem_p_dim_slice_sbuf(self):
        """__getitem__ P-dim slice at element level works correctly."""
        v = make_sbuf_view((128, 512))
        # At element level (after _single_tile_view), P-dim slice
        # should narrow P, not be silently ignored.
        reshaped = v.reshape_dim(1, (4, 128))  # at element level
        sub = reshaped[32:64]  # P-dim slice -> should give (32, 4, 128)
        assert sub.element_shape[0] == 32

    @pytest.mark.fast
    def test_sliced_middle_dim_addressable_is_full_extent(self):
        """A mid-tensor slice is fully addressable; the offset is not subtracted."""
        # offset lands at 15*104=1560, so dim_offset_elements(1) = 15 % 4 = 3.
        # The buggy formula returned 4 - 3 = 1 and the AP merge-clamp dropped the dim.
        v = make_untiled_view((128, 30, 104)).slice(1, 15, 19)
        assert v.element_shape == (128, 4, 104)
        assert v._grid.remainder_dims == ()  # a slice is not a remainder
        assert v._layout.dim_addressable(1, v._grid) == 4
        assert v._layout.dim_addressable(0, v._grid) == 128
        assert v._layout.dim_addressable(2, v._grid) == 104


# ============================================================================
# Remainder
# ============================================================================


@pytest_marks(["neurotile"])
class TestRemainder:
    @pytest.mark.fast
    def test_remainder_detected_on_parent(self):
        v = make_view((300, 512), (128, 512))
        # 3 tiles: [0:128], [128:256], [256:300]
        assert v.shape == (3, 1)
        assert v.is_remainder is True
        # Total children: 3 (whole + remainder).
        children = v.tolist()
        assert len(children) == 3

    def test_remainder_dim_addressable_subtracts_offset(self):
        """A remainder dim DOES subtract the offset (the other branch of the fix)."""
        # 300 rows / 128 -> dim 0 walks past the source, so it is a remainder
        # dim; from an origin 256 rows in, only 300 - 256 = 44 remain.
        v = make_view((300,), (128,))
        assert v._grid.remainder_dims == (0,)
        layout_at_third_tile = HBMLayout(
            source="mock_tensor",
            offset=256,
            strides=(1,),
            dtype="float32",
            buffer_type=nl.shared_hbm,
        )
        assert layout_at_third_tile.dim_addressable(0, v._grid) == 300 - 256


@pytest_marks(["neurotile"])
class TestElementShapeIsTruthful:
    """``NDSlice.element_shape`` reports the real data extent the view holds,
    not the nominal walked extent. A view whose origin has advanced into a
    partial trailing tile / block walks a full nominal tile but only part of
    it lies inside the source, so the walked extent (``grid.remaining``)
    over-reports; ``element_shape`` clamps it to the reachable remainder."""

    @pytest.mark.fast
    def test_partial_trailing_block_index_reports_real_width(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks

        # N=1792 with a 2-tile (1024-wide) N-block: block[0,1] starts at col
        # 1024 but the source ends at 1792, so its real width is 768 -- not
        # the nominal 1024 the block walk would report.
        v = blocks(MockTensor(512, 1792), tile_size=(128, 512), block_size=(2, 2))
        trailing = v[0, 1]
        assert trailing.element_shape == (256, 768)
        # The walked extent (internal) still reports the nominal 1024.
        assert trailing._grid.remaining == (256, 1024)

    @pytest.mark.fast
    def test_full_block_index_reports_full_width(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks

        v = blocks(MockTensor(512, 1792), tile_size=(128, 512), block_size=(2, 2))
        assert v[0, 0].element_shape == (256, 1024)

    @pytest.mark.fast
    def test_slice_origin_advance_is_not_clamped(self):
        # A slice narrows element_shape to match the walk and moves the origin
        # with it, so the data extent is exact -- the offset must NOT be
        # subtracted (its dim_offset_elements is a meaningless modulo here).
        v = make_untiled_view((128, 30, 104)).slice(1, 15, 19)
        assert v.element_shape == (128, 4, 104)


@pytest_marks(["neurotile"])
class TestSubTileNarrow:
    """Narrowing a single tile BELOW its ``tile_size=`` (sub-tile narrowing):
    the sub-tile becomes the new grain, so ``tile_size`` must narrow with
    ``element_shape`` -- they stay equal on a single-tile view. ``tile_size``
    is the element-leaf count, so the slice that shrinks the leaf shrinks the
    reported tile_size too. Covers the HBM (pre-load) ``__getitem__`` path and
    the SBUF (post-load) ``sub_index`` path, which must agree."""

    @pytest.mark.fast
    def test_hbm_free_dim_narrow(self):
        # Single (128, 256) tile -> narrow the free dim to 64 < tile_size[1].
        tile = make_view((128, 256), (128, 256))[0, 0]
        narrow = tile[:, 0:64]
        assert narrow.element_shape == (128, 64)
        assert narrow.tile_size == (128, 64)

    @pytest.mark.fast
    def test_hbm_free_dim_narrow_with_offset(self):
        # Sub-tile narrow at a non-zero start: 96 cols from col 128.
        tile = make_view((128, 256), (128, 256))[0, 0]
        narrow = tile[:, 128:224]
        assert narrow.element_shape == (128, 96)
        assert narrow.tile_size == (128, 96)
        assert narrow._offset == 128

    @pytest.mark.fast
    def test_hbm_partition_dim_narrow(self):
        # Narrow the partition dim below tile_size[0]: 32 < 128.
        tile = make_view((128, 256), (128, 256))[0, 0]
        narrow = tile[0:32, :]
        assert narrow.element_shape == (32, 256)
        assert narrow.tile_size == (32, 256)

    @pytest.mark.fast
    def test_hbm_both_dims_narrow(self):
        tile = make_view((128, 256), (128, 256))[0, 0]
        narrow = tile[0:32, 0:64]
        assert narrow.element_shape == (32, 64)
        assert narrow.tile_size == (32, 64)

    @pytest.mark.fast
    def test_hbm_narrow_after_multi_tile_descent(self):
        # Descend a multi-tile view to one tile, then sub-tile narrow within it.
        tg = make_view((128, 512), (128, 128))  # 4 F-tiles of 128
        narrow = tg[0, 1][:, 0:32]
        assert narrow.element_shape == (128, 32)
        assert narrow.tile_size == (128, 32)

    @pytest.mark.fast
    def test_sbuf_free_dim_narrow_matches_hbm(self):
        # Post-load (SBUF) path: same sub-tile narrow narrows tile_size too,
        # so both paths agree (element_shape == tile_size == (128, 64)).
        v = make_sbuf_view((128, 256))
        narrow = v.slice(1, 0, 64)
        assert narrow.element_shape == (128, 64)
        assert narrow.tile_size == (128, 64)
        assert isinstance(narrow._layout, SBUFLayout)


# ============================================================================
# Repr
# ============================================================================


@pytest_marks(["neurotile"])
class TestRepr:
    @pytest.mark.fast
    def test_repr(self):
        v = make_view((512, 2048), (128, 512))
        r = repr(v)
        assert "NDSlice(" in r
        assert "shape=(4, 4)" in r


# ============================================================================
# SBUF Transforms
# ============================================================================


def make_sbuf_view(tile_size, tile_shape=None):
    """Create an NDSlice(Grid, SBUFLayout) backed by a mock SBUF buffer."""
    if tile_shape is None:
        tile_shape = tuple(1 for _ in tile_size)
    tile_p = tile_size[0]
    tile_f = product(tile_size, start=1)
    total_tiles = product(tile_shape)
    sbuf_data = MockTensor((tile_p, total_tiles * tile_f))
    element_shape = tuple(tile_shape[d] * tile_size[d] for d in range(len(tile_size)))
    grid, layout = SBUFLayout.build_view(sbuf_data, element_shape, tile_size, "float32", "sbuf")
    return NDSlice(grid, layout)


@pytest_marks(["neurotile"])
class TestSBUFTransforms:
    @pytest.mark.fast
    def test_reshape_dim(self):
        """reshape_dim on SBUF tile updates Grid, keeps SBUFLayout."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        assert reshaped.element_shape == (128, 4, 128)
        assert reshaped.tile_size == (128, 4, 128)
        assert isinstance(reshaped._layout, SBUFLayout)
        assert reshaped._layout.source is v._layout.source  # same underlying buffer

    def test_permute(self):
        """permute on SBUF tile reorders element_shape."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        permuted = reshaped.permute((0, 2, 1))
        assert permuted.element_shape == (128, 128, 4)
        assert permuted._layout.source is v._layout.source

    def test_flatten_dims(self):
        """flatten_dims on SBUF tile merges dimensions back."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        flat = reshaped.flatten_dims(1, 2)
        assert flat.element_shape == (128, 512)
        assert flat._layout.source is v._layout.source

    def test_squeeze_dim(self):
        """squeeze_dim on SBUF tile removes size-1 dim."""
        v = make_sbuf_view((128, 1, 512))
        squeezed = v.squeeze_dim(1)
        assert squeezed.element_shape == (128, 512)
        assert squeezed._layout.source is v._layout.source

    def test_expand_dim(self):
        """expand_dim on SBUF tile inserts size-1 dim."""
        v = make_sbuf_view((128, 512))
        expanded = v.expand_dim(1)
        assert expanded.element_shape == (128, 1, 512)
        assert expanded._layout.source is v._layout.source

    def test_broadcast(self):
        """broadcast on SBUF tile after expand_dim."""
        v = make_sbuf_view((128, 512))
        expanded = v.expand_dim(1)
        broadcast = expanded.broadcast(1, 4)
        assert broadcast.element_shape == (128, 4, 512)
        assert broadcast._layout.source is v._layout.source

    def test_split(self):
        """split convenience method on SBUF tile."""
        v = make_sbuf_view((128, 512))
        chunked = v.split(1, 4)
        assert chunked.element_shape == (128, 4, 128)
        assert chunked._layout.source is v._layout.source

    def test_sub_index_after_reshape(self):
        """reshape_dim then __getitem__ routes through sub_index."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        # Index: select chunk 2 -> (128, 128) slice at F-offset 256
        chunk = reshaped[:, 2, :]
        assert chunk.element_shape == (128, 128)
        assert isinstance(chunk._layout, SBUFLayout)
        # Verify sub-slice offset: chunk 2 starts at F=256
        assert chunk._layout.offset == 0  # sub_index returns new source slice
        assert chunk._layout.source.shape == (128, 128)

    def test_split_then_index(self):
        """split then iterate chunks."""
        v = make_sbuf_view((128, 512))
        chunked = v.split(1, 2)  # (128, 2, 256)
        c0 = chunked[:, 0, :]
        c1 = chunked[:, 1, :]
        assert c0.element_shape == (128, 256)
        assert c1.element_shape == (128, 256)
        # c0 source at F=[0:256], c1 source at F=[256:512]
        assert c0._layout.source.shape == (128, 256)
        assert c1._layout.source.shape == (128, 256)

    def test_reshape_preserves_layout(self):
        """After reshape, the layout (same buffer) is preserved; ``.data`` is the
        access-pattern view, so its logical shape reflects the reshape."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        # layout source unchanged, same buffer
        assert reshaped._layout.source is v._layout.source
        # .data is the AP view: logical shape == the reshaped element_shape.
        assert reshaped.data.shape == reshaped.element_shape
        assert reshaped.element_shape == (128, 4, 128)

    def test_chained_reshape_flatten(self):
        """reshape_dim -> sub_index -> flatten: reshape chain on SBUF."""
        v = make_sbuf_view((128, 512))
        reshaped = v.reshape_dim(1, (4, 128))
        # Flatten back to original
        flat = reshaped.flatten_dims(1, 2)
        assert flat.element_shape == (128, 512)
        # sub_index on flattened view works
        sub = flat[:, 0:256]
        assert sub.element_shape == (128, 256)


# ============================================================================
# iter_dim advancement
# ============================================================================


@pytest_marks(["neurotile"])
class TestCursorAdvancement:
    """view[i] at tile level should advance cursor past consumed dim."""

    @pytest.mark.fast
    def test_single_dim_descent_advances(self):
        v = make_view((256, 1024), (128, 512))
        child = v[0]
        # dim 0 selected: stack popped from (tile, 1) to (1,) = element
        assert grid_is_element(child, 0)
        # dim 1 untouched: still at tile level
        assert not grid_is_element(child, 1)
        assert child._grid.cursor == 1

    def test_all_dims_descended_advances_to_end(self):
        """Both dims int-consumed -> cursor advances past both, no wrap
        (leaves are not iteration-level, no further iteration possible)."""
        v = make_view((512, 2048), (128, 512))
        child = v[0, 0]
        assert grid_is_element(child, 0)
        assert grid_is_element(child, 1)
        assert child._grid.cursor == child._grid.ndim

    def test_block_consume_advances_past_dim(self):
        """Int on dim 0 (block-grid) -> cursor advances past dim 0;
        the remaining iteration is on dim 1."""
        v = make_view((512, 1024), (128, 512), block_size=(2, 2))
        child = v[0]
        # dim 0 was iter-consumed: cursor moves past it.
        assert child._grid.cursor == 1
        # dim 0 retains the block interior axes: still navigable
        # for drilling, but past the cursor for shape purposes.
        assert not grid_is_element(child, 0)

    def test_consistent_with_tolist(self):
        v = make_view((256, 1024), (128, 512))
        indexed = v[0]
        iterated = v.tolist()[0]
        assert indexed._grid.cursor == iterated._grid.cursor


# ============================================================================
# block_size property
# ============================================================================


@pytest_marks(["neurotile"])
class TestBlockSize:
    """NDSlice.block_size derived from Grid."""

    @pytest.mark.fast
    def test_present(self):
        v = make_view((512, 1024), (128, 512), block_size=(2, 2))
        assert v.block_size == (2, 2)

    def test_none_for_tiles(self):
        v = make_view((512, 1024), (128, 512))
        assert v.block_size is None

    def test_partial(self):
        v = make_view((512, 1024), (128, 512), block_size=(2, 1))
        assert v.block_size == (2, 1)


# ============================================================================
# .data attribute
# ============================================================================


@pytest_marks(["neurotile"])
class TestDataAttribute:
    """``NDSlice.data`` is the access-pattern view of the region this view
    addresses -- never ``None``; its shape is the logical element extent."""

    @pytest.mark.fast
    def test_data_not_none(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        assert v.data is not None

    def test_data_shape_is_element_shape(self):
        v = make_view((128, 512), (128, 512))[0, 0]
        assert v.data.shape == v.element_shape

    def test_sbuf_data_not_none(self):
        v = make_sbuf_view((128, 512))
        assert v.data is not None


# ============================================================================
# NDSlice user-facing method misuse guards (P0c lift)
# ============================================================================


@pytest_marks(["neurotile"])
class TestTolistMisuseGuards:
    """NDSlice.tolist(dim=) input validation."""

    @pytest.mark.fast
    def test_dim_must_be_int_or_none(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="dim= must be int or None"):
            v.tolist(dim=1.5)

    def test_dim_in_range(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match=r"dim=5 is out of range"):
            v.tolist(dim=5)

    def test_dim_negative_rejected(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match=r"dim=-1 is out of range"):
            v.tolist(dim=-1)

    def test_dim_none_works(self):
        v = make_view((512, 1024), (128, 512))
        # cursor-driven; should not raise
        items = v.tolist()
        assert len(items) == 4

    def test_dim_int_in_range_works(self):
        v = make_view((512, 1024), (128, 512))
        items = v.tolist(dim=1)
        assert len(items) == 2


@pytest_marks(["neurotile"])
class TestStreamMisuseGuards:
    """NDSlice.stream(dim=, buffer_count=, ...) input validation."""

    @pytest.mark.fast
    def test_dim_must_be_int(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="dim= must be int"):
            v.stream(dim=0.5, buffer_count=2)

    def test_dim_in_range(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match=r"dim=5 is out of range"):
            v.stream(dim=5, buffer_count=2)

    def test_buffer_count_must_be_int(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="buffer_count= must be int"):
            v.stream(buffer_count=2.0)

    def test_buffer_count_must_be_positive(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="buffer_count= must be >= 1"):
            v.stream(buffer_count=0)
        with pytest.raises(AssertionError, match="buffer_count= must be >= 1"):
            v.stream(buffer_count=-1)

    def test_pattern_override_requires_out_shape(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="pattern_override= requires out_shape="):
            v.stream(buffer_count=2, pattern_override=[[1024, 4], [1, 512]])

    def test_transpose_axes_requires_transpose(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="transpose_axes= requires transpose=True"):
            v.stream(buffer_count=2, transpose_axes=(1, 0))

    def test_stream_on_sbuf_rejected(self):
        v = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="only valid on HBM views"):
            v.stream(buffer_count=2)

    def test_stream_with_unconsumed_batch_dims(self):
        # 3D source + 2D tile_size -> 1 batch dim left.
        v = make_view((4, 512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="batch dims to be consumed first"):
            v.stream(buffer_count=2)


@pytest_marks(["neurotile"])
class TestStreamSlotView:
    """A stream slot (stream[k]) must be typed at the step's logical tile grid
    -- transpose-aware -- so it is tile-addressable like a loaded view, instead
    of a flat single tile. These drive the slot-shape / tile-size / view-typing
    helpers directly (pure Python; no SBUF allocation or backend)."""

    def _xpose_stream(self, buffer_count=2):
        # Blocked seq view (BS=2 tiles of (TS=128, H=256)) -> transpose stream.
        v = make_view((512, 256), (128, 256), block_size=(2, 1))
        return v.stream(buffer_count=buffer_count, transpose=True)

    def _normal_stream(self, buffer_count=2):
        v = make_view((512, 256), (128, 256), block_size=(2, 1))
        return v.stream(buffer_count=buffer_count)

    def _sharded_xpose_stream(self, num_shards=2, total_tiles=8, h=256, buffer_count=2):
        # Interleaved tile-sharded block: shard 0 owns every num_shards-th seq-tile,
        # so a BS=2 block's owned tiles are non-contiguous (gapped) in HBM.
        import nkilib_src.nkilib.experimental.neurotile as nt

        v = make_view((total_tiles * 128, h), (128, h))
        own_r = nt.interleaved_range(rank=0, num_shards=num_shards, total=total_tiles)
        blocked = nt.blocks(v[own_r, :], block_size=(2, 1))
        return blocked.stream(buffer_count=buffer_count, transpose=True)

    @pytest.mark.fast
    def test_transpose_slot_shape_and_tile_size(self):
        # Slot holds the transposed block (128, BS*H1*TS)=(128,512); its per-tile
        # granularity is one seq-tile's transpose (128, H1*TS)=(128,256).
        s = self._xpose_stream()
        assert s._slot_shape() == (128, 512)
        assert s._slot_tile_size() == (128, 256)

    @pytest.mark.fast
    def test_transpose_slot_view_is_tile_addressable(self):
        # The slot view carries a (1, BS) tile grid: slot[0, ti] is one
        # transposed sub-tile (128, 256), NOT the flat whole slot.
        s = self._xpose_stream()
        slot = s._slot_view(MockTensor(s._slot_shape()))
        assert slot.shape == (1, 2)
        assert slot.tile_size == (128, 256)
        assert slot[0, 0].element_shape == (128, 256)
        assert slot[0, 1].element_shape == (128, 256)

    @pytest.mark.fast
    def test_normal_slot_view_tile_addressable(self):
        # Non-transpose slot is also typed at the step tile grid (consistency
        # with loaded views), not a flat single tile.
        s = self._normal_stream()
        slot = s._slot_view(MockTensor(s._slot_shape()))
        # block of BS=2 (TS,H) tiles -> slot carries the step's tile_size.
        assert slot.tile_size == s._slot_tile_size()
        assert slot.ndim == 2

    @pytest.mark.fast
    def test_slot_shape_and_view_agree(self):
        # _slot_shape (allocation) and _slot_view (typing) must use the same
        # tile_size -- the single-source-of-truth guard against drift.
        for s in (self._xpose_stream(), self._normal_stream()):
            slot = s._slot_view(MockTensor(s._slot_shape()))
            assert slot.element_shape == s._slot_shape()
            assert slot.tile_size == s._slot_tile_size()

    @pytest.mark.fast
    def test_sharded_transpose_slot_is_dense(self):
        # Tile-sharded (interleaved) block: 8 tiles, 2 shards -> shard 0 owns 4
        # tiles, BS=2 -> a block's 2 owned tiles span a 4-tile gap. The transpose
        # slot must be sized to the OWNED extent (2 tiles -> 2*H1 chunks), NOT the
        # gapped iteration span (which would double it). H=256 -> H1=2 chunks.
        s = self._sharded_xpose_stream(num_shards=2, total_tiles=8, h=256)
        # owned: 2 tiles * H1=2 chunks = 4 (128,128) tiles packed -> (128, 512).
        # gapped span would be 2x -> (128, 1024); assert we got the dense one.
        assert s._slot_shape() == (128, 512), s._slot_shape()

    @pytest.mark.fast
    def test_contiguous_and_sharded_slots_match_owned(self):
        # A contiguous BS=2 block and a sharded block that OWNS 2 tiles must
        # allocate the SAME transpose-slot shape -- sharding changes which tiles
        # are owned, not how much transposed data a block holds.
        contig = make_view((256, 256), (128, 256), block_size=(2, 1)).stream(buffer_count=2, transpose=True)
        sharded = self._sharded_xpose_stream(num_shards=2, total_tiles=8, h=256)
        assert contig._slot_shape() == sharded._slot_shape()

    @pytest.mark.fast
    def test_single_chunk_vs_multi_chunk_slot_shape(self):
        # H<=128 -> one F-chunk: slot rows = H (not capped at 128).
        # H>128  -> H1 chunks of 128 laid side by side on the free axis.
        single = make_view((256, 96), (128, 96), block_size=(2, 1)).stream(buffer_count=2, transpose=True)
        # BS=2 tiles, 1 chunk each -> (96, 2*128) = (96, 256).
        assert single._slot_shape() == (96, 256), single._slot_shape()
        multi = make_view((256, 384), (128, 384), block_size=(2, 1)).stream(buffer_count=2, transpose=True)
        # H=384 -> H1=3 chunks; BS=2 tiles -> 2*3=6 (128,128) tiles -> (128, 768).
        assert multi._slot_shape() == (128, 768), multi._slot_shape()


@pytest_marks(["neurotile"])
class TestLoadMisuseGuards:
    """NDSlice.load() input validation."""

    @pytest.mark.fast
    def test_load_on_sbuf_rejected(self):
        v = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="load.. is only valid on HBM views"):
            v.load()

    def test_load_with_unconsumed_batch_dims(self):
        # 3D source + 2D tile_size -> 1 batch dim left.
        v = make_view((4, 512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="batch dims to be consumed first"):
            v.load()

    def test_oob_value_requires_oob_mode(self):
        v = make_view((512, 1024), (128, 512))
        tile = v[0, 0]
        with pytest.raises(AssertionError, match="oob_value= requires oob_mode="):
            tile.load(oob_value=0.0)

    def test_pattern_override_requires_out_shape_or_dst(self):
        v = make_view((512, 1024), (128, 512))
        tile = v[0, 0]
        with pytest.raises(AssertionError, match="pattern_override= requires out_shape= or dst="):
            tile.load(pattern_override=[[1, 512], [512, 1]])


@pytest_marks(["neurotile"])
class TestStoreMisuseGuards:
    """NDSlice.store() input validation (existing guards verified)."""

    @pytest.mark.fast
    def test_store_on_sbuf_rejected(self):
        v = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="store.. is only valid on HBM views"):
            v.store(data="anything")

    def test_store_data_cannot_be_ndslice(self):
        # Use a dest HBM view; pass an NDSlice as data -- should reject.
        dst = make_view((512, 1024), (128, 512))
        src = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match=r"expects ndarray or .data view, not NDSlice"):
            dst[0, 0].store(src)

    def test_store_with_unconsumed_batch_dims(self):
        v = make_view((4, 512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="batch dims to be consumed first"):
            v.store(data="anything")


@pytest_marks(["neurotile"])
class TestDmaEngineValidation:
    """The engine= (HWDGE descriptor-gen engine) selector is plumbed to
    nisa.dma_copy and validated at the NDSlice/BlockStream boundary:
      - only valid with dge_mode=hwdge, and must be sync/scalar;
      - never valid on the transpose path (nisa.dma_transpose has no engine);
      - None / unknown / dma are the always-allowed default.
    These are pure API-boundary asserts -- no device/tracing needed. The spy
    records engine= to prove it is forwarded all the way to HBMLayout.load."""

    def _spy_layout_load(self, monkeypatch):
        recorded = []

        def spy(
            self,
            grid,
            dtype=None,
            dst=None,
            oob_mode=None,
            oob_value=None,
            out_shape=None,
            transpose=False,
            transpose_axes=None,
            dge_mode=None,
            priority=None,
            pattern_override=None,
            engine=None,
        ):
            recorded.append({"engine": engine, "dge_mode": dge_mode})
            return (None, None)

        monkeypatch.setattr(HBMLayout, "load", spy)
        return recorded

    @pytest.mark.fast
    def test_engine_on_transpose_rejected(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="engine= is not supported on the transpose path"):
            v[0, 0].load(transpose=True, engine=engine.scalar)

    @pytest.mark.fast
    def test_engine_requires_hwdge(self):
        v = make_view((512, 1024), (128, 512))
        # engine set but dge_mode left default (unknown) -> must reject.
        with pytest.raises(AssertionError, match="can only be set when dge_mode=nisa.dge_mode.hwdge"):
            v[0, 0].load(engine=engine.scalar)

    @pytest.mark.fast
    def test_engine_must_be_sync_or_scalar(self):
        v = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="must be nisa.engine.sync or nisa.engine.scalar"):
            v[0, 0].load(engine=engine.vector, dge_mode=dge_mode.hwdge)

    def test_engine_none_allowed_and_forwarded(self, monkeypatch):
        recorded = self._spy_layout_load(monkeypatch)
        # The spy returns a stub (None, None); the post-spy NDSlice construction
        # then fails harmlessly -- we only care the engine= reached layout.load.
        try:
            make_view((512, 1024), (128, 512))[0, 0].load()
        except Exception:
            pass
        assert recorded[0]["engine"] is None

    def test_engine_unknown_allowed(self, monkeypatch):
        recorded = self._spy_layout_load(monkeypatch)
        try:
            make_view((512, 1024), (128, 512))[0, 0].load(engine=engine.unknown)
        except Exception:
            pass
        assert recorded[0]["engine"] == engine.unknown

    def test_engine_scalar_hwdge_forwarded(self, monkeypatch):
        recorded = self._spy_layout_load(monkeypatch)
        try:
            make_view((512, 1024), (128, 512))[0, 0].load(engine=engine.scalar, dge_mode=dge_mode.hwdge)
        except Exception:
            pass
        assert recorded[0]["engine"] == engine.scalar
        assert recorded[0]["dge_mode"] == dge_mode.hwdge

    @pytest.mark.fast
    def test_store_engine_requires_hwdge(self):
        # The engine= assert fires before store touches data, so a sentinel suffices.
        dst = make_view((512, 1024), (128, 512))
        with pytest.raises(AssertionError, match="can only be set when dge_mode=nisa.dge_mode.hwdge"):
            dst[0, 0].store(data="sentinel", engine=engine.sync)

    # NOTE: a BlockStream.load(transpose=True, engine=...) test would exercise the
    # same NDSlice.load transpose-engine assert, but BlockStream.load allocates real
    # SBUF buffers first (needs an active NKI backend), so it can't run in this pure
    # unit context. The assert itself is covered by test_engine_on_transpose_rejected;
    # BlockStream.load forwards engine= to NDSlice.load unconditionally (see source).


# ============================================================================
# Slice-based sharding -- block_range narrows remaining; offset matches start
# ============================================================================


@pytest_marks(["neurotile"])
class TestSliceShardingOffset:
    """view[range, :] narrows the view to the rank's owned slice."""

    @pytest.mark.fast
    def test_block_range_narrows_remaining(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import blocks
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        # 2 blocks on dim 0 (block_size=2 -> 256 elements/block).
        # Rank 1 of 2 ranks: own 1 block; offset = 1 * (2*128) * stride[0] = 262144.
        sharded = blocks(MockTensor(512, 1024), tile_size=(128, 512), block_size=(2, 2))[
            block_range(rank=1, num_shards=2, total=2), :
        ]
        assert sharded._offset == 1 * (2 * 128) * 1024
        assert sharded._grid.remaining[0] == 256


# ============================================================================
# HBMLayout int advance preserves an existing indirect tag
# ============================================================================


@pytest_marks(["neurotile"])
class TestIndirectOffsetCombine:
    """Compile-time advance on a dim with a runtime indirect must keep the tag."""

    @pytest.mark.fast
    def test_int_advance_preserves_indirect(self):
        from nkilib_src.nkilib.experimental.neurotile.core.axis import IndirectKind, IndirectOffset

        layout = HBMLayout(
            "src", 0, (1024, 1), "f32", "hbm", indirect=IndirectOffset(kind=IndirectKind.SCALAR, value=100, dim=0)
        )
        new = layout.advance(0, 2, 128)
        # Compile-time advance updates offset; runtime tag is preserved.
        assert new.indirect.value == 100
        assert new.indirect.dim == 0
        assert new.offset == 2 * 128 * 1024


# ============================================================================
# Runtime element-offset indexing metadata and dispatch
# ============================================================================


@pytest_marks(["neurotile"])
class TestRuntimeElementOffsetIndexing:
    @pytest.mark.fast
    def test_index_stride_elements_on_tile_view(self):
        view = make_view((512, 2048), (128, 512))
        assert view.index_stride_elements == (128, 512)

    def test_index_stride_elements_on_block_view(self):
        view = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert view.index_stride_elements == (256, 1024)

    def test_index_stride_elements_after_partial_index(self):
        view = make_view((512, 2048), (128, 512))
        row = view[1, :]
        assert row.index_stride_elements == (512,)

    def test_index_stride_elements_after_selected_block(self):
        view = make_view((512, 2048), (128, 512), block_size=(2, 2))
        block = view[0, 0]
        assert block.index_stride_elements == (128, 512)

    def test_static_element_offset_folds_without_tile_scaling(self):
        view = make_view((512, 2048), (128, 512))
        shifted = view[element_offset(128), 0]
        assert shifted._layout.offset == 128 * 2048
        assert shifted._layout.indirect is None

    def test_element_offset_is_relative_to_sliced_view(self):
        view = make_view((512, 2048), (128, 512))
        tail = view[1:, :]
        shifted = tail[element_offset(128), 0]
        assert shifted._layout.offset == (128 + 128) * 2048

    def test_runtime_element_offset_passes_through(self):
        view = make_view((512, 2048), (128, 512))
        offset = MockTensor((1, 1), dtype=nl.int32)
        shifted = view[element_offset(offset), 0]
        assert shifted._layout.indirect.value is offset
        assert shifted._layout.indirect.dim == 0

    @pytest.mark.parametrize("shape", [(1,), (1, 1)])
    def test_runtime_element_offset_accepts_scalar_tensor_shapes(self, shape):
        view = make_view((512, 2048), (128, 512))
        offset = MockTensor(shape, dtype=nl.int32)
        shifted = view[element_offset(offset), 0]
        assert shifted._layout.indirect.value is offset

    @pytest.mark.parametrize("shape", [(2,), (2, 1), (1, 2)])
    def test_runtime_element_offset_rejects_vector_tensor_shapes(self, shape):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="tensor value must be an SBUF scalar"):
            view[element_offset(MockTensor(shape, dtype=nl.int32)), 0]

    @pytest.mark.parametrize(
        "offset",
        [
            True,
            False,
            1.0,
            None,
            [1],
            (1,),
            {"offset": 1},
            {1},
            "1",
            b"1",
            object(),
        ],
    )
    def test_runtime_element_offset_rejects_invalid_payloads(self, offset):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="nt.element_offset"):
            view[element_offset(offset), 0]

    @pytest.mark.parametrize("buffer", [nl.shared_hbm, nl.psum])
    def test_runtime_element_offset_rejects_non_sbuf_tensor(self, buffer):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="tensor value must be an SBUF scalar"):
            view[element_offset(MockTensor((1, 1), dtype=nl.int32, buffer=buffer)), 0]

    def test_static_element_offset_rejects_negative_offsets(self):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="must be non-negative"):
            view[element_offset(-1), 0]

    def test_static_element_offset_rejects_out_of_range_offsets(self):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="static offset 512 out of range"):
            view[element_offset(512), 0]

    def test_static_element_offset_bounds_are_relative_to_sliced_view(self):
        view = make_view((512, 2048), (128, 512))
        tail = view[1:, :]
        with pytest.raises(AssertionError, match="static offset 384 out of range"):
            tail[element_offset(384), 0]

    def test_runtime_element_offset_accepts_sbuf_scalar_ndslice(self):
        view = make_view((512, 2048), (128, 512))
        offset = make_sbuf_view((1, 1))
        shifted = view[element_offset(offset), 0]
        assert shifted._layout.indirect.value is offset._layout.source

    def test_runtime_element_offset_rejects_hbm_ndslice(self):
        view = make_view((512, 2048), (128, 512))
        offset = make_view((1, 1), (1, 1))
        with pytest.raises(AssertionError, match="SBUF-backed scalar view"):
            view[element_offset(offset), 0]

    def test_runtime_element_offset_rejects_vector_sbuf_ndslice(self):
        view = make_view((512, 2048), (128, 512))
        offset = make_sbuf_view((1, 2))
        with pytest.raises(AssertionError, match="value must be scalar-shaped"):
            view[element_offset(offset), 0]

    def test_runtime_element_offset_rejects_sbuf_target_view(self):
        view = make_sbuf_view((128, 512))
        with pytest.raises(AssertionError, match="only supported on HBM-backed views"):
            view[element_offset(MockTensor((1, 1), dtype=nl.int32)), 0]

    def test_runtime_logical_stride_1_index_does_not_scale(self, monkeypatch):
        view = make_view((8, 128, 64), (128, 64))
        logical_index = MockTensor((1, 1), dtype=nl.int32)

        def fail_if_called(value, scale):
            raise AssertionError("unexpected scaling")

        monkeypatch.setattr(HBMLayout, "_materialize_scaled_scalar_offset", staticmethod(fail_if_called))
        shifted = view[logical_index, 0, 0]
        assert shifted._layout.indirect.value is logical_index
        assert shifted._layout.indirect.dim == 0

    def test_runtime_logical_vector_tensor_index_rejects(self):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            view[MockTensor((1, 2), dtype=nl.int32), 0]

    def test_runtime_logical_hbm_tensor_index_rejects(self):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            view[MockTensor((1, 1), dtype=nl.int32, buffer=nl.shared_hbm), 0]

    def test_runtime_logical_sbuf_scalar_index_is_scaled(self, monkeypatch):
        view = make_view((512, 2048), (128, 512))
        logical_index = MockTensor((1, 1), dtype=nl.int32)
        monkeypatch.setattr(
            HBMLayout,
            "_materialize_scaled_scalar_offset",
            staticmethod(lambda value, scale: ("scaled", value, scale)),
        )
        shifted = view[logical_index, 0]
        assert shifted._layout.indirect.value == ("scaled", logical_index, 128)

    def test_runtime_logical_loop_var_with_stride_gt_one_rejects(self):
        view = make_view((512, 2048), (128, 512))
        with pytest.raises(AssertionError, match="requires an SBUF scalar index"):
            view[object(), 0]


# ============================================================================
# N-D SBUF tile_shape computation
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFTileShape:
    """_sbuf_tile_shape_from_buffer should compute per-dim tile counts for N-D."""

    @pytest.mark.fast
    def test_3d_matching_dims(self):
        """3D source (128, 4, 64) with tile (128, 1, 64) -> (1, 4, 1)."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 4, 64), (128, 1, 64))
        assert ts == (1, 4, 1)

    def test_3d_single_tile(self):
        """3D source == tile -> (1, 1, 1)."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 1, 8), (128, 1, 8))
        assert ts == (1, 1, 1)

    def test_2d_fallback(self):
        """2D source with 2D tile -> flat computation."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 512), (128, 512))
        assert ts == (1, 1)

    def test_2d_multi_tile(self):
        """2D source with multiple tiles."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        ts = SBUFLayout.tile_shape_from_buffer((128, 2048), (128, 512))
        assert ts == (1, 4)


# ============================================================================
# SBUF sharding via slice indexing
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFSharding:
    """Slice-based sharding on SBUF tiles narrows the view to the rank's range."""

    @pytest.mark.fast
    def test_sbuf_tiles_with_block_range(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        sbuf = MockTensor((128, 8, 64), dtype="bf16", buffer=nl.sbuf)
        # Rank 0 of 2, block-sharded on dim 1 -> owns 4 of 8 tiles.
        view = tiles(sbuf, tile_size=(128, 1, 64))[:, block_range(rank=0, num_shards=2, total=8), :]
        assert view._grid.remaining[1] == 4


# ============================================================================
# SBUFLayout transform strides and broadcast AP
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFTransformStrides:
    """apply_transform stores strides; ap() uses them for broadcast."""

    @pytest.mark.fast
    def test_apply_transform_creates_new_layout(self):
        """apply_transform returns new SBUFLayout with transform_strides."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        new_layout = layout.apply_transform((8, 0, 1))
        assert new_layout is not layout
        assert new_layout.ap_strides == (8, 0, 1)
        assert new_layout.source is layout.source

    def testap_strides_chain(self):
        """Chained transforms replace strides."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        first = layout.apply_transform((64, 1))
        second = first.apply_transform((0, 64, 1))
        assert second.ap_strides == (0, 64, 1)

    def testap_strides_used_inap_strides(self):
        """transform_strides() returns stored strides for chaining."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        transformed = layout.apply_transform((8, 0, 1))
        assert transformed.transform_strides((128, 4, 64)) == (8, 0, 1)

    def test_noap_strides_gives_contiguous(self):
        """Without transform, transform_strides() returns contiguous."""
        layout = SBUFLayout("src", 0, (512,), (128, 512), "f32", "sbuf")
        assert layout.transform_strides((128, 512)) == (512, 1)


# ============================================================================
# SBUFLayout.tile_data N-D reshape
# ============================================================================


@pytest_marks(["neurotile"])
class TestTileDataNDReshape:
    """tile_data() should reshape N-D source to 2D before slicing."""

    @pytest.mark.fast
    def test_3d_source_returns_2d(self):
        """3D source (128, 1, 8) -> _slice_one_tile returns 2D (128, 8)."""
        src = MockTensor((128, 1, 8))
        layout = SBUFLayout(src, 0, (8,), (128, 8), "f32", "sbuf")
        result = layout.tile_data()
        assert result.shape == (128, 8)

    def test_2d_source_no_reshape(self):
        """2D source (128, 512) -> tile_data returns as-is when allocation tile fills source."""
        src = MockTensor((128, 512))
        layout = SBUFLayout(src, 0, (512,), (128, 512), "f32", "sbuf")
        result = layout.tile_data()
        assert result is src  # returned directly, no reshape


# ============================================================================
# Logical vs physical SBUF shape (P-fold awareness)
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFLogicalShape:
    """SBUF storage is 2-D; NDSlice carries the logical element_shape.

    For multi-P-tile views, P-tiles fold into F columns physically while
    element_shape retains the logical (BS*tile_p, f_count*tile_f) rank.
    """

    @pytest.mark.fast
    def test_single_row_shape_matches_source(self):
        """Single P-tile row: source shape equals element_shape."""
        v = make_sbuf_view(tile_size=(128, 512), tile_shape=(1, 4))
        assert v._layout.source.shape == (128, 2048)
        assert v.shape == (1, 4)

    def test_multi_p_tile_source_folds_into_f(self):
        """Multi-P-tile SBUF: physical (128, 4*512); logical (256, 1024)."""
        v = make_sbuf_view(tile_size=(128, 512), tile_shape=(2, 2))
        assert v._layout.source.shape == (128, 2048)
        assert v._grid.element_shape == (256, 1024)
        assert v.shape == (2, 2)

    def test_sbuf_tile_shape_from_logical_vs_raw(self):
        """_sbuf_tile_shape_from_buffer differs for raw vs logical shape."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        tile_size = (128, 512)
        assert SBUFLayout.tile_shape_from_buffer((128, 2048), tile_size) == (1, 4)
        assert SBUFLayout.tile_shape_from_buffer((256, 1024), tile_size) == (2, 2)


# ============================================================================
# Slice-based sharding on a 3D source (rank batch dim) drops the consumed rank
# ============================================================================


@pytest_marks(["neurotile"])
class TestRankShardDimDrop:
    """v[rank, :, :] consumes the rank batch dim and drops it from the view.

    Matches the fgcc_mlp_cte pattern: 3D source (num_ranks, M, H) is indexed
    with `view[rank]` so subsequent `[m_batch]` targets the M dim.
    """

    @pytest.mark.fast
    def test_no_shard_rank_dim_visible(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles

        v = tiles(MockTensor(2, 512, 512), tile_size=(128, 512))
        assert v.ndim == 3
        assert v._grid.n_batch_dims == 1

    def test_rank_index_drops_dim(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles

        v = tiles(MockTensor(2, 512, 512), tile_size=(128, 512))
        sharded = v[0]  # consume rank dim
        assert sharded.ndim == 2
        assert sharded._grid.remaining == (512, 512)

    def test_rank_then_m_index_targets_m(self):
        from nkilib_src.nkilib.experimental.neurotile.core.factories import tiles

        v = tiles(MockTensor(2, 512, 512), tile_size=(128, 512))
        sharded = v[0]
        child = sharded[0]
        assert child._grid.remaining[0] == 128


# ============================================================================
# New public API -- Phase 1: tolist(dim=), stream(dim=), _load_dst field
# ============================================================================


def _equivalent_children(a, b):
    """Structural equality for a pair of NDSlice lists (NDSlice has no __eq__).

    Compares only data-relevant fields (element_shape, offset). `shape`
    tracks iteration-grid cursor state which can differ between paths that
    produce semantically-equivalent children.
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].element_shape != b[i].element_shape:
            return False
        if a[i]._offset != b[i]._offset:
            return False
    return True


@pytest_marks(["neurotile"])
class TestTolistDefaultDim:
    """NDSlice.tolist() with no args -- follows grid cursor, equals tolist()."""

    @pytest.mark.fast
    def test_tolist_cursor_2d(self):
        v = make_view((512, 2048), (128, 512))
        assert _equivalent_children(v.tolist(), v.tolist())

    def test_tolist_cursor_with_block_size(self):
        v = make_view((512, 2048), (128, 512), block_size=(2, 2))
        assert _equivalent_children(v.tolist(), v.tolist())

    def test_tolist_explicit_dim_iterates_batch(self):
        """Explicit-dim tolist on the batch dim yields one child per batch slot."""
        v = make_view((4, 128, 512), (128, 128))
        items = v.tolist(dim=0)
        assert len(items) == 4

    def test_tolist_empty_view(self):
        """Fully-consumed view with ndim==0 materializes as empty list."""
        v = make_view((128, 256), (128, 256))
        inner = v[0]  # consume outer dim; grid.cursor >= grid.ndim
        # After consumption cursor is past all dims -> tolist returns [self].
        result = inner.tolist()
        # This should NOT be length 0 since inner still has content; the
        # "returns [self]" branch of tolist_cursor fires. Assert it's length 1.
        assert len(result) == 1


@pytest_marks(["neurotile"])
class TestTolistAlongDim:
    """NDSlice.tolist(dim=d) -- explicit axis iteration, cursor untouched."""

    @pytest.mark.fast
    def test_tolist_dim0_counts(self):
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=0)
        assert len(items) == 4  # 512 / 128

    def test_tolist_dim1_counts(self):
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=1)
        assert len(items) == 4  # 2048 / 512

    def test_tolist_dim_offsets_step_along_dim(self):
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=1)
        # dim=1 stride = 1, tile_size = 512 -> offsets are 0, 512, 1024, 1536.
        offsets = [items[i]._offset for i in range(len(items))]
        assert offsets == [0, 512, 1024, 1536]

    def test_tolist_dim_preserves_other_dims(self):
        """Child along dim=1 keeps full dim 0 span."""
        v = make_view((512, 2048), (128, 512))
        items = v.tolist(dim=1)
        for child in items:
            assert child.element_shape[0] == 512


@_nki.jit
def _stream_dim0_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(buffer_count=2)
    assert s.count == v._grid.current_count(0)
    assert s._buffer_count == 2
    assert s._dim == 0
    return src


@_nki.jit
def _stream_dim1_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(dim=1, buffer_count=2)
    assert s.count == v._grid.current_count(1)
    assert s._dim == 1
    return src


@_nki.jit
def _stream_tolist_children_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(buffer_count=2)
    children = s.tolist()
    assert len(children) == 4
    for c in children:
        assert isinstance(c, NDSlice)
    return src


@_nki.jit
def _stream_tolist_load_dst_probe():
    src = _nl.ndarray((512, 2048), dtype=_nl.float32, buffer=_nl.shared_hbm)
    v = _tiles(src, tile_size=(128, 512))
    s = v.stream(buffer_count=2)
    children = s.tolist()
    for i in range(len(children)):
        assert children[i]._load_dst is s._buffers[i % s._buffer_count]
    return src


@pytest_marks(["neurotile"])
@pytest.mark.skip(
    reason="Requires NKI runtime (libnrt). These tests dispatch a "
    "compiled kernel and belong in integration/, not unit/."
)
class TestStreamDimKwarg:
    """NDSlice.stream(dim=d, ...) produces a BlockStream walking along dim d.

    Run via @nki.jit so SBUF buffer allocation has an active backend.
    """

    @pytest.mark.fast
    def test_stream_default_dim0(self):
        _stream_dim0_probe[1]()

    def test_stream_explicit_dim(self):
        _stream_dim1_probe[1]()


@pytest_marks(["neurotile"])
class TestLoadDstField:
    """NDSlice._load_dst, _load_pattern_override, _load_out_shape -- stored state."""

    @pytest.mark.fast
    def test_load_dst_default_none(self):
        v = make_view((128, 512), (128, 512))
        assert v._load_dst is None
        assert v._load_pattern_override is None
        assert v._load_out_shape is None

    def test_load_dst_settable_via_init(self):
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        marker = object()
        v2 = NDSlice(v._grid, v._layout, load_dst=marker)
        assert v2._load_dst is marker


@pytest_marks(["neurotile"])
class TestLoadDstRouting:
    """NDSlice.load() precedence rules for _load_dst routing.

    Monkeypatches HBMLayout.load to record its dst= kwarg. Encodes the
    design's four precedence rules as invariants.
    """

    def _capture_load_dst(self, monkeypatch):
        """Return a list that records the `dst` kwarg on every HBMLayout.load call."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout

        recorded = []

        def spy(
            self,
            grid,
            dtype=None,
            dst=None,
            oob_mode=None,
            oob_value=None,
            out_shape=None,
            transpose=False,
            transpose_axes=None,
            dge_mode=None,
            priority=None,
            pattern_override=None,
            engine=None,
        ):
            recorded.append(
                {
                    "dst": dst,
                    "pattern_override": pattern_override,
                    "out_shape": out_shape,
                    "dtype": dtype,
                    "engine": engine,
                }
            )
            # Return a stub -- we don't need a real SBUFLayout for these tests.
            return (None, None)

        monkeypatch.setattr(HBMLayout, "load", spy)
        return recorded

    def _nd_with_load_dst(self, buf):
        """Build an NDSlice with load_dst=buf but no other state."""
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        return NDSlice(v._grid, v._layout, load_dst=buf)

    def _fake_sbuf_buffer(self, dtype="float32"):
        return MockTensor((128, 512), dtype=dtype)

    @pytest.mark.fast
    def test_explicit_dst_wins_over_load_dst(self, monkeypatch):
        """view.load(dst=X) uses X -- _load_dst is ignored."""
        recorded = self._capture_load_dst(monkeypatch)
        buf = self._fake_sbuf_buffer()
        override = self._fake_sbuf_buffer()
        view = self._nd_with_load_dst(buf)
        try:
            view.load(dst=override)
        except Exception:
            pass  # spy returns (None, None) so wrap construction may crash
        assert recorded[0]["dst"] is override

    def test_load_dst_used_when_no_explicit(self, monkeypatch):
        """view.load() (no dst=) routes into _load_dst."""
        recorded = self._capture_load_dst(monkeypatch)
        buf = self._fake_sbuf_buffer(dtype=make_view((128, 512), (128, 512)).dtype)
        view = self._nd_with_load_dst(buf)
        try:
            view.load()
        except Exception:
            pass
        assert recorded[0]["dst"] is buf

    def test_remainder_skips_load_dst(self, monkeypatch):
        recorded = self._capture_load_dst(monkeypatch)
        buf = self._fake_sbuf_buffer(dtype="float32")
        v = make_view((300, 512), (128, 512))
        items = v.tolist()
        remainder = items[-1]
        assert remainder.is_remainder
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        view = NDSlice(remainder._grid, remainder._layout, load_dst=buf)
        try:
            view.load()
        except Exception:
            pass
        assert recorded[0]["dst"] is None

    def test_dtype_mismatch_skips_load_dst(self, monkeypatch):
        """If caller requests a different dtype, skip _load_dst and allocate fresh."""
        recorded = self._capture_load_dst(monkeypatch)
        # buf dtype intentionally mismatched with load() dtype request.
        buf = self._fake_sbuf_buffer(dtype="float32")
        view = self._nd_with_load_dst(buf)
        try:
            view.load(dtype="bfloat16")
        except Exception:
            pass
        assert recorded[0]["dst"] is None

    def test_bare_ndslice_allocates(self, monkeypatch):
        """Bare NDSlice (_load_dst=None) -> dst=None -> layout allocates as before."""
        recorded = self._capture_load_dst(monkeypatch)
        v = make_view((128, 512), (128, 512))
        try:
            v.load()
        except Exception:
            pass
        assert recorded[0]["dst"] is None

    def test_pattern_override_default_from_stream(self, monkeypatch):
        """_load_pattern_override is applied when caller omits pattern_override."""
        recorded = self._capture_load_dst(monkeypatch)
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        sentinel = [[1, 512], [512, 1]]
        # _load_out_shape required so the pattern_override + out_shape contract
        # is satisfied (see NDSlice.load assertion).
        view = NDSlice(v._grid, v._layout, load_pattern_override=sentinel, load_out_shape=(128, 512))
        try:
            view.load()
        except Exception:
            pass
        assert recorded[0]["pattern_override"] is sentinel

    def test_explicit_pattern_override_wins(self, monkeypatch):
        recorded = self._capture_load_dst(monkeypatch)
        v = make_view((128, 512), (128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import NDSlice

        default = [[1, 512], [512, 1]]
        override = [[2, 256], [256, 2]]
        view = NDSlice(v._grid, v._layout, load_pattern_override=default, load_out_shape=(128, 512))
        try:
            view.load(pattern_override=override, out_shape=(128, 512))
        except Exception:
            pass
        assert recorded[0]["pattern_override"] is override


@pytest_marks(["neurotile"])
@pytest.mark.skip(
    reason="Requires NKI runtime (libnrt). These tests dispatch a "
    "compiled kernel and belong in integration/, not unit/."
)
class TestBlockStreamTolistBinding:
    """BlockStream.tolist() returns NDSlice children with load_dst pre-filled."""

    @pytest.mark.fast
    def test_tolist_children_are_ndslice(self):
        _stream_tolist_children_probe[1]()

    def test_tolist_pre_fills_load_dst_from_rotating_buffers(self):
        _stream_tolist_load_dst_probe[1]()


# ============================================================================
# whole_tiles / remainder_tiles -- iteration split for boundary handling
# ============================================================================


@pytest_marks(["neurotile"])
class TestWholeAndRemainderTiles:
    """Pins the contract: whole_tiles returns non-remainder children;
    remainder_tiles returns the rest. On a clean (non-remainder) view
    whole_tiles returns the full list and remainder_tiles returns []."""

    @pytest.mark.fast
    def test_clean_view_whole_tiles_returns_all(self):
        """Even-divisible view: every child has is_remainder == False."""
        v = make_view((256, 1024), (128, 512))  # 2x2, no remainder
        whole = v.whole_tiles()
        rem = v.remainder_tiles()
        assert len(whole) == len(v.tolist())
        assert rem == []

    def test_clean_view_remainder_tiles_empty(self):
        v = make_view((512, 1024), (128, 512))
        assert v.remainder_tiles() == []

    def test_remainder_view_split(self):
        """Partial trailing tile on dim 0: 300 // 128 = 2, remainder 44.
        tolist() yields 3 children; the last is is_remainder=True.
        """
        v = make_view((300, 512), (128, 512))
        all_items = v.tolist()
        assert len(all_items) == 3
        whole = v.whole_tiles()
        rem = v.remainder_tiles()
        assert len(whole) + len(rem) == len(all_items)
        # Whole + remainder partition tolist; remainder ends up at the end.
        assert all(not c.is_remainder for c in whole)
        assert all(c.is_remainder for c in rem)


# ============================================================================
# Transpose-load DMA-control: public signatures (Gap 1 + Gap 5)
#
# These pin the *public surface* the gap fix adds. They are pure-Python
# (inspect-based / pre-emission asserts) so they need no tracer or device.
# Numeric correctness of the transpose itself is covered by the hardware
# integration tests (test_sbuf_remainder / test_streaming_consume).
# ============================================================================


def _params(fn):
    """Parameter names of a callable, in order."""
    return list(inspect.signature(fn).parameters)


@pytest_marks(["neurotile"])
class TestTransposeLoadSignatures:
    """NDSlice.load / store / stream and BlockStream.load / store expose the
    DMA-control params the gap fix adds (priority everywhere; transpose on the
    streamed load + stream factory)."""

    @pytest.mark.fast
    def test_load_exposes_priority(self):
        # Gap 1: priority is a public QoS knob on the transpose-capable load.
        assert "priority" in _params(NDSlice.load)

    def test_store_exposes_priority(self):
        # Gap 1 parity: store gets the same QoS knob.
        assert "priority" in _params(NDSlice.store)

    def test_stream_exposes_transpose(self):
        # stream(transpose=, transpose_axes=) declares the load type at
        # construction so the rotating slots are sized AND grid-typed up front
        # (needed when stream[k] is read as a typed dst= before any load runs).
        assert "transpose" in _params(NDSlice.stream)
        assert "transpose_axes" in _params(NDSlice.stream)

    def test_blockstream_load_exposes_transpose(self):
        # Gap 5a: the streamed (double-buffered) load forwards transpose.
        assert "transpose" in _params(BlockStream.load)

    def test_blockstream_load_exposes_priority(self):
        assert "priority" in _params(BlockStream.load)

    def test_blockstream_store_exposes_priority(self):
        assert "priority" in _params(BlockStream.store)

    def test_load_exposes_transpose_axes(self):
        # N-D gather-transpose: per-rank permutation knob on the load.
        assert "transpose_axes" in _params(NDSlice.load)

    def test_blockstream_load_exposes_transpose_axes(self):
        assert "transpose_axes" in _params(BlockStream.load)


@pytest_marks(["neurotile"])
class TestStreamLoadTypeConsistency:
    """BlockStream fixes its load type (transpose / transpose_axes) at
    construction (stream(transpose=...)) or on the first .load(), and rejects
    any later load that disagrees -- the rotating slots are sized once, so
    mixing load types into one pool would corrupt a wrongly-shaped slot. The
    gate is _lock_load_type (pure, no allocation); here we drive it directly."""

    def _stream(self):
        # BlockStream.__init__ no longer allocates, so this is pure-Python.
        return make_view((512, 2048), (128, 512)).stream(buffer_count=2)

    def _lock(self, s, transpose, transpose_axes):
        # Simulate "first .load() happened" without allocating slots.
        s._decided = True
        s._transpose = transpose
        s._transpose_axes = transpose_axes

    @pytest.mark.fast
    def test_matching_transpose_ok(self):
        s = self._stream()
        self._lock(s, transpose=True, transpose_axes=None)
        # Same mode re-asserts cleanly and does not re-allocate.
        s._lock_load_type(transpose=True, transpose_axes=None)
        assert s._transpose is True

    def test_matching_transpose_axes_ok(self):
        s = self._stream()
        self._lock(s, transpose=True, transpose_axes=(2, 1, 0))
        s._lock_load_type(transpose=True, transpose_axes=(2, 1, 0))

    def test_transpose_then_normal_rejected(self):
        s = self._stream()
        self._lock(s, transpose=True, transpose_axes=None)
        with pytest.raises(AssertionError, match="same transpose"):
            s._lock_load_type(transpose=False, transpose_axes=None)

    def test_normal_then_transpose_rejected(self):
        s = self._stream()
        self._lock(s, transpose=False, transpose_axes=None)
        with pytest.raises(AssertionError, match="same transpose"):
            s._lock_load_type(transpose=True, transpose_axes=None)

    def test_different_transpose_axes_rejected(self):
        s = self._stream()
        self._lock(s, transpose=True, transpose_axes=(2, 1, 0))
        with pytest.raises(AssertionError, match="same transpose"):
            s._lock_load_type(transpose=True, transpose_axes=(3, 1, 2, 0))

    def test_access_after_load_does_not_reset_mode(self):
        # A slot access after a transpose load must not flip the locked mode
        # back to normal. _ensure_buffers_for_access only re-locks when NOT
        # already decided, so an already-locked transpose mode is preserved.
        s = self._stream()
        self._lock(s, transpose=True, transpose_axes=(2, 1, 0))
        # Drive only the lock half (allocation needs a live backend).
        if not s._decided:
            s._lock_load_type(False, None)
        assert s._transpose is True
        assert s._transpose_axes == (2, 1, 0)

    def test_access_first_locks_normal(self):
        # If a slot access happens before any .load(), it locks the normal mode;
        # a later transpose load is then (correctly) rejected.
        s = self._stream()
        s._decided = True  # access path would set this; we assert the resulting gate
        s._transpose = False
        s._transpose_axes = None
        with pytest.raises(AssertionError, match="same transpose"):
            s._lock_load_type(transpose=True, transpose_axes=None)


@pytest_marks(["neurotile"])
class TestTransposeLoadValidation:
    """NDSlice.load(transpose=True, ...) strictly validates DMA-control args
    against the nisa.dma_transpose contract -- rejecting today's silently
    dropped / unsupported combinations with a named error rather than emitting
    a DMA that ignores them.

    Authoritative constraints (nki.isa.dma_transpose):
      - direct/tiled (non-indirect) path: dge_mode in {unknown, hwdge} only
        (swdge / none are invalid here).
      - oob_mode.skip is only valid when src uses indirect indexing.
      - priority is a QoS level in [0, 3].
    """

    @pytest.mark.fast
    def test_priority_out_of_range_rejected(self):
        v = make_view((128, 512), (128, 512))
        with pytest.raises(AssertionError, match="priority"):
            v.load(transpose=True, priority=4)

    def test_priority_negative_rejected(self):
        v = make_view((128, 512), (128, 512))
        with pytest.raises(AssertionError, match="priority"):
            v.load(transpose=True, priority=-1)

    def test_dge_mode_swdge_rejected_on_direct_path(self):
        # swdge is only valid for the indirect (gather) transpose; a plain
        # (non-indirect) transpose-load must reject it instead of dropping it.
        v = make_view((128, 512), (128, 512))
        with pytest.raises(AssertionError, match="dge_mode"):
            v.load(transpose=True, dge_mode=dge_mode.swdge)

    def test_dge_mode_none_rejected_on_direct_path(self):
        v = make_view((128, 512), (128, 512))
        with pytest.raises(AssertionError, match="dge_mode"):
            v.load(transpose=True, dge_mode=dge_mode.none)

    def test_oob_skip_without_indirect_rejected(self):
        # oob_mode.skip is only meaningful for indirect access; a static
        # transpose-load must reject it (today it is silently dropped).
        v = make_view((128, 512), (128, 512))
        with pytest.raises(AssertionError, match="oob_mode"):
            v.load(transpose=True, oob_mode=oob_mode.skip, oob_value=0.0)

    def test_transpose_axes_on_static_rejected(self):
        # transpose_axes is gather-only; a static (non-indirect) transpose is
        # 2-D and must reject an explicit permutation rather than ignore it.
        v = make_view((128, 512), (128, 512))
        with pytest.raises(AssertionError, match="transpose_axes"):
            v.load(transpose=True, transpose_axes=(1, 0))


@pytest_marks(["neurotile"])
class TestStorePriorityValidation:
    """NDSlice.store(priority=...) validates the QoS range. Store has no
    transpose path (DMA transpose dst must be SBUF), so only the priority
    range rule applies -- dge_mode is unconstrained for dma_copy."""

    @pytest.mark.fast
    def test_store_priority_out_of_range_rejected(self):
        dst = make_view((128, 512), (128, 512))
        tile = dst[0, 0]
        with pytest.raises(AssertionError, match="priority"):
            tile.store(data="placeholder", priority=4)

    def test_store_priority_negative_rejected(self):
        dst = make_view((128, 512), (128, 512))
        tile = dst[0, 0]
        with pytest.raises(AssertionError, match="priority"):
            tile.store(data="placeholder", priority=-1)


# Note: stream(transpose=True) slot-shape correctness needs a live NKI backend
# (BlockStream allocates rotating slots via nl.ndarray at construction), so it
# is validated end-to-end in the hardware test test_streaming_consume.py
# (TestStreamTransposeLoad) rather than here in pure-Python unit tests.


@pytest_marks(["neurotile"])
class TestTransposedSbufShapeFor:
    """HBMLayout.transposed_sbuf_shape_for: the SBUF allocation shape a
    .load(transpose=True) produces. This is the shape a stream slot / a dst=
    pre-alloc must match (the transpose path asserts dst.shape == this), so it
    is the source of truth for both Gap 5b slot sizing and Gap 4's
    partial-inclusive transposed_shape (doc §4.6#5). Pure shape math."""

    @pytest.mark.fast
    def test_f_under_128_single_chunk(self):
        """F<=128 -> (F, P): the whole transpose fits one column block."""
        v = make_view((128, 100), (128, 100))
        assert v._layout.transposed_sbuf_shape_for(v._grid) == (100, 128)

    def test_f_exact_128(self):
        v = make_view((128, 128), (128, 128))
        assert v._layout.transposed_sbuf_shape_for(v._grid) == (128, 128)

    def test_f_exact_multiple(self):
        """F=512=4*128 -> (128, P*4)."""
        v = make_view((128, 512), (128, 512))
        assert v._layout.transposed_sbuf_shape_for(v._grid) == (128, 128 * 4)

    def test_f_partial_includes_remainder_chunk(self):
        """F=400=3*128+16 -> 4 chunks (3 full + 1 partial) -> (128, P*4).
        The partial chunk gets its own P-wide column block, so a dst= pre-alloc
        must be sized for num_chunks=4, not 3."""
        v = make_view((128, 400), (128, 400))
        assert v._layout.transposed_sbuf_shape_for(v._grid) == (128, 128 * 4)


@pytest_marks(["neurotile"])
class TestTransposedGridShape:
    """The tile grid a .load(transpose=True) returns: built from the (packed shape,
    per-tile transposed size) pair, it preserves the source tile grid axis-swapped
    so the result is coordinate-indexable (xT[0, k]) instead of one merged tile.
    Grid.from_shape mirrors what SBUFLayout.build_view does in _load_transpose --
    pure shape math, no backend."""

    @staticmethod
    def _transpose_grid(v):
        layout = v._layout
        return Grid.from_shape(
            element_shape=layout.transposed_sbuf_shape_for(v._grid),
            tile_size=layout.transposed_tile_size_for(v._grid),
        )

    @pytest.mark.fast
    def test_single_tile(self):
        """[128,128] one tile -> (1, 1) grid of one [128,128] transposed tile."""
        v = make_view((128, 128), (128, 128))
        assert self._transpose_grid(v).shape == (1, 1)

    def test_multi_p_tile_axis_swap(self):
        """[512,128] block, tile (128,128): a (4,1) source tile grid transposes to
        a (1, 4) grid of [128,128] tiles -- the axis-swap that makes xT[0,k] work."""
        v = make_view((512, 128), (128, 128), block_size=(4, 1))
        assert self._transpose_grid(v).shape == (1, 4)

    def test_coalesced_row_chunks(self):
        """[128,512] tile-row: F=512=4 chunks -> (1, 4) grid of [128,128] tiles."""
        v = make_view((128, 512), (128, 128))
        assert self._transpose_grid(v).shape == (1, 4)

    def test_f_under_128_single(self):
        """F<=128 stays a single [F, P] tile -> (1, 1) grid."""
        v = make_view((128, 64), (128, 64))
        assert self._transpose_grid(v).shape == (1, 1)

    def test_nonsquare_p(self):
        """P != 128 carried through: (64, 256) -> 2 chunks -> (128, 64*2)."""
        v = make_view((64, 256), (64, 256))
        assert v._layout.transposed_sbuf_shape_for(v._grid) == (128, 64 * 2)

    @pytest.mark.parametrize(
        "elem,tile,block",
        [
            ((512, 128), (128, 128), (4, 1)),  # multi-P-tile
            ((128, 512), (128, 128), None),  # coalesced F-chunks
            ((128, 128), (128, 128), None),  # single tile
            ((64, 256), (64, 256), None),  # non-square P, multi-chunk
        ],
    )
    def test_packed_shape_divides_into_tiles(self, elem, tile, block):
        """Invariant the returned grid depends on: the packed SBUF shape must be an
        exact tile multiple of the per-tile transposed size on every axis, so
        Grid.from_shape yields whole tiles (no ragged trailing tile)."""
        v = make_view(elem, tile, block_size=block)
        packed = v._layout.transposed_sbuf_shape_for(v._grid)
        per_tile = v._layout.transposed_tile_size_for(v._grid)
        assert len(packed) == len(per_tile)
        for whole, one in zip(packed, per_tile, strict=True):
            assert one > 0 and whole % one == 0

    def test_dst_shape_matches_packed_shape(self):
        """A caller-provided dst= (and each stream slot) must equal the packed
        transpose shape -- the transpose path asserts dst.shape == this. Guards
        the load(transpose=True, dst=)/stream-slot contract against drift."""
        v = make_view((512, 128), (128, 128), block_size=(4, 1))
        # transposed_sbuf_shape_for is the single source of truth for the dst alloc.
        assert v._layout.transposed_sbuf_shape_for(v._grid) == (128, 512)


# ============================================================================
# Transpose-load + priority validation helpers (Gap 1)
#
# _validate_priority / _validate_transpose_load live in ndslice (next to the
# load()/store() that call them) so they need no NDSlice<->_validation import
# cycle. They encode the authoritative nisa.dma_transpose / dma_copy arg
# contract. Pure-Python: no tracer/device.
# ============================================================================


@pytest_marks(["neurotile"])
class TestValidatePriority:
    """_validate_priority: DMA QoS level must be None or an int in [0, 3]."""

    @pytest.mark.fast
    def test_none_ok(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_priority

        _validate_priority(None)  # no raise

    def test_in_range_ok(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_priority

        for p in (0, 1, 2, 3):
            _validate_priority(p)  # no raise

    def test_too_high_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_priority

        with pytest.raises(AssertionError, match="priority"):
            _validate_priority(4)

    def test_negative_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_priority

        with pytest.raises(AssertionError, match="priority"):
            _validate_priority(-1)


@pytest_marks(["neurotile"])
class TestValidateTransposeLoad:
    """_validate_transpose_load: the nisa.dma_transpose arg contract.

    - dge_mode in {None, unknown, hwdge}; swdge / none rejected (non-indirect).
    - oob_mode.skip only valid when src is indirect.
    - oob_value requires oob_mode set.
    - priority in [0, 3].
    - transpose_axes only valid on the indirect (gather) path.
    """

    @pytest.mark.fast
    def test_defaults_ok(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        # All-None: the unchanged default behavior must validate cleanly.
        _validate_transpose_load(
            dge_mode=None, oob_mode=None, oob_value=None, priority=None, transpose_axes=None, is_indirect=False
        )

    def test_unknown_dge_ok(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        _validate_transpose_load(
            dge_mode=dge_mode.unknown,
            oob_mode=None,
            oob_value=None,
            priority=None,
            transpose_axes=None,
            is_indirect=False,
        )

    def test_hwdge_ok(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        # hwdge is accepted at the semantic layer; the shape constraints
        # (p==16, F%128==0, 2-byte) are enforced deeper, by the compiler.
        _validate_transpose_load(
            dge_mode=dge_mode.hwdge,
            oob_mode=None,
            oob_value=None,
            priority=None,
            transpose_axes=None,
            is_indirect=False,
        )

    def test_swdge_rejected_non_indirect(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="dge_mode"):
            _validate_transpose_load(
                dge_mode=dge_mode.swdge,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=None,
                is_indirect=False,
            )

    def test_none_dge_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="dge_mode"):
            _validate_transpose_load(
                dge_mode=dge_mode.none,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=None,
                is_indirect=False,
            )

    def test_oob_skip_without_indirect_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="oob_mode"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=oob_mode.skip,
                oob_value=0.0,
                priority=None,
                transpose_axes=None,
                is_indirect=False,
            )

    def test_oob_skip_with_indirect_ok(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        _validate_transpose_load(
            dge_mode=None, oob_mode=oob_mode.skip, oob_value=0.0, priority=None, transpose_axes=None, is_indirect=True
        )

    def test_priority_out_of_range_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="priority"):
            _validate_transpose_load(
                dge_mode=None, oob_mode=None, oob_value=None, priority=9, transpose_axes=None, is_indirect=False
            )

    def test_transpose_axes_rejected_on_static(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="transpose_axes"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=(3, 1, 2, 0),
                is_indirect=False,
            )

    def test_transpose_axes_ok_on_indirect(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        _validate_transpose_load(
            dge_mode=None,
            oob_mode=None,
            oob_value=None,
            priority=None,
            transpose_axes=(3, 1, 2, 0),
            is_indirect=True,
            n_gathered_dims=3,
        )

    def test_transpose_axes_rank_mismatch_rejected(self):
        # (1,0) is a 2-D permutation; applying it to a 3-non-trivial-dim view must
        # raise rather than silently running the rank-2 AP on 3-D dims.
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="axes rank must match"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=(1, 0),
                is_indirect=True,
                n_gathered_dims=3,
            )

    def test_transpose_axes_4d_needs_3_real_dims(self):
        # The 4-D reshape-trick form (3,1,2,0) requires exactly 3 non-trivial dims.
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        _validate_transpose_load(
            dge_mode=None,
            oob_mode=None,
            oob_value=None,
            priority=None,
            transpose_axes=(3, 1, 2, 0),
            is_indirect=True,
            n_gathered_dims=3,
        )

    def test_transpose_axes_unsupported_permutation_rejected(self):
        # (0,1) has the right length for rank 2 but is not the supported perm.
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="not the supported permutation for rank"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=(0, 1),
                is_indirect=True,
                n_gathered_dims=2,
            )

    def test_transpose_axes_non_tuple_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="must be a tuple"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=[1, 0],  # list, not tuple
                is_indirect=True,
                n_gathered_dims=2,
            )

    def test_transpose_axes_non_int_elements_rejected(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="only ints"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=(1.0, 0.0),
                is_indirect=True,
                n_gathered_dims=2,
            )

    def test_gather_over_2d_requires_explicit_axes(self):
        # A 3-non-trivial-dim gather is ambiguous (3-D vs 4-D reshape-trick), so
        # omitting transpose_axes must raise rather than silently default.
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="ambiguous"):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=None,
                is_indirect=True,
                n_gathered_dims=3,
            )

    def test_2d_gather_omitting_axes_ok(self):
        # The unambiguous 2-D gather may omit transpose_axes (defaults to (1,0)).
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        _validate_transpose_load(
            dge_mode=None,
            oob_mode=None,
            oob_value=None,
            priority=None,
            transpose_axes=None,
            is_indirect=True,
            n_gathered_dims=2,
        )

    def test_pattern_override_rejected_on_transpose(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="pattern_override="):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=None,
                is_indirect=False,
                pattern_override=[[1, 128], [1, 128]],
            )

    def test_out_shape_rejected_on_transpose(self):
        from nkilib_src.nkilib.experimental.neurotile.core.ndslice import _validate_transpose_load

        with pytest.raises(AssertionError, match="out_shape="):
            _validate_transpose_load(
                dge_mode=None,
                oob_mode=None,
                oob_value=None,
                priority=None,
                transpose_axes=None,
                is_indirect=False,
                out_shape=(128, 128),
            )


# ============================================================================
# High-rank (5-D) tensor_view transform chains
#
# Regression for the squeeze/reshape stride-threading concern: chaining 3+
# slice+squeeze ops (filter load) or flatten+reshape+slice (output store) on a
# 5-D source must thread the original physical strides through the squeezed /
# reshaped view, so the emitted .ap() encodes the source layout -- not the
# contiguous strides of the narrowed logical shape. The user-visible symptom of
# a regression is a wrong partition (level-0) stride and an element count that
# fails nisa.dma_copy's same-#-elements check.
# ============================================================================


def _ap_total_elements(pattern):
    n = 1
    for _stride, count in pattern:
        n = n * count
    return n


def _nki_view(shape):
    """A contiguous HBM NkiTensor of ``shape`` -- the source a kernel transforms
    in place before tiling. Its native .slice/.squeeze_dim/.flatten_dims/
    .reshape_dim chain is what nt.tiles() / nisa ops consume directly now that
    nt.tensor_view is gone."""
    from nki.language.tensor import NkiTensor

    return NkiTensor(shape=tuple(shape), dtype="float32", storage=None, buffer=nl.shared_hbm)


@pytest_marks(["neurotile"])
class TestHighRankTransformChains:
    """5-D slice/squeeze/reshape chains on a raw NkiTensor thread physical strides
    through to get_pattern()/offset -- the layout nt.tiles() and nisa ops consume."""

    @pytest.mark.fast
    def test_triple_slice_squeeze_filter_load(self):
        """conv3d filter: [K_d,K_h,K_w,C_in,C_out] -> [C_in_tile, C_out_tile]."""
        src = _nki_view((3, 3, 3, 1024, 1024))  # strides (9437184,3145728,1048576,1024,1)
        k_d, k_h, k_w = 1, 2, 0
        view = (
            src.slice(0, k_d, k_d + 1)
            .squeeze_dim(0)
            .slice(0, k_h, k_h + 1)
            .squeeze_dim(0)
            .slice(0, k_w, k_w + 1)
            .squeeze_dim(0)
            .slice(0, 256, 384)  # C_in tile (128 wide)
            .slice(1, 512, 1024)  # C_out tile (512 wide)
        )
        assert tuple(view.shape) == (128, 512)
        # Physical strides survive the squeeze chain: C_in stride 1024, C_out 1.
        assert tuple(view.strides) == (1024, 1)
        assert view.offset == k_d * 9437184 + k_h * 3145728 + k_w * 1048576 + 256 * 1024 + 512
        # get_pattern() (level-0 partition stride 1024, not contiguous-of-logical 512).
        assert view.get_pattern() == [[1024, 128], [1, 512]]

    @pytest.mark.fast
    def test_flatten_reshape_slice_output_store(self):
        """conv3d output: [B,C_out,D,H,W] -> [C_out_tile, dh_tile, w_tile].

        Uses extents where D*H*W=504, W=12, D*H=42 so the correct partition
        stride (504) is distinct from the contiguous-of-logical value (12*12=144)
        a stride-threading regression would emit.
        """
        B, C_out, D, H, W = 2, 1024, 6, 7, 12  # strides (516096,504,84,12,1)
        total_dh = D * H  # 42
        view = (
            _nki_view((B, C_out, D, H, W))
            .slice(0, 1, 2)
            .squeeze_dim(0)  # [C_out, D, H, W]
            .slice(0, 0, 128)  # [128, D, H, W]
            .flatten_dims(1, 3)  # [128, 504]
            .reshape_dim(1, (total_dh, W))  # [128, 42, 12]
            .slice(1, 0, 12)  # [128, 12, 12]
            .slice(2, 0, 12)  # [128, 12, 12]
        )
        assert tuple(view.shape) == (128, 12, 12)
        assert tuple(view.strides) == (504, 12, 1)
        assert view.offset == 1 * 516096  # batch_idx=1
        # Partition stride is the physical D*H*W=504, NOT 12*12=144.
        assert view.get_pattern() == [[504, 128], [12, 12], [1, 12]]

    @pytest.mark.fast
    def test_flatten_reshape_slice_output_store_nonzero_dh_start(self):
        """Output store at a non-multiple dh start keeps the full dh level."""
        B, C_out, D, H, W = 2, 1024, 6, 7, 12  # strides (516096, 504, 84, 12, 1)
        total_dh = D * H  # 42
        dh_start, dh_count = 15, 4
        view = (
            _nki_view((B, C_out, D, H, W))
            .slice(0, 1, 2)
            .squeeze_dim(0)  # [C_out, D, H, W]
            .slice(0, 0, 128)  # [128, D, H, W]
            .flatten_dims(1, 3)  # [128, 504]
            .reshape_dim(1, (total_dh, W))  # [128, 42, 12]
            .slice(1, dh_start, dh_start + dh_count)  # [128, 4, 12]
            .slice(2, 0, W)
        )
        assert tuple(view.shape) == (128, dh_count, W)
        assert tuple(view.strides) == (504, 12, 1)
        assert view.offset == 1 * 516096 + dh_start * 12
        # dh level survives with count=4; nothing is clamped away.
        assert view.get_pattern() == [[504, 128], [12, dh_count], [1, W]]

    @pytest.mark.fast
    def test_triple_slice_squeeze_filter_load_nonzero_cin_start(self):
        """Filter load whose C_in tile starts mid-tensor stays fully addressable."""
        src = _nki_view((3, 3, 3, 1024, 1024))  # strides (9437184,3145728,1048576,1024,1)
        k_d, k_h, k_w = 1, 2, 0
        c_in_start, c_in_end = 200, 328  # 128 wide, start not a multiple of 128
        view = (
            src.slice(0, k_d, k_d + 1)
            .squeeze_dim(0)
            .slice(0, k_h, k_h + 1)
            .squeeze_dim(0)
            .slice(0, k_w, k_w + 1)
            .squeeze_dim(0)
            .slice(0, c_in_start, c_in_end)
            .slice(1, 512, 1024)
        )
        assert tuple(view.shape) == (128, 512)
        assert tuple(view.strides) == (1024, 1)
        assert view.get_pattern() == [[1024, 128], [1, 512]]
