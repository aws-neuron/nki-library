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
Tests for factory functions: tiles(), blocks().

Full pipeline: factory -> __getitem__ -> tolist -> verify offsets.
Pure Python -- no NKI, no tracer, no device required.
"""

import nki.language as nl
import pytest
from nkilib_src.nkilib.experimental.neurotile.core._helpers import buffer_space
from nkilib_src.nkilib.experimental.neurotile.core.factories import (
    NDSlice,
    alloc_blocks,
    alloc_tiles,
    blocks,
    tiles,
)

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.negative_test_helpers import call_with_invalid_argument
from test.utils.pytest_test_metadata import pytest_marks

# ============================================================================
# tiles() -- basic construction
# ============================================================================


@pytest_marks(["neurotile"])
class TestTilesBasic:
    @pytest.mark.fast
    def test_2d_tiles(self):
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        assert isinstance(v, NDSlice)
        assert v.shape == (4, 4)
        assert v.element_shape == (512, 2048)
        assert v.tile_size == (128, 512)
        assert v.ndim == 2
        assert v._offset == 0
        assert v._source is src

    def test_3d_tiles_with_iteration_dim(self):
        """2D tile on 3D tensor -> dim 0 is the batch dim."""
        src = MockTensor(8, 128, 64)
        v = tiles(src, tile_size=(128, 64))
        assert v.ndim == 3
        # Cursor defaults to n_batch_dims -> shape skips batch.
        assert v.shape == (1, 1)
        # tile_size_of: dim 0 leaf has count == element_shape[0] (batch).
        assert v.tile_size == (8, 128, 64)
        assert v._grid.n_batch_dims == 1
        assert v._grid.outer_axis(0).count == 8

    def test_4d_tiles_with_2_iteration_dims(self):
        """2D tile on 4D tensor -> dims 0,1 are batch."""
        src = MockTensor(4, 8, 128, 64)
        v = tiles(src, tile_size=(128, 64))
        assert v.ndim == 4
        assert v._grid.n_batch_dims == 2
        assert v.shape == (1, 1)
        assert v._grid.outer_axis(0).count == 4
        assert v._grid.outer_axis(1).count == 8


# ============================================================================
# tiles() -- strides and offsets
# ============================================================================


@pytest_marks(["neurotile"])
class TestTilesStrides:
    @pytest.mark.fast
    def test_contiguous_strides(self):
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        assert v._strides == (2048, 1)


# ============================================================================
# HBM source validation -- a sliced source self-addresses
# ============================================================================


@pytest_marks(["neurotile"])
class TestHbmSourceValidation:
    """A sliced source tiles directly from its own self-describing layout."""

    @pytest.mark.fast
    def test_sliced_source_self_addresses(self):
        """A sliced source tiles directly: strides from its own pattern, offset
        from the slice itself (not double-applied)."""
        sliced = MockTensor(
            (8192, 128),
            pattern=[[1152, 8192], [1, 128]],  # parent row stride 1152
            offset=1024,
        )
        v = tiles(sliced, tile_size=(128, 128))
        assert v._source is sliced
        assert v._strides == (1152, 1)  # from the slice's own get_pattern()
        assert v._offset == 0  # slice self-addresses; offset is NOT re-applied

    def test_sliced_source_blocks_self_addresses(self):
        sliced = MockTensor(
            (8192, 128),
            pattern=[[1152, 8192], [1, 128]],
            offset=1024,
        )
        v = blocks(sliced, tile_size=(128, 128), block_size=(8, 1))
        assert v._source is sliced
        assert v._strides == (1152, 1)
        assert v._offset == 0

    def test_top_level_source_passes(self):
        src = MockTensor((1024, 128))
        v = tiles(src, tile_size=(128, 128))
        assert v._source is src
        # Contiguous: stride 0 = inner product = 128
        assert v._strides == (128, 1)

    def test_non_nki_source_is_native(self):
        """A contiguous offset-0 MockTensor tiles directly."""
        src = MockTensor(1024, 128)
        v = tiles(src, tile_size=(128, 128))
        assert v._source is src


@pytest_marks(["neurotile"])
class TestPhysicalStrides:
    """physical_strides reads NkiTensor.get_pattern() when available."""

    @pytest.mark.fast
    def test_reads_parent_strides_from_sliced_view(self):
        """A sliced view's get_pattern() reports the parent row stride, so it
        self-addresses: strides come from the slice itself, offset stays 0."""
        sliced = MockTensor(
            (8192, 128),
            pattern=[[1152, 8192], [1, 128]],  # parent row stride 1152
            offset=1024,
        )
        v = tiles(sliced, tile_size=(128, 128))
        assert v._source is sliced
        assert v._strides == (1152, 1)  # from the slice's own get_pattern()
        assert v._offset == 0  # slice self-addresses; offset not re-applied

    def test_multi_p_tile_block_from_sliced_source(self):
        """Regression: multi-P-tile block load on a column-sliced source.

        Before the self-addressing fix, block would derive strides from the
        slice's logical shape (128, 1), not the parent's (1152, 1), silently
        producing wrong APs. Exercises the attention_cte kv_producer shape.
        """
        sliced = MockTensor(
            (8192, 128),
            pattern=[[1152, 8192], [1, 128]],
            offset=1024,
        )
        v = blocks(sliced, tile_size=(128, 128), block_size=(8, 1))
        assert v._strides == (1152, 1)
        assert v._offset == 0
        # 8 P-tiles x 128 rows each, 1 F-tile x 128 cols
        assert v.shape == (8, 1)


# ============================================================================
# tiles() -- RemainderPolicy
# ============================================================================


@pytest_marks(["neurotile"])
class TestTilesRemainder:
    @pytest.mark.fast
    def test_default_handles_remainder(self):
        src = MockTensor(300, 512)
        v = tiles(src, tile_size=(128, 512))
        # Source extent is (300, 512); the iteration walk is ceil(300/128)*128 = 384.
        assert v._grid.element_shape == (300, 512)
        assert v.is_remainder is True
        assert v.shape == (3, 1)  # ceil(300/128)=3

    def test_skip_truncates(self):
        src = MockTensor(300, 512)
        v = tiles(src, tile_size=(128, 512), remainder="skip")
        assert v.element_shape == (256, 512)  # 300 // 128 * 128 = 256
        assert v.is_remainder is False
        assert v.shape == (2, 1)


# ============================================================================
# blocks()
# ============================================================================


@pytest_marks(["neurotile"])
class TestBlocks:
    @pytest.mark.fast
    def test_basic_blocks(self):
        src = MockTensor(512, 2048)
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        assert v.shape == (2, 2)  # block-level counts
        assert v.element_shape == (512, 2048)
        assert v.tile_size == (128, 512)

    def test_block_axes_layout(self):
        """Blocks introduce a (block, tile, leaf) axis triple per dim."""
        from nkilib_src.nkilib.experimental.neurotile.core.axis import AxisLabel

        src = MockTensor(512, 2048)
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        # Per dim: 3 axes -- block + tile + leaf.
        for d in (0, 1):
            axes = v._grid.axes_for(d)
            assert len(axes) == 3
            assert axes[0].label == AxisLabel.BLOCK
            assert axes[1].label == AxisLabel.TILE
        # Block step on dim 0 = block_size * tile_size = 2 * 128 = 256.
        assert v._grid.axes_for(0)[0].step == 256
        # Block step on dim 1 = 2 * 512 = 1024.
        assert v._grid.axes_for(1)[0].step == 1024

    def test_blocks_with_iteration_dim(self):
        """3D tensor with 2D tile + blocks: dim 0 is the batch dim."""
        src = MockTensor(8, 512, 2048)
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        assert v.ndim == 3
        assert v._grid.n_batch_dims == 1
        # Batch dim has a single elem axis.
        assert len(v._grid.axes_for(0)) == 1
        assert v._grid.axes_for(0)[0].count == 8
        # Tile dims have block+tile+leaf.
        assert len(v._grid.axes_for(1)) == 3

    @pytest.mark.fast
    def test_partial_block_clamps_when_descended_to_tiles(self):
        """A partial trailing block clamps its remainder at TILE granularity,
        not at the block as a whole.

        N=1792 with a 2-tile (2*512=1024 wide) N-block yields ceil(1792/1024)
        = 2 N-blocks. Block 1 spans columns [1024, 2048) nominally but the
        source ends at 1792, so only 768 columns (one full 512 tile + a 256
        partial) exist. The neurotile contract is: load a coalesced block,
        then descend to a tile-grid (``nt.tiles(block)``) for compute/store --
        a block view itself is iterate-next, not addressable as a monolith.

        The descent is where the remainder resolves: the trailing tile must
        auto-clamp to its real 256 width while full tiles stay 512. This is
        what lets a single matmul kernel handle remainder N without any
        special-casing -- the per-tile store walks only the real extent.
        """
        src = MockTensor(512, 1792)
        block_view = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        # 2 M-blocks (512/256), 2 N-blocks (ceil(1792/1024)); dim 1 is partial.
        assert block_view.shape == (2, 2)

        # Descend the trailing N-block to its tile-grid (no re-tile: inherits
        # the block's (128, 512) tiling).
        trailing_tiles = tiles(block_view[0, 1])
        assert trailing_tiles.shape == (2, 2)  # 2 P-tiles x 2 F-tiles

        # Full F-tile keeps the nominal 512; partial F-tile clamps to 256 and
        # is flagged a remainder.
        full_tile = trailing_tiles[0, 0]
        assert full_tile.element_shape == (128, 512)
        assert full_tile._grid.is_remainder is False

        partial_tile = trailing_tiles[0, 1]
        assert partial_tile.element_shape == (128, 256)
        assert partial_tile._grid.is_remainder is True


@pytest_marks(["neurotile"])
class TestBufferSpace:
    """``buffer_space`` reads the source's memory space from ``source.buffer``.
    A source without ``.buffer`` (test mock / numpy) is HBM by definition."""

    @pytest.mark.fast
    def test_detects_sbuf_from_source(self):
        assert buffer_space(MockTensor(128, 512, buffer=nl.sbuf)) == nl.sbuf

    @pytest.mark.fast
    def test_detects_psum_from_source(self):
        assert buffer_space(MockTensor(128, 512, buffer=nl.psum)) == nl.psum

    @pytest.mark.fast
    def test_hbm_source_resolves_hbm(self):
        assert buffer_space(MockTensor(128, 512, buffer=nl.shared_hbm)) == nl.shared_hbm

    @pytest.mark.fast
    def test_no_buffer_attr_defaults_hbm(self):
        # MockTensor has no .buffer -> HBM (keeps mock / numpy sources working).
        assert buffer_space(MockTensor(128, 512)) == nl.shared_hbm


@pytest_marks(["neurotile"])
class TestAllocTilesMisuseGuards:
    """alloc_tiles input validation."""

    # tile_size validation reuses _validate_tile_size.

    @pytest.mark.fast
    def test_tile_size_must_be_tuple(self):
        with pytest.raises(AssertionError, match="tile_size= must be a tuple"):
            call_with_invalid_argument(alloc_tiles, tile_size=128, buffer_type=nl.sbuf, dtype="float32")

    def test_tile_size_rank_at_least_2(self):
        with pytest.raises(AssertionError, match="at least 2 dims"):
            alloc_tiles(tile_size=(128,), buffer_type=nl.sbuf, dtype="float32")

    def test_tile_size_must_be_positive(self):
        with pytest.raises(AssertionError, match=r"tile_size\[0\] must be > 0"):
            alloc_tiles(tile_size=(0, 256), buffer_type=nl.sbuf, dtype="float32")

    # buffer_type / dtype required + named messages.

    def test_buffer_type_required(self):
        with pytest.raises(AssertionError, match=r"nt\.alloc_tiles\(\.\.\.\): buffer_type= is required"):
            alloc_tiles(tile_size=(128, 256), dtype="float32")

    def test_dtype_required(self):
        with pytest.raises(AssertionError, match=r"nt\.alloc_tiles\(\.\.\.\): dtype= is required"):
            alloc_tiles(tile_size=(128, 256), buffer_type=nl.sbuf)

    def test_buffer_type_psum_rejected(self):
        with pytest.raises(AssertionError, match="buffer_type=nl.psum is not supported"):
            alloc_tiles(tile_size=(128, 256), buffer_type=nl.psum, dtype="float32")

    def test_buffer_type_string_rejected(self):
        with pytest.raises(AssertionError, match="buffer_type= must be an nl.MemoryRegion"):
            call_with_invalid_argument(alloc_tiles, tile_size=(128, 256), buffer_type="sbuf", dtype="float32")

    # grid xor element_shape.

    def test_grid_and_element_shape_mutex(self):
        with pytest.raises(AssertionError, match=r"pass either grid= or element_shape=, not both"):
            alloc_tiles(
                tile_size=(128, 256),
                grid=(2, 2),
                element_shape=(256, 512),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    # grid validation.

    def test_grid_rank_must_match_tile_size(self):
        with pytest.raises(AssertionError, match="grid has 1 dims but tile_size has 2"):
            alloc_tiles(
                tile_size=(128, 256),
                grid=(2,),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    def test_grid_entry_must_be_positive(self):
        with pytest.raises(AssertionError, match=r"grid\[0\] must be > 0"):
            alloc_tiles(
                tile_size=(128, 256),
                grid=(0, 2),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    def test_grid_entry_must_be_int(self):
        with pytest.raises(AssertionError, match=r"grid\[0\] must be int"):
            call_with_invalid_argument(
                alloc_tiles,
                tile_size=(128, 256),
                grid=(2.0, 2),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    # element_shape validation.

    def test_element_shape_rank_must_match_tile_size(self):
        with pytest.raises(AssertionError, match="element_shape has 1 dims but tile_size has 2"):
            alloc_tiles(
                tile_size=(128, 256),
                element_shape=(1024,),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    def test_element_shape_entry_must_be_positive(self):
        with pytest.raises(AssertionError, match=r"element_shape\[1\] must be > 0"):
            alloc_tiles(
                tile_size=(128, 256),
                element_shape=(128, 0),
                buffer_type=nl.sbuf,
                dtype="float32",
            )


@pytest_marks(["neurotile"])
class TestAllocBlocksMisuseGuards:
    """alloc_blocks input validation. Inherits all alloc_tiles rules and
    adds block_size-specific guards."""

    # Inherited from alloc_tiles validation.

    @pytest.mark.fast
    def test_buffer_type_required(self):
        with pytest.raises(AssertionError, match=r"buffer_type= is required"):
            alloc_blocks(tile_size=(128, 256), block_size=(2, 2), dtype="float32")

    def test_dtype_required(self):
        with pytest.raises(AssertionError, match=r"dtype= is required"):
            alloc_blocks(tile_size=(128, 256), block_size=(2, 2), buffer_type=nl.sbuf)

    def test_buffer_type_psum_rejected(self):
        with pytest.raises(AssertionError, match="buffer_type=nl.psum is not supported"):
            alloc_blocks(
                tile_size=(128, 256),
                block_size=(2, 2),
                buffer_type=nl.psum,
                dtype="float32",
            )

    def test_grid_and_element_shape_mutex(self):
        with pytest.raises(AssertionError, match=r"pass either grid= or element_shape=, not both"):
            alloc_blocks(
                tile_size=(128, 256),
                block_size=(2, 2),
                grid=(2, 2),
                element_shape=(512, 1024),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    # block_size-specific guards.

    def test_block_size_must_be_tuple(self):
        with pytest.raises(AssertionError, match="block_size= must be a tuple"):
            call_with_invalid_argument(
                alloc_blocks,
                tile_size=(128, 256),
                block_size=2,
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    def test_block_size_rank_at_least_2(self):
        with pytest.raises(AssertionError, match="at least 2 dims"):
            alloc_blocks(
                tile_size=(128, 256),
                block_size=(2,),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    def test_block_size_must_be_positive(self):
        with pytest.raises(AssertionError, match=r"block_size\[0\] must be > 0"):
            alloc_blocks(
                tile_size=(128, 256),
                block_size=(0, 2),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    def test_block_size_rank_must_match_tile_size(self):
        with pytest.raises(AssertionError, match="block_size has 3 dims but tile_size has 2"):
            alloc_blocks(
                tile_size=(128, 256),
                block_size=(2, 2, 2),
                buffer_type=nl.sbuf,
                dtype="float32",
            )

    # Happy-path coverage lives in test_sbuf_remainder.py (inside @nki.jit
    # so the backend is active for nl.ndarray allocations).


# psum_pool input validation lives in test/core/test_psum_pool_api.py.


# ============================================================================
# Sharding through factory
# ============================================================================


@pytest_marks(["neurotile"])
class TestSliceSharding:
    """Slice-based sharding: view[block_range(...), :] narrows the view."""

    @pytest.mark.fast
    def test_concrete_block_shard_first_rank(self):
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        # 4 tiles on dim 0; num_shards=2 -> 2 owned per core; rank 0 owns tiles 0-1.
        v = tiles(src, tile_size=(128, 512))[block_range(rank=0, num_shards=2, total=4), :]
        assert v._grid.remaining[0] == 256  # 2 * 128
        assert v._offset == 0

    def test_concrete_block_shard_second_rank(self):
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        # rank 1 owns tiles [2, 3] -> offset = 2 * 128 * stride[0] = 2 * 128 * 2048.
        v = tiles(src, tile_size=(128, 512))[block_range(rank=1, num_shards=2, total=4), :]
        assert v._grid.remaining[0] == 256
        assert v._offset == 2 * 128 * 2048

    # Runtime-rank slice paths require CExpr-style operands (program_id);
    # they're exercised end-to-end by the device tests in
    # test/core/test_sharding_correctness.py and test_multicore_sharding.py.

    def test_two_dim_block_shard(self):
        """Slice on both dims composes."""
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))[
            block_range(rank=0, num_shards=2, total=4),
            block_range(rank=0, num_shards=2, total=4),
        ]
        # Each rank owns 2 tiles on each dim.
        assert v._grid.remaining[0] == 256
        assert v._grid.remaining[1] == 1024


# ============================================================================
# access_pattern=
# ============================================================================


@pytest_marks(["neurotile"])
class TestInputValidation:
    """tiles() rejects malformed inputs with clear errors instead of producing
    wrong output or letting internal code raise later."""

    @pytest.mark.fast
    def test_size_must_be_tuple(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="tile_size= must be a tuple"):
            call_with_invalid_argument(tiles, src, tile_size=128)

    def test_size_rank_at_least_2(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="at least 2 dims"):
            tiles(src, tile_size=())
        with pytest.raises(AssertionError, match="at least 2 dims"):
            tiles(src, tile_size=(128,))

    def test_size_must_be_positive(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"tile_size\[0\] must be > 0"):
            tiles(src, tile_size=(0, 256))
        with pytest.raises(AssertionError, match=r"tile_size\[0\] must be > 0"):
            tiles(src, tile_size=(-128, 256))

    def test_size_must_be_int(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="tile_size\\[0\\] must be int"):
            tiles(src, tile_size=(128.5, 256))

    def test_size_rank_cannot_exceed_view_rank(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="tile_size must not exceed the view's rank"):
            tiles(src, tile_size=(128, 256, 512, 1024))

    def test_size_exceeding_source_extent_is_single_partial_tile(self):
        # tile_size larger than the source extent on a dim is allowed: the grid
        # is a single partial tile there, with tile_size clamped to the extent.
        src = MockTensor(128, 64)
        v = tiles(src, tile_size=(128, 128))
        assert v.shape == (1, 1)  # ceil(64/128) == 1
        assert v.element_shape == (128, 64)
        assert v.tile_size == (128, 64)  # clamped to source extent on dim 1

    def test_remainder_invalid_string(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="remainder=bogus"):
            tiles(src, tile_size=(128, 256), remainder="bogus")

    def test_remainder_wrong_case(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="remainder=SKIP"):
            tiles(src, tile_size=(128, 256), remainder="SKIP")

    def test_remainder_wrong_type(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="remainder=42"):
            call_with_invalid_argument(tiles, src, tile_size=(128, 256), remainder=42)

    def test_ap_level_wrong_length(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"access_pattern\[0\] must be \[stride, count\]"):
            tiles(src, access_pattern=[[1024], [1, 1024]], tile_size=(128, 256))

    def test_ap_stride_must_be_positive(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"access_pattern\[0\]\[0\] \(stride\) must be > 0"):
            tiles(src, access_pattern=[[-1024, 512], [1, 1024]], tile_size=(128, 256))

    def test_ap_count_must_be_positive(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"access_pattern\[0\]\[1\] \(count\) must be > 0"):
            tiles(src, access_pattern=[[1024, 0], [1, 1024]], tile_size=(128, 256))

    def test_ap_higher_rank_than_source_accepted(self):
        # AP defines the view's rank; tile_size must match the AP rank.
        # Higher-rank AP currently requires single-tile coverage (TODO).
        src = MockTensor(512, 1024)
        v = tiles(
            src,
            access_pattern=[[1024, 512], [1, 2], [2, 512]],
            tile_size=(512, 2, 512),
        )
        assert v.element_shape == (512, 2, 512)
        assert v.tile_size == (512, 2, 512)

    def test_indirect_handle_source_rejected(self):
        # A handle that already carries a runtime (gather / dynamic-select)
        # offset must NOT be tiled directly: _resolve_source reads only the
        # compile-time .offset and .ap() drops the runtime offset, so the DMA
        # would silently read from the base (wrong data, no error). The
        # supported idiom is nt.tiles(base)[k]. Guards a real gap left when the
        # _pattern-keyed sliced-source check went dead on the NkiTensor migration.
        src = MockTensor(512, 1024, indirect=True)
        with pytest.raises(AssertionError, match="runtime"):
            tiles(src, tile_size=(128, 256))

    def test_indirect_handle_source_rejected_blocks(self):
        src = MockTensor(512, 1024, indirect=True)
        with pytest.raises(AssertionError, match="runtime"):
            blocks(src, tile_size=(128, 256), block_size=(2, 2))

    def test_non_indirect_handle_accepted(self):
        # The companion to the reject above: a plain (non-indirect) handle is
        # accepted, so the guard cannot regress into over-rejecting normal
        # sources. is_indirect() exists but returns False.
        src = MockTensor(512, 1024, indirect=False)
        v = tiles(src, tile_size=(128, 256))
        assert v.shape == (4, 4)


@pytest_marks(["neurotile"])
class TestMisuseGuards:
    """Reject argument combinations that are silently dropped or produce
    corrupted state. Each guard maps to a specific misuse pattern."""

    # Guard: NDSlice source rejects raw-source-only kwargs.

    def test_ndslice_source_rejects_remainder(self):
        src = MockTensor(512, 1024)
        v = tiles(src, tile_size=(128, 256))
        with pytest.raises(AssertionError, match="remainder= applies at construct time only"):
            tiles(v, tile_size=(128, 256), remainder="skip")

    # Guard: memory space is detected from the source -- SBUF builds an
    # SBUFLayout, PSUM is rejected (use psum_pool; operate on a bank's .data).

    @pytest.mark.fast
    def test_sbuf_source_builds_sbuf_layout(self):
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        src = MockTensor((512, 1024), buffer=nl.sbuf)
        v = tiles(src, tile_size=(128, 256))
        assert isinstance(v._layout, SBUFLayout)
        assert v.shape == (4, 4)

    def test_psum_source_rejected(self):
        src = MockTensor((128, 512), buffer=nl.psum)
        with pytest.raises(AssertionError, match="PSUM sources are not supported"):
            tiles(src, tile_size=(128, 256))

    def test_hbm_source_builds_hbm_layout(self):
        from nkilib_src.nkilib.experimental.neurotile.core.layout_hbm import HBMLayout

        src = MockTensor((512, 1024))  # no .buffer -> HBM
        v = tiles(src, tile_size=(128, 256))
        assert isinstance(v._layout, HBMLayout)
        assert v.shape == (4, 4)

    # Guard: bad source type -> named error, not a downstream AttributeError.

    @pytest.mark.fast
    def test_source_none_rejected(self):
        with pytest.raises(AssertionError, match="source is required"):
            tiles(None, tile_size=(128, 256))

    def test_source_scalar_rejected(self):
        with pytest.raises(AssertionError, match=r"must be a tensor-like object with a .shape"):
            tiles(5, tile_size=(128, 256))

    def test_blocks_source_none_rejected(self):
        with pytest.raises(AssertionError, match="source is required"):
            blocks(None, tile_size=(128, 256), block_size=(2, 2))

    # Guard: access_pattern= on an SBUF source would silently drop the AP's
    # strides (the on-chip layout recomputes strides from the tile grid).

    @pytest.mark.fast
    def test_sbuf_source_rejects_access_pattern(self):
        src = MockTensor((128, 1024), buffer=nl.sbuf)
        with pytest.raises(AssertionError, match="access_pattern= is not supported for SBUF sources"):
            tiles(src, tile_size=(128, 256), access_pattern=[[1024, 128], [1, 1024]])

    # Guard 5+: spec-based sharding validation removed with the spec class.
    # Slice-based sharding validates through validate_index_key
    # (covered in test_indexing_validation.py).

    # Guard 9: AP must not address bytes past source's flat extent.

    def test_ap_addresses_past_source_rejected(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"access_pattern addresses element offset"):
            tiles(src, tile_size=(128, 256), access_pattern=[[1024, 1024], [1, 1024]])

    def test_ap_at_source_extent_accepted(self):
        # Contiguous case -- AP exactly fills the source.
        src = MockTensor(512, 1024)
        v = tiles(src, tile_size=(128, 256), access_pattern=[[1024, 512], [1, 1024]])
        assert v.element_shape == (512, 1024)

    def test_ap_count_below_source_shape_accepted(self):
        # Strided / windowed views: count < source.shape[d].
        src = MockTensor(512, 1024)
        v = tiles(src, tile_size=(128, 256), access_pattern=[[2048, 256], [1, 1024]])
        assert v.element_shape == (256, 1024)


@pytest_marks(["neurotile"])
class TestAccessPattern:
    @pytest.mark.fast
    def test_basic_pattern(self):
        # AP describes source layout (strides + element_shape). Tile size
        # is orthogonal and must be passed explicitly.
        src = MockTensor(512, 2048)
        v = tiles(src, access_pattern=[[2048, 512], [1, 2048]], tile_size=(128, 512))
        assert v._strides == (2048, 1)
        assert v.tile_size == (128, 512)
        assert v.shape == (4, 4)

    def test_ap_requires_size(self):
        src = MockTensor(512, 2048)
        with pytest.raises(AssertionError, match="tile_size= is required"):
            tiles(src, access_pattern=[[2048, 512], [1, 2048]])

    def test_ap_on_ndslice_rejected(self):
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        with pytest.raises(AssertionError, match="access_pattern= is not supported"):
            tiles(v, access_pattern=[[2048, 512], [1, 2048]], tile_size=(128, 512))

    def test_ap_compose_with_slice_sharding(self):
        """AP + slice indexing: multi-core view over a custom-strided source."""
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(
            src,
            access_pattern=[[2048, 512], [1, 2048]],
            tile_size=(128, 512),
        )[block_range(rank=0, num_shards=2, total=4), :]
        # Each core owns 2 of 4 row-tiles.
        assert v.shape == (2, 4)

    def test_ap_compose_with_remainder_skip(self):
        src = MockTensor(300, 500)
        v = tiles(
            src,
            access_pattern=[[500, 300], [1, 500]],
            tile_size=(128, 128),
            remainder="skip",
        )
        assert v.shape == (2, 3)
        assert v.is_remainder is False

    def test_ap_higher_rank_than_source(self):
        """3-level AP on a 2-D source produces a 3-D logical view.

        AP defines the view's rank, shape, and strides; source rank is
        only relevant to byte-range validation. tile_size matches the
        AP rank (3-D). Higher-rank AP currently requires single-tile
        coverage (TODO: lift this to support tile-grid iteration over
        a multi-level walk).
        """
        src = MockTensor(512, 1024)
        v = tiles(
            src,
            access_pattern=[[1024, 512], [1, 2], [2, 512]],
            tile_size=(512, 2, 512),  # single tile = whole view
        )
        assert v.element_shape == (512, 2, 512)
        assert v.tile_size == (512, 2, 512)
        # Strides reflect the 3-level walk (no rank conflation).
        assert v._strides == (1024, 1, 2)

    def test_ap_higher_rank_than_source_multi_tile_rejected(self):
        """Higher-rank AP with multi-tile grid is rejected (TODO).

        A multi-level walk's per-tile DMA descriptor needs offset
        adjustments that the AP emitter does not yet derive. Until
        that's added, restrict to the single-tile case.
        """
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="single-tile load"):
            tiles(
                src,
                access_pattern=[[1024, 512], [1, 2], [2, 512]],
                tile_size=(128, 1, 128),  # multi-tile -- not yet supported
            )

    def test_ap_lower_rank_than_source(self):
        """2-level AP on a 3-D source flattens it to a 2-D logical view.

        The source is a 3-D tensor (B, M, N). The user collapses it to
        a 2-D (B*M, N) view via a 2-level AP. Resulting view operates
        as a plain 2-D tile grid -- B and M are now folded into the P
        dim of the AP, and source.shape's 3-D structure is irrelevant
        to the view's logical 2-D shape.
        """
        src = MockTensor(2, 256, 512)
        # AP: P stride = 512 (one source row), count = 2*256 = 512 (B*M);
        #     F stride = 1, count = 512 (N).
        v = tiles(
            src,
            access_pattern=[[512, 512], [1, 512]],
            tile_size=(128, 256),
        )
        assert v.element_shape == (512, 512)
        assert v.tile_size == (128, 256)
        assert v.shape == (4, 2)
        assert v._strides == (512, 1)


# ============================================================================
# NDSlice source pass-through
# ============================================================================


@pytest_marks(["neurotile"])
class TestNDSliceSource:
    @pytest.mark.fast
    def test_pass_through(self):
        """tiles(existing_ndslice) rebuilds with tile-level stacks."""
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        v2 = tiles(v)
        # Same shape, same element_shape, same strides -- just fresh Grid
        assert v2.shape == v.shape
        assert v2.element_shape == v.element_shape
        assert v2._strides == v._strides

    def test_pass_through_strips_blocks(self):
        """tiles(block_view) strips block level -> tile iteration."""
        src = MockTensor(512, 2048)
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        assert v.shape == (2, 2)  # block level
        v2 = tiles(v)
        assert v2.shape == (4, 4)  # tile level (blocks stripped)

    def test_retile(self):
        """tiles(existing_ndslice, tile_size=...) re-tiles."""
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        v2 = tiles(v, tile_size=(256, 1024))
        assert v2.tile_size == (256, 1024)
        assert v2.shape == (2, 2)

    def test_retile_then_slice(self):
        """tiles(view)[slice, :] narrows the rebuilt view."""
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        v2 = tiles(v)[block_range(rank=1, num_shards=2, total=4), :]
        assert v2._grid.remaining[0] == 256
        assert v2._offset == 2 * 128 * 2048


# ============================================================================
# Re-tile on a sharded source preserves sharding state
# ============================================================================


@pytest_marks(["neurotile"])
class TestRetileOnShardedSource:
    """Re-tile on a sliced (sharded) view inherits the slice's narrowing."""

    @pytest.mark.fast
    def test_retile_block_sliced_narrower_tile(self):
        """Block-sliced view -> re-tile to smaller tile -> still sees owned rows."""
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))[block_range(rank=1, num_shards=2, total=4), :]
        assert v._grid.remaining[0] == 256
        assert v._offset == 2 * 128 * 2048

        # Re-tile to 64-row tiles within that shard.
        v2 = tiles(v, tile_size=(64, 512))
        assert v2.tile_size == (64, 512)
        assert v2._grid.remaining[0] == 256
        assert v2._offset == 2 * 128 * 2048
        assert v2.shape == (4, 4)


@pytest_marks(["neurotile"])
class TestBlocksWithSlicedSharding:
    """blocks(...)[block_range(...), :] narrows the block view."""

    @pytest.mark.fast
    def test_block_shard_on_block_dim(self):
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        # 2 blocks on dim 0 (block_size=2).
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))[block_range(rank=1, num_shards=2, total=2), :]
        # rank 1 owns 1 block starting at block index 1.
        assert v._grid.remaining[0] == 256
        assert v._offset == 2 * 128 * 2048


@pytest_marks(["neurotile"])
class TestRetileBlockShardSubdivide:
    """Sliced + re-tile to smaller tile."""

    def _sliced(self, rank=0):
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(1024, 2048)
        return tiles(src, tile_size=(128, 512))[block_range(rank=rank, num_shards=2, total=8), :]

    @pytest.mark.fast
    def test_half_tile_size_divides(self):
        v = self._sliced()
        v2 = tiles(v, tile_size=(64, 512))
        assert v2.tile_size == (64, 512)
        assert v2.shape[0] == v._grid.remaining[0] // 64

    def test_quarter_tile_size(self):
        v = self._sliced()
        v2 = tiles(v, tile_size=(32, 512))
        assert v2.shape[0] == v._grid.remaining[0] // 32

    def test_equal_tile_size_noop(self):
        v = self._sliced()
        v2 = tiles(v, tile_size=(128, 512))
        assert v2._grid.remaining == v._grid.remaining
        assert v2._offset == v._offset


@pytest_marks(["neurotile"])
class TestRetileMultiDimShard:
    """Slice on both dims, then re-tile."""

    @pytest.mark.fast
    def test_block_block_both_dims_compatible(self):
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))[
            block_range(rank=0, num_shards=2, total=4),
            block_range(rank=0, num_shards=2, total=4),
        ]
        v2 = tiles(v, tile_size=(64, 256))
        assert v2.tile_size == (64, 256)


# ============================================================================
# Full pipeline: factory -> index -> iterate -> verify offsets
# ============================================================================


@pytest_marks(["neurotile"])
class TestFullPipeline:
    @pytest.mark.fast
    def test_tiles_index_and_offset(self):
        """tiles()[i, j] produces correct offset."""
        src = MockTensor(256, 1024)
        v = tiles(src, tile_size=(128, 512))
        # strides = (1024, 1)

        tile = v[1, 1]
        # Offset: 1 * 128 * 1024 + 1 * 512 * 1 = 131072 + 512
        assert tile._offset == 131072 + 512
        assert tile.element_shape == (128, 512)

    def test_blocks_index_and_offset(self):
        """blocks()[bi, bj][ti, tj] -> correct compound offset."""
        src = MockTensor(512, 2048)
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        # strides = (2048, 1)

        block = v[1, 0]  # block (1, 0): step=256, offset=1*256*2048
        assert block._offset == 1 * 256 * 2048
        assert block.element_shape == (256, 1024)

        tile = block[0, 1]  # tile (0, 1) within block: step=128, offset += 1*512*1
        assert tile._offset == 1 * 256 * 2048 + 1 * 512 * 1

    def test_iterate_all_tiles(self):
        """Enumerate all tiles -- verify count and offset progression."""
        src = MockTensor(256, 1024)
        v = tiles(src, tile_size=(128, 512))
        # 2x2 grid

        all_tiles = []
        for row in v.tolist():
            for tile in row.tolist():
                all_tiles.append(tile)

        assert len(all_tiles) == 4
        # Offsets: (0,0)=0, (0,1)=512, (1,0)=128*1024, (1,1)=128*1024+512
        assert all_tiles[0]._offset == 0
        assert all_tiles[1]._offset == 512
        assert all_tiles[2]._offset == 128 * 1024
        assert all_tiles[3]._offset == 128 * 1024 + 512

    def test_iteration_dim_enumerate(self):
        """3D tensor: explicit-dim iterate the batch dim."""
        src = MockTensor(4, 128, 64)
        v = tiles(src, tile_size=(128, 64))

        # Cursor doesn't auto-iterate the batch dim. Use dim=0 explicitly.
        children = v.tolist(dim=0)
        assert len(children) == 4
        for i, child in enumerate(children):
            assert child._offset == i * 128 * 64

    def test_blocks_enumerate_full(self):
        """Enumerate blocks, verify each block's scope and offset."""
        src = MockTensor(512, 2048)
        v = blocks(src, tile_size=(128, 512), block_size=(2, 2))

        block_list = []
        for row in v.tolist():
            for block in row.tolist():
                block_list.append(block)

        assert len(block_list) == 4  # 2x2 blocks
        # Block (0,0): offset=0, remaining=(256, 1024)
        assert block_list[0]._offset == 0
        assert block_list[0].element_shape == (256, 1024)
        # Block (1,1): offset = 1*256*2048 + 1*1024*1 = 525312
        assert block_list[3]._offset == 256 * 2048 + 1024

    def test_tolist_dim_provides_items(self):
        """view.tolist(dim=d) materializes sub-views along dim d."""
        src = MockTensor(256, 1024)
        v = tiles(src, tile_size=(128, 512))

        items = v.tolist(dim=0)
        assert len(items) == 2
        assert items[0].element_shape[0] == 128
        assert items[1].element_shape[0] == 128
        assert items[0]._offset == 0
        assert items[1]._offset == 128 * 1024

    def test_remainder_present_at_parent(self):
        """Parent view's is_remainder flag is True for partial trailing tile."""
        src = MockTensor(300, 512)
        v = tiles(src, tile_size=(128, 512))
        assert v.is_remainder is True
        # 3 children: two whole + one trailing partial.
        items = v.tolist()
        assert len(items) == 3

    def test_enumerate_indices(self):
        """_enumerate returns correct indices."""
        src = MockTensor(256, 1024)
        v = tiles(src, tile_size=(128, 512))
        pairs = v._enumerate()
        assert pairs[0][0] == 0
        assert pairs[1][0] == 1

    def test_chained_transforms(self):
        """Factory -> index -> transform -> verify strides."""
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        tile = v[0, 0]
        reshaped = tile.reshape_dim(1, (8, 64))
        assert reshaped.element_shape == (128, 8, 64)
        assert reshaped.tile_size == (128, 8, 64)  # single tile covering new shape
        assert reshaped._offset == 0


# ============================================================================
# SBUF blocks
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFBlocks:
    @pytest.mark.fast
    def test_blocks_sbuf_basic(self):
        """blocks(sbuf, tile_size, block_size) on an SBUF source creates block-level Grid."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        sbuf = MockTensor((128, 1024), buffer=nl.sbuf)
        v = blocks(sbuf, tile_size=(128, 512), block_size=(1, 2))
        # 1024 / 512 = 2 tiles in F, block_size (1,2) = 1 block.
        assert v.shape == (1, 1)
        assert isinstance(v._layout, SBUFLayout)
        # Per-dim layout: dim 1 has block + tile + leaf.
        assert len(v._grid.axes_for(1)) == 3
        assert v._grid.is_blocked(1)


# ============================================================================
# Fold recipe
# ============================================================================


@pytest_marks(["neurotile"])
class TestFoldRecipe:
    @pytest.mark.fast
    def test_free_dim_fold_ap_override(self):
        """fold(1, 2) on (128, 4, 512) -- both free dims, gets AP override."""
        src = MockTensor(128, 4, 512)
        v = tiles(src, tile_size=(128, 4, 512))
        tile = v[0, 0, 0]
        folded = tile.fold(1, 2)
        assert folded._dma_override is not None
        assert folded._dma_override[0] == "ap_override"
        assert folded.element_shape == (128, 2048)

    def test_partition_fold_has_recipe(self):
        """fold(1, 0) on (128, 4, 512) -- into P-dim, recipe attached."""
        src = MockTensor(128, 4, 512)
        v = tiles(src, tile_size=(128, 4, 512))
        tile = v[0, 0, 0]
        folded = tile.fold(1, 0)
        assert folded._dma_override is not None
        K, P_per, fold_stride, base_pattern = folded._dma_override[1]
        assert K == 4  # 4 slices from dim 1
        assert P_per == 128  # P-dim size

    def test_fold_from_p_dim_has_recipe(self):
        """fold(0, 1) on (4, 128, 512) -- src_dim is P, recipe attached."""
        src = MockTensor(4, 128, 512)
        v = tiles(src, tile_size=(4, 128, 512))
        tile = v[0, 0, 0]
        folded = tile.fold(0, 1)
        assert folded._dma_override is not None
        K = folded._dma_override[1][0]
        assert K == 4


# ============================================================================
# SBUF tiles -- remainder handling (pure Python: grid shape + strides only)
# ============================================================================


@pytest_marks(["neurotile"])
class TestSBUFTilesRemainder:
    """Test SBUF tile remainder: ceiling division, stride correctness.

    Tests that require indexing/data access use NKI and live in
    test/test_sbuf_remainder.py.
    """

    def _make_sbuf(self, shape):
        return MockTensor(shape, buffer=nl.sbuf)

    @pytest.mark.fast
    def test_sbuf_tiles_exact_fit(self):
        """No remainder: 1024 / 512 = 2 tiles exactly."""
        sbuf = self._make_sbuf((128, 1024))
        v = tiles(sbuf, tile_size=(128, 512))
        assert v.shape == (1, 2)
        assert v.is_remainder is False

    def test_sbuf_tiles_f_remainder(self):
        """F-remainder: 1792 / 512 = 3 full + 1 remainder (256)."""
        sbuf = self._make_sbuf((128, 1792))
        v = tiles(sbuf, tile_size=(128, 512))
        assert v.shape == (1, 4)
        assert v.is_remainder is True

    def test_sbuf_tiles_small_remainder(self):
        """Small F-remainder: 640 / 512 = 1 full + 1 remainder (128)."""
        sbuf = self._make_sbuf((128, 640))
        v = tiles(sbuf, tile_size=(128, 512))
        assert v.shape == (1, 2)
        assert v.is_remainder is True

    def test_sbuf_tiles_p_remainder(self):
        """P-remainder: 300 / 128 = 2 full + 1 remainder (44)."""
        sbuf = self._make_sbuf((300, 512))
        v = tiles(sbuf, tile_size=(128, 512))
        assert v.shape == (3, 1)
        assert v.is_remainder is True

    def test_sbuf_stride_no_remainder(self):
        """Non-remainder strides: d=0 = F-width, d=1 = tile_f."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        sbuf = self._make_sbuf((128, 2048))
        v = tiles(sbuf, tile_size=(128, 512))
        assert isinstance(v._layout, SBUFLayout)
        assert v._layout.strides == (2048, 512)

    def test_sbuf_stride_with_remainder(self):
        """Remainder strides: d=0 uses actual remaining[1], not padded."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        sbuf = self._make_sbuf((128, 1792))
        v = tiles(sbuf, tile_size=(128, 512))
        assert isinstance(v._layout, SBUFLayout)
        assert v._layout.strides == (1792, 512)


# ============================================================================
# nt.blocks() full contract (PR 1): NDSlice dispatch + rejections
# ============================================================================


@pytest_marks(["neurotile"])
class TestBlocksRequiresBlockSize:
    """block_size= is mandatory; no form of nt.blocks() returns a tile-level view."""

    @pytest.mark.fast
    def test_explicit_none_block_size_rejects(self):
        src = MockTensor(512, 2048)
        with pytest.raises(AssertionError, match="block_size= is required"):
            call_with_invalid_argument(blocks, src, tile_size=(128, 512), block_size=None)


@pytest_marks(["neurotile"])
class TestBlocksMisuseGuards:
    """blocks()-specific guards: block_size validation and clear error
    attribution for blocks-level rules (vs forwarded tiles() messages)."""

    # Guard BG3/BG7: block_size error messages reference "block_size", not "size".

    @pytest.mark.fast
    def test_block_size_must_be_tuple(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="block_size= must be a tuple"):
            call_with_invalid_argument(blocks, src, tile_size=(128, 256), block_size=2)

    def test_block_size_rank_at_least_2(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="block_size= must have at least 2 dims"):
            blocks(src, tile_size=(128, 256), block_size=(2,))

    def test_block_size_must_be_positive(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"block_size\[0\] must be > 0"):
            blocks(src, tile_size=(128, 256), block_size=(0, 2))
        with pytest.raises(AssertionError, match=r"block_size\[0\] must be > 0"):
            blocks(src, tile_size=(128, 256), block_size=(-1, 2))

    def test_block_size_must_be_int(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"block_size\[0\] must be int"):
            blocks(src, tile_size=(128, 256), block_size=(2.0, 2))

    def test_block_size_empty_rejects(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match="block_size= must have at least 2 dims"):
            blocks(src, tile_size=(128, 256), block_size=())

    # Guard BG2: block_size rank must not exceed source rank.

    def test_block_size_rank_exceeds_source_rank(self):
        src = MockTensor(512, 1024)  # 2-D
        with pytest.raises(AssertionError, match="block_size has 3 dims but source has 2"):
            blocks(src, tile_size=(128, 256), block_size=(2, 2, 2))

    def test_block_size_rank_equal_to_source_rank_accepted(self):
        # 2-D source with 2-D block_size: block grid (2, 2).
        src = MockTensor(512, 1024)
        v = blocks(src, tile_size=(128, 256), block_size=(2, 2))
        assert v.shape == (2, 2)

    def test_block_size_rank_below_source_rank_accepted(self):
        # 3-D source with 2-D block_size is fine; block_size auto-pads.
        src = MockTensor(2, 512, 1024)
        v = blocks(src, tile_size=(128, 256), block_size=(2, 2))
        assert v.element_shape == (2, 512, 1024)

    # Guard BG8: tile_size required for raw sources, error names blocks().

    def test_raw_source_requires_tile_size_message_names_blocks(self):
        src = MockTensor(512, 1024)
        with pytest.raises(AssertionError, match=r"nt\.blocks\(<raw source>, \.\.\.\): tile_size= is required"):
            blocks(src, block_size=(2, 2))


@pytest_marks(["neurotile"])
class TestBlocksNDSlicePromoteHBM:
    """Row 1 (HBM): nt.blocks(hbm_tile_view, block_size=) -- pure promote."""

    @pytest.mark.fast
    def test_promote_unsharded(self):
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))  # (4, 4) tiles
        b = blocks(v, block_size=(2, 2))
        assert isinstance(b, NDSlice)
        assert b.tile_size == (128, 512)
        assert b.element_shape == (512, 2048)
        # Each dim now has block + tile + leaf.
        for d in (0, 1):
            assert len(b._grid.axes_for(d)) == 3
        assert b._grid.block_size == (2, 2)

    def test_promote_already_blocked_view(self):
        """Row 1 + pre-blocked input: strip existing block level, then re-block."""
        src = MockTensor(512, 2048)
        b1 = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        b2 = blocks(b1, block_size=(4, 2))
        assert b2._grid.block_size == (4, 2)
        assert b2.tile_size == (128, 512)


@pytest_marks(["neurotile"])
class TestBlocksNDSliceRetilePromoteHBM:
    """Row 2: nt.blocks(hbm_view, tile_size=, block_size=) -- re-tile + promote."""

    @pytest.mark.fast
    def test_retile_unsharded(self):
        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        b = blocks(v, tile_size=(64, 512), block_size=(2, 2))
        assert b.tile_size == (64, 512)
        # block + tile + leaf on dim 0.
        assert len(b._grid.axes_for(0)) == 3


@pytest_marks(["neurotile"])
class TestBlocksNDSlicePromoteSBUF:
    """Row 3: nt.blocks(sbuf_view, block_size=) + row 3 re-tile on unsharded SBUF."""

    def _make_sbuf_tiles(self, shape=(128, 1024), tile_size=(128, 512)):
        """Construct an unsharded SBUF tile-level view."""
        sbuf = MockTensor(shape, buffer=nl.sbuf)
        return tiles(sbuf, tile_size=tile_size)

    @pytest.mark.fast
    def test_promote_unsharded_sbuf(self):
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        v = self._make_sbuf_tiles()
        b = blocks(v, block_size=(1, 2))
        assert isinstance(b._layout, SBUFLayout)
        assert b._grid.block_size == (1, 2)
        # Same source, same offset -- no Layout rebuild
        assert b._layout.source is v._layout.source
        assert b._layout.offset == v._layout.offset

    def test_retile_and_promote_unsharded_sbuf(self):
        """Row 3 extended: re-tile on unsharded SBUF + promote."""
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        v = self._make_sbuf_tiles()
        b = blocks(v, tile_size=(128, 256), block_size=(1, 2))
        assert isinstance(b._layout, SBUFLayout)
        assert b.tile_size == (128, 256)
        assert b._grid.block_size == (1, 2)


@pytest_marks(["neurotile"])
class TestBlocksOnSlicedHBM:
    """blocks() on a slice-sharded view inherits the slice narrowing."""

    @pytest.mark.fast
    def test_blocks_on_slice_sharded(self):
        """Slice the tile view first, then promote to blocks."""
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))[block_range(rank=1, num_shards=2, total=4), :]
        b = blocks(v, block_size=(2, 2))
        # block_size triple on each dim.
        assert len(b._grid.axes_for(0)) == 3
        assert b._offset == 2 * 128 * 2048
        assert b._grid.remaining[0] == 256

    def test_blocks_with_retile_then_slice(self):
        """Re-tile + promote + slice on the resulting view."""
        from nkilib_src.nkilib.experimental.neurotile.core.shard_helpers import block_range

        src = MockTensor(512, 2048)
        v = tiles(src, tile_size=(128, 512))
        b = blocks(v, tile_size=(64, 512), block_size=(2, 2))[
            block_range(rank=0, num_shards=2, total=4),
            :,
        ]
        assert b.tile_size == (64, 512)
        assert b._grid.remaining[0] == 256


@pytest_marks(["neurotile"])
class TestBlocksNDSliceRawOnlyParams:
    """raw-source-only params (access_pattern=) rejected on an NDSlice source."""

    def _view(self):
        return tiles(MockTensor(512, 2048), tile_size=(128, 512))

    @pytest.mark.fast
    def test_reject_access_pattern(self):
        v = self._view()
        with pytest.raises(AssertionError, match="access_pattern= is only for raw sources"):
            blocks(v, access_pattern=[[2048, 1], [1, 2048]], block_size=(2, 2))


@pytest_marks(["neurotile"])
class TestBlocksEquivalence:
    """nt.blocks(raw, tile_size=, block_size=) equivalent to
    nt.blocks(nt.tiles(raw, tile_size=), block_size=)."""

    @pytest.mark.fast
    def test_promote_matches_construct_hbm(self):
        src = MockTensor(512, 2048)
        b_construct = blocks(src, tile_size=(128, 512), block_size=(2, 2))
        b_promote = blocks(tiles(src, tile_size=(128, 512)), block_size=(2, 2))
        # Both produce equivalent block-level views.
        assert _axis_tuples_equal(b_construct._grid.axes, b_promote._grid.axes)
        assert b_construct.element_shape == b_promote.element_shape
        assert b_construct._offset == b_promote._offset
        assert b_construct._grid.block_size == b_promote._grid.block_size


def _axis_tuples_equal(a, b):
    """Structural equality on axis tuples (NKIObject lacks __eq__)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if (x.count, x.step, x.dim, x.label) != (y.count, y.step, y.dim, y.label):
            return False
    return True

    def test_promote_matches_construct_sbuf(self):
        sbuf = MockTensor((128, 1024), buffer=nl.sbuf)
        b_construct = blocks(sbuf, tile_size=(128, 512), block_size=(1, 2))
        v_tiles = tiles(sbuf, tile_size=(128, 512))
        b_promote = blocks(v_tiles, block_size=(1, 2))
        # Both paths produce equivalent block-level views: same tile_size,
        # same element_shape, same offset.
        assert b_construct.tile_size == b_promote.tile_size
        assert b_construct.element_shape == b_promote.element_shape
        assert b_construct._layout.offset == b_promote._layout.offset


@pytest_marks(["neurotile"])
class TestSBUFRetileUnsharded:
    """_retile_sbuf_ndslice on an unsharded view."""

    @pytest.mark.fast
    def test_retile_unsharded_sbuf_still_works(self):
        """Regression: re-tile on a plain unsharded SBUF view."""
        sbuf = MockTensor((128, 1024), buffer=nl.sbuf)
        v = tiles(sbuf, tile_size=(128, 512))
        v2 = tiles(v, tile_size=(128, 256))
        assert v2.tile_size == (128, 256)
        assert not v2._grid.is_sharded(0)
        assert not v2._grid.is_sharded(1)


@pytest_marks(["neurotile"])
class TestSBUFLayoutRetile:
    """SBUFLayout.retile: stride recalc for a new tile_size, same memory."""

    @pytest.mark.fast
    def test_retile_rebuilds_strides(self):
        from nkilib_src.nkilib.experimental.neurotile.core.layout_sbuf import SBUFLayout

        sbuf = MockTensor((128, 1024), buffer=nl.sbuf)
        v = tiles(sbuf, tile_size=(128, 512))
        old_strides = v._layout.strides
        new_layout = v._layout.retile((128, 256), (128, 1024))
        assert isinstance(new_layout, SBUFLayout)
        assert new_layout.alloc_tile_size == (128, 256)
        # Source + offset preserved
        assert new_layout.source is v._layout.source
        assert new_layout.offset == v._layout.offset
        # Strides recomputed (different tile_size -> different F-strides)
        assert new_layout.strides != old_strides


# ============================================================================
# Shape/level attributes: tile_shape, block_shape, is_tiled, is_blocked
# ============================================================================


@pytest_marks(["neurotile"])
class TestShapeLevelAttributes:
    """``view.tile_shape`` / ``view.block_shape`` / ``view.is_tiled`` /
    ``view.is_blocked`` -- consume-stable per-level descriptors."""

    # --- is_tiled / is_blocked ---

    @pytest.mark.fast
    def test_tiles_is_tiled_not_blocked(self):
        v = tiles(MockTensor(512, 2048), tile_size=(128, 512))
        assert v.is_tiled is True
        assert v.is_blocked is False

    def test_blocks_is_tiled_and_blocked(self):
        v = blocks(MockTensor(512, 2048), tile_size=(128, 512), block_size=(2, 2))
        assert v.is_tiled is True
        assert v.is_blocked is True

    # --- tile_shape ---

    def test_tiles_tile_shape_2d(self):
        v = tiles(MockTensor(512, 2048), tile_size=(128, 512))
        # 512/128 = 4, 2048/512 = 4
        assert v.tile_shape == (4, 4)

    def test_tiles_tile_shape_with_batch_dim(self):
        """3-D tensor with 2-D tile_size: batch dim 0 reports 1 (one batch-slab per
        batch position is the unit), then 256/128=2, 1024/512=2."""
        v = tiles(MockTensor(8, 256, 1024), tile_size=(128, 512))
        assert v.tile_shape == (1, 2, 2)

    def test_tiles_tile_shape_uneven_extent_ceils(self):
        """Tile count uses ceil-div (partial trailing tile counts as 1 full)."""
        v = tiles(MockTensor(384, 2048), tile_size=(128, 512))  # 384/128 = 3 exact
        assert v.tile_shape == (3, 4)

    def test_blocks_tile_shape(self):
        v = blocks(MockTensor(1024, 2048), tile_size=(128, 512), block_size=(2, 2))
        # tile_shape ignores block grouping -- it's element_shape / tile_size
        assert v.tile_shape == (8, 4)

    # --- block_shape ---

    def test_blocks_block_shape_2d(self):
        v = blocks(MockTensor(1024, 2048), tile_size=(128, 512), block_size=(2, 2))
        # block_shape = element_shape / (block_size * tile_size)
        # = (1024 / (2*128), 2048 / (2*512)) = (4, 2)
        assert v.block_shape == (4, 2)

    def test_blocks_block_shape_with_batch_dim(self):
        """3-D tensor, 2-D tile/block. Batch dim has no block axis -> count 1."""
        v = blocks(MockTensor(4, 1024, 2048), tile_size=(128, 512), block_size=(2, 2))
        # Block-grid count for batch dim 0 is 1 (no block axis on it).
        assert v.block_shape == (1, 4, 2)

    def test_tiles_block_shape_is_none(self):
        v = tiles(MockTensor(512, 2048), tile_size=(128, 512))
        assert v.block_shape is None

    # --- consume stability: tile_shape survives indexing past outer axes ---

    def test_tile_shape_survives_block_index(self):
        """Indexing into a block doesn't change the loaded block's tile_shape."""
        v = blocks(MockTensor(1024, 2048), tile_size=(128, 512), block_size=(2, 2))
        # block_size=(2, 2) -> each block is (2*128, 2*512) = (256, 1024) elements
        # -> tile_shape inside one block is (2, 2)
        inner = v[0, 0]
        assert inner.tile_shape == (2, 2)
        assert inner.tile_size == (128, 512)

    def test_tile_shape_for_single_tile_view(self):
        """After full tile-grid indexing, tile_shape collapses to (1, 1)."""
        v = tiles(MockTensor(512, 2048), tile_size=(128, 512))
        single = v[0, 0]
        assert single.tile_shape == (1, 1)
