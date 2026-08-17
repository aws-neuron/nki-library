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

"""Validator + PSUMLayout coverage for the Grid-shaped ``nt.psum_pool`` API.

Exercises the validator decision matrix (``bank_axis`` / ``bank_ids``
combinations), the slot-stride rule
(``slot_stride = max(512, 4 * tile_f)`` for tiles sharing a bank), and
the PSUMLayout cursor / tile-strides invariants. ``psum_pool()`` itself
needs an NKI backend so most tests drive the validator + factory
helpers directly.
"""

import pytest
from nkilib_src.nkilib.experimental.neurotile.core._validation import _validate_psum_pool_grid_args
from nkilib_src.nkilib.experimental.neurotile.core.layout_psum import PSUMLayout

from test.unit.nkilib.experimental.neurotile._mocks import MockTensor
from test.utils.pytest_test_metadata import pytest_marks

# ============================================================================
# Validator: allocation modes (bank_axis + bank_ids matrix)
# ============================================================================


@pytest_marks(["neurotile"])
class TestValidatorAllocationModes:
    """Validator decision matrix:

    +-------------+-------------+--------------------------------+
    | bank_axis   | bank_ids    | accepted                        |
    +=============+=============+================================+
    | None        | None        | yes (compiler-managed)          |
    | None        | provided    | yes (every tile own bank)       |
    | int         | provided    | yes (slot-packed within bank)   |
    | int         | None        | rejected                        |
    +-------------+-------------+--------------------------------+
    """

    @pytest.mark.fast
    def test_compiler_managed_no_bank_axis_no_bank_ids(self):
        tile_grid_shape, num_banks, slots_per_bank, tile_f = _validate_psum_pool_grid_args(
            tile_size=(128, 128),
            grid=(2, 4),
            element_shape=None,
            bank_axis=None,
            bank_ids=None,
        )
        # Compiler-managed: tile_count is product(grid), slots_per_bank == 1.
        assert tile_grid_shape == (2, 4)
        assert num_banks == 8  # 2 * 4 grid tiles
        assert slots_per_bank == 1
        assert tile_f == 128

    def test_all_fanout_bank_axis_none_with_bank_ids(self):
        """bank_axis=None + bank_ids -> every tile its own bank."""
        tile_grid_shape, num_banks, slots_per_bank, _ = _validate_psum_pool_grid_args(
            tile_size=(128, 256),
            grid=(2, 3),
            element_shape=None,
            bank_axis=None,
            bank_ids=(0, 1, 2, 3, 4, 5),
        )
        assert num_banks == 6
        assert slots_per_bank == 1

    def test_slot_packed_bank_axis_int_with_bank_ids(self):
        """bank_axis=int -> fan one dim, pack the other as slots."""
        tile_grid_shape, num_banks, slots_per_bank, _ = _validate_psum_pool_grid_args(
            tile_size=(128, 128),
            grid=(2, 4),
            element_shape=None,
            bank_axis=1,
            bank_ids=(0, 1, 2, 3),
        )
        assert num_banks == 4
        assert slots_per_bank == 2

    def test_bank_axis_int_without_bank_ids_rejected(self):
        with pytest.raises(AssertionError, match="bank_axis= without bank_ids="):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(2, 4),
                element_shape=None,
                bank_axis=0,
                bank_ids=None,
            )


# ============================================================================
# Validator: bank capacity + slot-stride rule
# ============================================================================


@pytest_marks(["neurotile"])
class TestValidatorBankCapacity:
    """Slot stride rule (verified via raw-NKI probes):

    ``slot_stride = max(512, 4 * tile_f)``;
    ``slots_per_bank * slot_stride <= 2048``.
    """

    @pytest.mark.fast
    def test_slot_stride_512_for_small_tile_f(self):
        # tile_f=128 -> stride=512 -> up to 4 slots fit per bank.
        _, _, slots_per_bank, tile_f = _validate_psum_pool_grid_args(
            tile_size=(128, 128),
            grid=(2, 4),
            element_shape=None,
            bank_axis=1,
            bank_ids=(0, 1, 2, 3),
        )
        assert slots_per_bank == 2 and tile_f == 128
        # 2 slots * stride 512 = 1024, fits.

    def test_slot_stride_4x_for_medium_tile_f(self):
        # tile_f=256 -> stride=1024 -> 2 slots fit per bank.
        # grid=(2, 2), bank_axis=1 -> num_banks=2, slots_per_bank=2.
        _, _, slots_per_bank, tile_f = _validate_psum_pool_grid_args(
            tile_size=(128, 256),
            grid=(2, 2),
            element_shape=None,
            bank_axis=1,
            bank_ids=(0, 1),
        )
        assert slots_per_bank == 2 and tile_f == 256

    def test_slot_packing_rejects_tile_f_512_with_two_slots(self):
        """tile_f=512 needs full bank -> slots_per_bank=1, no packing."""
        # bank_axis=1: num_banks=4, slots_per_bank=2 (BXS_SUB).
        # 2 slots * stride 2048 = 4096 > 2048 bank size -> reject.
        with pytest.raises(AssertionError, match="exceeding the 2048-element"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 512),
                grid=(2, 4),
                element_shape=None,
                bank_axis=1,
                bank_ids=(0, 1, 2, 3),
            )

    def test_tile_f_512_with_all_fanout_passes(self):
        """Same grid as above, but bank_axis=None -> 1 tile per bank."""
        _, num_banks, slots_per_bank, _ = _validate_psum_pool_grid_args(
            tile_size=(128, 512),
            grid=(2, 4),
            element_shape=None,
            bank_axis=None,
            bank_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        )
        assert num_banks == 8 and slots_per_bank == 1


# ============================================================================
# Validator: bank count + bank_ids length checks
# ============================================================================


@pytest_marks(["neurotile"])
class TestValidatorBankCount:
    @pytest.mark.fast
    def test_more_grid_tiles_than_hw_banks_rejected_in_fanout(self):
        # 9 tiles; pass 9 bank_ids cycling within [0, 8) so the
        # range check passes before the tile-count check fires.
        with pytest.raises(AssertionError, match="grid tiles exceed"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(3, 3),
                element_shape=None,
                bank_axis=None,
                bank_ids=(0, 1, 2, 3, 4, 5, 6, 7, 0),
            )

    def test_bank_axis_int_count_exceeds_hw_banks_rejected(self):
        with pytest.raises(AssertionError, match="exceeds the 8 available"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(9, 1),
                element_shape=None,
                bank_axis=0,
                bank_ids=(0, 1, 2, 3, 4, 5, 6, 7, 0),
            )

    def test_bank_ids_length_must_match_bank_axis(self):
        with pytest.raises(AssertionError, match="must match"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(2, 4),
                element_shape=None,
                bank_axis=1,
                bank_ids=(0, 1, 2),  # length 3, but grid[1] = 4
            )

    def test_bank_ids_length_must_match_grid_tile_count_in_fanout(self):
        with pytest.raises(AssertionError, match="one bank per grid tile"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(2, 4),
                element_shape=None,
                bank_axis=None,
                bank_ids=(0, 1, 2, 3),  # length 4, but product(grid) = 8
            )

    def test_bank_id_out_of_range_rejected(self):
        with pytest.raises(AssertionError, match="is out of range"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(2, 1),
                element_shape=None,
                bank_axis=0,
                bank_ids=(0, 8),  # 8 is out of range [0, 8)
            )

    def test_duplicate_bank_ids_rejected_in_fanout_mode(self):
        """In all-fanout mode every tile must have its own bank."""
        with pytest.raises(AssertionError, match="unique bank_ids"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(2, 2),
                element_shape=None,
                bank_axis=None,
                bank_ids=(0, 1, 1, 2),  # bank_id 1 used twice
            )


# ============================================================================
# Validator: grid + element_shape combinations
# ============================================================================


@pytest_marks(["neurotile"])
class TestValidatorGridShape:
    @pytest.mark.fast
    def test_grid_alone_derives_tile_grid_shape(self):
        tile_grid_shape, *_ = _validate_psum_pool_grid_args(
            tile_size=(128, 128),
            grid=(2, 3),
            element_shape=None,
            bank_axis=None,
            bank_ids=None,
        )
        assert tile_grid_shape == (2, 3)

    def test_element_shape_alone_ceil_divides(self):
        # element_shape=(256, 384), tile=(128, 128) -> grid (2, 3).
        tile_grid_shape, *_ = _validate_psum_pool_grid_args(
            tile_size=(128, 128),
            grid=None,
            element_shape=(256, 384),
            bank_axis=None,
            bank_ids=None,
        )
        assert tile_grid_shape == (2, 3)

    def test_partial_element_shape_rounds_up(self):
        # 1792 / 512 -> ceil 4
        tile_grid_shape, *_ = _validate_psum_pool_grid_args(
            tile_size=(128, 512),
            grid=None,
            element_shape=(128, 1792),
            bank_axis=None,
            bank_ids=None,
        )
        assert tile_grid_shape == (1, 4)

    def test_grid_and_element_shape_mutually_exclusive(self):
        """Match alloc_tiles/alloc_blocks -- both args together is ambiguous."""
        with pytest.raises(AssertionError, match="pass either grid= or element_shape=, not both"):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=(2, 4),
                element_shape=(256, 320),
                bank_axis=None,
                bank_ids=None,
            )

    def test_neither_grid_nor_element_shape_rejected(self):
        with pytest.raises(AssertionError, match="pass either grid= .* or element_shape="):
            _validate_psum_pool_grid_args(
                tile_size=(128, 128),
                grid=None,
                element_shape=None,
                bank_axis=None,
                bank_ids=None,
            )


# ============================================================================
# PSUMLayout: tile_strides flatten grid coords to flat-tile index
# ============================================================================


@pytest_marks(["neurotile"])
class TestTileStrides:
    """``tile_strides[d] * k`` is the flat-tile-index advance per item on dim d.

    For the three allocation modes, the tile layout differs:

    - ``bank_axis=None``: row-major over all dims.
    - ``bank_axis=int``: bank-major, slot dims mixed-radix in original order.
    """

    @pytest.mark.fast
    def test_compiler_managed_or_fanout_is_row_major(self):
        # bank_axis=None -> row-major over (2, 4):
        # (s, i) -> s*4 + i  =>  strides = (4, 1)
        strides = PSUMLayout._tile_strides_for(grid_shape=(2, 4), bank_axis=None, slots_per_bank=1)
        assert strides == (4, 1)

    def test_bank_axis_0_packs_slots_after_bank(self):
        # grid=(2, 4), bank_axis=0, slots_per_bank=4:
        # bank=s, slot=i, flat = s*slots_per_bank + i = s*4 + i
        # strides = (slots_per_bank=4, 1)
        strides = PSUMLayout._tile_strides_for(grid_shape=(2, 4), bank_axis=0, slots_per_bank=4)
        assert strides == (4, 1)

    def test_bank_axis_1_packs_other_dim_as_slots(self):
        # grid=(2, 4), bank_axis=1, slots_per_bank=2:
        # bank=i, slot=s, flat = i*slots_per_bank + s = i*2 + s
        # strides[1] = slots_per_bank = 2; strides[0] = 1.
        strides = PSUMLayout._tile_strides_for(grid_shape=(2, 4), bank_axis=1, slots_per_bank=2)
        assert strides == (1, 2)

    def test_3d_grid_row_major_in_fanout_mode(self):
        # grid=(2, 3, 4), bank_axis=None: row-major.
        # (s, i, j) -> s*12 + i*4 + j  =>  strides = (12, 4, 1)
        strides = PSUMLayout._tile_strides_for(grid_shape=(2, 3, 4), bank_axis=None, slots_per_bank=1)
        assert strides == (12, 4, 1)


# ============================================================================
# PSUMLayout: cursor advance maps grid moves to flat tile offset
# ============================================================================


def _make_layout(grid_shape, bank_axis, slots_per_bank, num_tiles, alloc_tile_size=(128, 128)):
    """Build a PSUMLayout with stub tile ndarrays for cursor logic tests."""
    # Use tuple of None placeholders -- advance only manipulates offset.
    tile_arrays = tuple([None] * num_tiles)
    return PSUMLayout(
        tile_arrays=tile_arrays,
        offset=0,
        alloc_tile_size=alloc_tile_size,
        bank_axis=bank_axis,
        slots_per_bank=slots_per_bank,
        dtype=None,
        grid_shape=grid_shape,
    )


@pytest_marks(["neurotile"])
class TestPSUMLayoutAdvance:
    @pytest.mark.fast
    def test_advance_on_bank_axis_walks_banks(self):
        # grid=(2, 4), bank_axis=0, slots_per_bank=4:
        # advance(0, 1) -> flat = 1 * 4 = 4 (bank 1, slot 0)
        layout = _make_layout(grid_shape=(2, 4), bank_axis=0, slots_per_bank=4, num_tiles=8)
        advanced = layout.advance(dim=0, k=1, step=1)
        assert advanced.offset == 4
        assert advanced.bank_idx == 1
        assert advanced.slot_idx == 0

    def test_advance_on_slot_axis_walks_slots(self):
        # bank_axis=0, advance dim 1 -> slot+=1
        layout = _make_layout(grid_shape=(2, 4), bank_axis=0, slots_per_bank=4, num_tiles=8)
        advanced = layout.advance(dim=1, k=2, step=1)
        assert advanced.offset == 2
        assert advanced.bank_idx == 0
        assert advanced.slot_idx == 2

    def test_advance_in_fanout_mode_walks_banks(self):
        # bank_axis=None, slots_per_bank=1: every advance is a bank step.
        # tile_strides = (4, 1) for grid=(2, 4) row-major.
        layout = _make_layout(grid_shape=(2, 4), bank_axis=None, slots_per_bank=1, num_tiles=8)
        advanced = layout.advance(dim=0, k=1, step=1)
        assert advanced.offset == 4
        # In fanout mode each tile IS a bank.
        assert advanced.bank_idx == 4
        assert advanced.slot_idx == 0

    def test_runtime_k_routes_to_set_indirect(self):
        """Non-int k should not crash with int+object arithmetic."""
        layout = _make_layout(grid_shape=(2, 4), bank_axis=0, slots_per_bank=4, num_tiles=8)

        class FakeCExpr:
            pass

        runtime_k = FakeCExpr()
        advanced = layout.advance(dim=0, k=runtime_k, step=1)
        # Routes to set_indirect: indirect populated, offset unchanged.
        assert advanced.indirect is not None
        assert advanced.offset == 0


# ============================================================================
# PSUMLayout.get_data: per-tile .data auto-clamps to grid.element_shape
# ============================================================================


def _make_full_tile_layout(grid_shape, alloc_tile_size, bank_axis=None, slots_per_bank=None):
    """Build a PSUMLayout with tile ndarrays sized to ``alloc_tile_size``."""
    tile_count = 1
    for d in range(len(grid_shape)):
        tile_count = tile_count * grid_shape[d]
    spb = slots_per_bank if slots_per_bank is not None else 1
    tiles = tuple(MockTensor(alloc_tile_size) for _ in range(tile_count))
    return PSUMLayout(
        tile_arrays=tiles,
        offset=0,
        alloc_tile_size=alloc_tile_size,
        bank_axis=bank_axis,
        slots_per_bank=spb,
        dtype=None,
        grid_shape=grid_shape,
    )


@pytest_marks(["neurotile"])
class TestPSUMLayoutGetData:
    """``get_data`` returns a per-tile ndarray clamped to ``grid.element_shape``.

    This is what lets kernels write ``psums[s, i].data`` instead of the
    manual ``.data[:, :actual_f]`` slice on partial trailing tiles.
    """

    @pytest.mark.fast
    def test_full_tile_returns_alloc_sized_ndarray(self):
        """Non-trailing tiles: get_data == tile_data == full alloc."""
        # grid (2, 4), alloc tile (128, 128), element_shape full.
        layout = _make_full_tile_layout(
            grid_shape=(2, 4),
            alloc_tile_size=(128, 128),
        )
        # Build a Grid that reports remaining = alloc_tile_size on each dim
        # (single-tile view, no clamp).
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        grid = Grid.from_shape(
            element_shape=(128, 128),
            tile_size=(128, 128),
        )
        data = layout.get_data(grid)
        assert data.shape == (128, 128)

    def test_partial_trailing_tile_clamps_data_f(self):
        """Trailing tile: alloc 512 F, addressable 256 F -> data is (P, 256)."""
        layout = _make_full_tile_layout(
            grid_shape=(1, 1),
            alloc_tile_size=(128, 512),
        )
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        # Single-tile view, leaf clamped to 256 (partial trailing).
        grid = Grid.from_shape(
            element_shape=(128, 256),
            tile_size=(128, 512),
        )
        # remaining is min(walked, element_shape) = (128, 256).
        assert grid.remaining == (128, 256)
        data = layout.get_data(grid)
        assert data.shape == (128, 256)

    def test_addressable_equal_to_alloc_returns_full_tile(self):
        """``addressable_f >= alloc_tile_f`` short-circuits to tile_data."""
        layout = _make_full_tile_layout(
            grid_shape=(1, 1),
            alloc_tile_size=(128, 256),
        )
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        grid = Grid.from_shape(
            element_shape=(128, 256),
            tile_size=(128, 256),
        )
        data = layout.get_data(grid)
        # No slice; the original tile ndarray is returned.
        assert data is layout.tile_arrays[0]
        assert data.shape == (128, 256)

    def test_grid_none_returns_full_tile(self):
        """No grid context -> no clamp possible, return the alloc tile."""
        layout = _make_full_tile_layout(
            grid_shape=(1, 1),
            alloc_tile_size=(128, 512),
        )
        data = layout.get_data(grid=None)
        assert data is layout.tile_arrays[0]


@pytest_marks(["neurotile"])
class TestPSUMLayoutApClamp:
    """``PSUMLayout.ap`` clamps the active tile's AP to ``grid.remaining`` -- the
    same partial-trailing-tile width ``get_data`` reports (consistent with the
    SBUF/HBM layouts). A partial PSUM tile that reported the full alloc width
    here broke ``nc_matmul`` operand matching (dst free product != moving free
    product)."""

    @pytest.mark.fast
    def test_full_tile_ap_is_alloc_width(self):
        layout = _make_full_tile_layout(grid_shape=(1, 1), alloc_tile_size=(128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        grid = Grid.from_shape(element_shape=(128, 512), tile_size=(128, 512))
        assert layout.ap(grid).shape == (128, 512)

    @pytest.mark.fast
    def test_partial_trailing_tile_ap_clamps_to_remaining(self):
        """Trailing tile: alloc 512 F, addressable 256 F -> ap is (P, 256),
        matching get_data (regression guard: ap used to report the full 512)."""
        layout = _make_full_tile_layout(grid_shape=(1, 1), alloc_tile_size=(128, 512))
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        grid = Grid.from_shape(element_shape=(128, 256), tile_size=(128, 512))
        assert grid.remaining == (128, 256)
        assert layout.ap(grid).shape == (128, 256)
        # ap and get_data agree on the partial tile width.
        assert layout.ap(grid).shape == layout.get_data(grid).shape

    @pytest.mark.fast
    def test_subwidth_slice_pins_physical_partition_stride(self):
        """A tile backed by a WIDER bank (slice of a 512-wide bank viewed as 144)
        must emit AP level-0 stride = the physical row width 512, not the
        contiguous-of-logical 144. Regression guard for the on-chip
        partition-stride check ("Partition step 144 must equal 512")."""
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        # Tile mock whose logical shape is 144 but physical storage is 512 wide
        # (what `bank.data[:, :144]` reports: .shape=(128,144), _storage_shape=(128,512)).
        tile = MockTensor((128, 144), storage_shape=(128, 512))
        layout = PSUMLayout(
            tile_arrays=(tile,),
            offset=0,
            alloc_tile_size=(128, 144),
            bank_axis=None,
            slots_per_bank=1,
            dtype=None,
            grid_shape=(1, 1),
        )
        grid = Grid.from_shape(element_shape=(128, 144), tile_size=(128, 144))
        ap = layout.ap(grid)
        assert ap.shape == (128, 144)
        assert ap.strides[0] == 512  # physical row width, not the logical 144

        # Reshape path: a transform sets ap_strides to the contiguous-of-logical
        # values (here level-0 would be 144). ap() must still pin level-0 to the
        # physical 512 and keep the transform's inner strides.
        reshaped_layout = layout.apply_transform((144, 12, 1))
        reshaped = reshaped_layout.ap(Grid.from_shape(element_shape=(128, 12, 12), tile_size=(128, 12, 12)))
        assert reshaped.strides[0] == 512  # physical row width, NOT the transform's 144
        assert reshaped.strides[1:] == (12, 1)  # inner levels keep the transform strides

    @pytest.mark.fast
    def test_narrow_free_advances_offset_not_tile_index(self):
        """A free-dim slice narrows WITHIN the active tile: ``narrow_free``
        advances ``free_offset`` (via the element strides) and ``ap()`` emits it
        as the AP offset, while the tile index (``offset``) is untouched. The
        W-padded conv3d chain: reshape (4, 12) then slice dh and W."""
        from nkilib_src.nkilib.experimental.neurotile.core.grid import Grid

        tile = MockTensor((128, 48), storage_shape=(128, 512))
        layout = PSUMLayout(
            tile_arrays=(tile,),
            offset=0,
            alloc_tile_size=(128, 48),
            bank_axis=None,
            slots_per_bank=1,
            dtype=None,
            grid_shape=(1, 1),
        )
        # reshape flat 48 -> (4 dh, 12 w): element strides become (48, 12, 1).
        reshaped = layout.apply_transform((48, 12, 1))
        # slice dh[1:3] (dim 1, start 1) then w[3:10] (dim 2, start 3) on a
        # (128, 4, 12) element shape.
        dh_sliced = reshaped.narrow_free(1, 1, (128, 4, 12))
        wh_sliced = dh_sliced.narrow_free(2, 3, (128, 4, 12))

        # free_offset = 1*12 (dh) + 3*1 (w) = 15; tile index stays 0.
        assert wh_sliced.free_offset == 15
        assert wh_sliced.offset == 0  # tile selection untouched
        assert wh_sliced.tile_arrays is layout.tile_arrays

        # ap() emits the strided sub-region at that offset, level-0 still 512.
        ap = wh_sliced.ap(Grid.from_shape(element_shape=(128, 2, 7), tile_size=(128, 2, 7)))
        assert ap.offset == 15
        assert ap.strides[0] == 512  # physical row width
        assert ap.strides[1:] == (12, 1)  # dh stride 12, w stride 1
        assert ap.shape == (128, 2, 7)


# ============================================================================
# psum_pool factory: element_shape narrows the trailing tile .data extent
# ============================================================================


@pytest_marks(["neurotile"])
class TestPsumPoolElementShape:
    """``element_shape`` is the partial-aware path: factory ceiling-divides
    by ``tile_size`` to derive the grid and flags a partial trailing tile
    when ``element_shape[d]`` is not a multiple of ``tile_size[d]``.
    """

    @pytest.mark.fast
    def test_element_shape_alone_derives_partial_trailing_grid(self):
        # 1280 = 2*512 + 256 -> ceil(1280/512) = 3 tiles on dim 1.
        tile_grid_shape, num_banks, slots_per_bank, tile_f = _validate_psum_pool_grid_args(
            tile_size=(128, 512),
            grid=None,
            element_shape=(128, 1280),
            bank_axis=None,
            bank_ids=(0, 1, 2),
        )
        assert tile_grid_shape == (1, 3)
        assert num_banks == 3
        assert slots_per_bank == 1
        assert tile_f == 512
