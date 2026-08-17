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
import nki.language as nl

from ._helpers import contiguous_strides, physical_row_width, product, reachable_dim_extent, remove_at
from .axis import IndirectKind, IndirectOffset


class PSUMLayout(nl.NKIObject):
    """Physical addressing for a PSUM-backed view.

    One ``nl.ndarray(buffer=psum)`` per grid tile, ordered in a flat tuple.
    The cursor is a flat tile offset; ``tile_strides`` maps N-D grid coords
    to the flat index (``offset = sum(grid_idx[d] * tile_strides[d])``).
    Mirrors :class:`SBUFLayout`'s public surface.

    Two independent address axes:
      - ``offset`` selects which tile (``tile_arrays[offset]``); tile / bank
        navigation (``psums[i, j]``) moves it.
      - ``free_offset`` addresses within the active tile's partition row; an
        in-tile free-dim ``.slice()`` (after ``reshape_dim``) advances it.
    Keeping them separate is what lets a strided W-padded sub-region be
    expressed without disturbing tile selection.

    Attributes:
        tile_arrays (tuple): One ``nl.ndarray`` per grid tile.
        offset (int): Flat tile index of the active tile.
        free_offset (int): Element offset along the free axis within the
            active tile (0 unless an in-tile free-dim slice advanced it).
        alloc_tile_size (tuple[int, int]): Per-tile (P, F) span.
        bank_axis (int | None): Which Grid dim spreads tiles across PSUM
            banks. ``None`` means "every tile on its own bank" or
            "compiler-managed" (validator distinguishes).
        slots_per_bank (int): Tiles per bank (1 for bank_axis=None).
        tile_strides (tuple[int, ...]): Per-dim strides into the flat
            tile array.
        dtype: Element dtype.
        buffer_type: ``nl.psum``.
        ap_strides (tuple | None): Element-level transform strides.
        indirect (IndirectOffset | None): API-parity slot for runtime
            offsets; PSUM does not currently use it.
        root_source: Provided for API parity with SBUFLayout.
    """

    def __init__(
        self,
        tile_arrays,
        offset,
        alloc_tile_size,
        bank_axis,
        slots_per_bank,
        dtype,
        tile_strides=None,
        grid_shape=None,
        buffer_type=None,
        indirect=None,
        root_source=None,
        transform_strides=None,
        free_offset=0,
    ):
        self.tile_arrays = tuple(tile_arrays)
        self.alloc_tile_size = tuple(alloc_tile_size)
        self.bank_axis = bank_axis
        self.slots_per_bank = slots_per_bank
        self.dtype = dtype
        self.buffer_type = buffer_type if buffer_type is not None else nl.psum
        self.indirect = indirect
        self.ap_strides = transform_strides

        self.offset = offset
        self.free_offset = free_offset
        if tile_strides is not None:
            self.tile_strides = tuple(tile_strides)
        else:
            assert grid_shape is not None, (
                "PSUMLayout: either tile_strides= or grid_shape= must be "
                "provided so the cursor can map grid coords to flat tile index."
            )
            self.tile_strides = PSUMLayout._tile_strides_for(
                tuple(grid_shape),
                bank_axis,
                slots_per_bank,
            )
        spb = max(slots_per_bank, 1)
        self.bank_idx = offset // spb
        self.slot_idx = offset % spb
        self.num_banks = (len(self.tile_arrays) + spb - 1) // spb

        self.source = self.tile_arrays[self.offset] if len(self.tile_arrays) > 0 else None
        self.root_source = root_source if root_source is not None else self.source

        ts_strides = []
        for _d in range(len(self.alloc_tile_size)):
            ts_strides.append(1)
        self.strides = tuple(ts_strides)

    @staticmethod
    def _tile_strides_for(grid_shape, bank_axis, slots_per_bank):
        """Per-dim strides into the flat tile array.

        Tiles are stored in alloc order:

        - ``bank_axis is None``: row-major over all dims. Flat tile
          index is the standard mixed-radix sum
          ``sum(grid_idx[d] * tile_strides[d])``.
        - ``bank_axis: int``: bank-major. ``bank_idx = grid_idx[bank_axis]``;
          ``slot_idx`` is mixed-radix over the non-bank dims (in
          original order). Flat tile index = ``bank_idx * slots_per_bank
          + slot_idx``.
        """
        ndim = len(grid_shape)
        strides = [0] * ndim
        if bank_axis is None:
            # Row-major over all dims.
            s = 1
            for d in range(ndim - 1, -1, -1):
                strides[d] = s
                s = s * grid_shape[d]
            return tuple(strides)
        # Slot dims first (mixed-radix in original order), then bank dim.
        s = 1
        for d in range(ndim - 1, -1, -1):
            if d == bank_axis:
                continue
            strides[d] = s
            s = s * grid_shape[d]
        # Bank dim stride = product of slot dim sizes = slots_per_bank.
        strides[bank_axis] = slots_per_bank
        return tuple(strides)

    # ================================================================
    # Tile data
    # ================================================================

    def _alloc_tile_f(self):
        """F-extent of one allocation tile."""
        return product(self.alloc_tile_size, start=1)

    def tile_data(self):
        """Return the active tile's ndarray."""
        return self.tile_arrays[self.offset]

    # ================================================================
    # Data access
    # ================================================================

    def get_data(self, grid):
        """Return the active tile's ndarray, F-clamped to ``grid.remaining``.

        The tile ndarray is allocated at the uniform per-tile F extent
        (``alloc_tile_size[1:]``) so tiles always sit at distinct
        accumulator regions; ``grid.remaining`` carries the per-tile
        addressable F width, which is shorter than alloc on a partial
        trailing tile (caller passed ``element_shape`` narrower than
        ``grid * tile_size``). Clamping here lets kernels write
        ``psums[s, i].data`` instead of ``.data[:, :actual_f]``.

        PSUMLayout has no multi-tile ndarray composition -- callers
        iterate per-tile for bulk patterns.
        """
        data = self.tile_data()
        if grid is None or grid.ndim < 2:
            return data
        addressable_f = 1
        for d in range(1, len(self.alloc_tile_size)):
            es = grid.remaining[d] if d < len(grid.remaining) else self.alloc_tile_size[d]
            addressable_f = addressable_f * es
        full_f = self._alloc_tile_f()
        if addressable_f >= full_f:
            return data
        return data[:, 0:addressable_f]

    # ================================================================
    # Strides + transforms
    # ================================================================

    def transform_strides(self, element_shape):
        if self.ap_strides is not None:
            return self.ap_strides
        return contiguous_strides(element_shape)

    def apply_transform(self, new_strides):
        return PSUMLayout(
            tile_arrays=self.tile_arrays,
            offset=self.offset,
            alloc_tile_size=self.alloc_tile_size,
            bank_axis=self.bank_axis,
            slots_per_bank=self.slots_per_bank,
            tile_strides=self.tile_strides,
            dtype=self.dtype,
            buffer_type=self.buffer_type,
            indirect=self.indirect,
            root_source=self.root_source,
            transform_strides=new_strides,
            free_offset=self.free_offset,
        )

    def narrow_free(self, dim, start, element_shape):
        """In-tile free-dim slice: advance the free offset by ``start`` along
        ``dim`` (using the element strides), leaving tile selection untouched.

        ``dim`` is a free axis (>= 1); ``element_shape`` is the pre-slice
        element shape (so the stride for ``dim`` is read from the transform /
        contiguous strides). Returns a new PSUMLayout with the advanced
        ``free_offset``; the caller narrows the Grid counts.
        """
        strides = self.ap_strides if self.ap_strides is not None else contiguous_strides(tuple(element_shape))
        return PSUMLayout(
            tile_arrays=self.tile_arrays,
            offset=self.offset,
            alloc_tile_size=self.alloc_tile_size,
            bank_axis=self.bank_axis,
            slots_per_bank=self.slots_per_bank,
            tile_strides=self.tile_strides,
            dtype=self.dtype,
            buffer_type=self.buffer_type,
            indirect=self.indirect,
            root_source=self.root_source,
            transform_strides=self.ap_strides,
            free_offset=self.free_offset + start * strides[dim],
        )

    # ================================================================
    # Navigation
    # ================================================================

    def advance_step(self, grid, dim):
        return 1

    def advance(self, dim, k, step):
        if not isinstance(k, int):
            return self.set_indirect(IndirectKind.SCALAR, k, dim)
        new_offset = self.offset + k * self.tile_strides[dim]
        return PSUMLayout(
            tile_arrays=self.tile_arrays,
            offset=new_offset,
            alloc_tile_size=self.alloc_tile_size,
            bank_axis=self.bank_axis,
            slots_per_bank=self.slots_per_bank,
            tile_strides=self.tile_strides,
            dtype=self.dtype,
            buffer_type=self.buffer_type,
            indirect=self.indirect,
            root_source=self.root_source,
            transform_strides=self.ap_strides,
            free_offset=self.free_offset,
        )

    def set_indirect(self, kind, value, dim):
        return PSUMLayout(
            tile_arrays=self.tile_arrays,
            offset=self.offset,
            alloc_tile_size=self.alloc_tile_size,
            bank_axis=self.bank_axis,
            slots_per_bank=self.slots_per_bank,
            tile_strides=self.tile_strides,
            dtype=self.dtype,
            buffer_type=self.buffer_type,
            indirect=IndirectOffset(kind=kind, value=value, dim=dim),
            root_source=self.root_source,
            transform_strides=self.ap_strides,
            free_offset=self.free_offset,
        )

    def drop_dim(self, dim):
        new_alloc = remove_at(self.alloc_tile_size, dim)
        new_strides = remove_at(self.tile_strides, dim)
        new_bank_axis = self.bank_axis
        if isinstance(new_bank_axis, int):
            if dim < new_bank_axis:
                new_bank_axis = new_bank_axis - 1
            elif dim == new_bank_axis:
                # Bank dim collapsed; remaining tiles still flat-indexed.
                new_bank_axis = None
        return PSUMLayout(
            tile_arrays=self.tile_arrays,
            offset=self.offset,
            alloc_tile_size=new_alloc,
            bank_axis=new_bank_axis,
            slots_per_bank=self.slots_per_bank,
            tile_strides=new_strides,
            dtype=self.dtype,
            buffer_type=self.buffer_type,
            indirect=self.indirect,
            root_source=self.root_source,
            transform_strides=self.ap_strides,
            free_offset=self.free_offset,
        )

    def drop_dims(self, dims):
        result = self
        for d_idx in range(len(dims) - 1, -1, -1):
            result = result.drop_dim(dims[d_idx])
        return result

    # ================================================================
    # Remainder
    # ================================================================

    def dim_offset_elements(self, dim, element_shape):
        if dim == 0:
            return 0
        ts = self.alloc_tile_size[dim]
        # Decode the dim's grid coordinate from the flat offset using tile_strides.
        # Walking the dim by 1 advances ``tile_strides[dim]`` in flat index, so
        # the dim's coord = (offset // tile_strides[dim]) mod ceil(es/ts).
        stride = self.tile_strides[dim]
        if stride == 0:
            return 0
        dim_count = max(1, (element_shape[dim] + ts - 1) // ts) if element_shape[dim] > 0 else 1
        return ((self.offset // stride) % dim_count) * ts

    def dim_addressable(self, dim, grid):
        """Source elements reachable on `dim` before walking off the source.

        See :func:`reachable_dim_extent` for the shared offset-clamp rule;
        this is the PSUM-layout entry point into it.
        """
        return reachable_dim_extent(grid, self, dim)

    def is_remainder(self, grid):
        if self.indirect is not None:
            return True
        if grid is not None and grid.is_remainder:
            return True
        return False

    # ================================================================
    # AP construction
    # ================================================================

    def _partition_row_stride(self):
        """Physical free width of the active tile (the AP's level-0 stride).

        Read from the underlying ndarray's storage shape, which survives
        ``[:, slice]`` so a sub-width view of a wider bank reports the bank's
        true row width.
        """
        return physical_row_width(self.tile_data())

    def ap(self, grid):
        """AP over the active tile; counts clamped to ``grid.remaining`` for a
        partial trailing tile. Level-0 stride is the physical row width (so a
        sub-width slice of a wider bank threads a valid partition stride, as the
        compiler requires); inner levels use the transform or contiguous strides.
        ``free_offset`` is the in-tile column offset from a free-dim slice (the
        AP starts there, e.g. a W-padded strided sub-region)."""
        data = self.tile_data()
        counts = grid.remaining if grid is not None else tuple(data.shape)
        inner_strides = self.ap_strides if self.ap_strides is not None else contiguous_strides(tuple(counts))
        pattern = [[self._partition_row_stride(), counts[0]]]
        for d in range(1, len(counts)):
            pattern.append([inner_strides[d], counts[d]])
        return data.ap(pattern=pattern, offset=self.free_offset)

    # ================================================================
    # Repr
    # ================================================================

    def __repr__(self):
        return (
            "PSUMLayout(offset="
            + str(self.offset)
            + ", num_banks="
            + str(self.num_banks)
            + ", slots_per_bank="
            + str(self.slots_per_bank)
            + ", alloc_tile_size="
            + str(self.alloc_tile_size)
            + ", bank_axis="
            + str(self.bank_axis)
            + ")"
        )


__all__ = ["PSUMLayout"]
