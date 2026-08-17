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

from ._helpers import (
    ceiling_div,
    contiguous_ap_pattern,
    contiguous_strides,
    p_tile_count,
    physical_row_width,
    product,
    reachable_dim_extent,
    remove_at,
    validate_index_key,
)
from .axis import IndirectKind, IndirectOffset
from .grid import Grid


class SBUFLayout(nl.NKIObject):
    """
    Physical addressing for an SBUF / PSUM-backed view.

    SBUF is the on-chip tile buffer (P=128 partition rows × F free
    columns); the hardware addresses it as one flat F-offset per row.
    Multi-P-tile regions fold their extra P-tiles into the F dimension
    (the rotating-buffer pool laid out flat). This layout encapsulates
    that flattening so the rest of the system sees a logical N-D tile
    grid.

    Pairs with a :class:`Grid` inside an
    :class:`~neurotile.core.ndslice.NDSlice` to back ``.load()`` /
    ``.store()`` results, on-chip indexing, and re-tiling.

    ``alloc_tile_size`` is the allocation-side per-tile span (uniform
    across slots; what the SBUF buffer actually reserves per tile).
    Per-tile *addressable* extent (which can be partial for trailing
    tiles) lives on the Grid via ``element_shape``. The two are equal
    for non-partial views; for partial-trailing-tile views the Grid
    reports the smaller extent so downstream ``[k, i].data`` slices
    correctly, while AP / DMA emission uses the uniform allocation
    stride from ``strides``.

    Immutable: ``advance`` / ``apply_transform`` / ``retile`` /
    ``drop_dim`` / ``sub_index`` all return fresh ``SBUFLayout``
    instances.

    Attributes:
        source: Root SBUF ``nl.ndarray``.
        root_source: Parent SBUF buffer (defaults to ``source``).
        offset (int): F-column offset into ``source`` (flat navigation).
        strides (tuple[int, ...]): Per-dim tile-level F-strides -- one
            tile-grid step on dim ``d`` advances the F-offset by
            ``strides[d]``. Allocation-uniform.
        alloc_tile_size (tuple[int, ...]): Per-dim element extent the
            SBUF buffer reserves per tile. Stays uniform under partial
            trailing tiles. The Grid's per-tile *addressable* extent
            can be smaller (Grid.element_shape clamps for partials).
        dtype: Element dtype.
        buffer_type: ``nl.sbuf`` or ``nl.psum``.
        ap_strides (tuple[int, ...] | None): Element-level AP strides
            for transformed views (broadcast / non-contiguous).
            ``None`` when the view is plain row-major.
        indirect (IndirectOffset | None): Carried for API parity with
            HBMLayout; rarely used on SBUF (sub-tile indexing uses
            ``sub_index`` instead of an indirect offset).
    """

    def __init__(
        self,
        source,
        offset,
        strides,
        alloc_tile_size,
        dtype,
        buffer_type=None,
        transform_strides=None,
        root_source=None,
        indirect=None,
    ):
        self.source = source
        self.root_source = root_source if root_source is not None else source
        self.offset = offset
        self.strides = strides
        self.alloc_tile_size = alloc_tile_size
        self.dtype = dtype
        self.buffer_type = buffer_type
        self.ap_strides = transform_strides
        self.indirect = indirect

    # ================================================================
    # Static factories
    # ================================================================

    @staticmethod
    def build_view(
        sbuf_source,
        element_shape,
        alloc_tile_size,
        dtype,
        buffer_type,
        offset=0,
        block_size=None,
    ):
        """Create (Grid, SBUFLayout) for an SBUF tile grid.

        Single construction path for all SBUF views: load(), tiles(sbuf), re-tile.
        ``element_shape`` is the addressable per-dim extent (caller clamps
        for partial trailing tiles); ``alloc_tile_size`` is the allocation
        per-tile extent (uniform). Strides are computed from
        ``alloc_tile_size`` so AP emission walks the uniform allocation
        stride; Grid carries ``element_shape`` for partial-aware view
        typing.
        """
        element_shape = tuple(element_shape)
        alloc_tile_size = tuple(alloc_tile_size)
        # Grid uses alloc_tile_size for tile-grid layout; partial-F is reflected
        # in element_shape (the trailing tile's leaf axis count gets clamped
        # via Grid's axes-from-shape derivation).
        grid = Grid.from_shape(
            element_shape=element_shape,
            tile_size=alloc_tile_size,
            block_size=block_size,
        )
        # Strides at allocation extent: keeps tile-grid stride uniform across
        # slots (matters for the rotating-buffer contract).
        stride_basis = []
        for d in range(len(alloc_tile_size)):
            stride_basis.append(max(element_shape[d], alloc_tile_size[d]))
        strides = SBUFLayout._tile_grid_strides(alloc_tile_size, tuple(stride_basis))
        layout = SBUFLayout(
            sbuf_source,
            offset,
            strides,
            alloc_tile_size,
            dtype,
            buffer_type,
        )
        return grid, layout

    @staticmethod
    def contiguous_ap(sbuf):
        """Build contiguous AP for an SBUF ndarray."""
        return sbuf.ap(pattern=contiguous_ap_pattern(tuple(sbuf.shape)), offset=0)

    @staticmethod
    def f_extent(remaining, tile_p):
        """Total F-columns in SBUF for a loaded region."""
        total_f = product(remaining, start=1)
        p_tiles = p_tile_count(remaining[0], tile_p)
        return total_f * p_tiles

    @staticmethod
    def tile_shape_from_buffer(sbuf_shape, tile_size):
        """Derive tile_shape from SBUF buffer shape and tile_size."""
        if len(sbuf_shape) == len(tile_size):
            tile_shape = []
            for d in range(len(tile_size)):
                if tile_size[d] > 0:
                    tile_shape.append((sbuf_shape[d] + tile_size[d] - 1) // tile_size[d])
                else:
                    tile_shape.append(1)
            return tuple(tile_shape)

        tile_p = tile_size[0]
        tile_f = product(tile_size, start=1)
        sbuf_f = product(sbuf_shape, start=1)
        p_tiles = (sbuf_shape[0] + tile_p - 1) // tile_p if tile_p > 0 else 1
        f_tiles = (sbuf_f + tile_f - 1) // tile_f if tile_f > 0 else 1

        tile_shape = []
        for d in range(len(tile_size)):
            if d == 0:
                tile_shape.append(p_tiles)
            elif d == len(tile_size) - 1:
                tile_shape.append(f_tiles)
            else:
                tile_shape.append(1)
        return tuple(tile_shape)

    # ================================================================
    # Data access
    # ================================================================

    def _alloc_tile_f(self):
        """F-extent of one allocation tile (uniform; for buffer addressing)."""
        return product(self.alloc_tile_size, start=1)

    def tile_data(self):
        """Return SBUF ndarray for one allocation tile as 2D (P, alloc_tile_f)."""
        src = self._as_2d(self.source)
        alloc_tile_f = self._alloc_tile_f()
        if self.offset == 0 and alloc_tile_f >= src.shape[-1]:
            return src
        return src[:, nl.ds(self.offset, alloc_tile_f)]

    def get_data(self, grid):
        """Return raw data for ISA ops, or None for invalid multi-tile access.

        Single-tile: tile data at current offset, clamped to actual remaining.
        Multi-tile, contiguous: region spanning all walked F-columns.
        Multi-tile, sharded: returns None (gap bytes; index a tile first).
        """
        alloc_tile_f = self._alloc_tile_f()
        if not self._is_multi_tile(grid):
            actual_f = product(grid.remaining, start=1)
            src = self._as_2d(self.source)
            physical_f = min(alloc_tile_f, src.shape[-1] - self.offset)
            clamp_f = min(actual_f, physical_f)
            if clamp_f < alloc_tile_f:
                return src[:, nl.ds(self.offset, clamp_f)]
            return self.tile_data()
        if grid is not None and grid.has_any_gapped_axis():
            return None
        total_f = self._walked_f_extent(grid)
        src = self._as_2d(self.source)
        if self.offset == 0 and total_f >= src.shape[-1]:
            return src
        return src[:, self.offset : self.offset + total_f]

    # ================================================================
    # Strides and transforms
    # ================================================================

    def transform_strides(self, element_shape):
        """Strides for transform computation."""
        if self.ap_strides is not None:
            return self.ap_strides
        return contiguous_strides(element_shape)

    def apply_transform(self, new_strides):
        """Return layout with AP strides for broadcast/non-contiguous patterns."""
        return SBUFLayout(
            self.source,
            self.offset,
            self.strides,
            self.alloc_tile_size,
            self.dtype,
            self.buffer_type,
            transform_strides=new_strides,
            root_source=self.root_source,
            indirect=self.indirect,
        )

    def retile(self, new_tile_size, new_remaining):
        """Return a new SBUFLayout for the same memory with a different tile grid."""
        new_tile_size = tuple(new_tile_size)
        strides = SBUFLayout._tile_grid_strides(new_tile_size, tuple(new_remaining))
        return SBUFLayout(
            self.source,
            self.offset,
            strides,
            new_tile_size,
            self.dtype,
            self.buffer_type,
            root_source=self.root_source,
        )

    # ================================================================
    # AP construction
    # ================================================================

    def ap(self, grid):
        """Build SBUF AP. Dispatches to transform, standard, or contiguous path."""
        if self.ap_strides is not None:
            return self._ap_with_transform(grid)
        if grid is not None and grid.tile_size is not None:
            return self._ap_standard(grid)
        return self._ap_contiguous(grid)

    # ================================================================
    # Navigation
    # ================================================================

    def advance_step(self, grid, dim):
        """Per-item advance step on `dim` in this layout's native unit.

        SBUF strides are tile-slot granularity, so the native step is the
        Grid's per-item element step divided by the allocation tile size
        on that dim.
        """
        elem_step = grid.current_step(dim)
        tile_d = self.alloc_tile_size[dim]
        if tile_d <= 0:
            return elem_step
        return elem_step // tile_d

    def advance(self, dim, k, step):
        """Advance by ``k`` items along ``dim``, each item ``step`` tile-slots wide.

        ``step`` is in tile-slot units (the layout's native stride unit).
        Callers obtain it from ``advance_step(grid, dim)`` so they don't
        need to know the SBUF flat-F mapping. Runtime ``k`` (CExpr /
        LoopVar) routes to ``set_indirect`` -- the parser would otherwise
        choke on ``int + object`` when statically tracing the runtime
        branch of caller dispatch (the compile-time branch never executes
        with runtime ``k`` at runtime, but the parser checks both).
        """
        if not isinstance(k, int):
            return self.set_indirect(IndirectKind.SCALAR, k, dim)
        return SBUFLayout(
            self.source,
            self.offset + k * step * self.strides[dim],
            self.strides,
            self.alloc_tile_size,
            self.dtype,
            self.buffer_type,
            transform_strides=self.ap_strides,
            root_source=self.root_source,
            indirect=self.indirect,
        )

    def set_indirect(self, kind, value, dim):
        """SBUF doesn't currently use indirect offsets; provided for API parity."""
        return SBUFLayout(
            self.source,
            self.offset,
            self.strides,
            self.alloc_tile_size,
            self.dtype,
            self.buffer_type,
            transform_strides=self.ap_strides,
            root_source=self.root_source,
            indirect=IndirectOffset(kind=kind, value=value, dim=dim),
        )

    def drop_dim(self, dim):
        """Remove dim from strides + alloc_tile_size."""
        return SBUFLayout(
            self.source,
            self.offset,
            remove_at(self.strides, dim),
            remove_at(self.alloc_tile_size, dim),
            self.dtype,
            self.buffer_type,
            transform_strides=self.ap_strides,
            root_source=self.root_source,
            indirect=self.indirect,
        )

    def drop_dims(self, dims):
        """Drop multiple dims in reverse index order."""
        result = self
        for d_idx in range(len(dims) - 1, -1, -1):
            result = result.drop_dim(dims[d_idx])
        return result

    # ================================================================
    # Remainder
    # ================================================================

    def dim_offset_elements(self, dim, element_shape):
        """Per-dim element offset from the SBUF allocation origin.

        Recovered from the flat F-offset by dividing through the
        inner-dim element span. Dim 0 (P) returns 0: SBUFLayout never
        advances P (P-narrow uses ``narrow``).

        Note: when prior advances accumulated on a different dim, the
        flat-F decomposition isn't perfectly recoverable. ``Grid.truncate_to_source``
        guards against this by skipping when the leaf is already
        clamped to ``element_shape`` (the build_view path bakes the
        partial extent into the leaf at construction).
        """
        if dim == 0:
            return 0
        inner_span = 1
        for d in range(dim + 1, len(element_shape)):
            inner_span = inner_span * element_shape[d]
        if inner_span == 0:
            return 0
        return (self.offset // inner_span) % element_shape[dim]

    def dim_addressable(self, dim, grid):
        """Source elements reachable on `dim` before walking off the source.

        See :func:`reachable_dim_extent` for the shared offset-clamp rule;
        this is the SBUF-layout entry point into it.
        """
        return reachable_dim_extent(grid, self, dim)

    def is_remainder(self, grid):
        """True when this view sits on a partial trailing tile.

        Sources: Grid's own bit (multi-tile parent / truncated leaf),
        the offset crossing element_shape on any dim, or an indirect
        offset (runtime; treated as worst case).
        """
        if self.indirect is not None or grid.is_remainder:
            return True
        for d in range(grid.ndim):
            outer = grid.outer_axis(d)
            if outer is None or outer.step == 0:
                continue
            dim_offset = self.dim_offset_elements(d, grid.element_shape)
            if dim_offset + outer.count * outer.step > grid.element_shape[d]:
                return True
        return False

    # ================================================================
    # Sub-tile indexing
    # ================================================================

    def sub_index(self, key, element_shape):
        """Sub-tile indexing: P-narrowing + F-collapsing.

        Bypasses Grid. Called on single-tile SBUF views.
        Dim 0 (P): int narrows to 1 row, slice narrows to range.
        Dims 1+ (F): int collapses dim (offset +=), slice narrows.
        Min 2D maintained.

        Returns (new_SBUFLayout, result_element_shape).
        """
        if not isinstance(key, tuple):
            key = (key,)

        ndim = len(element_shape)
        num_keys = len(key)

        assert num_keys <= ndim, (
            f"sub-tile view[...]: too many keys ({num_keys}) for {ndim}-D element. Drop trailing keys."
        )
        for d in range(num_keys):
            validate_index_key(
                key[d],
                d,
                element_shape[d],
                grid=None,
                context="sub-tile " + ("P" if d == 0 else f"F[{d}]"),
            )

        # Free-dim strides: row-major over element_shape[1:]
        free_strides = []
        for i in range(1, ndim):
            stride = 1
            for j in range(i + 1, ndim):
                stride = stride * element_shape[j]
            free_strides.append(stride)

        # Dim 0 (P): int narrows to 1 row, slice narrows to range.
        # Runtime keys are not supported at sub-tile level (SBUF has no
        # indirect path) -- reject explicitly rather than silently keeping
        # the dim full.
        if num_keys > 0 and isinstance(key[0], int):
            p_start = key[0]
            p_size = 1
        elif num_keys > 0 and isinstance(key[0], slice):
            sl = key[0]
            p_start = sl.start if sl.start is not None else 0
            p_stop = sl.stop if sl.stop is not None else element_shape[0]
            p_size = p_stop - p_start
        elif num_keys > 0:
            assert False, (
                "sub-tile view[...]: dim 0 key must be int or slice on an "
                "SBUF sub-tile view (runtime keys are not supported at "
                "sub-tile level), got " + str(type(key[0]).__name__) + "."
            )
        else:
            p_start = 0
            p_size = element_shape[0]

        # Dims 1+ (F): collapse on int, narrow on slice, keep unkeyed.
        # Runtime keys are rejected for the same reason as dim 0.
        f_offset = 0
        surviving_f = []
        for i in range(1, ndim):
            if i >= num_keys:
                surviving_f.append(element_shape[i])
                continue
            ki = key[i]
            if isinstance(ki, int):
                f_offset = f_offset + ki * free_strides[i - 1]
            elif isinstance(ki, slice):
                sl_start = ki.start if ki.start is not None else 0
                sl_stop = ki.stop if ki.stop is not None else element_shape[i]
                f_offset = f_offset + sl_start * free_strides[i - 1]
                surviving_f.append(sl_stop - sl_start)
            else:
                assert False, (
                    "sub-tile view[...]: dim " + str(i) + " key must be int "
                    "or slice on an SBUF sub-tile view (runtime keys are "
                    "not supported at sub-tile level), got " + str(type(ki).__name__) + "."
                )

        # Min 2D: at least 1 F dim
        if len(surviving_f) == 0:
            surviving_f.append(1)

        result_f = product(tuple(surviving_f))
        result_shape = tuple([p_size] + surviving_f)

        # Slice: P-rows x F-columns (independent axes). N-D SBUF sources
        # (e.g. higher-rank loaded tiles) are flattened to 2-D first --
        # F-offset arithmetic above is already flat-based via free_strides.
        # The resulting view carries its own storage offset; subsequent
        # ``ap()`` calls inherit that offset to address the slice's start.
        total_f_offset = self.offset + f_offset
        flat_src = self._as_2d(self.source)
        new_data = flat_src[p_start : p_start + p_size, total_f_offset : total_f_offset + result_f]

        new_layout = SBUFLayout(
            new_data,
            0,
            (result_f,),
            (p_size, result_f),
            self.dtype,
            self.buffer_type,
        )

        return (new_layout, result_shape)

    # ================================================================
    # Repr
    # ================================================================

    def __repr__(self):
        return (
            "SBUFLayout(offset="
            + str(self.offset)
            + ", strides="
            + str(self.strides)
            + ", alloc_tile_size="
            + str(self.alloc_tile_size)
            + ")"
        )

    # ================================================================
    # Private instance helpers
    # ================================================================

    def _as_2d(self, src):
        """Reshape N-D SBUF source to 2D (P, F). No-op for already-2D."""
        if len(src.shape) > 2:
            p = src.shape[0]
            f = product(tuple(src.shape), start=1)
            return src.reshape((p, f))
        return src

    def _is_multi_tile(self, grid):
        """True if grid spans multiple allocation tiles in any dimension.

        Uses ``alloc_tile_size`` (Layout-side, allocation-uniform) rather
        than ``grid.tile_size`` so broadcast / transform views with
        expanded logical dims still report False on the broadcast dim.
        """
        for d in range(min(grid.ndim, len(self.alloc_tile_size))):
            if grid.remaining[d] > self.alloc_tile_size[d]:
                return True
        return False

    def _ap_with_transform(self, grid):
        """AP from ap_strides (may include stride 0 for broadcast)."""
        remaining = grid.remaining
        alloc_tile_f = self._alloc_tile_f()
        actual_f = product(remaining, start=1)
        if actual_f < alloc_tile_f:
            src = self._as_2d(self.source)
            data = src[:, nl.ds(self.offset, actual_f)]
        else:
            data = self.tile_data()
        # Level 0 pins to the buffer's physical row width (what the compiler
        # validates), not the logical tile width -- a sub-tile of a multi-tile
        # allocation is backed by a wider buffer. Inner levels keep the
        # transform's strides (stride-0 broadcast, contiguous, etc).
        pattern = [[SBUFLayout._partition_row_stride(data), remaining[0]]]
        for d in range(1, len(remaining)):
            pattern.append([self.ap_strides[d], remaining[d]])
        return data.ap(pattern=pattern)

    def _ap_standard(self, grid):
        """Standard tiled AP: contiguous for single-P, flat for multi-P."""
        remaining = grid.remaining
        tile_p = self.alloc_tile_size[0]
        p_tiles = p_tile_count(remaining[0], tile_p)
        # Single P-tile: the AP level-0 count is the ADDRESSABLE P (grid.remaining[0]),
        # not the uniform alloc tile_p -- so .data on a partial P-tile (element_shape[0]
        # < tile_p) reports its real height (e.g. 44), matching how a partial trailing
        # F-tile already reports its real width. Multi-P-tile (p_tiles > 1) keeps tile_p:
        # the fold packs full-height P-tiles into F, and any trailing P-partial is
        # resolved per-tile by the Grid, not here.
        ap_tile_p = min(tile_p, remaining[0]) if p_tiles == 1 else tile_p
        slice_f = self._walked_f_extent(grid)
        if self.offset + slice_f <= self.source.shape[-1]:
            data = self.source[:, self.offset : self.offset + slice_f]
        else:
            data = self.tile_data()
        return SBUFLayout._build_ap(data, remaining, ap_tile_p, p_tiles, grid=grid)

    def _walked_f_extent(self, grid):
        """F-column span actually walked from the current offset.

        Non-sharded dims contribute `remaining[d]`. Dims with a gap walk
        (explicit SHARD label OR slice-step-induced strided outer axis)
        contribute `(owned - 1) * outer_step + inner_walk` -- only the
        owned tiles, gaps excluded.
        """
        tile_p = self.alloc_tile_size[0]
        if not grid.has_any_gapped_axis():
            return SBUFLayout.f_extent(grid.remaining, tile_p)
        f_extent = 1
        for d in range(1, len(grid.remaining)):
            f_extent = f_extent * grid.walked_extent(d)
        p_count = p_tile_count(grid.remaining[0], tile_p)
        return f_extent * p_count if p_count > 1 else f_extent

    def _ap_contiguous(self, grid=None):
        """Fallback: contiguous AP from tile_data shape."""
        alloc_tile_f = self._alloc_tile_f()
        if grid is not None:
            actual_f = product(grid.remaining, start=1)
            if actual_f < alloc_tile_f:
                src = self._as_2d(self.source)
                data = src[:, nl.ds(self.offset, actual_f)]
                return SBUFLayout._build_default_ap(data, tuple(data.shape), data.shape[0], 1)
        data = self.tile_data()
        return SBUFLayout._build_default_ap(data, tuple(data.shape), data.shape[0], 1)

    # ================================================================
    # Private static helpers
    # ================================================================

    @staticmethod
    def _tile_grid_strides(tile_size, remaining):
        """Compute tile-level F-strides for an SBUF tile grid.

        Flat F-axis is row-major over (dim1, dim2, ...); a tile at tile-grid
        position [t_0, t_1, ...] starts at F-offset = sum_d t_d * stride[d].

        - P-dim (d=0) tile stride = full per-P-tile F-extent (covers inner dims).
        - Innermost F-dim stride = tile_size[d] (contiguous span per tile).
        - Middle F-dim stride = tile_size[d] * inner F-extent.
        """
        ndim = len(tile_size)
        strides = []
        for d in range(ndim):
            if d == 0:
                stride = 1
                for d2 in range(1, ndim):
                    stride = stride * remaining[d2]
            elif d == ndim - 1:
                stride = tile_size[d]
            else:
                stride = tile_size[d]
                for d2 in range(d + 1, ndim):
                    stride = stride * remaining[d2]
            strides.append(stride)
        return tuple(strides)

    @staticmethod
    def _partition_row_stride(sbuf):
        """Element distance between adjacent SBUF partition rows.

        SBUF is partition-major: the new ``nki`` compiler validates that level-0
        of every emitted AP pattern equals the underlying ndarray's physical free
        width -- not a sliced view's narrow free width -- so this reads the
        physical row width (preserved through ``[:, slice]`` indexing).
        """
        return physical_row_width(sbuf)

    @staticmethod
    def _build_default_ap(sbuf, remaining, tile_p, p_tiles):
        """AP for the default SBUF tile walk: no gapped axes, no transform.

        Pattern shape (outer-to-inner):
          level 0: [partition_stride, tile_p]  -- pinned to the underlying
                   ndarray's free width via :meth:`_partition_row_stride`,
                   which is what the new compiler validates against.
          level 1: [walked_F, p_tiles]         -- only when stacking P-tiles
                                                  (rotating-buffer flatten)
          inner:   contiguous walk over free dims (``remaining[1:]``)

        Inheriting the operand's storage offset (``offset=None``) lets
        sub-tile views (whose internal offset addresses the slice's start)
        produce correct runtime addressing without overriding to 0.
        """
        partition_stride = SBUFLayout._partition_row_stride(sbuf)
        levels = [[partition_stride, tile_p]]
        if p_tiles > 1:
            walked_f = SBUFLayout.f_extent(remaining, tile_p)
            levels.append([1, walked_f])
        elif len(remaining) > 1:
            levels.extend(contiguous_ap_pattern(tuple(remaining[1:])))
        return sbuf.ap(pattern=levels)

    @staticmethod
    def _build_ap(sbuf, remaining, tile_p, p_tiles, effective_tile_size=None, tile_shape=None, grid=None):
        """Build SBUF AP with level structure matching HBM AP.

        When `grid` carries a strided axis on an F-dim, emits a multi-level
        AP pattern for that dim: outer stride walks owned tiles at
        `shard_step x tile_size` F-column steps; inner stride walks a
        contiguous tile. Dims without a strided axis emit a single
        contiguous level.
        """
        if effective_tile_size is None or tile_shape is None:
            if grid is None or not grid.has_any_gapped_axis():
                return SBUFLayout._build_default_ap(sbuf, remaining, tile_p, p_tiles)
            # Synthesize from grid for the structured-path code below.
            assert grid.tile_size is not None, "Sharded SBUF grid must have tile_size set"
            effective_tile_size = tuple(grid.tile_size)
            ts_list = []
            for d in range(len(remaining)):
                if d < len(effective_tile_size) and effective_tile_size[d] > 0:
                    ts_list.append(ceiling_div(remaining[d], effective_tile_size[d]))
                else:
                    ts_list.append(1)
            tile_shape = tuple(ts_list)

        partition_stride = SBUFLayout._partition_row_stride(sbuf)
        col_width = product(remaining, start=1)

        levels = [[partition_stride, tile_p]]

        if p_tiles > 1:
            levels.append([col_width, p_tiles])

        f_dim_count = len(tile_shape) - 1 if len(tile_shape) > 0 else 0
        for f_idx in range(f_dim_count):
            abs_dim = f_idx + 1
            f_tile_size_val = effective_tile_size[abs_dim]
            f_tile_count = tile_shape[abs_dim]
            f_stride = 1
            for inner_dim in range(abs_dim + 1, len(remaining)):
                f_stride = f_stride * remaining[inner_dim]

            # Strided F-dim: emit two AP levels (outer = owned-tile walk,
            # inner = per-tile walk). Skips gaps.
            # Two ways a dim can be strided:
            #   1. Explicit AxisLabel.SHARD axis.
            #   2. Slice-induced: outer axis with step exceeding inner-walk
            #      extent (e.g., view[a:b:s] retains an owned axis with
            #      step = original_step * s, no SHARD label).
            strided_outer_step = grid.strided_outer_step(abs_dim) if grid is not None else None
            if strided_outer_step is not None:
                owned_count = remaining[abs_dim] // strided_outer_step
                levels.append([f_stride * strided_outer_step, owned_count])
                levels.append([f_stride, f_tile_size_val])
                continue

            if f_idx == f_dim_count - 1:
                total_f = f_tile_count * f_tile_size_val
                if abs_dim < len(remaining):
                    total_f = min(total_f, remaining[abs_dim])
                levels.append([f_stride, total_f])
            else:
                if f_tile_count > 1:
                    levels.append([f_stride * f_tile_size_val, f_tile_count])
                else:
                    levels.append([f_stride, f_tile_size_val])

        return sbuf.ap(pattern=levels)
