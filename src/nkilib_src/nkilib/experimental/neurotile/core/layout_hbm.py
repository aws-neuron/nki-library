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
import nki.isa as nisa
import nki.language as nl

from ._helpers import (
    MAX_SBUF_PARTITION_ROWS,
    ceiling_div,
    reachable_dim_extent,
    remove_at,
    sbuf_buffer_type,
)
from .ap_emitter import APEmitter
from .axis import IndirectKind, IndirectOffset
from .grid import Grid
from .indexing import assert_valid_element_offset_value, is_sbuf_scalar_value
from .layout_sbuf import SBUFLayout


class HBMLayout(nl.NKIObject):
    """
    Physical addressing for an HBM-backed view.

    Holds the byte-level information :class:`Grid` doesn't carry: source
    tensor, compile-time offset, per-dim strides, and an optional
    :class:`~neurotile.core.axis.IndirectOffset` for runtime indexing.
    Pairs with a Grid inside an :class:`~neurotile.core.ndslice.NDSlice`
    to back ``.load()`` / ``.store()`` and to build access patterns via
    ``.ap()``.

    Immutable: ``advance`` / ``set_indirect`` / ``drop_dim`` /
    ``apply_transform`` all return fresh ``HBMLayout`` instances rather
    than mutating in place. The Grid composes those operations; HBMLayout
    never inspects axes directly except to emit the AP.

    Attributes:
        source: Root HBM tensor used by ``.ap()`` calls.
        root_source: Parent tensor for stride anchoring on sliced views;
            equals ``source`` when not set explicitly.
        offset (int): Compile-time element offset into ``source``.
        strides (tuple[int, ...]): Per-dim element strides; multiplied
            by ``axis.step`` at AP emission to get physical byte stride.
        dtype: Element dtype.
        buffer_type: ``nl.shared_hbm`` / ``nl.private_hbm`` (or a
            sentinel in test mode).
        indirect (IndirectOffset | None): Runtime scalar / vector offset
            for indirect indexing (``view[k]`` with runtime ``k``); the
            AP emitter routes it through ``scalar_offset=`` /
            ``vector_offset=`` instead of folding into ``offset``.
    """

    def __init__(
        self,
        source,
        offset,
        strides,
        dtype,
        buffer_type=None,
        indirect=None,
        root_source=None,
    ):
        self.source = source
        self.root_source = root_source if root_source is not None else source
        self.offset = offset
        self.strides = strides
        self.dtype = dtype
        self.buffer_type = buffer_type
        self.indirect = indirect

    # ================================================================
    # Position primitives
    # ================================================================

    def advance_step(self, grid, dim):
        """Per-item advance step on `dim` in this layout's native unit.

        HBM strides are at source-element granularity, so the native step
        is simply the Grid's per-item element step.
        """
        return grid.current_step(dim)

    def advance(self, dim, k, step):
        """Advance offset along `dim` by `k` items at `step` source-units per item.

        For compile-time int `k`, folds k * step * strides[dim] into self.offset.
        For runtime k, lower the logical index to a source-element scalar
        offset. SBUF scalar indices with step > 1 are scaled internally;
        loop variables with step > 1 are rejected because NKI cannot
        currently materialize that scale.
        """
        return self.advance_by_logical_index(dim, k, step)

    def advance_by_logical_index(self, dim, index, index_stride_elements):
        """Advance by a logical grid index on `dim`."""
        self._assert_valid_dim(dim)
        HBMLayout._assert_positive_index_stride(index_stride_elements)
        assert not isinstance(index, bool), "Runtime logical indexing does not accept bool indices."
        if isinstance(index, int):
            new_offset = self.offset + index * index_stride_elements * self.strides[dim]
            return HBMLayout(
                self.source,
                new_offset,
                self.strides,
                self.dtype,
                self.buffer_type,
                self.indirect,
                self.root_source,
            )

        if index_stride_elements == 1:
            assert not hasattr(index, "shape") or is_sbuf_scalar_value(index), (
                "Runtime logical indexing requires an SBUF scalar index; got shape=" + str(tuple(index.shape))
            )
            element_offset = index
        else:
            assert is_sbuf_scalar_value(index), (
                "Runtime logical indexing with index stride "
                + str(index_stride_elements)
                + " requires an SBUF scalar index so NeuroTile can materialize "
                + "the source-element offset. Use nt.element_offset(offset) "
                + "with a counter already in source-element units for loop variables."
            )
            element_offset = HBMLayout._materialize_scaled_scalar_offset(index, index_stride_elements)

        if self.indirect is not None and self.indirect.kind == IndirectKind.SCALAR and self.indirect.dim == dim:
            element_offset = self.indirect.value + element_offset
        return self._with_scalar_indirect(element_offset, dim)

    def advance_by_element_offset(self, dim, offset):
        """Advance by a source-element offset relative to this view."""
        self._assert_valid_dim(dim)
        assert_valid_element_offset_value(offset)
        if isinstance(offset, int):
            new_offset = self.offset + offset * self.strides[dim]
            return HBMLayout(
                self.source,
                new_offset,
                self.strides,
                self.dtype,
                self.buffer_type,
                self.indirect,
                self.root_source,
            )
        if self.indirect is not None and self.indirect.kind == IndirectKind.SCALAR and self.indirect.dim == dim:
            offset = self.indirect.value + offset
        return self._with_scalar_indirect(offset, dim)

    def _with_scalar_indirect(self, value, dim):
        """Return a layout with a scalar indirect source-element offset."""
        return HBMLayout(
            self.source,
            self.offset,
            self.strides,
            self.dtype,
            self.buffer_type,
            IndirectOffset(kind=IndirectKind.SCALAR, value=value, dim=dim),
            self.root_source,
        )

    def _assert_valid_dim(self, dim):
        """Validate a source-tensor dimension index."""
        assert not isinstance(dim, bool) and isinstance(dim, int) and 0 <= dim < len(self.strides), (
            "dim must be an int in [0, " + str(len(self.strides)) + "); got " + str(dim)
        )

    @staticmethod
    def _assert_positive_index_stride(index_stride_elements):
        """Validate the logical-index stride metadata."""
        assert (
            not isinstance(index_stride_elements, bool)
            and isinstance(index_stride_elements, int)
            and index_stride_elements > 0
        ), "index_stride_elements must be a positive int; got " + str(index_stride_elements)

    @staticmethod
    def _materialize_scaled_scalar_offset(value, scale):
        """Materialize value * scale as an SBUF scalar."""
        dtype = value.dtype if hasattr(value, "dtype") else nl.int32
        scaled = nl.ndarray((1, 1), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=value, op0=nl.multiply, operand0=scale)
        return scaled

    def set_indirect(self, kind, value, dim):
        """Replace self.indirect with a new tagged offset."""
        return HBMLayout(
            self.source,
            self.offset,
            self.strides,
            self.dtype,
            self.buffer_type,
            IndirectOffset(kind=kind, value=value, dim=dim),
            self.root_source,
        )

    def drop_dim(self, dim):
        """Remove dim from strides. indirect.dim stays unchanged (source-relative)."""
        return HBMLayout(
            self.source,
            self.offset,
            remove_at(self.strides, dim),
            self.dtype,
            self.buffer_type,
            self.indirect,
            self.root_source,
        )

    def drop_dims(self, dims):
        """Drop multiple dims in reverse index order."""
        result = self
        for d_idx in range(len(dims) - 1, -1, -1):
            result = result.drop_dim(dims[d_idx])
        return result

    # ================================================================
    # Strides / transforms
    # ================================================================

    def transform_strides(self, element_shape):
        """Strides for transform computation."""
        return self.strides

    def apply_transform(self, new_strides):
        """Return layout for a transformed view (reshape/permute/broadcast)."""
        return HBMLayout(
            self.source,
            self.offset,
            new_strides,
            self.dtype,
            self.buffer_type,
            self.indirect,
            self.root_source,
        )

    def retile(self, new_tile_size, new_remaining):
        """Return a fresh layout with the same fields.

        HBM strides are source-element granularity, so re-tiling does
        not change the layout's contents. We still return a new
        instance to keep ``Layout.retile`` aligned with every other
        layout transform (each returns a fresh value, never aliases
        ``self``), so callers can treat layouts as immutable values.
        """
        return HBMLayout(
            self.source,
            self.offset,
            self.strides,
            self.dtype,
            self.buffer_type,
            self.indirect,
            self.root_source,
        )

    # ================================================================
    # Data access
    # ================================================================

    def tile_data(self):
        """HBM source tensor for direct NKI indexing."""
        return self.source

    def get_data(self, grid):
        """HBM views have no tile-local data; returns None."""
        return None

    def ap(self, grid):
        """Build the N-D HBM access pattern from grid.axes + self.strides."""
        levels = APEmitter.emit(grid.axes, self.strides, self.addressable_extents(grid))
        return HBMLayout._apply_ap(self.source, self.offset, levels, self.indirect)

    # ================================================================
    # Load / store
    # ================================================================

    def load(
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
        """Load from HBM to SBUF. Returns (Grid, SBUFLayout)."""
        load_dtype = dtype if dtype is not None else self.dtype

        if transpose:
            # engine= steers HWDGE descriptor gen on the dma_copy path only;
            # nisa.dma_transpose has no engine param, so it cannot be honored here.
            assert engine is None, (
                "load(transpose=True, engine=...): engine= is not supported on the "
                "transpose path (nisa.dma_transpose has no engine param). Drop engine= "
                "or use a non-transpose load."
            )
            return self._load_transpose(
                grid,
                load_dtype,
                dst=dst,
                transpose_axes=transpose_axes,
                dge_mode=dge_mode,
                oob_mode=oob_mode,
                oob_value=oob_value,
                priority=priority,
            )

        remaining = grid.remaining
        owned_extents = grid.owned_extents()
        effective_tile_size, tile_shape = HBMLayout.compute_effective_tiles(
            remaining,
            grid.tile_size,
            owned_extents=owned_extents,
        )
        sbuf_shape, p_tiles = HBMLayout.compute_sbuf_alloc(
            remaining,
            grid.tile_size,
            owned_extents=owned_extents,
        )
        tile_p = effective_tile_size[0]

        assert p_tiles == 1 or remaining[0] % tile_p == 0, (
            "Multi-tile load with P-remainder: remaining[0]="
            + str(remaining[0])
            + " not divisible by tile_p="
            + str(tile_p)
        )

        sbuf = self._resolve_sbuf(dst, out_shape, sbuf_shape, load_dtype)
        if oob_value is not None:
            nisa.memset(sbuf, oob_value)

        self._issue_load_dma(
            grid,
            sbuf,
            effective_tile_size,
            tile_shape,
            tile_p,
            p_tiles,
            pattern_override,
            dst,
            out_shape,
            oob_mode,
            dge_mode,
            priority,
            engine=engine,
        )

        return HBMLayout._wrap_sbuf_load(
            sbuf,
            grid,
            self,
            effective_tile_size,
            tile_shape,
            load_dtype,
            out_shape=out_shape,
        )

    def store(self, data, grid, oob_mode=None, dge_mode=None, priority=None, pattern_override=None, engine=None):
        """Build HBM AP, issue DMA from SBUF to HBM."""
        if pattern_override is not None:
            hbm_ap = HBMLayout._apply_ap(self.source, self.offset, pattern_override, self.indirect)
        else:
            hbm_ap = self.ap(grid)
        HBMLayout._dma_copy(
            dst=hbm_ap, src=data, oob_mode=oob_mode, dge_mode=dge_mode, priority=priority, engine=engine
        )

    def sbuf_shape_for(self, grid):
        """Compute SBUF allocation shape for a load from this region."""
        return HBMLayout.compute_sbuf_alloc(
            grid.remaining,
            grid.tile_size,
            owned_extents=grid.owned_extents(),
        )[0]

    def transposed_sbuf_shape_for(self, grid):
        """SBUF slot shape for a .load(transpose=True), sized from owned extents
        so the slot stays dense under interleaved sharding."""
        rows, p_dim, num_chunks = self._transposed_dims_for(grid, grid.owned_extents())
        return (rows, p_dim * num_chunks)

    def transposed_tile_size_for(self, grid):
        """Per-tile granularity of a .load(transpose=True) output: one source
        tile's transposed shape (sized from grid.tile_size, not the whole block)."""
        rows, tile_p, num_chunks = self._transposed_dims_for(grid, grid.tile_size)
        return (rows, tile_p * num_chunks)

    def _transposed_dims_for(self, grid, extents=None):
        """Shared transpose-output dim math ``(rows, p_dim, num_chunks)`` for both
        the packed shape and the per-tile size. ``extents`` sets the granularity."""
        if extents is None:
            extents = grid.remaining
        p_idx, f_idx = HBMLayout._transpose_pf_dims(grid)
        p_dim, f_dim = extents[p_idx], extents[f_idx]
        num_chunks = len(HBMLayout._f_chunks(f_dim))
        rows = f_dim if num_chunks == 1 else MAX_SBUF_PARTITION_ROWS
        return (rows, p_dim, num_chunks)

    @staticmethod
    def _transpose_pf_dims(grid):
        """Pick the (P, F) dim indices for a 2-D DMA transpose.

        Dims carrying extent (remaining > 1) come first, then the trivial
        size-1 dims (from a consumed batch / block iter) pad to two indices.
        """
        extentful = []
        trivial = []
        for d in range(grid.ndim):
            if grid.remaining[d] > 1:
                extentful.append(d)
            else:
                trivial.append(d)
        ordered = extentful + trivial
        return ordered[0], ordered[1]

    # ================================================================
    # Remainder
    # ================================================================

    def dim_offset_elements(self, dim, element_shape):
        """Elements consumed on `dim` from the source origin to this view.

        Precondition: ``element_shape[dim]`` is the source extent -- it is
        the modular period. Holds for remainder dims, NOT after a ``slice``
        / ``reshape_dim`` narrows it (then the result is meaningless).
        """
        if self.strides is None:
            return 0
        if dim >= len(self.strides):
            return 0
        stride = self.strides[dim]
        if stride == 0:
            return 0
        period = element_shape[dim] if dim < len(element_shape) else 1
        if period <= 0:
            return 0
        return (self.offset // stride) % period

    def dim_addressable(self, dim, grid):
        """Source elements reachable on `dim` before walking off the source.

        See :func:`reachable_dim_extent` for the shared offset-clamp rule;
        this is the HBM-layout entry point into it.
        """
        return reachable_dim_extent(grid, self, dim)

    def addressable_extents(self, grid):
        """Per-dim AP merge-clamp ceiling.

        Unlike ``sbuf_load_extents``, does NOT clamp to owned extents: the
        AP walks ``grid.axes`` directly, which already encode shard gaps.
        """
        result = []
        for d in range(grid.ndim):
            result.append(self.dim_addressable(d, grid))
        return tuple(result)

    def sbuf_load_extents(self, grid):
        """Per-dim element counts the SBUF load destination is sized to:
        owned (excludes shard gaps) intersected with source-reachable."""
        owned = grid.owned_extents()
        result = []
        for d in range(grid.ndim):
            result.append(min(owned[d], self.dim_addressable(d, grid)))
        return tuple(result)

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
    # Static helpers (tile/SBUF allocation math, DMA emission)
    # ================================================================

    @staticmethod
    def compute_effective_tiles(remaining, tile_size, owned_extents=None):
        """Compute effective tile size and tile shape from region and tile config.

        `owned_extents` (when provided) supplies per-dim owned-element counts
        so tile_shape reflects owned tiles (skipping shard gaps).
        """
        effective_tile_size = []
        tile_shape = []
        for d in range(len(tile_size)):
            if d < len(remaining):
                ets = min(tile_size[d], remaining[d])
            else:
                ets = tile_size[d]
            effective_tile_size.append(ets)
            if ets <= 0:
                tile_shape.append(1)
                continue
            if owned_extents is not None and d < len(owned_extents):
                tile_shape.append(ceiling_div(owned_extents[d], ets))
            elif d < len(remaining):
                tile_shape.append(ceiling_div(remaining[d], ets))
            else:
                tile_shape.append(1)
        return (tuple(effective_tile_size), tuple(tile_shape))

    @staticmethod
    def compute_sbuf_alloc(remaining, tile_size, owned_extents=None):
        """Compute SBUF allocation shape and owned P-tile count."""
        eff_ts, _ = HBMLayout.compute_effective_tiles(
            remaining,
            tile_size,
            owned_extents=owned_extents,
        )
        tile_p = eff_ts[0]
        p_span = owned_extents[0] if owned_extents is not None else remaining[0]
        p_tiles = p_span // tile_p
        if p_tiles > 1:
            f_span = (p_span,) + tuple(remaining[1:])
            total_f = SBUFLayout.f_extent(f_span, tile_p)
            return (tile_p, total_f), p_tiles
        return tuple(remaining), p_tiles

    @staticmethod
    def load_partition_fold(layout, grid, recipe, dtype, dge_mode, oob_mode=None, priority=None, engine=None):
        """Load with K separate DMAs for partition fold. Returns (Grid, SBUFLayout)."""
        sbuf_buffer = HBMLayout._partition_fold_load_dma(
            layout.source,
            layout.offset,
            recipe,
            grid.remaining,
            dtype,
            dge_mode,
            oob_mode=oob_mode,
            priority=priority,
            engine=engine,
        )
        return SBUFLayout.build_view(
            sbuf_buffer,
            grid.tile_size,
            grid.tile_size,
            dtype,
            sbuf_buffer_type(),
        )

    @staticmethod
    def store_partition_fold(
        source, offset, fold_recipe, element_shape, data, dge_mode, priority=None, oob_mode=None, engine=None
    ):
        """Store with K separate DMAs for partition fold."""
        K, P_per_slice, fold_stride, base_pattern = fold_recipe
        f_total = 1
        for d in range(1, len(element_shape)):
            f_total = f_total * element_shape[d]
        f_per_slice = f_total
        for k in range(K):
            dst_offset = offset + k * fold_stride
            hbm_ap = source.ap(pattern=base_pattern, offset=dst_offset)
            p_start = k * P_per_slice
            sbuf_offset = p_start * f_total
            sbuf_ap = data.ap(
                pattern=[[f_total, P_per_slice], [1, f_per_slice]],
                offset=sbuf_offset,
            )
            HBMLayout._dma_copy(
                dst=hbm_ap, src=sbuf_ap, oob_mode=oob_mode, dge_mode=dge_mode, priority=priority, engine=engine
            )

    # ================================================================
    # Repr
    # ================================================================

    def __repr__(self):
        parts = "HBMLayout(offset=" + str(self.offset)
        parts = parts + ", strides=" + str(self.strides)
        if self.indirect is not None:
            parts = parts + ", indirect=" + str(self.indirect)
        return parts + ")"

    # ================================================================
    # Private instance helpers
    # ================================================================

    def _load_transpose(
        self, grid, dtype, dst=None, transpose_axes=None, dge_mode=None, oob_mode=None, oob_value=None, priority=None
    ):
        """DMA transpose path. Returns (Grid, SBUFLayout).

        Static (non-indirect) views take the 2-D chunked transpose; indirect
        views gather rows then transpose per ``transpose_axes``.
        """
        if self.indirect is not None:
            # Gather transpose: one gathered+transposed result, typed as a single tile
            # (the hardware bounds it to src.shape[-1] <= 128, so no chunk grid).
            sbuf, packed_shape, _ = self._gather_transpose(
                grid,
                dtype,
                transpose_axes=transpose_axes,
                dst=dst,
                dge_mode=dge_mode,
                oob_mode=oob_mode,
                oob_value=oob_value,
                priority=priority,
            )
            return SBUFLayout.build_view(sbuf, packed_shape, packed_shape, dtype, sbuf_buffer_type())

        assert transpose_axes is None, (
            "NDSlice.load(transpose=True, transpose_axes=...): transpose_axes is only "
            "supported on the indirect (gather) transpose; a static transpose is 2-D."
        )
        p_dim_idx, f_dim_idx = HBMLayout._transpose_pf_dims(grid)
        remaining_2d = (grid.remaining[p_dim_idx], grid.remaining[f_dim_idx])
        strides_2d = (self.strides[p_dim_idx], self.strides[f_dim_idx])
        sbuf = HBMLayout._dma_transpose(
            self.source,
            self.offset,
            strides_2d,
            remaining_2d,
            dtype,
            dst=dst,
            dge_mode=dge_mode,
            oob_mode=oob_mode,
            oob_value=oob_value,
            priority=priority,
        )
        # Preserve the tile grid across the transpose (axis-swap): the packed SBUF
        # buffer holds one transposed tile per source tile, laid out along the free
        # axis. Typing the view at (whole packed shape, per-tile transposed size)
        # lets Grid derive the swapped tile grid -- e.g. a (K, 1) source tile grid
        # becomes a (1, K) grid of [F, P] tiles, indexable by coordinate. This is the
        # same (element_shape, tile_size) build a streamed transpose slot already uses.
        # Axis-swap (not "keep the source grid shape") is the convention because a
        # transpose is a logical axis permutation everywhere -- the dma_transpose ISA op
        # (a [1,0]/[2,1,0]/... reversal), numpy .T, jax lax.transpose -- so the grid
        # transposes with the data and the result indexes like a non-transpose load.
        return SBUFLayout.build_view(
            sbuf,
            self.transposed_sbuf_shape_for(grid),
            self.transposed_tile_size_for(grid),
            dtype,
            sbuf_buffer_type(),
        )

    def _gather_transpose(
        self, grid, dtype, transpose_axes=None, dst=None, dge_mode=None, oob_mode=None, oob_value=None, priority=None
    ):
        """Indirect (gather) transpose via nisa.dma_transpose with a vector_offset
        AP. ``transpose_axes`` selects the rank (validated at the NDSlice.load
        boundary); ``None`` defaults to the permutation for the view's dim count.
        """
        dims = grid.gathered_dims()
        # The 4-D reshape-trick form must be requested explicitly -- 3 real dims
        # cannot be inferred as 4-D.
        if transpose_axes is None:
            transpose_axes = HBMLayout._default_transpose_axes(max(len(dims), 2))

        hbm_pattern, sbuf_pattern, transposed_shape = HBMLayout._gather_transpose_aps(dims, transpose_axes)
        if dst is not None:
            assert tuple(dst.shape) == transposed_shape, (
                "gather transpose dst shape " + str(tuple(dst.shape)) + " != expected " + str(transposed_shape)
            )
            sbuf = dst
        else:
            sbuf = nl.ndarray(transposed_shape, dtype=dtype, buffer=nl.sbuf)
        if oob_value is not None:
            nisa.memset(sbuf, oob_value)

        # TODO(oob-gather-transpose): oob_mode.skip / oob_value is plumbed but its
        # per-row skip semantics are UNVERIFIED on hardware. The OOB sentinel is -1
        # as int32 then .view(uint32) -> 0xFFFFFFFF; re-probe and add an asserting
        # test before relying on it. Tracked in neurotile_transpose_load_gaps.md 6.3.
        HBMLayout._issue_dma_transpose(
            self.source,
            self.offset,
            hbm_pattern,
            sbuf,
            sbuf_pattern,
            0,
            self.indirect,
            dge_mode=dge_mode,
            oob_mode=oob_mode,
            priority=priority,
            axes=transpose_axes,
        )
        ones = []
        for _ in range(len(transposed_shape)):
            ones.append(1)
        tile_shape = tuple(ones)
        return (sbuf, transposed_shape, tile_shape)

    @staticmethod
    def _default_transpose_axes(rank):
        """nisa.dma_transpose's supported axis permutation for a transpose rank,
        or None if unsupported. 2-D->(1,0), 3-D->(2,1,0), 4-D->(3,1,2,0)."""
        if rank == 2:
            return (1, 0)
        if rank == 3:
            return (2, 1, 0)
        if rank == 4:
            return (3, 1, 2, 0)
        return None

    @staticmethod
    def _gather_transpose_aps(dims, transpose_axes):
        """Build (hbm_pattern, sbuf_pattern, dst_shape) for a gather-transpose.

        ``dims`` are the gathered logical extents, dim 0 the gathered-rows dim.
        The 4-D form inserts the size-1 padding dim the hardware requires.
        """
        rank = len(transpose_axes)
        if rank == 2:
            rows, d = dims[0], dims[1]
            hbm = [[d, rows], [1, d]]
            sbuf = [[rows, d], [1, rows]]
            return hbm, sbuf, (d, rows)
        if rank == 3:
            rows, n_tiles, tile = dims[0], dims[1], dims[2]
            hbm = [[tile * n_tiles, rows], [tile, n_tiles], [1, tile]]
            sbuf = [[rows * n_tiles, tile], [rows, n_tiles], [1, rows]]
            return hbm, sbuf, (tile, n_tiles, rows)
        # rank == 4: src (rows, 1, f_tiles, P) -> dst (P, 1, f_tiles, rows)
        rows, f_tiles, p = dims[0], dims[1], dims[2]
        hbm = [[f_tiles * p, rows], [1, 1], [p, f_tiles], [1, p]]
        sbuf = [[f_tiles * rows, p], [1, 1], [rows, f_tiles], [1, rows]]
        return hbm, sbuf, (p, 1, f_tiles, rows)

    def _issue_load_dma(
        self,
        grid,
        sbuf,
        effective_tile_size,
        tile_shape,
        tile_p,
        p_tiles,
        pattern_override,
        dst,
        out_shape,
        oob_mode,
        dge_mode,
        priority,
        engine=None,
    ):
        """Build HBM + SBUF APs and issue DMA copy.

        Both APs walk the *addressable* extent (post-offset), not the
        SBUF buffer's allocation. The buffer can be larger (uniform
        rotating-pool slot, or pre-allocated dst) -- the trailing
        partial region is left uninitialized; the partial-aware Grid
        on the returned view ensures downstream consumers slice to
        the actual extent.
        """
        if pattern_override is not None:
            hbm_ap = HBMLayout._apply_ap(self.source, self.offset, pattern_override, self.indirect)
        else:
            hbm_ap = self.ap(grid)

        if out_shape is not None:
            # Custom load: SBUF buffer is exactly out_shape; walk it as a
            # single contiguous tile rather than re-using the HBM grid's
            # tile structure.
            sbuf_remaining = tuple(out_shape)
            sbuf_tile_p = out_shape[0]
            sbuf_p_tiles = 1
            ets = None
            ts = None
        else:
            sbuf_remaining = self.sbuf_load_extents(grid)
            sbuf_tile_p = tile_p
            sbuf_p_tiles = p_tiles
            use_matched = pattern_override is None and dst is None
            ets = effective_tile_size if use_matched else None
            ts = tile_shape if use_matched else None
        sbuf_ap = SBUFLayout._build_ap(sbuf, sbuf_remaining, sbuf_tile_p, sbuf_p_tiles, ets, ts)
        HBMLayout._dma_copy(
            dst=sbuf_ap, src=hbm_ap, oob_mode=oob_mode, dge_mode=dge_mode, priority=priority, engine=engine
        )

    def _resolve_sbuf(self, dst, out_shape, default_shape, dtype):
        """Resolve SBUF buffer: use dst, allocate from out_shape, or default."""
        if dst is not None:
            return dst
        if out_shape is not None:
            return nl.ndarray(tuple(out_shape), dtype=dtype, buffer=nl.sbuf)
        return nl.ndarray(default_shape, dtype=dtype, buffer=nl.sbuf)

    # ================================================================
    # Private static helpers (DMA emission, transpose paths, fold)
    # ================================================================

    @staticmethod
    def _apply_ap(source, offset, pattern, indirect):
        """Build AP with optional indirect dispatch. Single source of truth."""
        if indirect is None:
            return source.ap(pattern=pattern, offset=offset)
        if indirect.kind == IndirectKind.SCALAR:
            return source.ap(
                pattern=pattern,
                offset=offset,
                scalar_offset=indirect.value,
                indirect_dim=indirect.dim,
            )
        return source.ap(
            pattern=pattern,
            offset=offset,
            vector_offset=indirect.value,
            indirect_dim=indirect.dim,
        )

    @staticmethod
    def _dma_copy(dst, src, oob_mode=None, dge_mode=None, priority=None, engine=None):
        """Issue nisa.dma_copy; None args map to each op's native default.

        ``engine`` selects the HWDGE descriptor-generation engine
        (``nisa.engine.sync`` / ``nisa.engine.scalar``); only honored when
        ``dge_mode=hwdge`` (nisa ignores it otherwise). Lets a caller steer
        descriptor generation off a contended engine queue."""
        if oob_mode is None:
            oob_mode = nisa.oob_mode.error
        if dge_mode is None:
            dge_mode = nisa.dge_mode.unknown
        if engine is None:
            engine = nisa.engine.unknown
        nisa.dma_copy(dst=dst, src=src, oob_mode=oob_mode, dge_mode=dge_mode, priority=priority, engine=engine)

    @staticmethod
    def _f_chunks(f_dim):
        """Tile the transpose free dim F into chunks of <= MAX_SBUF_PARTITION_ROWS
        (F maps onto SBUF partition rows, capped at 128). Returns
        ``(col_offset, width)`` pairs covering ``[0, f_dim)`` left to right.
        """
        chunks = []
        col = 0
        while col < f_dim:
            width = min(MAX_SBUF_PARTITION_ROWS, f_dim - col)
            chunks.append((col, width))
            col = col + width
        return chunks

    @staticmethod
    def _dma_transpose(
        source,
        offset,
        strides,
        remaining,
        dtype,
        dst=None,
        dge_mode=None,
        oob_mode=None,
        oob_value=None,
        priority=None,
    ):
        """Transpose-load an HBM (p_dim, f_dim) tile into SBUF via nisa.dma_transpose.

        Returns the SBUF ndarray. F maps onto SBUF partition rows (capped at 128),
        so it is one <=128 chunk (F <= 128) or several full 128-chunks laid
        side-by-side (F a multiple of 128) -- both a single coalesced DMA. The
        single-DMA assert rejects the two-DMA case (F > 128 and not a multiple).
        """
        assert len(remaining) == 2
        p_dim = remaining[0]
        f_dim = remaining[1]
        row_stride = strides[0]

        chunks = HBMLayout._f_chunks(f_dim)
        num_chunks = len(chunks)
        # Single-DMA contract: a transposed free dim that is both wider than one
        # 128-chunk AND not a multiple of 128 would need two dma_transpose calls
        # (batched full chunks + a trailing partial). Rather than silently break
        # the one-load-one-DMA contract, reject it and tell the caller to split.
        assert f_dim <= MAX_SBUF_PARTITION_ROWS or f_dim % MAX_SBUF_PARTITION_ROWS == 0, (
            "NDSlice.load(transpose=True): the transposed free dim F="
            + str(f_dim)
            + " is > "
            + str(MAX_SBUF_PARTITION_ROWS)
            + " and not a multiple of "
            + str(MAX_SBUF_PARTITION_ROWS)
            + ", so it cannot be one coalesced DMA (the single-DMA contract). "
            "Split the view and load each part separately, e.g.\n"
            "    for t in view.whole_tiles():     t.load(transpose=True)\n"
            "    for r in view.remainder_tiles(): r.load(transpose=True)\n"
            "or choose a tile_size whose free extent is <= " + str(MAX_SBUF_PARTITION_ROWS) + " or a multiple of it."
        )
        rows = f_dim if num_chunks == 1 else MAX_SBUF_PARTITION_ROWS
        transposed_shape = (rows, p_dim * num_chunks)
        if dst is not None:
            assert tuple(dst.shape) == transposed_shape, (
                "transpose load dst shape " + str(tuple(dst.shape)) + " != expected " + str(transposed_shape)
            )
            sbuf = dst
        else:
            sbuf = nl.ndarray(transposed_shape, dtype=dtype, buffer=nl.sbuf)
        if oob_value is not None:
            nisa.memset(sbuf, oob_value)

        # Level-0 stride must be the buffer's physical row width (what the
        # compiler validates), not logical p_dim*num_chunks -- a `dst` sub-tile
        # of a wider slot is backed by a wider buffer.
        part_stride = SBUFLayout._partition_row_stride(sbuf)
        num_full = f_dim // MAX_SBUF_PARTITION_ROWS
        # The single-DMA assert above leaves exactly one of these two blocks live:
        # (a) fires for F == 128 or a multiple of 128 (all chunks full-width);
        # (b) fires for F < 128 (one sub-128 chunk). They never both run.
        # (a) Batch every full 128-wide chunk into a single dma_transpose.
        if num_full > 0:
            P = MAX_SBUF_PARTITION_ROWS
            hbm_pattern = [[row_stride, p_dim], [1, 1], [P, num_full], [1, P]]
            sbuf_pattern = [[part_stride, P], [1, 1], [p_dim, num_full], [1, p_dim]]
            HBMLayout._issue_dma_transpose(
                source,
                offset,
                hbm_pattern,
                sbuf,
                sbuf_pattern,
                0,
                None,
                dge_mode=dge_mode,
                oob_mode=oob_mode,
                priority=priority,
            )
        # (b) Single sub-128 chunk (F < 128) -> its own column block.
        if num_chunks > num_full:
            col_offset, rem = chunks[num_full]
            hbm_pattern = [[row_stride, p_dim], [row_stride, 1], [row_stride, 1], [1, rem]]
            sbuf_pattern = [[part_stride, rem], [part_stride, 1], [part_stride, 1], [1, p_dim]]
            HBMLayout._issue_dma_transpose(
                source,
                offset + col_offset,
                hbm_pattern,
                sbuf,
                sbuf_pattern,
                num_full * p_dim,
                None,
                dge_mode=dge_mode,
                oob_mode=oob_mode,
                priority=priority,
            )
        return sbuf

    @staticmethod
    def _issue_dma_transpose(
        source,
        offset,
        hbm_pattern,
        sbuf,
        sbuf_pattern,
        sbuf_offset,
        indirect,
        dge_mode=None,
        oob_mode=None,
        priority=None,
        axes=None,
    ):
        """Build APs and issue nisa.dma_transpose; None args map to nisa defaults.
        axes=(1, 0) is required for the indirect (gather) transpose.

        NOTE: nisa.dma_transpose has NO ``engine=`` param (unlike nisa.dma_copy) --
        its HWDGE descriptor-gen engine is not caller-selectable, so the engine
        plumbing stops at the dma_copy path."""
        hbm_ap = HBMLayout._apply_ap(source, offset, hbm_pattern, indirect)
        sbuf_ap = sbuf.ap(pattern=sbuf_pattern, offset=sbuf_offset)
        if dge_mode is None:
            dge_mode = nisa.dge_mode.unknown
        if oob_mode is None:
            oob_mode = nisa.oob_mode.error
        nisa.dma_transpose(dst=sbuf_ap, src=hbm_ap, axes=axes, dge_mode=dge_mode, oob_mode=oob_mode, priority=priority)

    @staticmethod
    def _partition_fold_load_dma(
        source, offset, fold_recipe, element_shape, dtype, dge_mode, oob_mode=None, priority=None, engine=None
    ):
        """Load with K separate DMAs for partition fold. Returns SBUF ndarray."""
        K, P_per_slice, fold_stride, base_pattern = fold_recipe
        total_P = K * P_per_slice
        f_total = 1
        for d in range(1, len(element_shape)):
            f_total = f_total * element_shape[d]
        f_per_slice = f_total
        sbuf = nl.ndarray((total_P, f_total), dtype=dtype, buffer=nl.sbuf)
        for k in range(K):
            src_offset = offset + k * fold_stride
            hbm_ap = source.ap(pattern=base_pattern, offset=src_offset)
            p_start = k * P_per_slice
            sbuf_offset = p_start * f_total
            sbuf_ap = sbuf.ap(
                pattern=[[f_total, P_per_slice], [1, f_per_slice]],
                offset=sbuf_offset,
            )
            HBMLayout._dma_copy(
                dst=sbuf_ap, src=hbm_ap, oob_mode=oob_mode, dge_mode=dge_mode, priority=priority, engine=engine
            )
        return sbuf

    @staticmethod
    def _wrap_sbuf_load(
        sbuf,
        hbm_grid,
        hbm_layout,
        effective_tile_size,
        tile_shape,
        dtype,
        out_shape=None,
    ):
        """Wrap raw SBUF buffer as (Grid, SBUFLayout) after a load.

        Types the SBUF view at the HBM source's *addressable* per-dim
        extent: the Grid carries the partial-trailing-tile extent
        (clamped via the layout offset) so downstream ``[k, i].data``
        reports per-tile partial widths correctly. The SBUF buffer's
        physical allocation may be larger -- AP / DMA emission walks at
        the uniform allocation stride, while view typing uses the
        addressable extent. Single rule, no conditional fallback.

        With ``out_shape`` (load() with custom ``pattern_override`` /
        ``out_shape``), the loaded SBUF tile has ``out_shape`` extent --
        independent of the HBM source tile_size -- so the view is typed
        at ``out_shape`` as a single tile.
        """
        if out_shape is not None:
            sbuf_grid, sbuf_layout = SBUFLayout.build_view(
                sbuf,
                tuple(out_shape),
                tuple(out_shape),
                dtype,
                sbuf_buffer_type(),
                block_size=hbm_grid.block_size,
            )
            return sbuf_grid, sbuf_layout

        sbuf_grid, sbuf_layout = SBUFLayout.build_view(
            sbuf,
            hbm_layout.sbuf_load_extents(hbm_grid),
            tuple(effective_tile_size),
            dtype,
            sbuf_buffer_type(),
            block_size=hbm_grid.block_size,
        )

        # The loaded SBUF view mirrors the HBM iteration state: same per-dim
        # axis depth, same cursor. Truncate the freshly-built SBUF grid to
        # match HBM's per-dim depth (drop outer TILE/BLOCK wrappers HBM has
        # already consumed) and copy HBM's cursor so .shape reports the
        # right iteration extent.
        new_axes = []
        for d in range(sbuf_grid.ndim):
            sbuf_axes_d = sbuf_grid.axes_for(d)
            if d < hbm_grid.ndim:
                hbm_count = len(hbm_grid.axes_for(d))
            else:
                hbm_count = len(sbuf_axes_d)
            keep = min(len(sbuf_axes_d), hbm_count)
            kept = sbuf_axes_d[len(sbuf_axes_d) - keep :]
            for ax in kept:
                new_axes.append(ax)

        sbuf_grid = Grid(
            element_shape=sbuf_grid.element_shape,
            axes=tuple(new_axes),
            cursor=hbm_grid.cursor,
            n_batch_dims=sbuf_grid.n_batch_dims,
            tiled=sbuf_grid.tiled,
        )

        return sbuf_grid, sbuf_layout
