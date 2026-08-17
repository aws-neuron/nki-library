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
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ._helpers import (
    nki_strided_view,
    reachable_dim_extent,
    replace_at,
    sbuf_buffer_type,
    validate_index_key,
)
from .axis import Axis, AxisLabel, IndirectKind
from .grid import Grid
from .indexing import ElementOffset, assert_valid_element_offset_value, is_scalar_shape
from .layout_hbm import HBMLayout
from .layout_psum import PSUMLayout
from .layout_sbuf import SBUFLayout
from .transforms import (
    # All metadata transforms (reshape / reshape_dim / permute / flatten_dims /
    # squeeze_dim / broadcast / expand_dim) delegate to the source NkiTensor's
    # native view ops (see _as_nki_view). compute_fold stays: it is a DMA recipe
    # for merging non-adjacent / partition dims, with no NkiTensor equivalent.
    compute_fold,
)

# ============================================================================
# NDSlice
# ============================================================================


def _shift_dims_after_drop(dims, dropped_dims):
    """Re-number ``dims`` after ``dropped_dims`` were removed from the grid.

    A dim id ``D`` becomes ``D - count(d in dropped_dims where d < D)``;
    if ``D`` itself was dropped, it's removed from the result.
    """
    result = []
    for d in dims:
        if d in dropped_dims:
            continue
        shift = 0
        for dd in dropped_dims:
            if dd < d:
                shift = shift + 1
        result.append(d - shift)
    return result


def _is_post_consume_leaf(grid, dim):
    """True when ``dim`` has exactly one axis left and that axis is a
    sub-tile leaf left over after a parent index consumed the iteration
    level above it.

    Triggered states:
      - PARTITION: only the partition dim of a tiled view carries this
        label, and only after the TILE / BLOCK above it was consumed.
      - ELEM with ``walked < element_shape[dim]``: the elem-leaf was
        clamped after a prior index (post-consume sub-tile span).

    NOT triggered for untouched batch / untiled ELEM (``walked ==
    element_shape[dim]``) or broadcast (``step == 0``) -- those are
    still indexable by partial keys.
    """
    axes = grid.axes_for(dim)
    if len(axes) != 1:
        return False
    only = axes[0]
    if only.label == AxisLabel.PARTITION:
        return True
    if only.label not in (AxisLabel.ELEM, AxisLabel.BROADCAST):
        return False
    if only.step == 0:
        return False
    return only.count * only.step < grid.element_shape[dim]


def _is_partition_leaf_only(grid, dim):
    """True when ``dim`` has exactly one axis and it's the partition
    leaf. Int indexing on this state narrows the partition to count=1
    (sub-tile P-row) rather than fully consuming the dim -- SBUF
    allocations must keep a partition-row axis (2-D physical memory).
    """
    axes = grid.axes_for(dim)
    return len(axes) == 1 and axes[0].label == AxisLabel.PARTITION


def _element_offset_value(value):
    """Return raw scalar data for an ElementOffset value."""
    if isinstance(value, NDSlice):
        assert isinstance(value._layout, SBUFLayout), (
            "nt.element_offset(NDSlice): value must be an SBUF-backed scalar view."
        )
        assert is_scalar_shape(value.element_shape), (
            "nt.element_offset(NDSlice): value must be scalar-shaped; got element_shape=" + str(value.element_shape)
        )
        value = value._layout.source
    assert_valid_element_offset_value(value)
    return value


def _assert_static_element_offset_in_bounds(offset, grid, layout, dim):
    """Validate static ElementOffset bounds for the current view."""
    if not isinstance(offset, int):
        return
    extent = reachable_dim_extent(grid, layout, dim)
    assert offset < extent, (
        "nt.element_offset(value): static offset "
        + str(offset)
        + " out of range on dim "
        + str(dim)
        + " (valid range: [0, "
        + str(extent)
        + "))."
    )


# DMA quality-of-service priority range (nisa: lower value = higher priority).
_MIN_PRIORITY = 0
_MAX_PRIORITY = 3


def _validate_priority(priority):
    """DMA QoS priority must be None or an int in [0, 3]."""
    if priority is None:
        return
    assert isinstance(priority, int) and not isinstance(priority, bool), (
        "NDSlice DMA: priority= must be an int in ["
        + str(_MIN_PRIORITY)
        + ", "
        + str(_MAX_PRIORITY)
        + "], got "
        + str(priority)
    )
    assert _MIN_PRIORITY <= priority <= _MAX_PRIORITY, (
        "NDSlice DMA: priority="
        + str(priority)
        + " is out of range; valid QoS levels are ["
        + str(_MIN_PRIORITY)
        + ", "
        + str(_MAX_PRIORITY)
        + "] (lower = higher priority)."
    )


def _validate_dma_engine(engine, dge_mode, where):
    """The HWDGE descriptor-generation engine selector (nisa.dma_copy ``engine=``)
    is only valid with ``dge_mode=hwdge`` and must be sync/scalar -- mirror nisa's
    own contract (validate_dma_copy_engine) but raise at the neurotile API boundary
    with a clear message instead of deep in tracing. None / unknown / dma = the
    default (compiler/DMA-engine picks), always allowed."""
    if engine is None or engine == nisa.engine.unknown or engine == nisa.engine.dma:
        return
    assert dge_mode == nisa.dge_mode.hwdge, (
        where + ": engine= (HWDGE descriptor-gen engine) can only be set when "
        "dge_mode=nisa.dge_mode.hwdge, got dge_mode=" + str(dge_mode) + "."
    )
    assert engine in (nisa.engine.sync, nisa.engine.scalar), (
        where + ": engine= must be nisa.engine.sync or nisa.engine.scalar "
        "(the HWDGE descriptor-gen engines), got " + str(engine) + "."
    )


def _transpose_ranks_for_gathered_dims(n_gathered_dims):
    """Transpose ranks a gather view with ``n_gathered_dims`` non-trivial dims
    may use. A 3-dim gather is ambiguous: rank-3 ``(2,1,0)`` or the rank-4
    ``(3,1,2,0)`` reshape-trick form. Empty tuple if the dim count is unsupported."""
    if n_gathered_dims == 2:
        return (2,)
    if n_gathered_dims == 3:
        return (3, 4)
    return ()


def _validate_transpose_axes(transpose_axes, is_indirect, n_gathered_dims):
    """Validate transpose_axes= against the view: gather-only, a tuple of ints,
    rank valid for the gathered-dim count, and exactly nisa.dma_transpose's
    permutation for that rank. Only the unambiguous 2-D gather may omit it."""
    valid_ranks = _transpose_ranks_for_gathered_dims(n_gathered_dims)

    if transpose_axes is None:
        # Ambiguous when a gather view admits more than one transpose rank.
        assert not (is_indirect and len(valid_ranks) > 1), (
            "NDSlice.load(transpose=True): a gather transpose with "
            + str(n_gathered_dims)
            + " non-trivial dims is ambiguous (3-D (2,1,0) vs the 4-D (3,1,2,0) "
            "reshape-trick form), so transpose_axes= must be given explicitly. Pass "
            "transpose_axes=(2,1,0) or (3,1,2,0)."
        )
        return

    assert is_indirect, (
        "NDSlice.load(transpose=True, transpose_axes=...): transpose_axes is only "
        "supported on the indirect (gather) transpose (the view must carry an index); "
        "a static transpose is 2-D and needs no axes."
    )
    assert isinstance(transpose_axes, tuple), (
        "NDSlice.load(transpose=True): transpose_axes= must be a tuple of ints, got " + str(transpose_axes) + "."
    )
    for a in transpose_axes:
        assert isinstance(a, int) and not isinstance(a, bool), (
            "NDSlice.load(transpose=True): transpose_axes= must contain only ints, got " + str(transpose_axes)
        )
    axes_rank = len(transpose_axes)
    assert axes_rank in valid_ranks, (
        "NDSlice.load(transpose=True): transpose_axes="
        + str(transpose_axes)
        + " is a rank-"
        + str(axes_rank)
        + " permutation but the view has "
        + str(n_gathered_dims)
        + " non-trivial gathered dim(s) (valid transpose ranks: "
        + str(valid_ranks)
        + "). The axes rank must match the gather view."
    )
    expected = HBMLayout._default_transpose_axes(axes_rank)
    assert transpose_axes == expected, (
        "NDSlice.load(transpose=True): transpose_axes="
        + str(transpose_axes)
        + " is not the supported permutation for rank "
        + str(axes_rank)
        + " (expected "
        + str(expected)
        + "); nisa.dma_transpose supports only (1,0), (2,1,0), or (3,1,2,0)."
    )


def _validate_transpose_load(
    dge_mode,
    oob_mode,
    oob_value,
    priority,
    transpose_axes,
    is_indirect,
    n_gathered_dims=2,
    pattern_override=None,
    out_shape=None,
):
    """Validate DMA-control args for .load(transpose=True): reject combos
    nisa.dma_transpose can't honor. Hardware shape/dtype rules are left to the
    compiler."""
    _validate_transpose_axes(transpose_axes, is_indirect, n_gathered_dims)
    # The transpose path builds its own AP and SBUF shape, so pattern_override /
    # out_shape would be silently ignored -- reject rather than mislead.
    assert pattern_override is None, (
        "NDSlice.load(transpose=True): pattern_override= is not supported on the "
        "transpose path (the transpose builds its own access pattern); drop it or "
        "use a non-transpose load."
    )
    assert out_shape is None, (
        "NDSlice.load(transpose=True): out_shape= is not supported on the transpose "
        "path (the output shape is derived from the transpose); drop it."
    )
    if dge_mode is not None and not is_indirect:
        assert dge_mode.name in ("unknown", "hwdge"), (
            "NDSlice.load(transpose=True): dge_mode="
            + str(dge_mode.name)
            + " is not supported on the direct/tiled transpose path; only "
            "dge_mode.unknown or dge_mode.hwdge are valid (swdge is for the "
            "indirect/gather transpose)."
        )
    if oob_mode is not None and oob_mode.name == "skip":
        assert is_indirect, (
            "NDSlice.load(transpose=True): oob_mode.skip is only valid when "
            "the source uses indirect indexing; a static transpose-load has "
            "no out-of-bounds indices to skip."
        )
    if oob_value is not None:
        assert oob_mode is not None, (
            "NDSlice.load(transpose=True): oob_value= requires oob_mode= "
            "(typically nisa.oob_mode.skip) -- without it the value is never written."
        )
    _validate_priority(priority)


class NDSlice(nl.NKIObject):
    """A *view* over a tensor, decomposed into a grid of tiles.

    An ``NDSlice`` is the view object that every NeuroTile factory returns --
    ``tiles``, ``blocks``, ``alloc_tiles``, and ``alloc_blocks`` each
    produce one. It is the central value in a NeuroTile kernel: it is indexed to
    select tiles, iterated over, used to move data between HBM and SBUF, and reshaped
    to change its layout. Instances are obtained only from the factories; the class
    is not meant to be instantiated directly.

    A view holds no data of its own. Indexing or transforming a view produces a new
    view rather than modifying the original, and data moves only when ``load()`` or
    ``store()`` is called.

    The attributes below describe the shape and structure of a view. The operations
    it supports are grouped under :doc:`Data movement <data-movement>`,
    :doc:`Indexing & iteration <indexing-iteration>`, and
    :doc:`View transforms <view-transforms>`.

    .. warning::

       This API is experimental and may change in future releases.

    Attributes:
        shape (tuple[int, ...]): The number of tiles along each dimension that remain
            to be iterated, suitable for driving a loop (``for i in
            range(view.shape[0])``). For a freshly created tile view this is the full
            tile grid; indexing a dimension drops that dimension from ``shape``.
        element_shape (tuple[int, ...]): The number of individual elements (not tiles)
            along each dimension that the view covers. Selecting a whole tile leaves
            it unchanged -- ``view[i, j]`` still covers a full ``tile_size`` tile --
            whereas slicing to part of a tile (for example ``tile[:, 0:64]``) reduces
            it.
        tile_size (tuple[int, ...] | None): The shape of one tile -- the granularity
            each ISA instruction operates on after a ``load()``. ``None`` for an
            untiled view (e.g. the single-tile result of a transform).
        tile_shape (tuple[int, ...] | None): The number of tiles along each dimension
            of the whole view: ``element_shape`` divided by ``tile_size`` and rounded
            up. Unlike ``shape``, it always reflects the complete tile grid and does
            not change as the view is indexed. ``None`` for an untiled view.
        index_stride_elements (tuple[int, ...]): For each current public index dim,
            the number of source elements advanced by one logical index step. Use it
            to maintain runtime counters passed through ``nt.element_offset(...)``.
        block_size (tuple[int, ...] | None): The number of tiles per block along each
            dimension, for a view created by ``blocks`` / ``alloc_blocks``. ``None``
            when the view has no block grouping.
        block_shape (tuple[int, ...] | None): The number of blocks along each
            dimension. ``None`` when the view has no block grouping.
        is_tiled (bool): ``True`` if the view was created with a ``tile_size`` (every
            view except an untiled one).
        is_blocked (bool): ``True`` if the view groups its tiles into blocks.
        ndim (int): The number of dimensions of the view.
        is_remainder (bool): ``True`` if the region this view covers ends on a partial
            tile -- that is, the extent is not an exact multiple of ``tile_size``, so
            the last tile along some dimension is smaller than a full tile (the view's
            origin offset is taken into account, so an offset that pushes the trailing
            tile past the source also sets it) -- or if the view carries a runtime
            (gather / scatter or runtime-scalar) index, which is conservatively treated
            as a remainder. A broadcast (stride-0) dimension never raises the flag on
            its own.
        dtype: The element data type.
        buffer_type: The memory region the view lives in: ``nl.shared_hbm``,
            ``nl.private_hbm``, ``nl.sbuf``, or ``nl.psum``.
        data: The view's region as the underlying NKI tensor -- the single operand
            handle to pass to ``nisa.*`` compute ops, ``.store()``, and manual DMAs.
            Carries the view's offset / strides; same role for every buffer type.

    """

    def __init__(self, grid, layout, dma_override=None, load_dst=None, load_pattern_override=None, load_out_shape=None):
        # _grid / _layout are internal: NDSlice = Grid + Layout is an
        # implementation detail, not part of the public view contract.
        self._grid = grid
        self._layout = layout

        self.shape = grid.shape
        # User-facing data extent: grid.remaining clamped to what's reachable
        # (it over-reports on a partial trailing tile/block). See below.
        self.element_shape = NDSlice._reachable_element_shape(grid, layout)
        self.tile_size = grid.tile_size
        self.tile_shape = grid.tile_shape if grid.is_tiled() else None
        self.block_size = grid.block_size
        self.block_shape = grid.block_shape
        self.index_stride_elements = grid.index_stride_elements()
        self.is_tiled = grid.is_tiled()
        self.is_blocked = grid.is_blocked()
        self.ndim = grid.ndim
        # Layout owns the offset, so the partial-trailing-tile check is
        # Layout's responsibility (it ORs Grid's flag with offset+axis arithmetic).
        self.is_remainder = layout.is_remainder(grid)

        # Forwarded from Layout
        self.dtype = layout.dtype
        self.buffer_type = layout.buffer_type

        # Non-public: these mean different things per buffer type (element offset
        # vs. tile index, element vs. tile-grid strides, whole-buffer vs.
        # active-bank handle), so they are NOT part of the public NDSlice contract.
        # Kernels operate via .data (the operand/AP) and the indexing/transform API.
        self._source = layout.source
        self._offset = layout.offset
        self._strides = layout.strides

        self.data = layout.ap(self._grid)

        # DMA override for non-standard load/store paths (immutable after construction).
        # ("partition_fold", recipe) -- multi-DMA for P-dim folds
        # ("ap_override", pattern) -- custom HBM AP for free-dim folds
        self._dma_override = dma_override

        # Stream defaults -- populated by BlockStream.tolist() to route DMAs
        # into rotating buffer slots without the caller having to thread
        # dst=/pattern_override=/out_shape= through every load() call. Each
        # is used only when the corresponding kwarg on load() is None.
        self._load_dst = load_dst
        self._load_pattern_override = load_pattern_override
        self._load_out_shape = load_out_shape

    @staticmethod
    def _reachable_element_shape(grid, layout):
        """Per-dim source elements this view actually addresses (data extent).

        Clamps ``grid.remaining`` (the walked extent, which over-reports on a
        partial trailing tile/block) to ``layout.dim_addressable``, so the
        reported shape and the DMA'd data agree across all layouts.
        """
        reachable = []
        for d in range(grid.ndim):
            reachable.append(min(grid.remaining[d], layout.dim_addressable(d, grid)))
        return tuple(reachable)

    # ================================================================
    # Indexing
    # ================================================================

    def _is_sbuf_element_level(self):
        """True if SBUF view at element level -- enables sub_index path.

        Checks: Layout is SBUFLayout AND all dims at element level (step==1)
        AND remaining fits in a single tile (no tile navigation needed).
        """
        if isinstance(self._layout, SBUFLayout):
            grid = self._grid
            for d in range(grid.ndim):
                if not grid.is_element(d):
                    return False
            if grid.tile_size is not None:
                for d in range(grid.ndim):
                    if grid.remaining[d] > grid.tile_size[d]:
                        return False
            return True
        return False

    def __getitem__(self, key):
        """__getitem__(key) -> NDSlice

        Select part of a view by indexing it, following NumPy's indexing conventions.
        This is how a kernel picks the tile -- or row, column, or sub-grid of tiles -- it
        operates on. No data is moved: indexing returns a new view, and the DMA happens
        later when ``load()`` or ``store()`` is called on the result.

        Indexing is the most-used NeuroTile operation and accepts a range of keys, from a
        plain integer to a runtime index tensor for gather and scatter. The sections
        below progress from the simplest case.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            key (int | slice | tuple | NDSlice | nl.ndarray): What to select along one or
                more dimensions. An ``int`` or ``slice`` selects at tile granularity (or
                block granularity for a block view); a tuple applies one key per leading
                dimension, left to right; a loaded index view (or raw SBUF ``nl.ndarray``)
                performs a runtime gather or scatter. See the sections below.

        Returns:
            NDSlice: A new view over the selected region -- a single tile, a row or column
            of tiles, a rectangular sub-grid, or a gathered set of rows. Calling
            ``load()`` / ``store()`` on it moves the data. Indexing a block of a block view
            (``blocks[bi, bj]``) returns the tile grid inside that block, addressed by
            interior tile coordinate (``block[ti, tj]``).

        Raises ``AssertionError`` in any of these cases:

        - more keys are given than the view has dimensions.
        - a compile-time integer index is out of range (valid range ``[-extent, extent)``
          on that dim).
        - a compile-time slice has step ``< 1``, is out of range (outside ``[0, extent]``),
          or is empty (``stop <= start``).
        - a slice is applied to a dimension whose current extent is unknown.
        - a bool key (``True`` / ``False``) is given -- it must be written as an explicit
          ``int``.
        - the key is an unsupported type -- a ``list``, ``dict``, ``str``, or ``float``
          (Boolean masks and lists of indices are rejected).

        Runtime keys (a runtime scalar / loop variable, or a loaded index view) and runtime
        slice bounds are not bounds-checked at trace time; NKI bounds-checks them at run
        time. An ellipsis (``...``) or ``None`` key is not supported and is not validated --
        it is silently mishandled rather than rejected, so it must not be passed.

        Basic indexing
        --------------
        A tile grid is indexed by tile coordinate. Each key selects along one dimension,
        in order, as in NumPy:

        .. code-block:: python

            tiles = nt.tiles(src, tile_size=(128, 512))   # a 2-D grid of tiles

            one  = tiles[i, j]      # the single tile at row i, column j
            row  = tiles[i]         # the whole i-th row of tiles (same as tiles[i, :])
            col  = tiles[:, j]      # the whole j-th column of tiles
            sub  = tiles[1:3, 0:2]  # a 2x2 rectangular sub-grid of tiles
            last = tiles[-1, -1]    # negative indices count from the end

        Each result is itself a view: ``tiles[i, j].load()`` brings one tile into SBUF,
        while ``tiles[i].load()`` brings a whole row in as one coalesced DMA. The ``##``
        cells below show what each key selects from a ``4 x 4`` grid:

        .. code-block:: text

            tiles[1, 2]        tiles[0]            tiles[:, 3]
            (one tile)         (a row)             (a column)
            ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┐
            │  │  │  │  │      │##│##│##│##│       │  │  │  │##│
            ├──┼──┼──┼──┤      ├──┼──┼──┼──┤       ├──┼──┼──┼──┤
            │  │  │##│  │      │  │  │  │  │       │  │  │  │##│
            ├──┼──┼──┼──┤      ├──┼──┼──┼──┤       ├──┼──┼──┼──┤
            │  │  │  │  │      │  │  │  │  │       │  │  │  │##│
            └──┴──┴──┴──┘      └──┴──┴──┴──┘       └──┴──┴──┴──┘

        Slicing for sharding
        --------------------
        A strided slice (``start:stop:step``) selects every ``step``-th tile. Combined
        with a per-core rank this is how a kernel shards work across cores -- the
        sharding helpers (``block_range``, ``interleaved_range``, ...) return exactly
        these slices:

        .. code-block:: python

            own = nt.interleaved_range(nl.program_id(0), nl.num_programs(0), tiles.shape[0])
            local = tiles[own, :]   # this core's tiles only

        Gather and scatter (runtime indices)
        ------------------------------------
        To select rows whose positions are known only at runtime, a **loaded index view**
        is passed as the key in place of an integer. Each entry of the index selects one
        source row, and the hardware fetches them all in a single indirect DMA -- a
        *gather* on ``load()`` and a *scatter* on ``store()``.

        The index is an SBUF tile of shape ``(K, 1)``: ``K`` row positions, one per
        partition. It is placed in the position of the dimension being gathered along:

        .. code-block:: python

            # Gather K rows of `data` named by `indices` (shape (K, 1)): out[i] = data[indices[i]]
            data_iter = nt.tiles(data, tile_size=(N, T_D))     # full N rows, T_D-wide columns
            idx_iter  = nt.tiles(indices, tile_size=(K, 1))

            idx = idx_iter[0, 0].load()                        # the index tile, in SBUF
            gathered = data_iter[idx, j].load()                # gather -> (K, T_D) in SBUF

            # Scatter is the mirror, on store():
            out_iter[idx, j].store(values.data)                # out[indices[i]] = values[i]

        The index names which source row lands in each output row:

        .. code-block:: text

            index (SBUF)      data (HBM)         gathered (SBUF)
            ┌─────────┐       ┌──────────┐       ┌──────────┐
            │ idx[0]=2│       │ row 0    │       │ row 2    │  <- data[2]
            │ idx[1]=0│  ──>  │ row 1    │  ──>  │ row 0    │  <- data[0]
            │ idx[2]=3│       │ row 2    │       │ row 3    │  <- data[3]
            └─────────┘       │ row 3    │       └──────────┘
                              │ ...      │
                              └──────────┘

        Selecting one dynamic row or column (scalar index)
        --------------------------------------------------
        A loaded index of shape ``(1, 1)`` holds a single runtime position and shifts the
        whole window by it. Its position in the key chooses which dimension it shifts:

        .. code-block:: python

            pos = idx_iter[0, 0].load()          # one runtime position, shape (1, 1)
            row = data_iter[pos, 0].load()       # dynamic row   -> (1, D)
            col = data_iter[0, pos].load()       # dynamic column -> (N, 1)

        A plain runtime scalar key is a logical coordinate at the current view level. On
        tile/block axes, NeuroTile may scale an SBUF scalar internally to produce the
        source-element ``scalar_offset``. If the runtime value is already an element offset
        (for example a counter incremented by ``view.index_stride_elements[dim]``), wrap it
        with ``nt.element_offset(offset)`` so NeuroTile passes it through without scaling:

        .. code-block:: python

            tile = data_iter[tile_idx, j].load()                 # logical tile coordinate
            tile = data_iter[nt.element_offset(row_offset), j].load()  # element offset

        Mixing fixed and dynamic indices
        --------------------------------
        Static indices (an ``int`` or loop variable) and one runtime index combine in a
        single key: the static parts select a fixed base, and the runtime part selects
        within it. A KV-cache read is the canonical example:

        .. code-block:: python

            kv = nt.tiles(kv_cache, tile_size=(1, D))    # kv_cache: [B, S, D]
            seq = seq_iter[0, 0].load()                  # runtime sequence position (1, 1)
            vec = kv[batch_id, seq].load()               # batch_id fixed, seq dynamic -> (1, D)

        Granularity: tile, block, and element level
        --------------------------------------------
        Indexing an HBM, tile, or block view operates at tile (or block) granularity. On an
        SBUF view, indexing entirely within one tile (every key at element granularity)
        slices the SBUF array directly and returns an element-level SBUF view -- this is how
        a loaded tile is sub-addressed. One special case: an integer index on the partition
        (P) dimension narrows it to a single row rather than removing the dimension, because
        SBUF data always keeps its partition dimension (the result stays 2-D).

        Indexing a batch dimension
        --------------------------
        When a view's only remaining iteration dimension is a batch dimension and the
        trailing tile grid is a single tile (``(1, ..., 1)`` -- for example
        ``nt.tiles(src, tile_size=(P, F))`` on a ``(B, P, F)`` source), indexing the batch
        dimension lands directly on that single tile at element level. The result is ready
        to ``load()`` as-is; do not subscript it further:

        .. code-block:: python

            src_tiles = nt.tiles(src, tile_size=(P, F))   # src.shape == (B, P, F)
            slab = src_tiles[b]                            # already a single tile
            tile = slab.load()                             # not slab[0, 0].load()

        Notes:
            A key is expected to carry **at most one** runtime (gather/scatter or scalar)
            index, with a gather index (``(K, 1)``) in the first position; these are usage
            preconditions for a correct result, not validated constraints, so a second
            runtime index or a misplaced gather is silently mishandled rather than
            rejected. The index tensor must reside in SBUF (a loaded view, or a raw
            ``nl.ndarray``) and the data tensor must reside in HBM.

        .. warning::

            A gather index addresses rows of the **underlying tensor's first dimension**,
            not of any reshaped logical view. To gather along a flattened axis (for
            example, a ``[B, S, D]`` cache flattened to ``[B*S, D]``), reshape the **raw**
            tensor with ``tensor.reshape(...)`` before calling ``nt.tiles``. A metadata
            view transform such as ``flatten_dims`` does not change which axis the index
            walks, so the gather would read the wrong rows.

        See Also:
            load / store: move the selected (or gathered) data between HBM and SBUF.
            slice: narrow a view to part of a single tile, at element granularity.
            block_range / interleaved_range: produce the slices used for sharding.

        """
        if self._is_sbuf_element_level():
            return self._sub_index(key)

        if isinstance(key, tuple):
            return self._index_multi(key)
        return self._index_multi((key,))

    def _sub_index(self, key):
        """SBUF sub-tile indexing -- bypasses Grid entirely."""
        new_layout, result_shape = self._layout.sub_index(key, self.element_shape)

        result_shape = tuple(result_shape)
        # Element-level grid: one elem axis per dim, count = result_shape[d].
        minimal_grid = Grid.from_shape(
            element_shape=result_shape,
            tile_size=result_shape,
        )

        return NDSlice(minimal_grid, new_layout)

    def _index_multi(self, keys):
        """Process multi-dimensional index keys left-to-right.

        Keys map to source dims 0..len(keys)-1 (numpy convention).
        Partial indexing (``len(keys) < ndim``) leaves trailing dims at
        their current level.

        Per-key dispatch:
          - int            : grid.consume + layout.advance(stride, k)
          - slice [a:b]    : grid.narrow + layout.advance(stride, a)
          - slice [a:b:s]  : grid.split (peer-walk) + grid.consume +
                             grid.narrow + layout.advance(stride, a)
          - SBUF vector    : grid.consume + layout.set_indirect(VECTOR, ...)
          - runtime scalar : grid.consume + layout.set_indirect(SCALAR, ...)

        Sub-tile guard: when an int key lands on a dim whose only
        remaining axis is a sub-tile leaf (PART / consumed ELEM) AND
        the cursor has already advanced past this dim, the key is
        deflected to the cursor's dim. This handles ``row[j]`` after
        ``row = v[i, :].load()`` -- the user's single key targets the
        surviving iteration dim, not the consumed PARTITION leaf.

        After all keys, the cursor advances past every consumed dim and
        any dim whose surviving outer is no longer iteration-level (e.g.
        a slice that landed exactly on a tile-leaf). If the cursor would
        run off the end and any other dim still has an iteration-level
        outer, it wraps to expose the next-level grid (e.g. a consumed
        block reveals its interior tile grid).

        cleanup() drops fully-consumed dims and auto-pops trivial tile
        levels after batch collapse; ``drop_dim`` adjusts the cursor for
        dropped dims so no remapping is needed here.
        """
        grid = self._grid
        layout = self._layout

        assert len(keys) <= grid.ndim, (
            f"view[...]: too many keys ({len(keys)}) for {grid.ndim}-D "
            "view. Drop trailing keys or index through a higher-rank view."
        )

        # The new cursor must land past every consumed key (consume
        # removes the iter-level axis on its dim, so the dim is no
        # longer at the same iteration level).
        consumed_dims = []

        for pos in range(len(keys)):
            dim = pos
            k = keys[pos]

            # Partial-key deflect: when a single int key lands on a
            # non-batch dim the cursor has already advanced past,
            # redirect to the cursor's dim. The cursor identifies the
            # next iteration level the user wants to step through; dims
            # before it (and past the batch zone) are "inside" a
            # parent's iteration step (sub-tile leaves or per-block
            # tile grids inside a consumed block) and should not be
            # re-indexed implicitly. Batch dims (``dim < n_batch_dims``)
            # are always user-addressable -- ``v[batch_idx]`` on a
            # batched view consumes the slab regardless of cursor.
            if (
                len(keys) == 1
                and isinstance(k, (int, ElementOffset))
                and grid.n_batch_dims <= dim < grid.cursor
                and grid.cursor < grid.ndim
            ):
                dim = grid.cursor

            validate_index_key(
                k,
                dim,
                grid.current_count(dim),
                grid=grid,
                context="view[...]",
            )

            advance_step = layout.advance_step(grid, dim)

            if isinstance(k, int):
                normalized_k = k if k >= 0 else grid.current_count(dim) + k
                # Sub-tile P-narrow: an int on a dim whose only remaining
                # axis is the partition leaf shrinks PART to count=1
                # rather than dropping it. SBUF allocation must stay 2-D
                # (partition row axis required), so we keep the axis with
                # count=1 instead of consuming it. The layout still
                # advances by k partition-rows.
                if _is_partition_leaf_only(grid, dim):
                    grid = grid.narrow(dim, 1)
                    layout = layout.advance(dim, normalized_k, advance_step)
                else:
                    grid = grid.consume(dim)
                    layout = layout.advance(dim, normalized_k, advance_step)
                    # Elements consumed on this dim = index * tile extent. The P dim
                    # folds its tiles into F, so layout.dim_offset_elements(P) is always
                    # 0 -- use the P-tile index * tile_p directly so a trailing partial
                    # P-tile (element_shape[P] not a multiple of tile_p) is narrowed and
                    # flagged is_remainder, just like a trailing F-tile.
                    leaf = grid._elem_leaf_on_dim(dim)
                    is_folded_partition = (
                        leaf is not None and leaf.label == AxisLabel.PARTITION and hasattr(layout, "alloc_tile_size")
                    )
                    if is_folded_partition:
                        p_consumed = normalized_k * layout.alloc_tile_size[dim]
                    else:
                        p_consumed = layout.dim_offset_elements(dim, grid.element_shape)
                    grid = grid.truncate_to_source(dim, p_consumed)
                consumed_dims.append(dim)

            elif isinstance(k, ElementOffset):
                assert hasattr(layout, "advance_by_element_offset"), (
                    "nt.element_offset(...) indexing is only supported on HBM-backed views before load/store."
                )
                element_offset_value = _element_offset_value(k.value)
                _assert_static_element_offset_in_bounds(element_offset_value, grid, layout, dim)
                grid = grid.consume(dim)
                layout = layout.advance_by_element_offset(dim, element_offset_value)
                consumed_dims.append(dim)

            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else grid.current_count(dim)
                slice_step = k.step if k.step is not None else 1

                if slice_step != 1:
                    # Stepped slice: factor a peer-walk axis (split + pop peer).
                    # peer is outer with step = self.step (one item per peer slot);
                    # owned is inner with count = self.count // step,
                    # step = self.step * step (skips past peers).
                    grid = grid.split_peers(
                        dim,
                        num_peers=slice_step,
                        peer_label=AxisLabel.SHARD,
                        owned_label=AxisLabel.TILE,
                    )
                    grid = grid.consume(dim)  # drop peer axis -- this rank's slot
                    # ceil((stop - start) / step) -- the count of owned items.
                    count = (stop - start + slice_step - 1) // slice_step
                else:
                    count = stop - start

                grid = grid.narrow(dim, count)
                layout = layout.advance(dim, start, advance_step)
                grid = grid.truncate_to_source(dim, layout.dim_offset_elements(dim, grid.element_shape))

            else:
                # Runtime scalar OR vector tensor. Vector inputs may arrive
                # as a loaded NDSlice (over SBUF) OR a raw nl.ndarray with
                # shape[0] > 1 -- both route to vector_offset.
                if isinstance(k, NDSlice) and isinstance(k._layout, SBUFLayout):
                    vector_count = k.element_shape[0]
                    is_vector = vector_count > 1
                    index_data = k._layout.source
                elif hasattr(k, "shape") and len(k.shape) >= 1:
                    vector_count = k.shape[0]
                    is_vector = vector_count > 1
                    index_data = k
                else:
                    vector_count = 1
                    is_vector = False
                    index_data = k

                if is_vector:
                    # Vector gather: pop the outer axis on dim, then narrow
                    # the next axis (the partition/leaf walk) to the vector
                    # count -- this rank visits exactly ``vector_count``
                    # gathered positions on this dim.
                    grid = grid.consume(dim)
                    if grid.outer_axis(dim) is not None:
                        grid = grid.narrow(dim, vector_count)
                    layout = layout.set_indirect(IndirectKind.VECTOR, index_data, dim)
                else:
                    grid = grid.consume(dim)
                    layout = layout.advance(dim, index_data, advance_step)
                consumed_dims.append(dim)

        grid, dropped_dims = grid.cleanup()
        if hasattr(layout, "drop_dims"):
            layout = layout.drop_dims(dropped_dims)
        consumed_dims = _shift_dims_after_drop(consumed_dims, dropped_dims)

        grid = grid.with_cursor_past_consumed(consumed_dims)
        return NDSlice(grid, layout)

    # ================================================================
    # Iteration
    # ================================================================

    def tolist(self, dim: Optional[int] = None) -> list:
        """tolist(dim=None) -> list

        Return the view's tiles (or rows, columns, or blocks) as a plain Python list, for
        iteration with a ``for`` loop that does not track indices.

        This is the form to use when the loop body does the same thing to every tile and
        does not need the tile's position. A bare ``for x in view`` is not supported --
        the NKI tracer cannot iterate a view directly -- so ``tolist()`` provides the
        iterable.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int | None): Which dimension to iterate over. With the default ``None``,
                iteration follows the view's natural outer dimension: if leading batch dims
                remain, the leftmost is walked first and each sub-view drops that dimension;
                once batch dims are gone it walks the outer tile dimension. An integer
                iterates a specific dimension instead -- for example, ``dim=1`` walks columns
                rather than rows -- without dropping batch dims and without advancing the
                natural iteration position (each sub-view keeps every other dimension).

        Returns:
            list[NDSlice]: one view per position along the chosen dimension (the count
            equals the current item count along that dimension); each element is itself a
            view that can be indexed, loaded, or iterated further. With ``dim=None``, a
            0-dim view (everything indexed away) returns an empty list, and a view with no
            iteration dimension left returns a single-element list holding the view itself.

        Raises ``AssertionError`` if ``dim`` is neither ``None`` nor an integer (a bool is
        rejected), or is outside ``[0, ndim)``.

        Example:
            .. code-block:: python

                for tile in tiles.tolist():            # walk every tile, no index needed
                    tile.store(process(tile.load()).data)

                for col in tiles.tolist(dim=1):        # iterate columns instead of rows
                    for tile in col.tolist():
                        ...

        Notes:
            ``len(view)`` is the product of every per-dimension item count (the total number
            of tiles / positions across all dims), **not** the number of sub-views
            ``tolist()`` yields along the single iteration dimension. Use
            ``len(view.tolist())`` for the iteration-step count.

        See Also:
            whole_tiles / remainder_tiles: split iteration by remainder status.

        """
        if dim is None:
            return self._tolist_cursor()
        assert isinstance(dim, int) and not isinstance(dim, bool), (
            "NDSlice.tolist: dim= must be int or None, got " + str(dim)
        )
        assert 0 <= dim < self.ndim, (
            "NDSlice.tolist: dim=" + str(dim) + " is out of range for ndim=" + str(self.ndim) + "."
        )
        return self._tolist_along_dim(dim)

    def _tolist_cursor(self):
        """Iterate the next active dim -- batch dims first, then cursor.

        Unconsumed batch dims always come first (each iteration drops the
        dim entirely so a child sees one less batch dim). Once batch dims
        are exhausted, the iteration follows the grid cursor through the
        tile dims, advancing the cursor on each child so the next
        ``tolist()`` walks the next tile dim.
        """
        if self.ndim == 0:
            return []

        # Walk the leftmost still-present batch dim first.
        if self._grid.n_batch_dims > 0:
            return self._iterate_dim(dim=0, drop=True)

        cursor_dim = self._grid.cursor
        if cursor_dim >= self._grid.ndim:
            return [self]
        return self._iterate_dim(dim=cursor_dim, drop=False, advance_cursor=True)

    def _iterate_dim(self, dim, drop=False, advance_cursor=False):
        """Materialize one child per outer-axis step on ``dim``.

        Args:
            dim: dim to iterate.
            drop: drop ``dim`` from the child Grid + Layout (batch consumption).
            advance_cursor: bump child cursor to ``dim + 1`` (tile dim walk).
        """
        step = self._layout.advance_step(self._grid, dim)
        count = self._grid.current_count(dim)
        result = []
        for i in range(count):
            child_grid = self._grid.narrow(dim, 1)
            child_layout = self._layout.advance(dim, i, step)
            if drop:
                child_grid = child_grid.drop_dim(dim)
                child_layout = child_layout.drop_dim(dim)
            elif advance_cursor:
                child_grid = child_grid.with_cursor(dim + 1)
            result.append(NDSlice(child_grid, child_layout))
        return result

    def _tolist_along_dim(self, dim):
        """Iterate along an explicit dim -- cursor untouched, no batch-drop."""
        return self._iterate_dim(dim)

    def _enumerate(self, start=0, mode=None):
        """Return list of (index, sub_view) pairs -- always local indices."""
        items = self.tolist()
        result = []
        for i in range(len(items)):
            result.append((start + i, items[i]))
        return result

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        total = 1
        for s in self.shape:
            total = total * s
        return total

    # ================================================================
    # Dimensional iteration
    # ================================================================

    def stream(
        self,
        dim: Optional[int] = None,
        buffer_count: int = 2,
        dtype=None,
        transpose: bool = False,
        transpose_axes: Optional[tuple] = None,
        pattern_override: Optional[list] = None,
        out_shape: Optional[tuple[int, ...]] = None,
    ):
        """stream(dim=None, buffer_count=2, dtype=None, transpose=False, transpose_axes=None, pattern_override=None, out_shape=None) -> BlockStream

        Create a rotating-buffer stream along ``dim`` that overlaps each step's DMA with
        the previous step's compute.

        A stream is the tool for *flowing* operands -- the K dimension of a matmul, or
        weight blocks fed through compute -- where double-buffering hides DMA latency. A
        *stationary* operand is instead loaded once with ``load()``. Streams apply to HBM
        views only; the rotating buffers themselves live in SBUF.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int | None): The dimension to iterate over. The default ``None`` uses the
                view's outermost dimension that is still available to iterate (dimension 0
                for a freshly created view; dimension 1 for a row obtained by
                ``parent[i, :]``). An integer streams a specific dimension instead.
            buffer_count (int): The number of rotating buffers: 2 for double-buffering, 3
                for triple-buffering. Defaults to 2.
            dtype: The dtype for the rotating buffers. Defaults to the source dtype. A
                ``stream.load(k, dtype=...)`` that does not match this allocates a fresh
                per-call buffer instead of using a rotating one, so that step does not
                overlap.
            transpose (bool): When ``True``, the stream is a transpose stream: each buffer
                is sized and typed for the transposed output, and every ``load()`` must
                repeat the same ``transpose`` / ``transpose_axes``.
            transpose_axes (tuple[int] | None): The gather-transpose permutation; requires
                ``transpose=True``. See ``NDSlice.load``.
            pattern_override (list[[int, int]] | None): A custom HBM access pattern applied
                per step (advanced). Requires ``out_shape``.
            out_shape (tuple[int, int] | None): The per-buffer SBUF shape. Required when
                ``pattern_override`` is set.

        Returns:
            BlockStream: The rotating-buffer pipeline, driven by ``stream.load(k)`` /
            ``stream[k]`` / ``stream.store(k)`` or by iterating it.

        Raises ``AssertionError`` in any of these cases:

        - called on an SBUF view.
        - the view still has leading batch dimensions that have not been indexed down to
          a single tile.
        - ``dim`` is not an int or is out of range.
        - ``dim`` is omitted and the view has no iteration dimension left (index a parent
          view or pass ``dim=`` explicitly).
        - ``buffer_count`` is not a positive int.
        - ``transpose_axes`` is set without ``transpose=True``.
        - ``pattern_override`` is set without ``out_shape``.

        Example:
            .. code-block:: python

                stream = tiles[:, 0].stream(buffer_count=2)
                for k in nl.affine_range(K):
                    tile = stream.load(k)          # DMA into slot k%2, overlaps compute
                    nisa.nc_matmul(acc, lhs.data, tile.data)

        With ``buffer_count=2``, two SBUF slots are reused across steps so that the load
        of step ``k`` overlaps the compute of step ``k-1``::

                     slot 0      slot 1
                    ┌───────┐   ┌───────┐
            k=0     │ tile0 │   │       │   load 0
                    ├───────┤   ├───────┤
            k=1     │ tile0 │   │ tile1 │   compute 0 || load 1
                    ├───────┤   ├───────┤
            k=2     │ tile2 │   │ tile1 │   compute 1 || load 2   (slot 0 reused)
                    ├───────┤   ├───────┤
            k=3     │ tile2 │   │ tile3 │   compute 2 || load 3   (slot 1 reused)
                    └───────┘   └───────┘

        **SBUF cost.** Each buffer holds exactly one step along ``dim`` -- the same chunk
        that ``view[k]`` would yield -- so the total SBUF footprint is ``buffer_count``
        times that chunk::

            source view                       one buffer holds
            --------------------------------  --------------------------------
            tile column/row (tiles[:, m])     one tile
            multi-tile slice along the dim    that row/column of tiles, packed
            block column (blocks[:, b])       one block (block_size tiles)

        ``buffer_count=2`` is the default. A value of 3 helps only when the consumer is
        small enough that two stages cannot fully hide the DMA latency.

        **Load-type lock.** A stream's ``(transpose, transpose_axes)`` is fixed once --
        at construction (``stream(transpose=...)``) or by the first ``load()`` -- and it
        sizes every buffer, so all later loads must match. A given stream therefore
        serves one load type. Rotating gather-transpose streams are not supported; a
        non-streamed ``view[idx].load(transpose=True)`` is used instead.

        See Also:
            load: one-shot load for stationary operands.
            BlockStream: the returned pipeline object.

        """
        assert isinstance(self._layout, HBMLayout), (
            "NDSlice.stream: only valid on HBM views (the rotating "
            "buffers live in SBUF; the view is the HBM source). For an "
            "already-loaded SBUF view, iterate it with .tolist() or "
            "indexing instead."
        )
        assert self._grid.n_batch_dims == 0, (
            "NDSlice.stream: requires all batch dims to be consumed "
            "first (got " + str(self._grid.n_batch_dims) + " unconsumed). "
            "Index or iterate the batch dims before calling .stream()."
        )
        if dim is None:
            dim = self._grid.cursor
            assert dim < self.ndim, (
                "NDSlice.stream: cursor (=" + str(dim) + ") is past the "
                "view's ndim=" + str(self.ndim) + ". The view has no "
                "iteration dim left; either pass dim= explicitly or "
                "stream a parent view that still has iteration axes."
            )
        else:
            assert isinstance(dim, int) and not isinstance(dim, bool), "NDSlice.stream: dim= must be int, got " + str(
                dim
            )
            assert 0 <= dim < self.ndim, (
                "NDSlice.stream: dim=" + str(dim) + " is out of range for ndim=" + str(self.ndim) + "."
            )
        assert isinstance(buffer_count, int) and not isinstance(buffer_count, bool), (
            "NDSlice.stream: buffer_count= must be int, got " + str(buffer_count)
        )
        assert buffer_count >= 1, "NDSlice.stream: buffer_count= must be >= 1, got " + str(buffer_count)
        assert transpose or transpose_axes is None, (
            "NDSlice.stream: transpose_axes= requires transpose=True (it is the "
            "gather-transpose permutation; meaningless on a non-transpose stream)."
        )
        if pattern_override is not None:
            assert out_shape is not None, (
                "NDSlice.stream: pattern_override= requires out_shape="
                "(<P>, <F>) so the SBUF rotating buffers can be allocated."
            )
        return BlockStream(
            view=self,
            dim=dim,
            buffer_count=buffer_count,
            dtype=dtype,
            transpose=transpose,
            transpose_axes=transpose_axes,
            pattern_override=pattern_override,
            out_shape=out_shape,
        )

    # ================================================================
    # DMA
    # ================================================================

    def load(
        self,
        dtype=None,
        transpose: bool = False,
        transpose_axes: Optional[tuple] = None,
        dge_mode=None,
        oob_mode=None,
        oob_value: Optional[float] = None,
        priority: Optional[int] = None,
        dst: Optional[nl.ndarray] = None,
        pattern_override: Optional[list] = None,
        out_shape: Optional[tuple[int, ...]] = None,
        engine=None,
    ) -> "NDSlice":
        """load(dtype=None, transpose=False, transpose_axes=None, dge_mode=None, oob_mode=None, oob_value=None, priority=None, dst=None, pattern_override=None, out_shape=None, engine=None) -> NDSlice

        Transfer this view's region from HBM to SBUF and return a view over the SBUF
        result.

        The transfer is a single ``nisa.dma_copy`` (the single-DMA contract), so a
        coalesced slice such as ``tiles[:, m]`` becomes one strided DMA rather than a
        per-tile loop.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dtype: Allocate the SBUF destination at this dtype; the DMA hardware auto-casts
                on a mismatch with the HBM source (no separate cast op). Defaults to the
                source dtype.
            transpose (bool): Take the DMA-transpose path (requires a 2-byte dtype). See
                Transpose below.
            transpose_axes (tuple[int] | None): Gather-transpose permutation -- one of
                ``(1, 0)`` / ``(2, 1, 0)`` / ``(3, 1, 2, 0)``. Indirect transpose only;
                optional on a 2-D gather, required on a >2-D gather.
            dge_mode (nisa.dge_mode.* | None): DMA-generation mode hint. On a static
                transpose only ``unknown`` / ``hwdge`` are valid (``swdge`` is gather-only).
            oob_mode (nisa.oob_mode.skip | None): Suppress out-of-bounds DMA faults at
                boundaries; pair with ``oob_value`` to pre-fill. On a transpose, valid only
                on the indirect (gather) path.
            oob_value (float | None): Pre-fill SBUF with this value before the DMA.
                Requires ``oob_mode``.
            priority (int | None): DMA QoS level in ``[0, 3]`` (lower = higher priority);
                omitted when None.
            dst (nl.ndarray | .data view | None): Pre-allocated SBUF destination. The
                library allocates one (at ``dtype``) when omitted. On a transpose its shape
                must equal the transposed output shape.
            pattern_override (list[[int, int]] | None): Custom HBM access pattern replacing
                the auto-generated one. Requires ``out_shape`` (or ``dst``). Not supported
                with ``transpose=True``.
            out_shape (tuple[int, int] | None): SBUF allocation shape ``(P, F)``. Required
                with ``pattern_override``. Not supported with ``transpose=True``.
            engine (nisa.engine.* | None): The HWDGE descriptor-generation engine for the
                transfer. ``None`` (default), ``nisa.engine.unknown``, and
                ``nisa.engine.dma`` let the compiler / DMA engine pick, regardless of
                ``dge_mode``. Only ``nisa.engine.sync`` or ``nisa.engine.scalar`` pin a
                specific engine, and only with ``dge_mode=nisa.dge_mode.hwdge``. Not
                supported on the transpose path.

        Returns:
            NDSlice: A view over the freshly loaded (or caller-provided) SBUF buffer --
            see ``NDSlice`` for the view's attributes. ``.data`` is the underlying
            ``nl.ndarray``. A coalesced whole-row / whole-column selection (``tiles[i]`` or
            ``tiles[:, m]``) packs the selected tiles contiguously along the SBUF free axis
            in one DMA, and the result is indexed in SBUF with no further DMA.

        Raises ``AssertionError`` in any of these cases:

        - called on an untiled view (one with no ``tile_size``).
        - called on an SBUF view (``load()`` reads from HBM).
        - the view still has leading batch dimensions that have not been indexed down to
          a single tile.
        - ``oob_value`` is given without ``oob_mode``.
        - ``pattern_override`` is given without ``out_shape`` / ``dst``.
        - ``priority`` is outside ``[0, 3]``.
        - ``engine`` is set without ``dge_mode=hwdge`` or is not ``nisa.engine.sync`` /
          ``nisa.engine.scalar``.
        - a transpose rule is violated (``transpose_axes`` on a static transpose, an
          unsupported permutation, ``swdge`` / ``oob_mode.skip`` on a static transpose,
          or ``pattern_override`` / ``out_shape`` / ``engine`` with ``transpose=True``).
        - a transpose ``dst`` is given whose shape does not equal the transposed output
          shape (checked on both the static and gather paths).
        - a multi-P-tile load has a partition extent not divisible by the per-tile
          partition size (a multi-tile load with a P-remainder is unsupported).
        - the region cannot be expressed as one coalesced DMA.

        Example:
            .. code-block:: python

                tile = tiles[i, j].load()                       # auto-allocate
                col  = tiles[:, m].load()                       # coalesced column load
                tile = tiles[i, j].load(dst=buf.data)           # into a caller buffer
                rem  = tiles[i, j].load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)
                xT   = src_tiles[mi, ni].load(transpose=True)   # DMA-transpose

        **Single-DMA contract.** Every ``.load()`` emits one ``nisa.dma_copy``. If the
        region cannot be captured by a single coalesced access pattern, the call raises
        at trace time rather than silently issuing several -- loop and load tile-by-tile
        in that case. The sole exception is a partition-dim ``.fold()``, which emits K
        coalesced DMAs.

        **Transpose (transpose=True).** Requires a 2-byte dtype (bf16 / fp16). Two
        mechanisms, by whether the view carries an index:

        * Static (no index): transposes the view's tiles and **preserves the tile grid,
          axis-swapped** -- a ``(K, 1)`` source tile grid of ``(P, F)`` tiles returns a
          ``(1, K)`` grid of ``(F, P)`` tiles, indexable by coordinate (``xT[0, k]``), just
          like a non-transpose load. Single-DMA contract: F maps onto SBUF partition rows
          (<= 128), so ``F <= 128`` or an ``F`` that is a multiple of 128 is one coalesced
          DMA; an ``F`` that is ``> 128`` and *not* a multiple of 128 would need two DMAs
          and is **rejected** -- split with ``whole_tiles()`` / ``remainder_tiles()`` (or
          pick a tile_size whose F is <= 128 or a multiple of it).
        * Indirect / gather (``view[idx]``): gathers the indexed rows and transposes in
          one pass. ``transpose_axes`` picks the rank -- ``(1, 0)`` for 2-D,
          ``(2, 1, 0)`` for 3-D, ``(3, 1, 2, 0)`` for the 4-D reshape-trick form.

        ``pattern_override`` / ``out_shape`` are rejected on the transpose path -- it
        builds its own AP and output shape. A gather transpose returns a single tile:
        ``(D, N)`` for a 2-D ``(N, D)`` gather, ``(d2, d1, d0)`` for a 3-D ``(2, 1, 0)``,
        and ``(P, 1, f_tiles, rows)`` for the 4-D reshape-trick form.

        .. note::

            A gather (indirect) transpose carries compiler-enforced hardware constraints
            that NeuroTile does not re-validate: the index must be a uint32 2-D tensor, the
            gathered-row count must be a multiple of 16 in ``[16, 128]``, the transposed
            free dim must be ``<= 128``, and it requires TRN2 or later (``dge_mode=hwdge``
            on this path additionally requires ``P == 16`` and ``F % 128 == 0``). The
            ``oob_mode.skip`` / ``oob_value`` per-row skip on the gather-transpose path is
            plumbed through but unverified on hardware; do not rely on it without
            re-checking.

        **Out-of-bounds handling.** ``oob_mode=nisa.oob_mode.skip`` suppresses boundary
        DMA faults; skipped lanes keep their prior SBUF contents. Add ``oob_value`` to
        pre-fill the destination (one targeted ``memset`` before the DMA) so remainder
        lanes read a known value.

        Indirect indexing with blocks
        ------------------------------
        When the view carries a runtime index and a block grouping, the DMA count depends
        on the index kind: a scalar offset shifts the whole block base uniformly in a
        single coalesced DMA, while a vector (gather) offset issues one indirect DMA per
        tile within the block.

        Memory layout
        -------------
        SBUF has only 128 partition rows, so a load whose partition extent spans more than
        one P-tile does not stack tiles on the partition axis -- the extra P-tiles are
        packed into the free axis, and the returned buffer is ``(tile_P, total_F)`` with the
        P-tiles laid out flat along F. With ``out_shape`` set (and ``pattern_override``),
        the loaded buffer is typed at ``out_shape`` as a single tile, independent of the HBM
        source ``tile_size``.

        See Also:
            stream: rotating-buffer load that overlaps DMA with compute.
            store: the SBUF-to-HBM mirror.
            data: the underlying NKI tensor view for a manual ``nisa.dma_copy``.

        """
        assert self._grid.tile_size is not None
        assert isinstance(self._layout, HBMLayout), "load() is only valid on HBM views"
        assert self._grid.n_batch_dims == 0, (
            "load() requires all batch dims to be consumed first (got "
            + str(self._grid.n_batch_dims)
            + " unconsumed batch dim(s), element_shape="
            + str(self._grid.remaining)
            + "). Index or iterate the batch dims before calling .load()."
        )
        if transpose:
            # engine= steers HWDGE descriptor gen on the dma_copy path; the
            # transpose path uses nisa.dma_transpose, which has no engine param.
            assert engine is None, (
                "NDSlice.load(transpose=True, engine=...): engine= is not supported "
                "on the transpose path (nisa.dma_transpose has no engine param). "
                "Drop engine= or use a non-transpose load."
            )
            _validate_transpose_load(
                dge_mode,
                oob_mode,
                oob_value,
                priority,
                transpose_axes,
                is_indirect=self._layout.indirect is not None,
                n_gathered_dims=len(self._grid.gathered_dims()),
                pattern_override=pattern_override,
                out_shape=out_shape,
            )
        else:
            if oob_value is not None:
                assert oob_mode is not None, (
                    "NDSlice.load: oob_value= requires oob_mode= "
                    "(typically nisa.oob_mode.skip) -- without an oob_mode, the "
                    "value is never written."
                )
            _validate_priority(priority)
            _validate_dma_engine(engine, dge_mode, "NDSlice.load")

        # Partition fold: multi-DMA path
        if self._dma_override is not None and self._dma_override[0] == "partition_fold":
            load_dtype = dtype if dtype is not None else self.dtype
            sbuf_grid, sbuf_layout = HBMLayout.load_partition_fold(
                self._layout,
                self._grid,
                self._dma_override[1],
                load_dtype,
                dge_mode,
                oob_mode=oob_mode,
                priority=priority,
                engine=engine,
            )
            return NDSlice(sbuf_grid, sbuf_layout)

        # AP override (free-dim fold): use stored pattern for HBM side
        if self._dma_override is not None and self._dma_override[0] == "ap_override":
            if pattern_override is None:
                pattern_override = self._dma_override[1]

        # Stream defaults: fall through to BlockStream-provided rotating buffer,
        # pattern, and out_shape when the caller didn't pass them. Preserves
        # the load semantics BlockStream iteration relies on.
        if dst is None and self._load_dst is not None:
            effective_dtype = dtype if dtype is not None else self.dtype
            if not self.is_remainder and effective_dtype == self._load_dst.dtype:
                dst = self._load_dst
        if pattern_override is None and self._load_pattern_override is not None:
            pattern_override = self._load_pattern_override
        if out_shape is None and self._load_out_shape is not None:
            out_shape = self._load_out_shape

        # After internal defaults resolve, pattern_override needs out_shape (or dst)
        # so the SBUF destination shape is known.
        if pattern_override is not None and dst is None:
            assert out_shape is not None, (
                "NDSlice.load: pattern_override= requires out_shape= or dst= so the SBUF destination shape is known."
            )

        # Standard path: layout handles DMA + SBUF wrapping
        sbuf_grid, sbuf_layout = self._layout.load(
            self._grid,
            dtype=dtype,
            dst=dst,
            oob_mode=oob_mode,
            oob_value=oob_value,
            out_shape=out_shape,
            transpose=transpose,
            transpose_axes=transpose_axes,
            dge_mode=dge_mode,
            priority=priority,
            pattern_override=pattern_override,
            engine=engine,
        )
        return NDSlice(sbuf_grid, sbuf_layout)

    def store(
        self,
        data,
        dtype=None,
        dge_mode=None,
        oob_mode=None,
        priority: Optional[int] = None,
        pattern_override: Optional[list] = None,
        engine=None,
    ) -> None:
        """store(data, dtype=None, dge_mode=None, oob_mode=None, priority=None, pattern_override=None, engine=None) -> None

        Transfer SBUF data into this view's HBM region -- the mirror of ``load()``.

        The transfer is a single ``nisa.dma_copy`` (the single-DMA contract), so a
        coalesced multi-tile store goes out as one strided DMA.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            data (nl.ndarray | .data view): The SBUF source -- either a raw ``nl.ndarray``
                or the ``.data`` view of a loaded view. Pass ``view.data``, not the view
                object itself.
            dtype: Currently ignored -- ``store()`` performs no cast (the destination dtype
                is the HBM tensor's own). Any cast must be applied to ``data`` before the
                store. (Accepted for signature symmetry with ``load()``.)
            dge_mode (nisa.dge_mode.* | None): A DMA-generation mode hint.
            oob_mode (nisa.oob_mode.skip | None): When ``skip``, suppresses out-of-bounds
                DMA faults at boundaries; the skipped positions are simply not written.
            priority (int | None): The DMA QoS level in ``[0, 3]``; omitted when None.
            pattern_override (list[[int, int]] | None): A custom HBM access pattern, each
                level a ``[stride, count]`` pair.
            engine (nisa.engine.* | None): The HWDGE descriptor-generation engine for the
                transfer -- ``nisa.engine.sync`` or ``nisa.engine.scalar``. It applies only
                with ``dge_mode=nisa.dge_mode.hwdge``; with the default None the compiler
                selects the engine.

        Returns:
            None.

        Raises ``AssertionError`` in any of these cases:

        - called on an untiled view (one with no ``tile_size``).
        - called on an SBUF view (``store()`` writes to HBM).
        - ``data`` is an ``NDSlice`` rather than its ``.data`` view.
        - the view still has leading batch dimensions that have not been indexed down to
          a single tile.
        - ``priority`` is outside ``[0, 3]``.
        - ``engine`` is set without ``dge_mode=hwdge`` or is not ``nisa.engine.sync`` /
          ``nisa.engine.scalar``.
        - ``pattern_override`` levels are not ``[stride, count]`` pairs.

        Example:
            .. code-block:: python

                dst_tiles[i, j].store(tile.data)          # store a loaded tile
                dst_tiles[i, j].store(sbuf_result)        # store a raw nl.ndarray
                out_blocks[0, nb].store(acc.data)         # coalesced multi-tile store

        See Also:
            load: the HBM-to-SBUF mirror.
            data: the underlying NkiTensor view this method consumes.

        """
        assert self._grid.tile_size is not None
        assert isinstance(self._layout, HBMLayout), "store() is only valid on HBM views"
        assert not isinstance(data, NDSlice), "store() expects ndarray or .data view, not NDSlice. Use data.data"
        assert self._grid.n_batch_dims == 0, (
            "store() requires all batch dims to be consumed first (got "
            + str(self._grid.n_batch_dims)
            + " unconsumed batch dim(s), element_shape="
            + str(self._grid.remaining)
            + "). Index or iterate the batch dims before calling .store()."
        )
        _validate_priority(priority)
        _validate_dma_engine(engine, dge_mode, "NDSlice.store")
        if pattern_override is not None:
            assert isinstance(pattern_override, (list, tuple)), (
                "store() pattern_override= must be a list/tuple of [stride, count] pairs."
            )
            for level in pattern_override:
                assert isinstance(level, (list, tuple)) and len(level) == 2, (
                    "store() pattern_override= levels must each be [stride, count] pairs."
                )

        # Partition fold store: K separate DMAs (reverse of load)
        if self._dma_override is not None and self._dma_override[0] == "partition_fold":
            HBMLayout.store_partition_fold(
                self._source,
                self._offset,
                self._dma_override[1],
                self.element_shape,
                data,
                dge_mode,
                priority=priority,
                oob_mode=oob_mode,
                engine=engine,
            )
            return

        # AP override (free-dim fold): use stored pattern for HBM side
        if self._dma_override is not None and self._dma_override[0] == "ap_override":
            if pattern_override is None:
                pattern_override = self._dma_override[1]

        self._layout.store(
            data,
            self._grid,
            oob_mode=oob_mode,
            dge_mode=dge_mode,
            priority=priority,
            pattern_override=pattern_override,
            engine=engine,
        )

    # ================================================================
    # Transforms -- delegate to standalone functions + Layout
    # ================================================================

    def _logical_strides(self):
        """The view's logical element strides (the coordinate space transforms use)."""
        return self._layout.transform_strides(self.element_shape)

    def _as_nki_view(self):
        """This view's current layout as a single ``NkiTensor`` (for transforms)."""
        return nki_strided_view(self.element_shape, self._logical_strides(), self.dtype, self.buffer_type)

    def _rebuild_from(self, transformed_view):
        """Rebuild this view from a transformed ``NkiTensor``'s shape + strides."""
        # We read back only shape/strides, so a transform that shifted the offset
        # would silently lose it -- guard so misuse fails loudly.
        assert transformed_view.offset == 0, (
            "NDSlice._rebuild_from: routed transform changed offset; only offset-preserving "
            "transforms may go through _as_nki_view (use slice()/__getitem__ for offset shifts)."
        )
        return self._single_tile_view(tuple(transformed_view.shape), tuple(transformed_view.strides))

    def _single_tile_view(self, new_element_shape, new_strides):
        """Build a single-tile NDSlice for a transformed (reshaped/permuted/...) view."""
        tile_size = tuple(new_element_shape)
        new_layout = self._layout.apply_transform(new_strides)

        # SBUF: element-level grid with one ELEM axis per dim sized to the new
        # element_shape -- NDSlice.element_shape reads grid.remaining (count*step
        # of outer axis), so axes must carry the actual counts.
        if isinstance(self._layout, SBUFLayout):
            _elem_axes = []
            for _d in range(len(new_element_shape)):
                _elem_axes.append(Axis(count=new_element_shape[_d], step=1, dim=_d, label=AxisLabel.ELEM))
            new_grid = Grid.from_shape(
                element_shape=new_element_shape,
                tile_size=tile_size,
                axes=tuple(_elem_axes),
            )
        else:
            new_grid = Grid.from_shape(element_shape=new_element_shape, tile_size=tile_size)

        return NDSlice(new_grid, new_layout)

    def _validate_dim(self, dim, label="dim"):
        """Shared dim validator for transform methods."""
        assert isinstance(dim, int) and not isinstance(dim, bool), "NDSlice." + label + " must be int, got " + str(dim)
        assert 0 <= dim < self.ndim, (
            "NDSlice." + label + "=" + str(dim) + " is out of range for ndim=" + str(self.ndim) + "."
        )

    def reshape_dim(self, dim: int, shape: tuple) -> "NDSlice":
        """reshape_dim(dim, shape) -> NDSlice

        Split one dimension into several sub-dimensions. Metadata only -- no data is
        moved and nothing is allocated.

        This exposes interior structure on the free axis -- a parity split, or a split of
        a packed head dimension, for example -- before tiling or a per-sub-dimension
        operation. The strides of the new sub-dimensions subdivide the original
        dimension's stride from the innermost outward. Like every transform, it applies to
        an HBM view (before ``load()``) and an SBUF view (after ``load()``): see
        ``reshape`` for the shared HBM-versus-SBUF behavior.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int): The dimension to split, in ``[0, ndim)``.
            shape (tuple[int, ...]): The sub-dimension sizes; their product must equal
                ``element_shape[dim]`` (validated -- a mismatch raises).

        Returns:
            NDSlice: a new view with ``dim`` replaced by ``len(shape)`` dims. The result is a
            single tile (any prior tile/block grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` if ``dim`` is not an int in ``[0, ndim)`` (a bool is
        rejected), or if ``prod(shape) != element_shape[dim]``.

        Example:
            .. code-block:: python

                view = src.reshape_dim(1, (8, 128))   # (128, 1024) -> (128, 8, 128), then nt.tiles(view, ...)

        See Also:
            flatten_dims: the inverse (merge dims).
            split: equal-chunk shorthand.

        """
        self._validate_dim(dim, "reshape_dim: dim")
        return self._rebuild_from(self._as_nki_view().reshape_dim(dim, tuple(shape)))

    def reshape(self, new_shape: tuple) -> "NDSlice":
        """reshape(new_shape) -> NDSlice

        Reshape the whole view to ``new_shape``. Metadata only -- no data is moved and
        nothing is allocated.

        This is for a wholesale relayout, such as ``(B, S, H) -> (B*S, H)``, when the
        layout is contiguous. For a surgical change to a single axis, ``reshape_dim`` or
        ``flatten_dims`` is the better fit.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            new_shape (tuple[int, ...]): The new element shape; its element count must
                match the current one, and the current layout must be reshape-compatible
                (contiguous over the merged dims). Both are validated -- a mismatch or a
                non-contiguous layout raises.

        Returns:
            NDSlice: a new view over the same memory with the new shape. The result is a
            single tile (any prior tile/block grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` if the element count changes, or the layout is not
        contiguous enough to express ``new_shape`` as a view (no copy is made).

        Example:
            .. code-block:: python

                view = src.reshape((B * S, H))   # then nt.tiles(view, ...)

        Applies to HBM and SBUF views
        -----------------------------
        Every transform (``reshape``, ``reshape_dim``, ``permute``, ``flatten_dims``,
        ``squeeze_dim``, ``expand_dim``, ``broadcast``, ``slice``, ``split``) works
        uniformly on an HBM view (before ``load()``) and an SBUF view (after ``load()``),
        subject only to the SBUF partition-axis restrictions (``permute`` requires
        ``dims[0] == 0``; an element-level ``slice`` on dim 0 is routed through SBUF
        sub-indexing). The two differ in what comes back: a transformed **HBM** view is a
        single tile spanning the new shape (``tile_size`` equals the new ``element_shape``),
        so it is materialized with ``.load()`` first; a transformed **SBUF** view
        reinterprets the existing buffer in place (no copy) and comes back at element
        granularity, so it is indexed at element level and fed straight to a compute op
        without a load.

        See Also:
            reshape_dim / flatten_dims: surgical single-axis reshapes.

        """
        return self._rebuild_from(self._as_nki_view().reshape(tuple(new_shape)))

    def permute(self, dims: tuple) -> "NDSlice":
        """permute(dims) -> NDSlice

        Reorder the dimensions of the view. Metadata only -- no data is moved and nothing
        is allocated.

        A common use is bringing an axis into a matmul-friendly position before tiling.
        ``dims[i]`` names the original dimension that moves to position ``i``.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dims (tuple[int, ...]): A permutation of ``range(ndim)``; ``dims[i]`` names the
                original dimension that moves to position ``i``. Its length must equal
                ``ndim``.

        Returns:
            NDSlice: a new view with the dimensions and their strides reordered. The result is
            a single tile (any prior tile/block grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` if ``dims`` is not a permutation of ``range(ndim)`` (wrong
        length, a repeated or out-of-range index, or a non-int entry), or -- on an on-chip
        (SBUF / PSUM) view -- if ``dims[0] != 0`` (the partition axis is hardware-fixed). On
        an HBM view dimension 0 may move freely.

        Example:
            .. code-block:: python

                view = w.reshape((H1, P)).permute((1, 0))   # then nt.tiles(view, ...)

        See Also:
            reshape_dim / flatten_dims: split / merge dims.

        """
        # dims must be a true permutation of range(ndim): NkiTensor accepts a
        # duplicate index (e.g. (0, 0)) and silently drops a dim, so guard here.
        assert isinstance(dims, (tuple, list)), "NDSlice.permute: dims= must be a tuple or list of ints, got " + str(
            dims
        )
        assert len(dims) == self.ndim, (
            "NDSlice.permute: dims= has " + str(len(dims)) + " entries but ndim=" + str(self.ndim) + "; must match."
        )
        for i in range(len(dims)):
            d = dims[i]
            assert isinstance(d, int) and not isinstance(d, bool), (
                "NDSlice.permute: dims[" + str(i) + "] must be int, got " + str(d)
            )
        # Tracer-safe permutation check: the NKI ParserFrontend cannot resolve
        # builtins.sorted at trace time, so verify every axis in range(ndim)
        # appears in dims. Combined with the len(dims) == ndim guard above, that
        # is exactly a permutation (no duplicates, none out of range).
        for expected in range(self.ndim):
            found = False
            for i in range(len(dims)):
                if dims[i] == expected:
                    found = True
            assert found, (
                "NDSlice.permute: dims=" + str(tuple(dims)) + " must be a permutation of range(" + str(self.ndim) + ")."
            )
        # NkiTensor.permute enforces the partition-axis rule (dims[0] == 0) for
        # on-chip views natively, so no separate SBUF guard is needed here.
        return self._rebuild_from(self._as_nki_view().permute(tuple(dims)))

    def flatten_dims(self, start_dim: int, end_dim: int) -> "NDSlice":
        """flatten_dims(start_dim, end_dim) -> NDSlice

        Merge the contiguous dimensions ``[start_dim..end_dim]`` into one. Metadata only
        -- no data is moved and nothing is allocated.

        The merged dimension takes the innermost stride, so the dimensions being merged
        must be contiguous in the layout (validated -- merging non-contiguous dims raises).
        Non-adjacent dimensions are merged at DMA time with ``fold`` instead.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            start_dim (int): The first dimension to merge, in ``[0, ndim)``.
            end_dim (int): The last dimension to merge, in ``[start_dim, ndim)``.

        Returns:
            NDSlice: a new view whose merged dimension has size equal to the product of
            the merged sizes and takes the innermost merged dimension's stride.
            ``start_dim == end_dim`` is accepted as a no-op merge (it leaves the element
            shape unchanged). The result is a single tile (any prior tile/block grid is
            collapsed); re-tile to regrid.

        Raises ``AssertionError`` if ``start_dim`` or ``end_dim`` is not an int in
        ``[0, ndim)``, ``start_dim > end_dim`` (equality is allowed), or the merged dims
        are not contiguous in the layout.

        Example:
            .. code-block:: python

                flat = x.flatten_dims(0, 1)   # (B, S, H) -> (B*S, H), then nt.tiles(flat, ...)

        See Also:
            reshape_dim: the inverse (split a dim).
            fold: merge non-adjacent dims via a DMA recipe.

        """
        self._validate_dim(start_dim, "flatten_dims: start_dim")
        self._validate_dim(end_dim, "flatten_dims: end_dim")
        assert start_dim <= end_dim, (
            "NDSlice.flatten_dims: start_dim=" + str(start_dim) + " must be <= end_dim=" + str(end_dim) + "."
        )
        return self._rebuild_from(self._as_nki_view().flatten_dims(start_dim, end_dim))

    def rearrange(self, src_pattern: tuple, dst_pattern: tuple, fixed_sizes: Optional[dict] = None) -> "NDSlice":
        """rearrange(src_pattern, dst_pattern, fixed_sizes=None) -> NDSlice

        Split, reorder, and merge dimensions in one einops-style named operation.
        Metadata only -- no data is moved and nothing is allocated.

        This is the single-call form of a ``reshape_dim`` / ``permute`` / ``flatten_dims``
        chain: name the axes on each side and let matching names drive the transform.
        A grouped tuple in ``src_pattern`` splits that dimension; a grouped tuple in
        ``dst_pattern`` merges its members. It is the natural way to express a coalesced
        layout regroup -- e.g. mapping a transpose-load's packed free axis onto an HBM
        destination's row bands -- without deriving strides by hand.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            src_pattern (tuple): Source axis names, one entry per current dim; a grouped
                tuple of names splits that dim (sizes come from ``fixed_sizes``).
            dst_pattern (tuple): Destination axis names -- the same names as ``src_pattern``
                reordered; a grouped tuple merges its members into one dim.
            fixed_sizes (dict[str, int] | None): Known sizes for split axes, so the
                remaining split size can be inferred.

        Returns:
            NDSlice: a new view with the dimensions split / reordered / merged. The result is
            a single tile (any prior tile/block grid is collapsed); re-tile to regrid.

        Example:
            .. code-block:: python

                # (ni p) q -> p ni q : lift the packed row-band axis out in front.
                dst = out_tiles[:, mi].rearrange((("ni", "p"), "q"), ("p", "ni", "q"), {"p": 128})

        See Also:
            reshape_dim / permute / flatten_dims: the split / reorder / merge primitives
            this composes.

        """
        return self._rebuild_from(self._as_nki_view().rearrange(src_pattern, dst_pattern, fixed_sizes))

    def squeeze_dim(self, dim: int) -> "NDSlice":
        """squeeze_dim(dim) -> NDSlice

        Remove a size-1 dimension. Metadata only -- no data is moved and nothing is
        allocated.

        This drops a unit axis left over from an indexing or reshape chain.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int): The dimension to remove, in ``[0, ndim)``. It must have
                ``element_shape[dim] == 1``.

        Returns:
            NDSlice: a new view with the size-1 dimension removed. The result is a single tile
            (any prior tile/block grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` if ``dim`` is not an int in ``[0, ndim)``, or the named
        dimension is not size 1.

        Example:
            .. code-block:: python

                v = view.squeeze_dim(1)   # (128, 1, 512) -> (128, 512)

        See Also:
            expand_dim: the inverse (insert a size-1 dim).

        """
        self._validate_dim(dim, "squeeze_dim: dim")
        return self._rebuild_from(self._as_nki_view().squeeze_dim(dim))

    def expand_dim(self, dim: int) -> "NDSlice":
        """expand_dim(dim) -> NDSlice

        Insert a size-1 dimension at position ``dim``. Metadata only -- no data is moved
        and nothing is allocated. It pairs with ``broadcast(dim, size)``, which grows the
        size-1 dimension to a real size for an elementwise operation.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int): The insert position, in ``[0, ndim]`` -- the end position is allowed,
                for a trailing insertion.

        Returns:
            NDSlice: a new view with one additional size-1 dimension inserted at ``dim``. The
            result is a single tile (any prior tile/block grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` if ``dim`` is not an int in ``[0, ndim]`` (a bool is
        rejected; the upper bound is inclusive to allow a trailing insert).

        Example:
            .. code-block:: python

                bc = inv_rms.expand_dim(2).broadcast(2, S)

        See Also:
            broadcast: grow the inserted dim to a real size.
            squeeze_dim: the inverse (remove a size-1 dim).

        """
        # Insert position is [0, ndim] inclusive (trailing insert allowed).
        # NkiTensor silently accepts OOB / negative / bool, so guard here.
        assert isinstance(dim, int) and not isinstance(dim, bool), "NDSlice.expand_dim: dim must be int, got " + str(
            dim
        )
        assert 0 <= dim <= self.ndim, (
            "NDSlice.expand_dim: dim="
            + str(dim)
            + " is out of range for ndim="
            + str(self.ndim)
            + " (valid [0, ndim])."
        )
        return self._rebuild_from(self._as_nki_view().expand_dim(dim))

    def broadcast(self, dim: int, size: int) -> "NDSlice":
        """broadcast(dim, size) -> NDSlice

        Broadcast a size-1 dimension to ``size`` with stride 0. Metadata only -- the data
        is logically repeated, not copied.

        This aligns a row or column vector against a larger tile for a tensor-tensor
        operation, without materializing the repeated copies.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int): The dimension to broadcast, in ``[0, ndim)``. Its current size must
                be 1.
            size (int): The target size, a positive int.

        Returns:
            NDSlice: a new view with the dimension grown to ``size`` (its stride stays 0). The
            result is a single tile (any prior tile/block grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` in any of these cases:

        - ``dim`` is not an int in ``[0, ndim)``.
        - ``size`` is not a positive int.
        - the dimension's current size is not 1.

        Example:
            .. code-block:: python

                gamma_bc = gamma.expand_dim(1).broadcast(1, BxS)

        See Also:
            expand_dim: insert the size-1 dim to broadcast.

        """
        self._validate_dim(dim, "broadcast: dim")
        assert isinstance(size, int) and not isinstance(size, bool) and size > 0, (
            "NDSlice.broadcast: size= must be a positive int, got " + str(size)
        )
        return self._rebuild_from(self._as_nki_view().broadcast(dim, size))

    def slice(self, dim: int, start: int, end: int) -> "NDSlice":
        """slice(dim, start, end) -> NDSlice

        Narrow ``dim`` to the elements ``[start, end)``, at element granularity. Metadata
        only -- no data is moved and nothing is allocated.

        Where ``__getitem__`` operates at the tile or block grid level, ``slice()`` always
        narrows at the level of individual elements, and it works on HBM, SBUF, and PSUM
        views. The bounds must be compile-time ints -- unlike slice-style ``__getitem__``
        indexing, a runtime offset (a loop variable or other runtime expression) is
        rejected.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int): The dimension to narrow, in ``[0, ndim)``.
            start (int): The start element index, inclusive (a plain Python int).
            end (int): The end element index, exclusive (a plain Python int). It must
                satisfy ``0 <= start < end <= element_shape[dim]``.

        Returns:
            NDSlice: a new view over the narrowed region (the dimension's element extent
            reduced to ``end - start``). On an HBM view the result is a single tile spanning
            the narrowed region (any prior tile / block grid is not carried through), so it
            is directly loadable as one tile; on an SBUF view it narrows the chosen dim via
            SBUF sub-indexing, and an element-level P-narrow (dim 0) is valid this way; on a
            PSUM view it narrows the dimension within the already-selected tile -- a free
            dimension (``dim >= 1``) shifts the in-tile free-axis offset by ``start`` (so a
            strided or padded sub-region of the tile can be addressed) and ``dim 0`` (the
            partition dimension) reduces only the element count, without selecting a
            different tile.

        Raises ``AssertionError`` in any of these cases:

        - ``dim`` is not an int in ``[0, ndim)``.
        - ``start`` or ``end`` is not an int (a bool, or a runtime expression, is
          rejected).
        - the range is empty or out of bounds (``0 <= start < end <= element_shape[dim]``
          must hold).

        Example:
            .. code-block:: python

                half = tile.slice(1, 0, 256)   # (128, 512) -> (128, 256)

        See Also:
            __getitem__: tile/block-level indexing.

        """
        self._validate_dim(dim, "slice: dim")
        assert isinstance(start, int) and not isinstance(start, bool), "NDSlice.slice: start must be int, got " + str(
            start
        )
        assert isinstance(end, int) and not isinstance(end, bool), "NDSlice.slice: end must be int, got " + str(end)
        assert 0 <= start < end <= self.element_shape[dim], (
            "NDSlice.slice: invalid range ["
            + str(start)
            + ", "
            + str(end)
            + ") for dim="
            + str(dim)
            + " (element_shape[dim]="
            + str(self.element_shape[dim])
            + ")."
        )
        count = end - start

        # SBUF: delegate to sub_index for all dims (P and F)
        if isinstance(self._layout, SBUFLayout):
            key = []
            for d in range(self.ndim):
                if d == dim:
                    key.append(slice(start, end))
                else:
                    key.append(slice(0, self.element_shape[d]))
            return self._sub_index(tuple(key))

        new_element_shape = replace_at(self.element_shape, dim, count)
        new_grid = Grid.from_shape(new_element_shape, tile_size=new_element_shape)

        # PSUM: narrow within the active tile -- advance the free-dim element
        # offset, leaving tile selection (tile_arrays index) untouched. dim 0
        # (partition) narrows via Grid counts only; the offset is on free dims.
        if isinstance(self._layout, PSUMLayout):
            if dim == 0:
                new_layout = self._layout
            else:
                new_layout = self._layout.narrow_free(dim, start, self.element_shape)
            return NDSlice(new_grid, new_layout)

        # HBM: advance the element offset by `start` along dim + narrow shape.
        new_layout = self._layout.advance(dim, start, 1)
        return NDSlice(new_grid, new_layout)

    def fold(self, src_dim: int, into_dim: int, position: str = "outer") -> "NDSlice":
        """fold(src_dim, into_dim, position="outer") -> NDSlice

        Merge ``src_dim`` into ``into_dim``, reducing ``ndim`` by 1, via a DMA recipe.

        Unlike ``flatten_dims`` (adjacent dims, pure metadata), ``fold`` records a recipe
        that the next ``.load()`` / ``.store()`` applies, so it can merge *non-adjacent*
        dims at DMA time. It is symmetric for load and store (round-trip safe), and folds
        compose.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            src_dim (int): Dim folded away, in ``[0, ndim)``.
            into_dim (int): Dim that absorbs ``src_dim``, in ``[0, ndim)``; must differ from
                ``src_dim``.
            position (str): ``"outer"`` (default) places ``src_dim`` outside ``into_dim`` in
                the merged stride; ``"inner"`` places it inside.

        Returns:
            NDSlice: a new view with ``ndim - 1`` dims and a fold recipe attached as an
            immutable DMA override; the recipe rides along until the next ``.load()`` /
            ``.store()`` consumes it. The merged dimension's size is the product of the two
            dims' sizes; its stride is taken from ``into_dim`` for ``position="outer"`` and
            from ``src_dim`` for ``position="inner"`` (to preserve the HBM access pattern --
            the SBUF side is built contiguously regardless). A free-dim fold pre-binds both
            the merged-axis access pattern and the post-fold SBUF destination shape onto the
            view, so the following ``.load()`` / ``.store()`` is fully self-describing and
            needs no ``out_shape=`` / ``pattern_override=``. Folds compose: folding a view
            that already carries a free-dim recipe threads the existing access-pattern base
            through the chain rather than rebuilding it.

        Raises ``AssertionError`` in any of these cases:

        - ``src_dim`` or ``into_dim`` is not an int in ``[0, ndim)``.
        - they are equal.
        - ``position`` is not ``"outer"`` / ``"inner"``.

        Example:
            .. code-block:: python

                folded = src_tiles[0, 0, 0].fold(2, 1)   # (P, F, K) -> (P, F*K), one DMA
                tile = folded.load()

        A free-dim fold merges the ``K`` sub-columns of each F position into one wider
        free axis::

                F=2, K=3                       F*K = 6
               ┌────┬────┬────┐               ┌──┬──┬──┬──┬──┬──┐
            P  │ k0 │ k1 │ k2 │   fold(2,1)   │k0│k1│k2│k0│k1│k2│
            v  │ .. │ .. │ .. │  ──────────>  │..│..│..│..│..│..│
               └────┴────┴────┘               └──┴──┴──┴──┴──┴──┘

        **Free-dim fold vs partition fold.** A free-dim fold (``src_dim`` and ``into_dim``
        both > 0) stays a single coalesced DMA. A partition fold (either dim == 0) emits K
        separate DMAs -- one per slice along the folded partition dim -- since one access
        pattern cannot address disjoint partition ranges. This is the single exception to
        ``load`` / ``store``'s single-DMA contract.

        See Also:
            flatten_dims: metadata-only merge of adjacent dims.

        """
        self._validate_dim(src_dim, "fold: src_dim")
        self._validate_dim(into_dim, "fold: into_dim")
        assert src_dim != into_dim, "NDSlice.fold: src_dim and into_dim must differ, got " + str(src_dim)
        assert position in ("outer", "inner"), "NDSlice.fold: position= must be 'outer' or 'inner', got " + str(
            position
        )
        strides = self._logical_strides()
        result = compute_fold(self.element_shape, strides, src_dim, into_dim, position)
        view = self._single_tile_view(result[0], result[1])

        # Determine AP base: use existing override if present (chained fold),
        # otherwise build from current strides + element_shape.
        if self._dma_override is not None and self._dma_override[0] == "ap_override":
            ap_base = self._dma_override[1]
        else:
            ap_base = []
            for d in range(len(self.element_shape)):
                ap_base.append([strides[d], self.element_shape[d]])

        is_partition_fold = src_dim == 0 or into_dim == 0

        if is_partition_fold:
            K = self.element_shape[src_dim]
            P_per_slice = self.element_shape[into_dim]
            fold_stride = strides[src_dim]
            base_pattern = []
            for i in range(len(ap_base)):
                if i != src_dim:
                    base_pattern.append(ap_base[i])
            override = ("partition_fold", (K, P_per_slice, fold_stride, base_pattern))
            # Partition fold's multi-DMA path resolves shape internally;
            # no out_shape needed.
            return NDSlice(view._grid, view._layout, dma_override=override)

        # Free-dim fold: pattern_override on .load() needs the post-fold
        # element shape so the SBUF destination can be sized.
        override = ("ap_override", ap_base)
        return NDSlice(
            view._grid,
            view._layout,
            dma_override=override,
            load_out_shape=tuple(view.element_shape),
        )

    def split(self, dim: int, n: int) -> "NDSlice":
        """split(dim, n) -> NDSlice

        Split ``dim`` into ``n`` equal chunks. Metadata only -- no data is moved and
        nothing is allocated.

        This is shorthand for ``reshape_dim(dim, (n, element_shape[dim] // n))`` when the
        chunks are equal-sized.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            dim (int): The dimension to split, in ``[0, ndim)``.
            n (int): The number of chunks, a positive int that must evenly divide
                ``element_shape[dim]``.

        Returns:
            NDSlice: a new view with ``dim`` replaced by
            ``(n, element_shape[dim] // n)``. The result is a single tile (any prior tile/block
            grid is collapsed); re-tile to regrid.

        Raises ``AssertionError`` in any of these cases:

        - ``dim`` is not an int in ``[0, ndim)``.
        - ``n`` is not a positive int.
        - ``n`` does not divide ``element_shape[dim]``.

        Example:
            .. code-block:: python

                view.split(1, 4)   # (B, S) -> (B, 4, S/4)

        See Also:
            reshape_dim: split into arbitrary (non-equal) sub-dims.

        """
        self._validate_dim(dim, "split: dim")
        assert isinstance(n, int) and not isinstance(n, bool) and n > 0, (
            "NDSlice.split: n must be a positive int, got " + str(n)
        )
        assert self.element_shape[dim] % n == 0, (
            "NDSlice.split: element_shape["
            + str(dim)
            + "]="
            + str(self.element_shape[dim])
            + " is not divisible by n="
            + str(n)
            + "."
        )
        chunk_size = self.element_shape[dim] // n
        return self.reshape_dim(dim, (n, chunk_size))

    def _drop_unit_batch_dims(self):
        """Drop batch dims narrowed to a single tile (e.g., rank dim after shard).

        Any batch dim whose remaining fits in a single tile is consumed and
        dropped -- matches the source-tensor convention that rank dims
        collapse once the rank shard is applied.
        """
        grid = self._grid
        layout = self._layout
        drop_dims = []
        for d in range(grid.n_batch_dims):
            tile_step = grid.tile_size[d] if grid.tile_size is not None else 1
            if grid.remaining[d] <= tile_step:
                drop_dims.append(d)

        for d_idx in range(len(drop_dims) - 1, -1, -1):
            d = drop_dims[d_idx]
            # drop_dim removes all axes on `d` and shrinks element_shape;
            # an explicit consume() before it would be redundant.
            grid = grid.drop_dim(d)
            layout = layout.drop_dim(d)

        return NDSlice(grid, layout)

    # ================================================================
    # Remainder
    # ================================================================

    def whole_tiles(self) -> list:
        """whole_tiles() -> list

        Return only the full (non-partial) sub-views of this view -- the clean interior
        of a view whose extent is not an exact multiple of the tile size.

        It pairs with ``remainder_tiles()`` so that each half is handled with the right
        DMA policy: a plain ``load()`` for the full tiles, and ``oob_mode=skip`` for the
        partial tiles at the boundary. It takes no arguments and filters the default
        (``dim=None``) iteration, so it inherits ``tolist()``'s default-dimension and
        batch-dim behavior.

        .. warning::

           This API is experimental and may change in future releases.

        Returns:
            list[NDSlice]: the sub-views whose ``is_remainder`` attribute is ``False``.

        Example:
            .. code-block:: python

                for tile in tiles[i, :].whole_tiles():
                    tile.load()                                 # full-extent DMA
                for rem in tiles[i, :].remainder_tiles():
                    rem.load(oob_mode=nisa.oob_mode.skip, oob_value=0.0)

        See Also:
            remainder_tiles: the boundary half.
            tolist: the underlying iteration.

        """
        items = self.tolist()
        result = []
        for item in items:
            if not item.is_remainder:
                result.append(item)
        return result

    def remainder_tiles(self) -> list:
        """remainder_tiles() -> list

        Return only the partial sub-views of this view -- the boundary tiles of a view
        whose extent is not an exact multiple of the tile size.

        It pairs with ``whole_tiles()`` to split iteration into a fast pass over the full
        tiles and a boundary pass that uses ``oob_mode`` / ``oob_value`` (see ``load``).

        .. warning::

           This API is experimental and may change in future releases.

        Returns:
            list[NDSlice]: the sub-views whose ``is_remainder`` attribute is ``True``.

        The ``whole_tiles()`` example shows the two halves used together.

        See Also:
            whole_tiles: the full (non-partial) tiles, with a combined example.
            load: the ``oob_mode`` / ``oob_value`` handling for remainder DMAs.

        """
        items = self.tolist()
        result = []
        for item in items:
            if item.is_remainder:
                result.append(item)
        return result

    # ================================================================
    # Representation
    # ================================================================

    def __repr__(self):
        return (
            "NDSlice(shape="
            + str(self.shape)
            + ", element_shape="
            + str(self.element_shape)
            + ", buffer_type="
            + str(self.buffer_type)
            + ")"
        )


# ============================================================================
# BlockStream
# ============================================================================


class BlockStream(nl.NKIObject):
    """A rotating-buffer DMA pipeline along one dimension of an HBM view.

    A ``BlockStream`` is the rotating-buffer DMA pipeline that ``NDSlice.stream()``
    returns; it is not constructed directly. The source is an HBM view and the rotating
    buffers live in SBUF. It allocates ``buffer_count`` SBUF buffers lazily on first
    access (the first load, iteration, indexing, or store -- constructing the stream
    allocates nothing), each sized for one step along the streamed dimension, and
    overlaps each step's DMA with the previous step's compute -- the standard
    double-buffering pattern for flowing operands. Each buffer holds exactly one step's
    worth of data: the same chunk a single-index subscript on that dimension yields
    (one tile for a tile column/row, one packed row/column of tiles for a multi-tile
    slice, one block for a block column). Buffers are sized from the view's *owned*
    extents, so under interleaved (gapped) sharding a slot holds only the owned tiles --
    dense SBUF, not the gapped iteration span. Stream position ``index`` maps to
    rotating slot ``index % buffer_count``, so slots are reused cyclically.

    The same rotating pool is driven in either of two ways:

    * By iteration -- ``for sv in stream: sv.load()`` (or ``stream.tolist()``). Each
      step's view comes with its rotating buffer already bound, so ``sv.load()``
      transfers into the correct buffer automatically.
    * By random access -- ``stream.load(k)`` / ``stream[k]`` / ``stream.store(k)``,
      for hand-written prolog / steady-state / epilog pipelining.

    The load type (``transpose`` / ``transpose_axes``) is fixed once -- at
    construction or by the first ``load()`` -- and sizes every buffer, so all later
    loads must match and a given stream serves one load type. Declaring
    ``transpose=True`` at construction sizes and grid-types every slot to the
    transposed output before any load runs, so ``stream[k]`` is a transposed-block view
    usable as a typed per-tile ``dst=`` target. The rotating-buffer dtype defaults to
    the source view's dtype. The step count and per-step stride are captured once, from
    the streamed dimension at construction time, and are not re-read; later indexing of
    the source view does not change them. Constructing a stream requires the source
    view's batch dims to be indexed away (see ``NDSlice.stream``).

    .. warning::

       This API is experimental and may change in future releases.

    Attributes:
        count (int): The number of steps along the streamed dimension. ``len(stream)``
            returns the same value.

    """

    def __init__(
        self,
        view,
        dim,
        buffer_count=2,
        dtype=None,
        transpose=False,
        transpose_axes=None,
        pattern_override=None,
        out_shape=None,
    ):
        self._view = view
        self._dim = dim
        self._step = view._layout.advance_step(view._grid, dim)
        self.count = view._grid.current_count(dim)
        self._buffer_count = buffer_count
        self._dtype = dtype if dtype is not None else view.dtype
        self._pattern_override = pattern_override
        self._out_shape = out_shape
        assert view._grid.n_batch_dims == 0, (
            "stream() requires all batch dims to be consumed first (got "
            + str(view._grid.n_batch_dims)
            + " unconsumed batch dim(s), element_shape="
            + str(view._grid.remaining)
            + "). Index or iterate the batch dims before calling .stream()."
        )
        # Load type (transpose / transpose_axes) fixes the slots' shape and grid:
        # declared here when transpose=True, else locked by the first access.
        # `_decided` guards the lock so slots are sized exactly once.
        self._transpose = transpose
        self._transpose_axes = transpose_axes
        self._decided = transpose
        self._buffers = None

    def _child(self, i):
        """Build the NDSlice view at stream position i (no buffer binding).

        A stream step consumes the streamed-dim's iteration level (same
        semantics as a single-int index on the dim) and places the
        cursor past the consumed dim. The loaded view then exposes the
        next-level grid (e.g. one block's interior tile grid after a
        per-block stream step).
        """
        child_grid = self._consumed_grid_for_step()
        child_layout = self._view._layout.advance(self._dim, i, self._step)
        # Clamp the child grid for the last step if it's a partial (remainder) block.
        child_grid = child_grid.truncate_to_source(
            self._dim, child_layout.dim_offset_elements(self._dim, child_grid.element_shape)
        )
        return NDSlice(child_grid, child_layout)

    def _bound_child(self, i):
        """Build the NDSlice at stream position i with stream-default kwargs
        pre-bound so sv.load() routes into the rotating buffer."""
        self._ensure_buffers_for_access()
        child_grid = self._consumed_grid_for_step()
        child_layout = self._view._layout.advance(self._dim, i, self._step)
        # Clamp the child grid for the last step if it's a partial (remainder) block.
        child_grid = child_grid.truncate_to_source(
            self._dim, child_layout.dim_offset_elements(self._dim, child_grid.element_shape)
        )
        return NDSlice(
            child_grid,
            child_layout,
            load_dst=self._buffers[i % self._buffer_count],
            load_pattern_override=self._pattern_override,
            load_out_shape=self._out_shape,
        )

    def _consumed_grid_for_step(self):
        """Grid representing one stream step: consume the streamed dim's
        outer axis and place the cursor at the next iterable position."""
        return self._view._grid.consume(self._dim).with_cursor_past_consumed((self._dim,))

    def _lock_load_type(self, transpose, transpose_axes):
        """Fix (or re-validate) the stream's load type. No allocation.

        Fixed at construction or by the first ``.load()``; once fixed every
        load must match, since the slots are sized once and cannot be resized.
        """
        if self._decided:
            assert transpose == self._transpose and transpose_axes == self._transpose_axes, (
                "BlockStream.load: all loads on a stream must use the same "
                "transpose/transpose_axes -- the load type is fixed (transpose="
                + str(self._transpose)
                + ", transpose_axes="
                + str(self._transpose_axes)
                + ") by stream(transpose=...) or the first load; this load passed "
                "transpose="
                + str(transpose)
                + ", transpose_axes="
                + str(transpose_axes)
                + ". Use one stream per load type."
            )
            return
        self._transpose = transpose
        self._transpose_axes = transpose_axes
        self._decided = True

    def _ensure_buffers(self, transpose, transpose_axes):
        """Lock the load type, then allocate the rotating slots (idempotent)."""
        self._lock_load_type(transpose, transpose_axes)
        self._allocate_buffers_once()

    def _ensure_buffers_for_access(self):
        """Lock + allocate for a mode-agnostic access (iteration / ``stream[k]`` /
        ``store``). Defaults to a normal load when the type isn't fixed yet."""
        if not self._decided:
            self._lock_load_type(False, None)
        self._allocate_buffers_once()

    def _allocate_buffers_once(self):
        """Allocate the rotating slots on first need (idempotent)."""
        if self._buffers is None:
            self._buffers = self._allocate_buffers()

    def _slot_shape(self):
        """SBUF allocation shape for one stream step, transpose-aware. Single
        source of truth for both the allocation and the slot view typing, so the
        two can't drift. ``out_shape`` overrides; a transpose step uses the
        transposed output shape."""
        if self._out_shape is not None:
            return tuple(self._out_shape)
        child = self._child(0)
        if self._transpose:
            return child._layout.transposed_sbuf_shape_for(child._grid)
        return child._layout.sbuf_shape_for(child._grid)

    def _slot_tile_size(self):
        """Per-tile granularity of a slot's logical grid, transpose-aware, so
        ``stream[k][ti, tj]`` indexes interior tiles like a loaded view."""
        if self._out_shape is not None:
            return tuple(self._out_shape)
        step_grid = self._consumed_grid_for_step()
        if self._transpose:
            return self._view._layout.transposed_tile_size_for(step_grid)
        return step_grid.tile_size

    def _slot_view(self, sbuf):
        """Type a rotating-slot buffer at the step's tile grid so ``stream[k]`` is
        tile-addressable, consistent with what ``stream.load(k)`` returns."""
        slot_shape = tuple(sbuf.shape)
        grid, layout = SBUFLayout.build_view(sbuf, slot_shape, self._slot_tile_size(), self._dtype, sbuf_buffer_type())
        return NDSlice(grid, layout)

    def _allocate_buffers(self):
        """Allocate buffer_count SBUF ndarrays sized for one stream step."""
        slot_shape = self._slot_shape()
        buffers = []
        for i in range(self._buffer_count):
            buffers.append(nl.ndarray(slot_shape, dtype=self._dtype, buffer=nl.sbuf))
        return buffers

    # ================================================================
    # Iteration
    # ================================================================

    def tolist(self) -> list:
        """tolist() -> list

        Return the stream's positions as a list of views, each with its rotating buffer
        already bound.

        Calling ``load()`` on the view for step ``k`` transfers into buffer
        ``k % buffer_count``. This is the form for consuming the stream end to end without
        explicit index arithmetic. It takes no arguments (it always walks the streamed
        dimension), unlike ``NDSlice.tolist(dim=)``, and allocates the rotating pool on
        first access (defaulting the load type to non-transpose if not already fixed).
        Iterating the stream directly (``for sv in stream``) is equivalent to iterating
        ``stream.tolist()``.

        .. warning::

           This API is experimental and may change in future releases.

        Returns:
            list[NDSlice]: one view per stream step (``count`` items), each with its
            rotating buffer bound.

        Example:
            .. code-block:: python

                for sv in tiles[:, 0].stream(buffer_count=2).tolist():
                    tile = sv.load()      # rotates buffers automatically

        Notes:
            When the streamed dimension's last step is a partial (remainder) tile, that
            final step is not routed into its rotating slot: ``sv.load()`` on a remainder
            step allocates a fresh SBUF buffer, so it gets no DMA / compute overlap. Only
            whole (non-remainder) steps reuse the pre-bound rotating buffer.

        See Also:
            load: random-access load by index.

        """
        result = []
        for i in range(self.count):
            result.append(self._bound_child(i))
        return result

    def _enumerate(self, start=0, mode=None):
        """Return (index, NDSlice) pairs."""
        items = self.tolist()
        result = []
        for i in range(len(items)):
            result.append((start + i, items[i]))
        return result

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        return self.count

    # ================================================================
    # Direct access by index -- random stream position (prolog/epilog)
    # ================================================================

    def load(
        self,
        index: int,
        dtype=None,
        transpose: bool = False,
        transpose_axes: Optional[tuple] = None,
        dge_mode=None,
        oob_mode=None,
        oob_value: Optional[float] = None,
        priority: Optional[int] = None,
        engine=None,
    ) -> "NDSlice":
        """load(index, dtype=None, transpose=False, transpose_axes=None, dge_mode=None, oob_mode=None, oob_value=None, priority=None, engine=None) -> NDSlice

        DMA stream position ``index`` into rotating slot ``index % buffer_count`` and return
        the slot's ``NDSlice``.

        This is the steady-state call inside a streamed loop: with ``buffer_count=2``, step
        ``k`` overlaps the DMA into slot ``(k+1) % 2`` with the compute on slot ``k % 2``.

        There is no ``dst=`` parameter -- the destination is always the rotating slot (or a
        fresh buffer on a dtype mismatch). To load into a caller buffer, use
        ``NDSlice.load`` directly.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            index (int): Stream position (0-indexed). The data lands in slot
                ``index % buffer_count``.
            dtype: Cast on DMA. Defaults to the stream dtype; a different dtype bypasses the
                rotating slot (a fresh SBUF buffer is allocated for this call, so that step
                does not overlap). The same dtype-match guard applies on the iteration path:
                ``sv.load(dtype=...)`` differing from the bound buffer's dtype also bypasses
                the slot.
            transpose (bool): DMA-transpose this step into its slot. The first ``.load()``
                fixes the stream's load type and sizes the slots; later loads must repeat
                the same ``transpose`` / ``transpose_axes``.
            transpose_axes (tuple[int] | None): Per-rank gather-transpose permutation
                (indirect only); see ``NDSlice.load``. Must match across all loads.
            dge_mode / oob_mode / oob_value / priority / engine: Forwarded to
                ``NDSlice.load`` on the non-transpose path -- so ``priority`` must be an int
                in ``[0, 3]``, and a pinned ``engine`` (``nisa.engine.sync`` /
                ``nisa.engine.scalar``) requires ``dge_mode=nisa.dge_mode.hwdge`` (``None`` /
                ``unknown`` / ``dma`` let the compiler pick). A transpose load takes no
                ``engine``. The stream's own ``pattern_override`` / ``out_shape`` apply on
                the non-transpose path only (the transpose slot is pre-sized).

        Returns:
            NDSlice: the rotating slot's view (or a fresh SBUF buffer on a dtype mismatch).
            ``.data`` is the underlying ``nl.ndarray``. A step advances one position along
            the streamed dimension (like a single integer index on that dimension), so the
            loaded view exposes whatever one step contains -- for example the tile grid
            inside one block when streaming a block dimension -- for further indexing.

        Raises ``AssertionError`` if the load type disagrees with the stream's fixed
        ``transpose`` / ``transpose_axes``, plus every forwarded ``NDSlice.load`` check
        (``priority`` outside ``[0, 3]``; a pinned ``engine`` without ``dge_mode=hwdge`` or
        not ``sync`` / ``scalar``; ``oob_value`` without ``oob_mode``; ``engine`` on a
        transpose load).

        Example:
            .. code-block:: python

                stream = tiles[:, 0].stream(buffer_count=2)
                for k in nl.affine_range(K):
                    tile = stream.load(k)
                    nisa.nc_matmul(acc, lhs.data, tile.data)

        See Also:
            NDSlice.stream: constructs the stream.
            __getitem__ / store: slot read / write-back.

        """
        self._ensure_buffers(transpose, transpose_axes)
        child = self._child(index)
        dst = self._buffers[index % self._buffer_count]
        if dtype is not None and dtype != self._dtype:
            dst = None
        # transpose ignores pattern_override/out_shape (its dst slot is pre-sized).
        # engine= is invalid on the transpose path (NDSlice.load asserts); pass it
        # through so that assert fires with a clear message rather than silently.
        if transpose:
            return child.load(
                dst=dst,
                dtype=dtype,
                transpose=True,
                transpose_axes=transpose_axes,
                dge_mode=dge_mode,
                oob_mode=oob_mode,
                oob_value=oob_value,
                priority=priority,
                engine=engine,
            )
        return child.load(
            dst=dst,
            dtype=dtype,
            dge_mode=dge_mode,
            oob_mode=oob_mode,
            oob_value=oob_value,
            priority=priority,
            pattern_override=self._pattern_override,
            out_shape=self._out_shape,
            engine=engine,
        )

    def __getitem__(self, index: int) -> "NDSlice":
        """__getitem__(index) -> NDSlice

        Return the view over rotating buffer ``index % buffer_count``. No data is moved
        (the buffer pool is allocated on first access if needed).

        The buffer is the pre-allocated SBUF region, typed at the step's tile grid (and
        transpose-aware), so it is tile-addressable (``stream[k][ti, tj]``) and its
        ``data`` attribute is the raw ``nl.ndarray``. It serves three roles: output
        streaming, reading a buffer that an earlier step loaded, and acting as a per-tile
        ``dst=`` target. Indexing is **not** ordinary sequence indexing -- ``index`` is
        taken modulo ``buffer_count`` to pick a rotating slot, so out-of-range or large
        indices wrap rather than raising, and the same physical buffer backs many positions.

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            index (int): The stream position, 0-indexed.

        Returns:
            NDSlice: the rotating buffer's view, typed at the step's tile grid.

        Example:
            .. code-block:: python

                # Output streaming: fill the buffer, then transfer it back to HBM.
                nisa.tensor_copy(out_stream[k].data, result)
                out_stream.store(k)

        .. warning::

            Indexing, iteration, and ``store`` are mode-agnostic: on a stream whose load
            type is not yet fixed, any of them locks it to **non-transpose** before
            allocating the slots, and a later ``load(transpose=True)`` then raises
            (the slots are already sized for a non-transpose load). To transpose-stream,
            declare ``transpose=True`` at construction or issue a transpose ``load()``
            before any plain access.

        See Also:
            load: DMA a step into its slot.
            store: write a slot back to HBM.

        """
        self._ensure_buffers_for_access()
        return self._slot_view(self._buffers[index % self._buffer_count])

    def store(self, index: int, oob_mode=None, dge_mode=None, priority: Optional[int] = None, engine=None) -> None:
        """store(index, oob_mode=None, dge_mode=None, priority=None, engine=None) -> None

        Transfer rotating buffer ``index % buffer_count`` back to its HBM position -- the
        mirror of ``load(index)``.

        This is the output-streaming step: a buffer obtained from ``stream[index]`` is
        filled, and ``stream.store(index)`` then writes it out. Unlike ``NDSlice.store``,
        there is no ``data=`` argument -- ``store`` reads the whole rotating slot as a
        contiguous SBUF access pattern -- and no ``dtype=`` argument (it performs no cast;
        the destination dtype is the HBM tensor's own).

        .. warning::

           This API is experimental and may change in future releases.

        Args:
            index (int): The stream position, 0-indexed. The slot ``index % buffer_count``
                is written to the HBM position computed for this stream position.
            oob_mode (nisa.oob_mode.skip | None): When ``skip``, suppresses out-of-bounds
                DMA faults at boundaries.
            dge_mode (nisa.dge_mode.* | None): A DMA-generation mode hint.
            priority (int | None): The DMA QoS level in ``[0, 3]``; forwarded to
                ``NDSlice.store``.
            engine (nisa.engine.* | None): The HWDGE descriptor-generation engine;
                forwarded to ``NDSlice.store`` (applies only with ``dge_mode=hwdge``).

        Returns:
            None.

        Raises ``AssertionError`` if ``priority`` is outside ``[0, 3]``, or if ``engine``
        is set without ``dge_mode=hwdge`` or is not ``nisa.engine.sync`` /
        ``nisa.engine.scalar`` -- plus the forwarded ``NDSlice.store`` checks.

        Example:
            .. code-block:: python

                for k in nl.affine_range(K):
                    nisa.tensor_copy(out_stream[k].data, psum_acc)
                    out_stream.store(k)

        Notes:
            The slot is selected by ``index % buffer_count``, so storing position ``k`` after
            that slot has been reused by a later step writes the later step's data. Store a
            position before its slot is overwritten.

        See Also:
            __getitem__: get the slot to fill.
            load: the read mirror.

        """
        self._ensure_buffers_for_access()
        sbuf = self._buffers[index % self._buffer_count]
        child = self._child(index)
        child.store(
            SBUFLayout.contiguous_ap(sbuf),
            oob_mode=oob_mode,
            dge_mode=dge_mode,
            priority=priority,
            pattern_override=self._pattern_override,
            engine=engine,
        )
