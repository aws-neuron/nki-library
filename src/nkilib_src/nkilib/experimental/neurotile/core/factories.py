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
from typing import Any, Optional

import nki.language as nl

from ._helpers import (
    buffer_space,
    contiguous_strides,
    hbm_buffer_type,
    physical_strides,
    product,
)
from ._validation import (
    _validate_alloc_blocks_args,
    _validate_alloc_tiles_args,
    _validate_block_size,
    _validate_block_size_rank,
    _validate_source,
    _validate_tiles_args,
)
from .axis import Axis, AxisLabel
from .grid import Grid
from .layout_hbm import HBMLayout
from .layout_psum import PSUMLayout
from .layout_sbuf import SBUFLayout
from .ndslice import NDSlice

# ============================================================================
# Helpers
# ============================================================================


def _pad_tile_size(size, n_batch_dims):
    """Pad tile_size with 1s for leading iteration dims.

    Example: tile_size=(128, 64) on 3D tensor -> (1, 128, 64), n_batch_dims=1.
    """
    if n_batch_dims <= 0:
        return tuple(size)
    padded = []
    for d in range(n_batch_dims):
        padded.append(1)
    for d in range(len(size)):
        padded.append(size[d])
    return tuple(padded)


def _truncate_for_skip(remaining, tile_size):
    """Truncate remaining to evenly divisible (RemainderPolicy.SKIP)."""
    result = []
    for d in range(len(remaining)):
        if d < len(tile_size) and tile_size[d] > 0:
            result.append((remaining[d] // tile_size[d]) * tile_size[d])
        else:
            result.append(remaining[d])
    return tuple(result)


def _resolve_source(source):
    """Normalize a raw tensor source to ``(storage, offset, dtype)``.

    A sliced/transformed source self-addresses: its handle already embeds the
    offset, so the layout reads it at offset 0 (re-applying ``source.offset``
    would double-count it).
    """
    return source, 0, source.dtype


# ============================================================================
# SBUF view construction -- single path for all SBUF Grid+Layout creation
# ============================================================================


def _make_sbuf_ndslice(sbuf_source, element_shape, tile_size, dtype, offset=0):
    """Create NDSlice(Grid, SBUFLayout) for an SBUF source.

    Caller supplies the addressable per-dim ``element_shape`` and the
    per-tile ``tile_size`` (uniform allocation extent). Returns the
    wrapped NDSlice.
    """
    grid, layout = SBUFLayout.build_view(
        sbuf_source,
        element_shape,
        tile_size,
        dtype,
        nl.sbuf,
        offset,
    )
    return NDSlice(grid, layout)


# ============================================================================
# PSUM view construction
# ============================================================================


def _make_psum_ndslice(
    tile_arrays,
    alloc_tile_size,
    grid_shape,
    bank_axis,
    slots_per_bank,
    dtype,
    element_shape=None,
):
    """Create NDSlice(Grid, PSUMLayout) for per-tile PSUM ndarrays.

    Args:
        tile_arrays: tuple of ``nl.ndarray(buffer=nl.psum)`` -- one per
            grid tile, in bank-major order (tile index =
            ``bank_idx * slots_per_bank + slot_idx``).
        alloc_tile_size: per-tile (P, F) shape.
        grid_shape: tile-grid shape (one count per dim of alloc_tile_size).
            ``grid_shape[bank_axis]`` is the number of banks; the product
            of the rest is ``slots_per_bank``.
        bank_axis: which Grid dim spreads tiles across PSUM banks.
        slots_per_bank: number of tiles co-located on each bank.
        dtype: element dtype.
        element_shape: optional per-dim addressable element extent. When
            omitted, defaults to ``grid_shape * alloc_tile_size`` (full
            tile-aligned extent).

    Returns:
        NDSlice wrapping a PSUMLayout + Grid.
    """
    if element_shape is None:
        es = []
        for d in range(len(grid_shape)):
            es.append(grid_shape[d] * alloc_tile_size[d])
        element_shape = tuple(es)
    grid = Grid.from_shape(
        element_shape=tuple(element_shape),
        tile_size=tuple(alloc_tile_size),
    )
    layout = PSUMLayout(
        tile_arrays=tile_arrays,
        offset=0,
        alloc_tile_size=tuple(alloc_tile_size),
        bank_axis=bank_axis,
        slots_per_bank=slots_per_bank,
        grid_shape=tuple(grid_shape),
        dtype=dtype,
        buffer_type=nl.psum,
    )
    return NDSlice(grid, layout)


# ============================================================================
# tiles()
# ============================================================================


def tiles(
    source: Any,
    tile_size: Optional[tuple] = None,
    access_pattern: Optional[list] = None,
    remainder: Optional[str] = None,
) -> "NDSlice":
    """tiles(source, tile_size=None, access_pattern=None, remainder=None) -> NDSlice

    Decompose a tensor into a logical grid of fixed-size tiles. No data is moved -- the
    grid is a logical *view*, indexed by tile coordinate (``view[i, j]``) rather than by
    hand-computed offsets and strides, and sharded across cores by slicing it. The
    buffer space (HBM or SBUF) is detected from the source, so an HBM tensor and a raw
    SBUF ``nl.ndarray`` are tiled the same way; a PSUM source is rejected (operate on a
    ``nt.psum_pool()`` bank's ``.data`` directly instead).

    This is the starting point for most kernels. ``tile_size`` defines the *compute
    grain* -- the shape each ISA instruction
    (``nisa.nc_matmul``, ``nisa.tensor_*``) operates on once a tile is loaded. When
    several tiles should move in a single DMA while retaining that compute grain,
    they are grouped with ``nt.blocks()``, which establishes a separate, coarser
    *DMA grain*.

    For example, ``tile_size=(128, 256)`` over a ``(512, 1024)`` tensor produces a
    ``4 x 4`` grid (``shape == (4, 4)``); each tile spans ``128`` elements on the P
    (partition) axis and ``256`` on the F (free) axis. ``tiles[i, j]`` selects one
    cell, ``tiles[i]`` a whole row, and ``tiles[:, j]`` a whole column::

                  256       256       256       256      <- F (free): 256 per tile
           ┌─────────┬─────────┬─────────┬─────────┐
        P  │  (0,0)  │  (0,1)  │  (0,2)  │  (0,3)  │
        |  ├─────────┼─────────┼─────────┼─────────┤
        v  │  (1,0)  │  (1,1)  │  (1,2)  │  (1,3)  │
           ├─────────┼─────────┼─────────┼─────────┤
           │  (2,0)  │  (2,1)  │  (2,2)  │  (2,3)  │
           ├─────────┼─────────┼─────────┼─────────┤
           │  (3,0)  │  (3,1)  │  (3,2)  │  (3,3)  │
           └─────────┴─────────┴─────────┴─────────┘
        P (partition): 128 per tile

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        source (HBM tensor | nl.ndarray | NDSlice): The tensor to tile. A top-level
            HBM tensor; a sliced or transformed tensor (tiled directly using its own
            self-describing layout); a raw SBUF ``nl.ndarray`` (the SBUF space is
            detected from the source); or an existing ``NDSlice`` (re-tiled with
            ``tile_size=``, otherwise its block grouping is dropped). Required.
        tile_size (tuple[int, ...]): The per-tile shape, at least 2-D
            ``(P, F[, ...])``, every entry a positive int. Defines the *compute grain*.
            Required for a raw source; optional for an ``NDSlice``. Fewer dims than
            ``source`` makes the leading dims batch dims (see *Batch dimensions*). A
            ``tile_size`` larger than the source extent on a dim is accepted and yields
            a single partial tile there.
        access_pattern (list[[int, int]] | None): The source layout as
            ``[[stride, count], ...]``, one level per view dim, every entry a positive
            int. Its counts become the view's ``element_shape`` and its strides become
            the layout strides, overriding the source's own shape and physical strides.
            Raw HBM sources only -- a raw SBUF source rejects it (reshape / permute the
            view after construction instead). Defaults to the source's contiguous
            layout.
        remainder (str | None): ``"skip"`` floor-divides the grid (drops partial
            trailing tiles); None (default) keeps them, surfaced via
            ``view.is_remainder``. See *Remainder handling*.

    Returns:
        NDSlice: A tile-grid view over ``source`` -- see ``NDSlice`` for the view's
        attributes and operations. Indexing it selects tiles (``view[i, j]``); slicing
        it shards across cores (``nt.tiles(...)[nt.block_range(rank, n, total), :]``).

    Raises ``AssertionError`` in any of these cases:

    - ``source`` is None or is not a tensor-like object with a ``.shape`` (nor an
      ``NDSlice``).
    - ``source`` already carries a runtime (gather / dynamic-select) offset
      (apply the runtime index on the view instead: ``nt.tiles(t, ...)[k]``).
    - ``tile_size`` is missing on a raw source, is not a tuple/list of positive
      ints, has fewer than 2 dims, or its rank exceeds the view rank (the
      ``access_pattern`` level count if given, else the source rank).
    - ``access_pattern`` is malformed (not a list of positive ``[stride, count]``
      pairs) or its largest addressed offset reaches past the source's end.
    - a higher-rank ``access_pattern`` is paired with a ``tile_size`` that does not
      cover the whole view in one tile (only single-tile coverage is supported
      today).
    - the ``source`` resides in PSUM (a PSUM source is rejected -- operate on a
      ``nt.psum_pool()`` bank's ``.data`` directly).
    - ``access_pattern`` is given for an SBUF source (an SBUF source rejects
      ``access_pattern`` -- reshape / permute the view after construction instead).
    - an ``NDSlice`` source is combined with ``access_pattern`` or ``remainder``
      (both raw-source-only).

    Example:
        .. code-block:: python

            # Plain tile grid on a top-level HBM tensor: (512, 1024) -> 4x4 grid.
            src_tiles = nt.tiles(src, tile_size=(128, 256))
            for i in range(src_tiles.shape[0]):
                for j in range(src_tiles.shape[1]):
                    tile = src_tiles[i, j].load()

            # Shard across cores by slicing the view.
            own = nt.block_range(nl.program_id(0), nl.num_programs(0), src_tiles.shape[0])
            local = nt.tiles(src, tile_size=(128, 256))[own, :]

            # Strided source layout (every other row).
            view = nt.tiles(src, access_pattern=[[2 * N, M // 2], [1, N]], tile_size=(64, N))

    Source kinds
    ------------
    The operation depends on what ``source`` is and which other arguments are set::

        source             other args             result
        -----------------  ---------------------  --------------------------------
        raw HBM tensor     tile_size= (required)  tile-grid view over HBM
        raw HBM tensor     + access_pattern=      strides taken from the AP
        raw HBM tensor     + remainder="skip"     grid floor-divided (no partials)
        raw SBUF ndarray   tile_size= (required)  tile-grid view over SBUF
        sliced view        tile_size= (required)  strides/offset from the slice itself
        NDSlice            (no tile_size=)        drop the block grouping, keep tiles
        NDSlice            tile_size=             re-tile: rebuild grid at new size

    **Re-tile versus drop-block.** Both ``NDSlice`` forms return a view with
    ``block_size is None``; passing ``tile_size=`` rebuilds the tile grid at the new
    size, while omitting it keeps the existing tile structure. Outer shard / broadcast
    axes are preserved across both, and HBM and SBUF behave symmetrically.

    **SBUF sources.** An SBUF source is recognized from the source itself -- the
    canonical form is an ``NDSlice`` already carrying an SBUF layout (it supplies the
    buffer, offset, and dtype), and a raw SBUF ``nl.ndarray`` is accepted with offset 0.
    A raw *sliced* SBUF view is tiled directly from its own layout, the same as an HBM
    slice.

    Batch dimensions
    ----------------
    A ``tile_size`` with fewer dims than ``source`` is left-padded with 1s, turning the
    leading source dims into batch dims (each iterated over one slab at a time). The
    view's ``shape`` then leads with the batch extents -- a 3-D source with a 2-D
    ``tile_size`` gives ``shape == (B, M // tile_size[0], N // tile_size[1])``, where
    the first index selects a batch slab and the rest address the tile grid. Batch dims
    are indexed away (``view[b, i, j]``) or iterated over before ``.load()`` /
    ``.store()`` / ``.stream()``, which assert all batch dims are gone. ``tile_size``
    must be at least 2-D: write ``(N, 1)`` for a P-column tile or ``(1, N)`` for an
    F-row tile, never ``(N,)``.

    Access patterns
    ---------------
    The ``access_pattern`` level count is the *view's* rank, independent of the source
    rank: a 3-level AP on a 2-D source yields a 3-D view, and a 1-level AP flattens an
    N-D source. The source need only hold the AP's largest addressed offset. A
    higher-rank AP (more levels than the source rank) is supported only for single-tile
    coverage today -- the padded ``tile_size`` must equal the view shape exactly;
    multi-tile iteration over such a view is not yet supported.

    Remainder handling
    ------------------
    ``remainder="skip"`` truncates each tiled dim's extent down to an exact multiple of
    the padded ``tile_size`` before the grid is built (a per-dim floor-divide); batch
    dims and dims with no tile entry keep their full extent. The default (None) keeps
    the partial trailing tiles, each surfaced through ``view.is_remainder`` for the
    boundary DMA policy (see ``load`` / ``whole_tiles`` / ``remainder_tiles``).

    Memory layout
    -------------
    ``tile_size`` maps position-by-position onto the source's trailing dims: dim 0 is
    the SBUF partition (P) axis, and the rest are free (F) subdivisions kept as logical
    structure (not flattened), so a 3-D tile stays 3-D after ``.load()``. The physical
    SBUF backing is always 2-D ``(P, F_flat)`` -- a ``(P, F1, F2)`` tile is stored as
    ``(P, F1 * F2)`` and ``tile.data.shape`` reports the 2-D shape -- while the logical
    shape after ``.load()`` equals the tile's ``element_shape`` at load time::

        tile_size      source         batch dims   SBUF after .load()
        -------------  -------------  -----------  ------------------
        (P, F)         (M, N)         0            (P, F)
        (P, F)         (B, M, N)      1 (= B)      (P, F)
        (P, F1, F2)    (M, N1, N2)    0            (P, F1, F2)
        (P, 1)         (M, N)         0            (P, 1)
        (1, F)         (M, N)         0            (1, F)

    Sub-tile slicing (for example ``view[i, j][:, 0:64]``) narrows the tile's
    ``element_shape`` while preserving ``tile_size``; the P axis stays on dim 0 and can
    be narrowed but not rotated onto another dim. To map a different source dim onto P,
    reshape or permute the source first, or use ``load(transpose=True)``.

    See Also:
        blocks: coarser DMA grain (coalesced multi-tile DMA) over the same tiles.
        alloc_tiles: allocate a fresh tiled SBUF / HBM buffer.

    """
    _validate_tiles_args(
        source=source,
        size=tile_size,
        access_pattern=access_pattern,
        remainder=remainder,
    )

    if isinstance(source, NDSlice):
        assert access_pattern is None, (
            "nt.tiles(source=NDSlice): access_pattern= is not supported -- "
            "the NDSlice already carries its own layout. Pass the raw "
            "tensor with access_pattern= instead."
        )
        assert remainder is None, (
            "nt.tiles(source=NDSlice, remainder=...): remainder= applies at "
            "construct time only. Re-tile inherits the source view's "
            "remaining region."
        )
        return _retile_ndslice(source, tile_size)

    return _build_tiled_ndslice(
        source=source,
        size=tile_size,
        access_pattern=access_pattern,
        remainder=remainder,
    )


def _retile_ndslice(source, tile_size):
    """Handle ``nt.tiles(source=NDSlice)`` -- descend or re-tile.

    Both paths strip block axes first (re-tiling a block view descends
    to tile granularity) and then reset the cursor to the first
    non-batch dim so iteration starts on the resulting tile grid.

    - ``tile_size is None``  -> descend: drop block axes, keep the
      view's existing tile structure.
    - ``tile_size`` given    -> retile: drop block axes, rebuild the
      tile grid at the new ``tile_size`` via ``Grid.tile()``. Outer
      shard / broadcast axes are preserved.

    The Layout is asked to ``retile`` only when ``tile_size`` is given.
    HBMLayout returns self (HBM strides are source-element units,
    independent of tile grain); SBUFLayout recomputes its strides for
    the new tile grain.
    """
    if tile_size is None and source._grid.tile_size is None:
        return source

    base_grid = source._grid.strip_block()

    if tile_size is None:
        new_grid = base_grid.with_cursor(base_grid.n_batch_dims)
        return NDSlice(new_grid, source._layout)

    new_tile_size = tuple(tile_size)
    source._grid.validate_retile_to(new_tile_size)
    new_grid = base_grid.tile(new_tile_size).with_cursor(base_grid.n_batch_dims)
    new_layout = source._layout.retile(new_tile_size, new_grid.remaining)
    return NDSlice(new_grid, new_layout)


def _split_access_pattern(access_pattern):
    """Split AP into (strides, element_shape) tuples.

    access_pattern is `[[stride_0, count_0], ...]` -- stride_d is the element
    stride along dim d; count_d is the element extent along dim d.
    """
    strides = []
    element_shape = []
    for level in access_pattern:
        strides.append(level[0])
        element_shape.append(level[1])
    return tuple(strides), tuple(element_shape)


def _resolve_element_shape(source, access_pattern):
    """Pick the canonical element_shape.

    When `access_pattern` is given, its counts supply the element extents;
    otherwise fall back to `source.shape`.
    """
    if access_pattern is not None:
        _, ap_shape = _split_access_pattern(access_pattern)
        return ap_shape
    return tuple(source.shape)


def _build_tiled_ndslice(
    source,
    size,
    access_pattern,
    remainder,
):
    """Single-pipeline NDSlice builder for raw-tensor sources.

    All composition axes flow through the same steps:
      1. Resolve the memory space from the source; normalize to
         (storage tensor, offset, dtype).
      2. Resolve element_shape (from AP counts / source.shape).
      3. Resolve strides (from AP if given, else physical layout).
      4. Apply remainder policy to compute `remaining`.
      5. Build Grid + Layout.
      6. Drop unit batch dims, return NDSlice.
    """
    # Step 1: route by memory space, detected from the source. tiles/blocks
    # build HBM and SBUF views; a PSUM source is rejected -- PSUM accumulators
    # come from nt.psum_pool(); operate on a bank's .data directly.
    space = buffer_space(source)
    assert space != nl.psum, (
        "nt.tiles/nt.blocks: PSUM sources are not supported. PSUM accumulators "
        "are produced by nt.psum_pool(); operate on a bank's .data directly."
    )
    is_sbuf = space == nl.sbuf

    # Normalize source -> (storage tensor, offset, dtype). A sliced/transformed
    # source self-addresses (offset 0); same rule for HBM and SBUF.
    storage, offset, dtype = _resolve_source(source)

    # Step 2: element_shape. AP counts override source.shape.
    element_shape = _resolve_element_shape(source, access_pattern)

    # Step 3: strides. AP overrides physical layout.
    # Strides come from the source because a leading-index slice like weights[i]
    # reduces rank -- the slice's get_pattern() reports correct rank-matched
    # strides.
    if access_pattern is not None:
        strides, _ = _split_access_pattern(access_pattern)
    elif is_sbuf:
        strides = None  # SBUFLayout computes strides from tile grid; see below.
    else:
        strides = physical_strides(source, source.shape)

    # Step 4: tile size + remainder policy.
    tile_size = tuple(size)
    ndim = len(element_shape)
    n_batch_dims = ndim - len(tile_size)
    padded_tile_size = _pad_tile_size(tile_size, n_batch_dims)

    effective_element_shape = element_shape
    if remainder == "skip":
        effective_element_shape = _truncate_for_skip(element_shape, padded_tile_size)

    # Step 5: Grid + Layout.
    grid = Grid.from_shape(
        element_shape=effective_element_shape,
        tile_size=padded_tile_size,
        n_batch_dims=n_batch_dims,
    )

    if is_sbuf:
        view = _make_sbuf_ndslice(
            storage,
            element_shape,
            padded_tile_size,
            dtype,
            offset=offset,
        )
        layout = view._layout
        grid = view._grid if remainder != "skip" else grid
    else:
        layout = HBMLayout(
            source=storage,
            offset=offset,
            strides=strides,
            dtype=dtype,
            buffer_type=hbm_buffer_type(),
        )

    return NDSlice(grid, layout)


# ============================================================================
# blocks()
# ============================================================================


def blocks(
    source: Any,
    block_size: tuple,
    tile_size: Optional[tuple] = None,
    access_pattern: Optional[list] = None,
) -> "NDSlice":
    """blocks(source, block_size, tile_size=None, access_pattern=None) -> NDSlice

    Group a tensor's tiles into fixed-size rectangular blocks. No data is moved -- the
    block grid is a logical *view*, indexed by block coordinate (``blocks[bi, bj]``),
    that separates the *DMA grain* from the *compute grain*. The buffer space (HBM or
    SBUF) is detected from the source, so a raw SBUF ``nl.ndarray`` and an HBM tensor
    are block-tiled the same way; a PSUM source is rejected (operate on a
    ``nt.psum_pool()`` bank's ``.data`` directly instead).

    The block is the unit that moves between HBM and SBUF, while the underlying
    ``tile_size`` remains the unit each ISA instruction operates on. Three properties
    follow:

    * **Coalesced DMA.** A whole block transfers in a single ``nisa.dma_copy``
      rather than one DMA per tile, which reduces descriptor count and DMA-queue
      pressure and replaces many small strided transfers with one larger, more
      contiguous transfer.
    * **Compute grain preserved.** After a block is loaded, its individual
      ``tile_size`` tiles are still addressed and computed on independently
      (``block[ti, tj]``); the coarser DMA does not force a coarser matmul or
      tensor-op shape.
    * **Reuse and pipelining.** A resident block feeds many compute steps from SBUF,
      and ``blocks[:, b].stream()`` double-buffers one block per step.

    ``blocks.shape`` reports the block grid; indexing a block (``blocks[bi, bj]``)
    yields the tile grid inside that block. For example,
    ``block_size=(2, 2)`` over a ``4 x 4`` tile grid gives a ``2 x 2`` block grid; each
    block (heavy ``┏━┓`` box) is the DMA grain, while the tiles inside (light ``┌─┐``
    cells) remain the compute grain::

           block (0,0)         block (0,1)
           ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
        P  ┃ ┌─────┐ ┌─────┐ ┃ ┌─────┐ ┌─────┐ ┃
        |  ┃ │(0,0)│ │(0,1)│ ┃ │(0,2)│ │(0,3)│ ┃
        v  ┃ └─────┘ └─────┘ ┃ └─────┘ └─────┘ ┃
           ┃ ┌─────┐ ┌─────┐ ┃ ┌─────┐ ┌─────┐ ┃
           ┃ │(1,0)│ │(1,1)│ ┃ │(1,2)│ │(1,3)│ ┃
           ┃ └─────┘ └─────┘ ┃ └─────┘ └─────┘ ┃
           ┣━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━┫
           ┃ ┌─────┐ ┌─────┐ ┃ ┌─────┐ ┌─────┐ ┃
           ┃ │(2,0)│ │(2,1)│ ┃ │(2,2)│ │(2,3)│ ┃
           ┃ └─────┘ └─────┘ ┃ └─────┘ └─────┘ ┃
           ┃ ┌─────┐ ┌─────┐ ┃ ┌─────┐ ┌─────┐ ┃
           ┃ │(3,0)│ │(3,1)│ ┃ │(3,2)│ │(3,3)│ ┃
           ┃ └─────┘ └─────┘ ┃ └─────┘ └─────┘ ┃
           ┗━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━┛
           block (1,0)         block (1,1)
        P = partition axis (down), F = free axis (across); cells are global tile coords

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        source (HBM tensor | nl.ndarray | NDSlice): The tensor to block-tile. The
            accepted source kinds are the same as for ``tiles()`` -- a top-level or
            sliced HBM tensor, a raw SBUF ``nl.ndarray``, or an existing ``NDSlice``.
            Required.
        block_size (tuple[int, ...]): The number of tiles per block along each
            dimension, at least 2-D, every entry a positive int. Required. A per-dim
            entry of 1 still groups that dimension (one tile per block there). Its rank
            must not exceed the source rank -- for an ``NDSlice`` source this is the
            view's current ``ndim``, for a raw source it is ``len(source.shape)``.
        tile_size (tuple[int, ...] | None): The per-tile (compute-grain) shape (at
            least 2-D). Required for a raw tensor source; for an ``NDSlice`` source it
            is inherited unless given, in which case the view is re-tiled before it is
            grouped into blocks.
        access_pattern (list[[int, int]] | None): The source layout as
            ``[[stride, count], ...]``. Raw HBM sources only -- a raw SBUF source
            rejects it (reshape / permute the view after construction instead).

    There is no ``buffer_type=`` parameter (the space is detected from the source) and
    no ``remainder=`` parameter: unlike ``tiles()``, partial-tile policy
    cannot be chosen at ``blocks()`` construction time, and boundary tiles are always
    kept (surfaced via ``view.is_remainder``). To drop partial tiles, build the tile
    view with ``nt.tiles(..., remainder="skip")`` first and group it into blocks.

    Returns:
        NDSlice: A block-grid view over ``source`` -- see ``NDSlice`` for the view's
        attributes and operations. Indexing a block (``blocks[bi, bj]``) yields the
        tile grid inside it (``block[ti, tj]``); slicing the block grid shards work
        at block granularity across cores.

    Raises ``AssertionError`` in any of these cases:

    - ``block_size`` is missing, is not a tuple/list of positive ints, has fewer
      than 2 dims, or its rank exceeds the source rank.
    - ``source`` already carries a runtime (gather / dynamic-select) offset
      (apply the runtime index on the view instead: ``nt.tiles(t, ...)[k]``).
    - ``tile_size`` is missing on a raw source.
    - an ``NDSlice`` source is combined with ``access_pattern`` (raw-source-only).
    - ``access_pattern`` is given for an SBUF source (an SBUF source rejects
      ``access_pattern``).
    - ``source`` resides in PSUM (PSUM accumulators come from ``nt.psum_pool()``;
      operate on a bank's ``.data`` directly).
    - any ``tiles()`` rule on ``tile_size`` / ``access_pattern`` is violated.

    Example:
        .. code-block:: python

            # 128x512 tiles grouped into 2x2-tile blocks; coalesced load, per-tile
            # compute, coalesced store -- the canonical block loop.
            src_blocks = nt.blocks(src, tile_size=(128, 512), block_size=(2, 2))
            dst_blocks = nt.blocks(dst, tile_size=(128, 512), block_size=(2, 2))
            for bi in range(src_blocks.shape[0]):
                for bj in range(src_blocks.shape[1]):
                    block = src_blocks[bi, bj].load()      # one DMA -> 2x2 tiles in SBUF
                    for ti in range(block.shape[0]):       # iterate the interior tile grid
                        for tj in range(block.shape[1]):
                            tile = block[ti, tj]           # one tile_size tile -- no DMA
                            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
                    dst_blocks[bi, bj].store(block.data)    # one DMA writes the block back

            # Promote an existing (already sharded) tile view to block granularity.
            tv = nt.tiles(src, tile_size=(128, 512))[own, :]
            blks = nt.blocks(tv, block_size=(2, 2))

    Source kinds
    ------------
    The operation depends on what ``source`` is::

        source             other args                  result
        -----------------  --------------------------  ---------------------------
        raw HBM tensor     tile_size=, block_size=     block view over HBM
        raw SBUF ndarray   tile_size=, block_size=     block view over SBUF
        NDSlice            block_size= only            group its tiles into blocks
        NDSlice            block_size= + tile_size=    re-tile, then group into blocks

    ``block_size`` is mandatory; for a tile view with no block grouping, use
    ``nt.tiles()``. Every ``tiles()`` argument rule applies here as well.

    How blocks are formed
    ---------------------
    For a raw source, the tile grid is formed first (exactly as ``tiles()`` would, with
    the same ``tile_size`` / ``access_pattern``) and its tiles are then
    grouped into blocks of ``block_size`` tiles. For an ``NDSlice``
    source, ``block_size`` alone groups the existing tile grid, and ``block_size`` with
    ``tile_size`` re-tiles first and then groups; shard / broadcast structure is
    preserved either way. The block-grid extent on a dimension is
    ``ceil(tile_grid[dim] / block_size[dim])``.

    See Also:
        tiles: a tile view with no block grouping.
        alloc_blocks: allocate a fresh block-structured buffer.

    """
    _validate_source(source, caller="nt.blocks")
    assert block_size is not None, "nt.blocks(...): block_size= is required. For a tile-level view, use nt.tiles()."
    _validate_block_size(block_size)
    _validate_block_size_rank(block_size, source)

    # Raw sources require tile_size= (NDSlice inherits its tile_size). Check
    # here so the error names blocks(), not the forwarded tiles() call.
    if not isinstance(source, NDSlice):
        assert tile_size is not None, (
            "nt.blocks(<raw source>, ...): tile_size= is required for raw "
            "tensor sources. For an existing NDSlice, tile_size= is "
            "optional (inherited from the view)."
        )

    # Validate tile_size / AP with the same rules as nt.tiles(). tiles() is
    # re-called for the raw-source path below, but that dispatcher checks only
    # its own args -- so validate explicitly here to cover the NDSlice case too.
    _validate_tiles_args(
        source=source,
        size=tile_size,
        access_pattern=access_pattern,
        remainder=None,
    )

    # --- NDSlice source: promote / re-tile + promote ---
    if isinstance(source, NDSlice):
        assert access_pattern is None, "nt.blocks(NDSlice, ...): access_pattern= is only for raw sources."

        view = _retile_ndslice(source, tile_size)
        return _prepend_block_level(view, block_size)

    # --- Raw source: build tile-level view -> prepend block axis ---
    tile_view = tiles(
        source,
        tile_size=tile_size,
        access_pattern=access_pattern,
    )
    return _prepend_block_level(tile_view, block_size)


def _prepend_block_level(view, block_size):
    """Prepend a block axis above the outermost tile axis on each non-batch dim.

    Delegates to Grid.with_block, which inserts a block axis with
    step = block_size * tile_step on each dim with bs >= 1, preserving
    any outer shard / broadcast axes.
    """
    grid = view._grid
    padded_block_size = _pad_tile_size(block_size, grid.n_batch_dims)
    # Strip leading batch padding (with_block reads tail offset).
    block_for_dims = padded_block_size[grid.n_batch_dims :]
    new_grid = grid.with_block(block_for_dims)
    return NDSlice(new_grid, view._layout)


# ============================================================================
# alloc -- stubs
# ============================================================================


def alloc_tiles(
    tile_size: tuple[int, ...],
    grid: Optional[tuple[int, ...]] = None,
    buffer_type: Optional[nl.MemoryRegion] = None,
    dtype=None,
    element_shape: Optional[tuple[int, ...]] = None,
) -> NDSlice:
    """alloc_tiles(tile_size, grid=None, buffer_type=None, dtype=None, element_shape=None) -> NDSlice

    Allocate a fresh tiled SBUF or HBM buffer and return an ``NDSlice`` over it.

    This is the buffer behind output accumulators and scratch space: it is addressed
    by tile coordinate, with no hand-sizing of the buffer or hand-computed per-tile
    offsets.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        tile_size (tuple[int, ...]): The per-tile shape (at least 2-D), positive ints.
            Dimension 0 maps to the SBUF partition (P) axis.
        grid (tuple[int, ...] | None): The tile-grid shape, one count per ``tile_size``
            dimension. Mutually exclusive with ``element_shape``.
        buffer_type (nl.MemoryRegion): ``nl.sbuf`` for SBUF; ``nl.shared_hbm`` /
            ``nl.private_hbm`` for HBM. Required. ``nl.psum`` is rejected -- PSUM
            buffers come from ``nt.psum_pool()``.
        dtype: The element data type (for example ``nl.bfloat16``). Required.
        element_shape (tuple[int, ...] | None): The exact logical extent per dimension,
            ceiling-divided into the tile grid. Mutually exclusive with ``grid``.

    Returns:
        NDSlice: A tile-grid view over the freshly allocated buffer (newly created, not
        a view over caller data) -- see ``NDSlice`` for the view's attributes.
        ``acc[m, n].data`` is the per-tile ``nl.ndarray`` (an ISA-op operand) and
        ``acc.data`` is the view over the whole buffer (for example, for one
        coalesced store).

    Raises ``AssertionError`` in any of these cases:

    - ``buffer_type`` or ``dtype`` is missing.
    - ``buffer_type`` is ``nl.psum`` or not an ``nl.MemoryRegion``.
    - both ``grid`` and ``element_shape`` are given.
    - ``tile_size`` is not a >= 2-D positive-int tuple.
    - ``grid`` / ``element_shape`` rank or entries disagree with ``tile_size``, or any
      entry is non-positive or a bool (a ``True`` / ``False`` entry is rejected even
      though Python treats bool as an int).

    Example:
        .. code-block:: python

            # Tile-aligned SBUF accumulator, then zero it.
            acc = nt.alloc_tiles(tile_size=(128, 512), grid=(4, 2), buffer_type=nl.sbuf, dtype=nl.float32)
            nisa.memset(acc.data, 0.0)

            # Exact-extent buffer whose last F-tile is a 256-wide partial (1792 = 3*512 + 256).
            out = nt.alloc_tiles(tile_size=(128, 512), element_shape=(128, 1792),
                                 buffer_type=nl.shared_hbm, dtype=nl.bfloat16)

    Sizing: grid versus element_shape
    ---------------------------------
    ``grid`` allocates exactly ``grid * tile_size`` elements, which is tile-aligned.
    ``element_shape`` allocates the exact extent and ceiling-divides it into the tile
    grid, so the last tile along a dimension is a partial tile when the extent is not
    divisible -- and the returned view then reports ``is_remainder == True`` on that
    dim, so a caller detects the partial last tile without re-deriving the division (a
    tile-aligned ``grid`` allocation reports ``is_remainder == False``).
    ``element_shape`` is preferred unless the extent is known tile-aligned. With both
    omitted, the result is a single-tile allocation.

    Memory layout
    -------------
    An SBUF allocation is physically a flat 2-D ``nl.ndarray`` of shape
    ``(tile_size[0], total_tiles * tile_F)``, where ``tile_F`` is the product of
    ``tile_size[1:]`` and ``total_tiles`` counts the whole grid: every P-tile folds into
    the free (F) columns, so the physical partition extent never exceeds one tile's P
    dimension and a larger ``element_shape[0]`` widens F, never the partition axis. An
    HBM allocation is laid out row-major contiguous over the ``element_shape`` (which
    defaults to ``grid * tile_size`` per dim when ``grid`` is given). With both ``grid``
    and ``element_shape`` omitted, the single tile is element-addressable so a
    subsequent ``.data`` covers the whole tile; pass ``grid=(1, 1, ...)`` instead to
    keep tile-level navigation granularity.

    See Also:
        alloc_blocks: allocate a buffer whose tiles are grouped into blocks.
        psum_pool: allocate PSUM accumulator banks.

    """
    _validate_alloc_tiles_args(tile_size, grid, element_shape, buffer_type, dtype)

    tile_size = tuple(tile_size)
    tile_p = tile_size[0]
    tile_f = product(tile_size, start=1)

    # Derive grid from element_shape if not provided
    if element_shape is not None:
        element_shape = tuple(element_shape)
        if grid is None:
            tile_shape = []
            for d in range(len(tile_size)):
                if d < len(element_shape):
                    tile_shape.append((element_shape[d] + tile_size[d] - 1) // tile_size[d])
                else:
                    tile_shape.append(1)
            tile_shape = tuple(tile_shape)
        else:
            tile_shape = tuple(grid)
    else:
        # Tile grid: (1, 1, ...) for single tile, or user-specified
        if grid is None:
            tile_shape = []
            for d in range(len(tile_size)):
                tile_shape.append(1)
            tile_shape = tuple(tile_shape)
        else:
            tile_shape = tuple(grid)

    total_tiles = 1
    for g in tile_shape:
        total_tiles = total_tiles * g

    # Allocate buffer
    if buffer_type == hbm_buffer_type():
        # HBM allocation
        if element_shape is None:
            element_shape = []
            for d in range(len(tile_size)):
                if d < len(tile_shape):
                    element_shape.append(tile_shape[d] * tile_size[d])
                else:
                    element_shape.append(tile_size[d])
            element_shape = tuple(element_shape)

        hbm = nl.ndarray(element_shape, dtype=dtype, buffer=buffer_type)
        strides = contiguous_strides(element_shape)
        alloc_grid = Grid.from_shape(element_shape, tile_size)
        layout = HBMLayout(hbm, 0, strides, dtype, buffer_type)
        return NDSlice(alloc_grid, layout)
    else:
        # SBUF / PSUM allocation -- flat 2D (P-tiles fold into F-columns)
        if element_shape is not None:
            p_tiles = (element_shape[0] + tile_p - 1) // tile_p
            actual_f = product(element_shape, start=1) * p_tiles
            sbuf = nl.ndarray((tile_p, actual_f), dtype=dtype, buffer=buffer_type)
            view_element_shape = element_shape
        else:
            sbuf = nl.ndarray((tile_p, total_tiles * tile_f), dtype=dtype, buffer=buffer_type)
            view_element_shape = []
            for d in range(len(tile_size)):
                if d < len(tile_shape):
                    view_element_shape.append(tile_shape[d] * tile_size[d])
                else:
                    view_element_shape.append(tile_size[d])
            view_element_shape = tuple(view_element_shape)
        grid_obj, layout = SBUFLayout.build_view(
            sbuf,
            view_element_shape,
            tile_size,
            dtype,
            buffer_type,
        )

        # grid=None (single tile, no grid): collapse the tile axis into a
        # single elem axis whose count is the full per-dim element extent.
        # The walk is unchanged (one tile == full extent); the descent shape
        # changes from "tile-level (count=1)" to "element-level (count=ext)"
        # so that downstream `.data` addresses the whole tile.
        # grid=(M,N) (explicit grid, even 1x1): keep tile level for navigation.
        if grid is None and element_shape is None:
            _elem_axes = []
            for _d in range(grid_obj.ndim):
                _ext = grid_obj.element_shape[_d]
                _elem_axes.append(Axis(count=_ext, step=1, dim=_d, label=AxisLabel.ELEM))
            grid_obj = Grid(
                element_shape=grid_obj.element_shape,
                axes=tuple(_elem_axes),
                cursor=grid_obj.cursor,
                n_batch_dims=grid_obj.n_batch_dims,
                tiled=grid_obj.tiled,
            )

        return NDSlice(grid_obj, layout)


def alloc_blocks(
    tile_size: tuple[int, ...],
    block_size: tuple[int, ...],
    grid: Optional[tuple[int, ...]] = None,
    buffer_type: Optional[nl.MemoryRegion] = None,
    dtype=None,
    element_shape: Optional[tuple[int, ...]] = None,
) -> NDSlice:
    """alloc_blocks(tile_size, block_size, grid=None, buffer_type=None, dtype=None, element_shape=None) -> NDSlice

    Allocate a fresh block-structured SBUF or HBM buffer and return an ``NDSlice``
    whose tiles are grouped into blocks.

    This is the block-view counterpart of ``alloc_tiles``: because the result is a
    block view, a whole block is written back in a single coalesced DMA once it has
    been accumulated.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        tile_size (tuple[int, ...]): The per-tile shape (at least 2-D), positive ints.
            Dimension 0 maps to the SBUF partition (P) axis.
        block_size (tuple[int, ...]): The number of tiles per block, the same rank as
            ``tile_size`` (an exact equality here -- unlike ``nt.blocks()`` on a raw
            source, which only requires the block rank not to exceed the source rank),
            positive ints.
        grid (tuple[int, ...] | None): The block-grid shape. Mutually exclusive with
            ``element_shape``.
        buffer_type (nl.MemoryRegion): ``nl.sbuf`` / ``nl.shared_hbm`` /
            ``nl.private_hbm``. Required. ``nl.psum`` is rejected.
        dtype: The element data type. Required.
        element_shape (tuple[int, ...] | None): The exact logical extent per dimension.
            Mutually exclusive with ``grid``.

    Returns:
        NDSlice: A block-grid view over the freshly allocated buffer (its tiles grouped
        into blocks) -- see ``NDSlice`` for the view's attributes. A whole block can be
        written back in one coalesced DMA once it is accumulated.

    Raises ``AssertionError`` in any of these cases:

    - any ``alloc_tiles`` rule is violated (missing ``buffer_type`` / ``dtype``,
      ``nl.psum``, both ``grid`` and ``element_shape``, or a ``tile_size`` /
      ``grid`` / ``element_shape`` shape error).
    - ``block_size`` is not a >= 2-D positive-int tuple.
    - ``block_size`` rank does not equal ``tile_size`` rank.

    Example:
        .. code-block:: python

            # 4x2 block grid of 2x2-tile blocks, 128x512 tiles, in SBUF.
            out = nt.alloc_blocks(tile_size=(128, 512), block_size=(2, 2), grid=(4, 2),
                                  buffer_type=nl.sbuf, dtype=nl.bfloat16)

    Sizing: grid versus element_shape
    ---------------------------------
    ``grid`` is the block grid, so the tile grid is ``grid * block_size``.
    ``element_shape`` is the exact extent, ceiling-divided first into the tile grid and
    then into the block grid. With both omitted, the result is a single-block
    allocation. The SBUF / HBM physical backing follows the ``alloc_tiles`` rules (see
    its *Memory layout*).

    See Also:
        alloc_tiles: allocate a tile buffer with no block grouping.

    """
    _validate_alloc_blocks_args(
        tile_size,
        block_size,
        grid,
        element_shape,
        buffer_type,
        dtype,
    )

    # Allocate as a flat tile grid; alloc_tiles handles both grid= and
    # element_shape= paths. Then promote with _prepend_block_level (the same
    # helper used by nt.blocks(NDSlice, block_size=)) so block layering is
    # consistent across all block-producing factories.
    if element_shape is not None:
        # Forward exact extent; alloc_tiles ceil-divides into the tile grid.
        flat = alloc_tiles(
            tile_size,
            element_shape=element_shape,
            buffer_type=buffer_type,
            dtype=dtype,
        )
    elif grid is not None:
        # Block grid -> flat tile grid by element-wise multiply.
        tile_grid = []
        for d in range(len(block_size)):
            tile_grid.append(grid[d] * block_size[d])
        flat = alloc_tiles(
            tile_size,
            grid=tuple(tile_grid),
            buffer_type=buffer_type,
            dtype=dtype,
        )
    else:
        # Single block: tile grid = block_size.
        flat = alloc_tiles(
            tile_size,
            grid=tuple(block_size),
            buffer_type=buffer_type,
            dtype=dtype,
        )

    return _prepend_block_level(flat, tuple(block_size))


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "tiles",
    "blocks",
    "alloc_tiles",
    "alloc_blocks",
    "NDSlice",
    "Grid",
    "HBMLayout",
    "SBUFLayout",
]
