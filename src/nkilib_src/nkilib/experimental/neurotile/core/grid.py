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

from ._helpers import MIN_TILED_DIMS, ceiling_div
from .axis import Axis, AxisLabel

# ============================================================================
# Grid
# ============================================================================


class Grid(nl.NKIObject):
    """
    The iteration schedule: which tiles exist, in what order, at what level.

    A ``Grid`` carries a flat tuple of :class:`Axis` records describing
    how to walk the source tensor's tile structure -- block grid, tile
    grid, partition rows, inner elements -- plus a cursor pointing at
    the next dim to iterate. :class:`NDSlice` pairs a Grid with a
    :class:`Layout` to produce a complete view; indexing and iteration
    primitives mutate Grid via small composable transforms (consume,
    narrow, split, merge, drop_dim, ...) and never touch Layout's
    physical strides.

    Immutable value type: every primitive returns a fresh Grid. Derived
    fields (``shape``, ``remaining``, ``tile_size``, ...) are computed
    once in ``__init__`` and stored as plain attributes.

    Attributes:
        element_shape (tuple[int, ...]): Source-tensor extent per dim.
            Constant for the lifetime of the Grid -- shrinks happen by
            building a new Grid via ``drop_dim`` / ``narrow``.
        axes (tuple[Axis, ...]): Axes in outer-to-inner reading order,
            grouped by dim. Multiple axes per dim are common
            (block + tile + partition).
        cursor (int): The next dim ``tolist()`` / ``__iter__`` walks.
            Indexing advances the cursor as dims get consumed; batch
            dim selection drops the dim entirely.
        n_batch_dims (int): Count of leading dims iterated as plain
            batch slabs (one elem axis each, no tile/block structure).
            ``.load()`` / ``.store()`` / ``.stream()`` assert this is 0
            -- callers must consume batch dims first.
        ndim (int): ``len(element_shape)``.
        shape (tuple[int, ...]): Per-dim outer axis count, from
            ``cursor`` onward (the shape a user sees on the view).
        remaining (tuple[int, ...]): Per-dim element extent the current
            outer axis still walks (clamped to source).
        tile_size (tuple[int, ...] | None): Per-dim leaf-axis count;
            ``None`` when untiled.
        block_size (tuple[int, ...] | None): Per-dim BLOCK axis count;
            ``None`` when no block layer.
        tile_shape (tuple[int, ...]): Per-dim tile-grid extent
            (``ceil(element_shape[d] / tile_size[d])``).
        is_remainder (bool): True if any axis walks past
            ``element_shape`` -- partial trailing tile present.
    """

    def __init__(self, element_shape, axes, cursor, n_batch_dims, is_remainder=False, remainder_dims=(), tiled=None):
        self.element_shape = element_shape
        self.axes = axes
        self.cursor = cursor
        self.n_batch_dims = n_batch_dims

        # Derived attributes (computed once; tracer needs real attrs not properties).
        self.ndim = len(element_shape)
        self.tile_size = self._compute_tile_size()
        # ``tiled`` records whether the Grid was built with a tile_size.
        # Preserved across primitives so MIN_TILED_DIMS still applies
        # after all TILE / BLOCK axes have been consumed. Set before
        # ``_compute_shape`` runs -- ``_iter_count`` reads it to decide
        # whether a leaf-only dim is iterable.
        if tiled is None:
            self.tiled = self.tile_size is not None
        else:
            self.tiled = tiled
        self.shape = self._compute_shape()
        # Per-dim partial-trailing-tile flags. ``remainder_dims`` is set by
        # ``truncate_to_source`` for the specific dim it clamped; combined
        # with the multi-tile parent overshoot detection so multi-tile
        # views over a non-multiple extent surface as remainders too.
        # Build sorted in-place (NKI tracer rejects ``set``, ``.sort()``
        # method, and ``sorted`` builtin).
        derived = self._dims_walking_past_source()
        merged = []
        for d in range(self.ndim):
            if d in remainder_dims or d in derived:
                merged.append(d)
        self.remainder_dims = tuple(merged)
        self.is_remainder = is_remainder or len(self.remainder_dims) > 0
        self.block_size = self._compute_block_size()
        self.remaining = self._compute_remaining()
        self.tile_shape = self._compute_tile_shape()
        self.block_shape = self._compute_block_shape()

    def _replace(
        self, element_shape=None, axes=None, cursor=None, n_batch_dims=None, is_remainder=False, remainder_dims=None
    ):
        """Return a copy with the named fields overridden (others preserved).

        ``is_remainder`` and ``remainder_dims`` are re-derived in
        ``__init__`` from the new axes; only ``truncate_to_source``
        passes them explicitly when it clamps a leaf.
        """
        if remainder_dims is None:
            remainder_dims = ()
        return Grid(
            element_shape=self.element_shape if element_shape is None else element_shape,
            axes=self.axes if axes is None else axes,
            cursor=self.cursor if cursor is None else cursor,
            n_batch_dims=self.n_batch_dims if n_batch_dims is None else n_batch_dims,
            is_remainder=is_remainder,
            remainder_dims=remainder_dims,
            tiled=self.tiled,
        )

    # ================================================================
    # Static factories
    # ================================================================

    @staticmethod
    def from_shape(element_shape, tile_size=None, block_size=None, n_batch_dims=0, cursor=None, axes=None):
        """Construct a Grid from element_shape + tile/block config.

        Args:
            element_shape: per-dim source extent (tuple of ints).
            tile_size:     per-dim tile granularity, or None for untiled.
            block_size:    per-dim tiles-per-block, or None for no block level.
            n_batch_dims:  count of leading dims that are batch (single elem
                           per item, no tile/block structure).
            cursor:        next dim to iterate (defaults to n_batch_dims).
            axes:          override the computed axes (used by primitives that
                           rebuild axes directly).
        """
        if axes is None:
            axes = Axis.build_axes_for_grid(
                element_shape,
                tile_size,
                block_size,
                n_batch_dims,
            )
        if cursor is None:
            cursor = n_batch_dims if tile_size is not None else 0
        return Grid(
            element_shape=tuple(element_shape),
            axes=axes,
            cursor=cursor,
            n_batch_dims=n_batch_dims,
            tiled=tile_size is not None,
        )

    # ================================================================
    # Derived attributes (computed in __init__)
    # ================================================================

    def _compute_shape(self):
        """Iteration-grid count per dim from cursor onward.

        Uses ``_iter_count`` so consumed dims (post-int leaves) report
        1 instead of their per-tile element count.
        """
        result = []
        for d in range(self.cursor, self.ndim):
            result.append(self._iter_count(d))
        return tuple(result)

    def _dims_walking_past_source(self):
        """Tuple of dims whose outermost axis walks past ``element_shape[d]``.

        Multi-tile parent over a non-multiple source: outer axis
        ``count*step`` exceeds the dim extent, so the walk crosses a
        trailing partial tile. Single-tile children that already
        truncated to the partial size are flagged via
        ``truncate_to_source`` (``remainder_dims=`` constructor arg)
        instead.
        """
        result = []
        for d in range(self.ndim):
            outer = self.outer_axis(d)
            if outer is None:
                continue
            if outer.step == 0:
                continue  # broadcast: doesn't address source memory
            if outer.count * outer.step > self.element_shape[d]:
                result.append(d)
        return tuple(result)

    def _outer_count(self, dim):
        """Outermost axis count on dim, or 1 if no axis remains.

        Used by ``current_count`` for index validation -- returns the
        addressable extent at the dim's current granularity.
        """
        outer = self.outer_axis(dim)
        return 1 if outer is None else outer.count

    def _iter_count(self, dim):
        """Iteration-grid count on `dim` for ``shape`` computation.

        Like ``_outer_count`` but returns 1 for dims whose iteration
        level was consumed (outer is a sub-tile leaf left behind after
        stripping a TILE / BLOCK above it).

        Cases:
          - No outer (consumed): 1.
          - Iteration-level outer (TILE / BLOCK / SHARD): axis count.
          - Broadcast (``outer.step == 0``): axis count -- intentional
            logical repetition.
          - Leaf-only outer (PART / ELEM / REPETITION) where the dim is
            iterable in its own right (batch slab, or an untiled view's
            sole dim): axis count.
          - Leaf-only outer in any other case (post-consume sub-tile
            leaf inside a tiled view): 1.

        Structural markers (``n_batch_dims`` / ``tiled``) drive the
        discriminator, not a heuristic on walked extent vs source size.
        """
        outer = self.outer_axis(dim)
        if outer is None:
            return 1
        if outer.label in (
            AxisLabel.TILE,
            AxisLabel.BLOCK,
            AxisLabel.SHARD,
        ):
            return outer.count
        if outer.step == 0:
            return outer.count
        if dim < self.n_batch_dims:
            return outer.count
        if not self.is_tiled():
            return outer.count
        return 1

    # ================================================================
    # Queries
    # ================================================================

    def outer_axis(self, dim):
        """Return the outermost axis on `dim`, or None if dim is consumed."""
        for ax in self.axes:
            if ax.dim == dim:
                return ax
        return None

    def axes_for(self, dim, label=None):
        """All axes on `dim`, optionally filtered by label, in outer-to-inner order."""
        result = []
        for ax in self.axes:
            if ax.dim != dim:
                continue
            if label is None or ax.label == label:
                result.append(ax)
        return tuple(result)

    def has_label(self, dim, label):
        """True if `dim` has any axis with the given label."""
        for ax in self.axes:
            if ax.dim == dim and ax.label == label:
                return True
        return False

    def is_tiled(self):
        """True if this Grid was constructed with a tile_size.

        Tracked via the ``tiled`` field (set in __init__) so it stays
        True even after all TILE / BLOCK / leaf axes have been consumed.
        Cleanup uses this to apply MIN_TILED_DIMS even on heavily-
        consumed views.
        """
        return self.tiled

    def is_blocked(self, dim=None):
        """True if `dim` (or any dim, when None) has a block axis."""
        for ax in self.axes:
            if dim is not None and ax.dim != dim:
                continue
            if ax.label == AxisLabel.BLOCK:
                return True
        return False

    def is_sharded(self, dim):
        """True if `dim` has a strided shard outer axis."""
        return self.has_label(dim, AxisLabel.SHARD)

    def has_iter_level_outer(self, dim):
        """True if `dim`'s outer axis is an iteration-level grid axis.

        Iteration-level labels (TILE / BLOCK / SHARD) are the ones a
        ``for`` loop walks: they survive consume / narrow as long as the
        grid still has tiles to visit on that dim. Element-level labels
        (PARTITION / ELEM / BROADCAST / REPETITION) are leaves -- not
        iterable, addressed inside a single tile.

        Used by indexing to find the next dim a cursor can land on.
        """
        outer = self.outer_axis(dim)
        if outer is None:
            return False
        return outer.label in (
            AxisLabel.TILE,
            AxisLabel.BLOCK,
            AxisLabel.SHARD,
        )

    def dim_consumed(self, dim):
        """True if `dim` has no axes left (fully indexed)."""
        return self.outer_axis(dim) is None

    def is_element(self, dim):
        """True if `dim` is at element granularity (one axis or none)."""
        return len(self.axes_for(dim)) <= 1

    def current_step(self, dim):
        """Step in source-dim units of the outermost axis on dim, or 1."""
        outer = self.outer_axis(dim)
        return 1 if outer is None else outer.step

    def index_stride_elements(self):
        """Source-element stride for each current public index dim."""
        result = []
        for d in range(self.cursor, self.ndim):
            result.append(self.current_step(d))
        return tuple(result)

    def current_count(self, dim):
        """Outermost-axis count on dim, or 1."""
        return self._outer_count(dim)

    def tile_size_of(self, dim):
        """Per-tile element extent on `dim` (count of the elem axis), or None."""
        elem_axes = self.axes_for(dim, label=AxisLabel.ELEM)
        if len(elem_axes) > 0:
            return elem_axes[0].count
        # Partition dim: leaf is labeled 'partition', not 'elem'.
        partition_axes = self.axes_for(dim, label=AxisLabel.PARTITION)
        if len(partition_axes) > 0:
            return partition_axes[0].count
        return None

    def block_size_of(self, dim):
        """Tiles-per-block on `dim`, or None when no block axis."""
        block_axes = self.axes_for(dim, label=AxisLabel.BLOCK)
        if len(block_axes) == 0:
            return None
        # block.step / tile.step = block.count's worth of tiles per block.
        block_ax = block_axes[0]
        tile_axes = self.axes_for(dim, label=AxisLabel.TILE)
        if len(tile_axes) == 0:
            return 1
        tile_ax = tile_axes[0]
        if tile_ax.step == 0:
            return 1
        return block_ax.step // tile_ax.step

    def _compute_tile_size(self):
        """Tuple of per-dim tile sizes (or None if any dim is untiled)."""
        result = []
        for d in range(self.ndim):
            ts = self.tile_size_of(d)
            if ts is None:
                return None
            result.append(ts)
        return tuple(result)

    def _compute_block_size(self):
        """Tuple of per-dim block sizes; entries are None where no block axis.

        Returns None overall when no dim has a block axis. Otherwise each
        entry is either the tiles-per-block on that dim (>=1) or None
        (consumed / no block level). Callers must handle None entries; the
        previous "0" sentinel was ambiguous with "block of size 0".
        """
        if not self.is_blocked():
            return None
        result = []
        for d in range(self.ndim):
            result.append(self.block_size_of(d))
        return tuple(result)

    def _compute_remaining(self):
        """Per-dim addressable element span, clamped to the source extent.

        Each dim falls into one of three cases:

          1. No outer axis (consumed): ``element_shape[d]``.
          2. Broadcast (``outer.step == 0``): ``outer.count``.
          3. Walking axis: walked extent clamped to ``element_shape[d]``.
        """
        result = []
        for d in range(self.ndim):
            outer = self.outer_axis(d)
            if outer is None:
                result.append(self.element_shape[d])
                continue
            if outer.step == 0:
                result.append(outer.count)
                continue
            walked = self._walked_extent_on_dim(d, outer)
            result.append(min(walked, self.element_shape[d]))
        return tuple(result)

    def _walked_extent_on_dim(self, dim, outer):
        """Element extent the outer axis walks on ``dim``.

        Normal walk (``count * step``) is the default. When ``dim`` is a
        partial-trailing-tile dim (``truncate_to_source`` clamped its
        elem-leaf below ``outer.step``), the last slot is short by
        ``outer.step - leaf.count`` -- correct walked extent is
        ``(count - 1) * step + leaf.count``.

        Sharded / interleaved views are not in ``remainder_dims`` so
        they always take the normal-walk branch.
        """
        if dim not in self.remainder_dims:
            return outer.count * outer.step
        elem_leaf = self._elem_leaf_on_dim(dim)
        if elem_leaf is None or elem_leaf == outer:
            return outer.count * outer.step
        return (outer.count - 1) * outer.step + elem_leaf.count

    def _elem_leaf_on_dim(self, dim):
        """Innermost ``step == 1`` axis on ``dim``, or None if none exists."""
        result = None
        for ax in self.axes:
            if ax.dim == dim and ax.step == 1:
                result = ax
        return result

    def _compute_tile_shape(self):
        """Per-dim tile-grid count."""
        result = []
        for d in range(self.ndim):
            ts = self.tile_size_of(d)
            if ts is None or ts == 0:
                result.append(1)
            else:
                total_elements = self._dim_element_walk(d)
                result.append(ceiling_div(total_elements, ts))
        return tuple(result)

    def _compute_block_shape(self):
        """Per-dim block-grid count, or None on non-block grids.

        On block grids, dims with no block axis contribute 1 so the result
        rank matches ``ndim``.
        """
        if not self.is_blocked():
            return None
        result = []
        for d in range(self.ndim):
            block = self.block_size_of(d)
            tile = self.tile_size_of(d)
            if block is None or tile is None or block * tile <= 0:
                result.append(1)
                continue
            result.append(ceiling_div(self.element_shape[d], block * tile))
        return tuple(result)

    def _dim_element_walk(self, dim):
        """Total element walk on `dim` from outermost axis (count * step)."""
        outer = self.outer_axis(dim)
        if outer is None:
            return self.element_shape[dim]
        if outer.step == 0:
            return outer.count
        return outer.count * outer.step

    def owned_extents(self):
        """Per-dim owned element count, excluding gaps from shard axes.

        For a non-strided dim: outermost axis walk (count * step).
        For a sharded dim: walk all non-strided axes in the group; gapped
        axes contribute count * inner_step (one inner walk per owned tile,
        gaps excluded).
        """
        result = []
        for d in range(self.ndim):
            result.append(self._owned_extent(d))
        return tuple(result)

    def gathered_dims(self):
        """The non-trivial (extent > 1) remaining dims a gather/transpose operates
        on, outermost-first -- size-1 dims (consumed batch / partition leaf) drop."""
        result = []
        for extent in self.remaining:
            if extent > 1:
                result.append(extent)
        return tuple(result)

    def has_strided_axis(self, dim):
        """True if `dim` has any axis whose step exceeds its inner walk extent.

        Walks innermost-to-outermost. The innermost axis has no inner walk
        (its walk extent is `count`). For each outer axis, the per-item
        step exceeds the inner walk's element extent when the axis leaves
        gaps (slice-induced interleaved walk; shard outer; broadcast).
        """
        ax_list = self.axes_for(dim)
        if len(ax_list) < 2:
            return False
        inner_walk = ax_list[-1].count if ax_list[-1].step > 0 else 1
        for i in range(len(ax_list) - 2, -1, -1):
            ax = ax_list[i]
            if ax.step == 0:
                continue  # broadcast: not a gap walk
            if ax.step > inner_walk:
                return True
            inner_walk = ax.count * ax.step
        return False

    def has_any_gapped_axis(self):
        """True if any dim has a gap-walking axis (shard label OR strided step).

        Two sources of gaps:
          1. Explicit AxisLabel.SHARD axis.
          2. Any non-broadcast axis whose step exceeds the inner walk's
             extent (slice with step!=1 produces this without using the
             SHARD label on the kept owned axis).
        """
        for d in range(self.ndim):
            if self.is_sharded(d) or self.has_strided_axis(d):
                return True
        return False

    def strided_outer_step(self, dim):
        """Outer step of a strided axis on `dim`, or None.

        A dim is "strided" when:
          1. It carries an explicit AxisLabel.SHARD axis, OR
          2. Its outer axis has `step > inner_walk_extent`, i.e. the
             per-item stride leaves a gap before the next owned item.

        The returned step is the gap stride between owned items; the AP
        emitter uses it as the outer-level stride and walks `owned_count`
        items above an inner per-tile walk.
        """
        shard_axes = self.axes_for(dim, label=AxisLabel.SHARD)
        if len(shard_axes) > 0:
            return shard_axes[0].step
        ax_list = self.axes_for(dim)
        if len(ax_list) < 2:
            return None
        inner_walk = ax_list[-1].count if ax_list[-1].step > 0 else 1
        for i in range(len(ax_list) - 2, -1, -1):
            ax = ax_list[i]
            if ax.step == 0:
                continue
            if ax.step > inner_walk:
                return ax.step
            inner_walk = ax.count * ax.step
        return None

    def walked_extent(self, dim):
        """Element span actually walked on `dim` (gaps excluded).

        For a non-strided dim: full `remaining[d]`. For a dim whose outer
        axis skips past peer items, the walk visits
        `(owned - 1) * outer_step + inner_walk` elements -- all owned
        tiles, no gap bytes.
        """
        if not self.has_strided_axis(dim) and not self.is_sharded(dim):
            return self.remaining[dim]
        ax_list = self.axes_for(dim)
        inner_walk = ax_list[-1].count if ax_list[-1].step > 0 else 1
        cumulative = inner_walk
        extent = inner_walk
        for i in range(len(ax_list) - 2, -1, -1):
            ax = ax_list[i]
            if ax.step == 0:
                continue
            extent = (ax.count - 1) * ax.step + cumulative
            cumulative = ax.count * ax.step
        return extent

    def _owned_extent(self, dim):
        """Element count on dim excluding shard-axis gaps, clamped to source."""
        ax_list = self.axes_for(dim)
        if len(ax_list) == 0:
            return self.element_shape[dim]

        # Walk innermost-to-outermost; multiply by count except for shard
        # axes (which contribute count copies of the inner walk, no gap).
        owned = 1
        for i in range(len(ax_list) - 1, -1, -1):
            ax = ax_list[i]
            if ax.label == AxisLabel.SHARD:
                # Gapped: each owned item's footprint is just the inner walk.
                # Multiply by count (number of owned items).
                owned = owned * ax.count
            elif ax.step == 0:
                # Broadcast: revisits same memory; doesn't add to walk.
                pass
            else:
                # Contiguous level: contributes count * inner-walk
                # (which we have so far in `owned`).
                if i == len(ax_list) - 1:
                    # Innermost level: count is the walk.
                    owned = ax.count
                else:
                    owned = owned * ax.count
        # A partial trailing tile makes the unclamped walk overshoot;
        # cap it at the source extent so SBUF allocation matches the
        # in-bounds DMA volume.
        return min(owned, self.element_shape[dim])

    # ================================================================
    # Primitives -- six orthogonal operations on axes
    # ================================================================

    def consume(self, dim):
        """Remove the outermost axis on `dim`.

        Each int / runtime / vector key consumes one grid level: the
        outermost axis on `dim` is dropped, exposing the next-level
        axis (or leaving only a leaf, in which case the dim is at
        element-level). Layout records the offset shift separately.

        A dim with no axes left is fully consumed and ``cleanup`` will
        drop it (subject to MIN_TILED_DIMS for tiled views).
        """
        new_axes = []
        removed = False
        for ax in self.axes:
            if ax.dim == dim and not removed:
                removed = True
                continue
            new_axes.append(ax)
        return self._replace(axes=tuple(new_axes))

    def narrow(self, dim, count):
        """Shrink the outermost axis on `dim` to `count` items.

        Used by slice indexing (start..stop): caller has already advanced
        Layout offset by `start * outer.step`; this restricts the count.
        """
        new_axes = []
        narrowed = False
        for ax in self.axes:
            if ax.dim == dim and not narrowed:
                narrowed = True
                new_axes.append(ax.with_count(count))
            else:
                new_axes.append(ax)
        return self._replace(axes=tuple(new_axes))

    def truncate_to_source(self, dim, elements_consumed):
        """Clamp ``dim``'s tile walk to the addressable remainder
        (``element_shape[dim] - elements_consumed``) and flag it partial.

        Clamps the TILE axis to the tiles the remainder fills
        (``ceil(addressable / tile_extent)``), and the element leaf to a lone
        partial tile's width. The leaf is clamped only for a single partial
        tile -- across several tiles the trailing partial resolves per tile at
        descent time, so the shared leaf stays full width.

        No-op when ``elements_consumed`` is not a compile-time int, ``dim`` has
        no element leaf, the leaf already spans the source extent, the
        remainder is non-positive, or the walk already fits.
        """
        if not isinstance(elements_consumed, int) or elements_consumed < 0:
            return self
        elem_leaf = self._elem_leaf_on_dim(dim)
        if elem_leaf is None:
            return self
        # Leaf already capped at the per-dim element extent: no multi-tile
        # parent to truncate from.
        if elem_leaf.count >= self.element_shape[dim]:
            return self
        addressable = self.element_shape[dim] - elements_consumed
        if addressable <= 0:
            return self

        tile_extent = elem_leaf.count
        reachable_tiles = ceiling_div(addressable, tile_extent)
        last_tile_width = addressable - (reachable_tiles - 1) * tile_extent
        # A lone partial tile narrows the leaf; multiple tiles keep the leaf
        # full (the trailing partial resolves per tile at descent time).
        clamp_leaf = reachable_tiles == 1 and last_tile_width < elem_leaf.count

        # The elem-leaf being the dim's outermost axis (no TILE wrapper, e.g. a
        # consumed slice) means ``narrow`` is the right primitive.
        if self.outer_axis(dim) == elem_leaf:
            if not clamp_leaf:
                return self
            return self.narrow(dim, last_tile_width)._replace(
                is_remainder=True,
                remainder_dims=(dim,),
            )

        changed = False
        new_axes = []
        for ax in self.axes:
            if ax.dim == dim and ax.label == AxisLabel.TILE and ax.count > reachable_tiles:
                new_axes.append(ax.with_count(reachable_tiles))
                changed = True
            elif ax.dim == dim and ax.step == 1 and ax.count == tile_extent and clamp_leaf:
                new_axes.append(ax.with_count(last_tile_width))
                changed = True
            else:
                new_axes.append(ax)
        if not changed:
            return self
        return self._replace(
            axes=tuple(new_axes),
            is_remainder=True,
            remainder_dims=(dim,),
        )

    def split(self, dim, factor, outer_label, inner_label):
        """Factor the outermost axis on `dim` into outer + inner (tile-style).

        outer.count == self.count // factor    (groups of `factor` items)
        outer.step  == self.step * factor
        inner.count == factor                   (items within a group)
        inner.step  == self.step

        Used for tile/block introduction.
        """
        new_axes = []
        split_done = False
        for ax in self.axes:
            if ax.dim == dim and not split_done:
                split_done = True
                outer, inner = ax.split(factor, outer_label, inner_label)
                new_axes.append(outer)
                new_axes.append(inner)
            else:
                new_axes.append(ax)
        return self._replace(axes=tuple(new_axes))

    def split_peers(self, dim, num_peers, peer_label, owned_label):
        """Factor the outermost axis for round-robin sharding (interleaved).

        peer.count = num_peers; peer.step = self.step (one item per peer)
        owned.count = self.count // num_peers; owned.step = self.step * num_peers

        Used by stepped-slice indexing: this rank's owned axis is `owned`,
        and the `peer` axis is then consumed (one rank's slot).
        """
        new_axes = []
        split_done = False
        for ax in self.axes:
            if ax.dim == dim and not split_done:
                split_done = True
                peer, owned = ax.split_peers(num_peers, peer_label, owned_label)
                new_axes.append(peer)
                new_axes.append(owned)
            else:
                new_axes.append(ax)
        return self._replace(axes=tuple(new_axes))

    def merge(self, dim, n_axes, label):
        """Merge the outermost `n_axes` contiguous axes on `dim` into one.

        Inverse of repeated splits. Asserts each pair is contiguous.
        """
        assert n_axes >= 1, "merge: n_axes must be >= 1, got " + str(n_axes)
        if n_axes == 1:
            return self

        # Find the outermost run of n_axes consecutive axes on dim.
        run_start = None
        run = []
        for i in range(len(self.axes)):
            ax = self.axes[i]
            if ax.dim == dim:
                if run_start is None:
                    run_start = i
                run.append((i, ax))
                if len(run) == n_axes:
                    break

        assert len(run) == n_axes, "merge: dim " + str(dim) + " has fewer than " + str(n_axes) + " axes available"

        # Merge from inner to outer.
        merged = run[-1][1]
        for i in range(len(run) - 2, -1, -1):
            merged = run[i][1].merge(merged, label=label)

        # Build the new axis tuple: drop the merged span, insert the merged axis.
        first_idx = run[0][0]
        last_idx = run[-1][0]
        new_axes = list(self.axes[:first_idx]) + [merged] + list(self.axes[last_idx + 1 :])
        return self._replace(axes=tuple(new_axes))

    def reorder(self, perm):
        """Permute dims: new dim d gets the old dim perm[d]'s axes and shape.

        Used by view.permute(). Updates element_shape and re-labels each
        axis's dim id according to perm. Axes are regrouped in new dim
        order.
        """
        assert len(perm) == self.ndim, "reorder: perm length " + str(len(perm)) + " != ndim " + str(self.ndim)
        # inverse_perm[old_dim] = new_dim
        inverse = [0] * self.ndim
        for new_d in range(self.ndim):
            inverse[perm[new_d]] = new_d

        relabeled = []
        for ax in self.axes:
            relabeled.append(Axis(count=ax.count, step=ax.step, dim=inverse[ax.dim], label=ax.label))
        # Regroup by new dim, preserving outer-to-inner within each dim.
        regrouped = []
        for new_d in range(self.ndim):
            for ax in relabeled:
                if ax.dim == new_d:
                    regrouped.append(ax)

        new_es = []
        for d in range(self.ndim):
            new_es.append(self.element_shape[perm[d]])
        new_element_shape = tuple(new_es)
        return self._replace(element_shape=new_element_shape, axes=tuple(regrouped))

    def broadcast(self, dim, size):
        """Add a broadcast axis (step=0) on `dim` with `size` count.

        Used by view.broadcast(): repeats the same source element `size`
        times. Inserted as the outermost axis on `dim`.
        """
        broadcast_axis = Axis(count=size, step=0, dim=dim, label=AxisLabel.BROADCAST)
        new_axes = [broadcast_axis] + list(self.axes)
        # Regroup so dim's axes stay contiguous.
        regrouped = []
        for d in range(self.ndim):
            for ax in new_axes:
                if ax.dim == d:
                    regrouped.append(ax)
        return self._replace(axes=tuple(regrouped))

    # ================================================================
    # Composed operations
    # ================================================================

    def drop_dim(self, dim):
        """Remove `dim` entirely: drop all its axes and shrink element_shape."""
        new_axes = []
        for ax in self.axes:
            if ax.dim == dim:
                continue
            # Decrement dim id for axes after the dropped dim.
            if ax.dim > dim:
                new_axes.append(Axis(count=ax.count, step=ax.step, dim=ax.dim - 1, label=ax.label))
            else:
                new_axes.append(ax)

        new_es = []
        for d in range(self.ndim):
            if d != dim:
                new_es.append(self.element_shape[d])
        new_element_shape = tuple(new_es)
        new_cursor = self.cursor
        if dim < self.cursor:
            new_cursor = new_cursor - 1
        new_n_batch = self.n_batch_dims
        if dim < self.n_batch_dims:
            new_n_batch = new_n_batch - 1
        # Re-number remainder_dims: drop entries for the removed dim,
        # shift entries past it down by one. is_remainder is preserved
        # if any remainder dims survive (or the parent's flag was set
        # for some other reason already captured in _dims_walking_past_source).
        new_remainder = []
        for rd in self.remainder_dims:
            if rd == dim:
                continue
            if rd > dim:
                new_remainder.append(rd - 1)
            else:
                new_remainder.append(rd)
        return self._replace(
            element_shape=new_element_shape,
            axes=tuple(new_axes),
            cursor=new_cursor,
            n_batch_dims=new_n_batch,
            is_remainder=len(new_remainder) > 0,
            remainder_dims=tuple(new_remainder),
        )

    def with_cursor(self, new_cursor):
        """Return Grid with cursor advanced to `new_cursor`.

        Preserves remainder flags -- a cursor change is structural, not
        a re-derivation of the dim's partial-tile status.
        """
        return self._replace(
            cursor=new_cursor,
            is_remainder=self.is_remainder,
            remainder_dims=self.remainder_dims,
        )

    def with_cursor_dim(self, dim):
        """Set cursor to `dim` -- used by tolist(dim=) for non-default iteration."""
        return self.with_cursor(dim)

    def with_cursor_past_consumed(self, consumed_dims):
        """Place the cursor on the first dim that's still iterable after
        consuming ``consumed_dims``.

        Skips past every consumed dim and any non-iter-level dim (e.g.
        a dim narrowed to its tile-leaf). If nothing past the cursor is
        iterable but some other dim still has an iter-level outer,
        wraps to that dim to expose the next-level grid (e.g. block
        interior after consuming a block).

        Used by both indexing (after one or more keys consume) and
        streaming (after a single stream step consumes the streamed
        dim) -- both represent "iteration level on these dims has been
        taken; place the cursor on the next iterable position".
        """
        cursor = self.cursor
        while cursor < self.ndim and (cursor in consumed_dims or not self.has_iter_level_outer(cursor)):
            cursor = cursor + 1

        if cursor >= self.ndim:
            for d in range(self.ndim):
                if self.has_iter_level_outer(d):
                    cursor = d
                    break

        if cursor == self.cursor:
            return self
        return self.with_cursor(cursor)

    def cleanup(self):
        """Drop dims whose axis lists are empty; auto-pop trivial tiles after batch.

        Returns (new_grid, dropped_dims). Dropped dims is the tuple of
        source-dim ids that were removed; callers (NDSlice) pass this to
        Layout.drop_dims to keep `strides` aligned with the surviving dims.

        The "auto-pop after batch collapse" rule: if a batch dim is
        removed AND all remaining tile dims have count==1 on their outer
        axis, pop those outer axes (descend to element level so
        subsequent indexing is at element granularity).
        """
        consumed = []
        for d in range(self.ndim):
            if self.dim_consumed(d):
                consumed.append(d)
        if len(consumed) == 0:
            return self, ()

        # Tiled views must keep at least MIN_TILED_DIMS dims.
        max_drops = len(consumed)
        if self.is_tiled():
            available = self.ndim - MIN_TILED_DIMS
            if available < max_drops:
                max_drops = max(0, available)
        to_drop = consumed[:max_drops]

        result = self
        # Drop in reverse so dim ids stay valid.
        for d_idx in range(len(to_drop) - 1, -1, -1):
            result = result.drop_dim(to_drop[d_idx])

        # Auto-pop: after batch dims drop, if all remaining tile dims have
        # count==1 on the outer axis, descend to element level.
        dropped_batch = False
        for d in to_drop:
            if d < self.n_batch_dims:
                dropped_batch = True
                break
        if dropped_batch and result.is_tiled():
            all_single = True
            need_pop = False
            for d in range(result.ndim):
                outer = result.outer_axis(d)
                if outer is None:
                    continue
                if outer.count > 1:
                    all_single = False
                    break
                if len(result.axes_for(d)) > 1:
                    need_pop = True
            if all_single and need_pop:
                for d in range(result.ndim):
                    outer = result.outer_axis(d)
                    if outer is not None and outer.count == 1 and len(result.axes_for(d)) > 1:
                        # Pop the count==1 outer axis; the inner axis
                        # describes the dim's element-level layout.
                        result = result.consume(d)

        return result, tuple(to_drop)

    def tile(self, new_tile_size):
        """Re-tile: merge tile+elem axes back into one elem axis, then split anew.

        Preserves any outer shard / block / broadcast axes; only the
        innermost (tile, elem) pair on each non-batch dim is rebuilt.
        """
        result = self
        for d in range(self.n_batch_dims, self.ndim):
            ts = new_tile_size[d - self.n_batch_dims] if d >= self.n_batch_dims else None
            if ts is None:
                continue
            # Merge tile + elem on this dim if both exist.
            tile_axes = result.axes_for(d, label=AxisLabel.TILE)
            if len(tile_axes) > 0:
                # Find the innermost tile + leaf pair, merge to elem.
                result = result._merge_innermost_pair(d)
            # Now split the leaf elem at the new tile size.
            result = result._split_leaf_for_tile(d, ts)
        return result

    def _merge_innermost_pair(self, dim):
        """Merge the innermost (tile, leaf) pair on dim into one elem axis.

        The merged count is the outer/inner count product, clamped to the
        addressable extent ``element_shape[dim]``: when the leaf was
        carrying a full per-tile span on a multi-tile parent over a
        partial trailing tile, the unclamped product walks past the
        source. Clamping here keeps the re-tiled view in bounds.
        """
        ax_list = list(self.axes_for(dim))
        if len(ax_list) < 2:
            return self
        # Innermost two axes on dim.
        inner_two = ax_list[-2:]
        # Locate them in self.axes for replacement.
        first_indices = []
        for i in range(len(self.axes)):
            if self.axes[i].dim == dim:
                first_indices.append(i)
        outer_idx = first_indices[-2]
        inner_idx = first_indices[-1]
        merged = inner_two[0].merge(inner_two[1], label=AxisLabel.ELEM)
        # Clamp the merged count at the addressable extent.
        addressable = self.element_shape[dim]
        if merged.step > 0 and merged.count * merged.step > addressable:
            merged = merged.with_count(addressable // merged.step)
        new_axes = list(self.axes[:outer_idx]) + [merged] + list(self.axes[inner_idx + 1 :])
        return self._replace(axes=tuple(new_axes))

    def _split_leaf_for_tile(self, dim, new_tile_size):
        """Split the innermost elem axis on `dim` into (tile, leaf) at the new size.

        Always produces a (TILE, leaf) pair on `dim`, even when the leaf
        already matches new_tile_size -- the tile axis with count=1 keeps
        the dim's structure consistent with newly-tiled views and lets
        the leaf carry the right label (PARTITION on the partition dim,
        ELEM elsewhere).
        """
        # Locate the innermost elem/partition axis on dim.
        ax_list = self.axes_for(dim)
        if len(ax_list) == 0:
            return self
        leaf = ax_list[-1]
        if leaf.count < new_tile_size:
            return self  # below new tile size; nothing to split
        # Determine the proper leaf label for this dim.
        partition_dim = self.n_batch_dims
        leaf_label = AxisLabel.PARTITION if dim == partition_dim else AxisLabel.ELEM
        # Find the leaf axis index.
        leaf_idx = None
        for i in range(len(self.axes) - 1, -1, -1):
            ax = self.axes[i]
            if ax.dim == dim and (ax.label == AxisLabel.ELEM or ax.label == AxisLabel.PARTITION):
                leaf_idx = i
                break
        if leaf_idx is None:
            return self
        # Split: outer (TILE) walks tile-grain; inner is the leaf axis.
        n_outer = leaf.count // new_tile_size
        outer_axis = Axis(count=n_outer, step=new_tile_size, dim=dim, label=AxisLabel.TILE)
        inner_axis = Axis(count=new_tile_size, step=leaf.step, dim=dim, label=leaf_label)
        new_axes = list(self.axes[:leaf_idx]) + [outer_axis, inner_axis] + list(self.axes[leaf_idx + 1 :])
        return self._replace(axes=tuple(new_axes))

    def with_block(self, block_size):
        """Prepend a block axis on each non-batch dim with bs >= 1.

        Used by nt.blocks() promotion. Block step = bs * tile_step.
        Preserves any outer shard axis (block goes immediately above tile).
        """
        result = self
        for d in range(self.n_batch_dims, self.ndim):
            d_idx = d - self.n_batch_dims
            if d_idx >= len(block_size):
                continue
            bs = block_size[d_idx]
            if bs is None or bs < 1:
                continue
            result = result._prepend_block_on_dim(d, bs)
        return result

    def _prepend_block_on_dim(self, dim, block_size):
        """Insert a block axis above the outermost tile axis on `dim`."""
        # Find the outermost tile axis on dim.
        tile_axes = self.axes_for(dim, label=AxisLabel.TILE)
        if len(tile_axes) == 0:
            return self
        tile_ax = tile_axes[0]
        # Locate it in self.axes.
        tile_idx = None
        for i in range(len(self.axes)):
            ax = self.axes[i]
            if ax.dim == dim and ax.label == AxisLabel.TILE:
                tile_idx = i
                break
        if tile_idx is None:
            return self
        n_tiles = tile_ax.count
        n_blocks = ceiling_div(n_tiles, block_size)
        block_axis = Axis(
            count=n_blocks,
            step=block_size * tile_ax.step,
            dim=dim,
            label=AxisLabel.BLOCK,
        )
        # The tile axis below now has count = block_size.
        new_tile_axis = Axis(
            count=block_size,
            step=tile_ax.step,
            dim=dim,
            label=AxisLabel.TILE,
        )
        new_axes = list(self.axes[:tile_idx]) + [block_axis, new_tile_axis] + list(self.axes[tile_idx + 1 :])
        return self._replace(axes=tuple(new_axes))

    def strip_block(self):
        """Remove block axes by merging block+tile back into a single tile axis.

        Used by nt.tiles(view) when the input is a block-level view --
        descends to tile granularity by collapsing the block level.
        """
        result = self
        for d in range(self.ndim):
            if result.is_blocked(d):
                result = result._merge_block_into_tile(d)
        return result

    def _merge_block_into_tile(self, dim):
        """Merge the block axis on `dim` with the tile axis below it."""
        block_axes = self.axes_for(dim, label=AxisLabel.BLOCK)
        tile_axes = self.axes_for(dim, label=AxisLabel.TILE)
        if len(block_axes) == 0 or len(tile_axes) == 0:
            return self
        block_ax = block_axes[0]
        tile_ax = tile_axes[0]
        # Locate indices.
        block_idx = None
        tile_idx = None
        for i in range(len(self.axes)):
            ax = self.axes[i]
            if ax.dim == dim and ax.label == AxisLabel.BLOCK and block_idx is None:
                block_idx = i
            if ax.dim == dim and ax.label == AxisLabel.TILE and tile_idx is None and block_idx is not None:
                tile_idx = i
                break
        if block_idx is None or tile_idx is None:
            return self
        merged_tile = block_ax.merge(tile_ax, label=AxisLabel.TILE)
        new_axes = list(self.axes[:block_idx]) + [merged_tile] + list(self.axes[tile_idx + 1 :])
        return self._replace(axes=tuple(new_axes))

    # ================================================================
    # Validation
    # ================================================================

    def validate_retile_to(self, new_tile_size):
        """Reject re-tiles that corrupt iteration on a sharded dim.

        Per-dim rules:
          - Un-sharded or block-sharded: subdivide and aggregate fine
            as long as new tile divides/is divided-by current tile size.
          - Interleaved-sharded: only subdivide is safe. Aggregating
            would straddle the gap between owned tiles.
        """
        for d_idx in range(len(new_tile_size)):
            abs_dim = self.n_batch_dims + d_idx
            new_t = new_tile_size[d_idx]
            old_t = self.tile_size_of(abs_dim)
            if old_t is None or new_t == old_t:
                continue

            is_interleaved = self.is_sharded(abs_dim)

            if new_t < old_t:
                assert old_t % new_t == 0, (
                    "re-tile size "
                    + str(new_t)
                    + " on dim "
                    + str(abs_dim)
                    + " must divide current tile_size "
                    + str(old_t)
                )
                continue

            # new_t > old_t: aggregate.
            assert not is_interleaved, (
                "re-tile size "
                + str(new_t)
                + " on dim "
                + str(abs_dim)
                + " exceeds contiguous run "
                + str(old_t)
                + "; cannot aggregate across interleaved-shard gaps"
            )
            assert new_t % old_t == 0, (
                "re-tile size "
                + str(new_t)
                + " on dim "
                + str(abs_dim)
                + " must be a multiple of current tile_size "
                + str(old_t)
                + " when aggregating"
            )

    # ================================================================
    # Untiled view
    # ================================================================

    def as_untiled(self, new_element_shape):
        """Return an untiled Grid (after HBM transforms like reshape/permute)."""
        new_axes = []
        for d in range(len(new_element_shape)):
            new_axes.append(Axis.elem(d, count=new_element_shape[d]))
        return Grid(
            element_shape=tuple(new_element_shape),
            axes=tuple(new_axes),
            cursor=0,
            n_batch_dims=0,
            tiled=False,
        )

    # ================================================================
    # Representation
    # ================================================================

    def __repr__(self):
        return (
            "Grid(shape="
            + str(self.shape)
            + ", element_shape="
            + str(self.element_shape)
            + ", cursor="
            + str(self.cursor)
            + ", axes="
            + str(self.axes)
            + ")"
        )
