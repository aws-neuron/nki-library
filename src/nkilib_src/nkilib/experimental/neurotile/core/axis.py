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
from enum import Enum
from typing import Any, Optional, Tuple

import nki.language as nl

from ._helpers import P_DIM, ceiling_div

# ============================================================================
# Axis labels
# ============================================================================


class AxisLabel(Enum):
    """
    Role tag for an Axis.

    Labels record the user-visible role of an axis. They drive structural
    decisions (re-tile rules, AP emission ordering) but never carry
    behavior that isn't expressible from ``count`` and ``step`` alone --
    the label is documentation for the human reader and a tag for cheap
    filtering.

    Members:
        ELEM:       Innermost element walk (one element per step).
        TILE:       Tile-grid axis (one tile per step).
        BLOCK:      Block-grid axis (one block per step).
        SHARD:      Interleaved-shard outer (gapped walk).
        PARTITION:  SBUF partition-row axis (P regime).
        BROADCAST:  Broadcast axis (step == 0).
        REPETITION: SBUF P-tile repetition folded into F.
    """

    ELEM = "elem"
    TILE = "tile"
    BLOCK = "block"
    SHARD = "shard"
    PARTITION = "partition"
    BROADCAST = "broadcast"
    REPETITION = "repetition"


class IndirectKind(Enum):
    """
    Indirect-offset variant tag.

    Distinguishes a single runtime scalar offset from a vector gather.

    Members:
        SCALAR: A single runtime int (LoopVar or mutated SBUF tensor).
        VECTOR: An SBUF tensor of indices for vector gather.
    """

    SCALAR = "scalar"
    VECTOR = "vector"


# ============================================================================
# Axis
# ============================================================================


class Axis(nl.NKIObject):
    """
    One iteration level over a source-tensor dim.

    An ``Axis`` describes a single layer of a tile-grid walk: it visits
    ``count`` items along source dim ``dim``, advancing ``step`` source-
    dim elements between successive items. A :class:`~neurotile.core.grid.Grid`
    is a flat tuple of ``Axis`` instances ordered outer-to-inner; the
    AP emitter walks them in that order to produce the per-level DMA
    pattern.

    Conceptually, a tile-grid walk decomposes into independent levels
    along each dim. For ``nt.tiles(src, tile_size=(128, 256))`` on a
    ``(512, 1024)`` source, the grid carries::

        Axis(count=4, step=128, dim=0, label=TILE)        # 4 row-tiles
        Axis(count=128, step=1, dim=0, label=PARTITION)   # 128 P rows
        Axis(count=4, step=256, dim=1, label=TILE)        # 4 col-tiles
        Axis(count=256, step=1, dim=1, label=ELEM)        # 256 F elements

    A ``BLOCK`` axis stacks above ``TILE`` to introduce coalesced
    multi-tile DMAs; a ``SHARD`` axis carries a gap (``step >
    inner.count * inner.step``) to skip past peer ranks for interleaved
    sharding; a broadcast axis sets ``step = 0`` to revisit the same
    memory.

    The class is an immutable value type: ``split``, ``merge``, and
    ``with_count`` produce new ``Axis`` instances rather than mutating
    in place. The Grid composes these operations to transform whole
    axis tuples.

    Attributes:
        count (int): Number of items the walk visits at this level.
            ``count >= 1``.
        step (int): Elements between successive items, in source-dim
            units (NOT bytes or physical strides). Layout multiplies by
            ``source.strides[dim]`` at AP emission to convert to a
            physical byte stride. ``step == 0`` marks a broadcast axis
            (revisits the same memory); ``step > inner.count * inner.step``
            marks a strided axis (gaps between successive items).
        dim (int): Source-tensor dim id this axis walks. ``dim >= 0``.
            Multiple axes may share a dim (block / tile / partition).
        label (AxisLabel): Role tag -- see :class:`AxisLabel`. Drives
            structural decisions (re-tile rules, AP emission ordering)
            but never carries behavior unexpressible from
            ``count`` / ``step`` alone.

    Invariants:
        ``count >= 1``, ``step >= 0``, ``dim >= 0``,
        ``isinstance(label, AxisLabel)``.
    """

    def __init__(self, count: int, step: int, dim: int, label: AxisLabel):
        """
        Build an Axis.

        Args:
            count (int): Number of items walked at this level. Must be >= 1.
            step (int): Source-dim elements per item. Must be >= 0.
            dim (int): Source-tensor dim id this axis belongs to. Must be >= 0.
            label (AxisLabel): Role of this axis (see :class:`AxisLabel`).

        Raises:
            AssertionError: ``count < 1``, ``step < 0``, ``dim < 0``, or
                ``label`` is not a :class:`AxisLabel` enum value.
        """
        assert count >= 1, "Axis.count must be >= 1, got " + str(count)
        assert step >= 0, "Axis.step must be >= 0, got " + str(step)
        assert dim >= 0, "Axis.dim must be >= 0, got " + str(dim)
        assert isinstance(label, AxisLabel), "Axis.label must be an AxisLabel enum, got " + str(label)
        self.count = count
        self.step = step
        self.dim = dim
        self.label = label

    # ================================================================
    # Public methods
    # ================================================================

    def is_strided(self, inner_step: int) -> bool:
        """
        True if this axis has gaps relative to a contiguous inner walk.

        An axis is contiguous when its source-dim span (``count * step``)
        equals the next inner level's step times its count -- i.e. each
        item lands exactly where the previous item's inner walk ended.
        Strided axes (shard outer) leave gaps; broadcast axes (step=0)
        also count as strided (they revisit the same source location).

        Args:
            inner_step (int): Step of the axis immediately inside this
                one (or 1 for the innermost axis).

        Returns:
            bool: True when this axis walks with gaps relative to its
            inner walk, False when contiguous.
        """
        if self.step == 0:
            return True
        return self.step > inner_step * self.count

    def split(
        self,
        factor: int,
        outer_label: AxisLabel,
        inner_label: AxisLabel,
    ) -> Tuple["Axis", "Axis"]:
        """
        Factor this axis into outer + inner with the given labels.

        Tile-style split: outer walks tile groups, inner walks within a
        tile. Used to introduce tile/block structure::

            outer.count = self.count // factor
            outer.step  = self.step * factor   (group of `factor` items)
            inner.count = factor               (one item per group)
            inner.step  = self.step

        Args:
            factor (int): Group size. Must divide ``self.count`` evenly.
            outer_label (AxisLabel): Label for the outer axis (e.g. BLOCK or TILE).
            inner_label (AxisLabel): Label for the inner axis (e.g. TILE or ELEM).

        Returns:
            tuple[Axis, Axis]: ``(outer_axis, inner_axis)`` in
            outer-to-inner reading order.

        Raises:
            AssertionError: ``factor < 1`` or ``self.count % factor != 0``.
        """
        assert factor >= 1, "split factor must be >= 1, got " + str(factor)
        assert self.count % factor == 0, (
            "Axis.split: factor " + str(factor) + " does not divide count " + str(self.count)
        )
        outer = Axis(
            count=self.count // factor,
            step=self.step * factor,
            dim=self.dim,
            label=outer_label,
        )
        inner = Axis(
            count=factor,
            step=self.step,
            dim=self.dim,
            label=inner_label,
        )
        return outer, inner

    def split_peers(
        self,
        num_peers: int,
        peer_label: AxisLabel,
        owned_label: AxisLabel,
    ) -> Tuple["Axis", "Axis"]:
        """
        Factor this axis for round-robin peer ownership (interleaved shard).

        Produces ``(peer_axis, owned_axis)`` where the peer axis walks
        the first item of each peer slot, and the owned axis walks every
        Nth item from a given peer's starting position::

            peer.count   = num_peers
            peer.step    = self.step              (one item per peer slot)
            owned.count  = self.count // num_peers
            owned.step   = self.step * num_peers  (skip past peers)

        Used for interleaved sharding (stepped slice): the peer axis is
        consumed (one peer's slot) and the owned axis is what this rank
        actually iterates.

        Args:
            num_peers (int): Total peer count (== num_shards). Must
                divide ``self.count`` evenly.
            peer_label (AxisLabel): Label for the peer axis (typically SHARD).
            owned_label (AxisLabel): Label for the owned axis (typically TILE).

        Returns:
            tuple[Axis, Axis]: ``(peer_axis, owned_axis)``.

        Raises:
            AssertionError: ``num_peers < 1`` or
                ``self.count % num_peers != 0``.
        """
        assert num_peers >= 1, "split_peers: num_peers >= 1, got " + str(num_peers)
        assert self.count % num_peers == 0, (
            "Axis.split_peers: num_peers " + str(num_peers) + " does not divide count " + str(self.count)
        )
        peer = Axis(
            count=num_peers,
            step=self.step,
            dim=self.dim,
            label=peer_label,
        )
        owned = Axis(
            count=self.count // num_peers,
            step=self.step * num_peers,
            dim=self.dim,
            label=owned_label,
        )
        return peer, owned

    def merge(self, inner: "Axis", label: AxisLabel) -> "Axis":
        """
        Merge this (outer) axis with the inner axis immediately below it.

        Inverse of :meth:`split`. Requires the outer-inner pair to be
        contiguous (``outer.step == inner.step * inner.count``);
        otherwise the merge would lose the gap.

        Args:
            inner (Axis): The axis immediately inside this one in
                iteration order. Must share ``dim`` with this axis.
            label (AxisLabel): Label for the merged axis.

        Returns:
            Axis: New axis with ``count = self.count * inner.count``,
            ``step = inner.step``, same ``dim``, given ``label``.

        Raises:
            AssertionError: Dim mismatch between outer and inner; or the
                pair is not contiguous (``outer.step !=
                inner.step * inner.count``).
        """
        assert self.dim == inner.dim, "Axis.merge: dim mismatch (" + str(self.dim) + " vs " + str(inner.dim) + ")"
        assert self.step == inner.step * inner.count, (
            "Axis.merge: outer.step "
            + str(self.step)
            + " != inner.step * inner.count "
            + str(inner.step * inner.count)
            + " -- cannot merge across a gap"
        )
        return Axis(
            count=self.count * inner.count,
            step=inner.step,
            dim=self.dim,
            label=label,
        )

    def with_count(self, new_count: int) -> "Axis":
        """
        Return an Axis with the same step/dim/label but a different count.

        Args:
            new_count (int): New count. Must be >= 1.

        Returns:
            Axis: Fresh Axis with the new count; other fields unchanged.

        Raises:
            AssertionError: ``new_count < 1`` (via ``Axis.__init__``).
        """
        return Axis(new_count, self.step, self.dim, self.label)

    # ================================================================
    # Static factories
    # ================================================================

    @staticmethod
    def elem(dim: int, count: int) -> "Axis":
        """
        Build a plain element axis -- step 1, label ELEM.

        Args:
            dim (int): Source-tensor dim id.
            count (int): Number of elements walked.

        Returns:
            Axis: An Axis with ``step=1`` and ``label=AxisLabel.ELEM``.
        """
        return Axis(count=count, step=1, dim=dim, label=AxisLabel.ELEM)

    @staticmethod
    def leaf(dim: int, count: int) -> "Axis":
        """
        Build the innermost axis under a tile/block.

        Partition-labeled on the first dim (P-regime on SBUF), elem
        otherwise.

        Args:
            dim (int): Source-tensor dim id.
            count (int): Tile size on this dim (one element per step).

        Returns:
            Axis: ``label=PARTITION`` if ``dim == P_DIM`` else
            ``label=ELEM``; ``step=1`` either way.
        """
        if dim == P_DIM:
            return Axis(count=count, step=1, dim=dim, label=AxisLabel.PARTITION)
        return Axis(count=count, step=1, dim=dim, label=AxisLabel.ELEM)

    @staticmethod
    def build_axes_for_grid(
        element_shape: tuple,
        tile_size: Optional[tuple],
        block_size: Optional[tuple],
        n_batch_dims: int,
    ) -> tuple:
        """
        Build a flat axis tuple from a tile/block configuration.

        Per-dim axis layout (outer-to-inner reading order):

        - Untiled (tile_size is None): one elem axis spanning
          ``element_shape[d]``.
        - Batch dim (``d < n_batch_dims``): one elem axis spanning
          ``element_shape[d]``.
        - Tile dim with no block: tile axis + leaf axis.
        - Tile dim with block: block axis + tile axis + leaf axis.

        The first non-batch tile dim is the partition dim (its leaf is
        labeled ``PARTITION`` instead of ``ELEM``); the AP emitter hoists
        it to the outermost AP level on SBUF.

        Args:
            element_shape (tuple[int, ...]): Per-dim source extent.
            tile_size (tuple[int, ...] | None): Per-dim tile shape, or
                None for an untiled grid.
            block_size (tuple[int, ...] | None): Per-dim tiles-per-block,
                or None when block-iteration is not configured.
            n_batch_dims (int): Number of leading dims iterated as batch
                (one elem axis each, no tile/block structure).

        Returns:
            tuple[Axis, ...]: Flat axis tuple in outer-to-inner order.
        """
        axes = []
        ndim = len(element_shape)
        partition_dim = n_batch_dims  # first non-batch tile dim
        for d in range(ndim):
            if tile_size is None:
                axes.append(Axis.elem(d, count=element_shape[d]))
                continue
            if d < n_batch_dims:
                axes.append(Axis.elem(d, count=element_shape[d]))
                continue

            ts = tile_size[d]
            es = element_shape[d]
            n_tiles = ceiling_div(es, ts)

            if Axis._has_block(block_size, d):
                bs = block_size[d]
                n_blocks = ceiling_div(n_tiles, bs)
                axes.append(Axis(count=n_blocks, step=bs * ts, dim=d, label=AxisLabel.BLOCK))
                axes.append(Axis(count=bs, step=ts, dim=d, label=AxisLabel.TILE))
            else:
                axes.append(Axis(count=n_tiles, step=ts, dim=d, label=AxisLabel.TILE))

            # Leaf count caps at element extent: for a single-tile partial
            # (ceil(es/ts) == 1 with es < ts) the leaf walks only the actual
            # ``es`` elements. Multi-tile views report leaf=ts and surface
            # the partial trailing tile via Grid.is_remainder + the count*step
            # > es check.
            leaf_count = ts if n_tiles > 1 else min(ts, es)
            leaf_label = AxisLabel.PARTITION if d == partition_dim else AxisLabel.ELEM
            axes.append(Axis(count=leaf_count, step=1, dim=d, label=leaf_label))
        return tuple(axes)

    def __repr__(self):
        return (
            "Axis(count="
            + str(self.count)
            + ", step="
            + str(self.step)
            + ", dim="
            + str(self.dim)
            + ", label="
            + self.label.name
            + ")"
        )

    # ================================================================
    # Private helpers
    # ================================================================

    @staticmethod
    def _has_block(block_size, dim):
        """True if `dim` has a block level configured (count >= 1)."""
        if block_size is None or dim >= len(block_size):
            return False
        bs = block_size[dim]
        return bs is not None and bs >= 1


# ============================================================================
# IndirectOffset -- runtime offset carried by Layout
# ============================================================================


class IndirectOffset(nl.NKIObject):
    """
    Runtime source-element offset attached to a Layout for indirect / vector indexing.

    Layouts carry a compile-time ``offset`` (folded at trace time) plus,
    optionally, a single ``IndirectOffset`` for the cases where indexing
    must resolve at runtime: a scalar source-element offset (e.g.
    ``view[nt.element_offset(offset)]`` or a logical index already lowered
    to element units) or a vector gather (e.g. ``view[idx_tensor]``).
    When set, NKI's DMA path routes through ``scalar_offset=`` /
    ``vector_offset=`` on the AP instead of folding the offset into the
    pattern.

    Example::

        # Inside a kernel:
        for k in nl.sequential_range(N):
            tile = view[k].load()    # k is a LoopVar (runtime scalar)
            #                ^ creates an IndirectOffset(SCALAR, k, dim=0)
            #                  on the resulting NDSlice's Layout.

        gathered = view[index_tensor].load()
        # ^ index_tensor is an SBUF ndarray of ints
        #   creates an IndirectOffset(VECTOR, index_tensor, dim=0)

    Attributes:
        kind (IndirectKind): ``SCALAR`` for single-int runtime offset,
            ``VECTOR`` for gather. See :class:`IndirectKind`.
        value (Any): For ``SCALAR``, a runtime expression or SBUF tensor
            in source-element offset units. For ``VECTOR``, an SBUF
            ndarray of integer indices.
        dim (int): Absolute source-tensor dim id that the offset
            addresses. The id is ABSOLUTE (it never shifts when a batch
            dim drops out of the Grid) because runtime offsets address
            the original source layout, not the current view's
            element_shape.
    """

    def __init__(self, kind: IndirectKind, value: Any, dim: int):
        """
        Build an IndirectOffset.

        Args:
            kind (IndirectKind): SCALAR for a single runtime int, VECTOR for a
                gather-vector tensor.
            value (Any): Runtime expression (SCALAR) or SBUF index
                tensor (VECTOR).
            dim (int): Source-tensor dim id this offset addresses.
                Must be >= 0.

        Raises:
            AssertionError: ``kind`` is not a :class:`IndirectKind` enum value,
                or ``dim < 0``.
        """
        assert isinstance(kind, IndirectKind), "IndirectOffset.kind must be an IndirectKind enum, got " + str(kind)
        assert dim >= 0, "IndirectOffset.dim must be >= 0, got " + str(dim)
        self.kind = kind
        self.value = value
        self.dim = dim

    def __repr__(self):
        return "IndirectOffset(kind=" + self.kind.name + ", dim=" + str(self.dim) + ")"


__all__ = ["Axis", "IndirectOffset", "AxisLabel", "IndirectKind"]
