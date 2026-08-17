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

from .axis import AxisLabel


class APEmitter(nl.NKIObject):
    """Emit DMA access-pattern levels from a list of typed axes.

    The emitter's contract: produce an AP that walks within
    [0, element_shape[d] * source_strides[d]) on every source dim. The
    walk is a flat outer-to-inner iteration over `axes`, transformed by
    three peephole passes:

      - merge_contiguous: fold contiguous outer+inner pairs into one AP
        level (with the inner walk clamped to source extent so partial
        trailing tiles don't overshoot). Refuses to merge across the
        partition leaf (SBUF needs a distinct P-walk) and across dim
        boundaries (NKI bounds-checks per source-tensor dim).
      - hoist_partition: move the partition-row level to the outermost
        AP position so HBM iteration order matches SBUF P-walk.
      - drop_unit_levels: drop [stride, 1] entries; keep at most one
        matching the partition stride.

    Output: list of [stride, count] pairs. Stride per AP level is
    `axis.step * source_strides[axis.dim]`. No level-count cap -- the compiler
    validates descriptor depth when the AP is consumed.
    """

    @staticmethod
    def emit(
        axes: tuple,
        source_strides: tuple,
        element_shape: tuple,
    ) -> list:
        """
        Walk axes outer-to-inner; emit one [stride, count] AP level per axis.

        Args:
            axes (tuple[Axis, ...]): Iteration sequence to emit, in
                outer-to-inner reading order.
            source_strides (tuple[int, ...]): Physical element strides
                of the source tensor, indexed by ``axis.dim``.
            element_shape (tuple[int, ...]): Per-dim source extent. The
                merge pass clamps the merged inner walk to the source
                extent so partial trailing tiles do not overshoot.

        Returns:
            list[list[int]]: Sequence of ``[stride, count]`` pairs.
            Drops ``count == 1`` levels (except the partition axis, which
            is structurally required for SBUF AP rank consistency).
            Applies merge-contiguous (with source clamp) and
            partition-hoist peephole passes. No level-count cap (the
            compiler validates descriptor depth).
        """
        levels = APEmitter._emit_raw(axes, source_strides)
        levels = APEmitter._merge_contiguous(axes, levels, element_shape)
        levels = APEmitter._hoist_partition(axes, source_strides, levels)
        partition_stride = APEmitter._partition_stride(axes, source_strides)
        levels = APEmitter._drop_unit_levels(levels, partition_stride)
        return levels

    # ================================================================
    # Private helpers
    # ================================================================

    @staticmethod
    def _emit_raw(axes, source_strides):
        """One [stride_bytes, count] per axis, in iteration order. Pre-peephole.

        stride_bytes = axis.step * source_strides[axis.dim].
        For broadcast axes (step=0), stride is 0 (revisits same memory).
        """
        levels = []
        for ax in axes:
            if ax.step == 0:
                levels.append([0, ax.count])
            else:
                phys_stride = ax.step * source_strides[ax.dim]
                levels.append([phys_stride, ax.count])
        return levels

    @staticmethod
    def _merge_contiguous(axes, levels, element_shape):
        """Merge contiguous outer+inner pairs into a single AP level.

        Two pairs merge when (1) they sit on the same source dim, (2)
        neither is the SBUF partition axis (which must stay distinct so
        the P-walk and F-walk remain on separate AP levels), and (3) the
        outer's stride equals inner_stride * inner_count -- i.e. there's
        no gap between successive outer items. The merged count is then
        clamped to the source extent on that dim so a partial trailing
        tile doesn't push the walk past element_shape.

        Walks right-to-left so chains collapse to one level.
        """
        if len(levels) < 2 or len(axes) != len(levels):
            return APEmitter._merge_contiguous_no_axes(levels)

        result = list(levels)
        axis_per_level = list(axes)
        i = len(result) - 1
        while i > 0:
            outer_level = result[i - 1]
            inner_level = result[i]
            if APEmitter._can_merge(outer_level, inner_level, axis_per_level[i - 1], axis_per_level[i]):
                inner_axis = axis_per_level[i]
                # Walk extent of one inner-level iteration on the same dim:
                # sum of step*count across axes nested below inner on that dim.
                # When the inner axis itself is gappy (step > nested walk), the
                # last iteration's data span is < step, so the merged count
                # clamp must be (element - inner_walk) / step + 1, not
                # element / step.
                inner_walk = APEmitter._inner_walk_on_dim(
                    axis_per_level,
                    i,
                    inner_axis.dim,
                )
                merged_count = APEmitter._merged_count_clamped(
                    outer_level,
                    inner_level,
                    inner_axis,
                    element_shape,
                    inner_walk,
                )
                inner_stride = inner_level[0]
                result = result[: i - 1] + [[inner_stride, merged_count]] + result[i + 1 :]
                axis_per_level = axis_per_level[: i - 1] + [axis_per_level[i]] + axis_per_level[i + 1 :]
            i = i - 1
        return result

    @staticmethod
    def _inner_walk_on_dim(axis_per_level, inner_idx, dim):
        """Per-iteration data walk for the inner level on `dim`.

        Sums step*count for all axes nested below `inner_idx` (i.e., at
        index > inner_idx) on the same dim. For a non-gappy inner this
        equals inner_axis.step; for a gappy inner (step > nested walk)
        it's strictly smaller and is needed to compute a correct
        partial-trailing-tile clamp.
        """
        walk = 0
        for j in range(inner_idx + 1, len(axis_per_level)):
            ax = axis_per_level[j]
            if ax.dim != dim:
                continue
            if ax.step <= 0:
                continue
            walk = walk + ax.count * ax.step
        return walk

    @staticmethod
    def _can_merge(outer_level, inner_level, outer_axis, inner_axis):
        """True when outer+inner sit on the same dim, are non-partition, and contiguous."""
        if outer_axis.label == AxisLabel.PARTITION or inner_axis.label == AxisLabel.PARTITION:
            return False
        if outer_axis.dim != inner_axis.dim:
            return False
        outer_stride = outer_level[0]
        inner_stride, inner_count = inner_level
        return outer_stride == inner_stride * inner_count

    @staticmethod
    def _merged_count_clamped(outer_level, inner_level, inner_axis, element_shape, inner_walk=0):
        """Merged count = outer_level.count * inner_level.count, clamped to source extent.

        Counts come from the levels (post-merge state), not from the
        original axes, since this helper is called inside a chain-collapse
        loop where earlier merges have already grown the inner_level
        count.

        The clamp ensures the LAST merged iteration's data still fits in
        ``element_shape[dim]``. Iteration ``k`` lands at element position
        ``k * step`` and walks ``inner_walk`` data elements (sum of
        nested-axis walks on the same dim). So:

            (count - 1) * step + inner_walk <= element_shape[dim]
            count <= (element_shape[dim] - inner_walk) // step + 1

        For a non-gappy inner (step == inner_walk) this reduces to the
        familiar ``element_shape // step``; for a gappy inner
        (step > inner_walk, e.g. interleaved-shard tile-walk) it admits
        more iterations because each iteration's data fits inside the
        gap.
        """
        outer_count = outer_level[1]
        inner_count = inner_level[1]
        merged_count = outer_count * inner_count
        dim = inner_axis.dim
        if dim >= len(element_shape) or inner_axis.step <= 0:
            return merged_count
        if inner_walk <= 0:
            inner_walk = inner_axis.step
        if element_shape[dim] < inner_walk:
            return merged_count
        max_count_in_bounds = (element_shape[dim] - inner_walk) // inner_axis.step + 1
        if max_count_in_bounds <= 0:
            return merged_count
        return min(merged_count, max_count_in_bounds)

    @staticmethod
    def _merge_contiguous_no_axes(levels):
        """Fallback merge without axis info."""
        if len(levels) < 2:
            return levels
        result = list(levels)
        i = len(result) - 1
        while i > 0:
            outer_stride, outer_count = result[i - 1]
            inner_stride, inner_count = result[i]
            if outer_stride == inner_stride * inner_count:
                merged = [inner_stride, outer_count * inner_count]
                result = result[: i - 1] + [merged] + result[i + 1 :]
            i = i - 1
        return result

    @staticmethod
    def _hoist_partition(axes, source_strides, levels):
        """Move the partition-row level (if any) to the outermost AP position.

        SBUF stores by partition row first, then F columns. For HBM
        iteration to match that order, the partition walk (one element
        per row) must be the outermost AP level.
        """
        partition_axis = APEmitter._find_axis_by_label(axes, AxisLabel.PARTITION)
        if partition_axis is None:
            return levels
        partition_stride = partition_axis.step * source_strides[partition_axis.dim]

        for i in range(len(levels)):
            if levels[i][0] == partition_stride and levels[i][1] >= partition_axis.count:
                if i == 0:
                    return levels
                return [levels[i]] + levels[:i] + levels[i + 1 :]
        return levels

    @staticmethod
    def _drop_unit_levels(levels, partition_stride=None):
        """Drop [stride, 1] entries; keep at most one matching the partition stride.

        The partition axis is structurally required (SBUF AP must have a
        P-walk level even when only one partition row is written), so we
        preserve a single count=1 level matching the partition stride.
        Other count=1 levels (e.g. block / tile wrappers above the
        partition) are dropped.
        """
        result = []
        partition_seen = False
        for level in levels:
            if level[1] > 1:
                result.append(level)
                continue
            if partition_stride is not None and level[0] == partition_stride and not partition_seen:
                result.append(level)
                partition_seen = True
        return result

    @staticmethod
    def _partition_stride(axes, source_strides):
        """Return the partition axis's physical stride, or None if no partition axis."""
        for ax in axes:
            if ax.label == AxisLabel.PARTITION:
                return ax.step * source_strides[ax.dim]
        return None

    @staticmethod
    def _find_axis_by_label(axes, label):
        """Return the first axis with the given label, or None."""
        for ax in axes:
            if ax.label == label:
                return ax
        return None


__all__ = ["APEmitter"]
