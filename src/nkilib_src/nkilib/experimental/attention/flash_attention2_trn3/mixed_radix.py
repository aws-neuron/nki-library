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

"""Flattening and unflattening a loop nest's index space, in row-major order.

A multi-dimensional loop index is a number in a mixed-radix system whose digits are the grid's
extents, which is what lets the pipeline address "the point ``n`` steps from here" as one operation
instead of per-dimension bookkeeping. Deliberately three trace-time functions rather than a class:
the kernel only ever needs a linear step converted to digits and back.
"""

from typing import Optional, Tuple

from ....core.utils.kernel_assert import kernel_assert

IDX = Tuple[int, ...]


def radix_size(grid: IDX) -> int:
    """Total number of points in ``grid``."""
    total = 1
    for dim in range(len(grid)):
        total = total * grid[dim]
    return total


def unflatten(step: int, grid: IDX) -> IDX:
    """Convert a linear position into the grid index it names, least-significant dimension last.

    Args:
        step: linear position, ``0 <= step < radix_size(grid)``. Out of range is a caller bug, not
            a wrap: every caller derives ``step`` from the walk length.
        grid: the iteration space's extents.

    Returns:
        The grid index of ``step``.
    """
    kernel_assert(0 <= step < radix_size(grid), f"step {step} is outside grid {grid}")
    out = [0] * len(grid)
    carry = step
    for dim in range(len(grid) - 1, -1, -1):
        out[dim] = carry % grid[dim]
        carry = carry // grid[dim]
    return tuple(out)


def flatten(index: Tuple[Optional[int], ...], grid: IDX) -> int:
    """Convert a grid index into a linear position over the dimensions it names.

    ``index`` may hold ``None`` for dimensions the caller's quantity does not depend on; those
    dimensions are dropped rather than treated as zero, so the result is the row-major linear
    position within the sub-grid of the retained dimensions. That is injective in the retained
    digits, but it is *not* the position within the full ``grid``, so two indices with different
    ``None`` patterns are not comparable.

    Args:
        index: one entry per dimension of ``grid``, ``None`` for dimensions to drop.
        grid: the iteration space's extents.

    Returns:
        The linear position of ``index`` within the retained dimensions.
    """
    kernel_assert(len(index) == len(grid), f"index rank {len(index)} != grid rank {len(grid)}")
    step = 0
    radix = 1
    for dim in range(len(grid) - 1, -1, -1):
        if index[dim] != None:
            kernel_assert(0 <= index[dim] < grid[dim], f"index {index} is outside grid {grid}")
            step = step + index[dim] * radix
            radix = radix * grid[dim]
    return step
