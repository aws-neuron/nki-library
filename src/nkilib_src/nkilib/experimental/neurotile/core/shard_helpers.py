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
from typing import Any, Optional, Union

import nki.language as nl


def _validate_shard_args(num_shards, total, caller):
    """Shared positive-int check for shard helper arg dims.

    `num_shards` and `total` must be compile-time positive ints.
    `rank` is checked at the call site (it can be a runtime expression).
    """
    assert isinstance(num_shards, int) and not isinstance(num_shards, bool), (
        caller + ": num_shards must be an int, got " + str(num_shards)
    )
    assert num_shards >= 1, caller + ": num_shards must be >= 1, got " + str(num_shards)
    assert isinstance(total, int) and not isinstance(total, bool), caller + ": total must be an int, got " + str(total)
    assert total >= 1, caller + ": total must be >= 1, got " + str(total)


def block_range(rank: Union[int, Any], num_shards: int, total: int) -> slice:
    """block_range(rank, num_shards, total) -> slice

    Build a contiguous-block shard range for one core: each core owns
    ``total // num_shards`` consecutive units along the shard dimension.

    Sharding in NeuroTile is expressed as slicing. The returned ``slice`` is used to
    index a view, which restricts that view to one core's units; there is no separate
    sharding API. Like ``uneven_block_range`` and ``interleaved_range``, this helper
    takes only scalar counts and returns a plain Python ``slice`` -- it touches no
    tensor or view and performs no allocation or DMA, so one slice can be reused across
    any views that share the same unit count on the shard dimension.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        rank (int | runtime scalar): This core's rank index. A compile-time ``int``
            produces a fixed range; a runtime value such as ``nl.program_id(0)``
            produces a range resolved per core at run time. ``rank`` is not
            type-validated. When each core owns more than one unit and ``rank`` is a
            runtime value, the slice start is the runtime product ``rank * owned``; the
            ``owned == 1`` case uses ``rank`` directly with no multiply.
        num_shards (int): The total number of cores, a positive int. It must divide
            ``total`` evenly; when it may not, use ``uneven_block_range``.
        total (int): The total number of units along the shard dimension, a positive
            int. A unit is whatever the slice indexes -- tiles for ``nt.tiles(...)``,
            blocks for ``nt.blocks(...)`` -- and is computed with
            ``nt.ceiling_div(dim, grain)``.

    Returns:
        slice: A ``slice(start, start + owned)`` with ``owned = total // num_shards``,
        covering this core's contiguous range of units.

    Raises ``AssertionError`` in any of these cases:

    - ``num_shards`` is not a positive int (a bool is rejected), or ``num_shards < 1``.
    - ``total`` is not a positive int (a bool is rejected), or ``total < 1``.
    - ``total`` is not divisible by ``num_shards`` (the message directs the caller to
      ``uneven_block_range`` for remainder distribution).

    Example:
        .. code-block:: python

            own = nt.block_range(nl.program_id(0), nl.num_programs(0), nt.ceiling_div(M, 128))
            view = nt.tiles(src, tile_size=(128, 512))[own, :]

    With 8 units and ``num_shards=4``, each core owns 2 consecutive units::

        unit:   0   1   2   3   4   5   6   7
              ┌───┬───┬───┬───┬───┬───┬───┬───┐
              │ c0│ c0│ c1│ c1│ c2│ c2│ c3│ c3│
              └───┴───┴───┴───┴───┴───┴───┴───┘
        core0 = [0,1]  core1 = [2,3]  core2 = [4,5]  core3 = [6,7]

    See Also:
        uneven_block_range: contiguous, with the remainder spread to early cores.
        interleaved_range: round-robin distribution.

    """
    _validate_shard_args(num_shards, total, "block_range")
    assert total % num_shards == 0, (
        "block_range requires total ("
        + str(total)
        + ") divisible by num_shards ("
        + str(num_shards)
        + "). Use uneven_block_range for remainder distribution."
    )
    owned = total // num_shards
    # Skip `rank * 1` -- Beta 3 parser rejects runtime_scalar * 1.
    if owned == 1:
        start = rank
    else:
        start = rank * owned
    return slice(start, start + owned)


def uneven_block_range(rank: Union[int, Any], num_shards: int, total: int) -> slice:
    """uneven_block_range(rank, num_shards, total) -> slice

    Build a contiguous-block shard range with the remainder spread to the early
    cores: when ``total`` does not divide evenly, cores ``[0, total % num_shards)``
    each receive one extra unit.

    This is the choice when the unit count is not a multiple of the core count and
    every unit must still be covered exactly once.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        rank (int | runtime scalar): This core's rank index. It must be a compile-time
            ``int`` when ``total`` does not divide evenly, because the owned count then
            varies per core and cannot be resolved from a runtime value. A runtime
            value is accepted only when ``total`` divides evenly, in which case this
            delegates to ``block_range``.
        num_shards (int): The total number of cores, a positive int. Unlike
            ``block_range`` it need not divide ``total`` evenly (the remainder is
            distributed).
        total (int): The total number of units along the shard dimension, a positive
            int.

    Returns:
        slice: A ``slice(start, start + owned)`` for this core's contiguous range.
        Cores with ``rank < (total % num_shards)`` own ``base + 1`` units each (where
        ``base = total // num_shards``); the rest own ``base``. Consecutive ranks'
        ranges abut with no gap or overlap, so the union over all ranks tiles
        ``[0, total)`` exactly once and the last rank's stop equals ``total``.

    Raises ``AssertionError`` in any of these cases:

    - ``num_shards`` or ``total`` is not a positive int (a bool is rejected, and each
      must be ``>= 1``).
    - ``rank`` is a runtime (non-int) value while ``total`` is not divisible by
      ``num_shards`` (the per-core owned count then varies and cannot be resolved from
      a runtime value).

    Example:
        .. code-block:: python

            # 5 cores, 12 units -> ranks 0,1 own 3 each; ranks 2,3,4 own 2.
            r = nt.uneven_block_range(rank=nl.program_id(0), num_shards=nl.num_programs(0), total=N)
            view = nt.tiles(src, tile_size=(128, 512))[r, :]

    See Also:
        block_range: even contiguous distribution.

    """
    _validate_shard_args(num_shards, total, "uneven_block_range")
    base_units = total // num_shards
    remainder = total % num_shards

    if not isinstance(rank, int):
        # Runtime rank: must divide evenly (per-rank branch impossible).
        assert remainder == 0, (
            "uneven_block_range with runtime rank requires total ("
            + str(total)
            + ") divisible by num_shards ("
            + str(num_shards)
            + "). Use block_range or adjust dimensions."
        )
        return block_range(rank, num_shards, total)

    # Compile-time rank: ranks [0, remainder) get base+1, rest get base.
    if rank < remainder:
        owned = base_units + 1
        extra_before = rank
    else:
        owned = base_units
        extra_before = remainder
    start = rank * base_units + extra_before
    return slice(start, start + owned)


def interleaved_range(rank: Union[int, Any], num_shards: int, total: int) -> slice:
    """interleaved_range(rank, num_shards, total) -> slice

    Build a round-robin shard range for one core: core ``r`` owns every
    ``num_shards``-th unit, starting at ``r``.

    Round-robin balances non-uniform per-unit work better than contiguous blocks --
    for example causal attention, where later tiles cost more than earlier ones, so a
    contiguous split would leave the last core with the most work. The returned slice
    is strided, with ``step == num_shards``.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        rank (int | runtime scalar): This core's rank index, used directly as the slice
            start. A compile-time int or a runtime scalar is accepted with no extra
            restriction (unlike ``uneven_block_range``, this helper has no per-rank
            branching, so it never requires a compile-time rank). Not type-validated.
        num_shards (int): The total number of cores, a positive int. It must divide
            ``total``, and becomes the slice step.
        total (int): The total number of units along the shard dimension, a positive
            int; it becomes the slice stop.

    Returns:
        slice: A ``slice(rank, total, num_shards)`` -- a strided range with
        ``step == num_shards`` that addresses only this core's units (core ``r`` owns
        ``r``, ``r + num_shards``, ``r + 2 * num_shards``, ...).

    Raises ``AssertionError`` in any of these cases:

    - ``num_shards`` or ``total`` is not a positive int (a bool is rejected, and each
      must be ``>= 1``).
    - ``total`` is not divisible by ``num_shards``.

    Example:
        .. code-block:: python

            own = nt.interleaved_range(nl.program_id(0), nl.num_programs(0), nt.ceiling_div(M, 128))
            view = nt.tiles(src, tile_size=(128, 256))[own, :]

    With 8 units and ``num_shards=4``, each core owns every 4th unit (contrast
    ``block_range``, where each core owns a contiguous pair)::

        unit:   0   1   2   3   4   5   6   7
              ┌───┬───┬───┬───┬───┬───┬───┬───┐
              │ c0│ c1│ c2│ c3│ c0│ c1│ c2│ c3│
              └───┴───┴───┴───┴───┴───┴───┴───┘
        core0 = [0,4]  core1 = [1,5]  core2 = [2,6]  core3 = [3,7]

    See Also:
        block_range: contiguous distribution.

    """
    _validate_shard_args(num_shards, total, "interleaved_range")
    assert total % num_shards == 0, (
        "interleaved_range requires total (" + str(total) + ") divisible by num_shards (" + str(num_shards) + ")."
    )
    return slice(rank, total, num_shards)


def get_shard_info(
    tensor_shape: tuple,
    tile_size: tuple,
    shard_dim: int = 0,
    num_shards: Optional[int] = None,
    shard_id: Optional[Any] = None,
) -> dict:
    """get_shard_info(tensor_shape, tile_size, shard_dim=0, num_shards=None, shard_id=None) -> dict

    Compute a partition summary for a sharded tile grid.

    This is a diagnostic helper. It does not drive the runtime DMA path; it is
    intended for trace-time printing or for asserting the expected partition
    structure.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        tensor_shape (tuple[int, ...]): The element shape of the source tensor. Must be
            a tuple/list; its individual entries are not checked to be ints.
        tile_size (tuple[int, ...]): The tile shape (same rank as ``tensor_shape``),
            which determines the tile-grid extent. Must be a tuple/list; its individual
            entries are not checked to be ints.
        shard_dim (int): The dimension along which sharding splits tiles. Defaults
            to 0. Must be an int in ``[0, len(tensor_shape))``.
        num_shards (int | None): The total number of cores. Defaults to
            ``nl.num_programs(0)`` (a runtime scalar). Not validated -- see Notes.
        shard_id (int | runtime scalar | None): This core's rank. Defaults to
            ``nl.program_id(0)``. Accepted as an int, runtime scalar, or None, and
            echoed into the result unchanged; it is neither type-checked nor compared
            against ``num_shards``.

    Returns:
        dict: Keys ``total_tiles``, ``tiles_per_shard``, ``shard_id``, ``num_shards``,
        ``shard_dim``, where ``total_tiles = tensor_shape[shard_dim] //
        tile_size[shard_dim]`` and ``tiles_per_shard = total_tiles // num_shards``
        (floor division; when ``num_shards`` does not divide ``total_tiles`` the summary
        undercounts and no error is raised). When ``num_shards`` / ``shard_id`` come
        from their ``nl.*`` defaults the corresponding values are runtime scalars, so
        the dict may hold runtime-valued entries rather than plain ints.

    Raises ``AssertionError`` in any of these cases:

    - ``tensor_shape`` or ``tile_size`` is not a tuple/list.
    - their ranks (lengths) differ.
    - ``shard_dim`` is not an int in ``[0, len(tensor_shape))`` (a bool is rejected).
    - ``tensor_shape[shard_dim]`` is not divisible by ``tile_size[shard_dim]``.

    Example:
        .. code-block:: python

            info = nt.get_shard_info((1024, 4096), (128, 512), shard_dim=0, num_shards=2, shard_id=0)
            # info["total_tiles"] == 8, info["tiles_per_shard"] == 4

    Notes:
        This helper validates loosely because it is a diagnostic, not a runtime-DMA
        driver. It checks only: ``tensor_shape`` / ``tile_size`` are equal-length
        tuples/lists, ``shard_dim`` is in range, and the shard dimension divides evenly
        (in that order -- a bad ``shard_dim`` is reported before a divisibility
        problem). It does **not** validate ``num_shards`` (a zero, negative, bool, or
        runtime-scalar value passes; a literal ``0`` raises ``ZeroDivisionError`` rather
        than ``AssertionError`` when ``tiles_per_shard`` is computed), does not check
        that the shape entries are ints, checks divisibility only on the shard
        dimension, and never asserts ``shard_id < num_shards``.

    See Also:
        block_range: the slice helper that performs the actual sharding.

    """
    assert isinstance(tensor_shape, (tuple, list)), "get_shard_info: tensor_shape must be a tuple/list, got " + str(
        tensor_shape
    )
    assert isinstance(tile_size, (tuple, list)), "get_shard_info: tile_size must be a tuple/list, got " + str(tile_size)
    assert len(tensor_shape) == len(tile_size), (
        "get_shard_info: tensor_shape and tile_size must have the same "
        "rank; got tensor_shape rank " + str(len(tensor_shape)) + " vs tile_size rank " + str(len(tile_size)) + "."
    )
    assert isinstance(shard_dim, int) and not isinstance(shard_dim, bool), (
        "get_shard_info: shard_dim must be an int, got " + str(shard_dim)
    )
    assert 0 <= shard_dim < len(tensor_shape), (
        "get_shard_info: shard_dim="
        + str(shard_dim)
        + " is out of range for tensor_shape rank "
        + str(len(tensor_shape))
        + "."
    )

    if num_shards is None:
        num_shards = nl.num_programs(0)
    if shard_id is None:
        shard_id = nl.program_id(0)

    assert tensor_shape[shard_dim] % tile_size[shard_dim] == 0, (
        "get_shard_info: tensor_shape["
        + str(shard_dim)
        + "]="
        + str(tensor_shape[shard_dim])
        + " not divisible by tile_size["
        + str(shard_dim)
        + "]="
        + str(tile_size[shard_dim])
    )
    total_tiles = tensor_shape[shard_dim] // tile_size[shard_dim]
    tiles_per_shard = total_tiles // num_shards

    return {
        "total_tiles": total_tiles,
        "tiles_per_shard": tiles_per_shard,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "shard_dim": shard_dim,
    }


__all__ = [
    "block_range",
    "uneven_block_range",
    "interleaved_range",
    "get_shard_info",
]
