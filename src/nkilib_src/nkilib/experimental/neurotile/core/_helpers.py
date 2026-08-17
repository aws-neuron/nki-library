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

# ============================================================================
# Hardware constants
# ============================================================================

P_DIM = 0
MIN_TILED_DIMS = 2
MAX_SBUF_PARTITION_ROWS = 128

# PSUM hardware capacity (per bank) and bank count.
PSUM_BANK_SIZE = 2048
NUM_HW_BANKS = 8


# ============================================================================
# Partial-trailing-tile / block reachability -- shared offset-clamp rule.
# ============================================================================


def reachable_dim_extent(grid, layout, dim):
    """Source elements on ``dim`` reachable from the view's origin.

    Shared by view-shape reporting and DMA sizing so they agree. Subtracts the
    origin offset on a partial trailing tile/block (``dim in remainder_dims``
    or ``element_shape[dim] > remaining[dim]``); a slice/reshape_dim is left
    untouched since it already narrows ``element_shape`` to the walk.
    """
    source_extent = grid.element_shape[dim]
    if dim in grid.remainder_dims or source_extent > grid.remaining[dim]:
        return source_extent - layout.dim_offset_elements(dim, grid.element_shape)
    return source_extent


# ============================================================================
# Indexing validation -- single chokepoint for user-facing key checks.
# Used by NDSlice._index_multi and SBUFLayout.sub_index.
# ============================================================================


def level_hint(grid, dim):
    """Return an actionable hint for indexing surprises on `dim`.

    Users most often hit these when they expect tile-level semantics but
    the view is at a coarser level (block or shard outer). The hint
    points at the exact descent needed to restore intuition.
    """
    if grid is None:
        return ""
    if grid.is_sharded(dim):
        return (
            " This view has a shard axis on this dim; "
            "index a specific tile first (e.g. view[..., i]) or iterate "
            "via .tolist() to walk owned tiles."
        )
    if grid.is_blocked(dim):
        bs = grid.block_size_of(dim)
        if bs is not None and bs > 1:
            return (
                " This is a block-level view on this dim; if you meant tile indices, descend first with nt.tiles(view)."
            )
    return ""


def validate_index_key(key, dim, extent, grid=None, context=""):
    """Validate an indexing key against the current-level extent on `dim`.

    Accepted key types:
      - ``int`` (compile-time): ``-extent <= key < extent``.
      - ``slice``: step >= 1; non-empty range within ``[0, extent]``.
        Step > 1 routes through interleaved-shard handling.
      - Runtime scalar / vector (NKI CExpr, LoopVar, or 1-D ``NDSlice``):
        bounds-checked at runtime by NKI; trace-time skip.

    Rejected (with named asserts):
      - ``bool`` (``True``/``False`` look like ``int 0/1`` to Python; reject
        explicitly so users hit a clear error rather than silent off-by-one).
      - Plain Python collections / scalars (``list``, ``dict``, ``str``,
        ``float``, ``None``, ``Ellipsis``) -- not valid index types.
    """
    ctx = context if context else "index"

    # Reject bool first -- isinstance(True, int) is True.
    assert not isinstance(key, bool), (
        ctx + ": bool index " + str(key) + " is not allowed on dim " + str(dim) + "; use an int (0 or 1) explicitly."
    )

    if isinstance(key, int):
        assert extent is not None and extent > 0 and -extent <= key < extent, (
            ctx + f": int index {key} out of range on dim {dim} "
            f"(valid range: [{-extent}, {extent}))." + level_hint(grid, dim)
        )
        return

    if isinstance(key, slice):
        step = key.step if key.step is not None else 1
        assert step >= 1, ctx + f": slice step={step} must be >= 1 on dim {dim}."
        start = key.start if key.start is not None else 0
        stop = key.stop if key.stop is not None else extent

        # Skip bounds checks when start or stop is runtime (non-int) --
        # the tracer can't evaluate the comparison.
        if not isinstance(start, int) or not isinstance(stop, int):
            return

        assert extent is not None and extent >= 0, ctx + f": cannot slice dim {dim} with unknown extent."
        assert 0 <= start <= stop <= extent, (
            ctx + f": slice [{start}:{stop}:{step}] out of range on dim {dim} "
            f"(valid count: {extent})." + level_hint(grid, dim)
        )
        assert stop > start, (
            ctx + f": empty slice [{start}:{stop}] on dim {dim}. "
            "Empty views have no valid downstream use; use [a:a+1] for a "
            "single item, or drop the dim entirely."
        )
        return

    # Reject plain Python collections and string/float/bytes types up
    # front so misuse hits a clear named error instead of cryptic
    # trace-time failures inside Layout/Grid. Runtime scalars / vectors
    # (NKI CExpr, LoopVar, NDSlice) are NKIObject subclasses; they pass
    # this check and NKI bounds-checks them at runtime.
    assert not isinstance(key, (list, dict, str, float)), (
        ctx
        + ": unsupported index type on dim "
        + str(dim)
        + " (got "
        + str(key)
        + "). Indexing accepts int, slice, or NKI runtime "
        "expression (LoopVar / SBUF NDSlice)."
    )


# ============================================================================
# Tuple operations
# ============================================================================


def replace_at(tup, index, value):
    """Return tuple with element at index replaced by value."""
    result = []
    for i in range(len(tup)):
        if i == index:
            result.append(value)
        else:
            result.append(tup[i])
    return tuple(result)


def remove_at(tup, index):
    """Return tuple with element at index removed."""
    result = []
    for i in range(len(tup)):
        if i != index:
            result.append(tup[i])
    return tuple(result)


def insert_at(tup, index, value):
    """Return tuple with value inserted at index."""
    result = []
    for i in range(len(tup)):
        if i == index:
            result.append(value)
        result.append(tup[i])
    if index == len(tup):
        result.append(value)
    return tuple(result)


def zeros(n):
    """Tuple of n zeros (NKI-safe: no generator expressions)."""
    result = []
    for _ in range(n):
        result.append(0)
    return tuple(result)


def product(tup, start=0):
    """Product of tup[start:]."""
    result = 1
    for i in range(start, len(tup)):
        result = result * tup[i]
    return result


def ceiling_div(a: int, b: int) -> int:
    """ceiling_div(a, b) -> int

    Integer ceiling division: the smallest int ``q`` with ``q * b >= a``.

    Pure Python and safe at kernel trace time. It is the correct operator for
    tile-count math when an extent is not divisible by the tile size, where plain
    ``//`` would floor and silently drop the remainder tile.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        a (int): Dividend; non-negative int (zero is allowed).
        b (int): Divisor; positive int (strictly ``> 0``).

    Returns:
        int: ``ceil(a / b)``, computed as ``(a + b - 1) // b``. ``ceiling_div(0, b)`` is
        ``0`` for any positive ``b`` (a zero dividend yields a zero count, not 1).

    Raises ``AssertionError`` if ``a`` is not a non-negative int (a bool or a negative
    value is rejected), or ``b`` is not a positive int (a bool, zero, or negative value
    is rejected).

    Example:
        .. code-block:: python

            n_tiles = nt.ceiling_div(M, 128)   # 130 -> 2, 256 -> 2, 257 -> 3

    """
    assert isinstance(a, int) and not isinstance(a, bool) and a >= 0, (
        "nt.ceiling_div: a must be a non-negative int, got " + str(a)
    )
    assert isinstance(b, int) and not isinstance(b, bool) and b > 0, (
        "nt.ceiling_div: b must be a positive int, got " + str(b)
    )
    return (a + b - 1) // b


def largest_divisor(n: int, max_val: int) -> int:
    """largest_divisor(n, max_val) -> int

    Largest divisor of ``n`` that is ``<= max_val`` (or 1 when none exists, for
    example when ``n`` is prime and ``max_val < n``).

    Pure Python and safe at kernel trace time. A typical use is tile- or block-group
    sizing -- for example, choosing a K-group size that evenly divides the K-tile
    count while still fitting within the PSUM budget.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        n (int): Number to find a divisor of; positive int (``>= 1``).
        max_val (int): Upper bound on the divisor; positive int (``>= 1``).

    Returns:
        int: The largest ``d`` with ``d <= max_val`` and ``n % d == 0``, found by
        scanning downward from ``min(n, max_val)`` and returning the first divisor (the
        descending scan makes that first hit the largest). When ``max_val >= n`` the
        result is ``n`` itself, so the divisor is capped at ``n`` regardless of
        ``max_val``. Returns 1 when no divisor in ``[1, max_val]`` exists (since 1
        always divides ``n``, the result is never absent).

    Raises ``AssertionError`` if ``n`` or ``max_val`` is not a positive int (a bool,
    zero, or negative value is rejected).

    Example:
        .. code-block:: python

            K_GROUP_SIZE = nt.largest_divisor(K_tiles, max_val=8)

    """
    assert isinstance(n, int) and not isinstance(n, bool) and n >= 1, (
        "nt.largest_divisor: n must be a positive int, got " + str(n)
    )
    assert isinstance(max_val, int) and not isinstance(max_val, bool) and max_val >= 1, (
        "nt.largest_divisor: max_val must be a positive int, got " + str(max_val)
    )
    result = 1
    for candidate in range(min(n, max_val), 0, -1):
        if n % candidate == 0:
            result = candidate
            break
    return result


def tiles_along_dim(remaining_dim, tile_size_dim):
    """Iteration tile count: ceil(remaining / tile_size). For Grid shape computation."""
    if tile_size_dim <= 0:
        return 1
    return ceiling_div(remaining_dim, tile_size_dim)


def p_tile_count(p_remaining, tile_p):
    """Number of P-tile repetitions in a multi-P-tile SBUF region.

    Ceiling division so a partial trailing P-tile counts as one tile: a sub-tile
    extent (``p_remaining < tile_p``, e.g. an element_shape whose partition
    extent is smaller than tile_p) yields 1, not 0. Bare floor division returned
    0, which propagated into an AP level count and produced an invalid
    ``(partition_stride, 0)`` pattern.
    """
    if tile_p <= 0:
        return 1
    return ceiling_div(p_remaining, tile_p)


def contiguous_strides(shape):
    """Row-major contiguous strides: stride[d] = product(shape[d+1:])."""
    result = []
    for d in range(len(shape)):
        stride = 1
        for d2 in range(d + 1, len(shape)):
            stride = stride * shape[d2]
        result.append(stride)
    return tuple(result)


def contiguous_ap_pattern(shape):
    """Build N-D contiguous AP pattern: [[stride_0, D0], ..., [1, Dn]].

    stride_i = product(shape[i+1:]).
    """
    ndim = len(shape)
    strides = []
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides.append(stride)
        stride = stride * shape[i]
    strides.reverse()

    pattern = []
    for i in range(ndim):
        pattern.append([strides[i], shape[i]])
    return pattern


# ============================================================================
# Buffer type constants
# ============================================================================


def hbm_buffer_type():
    """HBM buffer type constant."""
    return nl.shared_hbm


def sbuf_buffer_type():
    """SBUF buffer type constant."""
    return nl.sbuf


# ============================================================================
# Source tensor introspection -- used by factories to distinguish top-level
# NKI tensors from sliced views, and to read physical memory layout.
# ============================================================================


def physical_strides(tensor, fallback_shape):
    """Element strides of `tensor`'s physical memory layout.

    For NKI tensors, reads `get_pattern()` which reports parent strides
    on sliced views. For non-NKI sources (test mocks, numpy) that lack
    `get_pattern`, falls back to contiguous strides of `fallback_shape`.
    """
    if hasattr(tensor, "get_pattern"):
        pattern = tensor.get_pattern()
        result = []
        for level in pattern:
            result.append(level[0])
        return tuple(result)
    return contiguous_strides(tuple(fallback_shape))


def buffer_space(source):
    """Memory space of ``source`` -> ``nl.sbuf`` / ``nl.psum`` / ``nl.shared_hbm``.

    Reads ``source.buffer`` (an ``nl.MemoryRegion`` that survives slicing and is
    carried by every NkiTensor, including ``NDSlice.data`` views). Anything that
    is not SBUF/PSUM -- a named HBM region, or a non-NKI source (numpy, test
    mock) with no ``.buffer`` -- is HBM.
    """
    if hasattr(source, "buffer"):
        if source.buffer == nl.sbuf:
            return nl.sbuf
        if source.buffer == nl.psum:
            return nl.psum
    return nl.shared_hbm


# ============================================================================
# NkiTensor private-API adapters -- the ONLY places neurotile touches a private
# NkiTensor member. NkiTensor ships in an external wheel, so confining the
# private access here gives a single fix-point if the upstream contract drifts;
# test_nkitensor_contract.py guards these so a drift fails fast.
# ============================================================================


def nki_strided_view(shape, strides, dtype, buffer):
    """A memory-less ``NkiTensor`` carrying ``shape`` + arbitrary ``strides``.

    Used purely as a stride calculator for transforms (storage is None, so it
    never materializes). ``.ap()`` cannot build this -- it rejects a logical
    partition stride -- so this is the one path that uses ``NkiTensor._copy``.
    """
    view = nl.NkiTensor(shape=tuple(shape), dtype=dtype, storage=None, buffer=buffer)
    return view._copy(shape=tuple(shape), strides=tuple(strides))


def physical_row_width(tensor):
    """Element distance between adjacent partition rows of ``tensor``'s storage.

    Reads the physical allocation free width (``_storage_shape``), which NKI
    preserves through ``[:, slice]`` indexing -- so a sliced view still reports
    the parent's row width. Non-NKI sources (test mock) fall back to ``shape``.
    ``hasattr`` (not ``getattr``) because the tracer rejects builtins.getattr.
    """
    if hasattr(tensor, "_storage_shape"):
        underlying = tensor._storage_shape
    else:
        underlying = tensor.shape
    return product(tuple(underlying), start=1)
