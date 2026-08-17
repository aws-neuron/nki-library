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
from typing import TYPE_CHECKING, Optional, Union

import nki.language as nl

from ._helpers import NUM_HW_BANKS, PSUM_BANK_SIZE
from ._validation import _validate_psum_pool_grid_args
from .factories import _make_psum_ndslice

if TYPE_CHECKING:
    from .ndslice import NDSlice


def psum_pool(
    tile_size: tuple,
    grid: Optional[tuple] = None,
    element_shape: Optional[tuple] = None,
    bank_axis: Optional[int] = None,
    bank_ids: Optional[Union[tuple, list]] = None,
    dtype=None,
) -> "NDSlice":
    """psum_pool(tile_size, grid=None, element_shape=None, bank_axis=None, bank_ids=None, dtype=None) -> NDSlice

    Allocate one PSUM tile per grid cell and return them as an ``NDSlice`` indexed by
    grid coordinate (``psums[s, i].data``).

    PSUM is the per-NeuronCore matmul-output memory: 8 banks of 2048 elements each,
    written by ``nisa.nc_matmul`` with hardware-accumulate semantics. A pool holds
    several matmul accumulators and addresses them by coordinate, rather than by
    hand-assigned bank IDs.

    .. warning::

       This API is experimental and may change in future releases.

    Args:
        tile_size (tuple[int, ...]): Per-tile (P, F) shape (>= 2-D), positive ints.
        grid (tuple[int, ...] | None): Tile-aligned tile-grid shape (extent =
            ``grid * tile_size``). Mutually exclusive with ``element_shape``; prefer
            ``element_shape`` unless the extent is exactly tile-aligned.
        element_shape (tuple[int, ...] | None): Recommended. Actual per-dim extent,
            ceiling-divided by ``tile_size`` to derive the grid; a trailing partial
            tile auto-clamps. Mutually exclusive with ``grid``.
        bank_axis (int | None): The allocation-mode selector (see Allocation modes).
        bank_ids (tuple[int, ...] | list[int] | None): The hardware bank indices in
            ``[0, 8)``. None defers placement to the compiler; otherwise the length
            must match the selected mode.
        dtype: The element data type. Defaults to ``nl.float32``.

    Returns:
        NDSlice: A view over the freshly allocated banks, indexed by grid coordinate
        (``psums[s, i]``) -- see ``NDSlice`` for the view's attributes. ``.data`` is the
        per-tile ``nl.ndarray`` used as an ISA-op operand. Each grid tile is a separate
        ``nl.ndarray(buffer=nl.psum)``, keeping each matmul accumulator independent;
        packing several into one ndarray would tie them into a single accumulation group
        and fail to compile.

    Raises ``AssertionError`` in any of these cases:

    - ``tile_size`` is not a >= 2-D positive-int tuple.
    - neither or both of ``grid`` / ``element_shape`` is given.
    - ``grid`` / ``element_shape`` rank or entries disagree with ``tile_size``, or any
      entry is non-positive or a bool.
    - ``bank_axis`` is an int but ``bank_ids`` is None.
    - ``bank_ids`` is not a tuple/list of ints in ``[0, 8)`` (a bool entry is
      rejected).
    - the ``bank_ids`` length mismatches the mode (``product(grid)`` for all-fanout,
      ``grid[bank_axis]`` for slot-packed). When ``element_shape`` is given, these
      lengths are checked against the *derived* tile grid
      ``ceil(element_shape[d] / tile_size[d])``, not against ``element_shape`` itself.
    - all-fanout ``bank_ids`` are not unique or exceed 8 banks. (Slot-packed mode does
      not require unique ``bank_ids`` -- duplicates are accepted there.)
    - slot-packed ``bank_axis`` is out of range, the count along ``bank_axis``
      exceeds the 8 available PSUM banks, or the per-bank slot usage exceeds the
      2048-element bank capacity.

    Example:
        .. code-block:: python

            # Recommended: element_shape drives the allocation (partial tiles auto-clamp).
            psums = nt.psum_pool(tile_size=(128, 512), element_shape=(256, 2048))
            for s in range(2):
                for i in range(4):
                    nisa.nc_matmul(psums[s, i].data, x, w[s, i])

            # All-fanout: every tile on its own bank (needed once the non-bank slot
            # count would exceed the per-bank cap -- 2 slots at tile_f==256, 1 at
            # tile_f>=512; here the (2, 4) grid leaves no slot-packed option).
            psums = nt.psum_pool(tile_size=(128, 512), element_shape=(256, 2048),
                                 bank_ids=(0, 1, 2, 3, 4, 5, 6, 7))

            # Slot-packed: grid (2, 4), bank_axis=1 -> 4 banks, 2 slots/bank
            # (stride 512; the cap is 4 slots/bank at tile_f<=128).
            psums = nt.psum_pool(tile_size=(128, 128), element_shape=(256, 512),
                                 bank_axis=1, bank_ids=(0, 1, 2, 3))

    **Allocation modes (bank_axis + bank_ids)**::

        bank_axis  bank_ids   mode
        ---------  ---------  ----------------------------------------------------
        None       None       compiler-managed -- compiler picks banks
        None       provided   all-fanout -- every grid tile on its own bank
                              (len(bank_ids) == product(grid), entries unique)
        int        provided   slot-packed -- bank_axis fans across banks; the
                              other grid dims pack as slots within each bank
                              (len(bank_ids) == grid[bank_axis])
        int        None       rejected

    **Slot-stride rule (slot-packed mode).** ``tile_f`` -- the per-tile free-element
    count in this rule -- is the product of ``tile_size[1:]`` (every tile dim except the
    partition dim). Tiles sharing a bank must sit at least
    ``slot_stride = max(512, 4 * tile_f)`` F-elements apart, and
    ``slots_per_bank * slot_stride`` must fit the 2048-element bank. This caps
    slots/bank at 4 (``tile_f <= 128``), 2 (``tile_f == 256``), 1 (``tile_f >= 512``).
    For non-matmul accumulation (elementwise reductions), use FP32 SBUF instead --
    ``store()`` casts at DMA time, so PSUM banks should be reserved for
    ``nc_matmul`` outputs.

    Memory layout
    -------------
    Each per-tile PSUM ``nl.ndarray`` is allocated at exactly
    ``(tile_size[0], product(tile_size[1:]))`` -- partition extent ``tile_size[0]``, free
    extent the product of the remaining tile dims -- independent of ``grid`` /
    ``element_shape``. Unlike an SBUF ``alloc_tiles`` buffer, PSUM tiles are never
    P-folded into the free axis: a multi-P-tile grid yields multiple distinct
    ``(tile_size[0], tile_f)`` banks, so ``psums[s, i].data`` always has partition extent
    ``tile_size[0]``. On a partial trailing tile (an ``element_shape`` narrower than
    ``grid * tile_size``), ``psums[s, last].data`` auto-clamps to the addressable F
    width, so kernels write ``psums[s, i].data`` directly rather than slicing
    ``.data[:, :actual_f]``; the tile is still allocated at the uniform per-tile F
    extent so tiles land at distinct accumulator regions. Using ``grid`` instead
    over-allocates the trailing partial extent (rounded up, no clamp).

    See Also:
        alloc_tiles: allocate FP32 SBUF for non-matmul accumulation.

    """
    return _build_psum_ndslice(
        tile_size=tuple(tile_size),
        grid=tuple(grid) if grid is not None else None,
        element_shape=tuple(element_shape) if element_shape is not None else None,
        bank_axis=bank_axis,
        bank_ids=bank_ids,
        dtype=dtype,
    )


def _build_psum_ndslice(tile_size, grid, element_shape, bank_axis, bank_ids, dtype):
    """Validate, allocate one ndarray per grid tile, wrap as an NDSlice.

    Per-tile ndarrays keep each matmul accumulator independent: the
    compiler infers accumulation groups per ``nl.ndarray``, so packing
    multiple accumulators into one ndarray would tie them together and
    trip ``NCC_ISCH714``.

    Tile ``c`` is placed at
    ``address=(0, bank_id * PSUM_BANK_SIZE + slot_idx * slot_stride)``
    where ``slot_stride = max(512, 4 * tile_f)``. That stride matches
    the compiler's matmul accumulator-region quantum -- successive
    ndarrays in the same physical bank must be at least
    ``slot_stride`` F-elements apart or NCC reports ``NCC_ISCH714`` /
    ``NCC_IBIR110``. Some F-space inside each bank is intentionally
    unused so successive tiles land at distinct accumulator regions.
    """
    tile_grid_shape, num_banks, slots_per_bank, tile_f = _validate_psum_pool_grid_args(
        tile_size=tile_size,
        grid=grid,
        element_shape=element_shape,
        bank_axis=bank_axis,
        bank_ids=bank_ids,
        psum_bank_size=PSUM_BANK_SIZE,
        num_hw_banks=NUM_HW_BANKS,
    )

    _dtype = dtype if dtype is not None else nl.float32
    tile_p = tile_size[0]
    matmul_quantum = 512
    slot_stride = max(matmul_quantum, 4 * tile_f)

    tile_arrays = []
    if bank_ids is None:
        # Compiler-managed: no explicit address=, one ndarray per tile.
        for _ in range(num_banks * slots_per_bank):
            tile_arrays.append(
                nl.ndarray(
                    (tile_p, tile_f),
                    dtype=_dtype,
                    buffer=nl.psum,
                )
            )
    elif bank_axis is None:
        # Every tile on its own bank (slots_per_bank == 1).
        for tile_idx in range(num_banks):
            bank_id = bank_ids[tile_idx]
            tile_arrays.append(
                nl.ndarray(
                    (tile_p, tile_f),
                    dtype=_dtype,
                    buffer=nl.psum,
                    address=(0, bank_id * PSUM_BANK_SIZE),
                )
            )
    else:
        # Pack non-bank dims as slots within each bank.
        for bank_idx in range(num_banks):
            bank_id = bank_ids[bank_idx]
            for slot_idx in range(slots_per_bank):
                f_offset = bank_id * PSUM_BANK_SIZE + slot_idx * slot_stride
                tile_arrays.append(
                    nl.ndarray(
                        (tile_p, tile_f),
                        dtype=_dtype,
                        buffer=nl.psum,
                        address=(0, f_offset),
                    )
                )

    return _make_psum_ndslice(
        tile_arrays=tuple(tile_arrays),
        alloc_tile_size=tile_size,
        grid_shape=tile_grid_shape,
        bank_axis=bank_axis,
        slots_per_bank=slots_per_bank,
        dtype=_dtype,
        element_shape=element_shape,
    )
