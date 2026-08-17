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

"""HBMStream: Tile iteration abstraction for HBM tensors."""

from typing import Tuple, Union

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ..iter_order import ColMajor, RowMajor


def div_ceil(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def prepare_hbm_view(
    hbm_source: Union[nl.NkiTensor, nl.NkiTensor],
    tile_shape: Tuple[int, ...],
) -> Tuple[nl.NkiTensor, Tuple[int, ...]]:
    """Prepare HBM view for tiled access and compute tile_dims.

    Returns (view, tile_dims) where tile_dims maps tile dimensions to physical dims.
    P is always placed at dim 0; F is always the last dim."""
    if isinstance(hbm_source, nl.NkiTensor):
        view = hbm_source
    else:
        view = hbm_source

    ndim = len(view.shape)

    # HBM view shape must match tile_shape dimensions
    # For multi-P-tile access, caller must reshape HBM to (pdim, n_p_tiles, *F) to match SBUF

    physical_ndim = len(view.shape)
    tile_ndim = len(tile_shape)
    tile_dims = [0]
    for i in range(1, tile_ndim):
        tile_dims.append(physical_ndim - tile_ndim + i)

    return view, tuple(tile_dims)


class HBMStream(nl.NKIObject):
    """Tile iteration over HBM tensor, matching TileStream's iteration pattern."""

    def __init__(
        self,
        hbm_source: Union[nl.NkiTensor, nl.NkiTensor],
        tile_shape: Tuple[int, ...],
        iter_order: Union[RowMajor, ColMajor] = None,
        tile_dims: Tuple[int, ...] = None,
    ) -> None:
        """
        Args:
            hbm_source: HBM tensor or nl.NkiTensor
            tile_shape: Shape of each tile
            iter_order: Iteration order (RowMajor or ColMajor). Defaults to RowMajor.
            tile_dims: Which logical dims to tile. If None, defaults to trailing dims.
        """
        if isinstance(hbm_source, nl.NkiTensor):
            self._hbm_view = hbm_source
        else:
            self._hbm_view = hbm_source

        self._tile_shape = tile_shape
        self._iter_order = iter_order if iter_order is not None else RowMajor()
        self._physical_shape = tuple(self._hbm_view.shape)

        kernel_assert(tile_dims is not None, "HBMStream requires explicit tile_dims")
        self._tile_dims = tile_dims

        # Compute grid_dims (dims not in tile_dims)
        self._grid_dims = self._compute_grid_dims()

        # Pre-compute is_grid_only lookup
        physical_ndim = len(self._physical_shape)
        self._is_grid_only = []
        for d in range(physical_ndim):
            self._is_grid_only.append(d not in self._tile_dims)

        # Pre-compute tile size for each dim (1 for grid-only)
        self._tile_sizes = []
        for d in range(physical_ndim):
            self._tile_sizes.append(1)
        for i in range(len(self._tile_dims)):
            self._tile_sizes[self._tile_dims[i]] = self._tile_shape[i]

        # Compute tile grid
        self._tile_grid = self._compute_tile_grid()

        # Current tile position
        self._cur_tile = []
        for _ in range(len(self._tile_grid)):
            self._cur_tile.append(0)

    def _compute_grid_dims(self) -> Tuple[int, ...]:
        """Compute which logical dims are grid-only."""
        physical_ndim = len(self._physical_shape)
        grid_dims = []
        for dim in range(physical_ndim):
            if dim not in self._tile_dims:
                grid_dims.append(dim)
        return tuple(grid_dims)

    def _compute_tile_grid(self) -> Tuple[int, ...]:
        """Compute tile grid dimensions in logical (physical) dimension order.

        For each dim: if tiled, grid = ceil(size/tile), else grid = size.
        """
        physical = self._physical_shape
        tile = self._tile_shape

        grid = []
        for dim in range(len(physical)):
            if self._is_grid_only[dim]:
                grid.append(physical[dim])
            else:
                grid.append(div_ceil(physical[dim], self._tile_sizes[dim]))
        return tuple(grid)

    def get_tile(self) -> nl.NkiTensor:
        """Get current tile and advance iterator."""
        kernel_assert(self._cur_tile[0] < self._tile_grid[0], "HBMStream tile grid exhausted")

        tile = self.get_tile_at_index(tuple(self._cur_tile))
        self._iter_order.advance(self._cur_tile, self._tile_grid)
        return tile

    def get_tile_at_index(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Get tile at specified grid position (in logical dimension order)."""
        physical = self._physical_shape
        physical_ndim = len(physical)

        view = self._hbm_view

        # Process dims highest-to-lowest so select (which removes a dim)
        # never shifts the index of dims we still need to process.
        for d in range(physical_ndim - 1, -1, -1):
            gp = grid_pos[d]
            tile_sz = self._tile_sizes[d]

            if self._is_grid_only[d]:
                # Grid-only dim: select (removes dimension)
                view = view.select(d, gp)
            else:
                # Tiled dim: slice
                start = gp * tile_sz
                end = min(start + tile_sz, physical[d])
                view = view.slice(d, start, end)

        return view

    def get_physical_shape(self) -> Tuple[int, ...]:
        return self._physical_shape

    def get_tile_shape(self) -> Tuple[int, ...]:
        return self._tile_shape

    def get_tile_grid(self) -> Tuple[int, ...]:
        return self._tile_grid

    def get_tile_dims(self) -> Tuple[int, ...]:
        return self._tile_dims

    def get_grid_dims(self) -> Tuple[int, ...]:
        return self._grid_dims

    def get_num_tiles(self) -> int:
        total = 1
        for g in self._tile_grid:
            total = total * g
        return total

    def reset_cur_tile(self) -> None:
        for i in range(len(self._cur_tile)):
            self._cur_tile[i] = 0
