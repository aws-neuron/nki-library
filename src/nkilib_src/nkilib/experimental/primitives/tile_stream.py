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

"""TileStream: tiling and iteration abstractions for (P, *F) logical layout."""

from typing import Optional, Tuple, Union

import nki.language as nl

from ...core.utils.allocator import BufferManager
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from .iter_order import IterOrder, RowMajor
from .view_spec import Broadcast, Permute, ReshapeDim, Select, Slice, ViewSpec

Tensor = Union[nl.NkiTensor, nl.NkiTensor]


def get_logical_shape(tensor: Tensor) -> Tuple[int, ...]:
    """Convert container shape to logical shape.

    Container shape from alloc_logical is (pdim, n_p_tiles, *F).
    Logical shape is (pdim * n_p_tiles, *F).
    """
    view = tensor
    container_shape = tuple(view.shape)
    if len(container_shape) >= 2:
        return (container_shape[0] * container_shape[1],) + container_shape[2:]
    return container_shape


class TileStream(nl.NKIObject):
    """
    Tiled view of a tensor with iteration support.

    Created by calling tile() on a tensor.
    Provides iteration over tiles in specified order.

    Logical layout: (P, *F) where P is partition dimension and F are free dimensions.
    Container layout: (p, p_tile, *F) where P = p * p_tile.
    """

    def __init__(
        self,
        tensor: Tensor,
        tile_shape: Tuple[int, ...],
        iter_order: IterOrder = None,
        tile_view: ViewSpec = None,
        tile_dims: Tuple[int, ...] = None,
        virtual_grid: Tuple[int, ...] = None,
        logical_p: Optional[int] = None,
    ):
        self._tensor = tensor
        self._tile_shape = tile_shape
        self._iter_order = iter_order if iter_order != None else RowMajor()
        self._name = self._tensor.name + "_tiled"
        self._tile_view = tile_view

        # Virtual grid: outer iteration dims that don't allocate SBUF (data reuse)
        self._virtual_grid = virtual_grid if virtual_grid != None else ()
        self._num_virtual = len(self._virtual_grid)

        # Store logical P size for partial tile handling
        # If not provided, compute from container (may over-estimate for partial last p_tile)
        tensor_shape = self._tensor.shape
        self._logical_p = logical_p if logical_p is not None else tensor_shape[0] * tensor_shape[1]

        # Set tile_dims: if None, default to dim 0 (partition) + trailing F dims
        # Logical ndim: container is (p, p_tile, *F), logical is (P, *F)
        logical_ndim = len(tensor.shape) - 1
        tile_ndim = len(tile_shape)
        if tile_dims == None:
            # tile_shape[0] always maps to dim 0 (partition)
            # remaining tile_shape dims map to rightmost logical dims
            dims = [0]
            trailing_start = logical_ndim - (tile_ndim - 1)
            for i in range(tile_ndim - 1):
                dims.append(trailing_start + i)
            self._tile_dims = tuple(dims)
        else:
            self._tile_dims = tile_dims

        # Validate tile_dims and compute grid_dims (dims not in tile_dims)
        self._grid_dims = self._validate_and_compute_grid_dims()

        # Pre-compute boolean lookup for grid-only dims - O(n) not O(n²)
        # _is_grid_only[i] = True if logical dim i is grid-only, False if tiled
        self._is_grid_only = []
        for _ in range(logical_ndim):
            self._is_grid_only.append(True)  # Initialize all as grid-only
        for td in self._tile_dims:
            self._is_grid_only[td] = False  # Mark tiled dims as not grid-only

        # Pre-compute tile size for each logical dim
        # Grid-only dims have tile_size=1 (select single index)
        self._tile_sizes = []
        for _ in range(logical_ndim):
            self._tile_sizes.append(1)
        for i in range(len(self._tile_dims)):
            self._tile_sizes[self._tile_dims[i]] = self._tile_shape[i]

        # Compute effective tile shape after tile_view transforms
        effective_tile_shape, n_p_tiles = self._compute_effective_tile_shape()
        self._effective_tile_shape = effective_tile_shape
        self._n_p_tiles = n_p_tiles

        # Compute tile grid: virtual_grid + (grid_only_dims + tiled_dims)
        self._tile_grid = self._compute_tile_grid()

        # Compute iteration grid (may differ from tile_grid when iter_order has splits)
        self._iter_grid = self._iter_order.get_iter_grid(self._tile_grid)

        # Current tile position for iteration (sized to iter grid)
        self._cur_tile = []
        for _ in range(len(self._iter_grid)):
            self._cur_tile.append(0)

    def _normalize_dim(self, dim: int, ndim: int) -> int:
        """Normalize negative dim to positive index."""
        if dim < 0:
            return ndim + dim
        return dim

    def _get_logical_shape(self) -> Tuple[int, ...]:
        """Get logical shape (P, *F).

        Uses stored logical_p for accurate partial tile handling.
        """
        tensor_shape = self._tensor.shape
        # F dims start at index 2 in container
        return (self._logical_p,) + tensor_shape[2:]

    def _validate_and_compute_grid_dims(self) -> Tuple[int, ...]:
        """Validate tile_dims and compute which logical dims are grid-only."""
        logical_ndim = len(self._tensor.shape) - 1  # Container (p, p_tile, *F) -> logical (P, *F)

        kernel_assert(
            len(self._tile_dims) == len(self._tile_shape),
            f"tile_dims length ({len(self._tile_dims)}) must match tile_shape length ({len(self._tile_shape)})",
        )

        for i in range(len(self._tile_dims)):
            td = self._tile_dims[i]
            kernel_assert(
                td >= 0 and td < logical_ndim,
                f"tile_dims[{i}]={td} out of range for logical_shape with {logical_ndim} dims",
            )

        # Find dims not in tile_dims
        grid_dims = []
        for dim in range(logical_ndim):
            if dim not in self._tile_dims:
                grid_dims.append(dim)
        return tuple(grid_dims)

    def _compute_effective_tile_shape(self) -> Tuple[int, ...]:
        """Compute tile shape after tile_view transformations.

        Returns the shape that get_tile() will actually produce after applying
        tile_view operations to the physical tile.

        With (P, *F) layout, logical == physical (no reordering needed).
        """
        # Start with physical tile shape, P clamped to pdim_size
        ndim = len(self._tile_shape)
        pdim_size = self._tensor.shape[0]

        p_tile = self._tile_shape[0]
        n_p_tiles = div_ceil(p_tile, pdim_size)
        shape = [min(p_tile, pdim_size)]
        if n_p_tiles > 1:
            shape.append(n_p_tiles)
        # Add remaining free dimensions
        for i in range(1, ndim):
            shape.append(self._tile_shape[i])

        if self._tile_view == None:
            return tuple(shape), n_p_tiles

        # Apply transforms to physical shape - with (P, *F) layout, logical == physical
        cur_ndim = len(shape)
        for op in self._tile_view.get_ops():
            if isinstance(op, Broadcast):
                dim = self._normalize_dim(op.dim, cur_ndim)
                shape[dim] = shape[dim] * op.size
            elif isinstance(op, Slice):
                dim = self._normalize_dim(op.dim, cur_ndim)
                shape[dim] = op.end - op.start
            elif isinstance(op, Select):
                dim = self._normalize_dim(op.dim, cur_ndim)
                shape = shape[:dim] + shape[dim + 1 :]
                cur_ndim = len(shape)
            elif isinstance(op, ReshapeDim):
                dim = self._normalize_dim(op.dim, cur_ndim)
                shape = shape[:dim] + list(op.shape) + shape[dim + 1 :]
                cur_ndim = len(shape)
            elif isinstance(op, Permute):
                phys_perm = []
                for d in op.dims:
                    phys_perm.append(self._normalize_dim(d, cur_ndim))
                new_shape = []
                for d in phys_perm:
                    new_shape.append(shape[d])
                shape = new_shape

        return tuple(shape), n_p_tiles

    def _compute_tile_grid(self) -> Tuple[int, ...]:
        """Compute number of tiles in each dimension, in logical dimension order.

        Grid structure: (virtual_grid..., logical_dim_0, logical_dim_1, ...)
        For each logical dim:
        - If in tile_dims: ceil(logical_size / tile_size)
        - If not in tile_dims (grid-only): full logical size

        This keeps grid aligned with logical dimensions for intuitive iteration.
        """
        logical = self._get_logical_shape()

        grid = []
        for dim in range(len(logical)):
            if self._is_grid_only[dim]:
                grid.append(logical[dim])
            else:
                grid.append(div_ceil(logical[dim], self._tile_sizes[dim]))
        # Prepend virtual grid
        return self._virtual_grid + tuple(grid)

    def get_name(self) -> str:
        """Get the name."""
        return self._name

    def get_logical_shape(self) -> Tuple[int, ...]:
        """Get the logical shape (P, *F)."""
        return self._get_logical_shape()

    def get_tile_shape(self) -> Tuple[int, ...]:
        """Get the effective tile shape (after tile_view transforms)."""
        return self._effective_tile_shape

    def get_pdim_size(self) -> int:
        """Get the partition dimension size (p from container)."""
        return self._tensor.shape[0]

    def get_n_p_tiles(self) -> int:
        """Get the number of p_tiles this tile spans."""
        return self._n_p_tiles

    def get_base_tile_shape(self) -> Tuple[int, ...]:
        """Get the base tile shape (logical, before tile_view transforms)."""
        return self._tile_shape

    def get_tile_grid(self) -> Tuple[int, ...]:
        """Get the tile grid dimensions."""
        return self._tile_grid

    def get_iter_order(self):
        """Get the iteration order."""
        return self._iter_order

    def get_tile_view(self) -> ViewSpec:
        """Get the tile view spec, or None if no tile transformations."""
        return self._tile_view

    def get_tile_dims(self) -> Tuple[int, ...]:
        """Get the tile_dims, or None if tiling all dims."""
        return self._tile_dims

    def get_grid_dims(self) -> Tuple[int, ...]:
        """Get the grid-only dims (not in tile_dims)."""
        return self._grid_dims

    def get_virtual_grid(self) -> Tuple[int, ...]:
        """Get the virtual grid dims (for iteration count matching)."""
        return self._virtual_grid

    def get_num_virtual(self) -> int:
        """Get number of virtual grid dims."""
        return self._num_virtual

    def get_dtype(self):
        """Get the data type."""
        return self._tensor.dtype

    def get_num_tiles(self) -> int:
        """Get total number of tiles."""
        total = 1
        for g in self._tile_grid:
            total = total * g
        return total

    def get_container(self) -> nl.NkiTensor:
        """Get the underlying container nl.NkiTensor."""
        return self._tensor

    def reset_cur_tile(self) -> None:
        """Reset iteration to the first tile."""
        for i in range(len(self._cur_tile)):
            self._cur_tile[i] = 0
        self._iter_order.reset()

    def get_logical_tile_at_index(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Get tile at grid position with p_tile dim always preserved.

        Returns a view with container structure intact: (pdim, n_p_tiles, *F_sliced).
        This view can be passed to another tile_stream.tile() for inner tiling (nesting).
        No tile_view is applied.

        Args:
            grid_pos: Tile indices in logical grid coordinates.

        Returns:
            nl.NkiTensor with p_tile dim preserved (no tile_view).
        """
        if self._num_virtual > 0:
            grid_pos = grid_pos[self._num_virtual :]
        return self._get_tile_impl(grid_pos)

    def ltile_at(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Short alias for get_logical_tile_at_index."""
        return self.get_logical_tile_at_index(grid_pos)

    def tile_at(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Short alias for get_tile_at_index."""
        return self.get_tile_at_index(grid_pos)

    def get_tile_at_index(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Get tile at grid position (p_tile dim collapsed, tile_view applied).

        Existing behavior for primitive consumption (Matmul, Load, etc).

        Args:
            grid_pos: Tile indices in logical grid coordinates.

        Returns:
            nl.NkiTensor for this tile (with tile_view transforms applied).
        """
        view = self.get_logical_tile_at_index(grid_pos)
        if self._n_p_tiles == 1:
            view = view.select(1, 0)
        return self._apply_tile_view(view)

    def _get_tile_impl(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Core tile slicing — always preserves p_tile dim via slice.

        Returns (pdim, n_p_tiles_needed, *F_sliced) with batch_dim_offset always 2.
        """
        view = self._tensor
        logical = self._get_logical_shape()
        ndim = len(logical)

        p_grid_idx = grid_pos[0]
        p_tile_size = self._tile_sizes[0]
        pdim_size = self._tensor.shape[0]

        p_start_row = p_grid_idx * p_tile_size
        container_p_idx = p_start_row // pdim_size
        logical_p = self._logical_p
        actual_p_tile_size = min(p_tile_size, logical_p - p_start_row)

        n_p_tiles_needed = div_ceil(p_tile_size, pdim_size)
        view = view.slice(1, container_p_idx, container_p_idx + n_p_tiles_needed)

        if n_p_tiles_needed == 1:
            p_offset = p_start_row % pdim_size
            p_end = min(p_offset + actual_p_tile_size, pdim_size)
            view = view.slice(0, p_offset, p_end)

        # F dimensions: container offset is always 2 (pdim, p_tile, *F)
        for i in range(ndim - 1, 0, -1):  # Skip dim 0 (partition), process dims 1..ndim-1
            grid_idx = grid_pos[i]
            dim_tile_size = self._tile_sizes[i]
            container_dim = i - 1 + 2  # Map logical dim i to container dim
            if self._is_grid_only[i]:
                view = view.select(container_dim, grid_idx)
            else:
                dim_start = grid_idx * dim_tile_size
                dim_end = min(dim_start + dim_tile_size, logical[i])
                view = view.slice(container_dim, dim_start, dim_end)

        return view

    def _apply_tile_view(self, tile: nl.NkiTensor) -> nl.NkiTensor:
        """Apply tile_view transformations to a physical tile.

        With (P, *F) layout, logical == physical, so dim conversion is just normalization.

        Args:
            tile: The physical tile as nl.NkiTensor in (P, *F) order

        Returns:
            Transformed tile with tile_view operations applied
        """
        if self._tile_view == None:
            return tile

        ndim = len(tile.shape)
        result = tile
        for op in self._tile_view.get_ops():
            if isinstance(op, Broadcast):
                phys_dim = self._normalize_dim(op.dim, ndim)
                dim_size = result.shape[phys_dim]
                if dim_size == 1:
                    result = result.broadcast(phys_dim, op.size)
                else:
                    result = result.reshape_dim(phys_dim, (dim_size, 1))
                    ndim = len(result.shape)
                    result = result.broadcast(phys_dim + 1, op.size)
            elif isinstance(op, Slice):
                phys_dim = self._normalize_dim(op.dim, ndim)
                result = result.slice(phys_dim, op.start, op.end)
            elif isinstance(op, Select):
                phys_dim = self._normalize_dim(op.dim, ndim)
                result = result.select(phys_dim, op.index)
                ndim = len(result.shape)
            elif isinstance(op, ReshapeDim):
                phys_dim = self._normalize_dim(op.dim, ndim)
                result = result.reshape_dim(phys_dim, op.shape)
                ndim = len(result.shape)
            elif isinstance(op, Permute):
                phys_perm = []
                for d in op.dims:
                    phys_perm.append(self._normalize_dim(d, ndim))
                result = result.permute(tuple(phys_perm))

        return result

    def get_tile(self) -> nl.NkiTensor:
        """
        Get current tile and advance. Collapses p_tile dim, applies tile_view.

        Returns:
            nl.NkiTensor for the current tile (with tile_view transforms applied),
            or None if the current position is filtered by a split dim.
        """
        # Check if exhausted
        kernel_assert(
            self._cur_tile[0] < self._iter_grid[0],
            f"TileStream '{self._name}' tile grid exhausted, cur_tile={self._cur_tile}, iter_grid={self._iter_grid}",
        )

        # Convert iter pos to real grid pos, skip filtered positions
        grid_pos = self._iter_order.to_grid_pos(tuple(self._cur_tile), self._tile_grid)
        while grid_pos is None:
            self._iter_order.advance(self._cur_tile, self._iter_grid)
            grid_pos = self._iter_order.to_grid_pos(tuple(self._cur_tile), self._tile_grid)

        tile = self.get_tile_at_index(grid_pos)

        # Advance to next tile using iter_order
        self._iter_order.advance(self._cur_tile, self._iter_grid)

        return tile

    def get_logical_tile(self) -> nl.NkiTensor:
        """Get current tile with p_tile dim preserved, then advance.

        Returns (pdim, n_p_tiles, *F_sliced) with no tile_view applied.
        Use for nesting with inner tile_stream.tile().

        Returns:
            nl.NkiTensor with p_tile dim preserved.
        """
        kernel_assert(
            self._cur_tile[0] < self._iter_grid[0],
            f"TileStream '{self._name}' tile grid exhausted, cur_tile={self._cur_tile}, iter_grid={self._iter_grid}",
        )

        grid_pos = self._iter_order.to_grid_pos(tuple(self._cur_tile), self._tile_grid)
        while grid_pos is None:
            self._iter_order.advance(self._cur_tile, self._iter_grid)
            grid_pos = self._iter_order.to_grid_pos(tuple(self._cur_tile), self._tile_grid)

        tile = self.get_logical_tile_at_index(grid_pos)
        self._iter_order.advance(self._cur_tile, self._iter_grid)
        return tile


def alloc_logical(
    logical_shape: Tuple[int, ...],
    pdim_size: int,
    dtype,
    name: Optional[str] = None,
    sbm: Optional[BufferManager] = None,
    collapse_trivial_p_tile: bool = False,
    buffer=nl.sbuf,
    align: Optional[int] = None,
) -> nl.NkiTensor:
    """Helper to allocate a logical shape with pdim tiled.

    Logical shape is (P, *F) where P is partition dimension.
    Container shape is always (pdim_size, n_p_tiles, *F) by default.

    When collapse_trivial_ptile=True and n_p_tiles == 1, container shape is collapsed to (P, *F).

    The correspondence is: logical P = p * p_tile (container dims 0 and 1).

    Args:
        buffer: Memory buffer type (nl.sbuf or nl.psum). Default: nl.sbuf.
        align: Alignment requirement in bytes (e.g., 32 for DMA transpose). Default: None.
    """
    kernel_assert(
        len(logical_shape) >= 2,
        f"logical tensor allocation requires logical_shape with at least 2 dimensions, got {logical_shape}",
    )

    # Partition dimension gets divided by pdim_size
    n_p_tiles = div_ceil(logical_shape[0], pdim_size)

    # Always include p_tile dim unless explicitly collapsed
    if collapse_trivial_p_tile and n_p_tiles == 1:
        container_shape = logical_shape
    else:
        # Shape: (pdim_size, n_p_tiles, *F) - always has p_tile dim
        container_shape = (pdim_size, n_p_tiles) + logical_shape[1:]

    if sbm:
        container = sbm.alloc_stack(container_shape, dtype=dtype, buffer=buffer, name=name, align=align)
    else:
        container = nl.ndarray(container_shape, dtype=dtype, buffer=buffer, name=name)

    return container


class HBMStream(nl.NKIObject):
    """Tile iteration over HBM tensor. All dimensions are free (no partition dimension).

    Unlike TileStream which assumes SBUF container format (pdim, n_p_tiles, *F),
    HBMStream treats the tensor shape directly as the logical shape.
    Default tile_dims maps to trailing rightmost dimensions.

    Usage:
        hbm_ts = tile_hbm(hidden_view, (1, 1, 1, n_H_grp), iter_order=RowMajor())
        dma.Load(dst=sbuf_ts, src=hbm_ts).execute()
    """

    def __init__(
        self,
        tensor: Tensor,
        tile_shape: Tuple[int, ...],
        iter_order: IterOrder = None,
        tile_dims: Tuple[int, ...] = None,
    ) -> None:
        """
        Args:
            tensor: HBM tensor or nl.NkiTensor
            tile_shape: Shape of each tile. Length determines how many dims are tiled.
            iter_order: Iteration order (RowMajor, ColMajor, DimOrder). Defaults to RowMajor.
            tile_dims: Which dims to tile. If None, defaults to trailing rightmost dims.
        """
        if isinstance(tensor, nl.NkiTensor):
            self._tensor = tensor
        else:
            self._tensor = tensor

        self._tile_shape = tile_shape
        self._iter_order = iter_order if iter_order is not None else RowMajor()
        self._name = self._tensor.name + "_hbm_tiled"

        shape = tuple(self._tensor.shape)
        ndim = len(shape)
        tile_ndim = len(tile_shape)

        # Default tile_dims: trailing rightmost dims
        if tile_dims is None:
            trailing_start = ndim - tile_ndim
            dims = []
            for i in range(tile_ndim):
                dims.append(trailing_start + i)
            self._tile_dims = tuple(dims)
        else:
            self._tile_dims = tile_dims

        # Validate
        kernel_assert(
            len(self._tile_dims) == tile_ndim,
            f"HBMStream '{self._name}': tile_dims length ({len(self._tile_dims)}) "
            f"must match tile_shape length ({tile_ndim})",
        )
        for i in range(len(self._tile_dims)):
            td = self._tile_dims[i]
            kernel_assert(
                td >= 0 and td < ndim,
                f"HBMStream '{self._name}': tile_dims[{i}]={td} out of range for {ndim} dims",
            )

        # Pre-compute is_grid_only and tile_sizes
        self._is_grid_only = []
        for d in range(ndim):
            self._is_grid_only.append(d not in self._tile_dims)

        self._tile_sizes = []
        for d in range(ndim):
            self._tile_sizes.append(1)
        for i in range(len(self._tile_dims)):
            self._tile_sizes[self._tile_dims[i]] = self._tile_shape[i]

        # Compute tile grid
        self._tile_grid = self._compute_tile_grid()

        # Compute iteration grid (may differ from tile_grid when iter_order has splits)
        self._iter_grid = self._iter_order.get_iter_grid(self._tile_grid)

        # Current tile position for iteration (sized to iter grid)
        self._cur_tile = []
        for _ in range(len(self._iter_grid)):
            self._cur_tile.append(0)

    def _compute_tile_grid(self) -> Tuple[int, ...]:
        shape = self._tensor.shape
        grid = []
        for dim in range(len(shape)):
            if self._is_grid_only[dim]:
                grid.append(shape[dim])
            else:
                grid.append(div_ceil(shape[dim], self._tile_sizes[dim]))
        return tuple(grid)

    def get_tile(self) -> nl.NkiTensor:
        kernel_assert(
            self._cur_tile[0] < self._iter_grid[0],
            f"HBMStream '{self._name}' tile grid exhausted",
        )

        grid_pos = self._iter_order.to_grid_pos(tuple(self._cur_tile), self._tile_grid)
        while grid_pos is None:
            self._iter_order.advance(self._cur_tile, self._iter_grid)
            grid_pos = self._iter_order.to_grid_pos(tuple(self._cur_tile), self._tile_grid)

        tile = self.get_tile_at_index(grid_pos)
        self._iter_order.advance(self._cur_tile, self._iter_grid)
        return tile

    def get_tile_at_index(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        shape = self._tensor.shape
        ndim = len(shape)
        view = self._tensor

        for d in range(ndim - 1, -1, -1):
            gp = grid_pos[d]
            tile_sz = self._tile_sizes[d]

            if self._is_grid_only[d]:
                view = view.select(d, gp)
            else:
                start = gp * tile_sz
                end = min(start + tile_sz, shape[d])
                view = view.slice(d, start, end)

        return view

    def tile_at(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Get tile at grid position."""
        return self.get_tile_at_index(grid_pos)

    def ltile_at(self, grid_pos: Tuple[int, ...]) -> nl.NkiTensor:
        """Alias for tile_at (HBM has no p_tile distinction)."""
        return self.get_tile_at_index(grid_pos)

    def get_name(self) -> str:
        return self._name

    def get_tile_shape(self) -> Tuple[int, ...]:
        return self._tile_shape

    def get_tile_grid(self) -> Tuple[int, ...]:
        return self._tile_grid

    def get_tile_dims(self) -> Tuple[int, ...]:
        return self._tile_dims

    def get_num_tiles(self) -> int:
        total = 1
        for g in self._tile_grid:
            total = total * g
        return total

    def get_iter_order(self) -> IterOrder:
        return self._iter_order

    def reset_cur_tile(self) -> None:
        for i in range(len(self._cur_tile)):
            self._cur_tile[i] = 0
        self._iter_order.reset()

    def get_container(self) -> nl.NkiTensor:
        return self._tensor

    def get_dtype(self):
        return self._tensor.dtype


def tile_hbm(
    tensor: Optional[Tensor],
    tile_shape: Tuple[int, ...],
    tile_dims: Tuple[int, ...] = None,
    iter_order: IterOrder = None,
) -> Optional[HBMStream]:
    """Create an HBMStream for tiling a tensor without partition dimension.

    All dimensions are free. Default tile_dims = trailing rightmost dims.
    """
    if tensor is None:
        return None

    if iter_order is None:
        iter_order = RowMajor()

    return HBMStream(
        tensor=tensor,
        tile_shape=tile_shape,
        iter_order=iter_order,
        tile_dims=tile_dims,
    )


def tile(
    tensor: Optional[Tensor],
    tile_shape: Tuple[int, ...],
    tile_dims: Tuple[int, ...] = None,
    tile_view: ViewSpec = None,
    virtual_grid: Tuple[int, ...] = None,
    iter_order: IterOrder = None,
    has_p_tile_dim: bool = True,
    logical_p: Optional[int] = None,
) -> Optional[Union[TileStream, HBMStream]]:
    """
    Create a TileStream (SBUF) or HBMStream (HBM) with the specified tiling.

    Detects the tensor's buffer location automatically:
    - SBUF tensors → TileStream (partition dim handling, alloc_logical layout)
    - HBM tensors → HBMStream (no partition dim, all dims are free)

    Args:
        tensor: Tensor to be tiled, or None (returns None)
        tile_shape: Shape of each tile.
            SBUF: (p_tile, *f_tiles) — first dim is partition tile size.
            HBM: (*tile_sizes) — all dims are free.
        tile_view: Optional ViewSpec for transformations on extracted tiles
            (e.g., broadcast, reshape). SBUF only.
        tile_dims: Which dimensions form the tile. If None, defaults to all dims
            (SBUF) or trailing rightmost dims (HBM).
        virtual_grid: Additional outer grid dimensions for iteration that don't
            allocate extra SBUF. SBUF only.
        iter_order: Iteration order (RowMajor, ColMajor, DimOrder). Defaults to RowMajor().
        has_p_tile_dim: If False, tensor lacks p_tile dim (expands it). SBUF only.
        logical_p: Actual logical P size (for partial tile handling). If None,
            computed from container shape. SBUF only.

    Returns:
        TileStream (SBUF) or HBMStream (HBM) with tiling and iteration support.

    Example:
        # SBUF
        buf = tile_stream.alloc_logical((P, F), pdim_size, dtype, "buf")
        ts = tile_stream.tile(buf, tile_shape=(p_tile, f_tile))

        # HBM
        hbm_view = hbm_tensor.reshape_dim(...)
        ts = tile_stream.tile(hbm_view, tile_shape=(1, 1, H, 128), tile_dims=(0, 1))
    """
    # Allow None passthrough for conditional tiling
    if tensor == None:
        return None

    tensor = tensor

    # HBM path: no partition dimension
    if tensor.is_hbm():
        if iter_order is None:
            iter_order = RowMajor()
        return HBMStream(
            tensor=tensor,
            tile_shape=tile_shape,
            iter_order=iter_order,
            tile_dims=tile_dims,
        )

    # SBUF path: partition dimension handling
    if iter_order == None:
        iter_order = RowMajor()

    # Add a dimension of size 1 for p_tile_dim when tensor doesn't have it
    if not has_p_tile_dim:
        tensor = tensor.expand_dim(dim=1)

    p_tile = tile_shape[0]
    pdim_size = tensor.shape[0]
    if p_tile > pdim_size:
        kernel_assert(
            p_tile % pdim_size == 0,
            f"Partition tile size ({p_tile}) must be a multiple of pdim_size ({pdim_size}) when greater",
        )

    return TileStream(
        tensor=tensor,
        tile_shape=tile_shape,
        iter_order=iter_order,
        tile_view=tile_view,
        tile_dims=tile_dims,
        virtual_grid=virtual_grid,
        logical_p=logical_p,
    )
