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


"""TiledTensor: A tensor of tiles over a base tensor.

This module provides a zero-cost abstraction for tiling tensors, where each
element of the TiledTensor represents a tile in the base tensor. TiledTensor
is fully interoperable with NkiTensor.
"""

from typing import Tuple

import nki.language as nl


def _trivial_strides(shape, base_stride: int = 1):
    """Row-major (C-style) strides for a shape; innermost stride = base_stride."""
    strides = [base_stride]
    for i in range(1, len(shape)):
        strides.append(strides[i - 1] * shape[len(shape) - i])
    ret = []
    for i in range(len(shape)):
        ret.append(strides[len(shape) - i - 1])
    return tuple(ret)


from .allocator import sizeinbytes
from .kernel_assert import kernel_assert
from .logging import Logger

logger = Logger("TiledTensor")


class TiledTensor(nl.NKIObject):
    """A tensor of tiles over a base tensor.

    Each element of the TiledTensor represents a rectangular tile in the source
    tensor. TiledTensor tracks tile positions using a strides-based grid model,
    enabling zero-cost grid manipulations (select, slice, reshape_dim, etc.)
    without generating any instructions.

    The source tensor can be either an nl.ndarray or a NkiTensor. When a
    NkiTensor is passed, tiling operates on its logical shape.

    Attributes:
        shape: The tile grid dimensions (number of tiles per dimension).
        tile_size: The tile size for each source dimension.
    """

    # The source tensor wrapped as a NkiTensor for uniform handling
    _source_view: nl.NkiTensor
    # The logical shape of the source tensor (used for tile boundary clipping)
    _source_shape: Tuple[int, ...]
    # The tile size for each source dimension (one entry per source dim)
    _tile_size: Tuple[int, ...]
    # The original grid shape at construction time (one entry per source dim),
    # used to unravel flat tile indices back to per-source-dim coordinates
    _full_grid_shape: Tuple[int, ...]
    # The current grid shape after select/slice/reshape_dim/etc. operations
    _grid_shape: Tuple[int, ...]
    # Strides in flat tile-index space, mirroring NkiTensor's stride model;
    # grid_strides[i] is the flat-index step when advancing one position in grid dim i
    _grid_strides: Tuple[int, ...]
    # Offset into the flat tile-index space (the flat index of the first selected tile)
    _grid_offset: int
    # Maps each current grid dimension to its originating source dimension index;
    # -1 indicates a virtual dim (from expand_dim) or mixed dim (from flatten across source dims)
    _dim_map: Tuple[int, ...]
    # The dtype of the source tensor (may change via reinterpret_cast)
    _dtype: object

    def __init__(self, source, tile_size):
        """Construct a TiledTensor by tiling a source tensor.

        Args:
            source: The base tensor (nl.ndarray) or NkiTensor to tile.
            tile_size: Size of each tile per dimension. Must have the same
                number of dimensions as the source.

        Raises:
            AssertionError: If tile_size has wrong number of dimensions,
                contains zero or negative values.
        """
        self._source_view = source
        self._source_shape = tuple(source.shape)
        self._dtype = source.dtype

        kernel_assert(
            len(tile_size) == len(self._source_shape),
            f"tile_size has {len(tile_size)} dimensions but source has {len(self._source_shape)} dimensions",
        )
        for i in range(len(tile_size)):
            kernel_assert(tile_size[i] > 0, f"tile_size[{i}] must be positive, got {tile_size[i]}")

        self._tile_size = tuple(tile_size)

        grid_shape = []
        for d in range(len(self._source_shape)):
            grid_shape.append((self._source_shape[d] + self._tile_size[d] - 1) // self._tile_size[d])
        self._full_grid_shape = tuple(grid_shape)
        self._grid_shape = tuple(grid_shape)
        self._grid_strides = _trivial_strides(self._grid_shape)
        self._grid_offset = 0

        dim_map = []
        for d in range(len(self._source_shape)):
            dim_map.append(d)
        self._dim_map = tuple(dim_map)

    def get_shape(self):
        """Return the tile grid dimensions."""
        return self._grid_shape

    @property
    def shape(self):
        """Return the tile grid dimensions."""
        return self._grid_shape

    def get_tile_size(self):
        """Return the tile size for each source dimension."""
        return self._tile_size

    @property
    def tile_size(self):
        """Return the tile size for each source dimension."""
        return self._tile_size

    def _get_ndim(self):
        return len(self._grid_shape)

    def _copy_from(self, other):
        """Copy all fields from another TiledTensor."""
        self._source_view = other._source_view
        self._source_shape = other._source_shape
        self._tile_size = other._tile_size
        self._full_grid_shape = other._full_grid_shape
        self._grid_shape = other._grid_shape
        self._grid_strides = other._grid_strides
        self._grid_offset = other._grid_offset
        self._dim_map = other._dim_map
        self._dtype = other._dtype

    def _make_copy(
        self,
        grid_shape=None,
        grid_strides=None,
        grid_offset=None,
        dim_map=None,
        tile_size=None,
        source_shape=None,
        source_view=None,
        dtype=None,
    ):
        """Create a new TiledTensor with optionally modified fields.

        Uses a dummy source to construct, then overwrites all fields.
        """
        # Create via constructor with a minimal valid call, then overwrite
        # We use the source_view's base_tensor and current tile_size to construct
        ts = self._tile_size if tile_size is None else tile_size
        new = TiledTensor(self._source_view, ts)

        new._source_view = source_view if source_view is not None else self._source_view
        new._source_shape = source_shape if source_shape is not None else self._source_shape
        new._tile_size = tile_size if tile_size is not None else self._tile_size
        new._full_grid_shape = self._full_grid_shape
        new._grid_shape = tuple(grid_shape) if grid_shape is not None else self._grid_shape
        new._grid_strides = tuple(grid_strides) if grid_strides is not None else self._grid_strides
        new._grid_offset = grid_offset if grid_offset is not None else self._grid_offset
        new._dim_map = tuple(dim_map) if dim_map is not None else self._dim_map
        new._dtype = dtype if dtype is not None else self._dtype

        kernel_assert(
            len(new._grid_shape) == len(new._grid_strides),
            f"Grid shape length {len(new._grid_shape)} != strides length {len(new._grid_strides)}",
        )
        kernel_assert(
            len(new._grid_shape) == len(new._dim_map),
            f"Grid shape length {len(new._grid_shape)} != dim_map length {len(new._dim_map)}",
        )
        kernel_assert(new._grid_offset >= 0, "Grid offset must be non-negative")
        return new

    def _unravel(self, flat_idx, shape):
        """Convert a flat index to coordinates in the given shape (row-major)."""
        coords = []
        for d in range(len(shape)):
            stride = 1
            for d2 in range(d + 1, len(shape)):
                stride *= shape[d2]
            coords.append(flat_idx // stride)
            flat_idx = flat_idx % stride
        return tuple(coords)

    def _get_single_tile_view(self, flat_tile_idx, keep_dim=True):
        """Get the NkiTensor for a single tile given its flat index in the original grid.

        Args:
            flat_tile_idx: Flat index into the full grid.
            keep_dim: If True (default), all dims are kept (sliced to size 1).
                      If False, size-1 tile dims are removed via select.
        """
        coords = self._unravel(flat_tile_idx, self._full_grid_shape)
        view = self._source_view
        if keep_dim:
            for d in range(len(self._source_shape)):
                start = coords[d] * self._tile_size[d]
                end = min(start + self._tile_size[d], self._source_shape[d])
                view = view.slice(d, start, end)
        else:
            squeezed = 0
            for d in range(len(self._source_shape)):
                start = coords[d] * self._tile_size[d]
                end = min(start + self._tile_size[d], self._source_shape[d])
                if self._tile_size[d] == 1 and d - squeezed > 0:
                    view = view.select(d - squeezed, start)
                    squeezed = squeezed + 1
                else:
                    view = view.slice(d - squeezed, start, end)
        return view

    def select(self, dim, index):
        """Select one row/column of tiles along a dimension, reducing dimensionality.

        Args:
            dim: Grid dimension to select from.
            index: Index of the row/column to select.

        Returns:
            A new TiledTensor with one fewer grid dimension.
        """
        kernel_assert(dim >= 0, f"Dimension must be non-negative, got {dim}")
        kernel_assert(dim < self._get_ndim(), f"Dimension {dim} out of range for {self._get_ndim()}D grid")
        kernel_assert(index >= 0, f"Index must be non-negative, got {index}")
        kernel_assert(
            index < self._grid_shape[dim],
            f"Index {index} out of range for grid dimension {dim} with size {self._grid_shape[dim]}",
        )

        new_offset = self._grid_offset + index * self._grid_strides[dim]
        new_shape = self._grid_shape[:dim] + self._grid_shape[dim + 1 :]
        new_strides = self._grid_strides[:dim] + self._grid_strides[dim + 1 :]
        new_dim_map = self._dim_map[:dim] + self._dim_map[dim + 1 :]

        return self._make_copy(
            grid_shape=new_shape, grid_strides=new_strides, grid_offset=new_offset, dim_map=new_dim_map
        )

    def slice(self, dim, start, end):
        """Slice the tile grid along a dimension.

        Args:
            dim: Grid dimension to slice.
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            A new TiledTensor with a sliced grid.
        """
        kernel_assert(dim >= 0, f"Dimension must be non-negative, got {dim}")
        kernel_assert(dim < self._get_ndim(), f"Dimension {dim} out of range for {self._get_ndim()}D grid")
        kernel_assert(start >= 0, f"Start index must be non-negative, got {start}")
        kernel_assert(end > start, f"End index {end} must be greater than start index {start}")
        kernel_assert(
            end <= self._grid_shape[dim],
            f"End index {end} out of range for grid dimension {dim} with size {self._grid_shape[dim]}",
        )

        new_offset = self._grid_offset + start * self._grid_strides[dim]
        new_shape = list(self._grid_shape)
        new_shape[dim] = end - start
        return self._make_copy(grid_shape=tuple(new_shape), grid_offset=new_offset)

    def get_tile(self, indices, keep_dim=True):
        """Access a single tile by its grid coordinates.

        Args:
            indices: Tuple of indices, one per grid dimension.
            keep_dim: If False (default), size-1 tile dims are removed.
                      If True, all dims are kept.

        Returns:
            NkiTensor for the specified tile.
        """
        kernel_assert(
            len(indices) == self._get_ndim(),
            f"Expected {self._get_ndim()} indices, got {len(indices)}",
        )

        flat_idx = self._grid_offset
        for i in range(len(indices)):
            kernel_assert(indices[i] >= 0, f"Index must be non-negative, got {indices[i]} at dimension {i}")
            kernel_assert(
                indices[i] < self._grid_shape[i],
                f"Index {indices[i]} out of range for grid dimension {i} with size {self._grid_shape[i]}",
            )
            flat_idx = flat_idx + indices[i] * self._grid_strides[i]
        return self._get_single_tile_view(flat_idx, keep_dim=keep_dim)

    def reshape_dim(self, dim, shape):
        """Split a grid dimension into multiple dimensions.

        Args:
            dim: Grid dimension to reshape.
            shape: New sizes for the split dimensions. Product must equal
                the current size of the dimension.

        Returns:
            A new TiledTensor with the grid dimension split.
        """
        kernel_assert(dim >= 0, f"Dimension must be non-negative, got {dim}")
        kernel_assert(dim < self._get_ndim(), f"Dimension {dim} out of range for {self._get_ndim()}D grid")

        size_prod = 1
        for s in shape:
            size_prod = size_prod * s
        kernel_assert(
            self._grid_shape[dim] == size_prod,
            f"Product of new shape {shape} is {size_prod}, but grid dimension {dim} has size {self._grid_shape[dim]}",
        )

        new_grid_shape = self._grid_shape[:dim] + tuple(shape) + self._grid_shape[dim + 1 :]
        reshaped_strides = _trivial_strides(shape, base_stride=self._grid_strides[dim])
        new_grid_strides = self._grid_strides[:dim] + reshaped_strides + self._grid_strides[dim + 1 :]
        src_dim = self._dim_map[dim]
        new_dim_map_list = []
        for i in range(len(self._dim_map[:dim])):
            new_dim_map_list.append(self._dim_map[i])
        for i in range(len(shape)):
            new_dim_map_list.append(src_dim)
        for i in range(dim + 1, len(self._dim_map)):
            new_dim_map_list.append(self._dim_map[i])
        new_dim_map = tuple(new_dim_map_list)

        return self._make_copy(grid_shape=new_grid_shape, grid_strides=new_grid_strides, dim_map=new_dim_map)

    def flatten_dims(self, start_dim, end_dim):
        """Flatten a range of grid dimensions into a single dimension.

        Args:
            start_dim: First dimension to flatten (inclusive).
            end_dim: Last dimension to flatten (inclusive).

        Returns:
            A new TiledTensor with the specified dimensions flattened.
        """
        kernel_assert(start_dim >= 0, f"Start dimension must be non-negative, got {start_dim}")
        kernel_assert(
            start_dim < self._get_ndim(),
            f"Start dimension {start_dim} out of range for {self._get_ndim()}D grid",
        )
        kernel_assert(
            end_dim < self._get_ndim(),
            f"End dimension {end_dim} out of range for {self._get_ndim()}D grid",
        )
        kernel_assert(start_dim < end_dim, f"Start dimension {start_dim} must be less than end dimension {end_dim}")

        for i in range(start_dim, end_dim):
            kernel_assert(
                self._grid_strides[i] == self._grid_shape[i + 1] * self._grid_strides[i + 1],
                f"Grid dimensions {i} and {i + 1} are not contiguous "
                f"(stride[{i}]={self._grid_strides[i]} != shape[{i + 1}]*stride[{i + 1}]="
                f"{self._grid_shape[i + 1] * self._grid_strides[i + 1]})",
            )

        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size = flattened_size * self._grid_shape[i]

        new_grid_shape = self._grid_shape[:start_dim] + (flattened_size,) + self._grid_shape[end_dim + 1 :]
        new_grid_strides = (
            self._grid_strides[:start_dim] + (self._grid_strides[end_dim],) + self._grid_strides[end_dim + 1 :]
        )

        # Determine dim_map: check if all flattened dims map to same source dim
        flat_dim_val = self._dim_map[start_dim]
        all_same = True
        for i in range(start_dim + 1, end_dim + 1):
            if self._dim_map[i] != flat_dim_val:
                all_same = False
                break
        if not all_same:
            flat_dim_val = -1

        new_dim_map = self._dim_map[:start_dim] + (flat_dim_val,) + self._dim_map[end_dim + 1 :]

        return self._make_copy(grid_shape=new_grid_shape, grid_strides=new_grid_strides, dim_map=new_dim_map)

    def expand_dim(self, dim):
        """Insert a size-1 grid dimension at the specified position.

        Args:
            dim: Position to insert the new dimension.

        Returns:
            A new TiledTensor with an additional size-1 grid dimension.
        """
        kernel_assert(
            dim >= 0 and dim <= self._get_ndim(),
            f"Dimension {dim} out of range for {self._get_ndim()}D grid",
        )

        if dim == self._get_ndim():
            new_stride = 1
        else:
            new_stride = self._grid_strides[dim] * self._grid_shape[dim]

        new_grid_shape = self._grid_shape[:dim] + (1,) + self._grid_shape[dim:]
        new_grid_strides = self._grid_strides[:dim] + (new_stride,) + self._grid_strides[dim:]
        new_dim_map = self._dim_map[:dim] + (-1,) + self._dim_map[dim:]

        return self._make_copy(grid_shape=new_grid_shape, grid_strides=new_grid_strides, dim_map=new_dim_map)

    def squeeze_dim(self, dim):
        """Remove a size-1 grid dimension.

        Args:
            dim: Dimension to remove (must have size 1).

        Returns:
            A new TiledTensor with the dimension removed.
        """
        kernel_assert(dim >= 0, f"Dimension must be non-negative, got {dim}")
        kernel_assert(dim < self._get_ndim(), f"Dimension {dim} out of range for {self._get_ndim()}D grid")
        kernel_assert(
            self._grid_shape[dim] == 1,
            f"Can only squeeze size-1 dimensions, got size {self._grid_shape[dim]} at dim {dim}",
        )

        new_grid_shape = self._grid_shape[:dim] + self._grid_shape[dim + 1 :]
        new_grid_strides = self._grid_strides[:dim] + self._grid_strides[dim + 1 :]
        new_dim_map = self._dim_map[:dim] + self._dim_map[dim + 1 :]

        return self._make_copy(grid_shape=new_grid_shape, grid_strides=new_grid_strides, dim_map=new_dim_map)

    def reshape(self, new_shape):
        """Reshape the tile grid to new dimensions.

        The total number of tiles must remain the same. After reshape, the
        dim_map is set to -1 for all dimensions (source dim mapping is lost).

        Args:
            new_shape: New grid dimensions. Total product must match current.

        Returns:
            A new TiledTensor with reshaped grid.
        """
        old_total = 1
        for s in self._grid_shape:
            old_total = old_total * s
        new_total = 1
        for s in new_shape:
            new_total = new_total * s
        kernel_assert(
            old_total == new_total,
            f"Cannot reshape grid from {self._grid_shape} (total {old_total}) to {new_shape} (total {new_total})",
        )

        expected_strides = _trivial_strides(self._grid_shape)
        kernel_assert(
            self._grid_strides == expected_strides,
            f"Cannot reshape non-contiguous grid. Grid strides {self._grid_strides} are not trivial {expected_strides}",
        )

        new_strides = _trivial_strides(new_shape)
        new_dim_map_list = []
        for _ in range(len(new_shape)):
            new_dim_map_list.append(-1)
        new_dim_map = tuple(new_dim_map_list)

        return self._make_copy(grid_shape=new_shape, grid_strides=new_strides, dim_map=new_dim_map)

    def reinterpret_cast(self, new_dtype):
        """Reinterpret the underlying tensor as a different dtype.

        Adjusts the last dimension of each tile by the dtype size ratio.
        The grid shape does not change.

        Args:
            new_dtype: Target dtype.

        Returns:
            A new TiledTensor with adjusted tile sizes and dtype.
        """
        old_size = sizeinbytes(self._dtype)
        new_size = sizeinbytes(new_dtype)

        if old_size == new_size:
            return self._make_copy(dtype=new_dtype)

        last_dim = len(self._tile_size) - 1

        if new_size > old_size:
            ratio = new_size // old_size
            kernel_assert(
                self._tile_size[last_dim] % ratio == 0,
                f"Last tile dimension {self._tile_size[last_dim]} not divisible by dtype ratio {ratio}",
            )
            new_tile_last = self._tile_size[last_dim] // ratio
            kernel_assert(
                self._source_shape[last_dim] % ratio == 0,
                f"Last source dimension {self._source_shape[last_dim]} not divisible by dtype ratio {ratio}",
            )
            new_source_last = self._source_shape[last_dim] // ratio
        else:
            ratio = old_size // new_size
            new_tile_last = self._tile_size[last_dim] * ratio
            new_source_last = self._source_shape[last_dim] * ratio

        new_tile_size = self._tile_size[:last_dim] + (new_tile_last,)
        new_source_shape = self._source_shape[:last_dim] + (new_source_last,)

        new_source_view = self._source_view.view(new_dtype)

        return self._make_copy(
            tile_size=new_tile_size,
            source_shape=new_source_shape,
            source_view=new_source_view,
            dtype=new_dtype,
        )

    def get_view(self):
        """Return a NkiTensor covering all currently selected tiles.

        Performs a contiguity check to verify that the selected tiles form a
        single contiguous rectangular block in the source tensor. If not,
        raises an error directing the user to use force_get_view() instead.

        Returns:
            A NkiTensor representing the coalesced tile region.
        """
        return self._build_view(True)

    def force_get_view(self):
        """Return a NkiTensor without performing contiguity checks.

        When tiles are non-contiguous, the returned NkiTensor may have a
        higher rank than the tile_size, with extra dimensions representing
        the tile groups.

        Returns:
            A NkiTensor representing the tile region.
        """
        return self._build_view(False)

    def _build_contiguous_view(self):
        """Build a NkiTensor for contiguous tile selections.

        Used by get_view() after contiguity has been verified.
        Computes tile starts and counts per source dim using min/max unravel,
        independent of dim_map.
        """
        ndim = self._get_ndim()
        source_ndim = len(self._source_shape)
        view = self._source_view

        # Compute min and max flat tile indices
        min_flat = self._grid_offset
        max_flat = self._grid_offset
        for i in range(ndim):
            max_flat = max_flat + (self._grid_shape[i] - 1) * self._grid_strides[i]

        min_coords = self._unravel(min_flat, self._full_grid_shape)
        max_coords = self._unravel(max_flat, self._full_grid_shape)

        for d in range(source_ndim):
            tile_start = min_coords[d]
            tile_count = max_coords[d] - min_coords[d] + 1
            start = tile_start * self._tile_size[d]
            end = min((tile_start + tile_count) * self._tile_size[d], self._source_shape[d])
            view = view.slice(d, start, end)
        return view

    def _build_force_view(self):
        """Build a NkiTensor for potentially non-contiguous tile selections.

        Each grid dimension becomes one or more view dimensions. When a grid
        dimension has a stride larger than what contiguous tiling would produce,
        extra view dimensions are introduced to represent the strided access,
        resulting in a higher-rank NkiTensor.

        For example, selecting every-other tile row from a (4,4) grid where
         each tile is (128, 256) produces
        grid shape (2,4) with strides (8,1). The source dim 0 (512 elements,
        tile size 128) gets split into (2, 128) via reshape_dim + slice with
        step = 2*128 = 256, yielding a view of shape (2, 128, 1024).
        """
        groups = self._build_dim_groups()
        view = self._source_view

        full_coords = self._unravel(self._grid_offset, self._full_grid_shape)

        # For each source dim, figure out what the grid says about it
        source_ndim = len(self._source_shape)

        # Collect per-source-dim info from groups
        src_dim_info = []
        for d in range(source_ndim):
            src_dim_info.append(None)

        for g_idx in range(len(groups)):
            src_dim = groups[g_idx][0]
            gdims = groups[g_idx][1]
            if src_dim >= 0:
                src_dim_info[src_dim] = gdims

        # Process each source dim: slice from source, potentially reshape for strides.
        # view_dim tracks the current view dimension index, which shifts when
        # reshape_dim inserts extra dimensions for non-contiguous source dims.
        view_dim = 0
        for d in range(source_ndim):
            gdims = src_dim_info[d]
            if gdims is None:
                # Source dim was fully selected away (via select) - single tile
                start = full_coords[d] * self._tile_size[d]
                end = min(start + self._tile_size[d], self._source_shape[d])
                view = view.slice(view_dim, start, end)
                view_dim = view_dim + 1
            else:
                total_count = 1
                for gd in gdims:
                    total_count = total_count * self._grid_shape[gd]

                start_tile = full_coords[d]
                start_elem = start_tile * self._tile_size[d]

                # Check if this group's grid strides are contiguous
                is_contiguous = True
                if len(gdims) == 1:
                    gd = gdims[0]
                    expected_flat_stride = 1
                    for d2 in range(d + 1, source_ndim):
                        expected_flat_stride = expected_flat_stride * self._full_grid_shape[d2]
                    if self._grid_strides[gd] != expected_flat_stride:
                        is_contiguous = False
                else:
                    for j in range(len(gdims) - 1):
                        if (
                            self._grid_strides[gdims[j]]
                            != self._grid_shape[gdims[j + 1]] * self._grid_strides[gdims[j + 1]]
                        ):
                            is_contiguous = False
                            break
                    if is_contiguous:
                        expected_flat_stride = 1
                        for d2 in range(d + 1, source_ndim):
                            expected_flat_stride = expected_flat_stride * self._full_grid_shape[d2]
                        if self._grid_strides[gdims[-1]] != expected_flat_stride:
                            is_contiguous = False

                if is_contiguous:
                    end_elem = min((start_tile + total_count) * self._tile_size[d], self._source_shape[d])
                    view = view.slice(view_dim, start_elem, end_elem)
                    view_dim = view_dim + 1
                else:
                    # Non-contiguous: introduce strided dimensions.
                    # tile_step = how many tiles we skip per grid step in this source dim
                    # elem_step = tile_step * tile_size[d] = element stride between blocks
                    suffix_product = 1
                    for d2 in range(d + 1, source_ndim):
                        suffix_product = suffix_product * self._full_grid_shape[d2]

                    gd = gdims[0]
                    tile_step = self._grid_strides[gd] // suffix_product
                    elem_step = tile_step * self._tile_size[d]
                    n_tiles = self._grid_shape[gd]

                    if elem_step > self._tile_size[d]:
                        # Take a range divisible by elem_step from start_elem
                        n_full_blocks = self._source_shape[d] // elem_step
                        range_end = n_full_blocks * elem_step
                        view = view.slice(view_dim, start_elem, range_end)
                        # Reshape into (n_full_blocks, elem_step)
                        view = view.reshape_dim(view_dim, (n_full_blocks, elem_step))
                        # Slice outer dim to select only the n_tiles blocks
                        view = view.slice(view_dim, 0, n_tiles)
                        # Slice inner dim to tile_size
                        view = view.slice(view_dim + 1, 0, self._tile_size[d])
                        # This source dim produced 2 view dims
                        view_dim = view_dim + 2
                    else:
                        end_elem = min(start_elem + total_count * self._tile_size[d], self._source_shape[d])
                        view = view.slice(view_dim, start_elem, end_elem)
                        view_dim = view_dim + 1

        return view

    def _build_view(self, check_contiguity):
        """Internal method to build a NkiTensor from the current grid state."""
        ndim = self._get_ndim()

        if ndim == 0:
            return self._get_single_tile_view(self._grid_offset)

        if check_contiguity:
            self._check_contiguity()
            return self._build_contiguous_view()
        else:
            return self._build_force_view()

    def _check_contiguity(self):
        """Verify that the current grid selection forms a contiguous rectangular block.

        The check works purely against the underlying memory layout, independent
        of dim_map. It computes the min and max flat tile index across the entire
        grid, unravels both to original grid coordinates, and verifies that for
        each source dimension the span equals the count (no gaps).

        This correctly handles dim_map=-1 entries from reshape/expand_dim,
        and non-trivial stride artifacts from reshape_dim + slice.
        """
        ndim = self._get_ndim()
        source_ndim = len(self._source_shape)

        # Compute total number of selected tiles
        total_tiles = 1
        for i in range(ndim):
            total_tiles = total_tiles * self._grid_shape[i]

        # Compute the min and max flat tile index across the full grid selection.
        # Min is at all-zero grid coords, max is at all-(shape-1) coords.
        min_flat = self._grid_offset
        max_flat = self._grid_offset
        for i in range(ndim):
            # Add the contribution of grid dim i at its max position
            max_flat = max_flat + (self._grid_shape[i] - 1) * self._grid_strides[i]

        # Unravel to original grid coordinates
        min_coords = self._unravel(min_flat, self._full_grid_shape)
        max_coords = self._unravel(max_flat, self._full_grid_shape)

        # For each source dim, verify the span is contiguous and the total
        # tile count decomposes correctly across source dims.
        # A contiguous rectangle requires: for each source dim d,
        #   span_d = max_coords[d] - min_coords[d] + 1
        # and product of all span_d == total_tiles
        product_of_spans = 1
        for d in range(source_ndim):
            span = max_coords[d] - min_coords[d] + 1
            product_of_spans = product_of_spans * span
            kernel_assert(
                min_coords[d] + span <= self._full_grid_shape[d],
                f"Tile range [{min_coords[d]}, {min_coords[d] + span}) "
                f"exceeds grid extent {self._full_grid_shape[d]} for source dim {d}. "
                f"Use force_get_view() instead.",
            )

        kernel_assert(
            product_of_spans == total_tiles,
            f"Tiles are not contiguous: selected {total_tiles} tiles but the bounding "
            f"box has {product_of_spans} tiles (min_coords={min_coords}, max_coords={max_coords}). "
            f"Use force_get_view() instead.",
        )

    def _build_dim_groups(self):
        """Group consecutive grid dims by their source dim mapping.

        Returns:
            List of [source_dim, [grid_dim_indices]] lists.
        """
        groups = []
        prev_src_dim = -2
        for i in range(self._get_ndim()):
            sd = self._dim_map[i]
            if sd == prev_src_dim:
                groups[-1][1].append(i)
            else:
                groups.append([sd, [i]])
                prev_src_dim = sd
        return groups

    # ─── Allocation and list-of-tiles support ────────────────────────────

    # When _tiles is set, this TiledTensor is backed by a list of individual
    # tensors rather than a single contiguous source.
    _tiles = None
    _num_banks = None

    def __getitem__(self, idx):
        """Access a tile by (i, j) index. Returns the raw tensor or NkiTensor.

        For contiguous-source TiledTensors, delegates to get_tile().
        For list-backed TiledTensors, returns the tile directly with rotation/bank logic.
        """
        if isinstance(idx, tuple):
            indices = idx
        else:
            indices = (idx,)

        if self._tiles is not None:
            ndim = len(self._grid_shape)
            if self._num_banks is not None:
                flat = 0
                stride = 1
                for d in range(ndim - 1, -1, -1):
                    flat = flat + indices[d] * stride
                    stride = stride * self._grid_shape[d]
                return self._tiles[flat % self._num_banks]
            # Rotation support
            phys_indices = list(indices)
            if self._rotate_dim is not None:
                phys_indices[self._rotate_dim] = indices[self._rotate_dim] % self._rotate_count
            # Compute flat index into physical tiles
            flat = 0
            stride = 1
            for d in range(ndim - 1, -1, -1):
                flat = flat + phys_indices[d] * stride
                if self._rotate_dim is not None and self._rotate_dim == d:
                    stride = stride * self._rotate_count
                else:
                    stride = stride * self._grid_shape[d]
            return self._tiles[flat]

        return self.get_tile(indices)

    def permute(self, perm):
        """Permute dimensions within each tile. Returns a new TiledTensor."""
        new_ts = []
        for p in perm:
            new_ts.append(self._tile_size[p])
        new_tile_size = tuple(new_ts)
        if self._tiles is not None:
            new_tiles = []
            for t in self._tiles:
                new_tiles.append(t.permute(perm))
            result = TiledTensor._make_tile_list(
                new_tiles,
                self._grid_shape,
                new_tile_size,
                num_banks=self._num_banks,
                rotate_dim=self._rotate_dim if self._rotate_dim is not None else None,
                rotate_count=self._rotate_count if self._rotate_count is not None else None,
            )
            return result
        # View-backed: permute the source and adjust tile_size
        new_source = self._source_view.permute(perm)
        return TiledTensor(new_source, new_tile_size)

    def broadcast(self, dim, size):
        """Broadcast a dimension within each tile. Returns a new TiledTensor."""
        if self._tiles is not None:
            new_tiles = []
            for t in self._tiles:
                new_tiles.append(t.broadcast(dim=dim, size=size))
            new_tile_size = list(self._tile_size)
            new_tile_size[dim] = size
            return TiledTensor._make_tile_list(
                new_tiles,
                self._grid_shape,
                tuple(new_tile_size),
                num_banks=self._num_banks,
                rotate_dim=self._rotate_dim if self._rotate_dim is not None else None,
                rotate_count=self._rotate_count if self._rotate_count is not None else None,
            )
        new_source = self._source_view.broadcast(dim=dim, size=size * self._grid_shape[dim])
        new_tile_size = list(self._tile_size)
        new_tile_size[dim] = size
        return TiledTensor(new_source, tuple(new_tile_size))

    @staticmethod
    def alloc(
        grid, tile_size, dtype, buffer=nl.sbuf, sbm=None, align=None, heap=False, rotate=None, num_banks=None, name=None
    ):
        """Allocate a TiledTensor grid of independent tiles.

        Args:
            grid: Tuple of logical tile counts per dimension.
            tile_size: Size of each physical tile.
            dtype: Data type for allocation.
            buffer: nl.sbuf or nl.psum.
            sbm: BufferManager. If set, uses sbm.alloc_stack per tile.
            align: Alignment for sbm.alloc_stack.
            rotate: (dim, count) — only allocate count physical tiles on that dim,
                    logical indices wrap modulo count.
            num_banks: Allocate this many physical tiles as a flat pool,
                      indices wrap modulo num_banks.
            name: Prefix for tensor names.

        Returns:
            TiledTensor backed by individually allocated tiles.
        """
        if num_banks is not None:
            tiles = []
            for b in range(num_banks):
                tile_name = name + "_bank" + str(b) if name is not None else None
                if sbm is not None:
                    tiles.append(
                        (sbm.alloc_heap if heap else sbm.alloc_stack)(
                            tile_size, dtype=dtype, buffer=buffer, name=tile_name, align=align
                        )
                    )
                else:
                    tiles.append(nl.ndarray(tile_size, dtype=dtype, buffer=buffer))
            return TiledTensor._make_tile_list(tiles, grid, tile_size, num_banks=num_banks)

        # Determine physical grid
        rotate_dim = None
        rotate_count = None
        phys_grid = list(grid)
        if rotate is not None:
            rotate_dim = rotate[0]
            rotate_count = rotate[1]
            phys_grid[rotate_dim] = rotate_count

        # Allocate tiles in row-major order of physical grid
        tiles = []
        total_phys = 1
        for d in range(len(phys_grid)):
            total_phys = total_phys * phys_grid[d]
        for i in range(total_phys):
            tile_name = name + "_" + str(i) if name is not None else None
            if sbm is not None:
                tiles.append(
                    (sbm.alloc_heap if heap else sbm.alloc_stack)(
                        tile_size, dtype=dtype, buffer=buffer, name=tile_name, align=align
                    )
                )
            else:
                tiles.append(nl.ndarray(tile_size, dtype=dtype, buffer=buffer))

        return TiledTensor._make_tile_list(tiles, grid, tile_size, rotate_dim=rotate_dim, rotate_count=rotate_count)

    @staticmethod
    def _make_tile_list(tiles, grid, tile_size, num_banks=None, rotate_dim=None, rotate_count=None):
        """Create a list-backed TiledTensor.

        Uses a dummy source to satisfy the base constructor, then overrides
        internals for list-based access.
        """
        # Create a minimal instance — use first tile as dummy source
        obj = TiledTensor(tiles[0], tile_size)
        obj._tiles = tiles
        obj._grid_shape = tuple(grid)
        obj._num_banks = num_banks
        obj._rotate_dim = rotate_dim
        obj._rotate_count = rotate_count
        # tile_size for matmul_loop_nest compatibility
        obj._tile_size = tuple(tile_size)
        return obj

    def sub_tile(self, dim, size, actual_size=None):
        """Sub-divide tiles along a dimension, expanding the grid.

        Each tile's dimension `dim` is split into chunks of `size`.
        The grid expands on that axis.

        Args:
            dim: Source dimension to sub-tile (0-indexed within tile_size).
                 Use -1 for last dimension.
            size: Sub-tile size along that dimension.
            actual_size: If set, only produce ceil(actual_size/size) sub-tiles
                        instead of using the full tile extent.

        Returns:
            New TiledTensor with expanded grid.
        """
        if dim < 0:
            dim = len(self._tile_size) + dim

        src_dim_size = actual_size if actual_size is not None else self._tile_size[dim]
        num_sub = (src_dim_size + size - 1) // size

        if self._tiles is None:
            # Source-backed: construct new TiledTensor with smaller tile_size,
            # then reshape to separate the head and sub-tile axes.
            new_tile_size = list(self._tile_size)
            new_tile_size[dim] = size
            new_tt = TiledTensor(self._source_view, tuple(new_tile_size))
            # The new grid's dim `dim` has ceil(source_shape[dim] / size) tiles.
            # We need to reshape it into (original_grid[dim], num_sub).
            # This only works if original tiles are evenly sub-divided.
            # If not, fall through to list-based expansion.
            orig_count = self._grid_shape[dim]
            new_count = new_tt._grid_shape[dim]
            if orig_count * num_sub == new_count:
                return new_tt.reshape_dim(dim, (orig_count, num_sub))
            # Fall through to list-based for non-even case

        # List-backed (or non-even source-backed): expand each tile into sub-tiles
        new_tiles = []
        num_orig_tiles = 1
        for d in range(len(self._grid_shape)):
            num_orig_tiles = num_orig_tiles * self._grid_shape[d]

        for t_idx in range(num_orig_tiles):
            tile = (
                self._tiles[t_idx] if self._tiles is not None else self._get_single_tile_view(self._grid_offset + t_idx)
            )
            for s in range(num_sub):
                start = s * size
                end = min(start + size, src_dim_size)
                if dim == 0:
                    new_tiles.append(tile[start:end])
                elif dim == 1 or (dim == -1 and len(self._tile_size) == 2):
                    new_tiles.append(tile[:, start:end])
                else:
                    new_tiles.append(tile[:, :, start:end])

        # Expand grid: insert num_sub on the sub-tiled axis
        new_grid = list(self._grid_shape)
        if len(new_grid) == 2 and new_grid[1] == 1 and dim == len(self._tile_size) - 1:
            new_grid[1] = num_sub
        elif dim < len(new_grid) and new_grid[dim] == 1:
            new_grid[dim] = num_sub
        else:
            # Multiply the last grid dim
            new_grid[-1] = new_grid[-1] * num_sub

        new_tile_size = list(self._tile_size)
        new_tile_size[dim] = size
        return TiledTensor._make_tile_list(new_tiles, tuple(new_grid), tuple(new_tile_size))
