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


"""TensorView: A wrapper for NKI tensor array pattern operations.

This module provides a high-level interface for tensor view operations on NKI arrays,
similar to PyTorch tensor views. It allows for efficient tensor manipulation without
data copying by using NKI's array pattern (ap) functionality.
"""

from typing import Dict, List, Optional, Tuple, Union

import nki.language as nl

from .kernel_assert import kernel_assert
from .logging import Logger

# Create logger instance
logger = Logger("TensorView")


class TensorView(nl.NKIObject):
    """A view wrapper around NKI tensors that supports various tensor operations.

    TensorView provides a convenient interface for tensor manipulation operations
    like slicing, permuting, broadcasting, and reshaping without copying data.
    It maintains metadata about tensor dimensions, shape, strides, and offset
    to efficiently generate NKI array patterns.

    Attributes:
        base_tensor (nl.ndarray): The underlying NKI tensor
        shape (Tuple[int, ...]): Size of each dimension
        strides (Tuple[int, ...]): Stride of each dimension in elements
        offset (int): Offset from the base tensor start in elements
    """

    base_tensor: nl.ndarray
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int
    dtype: object
    scalar_offset: nl.ndarray = None
    indirect_dim: Optional[int] = None

    def get_dim(self) -> int:
        return len(self.shape)

    def is_sbuf(self) -> bool:
        return self.base_tensor.buffer == nl.sbuf

    @staticmethod
    def get_trivial_strides(shape: Tuple[int, ...], base_stride: int = 1) -> Tuple[int, ...]:
        """Compute row-major (C-style) strides for given tensor shape.
        Args:
            shape: Tuple of dimension sizes
            base_stride: Stride of the innermost dimension (default: 1)
        Returns:
            Tuple of strides in row-major order
        Example:
            For shape (2, 3, 4), returns (12, 4, 1) (assuming base_stride=1)
        """
        # Build strides from innermost to outermost dimension
        strides = [base_stride]
        for i in range(1, len(shape)):
            # Each stride is the product of inner dimension size and previous stride
            strides.append(strides[i - 1] * shape[len(shape) - i])

        # Reverse to get row-major order (outermost to innermost)
        ret = []
        for i in range(len(shape)):
            ret.append(strides[len(shape) - i - 1])
        return tuple(ret)

    def __init__(self, base_tensor: nl.ndarray):
        """Initialize a TensorView.
        Args:
            base_tensor: The underlying NKI tensor
        Raises:
            AssertionError: If base_tensor is None
        """
        kernel_assert(base_tensor is not None, "Base tensor cannot be None")
        self.base_tensor = base_tensor
        self.shape = tuple(base_tensor.shape)
        self.strides = TensorView.get_trivial_strides(self.shape)
        self.offset = 0
        self.dtype = base_tensor.dtype
        self.scalar_offset = None
        self.indirect_dim = None

    def _copy(
        self,
        shape: Tuple[int, ...] = None,
        strides: Tuple[int, ...] = None,
        offset: int = None,
        scalar_offset: nl.ndarray = None,
        indirect_dim: Optional[int] = None,
    ) -> "TensorView":
        """Create a copy of this TensorView with optionally modified shape, strides, or offset.
        Args:
            shape: New shape (defaults to current shape)
            strides: New strides (defaults to current strides)
            offset: New offset (defaults to current offset)
            scalar_offset: New scalar_offset (defaults to current scalar_offset)
            indirect_dim: New indirect_dim (defaults to current indirect_dim)
        Returns:
            New TensorView with specified modifications
        Raises:
            AssertionError: If strides contain negative values or dimensions mismatch
        """
        view = TensorView(self.base_tensor)
        view.shape = tuple(shape) if shape is not None else self.shape
        view.strides = tuple(strides) if strides is not None else self.strides
        view.offset = offset if offset is not None else self.offset
        view.scalar_offset = scalar_offset if scalar_offset is not None else self.scalar_offset
        view.indirect_dim = indirect_dim if indirect_dim is not None else self.indirect_dim

        # Validate strides are non-negative (required for valid memory access)
        for i in range(len(view.strides)):
            kernel_assert(view.strides[i] >= 0, f"Stride at dimension {i} must be non-negative, got {view.strides[i]}")
        # Ensure all dimension metadata is consistent
        kernel_assert(len(view.shape) == len(view.strides), "Dimension count mismatch")
        kernel_assert(view.offset >= 0, "Offset must be non-negative")
        return view

    def _get_pattern_and_offset(self):
        """Generate the NKI tensor view pattern and offset.

        This helper is useful when debugging or porting existing patterns to TensorView.
        Returns:
            Pattern and offset corresponding to the view
        """
        ap_pattern = []
        for i in range(self.get_dim()):
            ap_pattern.append((self.strides[i], self.shape[i]))
        return ap_pattern, self.offset

    def get_view(self) -> nl.ndarray:
        """Generate the actual NKI tensor view using array pattern.
        Returns:
            NKI tensor with the specified view pattern applied
        """
        kernel_assert(len(self.shape) == len(self.strides), "len(self.shape) == len(self.strides)")
        # Build array pattern as list of (stride, size) tuples
        ap_pattern, offset = self._get_pattern_and_offset()

        if self.indirect_dim != None:
            result = self.base_tensor.ap(
                pattern=ap_pattern,
                offset=offset,
                scalar_offset=self.scalar_offset,
                indirect_dim=self.indirect_dim,
            )
        else:
            result = self.base_tensor.ap(pattern=ap_pattern, offset=offset)
        return result

    def slice(self, dim: int, start: int, end: int, step: int = 1) -> "TensorView":
        """Create a sliced view along a specific dimension.
        Args:
            dim: Dimension to slice
            start: Start index (inclusive)
            end: End index (exclusive), clamped to shape[dim] if out of bounds
            step: Step size (default: 1)
        Returns:
            New TensorView with the sliced dimension
        Example:
            for shape [X,Y,Z] and parameters (dim=1, start=1, end=4, step=2) we will get a shape of [X,2,Z]
        Raises:
            AssertionError: If slice parameters are invalid
        """
        kernel_assert(dim < self.get_dim(), f"Dimension {dim} out of range for {self.get_dim()}D tensor")
        kernel_assert(start >= 0, "Start index must be non-negative")
        kernel_assert(end > start, "End index must be greater than start")

        # Clamp end to be within bounds of shape[dim]
        end = min(end, self.shape[dim])

        new_shape = []
        new_strides = []
        for i in range(self.get_dim()):
            if i == dim:
                # Calculate new size accounting for step size
                new_shape.append((end - start + step - 1) // step)
                # Adjust stride by step size
                new_strides.append(self.strides[i] * step)
            else:
                # Other dimensions remain unchanged
                new_shape.append(self.shape[i])
                new_strides.append(self.strides[i])

        # Adjust offset to account for start position
        new_offset = self.offset + self.strides[dim] * start
        return self._copy(shape=new_shape, strides=new_strides, offset=new_offset)

    @staticmethod
    def validate_permutation(permutation: Tuple[int, ...], dim: int, is_sbuf: bool) -> None:
        kernel_assert(len(permutation) == dim, f"Permutation length {len(permutation)} != dimension count {dim}")
        for i in range(dim):
            kernel_assert(permutation[i] < dim, f"Permutation index {permutation[i]} >= dimension count {dim}")
            kernel_assert(permutation[i] >= 0, f"Permutation index {permutation[i]} must be non-negative")
            # Check for duplicates
            for j in range(i):
                kernel_assert(permutation[i] != permutation[j], f"Duplicate dimension {permutation[i]} in permutation")
        if is_sbuf:
            kernel_assert(permutation[0] == 0, "Partition dimension stay the outermost dimension")

    def permute(self, dims: Tuple[int, ...]) -> "TensorView":
        """Create a permuted view by reordering dimensions.
        Args:
            dims: New order of dimensions (tuple of dimension indices)
        Returns:
            New TensorView with permuted dimensions
        Example:
            For a 3D tensor (X,Y,Z) and dims=(2, 0, 1) we will get a (Z,X,Y) view.
        """
        TensorView.validate_permutation(dims, self.get_dim(), self.is_sbuf())
        # verify correctness of partition dim
        new_shape = []
        new_strides = []
        # Reorder shape and strides according to permutation
        for i in range(len(dims)):
            d = dims[i]
            kernel_assert(d < self.get_dim(), "Dimension index out of range")  # Additional safety check
            new_shape.append(self.shape[d])
            new_strides.append(self.strides[d])

        return self._copy(shape=new_shape, strides=new_strides)

    def broadcast(self, dim: int, size: int) -> "TensorView":
        """Create a broadcasted view by expanding a size-1 dimension.
        Args:
            dim: Dimension to broadcast (must have size 1)
            size: New size for the dimension
        Returns:
            New TensorView with broadcasted dimension
        Example:
            for shape [X,1,Z] and parameters (dim=1, size=8) we will get a shape of [X,8,Z]
        Note:
            Broadcasting sets stride to 0, so the same element is repeated
        """
        kernel_assert(dim < self.get_dim(), f"Dimension {dim} out of range")
        kernel_assert(self.shape[dim] == 1, f"Can only broadcast size-1 dimensions, got size {self.shape[dim]}")
        if self.is_sbuf():
            kernel_assert(dim != 0, "Cannot broadcast on partition dimension (dim=0) for SBUF tensors")
        new_shape = []
        new_strides = []
        for i in range(self.get_dim()):
            if i == dim:
                new_shape.append(size)
                # Set stride to 0 for broadcasting (same element repeated)
                new_strides.append(0)
            else:
                # Other dimensions remain unchanged
                new_shape.append(self.shape[i])
                new_strides.append(self.strides[i])

        return self._copy(shape=new_shape, strides=new_strides)

    def _reshape_dim_handle_minus_one(self, dim: int, shape: Tuple[int]) -> Tuple[int]:
        """Handle -1 in reshape shape by computing the inferred dimension size.
        Args:
            dim: Dimension being reshaped
            shape: Shape with possibly one -1 element
        Returns:
            Shape with -1 replaced by computed value
        """
        # Handle -1 in shape
        minus_one_index = None
        prod_shape = 1
        for i in range(len(shape)):
            if shape[i] == -1:
                kernel_assert(minus_one_index is None, "Only one dimension can be reshaped to -1")
                minus_one_index = i
            else:
                prod_shape *= shape[i]

        if minus_one_index is None:
            # No -1, return original shape
            return shape

        kernel_assert(self.shape[dim] % prod_shape == 0, "Cannot reshape with -1")
        new_shape = []
        for i in range(len(shape)):
            if i != minus_one_index:
                new_shape.append(shape[i])
            else:
                new_shape.append(self.shape[dim] // prod_shape)
        return tuple(new_shape)

    def reshape_dim(self, dim: int, shape: Tuple[int, ...]) -> "TensorView":
        """Reshape a single dimension into multiple dimensions.
        Args:
            dim: Dimension to reshape
            shape: New sizes for the reshaped dimensions (can contain at most one -1)
        Returns:
            New TensorView with reshaped dimension
        Example:
            for shape (X,24,Z) and parameters (dim=1, shape=(2,3,4)) we will get a shape of (X,2,3,4,Z)
            for shape (X,24,Z) and parameters (dim=1, shape=(2,-1,4)) we will get a shape of (X,2,3,4,Z)
        Note:
            The product of new shape must equal the original dimension size
        """
        kernel_assert(dim < self.get_dim(), f"Dimension {dim} out of range")
        if self.is_sbuf():
            # allow trivial reshape that does nothing
            kernel_assert((dim > 0) or (len(shape) == 1), "partition dim cannot be reshaped")

        shape = self._reshape_dim_handle_minus_one(dim, shape)
        # Verify that new sizes have same total elements
        size_prod = 1
        for i in range(len(shape)):
            size_prod *= shape[i]
        kernel_assert(self.shape[dim] == size_prod, f"Size mismatch: {self.shape[dim]} != {size_prod}")

        # Build new shape by replacing the target dimension
        if self.get_dim() > 1:
            new_shape = tuple(list(self.shape[:dim]) + list(shape) + list(self.shape[dim + 1 :]))
        else:
            new_shape = shape

        # Compute strides for the reshaped dimensions
        reshaped_strides = TensorView.get_trivial_strides(shape, base_stride=self.strides[dim])
        new_strides = tuple(list(self.strides[:dim]) + list(reshaped_strides) + list(self.strides[dim + 1 :]))

        return self._copy(shape=new_shape, strides=new_strides)

    def flatten_dims(self, start_dim: int, end_dim: int) -> "TensorView":
        """Flatten a range of dimensions into a single dimension.
        Args:
            start_dim: First dimension to flatten (inclusive)
            end_dim: Last dimension to flatten (inclusive)
        Returns:
            New TensorView with flattened dimensions
        Example:
            for shape [X,2,3,4,Z] and parameters (start_dim=1, end_dim=3) we will get a shape of [X,24,Z]
        Note:
            Dimensions must be contiguous in memory for flattening to work
        """
        kernel_assert(start_dim < end_dim, "Start dimension must be less than end dimension")
        kernel_assert(start_dim < self.get_dim(), f"Start dimension {start_dim} out of range")
        kernel_assert(end_dim < self.get_dim(), f"End dimension {end_dim} out of range")
        if self.is_sbuf():
            kernel_assert(start_dim > 0, "partition dim cannot be flattened")

        # Verify dimensions are contiguous in memory
        for i in range(start_dim, end_dim):
            kernel_assert(
                self.strides[i] == self.shape[i + 1] * self.strides[i + 1],
                f"Dimensions {i} and {i+1} are not contiguous in memory",
            )

        # Calculate total size of flattened dimension
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= self.shape[i]

        # Build new shape and strides
        new_shape = tuple(list(self.shape[:start_dim]) + [flattened_size] + list(self.shape[end_dim + 1 :]))
        new_strides = tuple(
            list(self.strides[:start_dim]) + [self.strides[end_dim]] + list(self.strides[end_dim + 1 :])
        )

        return self._copy(shape=new_shape, strides=new_strides)

    def expand_dim(self, dim: int) -> "TensorView":
        """Add a new dimension of size 1 at the specified position.
        Args:
            dim: Position to insert the new dimension
        Returns:
            New TensorView with an additional dimension
        Example:
            for shape [X,Y,Z] and parameters (dim=1) we will get a shape of [X,1,Y,Z]
        """
        kernel_assert(dim <= self.get_dim(), f"Dimension {dim} out of range")
        if self.is_sbuf():
            kernel_assert(dim > 0, "partition dim cannot be expanded")

        # Insert a new dimension of size 1 at the specified position
        # Stride for new dim = stride needed to skip over elements at that position
        if dim == self.get_dim():
            new_stride = 1
        else:
            new_stride = self.strides[dim] * self.shape[dim]
        new_shape = tuple(list(self.shape[:dim]) + [1] + list(self.shape[dim:]))
        new_strides = tuple(list(self.strides[:dim]) + [new_stride] + list(self.strides[dim:]))

        return self._copy(shape=new_shape, strides=new_strides)

    def squeeze_dim(self, dim: int) -> "TensorView":
        """Remove a dimension of size 1.
        Args:
            dim: Dimension to remove (must have size 1)
        Returns:
            New TensorView with the dimension removed
        Example:
            for shape [X,1,Y,Z] and parameters (dim=1) we will get a shape of [X,Y,Z]
        """
        kernel_assert(dim < self.get_dim(), f"Dimension {dim} out of range")
        kernel_assert(self.shape[dim] == 1, f"Can only squeeze size-1 dimensions, got size {self.shape[dim]}")
        if self.is_sbuf():
            kernel_assert(dim > 0, "partition dim cannot be squeezed")

        # Remove the specified dimension
        new_shape = tuple(list(self.shape[:dim]) + list(self.shape[dim + 1 :]))
        new_strides = tuple(list(self.strides[:dim]) + list(self.strides[dim + 1 :]))

        return self._copy(shape=new_shape, strides=new_strides)

    def _dynamic_select(self, dim: int, index: nl.ndarray) -> "TensorView":
        """Dynamic select - find base tensor dim by stride matching.

        Args:
            dim: View dimension to select from
            index: Dynamic index tensor (scalar in SBUF)
        Returns:
            New TensorView with dynamic indexing configured
        """
        kernel_assert(self.indirect_dim is None, "Cannot have multiple dynamic selects")
        kernel_assert(self.strides[dim] != 0, "Cannot dynamic select on broadcast dimension (stride=0)")

        view_stride = self.strides[dim]
        base_strides = TensorView.get_trivial_strides(self.base_tensor.shape)

        # Find base dim with matching stride
        base_dim = None
        for i in range(len(base_strides)):
            if base_strides[i] == view_stride:
                base_dim = i
                break
        kernel_assert(
            base_dim is not None,
            f"No base dim with stride {view_stride}; dynamic select not supported after slice(step>1) or reshape",
        )

        # Remove the selected dimension from view
        new_shape = self.shape[:dim] + self.shape[dim + 1 :]
        new_strides = self.strides[:dim] + self.strides[dim + 1 :]

        return self._copy(shape=new_shape, strides=new_strides, scalar_offset=index, indirect_dim=base_dim)

    def select(self, dim: int, index: Union[int, nl.ndarray]) -> "TensorView":
        """Select a single element along a dimension, reducing dimensionality.
        Args:
            dim: Dimension to select from
            index: Index to select (int for static, nl.ndarray[shape=(1,1)] for dynamic indexing)
        Returns:
            New TensorView with one fewer dimension
        Example:
            Static: for shape [X,Y,Z] and parameters (dim=1, index=2) we will get a shape of [X,Z]
            Dynamic: for shape [E,X,Y] and parameters (dim=0, index=scalar_tensor) we will get a shape of [X,Y]
        """
        if not isinstance(index, int):
            return self._dynamic_select(dim, index)
        # Static select by slicing a single element and then squeezing
        new_view = self.slice(dim, index, index + 1)
        return new_view.squeeze_dim(dim)

    # delete this once "key in dict" is supported [NKIFE-594]
    @staticmethod
    def key_in_dict(key, dicti):
        for k in dicti.keys():
            if k == key:
                return True
        return False

    @staticmethod
    def _rearrange_detect_src_reshapes(
        src_pattern: Tuple[Union[str, Tuple[str]]], fixed_sizes: Dict[str, int]
    ) -> List[Dict]:
        """Detect reshape operations needed in source pattern based on grouped dimensions.

        Args:
            src_pattern: Source einops-style dimensions pattern (with nesting)
            fixed_sizes: Dictionary mapping dimension names to their known sizes

        Returns:
            List of reshape operations, each dict contains reshape_dim params (dim, shape as tuple)
        """
        src_reshapes = []
        dim_offset = 0
        for i in range(len(src_pattern)):
            if isinstance(src_pattern[i], tuple):
                shape = []
                for j in range(len(src_pattern[i])):
                    if TensorView.key_in_dict(src_pattern[i][j], fixed_sizes):
                        shape.append(fixed_sizes[src_pattern[i][j]])
                    else:
                        shape.append(-1)
                src_reshapes.append({'dim': i + dim_offset, 'shape': tuple(shape)})
                dim_offset += len(shape) - 1
        return src_reshapes

    @staticmethod
    def _rearrange_detect_dst_flattens(dst_pattern: Tuple[Union[str, Tuple[str]]]) -> List[Dict]:
        """Detect flatten operations needed in destination pattern based on grouped dimensions.

        Args:
            dst_pattern: Destination einops-style dimensions pattern (with nesting)

        Returns:
            List of flatten operations, each dict contains flatten_dims params (start_dim, end_dim)
        """
        dst_flattens = []
        dim_offset = 0
        for i in range(len(dst_pattern)):
            if isinstance(dst_pattern[i], tuple):
                dst_flattens.append({'start_dim': i + dim_offset, 'end_dim': i + dim_offset + len(dst_pattern[i]) - 1})
                dim_offset += len(dst_pattern[i]) - 1
        return dst_flattens

    @staticmethod
    def _rearrange_expand_pattern(pattern: Tuple[Union[str, Tuple[str]]]) -> Tuple[str]:
        """Expand grouped dimension patterns into flat list of dimension names.

        Args:
            pattern: einops-style dimension pattern (with nesting)

        Returns:
            Flat tuple of all dimension names in order
        """
        ret = []
        for i in range(len(pattern)):
            if isinstance(pattern[i], tuple):
                for j in range(len(pattern[i])):
                    ret.append(pattern[i][j])
            else:
                ret.append(pattern[i])
        return tuple(ret)

    @staticmethod
    def _rearrange_get_permutation(src_pattern: Tuple[str], dst_pattern: Tuple[str]) -> Tuple[int, ...]:
        """Calculate permutation indices to reorder dimensions from source to destination pattern.

        Args:
            src_pattern: Flat Tuple of source dimension names in current order
            dst_pattern: Flat Tuple of destination dimension names in desired order

        Returns:
            Tuple of indices indicating how to permute source dimensions to match destination
        """
        permutation = []
        for i in range(len(dst_pattern)):
            for j in range(len(src_pattern)):
                if src_pattern[j] == dst_pattern[i]:
                    permutation.append(j)
                    break
        return tuple(permutation)

    def rearrange(
        self,
        src_pattern: Tuple[Union[str, Tuple[str]]],
        dst_pattern: Tuple[Union[str, Tuple[str]]],
        fixed_sizes: Dict[str, int] = None,
    ) -> "TensorView":
        """Rearrange tensor dimensions using einops-style patterns.

        Args:
            src_pattern: Source dimension pattern with named dimensions, grouped dimensions in tuples
            dst_pattern: Destination dimension pattern with named dimensions, grouped dimensions in tuples
            fixed_sizes: Dictionary mapping dimension names to their sizes for reshaping

        Returns:
            New TensorView with rearranged dimensions

        Example:
            # Reshape and transpose: (batch, height*width, channels) -> (batch, channels, height, width)
            tensor.rearrange(('b', ('h', 'w'), 'c'), ('b', 'c', 'h', 'w'), {'h': 32})

        Note:
            Combines reshape, permute, and flatten operations to transform tensor layout
        """
        fixed_sizes = {} if fixed_sizes is None else fixed_sizes
        src_reshapes = TensorView._rearrange_detect_src_reshapes(src_pattern, fixed_sizes)
        src_ordering = TensorView._rearrange_expand_pattern(src_pattern)
        dst_flattens = TensorView._rearrange_detect_dst_flattens(dst_pattern)
        dst_ordering = TensorView._rearrange_expand_pattern(dst_pattern)
        permutation = TensorView._rearrange_get_permutation(src_ordering, dst_ordering)

        t = self._copy()
        for reshape in src_reshapes:
            t = t.reshape_dim(reshape['dim'], reshape['shape'])
        t = t.permute(permutation)
        for flatten in dst_flattens:
            t = t.flatten_dims(flatten['start_dim'], flatten['end_dim'])
        return t

    def reshape(self, new_shape: Tuple[int, ...]) -> "TensorView":
        """Reshape the tensor to new dimensions.
        Args:
            new_shape: New dimension shape
        Returns:
            New TensorView with reshaped dimensions
        Note:
            Currently not implemented. Would require checking memory contiguity
            and computing appropriate strides for the new shape.
        """
        # TODO: Implement general reshape functionality
        # This requires checking if the tensor is contiguous and computing
        # new strides that maintain the same memory layout
        kernel_assert(False, "General reshape not yet implemented")

    def has_dynamic_access(self) -> bool:
        """Check if the tensor has dynamic access (i.e., non-contiguous memory layout).
        Returns:
            True if the tensor has dynamic access, False otherwise
        """
        return self.scalar_offset != None and self.indirect_dim != None
