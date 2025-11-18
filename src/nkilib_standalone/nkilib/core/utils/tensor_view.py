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

from typing import List

import nki.language as nl


class TensorView(nl.NKIObject):
    """A view wrapper around NKI tensors that supports various tensor operations.

    TensorView provides a convenient interface for tensor manipulation operations
    like slicing, permuting, broadcasting, and reshaping without copying data.
    It maintains metadata about tensor dimensions, sizes, strides, and offset
    to efficiently generate NKI array patterns.

    Attributes:
        base_tensor (nl.ndarray): The underlying NKI tensor
        sizes (List[int]): Size of each dimension
        strides (List[int]): Stride of each dimension in elements
        offset (int): Offset from the base tensor start in elements
    """

    base_tensor: nl.ndarray
    sizes: List[int]
    strides: List[int]
    offset: int

    def get_dim(self) -> int:
        return len(self.sizes)

    def is_sbuf(self) -> bool:
        return self.base_tensor.buffer == nl.sbuf

    @staticmethod
    def get_trivial_strides(sizes: List[int], base_stride: int = 1) -> List[int]:
        """Compute row-major (C-style) strides for given tensor sizes.
        Args:
            sizes: List of dimension sizes
            base_stride: Stride of the innermost dimension (default: 1)
        Returns:
            List of strides in row-major order
        Example:
            For sizes [2, 3, 4], returns [12, 4, 1] (assuming base_stride=1)
        """
        # Build strides from innermost to outermost dimension
        strides = [base_stride]
        for i in range(1, len(sizes)):
            # Each stride is the product of inner dimension size and previous stride
            strides.append(strides[i - 1] * sizes[len(sizes) - i])

        # Reverse to get row-major order (outermost to innermost)
        ret = []
        for i in range(len(sizes)):
            ret.append(strides[len(sizes) - i - 1])
        return ret

    def __init__(
        self,
        base_tensor: nl.ndarray,
        sizes: List[int] = None,
        strides: List[int] = None,
        offset: int = 0,
    ):
        """Initialize a TensorView.
        Args:
            base_tensor: The underlying NKI tensor
            sizes: Dimension sizes (defaults to base_tensor.shape)
            strides: Dimension strides (defaults to row-major strides)
            offset: Offset from base tensor start (default: 0)
        Raises:
            AssertionError: If strides contain non-positive values or dimensions mismatch
        """
        self.base_tensor = base_tensor
        # Use base tensor shape if sizes not provided
        self.sizes = list(base_tensor.shape) if sizes == None else sizes
        # Compute trivial strides if not provided
        self.strides = TensorView.get_trivial_strides(self.sizes) if strides == None else strides
        self.offset = 0 if offset == None else offset

        # Validate strides are non-negative (required for valid memory access)
        for i in range(len(self.strides)):
            assert self.strides[i] >= 0, f"Stride at dimension {i} must be non-negative, got {self.strides[i]}"
        # Ensure all dimension metadata is consistent
        assert len(self.sizes) == len(self.strides), "Dimension count mismatch"
        assert self.offset >= 0, "Offset must be non-negative"

    def get_view(self) -> nl.ndarray:
        """Generate the actual NKI tensor view using array pattern.
        Returns:
            NKI tensor with the specified view pattern applied
        """
        assert len(self.sizes) == len(self.strides)
        # Build array pattern as list of (stride, size) tuples
        ap_pattern = []
        for i in range(self.get_dim()):
            ap_pattern.append((self.strides[i], self.sizes[i]))
        return self.base_tensor.ap(pattern=ap_pattern, offset=self.offset)

    def slice(self, dim: int, start: int, end: int, step: int = 1) -> "TensorView":
        """Create a sliced view along a specific dimension.
        Args:
            dim: Dimension to slice
            start: Start index (inclusive)
            end: End index (exclusive)
            step: Step size (default: 1)
        Returns:
            New TensorView with the sliced dimension
        Example:
            for shape [X,Y,Z] and parameters (dim=1, start=1, end=4, step=2) we will get a shape of [X,2,Z]
        Raises:
            AssertionError: If slice parameters are invalid
        """
        assert dim < self.get_dim(), f"Dimension {dim} out of range for {self.get_dim()}D tensor"
        assert start >= 0, "Start index must be non-negative"
        assert end > start, "End index must be greater than start"
        assert end <= self.sizes[dim], f"End index {end} exceeds dimension size {self.sizes[dim]}"

        new_sizes = []
        new_strides = []
        for i in range(self.get_dim()):
            if i == dim:
                # Calculate new size accounting for step size
                new_sizes.append((end - start + step - 1) // step)
                # Adjust stride by step size
                new_strides.append(self.strides[i] * step)
            else:
                # Other dimensions remain unchanged
                new_sizes.append(self.sizes[i])
                new_strides.append(self.strides[i])

        # Adjust offset to account for start position
        new_offset = self.offset + self.strides[dim] * start
        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=new_offset)

    @staticmethod
    def validate_permutation(permutation: List[int], dim: int, is_sbuf: bool) -> None:
        assert len(permutation) == dim, f"Permutation length {len(permutation)} != dimension count {dim}"
        for i in range(dim):
            assert permutation[i] < dim, f"Permutation index {permutation[i]} >= dimension count {dim}"
            assert permutation[i] >= 0, f"Permutation index {permutation[i]} must be non-negative"
            # Check for duplicates
            for j in range(i):
                assert permutation[i] != permutation[j], f"Duplicate dimension {permutation[i]} in permutation"
        if is_sbuf:
            assert permutation[0] == 0, "Partition dimension stay the outermost dimension"

    def permute(self, dims: List[int]) -> "TensorView":
        """Create a permuted view by reordering dimensions.
        Args:
            dims: New order of dimensions (list of dimension indices)
        Returns:
            New TensorView with permuted dimensions
        Example:
            For a 3D tensor [X,Y,Z] and dims=[2, 0, 1] we will get a [Z,X,Y] view.
        """
        TensorView.validate_permutation(dims, self.get_dim(), self.is_sbuf())
        # verify correctness of partition dim
        new_sizes = []
        new_strides = []
        # Reorder sizes and strides according to permutation
        for i in range(len(dims)):
            d = dims[i]
            assert d < self.get_dim()  # Additional safety check
            new_sizes.append(self.sizes[d])
            new_strides.append(self.strides[d])
        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=self.offset)

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
        assert dim < self.get_dim(), f"Dimension {dim} out of range"
        assert self.sizes[dim] == 1, f"Can only broadcast size-1 dimensions, got size {self.sizes[dim]}"
        if self.is_sbuf():
            assert (dim != 0) or (
                size < nl.tile_size.pmax
            ), f"partition dim cannot be broadcasted into more than {nl.tile_size.pmax}"
        new_sizes = []
        new_strides = []
        for i in range(self.get_dim()):
            if i == dim:
                new_sizes.append(size)
                # Set stride to 0 for broadcasting (same element repeated)
                new_strides.append(0)
            else:
                # Other dimensions remain unchanged
                new_sizes.append(self.sizes[i])
                new_strides.append(self.strides[i])
        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=self.offset)

    def reshape_dim(self, dim: int, sizes: List[int]) -> "TensorView":
        """Reshape a single dimension into multiple dimensions.
        Args:
            dim: Dimension to reshape
            sizes: New sizes for the reshaped dimensions
        Returns:
            New TensorView with reshaped dimension
        Example:
            for shape [X,24,Z] and parameters (dim=1, sizes=[2,3,4]) we will get a shape of [X,2,3,4,Z]
        Note:
            The product of new sizes must equal the original dimension size
        """
        assert dim < self.get_dim(), f"Dimension {dim} out of range"
        if self.is_sbuf():
            # allow trivial reshape that does nothing
            assert (dim > 0) or (len(sizes) == 1), "partition dim cannot be reshaped"

        # Verify that new sizes have same total elements
        size_prod = 1
        for i in range(len(sizes)):
            size_prod *= sizes[i]
        assert self.sizes[dim] == size_prod, f"Size mismatch: {self.sizes[dim]} != {size_prod}"

        # Build new sizes by replacing the target dimension
        new_sizes = self.sizes[:dim] + sizes + self.sizes[dim + 1 :]

        # Compute strides for the reshaped dimensions
        reshaped_strides = TensorView.get_trivial_strides(sizes, base_stride=self.strides[dim])
        new_strides = self.strides[:dim] + reshaped_strides + self.strides[dim + 1 :]

        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=self.offset)

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
        assert start_dim < end_dim, "Start dimension must be less than end dimension"
        assert start_dim < self.get_dim(), f"Start dimension {start_dim} out of range"
        assert end_dim < self.get_dim(), f"End dimension {end_dim} out of range"
        if self.is_sbuf():
            assert start_dim > 0, "partition dim cannot be flattened"

        # Verify dimensions are contiguous in memory
        for i in range(start_dim, end_dim):
            assert (
                self.strides[i] == self.sizes[i + 1] * self.strides[i + 1]
            ), f"Dimensions {i} and {i+1} are not contiguous in memory"

        # Calculate total size of flattened dimension
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= self.sizes[i]

        # Build new sizes and strides
        new_sizes = self.sizes[:start_dim] + [flattened_size] + self.sizes[end_dim + 1 :]
        new_strides = self.strides[:start_dim] + [self.strides[end_dim]] + self.strides[end_dim + 1 :]

        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=self.offset)

    def expand_dim(self, dim: int) -> "TensorView":
        """Add a new dimension of size 1 at the specified position.
        Args:
            dim: Position to insert the new dimension
        Returns:
            New TensorView with an additional dimension
        Example:
            for shape [X,Y,Z] and parameters (dim=1) we will get a shape of [X,1,Y,Z]
        """
        assert dim <= self.get_dim(), f"Dimension {dim} out of range"
        if self.is_sbuf():
            assert dim > 0, "partition dim cannot be expanded"

        # Insert a new dimension of size 1 at the specified position
        new_stride = 1 if dim == self.get_dim() else self.strides[dim]
        new_sizes = self.sizes[:dim] + [1] + self.sizes[dim:]
        new_strides = self.strides[:dim] + [new_stride] + self.strides[dim:]
        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=self.offset)

    def squeeze_dim(self, dim: int) -> "TensorView":
        """Remove a dimension of size 1.
        Args:
            dim: Dimension to remove (must have size 1)
        Returns:
            New TensorView with the dimension removed
        Example:
            for shape [X,1,Y,Z] and parameters (dim=1) we will get a shape of [X,Y,Z]
        """
        assert dim < self.get_dim(), f"Dimension {dim} out of range"
        assert self.sizes[dim] == 1, f"Can only squeeze size-1 dimensions, got size {self.sizes[dim]}"
        if self.is_sbuf():
            assert dim > 0, "partition dim cannot be squeezed"

        # Remove the specified dimension
        new_sizes = self.sizes[:dim] + self.sizes[dim + 1 :]
        new_strides = self.strides[:dim] + self.strides[dim + 1 :]
        return TensorView(self.base_tensor, sizes=new_sizes, strides=new_strides, offset=self.offset)

    def select(self, dim: int, index: int) -> "TensorView":
        """Select a single element along a dimension, reducing dimensionality.
        Args:
            dim: Dimension to select from
            index: Index to select
        Returns:
            New TensorView with one fewer dimension
        Example:
            for shape [X,Y,Z] and parameters (dim=1, index=2) we will get a shape of [X,Z]
        """
        # Select by slicing a single element and then squeezing
        return self.slice(dim, index, index + 1).squeeze_dim(dim)

    def reshape(self, new_sizes: List[int]) -> "TensorView":
        """Reshape the tensor to new dimensions.
        Args:
            new_sizes: New dimension sizes
        Returns:
            New TensorView with reshaped dimensions
        Note:
            Currently not implemented. Would require checking memory contiguity
            and computing appropriate strides for the new shape.
        """
        # TODO: Implement general reshape functionality
        # This requires checking if the tensor is contiguous and computing
        # new strides that maintain the same memory layout
        assert False, "General reshape not yet implemented"
