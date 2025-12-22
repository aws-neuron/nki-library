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

"""
Input validation functions for loss kernels.
"""

import nki.language as nl

from ..utils.kernel_assert import kernel_assert

# Hardware constraints for Trainium2 SBUF allocations
P_MAX = 128  # Maximum elements in partition dimension
F_MAX = 65536  # Maximum elements in free dimension (64K)
MAX_ALLOCATION_BYTES = 24 * 1024 * 1024  # 24 MiB SBUF limit


def validate_cross_entropy_forward_inputs(
    logits_hbm: nl.ndarray,
    targets_hbm: nl.ndarray,
    positions_per_batch: int,
    chunk_size: int,
    func_name: str = "cross_entropy_forward",
) -> None:
    """
    Validate cross entropy forward kernel inputs against hardware constraints.

    Ensures that input tensors and parameters meet the kernel's requirements before execution
    to catch errors early with clear messages. Validates against Trainium2 hardware limits:
    P_MAX=128, F_MAX=65536, MAX_ALLOCATION=24 MiB.

    Dimensions:
        num_positions: Total positions (B * T)
        V: Vocabulary size
        positions_per_batch: Positions processed together (must be ≤ P_MAX)
        chunk_size: Vocabulary chunk size (must be ≤ F_MAX)

    Args:
        logits_hbm (nl.ndarray): Input logits tensor to validate.
        targets_hbm (nl.ndarray): Target indices tensor to validate.
        positions_per_batch (int): Number of positions to process per batch.
        chunk_size (int): Size of vocabulary chunks for processing.
        func_name (str): Name of calling function for error messages. Default: "cross_entropy_forward_kernel".

    Returns:
        None: Raises AssertionError if validation fails.

    Notes:
        - Validation checks include: tensor ranks, shape compatibility, data types, dimension constraints,
          chunk size requirements, and batch size limits
        - logits_hbm must be 2D [num_positions, vocab_size]
        - targets_hbm must be 1D [num_positions]
        - targets_hbm must be int32, logits_hbm must be bfloat16 or float32
        - positions_per_batch must be in (0, MAX_PARTITION_DIM]
        - chunk_size must be in (0, vocab_size] and not exceed MAX_FREE_DIM
        - Per-allocation size: positions_per_batch × chunk_size × dtype_bytes ≤ MAX_ALLOCATION_BYTES

    Pseudocode:
        # Validate tensor structure
        assert len(logits_hbm.shape) == 2
        assert len(targets_hbm.shape) == 1
        assert targets_hbm.shape[0] == logits_hbm.shape[0]

        # Validate dtypes
        assert targets_hbm.dtype == int32
        assert logits_hbm.dtype in (bfloat16, float32)

        # Validate dimensions
        assert num_positions > 0 and vocab_size > 0
        assert 0 < chunk_size <= min(vocab_size, F_MAX)
        assert 0 < positions_per_batch <= P_MAX

        # Validate memory constraints
        allocation_bytes = positions_per_batch * chunk_size * dtype_bytes
        assert allocation_bytes <= MAX_ALLOCATION_BYTES
    """
    # Validate tensor ranks
    kernel_assert(
        len(logits_hbm.shape) == 2,
        f"{func_name}: logits_hbm must be 2D [num_positions, vocab_size], got shape {logits_hbm.shape}",
    )
    kernel_assert(
        len(targets_hbm.shape) == 1,
        f"{func_name}: targets_hbm must be 1D [num_positions], got shape {targets_hbm.shape}",
    )

    # Get dimensions
    num_positions = logits_hbm.shape[0]
    vocab_size = logits_hbm.shape[1]

    # Validate shapes match
    kernel_assert(
        targets_hbm.shape[0] == num_positions,
        f"{func_name}: targets_hbm length {targets_hbm.shape[0]} must match logits_hbm first dim {num_positions}",
    )

    # Validate dtypes
    kernel_assert(targets_hbm.dtype == nl.int32, f"{func_name}: targets_hbm must be int32, got {targets_hbm.dtype}")
    kernel_assert(
        logits_hbm.dtype in (nl.bfloat16, nl.float32),
        f"{func_name}: logits_hbm must be bfloat16 or float32, got {logits_hbm.dtype}",
    )

    # Validate dimensions
    kernel_assert(num_positions > 0, f"{func_name}: num_positions must be > 0, got {num_positions}")
    kernel_assert(vocab_size > 0, f"{func_name}: vocab_size must be > 0, got {vocab_size}")

    # Validate chunk_size
    kernel_assert(chunk_size > 0, f"{func_name}: chunk_size must be > 0, got {chunk_size}")
    kernel_assert(chunk_size <= vocab_size, f"{func_name}: chunk_size {chunk_size} must be <= vocab_size {vocab_size}")
    kernel_assert(
        chunk_size <= F_MAX,
        f"{func_name}: chunk_size {chunk_size} exceeds hardware limit {F_MAX}",
    )

    # Validate positions_per_batch
    kernel_assert(
        positions_per_batch > 0 and positions_per_batch <= P_MAX,
        f"{func_name}: positions_per_batch must be in (0, {P_MAX}], got {positions_per_batch}",
    )

    # Validate per-allocation size constraint (critical for compilation)
    dtype_bytes = 2 if logits_hbm.dtype == nl.bfloat16 else 4
    allocation_bytes = positions_per_batch * chunk_size * dtype_bytes
    kernel_assert(
        allocation_bytes <= MAX_ALLOCATION_BYTES,
        f"{func_name}: allocation size {allocation_bytes} bytes (positions_per_batch={positions_per_batch} × "
        f"chunk_size={chunk_size} × {dtype_bytes} bytes) exceeds hardware limit {MAX_ALLOCATION_BYTES} bytes. "
        f"Reduce positions_per_batch or chunk_size.",
    )
