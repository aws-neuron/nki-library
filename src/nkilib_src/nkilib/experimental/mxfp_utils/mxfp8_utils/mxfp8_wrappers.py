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

"""Wrapper functions for MXFP8 quantization and matrix multiplication operations using BIR AP patterns."""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from .common_utils import get_active_sbm


def interleave_load_wrapper(dst_sbuf, src_tensor):
    """
    Load tensor from HBM to SBUF with interleaved layout for MX instructions.

    Divides input into 4 sections along P dimension and interleaves them element-wise
    along F dimension in SBUF.

    Args:
            dst_sbuf: Pre-allocated SBUF tensor with shape (P // 4, F * 4)
            src_tensor: Input HBM tensor with shape (P, F)

    Uses BIR AP to directly load from HBM to interleaved SBUF layout without temp buffer.
    """
    P, F = src_tensor.shape
    kernel_assert(P % 4 == 0, f"P must be divisible by 4, got {P=}")

    """
    Use BIR AP to directly load from HBM to interleaved destination.
    dst_sbuf shape: (P // 4, F * 4)
    Pattern: [[step_p, count_p], [step_f, count_f]]
    For interleaved layout: step_f = 4 (stride of 4 in free dim), count_f = F
    step_p = F * 4 (full row stride); offset = i (0, 1, 2, 3 for each quarter)
    """
    for i in range(4):
        p_start = i * (P // 4)
        nisa.dma_copy(
            src=src_tensor[p_start : p_start + P // 4, 0:F],
            dst=dst_sbuf.ap(pattern=[[F * 4, P // 4], [4, F]], offset=i),
        )


def quantize_mx_x4_wrapper(dst_data, dst_scale, src_sbuf, P: int = None, F: int = None, scale_partition_index: int = 0):
    """
    Quantize BF16 tensor to MX format with x4 data types.

    Supports both regular tensors and BIR AP views directly - no extra copies needed.

    Args:
            dst_data: Pre-allocated SBUF tensor or BIR AP view for quantized data
            dst_scale: Pre-allocated SBUF scale tensor or BIR AP view in FLOAT8_E8M0FNU
            src_sbuf: Input SBUF tensor or BIR AP view in BF16
            P: Partition dimension size (required when using BIR AP views)
            F: Free dimension size (required when using BIR AP views)
            scale_partition_index: Scale packing index (0, 4, 8, 12) when P > 32

    Example with BIR AP (no extra copies):
            OLD: tensor = nl.ndarray((N, D, H))
                 for tile in range(num_tiles):
                     i_n, i_d = nl.mgrid[0:tile_n, 0:tile_d]
                     tensor[tile*tile_n + i_n, i_d, offset]

            NEW: for tile in range(num_tiles):
                     result = tensor.ap(pattern=[[D*H, tile_n], [H, tile_d]],
                                       offset=tile*tile_n*D*H + offset)
    """
    # Get dimensions from shape if not provided (for regular tensors)
    if P is None or F is None:
        P, F = src_sbuf.shape

    kernel_assert(F % 4 == 0, f"F must be divisible by 4, got {F=}")
    kernel_assert(P % 32 == 0, f"P must be divisible by 32, got {P=}")

    # Use BIR AP views directly in nisa.quantize_mx - no intermediate copies
    if P <= 32:
        nisa.quantize_mx(src=src_sbuf, dst=dst_data, dst_scale=dst_scale)
    else:
        nisa.quantize_mx(src=src_sbuf, dst=dst_data, dst_scale=dst_scale)


def matmul_mx_x4_wrapper(
    out_psum,
    stationary_data,
    stationary_scale,
    moving_data,
    moving_scale,
    stationary_scale_partition_index: int = 0,
    moving_scale_partition_index: int = 0,
    accumulate: bool = True,
    P: int = None,
    F_m: int = None,
    F_n: int = None,
):
    """
    Perform matrix multiplication on MX quantized tensors.

    Supports both regular tensors and BIR AP views directly - no extra copies.

    Args:
            out_psum: Pre-allocated PSUM tensor, shape (F_m, F_n)
            stationary_data: Quantized stationary matrix (P, F_m) in MX x4 format or BIR AP view
            stationary_scale: Scale tensor for stationary matrix or BIR AP view
            moving_data: Quantized moving matrix (P, F_n) in MX x4 format or BIR AP view
            moving_scale: Scale tensor for moving matrix or BIR AP view
            stationary_scale_partition_index: Scale packing index when P > 32
            moving_scale_partition_index: Scale packing index when P > 32
            accumulate: Whether to accumulate into out_psum or overwrite
            P: Partition dimension size (required when using BIR AP views)
            F_m: Stationary free dimension size (required when using BIR AP views)
            F_n: Moving free dimension size (required when using BIR AP views)
    """
    # Get dimensions from shape if not provided (for regular tensors)
    if P is None or F_m is None:
        P, F_m = stationary_data.shape
    if F_n is None:
        _, F_n = moving_data.shape

    kernel_assert(F_m % 2 == 0, f"Stationary matrix F_m must be even for MX matmul, got {F_m=}")
    kernel_assert(F_n % 2 == 0, f"Moving matrix F_n must be even for MX matmul, got {F_n=}")

    if P > 32:
        kernel_assert(F_m >= 2, f"F_m must be >= 2 for P > 32, got {F_m=}, {P=}")
        kernel_assert(F_n >= 2, f"F_n must be >= 2 for P > 32, got {F_n=}, {P=}")

    # Perform matmul directly into out_psum - no intermediate copies
    # Use dst and accumulate parameters
    nisa.nc_matmul_mx(
        stationary=stationary_data,
        moving=moving_data,
        stationary_scale=stationary_scale,
        moving_scale=moving_scale,
        dst=out_psum[0:F_m, 0:F_n],
    )


def mx_quantize_and_matmul_x4(matrices_a, matrices_b, out, m: int, n: int, k: int, compute_dtype=nl.float8_e5m2_x4):
    """
    Perform MX quantization and matmul in a single operation.

    Args:
            matrices_a: Input matrix A from HBM, shape (k, m)
            matrices_b: Input matrix B from HBM, shape (k, n)
            out: Output matrix in HBM, shape (m, n)
            m, n, k: Matrix dimensions
            compute_dtype: MX compute dtype (default: float8_e5m2_x4)

    Returns:
            Output matrix with matmul result
    """
    sbm = get_active_sbm()

    scale_dtype = nl.float8_e8m0fnu

    stationary_sbuf = sbm.alloc_stack(shape=(k // 4, m * 4), dtype=matrices_a.dtype, buffer=nl.sbuf)
    moving_sbuf = sbm.alloc_stack(shape=(k // 4, n * 4), dtype=matrices_b.dtype, buffer=nl.sbuf)

    interleave_load_wrapper(stationary_sbuf, matrices_a)
    interleave_load_wrapper(moving_sbuf, matrices_b)

    mx_a_data = sbm.alloc_stack(shape=(k // 4, m), dtype=compute_dtype, buffer=nl.sbuf)
    mx_b_data = sbm.alloc_stack(shape=(k // 4, n), dtype=compute_dtype, buffer=nl.sbuf)

    if k // 4 <= 32:
        scale_shape_p = (k // 4) // 8
    else:
        scale_shape_p = k // 4

    mx_a_scale = sbm.alloc_stack(shape=(scale_shape_p, m), dtype=scale_dtype, buffer=nl.sbuf)
    mx_b_scale = sbm.alloc_stack(shape=(scale_shape_p, n), dtype=scale_dtype, buffer=nl.sbuf)

    quantize_mx_x4_wrapper(mx_a_data, mx_a_scale, stationary_sbuf)
    quantize_mx_x4_wrapper(mx_b_data, mx_b_scale, moving_sbuf)

    result = nl.ndarray((m, n), dtype=nl.bfloat16, buffer=nl.psum)
    matmul_mx_x4_wrapper(result, mx_a_data, mx_a_scale, mx_b_data, mx_b_scale, accumulate=False)

    result_sbuf = sbm.alloc_stack(shape=(m, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(result_sbuf[0:m, 0:n], result[0:m, 0:n])
    nisa.dma_copy(out[0:m, 0:n], result_sbuf[0:m, 0:n])
    return out


def mx_quantize_and_matmul_x4_packed_scale(
    matrices_a, matrices_b, out, m: int, n: int, k: int, compute_dtype=nl.float8_e5m2_x4
):
    """
    Perform MX quantization and matmul with packed scales for memory efficiency.

    Args:
            matrices_a: Input matrix A from HBM, shape (k, m)
            matrices_b: Input matrix B from HBM, shape (k, n)
            out: Output matrix in HBM, shape (m, n)
            m, n, k: Matrix dimensions
            compute_dtype: MX compute dtype (default: float8_e5m2_x4)

    Returns:
            Output matrix with matmul result
    """
    sbm = get_active_sbm()

    scale_dtype = nl.float8_e8m0fnu

    stationary_sbuf = sbm.alloc_stack(shape=(k // 4, m * 4), dtype=matrices_a.dtype, buffer=nl.sbuf)
    moving_sbuf = sbm.alloc_stack(shape=(k // 4, n * 4), dtype=matrices_b.dtype, buffer=nl.sbuf)

    interleave_load_wrapper(stationary_sbuf, matrices_a)
    interleave_load_wrapper(moving_sbuf, matrices_b)

    mx_a_data = sbm.alloc_stack(shape=(k // 4, m), dtype=compute_dtype, buffer=nl.sbuf)
    mx_b_data = sbm.alloc_stack(shape=(k // 4, n), dtype=compute_dtype, buffer=nl.sbuf)

    if k // 4 <= 32:
        scale_shape_p = (k // 4) // 8
    else:
        scale_shape_p = k // 4

    # Shared scale buffer for both matrices
    max_tiles = max(m, n)
    shared_scale = sbm.alloc_stack(shape=(scale_shape_p, max_tiles), dtype=scale_dtype, buffer=nl.sbuf)

    # Quantize with packed scales
    quantize_mx_x4_wrapper(mx_a_data, shared_scale, stationary_sbuf, scale_partition_index=0)
    quantize_mx_x4_wrapper(mx_b_data, shared_scale, moving_sbuf, scale_partition_index=4)

    # Perform matmul
    result = nl.ndarray((m, n), dtype=nl.bfloat16, buffer=nl.psum)
    matmul_mx_x4_wrapper(
        result,
        mx_a_data,
        shared_scale,
        mx_b_data,
        shared_scale,
        stationary_scale_partition_index=0,
        moving_scale_partition_index=4,
        accumulate=False,
    )

    # Store result - PSUM cannot be directly copied to HBM
    # First copy PSUM -> SBUF using tensor_copy, then SBUF -> HBM using dma_copy
    result_sbuf = sbm.alloc_stack(shape=(m, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(result_sbuf[0:m, 0:n], result[0:m, 0:n])
    nisa.dma_copy(out[0:m, 0:n], result_sbuf[0:m, 0:n])
    return out
