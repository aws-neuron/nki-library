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
Cross-partition copy utility for SBUF tensor operations.

This utility enables copying data between SBUF tensors at arbitrary partition offsets.
It uses nc_stream_shuffle when direct tensor_copy cannot be used due to hardware partition
alignment constraints, and falls back to efficient tensor_copy when alignment permits.
"""

import nki.isa as nisa
import nki.language as nl

QUADRANT = 32
MASK_NO_CHANGE = 255  # Special value meaning the output tensor in partition [i] will be unmodified.


def cross_partition_copy(
    src: nl.ndarray,
    dst: nl.ndarray,
    src_start_partition: int,
    dst_start_partition: int,
    num_partitions_to_copy: int,
    free_dim_size: int,
):
    """
    Copy partitions from src to dst at specified partition offsets.

    Copies data from src[src_start_partition:src_start_partition+num_partitions_to_copy]
    to dst[dst_start_partition:dst_start_partition+num_partitions_to_copy].

    Uses nc_stream_shuffle to handle copies where the destination partition offset is not
    aligned to quadrant boundaries (multiples of 32). When alignment permits, uses efficient
    direct tensor_copy instead.

    Args:
        src (nl.ndarray): Source tensor in SBUF
        dst (nl.ndarray): Destination tensor in SBUF
        src_start_partition (int): Starting partition index in src
        dst_start_partition (int): Starting partition index in dst
        num_partitions_to_copy (int): Number of partitions to copy
        free_dim_size (int): Size of the free dimension

    Returns:
        None: Copies data in-place to dst tensor

    Notes:
        - If dst_start_partition is aligned to 32-partition boundary, uses direct tensor_copy
        - If copy stays within a single quadrant, uses single-quadrant shuffle
        - Otherwise splits into first-quadrant shuffle + remaining aligned copy
    """
    dst_end = dst_start_partition + num_partitions_to_copy

    # Case 1: destination is aligned to 32-partition boundary, tensor_copy directly
    if dst_start_partition % QUADRANT == 0:
        nisa.tensor_copy(
            src=src[src_start_partition : src_start_partition + num_partitions_to_copy, :free_dim_size],
            dst=dst[dst_start_partition : dst_start_partition + num_partitions_to_copy, :free_dim_size],
        )
        return

    # Calculate quadrant info for dst
    dst_start_quadrant = dst_start_partition // QUADRANT
    dst_offset_in_quadrant = dst_start_partition % QUADRANT
    dst_end_quadrant = (dst_end - 1) // QUADRANT  # -1 because dst_end is exclusive

    # Case 2: Copy stays within a single quadrant
    if dst_start_quadrant == dst_end_quadrant:
        _shuffle_within_quadrant(
            src,
            dst,
            src_start_partition,
            dst_start_partition,
            num_partitions_to_copy,
            free_dim_size,
        )
        return

    # Case 3: Cross-quadrant copy that spans multiple quadrants
    first_portion_count = QUADRANT - dst_offset_in_quadrant
    _shuffle_within_quadrant(
        src,
        dst,
        src_start_partition,
        dst_start_partition,
        first_portion_count,
        free_dim_size,
    )

    # Handle remaining data in chunks of QUADRANT to support large T
    remaining_src_start = src_start_partition + first_portion_count
    remaining_dst_start = (dst_start_quadrant + 1) * QUADRANT
    remaining_count = num_partitions_to_copy - first_portion_count

    while remaining_count > 0:
        # Process at most QUADRANT elements per iteration
        chunk_count = min(remaining_count, QUADRANT)

        if remaining_src_start % QUADRANT == 0:
            nisa.tensor_copy(
                src=src[remaining_src_start : remaining_src_start + chunk_count, :free_dim_size],
                dst=dst[remaining_dst_start : remaining_dst_start + chunk_count, :free_dim_size],
            )
        else:
            _aligned_copy_from_unaligned_src(
                src,
                dst,
                remaining_src_start,
                remaining_dst_start,
                chunk_count,
                free_dim_size,
            )

        remaining_src_start += chunk_count
        remaining_dst_start += chunk_count
        remaining_count -= chunk_count


def _shuffle_within_quadrant(
    src: nl.ndarray,
    dst: nl.ndarray,
    src_start_partition: int,
    dst_start_partition: int,
    num_partitions_to_copy: int,
    free_dim_size: int,
):
    """
    Shuffle data within a single 32-partition quadrant using temp buffers.

    nc_stream_shuffle requires operating on full 32-partition quadrants, but src/dst
    tensors may be smaller. We use quadrant-sized temp buffers to satisfy the hardware requirements.

    Args:
        src (nl.ndarray): Source tensor
        dst (nl.ndarray): Destination tensor
        src_start_partition (int): Starting partition in source
        dst_start_partition (int): Starting partition in destination
        num_partitions_to_copy (int): Number of partitions to copy
        free_dim_size (int): Size of the free dimension

    Returns:
        None: Copies data in-place to dst tensor
    """
    dst_quadrant = dst_start_partition // QUADRANT
    dst_offset_in_quadrant = dst_start_partition % QUADRANT
    dst_quadrant_base = dst_quadrant * QUADRANT

    # If dst offset within quadrant is 0, use direct tensor_copy
    if dst_offset_in_quadrant == 0:
        nisa.tensor_copy(
            src=src[src_start_partition : src_start_partition + num_partitions_to_copy, :free_dim_size],
            dst=dst[dst_start_partition : dst_start_partition + num_partitions_to_copy, :free_dim_size],
        )
        return

    # SHUFFLE PATH
    temp_src = nl.ndarray((QUADRANT, free_dim_size), dtype=src.dtype, buffer=nl.sbuf)
    temp_dst = nl.ndarray((QUADRANT, free_dim_size), dtype=dst.dtype, buffer=nl.sbuf)
    nisa.memset(dst=temp_src, value=0.0)
    nisa.memset(dst=temp_dst, value=0.0)

    nisa.tensor_copy(
        src=src[src_start_partition : src_start_partition + num_partitions_to_copy, :free_dim_size],
        dst=temp_src[0:num_partitions_to_copy, :free_dim_size],
    )

    if dst_offset_in_quadrant > 0:
        nisa.tensor_copy(
            src=dst[dst_quadrant_base : dst_quadrant_base + dst_offset_in_quadrant, :free_dim_size],
            dst=temp_dst[0:dst_offset_in_quadrant, :free_dim_size],
        )

    shuffle_mask = [MASK_NO_CHANGE] * QUADRANT
    for i in range(num_partitions_to_copy):
        shuffle_mask[dst_offset_in_quadrant + i] = i

    nisa.nc_stream_shuffle(
        dst=temp_dst,
        src=temp_src,
        shuffle_mask=shuffle_mask,
    )

    copy_back_size = min(dst_offset_in_quadrant + num_partitions_to_copy, QUADRANT)
    nisa.tensor_copy(
        src=temp_dst[0:copy_back_size, :free_dim_size],
        dst=dst[dst_quadrant_base : dst_quadrant_base + copy_back_size, :free_dim_size],
    )


def _aligned_copy_from_unaligned_src(
    src: nl.ndarray,
    dst: nl.ndarray,
    src_start_partition: int,
    dst_start_partition: int,
    num_partitions_to_copy: int,
    free_dim_size: int,
):
    """
    Copy data from non-aligned source to aligned destination using shuffle.

    When src_start_partition is not aligned to QUADRANT boundary but dst_start_partition is,
    we use shuffle to align the source data before copying. Handles source data spanning
    multiple quadrants using double-shuffle approach.

    Args:
        src (nl.ndarray): Source tensor
        dst (nl.ndarray): Destination tensor (dst_start_partition must be QUADRANT-aligned)
        src_start_partition (int): Starting partition in source (may be non-aligned)
        dst_start_partition (int): Starting partition in destination (must be aligned)
        num_partitions_to_copy (int): Number of partitions to copy
        free_dim_size (int): Size of the free dimension
    """
    src_quadrant = src_start_partition // QUADRANT
    src_offset_in_quadrant = src_start_partition % QUADRANT
    src_quadrant_base = src_quadrant * QUADRANT

    # Calculate how much data is in first quadrant vs spills to next quadrant
    first_q_available = QUADRANT - src_offset_in_quadrant
    first_q_count = min(first_q_available, num_partitions_to_copy)
    second_q_count = num_partitions_to_copy - first_q_count

    # Check if source spans multiple quadrants
    if second_q_count > 0:
        temp_src0 = nl.ndarray((QUADRANT, free_dim_size), dtype=src.dtype, buffer=nl.sbuf)
        temp_src1 = nl.ndarray((QUADRANT, free_dim_size), dtype=src.dtype, buffer=nl.sbuf)
        temp_aligned = nl.ndarray((QUADRANT, free_dim_size), dtype=src.dtype, buffer=nl.sbuf)
        nisa.memset(dst=temp_src0, value=0.0)
        nisa.memset(dst=temp_src1, value=0.0)
        nisa.memset(dst=temp_aligned, value=0.0)

        nisa.tensor_copy(
            src=src[src_quadrant_base : src_quadrant_base + QUADRANT, :free_dim_size],
            dst=temp_src0[0:QUADRANT, :free_dim_size],
        )

        align_mask0 = [MASK_NO_CHANGE] * QUADRANT
        for i in range(first_q_count):
            align_mask0[i] = src_offset_in_quadrant + i

        nisa.nc_stream_shuffle(dst=temp_aligned, src=temp_src0, shuffle_mask=align_mask0)
        next_src_quadrant_base = (src_quadrant + 1) * QUADRANT
        nisa.tensor_copy(
            src=src[next_src_quadrant_base : next_src_quadrant_base + second_q_count, :free_dim_size],
            dst=temp_src1[0:second_q_count, :free_dim_size],
        )

        align_mask1 = [MASK_NO_CHANGE] * QUADRANT
        for i in range(second_q_count):
            align_mask1[first_q_count + i] = i
        nisa.nc_stream_shuffle(dst=temp_aligned, src=temp_src1, shuffle_mask=align_mask1)
        nisa.tensor_copy(
            src=temp_aligned[0:num_partitions_to_copy, :free_dim_size],
            dst=dst[dst_start_partition : dst_start_partition + num_partitions_to_copy, :free_dim_size],
        )
    else:
        temp_src = nl.ndarray((QUADRANT, free_dim_size), dtype=src.dtype, buffer=nl.sbuf)
        temp_aligned = nl.ndarray((QUADRANT, free_dim_size), dtype=src.dtype, buffer=nl.sbuf)
        nisa.memset(dst=temp_src, value=0.0)
        nisa.memset(dst=temp_aligned, value=0.0)
        src_copy_end = src_offset_in_quadrant + num_partitions_to_copy
        nisa.tensor_copy(
            src=src[src_quadrant_base : src_quadrant_base + src_copy_end, :free_dim_size],
            dst=temp_src[0:src_copy_end, :free_dim_size],
        )

        align_mask = [MASK_NO_CHANGE] * QUADRANT
        for i in range(num_partitions_to_copy):
            align_mask[i] = src_offset_in_quadrant + i

        nisa.nc_stream_shuffle(dst=temp_aligned, src=temp_src, shuffle_mask=align_mask)

        nisa.tensor_copy(
            src=temp_aligned[0:num_partitions_to_copy, :free_dim_size],
            dst=dst[dst_start_partition : dst_start_partition + num_partitions_to_copy, :free_dim_size],
        )
