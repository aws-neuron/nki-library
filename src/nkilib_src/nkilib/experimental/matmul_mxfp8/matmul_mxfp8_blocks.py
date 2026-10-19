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

"""Blocked matrix multiplication operations for MXFP8 quantized tensors using BIR AP patterns."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ..mxfp_utils.mxfp8_utils import _quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils._mxfp8_wrappers import matmul_mx_x4_wrapper
from ..mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor, _RatioSchedule


def _get_bir_ap_params(
    tensor_shape: tuple,
    k_idx: int,
    f_start: int,
    f_size: int,
    enable_scale_packing: bool = False,
) -> tuple:
    """
    Calculate BIR AP parameters for accessing 3D quantized tensor.

    Args:
        tensor_shape (Tuple[int, int, int]): Tensor shape [P, NUM_K_TILES, F].
            Note: for packed scale tensor NUM_K_TILES is NUM_SCALING_GROUPS.
        k_idx (int): K tile index (relative to block start)
        f_start (int): Starting offset in F dimension
        f_size (int): Size in F dimension
        enable_scale_packing (bool): Whether this is an SBUF scale buffer packed up to 4 tiles per
            scaling group (default: False)

    Returns:
        Tuple[List, int]: (pattern, offset) for BIR AP access

    Notes:
        For tensor shape [P, NUM_K_TILES, F]:
        - step_p = NUM_K_TILES * F (stride to next partition)
        - offset = k_idx * F + f_start + scale_partition_offset * step_p

        For packed scales, k_idx (block-local) maps to scaling group k_idx // 4 at partition
        offset 4 * (k_idx % 4). Every quantized operand -- quantized in-kernel, pre-quantized,
        or spill-reloaded -- is packed into SBUF this same block-local way.
    """
    P, NUM_K_TILES, F = tensor_shape
    step_p = NUM_K_TILES * F
    P_count = P

    if enable_scale_packing:
        tile_scaling_group_idx, _, scale_partition_offset = _quantize_mxfp8_utils.get_scale_packing_info(k_idx)
        offset = tile_scaling_group_idx * F + f_start + scale_partition_offset * step_p
        P_count = P - scale_partition_offset
    else:
        offset = k_idx * F + f_start

    return [[step_p, P_count], [1, f_size]], offset


def _k_inner_loop_helper(
    output_tensor: nl.NkiTensor,
    lhs_td: TensorDescriptor,
    rhs_td: TensorDescriptor,
    lhs_f_idx: int,
    rhs_f_idx: int,
    LHS_TILE_F: int,
    RHS_TILE_F: int,
    NUM_K_TILES: int,
    accumulate_output: bool,
    m_log_end: int,
    n_log_end: int,
    NUM_LHS_F_TILES: int,
    RHS_F: int,
    enable_scale_packing: bool = False,
    psum_dtype=nl.float32,
    psum_copy_in_schedule: Optional[_RatioSchedule] = None,
) -> None:
    """
    Process a single output tile by accumulating across K dimension using BIR AP.

    Args:
        output_tensor (nl.NkiTensor): Output tensor in SBUF
        lhs_td (TensorDescriptor): LHS tensor descriptor with .data, .scales
        rhs_td (TensorDescriptor): RHS tensor descriptor with .data, .scales
        lhs_f_idx (int): LHS tile index in F dimension
        rhs_f_idx (int): RHS tile index in F dimension
        LHS_TILE_F (int): LHS tile size in F dimension
        RHS_TILE_F (int): RHS tile size in F dimension
        NUM_K_TILES (int): Number of tiles in K dimension
        accumulate_output (bool): Whether to accumulate to existing output
        m_log_end (Optional[int]): Valid M dimension size for masking
        n_log_end (Optional[int]): Valid N dimension size for masking
        NUM_LHS_F_TILES (int): Number of LHS tiles in F dimension
        RHS_F (int): Total RHS F dimension size
        enable_scale_packing (bool): Whether the SBUF scale buffers are packed up to 4 tiles per scaling group (default: False)
        psum_copy_in_schedule: Schedule selecting eligible FP32 accumulator
            copies into PSUM. None disables copy-in.

    Returns:
        None
    """
    lhs_f_start = lhs_f_idx * LHS_TILE_F
    rhs_f_start = rhs_f_idx * RHS_TILE_F

    num_m = max(1, min(LHS_TILE_F, m_log_end - lhs_f_start)) if m_log_end != None else LHS_TILE_F
    num_n = max(1, min(RHS_TILE_F, n_log_end - rhs_f_start)) if n_log_end != None else RHS_TILE_F
    if m_log_end != None and lhs_f_start >= m_log_end:
        return
    if n_log_end != None and rhs_f_start >= n_log_end:
        return

    out_psum = nl.ndarray((num_m, num_n), psum_dtype, buffer=nl.psum)

    P = lhs_td.data.shape[0]

    # Output access pattern into the running SBUF accumulator.
    out_pattern = [[NUM_LHS_F_TILES * RHS_F, num_m], [1, num_n]]
    out_offset = lhs_f_idx * RHS_F + rhs_f_start
    out_ap = output_tensor.ap(pattern=out_pattern, offset=out_offset)

    # Copy-in requires an existing FP32 accumulator and a selected schedule entry.
    use_copy_method = (
        accumulate_output
        and psum_dtype == nl.float32
        and psum_copy_in_schedule is not None
        and psum_copy_in_schedule.select()
    )

    if use_copy_method:
        # Copy-in: load the running accumulator into PSUM so the matmuls below
        # can accumulate on top of it.
        nisa.tensor_copy(dst=out_psum, src=out_ap)

    # First K iteration. Accumulate onto the copied-in accumulator when the copy
    # method pre-loaded PSUM, otherwise start fresh.
    k_idx = 0

    lhs_data_pattern, lhs_data_offset = _get_bir_ap_params(lhs_td.data.shape, k_idx, lhs_f_start, num_m)
    lhs_scale_pattern, lhs_scale_offset = _get_bir_ap_params(
        lhs_td.scales.shape,
        k_idx,
        lhs_f_start,
        num_m,
        enable_scale_packing,
    )
    rhs_data_pattern, rhs_data_offset = _get_bir_ap_params(rhs_td.data.shape, k_idx, rhs_f_start, num_n)
    rhs_scale_pattern, rhs_scale_offset = _get_bir_ap_params(
        rhs_td.scales.shape,
        k_idx,
        rhs_f_start,
        num_n,
        enable_scale_packing,
    )

    matmul_mx_x4_wrapper(
        out_psum=out_psum,
        stationary_data=lhs_td.data.ap(pattern=lhs_data_pattern, offset=lhs_data_offset),
        stationary_scale=lhs_td.scales.ap(pattern=lhs_scale_pattern, offset=lhs_scale_offset),
        moving_data=rhs_td.data.ap(pattern=rhs_data_pattern, offset=rhs_data_offset),
        moving_scale=rhs_td.scales.ap(pattern=rhs_scale_pattern, offset=rhs_scale_offset),
        accumulate=use_copy_method,
        P=P,
        F_m=num_m,
        F_n=num_n,
    )

    # Remaining K iterations with accumulation
    for k_idx in range(1, NUM_K_TILES):
        lhs_data_pattern, lhs_data_offset = _get_bir_ap_params(lhs_td.data.shape, k_idx, lhs_f_start, num_m)
        lhs_scale_pattern, lhs_scale_offset = _get_bir_ap_params(
            lhs_td.scales.shape,
            k_idx,
            lhs_f_start,
            num_m,
            enable_scale_packing,
        )
        rhs_data_pattern, rhs_data_offset = _get_bir_ap_params(rhs_td.data.shape, k_idx, rhs_f_start, num_n)
        rhs_scale_pattern, rhs_scale_offset = _get_bir_ap_params(
            rhs_td.scales.shape,
            k_idx,
            rhs_f_start,
            num_n,
            enable_scale_packing,
        )

        matmul_mx_x4_wrapper(
            out_psum=out_psum,
            stationary_data=lhs_td.data.ap(pattern=lhs_data_pattern, offset=lhs_data_offset),
            stationary_scale=lhs_td.scales.ap(pattern=lhs_scale_pattern, offset=lhs_scale_offset),
            moving_data=rhs_td.data.ap(pattern=rhs_data_pattern, offset=rhs_data_offset),
            moving_scale=rhs_td.scales.ap(pattern=rhs_scale_pattern, offset=rhs_scale_offset),
            accumulate=True,
            P=P,
            F_m=num_m,
            F_n=num_n,
        )

    if accumulate_output and not use_copy_method:
        # Default path: Vector-engine add of the PSUM partial product into the
        # running accumulator.
        nisa.tensor_tensor(out_ap, out_psum, out_ap, op=nl.add)
    else:
        # Either a fresh write (accumulate_output=False) or the copy method has
        # already folded the previous accumulator into PSUM, so PSUM holds the
        # complete result.
        nisa.tensor_copy(dst=out_ap, src=out_psum)


def matmul_mxfp8_blocks(
    output_tensor: nl.NkiTensor,
    lhs_td: TensorDescriptor,
    LHS_tile_shape: tuple,
    rhs_td: TensorDescriptor,
    RHS_tile_shape: tuple,
    loop_order: str = "mnk",
    accumulate_output: bool = False,
    m_log_end: int = None,
    n_log_end: int = None,
    enable_scale_packing: bool = False,
    psum_dtype=nl.float32,
    num_k_tiles: int = None,
    psum_copy_in_schedule: Optional[_RatioSchedule] = None,
) -> None:
    """
    Perform blocked matmul between quantized tensors and store result in SBUF.

    # TODO: Consider wrapping output_tensor in a TensorDescriptor

    Args:
        output_tensor (nl.NkiTensor): Pre-allocated SBUF tensor of shape [NUM_LHS_F_TILES, 128, N]
        lhs_td (TensorDescriptor): LHS tensor descriptor with .data, .scales
            (each 3D [TILE_K, NUM_K_TILES, F])
        LHS_tile_shape (Tuple[int, int]): LHS tile shape [LHS_TILE_K, LHS_TILE_F]
        rhs_td (TensorDescriptor): RHS tensor descriptor with .data, .scales
            (each 3D [TILE_K, NUM_K_TILES, F])
        RHS_tile_shape (Tuple[int, int]): RHS tile shape [RHS_TILE_K, RHS_TILE_F]
        loop_order (str): Tile processing order ('mnk' or 'nmk')
        accumulate_output (bool): If True, accumulate to existing output (+=), else overwrite (=)
        m_log_end (Optional[int]): Valid M dimension size (block-relative) for masking
        n_log_end (Optional[int]): Valid N dimension size (block-relative) for masking
        enable_scale_packing (bool): Whether the SBUF scale buffers are packed up to 4 tiles per scaling group (default: False)
        psum_copy_in_schedule: Schedule selecting eligible FP32 accumulator
            copies into PSUM. None disables copy-in.
        num_k_tiles (Optional[int]): Number of valid K tiles in this block. When
            omitted, all K tiles in the loaded buffers are consumed. A partial
            final K block passes a smaller value so padded buffer slots are not
            multiplied.

    Returns:
        None

    Notes:
        - Output is written directly to output_tensor in SBUF
        - Supports two loop orders for different performance characteristics

    Pseudocode:
        TODO: Add pseudocode description

    Example:
        matmul_mxfp8_blocks(
            output_tensor=output_sbuf,
            lhs_td=TensorDescriptor(data=lhs_data, scales=lhs_scales),
            LHS_tile_shape=(128, 128),
            rhs_td=TensorDescriptor(data=rhs_td.data, scales=rhs_td.scales),
            RHS_tile_shape=(128, 128),
            loop_order='mnk'
        )
    """

    kernel_assert(
        len(LHS_tile_shape) == 2 and len(RHS_tile_shape) == 2,
        f"LHS_tile_shape and RHS_tile_shape must be tuples of length 2, got {len(LHS_tile_shape)=}, {len(RHS_tile_shape)=}",
    )
    kernel_assert(
        len(lhs_td.data.shape) == 3
        and len(lhs_td.scales.shape) == 3
        and len(rhs_td.data.shape) == 3
        and len(rhs_td.scales.shape) == 3,
        f"lhs_td.data, lhs_td.scales, rhs_td.data, and rhs_td.scales must be 3D tensors, got shapes {lhs_td.data.shape=}, {lhs_td.scales.shape=}, {rhs_td.data.shape=}, {rhs_td.scales.shape=}",
    )
    LHS_INPUT_TILE_K, LHS_NUM_K_TILES, LHS_F = lhs_td.data.shape
    RHS_INPUT_TILE_K, RHS_NUM_K_TILES, RHS_F = rhs_td.data.shape
    LHS_TILE_K, LHS_TILE_F = LHS_tile_shape
    RHS_TILE_K, RHS_TILE_F = RHS_tile_shape

    kernel_assert(LHS_TILE_K == RHS_TILE_K, f"LHS_TILE_K and RHS_TILE_K must match, got {LHS_TILE_K=}, {RHS_TILE_K=}")
    kernel_assert(
        LHS_NUM_K_TILES == RHS_NUM_K_TILES,
        f"LHS_NUM_K_TILES and RHS_NUM_K_TILES must match, got {LHS_NUM_K_TILES=}, {RHS_NUM_K_TILES=}",
    )
    if num_k_tiles is None:
        num_k_tiles = LHS_NUM_K_TILES
    kernel_assert(
        0 < num_k_tiles <= LHS_NUM_K_TILES,
        f"num_k_tiles must be in [1, {LHS_NUM_K_TILES}], got {num_k_tiles}",
    )
    NUM_K_TILES = num_k_tiles
    kernel_assert(LHS_TILE_F <= 128, f"LHS Tile F is used as stationary, cannot exceed 128, got {LHS_TILE_F=}")
    kernel_assert(
        LHS_TILE_K == LHS_INPUT_TILE_K and RHS_TILE_K == RHS_INPUT_TILE_K,
        f"LHS_TILE_K and RHS_TILE_K must match LHS_INPUT_TILE_K and RHS_INPUT_TILE_K, got {LHS_TILE_K=}, {RHS_TILE_K=}, {LHS_INPUT_TILE_K=}, {RHS_INPUT_TILE_K=}",
    )
    kernel_assert(
        LHS_TILE_F <= LHS_F and RHS_TILE_F <= RHS_F,
        f"LHS_TILE_F and RHS_TILE_F must be less than or equal to LHS_F and RHS_F, got {LHS_TILE_F=}, {RHS_TILE_F=}, {LHS_F=}, {RHS_F=}",
    )

    # Calculate number of tiles in F dimensions
    NUM_LHS_F_TILES = LHS_F // LHS_TILE_F
    NUM_RHS_F_TILES = RHS_F // RHS_TILE_F

    if loop_order == 'nmk' or loop_order == 'NMK':
        for rhs_f_idx in range(NUM_RHS_F_TILES):
            for lhs_f_idx in range(NUM_LHS_F_TILES):
                _k_inner_loop_helper(
                    output_tensor,
                    lhs_td,
                    rhs_td,
                    lhs_f_idx,
                    rhs_f_idx,
                    LHS_TILE_F,
                    RHS_TILE_F,
                    NUM_K_TILES,
                    accumulate_output,
                    m_log_end,
                    n_log_end,
                    NUM_LHS_F_TILES,
                    RHS_F,
                    enable_scale_packing,
                    psum_dtype,
                    psum_copy_in_schedule,
                )
    elif loop_order == 'mnk' or loop_order == 'MNK':
        for lhs_f_idx in range(NUM_LHS_F_TILES):
            for rhs_f_idx in range(NUM_RHS_F_TILES):
                _k_inner_loop_helper(
                    output_tensor,
                    lhs_td,
                    rhs_td,
                    lhs_f_idx,
                    rhs_f_idx,
                    LHS_TILE_F,
                    RHS_TILE_F,
                    NUM_K_TILES,
                    accumulate_output,
                    m_log_end,
                    n_log_end,
                    NUM_LHS_F_TILES,
                    RHS_F,
                    enable_scale_packing,
                    psum_dtype,
                    psum_copy_in_schedule,
                )
    else:
        kernel_assert(False, f"Unsupported loop_order '{loop_order}'. Supported loop orders are: 'mnk', 'nmk'.")
