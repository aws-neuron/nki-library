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

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ..mxfp_utils.mxfp8_utils import quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ..mxfp_utils.mxfp8_utils.mxfp8_wrappers import matmul_mx_x4_wrapper


def _get_bir_ap_params(
    tensor_shape: tuple,
    k_idx: int,
    f_start: int,
    f_size: int,
    is_packed_scales: bool = False,
    enable_scale_packing: bool = False,
    global_block_tile_k_idx: int = 0,
) -> tuple:
    """
    Calculate BIR AP parameters for accessing 3D quantized tensor.

    Args:
        tensor_shape (Tuple[int, int, int]): Tensor shape [P, NUM_K_TILES, F].
            Note: for packed scale tensor NUM_K_TILES is NUM_SCALING_GROUPS.
        k_idx (int): K tile index (relative to block start)
        f_start (int): Starting offset in F dimension
        f_size (int): Size in F dimension
        is_packed_scales (bool): Whether this is for packed scales tensor from prequantized load (default: False)
        enable_scale_packing (bool): Whether scale packing is enabled (default: False)
        global_block_tile_k_idx (int): Global K tile index at start of block (default: 0)

    Returns:
        Tuple[List, int]: (pattern, offset) for BIR AP access

    Notes:
        For tensor shape [P, NUM_K_TILES, F]:
        - step_p = NUM_K_TILES * F (stride to next partition)
        - offset = k_idx * F + f_start + scale_partition_offset * step_p

        For packed scales, k_idx is mapped to scaling_group_idx (k_idx // 4)
    """
    P, NUM_K_TILES, F = tensor_shape
    step_p = NUM_K_TILES * F
    P_count = P

    """
    For packed scales, use k_idx to calculate where in the scale tensor we should access.

    If is_packed_scales is true, the scale tensor was pre-packed in HBM,
    and we need to access it using global index.

    If is_packed_scales is false but enable_scale_packing is true, we need
    to access scale tensor using local index relative to the block.

    Otherwise we do a simple access pattern into tensor to get tile at
    k_idx from f_start.
    """
    if is_packed_scales:
        global_data_tile_k_idx = global_block_tile_k_idx + k_idx

        block_scaling_group_idx, _, _ = quantize_mxfp8_utils.get_scale_packing_info(global_block_tile_k_idx)
        tile_scaling_group_idx, _, scale_partition_offset = quantize_mxfp8_utils.get_scale_packing_info(
            global_data_tile_k_idx
        )
        sbuf_scaling_group_idx = tile_scaling_group_idx - block_scaling_group_idx

        offset = sbuf_scaling_group_idx * F + f_start + scale_partition_offset * step_p

        P_count = P - scale_partition_offset
    elif enable_scale_packing:
        tile_scaling_group_idx, _, scale_partition_offset = quantize_mxfp8_utils.get_scale_packing_info(k_idx)
        offset = tile_scaling_group_idx * F + f_start + scale_partition_offset * step_p
        P_count = P - scale_partition_offset
    else:
        offset = k_idx * F + f_start

    return [[step_p, P_count], [1, f_size]], offset


def _k_inner_loop_helper(
    output_tensor: nl.ndarray,
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
    global_block_tile_k_idx: int = 0,
    psum_dtype=nl.float32,
    tile_iter_idx: int = 0,
    enable_psum_copy_in: bool = False,
    inputs_were_prequantized: bool = False,
) -> None:
    """
    Process a single output tile by accumulating across K dimension using BIR AP.

    Args:
        output_tensor (nl.ndarray): Output tensor in SBUF
        lhs_td (TensorDescriptor): LHS tensor descriptor with .data, .scales, .scales_are_packed
        rhs_td (TensorDescriptor): RHS tensor descriptor with .data, .scales, .scales_are_packed
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
        enable_scale_packing (bool): Whether scale packing is enabled (default: False)
        global_block_tile_k_idx (int): Global K tile index at start of block (default: 0)
        tile_iter_idx (int): Sequential index of this M/N output tile within the block.
            Used to cycle the copy method — only applied every 3rd tile
            (tile_iter_idx % 3 == 1) to avoid oversubscribing the scalar engine
            (default: 0).
        enable_psum_copy_in (bool): Whether to use the PSUM copy-in method for
            cross-block accumulation. Skipped when inputs_were_prequantized is True.
            (default: False).
        inputs_were_prequantized (bool): Whether the original HBM inputs were
            pre-quantized (i.e. no online BF16->MXFP8 quantization). When True,
            the copy method is skipped since there is no online quant to overlap.
            (default: False).

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

    # Copy the running SBUF accumulator into PSUM before matmul to avoid tensor-tensor
    # adds on the vector engine. Conditions:
    #   1. accumulate_output: multiple K-blocks require cross-block accumulation
    #   2. not inputs_were_prequantized: only useful when online quantization loads the
    #      vector engine (pre-quantized inputs have no quant work to overlap with)
    #   3. enable_psum_copy_in: caller opted in to the copy-in method
    #   4. tile_iter_idx % 3 == 1: amortize scalar engine load (tuned hyper-parameter)
    #   5. psum_dtype == nl.float32: TensorCopy into PSUM requires 4-byte aligned dtype
    #      on core_v4 (trn3); bfloat16 PSUM writes are only legal from matmult instructions
    use_copy_method = (
        accumulate_output
        and not inputs_were_prequantized
        and enable_psum_copy_in
        and (tile_iter_idx % 3 == 1)
        and psum_dtype == nl.float32
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
        lhs_td.scales_are_packed,
        enable_scale_packing,
        global_block_tile_k_idx,
    )
    rhs_data_pattern, rhs_data_offset = _get_bir_ap_params(rhs_td.data.shape, k_idx, rhs_f_start, num_n)
    rhs_scale_pattern, rhs_scale_offset = _get_bir_ap_params(
        rhs_td.scales.shape,
        k_idx,
        rhs_f_start,
        num_n,
        rhs_td.scales_are_packed,
        enable_scale_packing,
        global_block_tile_k_idx,
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
            lhs_td.scales_are_packed,
            enable_scale_packing,
            global_block_tile_k_idx,
        )
        rhs_data_pattern, rhs_data_offset = _get_bir_ap_params(rhs_td.data.shape, k_idx, rhs_f_start, num_n)
        rhs_scale_pattern, rhs_scale_offset = _get_bir_ap_params(
            rhs_td.scales.shape,
            k_idx,
            rhs_f_start,
            num_n,
            rhs_td.scales_are_packed,
            enable_scale_packing,
            global_block_tile_k_idx,
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
    output_tensor: nl.ndarray,
    lhs_td: TensorDescriptor,
    LHS_tile_shape: tuple,
    rhs_td: TensorDescriptor,
    RHS_tile_shape: tuple,
    loop_order: str = "mnk",
    accumulate_output: bool = False,
    m_log_end: int = None,
    n_log_end: int = None,
    enable_scale_packing: bool = False,
    global_block_tile_k_idx: int = 0,
    psum_dtype=nl.float32,
    enable_psum_copy_in: bool = False,
    inputs_were_prequantized: bool = False,
) -> None:
    """
    Perform blocked matmul between quantized tensors and store result in SBUF.

    # TODO: Consider wrapping output_tensor in a TensorDescriptor

    Args:
        output_tensor (nl.ndarray): Pre-allocated SBUF tensor of shape [NUM_LHS_F_TILES, 128, N]
        lhs_td (TensorDescriptor): LHS tensor descriptor with .data, .scales
            (each 3D [TILE_K, NUM_K_TILES, F]), .scales_are_packed
        LHS_tile_shape (Tuple[int, int]): LHS tile shape [LHS_TILE_K, LHS_TILE_F]
        rhs_td (TensorDescriptor): RHS tensor descriptor with .data, .scales
            (each 3D [TILE_K, NUM_K_TILES, F]), .scales_are_packed
        RHS_tile_shape (Tuple[int, int]): RHS tile shape [RHS_TILE_K, RHS_TILE_F]
        loop_order (str): Tile processing order ('mnk' or 'nmk')
        accumulate_output (bool): If True, accumulate to existing output (+=), else overwrite (=)
        m_log_end (Optional[int]): Valid M dimension size (block-relative) for masking
        n_log_end (Optional[int]): Valid N dimension size (block-relative) for masking
        enable_scale_packing (bool): Whether scale packing is enabled (default: False)
        global_block_tile_k_idx (int): Global K tile index at start of block (default: 0)

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
    NUM_K_TILES = LHS_NUM_K_TILES
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
        tile_iter_idx = 0
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
                    global_block_tile_k_idx,
                    psum_dtype,
                    tile_iter_idx,
                    enable_psum_copy_in,
                    inputs_were_prequantized,
                )
                tile_iter_idx += 1
    elif loop_order == 'mnk' or loop_order == 'MNK':
        tile_iter_idx = 0
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
                    global_block_tile_k_idx,
                    psum_dtype,
                    tile_iter_idx,
                    enable_psum_copy_in,
                    inputs_were_prequantized,
                )
                tile_iter_idx += 1
    else:
        kernel_assert(False, f"Unsupported loop_order '{loop_order}'. Supported loop orders are: 'mnk', 'nmk'.")
