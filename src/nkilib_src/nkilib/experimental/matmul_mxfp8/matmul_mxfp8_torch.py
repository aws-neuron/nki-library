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

"""Torch reference implementations for matmul_mxfp8 and pipeline kernels.

Computes golden outputs using CPU-based MX quantization and matmul for
validation against NKI kernel outputs.
"""

import neuron_dtypes as dtype
import numpy as np
from neuronxcc.nki._private.private_api import float8_e4m3fn_x4, float8_e5m2_x4
from neuronxcc.nki._private.test import mx_util

# Swizzle interleave factor (must match hardware constant)
_INTERLEAVE_FACTOR = 4


def _get_mx_max_exp(dst_dtype):
    """Get maximum exponent for MX format, adjusted for TRN3 rounding mode."""
    return {float8_e5m2_x4: 14, float8_e4m3fn_x4: 7}[dst_dtype]


def _raw_exponent_scale(scale):
    """Return the MX scale tensor's raw biased exponent bytes as uint8.

    MX scales encode a biased exponent as one byte. ``mx_util.nc_matmul_mx_golden``
    expects the integer byte value (0-255) via ``.astype``. When the pre-quantized
    scale input is float8_e8m0fnu, ``.astype`` would convert the represented value
    (2^(byte-127)) rather than the byte, corrupting the golden scale factors.
    Reinterpret the bytes via ``.view(uint8)``; no-op for scales already uint8.
    """
    import nki.language as nl

    if scale.dtype == nl.float8_e8m0fnu:
        return scale.view(np.uint8)
    return scale


# NOTE: This duplicates test/integration/.../matmul_mxfp8/utils.py:swizzle_tensor
# intentionally — src/ modules must not import from test/.
def _swizzle(src_tensor, TILE_P=512):
    """
    Golden reference for interleave loading.

    When P is not divisible by TILE_P, the remainder is decomposed into 256
    and/or 128 sub-tiles to match the DGT hardware's remainder handling in
    load_block.py. Without this, the interleaving pattern would differ from
    what DGT produces, causing mismatched quantization groups in matmuls
    where one operand is pre-swizzled and the other is DGT-loaded.

    Args:
        src_tensor (np.ndarray): Source tensor of shape (P, F).
        TILE_P (int): Tile size in P dimension.

    Returns:
        np.ndarray: Swizzled tensor of shape (P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR).
    """
    P, F = src_tensor.shape

    # Validate that P is divisible by INTERLEAVE_FACTOR
    if P % _INTERLEAVE_FACTOR != 0:
        raise ValueError(f"P ({P}) must be divisible by INTERLEAVE_FACTOR ({_INTERLEAVE_FACTOR})")

    remainder = P % TILE_P
    if remainder != 0 and remainder % 128 == 0:
        # Decompose into full tiles + DGT-compatible remainder sub-tiles (256 and/or 128)
        full_p = P - remainder
        parts = []
        if full_p > 0:
            parts.append(_swizzle(src_tensor[:full_p, :], TILE_P))
        k_off = full_p
        rem = remainder
        if rem >= 256:
            parts.append(_swizzle(src_tensor[k_off : k_off + 256, :], TILE_P=256))
            k_off += 256
            rem -= 256
        if rem >= 128:
            parts.append(_swizzle(src_tensor[k_off : k_off + 128, :], TILE_P=128))
        return np.concatenate(parts, axis=0)

    dst_tensor = np.zeros((P // _INTERLEAVE_FACTOR, F * _INTERLEAVE_FACTOR), dtype=np.float32)

    NUM_TILES_P = (P + TILE_P - 1) // TILE_P  # Ceiling division to include partial tiles
    SUB_TILE_P = TILE_P // _INTERLEAVE_FACTOR

    for tp in range(NUM_TILES_P):
        # Calculate actual tile size (handles partial last tile)
        current_tile_size = min(TILE_P, P - tp * TILE_P)
        current_sub_tile_p = current_tile_size // _INTERLEAVE_FACTOR

        for sub_tp in range(_INTERLEAVE_FACTOR):
            for p in range(current_sub_tile_p):
                src_p = tp * TILE_P + sub_tp * current_sub_tile_p + p
                if src_p < P:  # Safety check
                    for f in range(F):
                        dst_p = tp * SUB_TILE_P + p
                        dst_tensor[dst_p, f * _INTERLEAVE_FACTOR + sub_tp] = src_tensor[src_p, f]

    return dst_tensor.astype(src_tensor.dtype)


# NOTE: This duplicates test/integration/.../matmul_mxfp8/utils.py:swizzle_tensor_1x32
# intentionally — src/ modules must not import from test/.
def _swizzle_1x32(src_tensor):
    """
    Golden reference for the 1x32 (contiguous-K) interleave layout.

    Packs INTERLEAVE_FACTOR consecutive K values of a feature into its four
    adjacent output columns (contrast with wrapX ``_swizzle``, which scatters
    K into four quarters):

        dst[p, f * INTERLEAVE_FACTOR + c] = src[INTERLEAVE_FACTOR * p + c, f]

    Matches the hardware ``load_tile_PE_Swizzle_1x32`` loader. The mapping is
    local to each group of INTERLEAVE_FACTOR rows, so no tile-remainder handling
    is needed.

    Args:
        src_tensor (np.ndarray): Source tensor of shape (P, F).

    Returns:
        np.ndarray: Swizzled tensor of shape (P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR).
    """
    P, F = src_tensor.shape
    if P % _INTERLEAVE_FACTOR != 0:
        raise ValueError(f"P ({P}) must be divisible by INTERLEAVE_FACTOR ({_INTERLEAVE_FACTOR})")
    return (
        src_tensor.reshape(P // _INTERLEAVE_FACTOR, _INTERLEAVE_FACTOR, F)
        .transpose(0, 2, 1)
        .reshape(P // _INTERLEAVE_FACTOR, F * _INTERLEAVE_FACTOR)
    )


def _resolve_x4_dtype(float8_dtype_str):
    """Convert float8 dtype string to neuron x4 dtype."""
    name = float8_dtype_str if float8_dtype_str.endswith("_x4") else float8_dtype_str + "_x4"
    return getattr(dtype, name)


def golden_matmul(lhs_swizzled, rhs_swizzled, compute_dtype_x4):
    """Quantize swizzled BF16 inputs and compute MX matmul golden."""
    a_data, a_scale = mx_util.quantize_mx_golden(lhs_swizzled, compute_dtype_x4, custom_mx_max_exp=_get_mx_max_exp)
    b_data, b_scale = mx_util.quantize_mx_golden(rhs_swizzled, compute_dtype_x4, custom_mx_max_exp=_get_mx_max_exp)
    return mx_util.nc_matmul_mx_golden(a_data, b_data, a_scale, b_scale)


def matmul_mxfp8_torch_ref(
    lhs,
    rhs,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    block_loop_order='mnk',
    tile_loop_order='mnk',
    float8_dtype="float8_e5m2",
    output_dtype=None,
    run_with_lnc2=True,
    lnc_2_shard_rhs=True,
    lhs_scales=None,
    rhs_scales=None,
    use_scale_packing=False,
    spill_reload=False,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    load_with_PE_swizzle=False,
    lhs_is_f_by_k=True,
    rhs_is_f_by_k=True,
    fast_dma_transpose=False,
    enable_psum_copy_in=None,
    quant_scheme="wrapX",
):
    """Compute golden matmul output for MXFP8 matrix multiplication.

    Handles both BF16 inputs (swizzled or unswizzled) and pre-quantized MXFP8 inputs.
    For BF16 inputs, swizzles if needed, quantizes to MXFP8, and computes matmul.
    For pre-quantized inputs, uses the quantized data and scales directly.

    Returns:
        dict: {"out": np.ndarray} with the golden matmul result.
    """
    import nki.language as nl

    compute_dtype_x4 = _resolve_x4_dtype(float8_dtype)
    out_dt = output_dtype if output_dtype is not None else nl.float32

    # Both operands' K axis must use the same permutation for the matmul to be
    # valid, so a single quant_scheme selects the swizzle for every BF16 operand.
    swizzle = _swizzle_1x32 if quant_scheme == "1x32" else _swizzle

    lhs_prequantized = lhs_scales is not None
    rhs_prequantized = rhs_scales is not None

    # Normalize the layout flags: None (auto/unset) means F-by-K, matching the kernel default.
    # Doing this once lets the branches below use the simpler `not lhs_is_f_by_k` form while
    # still treating an unset (None) flag as F-by-K rather than K-by-F.
    lhs_is_f_by_k = lhs_is_f_by_k is not False
    rhs_is_f_by_k = rhs_is_f_by_k is not False

    if not lhs_prequantized and not rhs_prequantized:
        # Both BF16: swizzle if needed, then quantize and matmul
        if lhs_is_swizzled:
            lhs_sw = lhs
        elif not lhs_is_f_by_k:
            # lhs is [K, M] already (K-by-F); swizzle directly
            lhs_sw = swizzle(lhs.copy())
        else:
            # lhs is [M, K] unswizzled (F-by-K, the default); transpose to [K, M] then swizzle
            lhs_sw = swizzle(lhs.T.copy())

        if rhs_is_swizzled:
            rhs_sw = rhs
        elif not rhs_is_f_by_k:
            # rhs is [K, N] already (K-by-F); swizzle directly
            rhs_sw = swizzle(rhs.copy())
        else:
            # rhs is [N, K] unswizzled (F-by-K, the default); transpose to [K, N] then swizzle
            rhs_sw = swizzle(rhs.T.copy())

        result = golden_matmul(lhs_sw, rhs_sw, compute_dtype_x4)
    else:
        # At least one operand is pre-quantized.
        # For BF16 operands, swizzle and quantize. For pre-quantized, use directly.
        if not lhs_prequantized:
            lhs_sw = lhs if lhs_is_swizzled else (swizzle(lhs.copy()) if not lhs_is_f_by_k else swizzle(lhs.T.copy()))
            a_data, a_scale = mx_util.quantize_mx_golden(lhs_sw, compute_dtype_x4, custom_mx_max_exp=_get_mx_max_exp)
        else:
            # Pre-quantized: data may be non-x4 dtype, view as x4
            a_data = lhs
            if str(a_data.dtype) != str(compute_dtype_x4):
                a_data = a_data.view(compute_dtype_x4)
            K = a_data.shape[0] * _INTERLEAVE_FACTOR
            F = lhs_scales.shape[1]
            if use_scale_packing:
                a_scale = _unpack_packed_scales(lhs_scales, K, F)
            else:
                a_scale = _compact_scales(lhs_scales)
            a_scale = _raw_exponent_scale(a_scale)

        if not rhs_prequantized:
            rhs_sw = rhs if rhs_is_swizzled else (swizzle(rhs.copy()) if not rhs_is_f_by_k else swizzle(rhs.T.copy()))
            b_data, b_scale = mx_util.quantize_mx_golden(rhs_sw, compute_dtype_x4, custom_mx_max_exp=_get_mx_max_exp)
        else:
            b_data = rhs
            if str(b_data.dtype) != str(compute_dtype_x4):
                b_data = b_data.view(compute_dtype_x4)
            K = b_data.shape[0] * _INTERLEAVE_FACTOR
            F = rhs_scales.shape[1]
            if use_scale_packing:
                b_scale = _unpack_packed_scales(rhs_scales, K, F)
            else:
                b_scale = _compact_scales(rhs_scales)
            b_scale = _raw_exponent_scale(b_scale)

        result = mx_util.nc_matmul_mx_golden(a_data, b_data, a_scale, b_scale)

    return {"out": result.astype(out_dt)}


def _compact_scales(oversized_scales):
    """Extract compact scales [P//8, F//4] from oversized layout [P, F//4]."""
    P, F_div4 = oversized_scales.shape
    compact_rows = P // 8

    compact = np.zeros((compact_rows, F_div4), dtype=oversized_scales.dtype)
    for idx in range(compact_rows):
        hbm_row = (idx // 4) * 32 + (idx % 4)
        if hbm_row < P:
            compact[idx, :] = oversized_scales[hbm_row, :]
    return compact


def _unpack_packed_scales(packed_scales, K, F):
    """Extract compact scales [K//4//8, F] from packed layout.

    Reverses _pack_scales (from quantize_mxfp8_torch.py) by reading scale values
    from their packed positions back into dense row order.
    """
    from ..mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import (
        INTERLEAVE_FACTOR,
        Q_TILE_K,
        get_remainder_partition_offset,
        get_scale_packing_info,
    )

    L_TILE_K = 512
    NUM_TILES_IN_K = K // L_TILE_K
    REMAINDER_K = K % L_TILE_K
    HAS_REMAINDER_256 = REMAINDER_K >= 256
    HAS_REMAINDER_128 = REMAINDER_K % 256 >= 128

    total_compact_rows = K // INTERLEAVE_FACTOR // 8
    compact = np.zeros((total_compact_rows, F), dtype=packed_scales.dtype)

    golden_offset = 0
    tile_idx = 0

    def _unpack_tile(tile_k_size, golden_offset, tile_idx, k_idx_within_tile=0):
        scaling_group_idx, _, slot_partition_offset = get_scale_packing_info(tile_idx, True)
        remainder_partition_offset = get_remainder_partition_offset(k_idx_within_tile, Q_TILE_K)
        scale_p_start = scaling_group_idx * Q_TILE_K
        num_rows = (tile_k_size // INTERLEAVE_FACTOR) // 8
        for row_idx in range(num_rows):
            packed_row = (row_idx // 4) * 32 + (row_idx % 4)
            compact[golden_offset + row_idx, :] = packed_scales[
                scale_p_start + slot_partition_offset + packed_row + remainder_partition_offset, :
            ]
        return golden_offset + num_rows

    for i in range(NUM_TILES_IN_K):
        golden_offset = _unpack_tile(L_TILE_K, golden_offset, tile_idx)
        tile_idx += 1

    if HAS_REMAINDER_256:
        golden_offset = _unpack_tile(256, golden_offset, tile_idx)

    if HAS_REMAINDER_128:
        k_idx_within_tile = 256 if HAS_REMAINDER_256 else 0
        golden_offset = _unpack_tile(128, golden_offset, tile_idx, k_idx_within_tile)

    # Handle small remainders (< 128 logical elements) that still have scale data
    remaining = REMAINDER_K - (256 if HAS_REMAINDER_256 else 0) - (128 if HAS_REMAINDER_128 else 0)
    if remaining > 0 and remaining % INTERLEAVE_FACTOR == 0:
        k_idx_within_tile = REMAINDER_K - remaining
        golden_offset = _unpack_tile(remaining, golden_offset, tile_idx, k_idx_within_tile)

    return compact


def quantize_lhs_matmul_pipeline_torch_ref(
    lhs_bf16,
    rhs_sw,
    return_fp8_dtype=None,
    run_with_lnc2=None,
    enable_scale_packing=None,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    block_loop_order=None,
    tile_loop_order=None,
    output_dtype=None,
    float8_dtype=None,
    use_scale_packing=None,
    spill_reload=None,
):
    """Golden for quantize-LHS + matmul pipeline.

    LHS is BF16 [M, K] (unswizzled), RHS is BF16 swizzled.
    Swizzle LHS, quantize both, and compute matmul.
    """
    compute_dtype_x4 = _resolve_x4_dtype(float8_dtype or return_fp8_dtype)
    lhs_sw = _swizzle(lhs_bf16.T.copy())
    result = golden_matmul(lhs_sw, rhs_sw, compute_dtype_x4)
    return {"out": result.astype(output_dtype)}


def quantize_rhs_matmul_pipeline_torch_ref(
    lhs_sw,
    rhs_bf16,
    return_fp8_dtype=None,
    run_with_lnc2=None,
    enable_scale_packing=None,
    TILES_IN_BLOCK_M=None,
    TILES_IN_BLOCK_N=None,
    TILES_IN_BLOCK_K=None,
    TILES_IN_LOAD_M=None,
    TILES_IN_LOAD_N=None,
    lhs_matmul_tile_shape_logical=None,
    rhs_matmul_tile_shape_logical=None,
    block_loop_order=None,
    tile_loop_order=None,
    output_dtype=None,
    float8_dtype=None,
    use_scale_packing=None,
    spill_reload=None,
):
    """Golden for quantize-RHS + matmul pipeline.

    LHS is BF16 swizzled, RHS is BF16 [N, K] (unswizzled).
    Swizzle RHS, quantize both, and compute matmul.
    """
    compute_dtype_x4 = _resolve_x4_dtype(float8_dtype or return_fp8_dtype)
    rhs_sw = _swizzle(rhs_bf16.T.copy())
    result = golden_matmul(lhs_sw, rhs_sw, compute_dtype_x4)
    return {"out": result.astype(output_dtype)}
