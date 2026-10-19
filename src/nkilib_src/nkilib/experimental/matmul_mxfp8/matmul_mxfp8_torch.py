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

from ...core.utils.mx_torch_common import nc_matmul_mx_golden, quantize_to_mx
from ..mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from ..quantize_mxfp8.quantize_mxfp8_torch import interleave_tensor as quantize_interleave_tensor
from .matmul_mxfp8_constants import QuantScheme

# Swizzle interleave factor (must match hardware constant)
_INTERLEAVE_FACTOR = 4


def _raw_exponent_scale(scale):
    """Return the MX scale tensor's raw biased exponent bytes as uint8.

    MX scales encode a biased exponent as one byte. ``nc_matmul_mx_golden``
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
def _swizzle(src_tensor):
    """wrapX-swizzle a [P, F] operand as the loaders do: a P % 512 remainder is split into
    256 / 128 / sub-128 tiles (decompose_k_remainder)."""
    return quantize_interleave_tensor(src_tensor, QuantScheme.WRAPX.value)


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


def _golden_matmul(lhs_swizzled, rhs_swizzled, compute_dtype_x4):
    """Quantize swizzled BF16 inputs and compute MX matmul golden."""
    a_data, a_scale = quantize_to_mx(lhs_swizzled, compute_dtype_x4)
    b_data, b_scale = quantize_to_mx(rhs_swizzled, compute_dtype_x4)
    return nc_matmul_mx_golden(a_data, b_data, a_scale, b_scale)


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
    lhs_orientation=None,
    rhs_orientation=None,
    fast_dma_transpose=False,
    quant_scheme="wrapX",
    disable_dma_transpose=False,
    psum_copy_in_ratio=None,
    strided_copy_swizzle_1x32_ratio=None,
):
    """Compute golden matmul output for MXFP8 matrix multiplication.

    Handles both BF16 inputs (swizzled or unswizzled) and pre-quantized MXFP8 inputs.
    For BF16 inputs, swizzles if needed, quantizes to MXFP8, and computes matmul.
    For pre-quantized inputs, uses the quantized data and dense [K // 32, F] scales directly.
    use_scale_packing only changes the kernel's SBUF layout, so the golden ignores it.
    Feature ratios only change scheduling and load mechanisms.

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

    # None (auto/unset) and F_BY_K both mean F-by-K (the kernel default); only an explicit
    # K_BY_F selects the K-by-F layout. This matches TensorDescriptor's resolution and the
    # kernel's `orientation=None if is_f_by_k else K_BY_F` mapping exactly.
    if not lhs_prequantized and not rhs_prequantized:
        # Both BF16: swizzle if needed, then quantize and matmul
        if lhs_is_swizzled:
            lhs_sw = lhs
        elif lhs_orientation == TensorOrientation.K_BY_F:
            # lhs is [K, M] already (K-by-F); swizzle directly
            lhs_sw = swizzle(lhs.copy())
        else:
            # lhs is [M, K] unswizzled (F-by-K, the default); transpose to [K, M] then swizzle
            lhs_sw = swizzle(lhs.T.copy())

        if rhs_is_swizzled:
            rhs_sw = rhs
        elif rhs_orientation == TensorOrientation.K_BY_F:
            # rhs is [K, N] already (K-by-F); swizzle directly
            rhs_sw = swizzle(rhs.copy())
        else:
            # rhs is [N, K] unswizzled (F-by-K, the default); transpose to [K, N] then swizzle
            rhs_sw = swizzle(rhs.T.copy())

        result = _golden_matmul(lhs_sw, rhs_sw, compute_dtype_x4)
    else:
        # At least one operand is pre-quantized.
        # For BF16 operands, swizzle and quantize. For pre-quantized, use directly.
        if not lhs_prequantized:
            if lhs_is_swizzled:
                lhs_sw = lhs
            elif lhs_orientation == TensorOrientation.K_BY_F:
                lhs_sw = swizzle(lhs.copy())
            else:
                lhs_sw = swizzle(lhs.T.copy())
            a_data, a_scale = quantize_to_mx(lhs_sw, compute_dtype_x4)
        else:
            # Pre-quantized: data may be non-x4 dtype, view as x4
            a_data = lhs
            if str(a_data.dtype) != str(compute_dtype_x4):
                a_data = a_data.view(compute_dtype_x4)
            # HBM scales are dense [K // 32, F], the layout the golden consumes.
            a_scale = _raw_exponent_scale(lhs_scales)

        if not rhs_prequantized:
            if rhs_is_swizzled:
                rhs_sw = rhs
            elif rhs_orientation == TensorOrientation.K_BY_F:
                rhs_sw = swizzle(rhs.copy())
            else:
                rhs_sw = swizzle(rhs.T.copy())
            b_data, b_scale = quantize_to_mx(rhs_sw, compute_dtype_x4)
        else:
            b_data = rhs
            if str(b_data.dtype) != str(compute_dtype_x4):
                b_data = b_data.view(compute_dtype_x4)
            # HBM scales are dense [K // 32, F], the layout the golden consumes.
            b_scale = _raw_exponent_scale(rhs_scales)

        result = nc_matmul_mx_golden(a_data, b_data, a_scale, b_scale)

    return {"out": result.astype(out_dt)}


def quantize_lhs_matmul_pipeline_torch_ref(
    lhs_bf16,
    rhs_sw,
    return_fp8_dtype=None,
    run_with_lnc2=None,
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
    quant_scheme=None,
):
    """Golden for quantize-LHS + matmul pipeline.

    LHS is BF16 [M, K] (unswizzled), RHS is BF16 swizzled.
    Swizzle LHS in the pipeline's scheme, quantize both, and compute matmul.
    """
    compute_dtype_x4 = _resolve_x4_dtype(float8_dtype or return_fp8_dtype)
    lhs_sw = _swizzle_1x32(lhs_bf16.T.copy()) if quant_scheme == "1x32" else _swizzle(lhs_bf16.T.copy())
    result = _golden_matmul(lhs_sw, rhs_sw, compute_dtype_x4)
    return {"out": result.astype(output_dtype)}


def quantize_rhs_matmul_pipeline_torch_ref(
    lhs_sw,
    rhs_bf16,
    return_fp8_dtype=None,
    run_with_lnc2=None,
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
    quant_scheme=None,
):
    """Golden for quantize-RHS + matmul pipeline.

    LHS is BF16 swizzled, RHS is BF16 [N, K] (unswizzled).
    Swizzle RHS in the pipeline's scheme, quantize both, and compute matmul.
    """
    compute_dtype_x4 = _resolve_x4_dtype(float8_dtype or return_fp8_dtype)
    rhs_sw = _swizzle_1x32(rhs_bf16.T.copy()) if quant_scheme == "1x32" else _swizzle(rhs_bf16.T.copy())
    result = _golden_matmul(lhs_sw, rhs_sw, compute_dtype_x4)
    return {"out": result.astype(output_dtype)}
