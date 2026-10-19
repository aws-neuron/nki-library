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

"""PyTorch/numpy reference implementation for quantize_block_mxfp8_kernel.

Computes the expected quantized FP8 data and dense float8_e8m0fnu scales in the
layout the kernel produces, using only src-tree utilities (no neuronxcc test internals).
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np

from ..mxfp_utils.mxfp8_utils._quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    QUANT_SCHEME_1X32,
    QUANT_SCHEME_WRAPX,
    decompose_k_remainder,
)
from ..mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation

# Alternative emax values used by the kernel (--enable-mx-alternative-emax).
# Standard MX emax is 15/8; alternative is 14/7.
_ALT_MAX_EXP = {
    nl.float8_e5m2_x4: 14,
    nl.float8_e4m3fn_x4: 7,
}

# Clipping max values are the standard FP8 max representable values
# (NOT affected by alternative emax — only scale computation changes).
_FP8_MAX_VAL = {
    nl.float8_e5m2_x4: 57344.0,
    nl.float8_e4m3fn_x4: 448.0,
}

_FP8_DTYPE_MAP = {
    "float8_e5m2": nl.float8_e5m2_x4,
    "float8_e4m3fn": nl.float8_e4m3fn_x4,
}


def quantize_block_mxfp8_torch_ref(
    src_tensor: np.ndarray,
    return_fp8_dtype: str,
    run_with_lnc2: bool = False,
    quant_scheme: str = None,
    use_fast_dma_transpose: bool = False,
    src_orientation=None,
) -> dict[str, np.ndarray]:
    """Golden reference for quantize_block_mxfp8_kernel.

    Computes the expected quantized data and scales using MX block
    quantization with alternative emax, in the layout the kernel produces:
    interleaved data and dense [K // 32, F] scales.

    use_fast_dma_transpose only selects which loader the kernel uses; every loader produces
    the same layout, so the golden ignores it. src_orientation K_BY_F means src_tensor is
    [K, F]; the output is that of its [F, K] transpose.

    Returns:
        dict with "quantized_data_hbm" (FP8 ndarray) and "quantized_scales_hbm" ([K // 32, F] float8_e8m0fnu ndarray).
    """
    assert quant_scheme is not None, "quant_scheme is required"  # noqa: S101
    if src_orientation == TensorOrientation.K_BY_F:
        src_tensor = np.ascontiguousarray(src_tensor.T)
    F, K = src_tensor.shape
    non_x4_dtype = _get_non_x4_dtype(return_fp8_dtype)
    x4_dtype = _get_x4_dtype(return_fp8_dtype)

    # Interleave input, quantize, reshape to kernel output layout
    interleaved = interleave_tensor(src_tensor.T, quant_scheme)  # (K, F) -> (K//4, F*4)
    golden_data_x4, golden_scales = _quantize_mx_alt_emax(interleaved, x4_dtype)
    golden_data = golden_data_x4.view(non_x4_dtype).reshape(K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR)

    return {
        "quantized_data_hbm": golden_data,
        # View the uint8 exponents as float8_e8m0fnu so the binary representation is preserved.
        "quantized_scales_hbm": golden_scales.view(nl.float8_e8m0fnu),
    }


"""Private helpers for quantization reference implementation."""


def _get_x4_dtype(return_fp8_dtype: str):
    return _FP8_DTYPE_MAP[return_fp8_dtype]


def _get_non_x4_dtype(return_fp8_dtype: str):
    return getattr(dt, return_fp8_dtype)


def _quantize_mx_alt_emax(data: np.ndarray, x4_dtype) -> tuple[np.ndarray, np.ndarray]:
    """MX block quantization with alternative emax values.

    Same algorithm as quantize_to_mx in mx_torch_common.py but uses the
    alternative max-exponent values (14/7 instead of 15/8) that the kernel
    uses when compiled with --enable-mx-alternative-emax.

    Args:
        data: float32 array [P, F], P%8==0, F%4==0.
        x4_dtype: Target nl x4 dtype.

    Returns:
        (x4_packed [P, F//4], scale_uint8 [P//8, F//4])
    """
    data_f32 = dt.static_cast(data, np.float32) if data.dtype != np.float32 else data
    max_exp = _ALT_MAX_EXP[x4_dtype]
    max_val = _FP8_MAX_VAL[x4_dtype]

    P, F = data_f32.shape
    exp_field = ((data_f32.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
    block_max_exp = exp_field.reshape(P // 8, 8, F // 4, 4).max(axis=(1, 3))
    # Scale as int (matches neuronxcc: uint8 - int -> int64, no clip)
    scale_int = block_max_exp.astype(np.uint8).astype(np.int64) - max_exp

    scale_factors = np.power(2.0, (scale_int - 127).astype(np.float64))
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)
    clipped = np.clip(data_f32.astype(np.float64) / scale_expanded, -max_val, max_val)

    return dt.static_cast(clipped.astype(np.float32), x4_dtype), scale_int.astype(np.uint8)


def _interleave_chunk(
    src_tensor: np.ndarray, dst_tensor: np.ndarray, src_p_start: int, dst_p_start: int, chunk_size: int, F: int
) -> None:
    """Interleave a single chunk of chunk_size P-elements into dst_tensor."""
    sub_tile_p = chunk_size // INTERLEAVE_FACTOR
    for sub_tile_idx in range(INTERLEAVE_FACTOR):
        for p_idx in range(sub_tile_p):
            src_p = src_p_start + sub_tile_idx * sub_tile_p + p_idx
            for f_idx in range(F):
                dst_tensor[dst_p_start + p_idx, f_idx * INTERLEAVE_FACTOR + sub_tile_idx] = src_tensor[src_p, f_idx]


def _interleave_tensor(src: np.ndarray, TILE_P: int = 512) -> np.ndarray:
    """wrapX interleave of full TILE_P tiles plus the P % TILE_P remainder, [P, F] -> [P//4, F*4].

    The remainder is split by ``decompose_k_remainder``, as every wrapX loader and the quantize
    kernel split it.
    """
    P, F = src.shape
    dst = np.zeros((P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR), dtype=np.float32)

    NUM_FULL_TILES = P // TILE_P
    SUB_TILE_P = TILE_P // INTERLEAVE_FACTOR

    # Process full tiles
    for tile_idx in range(NUM_FULL_TILES):
        _interleave_chunk(
            src, dst, src_p_start=tile_idx * TILE_P, dst_p_start=tile_idx * SUB_TILE_P, chunk_size=TILE_P, F=F
        )

    src_offset = NUM_FULL_TILES * TILE_P
    dst_offset = NUM_FULL_TILES * SUB_TILE_P

    remainder = P % TILE_P
    for chunk_size in decompose_k_remainder(remainder):
        _interleave_chunk(src, dst, src_p_start=src_offset, dst_p_start=dst_offset, chunk_size=chunk_size, F=F)
        src_offset += chunk_size
        dst_offset += chunk_size // INTERLEAVE_FACTOR

    return dst.astype(src.dtype)


def _interleave_tensor_1x32(src: np.ndarray) -> np.ndarray:
    """1x32 interleave: [P, F] -> [P//4, F*4].

    Packs INTERLEAVE_FACTOR consecutive P values of a feature into its four adjacent
    output columns:

        dst[p, f * INTERLEAVE_FACTOR + c] = src[INTERLEAVE_FACTOR * p + c, f]

    Matches the hardware ``load_tile_PE_Swizzle_1x32`` loader. The mapping is local to each
    group of INTERLEAVE_FACTOR rows, so unlike wrapX it is independent of the kernel's
    tiling and needs no remainder handling.
    """
    P, F = src.shape
    return (
        src.reshape(P // INTERLEAVE_FACTOR, INTERLEAVE_FACTOR, F)
        .transpose(0, 2, 1)
        .reshape(P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR)
    )


def interleave_tensor(src: np.ndarray, quant_scheme: str = QUANT_SCHEME_WRAPX, TILE_P: int = 512) -> np.ndarray:
    """Interleave [P, F] -> [P//4, F*4] into the given quantization scheme's layout."""
    if quant_scheme == QUANT_SCHEME_1X32:
        return _interleave_tensor_1x32(src)
    return _interleave_tensor(src, TILE_P)
