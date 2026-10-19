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

"""Utility functions for MXFP8 matmul tests."""

import hashlib
import json

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._quantize_mxfp8_utils import QUANT_SCHEME_WRAPX
from nkilib_src.nkilib.experimental.quantize_mxfp8.quantize_mxfp8_torch import interleave_tensor

from test.integration.nkilib.experimental.matmul_mxfp8.constants import (
    BYTES_PER_DTYPE,
    INTERLEAVE_FACTOR,
    MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE,
    MXFP8_COSINE_SIMILARITY_THRESHOLD,
    MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD,
)

# ============================================================================
# Utility Functions
# ============================================================================


def swizzle_tensor(src_tensor, TILE_P=512):
    """wrapX-swizzle a [P, F] tensor as the loaders do: a P % TILE_P remainder is split into
    256 / 128 / sub-128 tiles (decompose_k_remainder)."""
    return interleave_tensor(src_tensor, QUANT_SCHEME_WRAPX, TILE_P)


def swizzle_tensor_1x32(src_tensor):
    """
    Golden reference for the 1x32 (contiguous-K) interleave layout.

    Unlike wrapX (``swizzle_tensor``), which splits a feature's K into four
    far-apart quarters, the 1x32 layout packs four *consecutive* K values of a
    feature into its four adjacent output columns:

        dst[p, f * INTERLEAVE_FACTOR + c] = src[INTERLEAVE_FACTOR * p + c, f]

    This matches the layout produced by the hardware ``load_tile_PE_Swizzle_1x32``
    loader (fp32-reinterpret PE transpose). The mapping is local to each group of
    INTERLEAVE_FACTOR rows and independent of tile boundaries, so no TILE_P
    remainder decomposition is needed.

    Args:
        src_tensor (np.ndarray): Source tensor of shape (P, F).

    Returns:
        np.ndarray: Swizzled tensor of shape (P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR).
    """
    P, F = src_tensor.shape

    if P % INTERLEAVE_FACTOR != 0:
        raise ValueError(f"P ({P}) must be divisible by INTERLEAVE_FACTOR ({INTERLEAVE_FACTOR})")

    return (
        src_tensor.reshape(P // INTERLEAVE_FACTOR, INTERLEAVE_FACTOR, F)
        .transpose(0, 2, 1)
        .reshape(P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR)
    )


def unswizzle_tensor(src_tensor, TILE_P=512):
    """
    Golden reference for reversing interleave loading.

    Handles DGT-compatible remainder decomposition (256 and/or 128 sub-tiles)
    to match swizzle_tensor.

    Args:
        src_tensor (np.ndarray): Swizzled tensor of shape (P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR).
        TILE_P (int): Tile size in P dimension.

    Returns:
        np.ndarray: Unswizzled tensor of shape (P, F).
    """
    P_div_INTERLEAVE_FACTOR, F_times_INTERLEAVE_FACTOR = src_tensor.shape
    P = P_div_INTERLEAVE_FACTOR * INTERLEAVE_FACTOR
    F = F_times_INTERLEAVE_FACTOR // INTERLEAVE_FACTOR

    remainder = P % TILE_P
    if remainder != 0 and remainder % 128 == 0:
        full_p = P - remainder
        full_p_phys = full_p // INTERLEAVE_FACTOR
        parts = []
        if full_p > 0:
            parts.append(unswizzle_tensor(src_tensor[:full_p_phys, :], TILE_P))
        k_off_phys = full_p_phys
        rem = remainder
        if rem >= 256:
            chunk_phys = 256 // INTERLEAVE_FACTOR
            parts.append(unswizzle_tensor(src_tensor[k_off_phys : k_off_phys + chunk_phys, :], TILE_P=256))
            k_off_phys += chunk_phys
            rem -= 256
        if rem >= 128:
            chunk_phys = 128 // INTERLEAVE_FACTOR
            parts.append(unswizzle_tensor(src_tensor[k_off_phys : k_off_phys + chunk_phys, :], TILE_P=128))
        return np.concatenate(parts, axis=0)

    dst_tensor = np.zeros((P, F), dtype=src_tensor.dtype)

    NUM_TILES_P = (P + TILE_P - 1) // TILE_P  # Ceiling division to include partial tiles
    SUB_TILE_P = TILE_P // INTERLEAVE_FACTOR

    for tp in range(NUM_TILES_P):
        # Calculate actual tile size (handles partial last tile)
        current_tile_size = min(TILE_P, P - tp * TILE_P)
        current_sub_tile_p = current_tile_size // INTERLEAVE_FACTOR

        for sub_tp in range(INTERLEAVE_FACTOR):
            for p in range(current_sub_tile_p):
                dst_p = tp * TILE_P + sub_tp * current_sub_tile_p + p
                if dst_p < P:  # Safety check
                    for f in range(F):
                        src_p = tp * SUB_TILE_P + p
                        dst_tensor[dst_p, f] = src_tensor[src_p, f * INTERLEAVE_FACTOR + sub_tp]

    return dst_tensor


def cosine_sim(a, b):
    """
    Computes cosine similarity between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        float: Cosine similarity value.
    """
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return dot_product / (norm_a * norm_b)


def check_correctness(kernel_result, golden_result, rtol=1e-3):
    """
    Compare two tensors or arrays by flattening them.

    Args:
        kernel_result (np.ndarray): Kernel output array.
        golden_result (np.ndarray): Golden reference array.
        rtol (float): Relative tolerance for comparison.

    Returns:
        tuple: (passed boolean, metrics dict).
    """
    if type(kernel_result) is not type(golden_result):
        raise TypeError(f"Inputs must be of the same type. {type(kernel_result)=} && {type(golden_result)=}")

    kernel_result_flat = kernel_result.flatten().astype(np.float32)
    golden_result_flat = golden_result.flatten().astype(np.float32)
    cos_sim = cosine_sim(kernel_result_flat, golden_result_flat)
    euclid_dist = np.linalg.norm(kernel_result_flat - golden_result_flat) / (
        np.linalg.norm(kernel_result_flat) + np.linalg.norm(golden_result_flat) + 1e-12
    )

    is_close = np.allclose(
        kernel_result_flat,
        golden_result_flat,
        atol=np.abs(golden_result_flat).max() * MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE,
        rtol=rtol,
    )
    cosine_sim_passed = cos_sim >= MXFP8_COSINE_SIMILARITY_THRESHOLD
    norm_euclidean_passed = euclid_dist <= MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD
    passed = is_close and cosine_sim_passed and norm_euclidean_passed

    return passed, {
        "cosine_similarity": float(cos_sim),
        "normalized_euclidean_distance": euclid_dist,
        "all_close": is_close,
    }


def get_size(dtype: str) -> int:
    """
    Get byte size for a given matrix precision type.

    Args:
        dtype (str): Matrix precision string (e.g. 'mxfp8', 'bfloat16', 'fp32').

    Returns:
        int: Number of bytes per element.
    """
    if dtype in BYTES_PER_DTYPE:
        return BYTES_PER_DTYPE[dtype]
    else:
        raise ValueError(f"dtype {dtype} not in BYTES_PER_DTYPE: {BYTES_PER_DTYPE}")


def dict_hash(dictionary) -> str:
    """Create a unique hash for a dictionary"""
    # Convert dict to JSON string with sorted keys for consistency
    dict_string = json.dumps(dictionary, sort_keys=True)
    # Create hash
    return hashlib.md5(dict_string.encode()).hexdigest()


def to_hbm_scales(compact_scales):
    """
    Lay out golden MX scales the way a pre-quantized operand carries them on HBM.

    HBM scales are dense -- one row per 8 data rows, [K // 32, F] -- which is the compact
    layout quantize_mx_golden already returns, so only the dtype changes: the uint8 exponent
    bytes are viewed as float8_e8m0fnu, the dtype the quantize kernel writes.
    """
    return compact_scales.view(nl.float8_e8m0fnu)
