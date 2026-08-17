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

"""Shared constants for MXFP8 matmul kernel and tests."""

from enum import Enum

# ---------------------------------------------------------------------------
# Precision string constants (NKI-compatible, used in kernel code)
# ---------------------------------------------------------------------------
PRECISION_MXFP8 = "mxfp8"
PRECISION_MXFP8_X4 = "mxfp8_x4"
PRECISION_BFLOAT16 = "bfloat16"
PRECISION_FP32 = "fp32"


# ---------------------------------------------------------------------------
# MatrixPrecision enum (used in test infrastructure)
# ---------------------------------------------------------------------------
class MatrixPrecision(str, Enum):
    MXFP8 = PRECISION_MXFP8
    MXFP8_X4 = PRECISION_MXFP8_X4
    BFLOAT16 = PRECISION_BFLOAT16
    FP32 = PRECISION_FP32


# ---------------------------------------------------------------------------
# Hardware / tile constants
# ---------------------------------------------------------------------------
TILE_SIZE_P_MAX_LOGICAL = 512
INTERLEAVE_FACTOR = 4

# ---------------------------------------------------------------------------
# Dtype byte sizes (keyed by both PRECISION_* constants and MatrixPrecision enum)
# ---------------------------------------------------------------------------
BYTES_PER_DTYPE = {
    PRECISION_MXFP8: 1,
    PRECISION_MXFP8_X4: 1,
    PRECISION_BFLOAT16: 2,
    PRECISION_FP32: 4,
    MatrixPrecision.MXFP8: 1,
    MatrixPrecision.MXFP8_X4: 1,
    MatrixPrecision.BFLOAT16: 2,
    MatrixPrecision.FP32: 4,
}

# ---------------------------------------------------------------------------
# SBUF / blocking limits
# ---------------------------------------------------------------------------
SBUF_LIMIT_BYTES = 32 * 1024 * 1024  # 32 MB
SBUF_F_DIM_LIMIT_BYTES = 256 * 1024  # 256 KB
MAX_BLOCK_M = 2048
MAX_BLOCK_N = 2048

# ---------------------------------------------------------------------------
# Default tile sizes for auto-generation
# ---------------------------------------------------------------------------
TILE_M_DEFAULTS = [128]
TILE_K_DEFAULTS = [512, 256, 128]
TILE_N_DEFAULTS = [2048, 1024, 512]


# ---------------------------------------------------------------------------
# Autotune cache key construction
# ---------------------------------------------------------------------------
# Single source of truth for the cache key format, shared by the kernel
# (matmul_mxfp8_config.auto_generate_default, the read path) and the offline
# cache updater (autotune_update_cache.py, the write path). Kept here because
# this module is NKI-free and importable standalone; the config module pulls in
# nki and cannot be imported outside the kernel runtime.


def operand_dtype_key(dtype):
    """Canonical operand dtype token for the autotune cache key ('mxfp8' or 'bf16')."""
    return "mxfp8" if dtype in (PRECISION_MXFP8, PRECISION_MXFP8_X4) else "bf16"


def load_method_key(lhs_is_swizzled, rhs_is_swizzled, load_with_PE_swizzle, quant_scheme):
    """Canonical load-method token for the autotune cache key.

    Distinguishes the input-preparation path, which the tuned tiling depends on:
      - "swizzled": both operands pre-swizzled (no on-chip transpose).
      - "dgt": an unswizzled BF16 operand loaded via DMA gather-transpose.
      - "pe_<scheme>": an unswizzled operand loaded via PE-swizzle transpose,
        where <scheme> is the quant scheme ("1x32" or "wrapx").

    Uses explicit string comparisons (no str methods) so it stays resolvable
    when this runs inside the NKI-traced kernel.
    """
    if lhs_is_swizzled and rhs_is_swizzled:
        return "swizzled"
    if load_with_PE_swizzle:
        if quant_scheme == "1x32":
            return "pe_1x32"
        return "pe_wrapx"
    return "dgt"


def effective_shard_dims(M, N, run_with_lnc2, lnc_2_shard_rhs):
    """Per-core (M, N) after LNC2 sharding.

    The autotune cache is keyed by the shape each core actually computes, since
    the optimal tiling is a function of the per-core matmul. With LNC2 the larger
    dim is halved (N when lnc_2_shard_rhs, else M); without LNC2 the core sees the
    full shape. K is never sharded.
    """
    if run_with_lnc2:
        if lnc_2_shard_rhs:
            return M, N // 2
        return M // 2, N
    return M, N


def autotune_cache_key(
    M,
    K,
    N,
    lhs_dtype,
    rhs_dtype,
    lhs_is_swizzled,
    rhs_is_swizzled,
    load_with_PE_swizzle,
    quant_scheme,
    run_with_lnc2,
    lnc_2_shard_rhs,
):
    """Build the autotune cache key: {eM}x{K}x{eN}_{lhs}_{rhs}_{loadmethod}.

    (eM, eN) are the per-core dims after LNC2 sharding (see effective_shard_dims),
    so a shape run with LNC2 looks up the tiling tuned for the shape each core
    actually computes. Takes plain scalar fields (not a config object) so both the
    kernel and the offline updater call it directly with the values they have.
    """
    eff_m, eff_n = effective_shard_dims(M, N, run_with_lnc2, lnc_2_shard_rhs)
    lhs_key = operand_dtype_key(lhs_dtype)
    rhs_key = operand_dtype_key(rhs_dtype)
    load_key = load_method_key(lhs_is_swizzled, rhs_is_swizzled, load_with_PE_swizzle, quant_scheme)
    return f"{eff_m}x{K}x{eff_n}_{lhs_key}_{rhs_key}_{load_key}"
