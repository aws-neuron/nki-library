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


class QuantScheme(Enum):
    """Quantization scheme for MXFP8 PE swizzle layout.

    Defined here (NKI-free) so both the kernel-side TensorDescriptor and the
    NKI-free autotune-key logic in this module can compare the same enum members.
    Re-exported from common_dataclasses for existing importers.
    """

    WRAPX = "wrapX"
    _1x32 = "1x32"


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


def operand_precision(is_quantized, is_x4):
    """PRECISION_* for an operand from its quantization/packing facts."""
    if not is_quantized:
        return PRECISION_BFLOAT16
    return PRECISION_MXFP8_X4 if is_x4 else PRECISION_MXFP8


def operand_load_method(is_swizzled, is_quantized, load_with_PE_swizzle, fast_dma_transpose, quant_scheme):
    """Per-operand load-method token for the autotune cache key.

    Each operand is loaded through exactly one path (load_apis.load_tile dispatches
    on these same TensorDescriptor fields), and the optimal tiling depends on which
    path each operand takes -- so the key encodes the LHS and RHS methods
    independently. The six tokens are:

      - "prequant": pre-quantized MXFP8 (data + scales), swizzled in HBM. dma_copy.
      - "swizzled": pre-swizzled BF16. dma_copy, no transpose.
      - "pe_1x32":  unswizzled BF16, PE-swizzle transpose, 1x32 quant scheme.
      - "pe_wrapx": unswizzled BF16, PE-swizzle transpose, wrapX quant scheme.
      - "dgt_fast": unswizzled BF16, DMA gather-transpose, fast_dma_transpose 4D path.
      - "dgt":      unswizzled BF16, DMA gather-transpose, default path.

    fast_dma_transpose splits "dgt" from "dgt_fast" because the two use different
    access patterns and K alignment (32 vs 128), so their best tilings differ even
    though load_tile routes both through load_tile_dgt.

    quant_scheme is compared as a QuantScheme enum member (no .value / str ops) so
    it stays resolvable inside the NKI-traced kernel.
    """
    if is_quantized:
        return "prequant"
    if is_swizzled:
        return "swizzled"
    if load_with_PE_swizzle:
        if quant_scheme == QuantScheme._1x32:
            return "pe_1x32"
        return "pe_wrapx"
    if fast_dma_transpose:
        return "dgt_fast"
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
    lhs_load_with_PE_swizzle,
    rhs_load_with_PE_swizzle,
    quant_scheme,
    run_with_lnc2,
    lnc_2_shard_rhs,
    lhs_fast_dma_transpose=False,
    rhs_fast_dma_transpose=False,
    disable_dma_transpose=False,
):
    """Build the autotune cache key: {eM}x{K}x{eN}_{lhs_method}_{rhs_method}.

    Each operand contributes its own per-operand load-method token (see
    operand_load_method), so the key distinguishes both the input-preparation path
    on each side and mixed combinations (e.g. an LHS on the PE-1x32 path with a
    pre-quantized RHS). The method token subsumes the operand dtype -- "prequant"
    is MXFP8, every other token is BF16.

    The PE-swizzle fact is taken per operand (lhs_load_with_PE_swizzle /
    rhs_load_with_PE_swizzle): each descriptor owns whether it is PE-transposed, so
    a mixed PE-LHS / DGT-RHS pair keys as pe_*/dgt rather than being OR-reduced to
    pe_*/pe_*. Passing the same flag for both operands reproduces the legacy key.

    (eM, eN) are the per-core dims after LNC2 sharding (see effective_shard_dims),
    so a shape run with LNC2 looks up the tiling tuned for the shape each core
    actually computes. Takes plain scalar fields (not a config object) so both the
    kernel and the offline updater call it directly with the values they have.
    """
    eff_m, eff_n = effective_shard_dims(M, N, run_with_lnc2, lnc_2_shard_rhs)
    lhs_quantized = operand_dtype_key(lhs_dtype) == "mxfp8"
    rhs_quantized = operand_dtype_key(rhs_dtype) == "mxfp8"
    lhs_key = operand_load_method(
        lhs_is_swizzled, lhs_quantized, lhs_load_with_PE_swizzle, lhs_fast_dma_transpose, quant_scheme
    )
    rhs_key = operand_load_method(
        rhs_is_swizzled, rhs_quantized, rhs_load_with_PE_swizzle, rhs_fast_dma_transpose, quant_scheme
    )
    # No-DMA-transpose (1x32 K-by-F loaded via dma_copy + nc_transpose instead of a DMA
    # transpose) is a distinct performance regime, so it gets its own cache entry. The
    # suffix is only appended when the flag is set, so every existing (DMA) key is byte
    # identical -- no re-keying of the current cache.
    suffix = "_nodma" if disable_dma_transpose else ""
    return f"{eff_m}x{K}x{eff_n}_{lhs_key}_{rhs_key}{suffix}"
