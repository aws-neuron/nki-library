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


def operand_precision(is_quantized, is_x4):
    """PRECISION_* for an operand from its quantization/packing facts."""
    if not is_quantized:
        return PRECISION_BFLOAT16
    return PRECISION_MXFP8_X4 if is_x4 else PRECISION_MXFP8
