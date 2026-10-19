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

"""Constants for MXFP8 matmul tests."""

# Re-export shared constants so existing imports keep working
from nkilib_src.nkilib.experimental.matmul_mxfp8._matmul_mxfp8_constants import (  # noqa: F401
    BYTES_PER_DTYPE,
    INTERLEAVE_FACTOR,
    MAX_BLOCK_M,
    MAX_BLOCK_N,
    SBUF_F_DIM_LIMIT_BYTES,
    SBUF_LIMIT_BYTES,
    TILE_K_DEFAULTS,
    TILE_M_DEFAULTS,
    TILE_N_DEFAULTS,
    TILE_SIZE_P_MAX_LOGICAL,
    MatrixPrecision,
)

# ---------------------------------------------------------------------------
# Test-only constants
# ---------------------------------------------------------------------------
MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE = 0.2
MXFP8_COSINE_SIMILARITY_THRESHOLD = 0.995
MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD = 0.10

DEFAULT_STRIDE = 128
