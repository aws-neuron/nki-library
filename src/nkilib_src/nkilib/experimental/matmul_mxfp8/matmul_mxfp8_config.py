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

"""Configuration dataclasses for MXFP8 matmul kernel."""

import random
from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import nki
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ..mxfp_utils.mxfp8_utils import quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils.common_dataclasses import (
    BlockDescriptor,
    LncShardingMode,
    LoopOrder,
    QuantScheme,
    TensorDescriptor,
    TensorOrientation,
)
from .matmul_mxfp8_constants import (
    BYTES_PER_DTYPE,
    INTERLEAVE_FACTOR,
    MAX_BLOCK_M,
    MAX_BLOCK_N,
    PRECISION_BFLOAT16,
    PRECISION_FP32,
    PRECISION_MXFP8,
    PRECISION_MXFP8_X4,
    SBUF_F_DIM_LIMIT_BYTES,
    SBUF_LIMIT_BYTES,
    TILE_K_DEFAULTS,
    TILE_M_DEFAULTS,
    TILE_N_DEFAULTS,
    TILE_SIZE_P_MAX_LOGICAL,
    autotune_cache_key,
    effective_shard_dims,
    operand_precision,
)

# Autotune cache: optimal configs discovered by sweep for known shapes.
# Key format: "{eM}x{K}x{eN}_{lhs_method}_{rhs_method}" -- per-core dims after LNC2
# sharding, then a per-operand load-method token for each side. Tokens (see
# operand_load_method): 'prequant', 'swizzled', 'pe_1x32', 'pe_wrapx', 'dgt_fast',
# 'dgt'. The token subsumes the dtype ('prequant' is MXFP8, all others BF16).
#
# NOTE: several "2048x4096x2304_*" QKV entries use a hand-tuned non-power-of-two tile_n
# (dgt_fast: 384; pe_1x32_pe_1x32: 576) that the sweep's power-of-two tile_n options do not
# generate. They were hand-tuned and verified on B1: 384/576 divide N=2304 evenly, so the
# output write is uniform (no thin remainder block) -- pe_1x32 tile_n=512 left a 256-wide
# remainder whose small write-out extended the tail. A plain re-sweep would propose tile_n=256
# instead, so preserve these unless you re-tune the non-power-of-two tile_n by hand.
# fmt: off
_AUTOTUNE_CACHE = {
    "1024x1024x1792_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 7, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 7},
    "1024x1024x1792_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 7, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 7},
    "1024x1536x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 3, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "1024x2048x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3},
    "1024x2048x1536_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3},
    "1024x2048x512_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 1},
    "1024x2048x512_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1},
    "1024x3072x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "1024x4096x1536_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "1024x4096x1536_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': True},
    "1024x4096x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "1024x4096x1536_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "1024x4096x768_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 768, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 1, 'spill_reload': True},
    "1024x4096x768_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': True},
    "1024x4096x768_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "1024x4096x768_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 768, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': True},
    "1024x512x1280_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1280, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1},
    "1024x512x1280_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5},
    "1024x768x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "1152x4096x4096_dgt_fast_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "1152x768x1088_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 128, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 9},
    "1152x768x1088_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 128, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 9},
    "1440x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "1440x4096x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "1536x1024x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "1536x1920x1024_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 12, 'TILES_IN_LOAD_N': 2},
    "1536x1920x1024_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2},
    "1536x2048x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "1536x3200x1792_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1792, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 7, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1},
    "1536x3200x1792_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1792, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1},
    "1536x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 12, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "1536x4096x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2},
    "1536x512x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 6, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 6, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "1664x3200x1664_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1664, 'TILES_IN_BLOCK_M': 13, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 7, 'TILES_IN_LOAD_M': 13, 'TILES_IN_LOAD_N': 1},
    "1664x3200x1664_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1664, 'TILES_IN_BLOCK_M': 13, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 13, 'TILES_IN_LOAD_N': 1},
    "1664x3456x896_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 896, 'TILES_IN_BLOCK_M': 13, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 7, 'TILES_IN_LOAD_M': 13, 'TILES_IN_LOAD_N': 1},
    "1664x3456x896_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 896, 'TILES_IN_BLOCK_M': 13, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 13, 'TILES_IN_LOAD_N': 1},
    "2048x1024x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x1024x3072_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 6, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x1024x768_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x1536x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 3, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x2048x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x2048x2880_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x2048x2880_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x2048x3072_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x2048x4096_dgt_fast_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 8, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x2048x4096_pe_1x32_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "2048x2048x768_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x2304x4096_dgt_fast_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "2048x2304x4096_pe_1x32_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "2048x2560x2880_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x2560x2880_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 5, 'spill_reload': True},
    "2048x2880x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x2880x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "2048x2880x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "2048x2880x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 5, 'spill_reload': True},
    "2048x3072x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "2048x3584x1024_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 7, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "2048x3584x1024_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 7, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "2048x4096x1024_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "2048x4096x1024_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "2048x4096x128_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 128, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "2048x4096x128_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 128, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "2048x4096x1536_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x1536_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 3, 'spill_reload': True},
    "2048x4096x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x4096x1536_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': True},
    "2048x4096x2048_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2048_dgt_fast_dgt_fast": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "2048x4096x2048_dgt_fast_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x4096x2048_dgt_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2048_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2048_pe_1x32_pe_1x32_nodma": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2048_pe_1x32_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "2048x4096x2048_pe_wrapx_pe_wrapx": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "2048x4096x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x4096x2304_dgt_fast_dgt_fast": {'tile_m': 128, 'tile_k': 512, 'tile_n': 384, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x4096x2304_dgt_fast_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 384, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 6, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x4096x2304_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 576, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2304_pe_1x32_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 576, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x4096x2304_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 9, 'spill_reload': False},
    "2048x4096x2304_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 9, 'spill_reload': True},
    "2048x4096x3072_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x4096x3072_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "2048x4096x768_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x5120x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "2048x5120x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2048x5120x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "2048x5120x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "2048x512x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "2048x512x3072_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 6, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2048x512x768_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 768, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "2048x768x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2304x4096x2048_dgt_fast_dgt_fast": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "2304x4096x2048_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 18, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 18, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2304x4096x2048_pe_1x32_pe_1x32_nodma": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "2304x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2304x4096x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2304x4096x4096_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 9, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 9, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2560x2560x1280_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 20, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5},
    "2560x2560x1280_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 10, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 10, 'TILES_IN_LOAD_N': 5},
    "2560x4096x1440_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1440, 'TILES_IN_BLOCK_M': 20, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "2560x4096x1440_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 1440, 'TILES_IN_BLOCK_M': 10, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 10, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "2560x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 10, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 10, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2560x4096x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 10, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 10, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2560x4096x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 20, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 20, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "2560x4096x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 10, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 10, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "3072x4096x4096_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 12, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2},
    "3072x4096x4096_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 24, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 24, 'TILES_IN_LOAD_N': 2},
    "3200x768x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 4},
    "3200x768x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 2},
    "4096x1024x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 4},
    "4096x1024x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 4},
    "4096x12800x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "4096x12800x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': True},
    "4096x128x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 128, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "4096x128x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 128, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "4096x1536x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 3, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x1536x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 3, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 4},
    "4096x2048x2048_dgt_fast_dgt_fast": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x2048x2048_dgt_fast_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "4096x2048x2048_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "4096x2048x2048_pe_1x32_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x2048x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "4096x2048x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x2048x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "4096x2048x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "4096x2304x2048_dgt_fast_dgt_fast": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "4096x2304x2048_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x2304x2048_pe_1x32_pe_1x32_nodma": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x2304x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x2304x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "4096x2560x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "4096x2560x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': True},
    "4096x3072x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 32, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 6, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x3072x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 3, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "4096x4096x2048_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x4096x2048_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "4096x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x4096x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x4096x2304_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x4096x2304_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 9, 'spill_reload': True},
    "4096x4096x2304_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 9, 'spill_reload': False},
    "4096x4096x2304_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 9, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 9, 'spill_reload': True},
    "4096x4096x3072_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "4096x4096x3072_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 6, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "4096x4096x4096_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x4096x4608_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x4608x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 9, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x5120x3200_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x5120x3200_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "4096x5120x6400_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 5, 'TILES_IN_LOAD_M': 16, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x5120x6400_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': True},
    "4096x6144x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2},
    "4096x6144x2048_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4},
    "4096x6400x2560_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "4096x6400x2560_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 5, 'spill_reload': True},
    "4096x768x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x8192x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4096x9216x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "4608x4096x4096_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "5120x4096x3200_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "5120x4096x3200_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 16, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 8, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "512x1536x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 3, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "512x3072x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
    "512x4096x1536_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "512x4096x384_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 384, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x4096x768_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 768, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x4096x768_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 3, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 3, 'spill_reload': False},
    "512x4096x768_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 768, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x4096x768_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 768, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 8, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x512x256_dgt_dgt": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': True},
    "512x512x256_pe_1x32_pe_1x32": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x512x256_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x512x256_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 256, 'TILES_IN_BLOCK_M': 2, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 2, 'TILES_IN_LOAD_N': 1, 'spill_reload': False},
    "512x512x512_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1},
    "512x512x512_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 1, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 1},
    "512x768x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 4, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "6400x4096x5120_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 8, 'TILES_IN_BLOCK_N': 5, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 4, 'TILES_IN_LOAD_N': 5, 'spill_reload': False},
    "6400x4096x5120_swizzled_swizzled": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 25, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 25, 'TILES_IN_LOAD_N': 2, 'spill_reload': True},
    "768x1024x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 6, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 2, 'TILES_IN_LOAD_M': 6, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "768x2048x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 6, 'TILES_IN_BLOCK_N': 2, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 6, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "768x4096x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 6, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 4, 'TILES_IN_LOAD_M': 6, 'TILES_IN_LOAD_N': 2, 'spill_reload': False},
    "768x512x2048_prequant_prequant": {'tile_m': 128, 'tile_k': 512, 'tile_n': 512, 'TILES_IN_BLOCK_M': 6, 'TILES_IN_BLOCK_N': 4, 'TILES_IN_BLOCK_K': 1, 'TILES_IN_LOAD_M': 6, 'TILES_IN_LOAD_N': 4, 'spill_reload': False},
}
# fmt: on


@dataclass
class MatmulMxfp8KernelConfig(nl.NKIObject):
    """Kernel-level configuration for MXFP8 matmul, encapsulating tiling and blocking parameters.

    This class is NKI-compatible (inherits NKIObject) so it can be used inside NKI kernels.
    Complex operations (auto_generate_default, validate_shapes, etc.) are standalone functions.
    """

    M: int
    K: int
    N: int
    tile_m: Optional[int] = None
    tile_k: Optional[int] = None
    tile_n: Optional[int] = None
    TILES_IN_BLOCK_M: Optional[int] = None
    TILES_IN_BLOCK_N: Optional[int] = None
    TILES_IN_BLOCK_K: Optional[int] = None
    TILES_IN_LOAD_M: Optional[int] = None
    TILES_IN_LOAD_N: Optional[int] = None
    # Tiles coalesced per K DMA on the F-by-K PE-swizzle 1x32 load path only. None = the
    # whole K-block (LOAD_TILES_IN_K). A smaller value splits the block's contiguous read
    # into groups (trading DMA burst size for load/compute overlap); it never exceeds the
    # block. Ignored on every other load path.
    TILES_IN_LOAD_K: Optional[int] = None
    block_loop_order: LoopOrder = LoopOrder.MNK
    tile_loop_order: LoopOrder = LoopOrder.MNK
    float8_dtype: nki.dtype = nl.float8_e4m3fn
    enable_scale_packing: bool = False
    lnc_sharding: LncShardingMode = LncShardingMode.AUTO
    spill_reload: bool = False
    enable_psum_copy_in: bool = True
    quant_scheme: QuantScheme = QuantScheme.WRAPX
    output_dtype: Optional[nki.dtype] = None
    # Computed by validate_shapes():
    bd: Optional[BlockDescriptor] = None
    lhs_matmul_tile_shape_physical: Optional[tuple] = None
    rhs_matmul_tile_shape_physical: Optional[tuple] = None
    lhs_load_tile_shape: Optional[tuple] = None
    rhs_load_tile_shape: Optional[tuple] = None
    lhs_quantize_tile_shape: Optional[tuple] = None
    rhs_quantize_tile_shape: Optional[tuple] = None
    BLOCKS_IN_M: Optional[int] = None
    BLOCKS_IN_N: Optional[int] = None
    BLOCKS_IN_K: Optional[int] = None

    # Computed tile shape tuples (set by validate_shapes or manually):
    lhs_matmul_tile_shape_logical: Optional[tuple] = None
    rhs_matmul_tile_shape_logical: Optional[tuple] = None

    @property
    def run_with_lnc2(self) -> bool:
        """Back-compat bool view of lnc_sharding (the stored source of truth)."""
        return self.lnc_sharding.run_with_lnc2

    @property
    def lnc_2_shard_rhs(self):
        """Back-compat tri-state view of lnc_sharding (None on OFF/AUTO)."""
        return self.lnc_sharding.lnc_2_shard_rhs


# ------------------------------------------------------------------
# Standalone functions operating on MatmulMxfp8KernelConfig
# ------------------------------------------------------------------


def _max_tiles(dim, tile, cap=0):
    t = max(1, div_ceil(dim, tile))
    return min(t, cap) if cap > 0 else t


def calculate_sbuf_usage(
    config,
    lhs_dtype,
    rhs_dtype,
    output_dtype_str,
):
    """Calculate SBUF memory usage in bytes."""

    BLOCK_M = config.TILES_IN_BLOCK_M * config.tile_m
    BLOCK_K = config.TILES_IN_BLOCK_K * config.tile_k
    BLOCK_N = config.TILES_IN_BLOCK_N * config.tile_n
    total = 0
    if lhs_dtype == PRECISION_BFLOAT16:
        total += BLOCK_M * BLOCK_K * BYTES_PER_DTYPE[lhs_dtype]
    if rhs_dtype == PRECISION_BFLOAT16:
        total += BLOCK_N * BLOCK_K * BYTES_PER_DTYPE[rhs_dtype]
    total += BLOCK_M * BLOCK_K * 2
    total += BLOCK_N * BLOCK_K * 2
    total += BLOCK_M * BLOCK_N * BYTES_PER_DTYPE[output_dtype_str]
    return total


def calc_sbuf_free_dim_size(
    config,
    lhs_dtype,
    rhs_dtype,
    output_dtype_str,
):
    """Calculate maximum SBUF free dimension size in bytes."""

    BLOCK_M = config.TILES_IN_BLOCK_M * config.tile_m
    BLOCK_N = config.TILES_IN_BLOCK_N * config.tile_n
    max_fdim = 0
    if lhs_dtype == PRECISION_BFLOAT16:
        max_fdim = max(max_fdim, config.TILES_IN_BLOCK_K * BLOCK_M * BYTES_PER_DTYPE[lhs_dtype])
    max_fdim = max(max_fdim, config.TILES_IN_BLOCK_K * BLOCK_M * INTERLEAVE_FACTOR)
    if rhs_dtype == PRECISION_BFLOAT16:
        max_fdim = max(max_fdim, config.TILES_IN_BLOCK_K * BLOCK_N * BYTES_PER_DTYPE[rhs_dtype])
    max_fdim = max(max_fdim, config.TILES_IN_BLOCK_K * BLOCK_N * INTERLEAVE_FACTOR)
    effective_order = config.block_loop_order or LoopOrder.MNK
    if effective_order == LoopOrder.MNK:
        max_fdim = max(max_fdim, config.TILES_IN_BLOCK_M * BLOCK_N * BYTES_PER_DTYPE[output_dtype_str])
    return max_fdim


def fits_in_sbuf(
    config,
    lhs_dtype,
    rhs_dtype,
    output_dtype_str,
):
    """Check if this configuration fits within SBUF limits."""

    if config.tile_k == None or config.tile_k == 0:
        return False
    sbuf_limit = SBUF_LIMIT_BYTES / (TILE_SIZE_P_MAX_LOGICAL / config.tile_k)
    if (
        config.tile_m == None
        or config.tile_n == None
        or config.TILES_IN_BLOCK_M == None
        or config.TILES_IN_BLOCK_K == None
        or config.TILES_IN_BLOCK_N == None
    ):
        return False
    return (
        calculate_sbuf_usage(config, lhs_dtype, rhs_dtype, output_dtype_str) < sbuf_limit
        and calc_sbuf_free_dim_size(config, lhs_dtype, rhs_dtype, output_dtype_str) < SBUF_F_DIM_LIMIT_BYTES
    )


def _fits_sbuf(bm, bn, bk, tile_m, tile_n, tile_k, k_bytes, out_bytes, sbuf_lim):
    block_m, block_n, block_k = bm * tile_m, bn * tile_n, bk * tile_k
    total = k_bytes * block_k * (block_m + block_n) + block_m * block_n * out_bytes
    if total >= sbuf_lim:
        return False
    fdim_m = bk * block_m * INTERLEAVE_FACTOR
    fdim_n = bk * block_n * INTERLEAVE_FACTOR
    fdim_out = bm * block_n * out_bytes
    return fdim_m <= SBUF_F_DIM_LIMIT_BYTES and fdim_n <= SBUF_F_DIM_LIMIT_BYTES and fdim_out <= SBUF_F_DIM_LIMIT_BYTES


def _max_tiles_for_dgt(tile_f):
    """Return the maximum tiles_in_load value that satisfies the DGT transpose limit.

    In _generate_vector_offset_pattern, NUM_PARTITIONS = (INTERLEAVE_FACTOR * load_tile_f) // P_MAX.
    The nc_transpose requires NUM_PARTITIONS <= TRANSPOSE_CHUNK_SIZE (32).
    """
    P_MAX = 128
    TRANSPOSE_CHUNK_SIZE = 32
    return (TRANSPOSE_CHUNK_SIZE * P_MAX) // (INTERLEAVE_FACTOR * tile_f)


def _largest_divisor_within(n, limit):
    """Return the largest divisor of n that is <= limit."""
    for d in range(min(n, limit), 0, -1):
        if n % d == 0:
            return d
    return 1


def resolve_lnc2_sharding(
    M,
    N,
    lnc_sharding: LncShardingMode,
    tile_m=128,
    tile_n=128,
    lhs_is_prequant=False,
    rhs_is_prequant=False,
) -> LncShardingMode:
    """Resolve an LNC2 sharding mode: pick the AUTO axis, then disable if it fits one tile.

    Takes a LncShardingMode in and returns a fully resolved mode (OFF / SHARD_M /
    SHARD_N -- never AUTO):
      - OFF stays OFF.
      - AUTO picks the shard axis. If exactly one operand is pre-quantized, shard the
        OTHER operand's free dim -- the pre-quantized operand is loaded as-is (no
        on-chip quantize/transpose), so the on-chip prep cost is dominated by the
        non-pre-quantized operand, whose free dim is split across the two cores:
          LHS pre-quantized -> SHARD_N (split the RHS free dim).
          RHS pre-quantized -> SHARD_M (split the LHS free dim).
        Otherwise (neither or both pre-quantized) shard the larger dim
        (N >= M -> SHARD_N, else SHARD_M).
      - An explicit SHARD_M / SHARD_N is kept as-is (no auto-pick).

    Then the single-tile auto-disable: SHARD_N with N <= tile_n, or SHARD_M with
    M <= tile_m, collapses to OFF.
    """
    if lnc_sharding == LncShardingMode.OFF:
        return LncShardingMode.OFF
    if lnc_sharding == LncShardingMode.AUTO:
        if lhs_is_prequant and not rhs_is_prequant:
            lnc_sharding = LncShardingMode.SHARD_N  # split the non-pre-quantized RHS
        elif rhs_is_prequant and not lhs_is_prequant:
            lnc_sharding = LncShardingMode.SHARD_M  # split the non-pre-quantized LHS
        else:
            lnc_sharding = LncShardingMode.SHARD_N if N >= M else LncShardingMode.SHARD_M
    if lnc_sharding == LncShardingMode.SHARD_N and N <= tile_n:
        return LncShardingMode.OFF
    if lnc_sharding == LncShardingMode.SHARD_M and M <= tile_m:
        return LncShardingMode.OFF
    return lnc_sharding


def auto_generate_default(
    config,
    lhs_dtype,
    rhs_dtype,
    output_dtype_str,
    use_cache=True,
    enable_psum_copy_in=None,
    fast_dma_transpose=False,
    lhs_is_swizzled=True,
    rhs_is_swizzled=True,
    lhs_load_with_PE_swizzle=False,
    rhs_load_with_PE_swizzle=False,
    disable_dma_transpose=False,
):
    """Fill None fields on config using the block_count_reducer strategy. Returns config.

    The operand-layout facts (whether each side is pre-swizzled and whether it is
    loaded via PE transpose) are passed in per operand rather than read off the
    config: they describe the input tensors, which the config no longer duplicates.
    Callers source them from the two TensorDescriptors (is_swizzled / the
    uses_pe_swizzle property); the MoE forward, which has no descriptors here,
    passes the facts for its unswizzled BF16 GEMMs directly. The defaults
    (swizzled, no PE) preserve the standalone-matmul behavior.

    Precision parameters use plain string values ('mxfp8', 'mxfp8_x4', 'bfloat16', 'fp32')
    and can be resolved from config.lhs_precision/rhs_precision/output_precision fields.

    config.M/config.N are full (pre-LNC2-shard) dims, matching config.BLOCKS_IN_*, which
    validate_shapes derives from the unsharded logical shape. This function is the single
    place the LNC2 shard is applied: it resolves which axis is sharded from the full dims
    (a full-shape decision) and derives the per-core dims used for tiling and the cache key.

    Args:
        config: MatmulMxfp8KernelConfig with full M/K/N and tile sizes set.
        lhs_dtype: Precision string for left-hand side operand.
        rhs_dtype: Precision string for right-hand side operand.
        output_dtype_str: Precision string for output accumulator.
        use_cache: Whether to look up the autotune cache for known-good configs.
            Set to False when the cached configs (tuned for standalone matmul) may not
            be optimal for the caller's context (e.g., MLP backward phases).
    """
    # Resolve the LNC2 flags from the full dims (both the sharded axis and the
    # single-tile auto-disable are full-shape decisions), then shard once. Write the
    # resolved flags back so the tiling, the cache key, and any caller reading these
    # fields afterwards all describe the same per-core matmul. Prequant flags steer the
    # auto axis choice toward sharding the non-pre-quantized operand (see resolve_lnc2_sharding).
    lhs_is_prequant = lhs_dtype in (PRECISION_MXFP8, PRECISION_MXFP8_X4)
    rhs_is_prequant = rhs_dtype in (PRECISION_MXFP8, PRECISION_MXFP8_X4)
    config.lnc_sharding = resolve_lnc2_sharding(
        config.M,
        config.N,
        config.lnc_sharding,
        lhs_is_prequant=lhs_is_prequant,
        rhs_is_prequant=rhs_is_prequant,
    )
    effective_m, effective_n = effective_shard_dims(config.M, config.N, config.run_with_lnc2, config.lnc_2_shard_rhs)

    # Check autotune cache for known-good config. The key encodes shape, per-operand
    # dtype, and load method (swizzled / dgt / pe_<scheme>), since the optimal tiling
    # depends on the input-preparation path, not just the shape.
    #
    # Only consult the cache when the caller left the blocking/load tiling unspecified,
    # i.e. is actually asking to be autotuned. A caller that fully specifies the tiling
    # (all TILES_IN_BLOCK_*/TILES_IN_LOAD_* set) has opted out of autotuning, so the
    # cache must not run — otherwise its cached spill_reload / enable_psum_copy_in would
    # silently override the caller's explicit values.
    _tiling_unspecified = (
        config.TILES_IN_BLOCK_M is None
        or config.TILES_IN_BLOCK_N is None
        or config.TILES_IN_BLOCK_K is None
        or config.TILES_IN_LOAD_M is None
        or config.TILES_IN_LOAD_N is None
    )
    # Full dims + resolved flags, the same way the offline cache writer builds the key.
    # fast_dma_transpose only takes effect on an unswizzled operand (matmul_mxfp8
    # zeroes it for a swizzled side when building the TensorDescriptor), so mirror
    # that per-operand here or the key would not match what actually loads.
    _cache_key = autotune_cache_key(
        config.M,
        config.K,
        config.N,
        lhs_dtype,
        rhs_dtype,
        lhs_is_swizzled,
        rhs_is_swizzled,
        lhs_load_with_PE_swizzle,
        rhs_load_with_PE_swizzle,
        config.quant_scheme,
        config.run_with_lnc2,
        config.lnc_2_shard_rhs,
        lhs_fast_dma_transpose=fast_dma_transpose and not lhs_is_swizzled,
        rhs_fast_dma_transpose=fast_dma_transpose and not rhs_is_swizzled,
        disable_dma_transpose=disable_dma_transpose,
    )
    _cached = _tiling_unspecified and _AUTOTUNE_CACHE.get(_cache_key)
    if use_cache and _cached:
        if config.tile_m == None:
            config.tile_m = _cached.get('tile_m', 128)
        if config.tile_k == None:
            config.tile_k = _cached.get('tile_k', 512)
        if config.tile_n == None:
            config.tile_n = _cached.get('tile_n', 512)
        config.lhs_matmul_tile_shape_logical = (config.tile_k, config.tile_m)
        config.rhs_matmul_tile_shape_logical = (config.tile_k, config.tile_n)
        if config.TILES_IN_BLOCK_M == None:
            config.TILES_IN_BLOCK_M = _cached['TILES_IN_BLOCK_M']
        if config.TILES_IN_BLOCK_N == None:
            config.TILES_IN_BLOCK_N = _cached['TILES_IN_BLOCK_N']
        if config.TILES_IN_BLOCK_K == None:
            config.TILES_IN_BLOCK_K = _cached['TILES_IN_BLOCK_K']
        if config.TILES_IN_LOAD_M == None:
            config.TILES_IN_LOAD_M = _cached['TILES_IN_LOAD_M']
        if config.TILES_IN_LOAD_N == None:
            config.TILES_IN_LOAD_N = _cached['TILES_IN_LOAD_N']
        # Optional K coalescing knob; absent in most entries (None keeps whole-block reads).
        if config.TILES_IN_LOAD_K == None and 'TILES_IN_LOAD_K' in _cached:
            config.TILES_IN_LOAD_K = _cached['TILES_IN_LOAD_K']
        config.enable_psum_copy_in = _cached.get('enable_psum_copy_in', True)
        if 'spill_reload' in _cached:
            config.spill_reload = _cached['spill_reload']
        if enable_psum_copy_in is not None:
            config.enable_psum_copy_in = enable_psum_copy_in
        return config

    if config.tile_m == None:
        config.tile_m = 128
    if config.tile_k == None:
        if not lhs_is_swizzled or not rhs_is_swizzled:
            # DGT requires tile_k=512; kernel handles K < 512 as remainder
            config.tile_k = 512
        elif config.K >= 512:
            config.tile_k = 512
        elif config.K >= 256:
            config.tile_k = 256
        else:
            config.tile_k = 128

    if config.tile_n == None:
        config.tile_n = 128
        for candidate in [512, 256, 128]:
            if effective_n >= candidate and effective_n % candidate == 0:
                config.tile_n = candidate
                break

    tile_m, tile_k, tile_n = config.tile_m, config.tile_k, config.tile_n
    config.lhs_matmul_tile_shape_logical = (tile_k, tile_m)
    config.rhs_matmul_tile_shape_logical = (tile_k, tile_n)
    max_m = _max_tiles(effective_m, tile_m, MAX_BLOCK_M // tile_m)
    max_n = _max_tiles(effective_n, tile_n, MAX_BLOCK_N // tile_n)
    max_k = _max_tiles(config.K, tile_k)

    if lhs_is_prequant and rhs_is_prequant:
        k_bytes = 2
    elif not lhs_is_prequant and not rhs_is_prequant:
        k_bytes = 4
    else:
        k_bytes = 3
    out_bytes = BYTES_PER_DTYPE[output_dtype_str]
    sbuf_lim = SBUF_LIMIT_BYTES / (TILE_SIZE_P_MAX_LOGICAL / tile_k) * 0.8

    bm = min(16, max_m)
    bn = min(2, max_n)
    bk = min(8, max_k)
    while not _fits_sbuf(bm, bn, bk, tile_m, tile_n, tile_k, k_bytes, out_bytes, sbuf_lim) and (
        bm > 1 or bn > 1 or bk > 1
    ):
        k_cost = bk * tile_k * (bm * tile_m + bn * tile_n)
        mn_cost = bm * tile_m * bn * tile_n
        if k_cost >= mn_cost and bk > 1:
            bk = max(1, bk // 2)
        elif bm >= bn and bm > 1:
            bm = max(1, bm // 2)
        elif bn > 1:
            bn = max(1, bn // 2)
        else:
            bk = max(1, bk // 2)

    # Grow phase
    for try_bk in range(bk + 1, max_k + 1):
        if not _fits_sbuf(bm, bn, try_bk, tile_m, tile_n, tile_k, k_bytes, out_bytes, sbuf_lim):
            break
        if div_ceil(config.K, try_bk * tile_k) < div_ceil(config.K, bk * tile_k):
            bk = try_bk
    for try_bm in range(bm + 1, max_m + 1):
        if not _fits_sbuf(try_bm, bn, bk, tile_m, tile_n, tile_k, k_bytes, out_bytes, sbuf_lim):
            break
        if div_ceil(effective_m, try_bm * tile_m) < div_ceil(effective_m, bm * tile_m):
            bm = try_bm
    for try_bn in range(bn + 1, max_n + 1):
        if not _fits_sbuf(bm, try_bn, bk, tile_m, tile_n, tile_k, k_bytes, out_bytes, sbuf_lim):
            break
        if div_ceil(effective_n, try_bn * tile_n) < div_ceil(effective_n, bn * tile_n):
            bn = try_bn

    if config.TILES_IN_BLOCK_M == None:
        config.TILES_IN_BLOCK_M = bm
    if config.TILES_IN_BLOCK_N == None:
        config.TILES_IN_BLOCK_N = bn
    if config.TILES_IN_BLOCK_K == None:
        config.TILES_IN_BLOCK_K = bk
    if config.TILES_IN_LOAD_M == None:
        if not lhs_is_swizzled and lhs_dtype == PRECISION_BFLOAT16:
            max_load_m = _max_tiles_for_dgt(tile_m)
            config.TILES_IN_LOAD_M = _largest_divisor_within(config.TILES_IN_BLOCK_M, max_load_m)
        else:
            config.TILES_IN_LOAD_M = config.TILES_IN_BLOCK_M
    if config.TILES_IN_LOAD_N == None:
        if not rhs_is_swizzled and rhs_dtype == PRECISION_BFLOAT16:
            max_load_n = _max_tiles_for_dgt(tile_n)
            config.TILES_IN_LOAD_N = _largest_divisor_within(config.TILES_IN_BLOCK_N, max_load_n)
        else:
            config.TILES_IN_LOAD_N = config.TILES_IN_BLOCK_N

    if enable_psum_copy_in is not None:
        config.enable_psum_copy_in = enable_psum_copy_in
    return config


def validate_shapes(config, lhs_td, rhs_td):
    """Validate inputs and compute derived shape attributes on config. Returns config."""
    # K-by-F (is_f_by_k=False) unswizzled BF16 inputs require F (M for LHS, N for RHS) to be a
    # multiple of MIN_F_FOR_QUANTIZATION -- the DGT flexible-F floor; the PE-transpose loader masks
    # partial (<128) final sub-tiles down to that granularity.
    min_f = quantize_mxfp8_utils.MIN_F_FOR_QUANTIZATION
    if (
        lhs_td.orientation == TensorOrientation.K_BY_F
        and not lhs_td.is_swizzled
        and not lhs_td.is_quantized
        and lhs_td.logical_shape is not None
    ):
        lhs_F = lhs_td.logical_shape[1]
        kernel_assert(lhs_F % min_f == 0, f"K-by-F LHS requires F dimension ({lhs_F}) to be divisible by {min_f}.")
    if (
        rhs_td.orientation == TensorOrientation.K_BY_F
        and not rhs_td.is_swizzled
        and not rhs_td.is_quantized
        and rhs_td.logical_shape is not None
    ):
        rhs_F = rhs_td.logical_shape[1]
        kernel_assert(rhs_F % min_f == 0, f"K-by-F RHS requires F dimension ({rhs_F}) to be divisible by {min_f}.")

    tile_k = config.tile_k
    tile_m = config.tile_m
    tile_n = config.tile_n

    MATMUL_TILE_K_PHYSICAL = tile_k // quantize_mxfp8_utils.INTERLEAVE_FACTOR
    config.lhs_matmul_tile_shape_physical = (MATMUL_TILE_K_PHYSICAL, tile_m)
    config.rhs_matmul_tile_shape_physical = (MATMUL_TILE_K_PHYSICAL, tile_n)

    if lhs_td.is_quantized:
        config.lhs_load_tile_shape = config.lhs_matmul_tile_shape_physical
        lhs_td.scales_are_packed = quantize_mxfp8_utils.are_scales_packed(lhs_td.data.shape, lhs_td.scales.shape)
        kernel_assert(
            quantize_mxfp8_utils.scale_packing_not_possible(lhs_td.data.shape, lhs_td.scales.shape)
            or config.enable_scale_packing == lhs_td.scales_are_packed,
            f"use_scale_packing={config.enable_scale_packing} but lhs_scales_are_packed={lhs_td.scales_are_packed}",
        )
    elif lhs_td.is_swizzled:
        config.lhs_load_tile_shape = (MATMUL_TILE_K_PHYSICAL, tile_m * 4)
    else:
        config.lhs_load_tile_shape = (tile_k, tile_m)

    if rhs_td.is_quantized:
        config.rhs_load_tile_shape = config.rhs_matmul_tile_shape_physical
        rhs_td.scales_are_packed = quantize_mxfp8_utils.are_scales_packed(rhs_td.data.shape, rhs_td.scales.shape)
        kernel_assert(
            quantize_mxfp8_utils.scale_packing_not_possible(rhs_td.data.shape, rhs_td.scales.shape)
            or config.enable_scale_packing == rhs_td.scales_are_packed,
            f"use_scale_packing={config.enable_scale_packing} but rhs_scales_are_packed={rhs_td.scales_are_packed}",
        )
    elif rhs_td.is_swizzled:
        config.rhs_load_tile_shape = (MATMUL_TILE_K_PHYSICAL, tile_n * 4)
    else:
        config.rhs_load_tile_shape = (tile_k, tile_n)

    config.lhs_quantize_tile_shape = (MATMUL_TILE_K_PHYSICAL, tile_m * 4)
    config.rhs_quantize_tile_shape = (MATMUL_TILE_K_PHYSICAL, tile_n * 4)

    if config.enable_scale_packing and MATMUL_TILE_K_PHYSICAL < quantize_mxfp8_utils.Q_TILE_K:
        kernel_assert(
            not (lhs_td.is_quantized and lhs_td.scales_are_packed),
            f"Pre-quantized LHS with packed scales requires MATMUL_TILE_K_PHYSICAL >= Q_TILE_K={quantize_mxfp8_utils.Q_TILE_K}",
        )
        kernel_assert(
            not (rhs_td.is_quantized and rhs_td.scales_are_packed),
            f"Pre-quantized RHS with packed scales requires MATMUL_TILE_K_PHYSICAL >= Q_TILE_K={quantize_mxfp8_utils.Q_TILE_K}",
        )

    kernel_assert(
        config.TILES_IN_LOAD_M <= config.TILES_IN_BLOCK_M,
        f"TILES_IN_LOAD_M ({config.TILES_IN_LOAD_M}) must be <= TILES_IN_BLOCK_M ({config.TILES_IN_BLOCK_M})",
    )
    kernel_assert(
        config.TILES_IN_LOAD_N <= config.TILES_IN_BLOCK_N,
        f"TILES_IN_LOAD_N ({config.TILES_IN_LOAD_N}) must be <= TILES_IN_BLOCK_N ({config.TILES_IN_BLOCK_N})",
    )
    kernel_assert(
        config.TILES_IN_BLOCK_M % config.TILES_IN_LOAD_M == 0, f"TILES_IN_BLOCK_M must be a multiple of TILES_IN_LOAD_M"
    )
    kernel_assert(
        config.TILES_IN_BLOCK_N % config.TILES_IN_LOAD_N == 0, f"TILES_IN_BLOCK_N must be a multiple of TILES_IN_LOAD_N"
    )

    config.bd = BlockDescriptor(
        TILES_IN_BLOCK_M=config.TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=config.TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=config.TILES_IN_BLOCK_K,
        lhs_matmul_tile_shape_logical=config.lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=config.rhs_matmul_tile_shape_logical,
        lhs_load_tile_shape=config.lhs_load_tile_shape,
        rhs_load_tile_shape=config.rhs_load_tile_shape,
    )

    K_LOGICAL_LHS, M_LOGICAL = lhs_td.logical_shape
    K_LOGICAL_RHS, N_LOGICAL = rhs_td.logical_shape
    kernel_assert(K_LOGICAL_LHS == K_LOGICAL_RHS, f"K dimension mismatch: {K_LOGICAL_LHS} vs {K_LOGICAL_RHS}")
    K_LOGICAL = K_LOGICAL_LHS
    _, M_PHYSICAL = lhs_td.physical_shape
    _, N_PHYSICAL = rhs_td.physical_shape

    kernel_assert(config.bd.BLOCK_M_LOGICAL % tile_m == 0, f"BLOCK_M_LOGICAL must be divisible by tile_m")
    kernel_assert(config.bd.BLOCK_K_LOGICAL % tile_k == 0, f"BLOCK_K_LOGICAL must be divisible by tile_k")
    kernel_assert(config.bd.BLOCK_N_LOGICAL % tile_n == 0, f"BLOCK_N_LOGICAL must be divisible by tile_n")
    kernel_assert(
        MATMUL_TILE_K_PHYSICAL <= nl.tile_size.pmax
        and MATMUL_TILE_K_PHYSICAL % quantize_mxfp8_utils.MX_PARTITION_SIZE == 0,
        f"MATMUL_TILE_K_PHYSICAL ({MATMUL_TILE_K_PHYSICAL}) invalid",
    )
    kernel_assert(
        tile_m <= nl.tile_size.gemm_stationary_fmax and tile_m % quantize_mxfp8_utils.MX_PARTITION_SIZE == 0,
        f"tile_m ({tile_m}) invalid",
    )
    kernel_assert(
        tile_n <= quantize_mxfp8_utils.TILE_SIZE_GEMM_MOVING_MAX
        and tile_n % quantize_mxfp8_utils.MX_PARTITION_SIZE == 0,
        f"tile_n ({tile_n}) invalid",
    )
    kernel_assert(config.bd.BLOCK_M_PHYSICAL % tile_m == 0, f"BLOCK_M_PHYSICAL must be divisible by tile_m")
    kernel_assert(config.bd.BLOCK_K_PHYSICAL_LHS % MATMUL_TILE_K_PHYSICAL == 0, f"BLOCK_K_PHYSICAL_LHS invalid")
    kernel_assert(config.bd.BLOCK_K_PHYSICAL_RHS % MATMUL_TILE_K_PHYSICAL == 0, f"BLOCK_K_PHYSICAL_RHS invalid")
    kernel_assert(config.bd.BLOCK_N_PHYSICAL % tile_n == 0, f"BLOCK_N_PHYSICAL must be divisible by tile_n")
    kernel_assert(2 * M_PHYSICAL >= config.bd.BLOCK_M_PHYSICAL or M_PHYSICAL < tile_m, f"M_PHYSICAL too small")
    kernel_assert(2 * M_LOGICAL >= config.bd.BLOCK_M_LOGICAL or M_LOGICAL < tile_m, f"M_LOGICAL too small")
    kernel_assert(2 * N_PHYSICAL >= config.bd.BLOCK_N_PHYSICAL or N_PHYSICAL < tile_n, f"N_PHYSICAL too small")
    kernel_assert(2 * N_LOGICAL >= config.bd.BLOCK_N_LOGICAL or N_LOGICAL < tile_n, f"N_LOGICAL too small")
    kernel_assert(2 * K_LOGICAL >= config.bd.BLOCK_K_LOGICAL or K_LOGICAL < tile_k, f"K_LOGICAL too small")

    config.BLOCKS_IN_M = div_ceil(M_LOGICAL, config.bd.BLOCK_M_LOGICAL)
    config.BLOCKS_IN_N = div_ceil(N_LOGICAL, config.bd.BLOCK_N_LOGICAL)
    config.BLOCKS_IN_K = div_ceil(K_LOGICAL, config.bd.BLOCK_K_LOGICAL)

    if not lhs_td.is_swizzled or not rhs_td.is_swizzled:
        # DGT requires K divisible by 128; the fast DMA transpose path relaxes this to
        # MX_PARTITION_SIZE (32) since it gathers any 32-aligned K in a single op.
        unswizzled_fast_dma = (not lhs_td.is_swizzled and lhs_td.fast_dma_transpose) or (
            not rhs_td.is_swizzled and rhs_td.fast_dma_transpose
        )
        k_alignment = quantize_mxfp8_utils.MX_PARTITION_SIZE if unswizzled_fast_dma else 128
        kernel_assert(K_LOGICAL % k_alignment == 0, f"K must be divisible by {k_alignment} for DGT")

    return config


def resolve_matmul_config_with_validation(
    config: Optional[MatmulMxfp8KernelConfig],
    lhs_td: TensorDescriptor,
    rhs_td: TensorDescriptor,
    run_with_lnc2: bool,
    spill_reload: bool,
    use_scale_packing: bool,
    lnc_2_shard_rhs: Optional[bool] = None,
) -> MatmulMxfp8KernelConfig:
    """Resolve a single matmul config following the standalone matmul pattern.

    Derives M/K/N from TensorDescriptors, calls auto_generate_default
    (bypassing cache), then validate_shapes. Suitable for use in MLP, MoE,
    or any multi-matmul kernel that needs per-phase config resolution.

    Args:
        lnc_2_shard_rhs: Override for RHS sharding. None lets auto_generate_default
            decide based on M vs N.
    """
    # Per-core dims: callers here (MLP/MoE multi-matmul kernels) hand descriptors that
    # are already LNC2-sharded, and drive their own loop bounds from the sharded shape.
    # This path also passes use_cache=False, so no autotune key is built -- the shape only
    # feeds the auto-sizer, which must size the per-core matmul the kernel actually runs.
    K_lhs, M = lhs_td.sharded_logical_shape
    K_rhs, N = rhs_td.sharded_logical_shape
    kernel_assert(K_lhs == K_rhs, f"K dimension mismatch: LHS K={K_lhs} vs RHS K={K_rhs}")
    K = K_lhs

    if config is None:
        config = MatmulMxfp8KernelConfig(M=M, K=K, N=N)
    else:
        if config.M is None:
            config.M = M
        if config.K is None:
            config.K = K
        if config.N is None:
            config.N = N

    config.spill_reload = spill_reload
    config.enable_scale_packing = use_scale_packing
    # Preserve the existing shard axis when the caller passes lnc_2_shard_rhs=None.
    _shard = lnc_2_shard_rhs if lnc_2_shard_rhs is not None else config.lnc_2_shard_rhs
    config.lnc_sharding = LncShardingMode.from_bools(run_with_lnc2, _shard)

    lhs_precision = operand_precision(lhs_td.is_quantized, lhs_td.is_x4)
    rhs_precision = operand_precision(rhs_td.is_quantized, rhs_td.is_x4)

    auto_generate_default(
        config,
        lhs_precision,
        rhs_precision,
        PRECISION_FP32,
        use_cache=False,
        lhs_is_swizzled=lhs_td.is_swizzled,
        rhs_is_swizzled=rhs_td.is_swizzled,
        lhs_load_with_PE_swizzle=lhs_td.uses_pe_swizzle,
        rhs_load_with_PE_swizzle=rhs_td.uses_pe_swizzle,
    )

    validate_shapes(config, lhs_td, rhs_td)

    return config


def auto_generate_random(
    config,
    lhs_dtype,
    rhs_dtype,
    output_dtype_str,
):
    """Generate a new KernelConfig with random valid values from dimension chains."""

    m_chains = _generate_m_chains(config)
    n_chains = _generate_n_chains(config)
    k_chains = _generate_k_chains(config)

    m_chain = random.choice(m_chains)
    n_chain = random.choice(n_chains)
    k_chain = random.choice(k_chains)

    return MatmulMxfp8KernelConfig(
        M=config.M,
        K=config.K,
        N=config.N,
        tile_m=m_chain[0],
        TILES_IN_BLOCK_M=m_chain[1],
        TILES_IN_LOAD_M=m_chain[2],
        tile_n=n_chain[0],
        TILES_IN_BLOCK_N=n_chain[1],
        TILES_IN_LOAD_N=n_chain[2],
        tile_k=k_chain[0],
        TILES_IN_BLOCK_K=k_chain[1],
        block_loop_order=config.block_loop_order,
        tile_loop_order=config.tile_loop_order,
        float8_dtype=config.float8_dtype,
        enable_scale_packing=config.enable_scale_packing,
        lnc_sharding=config.lnc_sharding,
        spill_reload=config.spill_reload,
        output_dtype=config.output_dtype,
    )


def _generate_m_chains(config):
    chains = []
    if config.tile_m != None:
        tile_m_options = [config.tile_m]
    else:
        min_d = min(TILE_M_DEFAULTS)
        if config.M < min_d:
            tile_m_options = [config.M] if config.M * 2 < min_d else [min_d]
        else:
            tile_m_options = [t for t in TILE_M_DEFAULTS if t <= config.M]
    for tm in tile_m_options:
        if config.TILES_IN_BLOCK_M != None:
            bm_opts = [config.TILES_IN_BLOCK_M]
        else:
            bm_opts = []
            bm = 1
            while bm * tm < config.M and bm * tm <= MAX_BLOCK_M:
                bm_opts.append(bm)
                bm *= 2
            if bm * tm <= MAX_BLOCK_M:
                bm_opts.append(bm)
        for bm in bm_opts:
            lm_opts = (
                [config.TILES_IN_LOAD_M]
                if config.TILES_IN_LOAD_M != None
                else [bm] + ([bm // 2] if bm % 2 == 0 else []) + ([bm // 4] if bm % 4 == 0 else [])
            )
            for lm in lm_opts:
                chains.append((tm, bm, lm))
    return chains


def _generate_n_chains(config):
    chains = []
    if config.tile_n != None:
        tile_n_options = [config.tile_n]
    else:
        min_d = min(TILE_N_DEFAULTS)
        if config.N < min_d:
            tile_n_options = [config.N] if config.N * 2 < min_d else [min_d]
        else:
            tile_n_options = [t for t in TILE_N_DEFAULTS if t <= config.N]
    for tn in tile_n_options:
        if config.TILES_IN_BLOCK_N != None:
            bn_opts = [config.TILES_IN_BLOCK_N]
        else:
            bn_opts = []
            bn = 1
            while bn * tn < config.N and bn * tn <= MAX_BLOCK_N:
                bn_opts.append(bn)
                bn *= 2
            if bn * tn <= MAX_BLOCK_N:
                bn_opts.append(bn)
        for bn in bn_opts:
            ln_opts = (
                [config.TILES_IN_LOAD_N]
                if config.TILES_IN_LOAD_N != None
                else [bn] + ([bn // 2] if bn % 2 == 0 else []) + ([bn // 4] if bn % 4 == 0 else [])
            )
            for ln in ln_opts:
                chains.append((tn, bn, ln))
    return chains


def _generate_k_chains(config):
    chains = []
    if config.tile_k != None:
        tile_k_options = [config.tile_k]
    else:
        min_d = min(TILE_K_DEFAULTS)
        if config.K < min_d:
            tile_k_options = [config.K] if config.K * 2 < min_d else [min_d]
        else:
            tile_k_options = [t for t in TILE_K_DEFAULTS if t <= config.K]
    for tk in tile_k_options:
        if config.TILES_IN_BLOCK_K != None:
            bk_opts = [config.TILES_IN_BLOCK_K]
        else:
            bk_opts = []
            bk = 1
            while bk * tk < config.K:
                bk_opts.append(bk)
                bk *= 2
            bk_opts.append(bk)
        for bk in bk_opts:
            chains.append((tk, bk))
    return chains


def _dedup_positive(values):
    """Deduplicate a list of ints, keeping only values >= 1, preserving order."""
    seen = set()
    out = []
    for v in values:
        if v >= 1 and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def generate_autotune_candidates(
    config: 'MatmulMxfp8KernelConfig',
    lhs_is_prequant: bool = False,
    rhs_is_prequant: bool = False,
) -> 'List[MatmulMxfp8KernelConfig]':
    """Generate valid candidate configs for auto-tuning a given shape.

    Takes a MatmulMxfp8KernelConfig with M, K, N set (and optionally run_with_lnc2,
    lnc_2_shard_rhs). Returns a list of fully-populated MatmulMxfp8KernelConfig
    objects covering the ablation space.

    lhs_is_prequant / rhs_is_prequant steer the auto LNC2 axis choice exactly as
    auto_generate_default does (shard the non-pre-quantized operand when exactly one
    side is pre-quantized), so the candidates are tiled for the same per-core shape
    the cache reader looks up. They default to False, which reproduces the plain
    shard-the-larger-dim behavior for the neither/both-prequant methods.

    Autotune design:
      - tile_m is fixed at 128, tile_k is fixed at 512 (128 when K<=128).
      - tile_n is ablated over 3 strategies: divisibility-scan, threshold-based, and
        largest non-power-of-two exact divisor.
      - Blocking: BM from 4 options (full/half/capped), BN from 2 options (full/half),
        BK from 4 options (full/half/capped).
      - Load tiling: LM from {4, 8, BM}, LN from {BN, min(BN, 2)}.
      - Candidates are filtered by SBUF capacity and BM >= BN constraint.
      - Each candidate is emitted with both spill_reload=True and spill_reload=False.
      - Best configs per shape are stored in _AUTOTUNE_CACHE and used by
        auto_generate_default() as a lookup before falling back to heuristics.
    """
    M, K, N = config.M, config.K, config.N

    # Step 0: Sharding, resolved exactly as auto_generate_default does so the candidates
    # this emits are tiled for the same per-core shape the cache reader looks up. The
    # prequant flags steer the auto axis toward sharding the non-pre-quantized operand.
    resolved_lnc_sharding = resolve_lnc2_sharding(
        M,
        N,
        config.lnc_sharding,
        lhs_is_prequant=lhs_is_prequant,
        rhs_is_prequant=rhs_is_prequant,
    )
    effective_m, effective_n = effective_shard_dims(
        M, N, resolved_lnc_sharding.run_with_lnc2, resolved_lnc_sharding.lnc_2_shard_rhs
    )

    # Step 1: Fixed tile sizes (from cache analysis: tile_m=128 always, tile_k=512 96%)
    tile_m = 128
    if K >= 512:
        tile_k_default = 512
    elif K >= 256:
        tile_k_default = 256
    else:
        tile_k_default = 128

    # tile_k options: the fixed default, plus exact multiple-of-128 divisors of K in
    # (128, 512] when the default leaves a K remainder (non-power-of-two K such as 2304,
    # where 512 covers 4.5 tiles). An exact divisor gives every K-block a uniform tile
    # count, avoiding the thin remainder tile. K divisible by the default keeps a single
    # option, so divisible-K shapes generate byte-identical candidates (cache-preserving).
    tile_k_options = [tile_k_default]
    if K % tile_k_default != 0:
        for c in (384, 256, 128):
            if c < K and K % c == 0 and c not in tile_k_options:
                tile_k_options.append(c)
    tile_k_options = _dedup_positive(tile_k_options)

    # tile_n — 2 options
    # Option A: power-of-two divisibility scan
    tile_n_a = 128
    for c in [512, 256, 128]:
        if effective_n >= c and effective_n % c == 0:
            tile_n_a = c
            break
    # Option B: threshold
    if effective_n >= 2048:
        tile_n_b = 512
    elif effective_n <= 128:
        tile_n_b = 128
    else:
        tile_n_b = effective_n
    tile_n_options = _dedup_positive([tile_n_a, tile_n_b])

    # Step 2: Blocking options
    tiles_in_m = max(1, effective_m // tile_m)
    bm_options = _dedup_positive([tiles_in_m, tiles_in_m // 2, min(8, tiles_in_m), min(16, tiles_in_m)])

    out_bytes = BYTES_PER_DTYPE[PRECISION_FP32]

    # Build candidates
    seen = set()
    candidates = []

    for tile_k in tile_k_options:
        tiles_in_k = max(1, K // tile_k)
        bk_options = _dedup_positive([tiles_in_k, tiles_in_k // 2, min(2, tiles_in_k), min(4, tiles_in_k)])

        # SBUF limits for filtering (use mxfp8 k_bytes=2 as the permissive check).
        # Multiplier 1.5 is used because the kernel reuses SBUF across loop iterations,
        # so the static estimate is conservative. 1.5 preserves all known-good cached configs
        # (max observed ratio is 1.46) while still eliminating configs that are 2x+ over limit.
        sbuf_lim = SBUF_LIMIT_BYTES / (TILE_SIZE_P_MAX_LOGICAL / tile_k) * 1.5

        for tile_n in tile_n_options:
            tiles_in_n = max(1, effective_n // tile_n)
            # BN options
            bn_options = _dedup_positive([tiles_in_n, tiles_in_n // 2, min(2, tiles_in_n), min(4, tiles_in_n)])

            for bm, bn, bk in product(bm_options, bn_options, bk_options):
                # Filter: BM >= BN (100% of cached optima satisfy this)
                if bm < bn:
                    continue

                # SBUF filter: reject if config doesn't fit for mxfp8 (most permissive)
                if not _fits_sbuf(bm, bn, bk, tile_m, tile_n, tile_k, 2, out_bytes, sbuf_lim):
                    continue

                # Step 3: Load tiling
                lm_options = _dedup_positive([min(bm, 8), bm, min(4, bm)])
                ln_options = _dedup_positive([bn, min(bn, 2)])

                for lm, ln in product(lm_options, ln_options):
                    # Validity checks
                    if lm > bm or ln > bn:
                        continue
                    if bm % lm != 0 or bn % ln != 0:
                        continue

                    key = (tile_n, bm, bn, bk, lm, ln, tile_m, tile_k)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Emit both spill_reload=False and spill_reload=True
                    for spill in (False, True):
                        candidates.append(
                            MatmulMxfp8KernelConfig(
                                M=M,
                                K=K,
                                N=N,
                                tile_m=tile_m,
                                tile_k=tile_k,
                                tile_n=tile_n,
                                TILES_IN_BLOCK_M=bm,
                                TILES_IN_BLOCK_N=bn,
                                TILES_IN_BLOCK_K=bk,
                                TILES_IN_LOAD_M=lm,
                                TILES_IN_LOAD_N=ln,
                                lnc_sharding=resolved_lnc_sharding,
                                spill_reload=spill,
                            )
                        )

    return candidates
