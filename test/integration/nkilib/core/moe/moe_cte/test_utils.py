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

"""
Test utilities for MoE BWMM MXFP4/MXFP8 CTE kernel tests.

Provides input builders and golden functions for testing the blockwise
matrix multiplication kernel with MXFP4/MXFP8 quantization.
"""

import hashlib
import math
import os
import pickle
from typing import Optional

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
)
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert

from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import (
    generate_token_position_to_id_and_experts,
    get_n_blocks,
    map_skip_mode,
)
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.utils.mx_utils import dequantize_mx_golden

# MXFP4 quantization block dimensions
_q_width = 4  # quantization width
_q_height = 8  # quantization height
_pmax = 128  # sbuf max partition dim (128)

# --- Scale packing constants -----------------------------------------------
# Each MX scale tile occupies a 4-partition stripe in each 32-partition
# quadrant. The top half of a quadrant (16 partitions) can hold up to 4 such
# stripes, so a single packed buffer carries up to 4 tiles' scales. With more
# tiles, we use multiple packed buffers; the last one may be partially filled.
QUADRANT_SIZE = 32
N_QUADRANTS = _pmax // QUADRANT_SIZE  # 4
P_SCALE = _pmax // _q_height  # 16 source rows per scale tile
ROWS_PER_SLOT = P_SCALE // N_QUADRANTS  # 4 partitions per slot
SLOTS_PER_PACKED_BUFFER = (QUADRANT_SIZE // 2) // ROWS_PER_SLOT  # 4 tiles per packed buffer


def n_packed_buffers_for(n_tiles: int) -> int:
    """Number of packed buffers needed to hold n_tiles scale tiles."""
    return (n_tiles + SLOTS_PER_PACKED_BUFFER - 1) // SLOTS_PER_PACKED_BUFFER


def build_prequantized_hidden_concat(hidden_T: int, n_H512_tile: int, H: int, expert_affinities=None):
    """Build the pre-quantized fp8 hidden-state concat tensor for the fp8-hidden kernel path.

    Mirrors the packed layout rmsnorm_mx_prefill produces: each token row is
        [ hidden_quant (H fp8 bytes) | hidden_scale (packed scale_region uint8 MX-scale bytes) ]
    where scale_region = ceil(n_H512_tile/4)*128. The uint8 scales are folded 4-of-32 per quadrant
    AND 4 H512 tiles per 128-wide block (tile k at within-quadrant offset (k%4)*4 in pack k//4),
    matching the kernel's packed transpose_fp8_hidden_states + is_packed_moving_scale matmul.

    When expert_affinities ([hidden_T, E] bf16) is given, the dense affinity vector is appended after
    the scale region as bf16 reinterpreted into fp8 columns, and the whole row is padded to a multiple
    of 4 fp8 columns (the kernel's hidden fp32-reinterpret transpose requires it) -- exactly the
    rmsnorm_mx_prefill pack_affinities layout. The kernel then extracts each block's expert column
    on-chip instead of a separate affinity gather.

    Returns:
        hidden_concat (np.ndarray): [hidden_T, H + scale_region (+ affin tail)] viewed as fp8_e4m3fn.
        hidden_fp32 (np.ndarray): [hidden_T, H] dequantized fp32 hidden (for the torch reference,
            so kernel and golden compare the same numbers).
    """
    mx_unpacked_dtype = nl.float8_e4m3fn

    # Same generator the bf16 path uses; we keep BOTH the fp8 data + uint8 scales this time.
    hidden_fp32_blk, hidden_quant, hidden_scale_tmp = generate_stabilized_mx_data(
        mx_dtype=nl.float8_e4m3fn_x4,
        shape=(hidden_T * n_H512_tile * _pmax, _q_width),
        val_range=5,
    )

    # Dequantized fp32 hidden in the natural [hidden_T, H] layout for the reference.
    hidden_fp32 = (
        dequantize_mx_golden(hidden_quant, hidden_scale_tmp)
        .reshape(hidden_T, n_H512_tile, _pmax, _q_width)
        .transpose(0, 3, 1, 2)
        .reshape(hidden_T, H)
        .astype(np.float32)
    )

    # fp8 quant region: [hidden_T, n_H512_tile * 128_H] bytes (= H fp8 bytes per row).
    hidden_quant = hidden_quant.reshape(hidden_T, n_H512_tile * _pmax).view(mx_unpacked_dtype)

    # uint8 scale region (PACKED): fold 4 H512 tiles into one 128-wide block. Tile k lives in pack
    # k//4 at within-quadrant offset (k%4)*4 (4 valid rows per 32-col quadrant). scale_region =
    # n_packed * 128. hidden_scale_tmp is [hidden_T * n_H512_tile * 16, 1] -> [16, n_H512_tile, T].
    n_packed = n_packed_buffers_for(n_H512_tile)
    hidden_scale_tmp = (
        hidden_scale_tmp.reshape(hidden_T, n_H512_tile, _pmax // _q_height).transpose(2, 1, 0)  # [16, n_H512_tile, T]
    )
    hidden_scale = np.zeros((_pmax, n_packed, hidden_T), dtype=np.uint8)  # [128, n_packed, T]
    for tile_idx in range(n_H512_tile):
        pack_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
        slot_idx = tile_idx % SLOTS_PER_PACKED_BUFFER
        for quadrant in range(N_QUADRANTS):
            dst = QUADRANT_SIZE * quadrant + slot_idx * ROWS_PER_SLOT
            src = quadrant * ROWS_PER_SLOT
            hidden_scale[dst : dst + ROWS_PER_SLOT, pack_idx, :] = hidden_scale_tmp[
                src : src + ROWS_PER_SLOT, tile_idx, :
            ]
    hidden_scale = hidden_scale.transpose(2, 1, 0).reshape(hidden_T, n_packed * _pmax).view(mx_unpacked_dtype)

    hidden_concat = np.concatenate((hidden_quant, hidden_scale), axis=1)

    if expert_affinities is not None:
        # Append the dense [hidden_T, E] affinities as bf16 reinterpreted into fp8 columns, then pad
        # the whole row to a multiple of 4 fp8 cols (matches rmsnorm_mx_prefill pack_affinities).
        affin_fp8 = expert_affinities.astype(nl.bfloat16).view(mx_unpacked_dtype)  # [hidden_T, E*2]
        row = np.concatenate((hidden_concat, affin_fp8), axis=1)
        pad = (-row.shape[1]) % 4
        if pad:
            row = np.concatenate((row, np.zeros((hidden_T, pad), dtype=mx_unpacked_dtype)), axis=1)
        hidden_concat = row

    return hidden_concat, hidden_fp32


def dequant_prequantized_hidden_concat(hidden_concat, H: int):
    """Inverse of build_prequantized_hidden_concat: recover fp32 [T, H] hidden for the reference.

    Splits the concat [T, H + scale_region] back into the fp8 quant region and the PACKED uint8
    scale region (scale_region = ceil(n_H512_tile/4)*128), reverses the 4-tiles-per-128-block +
    4-of-32 quadrant fold, and dequantizes via dequantize_mx_golden so the torch reference sees the
    exact same numbers the kernel's matmul consumes.
    """
    hidden_T = hidden_concat.shape[0]
    n_H512_tile = H // (_pmax * _q_width)
    n_packed = n_packed_buffers_for(n_H512_tile)
    scale_region = n_packed * _pmax

    # fp8 quant region [T, H] -> x4 [T*n_H512*128, 1].
    quant_region = hidden_concat[:, :H]
    hidden_quant_x4 = quant_region.view(nl.float8_e4m3fn_x4).reshape(hidden_T * n_H512_tile * _pmax, 1)

    # PACKED uint8 scale region [T, scale_region] -> reverse the fold -> [T*n_H512*16, 1].
    scale_bytes = hidden_concat[:, H : H + scale_region].view(np.uint8)
    scale_packed = scale_bytes.reshape(hidden_T, n_packed, _pmax).transpose(2, 1, 0)  # [128, n_packed, T]
    scale_unpacked = np.zeros((_pmax // _q_height, n_H512_tile, hidden_T), dtype=np.uint8)  # [16, n_H512, T]
    for tile_idx in range(n_H512_tile):
        pack_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
        slot_idx = tile_idx % SLOTS_PER_PACKED_BUFFER
        for quadrant in range(N_QUADRANTS):
            src = QUADRANT_SIZE * quadrant + slot_idx * ROWS_PER_SLOT
            dst = quadrant * ROWS_PER_SLOT
            scale_unpacked[dst : dst + ROWS_PER_SLOT, tile_idx, :] = scale_packed[
                src : src + ROWS_PER_SLOT, pack_idx, :
            ]
    hidden_scale = scale_unpacked.transpose(2, 1, 0).reshape(hidden_T * n_H512_tile * (_pmax // _q_height), 1)

    hidden_fp32 = (
        dequantize_mx_golden(hidden_quant_x4, hidden_scale)
        .reshape(hidden_T, n_H512_tile, _pmax, _q_width)
        .transpose(0, 3, 1, 2)
        .reshape(hidden_T, H)
        .astype(np.float32)
    )
    return hidden_fp32


def _scatter_to_packed_gate_up(scale_std: np.ndarray) -> np.ndarray:
    """Pack gate/up weight scale from standard to packed HBM layout.

    Args:
        scale_std: uint8[E, P_SCALE=16, 2, n_H512_tile, I]
    Returns:
        uint8[E, _pmax=128, n_packed_buffers, 2, I]

    Layout: H/512 tile `tile_idx` lives in packed buffer
        packed_buffer_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
    at within-quadrant partition offset
        slot_idx = tile_idx %  SLOTS_PER_PACKED_BUFFER
    occupying partitions [quadrant_idx*32 + slot_idx*4 : +4] for each of the
    4 quadrants. Bottom half of each quadrant (and unused slots in the last
    packed buffer) stays zero.
    """
    E, p_scale, two, n_H512_tile, I = scale_std.shape
    assert p_scale == P_SCALE and two == 2, f"unexpected shape {scale_std.shape}"
    n_packed = n_packed_buffers_for(n_H512_tile)
    packed = np.zeros((E, _pmax, n_packed, 2, I), dtype=np.uint8)
    for tile_idx in range(n_H512_tile):
        packed_buffer_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
        slot_idx = tile_idx % SLOTS_PER_PACKED_BUFFER
        for quadrant_idx in range(N_QUADRANTS):
            dst_partition_start = quadrant_idx * QUADRANT_SIZE + slot_idx * ROWS_PER_SLOT
            src_partition_start = quadrant_idx * ROWS_PER_SLOT
            packed[:, dst_partition_start : dst_partition_start + ROWS_PER_SLOT, packed_buffer_idx, :, :] = scale_std[
                :, src_partition_start : src_partition_start + ROWS_PER_SLOT, :, tile_idx, :
            ]
    return packed


def gather_from_packed_gate_up(scale_packed: np.ndarray, n_H512_tile: int) -> np.ndarray:
    """Inverse of _scatter_to_packed_gate_up. Reads only the populated stripes;
    zero-pad regions of the packed buffer are not consulted.

    Args:
        scale_packed: uint8[E, _pmax=128, n_packed_buffers, 2, I]
        n_H512_tile:  number of real H/512 tiles in the source (the last
                      packed buffer may have unused slots beyond this count)
    Returns:
        uint8[E, P_SCALE=16, 2, n_H512_tile, I]
    """
    E, p, n_packed, two, I = scale_packed.shape
    assert p == _pmax and two == 2, f"unexpected packed shape {scale_packed.shape}"
    out = np.zeros((E, P_SCALE, 2, n_H512_tile, I), dtype=np.uint8)
    for tile_idx in range(n_H512_tile):
        packed_buffer_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
        slot_idx = tile_idx % SLOTS_PER_PACKED_BUFFER
        for quadrant_idx in range(N_QUADRANTS):
            src_partition_start = quadrant_idx * QUADRANT_SIZE + slot_idx * ROWS_PER_SLOT
            dst_partition_start = quadrant_idx * ROWS_PER_SLOT
            out[:, dst_partition_start : dst_partition_start + ROWS_PER_SLOT, :, tile_idx, :] = scale_packed[
                :, src_partition_start : src_partition_start + ROWS_PER_SLOT, packed_buffer_idx, :, :
            ]
    return out


def gather_from_packed_down(scale_packed: np.ndarray, n_I512_tile: int, p_scale: int = P_SCALE) -> np.ndarray:
    """Inverse of _scatter_to_packed_down.

    Args:
        scale_packed: uint8[E, _pmax=128, n_packed_buffers, H]
        n_I512_tile:  number of real I/512 tiles in the source
        p_scale:      number of source partition rows in the standard layout
                      (= I_TP_par_dim // _q_height; ≤ P_SCALE=16)
    Returns:
        uint8[E, p_scale, n_I512_tile, H]
    """
    E, p, n_packed, H_ = scale_packed.shape
    assert p == _pmax, f"unexpected packed shape {scale_packed.shape}"
    n_quadrants_filled = p_scale // ROWS_PER_SLOT
    out = np.zeros((E, p_scale, n_I512_tile, H_), dtype=np.uint8)
    for tile_idx in range(n_I512_tile):
        packed_buffer_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
        slot_idx = tile_idx % SLOTS_PER_PACKED_BUFFER
        for quadrant_idx in range(n_quadrants_filled):
            src_partition_start = quadrant_idx * QUADRANT_SIZE + slot_idx * ROWS_PER_SLOT
            dst_partition_start = quadrant_idx * ROWS_PER_SLOT
            out[:, dst_partition_start : dst_partition_start + ROWS_PER_SLOT, tile_idx, :] = scale_packed[
                :, src_partition_start : src_partition_start + ROWS_PER_SLOT, packed_buffer_idx, :
            ]
    return out


def _scatter_to_packed_down(scale_std: np.ndarray) -> np.ndarray:
    """Pack down weight scale from standard to packed HBM layout.

    Args:
        scale_std: uint8[E, p_scale, n_total_I512_tile, H]
            Down-projection scale; no gate/up axis. ``p_scale`` is
            ``I_TP_par_dim // _q_height`` and may be ≤ P_SCALE=16 when
            ``I_TP_par_dim < _pmax``. Quadrants beyond what fits in the
            available source rows are zero-padded (the matmul OOB-skips them).
    Returns:
        uint8[E, _pmax=128, n_packed_buffers, H]

    Layout: I/512 tile `tile_idx` lives in packed buffer
        packed_buffer_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
    at within-quadrant partition offset
        slot_idx = tile_idx %  SLOTS_PER_PACKED_BUFFER
    same as the gate/up packing.
    """
    E, p_scale, n_I512_tile, H_ = scale_std.shape
    n_quadrants_filled = p_scale // ROWS_PER_SLOT
    assert n_quadrants_filled * ROWS_PER_SLOT == p_scale, (
        f"down scale packing requires p_scale to be a multiple of ROWS_PER_SLOT={ROWS_PER_SLOT}, got {p_scale}"
    )
    n_packed = n_packed_buffers_for(n_I512_tile)
    packed = np.zeros((E, _pmax, n_packed, H_), dtype=np.uint8)
    for tile_idx in range(n_I512_tile):
        packed_buffer_idx = tile_idx // SLOTS_PER_PACKED_BUFFER
        slot_idx = tile_idx % SLOTS_PER_PACKED_BUFFER
        # Only fill the quadrants that have source data; the rest stay zero.
        for quadrant_idx in range(n_quadrants_filled):
            dst_partition_start = quadrant_idx * QUADRANT_SIZE + slot_idx * ROWS_PER_SLOT
            src_partition_start = quadrant_idx * ROWS_PER_SLOT
            packed[:, dst_partition_start : dst_partition_start + ROWS_PER_SLOT, packed_buffer_idx, :] = scale_std[
                :, src_partition_start : src_partition_start + ROWS_PER_SLOT, tile_idx, :
            ]
    return packed


# --- Block-128 (DeepSeek ue8m0) scale helpers ------------------------------
# Block-128 packs one uint8 per 128(K) x 128(N) weight block. The kernel
# materializes it into the same SBUF layout the native MX path produces, where
# each 32-partition quadrant = 128 contiguous K-elements = one block-128 K-row,
# and each scale repeats across 128 N-columns. The native-layout HBM scale has
# P_SCALE=16 K-rows (4 per quadrant); block-128 collapses those 4 rows into one
# per quadrant, hence 4 block-rows total.
SCALE_BLOCK = 128


def native_to_block128_gate_up(scale_std: np.ndarray) -> np.ndarray:
    """Subsample native gate/up scale into compact block-128 layout.

    Args:
        scale_std: uint8[E, P_SCALE=16, 2, n_H512_tile, I]
    Returns:
        uint8[E, N_QUADRANTS=4, 2, n_H512_tile, ceil(I/128)]
    Picks one representative scale per 128(K) x 128(N) block: native K-row
    ``4*q`` (first row of quadrant q) and native N-column ``nb*128``.
    """
    E, p_scale, two, n_H512_tile, I = scale_std.shape
    assert p_scale == P_SCALE and two == 2, f"unexpected shape {scale_std.shape}"
    n_blocks = (I + SCALE_BLOCK - 1) // SCALE_BLOCK
    out = np.zeros((E, N_QUADRANTS, 2, n_H512_tile, n_blocks), dtype=np.uint8)
    for q in range(N_QUADRANTS):
        for nb in range(n_blocks):
            out[:, q, :, :, nb] = scale_std[:, q * ROWS_PER_SLOT, :, :, nb * SCALE_BLOCK]
    return out


def block128_to_native_gate_up(scale_blk: np.ndarray, n_H512_tile: int, I: int) -> np.ndarray:
    """Expand compact block-128 gate/up scale to the native [E,16,2,n_H512,I] layout
    that the kernel materializes (each block-row repeated across its 4 quadrant
    K-rows, each block scale repeated across 128 N-columns)."""
    E, n_q, two, _, n_blocks = scale_blk.shape
    assert n_q == N_QUADRANTS and two == 2, f"unexpected block128 shape {scale_blk.shape}"
    out = np.zeros((E, P_SCALE, 2, n_H512_tile, I), dtype=np.uint8)
    for row in range(P_SCALE):
        q = row // ROWS_PER_SLOT
        for n in range(I):
            out[:, row, :, :, n] = scale_blk[:, q, :, :, n // SCALE_BLOCK]
    return out


def native_to_block128_down(scale_std: np.ndarray) -> np.ndarray:
    """Subsample native down scale into compact block-128 layout.

    Args:
        scale_std: uint8[E, p_scale, n_I512_tile, H]  (p_scale = I_TP_par_dim//8)
    Returns:
        uint8[E, N_QUADRANTS=4, n_I512_tile, ceil(H/128)]
    """
    E, p_scale, n_I512_tile, H = scale_std.shape
    n_quadrants_filled = p_scale // ROWS_PER_SLOT
    n_blocks = (H + SCALE_BLOCK - 1) // SCALE_BLOCK
    out = np.zeros((E, N_QUADRANTS, n_I512_tile, n_blocks), dtype=np.uint8)
    for q in range(n_quadrants_filled):
        for nb in range(n_blocks):
            out[:, q, :, nb] = scale_std[:, q * ROWS_PER_SLOT, :, nb * SCALE_BLOCK]
    return out


def block128_to_native_down(scale_blk: np.ndarray, n_I512_tile: int, H: int, p_scale: int = P_SCALE) -> np.ndarray:
    """Expand compact block-128 down scale to native [E, p_scale, n_I512_tile, H]."""
    E, n_q, _, n_blocks = scale_blk.shape
    assert n_q == N_QUADRANTS, f"unexpected block128 down shape {scale_blk.shape}"
    out = np.zeros((E, p_scale, n_I512_tile, H), dtype=np.uint8)
    for row in range(p_scale):
        q = row // ROWS_PER_SLOT
        for n in range(H):
            out[:, row, :, n] = scale_blk[:, q, :, n // SCALE_BLOCK]
    return out


# Explicit parameter ordering per kernel variant, matching kernel function signatures.
# From bwmm_shard_on_block_mx.py::bwmm_shard_on_block_mx
_SHARD_ON_BLOCK_MX_ORDER = [
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'conditions',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'n_static_blocks',
    'n_dynamic_blocks',
    'gate_up_activations_T',
    'down_activations',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
    'use_packed_scales',
    'quantization_type',
    'gate_up_in_scale',
    'down_in_scale',
]

# From bwmm_shard_on_I_mx.py::blockwise_mm_shard_intermediate_mx
_SHARD_ON_I_MX_ORDER = [
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
    'use_block128_scales',
]

# From bwmm_shard_on_I_mx.py::blockwise_mm_shard_intermediate_mx_hybrid
_SHARD_ON_I_MX_HYBRID_ORDER = [
    'conditions',
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'num_static_block',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
    'use_block128_scales',
]

_KERNEL_INPUT_ORDER = {
    'shard_on_block_mx': _SHARD_ON_BLOCK_MX_ORDER,
    'shard_on_I_mx': _SHARD_ON_I_MX_ORDER,
    'shard_on_I_mx_hybrid': _SHARD_ON_I_MX_HYBRID_ORDER,
}


def order_kernel_input(kernel_input, variant):
    """Reorder kernel_input dict to match kernel function signature ordering.

    Args:
        kernel_input: Dict from build_moe_bwmm_mx_cte.
        variant: One of 'shard_on_block_mx', 'shard_on_I_mx', 'shard_on_I_mx_hybrid'.

    Returns:
        New dict with keys ordered to match the kernel's parameter list.
        Internal keys (prefixed with '_') are excluded.
    """
    key_order = _KERNEL_INPUT_ORDER[variant]
    ordered = {}
    for key in key_order:
        if key in kernel_input:
            ordered[key] = kernel_input[key]
    for key in kernel_input:
        if key not in ordered and not key.startswith('_'):
            ordered[key] = kernel_input[key]
    return ordered


# Golden cache directory (same pattern as test_nki_moe.py)
_GOLDEN_CACHE_DIR = os.path.expanduser('~/unit_test_input_golden_cache/moe_bwmm_mxfp4_cte')


def _compute_golden_cache_key(
    H: int,
    T: int,
    E: int,
    B: int,
    TOPK: int,
    I_TP: int,
    dtype,
    weight_dtype,
    skip_mode: int,
    bias: bool,
    activation_function,
    expert_affinities_scaling_mode,
    is_dynamic: bool,
    vnc_degree: int,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    alpha,
) -> str:
    """Compute a hash key from test parameters for caching."""
    key_data = (
        H,
        T,
        E,
        B,
        TOPK,
        I_TP,
        str(dtype),
        str(weight_dtype),
        skip_mode,
        bias,
        activation_function.value if hasattr(activation_function, 'value') else activation_function,
        expert_affinities_scaling_mode.value
        if hasattr(expert_affinities_scaling_mode, 'value')
        else expert_affinities_scaling_mode,
        is_dynamic,
        vnc_degree,
        gate_clamp_upper_limit,
        gate_clamp_lower_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        alpha,
    )
    key_str = str(key_data)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _generate_token_experts_by_count(
    T: int,
    E: int,
    num_non_zero: int,
    alpha: np.float32 = None,
) -> np.ndarray:
    """Generate a [T, E] binary matrix with exactly num_non_zero ones.

    Args:
        T: Number of tokens.
        E: Number of experts.
        num_non_zero: Total number of nonzero (token, expert) entries to place.
        alpha: Skew parameter for expert selection. Larger alpha = more skewed. None = uniform.

    Returns:
        token_experts: [T, E] binary ndarray.
    """
    assert num_non_zero <= T * E, f"num_non_zero ({num_non_zero}) cannot exceed T*E ({T * E})"

    np.random.seed(0)
    token_experts = np.zeros((T, E))

    if alpha is not None and alpha > 0:
        expert_probs = np.random.dirichlet(np.ones(E) * (1.0 / alpha))
    else:
        expert_probs = np.ones(E) / E

    placed = 0
    while placed < num_non_zero:
        t = np.random.randint(0, T)
        e = np.random.choice(E, p=expert_probs)
        if token_experts[t, e] == 0:
            token_experts[t, e] = 1
            placed += 1

    return token_experts


# Input Builder
def build_moe_bwmm_mx_cte_from_model_test_config(
    H: int,
    T: int,
    E: int,
    B: int,
    I_TP: int,
    skewness_pct: float,
    global_top_k: int,
    ep_degree: int,
    dtype=nl.bfloat16,
    weight_dtype=nl.float4_e2m1fn_x4,
    skip_mode: int = 0,
    bias: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    is_dynamic: bool = True,
    vnc_degree: int = 2,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    alpha: Optional[float] = None,
    is_shard_on_I: bool = False,
    use_packed_scales: bool = False,
    quantization_type: QuantizationType = QuantizationType.MX,
) -> dict:
    """Build input tensors for MoE BWMM MX CTE model test configs using skewness-based routing.

    Unlike build_moe_bwmm_mx_cte which uses TOPK-based routing, this function uses
    skewness_pct, global_top_k, and ep_degree to control expert affinity distribution,
    simulating realistic model routing patterns.

    Args:
        H: Hidden dimension size
        T: Total number of tokens
        E: Number of local experts (after EP sharding)
        B: Block size (tokens per block)
        I_TP: Intermediate size per TP degree
        skewness_pct: Float in [0.0, 1.0] that interpolates between the best-case and
            worst-case number of nonzero expert affinities:

                num_non_zero = best + skewness_pct * (worst - best)

            where:
                best  = T * global_top_k / ep_degree   (perfectly balanced across EP shards)
                worst = T * min(E, global_top_k)        (all tokens routed to every local expert)

            Examples (T=4096, global_top_k=8, total_experts=128):
                EP64 (E=2):
                    best=512, worst=8192
                    skew 0.0 → 512,  skew 0.5 → 4352,  skew 1.0 → 8192
                EP8 (E=16):
                    best=4096, worst=32768
                    skew 0.0 → 4096, skew 1.0 → 32768
        global_top_k: Global top-K experts per token before EP sharding
        ep_degree: Expert parallelism degree
        dtype: Data type for activations
        weight_dtype: Data type for weights
        skip_mode: DMA skip mode (0-3)
        bias: Whether to include bias tensors
        activation_function: Activation function type
        expert_affinities_scaling_mode: Expert affinity scaling mode
        is_dynamic: Whether to use dynamic loop
        vnc_degree: LNC sharding degree
        gate_clamp_upper_limit: Upper clamp limit for gate projection
        gate_clamp_lower_limit: Lower clamp limit for gate projection
        up_clamp_upper_limit: Upper clamp limit for up projection
        up_clamp_lower_limit: Lower clamp limit for up projection
        alpha: Expert distribution skew parameter for _generate_token_experts_by_count
        is_shard_on_I: Whether to use shard-on-I variant

    Returns:
        Dictionary with all kernel input tensors and parameters
    """
    np.random.seed(0)

    dma_skip = map_skip_mode(skip_mode)
    is_block_parallel = not is_shard_on_I

    # Compute N (total blocks) for skewness-based routing
    n_block_per_iter_eff = vnc_degree if is_block_parallel else 1
    N = math.ceil((T * min(E, global_top_k) - (E - 1)) / B) + E - 1
    N = n_block_per_iter_eff * math.ceil(N / n_block_per_iter_eff)

    # Compute num_non_zero expert affinities based on skewness
    best = T * global_top_k // ep_degree
    worst = T * min(E, global_top_k)
    num_non_zero = int(best + skewness_pct * (worst - best))

    # Generate token-expert assignments using count-based method
    token_experts = _generate_token_experts_by_count(T, E, num_non_zero, alpha)

    blocks_per_expert = np.ceil(token_experts.sum(0) / B).astype(np.int32)
    n_padding_block = N - np.sum(blocks_per_expert)
    blocks_per_expert[E - 1] += n_padding_block

    cumulative_blocks_per_expert = np.cumsum(blocks_per_expert)
    block_to_expert = np.arange(E).repeat(blocks_per_expert).astype(np.int32)

    token_position_by_id_and_expert = np.cumsum(token_experts, axis=0)
    expert_block_offsets = cumulative_blocks_per_expert * B
    token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
    token_position_by_id_and_expert = np.where(token_experts, token_position_by_id_and_expert, 0).astype(np.int32)

    if dma_skip.skip_token:
        token_position_to_id = np.full((int(N * B + 1),), -1)
    else:
        token_position_to_id = np.full((int(N * B + 1),), T)

    tokens_ids = np.arange(T)
    token_position_to_id[token_position_by_id_and_expert] = np.expand_dims(tokens_ids, 1)
    token_position_to_id = token_position_to_id[1:]
    token_position_to_id = token_position_to_id.astype(np.int32)

    # Generate conditions
    if not is_block_parallel:
        conditions = np.ones((N + 1,), dtype=np.int32)
        conditions[-(n_padding_block + 1) :] = 0
    else:
        conditions = np.ones((N + 2,), dtype=np.int32)
        conditions[-(n_padding_block + 2) :] = 0

    num_static_block = math.ceil(math.ceil(T * global_top_k / ep_degree) / B)

    return _build_kernel_input_from_routing(
        H=H,
        T=T,
        E=E,
        B=B,
        I_TP=I_TP,
        expert_masks=token_experts,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        N=N,
        dma_skip=dma_skip,
        dtype=dtype,
        weight_dtype=weight_dtype,
        bias=bias,
        activation_function=activation_function,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        is_tensor_update_accumulating=min(E, global_top_k) > 1,
        is_dynamic=is_dynamic,
        is_shard_on_I=is_shard_on_I,
        n_static_blocks=num_static_block,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        use_packed_scales=use_packed_scales,
        quantization_type=quantization_type,
    )


def build_moe_bwmm_mx_cte(
    H: int,
    T: int,
    E: int,
    B: int,
    TOPK: int,
    I_TP: int,
    dtype=nl.bfloat16,
    weight_dtype=nl.float4_e2m1fn_x4,
    skip_mode: int = 0,
    bias: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    is_dynamic: bool = False,
    vnc_degree: int = 2,
    n_dynamic_blocks: int = 55,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    alpha: Optional[float] = None,
    use_cache: bool = False,
    is_shard_on_I: bool = False,
    n_static_blocks: Optional[int] = None,
    use_packed_scales: bool = False,
    use_block128_scales: bool = False,
    quantization_type: QuantizationType = QuantizationType.MX,
    use_prequant_hidden: bool = False,
    pack_affinities_into_hidden: bool = False,
) -> dict:
    """
    Build input tensors for MoE BWMM MXFP4/MXFP8 CTE kernel testing.

    Args:
        H: Hidden dimension size
        T: Total number of tokens
        E: Number of experts
        B: Block size (tokens per block)
        TOPK: Top-K experts per token
        I_TP: Intermediate size per TP degree
        dtype: Data type for activations
        weight_dtype: Data type for weights (e.g., nl.float4_e2m1fn_x4 for MXFP4,
                     nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4 for MXFP8)
        skip_mode: DMA skip mode (0-3)
        bias: Whether to include bias tensors
        activation_function: Activation function type
        expert_affinities_scaling_mode: Expert affinity scaling mode
        is_dynamic: Whether to use dynamic loop
        vnc_degree: LNC sharding degree
        n_dynamic_blocks: Number of blocks to process with dynamic loop (default: 55)
        gate_clamp_upper_limit: Upper clamp limit for gate projection
        gate_clamp_lower_limit: Lower clamp limit for gate projection
        up_clamp_upper_limit: Upper clamp limit for up projection
        up_clamp_lower_limit: Lower clamp limit for up projection
        alpha: Expert distribution sparsity parameter (None for uniform distribution)
        use_cache: Whether to use cached inputs if available (default: False)

    Returns:
        Dictionary with all kernel input tensors and parameters
    """
    # Check for cached inputs
    cache_key = _compute_golden_cache_key(
        H,
        T,
        E,
        B,
        TOPK,
        I_TP,
        dtype,
        weight_dtype,
        skip_mode,
        bias,
        activation_function,
        expert_affinities_scaling_mode,
        is_dynamic,
        vnc_degree,
        gate_clamp_upper_limit,
        gate_clamp_lower_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        alpha,
    )
    cache_file = os.path.join(_GOLDEN_CACHE_DIR, f"input_{cache_key}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Found cached inputs in {cache_file}, reusing...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    np.random.seed(0)

    dma_skip = map_skip_mode(skip_mode)
    if is_shard_on_I:
        N = get_n_blocks(T, TOPK, E, B, n_block_per_iter=1)
    else:
        N = get_n_blocks(T, TOPK, E, B, n_block_per_iter=vnc_degree)

    # Generate token assignments
    expert_masks, token_position_to_id, block_to_expert, conditions = generate_token_position_to_id_and_experts(
        T,
        TOPK,
        E,
        B,
        dma_skip,
        N,
        vnc_degree=vnc_degree,
        alpha=alpha,
        is_block_parallel=False if is_shard_on_I else True,
        quantize=weight_dtype,
    )

    kernel_input = _build_kernel_input_from_routing(
        H=H,
        T=T,
        E=E,
        B=B,
        I_TP=I_TP,
        expert_masks=expert_masks,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        N=N,
        dma_skip=dma_skip,
        dtype=dtype,
        weight_dtype=weight_dtype,
        bias=bias,
        activation_function=activation_function,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        is_tensor_update_accumulating=TOPK != 1,
        is_dynamic=is_dynamic,
        is_shard_on_I=is_shard_on_I,
        n_dynamic_blocks=n_dynamic_blocks,
        n_static_blocks=n_static_blocks,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        use_packed_scales=use_packed_scales,
        use_block128_scales=use_block128_scales,
        quantization_type=quantization_type,
        use_prequant_hidden=use_prequant_hidden,
        pack_affinities_into_hidden=pack_affinities_into_hidden,
    )

    # Cache the generated inputs for future reuse
    if use_cache:
        try:
            os.makedirs(_GOLDEN_CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(kernel_input, f)
            print(f"Cached inputs saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache inputs to {cache_file}: {e}")

    return kernel_input


def _build_kernel_input_from_routing(
    *,
    H: int,
    T: int,
    E: int,
    B: int,
    I_TP: int,
    expert_masks,
    token_position_to_id,
    block_to_expert,
    conditions,
    N: int,
    dma_skip: SkipMode,
    dtype,
    weight_dtype,
    bias: bool,
    activation_function: ActFnType,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    is_tensor_update_accumulating: bool,
    is_dynamic: bool,
    is_shard_on_I: bool,
    n_dynamic_blocks: int = 55,
    n_static_blocks: Optional[int] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    use_packed_scales: bool = False,
    use_block128_scales: bool = False,
    quantization_type: QuantizationType = QuantizationType.MX,
    use_prequant_hidden: bool = False,
    pack_affinities_into_hidden: bool = False,
) -> dict:
    """Build kernel input tensors and dict from pre-computed routing assignments.

    This is the shared implementation used by both build_moe_bwmm_mx_cte (TOPK routing)
    and build_moe_bwmm_mx_cte_from_model_test_config (skewness routing).

    When ``use_packed_scales=True``, gate_up_proj_scale and down_proj_scale are
    handed to the kernel in packed HBM layouts:
        gate_up_proj_scale: uint8[E, _pmax=128, n_packed_gup, 2, I]
        down_proj_scale:    uint8[E, _pmax=128, n_packed_down, H]
    The kernel detects the packed layout from the partition dim (128 vs 16)
    and dispatches to the packed sub-kernel variants. The torch reference
    receives the standard (unpacked) scale arrays via ``_internal`` so it
    keeps using the existing reference path.
    """
    # Calculate MXFP4 tensor dimensions
    kernel_assert(H % (_pmax * _q_width) == 0, f"H must be divisible by {_pmax * _q_width}, got {H}")
    n_H512_tile = H // (_pmax * _q_width)

    kernel_assert(
        I_TP % (_q_height * _q_width) == 0,
        f"I_TP must be divisible by {_q_height * _q_width}, got {I_TP}",
    )
    n_I512_tile, r_I512_tile = divmod(I_TP, _pmax * _q_width)
    n_total_I512_tile = n_I512_tile + (1 if r_I512_tile > 0 else 0)
    # Match kernel's p_I logic (moe_cte_mx_utils.BWMMMXDimensionSizes.p_I):
    #   p_I = _pmax if I > 512 else I // _q_width
    # Previously used r_I512_tile which is 0 when I_TP is an exact multiple of 512,
    # producing a 0-sized partition dim that fails NkiTensor shape validation.
    I_TP_par_dim = _pmax if I_TP > _pmax * _q_width else I_TP // _q_width

    # Generate hidden states with MXFP4-compatible layout
    # When skip_token is True, we use T tokens; otherwise T+1 (with padding token)
    if dma_skip.skip_token:
        hidden_T = T
    else:
        hidden_T = T + 1  # Include padding token

    # Generate expert affinities first: the packed-affinity prequant path folds them into the hidden
    # concat row, so they must exist before build_prequantized_hidden_concat.
    if dma_skip.skip_token:
        expert_affinities_masked = np.random.random_sample([T, E]).astype(dtype)
        expert_affinities_masked = (expert_affinities_masked * expert_masks).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([T + 1, E]).astype(dtype)
        expert_affinities_masked[:T] = (expert_affinities_masked[:T] * expert_masks).astype(dtype)
        expert_affinities_masked[T] = 0  # Zero padding token affinities

    kernel_assert(
        not (pack_affinities_into_hidden and not use_prequant_hidden),
        "pack_affinities_into_hidden requires use_prequant_hidden",
    )

    hidden_states_ref_fp32 = None
    if use_prequant_hidden:
        # Pre-quantized fp8 hidden (real MX): kernel gets the concat [T, H + scale_region (+ affinity
        # tail)] fp8 tensor; the reference gets the dequantized fp32 hidden (same numbers the kernel's
        # matmul sees). When pack_affinities_into_hidden, the dense affinities ride the row tail.
        _packed_affin = expert_affinities_masked if pack_affinities_into_hidden else None
        hidden_states, hidden_states_ref_fp32 = build_prequantized_hidden_concat(
            hidden_T, n_H512_tile, H, expert_affinities=_packed_affin
        )
        if not dma_skip.skip_token:
            hidden_states[T, :] = 0
            hidden_states_ref_fp32[T, :] = 0
    else:
        hidden_states_fp32, _, _ = generate_stabilized_mx_data(
            mx_dtype=nl.float8_e4m3fn_x4,
            shape=(hidden_T * n_H512_tile * _pmax, _q_width),
            val_range=5,
        )
        hidden_states = (
            hidden_states_fp32.reshape(hidden_T, n_H512_tile, _pmax, _q_width)
            .transpose(0, 3, 1, 2)
            .reshape(hidden_T, H)
            .astype(dtype)
        )

        # Zero out padding token (only when not skipping tokens)
        if not dma_skip.skip_token:
            hidden_states[T, :] = 0

    # Generate MXFP4 gate/up projection weights
    gate_up_proj_weights_fp32, gate_up_proj_weights, gate_up_proj_scale = generate_stabilized_mx_data(
        mx_dtype=weight_dtype,
        shape=(E * _pmax, 2 * n_H512_tile * I_TP * _q_width),
    )
    gate_up_proj_weights = gate_up_proj_weights.reshape(E, _pmax, 2, n_H512_tile, I_TP)
    gate_up_proj_scale = gate_up_proj_scale.reshape(E, _pmax // _q_height, 2, n_H512_tile, I_TP)

    # Generate MXFP4 down projection weights
    down_proj_weights_fp32, down_proj_weights, down_proj_scale = generate_stabilized_mx_data(
        mx_dtype=weight_dtype,
        shape=(E * I_TP_par_dim, n_total_I512_tile * H * _q_width),
    )
    down_proj_weights = down_proj_weights.reshape(E, I_TP_par_dim, n_total_I512_tile, H)
    down_proj_scale = down_proj_scale.reshape(E, I_TP_par_dim // _q_height, n_total_I512_tile, H)

    # Zero-pad unused partitions of the remainder tile.
    # The kernel expects the HBM weight to be pre-zeroed when p_I == _pmax
    # (it only memsets when p_I < _pmax). For the remainder tile, only
    # n_par_r_I512_tile partitions carry real data.
    if r_I512_tile > 0 and I_TP_par_dim == _pmax:
        n_par_r = r_I512_tile // _q_width
        down_proj_weights_fp32_reshaped = down_proj_weights_fp32.reshape(
            E, I_TP_par_dim, n_total_I512_tile, H * _q_width
        )
        down_proj_weights_fp32_reshaped[:, n_par_r:, -1, :] = 0
        down_proj_weights_fp32 = down_proj_weights_fp32_reshaped.reshape(
            E * I_TP_par_dim, n_total_I512_tile * H * _q_width
        )
        down_proj_weights[:, n_par_r:, -1, :] = 0
        down_proj_scale[:, n_par_r // _q_height :, -1, :] = 0

    # Build kernel input dictionary in exact compiler test order
    # Order must match build_blockwise_mm input_list:
    # [hidden_states, expert_affinities, gate_and_up_proj_weights, down_proj_weights,
    #  token_position_to_id, block_to_expert]
    # then: conditions (if dynamic), bias tensors (if bias), scale tensors (if quantize)

    kernel_input = {
        'hidden_states': hidden_states,
        'expert_affinities_masked': expert_affinities_masked.reshape(-1, 1),
        'gate_up_proj_weight': gate_up_proj_weights,
        'down_proj_weight': down_proj_weights,
        'block_size': B,
        'token_position_to_id': token_position_to_id,
        'block_to_expert': block_to_expert,
        'skip_dma': dma_skip,
        'compute_dtype': dtype,
        'is_tensor_update_accumulating': is_tensor_update_accumulating,
        'expert_affinities_scaling_mode': expert_affinities_scaling_mode,
    }

    if is_dynamic and not is_shard_on_I:
        kernel_input['n_dynamic_blocks'] = n_dynamic_blocks
    if n_static_blocks is not None:
        if is_shard_on_I:
            kernel_input['num_static_block'] = n_static_blocks
        else:
            kernel_input['n_static_blocks'] = n_static_blocks

    # Add clamp limits only if they have non-None values
    if gate_clamp_upper_limit is not None:
        kernel_input['gate_clamp_upper_limit'] = gate_clamp_upper_limit
    if gate_clamp_lower_limit is not None:
        kernel_input['gate_clamp_lower_limit'] = gate_clamp_lower_limit
    if up_clamp_lower_limit is not None:
        kernel_input['up_clamp_lower_limit'] = up_clamp_lower_limit
    if up_clamp_upper_limit is not None:
        kernel_input['up_clamp_upper_limit'] = up_clamp_upper_limit

    # Add activation function after clamp limits
    kernel_input['activation_function'] = activation_function

    # Add weight_dtype to specify target MXFP format
    kernel_input['weight_dtype'] = weight_dtype

    # Add dynamic conditions BEFORE bias (matches build_blockwise_mm order)
    if is_dynamic:
        kernel_input['conditions'] = conditions

    # Add bias tensors (matches build_blockwise_mm order: gate_and_up_proj_bias, down_proj_bias)
    if bias:
        gate_and_up_proj_bias = np.random.uniform(
            -2.0625, 0.52, size=(E, I_TP_par_dim, 2, n_total_I512_tile, _q_width)
        ).astype(dtype)
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[E, H]).astype(dtype)
        kernel_input['gate_and_up_proj_bias'] = gate_and_up_proj_bias
        kernel_input['down_proj_bias'] = down_proj_bias
    else:
        kernel_input['gate_and_up_proj_bias'] = None
        kernel_input['down_proj_bias'] = None

    # Add scale tensors AFTER bias (matches build_blockwise_mm order).
    # If packing is requested, hand the kernel the packed copies. The torch
    # reference inverts the packing via gather_from_packed_* in the wrapper.
    assert not (use_packed_scales and use_block128_scales), "packed and block-128 scales are mutually exclusive"
    if use_packed_scales:
        # Both gate/up (always P_SCALE=16 source rows) and down (p_scale =
        # I_TP_par_dim // _q_height, may be ≤ 16) are supported. For down,
        # quadrants beyond what the source covers stay zero-padded.
        kernel_input['gate_up_proj_scale'] = _scatter_to_packed_gate_up(gate_up_proj_scale)
        kernel_input['down_proj_scale'] = _scatter_to_packed_down(down_proj_scale)
    elif use_block128_scales:
        # Subsample the fine-grained MX scales into compact block-128 layout for
        # the kernel. The torch reference expands the SAME compact array back to
        # the (coarse) native layout in the wrapper, so kernel and ref agree.
        kernel_input['gate_up_proj_scale'] = native_to_block128_gate_up(gate_up_proj_scale)
        kernel_input['down_proj_scale'] = native_to_block128_down(down_proj_scale)
    else:
        kernel_input['gate_up_proj_scale'] = gate_up_proj_scale
        kernel_input['down_proj_scale'] = down_proj_scale

    # Forward the scale-format flag to the kernel — only emit when requested so
    # kernel variants that don't accept the kwarg aren't broken by an unknown key.
    if use_packed_scales:
        kernel_input['use_packed_scales'] = True
    if use_block128_scales:
        kernel_input['use_block128_scales'] = True

    # ── STATIC_MX support ──
    # STATIC_MX reuses the existing weight-scale tensors instead of adding new params:
    #   gate_up_proj_scale carries the per-expert gate/up weight scales as [E, 2, 1]
    #     (idx 0 = gate, idx 1 = up); down_proj_scale carries down weight scale as [E, 1].
    #   gate_up_in_scale: [E, 1], down_in_scale: [E, 1] — per-tensor input scales (HF
    #     checkpoints emit one input_scale per (expert, proj_type)).
    # Flat scale args at the kernel signature (matches MLP / MoE TKG conventions for @nki.jit tracing).
    kernel_input['quantization_type'] = quantization_type
    kernel_input['gate_up_in_scale'] = None
    kernel_input['down_in_scale'] = None
    if quantization_type == QuantizationType.STATIC_MX:
        # Two scales serve different purposes — calibrate them independently:
        #   - in_scale: framework's static fp8 quant scale for hidden states. Must match
        #     the hidden's range so `hidden / in_scale` lands within ±fp8_max (no clipping).
        #     CTE generates bf16 hidden with val_range=5, so in_scale ≈ 5/448 ≈ 0.011.
        #     (TKG generates fp8-byte hidden directly, so it can use 1/512 here — that
        #     would over-clip our bf16 hidden.)
        #   - w_scale: framework's static fp8 quant scale for weights. Stored fp8 bytes
        #     are ±fp8_max; w_scale dequantizes them to the network's "real" weight
        #     magnitude. Use TKG's convention: 1/(2 × fp8_max).
        _W_SCALE_MAP = {
            nl.float8_e4m3fn_x4: 1.0 / 512.0,  # 1/(2×448)
            nl.float8_e5m2_x4: 1.0 / 65536.0,  # 1/(2×57344) rounded
        }
        _w_scale_max = _W_SCALE_MAP.get(weight_dtype, 1.0 / 512.0)
        _IN_SCALE_TARGET = 5.0 / 448.0  # val_range=5 / fp8_e4m3fn max
        # Reuse gate_up_proj_scale / down_proj_scale to carry the per-expert weight scales.
        kernel_input['gate_up_proj_scale'] = np.random.random_sample(size=(E, 2, 1)).astype(np.float32) * _w_scale_max
        kernel_input['down_proj_scale'] = np.random.random_sample(size=(E, 1)).astype(np.float32) * _w_scale_max
        # Per-expert input scales: each expert gets a slightly different in_scale to exercise
        # the kernel's per-expert lookup path.
        kernel_input['gate_up_in_scale'] = (_IN_SCALE_TARGET * np.random.uniform(0.8, 1.25, size=(E, 1))).astype(
            np.float32
        )
        kernel_input['down_in_scale'] = (_IN_SCALE_TARGET * np.random.uniform(0.8, 1.25, size=(E, 1))).astype(
            np.float32
        )

    # Store additional data needed for golden computation
    kernel_input['_internal'] = {
        'gate_up_proj_weights_fp32': gate_up_proj_weights_fp32,
        'down_proj_weights_fp32': down_proj_weights_fp32,
        'expert_masks': expert_masks,
        'N': N,
        'n_H512_tile': n_H512_tile,
        'n_I512_tile': n_I512_tile,
        'n_total_I512_tile': n_total_I512_tile,
        'I_TP_par_dim': I_TP_par_dim,
        'use_packed_scales': use_packed_scales,
        'quantization_type': quantization_type,
        'use_prequant_hidden': use_prequant_hidden,
        'hidden_states_ref_fp32': hidden_states_ref_fp32,
    }

    return kernel_input
