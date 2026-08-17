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

"""MXFP8-accurate golden reference for attention_mxfp8_tkg kernel.

Models all quantization sources in the kernel's compute path:
Q round-trip, BF16 score truncation, online softmax chunking, score re-quantization.

For a pure FP32 attention reference (no quantization), use:
    nkilib.core.attention.attention_tkg_torch
"""

import math

import ml_dtypes
import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch

from ...core.utils.mx_torch_common import quantize_to_mx, unpack_float8_e4m3fn_x4

# Block geometry constants (match TileParams in attention_mxfp8_tkg)
_P_PER_BLOCK = 32
_BLOCK_LEN = 128
_D_HEAD = 128
_PACKED_COLS = 160  # 128 data cols (x4 as fp32) + 32 scale cols (4 uint8 per fp32)
_CHUNK_TOKENS = 2048
_BLOCKS_PER_FOLD = 4
_FOLDS_PER_CHUNK = 4
_BLOCKS_PER_CHUNK = _BLOCKS_PER_FOLD * _FOLDS_PER_CHUNK
_SCORE_NEG_INF = -65504.0  # matches the kernel's select_reduce on_false value
_RUNNING_MAX_INIT = -1e38  # matches the kernel's running-max initialization sentinel


def attention_mxfp8_tkg_torch_ref(
    q,
    k_active,
    v_active,
    k_prior,
    v_prior,
    mask,
    identity_hbm=None,
    active_blocks_table=None,
    sbm=None,
):
    """MXFP8-accurate golden reference matching attention_mxfp8_tkg kernel signature.

    Unpacks the packed MXFP8 KV cache blocks, gathers them per batch through
    active_blocks_table, and computes chunked online-softmax attention modeling
    the kernel's quantization path (Q round-trip, BF16 score truncation, score
    re-quantization, mask application, active-token update).

    Args:
        q: Query tensor [B, H, 1, d] bfloat16/float32.
        k_active: Active key [B, d] bfloat16/float32.
        v_active: Active value [B, d] bfloat16/float32.
        k_prior: MXFP8 K cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        v_prior: MXFP8 V cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
        mask: Per-head token mask [B, H, 1, s_prior] uint8.
        identity_hbm: [128, 128] bfloat16 identity matrix (unused by ref).
        active_blocks_table: Block indices [B, num_blocks] int32. None = sequential.
        sbm: Optional SbufManager (unused by ref).

    Returns:
        dict with 'out_hbm': [B, H, d] bfloat16 attention output.
    """
    q_np = _to_fp32(q)
    k_active_np = _to_fp32(k_active)
    v_active_np = _to_fp32(v_active)
    k_prior_np = _to_fp32(k_prior).reshape(-1, _P_PER_BLOCK, _PACKED_COLS)
    v_prior_np = _to_fp32(v_prior).reshape(-1, _P_PER_BLOCK, _PACKED_COLS)
    mask_np = np.asarray(mask)  # [B, H, 1, s_prior]

    bs, n_heads, _, d_head = q_np.shape
    scale = 1.0 / math.sqrt(d_head)
    num_blocks = k_prior_np.shape[0]
    num_chunks = num_blocks // _BLOCKS_PER_CHUNK

    # Unpack K and V blocks back to [num_blocks, BLOCK_LEN, d_head]
    k_unpacked = np.zeros((num_blocks, _BLOCK_LEN, d_head), dtype=np.float32)
    v_unpacked = np.zeros((num_blocks, _BLOCK_LEN, d_head), dtype=np.float32)
    for block_idx in range(num_blocks):
        k_unpacked[block_idx] = _unpack_mx_block(k_prior_np[block_idx], d_head, "d_head")
        v_unpacked[block_idx] = _unpack_mx_block(v_prior_np[block_idx], d_head, "block_len")

    # Q MXFP8 round-trip (kernel quantizes scaled Q for the block matmuls)
    q_scaled = q_np[:, :, 0, :] * scale
    q_rt = np.zeros((bs, n_heads, d_head), dtype=np.float32)
    for batch_idx in range(bs):
        q_rt[batch_idx] = mxfp8_round_trip_2d(q_scaled[batch_idx])

    result = np.zeros((bs, n_heads, d_head), dtype=np.float32)
    for batch_idx in range(bs):
        # Gather this batch's blocks through the indirection table
        if active_blocks_table is not None:
            table_b = np.asarray(active_blocks_table)[batch_idx].astype(np.int64)
        else:
            table_b = np.arange(num_blocks, dtype=np.int64)
        k_flat = k_unpacked[table_b].reshape(-1, d_head)
        v_flat = v_unpacked[table_b].reshape(-1, d_head)

        mask_b = mask_np[batch_idx, :, 0, :]  # [H, s_prior]
        s_prior = mask_b.shape[1]

        running_max = np.full((n_heads, 1), _RUNNING_MAX_INIT, dtype=np.float32)
        running_sum = np.zeros((n_heads, 1), dtype=np.float32)
        running_out = np.zeros((n_heads, d_head), dtype=np.float32)

        for chunk_idx in range(num_chunks):
            t_start = chunk_idx * _CHUNK_TOKENS
            t_end = t_start + _CHUNK_TOKENS
            k_chunk = k_flat[t_start:t_end]
            v_chunk = v_flat[t_start:t_end]

            # Slice mask for this chunk; pad with 0 if chunk extends beyond s_prior
            valid_chunk = np.zeros((n_heads, _CHUNK_TOKENS), dtype=np.uint8)
            copy_end = min(t_end, s_prior)
            if t_start < s_prior:
                valid_chunk[:, : copy_end - t_start] = mask_b[:, t_start:copy_end]

            scores = np.einsum("hd,td->ht", q_rt[batch_idx], k_chunk)
            # Kernel evicts MM1 scores to BF16 with masked positions at -65504
            scores = np.where(valid_chunk.astype(bool), scores, _SCORE_NEG_INF)
            scores = scores.astype(ml_dtypes.bfloat16).astype(np.float32)

            m_local = scores.max(axis=-1, keepdims=True)
            m_new = np.maximum(running_max, m_local)
            correction = np.exp(running_max - m_new)

            running_out = running_out * correction
            running_sum = running_sum * correction

            exp_scores = np.exp(scores - m_new)
            running_sum = running_sum + exp_scores.sum(axis=-1, keepdims=True)

            # Kernel re-quantizes exp(scores) to MXFP8 before the V matmul
            exp_scores_rt = mxfp8_round_trip_2d(exp_scores)

            running_out = running_out + np.einsum("ht,td->hd", exp_scores_rt, v_chunk)
            running_max = m_new

        # Active-token update: scalar softmax step with the un-quantized scaled Q
        score_new = q_scaled[batch_idx] @ k_active_np[batch_idx]  # [n_heads]
        m_new = np.maximum(running_max[:, 0], score_new)
        correction = np.exp(running_max[:, 0] - m_new)
        running_out = running_out * correction[:, np.newaxis]
        running_sum = running_sum * correction[:, np.newaxis]
        exp_new = np.exp(score_new - m_new)
        running_sum = running_sum + exp_new[:, np.newaxis]
        running_out = running_out + exp_new[:, np.newaxis] * v_active_np[batch_idx][np.newaxis, :]

        result[batch_idx] = running_out / running_sum

    return {"out_hbm": dt.static_cast(result.astype(np.float32), ml_dtypes.bfloat16)}


def _to_fp32(tensor):
    """Convert a torch tensor or numpy array of any float dtype to fp32 numpy."""
    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()
    return np.asarray(tensor).astype(np.float32)


def _unpack_mx_block(block_packed, d_head, contraction_dim):
    """Unpack one packed MXFP8 KV block [32, 160] fp32 back to [BLOCK_LEN, d_head] fp32.

    Inverse of the host-side packing: the first 128 columns hold float8_e4m3fn_x4
    data (viewed as fp32) in the swizzled [H//4, T*4] layout; the last 32 columns
    hold the MX scales (4 uint8 per fp32 word), valid in partitions [0:4].

    Args:
        block_packed: [32, 160] float32 packed block.
        d_head: Head dimension (128).
        contraction_dim: 'd_head' for K blocks, 'block_len' for V blocks —
            selects which axis was the contraction (partition) dim when packing.

    Returns:
        [BLOCK_LEN, d_head] float32 dequantized block.
    """
    data_cols = block_packed.shape[1] * 4 // 5  # F + F//4 packed cols -> F data cols
    scale_p = _P_PER_BLOCK // 8  # 4 scale partitions for 32 data partitions

    data_words = np.ascontiguousarray(block_packed[:, :data_cols])
    decoded = unpack_float8_e4m3fn_x4(data_words).numpy()  # [32, 512]

    scale_words = np.ascontiguousarray(block_packed[:scale_p, data_cols:])
    scale_u8 = scale_words.view(np.uint8).reshape(scale_p, data_cols)  # [4, 128]

    scale_factors = np.power(2.0, scale_u8.astype(np.int32) - 127)
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)  # [32, 512]
    deq_swizzled = decoded * scale_expanded

    # Inverse of swizzle_for_mx: [H//4, T*4] -> [T, H]
    P_out, TF = deq_swizzled.shape
    T = TF // 4
    deq = deq_swizzled.reshape(P_out, T, 4).transpose(1, 0, 2).reshape(T, P_out * 4)

    # For V the packed layout was built from the transposed block
    return deq if contraction_dim == "d_head" else deq.T


def mxfp8_round_trip_2d(tensor_2d):
    """Quantize [P, F] to MXFP8 and dequantize back to FP32."""
    data_f32 = tensor_2d.astype(np.float32)
    P, F = data_f32.shape

    # Pad to multiples of 8x4 if needed
    pad_p = (8 - P % 8) % 8
    pad_f = (4 - F % 4) % 4
    if pad_p or pad_f:
        data_f32 = np.pad(data_f32, ((0, pad_p), (0, pad_f)))

    mx_data_x4, mx_scale = quantize_to_mx(data_f32, nl.float8_e4m3fn_x4)

    # Dequantize: static_cast x4 back to float32, multiply by scale
    data_unpacked = dt.static_cast(mx_data_x4, np.float32)

    scale_exp = mx_scale.astype(np.int32) - 127
    scale_exp = np.clip(scale_exp, -127, 127)
    scale_factors = np.power(2.0, scale_exp)
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)

    result = data_unpacked * scale_expanded
    return result[:P, :F]
