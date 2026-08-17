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

"""PyTorch reference for the SAI bf16-score kernel variant.

Mirrors ``sparse_attention_indexer_mx_bf16score``:
  * MX projection noise on Q/K activations (kernel still uses nc_matmul_mx
    for projections).
  * NO Hadamard rotation.
  * NO MX quantization on the score matmul side (Q and K stay bf16-cast).
  * Score matmul performed with bf16 inputs (cast → matmul → fp32 accum).

Imports the shared helpers from sparse_attention_indexer_torch.py so we
don't duplicate the LayerNorm / RoPE / MX-block routines.
"""

import math

import neuron_dtypes as dt
import numpy as np
import torch
from neuron_dtypes import static_cast

from .sparse_attention_indexer_torch import (
    _block32_mx_quant_dequant,
    _layernorm_ref,
    _rope_non_interleaved_ref,
)

_P_MAX = 128
_H_PACK = 4


def _dequant_swizzled_qr(qr_qtz, qr_scale, M, q_lora_rank):
    """Recover ``[M, q_lora_rank]`` fp32 qr from the kernel's pre-quantized latent.

    The upstream QKV kernel exports qr already transposed + swizzled + block-32
    MX-quantized in the ``[num_S_tiles, P_MAX, num_K_tiles, P_MAX]`` fp8x4 (viewed
    uint32) / uint8 layout the indexer matmuls. This undoes that swizzle+quant so
    the Q projection replays the exact MX-rounded qr the kernel consumes.

    Args:
        qr_qtz: ``[num_S_tiles, P_MAX, num_K_tiles, P_MAX]`` uint32 (fp8x4 as u32).
        qr_scale: ``[num_S_tiles, P_MAX, num_K_tiles, P_MAX]`` uint8 (HW-quadrant scale).
    Returns: ``[M, q_lora_rank]`` float32.
    """
    qr_qtz = qr_qtz.numpy() if hasattr(qr_qtz, "numpy") else np.asarray(qr_qtz)
    qr_scale = qr_scale.numpy() if hasattr(qr_scale, "numpy") else np.asarray(qr_scale)
    qr_qtz = qr_qtz.astype(np.uint32)
    qr_scale = qr_scale.astype(np.uint8)

    K_TILE_FREE = _P_MAX * _H_PACK  # 512
    num_K_tiles = q_lora_rank // K_TILE_FREE
    # CEIL: qr_qtz is stored in whole P_MAX tiles (the QKV kernel zero-pads a partial last
    # tile), so a CP shard with M < P_MAX still has one tile. Decode the padded rows and slice
    # back to M below (FLOOR gave 0 tiles -> reshape/decode failed for M < P_MAX).
    num_S_tiles = (M + _P_MAX - 1) // _P_MAX
    M_pad = num_S_tiles * _P_MAX
    SCALE_QUADRANT_SIZE = 32
    F32_EXP_BIAS = 127

    qr_qtz_x4 = qr_qtz.reshape(num_S_tiles, _P_MAX, num_K_tiles * _P_MAX).view(dt.float8_e4m3fn_x4)
    qr_scale_hw = qr_scale.reshape(num_S_tiles, _P_MAX, num_K_tiles * _P_MAX)

    out = np.zeros((M_pad, q_lora_rank), dtype=np.float32)
    for s_tile in range(num_S_tiles):
        # Rebuild the compact [16, num_K_tiles*P_MAX] scale from the HW-quadrant
        # rows ([0..3, 32..35, 64..67, 96..99]) the qkv export writes.
        compact = np.zeros((_P_MAX // 8, num_K_tiles * _P_MAX), dtype=np.uint8)
        for i in range(_P_MAX // 8):
            hw_row = (i // 4) * SCALE_QUADRANT_SIZE + (i % 4)
            compact[i, :] = qr_scale_hw[s_tile, hw_row, :]
        # Dequant fp8x4 -> fp32 [P_MAX, num_K_tiles*K_TILE_FREE]: static_cast
        # expands x4 [P, F//4] -> [P, F], then scale each 8x4 block by
        # 2^(e8m0 - 127).
        data_f32 = static_cast(qr_qtz_x4[s_tile], np.float32)
        scale_exp = np.clip(compact.astype(np.int32) - F32_EXP_BIAS, -127, 127)
        scale_factors = 2.0**scale_exp
        scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)
        transposed = (data_f32 * scale_expanded).reshape(_P_MAX, num_K_tiles, K_TILE_FREE)
        # Undo swizzle: x[s, k_tile*512 + 4j + h] = transposed[j, k_tile, 4s+h].
        for k_tile in range(num_K_tiles):
            base = k_tile * K_TILE_FREE
            for h_sub in range(_H_PACK):
                src_T = transposed[:, k_tile, h_sub::_H_PACK]  # [P_MAX(j), P_MAX(s)]
                out[s_tile * _P_MAX : (s_tile + 1) * _P_MAX, base + h_sub : base + K_TILE_FREE : _H_PACK] = src_T.T
    return out[:M]  # drop the padded tail rows of a partial last tile


def sparse_attention_indexer_mx_bf16score_torch_ref(
    x,
    wq_b,
    wk,
    k_norm_gamma,
    k_norm_beta,
    weights_proj,
    cos,
    sin,
    k_cache,
    mask,
    n_heads,
    head_dim,
    rope_head_dim,
    index_topk,
    start_pos,
    use_hadamard=False,
    batch_size=1,
    wq_b_scale=None,
    wk_scale=None,
    k_scale_cache=None,
    x_non_mx=None,
    x_mx_data=None,
    x_mx_scale=None,
    qr_qtz_hbm=None,
    qr_scale_hbm=None,
    phase="all",
    k_seq_out_hbm=None,
    end_pos_arg=None,
    emit_flat_topk=False,
    emit_tiled_topk=False,
):
    """Reference for the MX-projections + BF16-score kernel.

    Signature mirrors ``sparse_attention_indexer_mx_bf16score``: qr is consumed
    pre-quantized via ``qr_qtz_hbm``/``qr_scale_hbm`` (an upstream QKV kernel),
    which this ref dequantizes back to ``[M, q_lora_rank]`` so the Q projection
    replays the exact MX-rounded qr the kernel matmuls. Weight tensors are the
    dequantized fp32 forms (same convention as the MX ref); activation MX quant
    noise on the projection inputs is replayed here to match the kernel.

    Only the default fused ``phase="all"`` self-attention path is modelled; the
    CP phase / topk-output-format flags are accepted for signature parity with
    the kernel and are otherwise unused here.
    """
    del wq_b_scale, wk_scale, k_scale_cache, x_non_mx, x_mx_data, x_mx_scale
    del use_hadamard  # bf16-score kernel never applies Hadamard
    del k_seq_out_hbm, end_pos_arg, emit_flat_topk, emit_tiled_topk
    M, dim = x.shape
    S = M // batch_size
    q_lora_rank = wq_b.shape[0]
    end_pos = start_pos + S

    # CP phased split: phase="kproj" only projects this shard's K (wk + LayerNorm + RoPE) and
    # returns it seq-major [M, head_dim] (the CP parent all-gathers it, then feeds phase="score").
    # The Q / weights / score / topk are all skipped -- this validates just the K hand-off buffer.
    do_kproj_only = phase == "kproj"
    k_seq_out = torch.zeros((M, head_dim), dtype=torch.float32) if do_kproj_only else None

    # Reconstruct the MX-rounded qr the kernel consumes from its pre-quantized latent.
    qr = torch.from_numpy(_dequant_swizzled_qr(qr_qtz_hbm, qr_scale_hbm, M, q_lora_rank))
    combined_scale = (1.0 / math.sqrt(n_heads)) * (1.0 / math.sqrt(head_dim))

    # Prior positions [0, start_pos) score against the zero-filled prior K cache,
    # yielding finite (0.0) scores — the kernel scores the full [0, end_pos) range.
    # Init zeros (not -inf) so the causally-valid prior region matches the kernel.
    index_score = torch.zeros((M, end_pos), dtype=torch.float32)
    topk_indices = torch.zeros((M, index_topk), dtype=torch.int64)

    cos_t = cos[:S, :]
    sin_t = sin[:S, :]

    for batch_idx in range(batch_size):
        bs = batch_idx * S
        be = bs + S

        # MX-projection noise on activations (matches kernel's nc_matmul_mx
        # quantization of qr and x for Q-proj and K-proj).
        x_arg = _block32_mx_quant_dequant(x[bs:be])

        # K projection + LayerNorm + RoPE.
        k = x_arg @ wk.T
        k = _layernorm_ref(k, k_norm_gamma, k_norm_beta)
        k_pe = k[:, :rope_head_dim]
        k_nope = k[:, rope_head_dim:]
        k_pe = _rope_non_interleaved_ref(k_pe, cos_t, sin_t)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # phase="kproj": emit only the projected K (seq-major), skip Q/score/topk.
        if do_kproj_only:
            k_seq_out[bs:be] = k
            continue

        # Q projection.
        qr_arg = _block32_mx_quant_dequant(qr[bs:be])
        q = qr_arg @ wq_b
        q = q.reshape(S, n_heads, head_dim)

        # Q RoPE (per head).
        q_pe = q[:, :, :rope_head_dim]
        q_nope = q[:, :, rope_head_dim:]
        q_pe = _rope_non_interleaved_ref(q_pe, cos_t.unsqueeze(1), sin_t.unsqueeze(1))
        q = torch.cat([q_pe, q_nope], dim=-1)

        # NO Hadamard. NO MX quant on Q/K. Cast to bf16 then back to f32 to
        # match the kernel's bf16 nc_matmul rounding.
        q_bf16 = q.to(torch.bfloat16).to(torch.float32)
        k_bf16 = k.to(torch.bfloat16).to(torch.float32)

        # Weights projection: kernel does this in bf16 with fp32 PSUM.
        x_bf16 = x[bs:be].to(torch.bfloat16).to(torch.float32)
        wp_bf16 = weights_proj.to(torch.bfloat16).to(torch.float32)
        weights = x_bf16 @ wp_bf16.T * combined_scale  # [S, n_heads]

        # Score: sum_h(relu(K @ q_h^T) * w_h) over current positions.
        # (start_pos == 0 in current bf16-score tests, so no prior cache.)
        for pos_range_start, pos_range_end, k_source in [
            (start_pos, end_pos, k_bf16),
        ]:
            tile_size = pos_range_end - pos_range_start
            for m in range(S):
                score_acc = torch.zeros(tile_size, dtype=torch.float32)
                for h in range(n_heads):
                    logits = k_source @ q_bf16[m, h, :]  # [tile_size]
                    logits = torch.relu(logits)
                    score_acc += logits * weights[m, h]
                index_score[bs + m, pos_range_start:pos_range_end] = score_acc

        # Mask add.
        index_score[bs:be, :end_pos] = index_score[bs:be, :end_pos] + mask[bs:be, :end_pos]

        # Topk (kept for signature parity; not validated downstream).
        actual_k = min(index_topk, end_pos)
        _, topk_idx = torch.topk(index_score[bs:be, :end_pos], actual_k, dim=-1)
        topk_indices[bs:be, :actual_k] = topk_idx

    if do_kproj_only:
        # Kernel emits bf16 k_seq_out; return the projected K (validator uses bf16 tolerance).
        return {"output_0": k_seq_out}
    return {"output_0": index_score, "output_1": topk_indices.to(torch.int32)}
