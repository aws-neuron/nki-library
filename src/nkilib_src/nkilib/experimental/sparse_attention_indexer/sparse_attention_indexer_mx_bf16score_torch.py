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

from ..mla.deepseek.mla_common_cte import MlaPrecision, RopeLayout
from .sparse_attention_indexer_torch import (
    _block32_mx_quant_dequant,
    _layernorm_ref,
    _rope_interleaved_ref,
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
    compact_scales=False,
    precision=MlaPrecision.MX,
    rope_layout=RopeLayout.HALF_SPLIT,
    qr_bf16_hbm=None,
):
    """Reference for the MX- or BF16-projections + BF16-score kernel.

    Signature mirrors ``sparse_attention_indexer_mx_bf16score``: qr is consumed
    pre-quantized via ``qr_qtz_hbm``/``qr_scale_hbm`` (an upstream QKV kernel),
    which this ref dequantizes back to ``[M, q_lora_rank]`` so the Q projection
    replays the exact MX-rounded qr the kernel matmuls. Weight tensors are the
    dequantized fp32 forms (same convention as the MX ref); activation MX quant
    noise on the projection inputs is replayed here to match the kernel.

    Only the default fused ``phase="all"`` self-attention path is modelled; the
    CP phase / topk-output-format / scale-layout flags are accepted for signature
    parity with the kernel and are otherwise unused here (the weights arrive already
    dequantized, so the compact-vs-native scale layout is a kernel-side detail).
    """
    del wq_b_scale, wk_scale, k_scale_cache, x_non_mx, x_mx_data, x_mx_scale
    del use_hadamard  # bf16-score kernel never applies Hadamard
    del k_seq_out_hbm, end_pos_arg, emit_flat_topk, emit_tiled_topk, compact_scales
    _bf16 = precision.is_bf16()
    _rope_ref = _rope_interleaved_ref if rope_layout.is_interleaved() else _rope_non_interleaved_ref
    M, dim = x.shape
    S = M // batch_size
    q_lora_rank = wq_b.shape[0]
    end_pos = start_pos + S

    # CP phased split: phase="kproj" only projects this shard's K (wk + LayerNorm + RoPE) and
    # returns it seq-major [M, head_dim] (the CP parent all-gathers it, then feeds phase="score").
    # The Q / weights / score / topk are all skipped -- this validates just the K hand-off buffer.
    do_kproj_only = phase == "kproj"
    k_seq_out = torch.zeros((M, head_dim), dtype=torch.float32) if do_kproj_only else None

    """
    qr: BF16 takes it verbatim (the bf16 QKV kernel exports plain [M, q_lora_rank]); MX
    reconstructs the MX-rounded values the kernel actually matmuls from its pre-quantized latent.
    """
    if _bf16:
        qr = qr_bf16_hbm if isinstance(qr_bf16_hbm, torch.Tensor) else torch.from_numpy(np.asarray(qr_bf16_hbm))
        qr = qr.to(torch.float32)
    else:
        qr = torch.from_numpy(_dequant_swizzled_qr(qr_qtz_hbm, qr_scale_hbm, M, q_lora_rank))
    combined_scale = (1.0 / math.sqrt(n_heads)) * (1.0 / math.sqrt(head_dim))

    """
    Prior positions [0, start_pos) are scored against the K CACHE below, not assumed zero.

    They used to be left at 0.0 on the grounds that the tests only ran start_pos == 0. Once
    start_pos > 0 cases were added that assumption silently gutted them: the test generator
    zero-fills the cache to match, so every prior column compared 0.0 against 0.0 and validated
    nothing. Measured on bf16proj_glm_skv8k_topk2k (start_pos=8064): all 1032192 prior cells were
    exactly zero, and a median 2020 of each query's top-2048 golden values were dead ties -- so
    the top-k could pick almost anything and still pass.

    Init zeros only as the accumulator; every column in [0, end_pos) is written below.
    """
    index_score = torch.zeros((M, end_pos), dtype=torch.float32)
    topk_indices = torch.zeros((M, index_topk), dtype=torch.int64)

    cos_t = cos[:S, :]
    sin_t = sin[:S, :]

    for batch_idx in range(batch_size):
        bs = batch_idx * S
        be = bs + S

        """
        Projection-input rounding. MX: block-32 quant/dequant, matching nc_matmul_mx's
        quantization of x and qr. BF16: plain bf16 rounding (the tensor engine still
        accumulates in fp32).
        """
        if _bf16:
            x_arg = x[bs:be].to(torch.bfloat16).to(torch.float32)
        else:
            x_arg = _block32_mx_quant_dequant(x[bs:be])

        # K projection + LayerNorm + RoPE.
        k = x_arg @ wk.T
        k = _layernorm_ref(k, k_norm_gamma, k_norm_beta)
        k_pe = k[:, :rope_head_dim]
        k_nope = k[:, rope_head_dim:]
        k_pe = _rope_ref(k_pe, cos_t, sin_t)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # phase="kproj": emit only the projected K (seq-major), skip Q/score/topk.
        if do_kproj_only:
            k_seq_out[bs:be] = k
            continue

        # Q projection.
        qr_arg = qr[bs:be].to(torch.bfloat16).to(torch.float32) if _bf16 else _block32_mx_quant_dequant(qr[bs:be])
        q = qr_arg @ wq_b
        q = q.reshape(S, n_heads, head_dim)

        # Q RoPE (per head).
        q_pe = q[:, :, :rope_head_dim]
        q_nope = q[:, :, rope_head_dim:]
        q_pe = _rope_ref(q_pe, cos_t.unsqueeze(1), sin_t.unsqueeze(1))
        q = torch.cat([q_pe, q_nope], dim=-1)

        # NO Hadamard. NO MX quant on Q/K. Cast to bf16 then back to f32 to
        # match the kernel's bf16 nc_matmul rounding.
        q_bf16 = q.to(torch.bfloat16).to(torch.float32)
        k_bf16 = k.to(torch.bfloat16).to(torch.float32)

        # Weights projection: kernel does this in bf16 with fp32 PSUM.
        x_bf16 = x[bs:be].to(torch.bfloat16).to(torch.float32)
        wp_bf16 = weights_proj.to(torch.bfloat16).to(torch.float32)
        weights = x_bf16 @ wp_bf16.T * combined_scale  # [S, n_heads]

        """
        Score: sum_h(relu(K @ q_h^T) * w_h) over [0, end_pos).

        Two K sources. Prior positions come from the cache the kernel reads (bf16, laid out
        [B, head_dim, max_seq_len], so transpose to [n_pos, head_dim] to match k_bf16); current
        positions come from the K just projected above. Rounded to bf16 like k_bf16 because the
        kernel's score matmul consumes a bf16 cache.

        MX path: its cache holds fp8 plus a separate scale cache, which this reference does not
        dequantize, so prior positions stay unmodelled there and generate_inputs keeps that cache
        zero-filled. Scoring an MX prior cache would need the k_scale_cache handling too.
        """
        score_ranges = []
        if start_pos > 0 and _bf16:
            _kc = (
                k_cache
                if isinstance(k_cache, torch.Tensor)
                else torch.from_numpy(np.ascontiguousarray(np.asarray(k_cache).astype(np.float32)))
            )
            k_prior = _kc[batch_idx, :, :start_pos].to(torch.float32).T.contiguous()
            score_ranges.append((0, start_pos, k_prior.to(torch.bfloat16).to(torch.float32)))
        score_ranges.append((start_pos, end_pos, k_bf16))
        for pos_range_start, pos_range_end, k_source in score_ranges:
            """
            Vectorized over HEADS: one [n_pos, head_dim] x [head_dim, n_heads] GEMM per query
            instead of n_heads separate matvecs. Same arithmetic, S iterations instead of
            S * n_heads -- 32x fewer at GLM's 32 heads, which took the [2048, 65536] golden from
            ~9.5 min to well under a minute. Kept per-query rather than fully batched because a
            [S, n_heads, n_pos] logits tensor would be 16 TB at the deployment shape.
            """
            for m in range(S):
                logits = k_source @ q_bf16[m].transpose(0, 1)  # [n_pos, n_heads]
                logits = torch.relu(logits)
                index_score[bs + m, pos_range_start:pos_range_end] = (logits * weights[m]).sum(dim=-1)

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
