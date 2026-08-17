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

"""PyTorch reference for the absorbed-latent MLA QKV CTE kernel.

Replays the pipeline of :func:`mla_qkv_cte_kernel` so numerics match
hardware: stage1 fused projection (MX), Q-norm + Q stage-2 matmul (MX), per-head
RoPE on q_pe, the per-head absorption matmul ``q_nope @ W_uk`` (bf16), and the
latent KV path ``c_kv = RMSNorm(kv) * gamma`` plus ``k_pe`` RoPE (no wkv_b
matmul).
"""

from typing import Dict

import ml_dtypes
import nki.language as nl
import numpy as np
import torch

from ....core.rmsnorm.rmsnorm_mx_prefill_torch import decode_packed_output
from ....core.subkernels.rmsnorm_torch import rms_norm_torch_ref
from ....core.utils.mx_torch_common import mx_matmul, quantize_to_mx, unpack_float8_e4m3fn_x4

_Q_WIDTH = 4
_PMAX = 128
_DS_SCALE_BLOCK = 128


def _broadcast_compact_scales(compact_scale, in_dim, out_dim):
    """Broadcast a DeepSeek compact block-128 scale [in//128, ceil(out/128)] to
    MX matmul layout [in//32, out]."""
    compact_np = compact_scale.cpu().numpy() if isinstance(compact_scale, torch.Tensor) else compact_scale
    full = np.repeat(compact_np, 4, axis=0)
    full = np.repeat(full, _DS_SCALE_BLOCK, axis=1)
    return full[: in_dim // 32, :out_dim].astype(np.uint8)


def mla_qkv_cte_torch_ref(
    x_hbm_mx,  # packed MX activation [B, S, H + scale_region] (rmsnorm_mx_prefill pack_scales=True)
    wqkv_a_hbm,  # MX x4 packed numpy [H//4, fused_dim]
    wqkv_a_scale_hbm,  # compact numpy [H//128, ceil(fused_dim/128)]
    wq_b_hbm,  # MX x4 packed numpy [qk_lora//4, q_out_dim]
    wq_b_scale_hbm,  # compact numpy
    q_norm_gamma_hbm: torch.Tensor,
    kv_norm_gamma_hbm: torch.Tensor,
    wuk_hbm,  # bf16 numpy/tensor [nope, n_heads*kv_lora]
    cos_cache_hbm: torch.Tensor,
    sin_cache_hbm: torch.Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float = 1e-6,
    # Optional qr export buffers, declared for kernel-signature parity; the reference does not model the export.
    qr_qtz_hbm=None,
    qr_scale_hbm=None,
) -> Dict[str, torch.Tensor]:
    """Returns dict with keys ``q_lift``, ``q_pe``, ``c_kv``, ``k_pe`` (bf16).

    Shapes:
        q_lift: [B, S, n_heads, kv_lora_rank]
        q_pe:   [B, S, n_heads, qk_rope_head_dim]
        c_kv:   [B, S, kv_lora_rank]
        k_pe:   [B, S, qk_rope_head_dim]

    Notes:
        Decodes the PACKED MX activation with the same helper the kernel's fp8 transpose
        mirrors. ``decode_packed_output`` returns NATURAL hidden order (h = h512*512 + 4p + q)
        — the order the kernel lands in x_qtz and that wqkv_a contracts against
        (pack_scales=True, folded scales). The framework upcasts the fp8 packed input to
        float32; it is re-viewed as fp8 bytes first.
    """
    x_mx_np = x_hbm_mx.cpu().numpy() if isinstance(x_hbm_mx, torch.Tensor) else np.asarray(x_hbm_mx)
    Bm, Sm, _row = x_mx_np.shape
    H = wqkv_a_hbm.shape[0] * _Q_WIDTH
    concat_fp8 = x_mx_np.reshape(Bm * Sm, _row).astype(ml_dtypes.float8_e4m3fn)
    x_nat = decode_packed_output(concat_fp8, Bm * Sm, H, pack_scales=True)
    x = torch.from_numpy(x_nat.astype(np.float32)).reshape(Bm, Sm, H)
    B, S = Bm, Sm
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    fused_out_dim = qk_lora_rank + kv_a_out_dim

    cos_cache = cos_cache_hbm.to(torch.float32)
    sin_cache = sin_cache_hbm.to(torch.float32)
    q_norm_gamma = q_norm_gamma_hbm.to(torch.float32)
    kv_norm_gamma = kv_norm_gamma_hbm.to(torch.float32)

    def _mx_matmul_torch(inp_bf16, weights_packed_np, weights_scale_np, in_dim, out_dim):
        b, s, _ = inp_bf16.shape
        hidden_np = inp_bf16.reshape(b * s, in_dim).T.numpy()
        hidden_np = (
            hidden_np.reshape(in_dim // _Q_WIDTH, _Q_WIDTH, b * s)
            .transpose(0, 2, 1)
            .reshape(in_dim // _Q_WIDTH, _Q_WIDTH * b * s)
            .astype(np.float32)
        )
        hidden_mx, hidden_scale = quantize_to_mx(hidden_np, nl.float8_e4m3fn_x4)
        hidden_mx_torch = unpack_float8_e4m3fn_x4(hidden_mx)
        weights_unpacked = unpack_float8_e4m3fn_x4(weights_packed_np)
        hidden_scale_torch = torch.from_numpy(hidden_scale.astype(np.float64))
        if isinstance(weights_scale_np, torch.Tensor):
            w_scale_torch = weights_scale_np.to(torch.float64)
        else:
            w_scale_torch = torch.from_numpy(weights_scale_np.astype(np.float64))
        result = mx_matmul(
            stationary=hidden_mx_torch,
            moving=weights_unpacked,
            stationary_scale=hidden_scale_torch,
            moving_scale=w_scale_torch,
        )
        return result.reshape(b, s, out_dim)

    wqkv_a_scale_full = _broadcast_compact_scales(wqkv_a_scale_hbm, H, fused_out_dim)
    wq_b_scale_full = _broadcast_compact_scales(wq_b_scale_hbm, qk_lora_rank, n_heads * qk_head_dim)

    # Step 1: fused QKV_A matmul.
    fused_out = _mx_matmul_torch(x, wqkv_a_hbm, wqkv_a_scale_full, H, fused_out_dim)

    def _unswizzle_cols(t, dim):
        num_512_tiles = dim // (_PMAX * _Q_WIDTH)
        idx = np.arange(dim).reshape(_Q_WIDTH, num_512_tiles, _PMAX).transpose(1, 2, 0).reshape(dim)
        return t[..., idx]

    qr_swizzled = fused_out[..., :qk_lora_rank]
    kv_raw_swizzled = fused_out[..., qk_lora_rank:]
    kv_swizzled = kv_raw_swizzled[..., :kv_lora_rank]
    k_pe = kv_raw_swizzled[..., kv_lora_rank:]  # not swizzled

    qr = _unswizzle_cols(qr_swizzled, qk_lora_rank)
    kv = _unswizzle_cols(kv_swizzled, kv_lora_rank)

    # Step 2: Q path - RMSNorm + Q_B matmul.
    qr = rms_norm_torch_ref(qr, q_norm_gamma, eps=norm_eps)
    q = _mx_matmul_torch(qr, wq_b_hbm, wq_b_scale_full, qk_lora_rank, n_heads * qk_head_dim)
    q = q.reshape(B, S, n_heads, qk_head_dim)
    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    """
    Step 3: q_pe RoPE. INTERLEAVED (adjacent-pair / complex) layout, matching the
    DeepSeek reference apply_rotary_emb(..., interleaved=True) used by the main MLA
    attention and the kernel's _apply_rope_*_interleaved (flag _USE_INTERLEAVED_ROPE).
    Pair j = (col 2j, col 2j+1) with angle theta_j:
      out[2j]   = x[2j]*cos_j - x[2j+1]*sin_j
      out[2j+1] = x[2j]*sin_j + x[2j+1]*cos_j
    cos/sin caches carry the R/2 frequencies theta_j in their first half.
    """
    half = qk_rope_head_dim // 2
    cos = cos_cache[:, :, :half].unsqueeze(2)  # [B, S, 1, half]
    sin = sin_cache[:, :, :half].unsqueeze(2)
    q_pe_even = q_pe[:, :, :, 0::2]
    q_pe_odd = q_pe[:, :, :, 1::2]
    q_out_even = q_pe_even * cos - q_pe_odd * sin
    q_out_odd = q_pe_even * sin + q_pe_odd * cos
    q_pe_rope = torch.stack([q_out_even, q_out_odd], dim=-1).flatten(-2)

    """
    Step 4: absorption q_nope @ W_uk per head (bf16).
    W_uk is [nope, n_heads * kv_lora]; head h owns columns
    [h*kv_lora, (h+1)*kv_lora). Contraction = nope. The kernel runs this in
    bf16, so round q_nope and W_uk to bf16 before the float32 matmul.
    """
    if isinstance(wuk_hbm, torch.Tensor):
        wuk_t = wuk_hbm.to(torch.bfloat16).to(torch.float32)
    else:
        wuk_t = torch.from_numpy(np.asarray(wuk_hbm)).to(torch.bfloat16).to(torch.float32)
    wuk_t = wuk_t.reshape(qk_nope_head_dim, n_heads, kv_lora_rank)  # [nope, h, kv_lora]
    q_nope_bf16 = q_nope.to(torch.bfloat16).to(torch.float32)  # [B, S, h, nope]

    q_lift = torch.zeros((B, S, n_heads, kv_lora_rank), dtype=torch.float32)
    for h in range(n_heads):
        # [B, S, nope] @ [nope, kv_lora] -> [B, S, kv_lora]
        q_lift[:, :, h, :] = q_nope_bf16[:, :, h, :] @ wuk_t[:, h, :]

    # Step 5: KV latent path - RMSNorm(kv) * gamma (no wkv_b).
    c_kv = rms_norm_torch_ref(kv, kv_norm_gamma, eps=norm_eps)

    # k_pe RoPE (shared single head). Same INTERLEAVED layout as q_pe above.
    k_pe = k_pe.unsqueeze(2)
    k_pe_even = k_pe[:, :, :, 0::2]
    k_pe_odd = k_pe[:, :, :, 1::2]
    k_out_even = k_pe_even * cos - k_pe_odd * sin
    k_out_odd = k_pe_even * sin + k_pe_odd * cos
    k_pe_rope = torch.stack([k_out_even, k_out_odd], dim=-1).flatten(-2).squeeze(2)

    return {
        "q_lift": q_lift.to(torch.bfloat16),
        "q_pe": q_pe_rope.to(torch.bfloat16),
        "c_kv": c_kv.to(torch.bfloat16),
        "k_pe": k_pe_rope.to(torch.bfloat16),
        "qr": qr.to(torch.bfloat16),
    }
