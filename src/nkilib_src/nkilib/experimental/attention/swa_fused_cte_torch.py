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

"""Torch reference for the fused GPT-OSS sliding-window-attention (SWA) block kernel.

The fused kernel spans QKV projection -> RoPE + Q-scale -> sliding-window attention
(with per-head attention sink + block KV cache prior) -> output projection, for a
single SWA layer of a TP-sharded GPT-OSS model.

This reference is the executable spec. Its parameter names must match the kernel
``swa_fused_cte`` entry exactly (UnitTestFramework.validate_torch_ref_signature),
and the returned dict keys must match the test's output_tensor_descriptor.

Layout conventions (see swa_fused_cte.py):
  hidden_states : [B, S, H]
  qkv_weight    : [H, I]      with I = (num_q_heads + 2*num_kv_heads) * d_head
                  columns ordered [Q (num_q_heads*d) | K (num_kv_heads*d) | V (num_kv_heads*d)]
  op_weight     : [num_q_heads*d_head, H]
  qkv_bias      : [1, I]                    added to the QKV projection (before RoPE/scale)
  op_bias       : [1, H]                    added to the output projection
  k_cache       : [num_blocks, num_kv_heads, block_size, d_head]  (post-RoPE K; FP8: packed to
                  [num_blocks, num_kv_heads, block_size//2, d_head, 2])
  v_cache       : [num_blocks, num_kv_heads, block_size, d_head]  (plain V; FP8: same shape, fp8 dtype)
  block_tables  : [B, max_blocks_per_seq]  int32, logical->physical block map
  cos_cache     : [B, S, d_head]           RoPE cos for the ACTIVE tokens
  sin_cache     : [B, S, d_head]           RoPE sin for the ACTIVE tokens
  sink          : [B, num_q_heads]         per (batch, q-head) sink logit
  prior_tokens  : int (compile-time)       number of valid prior tokens (<= sliding_window)

Returns dict with:
  out     : [B, S, H]
  k_cache : updated cache (active post-RoPE K scattered into blocks n_prior_blocks.. via block_tables)
  v_cache : updated cache (active plain V scattered into blocks n_prior_blocks.. via block_tables)
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch
import torch.nn.functional as F

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_fp8(t):
    return t.dtype in _FP8_DTYPES


def _fp8_max(dtype):
    return 448.0 if dtype == torch.float8_e4m3fn else 240.0


def _unpack_dequant_cache(cache, num_kv_heads, block_size, d_head, scale):
    """Packed-FP8 K cache (num_blocks, num_kv_heads, block_size//2, d_head, 2) fp8 -> float
    (num_blocks, num_kv_heads, block_size, d_head), dequantized by *scale. Inverse of _quant_pack_cache.
    Token 2i lives at [...,0], 2i+1 at [...,1]."""
    num_blocks = cache.shape[0]
    c = cache.float()  # (nb, n_kv, bs//2, d, 2)
    # Interleave the length-2 pack axis back to token order: (nb, n_kv, bs, d)
    tok = torch.zeros((num_blocks, num_kv_heads, block_size, d_head), dtype=torch.float32)
    tok[:, :, 0::2, :] = c[..., 0]
    tok[:, :, 1::2, :] = c[..., 1]
    return tok * scale


def _dequant_cache_unpacked(cache, scale):
    """Unpacked FP8 V cache (num_blocks, num_kv_heads, block_size, d_head) fp8 -> float, dequantized by
    *scale. V is stored token-major (same layout as the bf16 cache, only the dtype differs), so this is
    just a cast + scale -- no token-pair unpack. Inverse of _quant_cache_unpacked."""
    return cache.float() * scale


def _quant_pack_cache(cache_f, num_kv_heads, block_size, d_head, scale, nl_fp8_dtype):
    """Float (num_blocks, num_kv_heads, block_size, d_head) -> packed-FP8 K numpy
    (num_blocks, num_kv_heads, block_size//2, d_head, 2). Quantize clamp(x / scale, +-fp8_max) then
    pack 2 consecutive tokens into the trailing axis. Returns a numpy array cast to the FP8 dtype via
    dt.static_cast so the framework can compare against the kernel's FP8 cache (torch fp8 can't .numpy())."""
    fp8_max = 448.0 if nl_fp8_dtype == nl.float8_e4m3fn else 240.0
    # The kernel quantizes the freshly-computed bf16 K/V (clamp(x * (1/scale))); round the ref's fp32
    # values through bf16 first so the quantization input matches the kernel's bf16 precision.
    f = cache_f.to(torch.bfloat16).float().detach().cpu().numpy().astype(np.float32)
    q = np.clip(f / scale, -fp8_max, fp8_max)  # (nb, n_kv, bs, d)
    packed = np.stack([q[:, :, 0::2, :], q[:, :, 1::2, :]], axis=-1)  # (nb, n_kv, bs//2, d, 2)
    return dt.static_cast(packed, nl_fp8_dtype)


def _quant_cache_unpacked(cache_f, scale, nl_fp8_dtype):
    """Float (num_blocks, num_kv_heads, block_size, d_head) -> UNPACKED FP8 V numpy of the same
    (token-major) shape. Quantize clamp(x / scale, +-fp8_max) with no token-pair pack. Returns a numpy
    array cast to the FP8 dtype via dt.static_cast. Inverse of _dequant_cache_unpacked."""
    fp8_max = 448.0 if nl_fp8_dtype == nl.float8_e4m3fn else 240.0
    # Round the ref's fp32 values through bf16 first so the quantization input matches the kernel's bf16.
    f = cache_f.to(torch.bfloat16).float().detach().cpu().numpy().astype(np.float32)
    q = np.clip(f / scale, -fp8_max, fp8_max)  # (nb, n_kv, bs, d)
    return dt.static_cast(q, nl_fp8_dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard half-split (non-interleaved) RoPE.

    x, cos, sin : [..., S, d_head]. cos/sin are pre-duplicated across the two halves.
    rotate_half([x1, x2]) = [-x2, x1] with x1=first half, x2=second half.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotate_half * sin


def swa_fused_cte_torch_ref(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    op_weight: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    sink: torch.Tensor,
    prior_tokens: torch.Tensor,
    qkv_bias: torch.Tensor,
    op_bias: torch.Tensor,
    scale: float = 1.0,
    sliding_window: int = 128,
    block_size: int = 128,
    num_q_heads: int = 16,
    num_kv_heads: int = 2,
    d_head: int = 64,
    k_scale: torch.Tensor = None,
    v_scale: torch.Tensor = None,
) -> dict:
    # FP8 KV cache: the K cache arrives PACKED (num_blocks, num_kv_heads, block_size//2, d_head, 2) fp8;
    # the V cache arrives UNPACKED, token-major (num_blocks, num_kv_heads, block_size, d_head) fp8 (same
    # layout as the bf16 cache, only the dtype differs). Dequantize both to float for attention; the
    # returned golden caches are re-quantized (K packed, V unpacked) to match the kernel's FP8 write
    # bit-for-bit. k_scale/v_scale are per-tensor [.,.] fp32 (a single scalar). Detect FP8 by the K cache
    # rank/trailing-axis (the framework may hand the ref a float view of the fp8 tensor, so don't rely on
    # dtype): packed K is 5D with trailing axis 2, vs the bf16 K cache which is 4D. k_scale present is the
    # decisive signal.
    fp8_packed = k_scale is not None and k_cache.dim() == 5 and k_cache.shape[-1] == 2
    if fp8_packed:
        # nl FP8 dtype for the returned golden cache (e5m2 if the input was that, else e4m3fn).
        k_nl_dtype = nl.float8_e5m2 if k_cache.dtype == torch.float8_e5m2 else nl.float8_e4m3fn
        v_nl_dtype = nl.float8_e5m2 if v_cache.dtype == torch.float8_e5m2 else nl.float8_e4m3fn
        k_s = float(k_scale.flatten()[0].item())
        v_s = float(v_scale.flatten()[0].item())
        num_blocks = k_cache.shape[0]
        # Keep the ORIGINAL fp8 prior bytes: the kernel never rewrites prior blocks, so the golden must
        # reuse them verbatim (re-quantizing would double-round and mismatch by a ULP). K packed, V unpacked.
        k_packed_in = dt.static_cast(k_cache.float().detach().cpu().numpy(), k_nl_dtype)
        v_packed_in = dt.static_cast(v_cache.float().detach().cpu().numpy(), v_nl_dtype)
        k_cache = _unpack_dequant_cache(k_cache, num_kv_heads, block_size, d_head, k_s)
        v_cache = _dequant_cache_unpacked(v_cache, v_s)

    hidden_states = hidden_states.float()
    qkv_weight = qkv_weight.float()
    op_weight = op_weight.float()
    qkv_bias = qkv_bias.float()
    op_bias = op_bias.float()
    cos_cache = cos_cache.float()
    sin_cache = sin_cache.float()

    B, S, H = hidden_states.shape
    q_dim = num_q_heads * d_head
    kv_dim = num_kv_heads * d_head
    group_size = num_q_heads // num_kv_heads  # GQA: q-heads per kv-head
    # prior_tokens is a runtime [1,1] int32 tensor: how many of block 0's W slots are valid prior.
    prior_len = int(prior_tokens.flatten()[0].item())

    # --- 1. QKV projection (+ bias) ----------------------------------------
    qkv = torch.matmul(hidden_states, qkv_weight) + qkv_bias  # [B, S, I]
    q = qkv[:, :, :q_dim].reshape(B, S, num_q_heads, d_head)
    k = qkv[:, :, q_dim : q_dim + kv_dim].reshape(B, S, num_kv_heads, d_head)
    v = qkv[:, :, q_dim + kv_dim :].reshape(B, S, num_kv_heads, d_head)

    # --- 2. RoPE on Q and K (active positions), Q-scale --------------------
    cos = cos_cache.unsqueeze(2)  # [B, S, 1, d]
    sin = sin_cache.unsqueeze(2)
    q = _apply_rope(q, cos, sin) * scale
    k = _apply_rope(k, cos, sin)

    # --- 3. Scatter active (post-RoPE) K and (plain) V into the cache ------
    # The prior occupies logical blocks [0, n_prior_blocks); active token s lands in logical block
    # n_prior_blocks + s//block_size -> physical block_tables[b, ...], pos s%block_size.
    W = sliding_window
    n_prior_blocks = (prior_len + block_size - 1) // block_size
    k_out = k_cache.float().clone()
    v_out = v_cache.float().clone()
    for b in range(B):
        for s in range(S):
            lblk = n_prior_blocks + s // block_size
            blk = int(block_tables[b, lblk].item())
            off = s % block_size
            for g in range(num_kv_heads):
                k_out[blk, g, off, :] = k[b, s, g, :]
                v_out[blk, g, off, :] = v[b, s, g, :]

    total_len = prior_len + S

    # --- 4. Build prior+active K/V sequence per batch -----------------------
    # Prior absolute positions [0, prior_len) live in logical blocks [0, n_prior_blocks): position p
    # -> block p//block_size, slot p%block_size. Active = blocks n_prior_blocks.. The kernel only
    # *reads* the last prior block (the SWA window) and masks its leading invalid slots at runtime;
    # the reference attends the full prior+active sequence and lets the absolute-position window mask
    # below drop everything outside [pos-(W-1), pos], which is equivalent.
    # 5/6. Sliding-window causal attention with per-head sink, then OP proj.
    out = torch.zeros((B, S, H), dtype=torch.float32)
    for b in range(B):
        k_seq = torch.zeros((total_len, num_kv_heads, d_head), dtype=torch.float32)
        v_seq = torch.zeros((total_len, num_kv_heads, d_head), dtype=torch.float32)
        for p in range(prior_len):  # prior tokens at absolute positions [0, prior_len)
            blk = int(block_tables[b, p // block_size].item())
            off = p % block_size
            k_seq[p] = k_out[blk, :, off, :]
            v_seq[p] = v_out[blk, :, off, :]
        for s in range(S):  # active tokens at blocks n_prior_blocks..
            blk = int(block_tables[b, n_prior_blocks + s // block_size].item())
            off = s % block_size
            k_seq[prior_len + s] = k_out[blk, :, off, :]
            v_seq[prior_len + s] = v_out[blk, :, off, :]

        attn_out = torch.zeros((S, num_q_heads, d_head), dtype=torch.float32)
        # Absolute positions: active query s sits at prior_len + s.
        q_pos = torch.arange(S).unsqueeze(1) + prior_len  # [S, 1]
        k_pos = torch.arange(total_len).unsqueeze(0)  # [1, total_len]
        causal = k_pos > q_pos  # future keys masked
        window = k_pos < (q_pos - (sliding_window - 1))  # outside window masked
        mask = causal | window  # [S, total_len]

        for h in range(num_q_heads):
            g = h // group_size
            q_h = q[b, :, h, :]  # [S, d]
            k_h = k_seq[:, g, :]  # [total_len, d]
            v_h = v_seq[:, g, :]  # [total_len, d]
            scores = torch.matmul(q_h, k_h.t())  # [S, total_len] (scale already in q)
            scores = scores.masked_fill(mask, float("-inf"))
            # Append per-head sink logit as an extra denominator column.
            sink_col = sink[b, h].float().reshape(1, 1).expand(S, 1)
            scores = torch.cat([scores, sink_col], dim=-1)  # [S, total_len+1]
            weights = F.softmax(scores, dim=-1)[:, :-1]  # drop sink before PV
            attn_out[:, h, :] = torch.matmul(weights, v_h)

        # --- 7. Output projection (+ bias): [S, num_q_heads*d] @ [num_q_heads*d, H] --
        attn_flat = attn_out.reshape(S, q_dim)
        out[b] = torch.matmul(attn_flat, op_weight) + op_bias

    if fp8_packed:
        # Golden cache = original fp8 prior bytes (unchanged by the kernel) with the ACTIVE blocks
        # [n_prior_blocks:] overwritten by the quantized write-back. Only the active region is
        # re-quantized; the prior region keeps the input bytes verbatim (the kernel never rewrites it).
        # K is re-packed (5D); V is re-quantized token-major (unpacked, 4D).
        k_out = _quant_pack_cache(k_out, num_kv_heads, block_size, d_head, k_s, k_nl_dtype)
        v_out = _quant_cache_unpacked(v_out, v_s, v_nl_dtype)
        k_out[:n_prior_blocks] = k_packed_in[:n_prior_blocks]
        v_out[:n_prior_blocks] = v_packed_in[:n_prior_blocks]

    return {"out": out, "k_cache": k_out, "v_cache": v_out}
