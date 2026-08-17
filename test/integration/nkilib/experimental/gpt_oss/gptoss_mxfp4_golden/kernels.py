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
# SPDX-License-Identifier: Apache-2.0
"""
Pure-torch "kernel-oracle" functions for the GPT-OSS MXFP4 decode path.

These two functions mirror the call sites the production model uses
(``NF.attention_decode`` and ``NF.moe_block_tkg``) but contain only plain
``torch`` — no NKI, no vllm, single device, bf16 KV cache. They are transcribed
from the repo's own torch fallbacks:

  * ``attention_decode``  <- ``_torch_attention_decode_impl``
                             (attention_decode.py), stripped of DCP / attention-DP
                             / FP8 / packed-K branches.
  * ``moe_block_tkg``     <- ``_torch_moe_block_tkg_impl`` (moe_block_tkg.py),
                             but operating on **dense dequantized** expert weights
                             (the production torch fallback raises on MXFP4 scales;
                             the golden dequantizes up-front in loader.py). Uses an
                             HF-style per-expert token gather for speed.

A kernel author can feed identical inputs to their kernel and to these oracles
and diff the outputs.

Paged KV contract (mirrors production, so decode is also a paging oracle):
  K_cache / V_cache : [num_blocks, kv_heads, block_size, head_dim]
  block_table       : [B, max_blocks_per_seq] int32, -1 for unused blocks
  slot_mapping      : [B*S_decode] int64, absolute write slot = block*block_size + off

The gathered-buffer index ``s`` of the block-KV gather equals the *logical*
absolute position ``s`` (block_table[b][j] holds logical positions
[j*block_size, (j+1)*block_size) regardless of which physical block), which is
why fragmented and contiguous block tables produce identical logits.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

# ===========================================================================
# RoPE (contiguous half-split, matching the kernel and HF's _apply_rotary_emb)
# ===========================================================================


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_decode(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to decode Q/K.

    Args:
        q: [B, q_heads, S, head_dim]
        k: [B, kv_heads, S, head_dim]
        cos, sin: [B, S, head_dim//2] rotary tables (half-dim).

    Returns:
        (q_rot, k_rot) same shapes/dtype as inputs.

    The contiguous half-split ``q1*cos - q2*sin, q2*cos + q1*sin`` is identical
    to HF ``_apply_rotary_emb``.
    """
    cos_r = cos.unsqueeze(1)  # [B, 1, S, half_d]
    sin_r = sin.unsqueeze(1)
    cos_full = torch.cat([cos_r, cos_r], dim=-1).to(q.dtype)  # [B, 1, S, head_dim]
    sin_full = torch.cat([sin_r, sin_r], dim=-1).to(q.dtype)
    q_rot = (q * cos_full) + (_rotate_half(q) * sin_full)
    k_rot = (k * cos_full) + (_rotate_half(k) * sin_full)
    return q_rot, k_rot


# ===========================================================================
# Attention decode oracle
# ===========================================================================


def attention_decode(
    X: Tensor,  # [B, S_decode, H]
    W_qkv: Tensor,  # [H, q_size + 2*kv_size]
    bias_qkv: Optional[Tensor],  # [q_size + 2*kv_size]
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cos: Tensor,  # [B*S_decode, head_dim//2]
    sin: Tensor,  # [B*S_decode, head_dim//2]
    K_cache: Tensor,  # [num_blocks, kv_heads, block_size, head_dim]
    V_cache: Tensor,  # [num_blocks, kv_heads, block_size, head_dim]
    block_table: Tensor,  # [B, max_blocks_per_seq] int32, -1 pad
    slot_mapping: Tensor,  # [B*S_decode] int64 write slots
    pos_ids: Tensor,  # [B*S_decode] int absolute positions of active tokens
    sliding_window: Optional[int],
    sink: Optional[Tensor],  # [num_q_heads] or [num_q_heads, 1]
    softmax_scale: float,
    W_out: Tensor,  # [q_heads*head_dim, H]
    bias_out: Optional[Tensor],  # [H]
    update_cache: bool = True,
) -> Tensor:
    """Fused attention decode oracle (QKV proj -> RoPE -> paged attention with
    sinks + SWA -> KV cache write -> O proj).

    Mirrors ``_torch_attention_decode_impl`` for the single-device bf16 path.
    Returns the post-O-projection output ``[B*S_decode, H]``.

    Usage:
        >>> out = attention_decode(X, W_qkv, bias_qkv, 64, 8, 64, cos, sin,
        ...     K_cache, V_cache, block_table, slot_mapping, pos_ids,
        ...     sliding_window=128, sink=sinks, softmax_scale=1/8.0,
        ...     W_out=Wo, bias_out=bo)
    """
    compute_dtype = X.dtype
    B, S_tkg, H = X.shape
    num_blocks_total, kv_heads_cache, block_size, d_head = V_cache.shape
    assert d_head == head_dim
    num_kv_groups = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_table.shape[-1]
    S_ctx = max_blocks_per_seq * block_size

    # ── QKV projection ────────────────────────────────────────────────────
    qkv = X @ W_qkv  # [B, S, q_size+2*kv_size]
    if bias_qkv is not None:
        qkv = qkv + bias_qkv

    q_end = num_q_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim
    q = qkv[..., :q_end].view(B, S_tkg, num_q_heads, head_dim).transpose(1, 2)
    k = qkv[..., q_end:k_end].view(B, S_tkg, num_kv_heads, head_dim).transpose(1, 2)
    v = qkv[..., k_end:].view(B, S_tkg, num_kv_heads, head_dim).transpose(1, 2)

    # ── RoPE ──────────────────────────────────────────────────────────────
    cos_bshd = cos.view(B, S_tkg, head_dim // 2)
    sin_bshd = sin.view(B, S_tkg, head_dim // 2)
    q, k = apply_rope_decode(q, k, cos_bshd, sin_bshd)

    # ── Gather K/V from the paged block cache -> [B, kv_heads, S_ctx, hd] ──
    # -1 sentinels (unused blocks) are clamped to block 0; the mask masks them.
    safe_idx = torch.where(
        block_table < 0, torch.zeros_like(block_table), block_table
    ).long()  # [B, max_blocks_per_seq]
    flat_idx = safe_idx.reshape(-1)
    K_blocks = K_cache[flat_idx].to(compute_dtype)  # [B*mb, kv_heads, block_size, hd]
    V_blocks = V_cache[flat_idx].to(compute_dtype)
    K_gathered = (
        K_blocks.view(B, max_blocks_per_seq, kv_heads_cache, block_size, d_head)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, kv_heads_cache, S_ctx, d_head)
    )
    V_gathered = (
        V_blocks.view(B, max_blocks_per_seq, kv_heads_cache, block_size, d_head)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, kv_heads_cache, S_ctx, d_head)
    )

    # Place the freshly-computed active K/V into the LAST S_tkg positions of the
    # gathered buffer (the cache is not written until after attention, so the
    # active tokens' real slots hold stale data — masked out below).
    K_gathered[:, :, -S_tkg:, :] = k
    V_gathered[:, :, -S_tkg:, :] = v

    # GQA expand
    K_full = K_gathered.repeat_interleave(num_kv_groups, dim=1)  # [B, q_heads, S_ctx, hd]
    V_full = V_gathered.repeat_interleave(num_kv_groups, dim=1)

    # ── Scores ────────────────────────────────────────────────────────────
    scores = torch.matmul(q, K_full.transpose(-2, -1)) * softmax_scale  # [B,qh,S,S_ctx]
    scores = scores.to(torch.float32)

    # ── Mask (logical, un-shuffled) ───────────────────────────────────────
    mask = _build_decode_mask(
        pos_ids=pos_ids,
        B=B,
        S_tkg=S_tkg,
        S_ctx=S_ctx,
        sliding_window=sliding_window,
        device=X.device,
    )  # [B, 1, S_tkg, S_ctx] in {0,1}
    scores = scores.masked_fill(mask == 0, float("-inf"))

    # ── Attention sink ────────────────────────────────────────────────────
    if sink is not None:
        sink_score = sink.to(torch.float32).view(1, num_q_heads, 1, 1).expand(B, -1, S_tkg, -1)
        scores = torch.cat([scores, sink_score], dim=-1)  # [B, qh, S, S_ctx+1]

    attn_weights = torch.softmax(scores, dim=-1).to(q.dtype)
    if sink is not None:
        attn_weights = attn_weights[..., :-1]  # drop the sink column

    attn_out = torch.matmul(attn_weights, V_full)  # [B, q_heads, S_tkg, hd]

    # ── KV cache write (in place, via slot_mapping) ───────────────────────
    if update_cache:
        _write_kv_cache(K_cache, V_cache, k, v, slot_mapping, block_size, num_kv_heads)

    # ── Output projection ─────────────────────────────────────────────────
    attn_flat = attn_out.transpose(1, 2).reshape(B * S_tkg, num_q_heads * head_dim)
    output = attn_flat @ W_out  # [B*S_tkg, H]
    if bias_out is not None:
        output = output + bias_out
    return output


def _build_decode_mask(
    pos_ids: Tensor,
    B: int,
    S_tkg: int,
    S_ctx: int,
    sliding_window: Optional[int],
    device: torch.device,
) -> Tensor:
    """Build the decode attention mask in logical (un-shuffled) space.

    Returns [B, 1, S_tkg, S_ctx] in {0,1}. Mirrors
    ``_torch_gen_attention_decode_mask_impl`` after the Stage-7e un-shuffle:
      * prior region: causal ``s < min_pos`` (or SWA band ``start <= s < min_pos``),
        where ``min_pos`` is the min active position per batch (conservative for
        S_tkg > 1 spec decode).
      * the last S_tkg slots carry the active-token causal triangle.
    ``s`` is the gathered-buffer index == logical absolute position.
    """
    pos = pos_ids.view(B, S_tkg).to(torch.float32)  # [B, S_tkg]
    min_pos = pos.min(dim=1, keepdim=True).values  # [B, 1]

    s = torch.arange(S_ctx, device=device, dtype=torch.float32)  # [S_ctx]
    s_b11 = s.view(1, 1, 1, S_ctx)  # broadcast [., ., ., S_ctx]
    min_pos_b = min_pos.view(B, 1, 1, 1)

    if sliding_window is None:
        prior = (s_b11 < min_pos_b).float()  # [B,1,1,S_ctx]
        prior = prior.expand(B, 1, S_tkg, S_ctx).clone()
    else:
        # per-query window start (inclusive)
        start = torch.clamp(pos - sliding_window + 1, min=0)  # [B, S_tkg]
        per_start = start.view(B, 1, S_tkg, 1)  # [B,1,S_tkg,1]
        ge_start = s_b11 >= per_start  # [B,1,S_tkg,S_ctx]
        lt_end = s_b11 < min_pos_b  # [B,1,1,S_ctx]
        normal = ge_start & lt_end
        wrap = ge_start | lt_end
        is_wrap = per_start > min_pos_b  # [B,1,S_tkg,1]
        prior = torch.where(is_wrap, wrap, normal).float()

    # Overlay the active-token causal triangle onto the last S_tkg slots.
    # query q (0..S_tkg-1) attends to active key kk iff q >= kk.
    active = torch.zeros(B, 1, S_tkg, S_tkg, device=device, dtype=torch.float32)
    tri = torch.tril(torch.ones(S_tkg, S_tkg, device=device))  # [q, kk] q>=kk
    active[:, :, :, :] = tri.view(1, 1, S_tkg, S_tkg)
    prior[:, :, :, S_ctx - S_tkg :] = active
    return prior


def _write_kv_cache(
    K_cache: Tensor,
    V_cache: Tensor,
    k: Tensor,  # [B, kv_heads, S_tkg, hd]
    v: Tensor,
    slot_mapping: Tensor,  # [B*S_tkg]
    block_size: int,
    num_kv_heads: int,
) -> None:
    """Scatter the new K/V tokens into the paged cache at slot_mapping."""
    d_head = k.shape[-1]
    num_blocks_total = K_cache.shape[0]
    max_slot = num_blocks_total * block_size
    slot = slot_mapping.to(torch.int64).reshape(-1)  # [B*S_tkg]
    is_sentinel = (slot < 0) | (slot >= max_slot)
    slot = torch.where(is_sentinel, torch.full_like(slot, max_slot - 1), slot)

    block_indices = (slot // block_size).repeat(num_kv_heads)
    position_indices = (slot % block_size).repeat(num_kv_heads)
    head_indices = torch.arange(num_kv_heads, dtype=torch.long, device=k.device).repeat_interleave(slot.shape[0])

    # k,v: [B, kv_heads, S_tkg, hd] -> [kv_heads, B*S_tkg, hd] -> [kv_heads*B*S_tkg, hd]
    k_flat = k.transpose(0, 1).reshape(-1, d_head).to(K_cache.dtype)
    v_flat = v.transpose(0, 1).reshape(-1, d_head).to(V_cache.dtype)
    K_cache.index_put_((block_indices, head_indices, position_indices), k_flat)
    V_cache.index_put_((block_indices, head_indices, position_indices), v_flat)


# ===========================================================================
# MoE block decode oracle (dense dequantized weights)
# ===========================================================================


def moe_block_tkg(
    hidden_states: Tensor,  # [T, H]
    gamma: Tensor,  # [H] RMSNorm weight
    eps: float,
    router_weight: Tensor,  # [E, H]
    router_bias: Tensor,  # [E]
    gate_up_weight: Tensor,  # [E, H, 2, I]  (dense, dequantized)
    gate_up_bias: Tensor,  # [E, 2, I]     (up portion already +1)
    down_weight: Tensor,  # [E, I, H]     (dense, dequantized)
    down_bias: Tensor,  # [E, H]
    top_k: int,
    swiglu_limit: float,
    swiglu_alpha: float,
) -> Tensor:
    """MoE decode oracle: RMSNorm -> router top-k softmax -> SwiGLU experts.

    Transcribes ``_torch_moe_block_tkg_impl`` (SOFTMAX router, ``pre_norm=False``,
    POST_SCALE affinities, Swish activation) using an HF-style per-expert token
    gather. Operates on dense dequantized weights (the golden dequantizes MXFP4
    up front in loader.py).

    The ``gate_up_bias`` up portion already includes the +1 (``hidden_act_bias``);
    combined with ``up`` clamped to ``[-limit+1, limit+1]`` this equals HF's
    ``clamp(up, -limit, limit)`` then ``(up+1)*glu``.

    Returns [T, H].

    Usage:
        >>> out = moe_block_tkg(h, gamma, 1e-5, rw, rb, guw, gub, dw, db,
        ...     top_k=4, swiglu_limit=7.0, swiglu_alpha=1.702)
    """
    T, H = hidden_states.shape
    E = router_weight.shape[0]
    out_dtype = hidden_states.dtype

    # ── RMSNorm (fp32 accumulation) ───────────────────────────────────────
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = gamma.to(torch.float32) * normed  # [T, H] fp32

    # ── Router (top-k then softmax over the selected logits) ──────────────
    logits = torch.matmul(normed.to(torch.float32), router_weight.to(torch.float32).t()) + router_bias.to(
        torch.float32
    )  # [T, E]
    top_value, indices = torch.topk(logits, top_k, dim=-1)  # [T, top_k]
    scores = torch.softmax(top_value, dim=-1).to(torch.float32)  # [T, top_k]

    # ── Experts (per-expert token gather, HF-style) ───────────────────────
    next_states = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    guw = gate_up_weight.to(torch.float32)
    gub = gate_up_bias.to(torch.float32)
    dw = down_weight.to(torch.float32)
    db = down_bias.to(torch.float32)

    for e in range(E):
        # tokens (rows) and their top-k slot where expert e was selected
        tok, kpos = torch.where(indices == e)
        if tok.numel() == 0:
            continue
        cur = normed[tok]  # [n, H] fp32
        gate_up = torch.einsum("nh,hgi->ngi", cur, guw[e]) + gub[e]  # [n, 2, I]
        gate = gate_up[:, 0, :]
        up = gate_up[:, 1, :]
        gate = gate.clamp(max=swiglu_limit)  # lower = None
        up = up.clamp(min=-swiglu_limit + 1, max=swiglu_limit + 1)
        glu = gate * torch.sigmoid(gate * swiglu_alpha)
        intermediate = up * glu  # [n, I]
        out = torch.matmul(intermediate, dw[e]) + db[e]  # [n, H]
        weighted = out * scores[tok, kpos].unsqueeze(-1)
        next_states.index_add_(0, tok, weighted)

    return next_states.to(out_dtype)
