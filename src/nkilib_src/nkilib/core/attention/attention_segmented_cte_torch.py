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

"""Torch reference implementation for segmented attention with block-based KV cache."""

from typing import Optional

import torch
import torch.nn.functional as F


def attention_segmented_cte_torch_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    prior_tokens: torch.Tensor,
    block_size: int,
    prior_seg_size: int,
    scale: float = 1.0,
    tp_q: bool = True,
    tp_out: bool = False,
    sliding_window: Optional[int] = None,
    sink: Optional[torch.Tensor] = None,
    num_q_heads: int = 1,
    k_pre_transposed: bool = False,
    fp8_packed: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    kvp_q_offset: Optional[torch.Tensor] = None,
    kvp_rank_id: Optional[torch.Tensor] = None,
    kvp_group_size: int = 0,
    kvp_cp_offset_int: int = 0,
    kvp_seg_block_offset_int: int = 0,
    kvp_prior_load_blocks: int = 0,
    kvp_prior_fully_visible: bool = False,
) -> torch.Tensor:
    """
    Torch reference for segmented attention with block-based KV cache.

    Args:
        q: Query tensor, shape (bs_q, seqlen_q, head_dim) if tp_q else (bs_q, head_dim, seqlen_q)
        k_cache: KV cache for keys, shape (num_blocks, num_kv_heads, block_size, head_dim)
        v_cache: KV cache for values, shape (num_blocks, num_kv_heads, block_size, head_dim)
        block_tables: Block indices per batch, shape (bs, max_blocks_per_seq), -1 for padding
        prior_tokens: Number of prior tokens, shape (1, 1)
        block_size: Size of each block in KV cache
        prior_seg_size: Size of each prior KV segment. seqlen_q can differ from this.
        scale: Attention scale factor
        tp_q: If True, Q is (bs_q, seqlen_q, head_dim), else (bs_q, head_dim, seqlen_q)
        tp_out: If True, output is (bs_q, head_dim, seqlen_q), else (bs_q, seqlen_q, head_dim)
        sliding_window: Optional sliding window size for attention
        sink: Optional sink tokens (not used in this implementation)
        num_q_heads: Number of query heads per batch item
        k_pre_transposed: If True, K cache is stored in transposed layout
            (num_blocks * num_kv_heads, head_dim, block_size). Reversed internally.
        k_scale: Optional per-head-dim dequantization scale for K cache, shape (128, 1).
            When provided, K cache values are multiplied by k_scale after loading.
        v_scale: Optional per-head-dim dequantization scale for V cache, shape (128, 1).
            When provided, V cache values are multiplied by v_scale after loading.
        kvp_q_offset: Optional causal mask offset for KV-parallel mode (unused in reference).
        kvp_rank_id: Optional rank index for interleaved KV distribution (unused in reference).
        kvp_group_size: Number of ranks for round-robin KV distribution (unused in reference).

    Returns:
        Attention output, shape (bs_q, seqlen_q, head_dim) or (bs_q, head_dim, seqlen_q) depending on tp_out
    """
    # Convert to float32 for numerical stability
    q = q.float()
    k_cache = k_cache.float()
    v_cache = v_cache.float()

    if fp8_packed:
        # Unpack head-major packed layout:
        #   (num_blocks, num_kv_heads, block_size//2, head_dim, 2) -> (num_blocks, num_kv_heads, block_size, head_dim)
        # Last axis indexes the 2 consecutive seq positions packed into a row. Move it next to
        # block_size//2 and fold the pair back into the sequence dimension.
        n_blocks = k_cache.shape[0]
        num_kv_heads = k_cache.shape[1]
        block_size_half = k_cache.shape[2]
        head_dim_val = k_cache.shape[3]
        k_cache = k_cache.permute(0, 1, 2, 4, 3)  # (N, H, bs//2, 2, D)
        k_cache = k_cache.reshape(n_blocks, num_kv_heads, block_size_half * 2, head_dim_val)  # (N, H, bs, D)
    elif k_pre_transposed:
        num_kv_heads = v_cache.shape[1]
        block_size_val = k_cache.shape[2]
        head_dim_val = k_cache.shape[1]
        num_blocks = k_cache.shape[0] // num_kv_heads
        # Reverse the transpose: (N*H, head_dim, block_size) -> (N, H, block_size, head_dim)
        k_cache = k_cache.reshape(num_blocks, num_kv_heads, head_dim_val, block_size_val)
        k_cache = k_cache.permute(0, 1, 3, 2)  # -> (N, H, block_size, head_dim)

    # Apply dequantization scales if provided (simulates fp8 dequant: bf16_val = fp8_val * scale)
    if k_scale is not None:
        k_cache = k_cache * k_scale.float().flatten()[0].item()
    if v_scale is not None:
        v_cache = v_cache * v_scale.float().flatten()[0].item()
    # Transpose Q if needed to (bs_q, seqlen_q, head_dim)
    if not tp_q:
        q = q.transpose(1, 2)

    bs_q = q.shape[0]
    seqlen_q = q.shape[1]
    head_dim = q.shape[2]
    bs = block_tables.shape[0]
    num_kv_heads = k_cache.shape[1]
    prior_tokens_val = prior_tokens.item()
    active_tokens = seqlen_q
    total_tokens = prior_tokens_val + active_tokens

    # Compute attention per Q head with GQA mapping
    ref_list = []
    for b_idx in range(bs_q):
        # Map Q batch to block_tables batch and KV head
        batch_item_idx = b_idx // num_q_heads
        kv_head_idx = (b_idx % num_q_heads) * num_kv_heads // num_q_heads

        # Number of real (non-padding) blocks for this batch row. The test
        # pads block_tables to max_blocks_per_seq with either -1 (sentinel) or
        # a valid block index used as padding. Use the logical token counts to
        # derive the real block count so the torch ref works under either
        # padding strategy.
        num_real_blocks = (prior_tokens_val + active_tokens) // block_size
        valid_indices = block_tables[batch_item_idx, :num_real_blocks]

        # KV cache layout: (num_blocks, num_kv_head, block_size, head_dim)
        k_blocks = k_cache[valid_indices, kv_head_idx, :, :]
        v_blocks = v_cache[valid_indices, kv_head_idx, :, :]

        # Reshape to get full KV sequence
        k_seq = k_blocks.reshape(-1, head_dim)
        v_seq = v_blocks.reshape(-1, head_dim)

        # Compute attention scores for this Q head
        q_single = q[b_idx : b_idx + 1]  # (1, seqlen_q, head_dim)
        k_full = k_seq.permute(1, 0)  # (head_dim, total_tokens)
        scores = torch.matmul(q_single, k_full) * scale  # (1, seqlen_q, total_tokens)

        # Apply causal mask to active segment only (last seqlen_q tokens)
        causal_mask = torch.triu(torch.ones((seqlen_q, seqlen_q), dtype=torch.bool, device=scores.device), diagonal=1)
        scores[:, :, prior_tokens_val:].masked_fill_(causal_mask, float('-inf'))

        # Apply sliding window mask if provided
        if sliding_window is not None and sliding_window > 0:
            q_positions = torch.arange(seqlen_q, device=scores.device).unsqueeze(1) + prior_tokens_val
            k_positions = torch.arange(total_tokens, device=scores.device).unsqueeze(0)
            sw_mask = k_positions < q_positions - (sliding_window - 1)
            scores.masked_fill_(sw_mask.unsqueeze(0), float('-inf'))

        # Append sink to scores if provided
        if sink is not None:
            sink_val = sink[b_idx : b_idx + 1].reshape(1, 1, 1).expand(1, seqlen_q, 1)
            scores = torch.cat([scores, sink_val], dim=-1)

        # Compute attention output
        attn_weights = F.softmax(scores, dim=-1)
        if sink is not None:
            attn_weights = attn_weights[:, :, :-1]  # Drop sink weight before PV matmul
        result_single = torch.matmul(attn_weights, v_seq.unsqueeze(0))
        ref_list.append(result_single)

    result = torch.cat(ref_list, dim=0)  # (bs_q, seqlen_q, head_dim)

    # Transpose output if needed
    if tp_out:
        result = result.transpose(1, 2)

    return result
