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

"""PyTorch reference for mla_sparse_attention_cte_kernel (KERNEL A — attention only).

Sparse latent + RoPE attention -> latent value path out_attn[B, S, H*L] (row-major
h*L + l). This is stage 1 of the split sparse-MLA kernel; the o_proj is kernel B.
"""

from typing import Dict

import torch


def mla_sparse_attention_cte_torch_ref(
    q_lift_hbm: torch.Tensor,  # [B, S, H, L] bf16
    q_pe_hbm: torch.Tensor,  # [B, S, H, R] bf16 (pre-rotated RoPE queries)
    c_kv_hbm: torch.Tensor,  # [B, S_kv, L] bf16
    k_pe_hbm: torch.Tensor,  # [B, S_kv, R] bf16 (pre-rotated RoPE keys)
    softmax_scale: float,
    topk_indices_hbm: torch.Tensor = None,  # [B, S, K] int32; unused/None in dense mode
    # Kernel-only flag (partition-tiled topk layout). Declared to match the kernel
    # signature; the ref always consumes the flat [B, S, K] topk_indices.
    topk_tiled: bool = False,
    dense: bool = False,
    q_pos_offset: int = 0,
) -> Dict[str, torch.Tensor]:
    """Reference for sparse MLA latent + RoPE attention (the un-projected value path).

    Returns:
        Dict with key "out_attn": [B, S, H*L] bf16 latent attention output.
    """
    q = q_lift_hbm.to(torch.float32)
    qpe = q_pe_hbm.to(torch.float32)
    c = c_kv_hbm.to(torch.float32)
    kpe = k_pe_hbm.to(torch.float32)
    B, S, H, L = q.shape
    S_kv = c.shape[1]

    attn = torch.zeros((B, S, H, L), dtype=torch.float32)
    for b in range(B):
        if dense:
            # Dense: attend ALL S_kv keys (no gather), causal-mask key j > query pos. Queries
            # are seq-sharded across cores, so query s's global position is s (B==1, one shard
            # per core sees its own [s_start, s_start+s_per_core); the ref runs the full S here
            # and the kernel masks offset = q_pos_offset + s_start + s_local -> identical per-query
            # result).
            c_g = c[b]  # [S_kv, L]
            kpe_g = kpe[b]  # [S_kv, R]
            scores = torch.einsum("shd,jd->shj", q[b], c_g)  # [S, H, S_kv]
            scores = scores + torch.einsum("shr,jr->shj", qpe[b], kpe_g)
            scores = scores * softmax_scale
            q_global = q_pos_offset + torch.arange(S)
            causal = torch.arange(S_kv)[None, :] > q_global[:, None]  # [S, S_kv], True = future
            scores = scores.masked_fill(causal[:, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attn[b] = torch.einsum("shj,jd->shd", weights, c_g)  # [S, H, L]
        else:
            idx = topk_indices_hbm.to(torch.int64)
            c_g = c[b][idx[b]]  # [S, K, L]
            kpe_g = kpe[b][idx[b]]  # [S, K, R]
            scores = torch.einsum("shd,sjd->shj", q[b], c_g)
            scores = scores + torch.einsum("shr,sjr->shj", qpe[b], kpe_g)
            scores = scores * softmax_scale
            # Causal re-mask (matches the kernel and the reference indexer's index_mask += causal
            # mask): for a query whose GLOBAL position has fewer than K valid keys, the indexer
            # pads its topk with FUTURE positions (idx > q_global). Drop those from the softmax so
            # the query attends only its true causal prefix. No-op when every index is valid.
            q_global = q_pos_offset + torch.arange(S)
            future = idx[b] > q_global[:, None]  # [S, K], True = filler/future key
            scores = scores.masked_fill(future[:, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attn[b] = torch.einsum("shj,sjd->shd", weights, c_g)  # [S, H, L]

    """
    MX 4-pack column pre-permute (must match the kernel's MM2 write): within each
    head's L block, natural latent l = 4*group + sub is stored at physical column
    sub*(L//4) + group ("sub-major"). This lets the o_proj consumer transpose each
    sub's contiguous L//4 groups with a plain dma_transpose into the MX layout where
    partition p holds latents {4p, 4p+1, 4p+2, 4p+3}. See the IMPORTANT note at the
    MM2 write in sparse_mla_latent_attn_vupmx_oproj_cte.py.
    """
    H_PACK = 4
    attn = attn.reshape(B, S, H, L // H_PACK, H_PACK).permute(0, 1, 2, 4, 3).reshape(B, S, H * L)
    return {"out_attn": attn.to(torch.bfloat16)}
