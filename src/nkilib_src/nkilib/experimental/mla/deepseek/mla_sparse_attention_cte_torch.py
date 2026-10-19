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

from .mla_common_cte import MlaPrecision


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
    q_pos_offset_hbm: torch.Tensor = None,  # [1, 1] int32 global offset of this rank's q shard
    precision: MlaPrecision = MlaPrecision.MX,
    # Kernel-only flag: selects HOW the KV rows are gathered (SBUF residency + tensor indirection
    # vs DMA-indirect from HBM). Both gather the same rows, so the reference is identical.
    use_sbuf_indirect: bool = False,
) -> Dict[str, torch.Tensor]:
    """Reference for sparse MLA latent + RoPE attention (the un-projected value path).

    Returns:
        Dict with key "out_attn": [B, S, H*L] bf16 latent attention output.
    """
    # The kernel takes the offset as a tensor (one traced graph for all CP ranks); the ref only
    # needs its value. None = non-CP, offset 0.
    q_pos_offset = 0 if q_pos_offset_hbm is None else int(q_pos_offset_hbm.reshape(-1)[0])
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
            """
            Chunk over queries. The gathered c_g is [S, K, L], which at long context is far too
            large to materialize at once (S=2048, K=2048, L=512 fp32 is ~8.6 TB); a chunk of 64
            queries is ~270 MB. Chunking is arithmetically identical -- every query's attention is
            independent.
            """
            Q_CHUNK = 64
            for qs in range(0, S, Q_CHUNK):
                qe = min(qs + Q_CHUNK, S)
                idx_c = idx[b][qs:qe]
                c_g = c[b][idx_c]  # [chunk, K, L]
                kpe_g = kpe[b][idx_c]  # [chunk, K, R]
                sc = torch.einsum("shd,sjd->shj", q[b][qs:qe], c_g)
                sc = sc + torch.einsum("shr,sjr->shj", qpe[b][qs:qe], kpe_g)
                sc = sc * softmax_scale
                """
                Causal re-mask (matches the kernel and the reference indexer's
                index_mask += causal_mask): a query whose GLOBAL position has fewer than K valid
                keys gets its topk padded by the indexer with FUTURE positions (idx > q_global).
                Drop those from the softmax so the query attends only its true causal prefix;
                no-op when every index is valid.
                """
                q_global_c = q_pos_offset + torch.arange(qs, qe)
                future_c = idx_c > q_global_c[:, None]  # [chunk, K], True = filler/future key
                sc = sc.masked_fill(future_c[:, None, :], float("-inf"))
                w = torch.softmax(sc, dim=-1)
                attn[b, qs:qe] = torch.einsum("shj,sjd->shd", w, c_g)

    """
    Latent column layout, matching the kernel's MM2 write (see ``_mm2_out_aps``).

    MX 4-pack pre-permute: within each head's L block, natural latent l = 4*group + sub is
    stored at physical column sub*(L//4) + group ("sub-major"). This lets the MX o_proj consumer
    transpose each sub's contiguous L//4 groups with a plain dma_transpose into the MX layout
    where partition p holds latents {4p, 4p+1, 4p+2, 4p+3}.

    BF16: natural order — the bf16 o_proj transposes the latent on the Tensor Engine, so there
    is no MX layout to pre-feed.
    """
    if precision.is_bf16():
        attn = attn.reshape(B, S, H * L)
    else:
        H_PACK = 4
        attn = attn.reshape(B, S, H, L // H_PACK, H_PACK).permute(0, 1, 2, 4, 3).reshape(B, S, H * L)
    return {"out_attn": attn.to(torch.bfloat16)}
