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

import ml_dtypes
import numpy as np
import torch

from .mla_common_cte import MlaPrecision


def _owned_global_pos(rank, l, interleave_size: int, cp_world_size: int):
    """Global sequence position of a rank's ``l``-th owned token under CP interleave.

    Tokens are round-robin assigned to CP ranks in whole blocks of ``interleave_size``:
    global position ``p`` is owned by ``(p // interleave_size) % cp_world_size``. Inverting
    that, rank ``r``'s ``l``-th owned token (0-indexed over that rank's own stream) sits at::

        owned(r, l) = (l // I) * (I * W)  +  r * I  +  (l % I)      # I=interleave, W=cp_world

    Works on ints or int tensors (``l``/``rank`` may be a torch tensor). ``W == 1`` collapses to
    the identity ``owned(0, l) == l`` (non-CP).
    """
    P = interleave_size * cp_world_size
    return (l // interleave_size) * P + rank * interleave_size + (l % interleave_size)


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
    prior_tokens_hbm: torch.Tensor = None,  # [1, 1] int32 local prior/cached count on this rank
    cp_rank_hbm: torch.Tensor = None,  # [1, 1] int32 this rank's CP index
    cp_world_size: int = 1,  # CP degree W
    cp_kv_cache_interleave_size: int = 1,  # interleave granularity I (== block_size)
    precision: MlaPrecision = MlaPrecision.MX,
    # Kernel-only flag: selects HOW the KV rows are gathered (SBUF residency + tensor indirection
    # vs DMA-indirect from HBM). Both gather the same rows, so the reference is identical.
    use_sbuf_indirect: bool = False,
    kv_cache_packed: bool = None,
    kv_cache_fp8_dtype=None,
) -> Dict[str, torch.Tensor]:
    """Reference for sparse MLA latent + RoPE attention (the un-projected value path).

    Under Context Parallelism the KV cache is stored INTERLEAVED (physical slot i is the
    ``i``-th rank-major owned token, whose global position is ``_owned_global_pos``), and the
    topk indices are PHYSICAL slots. So the causal mask converts both the query rows and the
    gathered key slots to GLOBAL positions before comparing. ``cp_world_size == 1`` collapses to
    the natural non-CP case (global position == physical index, ``q_global == prior + row``).

    Returns:
        Dict with key "out_attn": [B, S, H*L] bf16 latent attention output.
    """
    # cp_rank / prior_tokens arrive as TENSORS (one traced graph for all CP ranks); the ref only
    # needs their values. None = non-CP (rank 0, no prior).
    prior_tokens = 0 if prior_tokens_hbm is None else int(prior_tokens_hbm.reshape(-1)[0])
    cp_rank = 0 if cp_rank_hbm is None else int(cp_rank_hbm.reshape(-1)[0])
    I = cp_kv_cache_interleave_size
    W = cp_world_size
    q = q_lift_hbm.to(torch.float32)
    qpe = q_pe_hbm.to(torch.float32)
    B, S, H, L = q.shape
    if kv_cache_packed is None:
        # Same rank-based auto-detect the kernel does: the fused cache is 2-D [S_kv, kv_row].
        kv_cache_packed = len(c_kv_hbm.shape) == 2
    if kv_cache_packed:
        """
        PACKED cache: one fused bf16 row per token -- L fp8 latent bytes, a bf16 scale, then R bf16
        k_pe. Decode and dequantize here so the comparison isolates the kernel's arithmetic rather
        than charging fp8 rounding to the tolerance.

        The harness upcasts inputs to fp32 losslessly, so casting back to bf16 recovers the exact
        bytes the kernel reads.
        """
        rows = np.asarray(c_kv_hbm.to(torch.float32) if isinstance(c_kv_hbm, torch.Tensor) else c_kv_hbm)
        rows = rows.astype(ml_dtypes.bfloat16)
        n_rows, row = rows.shape[-2], rows.shape[-1]
        rb = rows.reshape(n_rows, row).view(np.uint8)
        lat = rb[:, 0:L].copy().view(ml_dtypes.float8_e4m3).astype(np.float32)
        scale = rb[:, L : L + 2].copy().view(ml_dtypes.bfloat16).astype(np.float32).reshape(n_rows, 1)
        kpe_np = rb[:, L + 2 : L + 2 + 2 * q_pe_hbm.shape[3]].copy().view(ml_dtypes.bfloat16).astype(np.float32)
        c = torch.from_numpy((lat * scale).reshape(B, n_rows // B, L))
        kpe = torch.from_numpy(kpe_np.reshape(B, n_rows // B, -1))
    else:
        c = c_kv_hbm.to(torch.float32)
        kpe = k_pe_hbm.to(torch.float32)
    S_kv = c.shape[1]
    S_kv_local = S_kv // W  # per-rank stride of the rank-major interleaved cache

    # GLOBAL position of each local query row (row r is this rank's (prior + r)-th owned token).
    q_global_all = _owned_global_pos(cp_rank, prior_tokens + torch.arange(S), I, W)  # [S]
    # GLOBAL position of each physical cache slot j (dense attends every slot).
    _slots = torch.arange(S_kv)
    key_global_all = _owned_global_pos(_slots // S_kv_local, _slots % S_kv_local, I, W)  # [S_kv]

    attn = torch.zeros((B, S, H, L), dtype=torch.float32)
    for b in range(B):
        if dense:
            # Dense: attend ALL S_kv keys (no gather), causal-mask by GLOBAL position.
            c_g = c[b]  # [S_kv, L]
            kpe_g = kpe[b]  # [S_kv, R]
            scores = torch.einsum("shd,jd->shj", q[b], c_g)  # [S, H, S_kv]
            scores = scores + torch.einsum("shr,jr->shj", qpe[b], kpe_g)
            scores = scores * softmax_scale
            causal = key_global_all[None, :] > q_global_all[:, None]  # [S, S_kv], True = future
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
                keys gets its topk padded by the indexer with FUTURE positions. topk indices are
                PHYSICAL cache slots, so convert each to its GLOBAL position before comparing;
                drop keys whose global position exceeds the query's. Under CP the conversion is
                the interleave map; with cp_world_size==1 it is the identity (slot == position).
                """
                q_global_c = q_global_all[qs:qe]
                key_global_c = _owned_global_pos(idx_c // S_kv_local, idx_c % S_kv_local, I, W)  # [chunk, K]
                future_c = key_global_c > q_global_c[:, None]  # [chunk, K], True = filler/future key
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
