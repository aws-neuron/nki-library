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
"""PyTorch reference implementation for ring attention backward pass."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ...core.attention.attention_bwd_torch import compute_o_lse
from ..collectives.distributed_adapter import get_pg, get_rank


def _full_attention_bwd(q, k, v, dy, scale, causal=False, extra_mask=None):
    """
    Run attention backward on full (concatenated) tensors using chunked computation.

    Processes Q in chunks along the sequence dimension to avoid materializing
    the full (B, Hq, S, S) score matrix. Peak memory is O(S * chunk_size)
    instead of O(S^2).

    Args:
        q (torch.Tensor): (bs, seqlen_q, d)
        k (torch.Tensor): (bs, seqlen_k, d)
        v (torch.Tensor): (bs, seqlen_k, d)
        dy (torch.Tensor): (bs, seqlen_q, d)
        scale (float): Softmax scale factor.
        causal (bool): Whether to apply causal masking.
        extra_mask (np.ndarray, optional): (seqlen_q, seqlen_k) bool, True = attend.
    """
    bs, seqlen_q, d = q.shape
    seqlen_k = k.shape[1]

    # Choose chunk size: 2048 keeps peak at ~2GB instead of ~30GB for S=32K
    chunk_size = min(2048, seqlen_q)

    # Build bound_min/bound_max from extra_mask without materializing (S, S) int64
    bound_min = None
    bound_max = None
    if extra_mask is not None:
        bmin = np.empty(seqlen_q, dtype=np.int64)
        bmax = np.empty(seqlen_q, dtype=np.int64)
        for i in range(seqlen_q):
            row = extra_mask[i]
            nonzero = np.flatnonzero(row)
            if len(nonzero) > 0:
                bmin[i] = nonzero[0]
                bmax[i] = nonzero[-1] + 1
            else:
                bmin[i] = 0
                bmax[i] = 0
        bound_min = torch.from_numpy(bmin)
        bound_max = torch.from_numpy(bmax)

    # Precompute: k_t and v_t in (bs, d, seqlen_k) for matmul
    k_t = k.transpose(1, 2).float()  # (bs, d, S_k)
    v_f = v.float()  # (bs, S_k, d)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros(bs, seqlen_k, d, dtype=torch.float32)
    dv = torch.zeros(bs, seqlen_k, d, dtype=torch.float32)

    for q_start in range(0, seqlen_q, chunk_size):
        q_end = min(q_start + chunk_size, seqlen_q)
        clen = q_end - q_start

        q_chunk = q[:, q_start:q_end, :]  # (bs, clen, d)
        dy_chunk = dy[:, q_start:q_end, :]  # (bs, clen, d)

        # scores: (bs, clen, S_k)
        scores = torch.bmm(q_chunk.float(), k_t) * scale

        # Build mask for this chunk
        if causal or bound_min is not None:
            # Start with causal mask
            if causal:
                q_pos = torch.arange(q_start, q_end).unsqueeze(1)  # (clen, 1)
                k_pos = torch.arange(seqlen_k).unsqueeze(0)  # (1, S_k)
                chunk_mask = q_pos < k_pos  # (clen, S_k)
            else:
                chunk_mask = torch.zeros(clen, seqlen_k, dtype=torch.bool)

            # Sequence packing mask
            if bound_min is not None:
                k_idx = torch.arange(seqlen_k)
                bmin_chunk = bound_min[q_start:q_end].unsqueeze(1)  # (clen, 1)
                bmax_chunk = bound_max[q_start:q_end].unsqueeze(1)  # (clen, 1)
                pack_mask = (k_idx.unsqueeze(0) < bmin_chunk) | (k_idx.unsqueeze(0) >= bmax_chunk)
                chunk_mask = chunk_mask | pack_mask

            # Apply mask: broadcast (clen, S_k) -> (bs, clen, S_k)
            scores = scores.masked_fill(chunk_mask.unsqueeze(0), -float("inf"))

        # Softmax: (bs, clen, S_k)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        # dV += probs^T @ dy_chunk: (bs, S_k, clen) @ (bs, clen, d) -> (bs, S_k, d)
        dv += torch.bmm(probs.transpose(1, 2), dy_chunk.float())

        # dp = dy_chunk @ v^T: (bs, clen, d) @ (bs, d, S_k) -> (bs, clen, S_k)
        dp = torch.bmm(dy_chunk.float(), v_f.transpose(1, 2))

        # ds = probs * (dp - (dp * probs).sum(-1, keepdim=True))
        ds = probs * (dp - (dp * probs).sum(dim=-1, keepdim=True))

        # dQ chunk: ds @ k -> (bs, clen, d)
        dq[:, q_start:q_end, :] = torch.bmm(ds, k.float()) * scale

        # dK += ds^T @ q_chunk: (bs, S_k, clen) @ (bs, clen, d) -> (bs, S_k, d)
        dk += torch.bmm(ds.transpose(1, 2), q_chunk.float()) * scale

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _ring_attention_spmd_bwd_full(
    q_shards,
    k_shards,
    v_shards,
    dy_shards,
    scale,
    tp_degree,
    causal=False,
    striped=False,
    extra_mask=None,
):
    """
    PyTorch reference for ring attention backward.

    Computes golden dQ, dK, dV by concatenating all shards and running
    full attention backward using the existing attention_bwd_torch_ref.

    Args:
        q_shards (list[torch.Tensor]): Per-rank Q, each (bs, seqlen_per_rank, d).
        k_shards (list[torch.Tensor]): Per-rank K.
        v_shards (list[torch.Tensor]): Per-rank V.
        dy_shards (list[torch.Tensor]): Per-rank dY.
        scale (float): Softmax scale factor.
        tp_degree (int): Number of ranks.
        causal (bool): Whether to apply causal masking.
        striped (bool): Whether striped attention layout is used.

    Returns:
        tuple: (dq_shards, dk_shards, dv_shards) lists of torch.Tensor per rank.
    """
    seqlen_per_rank = q_shards[0].shape[1]

    if striped:
        full_seqlen = seqlen_per_rank * tp_degree
        bs, _, d = q_shards[0].shape

        q_full = torch.zeros(bs, full_seqlen, d)
        k_full = torch.zeros_like(q_full)
        v_full = torch.zeros_like(q_full)
        dy_full = torch.zeros_like(q_full)
        for rank in range(tp_degree):
            q_full[:, rank::tp_degree, :] = q_shards[rank]
            k_full[:, rank::tp_degree, :] = k_shards[rank]
            v_full[:, rank::tp_degree, :] = v_shards[rank]
            dy_full[:, rank::tp_degree, :] = dy_shards[rank]

        dq_full, dk_full, dv_full = _full_attention_bwd(
            q_full, k_full, v_full, dy_full, scale, causal=True, extra_mask=extra_mask
        )

        dq_shards = [dq_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
        dk_shards = [dk_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
        dv_shards = [dv_full[:, rank::tp_degree, :] for rank in range(tp_degree)]
    else:
        q_full = torch.cat(q_shards, dim=1)
        k_full = torch.cat(k_shards, dim=1)
        v_full = torch.cat(v_shards, dim=1)
        dy_full = torch.cat(dy_shards, dim=1)

        dq_full, dk_full, dv_full = _full_attention_bwd(q_full, k_full, v_full, dy_full, scale, causal)

        dq_shards = [dq_full[:, rank * seqlen_per_rank : (rank + 1) * seqlen_per_rank, :] for rank in range(tp_degree)]
        dk_shards = [dk_full[:, rank * seqlen_per_rank : (rank + 1) * seqlen_per_rank, :] for rank in range(tp_degree)]
        dv_shards = [dv_full[:, rank * seqlen_per_rank : (rank + 1) * seqlen_per_rank, :] for rank in range(tp_degree)]

    return dq_shards, dk_shards, dv_shards


def compute_per_rank_o_lse(q_shards, k_shards, v_shards, scale, tp_degree, causal=False, striped=False):
    """
    Compute per-rank O and LSE using full K/V (needed as kernel inputs).

    Args:
        q_shards (list[torch.Tensor]): Per-rank Q, each (bs_flat, seqlen_per_rank, d).
        k_shards (list[torch.Tensor]): Per-rank K.
        v_shards (list[torch.Tensor]): Per-rank V.
        scale (float): Softmax scale factor.
        tp_degree (int): Number of ranks.
        causal (bool): Whether to apply causal masking.
        striped (bool): Whether striped attention layout is used.

    Returns:
        tuple: (o_per_rank, lse_per_rank) lists of numpy arrays in kernel layout.
    """
    seqlen_per_rank = q_shards[0].shape[1]
    full_seqlen = seqlen_per_rank * tp_degree

    if striped:
        k_full = torch.zeros(k_shards[0].shape[0], full_seqlen, k_shards[0].shape[2])
        v_full = torch.zeros_like(k_full)
        for rank in range(tp_degree):
            k_full[:, rank::tp_degree, :] = k_shards[rank]
            v_full[:, rank::tp_degree, :] = v_shards[rank]
    else:
        k_full = torch.cat(k_shards, dim=1)
        v_full = torch.cat(v_shards, dim=1)

    o_per_rank = []
    lse_per_rank = []
    for rank in range(tp_degree):
        q_t = q_shards[rank].permute(0, 2, 1).unsqueeze(1).float()
        k_t = k_full.permute(0, 2, 1).unsqueeze(1).float()
        v_t = v_full.permute(0, 2, 1).unsqueeze(1).float()

        if causal:
            if striped:
                q_pos = torch.arange(seqlen_per_rank).unsqueeze(1) * tp_degree + rank
            else:
                q_pos = torch.arange(rank * seqlen_per_rank, (rank + 1) * seqlen_per_rank).unsqueeze(1)
            k_pos = torch.arange(full_seqlen).unsqueeze(0)
            causal_bias = torch.where(q_pos >= k_pos, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
            o_proj, lse, _ = compute_o_lse(q_t, k_t, v_t, False, True, softmax_scale=scale, logit_bias=causal_bias)
        else:
            o_proj, lse, _ = compute_o_lse(q_t, k_t, v_t, False, True, softmax_scale=scale)

        o_per_rank.append(o_proj.numpy())
        lse_per_rank.append(lse.numpy())

    return o_per_rank, lse_per_rank


def ring_attention_spmd_bwd_torch_ref(
    q_ref: np.ndarray,
    k_ref: np.ndarray,
    v_ref: np.ndarray,
    o_ref: np.ndarray,
    dy_ref: np.ndarray,
    lse_ref: np.ndarray,
    use_causal_mask: bool = False,
    mixed_precision: bool = True,
    softmax_scale: float = None,
    num_workers: int = 1,
    lnc_size: int = 1,
    replica_groups: tuple = None,
    striped_attention: bool = False,
    bound_min: np.ndarray = None,
    bound_max: np.ndarray = None,
) -> dict:
    """Per-rank PyTorch reference for ring_attention_spmd_bwd. Same signature as the kernel.

    Uses get_rank() and get_pg() to gather all ranks' data via all_gather,
    then computes full-context attention backward and returns this rank's gradient slice.
    """
    rank_id = get_rank()

    if isinstance(replica_groups, ReplicaGroup):
        groups = [list(g) for g in replica_groups._value]
    else:
        groups = [list(g) for g in replica_groups]

    my_group = next(g for g in groups if rank_id in g)
    my_worker_idx = my_group.index(rank_id)
    nw = len(my_group)

    rg = replica_groups if isinstance(replica_groups, ReplicaGroup) else ReplicaGroup(groups)
    pg = get_pg(rg)

    # Kernel layout: (bs, nheads, d, seqlen_per_rank) -> ref layout: (bs*nheads, seqlen_per_rank, d)
    def _to_ref_3d(arr):
        bs, nheads, d, spr = arr.shape
        return arr.transpose(0, 1, 3, 2).reshape(bs * nheads, spr, d).astype(np.float32)

    def _gather_all(local_3d):
        t = torch.from_numpy(local_3d)
        gathered = [torch.zeros_like(t) for _ in range(pg.size())]
        dist.all_gather(gathered, t, group=pg)
        return [g.numpy() for g in gathered]

    q_3d = _to_ref_3d(q_ref)
    k_3d = _to_ref_3d(k_ref)
    v_3d = _to_ref_3d(v_ref)
    dy_3d = _to_ref_3d(dy_ref)

    # Gather all ranks' data
    q_all = _gather_all(q_3d)
    k_all = _gather_all(k_3d)
    v_all = _gather_all(v_3d)
    dy_all = _gather_all(dy_3d)

    # Convert to torch tensors
    q_shards = [torch.from_numpy(s) for s in q_all]
    k_shards = [torch.from_numpy(s) for s in k_all]
    v_shards = [torch.from_numpy(s) for s in v_all]
    dy_shards = [torch.from_numpy(s) for s in dy_all]

    # Compute full backward using existing reference
    # Build same-document mask for packed attention
    extra_mask = None
    if bound_min is not None and striped_attention:
        # In striped packed mode, all ranks have identical bounds (broadcast from test).
        # Reconstruct global bounds by interleaving local bounds in striped order.
        spr = q_3d.shape[1]
        seqlen = spr * nw
        bmin_local = bound_min.reshape(q_3d.shape[0], spr).astype(np.float32)
        bmin_global = np.empty((q_3d.shape[0], seqlen), dtype=np.float32)
        for w in range(nw):
            bmin_global[:, w::nw] = bmin_local
        # Same-doc mask: tokens attend iff they have the same bound_min (vectorized)
        bmin_row = bmin_global[0]
        extra_mask = bmin_row[:, None] == bmin_row[None, :]

    dq_shards, dk_shards, dv_shards = _ring_attention_spmd_bwd_full(
        q_shards,
        k_shards,
        v_shards,
        dy_shards,
        softmax_scale,
        nw,
        causal=use_causal_mask,
        striped=striped_attention,
        extra_mask=extra_mask,
    )

    # Return this rank's slice in kernel layout: (bs, nheads, d, seqlen_per_rank)
    bs, nheads, d, spr = q_ref.shape
    dq_rank = dq_shards[my_worker_idx].numpy().reshape(bs, nheads, spr, d).transpose(0, 1, 3, 2)
    dk_rank = dk_shards[my_worker_idx].numpy().reshape(bs, nheads, spr, d).transpose(0, 1, 3, 2)
    dv_rank = dv_shards[my_worker_idx].numpy().reshape(bs, nheads, spr, d).transpose(0, 1, 3, 2)

    return {
        "out_dq_ref": dq_rank.astype(np.float32),
        "out_dk_ref": dk_rank.astype(np.float32),
        "out_dv_ref": dv_rank.astype(np.float32),
    }
