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

"""PyTorch reference for ring_attention_fwd kernel."""

import math

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives.distributed_adapter import get_pg, get_rank


def ref_attention_fwd(q, k, v, scale, causal=False, extra_mask=None):
    """
    Reference attention forward pass (numpy), chunked along Q to avoid O(S^2) memory.

    Args:
        q: (bs, seqlen_q, d)
        k: (bs, seqlen_k, d)
        v: (bs, seqlen_k, d)
        scale: float
        causal: apply causal mask
        extra_mask: optional (seqlen_q, seqlen_k) bool mask, True = attend

    Returns:
        o: (bs, seqlen_q, d), lse: (bs, seqlen_q)
    """
    bs, seqlen_q, d = q.shape
    seqlen_k = k.shape[1]
    chunk_size = min(2048, seqlen_q)

    # Extract per-row bounds from extra_mask to avoid materializing (S, S) arrays
    bound_min = None
    bound_max = None
    if extra_mask is not None:
        bound_min = np.empty(seqlen_q, dtype=np.int64)
        bound_max = np.empty(seqlen_q, dtype=np.int64)
        for i in range(seqlen_q):
            nonzero = np.flatnonzero(extra_mask[i])
            if len(nonzero) > 0:
                bound_min[i] = nonzero[0]
                bound_max[i] = nonzero[-1] + 1
            else:
                bound_min[i] = 0
                bound_max[i] = 0

    k_t = k.transpose(0, 2, 1)  # (bs, d, seqlen_k)
    o = np.empty_like(q)
    lse = np.empty((bs, seqlen_q), dtype=q.dtype)

    for q_start in range(0, seqlen_q, chunk_size):
        q_end = min(q_start + chunk_size, seqlen_q)
        q_chunk = q[:, q_start:q_end, :]

        # scores: (bs, clen, seqlen_k)
        scores = scale * (q_chunk @ k_t)

        # Causal mask
        if causal:
            q_pos = np.arange(q_start, q_end)[:, None]
            k_pos = np.arange(seqlen_k)[None, :]
            causal_mask = q_pos < k_pos  # (clen, seqlen_k)
            scores[:, causal_mask] = -float("inf")

        # Sequence packing mask
        if bound_min is not None:
            k_idx = np.arange(seqlen_k)
            bmin_chunk = bound_min[q_start:q_end, None]  # (clen, 1)
            bmax_chunk = bound_max[q_start:q_end, None]  # (clen, 1)
            pack_mask = (k_idx < bmin_chunk) | (k_idx >= bmax_chunk)
            scores[:, pack_mask] = -float("inf")

        row_max = scores.max(axis=-1, keepdims=True)
        # Fully-masked rows (all scores -inf) have row_max = -inf, which would make
        # exp/divide produce nan. The kernel writes o=0 and lse=0 for these rows, so
        # mirror that contract. For any row that attends to >=1 key this is a no-op.
        fully_masked = np.isneginf(row_max)
        row_max_safe = np.where(fully_masked, 0.0, row_max)
        exp_scores = np.exp(scores - row_max_safe)
        row_sum = exp_scores.sum(axis=-1, keepdims=True)
        row_sum_safe = np.where(fully_masked, 1.0, row_sum)
        softmax_weights = exp_scores / row_sum_safe

        o_chunk = softmax_weights @ v
        o[:, q_start:q_end, :] = np.where(fully_masked, 0.0, o_chunk)
        lse[:, q_start:q_end] = np.where(
            fully_masked.squeeze(-1),
            0.0,
            (row_max_safe + np.log(row_sum_safe)).squeeze(-1),
        )

    return o, lse


def ring_attention_spmd_fwd_torch_ref(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    replica_groups: tuple = None,
    num_workers: int = 1,
    softmax_scale: float = None,
    use_causal_mask: bool = False,
    striped_input: bool = False,
    training: bool = False,
    lse_dtype=None,
    tp_q: bool = False,
    tp_k: bool = False,
    bound_min: np.ndarray = None,
    bound_max: np.ndarray = None,
) -> dict:
    """PyTorch reference for ring_attention_spmd_fwd. Same signature as the kernel.

    Uses get_rank() and get_pg() to gather all ranks' data via all_gather,
    then computes full-context attention and returns this rank's output slice.
    """
    rank_id = get_rank()

    # Find this rank's sub-group and position within it
    if isinstance(replica_groups, ReplicaGroup):
        groups = [list(g) for g in replica_groups._value]
    else:
        groups = [list(g) for g in replica_groups]

    my_group = next(g for g in groups if rank_id in g)
    my_worker_idx = my_group.index(rank_id)
    nw = len(my_group)

    # Get the process group — try sub-group PG first, fall back to default all-ranks PG
    rg = replica_groups if isinstance(replica_groups, ReplicaGroup) else ReplicaGroup(groups)
    try:
        pg = get_pg(rg)
    except KeyError:
        # Framework didn't create a sub-group PG (replica_groups was a plain tuple).
        # Use default all-ranks PG and filter results to my sub-group.
        total_ranks = sum(len(g) for g in groups)
        default_rg = ReplicaGroup([list(range(total_ranks))])
        pg = get_pg(default_rg)

    # If PG covers more ranks than my sub-group, we need to gather from all
    # ranks and then filter to my sub-group members.
    use_subgroup_filter = pg.size() > nw

    def _to_ref_3d(arr, transposed):
        """Kernel 4-D layout (bs, h, spr, d) or (bs, h, d, spr) -> (bs*h, spr, d)."""
        if transposed:
            arr = arr.transpose(0, 1, 3, 2)
        bs, h, spr, d = arr.shape
        return arr.reshape(bs * h, spr, d).astype(np.float32)

    # Gather all workers' Q/K/V via all_gather
    def _gather_all(local_arr):
        t = torch.from_numpy(local_arr.astype(np.float32))
        gathered = [torch.zeros_like(t) for _ in range(pg.size())]
        dist.all_gather(gathered, t, group=pg)
        if use_subgroup_filter:
            # Filter to only my sub-group members (ordered by position in group)
            return [gathered[r].numpy() for r in my_group]
        return [g.numpy() for g in gathered]

    q_3d = _to_ref_3d(q, transposed=not tp_q)
    k_3d = _to_ref_3d(k, transposed=not tp_k)
    v_3d = _to_ref_3d(v, transposed=False)

    q_workers = _gather_all(q_3d)
    k_workers = _gather_all(k_3d)
    v_workers = _gather_all(v_3d)

    # Undo pre-scaling on Q when causal (test pre-scales Q and passes scale=1.0)
    actual_scale = softmax_scale
    if use_causal_mask and softmax_scale == 1.0:
        d = q_workers[0].shape[-1]
        actual_scale = 1.0 / math.sqrt(d)
        q_workers = [qw / actual_scale for qw in q_workers]

    # Compute reference attention
    if striped_input:
        bs_h, spr, d = q_workers[0].shape
        seqlen = spr * nw
        q_global = np.empty((bs_h, seqlen, d), dtype=np.float32)
        k_global = np.empty_like(q_global)
        v_global = np.empty_like(q_global)
        for w in range(nw):
            q_global[:, w::nw, :] = q_workers[w]
            k_global[:, w::nw, :] = k_workers[w]
            v_global[:, w::nw, :] = v_workers[w]
        # Sequence packing: build same-document mask from bounds
        extra_mask = None
        if bound_min is not None:
            # In striped packed mode, all ranks have identical bounds.
            # Reconstruct global bounds by interleaving local bounds in striped order.
            bmin_local = bound_min.reshape(bs_h, spr).astype(np.float32)
            bmin_global = np.empty((bs_h, seqlen), dtype=np.float32)
            for w in range(nw):
                bmin_global[:, w::nw] = bmin_local
            # Same-doc mask: tokens attend iff they have the same bound_min (vectorized)
            bmin_row = bmin_global[0]
            extra_mask = bmin_row[:, None] == bmin_row[None, :]
        o_global, lse_global = ref_attention_fwd(
            q_global, k_global, v_global, actual_scale, causal=True, extra_mask=extra_mask
        )
        o_ref = o_global[:, my_worker_idx::nw, :]
        lse_ref = lse_global[:, my_worker_idx::nw]
    else:
        # Concatenate all K/V, compute full attention for each rank's Q
        k_full = np.concatenate(k_workers, axis=1)
        v_full = np.concatenate(v_workers, axis=1)
        q_full = np.concatenate(q_workers, axis=1)
        o_full, lse_full = ref_attention_fwd(q_full, k_full, v_full, actual_scale, causal=use_causal_mask)
        spr = q_workers[0].shape[1]
        o_ref = o_full[:, my_worker_idx * spr : (my_worker_idx + 1) * spr, :]
        lse_ref = lse_full[:, my_worker_idx * spr : (my_worker_idx + 1) * spr]

    # Convert to kernel output layout
    bs, h = q.shape[0], q.shape[1]
    spr = o_ref.shape[1]
    d = o_ref.shape[2]
    out_o = o_ref.reshape(bs, h, spr, d).astype(q.dtype)
    result = {"out_o": out_o}

    # LSE is only emitted by the kernel when training=True, so only include it then (the
    # golden's keys must match the kernel's output count). Layout is (bs, h, 128, num_grps)
    # where query position p = g*128 + r maps to [b, h, r, g]. When seqlen is not a multiple
    # of 128 the final group is partial; the kernel zeros that group's padding rows, so we
    # pad the reference LSE with zeros to match before reshaping (no-op when divisible).
    if training:
        num_grps = math.ceil(spr / 128)
        padded = num_grps * 128
        lse_2d = lse_ref.reshape(bs, h, spr).astype(np.float32)
        if padded != spr:
            lse_padded = np.zeros((bs, h, padded), dtype=np.float32)
            lse_padded[..., :spr] = lse_2d
            lse_2d = lse_padded
        result["out_lse"] = lse_2d.reshape(bs, h, num_grps, 128).transpose(0, 1, 3, 2)

    return result
