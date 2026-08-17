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

"""PyTorch reference for attention_kv_parallel_segmented_cte kernel."""

from typing import Optional

import nki.language as nl
import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ...experimental.collectives.distributed_adapter import get_pg, get_rank


def attention_kv_parallel_segmented_cte_torch_ref(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_tables: np.ndarray,
    kvp_q_offset: np.ndarray,
    replica_groups: ReplicaGroup,
    group_size: int,
    block_size: int,
    seg_size: int,
    scale: float = 1.0,
    global_q_offset: int = 0,
    tp_out: bool = False,
    sliding_window: int = 0,
    kvp_rank_id: Optional[np.ndarray] = None,
    kvp_group_size: int = 0,
    apc_mode: bool = False,
    valid_num_prior_tokens: np.ndarray = None,
    fp8_packed: bool = False,
    k_scale: Optional[np.ndarray] = None,
    v_scale: Optional[np.ndarray] = None,
) -> dict:
    """PyTorch reference for attention_kv_parallel_segmented_cte. Same signature as the kernel.

    Uses get_rank() and get_pg(replica_groups) to gather all ranks' KV shards,
    then computes full-context causal attention and returns this rank's output.

    Args:
        q: [q_heads_per_rank, seq_len, head_dim] - this rank's Q heads
        k_cache: This rank's K shard. Standard shape is
            [num_blocks, num_kv_heads, block_size, head_dim]; with fp8_packed,
            [num_blocks, num_kv_heads, block_size // 2, head_dim, 2].
        v_cache: [num_blocks, num_kv_heads, block_size, head_dim] - this rank's KV shard
        block_tables: [1, num_blocks] - block indices (sequential)
        kvp_offset: [1, 1] - causal offset for this rank
        replica_groups: ReplicaGroup defining the collective topology
        group_size: number of logical ranks in the replica group
        block_size: KV cache block size
        seg_size: segment size for attention iteration
        scale: softmax scale factor
        global_q_offset: prior tokens offset
        tp_out: if True, transpose output to [q_heads_per_rank, head_dim, seq_len]
        fp8_packed: whether K uses the packed FP8 layout
        k_scale: optional K-cache dequantization scale, shape [128, 1]
        v_scale: optional V-cache dequantization scale, shape [128, 1]. All
            entries must repeat one scalar because the kernel applies it along
            the query partition, not the head dimension.
    """
    rank_id = get_rank()
    pg = get_pg(replica_groups)
    num_physical_ranks = pg.size()
    my_worker_idx = pg.rank()

    q_heads_per_rank = q.shape[0]
    seq_len = q.shape[1]
    head_dim = q.shape[2]

    if fp8_packed:
        num_blocks, num_kv_heads, block_size_half, cache_head_dim, pair = k_cache.shape
        if pair != 2:
            raise ValueError(f"packed K trailing dimension must be 2, got {pair}")
        k_cache = k_cache.transpose(0, 1, 2, 4, 3).reshape(
            num_blocks, num_kv_heads, block_size_half * 2, cache_head_dim
        )

    # Gather all ranks' K/V via all_gather
    # Flatten k_cache to [local_kv_len, head_dim] for gathering
    k_local = k_cache[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)
    v_local = v_cache[:, 0, :, :].reshape(-1, head_dim).astype(np.float32)

    def _expand_scale(
        scale: np.ndarray,
        name: str,
        *,
        require_replicated_scalar: bool = False,
    ) -> np.ndarray:
        values = np.asarray(scale, dtype=np.float32).reshape(-1)
        if values.size == 1:
            return values
        if values.size != nl.tile_size.pmax:
            raise ValueError(f"{name} must be scalar or have {nl.tile_size.pmax} values, got {values.size}")
        if require_replicated_scalar and not np.all(values == values[0]):
            raise ValueError(f"{name} must contain one scalar replicated {nl.tile_size.pmax} times")
        return np.resize(values, head_dim)

    if k_scale is not None:
        k_local *= _expand_scale(k_scale, "k_scale")
    if v_scale is not None:
        v_local *= _expand_scale(
            v_scale,
            "v_scale",
            require_replicated_scalar=True,
        )

    k_t = torch.from_numpy(k_local)
    v_t = torch.from_numpy(v_local)
    k_gathered = [torch.zeros_like(k_t) for _ in range(num_physical_ranks)]
    v_gathered = [torch.zeros_like(v_t) for _ in range(num_physical_ranks)]
    dist.all_gather(k_gathered, k_t, group=pg)
    dist.all_gather(v_gathered, v_t, group=pg)

    k_full = torch.cat(k_gathered, dim=0).numpy()
    v_full = torch.cat(v_gathered, dim=0).numpy()
    total_kv_len = k_full.shape[0]

    # Compute attention for each of this rank's Q heads
    outputs = []
    for head_idx in range(q_heads_per_rank):
        q_vec = q[head_idx].astype(np.float32)  # [seq_len, head_dim]

        # Compute attention scores
        scores = np.matmul(q_vec, k_full.T)  # [seq_len, total_kv_len]

        # Apply causal mask
        q_pos = np.arange(global_q_offset, global_q_offset + seq_len).reshape(-1, 1)
        k_pos = np.arange(total_kv_len).reshape(1, -1)
        causal_mask = q_pos < k_pos
        scores = np.where(causal_mask, -np.inf, scores)

        # Softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        max_scores = np.where(np.isinf(max_scores), 0, max_scores)
        exp_scores = np.exp(scores - max_scores)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        sum_exp = np.where(sum_exp == 0, 1, sum_exp)
        attn_weights = exp_scores / sum_exp

        out = np.matmul(attn_weights, v_full)  # [seq_len, head_dim]
        outputs.append(out)

    out_stacked = np.stack(outputs, axis=0).astype(nl.bfloat16)
    if tp_out:
        out_stacked = np.transpose(out_stacked, (0, 2, 1))

    return {"out": out_stacked}
