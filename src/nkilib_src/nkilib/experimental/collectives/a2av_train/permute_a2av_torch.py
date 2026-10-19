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
"""Torch reference for capacity-bounded permute_a2av."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives_torch import _to_numpy, _to_torch, get_pg
from ._packed_metadata_torch import _a2av_packed_regroup_indices_torch_ref


def _validate_routed_row_permutation(
    packed_source_indices: np.ndarray,
    rank_send_counts: np.ndarray,
) -> None:
    """Validate that peer segments partition a permutation of routed rows."""
    if packed_source_indices.ndim != 1 or packed_source_indices.dtype != np.int32:
        raise ValueError("packed_source_indices must be int32 [N]")
    num_rows = packed_source_indices.shape[0]
    rank_send_counts = np.asarray(rank_send_counts).reshape(-1)
    if rank_send_counts.dtype not in (np.int32, np.uint32, np.int64, np.uint64):
        raise ValueError("send_counts must be int32/uint32")
    if np.any(rank_send_counts < 0) or int(rank_send_counts.sum()) != num_rows:
        raise ValueError("send_counts must partition the packed vector")
    if not np.array_equal(np.sort(packed_source_indices), np.arange(num_rows)):
        raise ValueError("packed_source_indices must be a permutation of routed rows")


def permute_a2av_torch_ref(
    hidden_states: np.ndarray,
    packed_source_indices: np.ndarray,
    send_counts: np.ndarray,
    recv_capacity: int,
    replica_group: ReplicaGroup,
) -> dict:
    """Gather routed rows and emulate expert-major explicit-displacement A2AV."""
    num_rows, hidden = hidden_states.shape
    if send_counts.ndim != 2:
        raise ValueError(f"send_counts must have shape [EP, NLE], got {send_counts.shape}")
    ep_size = send_counts.shape[0]
    num_local_experts = send_counts.shape[1]
    rank_send_counts = send_counts.sum(axis=1).astype(np.int64)
    if int(rank_send_counts.sum()) != num_rows:
        raise ValueError(f"send_counts must sum to {num_rows}, got {rank_send_counts.sum()}")

    dtype = hidden_states.dtype
    pg = get_pg(replica_group)
    _validate_routed_row_permutation(packed_source_indices, rank_send_counts)

    send_count_chunks = [torch.from_numpy(send_counts[peer].copy()) for peer in range(ep_size)]
    recv_count_chunks = [torch.zeros(num_local_experts, dtype=send_count_chunks[0].dtype) for _ in range(ep_size)]
    dist.all_to_all(recv_count_chunks, send_count_chunks, group=pg)
    recv_expert_counts = np.stack([chunk.numpy() for chunk in recv_count_chunks]).astype(np.uint32)
    rank_recv_counts = recv_expert_counts.sum(axis=1).astype(np.uint32)
    recv_total = int(rank_recv_counts.sum())
    if recv_total > recv_capacity:
        raise ValueError(f"receive total {recv_total} exceeds capacity {recv_capacity}")

    torch_dtype = _to_torch(hidden_states[:1]).dtype
    send_chunks = []
    send_sizes = []
    offset = 0
    for peer in range(ep_size):
        count = int(rank_send_counts[peer])
        send_sizes.append(count * hidden)
        if count:
            indices = packed_source_indices[offset : offset + count].astype(np.intp)
            send_chunks.append(_to_torch(hidden_states[indices, :]).reshape(-1))
        else:
            send_chunks.append(torch.zeros(0, dtype=torch_dtype))
        offset += count
    send_flat = torch.cat(send_chunks)

    recv_sizes = [int(rank_recv_counts[peer]) * hidden for peer in range(ep_size)]
    recv_flat = torch.zeros(sum(recv_sizes), dtype=send_flat.dtype)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=pg,
    )

    recv_data = np.zeros((recv_capacity, hidden), dtype=dtype)
    if recv_total:
        source_major = _to_numpy(
            recv_flat.reshape(recv_total, hidden),
            dtype,
        )
        source_starts = np.concatenate([[0], np.cumsum(rank_recv_counts[:-1])])
        expert_offsets = np.cumsum(recv_expert_counts, axis=1) - recv_expert_counts
        output_offset = 0
        for expert in range(num_local_experts):
            for source in range(ep_size):
                count = int(recv_expert_counts[source, expert])
                input_offset = int(source_starts[source] + expert_offsets[source, expert])
                recv_data[output_offset : output_offset + count, :] = source_major[
                    input_offset : input_offset + count, :
                ]
                output_offset += count

    send_displs = np.concatenate([[0], np.cumsum(rank_send_counts[:-1])]).astype(np.uint32)
    recv_displs = np.concatenate([[0], np.cumsum(rank_recv_counts[:-1])]).astype(np.uint32)
    metadata = np.stack(
        [
            rank_send_counts.astype(np.uint32) * hidden,
            send_displs * hidden,
            rank_recv_counts * hidden,
            recv_displs * hidden,
        ]
    )
    if num_local_experts > 1:
        layout = _a2av_packed_regroup_indices_torch_ref(torch.from_numpy(recv_expert_counts), recv_capacity)
        regroup_indices = layout["regroup_indices"].numpy()
        bwmm_counts = layout["bwmm_counts"].numpy()
    else:
        regroup_indices = np.zeros(1, dtype=np.int32)
        bwmm_counts = np.array([recv_capacity], dtype=np.int32)
    return {
        "recv_data": recv_data,
        "metadata": metadata,
        "recv_counts": recv_expert_counts,
        "regroup_indices": regroup_indices,
        "bwmm_counts": bwmm_counts,
    }
