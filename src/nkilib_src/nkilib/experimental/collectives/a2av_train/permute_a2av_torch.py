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


def _validate_routed_row_permutation(
    packed_source_indices: np.ndarray,
    send_counts: np.ndarray,
) -> None:
    """Validate that peer segments partition a permutation of routed rows."""
    if packed_source_indices.ndim != 1 or packed_source_indices.dtype != np.int32:
        raise ValueError("packed_source_indices must be int32 [N]")
    num_rows = packed_source_indices.shape[0]
    if send_counts.ndim != 2 or send_counts.shape[0] != 1:
        raise ValueError("send_counts must have shape [1, EP]")
    if send_counts.dtype not in (np.int32, np.uint32):
        raise ValueError("send_counts must be int32/uint32")
    if np.any(send_counts < 0) or int(send_counts.sum()) != num_rows:
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
    """Gather routed rows and emulate packed explicit-displacement A2AV."""
    N, H = hidden_states.shape
    EP = send_counts.shape[1]
    dtype = hidden_states.dtype
    pg = get_pg(replica_group)
    _validate_routed_row_permutation(packed_source_indices, send_counts)
    if packed_source_indices.shape != (N,):
        raise ValueError("packed_source_indices must match the input row count")

    send_count_chunks = [torch.tensor([int(send_counts[0, peer])], dtype=torch.int64) for peer in range(EP)]
    recv_count_chunks = [torch.zeros(1, dtype=torch.int64) for _ in range(EP)]
    dist.all_to_all(recv_count_chunks, send_count_chunks, group=pg)
    recv_counts = np.array([[int(chunk[0]) for chunk in recv_count_chunks]], dtype=np.uint32)
    recv_total = int(recv_counts.sum())
    if recv_total > recv_capacity:
        raise ValueError(f"receive total {recv_total} exceeds capacity {recv_capacity}")

    torch_dtype = _to_torch(hidden_states[:1]).dtype
    send_chunks = []
    send_sizes = []
    offset = 0
    for peer in range(EP):
        count = int(send_counts[0, peer])
        send_sizes.append(count * H)
        if count:
            indices = packed_source_indices[offset : offset + count].astype(np.intp)
            send_chunks.append(_to_torch(hidden_states[indices, :]).reshape(-1))
        else:
            send_chunks.append(torch.zeros(0, dtype=torch_dtype))
        offset += count
    send_flat = torch.cat(send_chunks)

    recv_sizes = [int(recv_counts[0, peer]) * H for peer in range(EP)]
    recv_flat = torch.zeros(sum(recv_sizes), dtype=send_flat.dtype)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=pg,
    )

    recv_data = np.zeros((recv_capacity, H), dtype=dtype)
    if recv_total:
        recv_data[:recv_total, :] = _to_numpy(recv_flat.reshape(recv_total, H), dtype)

    send_displs = np.concatenate([[0], np.cumsum(send_counts[0, :-1])]).astype(np.uint32)
    recv_displs = np.concatenate([[0], np.cumsum(recv_counts[0, :-1])]).astype(np.uint32)
    metadata = np.stack(
        [
            send_counts[0].astype(np.uint32) * H,
            send_displs * H,
            recv_counts[0] * H,
            recv_displs * H,
        ]
    )
    return {"recv_data": recv_data, "metadata": metadata}
