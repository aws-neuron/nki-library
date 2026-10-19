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
"""Torch reference for capacity-bounded unpermute_a2av."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives_torch import _to_numpy, _to_torch, get_pg
from .permute_a2av_torch import _validate_routed_row_permutation


def unpermute_a2av_torch_ref(
    output: np.ndarray,
    packed_source_indices: np.ndarray,
    send_counts: np.ndarray,
    recv_counts: np.ndarray,
    replica_group: ReplicaGroup,
    regroup_indices: np.ndarray = None,
    num_local_experts: int = 1,
) -> dict:
    """Emulate packed reverse A2AV and restore unique routed rows."""
    N = packed_source_indices.shape[0]
    EP = send_counts.shape[1]
    _, H = output.shape
    dtype = output.dtype
    pg = get_pg(replica_group)
    _validate_routed_row_permutation(packed_source_indices, send_counts)

    if not 1 <= num_local_experts <= 128:
        raise ValueError("num_local_experts must be in [1, 128]")
    if num_local_experts > 1:
        if regroup_indices is None or regroup_indices.shape != (output.shape[0],):
            raise ValueError("regroup_indices must have shape [C] for multiple local experts")
        if not np.array_equal(np.sort(regroup_indices), np.arange(output.shape[0])):
            raise ValueError("regroup_indices must be a full-capacity row permutation")
        source_major = np.empty_like(output)
        source_major[regroup_indices.astype(np.intp)] = output
        output = source_major

    send_total = int(recv_counts.sum())
    send_flat = _to_torch(output[:send_total, :]).reshape(-1)
    input_sizes = [int(recv_counts[0, peer]) * H for peer in range(EP)]
    output_sizes = [int(send_counts[0, peer]) * H for peer in range(EP)]
    recv_flat = torch.zeros(sum(output_sizes), dtype=send_flat.dtype)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=output_sizes,
        input_split_sizes=input_sizes,
        group=pg,
    )

    result = np.zeros((N, H), dtype=dtype)
    result[packed_source_indices.astype(np.intp), :] = _to_numpy(recv_flat.reshape(N, H), dtype)
    return {"result": result}
