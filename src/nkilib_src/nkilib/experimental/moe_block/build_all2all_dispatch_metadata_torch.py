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

"""Torch reference for build_all2all_dispatch_metadata."""

import torch


def build_all2all_dispatch_metadata_torch_ref(
    expert_index,
    num_experts,
    num_elements_per_token,
    replica_group_size,
):
    """CPU reference implementation of build_all2all_dispatch_metadata.

    Args:
        expert_index: [T, K] int32 tensor of expert indices per token.
        num_experts: Total number of experts.
        num_elements_per_token: Elements per token (e.g. H_CONCAT).
        replica_group_size: Number of destination ranks.

    Returns:
        [4, replica_group_size] uint32 tensor.
    """
    T, K = expert_index.shape
    R = replica_group_size
    n_local_experts = num_experts // replica_group_size

    # Map experts to destination ranks
    dst_ranks = (expert_index // n_local_experts).to(torch.int32)  # [T, K]

    # Build [T, R] mask with deduplication via scatter
    rank_mask = torch.zeros(T, R, dtype=torch.int32)
    rank_mask.scatter_(1, dst_ranks.long(), 1)

    # Per-rank send counts in elements
    send_counts = rank_mask.sum(dim=0).to(torch.int32) * num_elements_per_token

    # Exclusive prefix sum for displacements
    send_displs = torch.zeros(R, dtype=torch.int32)
    send_displs[1:] = torch.cumsum(send_counts[:-1], dim=0)

    # Stack: [4, R]
    return torch.stack(
        [
            send_counts,
            send_displs,
            torch.zeros(R, dtype=torch.int32),
            torch.zeros(R, dtype=torch.int32),
        ],
        dim=0,
    ).to(torch.uint32)
