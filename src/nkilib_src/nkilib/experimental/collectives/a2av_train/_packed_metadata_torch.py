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
"""PyTorch reference for packed-row inverse indices and expert counts."""

import torch


def _a2av_packed_regroup_indices_torch_ref(
    recv_expert_counts: torch.Tensor,
    recv_capacity: int,
) -> dict:
    """Reorder source-major row IDs by expert, leaving the capacity tail fixed."""
    counts = recv_expert_counts.to(torch.int64)
    source_rows = torch.arange(recv_capacity, dtype=torch.int32, device=counts.device)
    segments = torch.split(source_rows[: int(counts.sum())], counts.reshape(-1).tolist())
    num_sources, num_local_experts = counts.shape
    expert_major_segments = [
        segments[source * num_local_experts + expert]
        for expert in range(num_local_experts)
        for source in range(num_sources)
    ]
    expert_major_segments.append(source_rows[int(counts.sum()) :])
    bwmm_counts = counts.sum(dim=0).to(torch.int32)
    bwmm_counts[-1] += recv_capacity - int(counts.sum())
    return {
        "regroup_indices": torch.cat(expert_major_segments),
        "bwmm_counts": bwmm_counts,
    }
