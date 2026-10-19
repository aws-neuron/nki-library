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

"""PyTorch reference for Kimi-K3's flat noaux_tc router."""

import torch


def kimi_k3_router_topk_torch_ref(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return compact indices and pre-bias normalized weights."""
    logits = hidden_states.float() @ gate_weight.float().T
    scores = torch.sigmoid(logits)
    scores_for_choice = scores + e_score_correction_bias.float().reshape(1, -1)

    selected = []
    work = scores_for_choice.clone()
    for _ in range(top_k):
        idx = torch.argmax(work, dim=1, keepdim=True)
        selected.append(idx)
        work.scatter_(1, idx, float("-inf"))
    topk_idx = torch.cat(selected, dim=1)

    topk_weight = torch.gather(scores, 1, topk_idx)
    topk_weight = topk_weight / (topk_weight.sum(dim=1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * topk_weight.new_full((), routed_scaling_factor)
    return {
        "topk_weight": topk_weight.to(torch.float32),
        "topk_idx": topk_idx.to(torch.int32),
    }
