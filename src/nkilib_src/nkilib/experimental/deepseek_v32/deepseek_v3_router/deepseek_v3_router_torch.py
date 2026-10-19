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

"""Torch reference for the DeepSeek-V3.2 group-limited ``noaux_tc`` router kernel."""

import torch

from . import RouterOutputs


def deepseek_v3_router_torch_ref(
    normed: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
    routed_scaling_factor: float,
    outputs: RouterOutputs = RouterOutputs.DENSE,
    output_in_sbuf: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Reference for the group-limited ``noaux_tc`` router.

    The correction bias is added after the sigmoid for **selection only**; the returned weights are
    gathered from the pre-bias scores, L1-normalized over the selected experts, then scaled.

    ``torch.topk`` stands in for the model's iterative amax/argmax formulation, which exists as a
    sort workaround and is numerically equivalent on tie-free inputs.

    Returns:
        ``output_0`` is the dense ``expert_affinities [n_tokens, n_experts]`` for ``DENSE``, or the
        ``topk_weight [n_tokens, top_k]`` for ``TOPK`` with ``output_1`` the indices. For
        ``DENSE_AND_TOPK`` the three are returned in the kernel's order: affinity, weight, indices.
    """

    # The memory space of the kernel's result has no meaning on the CPU side.
    del output_in_sbuf

    n_tokens = normed.shape[0]
    n_experts = router_weight.shape[0]
    experts_per_group = n_experts // n_group

    logits = torch.nn.functional.linear(normed.float(), router_weight.float())
    scores = logits.sigmoid()
    for_choice = scores + e_score_correction_bias.float().reshape(1, n_experts)

    # Each group is scored by the sum of its two best experts.
    grouped = for_choice.view(n_tokens, n_group, experts_per_group)
    group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)

    kept_groups = group_scores.topk(topk_group, dim=-1)[1]
    group_mask = torch.zeros(n_tokens, n_group, dtype=scores.dtype, device=normed.device)
    group_mask.scatter_(1, kept_groups, 1.0)
    keep = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(n_tokens, n_experts)

    masked = for_choice.masked_fill(keep == 0, float("-inf"))
    topk_idx = masked.topk(top_k, dim=-1)[1]

    topk_weight = scores.gather(1, topk_idx)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor

    affinity = torch.zeros(n_tokens, n_experts, dtype=torch.float32, device=normed.device)
    affinity.scatter_(1, topk_idx, topk_weight.to(torch.float32))

    if outputs == RouterOutputs.DENSE:
        return {"output_0": affinity}
    if outputs == RouterOutputs.TOPK:
        return {"output_0": topk_weight, "output_1": topk_idx.to(torch.int32)}
    return {"output_0": affinity, "output_1": topk_weight, "output_2": topk_idx.to(torch.int32)}
