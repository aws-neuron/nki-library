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

"""PyTorch reference implementation for router top-K kernel."""

import torch
import torch.nn.functional as F

from ..utils.common_types import RouterActFnType


def router_topk_torch_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bias: torch.Tensor,
    router_logits: torch.Tensor,
    expert_affinities: torch.Tensor,
    expert_index: torch.Tensor,
    act_fn: RouterActFnType,
    k: int,
    x_hbm_layout: int,
    x_sb_layout: int,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    use_column_tiling: bool = False,
    use_indirect_dma_scatter: bool = False,
    return_eager_affi: bool = False,
    use_PE_broadcast_w_bias: bool = False,
    shard_on_tokens: bool = False,
    skip_store_expert_index: bool = False,
    skip_store_router_logits: bool = False,
    x_input_in_sbuf: bool = False,
):
    """
    PyTorch reference implementation for router top-K kernel.

    Args:
        x: Input tensor, shape [H, T] or [T, H] depending on x_hbm_layout
        w: Weight tensor [H, E]
        w_bias: Optional bias tensor [1, E] or [E]
        router_logits: Output router logits [T, E] (unused, for signature match)
        expert_affinities: Output expert affinities [T, E] (unused, for signature match)
        expert_index: Output expert indices [T, K] (unused, for signature match)
        act_fn: Activation function (SOFTMAX or SIGMOID)
        k: Number of top experts to select
        x_hbm_layout: Layout of x in HBM (0=[H,T], 1=[T,H])
        x_sb_layout: Layout of x in SBUF (unused in reference)
        router_pre_norm: If True, apply activation before top-K
        norm_topk_prob: If True, normalize top-K probabilities with L1 norm
        x_input_in_sbuf: If True, x is in SBUF (affects layout interpretation)

    Returns:
        dict: Dictionary containing 'router_logits', 'expert_index', and 'expert_affinities' as torch tensors
    """
    # Determine x layout: True if x is [T, H], False if [H, T]
    x_th_layout = x_input_in_sbuf or x_hbm_layout == 1

    # Transpose x if needed to get [H, T]
    x_work = x.T if x_th_layout else x

    # Compute router logits: [T, E]
    router_logits_out = x_work.T @ w

    # Add bias if provided
    if w_bias is not None:
        router_logits_out = router_logits_out + w_bias

    # Get dimensions
    T, E = router_logits_out.shape

    # Get top-K indices
    ind = torch.argsort(-router_logits_out, dim=-1)
    expert_index_out = ind[..., :k]  # [T, k]

    # Compute expert affinities based on router_pre_norm flag
    if router_pre_norm:
        # ACT1 pipeline: activate full logits, then select top-K
        if act_fn == RouterActFnType.SOFTMAX:
            expert_affinities_full = F.softmax(router_logits_out, dim=-1)
        elif act_fn == RouterActFnType.SIGMOID:
            expert_affinities_full = torch.sigmoid(router_logits_out)
        else:
            raise NotImplementedError(f"Unsupported activation function: {act_fn}")

        if norm_topk_prob:
            # Scatter top-K values and normalize
            expert_affinities_select = torch.zeros((T, E))
            for token_idx in range(T):
                for topk_idx in range(k):
                    expert_idx = expert_index_out[token_idx][topk_idx]
                    expert_affinities_select[token_idx][expert_idx] = expert_affinities_full[token_idx][expert_idx]

            # L1 normalization per token
            expert_affinities_out = expert_affinities_select / torch.sum(expert_affinities_select, dim=1, keepdim=True)
        else:
            expert_affinities_out = expert_affinities_full
    else:
        # ACT2 pipeline: gather top-K, activate, then scatter
        top_k_values = torch.zeros((T, k))
        for token_idx in range(T):
            for topk_idx in range(k):
                top_k_values[token_idx][topk_idx] = router_logits_out[token_idx][expert_index_out[token_idx][topk_idx]]

        # Apply activation to top-K values
        if act_fn == RouterActFnType.SOFTMAX:
            expert_affinities_topk = F.softmax(top_k_values, dim=-1)
        elif act_fn == RouterActFnType.SIGMOID:
            expert_affinities_topk = torch.sigmoid(top_k_values)
        else:
            raise NotImplementedError(f"Unsupported activation function: {act_fn}")

        # Scatter activated top-K values back to [T, E]
        expert_affinities_out = torch.zeros((T, E))
        for token_idx in range(T):
            for topk_idx in range(k):
                expert_affinities_out[token_idx][expert_index_out[token_idx][topk_idx]] = expert_affinities_topk[
                    token_idx
                ][topk_idx]

    return {
        "router_logits": router_logits_out,
        "expert_index": expert_index_out,
        "expert_affinities": expert_affinities_out,
    }
