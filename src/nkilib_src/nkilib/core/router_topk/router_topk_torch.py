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

import numpy as np
import torch
import torch.nn.functional as F
from neuronxcc.starfish.support import dtype as dt

from ..utils.common_types import RouterActFnType


def router_topk_torch_ref(
    x: np.ndarray,
    w: np.ndarray,
    w_bias: np.ndarray,
    router_logits: np.ndarray,
    expert_affinities: np.ndarray,
    expert_index: np.ndarray,
    act_fn: RouterActFnType,
    k: int,
    x_hbm_layout: int,
    x_sb_layout: int,
    output_in_sbuf: bool = False,
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
    expert_affin_in_sb: bool = False,
):
    """
    PyTorch reference implementation for router top-K kernel.

    This is a reference implementation for testing the NKI router_topk kernel.
    It implements the same mathematical operation using PyTorch operations.

    Args:
        x (np.ndarray): Input tensor, shape [H, T] or [T, H] depending on x_hbm_layout
        w (np.ndarray): Weight tensor [H, E]
        w_bias (np.ndarray): Optional bias tensor [1, E] or [E]
        router_logits (np.ndarray): Output router logits [T, E] (unused, for signature match)
        expert_affinities (np.ndarray): Output expert affinities [T, E] (unused, for signature match)
        expert_index (np.ndarray): Output expert indices [T, K] (unused, for signature match)
        act_fn (RouterActFnType): Activation function (SOFTMAX or SIGMOID)
        k (int): Number of top experts to select
        x_hbm_layout (int): Layout of x in HBM (0=[H,T], 1=[T,H])
        x_sb_layout (int): Layout of x in SBUF (unused in reference)
        output_in_sbuf (bool): Whether outputs are in SBUF (unused in reference)
        router_pre_norm (bool): If True, apply activation before top-K
        norm_topk_prob (bool): If True, normalize top-K probabilities with L1 norm
        use_column_tiling (bool): Enable PE array column tiling (unused in reference)
        use_indirect_dma_scatter (bool): Use indirect DMA for scatter (unused in reference)
        return_eager_affi (bool): If True, return top-K affinities (unused in reference)
        use_PE_broadcast_w_bias (bool): Use tensor engine for bias broadcast (unused in reference)
        shard_on_tokens (bool): Enable LNC sharding on tokens (unused in reference)
        skip_store_expert_index (bool): Skip storing expert indices (unused in reference)
        skip_store_router_logits (bool): Skip storing router logits (unused in reference)
        x_input_in_sbuf (bool): If True, x is in SBUF (affects layout interpretation)
        expert_affin_in_sb (bool): Whether expert affinities are in SBUF (unused in reference)

    Returns:
        dict: Dictionary containing 'router_logits', 'expert_index', and 'expert_affinities'

    Notes:
        - This implementation prioritizes clarity over performance
        - Hardware-specific parameters are ignored as they don't affect the mathematical result
        - Uses torch.nn.functional for softmax and torch.sigmoid for activation
    """
    # Convert inputs to torch tensors
    x_torch = torch.from_numpy(dt.static_cast(x, np.float32))
    w_torch = torch.from_numpy(dt.static_cast(w, np.float32))

    # Determine x layout: True if x is [T, H], False if [H, T]
    x_th_layout = x_input_in_sbuf or x_hbm_layout == 1  # XHBMLayout_T_H__1 = 1

    # Transpose x if needed to get [H, T]
    if x_th_layout:
        x_torch = x_torch.T

    # Compute router logits: [T, E]
    router_logits_torch = x_torch.T @ w_torch

    # Add bias if provided
    has_bias = w_bias is not None
    if has_bias:
        w_bias_torch = torch.from_numpy(dt.static_cast(w_bias, np.float32))
        router_logits_torch += w_bias_torch

    # Get dimensions
    T, E = router_logits_torch.shape

    # Get top-K indices
    ind = torch.argsort(-router_logits_torch, dim=-1)
    expert_index_torch = ind[..., :k]  # [T, k]

    # Compute expert affinities based on router_pre_norm flag
    if router_pre_norm:
        # ACT1 pipeline: activate full logits, then select top-K
        if act_fn == RouterActFnType.SOFTMAX:
            expert_affinities_full = F.softmax(router_logits_torch, dim=-1)
        elif act_fn == RouterActFnType.SIGMOID:
            expert_affinities_full = torch.sigmoid(router_logits_torch)
        else:
            raise NotImplementedError(f"Unsupported activation function: {act_fn}")

        if norm_topk_prob:
            # Scatter top-K values and normalize
            expert_affinities_select = torch.zeros((T, E))
            for token_idx in range(T):
                for topk_idx in range(k):
                    expert_idx = expert_index_torch[token_idx][topk_idx]
                    expert_affinities_select[token_idx][expert_idx] = expert_affinities_full[token_idx][expert_idx]

            # L1 normalization per token
            expert_affinities_torch = expert_affinities_select / torch.sum(
                expert_affinities_select, dim=1, keepdim=True
            )
        else:
            expert_affinities_torch = expert_affinities_full
    else:
        # ACT2 pipeline: gather top-K, activate, then scatter
        top_k_values = torch.zeros((T, k))
        for token_idx in range(T):
            for topk_idx in range(k):
                top_k_values[token_idx][topk_idx] = router_logits_torch[token_idx][
                    expert_index_torch[token_idx][topk_idx]
                ]

        # Apply activation to top-K values
        if act_fn == RouterActFnType.SOFTMAX:
            expert_affinities_topk = F.softmax(top_k_values, dim=-1)
        elif act_fn == RouterActFnType.SIGMOID:
            expert_affinities_topk = torch.sigmoid(top_k_values)
        else:
            raise NotImplementedError(f"Unsupported activation function: {act_fn}")

        # Scatter activated top-K values back to [T, E]
        expert_affinities_torch = torch.zeros((T, E))
        for token_idx in range(T):
            for topk_idx in range(k):
                expert_affinities_torch[token_idx][expert_index_torch[token_idx][topk_idx]] = expert_affinities_topk[
                    token_idx
                ][topk_idx]

    # Infer dtype from router_logits output tensor
    dtype = router_logits.dtype

    # Convert outputs back to numpy with correct dtype
    outputs = {
        "router_logits": dt.static_cast(router_logits_torch.numpy(), dtype),
        "expert_index": dt.static_cast(expert_index_torch.numpy(), np.uint32),
        "expert_affinities": dt.static_cast(expert_affinities_torch.numpy(), dtype),
    }

    return outputs
