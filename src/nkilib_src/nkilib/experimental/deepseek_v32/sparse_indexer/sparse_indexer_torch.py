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

"""PyTorch reference implementation of the DeepSeek-V3.2 sparse indexer."""

import torch

from . import MASK_BOUND, get_deepseek_v32_sparse_indexer_shape


def deepseek_v32_sparse_indexer_torch_ref(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wk: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    weights_proj: torch.Tensor,
    hadamard: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    positions: torch.Tensor,
    index_topk: int,
    valid_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation  for ``deepseek_v32_sparse_indexer``."""

    if hasattr(wq_b, "dtype") and "x4" in str(wq_b.dtype):
        raise TypeError(
            f"sparse_indexer_torch_ref needs a dequantized wq_b; got MX-packed {wq_b.dtype}. "
            "Dequantize with its e8m0 scale before calling."
        )

    del wq_b_scale

    shapes = get_deepseek_v32_sparse_indexer_shape(
        hidden_states, qr, wk, weights_proj, cos, key_cache, index_topk, valid_len
    )

    n_tokens, n_heads = shapes.n_tokens, shapes.n_heads
    rope_dim, head_dim = shapes.rope_dim, shapes.head_dim
    nope_head_dim = head_dim - rope_dim

    # Query
    query = qr @ wq_b
    query = query.view(n_tokens, n_heads, head_dim)
    query_pe, query_nope = torch.split(query, [rope_dim, nope_head_dim], dim=-1)
    query_pe = _apply_rotary_embedding(query_pe, cos[:, None, :], sin[:, None, :])
    query = torch.cat([query_pe, query_nope], dim=-1)
    query = query @ hadamard

    # Key
    key = hidden_states @ wk
    key = _layer_norm(key, k_norm_weight, k_norm_bias)
    key_pe, key_nope = torch.split(key, [rope_dim, nope_head_dim], dim=-1)
    key_pe = _apply_rotary_embedding(key_pe, cos, sin)
    key = torch.cat([key_pe, key_nope], dim=-1)
    key = key @ hadamard

    position = positions.reshape(-1).long()

    # Score
    cache = key_cache.clone()
    cache_indices = torch.arange(n_tokens)
    cache[cache_indices, position] = key

    weights = (hidden_states @ weights_proj) * shapes.n_heads_scale
    weights = weights.unsqueeze(-1) * shapes.softmax_scale

    raw = torch.einsum("thd,tsd->ths", query, cache)
    scores = (torch.relu(raw) * weights).sum(dim=1)

    slot = torch.arange(scores.shape[-1]).view(1, -1)
    scores = scores.masked_fill(slot > position.view(-1, 1), torch.finfo(torch.float32).min)
    scores = scores.clamp_min(-MASK_BOUND)

    # TopK
    k = min(shapes.index_topk, scores.shape[-1])
    _, indices = torch.topk(scores.float(), k, dim=-1, largest=True, sorted=True)
    topk = indices.flip(-1).to(torch.int32)

    return {"output_0": scores, "output_1": topk}


def _layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Performs a LayerNorm in float32.


    This function is intentionally used over ``torch.functional.layer_norm`` which lowers with an float64 eps value
    that the compiler rejects.
    """

    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(variance + eps)
    return normalized * weight.reshape(-1) + bias.reshape(-1)


def _apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply non-interleaved RoPe."""

    half = cos.shape[-1]
    low, high = x[..., :half], x[..., half : 2 * half]
    return torch.cat((low * cos - high * sin, high * cos + low * sin), dim=-1)
