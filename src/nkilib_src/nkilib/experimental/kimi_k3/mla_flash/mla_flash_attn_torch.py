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
"""fp32 torch reference for ``kimi_k3_mla_flash_attn`` (same inputs, merged output)."""

import torch


def keys_to_blocks(k_t: torch.Tensor, block: int = 2048) -> torch.Tensor:
    """``[H, qk, n*block]`` feature-major keys -> ``[H, n, qk, block]`` blocks."""
    heads, dq, span = k_t.shape
    return k_t.reshape(heads, dq, span // block, block).permute(0, 2, 1, 3).contiguous()


def values_to_blocks(v: torch.Tensor, block: int = 2048) -> torch.Tensor:
    """``[H, n*block, dv]`` key-major values -> ``[H, n, 128, block // 128, dv]`` blocks."""
    heads, span, dv = v.shape
    return v.reshape(heads, span // block, block // 128, 128, dv).permute(0, 1, 3, 2, 4).contiguous()


def blocks_to_values(v_blk: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`values_to_blocks`."""
    heads, n_blocks, p, subs, dv = v_blk.shape
    return v_blk.permute(0, 1, 3, 2, 4).reshape(heads, n_blocks * p * subs, dv)


def kimi_k3_mla_flash_attn_torch_ref(
    q_t: torch.Tensor,
    k_blk: torch.Tensor,
    marker: torch.Tensor,
    v_blk: torch.Tensor,
    kc_t: torch.Tensor,
    mc: torch.Tensor,
    vc: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``softmax(q . [cache | chunk] * scale) @ [v | vc]`` with the marker added to the cache
    scores and the chunk causal, in fp32. Returns ``[H, Sq, dv]`` (the merged output)."""
    heads, n_blk, dq, blk = k_blk.shape
    k_t = k_blk.permute(0, 2, 1, 3).reshape(heads, dq, n_blk * blk).float()
    v = blocks_to_values(v_blk).float()
    q = q_t.float().transpose(1, 2)  # [H, Sq, qk]
    sq = q.shape[1]
    mc0 = mc[0] if mc.dim() == 3 else mc
    scores = (
        torch.cat((q @ k_t + marker.float().unsqueeze(1), q @ kc_t.float() + mc0.float().unsqueeze(1)), dim=-1) * scale
    )
    span = k_t.shape[2]
    causal = torch.ones(sq, sq, dtype=torch.bool, device=q.device).tril()
    scores[..., span:] = scores[..., span:].masked_fill(~causal, float("-inf"))
    return scores.softmax(dim=-1) @ torch.cat((v, vc.float()), dim=1)
