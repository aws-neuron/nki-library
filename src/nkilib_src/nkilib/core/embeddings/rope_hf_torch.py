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

"""Torch reference implementation for rope_hf kernel (HuggingFace format)."""

from typing import Optional

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims of the input."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotate_half_backward(x: torch.Tensor) -> torch.Tensor:
    """Backward rotation of half dims."""
    half = x.shape[-1] // 2
    return torch.cat((x[..., half:], -x[..., :half]), dim=-1)


def rope_hf_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    rope_cache: Optional[torch.Tensor] = None,
    backward: bool = False,
) -> dict:
    """Torch reference for rope_hf kernel.

    Args:
        q: [batch_size, q_heads, seq_len, head_dim]
        k: [batch_size, k_heads, seq_len, head_dim]
        q_out, k_out: output placeholders (unused)
        cos: [batch_size, seq_len, head_dim] or None
        sin: [batch_size, seq_len, head_dim] or None
        rope_cache: [seq_len, head_dim*2] or None
        backward: If True, compute backward pass
    """
    if rope_cache is not None:
        half = rope_cache.shape[-1] // 2
        cos_val = rope_cache[..., :half]
        sin_val = rope_cache[..., half:]
        if cos_val.ndim == 2:
            cos_val = cos_val.unsqueeze(0)
            sin_val = sin_val.unsqueeze(0)
    else:
        cos_val = cos
        sin_val = sin

    # unsqueeze_dim=1 to broadcast over heads
    cos_val = cos_val.unsqueeze(1).to(q.dtype)
    sin_val = sin_val.unsqueeze(1).to(q.dtype)

    if backward:
        q_embed = q * cos_val + _rotate_half_backward(q * sin_val)
        k_embed = k * cos_val + _rotate_half_backward(k * sin_val)
    else:
        q_embed = q * cos_val + _rotate_half(q) * sin_val
        k_embed = k * cos_val + _rotate_half(k) * sin_val

    return {"q_out": q_embed, "k_out": k_embed}
