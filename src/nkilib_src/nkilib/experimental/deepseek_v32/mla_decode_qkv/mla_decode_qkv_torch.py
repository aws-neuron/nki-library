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

"""Torch reference for the DeepSeek-V3.2 MLA decode QKV front-fold kernel."""

import torch


def mla_decode_qkv_torch_ref(
    hidden_states: torch.Tensor,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv_a: torch.Tensor,
    kv_norm_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_absorb_w: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    norm_eps: float = 1e-6,
    input_norm_w: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Reference for the MLA decode QKV front-fold.

    Emits the query tiles feature-major, matching the layout the kernel leaves in SBUF, plus this
    step's cache row.

    Note that ``wkv_a`` emits ``[kv_latent | k_pe]`` while a cache row stores ``[k_pe | kv_latent]``,
    so the two fields are concatenated in the inverted order here, exactly as the kernel writes them.

    Returns:
        ``output_0`` q_absorbed ``[kv_lora_rank, n_tokens, n_heads]``, ``output_1`` q_pe
        ``[qk_rope_dim, n_tokens, n_heads]``, ``output_2`` this step's cache row
        ``[n_tokens, kv_row_dim]``, and ``output_3`` the same row, which a caller reads back out of
        the cache to confirm the scatter landed on the slot ``slot_mapping`` names.
    """

    # kv_cache and slot_mapping only shape the scatter, which the caller verifies by reading the
    # written row back; the row contents this reference emits do not depend on either.
    del kv_cache, slot_mapping

    n_tokens = hidden_states.shape[0]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if input_norm_w is not None:
        hidden_states = _rmsnorm(hidden_states, input_norm_w, norm_eps)

    q_latent = _rmsnorm(torch.matmul(hidden_states, wq_a), q_norm_w, norm_eps)
    q = torch.matmul(q_latent, wq_b).view(n_tokens, n_heads, qk_head_dim)

    kv = torch.matmul(hidden_states, wkv_a)
    kv_latent = _rmsnorm(kv[..., :kv_lora_rank], kv_norm_w, norm_eps)
    k_pe = _rope_interleaved(kv[..., kv_lora_rank:].unsqueeze(1), cos.unsqueeze(1), sin.unsqueeze(1)).squeeze(1)

    q_nope = q[..., :qk_nope_head_dim]
    q_pe = _rope_interleaved(q[..., qk_nope_head_dim:], cos.unsqueeze(1), sin.unsqueeze(1))

    # [n_tokens, n_heads, kv_lora_rank] -> feature-major [kv_lora_rank, n_tokens, n_heads]
    q_absorbed = torch.einsum("thd,hdc->thc", q_nope, q_absorb_w)

    kv_row_cur = torch.cat([k_pe, kv_latent], dim=-1)

    return {
        "output_0": q_absorbed.permute(2, 0, 1).contiguous(),
        "output_1": q_pe.permute(2, 0, 1).contiguous(),
        "output_2": kv_row_cur,
        "output_3": kv_row_cur,
    }


def _rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMSNorm the way the model does, computing in fp32 with an fp32 gamma and casting back.

    Computes ``x * rsqrt(mean(x^2) + eps) * gamma``.
    """

    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps)
    return (gamma.to(torch.float32) * normalized).to(input_dtype)


def _rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies interleaved RoPE, pairing element ``2i`` with element ``2i + 1``.

    ``cos`` and ``sin`` carry half the rotated width, and the rotation is
    ``even' = even * cos - odd * sin`` and ``odd' = odd * cos + even * sin``.
    """

    even = x[..., ::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
