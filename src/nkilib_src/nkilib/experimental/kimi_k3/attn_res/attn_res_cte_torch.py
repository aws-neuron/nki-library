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
"""Torch reference for ``attn_res_cte`` --- Kimi-K3's residual-mixture operator.

This is the exact algebra the NKI kernel implements, expressed in fp32 torch so a
kernel bug is distinguishable from a wiring bug. It equals
``vllm_neuron.model.kimi_k3.attn_res.apply_attn_res`` to fp32 round-off, with the one
difference that the score vector arrives already fused: the caller precomputes
``score_w = norm.weight * proj.weight.squeeze(0)`` and passes it in, rather than the
two weights separately. Everything else --- the ``rsqrt`` factoring out of the dot
product, the softmax over ``C+1`` candidates, and the mixture over the RAW candidates
(not the normalised ones) --- is identical.

Two details that are silent when wrong (carried over from ``apply_attn_res``):

1. The scores are computed on the RMS-normalised candidates, but the mixture averages
   the RAW ones. ``probs @ v``, never ``probs @ rmsnorm(v)``.
2. ``rsqrt(ss/H + eps)`` is a per-candidate scalar, so it factors straight out of the
   dot ``sum_h v[h] * score_w[h]``: ``s_c = dot_c * rsqrt(ss_c / H + eps)``. The kernel
   exploits exactly this --- two free-axis reductions per (token, candidate), no
   normalised copy of ``v`` ever materialised.
"""

from __future__ import annotations

import torch


def attn_res_cte_torch_ref(
    bank: torch.Tensor,
    stream: torch.Tensor,
    score_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mix the residual stream with its banked snapshots --- the kernel's algebra.

    Args:
        bank: ``[T, C, H]`` the ``C`` banked snapshot candidates per token
            (``C`` may be 0). Row-major, ``H`` contiguous.
        stream: ``[T, H]`` the current residual stream, the ``(C+1)``-th candidate.
        score_w: ``[H]`` the fused score vector (``norm.weight * proj.weight``),
            precomputed by the caller.
        eps: RMSNorm epsilon.

    Returns:
        ``[T, H]`` in ``stream``'s dtype.

    Notes:
        Per token ``t``, with candidates ``v_c = bank[t, c]`` for ``c < C`` and
        ``v_C = stream[t]``::

            ss_c  = sum_h v_c[h] ** 2
            dot_c = sum_h v_c[h] * score_w[h]
            s_c   = dot_c * rsqrt(ss_c / H + eps)     # == rmsnorm(v_c) . score_w
            p     = softmax_c(s)                       # over the C+1 candidates
            out   = sum_c p_c * v_c                    # mixes the RAW candidates

    Example:
        A single candidate makes the softmax degenerate, so this is the identity:

        >>> h = torch.randn(3, 8)
        >>> empty = h.new_zeros(3, 0, 8)
        >>> out = attn_res_cte_torch_ref(empty, h, torch.randn(8), 1e-6)
        >>> torch.allclose(out, h, atol=1e-6)
        True

        Two identical candidates return that same vector:

        >>> out = attn_res_cte_torch_ref(h.unsqueeze(1), h, torch.randn(8), 1e-6)
        >>> torch.allclose(out, h, atol=1e-5)
        True
    """
    T, C, H = bank.shape
    bankf = bank.float()  # [T, C, H]
    streamf = stream.float()  # [T, H]
    sw = score_w.float()  # [H]

    def score(x: torch.Tensor) -> torch.Tensor:
        """RMS-normalise over H and project to one score per row; ``x`` is [..., H]."""
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + eps) * sw).sum(-1)

    scores = torch.cat((score(bankf), score(streamf).unsqueeze(-1)), dim=-1)  # [T, C+1]
    probs = scores.softmax(-1)
    # Note: mixes the RAW candidates, not the normalised ones.
    mixed = (probs[:, :-1].unsqueeze(-1) * bankf).sum(1) + probs[:, -1:] * streamf
    return mixed.to(stream.dtype)
