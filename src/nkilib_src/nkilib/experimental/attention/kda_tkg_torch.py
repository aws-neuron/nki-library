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

"""Torch reference for the Kimi Delta Attention (KDA) TKG kernel.

Mirrors ``kda_tkg`` op for op. All math runs in float32 for CPU-reference stability,
independent of the input dtype. The prior numpy implementation of the same math is kept
alongside as ``kda_tkg_torch_numpy_reference.py.bak``; it validated against the kernel
and is the tie-breaker if this file and the kernel ever disagree.
"""

import torch

# Matches the kernel's L2-norm epsilon.
_L2_EPS = 1.0e-6


def _l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 normalize with the epsilon inside the sqrt, matching the kernel's rsqrt(ss + eps)."""
    ss = x.pow(2).sum(dim=dim, keepdim=True)
    return x * torch.rsqrt(ss + _L2_EPS)


def kda_tkg_torch_ref(
    q: torch.Tensor,  # [B, dk, num_head]      raw query (pre-L2-norm), channel-major
    k: torch.Tensor,  # [B, dk, num_head]      raw key (pre-L2-norm), channel-major
    v: torch.Tensor,  # [B, num_head, dv]      head-major
    g: torch.Tensor,  # [B, dk, num_head]      raw per-channel gate (pre-activation)
    beta: torch.Tensor,  # [B, 1, num_head]    beta logits (pre-sigmoid)
    a_log: torch.Tensor,  # [1, num_head]      log-gate parameter (shared across batch)
    dt_bias: torch.Tensor,  # [dk, num_head]   gate bias (shared across batch)
    out: torch.Tensor,  # [B, num_head, dv]    output placeholder (unused; kernel destination)
    S: torch.Tensor,  # [B, num_head, dk, dv]  recurrent state (IN/OUT in the kernel)
    scale: torch.Tensor,  # [1, 1]             query scale (1/sqrt(dk))
    lower_bound: torch.Tensor,  # [1, 1]       gate lower bound (in [-5, 0])
) -> dict:
    """Kimi Delta Attention token generation: one token per request.

    q/k/g are channel-major ([B, dk, num_head]) to match the kernel's DMA layout;
    v, out and the state are head-major.

    Returns:
        dict with
            out: [B, num_head, dv]      -- attention output, head-major
            S:   [B, num_head, dk, dv]  -- updated recurrent state
    """
    del out  # the kernel writes into this; the reference returns a fresh tensor

    # Transpose the channel-major inputs to head-major so every term below indexes
    # [b, h, ...] uniformly and the L2 norm reduces over the last axis.
    q_raw = q.to(torch.float32).transpose(1, 2)  # [B, num_head, dk]
    k_raw = k.to(torch.float32).transpose(1, 2)
    g_raw = g.to(torch.float32).transpose(1, 2)
    v_f = v.to(torch.float32)  # [B, num_head, dv]
    S_in = S.to(torch.float32)  # [B, num_head, dk, dv]

    batch, num_head = q_raw.shape[0], q_raw.shape[1]
    dt_bias_hm = dt_bias.to(torch.float32).t()  # [num_head, dk]
    a_log_hm = a_log.to(torch.float32).reshape(num_head, 1)  # [num_head, 1]
    beta_logit = beta.to(torch.float32).reshape(batch, num_head, 1)
    scale_f = scale.to(torch.float32).reshape(-1)[0]
    lower_bound_f = lower_bound.to(torch.float32).reshape(-1)[0]

    # --- Fused preprocessing (mirrors the kernel) ---
    # a_log / dt_bias are per-head model parameters, so they broadcast over the batch.
    # g <= 0 since lower_bound <= 0 and sigmoid is in (0, 1), so exp(g) is in (0, 1].
    g_act = lower_bound_f * torch.sigmoid(torch.exp(a_log_hm) * (g_raw + dt_bias_hm))
    exp_g = torch.exp(g_act)  # [B, num_head, dk]
    q_n = _l2norm(q_raw) * scale_f
    k_n = _l2norm(k_raw)
    beta_f = torch.sigmoid(beta_logit)  # [B, num_head, 1]

    # --- Recurrence, one token ---
    # Decay folded into the key: (k * exp_g) @ S == k @ (Diag(exp_g) @ S).
    kv = torch.einsum("bhk,bhkv->bhv", k_n * exp_g, S_in)
    diff = (v_f - kv) * beta_f  # [B, num_head, dv]
    # S = Diag(exp_g) @ S + outer(k, diff)
    S_out = S_in * exp_g.unsqueeze(-1) + k_n.unsqueeze(-1) * diff.unsqueeze(-2)
    out_f = torch.einsum("bhk,bhkv->bhv", q_n, S_out)

    return {"out": out_f, "S": S_out}
