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
"""Torch reference for gdn_cte kernel (chunked GDN prefill with gate)."""

from typing import Optional

import torch


def gdn_cte_torch_nki_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token recurrence reference matching gdn_cte kernel semantics.

    Args:
        q, k, v: [B, S, D] float32
        beta: [B, S] float32, sigmoid-gated update strength
        gate: [B, S] float32 or None, log-decay per token (negative).
              If None, no decay (gate=0).
        scale: float, query scaling factor (typically 1/sqrt(D))

    Returns:
        out: [B, S, D] float32
        state: [B, D, D] float32 (final recurrent state)
    """
    B, S, D = q.shape
    device = q.device
    q = q.float() * scale
    k = k.float()
    v = v.float()
    beta = beta.float()

    state = torch.zeros(B, D, D, device=device, dtype=torch.float32)
    outs = []

    for t in range(S):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t]

        if gate is not None:
            g_t = gate[:, t].float().exp()
            state = state * g_t[:, None, None]

        v_old = (state * k_t[:, :, None]).sum(1)
        delta = (v_t - v_old) * beta_t[:, None]
        state = state + k_t[:, :, None] * delta[:, None, :]

        o_t = (state * q_t[:, :, None]).sum(1)
        outs.append(o_t)

    out = torch.stack(outs, dim=1)
    return out, state
