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

"""Torch/numpy reference for the Kimi Delta Attention (KDA) TKG kernel."""

import numpy as np

# Matches the kernel's L2-norm epsilon.
_L2_EPS = 1.0e-6


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _l2norm(x, axis=-1):
    """L2 normalize with epsilon inside the sqrt, matching the kernel's rsqrt(ss + eps)."""
    ss = np.sum(x * x, axis=axis, keepdims=True)
    return x * (1.0 / np.sqrt(ss + _L2_EPS))


def kda_tkg_torch_ref(
    q,  # [B, dk, num_head]      raw query (pre-L2-norm), channel-major
    k,  # [B, dk, num_head]      raw key (pre-L2-norm), channel-major
    v,  # [B, num_head, dv]      head-major
    g,  # [B, dk, num_head]      raw per-channel gate (pre-activation)
    beta,  # [B, 1, num_head]    beta logits (pre-sigmoid)
    a_log,  # [1, num_head]      log-gate parameter (shared across batch)
    dt_bias,  # [dk, num_head]   gate bias (shared across batch)
    out,  # [B, num_head, dv]        output placeholder (unused; kernel destination)
    S,  # [B, num_head, dk, dv]  recurrent state (IN/OUT in the kernel)
    scale,  # [1, 1]             query scale (1/sqrt(dk))
    lower_bound,  # [1, 1]       gate lower bound (in [-5, 0])
) -> dict:
    """Kimi Delta Attention token generation: one token per request.

    q/k/g are channel-major ([B, dk, num_head]) to match the kernel's DMA layout;
    v, out and the state are head-major.

    Returns:
        dict with
            out:   [B, num_head, dv]         -- attention output, head-major
            S: [B, num_head, dk, dv]  -- updated recurrent state
    """
    q_raw = np.asarray(q, dtype=np.float32)
    k_raw = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g_raw = np.asarray(g, dtype=np.float32)
    beta_logit = np.asarray(beta, dtype=np.float32)
    a_log = np.asarray(a_log, dtype=np.float32)
    dt_bias = np.asarray(dt_bias, dtype=np.float32)
    S_all = np.asarray(S, dtype=np.float32)
    scale = float(np.asarray(scale, dtype=np.float32).reshape(-1)[0])
    lower_bound = float(np.asarray(lower_bound, dtype=np.float32).reshape(-1)[0])

    B, dk, num_head = q_raw.shape
    dv = v.shape[-1]

    # --- Fused preprocessing (mirrors the kernel) ---
    # Transpose the channel-major inputs to head-major so the per-head math below
    # indexes [b, h] uniformly and the L2-norm reduces over the last axis.
    q_raw = np.ascontiguousarray(np.transpose(q_raw, (0, 2, 1)))  # [B, num_head, dk]
    k_raw = np.ascontiguousarray(np.transpose(k_raw, (0, 2, 1)))
    g_raw = np.ascontiguousarray(np.transpose(g_raw, (0, 2, 1)))
    dt_bias = np.ascontiguousarray(dt_bias.T)  # [num_head, dk]
    a_log = a_log.reshape(num_head, 1)  # [num_head, 1]
    beta_logit = beta_logit.reshape(B, num_head, 1)

    # a_log / dt_bias are per-head model parameters: broadcast over the batch.
    # g <= 0 since lower_bound <= 0 and sigmoid in (0, 1).
    g = lower_bound * _sigmoid(np.exp(a_log)[None, :, :] * (g_raw + dt_bias[None, :, :]))
    q = _l2norm(q_raw) * scale
    k = _l2norm(k_raw)
    beta = _sigmoid(beta_logit)  # [B, num_head, 1]

    out = np.zeros((B, num_head, dv), dtype=np.float32)
    S_out = np.zeros((B, num_head, dk, dv), dtype=np.float32)

    for b in range(B):
        for h in range(num_head):
            S_head = S_all[b, h]  # [dk, dv]
            exp_g = np.exp(g[b, h])  # [dk]

            # Decay folded into the key: (k * exp_g) @ S == k @ (Diag(exp_g) @ S).
            kv = (k[b, h] * exp_g) @ S_head  # [dv]
            diff = (v[b, h] - kv) * beta[b, h]  # [dv]
            S_head = S_head * exp_g[:, None] + np.outer(k[b, h], diff)

            out[b, h] = q[b, h] @ S_head
            S_out[b, h] = S_head

    return {"out": out, "S": S_out}
