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

"""Torch/numpy reference for the BATCHED Kimi Delta Attention (KDA) decode step.

This reference is the executable spec. Its parameter names must match the kernel
``kda_decode_batch`` exactly (UnitTestFramework.validate_torch_ref_signature),
and the returned dict keys must match the test's output_tensor_descriptor.

Per request b and head h, with a per-channel gate g[dk] (all entries <= 0) and
state S [dk, dv]:

    S    = S * exp(g)[:, None]      # per-channel decay along dk
    kv   = (k * exp(g)) @ S         # [dv]; equals k @ (Diag(exp(g)) @ S)
    diff = (v - kv) * beta
    S    = S + outer(k, diff)
    out  = q @ S                    # q carries the 1/sqrt(dk) scale

The batch axis is trivially parallel: requests share no state. ``a_log`` and
``dt_bias`` are model parameters and carry no batch axis.
"""

import numpy as np

# Matches the kernel's L2-norm epsilon.
_L2_EPS = 1.0e-6


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _l2norm(x, axis=-1):
    """L2 normalize with epsilon inside the sqrt, matching the kernel's rsqrt(ss + eps)."""
    ss = np.sum(x * x, axis=axis, keepdims=True)
    return x * (1.0 / np.sqrt(ss + _L2_EPS))


def kda_decode_batch_torch_ref(
    q_ref,  # [B, nv, dk]      raw query (pre-L2-norm)
    k_ref,  # [B, nv, dk]      raw key (pre-L2-norm)
    v_ref,  # [B, nv, dv]
    g_ref,  # [B, nv, dk]      raw per-channel gate (pre-activation)
    beta_ref,  # [B, nv, 1]       beta logits (pre-sigmoid)
    a_log_ref,  # [nv, 1]          log-gate parameter (shared across batch)
    dt_bias_ref,  # [nv, dk]         gate bias (shared across batch)
    out,  # [B, nv, dv]      output placeholder (unused; kernel destination)
    S_in_ref,  # [B, nv, dk, dv]  recurrent state (IN/OUT in the kernel)
    scale_ref,  # [1, 1]           query scale (1/sqrt(dk))
    lower_bound_ref,  # [1, 1]           gate lower bound (in [-5, 0])
) -> dict:
    """Batched single-token Kimi Delta Attention decode step.

    Returns:
        dict with
            out:   [B, nv, dv]      -- attention output
            S_in_ref: [B, nv, dk, dv]  -- updated recurrent state
    """
    q_raw = np.asarray(q_ref, dtype=np.float32)
    k_raw = np.asarray(k_ref, dtype=np.float32)
    v = np.asarray(v_ref, dtype=np.float32)
    g_raw = np.asarray(g_ref, dtype=np.float32)
    beta_logit = np.asarray(beta_ref, dtype=np.float32)
    a_log = np.asarray(a_log_ref, dtype=np.float32)
    dt_bias = np.asarray(dt_bias_ref, dtype=np.float32)
    S_in = np.asarray(S_in_ref, dtype=np.float32)
    scale = float(np.asarray(scale_ref, dtype=np.float32).reshape(-1)[0])
    lower_bound = float(np.asarray(lower_bound_ref, dtype=np.float32).reshape(-1)[0])

    B, nv, dk = q_raw.shape
    dv = v.shape[-1]

    # --- Fused preprocessing (mirrors the kernel) ---
    # a_log / dt_bias are [nv, ...] model parameters: broadcast over the batch.
    # g <= 0 since lower_bound <= 0 and sigmoid in (0, 1).
    g = lower_bound * _sigmoid(np.exp(a_log)[None, :, :] * (g_raw + dt_bias[None, :, :]))
    q = _l2norm(q_raw) * scale
    k = _l2norm(k_raw)
    beta = _sigmoid(beta_logit)  # [B, nv, 1]

    out = np.zeros((B, nv, dv), dtype=np.float32)
    S_out = np.zeros((B, nv, dk, dv), dtype=np.float32)

    for b in range(B):
        for h in range(nv):
            S = S_in[b, h]  # [dk, dv]
            exp_g = np.exp(g[b, h])  # [dk]

            # Decay folded into the key: (k * exp_g) @ S == k @ (Diag(exp_g) @ S).
            kv = (k[b, h] * exp_g) @ S  # [dv]
            diff = (v[b, h] - kv) * beta[b, h]  # [dv]
            S = S * exp_g[:, None] + np.outer(k[b, h], diff)

            out[b, h] = q[b, h] @ S
            S_out[b, h] = S

    return {"out": out, "S_in_ref": S_out}
