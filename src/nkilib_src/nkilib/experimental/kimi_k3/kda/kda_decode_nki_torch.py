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

"""Torch/numpy reference for the Kimi Delta Attention (KDA) decode step.

This reference is the executable spec. Its parameter names must match the kernel
``kda_decode_step`` exactly (UnitTestFramework.validate_torch_ref_signature), and
the returned dict keys must match the test's output_tensor_descriptor.

KDA (Kimi Delta Attention) is GatedDeltaNet with a **per-channel** (diagonal)
forget gate instead of a per-head scalar gate. The math mirrors
``fused_recurrent_kda`` from flash-linear-attention
(https://github.com/fla-org/flash-linear-attention, MIT License) and the
forward kernel of FlashKDA (https://github.com/MoonshotAI/FlashKDA).

Single-token recurrence, per head h (state S is [dk, dv]):

    g     = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))   # [dk], <= 0
    q, k  = l2norm(q_raw), l2norm(k_raw)
    beta  = sigmoid(beta_logit)
    S     = Diag(exp(g)) @ S            # per-channel decay along dk
    delta = (v - k @ S) * beta          # [dv]
    S     = S + outer(k, delta)
    out   = scale * (q @ S)             # [dv]

The gate activation, QK L2-norm and beta sigmoid are all fused into the kernel
(upstream-style API), so this reference performs them too and consumes raw
pre-activation inputs.
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


def kda_decode_step_torch_ref(
    q_ref,  # [nv, dk]     raw query (pre-L2-norm)
    k_ref,  # [nv, dk]     raw key (pre-L2-norm)
    v_ref,  # [nv, dv]
    g_ref,  # [nv, dk]     raw per-channel gate (pre-activation)
    beta_ref,  # [nv, 1]      beta logits (pre-sigmoid)
    a_log_ref,  # [nv, 1]      log-gate parameter
    dt_bias_ref,  # [nv, dk]     gate bias
    out,  # [nv, dv]     output placeholder (unused; kernel destination)
    S_in_ref,  # [nv, dk, dv] recurrent state (IN/OUT in the kernel)
    scale_ref,  # [1, 1]       query scale (1/sqrt(dk))
    lower_bound_ref,  # [1, 1]       gate lower bound (in [-5, 0])
) -> dict:
    """Single-token KDA decode step.

    Returns:
        dict with
            out:   [nv, dv]      -- attention output
            S_in_ref: [nv, dk, dv]  -- updated recurrent state
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

    nv, dk = q_raw.shape
    dv = v.shape[-1]

    # --- Fused preprocessing (mirrors the kernel) ---
    # Per-channel gate: g <= 0 since lower_bound <= 0 and sigmoid in (0, 1).
    g = lower_bound * _sigmoid(np.exp(a_log) * (g_raw + dt_bias))  # [nv, dk]
    exp_g = np.exp(g)  # [nv, dk], in (0, 1]
    q = _l2norm(q_raw) * scale  # [nv, dk]
    k = _l2norm(k_raw)  # [nv, dk]
    beta = _sigmoid(beta_logit)  # [nv, 1]

    out = np.zeros((nv, dv), dtype=np.float32)
    S_out = np.zeros((nv, dk, dv), dtype=np.float32)

    for h in range(nv):
        S = S_in[h].astype(np.float32)  # [dk, dv]
        # Per-channel decay applied along the dk (partition) axis.
        S = S * exp_g[h][:, None]
        # kv = k @ S. Equivalently (k * exp_g) @ S_undecayed -- the form the
        # kernel uses, folding the decay into the [dk] key vector.
        kv = k[h] @ S  # [dv]
        delta = (v[h] - kv) * beta[h, 0]  # [dv]
        S = S + np.outer(k[h], delta)  # [dk, dv]
        out[h] = q[h] @ S  # [dv]
        S_out[h] = S

    return {"out": out, "S_in_ref": S_out}
