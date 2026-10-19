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

"""Torch/numpy reference for the Kimi-K3 Delta Attention prefill kernel.

This reference is the executable spec. Its parameter names must match the kernel
``kda_prefill`` exactly (UnitTestFramework.validate_torch_ref_signature),
and the returned dict keys must match the test's output_tensor_descriptor.

KDA (Kimi Delta Attention) is GatedDeltaNet with a **per-channel** (diagonal)
forget gate instead of a per-head scalar gate. The math follows FlashKDA
(https://github.com/MoonshotAI/FlashKDA) and flash-linear-attention's
``chunk_kda`` (https://github.com/fla-org/flash-linear-attention, MIT License).

Per sub-chunk of ``SUB = 16`` tokens, with state S [dk, dv] (equal to upstream's
transposed ``[V, K]`` state):

    gcum      = cumsum(g, axis=tokens)          # [SUB, dk], <= 0
    g_total   = gcum[-1]                        # [dk]
    k_decayed = k * exp(gcum)                   # entries <= |k|
    q_decayed = q * exp(gcum) * scale
    k_inv     = k * exp(-gcum)                  # LARGE: up to exp(5*SUB)
    k_restr   = k * exp(g_total - gcum)         # stable, exponent <= 0
    L         = strict_lower(k_decayed @ k_inv.T) * beta
    Mqk       = lower(q_decayed @ k_inv.T)
    INV       = (I + L)^-1                      # L nilpotent, L^SUB = 0
    v'        = (v - k_decayed @ S) * beta
    U         = INV @ v'
    out       = q_decayed @ S + Mqk @ U
    S         = S * exp(g_total)[:, None] + k_restr.T @ U

**Why SUB = 16.** ``k_inv = k * exp(-gcum)`` grows as ``exp(|lower_bound| * SUB)``.
With ``lower_bound = -5`` that is ``exp(80) ~ 5.5e34`` at SUB=16 (fine in fp32,
whose max is ~3.4e38) but ``exp(160)`` at SUB=32, which overflows. Unlike GDN's
scalar gate, the per-channel gate does not factor the pairwise decay into a
``[C, C]`` mask, so this ``k_inv`` form -- and its range limit -- is unavoidable.
The sub-chunk size is therefore forced by numerics, not by any hardware detail.

``(I + L)^-1`` is computed here by the same Neumann repeated-squaring product the
kernel uses, ``(I - L)(I + L^2)(I + L^4)(I + L^8)``, which is *exact* for a
strictly-lower-triangular ``L`` of size 16 (``L^16 = 0``). Note the leading factor
is ``(I - L)``, not ``(I + L)``: the delta rule needs the inverse of ``(I + L)``,
matching upstream and the sibling GDN kernel (which pre-negates its A instead).
"""

import numpy as np

# Matches the kernel's L2-norm epsilon and sub-chunk size.
_L2_EPS = 1.0e-6
_SUB = 16
# Squarings needed for the exact inverse: (I - L) times log2(SUB) - 1 factors
# (I + L^2)(I + L^4)...(I + L^(SUB/2)). Derived from _SUB rather than hardcoded,
# since too few factors silently returns a wrong inverse instead of failing.
_NEUMANN_SQUARINGS = _SUB.bit_length() - 2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _l2norm(x, axis=-1):
    """L2 normalize with epsilon inside the sqrt, matching the kernel's rsqrt(ss + eps)."""
    ss = np.sum(x * x, axis=axis, keepdims=True)
    return x * (1.0 / np.sqrt(ss + _L2_EPS))


def kda_prefill_torch_ref(
    q_ref,  # [nv, T, dk]   raw query (pre-L2-norm)
    k_ref,  # [nv, T, dk]   raw key (pre-L2-norm)
    v_ref,  # [nv, T, dv]
    g_ref,  # [nv, T, dk]   raw per-channel gate (pre-activation)
    beta_ref,  # [nv, T, 1]    beta logits (pre-sigmoid)
    a_log_ref,  # [nv, 1]       log-gate parameter
    dt_bias_ref,  # [nv, dk]      gate bias
    blockdiag_ref,  # [C, C]        block-diagonal strict-lower mask (16x16 blocks)
    blockdiag_incl_ref,  # [C, C]   block-diagonal lower-inclusive mask
    eye_ref,  # [C, C]        identity
    out,  # [nv, T, dv]   output placeholder (unused; kernel destination)
    state_ref,  # [nv, dk, dv]  recurrent state seed (IN/OUT in the kernel)
    seed_scale_ref,  # [1, 1]   0.0 fresh prefill / 1.0 APC continuation
    scale_ref,  # [1, 1]        query scale (1/sqrt(dk))
    lower_bound_ref,  # [1, 1]  gate lower bound (in [-5, 0])
) -> dict:
    """Kimi Delta Attention prefill.

    Returns:
        dict with
            out:         [nv, T, dv]   -- attention output
            state_ref:   [nv, dk, dv]  -- final recurrent state
    """
    q_raw = np.asarray(q_ref, dtype=np.float32)
    k_raw = np.asarray(k_ref, dtype=np.float32)
    v = np.asarray(v_ref, dtype=np.float32)
    g_raw = np.asarray(g_ref, dtype=np.float32)
    beta_logit = np.asarray(beta_ref, dtype=np.float32)
    a_log = np.asarray(a_log_ref, dtype=np.float32)
    dt_bias = np.asarray(dt_bias_ref, dtype=np.float32)
    state_seed = np.asarray(state_ref, dtype=np.float32)
    seed_scale = float(np.asarray(seed_scale_ref, dtype=np.float32).reshape(-1)[0])
    scale = float(np.asarray(scale_ref, dtype=np.float32).reshape(-1)[0])
    lower_bound = float(np.asarray(lower_bound_ref, dtype=np.float32).reshape(-1)[0])

    nv, T, dk = q_raw.shape
    dv = v.shape[-1]

    # --- Fused preprocessing (mirrors the kernel) ---
    # Per-channel gate: g <= 0 since lower_bound <= 0 and sigmoid in (0, 1).
    # a_log is per-head [nv, 1]; broadcast over tokens and channels.
    g = lower_bound * _sigmoid(np.exp(a_log)[:, :, None] * (g_raw + dt_bias[:, None, :]))
    q = _l2norm(q_raw) * scale
    k = _l2norm(k_raw)
    beta = _sigmoid(beta_logit)  # [nv, T, 1]

    eye = np.eye(_SUB, dtype=np.float32)
    strict_lower = np.tril(np.ones((_SUB, _SUB), dtype=np.float32), -1)
    lower_incl = np.tril(np.ones((_SUB, _SUB), dtype=np.float32))

    out = np.zeros((nv, T, dv), dtype=np.float32)
    final_state = np.zeros((nv, dk, dv), dtype=np.float32)

    for h in range(nv):
        S = state_seed[h] * seed_scale  # [dk, dv]

        for s in range(T // _SUB):
            t0 = s * _SUB
            sl = slice(t0, t0 + _SUB)
            q_s, k_s, v_s, g_s = q[h, sl], k[h, sl], v[h, sl], g[h, sl]
            b_s = beta[h, sl]  # [SUB, 1]

            gcum = np.cumsum(g_s, axis=0)  # [SUB, dk]
            g_total = gcum[-1]  # [dk]

            exp_gcum = np.exp(gcum)
            k_decayed = k_s * exp_gcum
            q_decayed = q_s * exp_gcum
            k_inv = k_s * np.exp(-gcum)
            # Stable form: the exponent (g_total - gcum) is <= 0 by construction,
            # so this never overflows. Do NOT compute it as k_inv * exp(g_total),
            # which multiplies a huge value by a tiny one.
            k_restored = k_s * np.exp(g_total - gcum)

            L = ((k_decayed @ k_inv.T) * strict_lower) * b_s
            Mqk = (q_decayed @ k_inv.T) * lower_incl

            # (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8), exact for L^16 = 0.
            # Note the sign: the delta rule needs the inverse of (I + L), matching
            # upstream and the sibling GDN kernel (which pre-negates its A instead).
            INV = eye - L
            Lp = L
            for _ in range(_NEUMANN_SQUARINGS):
                Lp = Lp @ Lp
                INV = INV + INV @ Lp

            v_prime = (v_s - k_decayed @ S) * b_s
            U = INV @ v_prime
            out[h, sl] = q_decayed @ S + Mqk @ U
            S = S * np.exp(g_total)[:, None] + k_restored.T @ U

        final_state[h] = S

    return {"out": out, "state_ref": final_state}
