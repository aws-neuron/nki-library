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
"""Torch backward reference for the gdn_cte prefill family.

This module is the REFERENCE half of the gdn_cte backward: it defines what the device kernel
(`gdn_cte_bwd`) must compute, in two independent forms that are cross-checked against each
other in `test_gdn_cte_bwd.py`.

  `gdn_cte_bwd_torch_ref`  THE ORACLE. Runs `torch.autograd` straight through the per-token
                           forward reference (`gdn_cte_torch_ref`). No new math -- whatever
                           the forward computes, autograd differentiates exactly. This is the
                           ground truth the kernel is scored against.

  `gdn_cte_fwd_chunked` /  THE CHUNKED FORM the kernel implements. The forward is
  `gdn_cte_bwd_chunked`    re-expressed in the same chunk-level matmul primitives the schur
                           kernel uses (the decay matrix M, the interaction A, its inverse
                           T = (I + A)^-1, the per-token deltas U); the backward is the
                           hand-derived vector-Jacobian product of that graph, primitive by
                           primitive, plus the reverse-time state scan. Every step maps to a
                           matmul or an elementwise pass, so this is the derivation the kernel
                           follows -- listed step by step in `gdn_cte_bwd._backward_chunk`.

Both compute the same gradients. The chunked form is exact for ANY chunk length (chunking is
a pure tiling of the recurrence), so the tests run it at a tiny chunk against a finite-
difference check and at CHUNK=128 against the autograd oracle.

Gradients returned, for a scalar loss L: d_query, d_key, d_value, d_g_log, d_beta, and
d_recurrent_state (the gradient w.r.t. the initial state). All six are ALWAYS returned, as
`gdn_cte_bwd` returns them: dL/dS0 is the reverse scan's carry out of chunk 0 and is well
defined even when the forward started from an implicit zero state. Inputs to the backward
are `d_out` (dL/d out) and, optionally, `d_state` (dL/d final_state).
"""

from typing import NamedTuple, Optional

import torch

from ...core.utils.kernel_assert import kernel_assert
from .gdn_cte_utils import CHUNK, MAX_HEAD_DIM

_DEFAULT_GROUP_SIZE = 8
"""Mirrors `gdn_cte_bwd.DEFAULT_GROUP_SIZE`, exactly as `gdn_cte_torch` mirrors the forward's:
importing it would pull `nki` into a file that needs only torch. Inert here -- the oracle is
per-token -- but it must track the kernel default it claims signature parity with."""


class GdnCteGrads(NamedTuple):
    """Gradients of a scalar loss w.r.t. every differentiable gdn_cte input.

    Attributes:
        d_query (torch.Tensor): [B, Hk, S, Dk].
        d_key (torch.Tensor): [B, Hk, S, Dk].
        d_value (torch.Tensor): [B, Hv, S, Dv].
        d_g_log (torch.Tensor): [B, Hv, S].
        d_beta (torch.Tensor): [B, Hv, S].
        d_recurrent_state (torch.Tensor): [B, Hv, Dk, Dv]. Always present, even when the
            forward started from an implicit zero state -- that state is materialized so the
            gradient it receives is the same dL/dS0 the kernel returns.
    """

    d_query: torch.Tensor
    d_key: torch.Tensor
    d_value: torch.Tensor
    d_g_log: torch.Tensor
    d_beta: torch.Tensor
    d_recurrent_state: torch.Tensor


def _tri(mat: torch.Tensor, diagonal: int) -> torch.Tensor:
    """Lower-triangular part of the last two dims (batched `torch.tril`)."""
    return torch.tril(mat, diagonal=diagonal)


def _expand_kv_heads(query: torch.Tensor, key: torch.Tensor, ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicate key/query heads to the value-head count (GQA), matching the forward reference.

    Value head ``h`` reads key/query head ``h // ratio``; ``repeat_interleave`` on the head
    axis realizes exactly that mapping, and its adjoint is a group sum (see `_reduce_kv_heads`).
    """
    return query.repeat_interleave(ratio, dim=1), key.repeat_interleave(ratio, dim=1)


def _reduce_kv_heads(grad: torch.Tensor, n_k_heads: int, ratio: int) -> torch.Tensor:
    """Adjoint of `_expand_kv_heads`: sum the gradient over each shared head's value heads.

    Args:
        grad (torch.Tensor): [B, Hv, S, D] gradient on the expanded heads.
        n_k_heads (int): Hk.
        ratio (int): Hv // Hk.

    Returns:
        torch.Tensor: [B, Hk, S, D] gradient on the shared heads.
    """
    batch, _, seqlen, head_dim = grad.shape
    return grad.reshape(batch, n_k_heads, ratio, seqlen, head_dim).sum(dim=2)


def _forward_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    state0: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Chunk-parallel forward on heads already expanded to [B, Hv, ...].

    This is the schur kernel's own decomposition, in torch: per chunk it forms the decay
    matrix M, the strictly-lower interaction A, its inverse T = (I + A)^-1, the per-token
    deltas U = T b, then the chunk output and the next state. It is exact for any `chunk`.

    Args:
        query (torch.Tensor): [B, Hv, S, Dk] (already GQA-expanded).
        key (torch.Tensor): [B, Hv, S, Dk].
        value (torch.Tensor): [B, Hv, S, Dv].
        g_log (torch.Tensor): [B, Hv, S].
        beta (torch.Tensor): [B, Hv, S].
        scale (float): Query output scale.
        state0 (torch.Tensor): [B, Hv, Dk, Dv] initial state.
        chunk (int): Chunk length; S must be a multiple of it.

    Returns:
        tuple: (out [B, Hv, S, Dv], state_final [B, Hv, Dk, Dv], saved) where `saved` is a
            per-chunk list of the tensors the backward needs (incoming state and the chunk
            intermediates), so the backward does not recompute the forward.
    """
    seqlen = query.shape[2]
    n_chunks = seqlen // chunk
    eye = torch.eye(chunk, dtype=query.dtype, device=query.device)
    out = torch.zeros_like(value)
    state = state0
    saved = []
    for c in range(n_chunks):
        lo = c * chunk
        hi = lo + chunk
        q = query[:, :, lo:hi, :]
        k = key[:, :, lo:hi, :]
        v = value[:, :, lo:hi, :]
        g = g_log[:, :, lo:hi]
        b = beta[:, :, lo:hi]

        log_g = torch.cumsum(g, dim=-1)  # [B,Hv,C] inclusive prefix sum
        gate = torch.exp(log_g)  # G[s]
        gate_last = gate[..., -1:]  # G_last, [B,Hv,1]
        # M[r,s] = exp(log_g[r] - log_g[s]) for r >= s, else 0.
        decay = _tri(torch.exp(log_g[..., :, None] - log_g[..., None, :]), diagonal=0)

        kk = k @ k.transpose(-1, -2)  # [B,Hv,C,C] Gram over tokens
        qk = q @ k.transpose(-1, -2)
        # A = strict_lower(beta_r * M * (k_r . k_s)).
        a_mat = _tri(b[..., :, None] * decay * kk, diagonal=-1)
        inv = torch.linalg.inv(eye + a_mat)  # T = (I + A)^-1

        key_state = k @ state  # KS0[s] = S0^T k_s, [B,Hv,C,Dv]
        b_vec = b[..., :, None] * v - (b * gate)[..., :, None] * key_state
        deltas = inv @ b_vec  # U = T b, [B,Hv,C,Dv]

        query_state = q @ state  # QS0[r] = S0^T q_r
        w_qk = decay * qk  # M * (q_r . k_s), lower-triangular
        chunk_out = scale * (gate[..., :, None] * query_state + w_qk @ deltas)
        out[:, :, lo:hi, :] = chunk_out

        """ratio_s = G_last / G_s = exp(log_g_last - log_g_s). Form it from the log DIFFERENCE (<= 0,
        so ratio <= 1), NOT as the quotient G_last / G_s: under a large per-token decay both G_s and
        G_last underflow to 0 and the quotient is 0/0 = nan, exactly like `decay` above avoids."""
        ratio = torch.exp(log_g[..., -1:] - log_g)  # G_last / G_s
        k_decay = ratio[..., :, None] * k
        state_next = gate_last[..., None] * state + k_decay.transpose(-1, -2) @ deltas

        saved.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "b": b,
                "state": state,
                "log_g": log_g,
                "gate": gate,
                "gate_last": gate_last,
                "decay": decay,
                "kk": kk,
                "qk": qk,
                "inv": inv,
                "key_state": key_state,
                "deltas": deltas,
                "query_state": query_state,
                "w_qk": w_qk,
                "ratio": ratio,
                "k_decay": k_decay,
            }
        )
        state = state_next
    return out, state, saved


def _backward_chunk(saved: dict, d_out_chunk: torch.Tensor, d_state_new: torch.Tensor, scale: float) -> dict:
    """Vector-Jacobian product of one forward chunk.

    The step-by-step derivation is in `gdn_cte_bwd._backward_chunk`, whose docstring lists the
    whole chain in the order both this function and the kernel emit it.

    Args:
        saved (dict): The chunk's forward intermediates from `_forward_chunked`.
        d_out_chunk (torch.Tensor): [B, Hv, C, Dv], dL/d(this chunk's output).
        d_state_new (torch.Tensor): [B, Hv, Dk, Dv], dL/d(this chunk's OUTPUT state).
        scale (float): Query output scale.

    Returns:
        dict: d_q, d_k, d_v, d_g (log-gate grad), d_beta, and d_state0 (dL/d incoming state,
            handed to the previous chunk as its d_state_new).
    """
    q, k, v, b = saved["q"], saved["k"], saved["v"], saved["b"]
    state, gate, gate_last, decay = saved["state"], saved["gate"], saved["gate_last"], saved["decay"]
    kk, qk, inv = saved["kk"], saved["qk"], saved["inv"]
    key_state, deltas, query_state, w_qk = saved["key_state"], saved["deltas"], saved["query_state"], saved["w_qk"]
    ratio, k_decay = saved["ratio"], saved["k_decay"]
    inv_t = inv.transpose(-1, -2)

    # --- output: O = scale * (G * QS0 + Wqk @ U) ---
    d_os = scale * d_out_chunk
    d_query_state = gate[..., :, None] * d_os
    d_gate = (query_state * d_os).sum(-1)  # [B,Hv,C]
    d_w_qk = d_os @ deltas.transpose(-1, -2)  # [B,Hv,C,C]
    d_deltas = w_qk.transpose(-1, -2) @ d_os  # [B,Hv,C,Dv]

    # --- state out: S_new = G_last * S0 + Kdec^T @ U ---
    d_k_decay = deltas @ d_state_new.transpose(-1, -2)  # [B,Hv,C,Dk]
    d_deltas = d_deltas + k_decay @ d_state_new
    d_state0 = gate_last[..., None] * d_state_new  # G_last * dS_new
    # Kdec = ratio * k  ->  dk and d(ratio).
    d_k = ratio[..., :, None] * d_k_decay
    d_ratio = (k * d_k_decay).sum(-1)  # [B,Hv,C]
    """ratio_s = G_last / G_s (== `ratio`, saved by the forward, <= 1). Its gradient enters d_log_g in
    LOG-space, folded here, NOT through 1/G_s: writing it as d_gate += d_ratio*(-G_last/G_s^2) and
    d_gate_last += d_ratio/G_s and multiplying by G_s / G_last only at the log-space step below is
    ALGEBRAICALLY identical but OVERFLOWS -- once G_s = exp(cumsum g) underflows to 0 under a large
    per-token decay, 1/G_s^2 is +inf and the later *G_s gives inf*0 = nan. `ratio` is bounded, so
    d_ratio*ratio is finite; d_log_g[s] gets -d_ratio*ratio and d_log_g[last] gets sum_s(d_ratio*ratio)."""
    d_ratio_ratio = d_ratio * ratio  # [B,Hv,C], bounded -- the folded ratio-gradient
    d_gate_last_state = (state * d_state_new).sum((-1, -2)).unsqueeze(-1)  # S_new's G_last * S0

    # --- U = T b ---
    d_inv = d_deltas @ _b_vec(saved).transpose(-1, -2)  # dT = dU b^T
    d_b_vec = inv_t @ d_deltas  # db = T^T dU  (reuses T^T = the forward's Inv_T)

    # --- b = beta * V - (beta * G) * KS0 ---
    d_v = b[..., :, None] * d_b_vec
    d_beta = (v * d_b_vec).sum(-1) - gate * (key_state * d_b_vec).sum(-1)
    d_key_state = -(b * gate)[..., :, None] * d_b_vec
    d_gate = d_gate - b * (key_state * d_b_vec).sum(-1)

    # --- KS0 = k @ S0 ---
    d_k = d_k + d_key_state @ state.transpose(-1, -2)
    d_state0 = d_state0 + k.transpose(-1, -2) @ d_key_state

    # --- QS0 = q @ S0 ---
    d_q = d_query_state @ state.transpose(-1, -2)
    d_state0 = d_state0 + q.transpose(-1, -2) @ d_query_state

    # --- T = (I + A)^-1:  dA = -(T^T dT T^T), restricted to the strict-lower support of A ---
    d_a = -(inv_t @ d_inv @ inv_t)
    d_a = _tri(d_a, diagonal=-1)

    # --- A = strict_lower(beta_r * M * KK) ---
    d_beta = d_beta + (d_a * decay * kk).sum(-1)
    d_decay = b[..., :, None] * d_a * kk
    d_kk = b[..., :, None] * d_a * decay

    # --- Wqk = M * QK ---
    d_decay = d_decay + d_w_qk * qk
    d_qk = d_w_qk * decay

    # --- KK = k k^T ---
    d_k = d_k + (d_kk + d_kk.transpose(-1, -2)) @ k
    # --- QK = q k^T ---
    d_q = d_q + d_qk @ k
    d_k = d_k + d_qk.transpose(-1, -2) @ q

    # --- M[r,s] = exp(log_g[r] - log_g[s]) on r >= s ---
    d_decay_m = _tri(d_decay, diagonal=0) * decay
    d_log_g = d_decay_m.sum(-1) - d_decay_m.sum(-2)  # rowsum_s - colsum_r

    # --- G = exp(log_g), and G_last = exp(log_g[-1]) ---
    d_log_g = d_log_g + gate * d_gate - d_ratio_ratio  # ratio term folded (was gate * d_ratio*(-G_last/G_s^2))
    d_log_g[..., -1:] = (
        d_log_g[..., -1:] + gate_last * d_gate_last_state + d_ratio_ratio.sum(-1, keepdim=True)
    )  # ratio -> G_last term folded (was gate_last * sum_s(d_ratio/G_s))

    # --- log_g = cumsum(g_log): d(g_log)[i] = sum_{s >= i} d(log_g)[s] (reverse cumsum) ---
    d_g = torch.flip(torch.cumsum(torch.flip(d_log_g, dims=[-1]), dim=-1), dims=[-1])

    return {"d_q": d_q, "d_k": d_k, "d_v": d_v, "d_g": d_g, "d_beta": d_beta, "d_state0": d_state0}


def _b_vec(saved: dict) -> torch.Tensor:
    """Rebuild the chunk's b = beta * V - (beta * G) * KS0, exactly as in `_forward_chunked`.

    Needed by the dT = dU b^T step. Cheaper to recompute from saved intermediates than to
    also stash b in the forward.
    """
    b, v, gate, key_state = saved["b"], saved["v"], saved["gate"], saved["key_state"]
    return b[..., :, None] * v - (b * gate)[..., :, None] * key_state


def _validate_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    d_out: Optional[torch.Tensor] = None,
) -> None:
    """Shared shape/head-dim validation, mirroring the forward reference's contract.

    Every argument is checked here so a mismatch fails with a named tensor rather than as a torch
    broadcast error from inside `_forward_chunked`. `d_out` is only supplied on the backward path.
    """
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]
    kernel_assert(
        n_v_heads % n_k_heads == 0,
        f"n_v_heads must be divisible by n_k_heads, got {n_v_heads} and {n_k_heads}",
    )
    kernel_assert(
        key_head_dim == key.shape[-1],
        f"query and key must share a head dim, got {key_head_dim} and {key.shape[-1]}",
    )
    kernel_assert(
        key_head_dim <= MAX_HEAD_DIM and value_head_dim <= MAX_HEAD_DIM,
        f"head dims must be at most {MAX_HEAD_DIM}, got {key_head_dim} and {value_head_dim}",
    )
    kernel_assert(
        key.shape[:3] == (batch, n_k_heads, seqlen),
        f"key must be [{batch}, {n_k_heads}, {seqlen}, Dk], got {tuple(key.shape)}",
    )
    kernel_assert(
        value.shape[:3] == (batch, n_v_heads, seqlen),
        f"value must be [{batch}, {n_v_heads}, {seqlen}, Dv], got {tuple(value.shape)}",
    )
    # The gate and beta are per token, so they carry no head dim -- a [B, Hv, S, 1] gate would
    # otherwise broadcast silently against the [B, Hv, S, D] tiles instead of failing.
    kernel_assert(
        tuple(g_log.shape) == (batch, n_v_heads, seqlen),
        f"g_log must be [{batch}, {n_v_heads}, {seqlen}], got {tuple(g_log.shape)}",
    )
    kernel_assert(
        tuple(beta.shape) == (batch, n_v_heads, seqlen),
        f"beta must be [{batch}, {n_v_heads}, {seqlen}], got {tuple(beta.shape)}",
    )
    # `scale` is not checked and not taken: any real multiplier is meaningful.
    if d_out is not None:
        kernel_assert(
            tuple(d_out.shape) == (batch, n_v_heads, seqlen, value_head_dim),
            f"d_out must match value's shape [{batch}, {n_v_heads}, {seqlen}, {value_head_dim}], "
            f"got {tuple(d_out.shape)}",
        )


def gdn_cte_fwd_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    scale: float = 1.0,
    recurrent_state: Optional[torch.Tensor] = None,
    chunk_size: int = CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel forward reference (the schur decomposition), for validating the backward.

    Same result as `gdn_cte_torch_ref`, computed in the chunk-level matmul primitives the
    kernel uses. Exact for any `chunk_size` that divides S.

    Args mirror `gdn_cte_torch_ref`; `chunk_size` defaults to CHUNK=128.

    Returns:
        tuple: (out [B, Hv, S, Dv], final_state [B, Hv, Dk, Dv]).
    """
    _validate_shapes(query, key, value, g_log, beta)
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]
    kernel_assert(seqlen % chunk_size == 0, f"seqlen {seqlen} must be a multiple of chunk_size {chunk_size}")
    ratio = n_v_heads // n_k_heads

    query_e, key_e = _expand_kv_heads(query.double(), key.double(), ratio)
    if recurrent_state is None:
        state0 = torch.zeros((batch, n_v_heads, key_head_dim, value_head_dim), dtype=torch.float64)
    else:
        state0 = recurrent_state.double().clone()
    out, state_final, _ = _forward_chunked(
        query_e, key_e, value.double(), g_log.double(), beta.double(), scale, state0, chunk_size
    )
    return out, state_final


def gdn_cte_bwd_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    d_out: torch.Tensor,
    scale: float = 1.0,
    recurrent_state: Optional[torch.Tensor] = None,
    d_state: Optional[torch.Tensor] = None,
    chunk_size: int = CHUNK,
) -> GdnCteGrads:
    """Analytic chunked backward -- the derivation a device kernel would implement.

    Runs the chunked forward to recover each chunk's incoming state and intermediates, then a
    REVERSE-TIME scan over chunks: chunk c's state-gradient is the gradient the next chunk fed
    back (`d_state0`), plus `d_state` at the last chunk. Everything is computed in float64 for
    a clean finite-difference comparison; cast the inputs before calling to change that.

    Args:
        query, key, value, g_log, beta: The forward inputs (see `gdn_cte_torch_ref`).
        d_out (torch.Tensor): [B, Hv, S, Dv], dL/d out.
        scale (float): Query output scale.
        recurrent_state (Optional[torch.Tensor]): [B, Hv, Dk, Dv] initial state, or None.
        d_state (Optional[torch.Tensor]): [B, Hv, Dk, Dv], dL/d final_state, or None (zero).
        chunk_size (int): Chunk length; S must be a multiple of it.

    Returns:
        GdnCteGrads: gradients w.r.t. every differentiable input.
    """
    _validate_shapes(query, key, value, g_log, beta, d_out=d_out)
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]
    kernel_assert(seqlen % chunk_size == 0, f"seqlen {seqlen} must be a multiple of chunk_size {chunk_size}")
    ratio = n_v_heads // n_k_heads
    n_chunks = seqlen // chunk_size

    query_e, key_e = _expand_kv_heads(query.double(), key.double(), ratio)
    if recurrent_state is None:
        state0 = torch.zeros((batch, n_v_heads, key_head_dim, value_head_dim), dtype=torch.float64)
    else:
        state0 = recurrent_state.double().clone()

    _, _, saved = _forward_chunked(
        query_e, key_e, value.double(), g_log.double(), beta.double(), scale, state0, chunk_size
    )

    d_out_e = d_out.double()
    d_query_e = torch.zeros_like(query_e)
    d_key_e = torch.zeros_like(key_e)
    d_value = torch.zeros_like(value, dtype=torch.float64)
    d_g_log = torch.zeros_like(g_log, dtype=torch.float64)
    d_beta = torch.zeros_like(beta, dtype=torch.float64)

    d_state_new = torch.zeros_like(state0) if d_state is None else d_state.double().clone()
    for c in range(n_chunks - 1, -1, -1):
        lo = c * chunk_size
        hi = lo + chunk_size
        grads = _backward_chunk(saved[c], d_out_e[:, :, lo:hi, :], d_state_new, scale)
        d_query_e[:, :, lo:hi, :] = grads["d_q"]
        d_key_e[:, :, lo:hi, :] = grads["d_k"]
        d_value[:, :, lo:hi, :] = grads["d_v"]
        d_g_log[:, :, lo:hi] = grads["d_g"]
        d_beta[:, :, lo:hi] = grads["d_beta"]
        d_state_new = grads["d_state0"]

    d_query = _reduce_kv_heads(d_query_e, n_k_heads, ratio)
    d_key = _reduce_kv_heads(d_key_e, n_k_heads, ratio)
    # `d_state_new` is the scan's carry out of chunk 0 = dL/dS0, computed whether or not an
    # initial state was supplied. Returned on both paths, matching `gdn_cte_bwd`'s fixed arity.
    return GdnCteGrads(d_query, d_key, d_value, d_g_log, d_beta, d_state_new)


def gdn_cte_bwd_torch_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    d_out: torch.Tensor,
    mask_pack: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    group_size: int = _DEFAULT_GROUP_SIZE,
    recurrent_state: Optional[torch.Tensor] = None,
    d_state: Optional[torch.Tensor] = None,
) -> GdnCteGrads:
    """THE ORACLE: gradients via autograd straight through the per-token forward reference.

    This is the ground truth. It carries no backward math of its own -- it differentiates
    `gdn_cte_torch_ref` with `torch.autograd`, so it is correct by construction and is what
    both `gdn_cte_bwd_chunked` and the `gdn_cte_bwd` device kernel are scored against.

    The parameter list mirrors `gdn_cte_bwd`'s positionally: convention dispatch
    (`NKILIB_USE_TORCH_REF`) hands the ref the kernel's own call arguments.

    Args:
        query, key, value, g_log, beta: The forward inputs (see `gdn_cte_torch_ref`).
        d_out (torch.Tensor): [B, Hv, S, Dv], dL/d out.
        mask_pack (Optional[torch.Tensor]): [N_MASKS, CHUNK, CHUNK] constant mask pack.
            Accepted for signature parity with `gdn_cte_bwd`; carries no math here, and is
            VALIDATED when present so a corrupted pack fails with a clear message.
        scale (float): Query output scale.
        group_size (int): Accepted for signature parity with `gdn_cte_bwd`. Grouping is a
            scheduling transform and does not affect the result.
        recurrent_state (Optional[torch.Tensor]): [B, Hv, Dk, Dv] initial state, or None.
        d_state (Optional[torch.Tensor]): [B, Hv, Dk, Dv], dL/d final_state, or None (zero).

    Returns:
        GdnCteGrads: gradients w.r.t. every differentiable input.
    """
    # Imported here so the oracle path pulls in the forward reference only when used, keeping
    # this module importable without it for callers that only want the chunked form.
    from .gdn_cte_torch import _validate_mask_pack, gdn_cte_torch_ref
    from .gdn_cte_utils import ALGORITHM_NEUMANN

    kernel_assert(
        isinstance(group_size, int) and group_size >= 1,
        f"group_size must be a positive int, got {group_size}",
    )
    if mask_pack is not None:
        _validate_mask_pack(mask_pack)

    query = query.detach().float().requires_grad_(True)
    key = key.detach().float().requires_grad_(True)
    value = value.detach().float().requires_grad_(True)
    g_log = g_log.detach().float().requires_grad_(True)
    beta = beta.detach().float().requires_grad_(True)
    """dL/dS0 is returned on BOTH paths (as `gdn_cte_bwd` does), so the zero state an omitted
    `recurrent_state` implies is MATERIALIZED here: autograd cannot differentiate w.r.t. a
    value that never entered the graph. Zeros are what the forward reference uses for None,
    so the forward is unchanged -- this only makes the existing initial state visible."""
    state_in = (
        torch.zeros(value.shape[0], value.shape[1], query.shape[-1], value.shape[-1], dtype=torch.float32)
        if recurrent_state is None
        else recurrent_state.detach().float()
    ).requires_grad_(True)
    inputs = [query, key, value, g_log, beta, state_in]

    # The forward reference is algorithm-agnostic; ALGORITHM_NEUMANN just skips the mask-pack
    # requirement, which is a kernel input the math does not use.
    out, state_out = gdn_cte_torch_ref(
        query, key, value, g_log, beta, scale=scale, algorithm=ALGORITHM_NEUMANN, recurrent_state=state_in
    )
    d_out_t = d_out.float()
    d_state_t = torch.zeros_like(state_out) if d_state is None else d_state.float()
    grads = torch.autograd.grad(outputs=[out, state_out], grad_outputs=[d_out_t, d_state_t], inputs=inputs)
    return GdnCteGrads(grads[0], grads[1], grads[2], grads[3], grads[4], grads[5])
