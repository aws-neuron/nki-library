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
"""CPU tests for the gdn_cte backward REFERENCE (the torch forms, not the device kernel).

Three independent checks, chained so each backward form is pinned to something it cannot be
wrong about together with:

  1. chunked forward   == per-token forward reference        (the decomposition is exact)
  2. analytic backward == finite differences of the forward  (the hand-derived VJP is correct)
  3. analytic backward == autograd oracle                    (both tie to the repo reference)

All pure torch on CPU -- these validate math, not a kernel, so they need no Trainium. The
`gdn_cte_bwd` device kernel is scored against the same oracle in `test_gdn_cte_bwd_device.py`.
"""

import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.gdn.gdn_cte_bwd_torch import (
    gdn_cte_bwd_chunked,
    gdn_cte_bwd_torch_ref,
    gdn_cte_fwd_chunked,
)
from nkilib_src.nkilib.experimental.gdn.gdn_cte_torch import gdn_cte_torch_ref
from nkilib_src.nkilib.experimental.gdn.gdn_cte_utils import ALGORITHM_NEUMANN


def _make_inputs(batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, with_state, seed=42):
    """Forward inputs plus random cotangents d_out / d_state, in the forward's distribution."""
    rng = np.random.RandomState(seed)

    def l2_normed(shape):
        arr = rng.randn(*shape).astype(np.float64)
        return arr / np.sqrt((arr * arr).sum(axis=-1, keepdims=True) + 1e-6)

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    query = to_t(l2_normed((batch, n_k_heads, seqlen, key_head_dim)))
    key = to_t(l2_normed((batch, n_k_heads, seqlen, key_head_dim)))
    value = to_t(rng.randn(batch, n_v_heads, seqlen, value_head_dim).astype(np.float64) * 0.5)
    g_log = to_t(-np.abs(rng.randn(batch, n_v_heads, seqlen).astype(np.float64)) * 0.05)
    beta = to_t(1.0 / (1.0 + np.exp(-rng.randn(batch, n_v_heads, seqlen).astype(np.float64))))
    d_out = to_t(rng.randn(batch, n_v_heads, seqlen, value_head_dim).astype(np.float64))
    state = to_t(rng.randn(batch, n_v_heads, key_head_dim, value_head_dim).astype(np.float64)) if with_state else None
    d_state = to_t(rng.randn(batch, n_v_heads, key_head_dim, value_head_dim).astype(np.float64)) if with_state else None
    scale = float(1.0 / np.sqrt(key_head_dim))
    return query, key, value, g_log, beta, d_out, state, d_state, scale


# (batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size)
_FD_SHAPES = [
    (1, 1, 1, 16, 4, 4, 8),  # single head, two chunks
    (2, 1, 1, 24, 5, 4, 8),  # batch, Dk != Dv, three chunks
    (1, 1, 2, 16, 4, 6, 8),  # GQA: one key head shared by two value heads
]


@pytest.mark.parametrize("batch,n_k_heads,n_v_heads,seqlen,key_head_dim,value_head_dim,chunk_size", _FD_SHAPES)
@pytest.mark.parametrize("with_state", [False, True])
def test_chunked_forward_matches_per_token(
    batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size, with_state
):
    """The chunk-parallel forward reproduces the per-token forward reference exactly."""
    query, key, value, g_log, beta, _, state, _, scale = _make_inputs(
        batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, with_state
    )
    out_ref, state_ref = gdn_cte_torch_ref(
        query, key, value, g_log, beta, scale=scale, algorithm=ALGORITHM_NEUMANN, recurrent_state=state
    )
    out_chunk, state_chunk = gdn_cte_fwd_chunked(
        query, key, value, g_log, beta, scale=scale, recurrent_state=state, chunk_size=chunk_size
    )
    # The per-token reference computes in float32 (it casts inputs to .float()), so the
    # comparison floor is fp32 accumulation error, not the chunked form's own accuracy.
    torch.testing.assert_close(out_chunk, out_ref.double(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(state_chunk, state_ref.double(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch,n_k_heads,n_v_heads,seqlen,key_head_dim,value_head_dim,chunk_size", _FD_SHAPES)
@pytest.mark.parametrize("with_state", [False, True])
def test_analytic_backward_matches_finite_difference(
    batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size, with_state
):
    """The hand-derived chunked VJP matches central finite differences of the forward.

    The loss is L = <d_out, out> + <d_state, final_state>, so its input gradients are exactly
    what the backward returns for those cotangents.
    """
    query, key, value, g_log, beta, d_out, state, d_state, scale = _make_inputs(
        batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, with_state
    )

    def loss(q, k, v, g, b, st):
        out, state_final = gdn_cte_fwd_chunked(q, k, v, g, b, scale=scale, recurrent_state=st, chunk_size=chunk_size)
        total = (out * d_out).sum()
        if d_state is not None:
            total = total + (state_final * d_state).sum()
        return total

    grads = gdn_cte_bwd_chunked(
        query,
        key,
        value,
        g_log,
        beta,
        d_out,
        scale=scale,
        recurrent_state=state,
        d_state=d_state,
        chunk_size=chunk_size,
    )
    analytic = {
        "query": grads.d_query,
        "key": grads.d_key,
        "value": grads.d_value,
        "g_log": grads.d_g_log,
        "beta": grads.d_beta,
    }
    if with_state:
        analytic["state"] = grads.d_recurrent_state

    tensors = {"query": query, "key": key, "value": value, "g_log": g_log, "beta": beta}
    if with_state:
        tensors["state"] = state
    eps = 1e-6
    for name, tensor in tensors.items():
        fd = torch.zeros_like(tensor)
        flat = tensor.reshape(-1)
        fd_flat = fd.reshape(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            args = {k: (v if k != name else tensor) for k, v in tensors.items()}
            plus = loss(args["query"], args["key"], args["value"], args["g_log"], args["beta"], args.get("state"))
            flat[i] = orig - eps
            minus = loss(args["query"], args["key"], args["value"], args["g_log"], args["beta"], args.get("state"))
            flat[i] = orig
            fd_flat[i] = (plus - minus) / (2 * eps)
        torch.testing.assert_close(analytic[name], fd, rtol=1e-4, atol=1e-4, msg=f"grad mismatch for {name}")


# Kernel-scale shapes at the real chunk length, to confirm the derivation holds at CHUNK=128.
# (batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size)
_ORACLE_SHAPES = [
    (1, 1, 1, 256, 16, 16, 128),
    (2, 1, 1, 384, 16, 24, 128),
    (1, 2, 4, 256, 16, 16, 128),  # GQA
]


@pytest.mark.parametrize("batch,n_k_heads,n_v_heads,seqlen,key_head_dim,value_head_dim,chunk_size", _ORACLE_SHAPES)
@pytest.mark.parametrize("with_state", [False, True])
def test_analytic_backward_matches_autograd_oracle(
    batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size, with_state
):
    """The analytic chunked backward equals autograd through the per-token forward reference."""
    query, key, value, g_log, beta, d_out, state, d_state, scale = _make_inputs(
        batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, with_state
    )
    analytic = gdn_cte_bwd_chunked(
        query,
        key,
        value,
        g_log,
        beta,
        d_out,
        scale=scale,
        recurrent_state=state,
        d_state=d_state,
        chunk_size=chunk_size,
    )
    oracle = gdn_cte_bwd_torch_ref(
        query, key, value, g_log, beta, d_out, scale=scale, recurrent_state=state, d_state=d_state
    )
    # The oracle runs in float32 (the forward reference casts to float()); compare with a
    # tolerance that clears that floor without hiding a real derivation error.
    torch.testing.assert_close(analytic.d_query, oracle.d_query.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_key, oracle.d_key.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_value, oracle.d_value.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_g_log, oracle.d_g_log.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_beta, oracle.d_beta.double(), rtol=2e-3, atol=2e-3)
    # Compared on BOTH paths: dL/dS0 is returned even when the forward started from an implicit
    # zero state, so with_state=False checks the gradient of that materialized zero state too.
    torch.testing.assert_close(analytic.d_recurrent_state, oracle.d_recurrent_state.double(), rtol=2e-3, atol=2e-3)


# Large per-token decay. The gated delta-rule's log-decay is g = -exp(A_log)*softplus(a+dt_bias),
# which reaches |g| ~ 10 at init, so exp(cumsum g) UNDERFLOWS to 0 within a 128-chunk. Every other
# test here uses |g| ~ 0.05 (see `_make_inputs`), so none exercises that regime.
# (batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size)
_LARGE_DECAY_SHAPES = [
    (1, 1, 1, 256, 16, 16, 128),
    (1, 2, 4, 256, 16, 16, 128),  # GQA
]


@pytest.mark.parametrize("batch,n_k_heads,n_v_heads,seqlen,key_head_dim,value_head_dim,chunk_size", _LARGE_DECAY_SHAPES)
@pytest.mark.parametrize("decay_scale", [0.05, 5.0, 12.0])
def test_backward_stable_under_large_decay(
    batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, chunk_size, decay_scale
):
    """The analytic backward stays FINITE and matches the oracle even when exp(cumsum g) underflows.

    Under a large per-token decay, G_s = exp(cumsum g) underflows to 0 partway through a chunk. A
    gate gradient written with 1/G_s (and 1/G_s^2) -- cancelled only later by *G_s / *G_last -- then
    evaluates inf*0 = nan, even though the forward and the autograd oracle stay finite. This is the
    exact failure a training step hits (grad_norm = nan). The backward instead routes the ratio
    gradient through the bounded d_ratio*ratio in log-space, and the forward forms
    ratio = exp(log_g_last - log_g) rather than the quotient G_last/G_s. decay_scale=0.05 is the
    regression guard that the stable form still matches at ordinary decay.
    """
    query, key, value, _, beta, d_out, _, _, scale = _make_inputs(
        batch, n_k_heads, n_v_heads, seqlen, key_head_dim, value_head_dim, with_state=False
    )
    rng = np.random.RandomState(123)
    g_log = torch.from_numpy(
        -np.abs(rng.randn(batch, n_v_heads, seqlen).astype(np.float64)) * decay_scale
    )  # per-token |g| up to ~decay_scale

    analytic = gdn_cte_bwd_chunked(query, key, value, g_log, beta, d_out, scale=scale, chunk_size=chunk_size)
    for name in ("d_query", "d_key", "d_value", "d_g_log", "d_beta"):
        grad = getattr(analytic, name)
        assert torch.isfinite(grad).all(), f"{name} non-finite at decay_scale={decay_scale} (gate-gradient underflow)"

    oracle = gdn_cte_bwd_torch_ref(query, key, value, g_log, beta, d_out, scale=scale)
    torch.testing.assert_close(analytic.d_query, oracle.d_query.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_key, oracle.d_key.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_value, oracle.d_value.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_g_log, oracle.d_g_log.double(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(analytic.d_beta, oracle.d_beta.double(), rtol=2e-3, atol=2e-3)
