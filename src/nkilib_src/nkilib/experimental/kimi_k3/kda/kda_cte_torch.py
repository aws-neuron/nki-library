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
"""Torch references for ``kda_cte`` --- chunked delta rule with PER-CHANNEL decay.

Three functions, and the distinction between the first two is the whole point of the
file:

* :func:`kda_recurrent_ref` is the **sequential** recurrence, transcribed from
  ``fla.ops.kda.naive.naive_recurrent_kda``. It is the definition of the operator.
* :func:`kda_cte_chunked_ref` is the **chunked, anchored** form that the NKI kernel
  implements, expressed in torch. It exists so the kernel can be validated against
  something that shares its algebra, and so the algebra itself can be validated
  against the sequential definition first.
* :func:`kda_cte_torch_ref` is the convention reference for the ``kda_cte`` kernel:
  same argument names and ``[B, S, H, D]`` layouts as the kernel, delegating to
  :func:`kda_cte_chunked_ref` per (batch, head) at the kernel's chunk sizes.

Chaining the first two comparisons is what makes a kernel bug distinguishable from an
algebra bug.

Why per-channel decay is not a free generalisation of GDN
---------------------------------------------------------
``nkilib.experimental.gdn.gdn_cte`` takes ``gate`` as ``[B, S]`` --- one log-decay
scalar per token. The chunked delta rule then needs the pairwise decay

    decay[i, j] = exp(cg[i] - cg[j]),   cg = cumsum(gate)

as an explicit ``[CHUNK, CHUNK]`` tile, which it multiplies into ``k k^T``.

KDA's decay is **per channel**: ``g[t, d]``, so the pairwise factor is
``exp(cg[i, d] - cg[j, d])`` --- a ``[CHUNK, CHUNK, D]`` object that cannot be
materialised, and cannot be applied after a contraction over ``d``. It has to be
folded into the operands *before* the matmul::

    A[i, j] = -beta[i] * sum_d k[i, d] * exp(cg[i, d] - cg[j, d]) * k[j, d]
            = -beta[i] * sum_d khat[i, d] * kbar[j, d]

with ``khat[i, d] = k[i, d] * exp(cg[i, d])`` and ``kbar[j, d] = k[j, d] * exp(-cg[j, d])``.
That is exact, and it *overflows*: ``cg`` is a cumulative sum of values in
``(lower_bound, 0)``, so over a 64-token chunk with ``lower_bound = -5`` the
``exp(-cg)`` in ``kbar`` reaches ``exp(320)`` --- infinity in fp32 and bf16 alike.
No float format helps; ``exp(+320)`` is simply not representable.

Sub-block anchoring
-------------------
The factorisation is safe whenever the anchor sits *between* the two tokens, because
then both exponents are non-positive... in one direction. Concretely, anchor the pair
``(i, j)`` at ``a = ``the token just before ``j``'s sub-block:

    khat_a[i, d] = k[i, d] * exp(cg[i, d] - anchor[a, d])     exponent <= 0
    kbar[j, d]   = k[j, d] * exp(anchor[a, d] - cg[j, d])     exponent in [0, SUBBLK * |lb|]

and their product is exactly ``k[i,d] k[j,d] exp(cg[i,d] - cg[j,d])`` --- the anchor
cancels. ``kbar``'s exponent is bounded by ``SUBBLK * |lower_bound|`` because ``j`` is
at most ``SUBBLK`` tokens past the anchor, which is what makes the choice of
``SUBBLK`` a *numerical* constraint and not just a tiling one. At ``SUBBLK = 16`` and
``lower_bound = -5`` the bound is ``exp(80) = 5.5e34``, three orders of magnitude
below the bf16 maximum. :func:`assert_anchor_bound_safe` is the check; it is a
function of the *config*, never of the data, so it can and does fail closed.

``khat_a``'s exponent is unbounded below, but that direction underflows to zero
rather than overflowing, and it underflows exactly where the true pairwise decay is
negligible: ``cg[i] - cg[j] >= cg[i] - anchor[a]``, so a factor lost at ``exp(-100)``
can only have hidden a true contribution of at most ``exp(-100 + 80) = 2e-9``, which
is far below the bf16 precision of the terms that survive.

Only ``A`` and the intra-chunk ``q k^T`` need anchoring. Everything else in the chunk
already has a non-positive exponent and needs no factorisation at all:
``qhat = q * exp(cg)``, ``k_dec = k * exp(cg[-1] - cg)`` and the state decay
``exp(cg[-1])``. Note this makes the per-channel kernel *simpler* than the scalar one
in those places --- there is no ``[CHUNK, CHUNK]`` decay tile to build, transpose and
multiply; the decay rides along inside the operands.
"""

from typing import Optional

import torch

# Chunk of tokens carried by one state update, and the sub-block that bounds the
# anchored exponent. SUBBLK is a numerical parameter, not just a tiling one --- see
# the module docstring and assert_anchor_bound_safe.
CHUNK = 64
SUBBLK = 16

# exp(85) = 8.2e36, still inside bf16's 3.4e38. Above this the anchored kbar factor
# stops being representable and the kernel would return inf/NaN rather than a wrong
# number, so the bound is checked from the config rather than trusted.
MAX_SAFE_EXP_ARG = 85.0


def assert_anchor_bound_safe(lower_bound: float, subblk: int = SUBBLK, max_exp_arg: float = MAX_SAFE_EXP_ARG) -> None:
    """Fail closed unless sub-block anchoring cannot overflow for this gate bound.

    ``kbar``'s exponent is at most ``subblk * |lower_bound|`` (a token is at most
    ``subblk`` steps past its anchor, and each step contributes at most
    ``|lower_bound|``). This is a property of the model config, so it is checkable
    ahead of time --- unlike the data-dependent magnitudes it bounds.

    Args:
        lower_bound: KDA's ``gate_lower_bound`` (Kimi-K3 uses ``-5.0``). ``None`` is
            rejected: the unbounded GDN/Mamba gate branch has no bound to check, so
            the anchored factorisation cannot be shown safe for it.
        subblk: anchor spacing in tokens.
        max_exp_arg: largest exponent considered representable.

    Example:
        >>> assert_anchor_bound_safe(-5.0, 16)          # Kimi-K3: 80 <= 85
        >>> try:
        ...     assert_anchor_bound_safe(-6.0, 16)
        ... except ValueError as exc:
        ...     print(str(exc).split(".")[0])
        sub-block anchoring needs subblk * |lower_bound| <= 85
        >>> try:
        ...     assert_anchor_bound_safe(None)
        ... except ValueError as exc:
        ...     print(str(exc)[:48])
        sub-block anchoring requires a finite gate lower
    """
    if lower_bound is None:
        raise ValueError(
            "sub-block anchoring requires a finite gate lower_bound; the unbounded "
            "gate branch (lower_bound=None) has no bound on cumulative decay, so "
            "exp(anchor - cg) cannot be shown representable"
        )
    span = subblk * abs(float(lower_bound))
    if span > max_exp_arg:
        raise ValueError(
            f"sub-block anchoring needs subblk * |lower_bound| <= {max_exp_arg}, got "
            f"{subblk} * {abs(float(lower_bound))} = {span}. Halve SUBBLK (to "
            f"{subblk // 2}) or raise the gate bound; leaving it would make the "
            f"anchored kbar factor exp({span}) overflow bf16 and return inf."
        )


def kda_recurrent_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential KDA recurrence --- the definition. ``[BH, S, D]`` in, per head.

    Transcribed from ``fla.ops.kda.naive.naive_recurrent_kda``. Inputs are already
    post-L2-norm (``q``, ``k``) and post-sigmoid (``beta``); ``g`` is the log-decay.

    Args:
        q, k, v: ``[BH, S, D]``. ``BH`` is batch times heads, flattened.
        beta: ``[BH, S]`` update strength.
        g: ``[BH, S, D]`` per-channel log-decay, non-positive.
        scale: query scale, normally ``D ** -0.5``.
        initial_state: ``[BH, D, D]`` fp32, or None for a fresh sequence.

    Returns:
        ``(out, final_state)`` with ``out: [BH, S, D]`` and state ``[BH, D, D]`` fp32.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> q, k, v = (torch.randn(2, 4, 8) for _ in range(3))
        >>> g = -torch.rand(2, 4, 8)
        >>> beta = torch.rand(2, 4)
        >>> o, s = kda_recurrent_ref(q, k, v, beta, g)
        >>> o.shape, s.shape
        (torch.Size([2, 4, 8]), torch.Size([2, 8, 8]))
    """
    BH, S, D = q.shape
    qf = q.float() * scale
    kf, vf, gf, bf = k.float(), v.float(), g.float(), beta.float()

    state = qf.new_zeros(BH, D, vf.shape[-1])
    if initial_state is not None:
        state = state + initial_state.float()

    out = torch.zeros_like(vf)
    for t in range(S):
        state = state * gf[:, t][..., None].exp()
        pred = (kf[:, t][..., None] * state).sum(-2)
        state = state + torch.einsum("bk,bv->bkv", bf[:, t][..., None] * kf[:, t], vf[:, t] - pred)
        out[:, t] = torch.einsum("bk,bkv->bv", qf[:, t], state)
    return out.to(v.dtype), state


def _anchored_operands(
    kc: torch.Tensor, qc: torch.Tensor, cg: torch.Tensor, subblk: int
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Build ``kbar`` (one anchor per row) and ``khat_a`` / ``qhat_a`` per anchor.

    ``anchor[a]`` is the cumulative decay just *before* sub-block ``a`` starts, i.e.
    ``cg[a * subblk - 1]``, with ``anchor[0] = 0``. Returns ``kbar`` plus lists indexed
    by anchor.
    """
    chunk = cg.shape[1]
    n_sb = chunk // subblk
    zero = torch.zeros_like(cg[:, 0])
    anchors = [zero] + [cg[:, a * subblk - 1] for a in range(1, n_sb)]

    kbar = torch.empty_like(kc)
    for a in range(n_sb):
        rows = slice(a * subblk, (a + 1) * subblk)
        kbar[:, rows] = kc[:, rows] * (anchors[a][:, None] - cg[:, rows]).exp()

    # ⚠️ Clamp the anchored exponent at 0 before exponentiating. Rows *before*
    # anchor `a` have cg[i] > anchor[a], so the raw exponent is POSITIVE and reaches
    # +chunk*|lower_bound| (+320 for Kimi-K3) -- inf. Those entries are then masked
    # away, and `inf * 0` is NaN, which is how a correct-looking algebra produced a
    # NaN output while the state stayed exact.
    #
    # The clamp is provably free: an entry it changes has row i < a*subblk while its
    # column j is in sub-block a, so j > i and the entry lies in the strictly-upper
    # triangle that both masks discard. Every *used* entry has i >= a*subblk, hence
    # cg[i] <= anchor[a] and a non-positive exponent the clamp leaves alone.
    def anchored(x, a):
        return x * (cg - anchors[a][:, None]).clamp_max(0.0).exp()

    khat = [anchored(kc, a) for a in range(n_sb)]
    qhat = [anchored(qc, a) for a in range(n_sb)]
    return kbar, khat, qhat


def kda_cte_chunked_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    chunk: int = CHUNK,
    subblk: int = SUBBLK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked, sub-block-anchored KDA --- the algebra the NKI kernel implements.

    Same signature and semantics as :func:`kda_recurrent_ref`; ``S`` must be a
    multiple of ``chunk``. Everything is fp32 here: this reference isolates the
    *algebra*, so any disagreement with :func:`kda_recurrent_ref` is a derivation
    error, and disagreement between this and the kernel is an implementation or
    precision error.

    ⚠️ ``q`` and ``k`` must already be **L2-normalised per token** (KDA does that
    in-kernel under ``use_qk_l2norm_in_kernel=True``). That is not cosmetic: it is
    what bounds ``|k_i . k_j| <= 1`` and so keeps ``(I - A)`` well conditioned. With
    unnormalised keys the strictly-lower ``A`` has entries above 1 and its ``chunk``
    x ``chunk`` triangular inverse grows like ``|k.k| ** (chunk - 1)``, which
    overflows at ``chunk = 64`` -- a property of the operator, not of the chunking,
    and it will look exactly like a chunking bug.

    Example:
        Chunked and sequential agree to fp32 round-off, which is the claim the
        whole file exists to support:

        >>> _ = torch.manual_seed(0)
        >>> BH, S, D = 2, 128, 16
        >>> def l2(x):
        ...     return x / x.norm(dim=-1, keepdim=True)
        >>> q, k = l2(torch.randn(BH, S, D)), l2(torch.randn(BH, S, D))
        >>> v = torch.randn(BH, S, D)
        >>> g = -5.0 * torch.rand(BH, S, D)
        >>> beta = torch.rand(BH, S)
        >>> a, sa = kda_recurrent_ref(q, k, v, beta, g, scale=D ** -0.5)
        >>> b, sb = kda_cte_chunked_ref(q, k, v, beta, g, scale=D ** -0.5, chunk=64)
        >>> bool((a - b).abs().max() < 1e-4), bool((sa - sb).abs().max() < 1e-4)
        (True, True)

        A zero gate reduces it to the plain delta rule, and the sub-block anchors
        all collapse to zero -- so this case passes even if anchoring is wrong, and
        is therefore *not* a test of it:

        >>> c, _ = kda_cte_chunked_ref(q, k, v, beta, torch.zeros_like(g),
        ...                          scale=D ** -0.5, chunk=64)
        >>> d, _ = kda_recurrent_ref(q, k, v, beta, torch.zeros_like(g),
        ...                          scale=D ** -0.5)
        >>> bool((c - d).abs().max() < 1e-4)
        True
    """
    BH, S, D = q.shape
    if S % chunk != 0:
        raise ValueError(f"S={S} must be a multiple of chunk={chunk}")
    if chunk % subblk != 0:
        raise ValueError(f"chunk={chunk} must be a multiple of subblk={subblk}")
    n_sb = chunk // subblk

    qf, kf, vf, gf, bf = (t.float() for t in (q, k, v, g, beta))
    state = qf.new_zeros(BH, D, vf.shape[-1])
    if initial_state is not None:
        state = state + initial_state.float()

    idx = torch.arange(chunk, device=q.device)
    strict_lower = (idx[:, None] > idx[None, :]).float()
    lower_incl = (idx[:, None] >= idx[None, :]).float()
    eye = torch.eye(chunk, device=q.device).expand(BH, chunk, chunk)

    out = torch.zeros_like(vf)
    for c0 in range(0, S, chunk):
        sl = slice(c0, c0 + chunk)
        qc, kc, vc, gc, bc = qf[:, sl], kf[:, sl], vf[:, sl], gf[:, sl], bf[:, sl]
        cg = gc.cumsum(1)
        kbar, khat, qhat = _anchored_operands(kc, qc, cg, subblk)

        # A and the intra-chunk score matrix: column block `a` contracts against the
        # operands anchored at `a`, which is what keeps every exponent representable.
        amat = torch.zeros(BH, chunk, chunk, dtype=qf.dtype, device=q.device)
        qkt = torch.zeros_like(amat)
        for a in range(n_sb):
            cols = slice(a * subblk, (a + 1) * subblk)
            kbar_a_t = kbar[:, cols].transpose(-1, -2)
            amat[:, :, cols] = -bc[..., None] * (khat[a] @ kbar_a_t)
            qkt[:, :, cols] = qhat[a] @ kbar_a_t
        amat = amat * strict_lower
        qkt = qkt * lower_incl

        # (I - A) is unit-lower-triangular, hence exactly invertible. The kernel gets
        # the same operator by recursive doubling on the nilpotent strict-lower part.
        minv = torch.linalg.solve_triangular(eye - amat, eye, upper=False)
        w = minv @ (bc[..., None] * khat[0])
        u = minv @ (bc[..., None] * vc)

        v_new = u - w @ state
        out[:, sl] = (qhat[0] @ state + qkt @ v_new) * scale

        dec_last = cg[:, -1].exp()
        k_dec = kc * (cg[:, -1:] - cg).exp()
        state = state * dec_last[..., None] + k_dec.transpose(-1, -2) @ v_new

    return out.to(v.dtype), state


def kda_cte_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state_in: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the ``kda_cte`` kernel, in the kernel's own layouts.

    Args mirror ``kda_cte``: ``q``/``k``/``v``/``g`` are ``[B, S, H, D]``, ``beta`` is
    ``[B, S, H]`` and ``state_in`` is ``[B, H, D, D]``. Heads are independent, so each
    (batch, head) pair is one ``[S, D]`` sequence for :func:`kda_cte_chunked_ref`, run at
    the kernel's ``CHUNK``/``SUBBLK``. Returns ``(out, state_out)`` with ``out`` in
    ``q.dtype`` shaped ``[B, S, H, D]`` and ``state_out`` fp32 shaped ``[B, H, D, D]``.
    """
    B, S, H, D = q.shape
    # [B, S, H, D] -> [B*H, S, D]; beta [B, S, H] -> [B*H, S].
    to_bh = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, S, -1)  # noqa: E731
    out_bh, state_bh = kda_cte_chunked_ref(
        to_bh(q),
        to_bh(k),
        to_bh(v),
        beta.permute(0, 2, 1).reshape(B * H, S),
        to_bh(g),
        scale=scale,
        initial_state=state_in.reshape(B * H, D, D),
        chunk=CHUNK,
        subblk=SUBBLK,
    )
    out = out_bh.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()
    return out.to(q.dtype), state_bh.reshape(B, H, D, D)
