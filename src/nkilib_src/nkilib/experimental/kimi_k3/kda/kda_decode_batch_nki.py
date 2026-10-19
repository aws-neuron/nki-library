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

"""NKI kernel for BATCHED Kimi Delta Attention (KDA) decode: B requests per call.

Same one-token recurrence as ``kda_decode_step``, but all B requests are served
in a single invocation so the fixed launch cost and the Scalar-bound preprocessing
chain are paid once for the whole batch instead of once per request.

nc_matmul(dst, stationary, moving) = stationary.T @ moving
  - contraction on partition axis (dim 0, <=128)
  - dst in PSUM, inputs in SBUF

Per request b and head h, with a per-channel gate g[dk] (all entries <= 0):

  S    = Diag(exp(g)) @ S           # per-channel decay along dk
  kv   = (k * exp(g)) @ S           # [dv], contract dk
  diff = v - kv                     # beta is folded into the key
  S    = S + outer(k * beta, diff)  # [dk,dv]
  out  = scale * (q @ S)            # [dv]

The recurrence, the engine-placement policy and the dtype policy are inherited
verbatim from ``kda_decode_step``; see that kernel for their derivations. What is
new here is the (b, h) flattening and the packed prologue, specified on
``kda_decode_batch`` below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert

if TYPE_CHECKING:
    from nki.backends.mlir_tracer.tensor import Tensor

# L2-norm epsilon, folded into the rsqrt bias.
_L2_EPS = 1.0e-6

# Partition capacity of one packed prologue tile.
_PROLOGUE_P = 128


def _pack_runs(group):
    """Group a list of (b, h) work items into maximal contiguous HBM runs.

    Items are ordered by the flat index ``b * nv + h``, so a stretch of items
    sharing one ``b`` with consecutive ``h`` is one contiguous row range in every
    ``[B, nv, ...]`` input -- and, because ``a_log``/``dt_bias`` are indexed by
    ``h`` alone, in the ``[nv, ...]`` parameters too. Coalescing them means the
    packed tile is filled with a handful of DMAs rather than one per item.

    Returns:
        list of (b, dst_row, h_start, count)
    """
    runs = []
    for dst_row, (b, h) in enumerate(group):
        if runs:
            prev_b, _, prev_h, prev_n = runs[-1]
            if prev_b == b and prev_h + prev_n == h:
                runs[-1] = (prev_b, runs[-1][1], prev_h, prev_n + 1)
                continue
        runs.append((b, dst_row, h, 1))
    return runs


@nki.jit
def kda_decode_batch(
    q_ref: Tensor,  # [B, nv, dk]      raw query (pre-L2-norm)
    k_ref: Tensor,  # [B, nv, dk]      raw key (pre-L2-norm)
    v_ref: Tensor,  # [B, nv, dv]
    g_ref: Tensor,  # [B, nv, dk]      raw per-channel gate (pre-activation)
    beta_ref: Tensor,  # [B, nv, 1]       beta logits (pre-sigmoid)
    a_log_ref: Tensor,  # [nv, 1]          log-gate parameter (shared across batch)
    dt_bias_ref: Tensor,  # [nv, dk]         gate bias (shared across batch)
    out: Tensor,  # [B, nv, dv]      attention output (written)
    S_in_ref: Tensor,  # [B, nv, dk, dv]  recurrent state, IN/OUT (updated in place)
    scale_ref: Tensor,  # [1, 1]           query scale (1/sqrt(dk))
    lower_bound_ref: Tensor,  # [1, 1]           gate lower bound (in [-5, 0])
):
    """Batched single-token Kimi Delta Attention (KDA) decode: output + new state.

    Summary:
        Advances the KDA recurrence by exactly one token for each of B requests in
        a single invocation. Numerically identical to calling ``kda_decode_step``
        once per request; the difference is purely how the work is packed.

        The (b, h) pairs are FLATTENED into one work list, so the prologue runs on
        packed [min(B*nv, 128), dk] tiles and its Scalar activation chain covers up
        to 128 items per pass rather than nv.

        Intended usage: one call per decode step for the whole in-flight batch.
        Head counts nv <= 128 and dk, dv <= 128 as in the unbatched kernel; B is
        unconstrained (the work list is tiled).

    Performance:
        A functional reference, like the unbatched kernel, but batching addresses
        the two structural costs that dominate single-request decode. Measured on
        trn2 at nv = 12, dk = dv = 128, ``kda_decode_step`` takes 34.0 us of
        wall-clock for 19.3 us of active compute -- ~13 us, 43%, is fixed
        per-launch overhead, near-constant across shapes. Serving B requests one
        call at a time pays it B times; this pays it once. Second, that kernel's
        prologue is 11.2 us of the 19.3 us active time and Scalar-bound at 94%,
        yet occupies only nv of 128 partitions (9% of the machine at nv = 12).

        Decode is memory-bound (arithmetic intensity ~0.8 flop/byte against a
        ridge point of ~214), so MBU is the meaningful metric: this reaches ~20% at
        B = 32 versus ~10% unbatched. State traffic dominates -- 50 MB of fp32
        state per step at B = 32 -- so that is where the remaining headroom is.

    Dimensions:
        B:  Number of requests in the batch (unconstrained)
        nv: Number of value heads (GQA-expanded; each carries its own state), <= 128
        dk: Key/query head dimension, <= 128
        dv: Value head dimension, <= 128

    Args:
        q_ref (Tensor):
            Query tensor in HBM for the current token, raw (pre-L2-norm, pre-scale).
            Shape: [B, nv, dk].  Dtype: nl.float32
        k_ref (Tensor):
            Key tensor in HBM for the current token, raw (pre-L2-norm).
            Shape: [B, nv, dk].  Dtype: nl.float32
        v_ref (Tensor):
            Value tensor in HBM for the current token.
            Shape: [B, nv, dv].  Dtype: nl.float32
        g_ref (Tensor):
            Per-channel forget-gate tensor in HBM, raw (pre-activation; the kernel
            applies lower_bound * sigmoid(exp(a_log) * (g + dt_bias))).
            Shape: [B, nv, dk].  Dtype: nl.float32
        beta_ref (Tensor):
            Delta-rule beta LOGITS in HBM (the kernel applies the sigmoid).
            Shape: [B, nv, 1].  Dtype: nl.float32
        a_log_ref (Tensor):
            Per-head log-gate parameter in HBM; exp(a_log) scales the gate
            pre-activation. A model parameter, so it carries no batch axis and is
            indexed by head alone.
            Shape: [nv, 1].  Dtype: nl.float32
        dt_bias_ref (Tensor):
            Per-head, per-CHANNEL gate bias in HBM, added to g before the sigmoid.
            A model parameter, so it carries no batch axis.
            Shape: [nv, dk].  Dtype: nl.float32
        out (Tensor):
            Caller-allocated destination for this token's attention output.
            Shape: [B, nv, dv].  Dtype: nl.float32
        S_in_ref (Tensor):
            Recurrent state carried in from the previous token, in HBM. IN/OUT:
            each item's [dk, dv] slice is read to seed its step, then overwritten
            with the advanced state (see Returns).
            Shape: [B, nv, dk, dv].  Dtype: nl.float32
        scale_ref (Tensor):
            Query scale, normally 1/sqrt(dk); folded into the L2-normed query so
            the output needs no post-scale.
            Shape: [1, 1].  Dtype: nl.float32
        lower_bound_ref (Tensor):
            Gate lower bound, expected in [-5, 0]. The activated gate is
            lower_bound * sigmoid(...), hence <= 0, so exp(gate) is in (0, 1] and
            the decay can never amplify the state.
            Shape: [1, 1].  Dtype: nl.float32

    Returns:
        out (Tensor):
            The caller-provided output tensor, filled with this token's
            attention output for every request in the batch.
            Shape: [B, nv, dv].  Dtype: nl.float32
        S_in_ref (Tensor):
            The caller-provided state tensor, advanced in place by one token for
            every request. Each item's [dk, dv] slice is read once and written
            once, so no separate copy into the state cache is needed.
            Shape: [B, nv, dk, dv].  Dtype: nl.float32

    Notes:
        WORK FLATTENING AND TILING:
        ---------------------------
        (request, head) pairs are fully independent: each carries its own state
        and writes a disjoint output slice. They are therefore flattened into one
        list of B * nv items ordered by ``b * nv + h``, sharded across LNC cores,
        and then cut into groups of at most 128 -- the partition capacity of one
        prologue tile.

        Flattening rather than nesting two loops is what lets the prologue fill
        its partitions. With B = 32, nv = 12 the flat list is 384 items, so the
        Scalar activation chain runs 3 times on full 128-partition tiles instead
        of 32 times on 12-partition ones.

        Each group is processed as prologue-then-recurrence before the next group
        starts. That ordering bounds the live set: ``knb_rows`` alone is
        [1, group * dk] on a single partition (64 KB at group = dk = 128), so
        keeping every group's copy live at once would overflow partition 0's SBUF.

        HBM RUN COALESCING:
        -------------------
        Items in a group are consecutive in the flat index, so a stretch sharing
        one ``b`` with consecutive ``h`` is one contiguous row range in each
        [B, nv, ...] input. ``_pack_runs`` coalesces those, so filling a packed
        tile costs about ceil(group / nv) DMAs per input rather than one per item.
        The same runs index the [nv, ...] parameters, since ``a_log`` and
        ``dt_bias`` depend only on ``h``.

        TENSOR ORIENTATIONS:
        --------------------
        Identical to ``kda_decode_step`` with ``nv_local`` replaced by the group
        size m: prologue tiles are item-major [m, d] (partition = work item), while
        ``allT`` tiles are channel-major [dk, m] so item j is the free slice
        [:, j:j+1], the [dk, 1] nc_matmul stationary / tensor_scalar operand.
        ``knb_rows`` is [1, m * dk] on partition 0.

        SHARDING:
        ---------
        SPMD over the FLAT item list via ``nl.program_id(axis=0)``, not over heads
        as in the unbatched kernel. Sharding the flat list balances better: with
        nv = 12 on 2 cores, head sharding splits 12 heads 6/6 regardless of batch,
        whereas flat sharding splits B * 12 items evenly and stays balanced even
        when nv is small or odd. The last core absorbs the remainder, so B * nv
        need not divide the core count.

        CONSTRAINTS AND UNSUPPORTED CASES:
        ----------------------------------
          - nv <= 128, dk <= 128, dv <= 128 (partition dimensions).
          - Exactly one token per request per call. There is no multi-token or
            speculative path; use kda_prefill for sequences.
          - Uniform batch only: every request advances by exactly one token. No
            varlen / block-table / paged-state handling, and no per-request
            early-exit for finished sequences. The state is a dense
            [B, nv, dk, dv] tensor, not gathered by indirection from a unified
            paged state tensor, and has not been validated against a serving
            stack's cache layout.
          - a_log and dt_bias are shared across the batch (model parameters).
          - Forward only; upstream is also forward-only.
          - out and S_in_ref are caller-allocated destinations. S_in_ref is
            read and written in place, matching the KV-cache convention used
            elsewhere in this package (see core/attention/attention_tkg.py). The
            next item's state prefetch and the current item's writeback touch
            distinct (request, head) slices, so the in-place update is safe.

    Pseudocode:
        # (request, head) pairs are independent, so flatten them into one list
        # ordered by b * nv + h, shard it across cores, and cut it into groups
        # of at most 128 -- the partition capacity of one prologue tile.
        items = [(b, h) for b in range(B) for h in range(nv)]
        for each group of <= 128 items owned by this core:
            # Packed prologue: one DMA per contiguous HBM run, then the whole
            # Scalar activation chain runs once on a full-partition tile.
            q = l2_norm(q_raw) * scale
            k = l2_norm(k_raw)
            beta = sigmoid(beta_logits)
            exp_g = exp(lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias)))
            k_decayed = k * exp_g
            k_beta    = k * beta

            # Same one-token recurrence as kda_decode_step, per item.
            for j, (b, h) in enumerate(group):
                S         = S_in_ref[b, h]                  # [dk, dv]
                kv        = k_decayed[j] @ S                 # [dv]
                diff      = v[j] - kv
                S         = S * exp_g[j] + outer(k_beta[j], diff)
                out[b, h] = q[j] @ S
                S_in_ref[b, h] = S                           # in place
    """
    B, nv, dk = q_ref.shape
    dv = v_ref.shape[-1]

    # All preprocessing is batched on the partition axis, so each per-item vector
    # must fit one partition tile.
    kernel_assert(nv <= 128, f"nv must be <= 128, got {nv}")
    kernel_assert(dk <= 128, f"dk must be <= 128, got {dk}")
    kernel_assert(dv <= 128, f"dv must be <= 128, got {dv}")
    kernel_assert(B >= 1, f"B must be at least 1, got {B}")

    # SPMD sharding over the FLAT (request, head) work list.
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        n_progs = nl.num_programs(axes=0)
        prog_id = nl.program_id(axis=0)
    else:
        n_progs = 1
        prog_id = 0

    n_items = B * nv
    items_per_core = n_items // n_progs
    i_start = prog_id * items_per_core
    i_end = n_items if prog_id == n_progs - 1 else i_start + items_per_core

    # (b, h) for every item this core owns; resolved at trace time.
    my_items = [(i // nv, i % nv) for i in range(i_start, i_end)]

    scale_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_t, src=scale_ref[0:1, 0:1])
    lower_bound_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lower_bound_t, src=lower_bound_ref[0:1, 0:1])

    # Groups of at most _PROLOGUE_P items. Prologue and recurrence are emitted
    # per group so the group's live tiles can be reused by the next one.
    for g_start in range(0, len(my_items), _PROLOGUE_P):
        group = my_items[g_start : g_start + _PROLOGUE_P]
        m = len(group)
        runs = _pack_runs(group)

        # tensor_scalar requires operand0's partition count to match dst's, so the
        # [1,1] scalars must be explicitly partition-broadcast to the group tiles.
        scale_bc = nl.broadcast_to(scale_t, (m, 1))
        lower_bound_bc = nl.broadcast_to(lower_bound_t, (m, 1))

        # Packed loads: one DMA per contiguous run, not per item.
        q_in = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        k_in = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        g_in = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        dt_bias_in = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        a_log_in = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        beta_in = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        for b, r0, h0, cnt in runs:
            nisa.dma_copy(dst=q_in[r0 : r0 + cnt, 0:dk], src=q_ref[b, h0 : h0 + cnt, 0:dk])
            nisa.dma_copy(dst=k_in[r0 : r0 + cnt, 0:dk], src=k_ref[b, h0 : h0 + cnt, 0:dk])
            nisa.dma_copy(dst=g_in[r0 : r0 + cnt, 0:dk], src=g_ref[b, h0 : h0 + cnt, 0:dk])
            nisa.dma_copy(dst=beta_in[r0 : r0 + cnt, 0:1], src=beta_ref[b, h0 : h0 + cnt, 0:1])
            # a_log / dt_bias are model parameters: indexed by head only.
            nisa.dma_copy(dst=dt_bias_in[r0 : r0 + cnt, 0:dk], src=dt_bias_ref[h0 : h0 + cnt, 0:dk])
            nisa.dma_copy(dst=a_log_in[r0 : r0 + cnt, 0:1], src=a_log_ref[h0 : h0 + cnt, 0:1])

        # Fused L2-norm: square+sum in one Scalar pass, then rsqrt(ss + eps).
        q_ss = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        q_sq = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=q_sq, op=nl.square, data=q_in, reduce_op=nl.add, reduce_res=q_ss)
        q_rnorm = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=q_rnorm, op=nl.rsqrt, data=q_ss, bias=_L2_EPS)
        k_ss = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        k_sq = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=k_sq, op=nl.square, data=k_in, reduce_op=nl.add, reduce_res=k_ss)
        k_rnorm = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=k_rnorm, op=nl.rsqrt, data=k_ss, bias=_L2_EPS)

        # q is pre-scaled by `scale` so the output needs no scaling. The prologue is
        # Scalar-bound while Vector is nearly idle, so every tensor_scalar here is
        # pinned to Vector to run alongside activations it does not depend on.
        q_rnorm_scaled = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_rnorm_scaled, data=q_rnorm, op0=nl.multiply, operand0=scale_bc, engine=nisa.vector_engine
        )
        q_n = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=q_n, data=q_in, op0=nl.multiply, operand0=q_rnorm_scaled, engine=nisa.vector_engine)
        k_n = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=k_n, data=k_in, op0=nl.multiply, operand0=k_rnorm, engine=nisa.vector_engine)

        # g_act = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias)), batched.
        a_exp = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=a_exp, op=nl.exp, data=a_log_in)
        g_shift = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g_shift, data1=g_in, data2=dt_bias_in, op=nl.add)
        g_scaled = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g_scaled, data=g_shift, op0=nl.multiply, operand0=a_exp, engine=nisa.vector_engine)
        g_sig = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sig, op=nl.sigmoid, data=g_scaled)
        # g = lower_bound * sigmoid(...) <= 0, so exp(g) is in (0, 1] -- no overflow.
        g_act = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g_act, data=g_sig, op0=nl.multiply, operand0=lower_bound_bc, engine=nisa.vector_engine)
        exp_g_all = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g_all, op=nl.exp, data=g_act)

        beta_all = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=beta_all, op=nl.sigmoid, data=beta_in)
        # beta folds into the key feeding the rank-1 update, by bilinearity of the
        # outer product: outer(k, beta*(v - kv_mem)) == outer(k*beta, v - kv_mem).
        # The UNscaled k_n is still needed for kv_mem, which must not carry beta.
        k_nb = nl.ndarray((m, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=k_nb, data=k_n, op0=nl.multiply, operand0=beta_all, engine=nisa.vector_engine)

        # Transpose to [dk, m] so each item is a [dk,1] free slice.
        q_allT_p = nl.ndarray((dk, m), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_allT_p, data=q_n, engine=nisa.tensor_engine)
        q_allT = nl.ndarray((dk, m), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_allT, src=q_allT_p)
        k_allT_p = nl.ndarray((dk, m), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_allT_p, data=k_n, engine=nisa.tensor_engine)
        k_allT = nl.ndarray((dk, m), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_allT, src=k_allT_p)
        # The rank-1 update needs k*beta as a [1,dk] partition-0 ROW. Gather every
        # item's row onto partition 0 once with SBUF->SBUF DMAs (DMA may cross
        # partitions, compute engines may not), replacing a per-item PE transpose
        # plus PSUM drain with prologue DMA on otherwise-idle engines.
        knb_rows = nl.ndarray((1, m * dk), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(m):
            nisa.dma_copy(dst=knb_rows[0:1, j * dk : (j + 1) * dk], src=k_nb[j : j + 1, 0:dk])
        # exp_g as [dk, m] so eg_allT[:, j:j+1] is the [dk,1] channel vector.
        eg_allT_p = nl.ndarray((dk, m), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=eg_allT_p, data=exp_g_all, engine=nisa.tensor_engine)
        eg_allT = nl.ndarray((dk, m), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eg_allT, src=eg_allT_p)

        # Double-buffer the state DMA: items are independent, so prefetch item
        # j+1's state before computing j.
        S_bufs = [
            nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf),
            nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf),
        ]
        b0, h0 = group[0]
        nisa.dma_copy(dst=S_bufs[0], src=S_in_ref[b0, h0, 0:dk, 0:dv])

        for j in range(m):
            b, h = group[j]

            S_h = S_bufs[j % 2]
            if j + 1 < m:
                nb, nh = group[j + 1]
                nisa.dma_copy(dst=S_bufs[(j + 1) % 2], src=S_in_ref[nb, nh, 0:dk, 0:dv])

            # Prologue tiles are GROUP-relative, so they are indexed by j; only the
            # HBM DMAs use the absolute (b, h).
            q_h = q_allT[0:dk, j : j + 1]
            k_h = k_allT[0:dk, j : j + 1]
            exp_g = eg_allT[0:dk, j : j + 1]  # [dk,1] per-channel decay
            k_row = knb_rows[0:1, j * dk : (j + 1) * dk]  # [1,dk] k*beta

            v_h = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_h, src=v_ref[b, h : h + 1, 0:dv])

            # kv_mem = k @ (Diag(exp_g) @ S) = (k * exp_g) @ S. Folding the decay
            # into the [dk,1] key avoids a full [dk,dv] pass over the state, and
            # unlike GDN's scalar deferred-decay it stays correct for a diagonal
            # gate.
            k_dec = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=k_dec, data1=k_h, data2=exp_g, op=nl.multiply)

            kv_mem_p = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_mem_p, stationary=k_dec, moving=S_h)
            kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            # All PSUM->SBUF drains in this loop are pinned to Scalar: the
            # Vector-class steps between them occupy Vector, and Scalar would idle.
            nisa.tensor_copy(dst=kv_mem, src=kv_mem_p, engine=nisa.scalar_engine)

            # diff = v_h - kv_mem; beta is carried by k_row.
            diff = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff, data1=v_h, data2=kv_mem, op=nl.subtract)

            # S_h = Diag(exp_g) @ S_h + outer(k_h * beta, diff). The outer product
            # runs on the PE as a contraction-1 nc_matmul; the decay multiply and
            # additive update then fuse into one Vector pass, with the [dk,1]
            # per-channel exp_g as operand0 (no broadcast needed).
            outer_p = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_p, stationary=k_row, moving=diff)
            # The drain is not required by the operand rules (scalar_tensor_tensor
            # could read outer_p as operand1); it is kept for scheduling, as in the
            # unbatched kernel -- holding outer_p live until the Vector update
            # retires would extend the PSUM bank's lifetime into the next item's
            # matmul.
            outer_sbuf = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=outer_sbuf, src=outer_p, engine=nisa.scalar_engine)
            # A fresh tile, not in-place over S_h: S_h is a ping-pong buffer that
            # item j+2's prefetch reclaims, so an in-place update would make that
            # prefetch wait on this item's state DMA-out.
            S_updated = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=S_updated,
                data=S_h,
                op0=nl.multiply,
                operand0=exp_g,
                op1=nl.add,
                operand1=outer_sbuf,
            )
            S_h = S_updated

            # out_h = q_h.T @ S_h; q is already pre-scaled.
            out_h_p = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=out_h_p, stationary=q_h, moving=S_h)
            out_h = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_h, src=out_h_p, engine=nisa.scalar_engine)

            nisa.dma_copy(dst=out[b, h : h + 1, 0:dv], src=out_h)
            nisa.dma_copy(dst=S_in_ref[b, h, 0:dk, 0:dv], src=S_h)

    return out, S_in_ref
