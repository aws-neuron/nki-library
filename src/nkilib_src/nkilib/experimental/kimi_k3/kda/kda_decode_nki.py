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

"""NKI kernel for the Kimi Delta Attention (KDA) DECODE path (single-token).

KDA is GatedDeltaNet with a **per-channel** (diagonal) forget gate instead of a
per-head scalar gate. Ported from FlashKDA
(https://github.com/MoonshotAI/FlashKDA), whose CUDA kernel is forward-only.

nc_matmul(dst, stationary, moving) = stationary.T @ moving
  - contraction on partition axis (dim 0, <=128)
  - dst in PSUM, inputs in SBUF

Per head h, with a per-channel gate g[dk] (all entries <= 0):

  S_h    = Diag(exp(g_h)) @ S_h     # per-channel decay along dk
  kv_mem = k_h @ S_h                # [dv], contract dk
  delta  = (v_h - kv_mem) * beta_h  # [dv]
  S_h    = S_h + outer(k_h, delta)  # [dk,dv]
  out_h  = scale * (q_h @ S_h)      # [dv]

Status: functional reference. Not performance-competitive -- see the performance
note on ``kda_decode_step``.

Structure follows the sibling gdn_decode_step kernel (batched q/k transpose,
ping-pong state DMA, fused outer+decay via scalar_tensor_tensor, in-place state
buffer), with one substantive difference forced by the per-channel gate: GDN's
"deferred decay" -- compute kv on the UNdecayed state, then scale the tiny [1,dv]
result by the scalar exp_g -- is INVALID here. A diagonal gate does not commute
out of the contraction, since k @ (Diag(e) @ S) = sum_d k[d]*e[d]*S[d,:] weights
each contracted channel differently. The decay instead folds into the [dk,1] key
vector, k_decayed = k * exp_g, which is just as cheap.

Gate activation, QK L2-norm and beta sigmoid are fused in (upstream-style API),
so the kernel consumes raw pre-activation inputs.
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


@nki.jit
def kda_decode_step(
    q_ref: Tensor,  # [nv, dk]      raw query (pre-L2-norm)
    k_ref: Tensor,  # [nv, dk]      raw key (pre-L2-norm)
    v_ref: Tensor,  # [nv, dv]
    g_ref: Tensor,  # [nv, dk]      raw per-channel gate (pre-activation)
    beta_ref: Tensor,  # [nv, 1]       beta logits (pre-sigmoid)
    a_log_ref: Tensor,  # [nv, 1]       log-gate parameter
    dt_bias_ref: Tensor,  # [nv, dk]      gate bias
    out: Tensor,  # [nv, dv]      attention output (written)
    S_in_ref: Tensor,  # [nv, dk, dv]  recurrent state, IN/OUT (updated in place)
    scale_ref: Tensor,  # [1, 1]        query scale (1/sqrt(dk))
    lower_bound_ref: Tensor,  # [1, 1]        gate lower bound (in [-5, 0])
):
    """Single-token Kimi Delta Attention (KDA) decode step: output + new state.

    Summary:
        Advances the Kimi Delta Attention recurrence by exactly ONE token -- the
        token-generation (TKG) counterpart to ``kda_prefill``. KDA is a
        GatedDeltaNet variant whose forget gate is per-CHANNEL (diagonal) rather
        than a per-head scalar. Gate activation, query/key L2-norm, beta sigmoid
        and the query scale are all fused in, so the kernel consumes raw
        pre-activation model tensors and its output needs no post-scaling.

        Intended usage: one token per call, driven in a loop by the decode
        runtime. Head counts nv <= 128 (the whole prologue batches heads on the
        partition axis); nv = 12 (tp8) and nv = 96 (upstream production) are the
        exercised shapes. Head dimensions dk == dv == 128 as upstream KDA uses,
        with dk, dv <= 128 supported. There is no batch dimension: batching is the
        caller's concern (see ``kda_decode_batch`` for a batched variant).

        Sharding: SPMD over the nv heads across LNC cores via
        ``nl.program_id(axis=0)``; heads are independent, so no cross-core
        communication is needed. The prologue is batched over heads, so it is
        sized to the SHARD (nv_local), not to nv.

        Quantization: none. All inputs, outputs, the recurrent state and every
        matmul operand are fp32. bf16 operands were measured and rejected -- see
        the dtype note below.

    Performance:
        This is a FUNCTIONAL reference, not a performance-competitive kernel.
        Measured on trn2 under LNC2 at nv = 12, dk = dv = 128: ~19.4 us.

        Decode is MEMORY-bound, not FLOP-bound: at one token it moves the whole
        fp32 [dk, dv] state per head for O(dk*dv) flops, an arithmetic intensity
        of ~0.8 flop/byte against a ridge point of ~214, which caps MFU near 0.3%
        no matter what. MBU is therefore the only meaningful metric here, and the
        headroom is in state traffic, not arithmetic. This shape reaches ~10% MBU;
        the batched variant reaches ~20%.

    Dimensions:
        nv: Number of value heads (GQA-expanded; each carries its own state),
            <= 128
        dk: Key/query head dimension, <= 128
        dv: Value head dimension, <= 128

    Args:
        q_ref (Tensor):
            Query tensor in HBM for the current token, raw (pre-L2-norm,
            pre-scale).
            Shape: [nv, dk].  Dtype: nl.float32
        k_ref (Tensor):
            Key tensor in HBM for the current token, raw (pre-L2-norm).
            Shape: [nv, dk].  Dtype: nl.float32
        v_ref (Tensor):
            Value tensor in HBM for the current token.
            Shape: [nv, dv].  Dtype: nl.float32
        g_ref (Tensor):
            Per-channel forget-gate tensor in HBM, raw (pre-activation; the
            kernel applies lower_bound * sigmoid(exp(a_log) * (g + dt_bias))).
            Shape: [nv, dk].  Dtype: nl.float32
        beta_ref (Tensor):
            Delta-rule beta LOGITS in HBM (the kernel applies the sigmoid).
            Shape: [nv, 1].  Dtype: nl.float32
        a_log_ref (Tensor):
            Per-head log-gate parameter in HBM; exp(a_log) scales the gate
            pre-activation.
            Shape: [nv, 1].  Dtype: nl.float32
        dt_bias_ref (Tensor):
            Per-head, per-CHANNEL gate bias in HBM, added to g before the
            sigmoid.
            Shape: [nv, dk].  Dtype: nl.float32
        out (Tensor):
            Caller-allocated destination for this token's attention output.
            Shape: [nv, dv].  Dtype: nl.float32
        S_in_ref (Tensor):
            Recurrent state carried in from the previous token, in HBM. IN/OUT:
            read to seed the step, then overwritten with the advanced state (see
            Returns).
            Shape: [nv, dk, dv].  Dtype: nl.float32
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
            attention output.
            Shape: [nv, dv].  Dtype: nl.float32
        S_in_ref (Tensor):
            The caller-provided state tensor, advanced in place by one token.
            Each head's [dk, dv] slice is read once and written once, so no
            separate copy into the state cache is needed.
            Shape: [nv, dk, dv].  Dtype: nl.float32

    Notes:
        TENSOR ORIENTATIONS:
        --------------------
        HBM inputs are head-major (heads on axis 0), which is what the batched
        prologue wants: a [nv_local, d] DMA load puts heads on the PARTITION axis
        with channels on the free axis, so one instruction preprocesses every
        head. The L2-norm's sum over dk is a free-axis reduction, which is why the
        norm is computed in this layout and not after the transpose.

        The per-head recurrence needs the opposite orientation. Tiles suffixed
        ``allT`` are channel-major [dk, nv_local], so head h is the free slice
        [:, h_local : h_local+1] -- a [dk, 1] column, exactly the nc_matmul
        stationary operand and the tensor_scalar [P, 1] operand.

        ``knb_rows`` is [1, nv_local * dk], everything on partition 0: the rank-1
        update needs k*beta as a [1, dk] partition-0 ROW. Gathering all heads once
        with SBUF->SBUF DMAs (DMA may cross partitions, compute engines may not)
        replaces a per-head PE transpose plus PSUM drain with prologue DMA on
        otherwise-idle engines.

        ENGINE PLACEMENT IS PER-PHASE:
        ------------------------------
        Decode runs as two strictly sequential phases with opposite engine
        profiles, measured at the tp8 shape:

            Phase                     Span       Tensor   Vector   Scalar
            Prologue (preprocessing)  11,228 ns    5.3%    12.1%    94.2%
            Per-head recurrence loop   9,240 ns   90.5%    43.4%    33.7%

        So the engine= annotations deliberately point in opposite directions. In
        the prologue Scalar is saturated by the activation chain while Vector is
        ~88% idle, so every tensor_scalar is pinned to Vector; left unannotated the
        compiler put all six on Scalar, adding 2.7 us to the critical path. In the
        loop PE is the bottleneck and Scalar idles, so the PSUM->SBUF drains are
        pinned to Scalar.

        The prologue's four ACT_TABLE_LOAD swaps (46% of its Scalar time) are
        irreducible: the gate's algebra forces the chain exp -> rsqrt -> sigmoid
        -> exp, since exp_g = exp(lower_bound * sigmoid(...)) depends on the
        sigmoid that depends on the exp.

        BETA IS FOLDED INTO THE KEY:
        ---------------------------
        By bilinearity of the outer product,

            outer(k, beta * (v - kv_mem)) == outer(k * beta, v - kv_mem)

        so beta is multiplied into k once in the batched prologue and the per-head
        [1, dv] beta multiply disappears from the loop. The UNSCALED k_n is still
        required for kv_mem, which must NOT carry beta -- hence both k_n and k_nb.

        SHARDING:
        ---------
        Heads are independent, so the head loop shards over LNC cores; the last
        core absorbs the remainder, so nv need not divide n_progs. Because the
        prologue is BATCHED over heads it is sized to nv_local = h_end - h_start,
        not nv: sizing it to nv would make each core compute and then discard the
        other cores' heads. Consequently prologue tiles are indexed by the
        shard-relative h_local while HBM loads/stores use the absolute h -- mixing
        these up reads the wrong head's data without failing. If nv < n_progs some
        core gets zero heads; nv_local clamps to 1 so tiles stay well-formed.

        DTYPE POLICY:
        -------------
        Everything is fp32, including all three matmuls. This is a MEASURED
        choice, not an oversight:

          - The rank-1 state update outer(k*beta, diff) is UNSAFE in bf16. It is
            a K=1 contraction, so it gets none of the error averaging a long
            contraction provides, and its magnitude is ~1.0x that of the decayed
            state, so its relative error lands essentially undiluted on the
            RECURRENT CARRIER. Fed back over 256 decode steps the state error
            grows without converging. It is also the cheapest matmul to leave in
            fp32, since its operands are [1, dk] and [1, dv].
          - kv_mem and the readout are numerically safe in bf16 but casting them
            is not a win: nc_matmul forbids mixing fp32 and bf16 operands, so each
            needs a bf16 copy of the [dk, dv] state, and they read the pre- and
            post-update state respectively -- two distinct casts per head. The
            added Vector time cancels the PE saving; measurement confirmed no
            change.

        The general filter: the readout injects exactly zero state error because
        it touches only the output, whereas state error compounds across steps.

        CONSTRAINTS AND UNSUPPORTED CASES:
        ----------------------------------
          - nv <= 128, dk <= 128, dv <= 128 (all are partition dimensions:
            nv for the batched prologue, dk/dv for the state tile).
          - Exactly one token per call. There is no multi-token or speculative
            path; use kda_prefill for sequences.
          - No batch dimension, and no varlen / block-table / paged-state
            handling. The state is a dense per-request [nv, dk, dv] tensor, not
            gathered by indirection from a unified paged state tensor, and has not
            been validated against a serving stack's cache layout.
          - Forward only; upstream is also forward-only.
          - out and S_in_ref are caller-allocated destinations. S_in_ref is
            read and written in place, matching the KV-cache convention used
            elsewhere in this package (see core/attention/attention_tkg.py).

    Pseudocode:
        # Batched prologue: all heads this core owns are preprocessed at once,
        # heads on the partition axis.
        q = l2_norm(q_raw) * scale
        k = l2_norm(k_raw)
        beta = sigmoid(beta_logits)
        exp_g = exp(lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias)))
        k_decayed = k * exp_g          # diagonal gate folded into the key
        k_beta    = k * beta           # beta folded into the key (bilinearity)

        # Per-head recurrence; heads are independent and shard across cores.
        for h in this core's heads:
            S      = S_in_ref[h]                      # [dk, dv]
            kv_mem = k_decayed[h] @ S                 # [dv]; decay is INSIDE
            diff   = v[h] - kv_mem                    # [dv]
            S      = S * exp_g[h] + outer(k_beta[h], diff)
            out[h] = q[h] @ S                         # scale already fused in
            S_in_ref[h] = S                           # advanced in place
    """
    nv, dk = q_ref.shape
    dv = v_ref.shape[-1]

    # All preprocessing (L2-norm, gate activation, beta sigmoid) is batched over
    # this core's heads on the partition axis, so nv must fit one partition tile.
    kernel_assert(nv <= 128, f"nv must be <= 128, got {nv}")
    kernel_assert(dk <= 128, f"dk must be <= 128, got {dk}")
    kernel_assert(dv <= 128, f"dv must be <= 128, got {dv}")

    # SPMD head sharding: heads are independent (own state, disjoint output
    # slices), so the head loop shards across cores with no communication.
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        n_progs = nl.num_programs(axes=0)
        prog_id = nl.program_id(axis=0)
    else:
        n_progs = 1
        prog_id = 0

    heads_per_core = nv // n_progs
    h_start = prog_id * heads_per_core
    h_end = nv if prog_id == n_progs - 1 else h_start + heads_per_core

    # All preprocessing below is sized to nv_local, not nv, so a sharded core does
    # not compute and then discard the other cores' heads. When the shard is empty
    # (nv < n_progs) the tiles degenerate to width 1 and the loop does not run.
    n_heads = h_end - h_start
    nv_local = n_heads if n_heads > 0 else 1

    scale_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_t, src=scale_ref[0:1, 0:1])
    lower_bound_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lower_bound_t, src=lower_bound_ref[0:1, 0:1])
    # tensor_scalar requires operand0's partition count to match dst's, so the
    # [1,1] scalars must be explicitly partition-broadcast to the head tiles.
    scale_bc = nl.broadcast_to(scale_t, (nv_local, 1))
    lower_bound_bc = nl.broadcast_to(lower_bound_t, (nv_local, 1))

    # Batch the q/k loads + transposes out of the per-head loop: transpose the
    # whole [nv,dk] to [dk,nv] once, then free-axis slice per head. The L2-norm is
    # done BEFORE the transpose because its sum-of-squares is over dk, the free
    # axis there (activation_reduce only reduces along free axes).
    q_in = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_in, src=q_ref[h_start : h_start + nv_local, 0:dk])
    k_in = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_in, src=k_ref[h_start : h_start + nv_local, 0:dk])

    # Fused L2-norm: square+sum in one Scalar pass, then rsqrt(ss + eps).
    q_ss = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_sq = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=q_sq, op=nl.square, data=q_in, reduce_op=nl.add, reduce_res=q_ss)
    q_rnorm = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=q_rnorm, op=nl.rsqrt, data=q_ss, bias=_L2_EPS)
    k_ss = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    k_sq = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=k_sq, op=nl.square, data=k_in, reduce_op=nl.add, reduce_res=k_ss)
    k_rnorm = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=k_rnorm, op=nl.rsqrt, data=k_ss, bias=_L2_EPS)

    # q is pre-scaled by `scale` so the output needs no scaling. The prologue is
    # Scalar-bound while Vector is nearly idle, so every tensor_scalar here is
    # pinned to Vector to run alongside activations it does not depend on.
    q_rnorm_scaled = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=q_rnorm_scaled, data=q_rnorm, op0=nl.multiply, operand0=scale_bc, engine=nisa.vector_engine)
    q_n = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=q_n, data=q_in, op0=nl.multiply, operand0=q_rnorm_scaled, engine=nisa.vector_engine)
    k_n = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_n, data=k_in, op0=nl.multiply, operand0=k_rnorm, engine=nisa.vector_engine)

    # g_act = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias)), batched.
    g_in = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_in, src=g_ref[h_start : h_start + nv_local, 0:dk])
    dt_bias_in = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt_bias_in, src=dt_bias_ref[h_start : h_start + nv_local, 0:dk])
    a_log_in = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_log_in, src=a_log_ref[h_start : h_start + nv_local, 0:1])
    a_exp = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=a_exp, op=nl.exp, data=a_log_in)
    g_shift = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=g_shift, data1=g_in, data2=dt_bias_in, op=nl.add)
    g_scaled = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g_scaled, data=g_shift, op0=nl.multiply, operand0=a_exp, engine=nisa.vector_engine)
    g_sig = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=g_sig, op=nl.sigmoid, data=g_scaled)
    # g = lower_bound * sigmoid(...) <= 0, so exp(g) is in (0, 1] -- no overflow.
    g_act = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g_act, data=g_sig, op0=nl.multiply, operand0=lower_bound_bc, engine=nisa.vector_engine)
    exp_g_all = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_g_all, op=nl.exp, data=g_act)

    beta_in = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_in, src=beta_ref[h_start : h_start + nv_local, 0:1])
    beta_all = nl.ndarray((nv_local, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta_all, op=nl.sigmoid, data=beta_in)
    # beta folds into the key feeding the rank-1 update, by bilinearity of the
    # outer product: outer(k, beta*(v - kv_mem)) == outer(k*beta, v - kv_mem). The
    # UNscaled k_n is still needed for kv_mem, which must not carry beta.
    k_nb = nl.ndarray((nv_local, dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_nb, data=k_n, op0=nl.multiply, operand0=beta_all, engine=nisa.vector_engine)

    # Transpose to [dk, nv_local] so each head is a [dk,1] free slice.
    q_allT_p = nl.ndarray((dk, nv_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_allT_p, data=q_n, engine=nisa.tensor_engine)
    q_allT = nl.ndarray((dk, nv_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_allT, src=q_allT_p)
    k_allT_p = nl.ndarray((dk, nv_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=k_allT_p, data=k_n, engine=nisa.tensor_engine)
    k_allT = nl.ndarray((dk, nv_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_allT, src=k_allT_p)
    # The rank-1 update needs k*beta as a [1,dk] partition-0 ROW. Gather all heads'
    # rows onto partition 0 once with SBUF->SBUF DMAs (DMA may cross partitions,
    # compute engines may not), replacing a per-head PE transpose plus PSUM drain
    # with prologue DMA on otherwise-idle engines.
    knb_rows = nl.ndarray((1, nv_local * dk), dtype=nl.float32, buffer=nl.sbuf)
    for h_local in range(nv_local):
        nisa.dma_copy(
            dst=knb_rows[0:1, h_local * dk : (h_local + 1) * dk],
            src=k_nb[h_local : h_local + 1, 0:dk],
        )
    # exp_g as [dk, nv_local] so eg_allT[:, h:h+1] is the [dk,1] channel vector.
    eg_allT_p = nl.ndarray((dk, nv_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=eg_allT_p, data=exp_g_all, engine=nisa.tensor_engine)
    eg_allT = nl.ndarray((dk, nv_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eg_allT, src=eg_allT_p)

    # Double-buffer the state DMA: heads are independent, so prefetch head h+1's
    # state before computing h.
    S_bufs = [
        nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf),
        nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf),
    ]
    nisa.dma_copy(dst=S_bufs[0], src=S_in_ref[h_start, 0:dk, 0:dv])

    for h_local in range(n_heads):
        h = h_start + h_local

        S_h = S_bufs[h_local % 2]
        if h + 1 < h_end:
            nisa.dma_copy(dst=S_bufs[(h_local + 1) % 2], src=S_in_ref[h + 1, 0:dk, 0:dv])

        # Preprocessing tiles are shard-local, so they are indexed by the
        # SHARD-RELATIVE h_local; only the HBM DMAs use the absolute h.
        q_h = q_allT[0:dk, h_local : h_local + 1]
        k_h = k_allT[0:dk, h_local : h_local + 1]
        exp_g = eg_allT[0:dk, h_local : h_local + 1]  # [dk,1] per-channel decay
        k_row = knb_rows[0:1, h_local * dk : (h_local + 1) * dk]  # [1,dk] k*beta

        v_h = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_h, src=v_ref[h : h + 1, 0:dv])

        # kv_mem = k @ (Diag(exp_g) @ S) = (k * exp_g) @ S. Folding the decay into
        # the [dk,1] key avoids a full [dk,dv] pass over the state, and unlike
        # GDN's scalar deferred-decay it stays correct for a diagonal gate.
        k_dec = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_dec, data1=k_h, data2=exp_g, op=nl.multiply)

        kv_mem_p = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_p, stationary=k_dec, moving=S_h)
        kv_mem = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        # All PSUM->SBUF drains in this loop are pinned to Scalar: the Vector-class
        # steps between them occupy Vector, and Scalar would otherwise idle.
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_p, engine=nisa.scalar_engine)

        # diff = v_h - kv_mem; beta is carried by k_row.
        diff = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_h, data2=kv_mem, op=nl.subtract)

        # S_h = Diag(exp_g) @ S_h + outer(k_h * beta, diff). The outer product runs
        # on the otherwise-idle PE as a contraction-1 nc_matmul; the decay multiply
        # and additive update then fuse into one Vector pass, with the [dk,1]
        # per-channel exp_g as operand0 (no broadcast needed).
        outer_p = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_p, stationary=k_row, moving=diff)
        # This drain is not required by the operand rules -- outer_p could legally
        # be passed as operand1 below. It is kept for scheduling: it runs on Scalar
        # (~34% busy) while the loop is PE-bound at ~90%, and holding outer_p live
        # until the Vector update retires would extend the PSUM bank's lifetime into
        # the next head's matmul and make PE wait on Vector instead.
        outer_sbuf = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=outer_sbuf, src=outer_p, engine=nisa.scalar_engine)
        # A fresh tile, not in-place over S_h: S_h is a ping-pong buffer that head
        # h+2's prefetch reclaims, so an in-place update would make that prefetch
        # wait on this head's state DMA-out rather than just on its reads.
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

        nisa.dma_copy(dst=out[h : h + 1, 0:dv], src=out_h)
        nisa.dma_copy(dst=S_in_ref[h, 0:dk, 0:dv], src=S_h)

    return out, S_in_ref
