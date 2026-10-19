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

"""NKI kernel for Kimi-K3 Delta Attention (KDA) prefill.

KDA is GatedDeltaNet with a per-channel (diagonal) forget gate instead of a
per-head scalar gate. Ported from FlashKDA
(https://github.com/MoonshotAI/FlashKDA), which is forward-only.

Status: functional reference. Not performance-competitive -- see the performance
note on ``kda_prefill``.

"Chunk" here means the kernel's internal 128-token tiling only. This kernel does
NOT implement chunked prefill in the serving sense: there is no varlen path, no
block table, and no emission of block-aligned intermediate states.

nc_matmul(dst, stationary, moving) = stationary.T @ moving, contracting the
partition axis (<= 128) of both operands; dst in PSUM, inputs in SBUF. To
compute X @ Y, pass stationary=X_T.

WHY SUB = 16
------------
GDN's scalar gate makes the pairwise token decay factor into a [C,C] mask,
exp(gcum_i - gcum_j). KDA's per-channel gate puts the decay INSIDE the
contraction:

    A_ij = sum_d  kbeta[i,d] * k[j,d] * exp(gcum[i,d] - gcum[j,d])

which does not factor into a mask. The only separable form is

    k_decayed = k * exp(gcum)      # entries <= |k|
    k_inv     = k * exp(-gcum)     # entries up to exp(|lower_bound| * SUB)
    L         = k_decayed @ k_inv.T

so k_inv's range grows as exp(|lower_bound| * SUB). At lower_bound = -5,
exp(80) ~ 5.5e34 (SUB=16) is safe in fp32; exp(160) (SUB=32) overflows to inf.
The sub-chunk size is forced by NUMERICS, not by hardware. Masked-away entries
stay finite because exp(gcum_i) <= 1 bounds every product, so masking after the
matmul is safe.

Since nc_matmul costs ~max(moving_free_dim, 64) cycles regardless of the
contraction dim, the eight sub-chunks of a 128-token tile are batched into one
[128,128] matmul and masked block-diagonal rather than issued separately. The
inverse batches for free -- the inverse of a block-diagonal matrix is the
block-diagonal of the inverses. Within each 16x16 block L is strictly lower
triangular, so L^16 = 0 and (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8) is
exact in four factors. This trade costs issued FLOPs: 3.6x more than the
algorithm requires, since only the block diagonal is useful.

WY REFORMULATION
----------------
Naively, U_b = INV_b @ (v_b - k_decayed_b @ S_b) * beta_b puts an inverse-apply
inside the serial state recurrence. Expanding,

    U_b = INV_b @ vbeta_b - (INV_b @ beta_k_decayed_b) @ S_b = u_b - w_b @ S_b

so u and w are S-independent and hoist out of the loop. Mqk @ U defers too:
Mqk is block-diagonal, so (Mqk @ U)[b] depends only on U_b, and the whole term
applies as one matmul after the loop.

HAZARDS
-------
  - A compute-engine operand may not start at a partition offset violating the
    32-partition quadrant granularity: a 16-token band of a [C,d] tile is
    rejected ("Invalid access of 16 partitions starting at partition 16").
    Operands needing that shape are staged sub-chunk-major, [SUB, NSUB, d], so
    every band starts at partition 0.
  - A tensor_tensor-family op may read AT MOST ONE operand from PSUM (and for
    scalar_tensor_tensor, `data` and `operand1` may not both be in PSUM).
    Exceeding this yields silent garbage, not an error.
  - Recurrence updates must be written to a fresh buffer, never dst == data.
  - final_state must not be aliased to state_ref (see Returns).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ....core.utils.kernel_assert import kernel_assert

if TYPE_CHECKING:
    from nki.backends.mlir_tracer.tensor import Tensor

# L2-norm epsilon, folded into the rsqrt bias.
_L2_EPS = 1.0e-6

# Chunk tile and sub-chunk size. Exported because the caller-supplied constant
# masks are sized by them; see build_constant_masks.
CHUNK = 128
SUB_CHUNK = 16


def build_constant_masks():
    """Build the three loop-invariant [CHUNK, CHUNK] constants the kernel takes.

    These are passed in from HBM rather than generated on-chip: generating them
    costs ~11 extra [128, 128] SBUF tiles, which measured 3-5% slower on trn2
    (worst at high head counts, where SBUF pressure binds).

    Returns a dict keyed by kernel argument name, so callers need not know
    SUB_CHUNK -- an internal numerical constraint.
    """
    strict = np.zeros((CHUNK, CHUNK), dtype=np.float32)
    incl = np.zeros((CHUNK, CHUNK), dtype=np.float32)
    ones = np.ones((SUB_CHUNK, SUB_CHUNK), dtype=np.float32)
    for b in range(CHUNK // SUB_CHUNK):
        o = b * SUB_CHUNK
        strict[o : o + SUB_CHUNK, o : o + SUB_CHUNK] = np.tril(ones, -1)
        incl[o : o + SUB_CHUNK, o : o + SUB_CHUNK] = np.tril(ones)
    return {
        "blockdiag_ref": strict,
        "blockdiag_incl_ref": incl,
        "eye_ref": np.eye(CHUNK, dtype=np.float32),
    }


@nki.jit
def kda_prefill(
    q_ref: Tensor,  # [nv, T, dk]   raw query (pre-L2-norm)
    k_ref: Tensor,  # [nv, T, dk]   raw key (pre-L2-norm)
    v_ref: Tensor,  # [nv, T, dv]
    g_ref: Tensor,  # [nv, T, dk]   raw per-channel gate (pre-activation)
    beta_ref: Tensor,  # [nv, T, 1]    beta logits (pre-sigmoid)
    a_log_ref: Tensor,  # [nv, 1]       log-gate parameter
    dt_bias_ref: Tensor,  # [nv, dk]      gate bias
    blockdiag_ref: Tensor,  # [C, C]    block-diagonal strict-lower mask (16x16 blocks)
    blockdiag_incl_ref: Tensor,  # [C, C]   block-diagonal lower-inclusive mask
    eye_ref: Tensor,  # [C, C]        identity
    out: Tensor,  # [nv, T, dv]   attention output (written)
    state_ref: Tensor,  # [nv, dk, dv]  recurrent state, IN/OUT (seeded from, updated in place)
    seed_scale_ref: Tensor,  # [1, 1]   0.0 = fresh prefill / 1.0 = APC continuation
    scale_ref: Tensor,  # [1, 1]        query scale (1/sqrt(dk))
    lower_bound_ref: Tensor,  # [1, 1]  gate lower bound (in [-5, 0])
):
    """Kimi Delta Attention (KDA) prefill: attention output + final state.

    Summary:
        Computes the forward KDA recurrence over a whole fixed-length sequence,
        returning both the attention output and the final recurrent state. Gate
        activation, query/key L2-norm, beta sigmoid and the query scale are fused
        in, so the kernel consumes raw pre-activation model tensors.

        Sharding is SPMD over the nv heads across LNC cores via
        ``nl.program_id(axis=0)``; heads are independent, so no cross-core
        communication is needed.

        Inputs, outputs and the recurrent state are fp32. Matmul operands are
        internally downcast to bf16 with fp32 PSUM accumulation, except on the
        gate path -- see the dtype note below.

    Performance:
        This is a FUNCTIONAL reference, not a performance-competitive kernel.
        Measured on trn2 under LNC2 at nv = 12, dk = dv = 128, T = 512: 474 us,
        which is 0.95% MFU counting only the FLOPs the algorithm requires. The
        memory roofline at this arithmetic intensity (41 flop/byte) caps MFU at
        19%, so the kernel is neither FLOP- nor bandwidth-bound. The cost is
        latency: the per-band state recurrence is a serial dependency chain, and
        the block-diagonal batching issues 3.6x the required FLOPs (see the
        module docstring). Interleaving GROUP independent heads through that
        chain hides most of its latency -- see the note at the head loop.

        Note that profiler-reported MFU counts ISSUED FLOPs and so reads ~4.4x
        higher than the figure above; the useful-work number is the meaningful
        one. Cost is linear in nv and in T.

    Dimensions:
        nv:      Number of value heads (GQA-expanded; each carries its own state)
        T:       Sequence length in tokens; must be a positive multiple of C
        dk:      Key/query head dimension, <= 128
        dv:      Value head dimension, <= 128
        C:       Chunk size, fixed at 128 tokens (one full partition tile)
        SUB:     Sub-chunk size, fixed at 16 tokens (forced by numerics)
        NSUB:    Sub-chunks per chunk, C // SUB = 8
        nchunks: Chunks per sequence, T // C

    Args:
        q_ref (Tensor):
            Query tensor in HBM, raw (pre-L2-norm, pre-scale).
            Shape: [nv, T, dk].  Dtype: nl.float32
        k_ref (Tensor):
            Key tensor in HBM, raw (pre-L2-norm).
            Shape: [nv, T, dk].  Dtype: nl.float32
        v_ref (Tensor):
            Value tensor in HBM.
            Shape: [nv, T, dv].  Dtype: nl.float32
        g_ref (Tensor):
            Per-channel forget-gate tensor in HBM, raw (pre-activation; the
            kernel applies lower_bound * sigmoid(exp(a_log) * (g + dt_bias))).
            Shape: [nv, T, dk].  Dtype: nl.float32
        beta_ref (Tensor):
            Delta-rule beta LOGITS in HBM (the kernel applies the sigmoid).
            Shape: [nv, T, 1].  Dtype: nl.float32
        a_log_ref (Tensor):
            Per-head log-gate parameter in HBM; exp(a_log) scales the gate
            pre-activation.
            Shape: [nv, 1].  Dtype: nl.float32
        dt_bias_ref (Tensor):
            Per-head, per-CHANNEL gate bias in HBM, added to g before the
            sigmoid.
            Shape: [nv, dk].  Dtype: nl.float32
        blockdiag_ref (Tensor):
            Constant mask in HBM: 16x16 block-diagonal, STRICTLY lower
            triangular within each block, zero elsewhere. Applied to L.
            Shape: [C, C].  Dtype: nl.float32
        blockdiag_incl_ref (Tensor):
            Constant mask in HBM: 16x16 block-diagonal, lower triangular
            INCLUSIVE of the diagonal. Applied to Mqk.
            Shape: [C, C].  Dtype: nl.float32
        eye_ref (Tensor):
            Constant [C, C] identity in HBM, used to seed the Neumann inverse.
            Shape: [C, C].  Dtype: nl.float32
        out (Tensor):
            Caller-allocated destination for the attention output, token-major.
            Shape: [nv, T, dv].  Dtype: nl.float32
        state_ref (Tensor):
            Recurrent state in HBM. IN/OUT: read to seed the recurrence (scaled
            by seed_scale_ref), then overwritten with the final state (see
            Returns).
            Shape: [nv, dk, dv].  Dtype: nl.float32
        seed_scale_ref (Tensor):
            Runtime multiplier on the state seed: 0.0 for a fresh prefill
            (equivalent to a zero initial state), 1.0 to continue from
            state_ref (automatic prefix caching). A runtime tensor rather than a
            trace-time flag so both modes share one compiled graph.
            Shape: [1, 1].  Dtype: nl.float32
        scale_ref (Tensor):
            Query scale, normally 1/sqrt(dk); folded into the L2-normed query so
            the output needs no post-scale.
            Shape: [1, 1].  Dtype: nl.float32
        lower_bound_ref (Tensor):
            Gate lower bound, expected in [-5, 0]. The activated gate is
            lower_bound * sigmoid(...), hence <= 0, so exp(gate) <= 1.
            Shape: [1, 1].  Dtype: nl.float32

    Returns:
        out (Tensor):
            The caller-provided output tensor, filled with the attention output
            in token-major order.
            Shape: [nv, T, dv].  Dtype: nl.float32
        state_ref (Tensor):
            The caller-provided state tensor, updated in place to the final
            recurrent state. Each head's [dk, dv] slice is read once to seed the
            recurrence and written once at the end, so the caller needs no copy
            to persist it.
            Shape: [nv, dk, dv].  Dtype: nl.float32

    Notes:
        TENSOR ORIENTATIONS:
        --------------------
        HBM inputs are token-major (tokens on axis 1), matching what the DMA
        wants: a [C, d] SBUF tile with tokens on the partition axis.

        On chip the kernel holds two orientations, and mixing them up is the
        easiest way to break it. Tiles suffixed ``_T`` are CHANNEL-major [d, C]
        (partition = channel), so a sub-chunk is the free slice
        [:, 16b : 16b+16]; everything the serial recurrence touches lives here.
        Unsuffixed [C, d] tiles are TOKEN-major and are only ever used
        full-width. Tiles suffixed ``3`` are sub-chunk-major [SUB, NSUB, d].

        dt_bias is per-channel, so in the channel-major layout it must be a
        [dk, 1] COLUMN -- hence the transpose of the [1, dk] load, not a row.

        SERIAL RECURRENCE DEPENDENCY:
        -----------------------------
        Sub-chunk b+1 needs the state produced by b, and between the two
        accumulations the state is rescaled by a per-channel [dk, 1] decay:

            S_{b+1} = S_b * exp(g_total_b) + dS_b

        A single PSUM accumulator spanning the eight sub-chunks is therefore
        INVALID. Unrolling gives S_8 = sum_b dS_b * prod_{j>b} exp(g_total_j), so
        each contribution is decayed by every subsequent sub-chunk's gate. PSUM
        can only sum into a bank, never rescale what is already banked, so each
        dS_b must be combined with the rescaled state before the next matmul.

        Only U3[b], out_inter[b] and S are on this chain. The inverse, w, u,
        k_restored and the Mqk @ U term are all S-independent and hoisted out.

        DTYPE POLICY:
        -------------
        Matmul operands are bf16 with fp32 PSUM accumulation (~4x PE throughput);
        PSUM destinations and HBM outputs are fp32.

        The GATE PATH MUST STAY fp32 -- gcum_T, exp_gcum, exp_neg_gcum and
        k_restored_T. These are EXPONENTS, not values. At lower_bound = -5 and
        SUB = 16 the cumsum reaches |gcum| ~ 80, where bf16's spacing is 0.5, and
        a half-unit error in an exponent makes exp() wrong by ~28%. Rounding a
        value costs a relative epsilon; rounding an exponent costs exp(epsilon).
        That is why the products k_dec_T / k_inv_T / q_dec_T may be bf16 while
        the tensors feeding exp() may not.

        CONSTRAINTS AND UNSUPPORTED CASES:
        ----------------------------------
          - dk <= 128 and dv <= 128 (partition dimension); upstream uses
            dk == dv == 128.
          - T must be a positive multiple of C = 128.
          - Fixed-length sequences only: no varlen / cu_seqlens path, no block
            table, and no emission of block-aligned intermediate states, so this
            cannot currently populate a serving prefix cache.
          - The state layout is a dense per-request [nv, dk, dv] tensor and has
            not been validated against a serving stack's cache layout. There is
            no gather from a unified paged state tensor.
          - SUB = 16 is fixed by numerics (see the module docstring), NOT by
            hardware; NSUB = C // SUB = 8 follows.
          - NEUMANN = 4 is exact, not a truncation: L is strictly lower
            triangular within each 16x16 block, so L^16 = 0 terminates the series.
          - Forward only; upstream is also forward-only.
          - out and state_ref are caller-allocated destinations. state_ref is
            read to seed the recurrence and written back in place, matching the
            KV-cache convention used elsewhere in this package (see
            core/attention/attention_tkg.py).

    Pseudocode:
        # Per head h (heads are independent and shard across cores; GROUP of
        # them are stepped in lockstep to hide the serial chain's latency).
        S = state_ref[h] * seed_scale          # [dk, dv]; 0.0 = fresh prefill
        for each 128-token chunk of the sequence:
            # --- Off-chain preprocessing: nothing here reads S ---
            q = l2_norm(q_raw) * scale         # fused norm + query scale
            k = l2_norm(k_raw)
            beta = sigmoid(beta_logits)
            g = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias))
            gcum = cumsum(g) within each 16-token sub-chunk   # per channel
            k_decayed = k * exp(gcum)          # entries <= |k|
            k_inv     = k * exp(-gcum)         # range ~ exp(|lower_bound|*SUB)
            q_decayed = q * exp(gcum)

            # The per-channel gate does not factor into a [C,C] mask, so the
            # decay folds into the operands instead. All eight 16-token bands
            # are batched into one [128,128] matmul and masked block-diagonal.
            L   = blockdiag_mask * (k_decayed_beta @ k_inv.T)
            INV = (I - L)(I + L^2)(I + L^4)(I + L^8)    # exact: L^16 = 0
            u   = INV @ (v * beta)             # S-independent, hoisted out
            w   = INV @ (k_decayed * beta)     # S-independent, hoisted out
            Mqk = blockdiag_incl_mask * (q_decayed @ k_inv.T)

            # --- Serial chain: one step per 16-token band ---
            for b in range(8):
                U[b]         = u[b] - w[b] @ S           # WY reformulation
                out_inter[b] = q_decayed[b] @ S
                S            = S * exp(g_total[b]) + k_restored[b].T @ U[b]

            # --- Deferred intra-chunk output; Mqk is block-diagonal, so this
            #     whole term is one matmul after the loop ---
            out[h, chunk] = out_inter + Mqk @ U
        state_ref[h] = S                       # final state, written in place
    """
    nv, T, dk = q_ref.shape
    dv = v_ref.shape[-1]
    C = CHUNK
    SUB = SUB_CHUNK
    NSUB = C // SUB
    nchunks = T // C
    # 4 Neumann factors: (I - L)(I + L^2)(I + L^4)(I + L^8), exact for L^16 = 0.
    NEUMANN = 4

    kernel_assert(dk <= 128, f"dk must be <= 128, got {dk}")
    kernel_assert(dv <= 128, f"dv must be <= 128, got {dv}")
    kernel_assert(T % C == 0, f"T must be a multiple of {C}, got {T}")
    # T % C == 0 admits T == 0, which would leave `out` untouched instead of failing.
    kernel_assert(T >= C, f"T must be at least {C}, got {T}")

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

    # --- Constant [C,C] tiles, loaded once ---
    blockdiag = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=blockdiag, src=blockdiag_ref[0:C, 0:C])
    blockdiag_incl = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=blockdiag_incl, src=blockdiag_incl_ref[0:C, 0:C])
    eye = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=eye_ref[0:C, 0:C])
    # bf16 copy of eye for the Neumann loop (element-wise on bf16 tiles).
    eye_bf16 = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eye_bf16, src=eye)

    # --- Scalars ---
    scale_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_t, src=scale_ref[0:1, 0:1])
    lower_bound_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lower_bound_t, src=lower_bound_ref[0:1, 0:1])
    lower_bound_bc = nl.broadcast_to(lower_bound_t, (dk, 1))
    # `scale` folds into q's L2 normalizer, a [C,1] token-major operand.
    scale_bc = nl.broadcast_to(scale_t, (C, 1))
    seed_scale = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=seed_scale, src=seed_scale_ref[0:1, 0:1])
    seed_scale_bc = nl.broadcast_to(seed_scale, (dk, 1))

    # --- Scan constants for the per-sub-chunk cumsum over tokens ---
    # tensor_tensor_scan computes result[i] = data0[i] * result[i-1] + data1[i]
    # along the FREE axis, so data0 = ones gives a running sum.
    ones_scan = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_scan, value=1.0)
    zero_init = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_init, value=0.0)

    # --- Head interleaving ---------------------------------------------------
    # The per-band state recurrence is a serial dependency chain whose cost is
    # latency, not throughput: measured on trn2, an extra dependent op inside it
    # costs ~178 ns while an extra full-width off-chain copy costs ~0. So the
    # engines are mostly idle while the chain steps, and the way to use them is
    # to step GROUP independent heads in lockstep -- head i+1's band ops depend
    # on nothing head i produced, so the scheduler can hide each head's latency
    # behind the others' work.
    #
    # GROUP_MAX = 6 is measured, not derived: each head in flight holds its own
    # chunk-sized working set, so wider groups trade latency hiding against SBUF
    # pressure. On trn2, 6 is the best of {1, 2, 4, 6, 8} on every config -- at 8
    # the largest one (upstream_prod, 32 heads) turns back up, 3707 -> 3804 us.
    #
    # A group must divide the core's head count exactly, so the usable width is
    # the largest divisor at or below the cap: 6 heads/core takes 6, 8 takes 4.
    GROUP_MAX = 6
    GROUP = max(g for g in range(1, GROUP_MAX + 1) if (h_end - h_start) % g == 0)

    def _head_setup(h):
        """Per-head constants and the seeded state. Once per head, not per chunk."""

        # dt_bias is per-channel; in the channel-major [dk, C] layout that is the
        # PARTITION axis, so hold it as a [dk,1] tensor_scalar operand.
        dt_bias_row = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dt_bias_row, src=dt_bias_ref[h : h + 1, 0:dk])
        dt_bias_col_p = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=dt_bias_col_p, data=dt_bias_row, engine=nisa.tensor_engine)
        dt_bias_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dt_bias_col, src=dt_bias_col_p)

        a_log_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_log_t, src=a_log_ref[h : h + 1, 0:1])
        a_exp_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=a_exp_t, op=nl.exp, data=a_log_t)
        a_exp_bc = nl.broadcast_to(a_exp_t, (dk, 1))

        # Seed the state: seed_scale = 0.0 starts from zero, 1.0 continues from
        # state_ref.
        S_seed = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=S_seed, src=state_ref[h, 0:dk, 0:dv])
        S = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=S, data=S_seed, op0=nl.multiply, operand0=seed_scale_bc)

        return {"dt_bias_col": dt_bias_col, "a_exp_bc": a_exp_bc}, S

    def _preprocess(h, t0, hc):
        """Everything in a chunk that does not read S. Entirely off the chain."""
        dt_bias_col = hc["dt_bias_col"]
        a_exp_bc = hc["a_exp_bc"]
        # --- Load + fused preprocessing (token-major [C, d]) ---
        q_in = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_in, src=q_ref[h, t0 : t0 + C, 0:dk])
        k_in = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_in, src=k_ref[h, t0 : t0 + C, 0:dk])
        v_in = nl.ndarray((C, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_in, src=v_ref[h, t0 : t0 + C, 0:dv])
        g_in = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_in, src=g_ref[h, t0 : t0 + C, 0:dk])
        beta_in = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=beta_in, src=beta_ref[h, t0 : t0 + C, 0:1])

        beta = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=beta, op=nl.sigmoid, data=beta_in)

        # Fused QK L2-norm. The sum of squares is over d, the FREE axis here,
        # so it is a single activation_reduce pass.
        q_ss = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        q_sq = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=q_sq, op=nl.square, data=q_in, reduce_op=nl.add, reduce_res=q_ss)
        q_rnorm = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=q_rnorm, op=nl.rsqrt, data=q_ss, bias=_L2_EPS)
        k_ss = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        k_sq = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=k_sq, op=nl.square, data=k_in, reduce_op=nl.add, reduce_res=k_ss)
        k_rnorm = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=k_rnorm, op=nl.rsqrt, data=k_ss, bias=_L2_EPS)
        # Fold `scale` into q's normalizer so no output scaling is needed.
        q_rnorm_scaled = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_rnorm_scaled, data1=q_rnorm, data2=scale_bc, op=nl.multiply)
        q_n = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=q_n, data=q_in, op0=nl.multiply, operand0=q_rnorm_scaled)
        k_n = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=k_n, data=k_in, op0=nl.multiply, operand0=k_rnorm)

        # --- Transpose to channel-major [d, C] ---
        # Everything the sequential recurrence touches lives here, so that a
        # sub-chunk is a free-axis slice [:, 16b : 16b+16].
        q_T_p = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_p, data=q_n, engine=nisa.tensor_engine)
        q_T = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_p)
        k_T_p = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_p, data=k_n, engine=nisa.tensor_engine)
        k_T = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_p)
        g_T_p = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_T_p, data=g_in, engine=nisa.tensor_engine)
        g_T = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=g_T, src=g_T_p)

        # g_act = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias))
        # The add and the multiply are one instruction: tensor_scalar's second
        # operator is free (same cost as performing one).
        g_scaled = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=g_scaled,
            data=g_T,
            op0=nl.add,
            operand0=dt_bias_col,
            op1=nl.multiply,
            operand1=a_exp_bc,
        )
        g_sig = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sig, op=nl.sigmoid, data=g_scaled)
        # lower_bound <= 0 and sigmoid in (0,1), so g_act <= 0.
        g_act = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g_act, data=g_sig, op0=nl.multiply, operand0=lower_bound_bc)

        # Per-sub-chunk cumsum over tokens. Tokens are the free axis, which is
        # what tensor_tensor_scan scans; each sub-chunk restarts from zero, so
        # the scans are independent.
        gcum_T = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        for b in range(NSUB):
            o = b * SUB
            nisa.tensor_tensor_scan(
                dst=gcum_T[0:dk, o : o + SUB],
                data0=ones_scan[0:dk, 0:SUB],
                data1=g_act[0:dk, o : o + SUB],
                initial=zero_init[0:dk, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

        exp_gcum = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_gcum, op=nl.exp, data=gcum_T)  # in (0, 1]
        # exp(-gcum): the large one. Bounded by exp(|lower_bound| * SUB).
        exp_neg_gcum = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_neg_gcum, op=nl.exp, data=gcum_T, scale=-1.0)

        k_dec_T = nl.ndarray((dk, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_dec_T, data1=k_T, data2=exp_gcum, op=nl.multiply)
        k_inv_T = nl.ndarray((dk, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_inv_T, data1=k_T, data2=exp_neg_gcum, op=nl.multiply)
        q_dec_T = nl.ndarray((dk, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_dec_T, data1=q_T, data2=exp_gcum, op=nl.multiply)

        # k_restored = k * exp(g_total - gcum), computed with a non-positive
        # exponent -- NOT as k_inv * exp(g_total), which would multiply a ~1e34
        # value by a ~1e-34 one and lose the result to rounding.
        #
        # Done at FULL width, not per sub-chunk: 4 wide ops instead of 3 * NSUB
        # narrow ones, measured 1.3% faster on trn2. Each band needs its own
        # g_total, which is that band's last gcum column; a stride-0 innermost
        # free dim replicates it SUB times, so one copy materializes all NSUB
        # band totals as a [dk, C] tile. Materializing it is deliberate --
        # feeding that stride-0 access pattern straight into the subtract below
        # measured slower than this extra tensor_copy.
        gtot_full = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gtot_full.ap(pattern=[[C, dk], [SUB, NSUB], [1, SUB]]),
            # Innermost stride 0: read column (b * SUB + SUB - 1) SUB times.
            src=gcum_T.ap(pattern=[[C, dk], [SUB, NSUB], [0, SUB]], offset=SUB - 1),
        )
        gdiff = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gdiff, data1=gcum_T, data2=gtot_full, op=nl.subtract)
        # exp(-(gcum - g_total)) = exp(g_total - gcum), exponent <= 0.
        decay = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=decay, op=nl.exp, data=gdiff, scale=-1.0)
        k_restored_T = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_restored_T, data1=k_T, data2=decay, op=nl.multiply)

        # --- L and Mqk: one [128,128] matmul each, then block-diagonal mask ---
        # nc_matmul(stationary=k_dec_T, moving=k_inv_T) = k_dec @ k_inv.T
        L_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=L_p, stationary=k_dec_T, moving=k_inv_T)
        # Drained to SBUF before masking. Fusing the mask into a Vector read
        # straight out of L_p is legal but measurably SLOWER: it holds the PSUM
        # bank live across the Vector pass and the next nc_matmul into that bank
        # stalls behind it.
        L = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L, src=L_p)
        nisa.tensor_tensor(dst=L, data1=L, data2=blockdiag, op=nl.multiply)
        L_masked = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=L_masked, data=L, op0=nl.multiply, operand0=beta)

        Mqk_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Mqk_p, stationary=q_dec_T, moving=k_inv_T)
        Mqk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Mqk, src=Mqk_p)
        Mqk_masked = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=Mqk_masked, data1=Mqk, data2=blockdiag_incl, op=nl.multiply)
        # Mqk is applied as Mqk @ U, so the matmul needs its transpose.
        Mqk_T_p = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=Mqk_T_p, data=Mqk_masked, engine=nisa.tensor_engine)
        Mqk_T = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Mqk_T, src=Mqk_T_p, engine=nisa.scalar_engine)

        # --- INV = (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8) ---
        # Exact because L^16 = 0 within every 16x16 block. Only the transpose
        # INV_T is maintained; the non-transposed form is dead downstream.
        L_T_p = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=L_T_p, data=L_masked, engine=nisa.tensor_engine)
        L_T = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_T, src=L_T_p, engine=nisa.scalar_engine)

        # INV_T = (I - L).T = I - L_T
        INV_T = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=INV_T, data1=eye_bf16, data2=L_T, op=nl.subtract)

        # Lp = L @ L : nc_matmul(L_T, L) = L_T.T @ L = L @ L
        Lp_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Lp_p, stationary=L_T, moving=L_masked)
        Lp = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Lp, src=Lp_p, engine=nisa.scalar_engine)
        # Lp_T = (L @ L).T = L_T @ L_T : nc_matmul(L, L_T)
        Lp_T_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Lp_T_p, stationary=L_masked, moving=L_T)
        Lp_T = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Lp_T, src=Lp_T_p, engine=nisa.scalar_engine)

        # NEUMANN-2 full iterations, then one peeled (Lp is dead after it).
        for _ in nl.sequential_range(NEUMANN - 2):
            ipl = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ipl, data1=eye_bf16, data2=Lp, op=nl.add)
            # INV_T_new = (INV @ (I+Lp)).T = (I+Lp).T @ INV_T
            #           = nc_matmul(stationary=(I+Lp), moving=INV_T)
            inv_new_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=inv_new_p, stationary=ipl, moving=INV_T)
            nisa.tensor_copy(dst=INV_T, src=inv_new_p, engine=nisa.scalar_engine)
            # Lp_T_new = Lp_T @ Lp_T : nc_matmul(Lp, Lp_T)
            lpt_new_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=lpt_new_p, stationary=Lp, moving=Lp_T)
            # Lp_new = Lp @ Lp : nc_matmul(Lp_T, Lp)
            lp_new_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=lp_new_p, stationary=Lp_T, moving=Lp)
            nisa.tensor_copy(dst=Lp, src=lp_new_p, engine=nisa.scalar_engine)
            nisa.tensor_copy(dst=Lp_T, src=lpt_new_p, engine=nisa.scalar_engine)

        ipl_final = nl.ndarray((C, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ipl_final, data1=eye_bf16, data2=Lp, op=nl.add)
        inv_final_p = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=inv_final_p, stationary=ipl_final, moving=INV_T)
        nisa.tensor_copy(dst=INV_T, src=inv_final_p, engine=nisa.scalar_engine)

        # --- WY factors u and w: S-independent, hoisted out of the loop ---
        # u = INV @ (v * beta),  w = INV @ (k_decayed * beta). Both need a
        # token-major moving operand, so k_dec_T is transposed back.
        vbeta = nl.ndarray((C, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vbeta, data=v_in, op0=nl.multiply, operand0=beta)
        k_dec_p = nl.ndarray((C, dk), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=k_dec_p, data=k_dec_T, engine=nisa.tensor_engine)
        k_dec = nl.ndarray((C, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_dec, src=k_dec_p)
        kbeta_dec = nl.ndarray((C, dk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=kbeta_dec, data=k_dec, op0=nl.multiply, operand0=beta)

        # w_T = (INV @ kbeta_dec).T = kbeta_dec.T @ INV_T, so it comes straight
        # out of nc_matmul(stationary=kbeta_dec, moving=INV_T) channel-major,
        # with no transpose round-trip.
        #
        # u is not built full-width: the recurrence only reads per-sub-chunk
        # token-major bands, which the u3 staging below produces directly.
        w_T_p = nl.ndarray((dk, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=w_T_p, stationary=kbeta_dec, moving=INV_T)
        w_T = nl.ndarray((dk, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w_T, src=w_T_p)

        # --- Sub-chunk-major staging [SUB, NSUB, d] ---
        # The state update's operands (k_restored_b, U_b) must be token-major
        # [SUB, d]. The sub-chunk goes on the FREE axis so band b is
        # [0:SUB, b, :] -- 16 partitions at offset 0. A [16b, 16b+16) partition
        # range is NOT a legal compute operand: engine accesses respect the
        # 32-partition quadrant granularity and the backend rejects it
        # ("Invalid access of 16 partitions starting at partition 16").
        #
        # k_restored and u are S-independent, so this staging precedes the
        # recurrence.
        HALF = NSUB // 2
        kr3 = nl.ndarray((SUB, NSUB, dk), dtype=nl.bfloat16, buffer=nl.sbuf)
        u3 = nl.ndarray((SUB, NSUB, dv), dtype=nl.float32, buffer=nl.sbuf)
        for half in range(2):
            # PSUM free dim caps at 512, so stage HALF = 4 bands of d <= 128.
            kr3_p = nl.ndarray((SUB, HALF, dk), dtype=nl.float32, buffer=nl.psum)
            u3_p = nl.ndarray((SUB, HALF, dv), dtype=nl.float32, buffer=nl.psum)
            for bb in range(HALF):
                o = (half * HALF + bb) * SUB
                nisa.nc_transpose(
                    dst=kr3_p[0:SUB, bb, 0:dk],
                    data=k_restored_T[0:dk, o : o + SUB],
                    engine=nisa.tensor_engine,
                )
                nisa.nc_matmul(
                    dst=u3_p[0:SUB, bb, 0:dv],
                    stationary=INV_T[0:C, o : o + SUB],
                    moving=vbeta,
                )
            nisa.tensor_copy(dst=kr3[0:SUB, half * HALF : (half + 1) * HALF, 0:dk], src=kr3_p)
            nisa.tensor_copy(dst=u3[0:SUB, half * HALF : (half + 1) * HALF, 0:dv], src=u3_p)

        # --- Sequential sub-chunk recurrence (the serial chain) ---
        #   U_b         = u_b - w_b @ S_b
        #   out_inter_b = q_dec_b @ S_b
        #   S           = S * exp(g_total_b) + k_restored_b.T @ U_b
        # U3 is sub-chunk-major to match u3/kr3, so the state update needs no
        # transpose. out_inter is channel-major; it is off the serial chain, so
        # its orientation is free.
        U3 = nl.ndarray((SUB, NSUB, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        U_T = nl.ndarray((dv, C), dtype=nl.float32, buffer=nl.sbuf)
        out_inter_T = nl.ndarray((dv, C), dtype=nl.float32, buffer=nl.sbuf)

        return {
            "q_dec_T": q_dec_T,
            "exp_gcum": exp_gcum,
            "w_T": w_T,
            "u3": u3,
            "kr3": kr3,
            "U3": U3,
            "U_T": U_T,
            "out_inter_T": out_inter_T,
            "Mqk_T": Mqk_T,
        }

    def _band_step(P, S, b):
        """One step of the serial chain for one head. Returns the new state."""
        q_dec_T, exp_gcum, w_T = P["q_dec_T"], P["exp_gcum"], P["w_T"]
        u3, kr3, U3, U_T, out_inter_T = P["u3"], P["kr3"], P["U3"], P["U_T"], P["out_inter_T"]
        o = b * SUB
        S_bf16 = nl.ndarray((dk, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=S_bf16, src=S)

        # w_b @ S_b = nc_matmul(stationary=w_T_b, moving=S), already
        # token-major. The PSUM result is consumed directly as data2 --
        # legal because data1 (u3) is in SBUF; only *both* operands in PSUM
        # is illegal. Saves a drain on the serial chain.
        wS_p = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=wS_p, stationary=w_T[0:dk, o : o + SUB], moving=S_bf16)
        nisa.tensor_tensor(
            dst=U3[0:SUB, b, 0:dv],
            data1=u3[0:SUB, b, 0:dv],
            data2=wS_p,
            op=nl.subtract,
        )
        # The deferred Mqk @ U needs U channel-major. That consumer is off
        # the serial chain, so this transpose is not a dependency of the
        # next iteration.
        U_T_b_p = nl.ndarray((dv, SUB), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=U_T_b_p, data=U3[0:SUB, b, 0:dv], engine=nisa.tensor_engine)
        nisa.tensor_copy(dst=U_T[0:dv, o : o + SUB], src=U_T_b_p)

        # (q_dec_b @ S_b).T = S_b.T @ q_dec_b.T : nc_matmul(stationary=S, ...).
        oi_p = nl.ndarray((dv, SUB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=oi_p, stationary=S_bf16, moving=q_dec_T[0:dk, o : o + SUB])
        nisa.tensor_copy(dst=out_inter_T[0:dv, o : o + SUB], src=oi_p)

        # S = S * exp(g_total_b) + k_restored_b.T @ U_b. Both matmul operands
        # are token-major band slices at partition 0, so the serial chain
        # holds no transpose and no layout drain.
        dS_p = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=dS_p, stationary=kr3[0:SUB, b, 0:dk], moving=U3[0:SUB, b, 0:dv])
        # The decay multiply and the additive update fuse into ONE Vector
        # pass writing a FRESH buffer (dst == data would be a WAW on the
        # recurrence). dS_p is read straight from PSUM as operand1, legal
        # because `data` is in SBUF. exp(g_total_b) is exp_gcum's last column
        # for the sub-chunk, so it is reused rather than recomputed.
        S_new = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=S_new,
            data=S,
            op0=nl.multiply,
            operand0=exp_gcum[0:dk, o + SUB - 1 : o + SUB],
            op1=nl.add,
            operand1=dS_p,
        )
        return S_new

    def _epilogue(h, t0, P):
        """Deferred intra-chunk output. Reads only U_T, which the chain wrote."""
        U_T, out_inter_T, Mqk_T = P["U_T"], P["out_inter_T"], P["Mqk_T"]
        # --- Deferred intra-chunk output: out = out_inter + Mqk @ U ---
        # Mqk is block-diagonal, so (Mqk @ U)[b] depends only on U_b and the
        # whole term applies as ONE [128,128] x [128,dv] matmul after the loop.
        U_full_p = nl.ndarray((C, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=U_full_p, data=U_T, engine=nisa.tensor_engine)
        U_full = nl.ndarray((C, dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=U_full, src=U_full_p)
        out_intra_p = nl.ndarray((C, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=out_intra_p, stationary=Mqk_T, moving=U_full)

        # out_inter is transposed straight into the final add: out_intra_p is
        # read from PSUM as data2 (data1 is in SBUF), saving a drain.
        out_inter_p = nl.ndarray((C, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=out_inter_p, data=out_inter_T, engine=nisa.tensor_engine)
        out_inter = nl.ndarray((C, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_inter, src=out_inter_p)

        out_c = nl.ndarray((C, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out_c, data1=out_inter, data2=out_intra_p, op=nl.add)
        nisa.dma_copy(dst=out[h, t0 : t0 + C, 0:dv], src=out_c)

    for hg in range(0, h_end - h_start, GROUP):
        heads = [h_start + hg + i for i in range(GROUP)]
        hc = [None] * GROUP
        S = [None] * GROUP
        for i in range(GROUP):
            hc[i], S[i] = _head_setup(heads[i])

        for c in nl.sequential_range(nchunks):
            t0 = c * C
            P = [_preprocess(heads[i], t0, hc[i]) for i in range(GROUP)]
            # Bands are interleaved, not heads: all GROUP heads take band b before
            # any takes band b+1, so independent work always sits adjacent to the
            # chain op that is waiting.
            for b in range(NSUB):
                for i in range(GROUP):
                    S[i] = _band_step(P[i], S[i], b)
            for i in range(GROUP):
                _epilogue(heads[i], t0, P[i])

        for i in range(GROUP):
            nisa.dma_copy(dst=state_ref[heads[i], 0:dk, 0:dv], src=S[i])

    return out, state_ref
