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

"""
GDN TKG Kernel

Single-token gated delta-rule decode kernel with the surrounding per-head torch
ops fused in, so the Qwen3.5 GDN decode layer only calls this kernel + conv1d.

Takes RAW inputs and folds the small per-head ops directly into the kernel:

        +-------------------+   +-------------------+   +-----------------------+   +-------------------+
        | q,k,v,b,a,z (HBM) |-->|  l2norm(q,k)      |-->|  gated delta-rule     |-->| RMSNormGated(o,z) |--> out
        | RAW, GQA-expanded |   |  beta=sigmoid(b)  |   |  (recurrent, per-head)|   |                   |
        +-------------------+   |  g gate           |   +-----------------------+   +-------------------+
                                +-------------------+

Folded ops:
    1. beta = sigmoid(b)                          [b RAW, pre-sigmoid]
    2. g    = -exp(A_log) * softplus(a + dt_bias) [a RAW, per-v-head A_log/dt_bias]
    3. l2norm(q), l2norm(k) over K (eps=1e-6), composed with the 1/sqrt(K) Q-scale
    4. RMSNormGated output: (o*rsqrt(mean_V(o^2)+eps))*norm_weight[V]*silu(z)

The kernel still applies exp(g) and Q*(1/sqrt(K)) internally (do NOT pre-apply
upstream). GQA (repeat_interleave) is kept in Python: q/k arrive already
GQA-expanded to num_v_heads, matching the flat [BH, .] contract.

Math: single-token recurrent gated delta rule (unchanged core).
    S1    = S_in * exp(g)                  [K, V]
    kv    = k^T @ S1                       [1, V]
    delta = (v - kv) * beta                [1, V]
    S_out = S1 + outer(k, delta)           [K, V]
    out   = (q/sqrt(K))^T @ S_out          [1, V]
"""

import math
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.logging import get_logger
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

logger = get_logger("gdn_tkg")


@nki.jit
def gdn_tkg(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    b: nl.NkiTensor,
    a: nl.NkiTensor,
    A_log: nl.NkiTensor,
    dt_bias: nl.NkiTensor,
    z: nl.NkiTensor,
    norm_weight: nl.NkiTensor,
    state_in: nl.NkiTensor,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Fused single-token gated delta-rule decode step.

    Folds l2norm(q,k), beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias), and
    RMSNormGated(out, z) around the recurrent delta-rule core, then advances the
    per-head recurrent state by one token.

    Dimensions:
        BH: Flattened batch * num_v_heads (heads GQA-expanded, cycling fastest per batch)
        B: Batch size = BH // num_v_heads
        num_v_heads: Number of value heads
        K: Key/state head dimension (<= 128)
        V: Value head dimension (<= 128)

    Args:
        q (nl.NkiTensor): [BH, K] @ HBM, RAW query (GQA-expanded, not l2-normed).
        k (nl.NkiTensor): [BH, K] @ HBM, RAW key (GQA-expanded, not l2-normed).
        v (nl.NkiTensor): [BH, V] @ HBM, value.
        b (nl.NkiTensor): [BH] @ HBM, RAW beta pre-sigmoid.
        a (nl.NkiTensor): [BH] @ HBM, RAW gate input (pre-softplus).
        A_log (nl.NkiTensor): [num_v_heads] @ HBM, per-v-head log decay coefficient.
        dt_bias (nl.NkiTensor): [num_v_heads] @ HBM, per-v-head softplus bias.
        z (nl.NkiTensor): [BH, V] @ HBM, RMSNormGated gate input.
        norm_weight (nl.NkiTensor): [V] @ HBM, RMSNormGated weight (gamma).
        state_in (nl.NkiTensor): [BH, K, V] @ HBM, incoming recurrent state.

    Returns:
        out (nl.NkiTensor): [BH, V] @ HBM, RMSNormGated decode output (bf16).
        state_out (nl.NkiTensor): [BH, K, V] @ HBM, updated recurrent state (fp32).

    Notes:
        - Per-head independent recurrence (no cross-head reduction), so the BH
          axis is sharded across the SPMD grid; each core writes only its own
          contiguous head slice. LNC1 is byte-identical to single-core.
        - Do NOT pre-apply exp(g) or the 1/sqrt(K) Q-scale upstream; the kernel
          applies both internally.

    Pseudocode:
        beta = sigmoid(b)                             # [b RAW, pre-sigmoid]
        g    = -exp(A_log) * softplus(a + dt_bias)    # [a RAW, per-v-head A_log/dt_bias]
        q    = l2norm(q) * (1 / sqrt(K))              # l2norm over K (eps=1e-6), Q-scale composed
        k    = l2norm(k)                              # l2norm over K (eps=1e-6)
        # single-token recurrent gated delta rule (per head)
        S1    = S_in * exp(g)                          # [K, V]
        kv    = k^T @ S1                               # [1, V]
        delta = (v - kv) * beta                        # [1, V]
        S_out = S1 + outer(k, delta)                   # [K, V]
        out   = q^T @ S_out                            # [1, V]
        # RMSNormGated output
        out   = (out * rsqrt(mean_V(out^2) + eps)) * norm_weight[V] * silu(z)
    """
    BH, K = q.shape
    _, V = v.shape
    num_v_heads = A_log.shape[0]
    B = BH // num_v_heads
    kernel_assert(
        state_in.shape == (BH, K, V),
        f"state_in must be [BH, K, V], got {state_in.shape=}, {BH=}, {K=}, {V=}",
    )
    kernel_assert(K <= 128 and V <= 128, f"K and V must each be <= 128, got {K=}, {V=}")
    kernel_assert(
        dt_bias.shape[0] == num_v_heads,
        f"dt_bias must be [num_v_heads], got {dt_bias.shape[0]=}, {num_v_heads=}",
    )
    kernel_assert(B * num_v_heads == BH, f"BH must equal B * num_v_heads, got {B=}, {num_v_heads=}, {BH=}")

    # ========== LNC Sharding ==========
    # (heads are independent, so shard the BH axis across the SPMD grid)
    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0)
    BH_local = BH // n_prgs
    h_start = prg_id * BH_local
    if prg_id == n_prgs - 1:
        BH_local = BH - h_start  # last core takes the remainder

    out = nl.ndarray((BH, V), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    state_out = nl.ndarray((BH, K, V), dtype=nl.float32, buffer=nl.shared_hbm)
    scale = 1.0 / math.sqrt(K)

    # ========== Pre-allocated Buffers ==========
    # (sized to this core's head subset)
    S_all = nl.ndarray((K, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)
    S_bf_all = nl.ndarray((K, BH_local * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    Q_p = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    K_p = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    v_all = nl.ndarray((1, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)

    exp_g_K = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    beta_K = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)

    out_sb = nl.ndarray((1, BH_local * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    # ========== Bulk Preloads + Folded Pre-ops ==========
    # (each core loads only its head slice, offset by h_start)
    nisa.dma_copy(dst=Q_p, src=q.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))
    nisa.dma_copy(dst=K_p, src=k.ap(pattern=[[1, K], [K, BH_local]], offset=h_start * K))

    nisa.dma_copy(dst=v_all, src=v.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V])

    # Fold 4 inputs: gate z [BH, V] (per-head slice on free axis) + norm_weight [V].
    z_all = nl.ndarray((1, BH_local * V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=z_all, src=z.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V])
    nw_line = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_line, src=norm_weight.reshape((1, V)))

    k_f_all = nl.ndarray((1, BH_local * K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_f_all, src=k.reshape((1, BH * K))[:, h_start * K : h_start * K + BH_local * K])

    # g (raw a), beta (raw b): broadcast via partition 0 + stream_shuffle
    g_line = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    b_line = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(g_line, 0.0)
    nisa.memset(b_line, 0.0)
    nisa.dma_copy(dst=g_line[0:1, :], src=a.reshape((1, BH))[:, h_start : h_start + BH_local])
    nisa.dma_copy(dst=b_line[0:1, :], src=b.reshape((1, BH))[:, h_start : h_start + BH_local])

    """
    Fold 2: g = -exp(A_log) * softplus(a + dt_bias)  [per-v-head A_log, dt_bias]

    Head layout is BH = B * num_v_heads with the v-head index cycling fastest, so
    tiling the [num_v_heads] vectors B times gives the per-global-head value. Build
    full [1, BH] tiled lines then slice this core's contiguous head range.
    """
    alog_full = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
    dt_full = nl.ndarray((1, BH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=alog_full, src=A_log.ap(pattern=[[0, B], [1, num_v_heads]], offset=0))
    nisa.dma_copy(dst=dt_full, src=dt_bias.ap(pattern=[[0, B], [1, num_v_heads]], offset=0))
    alog_h = alog_full[0:1, nl.ds(h_start, BH_local)]
    dt_h = dt_full[0:1, nl.ds(h_start, BH_local)]

    gtmp = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    expA = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    # gtmp = softplus(a + dt_bias)
    nisa.tensor_tensor(dst=gtmp, data1=g_line[0:1, :], data2=dt_h, op=nl.add)
    nisa.activation(dst=gtmp, data=gtmp, op=nl.softplus)
    # expA = exp(A_log)
    nisa.activation(dst=expA, data=alog_h, op=nl.exp)
    # g = (softplus * -1) * exp(A_log)  -> stored back into g_line partition 0 (log-space)
    nisa.scalar_tensor_tensor(
        dst=g_line[0:1, :], data=gtmp, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=expA
    )

    stream_shuffle_broadcast(src=g_line, dst=g_line)
    stream_shuffle_broadcast(src=b_line, dst=b_line)

    # Kernel's existing exp path consumes log-space g.
    nisa.activation(dst=exp_g_K, data=g_line, op=nl.exp)
    # Fold 1: beta = sigmoid(b) fused (replaces plain copy of pre-sigmoid beta).
    nisa.activation(dst=beta_K, data=b_line, op=nl.sigmoid)

    # Fold 3: l2norm(q), l2norm(k) over K (partition axis), eps=1e-6.
    # Sum of squares over the partition (K) axis via matmul-with-ones (per-head/column).
    ones_K = nl.ndarray((K, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones_K, 1.0)
    q_sq = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    k_sq = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=q_sq, data=Q_p, op=nl.square)
    nisa.activation(dst=k_sq, data=K_p, op=nl.square)
    q_ss = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.psum)
    k_ss = nl.ndarray((1, BH_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(q_ss, ones_K, q_sq)
    nisa.nc_matmul(k_ss, ones_K, k_sq)
    # scale = rsqrt(sum(x^2) + eps) at partition 0, then broadcast over K partitions.
    q_scale = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    k_scale = nl.ndarray((K, BH_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=q_scale[0:1, :], data=q_ss, op=nl.rsqrt, bias=1e-6)
    nisa.activation(dst=k_scale[0:1, :], data=k_ss, op=nl.rsqrt, bias=1e-6)
    stream_shuffle_broadcast(src=q_scale, dst=q_scale)
    stream_shuffle_broadcast(src=k_scale, dst=k_scale)

    # Normalize K_p in place: K_p *= k_scale (per-column, over K partitions).
    nisa.tensor_tensor(dst=K_p, data1=K_p, data2=k_scale, op=nl.multiply)

    # Pre-scale Q: Q_scaled = Q_p * (1/sqrt(K)) * q_scale (l2norm composed with Q-scale).
    Q_scaled = nl.ndarray((K, BH_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=Q_scaled, data=Q_p, op0=nl.multiply, operand0=scale, op1=nl.multiply, operand1=q_scale
    )

    # Bulk load this core's states in one transposed DMA
    nisa.dma_copy(
        dst=S_all,
        src=state_in.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
    )

    # ========== Per-head Recurrence + Folded RMSNormGated ==========
    # (Python-unrolled so the compiler pipelines independent head streams)
    for head_idx in range(BH_local):
        S = S_all[:, nl.ds(head_idx * V, V)]
        S_bf = S_bf_all[:, nl.ds(head_idx * V, V)]
        k_p = K_p[:, head_idx : head_idx + 1]
        q_p = Q_scaled[:, head_idx : head_idx + 1]
        v_f = v_all[:, nl.ds(head_idx * V, V)]
        k_f = k_f_all[:, nl.ds(head_idx * K, K)]
        g_K_h = exp_g_K[:, head_idx : head_idx + 1]
        b_K_h = beta_K[:, head_idx : head_idx + 1]

        # Fold 3: normalize k_f (partition-1 layout of k) by this head's l2 scale.
        # k_scale[0, head_idx] is the per-head rsqrt(sum(k^2)+eps); broadcast scalar over free K.
        nisa.tensor_scalar(k_f, k_f, op0=nl.multiply, operand0=k_scale[0:1, head_idx : head_idx + 1])

        # Step 1: S *= exp(g)
        nisa.tensor_scalar(S, S, op0=nl.multiply, operand0=g_K_h)

        # Cast S to bf16
        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)

        # Step 2: kv_mem = k^T @ S
        kv_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(kv_p, k_p, S_bf)

        # Step 3: delta = (v - kv_mem) * beta
        diff = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_f, data2=kv_p, op=nl.subtract)
        delta = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(delta, diff, op0=nl.multiply, operand0=b_K_h[0:1, 0:1])

        # Step 4: S += outer(k, delta)
        delta_bf = nl.ndarray((1, V), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_bf, src=delta, engine=nisa.scalar_engine)
        outer_p = nl.ndarray((K, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(outer_p, k_f, delta_bf)
        nisa.tensor_tensor(dst=S, data1=S, data2=outer_p, op=nl.add)

        # Step 5: out = q_scaled^T @ S (refresh S_bf)
        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
        out_p = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(out_p, q_p, S_bf)

        # Fold 4: RMSNormGated. out = (o*rsqrt(mean_V(o^2)+eps)) * norm_weight * silu(z).
        # P=1 here (one head), V on free axis => reduce over free, all broadcasts free-axis.
        o_sb = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=out_p, engine=nisa.scalar_engine)
        ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        sq = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(sq, op=nl.square, data=o_sb, reduce_op=nl.add, reduce_res=ssum)
        # ssum = rsqrt(mean(o^2)+eps) = rsqrt(sum/V + eps)
        nisa.activation(dst=ssum, data=ssum, op=nl.rsqrt, scale=1.0 / V, bias=1e-6)
        # xf = (o * ssum) * norm_weight   (ssum free-bcast scalar, norm_weight elementwise on free)
        xf = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=xf,
            data=o_sb,
            op0=nl.multiply,
            operand0=ssum[0:1, 0:1],
            op1=nl.multiply,
            operand1=nw_line,
        )
        # silu(z) then out = xf * silu(z), cast to bf16 into out_sb slice.
        silu_z = nl.ndarray((1, V), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=silu_z, data=z_all[:, nl.ds(head_idx * V, V)], op=nl.silu)
        nisa.tensor_tensor(dst=out_sb[:, nl.ds(head_idx * V, V)], data1=xf, data2=silu_z, op=nl.multiply)

    # ========== Bulk Stores ==========
    # (each core writes ONLY its own head slice at offset h_start)
    nisa.dma_copy(dst=out.reshape((1, BH * V))[:, h_start * V : h_start * V + BH_local * V], src=out_sb)
    nisa.dma_copy(
        dst=state_out.ap(pattern=[[V, K], [K * V, BH_local], [1, V]], offset=h_start * K * V),
        src=S_all,
    )

    return out, state_out
