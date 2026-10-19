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
"""KDA chunked prefill --- gated delta rule with PER-CHANNEL log-decay.

Kimi Delta Attention. Same chunked delta rule as
``nkilib.experimental.gdn.gdn_cte``, but KDA's decay is a vector per token
(``g[t, d]``) rather than a scalar (``gate[t]``), which changes the algorithm
rather than just the shapes. See ``kda_cte_torch.py`` for the derivation, the
overflow analysis, and the reference this kernel is validated against.

Two consequences worth stating up front, because they run opposite to intuition:

1. **The per-channel version needs no ``[CHUNK, CHUNK]`` decay tile.** In the
   scalar kernel the pairwise factor ``exp(cg[i] - cg[j])`` is a matrix that gets
   built (cumsum, broadcast, transpose, subtract, clamp, exp) and multiplied into
   ``k k^T``. Per-channel, that factor cannot be applied after a contraction over
   ``d``, so it is folded into the operands instead --- and then the matrix, and all
   six instructions that built it, disappear.

2. **The folding overflows unless it is anchored.** ``cg`` is a cumulative sum of
   values in ``(lower_bound, 0)``, so the naive ``k * exp(-cg)`` operand reaches
   ``exp(320)`` over a 64-token chunk at Kimi's ``lower_bound = -5``: infinity in
   fp32 and bf16 alike. Each column block instead anchors on the cumulative decay
   just before its own sub-block, which bounds that operand at
   ``exp(_SUBBLK * |lower_bound|)``. ``_SUBBLK`` is therefore a *numerical*
   parameter; ``kda_cte_torch.assert_anchor_bound_safe`` is the check, and it is a
   function of the config, so it fails closed on the host.

Layout: tokens on the partition axis for everything that contracts over tokens
(the solve, the state update, the output), channels on the partition axis for
everything that contracts over channels (the operator ``A``, the intra-chunk
scores) *and* for the gate. Putting the gate channel-major is what lets the
cumulative sum be a single ``tensor_tensor_scan`` along the free axis in fp32, and
lets every anchor be a ``[D, 1]`` per-partition scalar rather than a partition
broadcast.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert

# Tokens per state update.
_CHUNK = 64
# Anchor spacing. NUMERICAL, not just tiling: the anchored operand is bounded by
# exp(_SUBBLK * |gate_lower_bound|), so halving this halves the exponent range.
_SUBBLK = 16
_N_SUBBLK = _CHUNK // _SUBBLK
# (I - A) is unit lower triangular with A strictly lower, hence nilpotent, so its inverse
# is a finite sum. ⚠️ It must NOT be formed by doubling over the whole chunk: on real
# activations (L2-normalised, correlated keys, beta -> 1) the powers A^2, A^4, ... A^32
# have huge entries with catastrophic cancellation. Measured on captured layer inputs,
# whole-chunk doubling was 346289% off at token 63 in bf16 and still 7.3% in fp32, while
# the exact solve is 0.000%. The kernel therefore inverts only the _SUBBLK x _SUBBLK
# diagonal blocks by doubling (A_bb ** _SUBBLK == 0 -> log2(_SUBBLK) rounds; their
# inverses are moderate) and does block forward substitution for the rest, all in fp32:
# X = D (I + A_off X) with D = blockdiag((I - A_bb)^-1), exact after _N_SUBBLK - 1
# iterations because A_off is nilpotent in block terms. Same structure as FLA's
# chunk_kda sub-chunking. 0.000% in fp32 and 0.41% with bf16 storage on the same inputs.
_N_DIAG_ROUNDS = math.ceil(math.log2(_SUBBLK))


@nki.jit
def kda_cte(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    beta: nl.NkiTensor,
    g: nl.NkiTensor,
    state_in: nl.NkiTensor,
    scale: float = 1.0,
):
    """Chunked KDA prefill with per-channel decay.

    Args:
        q, k, v: ``[B, S, H, D]`` --- the layout the model already has after
            ``view(batch, seq, num_heads, head_dim)``, so no transpose is needed on
            the torch side. ``D <= 128``; ``S`` must be a multiple of ``_CHUNK`` (64).
            Per (batch, head) the kernel gathers ``[_CHUNK, D]`` tiles with a strided
            DMA whose innermost run is ``D`` contiguous elements. ``q`` and ``k`` must ALREADY
            be L2-normalised per token --- that is what bounds ``|k_i . k_j| <= 1``
            and keeps ``(I - A)`` well conditioned. Feeding unnormalised keys makes
            the triangular inverse grow like ``|k.k| ** 63`` and overflow, which
            looks exactly like a chunking bug.
        beta: ``[B, S, H]`` fp32, already sigmoid-gated.
        g: ``[B, S, H, D]`` fp32 per-channel log-decay, non-positive. Pass fp32, not
            bf16: ``g`` is exponentiated after a 64-term sum, so bf16's 8-bit
            mantissa becomes a multiplicative error on the decay.
        state_in: ``[B, H, D, D]`` fp32 incoming recurrent state.
        scale: query scale, normally ``D ** -0.5``.

    Returns:
        ``(out, state_out)`` with ``out: [B, S, H, D]`` in ``q.dtype`` and
        ``state_out: [B, H, D, D]`` fp32.

    Notes:
        - Heads are independent, so the head axis carries no cross-head reduction
          and is safe to shard across the SPMD grid. LNC1 is what Kimi-K3 runs.
        - ``q``/``k`` L2-norm, ``beta``'s sigmoid and the log-gate formula stay
          outside: they are a handful of elementwise ops on ``[S, H*D]`` tensors and
          folding them in would couple this kernel to KDA's gate branch.
    """
    B, S, H, D = q.shape
    kernel_assert(S % _CHUNK == 0, f"kda_cte requires S % {_CHUNK} == 0, got S={S}")
    kernel_assert(D <= 128, f"kda_cte requires D <= 128 (partition axis), got D={D}")
    num_chunks = S // _CHUNK
    dtype = q.dtype
    # Strides of the [S, H, D] slab held by one batch entry.
    _TOK = H * D
    # (token, d) gather for one head: partition steps a whole token, free is the D
    # contiguous elements of that head.
    _HEAD_AP = [[_TOK, _CHUNK], [1, D]]

    out = nl.ndarray((B, S, H, D), dtype=dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((B, H, D, D), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- constants built once ----
    # data1 for the cumsum scan. tensor_tensor_scan computes
    #   r[i] = op1(op0(data0[i], r[i-1]), data1[i])
    # so op0=add with a zero data1 and op1=add is exactly an inclusive prefix sum,
    # accumulated in fp32 by the engine regardless of input dtype.
    scan_zeros = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(scan_zeros, 0.0)

    # Identity [_CHUNK, _CHUNK] in fp32: the start of the block doubling and the "mov"
    # operand that turns nc_matmul into a PE transpose (nc_matmul(X, I) = X^T @ I).
    ones_cc = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_cc, 1.0)
    ident_cc = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.affine_select(
        ident_cc,
        pattern=[[-1, _CHUNK]],
        offset=0,
        channel_multiplier=1,
        cmp_op=nl.equal,
        on_true_tile=ones_cc,
        on_false_value=0.0,
    )
    # Block-diagonal 0/1 mask (16x16 blocks) = B @ B^T with B[r, b] = 1 iff r // 16 == b.
    # floor(row/16) == floor(col/16) is not affine, so it cannot be one affine_select,
    # and a per-block memset is illegal on device ("Expected 'dst' partition start in
    # {0, 32, 64, 96} for a 16-partition access, but got 16"). B itself IS two affine
    # selects on a [_CHUNK, _N_SUBBLK] tile: r - 16 b >= 0 and 15 - r + 16 b >= 0.
    ones_cb = nl.ndarray((_CHUNK, _N_SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_cb, 1.0)
    blk_lo = nl.ndarray((_CHUNK, _N_SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.affine_select(
        blk_lo,
        pattern=[[-_SUBBLK, _N_SUBBLK]],
        offset=0,
        channel_multiplier=1,
        cmp_op=nl.greater_equal,
        on_true_tile=ones_cb,
        on_false_value=0.0,
    )
    blk_ind = nl.ndarray((_CHUNK, _N_SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.affine_select(
        blk_ind,
        pattern=[[_SUBBLK, _N_SUBBLK]],
        offset=_SUBBLK - 1,
        channel_multiplier=-1,
        cmp_op=nl.greater_equal,
        on_true_tile=blk_lo,
        on_false_value=0.0,
    )
    # B^T [_N_SUBBLK, _CHUNK] by PE transpose, then B @ B^T = nc_matmul(B^T, B^T).
    blk_ind_tp = nl.ndarray((_N_SUBBLK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(blk_ind_tp, blk_ind, ident_cc)
    blk_ind_t = nl.ndarray((_N_SUBBLK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=blk_ind_t, src=blk_ind_tp, engine=nisa.vector_engine)
    blkmask_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(blkmask_p, blk_ind_t, blk_ind_t)
    blkmask = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=blkmask, src=blkmask_p, engine=nisa.vector_engine)

    for b in nl.affine_range(B):
        for h in nl.affine_range(H):
            state = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=state, src=state_in[b, h])

            for chunk_idx in range(num_chunks):
                base = chunk_idx * _CHUNK
                head_off = base * _TOK + h * D

                # ================= loads =================
                # v is the only tensor wanted token-major; q, k and g are all consumed
                # channel-major, so they are loaded contiguously and transposed once.
                v_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=v_sb, src=v[b].ap(pattern=_HEAD_AP, offset=head_off))
                k_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k_sb, src=k[b].ap(pattern=_HEAD_AP, offset=head_off))
                q_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_sb, src=q[b].ap(pattern=_HEAD_AP, offset=head_off))
                g_sb = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=g_sb, src=g[b].ap(pattern=_HEAD_AP, offset=head_off))
                beta_sb = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=beta_sb,
                    src=beta[b].ap(pattern=[[H, _CHUNK], [1, 1]], offset=base * H + h),
                )

                k_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=k_t, src=k_sb)
                q_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=q_t, src=q_sb)
                g_t = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_transpose(dst=g_t, src=g_sb)

                # ============ cumulative gate ============
                # Channel-major, so the prefix sum runs along the FREE axis: one Vector
                # instruction, fp32 math, no matmul and no [CHUNK, CHUNK] tile.
                cg_t = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(
                    dst=cg_t,
                    data0=g_t,
                    data1=scan_zeros,
                    initial=0.0,
                    op0=nl.add,
                    op1=nl.add,
                )

                # ============ anchored operands ============
                # anchor[a] = cumulative decay just BEFORE sub-block a, i.e. cg at token
                # a*_SUBBLK - 1, with anchor[0] = 0. Channel-major makes each one a
                # [D, 1] free-axis view, directly usable as a per-partition operand.
                #
                # khat_a[d, i] = k[d, i] * exp(min(cg[d, i] - anchor[a, d], 0))
                # kbar[d, j]   = k[d, j] * exp(anchor[b(j), d] - cg[d, j])
                #
                # The min(., 0) clamp matters: rows before the anchor have a POSITIVE
                # exponent reaching +_CHUNK*|lower_bound|, i.e. inf, and they are then
                # masked away -- inf * 0 = NaN. The clamp only ever touches entries in
                # the strictly-upper triangle both masks discard, because a used entry
                # has i >= a*_SUBBLK and hence cg[i] <= anchor[a].
                khat = [None] * _N_SUBBLK
                qhat = [None] * _N_SUBBLK
                for a in range(_N_SUBBLK):
                    decay_a = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    if a == 0:
                        # anchor 0 is exactly zero, so cg is already the exponent and it
                        # is non-positive everywhere -- no subtract, no clamp.
                        nisa.activation(dst=decay_a, data=cg_t, op=nl.exp)
                    else:
                        anchor_a = cg_t[nl.ds(0, D), nl.ds(a * _SUBBLK - 1, 1)]
                        shifted = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar(
                            dst=shifted,
                            data=cg_t,
                            op0=nl.subtract,
                            operand0=anchor_a,
                            op1=nl.minimum,
                            operand1=0.0,
                        )
                        nisa.activation(dst=decay_a, data=shifted, op=nl.exp)
                    kh = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=kh, data1=k_t, data2=decay_a, op=nl.multiply)
                    qh = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=qh, data1=q_t, data2=decay_a, op=nl.multiply)
                    khat[a] = kh
                    qhat[a] = qh

                kbar = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                for a in range(_N_SUBBLK):
                    cols = nl.ds(a * _SUBBLK, _SUBBLK)
                    rise = nl.ndarray((D, _SUBBLK), dtype=nl.float32, buffer=nl.sbuf)
                    if a == 0:
                        # exp(0 - cg) = exp(-cg); activation's scale does the negation.
                        nisa.activation(dst=rise, data=cg_t[nl.ds(0, D), cols], op=nl.exp, scale=-1.0)
                    else:
                        # op(data*scale + bias) with scale=-1 and bias=anchor gives
                        # exp(anchor - cg) in one Scalar instruction.
                        nisa.activation(
                            dst=rise,
                            data=cg_t[nl.ds(0, D), cols],
                            op=nl.exp,
                            bias=cg_t[nl.ds(0, D), nl.ds(a * _SUBBLK - 1, 1)],
                            scale=-1.0,
                        )
                    nisa.tensor_tensor(
                        dst=kbar[nl.ds(0, D), cols],
                        data1=k_t[nl.ds(0, D), cols],
                        data2=rise,
                        op=nl.multiply,
                    )

                # ====== operator A and intra-chunk scores ======
                # Column block a contracts against the operands anchored at a, which is
                # what keeps every exponent representable. nc_matmul(stat, mov) contracts
                # the partition axis, so with channels on partition:
                #   nc_matmul(khat_a[D, C], kbar[:, cols_a][D, SB]) -> [C, SB]
                #   result[i, j'] = sum_d khat_a[d, i] * kbar[d, a*SB + j']
                # Four matmuls of N=_SUBBLK cover the same MACs as one N=_CHUNK matmul,
                # so anchoring costs instructions, not arithmetic.
                ahat = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                qkt = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                for a in range(_N_SUBBLK):
                    cols = nl.ds(a * _SUBBLK, _SUBBLK)
                    ap = nl.ndarray((_CHUNK, _SUBBLK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(ap, khat[a], kbar[nl.ds(0, D), cols])
                    nisa.tensor_copy(dst=ahat[nl.ds(0, _CHUNK), cols], src=ap)
                    qp = nl.ndarray((_CHUNK, _SUBBLK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(qp, qhat[a], kbar[nl.ds(0, D), cols])
                    nisa.tensor_copy(dst=qkt[nl.ds(0, _CHUNK), cols], src=qp)

                # A = -beta_i * ahat, strictly lower (read-before-write: token i sees
                # only j < i). beta is per-row, i.e. per-partition here.
                neg_beta = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=neg_beta, data=beta_sb, op0=nl.multiply, operand0=-1.0)
                a_full = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=a_full,
                    data=ahat,
                    op0=nl.multiply,
                    operand0=neg_beta,
                    engine=nisa.scalar_engine,
                )
                # fp32 from here to the solve: see the note at _N_DIAG_ROUNDS.
                a_masked = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.affine_select(
                    a_masked,
                    pattern=[[-1, _CHUNK]],
                    offset=-1,
                    channel_multiplier=1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=a_full,
                    on_false_value=0.0,
                )
                # Scores are lower-INCLUSIVE: token i does attend to itself.
                qkt_masked = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.affine_select(
                    qkt_masked,
                    pattern=[[-1, _CHUNK]],
                    offset=0,
                    channel_multiplier=1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=qkt,
                    on_false_value=0.0,
                )

                # ============== the solve ==============
                # M = (I - A)^-1 by block forward substitution (see _N_DIAG_ROUNDS).
                # nc_matmul(stat, mov) = stat^T @ mov, so nc_matmul(X, I) is X^T and
                # nc_matmul(MT, rhs) is M @ rhs. Everything here is [_CHUNK, _CHUNK] fp32.
                # (PE transposes are written out inline: the device NKI compiler rejects
                # direct calls to inner functions -- "inner functions can only be used as
                # fori_loop/while_loop body arguments" -- while the simulator accepts them.)
                # A = A_diag + A_off: the strictly-lower parts inside / outside the
                # 16x16 diagonal blocks.
                a_diag = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a_diag, data1=a_masked, data2=blkmask, op=nl.multiply)
                a_off = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a_off, data1=a_masked, data2=a_diag, op=nl.subtract)
                a_diag_t_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(a_diag_t_p, a_diag, ident_cc)
                a_diag_t = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=a_diag_t, src=a_diag_t_p, engine=nisa.vector_engine)
                a_off_t_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(a_off_t_p, a_off, ident_cc)
                a_off_t = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=a_off_t, src=a_off_t_p, engine=nisa.vector_engine)

                # D^T = ((I - A_diag)^-1)^T by doubling on the block-diagonal part only
                # (A_diag ** _SUBBLK == 0, so _N_DIAG_ROUNDS rounds are exact):
                #   DT <- I ; each round DT += (A_diag^(2^r))^T @ DT ; then square.
                dt = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=dt, src=ident_cc, engine=nisa.scalar_engine)
                pt_stat = a_diag  # A^(2^r), rows on partitions
                p_mat = a_diag_t  # its transpose
                for rd in range(_N_DIAG_ROUNDS):
                    pmt = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pmt, pt_stat, dt)
                    dt_new = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=dt_new, data1=dt, data2=pmt, op=nl.add)
                    dt = dt_new
                    if rd < _N_DIAG_ROUNDS - 1:
                        sq = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(sq, pt_stat, p_mat)
                        sq_t = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(sq_t, p_mat, pt_stat)
                        p_new = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=p_new, src=sq, engine=nisa.scalar_engine)
                        pt_new = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=pt_new, src=sq_t, engine=nisa.scalar_engine)
                        p_mat = p_new
                        pt_stat = pt_new

                # X_0 = D, then X <- D (I + A_off X); exact after _N_SUBBLK - 1 steps.
                #   A_off @ X  = nc_matmul(stat=A_off^T, mov=X)
                #   D @ Y      = nc_matmul(stat=D^T,     mov=Y)
                x_mat_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(x_mat_p, dt, ident_cc)
                x_mat = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x_mat, src=x_mat_p, engine=nisa.vector_engine)
                for _ in range(_N_SUBBLK - 1):
                    ax = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(ax, a_off_t, x_mat)
                    y_mat = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=y_mat, data1=ident_cc, data2=ax, op=nl.add)
                    dx = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dx, dt, y_mat)
                    x_new = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=x_new, src=dx, engine=nisa.vector_engine)
                    x_mat = x_new
                # MT = M^T is what nc_matmul(MT, rhs) = M @ rhs needs.
                mt_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(mt_p, x_mat, ident_cc)
                mt = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=mt, src=mt_p, engine=nisa.vector_engine)

                # rhs_w = beta * khat_0 (token-major), rhs_u = beta * v.
                khat0_tm = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=khat0_tm, src=khat[0])
                rhs_w = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=rhs_w,
                    data=khat0_tm,
                    op0=nl.multiply,
                    operand0=beta_sb,
                    engine=nisa.scalar_engine,
                )
                rhs_u = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=rhs_u,
                    data=v_sb,
                    op0=nl.multiply,
                    operand0=beta_sb,
                    engine=nisa.scalar_engine,
                )
                w_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(w_p, mt, rhs_w)
                u_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(u_p, mt, rhs_u)
                w_sb = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=w_sb, src=w_p, engine=nisa.vector_engine)

                # ========== output and state update ==========
                state_bf = nl.ndarray((D, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=state_bf, src=state, engine=nisa.vector_engine)

                # v_new = U - W @ state, against the state at the START of the chunk.
                w_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=w_t, src=w_sb)
                ws_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ws_p, w_t, state_bf)
                # tensor_tensor forbids both operands in PSUM, so evict U first.
                u_sb = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=u_sb, src=u_p, engine=nisa.vector_engine)
                v_new = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=v_new, data1=u_sb, data2=ws_p, op=nl.subtract)
                v_new_bf = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=v_new_bf, src=v_new, engine=nisa.vector_engine)

                # o_cross needs no transpose: qhat[0] is already channel-major, which is
                # exactly the layout the cross-chunk matmul wants.
                o_cross_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(o_cross_p, qhat[0], state_bf)
                qkt_t = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=qkt_t, src=qkt_masked)
                o_cross = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_cross, src=o_cross_p, engine=nisa.vector_engine)
                o_intra_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(o_intra_p, qkt_t, v_new_bf)
                o_sum = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o_sum, data1=o_cross, data2=o_intra_p, op=nl.add)
                o_out = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=o_out,
                    data=o_sum,
                    op0=nl.multiply,
                    operand0=scale,
                    engine=nisa.scalar_engine,
                )
                nisa.dma_copy(dst=out[b].ap(pattern=_HEAD_AP, offset=head_off), src=o_out)

                # state = state * exp(cg_last) + k_dec^T @ v_new, with
                # k_dec[j, d] = k[j, d] * exp(cg_last[d] - cg[j, d]) (exponent <= 0, so
                # it underflows to zero exactly where the true contribution vanishes).
                # cg_last is a [D, 1] free-axis view, i.e. a per-partition scalar for the
                # state decay -- the per-channel decay costs nothing here.
                cg_last = cg_t[nl.ds(0, D), nl.ds(_CHUNK - 1, 1)]
                fall = nl.ndarray((D, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=fall, data=cg_t, op=nl.exp, bias=cg_last, scale=-1.0)
                kdec_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=kdec_t, data1=k_t, data2=fall, op=nl.multiply)
                kdec_tm = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=kdec_tm, src=kdec_t)

                dec_last = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=dec_last, data=cg_last, op=nl.exp)
                state_dec = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=state_dec,
                    data=state,
                    op0=nl.multiply,
                    operand0=dec_last,
                    engine=nisa.scalar_engine,
                )
                su_p = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(su_p, kdec_tm, v_new_bf)
                state_new = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=state_new, data1=su_p, data2=state_dec, op=nl.add)
                nisa.tensor_copy(dst=state, src=state_new, engine=nisa.vector_engine)

            state_final = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=state_final, src=state, engine=nisa.vector_engine)
            nisa.dma_copy(dst=state_out[b, h], src=state_final)

    return out, state_out
