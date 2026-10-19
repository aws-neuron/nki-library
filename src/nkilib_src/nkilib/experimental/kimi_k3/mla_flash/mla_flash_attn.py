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
"""Flash attention for Kimi-K3 MLA over an expanded, blocked key/value cache.

The dense MLA kernels hold the whole key span on chip, so they stop at a few thousand
keys. This kernel streams the keys from HBM in 2048-key blocks with an online softmax, so
the span only bounds the cache, not the chip. Measured at a 128K span with 3 heads: a
decode step costs 2.2 ms on one core and 1.2 ms on the two cores of an LNC2 pair; a
256-query prefill chunk 4.0 / 2.2 ms.

Cache layout (``H`` heads, ``span`` a multiple of ``n_programs * 2048``):

* ``k_blk`` ``[H, span // 2048, qk, 2048]``: expanded keys, TRANSPOSED inside each block
  (feature-major, the PE moving-operand layout).
* ``marker`` ``[H, span]``: ``0`` for a written key, ``MARKER`` for an unwritten one. Added
  to every query's score through a ones-row matmul, so an unwritten key scores about
  ``MARKER * scale`` and gets probability 0: an UNMASKED stream over the whole span
  attends exactly the written prefix, with no mask tensor and no dynamic loop bound.
* ``v_blk`` ``[H, span // 2048, 128, 16, dv]``: key ``b*2048 + sub*128 + p`` at
  ``[h, b, p, sub, :]``, so a partition reads its 16 rows of a block as 4 KB contiguous.

See ``mla_flash_attn_torch.py`` for the fp32 reference with the same inputs.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
BLOCK = 2048  # keys per cache block (one DMA per operand per block)
KT = 512  # keys per softmax step (one PSUM bank of fp32 scores)
SUBS = BLOCK // P  # 128-key value rows per partition in a block
N_K_SUB = KT // P
MARKER = -3.0e4  # -29952 in bf16; times the score scale about -2000
_NEG_INF = -3.0e38


@nki.jit
def kimi_k3_mla_flash_attn(
    q_t: nl.NkiTensor,  # [H, qk, Sq] bf16, feature-major queries (nope | rope)
    k_blk: nl.NkiTensor,  # [H, n_blocks, qk, 2048] bf16, feature-major keys per block
    marker: nl.NkiTensor,  # [H, n_blocks * 2048] bf16, 0 written / MARKER unwritten
    v_blk: nl.NkiTensor,  # [H, n_blocks, 128, 16, dv] bf16, key b*2048 + sub*128 + p at [b, p, sub]
    kc_t: nl.NkiTensor,  # [H, qk, Sq] bf16, the call's own keys, feature-major
    mc: nl.NkiTensor,  # [n_programs, H, Sq] bf16, chunk markers per program
    vc: nl.NkiTensor,  # [H, Sq, dv] bf16, the call's own values
    scale: float,
):
    """Flash attention for Kimi-K3 MLA over an expanded, blocked key/value cache.

    One call attends the cache prefix (every block; the marker hides unwritten keys) and
    the call's own keys (causal by chunk-local index) in one online softmax, in 512-key
    steps. ``Sq`` is a multiple of 128 and at most 512. Under a 1-D SPMD grid program
    ``p`` takes blocks ``p, p + n, ...`` and the chunk keys with marker row ``mc[p]``;
    each program returns its unnormalized partial state and the caller merges them.

    Returns:
        ``(acc, m, l)``: ``[n_programs, H, Sq, dv]`` fp32 unnormalized output,
        ``[n_programs, H, Sq]`` fp32 running max and running sum. ``out = acc / l`` for
        one program; for several, weight each by ``exp(m_p - max_p m_p)`` first.
    """
    H, Dq, Sq = q_t.shape
    n_blocks = k_blk.shape[1]
    Dv = v_blk.shape[4]
    n_q_tiles = Sq // P
    steps_per_block = BLOCK // KT
    two_tiles = Dq > P
    hi_rows = Dq - P if two_tiles else 0  # 64 at Kimi dims (rope)
    lo_rows = P if two_tiles else Dq
    # SPMD over the LNC programs: program p takes blocks p, p + n, p + 2n, ... and
    # the chunk keys with its own marker row (the caller invalidates the chunk for
    # every program but the first). Each program returns its unnormalized partial
    # state; the caller merges the programs (an online-softmax step in torch).
    if nl.program_ndim() == 0:
        n_prog, pid = 1, 0
    else:
        n_prog, pid = nl.num_programs(0), nl.program_id(0)
    n_prog_static = mc.shape[0]
    acc_out = nl.ndarray((n_prog_static, H, Sq, Dv), dtype=nl.float32, buffer=nl.shared_hbm)
    m_out = nl.ndarray((n_prog_static, H, Sq), dtype=nl.float32, buffer=nl.shared_hbm)
    l_out = nl.ndarray((n_prog_static, H, Sq), dtype=nl.float32, buffer=nl.shared_hbm)
    ones_row = nl.ndarray((1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    for h in range(H):
        q_lo = nl.ndarray((P, Sq), dtype=nl.bfloat16, buffer=nl.sbuf)
        q_hi = nl.ndarray((P, Sq), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_lo[0:lo_rows, 0:Sq], src=q_t[h, 0:lo_rows, 0:Sq])
        if two_tiles:
            nisa.dma_copy(dst=q_hi[0:hi_rows, 0:Sq], src=q_t[h, P:Dq, 0:Sq])
        m_run = nl.ndarray((P, n_q_tiles), dtype=nl.float32, buffer=nl.sbuf)
        l_run = nl.ndarray((P, n_q_tiles), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P, n_q_tiles, Dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=m_run, value=_NEG_INF)
        nisa.memset(dst=l_run, value=0.0)
        nisa.memset(dst=acc, value=0.0)

        def attend_tile(k_lo, k_hi, mk_row, v_tiles, width, causal):
            """One key tile against every query tile. ``k_lo`` [P, width], ``k_hi``
            [hi_rows, width], ``mk_row`` [1, width] marker, ``v_tiles`` [P, n_sub, Dv]
            with ``n_sub * P == width``. ``causal`` adds the chunk-local mask."""
            n_sub = width // P
            for qt in range(n_q_tiles):
                qs = nl.ds(qt * P, P)
                scores_ps = nl.ndarray((P, KT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=scores_ps[0:P, 0:width],
                    stationary=q_lo[0:lo_rows, qs],
                    moving=k_lo,
                    accumulate=False,
                )
                if two_tiles:
                    nisa.nc_matmul(
                        dst=scores_ps[0:P, 0:width],
                        stationary=q_hi[0:hi_rows, qs],
                        moving=k_hi,
                        accumulate=True,
                    )
                # + marker[j] on every query row: a rank-1 matmul with a ones row.
                nisa.nc_matmul(
                    dst=scores_ps[0:P, 0:width],
                    stationary=ones_row[0:1, 0:P],
                    moving=mk_row[0:1, 0:width],
                    accumulate=True,
                )
                scores = nl.ndarray((P, KT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=scores[0:P, 0:width],
                    data=scores_ps[0:P, 0:width],
                    op0=nl.multiply,
                    operand0=scale,
                )
                if causal:
                    # j - i - qt*P <= 0 keeps the key. Static per query tile.
                    diff = nl.ndarray((P, KT), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.iota(
                        dst=diff[0:P, 0:width],
                        pattern=[[1, width]],
                        offset=-qt * P,
                        channel_multiplier=-1,
                    )
                    diff_f = nl.ndarray((P, KT), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=diff_f[0:P, 0:width], src=diff[0:P, 0:width])
                    keep_f = nl.ndarray((P, KT), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=keep_f[0:P, 0:width],
                        data=diff_f[0:P, 0:width],
                        op0=nl.less,
                        operand0=0.5,
                    )
                    keep = nl.ndarray((P, KT), dtype=nl.int8, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=keep[0:P, 0:width], src=keep_f[0:P, 0:width])
                    nisa.select_reduce(
                        dst=scores[0:P, 0:width],
                        predicate=keep[0:P, 0:width],
                        on_true=scores[0:P, 0:width],
                        on_false=_NEG_INF,
                    )
                tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=tile_max[0:P, 0:1],
                    op=nl.maximum,
                    data=scores[0:P, 0:width],
                    axis=1,
                )
                m_new = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=m_new[0:P, 0:1],
                    data1=m_run[0:P, qt : qt + 1],
                    data2=tile_max[0:P, 0:1],
                    op=nl.maximum,
                )
                neg_m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=neg_m[0:P, 0:1], data=m_new[0:P, 0:1], op0=nl.multiply, operand0=-1.0)
                alpha = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=alpha[0:P, 0:1],
                    op=nl.exp,
                    data=m_run[0:P, qt : qt + 1],
                    bias=neg_m[0:P, 0:1],
                )
                p = nl.ndarray((P, KT), dtype=nl.bfloat16, buffer=nl.sbuf)
                p_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=p[0:P, 0:width],
                    op=nl.exp,
                    data=scores[0:P, 0:width],
                    bias=neg_m[0:P, 0:1],
                    reduce_op=nl.add,
                    reduce_res=p_sum[0:P, 0:1],
                )
                nisa.scalar_tensor_tensor(
                    dst=l_run[0:P, qt : qt + 1],
                    data=l_run[0:P, qt : qt + 1],
                    op0=nl.multiply,
                    operand0=alpha[0:P, 0:1],
                    op1=nl.add,
                    operand1=p_sum[0:P, 0:1],
                )
                pv_ps = nl.ndarray((P, Dv), dtype=nl.float32, buffer=nl.psum)
                for sub in range(n_sub):
                    pT_ps = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=pT_ps[0:P, 0:P], data=p[0:P, sub * P : (sub + 1) * P])
                    pT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pT[0:P, 0:P], src=pT_ps[0:P, 0:P])
                    nisa.nc_matmul(
                        dst=pv_ps[0:P, 0:Dv],
                        stationary=pT[0:P, 0:P],
                        moving=v_tiles[0:P, sub, 0:Dv],
                        accumulate=sub > 0,
                    )
                nisa.scalar_tensor_tensor(
                    dst=acc[0:P, qt, 0:Dv],
                    data=acc[0:P, qt, 0:Dv],
                    op0=nl.multiply,
                    operand0=alpha[0:P, 0:1],
                    op1=nl.add,
                    operand1=pv_ps[0:P, 0:Dv],
                )
                nisa.tensor_copy(dst=m_run[0:P, qt : qt + 1], src=m_new[0:P, 0:1])

        # Cache prefix, one 2048-key block per iteration: four DMAs (K lo, K hi, V,
        # marker), then four 512-key softmax steps. The marker hides unwritten keys.
        for kb in nl.sequential_range(n_blocks // n_prog_static):
            blk = kb * n_prog + pid
            k_lo = nl.ndarray((P, BLOCK), dtype=nl.bfloat16, buffer=nl.sbuf)
            k_hi = nl.ndarray((P, BLOCK), dtype=nl.bfloat16, buffer=nl.sbuf)
            v_sb = nl.ndarray((P, SUBS, Dv), dtype=nl.bfloat16, buffer=nl.sbuf)
            mk_row = nl.ndarray((1, BLOCK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_lo[0:lo_rows, 0:BLOCK], src=k_blk[h, blk, 0:lo_rows, 0:BLOCK])
            if two_tiles:
                nisa.dma_copy(dst=k_hi[0:hi_rows, 0:BLOCK], src=k_blk[h, blk, P:Dq, 0:BLOCK])
            nisa.dma_copy(dst=v_sb[0:P, 0:SUBS, 0:Dv], src=v_blk[h, blk, 0:P, 0:SUBS, 0:Dv])
            nisa.dma_copy(dst=mk_row[0:1, 0:BLOCK], src=marker[h : h + 1, nl.ds(blk * BLOCK, BLOCK)])
            for t in range(steps_per_block):
                ks = nl.ds(t * KT, KT)
                attend_tile(
                    k_lo[0:lo_rows, ks],
                    k_hi[0:hi_rows, ks] if two_tiles else None,
                    mk_row[0:1, ks],
                    v_sb[0:P, nl.ds(t * N_K_SUB, N_K_SUB), 0:Dv],
                    KT,
                    False,
                )

        # The call's own keys, causal by chunk index (the chunk is at most one tile).
        kc_lo = nl.ndarray((P, KT), dtype=nl.bfloat16, buffer=nl.sbuf)
        kc_hi = nl.ndarray((P, KT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=kc_lo[0:lo_rows, 0:Sq], src=kc_t[h, 0:lo_rows, 0:Sq])
        if two_tiles:
            nisa.dma_copy(dst=kc_hi[0:hi_rows, 0:Sq], src=kc_t[h, P:Dq, 0:Sq])
        mc_row = nl.ndarray((1, KT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=mc_row[0:1, 0:Sq], src=mc[pid, h : h + 1, 0:Sq])
        vc_tiles = nl.ndarray((P, n_q_tiles, Dv), dtype=nl.bfloat16, buffer=nl.sbuf)
        for sub in range(n_q_tiles):
            nisa.dma_copy(dst=vc_tiles[0:P, sub, 0:Dv], src=vc[h, sub * P : (sub + 1) * P, 0:Dv])
        attend_tile(kc_lo[0:lo_rows, 0:Sq], kc_hi[0:hi_rows, 0:Sq] if two_tiles else None, mc_row, vc_tiles, Sq, True)

        for qt in range(n_q_tiles):
            qs = nl.ds(qt * P, P)
            nisa.dma_copy(dst=acc_out[pid, h, qs, 0:Dv], src=acc[0:P, qt, 0:Dv])
            nisa.dma_copy(dst=m_out[pid, h, qs].reshape((P, 1)), src=m_run[0:P, qt : qt + 1])
            nisa.dma_copy(dst=l_out[pid, h, qs].reshape((P, 1)), src=l_run[0:P, qt : qt + 1])
    return acc_out, m_out, l_out
