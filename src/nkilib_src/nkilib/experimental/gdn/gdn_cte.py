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
GDN Chunked Prefill Kernel (chunked gated delta-rule).

Same math as FLA/CUDA chunk_gated_delta_rule, but the intra-chunk triangular solve
uses a different numerical method suited to the Neuron tensor engine (see below).

Algorithm (per chunk of CHUNK tokens):
  1. Intra-chunk: solve (I - A) W = rhs_w and (I - A) U = rhs_u for W, U, where A is the
     strictly-lower-triangular chunk interaction matrix. The chunk is tiled into
     N_SUBBLK sub-blocks of SUBBLK rows:
       - INTER-block (bj < bi): block forward-substitution (subtract solved blocks).
       - INTRA-block (diagonal A_ii): RECURSIVE DOUBLING. A_ii is SUBBLK x SUBBLK
         strictly-lower-triangular => nilpotent (A_ii^SUBBLK = 0), so the solve is exact:
             (I - A_ii)^-1 rhs = sum_{j=0}^{SUBBLK-1} A_ii^j rhs.
         Doubling squares the operator each round, covering all SUBBLK terms in
         log2(SUBBLK) rounds (4 for SUBBLK=16) instead of the SUBBLK-1 rounds a linear
         fixed-point (one power of A per round) would need.
  2. Output: o = q @ h * exp(cg) * scale + qkt_masked @ v_new * scale
     where v_new = U - W @ h, h = state at START of this chunk
  3. State update: h = h * exp(g_last) + k_dec^T @ v_new

CHUNK=64, SUBBLK=16, N_SUBBLK=4 sub-blocks per chunk.
State [D,D] stays in f32 SBUF across all chunks.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

_CHUNK = 64
_SUBBLK = 16
_N_SUBBLK = _CHUNK // _SUBBLK
# Recursive-doubling rounds for the nilpotent SUBBLK x SUBBLK block solve: each round
# doubles the number of covered A^j terms, so ceil(log2(SUBBLK)) rounds cover all SUBBLK.
_N_ROUNDS = math.ceil(math.log2(_SUBBLK))

# Chunk-group batching: process up to _CHUNKS_PER_GROUP chunks together so their per-sub-block
# Phase-1 work packs onto the 128-partition axis (row-split) -> merged [.,D] vector ops (the
# VectorE bottleneck) and better PE-row use. Sub-blocks are _SUBBLK=16 rows and chunks stay
# independent (block-diagonal operator per group), so grouping adds NO matmuls; Phase 2/3
# (state carry) stays serial per chunk. Distinct from B_batch (the model's batch dim).
_CHUNKS_PER_GROUP = 8


@nki.jit
def gdn_cte(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    beta: nl.NkiTensor,
    gate: nl.NkiTensor,
    scale: float = 1.0,
):
    """
    GDN chunked prefill (chunked gated delta-rule). Same math as FLA/CUDA
    chunk_gated_delta_rule; the intra-chunk triangular solve uses a
    recursive-doubling method suited to the Neuron tensor engine (see module docstring).

    Chunks of _CHUNK=64 tokens are processed in groups of up to _CHUNKS_PER_GROUP=8
    (their per-sub-block Phase-1 work is batched onto the 128-partition axis via a
    block-diagonal operator, so chunks stay independent -- leak-free). The final group
    may be PARTIAL: if num_chunks is not a multiple of _CHUNKS_PER_GROUP, the last group
    processes only the remaining chunks (num_chunks % _CHUNKS_PER_GROUP). Therefore S
    need only be a multiple of _CHUNK (=64), not _CHUNKS_PER_GROUP*_CHUNK (=512).

    Args:
        q, k, v: [B, S, D] input tensors. S must be a multiple of _CHUNK (=64).
        beta: [B, S] per-token update strength (sigmoid-gated).
        gate: [B, S] per-token log-decay (negative). gate=0 means no decay.
        scale: query scaling factor (typically 1/sqrt(D)).

    Returns:
        result: [B, S, D] output.
        state_output: [B, D, D] final recurrent state (float32).
    """
    B_batch, S, D = q.shape
    # Fatal shape check: S must be a multiple of _CHUNK (partial final group handles the
    # sub-_CHUNKS_PER_GROUP remainder). kernel_assert is the NKI-safe assert (NKI rejects `raise`).
    kernel_assert(S % _CHUNK == 0, f"gdn_cte requires S to be a multiple of _CHUNK={_CHUNK}, got S={S}")
    num_chunks = S // _CHUNK
    # Groups of up to _CHUNKS_PER_GROUP chunks; the final group holds the remainder so
    # trailing chunks (S not a multiple of _CHUNKS_PER_GROUP*_CHUNK) are NOT dropped.
    n_full_groups = num_chunks // _CHUNKS_PER_GROUP
    rem_chunks = num_chunks % _CHUNKS_PER_GROUP
    group_sizes = [_CHUNKS_PER_GROUP] * n_full_groups + ([rem_chunks] if rem_chunks else [])
    dtype = q.dtype

    result = nl.ndarray(shape=(B_batch, S, D), dtype=dtype, buffer=nl.shared_hbm)
    state_output = nl.ndarray(shape=(B_batch, D, D), dtype=nl.float32, buffer=nl.shared_hbm)

    # Cumsum operator (built ONCE, reused every chunk/group): U[k,i] = 1 if k<=i else 0.
    # cg = U^T-style matmul: cg[i] = sum_k U[k,i]*gate[k] = sum_{k<=i} gate[k] (inclusive prefix
    # sum). nc_matmul contracts the partition axis, so feeding gate on partition gives cg on
    # partition ([_CHUNK,1] column) directly -- the layout exp_cg/decay want (kills the old
    # per-chunk scan + dma_transpose). U is 0/1 => bf16-exact; the sum accumulates in fp32 PSUM.
    U_ones = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(U_ones, 1.0)
    U_cumsum = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
    # keep where partition<=free: -1*p + 1*f + 0 >= 0  <=>  f >= p
    nisa.affine_select(
        U_cumsum,
        pattern=[[1, _CHUNK]],
        offset=0,
        channel_multiplier=-1,
        cmp_op=nl.greater_equal,
        on_true_tile=U_ones,
        on_false_value=0.0,
    )

    # Block-diagonal identity [_GROUP_ROWS,_GROUP_ROWS] for the shared-M solve (built once).
    # Diagonal 1s where partition==free: -1*p + 1*f == 0. Off-diag 0. The doubling accumulates
    # M = (I-A)^-1 = I + A + A^2 + ... starting from THIS identity, so W=M@rhs_w and U=M@rhs_u
    # share one operator (doubling runs once, not once-per-rhs).
    _GR = _CHUNKS_PER_GROUP * _SUBBLK
    _ones_gr = nl.ndarray((_GR, _GR), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(_ones_gr, 1.0)
    Ident_gr = nl.ndarray((_GR, _GR), dtype=dtype, buffer=nl.sbuf)
    nisa.affine_select(
        Ident_gr,
        pattern=[[-1, _GR]],
        offset=0,
        channel_multiplier=1,
        cmp_op=nl.equal,
        on_true_tile=_ones_gr,
        on_false_value=0.0,
    )

    for batch_id in nl.affine_range(B_batch):
        state = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(state, 0.0)

        for group_idx in range(len(group_sizes)):
            cpg = group_sizes[group_idx]
            # cpg = number of chunks in THIS group: _CHUNKS_PER_GROUP for full groups, or the
            # remainder for the final partial group. All per-group buffers/APs/row-counts use cpg
            # (not _CHUNKS_PER_GROUP) so the partial group loads/solves/carries only its real chunks.
            # ---- Pass 1a: per-chunk setup (cg, decay, A_masked); store for batched Phase 1 ----
            group_state = [None] * cpg
            _GROUP_ROWS = cpg * _SUBBLK
            # Block-diag identity for this group's solve: top-left cpg*_SUBBLK slice of Ident_gr.
            Ident_g = Ident_gr[nl.ds(0, _GROUP_ROWS), nl.ds(0, _GROUP_ROWS)]
            # Group-contiguous load: ONE strided-AP DMA per tensor pulls the whole group's tokens
            # into [_CHUNK, cpg*D] with chunks on the FREE axis (partition=row stride D,
            # free = chunk stride _CHUNK*D then d stride 1). Chunk c = free-view [:, c*D:(c+1)*D].
            # Replaces the per-chunk [_CHUNK,D] loads per tensor. Device-verified
            # (toy_grouploadview: layout, view-as-operand, batch-indexed .ap all OK).
            group_base = group_idx * _CHUNKS_PER_GROUP * _CHUNK  # token offset within the batch
            _NCD = cpg * D
            _GLOAD_AP = [[D, _CHUNK], [_CHUNK * D, cpg], [1, D]]
            kg = nl.ndarray((_CHUNK, _NCD), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=kg, src=k[batch_id].ap(pattern=_GLOAD_AP, offset=group_base * D))
            vg = nl.ndarray((_CHUNK, _NCD), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=vg, src=v[batch_id].ap(pattern=_GLOAD_AP, offset=group_base * D))
            qg = nl.ndarray((_CHUNK, _NCD), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=qg, src=q[batch_id].ap(pattern=_GLOAD_AP, offset=group_base * D))
            # Group-contiguous operator buffer: each chunk's masked A_bf is written into its free-slice
            # A_bf_grp[:, c*_CHUNK:] so the 8 operators are ONE contiguous SBUF tensor. This lets the
            # HBM staging write all 8 A's in ONE strided DMA (chunks on free) instead of 8 copies.
            A_bf_grp = nl.ndarray((_CHUNK, cpg * _CHUNK), dtype=dtype, buffer=nl.sbuf)
            # Contiguous rhs buffers (chunk c on free-slice c*D): let the rhs pack stage to HBM in one
            # strided write + per-chunk fold-reads instead of tiny SBUF->SBUF regroup DMAs/group.
            rhs_w_grp = nl.ndarray((_CHUNK, cpg * D), dtype=nl.float32, buffer=nl.sbuf)
            rhs_u_grp = nl.ndarray((_CHUNK, cpg * D), dtype=nl.float32, buffer=nl.sbuf)

            # ---- BATCHED CUMSUM (all 8 chunks in ONE matmul) ----
            # Load the group's gate as [_CHUNK, CPG] (partition=token t, free=chunk c): one strided DMA
            # replaces 8 per-chunk row-loads + 8 scan_ones memsets + 8 DVE scans. The cumsum is a matmul
            # against the shared U_cumsum operator -> cg_all[_CHUNK, CPG] in fp32 PSUM, one op on the
            # (idle) PE array. Per chunk, cg = cg_all[:, c:c+1] is a free-view [_CHUNK,1] column, exactly
            # the layout exp_cg/decay consume (the old per-chunk dma_transpose to column is gone).
            _GATE_AP = [[1, _CHUNK], [_CHUNK, cpg]]
            gate_all_f32 = nl.ndarray((_CHUNK, cpg), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gate_all_f32, src=gate[batch_id].ap(pattern=_GATE_AP, offset=group_base))
            gate_all = nl.ndarray((_CHUNK, cpg), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_all, src=gate_all_f32, engine=nisa.vector_engine)
            cg_all_p = nl.ndarray((_CHUNK, cpg), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(cg_all_p, U_cumsum, gate_all)
            cg_all = nl.ndarray((_CHUNK, cpg), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cg_all, src=cg_all_p, engine=nisa.vector_engine)

            for c in range(cpg):
                chunk_idx = group_idx * _CHUNKS_PER_GROUP + c
                base_off = chunk_idx * _CHUNK

                k_sb = kg[:, nl.ds(c * D, D)]
                v_sb = vg[:, nl.ds(c * D, D)]
                q_sb = qg[:, nl.ds(c * D, D)]
                beta_sb = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=beta_sb, src=beta[batch_id, nl.ds(base_off, _CHUNK)])

                # cg is a free-view of the batched cumsum result (column [_CHUNK,1], no per-chunk op).
                cg = cg_all[:, nl.ds(c, 1)]
                # cg_row [1,_CHUNK] is still needed by Phase-2/3 (last-element broadcast + dte subtract);
                # regenerate it from the column via one transpose (was: scan produced it directly).
                cg_row = nl.ndarray((1, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_transpose(dst=cg_row, src=cg)
                exp_cg = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=exp_cg, data=cg, op=nl.exp)

                ones_CC = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(ones_CC, 1.0)
                cg_rows = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(cg_rows, ones_CC, op0=nl.multiply, operand0=cg)
                cg_cols_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(cg_cols_p, cg_rows)
                log_decay = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(log_decay, cg_rows, cg_cols_p, op=nl.subtract)
                log_decay_safe = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(log_decay_safe, log_decay, op0=nl.minimum, operand0=0.0)
                decay = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=decay, data=log_decay_safe, op=nl.exp)

                k_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=k_t, src=k_sb)
                kkt_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(kkt_p, k_t, k_t)
                # kkt_decay = kkt * decay reading kkt straight from PSUM (drop the separate eviction copy).
                kkt_decay = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(kkt_decay, kkt_p, decay, op=nl.multiply)
                neg_beta = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(neg_beta, beta_sb, op0=nl.multiply, operand0=-1.0)
                A_full = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(A_full, kkt_decay, op0=nl.multiply, operand0=neg_beta, engine=nisa.scalar_engine)
                A_masked = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.affine_select(
                    A_masked,
                    pattern=[[-1, _CHUNK]],
                    offset=-1,
                    channel_multiplier=1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=A_full,
                    on_false_value=0.0,
                )
                # Write the bf16 masked operator into this chunk's free-slice of the group buffer.
                A_masked_bf16 = A_bf_grp[:, nl.ds(c * _CHUNK, _CHUNK)]
                nisa.tensor_copy(dst=A_masked_bf16, src=A_masked, engine=nisa.vector_engine)
                # Precompute the FULL-chunk rhs ONCE (not per sub-block bi): rhs_w = k*beta*exp_cg,
                # rhs_u = v*beta over the whole [_CHUNK,D]. The bi-loop then just SLICES [bi*16:]
                # instead of re-loading k/v/beta from HBM per (bi,chunk) (was 48 HBM loads/group).
                rhs_w_full = rhs_w_grp[:, nl.ds(c * D, D)]
                rhs_u_full = rhs_u_grp[:, nl.ds(c * D, D)]
                rwtmp = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(rwtmp, k_sb, op0=nl.multiply, operand0=beta_sb, engine=nisa.scalar_engine)
                nisa.tensor_scalar(rhs_w_full, rwtmp, op0=nl.multiply, operand0=exp_cg, engine=nisa.scalar_engine)
                nisa.tensor_scalar(rhs_u_full, v_sb, op0=nl.multiply, operand0=beta_sb, engine=nisa.scalar_engine)
                # Keep the group's per-chunk Phase-1 tensors RESIDENT in SBUF (no HBM staging).
                group_state[c] = {
                    "q_sb": q_sb,
                    "k_sb": k_sb,
                    "k_t": k_t,
                    "exp_cg": exp_cg,
                    "decay": decay,
                    "cg_row": cg_row,
                }

            # ---- Pass 1b: BATCHED Phase 1 across the group (row-split). For each sub-block bi,
            # stack the _CHUNKS_PER_GROUP chunks' [_SUBBLK,D] tiles into [_GROUP_ROWS,D] and solve
            # them together: operators are block-diagonal per 32-row quadrant, matmuls use
            # tile_position row bands (device-verified layout). Inter-block coupling (bj<bi) is
            # serial in bi but batched across chunks. Chunks stay independent (block-diagonal). ----
            # (_GROUP_ROWS = cpg * _SUBBLK, defined in the group header.)
            # Store solved W/U CONTIGUOUS (sub-block bi on free-slice bi*D) so the Phase-2/3 regroup
            # stages to HBM in ONE strided write + 8 per-chunk contiguous reads (9 ops/tensor) instead
            # of 64 tiny SBUF->SBUF DMAs. Wg[bi]/Ug[bi] are free-views for the coupling reads.
            Wg_grp = nl.ndarray((_GROUP_ROWS, _N_SUBBLK * D), dtype=nl.float32, buffer=nl.sbuf)
            Ug_grp = nl.ndarray((_GROUP_ROWS, _N_SUBBLK * D), dtype=nl.float32, buffer=nl.sbuf)
            Wg = [None] * _N_SUBBLK  # Wg[bi] = free-view Wg_grp[:, bi*D:] ([_GROUP_ROWS, D])
            Ug = [None] * _N_SUBBLK

            # PARALLEL PREP: build ALL block-diag operators up front, OUTSIDE the serial bi solve
            # loop. Every operator depends only on A_bf (ready after Pass-1a), never on the solve
            # W/U, so hoisting removes the false build<->solve serialization (measured: sync 55.7%,
            # TensorE only 13.7% -> DMA/sync-bound, not compute-bound). Same fill+transpose ops as
            # before, just no longer stalling their dependent matmuls on the serial chain.
            #  - Ndiag_pre[bi] = plain block-diag(A_ii for all chunks); Mdiag_pre[bi] = its transpose.
            #  - Acpl_t_pre[bi][bj] = transpose of block-diag(A[bi,bj] for all chunks), bj<bi.
            Ndiag_pre = [None] * _N_SUBBLK
            Mdiag_pre = [None] * _N_SUBBLK
            # coupling operators in a FLAT list indexed by the strictly-lower-tri offset
            # _cpl_off(bi,bj) = bi*(bi-1)//2 + bj  (bj<bi); dict/tuple-keys trip the NKI tracer.
            _n_cpl = _N_SUBBLK * (_N_SUBBLK - 1) // 2
            Acpl_t_pre = [None] * _n_cpl

            # HBM-STAGED FILL (Stage 3): stage each chunk's A_bf to HBM (plain identity write), then
            # read the WHOLE A back into its B chunk-band in ONE grid-layout command. The block-row
            # split (bi) rides the flat HBM source side, so a single DMA lands all _N_GRID blocks onto
            # the single dest band c*_SUBBLK (device-verified: toy_hbm_to_band HBM_TO_BAND_OK). B uses
            # a GRID layout: operator (bi,bj) at slot bi*_N_SUBBLK+bj (6 upper slots stay zero, unused).
            # 8 writes + 8 reads = 16 DMAs (was 32 strided fills). Extract reads grid slots as views.
            _N_GRID = _N_SUBBLK * _N_SUBBLK
            _ROWLEN = _N_GRID * _GROUP_ROWS
            Ball = nl.ndarray((_GROUP_ROWS, _ROWLEN), dtype=dtype, buffer=nl.sbuf)
            # memset(0) keeps every unused diagonal band ZERO so the block-diagonal operator stays
            # leak-free (chunks never contaminate each other) -- also for a partial group.
            nisa.memset(Ball, 0.0)
            # ONE strided write stages all cpg contiguous A's to HBM as [_CHUNK, cpg*_CHUNK]
            # (chunk c on free slice c*_CHUNK) — identity layout, just chunks packed on free.
            A_hbm = nl.ndarray((_CHUNK, cpg * _CHUNK), dtype=dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=A_hbm, src=A_bf_grp)
            _HBM_ROWLEN = cpg * _CHUNK
            for c in range(cpg):
                # dst: chunk-band c*_SUBBLK (leading r), grid-diagonal free (bi,bj,col).
                dst_ap = Ball.ap(
                    pattern=[
                        [_ROWLEN, _SUBBLK],
                        [_N_SUBBLK * _GROUP_ROWS, _N_SUBBLK],
                        [_GROUP_ROWS, _N_SUBBLK],
                        [1, _SUBBLK],
                    ],
                    offset=c * _SUBBLK * _ROWLEN + c * _SUBBLK,
                )
                # src: flat HBM, chunk c's A at free offset c*_CHUNK; (r, bi, bj, col).
                src_ap = A_hbm.ap(
                    pattern=[
                        [_HBM_ROWLEN, _SUBBLK],
                        [_SUBBLK * _HBM_ROWLEN, _N_SUBBLK],
                        [_SUBBLK, _N_SUBBLK],
                        [1, _SUBBLK],
                    ],
                    offset=c * _CHUNK,
                )
                nisa.dma_copy(dst=dst_ap, src=src_ap)

            # Extract operators as free-views and transpose each once. Grid slot = bi*_N_SUBBLK+bj.
            # Diagonal (bi,bi): plain view (Ndiag_pre) + transpose (Mdiag_pre). Coupling (bi,bj<bi):
            # only the transpose is used by the solve.
            for bi in range(_N_SUBBLK):
                for bj in range(bi + 1):
                    slot = bi * _N_SUBBLK + bj
                    op_view = Ball[:, nl.ds(slot * _GROUP_ROWS, _GROUP_ROWS)]
                    opt = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=dtype, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=opt, src=op_view)
                    if bj == bi:
                        Ndiag_pre[bi] = op_view
                        Mdiag_pre[bi] = opt
                    else:
                        Acpl_t_pre[bi * (bi - 1) // 2 + bj] = opt

            # HBM-STAGED RHS PACK: the old inner loop did 64 tiny SBUF->SBUF regroup DMAs/group
            # (fold each chunk's rhs_*_full sub-block bi -> band c*_SUBBLK of a stacked [GR,D]). That
            # regroup crosses partition bands (chunk-band <- subblock-band), so it can't merge SBUF-
            # side. Route through flat HBM instead: ONE strided write of the contiguous rhs_*_grp, then
            # per-chunk fold-reads that land ALL 4 sub-blocks of a chunk onto its band c*_SUBBLK with
            # the sub-block index on the FREE axis. Result rhs_*_stack[GR, _N_SUBBLK*D]; the solve for
            # sub-block bi reads the free-view [:, bi*D:(bi+1)*D]. 64 DMAs/group -> 2 writes + 16 reads.
            _RHS_ROWLEN = cpg * D
            rhs_w_hbm = nl.ndarray((_CHUNK, _RHS_ROWLEN), dtype=nl.float32, buffer=nl.shared_hbm)
            rhs_u_hbm = nl.ndarray((_CHUNK, _RHS_ROWLEN), dtype=nl.float32, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=rhs_w_hbm, src=rhs_w_grp)
            nisa.dma_copy(dst=rhs_u_hbm, src=rhs_u_grp)
            rhs_w_stack = nl.ndarray((_GROUP_ROWS, _N_SUBBLK * D), dtype=nl.float32, buffer=nl.sbuf)
            rhs_u_stack = nl.ndarray((_GROUP_ROWS, _N_SUBBLK * D), dtype=nl.float32, buffer=nl.sbuf)
            _RSTK = _N_SUBBLK * D
            _RSRC_AP = [[_RHS_ROWLEN, _SUBBLK], [_SUBBLK * _RHS_ROWLEN, _N_SUBBLK], [1, D]]
            _RDST_AP = [[_RSTK, _SUBBLK], [D, _N_SUBBLK], [1, D]]
            for c in range(cpg):
                # dst: chunk-band c*_SUBBLK (leading r=16), sub-block bi on free (stride D), then D cols.
                # src: flat HBM chunk c at free c*D; (r, bi, d): r stride _RHS_ROWLEN, bi stride _SUBBLK*
                # _RHS_ROWLEN (next sub-block = +16 partition rows in the [64,*] source), d stride 1.
                nisa.dma_copy(
                    dst=rhs_w_stack.ap(pattern=_RDST_AP, offset=c * _SUBBLK * _RSTK),
                    src=rhs_w_hbm.ap(pattern=_RSRC_AP, offset=c * D),
                )
                nisa.dma_copy(
                    dst=rhs_u_stack.ap(pattern=_RDST_AP, offset=c * _SUBBLK * _RSTK),
                    src=rhs_u_hbm.ap(pattern=_RSRC_AP, offset=c * D),
                )

            for bi in range(_N_SUBBLK):
                # stacked rhs for this sub-block = free-view of the HBM-staged stacks (no pack DMAs).
                rhs_w = nl.ndarray((_GROUP_ROWS, D), dtype=nl.float32, buffer=nl.sbuf)
                rhs_u = nl.ndarray((_GROUP_ROWS, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rhs_w, src=rhs_w_stack[:, nl.ds(bi * D, D)], engine=nisa.vector_engine)
                nisa.tensor_copy(dst=rhs_u, src=rhs_u_stack[:, nl.ds(bi * D, D)], engine=nisa.vector_engine)

                # inter-block coupling: rhs += A[bi,bj] @ W[bj] for bj<bi. BATCHED via probe_one idiom:
                # block-diag operator from PLAIN [16,16] blocks in HBM (dma_copy each -> Nh diag),
                # ONE dma_transpose of the whole [GR,GR] (= block-diag of A^T), stacked operand already
                # in Wg[bj]/Ug[bj] ([GR,D]), ONE matmul, PSUM->SBUF, add to rhs.
                # Coupling: ACCUMULATE all bj matmuls into ONE psum (accumulate=True), then a single
                # tensor_tensor add to rhs — removes the per-bj vector adds + intermediate rhs r/w
                # (fewer ops on the DMA<->Vector serial chain).
                if bi > 0:
                    cwp = nl.ndarray((_GROUP_ROWS, D), dtype=nl.float32, buffer=nl.psum)
                    cup = nl.ndarray((_GROUP_ROWS, D), dtype=nl.float32, buffer=nl.psum)
                    for bj in range(bi):
                        Aijt = Acpl_t_pre[bi * (bi - 1) // 2 + bj]  # pre-built coupling operator (hoisted)
                        Wj = nl.ndarray((_GROUP_ROWS, D), dtype=dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=Wj, src=Wg[bj], engine=nisa.vector_engine)
                        nisa.nc_matmul(
                            cwp,
                            Aijt,
                            Wj,
                            tile_position=(0, 0),
                            tile_size=(_GROUP_ROWS, _GROUP_ROWS),
                            accumulate=(bj > 0),
                        )
                        Uj = nl.ndarray((_GROUP_ROWS, D), dtype=dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=Uj, src=Ug[bj], engine=nisa.vector_engine)
                        nisa.nc_matmul(
                            cup,
                            Aijt,
                            Uj,
                            tile_position=(0, 0),
                            tile_size=(_GROUP_ROWS, _GROUP_ROWS),
                            accumulate=(bj > 0),
                        )
                    nisa.tensor_tensor(dst=rhs_w, data1=rhs_w, data2=cwp, op=nl.add)
                    nisa.tensor_tensor(dst=rhs_u, data1=rhs_u, data2=cup, op=nl.add)

                # SHARED-M solve: build MT = M^T = (I-A_ii)^-1 ^T ONCE via doubling on the identity,
                # then W = M@rhs_w = nc_matmul(MT, rhs_w) and U = nc_matmul(MT, rhs_u) share MT (the
                # doubling runs once, not once-per-rhs). nc_matmul(stat,mov)=stat^T@mov, so:
                #   MT += nc_matmul(PT_stat, MT)   (= P@MT, P=A_ii^T)
                #   square: P=nc_matmul(PT_stat,P), PT_stat=nc_matmul(P,PT_stat)  (CPU-verified 6.7e-16)
                # Operators: Ndiag_pre[bi]=A_ii (=PT_stat round0), Mdiag_pre[bi]=A_ii^T (=P round0).
                P = Mdiag_pre[bi]  # P = A_ii^T
                PT_stat = Ndiag_pre[bi]  # P^T = A_ii
                MT = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=MT, src=Ident_g, engine=nisa.scalar_engine)
                for _rd in range(_N_ROUNDS):
                    pMT = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pMT, PT_stat, MT, tile_position=(0, 0), tile_size=(_GROUP_ROWS, _GROUP_ROWS))
                    MT_new = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=MT_new, data1=MT, data2=pMT, op=nl.add)
                    MT = MT_new
                    if _rd < _N_ROUNDS - 1:
                        pp = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(pp, PT_stat, P, tile_position=(0, 0), tile_size=(_GROUP_ROWS, _GROUP_ROWS))
                        ppt = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(ppt, P, PT_stat, tile_position=(0, 0), tile_size=(_GROUP_ROWS, _GROUP_ROWS))
                        P = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=P, src=pp, engine=nisa.scalar_engine)
                        PT_stat = nl.ndarray((_GROUP_ROWS, _GROUP_ROWS), dtype=dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=PT_stat, src=ppt, engine=nisa.scalar_engine)
                # Apply shared MT to both rhs: W=nc_matmul(MT,rhs_w), U=nc_matmul(MT,rhs_u).
                rhs_w_b = nl.ndarray((_GROUP_ROWS, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rhs_w_b, src=rhs_w, engine=nisa.scalar_engine)
                rhs_u_b = nl.ndarray((_GROUP_ROWS, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rhs_u_b, src=rhs_u, engine=nisa.scalar_engine)
                yW = nl.ndarray((_GROUP_ROWS, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(yW, MT, rhs_w_b, tile_position=(0, 0), tile_size=(_GROUP_ROWS, _GROUP_ROWS))
                yU = nl.ndarray((_GROUP_ROWS, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(yU, MT, rhs_u_b, tile_position=(0, 0), tile_size=(_GROUP_ROWS, _GROUP_ROWS))
                # store into contiguous grp buffers (bi on free); expose free-views for coupling reads.
                nisa.tensor_copy(dst=Wg_grp[:, nl.ds(bi * D, D)], src=yW, engine=nisa.vector_engine)
                nisa.tensor_copy(dst=Ug_grp[:, nl.ds(bi * D, D)], src=yU, engine=nisa.vector_engine)
                Wg[bi] = Wg_grp[:, nl.ds(bi * D, D)]
                Ug[bi] = Ug_grp[:, nl.ds(bi * D, D)]

            # HBM-STAGED W/U REGROUP: per-band write (8) + ONE contiguous read. Each chunk-band c of
            # Wg_grp ([16, 4D] = chunk c's 4 sub-blocks) is written to HBM in the FINAL layout
            # W_all[bi*16+row][c][d] (partition=bi*16+row, chunk on free). HBM is flat so the write's
            # bi-on-free -> bi-on-partition swap rides the (unrestricted) HBM dst strides. Then ONE
            # contiguous read pulls all cpg chunks into W_all[_CHUNK, cpg*D]; W_full[c] is
            # the free-view W_all[:, c*D:]. Tiny DMAs -> cpg writes + 1 read per tensor.
            _WROWLEN = _N_SUBBLK * D  # Wg_grp row-length (4D)
            _WALL = cpg * D  # W_all row-length (cpg*D), chunk on free
            W_hbm = nl.ndarray((_CHUNK, _WALL), dtype=nl.float32, buffer=nl.shared_hbm)
            U_hbm = nl.ndarray((_CHUNK, _WALL), dtype=nl.float32, buffer=nl.shared_hbm)
            # per band c: src Wg_grp[c*16.., :] (row,bi,d); dst W_hbm[(bi*16+row)*cpg*D + c*D + d].
            _WBAND_SRC = [[_WROWLEN, _SUBBLK], [D, _N_SUBBLK], [1, D]]  # (row, bi, d)
            _WBAND_DST = [[_WALL, _SUBBLK], [_SUBBLK * _WALL, _N_SUBBLK], [1, D]]  # (row, bi, d) -> part=bi*16+row
            for c in range(cpg):
                nisa.dma_copy(
                    dst=W_hbm.ap(pattern=_WBAND_DST, offset=c * D),
                    src=Wg_grp.ap(pattern=_WBAND_SRC, offset=c * _SUBBLK * _WROWLEN),
                )
                nisa.dma_copy(
                    dst=U_hbm.ap(pattern=_WBAND_DST, offset=c * D),
                    src=Ug_grp.ap(pattern=_WBAND_SRC, offset=c * _SUBBLK * _WROWLEN),
                )
            # ONE contiguous read each: HBM -> W_all/U_all [_CHUNK, _WALL]; W_full[c]=free-view [:,c*D:].
            W_all = nl.ndarray((_CHUNK, _WALL), dtype=dtype, buffer=nl.sbuf)
            U_all = nl.ndarray((_CHUNK, _WALL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=W_all, src=W_hbm)
            nisa.dma_copy(dst=U_all, src=U_hbm)
            for c in range(cpg):
                chunk_idx = group_idx * _CHUNKS_PER_GROUP + c
                base_off = chunk_idx * _CHUNK
                # All Phase-1 tensors are resident SBUF (no reload).
                q_sb = group_state[c]["q_sb"]
                k_sb = group_state[c]["k_sb"]
                k_t = group_state[c]["k_t"]
                exp_cg = group_state[c]["exp_cg"]
                decay = group_state[c]["decay"]
                cg_row = group_state[c]["cg_row"]
                # === Phase 2+3 combined: Output + State update ===
                # For output we need state at START of chunk (current `state`).
                # For state update we need v_new = U - W @ state.
                # Compute both together per block.

                # Prepare state as bf16 for matmul (read-only, state stays f32)
                state_bf16 = nl.ndarray((D, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=state_bf16, src=state, engine=nisa.vector_engine)

                # o_cross = q @ state * exp_cg * scale (uses state at start of chunk)
                q_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=q_t, src=q_sb)
                o_cross_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(o_cross_p, q_t, state_bf16)
                exp_cg_scaled = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(exp_cg_scaled, exp_cg, op0=nl.multiply, operand0=scale)
                o_cross = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    o_cross, o_cross_p, op0=nl.multiply, operand0=exp_cg_scaled, engine=nisa.scalar_engine
                )

                # qkt_masked for intra-chunk output: lower-tri-inclusive
                qkt_p = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(qkt_p, q_t, k_t)
                qkt_s = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qkt_s, src=qkt_p, engine=nisa.scalar_engine)
                qkt_decay = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(qkt_decay, qkt_s, decay, op=nl.multiply)
                qkt_masked = nl.ndarray((_CHUNK, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.affine_select(
                    qkt_masked,
                    pattern=[[-1, _CHUNK]],
                    offset=0,
                    channel_multiplier=1,
                    cmp_op=nl.greater_equal,
                    on_true_tile=qkt_decay,
                    on_false_value=0.0,
                )

                # Compute v_new per block, accumulate o_intra and state update together
                # o_intra = sum_j qkt_masked[:, j*_SUBBLK:(j+1)*_SUBBLK]^T @ v_new_j * scale
                # state update: state = state * exp_g_last + sum_j k_dec_j^T @ v_new_j

                # Decay state for next chunk FIRST (v_new still uses pre-decay state_bf16)
                # exp_g_last = exp(cg[_CHUNK-1]) broadcast to [D, 1]
                ones_1D = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(ones_1D, 1.0)
                cg_last_1D = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    cg_last_1D, ones_1D, op0=nl.multiply, operand0=cg_row[nl.ds(0, 1), nl.ds(_CHUNK - 1, 1)]
                )
                exp_g_last_1D = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=exp_g_last_1D, data=cg_last_1D, op=nl.exp)
                exp_g_last_D1 = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_transpose(dst=exp_g_last_D1, src=exp_g_last_1D)

                state_decayed = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    state_decayed, state, op0=nl.multiply, operand0=exp_g_last_D1, engine=nisa.scalar_engine
                )
                nisa.tensor_copy(dst=state, src=state_decayed, engine=nisa.vector_engine)

                # === WHOLE-CHUNK FACTORED Phase-2/3 (math-identical to per-sub-block, verified ~1e-16).
                # The sub-blocking was redundant: v_new, o_intra, and the state update all factor into
                # single full-chunk matmuls. 3 matmuls + a few transposes replace 12 matmuls + 8 transposes.
                # W_full/U_full = free-view of the single-read W_all/U_all (chunk c on free). No DMA.
                W_full = W_all[:, nl.ds(c * D, D)]
                U_full = U_all[:, nl.ds(c * D, D)]

                # k_dec = k * decay_to_end over the WHOLE chunk: decay_to_end = exp(cg_last - cg).
                # Build in ROW form [1,64] (cg_last broadcasts on the FREE axis = HW-safe), then
                # transpose once to [64,1] to scale k_sb (partition-aligned per-token scalar).
                ones_1C = nl.ndarray((1, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(ones_1C, 1.0)
                cg_last_1C = nl.ndarray((1, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    cg_last_1C, ones_1C, op0=nl.multiply, operand0=cg_row[nl.ds(0, 1), nl.ds(_CHUNK - 1, 1)]
                )
                log_dte_1C = nl.ndarray((1, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(log_dte_1C, cg_last_1C, cg_row, op=nl.subtract)
                dte_1C = nl.ndarray((1, _CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=dte_1C, data=log_dte_1C, op=nl.exp)
                dte_C1 = nl.ndarray((_CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_transpose(dst=dte_C1, src=dte_1C)
                k_dec_full = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(k_dec_full, k_sb, op0=nl.multiply, operand0=dte_C1, engine=nisa.scalar_engine)

                # v_new = U - W @ S0  (S0=pre-decay state_bf16). W@S0 = matmul(W^T[D,64], S0[D,D]) -> [64,D]
                W_t = nl.ndarray((D, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=W_t, src=W_full)
                WS_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(WS_p, W_t, state_bf16)
                v_new = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(v_new, U_full, WS_p, op=nl.subtract)
                v_new_bf16 = nl.ndarray((_CHUNK, D), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=v_new_bf16, src=v_new, engine=nisa.vector_engine)

                # o_intra = qkt_masked @ v_new = matmul(qkt_masked^T[64,64], v_new[64,D]) -> [64,D]
                qkt_m_bf16 = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qkt_m_bf16, src=qkt_masked, engine=nisa.vector_engine)
                qkt_m_t = nl.ndarray((_CHUNK, _CHUNK), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=qkt_m_t, src=qkt_m_bf16)
                o_intra_p = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(o_intra_p, qkt_m_t, v_new_bf16)
                o_intra = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(o_intra, o_intra_p, op0=nl.multiply, operand0=scale, engine=nisa.scalar_engine)

                # state = state_decayed + k_dec^T @ v_new = matmul(k_dec[64,D], v_new[64,D]) -> [D,D] (no transpose)
                su_p = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(su_p, k_dec_full, v_new_bf16)
                state_new = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=state_new, data1=su_p, data2=state, op=nl.add)
                nisa.tensor_copy(dst=state, src=state_new, engine=nisa.vector_engine)

                # Final output = o_cross + o_intra
                o_out = nl.ndarray((_CHUNK, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o_out, data1=o_cross, data2=o_intra, op=nl.add)
                nisa.dma_copy(dst=result[batch_id, nl.ds(base_off, _CHUNK), nl.ds(0, D)], src=o_out)

        # Store final state
        state_for_dma = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(state_for_dma, state, op0=nl.multiply, operand0=1.0, engine=nisa.scalar_engine)
        nisa.dma_copy(dst=state_output[batch_id, nl.ds(0, D), nl.ds(0, D)], src=state_for_dma)

    return result, state_output
