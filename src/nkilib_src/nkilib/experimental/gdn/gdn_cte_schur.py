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
GDN chunked prefill, SCHUR-COMPLEMENT intra-chunk inverse (level-major grouped).

The default algorithm behind `gdn_cte`, and the faster of the two at every benchmarked
shape (see the family README for the table). Same math as FLA/CUDA
chunk_gated_delta_rule; specific to this implementation is how (I + A)^-1 is formed and
how the work is ordered.

  INVERSE   Schur complement on a 2x2 block-triangular matrix, applied recursively:
            inv([[P, 0], [R, Q]]) = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]]. Block size doubles
            per level from 1 to CHUNK, so a CHUNK x CHUNK inverse is log2(CHUNK) = 7 merge
            levels. Written literally that is 127 tiny block operations; because every
            block of one level lands in a DISJOINT region of the same tile, one tile-wide
            chain resolves a whole level at once -- 12 matmuls instead of 127. No power of
            A is formed, so nothing depends on A being small.
  ORDER     the LOOP NEST IS INVERTED relative to a per-chunk implementation: the merge
            LEVEL is the outer loop and the CHUNK the inner one.

Three stages, by what each depends on:

  STAGE 1  per chunk, STATE-INDEPENDENT: load, cumulative gate, decay matrix M, A, and
           every state-independent stage-3 operand.
  LEVEL 0  the first merge level, LEVEL-MAJOR over the group like stage 2 -- NOT part of
           stage 1, which is per chunk.
  STAGE 2  the remaining block-merge levels, LEVEL-MAJOR over the group. Do not merge
           these sub-loops back into a per-chunk loop.
  STAGE 3  the recurrent part, sequential over the group. The only stage that touches the
           carried state.

Algorithm (per chunk of CHUNK tokens):
  1. log_G[s] = cumsum(g_log[..s]), as a prefix sum along the FREE axis on the vector engine
     -- one instruction for the whole group at once -- then transposed once so each chunk's
     per-token column is a free slice.
  2. A = beta * M * (k^T k), strictly lower triangular.
  3. (I + A)^-1 by the recursive block merge above. Level 0 (block size 1) is pure
     elementwise because Inv == I there, and it builds Inv, so its transpose hands the
     first merge level both orientations. Inv_T is what the levels carry, in the MATMUL
     OPERAND dtype; Inv is re-derived per level as its transpose.
  4. o = G * (q^T S0) + (M * (q^T k)) @ delta,  where delta = (I+A)^-1 b_vec and
     b_vec = beta*v - (beta*G) * (k^T S0)
  5. S_new = G_last * S0 + ((G_last / G) * k)^T @ delta.

NUMERICS. The COMPUTE DTYPE is the dtype of `query` at trace time: it sets the matmul
operands, the q/k/v tiles and `out`. The recurrent state is always fp32 (it accumulates
across the whole sequence) and PSUM accumulates fp32 whatever the operand width.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .gdn_cte_utils import (
    CHUNK,
    COPY_ENGINE,
    IDX_EYE,
    IDX_NEG_OFF,
    IDX_OFF0,
    IDX_TRIL_STRICT,
    IDX_UPPER_NEG,
    MAX_HEAD_DIM,
    N_MASKS,
    N_MERGE_LEVELS,
    group_layout,
    transpose_tile,
)

# Chunks per grouped inverse. `group_layout` CLAMPS it to the chunk count, then splits into
# full groups of that width plus one residual group of the remainder.
DEFAULT_GROUP_SIZE = 8

# Upper bound on `group_size`: an SBUF/PSUM bound. At the stage-2 peak a group holds several
# (CHUNK, CHUNK) tiles PER MEMBER -- the interaction, Inv_T, Inv, R, Mid and W_T -- plus one
# live PSUM bank per member, so the footprint grows with `group_size` at that multiple.
MAX_GROUP_SIZE = 16


def gdn_cte_schur(
    query: nl.NkiTensor,
    key: nl.NkiTensor,
    value: nl.NkiTensor,
    g_log: nl.NkiTensor,
    beta: nl.NkiTensor,
    mask_pack: nl.NkiTensor,
    scale: float = 1.0,
    group_size: int = DEFAULT_GROUP_SIZE,
    recurrent_state: Optional[nl.NkiTensor] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Chunked gated delta-rule (GDN) prefill kernel, Schur-complement intra-chunk inverse.

    Computes the gated delta-rule linear-attention recurrence in chunk-parallel form,
    forming (I + A)^-1 by a recursive Schur-complement block merge with the merge level as
    the OUTER loop, so each sub-loop emits `group_size` independent same-kind operations and
    the per-level dependency stall is amortized across the group. Requires S a multiple of
    CHUNK=128 and head dims at most MAX_HEAD_DIM=128. The grouping needs enough chunks to
    amortize; below `group_size` chunks the width is clamped to the chunk count, so the whole
    sequence runs as ONE group of everything it has.

    Dimensions:
        B: Batch size
        Hk: Key/query head count
        Hv: Value head count, a multiple of Hk (GQA ratio Hv // Hk)
        Dk: Key/query/state head dimension (<= 128; 128 is the efficient value)
        Dv: Value/state head dimension (<= 128, and independent of Dk)
        S: Sequence length (must be a multiple of CHUNK)
        CHUNK: Chunk length (128)
        GROUP: Chunks per grouped inverse (`group_size`), plus a residual group of
            S/CHUNK mod GROUP when the split is not exact

    Args:
        query (nl.NkiTensor): [B, Hk, S, Dk], Query tensor in HBM, TOKEN-major. ITS DTYPE
            SETS THE COMPUTE DTYPE for the whole kernel. Must be L2-normalized, see Notes.
        key (nl.NkiTensor): [B, Hk, S, Dk], Key tensor in HBM, TOKEN-major, L2-normalized,
            same dtype as `query`.
        value (nl.NkiTensor): [B, Hv, S, Dv], Value tensor in HBM, same dtype as `query`.
        g_log (nl.NkiTensor): [B, Hv, S], Per-token log-decay (negative) in HBM, fp32.
        beta (nl.NkiTensor): [B, Hv, S], Per-token update strength in (0, 1) in HBM.
        mask_pack (nl.NkiTensor): [N_MASKS, CHUNK, CHUNK], Constant mask pack in HBM,
            fp32. Build it with `build_mask_pack()`.
        scale (float): Query scaling, typically 1/sqrt(Dk). Applied ON DEVICE: free on the
            `W_T` transpose evacuation, which runs either way, plus ONE (CHUNK, 1) pass per
            chunk to carry it on the gate column when `scale != 1.0`.
        group_size (int): Chunks per grouped inverse, honored whenever the sequence has at
            least that many chunks: with `group_width = min(group_size, n_chunks)`, the chunk
            count runs as full groups of `group_width` plus one residual group of
            `n_chunks % group_width`. Defaults to DEFAULT_GROUP_SIZE, the measured optimum.
        recurrent_state (Optional[nl.NkiTensor]): [B, Hv, Dk, Dv] fp32, OPTIONAL MUTABLE IN/OUT.
            Supplied, it is aliased to the second output, updated IN PLACE, and its
            incoming contents are the initial state. Omitted, the output is allocated and
            the state starts at zero with no load emitted.

    Returns:
        out (nl.NkiTensor): [B, Hv, S, Dv], Output hidden states in HBM in the compute
            dtype, HEAD-major and UN-normalized (no fused output epilogue).
        recurrent_state (nl.NkiTensor): [B, Hv, Dk, Dv], Final recurrent state, always fp32 --
            the caller's buffer when one was supplied.

    Notes:
        - S must be divisible by CHUNK=128; Dk and Dv must be at most MAX_HEAD_DIM=128.
          Dk = Dv is NOT required.
        - Hv must be divisible by Hk; value head h reads key/query head h // (Hv//Hk).
        - `group_size` is honored whenever the sequence has at least that many chunks, and
          clamped to the chunk count below that: full groups plus ONE residual group at its
          own width, so no padding beyond CHUNK is needed. The residual group
          changes only emission order -- output verified bit-identical to group_size=1.

    Pseudocode:
        # Every matmul below contracts its STATIONARY operand: nc_matmul(stationary=S,
        # moving=M) computes S^T @ M, so each product is written with that transpose explicit.
        state = recurrent_state or 0
        # group_width = min(group_size, n_chunks) chunks at a time, then one last group of
        # n_chunks % group_width if any
        for group in full_groups + ([residual_group] if n_chunks % group_width else []):
            log_G = cumsum(g_log[group])          # one vector instruction for the group
            # STAGE 1: per chunk, state-independent
            for member in range(n_group_chunks):
                M = exp(log_G[r] - log_G[s]) masked   # log_G from the group prefix sum
                A = beta * M * (k^T k)
                cache A, scale*W_T = (M * q^T k)^T, (G_last/G) * k^T, beta*v, and the
                      gate columns beta*G, scale*G, G_last
            # LEVEL 0 of the merge: elementwise, because Inv == I at block size 1
            for member: Inv[member]   = I - A[member] * off0
            for member: Inv_T[member] = transpose(Inv[member])
            # STAGE 2: block-merge levels, LEVEL-MAJOR over the group. Inv arrives from
            # level 0 already, so only later levels pay for the transpose.
            for level in merge_levels:            # block sizes 2, 4, ... 64
                if level > 0: for member: Inv[member] = transpose(Inv_T[member])
                for member: R[member]     = A[member] * neg_off_mask
                for member: Mid[member]   = R[member]^T @ Inv_T[member]
                for member: Off_T[member] = Inv[member]^T @ Mid[member]   # left in PSUM
                for member: Inv_T[member] = Inv_T[member] + Off_T[member]
            # STAGE 3: recurrent, sequential over the group
            for member in range(n_group_chunks):
                b_vec = beta*v - (beta*G) * (k^T state)
                delta = Inv_T^T @ b_vec                # = (I + A)^-1 b_vec
                out[chunk] = (scale*G) * (q^T state) + (scale*W_T)^T @ delta
                state = G_last * state + ((G_last/G) * k)^T @ delta
        return out, state
    """
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]

    # q and k contract against each other in k^T k and q^T k, so their head dims must match.
    kernel_assert(
        key_head_dim == key.shape[-1],
        f"query and key must have the same head dim, got {key_head_dim} and {key.shape[-1]}",
    )
    kernel_assert(
        key_head_dim <= MAX_HEAD_DIM and value_head_dim <= MAX_HEAD_DIM,
        f"head dims must be at most MAX_HEAD_DIM={MAX_HEAD_DIM}, got key_head_dim="
        f"{key_head_dim} and value_head_dim={value_head_dim}",
    )
    kernel_assert(
        seqlen >= CHUNK and seqlen % CHUNK == 0,
        f"seqlen must be a positive multiple of CHUNK={CHUNK}, got {seqlen}",
    )
    kernel_assert(
        n_v_heads % n_k_heads == 0,
        f"n_v_heads must be divisible by n_k_heads, got {n_v_heads} and {n_k_heads}",
    )
    kernel_assert(
        mask_pack.shape[0] == N_MASKS,
        f"mask_pack must have {N_MASKS} planes, got {mask_pack.shape[0]} -- build it with build_mask_pack()",
    )
    kernel_assert(
        mask_pack.shape[1] == CHUNK and mask_pack.shape[2] == CHUNK,
        f"mask_pack planes must be {CHUNK}x{CHUNK}, got {mask_pack.shape[1]}x{mask_pack.shape[2]}",
    )
    kernel_assert(
        1 <= group_size <= MAX_GROUP_SIZE,
        f"group_size must be in [1, {MAX_GROUP_SIZE}], got {group_size}.",
    )

    v_heads_per_k_head = n_v_heads // n_k_heads
    n_chunks = seqlen // CHUNK

    """SPMD SHARDING OVER (BATCH x VALUE HEAD) PAIRS. Call each pair a LANE: it owns one
    (Dk, Dv) recurrence, one slice of `out` and one slice of the state, and shares nothing with
    any other lane. So the whole `batch * n_v_heads` set is one flat list of independent work,
    and splitting it needs no communication and no reduction."""
    n_shards = nl.num_programs(axes=0)
    shard_id = nl.program_id(axis=0)
    n_lanes = batch * n_v_heads
    lanes_per_shard, extra_lanes = divmod(n_lanes, n_shards)
    if shard_id < extra_lanes:
        lanes_per_shard += 1
    lane_start = shard_id * (n_lanes // n_shards) + min(shard_id, extra_lanes)
    # `n_full_groups` groups of `group`, then ONE residual group of what is left, so
    # `group_size` holds at every sequence length (see `group_layout`).
    group_width, n_full_groups, residual = group_layout(n_chunks, group_size)
    group_widths = [group_width]
    group_counts = [n_full_groups]
    group_offsets = [0]
    if residual:
        group_widths.append(residual)
        group_counts.append(1)
        group_offsets.append(n_full_groups * group_width)

    # COMPUTE DTYPE from the query tensor: it governs the matmul operands, the q/k/v
    # tiles and `out`. The state and PSUM stay fp32.
    compute_dtype = query.dtype

    out = nl.ndarray((batch, n_v_heads, seqlen, value_head_dim), dtype=compute_dtype, buffer=nl.shared_hbm)

    """`recurrent_state` is an OPTIONAL MUTABLE IN/OUT. Supplied -> a device operand aliased
    to output 1, so its incoming contents are the initial state and the caller's buffer is
    updated in place. Omitted -> allocate the output and start from zero, emitting no load."""
    state_is_input = recurrent_state is not None
    if not state_is_input:
        recurrent_state = nl.ndarray(
            (batch, n_v_heads, key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.shared_hbm
        )

    query_view = query.reshape((batch, n_k_heads, n_chunks, CHUNK, key_head_dim))
    key_view = key.reshape((batch, n_k_heads, n_chunks, CHUNK, key_head_dim))
    value_view = value.reshape((batch, n_v_heads, n_chunks, CHUNK, value_head_dim))
    # g_log is read as ROWS, a whole group at a time: the cumulative gate is a prefix sum
    # along the free axis, and the vector engine does one for every chunk of the group at once.
    g_log_row_view = g_log.reshape((batch, n_v_heads, n_chunks, CHUNK))
    beta_view = beta.reshape((batch, n_v_heads, n_chunks, CHUNK, 1))
    out_view = out.reshape((batch, n_v_heads, n_chunks, CHUNK, value_head_dim))

    # Constant masks, loaded once.
    upper_neg = nl.load(mask_pack[IDX_UPPER_NEG, :, :])
    tril_strict = nl.load(mask_pack[IDX_TRIL_STRICT, :, :])
    # The identity serves the cumulative gate's partition spread, `off0` the elementwise
    # level 0 of the merge.
    eye_mat = nl.load(mask_pack[IDX_EYE, :, :])
    off0 = nl.load(mask_pack[IDX_OFF0, :, :])
    """`neg_off_masks[i]` is the off-block mask for block size 2^(i+1). Level 0 has no plane
    because it is elementwise, so there is one fewer of these than there are merge levels."""
    neg_off_masks = []
    for level in range(N_MERGE_LEVELS - 1):
        neg_off_masks.append(nl.load(mask_pack[IDX_NEG_OFF + level, :, :]))

    for lane_idx in range(lane_start, lane_start + lanes_per_shard):
        # Batch-major, so lane order matches the nested (batch, head) loops this replaced.
        batch_idx, v_head_idx = divmod(lane_idx, n_v_heads)
        k_head_idx = v_head_idx // v_heads_per_k_head
        if state_is_input:
            state = nl.load(recurrent_state[batch_idx, v_head_idx, :, :], dtype=nl.float32)
        else:
            # No DMA, but the tile must exist before the first chunk's k^T @ state.
            state = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=state, value=0.0)
        """`state` is carried fp32 and consumed by matmuls in the operand dtype, so both
        widths are live at once."""
        state_operand = nl.ndarray((key_head_dim, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
        nisa.activation(dst=state_operand, op=nl.copy, data=state)

        # One emission per DISTINCT group width.
        for schedule_idx in range(len(group_widths)):
            n_group_chunks = group_widths[schedule_idx]
            chunk_offset = group_offsets[schedule_idx]
            for group_idx in nl.sequential_range(group_counts[schedule_idx]):
                chunk_base = chunk_offset + group_idx * n_group_chunks
                """Per-chunk tiles live across the level loop. Indexed by a HOST int
                (static unroll), never a LoopVar: the tracer cannot index a Python list
                with a loop variable."""
                key_tiles = []
                key_t_tiles = []
                query_tiles = []
                beta_value_tiles = []
                beta_gate_tiles = []
                gate_scaled_tiles = []
                gate_last_tiles = []
                interaction_tiles = []
                inv_tiles = []
                inv_t_tiles = []
                w_t_tiles = []

                """THE CUMULATIVE GATE, for every chunk of the group in one pass.
                log_G[s] = cumsum(g_log[..s]) is a prefix sum along the free axis"""
                g_log_rows = nl.ndarray((n_group_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=g_log_rows,
                    src=g_log_row_view[batch_idx, v_head_idx, chunk_base : chunk_base + n_group_chunks, :],
                    dge_mode=nisa.dge_mode.none,
                )
                log_gate_rows = nl.ndarray((n_group_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar_cumulative(dst=log_gate_rows, src=g_log_rows, op0=nl.multiply, imm0=1.0, op1=nl.add)
                # (CHUNK, n_group_chunks): column `member_idx` is that chunk's log_G, per token.
                log_gate_cols = transpose_tile(log_gate_rows, n_group_chunks, CHUNK)

                """STAGE 1: per chunk, load + M + A + every state-INDEPENDENT stage-3
                operand. Emitted together so sibling chunks' vector work fills each other's
                matmul stalls. Level 0 of the inverse is NOT here -- it follows this loop,
                emitted level-major over the group like stage 2."""
                for member_idx in range(n_group_chunks):
                    chunk_idx = chunk_base + member_idx

                    """q/k arrive TOKEN-major. dma_transpose does the permute INSIDE the load."""
                    q_tile = nl.ndarray((key_head_dim, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.dma_transpose(
                        dst=q_tile,
                        src=query_view[batch_idx, k_head_idx, chunk_idx, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    k_tile = nl.ndarray((key_head_dim, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.dma_transpose(
                        dst=k_tile,
                        src=key_view[batch_idx, k_head_idx, chunk_idx, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    """k a SECOND time, un-transposed, because the state update contracts
                    over tokens and so needs the token axis on partitions. A second load
                    rather than a tensor-engine transpose of `k_tile`: a load costs no
                    engine instruction at all, only HBM bandwidth and DMA queue time, and
                    the DMA engines run at well under half the utilization of the three
                    compute engines this would otherwise occupy."""
                    key_token_major = nl.ndarray((CHUNK, key_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=key_token_major,
                        src=key_view[batch_idx, k_head_idx, chunk_idx, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    v_tile = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=v_tile,
                        src=value_view[batch_idx, v_head_idx, chunk_idx, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    beta_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=beta_col,
                        src=beta_view[batch_idx, v_head_idx, chunk_idx, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )

                    log_gate_col = log_gate_cols[:, member_idx : member_idx + 1]
                    log_gate_bcast = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=log_gate_bcast, stationary=log_gate_col.broadcast(1, CHUNK), moving=eye_mat)
                    """The LAST column of `log_gate_bcast` is the chunk's total decay, and it
                    is constant down the partitions -- so the one quantity that would
                    otherwise need a partition broadcast arrives already broadcast, and is
                    read straight out of PSUM by the two activations that want it below."""
                    log_gate_last = log_gate_bcast[:, CHUNK - 1 : CHUNK]

                    """M[r, s] = exp(log_G[r] - log_G[s]) for r >= s, strict upper masked out,
                    as ONE scalar_tensor_tensor. `reverse0=True` makes `operand0` the LEFT
                    operand, so the subtraction is (log_gate_col - log_gate_bcast): row index r
                    from the column, column index s from the spread. That order is what keeps
                    the exponent <= 0 on the retained triangle -- the gates are negative and
                    log_G is their prefix sum, so log_G[r] <= log_G[s] for r >= s."""
                    log_diff = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=log_diff,
                        data=log_gate_bcast,
                        op0=nl.subtract,
                        operand0=log_gate_col,
                        reverse0=True,
                        op1=nl.add,
                        operand1=upper_neg,
                    )
                    decay_matrix = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=decay_matrix, op=nl.exp, data=log_diff)

                    # A = beta * M * (k^T k), strictly lower triangular.
                    key_gram = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=key_gram, stationary=k_tile, moving=k_tile)
                    decay_gram = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=decay_gram, data1=decay_matrix, data2=key_gram, op=nl.multiply)
                    interaction = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=interaction,
                        data=decay_gram,
                        op0=nl.multiply,
                        operand0=beta_col,
                        op1=nl.multiply,
                        operand1=tril_strict,
                    )

                    """THE GATE COLUMNS, all (CHUNK, 1) and all state-independent. Each is
                    consumed as a per-partition operand in stage 3, and each is fused with
                    whatever multiply it would otherwise need a separate pass for:
                    `beta*G` and `beta*v` make stage 3's b_vec one instruction, `scale*G`
                    carries the query scale so `state` needs only one cast, and
                    `G_last/G` carries the state update's decay ratio."""
                    gate_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=gate_col, op=nl.exp, data=log_gate_col)
                    beta_gate_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=beta_gate_col, data1=gate_col, data2=beta_col, op=nl.multiply)
                    gate_scaled_col = gate_col
                    if scale != 1.0:
                        gate_scaled_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=gate_scaled_col, data=gate_col, op0=nl.multiply, operand0=scale)
                    # ratio[s] = exp(log_G_last - log_G[s]) as ONE activation.
                    ratio_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=ratio_col, op=nl.exp, data=log_gate_col, bias=log_gate_last, scale=-1.0)
                    gate_last_col = nl.ndarray((key_head_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=gate_last_col, op=nl.exp, data=log_gate_last[0:key_head_dim, :])
                    # beta*v, the state-independent half of stage 3's b_vec.
                    beta_value = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=beta_value, data=v_tile, op0=nl.multiply, operand0=beta_col)

                    """W_T = (M * (q^T k))^T -- the output's intra-chunk operand. QK stays
                    in PSUM for the multiply to read, and THE QUERY SCALE RIDES ON THE
                    TRANSPOSE'S EVACUATION. Putting it here rather than on `delta` is what
                    lets stage 3 feed one unweighted `delta` to both of its matmuls."""
                    query_key = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=query_key, stationary=q_tile, moving=k_tile)
                    weighted_qk = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=weighted_qk, data1=decay_matrix, data2=query_key, op=nl.multiply)
                    w_t = transpose_tile(weighted_qk, CHUNK, CHUNK, out_dtype=compute_dtype, scale=scale)

                    """THE DECAY RATIO RIDES ON k, not on delta: it is indexed by the token,
                    which is the state-update matmul's contraction index, so it can sit on
                    either operand -- and this one is state-independent, which takes a whole
                    instruction off the recurrence's critical path."""
                    key_t = nl.ndarray((CHUNK, key_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=key_t,
                        data=key_token_major,
                        op0=nl.multiply,
                        operand0=ratio_col,
                        engine=nisa.engine.scalar,
                    )

                    key_tiles.append(k_tile)
                    key_t_tiles.append(key_t)
                    query_tiles.append(q_tile)
                    beta_value_tiles.append(beta_value)
                    beta_gate_tiles.append(beta_gate_col)
                    gate_scaled_tiles.append(gate_scaled_col)
                    gate_last_tiles.append(gate_last_col)
                    interaction_tiles.append(interaction)
                    w_t_tiles.append(w_t)

                """LEVEL 0 of the merge, emitted LEVEL-MAJOR like stage 2 so the group's
                same-kind operations issue back to back. It produces Inv rather than Inv_T
                -- the transpose that follows then hands stage 2's first level both
                orientations, so that level pays for no transpose of its own."""
                # Block size 1 is PURE ELEMENTWISE: Inv == I there, so the merge reduces
                # to Inv = I - (A * off0).
                masked_tiles = []
                for member_idx in range(n_group_chunks):
                    masked = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=masked, data1=interaction_tiles[member_idx], data2=off0, op=nl.multiply)
                    masked_tiles.append(masked)
                for member_idx in range(n_group_chunks):
                    inv = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=inv, data1=eye_mat, data2=masked_tiles[member_idx], op=nl.subtract)
                    inv_tiles.append(inv)
                for member_idx in range(n_group_chunks):
                    inv_t_tiles.append(transpose_tile(inv_tiles[member_idx], CHUNK, CHUNK))

                """STAGE 2: block-merge levels, LEVEL-MAJOR over the group. Each
                sub-loop emits this group's `n_group_chunks` same-kind operations back to back
                with no dependency between them; do NOT merge these sub-loops into a
                per-chunk loop."""
                for level_idx in range(len(neg_off_masks)):
                    neg_off_mask = neg_off_masks[level_idx]
                    if level_idx > 0:
                        # Inv = P^-1, derived from the carried Inv_T rather than stored.
                        # Level 0 above already left it here.
                        inv_tiles = []
                        for member_idx in range(n_group_chunks):
                            inv_tiles.append(transpose_tile(inv_t_tiles[member_idx], CHUNK, CHUNK))
                    r_tiles = []
                    for member_idx in range(n_group_chunks):
                        """R = -(A restricted to this level's off-blocks); the negation
                        rides in the host mask, and A stands in for I + A because no off
                        mask ever touches the diagonal."""
                        r_tile = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.tensor_tensor(
                            dst=r_tile, data1=interaction_tiles[member_idx], data2=neg_off_mask, op=nl.multiply
                        )
                        r_tiles.append(r_tile)
                    mid_tiles = []
                    for member_idx in range(n_group_chunks):
                        # Mid = -R^T @ (Q^-1)^T
                        mid_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=mid_psum, stationary=r_tiles[member_idx], moving=inv_t_tiles[member_idx])
                        mid_tile = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=mid_tile, src=mid_psum, engine=COPY_ENGINE)
                        mid_tiles.append(mid_tile)
                    off_t_tiles = []
                    for member_idx in range(n_group_chunks):
                        # Off_T = (P^-1)^T @ Mid, LEFT IN PSUM and consumed straight
                        # from there by the accumulate below.
                        off_t_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=off_t_psum, stationary=inv_tiles[member_idx], moving=mid_tiles[member_idx])
                        off_t_tiles.append(off_t_psum)
                    for member_idx in range(n_group_chunks):
                        """Un-masked scatter-add, writing the compute dtype directly so
                        the accumulate and the operand format for the next level are ONE
                        stage."""
                        inv_t_next = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.tensor_tensor(
                            dst=inv_t_next, data1=inv_t_tiles[member_idx], data2=off_t_tiles[member_idx], op=nl.add
                        )
                        inv_t_tiles[member_idx] = inv_t_next

                for member_idx in range(n_group_chunks):
                    chunk_idx = chunk_base + member_idx
                    k_tile = key_tiles[member_idx]
                    q_tile = query_tiles[member_idx]

                    """b_vec = beta*v - (beta*G) * (k^T state), as ONE instruction: both
                    beta products are chunk-local and were hoisted into stage 1, and this
                    writes the MATMUL OPERAND dtype straight out of PSUM, so neither the
                    fp32 intermediate nor the cast that followed it exists any more."""
                    key_state = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=key_state, stationary=k_tile, moving=state_operand)
                    b_vec_operand = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=b_vec_operand,
                        data=key_state,
                        op0=nl.multiply,
                        operand0=beta_gate_tiles[member_idx],
                        op1=nl.subtract,
                        operand1=beta_value_tiles[member_idx],
                        reverse1=True,
                    )

                    # delta = (I + A)^-1 b_vec. Inv_T is exactly (I + A)^-T here and is
                    # ALREADY the matmul operand dtype, so no cast sits on this read.
                    delta_psum = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=delta_psum, stationary=inv_t_tiles[member_idx], moving=b_vec_operand)
                    """ONE evacuation of delta, feeding both matmuls below. The per-token weight each of
                    them wants -- the query scale, the decay ratio -- rides on the OTHER operand instead,
                    and both of those are state-independent."""
                    delta_operand = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.activation(dst=delta_operand, op=nl.copy, data=delta_psum)

                    # o = (scale*G) * (q^T state) + (scale*W_T)^T @ delta, per the transpose
                    # convention in the Pseudocode: the matmul contracts its stationary operand.
                    intra_chunk_psum = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=intra_chunk_psum, stationary=w_t_tiles[member_idx], moving=delta_operand)
                    intra_chunk = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=intra_chunk, src=intra_chunk_psum, engine=COPY_ENGINE)
                    query_state = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=query_state, stationary=q_tile, moving=state_operand)
                    chunk_out = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=chunk_out,
                        data=query_state,
                        op0=nl.multiply,
                        operand0=gate_scaled_tiles[member_idx],
                        op1=nl.add,
                        operand1=intra_chunk,
                    )

                    # state = G_last * state + ((G_last/G) * k)^T @ delta, from PSUM.
                    contribution = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=contribution, stationary=key_t_tiles[member_idx], moving=delta_operand)
                    state_next = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=state_next,
                        data=state,
                        op0=nl.multiply,
                        operand0=gate_last_tiles[member_idx],
                        op1=nl.add,
                        operand1=contribution,
                    )
                    """The same value again, narrow: the next chunk's matmul operand. Written from
                    `state` and `contribution` so it does not wait on `state_next`."""
                    state_operand = nl.ndarray((key_head_dim, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.scalar_tensor_tensor(
                        dst=state_operand,
                        data=state,
                        op0=nl.multiply,
                        operand0=gate_last_tiles[member_idx],
                        op1=nl.add,
                        operand1=contribution,
                    )
                    state = state_next

                    nisa.dma_copy(
                        dst=out_view[batch_idx, v_head_idx, chunk_idx, :, :],
                        src=chunk_out,
                        dge_mode=nisa.dge_mode.none,
                    )

        nl.store(recurrent_state[batch_idx, v_head_idx, :, :], state)

    return out, recurrent_state
