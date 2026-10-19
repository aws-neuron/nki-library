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

"""GDN chunked prefill BACKWARD (schur decomposition).

Reverse-mode VJP of `gdn_cte_schur`, derived primitive by primitive in `_backward_chunk`'s
docstring and validated against `gdn_cte_bwd_torch.py`. The forward chunk recurrence is:

    log_G = cumsum(g_log);  G = exp(log_G);  G_last = G[-1]
    M     = tril(exp(log_G[:,None] - log_G[None,:]))
    KK    = K K^T ;  QK = Q K^T
    A     = tril_-1(beta * M * KK) ;  T = (I + A)^-1
    KS0   = K S0
    b     = beta*V - (beta*G) KS0
    U     = T b
    QS0   = Q S0
    O     = scale (G*QS0 + (M*QK) U)
    S_new = G_last S0 + Kdec^T U,  Kdec = (G_last/G) K

The backward recomputes these per chunk (cheaper than stashing the O(C^2) tiles) and applies
the VJP in `_backward_chunk`. Two passes over the sequence:

  FORWARD STATE PASS   run the recurrence to recover each chunk's INCOMING state S0, stashing
                       the small (Dk, Dv) state per chunk in SBUF (NOT the C^2 intermediates).
  BACKWARD SCAN        reverse-time c = n-1 .. 0, carrying dS_new; recompute the chunk's
                       intermediates from its stashed S0 and emit dQ/dK/dV/dg/dbeta and dS0.

Recompute is what keeps the O(C^2) tiles out of the stash. The alternative -- fuse forward and
backward and keep each chunk's T^T and U SBUF-resident across the boundary -- would drop the
recompute entirely, but the reverse scan consumes chunks in the opposite order to the forward,
so it has to hold every chunk's O(C^2) tiles live at once. That is impossible past a few chunks,
hence the two passes above: stash only the (Dk, Dv) state, pay stage 1 + the schur merge twice.

The stash is still LINEAR in S, and it -- not the group footprint -- is what bounds the sequence
length: one (Dk, n_chunks*Dv) fp32 tile costs 4*n_chunks*Dv bytes on every partition, and on trn2
that consumes the whole 212,984 B/partition SBUF at S = 53248, Dv = 128.
`MAX_STATE_STASH_BYTES_PER_PARTITION` caps it a little below the measured wall (S <= 49152 at
Dv = 128, proportionally more at smaller Dv) so the kernel fails the shape up front with the
arithmetic rather than bottoming out in an allocation failure that names no cause. Past that,
move the stash to HBM: a DMA per chunk in each pass, and no SBUF cost at all.

LEVEL-MAJOR GROUPING. Both passes are emitted in GROUPS of `group_size` chunks, mirroring the
forward: the STATE-INDEPENDENT recompute (stage 1 -- gates/M/KK/QK/A/w_qk -- and the schur
merge, `_schur_group`) runs level-major across the group so the per-level matmul dependency
stall amortizes, while the state / state-gradient recurrences stay serial over the group. The
forward pass walks the groups in forward order; the backward scan walks them in reverse, which
still visits chunks in strictly-descending order, so the recurrence -- and the whole output --
is bit-identical to group_size=1.

The schur inverse (I+A)^-1 is formed exactly as the forward does (recursive block merge), and
the backward reuses both orientations T^T (=Inv_T) and T it produces:
    db = T^T dU ,  dA = tril_-1(-(T^T dT T^T)).

TOOLCHAIN FLOOR. `_backward_chunk` forms its seven feature-axis dot products with
`nisa.scalar_tensor_tensor`'s fused `reduce_res`, which nki gained in 0.7.0b1. On 0.6.0 the call
fails to trace on the unexpected keyword; there is no 0.6.0-compatible spelling, as that release
carries the fused reduce only on `tensor_scalar_reduce` and `activation`, neither of which takes
two tensor operands.

NUMERICS. The compute dtype is `query.dtype`, as in the forward; the recurrent state and all
gradient accumulation stay fp32. GQA: value head h shares key/query head h // (Hv//Hk); the
kernel shards over (batch x KEY head) so every dQ/dK slice is owned by one program, and the
head-repeat's adjoint (a sum over the shared group's value heads) is done by accumulating each
value head's dQ/dK into the shared slice.
"""

from typing import Any, Optional

import nki
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

DEFAULT_GROUP_SIZE = 8
# Upper bound on `group_size`, an SBUF/PSUM bound. The backward holds a full per-member ch dict
# (~7 CxC tiles: decay, KK, QK, A, w_qk, Inv_T, Inv) LIVE across the whole group -- more than the
# forward's stage-2 peak -- so the footprint grows with `group_size` at that multiple. Kept at
# the forward's ceiling; clamp lower here if a shape spills.
MAX_GROUP_SIZE = 16

MAX_STATE_STASH_BYTES_PER_PARTITION = 192 * 1024
"""SBUF budget for the forward state pass's per-chunk stash, which is one (Dk, n_chunks*Dv) fp32
tile -- 4*n_chunks*Dv bytes on EVERY partition, growing linearly in seqlen. It is by far the
largest consumer, so it, not the group's tiles, is what bounds the sequence length.

MEASURED on trn2 (SBUF is 212,984 B/partition): at Dv=128 the kernel compiles at S=52224 and
fails at S=53248, where the stash alone (212,992 B) exceeds SBUF. So everything else in flight
-- the group's CxC tiles at any `group_size` up to MAX_GROUP_SIZE, and the chunk working set --
fits in the remaining 4,088 B, the allocator having overlapped it against the group's liveness.
The wall did not move between bf16 and fp32, nor between group_size 8 and 16.

The budget sits at 192 KB rather than at that wall: 16,376 B of slack is 4x the remainder the
kernel needs today, so adding a tile later costs headroom rather than silently turning this
assert back into an opaque allocation failure. At Dv=128 it admits S <= 49152 (validated
end-to-end on trn2 against the autograd oracle at 32768 and 49152); smaller Dv scales up
proportionally. Past that, move the stash to HBM -- a DMA per chunk in each pass.
"""


@nki.jit
def gdn_cte_bwd(
    query: nl.NkiTensor,
    key: nl.NkiTensor,
    value: nl.NkiTensor,
    g_log: nl.NkiTensor,
    beta: nl.NkiTensor,
    d_out: nl.NkiTensor,
    mask_pack: nl.NkiTensor,
    scale: float = 1.0,
    group_size: int = DEFAULT_GROUP_SIZE,
    recurrent_state: Optional[nl.NkiTensor] = None,
    d_state: Optional[nl.NkiTensor] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Backward pass of the chunked gated delta-rule (schur) prefill kernel.

    Reverse-mode VJP of `gdn_cte_schur`, in two passes over the sequence: a forward state pass
    that stashes each chunk's incoming state, then a reverse-time scan that recomputes the chunk
    intermediates from it and emits every gradient. Requires S a multiple of CHUNK=128 and head
    dims at most MAX_HEAD_DIM=128. S is additionally bounded by the state stash, which is LINEAR
    in S: at Dv=128 the kernel admits S <= 49152, proportionally more at smaller Dv (see
    `MAX_STATE_STASH_BYTES_PER_PARTITION`) -- a bound the forward does not have. Costs roughly
    3.1-3.3x the forward's device time at the same shape.

    Dimensions:
        B: Batch size
        Hk: Key/query head count
        Hv: Value head count, a multiple of Hk (GQA ratio Hv // Hk)
        Dk: Key/query/state head dimension (<= MAX_HEAD_DIM=128)
        Dv: Value/state head dimension (<= MAX_HEAD_DIM=128, and independent of Dk)
        S: Sequence length (must be a multiple of CHUNK)
        CHUNK: Chunk length (128)
        GROUP: Chunks per grouped recompute (`group_size`), plus a residual group of
            S/CHUNK mod GROUP when the split is not exact

    Args:
        query, key (nl.NkiTensor): [B, Hk, S, Dk], TOKEN-major. `query.dtype` IS THE COMPUTE
            DTYPE for the whole kernel, and `key` must carry the same dtype -- both are DMA'd
            into compute-dtype tiles.
        value (nl.NkiTensor): [B, Hv, S, Dv], same dtype as `query`.
        g_log, beta (nl.NkiTensor): [B, Hv, S], fp32.
        d_out (nl.NkiTensor): [B, Hv, S, Dv], dL/d out, fp32 -- it is consumed in fp32 and the
            gate/beta gradients it feeds are fp32 outputs, so it does NOT follow the compute
            dtype. A bf16 upstream gradient must be cast by the caller.
        mask_pack (nl.NkiTensor): [N_MASKS, CHUNK, CHUNK] fp32, from build_mask_pack().
        scale (float): query output scale.
        group_size (int): chunks per grouped inverse. The state-independent recompute (stage 1
            + the schur merge) is emitted LEVEL-MAJOR across a group of this many chunks in both
            passes, so per-level dependency stalls amortize; the state / state-gradient
            recurrences stay serial. Scheduling only -- output is bit-identical to group_size=1.
        recurrent_state (Optional[nl.NkiTensor]): [B, Hv, Dk, Dv] fp32 initial state, or None.
        d_state (Optional[nl.NkiTensor]): [B, Hv, Dk, Dv] fp32 dL/d final state, or None (zero).

    Returns:
        (d_query [B,Hk,S,Dk], d_key [B,Hk,S,Dk], d_value [B,Hv,S,Dv], d_g_log [B,Hv,S],
         d_beta [B,Hv,S], d_recurrent_state [B,Hv,Dk,Dv]). d_query, d_key and d_value carry the
         COMPUTE DTYPE; d_g_log, d_beta and d_recurrent_state are always fp32, as their inputs
         are. ALWAYS six outputs, as the forward always returns (out, state_out): dL/dS0 is well
         defined even when the forward started from an implicit zero state, and callers that did
         not supply `recurrent_state` simply ignore the last one.

    Pseudocode:
        # Every matmul below contracts its STATIONARY operand: nc_matmul(stationary=S, moving=M)
        # computes S^T @ M, so each product is written with that transpose explicit.
        # group_width = min(group_size, n_chunks) chunks at a time, then one residual group of
        # n_chunks % group_width if any
        for lane in this shard's (batch x KEY head) lanes:     # SPMD over batch x key head
            for v_head in the key head's Hv // Hk value heads:
                # PASS 1, FORWARD ORDER: recover every chunk's INCOMING state
                state = recurrent_state or 0
                for group in groups:
                    log_G = cumsum(g_log[group])               # one prefix sum per GROUP
                    for member: ch[member] = stage1(...)       # load + M + KK + QK + A + w_qk
                    schur_group(ch, need_inv=False)            # T^T, LEVEL-MAJOR over the group
                    for member in 0 .. n-1:                    # serial: the state recurrence
                        states_sbuf[chunk] = state             # the stash, (Dk, n_chunks*Dv)
                        b = beta*V - (beta*G) (K state);  U = Inv_T^T @ b
                        state = G_last * state + ((G_last/G) * K)^T @ U
                # PASS 2, REVERSE ORDER: dS flows backward, chunks strictly descending
                d_state_cur = d_state or 0
                for group in reversed(groups):
                    log_G = cumsum(g_log[group])               # recomputed, never stashed
                    for member: ch[member] = stage1(...)       # same recompute as pass 1
                    schur_group(ch, need_inv=True)             # BOTH T^T and T
                    for member in n-1 .. 0:                    # serial: the dS recurrence
                        S0 = states_sbuf[chunk]                # from pass 1
                        U = Inv_T^T @ b                        # recomputed from S0
                        # emits dQ/dK/dV/dg_log/dbeta, and UPDATES d_state_cur IN PLACE to dS0
                        backward_chunk(ch, S0, U, d_out[chunk], d_state_cur)
                d_recurrent_state[batch, v_head] = d_state_cur  # the carry out of chunk 0
    """
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]

    kernel_assert(
        key_head_dim == key.shape[-1],
        f"query and key must have the same head dim, got {key_head_dim} and {key.shape[-1]}",
    )
    kernel_assert(
        key_head_dim <= MAX_HEAD_DIM and value_head_dim <= MAX_HEAD_DIM,
        f"head dims must be at most {MAX_HEAD_DIM}, got {key_head_dim} and {value_head_dim}",
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
    """g_log/beta/d_out and both state tensors are consumed in fp32 (the gate, beta and state
    gradients are fp32 outputs), so unlike q/k/v they do NOT follow the compute dtype. Caught
    here because the alternative is a dtype-mismatched dma_copy deep inside the scan."""
    kernel_assert(g_log.dtype == nl.float32, f"g_log must be float32, got {g_log.dtype}")
    kernel_assert(beta.dtype == nl.float32, f"beta must be float32, got {beta.dtype}")
    kernel_assert(d_out.dtype == nl.float32, f"d_out must be float32, got {d_out.dtype}")
    if recurrent_state is not None:
        kernel_assert(
            recurrent_state.dtype == nl.float32, f"recurrent_state must be float32, got {recurrent_state.dtype}"
        )
    if d_state is not None:
        kernel_assert(d_state.dtype == nl.float32, f"d_state must be float32, got {d_state.dtype}")

    state_stash_bytes = 4 * (seqlen // CHUNK) * value_head_dim
    kernel_assert(
        state_stash_bytes <= MAX_STATE_STASH_BYTES_PER_PARTITION,
        f"seqlen={seqlen} at value_head_dim={value_head_dim} needs {state_stash_bytes} B/partition for the "
        f"forward pass's state stash, over the {MAX_STATE_STASH_BYTES_PER_PARTITION} B budget; at this head dim "
        f"seqlen must be at most {MAX_STATE_STASH_BYTES_PER_PARTITION // (4 * value_head_dim) * CHUNK}",
    )

    v_heads_per_k_head = n_v_heads // n_k_heads
    n_chunks = seqlen // CHUNK
    compute_dtype = query.dtype
    state_is_input = recurrent_state is not None
    has_d_state = d_state is not None

    d_query = nl.ndarray((batch, n_k_heads, seqlen, key_head_dim), dtype=compute_dtype, buffer=nl.shared_hbm)
    d_key = nl.ndarray((batch, n_k_heads, seqlen, key_head_dim), dtype=compute_dtype, buffer=nl.shared_hbm)
    d_value = nl.ndarray((batch, n_v_heads, seqlen, value_head_dim), dtype=compute_dtype, buffer=nl.shared_hbm)
    d_g_log = nl.ndarray((batch, n_v_heads, seqlen), dtype=nl.float32, buffer=nl.shared_hbm)
    d_beta = nl.ndarray((batch, n_v_heads, seqlen), dtype=nl.float32, buffer=nl.shared_hbm)
    d_recurrent_state = nl.ndarray(
        (batch, n_v_heads, key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    query_view = query.reshape((batch, n_k_heads, n_chunks, CHUNK, key_head_dim))
    key_view = key.reshape((batch, n_k_heads, n_chunks, CHUNK, key_head_dim))
    value_view = value.reshape((batch, n_v_heads, n_chunks, CHUNK, value_head_dim))
    d_out_view = d_out.reshape((batch, n_v_heads, n_chunks, CHUNK, value_head_dim))
    g_log_row_view = g_log.reshape((batch, n_v_heads, n_chunks, CHUNK))
    beta_view = beta.reshape((batch, n_v_heads, n_chunks, CHUNK, 1))
    d_query_view = d_query.reshape((batch, n_k_heads, n_chunks, CHUNK, key_head_dim))
    d_key_view = d_key.reshape((batch, n_k_heads, n_chunks, CHUNK, key_head_dim))
    d_value_view = d_value.reshape((batch, n_v_heads, n_chunks, CHUNK, value_head_dim))
    d_g_log_view = d_g_log.reshape((batch, n_v_heads, n_chunks, 1, CHUNK))
    d_beta_view = d_beta.reshape((batch, n_v_heads, n_chunks, CHUNK, 1))

    upper_neg = _load_mask(mask_pack, IDX_UPPER_NEG)
    tril_strict = _load_mask(mask_pack, IDX_TRIL_STRICT)
    eye_mat = _load_mask(mask_pack, IDX_EYE)
    off0 = _load_mask(mask_pack, IDX_OFF0)
    neg_off_masks = []
    for level in range(N_MERGE_LEVELS - 1):
        neg_off_masks.append(_load_mask(mask_pack, IDX_NEG_OFF + level))
    masks = (upper_neg, tril_strict, eye_mat, off0, neg_off_masks)

    """Loop-invariant reduction operands (a matmul against a ones column is a free-axis sum).
    Hoisted out of `_backward_chunk`: they are constants, and memsetting them there costs two
    ops per chunk per value head per lane."""
    ones_c = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_c, value=1.0)
    ones_dk = nl.ndarray((key_head_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_dk, value=1.0)

    """GQA dQ/dK ACCUMULATOR, fp32. With Hv > Hk the head repeat's adjoint sums
    `v_heads_per_k_head` partial dQ/dK into ONE (q, k) slice. Accumulating in the compute dtype
    would round every partial before the add, and the error grows with the head ratio -- worst
    when the contributions partly cancel -- so the running sum lives in an fp32 HBM scratch and
    only the final total is rounded into the compute-dtype output. Dense (Hv == Hk) has a
    one-term sum and skips the scratch entirely, writing each chunk straight out."""
    is_gqa = v_heads_per_k_head > 1
    if is_gqa:
        d_query_acc = nl.ndarray(
            (batch, n_k_heads, n_chunks, CHUNK, key_head_dim), dtype=nl.float32, buffer=nl.private_hbm
        )
        d_key_acc = nl.ndarray(
            (batch, n_k_heads, n_chunks, CHUNK, key_head_dim), dtype=nl.float32, buffer=nl.private_hbm
        )

    # SPMD sharding over (batch x KEY head) so each dQ/dK slice is owned by one program.
    n_shards = nl.num_programs(axes=0)
    shard_id = nl.program_id(axis=0)
    n_lanes = batch * n_k_heads
    lanes_per_shard, extra_lanes = divmod(n_lanes, n_shards)
    if shard_id < extra_lanes:
        lanes_per_shard += 1
    lane_start = shard_id * (n_lanes // n_shards) + min(shard_id, extra_lanes)

    for lane_idx in range(lane_start, lane_start + lanes_per_shard):
        batch_idx, k_head_idx = divmod(lane_idx, n_k_heads)

        for vh_in_group in range(v_heads_per_k_head):
            v_head_idx = k_head_idx * v_heads_per_k_head + vh_in_group
            is_first_vhead = vh_in_group == 0

            """Group layout, matching the forward: `n_full_groups` groups of `group_width`, then
            ONE residual group of the remainder. The forward state pass walks these in FORWARD
            order (state flows forward); the backward scan walks them in REVERSE (dS flows
            backward), which visits chunks in the same strictly-descending order as a plain
            reverse loop -- so the recurrence is bit-identical to group_size=1."""
            group_width, n_full_groups, residual = group_layout(n_chunks, group_size)
            group_widths = [group_width]
            group_counts = [n_full_groups]
            group_offsets = [0]
            if residual:
                group_widths.append(residual)
                group_counts.append(1)
                group_offsets.append(n_full_groups * group_width)

            # ---- FORWARD STATE PASS (grouped): stash each chunk's incoming state ----
            states_sbuf = nl.ndarray((key_head_dim, n_chunks * value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
            if state_is_input:
                state = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=state, src=recurrent_state[batch_idx, v_head_idx, :, :], dge_mode=nisa.dge_mode.none)
            else:
                state = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=state, value=0.0)

            for schedule_idx in range(len(group_widths)):
                n_group_chunks = group_widths[schedule_idx]
                chunk_offset = group_offsets[schedule_idx]
                for group_idx in nl.sequential_range(group_counts[schedule_idx]):
                    chunk_base = chunk_offset + group_idx * n_group_chunks
                    # Stage 1 for every member, then the schur inverse LEVEL-MAJOR over the group.
                    log_gate_cols = _group_log_gate_cols(
                        g_log_row_view, batch_idx, v_head_idx, chunk_base, n_group_chunks
                    )
                    ch_list = []
                    for member in range(n_group_chunks):
                        ch_list.append(
                            _stage1(
                                query_view,
                                key_view,
                                value_view,
                                beta_view,
                                log_gate_cols[:, member : member + 1],
                                batch_idx,
                                k_head_idx,
                                v_head_idx,
                                chunk_base + member,
                                masks,
                                compute_dtype,
                                value_head_dim,
                                key_head_dim,
                            )
                        )
                    _schur_group(ch_list, masks, compute_dtype, need_inv=False)
                    # State recurrence: serial over the group, member 0 .. n-1.
                    for member in range(n_group_chunks):
                        chunk_idx = chunk_base + member
                        ch = ch_list[member]
                        nisa.tensor_copy(
                            dst=states_sbuf[:, chunk_idx * value_head_dim : (chunk_idx + 1) * value_head_dim],
                            src=state,
                        )
                        state_op = nl.ndarray((key_head_dim, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=state_op, src=state)
                        ks = _key_state(ch["k_f"], state_op, value_head_dim)
                        b_op = _b_vec(ch, ks, value_head_dim, compute_dtype)
                        u_op = _deltas(ch, b_op, value_head_dim, compute_dtype)
                        # Kdec = (G_last/G) K, token-major
                        k_decay = nl.ndarray((CHUNK, key_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=k_decay, data=ch["k_tm"], op0=nl.multiply, operand0=ch["ratio_col"])
                        contribution = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=contribution, stationary=k_decay, moving=u_op)
                        state_next = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.scalar_tensor_tensor(
                            dst=state_next,
                            data=state,
                            op0=nl.multiply,
                            operand0=ch["gate_last_col"],
                            op1=nl.add,
                            operand1=contribution,
                        )
                        state = state_next

            # ---- BACKWARD SCAN (grouped): reverse-time c = n-1 .. 0 ----
            d_state_cur = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
            if has_d_state:
                nisa.dma_copy(dst=d_state_cur, src=d_state[batch_idx, v_head_idx, :, :], dge_mode=nisa.dge_mode.none)
            else:
                nisa.memset(dst=d_state_cur, value=0.0)

            for schedule_idx in reversed(range(len(group_widths))):
                n_group_chunks = group_widths[schedule_idx]
                chunk_offset = group_offsets[schedule_idx]
                n_groups = group_counts[schedule_idx]
                for group_idx in nl.sequential_range(n_groups):
                    # Reverse the group order so chunks descend; members descend within a group.
                    chunk_base = chunk_offset + (n_groups - 1 - group_idx) * n_group_chunks
                    # Stage 1 for every member, then the schur inverse LEVEL-MAJOR over the group.
                    log_gate_cols = _group_log_gate_cols(
                        g_log_row_view, batch_idx, v_head_idx, chunk_base, n_group_chunks
                    )
                    ch_list = []
                    for member in range(n_group_chunks):
                        ch_list.append(
                            _stage1(
                                query_view,
                                key_view,
                                value_view,
                                beta_view,
                                log_gate_cols[:, member : member + 1],
                                batch_idx,
                                k_head_idx,
                                v_head_idx,
                                chunk_base + member,
                                masks,
                                compute_dtype,
                                value_head_dim,
                                key_head_dim,
                            )
                        )
                    _schur_group(ch_list, masks, compute_dtype, need_inv=True)
                    # State-gradient recurrence: serial over the group, member n-1 .. 0.
                    for member in reversed(range(n_group_chunks)):
                        chunk_idx = chunk_base + member
                        ch = ch_list[member]
                        s0 = nl.ndarray((key_head_dim, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(
                            dst=s0, src=states_sbuf[:, chunk_idx * value_head_dim : (chunk_idx + 1) * value_head_dim]
                        )
                        s0_op = nl.ndarray((key_head_dim, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=s0_op, src=s0)
                        ks = _key_state(ch["k_f"], s0_op, value_head_dim)
                        b_op = _b_vec(ch, ks, value_head_dim, compute_dtype)
                        u_op = _deltas(ch, b_op, value_head_dim, compute_dtype)

                        _backward_chunk(
                            ch=ch,
                            s0=s0,
                            s0_op=s0_op,
                            key_state=ks,
                            b_op=b_op,
                            u_op=u_op,
                            d_out_src=d_out_view[batch_idx, v_head_idx, chunk_idx, :, :],
                            d_state_cur=d_state_cur,
                            scale=scale,
                            masks=masks,
                            ones_c=ones_c,
                            ones_dk=ones_dk,
                            compute_dtype=compute_dtype,
                            value_head_dim=value_head_dim,
                            key_head_dim=key_head_dim,
                            d_query_dst=d_query_view[batch_idx, k_head_idx, chunk_idx, :, :],
                            d_key_dst=d_key_view[batch_idx, k_head_idx, chunk_idx, :, :],
                            d_value_dst=d_value_view[batch_idx, v_head_idx, chunk_idx, :, :],
                            d_g_dst=d_g_log_view[batch_idx, v_head_idx, chunk_idx, :, :],
                            d_beta_dst=d_beta_view[batch_idx, v_head_idx, chunk_idx, :, :],
                            acc_query_dst=(d_query_acc[batch_idx, k_head_idx, chunk_idx, :, :] if is_gqa else None),
                            acc_key_dst=(d_key_acc[batch_idx, k_head_idx, chunk_idx, :, :] if is_gqa else None),
                            is_first_vhead=is_first_vhead,
                            is_last_vhead=vh_in_group == v_heads_per_k_head - 1,
                        )

            """Stored unconditionally: dL/dS0 is the reverse scan's carry out of chunk 0, which
            the loop computes whether or not S0 came in as an argument. Returning it on both
            paths keeps the arity fixed, as the forward's (out, state_out) does."""
            nisa.dma_copy(
                dst=d_recurrent_state[batch_idx, v_head_idx, :, :],
                src=d_state_cur,
                dge_mode=nisa.dge_mode.none,
            )

    return d_query, d_key, d_value, d_g_log, d_beta, d_recurrent_state


def _load_mask(mask_pack: nl.NkiTensor, plane_idx: int) -> nl.NkiTensor:
    """One mask plane, HBM -> SBUF, in the pack's own dtype.

    `nisa.dma_copy` rather than `nl.load` so that every transfer in the kernel pins `dge_mode`
    explicitly: the descriptors for these are built before execution, as they are for every other
    DMA here. Ten loads once per launch, so the mode is inert either way -- this is uniformity, not
    performance.

    At MODULE level, not nested in `gdn_cte_bwd` where it is used: a helper defined inside the
    `@nki.jit` body and called directly is rejected at trace time (`inner functions can only be
    used as fori_loop/while_loop body arguments`), so every helper this kernel calls lives out here.

    Args:
        mask_pack (nl.NkiTensor): [N_MASKS, CHUNK, CHUNK] constant pack from build_mask_pack().
        plane_idx (int): which plane (one of the `IDX_*` constants).

    Returns:
        nl.NkiTensor: (CHUNK, CHUNK) SBUF tile holding that plane.
    """
    tile = nl.ndarray((CHUNK, CHUNK), dtype=mask_pack.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=mask_pack[plane_idx, :, :], dge_mode=nisa.dge_mode.none)
    return tile


def _schur_group(ch_list: list, masks: tuple, compute_dtype: Any, need_inv: bool) -> None:
    """Form (I + A)^-1 for a GROUP of chunks, emitted LEVEL-MAJOR like the forward.

    Same per-member op sequence as the original per-chunk `_schur_inverse` (recursive block
    merge: elementwise level 0, then one matmul pair per merge level), but the merge LEVEL is
    the outer loop and the group member the inner one, so each level's `group_size`
    independent matmuls issue back to back and the per-level dependency stall amortizes across
    the group. This is a pure scheduling transform -- output is bit-identical to group_size=1.

    Leaves ``ch["inv_t"]`` (T^T) valid for every member. ``ch["inv"]`` is written on every level
    either way, so the key ALWAYS exists -- but it holds T only when ``need_inv``; without it the
    value left behind is the transpose of the second-to-last level's partial inverse, which no
    caller may read. The forward state pass reads only ``inv_t``; the backward scan needs both.

    Args:
        ch_list (list): the per-member ch dicts from `_stage1` (must hold ``interaction``).
        masks (tuple): the shared mask tuple.
        compute_dtype (Any): matmul-operand dtype.
        need_inv (bool): also derive T (the backward needs both orientations; the forward only T^T).
    """
    _, _, eye_mat, off0, neg_off_masks = masks
    n_group_chunks = len(ch_list)

    # Level 0: pure elementwise (Inv == I at block size 1), builds Inv then its transpose.
    for member_idx in range(n_group_chunks):
        masked = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=masked, data1=ch_list[member_idx]["interaction"], data2=off0, op=nl.multiply)
        inv = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=inv, data1=eye_mat, data2=masked, op=nl.subtract)
        ch_list[member_idx]["inv"] = inv
    for member_idx in range(n_group_chunks):
        ch_list[member_idx]["inv_t"] = transpose_tile(ch_list[member_idx]["inv"], CHUNK, CHUNK)

    # Stage 2: remaining merge levels, LEVEL-MAJOR over the group.
    for level_idx in range(len(neg_off_masks)):
        neg_off_mask = neg_off_masks[level_idx]
        if level_idx > 0:
            for member_idx in range(n_group_chunks):
                ch_list[member_idx]["inv"] = transpose_tile(ch_list[member_idx]["inv_t"], CHUNK, CHUNK)
        r_tiles = []
        for member_idx in range(n_group_chunks):
            r_tile = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r_tile, data1=ch_list[member_idx]["interaction"], data2=neg_off_mask, op=nl.multiply)
            r_tiles.append(r_tile)
        mid_tiles = []
        for member_idx in range(n_group_chunks):
            mid_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=mid_psum, stationary=r_tiles[member_idx], moving=ch_list[member_idx]["inv_t"])
            mid_tile = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=mid_tile, src=mid_psum, engine=COPY_ENGINE)
            mid_tiles.append(mid_tile)
        off_t_tiles = []
        for member_idx in range(n_group_chunks):
            off_t_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=off_t_psum, stationary=ch_list[member_idx]["inv"], moving=mid_tiles[member_idx])
            off_t_tiles.append(off_t_psum)
        for member_idx in range(n_group_chunks):
            inv_t_next = nl.ndarray((CHUNK, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=inv_t_next, data1=ch_list[member_idx]["inv_t"], data2=off_t_tiles[member_idx], op=nl.add
            )
            ch_list[member_idx]["inv_t"] = inv_t_next

    if need_inv:
        for member_idx in range(n_group_chunks):
            ch_list[member_idx]["inv"] = transpose_tile(ch_list[member_idx]["inv_t"], CHUNK, CHUNK)


def _group_log_gate_cols(
    g_log_row_view: nl.NkiTensor,
    batch_idx: int,
    v_head_idx: int,
    chunk_base: int,
    n_group_chunks: int,
) -> nl.NkiTensor:
    """log_G = cumsum(g_log) for EVERY chunk of a group in one pass, as the forward does.

    Returns (CHUNK, n_group_chunks): column `member` is that chunk's per-token log_G, ready to
    slice as a (CHUNK, 1) operand. One DMA + one prefix sum + one transpose per GROUP; doing this
    per chunk instead costs `group_size` times as many.
    """
    g_log_rows = nl.ndarray((n_group_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=g_log_rows,
        src=g_log_row_view[batch_idx, v_head_idx, chunk_base : chunk_base + n_group_chunks, :],
        dge_mode=nisa.dge_mode.none,
    )
    log_gate_rows = nl.ndarray((n_group_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=log_gate_rows, src=g_log_rows, op0=nl.multiply, imm0=1.0, op1=nl.add)
    return transpose_tile(log_gate_rows, n_group_chunks, CHUNK)


def _stage1(
    query_view: nl.NkiTensor,
    key_view: nl.NkiTensor,
    value_view: nl.NkiTensor,
    beta_view: nl.NkiTensor,
    log_gate_col: nl.NkiTensor,
    batch_idx: int,
    k_head_idx: int,
    v_head_idx: int,
    chunk_idx: int,
    masks: tuple,
    compute_dtype: Any,
    value_head_dim: int,
    key_head_dim: int,
) -> dict:
    """Load one chunk and recompute its state-independent intermediates.

    STAGE 1 in the forward's sense: the loads of q/k (both layouts), v and beta, then the gates,
    M, KK, QK, A and w_qk. `log_gate_col` is this chunk's (CHUNK, 1) cumulative gate, already
    formed for the whole group by `_group_log_gate_cols` -- the prefix sum is one op per GROUP, as
    in the forward, not one per chunk.

    The schur inverse (T^T, T) is NOT built here: it is emitted level-major over the group by
    `_schur_group` after every member's stage 1 is in flight.

    Returns the chunk's ``ch`` dict of tiles, keyed as the backward scan reads them.
    """
    upper_neg, tril_strict, eye_mat, _, _ = masks

    k_f = nl.ndarray((key_head_dim, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.dma_transpose(dst=k_f, src=key_view[batch_idx, k_head_idx, chunk_idx, :, :], dge_mode=nisa.dge_mode.none)
    q_f = nl.ndarray((key_head_dim, CHUNK), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.dma_transpose(dst=q_f, src=query_view[batch_idx, k_head_idx, chunk_idx, :, :], dge_mode=nisa.dge_mode.none)
    k_tm = nl.ndarray((CHUNK, key_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_tm, src=key_view[batch_idx, k_head_idx, chunk_idx, :, :], dge_mode=nisa.dge_mode.none)
    q_tm = nl.ndarray((CHUNK, key_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_tm, src=query_view[batch_idx, k_head_idx, chunk_idx, :, :], dge_mode=nisa.dge_mode.none)
    v_tm = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_tm, src=value_view[batch_idx, v_head_idx, chunk_idx, :, :], dge_mode=nisa.dge_mode.none)
    beta_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_col, src=beta_view[batch_idx, v_head_idx, chunk_idx, :, :], dge_mode=nisa.dge_mode.none)

    ch = {}
    ch["k_f"] = k_f
    ch["q_f"] = q_f
    ch["k_tm"] = k_tm
    ch["q_tm"] = q_tm
    ch["v_tm"] = v_tm
    ch["beta_col"] = beta_col

    log_gate_bcast = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=log_gate_bcast, stationary=log_gate_col.broadcast(1, CHUNK), moving=eye_mat)
    log_gate_last = log_gate_bcast[:, CHUNK - 1 : CHUNK]

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
    decay = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=decay, op=nl.exp, data=log_diff)
    ch["decay"] = decay

    kk_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kk_psum, stationary=k_f, moving=k_f)
    kk = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kk, src=kk_psum, engine=COPY_ENGINE)
    ch["kk"] = kk

    qk_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_f, moving=k_f)
    qk = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk, src=qk_psum, engine=COPY_ENGINE)
    ch["qk"] = qk

    # A = tril_-1(beta * M * KK)
    decay_gram = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=decay_gram, data1=decay, data2=kk, op=nl.multiply)
    interaction = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=interaction,
        data=decay_gram,
        op0=nl.multiply,
        operand0=beta_col,
        op1=nl.multiply,
        operand1=tril_strict,
    )
    ch["interaction"] = interaction

    # w_qk = M * QK  (unscaled; the query scale rides on d_out in the backward)
    w_qk = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=w_qk, data1=decay, data2=qk, op=nl.multiply)
    ch["w_qk"] = w_qk

    # Gate columns.
    gate_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_col, op=nl.exp, data=log_gate_col)
    ch["gate_col"] = gate_col
    beta_gate_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=beta_gate_col, data1=gate_col, data2=beta_col, op=nl.multiply)
    ch["beta_gate_col"] = beta_gate_col
    ratio_col = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ratio_col, op=nl.exp, data=log_gate_col, bias=log_gate_last, scale=-1.0)
    ch["ratio_col"] = ratio_col
    gate_last_col = nl.ndarray((key_head_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_last_col, op=nl.exp, data=log_gate_last[0:key_head_dim, :])
    ch["gate_last_col"] = gate_last_col
    beta_value = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=beta_value, data=v_tm, op0=nl.multiply, operand0=beta_col)
    ch["beta_value"] = beta_value

    return ch


def _key_state(k_f: nl.NkiTensor, state_operand: nl.NkiTensor, value_head_dim: int) -> nl.NkiTensor:
    """KS0 = K S0, (CHUNK, Dv)."""
    ks_psum = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ks_psum, stationary=k_f, moving=state_operand)
    ks = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ks, src=ks_psum, engine=COPY_ENGINE)
    return ks


def _b_vec(ch: dict, key_state: nl.NkiTensor, value_head_dim: int, compute_dtype: Any) -> nl.NkiTensor:
    """b = beta*V - (beta*G) KS0, as a matmul operand, (CHUNK, Dv).

    Written straight out in the operand dtype as the forward does: nothing downstream reads b in
    fp32 (dT transposes this same operand), so the fp32 tile would only be rounded and dropped.
    """
    b_op = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=b_op,
        data=key_state,
        op0=nl.multiply,
        operand0=ch["beta_gate_col"],
        op1=nl.subtract,
        operand1=ch["beta_value"],
        reverse1=True,
    )
    return b_op


def _deltas(ch: dict, b_op: nl.NkiTensor, value_head_dim: int, compute_dtype: Any) -> nl.NkiTensor:
    """U = T b = inv_t^T @ b, (CHUNK, Dv), in the operand dtype (no fp32 consumer)."""
    u_psum = nl.ndarray((CHUNK, value_head_dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=u_psum, stationary=ch["inv_t"], moving=b_op)
    u_op = nl.ndarray((CHUNK, value_head_dim), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=u_op, src=u_psum, engine=COPY_ENGINE)
    return u_op


def _backward_chunk(
    ch: dict,
    s0: nl.NkiTensor,
    s0_op: nl.NkiTensor,
    key_state: nl.NkiTensor,
    b_op: nl.NkiTensor,
    u_op: nl.NkiTensor,
    d_out_src: nl.NkiTensor,
    d_state_cur: nl.NkiTensor,
    scale: float,
    masks: tuple,
    ones_c: nl.NkiTensor,
    ones_dk: nl.NkiTensor,
    compute_dtype: Any,
    value_head_dim: int,
    key_head_dim: int,
    d_query_dst: nl.NkiTensor,
    d_key_dst: nl.NkiTensor,
    d_value_dst: nl.NkiTensor,
    d_g_dst: nl.NkiTensor,
    d_beta_dst: nl.NkiTensor,
    acc_query_dst: Optional[nl.NkiTensor],
    acc_key_dst: Optional[nl.NkiTensor],
    is_first_vhead: bool,
    is_last_vhead: bool,
) -> None:
    """VJP of one forward chunk; writes dQ/dK/dV/dg/dbeta and updates d_state_cur to dS0.

    The whole chain, primitive by primitive, in the order the body emits it. `<x,y>` is a dot
    over the feature axis (one `scalar_tensor_tensor` with `reduce_res`), `Sum_s` a sum over tokens.
    Inputs are dO (dL/d this chunk's output) and dS_new (dL/d this chunk's OUTPUT state, the
    reverse scan's carry); dS0 is handed to the previous chunk as its dS_new.

        dOs     = scale dO
        # O = scale (G*QS0 + Wqk U)
        dQS0    = G dOs ;  dG = <QS0, dOs> ;  dWqk = dOs U^T ;  dU = Wqk^T dOs
        # S_new = G_last S0 + Kdec^T U
        dKdec   = U dS_new^T ;  dU += Kdec dS_new ;  dS0 = G_last dS_new
        # Kdec = ratio*K, ratio = G_last/G  -- routed in LOG space, see the note in the body
        dK      = ratio dKdec ;  dratio = <K, dKdec>
        dlogG[s]   += -dratio_s ratio_s
        dlogG[C-1] += Sum_s(dratio_s ratio_s) + G_last <S0, dS_new>
        # U = T b
        dT      = dU b^T ;  db = T^T dU          (reuses the forward's Inv_T)
        # b = beta*V - (beta*G) KS0
        dV      = beta db ;  dbeta = <V, db> - G <KS0, db> ;  dKS0 = -(beta*G) db
        dG     += -beta <KS0, db>
        # KS0 = K S0 ;  QS0 = Q S0
        dK     += dKS0 S0^T ;  dS0 += K^T dKS0
        dQ      = dQS0 S0^T ;  dS0 += Q^T dQS0
        # T = (I+A)^-1, on A's strictly-lower support
        dA      = tril_-1(-(T^T dT T^T))         (needs BOTH orientations; _schur_group emits them)
        # A = tril_-1(beta * M * KK)
        dbeta  += Sum_s(dA * M * KK) ;  dM = beta dA * KK ;  dKK = beta dA * M
        # Wqk = M * QK
        dM     += dWqk * QK ;  dQK = dWqk * M
        # KK = K K^T ;  QK = Q K^T
        dK     += (dKK + dKK^T) K ;  dQ += dQK K ;  dK += dQK^T Q
        # M[r,s] = exp(logG[r] - logG[s]) on r >= s
        dMM     = tril(dM) * M
        dlogG  += rowsum_s(dMM) - colsum_r(dMM) + G dG
        # logG = cumsum(g):  the adjoint of a prefix sum is a REVERSE prefix sum
        dg      = flip(cumsum(flip(dlogG)))

    G_last is the same variable as G[C-1], so the `ratio` term feeds both the numerator (routed
    to logG[C-1]) and the denominator (the full logG vector, whose s = C-1 entry cancels it,
    since ratio[C-1] == 1). Both land in the single dlogG accumulator, which is what keeps the
    contribution from being double counted.

    dQ/dK go to `d_query_dst`/`d_key_dst` directly when one value head owns the key head. Under
    GQA (`acc_query_dst`/`acc_key_dst` given) the group's partials are summed in the fp32 scratch
    they point at, and only the last value head's total is rounded into the output.
    """
    _, tril_strict, _, _, _ = masks  # the strict-lower mask is the only one dA needs
    C = CHUNK
    Dv = value_head_dim
    Dk = key_head_dim

    # d_os = scale * d_out
    d_out_tile = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=d_out_tile, src=d_out_src, dge_mode=nisa.dge_mode.none)
    d_os = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d_os, data=d_out_tile, op0=nl.multiply, operand0=float(scale))
    d_os_op = nl.ndarray((C, Dv), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_os_op, src=d_os)

    # transposes reused below
    d_os_t = transpose_tile(d_os_op, C, Dv)  # (Dv, C)
    u_t = transpose_tile(u_op, C, Dv)  # (Dv, C)

    # QS0 = Q S0
    qs0_psum = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qs0_psum, stationary=ch["q_f"], moving=s0_op)
    qs0 = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qs0, src=qs0_psum, engine=COPY_ENGINE)

    # d_query_state = G * d_os ; d_gate = sum_dv(QS0 * d_os)
    d_query_state = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d_query_state, data=d_os, op0=nl.multiply, operand0=ch["gate_col"])
    """Every `<x,y>` below is one `scalar_tensor_tensor` with `reduce_res`, not a
    `tensor_tensor` + `tensor_reduce` pair: the Vector engine reduces the elementwise result along
    the free axis in the same instruction, which drops the reduce's separate read pass over the
    (C, Dv) / (C, C) tile. `op0` is the identity `multiply` by 1.0 -- only the tensor-tensor product
    is wanted, and the docs put the scalar broadcast at no additional cost. `dst` is still written,
    so this saves the read, not the write. Needs nki >= 0.7.0b1."""
    tmp_cdv = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    d_gate = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=tmp_cdv,
        data=qs0,
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=d_os,
        reduce_op=nl.add,
        reduce_res=d_gate,
    )

    # d_w_qk = d_os @ U^T  (contract Dv)
    d_w_qk_psum = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_w_qk_psum, stationary=d_os_t, moving=u_t)
    d_w_qk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_w_qk, src=d_w_qk_psum, engine=COPY_ENGINE)

    # d_deltas = w_qk^T @ d_os + k_decay @ d_state_new
    d_state_op = nl.ndarray((Dk, Dv), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_state_op, src=d_state_cur)
    # k_decay^T (Dk, C) and k_decay (C, Dk)
    k_decay = nl.ndarray((C, Dk), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_decay, data=ch["k_tm"], op0=nl.multiply, operand0=ch["ratio_col"])
    k_decay_t = transpose_tile(k_decay, C, Dk)  # (Dk, C)
    w_qk_op = nl.ndarray((C, C), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_qk_op, src=ch["w_qk"])
    d_deltas_psum = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_deltas_psum, stationary=w_qk_op, moving=d_os_op)
    nisa.nc_matmul(dst=d_deltas_psum, stationary=k_decay_t, moving=d_state_op)
    d_deltas_op = nl.ndarray((C, Dv), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_deltas_op, src=d_deltas_psum, engine=COPY_ENGINE)
    d_deltas_t = transpose_tile(d_deltas_op, C, Dv)  # (Dv, C)

    # d_k_decay = U @ d_state_new^T (contract Dv)
    d_state_t = transpose_tile(d_state_op, Dk, Dv)  # (Dv, Dk)
    d_k_decay_psum = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_k_decay_psum, stationary=u_t, moving=d_state_t)
    d_k_decay = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_k_decay, src=d_k_decay_psum, engine=COPY_ENGINE)

    # d_state0 = G_last * d_state_new (init accumulator, fp32)
    d_state0 = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d_state0, data=d_state_cur, op0=nl.multiply, operand0=ch["gate_last_col"])

    # d_ratio = sum_dk(K * d_k_decay) ; d_k(part) = ratio * d_k_decay
    tmp_cdk = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
    d_ratio = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=tmp_cdk,
        data=ch["k_tm"],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=d_k_decay,
        reduce_op=nl.add,
        reduce_res=d_ratio,
    )
    d_k_from_ratio = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d_k_from_ratio, data=d_k_decay, op0=nl.multiply, operand0=ch["ratio_col"])

    """ratio_s = G_last/G_s enters d_log_g in LOG-space via the BOUNDED product d_ratio*ratio, NOT
    through 1/G_s. The algebraically-equal form -- d_gate += -d_ratio*ratio/G_s and
    d_gate_last_num = sum_s(d_ratio/G_s), each cancelled by the *G_s / *G_last at the log-space step
    below -- OVERFLOWS: once G_s = exp(cumsum g) underflows to 0 under a large per-token decay,
    1/G_s (nisa.reciprocal) is +inf and the later *G_s gives inf*0 = nan. d_ratio*ratio is finite
    (ratio <= 1), so route it straight: -d_ratio*ratio into d_log_g[s], +sum_s(d_ratio*ratio) into
    log_g[-1]. Only the state part of log_g[-1] keeps its *G_last factor."""
    d_ratio_ratio = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d_ratio_ratio, data1=d_ratio, data2=ch["ratio_col"], op=nl.multiply)
    num_psum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=num_psum, stationary=d_ratio_ratio, moving=ones_c)
    # d_gate_last_state = sum_{dk,dv}(S0 * d_state_new)
    s0_dstate = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.sbuf)
    s0_dstate_row = nl.ndarray((Dk, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=s0_dstate,
        data=s0,
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=d_state_cur,
        reduce_op=nl.add,
        reduce_res=s0_dstate_row,
    )
    state_psum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=state_psum, stationary=s0_dstate_row, moving=ones_dk)
    num_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)  # sum_s(d_ratio*ratio) -> log_g[-1] RAW
    nisa.tensor_copy(dst=num_sb, src=num_psum, engine=COPY_ENGINE)
    state_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)  # sum(S0*dS_new), the * G_last part
    nisa.tensor_copy(dst=state_sb, src=state_psum, engine=COPY_ENGINE)

    # db = T^T dU = inv @ d_deltas ; d_v = beta * db
    db_psum = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=db_psum, stationary=ch["inv"], moving=d_deltas_op)
    db = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=db, src=db_psum, engine=COPY_ENGINE)

    d_v = nl.ndarray((C, Dv), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d_v, data=db, op0=nl.multiply, operand0=ch["beta_col"])
    nisa.dma_copy(dst=d_value_dst, src=d_v, dge_mode=nisa.dge_mode.none)

    # d_beta(part) = sum_dv(V*db) - G * sum_dv(KS0*db) ; also feeds d_gate
    vdb = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    sum_vdb = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=vdb,
        data=ch["v_tm"],
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=db,
        reduce_op=nl.add,
        reduce_res=sum_vdb,
    )
    ksdb = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    t_kd = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=ksdb,
        data=key_state,
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=db,
        reduce_op=nl.add,
        reduce_res=t_kd,
    )
    d_beta = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=d_beta,
        data=t_kd,
        op0=nl.multiply,
        operand0=ch["gate_col"],
        op1=nl.subtract,
        operand1=sum_vdb,
        reverse1=True,
    )
    # d_gate += -beta * t_kd
    beta_tkd = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=beta_tkd, data1=ch["beta_col"], data2=t_kd, op=nl.multiply)
    nisa.tensor_tensor(dst=d_gate, data1=d_gate, data2=beta_tkd, op=nl.subtract)

    # d_key_state = -(beta*G) * db
    d_key_state = nl.ndarray((C, Dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=d_key_state, data=db, op0=nl.multiply, operand0=ch["beta_gate_col"], op1=nl.multiply, operand1=-1.0
    )
    d_key_state_op = nl.ndarray((C, Dv), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_key_state_op, src=d_key_state)
    d_key_state_t = transpose_tile(d_key_state_op, C, Dv)  # (Dv, C)

    # dT = d_deltas @ b^T (contract Dv). b is already the operand dtype, so transpose THAT: going
    # through the wider fp32 tile and narrowing after costs an extra (Dv, C) pass for the same bits.
    b_t_op = transpose_tile(b_op, C, Dv)  # (Dv, C)
    d_inv_psum = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_inv_psum, stationary=d_deltas_t, moving=b_t_op)
    d_inv = nl.ndarray((C, C), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_inv, src=d_inv_psum, engine=COPY_ENGINE)

    # dA = tril_-1( -(T^T dT T^T) )
    mid_psum = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=mid_psum, stationary=ch["inv"], moving=d_inv)  # (T^T dT)[a,j]
    mid = nl.ndarray((C, C), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mid, src=mid_psum, engine=COPY_ENGINE)
    mid_t = transpose_tile(mid, C, C)  # (j, a)
    da_pre_psum = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=da_pre_psum, stationary=mid_t, moving=ch["inv_t"])  # (T^T dT T^T)[a,b]
    d_a = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=d_a, data=da_pre_psum, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=tril_strict
    )

    # d_beta += sum_s(dA * M * KK)
    da_m = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=da_m, data1=d_a, data2=ch["decay"], op=nl.multiply)
    da_m_kk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    d_beta_a = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=da_m_kk,
        data=da_m,
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=ch["kk"],
        reduce_op=nl.add,
        reduce_res=d_beta_a,
    )
    nisa.tensor_tensor(dst=d_beta, data1=d_beta, data2=d_beta_a, op=nl.add)
    nisa.dma_copy(dst=d_beta_dst, src=d_beta, dge_mode=nisa.dge_mode.none)

    # d_decay = beta*dA*KK + d_w_qk*QK ; d_kk = beta*dA*M ; d_qk = d_w_qk*M
    beta_da = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=beta_da, data=d_a, op0=nl.multiply, operand0=ch["beta_col"])
    d_decay = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d_decay, data1=beta_da, data2=ch["kk"], op=nl.multiply)
    dwqk_qk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dwqk_qk, data1=d_w_qk, data2=ch["qk"], op=nl.multiply)
    nisa.tensor_tensor(dst=d_decay, data1=d_decay, data2=dwqk_qk, op=nl.add)
    d_kk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d_kk, data1=beta_da, data2=ch["decay"], op=nl.multiply)
    d_qk = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d_qk, data1=d_w_qk, data2=ch["decay"], op=nl.multiply)

    # d_decay_m = d_decay * M ; d_log_g = rowsum_s - colsum_r
    # d_decay_m is read again by the col_sum matmul below, so only the reduce's pass is saved here.
    d_decay_m = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    row_sum = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=d_decay_m,
        data=d_decay,
        op0=nl.multiply,
        operand0=1.0,
        op1=nl.multiply,
        operand1=ch["decay"],
        reduce_op=nl.add,
        reduce_res=row_sum,
    )
    col_sum_psum = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=col_sum_psum, stationary=d_decay_m, moving=ones_c)
    d_log_g = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d_log_g, data1=row_sum, data2=col_sum_psum, op=nl.subtract)

    # d_log_g += G * d_gate  - d_ratio*ratio  (the ratio term folded from the removed gate*(1/G) form)
    g_dgate = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=g_dgate, data1=ch["gate_col"], data2=d_gate, op=nl.multiply)
    nisa.tensor_tensor(dst=d_log_g, data1=d_log_g, data2=g_dgate, op=nl.add)
    nisa.tensor_tensor(dst=d_log_g, data1=d_log_g, data2=d_ratio_ratio, op=nl.subtract)

    # d_log_g[-1] += G_last * sum(S0*dS_new)  +  sum_s(d_ratio*ratio); [1,C] layout after transpose.
    # State part keeps its * G_last; the ratio part is added RAW (already folded, no 1/G).
    d_log_g_row = transpose_tile(d_log_g, C, 1)  # (1, C)
    gl_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gl_scalar, data1=state_sb, data2=ch["gate_last_col"][0:1, :], op=nl.multiply)
    nisa.tensor_tensor(dst=gl_scalar, data1=gl_scalar, data2=num_sb, op=nl.add)
    nisa.tensor_tensor(dst=d_log_g_row[0:1, C - 1 : C], data1=d_log_g_row[0:1, C - 1 : C], data2=gl_scalar, op=nl.add)

    # d_g = reverse_cumsum(d_log_g) = total - inclusive_prefix + d_log_g
    prefix = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=prefix, src=d_log_g_row, op0=nl.multiply, imm0=1.0, op1=nl.add)
    total = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=total, op=nl.add, data=d_log_g_row, axis=(1,))
    d_g = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=d_g, data=prefix, op0=nl.subtract, operand0=total, reverse0=True, op1=nl.add, operand1=d_log_g_row
    )
    nisa.dma_copy(dst=d_g_dst, src=d_g, dge_mode=nisa.dge_mode.none)

    # ---- d_q and d_k accumulation ----
    # d_q = d_query_state @ S0^T + d_qk @ K   (both token r on partition)
    s0_t = transpose_tile(s0_op, Dk, Dv)  # (Dv, Dk)
    d_query_state_op = nl.ndarray((C, Dv), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_query_state_op, src=d_query_state)
    d_query_state_t = transpose_tile(d_query_state_op, C, Dv)  # (Dv, C)
    d_qk_op = nl.ndarray((C, C), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_qk_op, src=d_qk)
    d_qk_t = transpose_tile(d_qk_op, C, C)  # (s, r)
    d_q_psum = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_q_psum, stationary=d_query_state_t, moving=s0_t)
    nisa.nc_matmul(dst=d_q_psum, stationary=d_qk_t, moving=ch["k_tm"])
    # Dense evacuates PSUM straight to the output dtype; GQA keeps fp32 for the head sum below.
    dq_dk_dtype = nl.float32 if acc_query_dst is not None else compute_dtype
    d_q = nl.ndarray((C, Dk), dtype=dq_dk_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_q, src=d_q_psum, engine=COPY_ENGINE)

    # d_k = ratio*d_k_decay + d_key_state @ S0^T + (d_kk+d_kk^T) @ K + d_qk^T @ Q
    d_kk_op = nl.ndarray((C, C), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d_kk_op, src=d_kk)
    d_kk_t = transpose_tile(d_kk_op, C, C)
    sym = nl.ndarray((C, C), dtype=compute_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sym, data1=d_kk_op, data2=d_kk_t, op=nl.add)
    d_k_psum = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_k_psum, stationary=d_key_state_t, moving=s0_t)
    nisa.nc_matmul(dst=d_k_psum, stationary=sym, moving=ch["k_tm"])
    nisa.nc_matmul(dst=d_k_psum, stationary=d_qk_op, moving=ch["q_tm"])
    d_k = nl.ndarray((C, Dk), dtype=dq_dk_dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=d_k, data=d_k_psum, op0=nl.multiply, operand0=1.0, op1=nl.add, operand1=d_k_from_ratio
    )

    if acc_query_dst is None:
        # Dense: one value head per key head, so this chunk's dQ/dK are already the whole sum.
        nisa.dma_copy(dst=d_query_dst, src=d_q, dge_mode=nisa.dge_mode.none)
        nisa.dma_copy(dst=d_key_dst, src=d_k, dge_mode=nisa.dge_mode.none)
    else:
        """GQA: this key head is shared, so sum the group's partials in the fp32 scratch --
        read-modify-write, safe because the shard owns the key head. The first value head has
        nothing to read; the last rounds the total into the output instead of storing it back,
        which is why the compute-dtype rounding happens ONCE per chunk however deep the group."""
        if is_first_vhead:
            total_q, total_k = d_q, d_k
        else:
            prev_q = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=prev_q, src=acc_query_dst, dge_mode=nisa.dge_mode.none)
            total_q = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=total_q, data1=d_q, data2=prev_q, op=nl.add)
            prev_k = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=prev_k, src=acc_key_dst, dge_mode=nisa.dge_mode.none)
            total_k = nl.ndarray((C, Dk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=total_k, data1=d_k, data2=prev_k, op=nl.add)

        if is_last_vhead:
            out_q = nl.ndarray((C, Dk), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_q, src=total_q)
            nisa.dma_copy(dst=d_query_dst, src=out_q, dge_mode=nisa.dge_mode.none)
            out_k = nl.ndarray((C, Dk), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_k, src=total_k)
            nisa.dma_copy(dst=d_key_dst, src=out_k, dge_mode=nisa.dge_mode.none)
        else:
            nisa.dma_copy(dst=acc_query_dst, src=total_q, dge_mode=nisa.dge_mode.none)
            nisa.dma_copy(dst=acc_key_dst, src=total_k, dge_mode=nisa.dge_mode.none)

    # ---- d_state0 += K^T d_key_state + Q^T d_query_state ----
    d_state_acc_psum = nl.ndarray((Dk, Dv), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=d_state_acc_psum, stationary=ch["k_tm"], moving=d_key_state_op)
    nisa.nc_matmul(dst=d_state_acc_psum, stationary=ch["q_tm"], moving=d_query_state_op)
    nisa.tensor_tensor(dst=d_state_cur, data1=d_state0, data2=d_state_acc_psum, op=nl.add)
