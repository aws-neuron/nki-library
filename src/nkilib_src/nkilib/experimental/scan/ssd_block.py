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

"""Chunk-outer SSD (State Space Duality) scan kernel for Mamba-2 prefill."""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .ssd_utils import compute_lnc_sharding

P_MAX = nl.tile_size.pmax
F_MAX = nl.tile_size.psum_fmax

# Per-partition SBUF bytes the resident SSM states may use (heads_per_core * headdim * fp32).
# Leaves ample room for the per-chunk tiles; larger configs park states in HBM.
_STATE_SBUF_BUDGET_BYTES = 64 * 1024


def ssd_block(
    x: nl.NkiTensor,
    dt: nl.NkiTensor,
    A: nl.NkiTensor,
    B: nl.NkiTensor,
    C: nl.NkiTensor,
    chunk_size: int = 128,
    D: nl.NkiTensor = None,
    initial_state: nl.NkiTensor = None,
    causal_mask: nl.NkiTensor = None,
) -> tuple:
    """Chunk-outer SSD (State Space Duality) scan for Mamba-2 prefill.

    Computes the SSD parallel scan with chunk-outer, head-inner loop nesting.
    B and C projections are loaded once per chunk and shared across all heads,
    eliminating redundant HBM traffic. LNC sharding distributes heads across
    NeuronCores with round-robin remainder handling.

    Most efficient at real model head counts where shared B/C projections
    amortize across many heads. Supports arbitrary headdim via automatic
    PSUM tiling when headdim > 512.

    Dimensions:
        batch: Batch size.
        nheads: Number of SSM heads.
        seqlen: Sequence length (must be divisible by chunk_size).
        headdim: Per-head dimension. Tiled in F_MAX (512) chunks when > 512.
        dstate: SSM state dimension (<= 128 for partition dimension).
        Q: Chunk size (= chunk_size, <= 128).

    Args:
        x (nl.NkiTensor): [batch, nheads, seqlen, headdim], Input activations.
        dt (nl.NkiTensor): [batch, nheads, seqlen], Softplus'd timesteps. Must be positive.
        A (nl.NkiTensor): [nheads], State transition scalars. Typically negative.
        B (nl.NkiTensor): [batch, seqlen, dstate], Input projection.
        C (nl.NkiTensor): [batch, seqlen, dstate], Output projection.
        chunk_size (int): Chunk size for parallel scan. Must be <= 128.
        D (nl.NkiTensor, optional): [nheads], Skip connection weights.
        initial_state (nl.NkiTensor, optional): [batch, nheads, dstate, headdim], Initial
            SSM state. Default: None (zeros).
        causal_mask (nl.NkiTensor): [Q, Q], Lower-triangular mask. Required.
            Pass np.tril(np.ones((Q, Q), dtype=np.float32)).

    Returns:
        tuple: (y, final_state)
            - y (nl.NkiTensor): [batch, nheads, seqlen, headdim], Output, same dtype as x.
            - final_state (nl.NkiTensor): [batch, nheads, dstate, headdim], Final SSM state
              in float32.

    Notes:
        - chunk_size <= 128 (must fit in partition dimension)
        - dstate <= 128 (must fit in partition dimension for SBUF state)
        - seqlen must be divisible by chunk_size
        - ngroups=1 (B/C shared across all heads)
        - Uses float32 accumulation internally for numerical stability
        - A should be negative for stable dynamics (decay < 1)
        - dt should be positive; discretization computes exp(dt * A)
        - TODO: dstate > 128 support via tiled state through HBM (needed for
          Falcon-H1 1.5B+ which uses dstate=256)
        - TODO: ngroups > 1 support via per-group B/C projections (needed for
          Zamba2-7B ngroups=2, Falcon-H1 34B ngroups=2). Assign groups to
          NeuronCores so each core only loads its groups' B/C — avoids
          replication when ngroups divides evenly across LNC cores.

    Pseudocode:
        for each chunk:
            B_T, C_T, B_f32, CB_masked = shared_projections(B, C, causal_mask)
            for each block of <= 128 heads (LNC sharded):
                cs[heads, Q] = cumsum(dt * A)       # one scan, heads on partitions
                w_intra, w_state = dt * exp(-cs), dt * exp(cs[-1] - cs)
                for each head in block:             # state resident in SBUF
                    P = CB_masked @ (x * w_intra) + C @ state    # one PSUM tile
                    y = exp(cs) * P + D * x
                    state = exp(cs[-1]) * state + B^T @ (x * w_state)
    """
    batch = x.shape[0]
    nheads = x.shape[1]
    seqlen = x.shape[2]
    headdim = x.shape[3]
    dstate = B.shape[2]
    Q = chunk_size
    num_chunks = div_ceil(seqlen, Q)

    HDIM_TILE = min(headdim, F_MAX)
    num_dstate_tiles = 1
    num_hdim_tiles = div_ceil(headdim, HDIM_TILE)

    heads_per_core, head_offset = compute_lnc_sharding(nheads)

    # Heads are processed in blocks of up to P_MAX so per-chunk decay terms for a
    # whole block can be computed with heads on the partition dimension.
    num_head_blocks = div_ceil(heads_per_core, P_MAX)

    A_2d = A.reshape((nheads, 1))
    dt_2d = dt.reshape((batch * nheads, seqlen))

    has_D = D != None
    if has_D:
        D_row = D.reshape((1, nheads))

    # Kernel-lifetime constants per head block: A as a per-partition scalar column,
    # D broadcast across the Q token partitions.
    A_cols = []
    D_bcasts = []
    for hb_idx in range(num_head_blocks):
        hb_start = hb_idx * P_MAX
        hb_size = min(P_MAX, heads_per_core - hb_start)
        g0 = head_offset + hb_start

        A_col = nl.ndarray((hb_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=A_col[0:hb_size, 0:1], src=A_2d[g0 : g0 + hb_size, 0:1])
        A_cols.append(A_col)

        if has_D:
            D_hb_row = nl.ndarray((1, hb_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=D_hb_row[0:1, 0:hb_size], src=D_row[0:1, g0 : g0 + hb_size])
            D_bcast = nl.ndarray((Q, hb_size), dtype=nl.float32, buffer=nl.sbuf)
            stream_shuffle_broadcast(src=D_hb_row, dst=D_bcast)
            D_bcasts.append(D_bcast)

    # Scan constants shared by all chunks and head blocks
    ones_blk = nl.ndarray((P_MAX, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_blk, value=1.0)
    zeros_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zeros_col, value=0.0)

    y = nl.ndarray((batch, nheads, seqlen, headdim), dtype=x.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray(
        (batch, nheads, dstate, headdim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    # Keep every head's running state resident in SBUF across chunks when it fits.
    # The chunk-outer loop revisits each head once per chunk, so all heads' states
    # are live at once; resident state removes a per-head, per-chunk HBM round trip
    # from the critical path. Otherwise fall back to parking states in HBM.
    state_resident = heads_per_core * headdim * 4 <= _STATE_SBUF_BUDGET_BYTES

    if state_resident:
        # One tile per head (not one wide tile) so heads carry no false dependencies
        state_tiles = []
        for local_head_idx in range(heads_per_core):
            state_tiles.append(nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf))
    else:
        # max(., 1) ensures valid allocation shape when a core has no heads assigned
        state_hbm = nl.ndarray(
            (batch, max(heads_per_core, 1), dstate, headdim),
            dtype=nl.float32,
            buffer=nl.shared_hbm,
        )

    for batch_idx in range(batch):
        # Initialize state
        for local_head_idx in range(heads_per_core):
            global_head_idx = head_offset + local_head_idx
            if state_resident:
                state_init = state_tiles[local_head_idx]
            else:
                state_init = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
            if initial_state != None:
                nisa.dma_copy(
                    dst=state_init[0:dstate, 0:headdim],
                    src=initial_state[batch_idx, global_head_idx, 0:dstate, 0:headdim],
                )
            else:
                nisa.memset(dst=state_init, value=0.0)
            if not state_resident:
                nisa.dma_copy(
                    dst=state_hbm[batch_idx, local_head_idx, 0:dstate, 0:headdim],
                    src=state_init[0:dstate, 0:headdim],
                )

        # Precompute upper-triangular mask = transpose(causal)
        causal_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=causal_sb[0:Q, 0:Q], src=causal_mask[0:Q, 0:Q])
        triu_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=triu_psum[0:Q, 0:Q], data=causal_sb[0:Q, 0:Q])
        triu_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=triu_sb[0:Q, 0:Q], src=triu_psum[0:Q, 0:Q])

        # One-hot selector on token row Q-1 (column Q-1 of the lower-triangular mask),
        # replicated across dstate: matmul(sel, M)[n, h] = M[Q-1, h] for every n.
        last_row_selector = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=last_row_selector, value=1.0)
        nisa.tensor_scalar(
            dst=last_row_selector[0:Q, 0:dstate],
            data=last_row_selector[0:Q, 0:dstate],
            op0=nl.multiply,
            operand0=causal_sb[0:Q, Q - 1 : Q],
        )

        # Chunk-outer loop
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * Q

            # Shared B/C projections
            B_T_tiles, C_T_tiles, B_f32_tiles, CB_masked = _compute_chunk_shared_projections_tiled(
                B,
                C,
                batch_idx,
                chunk_start,
                Q,
                dstate,
                dstate,
                num_dstate_tiles,
                triu_sb,
            )

            # Head-inner loop (LNC sharded), in blocks of up to P_MAX heads
            for hb_idx in range(num_head_blocks):
                hb_start = hb_idx * P_MAX
                hb_size = min(P_MAX, heads_per_core - hb_start)

                # Decay terms for every head in the block at once, as [Q, hb_size] columns
                w_intra_T, w_state_T, exp_cs_T, exp_cs_last_N = _compute_chunk_decay_block(
                    dt_2d,
                    A_cols[hb_idx],
                    ones_blk,
                    zeros_col,
                    last_row_selector,
                    batch_idx * nheads + head_offset + hb_start,
                    hb_size,
                    chunk_start,
                    Q,
                    dstate,
                )

                for blk_head_idx in range(hb_size):
                    local_head_idx = hb_start + blk_head_idx
                    global_head_idx = head_offset + local_head_idx
                    j = blk_head_idx

                    D_col = None
                    if has_D:
                        D_col = D_bcasts[hb_idx][0:Q, j : j + 1]

                    if state_resident:
                        state_sb = state_tiles[local_head_idx]
                    else:
                        # Load state into SBUF — persists across headdim tiles
                        state_sb = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(
                            dst=state_sb[0:dstate, 0:headdim],
                            src=state_hbm[batch_idx, local_head_idx, 0:dstate, 0:headdim],
                        )

                    # Tile over headdim (only matmul results need PSUM tiling)
                    for d_tile_idx in range(num_hdim_tiles):
                        d_start = d_tile_idx * HDIM_TILE
                        d_size = min(HDIM_TILE, headdim - d_start)

                        _compute_ssd_head_tile(
                            x,
                            y,
                            state_sb,
                            batch_idx,
                            global_head_idx,
                            chunk_start,
                            d_start,
                            d_size,
                            w_intra_T[0:Q, j : j + 1],
                            w_state_T[0:Q, j : j + 1],
                            exp_cs_T[0:Q, j : j + 1],
                            exp_cs_last_N[0:dstate, j : j + 1],
                            CB_masked,
                            B_T_tiles,
                            C_T_tiles,
                            B_f32_tiles,
                            D_col,
                            Q,
                            dstate,
                            dstate,
                            num_dstate_tiles,
                            has_D,
                        )

                    if not state_resident:
                        # Store state back to HBM after all headdim tiles
                        nisa.dma_copy(
                            dst=state_hbm[batch_idx, local_head_idx, 0:dstate, 0:headdim],
                            src=state_sb[0:dstate, 0:headdim],
                        )

        # Copy final state to output
        for local_head_idx in range(heads_per_core):
            global_head_idx = head_offset + local_head_idx
            if state_resident:
                final_state = state_tiles[local_head_idx]
            else:
                final_state = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=final_state[0:dstate, 0:headdim],
                    src=state_hbm[batch_idx, local_head_idx, 0:dstate, 0:headdim],
                )
            nisa.dma_copy(
                dst=final_state_out[batch_idx, global_head_idx, 0:dstate, 0:headdim],
                src=final_state[0:dstate, 0:headdim],
            )

    return y, final_state_out


def _compute_chunk_decay_block(
    dt_2d: nl.NkiTensor,
    A_col: nl.NkiTensor,
    ones_blk: nl.NkiTensor,
    zeros_col: nl.NkiTensor,
    last_row_selector: nl.NkiTensor,
    dt_row_start: int,
    hb_size: int,
    chunk_start: int,
    chunk_size: int,
    dstate: int,
) -> tuple:
    """Compute one chunk's decay terms for a block of heads at once.

    Heads sit on the partition dimension while the cumulative sum runs along the
    free (token) dimension, so the whole block needs one DMA, one scan, and a few
    transposes instead of a scalar chain per head.

    Args:
        dt_2d (nl.NkiTensor): [batch * nheads, seqlen], Timesteps in HBM.
        A_col (nl.NkiTensor): [hb_size, 1], A for each head in the block (SBUF).
        ones_blk (nl.NkiTensor): [>= hb_size, Q], Ones buffer for the scan (SBUF).
        zeros_col (nl.NkiTensor): [>= hb_size, 1], Zero scan initial values (SBUF).
        last_row_selector (nl.NkiTensor): [Q, dstate], One-hot on token row Q-1 (SBUF).
        dt_row_start (int): Row of dt_2d holding the block's first head.
        hb_size (int): Number of heads in the block (<= P_MAX).
        chunk_start (int): Starting sequence position for this chunk.
        chunk_size (int): Chunk size Q.
        dstate (int): State dimension N.

    Returns:
        tuple: (w_intra_T, w_state_T, exp_cs_T, exp_cs_last_N), column j = head j
            - w_intra_T: [Q, hb_size], dt * exp(-cs).
            - w_state_T: [Q, hb_size], dt * exp(cs[Q-1] - cs).
            - exp_cs_T: [Q, hb_size], exp(cs).
            - exp_cs_last_N: [dstate, hb_size], exp(cs[Q-1]) replicated over dstate.
    """
    Q = chunk_size
    H = hb_size

    # dt for all heads in the block: H contiguous rows of Q timesteps
    dt_blk = nl.ndarray((H, Q), dtype=dt_2d.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=dt_blk[0:H, 0:Q],
        src=dt_2d[dt_row_start : dt_row_start + H, chunk_start : chunk_start + Q],
    )
    dt_f32 = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dt_f32[0:H, 0:Q], src=dt_blk[0:H, 0:Q])

    # cs = cumsum(dt * A) along tokens, one scan for every head
    log_decay = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=log_decay[0:H, 0:Q],
        data=dt_f32[0:H, 0:Q],
        op0=nl.multiply,
        operand0=A_col[0:H, 0:1],
    )
    cs = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=cs[0:H, 0:Q],
        data0=ones_blk[0:H, 0:Q],
        data1=log_decay[0:H, 0:Q],
        initial=zeros_col[0:H, 0:1],
        op0=nl.multiply,
        op1=nl.add,
    )

    # decay = exp(cs_last - cs): cs_last is a per-partition bias in this layout
    decay = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        op=nl.exp,
        data=cs[0:H, 0:Q],
        bias=cs[0:H, Q - 1 : Q],
        scale=-1.0,
        dst=decay[0:H, 0:Q],
    )

    # Fold dt into the per-token scales once per block, so each head scales x once
    # per use instead of computing dt * x first.
    # w_intra = dt * exp(-cs)  (intra-chunk input scale)
    exp_neg_cs = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(op=nl.exp, data=cs[0:H, 0:Q], dst=exp_neg_cs[0:H, 0:Q], scale=-1.0)
    w_intra = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=w_intra[0:H, 0:Q], data1=dt_f32[0:H, 0:Q], data2=exp_neg_cs[0:H, 0:Q], op=nl.multiply)
    # w_state = dt * exp(cs_last - cs)  (state-update input scale)
    w_state = nl.ndarray((H, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=w_state[0:H, 0:Q], data1=dt_f32[0:H, 0:Q], data2=decay[0:H, 0:Q], op=nl.multiply)

    # Transpose to token-on-partition layout [Q, H]
    cs_T = _transpose_to_sbuf(cs, H, Q)
    w_intra_T = _transpose_to_sbuf(w_intra, H, Q)
    w_state_T = _transpose_to_sbuf(w_state, H, Q)

    exp_cs_T = nl.ndarray((Q, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(op=nl.exp, data=cs_T[0:Q, 0:H], dst=exp_cs_T[0:Q, 0:H])

    # exp(cs_last) replicated over dstate partitions: pick token row Q-1 via matmul
    exp_cs_last_psum = nl.ndarray((dstate, H), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=exp_cs_last_psum[0:dstate, 0:H],
        stationary=last_row_selector[0:Q, 0:dstate],
        moving=exp_cs_T[0:Q, 0:H],
    )
    exp_cs_last_N = nl.ndarray((dstate, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=exp_cs_last_N[0:dstate, 0:H], src=exp_cs_last_psum[0:dstate, 0:H])

    return w_intra_T, w_state_T, exp_cs_T, exp_cs_last_N


def _transpose_to_sbuf(src: nl.NkiTensor, rows: int, cols: int) -> nl.NkiTensor:
    """Transpose an SBUF [rows, cols] fp32 tile to a new SBUF [cols, rows] tile via PSUM."""
    t_psum = nl.ndarray((cols, rows), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=t_psum[0:cols, 0:rows], data=src[0:rows, 0:cols])
    t_sb = nl.ndarray((cols, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb[0:cols, 0:rows], src=t_psum[0:cols, 0:rows])
    return t_sb


def _compute_chunk_shared_projections_tiled(
    B_hbm: nl.NkiTensor,
    C_hbm: nl.NkiTensor,
    batch_idx: int,
    chunk_start: int,
    chunk_size: int,
    dstate: int,
    dstate_tile: int,
    num_dstate_tiles: int,
    triu_sb: nl.NkiTensor,
) -> tuple:
    """Compute B, C transposes and masked CB^T, tiled over dstate.

    Args:
        B_hbm (nl.NkiTensor): [batch, seqlen, dstate], Input projection B in HBM.
        C_hbm (nl.NkiTensor): [batch, seqlen, dstate], Output projection C in HBM.
        batch_idx (int): Current batch index.
        chunk_start (int): Starting sequence position for this chunk.
        chunk_size (int): Size of each chunk (Q).
        dstate (int): Full state dimension size.
        dstate_tile (int): Tile size for dstate (min(dstate, P_MAX)).
        num_dstate_tiles (int): Number of dstate tiles.
        triu_sb (nl.NkiTensor): [Q, Q], Upper-triangular mask in SBUF.

    Returns:
        tuple: (B_T_tiles, C_T_tiles, B_f32_tiles, CB_masked)
    """
    Q = chunk_size

    B_T_tiles = []
    C_T_tiles = []
    B_f32_tiles = []
    CB_accum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=CB_accum, value=0.0)

    for n_tile_idx in range(num_dstate_tiles):
        n_start = n_tile_idx * dstate_tile
        n_size = min(dstate_tile, dstate - n_start)

        B_sb = nl.ndarray((Q, n_size), dtype=B_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=B_sb[0:Q, 0:n_size],
            src=B_hbm[batch_idx, chunk_start : chunk_start + Q, n_start : n_start + n_size],
        )
        C_sb = nl.ndarray((Q, n_size), dtype=C_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=C_sb[0:Q, 0:n_size],
            src=C_hbm[batch_idx, chunk_start : chunk_start + Q, n_start : n_start + n_size],
        )

        B_f32 = nl.ndarray((Q, n_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=B_f32[0:Q, 0:n_size], src=B_sb[0:Q, 0:n_size])
        C_f32 = nl.ndarray((Q, n_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=C_f32[0:Q, 0:n_size], src=C_sb[0:Q, 0:n_size])

        C_T_psum = nl.ndarray((n_size, Q), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=C_T_psum[0:n_size, 0:Q], data=C_f32[0:Q, 0:n_size])
        C_T = nl.ndarray((n_size, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=C_T[0:n_size, 0:Q], src=C_T_psum[0:n_size, 0:Q])

        B_T_psum = nl.ndarray((n_size, Q), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=B_T_psum[0:n_size, 0:Q], data=B_f32[0:Q, 0:n_size])
        B_T = nl.ndarray((n_size, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=B_T[0:n_size, 0:Q], src=B_T_psum[0:n_size, 0:Q])

        CB_tile_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=CB_tile_psum[0:Q, 0:Q],
            stationary=B_T[0:n_size, 0:Q],
            moving=C_T[0:n_size, 0:Q],
        )
        CB_tile = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=CB_tile[0:Q, 0:Q], src=CB_tile_psum[0:Q, 0:Q])
        nisa.tensor_tensor(
            dst=CB_accum[0:Q, 0:Q],
            data1=CB_accum[0:Q, 0:Q],
            data2=CB_tile[0:Q, 0:Q],
            op=nl.add,
        )

        B_T_tiles.append(B_T)
        C_T_tiles.append(C_T)
        B_f32_tiles.append(B_f32)

    nisa.tensor_tensor(
        dst=CB_accum[0:Q, 0:Q],
        data1=CB_accum[0:Q, 0:Q],
        data2=triu_sb[0:Q, 0:Q],
        op=nl.multiply,
    )

    return B_T_tiles, C_T_tiles, B_f32_tiles, CB_accum


def _compute_ssd_head_tile(
    x_hbm: nl.NkiTensor,
    y_hbm: nl.NkiTensor,
    state_sb: nl.NkiTensor,
    batch_idx: int,
    global_head_idx: int,
    chunk_start: int,
    d_start: int,
    d_size: int,
    w_intra_col: nl.NkiTensor,
    w_state_col: nl.NkiTensor,
    exp_cs_col: nl.NkiTensor,
    exp_cs_last_N: nl.NkiTensor,
    CB_masked: nl.NkiTensor,
    B_T_tiles: list,
    C_T_tiles: list,
    B_f32_tiles: list,
    D_col: nl.NkiTensor,
    Q: int,
    dstate: int,
    dstate_tile: int,
    num_dstate_tiles: int,
    has_D: bool,
) -> None:
    """Compute one headdim tile of one head's SSD output and update state in-place.

    State lives in SBUF [dstate, headdim] and is sliced along headdim per tile.
    Only matmul results use PSUM (which has the F_MAX constraint).

    Both output terms share the exp(cs) scale, so they accumulate in one PSUM tile:
        y = exp(cs) * (CB_masked @ (x * w_intra) + C @ state) [+ D * x]
        state = exp(cs_last) * state + B^T @ (x * w_state)
    PSUM results are consumed directly, with no PSUM -> SBUF staging copies.

    Args:
        x_hbm (nl.NkiTensor): [batch, nheads, seqlen, headdim], Input in HBM.
        y_hbm (nl.NkiTensor): [batch, nheads, seqlen, headdim], Output in HBM.
        state_sb (nl.NkiTensor): [dstate, headdim], SSM state in SBUF (modified in-place).
        batch_idx (int): Current batch index.
        global_head_idx (int): Global head index.
        chunk_start (int): Starting sequence position.
        d_start (int): Starting headdim offset for this tile.
        d_size (int): Size of this headdim tile.
        w_intra_col (nl.NkiTensor): [Q, 1], dt * exp(-cs) column in SBUF.
        w_state_col (nl.NkiTensor): [Q, 1], dt * exp(cs[-1] - cs) column in SBUF.
        exp_cs_col (nl.NkiTensor): [Q, 1], exp(cs) column in SBUF.
        exp_cs_last_N (nl.NkiTensor): [dstate, 1], exp(cs[-1]) replicated over dstate in SBUF.
        CB_masked (nl.NkiTensor): [Q, Q], Causal-masked CB^T in SBUF.
        B_T_tiles (list): List of [tile_n, Q] transposed B tiles.
        C_T_tiles (list): List of [tile_n, Q] transposed C tiles.
        B_f32_tiles (list): List of [Q, tile_n] B tiles.
        D_col (nl.NkiTensor): [Q, 1], Skip connection weight replicated over tokens, or None.
        Q (int): Chunk size.
        dstate (int): Full state dimension.
        dstate_tile (int): Tile size for dstate.
        num_dstate_tiles (int): Number of dstate tiles.
        has_D (bool): Whether to apply skip connection.
    """
    # Load x tile
    x_sb = nl.ndarray((Q, d_size), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=x_sb[0:Q, 0:d_size],
        src=x_hbm[batch_idx, global_head_idx, chunk_start : chunk_start + Q, d_start : d_start + d_size],
    )

    # Matmul inputs: x scaled by the per-token columns (dt already folded in)
    X_scaled = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=X_scaled[0:Q, 0:d_size],
        data=x_sb[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=w_intra_col[0:Q, 0:1],
    )
    dtx_decayed = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=dtx_decayed[0:Q, 0:d_size],
        data=x_sb[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=w_state_col[0:Q, 0:1],
    )

    # Y_intra + Y_off accumulated in one PSUM tile. Reads state before it is updated below.
    y_psum = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=y_psum[0:Q, 0:d_size],
        stationary=CB_masked[0:Q, 0:Q],
        moving=X_scaled[0:Q, 0:d_size],
        accumulate=False,
    )
    for n_tile_idx in range(num_dstate_tiles):
        n_start = n_tile_idx * dstate_tile
        n_size = min(dstate_tile, dstate - n_start)
        nisa.nc_matmul(
            dst=y_psum[0:Q, 0:d_size],
            stationary=C_T_tiles[n_tile_idx][0:n_size, 0:Q],
            moving=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            accumulate=True,
        )

    # y = exp(cs) * y_psum, read straight from PSUM
    y_chunk = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=y_chunk[0:Q, 0:d_size],
        data=y_psum[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=exp_cs_col[0:Q, 0:1],
    )
    if has_D:
        # y += D * x in one fused op. Kept SBUF-only: pairing a sub-128-partition PSUM
        # tile with an SBUF tile in one instruction fails the partition verifier (Q < 128).
        nisa.scalar_tensor_tensor(
            dst=y_chunk[0:Q, 0:d_size],
            data=x_sb[0:Q, 0:d_size],
            op0=nl.multiply,
            operand0=D_col[0:Q, 0:1],
            op1=nl.add,
            operand1=y_chunk[0:Q, 0:d_size],
        )

    # State update in one fused op per dstate tile: state = state * exp(cs_last) + B^T @ dtx_decayed
    for n_tile_idx in range(num_dstate_tiles):
        n_start = n_tile_idx * dstate_tile
        n_size = min(dstate_tile, dstate - n_start)

        state_delta_psum = nl.ndarray((n_size, d_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=state_delta_psum[0:n_size, 0:d_size],
            stationary=B_f32_tiles[n_tile_idx][0:Q, 0:n_size],
            moving=dtx_decayed[0:Q, 0:d_size],
        )
        nisa.scalar_tensor_tensor(
            dst=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            data=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            op0=nl.multiply,
            operand0=exp_cs_last_N[n_start : n_start + n_size, 0:1],
            op1=nl.add,
            operand1=state_delta_psum[0:n_size, 0:d_size],
        )

    nisa.dma_copy(
        dst=y_hbm[batch_idx, global_head_idx, chunk_start : chunk_start + Q, d_start : d_start + d_size],
        src=y_chunk[0:Q, 0:d_size],
    )
