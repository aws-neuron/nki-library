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
from .ssd_utils import broadcast_scalar_to_column, compute_cumulative_decay, compute_lnc_sharding

P_MAX = nl.tile_size.pmax
F_MAX = nl.tile_size.psum_fmax


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
            for each head (LNC sharded):
                cs = cumsum(dt * A)
                Y_intra = exp(cs) * (CB_masked @ (exp(-cs) * dt * x))
                Y_off = exp(cs) * (C @ state)
                state = exp(cs[-1]) * state + B^T @ (dt * x * decay)
                y = Y_intra + Y_off + D * x
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

    A_2d = A.reshape((nheads, 1))
    dt_2d = dt.reshape((batch * nheads, seqlen))
    dt_flat = dt.reshape((batch * nheads * seqlen, 1))

    has_D = D != None
    if has_D:
        D_2d = D.reshape((nheads, 1))

    y = nl.ndarray((batch, nheads, seqlen, headdim), dtype=x.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray(
        (batch, nheads, dstate, headdim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

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
            state_init = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
            if initial_state != None:
                nisa.dma_copy(
                    dst=state_init[0:dstate, 0:headdim],
                    src=initial_state[batch_idx, global_head_idx, 0:dstate, 0:headdim],
                )
            else:
                nisa.memset(dst=state_init, value=0.0)
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

            # Scan constants shared across heads
            ones_row = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_row, value=1.0)
            zero_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zero_scalar, value=0.0)

            # Head-inner loop (LNC sharded)
            for local_head_idx in range(heads_per_core):
                global_head_idx = head_offset + local_head_idx

                A_head = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=A_head[0:1, 0:1],
                    src=A_2d[global_head_idx : global_head_idx + 1, 0:1],
                )

                dt_row_idx = batch_idx * nheads + global_head_idx
                dt_row = nl.ndarray((1, Q), dtype=dt.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dt_row[0:1, 0:Q],
                    src=dt_2d[dt_row_idx : dt_row_idx + 1, chunk_start : chunk_start + Q],
                )
                dt_flat_start = dt_row_idx * seqlen + chunk_start
                dt_col = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dt_col[0:Q, 0:1],
                    src=dt_flat[dt_flat_start : dt_flat_start + Q, 0:1],
                )

                cs_row, exp_cs_col, exp_neg_cs_col = compute_cumulative_decay(
                    dt_row,
                    A_head,
                    ones_row,
                    zero_scalar,
                    Q,
                )

                D_scalar = None
                if has_D:
                    D_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=D_scalar[0:1, 0:1],
                        src=D_2d[global_head_idx : global_head_idx + 1, 0:1],
                    )

                # State update decay factors (shared across headdim tiles)
                exp_cs_last = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(op=nl.exp, data=cs_row[0:1, Q - 1 : Q], dst=exp_cs_last[0:1, 0:1])
                exp_cs_last_Q = broadcast_scalar_to_column(exp_cs_last, Q)
                decay_factor = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=decay_factor[0:Q, 0:1],
                    data1=exp_neg_cs_col[0:Q, 0:1],
                    data2=exp_cs_last_Q[0:Q, 0:1],
                    op=nl.multiply,
                )

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
                        dt_col,
                        exp_cs_col,
                        exp_neg_cs_col,
                        exp_cs_last,
                        decay_factor,
                        CB_masked,
                        B_T_tiles,
                        C_T_tiles,
                        B_f32_tiles,
                        D_scalar,
                        Q,
                        dstate,
                        dstate,
                        num_dstate_tiles,
                        has_D,
                    )

                # Store state back to HBM after all headdim tiles
                nisa.dma_copy(
                    dst=state_hbm[batch_idx, local_head_idx, 0:dstate, 0:headdim],
                    src=state_sb[0:dstate, 0:headdim],
                )

        # Copy final state to output
        for local_head_idx in range(heads_per_core):
            global_head_idx = head_offset + local_head_idx
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
    dt_col: nl.NkiTensor,
    exp_cs_col: nl.NkiTensor,
    exp_neg_cs_col: nl.NkiTensor,
    exp_cs_last: nl.NkiTensor,
    decay_factor: nl.NkiTensor,
    CB_masked: nl.NkiTensor,
    B_T_tiles: list,
    C_T_tiles: list,
    B_f32_tiles: list,
    D_scalar: nl.NkiTensor,
    Q: int,
    dstate: int,
    dstate_tile: int,
    num_dstate_tiles: int,
    has_D: bool,
) -> None:
    """Compute one headdim tile of one head's SSD output and update state in-place.

    State lives in SBUF [dstate, headdim] and is sliced along headdim per tile.
    Only matmul results use PSUM (which has the F_MAX constraint).

    Args:
        x_hbm (nl.NkiTensor): [batch, nheads, seqlen, headdim], Input in HBM.
        y_hbm (nl.NkiTensor): [batch, nheads, seqlen, headdim], Output in HBM.
        state_sb (nl.NkiTensor): [dstate, headdim], SSM state in SBUF (modified in-place).
        batch_idx (int): Current batch index.
        global_head_idx (int): Global head index.
        chunk_start (int): Starting sequence position.
        d_start (int): Starting headdim offset for this tile.
        d_size (int): Size of this headdim tile.
        dt_col (nl.NkiTensor): [Q, 1], Timestep column in SBUF.
        exp_cs_col (nl.NkiTensor): [Q, 1], exp(cs) column in SBUF.
        exp_neg_cs_col (nl.NkiTensor): [Q, 1], exp(-cs) column in SBUF.
        exp_cs_last (nl.NkiTensor): [1, 1], exp(cs[-1]) scalar in SBUF.
        decay_factor (nl.NkiTensor): [Q, 1], exp(cs[-1] - cs) column in SBUF.
        CB_masked (nl.NkiTensor): [Q, Q], Causal-masked CB^T in SBUF.
        B_T_tiles (list): List of [tile_n, Q] transposed B tiles.
        C_T_tiles (list): List of [tile_n, Q] transposed C tiles.
        B_f32_tiles (list): List of [Q, tile_n] B tiles.
        D_scalar (nl.NkiTensor): [1, 1], Skip connection weight, or None.
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

    # dt * x
    dtx = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=dtx[0:Q, 0:d_size],
        data=x_sb[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=dt_col[0:Q, 0:1],
    )

    # Y_intra = exp(cs) * (CB_causal @ (exp(-cs) * dtx))
    X_scaled = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=X_scaled[0:Q, 0:d_size],
        data=dtx[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=exp_neg_cs_col[0:Q, 0:1],
    )
    Y_intra_psum = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=Y_intra_psum[0:Q, 0:d_size],
        stationary=CB_masked[0:Q, 0:Q],
        moving=X_scaled[0:Q, 0:d_size],
    )
    Y_intra = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Y_intra[0:Q, 0:d_size], src=Y_intra_psum[0:Q, 0:d_size])
    nisa.tensor_scalar(
        dst=Y_intra[0:Q, 0:d_size],
        data=Y_intra[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=exp_cs_col[0:Q, 0:1],
    )

    # Y_off = exp(cs) * (C @ state) — accumulate across dstate tiles, read state from SBUF
    Y_off = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=Y_off, value=0.0)
    for n_tile_idx in range(num_dstate_tiles):
        n_start = n_tile_idx * dstate_tile
        n_size = min(dstate_tile, dstate - n_start)
        Y_off_psum = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=Y_off_psum[0:Q, 0:d_size],
            stationary=C_T_tiles[n_tile_idx][0:n_size, 0:Q],
            moving=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
        )
        Y_off_tile = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Y_off_tile[0:Q, 0:d_size], src=Y_off_psum[0:Q, 0:d_size])
        nisa.tensor_tensor(
            dst=Y_off[0:Q, 0:d_size],
            data1=Y_off[0:Q, 0:d_size],
            data2=Y_off_tile[0:Q, 0:d_size],
            op=nl.add,
        )
    nisa.tensor_scalar(
        dst=Y_off[0:Q, 0:d_size],
        data=Y_off[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=exp_cs_col[0:Q, 0:1],
    )

    # State update: state = exp(cs_last) * state + B^T @ (dtx * decay)
    # State lives in SBUF — update in-place via slicing along headdim
    dtx_decayed = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=dtx_decayed[0:Q, 0:d_size],
        data=dtx[0:Q, 0:d_size],
        op0=nl.multiply,
        operand0=decay_factor[0:Q, 0:1],
    )
    for n_tile_idx in range(num_dstate_tiles):
        n_start = n_tile_idx * dstate_tile
        n_size = min(dstate_tile, dstate - n_start)

        state_delta_psum = nl.ndarray((n_size, d_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=state_delta_psum[0:n_size, 0:d_size],
            stationary=B_f32_tiles[n_tile_idx][0:Q, 0:n_size],
            moving=dtx_decayed[0:Q, 0:d_size],
        )
        state_delta = nl.ndarray((n_size, d_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_delta[0:n_size, 0:d_size], src=state_delta_psum[0:n_size, 0:d_size])

        exp_cs_last_N = broadcast_scalar_to_column(exp_cs_last, n_size)

        nisa.tensor_scalar(
            dst=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            data=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            op0=nl.multiply,
            operand0=exp_cs_last_N[0:n_size, 0:1],
        )
        nisa.tensor_tensor(
            dst=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            data1=state_sb[n_start : n_start + n_size, d_start : d_start + d_size],
            data2=state_delta[0:n_size, 0:d_size],
            op=nl.add,
        )

    # Combine: y = Y_intra + Y_off [+ D * x]
    y_chunk = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=y_chunk[0:Q, 0:d_size],
        data1=Y_intra[0:Q, 0:d_size],
        data2=Y_off[0:Q, 0:d_size],
        op=nl.add,
    )
    if has_D:
        D_col = broadcast_scalar_to_column(D_scalar, Q)
        Dx = nl.ndarray((Q, d_size), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=Dx[0:Q, 0:d_size],
            data=x_sb[0:Q, 0:d_size],
            op0=nl.multiply,
            operand0=D_col[0:Q, 0:1],
        )
        nisa.tensor_tensor(
            dst=y_chunk[0:Q, 0:d_size],
            data1=y_chunk[0:Q, 0:d_size],
            data2=Dx[0:Q, 0:d_size],
            op=nl.add,
        )

    nisa.dma_copy(
        dst=y_hbm[batch_idx, global_head_idx, chunk_start : chunk_start + Q, d_start : d_start + d_size],
        src=y_chunk[0:Q, 0:d_size],
    )
