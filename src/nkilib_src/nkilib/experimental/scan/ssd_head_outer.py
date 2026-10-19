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

"""Head-outer SSD (State Space Duality) scan kernel for Mamba-2 prefill."""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil
from .ssd_utils import broadcast_scalar_to_column, compute_cumulative_decay, compute_lnc_sharding

P_MAX = nl.tile_size.pmax


def ssd_head_outer(
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
    """Head-outer SSD scan for Mamba-2 prefill.

    Uses head-outer, chunk-inner loop nesting. State stays in SBUF across
    chunks within each head, avoiding HBM round-trips for state. B and C are
    loaded per head per chunk (no sharing). Better than chunk-outer when few
    heads make B/C sharing savings small relative to state HBM traffic.

    Dimensions:
        batch: Batch size.
        nheads: Number of SSM heads.
        seqlen: Sequence length (must be divisible by chunk_size).
        headdim: Per-head dimension (<= 512 for PSUM free dim limit).
        dstate: SSM state dimension (<= 128 for partition dimension).
        Q: Chunk size (= chunk_size, <= 128).

    Args:
        x (nl.NkiTensor): [batch, nheads, seqlen, headdim], Input activations.
        dt (nl.NkiTensor): [batch, nheads, seqlen], Softplus'd timesteps.
        A (nl.NkiTensor): [nheads], State transition scalars.
        B (nl.NkiTensor): [batch, seqlen, dstate], Input projection.
        C (nl.NkiTensor): [batch, seqlen, dstate], Output projection.
        chunk_size (int): Chunk size for parallel scan. Must be <= 128.
        D (nl.NkiTensor, optional): [nheads], Skip connection weights.
        initial_state (nl.NkiTensor, optional): [batch, nheads, dstate, headdim].
        causal_mask (nl.NkiTensor): [Q, Q], Lower-triangular mask. Required.

    Returns:
        tuple: (y, final_state)
            - y (nl.NkiTensor): [batch, nheads, seqlen, headdim], Output, same dtype as x.
            - final_state (nl.NkiTensor): [batch, nheads, dstate, headdim], float32.

    Notes:
        - headdim <= 512 (no PSUM tiling in this path)
        - State persists in SBUF across chunks — no HBM state traffic per chunk

    Pseudocode:
        for each head (LNC sharded):
            state = initial_state or zeros
            for each chunk:
                B_f32, C_T, CB_masked = projections(B, C, causal_mask)
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

    A_2d = A.reshape((nheads, 1))
    dt_2d = dt.reshape((batch * nheads, seqlen))
    dt_flat = dt.reshape((batch * nheads * seqlen, 1))

    y = nl.ndarray((batch, nheads, seqlen, headdim), dtype=x.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray(
        (batch, nheads, dstate, headdim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    has_D = D != None
    if has_D:
        D_2d = D.reshape((nheads, 1))

    heads_per_core, head_offset = compute_lnc_sharding(nheads)

    for batch_idx in range(batch):
        for local_head_idx in range(heads_per_core):
            head_idx = head_offset + local_head_idx

            A_h = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=A_h[0:1, 0:1], src=A_2d[head_idx : head_idx + 1, 0:1])

            causal_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=causal_sb[0:Q, 0:Q], src=causal_mask[0:Q, 0:Q])

            ones_sb = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_sb, value=1.0)
            zero_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zero_scalar, value=0.0)

            state_sb = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
            if initial_state != None:
                nisa.dma_copy(
                    dst=state_sb[0:dstate, 0:headdim],
                    src=initial_state[batch_idx, head_idx, 0:dstate, 0:headdim],
                )
            else:
                nisa.memset(dst=state_sb, value=0.0)

            D_Q = None
            if has_D:
                D_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=D_val[0:1, 0:1], src=D_2d[head_idx : head_idx + 1, 0:1])
                D_Q = broadcast_scalar_to_column(D_val, Q)

            triu_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=triu_psum[0:Q, 0:Q], data=causal_sb[0:Q, 0:Q])
            triu_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=triu_sb[0:Q, 0:Q], src=triu_psum[0:Q, 0:Q])

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * Q

                x_sb = nl.ndarray((Q, headdim), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=x_sb[0:Q, 0:headdim],
                    src=x[batch_idx, head_idx, chunk_start : chunk_start + Q, 0:headdim],
                )

                dt_row_idx = batch_idx * nheads + head_idx
                dt_row = nl.ndarray((1, Q), dtype=dt.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dt_row[0:1, 0:Q],
                    src=dt_2d[dt_row_idx : dt_row_idx + 1, chunk_start : chunk_start + Q],
                )

                B_sb = nl.ndarray((Q, dstate), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=B_sb[0:Q, 0:dstate],
                    src=B[batch_idx, chunk_start : chunk_start + Q, 0:dstate],
                )
                C_sb = nl.ndarray((Q, dstate), dtype=C.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=C_sb[0:Q, 0:dstate],
                    src=C[batch_idx, chunk_start : chunk_start + Q, 0:dstate],
                )

                cs_row, exp_cs_col, exp_neg_cs_col = compute_cumulative_decay(
                    dt_row,
                    A_h,
                    ones_sb,
                    zero_scalar,
                    Q,
                )

                dt_flat_start = dt_row_idx * seqlen + chunk_start
                dt_col = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=dt_col[0:Q, 0:1],
                    src=dt_flat[dt_flat_start : dt_flat_start + Q, 0:1],
                )
                dtx = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=dtx[0:Q, 0:headdim],
                    data=x_sb[0:Q, 0:headdim],
                    op0=nl.multiply,
                    operand0=dt_col[0:Q, 0:1],
                )

                B_f32, C_T, CB_masked = _compute_chunk_projections_ho(
                    B_sb,
                    C_sb,
                    triu_sb,
                    Q,
                    dstate,
                )

                y_chunk = _compute_ssd_chunk_ho(
                    x_sb,
                    dtx,
                    cs_row,
                    exp_cs_col,
                    exp_neg_cs_col,
                    state_sb,
                    B_f32,
                    C_T,
                    CB_masked,
                    D_Q,
                    Q,
                    headdim,
                    dstate,
                    has_D,
                )

                nisa.dma_copy(
                    dst=y[batch_idx, head_idx, chunk_start : chunk_start + Q, 0:headdim],
                    src=y_chunk[0:Q, 0:headdim],
                )

            nisa.dma_copy(
                dst=final_state_out[batch_idx, head_idx, 0:dstate, 0:headdim],
                src=state_sb[0:dstate, 0:headdim],
            )

    return y, final_state_out


def _compute_chunk_projections_ho(
    B_sb: nl.NkiTensor,
    C_sb: nl.NkiTensor,
    triu_sb: nl.NkiTensor,
    Q: int,
    dstate: int,
) -> tuple:
    """Compute B, C transposes and masked CB^T for one chunk.

    Args:
        B_sb (nl.NkiTensor): [Q, dstate], B chunk in SBUF.
        C_sb (nl.NkiTensor): [Q, dstate], C chunk in SBUF.
        triu_sb (nl.NkiTensor): [Q, Q], Upper-triangular mask in SBUF.
        Q (int): Chunk size.
        dstate (int): State dimension.

    Returns:
        tuple: (B_f32, C_T, CB_masked)
    """
    B_f32 = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=B_f32[0:Q, 0:dstate], src=B_sb[0:Q, 0:dstate])
    C_f32 = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=C_f32[0:Q, 0:dstate], src=C_sb[0:Q, 0:dstate])

    C_T_psum = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=C_T_psum[0:dstate, 0:Q], data=C_f32[0:Q, 0:dstate])
    C_T = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=C_T[0:dstate, 0:Q], src=C_T_psum[0:dstate, 0:Q])

    B_T_psum = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=B_T_psum[0:dstate, 0:Q], data=B_f32[0:Q, 0:dstate])
    B_T = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=B_T[0:dstate, 0:Q], src=B_T_psum[0:dstate, 0:Q])

    CB_T_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=CB_T_psum[0:Q, 0:Q],
        stationary=B_T[0:dstate, 0:Q],
        moving=C_T[0:dstate, 0:Q],
    )
    CB_masked = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=CB_masked[0:Q, 0:Q], src=CB_T_psum[0:Q, 0:Q])
    nisa.tensor_tensor(
        dst=CB_masked[0:Q, 0:Q],
        data1=CB_masked[0:Q, 0:Q],
        data2=triu_sb[0:Q, 0:Q],
        op=nl.multiply,
    )

    return B_f32, C_T, CB_masked


def _compute_ssd_chunk_ho(
    x_sb: nl.NkiTensor,
    dtx: nl.NkiTensor,
    cs_row: nl.NkiTensor,
    exp_cs_col: nl.NkiTensor,
    exp_neg_cs_col: nl.NkiTensor,
    state_sb: nl.NkiTensor,
    B_f32: nl.NkiTensor,
    C_T: nl.NkiTensor,
    CB_masked: nl.NkiTensor,
    D_Q: nl.NkiTensor,
    Q: int,
    headdim: int,
    dstate: int,
    has_D: bool,
) -> nl.NkiTensor:
    """Compute one chunk's SSD output and update state in-place.

    Args:
        x_sb (nl.NkiTensor): [Q, headdim], Input activations.
        dtx (nl.NkiTensor): [Q, headdim], dt * x.
        cs_row (nl.NkiTensor): [1, Q], Cumulative sum row.
        exp_cs_col (nl.NkiTensor): [Q, 1], exp(cs) column.
        exp_neg_cs_col (nl.NkiTensor): [Q, 1], exp(-cs) column.
        state_sb (nl.NkiTensor): [dstate, headdim], State (modified in-place).
        B_f32 (nl.NkiTensor): [Q, dstate], B in float32.
        C_T (nl.NkiTensor): [dstate, Q], Transposed C.
        CB_masked (nl.NkiTensor): [Q, Q], Causal-masked CB^T.
        D_Q (nl.NkiTensor): [Q, 1], Broadcast D or None.
        Q (int): Chunk size.
        headdim (int): Head dimension.
        dstate (int): State dimension.
        has_D (bool): Whether to apply skip connection.

    Returns:
        nl.NkiTensor: [Q, headdim], Output chunk.
    """
    # Y_intra = exp(cs) * (CB_causal @ (exp(-cs) * dtx))
    X_scaled = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=X_scaled[0:Q, 0:headdim],
        data=dtx[0:Q, 0:headdim],
        op0=nl.multiply,
        operand0=exp_neg_cs_col[0:Q, 0:1],
    )
    Y_intra_psum = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=Y_intra_psum[0:Q, 0:headdim],
        stationary=CB_masked[0:Q, 0:Q],
        moving=X_scaled[0:Q, 0:headdim],
    )
    Y_intra = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Y_intra[0:Q, 0:headdim], src=Y_intra_psum[0:Q, 0:headdim])
    nisa.tensor_scalar(
        dst=Y_intra[0:Q, 0:headdim],
        data=Y_intra[0:Q, 0:headdim],
        op0=nl.multiply,
        operand0=exp_cs_col[0:Q, 0:1],
    )

    # Y_off = exp(cs) * (C @ state)
    Y_off_psum = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=Y_off_psum[0:Q, 0:headdim],
        stationary=C_T[0:dstate, 0:Q],
        moving=state_sb[0:dstate, 0:headdim],
    )
    Y_off = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Y_off[0:Q, 0:headdim], src=Y_off_psum[0:Q, 0:headdim])
    nisa.tensor_scalar(
        dst=Y_off[0:Q, 0:headdim],
        data=Y_off[0:Q, 0:headdim],
        op0=nl.multiply,
        operand0=exp_cs_col[0:Q, 0:1],
    )

    # State update
    exp_cs_last = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(op=nl.exp, data=cs_row[0:1, Q - 1 : Q], dst=exp_cs_last[0:1, 0:1])
    exp_cs_last_Q = broadcast_scalar_to_column(exp_cs_last, Q)
    decay_to_end = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_to_end[0:Q, 0:1],
        data1=exp_neg_cs_col[0:Q, 0:1],
        data2=exp_cs_last_Q[0:Q, 0:1],
        op=nl.multiply,
    )
    dtx_state = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=dtx_state[0:Q, 0:headdim],
        data=dtx[0:Q, 0:headdim],
        op0=nl.multiply,
        operand0=decay_to_end[0:Q, 0:1],
    )
    chunk_state_psum = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=chunk_state_psum[0:dstate, 0:headdim],
        stationary=B_f32[0:Q, 0:dstate],
        moving=dtx_state[0:Q, 0:headdim],
    )
    chunk_state = nl.ndarray((dstate, headdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=chunk_state[0:dstate, 0:headdim],
        src=chunk_state_psum[0:dstate, 0:headdim],
    )
    exp_cs_last_N = broadcast_scalar_to_column(exp_cs_last, dstate)
    nisa.tensor_scalar(
        dst=state_sb[0:dstate, 0:headdim],
        data=state_sb[0:dstate, 0:headdim],
        op0=nl.multiply,
        operand0=exp_cs_last_N[0:dstate, 0:1],
    )
    nisa.tensor_tensor(
        dst=state_sb[0:dstate, 0:headdim],
        data1=state_sb[0:dstate, 0:headdim],
        data2=chunk_state[0:dstate, 0:headdim],
        op=nl.add,
    )

    # Combine output
    y_chunk = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=y_chunk[0:Q, 0:headdim],
        data1=Y_intra[0:Q, 0:headdim],
        data2=Y_off[0:Q, 0:headdim],
        op=nl.add,
    )
    if has_D:
        Dx = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=Dx[0:Q, 0:headdim],
            data=x_sb[0:Q, 0:headdim],
            op0=nl.multiply,
            operand0=D_Q[0:Q, 0:1],
        )
        nisa.tensor_tensor(
            dst=y_chunk[0:Q, 0:headdim],
            data1=y_chunk[0:Q, 0:headdim],
            data2=Dx[0:Q, 0:headdim],
            op=nl.add,
        )

    return y_chunk
