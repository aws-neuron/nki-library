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

"""Selective scan (SSM) kernel for NKI. Implements fused Mamba-style discretization + recurrence + output projection."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from .ssd_utils import compute_lnc_sharding

P_MAX = 128  # Partition dimension max
F_TILE_SIZE = 512  # Free dimension tile size (smaller than linear_scan due to more SBUF pressure)


@nki.jit
def selective_scan(
    x: nl.ndarray,
    dt: nl.ndarray,
    A: nl.ndarray,
    B: nl.ndarray,
    C: nl.ndarray,
    D: nl.ndarray = None,
    initial_state: nl.ndarray = None,
) -> tuple:
    """
    Selective scan (SSM) as in Mamba models.

    Performs fused discretization, linear recurrence, and output projection in a
    single kernel. For each state dimension n and time step t:

        decay[t] = exp(dt[t] * A[:, n])
        inp[t] = dt[t] * x[t] * B[:, n, t]
        state[t] = decay[t] * state[t-1] + inp[t]
        y[t] += C[:, n, t] * state[t]
    y += D * x  (optional skip connection)

    Dimensions:
        B_dim: Batch size
        channels: Number of channels (partition dimension, tiled at P_MAX=128)
        L: Sequence length (free dimension, tiled at F_TILE_SIZE=512)
        state_size: SSM state dimension

    Args:
        x (nl.ndarray): Input tensor of shape [B_dim, channels, L].
        dt (nl.ndarray): Time step tensor of shape [B_dim, channels, L]. Should be positive.
        A (nl.ndarray): State transition matrix of shape [channels, state_size]. Typically negative.
        B (nl.ndarray): Input projection matrix of shape [B_dim, state_size, L].
        C (nl.ndarray): Output projection matrix of shape [B_dim, state_size, L].
        D (nl.ndarray, optional): Skip connection weights of shape [channels]. Default: None.
        initial_state (nl.ndarray, optional): Initial hidden state of shape
            [B_dim, channels, state_size]. Default: None (zeros).

    Returns:
        tuple: (y, final_state)
            - y (nl.ndarray): Output tensor of shape [B_dim, channels, L] with same dtype as x.
            - final_state (nl.ndarray): Final hidden state of shape [B_dim, channels, state_size]
              in float32.

    Notes:
        - Uses float32 accumulation internally for numerical stability
        - A should be negative for stable recurrence (decay < 1)
        - dt should be positive; discretization computes exp(dt * A)
        - The scan is sequential along the L dimension but parallel across channels
        - Accumulation across state dimensions uses SBUF per free tile to avoid
          HBM read-modify-write (which requires trn2 shared memory)
        - Carries between free tiles are stored in the final_state HBM tensor

    Pseudocode:
        for each batch b:
            for each channel tile p_tile:
                for each free tile f_tile (sequential):
                    y_tile_accum = 0  (in SBUF)
                    for each state dim n (sequential):
                        carry = load(final_state[b, p_tile, n])
                        deltaA = exp(dt * A[:, n])
                        deltaBx = dt * x * B[:, n, :]
                        state = scan(deltaA, deltaBx, carry)
                        final_state[b, p_tile, n] = state[:, -1]
                        y_tile_accum += C[:, n, :] * state
                    if D: y_tile_accum += D * x
                    store y_tile_accum -> y[b, p_tile, f_tile]
    """
    # Input validation
    kernel_assert(len(x.shape) == 3, f"x must be 3D (B, channels, L), got shape {x.shape}")
    kernel_assert(len(dt.shape) == 3, f"dt must be 3D (B, channels, L), got shape {dt.shape}")
    kernel_assert(len(A.shape) == 2, f"A must be 2D (channels, state_size), got shape {A.shape}")
    kernel_assert(len(B.shape) == 3, f"B must be 3D (B, state_size, L), got shape {B.shape}")
    kernel_assert(len(C.shape) == 3, f"C must be 3D (B, state_size, L), got shape {C.shape}")

    batch_size = x.shape[0]
    channels = x.shape[1]
    L = x.shape[2]
    state_size = A.shape[1]

    kernel_assert(dt.shape == x.shape, f"dt shape {dt.shape} must match x shape {x.shape}")
    kernel_assert(A.shape[0] == channels, f"A.shape[0]={A.shape[0]} must equal channels={channels}")
    kernel_assert(B.shape[0] == batch_size, f"B batch dim {B.shape[0]} must match x batch dim {batch_size}")
    kernel_assert(B.shape[1] == state_size, f"B.shape[1]={B.shape[1]} must equal state_size={state_size}")
    kernel_assert(B.shape[2] == L, f"B.shape[2]={B.shape[2]} must equal L={L}")
    kernel_assert(C.shape == B.shape, f"C shape {C.shape} must match B shape {B.shape}")

    num_f_tiles = div_ceil(L, F_TILE_SIZE)
    num_p_tiles = div_ceil(channels, P_MAX)
    tiles_per_core, tile_offset = compute_lnc_sharding(num_p_tiles)

    # Allocate outputs on shared HBM (write-only from kernel perspective)
    y = nl.ndarray((batch_size, channels, L), dtype=x.dtype, buffer=nl.shared_hbm)
    final_state = nl.ndarray((batch_size, channels, state_size), dtype=nl.float32, buffer=nl.shared_hbm)

    # Reshape D to 2D for clean slicing in tile loops
    if D != None:
        D_2d = D.reshape((channels, 1))

    # Reshape B and C to 2D: (batch*state_size, L) for clean DMA access
    B_2d = B.reshape((batch_size * state_size, L))
    C_2d = C.reshape((batch_size * state_size, L))

    for batch_idx in range(batch_size):
        for local_tile_idx in range(tiles_per_core):
            global_tile_idx = tile_offset + local_tile_idx
            p_start = global_tile_idx * P_MAX
            p_size = min(P_MAX, channels - p_start)

            # Preload full A block once per (batch, p_tile)
            A_block = nl.ndarray((P_MAX, state_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=A_block, value=0.0)
            nisa.dma_copy(
                dst=A_block[0:p_size, 0:state_size],
                src=A[p_start : p_start + p_size, 0:state_size],
            )

            # Ones vector for matmul-based partition broadcast: (1, P_MAX)
            # Must match B/C dtype for nc_matmul (both inputs must share same dtype)
            ones_p = nl.ndarray((1, P_MAX), dtype=B.dtype, buffer=nl.sbuf)
            nisa.memset(dst=ones_p, value=1.0)

            # Carry state in SBUF — persists across f_tiles
            carry_all = nl.ndarray((P_MAX, state_size), dtype=nl.float32, buffer=nl.sbuf)
            if initial_state != None:
                nisa.dma_copy(
                    dst=carry_all[0:p_size, 0:state_size],
                    src=initial_state[batch_idx, p_start : p_start + p_size, 0:state_size],
                )
            else:
                nisa.memset(dst=carry_all, value=0.0)

            for f_tile_idx in nl.sequential_range(num_f_tiles):
                f_start = f_tile_idx * F_TILE_SIZE
                f_size = min(F_TILE_SIZE, L - f_start)

                # Load dt and x once per f_tile
                dt_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=dt.dtype, buffer=nl.sbuf)
                nisa.memset(dst=dt_sb, value=0.0)
                nisa.dma_copy(
                    dst=dt_sb[0:p_size, 0:f_size],
                    src=dt[batch_idx, p_start : p_start + p_size, f_start : f_start + f_size],
                )

                x_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
                nisa.memset(dst=x_sb, value=0.0)
                nisa.dma_copy(
                    dst=x_sb[0:p_size, 0:f_size],
                    src=x[batch_idx, p_start : p_start + p_size, f_start : f_start + f_size],
                )

                # Precompute dt * x
                dtx_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=dtx_sb[0:p_size, 0:f_size],
                    data1=dt_sb[0:p_size, 0:f_size],
                    data2=x_sb[0:p_size, 0:f_size],
                    op=nl.multiply,
                )

                # y accumulator
                y_tile_accum = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=y_tile_accum, value=0.0)

                for state_idx in nl.static_range(state_size):
                    A_i = A_block[0:p_size, state_idx : state_idx + 1]

                    # deltaA = exp(dt * A_i)
                    deltaA_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(
                        op=nl.exp,
                        data=dt_sb[0:p_size, 0:f_size],
                        scale=A_i,
                        dst=deltaA_sb[0:p_size, 0:f_size],
                    )

                    # Broadcast B row via matmul outer product: ones(1,P) @ B(1,F) -> B_full(P,F)
                    B_row_idx = batch_idx * state_size + state_idx
                    B_row = nl.ndarray((1, F_TILE_SIZE), dtype=B.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=B_row[0:1, 0:f_size],
                        src=B_2d[B_row_idx : B_row_idx + 1, f_start : f_start + f_size],
                    )
                    B_full_psum = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=B_full_psum[0:P_MAX, 0:f_size],
                        stationary=ones_p[0:1, 0:P_MAX],
                        moving=B_row[0:1, 0:f_size],
                    )
                    B_full = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=B_full[0:p_size, 0:f_size],
                        src=B_full_psum[0:p_size, 0:f_size],
                    )
                    deltaBx_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=deltaBx_sb[0:p_size, 0:f_size],
                        data1=dtx_sb[0:p_size, 0:f_size],
                        data2=B_full[0:p_size, 0:f_size],
                        op=nl.multiply,
                    )

                    # Scan (use carry_all slice directly, avoiding intermediate copy)
                    state_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor_scan(
                        dst=state_sb[0:p_size, 0:f_size],
                        data0=deltaA_sb[0:p_size, 0:f_size],
                        data1=deltaBx_sb[0:p_size, 0:f_size],
                        initial=carry_all[0:p_size, state_idx : state_idx + 1],
                        op0=nl.multiply,
                        op1=nl.add,
                    )

                    # Update carry
                    nisa.tensor_copy(
                        dst=carry_all[0:p_size, state_idx : state_idx + 1],
                        src=state_sb[0:p_size, f_size - 1 : f_size],
                    )

                    # Broadcast C row via matmul outer product
                    C_row_idx = batch_idx * state_size + state_idx
                    C_row = nl.ndarray((1, F_TILE_SIZE), dtype=C.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=C_row[0:1, 0:f_size],
                        src=C_2d[C_row_idx : C_row_idx + 1, f_start : f_start + f_size],
                    )
                    C_full_psum = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=C_full_psum[0:P_MAX, 0:f_size],
                        stationary=ones_p[0:1, 0:P_MAX],
                        moving=C_row[0:1, 0:f_size],
                    )
                    C_full = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=C_full[0:p_size, 0:f_size],
                        src=C_full_psum[0:p_size, 0:f_size],
                    )
                    Cs_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=Cs_sb[0:p_size, 0:f_size],
                        data1=C_full[0:p_size, 0:f_size],
                        data2=state_sb[0:p_size, 0:f_size],
                        op=nl.multiply,
                    )

                    # Accumulate
                    nisa.tensor_tensor(
                        dst=y_tile_accum[0:p_size, 0:f_size],
                        data1=y_tile_accum[0:p_size, 0:f_size],
                        data2=Cs_sb[0:p_size, 0:f_size],
                        op=nl.add,
                    )

                # Add D*x skip connection
                if D != None:
                    D_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(dst=D_sb, value=0.0)
                    nisa.dma_copy(
                        dst=D_sb[0:p_size, 0:1],
                        src=D_2d[p_start : p_start + p_size, 0:1],
                    )
                    Dx_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=Dx_sb[0:p_size, 0:f_size],
                        data=x_sb[0:p_size, 0:f_size],
                        op0=nl.multiply,
                        operand0=D_sb[0:p_size, 0:1],
                    )
                    nisa.tensor_tensor(
                        dst=y_tile_accum[0:p_size, 0:f_size],
                        data1=y_tile_accum[0:p_size, 0:f_size],
                        data2=Dx_sb[0:p_size, 0:f_size],
                        op=nl.add,
                    )

                # Store y tile
                nisa.dma_copy(
                    dst=y[batch_idx, p_start : p_start + p_size, f_start : f_start + f_size],
                    src=y_tile_accum[0:p_size, 0:f_size],
                )

            # Copy final carry to output
            nisa.dma_copy(
                dst=final_state[batch_idx, p_start : p_start + p_size, 0:state_size],
                src=carry_all[0:p_size, 0:state_size],
            )

    return y, final_state
