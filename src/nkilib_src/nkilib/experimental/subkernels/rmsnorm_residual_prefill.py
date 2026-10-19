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

"""Fused pre-norm residual add + RMSNorm in [T, H] layout"""

from typing import List, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.logging import get_logger

_SBUF_SCRATCH_RESERVE = 1024
_MAX_TILE_INTERLEAVE = 4

# H tile width for the split load; sets the DMA packet size to _H_TILE_SIZE * dtype bytes.
_H_TILE_SIZE = 1536


@nki.jit
def rmsnorm_residual_prefill(
    hidden: nl.NkiTensor,
    gamma: nl.NkiTensor,
    residual: Optional[nl.NkiTensor] = None,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
) -> List[nl.NkiTensor]:
    """
    Pre-norm residual add followed by RMSNorm, for prefill-shaped inputs (T >= 128).

    When residual is given the add runs in the DMA engines via dma_compute, so the sum is
    produced during the load; it is also spilled to HBM because it is the next block's
    residual. Without it the kernel is a plain RMSNorm and returns only out.

    Dimensions:
        T: Token count (batch x sequence, flattened)
        H: Hidden dimension size, padded

    Args:
        hidden (nl.NkiTensor): [T, H] or [B, S, H], Hidden states on HBM
        gamma (nl.NkiTensor): [1, H], Gamma tensor used in normalization on HBM
        residual (Optional[nl.NkiTensor]): [T, H] or [B, S, H], Residual to add before
            normalizing, on HBM. Omit for a plain RMSNorm
        eps (float): Epsilon to maintain numerical stability. Default is 1e-6
        hidden_actual (Optional[int]): True hidden width when H is padded, used as the
            mean divisor. Defaults to H

    Returns:
        out (nl.NkiTensor): [T, H], Normalized output
        residual_out (nl.NkiTensor): [T, H], hidden + residual, only when residual is given

    """
    kernel_assert(len(hidden.shape) in (2, 3), f"hidden must be [T, H] or [B, S, H], got {hidden.shape}")
    has_residual = residual != None
    if has_residual:
        kernel_assert(
            tuple(residual.shape) == tuple(hidden.shape),
            f"residual must match hidden {tuple(hidden.shape)}, got {tuple(residual.shape)}",
        )
    if len(hidden.shape) == 3:
        B, S, H = hidden.shape
        T = B * S
    else:
        T, H = hidden.shape
    hidden_view = hidden.reshape((T, H))
    residual_view = residual.reshape((T, H)) if has_residual else None
    kernel_assert(
        tuple(gamma.shape) == (1, H), f"Malformed shape of gamma, expected [1, {H}], got {tuple(gamma.shape)}"
    )

    T0 = nl.tile_size.pmax
    H0 = nl.tile_size.pmax
    inter_dtype = nl.float32
    h_tile_size = min(H, _H_TILE_SIZE)

    if hidden_actual is None:
        hidden_actual = H
    kernel_assert(0 < hidden_actual <= H, f"hidden_actual ({hidden_actual}) must be in (0, H={H}]")
    hidden_scale = 1.0 / hidden_actual

    _, num_shards, shard_id = get_verified_program_sharding_info("rmsnorm", (0, 1))
    nominal_shard = div_ceil(T, num_shards)
    shard_offset = shard_id * nominal_shard
    shard_size = min(nominal_shard, T - shard_offset) if shard_offset < T else 0

    # PSUM holds fp32 on every target and bf16 only from gen4, so stage the broadcast at
    # gamma's own dtype where the target allows it rather than always widening to fp32.
    gamma_psum_dtype = (
        gamma.dtype if (gamma.dtype == nl.bfloat16 and nisa.get_nc_version() >= nisa.nc_version.gen4) else inter_dtype
    )

    GAMMA_TILE = nl.tile_size.psum_bank_fmax
    num_h_tiles = div_ceil(H, h_tile_size)

    out = nl.ndarray((T, H), dtype=hidden.dtype, buffer=nl.shared_hbm)
    residual_out = nl.ndarray((T, H), dtype=hidden.dtype, buffer=nl.shared_hbm) if has_residual else None

    sb_upper_bound = nl.tile_size.total_available_sbuf_size - _SBUF_SCRATCH_RESERVE
    sbm = SbufManager(0, sb_upper_bound, get_logger("rmsnorm_residual_prefill"))

    sbm.open_scope(name="invariants")

    # Gamma arrives as a single row, so land it on one partition first.
    gamma_loaded = sbm.alloc_stack((1, H), dtype=gamma.dtype, name="gamma_loaded")
    nisa.dma_copy(dst=gamma_loaded, src=gamma, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    gamma_bc_ones = sbm.alloc_stack((1, H0), dtype=gamma.dtype, name="gamma_bc_ones")
    nisa.memset(gamma_bc_ones, value=1.0)

    # Broadcast that row to all partitions on the idle PE, an outer product with ones.
    gamma_sb = sbm.alloc_stack((H0, H), dtype=gamma.dtype, name="gamma_sb")
    for h_start in range(0, H, GAMMA_TILE):
        tile_H = min(H - h_start, GAMMA_TILE)
        broadcast_psum = nl.ndarray((H0, tile_H), dtype=gamma_psum_dtype, buffer=nl.psum)
        nisa.nc_matmul(
            dst=broadcast_psum[0:H0, 0:tile_H],
            stationary=gamma_bc_ones[0:1, 0:H0],
            moving=gamma_loaded[0:1, h_start : h_start + tile_H],
            is_stationary_onezero=True,
        )
        nisa.tensor_copy(dst=gamma_sb[0:H0, h_start : h_start + tile_H], src=broadcast_psum[0:H0, 0:tile_H])

    eps_bias = sbm.alloc_stack((T0, 1), dtype=inter_dtype, name="eps_bias")
    nisa.memset(eps_bias, value=eps)

    # Bytes one loop iteration costs per partition, which the interleave budget divides into
    # the SBUF left free by the loop-invariant allocations.
    per_tile_bytes = (
        2 * sizeinbytes(hidden.dtype) * H + sizeinbytes(inter_dtype) * h_tile_size + sizeinbytes(inter_dtype)
    )
    tile_interleave = max(1, min(_MAX_TILE_INTERLEAVE, sbm.get_free_space() // per_tile_bytes))

    sbm.open_scope(interleave_degree=tile_interleave, name="token_tiles")

    for t_start in range(shard_offset, shard_offset + shard_size, T0):
        tile_T = min(T0, shard_offset + shard_size - t_start)
        tile_idx = (t_start - shard_offset) // T0

        tile_buf = sbm.alloc_stack((T0, H), dtype=hidden.dtype, name=f"tile_buf_t{tile_idx}")
        norm_out = sbm.alloc_stack((T0, H), dtype=hidden.dtype, name=f"norm_out_t{tile_idx}")
        square_sb = sbm.alloc_stack((T0, h_tile_size), dtype=inter_dtype, name=f"square_sb_t{tile_idx}")
        reduced_sq = sbm.alloc_stack((T0, 1), dtype=inter_dtype, name=f"reduced_sq_t{tile_idx}")

        # Add the residual during the load, then square each H tile as soon as it lands.
        for h_idx, h_start in enumerate(range(0, H, h_tile_size)):
            tile_H = min(H - h_start, h_tile_size)
            if has_residual:
                nisa.dma_compute(
                    dst=tile_buf[0:tile_T, h_start : h_start + tile_H],
                    srcs=[
                        hidden_view[t_start : t_start + tile_T, h_start : h_start + tile_H],
                        residual_view[t_start : t_start + tile_T, h_start : h_start + tile_H],
                    ],
                    reduce_op=nl.add,
                )
            else:
                nisa.dma_copy(
                    dst=tile_buf[0:tile_T, h_start : h_start + tile_H],
                    src=hidden_view[t_start : t_start + tile_T, h_start : h_start + tile_H],
                    dge_mode=nisa.dge_mode.swdge,
                )
            nisa.activation(
                dst=square_sb[0:tile_T, 0:tile_H],
                op=nl.square,
                data=tile_buf[0:tile_T, h_start : h_start + tile_H],
                reduce_op=nl.add,
                reduce_cmd=nisa.reduce_cmd.reset_reduce if h_idx == 0 else nisa.reduce_cmd.reduce,
                reduce_res=reduced_sq[0:tile_T, 0:1] if h_idx == num_h_tiles - 1 else None,
            )

        if has_residual:
            nisa.dma_copy(
                dst=residual_out[t_start : t_start + tile_T, 0:H],
                src=tile_buf[0:tile_T, 0:H],
                dge_mode=nisa.dge_mode.swdge,
            )

        # Sum of squares -> 1/rms, with the mean divide and +eps folded into one op.
        nisa.activation(
            dst=reduced_sq[0:tile_T, 0:1],
            op=nl.rsqrt,
            data=reduced_sq[0:tile_T, 0:1],
            bias=eps_bias[0:tile_T, 0:1],
            scale=hidden_scale,
        )

        # Apply both the normalization and gamma in a single pass over the tile.
        for h_start in range(0, H, h_tile_size):
            tile_H = min(H - h_start, h_tile_size)
            nisa.scalar_tensor_tensor(
                dst=norm_out[0:tile_T, h_start : h_start + tile_H],
                data=tile_buf[0:tile_T, h_start : h_start + tile_H],
                op0=nl.multiply,
                operand0=reduced_sq[0:tile_T, 0:1],
                op1=nl.multiply,
                operand1=gamma_sb[0:tile_T, h_start : h_start + tile_H],
            )

        nisa.dma_copy(
            dst=out[t_start : t_start + tile_T, 0:H],
            src=norm_out[0:tile_T, 0:H],
            dge_mode=nisa.dge_mode.swdge,
        )

        sbm.increment_section()

    sbm.close_scope()
    sbm.close_scope()

    outputs = [out]
    if has_residual:
        outputs.append(residual_out)
    return outputs
