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
Down projection sub-kernels with LNC sharding on I (intermediate) dimension.

LNC Sharding Strategy: I dimension
- When LNC=2, weights and scales are sharded on I dimension
- Bias is sharded on H dimension (NC0 loads first half, NC1 loads second half)

These sub-kernels can be used by any algorithm that requires I-sharded down projection,
including all-expert, selective-load, or custom MoE implementations.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

# Shared MX constants
from ...mlp.mlp_tkg.projection_mx_constants import (
    NUM_QUADRANTS_IN_SBUF,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
)

# Common utils
from ...utils.common_types import ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast

# FIXME: add @nki.jit decorator to all sub-kernels when NKIFE-557 is resolved


def load_broadcast_down_weight_scale_bias(
    weight: nl.ndarray,
    scale: nl.ndarray,
    bias: Optional[nl.ndarray],
    expert_idx: int,
    H: int,
    tile_I: int,
    n_I512_tiles: int,
    tile_T: int,
    activation_compute_dtype=nl.bfloat16,
    use_PE_bias_broadcast: bool = True,
) -> tuple[nl.ndarray, nl.ndarray, Optional[nl.ndarray]]:
    """
    Load down projection weight, scale, and bias (optional) for one expert using static DMA.

    When executed with LNC=2, weights and scales are sharded on I dimension. Bias is sharded on H dimension,
    with NC0 loading the first half of H and NC1 loading the second half.

    Args:
        weight (nl.ndarray): [E_L, 128_I, I/512, H], Down projection weight tensor from HBM (4_I packed in x4 dtype).
        scale (nl.ndarray): [E_L, 16_I, I/512, H], Down projection MX scale tensor from HBM (uint8 MX scales).
        bias (Optional[nl.ndarray]): [E_L, H], Optional down projection bias tensor from HBM.
        expert_idx (int): Index of the current expert to load.
        H (int): Hidden dimension size.
        tile_I (int): Tile size for I dimension (typically 128).
        n_I512_tiles (int): Number of I/512 tiles to load (local shard size when LNC=2).
        tile_T (int): Tile size for T dimension (for bias broadcast).
        activation_compute_dtype: Data type for bias buffer (default: nl.bfloat16).
        use_PE_bias_broadcast (bool): If True, use PE (matmul with ones) for bias broadcast; else use DVE
            stream_shuffle_broadcast.

    Returns:
        weight_sb (nl.ndarray): [128_I, n_I512_tiles, H], Weight in SBUF (4_I packed in x4 dtype).
        scale_sb (nl.ndarray): [128_I, n_I512_tiles, H], Scales in SBUF (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.ndarray]): [tile_T, H], Broadcasted bias in SBUF (zeros when bias=None, sharded on H
            when LNC=2).

    Notes:
        - Based on experiments, static DMA demonstrates better performance
        - Can revert to DGE if HBM out-of-memory (OOM) issues occur
    """

    # Calculate shapes / tiling
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx_shard_I", (0, 1))
    weight_sb_shape = (tile_I, n_I512_tiles, H)
    bias_sb_shape = (tile_T, H)

    # Allocate buffers
    weight_sb = nl.ndarray(weight_sb_shape, dtype=weight.dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray(weight_sb_shape, dtype=scale.dtype, buffer=nl.sbuf)
    bias_sb: Optional[nl.ndarray] = None

    # Load weight: index expert, then slice I/512 tiles
    # Shape: [E_L, 128_I, I/512, H] -> [128_I, n_I512_tiles, H]
    nisa.dma_copy(
        src=weight[expert_idx, :, nl.ds(n_I512_tiles * prg_id, n_I512_tiles), :],
        dst=weight_sb[...],
    )

    # Load scale: index expert, then slice I/512 tiles
    # Shape: [E_L, 16_I, I/512, H] -> [16_I, n_I512_tiles, H]
    # Note: scales have 16_I (not 128_I), need to map to first 4 partitions of each quadrant
    # Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer
    for quadrant_idx in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
        nisa.dma_copy(
            src=scale[
                expert_idx,
                nl.ds(SCALE_P_ELEM_PER_QUADRANT * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT),
                nl.ds(n_I512_tiles * prg_id, n_I512_tiles),
                :,
            ],
            dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT), :, :],
        )

    # Load + broadcast bias, sharding on H dim when LNC=2
    # In LNC=2, NC0 bias_sb will have first half of H filled with bias, second half with zeros; NC1 will have the inverse
    # Shape: [E_L, H] -> [1, H_size_local]
    if bias != None:
        bias_sb = nl.ndarray(bias_sb_shape, dtype=activation_compute_dtype, buffer=nl.sbuf)
        nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        H_size_local = H // 2 if n_prgs > 1 else H
        H_offset = H_size_local * prg_id
        H_slice_local = nl.ds(H_offset, H_size_local)
        nisa.dma_copy(
            src=bias[expert_idx, H_slice_local],
            dst=bias_sb[0:1, H_slice_local],
        )

        # Broadcast bias using PE
        if use_PE_bias_broadcast:
            H_tile_size_local = min(H_size_local, nl.tile_size.gemm_moving_fmax)
            ones_mask = nl.ndarray((1, tile_T), dtype=bias_sb.dtype, buffer=nl.sbuf)
            nisa.memset(dst=ones_mask[...], value=1.0, engine=nisa.gpsimd_engine)
            n_H512_tiles = div_ceil(H_size_local, nl.tile_size.gemm_moving_fmax)
            for h512_tile_idx in nl.affine_range(n_H512_tiles):
                bias_bc_psum = nl.ndarray((tile_T, H_tile_size_local), dtype=nl.float32, buffer=nl.psum)
                H_tile_slice = nl.ds(H_offset + h512_tile_idx * H_tile_size_local, H_tile_size_local)
                nisa.nc_matmul(
                    dst=bias_bc_psum,
                    stationary=ones_mask[...],
                    moving=bias_sb[0:1, H_tile_slice],
                    is_stationary_onezero=True,
                )
                nisa.tensor_copy(
                    dst=bias_sb[:, H_tile_slice],
                    src=bias_bc_psum[...],
                    engine=nisa.scalar_engine,
                )

        # Broadcast on DVE
        else:
            stream_shuffle_broadcast(src=bias_sb, dst=bias_sb)

    return weight_sb, scale_sb, bias_sb


def down_projection_mx_shard_I(
    act_sb: nl.ndarray,
    act_scale_sb: nl.ndarray,
    weight_sb: nl.ndarray,
    weight_scale_sb: nl.ndarray,
    bias_sb: Optional[nl.ndarray],
    expert_affinities_masked_sb: nl.ndarray,
    expert_idx: int,
    out_sb: nl.ndarray,
    out_hbm: Optional[nl.ndarray] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    activation_compute_dtype=nl.bfloat16,
    is_first_expert: bool = False,
    is_last_expert: bool = False,
) -> nl.ndarray:
    """
    Computes down projection, expert affinity scaling, expert add, LNC reduction, and SB->HBM spill. When executed with LNC=2, inputs are
    expected to be sharded on I dimension, and compute will be sharded on I dimension for matrix multiplication and H dimenision for bias add.

    Algorithm Affinity:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring mx I-sharded down projection

    Input shapes:
        act_sb: [16_I * 8_I, I/512, T], 4_I is packed in x4 dtype
        act_scale_sb: [16_I * 8_I, I/512, T], scales located in leading 4P of each SBUF quadrant
        weight_sb: [16_I * 8_I, I/512, H], 4_I is packed in x4 dtype
        weight_scale_sb: [16_I * 8_I, I/512, H], scales located in leading 4P of each SBUF quadrant
        bias_sb: [1, H]
        expert_affinities_masked_sb: [T, E_L] or [128_T, T/128, E_L]
        out_sb: [min(T, 128), ⌈T/128⌉, H]
        out_hbm: [T, H]

    Output shape:
        out_sb: [min(T, 128), ⌈T/128⌉, H]
    """

    # Extract / validate shapes
    TILE_I, n_I512_tiles, T = act_sb.shape
    TILE_I_, n_I512_tiles_, H = weight_sb.shape
    kernel_assert(
        TILE_I == TILE_I_, f"Expected same number of partitions in activation and weight, got {TILE_I}, {TILE_I_}"
    )
    kernel_assert(
        n_I512_tiles == n_I512_tiles_,
        f"Expected same number of I tiles in activation and weight, got {n_I512_tiles}, {n_I512_tiles_}",
    )
    kernel_assert(H % 512 == 0, f"Expected H divisible by 512, got {H=}")
    kernel_assert(
        expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE,
        f"Expected expert_affinities_scaling_mode={ExpertAffinityScaleMode.POST_SCALE}, got: {expert_affinities_scaling_mode=}",
    )

    # LNC config
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx_shard_I", (0, 1))

    # Tiling strategy
    pmax = nl.tile_size.pmax
    TILE_T = min(T, pmax)  # T will be partition dim in output
    TILE_H = min(H, nl.tile_size.psum_fmax * 2)  # use 2 * fmax with bf16 PSUM
    num_T128_tiles = div_ceil(T, pmax)
    num_H1024_tiles = H // TILE_H

    # Cast expert affinities to fp32 for tensor_scalar on scalar engine
    is_3D_affinities = len(expert_affinities_masked_sb.shape) == 3
    expert_affinities_masked_fp32_sb = nl.ndarray((TILE_T, num_T128_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=expert_affinities_masked_fp32_sb[...],
        src=expert_affinities_masked_sb[:, :, expert_idx]
        if is_3D_affinities
        else expert_affinities_masked_sb[:, expert_idx],
        engine=nisa.scalar_engine,
    )

    # Tiled MM: compute activation_mxfp8 (stationary) @ W_mxfp4/8 (moving)
    for tile_t in nl.sequential_range(num_T128_tiles):
        # T dim slicing + masking for case when T tile < TILE_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)
        for tile_h in nl.sequential_range(num_H1024_tiles):
            # H dim slicing
            tile_H_offset = TILE_H * tile_h
            weight_H_slice = nl.ds(tile_H_offset, TILE_H)
            out_psum = nl.ndarray((TILE_T, TILE_H), dtype=nl.bfloat16, buffer=nl.psum)
            expert_out_tile_sb = nl.ndarray((TILE_T, TILE_H), dtype=activation_compute_dtype, buffer=nl.sbuf)
            for tile_i in nl.sequential_range(n_I512_tiles):
                nisa.nc_matmul_mx(
                    dst=out_psum[:tile_T_actual, :],
                    stationary=act_sb[:, tile_i, tile_T_slice],
                    moving=weight_sb[:, tile_i, weight_H_slice],
                    stationary_scale=act_scale_sb[:, tile_i, tile_T_slice],
                    moving_scale=weight_scale_sb[:, tile_i, weight_H_slice],
                )

            # Accumulate bias during PSUM eviction
            if bias_sb != None:
                nisa.tensor_tensor(
                    dst=expert_out_tile_sb[:tile_T_actual, :],
                    data1=out_psum[:tile_T_actual, :],
                    op=nl.add,
                    data2=bias_sb[:tile_T_actual, tile_H_offset : tile_H_offset + TILE_H],
                )
            else:
                nisa.tensor_copy(
                    dst=expert_out_tile_sb[:tile_T_actual, :],
                    src=out_psum[:tile_T_actual, :],
                )

            # Expert affinity scaling, expert add
            # Expert 0 can update out_sb directly without addition
            if is_first_expert:
                nisa.tensor_scalar(
                    dst=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                    data=expert_out_tile_sb.ap([[TILE_H, tile_T_actual], [1, TILE_H]]),
                    op0=nl.multiply,
                    operand0=expert_affinities_masked_fp32_sb[:tile_T_actual, tile_t : tile_t + 1],
                    engine=nisa.scalar_engine,
                )
            # Expert [1, ..., E] must compute out_sb += expert_out after affinity scaling
            else:
                nisa.tensor_scalar(
                    dst=expert_out_tile_sb[:tile_T_actual, :],
                    data=expert_out_tile_sb.ap([[TILE_H, tile_T_actual], [1, TILE_H]]),
                    op0=nl.multiply,
                    operand0=expert_affinities_masked_fp32_sb[:tile_T_actual, tile_t : tile_t + 1],
                    engine=nisa.scalar_engine,
                )
                # Expert add
                nisa.tensor_tensor(
                    dst=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                    data1=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                    op=nl.add,
                    data2=expert_out_tile_sb[:tile_T_actual, :],
                )

        # LNC reduce and SB->HBM spill when computing final expert
        # TODO: add support for output_in_sbuf=True
        if is_last_expert and out_hbm != None:
            # Sharded LNC reduction + SB->HBM spill when when LNC=2
            if n_prgs > 1:
                H_local = H // n_prgs
                H_offset_local = H_local * prg_id
                send_to_rank = recv_from_rank = 1 - prg_id
                H_local_slice = nl.ds(H_offset_local, H_local)
                H_send_slice = nl.ds(H_local * (1 - prg_id), H_local)
                out_sb_reduced = nl.ndarray((TILE_T, H_local), out_sb.dtype, buffer=nl.sbuf)
                PIPE_ID_OUTPUT = 0

                # NC0 recieves 1st half of output from NC1 and reduces locally on DVE; NC1 does the inverse
                # Create temporary buffers to hold the data for sendrecv
                out_sb_send = nl.ndarray((tile_T_actual, H_local), out_sb.dtype, buffer=nl.sbuf)
                out_sb_local = nl.ndarray((tile_T_actual, H_local), out_sb.dtype, buffer=nl.sbuf)

                # FIXME: remove extra TensorCopy instructions when NKIFE-472 is resolved
                # Copy data to temporary buffers
                nisa.tensor_copy(
                    dst=out_sb_send,
                    src=out_sb[:tile_T_actual, tile_t, H_send_slice],
                )
                nisa.tensor_copy(
                    dst=out_sb_local,
                    src=out_sb[:tile_T_actual, tile_t, H_local_slice],
                )

                # Perform sendrecv
                nisa.sendrecv(
                    send_to_rank=send_to_rank,
                    recv_from_rank=recv_from_rank,
                    src=out_sb_send,
                    dst=out_sb_reduced[:tile_T_actual, :],
                    pipe_id=PIPE_ID_OUTPUT,
                )

                # Reduce
                nisa.tensor_tensor(
                    dst=out_sb_reduced[:tile_T_actual, :],
                    data1=out_sb_local,
                    op=nl.add,
                    data2=out_sb_reduced[:tile_T_actual, :],
                )

                # Sharded SB->HBM spill
                nisa.dma_copy(
                    src=out_sb_reduced.ap([[H_local, tile_T_actual], [1, H_local]]),
                    dst=out_hbm[nl.ds(TILE_T * tile_t, tile_T_actual), nl.ds(H_offset_local, H_local)],
                )

            # LNC1
            else:
                # NC spills entire output to HBM
                nisa.dma_copy(
                    src=out_sb.ap([[num_T128_tiles * H, tile_T_actual], [H, 1]], offset=tile_t * H),
                    dst=out_hbm[nl.ds(TILE_T * tile_t, tile_T_actual), :],
                )

    return out_sb
