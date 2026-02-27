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
Gate/Up projection sub-kernels with LNC sharding on I (intermediate) dimension.

LNC Sharding Strategy: I dimension
- When LNC=2, weights, scales, and bias are sharded on I dimension
- Used by: all-expert MoE algorithm (but algorithm-independent)

These sub-kernels can be used by any algorithm that requires I-sharded gate/up projection,
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
    _q_width,
)

# Common utils
from ...utils.common_types import ActFnType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type

# FIXME: add @nki.jit decorator to all sub-kernels when NKIFE-557 is resolved


def load_gate_up_weight_scale_bias(
    weight: nl.ndarray,
    scale: nl.ndarray,
    bias: Optional[nl.ndarray],
    expert_idx: int,
    gate_or_up_idx: int,
    H: int,
    I_local: int,
    n_I512_tiles: int,
    prg_id: int,
) -> tuple[nl.ndarray, nl.ndarray, Optional[nl.ndarray]]:
    """
    Load gate or up projection weight, scale, and bias (optional) for one expert using static DMA.

    When executed with LNC=2, weights and scales are sharded on I dimension, and bias is sharded
    on I/512 tiles dimension.

    Args:
        weight (nl.ndarray): [E_L, 128_H, 2, H/512, I], Gate or up projection weight tensor from HBM
            (fused gate/up weights), 4_H packed in x4 dtype.
        scale (nl.ndarray): [E_L, 16_H, 2, H/512, I], Gate or up projection MX scale tensor from HBM
            (fused gate/up scales), uint8 MX scales.
        bias (Optional[nl.ndarray]): [E_L, 128_I, 2, I/512, 4_I], Optional gate or up projection bias
            tensor from HBM (fused gate/up biases).
        expert_idx (int): Index of the current expert to load.
        gate_or_up_idx (int): Index to select gate (0) or up (1) projection from fused tensor.
        H (int): Hidden dimension size.
        I_local (int): Local intermediate dimension size (full I when LNC=1, I/2 when LNC=2).
        n_I512_tiles (int): Number of I/512 tiles to load for bias (local shard size when LNC=2).
        prg_id (int): Program ID for LNC sharding (0 or 1).

    Returns:
        weight_sb (nl.ndarray): [128_H, H/512, I_local], Weight in SBUF (4_H packed in x4 dtype).
        scale_sb (nl.ndarray): [128_H, H/512, I_local], Scales in SBUF (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.ndarray]): [128_I, n_I512_tiles, 4_I], Bias in SBUF (None when bias not provided).

    Notes:
        - Based on experiments, static DMA demonstrates better performance
        - Can revert to DGE if HBM out-of-memory (OOM) issues occur
    """

    # Calculate shapes / tiling
    pmax = nl.tile_size.pmax
    TILE_H, n_H512_tiles = pmax, H // 512
    TILE_I, I_4 = pmax, 4
    weight_sb_shape = (TILE_H, n_H512_tiles, I_local)
    bias_sb_shape = (TILE_I, n_I512_tiles, I_4)
    is_bias = bias != None

    # Allocate buffers
    weight_sb = nl.ndarray(weight_sb_shape, dtype=weight.dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray(weight_sb_shape, dtype=scale.dtype, buffer=nl.sbuf)
    bias_sb = nl.ndarray(bias_sb_shape, dtype=bias.dtype, buffer=nl.sbuf) if is_bias else None

    # Load weight: index expert and gate/up, then slice I dimension
    # Shape: [E_L, 128_H, 2, H/512, I] -> [128_H, H/512, I_local]
    nisa.dma_copy(
        src=weight[expert_idx, :, gate_or_up_idx, :, nl.ds(I_local * prg_id, I_local)],
        dst=weight_sb[...],
    )

    # Load scale: index expert and gate/up, then slice I dimension
    # Shape: [E_L, 16_H, 2, H/512, I] -> [16_H, H/512, I_local]
    # Note: scales have 16_H (not 128_H), need to map to first 4 partitions of each quadrant
    # Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer
    for quadrant_idx in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
        nisa.dma_copy(
            src=scale[
                expert_idx,
                nl.ds(SCALE_P_ELEM_PER_QUADRANT * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT),
                gate_or_up_idx,
                :,
                nl.ds(I_local * prg_id, I_local),
            ],
            dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT), :, :],
        )

    # Load bias: index expert and gate/up, then slice I/512 tiles
    # Shape: [E_L, 128_I, 2, I/512, 4_I] -> [128_I, n_I512_tiles, 4_I]
    if is_bias:
        nisa.dma_copy(
            src=bias[expert_idx, :, gate_or_up_idx, nl.ds(n_I512_tiles * prg_id, n_I512_tiles), :],
            dst=bias_sb[...],
        )

    return weight_sb, scale_sb, bias_sb


def gate_up_projection_mx_shard_I(
    input_quant_sb: nl.ndarray,
    input_scale_sb: nl.ndarray,
    gate_weight_sb: nl.ndarray,
    up_weight_sb: nl.ndarray,
    gate_weight_scale_sb: nl.ndarray,
    up_weight_scale_sb: nl.ndarray,
    gate_bias_sb: Optional[nl.ndarray],
    up_bias_sb: Optional[nl.ndarray],
    lhs_rhs_swap: bool = True,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    hidden_act_fn: ActFnType = ActFnType.Swish,
    activation_compute_dtype=nl.bfloat16,
) -> tuple[nl.ndarray, nl.ndarray]:
    """
    Compute gate and up projections with clamping, activation function, and MX quantization.

    When executed with LNC=2, inputs are expected to be sharded on I dimension and compute
    is sharded on I dimension.

    Args:
        input_quant_sb (nl.ndarray): [16_H * 8_H, H/512, T], Quantized input in SBUF (4_H packed in x4 dtype).
        input_scale_sb (nl.ndarray): [16_H * 8_H, H/512, T], Input scales in SBUF (in leading 4P of each quadrant).
        gate_weight_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weights in SBUF
            (4_H packed in x4 dtype).
        up_weight_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weights in SBUF
            (4_H packed in x4 dtype).
        gate_weight_scale_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weight scales
            in SBUF (in leading 4P of each quadrant).
        up_weight_scale_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weight scales
            in SBUF (in leading 4P of each quadrant).
        gate_bias_sb (Optional[nl.ndarray]): [16_I * 8_I, I/512, 4_I], Gate bias in SBUF.
        up_bias_sb (Optional[nl.ndarray]): [16_I * 8_I, I/512, 4_I], Up bias in SBUF.
        lhs_rhs_swap (bool): Whether to swap LHS and RHS in matmul (default: True).
        gate_clamp_upper_limit (Optional[float]): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (Optional[float]): Lower clamp limit for gate projection.
        up_clamp_upper_limit (Optional[float]): Upper clamp limit for up projection.
        up_clamp_lower_limit (Optional[float]): Lower clamp limit for up projection.
        hidden_act_fn (ActFnType): Activation function type (default: Swish).
        activation_compute_dtype: Compute dtype for activations (default: bfloat16).

    Returns:
        out_quant_sb (nl.ndarray): [16_I * 8_I, I/512, T], Quantized output in SBUF (4_I packed in x4 dtype).
        out_scale_sb (nl.ndarray): [16_I * 8_I, I/512, T], Output scales in SBUF (in leading 4P of each quadrant).
    """

    # Step 1: Input validation
    TILE_H, n_H512_tiles, T = input_quant_sb.shape
    TILE_H_, n_H512_tiles_, I_local = gate_weight_sb.shape
    kernel_assert(
        gate_weight_sb.shape == up_weight_sb.shape,
        f"expected gate and up weights to have the same shapes, got {gate_weight_sb.shape=}, {up_weight_sb.shape=}",
    )
    kernel_assert(
        gate_weight_scale_sb.shape == up_weight_scale_sb.shape,
        f"expected gate and up scales to have the same shapes, got {gate_weight_scale_sb.shape=}, {up_weight_scale_sb.shape=}",
    )
    # Validate bias consistency: both must be None or both must have matching shapes
    if gate_bias_sb != None and up_bias_sb != None:
        kernel_assert(
            gate_bias_sb.shape == up_bias_sb.shape,
            f"expected gate and up biases to have the same shapes, got {gate_bias_sb.shape=}, {up_bias_sb.shape=}",
        )
    elif gate_bias_sb != None or up_bias_sb != None:
        kernel_assert(
            False,
            f"expected gate and up biases to be both None or both not None",
        )
    kernel_assert(TILE_H == TILE_H_, f"Expected same number of partitions in input and weight, got {TILE_H}, {TILE_H_}")
    kernel_assert(
        n_H512_tiles == n_H512_tiles_,
        f"Expected same number of H tiles in input and weight, got {n_H512_tiles}, {n_H512_tiles_}",
    )
    # TODO: remove assertion when support is added for incomplete tiles in I
    kernel_assert(I_local % 512 == 0, f"Expected I_local divisible by 1024, got {I_local=}.")
    kernel_assert(
        lhs_rhs_swap == True, f"Gate and up projection currently only support lhs_rhs_swap=True, got {lhs_rhs_swap=}"
    )

    # Tiling strategies for T, I
    TILE_T = min(256, T)  # We can use I_4 * TILE_T <= 1024 F dims with bf16 PSUM
    n_T256_tiles = div_ceil(T, TILE_T)
    n_I512_tiles, I_4, TILE_I = I_local // 512, 4, nl.tile_size.pmax

    # Step 2: Allocate output buffers
    out_shape = (TILE_I, n_I512_tiles, T, I_4)
    out_quant_shape = (TILE_I, n_I512_tiles, T)
    out_sb = nl.ndarray(out_shape, dtype=activation_compute_dtype, buffer=nl.sbuf)
    out_quant_sb = nl.ndarray(out_quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    out_scale_sb = nl.ndarray(out_quant_shape, dtype=nl.uint8, buffer=nl.sbuf)

    # Step 3: Fused gate projection, projection clamping (optional), activation function
    # Step 3.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving)
    # TODO: (1) consider changing loop order to T, H, I, 4_I
    #       (2) move duplicate matmul and bias add logic into sub-functions that are called by gate and up
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)
        for tile_i in nl.sequential_range(n_I512_tiles):
            out_psum = nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum)
            for q_width_I_idx in nl.sequential_range(_q_width):
                # I dim slicing
                weight_I_offset = tile_i * 512 + q_width_I_idx * TILE_I
                weight_I_slice = nl.ds(weight_I_offset, TILE_I)
                for tile_h in nl.sequential_range(n_H512_tiles):
                    nisa.nc_matmul_mx(
                        # dst=out_psum[:, q_width_I_idx, :],
                        dst=out_psum[:, q_width_I_idx, :tile_T_actual],
                        stationary=gate_weight_sb[:, tile_h, weight_I_slice],
                        moving=input_quant_sb[:, tile_h, tile_T_slice],
                        stationary_scale=gate_weight_scale_sb[:, tile_h, weight_I_slice],
                        moving_scale=input_scale_sb[:, tile_h, tile_T_slice],
                    )

            # Step 3.2: Accumulate bias during PSUM eviction
            # out_sb shape: [TILE_I, n_I512_tiles, T, I_4] = [128, 6, 32, 4]
            # out_psum shape: [TILE_I, I_4, TILE_T] = [128, 4, 256]
            # gate_bias_sb shape: [TILE_I, n_I512_tiles, I_4] = [128, 6, 4]
            # Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4]
            if gate_bias_sb != None:
                nisa.tensor_tensor(
                    dst=out_sb[:, tile_i, tile_T_slice, :],
                    data1=out_psum.ap([[I_4 * TILE_T, TILE_I], [1, tile_T_actual], [TILE_T, I_4]]),
                    op=nl.add,
                    data2=gate_bias_sb.ap(
                        [[n_I512_tiles * I_4, TILE_I], [0, tile_T_actual], [1, I_4]], offset=tile_i * I_4
                    ),
                )
            else:
                nisa.tensor_copy(
                    dst=out_sb[:, tile_i, tile_T_slice, :],
                    src=out_psum.ap([[I_4 * TILE_T, TILE_I], [1, tile_T_actual], [TILE_T, I_4]]),
                )

            # Step 3.3: Clamp projection output to [clamp_lower_limit, clamp_upper_limit] (optional)
            if gate_clamp_upper_limit != None or gate_clamp_lower_limit != None:
                nisa.tensor_scalar(
                    dst=out_sb[:, tile_i, tile_T_slice, :],
                    data=out_sb[:, tile_i, tile_T_slice, :],
                    op0=nl.minimum if gate_clamp_upper_limit != None else None,
                    operand0=gate_clamp_upper_limit,
                    op1=nl.maximum if gate_clamp_lower_limit != None else None,
                    operand1=gate_clamp_lower_limit,
                )

            # Step 3.4: Compute activation function
            if hidden_act_fn != None:
                nisa.activation(
                    dst=out_sb[:, tile_i, tile_T_slice, :],
                    data=out_sb[:, tile_i, tile_T_slice, :],
                    op=get_nl_act_fn_from_type(hidden_act_fn),
                )

    # Step 4: Fused up projection, projection clamp (optional), gate * up, MX quantization
    # Step 4.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving)
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)
        for tile_i in nl.sequential_range(n_I512_tiles):
            intermediate_tile_sb = nl.ndarray((TILE_I, 1, TILE_T, I_4), dtype=out_sb.dtype, buffer=nl.sbuf)
            out_psum = nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum)
            for q_width_I_idx in nl.sequential_range(_q_width):
                # I dim slicing
                weight_I_offset = tile_i * 512 + q_width_I_idx * TILE_I
                weight_I_slice = nl.ds(weight_I_offset, TILE_I)
                for tile_h in nl.sequential_range(n_H512_tiles):
                    nisa.nc_matmul_mx(
                        # dst=out_psum[:, q_width_I_idx, tile_T_slice],
                        # dst=out_psum[:, q_width_I_idx, :],
                        dst=out_psum[:, q_width_I_idx, :tile_T_actual],
                        stationary=up_weight_sb[:, tile_h, weight_I_slice],
                        moving=input_quant_sb[:, tile_h, tile_T_slice],
                        stationary_scale=up_weight_scale_sb[:, tile_h, weight_I_slice],
                        moving_scale=input_scale_sb[:, tile_h, tile_T_slice],
                    )

            # Step 4.2: Accumulate bias during PSUM eviction
            # intermediate_tile_sb shape: [TILE_I, 1, TILE_T, I_4] = [128, 1, 256, 4]
            # out_psum shape: [TILE_I, I_4, TILE_T] = [128, 4, 256]
            # up_bias_sb shape: [TILE_I, n_I512_tiles, I_4] = [128, 6, 4]
            # Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4]
            if up_bias_sb != None:
                nisa.tensor_tensor(
                    dst=intermediate_tile_sb[:, 0, :tile_T_actual, :],
                    data1=out_psum.ap([[I_4 * TILE_T, TILE_I], [1, tile_T_actual], [TILE_T, I_4]]),
                    op=nl.add,
                    data2=up_bias_sb.ap(
                        [[n_I512_tiles * I_4, TILE_I], [0, tile_T_actual], [1, I_4]], offset=tile_i * I_4
                    ),
                )
            else:
                nisa.tensor_copy(
                    dst=intermediate_tile_sb[:, 0, :tile_T_actual, :],
                    src=out_psum.ap([[I_4 * TILE_T, TILE_I], [1, tile_T_actual], [TILE_T, I_4]]),
                )

            # Step 4.3: Clamp projection output to [clamp_lower_limit, clamp_upper_limit]
            if up_clamp_upper_limit != None or up_clamp_lower_limit != None:
                nisa.tensor_scalar(
                    dst=intermediate_tile_sb[:, 0, :tile_T_actual, :],
                    data=intermediate_tile_sb[:, 0, :tile_T_actual, :],
                    op0=nl.minimum if up_clamp_upper_limit != None else None,
                    operand0=up_clamp_upper_limit,
                    op1=nl.maximum if up_clamp_lower_limit != None else None,
                    operand1=up_clamp_lower_limit,
                )

            # Step 4.4: Multiply completed up tile with corresponding gate tile
            nisa.tensor_tensor(
                dst=out_sb[:, tile_i, tile_T_slice, :],
                data1=out_sb[:, tile_i, tile_T_slice, :],
                op=nl.multiply,
                data2=intermediate_tile_sb[:, 0, :tile_T_actual, :],
            )

            # Step 4.5: MX quantize combined gate * up tile
            nisa.quantize_mx(
                src=out_sb[:, tile_i, tile_T_slice, :],
                dst=out_quant_sb[:, tile_i, tile_T_slice],
                dst_scale=out_scale_sb[:, tile_i, tile_T_slice],
            )

    return out_quant_sb, out_scale_sb
