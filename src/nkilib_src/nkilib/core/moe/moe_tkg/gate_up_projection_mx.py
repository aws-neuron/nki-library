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
Gate/Up projection sub-kernels with LNC sharding support.

Supports multiple LNC sharding strategies (see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py):
- NO_SHARD: No sharding, each NC computes full result independently. Used when LNC=1.
- SHARD_I: Shard on I (intermediate) dimension. Default for most workloads.
- TODO: SHARD_T: Shard on T (token) dimension. Useful when T is large.

These sub-kernels can be used by any algorithm that requires LNC-sharded gate/up projection,
including all-expert, selective-load, or custom MoE implementations.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

# Common utils
from ...utils.common_types import ActFnType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
from ...utils.tensor_view import as_nki_tensor

# Shared MX constants
from .projection_mx_constants import (
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    MIN_MATMULT_MX_P_DIM,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
    _psum_fmax,
    _q_height,
    _q_width,
    pad_to_valid_qmx_partitions,
)


def gate_up_projection_mx(
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    gate_weight_sb: nl.NkiTensor,
    up_weight_sb: nl.NkiTensor,
    gate_weight_scale_sb: nl.NkiTensor,
    up_weight_scale_sb: nl.NkiTensor,
    gate_bias_sb: Optional[nl.NkiTensor],
    up_bias_sb: Optional[nl.NkiTensor],
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    hidden_act_fn: ActFnType = ActFnType.Swish,
    activation_compute_dtype=nl.bfloat16,
    gate_dequant_scale: Optional[nl.NkiTensor] = None,
    up_dequant_scale: Optional[nl.NkiTensor] = None,
    input_dequant_scale: Optional[nl.NkiTensor] = None,
    input_quant_hbm: Optional[nl.NkiTensor] = None,
    input_scale_hbm: Optional[nl.NkiTensor] = None,
    is_software_quant: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Compute gate and up projections with clamping, activation function, and MX quantization.

    When executed with LNC=2, inputs are expected to be sharded on I dimension and compute
    is sharded on I dimension.

    Usage:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring mx I-sharded gate/up projection

    Args:
        input_quant_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Quantized input in SBUF (4_H packed in x4 dtype).
        input_scale_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Input scales in SBUF (in leading 4P of each quadrant).
        gate_weight_sb (nl.NkiTensor): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weights in SBUF
            (4_H packed in x4 dtype).
        up_weight_sb (nl.NkiTensor): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weights in SBUF
            (4_H packed in x4 dtype).
        gate_weight_scale_sb (nl.NkiTensor): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weight scales
            in SBUF (in leading 4P of each quadrant).
        up_weight_scale_sb (nl.NkiTensor): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weight scales
            in SBUF (in leading 4P of each quadrant).
        gate_bias_sb (Optional[nl.NkiTensor]): [16_I * 8_I, I/512, 4_I], Gate bias in SBUF.
        up_bias_sb (Optional[nl.NkiTensor]): [16_I * 8_I, I/512, 4_I], Up bias in SBUF.
        gate_clamp_upper_limit (Optional[float]): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (Optional[float]): Lower clamp limit for gate projection.
        up_clamp_upper_limit (Optional[float]): Upper clamp limit for up projection.
        up_clamp_lower_limit (Optional[float]): Lower clamp limit for up projection.
        hidden_act_fn (ActFnType): Activation function type (default: Swish).
        activation_compute_dtype: Compute dtype for activations (default: bfloat16).
        is_software_quant (bool): When True, weight scales are 2D [128, I] shared dummy tiles indexed
            as [:, :slice] instead of the normal 3D [:, tile_h, slice].

    Returns:
        out_quant_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Quantized output in SBUF (4_I packed in x4 dtype).
        out_scale_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Output scales in SBUF (in leading 4P of each quadrant).
    """

    # Step 1: Input validation
    TILE_H, n_H512_tiles, T = input_quant_sb.shape
    TILE_H_, n_H512_tiles_, I_local_padded = gate_weight_sb.shape
    I_local = I_local_padded
    kernel_assert(
        gate_weight_sb.shape == up_weight_sb.shape,
        f"expected gate and up weights to have the same shapes, got {gate_weight_sb.shape=}, {up_weight_sb.shape=}",
    )
    kernel_assert(
        gate_weight_scale_sb.shape == up_weight_scale_sb.shape,
        f"expected gate and up scales to have the same shapes, "
        f"got {gate_weight_scale_sb.shape=}, {up_weight_scale_sb.shape=}",
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

    # Tiling strategies for T, I
    TILE_T = min(_psum_fmax * 2 // _q_width, T)  # I_4 * TILE_T <= psum_fmax * 2 for bf16 PSUM
    n_T256_tiles = div_ceil(T, TILE_T)
    n_total_I512_tiles = div_ceil(I_local, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
    I_4, TILE_I = _q_width, nl.tile_size.pmax

    # Step 2: Allocate output buffers
    out_shape = (TILE_I, n_total_I512_tiles, T, I_4)
    out_quant_shape = (TILE_I, n_total_I512_tiles, T)
    out_sb = nl.ndarray(out_shape, dtype=activation_compute_dtype, buffer=nl.sbuf)
    out_quant_sb = nl.ndarray(out_quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    out_scale_sb = nl.ndarray(out_quant_shape, dtype=nl.uint8, buffer=nl.sbuf)

    # only memset in I padding case
    last_tile_I_size = I_local - (n_total_I512_tiles - 1) * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    last_I_pdim_sz = last_tile_I_size // _q_width
    if last_I_pdim_sz < TILE_I:
        nisa.memset(dst=out_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=out_quant_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        nisa.memset(dst=out_scale_sb[...], value=0.0, engine=nisa.gpsimd_engine)

    """
    Step 3: Fused gate projection, projection clamping (optional), activation function.
    Step 3.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving).
    """
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)

        # Per-tile HBM→SBUF load when input is in HBM
        if input_quant_hbm != None:
            input_tile_quant = nl.ndarray(
                (TILE_H, n_H512_tiles, tile_T_actual), dtype=input_quant_hbm.dtype, buffer=nl.sbuf
            )
            input_tile_scale = nl.ndarray(
                (TILE_H, n_H512_tiles, tile_T_actual), dtype=input_scale_hbm.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=input_tile_quant, src=input_quant_hbm[:, :, tile_T_slice])
            nisa.dma_copy(dst=input_tile_scale, src=input_scale_hbm[:, :, tile_T_slice])
            cur_input_quant = input_tile_quant
            cur_input_scale = input_tile_scale
            cur_tile_T_slice = nl.ds(0, tile_T_actual)
        else:
            cur_input_quant = input_quant_sb
            cur_input_scale = input_scale_sb
            cur_tile_T_slice = tile_T_slice

        # Pre-allocate PSUM for all I tiles upfront
        out_psum_lst = []
        for tile_i in range(n_total_I512_tiles):
            out_psum_lst.append(nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum))

        _projection_matmul_mx(
            out_psum_lst=out_psum_lst,
            weight_sb=gate_weight_sb,
            weight_scale_sb=gate_weight_scale_sb,
            input_quant_sb=cur_input_quant,
            input_scale_sb=cur_input_scale,
            tile_T_slice=cur_tile_T_slice,
            tile_T_actual=tile_T_actual,
            n_H512_tiles=n_H512_tiles,
            n_total_I512_tiles=n_total_I512_tiles,
            I_local=I_local,
            is_software_quant=is_software_quant,
        )

        # Step 3.2: PSUM eviction + bias + clamp + activation (after all H tiles complete)
        for tile_i in range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_local - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I_pdim_sz = cur_tile_I_size // _q_width

            """
            Accumulate bias during PSUM eviction (skip for STATIC_MX).
            out_sb shape: [TILE_I, n_total_I512_tiles, T, I_4]
            out_psum shape: [TILE_I, I_4, TILE_T]
            gate_bias_sb shape: [TILE_I, n_total_I512_tiles, I_4]
            Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4].
            """
            is_software_dequant = gate_dequant_scale != None
            if gate_bias_sb != None and not is_software_dequant:
                nisa.tensor_tensor(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    data1=out_psum_lst[tile_i].ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                    op=nl.add,
                    data2=gate_bias_sb.ap(
                        [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]], offset=tile_i * I_4
                    ),
                )
            else:
                nisa.tensor_copy(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    src=out_psum_lst[tile_i].ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                )

            # STATIC_MX: dequant then bias
            if gate_dequant_scale != None and gate_dequant_scale.shape[1] == 1:
                nisa.activation(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    op=nl.copy,
                    data=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    scale=gate_dequant_scale[:cur_I_pdim_sz, :],
                )
                if gate_bias_sb != None:
                    nisa.tensor_tensor(
                        dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                        data1=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                        op=nl.add,
                        data2=gate_bias_sb.ap(
                            [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]],
                            offset=tile_i * I_4,
                        ),
                    )

            # ROW_MX: per-column weight dequant, then per-token input dequant, then bias
            if gate_dequant_scale != None and gate_dequant_scale.shape[1] > 1:
                for i_q in nl.affine_range(I_4):
                    i_col = tile_i * I_4 + i_q
                    nisa.activation(
                        dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, i_q],
                        op=nl.copy,
                        data=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, i_q],
                        scale=gate_dequant_scale[:cur_I_pdim_sz, i_col : i_col + 1],
                    )
                for i_q in nl.affine_range(I_4):
                    nisa.tensor_tensor(
                        dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, i_q],
                        data1=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, i_q],
                        data2=input_dequant_scale[:cur_I_pdim_sz, :tile_T_actual, 0],
                        op=nl.multiply,
                    )
                if gate_bias_sb != None:
                    nisa.tensor_tensor(
                        dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                        data1=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                        op=nl.add,
                        data2=gate_bias_sb.ap(
                            [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]],
                            offset=tile_i * I_4,
                        ),
                    )

            # Step 3.3: Clamp projection output to [clamp_lower_limit, clamp_upper_limit] (optional)
            _clamp_tensor(
                tensor=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                clamp_upper_limit=gate_clamp_upper_limit,
                clamp_lower_limit=gate_clamp_lower_limit,
            )

            # Step 3.4: Compute activation function
            if hidden_act_fn != None:
                nisa.activation(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    data=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    op=get_nl_act_fn_from_type(hidden_act_fn),
                )

    """
    Step 4: Fused up projection, projection clamp (optional), gate * up, MX quantization.
    Step 4.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving).
    """
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)

        # Per-tile HBM→SBUF load when input is in HBM
        if input_quant_hbm != None:
            input_tile_quant = nl.ndarray(
                (TILE_H, n_H512_tiles, tile_T_actual), dtype=input_quant_hbm.dtype, buffer=nl.sbuf
            )
            input_tile_scale = nl.ndarray(
                (TILE_H, n_H512_tiles, tile_T_actual), dtype=input_scale_hbm.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=input_tile_quant, src=input_quant_hbm[:, :, tile_T_slice])
            nisa.dma_copy(dst=input_tile_scale, src=input_scale_hbm[:, :, tile_T_slice])
            cur_input_quant = input_tile_quant
            cur_input_scale = input_tile_scale
            cur_tile_T_slice = nl.ds(0, tile_T_actual)
        else:
            cur_input_quant = input_quant_sb
            cur_input_scale = input_scale_sb
            cur_tile_T_slice = tile_T_slice

        # Pre-allocate PSUM for all I tiles upfront (enables T → H → I → 4_I loop order)
        up_psum_lst = []
        for tile_i in range(n_total_I512_tiles):
            up_psum_lst.append(nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum))

        # Matmul compute with loop order H → I → 4_I (same input H-slice reused across all I tiles)
        _projection_matmul_mx(
            out_psum_lst=up_psum_lst,
            weight_sb=up_weight_sb,
            weight_scale_sb=up_weight_scale_sb,
            input_quant_sb=cur_input_quant,
            input_scale_sb=cur_input_scale,
            tile_T_slice=cur_tile_T_slice,
            tile_T_actual=tile_T_actual,
            n_H512_tiles=n_H512_tiles,
            n_total_I512_tiles=n_total_I512_tiles,
            I_local=I_local,
            is_software_quant=is_software_quant,
        )

        # Step 4.2: PSUM eviction + bias + clamp + gate*up + quantize (after all H tiles complete)
        for tile_i in range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_local - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I_pdim_sz = cur_tile_I_size // _q_width
            intermediate_tile_sb = nl.ndarray((TILE_I, 1, TILE_T, I_4), dtype=out_sb.dtype, buffer=nl.sbuf)

            """
            Accumulate bias during PSUM eviction (skip for STATIC_MX).
            intermediate_tile_sb shape: [TILE_I, 1, TILE_T, I_4]
            out_psum shape: [TILE_I, I_4, TILE_T]
            up_bias_sb shape: [TILE_I, n_total_I512_tiles, I_4]
            Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4].
            """
            is_up_software_dequant = up_dequant_scale != None
            if up_bias_sb != None and not is_up_software_dequant:
                nisa.tensor_tensor(
                    dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                    data1=up_psum_lst[tile_i].ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                    op=nl.add,
                    data2=up_bias_sb.ap(
                        [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]], offset=tile_i * I_4
                    ),
                )
            else:
                nisa.tensor_copy(
                    dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                    src=up_psum_lst[tile_i].ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                )

            # STATIC_MX: dequant then bias
            if up_dequant_scale != None and up_dequant_scale.shape[1] == 1:
                nisa.activation(
                    dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                    op=nl.copy,
                    data=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                    scale=up_dequant_scale[:cur_I_pdim_sz, :],
                )
                if up_bias_sb != None:
                    nisa.tensor_tensor(
                        dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                        data1=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                        op=nl.add,
                        data2=up_bias_sb.ap(
                            [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]],
                            offset=tile_i * I_4,
                        ),
                    )

            # ROW_MX: per-column weight dequant, then per-token input dequant, then bias
            if up_dequant_scale != None and up_dequant_scale.shape[1] > 1:
                for i_q in nl.affine_range(I_4):
                    i_col = tile_i * I_4 + i_q
                    nisa.activation(
                        dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, i_q],
                        op=nl.copy,
                        data=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, i_q],
                        scale=up_dequant_scale[:cur_I_pdim_sz, i_col : i_col + 1],
                    )
                for i_q in nl.affine_range(I_4):
                    nisa.tensor_tensor(
                        dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, i_q],
                        data1=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, i_q],
                        data2=input_dequant_scale[:cur_I_pdim_sz, :tile_T_actual, 0],
                        op=nl.multiply,
                    )
                if up_bias_sb != None:
                    nisa.tensor_tensor(
                        dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                        data1=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                        op=nl.add,
                        data2=up_bias_sb.ap(
                            [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]],
                            offset=tile_i * I_4,
                        ),
                    )

            # Step 4.3: Clamp projection output to [clamp_lower_limit, clamp_upper_limit]
            _clamp_tensor(
                tensor=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                clamp_upper_limit=up_clamp_upper_limit,
                clamp_lower_limit=up_clamp_lower_limit,
            )

            # Step 4.4: Multiply completed up tile with corresponding gate tile
            nisa.tensor_tensor(
                dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                data1=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                op=nl.multiply,
                data2=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
            )

            # Step 4.5: MX quantize combined gate * up tile
            # Skip for ROW_MX (uses external row_quantization) and STATIC_MX (uses external static_quantization)
            if not is_software_quant:
                # Pad partition count to valid quantize_mx size {32, 64, 96, 128}.
                # Extra zero-padded partitions are harmless: downstream weight is zero-padded.
                qmx_I_pdim_sz = pad_to_valid_qmx_partitions(cur_I_pdim_sz)
                nisa.quantize_mx(
                    src=out_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice, :],
                    dst=out_quant_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice],
                    dst_scale=out_scale_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice],
                )

    if is_software_quant:
        # ROW_MX and STATIC_MX: return bf16 gate*up result for external quantization
        return out_sb, None
    return out_quant_sb, out_scale_sb


def load_gate_up_weight_scale_bias(
    weight: nl.NkiTensor,
    scale: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    expert_idx: int,
    gate_or_up_idx: int,
    H: int,
    n_I512_tiles_local: int,
    I_local: int,
    I_offset: int,
    I_local_padded: int = 0,
    skip_scale_load: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor, Optional[nl.NkiTensor]]:
    """
    Load gate or up projection weight, scale, and bias (optional) for one expert using static DMA.

    When executed with LNC=2, weights and scales are sharded on I/512 tiles dimension (tile-based sharding).
    This ensures alignment with down_projection_mx which also uses tile-based I-sharding.

    Args:
        weight (nl.NkiTensor): [E_L, 128_H, 2, H/512, I], Gate or up projection weight tensor from HBM
            (fused gate/up weights), 4_H packed in x4 dtype.
        scale (nl.NkiTensor): [E_L, 16_H, 2, H/512, I], Gate or up projection MX scale tensor from HBM
            (fused gate/up scales), uint8 MX scales.
        bias (Optional[nl.NkiTensor]): [E_L, 128_I, 2, I/512, 4_I], Optional gate or up projection bias
            tensor from HBM (fused gate/up biases).
        expert_idx (int): Index of the current expert to load.
        gate_or_up_idx (int): Index to select gate (0) or up (1) projection from fused tensor.
        H (int): Hidden dimension size.
        n_I512_tiles_local (int): Number of I/512 tiles for this NC (may differ between NCs for odd tile counts).
        I_local (int): Local intermediate dimension size for this NC.
        I_offset (int): Starting I offset for this NC's tiles.
        I_local_padded (int): Padded I_local (nearest multiple of 8). If 0, defaults to I_local (no padding).

    Returns:
        weight_sb (nl.NkiTensor): [128_H, H/512, I_local_padded], Weight in SBUF (4_H packed in x4 dtype).
        scale_sb (nl.NkiTensor): [128_H, H/512, I_local_padded], Scales in SBUF (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.NkiTensor]): [128_I, n_I512_tiles_local, 4_I], Bias in SBUF (None when bias not provided).

    Notes:
        - Uses tile-based I-sharding to align with down_projection_mx
        - NC0 gets first n_I512_tiles_local tiles, NC1 gets the rest
        - Based on experiments, static DMA demonstrates better performance
    """

    # Calculate shapes / tiling
    I_buf = I_local_padded if I_local_padded > 0 else I_local
    kernel_assert(
        I_buf % MIN_MATMULT_MX_P_DIM == 0,
        f"Expected I_local (padded) divisible by {MIN_MATMULT_MX_P_DIM} for nc_matmul_mx even free-dim constraint, got {I_buf=}.",
    )
    needs_padding = I_buf > I_local
    pmax = nl.tile_size.pmax
    TILE_H, n_H512_tiles = pmax, H // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    TILE_I, I_4 = pmax, _q_width
    weight_sb_shape = (TILE_H, n_H512_tiles, I_buf)
    bias_sb_shape = (TILE_I, n_I512_tiles_local, I_4)
    is_bias = bias != None

    # Allocate buffers
    # SW quant: skip_scale_load=True, scale_sb=None (caller uses shared 2D dummy tile instead)
    base_weight = weight
    weight_sb = nl.ndarray(weight_sb_shape, dtype=base_weight.dtype, buffer=nl.sbuf)
    scale_dtype = nl.uint8 if skip_scale_load else scale.dtype
    scale_sb = None if skip_scale_load else nl.ndarray(weight_sb_shape, dtype=scale_dtype, buffer=nl.sbuf)
    bias_sb = nl.ndarray(bias_sb_shape, dtype=bias.dtype, buffer=nl.sbuf) if is_bias else None

    # Load weight: index expert and gate/up, then slice I dimension using tile-based offset
    # Shape: [E_L, 128_H, 2, H/512, I] -> [128_H, H/512, I_local] -> padded to [128_H, H/512, I_buf]
    weight_view = (
        base_weight.select(dim=0, index=expert_idx)
        .select(dim=1, index=gate_or_up_idx)
        .slice(dim=2, start=I_offset, end=I_offset + I_local)
    )
    if needs_padding:
        nisa.memset(dst=weight_sb[...], value=0, engine=nisa.gpsimd_engine)
        nisa.dma_copy(src=as_nki_tensor(weight_view), dst=weight_sb[:, :, :I_local], dge_mode=nisa.dge_mode.none)
    else:
        nisa.dma_copy(src=as_nki_tensor(weight_view), dst=weight_sb[...], dge_mode=nisa.dge_mode.none)
    weight_sb = weight_sb.view(weight.dtype)

    """
    Load scale: index expert and gate/up, then slice I dimension using tile-based offset.
    Shape: [E_L, 16_H, 2, H/512, I] -> [16_H, H/512, I_local]
    Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer.
    Skipped when skip_scale_load=True (SW quant): caller passes a shared 2D [128, F] dummy tile directly.
    """
    if not skip_scale_load:
        n_scale_partitions = TILE_H // _q_height
        n_quadrants_needed = div_ceil(n_scale_partitions, SCALE_P_ELEM_PER_QUADRANT)

        if needs_padding:
            nisa.memset(dst=scale_sb[...], value=0.0, engine=nisa.gpsimd_engine)

        DMA_FREE_DIM_TILE = 1024
        n_I_tiles = div_ceil(I_local, DMA_FREE_DIM_TILE)

        for quadrant_idx in nl.affine_range(n_quadrants_needed):
            for i_tile_idx in nl.affine_range(n_I_tiles):
                i_start = i_tile_idx * DMA_FREE_DIM_TILE
                i_size = min(DMA_FREE_DIM_TILE, I_local - i_start)
                scale_view = (
                    scale.select(dim=0, index=expert_idx)
                    .slice(
                        dim=0,
                        start=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx,
                        end=SCALE_P_ELEM_PER_QUADRANT * (quadrant_idx + 1),
                    )
                    .select(dim=1, index=gate_or_up_idx)
                    .slice(dim=2, start=I_offset + i_start, end=I_offset + i_start + i_size)
                )
                nisa.dma_copy(
                    src=scale_view,
                    dst=scale_sb[
                        nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT),
                        :,
                        i_start : i_start + i_size,
                    ],
                    dge_mode=nisa.dge_mode.none,
                )

    tile_offset = I_offset // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM

    # Load bias: index expert and gate/up, then slice I/512 tiles based on tile ownership
    # Shape: [E_L, I_p, 2, I/512, 4_I] -> [I_p, n_I512_tiles_local, 4_I] -> padded to [128_I, n_I512_tiles_local, 4_I]
    if is_bias:
        I_p_bias_in_hbm = bias.shape[1]

        if I_p_bias_in_hbm < TILE_I:
            nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)

        bias_view = (
            bias.select(dim=0, index=expert_idx)
            .select(dim=1, index=gate_or_up_idx)
            .slice(dim=1, start=tile_offset, end=tile_offset + n_I512_tiles_local)
        )
        nisa.dma_copy(
            src=as_nki_tensor(bias_view),
            dst=bias_sb[:I_p_bias_in_hbm, :, :],
            dge_mode=nisa.dge_mode.none,
        )

    return weight_sb, scale_sb, bias_sb


def _clamp_tensor(
    tensor: nl.NkiTensor,
    clamp_upper_limit: Optional[float],
    clamp_lower_limit: Optional[float],
) -> None:
    """
    Apply optional clamping to a tensor in-place.

    Args:
        tensor (nl.NkiTensor): Tensor slice to clamp.
        clamp_upper_limit (Optional[float]): Upper clamp limit (None to skip).
        clamp_lower_limit (Optional[float]): Lower clamp limit (None to skip).

    Returns:
        None. Tensor is clamped in-place.
    """
    if clamp_upper_limit != None or clamp_lower_limit != None:
        nisa.tensor_scalar(
            dst=tensor,
            data=tensor,
            op0=nl.minimum if clamp_upper_limit != None else None,
            operand0=clamp_upper_limit,
            op1=nl.maximum if clamp_lower_limit != None else None,
            operand1=clamp_lower_limit,
        )


def _projection_matmul_mx(
    out_psum_lst: list,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    input_quant_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    tile_T_slice,
    tile_T_actual: int,
    n_H512_tiles: int,
    n_total_I512_tiles: int,
    I_local: int,
    is_software_quant: bool = False,
) -> None:
    """
    Perform MX matmul accumulation over H tiles and I tiles.
    Loop order: H → I → 4_I for better input tensor reuse with larger T.

    Args:
        out_psum_lst (list): List of PSUM buffers, one per I512 tile.
        weight_sb (nl.NkiTensor): [128_H, H/512, I], Weight tensor in SBUF.
        weight_scale_sb (nl.NkiTensor): [128_H, H/512, I] normally, or [128, I] when is_software_quant
            (shared 2D dummy tile).
        input_quant_sb (nl.NkiTensor): [128_H, H/512, T], Quantized input in SBUF.
        input_scale_sb (nl.NkiTensor): [128_H, H/512, T], Input scale in SBUF.
        tile_T_slice: T dimension slice descriptor.
        tile_T_actual (int): Actual T tile size (may be < 256 for last tile).
        n_H512_tiles (int): Number of H/512 tiles.
        n_total_I512_tiles (int): Number of I/512 tiles.
        I_local (int): Local intermediate dimension size.
        is_software_quant (bool): When True, weight_scale_sb is a 2D dummy tile indexed
            as [:, :cur_I128_tile_sz] instead of the normal 3D [:, tile_h, slice].

    Returns:
        None. Results are accumulated in-place into out_psum_lst buffers.
    """
    for tile_h in range(n_H512_tiles):
        for tile_i in range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_local - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I128_tile_sz = cur_tile_I_size // _q_width
            for q_width_I_idx in range(_q_width):
                weight_I_offset = tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_width_I_idx * cur_I128_tile_sz
                weight_I_slice = nl.ds(weight_I_offset, cur_I128_tile_sz)
                nisa.nc_matmul_mx(
                    dst=out_psum_lst[tile_i][:cur_I128_tile_sz, q_width_I_idx, :tile_T_actual],
                    stationary=weight_sb[:, tile_h, weight_I_slice],
                    moving=input_quant_sb[:, tile_h, tile_T_slice],
                    stationary_scale=weight_scale_sb[:, :cur_I128_tile_sz]
                    if is_software_quant
                    else weight_scale_sb[:, tile_h, weight_I_slice],
                    moving_scale=input_scale_sb[:, tile_h, tile_T_slice],
                )
