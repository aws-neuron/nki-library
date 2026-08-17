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

"""MX / STATIC_MX quantized output projection implementation for TKG, not a public-facing API."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ..quantization.fp8_quantize import pre_combine_dequant_scales, static_quantization
from ..utils.common_types import QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_program_sharding_info
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ..utils.tiled_range import TiledRange

P_MAX = 128
F_MAX = 512

# MX packing: 4 elements packed into one float8_e4m3fn_x4 scalar
_MX_PACK_FACTOR = 4
# MX scale group size: one scale per 32 elements
_MX_SCALE_GROUP_SIZE = 32
# SBUF quadrant size for MX scale loading
_SBUF_QUADRANT_SIZE = 32


def _output_projection_tkg_mx(
    attention: nl.NkiTensor,
    weights_qtz: nl.NkiTensor,
    weight_scales_hbm: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    TRANSPOSE_OUT: bool = False,
    quantization_type=None,
    input_scale: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    MX / STATIC_MX quantized output projection for TKG.

    For MX: weights are pre-quantized to unpacked fp8_e4m3fn [NxD//4, H, 4], attention is
    quantized online via quantize_mx. Optimized for BxS <= 128.

    For STATIC_MX: weights are pre-quantized to unpacked fp8_e4m3fn [NxD//4, H, 4], attention
    is quantized via software static_quantization with dummy MX scales (127).
    Post-matmul dequantization applies combined_scale = input_scale * weight_scale.

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size
        NxD_packed: N x D // 4

    Args:
        attention (nl.NkiTensor): [D, B, N, S], Input attention tensor in HBM.
        weights_qtz (nl.NkiTensor): [NxD_packed, H, 4], Pre-quantized weights
            in unpacked fp8_e4m3fn dtype.
        weight_scales_hbm (nl.NkiTensor): MX weight scales in HBM.
            MX: [NxD // 32, H], nl.uint8.
            STATIC_MX: [P_MAX, 1], nl.float32 (per-tensor weight dequant scale).
        bias (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        TRANSPOSE_OUT (bool): Whether returned output should be transposed.
        quantization_type: QuantizationType.MX or QuantizationType.STATIC_MX.
        input_scale (Optional[nl.NkiTensor]): [P_MAX, 1], nl.float32.
            Per-tensor input dequant scale. Required for STATIC_MX, unused for MX.

    Returns:
        nl.NkiTensor:
            If not TRANSPOSE_OUT: [B*S, H] in HBM.
            If TRANSPOSE_OUT: [H1, H0, H2, BxS] in HBM, where
                H0 = n_prgs, H1 = P_MAX, H2 = H // n_prgs // P_MAX.

    Notes:
        - BxS <= 128, BxS % 4 == 0, NxD % 128 == 0, H % 4 == 0
        - TRANSPOSE_OUT not implemented yet.
        - Asserts are done in the top-level output_projection_tkg function.
    """
    if not TRANSPOSE_OUT:
        return _output_projection_tkg_mx_without_transpose_out(
            attention=attention,
            weights_qtz=weights_qtz,
            weight_scales_hbm=weight_scales_hbm,
            bias=bias,
            quantization_type=quantization_type,
            input_scale_hbm=input_scale,
        )
    else:
        return _output_projection_tkg_mx_with_transpose_out(
            attention=attention,
            weights_qtz=weights_qtz,
            weight_scales_hbm=weight_scales_hbm,
            bias=bias,
        )


def _change_layout_and_fold_attention(attn_sb):
    """Apply head-folding layout transform to attention already in SBUF.

    Transforms from [D_packed, 4, B, N, S] to [D_packed_folded, N_folded, BxS, 4].

    Step 1: Move pack dim to end -> [D_packed, B, N, S, 4]
    Step 2: Fold heads + swap B/N -> [D_packed_folded, N_folded, BxS, 4]

    Args:
        attn_sb (nl.NkiTensor): [D_packed, 4, B, N, S] in SBUF (any dtype).
        B (int): Batch size.
        N (int): Number of heads.
        S (int): Sequence length.

    Returns:
        tuple: (attn_folded_sb, D_packed_folded, N_folded)
            attn_folded_sb: [D_packed_folded, N_folded, BxS, 4] in SBUF.
    """

    D_packed, _MX_PACK_FACTOR, B, N, S = attn_sb.shape
    BxS = B * S
    data_dtype = attn_sb.dtype

    # Step 1: Move 4 to end: (D_packed, 4, B, N, S) -> (D_packed, B, N, S, 4)
    attn_move_pack_sb = nl.ndarray(
        (D_packed, B, N, S, _MX_PACK_FACTOR),
        dtype=data_dtype,
        buffer=nl.sbuf,
        name="attn_move_pack_sb",
    )
    nisa.tensor_copy(
        src=attn_sb.reshape((D_packed, B, N, S, _MX_PACK_FACTOR)).ap(
            pattern=[
                [B * N * S * _MX_PACK_FACTOR, D_packed],
                [N * S, B],
                [S, N],
                [1, S],
                [B * N * S, _MX_PACK_FACTOR],
            ],
        ),
        dst=attn_move_pack_sb.reshape((D_packed, B, N, S, _MX_PACK_FACTOR))[...],
    )

    D_packed_folded, N_folded, FOLD_FACTOR = _calculate_head_folded_dimensions(N, D_packed)

    # Step 2: Fold heads and swap B/N: (D_packed, B, N, S, 4) -> (D_packed_folded, N_folded, BxS, 4)
    attn_folded_sb = nl.ndarray(
        (D_packed_folded, N_folded, BxS, _MX_PACK_FACTOR),
        dtype=data_dtype,
        buffer=nl.sbuf,
        name="attn_folded_sb",
    )

    if D_packed >= 32:
        # Direct SBUF->SBUF fold (partition starts are 32-aligned when D_packed >= 32)
        for head_idx in nl.static_range(N):
            head_group_idx, head_offset = divmod(head_idx, FOLD_FACTOR)
            nisa.tensor_copy(
                src=attn_move_pack_sb.ap(
                    pattern=[
                        [B * N * S * _MX_PACK_FACTOR, D_packed],
                        [N * S * _MX_PACK_FACTOR, B],
                        [_MX_PACK_FACTOR, S],
                        [1, _MX_PACK_FACTOR],
                    ],
                    offset=head_idx * S * _MX_PACK_FACTOR,
                ),
                dst=attn_folded_sb.reshape((D_packed_folded, N_folded * BxS, _MX_PACK_FACTOR)).ap(
                    pattern=[
                        [N_folded * BxS * _MX_PACK_FACTOR, D_packed],
                        [S * _MX_PACK_FACTOR, B],
                        [_MX_PACK_FACTOR, S],
                        [1, _MX_PACK_FACTOR],
                    ],
                    offset=(
                        head_offset * D_packed * N_folded * BxS * _MX_PACK_FACTOR
                        + head_group_idx * BxS * _MX_PACK_FACTOR
                    ),
                ),
            )
    else:
        ''''
        D_packed < 32: tensor_copy dst requires 32-aligned partition starts.
        As a temporary solution, do HBM -> HBM roundtrip to fold D_packed.

        Few notes:
         - quantize_mx instruction also requires 32 partion_size, so we must pad or fold.
  
        NOTE: This solution is not performant. It is a temprarary, functionally correct solution.
        '''

        # Step 2a: Swap B and N in SBUF: (D_packed, B, N, S, 4) -> (D_packed, N, B, S, 4)
        attn_swapped_N_B_sb = nl.ndarray(
            (D_packed, N, B, S, _MX_PACK_FACTOR),
            dtype=data_dtype,
            buffer=nl.sbuf,
            name="attn_swapped_N_B_sb",
        )
        nisa.tensor_copy(
            src=attn_move_pack_sb.reshape((D_packed, N, B, S, _MX_PACK_FACTOR)).ap(
                pattern=[
                    [N * B * S * _MX_PACK_FACTOR, D_packed],
                    [S * _MX_PACK_FACTOR, N],
                    [N * S * _MX_PACK_FACTOR, B],
                    [_MX_PACK_FACTOR, S],
                    [1, _MX_PACK_FACTOR],
                ],
            ),
            dst=attn_swapped_N_B_sb[...],
        )

        # Step 2b: DMA to HBM
        attn_swapped_N_B_hbm = nl.ndarray(
            (D_packed, N, BxS, _MX_PACK_FACTOR),
            dtype=data_dtype,
            buffer=nl.shared_hbm,
            name="attn_swapped_hbm",
        )
        nisa.dma_copy(
            dst=attn_swapped_N_B_hbm,
            src=attn_swapped_N_B_sb.reshape((D_packed, N, BxS, _MX_PACK_FACTOR)),
        )

        # Step 2c: DMA back to SBUF with fold (N -> partition dim via FOLD_FACTOR)
        # HBM layout: [D_packed, N, BxS, 4] — D_packed partitions, N in free dim
        # Target SBUF: [D_packed_folded, N_folded, BxS, 4] — D_packed_folded partitions
        for head_idx in nl.static_range(N):
            head_group_idx, head_offset = divmod(head_idx, FOLD_FACTOR)
            nisa.dma_copy(
                dst=attn_folded_sb[nl.ds(head_offset * D_packed, D_packed), head_group_idx, :, :],
                src=attn_swapped_N_B_hbm[:, head_idx, :, :],
            )

    return attn_folded_sb


def _load_weights_folded(weights_qtz, D_packed_folded, N_folded, H_sharded, H_BLOCK_SIZE, n_prgs, prg_id):
    """Load pre-quantized weights with head folding.

    Loads H_shard of weights_qtz with shape [NxD_packed, H, 4] (unpacked fp8_e4m3fn) to SBUF.
    as a list of num_h_blocks x N_folded tensors with shapes [D_packed_folded][H_BLOCK_SIZE]
    in nl.float8_e4m3fn_x4 dtype (view-cast from fp8_e4m3fn after DMA).

    Args:
        weights_qtz (nl.NkiTensor): [NxD_packed, H, 4], Pre-quantized weights in fp8_e4m3fn on HBM.
        D_packed_folded (int): Folded partition dimension.
        N_folded (int): Number of head groups after folding.
        H_sharded (int): Per-core hidden dimension.
        n_prgs (int): Number of logical cores.
        prg_id (int): Current core index.

    Returns:
        list: weights_qtz_sb — [num_h_blocks][N_folded] of [D_packed_folded][H_BLOCK_SIZE] x4 tensors in SBUF.
    """
    _hbm_weight_dtype = weights_qtz.dtype
    weights_qtz_sb = []
    for h_block in TiledRange(H_sharded, H_BLOCK_SIZE):
        weights_h_block_sb = []
        h_offset_global = prg_id * H_sharded + h_block.start_offset
        for head_group_idx in nl.affine_range(N_folded):
            w_tensor = nl.ndarray(
                (D_packed_folded, h_block.size * _MX_PACK_FACTOR),
                dtype=_hbm_weight_dtype,
                buffer=nl.sbuf,
                name=f"weight_qtz_{h_block.index}_{head_group_idx}",
            )
            nisa.dma_copy(
                dst=w_tensor,
                src=weights_qtz[
                    nl.ds(head_group_idx * D_packed_folded, D_packed_folded),
                    nl.ds(h_offset_global, h_block.size),
                    nl.ds(0, _MX_PACK_FACTOR),
                ],
            )
            w_tensor = w_tensor.view(nl.float8_e4m3fn_x4)
            weights_h_block_sb.append(w_tensor)
        weights_qtz_sb.append(weights_h_block_sb)
    return weights_qtz_sb


def _load_weight_scales_folded(
    weight_scales_hbm, D_packed_folded, N_folded, H, H_sharded, H_BLOCK_SIZE, n_prgs, prg_id
):
    """Load MX weight scales with head folding and quadrant-based addressing.

    Loads H_shard of weights_scales_hbm with shape [NxD // 32, H] to SBUF.
    as a list of num_h_blocks x N_folded tensors with shapes [D_packed_folded][H_BLOCK_SIZE].

    Args:
        weight_scales_hbm (nl.NkiTensor): [NxD // 32, H], MX weight scales in uint8.
        D_packed_folded (int): Folded partition dimension.
        N_folded (int): Number of head groups after folding.
        H (int): Full hidden dimension.
        H_sharded (int): Per-core hidden dimension.
        n_prgs (int): Number of logical cores.
        prg_id (int): Current core index.

    Returns:
        list: weight_scales_sb — [num_h_blocks][N_folded] tensors in SBUF each with [D_packed_folded, H_BLOCK_SIZE] shape.
    """
    NUM_QUADRANTS = D_packed_folded // _SBUF_QUADRANT_SIZE
    SCALES_PER_QUADRANT = D_packed_folded // (_MX_SCALE_GROUP_SIZE // _MX_PACK_FACTOR) // NUM_QUADRANTS

    weight_scales_sb = []
    for h_block in TiledRange(H_sharded, H_BLOCK_SIZE):
        scales_h_block_sb = []
        h_offset_global = prg_id * H_sharded + h_block.start_offset
        for head_group_idx in nl.affine_range(N_folded):
            scale_tensor = nl.ndarray(
                (D_packed_folded, h_block.size),
                dtype=nl.uint8,
                buffer=nl.sbuf,
                name=f"scale_tensor_{h_block.index}_{head_group_idx}",
            )
            for quad_idx in nl.affine_range(NUM_QUADRANTS):
                hbm_row_offset = (
                    head_group_idx * (D_packed_folded // (_MX_SCALE_GROUP_SIZE // _MX_PACK_FACTOR))
                    + quad_idx * SCALES_PER_QUADRANT
                )
                sbuf_row_start = quad_idx * _SBUF_QUADRANT_SIZE
                nisa.dma_copy(
                    dst=scale_tensor[
                        sbuf_row_start : sbuf_row_start + SCALES_PER_QUADRANT,
                        : h_block.size,
                    ],
                    src=weight_scales_hbm.ap(
                        pattern=[[H, SCALES_PER_QUADRANT], [1, h_block.size]],
                        offset=hbm_row_offset * H + h_offset_global,
                        dtype=nl.uint8,
                    ),
                )
            scales_h_block_sb.append(scale_tensor)
        weight_scales_sb.append(scales_h_block_sb)
    return weight_scales_sb


def _output_projection_tkg_mx_without_transpose_out(
    attention: nl.NkiTensor,
    weights_qtz: nl.NkiTensor,
    weight_scales_hbm: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    quantization_type: QuantizationType = QuantizationType.MX,
    input_scale_hbm: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    MX / STATIC_MX quantized output projection with head folding optimization.

    For MX: uses hardware quantize_mx with real MX scales.
    For STATIC_MX: uses software static_quantization with dummy MX scales (127)
    and post-matmul dequantization via combined_scale = input_scale * weight_scale.

    Args:
        attention (nl.NkiTensor): [D, B, N, S], Input attention tensor in HBM.
        weights_qtz (nl.NkiTensor): [NxD_packed, H], Pre-quantized weights in fp8_x4.
        weight_scales_hbm (nl.NkiTensor): MX: [NxD // 32, H] uint8. STATIC_MX: [P_MAX, 1] fp32.
        bias (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        quantization_type (QuantizationType): MX or STATIC_MX.
        input_scale_hbm (Optional[nl.NkiTensor]): [P_MAX, 1] fp32. Required for STATIC_MX.

    Returns:
        nl.NkiTensor: [B*S, H] in HBM.
    """
    D, B, N, S = attention.shape
    NxD_packed, H, _ = weights_qtz.shape
    BxS = B * S
    D_packed = D // _MX_PACK_FACTOR

    io_dtype = attention.dtype
    _, n_prgs, prg_id = get_program_sharding_info()
    H_sharded = H // n_prgs

    is_static_mx = quantization_type == QuantizationType.STATIC_MX

    out_hbm_buffer = nl.ndarray((B * S, H), dtype=io_dtype, buffer=nl.shared_hbm)

    # Phase 0: Load and broadcast bias if present
    if bias != None:
        bias_sb = nl.ndarray(
            (P_MAX, H_sharded),
            dtype=bias.dtype,
            buffer=nl.sbuf,
            name='bias_sb',
        )
        nisa.dma_copy(
            src=bias.ap(
                pattern=[[H_sharded, 1], [1, H_sharded]],
                offset=prg_id * H_sharded,
            ),
            dst=bias_sb[0:1, :],
        )
        stream_shuffle_broadcast(bias_sb[0:1, :], bias_sb)

    # Phase 1: Load attention to SBUF [D, B, N, S] -> [D_packed, 4, B, N, S]
    # This is a contiguous load, we shuffle later on sbuf.
    attn_sb = nl.ndarray(
        (D_packed, _MX_PACK_FACTOR, B, N, S),
        dtype=io_dtype,
        buffer=nl.sbuf,
        name="attn_sb",
    )
    nisa.dma_copy(
        src=attention.reshape((D_packed, _MX_PACK_FACTOR, B * N * S)),
        dst=attn_sb.reshape((D_packed, _MX_PACK_FACTOR, B * N * S)),
    )

    # Phase 1.5: STATIC_MX software quantization (before layout transform)
    # Divide input by input_scale and clip to FP8 range.
    if is_static_mx:
        input_scale_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="input_scale_sb")
        nisa.dma_copy(dst=input_scale_sb, src=input_scale_hbm)
        attn_flat = attn_sb.reshape((D_packed, _MX_PACK_FACTOR * B * N * S))
        attn_quantized, _ = static_quantization(attn_flat, input_scale_sb[:D_packed, :])
        attn_sb = attn_quantized.reshape((D_packed, _MX_PACK_FACTOR, B, N, S))

    '''
     Phase 2: Layout transform + head folding (shared for both MX and STATIC_MX)
     Change layout:
     from: [D_packed, 4, B, N, S]
     to  : [D_packed_folded, N_folded, BxS, 4]

     There two parts to this layout_change:
     (1) If D_packed is small -> fold N into partition dimension
     (2) Move "4" all the way to the right + swap B & N.
    '''
    attn_for_quant_sb = _change_layout_and_fold_attention(attn_sb)
    D_packed_folded, N_folded, _, _ = attn_for_quant_sb.shape
    # Current shape: [D_packed_folded, N_folded, BxS, 4]

    # Pad BxS to next multiple of 4 for nc_matmul_mx even free-dim constraint.
    T_padded = div_ceil(BxS, _MX_PACK_FACTOR) * _MX_PACK_FACTOR
    if T_padded > BxS:
        attn_padded = nl.ndarray(
            (D_packed_folded, N_folded, T_padded, _MX_PACK_FACTOR),
            dtype=attn_for_quant_sb.dtype,
            buffer=nl.sbuf,
            name="attn_for_quant_padded",
        )
        nisa.memset(dst=attn_padded, value=0)
        nisa.tensor_copy(dst=attn_padded[:, :, :BxS, :], src=attn_for_quant_sb)
        attn_for_quant_sb = attn_padded

    # Phase 3: Quantize attention
    # [D_packed_folded, N_folded, T_padded, 4]  -> [D_packed_folded, N_folded, T_padded]
    if is_static_mx:
        # Reinterpret cast fp8 -> fp8_x4 + dummy MX scales
        attn_qtz_tv = (attn_for_quant_sb.reshape((D_packed_folded, N_folded, T_padded, _MX_PACK_FACTOR))).view(
            nl.float8_e4m3fn_x4
        )
        attn_qtz_sb = attn_qtz_tv.reshape((D_packed_folded, N_folded, T_padded))
        attn_scale_sb = nl.ndarray(
            (D_packed_folded, N_folded, T_padded), dtype=nl.uint8, buffer=nl.sbuf, name='attn_scale_dummy'
        )
        nisa.memset(dst=attn_scale_sb, value=127)
    else:
        attn_qtz_sb = nl.ndarray(
            (D_packed_folded, N_folded, T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name='attn_qtz_sb'
        )
        attn_scale_sb = nl.ndarray(
            (D_packed_folded, N_folded, T_padded), dtype=nl.uint8, buffer=nl.sbuf, name='attn_scale_sb'
        )
        nisa.quantize_mx(dst=attn_qtz_sb, src=attn_for_quant_sb, dst_scale=attn_scale_sb)

    # Phase 4: Load weights (and MX scales for MX path)
    H_BLOCK_SIZE = min(2048, H_sharded)
    # weights_qtz_sb: List of (num_h_blocks x N_folded) tensors with shapes [D_packed_folded][H_BLOCK_SIZE].
    weights_qtz_sb = _load_weights_folded(
        weights_qtz, D_packed_folded, N_folded, H_sharded, H_BLOCK_SIZE, n_prgs, prg_id
    )
    if is_static_mx:
        # Dummy weight scales (to be passed in matmul_mx), use minimum possible free dimension.
        weight_scale_dummy_sb = nl.ndarray(
            (D_packed_folded, H_BLOCK_SIZE),
            dtype=nl.uint8,
            buffer=nl.sbuf,
            name='weight_scale_dummy',
        )
        nisa.memset(dst=weight_scale_dummy_sb, value=127)

        # This are real weight scales, used for dequantization.
        # Pre-combine dequant scales: combined = input_scale * weight_scale
        weight_scale_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="weight_scale_sb")
        nisa.dma_copy(dst=weight_scale_sb, src=weight_scales_hbm)
        combined_dequant_scale = pre_combine_dequant_scales(input_scale_sb, weight_scale_sb)
    else:
        weight_scales_sb = _load_weight_scales_folded(
            weight_scales_hbm, D_packed_folded, N_folded, H, H_sharded, H_BLOCK_SIZE, n_prgs, prg_id
        )

    # Phase 5: Matmul + output
    output_sb = nl.ndarray((T_padded, H_sharded), dtype=io_dtype, buffer=nl.sbuf)

    for h_block in TiledRange(H_sharded, H_BLOCK_SIZE):
        for h_tile in TiledRange(h_block.size, F_MAX):
            mm_result_psum = nl.ndarray(
                (T_padded, h_tile.size),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for head_group_idx in nl.affine_range(N_folded):
                if is_static_mx:
                    moving_scale = weight_scale_dummy_sb[:, nl.ds(h_tile.start_offset, h_tile.size)]
                else:
                    moving_scale = weight_scales_sb[h_block.index][head_group_idx][
                        :, nl.ds(h_tile.start_offset, h_tile.size)
                    ]
                nisa.nc_matmul_mx(
                    dst=mm_result_psum[:T_padded, : h_tile.size],
                    stationary=attn_qtz_sb[:, head_group_idx, :],
                    moving=weights_qtz_sb[h_block.index][head_group_idx][:, nl.ds(h_tile.start_offset, h_tile.size)],
                    stationary_scale=attn_scale_sb[:, head_group_idx, :],
                    moving_scale=moving_scale,
                )

            h_offset = h_block.start_offset + h_tile.start_offset
            if is_static_mx:
                # Post-matmul dequant: copy from PSUM with combined scale
                nisa.activation(
                    dst=output_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                    op=nl.copy,
                    data=mm_result_psum[:T_padded, : h_tile.size],
                    scale=combined_dequant_scale[:T_padded, :],
                )
                if bias != None:
                    nisa.tensor_tensor(
                        dst=output_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                        data1=output_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                        data2=bias_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                        op=nl.add,
                    )
            else:
                if bias != None:
                    nisa.tensor_tensor(
                        dst=output_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                        data1=mm_result_psum[:T_padded, : h_tile.size],
                        data2=bias_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                        op=nl.add,
                    )
                else:
                    nisa.tensor_copy(
                        dst=output_sb[:T_padded, nl.ds(h_offset, h_tile.size)],
                        src=mm_result_psum[:T_padded, : h_tile.size],
                    )

    # Store only physical BxS rows to HBM output
    nisa.dma_copy(
        dst=out_hbm_buffer[:BxS, nl.ds(prg_id * H_sharded, H_sharded)],
        src=output_sb[:BxS, :H_sharded],
    )

    return out_hbm_buffer


def _output_projection_tkg_mx_with_transpose_out(
    attention: nl.NkiTensor,
    weights_qtz: nl.NkiTensor,
    weight_scales_hbm: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
) -> nl.NkiTensor:
    """
    MX quantized output projection with transposed output (not yet implemented).

    Computes: (attention @ weight + bias)^T.

    Args:
        attention (nl.NkiTensor): [D, B, N, S], Input attention tensor in HBM.
        weights_qtz (nl.NkiTensor): [NxD_packed, H, 4], Pre-quantized weights
            in unpacked fp8_e4m3fn dtype.
        weight_scales_hbm (nl.NkiTensor): [NxD // 32, H], MX weight scales
            in nl.uint8 dtype.
        bias (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.

    Returns:
        nl.NkiTensor: [H1, H0, H2, BxS] in HBM, where
            H0 = n_prgs, H1 = P_MAX, H2 = H // n_prgs // P_MAX.

    Notes:
        - TRANSPOSE_OUT not implemented yet.
        - BxS <= 128, BxS % 4 == 0, NxD % 128 == 0, H % 4 == 0.
    """
    kernel_assert(
        False,
        "[Output Projection TKG MX] TRANSPOSE_OUT feature not implemented yet",
    )


def _calculate_head_folded_dimensions(N: int, D: int) -> tuple[int, int, int]:
    """
    Maximize contraction dimension by folding heads N into D.

    D is usually <= 32 in the MX case since it is packed. nc_matmul_mx
    requires the partition dimension to be 32, 64, or 128. This function
    finds the largest FOLD_FACTOR such that D * FOLD_FACTOR hits one of
    those sizes and evenly divides N.

    Args:
        N (int): Number of heads.
        D (int): Packed head dimension size (D_original // 4).

    Returns:
        tuple[int, int, int]: (D_folded, N_folded, FOLD_FACTOR).
    """
    FOLD_FACTOR = 1
    # Valid partition sizes for nc_matmul_mx
    _VALID_PARTITION_SIZES = [128, 64, 32]
    for valid_size in _VALID_PARTITION_SIZES:
        if valid_size % D == 0:
            candidate = valid_size // D
            if candidate <= N and N % candidate == 0:
                FOLD_FACTOR = candidate
                break

    D_folded = D * FOLD_FACTOR
    N_folded = N // FOLD_FACTOR

    return D_folded, N_folded, FOLD_FACTOR
