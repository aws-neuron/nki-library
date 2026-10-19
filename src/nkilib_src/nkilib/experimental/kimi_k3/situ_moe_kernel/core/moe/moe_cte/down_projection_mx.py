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
Down projection sub-kernels with LNC sharding on H (hidden) dimension.

LNC Sharding Strategy: H dimension
- When LNC=2, weights and output are sharded on H dimension
- Used by: selective-expert MoE algorithm, dense MLP (but algorithm-independent)

These sub-kernels can be used by any algorithm that requires H-sharded down projection.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE, _psum_alloc, _sbm_alloc, div_ceil
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .projection_mx_constants import (
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
    ProjConfig,
    _pmax,
    _psum_fmax,
    _q_height,
    _q_width,
)

# Packed-scale layout: number of I/512 tiles folded into one packed buffer along the
# partition dim. Tile k occupies _q_width consecutive partitions starting at
# (k % _PACKED_TILES_PER_BUFFER) * _q_width within packed buffer k // _PACKED_TILES_PER_BUFFER.
_PACKED_TILES_PER_BUFFER = 4


def _down_proj_prep_inter_and_weights(
    inter_sb: nl.NkiTensor,
    weight: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    cfg: ProjConfig,
    sbm=None,
    name_prefix: str = None,
    inter_quant_recip: Optional[nl.NkiTensor] = None,
    dummy_inter_scale: Optional[nl.NkiTensor] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Prep intermediate and weights for down projection.

    For intermediate: reshape and quantize (and reshape back).
    For weight: load from HBM into SBUF.

    Args:
        inter_sb (nl.NkiTensor): bf16[_pmax, n_I512_tile, BxS, 4] @ SB. Dim I is shuffled on 128.
        weight (nl.NkiTensor): mxfp_x4[128_I, ceil(I/512), H] @ HBM. Expects zero-padding.
            I-contiguous x4 packing: element [p, tile, h] packs W[512*tile + 4p + q, h]
            for q=0..3 (4 consecutive I values at the same H column). Layout shared with
            the CTE MX down projection path.
        weight_scale (nl.NkiTensor): uint8[128_I // _q_height, ceil(I/512), H] @ HBM. Expects zero-padding.
        cfg (ProjConfig): Projection configuration.
        sbm (SbufManager, optional): SBUF allocator.
        name_prefix (str, optional): Prefix for SBUF buffer names.
        inter_quant_recip (nl.NkiTensor, optional): fp32[_pmax, 1] = 1 / down_in_scale. When
            provided, take the STATIC_MX path: quantize the intermediate via tensor_scalar
            (saturation cast to fp8) instead of nisa.quantize_mx. Caller must also pass
            dummy_inter_scale.
        dummy_inter_scale (nl.NkiTensor, optional): uint8[_pmax, BxS_tile_sz] all-127
            (per-tile shape, not the full 3D scale shape). Matmul site slices 2D
            per BxS tile under STATIC_MX.

    Returns:
        tuple: (inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale)
            - inter_qtz (nl.NkiTensor): mxfp8_x4[_pmax, cfg.n_total_I512_tile, BxS]
            - inter_qtz_scale (nl.NkiTensor): uint8[_pmax, cfg.n_total_I512_tile, BxS]
            - weight_qtz (nl.NkiTensor): mxfp_x4[_pmax, cfg.n_total_I512_tile, H_sharded]
            - weight_qtz_scale (nl.NkiTensor): uint8[_pmax, cfg.n_total_I512_tile, H_sharded]
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H, I, BxS = cfg.H, cfg.I, cfg.BxS
    H_sharded = H // n_prgs
    p_I = _pmax if I > _psum_fmax else I // _q_width  # we do not pad I if I<512 to save HBM

    is_static_quant = inter_quant_recip is not None
    if is_static_quant:
        kernel_assert(dummy_inter_scale is not None, "dummy_inter_scale required for STATIC_MX")

    """
    Quantize intermediate state to MXFP8.

    Quantize inter_sb into mxfp4_x4[_pmax, ceil(I/512), BxS] @ SB.
    When I%512 != 0, the final I512 tile of inter_sb will contain garbage.
    nc_matmul_mx requires 32/64/128 partitions input so all 128 partitions are used (including garbage).
    However, we memset the last tile of weight_qtz and weight_qtz_scale so the garbage does not matter.
    """
    inter_sb = inter_sb.reshape((_pmax, cfg.n_total_I512_tile * BxS * _q_width))
    inter_qtz = _sbm_alloc(
        sbm,
        (_pmax, cfg.n_total_I512_tile * BxS),
        dtype=nl.float8_e4m3fn_x4,
        name=f"{name_prefix}_inter_qtz" if name_prefix else None,
        align=SBUF_QUADRANT_SIZE,
    )
    if is_static_quant:
        # STATIC_MX: tensor_scalar(intermediate × 1/down_in_scale) with fp8 saturation cast.
        inter_qtz_fp8 = inter_qtz.view(nl.float8_e4m3fn)
        nisa.tensor_scalar(
            dst=inter_qtz_fp8,
            data=inter_sb,
            op0=nl.multiply,
            operand0=inter_quant_recip,
        )
        # Pass the [_pmax, 1] dummy buffer through; matmul site builds the per-tile broadcast view.
        inter_qtz_scale = dummy_inter_scale
    else:
        inter_qtz_scale = _sbm_alloc(
            sbm,
            inter_qtz.shape,
            dtype=nl.uint8,
            name=f"{name_prefix}_inter_qtz_scale" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.quantize_mx(dst=inter_qtz, src=inter_sb, dst_scale=inter_qtz_scale)
        inter_qtz_scale = inter_qtz_scale.reshape((_pmax, cfg.n_total_I512_tile, BxS))
    inter_qtz = inter_qtz.reshape((_pmax, cfg.n_total_I512_tile, BxS))

    if cfg.dbg_hidden:
        return inter_qtz, inter_qtz_scale, None, None  # DEBUG

    weight_qtz = None
    if weight.buffer == nl.sbuf:
        weight_qtz = weight
    else:
        # Load weight into [I0, ceil(I/512), H_sharded] NOTE: this is pre-quantized and each elt is mxfp_x4 (packed I)
        weight_qtz = _sbm_alloc(
            sbm,
            (_pmax, cfg.n_total_I512_tile, H_sharded),
            dtype=weight.dtype,
            name=f'{cfg.name_prefix}down_w_qtz_sb',
            align=SBUF_QUADRANT_SIZE,
        )
        # Memset weight if input weight HBM does not pad on par dim
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz[:, cfg.n_total_I512_tile - 1, :], value=0.0)

        kernel_assert(weight.shape == (p_I, cfg.n_total_I512_tile, H), "Incorrect weight shape")
        nisa.dma_copy(
            src=weight[:, :, prg_id * H_sharded : (prg_id + 1) * H_sharded],
            dst=weight_qtz[:p_I, :, :],
            dge_mode=nisa.dge_mode.hwdge,
        )

    # Check if weight scale is already in SBUF or needs to be loaded from HBM.
    # Packed layout: shape is (_pmax, n_packed, H_sharded) where n_packed =
    # ceil(n_total_I512_tile / 4). The matmul-site branch handles the packed
    # partition-offset slicing.
    weight_qtz_scale = None
    if weight_scale.buffer == nl.sbuf:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_total_I512_tile, H_sharded) or weight_scale.shape[0] == _pmax,
            f"Expect weight_scale in SBUF to have the shape of ({_pmax}, {cfg.n_total_I512_tile}, {H_sharded}) "
            f"or packed (_pmax, n_packed, H_sharded), got {weight_scale.shape}",
        )
        weight_qtz_scale = weight_scale
    else:
        # Load weight scale into [I0, ceil(I/512), H_sharded] NOTE: we have 1 scale per 8(p)x4(f) tile, but still span across full pdim with gaps
        weight_qtz_scale = _sbm_alloc(
            sbm,
            weight_qtz.shape,
            dtype=nl.uint8,
            name=f"{cfg.name_prefix}down_w_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        # Memset weight scale if input weight scale HBM does not pad on par dim
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz_scale[:, cfg.n_total_I512_tile - 1, :], value=0)

        # Load SCALE_P_ELEM_PER_QUADRANT partitions of scales for every quadrant
        n_quadrants_needed = _pmax // SBUF_QUADRANT_SIZE
        kernel_assert(weight_scale.shape == (p_I // _q_height, cfg.n_total_I512_tile, H), "Incorrect weight shape")
        for i_quad in range(n_quadrants_needed):
            # Scalar DGE needs AP to access either exactly 1 partitions or multiple of 16 partitions
            for i_scale_part in range(SCALE_P_ELEM_PER_QUADRANT):
                src_part = i_quad * SCALE_P_ELEM_PER_QUADRANT + i_scale_part
                if src_part < p_I // _q_height:
                    nisa.dma_copy(
                        src=weight_scale[src_part : src_part + 1, :, prg_id * H_sharded : (prg_id + 1) * H_sharded],
                        dst=weight_qtz_scale[
                            i_quad * SBUF_QUADRANT_SIZE + i_scale_part : i_quad * SBUF_QUADRANT_SIZE + i_scale_part + 1,
                            :,
                            :,
                        ],
                        dge_mode=nisa.dge_mode.hwdge,
                    )

    return inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale


def down_projection_mx(
    inter_sb: nl.NkiTensor,
    weight: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
    sbm=None,
    psum_bank_offset: int = 0,
    name_prefix: Optional[str] = None,
    out_sb: Optional[nl.NkiTensor] = None,
    is_packed_scale: bool = False,
    w_dequant_scale: Optional[nl.NkiTensor] = None,
    inter_quant_recip: Optional[nl.NkiTensor] = None,
    dummy_inter_scale: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    Perform down projection with MXFP quantization.

    Computes weight @ intermediate + bias using MXFP quantized weights,
    producing final MLP output. This version supports larger BxS values (CTE workloads)
    by tiling the BxS dimension.

    Args:
        inter_sb (nl.NkiTensor): Intermediate activations of shape [128, n_I512_tile, BxS, 4]
            in SBUF with I dimension shuffled on 128 partitions, bf16 type.
        weight (nl.NkiTensor): Quantized weights of shape [128, ceil(I/512), H] in HBM,
            mxfp_x4 type (supports MXFP4/MXFP8), zero-padded.
        weight_scale (nl.NkiTensor): Weight scales of shape [128//8, ceil(I/512), H] in HBM,
            uint8 type, zero-padded.
        bias_sb (nl.NkiTensor): Optional bias of shape [1, H_sharded] in SBUF, bf16 type.
        cfg (ProjConfig): Projection configuration with H, I, BxS, sharding info.
        is_packed_scale (bool): Set True when caller pre-packed weight_scale into the
            compressed layout (_pmax, n_packed, H_sharded) where multiple I/512 tiles
            are folded along the partition dim. Default False = standard
            (_pmax, n_total_I512_tile, H_sharded) layout.

    Returns:
        output (nl.NkiTensor): Down projection result of shape [128, ceil(BxS/128), H] in SBUF,
            bf16 type. Note: end of last tile contains garbage when BxS % 128 != 0.

    Notes:
        - Math: weight [I, H] @ inter_sb [I, BxS] → [BxS, H]
        - Quantizes intermediate activations online to MXFP8
        - Uses nc_matmul_mx for MXFP matrix multiplication
        - Supports both MXFP4 and MXFP8 weight dtypes
        - Tiles BxS dimension in chunks of 128
        - Bias added only by program 0 when using LNC sharding
        - Supports optional partition offset for TKG scenarios
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H, H0, H1, H1_sharded, I, BxS = cfg.H, cfg.H0, cfg.H1, cfg.H1_sharded, cfg.I, cfg.BxS
    H_sharded = H // n_prgs

    n_BxS_tile = div_ceil(BxS, _pmax)
    BxS_tile_sz = _pmax

    is_static_quant = inter_quant_recip is not None

    # Prep inputs
    inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale = _down_proj_prep_inter_and_weights(
        inter_sb,
        weight,
        weight_scale,
        cfg,
        sbm=sbm,
        name_prefix=name_prefix,
        inter_quant_recip=inter_quant_recip,
        dummy_inter_scale=dummy_inter_scale,
    )

    if cfg.dbg_weight:
        return weight_qtz, weight_qtz_scale

    """
    Bias handling with two paths based on input bias shape.
    
    Path 1: bias_sb is (1, H) - broadcast to (128, H_sharded) using configured method
    Path 2: bias_sb is already (128, H) - use tensor_tensor add after psum copy
    
    Broadcast methods (controlled by cfg.use_stream_shuffle_broadcast):
    - True (default): Use stream_shuffle_broadcast (nc_stream_shuffle)
    - False: Use PE broadcast via matmul with ones
    """
    bias_broadcasted = None
    if bias_sb and bias_sb.shape[0] == 1:
        bias_broadcasted = _sbm_alloc(
            sbm,
            (BxS_tile_sz, H_sharded),
            dtype=bias_sb.dtype,
            name=f"{name_prefix}_bias_bcast" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )

        if cfg.use_stream_shuffle_broadcast:
            # Stream shuffle broadcast: broadcast from partition 0 to all partitions
            stream_shuffle_broadcast(src=bias_sb, dst=bias_broadcasted)
        else:
            # PE broadcast via matmul with ones tiled by 512 chunks
            ones_sb = _sbm_alloc(
                sbm,
                (1, _pmax),
                dtype=nl.bfloat16,
                name=f"{name_prefix}_ones_sb" if name_prefix else None,
                align=SBUF_QUADRANT_SIZE,
            )
            nisa.memset(dst=ones_sb, value=1.0)

            n_bias_tiles = div_ceil(H_sharded, _psum_fmax)
            for i_bias_tile in nl.affine_range(n_bias_tiles):
                bias_h_offset = i_bias_tile * _psum_fmax
                bias_h_size = min(_psum_fmax, H_sharded - bias_h_offset)
                bias_h_slice = nl.ds(bias_h_offset, bias_h_size)
                bias_psum = _psum_alloc((BxS_tile_sz, bias_h_size), nl.float32, sbm, psum_bank_offset * PSUM_BANK_SIZE)
                nisa.nc_matmul(
                    dst=bias_psum,
                    stationary=ones_sb[:, :BxS_tile_sz],
                    moving=bias_sb[:, bias_h_slice],
                    is_stationary_onezero=True,
                )
                engine = nisa.vector_engine if i_bias_tile % 2 == 0 else nisa.scalar_engine
                nisa.tensor_copy(dst=bias_broadcasted[:, bias_h_slice], src=bias_psum, engine=engine)
    else:
        # Path 2: Bias already broadcasted to (128, H)
        kernel_assert(
            bias_sb == None or bias_sb.shape == (_pmax, H),
            f"Expects pre-broadcast bias of shape ({_pmax}, {H}), got {bias_sb.shape if bias_sb != None else None}",
        )
        bias_broadcasted = bias_sb

    # Allocate output buffer
    if out_sb != None:
        out_sb_p_start = 0
        out_sb_p_end = BxS_tile_sz
    elif cfg.out_p_offset != 0:
        out_sb = _sbm_alloc(
            sbm,
            (_pmax, n_BxS_tile, H),
            dtype=nl.bfloat16,
            name=f"{name_prefix}_out_sb" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
        out_sb_p_start = cfg.out_p_offset
        out_sb_p_end = cfg.out_p_offset + BxS
    else:
        out_sb = _sbm_alloc(
            sbm,
            (BxS_tile_sz, n_BxS_tile, H),
            dtype=nl.bfloat16,
            name=f"{name_prefix}_out_sb" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
        out_sb_p_start = 0
        out_sb_p_end = BxS_tile_sz

    for i_H_tile in nl.affine_range(cfg.n_H_tile_sharded):
        H_offset = i_H_tile * cfg.H_tile_size
        curr_H_slice = nl.ds(H_offset, cfg.H_tile_size)

        for i_BxS_tile in nl.affine_range(n_BxS_tile):
            BxS_offset = i_BxS_tile * BxS_tile_sz
            curr_BxS = min(BxS_tile_sz, BxS - BxS_offset)
            curr_BxS_slice = nl.ds(BxS_offset, curr_BxS)

            bank_id = (i_H_tile * n_BxS_tile + i_BxS_tile) % NUM_HW_PSUM_BANKS
            psum_bank = _psum_alloc((curr_BxS, cfg.H_tile_size), nl.bfloat16, sbm, bank_id * PSUM_BANK_SIZE)

            # STATIC_MX: both scale operands are dummy all-127 buffers with per-tile
            # shape (no I512 dim, no H_tile dim). The buffers are sized for one tile
            # — read the same first-tile bytes every iteration (all-127 anyway).
            if is_static_quant:
                stationary_scale_static = inter_qtz_scale[:, :curr_BxS]
                moving_scale_static = weight_qtz_scale[:, : cfg.H_tile_size]
            else:
                stationary_scale_static = None
                moving_scale_static = None

            for i_I512_tile in nl.affine_range(cfg.n_total_I512_tile):
                # Packed weight scale: I/512 tile is folded into the partition dim.
                # Tile k lives at within-quadrant offset (k % _PACKED_TILES_PER_BUFFER) * _q_width
                # in packed buffer k // _PACKED_TILES_PER_BUFFER.
                if is_static_quant:
                    moving_scale_view = moving_scale_static
                    stationary_scale_arg = stationary_scale_static
                elif is_packed_scale:
                    _packed_buf_idx = i_I512_tile // _PACKED_TILES_PER_BUFFER
                    _slot_part_off = (i_I512_tile % _PACKED_TILES_PER_BUFFER) * _q_width
                    moving_scale_view = weight_qtz_scale[_slot_part_off:, _packed_buf_idx, curr_H_slice]
                    stationary_scale_arg = inter_qtz_scale[:, i_I512_tile, curr_BxS_slice]
                else:
                    moving_scale_view = weight_qtz_scale[:, i_I512_tile, curr_H_slice]
                    stationary_scale_arg = inter_qtz_scale[:, i_I512_tile, curr_BxS_slice]
                nisa.nc_matmul_mx(
                    dst=psum_bank,
                    stationary=inter_qtz[:, i_I512_tile, curr_BxS_slice],
                    moving=weight_qtz[:, i_I512_tile, curr_H_slice],
                    stationary_scale=stationary_scale_arg,
                    moving_scale=moving_scale_view,
                )

            H_out_start = H_sharded * prg_id + i_H_tile * cfg.H_tile_size
            curr_H_out_slice = nl.ds(H_out_start, cfg.H_tile_size)

            # Compute the partition slice once: when out_p_offset == 0, this is
            # equivalent to [:curr_BxS]. When out_p_offset != 0, the caller's
            # out_sb has a partition prefix to skip past.
            out_sb_dst = out_sb[
                out_sb_p_start : out_sb_p_start + curr_BxS,
                i_BxS_tile,
                curr_H_out_slice,
            ]

            if w_dequant_scale is not None:
                # ── STATIC_MX: fused (psum * w_dequant_scale [+ bias]) in one op ──
                if bias_broadcasted is not None:
                    nisa.scalar_tensor_tensor(
                        dst=out_sb_dst,
                        data=psum_bank,
                        op0=nl.multiply,
                        operand0=w_dequant_scale,
                        op1=nl.add,
                        operand1=bias_broadcasted[:curr_BxS, curr_H_slice],
                    )
                else:
                    nisa.tensor_scalar(
                        dst=out_sb_dst,
                        data=psum_bank,
                        op0=nl.multiply,
                        operand0=w_dequant_scale,
                    )
            elif bias_broadcasted != None:
                nisa.tensor_tensor(
                    dst=out_sb_dst,
                    data1=psum_bank,
                    data2=bias_broadcasted[:curr_BxS, curr_H_slice],
                    op=nl.add,
                )
            else:
                engine = nisa.scalar_engine if i_BxS_tile % 2 == 0 else nisa.vector_engine
                nisa.tensor_copy(
                    dst=out_sb_dst,
                    src=psum_bank,
                    engine=engine,
                )

    # LNC sync
    if n_prgs > 1:
        other_prg = 1 - prg_id
        nisa.sendrecv(
            src=out_sb[out_sb_p_start:out_sb_p_end, :, H_sharded * prg_id : H_sharded * (prg_id + 1)],
            dst=out_sb[out_sb_p_start:out_sb_p_end, :, H_sharded * other_prg : H_sharded * (other_prg + 1)],
            send_to_rank=other_prg,
            recv_from_rank=other_prg,
            pipe_id=0,
        )

    return out_sb
