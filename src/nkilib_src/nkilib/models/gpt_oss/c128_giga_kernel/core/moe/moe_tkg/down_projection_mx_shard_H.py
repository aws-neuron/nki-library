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
from nki.isa.constants import oob_mode

from ...quantization.fp8_quantize import pre_combine_dequant_scales
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE, _psum_alloc, _sbm_alloc, div_ceil
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .projection_mx_constants import (
    SBUF_QUADRANT_SIZE,
    ProjConfig,
    _pmax,
    _psum_fmax,
    _q_height,
    _q_width,
)


def _alloc_down_weight_sb(
    down_weights: nl.NkiTensor,
    n_I512_tile: int,
    H_shard: int,
) -> list:
    """
    Allocate double-buffered SBUF buffers for down projection weight.

    :param down_weights: mxfp4_x4[E, p_I, n_I512_tile, H] @ HBM.
    :param n_I512_tile: Number of I512 tiles = ceil(I / 512).
    :param H_shard: Sharded hidden dimension size.
    :return: List (size=2) of SBUF tensors, each [_pmax, n_I512_tile, H_shard] @ SB (uninitialized).
    """
    base_dtype = down_weights.dtype
    weights_n_buff = []
    for _ in range(2):
        weights_n_buff.append(nl.ndarray((_pmax, n_I512_tile, H_shard), dtype=base_dtype, buffer=nl.sbuf))
    return weights_n_buff


def _load_down_weight(
    weight_sb: nl.NkiTensor,
    down_weights: nl.NkiTensor,
    expert_scalar: nl.NkiTensor,
    p_I: int,
    n_I512_tile: int,
    H_offset: int,
    H_shard: int,
) -> nl.NkiTensor:
    """
    Load down projection weight from HBM into a pre-allocated SBUF buffer.

    :param weight_sb: Pre-allocated SBUF buffer [_pmax, n_I512_tile, H_shard].
    :param down_weights: mxfp4_x4[E, p_I, n_I512_tile, H] @ HBM.
    :param expert_scalar: int32[1, 1] @ SB. Expert index for scalar DGE.
    :param p_I: Partition size of I dimension in HBM (_pmax if I > 512, else I // 4).
    :param n_I512_tile: Number of I512 tiles.
    :param H_offset: Offset into H dimension for sharding.
    :param H_shard: Sharded hidden dimension size.
    :return: weight_sb viewed as down_weights.dtype.
    """
    if p_I != _pmax:
        nisa.memset(dst=weight_sb[:, n_I512_tile - 1, :], value=0)

    down_weights_view = down_weights.select(dim=0, index=expert_scalar).slice(
        dim=2, start=H_offset, end=H_offset + H_shard
    )
    nisa.dma_copy(
        dst=weight_sb[:p_I, :, :],
        src=down_weights_view,
        dge_mode=nisa.dge_mode.hwdge,
    )
    return weight_sb.view(down_weights.dtype)


def _alloc_down_scale_sb(
    n_I512_tile: int,
    H_shard: int,
) -> list:
    """
    Allocate double-buffered SBUF buffers for down projection weight scale.

    :param n_I512_tile: Number of I512 tiles = ceil(I / 512).
    :param H_shard: Sharded hidden dimension size.
    :return: List (size=2) of SBUF tensors, each [_pmax, n_I512_tile, H_shard] @ SB (uninitialized).
    """
    scales_n_buff = []
    for _ in range(2):
        scales_n_buff.append(nl.ndarray((_pmax, n_I512_tile, H_shard), dtype=nl.uint8, buffer=nl.sbuf))
    return scales_n_buff


def _load_down_scale(
    down_scale_sb: nl.NkiTensor,
    down_w_scale: nl.NkiTensor,
    p_idx_vector: nl.NkiTensor,
    n_I512_tile: int,
    H_shard: int,
    H_offset: int,
):
    """
    Load down projection weight scale into pre-allocated SBUF via vector DGE.

    :param down_scale_sb: uint8[_pmax, n_I512_tile, H_shard] @ SB (pre-allocated).
    :param down_w_scale: uint8[E, p_I // _q_height, n_I512_tile, H] @ HBM.
    :param p_idx_vector: float32[_pmax, 1] @ SB. Prepared expert index for vector DGE.
    :param n_I512_tile: Number of I512 tiles.
    :param H_shard: Sharded hidden dimension size.
    :param H_offset: Offset into H dimension for sharding.
    """
    scale_shape = down_w_scale.shape
    down_scale_view = down_w_scale.reshape((scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3]))

    token_indices_on_p = nl.ndarray(p_idx_vector.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=token_indices_on_p, src=p_idx_vector)
    nisa.dma_copy(
        dst=down_scale_sb,
        src=down_scale_view.ap(
            pattern=[[n_I512_tile * down_w_scale.shape[3], _pmax], [down_w_scale.shape[3], n_I512_tile], [1, H_shard]],
            offset=H_offset,
            vector_offset=token_indices_on_p,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip,
    )


def _down_proj_prep_inter_and_weights(
    inter_sb: nl.NkiTensor,
    weight: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    cfg: ProjConfig,
    sbm=None,
    name_prefix: str = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Prep intermediate and weights for down projection:
        - for intermediate, reshape and quantize (and reshape back);
        - for weight, load from HBM into SBUF.

    :param inter_sb: bf16[_pmax, n_I512_tile, BxS, 4] @ SB. Dim I is shuffled on 128.
    :param weight: mxfp_x4[128_I, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
        I-contiguous x4 packing: element [p, tile, h] packs W[512*tile + 4p + q, h]
        for q=0..3 (4 consecutive I values at the same H column). This layout is
        shared with the CTE MX down projection path.
    :param weight_scale: uint8[128_I // _q_height, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :return:
        1. (inter_qtz)        mxfp8_x4[_pmax, cfg.n_total_I512_tile, BxS]
        2. (inter_qtz_scale)  uint8[_pmax, cfg.n_total_I512_tile, BxS]
        3. (weight_qtz)       mxfp_x4[_pmax, cfg.n_total_I512_tile, H_sharded]
        4. (weight_qtz_scale) uint8[_pmax, cfg.n_total_I512_tile, H_sharded]
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H, I, BxS = cfg.H, cfg.I, cfg.BxS
    H_sharded = H // n_prgs
    p_I = _pmax if I > _psum_fmax else I // _q_width  # we do not pad I if I<512 to save HBM

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
    inter_qtz_scale = _sbm_alloc(
        sbm,
        inter_qtz.shape,
        dtype=nl.uint8,
        name=f"{name_prefix}_inter_qtz_scale" if name_prefix else None,
        align=SBUF_QUADRANT_SIZE,
    )
    nisa.quantize_mx(dst=inter_qtz, src=inter_sb, dst_scale=inter_qtz_scale)
    inter_qtz = inter_qtz.reshape((_pmax, cfg.n_total_I512_tile, BxS))
    inter_qtz_scale = inter_qtz_scale.reshape(inter_qtz.shape)

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

    # Check if weight scale is already in SBUF or needs to be loaded from HBM
    weight_qtz_scale = None
    if weight_scale.buffer == nl.sbuf:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_total_I512_tile, H_sharded),
            f"Expect weight_scale in SBUF to have the shape of ({_pmax}, {cfg.n_total_I512_tile}, {H_sharded}), got {weight_scale.shape}",
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

        # Load 4 partitions of scales for every quadrant
        n_quadrants_needed = _pmax // SBUF_QUADRANT_SIZE
        for i_quad in range(n_quadrants_needed):
            kernel_assert(weight_scale.shape == (p_I // _q_height, cfg.n_total_I512_tile, H), "Incorrect weight shape")
            # Scalar DGE needs AP to access either exactly 1 partitions or multiple of 16 partitions
            for i_4 in range(4):
                if i_quad * 4 + i_4 < p_I // _q_height:
                    nisa.dma_copy(
                        src=weight_scale[
                            i_quad * 4 + i_4 : i_quad * 4 + i_4 + 1, :, prg_id * H_sharded : (prg_id + 1) * H_sharded
                        ],
                        dst=weight_qtz_scale[
                            i_quad * SBUF_QUADRANT_SIZE + i_4 : i_quad * SBUF_QUADRANT_SIZE + i_4 + 1, :, :
                        ],
                        dge_mode=nisa.dge_mode.hwdge,
                    )

    return inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale


def down_projection_mx_tp_shard_H(
    inter_sb: nl.NkiTensor,
    weight: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    cfg: ProjConfig,
    partial_output: bool = False,
    pre_quantized: bool = False,
    pre_quantized_scale: Optional[nl.NkiTensor] = None,
    w_dequant_scale=None,
    input_dequant_scale=None,
    name_prefix: str = None,
) -> nl.NkiTensor:
    """
    Performs the Down projection with H-dimension sharding. Math (Neuron matmul):
        inter_sb (moving) [I, BxS] @ weight (stationary) [I, H] → [H, BxS].

    NOTE: each matmul tile reads a weight of shape [_pmax (I), _pmax (H)], however the second dim (on H) is not
    a contiguous slice of 128 elts from the full H. Instead, those are 128 elts with a stride of H//128 (H1).
    This means the final output of shape [_pmax (H0), H//_pmax (H1), BxS] would have a contiguous H1 and strided H0.

    Weight x4 packing convention (I-contiguous, shared with CTE MX down projection):
        Element [p, tile, h] packs W[512*tile + 4p + q, h] for q=0..3
        (4 consecutive I values at the same H column).
        nc_matmul_mx contracts over partition × x4 = I dimension.

    :param inter_sb: bf16[_pmax, n_I512_tile, BxS, 4] @ SB (MX path), or fp8_x4[_pmax, n_I512_tile, BxS] @ SB (pre_quantized).
    :param weight: mxfp_x4[128_I, ceil(I/512), H] @ HBM. I-contiguous x4 packing. NOTE: expect zero-padding.
    :param weight_scale: uint8[128_I // _q_height, ceil(I/512), H] @ HBM (MX), or uint8[_pmax, n_I512_tile, H_sharded] @ SB (pre_quantized).
    :param bias_sb [OPTIONAL]: bf16[_pmax, H_sharded//_pmax] @ SB.
    :param partial_output: If True, skips LNC synchronization and returns only local shard.
    :param pre_quantized: If True, inter_sb is already fp8_x4 quantized with dummy scales (STATIC_MX path).
    :param pre_quantized_scale: uint8 dummy scale for pre-quantized intermediate, required when pre_quantized=True.
    :param w_dequant_scale [OPTIONAL]: Weight dequant scale in SBUF.
        [_pmax, 1] for STATIC_MX, [_pmax, H//128] for ROW_MX, None for MX.
    :param input_dequant_scale [OPTIONAL]: Input dequant scale in SBUF.
        [_pmax, 1] for STATIC_MX, [_pmax, T_padded, 1] for ROW_MX, None for MX.
    :return: bf16[_pmax, H1_sharded, BxS] @ SB if partial_output=True, else bf16[_pmax, H1, BxS] @ SB.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    kernel_assert(cfg.H_sharded % _pmax == 0, "down projection with [H, T] output layout requires H divisible by 128")
    kernel_assert(cfg.BxS <= 128, f"MX4 down proj with HT output layout only supports TKG but got {cfg.BxS=}")

    if pre_quantized:
        # ── STATIC_MX/ROW_MX path: intermediate already quantized, use dummy scales ──
        kernel_assert(pre_quantized_scale is not None, "pre_quantized_scale required when pre_quantized=True")
        inter_qtz_tv = inter_sb if isinstance(inter_sb, nl.NkiTensor) else inter_sb
        inter_qtz_scale = pre_quantized_scale  # Dummy uint8[_pmax, n_I512_tile, BxS] all-127

        weight_base = weight if isinstance(weight, nl.NkiTensor) else weight
        if weight_base.buffer == nl.sbuf:
            # Weight already in SBUF
            weight_qtz = weight_base
            weight_qtz_scale = weight_scale
        else:
            # Load weight from HBM
            p_I = _pmax if cfg.I > _psum_fmax else cfg.I // _q_width
            weight_qtz = nl.ndarray(
                (_pmax, cfg.n_total_I512_tile, cfg.H_sharded),
                dtype=weight.dtype,
                buffer=nl.sbuf,
                name=f'{cfg.name_prefix}down_w_qtz_sb',
            )
            if p_I != _pmax:
                nisa.memset(dst=weight_qtz[:, cfg.n_total_I512_tile - 1, :], value=0.0)
            kernel_assert(weight.shape == (p_I, cfg.n_total_I512_tile, cfg.H), "Incorrect weight shape")
            nisa.dma_copy(
                src=weight[:, :, prg_id * cfg.H_sharded : (prg_id + 1) * cfg.H_sharded],
                dst=weight_qtz[:p_I, :, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
            weight_qtz_scale = weight_scale
    else:
        # ── MX path: quantize intermediate and load weights ──
        inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale = _down_proj_prep_inter_and_weights(
            inter_sb, weight, weight_scale, cfg, name_prefix=name_prefix
        )
        # Wrap in nl.NkiTensor for consistent access in the matmul loop (inter_qtz_tv.slice)
        inter_qtz_tv = inter_qtz
    if cfg.dbg_weight:
        return weight_qtz, weight_qtz_scale

    # Matmul compute, tiles on H
    out_shape = (cfg.H0, cfg.H1_sharded, cfg.BxS) if partial_output else (cfg.H0, cfg.H1, cfg.BxS)
    out_sb = nl.ndarray(out_shape, dtype=nl.bfloat16, buffer=nl.sbuf)

    for i_H1 in range(cfg.H1_sharded):
        # Allocate psum for current H128 tile
        h128_psum = nl.ndarray((cfg.H0, cfg.BxS), dtype=nl.bfloat16, buffer=nl.psum)

        # Loop over I512 tiles
        for i_I512_tile in range(cfg.n_total_I512_tile):
            # Stationary accesses entire I0 because it's been zero-padded
            nisa.nc_matmul_mx(
                dst=h128_psum,
                stationary=weight_qtz[:, i_I512_tile, i_H1 * _pmax : (i_H1 + 1) * _pmax],
                moving=inter_qtz_tv.slice(1, i_I512_tile, i_I512_tile + 1),
                stationary_scale=weight_qtz_scale[:, i_I512_tile, i_H1 * _pmax : (i_H1 + 1) * _pmax],
                moving_scale=inter_qtz_scale[:, i_I512_tile, :],
            )

        # Copy out the current H128 tile to SB, use ACT because DVE is usually bottlenecked
        idx = i_H1 if partial_output else cfg.H1_sharded * prg_id + i_H1
        if w_dequant_scale is not None:
            # ── Software quant path: copy psum to sbuf first, dequant applied after sendrecv ──
            nisa.activation(dst=out_sb[:, idx, :], op=nl.copy, data=h128_psum)
        else:
            # ── MX path or no dequant: fuse bias into the copy ──
            act_bias_arg = bias_sb[:, i_H1] if bias_sb is not None else None
            nisa.activation(dst=out_sb[:, idx, :], op=nl.copy, data=h128_psum, bias=act_bias_arg)

    # Receive projection output from the other NC when LNC > 1
    # Skip sendrecv if partial_output=True (caller handles synchronization or only needs local shard)
    if not partial_output and n_prgs > 1:
        other_prg_id = 1 - prg_id
        nisa.sendrecv(
            src=out_sb[:, prg_id * cfg.H1_sharded : (prg_id + 1) * cfg.H1_sharded, :],
            dst=out_sb[:, other_prg_id * cfg.H1_sharded : (other_prg_id + 1) * cfg.H1_sharded, :],
            send_to_rank=other_prg_id,
            recv_from_rank=other_prg_id,
            pipe_id=0,
        )

    # ── Post-matmul dequant (STATIC_MX / ROW_MX): apply w_dequant_scale, input_dequant_scale, bias ──
    # Applied AFTER LNC reduce so bias is added once and scale is applied to the full sum.
    if w_dequant_scale is not None:
        H1_out = out_sb.shape[1]
        # When partial_output=True, out_sb only has H1_sharded columns corresponding to
        # the local shard, so we need H1_offset to index into the full weight scale.
        # When partial_output=False, out_sb has the full H1 range and indices are global.
        H1_offset = cfg.H1_sharded * prg_id if partial_output else 0
        if w_dequant_scale.shape[1] == 1:
            # ── STATIC_MX: both scales are [_pmax, 1], pre-combine and broadcast ──
            combined = pre_combine_dequant_scales(input_dequant_scale, w_dequant_scale)
            nisa.activation(
                dst=out_sb,
                op=nl.copy,
                data=out_sb,
                scale=combined,
            )
        else:
            # ── ROW_MX: fuse per-row weight scale × per-token input scale ──
            for i_H1 in nl.affine_range(H1_out):
                h_col = H1_offset + i_H1
                nisa.activation(
                    dst=out_sb[:, i_H1, :],
                    op=nl.copy,
                    data=out_sb[:, i_H1, :],
                    scale=w_dequant_scale[:, h_col : h_col + 1],
                )
            if input_dequant_scale is not None:
                # Broadcast input_dequant_scale [_pmax, BxS] across H1 dim to avoid per-H1 loop
                scale_broadcast = (
                    (input_dequant_scale[:, : cfg.BxS, 0])
                    .reshape_dim(dim=1, shape=(1, cfg.BxS))
                    .broadcast(dim=1, size=H1_out)
                )
                nisa.tensor_tensor(
                    dst=out_sb[:, :H1_out, :],
                    data1=out_sb[:, :H1_out, :],
                    data2=scale_broadcast,
                    op=nl.multiply,
                )

        # Add bias if present (after dequant for both STATIC_MX and ROW_MX)
        # Bias is [H0, H1_shard] — only covers the local shard's H1 range.
        # Apply to the correct slice of out_sb based on partial_output layout.
        if bias_sb is not None:
            for i_H1 in nl.affine_range(cfg.H1_sharded):
                out_idx = i_H1 if partial_output else cfg.H1_sharded * prg_id + i_H1
                nisa.activation(
                    dst=out_sb[:, out_idx, :],
                    op=nl.copy,
                    data=out_sb[:, out_idx, :],
                    bias=bias_sb[:, i_H1],
                )

    return out_sb


def down_projection_mx_shard_H(
    inter_sb: nl.NkiTensor,
    weight: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
    sbm=None,
    psum_bank_offset: int = 0,
    name_prefix: str = None,
    out_sb=None,
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

    # Prep inputs
    inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale = _down_proj_prep_inter_and_weights(
        inter_sb, weight, weight_scale, cfg, sbm=sbm, name_prefix=name_prefix
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
                nisa.tensor_copy(
                    dst=bias_broadcasted[:, bias_h_slice],
                    src=bias_psum,
                )
    else:
        # Path 2: Bias already broadcasted to (128, H)
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

            for i_I512_tile in nl.affine_range(cfg.n_total_I512_tile):
                nisa.nc_matmul_mx(
                    dst=psum_bank,
                    stationary=inter_qtz[:, i_I512_tile, curr_BxS_slice],
                    moving=weight_qtz[:, i_I512_tile, curr_H_slice],
                    stationary_scale=inter_qtz_scale[:, i_I512_tile, curr_BxS_slice],
                    moving_scale=weight_qtz_scale[:, i_I512_tile, curr_H_slice],
                )

            H_out_start = H_sharded * prg_id + i_H_tile * cfg.H_tile_size
            curr_H_out_slice = nl.ds(H_out_start, cfg.H_tile_size)

            # Copy psum to output and add bias via tensor_tensor
            if cfg.out_p_offset == 0:
                if bias_broadcasted is not None:
                    nisa.tensor_tensor(
                        dst=out_sb[:curr_BxS, i_BxS_tile, curr_H_out_slice],
                        data1=psum_bank,
                        data2=bias_broadcasted[:curr_BxS, curr_H_slice],
                        op=nl.add,
                    )
                else:
                    engine = nisa.scalar_engine if i_BxS_tile % 2 == 0 else nisa.vector_engine
                    nisa.tensor_copy(
                        dst=out_sb[:curr_BxS, i_BxS_tile, curr_H_out_slice],
                        src=psum_bank,
                        engine=engine,
                    )
            else:
                if bias_broadcasted is not None:
                    nisa.tensor_tensor(
                        dst=out_sb[
                            out_sb_p_start : out_sb_p_start + curr_BxS,
                            i_BxS_tile,
                            curr_H_out_slice,
                        ],
                        data1=psum_bank,
                        data2=bias_broadcasted[:curr_BxS, curr_H_slice],
                        op=nl.add,
                    )
                else:
                    engine = nisa.scalar_engine if i_BxS_tile % 2 == 0 else nisa.vector_engine
                    nisa.tensor_copy(
                        dst=out_sb[
                            out_sb_p_start : out_sb_p_start + curr_BxS,
                            i_BxS_tile,
                            curr_H_out_slice,
                        ],
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
