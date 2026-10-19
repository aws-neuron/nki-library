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
Gate/Up projection sub-kernels for MoE (no LNC sharding on H).
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ...quantization.fp8_quantize import pre_combine_dequant_scales
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import (
    PSUM_BANK_SIZE,
    _psum_alloc,
    _sbm_alloc,
    div_ceil,
)
from .projection_mx_constants import (
    SBUF_QUADRANT_SIZE,
    ProjConfig,
    _pmax,
    _psum_bmax,
    _psum_fmax,
    _q_width,
)

# H-chunk size for streamed weight loading (1536 = 3 H512 tiles).
# Used when dst_weight_sb is provided, to overlap weight DMA with matmul.
_H_CHUNK_SZ = 1536
_H_CHUNK_N_H512 = _H_CHUNK_SZ // (_pmax * _q_width)

# Packed-scale layout: number of H-tiles folded into one packed buffer along the
# partition dim. Tile k occupies _q_width consecutive partitions starting at
# (k % _PACKED_TILES_PER_BUFFER) * _q_width within packed buffer k // _PACKED_TILES_PER_BUFFER.
_PACKED_TILES_PER_BUFFER = 4


def _bulk_load_weight_from_hbm(
    weight_qtz: nl.NkiTensor,
    cfg: ProjConfig,
    sbm,
    name_prefix: str,
) -> nl.NkiTensor:
    """Load full H-sharded weight tile from HBM into a freshly-allocated SBUF buffer."""
    H0, I = cfg.H0, cfg.I
    prg_id = cfg.prg_id

    kernel_assert(
        weight_qtz.shape == (_pmax, cfg.n_H512_tile, I),
        f"Expect weight_qtz in HBM to be in shape (128, {cfg.n_H512_tile}, {I}), got {weight_qtz.shape}",
    )
    weight_qtz_sb = _sbm_alloc(
        sbm,
        (H0, cfg.n_H512_tile_sharded, I),
        dtype=weight_qtz.dtype,
        name=f"{name_prefix}_weight_qtz_sb" if name_prefix else None,
        align=SBUF_QUADRANT_SIZE,
    )
    nisa.dma_copy(
        dst=weight_qtz_sb,
        src=weight_qtz.slice(1, prg_id * cfg.n_H512_tile_sharded, (prg_id + 1) * cfg.n_H512_tile_sharded),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return weight_qtz_sb


def _matmul_one_H512_tile(
    i_H512_tile: int,
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scale_sb: nl.NkiTensor,
    weight_qtz_tv: nl.NkiTensor,
    weight_scale_tv: nl.NkiTensor,
    out_psum_lst: list,
    cfg: ProjConfig,
    cur_BxS_tile_offset: int,
    cur_BxS_tile_sz: int,
    is_packed_scale: bool = False,
    is_packed_moving_scale: bool = False,
    is_static_quant: bool = False,
):
    """
    Run nc_matmul_mx for all I512 x q_width sub-tiles of a single H512 tile.

    Args:
        i_H512_tile (int): Index of the H512 tile being multiplied.
        hidden_qtz_sb (nl.NkiTensor): Quantized hidden activations in SBUF.
        hidden_scale_sb (nl.NkiTensor): Hidden scale factors in SBUF.
        weight_qtz_tv (nl.NkiTensor): Quantized weight tile (SBUF or streamed buffer).
        weight_scale_tv (nl.NkiTensor): Weight scale factors (standard or packed layout).
        out_psum_lst (list): Per-I512-tile PSUM destinations to accumulate into.
        cfg (ProjConfig): Projection configuration (I, q_width, n_total_I512_tile, etc.).
        cur_BxS_tile_offset (int): Current BxS tile starting offset.
        cur_BxS_tile_sz (int): Current BxS tile size (may be < BxS_tile_sz at the tail).
        is_packed_scale (bool): If True, weight_scale_tv (stationary) is in packed layout.
        is_packed_moving_scale (bool): If True, hidden_scale_sb (moving) is in packed layout
            [_pmax, n_packed, B] — selected independently of the weight scale packing. Used by the
            fp8 pre-quantized hidden path, whose producer (rmsnorm_mx_prefill) folds 4 H512 tiles
            into one 128-wide scale block.

    Returns:
        None: writes into out_psum_lst in place.

    Notes:
        Packed scale SBUF layout (_pmax, n_packed, I): up to 4 H-tiles are folded into
        a single packed buffer along the partition dim. Tile k occupies 4 contiguous
        partitions starting at offset (k%4)*4 within packed buffer k//4. Addressed via
        base_tensor[slot_part_off:, packed_buf_idx, I_slice] because nl.NkiTensor cannot
        express partition-strided access.
    """
    I = cfg.I
    for i_I512_tile in range(cfg.n_total_I512_tile):
        cur_I512_tile_sz = min(512, I - i_I512_tile * 512)

        # Iterate _q_width number of I128 tiles, each uses 1/_q_width elts of an I512 tile (which may not be 512 for the last tile)
        for i_I_mm_tile in range(_q_width):
            cur_I128_tile_sz = cur_I512_tile_sz // _q_width
            weight_I_offset = i_I512_tile * 512 + i_I_mm_tile * cur_I128_tile_sz

            # Packed scale layout: the H tile is folded into the partition dim. Tile k lives at
            # within-quadrant offset (k%4)*4 in packed buffer k//4; the engine reads its
            # 4-per-quadrant stripe from that offset. Stationary (weight) and moving (hidden) scales
            # are packed independently — the fp8 hidden path packs only the moving operand.
            if is_static_quant:
                # STATIC_MX: weight_scale_tv and hidden_scale_sb are dummy [_pmax, _pmax]
                # all-127 buffers. Pass as 2D views directly (per-tile shape).
                stationary_scale_view = weight_scale_tv.slice(1, 0, cur_I128_tile_sz)
                moving_scale_view = hidden_scale_sb.slice(1, 0, cur_BxS_tile_sz)
            else:
                _packed_buf_idx = i_H512_tile // _PACKED_TILES_PER_BUFFER
                _slot_part_off = (i_H512_tile % _PACKED_TILES_PER_BUFFER) * _q_width
                if is_packed_scale:
                    stationary_scale_view = weight_scale_tv[
                        _slot_part_off:,
                        _packed_buf_idx,
                        weight_I_offset : weight_I_offset + cur_I128_tile_sz,
                    ]
                else:
                    stationary_scale_view = weight_scale_tv.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                        2, weight_I_offset, weight_I_offset + cur_I128_tile_sz
                    )
                if is_packed_moving_scale:
                    moving_scale_view = hidden_scale_sb[
                        _slot_part_off:,
                        _packed_buf_idx,
                        cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz,
                    ]
                else:
                    moving_scale_view = hidden_scale_sb.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                        2, cur_BxS_tile_offset, cur_BxS_tile_offset + cur_BxS_tile_sz
                    )
            nisa.nc_matmul_mx(
                dst=out_psum_lst[i_I512_tile][:cur_I128_tile_sz, i_I_mm_tile, :cur_BxS_tile_sz],
                stationary=weight_qtz_tv.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                    2, weight_I_offset, weight_I_offset + cur_I128_tile_sz
                ),
                moving=hidden_qtz_sb.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                    2, cur_BxS_tile_offset, cur_BxS_tile_offset + cur_BxS_tile_sz
                ),
                stationary_scale=stationary_scale_view,
                moving_scale=moving_scale_view,
            )


def _copy_psum_to_out_sb(
    out_sb: nl.NkiTensor,
    out_psum_lst: list,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
    cur_BxS_tile_offset: int,
    cur_BxS_tile_sz: int,
    use_software_quant_path: bool,
):
    """
    Copy PSUM tiles to out_sb, optionally fusing bias-add on NC0.

    Args:
        out_sb (nl.NkiTensor): Output SBUF buffer to write into.
        out_psum_lst (list): Per-I512-tile PSUM tiles produced by the matmul.
        bias_sb (nl.NkiTensor): Bias view (may be None).
        cfg (ProjConfig): Projection configuration.
        cur_BxS_tile_offset (int): Current BxS tile starting offset within out_sb.
        cur_BxS_tile_sz (int): Current BxS tile size (may be < BxS_tile_sz at the tail).
        use_software_quant_path (bool): When True (STATIC_MX/ROW_MX), skip bias fusion
            and emit a plain copy; bias is applied after dequant.

    Returns:
        None: writes into out_sb in place.
    """
    I = cfg.I
    for i_I512_tile in range(cfg.n_total_I512_tile):
        # Last tile of psum may have less partitions to copy
        cur_I_pdim_sz = min(_pmax, I // _q_width - i_I512_tile * _pmax)

        dst = out_sb[:cur_I_pdim_sz, i_I512_tile, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :]

        if not use_software_quant_path and bias_sb != None:
            # Slice and broadcast bias to match output shape
            bias_tile_view = bias_sb.slice(dim=0, start=0, end=cur_I_pdim_sz)
            if cfg.bias_t_shared_between_gate_up:
                gate_or_up_idx = 0 if cfg.bias_t_shared_base_offset == 0 else 1
                bias_tile_view = bias_tile_view.select(dim=1, index=gate_or_up_idx)
            bias_tile_view = bias_tile_view.slice(dim=1, start=i_I512_tile, end=i_I512_tile + 1)
            bias_tile_view = bias_tile_view.broadcast(dim=1, size=cur_BxS_tile_sz)

            nisa.tensor_tensor(
                dst=dst,
                data1=out_psum_lst[i_I512_tile].ap(
                    [
                        [_q_width * cur_BxS_tile_sz, cur_I_pdim_sz],
                        [1, cur_BxS_tile_sz],
                        [cur_BxS_tile_sz, _q_width],
                    ]
                ),
                data2=bias_tile_view,
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=dst,
                src=out_psum_lst[i_I512_tile].ap(
                    [
                        [_q_width * cur_BxS_tile_sz, cur_I_pdim_sz],
                        [1, cur_BxS_tile_sz],
                        [cur_BxS_tile_sz, _q_width],
                    ]
                ),
                engine=nisa.scalar_engine,
            )


def _psum_dequant_to_out_sb(
    out_sb: nl.NkiTensor,
    out_psum_lst: list,
    bias_2d,
    cfg: ProjConfig,
    cur_BxS_tile_offset: int,
    cur_BxS_tile_sz: int,
    combined,
    activation_op,
):
    """STATIC_MX fused: out_sb = activation_op(psum * combined + bias) — single write.

    Must be called inside the i_BxS_tile loop while psum is still live.
    """
    I = cfg.I
    _act_op = activation_op if activation_op is not None else nl.copy
    # Match _copy_psum_to_out_sb's partition coverage: write only the valid
    # partitions for this I512 tile (cur_I_pdim_sz). The last tile may have <_pmax
    # valid partitions; matmul writes only those, and nisa.activation requires
    # scale/bias/data partition dims to all match dst.
    for i_I512_tile in range(cfg.n_total_I512_tile):
        cur_I_pdim_sz = min(_pmax, I // _q_width - i_I512_tile * _pmax)
        for i_q in range(_q_width):
            col_idx = i_I512_tile * _q_width + i_q
            dst = out_sb[
                :cur_I_pdim_sz,
                i_I512_tile,
                cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz,
                i_q,
            ]
            data = out_psum_lst[i_I512_tile][:cur_I_pdim_sz, i_q, :cur_BxS_tile_sz]
            scale_slice = combined[:cur_I_pdim_sz, :]
            if bias_2d != None:
                nisa.activation(
                    dst=dst,
                    op=_act_op,
                    data=data,
                    scale=scale_slice,
                    bias=bias_2d[:cur_I_pdim_sz, col_idx],
                )
            else:
                nisa.activation(
                    dst=dst,
                    op=_act_op,
                    data=data,
                    scale=scale_slice,
                )


def _post_matmul_dequant(
    out_sb: nl.NkiTensor,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
    w_dequant_scale,
    input_dequant_scale,
    activation_op=None,
):
    """
    Apply STATIC_MX / ROW_MX post-matmul dequant + bias.

    Args:
        out_sb (nl.NkiTensor): Output SBUF buffer to dequantize and bias-add in place.
        bias_sb (nl.NkiTensor): Optional bias view; broadcast across BxS when present.
        cfg (ProjConfig): Projection configuration.
        w_dequant_scale (nl.NkiTensor): Weight dequant scale. [_pmax, 1] for STATIC_MX,
            [_pmax, n_total_I512_tile * _q_width] for ROW_MX.
        input_dequant_scale (nl.NkiTensor): Input dequant scale. [_pmax, 1] for STATIC_MX,
            [_pmax, T_padded, 1] for ROW_MX.

    Returns:
        None: writes into out_sb in place.

    Notes:
        STATIC_MX path pre-combines the two scalar dequant scales and broadcasts.
        ROW_MX path fuses per-row weight x per-token input dequant into a single
        scalar_tensor_tensor and applies bias in a follow-on activation.
    """
    BxS = cfg.BxS

    bias_2d = None
    if bias_sb != None:
        bias_2d = bias_sb.flatten_dims(1, 2)

    if w_dequant_scale.shape[1] == 1:
        # STATIC_MX: both scales are [_pmax, 1], pre-combine and broadcast.
        # When activation_op is provided (e.g. silu), fuse it into the dequant copy:
        #   out = activation_op(out * combined + bias). This eliminates a separate
        #   post-projection nisa.activation pass.
        # Caller may bake input_dequant into w_dequant_scale and pass input_dequant_scale=None; in that case skip the combine.
        combined = (
            pre_combine_dequant_scales(input_dequant_scale, w_dequant_scale)
            if input_dequant_scale is not None
            else w_dequant_scale
        )
        _act_op = activation_op if activation_op is not None else nl.copy
        if bias_2d != None:
            # Fuse scale + bias: out = activation_op(out * combined + bias)
            for i_tile in nl.affine_range(cfg.n_total_I512_tile):
                for i_q in nl.affine_range(_q_width):
                    col_idx = i_tile * _q_width + i_q
                    nisa.activation(
                        dst=out_sb[:, i_tile, :, i_q],
                        op=_act_op,
                        data=out_sb[:, i_tile, :, i_q],
                        scale=combined,
                        bias=bias_2d[:, col_idx],
                    )
        else:
            nisa.activation(
                dst=out_sb,
                op=_act_op,
                data=out_sb,
                scale=combined,
            )
    else:
        # ROW_MX: per-row weight scale × per-token input scale fused into one scalar_tensor_tensor;
        # optional bias added in a follow-on activation.
        for i_tile in nl.affine_range(cfg.n_total_I512_tile):
            for i_q in nl.affine_range(_q_width):
                i_col = i_tile * _q_width + i_q
                nisa.scalar_tensor_tensor(
                    dst=out_sb[:, i_tile, :, i_q],
                    data=out_sb[:, i_tile, :, i_q],
                    op0=nl.multiply,
                    operand0=w_dequant_scale[:, i_col : i_col + 1],
                    op1=nl.multiply,
                    operand1=input_dequant_scale[:, :BxS, 0],
                )
                if bias_2d != None:
                    nisa.activation(
                        dst=out_sb[:, i_tile, :, i_q],
                        op=nl.copy,
                        data=out_sb[:, i_tile, :, i_q],
                        bias=bias_2d[:, i_col],
                    )


def gate_up_projection_mx_tp(
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scale_sb: nl.NkiTensor,
    weight_qtz: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
    w_dequant_scale: Optional[nl.NkiTensor] = None,
    input_dequant_scale: Optional[nl.NkiTensor] = None,
    sbm=None,
    psum_bank_offset: int = 0,
    name_prefix: Optional[str] = None,
    out_sb: Optional[nl.NkiTensor] = None,
    dst_weight_sb: Optional[nl.NkiTensor] = None,
    skip_dma=None,
    is_packed_scale: bool = False,
    is_packed_moving_scale: bool = False,
    activation_op=None,
    is_static_quant: bool = False,
) -> nl.NkiTensor:
    """
    Gate/Up projection (TP variant: output transposed for the subsequent down projection).

    Math (Neuron matmul): hidden (moving) [H, BxS] @ weight (stationary) [H, I] → [I, BxS].

    Output is in SBUF with swizzle layout for subsequent quantization:
    [sb_p, I // sb_p // _q_width, BxS, _q_width].

    Three weight-loading modes:
        1. weight_qtz already in SBUF: use directly.
        2. weight_qtz in HBM, dst_weight_sb is None: bulk DMA into a freshly-allocated SBUF buffer.
        3. weight_qtz in HBM, dst_weight_sb provided: stream the weight in H-chunks of _H_CHUNK_SZ
           into dst_weight_sb. Each chunk's DMA overlaps with the previous chunk's matmul (better
           DMA/compute overlap, useful for large H tiles). skip_dma controls oob_mode.

    weight_scale is required to be in SBUF.

    H has a tile size of 512 because that is mx_matmul's contraction size (_pmax * _q_width).

    Args:
        hidden_qtz_sb (nl.NkiTensor): mxfp8_x4[_pmax, n_H512_tile_sharded, BxS] @ SB.
            Dim H is shuffled on _pmax.
        hidden_scale_sb (nl.NkiTensor): uint8[_pmax, n_H512_tile_sharded, BxS] @ SB.
            Dim H is shuffled on _pmax. Pdim has holes.
        weight_qtz (nl.NkiTensor): One of:
            - mxfp_x4[_pmax, n_H512_tile_sharded, I] @ SB, or
            - mxfp_x4[_pmax, n_H512_tile, I] @ HBM (full unsharded H; this fn slices the prg_id shard), or
            - mxfp_x4[_pmax, n_H512_tile_sharded, I] @ HBM when dst_weight_sb is provided (caller pre-sliced).
        weight_scale (nl.NkiTensor): uint8[_pmax, n_H512_tile_sharded, I] @ SB.
        bias_sb (nl.NkiTensor, optional): bf16[_pmax, n_I512_tile, _q_width] @ SB.
            For already-sliced (gate-only or up-only) bias, cfg.bias_t_shared_base_offset must be 0.
            For combined gate+up, cfg.bias_t_shared_base_offset specifies the up-projection offset.
        cfg (ProjConfig): Projection configuration (H, I, BxS, sharding info, layout flags).
        w_dequant_scale (nl.NkiTensor, optional): Weight dequant scale in SBUF.
            [_pmax, 1] for STATIC_MX, [_pmax, n_I512_tile * _q_width] for ROW_MX, None for MX.
            Not supported on the streaming path.
        input_dequant_scale (nl.NkiTensor, optional): Input dequant scale in SBUF.
            [_pmax, 1] for STATIC_MX, [_pmax, T_padded, 1] for ROW_MX, None for MX.
        sbm (SbufManager, optional): SBUF allocator. Required when out_sb is None or
            non-SBUF weight_qtz is provided.
        psum_bank_offset (int): Starting PSUM bank index for output tile rotation.
        name_prefix (str, optional): Prefix for SBUF buffer names; useful for debugging.
        out_sb (nl.NkiTensor, optional): Pre-allocated output buffer. If None, this fn allocates one.
        dst_weight_sb (nl.NkiTensor, optional): SBUF buffer (_pmax, n_H512_tile_sharded, I) to
            stream weights into. Enables H-chunked streaming. When provided, weight_qtz must
            be in HBM and already prg_id-sliced.
        skip_dma (SkipMode, optional): OOB handling for the streaming path. Required if
            dst_weight_sb is set.
        is_packed_scale (bool): True when caller pre-packed weight_scale (stationary) into the
            compressed layout (_pmax, n_packed, I) where multiple H-tiles are folded along the
            partition dim. Default False = standard (_pmax, n_H512_tile_sharded, I) layout.
        is_packed_moving_scale (bool): True when hidden_scale_sb (moving) is in the packed layout
            (_pmax, n_packed, BxS) — 4 H512 tiles folded into one 128-wide block, tile k at
            within-quadrant partition offset (k%4)*4 in pack k//4. Selected independently of
            is_packed_scale; the fp8 pre-quantized hidden path packs only the moving operand.
            Default False = standard (_pmax, n_H512_tile_sharded, BxS) layout.

    Returns:
        nl.NkiTensor: bf16[_pmax, ceil(I / 512), BxS, _q_width] @ SB.
    """
    H0, I, BxS = cfg.H0, cfg.I, cfg.BxS

    BxS_tile_sz = min(BxS, _psum_fmax * 2 // _q_width)  # double psum elts because out is in bf16
    n_BxS_tile = div_ceil(BxS, BxS_tile_sz)

    use_streaming = dst_weight_sb != None
    use_software_quant_path = w_dequant_scale != None

    weight_in_sbuf = weight_qtz.buffer == nl.sbuf

    kernel_assert(
        (weight_scale.buffer == nl.sbuf),
        f"Expect weight_scale in SBUF, got buffer={weight_scale.buffer}",
    )
    if not is_packed_scale and not is_static_quant:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_H512_tile_sharded, I),
            f"Expect weight_scale shape (128, {cfg.n_H512_tile_sharded}, {I}), got {weight_scale.shape}",
        )

    if use_streaming:
        kernel_assert(not weight_in_sbuf, "Streaming weight load requires weight_qtz in HBM")
        kernel_assert(skip_dma != None, "skip_dma is required when dst_weight_sb is provided")
        # STATIC_MX dequant on the streaming path is supported: per-chunk matmul still writes
        # to PSUM and `_post_matmul_dequant` applies dequant afterwards.

    # ── Resolve weight tensor (SBUF, bulk-DMA, or streamed-into-dst_weight_sb) ──
    if weight_in_sbuf:
        kernel_assert(
            weight_qtz.shape == (_pmax, cfg.n_H512_tile_sharded, I),
            f"Expect weight_qtz in SBUF to be in shape ({H0}, {cfg.n_H512_tile_sharded}, {I}), got {weight_qtz.shape}",
        )
        weight_qtz_tv = weight_qtz
    elif use_streaming:
        # DMA happens per-chunk inside the BxS loop below.
        weight_qtz_tv = dst_weight_sb
    else:
        weight_qtz_tv = _bulk_load_weight_from_hbm(weight_qtz, cfg, sbm, name_prefix)

    if cfg.dbg_hidden:
        return hidden_qtz_sb, hidden_scale_sb

    if cfg.dbg_weight:
        return weight_qtz_tv, weight_scale

    # ── Allocate output buffer ──
    out_sb = (
        out_sb
        if out_sb != None
        else _sbm_alloc(
            sbm,
            (_pmax, cfg.n_total_I512_tile, BxS, _q_width),
            dtype=nl.bfloat16,
            name=f"{name_prefix}_out_sb" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
    )

    # Zero-initialize the last I512 tile if it has fewer than _pmax valid partitions.
    # Prevents uninitialized SBUF data (potentially NaN) from corrupting downstream
    # operations like row_quantization's cross-partition reduce. Full-tile memset
    # (DMA engine) avoids contention with the matmul output copies that follow.
    if cfg.r_I512_tile > 0 and cfg.zero_unused_partitions:
        nisa.memset(dst=out_sb[:, cfg.n_total_I512_tile - 1, :, :], value=0.0)

    n_H_chunks = div_ceil(cfg.n_H512_tile_sharded, _H_CHUNK_N_H512) if use_streaming else 0

    # ── STATIC_MX fused-dequant prep ──
    # When w_dequant_scale is [_pmax, 1] (STATIC_MX), pre-compute combined scale and
    # bias view once.
    _fuse_static_dequant = use_software_quant_path and w_dequant_scale.shape[1] == 1
    fused_combined = None
    fused_bias_2d = None
    if _fuse_static_dequant:
        fused_combined = (
            pre_combine_dequant_scales(input_dequant_scale, w_dequant_scale)
            if input_dequant_scale is not None
            else w_dequant_scale
        )
        if bias_sb != None:
            fused_bias_2d = bias_sb.flatten_dims(1, 2)

    # ── Matmul loop ──
    for i_BxS_tile in range(n_BxS_tile):
        cur_BxS_tile_offset = i_BxS_tile * BxS_tile_sz
        cur_BxS_tile_sz = min(BxS_tile_sz, BxS - cur_BxS_tile_offset)

        # Allocate one psum tile per I512 tile
        out_psum_lst = []
        for i_I512_tile in range(cfg.n_total_I512_tile):
            psum_bank_id = (i_I512_tile + psum_bank_offset) % _psum_bmax
            out_psum_lst.append(
                _psum_alloc((_pmax, _q_width, cur_BxS_tile_sz), nl.bfloat16, sbm, psum_bank_id * PSUM_BANK_SIZE)
            )

        if use_streaming:
            # Stream weight in H-chunks: DMA load chunk, then matmul its H512 tiles.
            # overlap chunk i's DMA with chunk i-1's matmul.
            for i_chunk in nl.affine_range(n_H_chunks):
                h_start = i_chunk * _H_CHUNK_N_H512
                cur_chunk_n_H512 = min(_H_CHUNK_N_H512, cfg.n_H512_tile_sharded - h_start)

                nisa.dma_copy(
                    dst=dst_weight_sb.slice(1, h_start, h_start + cur_chunk_n_H512).slice(2, 0, I),
                    src=weight_qtz.slice(1, h_start, h_start + cur_chunk_n_H512).slice(2, 0, I),
                    oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                    dge_mode=dge_mode.hwdge,
                )

                for i_h_in_chunk in range(cur_chunk_n_H512):
                    _matmul_one_H512_tile(
                        i_H512_tile=h_start + i_h_in_chunk,
                        hidden_qtz_sb=hidden_qtz_sb,
                        hidden_scale_sb=hidden_scale_sb,
                        weight_qtz_tv=weight_qtz_tv,
                        weight_scale_tv=weight_scale,
                        out_psum_lst=out_psum_lst,
                        cfg=cfg,
                        cur_BxS_tile_offset=cur_BxS_tile_offset,
                        cur_BxS_tile_sz=cur_BxS_tile_sz,
                        is_packed_scale=is_packed_scale,
                        is_packed_moving_scale=is_packed_moving_scale,
                        is_static_quant=is_static_quant,
                    )
        else:
            for i_H512_tile in range(cfg.n_H512_tile_sharded):
                _matmul_one_H512_tile(
                    i_H512_tile=i_H512_tile,
                    hidden_qtz_sb=hidden_qtz_sb,
                    hidden_scale_sb=hidden_scale_sb,
                    weight_qtz_tv=weight_qtz_tv,
                    weight_scale_tv=weight_scale,
                    out_psum_lst=out_psum_lst,
                    cfg=cfg,
                    cur_BxS_tile_offset=cur_BxS_tile_offset,
                    cur_BxS_tile_sz=cur_BxS_tile_sz,
                    is_packed_scale=is_packed_scale,
                    is_packed_moving_scale=is_packed_moving_scale,
                    is_static_quant=is_static_quant,
                )

        if _fuse_static_dequant:
            # STATIC_MX: fuse psum→out_sb dequant + bias + activation in one write.
            _psum_dequant_to_out_sb(
                out_sb=out_sb,
                out_psum_lst=out_psum_lst,
                bias_2d=fused_bias_2d,
                cfg=cfg,
                cur_BxS_tile_offset=cur_BxS_tile_offset,
                cur_BxS_tile_sz=cur_BxS_tile_sz,
                combined=fused_combined,
                activation_op=activation_op,
            )
        else:
            # MX (no SW dequant): bias-fused tensor_tensor.
            # ROW_MX: plain tensor_copy here, dequant applied after the i_BxS_tile loop.
            _copy_psum_to_out_sb(
                out_sb=out_sb,
                out_psum_lst=out_psum_lst,
                bias_sb=bias_sb,
                cfg=cfg,
                cur_BxS_tile_offset=cur_BxS_tile_offset,
                cur_BxS_tile_sz=cur_BxS_tile_sz,
                use_software_quant_path=use_software_quant_path,
            )

    # Post-matmul dequant (ROW_MX only — STATIC_MX is fused inside the loop above).
    if use_software_quant_path and not _fuse_static_dequant:
        _post_matmul_dequant(out_sb, bias_sb, cfg, w_dequant_scale, input_dequant_scale, activation_op=activation_op)

    return out_sb
