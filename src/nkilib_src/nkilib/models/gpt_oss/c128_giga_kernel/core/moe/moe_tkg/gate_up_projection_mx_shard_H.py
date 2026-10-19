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
Gate/Up projection sub-kernels with LNC sharding on H (hidden) dimension.

LNC Sharding Strategy: H dimension
- When LNC=2, weights are sharded on H dimension (contraction dimension)
- LNC reduction via sendrecv after projection
- Used by: selective-expert MoE algorithm, dense MLP (but algorithm-independent)

These sub-kernels can be used by any algorithm that requires H-sharded gate/up projection.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from ...quantization.fp8_quantize import pre_combine_dequant_scales
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import (
    PSUM_BANK_SIZE,
    _psum_alloc,
    _sbm_alloc,
    div_ceil,
    get_nl_act_fn_from_type,
)
from .mlp_parameters import MLPParameters
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes
from .projection_mx_constants import (
    SBUF_QUADRANT_SIZE,
    ProjConfig,
    _pmax,
    _psum_bmax,
    _psum_fmax,
    _q_height,
    _q_width,
)


def gate_up_projection_mx_tp_shard_H(
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scale_sb: nl.NkiTensor,
    weight_qtz: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
    w_dequant_scale=None,
    input_dequant_scale=None,
    sbm=None,
    psum_bank_offset: int = 0,
    name_prefix: str = None,
    out_sb=None,
) -> nl.NkiTensor:
    """
    Performs the Gate/Up projection. This is the TP version of the projection, i.e. the output will be in transposed
    for down projection. Math (Neuron matmul):
        hidden (moving) [H, BxS] @ weight (stationary) [H, I] → [I, BxS].

    Further, the output will be in SBUF with swizzle layout for subsequent quantization, thus the output layout will
    be: [sb_p, I // sb_p // _q_width, BxS, _q_width].

    NOTE: In the shapes below, H has a tile size of 512 because it's the contraction size of mx_matmul (_pmax * _q_width).

    :param hidden_qtz_sb: mxfp8_x4[_pmax, n_H512_tile_sharded, BxS] @ SB. Dim H is shuffled on _pmax.
    :param hidden_scale_sb: uint8[_pmax, n_H512_tile_sharded, BxS] @ SB. Dim H is shuffled on _pmax. NOTE: pdim has holes
    :param weight_qtz:
        - mxfp_x4[_pmax, n_H512_tile_sharded, I] @ SB, or
        - mxfp_x4[_pmax, n_H512_tile, I] @ HBM.
    :param weight_scale:
        - uint8[_pmax, n_H512_tile_sharded, I] @ SB, or
        - uint8[_pmax // _q_height, n_H512_tile, I] @ HBM.
    :param bias_sb [OPTIONAL]: nl.NkiTensor of bf16[_pmax, n_I512_tile, _q_width] @ SB.
        - For already-sliced (gate or up only): cfg.bias_t_shared_base_offset should be 0
        - For combined gate+up: cfg.bias_t_shared_base_offset specifies offset for up projection
    :param w_dequant_scale [OPTIONAL]: Weight dequant scale in SBUF.
        [_pmax, 1] for STATIC_MX, [_pmax, n_I512_tile * _q_width] for ROW_MX, None for MX.
    :param input_dequant_scale [OPTIONAL]: Input dequant scale in SBUF.
        [_pmax, 1] for STATIC_MX, [_pmax, T_padded, 1] for ROW_MX, None for MX.
    :return: bf16[_pmax, ceil(I / 512), BxS, _q_width] @ SB.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H0, H1, H1_sharded, I, BxS = cfg.H0, cfg.H1, cfg.H1_sharded, cfg.I, cfg.BxS

    BxS_tile_sz = min(BxS, _psum_fmax * 2 // _q_width)  # double psum elts because out is in bf16
    n_BxS_tile = div_ceil(BxS, BxS_tile_sz)

    # Either load weight_qtz from HBM to sbuf or directly use it if it is already in SBUF
    if weight_qtz.buffer == nl.sbuf:
        kernel_assert(
            weight_qtz.shape == (_pmax, cfg.n_H512_tile_sharded, I),
            f"Expect weight_qtz in SBUF to be in shape ({H0}, {cfg.n_H512_tile_sharded}, {I}), got {weight_qtz.shape}",
        )
        weight_qtz_tv = weight_qtz
    else:
        kernel_assert(
            weight_qtz.shape == (_pmax, cfg.n_H512_tile, I),
            f"Expect weight_qtz in HBM to be in shape (128, {cfg.n_H512_tile}, {I}), got {weight_qtz.shape}",
        )
        # Load weight into [H0, cfg.n_H512_tile, I] NOTE: this is pre-quantized and each elt is mxfp_x4 (packed H)
        weight_qtz_sb = _sbm_alloc(
            sbm,
            (H0, cfg.n_H512_tile_sharded, I),
            dtype=weight_qtz.dtype,
            name=f"{name_prefix}_weight_qtz_sb" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.dma_copy(
            dst=weight_qtz_sb,
            src=weight_qtz[:, prg_id * cfg.n_H512_tile_sharded : (prg_id + 1) * cfg.n_H512_tile_sharded, :],
            dge_mode=nisa.dge_mode.hwdge,
        )
        weight_qtz_tv = weight_qtz_sb

    if cfg.dbg_hidden:
        return hidden_qtz_sb, hidden_scale_sb

    if weight_scale.buffer == nl.sbuf:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_H512_tile_sharded, I),
            f"Expect weight_scale in SBUF to have the shape of (128, {cfg.n_H512_tile_sharded}, {I}), got {weight_scale.shape}",
        )
        weight_scale_tv = weight_scale
    else:
        # Load weight scale into [H0, n_H512_tile, I] NOTE: we have 1 scale per 8(p)x4(f) tile, but still span across full pdim with gaps
        kernel_assert(
            weight_scale.shape == (_pmax // _q_height, cfg.n_H512_tile, I),
            f"Expect weight_scale in SBUF to have the shape of (16, {cfg.n_H512_tile}, {I}), got {weight_scale.shape}",
        )
        weight_scale_sb = _sbm_alloc(
            sbm,
            (H0, cfg.n_H512_tile_sharded, I),
            dtype=nl.uint8,
            name=f"{name_prefix}_weight_scale_sb" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
        # Load 4 partitions of scales for every quadrant
        n_quadrants_needed = H0 // SBUF_QUADRANT_SIZE
        for i_quad in range(n_quadrants_needed):
            nisa.dma_copy(
                src=weight_scale[
                    i_quad * 4 : (i_quad + 1) * 4,
                    prg_id * cfg.n_H512_tile_sharded : (prg_id + 1) * cfg.n_H512_tile_sharded,
                    :,
                ],
                dst=weight_scale_sb[i_quad * SBUF_QUADRANT_SIZE : i_quad * SBUF_QUADRANT_SIZE + 4, :, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
        weight_scale_tv = weight_scale_sb

    if cfg.dbg_weight:
        return weight_qtz_tv, weight_scale_tv

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

    """
    Zero-initialize the last I512 tile if it has fewer than _pmax valid partitions.

    This prevents uninitialized SBUF data (potentially NaN) from corrupting
    downstream operations like row_quantization's cross-partition reduce.
    Full-tile memset (DMA engine) is preferred over partial-partition tensor_scalar
    (ACT engine) to avoid contention with the matmul output copies that follow.
    """
    if cfg.r_I512_tile > 0 and cfg.zero_unused_partitions:
        nisa.memset(dst=out_sb[:, cfg.n_total_I512_tile - 1, :, :], value=0.0)

    # Loop over BxS tiles (each of size 256)
    for i_BxS_tile in range(n_BxS_tile):
        # For the last iter, we may have less than BxS_tile_sz to work with
        cur_BxS_tile_offset = i_BxS_tile * BxS_tile_sz
        cur_BxS_tile_sz = min(BxS_tile_sz, BxS - cur_BxS_tile_offset)

        # Allocate and init output psum and sbuf. Note that there are cfg.n_total_I512_tile instances of out_psum
        out_psum_lst = []
        for i_I512_tile in range(cfg.n_total_I512_tile):
            psum_bank_id = (i_I512_tile + psum_bank_offset) % _psum_bmax
            out_psum_lst.append(
                _psum_alloc((_pmax, _q_width, cur_BxS_tile_sz), nl.bfloat16, sbm, psum_bank_id * PSUM_BANK_SIZE)
            )

        # Matmul compute, tiles on H, then I, then _q_width (4)
        for i_H512_tile in range(cfg.n_H512_tile_sharded):
            for i_I512_tile in range(cfg.n_total_I512_tile):
                cur_I512_tile_sz = min(512, I - i_I512_tile * 512)

                # Iterate _q_width number of I128 tiles, each uses 1/4 elts of an I512 tile (which may not be 512 for the last tile)
                for i_I_mm_tile in range(_q_width):
                    cur_I128_tile_sz = cur_I512_tile_sz // 4
                    weight_I_offset = i_I512_tile * 512 + i_I_mm_tile * cur_I128_tile_sz

                    nisa.nc_matmul_mx(
                        dst=out_psum_lst[i_I512_tile][:cur_I128_tile_sz, i_I_mm_tile, :cur_BxS_tile_sz],
                        stationary=weight_qtz_tv.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                            2, weight_I_offset, weight_I_offset + cur_I128_tile_sz
                        ),
                        moving=hidden_qtz_sb.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                            2, cur_BxS_tile_offset, cur_BxS_tile_offset + cur_BxS_tile_sz
                        ),
                        stationary_scale=weight_scale_tv.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                            2, weight_I_offset, weight_I_offset + cur_I128_tile_sz
                        ),
                        moving_scale=hidden_scale_sb.slice(1, i_H512_tile, i_H512_tile + 1).slice(
                            2, cur_BxS_tile_offset, cur_BxS_tile_offset + cur_BxS_tile_sz
                        ),
                    )

        # Copy out psum to output sbuf NOTE: final tile may not use all partitions
        for i_I512_tile in range(cfg.n_total_I512_tile):
            # Last tile of psum may have less partitions to copy
            cur_I_pdim_sz = min(_pmax, I // _q_width - i_I512_tile * _pmax)

            """
            Copy output while adding bias if needed.
            
            Only NC0 needs to add bias because we shard on contraction dimension (H).
            out_sb shape: [_pmax, cfg.n_total_I512_tile, BxS, _q_width]
            out_psum shape: [_pmax, _q_width, BxS_tile_sz] (for each item in out_psum_lst)
            """
            if w_dequant_scale == None:
                # ── Standard path (MX with real scales, or no dequant needed) ──
                if (bias_sb != None) and (prg_id == 0):
                    # Slice and broadcast bias to match output shape
                    bias_tile_view = bias_sb.slice(dim=0, start=0, end=cur_I_pdim_sz)
                    if cfg.bias_t_shared_between_gate_up:
                        gate_or_up_idx = 0 if cfg.bias_t_shared_base_offset == 0 else 1
                        bias_tile_view = bias_tile_view.select(dim=1, index=gate_or_up_idx)
                    bias_tile_view = bias_tile_view.slice(dim=1, start=i_I512_tile, end=i_I512_tile + 1)
                    bias_tile_view = bias_tile_view.broadcast(dim=1, size=cur_BxS_tile_sz)

                    nisa.tensor_tensor(
                        dst=out_sb[
                            :cur_I_pdim_sz, i_I512_tile, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                        ],
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
                        dst=out_sb[
                            :cur_I_pdim_sz, i_I512_tile, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                        ],
                        src=out_psum_lst[i_I512_tile].ap(
                            [
                                [_q_width * cur_BxS_tile_sz, cur_I_pdim_sz],
                                [1, cur_BxS_tile_sz],
                                [cur_BxS_tile_sz, _q_width],
                            ]
                        ),
                    )
            else:
                # ── Software quant path (STATIC_MX / ROW_MX): copy psum to sbuf first ──
                nisa.tensor_copy(
                    dst=out_sb[
                        :cur_I_pdim_sz, i_I512_tile, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                    ],
                    src=out_psum_lst[i_I512_tile].ap(
                        [[_q_width * cur_BxS_tile_sz, cur_I_pdim_sz], [1, cur_BxS_tile_sz], [cur_BxS_tile_sz, _q_width]]
                    ),
                )

    # Receive projection output from the other NC when LNC > 1
    # IMPORTANT: LNC reduce must happen BEFORE dequant+bias to avoid double-counting bias
    if n_prgs > 1:
        recv = _sbm_alloc(
            sbm,
            out_sb.shape,
            dtype=out_sb.dtype,
            name=f"{name_prefix}_recv" if name_prefix else None,
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.sendrecv(src=out_sb, dst=recv, send_to_rank=(1 - prg_id), recv_from_rank=(1 - prg_id), pipe_id=0)
        nisa.tensor_tensor(dst=out_sb, data1=out_sb, data2=recv, op=nl.add)

    # ── Post-matmul dequant (STATIC_MX / ROW_MX): apply w_dequant_scale, input_dequant_scale, bias ──
    # Applied AFTER LNC reduce so bias is added once and scale is applied to the full sum.
    if w_dequant_scale != None:
        # Pre-compute bias in 2D layout for per-slice access (shared by both paths)
        bias_2d = None
        if bias_sb != None:
            bias_base = bias_sb if isinstance(bias_sb, nl.NkiTensor) else bias_sb
            bias_2d = bias_base.reshape((_pmax, cfg.n_total_I512_tile * _q_width))

        if w_dequant_scale.shape[1] == 1:
            # ── STATIC_MX: both scales are [_pmax, 1], pre-combine and broadcast ──
            combined = pre_combine_dequant_scales(input_dequant_scale, w_dequant_scale)
            if bias_2d != None:
                # Fuse scale + bias: out = out * combined + bias
                for i_tile in nl.affine_range(cfg.n_total_I512_tile):
                    for i_q in nl.affine_range(_q_width):
                        col_idx = i_tile * _q_width + i_q
                        nisa.activation(
                            dst=out_sb[:, i_tile, :, i_q],
                            op=nl.copy,
                            data=out_sb[:, i_tile, :, i_q],
                            scale=combined,
                            bias=bias_2d[:, col_idx],
                        )
            else:
                nisa.activation(
                    dst=out_sb,
                    op=nl.copy,
                    data=out_sb,
                    scale=combined,
                )
        else:
            # ── ROW_MX: per-row weight scale, then per-token input scale (+ optional bias) ──
            # Fuse weight dequant × input dequant into a single scalar_tensor_tensor:
            #   dst = data * w_dequant_scale[col] * input_dequant_scale[token]
            # This halves the dequant instruction count vs two separate loops.
            if bias_2d != None:
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

                        nisa.activation(
                            dst=out_sb[:, i_tile, :, i_q],
                            op=nl.copy,
                            data=out_sb[:, i_tile, :, i_q],
                            bias=bias_2d[:, i_col],
                        )
            else:
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

    return out_sb


def _lnc_reduce_proj_out(cur_nc_proj_out: nl.NkiTensor, shard_id: int):
    """In-place LNC2 reduction of projection output."""
    # SendRecv
    proj_out_recv = nl.ndarray(cur_nc_proj_out.shape, dtype=cur_nc_proj_out.dtype, buffer=nl.sbuf)
    nisa.sendrecv(
        src=cur_nc_proj_out, dst=proj_out_recv, send_to_rank=(1 - shard_id), recv_from_rank=(1 - shard_id), pipe_id=0
    )

    # Reduction, because each NC handled half of contraction (H)
    nisa.tensor_tensor(dst=cur_nc_proj_out, data1=cur_nc_proj_out, data2=proj_out_recv, op=nl.add)


def _alloc_gate_up_weights_sb(
    gate_up_weights: nl.NkiTensor,
    n_H512_tile_sharded: int,
    I: int,
) -> list:
    """
    Allocate double-buffered SBUF buffers for fused gate/up weights.

    :param gate_up_weights: mxfp4_x4[_pmax, 2, n_H512_tiles, I] @ HBM.
    :param n_H512_tile_sharded: Number of H512 tiles after sharding.
    :param I: Intermediate dimension size.
    :return: List (size=2) of SBUF tensors, each [_pmax, 2, n_H512_tile_sharded, I] @ SB (uninitialized).
    """
    base_dtype = gate_up_weights.dtype
    weights_n_buff = []
    for _ in range(2):
        weights_n_buff.append(nl.ndarray((_pmax, 2, n_H512_tile_sharded, I), dtype=base_dtype, buffer=nl.sbuf))
    return weights_n_buff


def _load_gate_up_weights(
    weight_sb: nl.NkiTensor,
    gate_up_weights: nl.NkiTensor,
    shard_id: int,
    n_H512_tile_sharded: int,
    gate_up_weights_E_offset: Optional[nl.NkiTensor],
) -> nl.NkiTensor:
    """
    Load fused gate/up weights from HBM into a pre-allocated SBUF buffer.

    :param weight_sb: Pre-allocated SBUF buffer.
    :param gate_up_weights: mxfp4_x4[_pmax, 2, n_H512_tiles, I] @ HBM (or [E, ...] when E_offset provided).
    :param shard_id: Shard index for H-dimension sharding.
    :param n_H512_tile_sharded: Number of H512 tiles after sharding.
    :param gate_up_weights_E_offset: int32[1, 1] @ SB. When provided, gate_up_weights has a leading E dim.
    :return: weight_sb viewed as gate_up_weights.dtype.
    """
    if gate_up_weights_E_offset == None:
        nisa.dma_copy(
            dst=weight_sb,
            src=gate_up_weights[:, :, shard_id : (shard_id + 1) * n_H512_tile_sharded, :],
            dge_mode=nisa.dge_mode.swdge,
        )
    else:
        gate_up_weights_view = gate_up_weights.select(dim=0, index=gate_up_weights_E_offset).slice(
            dim=2, start=shard_id * n_H512_tile_sharded, end=(shard_id + 1) * n_H512_tile_sharded
        )
        nisa.dma_copy(dst=weight_sb, src=gate_up_weights_view, dge_mode=nisa.dge_mode.hwdge)
    return weight_sb.view(gate_up_weights.dtype)


def _alloc_gate_up_scales_sb(n_H512_tile_sharded: int, I: int) -> list:
    """
    Allocate double-buffered SBUF buffers for fused gate/up weight scales.

    :param n_H512_tile_sharded: Number of H512 tiles after sharding.
    :param I: Intermediate dimension size.
    :return: List (size=2) of SBUF tensors, each [_pmax, 2, n_H512_tile_sharded, I] @ SB (uninitialized).
    """
    scales_n_buff = []
    for i in range(2):
        scales_n_buff.append(
            nl.ndarray(
                (_pmax, 2, n_H512_tile_sharded, I), dtype=nl.uint8, buffer=nl.sbuf, name=f'gate_up_w_scale_sb_{i}'
            )
        )
    return scales_n_buff


def _load_gate_up_scales(
    gate_up_scale: nl.NkiTensor,
    gate_up_scale_sb: nl.NkiTensor,
    p_idx_vector: nl.NkiTensor,
    shard_id: int,
    n_H512_tiles: int,
    n_H512_tile_sharded: int,
    I: int,
):
    """
    Load fused gate/up weight scales into pre-allocated SBUF via vector DGE.

    :param gate_up_scale: uint8[E, _pmax // _q_height, 2, n_H512_tiles, I] @ HBM.
    :param gate_up_scale_sb: uint8[_pmax, 2, n_H512_tile_sharded, I] @ SB (pre-allocated, pre-zeroed).
    :param p_idx_vector: float32[_pmax, ...] @ SB. Prepared expert index for vector DGE.
    :param shard_id: Shard index for H-dimension sharding.
    :param n_H512_tiles: Total number of H512 tiles (unsharded).
    :param n_H512_tile_sharded: Number of H512 tiles after sharding.
    :param I: Intermediate dimension size.
    """
    scale_shape = gate_up_scale.shape
    gup_scale_view = gate_up_scale.reshape(
        (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
    )  # [E * _pmax//_q_height, 2, n_H512_tiles, I]

    token_indices_on_p = nl.ndarray(p_idx_vector.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=token_indices_on_p, src=p_idx_vector)
    nisa.dma_copy(
        dst=gate_up_scale_sb,
        src=gup_scale_view.ap(
            pattern=[
                [2 * n_H512_tiles * I, _pmax],
                [n_H512_tiles * I, 2],
                [I, n_H512_tile_sharded],
                [1, I],
            ],
            offset=(shard_id * n_H512_tile_sharded) * I,
            vector_offset=token_indices_on_p,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip,
    )


def _alloc_gate_up_bias_sb(
    gate_up_bias: Optional[nl.NkiTensor],
    n_I512_tile: int,
) -> Optional[list]:
    """
    Allocate double-buffered SBUF buffers for fused gate/up bias.

    :param gate_up_bias: bias tensor @ HBM, or None if no bias.
    :param n_I512_tile: Number of I512 tiles = ceil(I / 512).
    :return: List (size=2) of SBUF tensors, each [_pmax, 2, n_I512_tile, _q_width] @ SB (uninitialized), or None.
    """
    if gate_up_bias == None:
        return None
    bias_n_buff = []
    for _ in range(2):
        bias_n_buff.append(nl.ndarray((_pmax, 2, n_I512_tile, _q_width), dtype=gate_up_bias.dtype, buffer=nl.sbuf))
    return bias_n_buff


def _load_gate_up_bias(
    bias_sb: nl.NkiTensor,
    gate_up_bias: nl.NkiTensor,
    n_I512_tile: int,
    I: int,
    gate_up_weights_E_offset: Optional[nl.NkiTensor],
    gate_up_bias_E_offset: Optional[nl.NkiTensor],
) -> nl.NkiTensor:
    """
    Load fused gate/up bias from HBM into a pre-allocated SBUF buffer, with zero-padding when I < 512.

    :param bias_sb: Pre-allocated SBUF buffer from _alloc_gate_up_bias_sb.
    :param gate_up_bias: bf16[I_p, 2, ceil(I/512), 4] @ HBM (or with leading E dim when E_offset provided).
    :param n_I512_tile: Number of I512 tiles = ceil(I / 512).
    :param I: Intermediate dimension size.
    :param gate_up_weights_E_offset: int32[1, 1] @ SB. Controls DGE mode selection.
    :param gate_up_bias_E_offset: int32[1, 1] @ SB. Scalar offset for expert-indexed bias load.
    :return: bias_sb.
    """
    if I < 512:  # when I<512, gate/up bias HBM is not padded so pad it here
        nisa.memset(dst=bias_sb[:, :, 0, :], value=0.0)
        if gate_up_weights_E_offset == None:
            nisa.dma_copy(dst=bias_sb[: I // 4, :, :, :], src=gate_up_bias, dge_mode=nisa.dge_mode.hwdge)
        else:
            nisa.dma_copy(
                dst=bias_sb[: I // 4, :, :, :],
                src=gate_up_bias.ap(
                    pattern=[
                        [2 * n_I512_tile * _q_width, I // 4],
                        [n_I512_tile * _q_width, 2],
                        [_q_width, n_I512_tile],
                        [1, _q_width],
                    ],
                    offset=0,
                    scalar_offset=gate_up_bias_E_offset,
                    indirect_dim=0,
                ),
            )
    else:
        if gate_up_weights_E_offset == None:
            nisa.dma_copy(dst=bias_sb, src=gate_up_bias, dge_mode=nisa.dge_mode.hwdge)
        else:
            nisa.dma_copy(
                dst=bias_sb,
                src=gate_up_bias.ap(
                    pattern=[
                        [2 * n_I512_tile * _q_width, _pmax],
                        [n_I512_tile * _q_width, 2],
                        [_q_width, n_I512_tile],
                        [1, _q_width],
                    ],
                    offset=0,
                    scalar_offset=gate_up_bias_E_offset,
                    indirect_dim=0,
                ),
            )
    return bias_sb


def process_fused_gate_up_projection_mxfp4(
    hidden: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    gate_up_scale_sb: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    output: nl.NkiTensor,
    attrs: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    gate_dequant_scale: Optional[nl.NkiTensor] = None,
    up_dequant_scale: Optional[nl.NkiTensor] = None,
    input_dequant_scale: Optional[nl.NkiTensor] = None,
):
    """
    Process gate and up projection, including the activation of gate projection and the final elem-wise multiply:
        output = act_fn(clamp(gate_proj(hidden))) * clamp(up_proj(hidden)).

    Weights, scales, and bias must be pre-loaded into SBUF by the caller (via _load_gate_up_* helpers).

    :param hidden: mxfp8_x4[_pmax, n_H512_tile_sharded, T] @ SB.
    :param hidden_scale: uint8[_pmax, n_H512_tile_sharded, T] @ SB.
    :param weight_sb: mxfp4_x4[_pmax, 2, n_H512_tile_sharded, I] @ SB. Pre-loaded gate/up weights.
    :param gate_up_scale_sb: uint8[_pmax, 2, n_H512_tile_sharded, I] @ SB. Pre-loaded gate/up scales.
    :param bias_sb: bf16[_pmax, 2, n_I512_tile, _q_width] @ SB, or None. Pre-loaded gate/up bias.
    :param output: bf16[_pmax, ceil(I/512), T, _q_width] @ SB.
    :param gate_dequant_scale: Optional fp32 @ SB. Dequant scale for gate projection.
        STATIC_MX: [_pmax, 1] combined (input * weight) scale. ROW_MX: [_pmax, n_I512*4] per-row weight scale.
    :param up_dequant_scale: Optional fp32 @ SB. Dequant scale for up projection.
        STATIC_MX: [_pmax, 1] combined (input * weight) scale. ROW_MX: [_pmax, n_I512*4] per-row weight scale.
    :param input_dequant_scale: Optional fp32[_pmax, T, 1] @ SB. ROW_MX per-token input dequant scale.

    NOTE: In the fused weights/scales/bias above, idx 0 is for gate and idx 1 is for up.
    """
    # Get sharding info on H
    shard_id, num_shards = (0, 1) if attrs.shard_on_h_disabled else (dims.shard_id, dims.num_shards)

    # Get dims and tiling info
    _, _, T = (
        hidden.shape
    )  # NOTE: this may be different from dims.T, e.g. all tokens would iter tokens 1-by-1 making T==1
    n_H512_tile_sharded = dims.H_shard // (_pmax * _q_width)
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))

    """
    Reshape workaround for NKI new FE indexing bug.
    
    NKI new FE has bug where indexing does not reduce number of dims.
    Need reshapes as workaround.
    """
    weight_sb = weight_sb.reshape((_pmax, 2 * n_H512_tile_sharded, dims.I))
    gate_up_scale_sb = gate_up_scale_sb.reshape((_pmax, 2 * n_H512_tile_sharded, dims.I))

    """
    Compute gate and up projections separately.
    
    Both projections' output shape is bf16[_pmax, n_I512_tile, T, _q_width].
    The bottom portion of the final I512 tile contains garbage.
    By providing prg_id even with n_prgs=1, we enforce only one NC to apply the bias
    (we shard on H for gate/up proj).
    """
    gate_proj_cfg = ProjConfig(
        H=dims.H_shard,
        I=dims.I,
        BxS=T,
        n_prgs=num_shards,
        prg_id=shard_id,
        bias_t_shared_between_gate_up=True,
        bias_t_shared_base_offset=0,
    )
    up_proj_cfg = ProjConfig(
        H=dims.H_shard,
        I=dims.I,
        BxS=T,
        n_prgs=num_shards,
        prg_id=shard_id,
        bias_t_shared_between_gate_up=True,
        bias_t_shared_base_offset=n_I512_tile * _q_width,
    )
    # Wrap tensors in nl.NkiTensor and use slice() for dimension 1
    hidden_tv = hidden
    hidden_scale_tv = hidden_scale
    weight_tv = weight_sb
    scale_tv = gate_up_scale_sb
    is_static_quant = attrs.quant_params.is_quant_static_mx()
    is_row_quant = attrs.quant_params.is_quant_row_mx()
    # STATIC_MX: bias applied after dequant in this function (not inside projection kernel)
    # ROW_MX/MX: bias applied inside projection kernel
    bias_tv = (bias_sb if bias_sb != None else None) if not is_static_quant else None

    # ROW_MX: pass weight dequant scales to underlying kernel (dispatches on shape internally)
    # STATIC_MX: dequant applied after LNC reduce in this function (not inside kernel)
    gate_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_tv,
        hidden_scale_sb=hidden_scale_tv,
        weight_qtz=weight_tv.slice(dim=1, start=0, end=n_H512_tile_sharded),
        weight_scale=scale_tv.slice(dim=1, start=0, end=n_H512_tile_sharded),
        bias_sb=bias_tv,
        cfg=gate_proj_cfg,
        w_dequant_scale=gate_dequant_scale if is_row_quant else None,
        input_dequant_scale=input_dequant_scale,
    )  # bf16[_pmax, n_I512_tile, T, _q_width]
    up_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_tv,
        hidden_scale_sb=hidden_scale_tv,
        weight_qtz=weight_tv.slice(dim=1, start=n_H512_tile_sharded, end=2 * n_H512_tile_sharded),
        weight_scale=scale_tv.slice(dim=1, start=n_H512_tile_sharded, end=2 * n_H512_tile_sharded),
        bias_sb=bias_tv,
        cfg=up_proj_cfg,
        w_dequant_scale=up_dequant_scale if is_row_quant else None,
        input_dequant_scale=input_dequant_scale,
    )  # bf16[_pmax, n_I512_tile, T, _q_width]

    # Perform SendRecv between two NCs to reduce/gather gate_proj results.
    # The SendRecv for up_proj results is postponed for ILP.
    if num_shards > 1:
        _lnc_reduce_proj_out(gate_proj_out_sb, shard_id)

    # STATIC_MX: post-matmul dequant + bias (ROW_MX dequant handled inside projection kernel)
    # Fuse scale+bias: out = out * scale + bias in one nisa.activation per (i_tile, i_q) slice.
    if is_static_quant:
        if bias_sb != None:
            bias_2d = bias_sb.reshape((_pmax, 2 * n_I512_tile * _q_width))
            for i_tile in range(n_I512_tile):
                for i_q in range(_q_width):
                    col_idx = i_tile * _q_width + i_q
                    nisa.activation(
                        dst=gate_proj_out_sb[:, i_tile, :, i_q],
                        op=nl.copy,
                        data=gate_proj_out_sb[:, i_tile, :, i_q],
                        scale=gate_dequant_scale,
                        bias=bias_2d[:, col_idx],
                    )
        else:
            nisa.activation(dst=gate_proj_out_sb, op=nl.copy, data=gate_proj_out_sb, scale=gate_dequant_scale)

    # Optionally perform clamping on gate projection results
    if attrs.gate_clamp_upper_limit != None or attrs.gate_clamp_lower_limit != None:
        nisa.tensor_scalar(
            dst=gate_proj_out_sb,
            data=gate_proj_out_sb,
            op0=nl.minimum if attrs.gate_clamp_upper_limit != None else None,
            operand0=attrs.gate_clamp_upper_limit,
            op1=nl.maximum if attrs.gate_clamp_lower_limit != None else None,
            operand1=attrs.gate_clamp_lower_limit,
        )

    # Compute activation(gate): it is either silu(gate) or swish(gate), based on attrs.act_fnd
    nisa.activation(dst=gate_proj_out_sb, op=get_nl_act_fn_from_type(attrs.activation_fn), data=gate_proj_out_sb)

    # Perform SendRecv between two NCs to reduce/gather up_proj results.
    if num_shards > 1:
        _lnc_reduce_proj_out(up_proj_out_sb, shard_id)

    # STATIC_MX: post-matmul dequant + bias (ROW_MX dequant handled inside projection kernel)
    if is_static_quant:
        if bias_sb != None:
            # bias_2d already computed above for gate; reuse the same reshape
            for i_tile in range(n_I512_tile):
                for i_q in range(_q_width):
                    col_idx = n_I512_tile * _q_width + i_tile * _q_width + i_q
                    nisa.activation(
                        dst=up_proj_out_sb[:, i_tile, :, i_q],
                        op=nl.copy,
                        data=up_proj_out_sb[:, i_tile, :, i_q],
                        scale=up_dequant_scale,
                        bias=bias_2d[:, col_idx],
                    )
        else:
            nisa.activation(dst=up_proj_out_sb, op=nl.copy, data=up_proj_out_sb, scale=up_dequant_scale)

    # Optionally perform clamping on up projection results
    if attrs.up_clamp_upper_limit != None or attrs.up_clamp_lower_limit != None:
        nisa.tensor_scalar(
            dst=up_proj_out_sb,
            data=up_proj_out_sb,
            op0=nl.minimum if attrs.up_clamp_upper_limit != None else None,
            operand0=attrs.up_clamp_upper_limit,
            op1=nl.maximum if attrs.up_clamp_lower_limit != None else None,
            operand1=attrs.up_clamp_lower_limit,
        )

    # Multiply gate and up projection outputs
    nisa.tensor_tensor(dst=output, data1=gate_proj_out_sb, data2=up_proj_out_sb, op=nl.multiply)

    return output
