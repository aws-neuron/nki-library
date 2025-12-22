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
from ...utils.kernel_helpers import div_ceil
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .projection_mx_constants import (
    SBUF_QUADRANT_SIZE,
    ProjConfig,
    _pmax,
    _psum_bmax,
    _psum_fmax,
    _q_height,
    _q_width,
)


def _down_proj_prep_inter_and_weights(
    inter_sb: nl.ndarray, weight: nl.ndarray, weight_scale: nl.ndarray, cfg: ProjConfig
) -> tuple[nl.ndarray, nl.ndarray, nl.ndarray, nl.ndarray]:
    """
    Prep intermediate and weights for down projection:
        - for intermediate, reshape and quantize (and reshape back);
        - for weight, load from HBM into SBUF.

    :param inter_sb: bf16[_pmax, n_I512_tile, BxS, 4] @ SB. Dim I is shuffled on 128.
    :param weight: mxfp4_x4[_pmax, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :param weight_scale: mxfp4_x4[_pmax // _q_height, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :return:
        1. (inter_qtz)        mxfp8_x4[_pmax, cfg.n_total_I512_tile, BxS]
        2. (inter_qtz_scale)  uint8[_pmax, cfg.n_total_I512_tile, BxS]
        3. (weight_qtz)       mxfp4_x4[_pmax, cfg.n_total_I512_tile, H_sharded]
        4. (weight_qtz_scale) uint8[_pmax, cfg.n_total_I512_tile, H_sharded]
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H, I, BxS = cfg.H, cfg.I, cfg.BxS
    H_sharded = H // n_prgs
    p_I = _pmax if I > 512 else I // 4  # we do not pad I if I<512 to save HBM

    # Quantize inter_sb into mxfp4_x4[_pmax, ceil(I/512), BxS] @ SB
    # NOTE: when I%512 != 0, the final I512 tile of inter_sb will contain garbage. Also nc_matmul_mx requires 32/64/128
    # partitions input so all 128 partitions are used (which includes garbage). However, we memset the last tile
    # of weight_qtz and weight_qtz_scale such that the garbage does not matter.
    inter_sb = inter_sb.reshape((_pmax, cfg.n_total_I512_tile * BxS * 4))  # flatten to 2D
    inter_qtz = nl.ndarray((_pmax, cfg.n_total_I512_tile * BxS), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    inter_qtz_scale = nl.ndarray(inter_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=inter_qtz, src=inter_sb, dst_scale=inter_qtz_scale)
    inter_qtz = inter_qtz.reshape((_pmax, cfg.n_total_I512_tile, BxS))
    inter_qtz_scale = inter_qtz_scale.reshape(inter_qtz.shape)

    if cfg.dbg_hidden:
        return inter_qtz, inter_qtz_scale, None, None  # DEBUG

    weight_qtz = None
    if weight.buffer == nl.sbuf:
        weight_qtz = weight
    else:
        # Load weight into [I0, ceil(I/512), H_sharded] NOTE: this is pre-quantized and each elt is mxfp4_x4 (packed I)
        weight_qtz = nl.ndarray(
            (_pmax, cfg.n_total_I512_tile, H_sharded), dtype=nl.float4_e2m1fn_x4, buffer=nl.sbuf, name='down_w_qtz_sb'
        )
        # Memset weight if input weight HBM does not pad on par dim
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz[:, cfg.n_total_I512_tile - 1, :], value=0.0)

        kernel_assert(weight.shape == (p_I, cfg.n_total_I512_tile, H), "Incorrect weight shape")
        nisa.dma_copy(
            src=weight[:, :, prg_id * H_sharded : (prg_id + 1) * H_sharded], dst=weight_qtz[:p_I, :, :], dge_mode=2
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
        weight_qtz_scale = nl.ndarray(weight_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf, name="down_w_scale_sb")
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
                        dge_mode=2,
                    )

    return inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale


def down_projection_mx_tp_shard_H(
    inter_sb: nl.ndarray, weight: nl.ndarray, weight_scale: nl.ndarray, bias_sb: Optional[nl.ndarray], cfg: ProjConfig
) -> nl.ndarray:
    """
    Performs the Down projection with H-dimension sharding. Math (Neuron matmul):
        inter_sb (moving) [I, BxS] @ weight (stationary) [I, H] → [H, BxS].

    NOTE: each matmul tile reads a weight of shape [_pmax (I), _pmax (H)], however the second dim (on H) is not
    a contiguous slice of 128 elts from the full H. Instead, those are 128 elts with a stride of H//128 (H1).
    This means the final output of shape [_pmax (H0), H//_pmax (H1), BxS] would have a contiguous H1 and strided H0.

    :param inter_sb: bf16[_pmax, n_I512_tile, BxS, 4] @ SB. Dim I is shuffled on 128.
    :param weight: mxfp4_x4[_pmax, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :param weight_scale: mxfp4_x4[_pmax // _q_height, ceil(I/512), H] @ HBM. NOTE: expect zero-padding.
    :param bias_sb [OPTIONAL]: bf16[_pmax, H_sharded//_pmax] @ SB.
    :return: bf16[_pmax, H//_pmax, BxS] @ SB.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    kernel_assert(cfg.H_sharded % _pmax == 0, "down projection with [H, T] output layout requires H divisible by 128")
    kernel_assert(cfg.BxS <= 128, f"MX4 down proj with HT output layout only supports TKG but got {cfg.BxS=}")

    # Prep inputs
    inter_qtz, inter_qtz_scale, weight_qtz, weight_qtz_scale = _down_proj_prep_inter_and_weights(
        inter_sb, weight, weight_scale, cfg
    )
    if cfg.dbg_weight:
        return weight_qtz, weight_qtz_scale

    # Matmul compute, tiles on H
    out_sb = nl.ndarray((cfg.H0, cfg.H1, cfg.BxS), dtype=nl.bfloat16, buffer=nl.sbuf)  # alloc space for full H

    for i_H1 in range(cfg.H1_sharded):
        # Allocate psum for current H128 tile
        h128_psum = nl.ndarray((cfg.H0, cfg.BxS), dtype=nl.bfloat16, buffer=nl.psum)

        # Loop over I512 tiles
        for i_I512_tile in range(cfg.n_total_I512_tile):
            # Stationary accesses entire I0 because it's been zero-padded
            nisa.nc_matmul_mx(
                dst=h128_psum,
                stationary=weight_qtz[:, i_I512_tile, i_H1 * _pmax : (i_H1 + 1) * _pmax],
                moving=inter_qtz[:, i_I512_tile, :],  # [_pmax (I), BxS]
                stationary_scale=weight_qtz_scale[:, i_I512_tile, i_H1 * _pmax : (i_H1 + 1) * _pmax],
                moving_scale=inter_qtz_scale[:, i_I512_tile, :],
            )

        # Copy out the current H128 tile to SB, use ACT because DVE is usually bottlenecked
        act_bias_arg = None
        if bias_sb is not None:
            act_bias_arg = bias_sb[:, i_H1]
        nisa.activation(dst=out_sb[:, i_H1 + cfg.H1_sharded * prg_id, :], op=nl.copy, data=h128_psum, bias=act_bias_arg)

    # Receive projection output from the other NC when LNC > 1
    if n_prgs > 1:
        other_prg_id = 1 - prg_id
        nisa.sendrecv(
            src=out_sb[:, prg_id * cfg.H1_sharded : (prg_id + 1) * cfg.H1_sharded, :],
            dst=out_sb[:, other_prg_id * cfg.H1_sharded : (other_prg_id + 1) * cfg.H1_sharded, :],
            send_to_rank=other_prg_id,
            recv_from_rank=other_prg_id,
            pipe_id=0,
        )

    return out_sb


def down_projection_mx_shard_H(
    inter_sb: nl.ndarray, weight: nl.ndarray, weight_scale: nl.ndarray, bias_sb: nl.ndarray, cfg: ProjConfig
) -> nl.ndarray:
    """
    Perform down projection with MXFP4 quantization.

    Computes weight @ intermediate + bias using MXFP4 quantized weights,
    producing final MLP output. This version supports larger BxS values (CTE workloads)
    by tiling the BxS dimension.

    Args:
        inter_sb (nl.ndarray): Intermediate activations of shape [128, n_I512_tile, BxS, 4]
            in SBUF with I dimension shuffled on 128 partitions, bf16 type.
        weight (nl.ndarray): Quantized weights of shape [128, ceil(I/512), H] in HBM,
            mxfp4_x4 type, zero-padded.
        weight_scale (nl.ndarray): Weight scales of shape [128//8, ceil(I/512), H] in HBM,
            uint8 type, zero-padded.
        bias_sb (nl.ndarray): Optional bias of shape [1, H_sharded] in SBUF, bf16 type.
        cfg (ProjConfig): Projection configuration with H, I, BxS, sharding info.

    Returns:
        output (nl.ndarray): Down projection result of shape [128, ceil(BxS/128), H] in SBUF,
            bf16 type. Note: end of last tile contains garbage when BxS % 128 != 0.

    Notes:
        - Math: weight [I, H] @ inter_sb [I, BxS] → [BxS, H]
        - Quantizes intermediate activations online to MXFP4
        - Uses nc_matmul_mx for MXFP4 matrix multiplication
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
        inter_sb, weight, weight_scale, cfg
    )

    if cfg.dbg_weight:
        return weight_qtz, weight_qtz_scale

    # Prepare bias broadcast if needed
    bias_broadcasted = None
    if bias_sb is not None:
        if bias_sb.shape[0] == 1:
            bias_broadcasted = nl.ndarray((BxS_tile_sz, H_sharded), dtype=bias_sb.dtype, buffer=nl.sbuf)
            stream_shuffle_broadcast(src=bias_sb, dst=bias_broadcasted)
        else:
            bias_broadcasted = bias_sb

    # Allocate output buffer
    if cfg.out_p_offset != 0:
        out_sb = nl.ndarray((_pmax, n_BxS_tile, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        out_sb_p_start = cfg.out_p_offset
        out_sb_p_end = cfg.out_p_offset + BxS
    else:
        out_sb = nl.ndarray((BxS_tile_sz, n_BxS_tile, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        out_sb_p_start = 0
        out_sb_p_end = BxS_tile_sz

    H_tile_size = 1024 if H_sharded % 1024 == 0 else 512
    n_H_tile_sharded = H_sharded // H_tile_size

    for i_H_tile in nl.affine_range(n_H_tile_sharded):
        for i_BxS_tile in nl.affine_range(n_BxS_tile):
            BxS_offset = i_BxS_tile * BxS_tile_sz
            curr_BxS = min(BxS_tile_sz, BxS - BxS_offset)

            psum_bank = nl.ndarray((curr_BxS, H_tile_size), dtype=nl.bfloat16, buffer=nl.psum)

            for i_I512_tile in nl.affine_range(cfg.n_total_I512_tile):
                H_offset = i_H_tile * H_tile_size
                nisa.nc_matmul_mx(
                    dst=psum_bank,
                    stationary=inter_qtz[:, i_I512_tile, BxS_offset : BxS_offset + curr_BxS],
                    moving=weight_qtz[:, i_I512_tile, H_offset : H_offset + H_tile_size],
                    stationary_scale=inter_qtz_scale[:, i_I512_tile, BxS_offset : BxS_offset + curr_BxS],
                    moving_scale=weight_qtz_scale[:, i_I512_tile, H_offset : H_offset + H_tile_size],
                )

            H_out_start = H_sharded * prg_id + i_H_tile * H_tile_size

            if cfg.out_p_offset == 0:
                if bias_sb is not None:
                    nisa.tensor_tensor(
                        dst=out_sb[:curr_BxS, i_BxS_tile, H_out_start : H_out_start + H_tile_size],
                        data1=psum_bank,
                        data2=bias_broadcasted[:curr_BxS, H_offset : H_offset + H_tile_size],
                        op=nl.add,
                    )
                else:
                    engine = nisa.scalar_engine if i_BxS_tile % 2 == 0 else nisa.vector_engine
                    nisa.tensor_copy(
                        dst=out_sb[:curr_BxS, i_BxS_tile, H_out_start : H_out_start + H_tile_size],
                        src=psum_bank,
                        engine=engine,
                    )
            else:
                if bias_sb is not None:
                    nisa.tensor_tensor(
                        dst=out_sb[
                            out_sb_p_start : out_sb_p_start + curr_BxS,
                            i_BxS_tile,
                            H_out_start : H_out_start + H_tile_size,
                        ],
                        data1=psum_bank,
                        data2=bias_broadcasted[:curr_BxS, H_offset : H_offset + H_tile_size],
                        op=nl.add,
                    )
                else:
                    engine = nisa.scalar_engine if i_BxS_tile % 2 == 0 else nisa.vector_engine
                    nisa.tensor_copy(
                        dst=out_sb[
                            out_sb_p_start : out_sb_p_start + curr_BxS,
                            i_BxS_tile,
                            H_out_start : H_out_start + H_tile_size,
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
