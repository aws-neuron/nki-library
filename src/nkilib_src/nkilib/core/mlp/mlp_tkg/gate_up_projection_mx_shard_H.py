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

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
from ...utils.tensor_view import TensorView
from ..mlp_parameters import MLPParameters
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
    hidden_qtz_sb: TensorView,
    hidden_scale_sb: TensorView,
    weight_qtz: TensorView,
    weight_scale: TensorView,
    bias_sb: TensorView,
    cfg: ProjConfig,
) -> nl.ndarray:
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
    :param bias_sb [OPTIONAL]: TensorView of bf16[_pmax, n_I512_tile, _q_width] @ SB.
        - For already-sliced (gate or up only): cfg.bias_t_shared_base_offset should be 0
        - For combined gate+up: cfg.bias_t_shared_base_offset specifies offset for up projection
    :return: bf16[_pmax, ceil(I / 512), BxS, _q_width] @ SB.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H0, H1, H1_sharded, I, BxS = cfg.H0, cfg.H1, cfg.H1_sharded, cfg.I, cfg.BxS

    BxS_tile_sz = min(BxS, _psum_fmax * 2 // _q_width)  # double psum elts because out is in bf16
    n_BxS_tile = div_ceil(BxS, BxS_tile_sz)

    # Either load weight_qtz from HBM to sbuf or directly use it if it is already in SBUF
    if weight_qtz.base_tensor.buffer == nl.sbuf:
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
        weight_qtz_sb = nl.ndarray((H0, cfg.n_H512_tile_sharded, I), dtype=weight_qtz.base_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=weight_qtz_sb,
            src=weight_qtz.base_tensor[:, prg_id * cfg.n_H512_tile_sharded : (prg_id + 1) * cfg.n_H512_tile_sharded, :],
        )
        weight_qtz_tv = TensorView(weight_qtz_sb)

    if cfg.dbg_hidden:
        return hidden_qtz_sb.base_tensor, hidden_scale_sb.base_tensor

    if weight_scale.base_tensor.buffer == nl.sbuf:
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
        weight_scale_sb = nl.ndarray((H0, cfg.n_H512_tile_sharded, I), dtype=nl.uint8, buffer=nl.sbuf)
        # Load 4 partitions of scales for every quadrant
        n_quadrants_needed = H0 // SBUF_QUADRANT_SIZE
        for i_quad in range(n_quadrants_needed):
            nisa.dma_copy(
                src=weight_scale.base_tensor[
                    i_quad * 4 : (i_quad + 1) * 4,
                    prg_id * cfg.n_H512_tile_sharded : (prg_id + 1) * cfg.n_H512_tile_sharded,
                    :,
                ],
                dst=weight_scale_sb[i_quad * SBUF_QUADRANT_SIZE : i_quad * SBUF_QUADRANT_SIZE + 4, :, :],
            )
        weight_scale_tv = TensorView(weight_scale_sb)

    if cfg.dbg_weight:
        return weight_qtz_tv.base_tensor, weight_scale_tv.base_tensor

    out_sb = nl.ndarray((_pmax, cfg.n_total_I512_tile, BxS, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)

    # Loop over BxS tiles (each of size 256)
    for i_BxS_tile in range(n_BxS_tile):
        # For the last iter, we may have less than BxS_tile_sz to work with
        cur_BxS_tile_offset = i_BxS_tile * BxS_tile_sz
        cur_BxS_tile_sz = min(BxS_tile_sz, BxS - cur_BxS_tile_offset)

        # Allocate and init output psum and sbuf. Note that there are cfg.n_total_I512_tile instances of out_psum
        out_psum_lst = []
        for i_I512_tile in range(cfg.n_total_I512_tile):
            out_psum_lst.append(nl.ndarray((_pmax, _q_width, cur_BxS_tile_sz), dtype=nl.bfloat16, buffer=nl.psum))

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
                        stationary=weight_qtz_tv.slice(1, i_H512_tile, i_H512_tile + 1)
                        .slice(2, weight_I_offset, weight_I_offset + cur_I128_tile_sz)
                        .get_view(),
                        moving=hidden_qtz_sb.slice(1, i_H512_tile, i_H512_tile + 1)
                        .slice(2, cur_BxS_tile_offset, cur_BxS_tile_offset + cur_BxS_tile_sz)
                        .get_view(),
                        stationary_scale=weight_scale_tv.slice(1, i_H512_tile, i_H512_tile + 1)
                        .slice(2, weight_I_offset, weight_I_offset + cur_I128_tile_sz)
                        .get_view(),
                        moving_scale=hidden_scale_sb.slice(1, i_H512_tile, i_H512_tile + 1)
                        .slice(2, cur_BxS_tile_offset, cur_BxS_tile_offset + cur_BxS_tile_sz)
                        .get_view(),
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
            if (bias_sb is not None) and (prg_id == 0):
                # Slice and broadcast bias to match output shape
                bias_tile_view = bias_sb.slice(dim=0, start=0, end=cur_I_pdim_sz)
                if cfg.bias_t_shared_between_gate_up:
                    # 4D [_pmax, 2, n_I512_tile, _q_width] becomes 3D via select
                    gate_or_up_idx = 0 if cfg.bias_t_shared_base_offset == 0 else 1
                    bias_tile_view = bias_tile_view.select(dim=1, index=gate_or_up_idx)
                # Now 3D: [cur_I_pdim_sz, n_I512_tile, _q_width]
                bias_tile_view = bias_tile_view.slice(dim=1, start=i_I512_tile, end=i_I512_tile + 1)
                bias_tile_view = bias_tile_view.broadcast(dim=1, size=cur_BxS_tile_sz)

                nisa.tensor_tensor(
                    dst=out_sb[
                        :cur_I_pdim_sz, i_I512_tile, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                    ],
                    data1=out_psum_lst[i_I512_tile].ap(
                        [[_q_width * cur_BxS_tile_sz, cur_I_pdim_sz], [1, cur_BxS_tile_sz], [cur_BxS_tile_sz, _q_width]]
                    ),  # strided read
                    data2=bias_tile_view.get_view(),
                    op=nl.add,
                )
            else:
                nisa.tensor_copy(
                    dst=out_sb[
                        :cur_I_pdim_sz, i_I512_tile, cur_BxS_tile_offset : cur_BxS_tile_offset + cur_BxS_tile_sz, :
                    ],
                    src=out_psum_lst[i_I512_tile].ap(
                        [[_q_width * cur_BxS_tile_sz, cur_I_pdim_sz], [1, cur_BxS_tile_sz], [cur_BxS_tile_sz, _q_width]]
                    ),  # strided read
                )

    # Receive projection output from the other NC when LNC > 1
    if n_prgs > 1:
        recv = nl.ndarray(out_sb.shape, dtype=out_sb.dtype, buffer=nl.sbuf)
        nisa.sendrecv(src=out_sb, dst=recv, send_to_rank=(1 - prg_id), recv_from_rank=(1 - prg_id), pipe_id=0)
        nisa.tensor_tensor(dst=out_sb, data1=out_sb, data2=recv, op=nl.add)

    return out_sb


def _lnc_reduce_proj_out(cur_nc_proj_out: nl.ndarray, shard_id: int):
    """In-place LNC2 reduction of projection output."""
    # SendRecv
    proj_out_recv = nl.ndarray(cur_nc_proj_out.shape, dtype=cur_nc_proj_out.dtype, buffer=nl.sbuf)
    nisa.sendrecv(
        src=cur_nc_proj_out, dst=proj_out_recv, send_to_rank=(1 - shard_id), recv_from_rank=(1 - shard_id), pipe_id=0
    )

    # Reduction, because each NC handled half of contraction (H)
    nisa.tensor_tensor(dst=cur_nc_proj_out, data1=cur_nc_proj_out, data2=proj_out_recv, op=nl.add)


def process_fused_gate_up_projection_mxfp4(
    hidden: nl.ndarray,
    hidden_scale: nl.ndarray,
    gate_up_weights: nl.ndarray,
    gate_up_scale: nl.ndarray,
    gate_up_bias: nl.ndarray,
    p_idx_vector: nl.ndarray,
    gate_up_scale_sb: nl.ndarray,
    output: nl.ndarray,
    attrs: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    gate_up_weights_E_offset: Optional[nl.ndarray],
    gate_up_bias_E_offset: Optional[nl.ndarray],
):
    """
    Process gate and up projection, including the activation of gate projection and the final elem-wise multiply:
        output = act_fn(clamp(gate_proj(hidden))) * clamp(up_proj(hidden)).

    :param hidden: mxfp8_x4[_pmax, n_H512_tile_sharded, T] @ SB.
    :param hidden_scale: uint8[_pmax, n_H512_tile_sharded, T] @ SB.
    :param gate_up_weights: mxfp4_x4[_pmax, 2, n_H512_tiles, I] @ HBM.
    :param gate_up_scale: uint8[E, _pmax // _q_height, 2, n_H512_tiles, I] @ HBM.
    :param gate_up_bias: bf16[I_p, 2, ceil(I/512), 4] @ HBM, where I_p = I//4 if I <= 512 else _pmax.
    :param gate_up_scale_sb: uint8[_pmax, 2, n_H512_tile_sharded, I] @ SB.
    :param output: bf16[_pmax, ceil(I/512), T, _q_width] @ SB.
    :param gate_up_weight_E_offset: int32[1, 1] @ SB. When this is provided, gate_up_weights has an additional leading E dim.
    :param gate_up_bias_E_offset: int32[1, 1] @ SB. When this is provided, gate_up_weights has an additional leading E dim.

    NOTE: In the fused weights/scales/bias above, idx 0 is for gate and idx 1 is for up.
    """
    # Get sharding info on H
    shard_id, num_shards = (0, 1) if attrs.shard_on_h_disabled else (dims.shard_id, dims.num_shards)

    # Get dims and tiling info
    _, _, T = (
        hidden.shape
    )  # NOTE: this may be different from dims.T, e.g. all tokens would iter tokens 1-by-1 making T==1
    n_H512_tile_sharded = dims.H_shard // (_pmax * _q_width)
    n_H512_tiles = dims.H // (_pmax * _q_width)
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))

    # Allocate and load weight sbuf shared between gate and up projection
    weight_sb = nl.ndarray((_pmax, 2, n_H512_tile_sharded, dims.I), dtype=gate_up_weights.dtype, buffer=nl.sbuf)
    if gate_up_weights_E_offset is None:
        nisa.dma_copy(
            dst=weight_sb,
            src=gate_up_weights[:, :, shard_id : (shard_id + 1) * n_H512_tile_sharded, :],
            dge_mode=nisa.dge_mode.swdge,
        )
    else:
        nisa.dma_copy(
            dst=weight_sb,
            src=gate_up_weights.ap(
                pattern=[
                    [2 * n_H512_tiles * dims.I, _pmax],
                    [n_H512_tiles * dims.I, 2],
                    [dims.I, n_H512_tile_sharded],
                    [1, dims.I],
                ],
                offset=shard_id * n_H512_tile_sharded * dims.I,
                scalar_offset=gate_up_weights_E_offset,
                indirect_dim=0,
            ),
        )

    # Alloc and load weight scale, which needs zero padding in sbuf
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
                [2 * n_H512_tiles * dims.I, _pmax],
                [n_H512_tiles * dims.I, 2],
                [dims.I, n_H512_tile_sharded],
                [1, dims.I],
            ],
            offset=(shard_id * n_H512_tile_sharded) * dims.I,
            vector_offset=token_indices_on_p,
            indirect_dim=0,
        ),
        oob_mode=oob_mode.skip,
    )

    # Alloc and load bias, which needs zero padding if I < 512
    bias_sb = nl.ndarray((_pmax, 2, n_I512_tile, _q_width), dtype=gate_up_bias.dtype, buffer=nl.sbuf)
    if dims.I < 512:  # when I<512, gate/up bias HBM is not padded so pad it here
        nisa.memset(dst=bias_sb[:, :, 0, :], value=0.0)
        if gate_up_weights_E_offset is None:
            nisa.dma_copy(dst=bias_sb[: dims.I // 4, :, :, :], src=gate_up_bias, dge_mode=nisa.dge_mode.hwdge)
        else:
            nisa.dma_copy(
                dst=bias_sb[: dims.I // 4, :, :, :],
                src=gate_up_bias.ap(
                    pattern=[
                        [2 * n_I512_tile * _q_width, dims.I // 4],
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
        if gate_up_weights_E_offset is None:
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
    # Wrap tensors in TensorView and use slice() for dimension 1
    hidden_tv = TensorView(hidden)
    hidden_scale_tv = TensorView(hidden_scale)
    weight_tv = TensorView(weight_sb)
    scale_tv = TensorView(gate_up_scale_sb)
    bias_tv = TensorView(bias_sb)

    gate_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_tv,
        hidden_scale_sb=hidden_scale_tv,
        weight_qtz=weight_tv.slice(dim=1, start=0, end=n_H512_tile_sharded),
        weight_scale=scale_tv.slice(dim=1, start=0, end=n_H512_tile_sharded),
        bias_sb=bias_tv,
        cfg=gate_proj_cfg,
    )  # bf16[_pmax, n_I512_tile, T, _q_width]
    up_proj_out_sb = gate_up_projection_mx_tp_shard_H(
        hidden_qtz_sb=hidden_tv,
        hidden_scale_sb=hidden_scale_tv,
        weight_qtz=weight_tv.slice(dim=1, start=n_H512_tile_sharded, end=2 * n_H512_tile_sharded),
        weight_scale=scale_tv.slice(dim=1, start=n_H512_tile_sharded, end=2 * n_H512_tile_sharded),
        bias_sb=bias_tv,
        cfg=up_proj_cfg,
    )  # bf16[_pmax, n_I512_tile, T, _q_width]

    # Perform SendRecv between two NCs to reduce/gather gate_proj results.
    # The SendRecv for up_proj results is postponed for ILP.
    if num_shards > 1:
        _lnc_reduce_proj_out(gate_proj_out_sb, shard_id)

    # Optionally perform clamping on gate projection results
    nisa.tensor_scalar(
        dst=gate_proj_out_sb,
        data=gate_proj_out_sb,
        op0=nl.minimum if attrs.gate_clamp_upper_limit is not None else None,
        operand0=attrs.gate_clamp_upper_limit,
        op1=nl.maximum if attrs.gate_clamp_lower_limit is not None else None,
        operand1=attrs.gate_clamp_lower_limit,
    )

    # Compute activation(gate): it is either silu(gate) or swish(gate), based on attrs.act_fnd
    nisa.activation(dst=gate_proj_out_sb, op=get_nl_act_fn_from_type(attrs.activation_fn), data=gate_proj_out_sb)

    # Perform SendRecv between two NCs to reduce/gather up_proj results.
    if num_shards > 1:
        _lnc_reduce_proj_out(up_proj_out_sb, shard_id)

    # Optionally perform clamping on up projection results
    nisa.tensor_scalar(
        dst=up_proj_out_sb,
        data=up_proj_out_sb,
        op0=nl.minimum if attrs.up_clamp_upper_limit is not None else None,
        operand0=attrs.up_clamp_upper_limit,
        op1=nl.maximum if attrs.up_clamp_lower_limit is not None else None,
        operand1=attrs.up_clamp_lower_limit,
    )

    # Multiply gate and up projection outputs
    nisa.tensor_tensor(dst=output, data1=gate_proj_out_sb, data2=up_proj_out_sb, op=nl.multiply)

    return output
