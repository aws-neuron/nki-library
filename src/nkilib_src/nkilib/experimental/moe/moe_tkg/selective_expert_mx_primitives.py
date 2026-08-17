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

"""Selective-expert MoE token generation implementation with MX (microscaling) FP4 quantization support."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from ....core.mlp.mlp_parameters import MLPParameters

# MLP utils
from ....core.mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstants
from ....core.mlp.mlp_tkg.projection_mx_constants import (
    SBUF_QUADRANT_SIZE,
    ProjConfig,
    _pmax,
    _psum_fmax,
    _q_height,
    _q_width,
)
from ....core.moe.moe_tkg.moe_tkg_utils import broadcast_token_affinity, gather_expert_affinities
from ....core.utils.allocator import SbufManager
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type

# Common utils
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...primitives import ColMajor, RowMajor, ViewOrder, blas, dma, tile_stream
from ...primitives.view_spec import ViewSpec, view


def _selective_expert_moe_tkg_mxfp4_primitives(
    params: MLPParameters,
    output: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    Perform selective-expert MoE MLP token generation with MXFP4 quantization.

    The input first goes through a layout adapter for desired MX-quantizable layout.

    Args:
        params (MLPParameters): MLPParameters containing all input tensors and configuration.
        output (nl.NkiTensor): [T, H], Output tensor in HBM.

    Returns:
        output (nl.NkiTensor): [T, H], Output tensor with MoE computation results in HBM.

    Notes:
        - This kernel only supports gate/up and down proj both swapped
        - gate_up_weights: mxfp4[E, _pmax, 2, n_H512_tiles, I] in HBM (2 dim means up & gate weights stacked)
        - down_weights: mxfp4[E, I_p, ceil(I/512), H] in HBM, where I_p = I//4 if I <= 512 else _pmax
        - gate_up_weights_scale: uint8[E, _pmax // _q_height, 2, n_H512_tiles, I] in HBM
        - down_weights_scale: uint8[E, I_p // _q_height, ceil(I/512), H] in HBM
        - gate_up_weights_bias: bf16[E, I_p, 2, ceil(I/512), 4] in HBM
        - down_weights_bias: bf16[E, H] in HBM (needs offline shuffling for down_lhs_rhs_swap)

    Pseudocode:
        # Layout adapter and quantization
        input_qtz, input_scale = layout_adapter(input)

        # Process each token
        for token_idx in range(T):
            for expert_k_idx in range(K):
                expert_idx = expert_index[token_idx, expert_k_idx]

                # Gate/up projection
                intermediate = gate_up_projection(input_qtz[token_idx], weights[expert_idx])

                # Down projection
                expert_out = down_projection(intermediate, down_weights[expert_idx])

                # Apply affinity and accumulate
                expert_out *= expert_affinities[token_idx, expert_idx]
                if expert_k_idx == 0:
                    output[token_idx] = expert_out
                else:
                    output[token_idx] += expert_out
    """
    # Init dims
    dims = MLPTKGConstants.calculate_constants(params)

    # This kernel uses auto allocation, init an auto allocator for subkernels that requires a sbm
    auto_sbm = SbufManager(0, 200 * 1024, use_auto_alloc=True)
    auto_sbm.open_scope()

    kernel_assert(not params.store_output_in_sbuf, "_all_token_mlp_mxfp4_kernel does not support sbuf output")
    kernel_assert(dims.T <= _pmax, "_all_token_mlp_mxfp4_kernel does not support T > 128")

    shard_on_K = True

    # Get intermediate dims
    kernel_assert(dims.H_shard % (_pmax * _q_width) == 0, "Expect H after sharding to be divisible by 512")
    n_H512_tile_sharded = dims.H_shard // (_pmax * _q_width)
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))
    T_padded = div_ceil(dims.T, 4) * 4

    # This is used iff. shard_on_K
    num_shards, shard_id = nl.num_programs(0), nl.program_id(0)
    K_sharded = dims.K
    if shard_on_K:
        kernel_assert(dims.K % num_shards == 0, "Selective load shard on K requires K divisible by num NC")
        K_sharded = dims.K // num_shards

    io_dtype = params.output_dtype

    # Use layout adapter to get quantizable layout for Gate/Up projection, runs this unsharded since we shard on K
    if params.input_in_sbuf:
        # input_sb_shfl = _layout_adapter_sb(params.hidden_tensor, n_prgs=1, prg_id=0)
        kernel_assert(False, "_layout_adapter_sb path not yet supported")
    else:
        input_sb_shfl = _layout_adapter_hbm(params.hidden_tensor, n_prgs=1, prg_id=0)

    inp_qtz = tile_stream.alloc_logical((n_H512_tile_sharded * _pmax, T_padded), _pmax, dtype=nl.float8_e4m3fn_x4)
    inp_scale = tile_stream.alloc_logical((n_H512_tile_sharded * _pmax, T_padded), _pmax, dtype=nl.uint8)
    nisa.quantize_mx(dst=inp_qtz, src=input_sb_shfl, dst_scale=inp_scale)
    inp_qtz = inp_qtz
    inp_scale = inp_scale

    # Allocate SBUF location to accumulate output which has shape [128, H_per_shard] to store the outputs for
    # four tokens on each of the four SBUF quadrants. This is to save sendrecvs (reduced by 4x).
    output_temp_shape = (dims.H0, dims.T, dims.H1_shard)
    output_temp = nl.ndarray(output_temp_shape, dtype=io_dtype, name=f"temp_output_sbuf", buffer=nl.sbuf)

    # Determine tiling on T. When down is not swapped (producing [T, H] output), it's tiled by 4. Otherwise we don't tile.
    sz_T_tile, n_T_tile = (dims.T, 1)

    # Allocate SBUF locations for gate/up projection results. NOTE: likely won't need fp32 precision, but keep this in mind
    intermediate_state_sb = nl.ndarray(
        (_pmax, n_I512_tile, 4, _q_width), dtype=io_dtype, name=f"intermediate_state_sbuf", buffer=nl.sbuf
    )

    # Load expert index
    if params.expert_params.expert_index.buffer == nl.sbuf:
        expert_idx = params.expert_params.expert_index
    else:
        expert_idx = nl.ndarray((dims.T, dims.K), dtype=params.expert_params.expert_index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=expert_idx, src=params.expert_params.expert_index[0 : dims.T, 0 : dims.K]
        )  # indices have to be in SBUF

    # Prepare expert index into broadcasted form for generating DGE indices
    # These scalars are broadcasted 4 times on the pdim for DGE indices
    expert_idx_f32 = nl.ndarray(expert_idx.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expert_idx_f32, op=nl.copy, data=expert_idx)
    expert_idx_scalar_broadcasted = nl.ndarray(
        (4, dims.T, K_sharded), dtype=params.expert_params.expert_index.dtype, buffer=nl.sbuf
    )
    K_start = shard_id * K_sharded if shard_on_K else 0
    blas.Transpose(
        dst=tile_stream.tile(
            expert_idx_scalar_broadcasted, tile_shape=(4, dims.T), tile_dims=(0, 1), has_p_tile_dim=False
        ),
        src=tile_stream.tile(
            expert_idx_f32.slice(dim=1, start=K_start, end=K_start + K_sharded),
            tile_shape=(dims.T, 1),
            has_p_tile_dim=False,
            tile_view=ViewSpec().broadcast(dim=1, size=4),
        ),
    ).execute()

    # Prepare expert index into vector DGE indices format
    p_idx_vector_gup = nl.ndarray((_pmax, dims.T, K_sharded), dtype=nl.float32, buffer=nl.sbuf, name="p_idx_vector_gup")
    nisa.memset(dst=p_idx_vector_gup, value=-1.0)

    p_idx_vector_down = nl.ndarray(
        (_pmax, dims.T, K_sharded), dtype=nl.float32, buffer=nl.sbuf, name="p_idx_vector_down"
    )
    nisa.memset(dst=p_idx_vector_down, value=-1.0)

    n_quadrants_needed = 4
    for i_quad in range(n_quadrants_needed):
        arange_4P = nl.ndarray((4, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=arange_4P, pattern=[[1, 1]], offset=i_quad * 4, channel_multiplier=1)

        # Generate indices for gate and up
        nisa.activation(
            dst=p_idx_vector_gup[i_quad * 32 : i_quad * 32 + 4, :, :],
            op=nl.copy,
            data=expert_idx_scalar_broadcasted,
            scale=float(16),
            bias=arange_4P,
        )

        # Generate indices for down
        nisa.activation(
            dst=p_idx_vector_down[i_quad * 32 : i_quad * 32 + 4, :, :],
            op=nl.copy,
            data=expert_idx_scalar_broadcasted,
            scale=float(params.quant_params.down_w_scale.shape[1]),
            bias=arange_4P,
        )

    # Load expert affinity
    expert_affinities_sb = nl.ndarray(
        (dims._pmax, dims.E), dtype=params.expert_params.expert_affinities.dtype, buffer=nl.sbuf
    )
    nisa.memset(dst=expert_affinities_sb, value=0.0)
    if params.expert_params.expert_affinities.buffer == nl.sbuf:
        nisa.tensor_copy(dst=expert_affinities_sb[: dims.T, :], src=params.expert_params.expert_affinities)
    else:
        # Prefetch expertIndices (Up to 128 tokens input)
        nisa.dma_copy(dst=expert_affinities_sb[: dims.T, :], src=params.expert_params.expert_affinities)

    # Gather expert affinities using utility function
    if params.expert_params.expert_affinities_eager != None:
        # broadcast expert_affinities_eager into a [128(P), K(F), T(F)] tensor
        expert_affi_eager_sb = nl.ndarray(
            (dims._pmax, K_sharded, dims.T), dtype=params.expert_params.expert_affinities.dtype, buffer=nl.sbuf
        )
        K_start = shard_id * K_sharded if shard_on_K else 0
        affi_eager_src = params.expert_params.expert_affinities_eager.slice(dim=0, start=0, end=dims.T).slice(
            dim=1, start=K_start, end=K_start + K_sharded
        )
        blas.Transpose(
            dst=tile_stream.tile(expert_affi_eager_sb, tile_shape=(1, dims.T), tile_dims=(0, 1), has_p_tile_dim=False),
            src=tile_stream.tile(affi_eager_src, tile_shape=(dims.T, 1), has_p_tile_dim=False),
        ).execute()
        expert_affi_eager_sb = expert_affi_eager_sb.reshape((dims._pmax, K_sharded * dims.T))
        stream_shuffle_broadcast(src=expert_affi_eager_sb, dst=expert_affi_eager_sb)
        expert_affi_eager_sb = expert_affi_eager_sb.reshape((dims._pmax, K_sharded, dims.T))

    else:
        gathered_affinities_sb = gather_expert_affinities(expert_affinities_sb, expert_idx, dims, auto_sbm)
        expert_affinity_sb = nl.ndarray((_pmax, dims.T, dims.K), dtype=gathered_affinities_sb.dtype, buffer=nl.sbuf)
        for i_t in range(dims.T):
            # Set SBM prefix to deduplicate
            auto_sbm.set_name_prefix(f"T{i_t}_")

            # In the new FE, t[:, i_t, :] is 3D instead of 2D. Reshape as a workaround
            expert_affinity_sb = expert_affinity_sb.reshape((_pmax, dims.T * dims.K))
            broadcast_token_affinity(
                dst=expert_affinity_sb[:, i_t * dims.K : (i_t + 1) * dims.K],
                gathered_affinities_sb=gathered_affinities_sb,
                token_index=i_t,
                dims=dims,
                sbm=auto_sbm,
            )
            expert_affinity_sb = expert_affinity_sb.reshape((_pmax, dims.T, dims.K))
        # Reset SBM prefix
        auto_sbm.set_name_prefix("")

    p_I = _pmax if dims.I > 512 else dims.I // 4

    # The SBUF scale tensor for gate_up/down projection can be reused by different expert iterations to avoid redundant memset
    gate_up_scale_sb = nl.ndarray(
        (_pmax, 2, n_H512_tile_sharded, dims.I), dtype=nl.uint8, buffer=nl.sbuf, name='gate_up_w_scale_sb'
    )
    nisa.memset(dst=gate_up_scale_sb, value=0)
    down_scale_sb = nl.ndarray((_pmax, n_I512_tile, dims.H_shard), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=down_scale_sb, value=0)

    for i_T_tile in range(n_T_tile):
        # For down proj with [T, H] layout, all four (at most) token outputs will write to the same output_temp on each of the four quadrants,
        # then one local CC + one DMA store is needed for saving these four (at most) outputs.
        for i_T_sub_tile in range(sz_T_tile):
            # Get true token index
            i_t = i_T_tile * sz_T_tile + i_T_sub_tile

            # Even with static ranges, NKI has undefined behaviour when using breaks
            if i_t < dims.T:
                inp_qtz_cur_t = nl.ndarray((_pmax, n_H512_tile_sharded, 4), dtype=inp_qtz.dtype, buffer=nl.sbuf)
                inp_scale_cur_t = nl.ndarray((_pmax, n_H512_tile_sharded, 4), dtype=inp_scale.dtype, buffer=nl.sbuf)
                nisa.memset(dst=inp_scale_cur_t, value=0)
                nisa.tensor_copy(
                    dst=inp_qtz_cur_t.ap(
                        pattern=[[n_H512_tile_sharded * 4, _pmax], [4, n_H512_tile_sharded]], offset=0, dtype=nl.float32
                    ),
                    src=inp_qtz.ap(
                        pattern=[[n_H512_tile_sharded * T_padded, _pmax], [T_padded, n_H512_tile_sharded]],
                        offset=i_t,
                        dtype=nl.float32,
                    ),
                    engine=nisa.vector_engine,
                )
                nisa.tensor_copy(
                    dst=inp_scale_cur_t[:, :, :1], src=inp_scale[:, :, i_t : i_t + 1], engine=nisa.vector_engine
                )

                for i_k in range(K_sharded):
                    i_k_lnc_adjusted = (i_k + shard_id * K_sharded) if shard_on_K else i_k

                    # Gate and Up projection
                    _process_fused_gate_up_projection_mxfp4(
                        hidden=inp_qtz_cur_t,  # [_pmax, n_H512_tile_sharded, 4_padded_from_1_t]
                        hidden_scale=inp_scale_cur_t,  # [_pmax, n_H512_tile_sharded, 4_padded_from_1_t]
                        gate_up_weights=params.gate_proj_weights_tensor,  # [E, _pmax, 2, n_H512_tiles, I]
                        gate_up_scale=params.quant_params.gate_w_scale,  # [E, _pmax // _q_height, 2, n_H512_tiles, I]
                        gate_up_bias=params.bias_params.gate_proj_bias_tensor,  # [E, I_p, 2, ceil(I/512), 4], I_p = I//4 if I <= 512 else _pmax
                        p_idx_vector=p_idx_vector_gup[
                            :, i_t, i_k : i_k + 1
                        ],  # prepared expert idx for vec dge, note it only contains data for cur K-shard
                        gate_up_scale_sb=gate_up_scale_sb,  # [_pmax, 2, n_H512_tile_sharded, dims.I]
                        output=intermediate_state_sb,  # [_pmax, ceil(I/512), 4_padded_from_1_t, _q_width]
                        attrs=params,
                        dims=dims,
                        gate_up_weights_E_offset=expert_idx.ap(
                            pattern=[[dims.K, 1], [1, 1]], offset=i_t * dims.K + i_k_lnc_adjusted
                        ),
                        gate_up_bias_E_offset=expert_idx.ap(
                            pattern=[[dims.K, 1], [1, 1]], offset=i_t * dims.K + i_k_lnc_adjusted
                        ),
                    )

                    # Efficiently load down projection scales using vector DGE indexing
                    scale_shape = params.quant_params.down_w_scale.shape

                    down_scale_view = params.quant_params.down_w_scale.reshape(
                        (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3])
                    )

                    n_quadrants_needed, n_remaining_partition = divmod(p_I, 32)
                    n_remaining_partition = n_remaining_partition // _q_height

                    # Handle remaining partitions if they exist
                    token_indices_on_p = nl.ndarray((_pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=token_indices_on_p, src=p_idx_vector_down[:, i_t, i_k])
                    nisa.dma_copy(
                        dst=down_scale_sb,
                        src=down_scale_view.ap(
                            pattern=[[n_I512_tile * dims.H, _pmax], [dims.H, n_I512_tile], [1, dims.H_shard]],
                            offset=0 if shard_on_K else (dims.shard_id * dims.H_shard),
                            vector_offset=token_indices_on_p,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                    )

                    # Load down proj weights into [I0, ceil(I/512), H_sharded] NOTE: this is pre-quantized and each elt is mx_x4 (packed I)
                    down_weight_qtz_sb = nl.ndarray(
                        (_pmax, n_I512_tile, dims.H_shard),
                        dtype=params.down_proj_weights_tensor.dtype,
                        buffer=nl.sbuf,
                    )
                    # Memset weight if input weight HBM does not pad on par dim
                    if p_I != _pmax:
                        nisa.memset(dst=down_weight_qtz_sb[:, n_I512_tile - 1, :], value=0)

                    # down_proj_weights_tensor shape: (E, p_I, n_I512_tile, H)
                    H_offset = 0 if shard_on_K else (dims.shard_id * dims.H_shard)
                    expert_scalar = expert_idx.slice(dim=0, start=i_t, end=i_t + 1).slice(
                        dim=1, start=i_k_lnc_adjusted, end=i_k_lnc_adjusted + 1
                    )
                    down_weights_view = (
                        (params.down_proj_weights_tensor)
                        .select(dim=0, index=expert_scalar)
                        .slice(dim=2, start=H_offset, end=H_offset + dims.H_shard)
                    )
                    nisa.dma_copy(
                        dst=down_weight_qtz_sb[:p_I, :, :],
                        src=down_weights_view,
                        dge_mode=nisa.dge_mode.hwdge,
                    )
                    down_weight_qtz_sb = down_weight_qtz_sb.view(params.down_proj_weights_tensor.dtype)

                    # Call down proj with out_p_offset (to ensure the data in cur_down_out is on the same partition as output_temp)
                    down_cfg = ProjConfig(
                        H=dims.H_shard,
                        I=dims.I,
                        BxS=4,
                        n_prgs=1,
                        prg_id=0,  # perform as LNC1
                        out_p_offset=0,
                    )
                    cur_down_out = _down_projection_mx_tp_shard_H(
                        inter_sb=intermediate_state_sb,
                        weight=down_weight_qtz_sb,
                        weight_scale=down_scale_sb,
                        bias_sb=None,  # NOTE: we assert down swap layout, postpone down bias to after down projection
                        cfg=down_cfg,
                    )  # NOTE: only the first 1_T partition has value

                    # cur_down_out has shape [H0, H1_shard, 4], slice out the part that has value, shape [H0, H1]
                    cur_down_out_view = cur_down_out[:, :, 0]

                    # Apply affinity and accumulate to SB
                    # output_temp shape: [H0, T, H1_shard]
                    cur_affinity = (
                        expert_affinity_sb[:, i_t, i_k_lnc_adjusted]
                        if params.expert_params.expert_affinities_eager == None
                        else expert_affi_eager_sb[:, i_k, i_t]
                    )
                    if i_k == 0:
                        nisa.tensor_scalar(
                            dst=output_temp[:, i_t, :], data=cur_down_out_view, op0=nl.multiply, operand0=cur_affinity
                        )
                    else:
                        nisa.scalar_tensor_tensor(
                            dst=output_temp[:, i_t, :],
                            data=cur_down_out_view,
                            op0=nl.multiply,
                            operand0=cur_affinity,
                            op1=nl.add,
                            operand1=output_temp[:, i_t, :],
                        )

        # If we shard on K, reduce result between two NCs
        if shard_on_K and (num_shards > 1):
            output_temp_recv = nl.ndarray(output_temp.shape, dtype=output_temp.dtype, buffer=nl.sbuf)
            nisa.sendrecv(
                src=output_temp,
                dst=output_temp_recv,
                send_to_rank=(1 - shard_id),
                recv_from_rank=(1 - shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=output_temp, data1=output_temp, data2=output_temp_recv, op=nl.add)

        # Now we have all tokens processed with all K experts, we shard on T to transpose and save (with optionally adding final down proj bias).
        T_sharded, T_has_remainder = dims.T // num_shards, (dims.T % num_shards > 0)

        # Compute expert_affinities scaled down projection bias with the pseudo code below:
        # (1) weighted_bias[T, H] = expert_affinities[T, E] @ down_bias[E, H]
        # (2) output_temp[T, H] += weighted_bias[T, H]
        if params.bias_params.down_proj_bias_tensor != None:
            kernel_assert(
                (dims.E <= _pmax),
                "MXFP4 down projection with LHS/RHS swapped only supports E <= 128 when down projection bias exists",
            )
            down_bias_sb = nl.ndarray(
                (dims.E, dims.H), dtype=params.bias_params.down_proj_bias_tensor.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=down_bias_sb, src=params.bias_params.down_proj_bias_tensor)

            # Transpose and down cast the expert affinities from FP32 to the same dtype as the down proj bias
            expert_affinity_tp = nl.ndarray(
                (dims.E, dims.T), dtype=params.bias_params.down_proj_bias_tensor.dtype, buffer=nl.sbuf
            )
            blas.transpose(
                dst=expert_affinity_tp,
                src=expert_affinities_sb[0 : dims.T, 0 : dims.E],
                src_has_p_tile_dim=False,
                dst_has_p_tile_dim=False,
            )

            # perform matmul with expert_affinity_tp to be LHS and down_bias_sb to be RHS
            # result has the layout of [dims.H0, dims.T, dims.H1_shard] layout to match output_temp
            scaled_bias_psum = nl.ndarray((dims.H0, dims.H1_shard, dims.T), dtype=nl.float32, buffer=nl.psum)
            for i_h1 in range(dims.H1_shard):
                nisa.nc_matmul(
                    dst=scaled_bias_psum[0 : dims.H0, i_h1, 0 : dims.T],
                    stationary=down_bias_sb[0 : dims.E, i_h1 * dims.H0 : (i_h1 + 1) * dims.H0],
                    moving=expert_affinity_tp[0 : dims.E, 0 : dims.T],
                )

            for i_t in range(dims.T):
                nisa.tensor_tensor(
                    dst=output_temp[:, i_t, :],
                    data1=output_temp[:, i_t, :],
                    data2=scaled_bias_psum[:, :, i_t],
                    op=nl.add,
                )

        t_start = shard_id * div_ceil(dims.T, num_shards)
        t_end = div_ceil(dims.T, num_shards) if shard_id == 0 else dims.T

        output_temp_tp = tile_stream.alloc_logical(
            (dims.H1_shard, dims.T, dims.H0), dims.H1_shard, dtype=output_temp.dtype
        )
        if t_end > t_start:
            # Transpose output since down proj is lhs/rhs swapped and producing HT layout
            blas.Transpose(
                dst=tile_stream.tile(
                    output_temp_tp.slice(dim=2, start=t_start, end=t_end),
                    tile_shape=(dims.H1_shard, dims.H0),
                ),
                src=tile_stream.tile(
                    output_temp.slice(dim=1, start=t_start, end=t_end),
                    tile_shape=(dims.H0, dims.H1_shard),
                    has_p_tile_dim=False,
                ),
            ).execute()

            # Store transposed output to HBM — output[T, H] reshaped as [T, H1, H0] to match SBUF tile shape
            dma.Store(
                dst=tile_stream.tile(
                    output.reshape_dim(dim=1, shape=(dims.H1, dims.H0)).slice(dim=0, start=t_start, end=t_end),
                    tile_shape=(dims.H1_shard, dims.H0),
                    tile_dims=(1, 2),
                ),
                src=tile_stream.tile(
                    output_temp_tp.slice(dim=2, start=t_start, end=t_end),
                    tile_shape=(dims.H1_shard, dims.H0),
                ),
            ).execute()

    auto_sbm.close_scope()
    return output


def _layout_adapter_hbm(src: nl.NkiTensor, n_prgs: int, prg_id: int):
    """
    Load and transpose input tensor from HBM to SBUF with swizzled layout.

    Performs the following transformations:
    1. Input layout in HBM: [T, H] with internally shuffled layout [T, 4_H, H/512, 16_H, 8_H]
    2. Load input to SBUF: [4_T * 4_H (P), T/4, H/512, 16_H * 8_H]
    3. Perform T/4 * H/512 transpose operations to swap outermost and innermost dims
    4. Obtain swizzle layout: [16_H * 8_H(P), H/512, T/4, 4_T * 4_H]

    Args:
        src (nl.NkiTensor): [T, H], 5D tensor in HBM with internally shuffled layout [T, 4_H, H/512, 16_H, 8_H].
        n_prgs (int): Number of programs.
        prg_id (int): Program ID.

    Returns:
        result (nl.NkiTensor): [16_H * 8_H(P), H/512, ceil(T/4) * 4, 4_H], 4D tensor in SBUF with swizzled layout.
    """
    _q_width = 4
    _pmax = nl.tile_size.pmax

    # Check for shape
    kernel_assert(len(src.shape) == 2, f"expect input to be of the shape [T, H], got {src.shape}")
    T, H = src.shape
    kernel_assert(H % 512 == 0, f"Expect H to be a multiple of 512, got {H}")
    H_div_512 = H // 512 // n_prgs
    T_div_4 = div_ceil(T, 4)
    T_padded = T_div_4 * 4

    # [16_H * 8_H(P), H/512, T/4, 4_T * 4_H]
    result = tile_stream.alloc_logical((H_div_512 * _pmax, T_padded * _q_width), _pmax, dtype=src.dtype)
    # [T, 4_H, H_div_512 * 16_H * 8_H]
    src = src.reshape_dim(dim=1, shape=(_q_width, n_prgs, H_div_512 * _pmax)).select(2, prg_id)
    # [4_T * 4_H (P), T/4, H/512, 16_H * 8_H]
    src_sbuf = tile_stream.alloc_logical((T_padded * _q_width, H_div_512 * _pmax), 4 * _q_width, dtype=src.dtype)
    nisa.memset(dst=src_sbuf, value=0.0)

    """
    Load [4_T * 4_H (P), T/4, H/512, 16_H * 8_H]@SBUF
    from [T, 4_H, H/512, 16_H, 8_H]@HBM.
    """
    dma.Load(
        dst=tile_stream.tile(
            tensor=src_sbuf,
            tile_shape=(1, H_div_512 * _pmax),
            iter_order=ViewOrder(ViewSpec().reshape_dim(0, (T_div_4, 4, 4))),
            logical_p=T * _q_width,
        ),
        src=tile_stream.tile(
            tensor=src,
            tile_shape=(1, H_div_512 * _pmax),
            tile_dims=(0, 2),
            iter_order=ViewOrder(ViewSpec().reshape_dim(0, (T_div_4, 4))),
        ),
    ).execute()

    # transpose [4_T * 4_H, 16_H*8_H] -> [16_H*8_H, 4_T * 4_H]
    blas.Transpose(
        dst=tile_stream.tile(result, (_pmax, 4 * _q_width), iter_order=ColMajor()),
        src=tile_stream.tile(src_sbuf, (4 * _q_width, _pmax), iter_order=RowMajor()),
    ).execute()

    return result


def _process_fused_gate_up_projection_mxfp4(
    hidden: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    gate_up_weights: nl.NkiTensor,
    gate_up_scale: nl.NkiTensor,
    gate_up_bias: nl.NkiTensor,
    p_idx_vector: nl.NkiTensor,
    gate_up_scale_sb: nl.NkiTensor,
    output: nl.NkiTensor,
    attrs: MLPParameters,
    dims,
    gate_up_weights_E_offset: Optional[nl.NkiTensor],
    gate_up_bias_E_offset: Optional[nl.NkiTensor],
):
    """
    Process gate and up projection, including the activation of gate projection and the final elem-wise multiply:
        output = act_fn(clamp(gate_proj(hidden))) * clamp(up_proj(hidden)).
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
    base_weight = gate_up_weights
    weight_sb = nl.ndarray((_pmax, 2, n_H512_tile_sharded, dims.I), dtype=base_weight.dtype, buffer=nl.sbuf)
    if gate_up_weights_E_offset is None:
        nisa.dma_copy(
            dst=weight_sb,
            src=base_weight[:, :, shard_id : (shard_id + 1) * n_H512_tile_sharded, :],
            dge_mode=nisa.dge_mode.swdge,
        )
    else:
        # gate_up_weights shape: (E, _pmax, 2, n_H512_tiles, I)
        gate_up_weights_view = base_weight.select(dim=0, index=gate_up_weights_E_offset).slice(
            dim=2, start=shard_id * n_H512_tile_sharded, end=(shard_id + 1) * n_H512_tile_sharded
        )
        nisa.dma_copy(dst=weight_sb, src=gate_up_weights_view)
    weight_sb = weight_sb.view(gate_up_weights.dtype)

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
    # Wrap tensors in nl.NkiTensor and use slice() for dimension 1
    hidden_tv = hidden
    hidden_scale_tv = hidden_scale
    weight_tv = weight_sb
    scale_tv = gate_up_scale_sb
    bias_tv = bias_sb

    gate_proj_out_sb = _gate_up_projection_mx_tp_shard_H_primitives(
        hidden_qtz_sb=hidden_tv,
        hidden_scale_sb=hidden_scale_tv,
        weight_qtz_tv=weight_tv.slice(dim=1, start=0, end=n_H512_tile_sharded),
        weight_scale_tv=scale_tv.slice(dim=1, start=0, end=n_H512_tile_sharded),
        bias_sb=bias_tv,
        cfg=gate_proj_cfg,
    )  # bf16[_pmax, n_I512_tile, T, _q_width]
    up_proj_out_sb = _gate_up_projection_mx_tp_shard_H_primitives(
        hidden_qtz_sb=hidden_tv,
        hidden_scale_sb=hidden_scale_tv,
        weight_qtz_tv=weight_tv.slice(dim=1, start=n_H512_tile_sharded, end=2 * n_H512_tile_sharded),
        weight_scale_tv=scale_tv.slice(dim=1, start=n_H512_tile_sharded, end=2 * n_H512_tile_sharded),
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


def _gate_up_projection_mx_tp_shard_H_primitives(
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scale_sb: nl.NkiTensor,
    weight_qtz_tv: nl.NkiTensor,
    weight_scale_tv: nl.NkiTensor,
    bias_sb: nl.NkiTensor,
    cfg: ProjConfig,
) -> nl.NkiTensor:
    """
    Primitives version of gate_up_projection_mx_tp_shard_H using blas.Matmul.

    Same semantics as gate_up_projection_mx_tp_shard_H but expects weight/scale
    already loaded into SBUF (no HBM loading path).

    :param hidden_qtz_sb: mxfp8_x4[_pmax, n_H512_tile_sharded, BxS] @ SB.
    :param hidden_scale_sb: uint8[_pmax, n_H512_tile_sharded, BxS] @ SB.
    :param weight_qtz_tv: nl.NkiTensor of mxfp_x4[_pmax, n_H512_tile_sharded, I] @ SB.
    :param weight_scale_tv: nl.NkiTensor of uint8[_pmax, n_H512_tile_sharded, I] @ SB.
    :param bias_sb [OPTIONAL]: nl.NkiTensor of bf16[_pmax, n_I512_tile, _q_width] @ SB.
    :return: bf16[_pmax, ceil(I / 512), BxS, _q_width] @ SB.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    I, BxS = cfg.I, cfg.BxS

    BxS_tile_sz = min(BxS, _psum_fmax * 2 // _q_width)
    I128_tile_sz = I // (_q_width * cfg.n_total_I512_tile) if cfg.n_total_I512_tile > 0 else _pmax

    # Allocate with _pmax partitions (downstream expects _pmax), flat F = BxS * _q_width
    out_sb = tile_stream.alloc_logical(
        (_pmax * cfg.n_total_I512_tile, BxS * _q_width),
        pdim_size=_pmax,
        dtype=nl.bfloat16,
    )
    # Slice to I128_tile_sz partitions before tiling so dst P matches stationary F
    out_sb_sliced = out_sb.slice(dim=0, start=0, end=I128_tile_sz)

    # Tile stationary weight: [_pmax, n_H512_tile_sharded, I] → tiles of (_pmax, I128_tile_sz)
    # K = P dim (contraction via _pmax partitions), stationary free = I in I128 chunks
    stat_ts = tile_stream.tile(
        weight_qtz_tv,
        tile_shape=(_pmax, I128_tile_sz),
        tile_dims=(0, 1),
        iter_order=ColMajor(),
    )
    stat_scale_ts = tile_stream.tile(
        weight_scale_tv,
        tile_shape=(_pmax, I128_tile_sz),
        tile_dims=(0, 1),
        iter_order=ColMajor(),
    )

    # Tile moving hidden: [_pmax, n_H512_tile_sharded, BxS] → tiles of (_pmax, BxS_tile_sz)
    # K = P dim (contraction via _pmax partitions), moving free = BxS
    mov_ts = tile_stream.tile(
        hidden_qtz_sb,
        tile_shape=(_pmax, BxS_tile_sz),
        tile_dims=(0, 1),
        iter_order=ColMajor(),
    )
    mov_scale_ts = tile_stream.tile(
        hidden_scale_sb,
        tile_shape=(_pmax, BxS_tile_sz),
        tile_dims=(0, 1),
        iter_order=ColMajor(),
    )

    # Tile dst: [_pmax, n_total_I512_tile, BxS, _q_width]
    # Output P = stationary F = I128_tile_sz (may be < _pmax when I < 512)
    # Pack factor = _q_width = 4: 4 stationary I128 tiles pack into one dst tile
    dst_ts = tile_stream.tile(
        out_sb_sliced,
        tile_shape=(I128_tile_sz, _q_width * BxS_tile_sz),
    )

    # Bias tiling: extract gate or up slice, then tile as (_pmax, _q_width) per I512 tile
    bias_ts = None
    if (bias_sb is not None) and (prg_id == 0):
        bias_view = bias_sb
        if cfg.bias_t_shared_between_gate_up:
            gate_or_up_idx = 0 if cfg.bias_t_shared_base_offset == 0 else 1
            bias_view = bias_view.select(dim=1, index=gate_or_up_idx)
        # Pre-slice to I128_tile_sz partitions so tile base has correct pdim
        bias_view = bias_view.slice(dim=0, start=0, end=I128_tile_sz)
        # bias_view is now [I128_tile_sz, n_I512_tile, _q_width]
        bias_ts = tile_stream.tile(
            bias_view,
            tile_shape=(I128_tile_sz, _q_width),
            tile_view=view().reshape_dim(1, (1, _q_width)).broadcast(1, BxS_tile_sz),
        )

    blas.Matmul(
        dst=dst_ts,
        moving=mov_ts,
        stationary=stat_ts,
        moving_scale=mov_scale_ts,
        stationary_scale=stat_scale_ts,
        bias=bias_ts,
        psum_evict_view=view().reshape_dim(1, (_q_width, BxS_tile_sz)).permute((0, 2, 1)),
    ).execute()

    # Unwrap to base ndarray, reshape to downstream expected layout
    out_sb = out_sb.reshape((_pmax, cfg.n_total_I512_tile, BxS, _q_width))

    # Zero unused partitions to prevent NaN from uninitialized SBUF
    if cfg.zero_unused_partitions and I128_tile_sz < _pmax:
        for i_I512_tile in range(cfg.n_total_I512_tile):
            cur_I_pdim_sz = min(_pmax, I // _q_width - i_I512_tile * _pmax)
            if cur_I_pdim_sz < _pmax:
                nisa.memset(out_sb[cur_I_pdim_sz:_pmax, i_I512_tile, :, :], value=0.0)

    # Receive projection output from the other NC when LNC > 1
    if n_prgs > 1:
        recv = nl.ndarray(out_sb.shape, dtype=out_sb.dtype, buffer=nl.sbuf)
        nisa.sendrecv(src=out_sb, dst=recv, send_to_rank=(1 - prg_id), recv_from_rank=(1 - prg_id), pipe_id=0)
        nisa.tensor_tensor(dst=out_sb, data1=out_sb, data2=recv, op=nl.add)

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


def _down_proj_prep_inter_and_weights(
    inter_sb: nl.NkiTensor, weight: nl.NkiTensor, weight_scale: nl.NkiTensor, cfg: ProjConfig
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Prep intermediate and weights for down projection:
        - for intermediate, reshape and quantize (and reshape back);
        - for weight, load from HBM into SBUF.
    """
    n_prgs, prg_id = cfg.n_prgs, cfg.prg_id
    H, I, BxS = cfg.H, cfg.I, cfg.BxS
    H_sharded = H // n_prgs
    p_I = _pmax if I > _psum_fmax else I // _q_width

    # Quantize intermediate state to MXFP8
    inter_sb = inter_sb.reshape((_pmax, cfg.n_total_I512_tile * BxS * _q_width))
    inter_qtz = nl.ndarray((_pmax, cfg.n_total_I512_tile * BxS), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    inter_qtz_scale = nl.ndarray(inter_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=inter_qtz, src=inter_sb, dst_scale=inter_qtz_scale)
    inter_qtz = inter_qtz.reshape((_pmax, cfg.n_total_I512_tile, BxS))
    inter_qtz_scale = inter_qtz_scale.reshape(inter_qtz.shape)

    if cfg.dbg_hidden:
        return inter_qtz, inter_qtz_scale, None, None

    weight_qtz = None
    if weight.buffer == nl.sbuf:
        weight_qtz = weight
    else:
        weight_qtz = nl.ndarray(
            (_pmax, cfg.n_total_I512_tile, H_sharded), dtype=weight.dtype, buffer=nl.sbuf, name='down_w_qtz_sb'
        )
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz[:, cfg.n_total_I512_tile - 1, :], value=0.0)

        kernel_assert(weight.shape == (p_I, cfg.n_total_I512_tile, H), "Incorrect weight shape")
        nisa.dma_copy(
            src=weight[:, :, prg_id * H_sharded : (prg_id + 1) * H_sharded],
            dst=weight_qtz[:p_I, :, :],
            dge_mode=nisa.dge_mode.hwdge,
        )

    weight_qtz_scale = None
    if weight_scale.buffer == nl.sbuf:
        kernel_assert(
            weight_scale.shape == (_pmax, cfg.n_total_I512_tile, H_sharded),
            f"Expect weight_scale in SBUF to have the shape of ({_pmax}, {cfg.n_total_I512_tile}, {H_sharded}), got {weight_scale.shape}",
        )
        weight_qtz_scale = weight_scale
    else:
        weight_qtz_scale = nl.ndarray(weight_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf, name="down_w_scale_sb")
        if p_I != _pmax:
            nisa.memset(dst=weight_qtz_scale[:, cfg.n_total_I512_tile - 1, :], value=0)

        n_quadrants_needed = _pmax // SBUF_QUADRANT_SIZE
        for i_quad in range(n_quadrants_needed):
            kernel_assert(weight_scale.shape == (p_I // _q_height, cfg.n_total_I512_tile, H), "Incorrect weight shape")
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


def _down_projection_mx_tp_shard_H(
    inter_sb: nl.NkiTensor,
    weight: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    cfg: ProjConfig,
    partial_output: bool = False,
) -> nl.NkiTensor:
    """
    Performs the Down projection with H-dimension sharding. Math (Neuron matmul):
        inter_sb (moving) [I, BxS] @ weight (stationary) [I, H] → [H, BxS].
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
                moving=inter_qtz[:, i_I512_tile, :],  # [_pmax (I), BxS]
                stationary_scale=weight_qtz_scale[:, i_I512_tile, i_H1 * _pmax : (i_H1 + 1) * _pmax],
                moving_scale=inter_qtz_scale[:, i_I512_tile, :],
            )

        # Copy out the current H128 tile to SB, use ACT because DVE is usually bottlenecked
        act_bias_arg = None
        if bias_sb is not None:
            act_bias_arg = bias_sb[:, i_H1]
        idx = i_H1 if partial_output else cfg.H1_sharded * prg_id + i_H1
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

    return out_sb
