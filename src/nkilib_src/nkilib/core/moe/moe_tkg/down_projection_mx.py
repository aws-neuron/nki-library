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
Down projection sub-kernels with LNC sharding support.

Supports multiple LNC sharding strategies (see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py):
- NO_SHARD: No sharding, each NC computes full result independently. Used when LNC=1.
- SHARD_I: Shard on I (intermediate) dimension. Default for most workloads.
- SHARD_T: Shard on T (token) dimension. Useful when T is large.
- TODO: SHARD_E: Shard on E (expert) dimension. When E_L is divisible by 2 and T is large,
  better to shard on E than I because we can support higher TP and get better DMA throughput
  by loading larger packets.

These sub-kernels can be used by any algorithm that requires LNC-sharded down projection,
including all-expert, selective-load, or custom MoE implementations.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

# Common utils
from ...utils.common_types import ExpertAffinityScaleMode, MoELNCShardingStrategy
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...utils.tensor_view import as_nki_tensor
from .all_expert_mx_utils import SUPPORTED_MOE_SHARDING_STRATEGIES

# Shared MX constants
from .projection_mx_constants import (
    NUM_QUADRANTS_IN_SBUF,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
)


def load_broadcast_down_weight_scale_bias(
    weight: nl.NkiTensor,
    scale: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    expert_idx: int,
    H: int,
    tile_I: int,
    n_I512_tiles: int,
    tile_offset: int,
    tile_T: int,
    activation_compute_dtype=nl.bfloat16,
    use_PE_bias_broadcast: bool = True,
    sharding_strategy: MoELNCShardingStrategy = MoELNCShardingStrategy.SHARD_I,
    skip_scale_load: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor, Optional[nl.NkiTensor]]:
    """
    Load down projection weight, scale, and bias (optional) for one expert using static DMA.

    When executed with LNC=2, weights and scales are sharded on I dimension. Bias is sharded on H dimension,
    with NC0 loading the first half of H and NC1 loading the second half.

    Args:
        weight (nl.NkiTensor): [E_L, 128_I, I/512, H], Down projection weight tensor from HBM (4_I packed in x4 dtype).
        scale (nl.NkiTensor): [E_L, 16_I, I/512, H], Down projection MX scale tensor from HBM (uint8 MX scales).
        bias (Optional[nl.NkiTensor]): [E_L, H], Optional down projection bias tensor from HBM.
        expert_idx (int): Index of the current expert to load.
        H (int): Hidden dimension size.
        tile_I (int): Tile size for I dimension (typically 128).
        n_I512_tiles (int): Number of I/512 tiles to load (local shard size when LNC=2).
        tile_offset (int): Starting tile offset for NC's tiles (pre-computed for tile-based sharding).
        tile_T (int): Tile size for T dimension (for bias broadcast).
        activation_compute_dtype: Data type for bias buffer (default: nl.bfloat16).
        use_PE_bias_broadcast (bool): If True, use PE (matmul with ones) for bias broadcast; else use DVE
            stream_shuffle_broadcast.
        sharding_strategy (MoELNCShardingStrategy): LNC sharding strategy. Determines bias H-sharding behavior.

    Returns:
        weight_sb (nl.NkiTensor): [128_I, n_I512_tiles, H], Weight in SBUF (4_I packed in x4 dtype).
        scale_sb (nl.NkiTensor): [128_I, n_I512_tiles, H], Scales in SBUF (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.NkiTensor]): [tile_T, H], Broadcasted bias in SBUF (zeros when bias=None, sharded on H
            when LNC=2).

    Notes:
        - tile_offset is pre-computed to ensure alignment with gate_up projection's tile-based I-sharding
        - Based on experiments, static DMA demonstrates better performance
        - Can revert to DGE if HBM out-of-memory (OOM) issues occur
    """

    # Calculate shapes / tiling
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx", (0, 1))
    weight_sb_shape = (tile_I, n_I512_tiles, H)
    bias_sb_shape = (tile_T, H)

    # Allocate buffers
    base_weight = weight
    weight_sb = nl.ndarray(weight_sb_shape, dtype=base_weight.dtype, buffer=nl.sbuf)
    scale_dtype = nl.uint8 if skip_scale_load else scale.dtype
    scale_sb = nl.ndarray(weight_sb_shape, dtype=scale_dtype, buffer=nl.sbuf)
    bias_sb: Optional[nl.NkiTensor] = None

    actual_prg_offset = tile_offset
    I_p_in_hbm = base_weight.shape[1]

    # Load weight: index expert, then slice I/512 tiles
    # Shape: [E_L, I_p, I/512, H] -> [I_p, n_I512_tiles, H] -> padded to [128_I, n_I512_tiles, H]
    if I_p_in_hbm < tile_I:
        nisa.memset(dst=weight_sb[...], value=0)
        weight_view = base_weight.select(dim=0, index=expert_idx).slice(
            dim=1, start=actual_prg_offset, end=actual_prg_offset + n_I512_tiles
        )
        nisa.dma_copy(src=as_nki_tensor(weight_view), dst=weight_sb[:I_p_in_hbm, :, :], dge_mode=nisa.dge_mode.none)
    else:
        weight_view = base_weight.select(dim=0, index=expert_idx).slice(
            dim=1, start=actual_prg_offset, end=actual_prg_offset + n_I512_tiles
        )
        nisa.dma_copy(src=as_nki_tensor(weight_view), dst=weight_sb[...], dge_mode=nisa.dge_mode.none)
    weight_sb = weight_sb.view(weight.dtype)

    """
    Load scale: index expert, then slice I/512 tiles.
    Shape: [E_L, I_p//8, I/512, H] -> [I_p//8, n_I512_tiles, H] -> padded to [128_I, n_I512_tiles, H]
    Note: scales have I_p//8 (not 128), need to map to first 4 partitions of each quadrant.
    Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer.
    Skipped when skip_scale_load=True (SW quant): caller passes a shared 2D [128, H] dummy tile directly.
    """
    if not skip_scale_load:
        I_p_scale_in_hbm = scale.shape[1]
        n_quadrants_needed = div_ceil(I_p_scale_in_hbm, SCALE_P_ELEM_PER_QUADRANT)

        if I_p_scale_in_hbm < tile_I // 8:
            for quadrant_idx in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
                nisa.memset(
                    dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT), :, :], value=0.0
                )

        for quadrant_idx in nl.affine_range(n_quadrants_needed):
            actual_scale_p = min(SCALE_P_ELEM_PER_QUADRANT, I_p_scale_in_hbm - SCALE_P_ELEM_PER_QUADRANT * quadrant_idx)
            if actual_scale_p > 1:
                scale_view = (
                    scale.select(dim=0, index=expert_idx)
                    .slice(
                        dim=0,
                        start=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx,
                        end=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx + actual_scale_p,
                    )
                    .slice(dim=1, start=actual_prg_offset, end=actual_prg_offset + n_I512_tiles)
                )
                nisa.dma_copy(
                    src=scale_view,
                    dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, actual_scale_p), :, :],
                    dge_mode=nisa.dge_mode.none,
                )
            else:
                hbm_p_idx = SCALE_P_ELEM_PER_QUADRANT * quadrant_idx
                sb_p_idx = SBUF_QUADRANT_SIZE * quadrant_idx
                scale_f_per_partition = n_I512_tiles * H
                expert_stride = I_p_scale_in_hbm * scale_f_per_partition
                hbm_offset = expert_idx * expert_stride + hbm_p_idx * scale_f_per_partition + actual_prg_offset * H
                nisa.dma_copy(
                    src=scale.ap(
                        pattern=[[scale_f_per_partition, 1], [1, scale_f_per_partition]],
                        offset=hbm_offset,
                    ),
                    dst=scale_sb.ap(
                        pattern=[[scale_f_per_partition, 1], [1, scale_f_per_partition]],
                        offset=sb_p_idx * scale_f_per_partition,
                    ),
                    dge_mode=nisa.dge_mode.none,
                )

    """
    Load + broadcast bias, sharding on H dim when LNC=2.
    In LNC=2, NC0 bias_sb will have first half of H filled with bias,
    second half with zeros; NC1 will have the inverse.
    Shape: [E_L, H] -> [1, H_size_local]
    """
    if bias != None:
        bias_sb = nl.ndarray(bias_sb_shape, dtype=activation_compute_dtype, buffer=nl.sbuf)
        H_size_local = H // 2 if (sharding_strategy == MoELNCShardingStrategy.SHARD_I and n_prgs > 1) else H
        H_offset = H_size_local * prg_id if (sharding_strategy == MoELNCShardingStrategy.SHARD_I and n_prgs > 1) else 0
        if H_size_local < H:
            other_H_offset = H_size_local * (1 - prg_id)
            nisa.memset(dst=bias_sb[:, nl.ds(other_H_offset, H_size_local)], value=0.0, engine=nisa.gpsimd_engine)
        else:
            nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)
        H_slice_local = nl.ds(H_offset, H_size_local)
        bias_view = bias.slice(dim=0, start=expert_idx, end=expert_idx + 1).slice(
            dim=1, start=H_offset, end=H_offset + H_size_local
        )
        nisa.dma_copy(src=as_nki_tensor(bias_view), dst=bias_sb[0:1, H_slice_local], dge_mode=nisa.dge_mode.none)

        # Broadcast bias using PE
        if use_PE_bias_broadcast:
            is_bias_16bit = activation_compute_dtype in (nl.bfloat16, nl.float16)
            psum_fmax = nl.tile_size.psum_fmax * 2 if is_bias_16bit else nl.tile_size.psum_fmax
            psum_dtype = activation_compute_dtype if is_bias_16bit else nl.float32
            H_tile_size_local = min(H_size_local, psum_fmax)
            ones_mask = nl.ndarray((1, tile_T), dtype=bias_sb.dtype, buffer=nl.sbuf)
            nisa.memset(dst=ones_mask[...], value=1.0, engine=nisa.gpsimd_engine)
            n_H_tiles_local = div_ceil(H_size_local, H_tile_size_local)
            for h_tile_idx in nl.affine_range(n_H_tiles_local):
                h_tile_actual = min(H_tile_size_local, H_size_local - h_tile_idx * H_tile_size_local)
                bias_bc_psum = nl.ndarray((tile_T, h_tile_actual), dtype=psum_dtype, buffer=nl.psum)
                H_tile_slice = nl.ds(H_offset + h_tile_idx * H_tile_size_local, h_tile_actual)
                nisa.nc_matmul(
                    dst=bias_bc_psum,
                    stationary=ones_mask[...],
                    moving=bias_sb[0:1, H_tile_slice],
                    is_stationary_onezero=True,
                )
                nisa.tensor_copy(
                    dst=bias_sb[:, H_tile_slice],
                    src=bias_bc_psum[...],
                    engine=nisa.scalar_engine,
                )

        # Broadcast on DVE
        else:
            stream_shuffle_broadcast(src=bias_sb, dst=bias_sb)

    return weight_sb, scale_sb, bias_sb


def down_projection_mx(
    act_sb: nl.NkiTensor,
    act_scale_sb: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    expert_affinities_masked_sb: nl.NkiTensor,
    expert_idx: int,
    out_sb: nl.NkiTensor,
    out_hbm: Optional[nl.NkiTensor] = None,
    token_position_to_id_T: Optional[nl.NkiTensor] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    activation_compute_dtype=nl.bfloat16,
    is_first_expert: bool = False,
    is_last_expert: bool = False,
    sharding_strategy: MoELNCShardingStrategy = MoELNCShardingStrategy.SHARD_I,
    T_offset: int = 0,
    down_dequant_scale: Optional[nl.NkiTensor] = None,
    down_input_dequant_scale: Optional[nl.NkiTensor] = None,
    output_t_offset: int = 0,
    is_software_quant: bool = False,
    T_physical: int = None,
) -> nl.NkiTensor:
    """
    Computes down projection, expert affinity scaling, expert add, LNC reduction, and SB->HBM spill.
    Supports multiple LNC sharding strategies (see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py).

    Usage:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring mx LNC-sharded down projection

    Args:
        act_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Activation tensor in SBUF (4_I packed in x4 dtype).
        act_scale_sb (nl.NkiTensor): [16_I * 8_I, I/512, T], Activation scales in SBUF
            (in leading 4P of each SBUF quadrant).
        weight_sb (nl.NkiTensor): [16_I * 8_I, I/512, H], Weight tensor in SBUF (4_I packed in x4 dtype).
        weight_scale_sb (nl.NkiTensor): [16_I * 8_I, I/512, H], Weight scales in SBUF
            (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.NkiTensor]): [1, H], Optional bias tensor in SBUF.
        expert_affinities_masked_sb (nl.NkiTensor): [T, E_L] or [128_T, T/128, E_L],
            Expert affinity scores in SBUF.
        expert_idx (int): Index of the current expert.
        out_sb (nl.NkiTensor): [min(T, 128), ⌈T/128⌉, H], Output tensor in SBUF.
        out_hbm (Optional[nl.NkiTensor]): [T, H], Optional output tensor in HBM for spill.
        token_position_to_id_T (Optional[nl.NkiTensor]): [128_T, T/128], Token position indices for indirect
            DMA scatter. When provided, enables blockwise output spill.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Scaling mode for expert affinities.
        activation_compute_dtype: Compute dtype for activations (default: bfloat16).
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.
        sharding_strategy (MoELNCShardingStrategy): LNC sharding strategy.
            Supported: see SUPPORTED_MOE_SHARDING_STRATEGIES in all_expert_mx_utils.py.
        T_offset (int): Offset for T dimension in HBM output (used with direct DMA).
        down_dequant_scale (Optional[nl.NkiTensor]): Dequant scale for down projection.
            STATIC_MX: [tile_T, 1] combined (input * weight) scale. ROW_MX: [tile_T, H//_pmax] per-row weight scale.
        down_input_dequant_scale (Optional[nl.NkiTensor]): [_pmax, T, 1], ROW_MX per-token intermediate dequant scale.
        is_software_quant (bool): When True, weight_scale_sb is a 2D [128, H] dummy tile indexed
            as [:, :TILE_H] instead of the normal 3D [:, tile_i, H_slice]; and act_scale_sb is a 2D [128, T] dummy tile indexed
            as [:, :T] instead of the normal 3D [:, tile_i, T_slice].

    Returns:
        out_sb (nl.NkiTensor): [min(T, 128), ⌈T/128⌉, H], Output tensor in SBUF with accumulated results.
    """

    # Validate sharding strategy is supported (use explicit equality checks for NKI tracing compatibility)
    _is_supported_strategy = (
        (sharding_strategy == MoELNCShardingStrategy.NO_SHARD)
        or (sharding_strategy == MoELNCShardingStrategy.SHARD_I)
        or (sharding_strategy == MoELNCShardingStrategy.SHARD_T)
        or (sharding_strategy == MoELNCShardingStrategy.SHARD_E)
    )
    kernel_assert(
        _is_supported_strategy,
        f"Unsupported sharding strategy: {sharding_strategy}. Supported: {SUPPORTED_MOE_SHARDING_STRATEGIES}",
    )

    # Extract / validate shapes
    TILE_I, n_I512_tiles, T = act_sb.shape
    T_out = T_physical if T_physical is not None else T
    TILE_I_, n_I512_tiles_, H = weight_sb.shape
    kernel_assert(
        TILE_I == TILE_I_, f"Expected same number of partitions in activation and weight, got {TILE_I}, {TILE_I_}"
    )
    kernel_assert(
        n_I512_tiles == n_I512_tiles_,
        f"Expected same number of I tiles in activation and weight, got {n_I512_tiles}, {n_I512_tiles_}",
    )
    kernel_assert(H % 512 == 0, f"Expected H divisible by 512, got {H=}")
    kernel_assert(
        expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE,
        f"Expected expert_affinities_scaling_mode={ExpertAffinityScaleMode.POST_SCALE}, "
        f"got: {expert_affinities_scaling_mode=}",
    )
    kernel_assert(
        out_hbm != None,
        f"Output in SBUF is not yet supported, got out_hbm=None",
    )

    # LNC config
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx", (0, 1))

    # When not sharding on I, treat as single-program for LNC reduction purposes
    need_cross_nc_reduce = (sharding_strategy == MoELNCShardingStrategy.SHARD_I) and (n_prgs > 1)

    # Algorithm + tiling strategy
    pmax = nl.tile_size.pmax
    TILE_T = min(T, pmax)  # T will be partition dim in output
    TILE_H = min(H, nl.tile_size.psum_fmax * 2)  # use 2 * fmax with bf16 PSUM
    n_T128_tiles = div_ceil(T, pmax)
    n_H1024_tiles = H // TILE_H
    need_down_dequant = down_dequant_scale != None
    is_blockwise = token_position_to_id_T != None

    # Cast expert affinities to fp32 for tensor_scalar on scalar engine
    # FIXME[perf]: skip TensorCopy and use strided AP in below TensorScalar when affinities are already fp32
    is_3D_affinities = len(expert_affinities_masked_sb.shape) == 3
    affinity_expert_idx = 0 if is_blockwise else expert_idx
    expert_affinities_masked_fp32_sb = nl.ndarray((TILE_T, n_T128_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=expert_affinities_masked_fp32_sb[...],
        src=expert_affinities_masked_sb[:, :, affinity_expert_idx]
        if is_3D_affinities
        else expert_affinities_masked_sb[:, affinity_expert_idx],
        engine=nisa.scalar_engine,
    )

    # HBM accumulation strategy: when out_sb exceeds 64KB per SBUF partition, it consumes
    # a significant fraction of usable SBUF (trn3: 240KB, trn2: 213KB), constraining the
    # compiler's ability to double-buffer weight/activation loads alongside compute.
    # Switching to HBM accumulation (dma_compute atomic add) eliminates out_sb entirely,
    # freeing SBUF for better DMA/compute overlap.
    #
    # Profiled crossover:
    #   out_sb=28KB  (T=256, H=7168): HBM accum regresses +4% (overhead > benefit)
    #   out_sb=96KB  (T=2048, H=3072): HBM accum improves -36%
    #   out_sb=224KB (T=2048, H=7168): HBM accum improves -40%
    #
    # Blockwise excluded: uses indirect DMA scatter incompatible with contiguous dma_compute.
    # HBM accumulation: eliminates out_sb buffer from SBUF, freeing space for better
    # weight/activation overlap. Only beneficial when out_sb is large enough to constrain
    # SBUF scheduling (>64KB per partition). Uses dma_compute per tile.
    # When out_sb is small, SBUF accumulation is cheaper (no HBM round-trips).
    _HBM_ACCUM_THRESHOLD_BYTES = 64 * 1024
    out_sb_per_partition_bytes = n_T128_tiles * H * 2  # bf16 = 2 bytes
    use_hbm_accumulation = (out_sb_per_partition_bytes > _HBM_ACCUM_THRESHOLD_BYTES) and not is_blockwise

    # Unified loop: always for_h{for_t}. Accumulation strategy selected by heuristic above.
    for tile_h in nl.sequential_range(n_H1024_tiles):
        tile_H_offset = TILE_H * tile_h
        weight_H_slice = nl.ds(tile_H_offset, TILE_H)

        for tile_t in nl.sequential_range(n_T128_tiles):
            tile_T_offset = TILE_T * tile_t
            tile_T_actual = min(TILE_T, T - tile_T_offset)
            tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)

            expert_out_tile_sb = _down_proj_tile_compute(
                act_sb=act_sb,
                act_scale_sb=act_scale_sb,
                weight_sb=weight_sb,
                weight_scale_sb=weight_scale_sb,
                bias_sb=bias_sb,
                expert_affinities_masked_fp32_sb=expert_affinities_masked_fp32_sb,
                down_dequant_scale=down_dequant_scale,
                down_input_dequant_scale=down_input_dequant_scale,
                t_tile_idx=tile_t,
                tile_T_offset=tile_T_offset,
                tile_T_actual=tile_T_actual,
                tile_T_slice=tile_T_slice,
                tile_H_offset=tile_H_offset,
                weight_H_slice=weight_H_slice,
                TILE_T=TILE_T,
                TILE_H=TILE_H,
                n_I512_tiles=n_I512_tiles,
                pmax=pmax,
                need_down_dequant=need_down_dequant,
                activation_compute_dtype=activation_compute_dtype,
                is_software_quant=is_software_quant,
            )

            tile_T_out_actual = min(tile_T_actual, T_out - tile_T_offset)

            if use_hbm_accumulation:
                # HBM accumulation: LNC reduce + write per tile per expert
                hbm_T_slice = nl.ds(T_offset + output_t_offset + TILE_T * tile_t, tile_T_out_actual)
                if need_cross_nc_reduce:
                    H_local = TILE_H // n_prgs
                    H_offset_local = H_local * prg_id
                    _lnc_reduce_and_write(
                        src_send=expert_out_tile_sb[:tile_T_out_actual, nl.ds(H_local * (1 - prg_id), H_local)],
                        src_local=expert_out_tile_sb[:tile_T_out_actual, nl.ds(H_offset_local, H_local)],
                        hbm_dst=out_hbm[hbm_T_slice, nl.ds(tile_H_offset + H_offset_local, H_local)],
                        tile_T_actual=tile_T_out_actual,
                        TILE_T=TILE_T,
                        H_local=H_local,
                        prg_id=prg_id,
                        activation_compute_dtype=activation_compute_dtype,
                        accumulate=not is_first_expert,
                    )
                else:
                    hbm_dst = out_hbm[hbm_T_slice, nl.ds(tile_H_offset, TILE_H)]
                    if is_first_expert:
                        nisa.dma_copy(
                            src=expert_out_tile_sb[:tile_T_out_actual, :], dst=hbm_dst, dge_mode=nisa.dge_mode.none
                        )
                    else:
                        nisa.dma_compute(
                            dst=hbm_dst,
                            srcs=[hbm_dst, expert_out_tile_sb[:tile_T_out_actual, :]],
                            scales=[1.0, 1.0],
                            reduce_op=nl.add,
                        )
            else:
                # SBUF accumulation: accumulate across experts, spill later
                if is_first_expert or is_blockwise:
                    nisa.tensor_copy(
                        dst=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                        src=expert_out_tile_sb[:tile_T_actual, :],
                    )
                else:
                    nisa.tensor_tensor(
                        dst=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                        data1=out_sb[:tile_T_actual, tile_t : tile_t + 1, tile_H_offset : tile_H_offset + TILE_H],
                        op=nl.add,
                        data2=expert_out_tile_sb[:tile_T_actual, :],
                    )

    # Deferred spill for SBUF accumulation path (after all H-tiles complete)
    if not use_hbm_accumulation:
        for tile_t in nl.sequential_range(n_T128_tiles):
            tile_T_offset = TILE_T * tile_t
            tile_T_actual = min(TILE_T, T - tile_T_offset)
            tile_T_out_actual = min(tile_T_actual, T_out - tile_T_offset)

            if is_blockwise:
                """
                In blockwise computation, PNC1 would send its data to PNC0 for
                reduction into the HBM. We currently do not have the ability
                to sync between PNC cores to do atomic write. As a result, PNC0 will
                serially accumulate the data into HBM for both cores.
                When LNC=1 (n_prgs==1), skip sendrecv/core_barrier and write directly.
                """
                out_src = out_sb[:tile_T_actual, tile_t : tile_t + 1, :H]
                dst_ap = out_hbm.ap(
                    pattern=[[H, tile_T_actual], [1, H]],
                    offset=0,
                    vector_offset=token_position_to_id_T.ap(
                        pattern=[[n_T128_tiles, tile_T_actual], [1, 1]], offset=tile_t
                    ),
                    indirect_dim=0,
                )
                kernel_assert(
                    sharding_strategy == MoELNCShardingStrategy.SHARD_I
                    or sharding_strategy == MoELNCShardingStrategy.SHARD_T
                    or sharding_strategy == MoELNCShardingStrategy.SHARD_E,
                    "Blockwise down_projection_mx must use shard_on_I, shard_on_T, or shard_on_E",
                )

                if n_prgs == 1:
                    # Single core: write directly without cross-NC synchronization.
                    if is_first_expert:
                        nisa.dma_copy(src=out_src, dst=dst_ap, oob_mode=oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
                    else:
                        nisa.dma_compute(
                            dst=dst_ap,
                            srcs=[dst_ap, out_src],
                            scales=[1.0, 1.0],
                            reduce_op=nl.add,
                            oob_mode=oob_mode.skip,
                        )
                    continue

                out_src_other = nl.ndarray((tile_T_actual, 1, H), dtype=out_sb.dtype, buffer=nl.sbuf)
                nisa.sendrecv(
                    src=out_src,
                    dst=out_src_other,
                    send_to_rank=1 - nl.program_id(0),
                    recv_from_rank=1 - nl.program_id(0),
                    pipe_id=0,
                )

                if sharding_strategy == MoELNCShardingStrategy.SHARD_E:
                    out_idx_other = nl.ndarray((tile_T_actual, 1), dtype=token_position_to_id_T.dtype, buffer=nl.sbuf)
                    nisa.sendrecv(
                        src=token_position_to_id_T.ap(pattern=[[n_T128_tiles, tile_T_actual], [1, 1]], offset=tile_t),
                        dst=out_idx_other,
                        send_to_rank=1 - nl.program_id(0),
                        recv_from_rank=1 - nl.program_id(0),
                        pipe_id=0,
                    )
                    dst_ap_other_core = out_hbm.ap(
                        pattern=[[H, tile_T_actual], [1, H]],
                        offset=0,
                        vector_offset=out_idx_other,
                        indirect_dim=0,
                    )
                    if nl.program_id(0) == 0:
                        if is_first_expert:
                            nisa.dma_copy(src=out_src, dst=dst_ap, oob_mode=oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
                            nisa.dma_compute(
                                srcs=[dst_ap_other_core, out_src_other],
                                dst=dst_ap_other_core,
                                oob_mode=oob_mode.skip,
                                reduce_op=nl.add,
                            )
                        else:
                            nisa.dma_compute(
                                dst=dst_ap,
                                srcs=[dst_ap, out_src],
                                scales=[1.0, 1.0],
                                reduce_op=nl.add,
                                oob_mode=oob_mode.skip,
                            )
                            nisa.dma_compute(
                                dst=dst_ap_other_core,
                                srcs=[dst_ap_other_core, out_src_other],
                                scales=[1.0, 1.0],
                                reduce_op=nl.add,
                                oob_mode=oob_mode.skip,
                            )
                elif sharding_strategy == MoELNCShardingStrategy.SHARD_I:
                    # This only works in shard on I, as both cores has the same index
                    out_src_agg = nl.ndarray((tile_T_actual, 1, H), dtype=out_sb.dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(out_src_agg, out_src_other, out_src, op=nl.add)
                    if nl.program_id(0) == 0:
                        if is_first_expert:
                            nisa.dma_copy(
                                src=out_src_agg, dst=dst_ap, oob_mode=oob_mode.skip, dge_mode=nisa.dge_mode.swdge
                            )
                        else:
                            nisa.dma_compute(
                                dst=dst_ap,
                                srcs=[dst_ap, out_src_agg],
                                scales=[1.0, 1.0],
                                reduce_op=nl.add,
                                oob_mode=oob_mode.skip,
                            )
                elif sharding_strategy == MoELNCShardingStrategy.SHARD_T:
                    out_idx_other = nl.ndarray((tile_T_actual, 1), dtype=token_position_to_id_T.dtype, buffer=nl.sbuf)
                    nisa.sendrecv(
                        src=token_position_to_id_T.ap(pattern=[[n_T128_tiles, tile_T_actual], [1, 1]], offset=tile_t),
                        dst=out_idx_other,
                        send_to_rank=1 - nl.program_id(0),
                        recv_from_rank=1 - nl.program_id(0),
                        pipe_id=0,
                    )
                    dst_ap_other_core = out_hbm.ap(
                        pattern=[[H, tile_T_actual], [1, H]],
                        offset=0,
                        vector_offset=out_idx_other,
                        indirect_dim=0,
                    )
                    if nl.program_id(0) == 0:
                        if is_first_expert:
                            nisa.dma_copy(src=out_src, dst=dst_ap, oob_mode=oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
                            nisa.dma_compute(
                                srcs=[dst_ap_other_core, out_src_other],
                                dst=dst_ap_other_core,
                                oob_mode=oob_mode.skip,
                                reduce_op=nl.add,
                            )
                        else:
                            nisa.dma_compute(
                                dst=dst_ap,
                                srcs=[dst_ap, out_src],
                                scales=[1.0, 1.0],
                                reduce_op=nl.add,
                                oob_mode=oob_mode.skip,
                            )
                            nisa.dma_compute(
                                dst=dst_ap_other_core,
                                srcs=[dst_ap_other_core, out_src_other],
                                scales=[1.0, 1.0],
                                reduce_op=nl.add,
                                oob_mode=oob_mode.skip,
                            )
                nisa.core_barrier(out_hbm, cores=[0, 1])

            elif is_last_expert:
                if need_cross_nc_reduce:
                    H_local = H // n_prgs
                    H_offset_local = H_local * prg_id
                    _lnc_reduce_and_write(
                        src_send=out_sb[:tile_T_out_actual, tile_t, nl.ds(H_local * (1 - prg_id), H_local)],
                        src_local=out_sb[:tile_T_out_actual, tile_t, nl.ds(H_offset_local, H_local)],
                        hbm_dst=out_hbm[
                            nl.ds(T_offset + output_t_offset + TILE_T * tile_t, tile_T_out_actual),
                            nl.ds(H_offset_local, H_local),
                        ],
                        tile_T_actual=tile_T_out_actual,
                        TILE_T=TILE_T,
                        H_local=H_local,
                        prg_id=prg_id,
                        activation_compute_dtype=out_sb.dtype,
                        accumulate=False,
                    )
                else:
                    _no_h_shard = (
                        n_prgs == 1
                        or sharding_strategy == MoELNCShardingStrategy.SHARD_T
                        or sharding_strategy == MoELNCShardingStrategy.SHARD_E
                    )
                    H_local = H if _no_h_shard else H // n_prgs
                    H_offset_local = 0 if _no_h_shard else H_local * prg_id
                    nisa.dma_copy(
                        src=out_sb[:tile_T_out_actual, tile_t : tile_t + 1, nl.ds(H_offset_local, H_local)],
                        dst=out_hbm[
                            nl.ds(T_offset + output_t_offset + TILE_T * tile_t, tile_T_out_actual),
                            nl.ds(H_offset_local, H_local),
                        ],
                        dge_mode=nisa.dge_mode.none,
                    )

    return out_sb


def _lnc_reduce_and_write(
    src_send: nl.NkiTensor,
    src_local: nl.NkiTensor,
    hbm_dst: nl.NkiTensor,
    tile_T_actual: int,
    TILE_T: int,
    H_local: int,
    prg_id: int,
    activation_compute_dtype: nki.dtype,
    accumulate: bool,
):
    """LNC2 cross-NC reduction + HBM write with pre-sliced source tensors.

    Performs sendrecv to exchange H-halves between NCs, reduces locally, then writes
    to HBM. Callers must pre-slice the source tensor into the send half and local half
    using 3D indexing (to ensure correct access pattern generation).

    Args:
        src_send: Pre-sliced SBUF tensor [tile_T_actual, H_local] — the H-half to send to the other NC.
        src_local: Pre-sliced SBUF tensor [tile_T_actual, H_local] — this NC's H-half to keep.
        hbm_dst: Pre-sliced HBM destination for this NC's reduced output.
        tile_T_actual: Number of valid T elements in this tile.
        TILE_T: Full T tile size (for buffer allocation).
        H_local: H dimension per NC (= TILE_H // n_prgs).
        prg_id: This NC's index (0 or 1).
        activation_compute_dtype: Dtype for intermediate reduction buffer.
        accumulate: If True, uses dma_compute atomic add. If False, overwrites with dma_copy.
    """
    out_sb_reduced = nl.ndarray((TILE_T, H_local), activation_compute_dtype, buffer=nl.sbuf)

    nisa.sendrecv(
        send_to_rank=1 - prg_id,
        recv_from_rank=1 - prg_id,
        src=src_send,
        dst=out_sb_reduced[:tile_T_actual, :],
        pipe_id=0,
    )
    nisa.tensor_tensor(
        dst=out_sb_reduced[:tile_T_actual, :],
        data1=src_local,
        op=nl.add,
        data2=out_sb_reduced[:tile_T_actual, :],
    )

    if accumulate:
        nisa.dma_compute(
            dst=hbm_dst,
            srcs=[hbm_dst, out_sb_reduced[:tile_T_actual, :]],
            scales=[1.0, 1.0],
            reduce_op=nl.add,
        )
    else:
        nisa.dma_copy(src=out_sb_reduced[:tile_T_actual, :], dst=hbm_dst, dge_mode=nisa.dge_mode.none)


def _apply_down_dequant(
    expert_out_tile_sb: nl.NkiTensor,
    down_dequant_scale: nl.NkiTensor,
    down_input_dequant_scale: Optional[nl.NkiTensor],
    bias_sb: Optional[nl.NkiTensor],
    tile_T_actual: int,
    tile_T_offset: int,
    tile_H_offset: int,
    TILE_T: int,
    TILE_H: int,
    pmax: int,
):
    """Apply post-matmul software dequantization and optional bias.

    Handles two quantization modes:
    - STATIC_MX (down_dequant_scale.shape[1] == 1): single combined scale broadcast over H.
    - ROW_MX (down_dequant_scale.shape[1] > 1): per-column weight dequant scale,
      followed by optional per-token input dequant scale.

    Bias is added after dequant (deferred from PSUM eviction to preserve precision).
    """
    if down_dequant_scale.shape[1] == 1:
        # STATIC_MX: combined input*weight scale broadcasts over TILE_H
        nisa.activation(
            dst=expert_out_tile_sb[:tile_T_actual, :],
            op=nl.copy,
            data=expert_out_tile_sb[:tile_T_actual, :],
            scale=down_dequant_scale[:tile_T_actual, :],
        )
    else:
        # ROW_MX: per-column weight dequant, then per-token input dequant
        n_H128_in_tile = TILE_H // pmax
        dequant_scale_view = down_dequant_scale.slice(dim=0, start=0, end=tile_T_actual)
        for i_h128 in nl.affine_range(n_H128_in_tile):
            h_col = tile_H_offset // pmax + i_h128
            h_slice = nl.ds(i_h128 * pmax, pmax)
            interleave_copy(
                dst=expert_out_tile_sb[:tile_T_actual, h_slice],
                src=expert_out_tile_sb[:tile_T_actual, h_slice],
                scale=dequant_scale_view.slice(dim=1, start=h_col, end=h_col + 1),
                index=i_h128,
            )
        if down_input_dequant_scale != None:
            token_scale_sb = nl.ndarray((TILE_T, 1), dtype=nl.float32, buffer=nl.sbuf)
            token_scale_psum = nl.ndarray((TILE_T, 1), dtype=nl.float32, buffer=nl.psum)
            token_scale_1d = down_input_dequant_scale[0:1, tile_T_offset : tile_T_offset + tile_T_actual, 0]
            nisa.nc_transpose(data=token_scale_1d, dst=token_scale_psum[:tile_T_actual, 0])
            nisa.tensor_copy(dst=token_scale_sb[:tile_T_actual, :], src=token_scale_psum[:tile_T_actual, :])
            nisa.activation(
                dst=expert_out_tile_sb[:tile_T_actual, :],
                op=nl.copy,
                data=expert_out_tile_sb[:tile_T_actual, :],
                scale=token_scale_sb[:tile_T_actual, :],
            )
    if bias_sb != None:
        nisa.tensor_tensor(
            dst=expert_out_tile_sb[:tile_T_actual, :],
            data1=expert_out_tile_sb[:tile_T_actual, :],
            op=nl.add,
            data2=bias_sb[:tile_T_actual, tile_H_offset : tile_H_offset + TILE_H],
        )


def _down_proj_tile_compute(
    act_sb: nl.NkiTensor,
    act_scale_sb: nl.NkiTensor,
    weight_sb: nl.NkiTensor,
    weight_scale_sb: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    expert_affinities_masked_fp32_sb: nl.NkiTensor,
    down_dequant_scale: Optional[nl.NkiTensor],
    down_input_dequant_scale: Optional[nl.NkiTensor],
    t_tile_idx: int,
    tile_T_offset: int,
    tile_T_actual: int,
    tile_T_slice: nl.DynamicSlice,
    tile_H_offset: int,
    weight_H_slice: nl.DynamicSlice,
    TILE_T: int,
    TILE_H: int,
    n_I512_tiles: int,
    pmax: int,
    need_down_dequant: bool,
    activation_compute_dtype: nki.dtype,
    is_software_quant: bool = False,
) -> nl.ndarray:
    """Compute one (T-tile, H-tile) of the down projection for a single expert.

    Performs: matmul over I tiles → optional dequant → expert affinity scaling.

    Steps:
    1. MX matmul: act[I, T_tile] × weight[I, H_tile] → out[T_tile, H_tile] in PSUM
    2. PSUM eviction to SBUF (fused with bias add if no software dequant needed)
    3. Software dequantization (if need_down_dequant):
       - STATIC_MX: single combined scale broadcast over H
       - ROW_MX: per-column weight scale + optional per-token input scale
       - Bias add (deferred to after dequant)
    4. Expert affinity scaling: element-wise multiply by per-token affinity score

    Args:
        act_sb: Activation in SBUF [128_I, n_I512_tiles, T].
        act_scale_sb: Activation MX scales in SBUF [128_I, n_I512_tiles, T].
        weight_sb: Weight in SBUF [128_I, n_I512_tiles, H].
        weight_scale_sb: Weight MX scales in SBUF [128_I, n_I512_tiles, H].
        bias_sb: Optional bias [TILE_T, H].
        expert_affinities_masked_fp32_sb: Per-token affinity scores [T, n_T_tiles].
        down_dequant_scale: Dequant scale. [T, 1] for STATIC_MX, [T, H//128] for ROW_MX, None if not needed.
        down_input_dequant_scale: Per-token input dequant scale for ROW_MX [128, T, 1], or None.
        t_tile_idx: Index of this T-tile (used for affinity indexing).
        tile_T_offset: Absolute T offset of this tile.
        tile_T_actual: Number of valid T elements (handles remainder).
        tile_T_slice: nl.ds slice for T dimension in act/scale tensors.
        tile_H_offset: Absolute H offset of this tile.
        weight_H_slice: nl.ds slice for H dimension in weight tensors.
        TILE_T: Full T tile size.
        TILE_H: Full H tile size.
        n_I512_tiles: Number of I-dimension tiles to contract over.
        pmax: Partition max (128).
        need_down_dequant: Whether software dequantization is needed.
        activation_compute_dtype: Compute dtype for the output SBUF buffer.
        is_software_quant (bool): When True, weight_scale_sb is a 2D [128, H] dummy tile indexed
            as [:, :TILE_H] instead of the normal 3D [:, tile_i, H_slice]; and act_scale_sb is a 2D [128, T] dummy tile indexed
            as [:, :T] instead of the normal 3D [:, tile_i, T_slice].

    Returns:
        expert_out_tile_sb: Result in SBUF [TILE_T, TILE_H], affinity-scaled.
    """
    out_psum = nl.ndarray((TILE_T, TILE_H), dtype=nl.bfloat16, buffer=nl.psum)
    expert_out_tile_sb = nl.ndarray((TILE_T, TILE_H), dtype=activation_compute_dtype, buffer=nl.sbuf)
    for tile_i in nl.sequential_range(n_I512_tiles):
        nisa.nc_matmul_mx(
            dst=out_psum[:tile_T_actual, :],
            stationary=act_sb[:, tile_i, tile_T_slice],
            moving=weight_sb[:, tile_i, weight_H_slice],
            stationary_scale=act_scale_sb[:, :tile_T_actual]
            if is_software_quant
            else act_scale_sb[:, tile_i, tile_T_slice],
            moving_scale=weight_scale_sb[:, :TILE_H]
            if is_software_quant
            else weight_scale_sb[:, tile_i, weight_H_slice],
        )

    if bias_sb != None and down_dequant_scale == None:
        nisa.tensor_tensor(
            dst=expert_out_tile_sb[:tile_T_actual, :],
            data1=out_psum[:tile_T_actual, :],
            op=nl.add,
            data2=bias_sb[:tile_T_actual, tile_H_offset : tile_H_offset + TILE_H],
        )
    else:
        nisa.tensor_copy(dst=expert_out_tile_sb[:tile_T_actual, :], src=out_psum[:tile_T_actual, :])

    if need_down_dequant:
        _apply_down_dequant(
            expert_out_tile_sb=expert_out_tile_sb,
            down_dequant_scale=down_dequant_scale,
            down_input_dequant_scale=down_input_dequant_scale,
            bias_sb=bias_sb,
            tile_T_actual=tile_T_actual,
            tile_T_offset=tile_T_offset,
            tile_H_offset=tile_H_offset,
            TILE_T=TILE_T,
            TILE_H=TILE_H,
            pmax=pmax,
        )

    # Expert affinity scaling
    nisa.tensor_scalar(
        dst=expert_out_tile_sb[:tile_T_actual, :],
        data=expert_out_tile_sb.ap([[TILE_H, tile_T_actual], [1, TILE_H]]),
        op0=nl.multiply,
        operand0=expert_affinities_masked_fp32_sb[:tile_T_actual, t_tile_idx : t_tile_idx + 1],
        engine=nisa.scalar_engine,
    )
    return expert_out_tile_sb
