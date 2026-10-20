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

"""Geometry-specific Conv3D compute kernels."""

from dataclasses import dataclass
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    _psum_num_banks,
    div_ceil,
    get_nl_act_fn_from_type,
    get_verified_program_sharding_info,
)
from ...core.utils.logging import get_logger
from .conv3d_helpers import (
    _BN_EVAL_COEFF_PER_TILE,
    _BN_EVAL_SCRATCH_NAMES,
    _BN_STATS_ELEMS,
    _BN_TRIPLE_ELEMS,
    _MAX_K_REP_AT_32,
    _MAX_K_REP_AT_64,
    _NUM_PSUM_BANKS,
    _PARTITION_STRIDE_32,
    _PARTITION_STRIDE_64,
    _PSUM_BANK_SIZE,
    _SBUF_RESERVE,
    Conv3dConfig,
    Conv3dMemoryConfig,
    Conv3dTileConfig,
    _allocate_bn_eval_buffers,
    _apply_bias_activation_and_copy,
    _band_part_stride,
    _BatchNormPostProcessBuffers,
    _BatchNormPostProcessConfig,
    _bn_finalize_reserve_bytes,
    _BnBandPlan,
    _BnFinalizePlan,
    _build_fallback_bn_finalize_plan,
    _build_memory_config,
    _build_tile_config,
    _column_bands,
    _compute_unwritten_free_ranges,
    _compute_valid_free_range,
    _Conv3dComputeResult,
    _decompose_k_position,
    _dma_low_chan_result_regions,
    _finalize_bn_stats,
    _flush_column_layout,
    _get_c_out_replication_params,
    _get_k_replication_params,
    _get_memset_engine,
    _get_tensor_copy_engine,
    _heap_alloc_cost,
    _identity_shuffle_mask,
    _load_bias_into_buf,
    _load_bn_eval_scales_for_c_out_group,
    _low_chan_post_group_extent,
    _LowChanOutputChunk,
    _LowChanOutputRegion,
)


def _conv3d_fallback_compute(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    gamma: Optional[nl.NkiTensor],
    beta: Optional[nl.NkiTensor],
    batch_norm_eps: float,
    running_means: Optional[nl.NkiTensor],
    running_variances: Optional[nl.NkiTensor],
    residuals_in: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    conv_out: nl.NkiTensor,
    means: Optional[nl.NkiTensor],
    variances: Optional[nl.NkiTensor],
    sbm: Optional[SbufManager],
    use_auto_allocation: bool,
) -> Optional[_Conv3dComputeResult]:
    dtype_size = sizeinbytes(x_in.dtype)

    if sbm != None:
        kernel_assert(
            sbm.is_auto_alloc() == use_auto_allocation,
            f"[conv3d] If an SbufManager is provided, sbm.is_auto_alloc() ({sbm.is_auto_alloc()}) "
            f"must equal use_auto_allocation ({use_auto_allocation}).",
        )
        if use_auto_allocation:
            total_sbuf_budget = nl.tile_size.total_available_sbuf_size
        else:
            total_sbuf_budget = sbm.get_free_space()
    else:
        total_sbuf_budget = nl.tile_size.total_available_sbuf_size

    tile_cfg = _build_tile_config(cfg, dtype_size, total_sbuf_budget)
    mem_cfg = _build_memory_config(cfg, tile_cfg, dtype_size, total_sbuf_budget)
    # SPMD sharding info: n_prgs = core count, prg_id = this core (0/1 at LNC=2).
    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d", (0, 1))
    if sbm == None:
        # Preserve the requested allocation mode; the arena size is unchanged
        sbm = SbufManager(
            0,
            mem_cfg.total_memory_required,
            logger=get_logger("conv3d"),
            use_auto_alloc=use_auto_allocation,
        )
    heap_depth = len(sbm.heap)

    # Statistics buffer, live for the whole kernel. Partition = channel within a C_out tile (band g at
    # g * col_size); free = one _BN_STATS_ELEMS-wide bn_stats group per (global C_out tile, flush).
    bn_stats_bufs = None
    if cfg.is_batch_norm_training():
        full_c_out_tile_count = div_ceil(cfg.C_out, tile_cfg.P_MAX)
        bn_acc_part_dim = tile_cfg.C_OUT_REP * _band_part_stride(tile_cfg)
        bn_stats_bufs = sbm.alloc_heap(
            shape=(bn_acc_part_dim, full_c_out_tile_count * tile_cfg.bn_partial_iters * _BN_STATS_ELEMS),
            dtype=nl.float32,
            name="bn_stats_bufs",
        )
        # A tail flush can activate fewer than C_OUT_REP bands; zero the unwritten group slots so their
        # count is 0 and bn_aggr ignores them. At C_OUT_REP == 1 one band covers every flush.
        if tile_cfg.C_OUT_REP > 1:
            nisa.memset(dst=bn_stats_bufs[:, :], value=0.0)

    # Eval-mode affine buffers (allocated before the hot-path buffers, freed after them).
    bn_eval_scratch_bufs = None
    bn_eval_true_gammas = None
    bn_eval_neg_true_betas = None
    # For the column-tiling coefficient replication in _compute_true_bn_scales.
    bn_eval_identity_mask = _identity_shuffle_mask()
    if cfg.is_batch_norm_eval():
        bn_eval_scratch_bufs, bn_eval_true_gammas, bn_eval_neg_true_betas = _allocate_bn_eval_buffers(
            sbm=sbm,
            coeff_part_dim=tile_cfg.C_OUT_REP * _band_part_stride(tile_cfg),
            c_out_interleave=mem_cfg.c_out_interleave,
        )

    pre_bias_bufs, pre_filter_bufs = _allocate_filter_bias_buffers(sbm, cfg, tile_cfg, mem_cfg, x_in.dtype)

    max_effective_free = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    # Column tiling: result buffer is C_OUT_REP bands (band g at g * col_size, dh_head_size*W_tile
    # wide); rep == 1 keeps the (c_out_tile, full-free) shape.
    if tile_cfg.C_OUT_REP > 1:
        result_part_dim = tile_cfg.C_OUT_REP * tile_cfg.col_size
        band_free = tile_cfg.dh_head_size * tile_cfg.W_tile
    else:
        result_part_dim = tile_cfg.c_out_tile_size_max
        band_free = max_effective_free
    # Store-batch merging: a slot holds store_batch_merge batches side by side (batch b at cols
    # [b * band_free, +band_free)); store_batch_merge * store_pipe == w_out_interleave.
    result_free_dim = mem_cfg.store_batch_merge * band_free
    result_sbuf_slots = _allocate_result_slot_grid(
        sbm, mem_cfg, (result_part_dim, result_free_dim), x_in.dtype, "result"
    )

    # Residual staging: same shape / rotation as result_sbuf_slots. An in-place residual
    # (Conv3dConfig.residual_in_place) stages into the result tile itself, so no second grid.
    residual_sbuf_slots = None
    if cfg.needs_residual_staging_grid():
        residual_sbuf_slots = _allocate_result_slot_grid(
            sbm, mem_cfg, (result_part_dim, result_free_dim), x_in.dtype, "residual"
        )
    elif cfg.residual_in_place():
        residual_sbuf_slots = result_sbuf_slots

    input_window_slots = []
    for input_window_slot_idx in range(mem_cfg.input_window_interleave):
        input_window_slots.append(
            sbm.alloc_heap(
                shape=(
                    tile_cfg.c_in_tile_size_max,
                    tile_cfg.d_window_max,
                    tile_cfg.h_window_max,
                    tile_cfg.w_window_max,
                ),
                dtype=x_in.dtype,
                name=f"input_window_{input_window_slot_idx}",
            )
        )

    stacked_input_slots = []
    for stacked_input_slot_idx in range(mem_cfg.stacked_input_interleave):
        stacked_input_bufs = []
        for k_outer_idx in range(tile_cfg.K_outer_tile_count):
            stacked_input_bufs.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.stacked_filter_dim_max, max_effective_free),
                    dtype=x_in.dtype,
                    name=f"stacked_input_{stacked_input_slot_idx}_k{k_outer_idx}",
                )
            )
        stacked_input_slots.append(stacked_input_bufs)

    for c_out_group_start in range(tile_cfg.C_out_start, tile_cfg.C_out_end, mem_cfg.c_out_interleave * tile_cfg.P_MAX):
        c_out_group_end = min(c_out_group_start + mem_cfg.c_out_interleave * tile_cfg.P_MAX, tile_cfg.C_out_end)
        actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)

        c_out_tile_sizes, bias_cache, filters_cache = _load_bias_and_filters_for_c_out_group(
            filters,
            bias,
            cfg,
            tile_cfg,
            c_out_group_start,
            c_out_group_end,
            pre_bias_bufs,
            pre_filter_bufs,
        )

        # Eval mode: derive this group's affine coefficients from the running statistics once, then
        # apply them on every eviction of the group's inner loop.
        if cfg.is_batch_norm_eval():
            _load_bn_eval_scales_for_c_out_group(
                gamma=gamma,
                beta=beta,
                running_means=running_means,
                running_variances=running_variances,
                bias_cache=bias_cache,
                cfg=cfg,
                tile_cfg=tile_cfg,
                batch_norm_eps=batch_norm_eps,
                c_out_tile_sizes=c_out_tile_sizes,
                c_out_group_start=c_out_group_start,
                scratch_bufs=bn_eval_scratch_bufs,
                true_gamma_bufs=bn_eval_true_gammas,
                neg_true_beta_bufs=bn_eval_neg_true_betas,
                identity_mask=bn_eval_identity_mask,
            )

        w_out_tile_counter = 0
        c_in_slot_counter = 0

        if mem_cfg.store_batch_merge > 1:
            # Store-batch-merge path: one batch-strided DMA per band over store_batch_merge
            # batches. Column-tiling only (which disables H-chunking).
            w_out_tile_counter, c_in_slot_counter = _conv3d_c_out_group_merged_store(
                x_in=x_in,
                conv_out=conv_out,
                residuals_in=residuals_in,
                filters_cache=filters_cache,
                bias_cache=bias_cache,
                result_sbuf_slots=result_sbuf_slots,
                residual_sbuf_slots=residual_sbuf_slots,
                input_window_slots=input_window_slots,
                stacked_input_slots=stacked_input_slots,
                cfg=cfg,
                tile_cfg=tile_cfg,
                mem_cfg=mem_cfg,
                c_out_tile_sizes=c_out_tile_sizes,
                actual_group_size=actual_group_size,
                c_out_group_start=c_out_group_start,
                bn_stats_bufs=bn_stats_bufs,
                bn_eval_true_gammas=bn_eval_true_gammas,
                bn_eval_neg_true_betas=bn_eval_neg_true_betas,
            )
            continue

        for batch_idx in range(cfg.B):
            x_in_batch = x_in.select(dim=0, index=batch_idx)

            chunk_preloaded_windows = None
            dh_group_idx = tile_cfg.dh_start
            while dh_group_idx < tile_cfg.dh_end:
                dh_positions = []
                first_d_out = None
                for dh_stacked_idx in range(tile_cfg.num_dh_stacked):
                    flat_idx = dh_group_idx + dh_stacked_idx
                    if flat_idx >= tile_cfg.dh_end:
                        break
                    d_out_idx, h_out_idx = divmod(flat_idx, cfg.H_out)
                    if first_d_out == None:
                        first_d_out = d_out_idx
                    elif d_out_idx != first_d_out:
                        break  # don't cross D boundary — avoids window spanning full H
                    dh_positions.append((d_out_idx, h_out_idx))
                num_dh_positions = len(dh_positions)

                # H-chunking participation for this dh position.
                d_out_current = dh_positions[0][0]
                h_out_current = dh_positions[0][1]
                h_chunk_enabled = tile_cfg.h_chunk_size > 1 and tile_cfg.num_dh_stacked == 1
                is_chunk_start = h_chunk_enabled and (h_out_current % tile_cfg.h_chunk_size == 0)

                # Build the chunk's dh_positions
                if is_chunk_start:
                    chunk_dh_positions = []
                    for h_chunk_idx in range(tile_cfg.h_chunk_size):
                        chunk_h_out = h_out_current + h_chunk_idx
                        if chunk_h_out >= cfg.H_out:
                            break
                        # Don't cross the d_out boundary.
                        chunk_flat = d_out_current * cfg.H_out + chunk_h_out
                        if chunk_flat >= tile_cfg.dh_end:
                            break
                        chunk_dh_positions.append((d_out_current, chunk_h_out))

                # Column-tiling bands for this D-H group, shared by the residual load and the store.
                col_rep_st, band_dh_st, col_size_st = _flush_column_layout(tile_cfg, num_dh_positions)
                store_bands = _column_bands(num_dh_positions, band_dh_st, col_rep_st, col_size_st)

                for w_start in range(0, cfg.W_out, tile_cfg.W_tile):
                    w_end = min(w_start + tile_cfg.W_tile, cfg.W_out)

                    # At an H-chunk start, preload the chunk's union input window for all c_in tiles.
                    if is_chunk_start:
                        chunk_preloaded_windows = []
                        c_in_tile_idx_load = 0
                        for c_in_start_load in range(0, cfg.C_in, tile_cfg.P_MAX):
                            c_in_end_load = min(c_in_start_load + tile_cfg.P_MAX, cfg.C_in)
                            global_idx_load = c_in_slot_counter + c_in_tile_idx_load
                            input_window_buf_load = input_window_slots[global_idx_load % len(input_window_slots)]
                            x_in_cin_load = x_in_batch.slice(dim=0, start=c_in_start_load, end=c_in_end_load, step=1)
                            loaded = _load_input_window_to_sbuf_3d(
                                x_in_cin=x_in_cin_load,
                                input_window_buf=input_window_buf_load,
                                cfg=cfg,
                                dh_positions=chunk_dh_positions,
                                w_start=w_start,
                                w_end=w_end,
                            )
                            chunk_preloaded_windows.append(loaded)
                            c_in_tile_idx_load += 1

                    slot_idx = w_out_tile_counter % mem_cfg.w_out_interleave
                    result_sbufs = result_sbuf_slots[slot_idx][:actual_group_size]
                    psum_bank_idx = (slot_idx * actual_group_size) % _NUM_PSUM_BANKS

                    # Preloaded windows only when H-chunking.
                    preloaded = chunk_preloaded_windows if h_chunk_enabled else None

                    # Same band layout as the store below (the add is only correct if they agree),
                    # issued before the compute so the DMA overlaps the matmuls.
                    residual_sbufs = None
                    if residual_sbuf_slots != None:
                        residual_sbufs = residual_sbuf_slots[slot_idx][:actual_group_size]
                        _dma_result_bands(
                            sbuf_bufs=residual_sbufs,
                            hbm_tensor=residuals_in,
                            cfg=cfg,
                            tile_cfg=tile_cfg,
                            c_out_tile_sizes=c_out_tile_sizes,
                            c_out_group_start=c_out_group_start,
                            batch_idx=batch_idx,
                            bands=store_bands,
                            dh_group_idx=dh_group_idx,
                            w_start=w_start,
                            w_end=w_end,
                            to_hbm=False,
                        )

                    _conv3d_output_tile(
                        x_in_batch=x_in_batch,
                        filters_cache=filters_cache,
                        bias_cache=bias_cache,
                        result_sbufs=result_sbufs,
                        input_window_slots=input_window_slots,
                        stacked_input_slots=stacked_input_slots,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                        psum_bank_idx=psum_bank_idx,
                        c_in_slot_offset=c_in_slot_counter,
                        preloaded_windows=preloaded,
                        bn_c_out_tile_base=c_out_group_start // tile_cfg.P_MAX,
                        bn_stats_bufs=bn_stats_bufs,
                        bn_partial_iters=tile_cfg.bn_partial_iters,
                        bn_iter_idx=w_out_tile_counter,
                        bn_eval_true_gammas=bn_eval_true_gammas,
                        bn_eval_neg_true_betas=bn_eval_neg_true_betas,
                        residual_sbufs=residual_sbufs,
                    )
                    c_in_slot_counter += tile_cfg.c_in_tile_count

                    # Store SBUF -> HBM, the mirror image of the residual load above.
                    _dma_result_bands(
                        sbuf_bufs=result_sbufs,
                        hbm_tensor=conv_out,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        c_out_group_start=c_out_group_start,
                        batch_idx=batch_idx,
                        bands=store_bands,
                        dh_group_idx=dh_group_idx,
                        w_start=w_start,
                        w_end=w_end,
                        to_hbm=True,
                    )

                    w_out_tile_counter += 1
                dh_group_idx += num_dh_positions

    # Free all heap allocations
    for _ in range(mem_cfg.stacked_input_interleave):
        for _ in range(tile_cfg.K_outer_tile_count):
            sbm.pop_heap()
    for _ in range(mem_cfg.input_window_interleave):
        sbm.pop_heap()
    # Result / residual grids, popped in reverse allocation order. result_bufs_per_slot is what the fit
    # search budgeted: 2 exactly when a separate residual grid was allocated, else 1.
    for _ in range(mem_cfg.result_bufs_per_slot * mem_cfg.store_pipe * mem_cfg.c_out_interleave):
        sbm.pop_heap()  # residual_* (if any), then result_*
    for _ in range(tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count):
        sbm.pop_heap()
    for _ in range(mem_cfg.c_out_interleave):
        if cfg.has_bias:
            sbm.pop_heap()
    if cfg.is_batch_norm_eval():
        # bn_eval coefficient columns (2 per interleave slot) then the 5 shared scratch columns.
        for _ in range(mem_cfg.c_out_interleave):
            sbm.pop_heap()  # bn_eval_neg_true_beta_*
            sbm.pop_heap()  # bn_eval_true_gamma_*
        for _ in range(len(_BN_EVAL_SCRATCH_NAMES)):
            sbm.pop_heap()

    if cfg.is_batch_norm_training():
        return _prepare_fallback_training_result(
            cfg=cfg,
            tile_cfg=tile_cfg,
            mem_cfg=mem_cfg,
            sbm=sbm,
            heap_depth=heap_depth,
            conv_dtype=conv_out.dtype,
            bn_stats_bufs=bn_stats_bufs,
            means=means,
            variances=variances,
            n_prgs=n_prgs,
            prg_id=prg_id,
        )
    return None


def _prepare_fallback_training_result(
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    mem_cfg: Conv3dMemoryConfig,
    sbm: SbufManager,
    heap_depth: int,
    conv_dtype: object,
    bn_stats_bufs: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    n_prgs: int,
    prg_id: int,
) -> _Conv3dComputeResult:
    p_max = tile_cfg.P_MAX
    full_c_out_tile_count = div_ceil(cfg.C_out, p_max)
    tile_size_full = min(p_max, cfg.C_out)
    total_spatial = cfg.D_out * cfg.H_out * cfg.W_out
    reload_free_tile = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    shard_stats_on_dh = tile_cfg.shard_on_dh and n_prgs > 1

    # Allocate persistent buffers before temporary finalization scratch
    recv_stats = sbm.alloc_heap(
        shape=(tile_size_full, full_c_out_tile_count * _BN_TRIPLE_ELEMS),
        dtype=nl.float32,
        name="bn_recv_stats",
    )
    means_sbuf = sbm.alloc_heap(
        shape=(tile_size_full, full_c_out_tile_count),
        dtype=nl.float32,
        name="bn_means_sbuf",
    )
    variances_sbuf = sbm.alloc_heap(
        shape=(tile_size_full, full_c_out_tile_count),
        dtype=nl.float32,
        name="bn_variances_sbuf",
    )

    c_out_rep = tile_cfg.C_OUT_REP
    rep_part_dim = c_out_rep * tile_size_full
    gamma_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_gamma")
    beta_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_beta")
    true_gamma_buf = sbm.alloc_heap(shape=(rep_part_dim, 1), dtype=nl.float32, name="bn_true_gamma")
    neg_true_beta_buf = sbm.alloc_heap(shape=(rep_part_dim, 1), dtype=nl.float32, name="bn_neg_true_beta")
    run_running_mean_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_running_mean")
    run_new_mean_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_new_mean")
    run_running_var_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_running_var")
    run_new_var_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_new_var")

    has_residual = cfg.has_residual_add()
    reload_slot_cost = _heap_alloc_cost(reload_free_tile * sizeinbytes(conv_dtype))
    if has_residual:
        reload_slot_cost = 2 * reload_slot_cost
    # Reserve space for the finalization frame allocated after reload_buf
    finalize_plan = _build_fallback_bn_finalize_plan(cfg, tile_cfg, shard_stats_on_dh, prg_id)
    free_slots = max(1, (sbm.get_free_space() - _bn_finalize_reserve_bytes(finalize_plan)) // reload_slot_cost)
    reload_interleave = min(mem_cfg.bn_reload_interleave, free_slots)
    reload_buf = sbm.alloc_heap(
        shape=(rep_part_dim, reload_interleave * reload_free_tile),
        dtype=conv_dtype,
        name="bn_reload",
    )
    residual_reload_buf = None
    if has_residual:
        residual_reload_buf = sbm.alloc_heap(
            shape=(rep_part_dim, reload_interleave * reload_free_tile),
            dtype=conv_dtype,
            name="bn_residual_reload",
        )

    identity_mask = _identity_shuffle_mask()
    _finalize_bn_stats(
        plan=finalize_plan,
        sbm=sbm,
        bn_stats_bufs=bn_stats_bufs,
        recv_triples=recv_stats,
        means_sbuf=means_sbuf,
        variances_sbuf=variances_sbuf,
        means=means,
        variances=variances,
        identity_mask=identity_mask,
    )

    # Each program owns either a C_out or D-H slice
    shard_c_out = n_prgs > 1 and not shard_stats_on_dh
    if shard_c_out:
        rescale_c_out_start, rescale_c_out_end = tile_cfg.C_out_start, tile_cfg.C_out_end
    else:
        rescale_c_out_start, rescale_c_out_end = 0, cfg.C_out
    if shard_stats_on_dh:
        rescale_spatial_start = tile_cfg.dh_start * cfg.W_out
        rescale_spatial_end = tile_cfg.dh_end * cfg.W_out
    else:
        rescale_spatial_start, rescale_spatial_end = 0, total_spatial
    # D-H shards share statistics; only core 0 updates momentum
    write_momentum = prg_id == 0 if shard_stats_on_dh else True

    return _Conv3dComputeResult(
        sbm=sbm,
        heap_depth=heap_depth,
        means_sbuf=means_sbuf,
        variances_sbuf=variances_sbuf,
        post_config=_BatchNormPostProcessConfig(
            p_max=p_max,
            batch_start=0,
            batch_end=cfg.B,
            c_out_start=rescale_c_out_start,
            c_out_end=rescale_c_out_end,
            spatial_start=rescale_spatial_start,
            spatial_end=rescale_spatial_end,
            reload_interleave=reload_interleave,
            reload_free_tile=reload_free_tile,
            fixed_c_out_rep=c_out_rep,
            low_chan_tile_layouts=None,
            write_momentum=write_momentum,
        ),
        post_buffers=_BatchNormPostProcessBuffers(
            gamma_buf=gamma_buf,
            beta_buf=beta_buf,
            true_gamma_buf=true_gamma_buf,
            neg_true_beta_buf=neg_true_beta_buf,
            run_running_mean_buf=run_running_mean_buf,
            run_new_mean_buf=run_new_mean_buf,
            run_running_var_buf=run_running_var_buf,
            run_new_var_buf=run_new_var_buf,
            reload_buf=reload_buf,
            residual_reload_buf=residual_reload_buf,
        ),
    )


def _conv3d_c_out_group_merged_store(
    x_in: nl.NkiTensor,
    conv_out: nl.NkiTensor,
    residuals_in: Optional[nl.NkiTensor],
    filters_cache: list,
    bias_cache: list,
    result_sbuf_slots: list,
    residual_sbuf_slots: Optional[list],
    input_window_slots: list,
    stacked_input_slots: list,
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    mem_cfg: "Conv3dMemoryConfig",
    c_out_tile_sizes: list[int],
    actual_group_size: int,
    c_out_group_start: int,
    bn_stats_bufs: Optional[nl.NkiTensor],
    bn_eval_true_gammas: Optional[list[nl.NkiTensor]] = None,
    bn_eval_neg_true_betas: Optional[list[nl.NkiTensor]] = None,
) -> tuple[int, int]:
    """
    Process one C_out group with store-batch merging (column tiling only).

    Reorders the loop nest so batch is inner to each (D-H group, W tile) flush: each of the
    store_batch_merge batches computes into its own free-column slice [b * band_free, +band_free) of
    one wide result slot, then the whole group goes to HBM in a single batch-strided DMA per band,
    cutting the store-trigger count by store_batch_merge. Per-batch compute and the batchnorm
    accumulation are unchanged; only the store is merged.

    Enabled by _build_memory_config only when every flush is full (D_out == 1 and the per-core D-H
    range divides evenly into num_dh_stacked), so each batch's result is exactly band_free wide. The
    returned counters are group-local, kept only for parity with the per-batch path.

    The store DMA lowers because the batch axis is an inner free stride and the channel axis (M) is
    the partition-aligned outer dim (verified against the device tracer) — unlike a cross-band merge,
    which cannot align band-major SBUF partitions to channel-major HBM.
    """
    merge = mem_cfg.store_batch_merge
    band_free = tile_cfg.dh_head_size * tile_cfg.W_tile
    P_MAX = tile_cfg.P_MAX
    w_out_tile_counter = 0
    c_in_slot_counter = 0
    store_group_idx = 0

    dh_group_idx = tile_cfg.dh_start
    while dh_group_idx < tile_cfg.dh_end:
        # Enable gate guarantees full flushes: the group is num_dh_stacked consecutive positions.
        dh_positions = []
        for dh_stacked_idx in range(tile_cfg.num_dh_stacked):
            flat_idx = dh_group_idx + dh_stacked_idx
            d_out_idx, h_out_idx = divmod(flat_idx, cfg.H_out)
            dh_positions.append((d_out_idx, h_out_idx))
        num_dh_positions = len(dh_positions)
        col_rep_st, band_dh_st, col_size_st = _flush_column_layout(tile_cfg, num_dh_positions)
        store_bands = _column_bands(num_dh_positions, band_dh_st, col_rep_st, col_size_st)

        for w_start in range(0, cfg.W_out, tile_cfg.W_tile):
            w_end = min(w_start + tile_cfg.W_tile, cfg.W_out)

            # Per batch-group: stage its batches into one rotating result slot, then merged-store.
            for batch_group_start in range(0, cfg.B, merge):
                group_batches = min(batch_group_start + merge, cfg.B) - batch_group_start

                slot_idx = store_group_idx % mem_cfg.store_pipe
                result_slot = result_sbuf_slots[slot_idx][:actual_group_size]
                residual_slot = None
                if residual_sbuf_slots != None:
                    residual_slot = residual_sbuf_slots[slot_idx][:actual_group_size]

                # One batch-strided DMA for the whole group, mirroring the merged store below, so the
                # load's trigger count matches it instead of being store_batch_merge x higher.
                if residual_slot != None:
                    _dma_merged_bands(
                        sbuf_slot=residual_slot,
                        hbm_tensor=residuals_in,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        c_out_group_start=c_out_group_start,
                        batch_group_start=batch_group_start,
                        group_batches=group_batches,
                        bands=store_bands,
                        dh_group_idx=dh_group_idx,
                        w_start=w_start,
                        w_end=w_end,
                        to_hbm=False,
                    )

                # Compute each batch into its own free-column slice [b_local * band_free, ...).
                for b_local in range(group_batches):
                    batch_idx = batch_group_start + b_local
                    x_in_batch = x_in.select(dim=0, index=batch_idx)
                    result_sbufs = _batch_column_views(result_slot, b_local, band_free)
                    # Residual: this batch's slice of the same wide slot, staged above.
                    residual_sbufs = None
                    if residual_slot != None:
                        residual_sbufs = _batch_column_views(residual_slot, b_local, band_free)
                    psum_bank_idx = (w_out_tile_counter * actual_group_size) % _NUM_PSUM_BANKS
                    _conv3d_output_tile(
                        x_in_batch=x_in_batch,
                        filters_cache=filters_cache,
                        bias_cache=bias_cache,
                        result_sbufs=result_sbufs,
                        input_window_slots=input_window_slots,
                        stacked_input_slots=stacked_input_slots,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                        psum_bank_idx=psum_bank_idx,
                        c_in_slot_offset=c_in_slot_counter,
                        preloaded_windows=None,
                        bn_c_out_tile_base=c_out_group_start // P_MAX,
                        bn_stats_bufs=bn_stats_bufs,
                        bn_partial_iters=tile_cfg.bn_partial_iters,
                        bn_iter_idx=w_out_tile_counter,
                        bn_eval_true_gammas=bn_eval_true_gammas,
                        bn_eval_neg_true_betas=bn_eval_neg_true_betas,
                        residual_sbufs=residual_sbufs,
                    )
                    c_in_slot_counter += tile_cfg.c_in_tile_count
                    w_out_tile_counter += 1

                # Merged store: one batch-strided DMA per (c_out tile, band), the mirror image of the
                # residual load above.
                _dma_merged_bands(
                    sbuf_slot=result_slot,
                    hbm_tensor=conv_out,
                    cfg=cfg,
                    tile_cfg=tile_cfg,
                    c_out_tile_sizes=c_out_tile_sizes,
                    c_out_group_start=c_out_group_start,
                    batch_group_start=batch_group_start,
                    group_batches=group_batches,
                    bands=store_bands,
                    dh_group_idx=dh_group_idx,
                    w_start=w_start,
                    w_end=w_end,
                    to_hbm=True,
                )
                store_group_idx += 1
        dh_group_idx += num_dh_positions

    return w_out_tile_counter, c_in_slot_counter


def _dma_merged_bands(
    sbuf_slot: list[nl.NkiTensor],
    hbm_tensor: nl.NkiTensor,
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    c_out_tile_sizes: list[int],
    c_out_group_start: int,
    batch_group_start: int,
    group_batches: int,
    bands: list[tuple[int, int, int]],
    dh_group_idx: int,
    w_start: int,
    w_end: int,
    to_hbm: bool,
) -> None:
    """
    DMA a whole batch group's store-batch-merged tiles between SBUF and a
    [B, C_out, D_out, H_out, W_out] HBM tensor: one batch-strided DMA per (C_out tile, column band).

    The SBUF side strides batch as an inner free dim (batch b at [b * band_free, +band_free)) and the
    HBM side permutes channel outermost to align with the SBUF partition dim. That lowering was
    verified against the device tracer and is why the batch axis must be the inner free stride.

    to_hbm selects the direction, so the merged result store and the merged residual load share one
    definition of the layout — the residual add is only correct if the two agree exactly. One DMA per
    (C_out tile, band) per batch group either way, which on the load side is store_batch_merge times
    fewer triggers than issuing it inside the per-batch compute loop.
    """
    total_dh = cfg.D_out * cfg.H_out
    band_free = tile_cfg.dh_head_size * tile_cfg.W_tile
    w_tile_size = w_end - w_start
    batch_group_end = batch_group_start + group_batches
    for c_out_tile_idx in range(len(sbuf_slot)):
        M = c_out_tile_sizes[c_out_tile_idx]
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_end = c_out_tile_start + M
        for band_idx in range(len(bands)):
            part_off = bands[band_idx][0]
            band_dh_lo = bands[band_idx][1]
            band_dh_hi = bands[band_idx][2]
            this_band_dh = band_dh_hi - band_dh_lo
            # SBUF side -> (M, group_batches, this_band_dh, w_tile_size).
            sbuf_view = (
                sbuf_slot[c_out_tile_idx]
                .slice(dim=0, start=part_off, end=part_off + M, step=1)
                .slice(dim=1, start=0, end=group_batches * band_free, step=1)
                .reshape_dim(1, (group_batches, band_free))
                .slice(dim=2, start=0, end=this_band_dh * w_tile_size, step=1)
                .reshape_dim(2, (this_band_dh, w_tile_size))
            )
            # HBM side: [batch_group, c_out_tile, dh_band, w] with channel permuted outermost.
            hbm_view = (
                hbm_tensor.slice(dim=0, start=batch_group_start, end=batch_group_end, step=1)
                .slice(dim=1, start=c_out_tile_start, end=c_out_tile_end, step=1)
                .flatten_dims(2, 4)
                .reshape_dim(2, (total_dh, cfg.W_out))
                .slice(dim=2, start=dh_group_idx + band_dh_lo, end=dh_group_idx + band_dh_hi, step=1)
                .slice(dim=3, start=w_start, end=w_end, step=1)
                .permute((1, 0, 2, 3))
            )
            nisa.dma_copy(
                dst=hbm_view if to_hbm else sbuf_view,
                src=sbuf_view if to_hbm else hbm_view,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nki.isa.engine.sync,
            )


def _batch_column_views(slot: list[nl.NkiTensor], b_local: int, band_free: int) -> list[nl.NkiTensor]:
    """
    Slice one batch's free-column window [b_local * band_free, +band_free) out of each buffer of a
    store-batch-merged slot. Explicit loop — the tracer rejects comprehensions.
    """
    views: list[nl.NkiTensor] = []
    for tile_idx in range(len(slot)):
        views.append(
            slot[tile_idx].slice(dim=1, start=b_local * band_free, end=b_local * band_free + band_free, step=1)
        )
    return views


def _dma_result_bands(
    sbuf_bufs: list[nl.NkiTensor],
    hbm_tensor: nl.NkiTensor,
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    c_out_tile_sizes: list[int],
    c_out_group_start: int,
    batch_idx: int,
    bands: list[tuple[int, int, int]],
    dh_group_idx: int,
    w_start: int,
    w_end: int,
    to_hbm: bool,
) -> None:
    """
    DMA one flush's result-shaped tiles between SBUF and a [B, C_out, D_out, H_out, W_out] HBM
    tensor: one DMA per (C_out tile, column-tiling band), band g covering D-H window
    [band_dh_lo, band_dh_hi) at SBUF partitions [g * col_size, ...) in band-local free coords.

    to_hbm selects the direction, so the result store and the residual load share one definition of
    the band layout — the residual add is only correct if the two agree exactly. Callers issue the
    load before the tile's compute so the DMA overlaps the matmuls. See _dma_merged_bands for the
    store-batch-merged equivalent.
    """
    total_dh = cfg.D_out * cfg.H_out
    w_tile_size = w_end - w_start
    for c_out_tile_idx in range(len(sbuf_bufs)):
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_end = c_out_tile_start + c_out_tile_size
        for band_idx in range(len(bands)):
            part_off = bands[band_idx][0]
            band_dh_lo = bands[band_idx][1]
            band_dh_hi = bands[band_idx][2]
            this_band_dh = band_dh_hi - band_dh_lo
            sbuf_view = (
                (sbuf_bufs[c_out_tile_idx])
                .slice(dim=0, start=part_off, end=part_off + c_out_tile_size, step=1)
                .slice(dim=1, start=0, end=this_band_dh * w_tile_size, step=1)
                .reshape_dim(1, (this_band_dh, w_tile_size))
            )
            hbm_view = (
                hbm_tensor.select(dim=0, index=batch_idx)
                .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                .flatten_dims(1, 3)
                .reshape_dim(1, (total_dh, cfg.W_out))
                .slice(dim=1, start=dh_group_idx + band_dh_lo, end=dh_group_idx + band_dh_hi, step=1)
                .slice(dim=2, start=w_start, end=w_end, step=1)
            )
            nisa.dma_copy(
                dst=hbm_view if to_hbm else sbuf_view,
                src=sbuf_view if to_hbm else hbm_view,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nki.isa.engine.sync,
            )


def _scatter_input_to_stacked(
    input_window: Optional[nl.NkiTensor],
    stacked_input_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_tile_size: int,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    valid_d_start: int,
    valid_d_end: int,
    valid_h_start: int,
    valid_h_end: int,
    valid_w_start: int,
    valid_w_end: int,
    used_k_positions: list[int],
    tensor_copy_engine_idx: int,
    memset_engine_idx: int = 0,
    tensor_copy_engine_modulo: int = 4,
    memset_engine_modulo: int = 2,
) -> tuple[list[Optional[nl.NkiTensor]], list[int], list[int]]:
    """
    Scatter input window data into K-replicated stacked buffers for matmul.

    For each K-outer tile, maps the relevant input positions from the loaded
    input window into a stacked buffer layout where multiple filter positions
    are packed along the partition dimension. Handles padding by zero-initializing
    buffers or adjusting the free-dimension and stride of input tiles. Returns None
    for fully-padded K-outer tiles to skip unnecessary matmul operations.

    This function is agnostic to C_out column tiling: it always produces the stacked
    input covering the whole num_dh_stacked free dim. Column tiling is applied purely
    downstream in _conv3d_matmul, which slices this stacked input per column-tiling group.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    K_total_used = len(used_k_positions)
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    effective_free_dim = num_dh_stacked * w_tile_size
    K_REP, partition_stride = _get_k_replication_params(c_in_tile_size, K_total_used) if c_in_tile_size > 0 else (1, 1)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)
    input_stacked_list: list[Optional[nl.NkiTensor]] = []
    valid_offsets: list[int] = []
    valid_sizes: list[int] = []
    valid_dh_starts: list[int] = []
    valid_dh_counts: list[int] = []
    valid_w_starts: list[int] = []
    valid_w_counts: list[int] = []
    use_strided_flags: list[bool] = []

    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual
        raw_buf = stacked_input_bufs[k_outer_idx]

        # Compute the valid free-dim range for this K-outer tile
        valid_free_start, valid_free_end = _compute_valid_free_range(
            cfg,
            dh_positions,
            w_start,
            w_end,
            used_k_positions,
            k_start,
            k_end,
        )

        if valid_free_start >= valid_free_end:
            input_stacked_list.append(None)
            valid_offsets.append(0)
            valid_sizes.append(0)
            valid_dh_starts.append(0)
            valid_dh_counts.append(0)
            valid_w_starts.append(0)
            valid_w_counts.append(0)
            use_strided_flags.append(False)
            continue

        valid_free_size = valid_free_end - valid_free_start
        first_valid_dh_idx_pre = valid_free_start // w_tile_size
        last_valid_dh_idx_pre = (valid_free_end - 1) // w_tile_size
        k_w_valid_start_pre = valid_free_start - first_valid_dh_idx_pre * w_tile_size
        k_w_valid_end_pre = valid_free_end - last_valid_dh_idx_pre * w_tile_size
        valid_dh_count_pre = last_valid_dh_idx_pre - first_valid_dh_idx_pre + 1
        has_w_padding = k_w_valid_start_pre > 0 or k_w_valid_end_pre < w_tile_size

        # All K positions share the same W valid range?
        all_k_same_w_range_pre = True
        if k_actual > 1 and valid_dh_count_pre > 1 and has_w_padding:
            for k_check_pre in range(k_actual):
                flat_k_pre = used_k_positions[k_start + k_check_pre]
                _, _, k_w_pre = _decompose_k_position(flat_k_pre, K_h, K_w)
                base_w_pre = w_start * cfg.stride_w + k_w_pre * cfg.dilation_w - cfg.pad_w_left
                this_w_start_pre = div_ceil(-base_w_pre, cfg.stride_w) if base_w_pre < 0 else 0
                last_w_pre = (w_end - 1) * cfg.stride_w + k_w_pre * cfg.dilation_w - cfg.pad_w_left
                this_w_end_pre = (
                    w_tile_size - div_ceil(last_w_pre - cfg.W + 1, cfg.stride_w) if last_w_pre >= cfg.W else w_tile_size
                )
                if this_w_start_pre != k_w_valid_start_pre or this_w_end_pre != k_w_valid_end_pre:
                    all_k_same_w_range_pre = False
                    break

        will_use_strided = all_k_same_w_range_pre and valid_dh_count_pre > 1 and has_w_padding

        # Check if any positions within the valid free-dim range need zero memset.
        needs_zero_init = partition_stride > c_in_tile_size
        if not needs_zero_init:
            first_valid_dh_idx = valid_free_start // w_tile_size
            last_valid_dh_idx = (valid_free_end - 1) // w_tile_size
            for dh_idx in range(first_valid_dh_idx, last_valid_dh_idx + 1):
                d_out, h_out = dh_positions[dh_idx]
                # Portion of this dh's W range within the valid range.
                dh_free_start = dh_idx * w_tile_size
                dh_free_end = (dh_idx + 1) * w_tile_size
                # Clip to the valid range
                check_w_start = max(dh_free_start, valid_free_start) - dh_free_start
                check_w_end = min(dh_free_end, valid_free_end) - dh_free_start
                for k_rep_idx in range(k_actual):
                    flat_k_pos = used_k_positions[k_start + k_rep_idx]
                    k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
                    d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                    h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                    if d_in < 0 or d_in >= cfg.D or h_in < 0 or h_in >= cfg.H:
                        needs_zero_init = True
                        break
                    # W range within the valid portion has any padding?
                    if not will_use_strided:
                        base_w_in = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
                        first_valid_w = 0 if base_w_in >= 0 else div_ceil(-base_w_in, cfg.stride_w)
                        last_w_in = (w_end - 1) * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
                        last_valid_w = (
                            w_tile_size
                            if last_w_in < cfg.W
                            else w_tile_size - div_ceil(last_w_in - cfg.W + 1, cfg.stride_w)
                        )
                        if first_valid_w > check_w_start or last_valid_w < check_w_end:
                            needs_zero_init = True
                            break
                if needs_zero_init:
                    break

        if needs_zero_init:
            # Memset only the valid-range portion.
            nisa.memset(
                dst=raw_buf[:stacked_filter_dim, valid_free_start:valid_free_end],
                value=0.0,
                engine=_get_memset_engine(memset_engine_idx + k_outer_idx, memset_engine_modulo),
            )

        if input_window != None:
            for k_rep_idx in range(k_actual):
                partition_offset = k_rep_idx * partition_stride
                flat_k_pos = used_k_positions[k_start + k_rep_idx]
                k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
                base_w_in = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left

                # Compute valid W range
                first_valid_out = max(
                    0,
                    min(
                        div_ceil(valid_w_start - base_w_in, cfg.stride_w) if base_w_in < valid_w_start else 0,
                        w_tile_size,
                    ),
                )
                last_valid_out = max(
                    0,
                    min(
                        div_ceil(valid_w_end - base_w_in, cfg.stride_w)
                        if base_w_in + (w_tile_size - 1) * cfg.stride_w >= valid_w_end
                        else w_tile_size,
                        w_tile_size,
                    ),
                )

                if first_valid_out >= last_valid_out:
                    continue

                src_w_start = base_w_in + first_valid_out * cfg.stride_w - valid_w_start
                num_w_elements = last_valid_out - first_valid_out
                src_w_end = src_w_start + (num_w_elements - 1) * cfg.stride_w + 1

                # Group consecutive dh_idx values into fewer, larger strided tensor_copy ops.
                dh_idx = 0
                while dh_idx < num_dh_stacked:
                    d_out, h_out = dh_positions[dh_idx]
                    d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                    h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                    if d_in < valid_d_start or d_in >= valid_d_end or h_in < valid_h_start or h_in >= valid_h_end:
                        dh_idx += 1
                        continue

                    d_local_start = d_in - valid_d_start
                    h_local_start = h_in - valid_h_start
                    group_start_dh = dh_idx

                    # Group consecutive H positions with the same d_in
                    num_h = 1
                    next_dh = dh_idx + 1
                    while next_dh < num_dh_stacked:
                        d_out_next, h_out_next = dh_positions[next_dh]
                        d_in_next = d_out_next * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                        h_in_next = h_out_next * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                        if d_in_next != d_in:
                            break
                        if h_in_next < valid_h_start or h_in_next >= valid_h_end:
                            break
                        expected_h_in = h_in + num_h * cfg.stride_h
                        if h_in_next != expected_h_in:
                            break
                        num_h += 1
                        next_dh += 1

                    # Extend across D: each D group needs num_h consecutive H at the same
                    # h_local_start, d_in advancing by stride_d.
                    num_d = 1
                    while next_dh + num_h - 1 < num_dh_stacked:
                        # Next num_h positions form a valid D+H group?
                        d_out_check, h_out_check = dh_positions[next_dh]
                        d_in_check = d_out_check * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                        h_in_check = h_out_check * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                        expected_d_in = d_in + num_d * cfg.stride_d
                        if d_in_check != expected_d_in:
                            break
                        if d_in_check < valid_d_start or d_in_check >= valid_d_end:
                            break
                        if h_in_check != h_in:
                            break
                        # All num_h positions in this D group match?
                        valid_d_group = True
                        for h_check_idx in range(1, num_h):
                            d_out_c, h_out_c = dh_positions[next_dh + h_check_idx]
                            d_in_c = d_out_c * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                            h_in_c = h_out_c * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                            if d_in_c != expected_d_in:
                                valid_d_group = False
                                break
                            expected_h_c = h_in + h_check_idx * cfg.stride_h
                            if h_in_c != expected_h_c:
                                valid_d_group = False
                                break
                            if h_in_c < valid_h_start or h_in_c >= valid_h_end:
                                valid_d_group = False
                                break
                        if not valid_d_group:
                            break
                        num_d += 1
                        next_dh += num_h

                    total_dh_in_group = num_d * num_h

                    if total_dh_in_group == 1:
                        # Single dh position, 1D strided copy
                        dst_start = group_start_dh * w_tile_size + first_valid_out
                        dst_end = dst_start + num_w_elements
                        nisa.tensor_copy(
                            dst=raw_buf[partition_offset : partition_offset + c_in_tile_size, dst_start:dst_end],
                            src=input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .select(dim=1, index=d_local_start)
                            .select(dim=1, index=h_local_start)
                            .slice(dim=1, start=src_w_start, end=src_w_end, step=cfg.stride_w),
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )
                    elif num_d == 1:
                        # Multiple H positions, 2D strided copy
                        h_local_end = h_local_start + (num_h - 1) * cfg.stride_h + 1
                        src_view = (
                            input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .select(dim=1, index=d_local_start)
                            .slice(dim=1, start=h_local_start, end=h_local_end, step=cfg.stride_h)
                            .slice(dim=2, start=src_w_start, end=src_w_end, step=cfg.stride_w)
                        )
                        dst_view = (
                            raw_buf.slice(
                                dim=0,
                                start=partition_offset,
                                end=partition_offset + c_in_tile_size,
                                step=1,
                            )
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_stacked, w_tile_size))
                            .slice(
                                dim=1,
                                start=group_start_dh,
                                end=group_start_dh + num_h,
                                step=1,
                            )
                            .slice(dim=2, start=first_valid_out, end=last_valid_out, step=1)
                        )
                        nisa.tensor_copy(
                            dst=dst_view,
                            src=src_view,
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )
                    else:
                        # Multiple D and H positions, 3D strided copy
                        d_local_end = d_local_start + (num_d - 1) * cfg.stride_d + 1
                        h_local_end = h_local_start + (num_h - 1) * cfg.stride_h + 1
                        src_view = (
                            input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .slice(dim=1, start=d_local_start, end=d_local_end, step=cfg.stride_d)
                            .slice(dim=2, start=h_local_start, end=h_local_end, step=cfg.stride_h)
                            .slice(dim=3, start=src_w_start, end=src_w_end, step=cfg.stride_w)
                        )
                        dst_view = (
                            raw_buf.slice(
                                dim=0,
                                start=partition_offset,
                                end=partition_offset + c_in_tile_size,
                                step=1,
                            )
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_stacked, w_tile_size))
                            .slice(
                                dim=1,
                                start=group_start_dh,
                                end=group_start_dh + total_dh_in_group,
                                step=1,
                            )
                            .reshape_dim(1, (num_d, num_h))
                            .slice(dim=3, start=first_valid_out, end=last_valid_out, step=1)
                        )
                        nisa.tensor_copy(
                            dst=dst_view,
                            src=src_view,
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )

                    dh_idx = next_dh

        # Per-K dh and W valid ranges for strided matmul (as in _compute_valid_free_range).
        first_valid_dh_idx, k_w_valid_start_val = divmod(valid_free_start, w_tile_size)
        last_valid_dh_idx = (valid_free_end - 1) // w_tile_size
        k_w_valid_end_val = valid_free_end - last_valid_dh_idx * w_tile_size
        valid_dh_count = last_valid_dh_idx - first_valid_dh_idx + 1
        valid_w_count = k_w_valid_end_val - k_w_valid_start_val

        # Strided matmul requires all K positions in the tile to share a W valid range
        # (trivially true for K_REP=1).
        all_k_same_w_range = True
        if k_actual > 1 and valid_dh_count > 1 and (k_w_valid_start_val > 0 or k_w_valid_end_val < w_tile_size):
            ref_w_start_check = k_w_valid_start_val
            ref_w_end_check = k_w_valid_end_val
            for k_check_idx in range(k_actual):
                flat_k_check = used_k_positions[k_start + k_check_idx]
                _, _, k_w_check = _decompose_k_position(flat_k_check, K_h, K_w)
                base_w_check = w_start * cfg.stride_w + k_w_check * cfg.dilation_w - cfg.pad_w_left
                if base_w_check < 0:
                    this_w_start = div_ceil(-base_w_check, cfg.stride_w)
                else:
                    this_w_start = 0
                last_w_check = (w_end - 1) * cfg.stride_w + k_w_check * cfg.dilation_w - cfg.pad_w_left
                if last_w_check >= cfg.W:
                    this_w_end = w_tile_size - div_ceil(last_w_check - cfg.W + 1, cfg.stride_w)
                else:
                    this_w_end = w_tile_size
                if this_w_start != ref_w_start_check or this_w_end != ref_w_end_check:
                    all_k_same_w_range = False
                    break

        use_strided = (
            all_k_same_w_range and valid_dh_count > 1 and (k_w_valid_start_val > 0 or k_w_valid_end_val < w_tile_size)
        )

        # Strided view of the stacked buffer, matching the PSUM strided view.
        if use_strided:
            # Strided case
            input_strided = (
                raw_buf.slice(dim=0, start=0, end=stacked_filter_dim, step=1)
                .slice(dim=1, start=0, end=effective_free_dim, step=1)
                .reshape_dim(1, (num_dh_stacked, w_tile_size))
                .slice(dim=1, start=first_valid_dh_idx, end=first_valid_dh_idx + valid_dh_count, step=1)
                .slice(dim=2, start=k_w_valid_start_val, end=k_w_valid_end_val, step=1)
            )
            input_stacked_list.append(input_strided)
        else:
            # Contiguous case
            input_stacked_list.append(raw_buf[:stacked_filter_dim, valid_free_start:valid_free_end])

        valid_offsets.append(valid_free_start)
        valid_sizes.append(valid_free_size)
        valid_dh_starts.append(first_valid_dh_idx)
        valid_dh_counts.append(valid_dh_count)
        valid_w_starts.append(k_w_valid_start_val)
        valid_w_counts.append(valid_w_count)
        use_strided_flags.append(use_strided)
    return (
        input_stacked_list,
        valid_offsets,
        valid_sizes,
        valid_dh_starts,
        valid_dh_counts,
        valid_w_starts,
        valid_w_counts,
        use_strided_flags,
    )


def _conv3d_matmul(
    input_stacked_list: list[Optional[nl.NkiTensor]],
    filters_stacked_list: list[list[nl.NkiTensor]],
    psum_tiles: list[nl.NkiTensor],
    valid_offsets: list[int],
    valid_sizes: list[int],
    effective_free_dim: int,
    num_dh_stacked: int,
    w_tile_size: int,
    valid_dh_starts: list[int],
    valid_dh_counts: list[int],
    valid_w_starts: list[int],
    valid_w_counts: list[int],
    use_strided_flags: list[bool],
    c_out_tile_sizes: Optional[list[int]] = None,
    C_OUT_REP: int = 1,
    col_size: int = 128,
    dh_head_size: int = 0,
) -> None:
    """
    Accumulate matmul results across K-outer tiles into PSUM for all C_out tiles.

    Uses strided 2D views to handle W-padding without memsets: reshapes the free
    dimension as (num_dh, w_tile) and slices both the dh and W dimensions to
    exclude padded positions. This creates a strided access pattern where each
    dh row contributes only valid_w_count elements with stride w_tile_size,
    matching the flat compiler's approach.

    Tensor Engine column tiling (C_OUT_REP > 1): the num_dh_stacked = C_OUT_REP * dh_head_size
    D-H positions of a flush form C_OUT_REP bands of dh_head_size positions each. Every K-outer
    matmul is distributed across the bands: band g's matmul processes only its D-H window,
    keeps the full dh_head_size * W_tile free dim, runs on PE column tile (0, g * col_size), and
    writes to PSUM partition band [g * col_size, g * col_size + M). The stacked input from
    _scatter_input_to_stacked is reused unchanged (column tiling only slices it downstream), so
    the same full / contiguous / strided-W-padding access patterns apply per band. For
    C_OUT_REP == 1 there is a single band spanning the whole flush at partition 0 with no PE
    column tiling, i.e. byte-identical to the non-column-tiled behavior.
    """
    bands = _column_bands(num_dh_stacked, dh_head_size if C_OUT_REP > 1 else num_dh_stacked, C_OUT_REP, col_size)

    for k_outer_idx in range(len(input_stacked_list)):
        input_stacked = input_stacked_list[k_outer_idx]
        if input_stacked == None:
            continue
        k_offset = valid_offsets[k_outer_idx]
        k_size = valid_sizes[k_outer_idx]
        dh_start = valid_dh_starts[k_outer_idx]
        dh_count = valid_dh_counts[k_outer_idx]
        w_start_k = valid_w_starts[k_outer_idx]
        w_count = valid_w_counts[k_outer_idx]
        use_strided = use_strided_flags[k_outer_idx]
        for c_out_idx in range(len(psum_tiles)):
            stationary = filters_stacked_list[c_out_idx][k_outer_idx]
            for band_idx in range(len(bands)):
                part_off = bands[band_idx][0]
                band_dh_lo = bands[band_idx][1]
                band_dh_hi = bands[band_idx][2]
                col_pos = part_off if C_OUT_REP > 1 else None
                _conv3d_matmul_band(
                    input_stacked=input_stacked,
                    stationary=stationary,
                    psum_tile=psum_tiles[c_out_idx],
                    k_offset=k_offset,
                    k_size=k_size,
                    dh_start=dh_start,
                    dh_count=dh_count,
                    w_start_k=w_start_k,
                    w_count=w_count,
                    use_strided=use_strided,
                    effective_free_dim=effective_free_dim,
                    w_tile_size=w_tile_size,
                    part_off=part_off,
                    band_dh_lo=band_dh_lo,
                    band_dh_hi=band_dh_hi,
                    col_pos=col_pos,
                    col_size=col_size,
                )


def _conv3d_matmul_band(
    input_stacked: nl.NkiTensor,
    stationary: nl.NkiTensor,
    psum_tile: nl.NkiTensor,
    k_offset: int,
    k_size: int,
    dh_start: int,
    dh_count: int,
    w_start_k: int,
    w_count: int,
    use_strided: bool,
    effective_free_dim: int,
    w_tile_size: int,
    part_off: int,
    band_dh_lo: int,
    band_dh_hi: int,
    col_pos: Optional[int],
    col_size: int,
) -> None:
    """
    Issue one K-outer matmul for a single column-tiling band.

    Restricts the K-outer tile's valid region (from the scatter, expressed over the whole flush)
    to this band's D-H window [band_dh_lo, band_dh_hi), remaps it to the band's local PSUM free
    coordinates (band-local free position = global free position - band_dh_lo * w_tile_size), and
    writes to PSUM partitions [part_off, part_off + M) where M is the stationary free (C_out)
    dim. When col_pos is not None the matmul selects PE column tile (0, col_pos) with tile_size
    (128, col_size) so bands run on disjoint PE columns.

    Reuses the same three access patterns as the non-column-tiled matmul (full free / contiguous
    sub-range / strided W-padding). For a single band spanning the whole flush (C_OUT_REP == 1,
    part_off == 0, col_pos is None), every branch reduces to the original non-tiled behavior.
    """
    band_free_lo = band_dh_lo * w_tile_size
    band_free_hi = band_dh_hi * w_tile_size
    M = stationary.shape[1]
    use_tiling = col_pos != None

    if not use_strided:
        # Flat access: intersect the contiguous valid range [k_offset, k_offset + k_size) with the
        # band's flat window.
        a = max(k_offset, band_free_lo)
        b = min(k_offset + k_size, band_free_hi)
        if a >= b:
            return
        moving = input_stacked.slice(dim=1, start=a - k_offset, end=b - k_offset, step=1)
        psum_sub = (
            (psum_tile)
            .slice(dim=0, start=part_off, end=part_off + M, step=1)
            .slice(dim=1, start=a - band_free_lo, end=b - band_free_lo, step=1)
        )
    else:
        # Strided access: input_stacked is a 3D (contraction, dh, w) view; intersect its dh range
        # [dh_start, dh_start + dh_count) with the band's dh window.
        dh_a = max(dh_start, band_dh_lo)
        dh_b = min(dh_start + dh_count, band_dh_hi)
        if dh_a >= dh_b:
            return
        moving = input_stacked.slice(dim=1, start=dh_a - dh_start, end=dh_b - dh_start, step=1)
        band_dh = band_dh_hi - band_dh_lo
        psum_sub = (
            (psum_tile)
            .slice(dim=0, start=part_off, end=part_off + M, step=1)
            .slice(dim=1, start=0, end=band_dh * w_tile_size, step=1)
            .reshape_dim(1, (band_dh, w_tile_size))
            .slice(dim=1, start=dh_a - band_dh_lo, end=dh_b - band_dh_lo, step=1)
            .slice(dim=2, start=w_start_k, end=w_start_k + w_count, step=1)
        )

    if use_tiling:
        nisa.nc_matmul(
            dst=psum_sub,
            stationary=stationary,
            moving=moving,
            tile_position=(0, col_pos),
            tile_size=(nl.tile_size.pmax, col_size),
        )
    else:
        nisa.nc_matmul(dst=psum_sub, stationary=stationary, moving=moving)


def _conv3d_cin_tile(
    input_window: Optional[nl.NkiTensor],
    filters_for_cin: list[list[nl.NkiTensor]],
    psum_tiles: list[nl.NkiTensor],
    stacked_input_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_tile_size: int,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    valid_d_start: int,
    valid_d_end: int,
    valid_h_start: int,
    valid_h_end: int,
    valid_w_start: int,
    valid_w_end: int,
    used_k_positions: list[int],
    tensor_copy_engine_idx: int,
    effective_free_dim: int,
    tensor_copy_engine_modulo: int = 4,
    memset_engine_modulo: int = 2,
    c_out_tile_sizes: Optional[list[int]] = None,
    C_OUT_REP: int = 1,
    col_size: int = 128,
    dh_head_size: int = 0,
) -> None:
    """
    Process one C_in tile: scatter input into stacked layout and accumulate matmul into PSUM.

    Combines the scatter and matmul steps for a single input channel tile. First
    scatters the input window into K-replicated stacked buffers, then performs
    matmul accumulation across all C_out tiles. When C_OUT_REP > 1 the matmul is
    column-tiled across C_OUT_REP PE column tiles (see _conv3d_matmul).
    """
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    (
        input_stacked_list,
        valid_offsets,
        valid_sizes,
        v_dh_starts,
        v_dh_counts,
        v_w_starts,
        v_w_counts,
        v_strided_flags,
    ) = _scatter_input_to_stacked(
        input_window=input_window,
        stacked_input_bufs=stacked_input_bufs,
        cfg=cfg,
        c_in_tile_size=c_in_tile_size,
        dh_positions=dh_positions,
        w_start=w_start,
        w_end=w_end,
        valid_d_start=valid_d_start,
        valid_d_end=valid_d_end,
        valid_h_start=valid_h_start,
        valid_h_end=valid_h_end,
        valid_w_start=valid_w_start,
        valid_w_end=valid_w_end,
        used_k_positions=used_k_positions,
        tensor_copy_engine_idx=tensor_copy_engine_idx,
        tensor_copy_engine_modulo=tensor_copy_engine_modulo,
        memset_engine_modulo=memset_engine_modulo,
    )
    _conv3d_matmul(
        input_stacked_list=input_stacked_list,
        filters_stacked_list=filters_for_cin,
        psum_tiles=psum_tiles,
        valid_offsets=valid_offsets,
        valid_sizes=valid_sizes,
        effective_free_dim=effective_free_dim,
        num_dh_stacked=num_dh_stacked,
        w_tile_size=w_tile_size,
        valid_dh_starts=v_dh_starts,
        valid_dh_counts=v_dh_counts,
        valid_w_starts=v_w_starts,
        valid_w_counts=v_w_counts,
        use_strided_flags=v_strided_flags,
        c_out_tile_sizes=c_out_tile_sizes,
        C_OUT_REP=C_OUT_REP,
        col_size=col_size,
        dh_head_size=dh_head_size,
    )


def _load_input_window_to_sbuf_3d(
    x_in_cin: nl.NkiTensor,
    input_window_buf: nl.NkiTensor,
    cfg: Conv3dConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
) -> tuple[Optional[nl.NkiTensor], int, int, int, int, int, int]:
    """
    Load the union input window from HBM into an SBUF buffer via DMA.

    Computes the bounding box of all receptive fields for the given D-H output
    positions and W tile range, clips to valid input bounds, and issues a single
    DMA copy. Returns None if the entire window falls in padding.
    """
    c_in_tile_size = x_in_cin.shape[0]

    union_d_start, union_d_end = cfg.D, 0
    union_h_start, union_h_end = cfg.H, 0
    for dh_idx in range(len(dh_positions)):
        d_out, h_out = dh_positions[dh_idx]
        field_d_start = d_out * cfg.stride_d - cfg.pad_d_left
        field_d_end = d_out * cfg.stride_d + (cfg.K_d - 1) * cfg.dilation_d - cfg.pad_d_left + 1
        field_h_start = h_out * cfg.stride_h - cfg.pad_h_top
        field_h_end = h_out * cfg.stride_h + (cfg.K_h - 1) * cfg.dilation_h - cfg.pad_h_top + 1
        union_d_start = min(union_d_start, field_d_start)
        union_d_end = max(union_d_end, field_d_end)
        union_h_start = min(union_h_start, field_h_start)
        union_h_end = max(union_h_end, field_h_end)

    field_w_start = w_start * cfg.stride_w - cfg.pad_w_left
    field_w_end = (w_end - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w - cfg.pad_w_left + 1

    valid_d_start, valid_d_end = max(0, union_d_start), min(cfg.D, union_d_end)
    valid_h_start, valid_h_end = max(0, union_h_start), min(cfg.H, union_h_end)
    valid_w_start, valid_w_end = max(0, field_w_start), min(cfg.W, field_w_end)

    if valid_d_start >= valid_d_end or valid_h_start >= valid_h_end or valid_w_start >= valid_w_end:
        return None, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end

    d_window_size = valid_d_end - valid_d_start
    h_window_size = valid_h_end - valid_h_start
    w_window_size = valid_w_end - valid_w_start

    nisa.dma_copy(
        dst=input_window_buf[:c_in_tile_size, :d_window_size, :h_window_size, :w_window_size],
        src=x_in_cin.slice(dim=1, start=valid_d_start, end=valid_d_end, step=1)
        .slice(dim=2, start=valid_h_start, end=valid_h_end, step=1)
        .slice(dim=3, start=valid_w_start, end=valid_w_end, step=1),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )
    return input_window_buf, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end


def _conv3d_output_tile(
    x_in_batch: nl.NkiTensor,
    filters_cache: list[list[list[nl.NkiTensor]]],
    bias_cache: list[Optional[nl.NkiTensor]],
    result_sbufs: list[nl.NkiTensor],
    input_window_slots: list[nl.NkiTensor],
    stacked_input_slots: list[list[nl.NkiTensor]],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    c_out_tile_sizes: list[int],
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    psum_bank_idx: int,
    c_in_slot_offset: int = 0,
    preloaded_windows: Optional[list[tuple]] = None,
    bn_c_out_tile_base: int = 0,
    bn_stats_bufs: Optional[nl.NkiTensor] = None,
    bn_partial_iters: int = 1,
    bn_iter_idx: int = 0,
    bn_eval_true_gammas: Optional[list[nl.NkiTensor]] = None,
    bn_eval_neg_true_betas: Optional[list[nl.NkiTensor]] = None,
    residual_sbufs: Optional[list[nl.NkiTensor]] = None,
) -> None:
    """
    Compute one output tile by iterating over C_in tiles with multi-buffered I/O.

    When the number of c_out tiles exceeds NUM_PSUM_BANKS, processes them in
    sub-groups that fit within the available PSUM banks. For each sub-group:
    iterates over all C_in tiles (load input, scatter, matmul), then flushes
    PSUM results to SBUF before reusing the banks for the next sub-group.
    """
    num_c_out_tiles = len(result_sbufs)
    w_tile_size = w_end - w_start
    num_dh_positions = len(dh_positions)

    # Column tiling packs col_rep bands of band_dh positions (band_dh == whole flush at col_rep 1;
    # a ragged tail keeps band_dh == dh_head_size with the last band partial).
    col_rep, band_dh, col_size = _flush_column_layout(tile_cfg, num_dh_positions)
    per_band_free = band_dh * w_tile_size  # per-band free width fed to PSUM / eviction
    effective_free_dim = num_dh_positions * w_tile_size  # total packed D-H positions this flush

    # Fully-padded free positions get no matmul (flush-global coords).
    unwritten_free_ranges = _compute_unwritten_free_ranges(cfg, tile_cfg, dh_positions, w_start, w_end)

    # Bands (partition_offset, dh_lo, dh_hi); single band at col_rep == 1.
    column_bands = _column_bands(num_dh_positions, band_dh, col_rep, col_size)
    if col_rep > 1:
        # One PSUM bank, col_rep * col_size partitions tall x per_band_free wide (<= F_MAX).
        psum_part_dim = col_rep * col_size
        psum_free_dim = per_band_free
        evict_partitions = col_rep * col_size
    else:
        psum_part_dim = None  # per-tile size below
        psum_free_dim = per_band_free
        evict_partitions = 0

    # Process c_out tiles in sub-groups that fit within PSUM banks
    for sub_group_start in range(0, num_c_out_tiles, _NUM_PSUM_BANKS):
        sub_group_end = min(sub_group_start + _NUM_PSUM_BANKS, num_c_out_tiles)
        sub_group_size = sub_group_end - sub_group_start
        sub_c_out_tile_sizes = c_out_tile_sizes[sub_group_start:sub_group_end]

        # Allocate PSUM banks for this sub-group
        psum_tiles = []
        for sg_idx in range(sub_group_size):
            c_out_tile_idx = sub_group_start + sg_idx
            part_dim = psum_part_dim if psum_part_dim != None else c_out_tile_sizes[c_out_tile_idx]
            psum_tiles.append(
                nl.ndarray(
                    shape=(part_dim, psum_free_dim),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, (psum_bank_idx + sg_idx) % _NUM_PSUM_BANKS * _PSUM_BANK_SIZE),
                )
            )

        # Iterate over all C_in tiles for this sub-group
        c_in_tile_idx = 0
        for c_in_start in range(0, cfg.C_in, tile_cfg.P_MAX):
            c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
            c_in_tile_size = c_in_end - c_in_start

            # Multi-buffer slot selection
            global_idx = c_in_slot_offset + c_in_tile_idx
            input_window_buf = input_window_slots[global_idx % len(input_window_slots)]
            stacked_input_bufs = stacked_input_slots[global_idx % len(stacked_input_slots)]

            if preloaded_windows != None:
                input_window, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end = (
                    preloaded_windows[c_in_tile_idx]
                )
            else:
                x_in_cin = x_in_batch.slice(dim=0, start=c_in_start, end=c_in_end, step=1)
                input_window, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end = (
                    _load_input_window_to_sbuf_3d(
                        x_in_cin=x_in_cin,
                        input_window_buf=input_window_buf,
                        cfg=cfg,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                    )
                )

            # Get filters for this sub-group
            filters_for_cin = []
            for sg_idx in range(sub_group_size):
                c_out_tile_idx = sub_group_start + sg_idx
                filters_for_cin.append(filters_cache[c_out_tile_idx][c_in_tile_idx])

            _conv3d_cin_tile(
                input_window=input_window,
                filters_for_cin=filters_for_cin,
                psum_tiles=psum_tiles,
                stacked_input_bufs=stacked_input_bufs,
                cfg=cfg,
                c_in_tile_size=c_in_tile_size,
                dh_positions=dh_positions,
                w_start=w_start,
                w_end=w_end,
                valid_d_start=valid_d_start,
                valid_d_end=valid_d_end,
                valid_h_start=valid_h_start,
                valid_h_end=valid_h_end,
                valid_w_start=valid_w_start,
                valid_w_end=valid_w_end,
                used_k_positions=tile_cfg.used_k_positions,
                tensor_copy_engine_idx=c_in_tile_idx * tile_cfg.K_total_used,
                effective_free_dim=effective_free_dim,
                tensor_copy_engine_modulo=tile_cfg.tensor_copy_activation_engine_offload,
                memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
                c_out_tile_sizes=sub_c_out_tile_sizes,
                C_OUT_REP=col_rep,
                col_size=col_size,
                dh_head_size=band_dh,
            )
            c_in_tile_idx += 1

        # Flush this sub-group's PSUM results to SBUF
        sub_result_sbufs = result_sbufs[sub_group_start:sub_group_end]
        sub_bias_sbufs = bias_cache[sub_group_start:sub_group_end]
        sub_true_gammas = None
        sub_neg_true_betas = None
        if cfg.is_batch_norm_eval():
            sub_true_gammas = bn_eval_true_gammas[sub_group_start:sub_group_end]
            sub_neg_true_betas = bn_eval_neg_true_betas[sub_group_start:sub_group_end]
        sub_residual_sbufs = None
        if residual_sbufs != None:
            sub_residual_sbufs = residual_sbufs[sub_group_start:sub_group_end]
        _apply_bias_activation_and_copy(
            psum_tiles=psum_tiles,
            result_sbufs=sub_result_sbufs,
            bias_sbufs=sub_bias_sbufs,
            c_out_tile_sizes=sub_c_out_tile_sizes,
            # Eval mode folds the bias into the affine coefficients, so it is not added here.
            has_bias=cfg.has_bias and not cfg.is_batch_norm_eval(),
            # Training-mode batchnorm defers activation to the rescale phase, so stats see raw conv
            # output; eval mode has no stats, so it applies the activation right after the norm.
            has_activation=cfg.has_activation and not cfg.is_batch_norm_training(),
            activation_fn=cfg.activation_fn,
            # Eviction is per band: each band is per_band_free wide, evict_partitions spans all bands.
            effective_free_dim=per_band_free,
            unwritten_free_ranges=unwritten_free_ranges,
            memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
            batch_norm_mode=cfg.batch_norm_mode,
            bn_stats_bufs=bn_stats_bufs,
            bn_partial_iters=bn_partial_iters,
            bn_iter_idx=bn_iter_idx,
            bn_tile_offset=bn_c_out_tile_base + sub_group_start,
            evict_partitions=evict_partitions,
            column_bands=column_bands,
            w_tile_size=w_tile_size,
            bn_eval_true_gammas=sub_true_gammas,
            bn_eval_neg_true_betas=sub_neg_true_betas,
            residual_sbufs=sub_residual_sbufs,
            residual_pre_act=cfg.residual_pre_activation(),
            residual_aliases_result=cfg.residual_in_place(),
        )


def _allocate_filter_bias_buffers(
    sbm: SbufManager,
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    mem_cfg: Conv3dMemoryConfig,
    dtype: object,
) -> tuple[list[Optional[nl.NkiTensor]], list[list[nl.NkiTensor]]]:
    """
    Allocate SBUF heap buffers for bias and filter storage.

    Creates interleaved bias buffers (one per C_out tile in the interleave group)
    and filter buffers indexed by [c_in_tile][k_outer_tile].
    """
    c_out_wide = mem_cfg.c_out_interleave * tile_cfg.c_out_tile_size_max

    # Column tiling replicates bias across the C_OUT_REP bands (free dim 1, no extra per-partition
    # SBUF) so each band's eviction reads its per-channel bias.
    bias_part_dim = tile_cfg.C_OUT_REP * tile_cfg.col_size if tile_cfg.C_OUT_REP > 1 else tile_cfg.c_out_tile_size_max
    bias_bufs: list[Optional[nl.NkiTensor]] = []
    for tile_idx in range(mem_cfg.c_out_interleave):
        if cfg.has_bias:
            bias_bufs.append(
                sbm.alloc_heap(
                    shape=(bias_part_dim, 1),
                    dtype=nl.float32,
                    name=f"bias_{tile_idx}",
                )
            )
        else:
            bias_bufs.append(None)

    filter_bufs: list[list[nl.NkiTensor]] = []
    for cin_idx in range(tile_cfg.c_in_tile_count):
        cin_filters: list[nl.NkiTensor] = []
        for k_idx in range(tile_cfg.K_outer_tile_count):
            cin_filters.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.stacked_filter_dim_max, c_out_wide),
                    dtype=dtype,
                    name=f"filter_cin{cin_idx}_k{k_idx}",
                )
            )
        cin_filters_list = cin_filters
        filter_bufs.append(cin_filters_list)

    return bias_bufs, filter_bufs


def _allocate_result_slot_grid(
    sbm: SbufManager,
    mem_cfg: "Conv3dMemoryConfig",
    shape: tuple[int, int],
    dtype: object,
    name_prefix: str,
) -> list[list[nl.NkiTensor]]:
    """
    Allocate a store_pipe x c_out_interleave grid of result-shaped SBUF buffers.

    Used for both the result tiles and (with a fused residual add) the identically shaped residual
    staging tiles, so the two always agree on shape and rotation.
    """
    slots: list[list[nl.NkiTensor]] = []
    for w_slot_idx in range(mem_cfg.store_pipe):
        slot: list[nl.NkiTensor] = []
        for c_out_slot_idx in range(mem_cfg.c_out_interleave):
            slot.append(
                sbm.alloc_heap(shape=shape, dtype=dtype, name=f"{name_prefix}_{w_slot_idx}_cout{c_out_slot_idx}")
            )
        slots.append(slot)
    return slots


def _load_filters_into_bufs(
    filters: nl.NkiTensor,
    filter_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_start: int,
    c_in_end: int,
    c_out_group_start: int,
    c_out_group_end: int,
    used_k_positions: list[int],
    memset_engine_idx: int = 0,
    memset_engine_modulo: int = 2,
) -> None:
    """
    Load filter weights from HBM into K-replicated stacked SBUF buffers.

    For each K-outer tile, packs multiple filter positions along the partition
    dimension with the appropriate partition stride. Zero-initializes buffers
    when the partition stride exceeds the C_in tile size to handle padding.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    K_total_used = len(used_k_positions)
    c_in_size = c_in_end - c_in_start
    c_out_wide = c_out_group_end - c_out_group_start
    K_REP, partition_stride = _get_k_replication_params(c_in_size, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)

    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual
        raw_buf = filter_bufs[k_outer_idx]

        if partition_stride > c_in_size:
            nisa.memset(
                dst=raw_buf[:stacked_filter_dim, :c_out_wide],
                value=0.0,
                engine=_get_memset_engine(memset_engine_idx + k_outer_idx, memset_engine_modulo),
            )

        for k_rep_idx in range(k_actual):
            partition_offset = k_rep_idx * partition_stride
            flat_k_pos = used_k_positions[k_start + k_rep_idx]
            k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
            nisa.dma_copy(
                dst=raw_buf[partition_offset : partition_offset + c_in_size, :c_out_wide],
                src=filters.select(dim=0, index=k_d_idx)
                .select(dim=0, index=k_h_idx)
                .select(dim=0, index=k_w_idx)
                .slice(dim=0, start=c_in_start, end=c_in_end, step=1)
                .slice(dim=1, start=c_out_group_start, end=c_out_group_start + c_out_wide, step=1),
                dge_mode=nisa.dge_mode.hwdge,
            )


def _load_bias_and_filters_for_c_out_group(
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    c_out_group_start: int,
    c_out_group_end: int,
    pre_bias_bufs: list[Optional[nl.NkiTensor]],
    pre_filter_bufs: list[list[nl.NkiTensor]],
) -> tuple[list[int], list[Optional[nl.NkiTensor]], list[list[list[nl.NkiTensor]]]]:
    """
    Load bias and filter weights for an entire C_out interleave group.

    Loads bias per c_out tile, then loads all filter weights for the entire
    c_out group. Returns sliced views for each c_out tile.
    """
    actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)
    c_out_tile_sizes = []
    bias_cache: list[Optional[nl.NkiTensor]] = []

    for c_out_tile_idx in range(actual_group_size):
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_end = min(c_out_tile_start + tile_cfg.P_MAX, tile_cfg.C_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        c_out_tile_sizes.append(c_out_tile_size)

        if cfg.has_bias:
            bias_cache.append(
                _load_bias_into_buf(
                    bias,
                    pre_bias_bufs[c_out_tile_idx],
                    c_out_tile_start,
                    c_out_tile_end,
                    C_OUT_REP=tile_cfg.C_OUT_REP,
                    col_size=tile_cfg.col_size,
                )
            )
        else:
            bias_cache.append(None)

    K_total_used = len(tile_cfg.used_k_positions)

    for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
        c_in_start = c_in_tile_idx * tile_cfg.P_MAX
        c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
        _load_filters_into_bufs(
            filters,
            pre_filter_bufs[c_in_tile_idx],
            cfg,
            c_in_start,
            c_in_end,
            c_out_group_start,
            c_out_group_end,
            used_k_positions=tile_cfg.used_k_positions,
            memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
        )

    filters_cache: list[list[list[nl.NkiTensor]]] = []
    for c_out_tile_idx in range(actual_group_size):
        c_out_offset = c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        tile_filters: list[list[nl.NkiTensor]] = []
        for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
            c_in_size = min(tile_cfg.P_MAX, cfg.C_in - c_in_tile_idx * tile_cfg.P_MAX)
            K_REP_cin, partition_stride_cin = _get_k_replication_params(c_in_size, K_total_used)
            K_outer_tile_count_cin = div_ceil(K_total_used, K_REP_cin)
            cin_k_views: list[nl.NkiTensor] = []
            for k_outer_idx in range(K_outer_tile_count_cin):
                k_start = k_outer_idx * K_REP_cin
                k_end = min(k_start + K_REP_cin, K_total_used)
                k_actual = k_end - k_start
                stacked_filter_dim = partition_stride_cin * k_actual
                wide_buf = pre_filter_bufs[c_in_tile_idx][k_outer_idx]
                cin_k_views.append(wide_buf[:stacked_filter_dim, c_out_offset : c_out_offset + c_out_tile_size])
            tile_filters.append(cin_k_views)
        filters_cache.append(tile_filters)

    return c_out_tile_sizes, bias_cache, filters_cache


_LOW_CHAN_MAX = _PARTITION_STRIDE_64
_LOW_CHAN_SPATIAL_TILE = 8 * 1024
_LOW_CHAN_REP4_SPATIAL_TILE = 12 * 1024
_LOW_CHAN_SUBTILE = nl.tile_size.psum_fmax
_DMA_TRANSPOSE_ALIGN = 32


@dataclass
class _LowChanConfig(nl.NKIObject):
    C_IN_REPL: int
    C_IN_STRIDE: int
    packed_c_in: int
    batch_capacity: int
    spatial_tile_size: int
    input_window_size: int
    result_band_size: int
    batch_start: int
    batch_end: int
    n_prgs: int
    prg_id: int
    max_c_out_repl: int
    bn_partial_iters: int


@dataclass
class _LowChanMemoryConfig(nl.NKIObject):
    cache_all_taps: bool
    filter_slots: int
    working_set_count: int
    total_memory_required: int


@dataclass
class _LowChanSpatialTile(nl.NKIObject):
    spatial_start: int
    spatial_end: int

    def size(self) -> int:
        return self.spatial_end - self.spatial_start


@dataclass
class _LowChanSpatialRun(nl.NKIObject):
    d_out: int
    h_out: int
    w_start: int
    w_end: int
    free_start: int

    def size(self) -> int:
        return self.w_end - self.w_start


@dataclass
class _LowChanTapAccess(nl.NKIObject):
    input_flat_start: int
    input_flat_end: int
    output_free_start: int
    output_free_end: int


@dataclass
class _LowChanInputWindowReuse(nl.NKIObject):
    previous_buffer_offset: int
    prefix_length: int
    fresh_input_start: int
    fresh_input_end: int
    active_partitions: int


@dataclass
class _LowChanWorkingSet(nl.NKIObject):
    input_window: nl.NkiTensor
    result_arena: nl.NkiTensor
    # Shares result_arena unless separately staged
    residual_arena: nl.NkiTensor


@dataclass
class _LowChanBuffers(nl.NKIObject):
    bias_buf: Optional[nl.NkiTensor]
    eval_scratch: Optional[list]
    eval_true_gamma: Optional[list]
    eval_neg_true_beta: Optional[list]
    filter_bufs: list
    working_sets: list


@dataclass
class _LowChanPostTileConfig(nl.NKIObject):
    P_MAX: int
    C_OUT_REP: int
    col_size: int
    c_out_tile_size_max: int


def _get_low_chan_replication(channel_count: int) -> tuple[int, int]:
    if channel_count <= _PARTITION_STRIDE_32:
        return _MAX_K_REP_AT_32, _PARTITION_STRIDE_32
    if channel_count <= _PARTITION_STRIDE_64:
        return _MAX_K_REP_AT_64, _PARTITION_STRIDE_64
    return 1, nl.tile_size.pmax


def _get_low_chan_batch_capacity(c_in_repl: int, batch_per_program: int) -> int:
    return min(c_in_repl, batch_per_program)


def _iter_low_chan_batch_groups(
    batch_start: int,
    batch_end: int,
    batch_capacity: int,
) -> list[tuple[int, int]]:
    groups = []
    for batch_group_start in range(batch_start, batch_end, batch_capacity):
        groups.append((batch_group_start, min(batch_capacity, batch_end - batch_group_start)))
    return groups


def _low_chan_result_arena_shape(batch_capacity: int, result_band_size: int) -> tuple[int, int, int]:
    return nl.tile_size.pmax, batch_capacity, result_band_size


def _use_low_chan_merged_dma(c_out_rep: int, batch_count: int, batch_capacity: int) -> bool:
    return c_out_rep > 1 and batch_count > 1 and batch_count == batch_capacity


def _use_low_chan_merged_input_dma(
    c_in: int,
    c_in_stride: int,
    c_in_repl: int,
    batch_count: int,
) -> bool:
    # Rep-2 input DMAs are already wide enough; merging batches reduces achieved bandwidth
    return c_in_repl == 4 and batch_count > 1 and c_in == c_in_stride


def _get_low_chan_spatial_tile_limit(c_in_repl: int) -> int:
    if c_in_repl == 4:
        return _LOW_CHAN_REP4_SPATIAL_TILE
    return _LOW_CHAN_SPATIAL_TILE


def _get_c_out_replication_range(c_out: int) -> tuple[int, int]:
    min_rep = 4
    max_rep = 1
    for c_out_start in range(0, c_out, nl.tile_size.pmax):
        c_out_size = min(nl.tile_size.pmax, c_out - c_out_start)
        c_out_rep, _ = _get_c_out_replication_params(c_out_size)
        min_rep = min(min_rep, c_out_rep)
        max_rep = max(max_rep, c_out_rep)
    return min_rep, max_rep


def _iter_low_chan_spatial_tiles(total_spatial: int, spatial_tile_size: int) -> list[_LowChanSpatialTile]:
    spatial_tiles = []
    for spatial_start in range(0, total_spatial, spatial_tile_size):
        spatial_tiles.append(
            _LowChanSpatialTile(
                spatial_start=spatial_start,
                spatial_end=min(spatial_start + spatial_tile_size, total_spatial),
            )
        )
    return spatial_tiles


def _low_chan_spatial_tiles(cfg: Conv3dConfig, low_cfg: _LowChanConfig) -> list[_LowChanSpatialTile]:
    return _iter_low_chan_spatial_tiles(cfg.D_out * cfg.H_out * cfg.W_out, low_cfg.spatial_tile_size)


def _build_low_chan_spatial_runs(
    cfg: Conv3dConfig,
    spatial_start: int,
    spatial_end: int,
) -> list[_LowChanSpatialRun]:
    runs = []
    cursor = spatial_start
    while cursor < spatial_end:
        dh_out, w_start = divmod(cursor, cfg.W_out)
        d_out, h_out = divmod(dh_out, cfg.H_out)
        run_size = min(spatial_end - cursor, cfg.W_out - w_start)
        runs.append(
            _LowChanSpatialRun(
                d_out=d_out,
                h_out=h_out,
                w_start=w_start,
                w_end=w_start + run_size,
                free_start=cursor - spatial_start,
            )
        )
        cursor += run_size
    return runs


def _get_low_chan_tap_access(
    cfg: Conv3dConfig,
    run: _LowChanSpatialRun,
    k_d: int,
    k_h: int,
    k_w: int,
) -> Optional[_LowChanTapAccess]:
    d_in = run.d_out * cfg.stride_d + k_d * cfg.dilation_d - cfg.pad_d_left
    h_in = run.h_out * cfg.stride_h + k_h * cfg.dilation_h - cfg.pad_h_top
    if d_in < 0 or d_in >= cfg.D or h_in < 0 or h_in >= cfg.H:
        return None

    first_w_in = run.w_start * cfg.stride_w + k_w * cfg.dilation_w - cfg.pad_w_left
    valid_run_start = max(0, div_ceil(-first_w_in, cfg.stride_w))
    valid_run_end = min(run.size(), div_ceil(cfg.W - first_w_in, cfg.stride_w))
    if valid_run_start >= valid_run_end:
        return None

    valid_w_start = first_w_in + valid_run_start * cfg.stride_w
    valid_count = valid_run_end - valid_run_start
    input_flat_start = (d_in * cfg.H + h_in) * cfg.W + valid_w_start
    return _LowChanTapAccess(
        input_flat_start=input_flat_start,
        input_flat_end=input_flat_start + (valid_count - 1) * cfg.stride_w + 1,
        output_free_start=run.free_start + valid_run_start,
        output_free_end=run.free_start + valid_run_end,
    )


def _get_low_chan_input_window(
    cfg: Conv3dConfig,
    spatial_tile: _LowChanSpatialTile,
) -> tuple[int, int]:
    input_flat_start = cfg.D * cfg.H * cfg.W
    input_flat_end = 0
    runs = _build_low_chan_spatial_runs(cfg, spatial_tile.spatial_start, spatial_tile.spatial_end)
    for k_d, k_h, k_w in _tap_positions(cfg):
        for run in runs:
            access = _get_low_chan_tap_access(cfg, run, k_d, k_h, k_w)
            if access != None:
                input_flat_start = min(input_flat_start, access.input_flat_start)
                input_flat_end = max(input_flat_end, access.input_flat_end)
    if input_flat_start >= input_flat_end:
        return 0, 0
    return input_flat_start, input_flat_end


def _get_low_chan_input_window_reuse(
    previous_input_start: int,
    previous_input_end: int,
    current_input_start: int,
    current_input_end: int,
    previous_working_set_idx: int,
    current_working_set_idx: int,
    working_set_count: int,
    batch_count: int,
    c_in_stride: int,
) -> Optional[_LowChanInputWindowReuse]:
    if working_set_count < 2 or previous_working_set_idx == current_working_set_idx:
        return None
    if current_input_start < previous_input_start or current_input_start >= previous_input_end:
        return None
    prefix_end = min(previous_input_end, current_input_end)
    prefix_length = prefix_end - current_input_start
    if prefix_length <= 0:
        return None
    return _LowChanInputWindowReuse(
        previous_buffer_offset=current_input_start - previous_input_start,
        prefix_length=prefix_length,
        fresh_input_start=prefix_end,
        fresh_input_end=current_input_end,
        active_partitions=batch_count * c_in_stride,
    )


def _low_chan_subtiles_per_band(spatial_size: int, c_out_rep: int) -> int:
    return div_ceil(div_ceil(spatial_size, _LOW_CHAN_SUBTILE), c_out_rep)


def _build_low_chan_output_regions(
    spatial_tile: _LowChanSpatialTile,
    c_out_rep: int,
) -> list[_LowChanOutputRegion]:
    subtiles_per_band = _low_chan_subtiles_per_band(spatial_tile.size(), c_out_rep)
    regions = []
    for band_idx in range(c_out_rep):
        region_start = spatial_tile.spatial_start + band_idx * subtiles_per_band * _LOW_CHAN_SUBTILE
        if region_start >= spatial_tile.spatial_end:
            break
        regions.append(
            _LowChanOutputRegion(
                band_idx=band_idx,
                spatial_start=region_start,
                spatial_end=min(
                    region_start + subtiles_per_band * _LOW_CHAN_SUBTILE,
                    spatial_tile.spatial_end,
                ),
            )
        )
    return regions


def _build_low_chan_output_tile_groups(
    spatial_tile: _LowChanSpatialTile,
    c_out_rep: int,
) -> list[list[_LowChanOutputChunk]]:
    subtiles_per_band = _low_chan_subtiles_per_band(spatial_tile.size(), c_out_rep)
    groups = []
    # Band 0 always contributes, so every free_tile_idx yields a non-empty group
    for free_tile_idx in range(subtiles_per_band):
        group = []
        for band_idx in range(c_out_rep):
            subtile_idx = band_idx * subtiles_per_band + free_tile_idx
            chunk_start = spatial_tile.spatial_start + subtile_idx * _LOW_CHAN_SUBTILE
            if chunk_start >= spatial_tile.spatial_end:
                continue
            group.append(
                _LowChanOutputChunk(
                    band_idx=band_idx,
                    free_offset=free_tile_idx * _LOW_CHAN_SUBTILE,
                    spatial_start=chunk_start,
                    spatial_end=min(chunk_start + _LOW_CHAN_SUBTILE, spatial_tile.spatial_end),
                )
            )
        groups.append(group)
    return groups


def _compute_low_chan_unwritten_free_ranges(
    cfg: Conv3dConfig,
    tile_group: list[_LowChanOutputChunk],
    chunk_runs: list[list[_LowChanSpatialRun]],
) -> list[tuple[int, int]]:
    """Compute global flattened-output ranges that no low-channel matmul writes."""
    unwritten_ranges = []
    taps = _tap_positions(cfg)
    for chunk_idx in range(len(tile_group)):
        chunk = tile_group[chunk_idx]
        written = [False] * chunk.size()
        for k_d, k_h, k_w in taps:
            for run in chunk_runs[chunk_idx]:
                access = _get_low_chan_tap_access(cfg, run, k_d, k_h, k_w)
                if access != None:
                    for free_idx in range(access.output_free_start, access.output_free_end):
                        written[free_idx] = True

        free_idx = 0
        while free_idx < chunk.size():
            if written[free_idx]:
                free_idx += 1
                continue
            gap_start = chunk.spatial_start + free_idx
            while free_idx < chunk.size() and not written[free_idx]:
                free_idx += 1
            gap_end = chunk.spatial_start + free_idx
            if len(unwritten_ranges) > 0 and unwritten_ranges[-1][1] == gap_start:
                unwritten_ranges[-1] = (unwritten_ranges[-1][0], gap_end)
            else:
                unwritten_ranges.append((gap_start, gap_end))
    return unwritten_ranges


def _max_low_chan_output_region_size(spatial_tile_size: int, c_out_rep: int) -> int:
    subtiles_per_band = _low_chan_subtiles_per_band(spatial_tile_size, c_out_rep)
    return min(spatial_tile_size, subtiles_per_band * _LOW_CHAN_SUBTILE)


def _count_low_chan_groups(
    spatial_tiles: list[_LowChanSpatialTile],
    c_out_rep: int,
) -> int:
    group_count = 0
    for spatial_tile in spatial_tiles:
        group_count += _low_chan_subtiles_per_band(spatial_tile.size(), c_out_rep)
    return group_count


def _get_low_chan_psum_banks(rotation_idx: int, batch_count: int) -> list[int]:
    """
    Rotate each batch tile across usable PSUM banks.

    Stepping by batch_count preserves the full rotation for partial groups. The final bank
    is excluded because dma_transpose may lower to an identity matmul that uses it.
    """
    num_banks = _psum_num_banks()
    bank_start = (rotation_idx * batch_count) % num_banks
    banks = []
    for batch_slot in range(batch_count):
        banks.append((bank_start + batch_slot) % num_banks)
    return banks


def _tap_positions(cfg: Conv3dConfig) -> list[tuple[int, int, int]]:
    taps = []
    for k_d in range(cfg.K_d):
        for k_h in range(cfg.K_h):
            for k_w in range(cfg.K_w):
                taps.append((k_d, k_h, k_w))
    return taps


def _low_chan_fixed_cost(cfg: Conv3dConfig, bn_partial_iters: int) -> int:
    one_column = _heap_alloc_cost(sizeinbytes(nl.float32))
    fixed_cost = one_column if cfg.has_bias else 0
    if cfg.is_batch_norm_eval():
        fixed_cost += (len(_BN_EVAL_SCRATCH_NAMES) + _BN_EVAL_COEFF_PER_TILE) * one_column
    if cfg.is_batch_norm_training():
        full_c_out_tile_count = div_ceil(cfg.C_out, nl.tile_size.pmax)
        stats_columns = full_c_out_tile_count * bn_partial_iters * _BN_STATS_ELEMS
        fixed_cost += _heap_alloc_cost(stats_columns * sizeinbytes(nl.float32))
    return fixed_cost


def _maximize_low_chan_working_sets(
    budget: int,
    fixed_cost: int,
    filter_cost: int,
    working_set_cost: int,
    max_working_sets: int,
) -> int:
    kernel_assert(working_set_cost > 0, "low-channel working-set cost must be positive")
    available = budget - fixed_cost - filter_cost
    working_set_count = min(max_working_sets, available // working_set_cost)
    kernel_assert(
        working_set_count >= 1,
        f"low-channel kernel requires one working set: budget={budget}, fixed={fixed_cost}, "
        f"filters={filter_cost}, working_set={working_set_cost}",
    )
    return working_set_count


def _build_low_chan_plan(
    cfg: Conv3dConfig,
    dtype_size: int,
    total_sbuf_budget: int,
    transpose_taps: bool = False,
) -> tuple[_LowChanConfig, _LowChanMemoryConfig]:
    budget = total_sbuf_budget - _SBUF_RESERVE
    c_in_repl, c_in_stride = _get_low_chan_replication(cfg.C_in)
    packed_c_in = c_in_repl * c_in_stride
    min_c_out_repl, max_c_out_repl = _get_c_out_replication_range(cfg.C_out)
    tap_count = cfg.K_d * cfg.K_h * cfg.K_w

    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d_low_chan", (0, 1))
    batch_per_program = div_ceil(cfg.B, n_prgs)
    batch_start = min(prg_id * batch_per_program, cfg.B)
    batch_end = min(batch_start + batch_per_program, cfg.B)
    local_batch_count = batch_end - batch_start
    batch_capacity = _get_low_chan_batch_capacity(c_in_repl, batch_per_program)

    total_spatial = cfg.D_out * cfg.H_out * cfg.W_out
    spatial_tile_size = min(total_spatial, _get_low_chan_spatial_tile_limit(c_in_repl))
    filter_slot_cost = _heap_alloc_cost(nl.tile_size.pmax * dtype_size)
    minimum_filter_slots = min(tap_count, 2)
    filter_alignment_slack = _DMA_TRANSPOSE_ALIGN - 1 if transpose_taps else 0
    minimum_filter_cost = minimum_filter_slots * filter_slot_cost + filter_alignment_slack

    found_fit = False
    while True:
        spatial_tiles = _iter_low_chan_spatial_tiles(total_spatial, spatial_tile_size)
        input_window_size = 1
        for spatial_tile in spatial_tiles:
            input_start, input_end = _get_low_chan_input_window(cfg, spatial_tile)
            input_window_size = max(input_window_size, input_end - input_start)

        result_band_size = _max_low_chan_output_region_size(spatial_tile_size, min_c_out_repl)
        groups_per_batch = _count_low_chan_groups(spatial_tiles, min_c_out_repl)
        bn_partial_iters = max(1, batch_per_program * groups_per_batch)
        fixed_cost = _low_chan_fixed_cost(cfg, bn_partial_iters)

        input_cost = _heap_alloc_cost(input_window_size * dtype_size)
        result_arena_cost = _heap_alloc_cost(batch_capacity * result_band_size * dtype_size)
        main_result_buffers = 1 + (1 if cfg.needs_residual_staging_grid() else 0)
        working_set_cost = input_cost + main_result_buffers * result_arena_cost

        result_cost = _heap_alloc_cost(result_band_size * dtype_size)
        rescale_result_buffers = 1 + (1 if cfg.is_batch_norm_training() and cfg.has_residual_add() else 0)
        post_peak_cost = (
            fixed_cost + rescale_result_buffers * result_cost if cfg.is_batch_norm_training() else fixed_cost
        )
        minimum_hot_cost = fixed_cost + minimum_filter_cost + working_set_cost
        if max(minimum_hot_cost, post_peak_cost) <= budget:
            found_fit = True
            break
        if spatial_tile_size == 1:
            break
        spatial_tile_size //= 2

    kernel_assert(
        found_fit,
        f"conv3d_low_chan cannot fit one row-tiled input/output working set and "
        f"{minimum_filter_slots} tap buffers in {budget} B/partition",
    )

    batch_group_count = max(1, div_ceil(local_batch_count, batch_capacity))
    c_out_tile_count = div_ceil(cfg.C_out, nl.tile_size.pmax)
    max_working_sets = max(1, batch_group_count * c_out_tile_count * len(spatial_tiles))
    working_set_count = _maximize_low_chan_working_sets(
        budget=budget,
        fixed_cost=fixed_cost,
        filter_cost=minimum_filter_cost,
        working_set_cost=working_set_cost,
        max_working_sets=max_working_sets,
    )

    cache_all_cost = tap_count * filter_slot_cost + filter_alignment_slack
    cache_all_taps = (
        max(
            fixed_cost + cache_all_cost + working_set_count * working_set_cost,
            post_peak_cost,
        )
        <= budget
    )
    filter_slots = tap_count if cache_all_taps else minimum_filter_slots
    filter_cost = filter_slots * filter_slot_cost + filter_alignment_slack
    used = max(
        fixed_cost + filter_cost + working_set_count * working_set_cost,
        post_peak_cost,
    )
    kernel_assert(used <= budget, f"low-channel memory plan uses {used} bytes/partition, budget={budget}")

    low_cfg = _LowChanConfig(
        C_IN_REPL=c_in_repl,
        C_IN_STRIDE=c_in_stride,
        packed_c_in=packed_c_in,
        batch_capacity=batch_capacity,
        spatial_tile_size=spatial_tile_size,
        input_window_size=input_window_size,
        result_band_size=result_band_size,
        batch_start=batch_start,
        batch_end=batch_end,
        n_prgs=n_prgs,
        prg_id=prg_id,
        max_c_out_repl=max_c_out_repl,
        bn_partial_iters=bn_partial_iters,
    )
    mem_cfg = _LowChanMemoryConfig(
        cache_all_taps=cache_all_taps,
        filter_slots=filter_slots,
        working_set_count=working_set_count,
        total_memory_required=used,
    )
    get_logger("conv3d_low_chan").info(
        f"LowChanConfig: C_in_repl={c_in_repl}, batch_capacity={batch_capacity}, "
        f"spatial_tile={spatial_tile_size}, "
        f"input_window={input_window_size}, result_band={result_band_size}, "
        f"batch=[{batch_start},{batch_end})"
    )
    get_logger("conv3d_low_chan").info(
        f"MemoryConfig: cache_all_taps={cache_all_taps}, filter_slots={filter_slots}, "
        f"working_sets={working_set_count}, total_memory_required={used}, "
        f"budget={budget}, unallocated={budget - used}"
    )
    return low_cfg, mem_cfg


def _load_low_chan_tap(
    dst: nl.NkiTensor,
    filters: nl.NkiTensor,
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    batch_count: int,
    tap: tuple[int, int, int],
    c_out_start: int,
    c_out_size: int,
    transpose_taps: bool,
) -> None:
    k_d, k_h, k_w = tap
    if not transpose_taps:
        nisa.dma_copy(
            dst=dst[: cfg.C_in, :c_out_size],
            src=filters.select(dim=0, index=k_d)
            .select(dim=0, index=k_h)
            .select(dim=0, index=k_w)
            .slice(dim=0, start=0, end=cfg.C_in, step=1)
            .slice(dim=1, start=c_out_start, end=c_out_start + c_out_size, step=1),
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        physical_k_d = cfg.K_d - 1 - k_d
        physical_k_h = cfg.K_h - 1 - k_h
        physical_k_w = cfg.K_w - 1 - k_w
        src_tap = (
            filters.select(dim=0, index=physical_k_d)
            .select(dim=0, index=physical_k_h)
            .select(dim=0, index=physical_k_w)
            .slice(dim=0, start=c_out_start, end=c_out_start + c_out_size, step=1)
            .slice(dim=1, start=0, end=cfg.C_in, step=1)
        )
        nisa.dma_transpose(dst=dst[: cfg.C_in, :c_out_size], src=src_tap)

    base_tap = dst[: low_cfg.C_IN_STRIDE, :c_out_size]
    for row_tile_idx in range(1, batch_count):
        row_start = row_tile_idx * low_cfg.C_IN_STRIDE
        nisa.tensor_copy(
            dst=dst[row_start : row_start + low_cfg.C_IN_STRIDE, :c_out_size],
            src=base_tap,
        )


def _load_low_chan_input_window(
    x_in: nl.NkiTensor,
    input_window: nl.NkiTensor,
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    batch_group_start: int,
    batch_count: int,
    input_flat_start: int,
    input_flat_end: int,
    destination_offset: int = 0,
) -> None:
    window_size = input_flat_end - input_flat_start
    if window_size == 0:
        return
    if _use_low_chan_merged_input_dma(
        cfg.C_in,
        low_cfg.C_IN_STRIDE,
        low_cfg.C_IN_REPL,
        batch_count,
    ):
        x_flat = (
            x_in.slice(
                dim=0,
                start=batch_group_start,
                end=batch_group_start + batch_count,
                step=1,
            )
            .slice(dim=1, start=0, end=cfg.C_in, step=1)
            .flatten_dims(2, 4)
            .slice(dim=2, start=input_flat_start, end=input_flat_end, step=1)
            .flatten_dims(0, 1)
        )
        nisa.dma_copy(
            dst=input_window[
                : batch_count * low_cfg.C_IN_STRIDE,
                destination_offset : destination_offset + window_size,
            ],
            src=x_flat,
            dge_mode=nisa.dge_mode.hwdge,
        )
        return
    for batch_slot in range(batch_count):
        batch_idx = batch_group_start + batch_slot
        row_start = batch_slot * low_cfg.C_IN_STRIDE
        x_flat = x_in.select(dim=0, index=batch_idx).slice(dim=0, start=0, end=cfg.C_in, step=1).flatten_dims(1, 3)
        nisa.dma_copy(
            dst=input_window[
                row_start : row_start + cfg.C_in,
                destination_offset : destination_offset + window_size,
            ],
            src=x_flat.slice(dim=1, start=input_flat_start, end=input_flat_end, step=1),
            dge_mode=nisa.dge_mode.hwdge,
        )


def _stage_low_chan_input_window(
    x_in: nl.NkiTensor,
    input_window: nl.NkiTensor,
    previous_input_window: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    batch_group_start: int,
    batch_count: int,
    reuse: Optional[_LowChanInputWindowReuse],
    input_flat_start: int,
    input_flat_end: int,
) -> None:
    if reuse == None:
        _load_low_chan_input_window(
            x_in=x_in,
            input_window=input_window,
            cfg=cfg,
            low_cfg=low_cfg,
            batch_group_start=batch_group_start,
            batch_count=batch_count,
            input_flat_start=input_flat_start,
            input_flat_end=input_flat_end,
        )
        return
    nisa.tensor_copy(
        dst=input_window[: reuse.active_partitions, : reuse.prefix_length],
        src=previous_input_window[
            : reuse.active_partitions,
            reuse.previous_buffer_offset : reuse.previous_buffer_offset + reuse.prefix_length,
        ],
        engine=nisa.engine.vector,
    )
    _load_low_chan_input_window(
        x_in=x_in,
        input_window=input_window,
        cfg=cfg,
        low_cfg=low_cfg,
        batch_group_start=batch_group_start,
        batch_count=batch_count,
        input_flat_start=reuse.fresh_input_start,
        input_flat_end=reuse.fresh_input_end,
        destination_offset=reuse.prefix_length,
    )


def _accumulate_low_chan_tap(
    input_window: nl.NkiTensor,
    stationary: nl.NkiTensor,
    psum: nl.NkiTensor,
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    tap: tuple[int, int, int],
    chunk: _LowChanOutputChunk,
    runs: list[_LowChanSpatialRun],
    input_flat_start: int,
    batch_slot: int,
    c_out_size: int,
    col_size: int,
) -> None:
    k_d, k_h, k_w = tap
    row_start = batch_slot * low_cfg.C_IN_STRIDE
    part_start = chunk.band_idx * col_size
    for run in runs:
        access = _get_low_chan_tap_access(cfg, run, k_d, k_h, k_w)
        if access == None:
            continue
        stage_start = access.input_flat_start - input_flat_start
        stage_end = access.input_flat_end - input_flat_start
        nisa.nc_matmul(
            dst=psum[
                part_start : part_start + c_out_size,
                access.output_free_start : access.output_free_end,
            ],
            stationary=stationary[
                row_start : row_start + low_cfg.C_IN_STRIDE,
                :c_out_size,
            ],
            moving=input_window[
                row_start : row_start + low_cfg.C_IN_STRIDE,
                stage_start : stage_end : cfg.stride_w,
            ],
            accumulate=None,
            tile_position=(row_start, part_start),
            tile_size=(low_cfg.C_IN_STRIDE, col_size),
        )


def _dma_merged_result_regions(
    hbm_tensor: nl.NkiTensor,
    result_arena: nl.NkiTensor,
    batch_group_start: int,
    active_batches: int,
    c_out_start: int,
    c_out_size: int,
    col_size: int,
    regions: list[_LowChanOutputRegion],
    to_hbm: bool,
) -> None:
    batch_group_end = batch_group_start + active_batches
    for region in regions:
        part_start = region.band_idx * col_size
        sbuf_view = (
            result_arena.slice(dim=0, start=part_start, end=part_start + c_out_size, step=1)
            .slice(dim=1, start=0, end=active_batches, step=1)
            .slice(dim=2, start=0, end=region.size(), step=1)
        )
        hbm_view = (
            hbm_tensor.slice(dim=0, start=batch_group_start, end=batch_group_end, step=1)
            .slice(dim=1, start=c_out_start, end=c_out_start + c_out_size, step=1)
            .flatten_dims(2, 4)
            .slice(
                dim=2,
                start=region.spatial_start,
                end=region.spatial_end,
                step=1,
            )
            .permute((1, 0, 2))
        )
        nisa.dma_copy(
            dst=hbm_view if to_hbm else sbuf_view,
            src=sbuf_view if to_hbm else hbm_view,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )


def _dma_low_chan_arena(
    hbm_tensor: nl.NkiTensor,
    arena: nl.NkiTensor,
    use_merged: bool,
    batch_group_start: int,
    batch_count: int,
    c_out_start: int,
    c_out_size: int,
    col_size: int,
    regions: list[_LowChanOutputRegion],
    to_hbm: bool,
) -> None:
    if use_merged:
        _dma_merged_result_regions(
            hbm_tensor=hbm_tensor,
            result_arena=arena,
            batch_group_start=batch_group_start,
            active_batches=batch_count,
            c_out_start=c_out_start,
            c_out_size=c_out_size,
            col_size=col_size,
            regions=regions,
            to_hbm=to_hbm,
        )
        return
    for batch_slot in range(batch_count):
        _dma_low_chan_result_regions(
            hbm_tensor=hbm_tensor,
            result_sbuf=arena.select(dim=1, index=batch_slot),
            batch_idx=batch_group_start + batch_slot,
            c_out_start=c_out_start,
            c_out_size=c_out_size,
            col_size=col_size,
            regions=regions,
            to_hbm=to_hbm,
        )


def _allocate_low_chan_buffers(
    sbm: SbufManager,
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    mem_cfg: _LowChanMemoryConfig,
    dtype: object,
    transpose_taps: bool = False,
) -> _LowChanBuffers:
    bias_buf = None
    if cfg.has_bias:
        bias_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bias")

    eval_scratch = None
    eval_true_gamma = None
    eval_neg_true_beta = None
    if cfg.is_batch_norm_eval():
        eval_scratch, eval_true_gamma, eval_neg_true_beta = _allocate_bn_eval_buffers(
            sbm=sbm,
            # Coefficients span all partition bands
            coeff_part_dim=nl.tile_size.pmax,
            c_out_interleave=1,
        )

    # Zero aligned rows not written by DMA; non-finite SBUF data would survive multiplication by 0
    has_partition_gap = cfg.C_in < low_cfg.C_IN_STRIDE

    filter_bufs = []
    for tap_slot in range(mem_cfg.filter_slots):
        filter_buf = sbm.alloc_heap(
            shape=(low_cfg.packed_c_in, nl.tile_size.pmax),
            dtype=dtype,
            name=f"low_filter_{tap_slot}",
            align=_DMA_TRANSPOSE_ALIGN if transpose_taps else None,
        )
        if has_partition_gap:
            # Zero band 0; tap replication copies the gap to active bands. Whole-band memset
            # is required because arbitrary C_in offsets are unsupported
            nisa.memset(dst=filter_buf[: low_cfg.C_IN_STRIDE, :], value=0.0)
        filter_bufs.append(filter_buf)

    working_sets = []
    for working_set_idx in range(mem_cfg.working_set_count):
        input_window = sbm.alloc_heap(
            shape=(low_cfg.packed_c_in, low_cfg.input_window_size),
            dtype=dtype,
            name=f"low_input_{working_set_idx}",
        )
        # Input windows lack tap replication, so clear every band at once
        if has_partition_gap:
            nisa.memset(dst=input_window, value=0.0)
        result_arena = sbm.alloc_heap(
            shape=_low_chan_result_arena_shape(low_cfg.batch_capacity, low_cfg.result_band_size),
            dtype=dtype,
            name=f"low_result_{working_set_idx}",
        )
        residual_arena = result_arena
        if cfg.needs_residual_staging_grid():
            residual_arena = sbm.alloc_heap(
                shape=_low_chan_result_arena_shape(low_cfg.batch_capacity, low_cfg.result_band_size),
                dtype=dtype,
                name=f"low_residual_{working_set_idx}",
            )
        working_sets.append(
            _LowChanWorkingSet(
                input_window=input_window,
                result_arena=result_arena,
                residual_arena=residual_arena,
            )
        )

    return _LowChanBuffers(
        bias_buf=bias_buf,
        eval_scratch=eval_scratch,
        eval_true_gamma=eval_true_gamma,
        eval_neg_true_beta=eval_neg_true_beta,
        filter_bufs=filter_bufs,
        working_sets=working_sets,
    )


def _load_output_tile_postprocess(
    bias: Optional[nl.NkiTensor],
    gamma: Optional[nl.NkiTensor],
    beta: Optional[nl.NkiTensor],
    running_means: Optional[nl.NkiTensor],
    running_variances: Optional[nl.NkiTensor],
    bias_buf: Optional[nl.NkiTensor],
    eval_scratch: Optional[list[nl.NkiTensor]],
    eval_true_gamma: Optional[list[nl.NkiTensor]],
    eval_neg_true_beta: Optional[list[nl.NkiTensor]],
    cfg: Conv3dConfig,
    c_out_start: int,
    c_out_size: int,
    c_out_rep: int,
    col_size: int,
    batch_norm_eps: float,
) -> tuple[Optional[nl.NkiTensor], Optional[list[nl.NkiTensor]], Optional[list[nl.NkiTensor]]]:
    bias_cache = None
    if cfg.has_bias:
        bias_cache = _load_bias_into_buf(
            bias=bias,
            bias_buf=bias_buf,
            c_out_start=c_out_start,
            c_out_end=c_out_start + c_out_size,
            C_OUT_REP=c_out_rep,
            col_size=col_size,
        )

    tile_true_gamma = None
    tile_neg_true_beta = None
    if cfg.is_batch_norm_eval():
        post_tile_cfg = _LowChanPostTileConfig(
            P_MAX=nl.tile_size.pmax,
            C_OUT_REP=c_out_rep,
            col_size=col_size,
            c_out_tile_size_max=c_out_size,
        )
        _load_bn_eval_scales_for_c_out_group(
            gamma=gamma,
            beta=beta,
            running_means=running_means,
            running_variances=running_variances,
            bias_cache=[bias_cache],
            cfg=cfg,
            tile_cfg=post_tile_cfg,
            batch_norm_eps=batch_norm_eps,
            c_out_tile_sizes=[c_out_size],
            c_out_group_start=c_out_start,
            scratch_bufs=eval_scratch,
            true_gamma_bufs=eval_true_gamma,
            neg_true_beta_bufs=eval_neg_true_beta,
            identity_mask=_identity_shuffle_mask(),
        )
        tile_true_gamma = eval_true_gamma
        tile_neg_true_beta = eval_neg_true_beta
    return bias_cache, tile_true_gamma, tile_neg_true_beta


def _flush_low_chan_tile_group(
    filters: nl.NkiTensor,
    filter_bufs: list[nl.NkiTensor],
    taps: list[tuple[int, int, int]],
    input_window: nl.NkiTensor,
    result_arena: nl.NkiTensor,
    residual_arena: nl.NkiTensor,
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    mem_cfg: _LowChanMemoryConfig,
    tile_group: list[_LowChanOutputChunk],
    batch_group_start: int,
    batch_count: int,
    c_out_start: int,
    c_out_size: int,
    col_size: int,
    input_flat_start: int,
    psum_rotation_idx: int,
    bn_iter_base: int,
    groups_per_batch: int,
    bias_cache: Optional[nl.NkiTensor],
    tile_true_gamma: Optional[list[nl.NkiTensor]],
    tile_neg_true_beta: Optional[list[nl.NkiTensor]],
    bn_stats_bufs: Optional[nl.NkiTensor],
    transpose_taps: bool,
) -> None:
    max_chunk_size, active_partitions, free_offset = _low_chan_post_group_extent(tile_group, c_out_size, col_size)
    chunk_runs = []
    for chunk in tile_group:
        chunk_runs.append(_build_low_chan_spatial_runs(cfg, chunk.spatial_start, chunk.spatial_end))
    unwritten_free_ranges = _compute_low_chan_unwritten_free_ranges(cfg, tile_group, chunk_runs)
    psum_banks = _get_low_chan_psum_banks(psum_rotation_idx, batch_count)
    psum_tiles = []
    for batch_slot in range(batch_count):
        psum_tiles.append(
            nl.ndarray(
                shape=(nl.tile_size.pmax, max_chunk_size),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, psum_banks[batch_slot] * _PSUM_BANK_SIZE),
            )
        )

    if not mem_cfg.cache_all_taps:
        _load_low_chan_tap(
            dst=filter_bufs[0],
            filters=filters,
            cfg=cfg,
            low_cfg=low_cfg,
            batch_count=batch_count,
            tap=taps[0],
            c_out_start=c_out_start,
            c_out_size=c_out_size,
            transpose_taps=transpose_taps,
        )

    for tap_idx in range(len(taps)):
        stationary = filter_bufs[tap_idx % mem_cfg.filter_slots]
        next_tap_idx = tap_idx + 1
        if not mem_cfg.cache_all_taps and next_tap_idx < len(taps):
            _load_low_chan_tap(
                dst=filter_bufs[next_tap_idx % mem_cfg.filter_slots],
                filters=filters,
                cfg=cfg,
                low_cfg=low_cfg,
                batch_count=batch_count,
                tap=taps[next_tap_idx],
                c_out_start=c_out_start,
                c_out_size=c_out_size,
                transpose_taps=transpose_taps,
            )

        for batch_slot in range(batch_count):
            for chunk_idx in range(len(tile_group)):
                _accumulate_low_chan_tap(
                    input_window=input_window,
                    stationary=stationary,
                    psum=psum_tiles[batch_slot],
                    cfg=cfg,
                    low_cfg=low_cfg,
                    tap=taps[tap_idx],
                    chunk=tile_group[chunk_idx],
                    runs=chunk_runs[chunk_idx],
                    input_flat_start=input_flat_start,
                    batch_slot=batch_slot,
                    c_out_size=c_out_size,
                    col_size=col_size,
                )

    column_bands = []
    for chunk in tile_group:
        column_bands.append((chunk.band_idx * col_size, chunk.spatial_start, chunk.spatial_end))

    for batch_slot in range(batch_count):
        result_window = result_arena.select(dim=1, index=batch_slot)[
            :,
            free_offset : free_offset + max_chunk_size,
        ]
        residual_window = None
        if cfg.residual_in_main_loop():
            residual_window = residual_arena.select(dim=1, index=batch_slot)[
                :,
                free_offset : free_offset + max_chunk_size,
            ]
        local_batch_idx = batch_group_start + batch_slot - low_cfg.batch_start
        _apply_bias_activation_and_copy(
            psum_tiles=[psum_tiles[batch_slot]],
            result_sbufs=[result_window],
            bias_sbufs=[bias_cache],
            c_out_tile_sizes=[c_out_size],
            has_bias=cfg.has_bias and not cfg.is_batch_norm_eval(),
            has_activation=cfg.has_activation,
            activation_fn=cfg.activation_fn,
            effective_free_dim=max_chunk_size,
            unwritten_free_ranges=unwritten_free_ranges,
            batch_norm_mode=cfg.batch_norm_mode,
            evict_partitions=active_partitions,
            column_bands=column_bands,
            w_tile_size=1,
            bn_eval_true_gammas=tile_true_gamma,
            bn_eval_neg_true_betas=tile_neg_true_beta,
            residual_sbufs=[residual_window] if residual_window != None else None,
            residual_pre_act=cfg.residual_pre_activation(),
            residual_aliases_result=cfg.residual_in_place(),
            bn_stats_bufs=bn_stats_bufs,
            bn_partial_iters=low_cfg.bn_partial_iters,
            bn_iter_idx=local_batch_idx * groups_per_batch + bn_iter_base,
            bn_tile_offset=c_out_start // nl.tile_size.pmax,
        )


def _execute_low_chan(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    gamma: Optional[nl.NkiTensor],
    beta: Optional[nl.NkiTensor],
    running_means: Optional[nl.NkiTensor],
    running_variances: Optional[nl.NkiTensor],
    conv_out: nl.NkiTensor,
    residuals_in: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    mem_cfg: _LowChanMemoryConfig,
    batch_norm_eps: float,
    bufs: _LowChanBuffers,
    bn_stats_bufs: Optional[nl.NkiTensor],
    transpose_taps: bool,
) -> None:
    bias_buf = bufs.bias_buf
    eval_scratch = bufs.eval_scratch
    eval_true_gamma = bufs.eval_true_gamma
    eval_neg_true_beta = bufs.eval_neg_true_beta
    filter_bufs = bufs.filter_bufs
    working_sets = bufs.working_sets
    taps = _tap_positions(cfg)
    spatial_tiles = _low_chan_spatial_tiles(cfg, low_cfg)
    tile_input_windows = []
    for spatial_tile in spatial_tiles:
        tile_input_windows.append(_get_low_chan_input_window(cfg, spatial_tile))
    working_set_idx = 0
    psum_rotation_idx = 0

    batch_groups = _iter_low_chan_batch_groups(
        batch_start=low_cfg.batch_start,
        batch_end=low_cfg.batch_end,
        batch_capacity=low_cfg.batch_capacity,
    )
    for batch_group_start, batch_count in batch_groups:
        for c_out_start in range(0, cfg.C_out, nl.tile_size.pmax):
            c_out_size = min(nl.tile_size.pmax, cfg.C_out - c_out_start)
            c_out_rep, col_size = _get_c_out_replication_params(c_out_size)
            bias_cache, tile_true_gamma, tile_neg_true_beta = _load_output_tile_postprocess(
                bias=bias,
                gamma=gamma,
                beta=beta,
                running_means=running_means,
                running_variances=running_variances,
                bias_buf=bias_buf,
                eval_scratch=eval_scratch,
                eval_true_gamma=eval_true_gamma,
                eval_neg_true_beta=eval_neg_true_beta,
                cfg=cfg,
                c_out_start=c_out_start,
                c_out_size=c_out_size,
                c_out_rep=c_out_rep,
                col_size=col_size,
                batch_norm_eps=batch_norm_eps,
            )

            if mem_cfg.cache_all_taps:
                for tap_idx in range(len(taps)):
                    _load_low_chan_tap(
                        dst=filter_bufs[tap_idx],
                        filters=filters,
                        cfg=cfg,
                        low_cfg=low_cfg,
                        batch_count=batch_count,
                        tap=taps[tap_idx],
                        c_out_start=c_out_start,
                        c_out_size=c_out_size,
                        transpose_taps=transpose_taps,
                    )

            groups_per_batch = _count_low_chan_groups(spatial_tiles, c_out_rep)
            group_prefix = 0
            previous_input_window = None
            previous_input_flat_start = 0
            previous_input_flat_end = 0
            previous_working_set_idx = -1
            for spatial_tile_idx in range(len(spatial_tiles)):
                spatial_tile = spatial_tiles[spatial_tile_idx]
                current_working_set_idx = working_set_idx % len(working_sets)
                working_set = working_sets[current_working_set_idx]
                input_window = working_set.input_window
                input_flat_start, input_flat_end = tile_input_windows[spatial_tile_idx]
                reuse = None
                # Rep-2 does not have enough Vector Engine slack to hide the prefix copy
                if previous_input_window != None and c_out_rep == 4:
                    reuse = _get_low_chan_input_window_reuse(
                        previous_input_start=previous_input_flat_start,
                        previous_input_end=previous_input_flat_end,
                        current_input_start=input_flat_start,
                        current_input_end=input_flat_end,
                        previous_working_set_idx=previous_working_set_idx,
                        current_working_set_idx=current_working_set_idx,
                        working_set_count=len(working_sets),
                        batch_count=batch_count,
                        c_in_stride=low_cfg.C_IN_STRIDE,
                    )
                _stage_low_chan_input_window(
                    x_in=x_in,
                    input_window=input_window,
                    previous_input_window=previous_input_window,
                    cfg=cfg,
                    low_cfg=low_cfg,
                    batch_group_start=batch_group_start,
                    batch_count=batch_count,
                    reuse=reuse,
                    input_flat_start=input_flat_start,
                    input_flat_end=input_flat_end,
                )

                regions = _build_low_chan_output_regions(spatial_tile, c_out_rep)
                tile_groups = _build_low_chan_output_tile_groups(spatial_tile, c_out_rep)
                use_merged_dma = _use_low_chan_merged_dma(
                    c_out_rep=c_out_rep,
                    batch_count=batch_count,
                    batch_capacity=low_cfg.batch_capacity,
                )
                if cfg.residual_in_main_loop():
                    _dma_low_chan_arena(
                        hbm_tensor=residuals_in,
                        arena=working_set.residual_arena,
                        use_merged=use_merged_dma,
                        batch_group_start=batch_group_start,
                        batch_count=batch_count,
                        c_out_start=c_out_start,
                        c_out_size=c_out_size,
                        col_size=col_size,
                        regions=regions,
                        to_hbm=False,
                    )

                for tile_group_idx in range(len(tile_groups)):
                    _flush_low_chan_tile_group(
                        filters=filters,
                        filter_bufs=filter_bufs,
                        taps=taps,
                        input_window=input_window,
                        result_arena=working_set.result_arena,
                        residual_arena=working_set.residual_arena,
                        cfg=cfg,
                        low_cfg=low_cfg,
                        mem_cfg=mem_cfg,
                        tile_group=tile_groups[tile_group_idx],
                        batch_group_start=batch_group_start,
                        batch_count=batch_count,
                        c_out_start=c_out_start,
                        c_out_size=c_out_size,
                        col_size=col_size,
                        input_flat_start=input_flat_start,
                        psum_rotation_idx=psum_rotation_idx,
                        bn_iter_base=group_prefix + tile_group_idx,
                        groups_per_batch=groups_per_batch,
                        bias_cache=bias_cache,
                        tile_true_gamma=tile_true_gamma,
                        tile_neg_true_beta=tile_neg_true_beta,
                        bn_stats_bufs=bn_stats_bufs,
                        transpose_taps=transpose_taps,
                    )
                    psum_rotation_idx += 1

                _dma_low_chan_arena(
                    hbm_tensor=conv_out,
                    arena=working_set.result_arena,
                    use_merged=use_merged_dma,
                    batch_group_start=batch_group_start,
                    batch_count=batch_count,
                    c_out_start=c_out_start,
                    c_out_size=c_out_size,
                    col_size=col_size,
                    regions=regions,
                    to_hbm=True,
                )
                group_prefix += len(tile_groups)
                previous_input_window = input_window
                previous_input_flat_start = input_flat_start
                previous_input_flat_end = input_flat_end
                previous_working_set_idx = current_working_set_idx
                working_set_idx += 1


def _get_low_chan_band_counts(
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    c_out_rep: int,
) -> list[int]:
    counts = []
    for _ in range(c_out_rep):
        counts.append(0)
    for spatial_tile in _low_chan_spatial_tiles(cfg, low_cfg):
        for region in _build_low_chan_output_regions(spatial_tile, c_out_rep):
            counts[region.band_idx] += region.size()
    return counts


def _build_low_chan_bn_finalize_plan(cfg: Conv3dConfig, low_cfg: _LowChanConfig) -> _BnFinalizePlan:
    p_max = nl.tile_size.pmax
    tile_size_full = min(p_max, cfg.C_out)
    local_batch = low_cfg.batch_end - low_cfg.batch_start
    spatial_out = cfg.D_out * cfg.H_out * cfg.W_out

    # Tail C_out tiles can use different band layouts; scale counts to this batch shard
    band_plans = []
    max_bands = 1
    for c_out_start in range(0, cfg.C_out, p_max):
        c_out_size = min(p_max, cfg.C_out - c_out_start)
        c_out_rep, col_size = _get_c_out_replication_params(c_out_size)
        per_batch_counts = _get_low_chan_band_counts(cfg, low_cfg, c_out_rep)
        band_counts = []
        for band_idx in range(c_out_rep):
            band_counts.append(local_batch * per_batch_counts[band_idx])
        kernel_assert(
            (c_out_rep - 1) * col_size + c_out_size <= p_max,
            f"band span {(c_out_rep - 1) * col_size + c_out_size} exceeds {p_max} partitions",
        )
        max_bands = max(max_bands, c_out_rep)
        band_plans.append(_BnBandPlan(n_bands=c_out_rep, band_stride=col_size, band_counts=band_counts))

    exchange = low_cfg.n_prgs > 1
    return _BnFinalizePlan(
        p_max=p_max,
        c_out_total=cfg.C_out,
        # Batch sharding preserves the full C_out range
        own_c_out_start=0,
        own_c_out_end=cfg.C_out,
        stats_per_tile=low_cfg.bn_partial_iters * _BN_STATS_ELEMS,
        own_count=local_batch * spatial_out,
        exchange=exchange,
        prg_id=low_cfg.prg_id,
        pool_triples=max(max_bands, 2 if exchange else 1),
        band_plans=band_plans,
    )


def _prepare_low_chan_training_result(
    cfg: Conv3dConfig,
    low_cfg: _LowChanConfig,
    sbm: SbufManager,
    heap_depth: int,
    conv_dtype: object,
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
) -> _Conv3dComputeResult:
    gamma_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_gamma")
    beta_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_beta")
    true_gamma_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_true_gamma")
    neg_true_beta_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_neg_true_beta")
    run_running_mean_buf = sbm.alloc_heap(
        shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_run_running_mean"
    )
    run_new_mean_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_run_new_mean")
    run_running_var_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_run_running_var")
    run_new_var_buf = sbm.alloc_heap(shape=(nl.tile_size.pmax, 1), dtype=nl.float32, name="low_bn_run_new_var")
    reload_buf = sbm.alloc_heap(
        shape=(nl.tile_size.pmax, low_cfg.result_band_size),
        dtype=conv_dtype,
        name="low_bn_reload",
    )
    residual_reload_buf = None
    if cfg.has_residual_add():
        residual_reload_buf = sbm.alloc_heap(
            shape=(nl.tile_size.pmax, low_cfg.result_band_size),
            dtype=conv_dtype,
            name="low_bn_residual_reload",
        )

    spatial_tiles = _low_chan_spatial_tiles(cfg, low_cfg)
    low_chan_tile_layouts = []
    for c_out_start in range(0, cfg.C_out, nl.tile_size.pmax):
        c_out_size = min(nl.tile_size.pmax, cfg.C_out - c_out_start)
        c_out_rep, _ = _get_c_out_replication_params(c_out_size)
        tile_layouts = []
        for spatial_tile in spatial_tiles:
            tile_layouts.append(
                (
                    _build_low_chan_output_regions(spatial_tile, c_out_rep),
                    _build_low_chan_output_tile_groups(spatial_tile, c_out_rep),
                )
            )
        low_chan_tile_layouts.append(tile_layouts)

    return _Conv3dComputeResult(
        sbm=sbm,
        heap_depth=heap_depth,
        means_sbuf=means_sbuf,
        variances_sbuf=variances_sbuf,
        post_config=_BatchNormPostProcessConfig(
            p_max=nl.tile_size.pmax,
            batch_start=low_cfg.batch_start,
            batch_end=low_cfg.batch_end,
            c_out_start=0,
            c_out_end=cfg.C_out,
            spatial_start=0,
            spatial_end=cfg.D_out * cfg.H_out * cfg.W_out,
            reload_interleave=1,
            reload_free_tile=low_cfg.result_band_size,
            fixed_c_out_rep=None,
            low_chan_tile_layouts=low_chan_tile_layouts,
            write_momentum=low_cfg.prg_id == 0,
        ),
        post_buffers=_BatchNormPostProcessBuffers(
            gamma_buf=gamma_buf,
            beta_buf=beta_buf,
            true_gamma_buf=true_gamma_buf,
            neg_true_beta_buf=neg_true_beta_buf,
            run_running_mean_buf=run_running_mean_buf,
            run_new_mean_buf=run_new_mean_buf,
            run_running_var_buf=run_running_var_buf,
            run_new_var_buf=run_new_var_buf,
            reload_buf=reload_buf,
            residual_reload_buf=residual_reload_buf,
        ),
    )


def _conv3d_low_chan_compute(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    gamma: Optional[nl.NkiTensor],
    beta: Optional[nl.NkiTensor],
    batch_norm_eps: float,
    running_means: Optional[nl.NkiTensor],
    running_variances: Optional[nl.NkiTensor],
    residuals_in: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    conv_out: nl.NkiTensor,
    means: Optional[nl.NkiTensor],
    variances: Optional[nl.NkiTensor],
    sbm: Optional[SbufManager],
    use_auto_allocation: bool,
    transpose_taps: bool,
) -> Optional[_Conv3dComputeResult]:
    # In transpose mode, cfg.C_in is the caller's filter C_out
    low_channel_name = "C_out" if transpose_taps else "C_in"
    kernel_assert(
        cfg.C_in <= _LOW_CHAN_MAX,
        f"conv3d_low_chan requires {low_channel_name} <= {_LOW_CHAN_MAX}, got {cfg.C_in}",
    )

    if sbm != None:
        kernel_assert(
            sbm.is_auto_alloc() == use_auto_allocation,
            f"[conv3d_low_chan] If an SbufManager is provided, sbm.is_auto_alloc() "
            f"({sbm.is_auto_alloc()}) must equal use_auto_allocation ({use_auto_allocation}).",
        )
        total_sbuf_budget = nl.tile_size.total_available_sbuf_size if use_auto_allocation else sbm.get_free_space()
    else:
        total_sbuf_budget = nl.tile_size.total_available_sbuf_size

    low_cfg, mem_cfg = _build_low_chan_plan(
        cfg=cfg,
        dtype_size=sizeinbytes(x_in.dtype),
        total_sbuf_budget=total_sbuf_budget,
        transpose_taps=transpose_taps,
    )

    if sbm == None:
        sbm = SbufManager(
            0,
            mem_cfg.total_memory_required,
            logger=get_logger("conv3d_low_chan"),
            use_auto_alloc=use_auto_allocation,
        )

    heap_depth = len(sbm.heap)
    bn_stats_bufs = None
    if cfg.is_batch_norm_training():
        full_c_out_tile_count = div_ceil(cfg.C_out, nl.tile_size.pmax)
        bn_stats_bufs = sbm.alloc_heap(
            shape=(
                nl.tile_size.pmax,
                full_c_out_tile_count * low_cfg.bn_partial_iters * _BN_STATS_ELEMS,
            ),
            dtype=nl.float32,
            name="low_bn_stats",
        )
        nisa.memset(dst=bn_stats_bufs[:, :], value=0.0)

    hot_heap_depth = len(sbm.heap)
    bufs = _allocate_low_chan_buffers(
        sbm=sbm,
        cfg=cfg,
        low_cfg=low_cfg,
        mem_cfg=mem_cfg,
        dtype=x_in.dtype,
        transpose_taps=transpose_taps,
    )

    _execute_low_chan(
        x_in=x_in,
        filters=filters,
        bias=bias,
        gamma=gamma,
        beta=beta,
        running_means=running_means,
        running_variances=running_variances,
        conv_out=conv_out,
        residuals_in=residuals_in,
        cfg=cfg,
        low_cfg=low_cfg,
        mem_cfg=mem_cfg,
        batch_norm_eps=batch_norm_eps,
        bufs=bufs,
        bn_stats_bufs=bn_stats_bufs,
        transpose_taps=transpose_taps,
    )

    while len(sbm.heap) > hot_heap_depth:
        sbm.pop_heap()

    if cfg.is_batch_norm_training():
        finalize_plan = _build_low_chan_bn_finalize_plan(cfg, low_cfg)
        full_c_out_tile_count = div_ceil(cfg.C_out, nl.tile_size.pmax)
        tile_size_full = min(nl.tile_size.pmax, cfg.C_out)
        # These buffers outlive finalization and feed the rescale pass
        means_sbuf = sbm.alloc_heap(
            shape=(tile_size_full, full_c_out_tile_count), dtype=nl.float32, name="low_bn_means"
        )
        variances_sbuf = sbm.alloc_heap(
            shape=(tile_size_full, full_c_out_tile_count), dtype=nl.float32, name="low_bn_variances"
        )
        recv_triples = None
        if finalize_plan.exchange:
            recv_triples = sbm.alloc_heap(
                shape=(tile_size_full, full_c_out_tile_count * _BN_TRIPLE_ELEMS),
                dtype=nl.float32,
                name="low_bn_recv_triples",
            )
        _finalize_bn_stats(
            plan=finalize_plan,
            sbm=sbm,
            bn_stats_bufs=bn_stats_bufs,
            recv_triples=recv_triples,
            means_sbuf=means_sbuf,
            variances_sbuf=variances_sbuf,
            means=means,
            variances=variances,
            identity_mask=_identity_shuffle_mask(),
        )
        if finalize_plan.exchange:
            sbm.pop_heap()
        return _prepare_low_chan_training_result(
            cfg=cfg,
            low_cfg=low_cfg,
            sbm=sbm,
            heap_depth=heap_depth,
            conv_dtype=conv_out.dtype,
            means_sbuf=means_sbuf,
            variances_sbuf=variances_sbuf,
        )

    return None


# ---------------------------------------------------------------------------------------------------------------
# Row-banded path for thin 2D layers (conv3d routes here on trn2 via _banded_wins; conv3d_transpose shares the
# band helpers below).
# ---------------------------------------------------------------------------------------------------------------

# Blocks whose input / output tiles the SBUF budget leaves room for at once, so the compiler can load
# the next block while the current one computes.
_BLOCKS_IN_FLIGHT = 2
# Per-partition SBUF bytes left to the compiler beyond the kernel's own tiles.
_BAND_SBUF_RESERVE = 8 * 1024
# The narrowest balanced matmul chunk worth keeping before falling back to the widest one.
_MIN_BALANCED_CHUNK = 128
# Stationaries start at multiples of this many elements within a partition, so the tensor engine loads
# each one from an aligned address whatever rows_per_block * C_out is.
_STATIONARY_ALIGN = 32

# Epilogue kinds applied as results leave PSUM.
_EPILOGUE_COPY = 0
_EPILOGUE_ACT_FN = 1

# Matmul chunks (one PSUM bank each) that one eviction instruction moves out of PSUM: fewer, wider evictions pay the
# engine's fixed per-instruction cost less often.
_EVICT_GROUP_CHUNKS = 4

# Cost model shared by the routers of conv3d (_banded_cost_wins) and conv3d_transpose (_banded_transpose_wins), which
# send a layer here when they expect to win. This path paid about one DMA per image row in and per output row out:
# 0.63 to 0.69 us measured on trn2 (bf16, one core) while those DMAs had their descriptors generated on GpSimd, taken as
# _BANDED_DMA_NS. Static descriptors and the fewer-pass plan have since made the path 1.0x to 5.8x faster per layer
# (bf16, one core, NEURON_RT_IO_RING_CACHE_SIZE=64), so the constants are now conservative: the routers send fewer
# layers here than they could. The original path pays a time per output pixel and filter tap: 0.38 to 0.78 ns measured
# for 3 x 3 filters at up to 64 channels (median 0.55), taken as _BANDED_TAP_PIXEL_NS_X20 / 20 = 0.45 ns, and at least
# _BANDED_MIN_TAPS taps' worth (1 x 1 filters still cost like a few taps). The constants are rounded middles of the
# measurements, not bounds: the two fastest measured layers (32 channels, rows 1100 pixels or wider) run 6 to 15% below
# 0.45 ns. The gain a layer must reach, _BANDED_MARGIN_X5 / 5 = 1.2, covers that.
_BANDED_DMA_NS = 700
_BANDED_TAP_PIXEL_NS_X20 = 9
_BANDED_MIN_TAPS = 5
_BANDED_MARGIN_X5 = 6


@dataclass
class _BandTileConfig(nl.NKIObject):
    """
    Band tiling plan shared by the banded convolution kernels.

    The tensor engine's 128 rows hold (input row, input channel) pairs and its 128 columns hold
    (output row, output channel) pairs. This records how many of each fit, the contraction tiles one
    block of output rows needs, the column window the SBUF budget allows and the PSUM epilogue.
    With taps_folded = f > 1 each input row sits f times on the contraction rows, at the column
    shifts of f consecutive filter columns, so one matmul covers f filter columns (forward conv only;
    the transposed conv keeps 1).
    """

    rows_per_tile: int
    rows_per_block: int
    block_count: int
    block_span: int
    tiles_per_block: int
    out_partitions: int
    stationary_count: int
    taps_folded: int
    rows_reversed: bool
    chunk_cols: int
    window_cols: int
    window_count: int
    epilogue: int


def _conv3d_banded_compute(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
) -> nl.NkiTensor:
    """
    Row-banded conv3d for thin 2D layers: the D = 1, K_d = 1 case, y = act(conv2d(x_in, filters) + bias).

    conv3d calls this on trn2 when its cost model expects it to be faster (conv3d._banded_wins); it takes conv3d's
    inputs and returns its [B, C_out, 1, H_out, W_out] output. Thin layers' channel counts are too small to fill the
    tensor engine on their own. A channels-only mapping puts C_in
    on the contraction rows and C_out on the stationary columns, leaving most of the 128 x 128
    engine idle for thin layers. This kernel instead packs rows_per_tile = 128 // C_in image rows
    with their channels on the contraction rows and rows_per_block = 128 // C_out output rows with
    their channels on the columns. The filter taps become banded block matrices (output row r of a
    block reads input rows stride_h * r + dilation_h * k_h) that stay loaded while whole image rows
    stream through as the moving operand, so a layer with C channels needs about 128 // C times
    fewer matmul passes. A tap's horizontal shift is an offset into the rows already in SBUF: the
    input is not copied into an im2col layout. Only when a block's rows leave most contraction rows
    empty (few input channels, e.g. RGB stems) does the kernel fold filter columns: each input row is
    loaded up to K_w times, shifted one filter column apart, so one matmul covers several filter
    columns. It folds on one program (LNC=1) only, see _tap_fold. The banded stationaries are
    built in the kernel from the filters, one DMA per contraction row. Supports stride, asymmetric
    zero padding, an optional bias, a fused ActFnType activation and LNC sharding over (batch, row
    block) work items.

    Args:
        x_in (nl.NkiTensor): [B, C_in, 1, H, W], conv3d's input on HBM.
        filters (nl.NkiTensor): [1, K_h, K_w, C_in, C_out], conv3d's filters on HBM.
        bias (Optional[nl.NkiTensor]): conv3d's bias; its first C_out elements are read.
        cfg (Conv3dConfig): conv3d's config (D = 1, K_d = 1).

    Returns:
        y_out (nl.NkiTensor): [B, C_out, 1, H_out, W_out], Output tensor on HBM in x_in's dtype.

    Notes:
        - x_in, filters and bias share one 16-bit dtype (conv3d routes only bfloat16 / float16 here).
          The matmuls run in that dtype and accumulate in fp32 PSUM; the result is rounded once.
        - With C_in or C_out above 64 a channels-only mapping already fills most of the engine and
          the gain shrinks; at 128 channels nothing stacks.
        - The input and output move one (image row, all channels) DMA at a time, the transfer the
          NCHW layout allows for a partition range, with static descriptors (dge_mode none) built
          before the run. When x_in or y_out is also an input or output of the NEFF, the runtime
          rewrites those descriptors for the call's tensors on every execution unless it caches
          them: set NEURON_RT_IO_RING_CACHE_SIZE (e.g. 64). Inside a larger graph they cost nothing
          per call. Software-generated descriptors bounded thin layers on GpSimd at about 0.6 us
          per DMA; static ones leave the tensor engine as the limit.
        - Measured on trn2 (bf16, one NeuronCore, 3x3 stride 1) with NEURON_RT_IO_RING_CACHE_SIZE=64,
          against the compiler's convolution: 66x for 16 -> 16 at 1080p (0.51 vs 33.7 ms), 64x for
          8 -> 8, 46x for 3 -> 3 at 4K, 11x for 32 -> 32 and 5.8x for 64 -> 64 at 1080p, 2.5x for a
          16 -> 32 stride-2 layer at 320 x 320. Against conv3d (nki-library main, measured with the
          previous, software-descriptor version of this kernel) it broke even at output rows of
          about 300 pixels for 16 to 64 channels and lost on narrow rows (0.4x to 0.5x at 120
          pixels) and at 128 channels below 512 pixels; conv3d's router still uses those
          measurements.
        - Image rows outside the input and padded columns are zero in SBUF; contraction tiles that
          hold no image row are skipped.

    Pseudocode:
        plan = build_band_plan(cfg)                    # fold f, rows per tile / block, column windows
        G = ceil(K_w / f)                              # matmuls per contraction tile
        stationaries = zeros(128, tiles_per_block * G, rows_per_block * C_out)
        for each contraction row (t, j) of a block and each fold shift q < f:
            DMA the filter rows it meets, filter columns g * f + q, into
            stationaries[(j, q, :), t * G : (t + 1) * G, band]
        bias_col = replicate(bias, rows_per_block)     # partition (r, c_out) <- bias[c_out]

        for (b, block) in this program's work items:
            for window in column windows:
                load the input tiles: tile t, partition (j, q, c_in) <- x_in[b, c_in, row(t, j),
                    cols + q * dilation_w]
                for each group of up to 4 matmul chunks (<= 512 output columns each, one PSUM bank):
                    for chunk in group, t, g < G:
                        psum[bank] += stationaries[t, g].T @ tile_t[:, stride_w * col + dilation_w * f * g]
                    result[:, group] = act(psum + bias_col)   # one eviction per group
                store each output row: y_out[b, :, row, window] <- result[(r, :), :]

    Tiling Strategy:
        Loop nest: (batch, row block) work items -> column windows -> groups of matmul chunks ->
        chunks -> contraction tiles -> filter-column groups.

        SBUF layout:
        - Stationaries: [128, tiles_per_block * G, rows_per_block * C_out], loaded once.
        - Input tiles: tiles_per_block x [128, window input columns], fresh per (block, window).
        - Result: [rows_per_block * C_out, window columns] in the output dtype.

        Tile size selection:
        - Fold f (1 to K_w, filter columns per matmul): the one with the fewest passes (blocks x
          contraction tiles x G, each with its own best rows per block), a larger fold only when it
          cuts the passes by at least a quarter (it costs one input DMA per row and fold step). On
          more than one program (LNC=2) f is 1, where the row DMAs rather than the passes set the time
          (_tap_fold).
        - rows_per_tile = 128 // (C_in * f); rows_per_block <= 128 // C_out, the row count with the
          fewest passes over the output height, balanced over the row blocks and reduced only if
          the stationaries leave no room for one matmul chunk.
        - The column window spans the full output width when the tiles of two blocks fit, otherwise
          the widest multiple of the matmul chunk that does.
        - Matmul chunk: the widest divisor of the output width in [128, 512], else 512.
    """
    x_nchw = x_in.reshape((cfg.B, cfg.C_in, cfg.H, cfg.W))
    filters_hwio = filters.reshape((cfg.K_h, cfg.K_w, cfg.C_in, cfg.C_out))
    # conv3d reads the first C_out elements of the bias without checking its shape: [C_out], [C_out, 1] or longer.
    bias_vec = None
    if bias != None:
        bias_vec = bias.slice(dim=0, start=0, end=cfg.C_out, step=1).reshape((cfg.C_out,))
    y_out = nl.ndarray(shape=(cfg.B, cfg.C_out, 1, cfg.H_out, cfg.W_out), dtype=x_in.dtype, buffer=nl.shared_hbm)
    _conv2d_banded_into(y_out.reshape((cfg.B, cfg.C_out, cfg.H_out, cfg.W_out)), x_nchw, filters_hwio, bias_vec, cfg)
    return y_out


def _conv2d_banded_into(
    y_out: nl.NkiTensor,
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
) -> None:
    """Compute the convolution into y_out [B, C_out, H_out, W_out]."""
    # SPMD sharding info: n_prgs = core count, prg_id = this core (0/1 at LNC=2).
    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d", (0, 1))
    tile_cfg = _build_conv2d_band_tile_config(cfg, sizeinbytes(x_in.dtype), n_prgs)
    work_start, work_end = _program_work_range(cfg.B * tile_cfg.block_count, n_prgs, prg_id)

    if work_start < work_end:
        stationaries = _load_band_stationaries(
            filters, _conv2d_stationary_dmas(cfg, tile_cfg), tile_cfg, cfg.C_in, cfg.C_out, cfg.K_w
        )
        bias_col = _load_partition_vector(bias, cfg.C_out, tile_cfg)

        groups = div_ceil(cfg.K_w, tile_cfg.taps_folded)
        for work_idx in range(work_start, work_end):
            batch_idx = work_idx // tile_cfg.block_count
            row0 = tile_cfg.rows_per_block * (work_idx % tile_cfg.block_count)
            rows_valid = min(tile_cfg.rows_per_block, cfg.H_out - row0)
            for window_idx in range(tile_cfg.window_count):
                col0 = window_idx * tile_cfg.window_cols
                cols = min(tile_cfg.window_cols, cfg.W_out - col0)
                tiles = _load_band_input_tiles(
                    x_in,
                    batch_idx,
                    cfg.stride_h * row0 - cfg.pad_h_top,
                    cfg.stride_w * col0 - cfg.pad_w_left,
                    cfg.stride_w * (cols - 1) + cfg.dilation_w * tile_cfg.taps_folded * (groups - 1) + 1,
                    tile_cfg,
                    cfg.dilation_w,
                )
                result = _alloc_band_result(cols, x_in.dtype, tile_cfg)
                # Whole chunks in groups of up to _EVICT_GROUP_CHUNKS PSUM banks, one eviction per group; a
                # partial last chunk alone.
                full_chunks = cols // tile_cfg.chunk_cols
                group_banks = max(1, min(_EVICT_GROUP_CHUNKS, full_chunks))
                for group_start in range(0, full_chunks, group_banks):
                    banks = min(group_banks, full_chunks - group_start)
                    psum = nl.ndarray(
                        (nl.tile_size.pmax, group_banks, nl.tile_size.psum_fmax), dtype=nl.float32, buffer=nl.psum
                    )
                    for bank in range(banks):
                        chunk_start = (group_start + bank) * tile_cfg.chunk_cols
                        _conv2d_band_matmuls(
                            psum, bank, stationaries, tiles, chunk_start, tile_cfg.chunk_cols, cfg, tile_cfg
                        )
                    _evict_band_banks(
                        result,
                        psum,
                        bias_col,
                        group_start * tile_cfg.chunk_cols,
                        banks,
                        tile_cfg.chunk_cols,
                        tile_cfg,
                        cfg.activation_fn,
                    )
                tail_start = full_chunks * tile_cfg.chunk_cols
                if tail_start < cols:
                    psum = nl.ndarray((nl.tile_size.pmax, 1, nl.tile_size.psum_fmax), dtype=nl.float32, buffer=nl.psum)
                    _conv2d_band_matmuls(psum, 0, stationaries, tiles, tail_start, cols - tail_start, cfg, tile_cfg)
                    _evict_band_banks(
                        result, psum, bias_col, tail_start, 1, cols - tail_start, tile_cfg, cfg.activation_fn
                    )
                _store_band_rows(y_out, result, batch_idx, row0, rows_valid, col0, cols, tile_cfg)


def _build_conv2d_band_tile_config(cfg: Conv3dConfig, compute_bytes: int, n_prgs: int = 1) -> _BandTileConfig:
    """
    Plan the band tiling of a forward convolution on n_prgs programs (LNC=2: 2).

    A block of R output rows reads (R - 1) * stride_h + dilation_h * (K_h - 1) + 1 input rows. The
    tap fold f (_tap_fold) sets the image rows per contraction tile, 128 // (C_in * f). R is the row
    count up to 128 // C_out with the fewest tensor-engine passes over the output height
    (_fewest_pass_rows), balanced over the blocks it needs, and shrinks only if the stationaries
    leave no room for one matmul chunk.
    """
    fold = _tap_fold(cfg, n_prgs)
    groups = div_ceil(cfg.K_w, fold)
    rows_per_tile = nl.tile_size.pmax // (cfg.C_in * fold)
    rows_per_block = _balanced_rows(cfg.H_out, _fewest_pass_rows(cfg, rows_per_tile, nl.tile_size.pmax // cfg.C_out), 1)
    epilogue = _band_epilogue(cfg.activation_fn)
    chunk_cols = _pick_chunk_cols(cfg.W_out)
    budget = nl.tile_size.sbuf_fmax_bytes - _BAND_SBUF_RESERVE
    window_cols = 0
    span = 0
    tiles_per_block = 0
    while window_cols == 0 and rows_per_block >= 1:
        span = (rows_per_block - 1) * cfg.stride_h + cfg.dilation_h * (cfg.K_h - 1) + 1
        tiles_per_block = div_ceil(span, rows_per_tile)
        fixed = _band_fixed_bytes(tiles_per_block * groups, rows_per_block * cfg.C_out, compute_bytes)
        window_cols = _fit_window_cols(
            cfg.W_out,
            chunk_cols,
            budget - fixed,
            tiles_per_block * compute_bytes * cfg.stride_w + compute_bytes,
            tiles_per_block * compute_bytes * (cfg.dilation_w * fold * (groups - 1) + 1 - cfg.stride_w),
        )
        if window_cols == 0:
            rows_per_block = rows_per_block - 1
    kernel_assert(
        window_cols > 0,
        f"conv3d (banded): the banded stationaries of K_h={cfg.K_h}, K_w={cfg.K_w}, C_in={cfg.C_in}, "
        f"C_out={cfg.C_out} do not fit the SBUF budget even with one output row per block.",
    )
    return _BandTileConfig(
        rows_per_tile=rows_per_tile,
        rows_per_block=rows_per_block,
        block_count=div_ceil(cfg.H_out, rows_per_block),
        block_span=span,
        tiles_per_block=tiles_per_block,
        out_partitions=rows_per_block * cfg.C_out,
        stationary_count=tiles_per_block * groups,
        taps_folded=fold,
        rows_reversed=True,
        chunk_cols=min(chunk_cols, window_cols),
        window_cols=window_cols,
        window_count=div_ceil(cfg.W_out, window_cols),
        epilogue=epilogue,
    )


def _tap_fold(cfg: Conv3dConfig, n_prgs: int = 1) -> int:
    """
    Filter columns folded onto the contraction rows, 1 (none) to K_w.

    With fold f a contraction tile holds 128 // (C_in * f) image rows, each f times at the column
    shifts of f consecutive filter columns, so one matmul covers f filter columns and a block needs
    ceil(K_w / f) matmuls per tile instead of K_w. That pays when a block's rows leave most
    contraction rows empty (few input channels, many output channels: RGB stems). Each fold step
    costs one more input DMA per image row, so a larger fold must cut the passes (blocks x tiles x
    filter-column groups, each fold with its own fewest-pass rows) by at least a quarter.

    On more than one program (LNC=2) the fold is always 1. There the compiler issues each of this
    path's static row DMAs on its own (each started by a Sync-engine write) instead of merging them,
    so the DMA count, not the tensor engine, sets the time, and the fold multiplies it. Measured on
    trn2 at LNC=2 (bf16, B=1): a 3 -> 16 3 x 3 stride-2 stem at 736 x 960 took 341 us with fold 3 and
    153 us with fold 1, a 3 -> 64 7 x 7 stride-2 stem at 544 x 960 608 us with fold 4 and 220 us
    with fold 1. On one program the fold still pays: 172 us with fold 4 against 409 us with fold 1.
    """
    if n_prgs > 1:
        return 1
    best_fold = 1
    best_passes = 0
    for fold in range(1, cfg.K_w + 1):
        if cfg.C_in * fold <= nl.tile_size.pmax:
            rows_per_tile = nl.tile_size.pmax // (cfg.C_in * fold)
            rows = _fewest_pass_rows(cfg, rows_per_tile, nl.tile_size.pmax // cfg.C_out)
            span = (rows - 1) * cfg.stride_h + cfg.dilation_h * (cfg.K_h - 1) + 1
            passes = div_ceil(cfg.H_out, rows) * div_ceil(span, rows_per_tile) * div_ceil(cfg.K_w, fold)
            if best_passes == 0 or 4 * passes <= 3 * best_passes:
                best_fold = fold
                best_passes = passes
    return best_fold


def _fewest_pass_rows(cfg: Conv3dConfig, rows_per_tile: int, rows_max: int) -> int:
    """
    Output rows per block, at most rows_max, with the fewest tensor-engine passes over the output height.

    Passes = blocks x contraction tiles per block (each tile is a fixed number of matmuls). A block
    whose input span just overflows a tile costs a whole extra tile of passes, so a slightly shorter
    block that needs one tile fewer is often cheaper overall. Ties keep more rows (fewer blocks).
    """
    best_rows = rows_max
    best_passes = 0
    for row_idx in range(rows_max):
        rows = rows_max - row_idx
        span = (rows - 1) * cfg.stride_h + cfg.dilation_h * (cfg.K_h - 1) + 1
        passes = div_ceil(cfg.H_out, rows) * div_ceil(span, rows_per_tile)
        if best_passes == 0 or passes < best_passes:
            best_rows = rows
            best_passes = passes
    return best_rows


def _conv2d_stationary_dmas(cfg: Conv3dConfig, tile_cfg: _BandTileConfig) -> list:
    """
    Filter-row DMAs that build the forward conv's banded stationaries.

    Contraction row (j, q) of tile t holds block input row s = t * rows_per_tile + j at fold shift
    q < taps_folded. Output row r of the block reads it through filter row k_h = (s - stride_h * r) /
    dilation_h when that is an integer in [0, K_h). Output rows sit in reversed slots
    (slot = rows_per_block - 1 - r) so the filter row grows with the slot and each (t, j, q) is one
    DMA with positive strides.
    """
    dmas = []
    for tile_idx in range(tile_cfg.tiles_per_block):
        for row_idx in range(tile_cfg.rows_per_tile):
            in_row = tile_idx * tile_cfg.rows_per_tile + row_idx
            if in_row < tile_cfg.block_span:
                slots = []
                k_rows = []
                for slot in range(tile_cfg.rows_per_block):
                    offset = in_row - cfg.stride_h * (tile_cfg.rows_per_block - 1 - slot)
                    if offset >= 0 and offset % cfg.dilation_h == 0 and offset // cfg.dilation_h < cfg.K_h:
                        slots.append(slot)
                        k_rows.append(offset // cfg.dilation_h)
                for shift in range(tile_cfg.taps_folded):
                    _append_band_dma(dmas, tile_idx, row_idx, slots, k_rows, shift)
    return dmas


def _conv2d_band_matmuls(
    psum: nl.NkiTensor,
    bank: int,
    stationaries: nl.NkiTensor,
    tiles: list,
    chunk_start: int,
    chunk: int,
    cfg: Conv3dConfig,
    tile_cfg: _BandTileConfig,
) -> None:
    """
    Accumulate one chunk of output columns into PSUM bank `bank` of a [128, banks, psum_fmax] PSUM
    tile, over the block's tiles and their ceil(K_w / taps_folded) filter-column groups.
    """
    parts = tile_cfg.out_partitions
    fold = tile_cfg.taps_folded
    groups = div_ceil(cfg.K_w, fold)
    for list_idx in range(len(tiles)):
        tile = tiles[list_idx][0]
        tile_rows = tiles[list_idx][1]
        tile_idx = tiles[list_idx][2]
        for group_idx in range(groups):
            col_start = cfg.stride_w * chunk_start + cfg.dilation_w * fold * group_idx
            if cfg.stride_w == 1:
                moving = tile[0:tile_rows, col_start : col_start + chunk]
            else:
                moving = tile[0:tile_rows, col_start : col_start + cfg.stride_w * (chunk - 1) + 1 : cfg.stride_w]
            nisa.nc_matmul(
                dst=psum[0:parts, bank, 0:chunk],
                stationary=stationaries[0:tile_rows, tile_idx * groups + group_idx, 0:parts],
                moving=moving,
                accumulate=(list_idx > 0 or group_idx > 0),
            )


def _balanced_rows(out_rows: int, rows_max: int, row_multiple: int) -> int:
    """Rows per block: the fewest blocks rows_max allows, then the smallest multiple of row_multiple that covers them."""
    block_count = div_ceil(out_rows, rows_max)
    return min(rows_max, div_ceil(div_ceil(out_rows, block_count), row_multiple) * row_multiple)


def _append_band_dma(dmas: list, tile_idx: int, row_idx: int, slots: list, k_rows: list, shift: int = 0) -> None:
    """
    Append (tile, row, fold shift, first slot, slot step, first filter row, filter row step, count)
    for one band. shift is the contraction row's fold shift (always 0 without a tap fold).
    """
    count = len(slots)
    if count > 0:
        slot_step = 1
        k_row_step = 1
        if count > 1:
            slot_step = slots[1] - slots[0]
            k_row_step = k_rows[1] - k_rows[0]
        for idx in range(count):
            kernel_assert(
                slots[idx] == slots[0] + idx * slot_step and k_rows[idx] == k_rows[0] + idx * k_row_step,
                "banded stationary rows must form an arithmetic progression",
            )
        kernel_assert(slot_step >= 1 and k_row_step >= 1, "banded stationary DMA strides must be positive")
        dmas.append((tile_idx, row_idx, shift, slots[0], slot_step, k_rows[0], k_row_step, count))


def _band_epilogue(activation_fn: Optional[ActFnType]) -> int:
    """Epilogue kind applied as results leave PSUM."""
    if activation_fn == None:
        return _EPILOGUE_COPY
    return _EPILOGUE_ACT_FN


def _pick_chunk_cols(cols: int) -> int:
    """Widest matmul chunk within the PSUM / moving free limits that divides cols, else the limit."""
    chunk_max = min(nl.tile_size.psum_fmax, nl.tile_size.gemm_moving_fmax)
    for chunk in range(min(cols, chunk_max), _MIN_BALANCED_CHUNK - 1, -1):
        if cols % chunk == 0:
            return chunk
    return min(cols, chunk_max)


def _band_fixed_bytes(stationary_count: int, out_partitions: int, compute_bytes: int) -> int:
    """Per-partition SBUF bytes held for the whole kernel: stationaries plus the partition vectors."""
    vectors = 2 * (out_partitions * (4 + compute_bytes) + 4) + 8
    return stationary_count * _stationary_stride(out_partitions) * compute_bytes + vectors


def _stationary_stride(out_partitions: int) -> int:
    """Elements between consecutive stationaries in a partition: out_partitions rounded up to _STATIONARY_ALIGN."""
    return div_ceil(out_partitions, _STATIONARY_ALIGN) * _STATIONARY_ALIGN


def _fit_window_cols(out_cols: int, chunk_cols: int, budget: int, bytes_per_col: int, bytes_fixed: int) -> int:
    """
    Widest column window whose per-block tiles fit the budget for _BLOCKS_IN_FLIGHT blocks.

    Returns out_cols when the whole width fits, else the widest multiple of chunk_cols that does,
    else 0.
    """
    window_cols = 0
    if budget > 0:
        if _BLOCKS_IN_FLIGHT * (bytes_fixed + bytes_per_col * out_cols) <= budget:
            window_cols = out_cols
        else:
            cols = (budget // _BLOCKS_IN_FLIGHT - bytes_fixed) // bytes_per_col
            if cols >= chunk_cols:
                window_cols = (cols // chunk_cols) * chunk_cols
    return window_cols


def _program_work_range(total: int, n_prgs: int, prg_id: int) -> tuple[int, int]:
    """Contiguous [start, end) slice of the (batch, row block) work items owned by this program."""
    per_program = div_ceil(total, n_prgs)
    start = min(total, per_program * prg_id)
    return start, min(total, start + per_program)


def _load_band_stationaries(
    filters: nl.NkiTensor, dmas: list, tile_cfg: _BandTileConfig, C_in: int, C_out: int, K_w: int
) -> nl.NkiTensor:
    """
    Build the banded stationaries [128, tiles_per_block * G, rows_per_block * C_out] in SBUF, with
    G = ceil(K_w / taps_folded) filter-column groups per tile (G = K_w without a tap fold).

    Zeroed once, then each DMA writes filter rows k_row0 + q * k_row_step, filter columns
    g * taps_folded + shift of every group g, into slots slot0 + q * slot_step of partitions
    [(row * taps_folded + shift) * C_in, + C_in) of tile t.
    """
    pmax = nl.tile_size.pmax
    parts = tile_cfg.out_partitions
    stride = _stationary_stride(parts)
    fold = tile_cfg.taps_folded
    groups = div_ceil(K_w, fold)
    stationaries = nl.ndarray((pmax, tile_cfg.stationary_count, stride), dtype=filters.dtype, buffer=nl.sbuf)
    nisa.memset(dst=stationaries[0:pmax, 0 : tile_cfg.stationary_count, 0:stride], value=0.0)
    for dma_idx in range(len(dmas)):
        tile_idx = dmas[dma_idx][0]
        row_idx = dmas[dma_idx][1]
        shift = dmas[dma_idx][2]
        slot0 = dmas[dma_idx][3]
        slot_step = dmas[dma_idx][4]
        k_row0 = dmas[dma_idx][5]
        k_row_step = dmas[dma_idx][6]
        count = dmas[dma_idx][7]
        shift_groups = div_ceil(K_w - shift, fold)
        part0 = (row_idx * fold + shift) * C_in
        dst = (
            stationaries[part0 : part0 + C_in, tile_idx * groups : tile_idx * groups + shift_groups, 0:parts]
            .reshape_dim(2, (tile_cfg.rows_per_block, C_out))
            .slice(dim=2, start=slot0, end=slot0 + (count - 1) * slot_step + 1, step=slot_step)
        )
        src = filters.ap(
            pattern=[
                [C_out, C_in],
                [fold * C_in * C_out, shift_groups],
                [k_row_step * K_w * C_in * C_out, count],
                [1, C_out],
            ],
            offset=k_row0 * K_w * C_in * C_out + shift * C_in * C_out,
        )
        _issue_setup_dma(dst, src, dma_idx)
    return stationaries


def _issue_setup_dma(dst, src, dma_idx: int) -> None:
    """
    Issue one of the many small stationary-building DMAs, rotating over three descriptor generators.

    Each carries C_out-element runs (a few bytes for thin outputs), so the descriptor rate rather than
    the bandwidth bounds the setup: the sync-engine DGE, the scalar-engine DGE and the GpSimd SWDGE
    process their share in parallel.
    """
    queue = dma_idx % 3
    if queue == 0:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge)
    elif queue == 1:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.scalar_engine)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge)


def _load_partition_vector(vec: Optional[nl.NkiTensor], C_out: int, tile_cfg: _BandTileConfig) -> nl.NkiTensor:
    """
    Load a [C_out] vector as an fp32 [128, 1] column with partition (slot, c_out) <- vec[c_out].

    One stride-0 DMA lays rows_per_block copies along partition 0, and a K = 1 matmul against a one
    turns that row into the column. A missing vector becomes zeros.
    """
    pmax = nl.tile_size.pmax
    parts = tile_cfg.out_partitions
    column = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    if vec == None:
        nisa.memset(dst=column[0:pmax, 0:1], value=0.0)
    else:
        row = nl.ndarray((1, parts), dtype=vec.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=row[0:1, 0:parts].reshape_dim(1, (tile_cfg.rows_per_block, C_out)),
            src=vec.ap(pattern=[[C_out, 1], [0, tile_cfg.rows_per_block], [1, C_out]], offset=0),
        )
        row_f32 = nl.ndarray((1, parts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row_f32[0:1, 0:parts], src=row[0:1, 0:parts])
        one = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=one[0:1, 0:1], value=1.0)
        column_psum = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=column_psum[0:parts, 0:1], stationary=row_f32[0:1, 0:parts], moving=one[0:1, 0:1])
        nisa.tensor_copy(dst=column[0:parts, 0:1], src=column_psum[0:parts, 0:1])
    return column


def _load_band_input_tiles(
    x_in: nl.NkiTensor,
    batch_idx: int,
    in_row0: int,
    in_col0: int,
    tile_cols: int,
    tile_cfg: _BandTileConfig,
    col_shift: int = 1,
) -> list:
    """
    Load one block's contraction tiles for one column window.

    Tile t, partition (j, q, c_in) holds x_in[batch_idx, c_in, in_row0 + t * rows_per_tile + j, :]
    at tile column c <- input column in_col0 + c + q * col_shift, for the fold shifts
    q < taps_folded (q = 0 only without a tap fold). Rows outside [0, H) and columns outside [0, W)
    are zero; only the rows the stationaries use are loaded. Tiles that hold no image row are
    skipped (their products are zero); if none is left, one zero tile keeps the PSUM
    initialization. Each row moves with static DMA descriptors. Returns
    [(tile, contraction rows, tile index)].
    """
    _, C_in, H, W = x_in.shape
    pmax = nl.tile_size.pmax
    fold = tile_cfg.taps_folded
    # Tile columns that hold image columns for every fold shift: [lo_max, hi_min).
    lo_max = 0
    hi_min = tile_cols
    any_cols = False
    for shift in range(fold):
        shifted = in_col0 + shift * col_shift
        lo_max = max(lo_max, max(0, -shifted))
        hi_min = min(hi_min, min(tile_cols, W - shifted))
        if max(0, -shifted) < min(tile_cols, W - shifted):
            any_cols = True
    tiles = []
    for tile_idx in range(tile_cfg.tiles_per_block):
        tile_row0 = tile_idx * tile_cfg.rows_per_tile
        rows = min(tile_cfg.rows_per_tile, tile_cfg.block_span - tile_row0)
        tile_parts = rows * fold * C_in
        first_row = in_row0 + tile_row0
        valid_lo = min(rows, max(0, -first_row))
        valid_hi = max(valid_lo, min(rows, H - first_row))
        if any_cols and valid_lo < valid_hi:
            tile = nl.ndarray((pmax, tile_cols), dtype=x_in.dtype, buffer=nl.sbuf)
            if valid_lo > 0 or valid_hi < rows or lo_max >= hi_min:
                # A partly padded tile: zero it whole (an instruction starting off partition 0 may span
                # only one quadrant), then the image rows land on top.
                nisa.memset(dst=tile[0:tile_parts, 0:tile_cols], value=0.0)
            else:
                if lo_max > 0:
                    nisa.memset(dst=tile[0:tile_parts, 0:lo_max], value=0.0)
                if hi_min < tile_cols:
                    nisa.memset(dst=tile[0:tile_parts, hi_min:tile_cols], value=0.0)
            for row_idx in range(valid_lo, valid_hi):
                for shift in range(fold):
                    shifted = in_col0 + shift * col_shift
                    col_lo = max(0, -shifted)
                    col_hi = min(tile_cols, W - shifted)
                    if col_lo < col_hi:
                        part0 = (row_idx * fold + shift) * C_in
                        nisa.dma_copy(
                            dst=tile[part0 : part0 + C_in, col_lo:col_hi],
                            src=x_in.ap(
                                pattern=[[H * W, C_in], [1, col_hi - col_lo]],
                                offset=(batch_idx * C_in * H + first_row + row_idx) * W + shifted + col_lo,
                            ),
                            dge_mode=nisa.dge_mode.none,
                        )
            tiles.append((tile, tile_parts, tile_idx))
    if len(tiles) == 0:
        tile = nl.ndarray((pmax, tile_cols), dtype=x_in.dtype, buffer=nl.sbuf)
        tile_parts = min(tile_cfg.rows_per_tile, tile_cfg.block_span) * fold * C_in
        nisa.memset(dst=tile[0:tile_parts, 0:tile_cols], value=0.0)
        tiles.append((tile, tile_parts, 0))
    return tiles


def _alloc_band_result(cols: int, out_dtype, tile_cfg: _BandTileConfig) -> nl.NkiTensor:
    """The stored tile of one column window, in the output dtype."""
    return nl.ndarray((nl.tile_size.pmax, cols), dtype=out_dtype, buffer=nl.sbuf)


def _evict_band_psum(
    result: nl.NkiTensor,
    psum: nl.NkiTensor,
    bias_col: nl.NkiTensor,
    col_start: int,
    count: int,
    col_step: int,
    tile_cfg: _BandTileConfig,
    activation_fn: Optional[ActFnType],
) -> None:
    """
    PSUM -> SBUF with the bias and the activation applied, into columns col_start, col_start + col_step, ... of
    the window. The stored tile is written directly, so the result is rounded once.
    """
    parts = tile_cfg.out_partitions
    dst_tile = result
    op = nl.copy
    if tile_cfg.epilogue == _EPILOGUE_ACT_FN:
        op = get_nl_act_fn_from_type(activation_fn)
    if col_step == 1:
        dst = dst_tile[0:parts, col_start : col_start + count]
    else:
        dst = dst_tile[0:parts, col_start : col_start + col_step * (count - 1) + 1 : col_step]
    nisa.activation(dst=dst, op=op, data=psum[0:parts, 0:count], bias=bias_col[0:parts, 0:1], scale=1.0)


def _evict_band_banks(
    result: nl.NkiTensor,
    psum: nl.NkiTensor,
    bias_col: nl.NkiTensor,
    col_start: int,
    banks: int,
    chunk: int,
    tile_cfg: _BandTileConfig,
    activation_fn: Optional[ActFnType],
) -> None:
    """
    _evict_band_psum for the forward conv's multi-bank PSUM tile: banks 0..banks-1 of a
    [128, banks, psum_fmax] tile (chunk columns each) -> window columns [col_start, col_start +
    banks * chunk), in one instruction.
    """
    parts = tile_cfg.out_partitions
    dst_tile = result
    op = nl.copy
    if tile_cfg.epilogue == _EPILOGUE_ACT_FN:
        op = get_nl_act_fn_from_type(activation_fn)
    nisa.activation(
        dst=dst_tile[0:parts, col_start : col_start + banks * chunk].reshape_dim(1, (banks, chunk)),
        op=op,
        data=psum[0:parts, 0:banks, 0:chunk],
        bias=bias_col[0:parts, 0:1],
        scale=1.0,
    )


def _store_band_rows(
    y_out: nl.NkiTensor,
    result: nl.NkiTensor,
    batch_idx: int,
    row0: int,
    rows_valid: int,
    col0: int,
    cols: int,
    tile_cfg: _BandTileConfig,
) -> None:
    """
    Store partitions (slot, c_out) as y_out[batch_idx, c_out, row0 + r, col0 : col0 + cols], one DMA
    per row with static descriptors.
    """
    _, C_out, H_out, W_out = y_out.shape
    for out_row in range(rows_valid):
        slot = out_row
        if tile_cfg.rows_reversed:
            slot = tile_cfg.rows_per_block - 1 - out_row
        nisa.dma_copy(
            dst=y_out.ap(
                pattern=[[H_out * W_out, C_out], [1, cols]],
                offset=(batch_idx * C_out * H_out + row0 + out_row) * W_out + col0,
            ),
            src=result[slot * C_out : (slot + 1) * C_out, 0:cols],
            dge_mode=nisa.dge_mode.none,
        )


# Row-banded path routing (_banded_wins). The cost constants _BANDED_* above are shared with conv3d_transpose; these
# are specific to conv3d.
_BANDED_MIN_WORK_NS = 100_000  # below this both paths sit at the launch floor (one core)
_BANDED_MAX_CHANNELS = 64
_BANDED_MAX_STRIDE = 2


def _banded_cost_wins(cfg: "Conv3dConfig") -> bool:
    """
    Cost model of the row-banded path: True when it is expected to beat the K-replication path.
    Per output row the banded path issues (span + rows) / rows DMAs, where rows = 128 // C_out output rows
    share the span input rows they read; this path costs a per-pixel, per-tap time times the row width. The cost
    constants (_BANDED_*) are rounded middles of the measurements, and the gain must reach
    _BANDED_MARGIN_X5 / 5 so a layer near the break-even stays on this path. Fitted against the K-replication path: on
    one core a routed layer was at least 1.6x faster on every one of 66 measured trn2 layers, and at least 1.2x on two
    cores (the weakest of 18 measured). Re-measured against the current conv3d (low-channel path included) together
    with conv3d_transpose's router: every routed layer of 41 measured trn2 layers (bf16, one core) was faster, 1.3x to
    28x, and none was slower.
    """
    rows_per_block = nl.tile_size.pmax // cfg.C_out
    span = (rows_per_block - 1) * cfg.stride_h + cfg.K_h
    banded_row_ns = _BANDED_DMA_NS * (span + rows_per_block) // rows_per_block
    taps = max(cfg.K_h * cfg.K_w, _BANDED_MIN_TAPS)
    conv3d_row_ns = _BANDED_TAP_PIXEL_NS_X20 * taps * cfg.W_out // 20
    return (
        5 * conv3d_row_ns >= _BANDED_MARGIN_X5 * banded_row_ns
        and cfg.B * cfg.H_out * conv3d_row_ns >= _BANDED_MIN_WORK_NS
    )


def _banded_wins(cfg: "Conv3dConfig", dtype, has_caller_sbm: bool) -> bool:
    """
    True when this call is routed to _conv3d_banded_compute: a 2D call (D = 1, K_d = 1, no depth padding, so D_out = 1) that
    path supports (no fused batchnorm or residual, no caller SbufManager, the activations this kernel supports,
    16-bit operands, unit dilation and strides up to 2, filters up to 8 x 8, at most 64 channels) where
    _banded_cost_wins expects it to be faster. conv3d additionally requires trn2, the only target the cost model was
    measured on.
    """
    structural = (
        not has_caller_sbm
        and not cfg.has_batch_norm()
        and not cfg.has_residual_add()
        and cfg.activation_fn != ActFnType.SquaredReLU
        and (dtype == nl.bfloat16 or dtype == nl.float16)
        and cfg.D == 1
        and cfg.K_d == 1
        and cfg.pad_d_left == 0
        and cfg.pad_d_right == 0
        and cfg.dilation_h == 1
        and cfg.dilation_w == 1
        and cfg.stride_h <= _BANDED_MAX_STRIDE
        and cfg.stride_w <= _BANDED_MAX_STRIDE
        and cfg.K_h <= 8
        and cfg.K_w <= 8
        and cfg.B <= 16
        and cfg.H <= 4096
        and cfg.W <= 4096
        and cfg.C_in <= _BANDED_MAX_CHANNELS
        and cfg.C_out <= _BANDED_MAX_CHANNELS
    )
    return structural and _banded_cost_wins(cfg)


# ---------------------------------------------------------------------------------------------------------------
# Block-diagonal banded path for grouped and depthwise 2D layers (conv3d runs every call with groups > 1 here, on
# trn2 only; depthwise_conv1d_implicit_gemm routes its calls here on trn2 via _grouped_1d_wins).
# ---------------------------------------------------------------------------------------------------------------

# Filter layouts the grouped path reads: conv3d's filters viewed as [K_h, K_w, C_in / groups, C_out], or the
# [C, 1, 1, S] one-row depthwise filters of depthwise_conv1d_implicit_gemm (only the one-row diagonal path reads those).
_FILTERS_HWIO = 0
_FILTERS_C11S = 1
# depthwise_conv1d_implicit_gemm routing (_grouped_1d_wins), from trn2 measurements of both kernels.
_GROUPED_1D_MAX_TAPS = 512  # the longest filter measured
_GROUPED_1D_MIN_CHANNELS = 128  # below, both kernels run at the launch floor
_GROUPED_1D_MIN_OUTPUTS = 16


@dataclass
class _GroupedBandPlan(nl.NKIObject):
    """
    Packing of one grouped convolution onto the tensor engine (see _conv3d_grouped_compute).

    A pass is one matmul per filter column over one block of rows_out output rows of g_pack groups. Its contraction
    rows hold in_parts = g_pack * C_in_g * span (input channel, input row) pairs and its stationary columns
    out_parts = g_pack * C_out_g * rows_out (output channel, output row) pairs; n_img images sit side by side on the
    free axis.
    """

    g_pack: int  # groups side by side on the partitions of one pass
    rows_out: int  # output rows of one block, per output channel
    span: int  # input rows of one block, per input channel
    in_parts: int  # g_pack * C_in_g * span contraction rows
    out_parts: int  # g_pack * C_out_g * rows_out stationary columns
    n_packs: int  # passes over the groups, ceil(groups / g_pack)
    n_blocks: int  # row blocks over the output height
    n_img: int  # images side by side on the free axis of one pass
    n_img_groups: int  # passes over the batch, ceil(B / n_img)
    seg_in: int  # input columns per image segment (padded width, a multiple of stride_w)
    seg_out: int  # output columns per image segment, seg_in // stride_w
    tile_cols: int  # input tile columns: the segments plus the reach of the last filter column


def _plan_grouped_band(
    B: int,
    C_in_g: int,
    C_out_g: int,
    groups: int,
    H_out: int,
    W: int,
    K_h: int,
    stride_h: int,
    stride_w: int,
    pad_w: int,
    dilation_h: int,
    reach_w: int,
) -> _GroupedBandPlan:
    """
    Choose groups per pass and output rows per block with the fewest passes, then images per pass.

    A block of R output rows reads span(R) = (R - 1) * stride_h + dilation_h * (K_h - 1) + 1 input rows of every
    input channel, so the contraction rows hold g_pack * C_in_g * span(R) <= 128 and the stationary columns
    g_pack * C_out_g * R <= 128. Passes = ceil(groups / g_pack) * ceil(H_out / R); ties keep fewer groups per pass
    (fewer setup DMAs per pass). Rows are then balanced over the blocks. Images go side by side on the free axis
    until one PSUM bank (512 output columns) is full, balanced over the image groups. pad_w is the total horizontal
    padding and reach_w = dilation_w * (K_w - 1) the columns the last filter column reaches past a segment.
    """
    pmax = nl.tile_size.pmax
    psum_cols = nl.tile_size.psum_fmax
    best_pack = 0
    best_rows = 0
    best_passes = 0
    for g_pack in range(1, groups + 1):
        if g_pack * C_in_g <= pmax and g_pack * C_out_g <= pmax:
            rows = min(H_out, pmax // (g_pack * C_out_g))
            while rows >= 1 and g_pack * C_in_g * ((rows - 1) * stride_h + dilation_h * (K_h - 1) + 1) > pmax:
                rows = rows - 1
            if rows >= 1:
                passes = div_ceil(groups, g_pack) * div_ceil(H_out, rows)
                if best_passes == 0 or passes < best_passes:
                    best_pack = g_pack
                    best_rows = rows
                    best_passes = passes
    kernel_assert(best_pack > 0, "conv3d grouped path: no block of output rows fits the 128 partitions")
    n_blocks = div_ceil(H_out, best_rows)
    rows_out = div_ceil(H_out, n_blocks)
    span = (rows_out - 1) * stride_h + dilation_h * (K_h - 1) + 1
    seg_in = div_ceil(W + pad_w, stride_w) * stride_w
    seg_out = seg_in // stride_w
    n_img = 1
    if seg_out <= psum_cols:
        n_img = max(1, min(B, psum_cols // seg_out))
    n_img_groups = div_ceil(B, n_img)
    n_img = div_ceil(B, n_img_groups)
    return _GroupedBandPlan(
        g_pack=best_pack,
        rows_out=rows_out,
        span=span,
        in_parts=best_pack * C_in_g * span,
        out_parts=best_pack * C_out_g * rows_out,
        n_packs=div_ceil(groups, best_pack),
        n_blocks=n_blocks,
        n_img=n_img,
        n_img_groups=n_img_groups,
        seg_in=seg_in,
        seg_out=seg_out,
        tile_cols=n_img * seg_in + reach_w,
    )


def _conv3d_grouped_compute(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    groups: int,
) -> nl.NkiTensor:
    """
    Grouped and depthwise conv3d for 2D layers: the groups > 1, D = 1, K_d = 1 case,
    y = act(conv2d(x_in, filters, groups=groups) + bias), with block-diagonal banded filter matrices.

    conv3d runs every call with groups > 1 here (its K-replication strategy contracts over all of C_in and has no
    grouped mapping); _validate_grouped_inputs checks the call first. It takes conv3d's inputs and returns its
    [B, C_out, 1, H_out, W_out] output. A depthwise or grouped layer gives a channels-only mapping almost nothing to
    contract over (one input channel per output channel for depthwise). Here every input channel contributes a block
    of image rows to the contraction rows and every output channel a block of output rows to the stationary columns;
    for each filter column the stationary holds, for every (input, output) channel pair of a group, the banded
    Toeplitz matrix of that filter column (output row r reads input rows stride_h * r + dilation_h * k_h), and zeros
    between groups. Several groups share one pass (g_pack), and several images sit side by side on the free axis, so
    a pass streams up to 512 output columns. The Toeplitz blocks are built on chip: a matmul by a 0/1
    block-expansion matrix gives every contraction row the zero-padded filter columns of its input channel, and one
    GpSimd local gather (nc_n_gather) with a fixed index pattern places them on the band, so the setup of a pass
    costs three instructions whatever the filter size. A filter column's horizontal shift is an offset into the rows
    already in SBUF. Depthwise layers with small planes and many channels run instead with one whole plane per
    partition and diagonal stationaries when the cost model (_grouped_planes_cheaper) expects that to be faster, and
    one-row depthwise layers (H = 1, K_h = 1) use diagonal stationaries directly.

    Args:
        x_in (nl.NkiTensor): [B, C_in, 1, H, W], conv3d's input on HBM.
        filters (nl.NkiTensor): [1, K_h, K_w, C_in / groups, C_out], conv3d's filters on HBM: the
            torch.nn.Conv3d(groups=groups) weight [C_out, C_in / groups, 1, K_h, K_w] permuted with (2, 3, 4, 1, 0).
            Output channel o belongs to group o // (C_out / groups) and reads its input channels
            (o // (C_out / groups)) * (C_in / groups) + i.
        bias (Optional[nl.NkiTensor]): conv3d's bias; its first C_out elements are read.
        cfg (Conv3dConfig): conv3d's config (D = 1, K_d = 1).
        groups (int): Number of channel groups, dividing C_in and C_out.

    Returns:
        y_out (nl.NkiTensor): [B, C_out, 1, H_out, W_out], Output tensor on HBM in x_in's dtype.

    Notes:
        - Supported on trn2 (NeuronCore-v3) only.
        - Intended usage: B 1-64; C_in / groups and C_out / groups at most 128, best below 16 (depthwise: 1);
          H, W 1-4096; K_h, K_w 1-31 (the 31 x 31 depthwise filters of large-kernel networks included); one-row
          depthwise inputs (H = 1, K_h = 1) take K_w up to 512; stride 1-4 and dilation 1-8 per dimension.
        - x_in, filters and bias share one dtype: float32, bfloat16 or float16. The matmuls run in that dtype and
          accumulate in fp32 PSUM; the result is rounded once.
        - LNC=2: the (pack, image group, row block) work items are split into two contiguous ranges.

    Pseudocode:
        plan = _plan_grouped_band(...)                   # g_pack, rows per block, images per pass
        band_idx = fixed gather pattern                   # (channel, row) x (k_w, out channel, out row)
        for pack of g_pack groups:
            V[i, (o, k_w, :)] = zero-padded filter columns of input channel i     # memset + one DMA per group
            rows = expand^T @ V                           # TensorE: every contraction row gets its channel's V
            stat = gather(rows, band_idx)                 # GpSimd: the block-diagonal Toeplitz blocks
            for image group, row block:
                tile[(c, j), image * seg + col] = x_in[image, c, row(j), col - pad_w_left]   (zero outside)
                for chunk of <= 512 output columns:
                    psum = sum over k_w of stat[k_w]^T @ tile[:, chunk * stride_w + k_w * dilation_w]
                    result[:, chunk] = act(psum + bias)
                y_out[image, o, rows, :] = result[(o, r), image * seg_out : + W_out]
    """
    C_in_g = filters.shape[3]
    x_nchw = x_in.reshape((cfg.B, cfg.C_in, cfg.H, cfg.W))
    filters_hwio = filters.reshape((cfg.K_h, cfg.K_w, C_in_g, cfg.C_out))
    # conv3d reads the first C_out elements of the bias without checking its shape: [C_out], [C_out, 1] or longer.
    bias_vec = None
    if bias != None:
        bias_vec = bias.slice(dim=0, start=0, end=cfg.C_out, step=1).reshape((cfg.C_out,))
    y_out = nl.ndarray(shape=(cfg.B, cfg.C_out, 1, cfg.H_out, cfg.W_out), dtype=x_in.dtype, buffer=nl.shared_hbm)
    _conv2d_grouped_banded_into(
        y_out=y_out.reshape((cfg.B, cfg.C_out, cfg.H_out, cfg.W_out)),
        x_in=x_nchw,
        filters=filters_hwio,
        bias=bias_vec,
        stride=(cfg.stride_h, cfg.stride_w),
        padding=(cfg.pad_h_top, cfg.pad_h_bottom, cfg.pad_w_left, cfg.pad_w_right),
        dilation=(cfg.dilation_h, cfg.dilation_w),
        groups=groups,
        activation_fn=cfg.activation_fn,
        filter_layout=_FILTERS_HWIO,
    )
    return y_out


def _validate_grouped_inputs(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    groups: int,
    has_caller_sbm: bool,
) -> None:
    """Validate a conv3d call with groups > 1 for _conv3d_grouped_compute (conv3d's own checks assume groups = 1)."""
    kernel_assert(
        nisa.get_nc_version() == nisa.nc_version.gen3,
        "conv3d with groups > 1 supports trn2 (NeuronCore-v3) only",
    )
    kernel_assert(
        not cfg.has_batch_norm() and not cfg.has_residual_add() and not has_caller_sbm,
        "conv3d with groups > 1 supports no batchnorm, no residual add and no caller SbufManager",
    )
    kernel_assert(
        cfg.D == 1 and cfg.K_d == 1 and cfg.pad_d_left == 0 and cfg.pad_d_right == 0,
        f"conv3d with groups > 1 supports 2D calls only (D = 1, K_d = 1, no depth padding), got D={cfg.D}, "
        f"K_d={cfg.K_d}",
    )
    kernel_assert(cfg.stride_h >= 1 and cfg.stride_w >= 1, f"bad stride ({cfg.stride_h}, {cfg.stride_w})")
    kernel_assert(cfg.dilation_h >= 1 and cfg.dilation_w >= 1, f"bad dilation ({cfg.dilation_h}, {cfg.dilation_w})")
    kernel_assert(
        cfg.pad_h_top >= 0 and cfg.pad_h_bottom >= 0 and cfg.pad_w_left >= 0 and cfg.pad_w_right >= 0,
        f"bad padding ({cfg.pad_h_top}, {cfg.pad_h_bottom}, {cfg.pad_w_left}, {cfg.pad_w_right})",
    )
    kernel_assert(
        groups >= 1 and cfg.C_in % groups == 0 and cfg.C_out % groups == 0,
        f"groups {groups} must divide C_in {cfg.C_in} and C_out {cfg.C_out}",
    )
    C_in_g = filters.shape[3]
    kernel_assert(
        C_in_g == cfg.C_in // groups, f"filters.shape[3] must be C_in / groups = {cfg.C_in // groups}, got {C_in_g}"
    )
    kernel_assert(
        C_in_g <= nl.tile_size.pmax and cfg.C_out // groups <= nl.tile_size.pmax, "at most 128 channels per group"
    )
    kernel_assert(
        x_in.dtype == nl.float32 or x_in.dtype == nl.bfloat16 or x_in.dtype == nl.float16,
        f"x_in dtype must be float32, bfloat16 or float16, got {x_in.dtype}",
    )
    kernel_assert(x_in.dtype == filters.dtype, f"dtype mismatch: {x_in.dtype} vs {filters.dtype}")
    if bias != None:
        kernel_assert(bias.dtype == x_in.dtype, f"bias dtype mismatch: {bias.dtype} vs {x_in.dtype}")
    if cfg.activation_fn != None:
        kernel_assert(
            isinstance(cfg.activation_fn, ActFnType), f"Unsupported activation function type: {cfg.activation_fn}"
        )
        kernel_assert(cfg.activation_fn != ActFnType.SquaredReLU, "conv3d with groups > 1 does not support SquaredReLU")
    kernel_assert(cfg.H_out >= 1 and cfg.W_out >= 1, f"empty output {cfg.H_out} x {cfg.W_out}")


def _depthwise_conv1d_grouped_compute(
    img_ref: nl.NkiTensor,
    filter_ref: nl.NkiTensor,
    padding: tuple,
    stride: tuple,
) -> nl.NkiTensor:
    """
    A depthwise_conv1d_implicit_gemm call on the one-row diagonal path of the grouped path: [N, C, 1, W] input,
    [C, 1, 1, S] filters (read in that layout), stride (1, stride_w), padding ((0, 0), (left, right)); returns
    [N, C, 1, Q]. depthwise_conv1d_implicit_gemm routes here when _grouped_1d_wins holds.
    """
    N, C, _, W = img_ref.shape
    S = filter_ref.shape[3]
    W_pad_l, W_pad_r = padding[1]
    Q = (W + W_pad_l + W_pad_r - S) // stride[1] + 1
    y_out = nl.ndarray((N, C, 1, Q), dtype=img_ref.dtype, buffer=nl.shared_hbm)
    _conv2d_grouped_banded_into(
        y_out=y_out,
        x_in=img_ref,
        filters=filter_ref,
        bias=None,
        stride=(1, stride[1]),
        padding=(0, 0, W_pad_l, W_pad_r),
        dilation=(1, 1),
        groups=C,
        activation_fn=None,
        filter_layout=_FILTERS_C11S,
    )
    return y_out


def _grouped_1d_wins(img_ref: nl.NkiTensor, filter_ref: nl.NkiTensor, padding: tuple, stride: tuple) -> bool:
    """
    True when a depthwise_conv1d_implicit_gemm call ([N, C, 1, W] input, [C, 1, 1, S] filters, stride (1, s), zero
    padding ((0, 0), (left, right)), no dilation) runs faster on the one-row diagonal path of the grouped path
    (_depthwise_conv1d_grouped_compute). That kernel multiplies one channel at a time: per channel and per 128 taps one
    matmul over its output columns, from S overlapping input windows it loads per channel. The diagonal path covers
    128 channels per matmul (one matmul per tap) and loads each channel's row once. Measured on trn2 (LNC=2):
    20x-190x on Mamba, Conformer and Evo 2 layers, 2x-35x at 256 and 512 taps, and no gain under 128 channels, where
    both kernels run at the launch floor. Taken from 128 channels and 16 outputs, up to 512 taps (the longest filter
    measured), when everything a pass holds in SBUF fits at once (_grouped_one_row_sbuf_bytes: the padded rows, the
    results and the S x 128 diagonal stationaries); every other call stays on the original kernel.
    """
    N, C, _, W = img_ref.shape
    S = filter_ref.shape[3]
    W_pad_l, W_pad_r = padding[1]
    Q = (W + W_pad_l + W_pad_r - S) // stride[1] + 1
    wins = (
        padding[0][0] == 0
        and padding[0][1] == 0
        and img_ref.dtype == filter_ref.dtype
        and (img_ref.dtype == nl.bfloat16 or img_ref.dtype == nl.float16 or img_ref.dtype == nl.float32)
        and C >= _GROUPED_1D_MIN_CHANNELS
        and S <= _GROUPED_1D_MAX_TAPS
        and Q >= _GROUPED_1D_MIN_OUTPUTS
    )
    if wins:
        plan = _plan_grouped_band(N, 1, 1, C, 1, W, 1, 1, stride[1], W_pad_l + W_pad_r, 1, S - 1)
        wins = (
            _grouped_one_row_sbuf_bytes(plan, S, sizeinbytes(img_ref.dtype)) <= nl.tile_size.total_available_sbuf_size
        )
    return wins


def _grouped_one_row_sbuf_bytes(plan: _GroupedBandPlan, S: int, dsize: int) -> int:
    """
    SBUF bytes per partition the one-row diagonal path holds in a pass of an S-tap filter: the input tile (the pass's
    padded rows and the filter's reach), the result tile, the diagonal stationaries (S x 128), the filter columns, and
    the identity with its int32 and fp32 build temporaries.
    """
    pmax = nl.tile_size.pmax
    return (plan.tile_cols + plan.n_img * plan.seg_out + (pmax + 1) * S + pmax) * dsize + 2 * 4 * pmax


def _conv2d_grouped_banded_into(
    y_out: nl.NkiTensor,
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
    dilation: tuple[int, int],
    groups: int,
    activation_fn: Optional[ActFnType],
    filter_layout: int,
) -> None:
    """
    Compute the grouped convolution into y_out [B, C_out, H_out, W_out] from x_in [B, C_in, H, W]. filter_layout is
    _FILTERS_HWIO for [K_h, K_w, C_in / groups, C_out] filters, or _FILTERS_C11S for the [C, 1, 1, S] one-row
    depthwise filters of depthwise_conv1d_implicit_gemm (only the one-row diagonal path reads that layout). padding is
    (top, bottom, left, right).
    """
    pmax = nl.tile_size.pmax
    psum_cols = nl.tile_size.psum_fmax
    B, C_in, H, W = x_in.shape
    if filter_layout == _FILTERS_C11S:
        K_h, K_w, C_in_g, C_out = (1, filters.shape[3], 1, filters.shape[0])
    else:
        K_h, K_w, C_in_g, C_out = filters.shape
    stride_h, stride_w = stride
    pad_t, pad_b, pad_l, pad_r = padding
    dil_h, dil_w = dilation
    C_out_g = C_out // groups
    _, _, H_out, W_out = y_out.shape
    plan = _plan_grouped_band(
        B, C_in_g, C_out_g, groups, H_out, W, K_h, stride_h, stride_w, pad_l + pad_r, dil_h, dil_w * (K_w - 1)
    )
    # SPMD sharding info: n_prgs = core count, prg_id = this core (0/1 at LNC=2).
    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d_grouped", (0, 1))
    act = nl.copy
    if activation_fn != None:
        act = get_nl_act_fn_from_type(activation_fn)
    if filter_layout == _FILTERS_HWIO and _grouped_planes_cheaper(
        plan, B, C_in, H, W, H_out, W_out, K_h, K_w, C_in_g, C_out_g, stride, padding, x_in.dtype
    ):
        _grouped_depthwise_planes(x_in, filters, bias, y_out, padding, dilation, act, n_prgs, prg_id)
        return
    total = plan.n_packs * plan.n_img_groups * plan.n_blocks
    per_prg = div_ceil(total, n_prgs)
    work_start = min(total, per_prg * prg_id)
    work_end = min(total, work_start + per_prg)

    # zero-padded filter columns: V[i, (o, k_w, pre + dil_h * k_h)] = filters[k_h, k_w, i, o], pre = stride_h *
    # (rows_out - 1), so that stationary entry (row j, slot s) = V[i, (o, k_w, pre + j - stride_h * s)]
    pre = stride_h * (plan.rows_out - 1)
    l_v = pre + plan.span
    # one row per channel and a one-row filter (1-D depthwise convolutions): the stationaries are diagonal
    diag = plan.span == 1 and plan.rows_out == 1 and K_h == 1 and C_in_g == 1 and C_out_g == 1
    kernel_assert(diag or filter_layout == _FILTERS_HWIO, "[C, 1, 1, S] filters need the one-row diagonal path")
    band_idx, expand, ident = (None, None, None)
    if work_start < work_end:
        if diag:
            ident = _grouped_identity(filters.dtype)
        else:
            band_idx, expand = _grouped_band_constants(plan, C_in_g, C_out_g, K_w, l_v, pre, stride_h, filters.dtype)

    # the last image of a pass needs only its W_out valid columns (with a long filter most of a segment is junk)
    n_cols = (plan.n_img - 1) * plan.seg_out + W_out
    stat = None
    bias_col = None
    current_pack = -1
    for work_idx in range(work_start, work_end):
        pack = work_idx // (plan.n_img_groups * plan.n_blocks)
        img_group = (work_idx // plan.n_blocks) % plan.n_img_groups
        block = work_idx % plan.n_blocks
        g0 = pack * plan.g_pack
        n_groups = min(plan.g_pack, groups - g0)
        if pack != current_pack:
            if diag:
                stat = _build_grouped_diag_stationaries(filters, ident, g0, n_groups, K_w, filter_layout)
            else:
                stat = _build_grouped_band_stationaries(
                    filters, band_idx, expand, plan, g0, n_groups, C_in_g, C_out_g, K_h, K_w, l_v, pre, dil_h
                )
            bias_col = _load_grouped_bias_column(bias, plan, g0 * C_out_g, n_groups * C_out_g)
            current_pack = pack
        b0 = img_group * plan.n_img
        n_img = min(plan.n_img, B - b0)
        row0 = block * plan.rows_out
        rows_valid = min(plan.rows_out, H_out - row0)
        tile = _load_grouped_input_block(
            x_in, plan, b0, n_img, g0 * C_in_g, n_groups * C_in_g, row0 * stride_h - pad_t, pad_l
        )
        result = nl.ndarray((pmax, plan.n_img * plan.seg_out), dtype=x_in.dtype, buffer=nl.sbuf)
        parts_in = n_groups * C_in_g * plan.span
        parts_out = n_groups * C_out_g * plan.rows_out
        for chunk in range(div_ceil(n_cols, psum_cols)):
            c0 = chunk * psum_cols
            cv = min(psum_cols, n_cols - c0)
            psum = nl.ndarray((pmax, psum_cols), dtype=nl.float32, buffer=nl.psum)
            for kw in range(K_w):
                start = stride_w * c0 + dil_w * kw
                if stride_w == 1:
                    moving = tile[0:parts_in, start : start + cv]
                else:
                    moving = tile[0:parts_in, start : start + stride_w * (cv - 1) + 1 : stride_w]
                nisa.nc_matmul(
                    dst=psum[0:parts_out, 0:cv],
                    stationary=stat[0:parts_in, kw, 0:parts_out],
                    moving=moving,
                    accumulate=(kw > 0),
                )
            nisa.activation(
                dst=result[0:parts_out, c0 : c0 + cv],
                op=act,
                data=psum[0:parts_out, 0:cv],
                bias=bias_col[0:parts_out, 0:1],
                scale=1.0,
            )
        _store_grouped_output_block(
            y_out, result, plan, b0, n_img, g0 * C_out_g, n_groups * C_out_g, row0, rows_valid, W_out
        )


def _grouped_planes_cheaper(
    plan: _GroupedBandPlan,
    B: int,
    C: int,
    H: int,
    W: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    C_in_g: int,
    C_out_g: int,
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
    dtype,
) -> bool:
    """
    True when a depthwise layer runs faster as whole planes with diagonal stationaries (_grouped_depthwise_planes)
    than banded. Rough per-core microsecond costs of each engine stream, the slowest one taken:
      banded: per pack the stationary build (GpSimd gather, about 3.4 ns per stationary element per partition);
        per pass K_w matmuls per chunk (TensorE, at least ~200 columns each for the weight load, throttled x1.5);
        the row DMAs (about 1 ns per image row).
      planes: per 128 planes K_h * K_w matmuls per chunk of output rows, the diagonal build (VectorE, K_h * K_w
        * 128 elements), the staging copy (GpSimd, H * W elements) and contiguous plane DMAs.
    Planes need stride 1, a padded row of at most 512 columns and at most 8192 pixels per plane.
    """
    pmax = nl.tile_size.pmax
    psum_cols = nl.tile_size.psum_fmax
    pad_t, pad_b, pad_l, pad_r = padding
    Wp = W + pad_l + pad_r
    if not (C_in_g == 1 and C_out_g == 1 and stride[0] == 1 and stride[1] == 1 and H > 1):
        return False
    if Wp > psum_cols or H * W > 8192 or K_h * K_w > 81:
        return False
    items = plan.n_packs * plan.n_img_groups * plan.n_blocks
    n_cols = (plan.n_img - 1) * plan.seg_out + W_out
    chunks = div_ceil(n_cols, psum_cols)
    cols = min(n_cols, psum_cols)
    width = K_w * plan.g_pack * plan.rows_out
    b_build = plan.n_packs * (0.6 + 0.0034 * width)
    b_pe = items * chunks * K_w * max(cols, 200) * 1.5 / 2400.0
    b_dma = items * plan.n_img * (plan.g_pack * plan.span + plan.g_pack * plan.rows_out) * 0.001
    banded = max(b_build, max(b_pe, b_dma))
    taps = K_h * K_w
    passes = div_ceil(B * C, pmax)
    rpc = max(1, min(H_out, psum_cols // Wp))
    p_chunks = div_ceil(H_out, rpc)
    p_cols = div_ceil(H_out, p_chunks) * Wp
    d_pe = passes * p_chunks * taps * max(p_cols, 200) * 1.5 / 2400.0
    d_build = passes * (0.3 + taps * pmax / 960.0)
    d_stage = passes * (0.3 + H * W / 1200.0)
    dsize = 4 if dtype == nl.float32 else 2
    d_dma = passes * 2 * pmax * H * W * dsize / 360000.0
    planes = max(max(d_pe, d_build), max(d_stage, d_dma))
    return planes < 0.8 * banded


def _grouped_depthwise_planes(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    y_out: nl.NkiTensor,
    padding: tuple[int, int, int, int],
    dilation: tuple[int, int],
    act,
    n_prgs: int,
    prg_id: int,
) -> None:
    """
    Stride-1 depthwise convolution with one whole (image, channel) plane per partition: 128 consecutive planes of
    x_in (one contiguous DMA) per pass, staged by GpSimd into a zero-padded row-major layout of pitch Wp, and for
    every filter tap one matmul by the diagonal stationary diag(filters[k_h, k_w, 0, channel]) over a shifted
    window of the padded rows (the shift is k_h * dilation_h * Wp + k_w * dilation_w). The padding columns of
    each row give junk outputs, dropped by the PSUM evacuation (bias and activation fused), and the result is one
    contiguous DMA. No per-pass setup but the diagonal (one VectorE multiply) and a few small filter DMAs.
    LNC=2: the passes are split into two contiguous ranges.
    """
    pmax = nl.tile_size.pmax
    psum_cols = nl.tile_size.psum_fmax
    B, C, H, W = x_in.shape
    K_h, K_w, _, _ = filters.shape
    _, _, H_out, W_out = y_out.shape
    pad_t, pad_b, pad_l, pad_r = padding
    dil_h, dil_w = dilation
    Wp = W + pad_l + pad_r
    Hp = H + pad_t + pad_b
    taps = K_h * K_w
    n_planes = B * C
    n_pass = div_ceil(n_planes, pmax)
    per_prg = div_ceil(n_pass, n_prgs)
    p_start = min(n_pass, per_prg * prg_id)
    p_end = min(n_pass, p_start + per_prg)
    rpc = max(1, min(H_out, psum_cols // Wp))
    n_chunks = div_ceil(H_out, rpc)
    rpc = div_ceil(H_out, n_chunks)
    tile_cols = (Hp + 1) * Wp + dil_w * (K_w - 1)
    ident = None
    if p_start < p_end:
        ident = _grouped_identity(filters.dtype)
    for k in range(p_start, p_end):
        q0 = k * pmax
        n_p = min(pmax, n_planes - q0)
        # the planes' filter taps and biases: runs of consecutive channels (a pass may wrap to the next image)
        fcol = nl.ndarray((pmax, taps), dtype=filters.dtype, buffer=nl.sbuf)
        bcol = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        if bias == None:
            nisa.memset(dst=bcol[0:n_p, 0:1], value=0.0)
        p = 0
        while p < n_p:
            c = (q0 + p) % C
            n_seg = min(n_p - p, C - c)
            nisa.dma_copy(dst=fcol[p : p + n_seg, 0:taps], src=filters.ap(pattern=[[1, n_seg], [C, taps]], offset=c))
            if bias != None:
                nisa.dma_copy(dst=bcol[p : p + n_seg, 0:1], src=bias.ap(pattern=[[1, n_seg], [1, 1]], offset=c))
            p = p + n_seg
        stat = nl.ndarray((pmax, taps, pmax), dtype=filters.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=stat[0:n_p, 0:taps, 0:n_p],
            data1=ident.ap(pattern=[[pmax, n_p], [0, taps], [1, n_p]], offset=0),
            data2=fcol.ap(pattern=[[taps, n_p], [1, taps], [0, n_p]], offset=0),
            op=nl.multiply,
        )
        stage = nl.ndarray((pmax, H * W), dtype=x_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=stage[0:n_p, 0 : H * W],
            src=x_in.ap(pattern=[[H * W, n_p], [1, H * W]], offset=q0 * H * W),
            dge_mode=nisa.dge_mode.none,
        )
        tile = nl.ndarray((pmax, tile_cols), dtype=x_in.dtype, buffer=nl.sbuf)
        rows = tile[0:n_p, pad_t * Wp : (pad_t + H) * Wp].reshape_dim(1, (H, Wp))
        if pad_t > 0:
            nisa.memset(dst=tile[0:n_p, 0 : pad_t * Wp], value=0.0)
        nisa.memset(dst=tile[0:n_p, (pad_t + H) * Wp : tile_cols], value=0.0)
        if pad_l > 0:
            nisa.memset(dst=rows.slice(dim=2, start=0, end=pad_l, step=1), value=0.0)
        if pad_r > 0:
            nisa.memset(dst=rows.slice(dim=2, start=pad_l + W, end=Wp, step=1), value=0.0)
        nisa.tensor_copy(
            dst=rows.slice(dim=2, start=pad_l, end=pad_l + W, step=1),
            src=stage[0:n_p, 0 : H * W].reshape_dim(1, (H, W)),
            engine=nisa.engine.gpsimd,
        )
        result = nl.ndarray((pmax, H_out * W_out), dtype=x_in.dtype, buffer=nl.sbuf)
        for g0 in range(0, n_chunks, 4):
            n_g = min(4, n_chunks - g0)
            psums = []
            for ch in range(n_g):
                psums.append(nl.ndarray((pmax, psum_cols), dtype=nl.float32, buffer=nl.psum))
            for t in range(taps):
                kh = t // K_w
                kw = t % K_w
                for ch in range(n_g):
                    r0 = (g0 + ch) * rpc
                    rv = min(rpc, H_out - r0)
                    start = (r0 + kh * dil_h) * Wp + kw * dil_w
                    nisa.nc_matmul(
                        dst=psums[ch][0:n_p, 0 : rv * Wp],
                        stationary=stat[0:n_p, t, 0:n_p],
                        moving=tile[0:n_p, start : start + rv * Wp],
                        accumulate=(t > 0),
                    )
            for ch in range(n_g):
                r0 = (g0 + ch) * rpc
                rv = min(rpc, H_out - r0)
                nisa.activation(
                    dst=result[0:n_p, r0 * W_out : (r0 + rv) * W_out].reshape_dim(1, (rv, W_out)),
                    op=act,
                    data=psums[ch][0:n_p, 0 : rv * Wp]
                    .reshape_dim(1, (rv, Wp))
                    .slice(dim=2, start=0, end=W_out, step=1),
                    bias=bcol[0:n_p, 0:1],
                    scale=1.0,
                )
        nisa.dma_copy(
            dst=y_out.ap(pattern=[[H_out * W_out, n_p], [1, H_out * W_out]], offset=q0 * H_out * W_out),
            src=result[0:n_p, 0 : H_out * W_out],
            dge_mode=nisa.dge_mode.none,
        )


def _grouped_identity(dtype) -> nl.NkiTensor:
    """[128, 128] identity in dtype (for the diagonal stationaries of depthwise filters)."""
    pmax = nl.tile_size.pmax
    off_i = nl.ndarray((pmax, pmax), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=off_i, pattern=[[1, pmax]], offset=0, channel_multiplier=-1)
    off = nl.ndarray((pmax, pmax), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=off, src=off_i, engine=nisa.engine.vector)
    ident = nl.ndarray((pmax, pmax), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident, data=off, op0=nl.equal, operand0=0.0)
    return ident


def _build_grouped_diag_stationaries(
    filters: nl.NkiTensor,
    ident: nl.NkiTensor,
    g0: int,
    n_ch: int,
    K_w: int,
    filter_layout: int,
) -> nl.NkiTensor:
    """
    stat[c, k_w, c'] = (c == c') * (filter tap k_w of channel g0 + c) for the n_ch channels from g0: one small DMA
    of the filter columns ([1, K_w, 1, C] filters: tap stride C; [C, 1, 1, S] filters: one contiguous row per
    channel) and one VectorE multiply of the identity (broadcast over k_w) by them (broadcast over c').
    """
    pmax = nl.tile_size.pmax
    fcol = nl.ndarray((pmax, K_w), dtype=filters.dtype, buffer=nl.sbuf)
    if filter_layout == _FILTERS_C11S:
        src = filters.ap(pattern=[[K_w, n_ch], [1, K_w]], offset=g0 * K_w)
    else:
        src = filters.ap(pattern=[[1, n_ch], [filters.shape[3], K_w]], offset=g0)
    nisa.dma_copy(dst=fcol[0:n_ch, 0:K_w], src=src)
    stat = nl.ndarray((pmax, K_w, pmax), dtype=filters.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=stat[0:n_ch, 0:K_w, 0:n_ch],
        data1=ident.ap(pattern=[[pmax, n_ch], [0, K_w], [1, n_ch]], offset=0),
        data2=fcol.ap(pattern=[[K_w, n_ch], [1, K_w], [0, n_ch]], offset=0),
        op=nl.multiply,
    )
    return stat


def _grouped_band_constants(
    plan: _GroupedBandPlan,
    C_in_g: int,
    C_out_g: int,
    K_w: int,
    l_v: int,
    pre: int,
    stride_h: int,
    dtype,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    The pass-independent tiles of the stationary build, made once.

    band_idx[(c, j), k_w, (o, s)] (uint32) is the element of contraction row (c, j)'s expanded filter row that
    stationary entry (k_w, o, s) takes: (o % C_out_g) * K_w * l_v + k_w * l_v + pre + j - stride_h * s when the
    input channel c and the output channel o belong to the same group of the pass, else the row's trailing zero.
    expand[c, (c', j)] = (c == c') (dtype) gives every contraction row of channel c that channel's filter row.
    """
    pmax = nl.tile_size.pmax
    span = plan.span
    rows = plan.rows_out
    n_ci = plan.g_pack * C_in_g
    n_co = plan.g_pack * C_out_g
    zero_at = C_out_g * K_w * l_v
    width = K_w * n_co * rows
    # per partition p = c * span + j: the channel c (number of block starts at or below p, minus one) and its group
    part_i = nl.ndarray((pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=part_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    part = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=part, src=part_i, engine=nisa.engine.vector)
    starts_i = nl.ndarray((pmax, n_ci), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=starts_i, pattern=[[span, n_ci]], offset=0, channel_multiplier=0)
    starts = nl.ndarray((pmax, n_ci), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=starts, src=starts_i, engine=nisa.engine.vector)
    seen = nl.ndarray((pmax, n_ci), dtype=nl.float32, buffer=nl.sbuf)
    chan = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(
        dst=seen, data=starts, op0=nl.less_equal, operand0=part, reduce_op=nl.add, reduce_res=chan
    )
    # chan = count - 1; group = floor(chan / C_in_g) the same way over the group starts
    nisa.tensor_scalar(dst=chan, data=chan, op0=nl.subtract, operand0=1.0)
    g_starts_i = nl.ndarray((pmax, plan.g_pack), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=g_starts_i, pattern=[[C_in_g, plan.g_pack]], offset=0, channel_multiplier=0)
    g_starts = nl.ndarray((pmax, plan.g_pack), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_starts, src=g_starts_i, engine=nisa.engine.vector)
    g_seen = nl.ndarray((pmax, plan.g_pack), dtype=nl.float32, buffer=nl.sbuf)
    group = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(
        dst=g_seen, data=g_starts, op0=nl.less_equal, operand0=chan, reduce_op=nl.add, reduce_res=group
    )
    nisa.tensor_scalar(dst=group, data=group, op0=nl.subtract, operand0=1.0)
    # row offset of the channel's block: -span * chan, so that p - span * chan = j
    shift = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shift, data=chan, op0=nl.multiply, operand0=float(-span))
    # value[p, (k_w, g, o, s)] = o * K_w * l_v + k_w * l_v + pre - stride_h * s + p, then + shift = ... + j
    val_i = nl.ndarray((pmax, width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        dst=val_i.reshape_dim(1, (K_w, plan.g_pack, C_out_g, rows)),
        pattern=[[l_v, K_w], [0, plan.g_pack], [K_w * l_v, C_out_g], [-stride_h, rows]],
        offset=pre,
        channel_multiplier=1,
    )
    val = nl.ndarray((pmax, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=val, src=val_i, engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=val, data=val, op0=nl.add, operand0=shift)
    # same group mask, then idx = zero_at + mask * (value - zero_at)
    col_g_i = nl.ndarray((pmax, width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        dst=col_g_i.reshape_dim(1, (K_w, plan.g_pack, C_out_g, rows)),
        pattern=[[0, K_w], [1, plan.g_pack], [0, C_out_g], [0, rows]],
        offset=0,
        channel_multiplier=0,
    )
    col_g = nl.ndarray((pmax, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=col_g, src=col_g_i, engine=nisa.engine.vector)
    mask = nl.ndarray((pmax, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mask, data=col_g, op0=nl.equal, operand0=group)
    idx_f = nl.ndarray((pmax, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=idx_f, data=val, op0=nl.subtract, operand0=float(zero_at), op1=nl.multiply, operand1=mask
    )
    nisa.tensor_scalar(dst=idx_f, data=idx_f, op0=nl.add, operand0=float(zero_at))
    band_idx = nl.ndarray((pmax, width), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=band_idx, src=idx_f, engine=nisa.engine.vector)
    # expand[c, q] = (0 <= q - span * c < span)
    off_i = nl.ndarray((pmax, pmax), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=off_i, pattern=[[1, pmax]], offset=0, channel_multiplier=-span)
    off = nl.ndarray((pmax, pmax), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=off, src=off_i, engine=nisa.engine.vector)
    lo = nl.ndarray((pmax, pmax), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo, data=off, op0=nl.greater_equal, operand0=0.0)
    hi = nl.ndarray((pmax, pmax), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi, data=off, op0=nl.less, operand0=float(span))
    expand = nl.ndarray((pmax, pmax), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=expand, data1=lo, data2=hi, op=nl.multiply)
    return band_idx, expand


def _build_grouped_band_stationaries(
    filters: nl.NkiTensor,
    band_idx: nl.NkiTensor,
    expand: nl.NkiTensor,
    plan: _GroupedBandPlan,
    g0: int,
    n_groups: int,
    C_in_g: int,
    C_out_g: int,
    K_h: int,
    K_w: int,
    l_v: int,
    pre: int,
    dil_h: int,
) -> nl.NkiTensor:
    """
    stat[(c, j), k_w, (o, s)] for the n_groups groups from g0: within a group, the Toeplitz block of input channel
    c and output channel o, T[j, s] = filters[k_h, k_w, c, o] where j - stride_h * s = dil_h * k_h; zero between
    groups. V (one row per input channel: its zero-padded filter columns for every output channel of its group, and
    a trailing zero) gets one small DMA per group; expand^T @ V copies each row to its channel's span contraction
    rows; nc_n_gather with band_idx places the values on the band.
    """
    pmax = nl.tile_size.pmax
    psum_cols = nl.tile_size.psum_fmax
    span = plan.span
    rows = plan.rows_out
    n_ci = n_groups * C_in_g
    parts_in = n_ci * span
    _, _, _, C_out = filters.shape
    row_len = C_out_g * K_w * l_v + 1
    v = nl.ndarray((pmax, row_len), dtype=filters.dtype, buffer=nl.sbuf)
    nisa.memset(dst=v[0:n_ci, 0:row_len], value=0.0)
    for gp in range(n_groups):
        nisa.dma_copy(
            dst=v[gp * C_in_g : (gp + 1) * C_in_g, 0 : row_len - 1]
            .reshape_dim(1, (C_out_g, K_w, l_v))
            .slice(dim=3, start=pre, end=pre + dil_h * (K_h - 1) + 1, step=dil_h),
            src=filters.ap(
                pattern=[[C_out, C_in_g], [1, C_out_g], [C_in_g * C_out, K_w], [K_w * C_in_g * C_out, K_h]],
                offset=(g0 + gp) * C_out_g,
            ),
        )
    rows_sb = nl.ndarray((pmax, row_len), dtype=filters.dtype, buffer=nl.sbuf)
    for ch in range(div_ceil(row_len, psum_cols)):
        c0 = ch * psum_cols
        cv = min(psum_cols, row_len - c0)
        rows_ps = nl.ndarray((pmax, psum_cols), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=rows_ps[0:parts_in, 0:cv], stationary=expand[0:n_ci, 0:parts_in], moving=v[0:n_ci, c0 : c0 + cv]
        )
        nisa.tensor_copy(
            dst=rows_sb[0:parts_in, c0 : c0 + cv], src=rows_ps[0:parts_in, 0:cv], engine=nisa.engine.scalar
        )
    width = K_w * plan.g_pack * C_out_g * rows
    stat = nl.ndarray((pmax, K_w, plan.g_pack * C_out_g * rows), dtype=filters.dtype, buffer=nl.sbuf)
    nisa.nc_n_gather(
        dst=stat[0:parts_in, 0:K_w, 0 : plan.g_pack * C_out_g * rows],
        data=rows_sb[0:parts_in, 0:row_len],
        indices=band_idx[0:parts_in, 0:width],
    )
    if filters.dtype == nl.float32:
        # With 4-byte data the matmuls of a pack were seen to read stationary columns before the multi-instruction
        # gather had written them (whole channels wrong, deterministically, on 31 x 31 filters with several images
        # per pass). A VectorE copy reads every gathered element, so the matmuls, which read the copy, wait for all
        # of the gather.
        settled = nl.ndarray((pmax, K_w, plan.g_pack * C_out_g * rows), dtype=filters.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=settled[0:parts_in, 0:K_w, 0 : plan.g_pack * C_out_g * rows],
            src=stat[0:parts_in, 0:K_w, 0 : plan.g_pack * C_out_g * rows],
            engine=nisa.engine.vector,
        )
        return settled
    return stat


def _load_grouped_bias_column(
    bias: Optional[nl.NkiTensor], plan: _GroupedBandPlan, o0: int, n_out: int
) -> nl.NkiTensor:
    """fp32 [128, 1] column with partition (o, s) <- bias[o0 + o] (zeros without a bias)."""
    pmax = nl.tile_size.pmax
    rows = plan.rows_out
    parts = n_out * rows
    column = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    if bias == None:
        nisa.memset(dst=column[0:pmax, 0:1], value=0.0)
    else:
        row = nl.ndarray((1, parts), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=row[0:1, 0:parts].reshape_dim(1, (n_out, rows)),
            src=bias.ap(pattern=[[n_out, 1], [1, n_out], [0, rows]], offset=o0),
        )
        row_f32 = nl.ndarray((1, parts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row_f32[0:1, 0:parts], src=row[0:1, 0:parts])
        one = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=one[0:1, 0:1], value=1.0)
        column_psum = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=column_psum[0:parts, 0:1], stationary=row_f32[0:1, 0:parts], moving=one[0:1, 0:1])
        nisa.tensor_copy(dst=column[0:parts, 0:1], src=column_psum[0:parts, 0:1])
    return column


def _load_grouped_input_block(
    x_in: nl.NkiTensor,
    plan: _GroupedBandPlan,
    b0: int,
    n_img: int,
    c0: int,
    n_ch: int,
    in_row0: int,
    pad_l: int,
) -> nl.NkiTensor:
    """
    tile[(c, j), img * seg_in + pad_l + col] = x_in[b0 + img, c0 + c, in_row0 + j, col] for the n_ch channels from
    c0, zero outside the image. One DMA for all images and channels when every row of the block lies inside the
    image and the block spans the image height (the channels' rows are then one contiguous run, which the DMA needs
    on the partition axis), else one DMA per channel (all images) over the valid rows.
    """
    pmax = nl.tile_size.pmax
    _, C_in, H, W = x_in.shape
    span = plan.span
    tile_cols = plan.tile_cols
    tile = nl.ndarray((pmax, tile_cols), dtype=x_in.dtype, buffer=nl.sbuf)
    parts = n_ch * span
    j_lo = max(0, -in_row0)
    j_hi = min(span, H - in_row0)
    seg = plan.seg_in
    if j_lo == 0 and j_hi == span:
        # every row is loaded: only the padding columns of each image segment and the tail need zeros
        segs = tile[0:parts, 0 : n_img * seg].reshape_dim(1, (n_img, seg))
        if pad_l > 0:
            nisa.memset(dst=segs.slice(dim=2, start=0, end=pad_l, step=1), value=0.0)
        if pad_l + W < seg:
            nisa.memset(dst=segs.slice(dim=2, start=pad_l + W, end=seg, step=1), value=0.0)
        if n_img * seg < tile_cols:
            nisa.memset(dst=tile[0:parts, n_img * seg : tile_cols], value=0.0)
    else:
        nisa.memset(dst=tile[0:parts, 0:tile_cols], value=0.0)
    if j_lo < j_hi:
        if j_lo == 0 and j_hi == span and span == H:
            nisa.dma_copy(
                dst=tile[0:parts, 0 : n_img * plan.seg_in]
                .reshape_dim(1, (n_img, plan.seg_in))
                .slice(dim=2, start=pad_l, end=pad_l + W, step=1),
                src=x_in.ap(
                    pattern=[[H * W, n_ch], [W, span], [C_in * H * W, n_img], [1, W]],
                    offset=((b0 * C_in + c0) * H + in_row0) * W,
                ),
                dge_mode=nisa.dge_mode.none,
            )
        else:
            for c in range(n_ch):
                nisa.dma_copy(
                    dst=tile[c * span + j_lo : c * span + j_hi, 0 : n_img * plan.seg_in]
                    .reshape_dim(1, (n_img, plan.seg_in))
                    .slice(dim=2, start=pad_l, end=pad_l + W, step=1),
                    src=x_in.ap(
                        pattern=[[W, j_hi - j_lo], [C_in * H * W, n_img], [1, W]],
                        offset=((b0 * C_in + c0 + c) * H + in_row0 + j_lo) * W,
                    ),
                    dge_mode=nisa.dge_mode.none,
                )
    return tile


def _store_grouped_output_block(
    y_out: nl.NkiTensor,
    result: nl.NkiTensor,
    plan: _GroupedBandPlan,
    b0: int,
    n_img: int,
    o0: int,
    n_out: int,
    row0: int,
    rows_valid: int,
    W_out: int,
) -> None:
    """
    y_out[b0 + img, o0 + o, row0 + r, :] = result[o * rows_out + r, img * seg_out : + W_out]: one DMA when the block
    spans the output height (the channels' rows are one contiguous run), else one per channel.
    """
    _, C_out, H_out, _ = y_out.shape
    rows = plan.rows_out
    seg = plan.seg_out
    if rows_valid == rows and rows == H_out:
        nisa.dma_copy(
            dst=y_out.ap(
                pattern=[[H_out * W_out, n_out], [W_out, rows], [C_out * H_out * W_out, n_img], [1, W_out]],
                offset=((b0 * C_out + o0) * H_out + row0) * W_out,
            ),
            src=result[0 : n_out * rows, 0 : n_img * seg]
            .reshape_dim(1, (n_img, seg))
            .slice(dim=2, start=0, end=W_out, step=1),
            dge_mode=nisa.dge_mode.none,
        )
    else:
        for o in range(n_out):
            nisa.dma_copy(
                dst=y_out.ap(
                    pattern=[[W_out, rows_valid], [C_out * H_out * W_out, n_img], [1, W_out]],
                    offset=((b0 * C_out + o0 + o) * H_out + row0) * W_out,
                ),
                src=result[o * rows : o * rows + rows_valid, 0 : n_img * seg]
                .reshape_dim(1, (n_img, seg))
                .slice(dim=2, start=0, end=W_out, step=1),
                dge_mode=nisa.dge_mode.none,
            )
