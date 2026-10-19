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

"""Shared Conv3D configuration and helpers."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    div_ceil,
    get_nl_act_fn_from_type,
    get_verified_program_sharding_info,
)
from ...core.utils.lnc_sendrecv import lnc_sendrecv
from ...core.utils.logging import get_logger

_PARTITION_STRIDE_32 = 32
_PARTITION_STRIDE_64 = 64
_MAX_K_REP_AT_32 = 4
_MAX_K_REP_AT_64 = 2
_PSUM_BANK_SIZE = nl.tile_size.psum_fmax * 4
_NUM_PSUM_BANKS = 8
_BIAS_DTYPE_SIZE = sizeinbytes(nl.float32)
_HEAP_ALIGN = nl.tile_size.sbuf_min_align
_SBUF_RESERVE = 1024
# Column-tiling column sizes: the 128 PE columns slice into 4x32 (gen3+) or 2x64 (gen2+).
_COLUMN_TILE_32 = 32
_COLUMN_TILE_64 = 64
# Eval-mode batchnorm: shared single-column scratch buffers (gamma, beta, running mean, running
# variance, bias-fold temporary) plus 2 persistent coefficient columns per C_out interleave slot.
_BN_EVAL_SCRATCH_NAMES = ["bn_eval_gamma", "bn_eval_beta", "bn_eval_mean", "bn_eval_var", "bn_eval_tmp"]
_BN_EVAL_COEFF_PER_TILE = 2
# bn_stats writes 6 elements/partition — a (count, mean, variance * count) triple for the even and
# the odd input elements — and bn_aggr consumes those triples, emitting 2 (mean, variance).
_BN_STATS_ELEMS = 6
_BN_AGGR_ELEMS = 2
# One (count, mean, variance * count) triple: the unit _finalize_stats_phase pools with.
_BN_TRIPLE_ELEMS = 3


class BatchNormMode(Enum):
    """
    Fused-batchnorm execution mode selected by the caller of conv3d.

    NONE: no fused batchnorm; the kernel returns the plain convolution output.
    EVAL: normalize with the supplied running statistics and do not update them, equivalent to
        torch.nn.BatchNorm2d in eval() / track_running_stats=False. The normalization is applied
        directly on the PSUM eviction, so no statistics accumulation, conv_out staging buffer or
        finalize pass is needed.
    TRAINING: compute the per-channel batch statistics, normalize with them, and momentum-update
        the running statistics, equivalent to torch.nn.BatchNorm2d in train(). This is the only
        mode that materializes the raw pre-batchnorm convolution output, so it is also the only
        mode in which output_pre_norm may be True.
    """

    NONE = 0
    EVAL = 1
    TRAINING = 2

    def is_fused(self) -> bool:
        """Whether a batchnorm is fused into the convolution at all."""
        return self != BatchNormMode.NONE


class ResidualAddLoc(Enum):
    """
    Insertion point of the fused residual add, selected by the caller of conv3d.

    NONE: no residual add; residuals_in is ignored.
    PRE_ACT: add the residual before the optional activation function, i.e. act(y + residual).
    POST_ACT: add the residual after the optional activation function, i.e. act(y) + residual.

    Both insertion points are valid in every BatchNormMode; the residual is always added to the
    fully post-processed convolution output (after the batchnorm, when one is fused). When no
    activation function is given the two insertion points are equivalent and behave identically.
    """

    NONE = 0
    PRE_ACT = 1
    POST_ACT = 2

    def is_fused(self) -> bool:
        """Whether a residual add is fused into the convolution at all."""
        return self != ResidualAddLoc.NONE


@dataclass
class _BnBandPlan(nl.NKIObject):
    """Batchnorm band layout and counts for one C_out tile."""

    n_bands: int
    band_stride: int
    # Per-band element counts; 0 marks an empty band
    band_counts: list


@dataclass
class _BnFinalizePlan(nl.NKIObject):
    """Geometry-independent batchnorm finalization plan."""

    p_max: int
    c_out_total: int
    # This core's C_out range; narrowed only for C_out sharding
    own_c_out_start: int
    own_c_out_end: int
    stats_per_tile: int
    # Per-channel element count; may be 0 for an empty shard
    own_count: int
    exchange: bool
    prg_id: int
    pool_triples: int
    # Indexed by global C_out tile
    band_plans: list


def _bn_finalize_reserve_bytes(plan: _BnFinalizePlan) -> int:
    stat_size = sizeinbytes(nl.float32)
    total = _heap_alloc_cost(plan.pool_triples * _BN_TRIPLE_ELEMS * stat_size)
    total += _heap_alloc_cost(_BN_AGGR_ELEMS * stat_size)
    if plan.exchange:
        full_c_out_tile_count = div_ceil(plan.c_out_total, plan.p_max)
        total += _heap_alloc_cost(full_c_out_tile_count * _BN_TRIPLE_ELEMS * stat_size)
    return total


@dataclass
class _BatchNormPostProcessConfig(nl.NKIObject):
    p_max: int
    batch_start: int
    batch_end: int
    c_out_start: int
    c_out_end: int
    spatial_start: int
    spatial_end: int
    reload_interleave: int
    reload_free_tile: int
    fixed_c_out_rep: Optional[int]
    # Low-channel per-tile reload layouts
    low_chan_tile_layouts: Optional[list]
    write_momentum: bool


@dataclass
class _BatchNormPostProcessBuffers(nl.NKIObject):
    gamma_buf: nl.NkiTensor
    beta_buf: nl.NkiTensor
    true_gamma_buf: nl.NkiTensor
    neg_true_beta_buf: nl.NkiTensor
    run_running_mean_buf: nl.NkiTensor
    run_new_mean_buf: nl.NkiTensor
    run_running_var_buf: nl.NkiTensor
    run_new_var_buf: nl.NkiTensor
    reload_buf: nl.NkiTensor
    residual_reload_buf: Optional[nl.NkiTensor]


@dataclass
class _Conv3dComputeResult(nl.NKIObject):
    sbm: SbufManager
    heap_depth: int
    means_sbuf: nl.NkiTensor
    variances_sbuf: nl.NkiTensor
    post_config: _BatchNormPostProcessConfig
    post_buffers: _BatchNormPostProcessBuffers


def _band_part_stride(tile_cfg: "Conv3dTileConfig") -> int:
    """
    Partition stride between consecutive column-tiling bands in a band-tall single-column buffer.

    Under column tiling that is the PE column-tile size; without it (C_OUT_REP == 1) there is one
    band, so the stride is just the tile's own channel extent. C_OUT_REP * this is the full span such
    a buffer needs, which is why the bias / affine-coefficient buffers are sized from it.
    """
    if tile_cfg.C_OUT_REP > 1:
        return tile_cfg.col_size
    return tile_cfg.c_out_tile_size_max


def _identity_shuffle_mask() -> list:
    """
    Identity nc_stream_shuffle mask (partition i <- i) over one 32-partition quadrant.

    Used by the column-tiling steps that move a single-column result from one partition band to
    another: the shuffle crosses bands while the mask leaves the within-quadrant order alone.
    Plain list built with a loop — the tracer rejects tuple(range(...)).
    """
    mask = []
    for mask_idx in range(_PARTITION_STRIDE_32):
        mask.append(mask_idx)
    return mask


def _momentum_update_stat(
    running_hbm: nl.NkiTensor,
    new_stat_col: nl.NkiTensor,
    updated_hbm: nl.NkiTensor,
    running_scratch: nl.NkiTensor,
    new_scratch: nl.NkiTensor,
    c_out_tile_start: int,
    c_out_tile_end: int,
    one_minus_momentum: float,
    momentum: float,
    new_scale: float,
) -> None:
    """
    Momentum-update one running statistic for a C_out tile (matches torch.nn.BatchNorm2d):
        updated = one_minus_momentum * running + momentum * (new_scale * new_stat)
    new_scale folds in any per-stat factor (1.0 for the mean, the correction-1 factor N/(N-1) for
    the variance) so both stats share one path. running_scratch / new_scratch are two disjoint
    SBUF scratch columns.
    """
    nisa.dma_copy(dst=running_scratch, src=running_hbm.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
    nisa.tensor_scalar(dst=running_scratch, data=running_scratch, op0=nl.multiply, operand0=one_minus_momentum)
    nisa.tensor_scalar(
        dst=new_scratch, data=new_stat_col, op0=nl.multiply, operand0=new_scale, op1=nl.multiply, operand1=momentum
    )
    nisa.tensor_tensor(dst=running_scratch, data1=running_scratch, data2=new_scratch, op=nl.add)
    nisa.dma_copy(dst=updated_hbm.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1), src=running_scratch)


def _build_fallback_bn_finalize_plan(
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    shard_stats_on_dh: bool,
    prg_id: int,
) -> _BnFinalizePlan:
    p_max = tile_cfg.P_MAX
    n_bands = tile_cfg.C_OUT_REP
    band_stride = tile_cfg.col_size
    band_counts = _bn_band_element_counts(cfg, tile_cfg)
    own_dh = tile_cfg.dh_end - tile_cfg.dh_start
    tile_size_full = min(p_max, cfg.C_out)
    kernel_assert(
        (n_bands - 1) * band_stride + tile_size_full <= p_max,
        f"band span {(n_bands - 1) * band_stride + tile_size_full} exceeds {p_max} partitions",
    )
    # The fallback uses the same band layout for every C_out tile
    band_plans = []
    for _tile_idx in range(div_ceil(cfg.C_out, p_max)):
        band_plans.append(_BnBandPlan(n_bands=n_bands, band_stride=band_stride, band_counts=band_counts))
    return _BnFinalizePlan(
        p_max=p_max,
        c_out_total=cfg.C_out,
        # D-H sharding keeps the full C_out range; C_out sharding narrows it
        own_c_out_start=tile_cfg.C_out_start,
        own_c_out_end=tile_cfg.C_out_end,
        stats_per_tile=tile_cfg.bn_partial_iters * _BN_STATS_ELEMS,
        own_count=cfg.B * own_dh * cfg.W_out,
        exchange=shard_stats_on_dh,
        prg_id=prg_id,
        pool_triples=n_bands + (1 if shard_stats_on_dh else 0),
        band_plans=band_plans,
    )


def _finalize_bn_stats(
    plan: _BnFinalizePlan,
    sbm: SbufManager,
    bn_stats_bufs: nl.NkiTensor,
    recv_triples: Optional[nl.NkiTensor],
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    identity_mask: list,
) -> None:
    """
    Finalize per-channel batchnorm statistics in HBM and SBUF.

    Bands and optional peer shards are pooled by weighted triples, including empty shards.
    After an exchange every core holds the full statistics, but only core 0 writes them to HBM.
    Temporary allocations are popped before return.
    """
    entry_depth = len(sbm.heap)
    full_c_out_tile_count = div_ceil(plan.c_out_total, plan.p_max)
    tile_size_full = min(plan.p_max, plan.c_out_total)

    pool_buf = sbm.alloc_heap(
        shape=(tile_size_full, plan.pool_triples * _BN_TRIPLE_ELEMS), dtype=nl.float32, name="bn_pool"
    )
    # Tall enough for a band-level bn_aggr, which writes on the band's own partitions.
    aggr_out = sbm.alloc_heap(shape=(plan.p_max, _BN_AGGR_ELEMS), dtype=nl.float32, name="bn_aggr_out")
    own_triples = None
    if plan.exchange:
        # sendrecv shapes must match; zeroing represents an empty shard
        own_triples = sbm.alloc_heap(
            shape=(tile_size_full, full_c_out_tile_count * _BN_TRIPLE_ELEMS),
            dtype=nl.float32,
            name="bn_own_triples",
        )
        nisa.memset(dst=own_triples[:, :], value=0.0)

    # Use absolute offsets because own_c_out_start need not be tile-aligned
    for c_out_tile_start in range(plan.own_c_out_start, plan.own_c_out_end, plan.p_max):
        c_out_tile_idx = c_out_tile_start // plan.p_max
        c_out_tile_size = min(c_out_tile_start + plan.p_max, plan.own_c_out_end) - c_out_tile_start
        band_plan = plan.band_plans[c_out_tile_idx]
        block_start = c_out_tile_idx * plan.stats_per_tile

        for band_idx in range(band_plan.n_bands):
            part_lo = band_idx * band_plan.band_stride
            tri_lo = band_idx * _BN_TRIPLE_ELEMS
            if band_plan.band_counts[band_idx] > 0:
                # bn_aggr reads and writes this band's partitions
                nisa.bn_aggr(
                    dst=aggr_out[part_lo : part_lo + c_out_tile_size, 0:_BN_AGGR_ELEMS],
                    data=bn_stats_bufs[
                        part_lo : part_lo + c_out_tile_size, block_start : block_start + plan.stats_per_tile
                    ],
                )
                _store_bn_triple(
                    dst=pool_buf,
                    tri_lo=tri_lo,
                    count=band_plan.band_counts[band_idx],
                    aggr_out=aggr_out,
                    part_lo=part_lo,
                    c_out_tile_size=c_out_tile_size,
                    identity_mask=identity_mask,
                )
            else:
                # Clear stale triples for empty bands
                nisa.memset(dst=pool_buf[:c_out_tile_size, tri_lo : tri_lo + _BN_TRIPLE_ELEMS], value=0.0)

        if plan.own_count > 0:
            # Pool this core's bands
            nisa.bn_aggr(
                dst=aggr_out[:c_out_tile_size, 0:_BN_AGGR_ELEMS],
                data=pool_buf[:c_out_tile_size, 0 : band_plan.n_bands * _BN_TRIPLE_ELEMS],
            )
            if plan.exchange:
                _store_bn_triple(
                    dst=own_triples,
                    tri_lo=c_out_tile_idx * _BN_TRIPLE_ELEMS,
                    count=plan.own_count,
                    aggr_out=aggr_out,
                    part_lo=0,
                    c_out_tile_size=c_out_tile_size,
                    identity_mask=identity_mask,
                )
            else:
                _write_final_stats(
                    c_out_tile_start,
                    c_out_tile_idx,
                    c_out_tile_size,
                    aggr_out,
                    means_sbuf,
                    variances_sbuf,
                    means,
                    variances,
                    write_hbm=True,
                )

    if plan.exchange:
        # Exchange compact triples; peer HBM reads would race
        peer = 1 - plan.prg_id
        lnc_sendrecv(
            src=own_triples[:, :],
            dst=recv_triples[:, :],
            send_to_rank=peer,
            recv_from_rank=peer,
            pipe_id=0,
        )
        pair_elems = 2 * _BN_TRIPLE_ELEMS
        for c_out_tile_start in range(0, plan.c_out_total, plan.p_max):
            c_out_tile_idx = c_out_tile_start // plan.p_max
            c_out_tile_size = min(plan.p_max, plan.c_out_total - c_out_tile_start)
            tri_lo = c_out_tile_idx * _BN_TRIPLE_ELEMS
            nisa.tensor_copy(
                dst=pool_buf[:c_out_tile_size, 0:_BN_TRIPLE_ELEMS],
                src=own_triples[:c_out_tile_size, tri_lo : tri_lo + _BN_TRIPLE_ELEMS],
            )
            nisa.tensor_copy(
                dst=pool_buf[:c_out_tile_size, _BN_TRIPLE_ELEMS:pair_elems],
                src=recv_triples[:c_out_tile_size, tri_lo : tri_lo + _BN_TRIPLE_ELEMS],
            )
            nisa.bn_aggr(
                dst=aggr_out[:c_out_tile_size, 0:_BN_AGGR_ELEMS], data=pool_buf[:c_out_tile_size, 0:pair_elems]
            )
            # Core 0 alone writes HBM, as for momentum: an empty-shard core's pooled mean can be non-finite
            _write_final_stats(
                c_out_tile_start,
                c_out_tile_idx,
                c_out_tile_size,
                aggr_out,
                means_sbuf,
                variances_sbuf,
                means,
                variances,
                write_hbm=plan.prg_id == 0,
            )

    if plan.exchange:
        sbm.pop_heap()
    sbm.pop_heap()
    sbm.pop_heap()
    kernel_assert(
        len(sbm.heap) == entry_depth,
        f"conv3d finalize heap frame unbalanced: {len(sbm.heap)} != {entry_depth}",
    )


def _store_bn_triple(
    dst: nl.NkiTensor,
    tri_lo: int,
    count: int,
    aggr_out: nl.NkiTensor,
    part_lo: int,
    c_out_tile_size: int,
    identity_mask: list,
) -> None:
    """
    Write one (count, mean, variance * count) triple at dst[:, tri_lo : tri_lo + 3].

    Rebuilds exactly what a first-level nisa.bn_stats group produced, scaled by this contribution's
    element count, so the next nisa.bn_aggr pools it exactly. The (mean, variance) pair is read from
    aggr_out at partitions [part_lo, +M) — a column-tiling band's bn_aggr writes on its own
    partitions — and lands at [0, M) so one pool covers every contribution.
    """
    nisa.memset(dst=dst[:c_out_tile_size, tri_lo : tri_lo + 1], value=float(count))
    if part_lo == 0:
        nisa.tensor_copy(dst=dst[:c_out_tile_size, tri_lo + 1 : tri_lo + 2], src=aggr_out[:c_out_tile_size, 0:1])
        nisa.tensor_scalar(
            dst=dst[:c_out_tile_size, tri_lo + 2 : tri_lo + 3],
            data=aggr_out[:c_out_tile_size, 1:2],
            op0=nl.multiply,
            operand0=float(count),
        )
    else:
        nisa.nc_stream_shuffle(
            dst=dst[:c_out_tile_size, tri_lo + 1 : tri_lo + 3],
            src=aggr_out[part_lo : part_lo + c_out_tile_size, 0:_BN_AGGR_ELEMS],
            shuffle_mask=identity_mask,
        )
        nisa.tensor_scalar(
            dst=dst[:c_out_tile_size, tri_lo + 2 : tri_lo + 3],
            data=dst[:c_out_tile_size, tri_lo + 2 : tri_lo + 3],
            op0=nl.multiply,
            operand0=float(count),
        )


def _write_final_stats(
    c_out_tile_start: int,
    c_out_tile_idx: int,
    c_out_tile_size: int,
    aggr_out: nl.NkiTensor,
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    write_hbm: bool,
) -> None:
    """
    Split one nisa.bn_aggr result into the persistent per-channel mean / variance SBUF columns (read
    by the later phases without a reload) and, when write_hbm is set, write both to their HBM outputs.
    """
    c_out_tile_end = c_out_tile_start + c_out_tile_size
    mean_col = means_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
    var_col = variances_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
    nisa.tensor_copy(dst=mean_col, src=aggr_out[:c_out_tile_size, 0:1])
    nisa.tensor_copy(dst=var_col, src=aggr_out[:c_out_tile_size, 1:2])
    if write_hbm:
        nisa.dma_copy(
            dst=means.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1),
            src=mean_col[:c_out_tile_size, 0],
        )
        nisa.dma_copy(
            dst=variances.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1),
            src=var_col[:c_out_tile_size, 0],
        )


def _flush_column_layout(tile_cfg: "Conv3dTileConfig", num_dh_positions: int) -> tuple[int, int, int]:
    """
    Decide the column-tiling layout for one flush of num_dh_positions D-H positions. Returns
    (col_rep, band_dh, col_size): col_rep = active bands (1 disables tiling), band_dh = per-band
    D-H count, col_size = PE column-tile / partition-band stride (P_MAX when col_rep == 1).

    Tiling applies when the flush spans > dh_head_size positions; band_dh stays dh_head_size (the
    last band may be partial) so every eviction fits the per-band result width. A rep-1 fallback
    would size the eviction at num_dh_positions * W_tile and overflow the band-width buffer. Shared
    by _conv3d_output_tile and the store loop so they agree on the layout.
    """
    dh_head = tile_cfg.dh_head_size
    if tile_cfg.C_OUT_REP > 1 and dh_head > 0 and num_dh_positions > dh_head:
        col_rep = min(tile_cfg.C_OUT_REP, div_ceil(num_dh_positions, dh_head))
        return col_rep, dh_head, tile_cfg.col_size
    return 1, num_dh_positions, tile_cfg.P_MAX


def _column_bands(
    num_dh_positions: int, dh_head_size: int, C_OUT_REP: int, col_size: int
) -> list[tuple[int, int, int]]:
    """
    Enumerate the column-tiling bands as (partition_offset, dh_lo, dh_hi). Band band_idx covers dh
    [band_idx * dh_head_size, min((band_idx+1) * dh_head_size, num_dh_positions)) at partition offset
    band_idx * col_size. A tail flush may activate fewer than C_OUT_REP bands (last one partial);
    C_OUT_REP == 1 is a single band at partition 0 over all positions.
    """
    if C_OUT_REP <= 1:
        return [(0, 0, num_dh_positions)]
    bands: list[tuple[int, int, int]] = []
    for band_idx in range(C_OUT_REP):
        dh_lo = band_idx * dh_head_size
        if dh_lo >= num_dh_positions:
            break
        dh_hi = min(dh_lo + dh_head_size, num_dh_positions)
        bands.append((band_idx * col_size, dh_lo, dh_hi))
    return bands


def _bn_band_element_counts(cfg: "Conv3dConfig", tile_cfg: "Conv3dTileConfig") -> list[int]:
    """
    Per-channel element count that each column-tiling band's nisa.bn_stats groups cover, summed over
    every flush of this core's D-H range and all batches.

    Mirrors the main loop's grouping (see _dh_group_sizes and _column_bands): a flush packs up to
    num_dh_stacked consecutive D-H positions without crossing a d_out boundary, and band g takes the
    positions [g * dh_head, (g+1) * dh_head) of that flush. Every output position therefore belongs
    to exactly one (flush, band) pair, so these counts sum to B * (dh_end - dh_start) * W_out.
    Needed because bn_aggr pools by count, so the second-level pool must know each band's weight.
    """
    counts = []
    for _band in range(tile_cfg.C_OUT_REP):
        counts.append(0)

    group_sizes = _dh_group_sizes(tile_cfg.dh_start, tile_cfg.dh_end, tile_cfg.num_dh_stacked, cfg.H_out)
    for group_idx in range(len(group_sizes)):
        # Bands of this flush, then its W tiles (a band's width is its D-H count * the W tile size).
        group_size = group_sizes[group_idx]
        col_rep, band_dh, col_size = _flush_column_layout(tile_cfg, group_size)
        bands = _column_bands(group_size, band_dh, col_rep, col_size)
        for band_idx in range(len(bands)):
            dh_count = bands[band_idx][2] - bands[band_idx][1]
            counts[band_idx] += cfg.B * dh_count * cfg.W_out

    return counts


def _compute_true_bn_scales(
    mean_col: nl.NkiTensor,
    var_col: nl.NkiTensor,
    gamma_col: nl.NkiTensor,
    beta_col: nl.NkiTensor,
    true_gamma_buf: nl.NkiTensor,
    neg_true_beta_buf: nl.NkiTensor,
    batch_norm_eps: float,
    c_out_tile_size: int,
    C_OUT_REP: int,
    band_stride: int,
    identity_mask: list,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Precompute the per-channel batchnorm affine coefficients for one C_out tile from the
    SBUF-resident mean / variance columns,
        true_gamma    = gamma * rsqrt(variances + batch_norm_eps)
        neg_true_beta = means * true_gamma - beta
    so a single tensor_scalar applies the whole normalization,
        y = x * true_gamma - neg_true_beta
    (== gamma * (x - mean) / sqrt(var + eps) + beta). See conv3d_design_spec.md.

    The coefficients are computed on band 0 (partitions [0, c_out_tile_size)) and replicated
    into bands 1..C_OUT_REP-1 at partition offsets band_idx * band_stride via
    nc_stream_shuffle, so one tensor_scalar can cover all C_OUT_REP column-tiling bands at
    once. Returns the (true_gamma, neg_true_beta) operand views spanning all bands.
    """
    true_gamma_base = true_gamma_buf[:c_out_tile_size, 0:1]
    neg_true_beta_base = neg_true_beta_buf[:c_out_tile_size, 0:1]

    # true_gamma = gamma * rsqrt(variances + eps)
    nisa.activation(dst=true_gamma_base, op=nl.rsqrt, data=var_col, bias=batch_norm_eps)
    nisa.tensor_tensor(dst=true_gamma_base, data1=true_gamma_base, data2=gamma_col, op=nl.multiply)
    # neg_true_beta = means * true_gamma - beta
    nisa.tensor_scalar(
        dst=neg_true_beta_base,
        data=mean_col,
        op0=nl.multiply,
        operand0=true_gamma_base,
        op1=nl.subtract,
        operand1=beta_col,
    )

    # Replicate coefficients into bands >= 1 via nc_stream_shuffle (dst_start quadrant-aligned).
    for band_idx in range(1, C_OUT_REP):
        dst_lo = band_idx * band_stride
        nisa.nc_stream_shuffle(
            dst=true_gamma_buf[dst_lo : dst_lo + c_out_tile_size, 0:1],
            src=true_gamma_base,
            shuffle_mask=identity_mask,
        )
        nisa.nc_stream_shuffle(
            dst=neg_true_beta_buf[dst_lo : dst_lo + c_out_tile_size, 0:1],
            src=neg_true_beta_base,
            shuffle_mask=identity_mask,
        )

    # Coefficient views spanning all replicated bands (tensor_scalar operands).
    rep_active_part = (C_OUT_REP - 1) * band_stride + c_out_tile_size if C_OUT_REP > 1 else c_out_tile_size
    return true_gamma_buf[:rep_active_part, 0:1], neg_true_beta_buf[:rep_active_part, 0:1]


def _apply_bn_affine_and_residual(
    cfg: "Conv3dConfig",
    compute_tile: nl.NkiTensor,
    residual_tile: Optional[nl.NkiTensor],
    true_gamma: nl.NkiTensor,
    neg_true_beta: nl.NkiTensor,
    true_beta: nl.NkiTensor,
    active_part: int,
    residual_fold: bool,
) -> None:
    """Apply batchnorm, activation, and residual fusion to one tile."""
    if residual_fold and cfg.residual_pre_activation():
        # Keep the affine intact before the residual and activation
        nisa.activation(
            dst=compute_tile,
            data=compute_tile,
            op=nl.copy,
            scale=true_gamma[:active_part, 0:1],
            bias=true_beta[:active_part, 0:1],
        )
        nisa.tensor_tensor(dst=compute_tile, data1=compute_tile, data2=residual_tile, op=nl.add)
        nisa.activation(dst=compute_tile, data=compute_tile, op=get_nl_act_fn_from_type(cfg.activation_fn))
    elif residual_fold:
        # POST_ACT: ActFn(compute_tile * true_gamma + true_beta) applies the whole affine
        # and the activation in one fp32-internal instruction, then add the residual.
        nisa.activation(
            dst=compute_tile,
            data=compute_tile,
            op=get_nl_act_fn_from_type(cfg.activation_fn),
            scale=true_gamma[:active_part, 0:1],
            bias=true_beta[:active_part, 0:1],
        )
        nisa.tensor_tensor(dst=compute_tile, data1=compute_tile, data2=residual_tile, op=nl.add)
    else:
        nisa.tensor_scalar(
            dst=compute_tile,
            data=compute_tile,
            op0=nl.multiply,
            operand0=true_gamma[:active_part, 0:1],
            op1=nl.subtract,
            operand1=neg_true_beta[:active_part, 0:1],
        )
        # Activation (post-batchnorm) on the normalized tile.
        if cfg.has_activation:
            nisa.activation(dst=compute_tile, data=compute_tile, op=get_nl_act_fn_from_type(cfg.activation_fn))
        if residual_tile != None:
            nisa.tensor_tensor(dst=compute_tile, data1=compute_tile, data2=residual_tile, op=nl.add)


@dataclass
class _LowChanOutputRegion(nl.NKIObject):
    band_idx: int
    spatial_start: int
    spatial_end: int

    def size(self) -> int:
        return self.spatial_end - self.spatial_start


@dataclass
class _LowChanOutputChunk(nl.NKIObject):
    band_idx: int
    free_offset: int
    spatial_start: int
    spatial_end: int

    def size(self) -> int:
        return self.spatial_end - self.spatial_start


def _dma_low_chan_result_regions(
    hbm_tensor: nl.NkiTensor,
    result_sbuf: nl.NkiTensor,
    batch_idx: int,
    c_out_start: int,
    c_out_size: int,
    col_size: int,
    regions: list[_LowChanOutputRegion],
    to_hbm: bool,
) -> None:
    """Move one batch's result or residual bands between SBUF and HBM."""
    hbm_flat = (
        hbm_tensor.select(dim=0, index=batch_idx)
        .slice(dim=0, start=c_out_start, end=c_out_start + c_out_size, step=1)
        .flatten_dims(1, 3)
    )
    for region in regions:
        part_start = region.band_idx * col_size
        sbuf_view = result_sbuf[part_start : part_start + c_out_size, : region.size()]
        hbm_view = hbm_flat.slice(
            dim=1,
            start=region.spatial_start,
            end=region.spatial_end,
            step=1,
        )
        nisa.dma_copy(
            dst=hbm_view if to_hbm else sbuf_view,
            src=sbuf_view if to_hbm else hbm_view,
            dge_mode=nisa.dge_mode.hwdge,
        )


def _low_chan_post_group_extent(
    tile_group: list[_LowChanOutputChunk],
    c_out_size: int,
    col_size: int,
) -> tuple[int, int, int]:
    max_chunk_size = 0
    active_partitions = 0
    for chunk in tile_group:
        max_chunk_size = max(max_chunk_size, chunk.size())
        active_partitions = max(active_partitions, chunk.band_idx * col_size + c_out_size)
    return max_chunk_size, active_partitions, tile_group[0].free_offset


def apply_bn_rescale(
    cfg: "Conv3dConfig",
    conv_out: nl.NkiTensor,
    y_out: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    batch_norm_eps: float,
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    post_config: _BatchNormPostProcessConfig,
    post_buffers: _BatchNormPostProcessBuffers,
    identity_mask: list,
    residuals_in: Optional[nl.NkiTensor] = None,
) -> None:
    """
    Summary:
        Normalize this program's output shard and apply optional activation and residual.

    Args:
        cfg (Conv3dConfig): Convolution geometry and fused post-processing configuration.
        conv_out (nl.NkiTensor): [B, C_out, D_out, H_out, W_out], raw convolution
            output on HBM.
        y_out (nl.NkiTensor): [B, C_out, D_out, H_out, W_out], normalized output on HBM.
        gamma (nl.NkiTensor): [C_out, 1], batch-normalization scale on HBM.
        beta (nl.NkiTensor): [C_out, 1], batch-normalization bias on HBM.
        batch_norm_eps (float): Epsilon added to the variance before normalization.
        means_sbuf (nl.NkiTensor): Finalized channel means in SBUF columns.
        variances_sbuf (nl.NkiTensor): Finalized channel variances in SBUF columns.
        post_config (_BatchNormPostProcessConfig): Program-owned channel, batch, and
            spatial ranges plus reload layout.
        post_buffers (_BatchNormPostProcessBuffers): SBUF affine, reload, residual,
            and running-stat scratch buffers.
        identity_mask (list): Identity shuffle mask for replicating affine coefficients.
        residuals_in (Optional[nl.NkiTensor]): [B, C_out, D_out, H_out, W_out],
            optional residual input on HBM.

    Returns:
        None: Writes the normalized shard to ``y_out``.

    Notes:
        - PRE_ACT keeps the full affine before bf16 rounding, then adds and activates.
          POST_ACT fuses the affine with activation before adding. Without activation,
          both locations match.
        - Low-channel layouts use region-based reloads; the fallback path reloads
          contiguous spatial chunks and rotates through the configured SBUF slots.
    """
    gamma_buf = post_buffers.gamma_buf
    beta_buf = post_buffers.beta_buf
    true_gamma_buf = post_buffers.true_gamma_buf
    neg_true_beta_buf = post_buffers.neg_true_beta_buf
    reload_buf = post_buffers.reload_buf
    residual_reload_buf = post_buffers.residual_reload_buf

    has_residual = residual_reload_buf != None
    residual_fold = has_residual and cfg.has_activation

    for c_out_tile_start in range(post_config.c_out_start, post_config.c_out_end, post_config.p_max):
        c_out_tile_idx = c_out_tile_start // post_config.p_max
        c_out_tile_end = min(c_out_tile_start + post_config.p_max, post_config.c_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        if post_config.fixed_c_out_rep == None:
            c_out_rep, band_stride = _get_c_out_replication_params(c_out_tile_size)
        else:
            c_out_rep = post_config.fixed_c_out_rep
            band_stride = c_out_tile_size

        # Stats read from SBUF columns (no reload); affine computed on band 0, replicated below.
        mean_col = means_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        var_col = variances_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        gamma_col = gamma_buf[:c_out_tile_size, 0:1]
        beta_col = beta_buf[:c_out_tile_size, 0:1]

        nisa.dma_copy(dst=gamma_col, src=gamma.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
        nisa.dma_copy(dst=beta_col, src=beta.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))

        # Affine coefficients, replicated across the C_OUT_REP reload_buf bands (band g at
        # [g * c_out_tile_size, +)), so one tensor_scalar rescales every band.
        true_gamma, neg_true_beta = _compute_true_bn_scales(
            mean_col=mean_col,
            var_col=var_col,
            gamma_col=gamma_col,
            beta_col=beta_col,
            true_gamma_buf=true_gamma_buf,
            neg_true_beta_buf=neg_true_beta_buf,
            batch_norm_eps=batch_norm_eps,
            c_out_tile_size=c_out_tile_size,
            C_OUT_REP=c_out_rep,
            band_stride=band_stride,
            identity_mask=identity_mask,
        )

        # The folds below add true_beta rather than subtracting neg_true_beta, so flip the sign in place
        # (no extra buffer). Hoisted: once per C_out tile, not per reloaded tile.
        true_beta = None
        if residual_fold:
            nisa.tensor_scalar(dst=neg_true_beta, data=neg_true_beta, op0=nl.multiply, operand0=-1.0)
            true_beta = neg_true_beta

        if post_config.low_chan_tile_layouts != None:
            tile_layouts = post_config.low_chan_tile_layouts[c_out_tile_idx]
            for batch_idx in range(post_config.batch_start, post_config.batch_end):
                for regions, tile_groups in tile_layouts:
                    _dma_low_chan_result_regions(
                        hbm_tensor=conv_out,
                        result_sbuf=reload_buf,
                        batch_idx=batch_idx,
                        c_out_start=c_out_tile_start,
                        c_out_size=c_out_tile_size,
                        col_size=band_stride,
                        regions=regions,
                        to_hbm=False,
                    )
                    if has_residual:
                        _dma_low_chan_result_regions(
                            hbm_tensor=residuals_in,
                            result_sbuf=residual_reload_buf,
                            batch_idx=batch_idx,
                            c_out_start=c_out_tile_start,
                            c_out_size=c_out_tile_size,
                            col_size=band_stride,
                            regions=regions,
                            to_hbm=False,
                        )

                    for tile_group in tile_groups:
                        max_chunk_size, active_partitions, free_offset = _low_chan_post_group_extent(
                            tile_group, c_out_tile_size, band_stride
                        )
                        compute_tile = reload_buf[
                            :active_partitions,
                            free_offset : free_offset + max_chunk_size,
                        ]
                        residual_tile = None
                        if has_residual:
                            residual_tile = residual_reload_buf[
                                :active_partitions,
                                free_offset : free_offset + max_chunk_size,
                            ]
                        _apply_bn_affine_and_residual(
                            cfg=cfg,
                            compute_tile=compute_tile,
                            residual_tile=residual_tile,
                            true_gamma=true_gamma,
                            neg_true_beta=neg_true_beta,
                            true_beta=true_beta,
                            active_part=active_partitions,
                            residual_fold=residual_fold,
                        )

                    _dma_low_chan_result_regions(
                        hbm_tensor=y_out,
                        result_sbuf=reload_buf,
                        batch_idx=batch_idx,
                        c_out_start=c_out_tile_start,
                        c_out_size=c_out_tile_size,
                        col_size=band_stride,
                        regions=regions,
                        to_hbm=True,
                    )
            continue

        # Reload conv_out and rescale into y_out. Column tiling handles C_OUT_REP spatial chunks per
        # iteration (rep_step), rep == 1 one tile; multibuffering rotates reload_buf for overlap.
        reload_slot_counter = 0
        rep_step = c_out_rep * post_config.reload_free_tile
        # Per-chunk plan (band_extents, max_sp_size) is batch-independent, so build it once.
        # band_extents = (band_idx, sp_start, sp_end, sp_size) per active band (tail bands dropped).
        chunk_plan = []
        for chunk_start in range(post_config.spatial_start, post_config.spatial_end, rep_step):
            band_extents = []
            max_sp_size = 0
            for band_idx in range(c_out_rep):
                band_sp_start = chunk_start + band_idx * post_config.reload_free_tile
                if band_sp_start >= post_config.spatial_end:
                    break
                band_sp_end = min(band_sp_start + post_config.reload_free_tile, post_config.spatial_end)
                band_sp_size = band_sp_end - band_sp_start
                band_extents.append((band_idx, band_sp_start, band_sp_end, band_sp_size))
                if band_sp_size > max_sp_size:
                    max_sp_size = band_sp_size
            chunk_plan.append((band_extents, max_sp_size))

        for batch_idx in range(post_config.batch_start, post_config.batch_end):
            conv_out_flat = (
                conv_out.select(dim=0, index=batch_idx)
                .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                .flatten_dims(1, 3)
            )
            y_out_flat = (
                y_out.select(dim=0, index=batch_idx)
                .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                .flatten_dims(1, 3)
            )
            residual_flat = None
            if has_residual:
                residual_flat = (
                    residuals_in.select(dim=0, index=batch_idx)
                    .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                    .flatten_dims(1, 3)
                )
            for plan_entry in chunk_plan:
                band_extents = plan_entry[0]
                max_sp_size = plan_entry[1]
                slot_idx = reload_slot_counter % post_config.reload_interleave
                slot_start = slot_idx * post_config.reload_free_tile

                # DMA each active band's spatial slice into its partition band.
                for band_ext in band_extents:
                    band_idx = band_ext[0]
                    band_sp_start = band_ext[1]
                    band_sp_end = band_ext[2]
                    band_sp_size = band_ext[3]
                    band_part_lo = band_idx * band_stride
                    conv_out_view = conv_out_flat.slice(dim=1, start=band_sp_start, end=band_sp_end, step=1)
                    band_load = reload_buf[
                        band_part_lo : band_part_lo + c_out_tile_size,
                        slot_start : slot_start + band_sp_size,
                    ]
                    nisa.dma_copy(dst=band_load, src=conv_out_view)
                    if has_residual:
                        residual_view = residual_flat.slice(dim=1, start=band_sp_start, end=band_sp_end, step=1)
                        nisa.dma_copy(
                            dst=residual_reload_buf[
                                band_part_lo : band_part_lo + c_out_tile_size,
                                slot_start : slot_start + band_sp_size,
                            ],
                            src=residual_view,
                        )

                # One tensor_scalar rescales all active bands with the replicated coeffs. A short
                # band's [band_sp_size:max_sp_size) tail is garbage but never stored.
                active_bands = len(band_extents)
                active_part = (active_bands - 1) * band_stride + c_out_tile_size
                compute_tile = reload_buf[:active_part, slot_start : slot_start + max_sp_size]
                residual_tile = None
                if has_residual:
                    residual_tile = residual_reload_buf[:active_part, slot_start : slot_start + max_sp_size]

                _apply_bn_affine_and_residual(
                    cfg=cfg,
                    compute_tile=compute_tile,
                    residual_tile=residual_tile,
                    true_gamma=true_gamma,
                    neg_true_beta=neg_true_beta,
                    true_beta=true_beta,
                    active_part=active_part,
                    residual_fold=residual_fold,
                )

                # DMA each active band's rescaled slice back to its y_out region.
                for band_ext in band_extents:
                    band_idx = band_ext[0]
                    band_sp_start = band_ext[1]
                    band_sp_end = band_ext[2]
                    band_sp_size = band_ext[3]
                    band_part_lo = band_idx * band_stride
                    y_out_view = y_out_flat.slice(dim=1, start=band_sp_start, end=band_sp_end, step=1)
                    band_store = reload_buf[
                        band_part_lo : band_part_lo + c_out_tile_size,
                        slot_start : slot_start + band_sp_size,
                    ]
                    nisa.dma_copy(dst=y_out_view, src=band_store)
                reload_slot_counter += 1


def compute_bn_momentum_update(
    cfg: "Conv3dConfig",
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    running_means: nl.NkiTensor,
    running_variances: nl.NkiTensor,
    updated_running_means: nl.NkiTensor,
    updated_running_variances: nl.NkiTensor,
    run_running_mean_buf: nl.NkiTensor,
    run_new_mean_buf: nl.NkiTensor,
    run_running_var_buf: nl.NkiTensor,
    run_new_var_buf: nl.NkiTensor,
    momentum: float,
    p_max: int,
    c_out_start: int,
    c_out_end: int,
) -> None:
    """
    Summary:
        Momentum-update phase (matches torch.nn.BatchNorm2d), sharded by C_out slice
        so each core only touches the stats it produced.

    Args:
        cfg (Conv3dConfig): Convolution geometry used to calculate the sample count N.
        means_sbuf (nl.NkiTensor): Finalized channel means in SBUF columns.
        variances_sbuf (nl.NkiTensor): Finalized channel variances in SBUF columns.
        running_means (nl.NkiTensor): [C_out, 1], existing running means on HBM.
        running_variances (nl.NkiTensor): [C_out, 1], existing running variances on HBM.
        updated_running_means (nl.NkiTensor): [C_out, 1], destination running means on HBM.
        updated_running_variances (nl.NkiTensor): [C_out, 1], destination running
            variances on HBM.
        run_running_mean_buf (nl.NkiTensor): SBUF scratch column for existing means.
        run_new_mean_buf (nl.NkiTensor): SBUF scratch column for new means.
        run_running_var_buf (nl.NkiTensor): SBUF scratch column for existing variances.
        run_new_var_buf (nl.NkiTensor): SBUF scratch column for new variances.
        momentum (float): Weight applied to the newly computed statistics.
        p_max (int): Maximum channel count processed per tile.
        c_out_start (int): Inclusive start of this program's output-channel slice.
        c_out_end (int): Exclusive end of this program's output-channel slice.

    Returns:
        None: Writes momentum-updated statistics to the two destination tensors.

    Notes:
        - Momentum equations:
            updated_running_means     = (1 - momentum) * running_means     + momentum * means
            updated_running_variances = (1 - momentum) * running_variances + momentum * (N/(N-1)) * var
        - The running variance uses the correction-1 (unbiased) variance. Reads the fresh mean /
          variance from the SBUF columns (no HBM reload).
    """
    one_minus_momentum = 1.0 - momentum
    N = cfg.B * cfg.D_out * cfg.H_out * cfg.W_out
    correct1_factor = N / (N - 1) if N > 1 else 1.0

    for c_out_tile_start in range(c_out_start, c_out_end, p_max):
        c_out_tile_idx = c_out_tile_start // p_max
        c_out_tile_end = min(c_out_tile_start + p_max, c_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        running_mean_col = run_running_mean_buf[:c_out_tile_size, 0:1]
        new_mean_col = run_new_mean_buf[:c_out_tile_size, 0:1]
        running_var_col = run_running_var_buf[:c_out_tile_size, 0:1]
        new_var_col = run_new_var_buf[:c_out_tile_size, 0:1]
        # Fresh stats read from the SBUF columns (no HBM reload).
        mean_sbuf_col = means_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        var_sbuf_col = variances_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]

        # Mean: new_scale = 1; variance: correction-1 factor. Disjoint scratch avoids serializing.
        _momentum_update_stat(
            running_hbm=running_means,
            new_stat_col=mean_sbuf_col,
            updated_hbm=updated_running_means,
            running_scratch=running_mean_col,
            new_scratch=new_mean_col,
            c_out_tile_start=c_out_tile_start,
            c_out_tile_end=c_out_tile_end,
            one_minus_momentum=one_minus_momentum,
            momentum=momentum,
            new_scale=1.0,
        )
        _momentum_update_stat(
            running_hbm=running_variances,
            new_stat_col=var_sbuf_col,
            updated_hbm=updated_running_variances,
            running_scratch=running_var_col,
            new_scratch=new_var_col,
            c_out_tile_start=c_out_tile_start,
            c_out_tile_end=c_out_tile_end,
            one_minus_momentum=one_minus_momentum,
            momentum=momentum,
            new_scale=correct1_factor,
        )


def _heap_align_waste(bytes_per_partition: int) -> int:
    """Compute alignment waste for a single SBM heap allocation."""
    return (_HEAP_ALIGN - (bytes_per_partition % _HEAP_ALIGN)) % _HEAP_ALIGN


def _heap_alloc_cost(bytes_per_partition: int) -> int:
    return bytes_per_partition + _heap_align_waste(bytes_per_partition)


def _decompose_k_position(k_position: int, K_h: int, K_w: int) -> tuple[int, int, int]:
    """Decompose a flat filter position index into (k_d, k_h, k_w) indices."""
    k_d_idx, k_hw_remainder = divmod(k_position, K_h * K_w)
    k_h_idx, k_w_idx = divmod(k_hw_remainder, K_w)
    return k_d_idx, k_h_idx, k_w_idx


def _compute_used_k_positions(cfg: "Conv3dConfig") -> list[int]:
    """
    Compute which flat filter positions are actually used.

    For each filter position (k_d, k_h, k_w), checks whether there exists any
    output position where the corresponding input position falls within valid
    (non-padded) bounds.

    Args:
        cfg: Conv3dConfig with all convolution parameters.

    Returns:
        List of flat filter position indices (into K_d*K_h*K_w) that are actually used.
    """
    K_d, K_h, K_w = cfg.K_d, cfg.K_h, cfg.K_w

    used_k_d = [False] * K_d
    for k_d in range(K_d):
        for d_out in range(cfg.D_out):
            d_in = d_out * cfg.stride_d + k_d * cfg.dilation_d - cfg.pad_d_left
            if 0 <= d_in < cfg.D:
                used_k_d[k_d] = True
                break

    used_k_h = [False] * K_h
    for k_h in range(K_h):
        for h_out in range(cfg.H_out):
            h_in = h_out * cfg.stride_h + k_h * cfg.dilation_h - cfg.pad_h_top
            if 0 <= h_in < cfg.H:
                used_k_h[k_h] = True
                break

    used_k_w = [False] * K_w
    for k_w in range(K_w):
        for w_out in range(cfg.W_out):
            w_in = w_out * cfg.stride_w + k_w * cfg.dilation_w - cfg.pad_w_left
            if 0 <= w_in < cfg.W:
                used_k_w[k_w] = True
                break

    used_positions = []
    for k_d in range(K_d):
        for k_h in range(K_h):
            for k_w in range(K_w):
                if used_k_d[k_d] and used_k_h[k_h] and used_k_w[k_w]:
                    flat_idx = k_d * K_h * K_w + k_h * K_w + k_w
                    used_positions.append(flat_idx)

    return used_positions


@dataclass
class Conv3dConfig(nl.NKIObject):
    """Configuration for 3D convolution kernel parameters and input/output dimensions."""

    stride_d: int
    stride_h: int
    stride_w: int
    pad_d_left: int
    pad_d_right: int
    pad_h_top: int
    pad_h_bottom: int
    pad_w_left: int
    pad_w_right: int
    dilation_d: int
    dilation_h: int
    dilation_w: int
    has_bias: bool
    has_activation: bool
    activation_fn: Optional[ActFnType]
    batch_norm_mode: BatchNormMode
    # Also return the raw pre-batchnorm convolution output; only ever True in TRAINING mode, the
    # only mode that materializes that tensor.
    output_pre_norm: bool
    residual_add_loc: ResidualAddLoc
    B: int
    C_in: int
    C_out: int
    D: int
    H: int
    W: int
    K_d: int
    K_h: int
    K_w: int
    D_out: int
    H_out: int
    W_out: int

    def has_batch_norm(self) -> bool:
        """Whether any fused batchnorm runs."""
        return self.batch_norm_mode.is_fused()

    def is_batch_norm_eval(self) -> bool:
        """Whether the batchnorm normalizes on the PSUM eviction from the given running stats."""
        return self.batch_norm_mode == BatchNormMode.EVAL

    def is_batch_norm_training(self) -> bool:
        """Whether the statistics-accumulation + finalize (rescale) pass runs."""
        return self.batch_norm_mode == BatchNormMode.TRAINING

    def has_residual_add(self) -> bool:
        """Whether a residual add is fused into the output post-processing."""
        return self.residual_add_loc.is_fused()

    def residual_in_main_loop(self) -> bool:
        """
        Whether the residual is staged in the main loop's result-shaped SBUF slots.

        True in every mode but TRAINING, which adds the residual in the rescale phase instead (the
        main loop's eviction there produces the raw pre-batchnorm conv output).
        """
        return self.has_residual_add() and not self.is_batch_norm_training()

    def residual_pre_activation(self) -> bool:
        """Whether the residual is added before the activation function."""
        return self.residual_add_loc == ResidualAddLoc.PRE_ACT

    def residual_in_place(self) -> bool:
        """
        Whether the staged residual can share the result tile instead of needing its own SBUF slot.

        True only when the residual is consumed by the very first op of the eviction — the one that
        reads PSUM — so the op's own write is what overwrites the staged value:
            result = psum + residual            (tensor_tensor)
            result = (psum + bias) + residual   (scalar_tensor_tensor)
        That is the BatchNormMode.NONE eviction with either no activation, or a PRE_ACT residual
        (which the same op consumes before the activation runs). Every other form writes the result
        tile before reading the residual — the batchnorm affine, and POST_ACT with an activation,
        both evict into the tile first — and would read back its own output as the residual, so those
        keep a separate staging tile. Halves the hot-path result footprint when it holds.
        """
        # has_batch_norm() implies the first conjunct's not-TRAINING, so residual_in_main_loop() would
        # be redundant here.
        return (
            self.has_residual_add()
            and not self.has_batch_norm()
            and (not self.has_activation or self.residual_pre_activation())
        )

    def needs_residual_staging_grid(self) -> bool:
        """
        Whether the main loop allocates a second, result-shaped SBUF slot grid to stage the residual.

        Single source of truth for the three sites that must agree — the allocation, the matching
        pop_heap unwind, and _build_memory_config's result_bufs_per_slot budget. A drift between the
        budget and the unwind is silent SBUF corruption rather than an error, so they read one name.
        """
        return self.residual_in_main_loop() and not self.residual_in_place()


@dataclass
class Conv3dTileConfig(nl.NKIObject):
    """Tiling configuration including partition sizes, tile counts, and sharding ranges."""

    P_MAX: int
    F_MAX: int
    PSUM_BANK_SIZE: int
    NUM_PSUM_BANKS: int
    W_tile: int
    c_out_tile_size_max: int
    c_in_tile_size_max: int
    c_in_tile_count: int
    c_out_tile_count: int
    K_outer_tile_count: int
    K_REP_max: int
    partition_stride_max: int
    stacked_filter_dim_max: int
    C_out_start: int
    C_out_end: int
    C_out_local: int
    num_dh_stacked: int
    # Column tiling: C_OUT_REP bands (1 disables), PE column-tile size col_size, per-band D-H count
    # dh_head_size. Band g runs on column tile / PSUM band g * col_size. See conv3d_design_spec.md.
    C_OUT_REP: int
    col_size: int
    dh_head_size: int
    d_window_max: int
    h_window_max: int
    w_window_max: int
    shard_on_dh: bool
    dh_start: int
    dh_end: int
    used_k_positions: list[int]
    K_total_used: int
    h_chunk_size: int
    # Eviction flushes per C_out tile (B * num_dh_groups * num_w_tiles); sizes bn_stats_bufs, which
    # holds one _BN_STATS_ELEMS-wide nisa.bn_stats group per flush.
    bn_partial_iters: int
    # TODO: Determine optimal engine offload values using a cost model.
    tensor_copy_activation_engine_offload: int
    memset_gpsimd_engine_offload: int


@dataclass
class Conv3dMemoryConfig(nl.NKIObject):
    """Memory layout configuration for SBUF buffer allocation and interleaving."""

    dtype_size: int
    TOTAL_SBUF: int
    bias_size_per_tile: int
    filters_per_c_out_tile: int
    per_c_out_tile_outer: int
    per_c_out_tile_inner: int
    input_window_memory: int
    stacked_input_memory: int
    c_out_interleave: int
    input_window_interleave: int
    stacked_input_interleave: int
    w_out_interleave: int
    num_w_out_tiles: int
    total_memory_required: int
    # Store-batch merging (column tiling): store_batch_merge batches per band DMA, store_pipe slots
    # overlap store with compute; product == w_out_interleave (== 1 disables). See _build_memory_config.
    store_batch_merge: int
    store_pipe: int
    # Result-shaped grids per (w_out, c_out) slot: 2 when a fused residual needs its own staging grid,
    # else 1. The fit search budgets and the main loop allocates/pops this many, so they cannot drift.
    result_bufs_per_slot: int
    # Maximum batchnorm reload interleave; the fallback fits the actual value
    bn_reload_interleave: int


def _build_conv3d_config(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int],
    dilation: tuple[int, int, int],
    activation_fn: Optional[ActFnType],
    batch_norm_mode: BatchNormMode,
    output_pre_norm: bool,
    residual_add_loc: ResidualAddLoc,
    transpose_taps: bool = False,
) -> Conv3dConfig:
    """
    Build Conv3dConfig from input tensors and convolution parameters.

    Extracts dimensions from input/filter shapes and computes output spatial
    dimensions based on stride, padding, and dilation settings.
    """
    # With no activation function the two residual insertion points are equivalent, so PRE_ACT is
    # normalized to POST_ACT here and only one code path has to handle it.
    normalized_residual_loc = residual_add_loc
    if residual_add_loc == ResidualAddLoc.PRE_ACT and activation_fn == None:
        normalized_residual_loc = ResidualAddLoc.POST_ACT
    stride_d, stride_h, stride_w = stride
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dilation_d, dilation_h, dilation_w = dilation
    B, C_in, D, H, W = x_in.shape
    K_d, K_h, K_w, filter_c_in, filter_c_out = filters.shape
    C_out = filter_c_in if transpose_taps else filter_c_out
    D_out = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1
    return Conv3dConfig(
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_d_left=pad_d_left,
        pad_d_right=pad_d_right,
        pad_h_top=pad_h_top,
        pad_h_bottom=pad_h_bottom,
        pad_w_left=pad_w_left,
        pad_w_right=pad_w_right,
        dilation_d=dilation_d,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        has_bias=bias != None,
        has_activation=activation_fn != None,
        activation_fn=activation_fn,
        batch_norm_mode=batch_norm_mode,
        output_pre_norm=output_pre_norm == True,
        residual_add_loc=normalized_residual_loc,
        B=B,
        C_in=C_in,
        C_out=C_out,
        D=D,
        H=H,
        W=W,
        K_d=K_d,
        K_h=K_h,
        K_w=K_w,
        D_out=D_out,
        H_out=H_out,
        W_out=W_out,
    )


def _get_k_replication_params(c_in_tile_size: int, K_total: int) -> tuple[int, int]:
    """Determine K-replication count and partition stride based on input channel tile size."""
    if c_in_tile_size <= _PARTITION_STRIDE_32:
        return min(K_total, _MAX_K_REP_AT_32), _PARTITION_STRIDE_32
    elif c_in_tile_size <= _PARTITION_STRIDE_64:
        return min(K_total, _MAX_K_REP_AT_64), _PARTITION_STRIDE_64
    else:
        return 1, c_in_tile_size


def _get_c_out_replication_params(c_out_tile_size: int) -> tuple[int, int]:
    """
    Determine the C_out column-tiling replication factor and PE column-tile size.

    When the C_out tile size is smaller than the 128 PE columns, the nc_matmul
    stationary free dim (= C_out) occupies only that many columns and leaves the rest
    of the systolic array idle. Column tiling packs C_OUT_REP D-H groups into a single
    output flush, each running its matmuls on a different PE column tile (start column
    g * col_size) and landing in a different partition band of one wide PSUM bank. Each
    group keeps the full baseline free dim, so the total matmul count and per-matmul free
    dim are unchanged relative to processing the groups one-flush-at-a-time; the win is
    parallel PE-column occupancy and a single wide eviction.

    C_OUT_REP is bounded by the column-tile size that fits C_out (4x32 or 2x64 columns)
    and the hardware (4x32 needs NeuronCore-v3+; 2x64 needs v2+).

    Returns (C_OUT_REP, col_size). C_OUT_REP == 1 means no column tiling (col_size 128).
    """
    nc_ver = nisa.get_nc_version()
    if c_out_tile_size <= _COLUMN_TILE_32 and nc_ver >= nisa.nc_version.gen3:
        return _MAX_K_REP_AT_32, _COLUMN_TILE_32
    elif c_out_tile_size <= _COLUMN_TILE_64:
        return _MAX_K_REP_AT_64, _COLUMN_TILE_64
    else:
        return 1, nl.tile_size.pmax


def _dh_group_sizes(dh_start: int, dh_end: int, num_dh_stacked: int, H_out: int) -> list[int]:
    """
    D-H output group sizes the main loop iterates over for a given dh range.

    Single source of truth for the grouping in the conv3d loop nest: starting at dh_start, each
    group packs up to num_dh_stacked consecutive flat (d_out, h_out) positions but never crosses a
    d_out boundary. Callers need either the count (to size the batchnorm partials buffer) or each
    group's size (to weight the batchnorm aggregation), so both come from here.
    """
    sizes = []
    dh_group_idx = dh_start
    while dh_group_idx < dh_end:
        first_d_out = None
        group_size = 0
        for dh_stacked_idx in range(num_dh_stacked):
            flat_idx = dh_group_idx + dh_stacked_idx
            if flat_idx >= dh_end:
                break
            d_out_idx = flat_idx // H_out
            if first_d_out == None:
                first_d_out = d_out_idx
            elif d_out_idx != first_d_out:
                break
            group_size += 1
        sizes.append(group_size)
        dh_group_idx += group_size
    return sizes


def _count_dh_groups(dh_start: int, dh_end: int, num_dh_stacked: int, H_out: int) -> int:
    """Number of D-H output groups the main loop iterates over. See _dh_group_sizes."""
    return len(_dh_group_sizes(dh_start, dh_end, num_dh_stacked, H_out))


def _calculate_union_window_size(cfg: Conv3dConfig, num_dh_stacked: int, w_tile: int) -> tuple[int, int, int]:
    """Calculate the union input window size for a group of stacked D-H output positions."""
    single_d_window = (cfg.K_d - 1) * cfg.dilation_d + 1
    single_h_window = (cfg.K_h - 1) * cfg.dilation_h + 1
    w_window = (w_tile - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w + 1
    if num_dh_stacked == 1:
        return single_d_window, single_h_window, w_window
    max_d_out_in_group = (num_dh_stacked - 1 + cfg.H_out - 1) // cfg.H_out
    d_window_max = min(single_d_window + max_d_out_in_group * cfg.stride_d, cfg.D)
    max_h_out_in_group = min(num_dh_stacked - 1, cfg.H_out - 1)
    h_window_max = min(single_h_window + max_h_out_in_group * cfg.stride_h, cfg.H)
    return d_window_max, h_window_max, w_window


def _build_tile_config(
    cfg: Conv3dConfig,
    dtype_size: int,
    total_sbuf_budget: Optional[int] = None,
) -> Conv3dTileConfig:
    """Build tiling configuration with LNC sharding and iterative memory fitting."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    TOTAL_SBUF = total_sbuf_budget if total_sbuf_budget != None else nl.tile_size.total_available_sbuf_size
    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d", (0, 1))

    full_c_out_tile_count = div_ceil(cfg.C_out, P_MAX)
    total_dh_positions = cfg.D_out * cfg.H_out

    dh_work_per_nc = div_ceil(total_dh_positions, 2)
    cout_tiles_per_nc = div_ceil(full_c_out_tile_count, 2)
    dh_waste = 2 * dh_work_per_nc - total_dh_positions
    cout_waste = 2 * cout_tiles_per_nc - full_c_out_tile_count
    shard_on_dh = (total_dh_positions >= 2) and (dh_waste <= cout_waste)

    if n_prgs > 1 and shard_on_dh:
        C_out_start, C_out_end, C_out_local = 0, cfg.C_out, cfg.C_out
        dh_per_nc = div_ceil(total_dh_positions, n_prgs)
        dh_start = dh_per_nc * prg_id
        dh_end = min(dh_start + dh_per_nc, total_dh_positions)
    elif n_prgs > 1:
        C_out_per_nc = div_ceil(cfg.C_out, n_prgs)
        C_out_start = C_out_per_nc * prg_id
        C_out_end = min(C_out_start + C_out_per_nc, cfg.C_out)
        C_out_local = C_out_end - C_out_start
        dh_start, dh_end = 0, total_dh_positions
    else:
        C_out_start, C_out_end, C_out_local = 0, cfg.C_out, cfg.C_out
        dh_start, dh_end = 0, total_dh_positions
        shard_on_dh = False

    c_out_tile_size_max = min(P_MAX, C_out_local)
    c_in_tile_size_max = min(P_MAX, cfg.C_in)
    used_k_positions = _compute_used_k_positions(cfg)
    K_total_used = len(used_k_positions)
    K_REP_max, partition_stride_max = _get_k_replication_params(c_in_tile_size_max, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP_max)
    stacked_filter_dim_max = partition_stride_max * K_REP_max
    c_in_tile_count = div_ceil(cfg.C_in, P_MAX)
    c_out_tile_count = div_ceil(C_out_local, P_MAX)

    # C_out column tiling (low C_out): pack C_OUT_REP D-H bands (dh_head each, num_dh_stacked =
    # C_OUT_REP * dh_head) onto parallel PE columns in one flush. See conv3d_design_spec.md.
    C_OUT_REP, col_size = _get_c_out_replication_params(c_out_tile_size_max)

    W_tile = min(cfg.W_out, F_MAX)
    dh_head = min(F_MAX // cfg.W_out, total_dh_positions) if cfg.W_out < F_MAX and total_dh_positions > 1 else 1

    while True:
        # Stack C_OUT_REP dh_head-wide bands, capped so the group doesn't exceed the total D-H work.
        c_out_rep_eff = min(C_OUT_REP, max(1, total_dh_positions // dh_head)) if C_OUT_REP > 1 else 1
        num_dh_stacked = c_out_rep_eff * dh_head
        per_band_free = dh_head * W_tile
        effective_free_dim = num_dh_stacked * W_tile
        d_window_max, h_window_max, w_window_max = _calculate_union_window_size(cfg, num_dh_stacked, W_tile)
        bias_mem = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
        filter_mem = c_in_tile_count * K_outer_tile_count * c_out_tile_size_max * dtype_size
        # PSUM/result are per_band_free wide (bands stacked on partitions); stacked input / input
        # window span the whole packed free dim.
        min_mem = (
            bias_mem
            + filter_mem
            + per_band_free * dtype_size
            + d_window_max * h_window_max * w_window_max * dtype_size
            + K_outer_tile_count * effective_free_dim * dtype_size
        )
        if min_mem <= TOTAL_SBUF:
            break
        if dh_head > 1:
            dh_head = max(1, dh_head // 2)
        elif W_tile > 1:
            W_tile = max(1, W_tile // 2)
        else:
            break

    c_out_rep_eff = min(C_OUT_REP, max(1, total_dh_positions // dh_head)) if C_OUT_REP > 1 else 1
    C_OUT_REP = c_out_rep_eff
    if C_OUT_REP == 1:
        col_size = P_MAX
    num_dh_stacked = C_OUT_REP * dh_head
    dh_head_size = dh_head
    d_window_max, h_window_max, w_window_max = _calculate_union_window_size(cfg, num_dh_stacked, W_tile)

    # H-chunk size: input reuse across consecutive h_out positions.
    single_h_window = (cfg.K_h - 1) * cfg.dilation_h + 1
    num_w_tiles = div_ceil(cfg.W_out, W_tile)
    if num_dh_stacked == 1 and cfg.K_h > 1 and cfg.H_out > 1 and num_w_tiles == 1:
        h_chunk_size = cfg.K_h

        h_chunk_h_window = single_h_window + (h_chunk_size - 1) * cfg.stride_h
        h_chunk_h_window = min(h_chunk_h_window, cfg.H)

        effective_free_dim_check = num_dh_stacked * W_tile
        while h_chunk_size > 1:
            h_chunk_h_window = min(single_h_window + (h_chunk_size - 1) * cfg.stride_h, cfg.H)
            chunk_input_window_mem = d_window_max * h_chunk_h_window * w_window_max * dtype_size
            bias_mem_check = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
            filter_mem_check = c_in_tile_count * K_outer_tile_count * c_out_tile_size_max * dtype_size

            min_mem_check = (
                bias_mem_check
                + filter_mem_check
                + effective_free_dim_check * dtype_size
                + chunk_input_window_mem
                + K_outer_tile_count * effective_free_dim_check * dtype_size
            )
            if min_mem_check <= TOTAL_SBUF:
                break
            h_chunk_size -= 1
        h_chunk_h_window_max = min(single_h_window + (h_chunk_size - 1) * cfg.stride_h, cfg.H)
    else:
        h_chunk_size = 1
        h_chunk_h_window_max = h_window_max

    # Flushes per C_out tile (batches x D-H groups x W tiles), one nisa.bn_stats group block each.
    # A flush is one C_OUT_REP-band group; h-chunking forces C_OUT_REP == 1, so they never mix.
    num_dh_groups = _count_dh_groups(dh_start, dh_end, num_dh_stacked, cfg.H_out)
    bn_partial_iters = cfg.B * num_dh_groups * num_w_tiles

    tile_cfg = Conv3dTileConfig(
        P_MAX=P_MAX,
        F_MAX=F_MAX,
        PSUM_BANK_SIZE=_PSUM_BANK_SIZE,
        NUM_PSUM_BANKS=_NUM_PSUM_BANKS,
        W_tile=W_tile,
        c_out_tile_size_max=c_out_tile_size_max,
        c_in_tile_size_max=c_in_tile_size_max,
        c_in_tile_count=c_in_tile_count,
        c_out_tile_count=c_out_tile_count,
        K_outer_tile_count=K_outer_tile_count,
        K_REP_max=K_REP_max,
        partition_stride_max=partition_stride_max,
        stacked_filter_dim_max=stacked_filter_dim_max,
        C_out_start=C_out_start,
        C_out_end=C_out_end,
        C_out_local=C_out_local,
        num_dh_stacked=num_dh_stacked,
        C_OUT_REP=C_OUT_REP,
        col_size=col_size,
        dh_head_size=dh_head_size,
        d_window_max=d_window_max,
        h_window_max=h_chunk_h_window_max,
        w_window_max=w_window_max,
        shard_on_dh=shard_on_dh,
        dh_start=dh_start,
        dh_end=dh_end,
        used_k_positions=used_k_positions,
        K_total_used=K_total_used,
        h_chunk_size=h_chunk_size,
        bn_partial_iters=bn_partial_iters,
        tensor_copy_activation_engine_offload=4,
        memset_gpsimd_engine_offload=2,
    )

    K_total_full = cfg.K_d * cfg.K_h * cfg.K_w
    logger = get_logger("conv3d")
    logger.debug(
        f"TileConfig: W_tile={tile_cfg.W_tile}, num_dh_stacked={tile_cfg.num_dh_stacked}, "
        f"c_out_tile_size_max={tile_cfg.c_out_tile_size_max}, c_in_tile_size_max={tile_cfg.c_in_tile_size_max}, "
        f"c_in_tile_count={tile_cfg.c_in_tile_count}, c_out_tile_count={tile_cfg.c_out_tile_count}, "
        f"K_outer_tile_count={tile_cfg.K_outer_tile_count}, K_REP_max={tile_cfg.K_REP_max}, "
        f"K_total_used={tile_cfg.K_total_used}/{K_total_full}, "
        f"h_chunk_size={tile_cfg.h_chunk_size}, h_window_max={tile_cfg.h_window_max}, "
        f"shard_on_dh={tile_cfg.shard_on_dh}, dh_range=[{tile_cfg.dh_start},{tile_cfg.dh_end}), "
        f"C_out_range=[{tile_cfg.C_out_start},{tile_cfg.C_out_end}), "
        f"C_OUT_REP={tile_cfg.C_OUT_REP}, col_size={tile_cfg.col_size}, dh_head_size={tile_cfg.dh_head_size}"
    )

    return tile_cfg


def _build_memory_config(
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    dtype_size: int,
    total_sbuf_budget: Optional[int] = None,
) -> Conv3dMemoryConfig:
    """Build memory configuration for SBUF buffer allocation and interleaving."""
    base_budget = total_sbuf_budget if total_sbuf_budget != None else nl.tile_size.total_available_sbuf_size
    TOTAL_SBUF = base_budget - _SBUF_RESERVE
    num_w_out_tiles = div_ceil(cfg.W_out, tile_cfg.W_tile)
    bias_size_per_tile = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
    filters_per_c_out_tile = (
        tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * tile_cfg.c_out_tile_size_max * dtype_size
    )
    per_c_out_tile_outer = bias_size_per_tile + filters_per_c_out_tile
    effective_free_dim = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    # Result/PSUM free footprint is one band's width (bands stack on partitions); stacked-input /
    # input-window span the full packed free dim. result_free_dim == effective_free_dim at rep 1.
    result_free_dim = tile_cfg.dh_head_size * tile_cfg.W_tile
    per_c_out_tile_inner = result_free_dim * dtype_size
    # A fused residual needs a second result-shaped tile per slot, unless the eviction consumes it in
    # place (Conv3dConfig.residual_in_place) and it is staged into the result tile itself.
    result_bufs_per_slot = 2 if cfg.needs_residual_staging_grid() else 1
    input_window_memory = tile_cfg.d_window_max * tile_cfg.h_window_max * tile_cfg.w_window_max * dtype_size
    stacked_input_memory = tile_cfg.K_outer_tile_count * effective_free_dim * dtype_size
    bias_align_waste = _heap_align_waste(_BIAS_DTYPE_SIZE) if cfg.has_bias else 0
    result_align_waste = _heap_align_waste(result_free_dim * dtype_size)
    input_window_align_waste = _heap_align_waste(input_window_memory)
    stacked_input_align_waste = _heap_align_waste(effective_free_dim * dtype_size)
    input_window_cost = input_window_memory + input_window_align_waste
    stacked_input_cost = stacked_input_memory + tile_cfg.K_outer_tile_count * stacked_input_align_waste

    # Reserve a fixed SBUF amount for the batchnorm accumulators; the fit search uses the rest.
    bn_memory = 0
    full_c_out_tile_count = div_ceil(cfg.C_out, tile_cfg.P_MAX)
    if cfg.is_batch_norm_training():
        # One bn_stats group (_BN_STATS_ELEMS fp32) per (C_out tile, flush). The finalize pass's own
        # scratch comes out of the free heap after the hot-path buffers, so it is not budgeted here.
        stats_bytes = full_c_out_tile_count * tile_cfg.bn_partial_iters * _BN_STATS_ELEMS * _BIAS_DTYPE_SIZE
        bn_memory = stats_bytes + _heap_align_waste(stats_bytes)
    elif cfg.is_batch_norm_eval():
        # Eval mode: single-column fp32 affine buffers, so alignment dominates. c_out_tile_count upper
        # bounds the c_out_interleave the fit search can pick.
        eval_col_bytes = _BIAS_DTYPE_SIZE + _heap_align_waste(_BIAS_DTYPE_SIZE)
        bn_memory = (len(_BN_EVAL_SCRATCH_NAMES) + _BN_EVAL_COEFF_PER_TILE * tile_cfg.c_out_tile_count) * eval_col_bytes
    budget = TOTAL_SBUF - bn_memory

    c_out_interleave = 1
    w_out_interleave = _NUM_PSUM_BANKS
    input_window_interleave = 2
    stacked_input_interleave = 2

    found_fit = False
    for min_buf_count in (2, 1):
        if found_fit:
            break
        for try_c_out in range(tile_cfg.c_out_tile_count, 0, -1):
            if found_fit:
                break
            c_out_wide_try = try_c_out * tile_cfg.c_out_tile_size_max
            filter_align_waste_try = _heap_align_waste(c_out_wide_try * dtype_size)
            fixed_filter_align = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * filter_align_waste_try
            outer_cost = try_c_out * per_c_out_tile_outer + try_c_out * bias_align_waste + fixed_filter_align
            for try_w_out in range(_NUM_PSUM_BANKS, 0, -1):
                inner_cost = try_w_out * try_c_out * result_bufs_per_slot * (per_c_out_tile_inner + result_align_waste)
                min_total = (
                    outer_cost + inner_cost + min_buf_count * input_window_cost + min_buf_count * stacked_input_cost
                )
                if min_total <= budget:
                    c_out_interleave, w_out_interleave = try_c_out, try_w_out
                    input_window_interleave = min_buf_count
                    stacked_input_interleave = min_buf_count
                    found_fit = True
                    break

    # The fit search is the capacity authority; name the up-front statistics reservation when that is
    # what shrank the budget. A fraction-of-SBUF pre-check was tried and removed (see design spec).
    bn_shape_hint = ""
    if not found_fit and cfg.is_batch_norm_training():
        bn_shape_hint = (
            f" BatchNormMode.TRAINING is not supported for this shape: the nisa.bn_stats buffer "
            f"reserves {bn_memory} B/partition of {TOTAL_SBUF} B up front, leaving too little for the "
            f"convolution's own tiles. It holds ceil(C_out/{tile_cfg.P_MAX}) * B * D-H groups * "
            f"W tiles * {_BN_STATS_ELEMS} fp32 groups — here {full_c_out_tile_count} * "
            f"{tile_cfg.bn_partial_iters} flushes — so reduce B, the output spatial size, or C_out. "
            f"BatchNormMode.EVAL and BatchNormMode.NONE accumulate no statistics and have no such "
            f"limit."
        )
    kernel_assert(
        found_fit,
        f"Failed to find an SBUF-fitting memory configuration for conv3d. "
        f"TOTAL_SBUF={TOTAL_SBUF}, per_c_out_tile_outer={per_c_out_tile_outer}, "
        f"per_c_out_tile_inner={per_c_out_tile_inner}, "
        f"input_window_cost={input_window_cost}, stacked_input_cost={stacked_input_cost}, "
        f"c_out_tile_count={tile_cfg.c_out_tile_count}.{bn_shape_hint}",
    )

    c_out_wide = c_out_interleave * tile_cfg.c_out_tile_size_max
    filter_align_waste = _heap_align_waste(c_out_wide * dtype_size)
    fixed_filter_align = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * filter_align_waste
    baseline = (
        c_out_interleave * per_c_out_tile_outer
        + c_out_interleave * bias_align_waste
        + fixed_filter_align
        + w_out_interleave * c_out_interleave * result_bufs_per_slot * (per_c_out_tile_inner + result_align_waste)
        + input_window_interleave * input_window_cost
        + stacked_input_interleave * stacked_input_cost
    )
    remaining = budget - baseline

    both_cost = input_window_cost + stacked_input_cost
    while remaining >= both_cost and both_cost > 0:
        input_window_interleave += 1
        stacked_input_interleave += 1
        remaining -= both_cost
    if remaining >= stacked_input_cost and stacked_input_cost > 0:
        stacked_input_interleave += 1
        remaining -= stacked_input_cost
    elif remaining >= input_window_cost and input_window_cost > 0:
        input_window_interleave += 1
        remaining -= input_window_cost

    # Account for SBM heap alignment overhead
    num_bias_allocs = c_out_interleave if cfg.has_bias else 0
    num_filter_allocs = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count
    num_result_allocs = w_out_interleave * c_out_interleave * result_bufs_per_slot
    num_input_window_allocs = input_window_interleave
    num_stacked_input_allocs = stacked_input_interleave * tile_cfg.K_outer_tile_count

    alignment_overhead = (
        num_filter_allocs * filter_align_waste
        + num_bias_allocs * bias_align_waste
        + num_result_allocs * result_align_waste
        + num_input_window_allocs * input_window_align_waste
        + num_stacked_input_allocs * stacked_input_align_waste
    )

    total_memory_required = (
        w_out_interleave * c_out_interleave * result_bufs_per_slot * per_c_out_tile_inner
        + input_window_interleave * input_window_memory
        + stacked_input_interleave * stacked_input_memory
        + c_out_interleave * per_c_out_tile_outer
        + alignment_overhead
        + bn_memory
    )

    # Store-batch merging (column tiling, B > 1): merge each band's store across store_batch_merge
    # batches via one strided DMA; needs full flushes (D_out == 1, dh_range % num_dh_stacked == 0).
    store_batch_merge = 1
    store_pipe = w_out_interleave
    dh_range = tile_cfg.dh_end - tile_cfg.dh_start
    merge_ok = (
        tile_cfg.C_OUT_REP > 1
        and cfg.B > 1
        and w_out_interleave >= 2
        and cfg.D_out == 1
        and tile_cfg.num_dh_stacked > 0
        and dh_range % tile_cfg.num_dh_stacked == 0
    )
    if merge_ok:
        store_pipe = 2
        store_batch_merge = min(cfg.B, w_out_interleave // store_pipe)
        if store_batch_merge < 1:
            store_batch_merge, store_pipe = 1, w_out_interleave

    # Upper bound on rescale-phase reload interleave (# reload iterations that exist); over-approx
    # B x C_out tiles x spatial tiles, divided by rep_step since each iteration fills C_OUT_REP bands.
    bn_reload_interleave = 1
    if cfg.is_batch_norm_training():
        reload_free_tile = tile_cfg.num_dh_stacked * tile_cfg.W_tile
        total_spatial = cfg.D_out * cfg.H_out * cfg.W_out
        rep_step = tile_cfg.C_OUT_REP * reload_free_tile
        max_reload_iters = cfg.B * full_c_out_tile_count * div_ceil(total_spatial, rep_step)
        bn_reload_interleave = max(1, max_reload_iters)

    mem_cfg = Conv3dMemoryConfig(
        dtype_size=dtype_size,
        TOTAL_SBUF=TOTAL_SBUF,
        bias_size_per_tile=bias_size_per_tile,
        filters_per_c_out_tile=filters_per_c_out_tile,
        per_c_out_tile_outer=per_c_out_tile_outer,
        per_c_out_tile_inner=per_c_out_tile_inner,
        input_window_memory=input_window_memory,
        stacked_input_memory=stacked_input_memory,
        c_out_interleave=c_out_interleave,
        input_window_interleave=input_window_interleave,
        stacked_input_interleave=stacked_input_interleave,
        w_out_interleave=w_out_interleave,
        num_w_out_tiles=num_w_out_tiles,
        total_memory_required=total_memory_required,
        store_batch_merge=store_batch_merge,
        result_bufs_per_slot=result_bufs_per_slot,
        store_pipe=store_pipe,
        bn_reload_interleave=bn_reload_interleave,
    )

    logger = get_logger("conv3d")
    logger.info(
        f"MemoryConfig: c_out_interleave={mem_cfg.c_out_interleave}, "
        f"w_out_interleave={mem_cfg.w_out_interleave}, "
        f"input_window_interleave={mem_cfg.input_window_interleave}, "
        f"stacked_input_interleave={mem_cfg.stacked_input_interleave}, "
        f"store_batch_merge={mem_cfg.store_batch_merge}, store_pipe={mem_cfg.store_pipe}, "
        f"alignment_overhead={alignment_overhead}, "
        f"total_memory_required={mem_cfg.total_memory_required}, TOTAL_SBUF={mem_cfg.TOTAL_SBUF}"
    )

    return mem_cfg


def _get_tensor_copy_engine(idx: int, modulo: int = 4) -> Any:
    """Alternate between vector and scalar engines for tensor copy pipelining.

    TODO: Determine optimal modulo using workload characteristics and cost model.
    """
    return nisa.scalar_engine if idx % modulo == 0 else nisa.vector_engine


def _get_memset_engine(idx: int, modulo: int = 2, dst: Optional[nl.NkiTensor] = None) -> Any:
    """Alternate between gpsimd and vector engines for memset pipelining.

    A PSUM destination always gets the Vector engine: nisa.memset accepts only vector / gpsimd, and
    asserts against gpsimd when dst is in PSUM. Deciding that here rather than at the call site keeps
    a caller from reintroducing the illegal pairing (pass dst whenever it may be a PSUM tile).

    TODO: Determine optimal modulo using workload characteristics and cost model.
    """
    if dst != None and nl.is_psum(dst.buffer):
        return nisa.vector_engine
    return nisa.gpsimd_engine if idx % modulo == 0 else nisa.vector_engine


def _validate_conv3d_inputs(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    gamma: Optional[nl.NkiTensor] = None,
    beta: Optional[nl.NkiTensor] = None,
    running_means: Optional[nl.NkiTensor] = None,
    running_variances: Optional[nl.NkiTensor] = None,
    residuals_in: Optional[nl.NkiTensor] = None,
    transpose_taps: bool = False,
) -> None:
    """Validate all input parameters for the 3D convolution kernel."""
    if cfg.has_residual_add():
        kernel_assert(
            residuals_in != None,
            "residual_add_loc is not ResidualAddLoc.NONE, which requires residuals_in to be provided.",
        )
        kernel_assert(
            tuple(residuals_in.shape) == (cfg.B, cfg.C_out, cfg.D_out, cfg.H_out, cfg.W_out),
            f"residuals_in must have shape [B, C_out, D_out, H_out, W_out] = "
            f"[{cfg.B}, {cfg.C_out}, {cfg.D_out}, {cfg.H_out}, {cfg.W_out}], got {tuple(residuals_in.shape)}",
        )
        kernel_assert(
            x_in.dtype == residuals_in.dtype,
            f"residuals_in dtype mismatch: {x_in.dtype} vs {residuals_in.dtype}",
        )
    if cfg.has_batch_norm():
        # Eval mode normalizes on the PSUM eviction, so the raw pre-batchnorm output is never
        # materialized and cannot be returned.
        kernel_assert(
            not (cfg.is_batch_norm_eval() and cfg.output_pre_norm),
            "batch_norm_mode=BatchNormMode.EVAL and output_pre_norm=True are mutually exclusive: eval "
            "mode normalizes in place on the PSUM eviction, so there is no pre-batchnorm output to return.",
        )
        kernel_assert(
            gamma != None and beta != None,
            "A fused batch_norm_mode requires both gamma and beta to be provided.",
        )
        kernel_assert(
            tuple(gamma.shape) == (cfg.C_out, 1),
            f"gamma must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(gamma.shape)}",
        )
        kernel_assert(
            tuple(beta.shape) == (cfg.C_out, 1),
            f"beta must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(beta.shape)}",
        )
        kernel_assert(
            running_means != None and running_variances != None,
            "A fused batch_norm_mode requires both running_means and running_variances to be provided.",
        )
        kernel_assert(
            tuple(running_means.shape) == (cfg.C_out, 1),
            f"running_means must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(running_means.shape)}",
        )
        kernel_assert(
            tuple(running_variances.shape) == (cfg.C_out, 1),
            f"running_variances must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(running_variances.shape)}",
        )
    kernel_assert(cfg.dilation_d >= 1, f"Dilation D must be >= 1, got {cfg.dilation_d}")
    kernel_assert(cfg.dilation_h >= 1, f"Dilation H must be >= 1, got {cfg.dilation_h}")
    kernel_assert(cfg.dilation_w >= 1, f"Dilation W must be >= 1, got {cfg.dilation_w}")
    kernel_assert(cfg.stride_d >= 1, f"Stride D must be >= 1, got {cfg.stride_d}")
    kernel_assert(cfg.stride_h >= 1, f"Stride H must be >= 1, got {cfg.stride_h}")
    kernel_assert(cfg.stride_w >= 1, f"Stride W must be >= 1, got {cfg.stride_w}")
    kernel_assert(
        cfg.pad_d_left >= 0 and cfg.pad_d_right >= 0,
        f"Padding D non-negative: ({cfg.pad_d_left}, {cfg.pad_d_right})",
    )
    kernel_assert(
        cfg.pad_h_top >= 0 and cfg.pad_h_bottom >= 0,
        f"Padding H non-negative: ({cfg.pad_h_top}, {cfg.pad_h_bottom})",
    )
    kernel_assert(
        cfg.pad_w_left >= 0 and cfg.pad_w_right >= 0,
        f"Padding W non-negative: ({cfg.pad_w_left}, {cfg.pad_w_right})",
    )
    kernel_assert(cfg.D_out >= 1, f"Output depth D_out must be >= 1, got {cfg.D_out}")
    kernel_assert(cfg.H_out >= 1, f"Output height H_out must be >= 1, got {cfg.H_out}")
    kernel_assert(cfg.W_out >= 1, f"Output width W_out must be >= 1, got {cfg.W_out}")
    filter_c_in = filters.shape[4] if transpose_taps else filters.shape[3]
    kernel_assert(x_in.shape[1] == filter_c_in, f"C_in mismatch: {x_in.shape[1]} vs {filter_c_in}")
    kernel_assert(x_in.dtype == filters.dtype, f"dtype mismatch: {x_in.dtype} vs {filters.dtype}")
    if bias != None:
        kernel_assert(x_in.dtype == bias.dtype, f"bias dtype mismatch: {x_in.dtype} vs {bias.dtype}")


# Memory Hierarchy 3: SBUF operations


def _compute_valid_free_range(
    cfg: Conv3dConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    used_k_positions: list[int],
    k_start: int,
    k_end: int,
) -> tuple[int, int]:
    """
    Compute the contiguous valid free-dim range for a K-outer tile.

    For each K position in [k_start, k_end), determines which (dh_idx, w_out)
    positions in the free dimension have at least one input position within valid
    bounds.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    full_free_dim = num_dh_stacked * w_tile_size

    # Start with empty range and take union
    global_valid_start = full_free_dim
    global_valid_end = 0

    for k_idx in range(k_start, k_end):
        flat_k_pos = used_k_positions[k_idx]
        k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)

        # Compute valid W range for this K position
        base_w_in_first = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
        # First valid w_out offset: input position >= 0
        if base_w_in_first < 0:
            k_w_valid_start = div_ceil(-base_w_in_first, cfg.stride_w)
        else:
            k_w_valid_start = 0
        # Last valid w_out offset: input position < W
        base_w_in_last = (w_end - 1) * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
        if base_w_in_last >= cfg.W:
            k_w_valid_end = w_tile_size - div_ceil(base_w_in_last - cfg.W + 1, cfg.stride_w)
        else:
            k_w_valid_end = w_tile_size

        if k_w_valid_start >= k_w_valid_end:
            # This K position has no valid W positions
            continue

        # Find first and last valid dh positions for this K position
        first_valid_dh = -1
        last_valid_dh = -1
        for dh_idx in range(num_dh_stacked):
            d_out, h_out = dh_positions[dh_idx]
            d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
            h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
            if 0 <= d_in < cfg.D and 0 <= h_in < cfg.H:
                if first_valid_dh == -1:
                    first_valid_dh = dh_idx
                last_valid_dh = dh_idx

        if first_valid_dh == -1:
            # No valid dh positions for this K position
            continue

        # Valid free-dim range: [first_valid_dh * w_tile_size + k_w_valid_start,
        #                        last_valid_dh * w_tile_size + k_w_valid_end)
        k_valid_start = first_valid_dh * w_tile_size + k_w_valid_start
        k_valid_end = last_valid_dh * w_tile_size + k_w_valid_end

        # Union with global range
        global_valid_start = min(global_valid_start, k_valid_start)
        global_valid_end = max(global_valid_end, k_valid_end)

    if global_valid_start >= global_valid_end:
        return 0, 0

    return global_valid_start, global_valid_end


def _compute_unwritten_free_ranges(
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
) -> list[tuple[int, int]]:
    """
    Compute free-dim ranges that no nc_matmul will ever write for this output tile.
    """
    K_total_used = tile_cfg.K_total_used
    K_REP = tile_cfg.K_REP_max
    K_outer_tile_count = tile_cfg.K_outer_tile_count
    w_tile_size = w_end - w_start
    effective_free_dim = len(dh_positions) * w_tile_size

    # Written mask = union of each K-outer tile's valid range; False positions are never written.
    written = [False] * effective_free_dim
    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        valid_start, valid_end = _compute_valid_free_range(
            cfg,
            dh_positions,
            w_start,
            w_end,
            tile_cfg.used_k_positions,
            k_start,
            k_end,
        )
        for pos in range(valid_start, valid_end):
            written[pos] = True

    # Coalesce contiguous unwritten positions into ranges
    unwritten_ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < effective_free_dim:
        if written[pos]:
            pos += 1
            continue
        gap_start = pos
        while pos < effective_free_dim and not written[pos]:
            pos += 1
        unwritten_ranges.append((gap_start, pos))
    return unwritten_ranges


# Memory Hierarchy 2: HBM <-> SBUF interleaved with SBUF operations


def _band_free_gap(
    band_dh_lo: int, w_tile_size: int, effective_free_dim: int, gap_start: int, gap_end: int
) -> tuple[int, int]:
    """
    Translate a flush-global never-written free range into one band's band-local free coords.

    Band positions start at flush-global free position band_dh_lo * w_tile_size and span
    effective_free_dim columns. Returns (lo, hi) clipped to the band (lo >= hi means the gap
    does not touch this band).
    """
    band_free_lo = band_dh_lo * w_tile_size
    lo = max(gap_start, band_free_lo) - band_free_lo
    hi = min(gap_end, band_free_lo + effective_free_dim) - band_free_lo
    return lo, hi


def _memset_band_free_gaps(
    psum_tile: nl.NkiTensor,
    column_bands: list[tuple[int, int, int]],
    c_out_tile_size: int,
    unwritten_free_ranges: list[tuple[int, int]],
    effective_free_dim: int,
    w_tile_size: int,
    memset_engine_modulo: int,
) -> None:
    """
    Zero the never-written free positions of each band directly in PSUM (fused eviction paths).

    _get_memset_engine forces Vector for a PSUM dst, so these do not participate in the gpsimd/vector
    rotation the SBUF gap fill uses — the ISA forbids gpsimd on PSUM.
    """
    gap_engine_ctr = 0
    for band_idx in range(len(column_bands)):
        bp_lo = column_bands[band_idx][0]
        bp_hi = bp_lo + c_out_tile_size
        band_dh_lo = column_bands[band_idx][1]
        for gap_idx in range(len(unwritten_free_ranges)):
            lo, hi = _band_free_gap(
                band_dh_lo,
                w_tile_size,
                effective_free_dim,
                unwritten_free_ranges[gap_idx][0],
                unwritten_free_ranges[gap_idx][1],
            )
            if lo < hi:
                gap_dst = psum_tile[bp_lo:bp_hi, lo:hi]
                nisa.memset(
                    dst=gap_dst,
                    value=0.0,
                    engine=_get_memset_engine(gap_engine_ctr, memset_engine_modulo, gap_dst),
                )
                gap_engine_ctr += 1


def _fill_band_free_gaps_sbuf(
    result_full: nl.NkiTensor,
    bias_sbuf: Optional[nl.NkiTensor],
    column_bands: list[tuple[int, int, int]],
    c_out_tile_size: int,
    unwritten_free_ranges: list[tuple[int, int]],
    effective_free_dim: int,
    w_tile_size: int,
    has_bias: bool,
    has_activation: bool,
    activation_fn: Optional[ActFnType],
    memset_engine_modulo: int,
) -> None:
    """
    Fill never-written free positions in the result SBUF with f(0 + bias) (non-batchnorm path).

    Per band, zero the band-local gap then apply the same bias / activation as the main copy.
    The bias for band g lives at partitions [g * col_size, ...) of the passed bias_sbuf (it is
    pre-replicated across bands), matching the band's result partitions.
    """
    gap_engine_ctr = 0
    for band_idx in range(len(column_bands)):
        bp_lo = column_bands[band_idx][0]
        bp_hi = bp_lo + c_out_tile_size
        band_dh_lo = column_bands[band_idx][1]
        band_bias = bias_sbuf[bp_lo:bp_hi, :] if (has_bias and bias_sbuf != None) else bias_sbuf
        for gap_idx in range(len(unwritten_free_ranges)):
            lo, hi = _band_free_gap(
                band_dh_lo,
                w_tile_size,
                effective_free_dim,
                unwritten_free_ranges[gap_idx][0],
                unwritten_free_ranges[gap_idx][1],
            )
            if lo >= hi:
                continue
            gap_result = result_full[bp_lo:bp_hi, lo:hi]
            nisa.memset(dst=gap_result, value=0.0, engine=_get_memset_engine(gap_engine_ctr, memset_engine_modulo))
            gap_engine_ctr += 1
            if has_bias and has_activation:
                nisa.tensor_scalar(dst=gap_result, data=gap_result, op0=nl.add, operand0=band_bias)
                nisa.activation(dst=gap_result, data=gap_result, op=get_nl_act_fn_from_type(activation_fn))
            elif has_bias:
                nisa.tensor_scalar(dst=gap_result, data=gap_result, op0=nl.add, operand0=band_bias)
            elif has_activation:
                nisa.activation(dst=gap_result, data=gap_result, op=get_nl_act_fn_from_type(activation_fn))


def _apply_bias_activation_and_copy(
    psum_tiles: list[nl.NkiTensor],
    result_sbufs: list[nl.NkiTensor],
    bias_sbufs: list[Optional[nl.NkiTensor]],
    c_out_tile_sizes: list[int],
    has_bias: bool,
    has_activation: bool,
    activation_fn: Optional[ActFnType],
    effective_free_dim: int,
    unwritten_free_ranges: Optional[list[tuple[int, int]]] = None,
    memset_engine_modulo: int = 2,
    batch_norm_mode: BatchNormMode = BatchNormMode.NONE,
    bn_stats_bufs: Optional[nl.NkiTensor] = None,
    bn_partial_iters: int = 1,
    bn_iter_idx: int = 0,
    bn_tile_offset: int = 0,
    evict_partitions: int = 0,
    column_bands: Optional[list[tuple[int, int, int]]] = None,
    w_tile_size: int = 1,
    bn_eval_true_gammas: Optional[list[nl.NkiTensor]] = None,
    bn_eval_neg_true_betas: Optional[list[nl.NkiTensor]] = None,
    residual_sbufs: Optional[list[nl.NkiTensor]] = None,
    residual_pre_act: bool = False,
    residual_aliases_result: bool = False,
) -> None:
    """
    Apply optional bias / activation and copy the PSUM result to SBUF. One path per BatchNormMode:

    * NONE: bias and/or activation on the copy-out (or a plain copy), filling never-written padded
      positions with f(0 + bias).
    * TRAINING: evict with an activation copy (folding in any bias), then take this flush's
      statistics with nisa.bn_stats, one group per column-tiling band into its own bn_stats_bufs
      slot. No activation fn (deferred to the rescale phase so the statistics see the raw conv
      output) and no per-flush read-modify-write; nisa.bn_aggr combines the groups later.
    * EVAL: apply the precomputed affine on the eviction — one tensor_scalar
      (result = psum * true_gamma - neg_true_beta, any bias folded in), or with an activation a
      single nisa.activation carrying the whole affine in scale / bias (its coefficient sign is
      pre-flipped by _load_bn_eval_scales_for_c_out_group). The eviction produces the final output.

    A fused residual add rides along in the op already evicting the tile rather than being appended,
    so the chain stays at two instructions; only the op on the far side of the activation stays
    separate. On all fused paths the padded positions are zeroed in PSUM first, so the same op gives
    them their true value. add_pre implies has_activation (cfg normalizes PRE_ACT without one).

    In the NONE residual branch the residual is read by the same op that first writes the result tile,
    which is what lets Conv3dConfig.residual_in_place alias the two buffers — the read and write are
    one instruction. The kernel_assert below enforces that only that branch sees them aliased.

    Column tiling: the C_OUT_REP bands share the band-local free layout [0, effective_free_dim) at
    partition offsets g * col_size, so one op over [0, evict_partitions) evicts all bands (the
    per-band split happens downstream). C_OUT_REP == 1 is byte-identical to the non-tiled path.
    """
    has_unwritten = unwritten_free_ranges != None and len(unwritten_free_ranges) > 0
    if column_bands == None:
        column_bands = [(0, 0, 0)]

    # Aliasing contract (see docstring): only the branch whose FIRST op reads the residual is safe to
    # alias, so reordering or adding a branch below fails loudly here rather than corrupting the add.
    kernel_assert(
        not residual_aliases_result
        or (batch_norm_mode == BatchNormMode.NONE and (not has_activation or residual_pre_act)),
        f"residual aliases the result tile but this eviction writes it before reading the residual: "
        f"batch_norm_mode={batch_norm_mode}, has_activation={has_activation}, "
        f"residual_pre_act={residual_pre_act}",
    )

    for c_out_tile_idx in range(len(result_sbufs)):
        psum_tile = psum_tiles[c_out_tile_idx]
        result_full = result_sbufs[c_out_tile_idx]
        bias_full = bias_sbufs[c_out_tile_idx]
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        # Active partition span over all bands (== c_out_tile_size when col_rep == 1). Bias is
        # pre-replicated across bands; slice it to evict_p (buffer may be taller on a rep-1 tail).
        evict_p = evict_partitions if evict_partitions > 0 else c_out_tile_size
        bias_sbuf = bias_full[:evict_p, :] if (has_bias and bias_full != None) else bias_full
        result = result_full[:evict_p, :effective_free_dim]
        psum_data = psum_tile[:evict_p, :effective_free_dim]
        # Residual tile: same band / free layout as the result tile it is added into. add_pre / add_post
        # select the side of the activation (see docstring).
        residual = None
        if residual_sbufs != None:
            residual = residual_sbufs[c_out_tile_idx][:evict_p, :effective_free_dim]
        add_pre = residual != None and residual_pre_act
        add_post = residual != None and not residual_pre_act

        # Fused paths produce the final value in one op reading PSUM, so zero the padded positions in
        # PSUM and let them ride through it; the plain path fills them in SBUF afterwards instead.
        if has_unwritten and (batch_norm_mode.is_fused() or residual != None):
            _memset_band_free_gaps(
                psum_tile,
                column_bands,
                c_out_tile_size,
                unwritten_free_ranges,
                effective_free_dim,
                w_tile_size,
                memset_engine_modulo,
            )

        if batch_norm_mode == BatchNormMode.EVAL:
            eval_gamma = bn_eval_true_gammas[c_out_tile_idx][:evict_p, 0:1]
            eval_beta = bn_eval_neg_true_betas[c_out_tile_idx][:evict_p, 0:1]
            if has_activation and not add_pre:
                # ActFn(psum * true_gamma + true_beta) in one fp32-internal instruction (the
                # coefficient's sign is pre-flipped for the adding operand), then any POST_ACT add.
                nisa.activation(
                    dst=result,
                    data=psum_data,
                    op=get_nl_act_fn_from_type(activation_fn),
                    scale=eval_gamma,
                    bias=eval_beta,
                )
                if add_post:
                    nisa.tensor_tensor(dst=result, data1=result, data2=residual, op=nl.add)
            elif has_activation:
                # PRE_ACT needs the residual between the normalization and the activation, so
                # normalize with the whole affine inside one instruction, then add, then activate.
                nisa.activation(dst=result, data=psum_data, op=nl.copy, scale=eval_gamma, bias=eval_beta)
                nisa.tensor_tensor(dst=result, data1=result, data2=residual, op=nl.add)
                nisa.activation(dst=result, data=result, op=get_nl_act_fn_from_type(activation_fn))
            else:
                # result = psum * true_gamma - neg_true_beta (bias folded into neg_true_beta).
                nisa.tensor_scalar(
                    dst=result,
                    data=psum_data,
                    op0=nl.multiply,
                    operand0=eval_gamma,
                    op1=nl.subtract,
                    operand1=eval_beta,
                )
                if residual != None:
                    nisa.tensor_tensor(dst=result, data1=result, data2=residual, op=nl.add)
        elif batch_norm_mode == BatchNormMode.TRAINING:
            # Evict with an activation copy, folding any bias into its bias operand. No activation fn:
            # TRAINING defers it to the rescale phase so the statistics see the raw conv output.
            if has_bias:
                nisa.activation(dst=result, data=psum_data, op=nl.copy, bias=bias_sbuf)
            else:
                nisa.activation(dst=result, data=psum_data, op=nl.copy)

            # One bn_stats group per band, at its own partitions. Each band passes its real free width, so a
            # ragged tail is not read — zero-filling would dilute the mean, since the count weights it.
            global_tile_idx = bn_tile_offset + c_out_tile_idx
            group_col = (global_tile_idx * bn_partial_iters + bn_iter_idx) * _BN_STATS_ELEMS
            for band_idx in range(len(column_bands)):
                bp_lo = column_bands[band_idx][0]
                band_real_free = (column_bands[band_idx][2] - column_bands[band_idx][1]) * w_tile_size
                nisa.bn_stats(
                    dst=bn_stats_bufs[bp_lo : bp_lo + c_out_tile_size, group_col : group_col + _BN_STATS_ELEMS],
                    data=result_full[bp_lo : bp_lo + c_out_tile_size, 0:band_real_free],
                )
        elif residual != None:
            # Residual fused into the op that reads PSUM, so the chain is at most two instructions and
            # the gap zeroing above carries the padded positions (see this function's docstring).
            if add_pre or not has_activation:
                # result = (psum + bias) + residual.
                if has_bias:
                    nisa.scalar_tensor_tensor(
                        dst=result, data=psum_data, op0=nl.add, operand0=bias_sbuf, op1=nl.add, operand1=residual
                    )
                else:
                    nisa.tensor_tensor(dst=result, data1=psum_data, data2=residual, op=nl.add)
                if add_pre:
                    nisa.activation(dst=result, data=result, op=get_nl_act_fn_from_type(activation_fn))
            else:
                # result = ActFn(psum + bias) + residual.
                if has_bias:
                    nisa.activation(
                        dst=result, data=psum_data, op=get_nl_act_fn_from_type(activation_fn), bias=bias_sbuf
                    )
                else:
                    nisa.activation(dst=result, data=psum_data, op=get_nl_act_fn_from_type(activation_fn))
                nisa.tensor_tensor(dst=result, data1=result, data2=residual, op=nl.add)
        else:
            if has_bias and has_activation:
                nisa.activation(dst=result, data=psum_data, op=get_nl_act_fn_from_type(activation_fn), bias=bias_sbuf)
            elif has_bias:
                nisa.tensor_scalar(dst=result, data=psum_data, op0=nl.add, operand0=bias_sbuf)
            elif has_activation:
                nisa.activation(dst=result, data=psum_data, op=get_nl_act_fn_from_type(activation_fn))
            else:
                nisa.tensor_copy(dst=result, src=psum_data)

            # Fill never-written free positions in the result SBUF (memset targets SBUF, not PSUM)
            # with f(0 + bias), per band in band-local free coords.
            if has_unwritten:
                _fill_band_free_gaps_sbuf(
                    result_full,
                    bias_sbuf,
                    column_bands,
                    c_out_tile_size,
                    unwritten_free_ranges,
                    effective_free_dim,
                    w_tile_size,
                    has_bias,
                    has_activation,
                    activation_fn,
                    memset_engine_modulo,
                )


# Memory Hierarchy 1: HBM <-> SBUF


def _allocate_bn_eval_buffers(
    sbm: SbufManager,
    coeff_part_dim: int,
    c_out_interleave: int,
) -> tuple[list[nl.NkiTensor], list[nl.NkiTensor], list[nl.NkiTensor]]:
    scratch_bufs: list[nl.NkiTensor] = []
    for scratch_idx in range(len(_BN_EVAL_SCRATCH_NAMES)):
        scratch_bufs.append(
            sbm.alloc_heap(
                shape=(coeff_part_dim, 1),
                dtype=nl.float32,
                name=_BN_EVAL_SCRATCH_NAMES[scratch_idx],
            )
        )
    true_gamma_bufs: list[nl.NkiTensor] = []
    neg_true_beta_bufs: list[nl.NkiTensor] = []
    for tile_idx in range(c_out_interleave):
        true_gamma_bufs.append(
            sbm.alloc_heap(shape=(coeff_part_dim, 1), dtype=nl.float32, name=f"bn_eval_true_gamma_{tile_idx}")
        )
        neg_true_beta_bufs.append(
            sbm.alloc_heap(shape=(coeff_part_dim, 1), dtype=nl.float32, name=f"bn_eval_neg_true_beta_{tile_idx}")
        )
    return scratch_bufs, true_gamma_bufs, neg_true_beta_bufs


def _load_bn_eval_scales_for_c_out_group(
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    running_means: nl.NkiTensor,
    running_variances: nl.NkiTensor,
    bias_cache: list[Optional[nl.NkiTensor]],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    batch_norm_eps: float,
    c_out_tile_sizes: list[int],
    c_out_group_start: int,
    scratch_bufs: list[nl.NkiTensor],
    true_gamma_bufs: list[nl.NkiTensor],
    neg_true_beta_bufs: list[nl.NkiTensor],
    identity_mask: list,
) -> None:
    """
    Precompute the eval-mode batchnorm affine coefficients for one C_out interleave group.

    Loads this group's gamma / beta / running_means / running_variances slices and derives
    true_gamma / neg_true_beta per C_out tile with _compute_true_bn_scales — using the *running*
    statistics, which eval mode normalizes with and never updates.

    When the conv has a bias, it is folded into the coefficients instead of being added separately:
        gamma * ((y + b) - mean) / sqrt(var + eps) + beta == y * true_gamma - (neg_true_beta - b * true_gamma)
    so the whole bias + batchnorm chain still costs one tensor_scalar on the eviction. The bias is
    read from the already-loaded (and band-replicated) SBUF bias buffers.

    With an activation the eviction takes the affine through nisa.activation's scale / bias operands,
    which add rather than subtract, so the sign is flipped here — once per C_out tile per group
    instead of per eviction, and in place, so it costs no extra buffer.
    """
    C_OUT_REP = tile_cfg.C_OUT_REP
    band_stride = _band_part_stride(tile_cfg)
    gamma_buf = scratch_bufs[0]
    beta_buf = scratch_bufs[1]
    mean_buf = scratch_bufs[2]
    var_buf = scratch_bufs[3]
    tmp_buf = scratch_bufs[4]

    for c_out_tile_idx in range(len(c_out_tile_sizes)):
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        c_out_tile_end = c_out_tile_start + c_out_tile_size

        gamma_col = gamma_buf[:c_out_tile_size, 0:1]
        beta_col = beta_buf[:c_out_tile_size, 0:1]
        mean_col = mean_buf[:c_out_tile_size, 0:1]
        var_col = var_buf[:c_out_tile_size, 0:1]
        nisa.dma_copy(dst=gamma_col, src=gamma.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
        nisa.dma_copy(dst=beta_col, src=beta.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
        nisa.dma_copy(dst=mean_col, src=running_means.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
        nisa.dma_copy(
            dst=var_col, src=running_variances.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
        )

        _compute_true_bn_scales(
            mean_col=mean_col,
            var_col=var_col,
            gamma_col=gamma_col,
            beta_col=beta_col,
            true_gamma_buf=true_gamma_bufs[c_out_tile_idx],
            neg_true_beta_buf=neg_true_beta_bufs[c_out_tile_idx],
            batch_norm_eps=batch_norm_eps,
            c_out_tile_size=c_out_tile_size,
            C_OUT_REP=C_OUT_REP,
            band_stride=band_stride,
            identity_mask=identity_mask,
        )

        if cfg.has_bias:
            # neg_true_beta -= bias * true_gamma, over the coefficients' full band span (the
            # per-band [c_out_tile_size, col_size) gaps hold garbage in both, and are never stored).
            fold_part = C_OUT_REP * band_stride if C_OUT_REP > 1 else c_out_tile_size
            tmp_col = tmp_buf[:fold_part, 0:1]
            nisa.tensor_tensor(
                dst=tmp_col,
                data1=bias_cache[c_out_tile_idx][:fold_part, 0:1],
                data2=true_gamma_bufs[c_out_tile_idx][:fold_part, 0:1],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=neg_true_beta_bufs[c_out_tile_idx][:fold_part, 0:1],
                data1=neg_true_beta_bufs[c_out_tile_idx][:fold_part, 0:1],
                data2=tmp_col,
                op=nl.subtract,
            )

        if cfg.has_activation:
            # Flip to true_beta for the eviction's fused nisa.activation(scale, bias), which adds.
            flip_part = C_OUT_REP * band_stride if C_OUT_REP > 1 else c_out_tile_size
            nisa.tensor_scalar(
                dst=neg_true_beta_bufs[c_out_tile_idx][:flip_part, 0:1],
                data=neg_true_beta_bufs[c_out_tile_idx][:flip_part, 0:1],
                op0=nl.multiply,
                operand0=-1.0,
            )


def _load_bias_into_buf(
    bias: nl.NkiTensor,
    bias_buf: nl.NkiTensor,
    c_out_start: int,
    c_out_end: int,
    C_OUT_REP: int = 1,
    col_size: int = 128,
) -> nl.NkiTensor:
    """
    Load a slice of the bias tensor from HBM into an SBUF buffer via DMA.

    Under column tiling (C_OUT_REP > 1) the bias is replicated into each of the C_OUT_REP
    partition bands (offsets band_idx * col_size), so the eviction of a band reads its per-channel
    bias directly. The band offsets are multiples of col_size (32 or 64), i.e. quadrant
    aligned, so a plain tensor_copy suffices. Returns the full (possibly replicated) buffer.
    """
    c_out_size = c_out_end - c_out_start
    nisa.dma_copy(
        dst=bias_buf[:c_out_size, 0],
        src=bias.slice(dim=0, start=c_out_start, end=c_out_end, step=1),
    )
    if C_OUT_REP > 1:
        for band_idx in range(1, C_OUT_REP):
            nisa.tensor_copy(
                dst=bias_buf[band_idx * col_size : band_idx * col_size + c_out_size, :],
                src=bias_buf[0:c_out_size, :],
            )
        return bias_buf
    return bias_buf.slice(dim=0, start=0, end=c_out_size, step=1)
