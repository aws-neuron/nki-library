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

"""Batch normalization backward kernel for NeuronCore."""

from dataclasses import dataclass, field
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, align_to, sizeinbytes
from ...core.utils.kernel_assert import assert_shape, kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.lnc_sendrecv import lnc_sendrecv
from ...core.utils.logging import get_logger


@nki.jit
def batch_norm_bwd(
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    eps: float = 1e-5,
    fuse_relu: Optional[bool] = False,
    relu_mask: Optional[nl.NkiTensor] = None,
    residuals_in: Optional[nl.NkiTensor] = None,
    compute_dresidual: bool = False,
) -> tuple:
    """
    Compute batch-normalization gradients with optional fused ReLU and pre-ReLU residual addition.

    Reductions span the batch and spatial dimensions. Channels are tiled by at most 128 and the
    flattened spatial dimension by at most 8,192 elements; channel and spatial tails are supported.

    Dimensions:
        B: Batch size
        C: Number of channels
        D: Depth
        H: Height
        W: Width

    Args:
        dy (nl.NkiTensor): [B, C, D, H, W] incoming gradient in HBM.
        x_in (nl.NkiTensor): [B, C, D, H, W] forward batch-normalization input in HBM.
        means (nl.NkiTensor): [C] float32 channel means saved by the forward pass.
        variances (nl.NkiTensor): [C] float32 biased channel variances saved by the forward pass.
        gamma (nl.NkiTensor): [C, 1] float32 channel scale.
        beta (nl.NkiTensor): [C, 1] float32 channel shift, read only when recomputing a ReLU mask.
        eps (float): Positive variance offset used by the forward pass. Defaults to 1e-5.
        fuse_relu (Optional[bool]): Whether the forward pass applied ReLU.
        relu_mask (Optional[nl.NkiTensor]): Optional [B, C, D, H, W] uint8 forward ReLU mask.
            When omitted with fuse_relu=True, the kernel recomputes the predicate.
        residuals_in (Optional[nl.NkiTensor]): Optional [B, C, D, H, W] residual added before
            ReLU. Must match dy's dtype and requires fuse_relu=True.
        compute_dresidual (bool): Whether to return the residual gradient. Requires residuals_in.

    Returns:
        dx (nl.NkiTensor): [B, C, D, H, W] input gradient, with dy's dtype.
        dgamma (nl.NkiTensor): [C, 1] float32 scale gradient.
        dbeta (nl.NkiTensor): [C, 1] float32 shift gradient.
        dresidual (nl.NkiTensor): Optional [B, C, D, H, W] residual gradient, returned only
            when compute_dresidual=True.

    Notes:
        - A supplied relu_mask is exact. A recomputed mask uses approximately 2^-18-relative
          scale/bias precision and can differ only near ReLU's zero-valued kink.
        - dx is rounded once, into dy's dtype: the masked dy is staged unscaled and phase 2 forms
          dy' * gamma * rstd - var_x with a float32 var_x. Reductions and channel coefficients use
          float32.

    Pseudocode:
        validate inputs and choose channel/spatial tiles
        phase 1:
            mask dy using relu_mask or batch_norm(x_in) + residuals_in
            optionally store masked dy as dresidual
            reduce dbeta and dgamma; stage masked dy
        phase 2:
            dx = masked dy * gamma * rstd - (mean and variance correction terms, float32)
        return dx, dgamma, dbeta[, dresidual]
    """
    _validate_inputs(
        dy,
        x_in,
        means,
        variances,
        gamma,
        beta,
        eps,
        fuse_relu,
        relu_mask,
        residuals_in,
        compute_dresidual,
    )

    # fuse_relu with no mask: rebuild the forward output per tile and mask on its sign instead of
    # reading a [B, C, D, H, W] mask from HBM.
    recompute_relu_mask = fuse_relu and relu_mask == None
    recompute_with_residual = recompute_relu_mask and residuals_in != None

    B, C, D, H, W = dy.shape
    spatial = D * H * W
    inv_n = 1.0 / (B * spatial)
    P_MAX = nl.tile_size.pmax
    n_c_tiles = div_ceil(C, P_MAX)
    dtype_size = sizeinbytes(dy.dtype)

    dx = nl.ndarray(shape=(B, C, D, H, W), dtype=dy.dtype, buffer=nl.shared_hbm)
    dgamma = nl.ndarray(shape=(C, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbeta = nl.ndarray(shape=(C, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dresidual = nl.ndarray(shape=(B, C, D, H, W), dtype=dy.dtype, buffer=nl.shared_hbm) if compute_dresidual else None

    # LNC sharding. Channel tiles first (per-channel reductions need no cross-core traffic); too few
    # for the cores, shard the batch at two cross-core reductions; with neither, compute redundantly.
    _, n_prgs, prg_id = get_verified_program_sharding_info("batch_norm_bwd", (0, 1))
    # The batch-sharded path reduces with one pairwise lnc_sendrecv, which only spans ALL cores at
    # n_prgs <= 2; at 4 the pairs {0,3} / {1,2} would each sum half the batch. Unreachable today.
    kernel_assert(n_prgs <= 2, f"[batch_norm_bwd] expected at most 2 logical cores, got {n_prgs}.")
    shard_on_b = False
    if n_prgs > 1 and n_c_tiles >= n_prgs:
        tiles_per_core = div_ceil(n_c_tiles, n_prgs)
        c_tile_lo = min(tiles_per_core * prg_id, n_c_tiles)
        c_tile_hi = min(c_tile_lo + tiles_per_core, n_c_tiles)
        b_lo, b_hi = 0, B
    elif n_prgs > 1 and B >= n_prgs:
        shard_on_b = True
        c_tile_lo, c_tile_hi = 0, n_c_tiles
        batch_per_core = div_ceil(B, n_prgs)
        b_lo = min(batch_per_core * prg_id, B)
        b_hi = min(b_lo + batch_per_core, B)
    else:
        c_tile_lo, c_tile_hi = 0, n_c_tiles
        b_lo, b_hi = 0, B

    # dy' crosses both phases, so it is staged in HBM, sized to this core's region only (a full
    # buffer wastes 134 MB/core at the stress config). Views rebase to this core's origin.
    owned_batch = b_hi - b_lo
    owned_c_lo = c_tile_lo * P_MAX
    owned_c = min(c_tile_hi * P_MAX, C) - owned_c_lo
    masked_dy_hbm = nl.ndarray(shape=(owned_batch, owned_c, spatial), dtype=dy.dtype, buffer=nl.private_hbm)

    # Sized after sharding: replication folds batches into one unit, shrinking the useful slot count.
    # Single derivation of the factor — the walk takes it, both phases read it back off the walk.
    replication = _replication_factor(min(P_MAX, C), C, owned_batch, P_MAX)
    mem_cfg = _build_memory_config(
        spatial,
        owned_batch,
        n_c_tiles,
        c_tile_hi - c_tile_lo,
        replication,
        dtype_size,
        fuse_relu,
        recompute_relu_mask,
        recompute_with_residual,
    )
    f_tile = mem_cfg.f_tile
    n_sp_tiles = mem_cfg.n_sp_tiles
    interleave = mem_cfg.interleave

    # Built once so the phases cannot disagree about the sequence the resident handoff depends on.
    tile_walk = _build_tile_walk(c_tile_lo, c_tile_hi, C, b_lo, b_hi, n_sp_tiles, replication, P_MAX)
    n_partials = _units_per_c_tile(owned_batch, replication, n_sp_tiles)

    logger = get_logger("batch_norm_bwd")
    logger.debug(
        f"batch_norm_bwd: shape=[{B}, {C}, {D}, {H}, {W}], spatial={spatial}, f_tile={f_tile}, "
        f"n_sp_tiles={n_sp_tiles}, n_c_tiles={n_c_tiles}, c_tile_range=[{c_tile_lo}, {c_tile_hi}), "
        f"shard_on_b={shard_on_b}, batch_range=[{b_lo}, {b_hi}), "
        f"replication={replication}, work_units={len(tile_walk)}, "
        f"fuse_relu={fuse_relu}, recompute_relu_mask={recompute_relu_mask}, "
        f"recompute_with_residual={recompute_with_residual}, compute_dresidual={compute_dresidual}, "
        f"interleave={interleave} "
        f"(p1 {mem_cfg.phase1_slot_bytes} B/slot, p2 {mem_cfg.phase2_slot_bytes} B/slot)"
    )

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, logger=logger)
    sbm.open_scope(name="batch_norm_bwd")

    # Per-channel columns shared by both phases; column index == channel tile index.
    means_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="means")
    gamma_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="gamma")
    rstd_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="rstd")
    gamma_rstd_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="gamma_rstd")
    dbeta_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="dbeta")
    dgamma_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="dgamma")
    # Phase-2 coefficients, one column per channel tile: phase 2 visits tiles out of walk order,
    # so consecutive units can be on different channel tiles and these cannot be shared scratch.
    mean_x_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="mean_x")
    intermediate_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="intermediate_scale")
    var_x_bias_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="var_x_bias")

    # dy' slots span both phases, so they are heap-allocated before phase 1's scope and freed
    # only after phase 2. Phase 1's last tiles stay resident for phase 2 to read without an HBM trip.
    masked_dy_slots = []
    for slot_idx in range(mem_cfg.interleave):
        masked_dy_slots.append(sbm.alloc_heap((P_MAX, f_tile), dtype=dy.dtype, name=f"masked_dy_tile_{slot_idx}"))

    _batch_norm_bwd_direct_and_affine_phase(
        dy=dy,
        x_in=x_in,
        means=means,
        variances=variances,
        gamma=gamma,
        beta=beta,
        eps=eps,
        dgamma=dgamma,
        dbeta=dbeta,
        relu_mask=relu_mask,
        residuals_in=residuals_in,
        dresidual=dresidual,
        masked_dy_hbm=masked_dy_hbm,
        sbm=sbm,
        means_all=means_all,
        gamma_all=gamma_all,
        rstd_all=rstd_all,
        gamma_rstd_all=gamma_rstd_all,
        dbeta_all=dbeta_all,
        dgamma_all=dgamma_all,
        masked_dy_slots=masked_dy_slots,
        tile_walk=tile_walk,
        mem_cfg=mem_cfg,
        n_partials=n_partials,
        spatial=spatial,
        b_lo=b_lo,
        owned_c_lo=owned_c_lo,
        P_MAX=P_MAX,
        fuse_relu=fuse_relu,
        recompute_relu_mask=recompute_relu_mask,
        recompute_with_residual=recompute_with_residual,
        shard_on_b=shard_on_b,
        prg_id=prg_id,
        n_prgs=n_prgs,
    )

    _batch_norm_bwd_dx_phase(
        x_in=x_in,
        dx=dx,
        masked_dy_hbm=masked_dy_hbm,
        sbm=sbm,
        means_all=means_all,
        gamma_all=gamma_all,
        rstd_all=rstd_all,
        gamma_rstd_all=gamma_rstd_all,
        dbeta_all=dbeta_all,
        dgamma_all=dgamma_all,
        mean_x_all=mean_x_all,
        intermediate_all=intermediate_all,
        var_x_bias_all=var_x_bias_all,
        masked_dy_slots=masked_dy_slots,
        tile_walk=tile_walk,
        dx_dtype=dy.dtype,
        mem_cfg=mem_cfg,
        C=C,
        spatial=spatial,
        inv_n=inv_n,
        b_lo=b_lo,
        owned_c_lo=owned_c_lo,
        c_tile_lo=c_tile_lo,
        c_tile_hi=c_tile_hi,
        P_MAX=P_MAX,
    )

    # Free the shared dy' slots (allocated on the heap before phase 1).
    for _slot_idx in range(mem_cfg.interleave):
        sbm.pop_heap()

    sbm.close_scope()

    if compute_dresidual:
        return dx, dgamma, dbeta, dresidual
    return dx, dgamma, dbeta


# Spatial (D * H * W) elements per partition in one tile; _build_memory_config reduces it to fit.
_MAX_F_TILE = 8192
_SBUF_RESERVE = 1024
_HEAP_ALIGN = nl.tile_size.sbuf_min_align
# fp32 columns living across both phases, indexed by channel tile: means, gamma, rstd,
# gamma_rstd, d_beta, d_gamma.
_NUM_PERSISTENT_COLS = 6
# Phase-1-only column (variances), per channel tile so each prologue DMA does not serialize behind
# the previous tile's rsqrt.
_NUM_PHASE1_COLS = 1
# Two more phase-1 columns, only when the mask is recomputed: beta and true_beta.
_NUM_RECOMPUTE_COLS = 2
# Phase-2 coefficients (mean_x, intermediate_scale, var_x_bias), one column per channel tile
# because phase 2 visits tiles out of walk order.
_NUM_PHASE2_COEFF_COLS = 3
# Slack for each phase's single-column scratch. Over-counted on purpose; a column is 4 B/partition.
_NUM_SCRATCH_COLS = 12
# Multibuffering depth cap. Two slots already overlap DMA with compute; beyond ~8 the extra slots
# just crowd out f_tile.
_MAX_INTERLEAVE = 8
_MIN_INTERLEAVE = 1
# nc_stream_shuffle needs quadrant-aligned start partitions, so replica bands sit on a 32 stride.
_QUADRANT = 32

# Dtype of a loaded ReLU mask and of the rebuilt forward output used to recompute it.
_COMPUTE_DTYPE = nl.bfloat16
# float32 so dx is rounded only once, into its own dtype.
_VAR_X_DTYPE = nl.float32


@dataclass
class BatchNormBwdMemoryConfig(nl.NKIObject):
    """
    SBUF layout for both phases: the spatial tile size and the multibuffering depth.

    Both are shared by the phases. f_tile sets the DMA granularity and both phases walk the same
    tiles; interleave is single because the phases rotate the *same* heap-allocated dy'
    slots (tile t uses slot t % interleave in each), so the depths cannot differ without
    breaking the SBUF-resident handoff between them.
    """

    f_tile: int
    """Spatial elements per partition in one tile, for both phases."""
    n_sp_tiles: int
    """Spatial tiles per (channel tile, batch) = ceil(spatial / f_tile)."""
    interleave: int
    """Rotating slots per phase, and the number of shared dy' slots."""
    phase1_slot_bytes: int
    """Per-partition bytes of one phase-1 multibuffered slot (all its tiles together)."""
    phase2_slot_bytes: int
    """Per-partition bytes of one phase-2 multibuffered slot (all its tiles together)."""
    phase1_total_bytes: int
    """Per-partition bytes the budget predicts phase 1 needs, asserted against the real mark."""
    phase2_total_bytes: int
    """Per-partition bytes the budget predicts phase 2 needs, asserted against the real mark."""


def _replication_factor(c_size: int, C: int, owned_batch: int, P_MAX: int) -> int:
    """
    How many batches to process at once by replicating across the partition dimension.

    A channel tile of c_size < 128 channels leaves 128 - c_size partitions idle. Replication
    fills them by giving replica r its own batch, at partitions [r * c_size, (r + 1) * c_size).

    All of the following must hold, so the factor is 1 (no replication, the original behaviour)
    unless every one is met:

    1. c_size is a whole number of 32-partition quadrants. The cross-partition gather (of the
       dbeta / sum(dy * x_in) partials) and the broadcast (of the per-channel coefficients) both
       go through nc_stream_shuffle, whose src/dst start partitions must be quadrant-aligned, so
       band r must begin on a quadrant boundary.
    2. c_size == C, i.e. this is the only channel tile. The single-DMA load merges the batch and
       channel axes of the HBM tensor into one partition axis, which is only a valid view when
       the channel slice spans the whole channel dimension — otherwise consecutive batches are
       not a fixed stride apart in the sliced view and the merge is rejected as non-contiguous.
    3. The factor divides owned_batch, so every work unit has the same number of active bands.
       A ragged tail unit would break the dbeta readout: reduce_regs is read out on the channel
       tile's last unit, and nisa.activation requires reduce_res to span exactly that
       instruction's partitions — a short tail could therefore never read back the bands that
       earlier units accumulated. Requiring an exact division sidesteps that entirely; the
       largest such factor is used, so e.g. owned_batch 6 with room for 4 bands uses 3.

    Returns the replication factor; 1 means no replication.
    """
    if c_size % _QUADRANT != 0 or c_size != C:
        return 1
    max_bands = max(1, P_MAX // c_size)
    factor = 1
    for candidate in range(2, max_bands + 1):
        if owned_batch % candidate == 0:
            factor = candidate
    return factor


def _build_tile_walk(
    c_tile_lo: int, c_tile_hi: int, C: int, b_lo: int, b_hi: int, n_sp_tiles: int, rep: int, P_MAX: int
) -> list:
    """
    Enumerate the (super-)tiles both phases walk, in order.

    Each entry is (c_tile_idx, c_size, rep, b_base, sp_tile_idx): one unit of work covering `rep`
    batches at once, band r holding batch b_base + r on partitions
    [r * c_size, (r + 1) * c_size). rep divides the owned batch exactly (see
    _replication_factor), so every unit has exactly rep active bands — there is no ragged tail.

    rep is passed in rather than recomputed here so the walk, the SBUF budget and phase 2 all
    share one derivation of it. It is uniform across channel tiles: replication requires
    c_size == C, so rep > 1 implies there is only one channel tile.

    Built once and shared by both phases so their tile sequences — and therefore the rotating
    slot assignment and the SBUF-resident dy' handoff — cannot drift apart.
    """
    walk = []
    for c_tile_idx in range(c_tile_lo, c_tile_hi):
        c_start = c_tile_idx * P_MAX
        c_size = min(c_start + P_MAX, C) - c_start
        # rep divides the owned batch exactly, so every unit has exactly rep active bands.
        for b_base in range(b_lo, b_hi, rep):
            for sp_tile_idx in range(n_sp_tiles):
                walk.append((c_tile_idx, c_size, rep, b_base, sp_tile_idx))
    return walk


def _units_per_c_tile(owned_batch: int, replication: int, n_sp_tiles: int) -> int:
    """
    Work units one channel tile contributes, i.e. phase 1's partial-column count.

    Replication folds `replication` batches into a single unit, and the factor divides
    owned_batch exactly (see _replication_factor), so every channel tile contributes the same
    number of units. Phase 1 relies on that: it sizes yx_partials from this count while indexing
    it with a per-channel-tile counter.
    """
    return (owned_batch // replication) * n_sp_tiles


def _replicated_hbm_view(
    tensor: nl.NkiTensor,
    b_base: int,
    n_bands: int,
    c_start: int,
    c_end: int,
    sp_start: int,
    sp_end: int,
) -> nl.NkiTensor:
    """
    View an HBM tensor as one (bands * channels, spatial) tile.

    Accepts either a [B, C, D, H, W] input/output tensor or the already-flat [B, C, spatial]
    dy' stage; the spatial dims are merged first when there are more than three.

    Selects batches [b_base, b_base + n_bands), channels [c_start, c_end) and the spatial slice,
    then merges batch and channel into a single partition axis so that partition (r * c_size + c)
    is batch b_base + r, channel c_start + c. That layout is exactly the band layout in SBUF, so
    one dma_copy fills every band.

    With a single band the merge is skipped: it is only a valid view when the channel slice spans
    the whole channel dimension (see _replication_factor), and an unreplicated tile has no reason
    to require that.
    """
    view = tensor.slice(dim=0, start=b_base, end=b_base + n_bands, step=1).slice(
        dim=1, start=c_start, end=c_end, step=1
    )
    if len(tensor.shape) > 3:
        view = view.flatten_dims(2, len(tensor.shape) - 1)
    view = view.slice(dim=2, start=sp_start, end=sp_end, step=1)
    if n_bands == 1:
        return view.select(dim=0, index=0)
    return view.flatten_dims(0, 1)


def _identity_shuffle_mask() -> list:
    """Identity mask for nc_stream_shuffle: each output partition takes the same input partition."""
    mask = []
    for partition_idx in range(_QUADRANT):
        mask.append(partition_idx)
    return mask


def _gather_bands_and_sum(
    dst_col: nl.NkiTensor,
    wide_col: nl.NkiTensor,
    gather_buf: nl.NkiTensor,
    c_size: int,
    n_bands: int,
    shuffle_mask: list,
) -> None:
    """
    Collapse a replicated per-band column into one canonical column at partitions [0, c_size).

    wide_col holds one value per (band, channel) at partitions [r * c_size + c). Each band is
    moved down to [0, c_size) with nc_stream_shuffle (band offsets are quadrant multiples, so
    this is legal) into its own column of gather_buf, then the n_bands columns are summed along
    the free dimension. Needed because dbeta and sum(dy * x_in) are reductions over the batch,
    and replication splits the batch across bands.

    n_bands must be the number of bands actually written, not the replication factor: a tail work
    unit can have fewer active bands, and the unwritten ones hold garbage.

    dst_col may alias wide_col's band 0, as the sum(dy' * x_in) caller does. That is safe: the
    reduce reads every column of gather_buf, so it flow-depends on all n_bands shuffles and cannot
    be scheduled ahead of the one that reads band 0.
    """
    if n_bands == 1:
        return
    for band_idx in range(n_bands):
        part_lo = band_idx * c_size
        nisa.nc_stream_shuffle(
            dst=gather_buf[:c_size, band_idx : band_idx + 1],
            src=wide_col[part_lo : part_lo + c_size, 0:1],
            shuffle_mask=shuffle_mask,
        )
    nisa.tensor_reduce(dst=dst_col, op=nl.add, data=gather_buf[:c_size, 0:n_bands], axis=1)


def _broadcast_to_bands(col_all: nl.NkiTensor, c_size: int, rep: int, shuffle_mask: list) -> None:
    """
    Replicate a per-channel column from partitions [0, c_size) into every other band.

    The per-channel scale / bias operands of the replicated activations and tensor_scalars must
    supply a value for every active partition, so each band needs its own copy. Band offsets are
    quadrant multiples, so nc_stream_shuffle can place them.
    """
    for band_idx in range(1, rep):
        part_lo = band_idx * c_size
        nisa.nc_stream_shuffle(
            dst=col_all[part_lo : part_lo + c_size, 0:1],
            src=col_all[:c_size, 0:1],
            shuffle_mask=shuffle_mask,
        )


def _slot_cost(f_tile: int, dtype_sizes: list[int]) -> int:
    """Per-partition cost of one multibuffered slot holding one tile of each given dtype."""
    cost = 0
    for dtype_idx in range(len(dtype_sizes)):
        tile_bytes = f_tile * dtype_sizes[dtype_idx]
        cost = cost + align_to(tile_bytes, _HEAP_ALIGN)
    return cost


def _build_memory_config(
    spatial: int,
    batch: int,
    n_c_tiles: int,
    n_owned_c_tiles: int,
    replication: int,
    dtype_size: int,
    fuse_relu: bool,
    recompute_relu_mask: bool,
    recompute_with_residual: bool,
) -> BatchNormBwdMemoryConfig:
    """
    Pick the spatial tile size and the shared multibuffering depth to fit the SBUF budget.

    Both phases are DMA bound, so the goal is to keep f_tile large (long, contiguous DMAs)
    while still affording enough rotating slots to overlap a tile's DMA with the previous
    tile's compute. The search therefore starts from the largest f_tile and the deepest
    interleave and gives up interleave depth first, halving f_tile only once even
    double-buffering no longer fits:

        for f_tile = _MAX_F_TILE, /2, ... :
            for interleave = _MAX_INTERLEAVE ... _MIN_INTERLEAVE:
                if both phases fit: take it

    The dy' slots are shared: they are heap-allocated once and live across both phases, so
    the tiles phase 1 wrote last are still SBUF-resident when phase 2 starts and can be consumed
    without a round trip through HBM. They are therefore charged once, to both phases' budgets,
    rather than counted in either phase's per-slot cost.

    The per-phase slots are:
      - phase 1: dy (input dtype), x_in (input dtype), plus one compute-dtype tile when fusing —
        the loaded relu_mask, or the rebuilt forward output that replaces it. Residual-aware
        rematerialization additionally holds one residual tile at the input dtype.
      - phase 2: x_in (input dtype), dx (input dtype)
    Phase 1's non-rotating tiles (the dy * x_in product, the partial columns) and phase 2's
    (var_x) plus the per-channel columns are charged once, outside the slot cost.
    """
    compute_size = sizeinbytes(_COMPUTE_DTYPE)
    var_x_size = sizeinbytes(_VAR_X_DTYPE)
    f32_size = sizeinbytes(nl.float32)
    total_sbuf = nl.tile_size.total_available_sbuf_size - _SBUF_RESERVE

    # Per-channel columns are allocated before either phase and live across both.
    persistent_bytes = _NUM_PERSISTENT_COLS * align_to(n_c_tiles * f32_size, _HEAP_ALIGN)
    coeff_bytes = _NUM_PHASE2_COEFF_COLS * align_to(n_c_tiles * f32_size, _HEAP_ALIGN)
    scratch_bytes = _NUM_SCRATCH_COLS * align_to(f32_size, _HEAP_ALIGN)
    # Phase-1-only columns: variances, plus beta / true_beta when the mask is recomputed.
    n_phase1_cols = _NUM_PHASE1_COLS
    if recompute_relu_mask:
        n_phase1_cols = n_phase1_cols + _NUM_RECOMPUTE_COLS
    phase1_col_bytes = n_phase1_cols * align_to(n_c_tiles * f32_size, _HEAP_ALIGN)

    # One slot's worth of each phase's rotating tiles; phase 1's dy lives in the shared dy' slot.
    phase1_slot_dtypes = [dtype_size]
    if fuse_relu:
        # One compute-dtype tile either way: the loaded relu_mask, or the rebuilt forward output
        # that replaces it on the recompute path.
        phase1_slot_dtypes.append(compute_size)
    if recompute_with_residual:
        phase1_slot_dtypes.append(dtype_size)
    phase2_slot_dtypes = [dtype_size, dtype_size]

    f_tile = min(spatial, _MAX_F_TILE)
    chosen_f_tile = 0
    chosen_interleave = 0
    chosen_p1_total = 0
    chosen_p2_total = 0
    while f_tile >= 1:
        n_sp_tiles = div_ceil(spatial, f_tile)
        n_partials = _units_per_c_tile(batch, replication, n_sp_tiles)
        partials_bytes = align_to(n_partials * f32_size, _HEAP_ALIGN)
        # Phase 1 also holds the non-rotating dy * x_in tile (float32, see phase 1) and the partials.
        dyx_bytes = align_to(f_tile * f32_size, _HEAP_ALIGN)
        common_fixed = persistent_bytes + coeff_bytes + scratch_bytes
        # Phase 1 also holds the dbeta activation's discard tile.
        discard_bytes = align_to(f_tile * compute_size, _HEAP_ALIGN)
        phase1_fixed = common_fixed + partials_bytes + dyx_bytes + phase1_col_bytes + discard_bytes
        # Phase 2 also holds the single (non-rotating) var_x tile.
        var_x_bytes = align_to(f_tile * var_x_size, _HEAP_ALIGN)
        phase2_fixed = common_fixed + var_x_bytes

        p1_slot = _slot_cost(f_tile, phase1_slot_dtypes)
        p2_slot = _slot_cost(f_tile, phase2_slot_dtypes)
        # The shared dy' slots are live during both phases, so they are added to each.
        dx_slot = _slot_cost(f_tile, [dtype_size])

        # Slots rotate over every work unit, channel tiles included. Capped at the unit count: extra
        # slots would trade f_tile away untouched, so n_partials must account for replication.
        total_tiles = n_owned_c_tiles * n_partials
        interleave = min(_MAX_INTERLEAVE, total_tiles)
        while interleave >= _MIN_INTERLEAVE:
            p1_total = phase1_fixed + interleave * (p1_slot + dx_slot)
            p2_total = phase2_fixed + interleave * (p2_slot + dx_slot)
            if p1_total <= total_sbuf and p2_total <= total_sbuf:
                chosen_f_tile = f_tile
                chosen_interleave = interleave
                chosen_p1_total = p1_total
                chosen_p2_total = p2_total
                break
            interleave = interleave - 1
        if chosen_f_tile > 0:
            break
        f_tile = f_tile // 2

    kernel_assert(
        chosen_f_tile > 0,
        f"[batch_norm_bwd] No SBUF-fitting memory configuration: spatial={spatial}, "
        f"batch={batch}, channel_tiles={n_c_tiles}, input_dtype_bytes={dtype_size}, "
        f"fuse_relu={fuse_relu}, recompute_relu_mask={recompute_relu_mask}, "
        f"recompute_with_residual={recompute_with_residual}, budget={total_sbuf}.",
    )

    return BatchNormBwdMemoryConfig(
        f_tile=chosen_f_tile,
        n_sp_tiles=div_ceil(spatial, chosen_f_tile),
        interleave=chosen_interleave,
        phase1_slot_bytes=_slot_cost(chosen_f_tile, phase1_slot_dtypes),
        phase2_slot_bytes=_slot_cost(chosen_f_tile, phase2_slot_dtypes),
        phase1_total_bytes=chosen_p1_total,
        phase2_total_bytes=chosen_p2_total,
    )


def _assert_sbuf_budget(sbm: SbufManager, predicted_bytes: int, phase_name: str) -> None:
    """
    Check a phase's real SBUF footprint against what _build_memory_config predicted for it.

    The budget is assembled from hand-maintained column counts (the _NUM_*_COLS constants), so it
    can silently drift from the allocations it is meant to model. Left unchecked, an under-estimate
    surfaces as an opaque stack-OOM inside SbufManager with nothing pointing back at the budget;
    this turns it into a trace-time failure naming both numbers.

    Measures the live stack plus heap at the call site rather than SbufManager's max_combined_usage,
    which is a running high-water mark that never resets: by phase 2 that mark still carries phase
    1's peak, so every drift would be attributed to phase 2. Call this right after a phase has
    allocated everything it needs and before it frees anything. The heap term is included because
    the shared dy' slots live there and are charged to both phases' budgets.
    """
    live_stack = sbm.stack_curr_addr - sbm.lower_bound
    live_heap = sbm.upper_bound - sbm.heap_curr_addr
    actual_bytes = live_stack + live_heap
    kernel_assert(
        actual_bytes <= predicted_bytes,
        f"[batch_norm_bwd] {phase_name} allocated {actual_bytes} B per partition of SBUF "
        f"({live_stack} B stack + {live_heap} B heap) but _build_memory_config budgeted "
        f"{predicted_bytes} B. The _NUM_*_COLS constants have drifted from the allocations; update "
        f"them so the budget covers what the phase allocates.",
    )


def _validate_inputs(
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    eps: float,
    fuse_relu: Optional[bool],
    relu_mask: Optional[nl.NkiTensor],
    residuals_in: Optional[nl.NkiTensor],
    compute_dresidual: bool,
) -> None:
    """Validate shapes, dtypes and eps. fuse_relu needs no check: relu_mask is optional either way."""
    # A non-positive eps defeats the guard: rsqrt gives +inf at zero variance, NaN below it.
    kernel_assert(eps > 0.0, f"[batch_norm_bwd] eps must be positive, received {eps}.")
    kernel_assert(
        len(dy.shape) == 5,
        f"[batch_norm_bwd] dy must be a 5D [B, C, D, H, W] tensor, received shape {dy.shape}.",
    )
    C = dy.shape[1]
    assert_shape(x_in, dy.shape, "x_in", "x_in must have the same shape as dy.")
    # The SBUF budget charges one slot per tensor at dy's dtype size (see _build_memory_config),
    # so a wider x_in would silently under-budget and fail as a stack OOM rather than here.
    kernel_assert(
        x_in.dtype == dy.dtype,
        f"[batch_norm_bwd] x_in dtype ({x_in.dtype}) must match dy dtype ({dy.dtype}).",
    )
    assert_shape(means, (C,), "means")
    assert_shape(variances, (C,), "variances")
    assert_shape(gamma, (C, 1), "gamma")
    assert_shape(beta, (C, 1), "beta")
    # The loading DMA would silently widen a narrower dtype instead of failing, and every coefficient
    # derives from these, so bf16 statistics would cost dgamma most of its digits.
    for stat_name, stat in (("means", means), ("variances", variances), ("gamma", gamma), ("beta", beta)):
        kernel_assert(
            stat.dtype == nl.float32,
            f"[batch_norm_bwd] {stat_name} must be float32, received {stat.dtype}.",
        )
    if relu_mask != None:
        # Checked whenever supplied, not just when used, so a wrongly-typed mask is not ignored.
        kernel_assert(
            relu_mask.dtype == nl.uint8,
            f"[batch_norm_bwd] relu_mask must be uint8, received {relu_mask.dtype}.",
        )
        assert_shape(relu_mask, dy.shape, "relu_mask", "relu_mask must have the same shape as dy.")
    if residuals_in != None:
        kernel_assert(
            fuse_relu,
            "[batch_norm_bwd] residuals_in requires fuse_relu=True because the residual is added before ReLU.",
        )
        assert_shape(
            residuals_in,
            dy.shape,
            "residuals_in",
            "residuals_in must have the same shape as dy.",
        )
        kernel_assert(
            residuals_in.dtype == dy.dtype,
            f"[batch_norm_bwd] residuals_in dtype ({residuals_in.dtype}) must match dy dtype ({dy.dtype}).",
        )
    if compute_dresidual:
        kernel_assert(
            residuals_in != None,
            "[batch_norm_bwd] compute_dresidual=True requires residuals_in.",
        )


@dataclass
class _Phase1Buffers(nl.NKIObject):
    """
    Phase 1's own SBUF buffers, allocated once by _alloc_phase1_buffers.

    Grouped into one object so the per-tile and per-channel-tile steps below take a single buffers
    argument instead of a dozen tensors each. The shared dy' slots are NOT here: they are
    caller-owned and outlive this phase.
    """

    dy_slots: list = field(default_factory=list)
    """Rotating dy tiles, one per multibuffer slot."""
    x_slots: list = field(default_factory=list)
    """Rotating x_in tiles, one per multibuffer slot."""
    mask_slots: list = field(default_factory=list)
    """Rotating loaded-relu_mask tiles; empty unless fusing from a supplied mask."""
    post_bn_slots: list = field(default_factory=list)
    """Rotating rebuilt-forward-output tiles; empty unless recomputing the mask."""
    residual_slots: list = field(default_factory=list)
    """Rotating residual tiles; empty unless rematerializing a residual-aware predicate."""
    discard: Optional[nl.NkiTensor] = None
    """Single tile: the unused destination of the dbeta-accumulating activation."""
    dyx: Optional[nl.NkiTensor] = None
    """Single (non-rotating) dy' * x_in product tile, float32."""
    yx_partials: Optional[nl.NkiTensor] = None
    """One float32 partial column of sum(dy' * x_in) per work unit."""
    var_all: Optional[nl.NkiTensor] = None
    """Loaded variances, one column per channel tile."""
    beta_all: Optional[nl.NkiTensor] = None
    """Loaded beta, one column per channel tile; None unless recomputing the mask."""
    true_beta_all: Optional[nl.NkiTensor] = None
    """beta - means * true_gamma, one column per channel tile; None unless recomputing."""
    yx_col: Optional[nl.NkiTensor] = None
    """Single-column scratch holding the collapsed sum(dy' * x_in)."""
    scratch_col: Optional[nl.NkiTensor] = None
    """Single-column scratch for the means * dbeta product."""
    band_gather: Optional[nl.NkiTensor] = None
    """Per-band gather scratch for the two cross-partition reductions."""
    dbeta_wide: Optional[nl.NkiTensor] = None
    """Full-height dbeta readout under replication, before the gather to [0, c_size)."""
    recv_dbeta: Optional[nl.NkiTensor] = None
    """Landing buffer for the peer's partial dbeta; None unless batch-sharded."""
    recv_yx: Optional[nl.NkiTensor] = None
    """Landing buffer for the peer's partial sum(dy' * x_in); None unless batch-sharded."""


def _alloc_phase1_buffers(
    sbm: SbufManager,
    x_dtype: nki.dtype,
    f_tile: int,
    interleave: int,
    n_partials: int,
    n_c_tiles: int,
    P_MAX: int,
    fuse_relu: bool,
    recompute_relu_mask: bool,
    recompute_with_residual: bool,
    shard_on_b: bool,
    masked_dy_slots: list,
) -> _Phase1Buffers:
    """
    Allocate every phase-1 buffer, in the order the SBUF budget charges them.

    Rotating slots come first (one x_in tile per slot, plus one compute-dtype tile when fusing; dy
    loads into the caller's shared slots),
    then the single-buffered tiles and the per-channel-tile and scratch columns. The dtype choices
    that are not simply the input dtype are justified at their allocation.
    """
    bufs = _Phase1Buffers()
    bufs.dy_slots = []
    bufs.x_slots = []
    # Exactly one is populated, on the same condition the compute branches on.
    bufs.mask_slots = []
    bufs.post_bn_slots = []
    bufs.residual_slots = []
    for slot_idx in range(interleave):
        # dy is loaded and masked in place in the shared slot, which phase 2 reads as dy'.
        bufs.dy_slots.append(masked_dy_slots[slot_idx])
        bufs.x_slots.append(sbm.alloc_stack((P_MAX, f_tile), dtype=x_dtype, name=f"x_in_tile_{slot_idx}"))
        if recompute_relu_mask:
            # The compute dtype suffices: bf16's exponent range keeps the tiny residue's sign. The
            # mask's real limit is the engines' operand path — see the tile loop.
            bufs.post_bn_slots.append(
                sbm.alloc_stack((P_MAX, f_tile), dtype=_COMPUTE_DTYPE, name=f"x_in_post_bn_tile_{slot_idx}")
            )
            if recompute_with_residual:
                bufs.residual_slots.append(
                    sbm.alloc_stack((P_MAX, f_tile), dtype=x_dtype, name=f"residual_tile_{slot_idx}")
                )
        elif fuse_relu:
            # The DMA casts uint8 -> compute dtype, so the masking multiply is same-dtype.
            # This is important for 2x perf mode on bf16 tensor_tensor instructions
            bufs.mask_slots.append(
                sbm.alloc_stack((P_MAX, f_tile), dtype=_COMPUTE_DTYPE, name=f"relu_mask_tile_{slot_idx}")
            )
    # float32: dgamma = (sum(dy' * x_in) - mean * dbeta) * rstd cancels at large mean(x), where a bf16
    # product costs ~20% error at mean 5 and ~69% at 20 vs ~1e-3 % here. Cheap: single-buffered.
    bufs.dyx = sbm.alloc_stack((P_MAX, f_tile), dtype=nl.float32, name="dy_times_x_tile")
    bufs.discard = sbm.alloc_stack((P_MAX, f_tile), dtype=_COMPUTE_DTYPE, name="dbeta_discard_tile")
    # One float32 column per work unit (full-height: a replicated unit writes one partial per band),
    # collapsed once the channel tile completes. tensor_reduce accumulates in float32 regardless.
    bufs.yx_partials = sbm.alloc_stack((P_MAX, n_partials), dtype=nl.float32, name="yx_partials")
    # One column per channel tile, not a shared one: sharing would put a WAR edge from each prologue
    # DMA onto the previous tile's rsqrt, serializing prologues to save 4 B per partition per tile.
    bufs.var_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="variances")
    if recompute_relu_mask:
        # beta and the affine bias derived from it, per channel tile for the same reason var_all is.
        bufs.beta_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="beta")
        bufs.true_beta_all = sbm.alloc_stack((P_MAX, n_c_tiles), dtype=nl.float32, name="true_beta")
    bufs.yx_col = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, name="yx_sum_col")
    bufs.scratch_col = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, name="affine_scratch_col")
    # Per-band gather scratch for the two cross-partition reductions (one column per band).
    bufs.band_gather = sbm.alloc_stack((P_MAX, P_MAX // _QUADRANT), dtype=nl.float32, name="band_gather")
    # dbeta lands here at full height (one value per band) before being gathered to [0, c_size).
    bufs.dbeta_wide = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, name="dbeta_wide_col")
    if shard_on_b:
        # Landing buffers for the peer's partial dbeta / sum(dy' * x_in).
        bufs.recv_dbeta = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, name="recv_dbeta_col")
        bufs.recv_yx = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, name="recv_yx_col")
    return bufs


def _phase1_channel_tile_prologue(
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    eps: float,
    means_all: nl.NkiTensor,
    gamma_all: nl.NkiTensor,
    rstd_all: nl.NkiTensor,
    gamma_rstd_all: nl.NkiTensor,
    bufs: _Phase1Buffers,
    c_tile_idx: int,
    c_start: int,
    c_end: int,
    c_size: int,
    rep: int,
    shuffle_mask: list,
    recompute_relu_mask: bool,
) -> None:
    """
    Load a channel tile's statistics and derive its per-channel coefficients.

    Runs on the first work unit of each channel tile: loads means / variances / gamma, forms
    rstd = rsqrt(var + eps) and gamma_rstd = gamma * rstd, and when the ReLU mask is recomputed
    also loads beta and forms true_beta. Every coefficient the replicated per-tile ops read is
    broadcast to all active bands here.
    """
    means_col = means_all[:c_size, c_tile_idx : c_tile_idx + 1]
    gamma_col = gamma_all[:c_size, c_tile_idx : c_tile_idx + 1]
    rstd_col = rstd_all[:c_size, c_tile_idx : c_tile_idx + 1]
    var_col = bufs.var_all[:c_size, c_tile_idx : c_tile_idx + 1]
    gamma_rstd_col = gamma_rstd_all[:c_size, c_tile_idx : c_tile_idx + 1]

    nisa.dma_copy(
        dst=means_all[:c_size, c_tile_idx],
        src=means.slice(dim=0, start=c_start, end=c_end, step=1),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )
    nisa.dma_copy(
        dst=bufs.var_all[:c_size, c_tile_idx],
        src=variances.slice(dim=0, start=c_start, end=c_end, step=1),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )
    nisa.dma_copy(
        dst=gamma_col,
        src=gamma.slice(dim=0, start=c_start, end=c_end, step=1),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )

    # rstd = rsqrt(variances + eps); gamma_rstd = gamma * rstd.
    nisa.activation(dst=rstd_col, op=nl.rsqrt, data=var_col, bias=eps)
    nisa.tensor_tensor(dst=gamma_rstd_col, data1=gamma_col, data2=rstd_col, op=nl.multiply)
    # The replicated activations below scale every active partition, so each band needs its own copy.
    _broadcast_to_bands(gamma_rstd_all[:, c_tile_idx : c_tile_idx + 1], c_size, rep, shuffle_mask)

    if recompute_relu_mask:
        # conv3d's fused-forward affine pair: x_in * true_gamma + true_beta == gamma *
        # (x_in - means) * rstd + beta. Un-negated bias because the activation adds it.
        beta_col = bufs.beta_all[:c_size, c_tile_idx : c_tile_idx + 1]
        true_beta_col = bufs.true_beta_all[:c_size, c_tile_idx : c_tile_idx + 1]
        nisa.dma_copy(
            dst=beta_col,
            src=beta.slice(dim=0, start=c_start, end=c_end, step=1),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nki.isa.engine.sync,
        )
        # true_beta = beta - means * true_gamma, conv3d's tensor_scalar with the subtract reversed.
        nisa.tensor_scalar(
            dst=true_beta_col,
            data=means_col,
            op0=nl.multiply,
            operand0=gamma_rstd_col,
            op1=nl.subtract,
            operand1=beta_col,
            reverse1=True,
        )
        # The activation below biases every active partition, so bands need their copy.
        _broadcast_to_bands(bufs.true_beta_all[:, c_tile_idx : c_tile_idx + 1], c_size, rep, shuffle_mask)


def _phase1_mask_dy(
    relu_mask: Optional[nl.NkiTensor],
    residuals_in: Optional[nl.NkiTensor],
    bufs: _Phase1Buffers,
    dy_tile: nl.NkiTensor,
    x_tile: nl.NkiTensor,
    gamma_rstd_active: nl.NkiTensor,
    true_beta_active: Optional[nl.NkiTensor],
    slot_idx: int,
    active_part: int,
    sp_size: int,
    band_extent: list,
    recompute_relu_mask: bool,
) -> None:
    """
    Apply the fused ReLU's mask to dy in place, so every later reader sees the masked gradient.

    With a supplied relu_mask the uint8 HBM tile is DMA'd in (the DMA casts) and multiplied against
    dy. With no mask, the batchnorm forward output the ReLU saw is rebuilt from the already-loaded
    x_in as x_in * true_gamma + true_beta, the residual is added when supplied, and dy is masked
    on the resulting sign — see the kernel docstring's Notes for why that rebuild is only exact
    to ~2^-18 and why the resulting flips are benign.

    band_extent carries the HBM view coordinates (b_base, rep, c_start, c_end, sp_start, sp_end)
    needed to load a supplied mask; it is unused on the recompute path.
    """
    if recompute_relu_mask:
        post_bn_tile = bufs.post_bn_slots[slot_idx][:active_part, :sp_size]
        residual_tile = None
        if residuals_in != None:
            residual_tile = bufs.residual_slots[slot_idx][:active_part, :sp_size]
            residual_view = _replicated_hbm_view(
                residuals_in,
                band_extent[0],
                band_extent[1],
                band_extent[2],
                band_extent[3],
                band_extent[4],
                band_extent[5],
            )
            nisa.dma_copy(
                dst=residual_tile,
                src=residual_view,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nki.isa.engine.sync,
            )
        nisa.activation(
            dst=post_bn_tile,
            op=nl.copy,
            data=x_tile,
            scale=gamma_rstd_active,
            bias=true_beta_active,
        )
        if residuals_in != None:
            nisa.tensor_tensor(
                dst=post_bn_tile,
                data1=post_bn_tile,
                data2=residual_tile,
                op=nl.add,
            )
        # dy' = (x_in_post_bn > 0) * dy. Strict greater matches relu's convention and (y > 0).
        nisa.scalar_tensor_tensor(
            dst=dy_tile,
            data=post_bn_tile,
            op0=nl.greater,
            operand0=0.0,
            op1=nl.multiply,
            operand1=dy_tile,
        )
        return
    mask_tile = bufs.mask_slots[slot_idx][:active_part, :sp_size]
    mask_view = _replicated_hbm_view(
        relu_mask, band_extent[0], band_extent[1], band_extent[2], band_extent[3], band_extent[4], band_extent[5]
    )
    # Default DGE mode, unlike every other dma_copy here: this is the one casting transfer
    # (uint8 HBM -> bf16 SBUF) and hwdge rejects it with NCC_IBIR098 "Unsupported DGE Type".
    nisa.dma_copy(dst=mask_tile, src=mask_view)
    nisa.tensor_tensor(dst=dy_tile, data1=dy_tile, data2=mask_tile, op=nl.multiply)


def _phase1_tile_compute(
    bufs: _Phase1Buffers,
    dy_tile: nl.NkiTensor,
    x_tile: nl.NkiTensor,
    dyx_tile: nl.NkiTensor,
    discard_tile: nl.NkiTensor,
    dbeta_readout: Optional[nl.NkiTensor],
    is_first_tile: bool,
    active_part: int,
    tile_idx: int,
) -> None:
    """
    One tile's compute: accumulate dbeta and reduce this tile's dy' * x_in.

    The Scalar Engine's identity activation accumulates dbeta into reduce_regs (its output goes
    to discard_tile) while the Vector Engine forms dy' * x_in and reduces it into a partial column.

    dbeta_readout is None except on the channel tile's last unit, where reduce_regs is evicted.
    """
    nisa.activation(
        dst=discard_tile,
        op=nl.copy,
        data=dy_tile,
        reduce_op=nl.add,
        reduce_res=dbeta_readout,
        reduce_cmd=nisa.reduce_cmd.reset_reduce if is_first_tile else nisa.reduce_cmd.reduce,
    )
    nisa.tensor_tensor(dst=dyx_tile, data1=dy_tile, data2=x_tile, op=nl.multiply)

    nisa.tensor_reduce(
        dst=bufs.yx_partials[:active_part, tile_idx : tile_idx + 1],
        op=nl.add,
        data=dyx_tile,
        axis=1,
    )


def _phase1_channel_tile_epilogue(
    dgamma: nl.NkiTensor,
    dbeta: nl.NkiTensor,
    bufs: _Phase1Buffers,
    means_col: nl.NkiTensor,
    rstd_col: nl.NkiTensor,
    dbeta_col: nl.NkiTensor,
    dgamma_col: nl.NkiTensor,
    c_start: int,
    c_end: int,
    c_size: int,
    rep: int,
    active_part: int,
    n_partials: int,
    shuffle_mask: list,
    shard_on_b: bool,
    prg_id: int,
    n_prgs: int,
) -> None:
    """
    Finish a channel tile's dbeta / dgamma and store both to HBM.

    Runs on the channel tile's last work unit. Collapses the per-band partials of both reductions
    down to [0, c_size), completes them across cores under batch sharding (each core reduced only
    its own batch slice), then forms dgamma = (sum(dy' * x_in) - means * dbeta) * rstd, which equals
    sum(dy' * x_hat). Both inputs are complete on every core by then, so each computes the same
    dgamma locally.
    """
    yx_col = bufs.yx_col[:c_size, 0:1]
    scratch_col = bufs.scratch_col[:c_size, 0:1]

    if rep > 1:
        _gather_bands_and_sum(dbeta_col, bufs.dbeta_wide, bufs.band_gather, c_size, rep, shuffle_mask)
        # Sum this channel tile's partial columns per band first, then across bands.
        nisa.tensor_reduce(
            dst=bufs.yx_col[:active_part, 0:1],
            op=nl.add,
            data=bufs.yx_partials[:active_part, 0:n_partials],
            axis=1,
        )
        # Collapses in place: yx_col is band 0 of the buffer being gathered. Safe by data
        # dependency, not by emission order — see _gather_bands_and_sum.
        _gather_bands_and_sum(yx_col, bufs.yx_col, bufs.band_gather, c_size, rep, shuffle_mask)
    else:
        nisa.tensor_reduce(dst=yx_col, op=nl.add, data=bufs.yx_partials[:c_size, 0:n_partials], axis=1)

    if shard_on_b:
        # Each core reduced only its own batch slice, so both columns are partial sums: swap with
        # the peer and add. Distinct pipe_ids keep the two exchanges independent, not one barrier.
        peer = n_prgs - 1 - prg_id
        recv_dbeta_col = bufs.recv_dbeta[:c_size, 0:1]
        recv_yx_col = bufs.recv_yx[:c_size, 0:1]
        lnc_sendrecv(src=dbeta_col, dst=recv_dbeta_col, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)
        lnc_sendrecv(src=yx_col, dst=recv_yx_col, send_to_rank=peer, recv_from_rank=peer, pipe_id=1)
        nisa.tensor_tensor(dst=dbeta_col, data1=dbeta_col, data2=recv_dbeta_col, op=nl.add)
        nisa.tensor_tensor(dst=yx_col, data1=yx_col, data2=recv_yx_col, op=nl.add)

    # dgamma = (sum(dy' * x_in) - means * dbeta) * rstd == sum(dy' * x_hat).
    nisa.tensor_tensor(dst=scratch_col, data1=means_col, data2=dbeta_col, op=nl.multiply)
    nisa.tensor_tensor(dst=dgamma_col, data1=yx_col, data2=scratch_col, op=nl.subtract)
    nisa.tensor_tensor(dst=dgamma_col, data1=dgamma_col, data2=rstd_col, op=nl.multiply)

    nisa.dma_copy(
        dst=dbeta.slice(dim=0, start=c_start, end=c_end, step=1),
        src=dbeta_col,
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )
    nisa.dma_copy(
        dst=dgamma.slice(dim=0, start=c_start, end=c_end, step=1),
        src=dgamma_col,
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )


def _batch_norm_bwd_direct_and_affine_phase(
    dy: nl.NkiTensor,
    x_in: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    eps: float,
    dgamma: nl.NkiTensor,
    dbeta: nl.NkiTensor,
    relu_mask: Optional[nl.NkiTensor],
    residuals_in: Optional[nl.NkiTensor],
    dresidual: Optional[nl.NkiTensor],
    masked_dy_hbm: nl.NkiTensor,
    sbm: SbufManager,
    means_all: nl.NkiTensor,
    gamma_all: nl.NkiTensor,
    rstd_all: nl.NkiTensor,
    gamma_rstd_all: nl.NkiTensor,
    dbeta_all: nl.NkiTensor,
    dgamma_all: nl.NkiTensor,
    masked_dy_slots: list,
    tile_walk: list,
    mem_cfg: BatchNormBwdMemoryConfig,
    n_partials: int,
    spatial: int,
    b_lo: int,
    owned_c_lo: int,
    P_MAX: int,
    fuse_relu: bool,
    recompute_relu_mask: bool,
    recompute_with_residual: bool,
    shard_on_b: bool,
    prg_id: int,
    n_prgs: int,
) -> None:
    """
    Phase 1: stage dy' to HBM and finish dbeta / dgamma.

    Per (batch, spatial) tile the Scalar and Vector engines pipeline against each other; see
    _phase1_tile_compute for that instruction sequence.

    dy / x_in / relu_mask are multibuffered over mem_cfg.interleave slots, as are the
    caller-owned dy' slots, so a tile's three loads and its dy' store overlap the
    previous tile's compute. dyx and the partial columns are not: dyx is consumed by the
    tensor_reduce in the same tile (no DMA waits on it) and the partials are written once per
    tile and read once at the end.

    The body is one flat loop over the shared tile walk. Everything not per-tile is delegated:
    _alloc_phase1_buffers takes the SBUF allocation, _phase1_channel_tile_prologue the statistics
    load and coefficient derivation, _phase1_mask_dy the fused ReLU's mask (loaded or recomputed),
    _phase1_tile_compute the per-tile engine pipeline, and _phase1_channel_tile_epilogue the
    dbeta / dgamma finalization and its cross-core exchange.

    The last mem_cfg.interleave tiles are the final write to each distinct dy' slot,
    so their dy' is still SBUF-resident when this phase ends. Their store to HBM is skipped
    and phase 2 consumes them straight from SBUF; every earlier tile still stores, because its
    slot gets overwritten before phase 2 runs.

    Partition replication: when a channel tile leaves partitions idle, one work unit covers
    `rep` consecutive batches at once, band r on partitions [r * c_size, (r+1) * c_size)
    (see _replication_factor). Consecutive batches sit a fixed stride apart in HBM, so all bands
    load in a single dma_copy per tensor: the [b_base, b_base + bands) x [c_start, c_end) x
    [sp_start, sp_end) HBM region is viewed with batch and channel merged into one partition axis.
    The dy' store is the same view in reverse. Since dbeta and sum(dy' * x_in) reduce over the
    batch, their per-band partials are gathered down to [0, c_size) and summed once the channel
    tile completes; gamma_rstd is broadcast to every band beforehand so the replicated
    activation has a scale for each active partition.
    """
    f_tile = mem_cfg.f_tile
    interleave = mem_cfg.interleave

    # The last `interleave` work units keep their dy' in SBUF for phase 2.
    total_tiles = len(tile_walk)
    resident_from = total_tiles - interleave
    shuffle_mask = _identity_shuffle_mask()

    sbm.open_scope(name="bn_bwd_affine")
    bufs = _alloc_phase1_buffers(
        sbm=sbm,
        x_dtype=x_in.dtype,
        f_tile=f_tile,
        interleave=interleave,
        n_partials=n_partials,
        n_c_tiles=means_all.shape[1],
        P_MAX=P_MAX,
        fuse_relu=fuse_relu,
        recompute_relu_mask=recompute_relu_mask,
        recompute_with_residual=recompute_with_residual,
        shard_on_b=shard_on_b,
        masked_dy_slots=masked_dy_slots,
    )
    _assert_sbuf_budget(sbm, mem_cfg.phase1_total_bytes, "phase 1")

    # Slot rotation and the partial-column index both span the whole walk; the walk is grouped by
    # channel tile, so a channel tile's partials are the contiguous run ending at its last unit.
    prev_c_tile_idx = -1
    tile_idx = 0
    for walk_idx in range(total_tiles):
        c_tile_idx = tile_walk[walk_idx][0]
        c_size = tile_walk[walk_idx][1]
        rep = tile_walk[walk_idx][2]
        b_base = tile_walk[walk_idx][3]
        sp_tile_idx = tile_walk[walk_idx][4]

        c_start = c_tile_idx * P_MAX
        c_end = c_start + c_size
        active_part = rep * c_size

        means_col = means_all[:c_size, c_tile_idx : c_tile_idx + 1]
        rstd_col = rstd_all[:c_size, c_tile_idx : c_tile_idx + 1]
        dbeta_col = dbeta_all[:c_size, c_tile_idx : c_tile_idx + 1]
        dgamma_col = dgamma_all[:c_size, c_tile_idx : c_tile_idx + 1]

        if c_tile_idx != prev_c_tile_idx:
            tile_idx = 0
            _phase1_channel_tile_prologue(
                means=means,
                variances=variances,
                gamma=gamma,
                beta=beta,
                eps=eps,
                means_all=means_all,
                gamma_all=gamma_all,
                rstd_all=rstd_all,
                gamma_rstd_all=gamma_rstd_all,
                bufs=bufs,
                c_tile_idx=c_tile_idx,
                c_start=c_start,
                c_end=c_end,
                c_size=c_size,
                rep=rep,
                shuffle_mask=shuffle_mask,
                recompute_relu_mask=recompute_relu_mask,
            )
            prev_c_tile_idx = c_tile_idx

        # Coefficient columns spanning the active bands (band r carries the same per-channel values).
        gamma_rstd_active = gamma_rstd_all[:active_part, c_tile_idx : c_tile_idx + 1]
        true_beta_active = None
        if recompute_relu_mask:
            true_beta_active = bufs.true_beta_all[:active_part, c_tile_idx : c_tile_idx + 1]

        sp_start = sp_tile_idx * f_tile
        sp_end = min(sp_start + f_tile, spatial)
        sp_size = sp_end - sp_start

        slot_idx = walk_idx % interleave
        dy_tile = bufs.dy_slots[slot_idx][:active_part, :sp_size]
        x_tile = bufs.x_slots[slot_idx][:active_part, :sp_size]
        dyx_tile = bufs.dyx[:active_part, :sp_size]

        # One DMA per tensor fills every band: batch and channel are merged into the partition
        # axis, so partition (r * c_size + c) is batch (b_base + r), channel (c_start + c).
        dy_view = _replicated_hbm_view(dy, b_base, rep, c_start, c_end, sp_start, sp_end)
        x_view = _replicated_hbm_view(x_in, b_base, rep, c_start, c_end, sp_start, sp_end)
        nisa.dma_copy(dst=dy_tile, src=dy_view, dge_mode=nisa.dge_mode.hwdge, engine=nki.isa.engine.sync)
        nisa.dma_copy(dst=x_tile, src=x_view, dge_mode=nisa.dge_mode.hwdge, engine=nki.isa.engine.sync)
        if fuse_relu:
            _phase1_mask_dy(
                relu_mask=relu_mask,
                residuals_in=residuals_in,
                bufs=bufs,
                dy_tile=dy_tile,
                x_tile=x_tile,
                gamma_rstd_active=gamma_rstd_active,
                true_beta_active=true_beta_active,
                slot_idx=slot_idx,
                active_part=active_part,
                sp_size=sp_size,
                band_extent=[b_base, rep, c_start, c_end, sp_start, sp_end],
                recompute_relu_mask=recompute_relu_mask,
            )
        if dresidual != None:
            dresidual_view = _replicated_hbm_view(
                dresidual,
                b_base,
                rep,
                c_start,
                c_end,
                sp_start,
                sp_end,
            )
            nisa.dma_copy(
                dst=dresidual_view,
                src=dy_tile,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nki.isa.engine.sync,
            )

        # dbeta = sum(dy') in reduce_regs, read out on the channel tile's last unit (per band under
        # replication) as marked by n_partials. The activation's own output goes to a discard tile.
        is_first_tile = tile_idx == 0
        is_last_of_c_tile = tile_idx == n_partials - 1
        dbeta_readout = None
        if is_last_of_c_tile:
            dbeta_readout = bufs.dbeta_wide[:active_part, 0:1] if rep > 1 else dbeta_col
        _phase1_tile_compute(
            bufs=bufs,
            dy_tile=dy_tile,
            x_tile=x_tile,
            dyx_tile=dyx_tile,
            discard_tile=bufs.discard[:active_part, :sp_size],
            dbeta_readout=dbeta_readout,
            is_first_tile=is_first_tile,
            active_part=active_part,
            tile_idx=tile_idx,
        )

        # Skip the store on each slot's last write: phase 2 reads those from SBUF. Earlier tiles
        # must store, since their slot is reused first.
        if walk_idx < resident_from:
            # masked_dy_hbm is core-local, so rebase the absolute indices to this core's origin.
            masked_dy_view = _replicated_hbm_view(
                masked_dy_hbm,
                b_base - b_lo,
                rep,
                c_start - owned_c_lo,
                c_end - owned_c_lo,
                sp_start,
                sp_end,
            )
            nisa.dma_copy(dst=masked_dy_view, src=dy_tile, dge_mode=nisa.dge_mode.hwdge, engine=nki.isa.engine.sync)
        tile_idx += 1

        if is_last_of_c_tile:
            _phase1_channel_tile_epilogue(
                dgamma=dgamma,
                dbeta=dbeta,
                bufs=bufs,
                means_col=means_col,
                rstd_col=rstd_col,
                dbeta_col=dbeta_col,
                dgamma_col=dgamma_col,
                c_start=c_start,
                c_end=c_end,
                c_size=c_size,
                rep=rep,
                active_part=active_part,
                n_partials=n_partials,
                shuffle_mask=shuffle_mask,
                shard_on_b=shard_on_b,
                prg_id=prg_id,
                n_prgs=n_prgs,
            )

    sbm.close_scope()


def _dx_tile_compute(
    x_tile: nl.NkiTensor,
    masked_dy_tile: nl.NkiTensor,
    var_x_tile: nl.NkiTensor,
    dx_tile: nl.NkiTensor,
    gamma_rstd_col: nl.NkiTensor,
    intermediate_col: nl.NkiTensor,
    var_x_bias_col: nl.NkiTensor,
) -> None:
    """
    One dx tile: var_x = x_in * intermediate + var_x_bias (float32), then
    dx = dy' * gamma_rstd - var_x.
    """
    nisa.activation(
        dst=var_x_tile,
        op=nl.copy,
        data=x_tile,
        scale=intermediate_col,
        bias=var_x_bias_col,
    )
    nisa.scalar_tensor_tensor(
        dst=dx_tile,
        data=masked_dy_tile,
        op0=nl.multiply,
        operand0=gamma_rstd_col,
        op1=nl.subtract,
        operand1=var_x_tile,
    )


def _batch_norm_bwd_dx_phase(
    x_in: nl.NkiTensor,
    dx: nl.NkiTensor,
    masked_dy_hbm: nl.NkiTensor,
    sbm: SbufManager,
    means_all: nl.NkiTensor,
    gamma_all: nl.NkiTensor,
    rstd_all: nl.NkiTensor,
    gamma_rstd_all: nl.NkiTensor,
    dbeta_all: nl.NkiTensor,
    dgamma_all: nl.NkiTensor,
    mean_x_all: nl.NkiTensor,
    intermediate_all: nl.NkiTensor,
    var_x_bias_all: nl.NkiTensor,
    masked_dy_slots: list,
    tile_walk: list,
    dx_dtype: nki.dtype,
    mem_cfg: BatchNormBwdMemoryConfig,
    C: int,
    spatial: int,
    inv_n: float,
    b_lo: int,
    owned_c_lo: int,
    c_tile_lo: int,
    c_tile_hi: int,
    P_MAX: int,
) -> None:
    """
    Phase 2: turn the staged dy' into dx.

    Reuses phase 1's SBUF (its scope was closed) for x_in, var_x and dx; the dy' slots are
    the shared, caller-owned ones phase 1 wrote. The two 1/N correction terms are folded into
    per-channel coefficients so each tile costs one activation plus one scalar_tensor_tensor:
        mean_x           = gamma * rstd * dbeta / N
        intermediate     = gamma * dgamma * rstd^2 / N
        var_x            = x_in * intermediate - means * intermediate + mean_x   (float32)
        dx               = dy' * gamma * rstd - var_x

    One loop over a rotated tile order. Phase 1 walked the same sequence (the shared tile_walk)
    and skipped the HBM store for its last `interleave` tiles, whose dy' is therefore still
    in SBUF, so the walk is visited tail-first: those resident units come first, before slot
    rotation reaches and overwrites them, then everything before them. The only per-unit
    difference is that a resident unit skips the dy' load and reads its slot directly.

    x_in / dx are multibuffered over mem_cfg.interleave slots so a tile's load and its dx
    store overlap the previous tile's compute. var_x is not: it is produced and consumed within
    one tile and no DMA waits on it.

    Partition replication mirrors phase 1: a work unit covers rep batches at once, all
    loaded and stored with one DMA each via the band-major HBM view. The three per-channel
    coefficients are broadcast to every band so the replicated activation and scalar_tensor_tensor
    have an operand value for each active partition.

    Needs no cross-core synchronization under either sharding: dx is elementwise in the staged
    dy', and every per-channel coefficient it reads was made complete on every core by the
    end of phase 1. Under batch sharding a core reads back exactly the dy' it staged itself,
    so the tile_walk's batch range is the only difference from the unsharded path.
    """
    f_tile = mem_cfg.f_tile
    # One shared interleave: tile t uses slot t % interleave in both phases, which is what makes the
    # resident handoff work.
    interleave = mem_cfg.interleave
    # Same sequence phase 1 walked, so the same tail is resident. Clamped at 0 because a short walk
    # leaves every unit in its own slot.
    total_tiles = len(tile_walk)
    resident_from = max(0, total_tiles - interleave)
    shuffle_mask = _identity_shuffle_mask()

    sbm.open_scope(name="bn_bwd_dx")
    # Rotating slots: slot s holds one x_in / dx tile. The dy' slots are the shared ones.
    x_slots = []
    dx_slots = []
    for slot_idx in range(interleave):
        x_slots.append(sbm.alloc_stack((P_MAX, f_tile), dtype=x_in.dtype, name=f"dx_x_in_tile_{slot_idx}"))
        dx_slots.append(sbm.alloc_stack((P_MAX, f_tile), dtype=dx_dtype, name=f"dx_tile_{slot_idx}"))
    var_x_sb = sbm.alloc_stack((P_MAX, f_tile), dtype=_VAR_X_DTYPE, name="var_x_tile")
    _assert_sbuf_budget(sbm, mem_cfg.phase2_total_bytes, "phase 2")

    # Uniform across channel tiles: replication requires c_size == C, so rep > 1 implies a single
    # channel tile. Guarded because batch sharding can leave a core an empty walk (b_lo == b_hi).
    rep = tile_walk[0][2] if total_tiles > 0 else 1

    # Computed up front: the tail-first visit order can reach a different channel tile than the rest.
    for c_tile_idx in range(c_tile_lo, c_tile_hi):
        c_start = c_tile_idx * P_MAX
        c_end = min(c_start + P_MAX, C)
        c_size = c_end - c_start

        means_col = means_all[:c_size, c_tile_idx : c_tile_idx + 1]
        gamma_col = gamma_all[:c_size, c_tile_idx : c_tile_idx + 1]
        rstd_col = rstd_all[:c_size, c_tile_idx : c_tile_idx + 1]
        gamma_rstd_col = gamma_rstd_all[:c_size, c_tile_idx : c_tile_idx + 1]
        dbeta_col = dbeta_all[:c_size, c_tile_idx : c_tile_idx + 1]
        dgamma_col = dgamma_all[:c_size, c_tile_idx : c_tile_idx + 1]
        mean_x_col = mean_x_all[:c_size, c_tile_idx : c_tile_idx + 1]
        intermediate_col = intermediate_all[:c_size, c_tile_idx : c_tile_idx + 1]
        var_x_bias_col = var_x_bias_all[:c_size, c_tile_idx : c_tile_idx + 1]

        # mean_x = gamma_rstd * dbeta / N
        nisa.tensor_scalar(
            dst=mean_x_col,
            data=dbeta_col,
            op0=nl.multiply,
            operand0=gamma_rstd_col,
            op1=nl.multiply,
            operand1=inv_n,
        )
        # intermediate = dgamma * rstd^2 * gamma / N, using rstd^2 so variance is only read via rstd.
        nisa.tensor_tensor(dst=intermediate_col, data1=rstd_col, data2=rstd_col, op=nl.multiply)
        nisa.tensor_tensor(dst=intermediate_col, data1=intermediate_col, data2=dgamma_col, op=nl.multiply)
        nisa.tensor_scalar(
            dst=intermediate_col,
            data=intermediate_col,
            op0=nl.multiply,
            operand0=gamma_col,
            op1=nl.multiply,
            operand1=inv_n,
        )
        # var_x_bias = mean_x - means * intermediate.
        nisa.tensor_scalar(
            dst=var_x_bias_col,
            data=intermediate_col,
            op0=nl.multiply,
            operand0=means_col,
            op1=nl.subtract,
            operand1=mean_x_col,
            reverse1=True,
        )
        # Every active partition needs its own copy of each coefficient under replication.
        _broadcast_to_bands(intermediate_all[:, c_tile_idx : c_tile_idx + 1], c_size, rep, shuffle_mask)
        _broadcast_to_bands(var_x_bias_all[:, c_tile_idx : c_tile_idx + 1], c_size, rep, shuffle_mask)

    # Visit the SBUF-resident tail first, then wrap to everything before it: the walk rotated by
    # resident_from. Only difference per unit is that a resident one skips the dy' load.
    for order_idx in range(total_tiles):
        walk_idx = (resident_from + order_idx) % total_tiles
        is_resident = walk_idx >= resident_from
        c_tile_idx = tile_walk[walk_idx][0]
        c_size = tile_walk[walk_idx][1]
        rep = tile_walk[walk_idx][2]
        b_base = tile_walk[walk_idx][3]
        sp_tile_idx = tile_walk[walk_idx][4]

        c_start = c_tile_idx * P_MAX
        c_end = c_start + c_size
        active_part = rep * c_size
        sp_start = sp_tile_idx * f_tile
        sp_end = min(sp_start + f_tile, spatial)
        sp_size = sp_end - sp_start

        slot_idx = walk_idx % interleave
        x_tile = x_slots[slot_idx][:active_part, :sp_size]
        masked_dy_tile = masked_dy_slots[slot_idx][:active_part, :sp_size]
        var_x_tile = var_x_sb[:active_part, :sp_size]
        dx_tile = dx_slots[slot_idx][:active_part, :sp_size]

        x_view = _replicated_hbm_view(x_in, b_base, rep, c_start, c_end, sp_start, sp_end)
        nisa.dma_copy(dst=x_tile, src=x_view, dge_mode=nisa.dge_mode.hwdge, engine=nki.isa.engine.sync)
        if not is_resident:
            # Not resident: phase 1 stored this tile. masked_dy_hbm is core-local, so rebase.
            masked_dy_view = _replicated_hbm_view(
                masked_dy_hbm,
                b_base - b_lo,
                rep,
                c_start - owned_c_lo,
                c_end - owned_c_lo,
                sp_start,
                sp_end,
            )
            nisa.dma_copy(
                dst=masked_dy_tile, src=masked_dy_view, dge_mode=nisa.dge_mode.hwdge, engine=nki.isa.engine.sync
            )

        _dx_tile_compute(
            x_tile=x_tile,
            masked_dy_tile=masked_dy_tile,
            var_x_tile=var_x_tile,
            dx_tile=dx_tile,
            gamma_rstd_col=gamma_rstd_all[:active_part, c_tile_idx : c_tile_idx + 1],
            intermediate_col=intermediate_all[:active_part, c_tile_idx : c_tile_idx + 1],
            var_x_bias_col=var_x_bias_all[:active_part, c_tile_idx : c_tile_idx + 1],
        )
        dx_view = _replicated_hbm_view(dx, b_base, rep, c_start, c_end, sp_start, sp_end)
        nisa.dma_copy(dst=dx_view, src=dx_tile, dge_mode=nisa.dge_mode.hwdge, engine=nki.isa.engine.sync)

    sbm.close_scope()
