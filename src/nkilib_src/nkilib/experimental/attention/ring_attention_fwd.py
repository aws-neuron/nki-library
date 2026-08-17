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

"""Ring Attention Forward pass kernel using attention_cte with HBM I/O and online softmax reduction."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.attention.attention_cte import (
    _MIN_SEQLEN_FOR_LNC2_SHARDING,
    _Q_GRP_SZ,
    _SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT,
    attention_cte,
)
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ...core.utils.modular_allocator import ModularAllocator
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

# Maximum number of Q groups to process per output tile in the reduction
# and normalization loops.
_MAX_GRPS_PER_TILE = 128
_FULLY_MASKED_LSE_THRESHOLD = -1.0e30


def _load_rank_sb(iota_nw, scalar_rank, qts):
    """
    Load a dynamic rank ID into a (qts, 1) float32 SBUF tensor.

    Uses indirect DMA with scalar_offset=scalar_rank to read iota_nw[scalar_rank]
    directly into SBUF, then stream_shuffle_broadcast to replicate across all
    partitions. This pattern (matching ring_attention_bwd) avoids register_store,
    whose scheduling can be reordered across ring steps on trn3 cp=4 lnc=2
    striped causal.

    Args:
        iota_nw (nl.NkiTensor): [1, num_workers] HBM iota table containing [0, 1, ..., num_workers-1].
        scalar_rank: Dynamic rank ID (index type from collective_permute).
        qts (int): Q sequence tile size (partition dimension).

    Returns:
        nl.NkiTensor: [qts, 1], Rank ID broadcast to all partitions in SBUF as float32.
    """
    sb = nl.ndarray((qts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=sb[0, 0],
        src=iota_nw.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=scalar_rank, indirect_dim=1),
    )
    stream_shuffle_broadcast(src=sb, dst=sb)
    return sb


def _reduce_attention_stats(
    neg_max_running,
    exp_sum_running,
    neg_max_curr,
    exp_sum_curr,
    allocator,
):
    """Compute combined softmax statistics and correction factors for all Q groups.

    Given the softmax stats (neg_max, sum) from two attention chunks, computes
    the global max, per-group correction factors, and combined sum using online
    softmax rescaling. All operations work on [128, num_grps] tensors — all
    groups are processed in a single set of ISA instructions.

    This is the first phase of the two-phase reduction in _reduce_one_batch.
    The second phase broadcasts the returned correction factors to [128, num_grps, d]
    and applies them to the output tiles via tensor_tensor.

    Formulas (using negated maxes as stored):
        neg_max_new    = min(neg_max_running, neg_max_curr)
        corr_running   = exp(neg_max_new - neg_max_running)
        corr_curr      = exp(neg_max_new - neg_max_curr)
        exp_sum_new    = exp_sum_running * corr_running + exp_sum_curr * corr_curr

    ISA instructions (8 total, independent of num_grps):
        1. tensor_tensor(min)   — neg_max_new
        2. tensor_tensor(sub)   — neg_max_new - neg_max_running
        3. activation(exp)      — corr_running
        4. tensor_tensor(sub)   — neg_max_new - neg_max_curr
        5. activation(exp)      — corr_curr
        6. tensor_tensor(mul)   — exp_sum_running * corr_running
        7. tensor_tensor(mul)   — exp_sum_curr * corr_curr
        8. tensor_tensor(add)   — exp_sum_new

    Args:
        neg_max_running (nl.NkiTensor): [128, num_grps], Running negated row max in SBUF.
        exp_sum_running (nl.NkiTensor): [128, num_grps], Running softmax denominator in SBUF.
        neg_max_curr (nl.NkiTensor): [128, num_grps], Current step negated row max in SBUF.
        exp_sum_curr (nl.NkiTensor): [128, num_grps], Current step softmax denominator in SBUF.
        allocator (ModularAllocator): SBUF allocator for temporary buffers.

    Returns:
        (neg_max_new, exp_sum_new, corr_running, corr_curr): All [128, num_grps] SBUF tensors.
    """
    S = neg_max_running.shape[0]  # 128
    G = neg_max_running.shape[1]  # num_grps

    neg_max_new = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    exp_sum_new = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    corr_running = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    corr_curr = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)

    # 1. Global max (negated): neg_max_new = min(neg_max_running, neg_max_curr)
    nisa.tensor_tensor(dst=neg_max_new, data1=neg_max_running, data2=neg_max_curr, op=nl.minimum)

    # 2. corr_running = exp(neg_max_new - neg_max_running)
    nisa.tensor_tensor(dst=corr_running, data1=neg_max_new, data2=neg_max_running, op=nl.subtract)
    nisa.activation(dst=corr_running, op=nl.exp, data=corr_running)

    # 3. corr_curr = exp(neg_max_new - neg_max_curr)
    nisa.tensor_tensor(dst=corr_curr, data1=neg_max_new, data2=neg_max_curr, op=nl.subtract)
    nisa.activation(dst=corr_curr, op=nl.exp, data=corr_curr)

    # 4. exp_sum_new = exp_sum_running * corr_running + exp_sum_curr * corr_curr
    nisa.tensor_tensor(dst=exp_sum_new, data1=exp_sum_running, data2=corr_running, op=nl.multiply)
    tmp = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    nisa.tensor_tensor(dst=tmp, data1=exp_sum_curr, data2=corr_curr, op=nl.multiply)
    nisa.tensor_tensor(dst=exp_sum_new, data1=exp_sum_new, data2=tmp, op=nl.add)

    return neg_max_new, exp_sum_new, corr_running, corr_curr


def _apply_correction_run(
    run_start,
    run_end,
    rows,
    max_grps_per_tile,
    batch_o_offset,
    sb_p,
    d,
    o_prev_hbm,
    o_curr_hbm,
    o_prev_sb,
    o_curr_sb,
    o_new,
    o_tmp,
    corr_running,
    corr_curr,
):
    """Apply online-softmax correction to output groups [run_start, run_end).

    Every group in the run has the same `rows` partition rows (sb_p for full groups,
    last_grp_rows for the partial final group). Tiled by max_grps_per_tile to bound
    SBUF usage. Defined at module scope because NKI does not allow nested functions.
    """
    for tile_start in range(run_start, run_end, max_grps_per_tile):
        tile_end = min(tile_start + max_grps_per_tile, run_end)
        tile_count = tile_end - tile_start
        tile_o_offset = batch_o_offset + tile_start * sb_p * d

        # Bulk DMA pattern for this tile
        o_tile_pat = [[d, rows], [128 * d, tile_count], [1, d]]

        # Load this tile's output groups into the front of the pre-allocated buffers
        nisa.dma_copy(dst=o_prev_sb[:rows, :tile_count, :], src=o_prev_hbm.ap(pattern=o_tile_pat, offset=tile_o_offset))
        nisa.dma_copy(dst=o_curr_sb[:rows, :tile_count, :], src=o_curr_hbm.ap(pattern=o_tile_pat, offset=tile_o_offset))

        # Broadcast correction factors for this tile's groups (clamp partition dim to rows
        # so the partial final group matches the (rows, tile_count, d) output view).
        corr_running_bc = (
            corr_running.slice(dim=0, start=0, end=rows)
            .slice(dim=1, start=tile_start, end=tile_end)
            .expand_dim(dim=2)
            .broadcast(dim=2, size=d)
        )
        corr_curr_bc = (
            corr_curr.slice(dim=0, start=0, end=rows)
            .slice(dim=1, start=tile_start, end=tile_end)
            .expand_dim(dim=2)
            .broadcast(dim=2, size=d)
        )

        # Views into the first tile_count groups of the pre-allocated buffers
        o_prev_view = o_prev_sb[:rows, :tile_count, :]
        o_curr_view = o_curr_sb[:rows, :tile_count, :]
        o_new_view = o_new[:rows, :tile_count, :]
        o_tmp_view = o_tmp[:rows, :tile_count, :]

        # o_new = o_prev * corr_running
        nisa.tensor_tensor(dst=o_new_view, data1=o_prev_view, data2=corr_running_bc, op=nl.multiply)
        # o_tmp = o_curr * corr_curr
        nisa.tensor_tensor(dst=o_tmp_view, data1=o_curr_view, data2=corr_curr_bc, op=nl.multiply)
        # o_new = o_new + o_tmp
        nisa.tensor_tensor(dst=o_new_view, data1=o_new_view, data2=o_tmp_view, op=nl.add)

        # Write back this tile
        nisa.dma_copy(dst=o_prev_hbm.ap(pattern=o_tile_pat, offset=tile_o_offset), src=o_new[:rows, :tile_count, :])


def _apply_normalize_run(
    run_start,
    run_end,
    rows,
    max_grps_per_tile,
    batch_o_offset,
    sb_p,
    d,
    o_prev_hbm,
    o_out,
    o_sb,
    sum_recip_sb,
):
    """Normalize output groups [run_start, run_end) by 1/S and write to o_out.

    Every group in the run has the same `rows` partition rows. o_prev_hbm and o_out
    share the (bs, seqlen, d) layout, so the same offset/pattern is used for load and
    store. Defined at module scope because NKI does not allow nested functions.
    """
    for tile_start in range(run_start, run_end, max_grps_per_tile):
        tile_end = min(tile_start + max_grps_per_tile, run_end)
        tile_count = tile_end - tile_start
        tile_o_offset = batch_o_offset + tile_start * sb_p * d

        # Bulk DMA pattern for this tile
        o_tile_pat = [[d, rows], [128 * d, tile_count], [1, d]]

        # Load this tile's output groups
        nisa.dma_copy(dst=o_sb[:rows, :tile_count, :], src=o_prev_hbm.ap(pattern=o_tile_pat, offset=tile_o_offset))

        # Broadcast 1/S for this tile's groups (clamp partition dim to rows so the
        # partial final group matches the (rows, tile_count, d) output view).
        sum_recip_bc = (
            sum_recip_sb.slice(dim=0, start=0, end=rows)
            .slice(dim=1, start=tile_start, end=tile_end)
            .expand_dim(dim=2)
            .broadcast(dim=2, size=d)
        )

        # Normalize: o_final = o_unnorm * (1/S)
        nisa.tensor_tensor(
            dst=o_sb[:rows, :tile_count, :],
            data1=o_sb[:rows, :tile_count, :],
            data2=sum_recip_bc,
            op=nl.multiply,
        )

        # Write to final output
        nisa.dma_copy(dst=o_out.ap(pattern=o_tile_pat, offset=tile_o_offset), src=o_sb[:rows, :tile_count, :])


def _reduce_one_batch(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    batch_idx,
    grp_start,
    grp_end,
    seqlen,
    d,
    num_grps,
    sb_p,
    softmax_pat,
    max_grps_per_tile,
    neg_max_prev_sb,
    sum_prev_sb,
    neg_max_curr_sb,
    sum_curr_sb,
    o_prev_sb,
    o_curr_sb,
    o_new,
    o_tmp,
    batch_loop_addr,
    allocator,
):
    """Reduce softmax stats and apply correction for one batch over [grp_start, grp_end).

    Loads softmax statistics from HBM, computes correction factors via
    _reduce_attention_stats, then applies corrections to output groups using
    nl.NkiTensor broadcast. Output groups are processed in tiles of up to
    max_grps_per_tile to bound SBUF usage.

    Args:
        o_prev_hbm (nl.NkiTensor): [bs, seqlen, d], Accumulated unnormalized output in HBM.
        neg_max_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Negated row max for accumulated output.
        sum_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Raw softmax denominator S for accumulated output.
        o_curr_hbm (nl.NkiTensor): [bs, seqlen, d], Current step unnormalized output in HBM.
        neg_max_curr_hbm (nl.NkiTensor): [bs, 128, num_grps], Negated row max for current step.
        sum_curr_hbm (nl.NkiTensor): [bs, 128, num_grps], Raw softmax denominator S for current step.
        batch_idx (int): Batch index to process.
        grp_start (int): First Q group to reduce (inclusive).
        grp_end (int): Last Q group to reduce (exclusive).
        seqlen (int): Real query sequence length (may not be a multiple of _Q_GRP_SZ).
        d (int): Head dimension size.
        num_grps (int): Total number of Q groups (ceil(seqlen / _Q_GRP_SZ)).
        sb_p (int): Partition tile size (128).
        softmax_pat: Array pattern for softmax stat tensors.
        max_grps_per_tile (int): Max groups per output tile (caps SBUF usage).
        neg_max_prev_sb (nl.NkiTensor): Pre-allocated SBUF for prev neg_max.
        sum_prev_sb (nl.NkiTensor): Pre-allocated SBUF for prev sum.
        neg_max_curr_sb (nl.NkiTensor): Pre-allocated SBUF for curr neg_max.
        sum_curr_sb (nl.NkiTensor): Pre-allocated SBUF for curr sum.
        o_prev_sb (nl.NkiTensor): [sb_p, max_grps_per_tile, d] Pre-allocated SBUF for prev output.
        o_curr_sb (nl.NkiTensor): [sb_p, max_grps_per_tile, d] Pre-allocated SBUF for curr output.
        o_new (nl.NkiTensor): [sb_p, max_grps_per_tile, d] Pre-allocated SBUF for corrected output.
        o_tmp (nl.NkiTensor): [sb_p, max_grps_per_tile, d] Pre-allocated SBUF for temporary.
        batch_loop_addr (int): Allocator checkpoint to reset temporaries.
        allocator (ModularAllocator): SBUF allocator for temporaries.
    """
    allocator.set_current_address(batch_loop_addr)

    grp_count = grp_end - grp_start

    """
    Final Q group may be partial when seqlen is not a multiple of _Q_GRP_SZ.
    The output tensors are (bs, seqlen, d), so the batch stride is seqlen * d and
    the last group occupies only last_grp_rows (<= sb_p) rows. Stats tensors stay
    (bs, sb_p, num_grps); their padding rows are independent per partition and unused.
    """
    last_grp_rows = seqlen - (num_grps - 1) * sb_p

    batch_softmax_offset = batch_idx * sb_p * num_grps
    batch_o_offset = batch_idx * seqlen * d

    nisa.dma_copy(dst=neg_max_prev_sb, src=neg_max_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
    nisa.dma_copy(dst=sum_prev_sb, src=sum_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
    nisa.dma_copy(dst=neg_max_curr_sb, src=neg_max_curr_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
    nisa.dma_copy(dst=sum_curr_sb, src=sum_curr_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))

    # Stats reduction — all groups at once on [sb_p, num_grps] tensors
    neg_max_new, exp_sum_new, corr_running, corr_curr = _reduce_attention_stats(
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb,
        allocator,
    )

    # Full (sb_p-row) groups first, then the partial final group if this shard owns it.
    has_partial = last_grp_rows < sb_p
    full_grp_end = (num_grps - 1) if has_partial else num_grps
    _apply_correction_run(
        grp_start,
        min(grp_end, full_grp_end),
        sb_p,
        max_grps_per_tile,
        batch_o_offset,
        sb_p,
        d,
        o_prev_hbm,
        o_curr_hbm,
        o_prev_sb,
        o_curr_sb,
        o_new,
        o_tmp,
        corr_running,
        corr_curr,
    )
    if has_partial and grp_end == num_grps:
        _apply_correction_run(
            num_grps - 1,
            num_grps,
            last_grp_rows,
            max_grps_per_tile,
            batch_o_offset,
            sb_p,
            d,
            o_prev_hbm,
            o_curr_hbm,
            o_prev_sb,
            o_curr_sb,
            o_new,
            o_tmp,
            corr_running,
            corr_curr,
        )

    # Write back stats — only the groups we processed
    wb_softmax_pat = [[num_grps, sb_p], [1, grp_count]]
    nisa.dma_copy(
        dst=neg_max_prev_hbm.ap(pattern=wb_softmax_pat, offset=batch_softmax_offset + grp_start),
        src=neg_max_new.ap(pattern=wb_softmax_pat, offset=grp_start),
    )
    nisa.dma_copy(
        dst=sum_prev_hbm.ap(pattern=wb_softmax_pat, offset=batch_softmax_offset + grp_start),
        src=exp_sum_new.ap(pattern=wb_softmax_pat, offset=grp_start),
    )


def _tiled_reduce_attention(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    bs,
    d,
    num_grps,
    seqlen,
    num_bs_per_shard,
    bs_offset,
    has_remainder,
    shard_id,
    allocator,
):
    """Combine attention outputs from two ring steps using bulk DMA and broadcast compute.

    Two-phase approach per batch:
      1. Reduce softmax statistics (neg_max, sum) and compute correction factors
         for all Q groups at once via _reduce_attention_stats on (128, num_grps)
         tensors.
      2. Apply correction factors to output groups in tiles of up to
         _MAX_GRPS_PER_TILE groups. Each tile is loaded via bulk DMA as
         [128, tile_count, d], correction factors are broadcast from
         [128, tile_count] to [128, tile_count, d], and element-wise ops
         process the tile in one shot. This tiling bounds SBUF usage while
         still processing many groups per DMA/compute round.

    For the LNC2 remainder batch (odd bs), attention_cte uses seqlen sharding
    where each NC computes a disjoint set of Q groups. Since each group is
    independent, each NC reduces only its own groups for the remainder batch.

    Args:
        o_prev_hbm (nl.NkiTensor): [bs, seqlen, d], Accumulated unnormalized output in HBM.
        neg_max_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Negated row max for accumulated output.
        sum_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Raw softmax denominator S for accumulated output.
        o_curr_hbm (nl.NkiTensor): [bs, seqlen, d], Current step unnormalized output in HBM.
        neg_max_curr_hbm (nl.NkiTensor): [bs, 128, num_grps], Negated row max for current step.
        sum_curr_hbm (nl.NkiTensor): [bs, 128, num_grps], Raw softmax denominator S for current step.
        bs (int): Total batch size (b * q_h).
        d (int): Head dimension size.
        num_grps (int): Number of Q groups (ceil(seqlen / _Q_GRP_SZ)).
        seqlen (int): Real query sequence length (may not be a multiple of _Q_GRP_SZ).
        num_bs_per_shard (int): Number of batches this shard processes (excluding remainder).
        bs_offset (int): Starting batch index for this shard.
        has_remainder (bool): Whether there is a remainder batch (odd bs with LNC2).
        shard_id (int): Current LNC shard ID (0 or 1).
        allocator (ModularAllocator): SBUF allocator for temporaries.

    Notes:
        After this call, o_prev_hbm/neg_max_prev_hbm/sum_prev_hbm hold the combined result.
    """
    sb_p = nl.tile_size.pmax  # 128

    # Save allocator checkpoint — all temporaries freed after this function
    init_addr = allocator.get_current_address()

    # Pre-allocate SBUF buffers (reused across batch iterations)
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    sum_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)

    softmax_pat = [[num_grps, sb_p], [1, num_grps]]

    # Cap output tile size to bound SBUF usage.
    max_grps_per_tile = min(num_grps, _MAX_GRPS_PER_TILE)

    # Pre-allocate output buffers: [sb_p, max_grps_per_tile, d] tensors.
    # Reused across batch iterations and group tiles.
    o_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, max_grps_per_tile, d), dtype=nl.bfloat16)
    o_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, max_grps_per_tile, d), dtype=nl.bfloat16)
    o_new = allocator.alloc_sbuf_tensor(shape=(sb_p, max_grps_per_tile, d), dtype=nl.bfloat16)
    o_tmp = allocator.alloc_sbuf_tensor(shape=(sb_p, max_grps_per_tile, d), dtype=nl.bfloat16)

    # Checkpoint after pre-allocated buffers — _reduce_attention_stats allocates
    # temporaries that must be freed after each batch iteration.
    batch_loop_addr = allocator.get_current_address()

    # Process batches assigned to this shard (all groups)
    for batch_local_idx in range(num_bs_per_shard):
        _reduce_one_batch(
            o_prev_hbm,
            neg_max_prev_hbm,
            sum_prev_hbm,
            o_curr_hbm,
            neg_max_curr_hbm,
            sum_curr_hbm,
            batch_local_idx + bs_offset,
            0,
            num_grps,
            seqlen,
            d,
            num_grps,
            sb_p,
            softmax_pat,
            max_grps_per_tile,
            neg_max_prev_sb,
            sum_prev_sb,
            neg_max_curr_sb,
            sum_curr_sb,
            o_prev_sb,
            o_curr_sb,
            o_new,
            o_tmp,
            batch_loop_addr,
            allocator,
        )

    """
    Handle LNC2 remainder batch, mirroring attention_cte's remainder sharding:
      - seqlen >= _MIN_SEQLEN_FOR_LNC2_SHARDING: both NCs seqlen-shard the batch, each
        reducing only its own groups (no cross-core reads).
      - seqlen <  _MIN_SEQLEN_FOR_LNC2_SHARDING: attention_cte has core 0 compute the
        whole batch, so core 0 also reduces the whole batch here. Splitting it would make
        core 1 read core 0's current-step output with no inter-core barrier (data race).
    """
    if has_remainder:
        if seqlen >= _MIN_SEQLEN_FOR_LNC2_SHARDING:
            batch_0_grp = int(num_grps * _SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT)
            remainder_grp_start = shard_id * batch_0_grp
            remainder_grp_end = batch_0_grp if shard_id == 0 else num_grps
            do_remainder = True
        else:
            remainder_grp_start = 0
            remainder_grp_end = num_grps
            do_remainder = shard_id == 0
        if do_remainder:
            _reduce_one_batch(
                o_prev_hbm,
                neg_max_prev_hbm,
                sum_prev_hbm,
                o_curr_hbm,
                neg_max_curr_hbm,
                sum_curr_hbm,
                bs - 1,
                remainder_grp_start,
                remainder_grp_end,
                seqlen,
                d,
                num_grps,
                sb_p,
                softmax_pat,
                max_grps_per_tile,
                neg_max_prev_sb,
                sum_prev_sb,
                neg_max_curr_sb,
                sum_curr_sb,
                o_prev_sb,
                o_curr_sb,
                o_new,
                o_tmp,
                batch_loop_addr,
                allocator,
            )

    # Free all temporaries
    allocator.set_current_address(init_addr)


def _compute_cp_offset(rank_id_sb, recv_rank_sb, cp_offset_hbm, seqlen, striped_input):
    """Compute cp_offset for a ring step and write it to HBM.

    For striped mode: cp_offset = 0 if q_rank >= kv_rank, else -1.
    For contiguous mode: cp_offset = (my_rank - recv_rank) * seqlen.

    Args:
        rank_id_sb (nl.NkiTensor): [_Q_GRP_SZ, 1], Current rank ID broadcast in SBUF.
        recv_rank_sb (nl.NkiTensor): [_Q_GRP_SZ, 1], Received rank ID broadcast in SBUF.
        cp_offset_hbm (nl.NkiTensor): [1, 1], Output HBM tensor for attention_cte.
        seqlen (int): Sequence length (used in contiguous mode).
        striped_input (bool): Whether input is striped across ranks.
    """
    if striped_input:
        # cp_offset = -float(q_rank < kv_rank) (i.e., 0.0 or -1.0)
        cp_off_sb = nl.ndarray((_Q_GRP_SZ, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(cp_off_sb, rank_id_sb, recv_rank_sb, op=nl.less)
        nisa.tensor_scalar(cp_off_sb, cp_off_sb, nl.multiply, -1.0)
    else:
        # cp_offset = (my_rank - recv_rank) * seqlen
        cp_off_sb = nl.ndarray((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(cp_off_sb, rank_id_sb, recv_rank_sb, op=nl.subtract)
        nisa.tensor_scalar(cp_off_sb, cp_off_sb, nl.multiply, float(seqlen))
    nisa.dma_copy(dst=cp_offset_hbm, src=cp_off_sb[0, 0])


def _normalize_one_batch(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_out,
    lse_out,
    batch_idx,
    seqlen,
    d,
    num_grps,
    training,
    lse_dtype,
    allocator,
    grp_start=0,
    grp_end=-1,
):
    """Normalize and write output for a single batch (or a subset of its groups).

    Loads all groups' output tiles in a single bulk DMA, applies normalization
    using nl.NkiTensor broadcast, and writes to final output.

    Args:
        o_prev_hbm (nl.NkiTensor): [bs, seqlen, d], Accumulated unnormalized output in HBM.
        neg_max_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Negated row max.
        sum_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Raw softmax denominator S.
        o_out (nl.NkiTensor): [bs, seqlen, d], Final output tensor in HBM.
        lse_out (nl.NkiTensor or None): [bs, 128, num_grps], LSE output (if training).
        batch_idx (int): Batch index to normalize.
        seqlen (int): Real query sequence length (may not be a multiple of _Q_GRP_SZ).
        d (int): Head dimension size.
        num_grps (int): Number of Q groups (ceil(seqlen / _Q_GRP_SZ)).
        training (bool): Whether to compute LSE.
        lse_dtype: Data type for LSE output.
        allocator (ModularAllocator): SBUF allocator for temporaries.
        grp_start (int): First group to normalize (default 0).
        grp_end (int): One past last group to normalize (default -1 = num_grps).
    """
    if grp_end < 0:
        grp_end = num_grps
    grp_count = grp_end - grp_start
    sb_p = nl.tile_size.pmax
    softmax_pat = [[num_grps, sb_p], [1, num_grps]]

    """
    Final Q group may be partial when seqlen is not a multiple of _Q_GRP_SZ.
    Output tensors are (bs, seqlen, d): batch stride is seqlen * d and the last
    group has only last_grp_rows (<= sb_p) valid rows.
    """
    last_grp_rows = seqlen - (num_grps - 1) * sb_p

    batch_softmax_offset = batch_idx * sb_p * num_grps
    batch_o_offset = batch_idx * seqlen * d

    init_addr = allocator.get_current_address()

    # Load sum (raw S) from HBM
    sum_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    nisa.dma_copy(dst=sum_sb, src=sum_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))

    # Compute 1/S for normalization
    sum_recip_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    nisa.reciprocal(sum_recip_sb, sum_sb)

    # Tiled normalization — process up to _MAX_GRPS_PER_TILE groups per iteration
    max_grps_per_tile = min(grp_count, _MAX_GRPS_PER_TILE)
    o_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, max_grps_per_tile, d), dtype=nl.bfloat16)

    # Full (sb_p-row) groups first, then the partial final group if this shard owns it.
    has_partial = last_grp_rows < sb_p
    full_grp_end = (num_grps - 1) if has_partial else num_grps
    _apply_normalize_run(
        grp_start,
        min(grp_end, full_grp_end),
        sb_p,
        max_grps_per_tile,
        batch_o_offset,
        sb_p,
        d,
        o_prev_hbm,
        o_out,
        o_sb,
        sum_recip_sb,
    )
    if has_partial and grp_end == num_grps:
        _apply_normalize_run(
            num_grps - 1,
            num_grps,
            last_grp_rows,
            max_grps_per_tile,
            batch_o_offset,
            sb_p,
            d,
            o_prev_hbm,
            o_out,
            o_sb,
            sum_recip_sb,
        )

    # Write LSE if training
    # lse = -neg_max + log(S)
    if training and lse_out != None:
        neg_max_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
        nisa.dma_copy(dst=neg_max_sb, src=neg_max_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
        # sum_sb already loaded above

        lse_tile = nl.ndarray((sb_p, num_grps), dtype=lse_dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(lse_tile, neg_max_sb, nl.multiply, -1.0)
        log_s = nl.ndarray((sb_p, num_grps), dtype=lse_dtype, buffer=nl.sbuf)
        nisa.activation(log_s, nl.log, sum_sb)
        nisa.tensor_tensor(lse_tile, lse_tile, log_s, op=nl.add)

        """
        Zero the partial final group's padding rows. Those rows correspond to no real
        query position; their raw stats (neg_max=_FLOAT32_MIN, sum=0) would otherwise
        yield -inf/NaN. Zeroing makes the LSE output deterministic and comparable.
        """
        if has_partial and grp_end == num_grps:
            nisa.memset(lse_tile[last_grp_rows:sb_p, num_grps - 1 : num_grps], 0.0)

        """
        A fully masked row retains the LSE sentinel, which overflows when backward reconstructs probabilities.
        Zeroing that LSE lets the existing score mask produce zero probability.
        """
        nisa.scalar_tensor_tensor(
            dst=lse_tile,
            data=lse_tile,
            op0=nl.greater,
            operand0=_FULLY_MASKED_LSE_THRESHOLD,
            op1=nl.multiply,
            operand1=lse_tile,
        )

        # Write only our groups' LSE using ap() on the raw tensor
        lse_ap_pat = [[num_grps, sb_p], [1, grp_count]]
        lse_batch_offset = batch_idx * sb_p * num_grps
        nisa.dma_copy(
            dst=lse_out.ap(pattern=lse_ap_pat, offset=lse_batch_offset + grp_start),
            src=lse_tile.ap(pattern=lse_ap_pat, offset=grp_start),
        )

    allocator.set_current_address(init_addr)


def _normalize_and_write_output(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_out,
    lse_out,
    bs,
    d,
    num_grps,
    seqlen,
    num_bs_per_shard,
    bs_offset,
    has_remainder,
    shard_id,
    training,
    lse_dtype,
    allocator,
):
    """Normalize accumulated unnormalized output and write to final output tensor.

    Also computes LSE = -neg_max + log(S) if training.
    Each NC has the full result in shared HBM (attention_cte uses default LNC sharding).
    Only writes this shard's assigned batches to the shared output tensors.
    For the LNC2 remainder batch (odd bs), each NC writes only its own
    seqlen groups, matching attention_cte's seqlen sharding split.

    Args:
        o_prev_hbm (nl.NkiTensor): [bs, seqlen, d], Accumulated unnormalized output in private HBM.
        neg_max_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Negated row max in private HBM.
        sum_prev_hbm (nl.NkiTensor): [bs, 128, num_grps], Raw softmax denominator S in private HBM.
        o_out (nl.NkiTensor): [bs, seqlen, d], Final output tensor in shared HBM.
        lse_out (nl.NkiTensor or None): [bs, 128, num_grps], LSE output in shared HBM (if training).
        bs (int): Total batch size.
        d (int): Head dimension size.
        num_grps (int): Number of Q groups (ceil(seqlen / _Q_GRP_SZ)).
        seqlen (int): Real query sequence length (may not be a multiple of _Q_GRP_SZ).
        num_bs_per_shard (int): Number of batches this shard processes (excluding remainder).
        bs_offset (int): Starting batch index for this shard.
        has_remainder (bool): Whether there is a remainder batch.
        shard_id (int): Current LNC shard ID.
        training (bool): Whether to compute LSE.
        lse_dtype: Data type for LSE output.
        allocator (ModularAllocator): SBUF allocator for temporaries.
    """
    # Process batches assigned to this shard
    for batch_local_idx in range(num_bs_per_shard):
        _normalize_one_batch(
            o_prev_hbm,
            neg_max_prev_hbm,
            sum_prev_hbm,
            o_out,
            lse_out,
            batch_local_idx + bs_offset,
            seqlen,
            d,
            num_grps,
            training,
            lse_dtype,
            allocator,
        )

    """
    Handle LNC2 remainder batch, mirroring attention_cte's remainder sharding:
      - seqlen >= _MIN_SEQLEN_FOR_LNC2_SHARDING: both NCs seqlen-shard, each normalizing
        and writing only its own groups.
      - seqlen <  _MIN_SEQLEN_FOR_LNC2_SHARDING: core 0 computed the whole batch, so core 0
        normalizes the whole batch (avoids core 1 reading core 0's output unsynchronized).
    """
    if has_remainder:
        if seqlen >= _MIN_SEQLEN_FOR_LNC2_SHARDING:
            batch_0_grp = int(num_grps * _SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT)
            remainder_grp_start = shard_id * batch_0_grp
            remainder_grp_end = batch_0_grp if shard_id == 0 else num_grps
            do_remainder = True
        else:
            remainder_grp_start = 0
            remainder_grp_end = num_grps
            do_remainder = shard_id == 0
        if do_remainder:
            _normalize_one_batch(
                o_prev_hbm,
                neg_max_prev_hbm,
                sum_prev_hbm,
                o_out,
                lse_out,
                bs - 1,
                seqlen,
                d,
                num_grps,
                training,
                lse_dtype,
                allocator,
                grp_start=remainder_grp_start,
                grp_end=remainder_grp_end,
            )


def _init_kv_send_buffers(dst_k, dst_v, src_k, src_v, bs, num_bs_per_shard, bs_offset, has_remainder, shard_id):
    """Copy local K/V into send buffers with LNC-sharded DMA work (one-time init).

    Each NC copies only its assigned batches, parallelizing the DMA work under LNC2.
    For the remainder batch (odd bs), shard 0 handles it.

    Args:
        dst_k (nl.NkiTensor): Destination K buffer in shared HBM. Layout matches input k.
        dst_v (nl.NkiTensor): [bs, seqlen, d], Destination V buffer in shared HBM.
        src_k (nl.NkiTensor): Source K buffer. Layout matches input k.
        src_v (nl.NkiTensor): [bs, seqlen, d], Source V buffer.
        bs (int): Total batch size.
        num_bs_per_shard (int): Number of batches this shard processes (excluding remainder).
        bs_offset (int): Starting batch index for this shard.
        has_remainder (bool): Whether there is a remainder batch (odd bs with LNC2).
        shard_id (int): Current LNC shard ID (0 or 1).
    """
    for batch_local_idx in range(num_bs_per_shard):
        batch_idx = batch_local_idx + bs_offset
        nisa.dma_copy(
            dst=dst_k.select(dim=0, index=batch_idx),
            src=src_k.select(dim=0, index=batch_idx),
        )
        nisa.dma_copy(
            dst=dst_v.select(dim=0, index=batch_idx),
            src=src_v.select(dim=0, index=batch_idx),
        )
    if has_remainder and shard_id == 0:
        last_batch = bs - 1
        nisa.dma_copy(
            dst=dst_k.select(dim=0, index=last_batch),
            src=src_k.select(dim=0, index=last_batch),
        )
        nisa.dma_copy(
            dst=dst_v.select(dim=0, index=last_batch),
            src=src_v.select(dim=0, index=last_batch),
        )


@nki.jit
def ring_attention_spmd_fwd(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    replica_groups: tuple = None,
    num_workers: int = 1,
    softmax_scale: float = None,
    use_causal_mask: bool = False,
    striped_input: bool = False,
    training: bool = False,
    lse_dtype: nki.dtype = nl.float32,
    tp_q: bool = False,
    tp_k: bool = False,
    bound_min: nl.NkiTensor = None,
    bound_max: nl.NkiTensor = None,
):
    """
    Ring attention forward using attention_cte with HBM I/O.

    Implements ring attention for context parallelism across multiple workers.
    All Q/K/V/O stay in HBM. After each attention_cte call, the output is
    corrected via a tiled HBM roundtrip (load one group at a time into SBUF,
    apply online softmax correction, write back). The collective permute runs
    in parallel with the correction for latency hiding.

    LNC sharding and batch iteration are handled internally by attention_cte,
    so this kernel does not manage LNC explicitly for the attention compute.
    However, the K/V DMA transfers (initial copy and per-step swap) are
    sharded across NCs so each NC handles only its assigned batches,
    parallelizing the DMA work under LNC2.

    Dimensions:
        b: Batch size
        h: Number of attention heads
        d: Head dimension (must be <= 128)
        seqlen: Sequence length. For training=False, arbitrary seqlen is supported (the
            final Q group may be partial). For training=True, must be divisible by _Q_GRP_SZ.

    Args:
        q (nl.NkiTensor): Query tensor. Shape depends on tp_q:
            [b, h, seqlen, d] when tp_q=True (non-transposed, kernel transposes internally via dma_transpose).
            [b, h, d, seqlen] when tp_q=False (pre-transposed layout).
        k (nl.NkiTensor): Key tensor. Shape depends on tp_k:
            [b, h, seqlen, d] when tp_k=True (non-transposed, kernel transposes internally via dma_transpose).
            [b, h, d, seqlen] when tp_k=False (pre-transposed layout).
        v (nl.NkiTensor): [b, h, seqlen, d], Value tensor (non-transposed layout).
        replica_groups (list): Replica groups for collective communication.
        num_workers (int): Number of workers in the ring.
        softmax_scale (float): Softmax scale factor. Default: 1/sqrt(d).
            When use_causal_mask=True, the caller must pre-scale Q by
            softmax_scale before calling this kernel, and pass
            softmax_scale=1.0 (attention_cte requires scale=1.0 in CP mode).
        use_causal_mask (bool): Whether to apply causal masking.
        striped_input (bool): Whether input is striped (requires use_causal_mask=True).
        training (bool): Whether to output LSE for backward pass. Default: False.
        lse_dtype (nki.dtype): Data type for the LSE output tensor. Default: nl.float32.
        tp_q (bool): Query tensor transpose flag. When True, q is in non-transposed layout
            (batch, seqlen, d) and attention_cte will transpose internally via dma_transpose.
            When False (default), q must be pre-transposed to (batch, d, seqlen).
        tp_k (bool): Key tensor transpose flag. When True, k is in non-transposed layout
            (batch, seqlen, d) and attention_cte will transpose internally via dma_transpose.
            When False (default), k must be pre-transposed to (batch, d, seqlen).
            The ring transfer buffers match the input k layout, so collective permute
            transfers k in whichever layout the caller provides.
        bound_min (nl.NkiTensor, optional): Sequence packing lower bound. Shape
            (b*q_h, seqlen_per_rank, 1), fp32. Per-local-Q-token inclusive lower bound
            on the local K index of the document this Q token belongs to. Must be
            provided together with bound_max. Requires use_causal_mask=True and
            striped_input=True. Under the invariant that each padded document length
            is a multiple of num_workers, the local document layout is identical on
            every rank, so the caller produces this tensor once and replicates it to
            every rank. Use cu_seqlens_to_striped_bounds() to build it from cu_seqlens.
        bound_max (nl.NkiTensor, optional): Sequence packing upper bound (exclusive).
            Same shape/dtype/semantics as bound_min.

    Returns:
        o (nl.NkiTensor): [b, h, seqlen, d], Attention output.
        lse (nl.NkiTensor): [b, h, 128, seqlen//128], Log-sum-exp (if training).

    Notes:
        - Requires Trainium2 or later (not supported on trn1)
        - MHA only (q_heads == kv_heads, must broadcast before calling)
        - Supports LNC1 and LNC2 (sharded on batch * heads by attention_cte)
        - Supports non-causal and causal attention
        - Supports striped causal masking (via cp_offset adjustment: 0 or -1)
        - Tested with up to 16k seqlen per rank (1M total sequence on trn2/trn3)

    Pseudocode:
        # Step 0: Run local attention on own Q, K, V
        o_prev, stats_prev = attention_cte(Q, K_local, V_local, unnormalized=True)

        # Steps 1..num_workers-1: Ring loop
        for step in range(1, num_workers):
            K_recv, V_recv = collective_permute(K, V)  # async
            o_curr, stats_curr = attention_cte(Q, K_recv, V_recv, unnormalized=True)
            o_prev, stats_prev = reduce_attention(o_prev, stats_prev, o_curr, stats_curr)

        # Normalize final output
        o_final = o_prev / sum_prev
    """
    kernel_assert(nisa.get_nc_version() > nisa.nc_version.gen2, "ring_attention_spmd_fwd not supported on trn1")

    if tp_q:
        b, q_h, seqlen, d = q.shape
    else:
        b, q_h, d, seqlen = q.shape

    _, k_h, _, _ = k.shape

    kernel_assert(q_h == k_h, "expects q_heads == kv_heads, broadcast before calling")
    kernel_assert(d <= 128, f"head_dim must be <= 128, got {d}")
    """
    Arbitrary seqlen is supported for both inference and training: the final Q group is
    handled as a ragged (< _Q_GRP_SZ rows) group in the reduction/normalization. For
    training, the LSE of the partial final group's padding rows is zeroed (see
    _normalize_one_batch) so the output is deterministic.

    Sequence packing: bound_min and bound_max must be provided together. Both
    must reference LOCAL (per-rank) positions. Under striped input with each
    padded doc length a multiple of num_workers, the local document layout is
    identical across ranks, so the caller can produce the bounds once and
    replicate them to every rank.
    """
    is_sequence_packed = bound_min != None
    kernel_assert(
        is_sequence_packed == (bound_max != None),
        "bound_min and bound_max must both be provided or both be None",
    )
    if is_sequence_packed:
        kernel_assert(use_causal_mask, "bound_min/bound_max require use_causal_mask=True")
        kernel_assert(striped_input, "bound_min/bound_max require striped_input=True")

    if replica_groups == None:
        replica_groups = ()

    bs = b * q_h
    num_grps = div_ceil(seqlen, _Q_GRP_SZ)

    # Reshape, attention_cte treats dim 0 as batch
    if tp_q:
        q = q.reshape((bs, seqlen, d))
    else:
        q = q.reshape((bs, d, seqlen))
    if tp_k:
        k = k.reshape((bs, seqlen, d))
    else:
        k = k.reshape((bs, d, seqlen))
    v = v.reshape((bs, seqlen, d))

    # Final output tensors
    o = nl.ndarray((bs, seqlen, d), dtype=q.dtype, buffer=nl.shared_hbm)
    lse = None
    if training:
        lse = nl.ndarray(
            (bs, nl.tile_size.pmax, num_grps),
            dtype=lse_dtype,
            buffer=nl.shared_hbm,
        )

    softmax_scale = softmax_scale or (1.0 / float(d**0.5))
    # When using causal mask with CP mode, attention_cte requires scale=1.0.
    # The caller must pre-scale Q by softmax_scale before invoking this kernel.
    attn_scale = 1.0 if use_causal_mask else softmax_scale

    """
    LNC info — needed for _normalize_and_write_output to write only this
    shard's batches to shared output, and for _tiled_reduce_attention.
    attention_cte handles LNC sharding internally, so under LNC2 it shards
    on batch: NC0 processes batches [0, bs//2), NC1 processes [bs//2, bs).
    Outputs are in shared_hbm.
    """
    program_ndims = nl.program_ndim()
    shard_id = nl.program_id(0) if program_ndims == 1 else 0
    num_shards = nl.num_programs(0) if program_ndims == 1 else 1
    num_bs_per_shard = bs // num_shards
    bs_offset = shard_id * num_bs_per_shard
    has_remainder = (bs % num_shards) != 0

    """
    Collective permute send/recv buffers for K/V in shared_hbm.
    K buffer layout matches input k layout (depends on tp_k).
    V buffer: (bs, seqlen, d) — V in non-transposed layout
    """
    if tp_k:
        # K in non-transposed layout: (bs, seqlen, d)
        send_k_buf = nl.ndarray(
            (bs, seqlen, d),
            dtype=k.dtype,
            buffer=nl.shared_hbm,
            name="send_k_buf",
        )
        recv_k_buf = nl.ndarray(
            (bs, seqlen, d),
            dtype=k.dtype,
            buffer=nl.shared_hbm,
            name="recv_k_buf",
        )
    else:
        # K in transposed layout: (bs, d, seqlen)
        send_k_buf = nl.ndarray(
            (bs, d, seqlen),
            dtype=k.dtype,
            buffer=nl.shared_hbm,
            name="send_k_buf",
        )
        recv_k_buf = nl.ndarray(
            (bs, d, seqlen),
            dtype=k.dtype,
            buffer=nl.shared_hbm,
            name="recv_k_buf",
        )
    send_v_buf = nl.ndarray(
        (bs, seqlen, d),
        dtype=v.dtype,
        buffer=nl.shared_hbm,
        name="send_v_buf",
    )
    recv_v_buf = nl.ndarray(
        (bs, seqlen, d),
        dtype=v.dtype,
        buffer=nl.shared_hbm,
        name="recv_v_buf",
    )

    # Create ReplicaGroup once and reuse across all ring steps.
    replica_group = ReplicaGroup(replica_groups)

    # For causal masking, rank IDs are needed to compute cp_offset per ring step.
    # cp_offset_hbm is a (1,1) shared HBM tensor that gets updated each ring step.
    cp_offset_hbm = None
    if use_causal_mask:
        iota_nw_sb = nl.ndarray((1, num_workers), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(iota_nw_sb, [[1, num_workers]], offset=0)
        iota_nw = nl.ndarray((1, num_workers), dtype=nl.float32, buffer=nl.shared_hbm, name="iota_nw")
        nisa.dma_copy(dst=iota_nw, src=iota_nw_sb)

        rank_id = ncc.collective_permute_implicit_current_processing_rank_id(
            iteration_id=0,
            replica_group=replica_group,
        )
        rank_id_sb = _load_rank_sb(iota_nw, rank_id, _Q_GRP_SZ)
        cp_offset_hbm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    allocator = ModularAllocator(initial_address=0)

    # Ring step 0: Local K/V (Q and K from same rank => cp_offset=0)
    if use_causal_mask:
        _cp_zero_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(_cp_zero_sb, 0.0)
        nisa.dma_copy(dst=cp_offset_hbm, src=_cp_zero_sb)

    """
    attention_cte handles LNC sharding and batch iteration internally.
    With skip_output_normalization=True, returns unnormalized output and raw S.
    Outputs are in shared_hbm. For LNC2 odd bs, the remainder batch is
    seqlen-sharded across NCs — core barrier needed before reduction.
    """
    o_prev, neg_max_prev, sum_prev = attention_cte(
        q,
        k,
        v,
        scale=attn_scale,
        causal_mask=use_causal_mask,
        tp_q=tp_q,
        tp_k=tp_k,
        tp_out=False,
        cache_softmax=True,
        cp_offset=cp_offset_hbm if use_causal_mask else None,
        global_cp_deg=num_workers if use_causal_mask else None,
        cp_striped_input=striped_input,
        skip_output_normalization=True,
        bound_min=bound_min,
        bound_max=bound_max,
    )

    # Initialize send KV buffers with local K/V.
    # Each NC copies only its assigned batches to parallelize DMA work under LNC2.
    _init_kv_send_buffers(send_k_buf, send_v_buf, k, v, bs, num_bs_per_shard, bs_offset, has_remainder, shard_id)

    """
    Ring steps 1..num_workers-1

    Ping-pong buffer design: instead of copying recv→send after each
    step, we swap which buffer is "current" (to send from) and which
    is "next" (to receive into). This eliminates per-step K/V DMA copy.

    CC pipeline: to hide collective latency, we pipeline the collective
    permute one step ahead: launch the collective for step N+1 in step N.
    The first collective is hoisted before the loop (overlaps with
    step 0's attention_cte). The last step skips the collective launch.
    """
    cur_k, nxt_k = send_k_buf, recv_k_buf
    cur_v, nxt_v = send_v_buf, recv_v_buf

    # Hoist first collective permute before the loop.
    if num_workers > 1:
        # Core barrier needed because NCs wrote the send buffers via _init_kv_send_buffers.
        if num_shards > 1:
            nisa.core_barrier(cur_k, (0, 1))
            nisa.core_barrier(cur_v, (0, 1))
        ncc.collective_permute_implicit(
            srcs_by_channel=[[cur_k]],
            dsts_by_channel=[[nxt_k]],
            replica_group=replica_group,
        )
        ncc.collective_permute_implicit(
            srcs_by_channel=[[cur_v]],
            dsts_by_channel=[[nxt_v]],
            replica_group=replica_group,
        )

    for ring_step in nl.sequential_range(1, num_workers):
        # For causal masking, compute cp_offset for this ring step.
        if use_causal_mask:
            recv_rank = ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=ring_step,
                replica_group=replica_group,
            )
            recv_rank_sb = _load_rank_sb(iota_nw, recv_rank, _Q_GRP_SZ)
            _compute_cp_offset(rank_id_sb, recv_rank_sb, cp_offset_hbm, seqlen, striped_input)

        # Run attention_cte with K/V in nxt buffers (collective already landed).
        o_curr, neg_max_curr, sum_curr = attention_cte(
            q,
            nxt_k,
            nxt_v,
            scale=attn_scale,
            causal_mask=use_causal_mask,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=False,
            cache_softmax=True,
            cp_offset=cp_offset_hbm if use_causal_mask else None,
            global_cp_deg=num_workers if use_causal_mask else None,
            cp_striped_input=striped_input,
            skip_output_normalization=True,
            bound_min=bound_min,
            bound_max=bound_max,
        )

        """
        Swap buffer roles BEFORE launching the next collective.
        attention_cte is done reading nxt; after swap, cur points to the
        just-received data (for sending) and nxt points to the free buffer
        (for receiving). The collective reads cur and writes nxt — no
        conflict with attention_cte which already finished reading.
        """
        cur_k, nxt_k = nxt_k, cur_k
        cur_v, nxt_v = nxt_v, cur_v

        # Launch next collective permute (skip on last step — no next step).
        if ring_step < num_workers - 1:
            ncc.collective_permute_implicit(
                srcs_by_channel=[[cur_k]],
                dsts_by_channel=[[nxt_k]],
                replica_group=replica_group,
            )
            ncc.collective_permute_implicit(
                srcs_by_channel=[[cur_v]],
                dsts_by_channel=[[nxt_v]],
                replica_group=replica_group,
            )

        # Tiled reduction: combine current step with accumulated results.
        _tiled_reduce_attention(
            o_prev,
            neg_max_prev,
            sum_prev,
            o_curr,
            neg_max_curr,
            sum_curr,
            bs,
            d,
            num_grps,
            seqlen,
            num_bs_per_shard,
            bs_offset,
            has_remainder,
            shard_id,
            allocator,
        )

    # Normalize output and compute LSE
    _normalize_and_write_output(
        o_prev,
        neg_max_prev,
        sum_prev,
        o,
        lse,
        bs,
        d,
        num_grps,
        seqlen,
        num_bs_per_shard,
        bs_offset,
        has_remainder,
        shard_id,
        training,
        lse_dtype,
        allocator,
    )

    o = o.reshape((b, q_h, seqlen, d))
    if training:
        lse = lse.reshape((b, q_h, nl.tile_size.pmax, num_grps))
        return o, lse
    return o
