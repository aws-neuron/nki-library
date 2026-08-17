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
Utility for combining two attention outputs using online softmax rescaling.

This module uses the unnormalized accumulation approach: outputs and sums
are kept unnormalized throughout the reduction chain, and the caller
normalizes once at the very end by dividing output by sum. This minimizes
ISA instructions per reduction.

The main entry point is ``reduce_attention_stats``, which computes the
combined softmax statistics (neg_max, sum) and correction factors for all
Q groups at once on (128, num_grps) tensors. The caller then applies the
per-group correction factors to the output tiles in its own loop.
"""

import nki.isa as nisa
import nki.language as nl

from .modular_allocator import ModularAllocator

# Maximum number of physical SBUF tiles to allocate for group-tiled buffers.
# Caps num_free_tiles to limit SBUF usage when num_grps is large.
_MAX_FREE_TILES = 8


def reduce_attention_stats(
    neg_max_running,
    exp_sum_running,
    neg_max_curr,
    exp_sum_curr,
    allocator: ModularAllocator,
):
    """
    Reduce/combine softmax statistics for all Q groups at once using online
    softmax rescaling, and return the per-group correction factors.

    Operates on (128, num_grps) SBUF tensors — all groups are processed in a
    single set of ISA instructions instead of looping per group. The caller
    is responsible for applying the returned correction factors to the
    per-group output tiles (which must be tiled through HBM due to SBUF size).

    Background:
        When attention is computed in chunks, each chunk computes unnormalized results:
            output_chunk = exp(scores - M_chunk) @ V_chunk   (unnormalized)
            S_chunk = sum(exp(scores - M_chunk))             (softmax denominator)
            M_chunk = max(scores)                            (row-wise max)

        We store neg_max = -M (negated so global max = min of neg_maxes) and
        sum = S (raw denominator, not reciprocal). The caller normalizes once
        at the end: final_output = output / sum.

    Derivation:
        Let M_p, M_c = positive maxes for prev/curr chunks.
        Let S_p, S_c = local softmax denominators.
        Let M_g = max(M_p, M_c) = global max, i.e. neg_max_new = min(neg_max_p, neg_max_c).

        Each chunk's unnormalized output and sum must be rescaled from its
        local max to the global max:
            corr_running = exp(M_p - M_g)  -- always in (0, 1] since M_g >= M_p, no overflow.
            corr_curr    = exp(M_c - M_g)  -- always in (0, 1] since M_g >= M_c, no overflow.
        Using negated maxes: corr = exp(-neg_max + neg_max_new) = exp(neg_max_new - neg_max).

        The combined unnormalized results are simply:
            exp_sum_new = S_p * corr_running + S_c * corr_curr

        The caller applies the correction factors to the output tiles:
            out_new = out_running * corr_running[:, grp_i] + out_curr * corr_curr[:, grp_i]

    Why unnormalized accumulation is better than normalized (reciprocal) approach:
        - Fewer ISA instructions: no reciprocal, no blending weights,
          no cross-multiplication with other chunk's 1/S
        - Simpler algebra: just "shift to common max, then add"

    Summary of formulas (using negated maxes as stored):
        neg_max_new                       = min(neg_max_running, neg_max_curr)
        flash_attn_correction_factor_running = exp(neg_max_new - neg_max_running)
        flash_attn_correction_factor_curr    = exp(neg_max_new - neg_max_curr)
        exp_sum_new                       = exp_sum_running * flash_attn_correction_factor_running + exp_sum_curr * flash_attn_correction_factor_curr

    ISA instruction breakdown (all groups at once, independent of num_grps):
        1. tensor_tensor(min)                          -- neg_max_new
        2. tensor_tensor(sub)                          -- neg_max_new - neg_max_running
        3. activation(exp)                             -- flash_attn_correction_factor_running
        4. tensor_tensor(sub)                          -- neg_max_new - neg_max_curr
        5. activation(exp)                             -- flash_attn_correction_factor_curr
        6. tensor_tensor(mul)                          -- exp_sum_running * flash_attn_correction_factor_running
        7. tensor_tensor(mul)                          -- exp_sum_curr * flash_attn_correction_factor_curr (tmp)
        8. tensor_tensor(add)                          -- exp_sum_new = (6) + (7)

    Args:
        neg_max_running (nl.ndarray): [128, num_grps], Running negative max (-M_p) in SBUF.
        exp_sum_running (nl.ndarray): [128, num_grps], Running softmax denominator (S_p) in SBUF.
        neg_max_curr (nl.ndarray): [128, num_grps], Current step negative max (-M_c) in SBUF.
        exp_sum_curr (nl.ndarray): [128, num_grps], Current step softmax denominator (S_c) in SBUF.
        allocator (ModularAllocator): ModularAllocator for temporary SBUF buffers.

    Returns:
        tuple of (neg_max_new, exp_sum_new, flash_attn_correction_factor_running, flash_attn_correction_factor_curr):
            neg_max_new (nl.ndarray): [128, num_grps], Combined negative max.
            exp_sum_new (nl.ndarray): [128, num_grps], Combined softmax denominator.
            flash_attn_correction_factor_running (nl.ndarray): [128, num_grps], Correction factor for running output tiles.
            flash_attn_correction_factor_curr (nl.ndarray): [128, num_grps], Correction factor for current output tiles.
        All SBUF tensors, unnormalized.
        Caller must apply correction factors to output tiles per group:
            out_new = out_running * flash_attn_correction_factor_running[:, grp_i] + out_curr * flash_attn_correction_factor_curr[:, grp_i]
        And normalize at the end: final_output = out_new * (1 / exp_sum_new).
    """
    S = neg_max_running.shape[0]  # 128
    G = neg_max_running.shape[1]  # num_grps

    # Allocate output buffers
    neg_max_new = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    exp_sum_new = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)

    # Allocate correction factor buffers
    flash_attn_correction_factor_running = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    flash_attn_correction_factor_curr = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)

    # 1. Global max (negated): neg_max_new = min(neg_max_running, neg_max_curr)
    nisa.tensor_tensor(dst=neg_max_new, data1=neg_max_running, data2=neg_max_curr, op=nl.minimum)

    # 2. flash_attn_correction_factor_running = exp(neg_max_new - neg_max_running)
    #    nisa.activation bias must be Nx1, so we compute the difference first
    #    then exponentiate separately.
    nisa.tensor_tensor(
        dst=flash_attn_correction_factor_running, data1=neg_max_new, data2=neg_max_running, op=nl.subtract
    )
    nisa.activation(dst=flash_attn_correction_factor_running, op=nl.exp, data=flash_attn_correction_factor_running)

    # 3. flash_attn_correction_factor_curr = exp(neg_max_new - neg_max_curr)
    nisa.tensor_tensor(dst=flash_attn_correction_factor_curr, data1=neg_max_new, data2=neg_max_curr, op=nl.subtract)
    nisa.activation(dst=flash_attn_correction_factor_curr, op=nl.exp, data=flash_attn_correction_factor_curr)

    # 4. exp_sum_new = exp_sum_running * flash_attn_correction_factor_running + exp_sum_curr * flash_attn_correction_factor_curr
    #    scalar_tensor_tensor requires operand0 to be Nx1, so we use
    #    tensor_tensor(mul) into a temp, then tensor_tensor(add).
    nisa.tensor_tensor(
        dst=exp_sum_new, data1=exp_sum_running, data2=flash_attn_correction_factor_running, op=nl.multiply
    )
    tmp = allocator.alloc_sbuf_tensor(shape=(S, G), dtype=nl.float32)
    nisa.tensor_tensor(dst=tmp, data1=exp_sum_curr, data2=flash_attn_correction_factor_curr, op=nl.multiply)
    nisa.tensor_tensor(dst=exp_sum_new, data1=exp_sum_new, data2=tmp, op=nl.add)

    return neg_max_new, exp_sum_new, flash_attn_correction_factor_running, flash_attn_correction_factor_curr


def reduce_one_batch(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    batch_idx,
    grp_start,
    grp_end,
    d,
    num_grps,
    sb_p,
    softmax_pat,
    o_pat,
    neg_max_prev_sb,
    sum_prev_sb,
    neg_max_curr_sb,
    sum_curr_sb,
    o_prev_sb,
    o_curr_sb,
    o_new,
    batch_loop_addr,
    allocator,
):
    """Reduce softmax stats and apply correction for one batch over [grp_start, grp_end).

    Loads softmax statistics from HBM, computes correction factors via
    reduce_attention_stats, applies per-group corrections to output tiles,
    and writes back the updated stats and output.

    Args:
        o_prev_hbm (nl.ndarray): [bs, seqlen, d], Accumulated unnormalized output in HBM.
        neg_max_prev_hbm (nl.ndarray): [bs, 128, num_grps], Negated row max for accumulated output.
        sum_prev_hbm (nl.ndarray): [bs, 128, num_grps], Raw softmax denominator S for accumulated output.
        o_curr_hbm (nl.ndarray): [bs, seqlen, d], Current step unnormalized output in HBM.
        neg_max_curr_hbm (nl.ndarray): [bs, 128, num_grps], Negated row max for current step.
        sum_curr_hbm (nl.ndarray): [bs, 128, num_grps], Raw softmax denominator S for current step.
        batch_idx (int): Batch index to process.
        grp_start (int): First Q group to reduce (inclusive).
        grp_end (int): Last Q group to reduce (exclusive).
        d (int): Head dimension size.
        num_grps (int): Total number of Q groups (seqlen // 128).
        sb_p (int): Partition tile size (128).
        softmax_pat: Array pattern for softmax stat tensors.
        o_pat: Array pattern for output tiles.
        neg_max_prev_sb (nl.ndarray): Pre-allocated SBUF for prev neg_max.
        sum_prev_sb (nl.ndarray): Pre-allocated SBUF for prev sum.
        neg_max_curr_sb (nl.ndarray): Pre-allocated SBUF for curr neg_max.
        sum_curr_sb (nl.ndarray): Pre-allocated SBUF for curr sum.
        o_prev_sb (nl.ndarray): Pre-allocated SBUF for prev output tiles.
        o_curr_sb (nl.ndarray): Pre-allocated SBUF for curr output tiles.
        o_new (nl.ndarray): Pre-allocated SBUF for corrected output tiles.
        batch_loop_addr (int): Allocator checkpoint to reset temporaries.
        allocator (ModularAllocator): SBUF allocator for temporaries.
    """
    allocator.set_current_address(batch_loop_addr)

    batch_softmax_offset = batch_idx * sb_p * num_grps
    batch_o_offset = batch_idx * num_grps * sb_p * d

    nisa.dma_copy(dst=neg_max_prev_sb, src=neg_max_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
    nisa.dma_copy(dst=sum_prev_sb, src=sum_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
    nisa.dma_copy(dst=neg_max_curr_sb, src=neg_max_curr_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
    nisa.dma_copy(dst=sum_curr_sb, src=sum_curr_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))

    neg_max_new, exp_sum_new, flash_attn_correction_factor_running, flash_attn_correction_factor_curr = (
        reduce_attention_stats(
            neg_max_prev_sb,
            sum_prev_sb,
            neg_max_curr_sb,
            sum_curr_sb,
            allocator,
        )
    )

    for grp_i in range(grp_start, grp_end):
        grp_o_offset = batch_o_offset + grp_i * sb_p * d
        nisa.dma_copy(dst=o_prev_sb[grp_i], src=o_prev_hbm.ap(pattern=o_pat, offset=grp_o_offset))
        nisa.dma_copy(dst=o_curr_sb[grp_i], src=o_curr_hbm.ap(pattern=o_pat, offset=grp_o_offset))
        nisa.tensor_scalar(
            dst=o_new[grp_i],
            data=o_prev_sb[grp_i],
            op0=nl.multiply,
            operand0=flash_attn_correction_factor_running[:, grp_i],
        )
        nisa.scalar_tensor_tensor(
            dst=o_new[grp_i],
            data=o_curr_sb[grp_i],
            op0=nl.multiply,
            operand0=flash_attn_correction_factor_curr[:, grp_i],
            op1=nl.add,
            operand1=o_new[grp_i],
        )
        nisa.dma_copy(dst=o_prev_hbm.ap(pattern=o_pat, offset=grp_o_offset), src=o_new[grp_i])

    # Write back stats — only the groups we processed
    grp_count = grp_end - grp_start
    wb_softmax_pat = [[num_grps, sb_p], [1, grp_count]]
    nisa.dma_copy(
        dst=neg_max_prev_hbm.ap(pattern=wb_softmax_pat, offset=batch_softmax_offset + grp_start),
        src=neg_max_new.ap(pattern=wb_softmax_pat, offset=grp_start),
    )
    nisa.dma_copy(
        dst=sum_prev_hbm.ap(pattern=wb_softmax_pat, offset=batch_softmax_offset + grp_start),
        src=exp_sum_new.ap(pattern=wb_softmax_pat, offset=grp_start),
    )


def normalize_one_batch(
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_out,
    lse_out,
    batch_idx,
    d,
    num_grps,
    training,
    lse_dtype,
    allocator,
    grp_start=0,
    grp_end=-1,
):
    """Normalize and write output for a single batch (or a subset of its groups).

    Computes final_output = o_unnorm / S and writes to o_out.
    Optionally computes LSE = -neg_max + log(S) for training.

    Args:
        o_prev_hbm (nl.ndarray): [bs, seqlen, d], Accumulated unnormalized output in HBM.
        neg_max_prev_hbm (nl.ndarray): [bs, 128, num_grps], Negated row max.
        sum_prev_hbm (nl.ndarray): [bs, 128, num_grps], Raw softmax denominator S.
        o_out (nl.ndarray): [bs, seqlen, d], Final output tensor in HBM.
        lse_out (nl.ndarray or None): [bs, 128, num_grps], LSE output (if training).
        batch_idx (int): Batch index to normalize.
        d (int): Head dimension size.
        num_grps (int): Number of Q groups (seqlen // 128).
        training (bool): Whether to compute LSE.
        lse_dtype: Data type for LSE output.
        allocator (ModularAllocator): SBUF allocator for temporaries.
        grp_start (int): First group to normalize (default 0).
        grp_end (int): One past last group to normalize (default -1 = num_grps).
    """
    if grp_end < 0:
        grp_end = num_grps
    sb_p = nl.tile_size.pmax
    softmax_pat = [[num_grps, sb_p], [1, num_grps]]
    o_tile_pat = [[d, sb_p], [1, d]]

    batch_softmax_offset = batch_idx * sb_p * num_grps
    batch_o_offset = batch_idx * num_grps * sb_p * d

    init_addr = allocator.get_current_address()

    # Load sum (raw S) from HBM
    sum_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    nisa.dma_copy(dst=sum_sb, src=sum_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))

    # Compute 1/S for normalization
    sum_recip_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
    nisa.reciprocal(sum_recip_sb, sum_sb)

    o_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, d),
        dtype=nl.float32,
        block_dim=[num_grps],
        num_free_tiles=[min(num_grps, _MAX_FREE_TILES)],
    )

    o_batch_view = o_out.select(dim=0, index=batch_idx)

    for grp_i in range(grp_start, grp_end):
        grp_o_offset = batch_o_offset + grp_i * sb_p * d

        nisa.dma_copy(
            dst=o_sb[grp_i],
            src=o_prev_hbm.ap(pattern=o_tile_pat, offset=grp_o_offset),
        )

        sum_recip_grp = sum_recip_sb.slice(dim=1, start=grp_i, end=grp_i + 1)
        # Normalize: o_final = o_unnorm * (1/S)
        nisa.tensor_scalar(dst=o_sb[grp_i], data=o_sb[grp_i], op0=nl.multiply, operand0=sum_recip_grp)

        # Write to final output
        o_grp_view = o_batch_view.reshape_dim(dim=0, shape=(num_grps, sb_p)).select(dim=0, index=grp_i)
        nisa.dma_copy(
            dst=o_grp_view,
            src=o_sb[grp_i],
        )

    # Write LSE if training
    # lse = -neg_max + log(S)
    if training and lse_out is not None:
        neg_max_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
        nisa.dma_copy(dst=neg_max_sb, src=neg_max_prev_hbm.ap(pattern=softmax_pat, offset=batch_softmax_offset))
        # sum_sb already loaded above

        lse_tile = nl.ndarray((sb_p, num_grps), dtype=lse_dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(lse_tile, neg_max_sb, nl.multiply, -1.0)
        log_s = nl.ndarray((sb_p, num_grps), dtype=lse_dtype, buffer=nl.sbuf)
        nisa.activation(log_s, nl.log, sum_sb)
        nisa.tensor_tensor(lse_tile, lse_tile, log_s, op=nl.add)
        # Write only our groups' LSE using ap() on the raw tensor
        grp_count = grp_end - grp_start
        lse_ap_pat = [[num_grps, sb_p], [1, grp_count]]
        lse_batch_offset = batch_idx * sb_p * num_grps
        nisa.dma_copy(
            dst=lse_out.ap(pattern=lse_ap_pat, offset=lse_batch_offset + grp_start),
            src=lse_tile.ap(pattern=lse_ap_pat, offset=grp_start),
        )

    allocator.set_current_address(init_addr)
