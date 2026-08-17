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
Fused segmented attention: replaces per-segment _attention_cte + external reduction
with a single flash-attention loop over all segments.

Uses kv_section_idx=0 so K/V indexing always starts at tile 0 (each segment has
its own K/V in SBUF), while section_idx > 0 triggers the flash attention
accumulation path in _write_back_impl and _update_max_impl. This keeps the PV
accumulation in float32 SBUF across segments, matching _attention_cte's internal
precision.
"""

import math

import nki.isa as nisa
import nki.language as nl

from ..utils.attention_reduce import _MAX_FREE_TILES, reduce_one_batch
from ..utils.kernel_assert import kernel_assert
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .attention_cte import (
    _D_TILE_SZ,
    _K_TILE_SZ,
    _V_D_TILE_SZ,
    _V_TILE_SZ,
    AttnConfig,
    AttnInternalBuffers,
    SectionParams,
    _allocate_attention_buffers,
    _compute_tile_parameters,
    _exp_impl,
    _fused_qkmax_and_pv_impl,
    _load_q_impl,
    _pv_impl,
    _qk_and_max_impl,
    _setup_range_select_bounds,
    _update_max_impl,
    _write_back_impl,
)


def _run_groups(grp_start, grp_end, ac, atp, sp, bufs, q, batch_id, o, sbuf_addr, sink=None):
    """Run Q-group loop with software pipelining."""
    n = grp_end - grp_start
    if n <= 1:
        _load_q_impl(grp_start, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_start, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_start, ac, atp, sp, bufs, sink)
        _exp_impl(grp_start, ac, atp, sp, bufs, sink)
        _pv_impl(grp_start, ac, atp, sp, bufs)
        _write_back_impl(grp_start, ac, atp, sp, bufs, o, batch_id)
    else:
        _load_q_impl(grp_start, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_start, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_start, ac, atp, sp, bufs, sink)
        _exp_impl(grp_start, ac, atp, sp, bufs, sink)

        _load_q_impl(grp_start + 1, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
        _qk_and_max_impl(grp_start + 1, ac, atp, sp, bufs, batch_id)
        _update_max_impl(grp_start + 1, ac, atp, sp, bufs, sink)

        for grp_i in range(grp_start, grp_end - 2):
            _load_q_impl(grp_i + 2, ac, atp, sp, bufs, q, batch_id, sbuf_addr)
            _exp_impl(grp_i + 1, ac, atp, sp, bufs, sink)
            _fused_qkmax_and_pv_impl(grp_i, ac, atp, sp, bufs, batch_id)
            _write_back_impl(grp_i, ac, atp, sp, bufs, o, batch_id)
            _update_max_impl(grp_i + 2, ac, atp, sp, bufs, sink)

        _pv_impl(grp_end - 2, ac, atp, sp, bufs)
        _write_back_impl(grp_end - 2, ac, atp, sp, bufs, o, batch_id)
        _exp_impl(grp_end - 1, ac, atp, sp, bufs, sink)
        _pv_impl(grp_end - 1, ac, atp, sp, bufs)
        _write_back_impl(grp_end - 1, ac, atp, sp, bufs, o, batch_id)


def _make_ac_atp(
    seqlen_q, seqlen_k, head_dim, dtype, causal, scale, tp_q, tp_out, num_sections, use_cp=False, global_cp_deg=None
):
    """Create AttnConfig + AttnTileParams."""
    ac = AttnConfig(
        seqlen_q=seqlen_q,
        seqlen_k_active=seqlen_k,
        seqlen_k_prior=None,
        d=head_dim,
        tp_q=tp_q,
        tp_k=False,
        tp_out=tp_out,
        is_prefix_caching=False,
        causal_mask=causal,
        use_swa=False,
        sliding_window=0,
        use_cp=use_cp,
        global_cp_deg=global_cp_deg,
        cp_strided_q_slicing=False,
        cp_striped_input=False,
        scale=scale,
        cache_softmax=True,
        skip_output_normalization=True,
        dtype=dtype,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        is_sequence_packed=False,
    )
    atp = _compute_tile_parameters(ac, is_seqlen_sharded=False)
    atp.num_sections = num_sections
    return ac, atp


def _kvp_partial_prior_attention(
    q_hbm,
    k_cache_sbuf,
    v_cache_sbuf,
    k_prior_sbuf,
    v_prior_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    kvp_offset_active_hbm,
    kvp_q_offset,
    prior_block_offset,
    partial_prior_tokens,
    num_k_tiles_active,
    num_v_tiles_active,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    n_grps,
    head_dim,
    block_size,
    sb_p,
    scale,
    tp_q,
    allocator,
    attention_cte_fn,
    sink=None,
    k_scale_sb=None,
    active_block_offset=None,
    kvp_rank_id=None,
    kvp_group_size=0,
    sliding_window=0,
    kvp_prior_load_blocks=0,
    kvp_prior_fully_visible=False,
):
    """KVP partial prior: handles the boundary segment that straddles active and prior KV.

    Splits into two calls to avoid the unsupported use_cp + is_prefix_caching combination:
      1. Active-only: causal_mask=True with cp_offset=kvp_offset_active
      2. Prior-only: causal_mask=False with effective_prior_used_len
    Results are reduced via online softmax into o_prev_hbm in-place.
    """
    init_sbuf_addr = allocator.get_current_address()

    # Create kvp_seg_block_offset HBM tensor for active segment
    seg_offset_hbm = None
    if kvp_group_size > 0:
        seg_offset_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        seg_offset_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)
        nisa.tensor_copy(dst=seg_offset_sbuf, src=active_block_offset)
        nisa.dma_copy(dst=seg_offset_hbm, src=seg_offset_sbuf)

    # Call 1: active-only with causal mask + cp_offset.
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=True,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_cache_sbuf[:num_k_tiles_active],
        v_cache_sbuf=v_cache_sbuf[:num_v_tiles_active],
        out_o_hbm=o_prev_hbm,
        out_neg_max_hbm=neg_max_prev_hbm,
        out_sum_hbm=sum_prev_hbm,
        init_sbuf_addr=init_sbuf_addr,
        sink=sink,
        cp_offset=kvp_offset_active_hbm,
        global_cp_deg=1,
        k_scale_sb=k_scale_sb,
        kvp_rank_id=kvp_rank_id,
        kvp_group_size=kvp_group_size,
        block_size=block_size,
        kvp_seg_block_offset=seg_offset_hbm if kvp_group_size > 0 else None,
        sliding_window=sliding_window,
    )
    allocator.set_current_address(init_sbuf_addr)

    # Compute effective_prior_used_len
    effective_prior_used_len = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    prior_seg_start_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    nisa.tensor_scalar(dst=prior_seg_start_sbuf, data=prior_block_offset, op0=nl.multiply, operand0=block_size)
    nisa.tensor_tensor(dst=effective_prior_used_len, data1=kvp_q_offset, data2=prior_seg_start_sbuf, op=nl.subtract)
    nisa.tensor_tensor(
        dst=effective_prior_used_len, data1=effective_prior_used_len, data2=partial_prior_tokens, op=nl.minimum
    )
    nisa.tensor_scalar(dst=effective_prior_used_len, data=effective_prior_used_len, op0=nl.maximum, operand0=0)

    # Re-zero active KV sbuf before Call 2 (prior-only, causal_mask=False).
    num_d_tiles_k = len(k_cache_sbuf[0])
    num_d_tiles_v = len(v_cache_sbuf[0])
    for k_idx in range(num_k_tiles_active):
        for d_idx in range(num_d_tiles_k):
            nisa.memset(k_cache_sbuf[k_idx][d_idx][...], value=0.0)
    for v_idx in range(num_v_tiles_active):
        for d_idx in range(num_d_tiles_v):
            nisa.memset(v_cache_sbuf[v_idx][d_idx][...], value=0.0)

    call2_sbuf_addr = allocator.get_current_address()
    if kvp_group_size > 0 and kvp_prior_fully_visible:
        # Interleaved with fully-visible prior: copy prior into k_cache_sbuf (already zeroed)
        # and pass only the loaded tiles with causal_mask=False.
        num_partial_k_tiles = (
            math.ceil(kvp_prior_load_blocks * block_size / _K_TILE_SZ)
            if kvp_prior_load_blocks > 0
            else num_k_tiles_per_seg
        )
        num_partial_v_tiles = num_partial_k_tiles * (_K_TILE_SZ // _V_TILE_SZ)
        for k_idx in range(num_partial_k_tiles):
            for d_idx in range(num_d_tiles_k):
                nisa.tensor_copy(dst=k_cache_sbuf[k_idx][d_idx][...], src=k_prior_sbuf[k_idx][d_idx][...])
        for v_idx in range(num_partial_v_tiles):
            for d_idx in range(num_d_tiles_v):
                nisa.tensor_copy(dst=v_cache_sbuf[v_idx][d_idx][...], src=v_prior_sbuf[v_idx][d_idx][...])

        attention_cte_fn(
            q_hbm,
            None,
            None,
            scale=scale,
            causal_mask=False,
            tp_q=tp_q,
            tp_k=False,
            tp_out=False,
            cache_softmax=True,
            skip_output_normalization=True,
            k_cache_sbuf=k_cache_sbuf[:num_partial_k_tiles],
            v_cache_sbuf=v_cache_sbuf[:num_partial_v_tiles],
            kv_used_len=partial_prior_tokens,
            out_o_hbm=o_curr_hbm,
            out_neg_max_hbm=neg_max_curr_hbm,
            out_sum_hbm=sum_curr_hbm,
            init_sbuf_addr=call2_sbuf_addr,
            block_size=block_size,
        )
    elif kvp_group_size > 0:
        # Interleaved without fully-visible prior: use causal masking (original path)
        for k_idx in range(num_k_tiles_per_seg):
            for d_idx in range(num_d_tiles_k):
                nisa.tensor_copy(dst=k_cache_sbuf[k_idx][d_idx][...], src=k_prior_sbuf[k_idx][d_idx][...])
        num_v_tiles_per_k_tile = _K_TILE_SZ // _V_TILE_SZ
        num_v_prior = num_k_tiles_per_seg * num_v_tiles_per_k_tile
        for v_idx in range(num_v_prior):
            for d_idx in range(num_d_tiles_v):
                nisa.tensor_copy(dst=v_cache_sbuf[v_idx][d_idx][...], src=v_prior_sbuf[v_idx][d_idx][...])

        seg_offset_prior_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        seg_offset_prior_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)
        nisa.tensor_copy(dst=seg_offset_prior_sbuf, src=prior_block_offset)
        nisa.dma_copy(dst=seg_offset_prior_hbm, src=seg_offset_prior_sbuf)

        attention_cte_fn(
            q_hbm,
            None,
            None,
            scale=scale,
            causal_mask=True,
            tp_q=tp_q,
            tp_k=False,
            tp_out=False,
            cache_softmax=True,
            skip_output_normalization=True,
            k_cache_sbuf=k_cache_sbuf[:num_k_tiles_per_seg],
            v_cache_sbuf=v_cache_sbuf[:num_v_tiles_per_seg],
            out_o_hbm=o_curr_hbm,
            out_neg_max_hbm=neg_max_curr_hbm,
            out_sum_hbm=sum_curr_hbm,
            init_sbuf_addr=call2_sbuf_addr,
            cp_offset=kvp_offset_active_hbm,
            global_cp_deg=1,
            kvp_rank_id=kvp_rank_id,
            kvp_group_size=kvp_group_size,
            block_size=block_size,
            kvp_seg_block_offset=seg_offset_prior_hbm,
            sliding_window=sliding_window,
        )
    else:
        # Contiguous: prior-only with causal_mask=False and effective_prior_used_len.
        attention_cte_fn(
            q_hbm,
            None,
            None,
            scale=scale,
            causal_mask=False,
            tp_q=tp_q,
            tp_k=False,
            tp_out=False,
            cache_softmax=True,
            skip_output_normalization=True,
            k_cache_sbuf=k_cache_sbuf[:num_k_tiles_per_seg],
            v_cache_sbuf=v_cache_sbuf[:num_v_tiles_per_seg],
            k_prior_sbuf=k_prior_sbuf,
            v_prior_sbuf=v_prior_sbuf,
            prior_used_len=effective_prior_used_len,
            out_o_hbm=o_curr_hbm,
            out_neg_max_hbm=neg_max_curr_hbm,
            out_sum_hbm=sum_curr_hbm,
            init_sbuf_addr=call2_sbuf_addr,
        )
    allocator.set_current_address(call2_sbuf_addr)

    # Reduce active + prior into o_prev_hbm.
    softmax_pat = [[n_grps, sb_p], [1, n_grps]]
    o_pat = [[head_dim, sb_p], [1, head_dim]]
    num_free = min(n_grps, _MAX_FREE_TILES)
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    o_prev_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_curr_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_new_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    reduce_batch_addr = allocator.get_current_address()
    reduce_one_batch(
        o_prev_hbm,
        neg_max_prev_hbm,
        sum_prev_hbm,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        0,
        0,
        n_grps,
        head_dim,
        n_grps,
        sb_p,
        softmax_pat,
        o_pat,
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb_buf,
        o_prev_sb,
        o_curr_sb,
        o_new_sb,
        reduce_batch_addr,
        allocator,
    )
    allocator.set_current_address(init_sbuf_addr)


def _nonkvp_partial_prior_attention(
    q_hbm,
    k_cache,
    v_cache,
    block_tables,
    k_cache_sbuf,
    v_cache_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    prior_block_offset,
    partial_prior_tokens,
    num_k_tiles_active,
    num_v_tiles_active,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    num_blocks_per_seg,
    num_v_tiles_for_prior,
    b_i,
    h_i,
    n_grps,
    head_dim,
    sb_p,
    scale,
    tp_q,
    allocator,
    attention_cte_fn,
    load_kv_cache_fn,
    sink=None,
    k_pre_transposed=False,
    fp8_packed=False,
):
    """Non-KVP partial prior: two sequential attention_cte calls then reduce.

    Mirrors _kvp_partial_prior_attention's 2-pass shape but without cp_offset /
    global_cp_deg, and uses the static partial_prior_tokens directly as
    prior_used_len (no dynamic effective_prior_used_len math needed).

    Pass 1: active-only, causal_mask=True, sink applied.
    ---- Allocate k_prior_sbuf/v_prior_sbuf ALIASED onto k_cache_sbuf /
         v_cache_sbuf (same physical SBUF region). Pass 1 has already reduced
         its active-K result into o_prev_hbm, so the active K data in
         k_cache_sbuf is no longer needed. Load prior K/V into the aliased
         region.
    Pass 2: prior-only, causal_mask=False, prior_used_len=partial_prior_tokens.
    Reduce via online softmax into o_prev_hbm in-place.

    Saves ~48 KB/partition of peak SBUF at head_dim=128 prior_seg_size=8192
    compared to the previous single-fused-call design (which held both
    k_cache_sbuf and a separate k_prior_sbuf concurrently live through APC).
    """
    init_sbuf_addr = allocator.get_current_address()

    # Pass 1: active-only with causal mask. No k_prior_sbuf reference.
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=True,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_cache_sbuf[:num_k_tiles_active],
        v_cache_sbuf=v_cache_sbuf[:num_v_tiles_active],
        out_o_hbm=o_prev_hbm,
        out_neg_max_hbm=neg_max_prev_hbm,
        out_sum_hbm=sum_prev_hbm,
        init_sbuf_addr=init_sbuf_addr,
        sink=sink,
    )
    allocator.set_current_address(init_sbuf_addr)

    # Alias k_prior_sbuf/v_prior_sbuf onto the first N tiles of
    # k_cache_sbuf/v_cache_sbuf via Python list slicing — same physical SBUF,
    # no new allocation. k_cache_sbuf is sized with
    # max(num_k_tiles_active, num_k_tiles_per_seg) at the caller, so the slice
    # is always in range.
    kernel_assert(
        num_k_tiles_per_seg <= len(k_cache_sbuf),
        "k_cache_sbuf must be sized >= num_k_tiles_per_seg for aliased reuse",
    )
    kernel_assert(
        num_v_tiles_for_prior <= len(v_cache_sbuf),
        "v_cache_sbuf must be sized >= num_v_tiles_for_prior for aliased reuse",
    )
    k_prior_sbuf = k_cache_sbuf[:num_k_tiles_per_seg]
    v_prior_sbuf = v_cache_sbuf[:num_v_tiles_for_prior]

    # Load prior K/V into the aliased region. This overwrites the active K/V
    # from Pass 1, which is safe because Pass 1's results are already in
    # o_prev_hbm.
    load_kv_cache_fn(
        k_cache,
        v_cache,
        block_tables,
        k_prior_sbuf,
        v_prior_sbuf,
        b_i,
        h_i,
        prior_block_offset,
        num_blocks_per_seg,
        allocator,
        k_pre_transposed=k_pre_transposed,
        fp8_packed=fp8_packed,
    )

    call2_sbuf_addr = allocator.get_current_address()
    # Pass 2: non-APC call treating the aliased prior data as the active K/V.
    # `kv_used_len=partial_prior_tokens` dynamically masks K positions beyond
    # the used prior range. Previously this was an APC call (k_prior_sbuf +
    # k_cache_sbuf both pointing to the aliased memory), but that caused the
    # kernel to attend the prior data TWICE — once as "active" (unmasked) and
    # once as "prior" (masked by prior_used_len) — inflating sum_curr_hbm by
    # ~2× and skewing the reduce_one_batch combination. Using kv_used_len in
    # non-APC mode keeps Bucket B's SBUF aliasing AND produces correct output.
    attention_cte_fn(
        q_hbm,
        None,
        None,
        scale=scale,
        causal_mask=False,
        tp_q=tp_q,
        tp_k=False,
        tp_out=False,
        cache_softmax=True,
        skip_output_normalization=True,
        k_cache_sbuf=k_prior_sbuf,
        v_cache_sbuf=v_prior_sbuf,
        kv_used_len=partial_prior_tokens,
        out_o_hbm=o_curr_hbm,
        out_neg_max_hbm=neg_max_curr_hbm,
        out_sum_hbm=sum_curr_hbm,
        init_sbuf_addr=call2_sbuf_addr,
    )
    allocator.set_current_address(call2_sbuf_addr)

    # Reduce active (o_prev_hbm) + prior (o_curr_hbm) into o_prev_hbm.
    softmax_pat = [[n_grps, sb_p], [1, n_grps]]
    o_pat = [[head_dim, sb_p], [1, head_dim]]
    num_free = min(n_grps, _MAX_FREE_TILES)
    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    o_prev_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_curr_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    o_new_sb = allocator.alloc_sbuf_tensor(
        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[n_grps], num_free_tiles=[num_free]
    )
    reduce_batch_addr = allocator.get_current_address()
    reduce_one_batch(
        o_prev_hbm,
        neg_max_prev_hbm,
        sum_prev_hbm,
        o_curr_hbm,
        neg_max_curr_hbm,
        sum_curr_hbm,
        0,
        0,
        n_grps,
        head_dim,
        n_grps,
        sb_p,
        softmax_pat,
        o_pat,
        neg_max_prev_sb,
        sum_prev_sb,
        neg_max_curr_sb,
        sum_curr_sb_buf,
        o_prev_sb,
        o_curr_sb,
        o_new_sb,
        reduce_batch_addr,
        allocator,
    )
    allocator.set_current_address(init_sbuf_addr)


def fused_segmented_attention_impl(
    q_hbm,
    num_batches,
    k_cache,
    v_cache,
    block_tables,
    k_cache_sbuf,
    v_cache_sbuf,
    o_prev_hbm,
    neg_max_prev_hbm,
    sum_prev_hbm,
    o_curr_hbm,
    neg_max_curr_hbm,
    sum_curr_hbm,
    prior_tokens_sbuf,
    num_full_prior_segments_i32,
    partial_prior_tokens,
    is_partial_prior_segment,
    is_not_partial_prior_segment,
    active_block_offset,
    prior_block_offset,
    allocator,
    prior_seg_size,
    block_size,
    scale,
    head_dim,
    num_grps,
    num_active_blocks,
    num_k_tiles_active,
    num_v_tiles_active,
    num_blocks_per_seg,
    num_k_tiles_per_seg,
    num_v_tiles_per_seg,
    b_i=0,
    h_i=0,
    tp_q=True,
    tp_out=False,
    load_kv_cache_fn=None,
    attention_cte_fn=None,
    sink=None,
    k_pre_transposed=False,
    fp8_packed=False,
    k_scale_sb=None,
    kvp_q_offset=None,
    kvp_rank_id=None,
    kvp_group_size=0,
    sliding_window=0,
    kvp_cp_offset_int: int = 0,
    kvp_seg_block_offset_int: int = 0,
    kvp_prior_load_blocks: int = 0,
    kvp_prior_fully_visible: bool = False,
):
    """Fused segmented-attention impl with SBUF aliasing across active/prior passes.

    Uses kv_section_idx=0 so K/V indexing starts at tile 0 for every segment,
    while section_idx controls flash attention accumulation:
      - Active segment: section_idx=0 (init running stats)
      - Prior segments: section_idx=1 (accumulate via _write_back_impl)

    The PV accumulation stays in float32 SBUF across all segments, matching
    _attention_cte's internal flash attention precision.
    """
    orig_addr = allocator.get_current_address()

    # D-tile parameters for 2D K/V SBUF layout
    num_d_tiles = math.ceil(head_dim / _D_TILE_SZ)
    d_tile_size_par_dim = min(head_dim, _D_TILE_SZ)
    num_d_tiles_free_dim = math.ceil(head_dim / (_V_D_TILE_SZ))
    d_tile_size_free_dim = min(head_dim, _V_D_TILE_SZ)

    # D-tile parameters for 2D K/V SBUF layout
    num_d_tiles = math.ceil(head_dim / _D_TILE_SZ)
    d_tile_size_par_dim = min(head_dim, _D_TILE_SZ)
    num_d_tiles_free_dim = math.ceil(head_dim / (_V_D_TILE_SZ))
    d_tile_size_free_dim = min(head_dim, _V_D_TILE_SZ)

    is_kvp = kvp_q_offset != None
    # KVP: compute kvp_offset_active = kvp_q_offset - prior_tokens_sbuf (for active segment cp_offset)
    # and allocate kvp_offset_prior_sbuf/hbm for per-iteration prior segment cp_offset.
    kvp_offset_active_hbm = None
    kvp_offset_prior_sbuf = None
    kvp_offset_prior_hbm = None
    if is_kvp:
        kvp_offset_active_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        kvp_offset_active_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)
        if kvp_group_size > 0:
            # Interleaved: cp_offset = global_q_offset (seg_block_offset handles K-side)
            nisa.tensor_copy(dst=kvp_offset_active_sbuf, src=kvp_q_offset)
        else:
            # Contiguous: kvp_offset_active = kvp_q_offset - prior_tokens
            nisa.tensor_tensor(dst=kvp_offset_active_sbuf, data1=kvp_q_offset, data2=prior_tokens_sbuf, op=nl.subtract)
        nisa.dma_copy(dst=kvp_offset_active_hbm, src=kvp_offset_active_sbuf)
        kvp_offset_prior_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        kvp_offset_prior_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)

    # Create kvp_seg_block_offset HBM buffers for round-robin KV distribution
    kvp_seg_offset_sbuf = None
    kvp_seg_offset_hbm = None
    if kvp_group_size > 0:
        kvp_seg_offset_sbuf = allocator.alloc_sbuf_tensor((1, 1), nl.int32)
        kvp_seg_offset_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)

    seqlen_q = q_hbm.shape[1] if tp_q else q_hbm.shape[2]
    seqlen_k_active = seqlen_q  # actual tokens, not rounded to tile boundary
    seqlen_k_prior = prior_seg_size  # actual tokens, not rounded to tile boundary

    # num_sections must be > 1 to enable flash attention accumulation path
    max_blocks_per_seq = block_tables.shape[1]
    max_prior_segments = math.ceil(max_blocks_per_seq * block_size / prior_seg_size)
    total_sections = max(max_prior_segments + 1, 2)

    # Prior config: KVP uses causal=True + cp to handle shifted causal mask; non-KVP uses causal=False
    ac_p, atp_p = _make_ac_atp(
        seqlen_q,
        seqlen_k_prior,
        head_dim,
        q_hbm.dtype,
        is_kvp,
        scale,
        tp_q,
        False,
        total_sections,
        use_cp=is_kvp,
        global_cp_deg=1 if is_kvp else None,
    )

    sb_p = atp_p.sb_p
    n_grps = atp_p.num_grps

    # Running buffers (persist across all segments in SBUF)
    bufs = AttnInternalBuffers()
    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)
    bufs.k_scale_sb = k_scale_sb
    bufs.mm1_running_max = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    bufs.exp_running_sum = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(shape=(sb_p, n_grps), dtype=nl.float32)

    # Static k_threshold pattern for interleaved KV masking (persists across all segments)
    if kvp_group_size > 0:
        stride = kvp_group_size * block_size
        section_size = max(num_k_tiles_active, num_k_tiles_per_seg) * _K_TILE_SZ
        blocks_per_section = section_size // block_size
        bufs.k_threshold_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, section_size), dtype=nl.float32)
        nisa.iota(
            bufs.k_threshold_sb,
            pattern=[[stride, blocks_per_section], [1, block_size]],
            offset=0,
            channel_multiplier=0,
        )

    if sink is not None:
        bufs.sink_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, 1), dtype=nl.float32)
        nisa.dma_copy(dst=bufs.sink_sb[0, 0], src=sink[0, 0])
        stream_shuffle_broadcast(src=bufs.sink_sb, dst=bufs.sink_sb)

    sbuf_outer = allocator.get_current_address()

    # --- Load active KV ---
    load_kv_cache_fn(
        k_cache,
        v_cache,
        block_tables,
        k_cache_sbuf,
        v_cache_sbuf,
        b_i,
        h_i,
        active_block_offset,
        num_active_blocks,
        allocator,
        k_pre_transposed=k_pre_transposed,
        fp8_packed=fp8_packed,
    )

    # --- ACTIVE SEGMENT (with optional partial prior) ---

    # Partial prior: use _attention_cte with prefix caching (active + partial prior in one call)
    # No partial prior: use fused sub-functions directly
    num_v_tiles_for_prior = num_k_tiles_per_seg * (_K_TILE_SZ // _V_TILE_SZ)

    is_partial_reg = nisa.register_alloc()
    nisa.register_load(dst=is_partial_reg, src=is_partial_prior_segment)

    def _attend_partial_prior(_):
        if is_kvp:
            # KVP path pre-allocates separate k_prior_sbuf / v_prior_sbuf buffers
            # because its helper expects prior K/V already loaded before the call.
            k_prior_sbuf = allocator.alloc_sbuf_tensor(
                shape=(d_tile_size_par_dim, _K_TILE_SZ),
                dtype=k_cache.dtype,
                block_dim=[num_k_tiles_per_seg, num_d_tiles],
                num_free_tiles=[num_k_tiles_per_seg, num_d_tiles],
                align_to=32,
            )
            v_prior_sbuf = allocator.alloc_sbuf_tensor(
                shape=(_V_TILE_SZ, d_tile_size_free_dim),
                dtype=v_cache.dtype,
                block_dim=[num_v_tiles_for_prior, num_d_tiles_free_dim],
                num_free_tiles=[num_v_tiles_for_prior, num_d_tiles_free_dim],
            )
            if kvp_prior_load_blocks > 0:
                for k_idx in range(num_k_tiles_per_seg):
                    for d_idx in range(num_d_tiles):
                        nisa.memset(k_prior_sbuf[k_idx][d_idx][...], value=0.0)
                for v_idx in range(num_v_tiles_for_prior):
                    for d_idx in range(num_d_tiles_free_dim):
                        nisa.memset(v_prior_sbuf[v_idx][d_idx][...], value=0.0)
            load_kv_cache_fn(
                k_cache,
                v_cache,
                block_tables,
                k_prior_sbuf,
                v_prior_sbuf,
                b_i,
                h_i,
                prior_block_offset,
                kvp_prior_load_blocks if kvp_prior_load_blocks > 0 else num_blocks_per_seg,
                allocator,
                k_pre_transposed=k_pre_transposed,
                fp8_packed=fp8_packed,
            )
        else:
            # Non-KVP aliased path: helper allocates k_prior_sbuf / v_prior_sbuf
            # as a list slice onto k_cache_sbuf / v_cache_sbuf after Pass 1
            # (active) has already reduced its result into o_prev_hbm, so the
            # active data isn't clobbered before being consumed.
            k_prior_sbuf = None
            v_prior_sbuf = None
        init_sbuf_addr = allocator.get_current_address()
        if is_kvp:
            _kvp_partial_prior_attention(
                q_hbm=q_hbm,
                k_cache_sbuf=k_cache_sbuf,
                v_cache_sbuf=v_cache_sbuf,
                k_prior_sbuf=k_prior_sbuf,
                v_prior_sbuf=v_prior_sbuf,
                o_prev_hbm=o_prev_hbm,
                neg_max_prev_hbm=neg_max_prev_hbm,
                sum_prev_hbm=sum_prev_hbm,
                o_curr_hbm=o_curr_hbm,
                neg_max_curr_hbm=neg_max_curr_hbm,
                sum_curr_hbm=sum_curr_hbm,
                kvp_offset_active_hbm=kvp_offset_active_hbm,
                kvp_q_offset=kvp_q_offset,
                prior_block_offset=prior_block_offset,
                partial_prior_tokens=partial_prior_tokens,
                num_k_tiles_active=num_k_tiles_active,
                num_v_tiles_active=num_v_tiles_active,
                num_k_tiles_per_seg=num_k_tiles_per_seg,
                num_v_tiles_per_seg=num_v_tiles_per_seg,
                n_grps=n_grps,
                head_dim=head_dim,
                block_size=block_size,
                sb_p=sb_p,
                scale=scale,
                tp_q=tp_q,
                allocator=allocator,
                attention_cte_fn=attention_cte_fn,
                sink=sink,
                k_scale_sb=k_scale_sb,
                active_block_offset=active_block_offset,
                kvp_rank_id=kvp_rank_id,
                kvp_group_size=kvp_group_size,
                sliding_window=sliding_window,
                kvp_prior_load_blocks=kvp_prior_load_blocks,
                kvp_prior_fully_visible=kvp_prior_fully_visible,
            )
            allocator.set_current_address(init_sbuf_addr)
        else:
            # Non-KVP 2-pass path: helper does Pass 1 (active) then aliases
            # k_prior_sbuf / v_prior_sbuf onto k_cache_sbuf / v_cache_sbuf via
            # list slice, loads prior K/V, runs Pass 2 (prior), and reduces
            # results into o_prev_hbm.
            _nonkvp_partial_prior_attention(
                q_hbm=q_hbm,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                k_cache_sbuf=k_cache_sbuf,
                v_cache_sbuf=v_cache_sbuf,
                o_prev_hbm=o_prev_hbm,
                neg_max_prev_hbm=neg_max_prev_hbm,
                sum_prev_hbm=sum_prev_hbm,
                o_curr_hbm=o_curr_hbm,
                neg_max_curr_hbm=neg_max_curr_hbm,
                sum_curr_hbm=sum_curr_hbm,
                prior_block_offset=prior_block_offset,
                partial_prior_tokens=partial_prior_tokens,
                num_k_tiles_active=num_k_tiles_active,
                num_v_tiles_active=num_v_tiles_active,
                num_k_tiles_per_seg=num_k_tiles_per_seg,
                num_v_tiles_per_seg=num_v_tiles_per_seg,
                num_blocks_per_seg=num_blocks_per_seg,
                num_v_tiles_for_prior=num_v_tiles_for_prior,
                b_i=b_i,
                h_i=h_i,
                n_grps=n_grps,
                head_dim=head_dim,
                sb_p=sb_p,
                scale=scale,
                tp_q=tp_q,
                allocator=allocator,
                attention_cte_fn=attention_cte_fn,
                load_kv_cache_fn=load_kv_cache_fn,
                sink=sink,
                k_pre_transposed=k_pre_transposed,
                fp8_packed=fp8_packed,
            )
            allocator.set_current_address(init_sbuf_addr)
        allocator.set_current_address(init_sbuf_addr)

    nl.fori_loop(0, is_partial_reg, _attend_partial_prior)

    is_not_partial_reg = nisa.register_alloc()
    nisa.register_load(dst=is_not_partial_reg, src=is_not_partial_prior_segment)

    def _attend_full_prior(_):
        if is_kvp:
            # KVP: use attention_cte_fn with cp_offset for correct shifted causal masking.
            # _run_groups with use_cp=True is not used here to avoid potential issues
            # with the dynamic range select + fused flash-attention interaction.
            allocator.set_current_address(sbuf_outer)
            init_sbuf_addr = allocator.get_current_address()
            if kvp_group_size > 0:
                nisa.tensor_copy(dst=kvp_seg_offset_sbuf, src=active_block_offset)
                nisa.dma_copy(dst=kvp_seg_offset_hbm, src=kvp_seg_offset_sbuf)
            attention_cte_fn(
                q_hbm,
                None,
                None,
                scale=scale,
                causal_mask=True,
                tp_q=tp_q,
                tp_k=False,
                tp_out=False,
                cache_softmax=True,
                skip_output_normalization=True,
                k_cache_sbuf=k_cache_sbuf[:num_k_tiles_active],
                v_cache_sbuf=v_cache_sbuf[:num_v_tiles_active],
                out_o_hbm=o_prev_hbm,
                out_neg_max_hbm=neg_max_prev_hbm,
                out_sum_hbm=sum_prev_hbm,
                init_sbuf_addr=init_sbuf_addr,
                sink=sink,
                cp_offset=kvp_offset_active_hbm,
                global_cp_deg=1,
                k_scale_sb=k_scale_sb,
                kvp_rank_id=kvp_rank_id,
                kvp_group_size=kvp_group_size,
                block_size=block_size,
                kvp_seg_block_offset=kvp_seg_offset_hbm if kvp_group_size > 0 else None,
                sliding_window=sliding_window,
                kvp_cp_offset_int=kvp_cp_offset_int,
                kvp_seg_block_offset_int=kvp_seg_block_offset_int,
                kvp_k_threshold_sb=bufs.k_threshold_sb,
            )
            allocator.set_current_address(init_sbuf_addr)
        else:
            # Non-KVP: use fused sub-functions for flash-attention accumulation
            ac_a, atp_a = _make_ac_atp(
                seqlen_q, seqlen_k_active, head_dim, q_hbm.dtype, True, scale, tp_q, False, total_sections
            )
            sp_active = SectionParams(
                section_idx=0,
                section_offset=0,
                section_offset_active=0,
                next_section_offset_active=seqlen_k_active,
                section_contains_prefix=False,
                next_section_contains_prefix=False,
                kv_section_idx=0,
            )
            allocator.set_current_address(sbuf_outer)
            _allocate_attention_buffers(allocator, ac_a, atp_a, bufs, sink, k_cache_sbuf, v_cache_sbuf)
            _setup_range_select_bounds(ac_a, atp_a, bufs, allocator, None, None, None, None, batch_id=0)
            sbuf_inner = allocator.get_current_address()
            _run_groups(0, n_grps, ac_a, atp_a, sp_active, bufs, q_hbm, 0, o_prev_hbm, sbuf_inner, sink=sink)

    nl.fori_loop(0, is_not_partial_reg, _attend_full_prior)

    # Load initial stats into SBUF running buffers
    # For partial prior: stats were written to HBM by _attention_cte, load them
    # For KVP no-partial-prior: stats were also written to HBM by attention_cte_fn, load them
    # For non-KVP no-partial-prior: stats are already in bufs from _run_groups
    sm_pat = [[num_grps, sb_p], [1, num_grps]]

    if is_kvp:
        # KVP always uses attention_cte_fn which writes stats to HBM
        nisa.dma_copy(dst=bufs.mm1_running_max, src=neg_max_prev_hbm.ap(pattern=sm_pat, offset=0))
        nisa.dma_copy(dst=bufs.exp_running_sum, src=sum_prev_hbm.ap(pattern=sm_pat, offset=0))
    else:
        is_partial_reg2 = nisa.register_alloc()
        nisa.register_load(dst=is_partial_reg2, src=is_partial_prior_segment)

        def _load_partial_stats(_):
            nisa.dma_copy(dst=bufs.mm1_running_max, src=neg_max_prev_hbm.ap(pattern=sm_pat, offset=0))
            nisa.dma_copy(dst=bufs.exp_running_sum, src=sum_prev_hbm.ap(pattern=sm_pat, offset=0))

        nl.fori_loop(0, is_partial_reg2, _load_partial_stats)

    # --- PRIOR SEGMENTS (section_idx=1, kv_section_idx=0, dynamic loop) ---
    # section_idx=1 triggers accumulation: _write_back_impl loads prev output from o_prev_hbm,
    # applies correction factor, adds fresh PV, writes back. Running stats update in SBUF.
    # kv_section_idx=0 ensures K/V indexing starts at tile 0 (each segment's own SBUF data).
    sp_prior = SectionParams(
        section_idx=1,
        section_offset=0,
        section_offset_active=0,
        next_section_offset_active=seqlen_k_prior,
        section_contains_prefix=False,
        next_section_contains_prefix=False,
        kv_section_idx=0,
    )

    prior_offset_save = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
    nisa.tensor_copy(dst=prior_offset_save, src=prior_block_offset)

    # For interleaved KVP: scratch tensor to detect fully-masked prior segments.
    # A segment is degenerate when min_global_K > max_Q (all K beyond Q's reach).
    kvp_prior_should_compute = None
    if is_kvp and kvp_group_size > 0:
        kvp_prior_should_compute = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)

    num_prior_reg = nisa.register_alloc()
    nisa.register_load(dst=num_prior_reg, src=num_full_prior_segments_i32)

    loop_addr = allocator.get_current_address()

    def _process_prior_segment(_):
        nisa.tensor_scalar(
            dst=prior_block_offset, data=prior_block_offset, op0=nl.subtract, operand0=num_blocks_per_seg
        )

        load_kv_cache_fn(
            k_cache,
            v_cache,
            block_tables,
            k_cache_sbuf,
            v_cache_sbuf,
            b_i,
            h_i,
            prior_block_offset,
            num_blocks_per_seg,
            allocator,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
        )

        allocator.set_current_address(sbuf_outer)
        _allocate_attention_buffers(allocator, ac_p, atp_p, bufs, sink, k_cache_sbuf, v_cache_sbuf)

        # KVP: compute kvp_offset_prior
        if is_kvp:
            if kvp_group_size > 0:
                # Interleaved: cp_offset = global_q_offset (seg_block_offset handles K-side)
                nisa.tensor_copy(dst=kvp_offset_prior_sbuf, src=kvp_q_offset)
            else:
                # Contiguous: kvp_offset_prior = kvp_q_offset - prior_block_offset * block_size
                nisa.tensor_scalar(
                    dst=kvp_offset_prior_sbuf, data=prior_block_offset, op0=nl.multiply, operand0=block_size
                )
                nisa.tensor_tensor(
                    dst=kvp_offset_prior_sbuf, data1=kvp_q_offset, data2=kvp_offset_prior_sbuf, op=nl.subtract
                )
            nisa.dma_copy(dst=kvp_offset_prior_hbm, src=kvp_offset_prior_sbuf)

        prior_cp_offset = kvp_offset_prior_hbm if is_kvp else None

        if is_kvp:
            # KVP: use attention_cte_fn with cp_offset, then reduce into accumulated output
            if kvp_group_size > 0:
                nisa.tensor_copy(dst=kvp_seg_offset_sbuf, src=prior_block_offset)
                nisa.dma_copy(dst=kvp_seg_offset_hbm, src=kvp_seg_offset_sbuf)
            init_sbuf_addr = allocator.get_current_address()
            if kvp_group_size > 0 and kvp_prior_fully_visible:
                # Prior segments are fully visible: use _run_groups with section_idx=1
                # for in-place SBUF accumulation (avoids HBM roundtrip via reduce_one_batch).
                ac_pv, atp_pv = _make_ac_atp(
                    seqlen_q,
                    seqlen_k_prior,
                    head_dim,
                    q_hbm.dtype,
                    False,
                    scale,
                    tp_q,
                    False,
                    total_sections,
                )
                _allocate_attention_buffers(allocator, ac_pv, atp_pv, bufs, sink, k_cache_sbuf, v_cache_sbuf)
                _setup_range_select_bounds(ac_pv, atp_pv, bufs, allocator, None, None, None, None, batch_id=0)
                sbuf_inner_p = allocator.get_current_address()
                _run_groups(0, n_grps, ac_pv, atp_pv, sp_prior, bufs, q_hbm, 0, o_prev_hbm, sbuf_inner_p)
            else:
                # Skip fully-masked prior segments in interleaved KVP mode.
                # A segment is degenerate when its minimum global K position exceeds
                # Q's maximum position: prior_block_offset * stride > cp_offset + seg_size - 1.
                # Merging degenerate segments via reduce_one_batch corrupts the accumulator.
                if kvp_prior_should_compute is not None:
                    stride = kvp_group_size * block_size
                    # should_compute = (prior_block_offset * stride <= kvp_q_offset + seg_size - 1)
                    nisa.tensor_scalar(
                        dst=kvp_prior_should_compute, data=prior_block_offset, op0=nl.multiply, operand0=stride
                    )
                    nisa.tensor_tensor(
                        dst=kvp_prior_should_compute, data1=kvp_q_offset, data2=kvp_prior_should_compute, op=nl.subtract
                    )
                    # result = kvp_q_offset - prior_block_offset * stride
                    # should_compute = (result >= -(seg_size - 1)), i.e. result + seg_size - 1 >= 0
                    nisa.tensor_scalar(
                        dst=kvp_prior_should_compute,
                        data=kvp_prior_should_compute,
                        op0=nl.add,
                        operand0=seqlen_q - 1,
                        op1=nl.greater_equal,
                        operand1=0,
                    )
                    kvp_prior_compute_reg = nisa.register_alloc()
                    nisa.register_load(dst=kvp_prior_compute_reg, src=kvp_prior_should_compute)
                else:
                    kvp_prior_compute_reg = None

                def _kvp_prior_body(_):
                    attention_cte_fn(
                        q_hbm,
                        None,
                        None,
                        scale=scale,
                        causal_mask=True,
                        tp_q=tp_q,
                        tp_k=False,
                        tp_out=False,
                        cache_softmax=True,
                        skip_output_normalization=True,
                        k_cache_sbuf=k_cache_sbuf[:num_k_tiles_per_seg],
                        v_cache_sbuf=v_cache_sbuf[:num_v_tiles_per_seg],
                        out_o_hbm=o_curr_hbm,
                        out_neg_max_hbm=neg_max_curr_hbm,
                        out_sum_hbm=sum_curr_hbm,
                        init_sbuf_addr=init_sbuf_addr,
                        cp_offset=prior_cp_offset,
                        global_cp_deg=1,
                        k_scale_sb=k_scale_sb,
                        kvp_rank_id=kvp_rank_id,
                        kvp_group_size=kvp_group_size,
                        block_size=block_size,
                        kvp_seg_block_offset=kvp_seg_offset_hbm,
                        sliding_window=sliding_window,
                        kvp_k_threshold_sb=bufs.k_threshold_sb,
                    )
                    allocator.set_current_address(init_sbuf_addr)

                    # Reduce current segment into accumulated output
                    softmax_pat = [[num_grps, sb_p], [1, num_grps]]
                    o_pat = [[head_dim, sb_p], [1, head_dim]]
                    num_free = min(num_grps, _MAX_FREE_TILES)
                    neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
                    sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
                    neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
                    sum_curr_sb_buf = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
                    o_prev_sb = allocator.alloc_sbuf_tensor(
                        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
                    )
                    o_curr_sb = allocator.alloc_sbuf_tensor(
                        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
                    )
                    o_new_sb = allocator.alloc_sbuf_tensor(
                        shape=(sb_p, head_dim), dtype=nl.float32, block_dim=[num_grps], num_free_tiles=[num_free]
                    )
                    batch_loop_addr = allocator.get_current_address()
                    reduce_one_batch(
                        o_prev_hbm,
                        neg_max_prev_hbm,
                        sum_prev_hbm,
                        o_curr_hbm,
                        neg_max_curr_hbm,
                        sum_curr_hbm,
                        0,
                        0,
                        num_grps,
                        head_dim,
                        num_grps,
                        sb_p,
                        softmax_pat,
                        o_pat,
                        neg_max_prev_sb,
                        sum_prev_sb,
                        neg_max_curr_sb,
                        sum_curr_sb_buf,
                        o_prev_sb,
                        o_curr_sb,
                        o_new_sb,
                        batch_loop_addr,
                        allocator,
                    )
                    # Reload updated stats into SBUF running buffers
                    softmax_pat_reload = [[num_grps, sb_p], [1, num_grps]]
                    nisa.dma_copy(
                        dst=bufs.mm1_running_max, src=neg_max_prev_hbm.ap(pattern=softmax_pat_reload, offset=0)
                    )
                    nisa.dma_copy(dst=bufs.exp_running_sum, src=sum_prev_hbm.ap(pattern=softmax_pat_reload, offset=0))

                nl.fori_loop(0, kvp_prior_compute_reg if kvp_prior_compute_reg is not None else 1, _kvp_prior_body)
        else:
            _setup_range_select_bounds(ac_p, atp_p, bufs, allocator, None, None, None, None, batch_id=0)
            sbuf_inner_p = allocator.get_current_address()
            _run_groups(0, n_grps, ac_p, atp_p, sp_prior, bufs, q_hbm, 0, o_prev_hbm, sbuf_inner_p)

        allocator.set_current_address(loop_addr)

    nl.fori_loop(0, num_prior_reg, _process_prior_segment)

    # Restore
    nisa.tensor_copy(dst=prior_block_offset, src=prior_offset_save)

    # Write running stats to HBM for caller's normalization
    sm_pat = [[num_grps, sb_p], [1, num_grps]]
    nisa.dma_copy(dst=neg_max_prev_hbm.ap(pattern=sm_pat, offset=0), src=bufs.mm1_running_max)
    nisa.dma_copy(dst=sum_prev_hbm.ap(pattern=sm_pat, offset=0), src=bufs.exp_running_sum)

    allocator.set_current_address(orig_addr)
