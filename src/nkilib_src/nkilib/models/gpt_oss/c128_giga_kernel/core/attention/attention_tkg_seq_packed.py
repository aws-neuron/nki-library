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
This kernel implements attention specifically optimized for Token Generation (TKG, also known as Decode)
scenarios where the active sequence length is small (typically 8 or smaller).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.isa import dge_mode, dma_engine, oob_mode

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import DtypeMode
from ..utils.cross_partition_copy import cross_partition_copy
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, resolve_fp8_e4m3_dtype
from ..utils.stream_shuffle_broadcast import (
    stream_shuffle_broadcast,
)
from ..utils.tp_broadcast import tp_broadcast
from .attention_tkg_utils import (
    AttnTKGConfig,
    TileConstants,
    is_fp8_e4m3,
    is_fp8_e5m2,
    resize_cache_block_len_for_attention_tkg_kernel,
    uses_flash_attention,
)

_MAX_D_HEAD = 128
_MAX_S_PRIOR_ACCURATE_ROPE = 2**17

_MIN_FLOAT32 = float(np.finfo(np.float32).min)
_MAX_FLOAT32 = float(np.finfo(np.float32).max)

# Sentinel value for inactive block slots in active_blocks_table.
# A batch only needs ceil((prior_tokens + active_tokens) / block_len) blocks;
# remaining ABT slots use this value. Indirect DMA loads of the KV cache use
# oob_mode.skip to skip these entries (since -1 is out of bounds), avoiding
# wasted memory bandwidth. The attention mask ensures they don't contribute
# to the output.
INACTIVE_BLOCK_IDX = np.int32(-1)


def attention_tkg_seq_packed(
    q: nl.NkiTensor,
    k_active: nl.NkiTensor,
    v_active: nl.NkiTensor,
    k_prior: nl.NkiTensor,
    v_prior: nl.NkiTensor,
    mask: nl.NkiTensor,
    out: nl.NkiTensor,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    active_blocks_table: nl.NkiTensor,
    q_index_table: nl.NkiTensor,
    seq_id_table: nl.NkiTensor,
    inv_freqs: Optional[nl.NkiTensor] = None,
    rope_pos_ids: Optional[nl.NkiTensor] = None,
    start_pos_ids: Optional[nl.NkiTensor] = None,
    sink: Optional[nl.NkiTensor] = None,
    k_out: Optional[nl.NkiTensor] = None,
    DBG_TENSORS: Optional[tuple] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    accumulator_partial_route_table: Optional[nl.NkiTensor] = None,
    accumulator_group_state_route_table: Optional[nl.NkiTensor] = None,
) -> Tuple[nl.NkiTensor, Optional[nl.NkiTensor]]:
    """Sequence-packed token-generation attention for small ``s_active``.

    This kernel consumes canonical sequence IDs, precomputed Q indexing, packed
    block routing, and prepacked attention masks. Optional hardware-ready
    accumulator routing tables enable row-local associative accumulation
    without specializing the compiled kernel to one schedule. Active K/V
    values are selected from canonical sequence IDs and folded once per
    sequence at finalization. It supports block KV cache and LNC1/LNC2 batch
    sharding only, and can optionally fuse RoPE.

    See nkilib/core/attention/attention_tkg_sequence_packing_design_spec.md for the
    schedule invariant, the metadata contract with a worked example, the accumulator
    merge, the padding contract, and the current limitations.

    Please refer to attention_tkg_torch.attention_tkg_torch for the attention math.

    NOTE: KV cache can have a batch size larger than B when kernel caller decides to add an extra buffer batch
    to KV cache to write garbage data. This is irrelevant to kernel impl which strictly uses the first B batches from
    KV cache in all cases. This is denoted as B+ in the shapes below.

    Dimensions:
        B: Batch size
        H: Number of query heads
        d: Head dimension
        s_active: Active sequence length (current tokens being processed)
        s_prior: Prior sequence length (KV cache length)
        block_len: Block length for block KV cache
        block_count: Number of blocks in KV cache

    Args:
      q: Query tensor. NOTE: Q is scaled with 1/sqrt(d_head) iff. cfg.fuse_rope!
        Shape: if cfg.qk_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, d, H, s_active]
      k_active: Active key tensor.
        Shape:  if cfg.qk_in_sb:
                  [d, B * s_active] (indexing: [d, b * s_active + s])
                else:
                  [B, d, s_active]
      v_active: Active value tensor. Shape [B, 1, s_active, d]
      k_prior: Prior key tensor from block KV cache. Shape
               [B+ * block_count, block_len, d], or
               [B+ * block_count, block_len // 2, d, 2] with fp8_packed.
      v_prior: Prior value tensor from block KV cache. Shape
               [B+ * block_count, block_len, d].
      mask: Prepacked attention masks. Shape
            [total_num_rows, p_max,
             tile_n_sprior * num_slots * H * s_active],
            dtype uint8.
      out: Output tensor.
        Shape: if cfg.out_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, H, d, s_active]
      cfg: Kernel configuration with shapes and performance flags. See `AttnTKGConfig`
      sbm: SBUF memory manager for allocating temporary buffers
      inv_freqs: Inverse frequencies for RoPE. Shape [d // 2, 1]. Required when cfg.fuse_rope is True
      rope_pos_ids: Position IDs for RoPE. Shape [B, s_active]. Required when cfg.fuse_rope or cfg.use_pos_id is True
      start_pos_ids: Per-query SWA window start positions. Shape [B, s_active]. Optional.
                    When None, standard attention (full context) is used.
                    When provided, per-query sliding window attention mask is generated.
      sink: Sink attention tokens. Shape [H, 1] for streaming attention sink tokens
      active_blocks_table: Packed physical block routes. Shape
                           [total_num_rows, num_slots * blocks_per_slot],
                           dtype int32.
      q_index_table: Reusable precomputed Q Tensor-Indirection indices in
                     16-lane snake layout. Shape
                     [total_num_rows, d_head, ceil(slot_bqh / 16)],
                     dtype uint16.
      seq_id_table: Absolute sequence ID for every packed slot. Shape
                    [total_num_rows, num_slots], dtype int32.
      k_out: Output key tensor after RoPE. Populated when cfg.fuse_rope is True, stores k_active after applying RoPE
        Shape: if cfg.k_out_in_sb:
                [d, B * s_active] (indexing [d, b * s_active + s])
              else:
                [B, 1, d, s_active].
      DBG_TENSORS: Optional tuple of 4-5 debug tensors with shared HBM type for intermediate value inspection.
                  Expects:
                    - QK: Result of Q@K^T.
                    - QK_MAX: Result of max reduction of QK.
                    - QK_EXP: Result of exp(QK).
                    - EXP_SUM: Result of sum(exp(QK)).
                    - ACTIVE_TABLE: (only use with block KV) Result after loading the active blocks table.
                  See implementation for shapes of these tensors.
      dtype_mode: Quantization dtype policy for the SBUF K/V tile allocations.
                  When ``k_prior.dtype`` / ``v_prior.dtype`` is concrete
                  (``nl.float8_e4m3`` or ``nl.float8_e4m3fn``), the kernel uses
                  it directly. When the caller leaves the dtype as the opaque
                  ``"float8e4"`` sentinel, ``dtype_mode`` selects the variant:
                  ``NON_OCP`` → ``nl.float8_e4m3``, ``OCP`` → ``nl.float8_e4m3fn``,
                  ``AUTO`` → ``nl.float8_e4m3fn`` on TRN3 else ``nl.float8_e4m3``.
                  Compiler enforces a single E4M3 variant per traced module
                  (``EOCP001``); pick one variant for the whole call graph.
      accumulator_partial_route_table: Optional Tensor-Indirection offsets that
                                       scatter partial records into group/source
                                       positions. Shape
                                       [total_num_rows, d_head,
                                        ceil(num_slots * H * s_active / 16)],
                                       dtype uint16.
      accumulator_group_state_route_table: Optional Tensor-Indirection offsets
                                           for gathering and scattering the
                                           persistent state of each group.
                                           Same shape and dtype as the partial
                                           route. Both routes are reused
                                           independently for max, sum, and PV.

    Returns:
      out: Attention output tensor.
        Shape: if cfg.out_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, H, d, s_active]
      k_out: Key output tensor.
        Shape: if cfg.k_out_in_sb:
                [d, B * s_active] (indexing [d, b * s_active + s])
              else:
                [B, 1, d, s_active]

    FEATURES:

    1. Flexible Tensor Placement:
      - q, k, k_out, and out tensors can be placed in either SBUF or HBM
      - When qk_in_sb=True, q and k tensors are pre-loaded in SBUF (required for block KV cache)
      - out_in_sb and k_out_in_sb flags control output tensor placement for reduced memory transfers
      - Use this feature for performance improvement when integrating this kernel into a larger kernel

    2. Batch-only LNC2 Sharding:
      - LNC1 runs the full batch on one NeuronCore
      - LNC2 assigns each NeuronCore a disjoint, equally sized batch slice
      - Sequence sharding is not supported for packed attention

    3. Packed Mask Loading:
      - Each physical schedule row has a pre-aligned mask for all compute slots
      - Sequence and tile routing, including terminal active tokens, is encoded by the caller

    4. Fused RoPE (Rotary Position Embedding):
      - fuse_rope integrates RoPE computation directly into the attention kernel
      - Applies rotary embeddings to Q and K tensors, scaling Q by 1/sqrt(d_head)
      - Reduces memory traffic by avoiding separate RoPE passes

    5. Block KV Cache:
      - Supports block-sparse KV cache with configurable block_len
      - Uses active_blocks_table to track which cache blocks are active per batch
      - Enables efficient long-context inference with sparse memory access patterns

    6. K_prior Transpose Handling:
      - Block KV requires tp_k_prior=True
      - K_prior is [num_blocks, block_len, d] and is transposed during block loading

    7. Strided Memory Access (strided_mm1):
      - Enables strided read patterns for K in first matmul
      - When enabled, allows MM2 to use sequential V reads for better DMA throughput
      - Trades off MM1 memory access for MM2 optimization

    8. Attention Sink:
      - Supports streaming attention with sink tokens for infinite context
      - Sink tokens maintain fixed attention scores across all positions
      - Folds each sink once into persistent softmax state after all packed rows

    9. GPSIMD SBUF-to-SBUF Transfers:
      - use_gpsimd_sb2sb enables high-performance GPSIMD instructions for inter-core communication
      - Optimizes LNC2 sharding by using extended instructions for SBUF-to-SBUF data transfers

    10. Context Length Management:
      - curr_sprior: Current prior sequence length (actual KV cache content for this invocation)
      - full_sprior: Full prior sequence length (maximum KV cache capacity allocated)
      - Allows progressive filling of KV cache during autoregressive generation

    11. Stack-based SBUF Allocation:
      - Uses SbufManager for efficient on-chip memory management
      - Hierarchical scoping with interleave_degree for multi-bank utilization
      - Automatic alignment and temporary buffer lifecycle management

    IMPLEMENTATION DETAILS:

      The kernel goes through the following steps:
        -1. Setup of intermediate buffers, mask, block KV, and debug tensors.
         0. Perform rope if fuse_rope is set
         1. Performs the KQ^T computation.
          - Loop over each batch
          - Load the current chunk of K based on configuration (block KV, transpose, etc.)
          - Tile over the multiplication of K and Q in groups of 4k size
         2. Compute the max reduction of KQ^T computation.
          - Compute the max in tiles of size 128 over bs * q_head * s_active
          - Prepare the sink if used
          - Transpose and broadcast along the partition dimension
         3. Compute Exp(KQ^T - max(KQ^T))
          - Add/subtract the max based on whether it was negated
          - Apply the exponentiation activation
         4. Compute sum reduction of the exponentiation result
          - Compute the sum in tiles of size 128 over bs * q_head * s_active
          - Perform additional reductions based on sink or other optimization flags
          - Compute the reciprocal with the same tiling scheme, and then broadcast
         5. Compute the product of the above and V and store the result
          - Loop over each batch
          - Load the current chunk of V based on configuration (same as step 1)
          - Perform the matmul over sprior tiles
          - If needed, copy information over core boundaries or to HBM

    INTENDED USAGE:

      This kernel is optimized for cases when there are few active tokens.
      Use with s_active <= 7, and with d_head <= 128.

    Notes:
        - KV cache can have batch size larger than B (denoted B+) for garbage data buffering
        - Q is scaled with 1/sqrt(d_head) only when cfg.fuse_rope is True
        - Block KV cache requires qk_in_sb=True
        - LNC2 uses batch sharding and requires B to divide evenly across logical NCs
        - Packed schedules require an even number of compute slots per NC
        - Extended GPSIMD instructions require 16-partition alignment

    Pseudocode:
        # Setup
        TC = get_tile_constants()
        atp = compute_tile_params(cfg, TC, q, active_blocks_table)
        bufs = allocate_internal_buffers()

        # Step 0: Optional RoPE
        if cfg.fuse_rope:
            q_sb, k_active_sb = apply_rope(q, k_active, inv_freqs, rope_pos_ids)

        loop over flash_attention_tile_idx:
            # Step 1: Compute KQ^T
            for batch_idx in range(bs):
                k_sb = load_k_prior_and_active(k_prior, k_active, batch_idx)
                qk[batch_idx] = matmul(k_sb, q_sb[batch_idx])  # Tiled in 4k groups

            # Step 2: Max reduction
            qk_max = reduce_max(qk, axis=s_prior)  # Cascaded reduction
            if sprior_n_prgs > 1:
                qk_max = sendrecv_and_reduce(qk_max)
            if sink is not None:
                qk_max = reduce_with_sink(qk_max, sink)

            # Step 3: Compute exp(QK - max)
            qk_exp = exp(qk - qk_max)

            # Step 4: Sum reduction and reciprocal
            exp_sum = reduce_sum(qk_exp, axis=s_prior)  # Cascaded reduction
            if sprior_n_prgs > 1:
                exp_sum = sendrecv_and_add(exp_sum)
            if sink is not None:
                exp_sum = add_sink_contribution(exp_sum, sink, qk_max)
            exp_sum_recip = reciprocal(exp_sum)

            # Step 5: Compute (exp @ V)^T
            for batch_idx in range(bs):
                v_sb = load_v_prior_and_active(v_prior, v_active, batch_idx)
                exp_v[batch_idx] = matmul(v_sb, qk_exp[batch_idx]) * exp_sum_recip[batch_idx]

            if sprior_n_prgs > 1:
                exp_v = sendrecv_and_add(exp_v)

        finalize_flash_attention_and_store_output(exp_v, out)
    """

    kernel_assert(
        mask != None,
        "Packed attention requires a prepacked attention mask.",
    )
    kernel_assert(q_index_table != None, "Packed attention requires a Q-index table.")
    kernel_assert(seq_id_table != None, "Packed attention requires a sequence-ID table.")
    kernel_assert(active_blocks_table != None, "Packed attention requires a packed block table.")

    TC = TileConstants.get_tile_constants()
    atp = _compute_tile_params(cfg, TC, q, k_prior, v_prior, k_active, v_active, active_blocks_table, dtype_mode)

    # Keep persistent batch ownership on atp; packed schedule and slot geometry
    # live independently on spp.
    _set_atp_batch_dims(atp, atp.bs_per_nc, TC)
    spp = _compute_sequence_packing_params(
        atp,
        cfg,
        TC,
        mask,
        q_index_table,
        seq_id_table,
        accumulator_partial_route_table,
        accumulator_group_state_route_table,
        active_blocks_table,
    )
    # Packed rows may cover an arbitrary subset of persistent sequences even
    # when each slot processes only one FA tile, so accumulation is always
    # required independently of the maximum sequence length.

    bufs = AttnInternalBuffers()

    _setup_block_kv_cache(
        k_prior,
        v_prior,
        k_active,
        v_active,
        active_blocks_table,
        atp,
        spp,
        cfg,
        TC,
        sbm,
        bufs,
    )

    if DBG_TENSORS:
        _setup_debug_tensors(DBG_TENSORS, atp, TC, bufs)

    bufs.one_vec = sbm.alloc_stack(
        (TC.p_max, 1), dtype=atp.io_type, buffer=nl.sbuf, align=4
    )  # align to 4 bytes to prevent race condition (TODO: fix properly in NKILIB-876)
    nisa.memset(bufs.one_vec, value=1.0)

    # Load position IDs (needed for RoPE and mask generation)
    _load_position_ids(rope_pos_ids, start_pos_ids, atp, cfg, TC, sbm, bufs)

    # Step 0. Optional RoPE
    if cfg.fuse_rope:
        _perform_rope(q, k_active, inv_freqs, k_out, atp, cfg, TC, sbm, bufs)
    else:
        kernel_assert(
            cfg.qk_in_sb,
            "Currently only suppport skipping fusing RoPE when QK is in SBUF (qk_in_sb==True).",
        )
        bufs.q_sb = q
        bufs.k_active_sb = k_active

    # =========================================================================
    # SEQUENCE PACKING SCHEDULE LOOP
    # =========================================================================
    # Replaces the batched outer loop (for batch_tile → for fa_tile) with a
    # schedule-driven loop over this NC's assigned rows from the global schedule.
    # In packed mode, total_num_rows comes directly from schedule metadata and
    # is independent of any kernel-derived tile count. Each row describes one slot-compute
    # iteration; the schedule determines which sequence tile occupies each slot.
    # =========================================================================

    # Persistent accumulators/output stay B-sized. Only the FA tile body switches
    # to the independently tuned slot width.
    persistent_bs = atp.bs_per_nc
    btc = BatchTileContext(
        batch_tile_idx=0,
        tile_bs=persistent_bs,
        tile_batch_offset=0,
        global_batch_offset=atp.bs_prg_id * atp.bs_per_nc,
    )
    # Route validation requires both tables, so the optimized merge is selected
    # whenever the caller supplies its runtime routing metadata.
    use_grouped_accumulator = accumulator_partial_route_table != None

    sbm.open_scope()

    _allocate_online_softmax_buffers(
        atp,
        cfg,
        sbm,
        bufs,
        use_grouped_accumulator=use_grouped_accumulator,
    )

    # atp remains in its original B-sized batch-tile geometry. Packed compute
    # helpers select their S-sized dimensions explicitly from spp.
    #
    # Keep two compact completed-row handoffs outside the row-local scope. While
    # row N+1 performs metadata/KV DMA and QK Tensor Engine work, row N can merge
    # into the persistent DVE accumulator without retaining row N's large QK
    # scratch buffers. The accumulator calls remain ordered across rows.
    handoff_ping = _allocate_packed_row_handoff(0, atp, spp, cfg, sbm)
    handoff_pong = _allocate_packed_row_handoff(1, atp, spp, cfg, sbm)

    # All per-row metadata is loop invariant in geometry, so fetch this NC's
    # whole slice once. This keeps small metadata DMAs off the critical path
    # ahead of each row's indirect K/V load.
    _preload_packed_row_metadata(active_blocks_table, q_index_table, atp, spp, cfg, TC, sbm, bufs)

    seq_ids_width = spp.num_iters_per_nc * spp.num_slots
    bufs.seq_ids_all_sb = sbm.alloc_stack((1, seq_ids_width), dtype=nl.int32, buffer=nl.sbuf, align=4)
    nisa.dma_copy(
        dst=bufs.seq_ids_all_sb,
        src=seq_id_table.reshape((seq_id_table.shape[0] * seq_id_table.shape[1],)).ap(
            [[0, 1], [1, seq_ids_width]],
            offset=spp.schedule_row_base * spp.num_slots,
        ),
        name=f"{sbm.get_name_prefix()}packed_seq_ids_load_all",
    )
    if use_grouped_accumulator:
        _preload_packed_accumulator_routes(
            accumulator_partial_route_table,
            accumulator_group_state_route_table,
            atp,
            spp,
            cfg,
            sbm,
            bufs,
        )

    for packed_iter_idx in range(spp.num_iters_per_nc):
        schedule_row = spp.schedule_row_base + packed_iter_idx
        fa_ctx = _compute_packed_fa_tile_context(packed_iter_idx, schedule_row, atp, TC)
        current_handoff = handoff_ping if packed_iter_idx % 2 == 0 else handoff_pong

        sbm.open_scope()
        _prepare_fa_tile_body(
            active_blocks_table,
            q_index_table,
            seq_id_table,
            mask,
            k_prior,
            DBG_TENSORS,
            atp,
            spp,
            cfg,
            TC,
            sbm,
            bufs,
            fa_ctx,
            btc,
        )

        # Prepare the current row before merging the previous handoff. This
        # exposes current DMA/QK work while the independent accumulator merge
        # uses compute engines, without retaining the previous QK scratch.
        if packed_iter_idx > 0:
            previous_iter_idx = packed_iter_idx - 1
            previous_handoff = handoff_ping if previous_iter_idx % 2 == 0 else handoff_pong
            if use_grouped_accumulator:
                _grouped_packed_accumulator_update(
                    atp,
                    spp,
                    cfg,
                    sbm,
                    bufs,
                    previous_handoff,
                    previous_iter_idx,
                )
            else:
                _sequence_packing_accumulator_update(
                    atp,
                    spp,
                    cfg,
                    sbm,
                    bufs,
                    previous_handoff,
                )

        _finish_fa_tile_body(
            v_prior,
            v_active,
            DBG_TENSORS,
            atp,
            spp,
            cfg,
            TC,
            sbm,
            bufs,
            fa_ctx,
            btc,
        )
        _snapshot_packed_row_handoff(current_handoff, atp, spp, cfg, bufs)
        sbm.close_scope()

    # The final handoff has no following row behind which to merge, so drain it
    # explicitly after the schedule loop.
    final_iter_idx = spp.num_iters_per_nc - 1
    final_handoff = handoff_ping if final_iter_idx % 2 == 0 else handoff_pong
    if use_grouped_accumulator:
        _grouped_packed_accumulator_update(
            atp,
            spp,
            cfg,
            sbm,
            bufs,
            final_handoff,
            final_iter_idx,
        )
    else:
        _sequence_packing_accumulator_update(
            atp,
            spp,
            cfg,
            sbm,
            bufs,
            final_handoff,
        )
    _materialize_packed_accumulator_stats(atp, sbm, bufs)

    _finalize_and_store(sink, out, atp, spp, cfg, TC, sbm, bufs, btc, DBG_TENSORS=DBG_TENSORS)

    sbm.close_scope()

    return out, k_out


def _gather_active_kv_by_seqid(
    seq_id_table,
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
    fa_ctx: "FATileContext",
):
    """Prepare active K/V routing for the current schedule row."""
    num_slots = spp.num_slots
    # Slice this row's IDs out of the table loaded once per NC.
    row_in_nc = fa_ctx.schedule_row - spp.schedule_row_base
    bufs.seq_ids_sb = bufs.seq_ids_all_sb[0:1, row_in_nc * num_slots : (row_in_nc + 1) * num_slots]

    # Padded slots retain -1 for grouping. Route a separate safe copy to
    # sequence zero; the packed mask keeps their active positions invisible.
    bufs.safe_seq_ids_sb = sbm.alloc_stack((1, num_slots), dtype=nl.int32, buffer=nl.sbuf, align=4)
    nisa.tensor_scalar(
        dst=bufs.safe_seq_ids_sb,
        data=bufs.seq_ids_sb,
        op0=nl.maximum,
        operand0=0,
    )


def _select_q_by_seqid(
    q_index_table,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    TC: "TileConstants",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
    fa_ctx: "FATileContext",
):
    """Gather Q columns by seq_id via DVE Tensor Indirection.

    q_index_table is the Q-select index table: [total_num_rows, d_head, cols_q]
    uint16 in snake layout. Each iteration's slice [d_head, cols_q] contains the column
    offsets to gather num_slots * s_active_qh columns from bufs.q_sb.

    Result is stored in bufs.q_gathered: [d_head, num_slots * s_active_qh] with slots
    in schedule order. The i_b loop in _compute_qk_matmul then indexes into this buffer
    instead of bufs.q_sb directly.
    """
    num_elem = spp.slot_s_active_bqh
    cols_q = (num_elem + 15) // 16  # ceil(num_elem / 16)

    # TI index tensors need dense storage strides, so copy this row's slice out
    # of the per-NC preload into its own dense buffer (DVE, not DMA).
    row_in_nc = fa_ctx.schedule_row - spp.schedule_row_base
    q_idx_sbuf = sbm.alloc_stack((cfg.d_head, cols_q), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=q_idx_sbuf,
        src=(bufs.q_index_all_sb).select(1, row_in_nc),
        engine=nisa.vector_engine,
    )

    # Gather Q columns via TI
    # Publish the indirection instead of materializing it. With row tiling the Q
    # replication pass below reads this view directly, so the gather costs no
    # extra pass over Q. Without row tiling there is no replication to fold
    # into, so materialize as before.
    bufs.q_ti_index = q_idx_sbuf
    bufs.q_ti_num_elem = num_elem
    if atp.qk_row_tile_factor > 1:
        bufs.q_gathered = None
    else:
        bufs.q_gathered = sbm.alloc_stack((cfg.d_head, num_elem), dtype=bufs.q_sb.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=bufs.q_gathered,
            src=bufs.q_sb.indirect(q_idx_sbuf, num_elem=num_elem),
            engine=nisa.vector_engine,
        )


def _transpose_sbuf_tiled(dst, src, partition_size, free_size):
    """Transpose SBUF views, tiling only when Vector Engine limits require it."""
    transpose_tile_size = 32
    if partition_size <= transpose_tile_size and free_size <= transpose_tile_size:
        nisa.nc_transpose(dst=dst, data=src, engine=nisa.vector_engine)
        return

    for partition_tile in range(div_ceil(partition_size, transpose_tile_size)):
        partition_start = partition_tile * transpose_tile_size
        partition_extent = min(transpose_tile_size, partition_size - partition_start)
        for free_tile in range(div_ceil(free_size, transpose_tile_size)):
            free_start = free_tile * transpose_tile_size
            free_extent = min(transpose_tile_size, free_size - free_start)
            nisa.nc_transpose(
                dst=dst[
                    free_start : free_start + free_extent,
                    partition_start : partition_start + partition_extent,
                ],
                data=src[
                    partition_start : partition_start + partition_extent,
                    free_start : free_start + free_extent,
                ],
                engine=nisa.vector_engine,
            )


def _transpose_sbuf_columns_to_row(dst, src, num_elem, partition_tile_size, n_tiles):
    """Flatten tiled SBUF columns into partition zero of a TI source."""
    for tile_idx in range(n_tiles):
        tile_start = tile_idx * partition_tile_size
        tile_extent = min(partition_tile_size, num_elem - tile_start)
        _transpose_sbuf_tiled(
            dst=dst[0:1, tile_start : tile_start + tile_extent],
            src=src[0:tile_extent, tile_idx : tile_idx + 1],
            partition_size=tile_extent,
            free_size=1,
        )


def _transpose_sbuf_row_to_columns(dst, src, num_elem, partition_tile_size, n_tiles):
    """Unflatten partition zero of a TI source into tiled SBUF columns."""
    for tile_idx in range(n_tiles):
        tile_start = tile_idx * partition_tile_size
        tile_extent = min(partition_tile_size, num_elem - tile_start)
        _transpose_sbuf_tiled(
            dst=dst[0:tile_extent, tile_idx : tile_idx + 1],
            src=src[0:1, tile_start : tile_start + tile_extent],
            partition_size=1,
            free_size=tile_extent,
        )


@dataclass
class PackedRowHandoff(nl.NKIObject):
    """Compact online-softmax result retained across schedule-row scopes.

    Two instances form a ping-pong buffer. They preserve only the data needed
    to merge a completed row while the following row reuses the large QK, exp,
    and PV scratch allocations.

    ``tile_max`` and ``tile_sum`` retain the column-tiled statistic layout
    ``[slot_s_active_bqh_tile, slot_n_bsq_tiles]``. ``tile_output`` stores the
    unnormalized PV result in slot-major ``[d_head, slot_s_active_bqh]`` layout.
    ``seq_ids`` stores the row's absolute sequence IDs as ``[1, num_slots]`` for
    the serial fallback accumulator.
    """

    tile_max: nl.NkiTensor = None
    tile_sum: nl.NkiTensor = None
    tile_output: nl.NkiTensor = None
    seq_ids: nl.NkiTensor = None


@dataclass
class PackedAccumulatorContext(nl.NKIObject):
    """SBUF workspace for merging one runtime-routed packed schedule row.

    Let ``D`` be ``d_head``, ``S`` the number of slots, ``A`` the active query
    lanes per slot, and ``G = S`` the maximum number of row-local sequence
    groups. Each softmax record contains three fields: max, sum, and PV.

    ``partial_records`` has shape ``[D, 3, S, A]`` and holds one completed
    record per slot. ``partial_records_flat`` is an alias used for allocation
    and copies.

    ``staged_records`` has shape ``[D, 3, G, S + 1, A]``. Source zero contains
    a group's prior persistent state; the remaining sources receive this row's
    routed slot records. ``staged_records_flat`` aliases the same storage.

    ``running_state_flat`` aliases the persistent per-sequence accumulator.
    ``prior_records`` gathers the state needed by this row into
    ``[D, 3, G, A]``; ``prior_records_flat`` is its allocation alias.

    ``new_state`` has shape ``[D, 3, G, A]`` and holds the merged group
    records before they are scattered back to persistent state.
    ``new_state_flat`` aliases the same storage.

    ``correction_factors`` has shape ``[D, G, A, S + 1]`` and contains each
    source's exponential correction to its group's new maximum.
    ``scaled_updates`` has shape ``[D, 2, G, A, S + 1]`` and holds the
    corrected sum and PV sources before reduction.

    ``partial_route_indices`` routes each slot/query lane to a group/source
    cell. ``group_state_route_indices`` gathers and scatters persistent state
    for each group, routing unused groups to the dummy state. Both are
    row-specific ``uint16`` Tensor Indirection indices copied from the per-NC
    tables preloaded in SBUF.
    """

    partial_records: nl.NkiTensor = None
    partial_records_flat: nl.NkiTensor = None
    staged_records: nl.NkiTensor = None
    staged_records_flat: nl.NkiTensor = None
    running_state_flat: nl.NkiTensor = None
    new_state: nl.NkiTensor = None
    new_state_flat: nl.NkiTensor = None
    correction_factors: nl.NkiTensor = None
    scaled_updates: nl.NkiTensor = None
    partial_route_indices: nl.NkiTensor = None
    group_state_route_indices: nl.NkiTensor = None
    prior_records: nl.NkiTensor = None
    prior_records_flat: nl.NkiTensor = None


def _allocate_packed_row_handoff(
    handoff_idx: int,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    sbm: "SbufManager",
) -> PackedRowHandoff:
    """Allocate one persistent compact row result for software pipelining."""
    handoff = PackedRowHandoff()
    handoff.tile_max = sbm.alloc_stack(
        (spp.slot_s_active_bqh_tile, spp.slot_n_bsq_tiles),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}packed_handoff_max_{handoff_idx}",
    )
    handoff.tile_sum = sbm.alloc_stack(
        (spp.slot_s_active_bqh_tile, spp.slot_n_bsq_tiles),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}packed_handoff_sum_{handoff_idx}",
    )
    handoff.tile_output = sbm.alloc_stack(
        (cfg.d_head, spp.slot_s_active_bqh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
        name=f"{sbm.get_name_prefix()}packed_handoff_output_{handoff_idx}",
    )
    handoff.seq_ids = sbm.alloc_stack(
        (1, spp.num_slots),
        dtype=nl.int32,
        buffer=nl.sbuf,
        align=4,
        name=f"{sbm.get_name_prefix()}packed_handoff_seq_ids_{handoff_idx}",
    )
    return handoff


def _snapshot_packed_row_handoff(
    handoff: PackedRowHandoff,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    bufs: "AttnInternalBuffers",
):
    """Retain only the completed FA partials needed by the accumulator."""
    nisa.tensor_copy(
        dst=handoff.tile_max,
        src=bufs.qk_max_buf[: spp.slot_s_active_bqh_tile, : spp.slot_n_bsq_tiles],
        engine=nisa.scalar_engine,
    )
    nisa.tensor_copy(
        dst=handoff.tile_sum,
        src=bufs.exp_sum[: spp.slot_s_active_bqh_tile, : spp.slot_n_bsq_tiles],
        engine=nisa.scalar_engine,
    )
    nisa.tensor_copy(
        dst=handoff.tile_output[:, 0 : spp.slot_s_active_bqh],
        src=bufs.exp_v[: cfg.d_head, :, :].reshape((cfg.d_head, spp.slot_s_active_bqh)),
        engine=nisa.scalar_engine,
    )
    nisa.tensor_copy(
        dst=handoff.seq_ids,
        src=bufs.seq_ids_sb,
        engine=nisa.scalar_engine,
    )


def _allocate_slot_local_accumulator_context(
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
) -> PackedAccumulatorContext:
    """Allocate grouped workspace for one packed row.

    For ``S`` slots, the fixed workspace has ``S`` possible groups and ``S+1``
    sources per group. Source zero is the group's prior persistent state;
    sources one through ``S`` are the current row's slot partials.
    """
    ctx = PackedAccumulatorContext()
    num_groups = spp.num_slots
    num_sources = spp.num_slots + 1
    record_size = 3 * atp.s_active_qh
    num_elem = spp.num_slots * record_size
    index_cols = div_ceil(spp.num_slots * atp.s_active_qh, 16)

    ctx.partial_records_flat = sbm.alloc_stack(
        (cfg.d_head, spp.num_slots * record_size),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    ctx.partial_records = ctx.partial_records_flat.reshape((cfg.d_head, 3, spp.num_slots, atp.s_active_qh))
    ctx.staged_records_flat = sbm.alloc_stack(
        (cfg.d_head, num_groups * num_sources * record_size),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    ctx.staged_records = ctx.staged_records_flat.reshape((cfg.d_head, 3, num_groups, num_sources, atp.s_active_qh))
    ctx.running_state_flat = bufs.running_state_flat
    ctx.new_state_flat = sbm.alloc_stack(
        (cfg.d_head, 3 * num_groups * atp.s_active_qh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    ctx.new_state = ctx.new_state_flat.reshape((cfg.d_head, 3, num_groups, atp.s_active_qh))
    ctx.correction_factors = sbm.alloc_stack(
        (cfg.d_head, num_groups, atp.s_active_qh, num_sources),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    ctx.scaled_updates = sbm.alloc_stack(
        (cfg.d_head, 2, num_groups, atp.s_active_qh, num_sources),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    ctx.partial_route_indices = sbm.alloc_stack(
        (cfg.d_head, index_cols),
        dtype=nl.uint16,
        buffer=nl.sbuf,
        align=4,
    )
    ctx.group_state_route_indices = sbm.alloc_stack(
        (cfg.d_head, index_cols),
        dtype=nl.uint16,
        buffer=nl.sbuf,
        align=4,
    )
    ctx.prior_records_flat = sbm.alloc_stack(
        (cfg.d_head, num_elem),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    ctx.prior_records = ctx.prior_records_flat.reshape((cfg.d_head, 3, num_groups, atp.s_active_qh))
    return ctx


def _initialize_slot_local_accumulator_context(
    ctx: PackedAccumulatorContext,
    local_row: int,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    bufs: "AttnInternalBuffers",
) -> None:
    """Load this row's routes and seed every group with persistent state.

    A used group gathers the state for its sequence. An unused group gathers
    the extra dummy record, so fixed-width routing remains valid for padding.
    """
    nisa.tensor_copy(
        dst=ctx.partial_route_indices,
        src=bufs.accumulator_partial_route_all_sb.select(dim=1, index=local_row),
        engine=nisa.scalar_engine,
    )
    nisa.tensor_copy(
        dst=ctx.group_state_route_indices,
        src=bufs.accumulator_group_state_route_all_sb.select(dim=1, index=local_row),
        engine=nisa.scalar_engine,
    )

    # Initialize all source cells to the online-softmax identity before routed
    # records overwrite selected cells. Sources absent from a group therefore
    # remain no-ops without data-dependent control flow.
    nisa.memset(ctx.staged_records[:, 1:3, :, :, :], value=0.0)
    nisa.memset(
        ctx.staged_records[:, 0, :, :, :],
        value=(_MAX_FLOAT32 if atp.max_negated else _MIN_FLOAT32),
    )

    field_num_elem = spp.num_slots * atp.s_active_qh
    running_state_fields = ctx.running_state_flat.reshape((cfg.d_head, 3, (atp.bs_per_nc + 1) * atp.s_active_qh))
    for field_idx in range(3):
        nisa.tensor_copy(
            dst=ctx.prior_records.select(dim=1, index=field_idx).reshape((cfg.d_head, field_num_elem)),
            src=running_state_fields.select(dim=1, index=field_idx).indirect(
                ctx.group_state_route_indices,
                num_elem=field_num_elem,
            ),
            engine=nisa.scalar_engine,
        )
    nisa.tensor_copy(
        dst=ctx.staged_records.select(dim=3, index=0),
        src=ctx.prior_records,
        engine=nisa.scalar_engine,
    )


def _prepare_slot_local_accumulator_partials(
    ctx: PackedAccumulatorContext,
    row_handoff: PackedRowHandoff,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
) -> None:
    """Convert a handoff into the common slot-major record layout.

    Max and sum arrive in the softmax reduction's tiled-column layout and must
    be transposed into flat Tensor Indirection sources. PV is already
    ``[d_head, slot_s_active_bqh]`` and only needs a copy.
    """
    num_elem = spp.num_slots * atp.s_active_qh
    partial_max = ctx.partial_records[:, 0, :, :].reshape((cfg.d_head, num_elem))
    _transpose_sbuf_columns_to_row(
        dst=partial_max,
        src=row_handoff.tile_max,
        num_elem=num_elem,
        partition_tile_size=spp.slot_s_active_bqh_tile,
        n_tiles=spp.slot_n_bsq_tiles,
    )
    partial_sum = ctx.partial_records[:, 1, :, :].reshape((cfg.d_head, num_elem))
    _transpose_sbuf_columns_to_row(
        dst=partial_sum,
        src=row_handoff.tile_sum,
        num_elem=num_elem,
        partition_tile_size=spp.slot_s_active_bqh_tile,
        n_tiles=spp.slot_n_bsq_tiles,
    )
    nisa.tensor_copy(
        dst=ctx.partial_records[:, 2, :, :].reshape((cfg.d_head, num_elem)),
        src=row_handoff.tile_output,
        engine=nisa.scalar_engine,
    )


def _grouped_packed_accumulator_update(
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
    row_handoff: PackedRowHandoff,
    local_row: int,
) -> None:
    """Merge one row with runtime routes rather than serial slot updates.

    Slots for the same sequence route to one row-local group and are reduced
    together with that sequence's prior state. The resulting group state is
    then scattered back to the persistent sequence accumulator.
    """
    sbm.open_scope(name="slot_local_packed_accumulator")
    ctx = _allocate_slot_local_accumulator_context(
        atp,
        spp,
        cfg,
        sbm,
        bufs,
    )
    _initialize_slot_local_accumulator_context(
        ctx,
        local_row,
        atp,
        spp,
        cfg,
        bufs,
    )
    _prepare_slot_local_accumulator_partials(
        ctx,
        row_handoff,
        atp,
        spp,
        cfg,
    )
    _aggregate_packed_accumulator(
        ctx,
        atp,
        spp.num_slots,
    )
    sbm.close_scope()


def _aggregate_packed_accumulator(
    ctx: PackedAccumulatorContext,
    atp: "AttnTileParams",
    total_slots: int,
) -> None:
    """Route one row, merge online-softmax records, and persist each group.

    Tensor Indirection scatters slot records into fixed group/source cells.
    Source zero already contains the group's previous accumulator. Identity
    values in all untouched cells make padding and absent sources no-ops.
    """
    num_sources = total_slots + 1
    stats_width = 2 * total_slots * atp.s_active_qh
    partial_stats = ctx.partial_records[:, 0:2, :, :].reshape((ctx.partial_records.shape[0], stats_width))
    # Max and sum are scalar per query lane and initially occupy partition zero;
    # PV already carries one value per d_head partition.
    stream_shuffle_broadcast(
        src=partial_stats[0:1, :],
        dst=partial_stats,
    )
    field_num_elem = total_slots * atp.s_active_qh
    staged_fields = ctx.staged_records.reshape(
        (
            ctx.staged_records.shape[0],
            3,
            total_slots * num_sources * atp.s_active_qh,
        )
    )
    partial_fields = ctx.partial_records.reshape((ctx.partial_records.shape[0], 3, field_num_elem))
    # Scatter each slot into its group's source cell. The route is shared by
    # max, sum, and PV because those fields use the same slot/group geometry.
    for field_idx in range(3):
        nisa.tensor_copy(
            dst=staged_fields.select(dim=1, index=field_idx).indirect(
                ctx.partial_route_indices,
                num_elem=field_num_elem,
            ),
            src=partial_fields.select(dim=1, index=field_idx),
            engine=nisa.scalar_engine,
        )

    # Put source on the reduction axis so each group's prior state and routed
    # slot partials can be combined in one operation per field.
    staged = ctx.staged_records.permute([0, 1, 2, 4, 3])
    max_sources = staged[:, 0, :, :, :]
    update_sources = staged[:, 1:3, :, :, :]
    new_max = ctx.new_state[:, 0, :, :]
    new_updates = ctx.new_state[:, 1:3, :, :]
    # Associatively combine online-softmax records: choose the new maximum,
    # rescale every source's sum/PV to that maximum, then add the sources.
    nisa.tensor_reduce(
        dst=new_max,
        data=max_sources,
        op=(nl.minimum if atp.max_negated else nl.maximum),
        axis=3,
    )
    new_max_bc = new_max.expand_dim(dim=3).broadcast(dim=3, size=num_sources)
    nisa.tensor_tensor(
        dst=ctx.correction_factors,
        data1=new_max_bc if atp.max_negated else max_sources,
        data2=max_sources if atp.max_negated else new_max_bc,
        op=nl.subtract,
    )
    nisa.activation(
        dst=ctx.correction_factors,
        op=nl.exp,
        data=ctx.correction_factors,
    )

    for field_idx in range(2):
        scaled_field = ctx.scaled_updates.select(dim=1, index=field_idx)
        nisa.tensor_tensor(
            dst=scaled_field,
            data1=update_sources.select(dim=1, index=field_idx),
            data2=ctx.correction_factors,
            op=nl.multiply,
        )
        nisa.tensor_reduce(
            dst=new_updates.select(dim=1, index=field_idx),
            data=scaled_field,
            op=nl.add,
            axis=3,
        )
    running_state_fields = ctx.running_state_flat.reshape(
        (ctx.running_state_flat.shape[0], 3, (atp.bs_per_nc + 1) * atp.s_active_qh)
    )
    new_state_fields = ctx.new_state.reshape((ctx.new_state.shape[0], 3, field_num_elem))
    # Used groups scatter to their NC-local sequence state; unused groups write
    # only the dummy record and cannot affect a real sequence.
    for field_idx in range(3):
        nisa.tensor_copy(
            dst=running_state_fields.select(dim=1, index=field_idx).indirect(
                ctx.group_state_route_indices,
                num_elem=field_num_elem,
            ),
            src=new_state_fields.select(dim=1, index=field_idx),
            engine=nisa.scalar_engine,
        )


def _sequence_packing_accumulator_update(
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
    row_handoff: PackedRowHandoff,
):
    """Merge one packed schedule row into persistent accumulators in slot order.

    Each slot consumes its fixed partial slice and updates its NC-local sequence
    destination immediately, so a later slot holding another tile of the same
    sequence observes the preceding write. Padded slots produce no-op partials
    through their all-zero packed mask and inactive block routes.
    """
    num_slots = spp.num_slots
    s_active_qh = atp.s_active_qh
    num_elem = num_slots * s_active_qh

    sbm.open_scope()

    schedule_nc_id = spp.schedule_row_base // spp.num_iters_per_nc
    sequence_offset = schedule_nc_id * atp.bs_per_nc
    local_destinations_i32 = sbm.alloc_stack((1, num_slots), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=local_destinations_i32,
        data=row_handoff.seq_ids,
        op0=nl.subtract,
        operand0=sequence_offset,
    )
    nisa.tensor_scalar(
        dst=local_destinations_i32,
        data=local_destinations_i32,
        op0=nl.maximum,
        operand0=0,
    )
    local_destinations = local_destinations_i32.view(nl.uint32)

    tile_max = row_handoff.tile_max
    tile_sum = row_handoff.tile_sum
    tile_output = row_handoff.tile_output

    partial_max_sources = sbm.alloc_stack((cfg.d_head, num_elem), dtype=atp.inter_type, buffer=nl.sbuf)
    _transpose_sbuf_columns_to_row(
        dst=partial_max_sources,
        src=tile_max,
        num_elem=num_elem,
        partition_tile_size=spp.slot_s_active_bqh_tile,
        n_tiles=spp.slot_n_bsq_tiles,
    )
    stream_shuffle_broadcast(
        src=partial_max_sources[0:1, 0:num_elem],
        dst=partial_max_sources[:, 0:num_elem],
    )

    partial_sum_sources = sbm.alloc_stack((cfg.d_head, num_elem), dtype=atp.inter_type, buffer=nl.sbuf)
    _transpose_sbuf_columns_to_row(
        dst=partial_sum_sources,
        src=tile_sum,
        num_elem=num_elem,
        partition_tile_size=spp.slot_s_active_bqh_tile,
        n_tiles=spp.slot_n_bsq_tiles,
    )
    stream_shuffle_broadcast(
        src=partial_sum_sources[0:1, 0:num_elem],
        dst=partial_sum_sources[:, 0:num_elem],
    )

    # Bundle each destination's max, sum, and output fields so every ordered
    # slot update needs only one dynamic gather and one immediate scatter.
    #
    # The merge stays sequential across slots on purpose. Slot work is unique
    # per row, but a row may still hold several tiles of the SAME sequence, so
    # two slots can share one accumulator destination. A row-wide merge would
    # read that destination once and lose the earlier slot's contribution.
    running_state_groups = bufs.running_state.reshape_dim(dim=2, shape=[atp.bs_per_nc, s_active_qh])

    state_record = sbm.alloc_stack((cfg.d_head, 3, s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf)
    state_max = state_record[:, 0, :]
    state_sum = state_record[:, 1, :]
    state_output = state_record[:, 2, :]
    new_max = sbm.alloc_stack((cfg.d_head, s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf)
    state_correction = sbm.alloc_stack((cfg.d_head, s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf)
    slot_correction = sbm.alloc_stack((cfg.d_head, s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf)
    scaled_sum = sbm.alloc_stack((cfg.d_head, s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf)
    scaled_output = sbm.alloc_stack((cfg.d_head, s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf)

    for slot_idx in range(num_slots):
        slot_start = slot_idx * s_active_qh
        slot_end = (slot_idx + 1) * s_active_qh
        partial_max = partial_max_sources[:, slot_start:slot_end]
        partial_sum = partial_sum_sources[:, slot_start:slot_end]
        partial_output = tile_output[:, slot_start:slot_end]

        destination = local_destinations[0:1, slot_idx : slot_idx + 1]
        selected_state = running_state_groups.select(dim=2, index=destination)
        nisa.tensor_copy(dst=state_record, src=selected_state, engine=nisa.scalar_engine)

        nisa.tensor_tensor(
            dst=new_max,
            data1=state_max,
            data2=partial_max,
            op=(nl.minimum if atp.max_negated else nl.maximum),
        )
        nisa.tensor_tensor(
            dst=state_correction,
            data1=new_max if atp.max_negated else state_max,
            data2=state_max if atp.max_negated else new_max,
            op=nl.subtract,
        )
        nisa.tensor_tensor(
            dst=slot_correction,
            data1=new_max if atp.max_negated else partial_max,
            data2=partial_max if atp.max_negated else new_max,
            op=nl.subtract,
        )
        nisa.activation(dst=state_correction, op=nl.exp, data=state_correction)
        nisa.activation(dst=slot_correction, op=nl.exp, data=slot_correction)

        nisa.tensor_tensor(
            dst=state_sum,
            data1=state_sum,
            data2=state_correction,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=scaled_sum,
            data1=partial_sum,
            data2=slot_correction,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=state_sum,
            data1=state_sum,
            data2=scaled_sum,
            op=nl.add,
        )

        nisa.tensor_tensor(
            dst=state_output,
            data1=state_output,
            data2=state_correction,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=scaled_output,
            data1=partial_output,
            data2=slot_correction,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=state_output,
            data1=state_output,
            data2=scaled_output,
            op=nl.add,
        )
        nisa.tensor_copy(dst=state_max, src=new_max, engine=nisa.scalar_engine)

        # Preserve dependency order for repeated destinations: the complete
        # record is visible before the next slot performs its dynamic gather.
        nisa.tensor_copy(dst=selected_state, src=state_record, engine=nisa.scalar_engine)

    sbm.close_scope()


def _materialize_packed_accumulator_stats(
    atp: "AttnTileParams",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
):
    """Materialize canonical partition-zero fields for finalization."""
    _transpose_sbuf_row_to_columns(
        dst=bufs.running_max,
        src=bufs.accumulator_max,
        num_elem=atp.s_active_bqh,
        partition_tile_size=atp.s_active_bqh_tile,
        n_tiles=atp.n_bsq_tiles,
    )
    _transpose_sbuf_row_to_columns(
        dst=bufs.running_sum,
        src=bufs.accumulator_sum,
        num_elem=atp.s_active_bqh,
        partition_tile_size=atp.s_active_bqh_tile,
        n_tiles=atp.n_bsq_tiles,
    )


def _prepare_fa_tile_body(
    active_blocks_table,
    q_index_table,
    seq_id_table,
    mask,
    k_prior,
    DBG_TENSORS,
    atp,
    spp,
    cfg,
    TC,
    sbm,
    bufs,
    fa_ctx,
    btc,
):
    """Prepare one row and launch its DMA/QK work before prior accumulation."""
    _load_and_reshape_active_blk_table(active_blocks_table, atp, spp, sbm, bufs, fa_ctx)
    _allocate_qk_buffers(atp, spp, TC, sbm, bufs, fa_ctx)
    _load_mask(mask, TC, bufs, fa_ctx, sbm)
    _gather_active_kv_by_seqid(seq_id_table, spp, cfg, sbm, bufs, fa_ctx)
    _select_q_by_seqid(q_index_table, atp, spp, cfg, TC, sbm, bufs, fa_ctx)
    _compute_qk_matmul(k_prior, DBG_TENSORS, atp, spp, cfg, TC, sbm, bufs, fa_ctx, btc)


def _finish_fa_tile_body(
    v_prior,
    v_active,
    DBG_TENSORS,
    atp,
    spp,
    cfg,
    TC,
    sbm,
    bufs,
    fa_ctx,
    btc,
):
    """Finish a sink-free row partial before persistent accumulation."""
    _cascaded_max_reduce(None, DBG_TENSORS, atp, spp, cfg, TC, sbm, bufs, fa_ctx, btc)
    _compute_exp_qk(DBG_TENSORS, atp, spp, TC, sbm, bufs, fa_ctx, btc)
    _cascaded_sum_reduction(None, DBG_TENSORS, atp, spp, cfg, TC, sbm, bufs, fa_ctx, btc)
    _compute_pv_matmul_and_store(v_prior, v_active, atp, spp, cfg, TC, sbm, bufs, fa_ctx, btc)


OOB_MODE_SKIP = oob_mode.skip  # FIXME: needs to be instantiated externally from kernel


@dataclass
class SequencePackingParams(nl.NKIObject):
    """Static packed-schedule and slot-compute geometry."""

    # Schedule geometry
    num_slots: int = 0
    num_iters_per_nc: int = 0
    schedule_row_base: int = 0

    # Slot-sized compute geometry
    slot_s_active_bqh: int = 0
    slot_s_active_bqh_remainder: int = 0
    slot_n_bsq_full_tiles: int = 0
    slot_n_bsq_tiles: int = 0
    slot_s_active_bqh_tile: int = 0
    slot_batch_interleave_degree: int = 0


@dataclass
class AttnTileParams(nl.NKIObject):
    """Computed tiling and dimension parameters for the attention kernel.

    This dataclass holds all the computed parameters needed for tiling the attention
    computation, including data types, sharding information, dimension calculations,
    and flash attention parameters.

    Fields are grouped into:
    - Global parameters: fixed for the entire kernel invocation
    - Persistent batch geometry
    - Softmax reduction parameters

    Packed schedule and slot-compute geometry intentionally live in
    ``SequencePackingParams`` and never overwrite these batch fields.
    """

    # ========== Global parameters (fixed for entire kernel invocation) ==========

    # Data types
    io_type = None
    """Data type for input/output tensors (e.g., bfloat16, float32). Derived from query tensor dtype."""

    inter_type = None
    """Data type for intermediate computations (typically float32 for numerical stability)."""

    k_prior_load_type = None
    """Data type used when loading k_prior. For FP8 KV cache, this is bfloat16; otherwise matches k_prior.dtype."""

    # Block KV cache flag
    is_block_kv: bool = None
    """Whether block KV cache is being used (True when active_blocks_table is provided)."""

    # KV FP8 Quantization Flag
    is_fp8_kv: bool = None
    """Whether FP8 quantization is used for KV cache tensors. When True, all KV
    tensors must have the same FP8 E4M3 dtype (``nl.float8_e4m3`` or
    ``nl.float8_e4m3fn``)."""

    kv_e4m3_tile_dtype: Any = None
    """Concrete FP8 E4M3 dtype for SBUF K/V tiles when ``is_fp8_kv`` is True.
    Uses caller's concrete ``k_prior.dtype`` when explicit; resolves from
    ``dtype_mode`` only when the caller passed the opaque ``"float8e4"``
    sentinel. ``None`` when ``is_fp8_kv`` is False."""

    # DMA transpose optimization flag
    use_dma_transpose: bool = None
    """Whether to use DMA transpose for block KV loading. True when d_head==128 and dtype is 2 bytes."""

    # Sharding parameters
    sprior_n_prgs: int = None
    """Number of programs (NeuronCores) that s_prior is sharded across. Either 1 or 2 for LNC2."""

    sprior_prg_id: int = None
    """Program ID for s_prior sharding (0 or 1). Each program handles AttnTileParams.s_prior // AttnTileParams.sprior_n_prgs elements."""

    bs_n_prgs: int = None
    """Number of programs (NeuronCores) that batch dimension is sharded across. Either 1 or 2 for LNC2."""

    bs_prg_id: int = None
    """Program ID for batch sharding (0 or 1). Each program handles AttnTileParams.bs // AttnTileParams.bs_n_prgs batches."""

    n_prgs: int = None
    """Total number of programs. Equals AttnTileParams.sprior_n_prgs * AttnTileParams.bs_n_prgs (either 1 or 2)."""

    # Batch size parameters
    bs_full: int = None
    """Full batch size before any sharding is applied. Equals AttnTKGConfig.bs."""

    bs_per_nc: int = None
    """Persistent sequence count on this NC before batch tiling."""

    # Sequence and head dimensions (not batch-dependent)
    s_prior: int = None
    """Prior sequence length per program after sharding. Equals AttnTKGConfig.curr_sprior // AttnTileParams.sprior_n_prgs."""

    s_active_qh: int = None
    """Flattened dimension of [q_head, s_active]. Equals AttnTKGConfig.s_active * AttnTKGConfig.q_head."""

    n_sprior_tile: int = None
    """Total number of TileConstants.p_max-sized tiles across the post-sharded s_prior dimension. Equals ceil(AttnTileParams.s_prior / TileConstants.p_max)."""

    qk_row_tile_factor: int = None
    """Row-tiling factor for d_head=64 block KV with DMA transpose. When 2, two K positions are
    packed per 128-partition column (using both halves of the partition dim), enabling row-tiled
    matmul with two nc_matmul calls per tile. 1 otherwise (standard single-matmul path)."""
    # Block KV cache parameters (if used)
    block_len: int = None
    """Block length for block KV cache. 0 for flat KV cache, or adjusted AttnTKGConfig.block_len after resizing."""

    blk_cache_resize_factor: int = None
    """Resize factor for block KV cache. Original block_len / resized block_len. 1 when no resize needed."""

    use_v_dma_skipping: bool = False
    """Whether to use DMA skipping for V load (block KV). DMA skipping currently doesn't help since we become gpsimd/compute bound
    in the majority of configurations, and it adds an extra memset. In addition, DMA batching does not currently support DMA skipping
    (uCode-407). In future we can flip this switch.
    """

    # Flash attention parameters
    use_fa: bool = None
    """Whether flash attention tiling is enabled. True when AttnTileParams.s_prior > FA_TILE_SIZE (8K)."""

    fa_tile_s_prior: int = None
    """Size of each flash attention tile in s_prior dimension. Equals FA_TILE_SIZE (8K) when FA enabled."""

    fa_n_sprior_tile: int = None
    """Number of TileConstants.p_max-sized tiles within each FA tile. Equals ceil(AttnTileParams.fa_tile_s_prior / TileConstants.p_max)."""

    # ========== Per-batch-tile parameters (recomputed by _update_atp_for_batch_tile) ==========

    bs: int = None
    """Batch size for the current batch tile. May be smaller than bs_per_nc for the last tile."""

    s_active_bqh: int = None
    """Flattened dimension of [bs, q_head, s_active] for the current batch tile. Equals AttnTileParams.bs * AttnTileParams.s_active_qh."""

    s_active_bqh_remainder: int = None
    """Remainder when AttnTileParams.s_active_bqh doesn't evenly divide into TileConstants.p_max tiles. Equals AttnTileParams.s_active_bqh % TileConstants.p_max."""

    n_bsq_full_tiles: int = None
    """Number of full TileConstants.p_max-sized tiles that fit in AttnTileParams.s_active_bqh. Equals AttnTileParams.s_active_bqh // TileConstants.p_max."""

    n_bsq_tiles: int = None
    """Total number of tiles needed for AttnTileParams.s_active_bqh (including partial). Equals AttnTileParams.n_bsq_full_tiles + (1 if remainder > 0)."""

    s_active_bqh_tile: int = None
    """Size of each BSQ tile. Equals TileConstants.p_max if multiple tiles needed, otherwise AttnTileParams.s_active_bqh."""

    batch_interleave_degree: int = None
    """Degree of interleaving across batches for PSUM bank utilization. Min of AttnTileParams.bs and TileConstants.psum_b_max (8)."""

    # Softmax reduction parameters ==========

    num_folds_per_batch: int = None
    """Number of 128-block folds for the current FA tile in block KV cache. Set by _load_and_reshape_active_blk_table.
    Equals (blocks_per_batch * resize_factor) / TileConstants.p_max."""

    softmax_final_reduction_length: int = None
    """Number of elements in final softmax reduction = 1 (local) + (sink slot if per-tile softmax sync
    is used — see AttnTileParams.sync_softmax_per_fa_tile). Sharded cases defer the cross-NC sync to
    _finalize_and_store; sink is loaded locally inside _finalize_and_store in that case."""

    softmax_final_reduction_local_idx: int = None
    """Index in reduction buffer for local NC's result. Always 0."""

    softmax_final_reduction_sink_idx: int = None
    """Index in reduction buffer for sink contribution. None unless sink is staged into qk_max_buf for
    per-tile softmax sync; else AttnTileParams.softmax_final_reduction_length - 1."""

    sync_softmax_per_fa_tile: bool = None
    """Whether to synchronize softmax statistics (max/sum) across NeuronCores after each FA tile.
    True when not sharded on s_prior. When False (sharded), the cross-NC sync is deferred to
    _finalize_and_store so sendrecv does not block GPSIMD V prefetches.
    """

    max_negated: bool = None
    """Whether the max values are stored negated (for exp computation optimization with sink)."""


@dataclass
class AttnInternalBuffers(nl.NKIObject):
    """Internal SBUF buffers needed across multiple steps of the attention kernel.

    This dataclass holds all temporary SBUF buffers that are allocated during
    kernel execution and shared across different computation steps.
    """

    # Core attention tensors
    qk: nl.NkiTensor = None
    """QK^T result buffer. Shape [TileConstants.p_max, FATileContext.tile_n_sprior * AttnTileParams.s_active_bqh]. Filled with -inf initially for masking."""

    qk_io_type: nl.NkiTensor = None
    """QK buffer in AttnTileParams.io_type (e.g., bfloat16) for matmuls. Same shape as qk, stores exp(QK - max) after softmax."""

    qk_max: nl.NkiTensor = None
    """Per-position max of QK^T for softmax stability. Shape [TileConstants.p_max, AttnTileParams.s_active_bqh]."""

    qk_max_buf: nl.NkiTensor = None
    """Buffer for max reduction across tiles and LNC2 cores. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles * AttnTileParams.softmax_final_reduction_length]."""

    exp_sum: nl.NkiTensor = None
    """Sum of exp(QK - max) for softmax normalization. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles * AttnTileParams.softmax_final_reduction_length]."""

    exp_sum_recip: nl.NkiTensor = None
    """Reciprocal of exp_sum, broadcasted for final normalization. Shape [TileConstants.p_max, AttnTileParams.s_active_bqh]. Not used when FA enabled."""

    exp_v: nl.NkiTensor = None
    """Result of softmax(QK) @ V matmul. Shape [AttnTKGConfig.d_head, AttnTileParams.bs, AttnTileParams.s_active_qh]. Contains unnormalized output for FA."""

    # Preprocessed inputs
    q_sb: nl.NkiTensor = None
    """Query tensor in SBUF after optional RoPE. Shape [AttnTKGConfig.d_head, AttnTileParams.bs_full * AttnTileParams.s_active_qh]. Scaled by 1/sqrt(AttnTKGConfig.d_head) if fuse_rope."""

    q_ti_index: nl.NkiTensor = None
    """This row's Q tensor-indirection index tensor, or None on the identity path."""

    q_ti_num_elem: int = None
    """Gathered Q width (num_slots * s_active_qh) for the indirection view."""

    q_gathered: nl.NkiTensor = None
    """TI-gathered Q columns for the current packed iteration. Shape [d_head, num_slots * s_active_qh].
    None when no packing table is provided (falls back to positional indexing via q_sb)."""

    k_active_sb: nl.NkiTensor = None
    """Active key tensor in SBUF after optional RoPE. Shape [d_head, bs_full * s_active]."""

    seq_ids_sb: nl.NkiTensor = None
    """Current schedule row's absolute sequence IDs. Shape [1, num_slots]."""

    seq_ids_all_sb: nl.NkiTensor = None
    """This NC's whole sequence-ID table, loaded once. Shape [1, num_iters_per_nc * num_slots]."""

    accumulator_partial_route_all_sb: nl.NkiTensor = None
    """All local rows' slot-to-group/source TI routes.
    Shape [d_head, num_iters_per_nc, index_cols]."""

    accumulator_group_state_route_all_sb: nl.NkiTensor = None
    """All local rows' group-to-persistent-state TI routes.
    Shape [d_head, num_iters_per_nc, index_cols]."""

    active_blocks_all_sb: nl.NkiTensor = None
    """This NC's whole packed block table, slot-major, loaded once.
    Shape [p_max, num_iters_per_nc, num_slots * num_folds]."""

    active_blocks_all_sb_u32: nl.NkiTensor = None
    """uint32 view of active_blocks_all_sb for indirect DMA indices."""

    q_index_all_sb: nl.NkiTensor = None
    """This NC's whole Q Tensor-Indirection table, loaded once.
    Shape [d_head, num_iters_per_nc, ceil(slot_bqh / 16)]."""

    safe_seq_ids_sb: nl.NkiTensor = None
    """Nonnegative sequence IDs for padded-slot K/V indirection. Shape [1, num_slots]."""

    pos_ids_sb: nl.NkiTensor = None
    """Position IDs broadcasted to all partitions for RoPE/mask generation. Shape [TileConstants.p_max, AttnTileParams.bs_per_nc * AttnTKGConfig.s_active]."""

    start_pos_sb: nl.NkiTensor = None
    """Per-query SWA window start positions broadcasted to all partitions. Shape [TileConstants.p_max, AttnTileParams.bs * AttnTKGConfig.s_active]. None when SWA is disabled."""

    mask_sb: nl.NkiTensor = None
    """Attention mask in SBUF. Shape [TileConstants.p_max, FATileContext.tile_n_sprior * AttnTileParams.s_active_bqh]. Values: 1 for valid, 0 for masked."""

    one_vec: nl.NkiTensor = None
    """Vector of ones for sum reduction via matmul. Shape [TileConstants.p_max, 1]."""

    # Block KV cache buffers
    active_blocks_sb: nl.NkiTensor = None
    """Active block indices in SBUF for block KV cache. Shape [TileConstants.p_max, AttnTileParams.num_folds_per_batch * AttnTileParams.bs].
    Loaded per FA tile and batch tile by _load_and_reshape_active_blk_table. Contains block indices."""

    active_blocks_sb_u32: nl.NkiTensor = None
    """Pre-cast uint32 copy of active_blocks_sb for DMA transpose path. Avoids per-fold int32→uint32 cast in hot loop."""

    v_active_reshaped: nl.NkiTensor = None
    """Reshaped v_active for block KV loading. Shape [AttnTKGConfig.bs, AttnTKGConfig.s_active * AttnTKGConfig.d_head]."""

    k_prior_reshaped: nl.NkiTensor = None
    """Reshaped k_prior cache for block-sparse access.
    Shape [num_blocks * resize_factor, block_len * d_head] (or [num_blocks * resize_factor, block_len//2 * d_head * 2] fp8 when fp8_packed)."""

    v_prior_reshaped: nl.NkiTensor = None
    """Reshaped v_prior cache for block-sparse access. Shape [num_blocks * resize_factor, AttnTileParams.block_len * AttnTKGConfig.d_head]."""

    # Debug tensors (reshaped from DBG_TENSORS)
    DBG_QK: nl.NkiTensor = None
    """Debug tensor for QK^T results. Shape [TileConstants.p_max, AttnTileParams.sprior_n_prgs, AttnTileParams.n_sprior_tile, AttnTileParams.bs_n_prgs, AttnTileParams.s_active_bqh]."""

    DBG_QK_MAX: nl.NkiTensor = None
    """Debug tensor for QK max values. Shape [AttnTileParams.bs_n_prgs, AttnTileParams.n_bsq_tiles, AttnTileParams.s_active_bqh_tile]."""

    DBG_QK_EXP: nl.NkiTensor = None
    """Debug tensor for exp(QK - max). Shape [TileConstants.p_max, AttnTileParams.sprior_n_prgs, AttnTileParams.n_sprior_tile, AttnTileParams.bs_n_prgs, AttnTileParams.s_active_bqh]."""

    DBG_EXP_SUM: nl.NkiTensor = None
    """Debug tensor for exp sum values. Shape [AttnTileParams.bs_n_prgs, AttnTileParams.n_bsq_tiles, AttnTileParams.s_active_bqh_tile]."""

    DBG_ACTIVE_TABLE: nl.NkiTensor = None
    """Debug tensor for active blocks table (block KV only). Shape [TileConstants.p_max, AttnTileParams.num_folds_per_batch * AttnTileParams.sprior_n_prgs, AttnTileParams.bs_full]."""

    # Flash attention buffers
    running_max: nl.NkiTensor = None
    """Running max across FA tiles for online softmax. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]. Updated each FA tile."""

    running_sum: nl.NkiTensor = None
    """Running sum of exp values across FA tiles. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]. Accumulated each FA tile."""

    running_state: nl.NkiTensor = None
    """Real-sequence view of running_state_flat.
    Shape [d_head, 3, s_active_bqh], with fields max, sum, and output."""

    running_state_flat: nl.NkiTensor = None
    """Dense backing allocation used by Tensor Indirection.
    Grouped routing appends one s_active_qh-wide dummy sequence record for
    unused groups; running_state excludes that record."""

    accumulator_max: nl.NkiTensor = None
    """Max-field view of running_state in [d_head, s_active_bqh] layout."""

    accumulator_sum: nl.NkiTensor = None
    """Sum-field view of running_state in [d_head, s_active_bqh] layout."""

    correction_factor: nl.NkiTensor = None
    """Correction factor exp(prev_max - curr_max) for rescaling. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]."""

    running_output: nl.NkiTensor = None
    """Output-field view of running_state in [d_head, s_active_bqh] layout."""


@dataclass
class FATileContext(nl.NKIObject):
    """Context for the current FA tile being processed.

    This holds tile-specific parameters that vary per FA tile iteration.
    Functions inside the FA loop should use these values instead of
    AttnTileParams.fa_tile_s_prior / AttnTileParams.fa_n_sprior_tile which are max values.

    Flash attention tiles process s_prior in chunks of fa_tile_size (8K).
    The last tile may be smaller than fa_tile_size if s_prior is not evenly divisible.
    """

    fa_tile_idx: int
    """NC-local packed schedule-row index."""

    schedule_row: int
    """Physical metadata row. Packed LNC2 tables flatten [NC0 rows][NC1 rows]."""

    tile_s_prior: int
    """Fixed s_prior width represented by each packed slot."""

    tile_n_sprior: int
    """Number of TileConstants.p_max-sized tiles within this FA tile. Equals ceil(FATileContext.tile_s_prior / TileConstants.p_max)."""

    tile_offset: int
    """Local packed-buffer offset. Always zero because metadata rows are pre-aligned."""

    is_last_fa_tile: bool
    """Always true for packed rows: every row is a complete, pre-aligned FA tile."""


def _compute_packed_fa_tile_context(
    local_iter: int,
    schedule_row: int,
    atp: AttnTileParams,
    TC: TileConstants,
) -> FATileContext:
    """Build the uniform slot-compute context for one packed schedule row.

    K/V block rows and masks are already pre-aligned by the framework, so the
    local buffer always represents one full FA tile starting at offset zero.
    Active K/V is not part of the row at all: it is folded into the running
    statistics once per sequence at finalization.
    """
    return FATileContext(
        fa_tile_idx=local_iter,
        schedule_row=schedule_row,
        tile_s_prior=atp.fa_tile_s_prior,
        tile_n_sprior=atp.fa_n_sprior_tile,
        tile_offset=0,
        is_last_fa_tile=True,
    )


@dataclass
class BatchTileContext(nl.NKIObject):
    """Context for the current batch tile being processed.

    When the full per-NC batch size is too large to fit in SBUF, the batch dimension
    is tiled. This dataclass tracks the current batch tile's parameters.
    """

    batch_tile_idx: int
    """Which batch tile is being processed (0-indexed)."""

    tile_bs: int
    """Batch size for this tile. May be smaller than batch_tile_size for the last tile."""

    tile_batch_offset: int
    """Offset within the NC's batch portion where this tile starts. Equals batch_tile_idx * batch_tile_size."""

    global_batch_offset: int
    """Offset into the full (all-NC) batch dimension. Equals bs_prg_id * bs_per_nc + tile_batch_offset.
    Use as: global_batch_offset + i_b to index into full-batch tensors (q_sb, k_prior, v_prior, etc.)."""


def _compute_sequence_packing_params(
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    mask,
    q_index_table,
    seq_id_table,
    accumulator_partial_route_table,
    accumulator_group_state_route_table,
    active_blocks_table,
) -> SequencePackingParams:
    """Validate packed metadata and derive static per-NC slot geometry.

    Accumulator routes are optional as a pair. Supplying both selects the
    grouped associative update; omitting both preserves the serial slot-order
    fallback.
    """
    kernel_assert(
        mask != None,
        "Packed attention requires a prepacked attention mask.",
    )
    kernel_assert(q_index_table != None, "Packed attention requires a Q-index table.")
    kernel_assert(seq_id_table != None, "Packed attention requires a sequence-ID table.")
    kernel_assert(active_blocks_table != None, "Packed attention requires a packed block table.")
    kernel_assert(atp.is_block_kv, "Packed attention requires block KV cache routing.")
    kernel_assert(cfg.block_len > 0, "Packed attention requires a positive block length.")
    kernel_assert(
        atp.fa_tile_s_prior % cfg.block_len == 0,
        "Packed FA tile size must divide evenly into cache blocks.",
    )
    kernel_assert(
        len(active_blocks_table.shape) == 2,
        f"Packed block table must be rank 2, got shape {active_blocks_table.shape}.",
    )
    kernel_assert(
        active_blocks_table.dtype == nl.int32,
        f"Packed block table must use int32, got {active_blocks_table.dtype}.",
    )
    kernel_assert(
        len(q_index_table.shape) == 3,
        f"Packed Q-index table must be rank 3, got shape {q_index_table.shape}.",
    )
    kernel_assert(
        q_index_table.dtype == nl.uint16,
        f"Packed Q-index table must use uint16, got {q_index_table.dtype}.",
    )
    kernel_assert(
        len(seq_id_table.shape) == 2,
        f"Packed sequence-ID table must be rank 2, got shape {seq_id_table.shape}.",
    )
    kernel_assert(
        seq_id_table.dtype == nl.int32,
        f"Packed sequence-ID table must use int32, got {seq_id_table.dtype}.",
    )
    has_any_accumulator_route = accumulator_partial_route_table != None or accumulator_group_state_route_table != None
    has_all_accumulator_routes = accumulator_partial_route_table != None and accumulator_group_state_route_table != None
    kernel_assert(
        not has_any_accumulator_route or has_all_accumulator_routes,
        "Packed accumulator TI tables must be supplied together.",
    )
    if has_all_accumulator_routes:
        for route_table in (
            accumulator_partial_route_table,
            accumulator_group_state_route_table,
        ):
            kernel_assert(
                len(route_table.shape) == 3,
                f"Packed accumulator TI table must be rank 3, got shape {route_table.shape}.",
            )
            kernel_assert(
                route_table.dtype == nl.uint16,
                f"Packed accumulator TI table must use uint16, got {route_table.dtype}.",
            )
    kernel_assert(
        len(mask.shape) == 3,
        f"Packed attention mask must be rank 3, got shape {mask.shape}.",
    )
    kernel_assert(
        mask.dtype == nl.uint8,
        f"Packed attention mask must use uint8, got {mask.dtype}.",
    )

    spp = SequencePackingParams()
    blocks_per_slot = atp.fa_tile_s_prior // cfg.block_len
    kernel_assert(
        active_blocks_table.shape[1] % blocks_per_slot == 0,
        "Packed block-table width must be an exact multiple of blocks_per_tile.",
    )
    spp.num_slots = active_blocks_table.shape[1] // blocks_per_slot
    kernel_assert(spp.num_slots > 0, "Packed attention requires at least one compute slot per NC.")
    kernel_assert(
        spp.num_slots % 2 == 0,
        "Packed attention requires an even number of compute slots per NC.",
    )
    kernel_assert(
        spp.num_slots <= atp.bs_per_nc,
        "Packed accumulator routing requires num_slots <= per-NC batch size.",
    )

    total_num_rows = q_index_table.shape[0]
    expected_q_index_columns = div_ceil(spp.num_slots * atp.s_active_qh, 16)
    expected_accumulator_index_columns = div_ceil(
        spp.num_slots * atp.s_active_qh,
        16,
    )
    expected_mask_width = atp.fa_n_sprior_tile * spp.num_slots * atp.s_active_qh
    kernel_assert(
        total_num_rows == active_blocks_table.shape[0],
        "Packed Q-index and block tables must have the same row count.",
    )
    kernel_assert(
        q_index_table.shape[1] == cfg.d_head and q_index_table.shape[2] == expected_q_index_columns,
        "Packed Q-index rows must have shape [d_head, ceil(slot_bqh / 16)].",
    )
    kernel_assert(
        seq_id_table.shape[0] == total_num_rows,
        "Packed sequence-ID and Q-index tables must have the same row count.",
    )
    kernel_assert(
        seq_id_table.shape[1] == spp.num_slots,
        "Packed sequence-ID table width must equal the number of compute slots.",
    )
    if has_all_accumulator_routes:
        for route_table in (
            accumulator_partial_route_table,
            accumulator_group_state_route_table,
        ):
            kernel_assert(
                route_table.shape[0] == total_num_rows
                and route_table.shape[1] == cfg.d_head
                and route_table.shape[2] == expected_accumulator_index_columns,
                "Packed accumulator TI rows must have shape [d_head, ceil(num_slots * s_active_qh / 16)].",
            )
    kernel_assert(
        mask.shape[0] == total_num_rows,
        "Packed mask and Q-index tables must have the same row count.",
    )
    kernel_assert(
        mask.shape[1] == TC.p_max and mask.shape[2] == expected_mask_width,
        "Packed mask rows must have shape [p_max, tile_n_sprior * slot_bqh].",
    )
    kernel_assert(
        total_num_rows % atp.n_prgs == 0,
        "Packed schedule rows must divide evenly across logical NCs.",
    )
    schedule_nc_id = atp.sprior_prg_id if atp.sprior_n_prgs > 1 else atp.bs_prg_id
    spp.num_iters_per_nc = total_num_rows // atp.n_prgs
    spp.schedule_row_base = schedule_nc_id * spp.num_iters_per_nc

    spp.slot_s_active_bqh = spp.num_slots * atp.s_active_qh
    spp.slot_s_active_bqh_remainder = spp.slot_s_active_bqh % TC.p_max
    spp.slot_n_bsq_full_tiles = spp.slot_s_active_bqh // TC.p_max
    spp.slot_n_bsq_tiles = spp.slot_n_bsq_full_tiles + (spp.slot_s_active_bqh_remainder > 0)
    spp.slot_s_active_bqh_tile = TC.p_max if spp.slot_n_bsq_tiles > 1 else spp.slot_s_active_bqh
    spp.slot_batch_interleave_degree = (
        cfg.seq_packed_slot_interleave_degree
        if cfg.seq_packed_slot_interleave_degree > 0
        else min(spp.num_slots, TC.psum_b_max)
    )
    return spp


def _set_atp_batch_dims(atp: AttnTileParams, batch_width: int, TC: TileConstants):
    """Set the active batch-shaped aliases used by the existing compute helpers."""
    atp.bs = batch_width
    atp.s_active_bqh = batch_width * atp.s_active_qh
    atp.s_active_bqh_remainder = atp.s_active_bqh % TC.p_max
    atp.n_bsq_full_tiles = atp.s_active_bqh // TC.p_max
    atp.n_bsq_tiles = atp.n_bsq_full_tiles + (atp.s_active_bqh_remainder > 0)
    atp.s_active_bqh_tile = TC.p_max if atp.n_bsq_tiles > 1 else atp.s_active_bqh
    atp.batch_interleave_degree = min(batch_width, TC.psum_b_max)


"""
Initialization functions
"""


def _compute_tile_params(
    cfg: AttnTKGConfig,
    TC: TileConstants,
    q,
    k_prior,
    v_prior,
    k_active,
    v_active,
    active_blocks_table,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> AttnTileParams:
    """Compute tiling and dimension parameters from configuration."""
    atp = AttnTileParams()

    atp.is_block_kv = True
    atp.block_len = 0  # Default for flat KV cache; overwritten in _setup_block_kv_cache for block KV
    atp.qk_row_tile_factor = 1  # Default; overridden in _setup_block_kv_cache for block KV

    # Determine FP8 KV status
    k_prior_fp8 = is_fp8_e4m3(k_prior.dtype)
    v_prior_fp8 = is_fp8_e4m3(v_prior.dtype)
    k_active_fp8 = is_fp8_e4m3(k_active.dtype)
    v_active_fp8 = is_fp8_e4m3(v_active.dtype)
    any_fp8 = k_prior_fp8 or v_prior_fp8 or k_active_fp8 or v_active_fp8
    all_fp8 = k_prior_fp8 and v_prior_fp8 and k_active_fp8 and v_active_fp8
    atp.is_fp8_kv = all_fp8

    kernel_assert(
        not cfg.fp8_packed or all_fp8,
        f"fp8_packed requires all KV tensors to be FP8. Got k_prior_fp8={k_prior_fp8}, "
        f"v_prior_fp8={v_prior_fp8}, k_active_fp8={k_active_fp8}, v_active_fp8={v_active_fp8}.",
    )

    # Resolve FP8 E4M3 dtype for SBUF K/V tile allocations: use caller's
    # concrete k_prior.dtype when explicit; resolve from dtype_mode only when
    # the caller passed the opaque "float8e4" sentinel.
    if all_fp8:
        if str(k_prior.dtype) == "float8e4":
            atp.kv_e4m3_tile_dtype = resolve_fp8_e4m3_dtype(dtype_mode)
        else:
            atp.kv_e4m3_tile_dtype = k_prior.dtype
    else:
        atp.kv_e4m3_tile_dtype = None

    # ========== Input validation (kernel asserts) ==========
    # Basic shape constraints
    kernel_assert(
        0 < cfg.bs,
        f"Batch size must be strictly positive, got bs={cfg.bs}.",
    )
    kernel_assert(
        0 < cfg.q_head,
        f"Number of Q heads must be strictly positive, got q_head={cfg.q_head}.",
    )
    kernel_assert(
        0 < cfg.s_active,
        f"Number of decode tokens must be strictly positive, got s_active={cfg.s_active}.",
    )
    kernel_assert(
        0 < cfg.curr_sprior <= cfg.full_sprior,
        f"curr_sprior must be <= full_sprior. Got curr_sprior={cfg.curr_sprior}, full_sprior={cfg.full_sprior}.",
    )
    kernel_assert(
        0 < cfg.d_head <= _MAX_D_HEAD,
        f"Unsupported d_head. Got d_head={cfg.d_head}, must be between 1 and {_MAX_D_HEAD}, inclusive.",
    )

    # FP8 dtype validation
    kernel_assert(
        not any_fp8 or all_fp8,
        f"FP8 KV cache requires all KV tensors to have the same FP8 E4M3 dtype "
        f"(nl.float8_e4m3 or nl.float8_e4m3fn). "
        f"Got k_prior.dtype={k_prior.dtype}, v_prior.dtype={v_prior.dtype}, "
        f"k_active.dtype={k_active.dtype}, v_active.dtype={v_active.dtype}.",
    )
    kernel_assert(
        not is_fp8_e5m2(k_prior.dtype)
        and not is_fp8_e5m2(v_prior.dtype)
        and not is_fp8_e5m2(k_active.dtype)
        and not is_fp8_e5m2(v_active.dtype),
        f"nl.float8_e5m2 is not supported for KV tensors. "
        f"Got k_prior.dtype={k_prior.dtype}, v_prior.dtype={v_prior.dtype}, "
        f"k_active.dtype={k_active.dtype}, v_active.dtype={v_active.dtype}.",
    )

    # FP8 KV configuration constraints
    kernel_assert(
        not atp.is_fp8_kv or not cfg.fuse_rope,
        f"fuse_rope must be False when using FP8 KV cache. Got fuse_rope={cfg.fuse_rope}.",
    )
    kernel_assert(
        not atp.is_fp8_kv or q.dtype != nl.float32,
        f"float32 query dtype is not supported with FP8 KV cache. Got q.dtype={q.dtype}. Use nl.bfloat16 instead.",
    )
    kernel_assert(
        not atp.is_fp8_kv or cfg.qk_in_sb,
        f"qk_in_sb must be True when using FP8 KV cache. Got qk_in_sb={cfg.qk_in_sb}.",
    )

    # assign to object (can't directly because of NKI limitation)
    sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id = _get_lnc_sharding(cfg)
    atp.sprior_n_prgs = sprior_n_prgs
    atp.sprior_prg_id = sprior_prg_id
    atp.bs_n_prgs = bs_n_prgs
    atp.bs_prg_id = bs_prg_id
    atp.n_prgs = atp.sprior_n_prgs * atp.bs_n_prgs

    # Get shapes and dtypes
    atp.k_prior_load_type = nl.bfloat16 if atp.is_fp8_kv else k_prior.dtype
    atp.use_dma_transpose = (sizeinbytes(atp.k_prior_load_type) == 2) and (
        not atp.is_fp8_kv or cfg.fp8_packed
    )  # use_dma_transpose may be further disabled in _setup_block_kv_cache for small block_len configs
    atp.io_type = q.dtype
    atp.inter_type = nl.float32
    atp.bs_full = cfg.bs
    atp.bs_per_nc = atp.bs_full // atp.bs_n_prgs  # full per-NC batch size before batch tiling
    atp.s_prior = cfg.curr_sprior // atp.sprior_n_prgs  # shard prior seqlen onto each prg
    atp.s_active_qh = cfg.s_active * cfg.q_head  # flattened dim of [q_heads, s_active]
    atp.n_sprior_tile = div_ceil(atp.s_prior, TC.p_max)  # total number of p_max-tiles across full s_prior

    # ========== Derived parameter validation ==========
    kernel_assert(
        atp.s_prior % TC.p_max == 0,
        f"Sharded s_prior must be divisible by p_max. Got sharded s_prior={atp.s_prior}, p_max={TC.p_max}.",
    )

    kernel_assert(
        not cfg.fuse_rope or atp.bs_n_prgs == 1,
        f"Fuse rope requires batch to not be sharded. See `is_batch_sharded`.",
    )
    kernel_assert(
        not cfg.fuse_rope or cfg.bs * cfg.q_head * cfg.s_active <= TC.p_max,
        f"Fuse rope requires batch * q_head * s_active to be fit on the partition dimension, got {cfg.bs * atp.s_active_qh}.",
    )

    # Flash attention parameters
    # Enable FA when s_prior exceeds the tile size threshold
    use_fa, fa_tile_size = uses_flash_attention(cfg.enable_fa_s_prior_tiling, atp.s_prior)
    atp.use_fa = use_fa
    if atp.use_fa:
        atp.fa_tile_s_prior = fa_tile_size
        atp.fa_n_sprior_tile = div_ceil(fa_tile_size, TC.p_max)
        # Last FA tile must be able to hold s_active (k_active is loaded at tile end)
        last_tile_s_prior = atp.s_prior % fa_tile_size if atp.s_prior % fa_tile_size != 0 else fa_tile_size
        kernel_assert(
            last_tile_s_prior >= cfg.s_active,
            f"Last FA tile size ({last_tile_s_prior}) must be >= s_active ({cfg.s_active})",
        )
    else:
        atp.fa_tile_s_prior = atp.s_prior
        atp.fa_n_sprior_tile = atp.n_sprior_tile

    # Whether softmax sync is complete within the _cascaded_* path for this tile.
    #   - single-NC: True on the (only) tile — there is nothing cross-NC to do.
    #   - sharded (FA or non-FA): False — cross-NC sync happens in _finalize_and_store via the
    #     running buffers.
    atp.sync_softmax_per_fa_tile = atp.sprior_n_prgs == 1

    return atp


def _setup_block_kv_cache(
    k_prior,
    v_prior,
    k_active,
    v_active,
    active_blocks_table,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """
    Setup block KV cache by validating shapes and reshaping tensors for block-sparse access.

    Validates that k_prior and v_prior have correct block_len and d_head dimensions, then reshapes
    v_active and cache tensors to enable efficient block-sparse memory access. When blocks per batch
    is less than 128, resizes block_len to make blocks per batch a multiple of 128 for optimal performance.
    Loads and reshapes the active blocks table to track which cache blocks are active per batch.
    """
    # Active blocks table is int32 with INACTIVE_BLOCK_IDX (-1) for invalid/padding blocks.
    # When use_dma_transpose is enabled, we convert to uint32
    # When use_dma_transpose is disabled, int32 indices with -1 enable oob_mode.skip in dma_copy.
    if active_blocks_table.dtype != nl.int32:
        print(
            f"WARNING: active_blocks_table dtype is {active_blocks_table.dtype}, not int32. "
            f"Use int32 with -1 for OOB indices to take advantage of DMA skipping."
        )
    # Check shapes
    kernel_assert(not cfg.strided_mm1, f"Block KV requires MM1 to not be strided.")
    kernel_assert(cfg.tp_k_prior, f"Block KV requires k_prior to not be transposed.")
    if cfg.fp8_packed:
        kernel_assert(
            cfg.block_len % 2 == 0,
            f"fp8_packed requires block_len to be even, got {cfg.block_len}.",
        )
        kernel_assert(
            k_prior.shape == (v_prior.shape[0], cfg.block_len // 2, cfg.d_head, 2),
            f"Block KV fp8_packed requires k_prior shape (*, {cfg.block_len // 2}, {cfg.d_head}, 2), "
            f"got {k_prior.shape}.",
        )
    else:
        kernel_assert(
            k_prior.shape[1] == cfg.block_len,
            f"Block KV requires k_prior input must be reshaped to have block_len as the second dimension, expected k_prior.shape[1]={cfg.block_len}, got {k_prior.shape[1]}.",
        )
        kernel_assert(
            k_prior.shape[2] == cfg.d_head,
            f"Block KV requires k_prior input must be reshaped to have d_head as the third dimension, expected k_prior.shape[2]={cfg.d_head}, got {k_prior.shape[2]}.",
        )
    kernel_assert(
        v_prior.shape[1:] == (cfg.block_len, cfg.d_head),
        f"Block KV requires v_prior shape (*, {cfg.block_len}, {cfg.d_head}), got v_prior.shape={v_prior.shape}.",
    )
    if cfg.fp8_packed:
        kernel_assert(
            k_prior.shape[0] == v_prior.shape[0],
            f"Block KV requires k_prior and v_prior to have the same number of blocks, "
            f"got k_prior.shape[0]={k_prior.shape[0]}, v_prior.shape[0]={v_prior.shape[0]}.",
        )
    else:
        kernel_assert(
            k_prior.shape == v_prior.shape,
            f"Block KV requires k_prior and v_prior shapes to match, got {k_prior.shape=}, {v_prior.shape=}",
        )
    kernel_assert(
        cfg.qk_in_sb,
        "Block KV loading from k_active is currently only supported when qk is in SBUF (qk_in_sb==True)",
    )
    kernel_assert(
        k_active.shape == (cfg.d_head, cfg.bs * cfg.s_active),
        f"Block KV requires k_active has the shape (d_head, bs * s_active), expected {(cfg.d_head, cfg.bs * cfg.s_active)}, got {k_active.shape}.",
    )  # This is equivalent to qk_in_sb, but just in case
    kernel_assert(
        active_blocks_table.ndim == 2,
        f"Packed block KV requires a 2D active_blocks_table, got shape {active_blocks_table.shape}.",
    )
    num_blocks_per_batch = cfg.curr_sprior // cfg.block_len

    # Reshape before performing modifications on the dimensions
    bufs.v_active_reshaped = v_active.reshape((cfg.bs, cfg.s_active * cfg.d_head))
    # For block cache support, the kernel requires the number of blocks per batch to be a multiple of 128.
    # When S_ctx is small and blocks per batch < 128, we will "resize" blocks to make blocks per batch a multiple of 128.
    n_prgs = nl.num_programs(0)
    block_len, blk_cache_resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_per_batch=num_blocks_per_batch,
        block_len=cfg.block_len,
        lnc=n_prgs,
        p_max=TC.p_max,
        bs=cfg.bs,
        q_head=cfg.q_head,
        s_active=cfg.s_active,
        full_sprior=cfg.full_sprior,
        enable_fa_s_prior_tiling=cfg.enable_fa_s_prior_tiling,
        fuse_rope=cfg.fuse_rope,
    )

    # fp8_packed: resized block_len must be >= 2 to keep packed pairs intact
    kernel_assert(
        not cfg.fp8_packed or block_len >= 2,
        f"fp8_packed requires resized block_len >= 2, got {block_len}. "
        f"Increase S_ctx so that blocks_per_batch >= 128 without shrinking block_len below 2.",
    )

    # Disable dma_transpose when block_len is very small, or block_len, d_head, and batches per core are all small on LNC2.
    # For these configs the DMA transpose overhead outweighs the benefits because there are too few batches to pipeline DMA bursts
    # and small block_len leads to too many folds per batch.
    # fp8_packed always uses DMA transpose (no PE transpose fallback for packed layout).
    if not cfg.fp8_packed:
        if (block_len < 8) or (block_len <= 16 and cfg.d_head <= 16 and atp.bs < 4 and atp.n_prgs == 2):
            atp.use_dma_transpose = False

    # assign to atp (can't do directly because function return value cannot be assigned to object)
    atp.block_len = block_len
    atp.blk_cache_resize_factor = blk_cache_resize_factor

    _min_block_len = 4 if cfg.fp8_packed else 2
    # Manual PSUM allocation does not currently reserve separate banks for the row tiles.
    atp.qk_row_tile_factor = (
        2
        if (sbm.is_auto_alloc() and cfg.d_head == 64 and atp.use_dma_transpose and atp.block_len >= _min_block_len)
        else 1
    )
    k_new_cache_shape = (k_prior.shape[0] * blk_cache_resize_factor, atp.block_len * cfg.d_head)
    bufs.k_prior_reshaped = k_prior.reshape(k_new_cache_shape)

    v_new_cache_shape = (
        v_prior.shape[0] * blk_cache_resize_factor,
        atp.block_len * cfg.d_head,
    )
    bufs.v_prior_reshaped = v_prior.reshape(v_new_cache_shape)

    # if using flash attention verify the fa tile size is divisible by atp.block_len * TC.p_max
    # since that is assumed during KV load. Last tile can be smaller. Note that the current resize
    # logic doesn't account for flash attention so it is possible below assertion breaks when the
    # flash attention tile size is too small.
    if atp.use_fa:
        kernel_assert(
            atp.fa_tile_s_prior % (atp.block_len * TC.p_max) == 0,
            f"Block KV requires the Flash attention tile size to be divisible by product of resized block len and max partitions, got {atp.fa_tile_s_prior=}, {atp.block_len=}, {TC.p_max=}",
        )
        # check last tile is also divisible
        if atp.s_prior % atp.fa_tile_s_prior != 0:
            last_tile_s_prior = atp.s_prior % atp.fa_tile_s_prior
            kernel_assert(
                last_tile_s_prior % (atp.block_len * TC.p_max) == 0,
                f"Block KV requires the Flash attention tile size to be divisible by product of resized block len and max partitions, got {last_tile_s_prior=}, {atp.block_len=}, {TC.p_max=}",
            )


def _setup_debug_tensors(DBG_TENSORS, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers):
    """Setup debug tensor references."""
    kernel_assert(
        len(DBG_TENSORS) == 4 + (1 if atp.is_block_kv else 0),
        f"Received {len(DBG_TENSORS)} debug tensors, when 4 are expected (or 5 if block KV is used)",
    )
    # Intermediate values for debugging.
    bufs.DBG_QK = DBG_TENSORS[0].reshape(
        (
            TC.p_max,
            atp.sprior_n_prgs,
            atp.n_sprior_tile,
            atp.bs_n_prgs,
            atp.s_active_bqh,
        )
    )
    bufs.DBG_QK_MAX = DBG_TENSORS[1].reshape((atp.bs_n_prgs, atp.n_bsq_tiles, atp.s_active_bqh_tile))
    bufs.DBG_QK_EXP = DBG_TENSORS[2].reshape(
        (
            TC.p_max,
            atp.sprior_n_prgs,
            atp.n_sprior_tile,
            atp.bs_n_prgs,
            atp.s_active_bqh,
        )
    )
    bufs.DBG_EXP_SUM = DBG_TENSORS[3].reshape((atp.bs_n_prgs, atp.n_bsq_tiles, atp.s_active_bqh_tile))
    if atp.is_block_kv:
        bufs.DBG_ACTIVE_TABLE = DBG_TENSORS[4]
        # DBG_ACTIVE_TABLE shape validation — compute full num_folds_per_batch from atp fields
        full_num_folds_per_batch = atp.s_prior // (atp.block_len * TC.p_max)
        kernel_assert(
            bufs.DBG_ACTIVE_TABLE.shape[1] == full_num_folds_per_batch * atp.sprior_n_prgs,
            "Active table debug tensor second dimension incorrect (needs to have shape (P_MAX, curr_sprior // block_len, batch_size)), "
            f"expected DBG_ACTIVE_TABLE.shape[1]={full_num_folds_per_batch * atp.sprior_n_prgs}, got {bufs.DBG_ACTIVE_TABLE.shape[1]}",
        )
        kernel_assert(
            bufs.DBG_ACTIVE_TABLE.shape[2] == atp.bs_full,
            "Active table debug tensor third dimension incorrect (needs to have shape (P_MAX, curr_sprior // block_len, batch_size))"
            f"expected DBG_ACTIVE_TABLE.shape[2]={atp.bs_full}, got {bufs.DBG_ACTIVE_TABLE.shape[2]}",
        )
        # Note: DBG_ACTIVE_TABLE store is done incrementally inside _load_and_reshape_active_blk_table


def _store_dbg_qk_max(
    src: nl.NkiTensor,
    max_is_negated: bool,
    name_suffix: str,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Transpose a max tensor with shape [s_active_bqh_tile, n_bsq_tiles] into a
    [n_bsq_tiles, s_active_bqh_tile] block in DBG_QK_MAX, un-negating if needed.

    Only writes the current NC's slice. Also pads the remainder columns with zeros when
    s_active_bqh is not a multiple of p_max. Caller must ensure atp.bs == atp.bs_per_nc
    (i.e. no batch tiling) because the offset-based write assumes full-batch layout.

    Args:
      src: Source tensor with shape [s_active_bqh_tile, n_bsq_tiles].
      max_is_negated: Whether `src` holds negated max values (requires multiply by -1 on dump).
      name_suffix: Unique suffix for the DMA ops' names.
    """
    sbm.open_scope()
    qk_max_dbg_psum = nl.ndarray(
        (atp.n_bsq_tiles, atp.s_active_bqh_tile),
        dtype=src.dtype,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, 0),
    )
    qk_max_dbg = sbm.alloc_stack((atp.n_bsq_tiles, atp.s_active_bqh_tile), dtype=src.dtype)
    nisa.nc_transpose(qk_max_dbg_psum, src[: atp.s_active_bqh_tile, : atp.n_bsq_tiles])
    if max_is_negated:
        nisa.tensor_copy(qk_max_dbg, qk_max_dbg_psum)
    else:
        # Multiply by -1 so DBG_QK_MAX always stores negated values (consistent with the
        # legacy per-tile classical path, which stored from the negated qk_max_buf).
        nisa.tensor_scalar(qk_max_dbg, qk_max_dbg_psum, op0=nl.multiply, operand0=-1)

    dbg_qk_max_view = (bufs.DBG_QK_MAX).select(0, atp.bs_prg_id)
    nisa.dma_copy(
        dbg_qk_max_view,
        qk_max_dbg,
        name=f"{sbm.get_name_prefix()}dbg_qk_max_store_{name_suffix}",
    )

    # Pad remainder-tile region with zeros (the last BSQ tile is smaller when s_active_bqh
    # is not a multiple of p_max; the remainder columns need defined values).
    if atp.n_bsq_full_tiles > 0 and atp.s_active_bqh_remainder > 0:
        zeros = sbm.alloc_stack((1, atp.s_active_bqh_tile - atp.s_active_bqh_remainder), dtype=src.dtype)
        nisa.memset(zeros, 0)
        nisa.dma_copy(
            dbg_qk_max_view.select(0, atp.n_bsq_full_tiles)
            .expand_dim(0)
            .slice(1, atp.s_active_bqh_remainder, atp.s_active_bqh_tile),
            zeros,
            name=f"{sbm.get_name_prefix()}dbg_qk_max_store_zeros_{name_suffix}",
        )
    sbm.close_scope()


def _store_dbg_exp_sum(
    src: nl.NkiTensor,
    name_suffix: str,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Transpose a sum tensor with shape [s_active_bqh_tile, n_bsq_tiles] into a
    [n_bsq_tiles, s_active_bqh_tile] block in DBG_EXP_SUM. See _store_dbg_qk_max for the
    full-batch / batch-tiling contract."""
    sbm.open_scope()
    exp_sum_dbg_psum = nl.ndarray(
        (atp.n_bsq_tiles, atp.s_active_bqh_tile),
        dtype=src.dtype,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, 0),
    )
    exp_sum_dbg = sbm.alloc_stack((atp.n_bsq_tiles, atp.s_active_bqh_tile), dtype=src.dtype)
    nisa.nc_transpose(exp_sum_dbg_psum, src[: atp.s_active_bqh_tile, : atp.n_bsq_tiles])
    nisa.tensor_copy(exp_sum_dbg, exp_sum_dbg_psum)

    dbg_exp_sum_view = (bufs.DBG_EXP_SUM).select(0, atp.bs_prg_id)
    nisa.dma_copy(
        dst=dbg_exp_sum_view,
        src=exp_sum_dbg,
        name=f"{sbm.get_name_prefix()}dbg_exp_sum_store_{name_suffix}",
    )

    if atp.n_bsq_full_tiles > 0 and atp.s_active_bqh_remainder > 0:
        zeros = sbm.alloc_stack((1, atp.s_active_bqh_tile - atp.s_active_bqh_remainder), dtype=src.dtype)
        nisa.memset(zeros, 0)
        nisa.dma_copy(
            dbg_exp_sum_view.select(0, atp.n_bsq_full_tiles)
            .expand_dim(0)
            .slice(1, atp.s_active_bqh_remainder, atp.s_active_bqh_tile),
            zeros,
            name=f"{sbm.get_name_prefix()}dbg_exp_sum_store_zeros_{name_suffix}",
        )
    sbm.close_scope()


def _store_dbg_qk_max_zeros_full_batch(atp: AttnTileParams, sbm: SbufManager, bufs: AttnInternalBuffers):
    """Fallback zero-fill for DBG_QK_MAX when batch tiling is active (full-batch offset writes
    aren't reliable). Writes once on the first batch tile using the debug tensor's full shape."""
    sbm.open_scope()
    dbg_qk_max_view = (bufs.DBG_QK_MAX).select(0, atp.bs_prg_id)
    dbg_zero = sbm.alloc_stack((dbg_qk_max_view.shape[0], 1), dtype=bufs.DBG_QK_MAX.dtype, buffer=nl.sbuf)
    nisa.memset(dbg_zero, 0.0)
    nisa.dma_copy(
        dbg_qk_max_view,
        (dbg_zero).broadcast(1, dbg_qk_max_view.shape[1]),
        name=f"{sbm.get_name_prefix()}dbg_qk_max_store_zeros_batch_tiling",
    )
    sbm.close_scope()


def _store_dbg_exp_sum_zeros_full_batch(atp: AttnTileParams, sbm: SbufManager, bufs: AttnInternalBuffers):
    """Fallback zero-fill for DBG_EXP_SUM when batch tiling is active."""
    sbm.open_scope()
    dbg_exp_sum_view = (bufs.DBG_EXP_SUM).select(0, atp.bs_prg_id)
    dbg_zero = sbm.alloc_stack((dbg_exp_sum_view.shape[0], 1), dtype=bufs.DBG_EXP_SUM.dtype, buffer=nl.sbuf)
    nisa.memset(dbg_zero, 0.0)
    nisa.dma_copy(
        dbg_exp_sum_view,
        (dbg_zero).broadcast(1, dbg_exp_sum_view.shape[1]),
        name=f"{sbm.get_name_prefix()}dbg_exp_sum_store_zeros_batch_tiling",
    )
    sbm.close_scope()


def _allocate_qk_buffers(
    atp: AttnTileParams,
    spp: SequencePackingParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Allocate core QK buffers for the current FA tile.

    Create KQ^T result mloc for all batches (filled with -INF for masking)
    The tensor has shape [p_max, tile_n_sprior * bs * s_active_qh], where on the free dimension, there are
      tile_n_sprior tiles, each tile contains bs number of subtiles, and each subtile is s_active_qh in length.
      I.e., the s_active_qh tiles are interleaved by batch on the free dimension.
    The cascaded max reduce later on will do a strided access on the free dimension.

    Uses fa_ctx.tile_n_sprior which is the actual tile size (may be smaller for last FA tile).
    """
    compute_s_active_bqh = spp.slot_s_active_bqh

    bufs.qk = sbm.alloc_stack(
        (TC.p_max, fa_ctx.tile_n_sprior * compute_s_active_bqh),
        dtype=atp.inter_type,
    )
    if atp.qk_row_tile_factor > 1:
        if nisa.get_nc_version() >= nisa.nc_version.gen4:
            # Trn3+ (gen4+): tile the memset with a fixed tile size to allow freedom for
            # scheduling. 512 is a rough estimate and may need tuning for different configs.
            qk_memset_tile_size = 512
            qk_free = bufs.qk.shape[1]
            qk_memset_tiles = div_ceil(qk_free, qk_memset_tile_size)
            for tile_idx in range(qk_memset_tiles):
                start = tile_idx * qk_memset_tile_size
                size = min(qk_memset_tile_size, qk_free - start)
                nisa.memset(bufs.qk[:, nl.ds(start, size)], -np.inf)
        else:
            nisa.memset(bufs.qk, -np.inf)
    bufs.qk_io_type = sbm.alloc_stack(bufs.qk.shape, dtype=atp.io_type)  # for matmults

    # Allocate mask buffer with same shape as qk
    bufs.mask_sb = sbm.alloc_stack(bufs.qk.shape, dtype=nl.uint8, buffer=nl.sbuf)


def _allocate_online_softmax_buffers(
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    use_grouped_accumulator: bool,
):
    """Allocate identity-initialized persistent statistics for packed rows.

    The grouped path adds one dummy sequence record after the real per-NC
    batch. Unused groups gather from and scatter to that record, which keeps
    routing fixed-width without affecting a real sequence.
    """
    bufs.running_max = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    nisa.memset(bufs.running_max, value=-np.inf)

    bufs.running_sum = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    nisa.memset(bufs.running_sum, value=0)

    running_state_width = atp.s_active_bqh + (atp.s_active_qh if use_grouped_accumulator else 0)
    bufs.running_state_flat = sbm.alloc_stack(
        (cfg.d_head, 3 * running_state_width),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    running_state_storage = bufs.running_state_flat.reshape((cfg.d_head, 3, running_state_width))
    bufs.running_state = running_state_storage[:, :, : atp.s_active_bqh]
    if use_grouped_accumulator:
        nisa.memset(running_state_storage[:, 1:3, :], value=0)
        nisa.memset(running_state_storage[:, 0, :], value=-np.inf)
    else:
        nisa.memset(bufs.running_state, value=0)
    bufs.accumulator_max = bufs.running_state[:, 0, :]
    bufs.accumulator_sum = bufs.running_state[:, 1, :]
    bufs.running_output = bufs.running_state[:, 2, :]
    if not use_grouped_accumulator:
        nisa.memset(bufs.accumulator_max, value=-np.inf)

    bufs.correction_factor = sbm.alloc_stack(
        (atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf
    )
    if not use_grouped_accumulator:
        nisa.memset(bufs.correction_factor, value=1.0)


def _update_correction_factor(
    atp: AttnTileParams, correction_factor: nl.NkiTensor, curr_running_max: nl.NkiTensor, prev_running_max: nl.NkiTensor
):
    """Updates correction_factor to exp(prev_running_max - curr_running_max)"""
    for i_bsq_tile in range(atp.n_bsq_tiles):
        nisa.activation(
            correction_factor[:, i_bsq_tile],
            nl.exp,
            (prev_running_max if atp.max_negated else curr_running_max)[:, i_bsq_tile],
            bias=(curr_running_max if atp.max_negated else prev_running_max)[:, i_bsq_tile],
            scale=-1.0,
        )


def _gather_and_compute_global_running_max_and_sum(
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    local_running_max: nl.NkiTensor,
    sink_values: Optional[nl.NkiTensor] = None,
):
    """Gather remote running max AND running sum in a single sendrecv to produce the global max
    and global running sum (sink folded in if present).

    Packs [local_max | local_sum] into one buffer so both stats cross in one rendezvous instead
    of two, removing one PSEUDO_CORE_BARRIER pair from the finalize path. Math matches the
    sequential two-sendrecv version: global_sum = exp(local_max - global_max) * local_sum +
    exp(remote_max - global_max) * remote_sum [+ exp(sink - global_max)].

    Called from _finalize_and_store (sharded path). On exit running_max holds the global max,
    running_sum holds the global sum, and correction_factor = exp(local_max - global_max) for
    rescaling running_output.
    """
    kernel_assert(not atp.max_negated, "Unexpected atp.max_negated=True when computing cross-NC max/sum")

    if sink_values is not None:
        # Fold sink into the local running max so the exchanged max already accounts for it.
        nisa.tensor_tensor(
            dst=bufs.running_max,
            data1=bufs.running_max,
            data2=sink_values,
            op=nl.maximum,
        )

    sbm.open_scope()
    # Pack into one buffer so a single sendrecv exchanges both:
    # max = [:, :n_bsq_tiles], sum = [:, n_bsq_tiles:2*n_bsq_tiles].
    local_pack = sbm.alloc_stack((atp.s_active_bqh_tile, 2 * atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    remote_pack = sbm.alloc_stack((atp.s_active_bqh_tile, 2 * atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    nisa.tensor_copy(local_pack[:, 0 : atp.n_bsq_tiles], bufs.running_max)
    nisa.tensor_copy(local_pack[:, atp.n_bsq_tiles : 2 * atp.n_bsq_tiles], bufs.running_sum)

    # GPSIMD SB-to-SB swap (no CoreBarrier) when enabled and the packed stats fit its
    # 1024-byte/partition limit; else fall back to the DMA path.
    use_gpsimd = cfg.use_gpsimd_sb2sb and (2 * atp.n_bsq_tiles * 4 <= 1024)
    nisa.sendrecv(
        src=local_pack,
        dst=remote_pack,
        send_to_rank=(1 - atp.sprior_prg_id),
        recv_from_rank=(1 - atp.sprior_prg_id),
        pipe_id=0,
        dma_engine=dma_engine.gpsimd_dma if use_gpsimd else dma_engine.dma,
    )

    remote_max = remote_pack[:, 0 : atp.n_bsq_tiles]
    remote_sum = remote_pack[:, atp.n_bsq_tiles : 2 * atp.n_bsq_tiles]

    # global_max = max(local_max[+sink], remote_max)
    nisa.tensor_tensor(
        dst=bufs.running_max,
        data1=bufs.running_max,
        data2=remote_max,
        op=nl.maximum,
    )

    # c_local = exp(local_max - global_max) -> bufs.correction_factor (rescales sum & output)
    _update_correction_factor(atp, bufs.correction_factor, bufs.running_max, local_running_max)

    # c_remote = exp(remote_max - global_max), staged in remote_max in place.
    for i_bsq_tile in range(atp.n_bsq_tiles):
        nisa.activation(
            dst=remote_max[:, i_bsq_tile],
            op=nl.exp,
            data=bufs.running_max[: atp.s_active_bqh_tile, i_bsq_tile],
            bias=remote_max[:, i_bsq_tile],
            scale=-1.0,
        )

    # global_sum = c_local * local_sum + c_remote * remote_sum
    #   step 1: local_sum *= c_local
    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=bufs.correction_factor,
        op=nl.multiply,
    )
    #   step 2: remote_sum *= c_remote (both reside in remote_pack now)
    nisa.tensor_tensor(
        dst=remote_sum,
        data1=remote_sum,
        data2=remote_max,
        op=nl.multiply,
    )
    #   step 3: running_sum += remote_sum
    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=remote_sum,
        op=nl.add,
    )
    sbm.close_scope()

    if sink_values is not None:
        # sink_exp = exp(sink_raw - global_max) in-place, then running_sum += sink_exp.
        for i_bsq_tile in range(atp.n_bsq_tiles):
            nisa.activation(
                dst=sink_values[:, i_bsq_tile],
                op=nl.exp,
                data=bufs.running_max[: atp.s_active_bqh_tile, i_bsq_tile],
                bias=sink_values[:, i_bsq_tile],
                scale=-1.0,
            )
        nisa.tensor_tensor(
            dst=bufs.running_sum,
            data1=bufs.running_sum,
            data2=sink_values,
            op=nl.add,
        )


def _fold_active_kv_into_packed_running_stats(
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
):
    """Merge each sequence's active K/V once, after all schedule rows.

    The online-softmax merge is associative over keys, so a group of one key is
    as valid as a tile of 8192. Rather than splicing active K/V into whichever
    tile happens to be terminal for a sequence - which costs a predicated
    read-modify-write per slot plus per-slot metadata naming the terminal slot -
    each active key is folded here as its own one-key partial, exactly as the
    attention sink already is.

    Scores are produced in [1, s_active_bqh] row layout with active K stationary,
    then transposed into the running statistics' column layout with the same
    helper the accumulator materialization uses, so any number of partition tiles
    is supported. Causality among active tokens is applied per active key by
    masking the query lanes that precede it.
    """
    kernel_assert(
        not atp.max_negated,
        "Packed active-KV finalization requires non-negated running maxima.",
    )

    sbm.open_scope()

    q_rows = (bufs.q_sb).reshape_dim(1, [atp.bs_full, atp.s_active_qh])
    k_active_rows = (bufs.k_active_sb).reshape_dim(1, [cfg.bs, cfg.s_active])
    v_active_flat = (bufs.v_active_reshaped).reshape((cfg.bs * cfg.s_active * cfg.d_head,))

    scores_row = sbm.alloc_stack((1, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
    scores_col = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)
    previous_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=atp.inter_type, buffer=nl.sbuf)
    correction_factor_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
    weight_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
    v_active_sb = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)

    for i_s_k in range(cfg.s_active):
        # Active K stationary puts query lanes on the free axis, one row per key.
        scores_psum = nl.ndarray((1, atp.s_active_bqh), dtype=nl.float32, buffer=nl.psum)
        for i_b in range(atp.bs_per_nc):
            global_b = btc.global_batch_offset + i_b
            lane_start = i_b * atp.s_active_qh
            nisa.nc_matmul(
                scores_psum[0:1, lane_start : lane_start + atp.s_active_qh],
                stationary=(k_active_rows).select(1, global_b).slice(1, start=i_s_k, end=i_s_k + 1),
                moving=(q_rows).select(1, global_b),
            )
        nisa.tensor_copy(dst=scores_row, src=scores_psum)

        # Causality: a query at active position s_q cannot see active key s_k > s_q.
        # Lanes run b * s_active_qh + h * s_active + s_q, so for each sequence and
        # head drive the first i_s_k lanes to the finite minimum.
        if i_s_k > 0:
            for i_b in range(atp.bs_per_nc):
                for i_h in range(cfg.q_head):
                    lane = i_b * atp.s_active_qh + i_h * cfg.s_active
                    nisa.memset(dst=scores_row[:, lane : lane + i_s_k], value=_MIN_FLOAT32)

        _transpose_sbuf_row_to_columns(
            dst=scores_col,
            src=scores_row,
            num_elem=atp.s_active_bqh,
            partition_tile_size=atp.s_active_bqh_tile,
            n_tiles=atp.n_bsq_tiles,
        )

        # Standard one-key online-softmax merge.
        nisa.tensor_copy(previous_running_max, bufs.running_max)
        nisa.tensor_tensor(
            dst=bufs.running_max,
            data1=bufs.running_max,
            data2=scores_col,
            op=nl.maximum,
        )
        _update_correction_factor(atp, bufs.correction_factor, bufs.running_max, previous_running_max)
        nisa.tensor_tensor(
            dst=bufs.running_sum,
            data1=bufs.running_sum,
            data2=bufs.correction_factor,
            op=nl.multiply,
        )
        # scores_col <- exp(score - new_max): this key's unnormalized weight.
        for i_bsq_tile in range(atp.n_bsq_tiles):
            nisa.activation(
                dst=scores_col[:, i_bsq_tile],
                op=nl.exp,
                data=bufs.running_max[:, i_bsq_tile],
                bias=scores_col[:, i_bsq_tile],
                scale=-1.0,
            )
        nisa.tensor_tensor(dst=bufs.running_sum, data1=bufs.running_sum, data2=scores_col, op=nl.add)

        # Rescale accumulated output to the new max, then add weight * v_active.
        _s_active_bqh_tile_transpose_broadcast(bufs.correction_factor, correction_factor_bc, atp, TC)
        nisa.tensor_tensor(
            dst=bufs.running_output,
            data1=bufs.running_output,
            data2=correction_factor_bc,
            op=nl.multiply,
        )
        _s_active_bqh_tile_transpose_broadcast(scores_col, weight_bc, atp, TC)

        # One transposing broadcast load: dst[d, b, h, s] = v_active[b, i_s_k, d].
        nisa.dma_copy(
            dst=v_active_sb.reshape_dim(1, [atp.bs_per_nc, cfg.q_head, cfg.s_active]),
            src=v_active_flat.ap(
                [
                    [1, cfg.d_head],
                    [cfg.s_active * cfg.d_head, atp.bs_per_nc],
                    [0, cfg.q_head],
                    [0, cfg.s_active],
                ],
                offset=(btc.global_batch_offset * cfg.s_active + i_s_k) * cfg.d_head,
            ),
            name=f"{sbm.get_name_prefix()}packed_active_v_fold_load_k{i_s_k}",
        )
        nisa.tensor_tensor(dst=v_active_sb, data1=v_active_sb, data2=weight_bc, op=nl.multiply)
        nisa.tensor_tensor(
            dst=bufs.running_output,
            data1=bufs.running_output,
            data2=v_active_sb,
            op=nl.add,
        )

    sbm.close_scope()


def _fold_sink_into_packed_running_stats(
    sink,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
):
    """Fold one sink contribution into each persistent query lane."""
    kernel_assert(not atp.max_negated, "Packed sink finalization requires non-negated running maxima")

    sbm.open_scope()
    sink_values = sbm.alloc_stack(
        (atp.s_active_bqh_tile, atp.n_bsq_tiles),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    _prep_sink(
        sink,
        sink_values,
        atp,
        spp,
        cfg,
        TC,
        sbm,
        btc,
        use_slot_geometry=False,
    )

    previous_running_max = sbm.alloc_stack(
        bufs.running_max.shape,
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(previous_running_max, bufs.running_max)
    nisa.tensor_tensor(
        dst=bufs.running_max,
        data1=bufs.running_max,
        data2=sink_values,
        op=nl.maximum,
    )
    _update_correction_factor(
        atp,
        bufs.correction_factor,
        bufs.running_max,
        previous_running_max,
    )

    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=bufs.correction_factor,
        op=nl.multiply,
    )
    for i_bsq_tile in range(atp.n_bsq_tiles):
        nisa.activation(
            dst=sink_values[:, i_bsq_tile],
            op=nl.exp,
            data=bufs.running_max[:, i_bsq_tile],
            bias=sink_values[:, i_bsq_tile],
            scale=-1.0,
        )
    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=sink_values,
        op=nl.add,
    )

    correction_factor_bc = sbm.alloc_stack(
        (cfg.d_head, atp.s_active_bqh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    _s_active_bqh_tile_transpose_broadcast(
        bufs.correction_factor,
        correction_factor_bc,
        atp,
        TC,
    )
    nisa.tensor_tensor(
        dst=bufs.running_output,
        data1=bufs.running_output,
        data2=correction_factor_bc,
        op=nl.multiply,
    )
    sbm.close_scope()


def _finalize_and_store(
    sink,
    out: nl.NkiTensor,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
    DBG_TENSORS=None,
):
    """Finalize flash attention output: sync softmax across NCs (if sharded), normalize by running
    sum and store to HBM.

    After all FA tiles are processed, for each NeuronCore we have:
    - running_max: local max over QK across all FA tiles (no sink, no remote).
    - running_sum: sum over local tiles of exp(qk - local_max).
    - running_output: sum over local tiles of exp(qk - local_max) @ V.

    When s_prior is sharded across NCs, we run the cross-NC softmax sync here (after the last PV
    matmul) so sendrecv does not block GPSIMD V prefetches on the last FA tile. The sync:
        1. Save local_max to a scope-local buffer (used as prev for the correction factor).
        2. If sink is used: load sink into a scope-local buffer via _prep_sink.
        3. Call _gather_and_compute_global_running_max_and_sum(local_running_max=local_max,
           sink_values=...) which packs [local_max | local_sum] and exchanges both in a single
           sendrecv, then folds sink, produces the global running_sum, and updates
           correction_factor to exp(local - global).
        4. Rescale running_output by correction_factor (transpose_broadcast to [d_head,
           s_active_bqh]); running_sum is already at global scale from step 3.

    For packed batch-owned execution, sink is folded once into the persistent
    running state after every schedule row has accumulated. Then normalize
    running_output by reciprocal(running_sum) and store.

    running_output has shape [d_head, s_active_bqh] (flat).
    running_sum has shape [s_active_bqh_tile, n_bsq_tiles].
    """
    sbm.open_scope()

    if atp.sprior_n_prgs > 1:
        kernel_assert(not atp.max_negated, "Unexpected atp.max_negated=True when deferring softmax gather")

        sbm.open_scope()

        # Scope-local sink buffer for the cross-NC sync; sink is consumed here rather than in
        # _cascaded_max_reduce / _cascaded_sum_reduction.
        sink_values = None
        if sink is not None:
            sink_values = sbm.alloc_stack(
                (atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf
            )
            _prep_sink(sink, sink_values, atp, spp, cfg, TC, sbm, btc, use_slot_geometry=False)

        # 1. Save local_max (used as prev when computing c_local = exp(local_max - global_max)
        #    and for rescaling running_sum / running_output to the global scale).
        local_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=atp.inter_type, buffer=nl.sbuf)
        nisa.tensor_copy(local_running_max, bufs.running_max)

        # 2. Single cross-NC exchanging BOTH local max and local sum in one sendrecv,
        #    fold sink, produce the global running_sum and correction_factor = exp(local - global).
        _gather_and_compute_global_running_max_and_sum(atp, cfg, sbm, bufs, local_running_max, sink_values=sink_values)

        # 3. Rescale running_output by c_local (broadcast to [d_head, s_active_bqh]).
        correction_factor_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
        _s_active_bqh_tile_transpose_broadcast(bufs.correction_factor, correction_factor_bc, atp, TC)
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            correction_factor_bc,
            op=nl.multiply,
        )

        sbm.close_scope()

    else:
        # Packed LNC1/LNC2 execution owns complete sequences on each NC, so the
        # active token and the sink are both folded once into persistent state
        # after all schedule rows are accumulated.
        _fold_active_kv_into_packed_running_stats(atp, spp, cfg, TC, sbm, bufs, btc)
        if sink != None:
            _fold_sink_into_packed_running_stats(sink, atp, spp, cfg, TC, sbm, bufs, btc)

    # Debug tensor writes for the online-softmax path (sharded; FA single-NC). Dumping from
    # running_max / running_sum captures the final global values. Batch-tiling case
    # Slot-width debug tensors are defined by zero-fill in _cascaded_*.
    if DBG_TENSORS is not None and atp.bs == atp.bs_per_nc:
        _store_dbg_qk_max(bufs.running_max, atp.max_negated, "finalize", atp, TC, sbm, bufs)
        _store_dbg_exp_sum(bufs.running_sum, "finalize", atp, TC, sbm, bufs)

    # Compute reciprocal of running sum in-place
    nisa.reciprocal(
        bufs.running_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        bufs.running_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
    )

    # Transpose and broadcast sum_recip to [d_head, s_active_bqh] for final normalization

    sum_recip_bc = sbm.alloc_stack((cfg.d_head, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)
    _s_active_bqh_tile_transpose_broadcast(bufs.running_sum, sum_recip_bc, atp, TC)

    # Normalize: running_output *= sum_recip_bc
    nisa.tensor_tensor(
        bufs.running_output,
        bufs.running_output,
        sum_recip_bc,
        op=nl.multiply,
    )
    sbm.close_scope()
    exp_v_sendrecv_gpsimd = (
        cfg.use_gpsimd_sb2sb and atp.n_prgs > 1 and atp.bs * atp.s_active_qh <= 256 and cfg.d_head % 16 == 0
    )
    _gather_and_store_output(out, bufs.running_output, exp_v_sendrecv_gpsimd, atp, cfg, sbm, btc)


def _load_and_broadcast_pos_ids(pos_ids, atp, cfg, TC, sbm, name):
    """Load position IDs and broadcast onto all 128 partitions (for TensorScalarPtr).

    Shared helper for loading both rope_pos_ids and start_pos_ids, which follow
    the same pattern: reshape → alloc_stack → dma_copy → alloc_stack → stream_shuffle_broadcast.
    """
    pos_ids_sb = sbm.alloc_stack((TC.p_max, atp.bs_per_nc * cfg.s_active), dtype=pos_ids.dtype)
    pos_ids = pos_ids.reshape([atp.bs_n_prgs, atp.bs_per_nc * cfg.s_active])

    sbm.open_scope()
    pos_ids_loaded = sbm.alloc_stack((1, atp.bs_per_nc * cfg.s_active), dtype=pos_ids.dtype, align=4)
    nisa.dma_copy(pos_ids_loaded, pos_ids[atp.bs_prg_id, :], name=f"{sbm.get_name_prefix()}{name}")
    stream_shuffle_broadcast(src=pos_ids_loaded, dst=pos_ids_sb)
    sbm.close_scope()
    return pos_ids_sb


def _load_position_ids(
    rope_pos_ids,
    start_pos_ids,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Load position IDs and optional SWA start positions."""
    bufs.pos_ids_sb = None
    if rope_pos_ids is None:
        # only two components that use pos ids
        kernel_assert(
            not cfg.use_pos_id and not cfg.fuse_rope,
            "To generate mask or fuse rope, rope_pos_ids tensor must be provided",
        )
    else:
        bufs.pos_ids_sb = _load_and_broadcast_pos_ids(rope_pos_ids, atp, cfg, TC, sbm, "rope_pos_ids_load")

    bufs.start_pos_sb = None
    if start_pos_ids is not None:
        bufs.start_pos_sb = _load_and_broadcast_pos_ids(start_pos_ids, atp, cfg, TC, sbm, "start_pos_ids_load")


def _load_mask(
    mask,
    TC: TileConstants,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    sbm: SbufManager,
):
    """Load one pre-aligned packed mask row."""
    kernel_assert(
        mask.shape[0] > fa_ctx.schedule_row,
        "Packed mask is missing the current physical schedule row.",
    )
    kernel_assert(
        mask.shape[1] == TC.p_max and mask.shape[2] >= bufs.mask_sb.shape[1],
        "Packed mask rows must have shape [p_max, tile_n_sprior * slot_bqh].",
    )
    nisa.dma_copy(
        dst=bufs.mask_sb,
        src=mask[fa_ctx.schedule_row, 0 : TC.p_max, 0 : bufs.mask_sb.shape[1]],
        name=f"{sbm.get_name_prefix()}packed_mask_load_row{fa_ctx.schedule_row}",
    )


def _perform_rope(
    q,
    k_active,
    inv_freqs,
    k_out,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 0. Optional RoPE"""
    kernel_assert(
        cfg.curr_sprior <= _MAX_S_PRIOR_ACCURATE_ROPE,
        f"Rope requires modulo, which for s_prior={cfg.curr_sprior} > {_MAX_S_PRIOR_ACCURATE_ROPE} is innacurate due to float32 error build-up.",
    )

    # If we fuse rope, Q and K_active would need be processed by RoPE first then be stored in Q_sb.
    bufs.q_sb = sbm.alloc_stack(
        (cfg.d_head, atp.bs_n_prgs * atp.bs_per_nc * atp.s_active_qh),
        dtype=atp.io_type,
        buffer=nl.sbuf,
    )
    bufs.k_active_sb = (
        k_out
        if cfg.k_out_in_sb
        else sbm.alloc_stack(
            (cfg.d_head, atp.bs_n_prgs * atp.bs_per_nc * cfg.s_active),
            dtype=atp.io_type,
            buffer=nl.sbuf,
        )
    )

    # Load inv_freqs
    sbm.open_scope()
    inv_freqs_sb = sbm.alloc_stack(inv_freqs.shape, dtype=inv_freqs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(inv_freqs_sb, inv_freqs, name=f"{sbm.get_name_prefix()}inv_freqs_load")

    # Compute RoPE coefficients, then apply (while loading) onto Q and K_active (only last NC handles K_active)
    cos, sin = _rope(
        inv_freqs_sb,
        bufs.pos_ids_sb,
        bs=atp.bs_per_nc,
        s_a=cfg.s_active,
        d_head=cfg.d_head,
        sbm=sbm,
    )
    _apply_rope(q, cos, sin, bufs.q_sb, cfg, sbm=sbm, name_suffix="q")
    if cfg.k_out_in_sb or (atp.sprior_prg_id == atp.sprior_n_prgs - 1):
        _apply_rope(k_active, cos, sin, bufs.k_active_sb, cfg, ignore_heads=True, sbm=sbm, name_suffix="k_active")
        # Store K to the second output if not kOutInSB; otherwise we already write to it via name alias to k_active_sb
        if not cfg.k_out_in_sb and k_out is not None:
            k_active_sb_view = (bufs.k_active_sb).reshape_dim(1, [cfg.bs, cfg.s_active])
            k_out_hbm_view = (k_out).squeeze_dim(1).permute([1, 0, 2])
            nisa.dma_copy(
                src=k_active_sb_view,
                dst=k_out_hbm_view,
                name=f"{sbm.get_name_prefix()}k_out_store_after_rope",
            )

    nisa.activation(bufs.q_sb, op=nl.copy, data=bufs.q_sb, scale=1 / math.sqrt(cfg.d_head))
    sbm.close_scope()


"""
Main computation blocks
"""


def _compute_qk_matmul(
    k_prior,
    DBG_TENSORS,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)"""
    num_slots = spp.num_slots
    compute_s_active_bqh = spp.slot_s_active_bqh
    slot_interleave_degree = spp.slot_batch_interleave_degree

    fa_tile_s_prior = fa_ctx.tile_s_prior
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_last_fa_tile = fa_ctx.is_last_fa_tile

    # Use per-tile s_prior for slot interleave calculation
    # For block KV with PE transpose path, each slot also accumulates k_loaded buffers across folds.
    sbuf_usage_per_slot = fa_tile_s_prior * sizeinbytes(k_prior.dtype)
    num_folds_this_tile = 0  # placeholder for flat KV
    if atp.is_block_kv:
        # For FA, compute which folds correspond to this tile
        # Each fold covers block_len * 128 elements of s_prior
        fold_s_prior = atp.block_len * TC.p_max
        fold_start = fa_tile_offset // fold_s_prior
        fold_end = div_ceil(fa_tile_offset + fa_tile_s_prior, fold_s_prior)
        num_folds_this_tile = fold_end - fold_start

        if not atp.use_dma_transpose:
            sbuf_usage_per_slot += num_folds_this_tile * atp.block_len * cfg.d_head * sizeinbytes(atp.k_prior_load_type)
    elif cfg.tp_k_prior and atp.is_fp8_kv:
        sbuf_usage_per_slot += TC.psum_b_max * cfg.d_head * sizeinbytes(atp.k_prior_load_type)
    slot_interleave_degree_safe = _get_safe_batch_interleave_degree(
        sbuf_usage_per_slot,
        slot_interleave_degree,
        sbm,
    )
    if atp.qk_row_tile_factor > 1:
        # Fold the seq-id gather into the replication: the first copy reads Q
        # through its tensor-indirection view, so no separate gathered buffer is
        # materialized. MM1 cannot take the indirect operand itself because row
        # tiling needs tile_position=(d_head, 0), which indirection forbids.
        if bufs.q_ti_index is not None:
            q_width = bufs.q_ti_num_elem
            q_src_view = bufs.q_sb.indirect(bufs.q_ti_index, num_elem=q_width)
        else:
            q_width = bufs.q_sb.shape[1]
            q_src_view = bufs.q_sb[0 : cfg.d_head, :]
        q_sb_128 = sbm.alloc_stack((TC.p_max, q_width), dtype=bufs.q_sb.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_sb_128[0 : cfg.d_head, :], src=q_src_view, engine=nisa.vector_engine)
        cross_partition_copy(
            src=q_sb_128,
            dst=q_sb_128,
            src_start_partition=0,
            dst_start_partition=cfg.d_head,
            num_partitions_to_copy=cfg.d_head,
            free_dim_size=q_width,
        )
    else:
        q_sb_128 = bufs.q_sb

    # Maximum multi-buffer degree inside a slot is 8 (banks) divided by the current scope's interleave degree.
    per_slot_interleave_degree = math.floor(float(TC.psum_b_max) / slot_interleave_degree_safe)

    # FP8 KV: use the resolved FP8 dtype (caller's concrete dtype, or
    # dtype_mode resolution for opaque "float8e4"). Otherwise pass through k_prior.dtype.
    _k_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else k_prior.dtype

    # Pre-compute dma_transpose batching params (before the per-slot loop).
    k_block_len_dma = 0
    k_block_len_row_tiled = 0
    k_prior_4d = None
    k_dma_batch_n_folds = 1
    k_dma_batch_n_slots = 1
    if atp.is_block_kv and atp.use_dma_transpose:
        # Pre-compute for dma_transpose path: reshape k_prior into 4-d tile for indirect transpose.
        if cfg.fp8_packed:
            # Reinterpret fp8 [N, elems_fp8] as bf16 [N, elems_fp8 // 2]
            k_prior_bf16 = bufs.k_prior_reshaped.view(nl.bfloat16)
            k_block_len_dma = bufs.k_prior_reshaped.shape[1] // (cfg.d_head * 2)
            k_block_len_row_tiled = k_block_len_dma // atp.qk_row_tile_factor
            k_prior_4d = k_prior_bf16.reshape(
                (bufs.k_prior_reshaped.shape[0], 1, k_block_len_row_tiled, cfg.d_head * atp.qk_row_tile_factor)
            )
        else:
            k_block_len_dma = atp.block_len
            k_block_len_row_tiled = k_block_len_dma // atp.qk_row_tile_factor
            k_prior_4d = bufs.k_prior_reshaped.reshape(
                (bufs.k_prior_reshaped.shape[0], 1, k_block_len_row_tiled, cfg.d_head * atp.qk_row_tile_factor)
            )

        # 1. use k_block_len_dma since row tiling doesn't reduce block len
        #    from dma perspective.
        # 2. cap at 32 since otherwise dma gather transpose starts to fill up
        #    DGE carve-out space in SBUF leading to poor pipelining.
        k_dma_batch_n_folds, k_dma_batch_n_slots = _compute_dma_batch_params(
            num_folds_this_tile, num_slots, k_block_len_dma, cap=32, sbm=sbm
        )

    k_sb_shared = None
    sbm.open_scope(interleave_degree=slot_interleave_degree_safe, name="qk_matmul")
    for i_s in range(num_slots):
        # Load the scheduled K_prior tile for the current slot into sbuf (tile portion for FA)
        # d_head=64 with block KV + DMA transpose: row-tiling (qk_row_tile_factor=2) packs 2 K positions
        # per 128-partition column, using both halves of the partition dim.
        # Requires sufficient s_prior for the packed DMA layout (small SWA windows excluded).
        # For fp8_packed: allocate as bf16 with half the seq length (each bf16 slot holds 2 fp8 values)
        # For slot batching: allocate a fresh shared buffer per group to avoid anti-deps.
        if k_dma_batch_n_slots > 1:
            if (i_s % k_dma_batch_n_slots) == 0:
                _k_sb_free_per_slot = num_folds_this_tile * k_block_len_row_tiled * TC.p_max
                k_sb_shared = sbm.alloc_stack(
                    (cfg.d_head * atp.qk_row_tile_factor, _k_sb_free_per_slot * k_dma_batch_n_slots),
                    dtype=nl.bfloat16 if cfg.fp8_packed else _k_sb_dtype,
                    buffer=nl.sbuf,
                    align=32,
                )
            k_sb = k_sb_shared
        elif cfg.fp8_packed:
            # fp8_packed: allocate as bf16 (each bf16 slot holds 2 fp8 values).
            # Partition dim = d_head * qk_row_tile_factor (128 when row-tiled, packs 2 positions per column).
            _k_sb_free_size = num_folds_this_tile * k_block_len_row_tiled * TC.p_max
            k_sb = sbm.alloc_stack(
                (cfg.d_head * atp.qk_row_tile_factor, _k_sb_free_size),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                align=32,
            )
        else:
            k_sb = sbm.alloc_stack(
                (cfg.d_head * atp.qk_row_tile_factor, fa_tile_s_prior // atp.qk_row_tile_factor),
                dtype=_k_sb_dtype,
                buffer=nl.sbuf,
                align=32,
            )
        if atp.is_block_kv:
            sbm.open_scope()
            for i_fold_rel in range(num_folds_this_tile):
                i_fold = fold_start + i_fold_rel
                slot_pos = i_s * num_folds_this_tile + i_fold_rel
                cur_blks = (bufs.active_blocks_sb).slice(dim=1, start=slot_pos, end=slot_pos + 1)
                kernel_assert(
                    cur_blks.shape == (TC.p_max, 1),
                    f"Internal error: unexpected shape error after loading current blocks, expected {(TC.p_max, 1)}, got {cur_blks.shape}.",
                )

                if atp.use_dma_transpose:
                    # DMA transpose path: indirect DMA transpose per fold (or batched across multiple folds).
                    # dma_transpose requires uint32 indices, but active_blocks_sb is kept as int32
                    # so V's dma_copy can use oob_mode.skip with -1 sentinels.
                    # Use pre-computed uint32 copy (int32(-1) → float32(-1.0) → uint32(0))
                    # via the vector engine cast done once upfront.
                    # Skip non-leading folds/slots within a batched group.
                    if i_fold_rel % k_dma_batch_n_folds != 0:
                        continue
                    if k_dma_batch_n_slots > 1 and i_s % k_dma_batch_n_slots != 0:
                        continue

                    k_dma_batch_size = k_dma_batch_n_folds * k_dma_batch_n_slots
                    idx_start = i_s * num_folds_this_tile + i_fold_rel

                    blks_u32 = (bufs.active_blocks_sb_u32).slice(
                        dim=1, start=idx_start, end=idx_start + k_dma_batch_size
                    )

                    dst_start = i_fold_rel * k_block_len_row_tiled * TC.p_max if k_dma_batch_n_slots == 1 else 0
                    dst_end = dst_start + k_dma_batch_size * k_block_len_row_tiled * TC.p_max
                    # Note: indirect dma_transpose requires src to be a 4-d tile
                    nisa.dma_transpose(
                        dst=(
                            (k_sb)
                            .slice(1, start=dst_start, end=dst_end)
                            .reshape_dim(1, (k_block_len_row_tiled, k_dma_batch_size * TC.p_max))
                            .expand_dim(1)
                        ),
                        # TODO: Port to NkiTensor once dynamic vector_offset is supported
                        src=k_prior_4d.ap(
                            [
                                [
                                    k_block_len_row_tiled * cfg.d_head * atp.qk_row_tile_factor,
                                    TC.p_max * k_dma_batch_size,
                                ],
                                [1, 1],
                                [cfg.d_head * atp.qk_row_tile_factor, k_block_len_row_tiled],
                                [1, cfg.d_head * atp.qk_row_tile_factor],
                            ],
                            offset=0,
                            vector_offset=blks_u32,
                            indirect_dim=0,
                        ),
                        axes=(3, 1, 2, 0),
                        dge_mode=dge_mode.swdge,
                    )
                else:
                    # PE transpose path: indirect DMA load + PE transposes per fold
                    k_loaded = sbm.alloc_stack(
                        (TC.p_max, atp.block_len * cfg.d_head),
                        dtype=atp.k_prior_load_type,
                        buffer=nl.sbuf,
                    )
                    # Memset K to 0 for easier accuracy debug: not strictly necessary since runtime does not
                    # throw NaN errors by default, and the QK result for skipped blocks is masked
                    # to -inf by the attention mask (so NaN from stale K data doesn't propagate).
                    # Can be removed for max perf since the NaN result is not copied back, and
                    # the SBUF value stays as -inf after softmax masking. Having memset helps debug if
                    # we enable NaN runtime error for issues elsewhere in graph.
                    # NOTE: Commented out for performance, re-enable for NaN debug purposes
                    # nisa.memset(k_loaded, value=0)
                    nisa.dma_copy(
                        dst=k_loaded,
                        # TODO: Port to NkiTensor once dynamic vector_offset is supported
                        src=bufs.k_prior_reshaped.ap(
                            [
                                [atp.block_len * cfg.d_head, TC.p_max],
                                [1, atp.block_len * cfg.d_head],
                            ],
                            offset=0,
                            vector_offset=cur_blks,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                        name=f"{sbm.get_name_prefix()}k_prior_block_load_indirect_fa{fa_ctx.fa_tile_idx}_s{i_s}_f{i_fold}_bt{btc.batch_tile_idx}",
                    )

                    # Transpose to [d_head, blk_len * 128blks]
                    # Explicitly group transposes that can share a single psum bank to allow compiler to fuse to a 1024-free-dim PSUM.
                    transpose_grp_size = min(
                        8, atp.block_len
                    )  # FIXME: parameterize this value 8 to psum free_dim size // data size
                    kernel_assert(
                        atp.block_len % transpose_grp_size == 0,
                        (
                            "Internal error: If block length is greater than 8, then it needs to be a multiple of 8 to allow tiling transpose. "
                            f"Instead got block length of {atp.block_len}."
                        ),
                    )
                    num_transpose_grps = atp.block_len // transpose_grp_size
                    for tp_grp_i in range(num_transpose_grps):
                        for tp_j_in_grp in range(transpose_grp_size):
                            blk_len_i = tp_grp_i * transpose_grp_size + tp_j_in_grp
                            tp_psum = nl.ndarray(
                                (cfg.d_head, TC.p_max),
                                dtype=atp.k_prior_load_type,
                                buffer=nl.psum,
                                address=None
                                if sbm.is_auto_alloc()
                                else (
                                    0,
                                    (tp_j_in_grp % per_slot_interleave_degree) * TC.psum_f_max_bytes,
                                ),
                            )
                            nisa.nc_transpose(
                                tp_psum,
                                k_loaded[:, nl.ds(blk_len_i * cfg.d_head, cfg.d_head)],
                            )

                            # Balance psum->sbuf copies across vector and scalar engines
                            cur_idx = i_fold_rel * atp.block_len + blk_len_i
                            if cur_idx % 2 == 0:
                                engine = nisa.vector_engine
                            else:
                                engine = nisa.scalar_engine

                            nisa.tensor_copy(k_sb[:, nl.ds(cur_idx * 128, 128)], tp_psum, engine=engine)
            sbm.close_scope()
        elif not cfg.tp_k_prior:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.d_head, cfg.full_sprior),
                f"k_prior[1:] expected to have shape {(1, cfg.d_head, cfg.full_sprior)=}, received {k_prior.shape[1:]=}",
            )
            # k_prior shape: [B+, 1, d, full_sprior]
            # K_prior is already transposed, insert flat load
            s_prior_pos = fa_tile_offset
            k_prior_view = (
                (k_prior)
                .select(0, btc.global_batch_offset + i_s)
                .squeeze_dim(0)
                .slice(1, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
            )
            nisa.dma_copy(
                k_sb,
                k_prior_view,
                name=f"{sbm.get_name_prefix()}k_prior_flat_load_transposed_fa{fa_ctx.fa_tile_idx}_s{i_s}_bt{btc.batch_tile_idx}",
            )
        else:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.full_sprior, cfg.d_head),
                f"k_prior[1:] expected to have shape {(1, cfg.full_sprior, cfg.d_head)=}, received {k_prior.shape[1:]=}",
            )

            if atp.is_fp8_kv:
                # Can't do DMA transpose for FP8, so load as BF16, transpose via PSUM, copy to k_sb (casts to FP8)
                sbm.open_scope(interleave_degree=TC.psum_b_max)
                for tp_grp_i in range(fa_tile_n_sprior):
                    tile_start = tp_grp_i * TC.p_max
                    tile_size = min(TC.p_max, fa_tile_s_prior - tile_start)
                    k_loaded = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=atp.k_prior_load_type, buffer=nl.sbuf)
                    s_prior_pos = fa_tile_offset + tile_start
                    k_prior_view = (
                        (k_prior)
                        .select(0, btc.global_batch_offset + i_s)
                        .squeeze_dim(0)
                        .slice(0, start=s_prior_pos, end=s_prior_pos + tile_size)
                    )
                    nisa.dma_copy(
                        dst=k_loaded[:tile_size, :],
                        src=k_prior_view,
                    )
                    tp_psum = nl.ndarray(
                        (cfg.d_head, TC.p_max),
                        dtype=atp.k_prior_load_type,
                        buffer=nl.psum,
                        address=None if sbm.is_auto_alloc() else (0, (tp_grp_i % TC.psum_b_max) * TC.psum_f_max_bytes),
                    )
                    nisa.nc_transpose(tp_psum[:, :tile_size], k_loaded[:tile_size, :])
                    nisa.tensor_copy(k_sb[:, nl.ds(tile_start, tile_size)], tp_psum[:, :tile_size])
                    sbm.increment_section()
                sbm.close_scope()

            else:
                # FIXME: 4d reshape_dim required here, while simple slicing should suffice
                k_sb_view = (k_sb).reshape_dim(1, [1, 1, fa_tile_s_prior])
                s_prior_pos = fa_tile_offset
                k_prior_view = (
                    (k_prior)
                    .select(0, btc.global_batch_offset + i_s)
                    .squeeze_dim(0)
                    .slice(0, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
                    .reshape_dim(1, [1, 1, cfg.d_head])
                )
                nisa.dma_transpose(k_sb_view, k_prior_view)

        # Active K is no longer appended to k_sb: it is folded into the running
        # statistics at finalization, so the cache tiles stay read-only.

        # Do MM1 in grps (default 4k grp size), make sure appropriate group size is selected s.t. psum free < hw limit
        mm1_grp_sz = 4 * 1024
        if (mm1_grp_sz // TC.p_max) * atp.s_active_qh > TC.psum_f_max:
            mm1_grp_sz = (TC.psum_f_max // atp.s_active_qh) * TC.p_max
        n_mm1_per_grp = mm1_grp_sz // TC.p_max

        # For fp8_packed, create the fp8 reinterpreted view once outside the matmul loop
        k_sb_fp8 = (k_sb).view(_k_sb_dtype) if cfg.fp8_packed else None

        # d_head=64: replicate Q to 128 partitions for row-tiled MM1
        # Use cross_partition_copy (nc_stream_shuffle) for cross-partition data movement
        """
        Tiling Strategy for MM1 (KQ^T computation):
        - K stationary: [d_head, s_prior] loaded per slot into k_sb
        - Q moving: [d_head, s_active_qh] per slot from q_sb
        - Tile size: mm1_grp_sz (default 4096 = 4k) to balance PSUM usage
        - PSUM allocation: [P_MAX, n_mm1_per_grp * s_active_qh]
          where n_mm1_per_grp = mm1_grp_sz / P_MAX
        - PSUM constraint: (mm1_grp_sz / P_MAX) * s_active_qh < psum_f_max
        - Output: qk [P_MAX, n_sprior_tile * s_active_bqh] with slot interleaving
        - Memory: Each tile processes P_MAX rows of K against full Q per slot
        """

        for i_mm1_grp in range(div_ceil(fa_tile_s_prior, mm1_grp_sz)):
            # PSUM allocation
            qk_psum = nl.ndarray(
                (TC.p_max, n_mm1_per_grp * atp.s_active_qh),
                dtype=nl.float32,
                buffer=nl.psum,
                address=None
                if sbm.is_auto_alloc()
                else (
                    0,
                    (i_mm1_grp % per_slot_interleave_degree) * TC.psum_f_max_bytes,
                ),
            )
            if atp.qk_row_tile_factor > 1:
                qk_psum_even = qk_psum
                qk_psum_odd = nl.ndarray(
                    (TC.p_max, n_mm1_per_grp * atp.s_active_qh),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    # Manual allocation + row tiling is unsupported.
                    address=None,
                )

            # Inner matmul loop
            for i_mm1 in range(n_mm1_per_grp):
                # K tile loading (shared)
                if (
                    cfg.strided_mm1
                ):  # optionally use strided read to K s.t. MM2 can also be strided with sequential read to V
                    k_tile_offset = i_mm1_grp * n_mm1_per_grp + i_mm1
                    num_acc = min(
                        TC.p_max,
                        (fa_tile_s_prior - 1 - k_tile_offset) // fa_tile_n_sprior + 1,
                    )
                    if num_acc <= 0:
                        break  # k_tile_offset is strictly increasing
                    k_tile = (k_sb).slice(
                        1,
                        start=k_tile_offset,
                        end=k_tile_offset + (num_acc - 1) * fa_tile_n_sprior + 1,
                        step=fa_tile_n_sprior,
                    )
                else:
                    k_tile_offset = i_mm1_grp * mm1_grp_sz + i_mm1 * TC.p_max
                    num_acc = min(TC.p_max, fa_tile_s_prior // atp.qk_row_tile_factor - k_tile_offset)
                    if num_acc <= 0:
                        break  # k_tile_offset is strictly increasing
                    logical_tile_idx = k_tile_offset // TC.p_max
                    if cfg.fp8_packed:
                        # fp8_packed: k_sb_fp8 has interleaved even/odd layout.
                        # Map logical fp8 tile → bf16 chunk (strips parity), then remap for kbld-outer.
                        parity = logical_tile_idx % 2
                        phys_tile = _k_tile_physical_index(
                            logical_tile_idx // 2,
                            k_block_len_row_tiled,
                            k_dma_batch_n_folds,
                            k_dma_batch_n_slots,
                            i_s,
                        )
                        phys_start = phys_tile * 2 * TC.p_max + parity
                        k_tile = k_sb_fp8.slice(1, start=phys_start, end=phys_start + (num_acc - 1) * 2 + 1, step=2)
                    else:
                        # Non-fp8: direct tile access, remapped for kbld-outer layout.
                        phys_tile = _k_tile_physical_index(
                            logical_tile_idx,
                            k_block_len_row_tiled,
                            k_dma_batch_n_folds,
                            k_dma_batch_n_slots,
                            i_s,
                        )
                        k_tile = k_sb[0 : cfg.d_head * atp.qk_row_tile_factor, nl.ds(phys_tile * TC.p_max, num_acc)]

                # Matmul (diverges based on atp.qk_row_tile_factor)
                if atp.qk_row_tile_factor > 1:
                    q_slot_offset = (
                        i_s * atp.s_active_qh
                        if bufs.q_ti_index is not None
                        else (btc.global_batch_offset + i_s) * atp.s_active_qh
                    )
                    nisa.nc_matmul(
                        qk_psum_even[0:num_acc, i_mm1 * atp.s_active_qh : (i_mm1 + 1) * atp.s_active_qh],
                        stationary=k_tile[0 : cfg.d_head, :],
                        moving=q_sb_128[0 : cfg.d_head, q_slot_offset : q_slot_offset + atp.s_active_qh],
                        tile_size=(cfg.d_head, TC.p_max),
                        tile_position=(0, 0),
                    )
                    nisa.nc_matmul(
                        qk_psum_odd[0:num_acc, i_mm1 * atp.s_active_qh : (i_mm1 + 1) * atp.s_active_qh],
                        stationary=k_tile[cfg.d_head : TC.p_max, :],
                        moving=q_sb_128[cfg.d_head : TC.p_max, q_slot_offset : q_slot_offset + atp.s_active_qh],
                        tile_size=(cfg.d_head, TC.p_max),
                        tile_position=(cfg.d_head, 0),
                    )
                else:
                    qk_psum_view = (
                        (qk_psum)
                        .reshape_dim(1, [n_mm1_per_grp, atp.s_active_qh])
                        .select(1, i_mm1)
                        .slice(0, start=0, end=num_acc)
                    )
                    if bufs.q_gathered is not None:
                        # TI path: Q already gathered by seq_id into q_gathered [d_head, num_slots * s_active_qh]
                        q_sb_view = (bufs.q_gathered).reshape_dim(1, [num_slots, atp.s_active_qh]).select(1, i_s)
                    else:
                        # Identity sequence-ID path: index persistent B-sized Q state by batch offset
                        q_sb_view = (
                            (bufs.q_sb)
                            .reshape_dim(1, [atp.bs_full, atp.s_active_qh])
                            .select(1, (btc.global_batch_offset + i_s))
                        )
                    nisa.nc_matmul(
                        qk_psum_view,
                        stationary=k_tile,
                        moving=q_sb_view,
                    )

            # Flush psum -> sb
            if atp.qk_row_tile_factor > 1:
                num_acc_cpy = min(
                    n_mm1_per_grp, fa_tile_s_prior // atp.qk_row_tile_factor // TC.p_max - i_mm1_grp * n_mm1_per_grp
                )
                if num_acc_cpy <= 0:
                    break  # i_mm1_grp * n_mm1_per_grp is strictly increasing

                sprior_tile_pos = i_mm1_grp * n_mm1_per_grp

                if cfg.fp8_packed:
                    # fp8: double interleave → group-of-4 pattern via [n_sprior//2, 2] reshape + stride-2
                    qk_sb_4d = bufs.qk.reshape_dim(1, [fa_tile_n_sprior // 2, 2, num_slots, atp.s_active_qh]).select(
                        3, i_s
                    )
                    mask_sb_4d = bufs.mask_sb.reshape_dim(
                        1, [fa_tile_n_sprior // 2, 2, num_slots, atp.s_active_qh]
                    ).select(3, i_s)
                    grp_start = sprior_tile_pos
                    grp_end = grp_start + num_acc_cpy
                    even_dst = qk_sb_4d[:, grp_start:grp_end:2]
                    even_mask = mask_sb_4d[:, grp_start:grp_end:2]
                    nisa.tensor_copy_predicated(
                        src=qk_psum_even[0:128, 0 : num_acc_cpy * atp.s_active_qh],
                        dst=even_dst,
                        predicate=even_mask,
                    )
                    odd_dst = qk_sb_4d[:, grp_start + 1 : grp_end : 2]
                    odd_mask = mask_sb_4d[:, grp_start + 1 : grp_end : 2]
                    nisa.tensor_copy_predicated(
                        src=qk_psum_odd[0:128, 0 : num_acc_cpy * atp.s_active_qh],
                        dst=odd_dst,
                        predicate=odd_mask,
                    )
                else:
                    # bf16: simple even/odd alternation → stride-2 on flat sprior axis
                    qk_sb_view = bufs.qk.reshape_dim(1, [fa_tile_n_sprior, num_slots, atp.s_active_qh]).select(2, i_s)
                    mask_sb_view = bufs.mask_sb.reshape_dim(1, [fa_tile_n_sprior, num_slots, atp.s_active_qh]).select(
                        2, i_s
                    )
                    even_start = sprior_tile_pos * 2
                    even_end = even_start + num_acc_cpy * 2
                    nisa.tensor_copy_predicated(
                        src=qk_psum_even[0:128, 0 : num_acc_cpy * atp.s_active_qh],
                        dst=qk_sb_view[:, even_start:even_end:2],
                        predicate=mask_sb_view[:, even_start:even_end:2],
                    )
                    nisa.tensor_copy_predicated(
                        src=qk_psum_odd[0:128, 0 : num_acc_cpy * atp.s_active_qh],
                        dst=qk_sb_view[:, even_start + 1 : even_end : 2],
                        predicate=mask_sb_view[:, even_start + 1 : even_end : 2],
                    )
            else:
                # Flush psum -> sb, the write to sb needs to be strided for slot interleaving
                num_acc_cpy = min(n_mm1_per_grp, fa_tile_s_prior // TC.p_max - i_mm1_grp * n_mm1_per_grp)

                if num_acc_cpy <= 0:
                    break  # i_mm1_grp * n_mm1_per_grp is strictly increasing

                qk_psum_view = (
                    (qk_psum).reshape_dim(1, [n_mm1_per_grp, atp.s_active_qh]).slice(1, start=0, end=num_acc_cpy)
                )

                sprior_tile_pos = i_mm1_grp * n_mm1_per_grp
                qk_sb_view = (
                    (bufs.qk)
                    .reshape_dim(1, [fa_tile_n_sprior, num_slots, atp.s_active_qh])
                    .slice(1, start=sprior_tile_pos, end=sprior_tile_pos + num_acc_cpy)
                    .select(2, i_s)
                )

                mask_sb_view = (
                    (bufs.mask_sb)
                    .reshape_dim(1, [fa_tile_n_sprior, num_slots, atp.s_active_qh])
                    .slice(1, start=sprior_tile_pos, end=sprior_tile_pos + num_acc_cpy)
                    .select(2, i_s)
                )
                if num_acc_cpy * atp.s_active_qh == 1:
                    # Use tensor_copy_predicated due to select_reduce bug with free dim 1
                    # Tracked in NKI-2209

                    # Memset QK to -inf
                    # This is necessary because tensor_copy_predicated only copies positions where mask=1,
                    # leaving positions where mask=0 with stale values
                    nisa.memset(qk_sb_view, value=-np.inf)
                    nisa.tensor_copy_predicated(
                        src=qk_psum_view,
                        dst=qk_sb_view,
                        predicate=mask_sb_view,
                    )
                else:
                    nisa.select_reduce(
                        dst=qk_sb_view,
                        predicate=mask_sb_view,
                        on_true=qk_psum_view,
                        on_false=-np.inf,
                    )
        sbm.increment_section()
    sbm.close_scope()

    if DBG_TENSORS:
        if cfg.strided_mm1 and (atp.use_fa or num_slots != atp.bs_per_nc):
            # strided_mm1 + FA has complex K column remapping — write zeros so the tensor is defined.
            # strided_mm1 + batch tiling: batch and sprior tiles are interleaved in QK buffer,
            # so per-tile slices don't concatenate to match the full-batch layout.
            sbm.open_scope()
            dbg_tile_offset = fa_ctx.fa_tile_idx * atp.fa_n_sprior_tile
            bqh_offset = btc.tile_batch_offset * atp.s_active_qh
            dbg_zero = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk.dtype, buffer=nl.sbuf)
            nisa.memset(dbg_zero, 0.0)
            dbg_zero_bc = (
                (dbg_zero)
                .reshape_dim(1, [1, 1, 1, 1])
                .broadcast(2, fa_tile_n_sprior)
                .broadcast(4, compute_s_active_bqh)
            )
            nisa.dma_copy(
                bufs.DBG_QK[
                    :,
                    atp.sprior_prg_id,
                    dbg_tile_offset : dbg_tile_offset + fa_tile_n_sprior,
                    atp.bs_prg_id,
                    bqh_offset : bqh_offset + compute_s_active_bqh,
                ],
                dbg_zero_bc,
                name=f"{sbm.get_name_prefix()}dbg_qk_store_zeros_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )
            sbm.close_scope()
        else:
            # For FA, copy to the slice corresponding to this FA tile
            dbg_tile_offset = fa_ctx.fa_tile_idx * atp.fa_n_sprior_tile
            bqh_offset = btc.tile_batch_offset * atp.s_active_qh
            nisa.dma_copy(
                bufs.DBG_QK[
                    :,
                    atp.sprior_prg_id,
                    dbg_tile_offset : dbg_tile_offset + fa_tile_n_sprior,
                    atp.bs_prg_id,
                    bqh_offset : bqh_offset + compute_s_active_bqh,
                ],
                bufs.qk.reshape((TC.p_max, 1, fa_tile_n_sprior, 1, compute_s_active_bqh)),
                name=f"{sbm.get_name_prefix()}dbg_qk_store_mm1_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )


def _cascaded_max_reduce(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 2. Cascaded max reduce of KQ^T"""
    compute_bs = spp.num_slots
    compute_s_active_bqh = spp.slot_s_active_bqh
    compute_s_active_bqh_remainder = spp.slot_s_active_bqh_remainder
    compute_n_bsq_full_tiles = spp.slot_n_bsq_full_tiles
    compute_n_bsq_tiles = spp.slot_n_bsq_tiles
    compute_s_active_bqh_tile = spp.slot_s_active_bqh_tile

    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    bufs.qk_max = sbm.alloc_stack((TC.p_max, compute_s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf, align=4)

    # Engine balancing splits the max reduce between a vectorized tensor_reduce on
    # DVE and a per-lane activate2 chain on ACT. Under manual allocation the ACT
    # path costs ~1.6 us per lane against ~0.04 us per lane on DVE, and its serial
    # chain gates the next schedule row's QK matmul, so keeping the whole reduce on
    # DVE is 8.5% faster. Under automatic allocation the full-width DVE reduce is
    # itself much more expensive, and balancing wins instead.
    use_engine_balancing_for_max_reduce = sbm.is_auto_alloc()
    if atp.s_prior <= 256:
        # For small S, engine balancing is inefficient.
        use_engine_balancing_for_max_reduce = False
    if nisa.get_nc_version() <= nisa.nc_version.gen3:
        # nisa.activate2 not available on older hardware.
        use_engine_balancing_for_max_reduce = False

    # TODO: revisit porting the QK-swap (transposed-score) path to the packed schedule.
    # The unpacked kernel's swap path folds this max reduction into the PSUM eviction
    # through select_reduce's reduction accumulator, removing this pass entirely. It is
    # not expressible in the layout below: the accumulator reduces the free axis to one
    # value per partition, but here partitions hold s_prior positions and the lane axis
    # must survive the reduction, so the lanes would have to move to the partition axis.
    # Measured on trn3, the swap is worth roughly 7% on the unpacked kernel at a
    # swap-eligible geometry (B=32, s_active_qh=8, d_head=64, 32K context). For the packed
    # schedule the lane count is num_slots * s_active_qh, which must divide the partition
    # dim, so eligibility is narrower than for the unpacked kernel.
    #
    # Step 2.1. Strided reduce from [p_max, tile_n_sprior * bs * s_active_qh] -> [p_max, bs * s_active_qh]
    if not use_engine_balancing_for_max_reduce:
        # This is small (e.g. if n=2, s_a=6, s_p=8192, then free dim is 64*12=768), reasonable to be done with one inst
        qk_view = (bufs.qk).reshape_dim(1, [fa_tile_n_sprior, compute_s_active_bqh]).permute([0, 2, 1])
        nisa.tensor_reduce(
            dst=bufs.qk_max, op=nl.maximum, data=qk_view, axis=[2], keepdims=False
        )  # The axis is modified here
    else:
        # Split across DVE (first half) and ACT/Scalar Engine (second half) for parallelism
        s_active_bqh_half = compute_s_active_bqh // 2
        s_active_bqh_first_half = compute_s_active_bqh - s_active_bqh_half  # ceiling half handles odd s_active_bqh
        s_active_bqh_second_half = s_active_bqh_half

        # First half with DVE (tensor_reduce)
        qk_view = (bufs.qk).reshape_dim(1, [fa_tile_n_sprior, compute_s_active_bqh]).permute([0, 2, 1])
        qk_view_first_half = qk_view.slice(1, start=0, end=s_active_bqh_first_half)
        nisa.tensor_reduce(
            dst=bufs.qk_max[:, :s_active_bqh_first_half],
            op=nl.maximum,
            data=qk_view_first_half,
            axis=[2],
            keepdims=False,
        )

        # Second half with ACT (activate2 with reduction accumulator)
        for i_s_active_bqh_half in nl.affine_range(s_active_bqh_second_half):
            qk_i_s_view = qk_view.select(1, s_active_bqh_first_half + i_s_active_bqh_half)
            nisa.activate2(
                dst=qk_i_s_view,
                op=nl.copy,
                data=qk_i_s_view,
                imm0=1.0,
                imm1=0.0,
                op0=nl.multiply,
                op1=nl.add,
                reduce_op=nl.max,
                reduce_res=bufs.qk_max[:, s_active_bqh_first_half + i_s_active_bqh_half],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

    # Sink prep placement: only the classical per-tile-sync path stages sink into qk_max_buf on the
    # first FA tile (single-NC) and folds it into tile_max during the final reduction below.
    # Sharded paths defer cross-NC sync (and sink fold) to _finalize_and_store.
    should_prep_sink_in_cascade = sink is not None and atp.sync_softmax_per_fa_tile and fa_ctx.fa_tile_idx == 0

    # The free-dim length reserves slots in qk_max_buf (and exp_sum) for:
    #   - 1: local reduction result (always).
    #   - +1 for sink: only when sink is staged in _cascaded_* (per-tile path).
    atp.softmax_final_reduction_length = 1 + should_prep_sink_in_cascade
    atp.softmax_final_reduction_local_idx = 0  # The reduction result from local qk goes to 1st entry.
    atp.softmax_final_reduction_sink_idx = (
        atp.softmax_final_reduction_length - 1 if should_prep_sink_in_cascade else None
    )

    if cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_qk_max_pdim = pad_partitions_for_ext_inst(compute_s_active_bqh_tile)
    else:
        padded_qk_max_pdim = compute_s_active_bqh_tile

    bufs.qk_max_buf = sbm.alloc_stack(
        (padded_qk_max_pdim, compute_n_bsq_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    # Step 2.2 Transpose to psum -> [bs * s_active_qh, p_max]
    sbm.open_scope()
    for i_bsq_tile in range(compute_n_bsq_full_tiles):
        _transpose_max_psum(i_bsq_tile, compute_s_active_bqh_tile, atp, spp, TC, bufs, sbm)

    if compute_s_active_bqh_remainder > 0:
        _transpose_max_psum(
            compute_n_bsq_full_tiles,
            compute_s_active_bqh_remainder,
            atp,
            spp,
            TC,
            bufs,
            sbm,
        )
    sbm.close_scope()

    # Step 2.3.1  If there is sink, load with the right layout.
    if should_prep_sink_in_cascade:
        # Stage sink into qk_max_buf for the per-tile final reduction below (single-NC path).
        sink_offset = compute_n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        _prep_sink(
            sink,
            bufs.qk_max_buf[:compute_s_active_bqh_tile, nl.ds(sink_offset, compute_n_bsq_tiles)],
            atp,
            spp,
            cfg,
            TC,
            sbm,
            btc,
            use_slot_geometry=True,
        )
    # Step 2.3.3  Do the final reduction (2 or 3 reduce to 1) -> [bs * s_active_qh, 1]
    #             Negate if we are doing the reduction to save one op for sink exponential.
    # Do this only if syncing softmax per tile (not deferring to FA finalization)
    atp.max_negated = False
    tile_max = bufs.qk_max_buf[:compute_s_active_bqh_tile, :compute_n_bsq_tiles]
    if atp.softmax_final_reduction_length > 1 and atp.sync_softmax_per_fa_tile:
        atp.max_negated = True
        for i_bsq_tile in range(compute_n_bsq_tiles):
            qk_max_buf_view = (
                (bufs.qk_max_buf)
                .slice(0, start=0, end=compute_s_active_bqh_tile)
                .reshape_dim(1, [atp.softmax_final_reduction_length, compute_n_bsq_tiles])
                .select(2, i_bsq_tile)
            )
            nisa.tensor_reduce(
                tile_max[:, i_bsq_tile],
                data=qk_max_buf_view,
                op=nl.maximum,
                axis=1,
                negate=True,
            )
    elif sink is not None and atp.use_fa and atp.sprior_n_prgs == 1:
        # need to negate in tile > 0 for consistency with 0th tile even though no sink
        atp.max_negated = True
        nisa.tensor_scalar(tile_max, tile_max, op0=nl.multiply, operand0=-1)

    _clamp_max_to_finite(dst=tile_max, src=tile_max, max_negated=atp.max_negated)

    # Packed running statistics are updated at the accumulator boundary.

    # Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]
    # (Either running_max or qk_max_buf depending on whether online softmax is used)
    for i_bsq_tile in range(compute_n_bsq_full_tiles):
        _transpose_broadcast_max(i_bsq_tile, compute_s_active_bqh_tile, atp, spp, TC, sbm, bufs)

    if compute_s_active_bqh_remainder > 0:
        _transpose_broadcast_max(
            compute_n_bsq_full_tiles,
            compute_s_active_bqh_remainder,
            atp,
            spp,
            TC,
            sbm,
            bufs,
        )

    if DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0 and compute_bs != atp.bs_per_nc:
        # Slot-shaped partials do not map directly to the persistent debug layout.
        _store_dbg_qk_max_zeros_full_batch(atp, sbm, bufs)


def _transpose_max_psum(
    index: int,
    tile_size: int,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    TC: TileConstants,
    bufs: AttnInternalBuffers,
    sbm: SbufManager,
):
    """
    Step 2.2 Transpose to psum -> [bs * s_active_qh, p_max]
    Step 2.3.0 Reduce the new 128 fdim while copying to sbuf -> [bs * s_active_qh, 1]
    """
    compute_s_active_bqh_tile = spp.slot_s_active_bqh_tile
    compute_n_bsq_tiles = spp.slot_n_bsq_tiles

    # Step 2.2
    qk_max_psum = nl.ndarray(
        (tile_size, TC.p_max),
        dtype=atp.inter_type,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )
    nisa.nc_transpose(
        qk_max_psum,
        bufs.qk_max[:, nl.ds(index * compute_s_active_bqh_tile, tile_size)],
    )

    # Step 2.3.0
    nisa.tensor_reduce(
        bufs.qk_max_buf[
            :tile_size,
            compute_n_bsq_tiles * atp.softmax_final_reduction_local_idx + index,
        ],
        op=nl.maximum,
        data=qk_max_psum,
        axis=1,
        keepdims=True,
    )


def _transpose_broadcast_max(
    index,
    tile_size,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]"""
    compute_s_active_bqh_tile = spp.slot_s_active_bqh_tile

    sbm.open_scope()
    qk_max_copy = sbm.alloc_stack((TC.p_max, tile_size), dtype=bufs.qk_max.dtype)

    # Packed running_max is updated after the row completes, so broadcast this row's tile max.
    max_src_tensor = bufs.qk_max_buf[:tile_size]

    tp_broadcast(
        src=max_src_tensor, dst=qk_max_copy, src_offset=index, psum_address=None if sbm.is_auto_alloc() else (0, 0)
    )
    nisa.tensor_copy(
        bufs.qk_max[:, nl.ds(index * compute_s_active_bqh_tile, tile_size)],
        qk_max_copy,
    )
    sbm.close_scope()


def _compute_exp_qk(
    DBG_TENSORS,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 3. Exp(KQ^T - max(KQ^T))"""
    compute_bs = spp.num_slots
    compute_s_active_bqh = spp.slot_s_active_bqh

    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    # Instruction startup time on TRN2 does not outweight pipelining advantages
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        for i_s_prior in range(fa_tile_n_sprior):
            qk_view = (
                (bufs.qk)
                .reshape_dim(1, [fa_tile_n_sprior, compute_s_active_bqh])
                .slice(1, start=i_s_prior, end=i_s_prior + 1)
            )

            nisa.tensor_tensor(qk_view, qk_view, bufs.qk_max, op=(nl.add if atp.max_negated else nl.subtract))

            qk_io_type_view = (
                (bufs.qk_io_type)
                .reshape_dim(1, [fa_tile_n_sprior, compute_s_active_bqh])
                .slice(1, start=i_s_prior, end=i_s_prior + 1)
            )
            nisa.activation(qk_io_type_view, op=nl.exp, data=qk_view)
    else:
        qk_max_view = (bufs.qk_max).expand_dim(1).broadcast(1, fa_tile_n_sprior)

        nisa.tensor_tensor(
            bufs.qk,
            bufs.qk,
            qk_max_view,
            op=(nl.add if atp.max_negated else nl.subtract),
        )
        nisa.activation(bufs.qk_io_type, op=nl.exp, data=bufs.qk)

    if DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0:
        # Packed row exponentials are relative to row-local maxima and are not
        # meaningful in the persistent debug layout; define the tensor as zero.
        sbm.open_scope()
        full_bqh = bufs.DBG_QK_EXP.shape[-1]
        dbg_zero = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk_io_type.dtype, buffer=nl.sbuf)
        nisa.memset(dbg_zero, 0.0)
        dbg_zero_bc = (dbg_zero).reshape_dim(1, [1, 1, 1, 1]).broadcast(2, atp.n_sprior_tile).broadcast(4, full_bqh)
        nisa.dma_copy(
            bufs.DBG_QK_EXP[:, atp.sprior_prg_id, :, atp.bs_prg_id, :],
            dbg_zero_bc,
            name=f"{sbm.get_name_prefix()}dbg_qk_exp_store_zeros",
        )
        sbm.close_scope()


def _cascaded_sum_reduction(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 4. Cascaded sum reduction of exp"""
    compute_bs = spp.num_slots
    compute_s_active_bqh = spp.slot_s_active_bqh
    compute_s_active_bqh_remainder = spp.slot_s_active_bqh_remainder
    compute_n_bsq_full_tiles = spp.slot_n_bsq_full_tiles
    compute_n_bsq_tiles = spp.slot_n_bsq_tiles
    compute_s_active_bqh_tile = spp.slot_s_active_bqh_tile

    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    if cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_sum_pdim = pad_partitions_for_ext_inst(compute_s_active_bqh_tile)
    else:
        padded_exp_sum_pdim = compute_s_active_bqh_tile
    bufs.exp_sum = sbm.alloc_stack(
        (padded_exp_sum_pdim, compute_n_bsq_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    sbm.open_scope()
    for i_bsq_tile in range(compute_n_bsq_full_tiles):
        _tile_sum_reduction(i_bsq_tile, compute_s_active_bqh_tile, fa_tile_n_sprior, atp, spp, TC, bufs, sbm)
    sbm.close_scope()

    if compute_s_active_bqh_remainder > 0:
        sbm.open_scope()
        _tile_sum_reduction(
            compute_n_bsq_full_tiles,
            compute_s_active_bqh_remainder,
            fa_tile_n_sprior,
            atp,
            spp,
            TC,
            bufs,
            sbm,
        )
        sbm.close_scope()

    if sink is not None and atp.sync_softmax_per_fa_tile and fa_ctx.fa_tile_idx == 0:
        kernel_assert(
            atp.max_negated,
            "Internal error: Unexpectedly found that maximum has not been negated when using sink",
        )
        kernel_assert(
            atp.softmax_final_reduction_sink_idx is not None,
            "Internal error: Unexpectedly found that softmax_final_reduction_sink_idx is None",
        )
        reduction_offset = compute_n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        for i_bsq_tile in range(compute_n_bsq_tiles):
            max_buf_for_sink = bufs.running_max
            nisa.tensor_scalar(
                bufs.qk_max_buf[:compute_s_active_bqh_tile, reduction_offset + i_bsq_tile],
                bufs.qk_max_buf[:compute_s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.add,
                max_buf_for_sink[:compute_s_active_bqh_tile, i_bsq_tile],
            )
            nisa.activation(
                bufs.exp_sum[:compute_s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.exp,
                bufs.qk_max_buf[:compute_s_active_bqh_tile, reduction_offset + i_bsq_tile],
            )

    if atp.softmax_final_reduction_length > 1 and atp.sync_softmax_per_fa_tile:
        for i_bsq_tile in range(compute_n_bsq_tiles):
            exp_sum_view = (
                (bufs.exp_sum)
                .slice(0, start=0, end=compute_s_active_bqh_tile)
                .reshape_dim(1, [atp.softmax_final_reduction_length, compute_n_bsq_tiles])
                .select(2, i_bsq_tile)
            )
            nisa.tensor_reduce(
                bufs.exp_sum[:compute_s_active_bqh_tile, i_bsq_tile],
                data=exp_sum_view,
                op=nl.add,
                axis=1,
            )

    # Packed running statistics are updated at the accumulator boundary.

    if DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0 and compute_bs != atp.bs_per_nc:
        # Slot-shaped partials do not map directly to the persistent debug layout.
        _store_dbg_exp_sum_zeros_full_batch(atp, sbm, bufs)


def _tile_sum_reduction(
    index,
    tile_size,
    tile_n_sprior,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    TC: TileConstants,
    bufs: AttnInternalBuffers,
    sbm: SbufManager,
):
    """
    Step 4.1. Each of the tile_n_sprior matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    Step 4.2. Copy partial reduce output from psum -> sb while reducing the free dim (num_sprior_t128)
    tile_size is either the selected full compute tile or its remainder.
    """
    compute_s_active_bqh = spp.slot_s_active_bqh
    compute_s_active_bqh_tile = spp.slot_s_active_bqh_tile
    compute_n_bsq_tiles = spp.slot_n_bsq_tiles

    sum_reduce_psum = nl.ndarray(
        (tile_size, tile_n_sprior),
        dtype=nl.float32,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )

    # Step 4.1. Each of the tile_n_sprior matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    for i_exp_reduce in range(tile_n_sprior):
        sum_reduce_psum_view = (sum_reduce_psum).slice(1, start=i_exp_reduce, end=i_exp_reduce + 1)
        s_active_bqh_pos = index * compute_s_active_bqh_tile
        qk_io_type_view = (
            (bufs.qk_io_type)
            .reshape_dim(1, [tile_n_sprior, compute_s_active_bqh])
            .select(1, i_exp_reduce)
            .slice(1, start=s_active_bqh_pos, end=s_active_bqh_pos + tile_size)
        )

        nisa.nc_matmul(
            sum_reduce_psum_view,
            stationary=qk_io_type_view,
            moving=bufs.one_vec,
        )

    # Step 4.2. Copy partial reduce output from psum -> sb while reducing the free dim (num_sprior_t128)
    nisa.tensor_reduce(
        bufs.exp_sum[
            :tile_size,
            compute_n_bsq_tiles * atp.softmax_final_reduction_local_idx + index,
        ],
        op=nl.add,
        data=sum_reduce_psum,
        axis=1,
    )


def _column_tile_transpose(src, dst, index, tile_size, tile_stride, TC: TileConstants):
    """Transpose a column tile to a row and place it at the correct offset in dst.

    Transposes src[0:tile_size, index:index+1] to dst[0:1, base_offset:base_offset+tile_size]
    where base_offset = index * tile_stride.

    Args:
        src: Source tensor with shape [tile_stride, num_tiles]
        dst: Destination tensor with shape [1, total_size] or broadcastable
        index: Which column tile to transpose (0-indexed)
        tile_size: Number of elements in this tile (may be less than tile_stride for remainder)
        tile_stride: Stride between tiles in the output
    """
    base_offset = index * tile_stride
    for quadrant_idx in range(div_ceil(tile_size, TC.sbuf_quadrant_size)):
        offset = quadrant_idx * TC.sbuf_quadrant_size
        full_offset = base_offset + offset
        tp_size = min(TC.sbuf_quadrant_size, tile_size - offset)
        # Even though TP on vector engine is slower, the vector engine is not busy while the tensor engine is
        nisa.nc_transpose(
            dst[:1, full_offset : full_offset + tp_size],
            src[offset : offset + tp_size, index : index + 1],
            engine=nisa.vector_engine,
        )


def _s_active_bqh_tile_transpose_broadcast(src, dst, atp: AttnTileParams, TC: TileConstants):
    """Transpose all tiles from src and broadcast to dst.

    Transposes src with shape [s_active_bqh_tile, n_bsq_tiles] to [1,s_active_bqh]
    and then broadcast to dst with shape [d_head, s_active_bqh].

    Args:
        src: Source tensor with shape [s_active_bqh_tile, n_bsq_tiles]
        dst: Destination tensor with shape [d_head, s_active_bqh]
    """
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _column_tile_transpose(src, dst, i_bsq_tile, atp.s_active_bqh_tile, atp.s_active_bqh_tile, TC)
    if atp.s_active_bqh_remainder > 0:
        _column_tile_transpose(src, dst, atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp.s_active_bqh_tile, TC)
    stream_shuffle_broadcast(src=dst[:1, : atp.s_active_bqh], dst=dst)


def _compute_pv_matmul_and_store(
    v_prior,
    v_active,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 5. Matmult 2 of (exp @ V)^T and store output"""
    num_slots = spp.num_slots
    compute_s_active_bqh = spp.slot_s_active_bqh
    slot_interleave_degree = spp.slot_batch_interleave_degree

    fa_tile_s_prior = fa_ctx.tile_s_prior
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_last_fa_tile = fa_ctx.is_last_fa_tile
    v_active_rows = bufs.v_active_reshaped.reshape_dim(1, [cfg.s_active, cfg.d_head])

    exp_v_sendrecv_gpsimd = (
        cfg.use_gpsimd_sb2sb and atp.n_prgs > 1 and num_slots * atp.s_active_qh <= 256 and cfg.d_head % 16 == 0
    )
    if exp_v_sendrecv_gpsimd:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_v_pdim = pad_partitions_for_ext_inst(cfg.d_head)
    else:
        padded_exp_v_pdim = cfg.d_head
    bufs.exp_v = sbm.alloc_stack(
        (padded_exp_v_pdim, num_slots, atp.s_active_qh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    slot_interleave_degree_safe = _get_safe_batch_interleave_degree(
        cfg.d_head * fa_tile_n_sprior * sizeinbytes(v_prior.dtype), slot_interleave_degree, sbm
    )

    # PV array tiling: with d_head < p_max the PE array is only partly used by a
    # single [p_max, d_head] stationary tile. Packing pv_dense_factor s_prior
    # tiles into one matmul fills the array and cuts the MM2 instruction count
    # proportionally. Results land on the diagonal blocks of the grid PSUM.
    # Tiling is along a slot's own s_prior axis, so slots stay independent.
    _PV_ARRAY_TILING_THRESHOLD = 8192
    pv_array_tiling_factor = TC.p_max // cfg.d_head
    pv_array_tiling_ok = (
        atp.is_block_kv
        and cfg.d_head < TC.p_max
        and cfg.d_head % 32 == 0
        and fa_tile_s_prior >= _PV_ARRAY_TILING_THRESHOLD
    )
    pv_dense_factor = (
        pv_array_tiling_factor
        if (pv_array_tiling_ok and pv_array_tiling_factor * atp.s_active_qh <= TC.psum_f_max)
        else 1
    )

    """
    Tiling Strategy for MM2 ((exp @ V)^T computation and output):
    - V stationary: [s_prior, d_head] loaded per slot into v_sb as [P_MAX, n_sprior_tile * d_head]
    - exp(QK) moving: [P_MAX, s_active_bqh] from qk_io_type (already computed and normalized)
    - Output: exp_v [d_head, num_slots, s_active_qh] accumulated in PSUM then copied to SBUF
    - PSUM allocation: [d_head, s_active_qh] per slot
    - Memory layout: V loaded horizontally tiled (strided if strided_mm1=False, sequential if True)
    - Slot interleaving: Uses slot_interleave_degree_safe for DMA/compute overlap
    - Final output: Gathered across cores if sprior_n_prgs > 1, then stored to HBM or kept in SBUF
    """

    # V-load DMA batching setup
    V_CAP = 64  # cap to 64 effective block size after batching to keep memory consumption reasonable while getting benefit of batching
    v_block_len = atp.block_len if atp.is_block_kv else 0
    v_dma_batch_n_folds = 1
    v_dma_batch_n_slots = 1
    v_idx_table = None
    if atp.is_block_kv and v_block_len > 0:
        v_dma_batch_n_folds, v_dma_batch_n_slots = _compute_dma_batch_params(
            atp.num_folds_per_batch, num_slots, v_block_len, cap=V_CAP, sbm=sbm
        )
    v_dma_batch_size = v_dma_batch_n_folds * v_dma_batch_n_slots
    v_idx_src = bufs.active_blocks_sb if atp.use_v_dma_skipping else bufs.active_blocks_sb_u32
    if v_dma_batch_size > 1:
        # V's batched dma_copy reads a [128, N] vector_offset in column-major ("snake") order, so the
        # index table must be pre-rearranged into that layout. The K path uses dma_transpose instead,
        # which consumes a contiguous slice of active_blocks_sb_u32 directly and needs no rearrange.
        v_idx_table = _rearrange_indices_for_batched_dma(
            v_idx_src, num_slots * atp.num_folds_per_batch, v_dma_batch_size, TC, sbm
        )

    # FP8 KV: use the resolved FP8 dtype (caller's concrete dtype, or
    # dtype_mode resolution for opaque "float8e4"). Otherwise pass through v_prior.dtype.
    _v_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else v_prior.dtype
    v_sb_shared = None

    sbm.open_scope(interleave_degree=slot_interleave_degree_safe, name="pv_matmul")
    for i_s in range(num_slots):
        seq_id_u32 = bufs.safe_seq_ids_sb[0:1, i_s : i_s + 1].view(nl.uint32)
        v_active_sequence = v_active_rows.select(dim=0, index=seq_id_u32)
        # Load V_prior from HBM [s_prior, d_head] into SB [128, (tile_s_prior / 128) * d_head]
        # Do strided load (horizontal tile) if not strided_mm1, otherwise load sequentially for better DMA throughput
        # V buffer allocation: shared for slot batching, per-slot otherwise.
        per_slot_v_size = cfg.d_head * fa_tile_n_sprior
        if v_dma_batch_n_slots > 1:
            slot_in_group = i_s % v_dma_batch_n_slots
            if slot_in_group == 0:
                v_sb_shared = sbm.alloc_stack(
                    (TC.p_max, per_slot_v_size * v_dma_batch_n_slots), dtype=_v_sb_dtype, buffer=nl.sbuf
                )
            v_sb = (v_sb_shared).slice(
                1, start=slot_in_group * per_slot_v_size, end=(slot_in_group + 1) * per_slot_v_size
            )
        else:
            v_sb = sbm.alloc_stack((TC.p_max, per_slot_v_size), dtype=_v_sb_dtype, buffer=nl.sbuf)
        v_sb_view = (v_sb).reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
        if atp.is_block_kv:
            # For FA, compute which folds correspond to this tile
            fold_s_prior = atp.block_len * TC.p_max
            fold_start = fa_tile_offset // fold_s_prior
            fold_end = div_ceil(fa_tile_offset + fa_tile_s_prior, fold_s_prior)
            num_folds_this_tile = fold_end - fold_start

            if atp.use_v_dma_skipping:
                # This memset is required for oob skip to prevent uninitialized NaNs from corrupting results.
                # With slot batching the load targets the whole v_sb_shared on the leading slot, so memset
                # the full shared buffer once (on slot_in_group == 0) and skip non-leading slots to avoid
                # zeroing data the leading-slot load already wrote.
                if v_dma_batch_n_slots > 1:
                    if slot_in_group == 0:
                        nisa.memset(v_sb_shared, value=0)
                else:
                    nisa.memset(v_sb, value=0)

            sbm.open_scope()
            for i_fold_rel in range(num_folds_this_tile):
                i_fold = fold_start + i_fold_rel
                if v_dma_batch_n_folds > 1 and i_fold_rel % v_dma_batch_n_folds != 0:
                    continue
                if v_dma_batch_n_slots > 1 and (i_s % v_dma_batch_n_slots) != 0:
                    continue

                idx_start = i_s * atp.num_folds_per_batch + i_fold_rel

                # Indices: use rearranged table when batching (v_dma_batch_size > 1), direct slice otherwise.
                if v_dma_batch_size > 1:
                    v_idx_slice = (v_idx_table).slice(dim=1, start=idx_start, end=idx_start + v_dma_batch_size)
                else:
                    v_idx_slice = v_idx_src.slice(dim=1, start=idx_start, end=idx_start + 1)

                load_len = v_dma_batch_size * atp.block_len * cfg.d_head
                if v_dma_batch_n_slots > 1:
                    load_dst_view = v_sb_shared
                else:
                    load_start = i_fold_rel * atp.block_len * cfg.d_head
                    load_dst_view = (v_sb).slice(1, start=load_start, end=load_start + load_len)

                nisa.dma_copy(
                    dst=load_dst_view,
                    # TODO: Port to NkiTensor once dynamic vector_offset is supported
                    src=bufs.v_prior_reshaped.ap(
                        [
                            [atp.block_len * cfg.d_head, TC.p_max * v_dma_batch_size],
                            [1, atp.block_len * cfg.d_head],
                        ],
                        offset=0,
                        vector_offset=v_idx_slice,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if atp.use_v_dma_skipping else oob_mode.error,
                    name=f"{sbm.get_name_prefix()}v_prior_block_load_indirect_fa{fa_ctx.fa_tile_idx}_s{i_s}_f{i_fold}_bt{btc.batch_tile_idx}",
                )
            sbm.close_scope()
        elif cfg.strided_mm1:
            s_prior_pos = fa_tile_offset
            v_prior_view = (
                (v_prior)
                .select(0, btc.global_batch_offset + i_s)
                .squeeze_dim(0)
                .slice(0, start=s_prior_pos, end=s_prior_pos + (TC.p_max * fa_tile_n_sprior))
                .reshape_dim(0, [TC.p_max, fa_tile_n_sprior])
            )
            nisa.dma_copy(
                v_sb_view,
                v_prior_view,
                name=f"{sbm.get_name_prefix()}v_prior_load_strided_mm1_fa{fa_ctx.fa_tile_idx}_s{i_s}_bt{btc.batch_tile_idx}",
            )
        else:
            s_prior_pos = fa_tile_offset
            v_prior_view = (
                (v_prior)
                .select(0, btc.global_batch_offset + i_s)
                .squeeze_dim(0)
                .slice(0, start=s_prior_pos, end=s_prior_pos + (TC.p_max * fa_tile_n_sprior))
                .reshape_dim(0, [fa_tile_n_sprior, TC.p_max])
                .permute((1, 0, 2))
            )
            nisa.dma_copy(
                dst=v_sb_view,
                src=v_prior_view,
                name=f"{sbm.get_name_prefix()}v_prior_load_sequential_fa{fa_ctx.fa_tile_idx}_s{i_s}_bt{btc.batch_tile_idx}",
            )

        # Load V_active to the last portion if needed (only on last FA tile)
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1 and is_last_fa_tile:
            if atp.is_block_kv:
                num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
                extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active

                v_sb_partition_base = TC.p_max - num_blks_covering_s_active
                v_sb_s_prior_base = (num_folds_this_tile - 1) * atp.block_len

                # Need to mask as dim_0 * blk_len + dim_1 >= extra_covered
                # Solving the above inequality with 0 <= dim_0 < num_blks_covering_s_active and 0 <= dim_1 < blk_len
                # we get (dim_0, dim_1) in {(0,[extra_covered, blk_len)) and ([1, num_blks_covering_s_active), [0, blk_len))}
                # Thus, if extra_covered != 0, we do an access pattern for dim_0 == 0 and dim_1 in [extra_covered, blk_len)
                # and for main copy we don't need any restrictions
                if extra_covered > 0:
                    if atp.block_len > extra_covered:
                        v_sb_view = (
                            (v_sb)
                            .slice(0, start=v_sb_partition_base, end=v_sb_partition_base + 1)
                            .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                            .slice(1, start=v_sb_s_prior_base + extra_covered, end=v_sb_s_prior_base + atp.block_len)
                        )

                        first_block_active = atp.block_len - extra_covered
                    if num_blks_covering_s_active > 1:
                        v_sb_view = (
                            (v_sb)
                            .slice(
                                0, start=v_sb_partition_base + 1, end=v_sb_partition_base + num_blks_covering_s_active
                            )
                            .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                            .slice(1, start=v_sb_s_prior_base, end=v_sb_s_prior_base + atp.block_len)
                        )

                        first_block_active = atp.block_len - extra_covered
                else:
                    v_sb_view = (
                        (v_sb)
                        .slice(0, start=v_sb_partition_base, end=v_sb_partition_base + num_blks_covering_s_active)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                        .slice(1, start=v_sb_s_prior_base, end=v_sb_s_prior_base + atp.block_len)
                    )

            elif cfg.strided_mm1:
                # Need to load V_active in a strided manner across the entire free dim, this requires two loads because we need
                # to load s_active rows of d_head into v_sb, which has free dim of (tile_n_sprior * d_head).
                load1_nrows = cfg.s_active % fa_tile_n_sprior
                load2_nrows = cfg.s_active - load1_nrows

                # Load 1. Load the first (s_active % tile_n_sprior) rows of V_active (less than one row in v_sb)
                if load1_nrows > 0:
                    load1_pidx = TC.p_max - (load2_nrows // fa_tile_n_sprior) - 1

                    v_sb_view = (
                        (v_sb)
                        .slice(0, start=load1_pidx, end=load1_pidx + 1)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                        .slice(1, start=fa_tile_n_sprior - load1_nrows, end=fa_tile_n_sprior)
                    )

                    v_active_view = (
                        (v_active).select(0, btc.global_batch_offset + i_s).slice(1, start=0, end=load1_nrows)
                    )

                    nisa.dma_copy(
                        v_sb_view,
                        v_active_view,
                        name=f"{sbm.get_name_prefix()}v_active_strided_load_partial_fa{fa_ctx.schedule_row}_s{i_s}_bt{btc.batch_tile_idx}",
                    )

                # Load 2. Load the remaining rows
                if load2_nrows > 0:
                    load2_pidx = TC.p_max - (load2_nrows // fa_tile_n_sprior)

                    v_sb_view = (
                        (v_sb)
                        .slice(0, start=load2_pidx, end=load2_pidx + load2_nrows // fa_tile_n_sprior)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                    )

                    v_active_view = (
                        (v_active)
                        .select(0, btc.global_batch_offset + i_s)
                        .squeeze_dim(0)
                        .slice(0, start=load1_nrows, end=load1_nrows + load2_nrows)
                        .reshape_dim(0, [load2_nrows // fa_tile_n_sprior, fa_tile_n_sprior])
                    )

                    nisa.dma_copy(
                        v_sb_view,
                        v_active_view,
                        name=f"{sbm.get_name_prefix()}v_active_strided_load_remaining_fa{fa_ctx.schedule_row}_s{i_s}_bt{btc.batch_tile_idx}",
                    )
            else:
                v_active_view = (v_active).select(0, btc.global_batch_offset + i_s).squeeze_dim(0)
                # Load to the bottom right part of last chunk of v_sb: [s_active, d_head]
                nisa.dma_copy(
                    v_sb[TC.p_max - cfg.s_active :, v_sb.shape[1] - cfg.d_head :],
                    v_active_view,
                    name=f"{sbm.get_name_prefix()}v_active_load_sequential_fa{fa_ctx.schedule_row}_s{i_s}_bt{btc.batch_tile_idx}",
                )

        # Perform V^T @ exp^T, which equals to (exp @ V)^T. Recall mm1 output is transposed - KQ^T
        exp_v_psum = nl.ndarray(
            (cfg.d_head * pv_dense_factor, atp.s_active_qh * pv_dense_factor),
            dtype=nl.float32,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, (i_s % slot_interleave_degree_safe) * TC.psum_f_max_bytes),
        )
        slot_s_active_qh_pos = i_s * atp.s_active_qh
        for i_t in range(0, fa_tile_n_sprior, pv_dense_factor):
            n_col_tiles = min(pv_dense_factor, fa_tile_n_sprior - i_t)
            # Stationary spans n_col_tiles contiguous [p_max, d_head] V tiles; the
            # moving operand takes the matching s_prior tiles for this slot only.
            # The stationary AP must stay one free dimension, so slice v_sb flat.
            v_sb_view = v_sb[:, i_t * cfg.d_head : (i_t + n_col_tiles) * cfg.d_head]
            qk_io_type_view = (
                (bufs.qk_io_type)
                .reshape_dim(1, [fa_tile_n_sprior, compute_s_active_bqh])
                .slice(1, start=i_t, end=i_t + n_col_tiles)
                .slice(2, start=slot_s_active_qh_pos, end=slot_s_active_qh_pos + atp.s_active_qh)
            )
            nisa.nc_matmul(
                exp_v_psum[: n_col_tiles * cfg.d_head, : n_col_tiles * atp.s_active_qh],
                stationary=v_sb_view,
                moving=qk_io_type_view,
            )

        # Copy mm2 output from psum -> sb while multiplying recip(sum)
        # When online softmax is active, we don't multiply by recip(sum) here — that's done in finalize.
        # exp_sum_recip = exp_sum_recip.reshape((p_max, num_slots, s_active_qh))
        exp_v_view = (bufs.exp_v).select(1, i_s).slice(0, start=0, end=cfg.d_head)
        # Packed accumulator normalization is deferred until finalization.
        # Dense tiling leaves one partial per diagonal block; fold them together.
        nisa.tensor_copy(exp_v_view, exp_v_psum[0 : cfg.d_head, 0 : atp.s_active_qh])
        actual_col_tiles_used = min(pv_dense_factor, fa_tile_n_sprior)
        for i_col in range(1, actual_col_tiles_used):
            nisa.tensor_tensor(
                exp_v_view,
                exp_v_view,
                exp_v_psum[
                    i_col * cfg.d_head : (i_col + 1) * cfg.d_head,
                    i_col * atp.s_active_qh : (i_col + 1) * atp.s_active_qh,
                ],
                op=nl.add,
            )
        sbm.increment_section()
    sbm.close_scope()

    # _sequence_packing_accumulator_update merges this row partial into persistent state.


def _gather_and_store_output(
    out: nl.NkiTensor,
    res: nl.NkiTensor,
    exp_v_sendrecv_gpsimd: bool,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    btc: BatchTileContext,
):
    """Gather partial results from other NC if sharded, then store output to HBM/SBUF.

    Args:
        out: Output tensor in HBM/SBUF
        res: Result tensor in SBUF with shape [d_head, s_active_bqh]
        btc: Batch tile context for correct output offset
    """
    sbm.open_scope()
    # Gather and add partial results from other NC if sprior is sharded
    if atp.sprior_n_prgs > 1:
        res_recv = sbm.alloc_stack(res.shape, res.dtype, buffer=nl.sbuf)
        nisa.sendrecv(
            src=res,
            dst=res_recv,
            send_to_rank=(1 - atp.sprior_prg_id),
            recv_from_rank=(1 - atp.sprior_prg_id),
            pipe_id=0,
            dma_engine=dma_engine.gpsimd_dma if exp_v_sendrecv_gpsimd else dma_engine.dma,
        )
        # Only NC0 adds partial results, unless we have out_in_sb then both cores will obtain the result
        if cfg.out_in_sb or (atp.sprior_prg_id == 0):
            nisa.tensor_tensor(res, res, res_recv, op=nl.add)

    # Store to output
    if cfg.out_in_sb:
        # exp_v and out may have different dtype, easier to just always keep this tensor copy to do the conversion
        res_reshaped = res.reshape((cfg.d_head, atp.s_active_bqh))

        out_offset = btc.global_batch_offset * atp.s_active_qh
        nisa.tensor_copy(out[0 : cfg.d_head, nl.ds(out_offset, atp.s_active_bqh)], src=res_reshaped)
        if atp.bs_n_prgs > 1:
            dst_bs_offset = ((1 - atp.bs_prg_id) * atp.bs_per_nc + btc.tile_batch_offset) * atp.s_active_qh
            nisa.sendrecv(
                src=out[0 : cfg.d_head, nl.ds(out_offset, atp.s_active_bqh)],
                dst=out[0 : cfg.d_head, nl.ds(dst_bs_offset, atp.s_active_bqh)],
                send_to_rank=(1 - atp.bs_prg_id),
                recv_from_rank=(1 - atp.bs_prg_id),
                pipe_id=0,
                dma_engine=dma_engine.gpsimd_dma if exp_v_sendrecv_gpsimd else dma_engine.dma,
            )
    else:
        # Save exp_v (output) into DRAM (each NC writes its own batches in case of batch sharded)
        # This needs to be strided save due to different layout in SBUF and DRAM:
        #   SBUF: [d_head, s_active * n_qhead_per_kvhead]
        #   DRAM: [n_qhead_per_kvhead, d_head, s_active]
        if atp.sprior_prg_id == 0:
            res_reshaped = res.reshape((cfg.d_head, atp.bs, cfg.q_head, cfg.s_active))
            batch_pos = btc.global_batch_offset
            out_view = (
                (out)  # [B, H, d, S_active]
                .slice(0, start=batch_pos, end=batch_pos + atp.bs)
                .permute((2, 0, 1, 3))
            )

            nisa.dma_copy(
                dst=out_view,
                src=res_reshaped,
                name=f"{sbm.get_name_prefix()}out_store_hbm_bt{btc.batch_tile_idx}",
            )
    sbm.close_scope()


"""
Sharding Logic
"""


def _get_lnc_sharding(cfg: AttnTKGConfig) -> Tuple[int, int, int, int]:
    """Return batch-only sharding parameters for packed attention."""
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    kernel_assert(
        n_prgs <= 2,
        f"Packed attention supports unsharded or LNC2 batch sharding; got a SPMD grid size of {n_prgs}.",
    )
    kernel_assert(
        cfg.bs % n_prgs == 0,
        f"Packed attention batch size must divide evenly across logical NCs; got bs={cfg.bs}, logical NCs={n_prgs}.",
    )

    return 1, 0, n_prgs, prg_id


"""
RoPE
"""


def _apply_rope(
    x_inp,
    cos,
    sin,
    x_embed,
    cfg: AttnTKGConfig,
    ignore_heads: bool = False,
    sbm: SbufManager = None,
    name_suffix: str = "",
):
    """Applies rotary embedding for x following this algorithm:
      def _rotate_half(x) -> Tensor:
        '''Rotates half the hidden dims of the input.'''
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

      x_embed = (x * cos) + (_rotate_half(x) * sin)

    Args:
      x_input: HBM input memloc, shape [bs, n_head, s_active, d_head]
      cos: SB input memloc, shape [par(d_head), bs * s_active]
      sin: SB input memloc, shape [par(d_head), bs * s_active]
      cfg: attention tokengen config, including shapes and optimization configs
      ignore_heads: For some inputs (e.g. K active), ignore the head dim in cfg
      sbm: SbufManager of calling kernel

    Returns:
      x_embed: SB output memloc, shape [par(d_head), bs * n_head * s_active]
    """
    # Get basic shapes
    bs, s_active, d_head = cfg.bs, cfg.s_active, cfg.d_head
    n_head = x_inp.shape[1] if not cfg.qk_in_sb else x_inp.shape[1] // (bs * s_active)
    n_head = 1 if ignore_heads else n_head

    x_f = bs * n_head * s_active  # free dim of x after load + transpose
    x = x_inp if cfg.qk_in_sb else sbm.alloc_stack((d_head, x_f), dtype=nl.float32, buffer=nl.sbuf)
    x_shape_expanded = (d_head, bs, n_head, s_active)

    # Load and transpose x_inp from BNSd to BNdS
    if not cfg.qk_in_sb:
        x_inp = x_inp.reshape((x_f, d_head))
        x_pre_tp = sbm.alloc_stack((x_f, d_head), dtype=x_inp.dtype, buffer=nl.sbuf)
        nisa.dma_copy(x_pre_tp, x_inp, name=f"{sbm.get_name_prefix()}rope_x_inp_load_{name_suffix}")
        # FIXME: Add arch check
        tp_dtype = x_pre_tp.dtype
        x_tp_psum = nl.ndarray(
            (d_head, x_f), dtype=tp_dtype, buffer=nl.psum, address=None if sbm.is_auto_alloc() else (0, 0)
        )
        nisa.nc_transpose(x_tp_psum, x_pre_tp)
        nisa.tensor_copy(x, x_tp_psum)

    # Compute x * cos, the read to cos is repeated n_head times
    x_cos = sbm.alloc_stack(
        x_shape_expanded, dtype=nl.float32, buffer=nl.sbuf
    )  # expand dims for more convenient indexing
    # cos[i_d, s_active*i_B + i_S]
    cos_view = (cos).reshape_dim(1, [bs, s_active]).expand_dim(2).broadcast(2, size=n_head)
    nisa.tensor_tensor(
        dst=x_cos,
        data1=x.reshape(x_shape_expanded),
        data2=cos_view,
        op=nl.multiply,
    )
    x_cos = x_cos.reshape(x.shape)

    # Compute _rotate_half(x)
    rotated_x = sbm.alloc_stack(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(rotated_x[d_head // 2 :, :], x[: d_head // 2, :])
    nisa.tensor_scalar(rotated_x[: d_head // 2, :], x[d_head // 2 :, :], op0=nl.multiply, operand0=-1.0)

    # Compute _rotate_half(x) * sin
    rotated_x = rotated_x.reshape(x_shape_expanded)
    sin_view = (sin).reshape_dim(1, [bs, s_active]).expand_dim(2).broadcast(2, size=n_head)
    nisa.tensor_tensor(
        dst=rotated_x,
        data1=rotated_x,
        data2=sin_view,
        op=nl.multiply,
    )
    rotated_x = rotated_x.reshape(x.shape)

    # Add two intermediates
    nisa.tensor_tensor(x_embed, x_cos, rotated_x, op=nl.add)


def _rope(inv_freqs, pos_ids, bs: int, s_a: int, d_head: int, sbm: SbufManager):
    """Computes rotary embedding for current pos_ids following this algorithm:
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()

    All inputs and outputs to this function are assumed to be in sbuf.

    Args:
      inv_freqs: input ndarray, shape [par(d_head // 2), 1]
      pos_ids: input ndarray, shape [par(p_max), bs * s_active], par(p_max) is broadcasted
      bs: batch size
      s_a: active seqeunce length
      d_head: head dimension

    Returns:
      cos: output ndarray, shape [par(d_head), bs * s_active]
      sin: output ndarray, shape [par(d_head), bs * s_active]
    """
    # Most of the computation handles half of d_head at a time.
    # [d_head_half : d_head_half + d_head_half] requires d_head_half to be a multiple of 32, which means d_head is a multiple of 64
    kernel_assert(d_head % 64 == 0, f"RoPE expects head dim ({d_head}) to be divisible by 64")
    d_head_half = d_head // 2

    # Create outputs
    cos = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="name_cos")
    sin = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="sin_rope")

    # Compute freqs = dot(inv_freqs, pos_ids), can be simplified to elem-wise multiply
    emb = sbm.alloc_stack((d_head_half, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="emb_rope")
    nisa.tensor_scalar(emb, pos_ids[0:d_head_half, :], op0=nl.multiply, operand0=inv_freqs)

    # Compute ((emb + π) % 2π) - π, note that sin(θ) = sin((θ + π) % 2π - π)
    # This is to reduce emb to [-π, π] which is the restriction for Sine on ACT engine
    emb4sin = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="eb4sin_rope")
    nisa.tensor_scalar(emb4sin[0:d_head_half, :], emb, op0=nl.add, operand0=math.pi)
    _modulo(x=emb4sin, y=2.0 * math.pi, out=emb4sin, sbm=sbm)
    nisa.tensor_scalar(
        emb4sin[0:d_head_half, :],
        emb4sin[0:d_head_half, :],
        op0=nl.add,
        operand0=-math.pi,
    )

    # Compute sin = sin(torch.cat((freqs, freqs), dim=-1))
    nisa.tensor_copy(emb4sin[d_head_half : d_head_half + d_head_half, :], emb4sin[0:d_head_half, :])
    nisa.activation(sin, op=nl.sin, data=emb4sin)

    # Compute ((emb + π/2 + π) % 2π) - π, note that cos(θ) = sin((θ + π/2 + π) % 2π) - π).
    # This is to reduce emb to [-π, π] which is the legal restriction for Act Sine (and we dont have Act Cosine).
    emb4cos = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name="emb4cos_rope")
    nisa.tensor_scalar(emb4cos[0:d_head_half, :], emb, op0=nl.add, operand0=1.5 * math.pi)
    _modulo(x=emb4cos, y=2.0 * math.pi, out=emb4cos, sbm=sbm)
    nisa.tensor_scalar(
        emb4cos[0:d_head_half, :],
        emb4cos[0:d_head_half, :],
        op0=nl.add,
        operand0=-math.pi,
    )

    # Compute cos = cos(torch.cat((freqs, freqs), dim=-1))
    nisa.tensor_copy(emb4cos[d_head_half : d_head_half + d_head_half, :], emb4cos[0:d_head_half, :])
    nisa.activation(cos, op=nl.sin, data=emb4cos)

    return cos, sin


"""
Other utilities
"""


def _modulo(x, y: float, out, sbm=None):
    """Computes modulo with the following algorithm:
      q = round(x/y - 0.5)
      res = x - q * y

    All inputs and outputs to this function are assumed to be in sbuf.
    This requires both x and y to be positive.

    Args:
      x: 2D input tensor
      y: input scalar

    Returns:
      out: output sbuf tensor of the same shape as x
    """
    kernel_assert(len(x.shape) == 2, "Expect 2D input x for modulo kernel.")
    p, f = x.shape

    # Compute q = round(x/y - 0.5)
    q_f32 = sbm.alloc_stack((p, f), dtype=nl.float32, buffer=nl.sbuf)
    q_i32 = sbm.alloc_stack((p, f), dtype=nl.int32, buffer=nl.sbuf)

    nisa.tensor_scalar(q_f32, x, nl.multiply, 1.0 / y, False, nl.add, -0.5, False)
    nisa.tensor_copy(q_i32, q_f32)

    # Compute q * y
    qy = sbm.alloc_stack((p, f), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(qy, q_i32, nl.multiply, y)

    # Compute x - (q * y)
    # out = x if in_place else nl.ndarray((p, f), dtype=nl.float32, buffer=nl.sbuf, name='modulo_out')
    nisa.tensor_tensor(out, x, qy, nl.subtract)

    return out


### Sink
def _prep_sink(
    sink_hbm: nl.NkiTensor,
    result: nl.NkiTensor,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    btc: BatchTileContext,
    use_slot_geometry: bool,
):
    """
    Load the attention sink and broadcast/transpose it into ``result``.

    Input ``sink_hbm`` is ``[H, 1]`` @ HBM where ``H = kv_heads * cfg.q_head``. This helper infers
    ``kv_heads`` from the shape, broadcasts over ``s_active``, and transposes the result into
    ``[s_active_bqh_tile, n_bsq_tiles]``.
    """
    compute_bs = spp.num_slots if use_slot_geometry else atp.bs
    compute_s_active_bqh = spp.slot_s_active_bqh if use_slot_geometry else atp.s_active_bqh
    compute_s_active_bqh_remainder = (
        spp.slot_s_active_bqh_remainder if use_slot_geometry else atp.s_active_bqh_remainder
    )
    compute_n_bsq_full_tiles = spp.slot_n_bsq_full_tiles if use_slot_geometry else atp.n_bsq_full_tiles
    compute_s_active_bqh_tile = spp.slot_s_active_bqh_tile if use_slot_geometry else atp.s_active_bqh_tile

    sbm.open_scope()

    kv_heads = sink_hbm.shape[0] // cfg.q_head  # 1 for per-head; kv_heads when folded into batch

    # With batch sharding this core shards on a batch-folded range starting at btc.global_batch_offset.
    # When batch is odd, the kv_head idx may not be 0 at the sharded index.
    kv_offset = btc.global_batch_offset % kv_heads
    n_repeats = div_ceil(kv_offset + compute_bs, kv_heads)

    sink_sb = sbm.alloc_stack((1, kv_heads, cfg.q_head), dtype=sink_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        sink_sb,
        sink_hbm.reshape((1, kv_heads, cfg.q_head)),
        name=f"{sbm.get_name_prefix()}sink_load_bt{btc.batch_tile_idx}",
    )

    sink_repeated = sbm.alloc_stack(
        (1, n_repeats * kv_heads * cfg.q_head * cfg.s_active), buffer=nl.sbuf, dtype=sink_hbm.dtype
    )
    nisa.tensor_copy(
        dst=sink_repeated.reshape((1, n_repeats, kv_heads, cfg.q_head, cfg.s_active)),
        src=sink_sb.expand_dim(3).broadcast(3, size=cfg.s_active).expand_dim(1).broadcast(1, size=n_repeats),
    )

    # Slice out only the section of sink included in this nc
    sink_repeated_view = sink_repeated[:, nl.ds(kv_offset * cfg.q_head * cfg.s_active, compute_s_active_bqh)]

    for i_bsq_tile in range(compute_n_bsq_full_tiles):
        _tile_sink_transpose(
            i_bsq_tile,
            compute_s_active_bqh_tile,
            compute_s_active_bqh_tile,
            sink_hbm,
            sink_repeated_view,
            result,
            atp,
            TC,
            sbm,
        )

    if compute_s_active_bqh_remainder > 0:
        _tile_sink_transpose(
            compute_n_bsq_full_tiles,
            compute_s_active_bqh_remainder,
            compute_s_active_bqh_tile,
            sink_hbm,
            sink_repeated_view,
            result,
            atp,
            TC,
            sbm,
        )

    sbm.close_scope()


def _tile_sink_transpose(
    index,
    tile_size,
    tile_stride,
    sink_hbm,
    sink_repeated,
    sink_tp_repeated,
    atp: AttnTileParams,
    TC: TileConstants,
    sbm: SbufManager,
):
    sink_tp_psum = nl.ndarray(
        (tile_size, 1),
        buffer=nl.psum,
        dtype=sink_hbm.dtype,
        address=None
        if sbm.is_auto_alloc()
        else (
            0,
            (index % TC.psum_b_max) * TC.psum_f_max_bytes,
        ),
    )
    nisa.nc_transpose(
        sink_tp_psum,
        sink_repeated[:, nl.ds(index * tile_stride, tile_size)],
    )
    nisa.tensor_copy(sink_tp_repeated[:tile_size, index], sink_tp_psum)


def _load_packed_block_table_unresized(
    active_blk_table,
    rows: int,
    base_row: int,
    num_slots: int,
    num_folds: int,
    caller_blocks_per_row: int,
    TC: "TileConstants",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
):
    """Load the packed block table when the caller's block_len needs no reduction.

    One DMA reads the whole per-NC slice with the fold/partition split expressed in
    the access pattern; a single DVE pass then reorders fold-major to slot-major.
    """
    fold_major = sbm.alloc_stack(
        (TC.p_max, rows, num_folds, num_slots),
        dtype=active_blk_table.dtype,
        buffer=nl.sbuf,
    )
    rows_view = (active_blk_table).slice(0, base_row, base_row + rows)
    if caller_blocks_per_row < active_blk_table.shape[1]:
        rows_view = rows_view.slice(1, 0, caller_blocks_per_row)
    # [rows, num_slots, num_folds, p_max] -> [p_max, rows, num_folds, num_slots]
    src_tv = rows_view.reshape((rows, num_slots, num_folds, TC.p_max)).permute([3, 0, 2, 1])
    nisa.dma_copy(src=src_tv, dst=fold_major, name=f"{sbm.get_name_prefix()}packed_active_blk_table_load_all")

    # Reorder fold-major -> slot-major so each row slice matches the per-slot layout.
    if num_slots > 1:
        slot_major = sbm.alloc_stack(
            (TC.p_max, rows, num_slots, num_folds),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(
            dst=slot_major.permute([0, 1, 3, 2]),
            src=fold_major,
            engine=nisa.vector_engine,
        )
        bufs.active_blocks_all_sb = slot_major.reshape((TC.p_max, rows, num_slots * num_folds))
    else:
        bufs.active_blocks_all_sb = fold_major.reshape((TC.p_max, rows, num_slots * num_folds))


def _load_packed_block_table_resized(
    active_blk_table,
    rows: int,
    base_row: int,
    num_slots: int,
    num_folds: int,
    caller_blocks_per_slot: int,
    resize_factor: int,
    TC: "TileConstants",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
):
    """Load the packed block table when the kernel reduced the cache block length.

    ``_setup_block_kv_cache`` reshapes the cache to
    ``[num_blocks * resize_factor, reduced_block_len * d_head]``, so caller block ``b``
    becomes the ``resize_factor`` consecutive physical blocks ``b * resize_factor + j``,
    in token order. Each caller index is therefore expanded in place to
    ``b * resize_factor + arange(resize_factor)``.

    The expansion cannot be folded into the load's access pattern: the destination
    partition would need two affine indices (``b`` and ``j``), which the DMA cannot
    express. So the indices are expanded along the free axis and each fold is then
    transposed onto the partition axis, matching the unpacked kernel's approach in
    ``attention_tkg._load_and_reshape_active_blk_table``.

    Padded slots hold INACTIVE_BLOCK_IDX (-1) and expand to ``-resize_factor + j``,
    which stays negative for every ``j``, so out-of-bounds skipping still suppresses
    their loads.
    """
    # One (row, slot) pair per partition lane: the caller's row is [num_slots,
    # caller_blocks_per_slot], so flattening row-major keeps each slot's blocks
    # contiguous.
    flat_pairs = rows * num_slots
    expanded = sbm.alloc_stack(
        (TC.p_max, rows, num_slots, num_folds),
        dtype=active_blk_table.dtype,
        buffer=nl.sbuf,
        align=4,
    )
    pairs_view = (active_blk_table).slice(0, base_row, base_row + rows).reshape((flat_pairs, caller_blocks_per_slot))
    # Fold-inner destination view so a chunk of consecutive (row, slot) pairs is one
    # strided slice per fold. Reshaped back to [p_max, rows, num_slots * num_folds]
    # below, this is already the slot-major layout the row loop slices.
    expanded_pairs = (expanded).reshape((TC.p_max, flat_pairs, num_folds))

    sbm.open_scope()
    pair_chunk = min(flat_pairs, TC.p_max)
    for pair_offset in range(0, flat_pairs, pair_chunk):
        chunk = min(pair_chunk, flat_pairs - pair_offset)

        caller_blocks = sbm.alloc_stack((chunk, caller_blocks_per_slot), dtype=active_blk_table.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=caller_blocks,
            src=(pairs_view).slice(0, pair_offset, pair_offset + chunk),
            name=f"{sbm.get_name_prefix()}packed_active_blk_table_load_caller_p{pair_offset}",
        )

        sub_block_offsets = sbm.alloc_stack((chunk, resize_factor), dtype=active_blk_table.dtype, buffer=nl.sbuf)
        nisa.iota(dst=sub_block_offsets, pattern=[[1, resize_factor]], offset=0)

        # physical = caller * resize_factor + j, for j in [0, resize_factor)
        physical = sbm.alloc_stack((chunk, caller_blocks_per_slot, resize_factor), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=physical,
            data=(caller_blocks).expand_dim(2).broadcast(2, size=resize_factor),
            op0=nl.multiply,
            operand0=float(resize_factor),
            op1=nl.add,
            operand1=(sub_block_offsets).expand_dim(1).broadcast(1, size=caller_blocks_per_slot),
        )
        physical = physical.reshape((chunk, num_folds * TC.p_max))

        # Move each fold's p_max sub-blocks onto the partition axis.
        sbm.open_scope()
        for fold_idx in range(num_folds):
            fold_psum = nl.ndarray(
                (TC.p_max, chunk),
                dtype=physical.dtype,
                buffer=nl.psum,
                address=None if sbm.is_auto_alloc() else (0, (fold_idx % TC.psum_b_max) * TC.psum_f_max_bytes),
            )
            nisa.nc_transpose(fold_psum, physical[:, nl.ds(fold_idx * TC.p_max, TC.p_max)])
            nisa.tensor_copy(
                dst=(expanded_pairs).slice(1, pair_offset, pair_offset + chunk).select(2, fold_idx),
                src=fold_psum,
                engine=nisa.vector_engine,
            )
        sbm.close_scope()
    sbm.close_scope()

    bufs.active_blocks_all_sb = expanded.reshape((TC.p_max, rows, num_slots * num_folds))


def _preload_packed_accumulator_routes(
    partial_route_table: nl.NkiTensor,
    group_state_route_table: nl.NkiTensor,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
) -> None:
    """Load this NC's preformatted accumulator Tensor Indirection routes.

    The caller stores rows globally as ``[NC0 rows][NC1 rows]`` with shape
    ``[total_rows, d_head, index_cols]``. This preload slices the current NC and
    transposes both tables to ``[d_head, local_rows, index_cols]``. Individual
    iterations then copy a dense row locally instead of issuing small HBM DMAs
    on the attention critical path.
    """
    rows = spp.num_iters_per_nc
    base_row = spp.schedule_row_base
    index_cols = div_ceil(spp.num_slots * atp.s_active_qh, 16)

    bufs.accumulator_partial_route_all_sb = sbm.alloc_stack(
        (cfg.d_head, rows, index_cols),
        dtype=nl.uint16,
        buffer=nl.sbuf,
    )
    bufs.accumulator_group_state_route_all_sb = sbm.alloc_stack(
        (cfg.d_head, rows, index_cols),
        dtype=nl.uint16,
        buffer=nl.sbuf,
    )

    nisa.dma_copy(
        dst=bufs.accumulator_partial_route_all_sb,
        src=partial_route_table[
            base_row : base_row + rows,
            : cfg.d_head,
            :index_cols,
        ].permute([1, 0, 2]),
        name=f"{sbm.get_name_prefix()}packed_accumulator_partial_routes_load_all",
    )
    nisa.dma_copy(
        dst=bufs.accumulator_group_state_route_all_sb,
        src=group_state_route_table[
            base_row : base_row + rows,
            : cfg.d_head,
            :index_cols,
        ].permute([1, 0, 2]),
        name=f"{sbm.get_name_prefix()}packed_accumulator_group_state_routes_load_all",
    )


def _preload_packed_row_metadata(
    active_blk_table,
    q_index_table,
    atp: "AttnTileParams",
    spp: "SequencePackingParams",
    cfg: "AttnTKGConfig",
    TC: "TileConstants",
    sbm: "SbufManager",
    bufs: "AttnInternalBuffers",
):
    """Load this NC's whole per-row metadata once, before the schedule-row loop.

    The block and Q-index tables are fetched here; the caller similarly hoists
    sequence IDs and optional accumulator routes. Loading any of this metadata
    per row puts a small DMA plus a reorder on the critical path immediately
    ahead of the indirect K/V load, preventing the K/V transfers from being
    pipelined as in the unpacked kernel. Hoisting removes that serialization;
    rows then slice these buffers for free.
    """
    rows = spp.num_iters_per_nc
    base_row = spp.schedule_row_base
    num_slots = spp.num_slots
    resize_factor = atp.blk_cache_resize_factor
    fold_s_prior = atp.block_len * TC.p_max
    num_folds = atp.fa_tile_s_prior // fold_s_prior
    num_blocks_per_slot = num_folds * TC.p_max
    # The caller's table is in its own block_len units. When the kernel reduces
    # block_len (see resize_cache_block_len_for_attention_tkg_kernel), each caller
    # block covers resize_factor physical blocks of the reshaped cache, so a slot's
    # row is resize_factor times narrower than the folds it feeds.
    caller_blocks_per_slot = num_blocks_per_slot // resize_factor
    caller_blocks_per_row = num_slots * caller_blocks_per_slot
    kernel_assert(
        caller_blocks_per_row <= active_blk_table.shape[1],
        "Packed block-table row is too short for this FA iteration.",
    )

    # --- packed block table -> [p_max, rows, num_slots * num_folds] (slot-major)
    if resize_factor == 1:
        _load_packed_block_table_unresized(
            active_blk_table, rows, base_row, num_slots, num_folds, caller_blocks_per_row, TC, sbm, bufs
        )
    else:
        _load_packed_block_table_resized(
            active_blk_table,
            rows,
            base_row,
            num_slots,
            num_folds,
            caller_blocks_per_slot,
            resize_factor,
            TC,
            sbm,
            bufs,
        )

    if atp.use_dma_transpose or not atp.use_v_dma_skipping:
        u32 = sbm.alloc_stack((TC.p_max, rows, num_slots * num_folds), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(u32, bufs.active_blocks_all_sb, engine=nisa.vector_engine)
        bufs.active_blocks_all_sb_u32 = u32

    # --- Q Tensor-Indirection indices: one [d_head, cols] block per row. Loaded
    # in bulk here; each row copies its slice into a dense buffer, because a TI
    # index tensor must own dense storage strides and a row slice of this buffer
    # does not. The copy is a cheap DVE op and keeps small DMAs out of the queues
    # the indirect K/V loads use.
    cols_q = div_ceil(spp.slot_s_active_bqh, 16)
    bufs.q_index_all_sb = sbm.alloc_stack((cfg.d_head, rows, cols_q), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bufs.q_index_all_sb,
        src=(q_index_table)
        .slice(0, base_row, base_row + rows)
        .slice(1, 0, cfg.d_head)
        .slice(2, 0, cols_q)
        .permute([1, 0, 2]),
        name=f"{sbm.get_name_prefix()}packed_q_index_load_all",
    )


def _load_and_reshape_active_blk_table(
    active_blk_table,
    atp: AttnTileParams,
    spp: SequencePackingParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Point this row at its slice of the per-NC block table loaded up front."""
    TC = TileConstants.get_tile_constants()
    fold_s_prior = atp.block_len * TC.p_max
    atp.num_folds_per_batch = fa_ctx.tile_s_prior // fold_s_prior
    row_in_nc = fa_ctx.schedule_row - spp.schedule_row_base
    bufs.active_blocks_sb = (bufs.active_blocks_all_sb).select(1, row_in_nc)
    if bufs.active_blocks_all_sb_u32 != None:
        bufs.active_blocks_sb_u32 = (bufs.active_blocks_all_sb_u32).select(1, row_in_nc)


### DMA Batching Helpers


def _compute_dma_batch_params(
    num_folds: int,
    bs: int,
    granularity: int,
    cap: int,
    sbm: SbufManager,
    batching_cap: int = 64,
) -> Tuple[int, int]:
    """Compute fold-batching and batch-batching parameters for DMA calls.

    Two-step algorithm:
      1. Find largest N dividing num_folds with N * granularity <= cap.
      2. If all folds are batched (N == num_folds) and bs > 1, find largest M
         dividing bs with N * M * granularity <= cap.

    Only activates when sbm uses auto allocation (manual alloc calculations not updated for batching).
    Returns (batch_n_folds, batch_n_batches).

    ``batching_cap`` limits the total number of folds times batches represented
    by one DMA instruction. This avoids exhausting the descriptor budget when
    cache-block resizing makes ``granularity`` very small.
    """
    if not sbm.is_auto_alloc():
        return 1, 1
    batch_n_folds = 1
    for n in range(num_folds, 0, -1):
        if num_folds % n == 0 and n <= batching_cap and n * granularity <= cap:
            batch_n_folds = n
            break
    batch_n_batches = 1
    if batch_n_folds == num_folds and bs > 1:
        for m in range(bs, 0, -1):
            if bs % m == 0 and batch_n_folds * m <= batching_cap and batch_n_folds * m * granularity <= cap:
                batch_n_batches = m
                break
    return batch_n_folds, batch_n_batches


def _rearrange_indices_for_batched_dma(active_blocks_sb_u32, total_columns, group_size, TC, sbm):
    """Rearrange block index table for batched dma_copy 2D vector_offset column-major semantics.

    dma_copy with a [128, N] vector_offset reads indices in a "snake" pattern (column-major):
    all 128 partitions' element 0, then all 128 partitions' element 1, etc. To ensure each
    partition gets its own N fold indices correctly, the SBUF index table must be pre-arranged
    into this snake layout.

    Approach: write SBUF to HBM with groups deinterleaved (AP-based), then dma_transpose reads
    each group's block column-major back into SBUF — producing the snake layout in one efficient
    DMA transfer (vs. the old 2-dma_copy approach which used a strided AP for the read-back).

    Example (P=8, N=4, num_groups=2, values shown as partition*100 + group*10 + fold):

      SBUF original [8, 8] (group-interleaved):
        p0: [  0,  1,  2,  3, 10, 11, 12, 13]   ← group0 folds | group1 folds
        p1: [100,101,102,103,110,111,112,113]
        ...

      After step 1 (AP deinterleave to HBM [num_groups, P, N] = [2, 8, 4]):
        Group 0: p0=[0,1,2,3], p1=[100,101,102,103], ..., p7=[700,701,702,703]
        Group 1: p0=[10,11,12,13], p1=[110,111,112,113], ...

      HBM reshaped as [num_groups*N=8, P=8] for dma_transpose:
        row 0 (g0,f0): [  0,  1,  2,  3,100,101,102,103]
        row 1 (g0,f1): [200,201,202,203,300,301,302,303]
        ...

      After step 2 (dma_transpose [8,8] → [8,8], column-major read):
        p0: [  0,200,400,600, 10,210,410,610]   ← group0 snake | group1 snake
        p1: [  1,201,401,601, 11,211,411,611]
        ...

    Args:
        active_blocks_sb_u32: source index table [128, total_columns] in SBUF (batch-major order)
        total_columns: total index columns (= bs * num_folds_per_batch)
        group_size: columns per batched DMA call (= v_dma_batch_size)
        TC: TileConstants (for TC.p_max = 128)
        sbm: SbufManager for output allocation

    Returns:
        Rearranged index table [128, total_columns] in SBUF with snake layout per group.
    """
    num_groups = total_columns // group_size
    N = group_size

    # Step 1: dma_copy deinterleaves groups from SBUF to HBM.
    # SBUF layout: [128, num_groups, N] (groups interleaved per partition).
    # HBM layout: [num_groups, P, N] — each group's [P, N] block contiguous, partition-outer within group.
    # Flat offset: hbm[g*P*N + p*N + n] = sbuf[p, g*N + n].
    # AP strides: partition stride=N, group stride=P*N, fold stride=1.
    v_idx_hbm = nl.ndarray((num_groups * TC.p_max * N,), dtype=nl.uint32, buffer=nl.private_hbm)
    nisa.dma_copy(
        dst=v_idx_hbm.ap(
            [
                [N, TC.p_max],  # partition p: stride N, count P
                [TC.p_max * N, num_groups],  # group g: stride P*N, count num_groups
                [1, N],  # fold n: stride 1, count N
            ],
            offset=0,
        ),
        src=(active_blocks_sb_u32).reshape_dim(1, [num_groups, N]),
    )
    # Step 2: reshape HBM as [num_groups*N, 128] and dma_transpose (1,0) → SBUF [128, num_groups*N].
    # The reshape interprets the flat HBM as num_groups*N rows of 128 columns each.
    # Group 0's N rows come first, then group 1's N rows, etc.
    # dma_transpose reads column-major: column 0 of all rows → partition 0's free-dim values.
    # Since groups occupy separate row blocks, the result is naturally group-outer in SBUF.
    idx_table = sbm.alloc_stack((TC.p_max, total_columns), dtype=nl.uint32, buffer=nl.sbuf, align=32)
    nisa.dma_transpose(
        dst=idx_table,
        src=v_idx_hbm.reshape((num_groups * N, TC.p_max)),
        axes=(1, 0),
    )
    return idx_table


def _k_tile_physical_index(logical_tile_idx, k_block_len_row_tiled, k_dma_batch_n_folds, k_dma_batch_n_batches, i_b):
    """Map logical tile index to physical tile position in k_sb, accounting for kbld-outer layout.

    When DMA batching is active, the block table is batch-major so index columns in k_sb are ordered
    as [b0_f0, b0_f1, ..., b1_f0, b1_f1, ...] within each position-in-fold group.
    Without batching, physical == logical.
    """
    k_dma_batch_size = k_dma_batch_n_folds * k_dma_batch_n_batches
    if k_dma_batch_size > 1:
        fold = logical_tile_idx // k_block_len_row_tiled
        pos_in_fold = logical_tile_idx % k_block_len_row_tiled
        group = fold // k_dma_batch_n_folds
        fold_in_group = fold % k_dma_batch_n_folds
        batch_in_group = i_b % k_dma_batch_n_batches
        # Block table is batch-major: ic = batch_in_group * n_folds + fold_in_group
        # group offset accounts for multiple fold-groups within a single k_sb
        group_offset = group * k_block_len_row_tiled * k_dma_batch_size
        return group_offset + pos_in_fold * k_dma_batch_size + batch_in_group * k_dma_batch_n_folds + fold_in_group
    else:
        return logical_tile_idx


### Other Helpers
def _get_safe_batch_interleave_degree(space_needed_per_batch: int, max_batch_interleave_degree: int, sbm: SbufManager):
    """
    Compute the batch interleave degree that will not overflow memory given the current memory available.
    space_needed_per_batch is in bytes (interleaved across batches).

    For auto_alloc sbm, return 1 (interleave degree is ignored in this case).
    """
    if sbm.is_auto_alloc():
        return 1
    # use 1 if auto alloc since otherwise it can fail despite enough space available
    space_available = sbm.get_free_space()
    kernel_assert(max_batch_interleave_degree > 0, "batch_interleave_degree must be greater than 0")

    result = min(max_batch_interleave_degree, space_available // space_needed_per_batch)

    kernel_assert(
        result > 0,
        (
            f"Insufficient memory to run batch loop, even at interleave_degree=1."
            f"Need {space_needed_per_batch} bytes, but only have {space_available} bytes available."
        ),
    )

    return result


# Extended instructions require input/output tensors have multiple of 16 partitions
# This is temporary, will go away once gpsimd sb2sb moves from extended isa to its final isa
def pad_partitions_for_ext_inst(partitions):
    PARTITIONS_PER_GPSIMD_CORE = 16
    return (partitions + PARTITIONS_PER_GPSIMD_CORE - 1) // PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE


def _clamp_max_to_finite(dst: nl.NkiTensor, src: nl.NkiTensor, max_negated: bool = False):
    """Clamp infinite max values to finite bounds to prevent NaN in exp(a - b).

    Fully masked tiles produce -Inf (or +Inf when negated) as the softmax max.
    When computing exp(prev_max - curr_max), -Inf - (-Inf) = NaN per IEEE 754.
    Clamping to a finite bound ensures exp produces 0 instead.

    When max_negated=False: -Inf -> _MIN_FLOAT32 (clamp up via maximum)
    When max_negated=True:  +Inf -> _MAX_FLOAT32 (clamp down via minimum)
    """
    op, bound = (nl.minimum, _MAX_FLOAT32) if max_negated else (nl.maximum, _MIN_FLOAT32)
    nisa.tensor_scalar(dst, src, op0=op, operand0=bound)
