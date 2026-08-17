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
import os as _os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.isa import dge_mode, dma_engine, oob_mode, reduce_cmd

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import DtypeMode
from ..utils.cross_partition_copy import cross_partition_copy
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, is_trn3_b1, resolve_fp8_e4m3_dtype
from ..utils.stream_shuffle_broadcast import (
    stream_shuffle_broadcast,
)
from .attention_tkg_utils import (
    AttnTKGConfig,
    TileConstants,
    is_batch_sharded,
    is_fp8_e4m3,
    is_fp8_e5m2,
    is_qk_swapped,
    is_s_prior_sharded,
    resize_cache_block_len_for_attention_tkg_kernel,
    uses_batch_tiling,
    uses_flash_attention,
)
from .gen_mask_tkg import gen_mask_tkg

_MAX_D_HEAD = 512
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


@nki.jit
def attention_tkg(
    q: nl.NkiTensor,
    k_active: nl.NkiTensor,
    v_active: nl.NkiTensor,
    k_prior: nl.NkiTensor,
    v_prior: nl.NkiTensor,
    mask: nl.NkiTensor,
    out: nl.NkiTensor,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    inv_freqs: Optional[nl.NkiTensor] = None,
    rope_pos_ids: Optional[nl.NkiTensor] = None,
    start_pos_ids: Optional[nl.NkiTensor] = None,
    sink: Optional[nl.NkiTensor] = None,
    active_blocks_table: Optional[nl.NkiTensor] = None,
    k_out: Optional[nl.NkiTensor] = None,
    cp_softmax_stats_out: Optional[dict] = None,
    DBG_TENSORS: Optional[tuple] = None,
    max_context_len: Optional[nl.NkiTensor] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> Tuple[nl.NkiTensor, Optional[nl.NkiTensor]]:
    """Attention specifically optimized for token-gen (where s_active is small). Can optionally fuse RoPE at the start.
    Please refer to attention_tkg_torch.attention_tkg_torch for an equivalent torch implementation.

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
      k_prior: Prior key tensor from KV cache. Shape [B+, 1, s_prior, d] if cfg.tp_k_prior else [B+, 1, d, s_prior].
               For block KV cache, shape is [B+ * block_count, block_len, d] (indexing: [b * block_count + blk, block_len, d])
               For block KV cache with fp8_packed, shape is [B+ * block_count, block_len // 2, d, 2] fp8
      v_prior: Prior value tensor from KV cache. Shape [B+, 1, s_prior, d].
               For block KV cache, shape is [B+ * block_count, block_len, d] (indexing: [b * block_count + blk, block_len, d])
      mask: Attention mask. Shape depends on cfg.use_pos_id and the QK-swap decision:
              - cfg.use_pos_id=True: the active mask [s_active, B, H, s_active].
              - cfg.use_pos_id=False (pre-generated full mask): the layout MUST match the QK-swap
                decision. Call is_qk_swapped(...) (see attention_tkg_utils) first to determine which
                path this config takes, then supply the matching layout:
                  * default (K-stationary): [s_prior, B, H, s_active]
                  * QK-swap (Q-stationary): [B, H, s_active, s_prior]
                gen_mask_tkg_hbm produces the correct layout when given the same transposed_out flag.
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
      active_blocks_table: Table of active blocks for block KV cache. Shape [B, num_blocks].
                          Required when using block KV cache
      k_out: Output key tensor after RoPE. Populated when cfg.fuse_rope is True, stores k_active after applying RoPE
        Shape: if cfg.k_out_in_sb:
                [d, B * s_active] (indexing [d, b * s_active + s])
              else:
                [B, 1, d, s_active].
      cp_softmax_stats_out: Output dict for exporting local softmax statistics.
                        Used by attention_block_tkg kernel with context parallel (CP > 1) for
                        distributed softmax correction. Caller provides pre-allocated tensors;
                        the kernel writes results into them.
                        Caller-provided keys:
                        - "fa_running_max" (nl.ndarray): Local softmax max.
                          Shape [s_active_bqh_tile, n_bsq_tiles].
                        - "fa_running_sum" (nl.ndarray): Local softmax sum of exp values.
                          Shape [s_active_bqh_tile, n_bsq_tiles].
                        Kernel-added keys:
                        - "max_negated" (bool): Whether max values are negated. Caller must pass
                          this to the CP correction to select the correct reduction op.
                        - "atp" (AttnTileParams): Tile parameters needed for broadcasting stats.
                        - "TC" (TileConstants): Tile constants needed for broadcasting stats.
                        Only populated when cfg.return_cp_softmax_stats=True.
      DBG_TENSORS: Optional tuple of 4-5 debug tensors with shared HBM type for intermediate value inspection.
                  Expects:
                    - QK: Result of Q@K^T.
                    - QK_MAX: Result of max reduction of QK.
                    - QK_EXP: Result of exp(QK).
                    - EXP_SUM: Result of sum(exp(QK)).
                    - ACTIVE_TABLE: (only use with block KV) Result after loading the active blocks table.
                  See implementation for shapes of these tensors.
      max_context_len: Optional scalar tensor for dynamic FA early exit. Shape [1], dtype int32.
                      When provided, the FA loop exits early after processing ceil(max_context_len / tile_size)
                      tiles instead of all tiles. Requires block KV cache and use_pos_id=True.
      dtype_mode: Quantization dtype policy for the SBUF K/V tile allocations.
                  When ``k_prior.dtype`` / ``v_prior.dtype`` is concrete
                  (``nl.float8_e4m3`` or ``nl.float8_e4m3fn``), the kernel uses
                  it directly. When the caller leaves the dtype as the opaque
                  ``"float8e4"`` sentinel, ``dtype_mode`` selects the variant:
                  ``NON_OCP`` → ``nl.float8_e4m3``, ``OCP`` → ``nl.float8_e4m3fn``,
                  ``AUTO`` → ``nl.float8_e4m3fn`` on TRN3 else ``nl.float8_e4m3``.
                  Compiler enforces a single E4M3 variant per traced module
                  (``EOCP001``); pick one variant for the whole call graph.

    Returns:
      out: Attention output tensor.
        Shape: if cfg.out_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, H, d, s_active]
        When cp_softmax_stats_out is provided, out is unnormalized
        (not divided by softmax_sum). Caller handles global normalization after CP correction.
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

    2. Adaptive LNC2 Sharding:
      - Automatically selects sharding strategy based on tensor dimensions
      - Batch sharding: Used when batch is even AND (s_prior < 256 OR b*q_head*s_active > 128)
      - Sequence sharding: Used when s_prior >= 256 and batch sharding criteria not met
      - Balances computation across 2 NeuronCores for improved throughput

    3. Mask Generation:
      - use_pos_id=False: Pre-generated mask loaded from HBM
      - use_pos_id=True: Mask generated in-kernel from position IDs
      - In-kernel generation reduces memory bandwidth but requires position ID input

    4. Fused RoPE (Rotary Position Embedding):
      - fuse_rope integrates RoPE computation directly into the attention kernel
      - Applies rotary embeddings to Q and K tensors, scaling Q by 1/sqrt(d_head)
      - Reduces memory traffic by avoiding separate RoPE passes

    5. Block KV Cache:
      - Supports block-sparse KV cache with configurable block_len
      - Uses active_blocks_table to track which cache blocks are active per batch
      - Enables efficient long-context inference with sparse memory access patterns

    6. K_prior Transpose Handling:
      - tp_k_prior flag indicates whether the kernel needs to transpose K_prior during load
      - Flat KV:
        - True: K_prior is [B, 1, s_prior, d] in HBM, kernel transposes to [d, s_prior] in SBUF
        - False: K_prior is [B, 1, d, s_prior] in HBM, kernel loads directly (already transposed)
      - Block KV: must be True. K_prior is [num_blocks, block_len, d], kernel always transposes during block loading

    7. Strided Memory Access (strided_mm1):
      - Enables strided read patterns for K in first matmul
      - When enabled, allows MM2 to use sequential V reads for better DMA throughput
      - Trades off MM1 memory access for MM2 optimization

    8. Attention Sink:
      - Supports streaming attention with sink tokens for infinite context
      - Sink tokens maintain fixed attention scores across all positions
      - Integrated into softmax reduction for minimal overhead

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
      Use with s_active <= 7, and with d_head <= 512.

    Notes:
        - KV cache can have batch size larger than B (denoted B+) for garbage data buffering
        - Q is scaled with 1/sqrt(d_head) only when cfg.fuse_rope is True
        - Block KV cache requires qk_in_sb=True
        - LNC2 sharding automatically selected based on tensor dimensions
        - Extended GPSIMD instructions require 16-partition alignment

    SBUF Internal Layouts (d-tiling, d_head > 128):
        When d_head > p_max (128), the head dimension is tiled into n_d_tiles = ceil(d_head/128) chunks.
        Key SBUF buffers use a packed layout with d-tiles concatenated along the free dimension:

        k_sb:  [d_tile_size, n_d_tiles * s_prior_tile]
               Each d-tile's K data occupies [d_tile_size, s_prior_tile] at offset i_d * s_prior_tile.
        qk:    [s_prior_tile, s_active_bqh]
               QK^T accumulates across d-tiles in PSUM; final shape is independent of n_d_tiles.
        exp_v (res): [d_tile_size, n_d_tiles * s_active_bqh]
               PV matmul output per d-tile at offset i_d * s_active_bqh.
        out (if out_in_sb): [d_tile_size, n_d_tiles * bs_full * s_active_qh]
               Final output in tiled layout; each d-tile at offset i_d * bs_full * s_active_qh.
        out (if HBM): [B, H, d_head, s_active]
               Stored per d-tile via strided DMA: slice(2, i_d*d_tile_size, (i_d+1)*d_tile_size).

        When d_head <= 128 (n_d_tiles == 1), all offsets are 0 and shapes collapse to the non-tiled case.

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

    TC = TileConstants.get_tile_constants()
    atp = _compute_tile_params(cfg, TC, q, k_prior, v_prior, k_active, v_active, active_blocks_table, dtype_mode)
    # Initialize batch-dependent fields with full per-NC batch (before any batch tiling)
    # This is used to set up the debug tensor shapes.
    _update_atp_for_batch_tile(atp, atp.bs_per_nc, TC, cfg)
    bufs = AttnInternalBuffers()

    # Validate cp_softmax_stats_out when return_cp_softmax_stats is enabled
    if cfg.return_cp_softmax_stats:
        kernel_assert(
            cp_softmax_stats_out is not None, "cp_softmax_stats_out dict is required when return_cp_softmax_stats=True"
        )

    if atp.is_block_kv:
        _setup_block_kv_cache(
            k_prior,
            v_prior,
            k_active,
            v_active,
            active_blocks_table,
            atp,
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

    # Row-tiling needs Q replicated across all partition halves; both MM1 paths then read bufs.q_sb.
    _replicate_q_for_row_tiling(bufs, atp, cfg, TC, sbm)

    # Compute batch tiling parameters
    _, batch_tile_size = uses_batch_tiling(
        atp.bs_per_nc,
        cfg.q_head,
        cfg.s_active,
        atp.fa_tile_s_prior,
        sbm.is_auto_alloc(),
        dtype_size=sizeinbytes(atp.io_type),
        qk_swapped=atp.qk_swapped,
    )
    num_batch_tiles = div_ceil(atp.bs_per_nc, batch_tile_size)

    # Pre-compute dynamic FA trip count (outside batch loop — doesn't depend on batch tile)
    num_non_last_tiles_reg = None
    if max_context_len is not None and atp.use_fa:
        kernel_assert(
            atp.is_block_kv,
            "Dynamic FA early exit (max_context_len) requires block KV cache. Flat KV is not supported.",
        )
        kernel_assert(
            cfg.use_pos_id,
            "Dynamic FA early exit (max_context_len) requires use_pos_id=True for in-kernel mask generation.",
        )
        kernel_assert(
            atp.s_prior % atp.fa_tile_s_prior == 0,
            f"Dynamic FA early exit (max_context_len) requires s_prior ({atp.s_prior}) to be a multiple of "
            f"fa_tile_size ({atp.fa_tile_s_prior}).",
        )
        max_ctx_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=max_ctx_sbuf, src=max_context_len)
        if atp.interleaved_fa_tiles:
            nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.right_shift, operand0=1)
        fa_tile_shift = int(math.log2(atp.fa_tile_s_prior))
        # Compute: num_non_last_tiles = max(ceil(mcl / tile_size) - 1, 0)
        #   Step 1: mcl += tile_size - 1          (prepare for integer ceil division)
        #   Step 2: mcl >>= log2(tile_size)       (= ceil(mcl / tile_size))
        #   Step 3: mcl = max(mcl - 1, 0)         (subtract 1 for non-last count, clamp to 0)
        nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.add, operand0=atp.fa_tile_s_prior - 1)
        nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.right_shift, operand0=fa_tile_shift)
        nisa.tensor_scalar(dst=max_ctx_sbuf, data=max_ctx_sbuf, op0=nl.add, operand0=-1, op1=nl.maximum, operand1=0)
        num_non_last_tiles_reg = nisa.register_alloc()
        nisa.register_load(num_non_last_tiles_reg, max_ctx_sbuf)

    # Batch outer loop - tiles the batch dimension to fit within SBUF memory budget
    # When batch_tile_size == atp.bs_per_nc, num_batch_tiles=1 so this is a single iteration
    for batch_tile_idx in range(num_batch_tiles):
        tile_batch_offset = batch_tile_idx * batch_tile_size
        tile_bs = min(batch_tile_size, atp.bs_per_nc - tile_batch_offset)
        btc = BatchTileContext(
            batch_tile_idx=batch_tile_idx,
            tile_bs=tile_bs,
            tile_batch_offset=tile_batch_offset,
            global_batch_offset=atp.bs_prg_id * atp.bs_per_nc + tile_batch_offset,
        )
        _update_atp_for_batch_tile(atp, tile_bs, TC, cfg)

        # Open scope for this batch tile's buffers (FA running buffers, etc.)
        sbm.open_scope()

        if atp.use_online_softmax:
            _allocate_online_softmax_buffers(atp, cfg, sbm, bufs, is_dynamic=(num_non_last_tiles_reg is not None))

        # Flash Attention loop - iterates over tiles of s_prior
        # When use_fa=False, num_fa_tiles=1 so this is a single iteration
        if num_non_last_tiles_reg is not None:
            # Dynamic FA early exit: split into N-1 dynamic iterations + 1 static last tile.
            # Trip count pre-computed in num_non_last_tiles_reg above.

            # Reset tile offset counters for this batch tile
            # Interleaved LNC2 (sprior-sharded): NC0 starts at offset 0, NC1 at tile_size.
            # When batch-sharded, interleaved_fa_tiles=False so nc_start_offset=0.
            nc_start_offset = atp.sprior_prg_id * atp.fa_tile_s_prior if atp.interleaved_fa_tiles else 0
            dynamic_tile_offset_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(dynamic_tile_offset_sbuf, value=nc_start_offset)
            TC = TileConstants.get_tile_constants()
            dynamic_tile_offset_f32 = nl.ndarray((TC.p_max, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dynamic_tile_offset_f32, value=float(nc_start_offset))

            def _process_non_last_tile(_):
                fa_ctx = FATileContext(
                    fa_tile_idx=0,  # placeholder, not used for branching in dynamic path
                    tile_s_prior=atp.fa_tile_s_prior,
                    tile_n_sprior=atp.fa_n_sprior_tile,
                    tile_offset=0,  # placeholder, dynamic_tile_offset_sbuf used instead
                    is_last_fa_tile=False,
                    dynamic_tile_offset_sbuf=dynamic_tile_offset_sbuf,
                    dynamic_tile_offset_f32=dynamic_tile_offset_f32,
                )

                sbm.open_scope()
                _execute_fa_tile_body(
                    active_blocks_table,
                    mask,
                    k_prior,
                    v_prior,
                    v_active,
                    out,
                    sink,
                    DBG_TENSORS,
                    atp,
                    cfg,
                    TC,
                    sbm,
                    bufs,
                    fa_ctx,
                    btc,
                )
                sbm.close_scope()

                # Increment tile offset (2 counters: int32 for DMA, float32 for iota)
                # Interleaved LNC2: stride by 2 tiles (skip the other NC's tile)
                tile_stride = atp.fa_tile_s_prior * atp.sprior_n_prgs
                nisa.tensor_scalar(
                    dst=dynamic_tile_offset_sbuf,
                    data=dynamic_tile_offset_sbuf,
                    op0=nl.add,
                    operand0=tile_stride,
                )
                nisa.tensor_scalar(
                    dst=dynamic_tile_offset_f32,
                    data=dynamic_tile_offset_f32,
                    op0=nl.add,
                    operand0=float(tile_stride),
                )

            nl.fori_loop(0, num_non_last_tiles_reg, _process_non_last_tile)

            # Static last tile
            last_fa_ctx = _compute_fa_tile_context(atp.num_fa_tiles - 1, atp, TC)
            last_fa_ctx.dynamic_tile_offset_sbuf = dynamic_tile_offset_sbuf
            last_fa_ctx.dynamic_tile_offset_f32 = dynamic_tile_offset_f32

            sbm.open_scope()
            _execute_fa_tile_body(
                active_blocks_table,
                mask,
                k_prior,
                v_prior,
                v_active,
                out,
                sink,
                DBG_TENSORS,
                atp,
                cfg,
                TC,
                sbm,
                bufs,
                last_fa_ctx,
                btc,
            )
            sbm.close_scope()
        else:
            # Original static FA loop
            for fa_tile_idx in range(atp.num_fa_tiles):
                fa_ctx = _compute_fa_tile_context(fa_tile_idx, atp, TC)

                sbm.open_scope()
                _execute_fa_tile_body(
                    active_blocks_table,
                    mask,
                    k_prior,
                    v_prior,
                    v_active,
                    out,
                    sink,
                    DBG_TENSORS,
                    atp,
                    cfg,
                    TC,
                    sbm,
                    bufs,
                    fa_ctx,
                    btc,
                )
                sbm.close_scope()

        # Final normalization and store for the online-softmax path (FA or sharded non-FA)
        if atp.use_online_softmax:
            _finalize_and_store(
                sink,
                out,
                atp,
                cfg,
                TC,
                sbm,
                bufs,
                btc,
                cp_softmax_stats_out=cp_softmax_stats_out,
                DBG_TENSORS=DBG_TENSORS,
            )

        # Close scope for this batch tile's buffers
        sbm.close_scope()

    return out, k_out


def _execute_fa_tile_body(
    active_blocks_table,
    mask,
    k_prior,
    v_prior,
    v_active,
    out,
    sink,
    DBG_TENSORS,
    atp,
    cfg,
    TC,
    sbm,
    bufs,
    fa_ctx,
    btc,
):
    """Execute one FA tile iteration (the function calls inside the FA loop)."""
    # Load active blocks table for this FA tile and batch tile (block KV only)
    if atp.is_block_kv:
        # The swap path's MM1 K-load also needs a fold-major index table (additionally_emit_fold_major) alongside
        # the batch-major one the V-load consumes. Unswapped uses batch-major only.
        _load_and_reshape_active_blk_table(
            active_blocks_table, atp, sbm, bufs, btc, fa_ctx, additionally_emit_fold_major=atp.qk_swapped
        )
    # Allocate QK and mask buffers
    _allocate_qk_buffers(atp, TC, sbm, bufs, fa_ctx)
    # Load mask for this FA tile
    _load_mask(mask, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)
    if atp.qk_swapped:
        _compute_kq_matmul_and_max_swapped(k_prior, sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
        # Step 2. Fold sink into the fused per-position max and update the FA running max
        _fold_sink_and_update_max_swapped(sink, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
        # Step 3+4. Exp(KQ^T - max(KQ^T)) with fused sum reduction, then transpose for PV
        _compute_exp_sum_and_transpose_swapped(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    else:
        _compute_qk_matmul(k_prior, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
        # Step 2. Cascaded max reduce of KQ^T (includes FA running max update)
        _cascaded_max_reduce(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
        # Step 3. Exp(KQ^T - max(KQ^T))
        _compute_exp_qk(DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
        # Step 4. Cascaded sum reduction of exp
        _cascaded_sum_reduction(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)
    # Step 5. Matmult 2 of (exp @ V)^T and store output
    _compute_pv_matmul_and_store(v_prior, v_active, out, atp, cfg, TC, sbm, bufs, fa_ctx, btc)


OOB_MODE_SKIP = nisa.oob_mode.skip  # FIXME: needs to be instantiated externally from kernel


def _copy_and_export_softmax_stats(
    cp_softmax_stats_out: dict,
    max_src: nl.ndarray,
    sum_src: nl.ndarray,
    max_negated: bool,
    atp: 'AttnTileParams',
    TC: 'TileConstants',
):
    """Export softmax stats (max, sum) to caller-provided tensors for CP distributed correction.

    Copies stats in tile layout [s_active_bqh_tile, n_bsq_tiles] directly
    to the output tensors without broadcasting across d_head. The CP correction
    broadcasts after the all_gather to minimize collective data size.

    Args:
        cp_softmax_stats_out: Dict with "fa_running_max" and "fa_running_sum" output tensors.
            "max_negated", "atp", "TC" are added by this function.
        max_src: Source max buffer [s_active_bqh_tile, n_bsq_tiles].
        sum_src: Source sum buffer [s_active_bqh_tile, n_bsq_tiles].
        max_negated: Whether max values are negated.
        atp: Attention tile parameters.
        TC: Tile constants.
    """
    cp_softmax_stats_out["max_negated"] = max_negated
    cp_softmax_stats_out["atp"] = atp
    cp_softmax_stats_out["TC"] = TC
    nisa.tensor_copy(cp_softmax_stats_out["fa_running_max"], max_src)
    nisa.tensor_copy(cp_softmax_stats_out["fa_running_sum"], sum_src)


def _normalize_output_divide_by_running_sum(
    bufs: 'AttnInternalBuffers',
    atp: 'AttnTileParams',
    cfg: AttnTKGConfig,
    TC: 'TileConstants',
    sbm: SbufManager,
):
    """Normalize running_output by dividing by running_sum.

    Equivalent:
        output[d_head, s_active_bqh] /= sum[s_active_bqh]

    Sum is stored in SBUF as [s_active_bqh_tile, n_bsq_tiles] with s_active_bqh
    tiled across partition dim.

    Kernel steps:
        1. reciprocal:  sum_recip[s_active_bqh_tile, n_bsq_tiles] = 1/sum[s_active_bqh_tile, n_bsq_tiles]
        2. sum_recip_bc[d_tile_size, n_d_tiles * s_active_bqh] =
              _s_active_bqh_tile_transpose_broadcast_d_tiled(sum_recip)
        3. multiply:    output[d_tile_size, n_d_tiles * s_active_bqh] *= sum_recip_bc

    Args:
        bufs: Internal buffers containing:
            - running_output [d_tile_size, n_d_tiles * s_active_bqh] @ SBUF
            - running_sum [s_active_bqh_tile, n_bsq_tiles] @ SBUF
        atp: Tile parameters with active tile sizes and layout info.
        cfg: Kernel config with d_head and dtype settings.
        TC: Tile constants for transpose/broadcast helpers.
        sbm: SBUF memory manager for intermediate allocation.
    """
    # Compute reciprocal of running sum in-place
    nisa.reciprocal(
        bufs.running_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        bufs.running_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
    )

    # Transpose and broadcast sum_recip to [d_tile_size, n_d_tiles * s_active_bqh] for final normalization
    sum_recip_bc = sbm.alloc_stack(
        (atp.d_tile_size, atp.n_d_tiles * atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf
    )
    _s_active_bqh_tile_transpose_broadcast_d_tiled(bufs.running_sum, sum_recip_bc, atp, TC)

    # Normalize: running_output *= sum_recip_bc
    nisa.tensor_tensor(
        bufs.running_output,
        bufs.running_output,
        sum_recip_bc,
        op=nl.multiply,
    )


@dataclass
class AttnTileParams(nl.NKIObject):
    """Computed tiling and dimension parameters for the attention kernel.

    This dataclass holds all the computed parameters needed for tiling the attention
    computation, including data types, sharding information, dimension calculations,
    and flash attention parameters.

    Fields are grouped into:
    - Global parameters: fixed for the entire kernel invocation
    - Per-batch-tile parameters: recomputed by _update_atp_for_batch_tile for each batch tile
    - Softmax reduction parameters
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
    """Whether to use indirect DMA transpose for block KV loading (the batched path that reshapes
    k_prior into 4-d tiles). True when d_head<=128 and dtype is 2 bytes. When False (d_head>128,
    FP8 without fp8_packed, or 1-byte dtype), block KV uses nc_transpose and flat KV non-FP8
    uses dma_transpose directly per d-tile."""

    qk_swapped: bool = None
    """Whether the QK swapped layout path is active. Enabled by default on compatible shapes
    (see is_qk_swapped); force off with NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=1."""

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
    """Full batch size per NC before batch tiling. Equals AttnTileParams.bs_full // AttnTileParams.bs_n_prgs."""

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

    # d_head tiling parameters (for d_head > p_max)
    n_d_tiles: int = None
    """Number of d_head tiles (each of size p_max=128). 1 when d_head <= p_max, else d_head / p_max.
    When n_d_tiles > 1, Q/K in SBUF use layout [p_max, n_d_tiles * free] with d_head tiled into free dim."""

    d_tile_size: int = None
    """Size of each d_head tile. Equals min(d_head, p_max). Always 128 when d_head >= 128."""

    exp_v_sendrecv_gpsimd: bool = None
    """Whether to use GPSIMD DMA for exp_v inter-core sendrecv. Requires n_d_tiles==1 and small tile sizes."""

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

    use_online_softmax: bool = None
    """Whether to use the online softmax running buffers (running_max/sum/output/correction_factor)
    and defer the cross-NC softmax sync to _finalize_and_store. True when AttnTileParams.use_fa is
    True (multi-tile accumulation needs running buffers) OR s_prior is sharded across NCs (running
    state must persist past the per-FA-tile scope so the cross-NC sync can run in
    _finalize_and_store). False for single-NC non-FA, where the classical per-tile softmax path
    is used."""

    num_fa_tiles: int = None
    """Number of flash attention tiles to iterate over. 1 if not using FA, otherwise ceil(AttnTileParams.s_prior / fa_tile_size)."""

    fa_tile_s_prior: int = None
    """Size of each flash attention tile in s_prior dimension. Equals FA_TILE_SIZE (8K) when FA enabled."""

    fa_n_sprior_tile: int = None
    """Number of TileConstants.p_max-sized tiles within each FA tile. Equals ceil(AttnTileParams.fa_tile_s_prior / TileConstants.p_max)."""

    interleaved_fa_tiles: bool = False
    """Whether FA tiles are interleaved across NCs for block KV DMA skipping load balancing.
    False for flat KV or batch sharding (uses sequential local offsets)."""

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
    """QK^T result buffer.
      Regular: [TileConstants.p_max (s_prior), FATileContext.tile_n_sprior * AttnTileParams.s_active_bqh].
      Swapped: [p_max (s_active_bqh), n_bsq_tiles * tile_n_sprior * p_max (s_prior)]."""

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

    k_active_sb: nl.NkiTensor = None
    """Active key tensor in SBUF after optional RoPE. Shape [AttnTKGConfig.d_head, AttnTileParams.bs_full * AttnTKGConfig.s_active]."""

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

    active_blocks_sb_u32_fold_major: nl.NkiTensor = None
    """Fold-major uint32 block-index table [TileConstants.p_max, num_folds * bs] for the QK-swap K-load.
    The default active_blocks_sb(_u32) is batch-major (required by the batched V-load / unswapped K-load),
    but the swap MM1 gathers, per fold, a contiguous run of all batches_per_psum batches' blocks, which
    is contiguous only in fold-major order. Populated by _load_and_reshape_active_blk_table (additionally_emit_fold_major=True)."""

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
    max_sum_pack: nl.NkiTensor = None
    """Packed [running_max | running_sum] buffer; the sharded finalize sends it directly with no packing copies. Shape [AttnTileParams.s_active_bqh_tile, 2 * AttnTileParams.n_bsq_tiles]."""

    running_max: nl.NkiTensor = None
    """Running max across FA tiles for online softmax; first half of max_sum_pack. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]. Updated each FA tile."""

    running_sum: nl.NkiTensor = None
    """Running sum of exp values across FA tiles; second half of max_sum_pack. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]. Accumulated each FA tile."""

    correction_factor: nl.NkiTensor = None
    """Correction factor exp(prev_max - curr_max) for rescaling. Shape [AttnTileParams.s_active_bqh_tile, AttnTileParams.n_bsq_tiles]."""

    running_output: nl.NkiTensor = None
    """Running accumulated PV output across FA tiles. Shape [AttnTKGConfig.d_head, AttnTileParams.s_active_bqh]. Normalized at end."""


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
    """Which FA tile is being processed (0-indexed). Ranges from 0 to AttnTileParams.num_fa_tiles - 1."""

    tile_s_prior: int
    """Actual s_prior for this tile. Equals fa_tile_size (8K) for all but last tile; last tile may be smaller."""

    tile_n_sprior: int
    """Number of TileConstants.p_max-sized tiles within this FA tile. Equals ceil(FATileContext.tile_s_prior / TileConstants.p_max)."""

    tile_offset: int
    """Global s_prior offset where this tile starts. Includes NC base offset."""

    is_last_fa_tile: bool
    """True if this is the final FA tile. Used to determine when to load k_active/v_active and finalize output."""

    dynamic_tile_offset_sbuf: nl.NkiTensor = None
    """SBUF (1,1) int32 scalar holding the dynamic tile offset (fa_tile_idx * fa_tile_s_prior).
    Non-None only in the dynamic FA loop path. When set, functions should use this
    instead of tile_offset for runtime-dependent offset computations."""

    dynamic_tile_offset_f32: nl.NkiTensor = None
    """SBUF (P_MAX, 1) float32 version of dynamic_tile_offset_sbuf for gen_mask_tkg iota bias."""


def _compute_fa_tile_context(fa_tile_idx: int, atp: AttnTileParams, TC: TileConstants) -> FATileContext:
    """Compute the context for a specific FA tile.

    For block KV, tile_offset is always a global s_prior offset (works for both interleaved
    and contiguous sharding). For flat KV, tile_offset is local to the NC's portion.
    """
    is_last_fa_tile = fa_tile_idx == atp.num_fa_tiles - 1

    # Compute tile offset (always global)
    if atp.interleaved_fa_tiles:
        tile_offset = _get_interleaved_fa_tile_offset(fa_tile_idx, atp)
    else:
        tile_offset = atp.sprior_prg_id * atp.s_prior + fa_tile_idx * atp.fa_tile_s_prior

    # Compute tile size (last tile may be smaller)
    if is_last_fa_tile and atp.use_fa:
        tile_s_prior = atp.s_prior - fa_tile_idx * atp.fa_tile_s_prior
        tile_s_prior = min(atp.fa_tile_s_prior, tile_s_prior)
        tile_n_sprior = div_ceil(tile_s_prior, TC.p_max)
    else:
        tile_s_prior = atp.fa_tile_s_prior
        tile_n_sprior = atp.fa_n_sprior_tile

    return FATileContext(
        fa_tile_idx=fa_tile_idx,
        tile_s_prior=tile_s_prior,
        tile_n_sprior=tile_n_sprior,
        tile_offset=tile_offset,
        is_last_fa_tile=is_last_fa_tile,
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


def _update_atp_for_batch_tile(atp: AttnTileParams, tile_bs: int, TC: TileConstants, cfg: AttnTKGConfig):
    """Recompute batch-dependent atp fields for a given batch tile size.

    These fields depend on the current batch tile's bs and must be recomputed
    each time the batch tile changes. When num_batch_tiles == 1, tile_bs == bs_per_nc
    and these are equivalent to the original (pre-tiling) values.
    """
    atp.bs = tile_bs
    atp.s_active_bqh = atp.bs * atp.s_active_qh  # flattened dim of [bs, q_heads, s_active] for this batch tile
    atp.s_active_bqh_remainder = atp.s_active_bqh % TC.p_max
    atp.n_bsq_full_tiles = atp.s_active_bqh // TC.p_max
    atp.n_bsq_tiles = atp.n_bsq_full_tiles + (atp.s_active_bqh_remainder > 0)
    atp.s_active_bqh_tile = TC.p_max if atp.n_bsq_tiles > 1 else atp.s_active_bqh
    atp.batch_interleave_degree = min(atp.bs, TC.psum_b_max)  # PSUM bank interleaving across batches
    atp.exp_v_sendrecv_gpsimd = (
        cfg.use_gpsimd_sb2sb
        and atp.n_prgs > 1
        and atp.n_d_tiles == 1
        and atp.bs * atp.s_active_qh <= 256
        and cfg.d_head % 16 == 0
    )


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

    atp.is_block_kv = active_blocks_table is not None
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

    # FP8 Q: attention tkg has no Q dequant path
    kernel_assert(
        not (is_fp8_e4m3(q.dtype) or is_fp8_e5m2(q.dtype)),
        f"FP8 query dtype is not supported. Got q.dtype={q.dtype}. Use nl.bfloat16.",
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
    atp.use_dma_transpose = (
        (sizeinbytes(atp.k_prior_load_type) == 2) and (not atp.is_fp8_kv or cfg.fp8_packed) and cfg.d_head <= 128
    )  # use_dma_transpose may be further disabled in _setup_block_kv_cache for small block_len configs
    atp.io_type = q.dtype
    atp.inter_type = nl.float32
    atp.bs_full = cfg.bs
    atp.bs_per_nc = atp.bs_full // atp.bs_n_prgs  # full per-NC batch size before batch tiling
    atp.s_prior = cfg.curr_sprior // atp.sprior_n_prgs  # shard prior seqlen onto each prg
    atp.s_active_qh = cfg.s_active * cfg.q_head  # flattened dim of [q_heads, s_active]

    # QK swap requires packing s_active_qh for column tiling and enough bs_per_nc to fill all partitions.
    # Q-tiling (s_active_qh > p_max, one batch spanning multiple tiles) is not yet supported. The
    # compatibility check lives in is_qk_swapped so the test infra can mirror it when laying out the
    # pre-generated mask.
    atp.qk_swapped = is_qk_swapped(
        bs=cfg.bs,
        q_head=cfg.q_head,
        d_head=cfg.d_head,
        s_active=cfg.s_active,
        curr_sprior=cfg.curr_sprior,
        lnc=nl.num_programs(0),
        p_max=TC.p_max,
        block_len=cfg.block_len,
        is_2byte_kv=sizeinbytes(k_prior.dtype) == 2,
        fp8_packed=cfg.fp8_packed,
        fuse_rope=cfg.fuse_rope,
    )
    atp.n_sprior_tile = div_ceil(atp.s_prior, TC.p_max)  # total number of p_max-tiles across full s_prior

    # d_head tiling for d_head > p_max
    atp.d_tile_size = min(cfg.d_head, TC.p_max)
    atp.n_d_tiles = div_ceil(cfg.d_head, TC.p_max)

    # ========== Derived parameter validation ==========
    kernel_assert(
        cfg.d_head % TC.p_max == 0 or cfg.d_head <= TC.p_max,
        f"d_head must be <= p_max or a multiple of p_max (ragged d_head not yet supported). "
        f"Got d_head={cfg.d_head}, p_max={TC.p_max}.",
    )
    kernel_assert(
        atp.n_d_tiles == 1 or not cfg.strided_mm1,
        f"strided_mm1 is not supported with d_head > p_max. Got d_head={cfg.d_head}.",
    )
    kernel_assert(
        atp.n_d_tiles == 1 or not cfg.fp8_packed,
        f"fp8_packed is not supported with d_head > p_max (requires DMA transpose). Got d_head={cfg.d_head}.",
    )
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
    kernel_assert(
        not cfg.fuse_rope or cfg.d_head <= TC.p_max,
        f"Fuse rope requires d_head <= p_max ({TC.p_max}). Got d_head={cfg.d_head}.",
    )

    # Flash attention parameters
    # Enable FA when s_prior exceeds the tile size threshold
    use_fa, fa_tile_size = uses_flash_attention(cfg.enable_fa_s_prior_tiling, atp.s_prior)
    atp.use_fa = use_fa
    if atp.use_fa:
        atp.num_fa_tiles = div_ceil(atp.s_prior, fa_tile_size)
        atp.fa_tile_s_prior = fa_tile_size
        atp.fa_n_sprior_tile = div_ceil(fa_tile_size, TC.p_max)
        # Last FA tile must be able to hold s_active (k_active is loaded at tile end)
        last_tile_s_prior = atp.s_prior % fa_tile_size if atp.s_prior % fa_tile_size != 0 else fa_tile_size
        kernel_assert(
            last_tile_s_prior >= cfg.s_active,
            f"Last FA tile size ({last_tile_s_prior}) must be >= s_active ({cfg.s_active})",
        )
    else:
        atp.num_fa_tiles = 1
        atp.fa_tile_s_prior = atp.s_prior
        atp.fa_n_sprior_tile = atp.n_sprior_tile

    # For block KV with LNC2 s_prior sharding, enable interleaved FA tile assignment.
    # Instead of each NC owning a contiguous half of s_prior, tiles alternate between NCs
    # for better load balancing when cache is partially filled (DMA skipping).
    if atp.is_block_kv and atp.sprior_n_prgs == 2:
        atp.interleaved_fa_tiles = True

    # Online softmax (running max/sum/output buffers) is used whenever:
    #   - FA tiling is active (running state must persist across FA tiles), OR
    #   - s_prior is sharded across NCs (the running state carries local values past the per-FA-tile
    #     scope into _finalize_and_store, where the cross-NC sync runs).
    # Only the classical single-NC non-FA path skips them.
    atp.use_online_softmax = atp.use_fa or atp.sprior_n_prgs > 1 or cfg.return_cp_softmax_stats

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
    _expected_k_active_shape = (atp.d_tile_size, atp.n_d_tiles * cfg.bs * cfg.s_active)
    kernel_assert(
        k_active.shape == _expected_k_active_shape,
        f"Block KV requires k_active has shape {_expected_k_active_shape}, got {k_active.shape}.",
    )
    kernel_assert(
        active_blocks_table.shape[0] == cfg.bs,
        f"Block KV requires active_blocks_table has the shape (bs, num_blocks_per_batch), expected active_blocks_table.shape[0]={cfg.bs}, got {active_blocks_table.shape[0]}",
    )
    kernel_assert(
        active_blocks_table.shape[1] * cfg.block_len == cfg.curr_sprior,
        f"Block KV requires the number of blocks per batch times the number of blocks to match the current context length, expected active_blocks_table.shape[1] * cfg.block_len={cfg.curr_sprior}, got {active_blocks_table.shape[1] * cfg.block_len}",
    )

    # Reshape before performing modifications on the dimensions
    bufs.v_active_reshaped = v_active.reshape((cfg.bs, cfg.s_active * cfg.d_head))

    # For block cache support, the kernel requires the number of blocks per batch to be a multiple of 128.
    # When S_ctx is small and blocks per batch < 128, we will "resize" blocks to make blocks per batch a multiple of 128.
    n_prgs = nl.num_programs(0)
    block_len, blk_cache_resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_per_batch=active_blocks_table.shape[1],
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


def _allocate_qk_buffers(
    atp: AttnTileParams, TC: TileConstants, sbm: SbufManager, bufs: AttnInternalBuffers, fa_ctx: FATileContext
):
    """Allocate core QK buffers for the current FA tile.

    Create KQ^T result mloc for all batches (filled with -INF for masking). See the SBUF layouts
    below for the per-path shapes. Cascaded max reduce does a strided access on the free dimension.

    Uses fa_ctx.tile_n_sprior which is the actual tile size (may be smaller for last FA tile).
    """

    # Layouts:
    #   unswapped: bufs.qk, mask are [p_max=s_prior, fa_tile_n_sprior * s_active_bqh]
    #   QK_SWAP  : bufs.qk, mask are [p_max=s_active_bqh, n_bsq_tiles * fa_tile_n_sprior * p_max=s_prior]
    io_free = fa_ctx.tile_n_sprior * atp.s_active_bqh
    if atp.qk_swapped:
        n_bsq_tiles = (atp.s_active_bqh + TC.p_max - 1) // TC.p_max
        qk_free = n_bsq_tiles * fa_ctx.tile_n_sprior * TC.p_max
    else:
        qk_free = io_free
    bufs.qk = sbm.alloc_stack((TC.p_max, qk_free), dtype=atp.inter_type)
    # Set to -inf only when the evict uses tensor_copy_predicated.
    if atp.qk_row_tile_factor > 1 and not atp.qk_swapped:
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

    # qk_io_type is [p_max=s_prior, fa_tile_n_sprior * s_active_bqh] in both paths (PV operand).
    bufs.qk_io_type = sbm.alloc_stack((TC.p_max, io_free), dtype=atp.io_type)

    # mask shares bufs.qk's [p_max, qk_free] layout in both paths.
    bufs.mask_sb = sbm.alloc_stack((TC.p_max, qk_free), dtype=nl.uint8, buffer=nl.sbuf)


def _allocate_online_softmax_buffers(
    atp: AttnTileParams, cfg: AttnTKGConfig, sbm: SbufManager, bufs: AttnInternalBuffers, is_dynamic: bool = False
):
    """Allocate the online-softmax running-statistics buffers (running max/sum/output and the
    correction-factor staging buffer).

    Only called when atp.use_online_softmax is True (i.e. atp.use_fa or atp.sprior_n_prgs > 1).
    When is_dynamic=True, buffers are initialized to identity values so the update math works
    correctly on the first dynamic iteration without a special case.
    """
    # running_max and running_sum are the two halves of one packed buffer so the sharded finalize
    # sends both stats in a single sendrecv with no packing copies.
    bufs.max_sum_pack = sbm.alloc_stack(
        (atp.s_active_bqh_tile, 2 * atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf
    )
    bufs.running_max = bufs.max_sum_pack[:, : atp.n_bsq_tiles]
    bufs.running_sum = bufs.max_sum_pack[:, atp.n_bsq_tiles :]
    if is_dynamic:
        nisa.memset(bufs.running_max, value=-np.inf)
        nisa.memset(bufs.running_sum, value=0)

    # Correction factor exp(prev_max - curr_max) - same shape as running_max
    bufs.correction_factor = sbm.alloc_stack(
        (atp.s_active_bqh_tile, atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf
    )
    if is_dynamic:
        nisa.memset(bufs.correction_factor, value=1.0)

    # Running output - accumulates PV results across tiles
    # For d_head > p_max, tile d_head into free dimension: [p_max, n_d_tiles * s_active_bqh]
    bufs.running_output = sbm.alloc_stack(
        (atp.d_tile_size, atp.n_d_tiles * atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf
    )
    if is_dynamic:
        nisa.memset(bufs.running_output, value=0)


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
    """Exchange the local [max | sum] pack with the other core in one sendrecv and reconstruct the
    global max and global running sum (folding in the sink if present).

    On exit running_max/running_sum hold the global max/sum and correction_factor =
    exp(local_max - global_max) for rescaling running_output.
    """
    kernel_assert(not atp.max_negated, "Unexpected atp.max_negated=True when computing cross-NC max/sum")

    sbm.open_scope()

    # Receives the other core's [max | sum] pack.
    remote_pack = sbm.alloc_stack((atp.s_active_bqh_tile, 2 * atp.n_bsq_tiles), dtype=atp.inter_type, buffer=nl.sbuf)

    # Swap the packed stats over GPSIMD SB-to-SB when it fits the 1024-byte/partition limit, else DMA.
    use_gpsimd = cfg.use_gpsimd_sb2sb and (2 * atp.n_bsq_tiles * sizeinbytes(atp.inter_type) <= 1024)
    nisa.sendrecv(
        src=bufs.max_sum_pack,
        dst=remote_pack,
        send_to_rank=(1 - atp.sprior_prg_id),
        recv_from_rank=(1 - atp.sprior_prg_id),
        pipe_id=0,
        dma_engine=dma_engine.gpsimd_dma if use_gpsimd else dma_engine.dma,
    )

    remote_max = remote_pack[:, 0 : atp.n_bsq_tiles]
    remote_sum = remote_pack[:, atp.n_bsq_tiles : 2 * atp.n_bsq_tiles]

    # global_max = max(local_max, remote_max)
    nisa.tensor_tensor(
        dst=bufs.running_max,
        data1=bufs.running_max,
        data2=remote_max,
        op=nl.maximum,
    )
    # Fold sink in only now, so the exchanged maxes stayed pure for the remote-sum rescale below.
    if sink_values is not None:
        nisa.tensor_tensor(
            dst=bufs.running_max,
            data1=bufs.running_max,
            data2=sink_values,
            op=nl.maximum,
        )

    # correction_factor = exp(local_max - global_max), rescales the local sum and output.
    _update_correction_factor(atp, bufs.correction_factor, bufs.running_max, local_running_max)

    # remote_max := exp(remote_max - global_max), the remote-sum rescale factor.
    for i_bsq_tile in range(atp.n_bsq_tiles):
        nisa.activation(
            dst=remote_max[:, i_bsq_tile],
            op=nl.exp,
            data=bufs.running_max[: atp.s_active_bqh_tile, i_bsq_tile],
            bias=remote_max[:, i_bsq_tile],
            scale=-1.0,
        )

    # global_sum = correction_factor * local_sum + exp(remote_max - global_max) * remote_sum
    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=bufs.correction_factor,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=remote_sum,
        data1=remote_sum,
        data2=remote_max,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=bufs.running_sum,
        data1=bufs.running_sum,
        data2=remote_sum,
        op=nl.add,
    )
    sbm.close_scope()

    if sink_values is not None:
        # global_sum += exp(sink - global_max), reusing sink_values in place.
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


def _update_running_max(
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Update flash attention running max after computing tile max.

    Called inside _cascaded_max_reduce after step 2.3 for each FA tile.
    Updates:
    - running_max: running max across tiles, shape (s_active_bqh_tile, n_bsq_tiles)
    - correction_factor: exp(prev_max - curr_max) for rescaling, same shape

    The tile max is in bufs.qk_max_buf[:, :n_bsq_tiles] after reduction.
    When max_negated=True, values are negated (so min gives true max).
    When max_negated=False, values are not negated (so max gives true max).

    Cross-NC sync (when sharded) is deferred to _finalize_and_store; running_max stays local
    here.
    """
    # Get current tile max from qk_max_buf (first n_bsq_tiles columns after reduction)
    tile_max = bufs.qk_max_buf[: atp.s_active_bqh_tile, : atp.n_bsq_tiles]

    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None
    if not is_dynamic and fa_ctx.fa_tile_idx == 0:
        # First tile (static path): just copy tile max, correction = 1.0
        nisa.tensor_copy(bufs.running_max, tile_max)
        nisa.memset(bufs.correction_factor, value=1.0)
    else:
        # Update running max and compute correction factor.
        # In dynamic path, running_max is identity-init (-inf) so first tile works correctly.
        sbm.open_scope()
        # Save previous running max
        prev_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=bufs.running_max.dtype)
        nisa.tensor_copy(prev_running_max, bufs.running_max, engine=nisa.scalar_engine)

        # Update running max: min if negated, max if not negated
        nisa.tensor_tensor(
            bufs.running_max, bufs.running_max, tile_max, op=(nl.minimum if atp.max_negated else nl.maximum)
        )

        # Local correction only. Global sync (for sharded) happens in _finalize_and_store.
        _update_correction_factor(atp, bufs.correction_factor, bufs.running_max, prev_running_max)

        sbm.close_scope()


def _update_running_sum(
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Update flash attention running sum after computing tile exp sum.

    Called in _cascaded_sum_reduction for each FA tile.
    Updates running_sum = running_sum * correction_factor + tile_sum

    All tensors have shape (s_active_bqh_tile, n_bsq_tiles).
    """
    # exp_sum has shape [s_active_bqh_tile, n_bsq_tiles * softmax_final_reduction_length]
    # After reduction, tile sum is in exp_sum[:, 0:n_bsq_tiles]
    tile_sum = bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles]

    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None
    if not is_dynamic and fa_ctx.fa_tile_idx == 0:
        # First tile (static path): just copy. Dynamic path uses identity-init (0).
        nisa.tensor_copy(bufs.running_sum, tile_sum)
    else:
        # running_sum = running_sum * correction_factor + tile_sum
        # Step 1: running_sum *= correction_factor
        nisa.tensor_tensor(
            bufs.running_sum,
            bufs.running_sum,
            bufs.correction_factor,
            op=nl.multiply,
        )
        # Step 2: running_sum += tile_sum
        nisa.tensor_tensor(
            bufs.running_sum,
            bufs.running_sum,
            tile_sum,
            op=nl.add,
        )


def _accumulate_output(
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
):
    """Accumulate PV output for flash attention.

    Called after PV matmul for each FA tile.
    Updates running_output = running_output * correction_factor + tile_output

    Note: exp_v has shape [d_tile_size, n_d_tiles * bs * s_active_qh] (tiled d_head in free dim).
    running_output has shape [d_tile_size, n_d_tiles * s_active_bqh] (tiled d_head in free dim).
    For FA, we don't multiply by exp_sum_recip here - that's done in finalize.
    """
    # Reshape exp_v to tiled layout [d_tile_size, n_d_tiles * s_active_bqh]
    exp_v_flat = bufs.exp_v.reshape((atp.d_tile_size, atp.n_d_tiles * atp.s_active_bqh))

    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None
    if not is_dynamic and fa_ctx.fa_tile_idx == 0:
        # First tile (static path): just copy. Dynamic path uses identity-init (0).
        nisa.tensor_copy(bufs.running_output, exp_v_flat)
    else:
        # running_output = running_output * correction_factor + tile_output
        # correction_factor has shape [s_active_bqh_tile, n_bsq_tiles]
        # running_output has shape [d_tile_size, n_d_tiles * s_active_bqh]
        #
        # Transpose correction_factor to [1, s_active_bqh] then broadcast to [d_tile_size, n_d_tiles * s_active_bqh]
        sbm.open_scope()
        # Broadcasted correction factor - shape [d_tile_size, n_d_tiles * s_active_bqh]
        correction_factor_bc = sbm.alloc_stack(
            (atp.d_tile_size, atp.n_d_tiles * atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf
        )
        _s_active_bqh_tile_transpose_broadcast_d_tiled(bufs.correction_factor, correction_factor_bc, atp, TC)

        # Now apply: running_output = running_output * correction_factor_bc + exp_v
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            correction_factor_bc,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            exp_v_flat,
            op=nl.add,
        )
        sbm.close_scope()


def _finalize_and_store(
    sink,
    out: nl.NkiTensor,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
    cp_softmax_stats_out=None,
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
           sendrecv, folds sink into the global max, produces the global running_sum, and updates
           correction_factor to exp(local - global).
        4. Rescale running_output by correction_factor (transpose_broadcast to [d_head,
           s_active_bqh]); running_sum is already at global scale from step 3.

    Then normalize running_output by reciprocal(running_sum) and store.

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
            _prep_sink(sink, sink_values, atp, cfg, TC, sbm, btc)

        # 1. Save local_max (used as prev when computing c_local = exp(local_max - global_max)
        #    and for rescaling running_sum / running_output to the global scale).
        local_running_max = sbm.alloc_stack(bufs.running_max.shape, dtype=atp.inter_type, buffer=nl.sbuf)
        nisa.tensor_copy(local_running_max, bufs.running_max)

        # 2. Single cross-NC exchanging BOTH local max and local sum in one sendrecv,
        #    fold sink into the global max, produce the global running_sum and
        #    correction_factor = exp(local - global).
        _gather_and_compute_global_running_max_and_sum(atp, cfg, sbm, bufs, local_running_max, sink_values=sink_values)

        # 3. Rescale running_output by c_local (broadcast to [d_tile_size, n_d_tiles * s_active_bqh]).
        correction_factor_bc = sbm.alloc_stack(
            (atp.d_tile_size, atp.n_d_tiles * atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf
        )
        _s_active_bqh_tile_transpose_broadcast_d_tiled(bufs.correction_factor, correction_factor_bc, atp, TC)
        nisa.tensor_tensor(
            bufs.running_output,
            bufs.running_output,
            correction_factor_bc,
            op=nl.multiply,
        )

        sbm.close_scope()

    # Debug tensor writes for the online-softmax path (sharded; FA single-NC). Dumping from
    # running_max / running_sum captures the final global values. Batch-tiling case
    # (bs != bs_per_nc) is handled by the zero-fill fallback in _cascaded_*.
    if DBG_TENSORS is not None and atp.bs == atp.bs_per_nc:
        _store_dbg_qk_max(bufs.running_max, atp.max_negated, "finalize", atp, TC, sbm, bufs)
        _store_dbg_exp_sum(bufs.running_sum, "finalize", atp, TC, sbm, bufs)

    # CP: export stats; non-CP: normalize output
    if cp_softmax_stats_out is not None:
        _copy_and_export_softmax_stats(
            cp_softmax_stats_out, bufs.running_max, bufs.running_sum, atp.max_negated, atp, TC
        )
    else:
        _normalize_output_divide_by_running_sum(bufs, atp, cfg, TC, sbm)
    sbm.close_scope()
    _gather_and_store_output(out, bufs.running_output, atp, cfg, sbm, btc)


def _load_and_broadcast_pos_ids(pos_ids, atp, cfg, TC, sbm, name):
    """Load position IDs and broadcast onto all 128 partitions (for TensorScalarPtr).

    Shared helper for loading both rope_pos_ids and start_pos_ids, which follow
    the same pattern: reshape → alloc_stack → dma_copy → alloc_stack → stream_shuffle_broadcast.
    """
    pos_ids_sb = sbm.alloc_stack((TC.p_max, atp.bs_per_nc * cfg.s_active), dtype=pos_ids.dtype)
    pos_ids = pos_ids.reshape([atp.bs_n_prgs, atp.bs_per_nc * cfg.s_active])

    sbm.open_scope()
    pos_ids_loaded = sbm.alloc_stack((1, atp.bs_per_nc * cfg.s_active), dtype=pos_ids.dtype, align=4)
    nisa.dma_copy(pos_ids_loaded, pos_ids[atp.bs_prg_id, :], name=sbm.get_name_prefix() + name)
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
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Load mask for the current FA tile and batch tile."""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_dynamic_mask = fa_ctx.dynamic_tile_offset_sbuf is not None

    # If we don't use pos_id, mask is already generated outside of kernel. Otherwise, generate prior mask in kernel and
    # load active mask at the end of the generated mask.
    if not cfg.use_pos_id:
        # Reshape mask with full per-NC bqh, then slice to batch tile
        full_s_active_bqh = atp.bs_per_nc * atp.s_active_qh
        # Compute source offset including FA tile offset and batch tile offset
        bqh_offset = btc.tile_batch_offset * atp.s_active_qh

        if atp.qk_swapped:
            # The HBM mask is generated in [bs_n_prgs, full_s_active_bqh, curr_sprior] layout.
            band_factor = _swap_band_factor(atp, TC)
            if band_factor == 1:
                # mask_sb is [p_max=s_active_bqh, s_prior] to match the swapped QK.
                n_bsq_tiles = atp.s_active_bqh // TC.p_max
                mask_hbm_view = (
                    mask.reshape((atp.bs_n_prgs, full_s_active_bqh, cfg.curr_sprior))
                    .select(dim=0, index=atp.bs_prg_id)
                    .slice(dim=0, start=bqh_offset, end=bqh_offset + atp.s_active_bqh)
                    .slice(dim=1, start=fa_tile_offset, end=fa_tile_offset + fa_ctx.tile_s_prior)
                    .reshape_dim(0, [n_bsq_tiles, TC.p_max])
                    .permute([1, 0, 2])
                )
                mask_sb_view = bufs.mask_sb.reshape_dim(1, [n_bsq_tiles, fa_ctx.tile_s_prior])
            else:
                # When banding is active we fold sprior onto partition to fill partition dimension. HBM
                # mask is in layout [bs_n_prgs, s_active_bqh * band_factor, curr_sprior // band_factor],
                # with contiguous sprior from the same batch packed adjacent in partition.
                band_s_prior = fa_ctx.tile_s_prior // band_factor
                band_free_offset = fa_tile_offset // band_factor
                banded_sprior = cfg.curr_sprior // band_factor
                mask_hbm_view = (
                    mask.reshape((atp.bs_n_prgs, TC.p_max, banded_sprior))
                    .select(dim=0, index=atp.bs_prg_id)
                    .slice(dim=1, start=band_free_offset, end=band_free_offset + band_s_prior)
                )
                mask_sb_view = bufs.mask_sb[:, :band_s_prior]
            nisa.dma_copy(
                dst=mask_sb_view,
                src=mask_hbm_view,
                name=f"{sbm.get_name_prefix()}mask_load_swap_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )
        else:
            # Reshape mask as full s_prior and slice by global tile_offset
            mask_hbm_view = (
                mask.reshape((cfg.curr_sprior, atp.bs_n_prgs, full_s_active_bqh))
                .select(dim=1, index=atp.bs_prg_id)
                .slice(dim=0, start=fa_tile_offset, end=fa_tile_offset + fa_ctx.tile_s_prior)
                .slice(dim=1, start=bqh_offset, end=bqh_offset + atp.s_active_bqh)
            )

            # gen_mask_tkg_hbm stores in n_sprior_tile-major layout:
            # [n_sprior_tile, P_MAX, ...]. After flatten to [s_prior, ...], the
            # load must undo the tiling to recover [P_MAX, n_sprior_tile] in SBUF.
            #
            # TODO: The strided_mm1 flat-KV branch below uses reshape_dim(0,
            # [P_MAX, n_sprior_tile]) which assumes P_MAX-major order. This is
            # inconsistent with the n_sprior_tile-major HBM layout and should
            # use the else path (reshape + permute) like all other cases.
            # Kept as-is pending end-to-end validation; tracked for follow-up.
            if cfg.strided_mm1 and not atp.is_block_kv:
                mask_hbm_view = mask_hbm_view.reshape_dim(0, [TC.p_max, fa_tile_n_sprior])
            else:
                mask_hbm_view = mask_hbm_view.reshape_dim(0, [fa_tile_n_sprior, TC.p_max]).permute([1, 0, 2])

            mask_sb_view = bufs.mask_sb.reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
            nisa.dma_copy(
                dst=mask_sb_view,
                src=mask_hbm_view,
                name=f"{sbm.get_name_prefix()}mask_load_pregenerated_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
            )
    else:
        # In-kernel mask generation supports both flat and block KV cache.
        if atp.qk_swapped:
            band_factor = _swap_band_factor(atp, TC)
            if band_factor == 1:
                # mask_gen_view is [p_max=s_active_bqh, n_bsq_tiles, s_prior] to match the swapped QK.
                n_bsq_tiles = atp.s_active_bqh // TC.p_max
                mask_gen_view = bufs.mask_sb.reshape((TC.p_max, n_bsq_tiles, fa_ctx.tile_s_prior))
            else:
                # When banding is active we fold sprior onto partition to fill partition dimension.
                # mask is in layout [bs_n_prgs, s_active_bqh * band_factor, curr_sprior // band_factor],
                # with contiguous sprior from the same batch packed adjacent in partition.
                band_s_prior = fa_ctx.tile_s_prior // band_factor
                mask_gen_view = bufs.mask_sb[:, :band_s_prior].reshape((TC.p_max, 1, band_s_prior))
        else:
            mask_gen_view = bufs.mask_sb.reshape((TC.p_max, fa_tile_n_sprior, atp.bs, cfg.q_head, cfg.s_active))

        # For FA, only load active mask on the last FA tile and last NC
        # For non-FA, load active mask on the last NC (sprior_prg_id == sprior_n_prgs - 1)
        load_active_mask = (atp.sprior_prg_id == atp.sprior_n_prgs - 1) and fa_ctx.is_last_fa_tile

        # Slice pos_ids_sb to the batch tile's portion
        pos_ids_offset = btc.tile_batch_offset * cfg.s_active
        pos_ids_tile = bufs.pos_ids_sb[:, nl.ds(pos_ids_offset, atp.bs * cfg.s_active)]

        start_pos_tile = None
        if bufs.start_pos_sb is not None:
            start_pos_tile = bufs.start_pos_sb[:, nl.ds(pos_ids_offset, atp.bs * cfg.s_active)]

        sbm_prefix = sbm.get_name_prefix()
        sbm.set_name_prefix(f"{sbm_prefix}_bt{btc.batch_tile_idx}_")
        # tile_offset is always global — so we pass full s_prior_per_shard
        # and is_s_prior_sharded=False.
        gen_mask_tkg(
            pos_ids=pos_ids_tile,
            mask_out=mask_gen_view,
            bs=atp.bs,
            q_head=cfg.q_head,
            s_active=cfg.s_active,
            s_prior_per_shard=cfg.curr_sprior,
            start_pos=start_pos_tile,
            s_prior_offset=fa_tile_offset,
            block_len=atp.block_len,
            strided_mm1=cfg.strided_mm1,
            active_mask=mask if load_active_mask else None,
            sbm=sbm,
            is_batch_sharded=atp.bs_n_prgs > 1,
            is_s_prior_sharded=False,
            batch_offset=btc.tile_batch_offset,
            dynamic_s_prior_offset=fa_ctx.dynamic_tile_offset_f32 if is_dynamic_mask else None,
            transposed_out=atp.qk_swapped,
        )
        sbm.set_name_prefix(sbm_prefix)


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
    nisa.dma_copy(inv_freqs_sb, inv_freqs, name=sbm.get_name_prefix() + "inv_freqs_load")

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
                name=sbm.get_name_prefix() + "k_out_store_after_rope",
            )

    nisa.activation(bufs.q_sb, op=nl.copy, data=bufs.q_sb, scale=1 / math.sqrt(cfg.d_head))
    sbm.close_scope()


def _replicate_q_for_row_tiling(bufs, atp, cfg, TC, sbm):
    """Row-tiling (qk_row_tile_factor == 2, d_head == 64) contracts Q against both partition halves of
    K, so Q must be present in both halves. bufs.q_sb comes in with d_head partitions filled; copy it
    into a 2*d_head-partition buffer and replicate the second half via cross_partition_copy. No-op when
    qk_row_tile_factor == 1. Shared by the swap and unswapped MM1 paths (both read bufs.q_sb).
    """
    if atp.qk_row_tile_factor <= 1:
        return
    kernel_assert(
        atp.qk_row_tile_factor == 2,
        f"Q row-tiling replication only supports qk_row_tile_factor 2, got {atp.qk_row_tile_factor}.",
    )
    q_sb_row_tile = sbm.alloc_stack((TC.p_max, bufs.q_sb.shape[1]), dtype=bufs.q_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_sb_row_tile[0 : cfg.d_head, :], src=bufs.q_sb[0 : cfg.d_head, :])
    cross_partition_copy(
        src=q_sb_row_tile,
        dst=q_sb_row_tile,
        src_start_partition=0,
        dst_start_partition=cfg.d_head,
        num_partitions_to_copy=cfg.d_head,
        free_dim_size=bufs.q_sb.shape[1],
    )
    bufs.q_sb = q_sb_row_tile


"""
Main computation blocks
"""


def _compute_qk_matmul(
    k_prior,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)"""
    fa_tile_s_prior = fa_ctx.tile_s_prior
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_last_fa_tile = fa_ctx.is_last_fa_tile

    # Use per-tile s_prior for batch interleave calculation
    # For block KV with PE transpose path, each batch also accumulates k_loaded buffers across folds.
    sbuf_usage_per_batch = fa_tile_s_prior * sizeinbytes(k_prior.dtype)
    num_folds_this_tile = 0  # placeholder for flat KV
    if atp.is_block_kv:
        # For FA, compute which folds correspond to this tile
        # Each fold covers block_len * 128 elements of s_prior
        fold_s_prior = atp.block_len * TC.p_max
        fold_start = fa_tile_offset // fold_s_prior
        fold_end = div_ceil(fa_tile_offset + fa_tile_s_prior, fold_s_prior)
        num_folds_this_tile = fold_end - fold_start

        if not atp.use_dma_transpose:
            sbuf_usage_per_batch += (
                num_folds_this_tile * atp.block_len * cfg.d_head * sizeinbytes(atp.k_prior_load_type)
            )
    elif cfg.tp_k_prior and atp.is_fp8_kv:
        sbuf_usage_per_batch += TC.psum_b_max * cfg.d_head * sizeinbytes(atp.k_prior_load_type)
    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(
        sbuf_usage_per_batch,
        atp.batch_interleave_degree,
        sbm,
    )

    # Maximum multi-buffer degree inside a batch is 8 (banks) // bs (multi-buffer degree on the current scope)
    per_batch_interleave_degree = math.floor(float(TC.psum_b_max) / batch_interleave_degree_safe)

    # FP8 KV: use the resolved FP8 dtype (caller's concrete dtype, or
    # dtype_mode resolution for opaque "float8e4"). Otherwise pass through k_prior.dtype.
    _k_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else k_prior.dtype

    # Pre-compute dma_transpose batching params (before the per-batch loop).
    k_block_len_dma = 0
    k_block_len_row_tiled = 0
    k_prior_4d = None
    k_dma_batch_n_folds = 1
    k_dma_batch_n_batches = 1
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
        # 2. cap at 64 (matching the V-load V_CAP) to batch more folds/batches per
        #    DMA gather transpose, reducing DMA call count.
        k_dma_batch_n_folds, k_dma_batch_n_batches = _compute_dma_batch_params(
            num_folds_this_tile, atp.bs, k_block_len_dma, cap=64, sbm=sbm
        )

    k_sb_shared = None
    sbm.open_scope(interleave_degree=batch_interleave_degree_safe, name="qk_matmul")
    for i_b in range(atp.bs):
        # Load entire K_prior for current batch into sbuf (tile portion for FA)
        # d_head=64 with block KV + DMA transpose: row-tiling (qk_row_tile_factor=2) packs 2 K positions
        # per 128-partition column, using both halves of the partition dim.
        # Requires sufficient s_prior for the packed DMA layout (small SWA windows excluded).
        # For fp8_packed: allocate as bf16 with half the seq length (each bf16 slot holds 2 fp8 values)
        # For batch-batching: allocate a fresh shared buffer per group to avoid anti-deps.
        if atp.use_dma_transpose and k_dma_batch_n_batches > 1:
            if (i_b % k_dma_batch_n_batches) == 0:
                _k_sb_free_per_batch = num_folds_this_tile * k_block_len_row_tiled * TC.p_max
                k_sb_shared = sbm.alloc_stack(
                    (cfg.d_head * atp.qk_row_tile_factor, _k_sb_free_per_batch * k_dma_batch_n_batches),
                    dtype=nl.bfloat16 if cfg.fp8_packed else _k_sb_dtype,
                    buffer=nl.sbuf,
                    align=32,
                )
            k_sb = k_sb_shared
        elif atp.use_dma_transpose and cfg.fp8_packed:
            # fp8_packed: allocate as bf16 (each bf16 slot holds 2 fp8 values).
            # Partition dim = d_head * qk_row_tile_factor (128 when row-tiled, packs 2 positions per column).
            _k_sb_free_size = num_folds_this_tile * k_block_len_row_tiled * TC.p_max
            k_sb = sbm.alloc_stack(
                (cfg.d_head * atp.qk_row_tile_factor, _k_sb_free_size),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                align=32,
            )
        elif atp.use_dma_transpose:
            k_sb = sbm.alloc_stack(
                (cfg.d_head * atp.qk_row_tile_factor, fa_tile_s_prior // atp.qk_row_tile_factor),
                dtype=_k_sb_dtype,
                buffer=nl.sbuf,
                align=32,
            )
        else:
            # Non-DMA-transpose path (flat KV or d_head > 128): tile d_head into free dim
            k_sb = sbm.alloc_stack(
                (atp.d_tile_size, atp.n_d_tiles * fa_tile_s_prior),
                dtype=_k_sb_dtype,
                buffer=nl.sbuf,
                align=32,
            )
        if atp.is_block_kv:
            sbm.open_scope()
            for i_fold_rel in range(num_folds_this_tile):
                i_fold = fold_start + i_fold_rel
                batch_pos = i_b * num_folds_this_tile + i_fold_rel
                cur_blks = (bufs.active_blocks_sb).slice(dim=1, start=batch_pos, end=batch_pos + 1)
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
                    # Skip non-leading folds/batches within a batched group.
                    if i_fold_rel % k_dma_batch_n_folds != 0:
                        continue
                    if k_dma_batch_n_batches > 1 and i_b % k_dma_batch_n_batches != 0:
                        continue

                    k_dma_batch_size = k_dma_batch_n_folds * k_dma_batch_n_batches
                    idx_start = i_b * num_folds_this_tile + i_fold_rel

                    blks_u32 = (bufs.active_blocks_sb_u32).slice(
                        dim=1, start=idx_start, end=idx_start + k_dma_batch_size
                    )

                    dst_start = i_fold_rel * k_block_len_row_tiled * TC.p_max if k_dma_batch_n_batches == 1 else 0
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
                        name=f"{sbm.get_name_prefix()}k_prior_block_load_indirect_fa{fa_ctx.fa_tile_idx}_b{i_b}_f{i_fold}_bt{btc.batch_tile_idx}",
                    )

                    # Transpose to [d_tile_size, blk_len * 128blks] per d_tile
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
                            for i_d in range(atp.n_d_tiles):
                                d_src_offset = blk_len_i * cfg.d_head + i_d * atp.d_tile_size
                                tp_psum = nl.ndarray(
                                    (atp.d_tile_size, TC.p_max),
                                    dtype=atp.k_prior_load_type,
                                    buffer=nl.psum,
                                    address=None
                                    if sbm.is_auto_alloc()
                                    else (
                                        0,
                                        (tp_j_in_grp % per_batch_interleave_degree) * TC.psum_f_max_bytes,
                                    ),
                                )
                                nisa.nc_transpose(
                                    tp_psum,
                                    k_loaded[:, nl.ds(d_src_offset, atp.d_tile_size)],
                                )

                                # Balance psum->sbuf copies across vector and scalar engines
                                cur_idx = i_fold_rel * atp.block_len + blk_len_i
                                k_sb_offset = i_d * fa_tile_s_prior + cur_idx * TC.p_max
                                if cur_idx % 2 == 0:
                                    engine = nisa.vector_engine
                                else:
                                    engine = nisa.scalar_engine

                                nisa.tensor_copy(k_sb[:, k_sb_offset : k_sb_offset + TC.p_max], tp_psum, engine=engine)
            sbm.close_scope()
        elif not cfg.tp_k_prior:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.d_head, cfg.full_sprior),
                f"k_prior[1:] expected to have shape {(1, cfg.d_head, cfg.full_sprior)=}, received {k_prior.shape[1:]=}",
            )
            # k_prior shape: [B+, 1, d, full_sprior]
            # K_prior is already transposed in HBM, load each d_tile to its section in k_sb
            s_prior_pos = fa_tile_offset
            for i_d in range(atp.n_d_tiles):
                d_start = i_d * atp.d_tile_size
                k_sb_offset = i_d * fa_tile_s_prior
                k_prior_view = (
                    (k_prior)
                    .select(0, btc.global_batch_offset + i_b)
                    .squeeze_dim(0)
                    .slice(0, start=d_start, end=d_start + atp.d_tile_size)
                    .slice(1, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
                )
                nisa.dma_copy(
                    k_sb[:, k_sb_offset : k_sb_offset + fa_tile_s_prior],
                    k_prior_view,
                    name=f"{sbm.get_name_prefix()}k_prior_flat_load_transposed_fa{fa_ctx.fa_tile_idx}_b{i_b}_d{i_d}_bt{btc.batch_tile_idx}",
                )
        else:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.full_sprior, cfg.d_head),
                f"k_prior[1:] expected to have shape {(1, cfg.full_sprior, cfg.d_head)=}, received {k_prior.shape[1:]=}",
            )

            if atp.is_fp8_kv:
                # PE transpose path: load [p_max, d_head] tiles and transpose via PSUM.
                # Required for FP8 (DMA transpose doesn't support FP8).
                sbm.open_scope(interleave_degree=TC.psum_b_max)
                for tp_grp_i in range(fa_tile_n_sprior):
                    tile_start = tp_grp_i * TC.p_max
                    tile_size = min(TC.p_max, fa_tile_s_prior - tile_start)
                    k_loaded = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=atp.k_prior_load_type, buffer=nl.sbuf)
                    s_prior_pos = fa_tile_offset + tile_start
                    k_prior_view = (
                        (k_prior)
                        .select(0, btc.global_batch_offset + i_b)
                        .squeeze_dim(0)
                        .slice(0, start=s_prior_pos, end=s_prior_pos + tile_size)
                    )
                    nisa.dma_copy(
                        dst=k_loaded[:tile_size, :],
                        src=k_prior_view,
                    )
                    for i_d in range(atp.n_d_tiles):
                        d_src_offset = i_d * atp.d_tile_size
                        tp_psum = nl.ndarray(
                            (atp.d_tile_size, TC.p_max),
                            dtype=atp.k_prior_load_type,
                            buffer=nl.psum,
                            address=None
                            if sbm.is_auto_alloc()
                            else (0, (tp_grp_i % TC.psum_b_max) * TC.psum_f_max_bytes),
                        )
                        nisa.nc_transpose(
                            tp_psum[:, :tile_size],
                            k_loaded[:tile_size, d_src_offset : d_src_offset + atp.d_tile_size],
                        )
                        k_sb_offset = i_d * fa_tile_s_prior + tile_start
                        nisa.tensor_copy(k_sb[:, k_sb_offset : k_sb_offset + tile_size], tp_psum[:, :tile_size])
                    sbm.increment_section()
                sbm.close_scope()

            elif cfg.d_head > TC.p_max:
                # DMA transpose path for d_head > p_max: transpose each d_tile directly from HBM.
                # Each d_tile is a [s_prior_tile, d_tile_size=128] slice that gets transposed to
                # [d_tile_size=128, s_prior_tile] in k_sb. This avoids the intermediate SBUF load
                # + PE transpose + tensor_copy pipeline, freeing TensorEngine for matmul.
                s_prior_pos = fa_tile_offset
                for i_d in range(atp.n_d_tiles):
                    d_start = i_d * atp.d_tile_size
                    k_sb_offset = i_d * fa_tile_s_prior
                    k_sb_d_view = (
                        (k_sb)
                        .slice(1, start=k_sb_offset, end=k_sb_offset + fa_tile_s_prior)
                        .reshape_dim(1, [1, 1, fa_tile_s_prior])
                    )
                    k_prior_view = (
                        (k_prior)
                        .select(0, btc.global_batch_offset + i_b)
                        .squeeze_dim(0)
                        .slice(0, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
                        .slice(1, start=d_start, end=d_start + atp.d_tile_size)
                        .reshape_dim(1, [1, 1, atp.d_tile_size])
                    )
                    nisa.dma_transpose(
                        k_sb_d_view,
                        k_prior_view,
                        name=f"{sbm.get_name_prefix()}k_prior_dma_transpose_d{i_d}_fa{fa_ctx.fa_tile_idx}_b{i_b}_bt{btc.batch_tile_idx}",
                    )

            else:
                # FIXME: 4d reshape_dim required here, while simple slicing should suffice
                k_sb_view = (k_sb).reshape_dim(1, [1, 1, fa_tile_s_prior])
                s_prior_pos = fa_tile_offset
                k_prior_view = (
                    (k_prior)
                    .select(0, btc.global_batch_offset + i_b)
                    .squeeze_dim(0)
                    .slice(0, start=s_prior_pos, end=s_prior_pos + fa_tile_s_prior)
                    .reshape_dim(1, [1, 1, cfg.d_head])
                )
                nisa.dma_transpose(k_sb_view, k_prior_view)

        # If on final NC and last FA tile, add K_active to the end of k_sb
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1 and is_last_fa_tile:
            _stitch_k_active(
                k_sb,
                bufs,
                atp,
                cfg,
                TC,
                btc,
                i_b,
                num_folds_this_tile,
                k_block_len_row_tiled,
                k_dma_batch_n_folds,
                k_dma_batch_n_batches,
                _k_sb_dtype,
            )

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
        - K stationary: [d_tile_size, n_d_tiles * s_prior] loaded per batch into k_sb
        - Q moving: [d_tile_size, n_d_tiles * s_active_qh] per batch from q_sb
        - For d_head > 128: accumulate across n_d_tiles via repeated nc_matmul calls
        - Tile size: mm1_grp_sz (default 4096 = 4k) to balance PSUM usage
        - PSUM allocation: [P_MAX, n_mm1_per_grp * s_active_qh]
          where n_mm1_per_grp = mm1_grp_sz / P_MAX
        - PSUM constraint: (mm1_grp_sz / P_MAX) * s_active_qh < psum_f_max
        - Output: qk [P_MAX, n_sprior_tile * s_active_bqh] with batch interleaving
        - Memory: Each tile processes P_MAX rows of K against full Q per batch
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
                    (i_mm1_grp % per_batch_interleave_degree) * TC.psum_f_max_bytes,
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
                            k_dma_batch_n_batches,
                            i_b,
                        )
                        phys_start = phys_tile * 2 * TC.p_max + parity
                        k_tile = k_sb_fp8.slice(1, start=phys_start, end=phys_start + (num_acc - 1) * 2 + 1, step=2)
                    elif atp.use_dma_transpose:
                        # DMA transpose path: k_sb has shape [d_head * qk_row_tile_factor, s_prior]
                        phys_tile = _k_tile_physical_index(
                            logical_tile_idx,
                            k_block_len_row_tiled,
                            k_dma_batch_n_folds,
                            k_dma_batch_n_batches,
                            i_b,
                        )
                        k_tile = k_sb[0 : cfg.d_head * atp.qk_row_tile_factor, nl.ds(phys_tile * TC.p_max, num_acc)]
                    else:
                        # Non-DMA-transpose path (d_head tiled): k_sb has shape [d_tile_size, n_d_tiles * s_prior]
                        k_tile = k_sb[0 : atp.d_tile_size, nl.ds(k_tile_offset, num_acc)]

                # Matmul (diverges based on atp.qk_row_tile_factor)
                if atp.qk_row_tile_factor > 1:
                    q_batch_offset = (btc.global_batch_offset + i_b) * atp.s_active_qh
                    nisa.nc_matmul(
                        qk_psum_even[0:num_acc, i_mm1 * atp.s_active_qh : (i_mm1 + 1) * atp.s_active_qh],
                        stationary=k_tile[0 : cfg.d_head, :],
                        moving=bufs.q_sb[0 : cfg.d_head, q_batch_offset : q_batch_offset + atp.s_active_qh],
                        tile_size=(cfg.d_head, TC.p_max),
                        tile_position=(0, 0),
                    )
                    nisa.nc_matmul(
                        qk_psum_odd[0:num_acc, i_mm1 * atp.s_active_qh : (i_mm1 + 1) * atp.s_active_qh],
                        stationary=k_tile[cfg.d_head : TC.p_max, :],
                        moving=bufs.q_sb[cfg.d_head : TC.p_max, q_batch_offset : q_batch_offset + atp.s_active_qh],
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
                    # Accumulate across d_head tiles: each d_tile contributes to the same PSUM output
                    for i_d in range(atp.n_d_tiles):
                        if i_d > 0:
                            k_sb_offset = i_d * fa_tile_s_prior + k_tile_offset
                            k_tile = k_sb[0 : atp.d_tile_size, nl.ds(k_sb_offset, num_acc)]

                        q_sb_d_offset = i_d * atp.bs_full * atp.s_active_qh
                        q_sb_view = (
                            (bufs.q_sb)
                            .slice(1, start=q_sb_d_offset, end=q_sb_d_offset + atp.bs_full * atp.s_active_qh)
                            .reshape_dim(1, [atp.bs_full, atp.s_active_qh])
                            .select(1, (btc.global_batch_offset + i_b))
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
                    qk_sb_4d = bufs.qk.reshape_dim(1, [fa_tile_n_sprior // 2, 2, atp.bs, atp.s_active_qh]).select(
                        3, i_b
                    )
                    mask_sb_4d = bufs.mask_sb.reshape_dim(
                        1, [fa_tile_n_sprior // 2, 2, atp.bs, atp.s_active_qh]
                    ).select(3, i_b)
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
                    qk_sb_view = bufs.qk.reshape_dim(1, [fa_tile_n_sprior, atp.bs, atp.s_active_qh]).select(2, i_b)
                    mask_sb_view = bufs.mask_sb.reshape_dim(1, [fa_tile_n_sprior, atp.bs, atp.s_active_qh]).select(
                        2, i_b
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
                # Flush psum -> sb, the write to sb needs to be strided for batch interleaving
                num_acc_cpy = min(n_mm1_per_grp, fa_tile_s_prior // TC.p_max - i_mm1_grp * n_mm1_per_grp)

                if num_acc_cpy <= 0:
                    break  # i_mm1_grp * n_mm1_per_grp is strictly increasing

                qk_psum_view = (
                    (qk_psum).reshape_dim(1, [n_mm1_per_grp, atp.s_active_qh]).slice(1, start=0, end=num_acc_cpy)
                )

                sprior_tile_pos = i_mm1_grp * n_mm1_per_grp
                qk_sb_view = (
                    (bufs.qk)
                    .reshape_dim(1, [fa_tile_n_sprior, atp.bs, atp.s_active_qh])
                    .slice(1, start=sprior_tile_pos, end=sprior_tile_pos + num_acc_cpy)
                    .select(2, i_b)
                )

                mask_sb_view = (
                    (bufs.mask_sb)
                    .reshape_dim(1, [fa_tile_n_sprior, atp.bs, atp.s_active_qh])
                    .slice(1, start=sprior_tile_pos, end=sprior_tile_pos + num_acc_cpy)
                    .select(2, i_b)
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

    _store_dbg_qk(DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)


def _compute_kq_matmul_and_max_swapped(
    k_prior,
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """
    Step 1. MM1 of KQ^T for the QK-swap (transposed-score, column-tiled) path.

    In situations where there aren't enough batches to fill partition, swapped path will fold sprior
    onto partition to fill all 128 channels.
    """

    batches_per_psum = TC.p_max // atp.s_active_qh
    gather_batches = min(atp.bs_per_nc, batches_per_psum)
    # Partition banding: when the per-NC batch is too small to fill the 128 output partitions
    # (bs_per_nc < batches_per_psum), each query's s_prior is folded across sprior_band_factor contiguous
    # bands in output partition via stationary column location. == 1 for unbanded case.
    sprior_band_factor = _swap_band_factor(atp, TC)
    # When sprior_band_factor==1 these are same as unbanded values.
    s_active_bqh_banded_tile = atp.s_active_bqh_tile * sprior_band_factor
    n_bsq_banded_tiles = div_ceil(atp.s_active_bqh * sprior_band_factor, TC.p_max)
    band_s_prior = fa_ctx.tile_s_prior // sprior_band_factor

    # Per-tile s_prior drives the batch-interleave budget. Block-KV folds covered by this
    # FA tile: each fold spans block_len * 128 of s_prior.
    sbuf_usage_per_batch = fa_ctx.tile_s_prior * sizeinbytes(k_prior.dtype)
    fold_s_prior = atp.block_len * TC.p_max
    fold_start = fa_ctx.tile_offset // fold_s_prior
    fold_end = div_ceil(fa_ctx.tile_offset + fa_ctx.tile_s_prior, fold_s_prior)
    num_folds_this_tile = fold_end - fold_start
    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(
        sbuf_usage_per_batch,
        atp.batch_interleave_degree,
        sbm,
    )

    _k_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else k_prior.dtype

    # Maximum multi-buffer degree inside a batch is 8 (banks) // bs
    per_batch_interleave_degree = TC.psum_b_max // batch_interleave_degree_safe

    # When a sink is present, reserve an extra reduction slot for the prepped sink and fold it into
    # the per-position max at FA tile 0 when unsharded. Sharded path folds the sink into the GLOBAL
    # max in _finalize_and_store.
    stage_sink = sink is not None and fa_ctx.fa_tile_idx == 0 and atp.sync_softmax_per_fa_tile
    atp.softmax_final_reduction_length = 1 + stage_sink
    atp.softmax_final_reduction_sink_idx = atp.softmax_final_reduction_length - 1 if stage_sink else None
    bufs.qk_max_buf = sbm.alloc_stack(
        (s_active_bqh_banded_tile, n_bsq_banded_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    # Row-tile to pack qk_row_tile_factor consecutive k_block_len_dma rows into K moving
    if cfg.fp8_packed:
        # fp8 [N, elems_fp8] viewed as bf16 [N, elems_fp8 // 2]
        k_prior_src = bufs.k_prior_reshaped.view(nl.bfloat16)
        k_block_len_dma = bufs.k_prior_reshaped.shape[1] // (cfg.d_head * 2)
    else:
        k_prior_src = bufs.k_prior_reshaped
        k_block_len_dma = atp.block_len
    k_d_head_row_tile = cfg.d_head * atp.qk_row_tile_factor
    k_block_len_row_tile = k_block_len_dma // atp.qk_row_tile_factor

    # Reinterpret K_prior for row-tile packing
    k_prior_4d = k_prior_src.reshape((bufs.k_prior_reshaped.shape[0], 1, k_block_len_row_tile, k_d_head_row_tile))

    # Column tiling to pack multiple s_active_qh into Q stationary.
    # When a batch is narrower than a 32-wide column tile (s_active_qh < 32), multiple batches share a
    # column tile. Each batch's Q occupies a disjoint sub-band of a zero-padded tile, and the batches'
    # results accumulate into staggered partition bands in PSUM over several matmul steps.
    # NOTE: widening this to 64 (packing eight batches per tile instead of four) doubles
    # per-instruction PE occupancy exactly as intended -- 64x32 -> 64x64 tiles, 2796 -> 5563
    # elem/ns -- but REGRESSES e2e by +0.15 ms, because the wider PSUM banding and the
    # _build_swap_q_packed_col padding move cost onto Vector/DMA. PE work is free at the margin
    # on this kernel; do not re-try. See scheduling-experiment/FULL128_TUNING.md.
    is_packed_col = atp.s_active_qh < 32
    col_tile_width = 32 if is_packed_col else atp.s_active_qh
    batches_per_col_tile = 32 // atp.s_active_qh if is_packed_col else 1

    if is_packed_col:
        # When a query is narrower than a 32-wide column tile (s_active_qh < 32)
        # the swap MM1 stages Q as a padded buffer.
        kernel_assert(
            col_tile_width == 32,
            f"swap column tiling assumes col_tile_width == 32, got {col_tile_width}.",
        )
        q_pad_sb, left_zeros = _build_swap_q_packed_col(bufs, atp, cfg, TC, sbm, sprior_band_factor, col_tile_width)

    # Free size per bsq tile: each bsq tile spans the full FA s_prior.
    bsq_tile_sz = fa_ctx.tile_n_sprior * TC.p_max
    # How many element pack into K dtype: fp8_packed is 2 fp8 per bf16, bf16 is 1
    k_dtype_pack_factor = 2 if cfg.fp8_packed else 1
    # Row-tiling: route each K tile's matmul(s) to a separate PSUM bank by its row-tile position.
    mm1_grp_sz = TC.psum_f_max * atp.qk_row_tile_factor  # s_prior positions per mm1 group (across all banks)
    n_mm1_per_grp = mm1_grp_sz // TC.p_max  # nc_matmuls per mm1 group (across all banks)
    n_mm1_per_bank = n_mm1_per_grp // atp.qk_row_tile_factor  # nc_matmuls per PSUM bank
    # Positions refer to p_max wide gather transpose tiles
    n_pos_total = num_folds_this_tile * k_block_len_row_tile
    n_pos_per_band = n_pos_total // sprior_band_factor
    n_pos_per_mm1_grp = n_mm1_per_bank // k_dtype_pack_factor

    # Max positions per indirect dma_transpose. NOTE: this bounds the GATHER, not the matmul
    # tiling -- raising it to 128/256 leaves the QK matmul count and shape byte-identical (verified
    # in the profile), so it is inert for this config. Do not expect it to change QK efficiency.
    K_CAP = 64
    all_groups_full = band_s_prior % mm1_grp_sz == 0
    n_mm1_grps_total = div_ceil(band_s_prior, mm1_grp_sz)
    if sprior_band_factor > 1:
        # Banding: the batches_per_psum iterations filling one mm1 group read every band's position
        # sub-range (band b at offset b * n_pos_per_band), so the k_tile cannot be split by position --
        # the whole tile's K must stay resident. Keep a single k_tile (n_k_tiles=1, all mm1 groups); the
        # gather loop below still bounds each dma_transpose to K_CAP // gather_batches positions.
        mm1_grps_per_k_tile, n_k_tiles = n_mm1_grps_total, 1
    else:
        # cap: max positions per indirect dma_transpose. Split into k_tiles (contiguous position ranges).
        mm1_grps_per_k_tile, n_k_tiles = _compute_k_tile_gather(
            n_mm1_grps_total, k_block_len_row_tile, n_pos_per_mm1_grp, gather_batches, all_groups_full, cap=K_CAP
        )

    # Split the per-position max reduce across two engines: even evicts fuse the reduce into the Vector
    # select_reduce (accumulator column 0), odd evicts skip the Vector reduce and reduce the evicted tile
    # on the Scalar engine via activate2 (column 1). The two accumulators are combined per bsq tile. Breaks
    # the single serial Vector reduce chain. activate2 is gen4+; older HW falls back to the Vector chain.
    split_max_reduce = nisa.get_nc_version() >= nisa.nc_version.gen4

    sbm.open_scope(interleave_degree=batch_interleave_degree_safe, name="qk_matmul")
    # Loop over s_active_bqh tiles: each iteration is the batches whose scores fill one PMAX of output.
    for i_bsq_tile in range(atp.n_bsq_tiles):
        sbm.open_scope()
        bsq_tile_batch_start = i_bsq_tile * batches_per_psum

        atp.max_negated = False
        # Per-bsq-tile two-column max accumulator (col 0 = Vector, col 1 = Scalar) and running call index.
        if split_max_reduce:
            qk_max_2col = sbm.alloc_stack((s_active_bqh_banded_tile, 2), dtype=atp.inter_type, buffer=nl.sbuf)
        reduce_call_idx = 0
        vector_reduce_used = False
        scalar_reduce_used = False
        for i_k_tile in range(n_k_tiles):
            sbm.open_scope()
            k_tile_mm1_grp_start = i_k_tile * mm1_grps_per_k_tile
            k_tile_mm1_grps = min(mm1_grps_per_k_tile, n_mm1_grps_total - k_tile_mm1_grp_start)
            pos_start = k_tile_mm1_grp_start * n_pos_per_mm1_grp
            # Gather covers the FULL tile positions (both bands' ranges) so a psum tile can read any band's
            # position sub-range. n_k_tiles==1 under banding, so k_tile spans all n_pos_total positions.
            k_tile_n_pos = (
                n_pos_total
                if sprior_band_factor > 1
                else min(k_tile_mm1_grps * n_pos_per_mm1_grp, n_pos_total - pos_start)
            )
            # k_sb comes off the reusable stack, so the allocator hands it an address that the
            # previous layer's MoE down-projection output tile also used. That creates a WAR
            # (ANTI_DEPENDENCE, confirmed in Flow.parquet: 3 edges from down_projection_mx.py:915
            # into the MM1 gather) which pins the K gather-transposes AFTER the MoE finishes --
            # even though the gather reads K_cache through the block table and has no real data
            # dependence on MoE. That is what leaves RS-EP's ~19 us shadow empty. Giving k_sb a
            # dedicated buffer removes the aliasing so the gather can float into the shadow.
            # SBUF is only ~40% utilised, so the extra live range fits.
            if _os.environ.get("VLLM_NEURON_MEGA_KSB_NOALIAS"):
                k_sb = nl.ndarray(
                    (k_d_head_row_tile, k_tile_n_pos * gather_batches * TC.p_max),
                    dtype=nl.bfloat16 if cfg.fp8_packed else _k_sb_dtype,
                    buffer=nl.sbuf,
                    name=f"{sbm.get_name_prefix()}ksb_noalias_fa{fa_ctx.fa_tile_idx}"
                    f"_bt{btc.batch_tile_idx}_bsq{i_bsq_tile}_kt{i_k_tile}",
                )
            else:
                k_sb = sbm.alloc_stack(
                    (k_d_head_row_tile, k_tile_n_pos * gather_batches * TC.p_max),
                    dtype=nl.bfloat16 if cfg.fp8_packed else _k_sb_dtype,
                    buffer=nl.sbuf,
                    align=32,
                )
            k_sb_4d = k_sb.reshape((k_d_head_row_tile, 1, k_tile_n_pos, gather_batches * TC.p_max))
            # k_tile position window (banded: pos_start is 0 and the window is the whole tile).
            band_pos_start = 0 if sprior_band_factor > 1 else pos_start
            pos_end = band_pos_start + k_tile_n_pos

            # Standard row-tiled gather of gather_batches queries over [band_pos_start, pos_end). Identical
            # to the non-banded gather (banding does not change how K is loaded).
            first_fold = band_pos_start // k_block_len_row_tile
            n_loads = div_ceil(pos_end, k_block_len_row_tile) - first_fold
            # K_CAP bounds positions x gather_batches per dma_transpose. When the whole tile stays resident
            # (banding, n_k_tiles==1) a full fold's load can exceed the cap, so chunk each fold's positions.
            # Non-banded k_tiles are pre-sized under the cap, so max_pos_per_dma covers the fold in one shot.
            max_pos_per_dma = max(1, K_CAP // gather_batches)
            for i_load in range(n_loads):
                # Clip this k_tile's window to load i_load's fold: [load_start, load_start + load_n_pos).
                fold_base = (first_fold + i_load) * k_block_len_row_tile
                load_start = max(band_pos_start, fold_base)
                load_n_pos = min(pos_end, fold_base + k_block_len_row_tile) - load_start
                blks_u32 = bufs.active_blocks_sb_u32_fold_major[
                    :,
                    (first_fold + i_load) * atp.bs + bsq_tile_batch_start : (first_fold + i_load) * atp.bs
                    + bsq_tile_batch_start
                    + gather_batches,
                ]
                for chunk_start in range(0, load_n_pos, max_pos_per_dma):
                    chunk_n_pos = min(max_pos_per_dma, load_n_pos - chunk_start)
                    src_pos_in_fold = (load_start - fold_base) + chunk_start  # HBM read offset in the fold
                    dst_pos_in_tile = (load_start - band_pos_start) + chunk_start  # write offset within k_sb
                    nisa.dma_transpose(
                        dst=k_sb_4d[:, 0:1, dst_pos_in_tile : dst_pos_in_tile + chunk_n_pos, :],
                        src=k_prior_4d.ap(
                            [
                                [k_block_len_row_tile * k_d_head_row_tile, TC.p_max * gather_batches],
                                [1, 1],
                                [k_d_head_row_tile, chunk_n_pos],
                                [1, k_d_head_row_tile],
                            ],
                            offset=src_pos_in_fold * k_d_head_row_tile,
                            vector_offset=blks_u32,
                            indirect_dim=0,
                        ),
                        axes=(3, 1, 2, 0),
                        dge_mode=dge_mode.swdge,
                    )

            # On the final NC + last FA tile, stitch each batch's K_active onto the end of its K.
            last_fold_pos_start = (num_folds_this_tile - 1) * k_block_len_row_tile
            overlaps_last_fold = (
                pos_start < last_fold_pos_start + k_block_len_row_tile and last_fold_pos_start < pos_end
            )
            if atp.sprior_prg_id == atp.sprior_n_prgs - 1 and fa_ctx.is_last_fa_tile and overlaps_last_fold:
                k_tile_pos_in_last_fold = max(0, pos_start - last_fold_pos_start)
                n_pos_in_last_fold = min(pos_end, last_fold_pos_start + k_block_len_row_tile) - max(
                    pos_start, last_fold_pos_start
                )
                last_fold_base_in_k_tile = max(0, last_fold_pos_start - pos_start)
                # NOTE: loading happens at a k_tile granularity that can be smaller than a fold, so the last
                # fold may span multiple k_tiles; the stitch runs once per k_tile overlapping it.
                _stitch_k_active_swap(
                    k_sb,
                    bufs,
                    atp,
                    cfg,
                    TC,
                    btc,
                    i_bsq_tile * batches_per_psum,
                    gather_batches,
                    k_tile_pos_in_last_fold,
                    n_pos_in_last_fold,
                    last_fold_base_in_k_tile,
                    _k_sb_dtype,
                )

            # For each mm1 group in the k_tile: KQ^T into PSUM, then mask + max-reduce evict to bufs.qk.
            for i_mm1_grp_in_k_tile in range(k_tile_mm1_grps):
                i_mm1_grp = k_tile_mm1_grp_start + i_mm1_grp_in_k_tile
                # This mm1 group's s_prior span (< mm1_grp_sz for a partial last group) and the derived
                # matmul-tile / per-bank counts. Full groups keep the original n_mm1_per_grp / n_mm1_per_bank.
                mm1_grp_s_prior = min(mm1_grp_sz, band_s_prior - i_mm1_grp * mm1_grp_sz)
                mm1_grp_n_mm1_per_bank = mm1_grp_s_prior * n_mm1_per_bank // mm1_grp_sz
                # This mm1 group's position offset within the k_tile.
                mm1_grp_pos_in_k_tile = i_mm1_grp_in_k_tile * n_pos_per_mm1_grp

                # This mm1 group spans qk_row_tile_factor PSUM banks, one per row-tile half.
                qk_psum_banks = []
                for i_row_tile in range(atp.qk_row_tile_factor):
                    psum_addr = (
                        0,
                        ((i_mm1_grp * atp.qk_row_tile_factor + i_row_tile) % per_batch_interleave_degree)
                        * TC.psum_f_max_bytes,
                    )
                    qk_psum_banks.append(
                        nl.ndarray(
                            (TC.p_max, mm1_grp_n_mm1_per_bank * TC.p_max),
                            dtype=nl.float32,
                            buffer=nl.psum,
                            # Manual allocation + row tiling is unsupported.
                            address=None if sbm.is_auto_alloc() or i_row_tile == 1 else psum_addr,
                        )
                    )

                # MM1: one nc_matmul per (batch, row-tile) into this mm1 group's PSUM banks.
                # TODO: reorder to emit the matmuls of one timestep together. A "timestep" is a single
                # s_prior position: each column-tile batch (i_b_local) packs its s_active_qh at a distinct
                # 32-aligned partition slot (bsq_row_offset), and each row-tile half (row_idx) contracts a
                # distinct d_head partition slab. The current nest (i_b_local outer, row_idx inner) walks all
                # row-tile banks of one batch before moving to the next batch, interleaving distinct timesteps.
                # QK_TIMESTEP_ORDER (the reorder the TODO above proposes): emit row-tile
                # outer / batch inner so consecutive matmuls contract the same d_head slab
                # and reuse the loaded stationary rows (halves LDWEIGHTS at this site).
                # The inner row_idx loop must then run only the ONE row this pair owns --
                # running the full range there double-accumulates into the same PSUM bank.
                _qk_ts = bool(_os.environ.get("VLLM_NEURON_MEGA_QK_TIMESTEP_ORDER"))
                _qk_pairs = (
                    [(_r, _b) for _r in range(atp.qk_row_tile_factor) for _b in range(batches_per_psum)]
                    if _qk_ts
                    else [(None, _b) for _b in range(batches_per_psum)]
                )
                for _row_outer, i_b_local in _qk_pairs:
                    i_b = i_bsq_tile * batches_per_psum + i_b_local
                    batch_local, i_band = divmod(i_b_local, sprior_band_factor)
                    band_pos_off = i_band * n_pos_per_band  # this band's position offset into k_sb
                    if not is_packed_col:
                        # When Q fills its column tile completely read Q directly.
                        q_stationary = bufs.q_sb.reshape((k_d_head_row_tile, atp.bs_full, atp.s_active_qh))[
                            :, btc.global_batch_offset + i_bsq_tile * gather_batches + batch_local
                        ]
                    else:
                        # Slice a col_tile_width-wide window from the padded Q buffer: q_idx is the query's
                        # GLOBAL batch, out_band_idx is the GLOBAL output index (which 32-tile column offset).
                        # For non-banded, q_idx and out_band_idx are the same.
                        win_start = _swap_q_packed_col_window_start(
                            q_idx=btc.global_batch_offset + i_bsq_tile * gather_batches + batch_local,
                            out_band_idx=btc.global_batch_offset + i_b,
                            left_zeros=left_zeros,
                            atp=atp,
                        )
                        q_stationary = q_pad_sb[:, win_start : win_start + col_tile_width]
                    for row_idx in [_row_outer] if _qk_ts else range(atp.qk_row_tile_factor):
                        row_slice = nl.ds(row_idx * cfg.d_head, cfg.d_head)
                        # Coalesce tiles of 128 into 512 moving to saturate PE.
                        if cfg.fp8_packed:
                            fp8_n_mm1_per_bank = mm1_grp_n_mm1_per_bank // 2
                            fp8_pos_in_k_tile = band_pos_off + mm1_grp_pos_in_k_tile
                            # Un-permute the parity interleave of packed FP8 format
                            moving = (
                                k_sb.view(_k_sb_dtype)
                                .reshape((k_d_head_row_tile, k_tile_n_pos, gather_batches, TC.p_max, 2))[
                                    row_slice,
                                    fp8_pos_in_k_tile : fp8_pos_in_k_tile + fp8_n_mm1_per_bank,
                                    batch_local,
                                    :,
                                    :,
                                ]
                                .permute([0, 1, 3, 2])
                            )
                        else:
                            pos_off = band_pos_off + mm1_grp_pos_in_k_tile
                            moving = k_sb.reshape((k_d_head_row_tile, k_tile_n_pos, gather_batches, TC.p_max))[
                                row_slice,
                                pos_off : pos_off + mm1_grp_n_mm1_per_bank,
                                batch_local,
                                :,
                            ]

                        bsq_tile_idx = i_b_local // batches_per_col_tile
                        bsq_row_offset = bsq_tile_idx * col_tile_width
                        nisa.nc_matmul(
                            dst=qk_psum_banks[row_idx][
                                nl.ds(bsq_row_offset, col_tile_width), : mm1_grp_n_mm1_per_bank * TC.p_max
                            ],
                            stationary=q_stationary[row_slice, :],
                            moving=moving,
                            tile_position=(row_idx * cfg.d_head, bsq_row_offset),
                            tile_size=(cfg.d_head, col_tile_width),
                        )

                # De-interleaved evict, one per row-tile bank. dst_base advances by the full group stride
                # (mm1_grp_sz) but a partial last group only writes mm1_grp_s_prior columns per bank.
                dst_base = i_bsq_tile * bsq_tile_sz + i_mm1_grp * mm1_grp_sz
                dst_mm1_grp = bufs.qk[:, dst_base : dst_base + mm1_grp_s_prior].reshape_dim(
                    1,
                    [
                        mm1_grp_n_mm1_per_bank // k_dtype_pack_factor,
                        atp.qk_row_tile_factor,
                        k_dtype_pack_factor * TC.p_max,
                    ],
                )
                mask_mm1_grp = bufs.mask_sb[:, dst_base : dst_base + mm1_grp_s_prior].reshape_dim(
                    1,
                    [
                        mm1_grp_n_mm1_per_bank // k_dtype_pack_factor,
                        atp.qk_row_tile_factor,
                        k_dtype_pack_factor * TC.p_max,
                    ],
                )
                # De-interleaved evict, one select_reduce per row-tile bank.
                for i_row_tile in range(atp.qk_row_tile_factor):
                    evicted = dst_mm1_grp[:, :, i_row_tile]
                    if not split_max_reduce:
                        is_first_reduce = i_mm1_grp == 0 and i_row_tile == 0
                        nisa.select_reduce(
                            dst=evicted,
                            predicate=mask_mm1_grp[:, :, i_row_tile],
                            on_true=qk_psum_banks[i_row_tile][:, : mm1_grp_n_mm1_per_bank * TC.p_max],
                            on_false=-np.inf,
                            reduce_res=bufs.qk_max_buf[:, i_bsq_tile : i_bsq_tile + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce if is_first_reduce else nisa.reduce_cmd.reduce,
                            reduce_op=nl.max,
                        )
                    elif reduce_call_idx % 2 == 0:
                        # Even: fuse the max reduce into the Vector select_reduce (accumulator col 0).
                        nisa.select_reduce(
                            dst=evicted,
                            predicate=mask_mm1_grp[:, :, i_row_tile],
                            on_true=qk_psum_banks[i_row_tile][:, : mm1_grp_n_mm1_per_bank * TC.p_max],
                            on_false=-np.inf,
                            reduce_res=qk_max_2col[:, 0:1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce
                            if not vector_reduce_used
                            else nisa.reduce_cmd.reduce,
                            reduce_op=nl.max,
                        )
                        vector_reduce_used = True
                    else:
                        # Odd: evict only (no Vector reduce), then reduce the evicted tile on the Scalar
                        # engine via activate2 (accumulator col 1) so the two reduces run concurrently.
                        nisa.select_reduce(
                            dst=evicted,
                            predicate=mask_mm1_grp[:, :, i_row_tile],
                            on_true=qk_psum_banks[i_row_tile][:, : mm1_grp_n_mm1_per_bank * TC.p_max],
                            on_false=-np.inf,
                            reduce_cmd=nisa.reduce_cmd.idle,
                        )
                        nisa.activate2(
                            dst=evicted,
                            op=nl.copy,
                            data=evicted,
                            imm0=1.0,
                            imm1=0.0,
                            op0=nl.multiply,
                            op1=nl.add,
                            reduce_op=nl.max,
                            reduce_res=qk_max_2col[:, 1:2],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce
                            if not scalar_reduce_used
                            else nisa.reduce_cmd.reduce,
                        )
                        scalar_reduce_used = True
                    reduce_call_idx += 1

            sbm.close_scope()  # i_k_tile

        # Combine the two per-engine max accumulators into the single downstream max column. If only the
        # Vector engine ran (a single reduce call), col 0 already holds the full max -- just copy it.
        if split_max_reduce:
            if scalar_reduce_used:
                nisa.tensor_tensor(
                    bufs.qk_max_buf[:, i_bsq_tile : i_bsq_tile + 1],
                    qk_max_2col[:, 0:1],
                    qk_max_2col[:, 1:2],
                    op=nl.maximum,
                )
            else:
                nisa.tensor_copy(bufs.qk_max_buf[:, i_bsq_tile : i_bsq_tile + 1], qk_max_2col[:, 0:1])

        sbm.close_scope()  # i_bsq_tile
        sbm.increment_section()
    sbm.close_scope()  # qk_matmul

    _store_dbg_qk(DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)


def _fold_sink_and_update_max_swapped(sink, atp, cfg, TC, sbm, bufs, fa_ctx, btc):
    """
    QK_SWAP: the per-position max along s_prior is computed in _compute_kq_matmul_and_max_swapped
    (fused into the select_reduce evict, in bufs.qk_max_buf). Fold the sink into that max and
    update the FA running max (online softmax).
    """

    # Reduce banded max across partitions
    if _swap_band_factor(atp, TC) > 1:
        for col in range(atp.n_bsq_tiles):
            _band_combine(bufs.qk_max_buf[:, col : col + 1], bufs.qk_max_buf[:, col : col + 1], nl.maximum, atp, TC)

    atp.max_negated = False
    # Sharded paths defer cross-NC sync (and sink fold) to _finalize_and_store.
    if sink is not None and fa_ctx.fa_tile_idx == 0 and atp.sync_softmax_per_fa_tile:
        # Prep the per-bqh-row sink logit into qk_max_buf's reserved sink slot then fold
        # it into the per-position max (slot 0) so exp(sink - max) stays bounded (e.g. fully-masked tiles).
        sink_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        _prep_sink(
            sink, bufs.qk_max_buf[: atp.s_active_bqh_tile, nl.ds(sink_offset, atp.n_bsq_tiles)], atp, cfg, TC, sbm, btc
        )
        for i_bsq in range(atp.n_bsq_tiles):
            bsq_size = min(TC.p_max, atp.s_active_bqh - i_bsq * TC.p_max)
            nisa.tensor_tensor(
                bufs.qk_max_buf[:bsq_size, i_bsq : i_bsq + 1],
                bufs.qk_max_buf[:bsq_size, i_bsq : i_bsq + 1],
                bufs.qk_max_buf[:bsq_size, sink_offset + i_bsq : sink_offset + i_bsq + 1],
                op=nl.maximum,
            )

    # A fully-masked s_prior shard (banding + s_prior sharding, cache_len smaller than this shard's range)
    # leaves the per-position max at -inf; exp(qk - (-inf)) is NaN. Clamp to a finite bound so exp yields 0,
    _clamp_max_to_finite(
        dst=bufs.qk_max_buf[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        src=bufs.qk_max_buf[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        max_negated=False,
    )

    if atp.use_online_softmax:
        _update_running_max(atp, sbm, bufs, fa_ctx)


def _compute_exp_sum_and_transpose_swapped(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """
    QK_SWAP softmax stage. bufs.qk holds QK in [s_active_bqh, s_prior] layout; compute exp(qk - max) and
    the per-position sum-reduce along the free (s_prior) dim, then nc_transpose the exp to [s_prior, s_active_bqh]
    (bufs.qk_io_type) for the PV matmul.
    """

    band_factor = _swap_band_factor(atp, TC)
    s_active_bqh_banded_tile = atp.s_active_bqh_tile * band_factor
    n_bsq_banded_tiles = div_ceil(atp.s_active_bqh * band_factor, TC.p_max)
    band_n_sprior = fa_ctx.tile_n_sprior // band_factor

    qk_io_transposed = sbm.alloc_stack(
        (s_active_bqh_banded_tile, band_n_sprior * TC.p_max), dtype=atp.io_type, buffer=nl.sbuf
    )

    # Allocate outputs that downstream expects. exp_sum is sized to the BANDED partition so the exp's
    # fused per-position reduce can land a partial sum on every banded row; the band-combine below folds
    # those to the un-banded s_active_bqh rows that running_sum / exp_sum_recip consume.
    if cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1:
        padded_exp_sum_pdim = pad_partitions_for_ext_inst(s_active_bqh_banded_tile)
    else:
        padded_exp_sum_pdim = s_active_bqh_banded_tile
    bufs.exp_sum = sbm.alloc_stack(
        (padded_exp_sum_pdim, n_bsq_banded_tiles),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    if not atp.use_online_softmax:
        bufs.exp_sum_recip = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)

    # exp_grp_size: free-dim 128-tiles per exp instruction. Empirically ~1K free (8 x 128) balances:
    #  - exp's per-instruction overhead
    #  - interaction with PV transpose, whose free is capped at 128
    #  Finer exp chunks lets transposes start sooner while balancing pipelining time to be equal.
    exp_grp_size = 8
    n_exp_grps = div_ceil(band_n_sprior, exp_grp_size)
    tile_free = band_n_sprior * TC.p_max

    for i_bsq in range(n_bsq_banded_tiles):
        bsq_start = i_bsq * s_active_bqh_banded_tile
        bsq_size = min(s_active_bqh_banded_tile, atp.s_active_bqh * band_factor - bsq_start)

        # bufs.qk holds this bsq tile's QK in [s_active_bqh_banded, band_s_prior] layout.
        qk_transposed = bufs.qk[:, i_bsq * tile_free : (i_bsq + 1) * tile_free]

        # Step 1: Max for the exponential. The running/qk max is in the un-banded s_active_bqh (64) layout;
        # broadcast it up to the banded rows so exp subtracts each query's max from all its band rows.
        max_src = (
            bufs.running_max[:, i_bsq : i_bsq + 1] if atp.use_online_softmax else bufs.qk_max_buf[:, i_bsq : i_bsq + 1]
        )
        if band_factor > 1:
            qk_max_tile = nl.ndarray((bsq_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            _band_broadcast(max_src, qk_max_tile, atp, TC)
            if atp.max_negated:
                nisa.tensor_scalar(qk_max_tile, qk_max_tile, op0=nl.multiply, operand0=-1.0)
        elif atp.max_negated:
            qk_max_tile = nl.ndarray((bsq_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(qk_max_tile, max_src[:bsq_size, :], op0=nl.multiply, operand0=-1.0)
        else:
            qk_max_tile = max_src[:bsq_size, :]

        # Step 2: exp(qk - max) with fused per-position sum accumulation into exp_sum.
        # trn3 has the Vector-Engine nisa.exponential (subtracts max_value directly).
        # Other nc versions uses nisa.activation (which adds a bias before exp),
        # so negate the max first and pass it as bias.
        if not is_trn3_b1():
            neg_qk_max_tile = nl.ndarray((bsq_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(neg_qk_max_tile, qk_max_tile, op0=nl.multiply, operand0=-1.0)
        for i_grp in range(n_exp_grps):
            grp_start = i_grp * exp_grp_size * TC.p_max
            grp_free = min(exp_grp_size, band_n_sprior - i_grp * exp_grp_size) * TC.p_max
            if is_trn3_b1():
                nisa.exponential(
                    dst=qk_io_transposed[:bsq_size, grp_start : grp_start + grp_free],
                    src=qk_transposed[:bsq_size, grp_start : grp_start + grp_free],
                    max_value=qk_max_tile,
                    reduce_res=bufs.exp_sum[:bsq_size, i_bsq : i_bsq + 1],
                    reduce_cmd=reduce_cmd.reset_reduce if i_grp == 0 else reduce_cmd.reduce,
                )
            else:
                nisa.activation(
                    dst=qk_io_transposed[:bsq_size, grp_start : grp_start + grp_free],
                    op=nl.exp,
                    data=qk_transposed[:bsq_size, grp_start : grp_start + grp_free],
                    bias=neg_qk_max_tile,
                    reduce_op=nl.add,
                    reduce_res=bufs.exp_sum[:bsq_size, i_bsq : i_bsq + 1],
                    reduce_cmd=reduce_cmd.reset_reduce if i_grp == 0 else reduce_cmd.reduce,
                )

        # Step 2.5: fold the sink token into the denominator once (FA tile 0), on the unsharded
        # (per-tile-sync) path only. Sharded paths defer this to _finalize_and_store, which adds
        # exp(sink - global_max) to the global sum after the cross-NC exchange.
        if sink is not None and fa_ctx.fa_tile_idx == 0 and atp.sync_softmax_per_fa_tile:
            sink_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
            sink_exp = nl.ndarray((bsq_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                sink_exp,
                bufs.qk_max_buf[:bsq_size, sink_offset + i_bsq : sink_offset + i_bsq + 1],
                qk_max_tile,
                op=nl.subtract,
            )
            nisa.activation(sink_exp, nl.exp, sink_exp)
            nisa.tensor_tensor(
                bufs.exp_sum[:bsq_size, i_bsq : i_bsq + 1],
                bufs.exp_sum[:bsq_size, i_bsq : i_bsq + 1],
                sink_exp,
                op=nl.add,
            )

        # Step 3: Transpose exp back into qk_io_type [tile_n_sprior, s_active_bqh(un-banded)] for PV.
        # Group tp_grp_sz transposes into one wide PSUM bank and evict with a single strided copy.
        tp_grp_sz = max(1, TC.psum_f_max // bsq_size)
        qk_io_dst = bufs.qk_io_type.reshape_dim(1, [fa_ctx.tile_n_sprior, atp.s_active_bqh])
        if band_factor == 1:
            # Original path: qk_io_transposed rows are the un-banded s_active_bqh queries; transpose each
            # 128-wide s_prior chunk onto qk_io_type's query columns at this bsq tile's offset.
            for i_grp in range(div_ceil(band_n_sprior, tp_grp_sz)):
                sp_start = i_grp * tp_grp_sz
                grp_n = min(tp_grp_sz, band_n_sprior - sp_start)
                tp_psum = nl.ndarray((TC.p_max, grp_n * bsq_size), dtype=atp.io_type, buffer=nl.psum)
                tp_psum_3d = tp_psum.reshape((TC.p_max, grp_n, bsq_size))
                for j in range(grp_n):
                    i_sp = sp_start + j
                    src_slice = qk_io_transposed[:bsq_size, i_sp * TC.p_max : (i_sp + 1) * TC.p_max]
                    nisa.nc_transpose(tp_psum_3d[:, j, :], src_slice)
                dst_view = qk_io_dst[:, sp_start : sp_start + grp_n, bsq_start : bsq_start + bsq_size]
                nisa.tensor_copy(
                    dst_view, tp_psum_3d, engine=nisa.scalar_engine if is_trn3_b1() else nisa.vector_engine
                )
        else:
            # Banded (n_bsq_tiles==1): transpose the full 128 banded rows in 128x128 s_prior tiles (same
            # as the non-banded path). After transpose the PSUM free axis is the banded row order
            # (q, band, p) = [bs_per_nc, band_factor, sqh]. The evict de-bands in ONE strided copy: band
            # is an affine axis on both sides -- src middle axis, dst s_prior-tile offset (band occupies
            # tiles [band*band_n_sprior : (band+1)*band_n_sprior), this tile at band*band_n_sprior + i_sp).
            # View qk_io_dst as [p_max, band, tile_in_band, bs_per_nc, sqh], select this tile across bands,
            # and permute src to the matching [p_max, band, bs_per_nc, sqh] axis order.
            sqh = atp.s_active_qh
            qk_io_dst_banded = bufs.qk_io_type.reshape_dim(1, [band_factor, band_n_sprior, atp.bs_per_nc, sqh])
            # Group grp_n s_prior tiles' transposes into one wide PSUM bank, then evict the whole group in
            # one strided copy (band de-interleave folded into the dst AP) -- same op-count reduction as the
            # non-banded path, avoiding one transpose+copy per single s_prior position.
            for i_grp in range(div_ceil(band_n_sprior, tp_grp_sz)):
                sp_start = i_grp * tp_grp_sz
                grp_n = min(tp_grp_sz, band_n_sprior - sp_start)
                tp_psum = nl.ndarray((TC.p_max, grp_n * s_active_bqh_banded_tile), dtype=atp.io_type, buffer=nl.psum)
                tp_psum_g = tp_psum.reshape((TC.p_max, grp_n, atp.bs_per_nc, band_factor, sqh))
                for j in range(grp_n):
                    i_sp = sp_start + j
                    nisa.nc_transpose(
                        tp_psum.reshape((TC.p_max, grp_n, s_active_bqh_banded_tile))[:, j, :],
                        qk_io_transposed[:s_active_bqh_banded_tile, i_sp * TC.p_max : (i_sp + 1) * TC.p_max],
                    )
                # dst [p_max, band, grp_n s_prior tiles, bs_per_nc, sqh]; src permute (q,band)->(band,q).
                nisa.tensor_copy(
                    qk_io_dst_banded[:, :, sp_start : sp_start + grp_n, :, :],
                    tp_psum_g.permute([0, 3, 1, 2, 4]),  # [p_max, grp_n, q, band, sqh] -> [p_max, band, grp_n, q, sqh]
                    engine=nisa.scalar_engine if is_trn3_b1() else nisa.vector_engine,
                )

    # Reduce banded exp_sum across partitions
    if band_factor > 1:
        for col in range(atp.n_bsq_tiles):
            _band_combine(bufs.exp_sum[:, col : col + 1], bufs.exp_sum[:, col : col + 1], nl.add, atp, TC)

    _store_dbg_qk_exp(DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)

    # Compute reciprocal for output normalization (skip under online softmax — applied in finalize).
    if not atp.use_online_softmax:
        nisa.reciprocal(
            bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
            bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        )
        _s_active_bqh_tile_transpose_broadcast(bufs.exp_sum, bufs.exp_sum_recip, atp, TC)

    if atp.use_online_softmax:
        _update_running_sum(atp, sbm, bufs, fa_ctx)


def _cascaded_max_reduce(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 2. Cascaded max reduce of KQ^T"""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    bufs.qk_max = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf, align=4)

    use_engine_balancing_for_max_reduce = True
    if atp.s_prior <= 256:
        # For small S, engine balancing is inefficient.
        use_engine_balancing_for_max_reduce = False
    if nisa.get_nc_version() <= nisa.nc_version.gen3:
        # nisa.activate2 not available on older hardware.
        use_engine_balancing_for_max_reduce = False

    # Step 2.1. Strided reduce from [p_max, tile_n_sprior * bs * s_active_qh] -> [p_max, bs * s_active_qh]
    if not use_engine_balancing_for_max_reduce:
        # This is small (e.g. if n=2, s_a=6, s_p=8192, then free dim is 64*12=768), reasonable to be done with one inst
        qk_view = (bufs.qk).reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh]).permute([0, 2, 1])
        nisa.tensor_reduce(
            dst=bufs.qk_max, op=nl.maximum, data=qk_view, axis=[2], keepdims=False
        )  # The axis is modified here
    else:
        # Split across DVE (first half) and ACT/Scalar Engine (second half) for parallelism
        s_active_bqh_half = atp.s_active_bqh // 2
        s_active_bqh_first_half = atp.s_active_bqh - s_active_bqh_half  # ceiling half handles odd s_active_bqh
        s_active_bqh_second_half = s_active_bqh_half

        # First half with DVE (tensor_reduce)
        qk_view = (bufs.qk).reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh]).permute([0, 2, 1])
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
        padded_qk_max_pdim = pad_partitions_for_ext_inst(atp.s_active_bqh_tile)
    else:
        padded_qk_max_pdim = atp.s_active_bqh_tile

    bufs.qk_max_buf = sbm.alloc_stack(
        (padded_qk_max_pdim, atp.n_bsq_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    # Step 2.2 Transpose to psum -> [bs * s_active_qh, p_max]
    sbm.open_scope()
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _transpose_max_psum(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, bufs, sbm)

    if atp.s_active_bqh_remainder > 0:
        _transpose_max_psum(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, bufs, sbm)
    sbm.close_scope()

    # Step 2.3.1  If there is sink, load with the right layout.
    if should_prep_sink_in_cascade:
        # Stage sink into qk_max_buf for the per-tile final reduction below (single-NC path).
        sink_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        _prep_sink(
            sink,
            bufs.qk_max_buf[: atp.s_active_bqh_tile, nl.ds(sink_offset, atp.n_bsq_tiles)],
            atp,
            cfg,
            TC,
            sbm,
            btc,
        )
    # Step 2.3.3  Do the final reduction (2 or 3 reduce to 1) -> [bs * s_active_qh, 1]
    #             Negate if we are doing the reduction to save one op for sink exponential.
    # Do this only if syncing softmax per tile (not deferring to FA finalization)
    atp.max_negated = False
    tile_max = bufs.qk_max_buf[: atp.s_active_bqh_tile, : atp.n_bsq_tiles]
    if atp.softmax_final_reduction_length > 1 and atp.sync_softmax_per_fa_tile:
        atp.max_negated = True
        for i_bsq_tile in range(atp.n_bsq_tiles):
            qk_max_buf_view = (
                (bufs.qk_max_buf)
                .slice(0, start=0, end=atp.s_active_bqh_tile)
                .reshape_dim(1, [atp.softmax_final_reduction_length, atp.n_bsq_tiles])
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

    # Step 2.3.4 Update running max (if online softmax is used — FA or sharded)
    if atp.use_online_softmax:
        _update_running_max(atp, sbm, bufs, fa_ctx)

    # Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]
    # (Either running_max or qk_max_buf depending on whether online softmax is used)
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _transpose_broadcast_max(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, sbm, bufs)

    if atp.s_active_bqh_remainder > 0:
        _transpose_broadcast_max(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, sbm, bufs)

    if DBG_TENSORS and not atp.use_online_softmax and atp.bs == atp.bs_per_nc:
        # Non-online-softmax path (single-NC non-FA). qk_max_buf holds the final max; dump it
        # directly. Online-softmax paths dump from _finalize_and_store instead.
        local_slice = bufs.qk_max_buf[
            : atp.s_active_bqh_tile,
            nl.ds(atp.softmax_final_reduction_local_idx * atp.n_bsq_tiles, atp.n_bsq_tiles),
        ]
        _store_dbg_qk_max(local_slice, atp.max_negated, "cascaded", atp, TC, sbm, bufs)
    elif DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0 and atp.bs != atp.bs_per_nc:
        # Batch tiling active (offset-based writes into the debug tensor unreliable): write zeros
        # once on the first batch tile using the debug tensor's full-batch shape. The
        # online-softmax + full-batch case is handled by _finalize_and_store.
        _store_dbg_qk_max_zeros_full_batch(atp, sbm, bufs)


def _transpose_max_psum(
    index: int,
    tile_size: int,
    atp: AttnTileParams,
    TC: TileConstants,
    bufs: AttnInternalBuffers,
    sbm: SbufManager,
):
    """
    Step 2.2 Transpose to psum -> [bs * s_active_qh, p_max]
    Step 2.3.0 Reduce the new 128 fdim while copying to sbuf -> [bs * s_active_qh, 1]
    """
    # Step 2.2
    qk_max_psum = nl.ndarray(
        (tile_size, TC.p_max),
        dtype=atp.inter_type,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )
    nisa.nc_transpose(
        qk_max_psum,
        bufs.qk_max[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
    )

    # Step 2.3.0
    nisa.tensor_reduce(
        bufs.qk_max_buf[
            :tile_size,
            atp.n_bsq_tiles * atp.softmax_final_reduction_local_idx + index,
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
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]"""
    sbm.open_scope()

    # Use running max when online softmax is active (FA or sharded), else tile max
    if atp.use_online_softmax:
        max_src_tensor = bufs.running_max[:tile_size]
    else:
        max_src_tensor = bufs.qk_max_buf[:tile_size]

    # Transpose column `index` and broadcast onto all p_max partitions. Native nc_transpose (not
    # tp_broadcast) so a strided running_max slice keeps its real partition stride.
    tp_psum = nl.ndarray(
        (TC.p_max, tile_size), dtype=nl.float32, buffer=nl.psum, address=None if sbm.is_auto_alloc() else (0, 0)
    )
    nisa.nc_transpose(tp_psum, max_src_tensor[:, index : index + 1].broadcast(1, TC.p_max))
    nisa.tensor_copy(bufs.qk_max[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)], tp_psum)
    sbm.close_scope()


def _compute_exp_qk(
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 3. Exp(KQ^T - max(KQ^T))"""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    # Instruction startup time on TRN2 does not outweigh pipelining advantages
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        for i_s_prior in range(fa_tile_n_sprior):
            qk_view = (
                (bufs.qk)
                .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
                .slice(1, start=i_s_prior, end=i_s_prior + 1)
            )

            nisa.tensor_tensor(qk_view, qk_view, bufs.qk_max, op=(nl.add if atp.max_negated else nl.subtract))

            qk_io_type_view = (
                (bufs.qk_io_type)
                .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
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

    _store_dbg_qk_exp(DBG_TENSORS, atp, cfg, TC, sbm, bufs, fa_ctx, btc)


def _cascaded_sum_reduction(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 4. Cascaded sum reduction of exp"""
    fa_tile_n_sprior = fa_ctx.tile_n_sprior

    if cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_sum_pdim = pad_partitions_for_ext_inst(atp.s_active_bqh_tile)
    else:
        padded_exp_sum_pdim = atp.s_active_bqh_tile
    bufs.exp_sum = sbm.alloc_stack(
        (padded_exp_sum_pdim, atp.n_bsq_tiles * atp.softmax_final_reduction_length),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )
    if not atp.use_online_softmax:
        # When online softmax is active, reciprocal is applied in _finalize_and_store
        bufs.exp_sum_recip = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)

    sbm.open_scope()
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_sum_reduction(i_bsq_tile, atp.s_active_bqh_tile, fa_tile_n_sprior, atp, TC, bufs, sbm)
    sbm.close_scope()

    if atp.s_active_bqh_remainder > 0:
        sbm.open_scope()
        _tile_sum_reduction(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, fa_tile_n_sprior, atp, TC, bufs, sbm)
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
        reduction_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        for i_bsq_tile in range(atp.n_bsq_tiles):
            # Use running max when online softmax is active (FA single-NC), else tile max from qk_max_buf
            max_buf_for_sink = bufs.running_max if atp.use_online_softmax else bufs.qk_max_buf
            nisa.tensor_scalar(
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.add,
                max_buf_for_sink[: atp.s_active_bqh_tile, i_bsq_tile],
            )
            nisa.activation(
                bufs.exp_sum[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.exp,
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
            )

    if atp.softmax_final_reduction_length > 1 and atp.sync_softmax_per_fa_tile:
        for i_bsq_tile in range(atp.n_bsq_tiles):
            exp_sum_view = (
                (bufs.exp_sum)
                .slice(0, start=0, end=atp.s_active_bqh_tile)
                .reshape_dim(1, [atp.softmax_final_reduction_length, atp.n_bsq_tiles])
                .select(2, i_bsq_tile)
            )
            nisa.tensor_reduce(
                bufs.exp_sum[: atp.s_active_bqh_tile, i_bsq_tile],
                data=exp_sum_view,
                op=nl.add,
                axis=1,
            )

    if atp.use_online_softmax:
        _update_running_sum(atp, sbm, bufs, fa_ctx)

    if DBG_TENSORS and fa_ctx.is_last_fa_tile and atp.bs == atp.bs_per_nc and not atp.use_online_softmax:
        # Non-online-softmax path (single-NC non-FA). exp_sum holds the final sum; dump it directly.
        # Online-softmax paths dump from _finalize_and_store where running_sum is the
        # globally-synced sum.
        local_slice = bufs.exp_sum[
            : atp.s_active_bqh_tile,
            nl.ds(atp.softmax_final_reduction_local_idx * atp.n_bsq_tiles, atp.n_bsq_tiles),
        ]
        _store_dbg_exp_sum(local_slice, "cascaded", atp, TC, sbm, bufs)
    elif DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0 and atp.bs != atp.bs_per_nc:
        # Batch tiling active: write zeros with full-batch shape so the tensor is defined.
        _store_dbg_exp_sum_zeros_full_batch(atp, sbm, bufs)

    # Skip reciprocal when online softmax is active — reciprocal is applied in _finalize_and_store.
    if atp.use_online_softmax:
        return

    # Take sum recip, transpose and broadcast on pdim
    nisa.reciprocal(
        bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
    )

    _s_active_bqh_tile_transpose_broadcast(bufs.exp_sum, bufs.exp_sum_recip, atp, TC)


def _tile_sum_reduction(
    index, tile_size, tile_n_sprior, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers, sbm: SbufManager
):
    """
    Step 4.1. Each of the tile_n_sprior matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    Step 4.2. Copy partial reduce output from psum -> sb while reducing the free dim (num_sprior_t128)
    tile_size is either atp.s_active_bqh_tile or atp.s_active_bqh_remainder
    """
    sum_reduce_psum = nl.ndarray(
        (tile_size, tile_n_sprior),
        dtype=nl.float32,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )

    # Step 4.1. Each of the tile_n_sprior matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    for i_exp_reduce in range(tile_n_sprior):
        sum_reduce_psum_view = (sum_reduce_psum).slice(1, start=i_exp_reduce, end=i_exp_reduce + 1)
        s_active_bqh_pos = index * atp.s_active_bqh_tile
        qk_io_type_view = (
            (bufs.qk_io_type)
            .reshape_dim(1, [tile_n_sprior, atp.s_active_bqh])
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
            atp.n_bsq_tiles * atp.softmax_final_reduction_local_idx + index,
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


def _s_active_bqh_tile_transpose_broadcast_d_tiled(src, dst, atp: AttnTileParams, TC: TileConstants):
    """Transpose and broadcast for d_head-tiled layout.

    Same as _s_active_bqh_tile_transpose_broadcast but for dst with shape
    [d_tile_size, n_d_tiles * s_active_bqh]. Transposes src to partition 0 of the first
    d_tile chunk, broadcasts within that chunk, then replicates to all d_tile chunks.

    Args:
        src: Source tensor with shape [s_active_bqh_tile, n_bsq_tiles]
        dst: Destination tensor with shape [d_tile_size, n_d_tiles * s_active_bqh]
    """
    # Transpose into the first d_tile chunk [0:1, 0:s_active_bqh]
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _column_tile_transpose(src, dst, i_bsq_tile, atp.s_active_bqh_tile, atp.s_active_bqh_tile, TC)
    if atp.s_active_bqh_remainder > 0:
        _column_tile_transpose(src, dst, atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp.s_active_bqh_tile, TC)
    # Broadcast partition 0 to all d_tile_size partitions within first chunk
    stream_shuffle_broadcast(src=dst[:1, : atp.s_active_bqh], dst=dst[: atp.d_tile_size, : atp.s_active_bqh])
    # Replicate first d_tile chunk to all remaining d_tile chunks
    for i_d in range(1, atp.n_d_tiles):
        d_offset = i_d * atp.s_active_bqh
        nisa.tensor_copy(
            dst[:, d_offset : d_offset + atp.s_active_bqh],
            dst[:, : atp.s_active_bqh],
        )


def _compute_pv_matmul_and_store(
    v_prior,
    v_active,
    out,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx: FATileContext,
    btc: BatchTileContext,
):
    """Step 5. Matmult 2 of (exp @ V)^T and store output"""
    fa_tile_s_prior = fa_ctx.tile_s_prior
    fa_tile_n_sprior = fa_ctx.tile_n_sprior
    fa_tile_offset = fa_ctx.tile_offset
    is_last_fa_tile = fa_ctx.is_last_fa_tile

    if atp.exp_v_sendrecv_gpsimd:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_v_pdim = pad_partitions_for_ext_inst(atp.d_tile_size)
    else:
        padded_exp_v_pdim = atp.d_tile_size
    # For d_head tiling: exp_v is [d_tile_size, n_d_tiles * bs * s_active_qh]
    bufs.exp_v = sbm.alloc_stack(
        (padded_exp_v_pdim, atp.n_d_tiles * atp.bs * atp.s_active_qh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(
        cfg.d_head * fa_tile_n_sprior * sizeinbytes(v_prior.dtype), atp.batch_interleave_degree, sbm
    )

    """
    Tiling Strategy for MM2 ((exp @ V)^T computation and output):
    - V stationary: [s_prior, d_head] loaded per batch into v_sb as [P_MAX, n_sprior_tile * d_head]
    - exp(QK) moving: [P_MAX, s_active_bqh] from qk_io_type (already computed and normalized)
    - Output: exp_v [d_tile_size, n_d_tiles * bs * s_active_qh] accumulated in PSUM then copied to SBUF
    - PSUM allocation: [d_tile_size, s_active_qh] per batch per d_tile
    - Memory layout: V loaded horizontally tiled (strided if strided_mm1=False, sequential if True)
    - Batch interleaving: Uses batch_interleave_degree_safe for DMA/compute overlap
    - Final output: Gathered across cores if sprior_n_prgs > 1, then stored to HBM or kept in SBUF
    """

    # V-load DMA batching setup
    V_CAP = 128  # cap effective block size after batching to keep V-load memory reasonable
    v_block_len = atp.block_len if atp.is_block_kv else 0
    v_dma_batch_n_folds = 1
    v_dma_batch_n_batches = 1
    v_idx_table = None
    if atp.is_block_kv and v_block_len > 0:
        v_dma_batch_n_folds, v_dma_batch_n_batches = _compute_dma_batch_params(
            atp.num_folds_per_batch, atp.bs, v_block_len, cap=V_CAP, sbm=sbm
        )
    v_dma_batch_size = v_dma_batch_n_folds * v_dma_batch_n_batches
    # Array tiling on sprior: pack (p_max // d_head) tiles into the PE array per outer iteration.
    # Two possible schemes: HW array tiling (col tiling) and dense array tiling (manual).
    # Only enabled for longer fa_tile_sprior to amortize sprior tiling costs without regression.
    # Block KV only: both schemes assume the block-KV V-load layout in v_sb. The flat-KV
    # (strided_mm1) load produces a different v_sb layout that the tile packing has not been
    # validated against, so array tiling is disabled there.
    _PV_ARRAY_TILING_THRESHOLD = 8192
    pv_array_tiling_factor = TC.p_max // cfg.d_head
    pv_array_tiling_ok = (
        atp.is_block_kv
        and cfg.d_head < TC.p_max
        and cfg.d_head % 32 == 0
        and fa_tile_s_prior >= _PV_ARRAY_TILING_THRESHOLD
    )
    # Dense tiling is preferred where usable (fewer instructions, never slower than col in testing).
    # Its grid PSUM is [g*d_head, g*s_active_qh], so the free dim must also fit psum_f_max. When above
    # that (s_active_qh > psum_f_max / factor) fall back to HW col tiling, whose free dim is s_active_qh.
    # At EP128/sl10240 fa_tile_s_prior is below the 8192 threshold, so PV runs untiled here.
    # Lowering the threshold to enable dense tiling was measured twice and lost (1024: tie at
    # 5.930 vs 5.923; 2048: 6.079). Leave the threshold at 8192.
    pv_dense_factor = (
        pv_array_tiling_factor
        if (pv_array_tiling_ok and pv_array_tiling_factor * atp.s_active_qh <= TC.psum_f_max)
        else 1
    )
    # HW col tiling. Mutually exclusive with dense tiling.
    pv_col_tile_factor = pv_array_tiling_factor if (pv_dense_factor == 1 and pv_array_tiling_ok) else 1
    v_idx_src = bufs.active_blocks_sb if atp.use_v_dma_skipping else bufs.active_blocks_sb_u32
    if v_dma_batch_size > 1:
        # V's batched dma_copy reads a [128, N] vector_offset in column-major ("snake") order, so the
        # index table must be pre-rearranged into that layout. The K path uses dma_transpose instead,
        # which consumes a contiguous slice of active_blocks_sb_u32 directly and needs no rearrange.
        v_idx_total_columns = atp.bs * atp.num_folds_per_batch
        v_idx_cache_key = None
        if _os.environ.get("VLLM_NEURON_MEGA_ABT_REUSE") and fa_ctx.dynamic_tile_offset_sbuf is None:
            v_idx_cache_key = (id(v_idx_src), v_idx_total_columns, v_dma_batch_size)
            v_idx_table = _ABT_V_REARRANGED_CACHE.get(v_idx_cache_key)
        if v_idx_table is None:
            v_idx_table = _rearrange_indices_for_batched_dma(v_idx_src, v_idx_total_columns, v_dma_batch_size, TC, sbm)
            if v_idx_cache_key is not None:
                _ABT_V_REARRANGED_CACHE[v_idx_cache_key] = v_idx_table

    # FP8 KV: use the resolved FP8 dtype (caller's concrete dtype, or
    # dtype_mode resolution for opaque "float8e4"). Otherwise pass through v_prior.dtype.
    _v_sb_dtype = atp.kv_e4m3_tile_dtype if atp.is_fp8_kv else v_prior.dtype
    v_sb_shared = None

    sbm.open_scope(interleave_degree=batch_interleave_degree_safe, name="pv_matmul")
    for i_b in range(atp.bs):
        # Load V_prior from HBM [s_prior, d_head] into SB [128, (tile_s_prior / 128) * d_head]
        # Do strided load (horizontal tile) if not strided_mm1, otherwise load sequentially for better DMA throughput
        # V buffer allocation: shared for batch-batching, per-batch otherwise.
        per_batch_v_size = cfg.d_head * fa_tile_n_sprior
        if v_dma_batch_n_batches > 1:
            b_in_group = i_b % v_dma_batch_n_batches
            if b_in_group == 0:
                v_sb_shared = sbm.alloc_stack(
                    (TC.p_max, per_batch_v_size * v_dma_batch_n_batches), dtype=_v_sb_dtype, buffer=nl.sbuf
                )
            v_sb = (v_sb_shared).slice(1, start=b_in_group * per_batch_v_size, end=(b_in_group + 1) * per_batch_v_size)
        else:
            v_sb = sbm.alloc_stack((TC.p_max, per_batch_v_size), dtype=_v_sb_dtype, buffer=nl.sbuf)
        v_sb_view = (v_sb).reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
        if atp.is_block_kv:
            # For FA, compute which folds correspond to this tile
            fold_s_prior = atp.block_len * TC.p_max
            fold_start = fa_tile_offset // fold_s_prior
            fold_end = div_ceil(fa_tile_offset + fa_tile_s_prior, fold_s_prior)
            num_folds_this_tile = fold_end - fold_start

            if atp.use_v_dma_skipping:
                # This memset is required for oob skip to prevent uninitialized NaNs from corrupting results.
                # With batch-batching the load targets the whole v_sb_shared on the leading batch, so memset
                # the full shared buffer once (on b_in_group == 0) and skip non-leading batches to avoid
                # zeroing data the leading-batch load already wrote.
                if v_dma_batch_n_batches > 1:
                    if b_in_group == 0:
                        nisa.memset(v_sb_shared, value=0)
                else:
                    nisa.memset(v_sb, value=0)

            sbm.open_scope()
            for i_fold_rel in range(num_folds_this_tile):
                i_fold = fold_start + i_fold_rel
                if v_dma_batch_n_folds > 1 and i_fold_rel % v_dma_batch_n_folds != 0:
                    continue
                if v_dma_batch_n_batches > 1 and (i_b % v_dma_batch_n_batches) != 0:
                    continue

                idx_start = i_b * atp.num_folds_per_batch + i_fold_rel

                # Indices: use rearranged table when batching (v_dma_batch_size > 1), direct slice otherwise.
                if v_dma_batch_size > 1:
                    v_idx_slice = (v_idx_table).slice(dim=1, start=idx_start, end=idx_start + v_dma_batch_size)
                else:
                    v_idx_slice = v_idx_src.slice(dim=1, start=idx_start, end=idx_start + 1)

                load_len = v_dma_batch_size * atp.block_len * cfg.d_head
                if v_dma_batch_n_batches > 1:
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
                    name=f"{sbm.get_name_prefix()}v_prior_block_load_indirect_fa{fa_ctx.fa_tile_idx}_b{i_b}_f{i_fold}_bt{btc.batch_tile_idx}",
                )
            sbm.close_scope()
        elif cfg.strided_mm1:
            s_prior_pos = fa_tile_offset
            v_prior_view = (
                (v_prior)
                .select(0, btc.global_batch_offset + i_b)
                .squeeze_dim(0)
                .slice(0, start=s_prior_pos, end=s_prior_pos + (TC.p_max * fa_tile_n_sprior))
                .reshape_dim(0, [TC.p_max, fa_tile_n_sprior])
            )
            nisa.dma_copy(
                v_sb_view,
                v_prior_view,
                name=f"{sbm.get_name_prefix()}v_prior_load_strided_mm1_fa{fa_ctx.fa_tile_idx}_b{i_b}_bt{btc.batch_tile_idx}",
            )
        else:
            s_prior_pos = fa_tile_offset
            v_prior_view = (
                (v_prior)
                .select(0, btc.global_batch_offset + i_b)
                .squeeze_dim(0)
                .slice(0, start=s_prior_pos, end=s_prior_pos + (TC.p_max * fa_tile_n_sprior))
                .reshape_dim(0, [fa_tile_n_sprior, TC.p_max])
                .permute((1, 0, 2))
            )
            nisa.dma_copy(
                dst=v_sb_view,
                src=v_prior_view,
                name=f"{sbm.get_name_prefix()}v_prior_load_sequential_fa{fa_ctx.fa_tile_idx}_b{i_b}_bt{btc.batch_tile_idx}",
            )

        # Load V_active to the last portion if needed (only on last FA tile)
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1 and is_last_fa_tile:
            if atp.is_block_kv:
                num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
                extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active

                v_sb_partition_base = TC.p_max - num_blks_covering_s_active
                v_sb_s_prior_base = (num_folds_this_tile - 1) * atp.block_len

                v_active_reshaped_batch_pos = btc.global_batch_offset + i_b

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

                        v_active_reshaped_view = (
                            (bufs.v_active_reshaped)
                            .slice(0, start=v_active_reshaped_batch_pos, end=v_active_reshaped_batch_pos + 1)
                            .reshape_dim(1, [cfg.s_active, cfg.d_head])
                            .slice(1, start=0, end=atp.block_len - extra_covered)
                        )
                        nisa.dma_copy(
                            dst=v_sb_view,
                            src=v_active_reshaped_view,
                            name=f"{sbm.get_name_prefix()}v_active_block_load_partial_rows_b{i_b}_bt{btc.batch_tile_idx}",
                        )
                    if num_blks_covering_s_active > 1:
                        v_sb_view = (
                            (v_sb)
                            .slice(
                                0, start=v_sb_partition_base + 1, end=v_sb_partition_base + num_blks_covering_s_active
                            )
                            .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                            .slice(1, start=v_sb_s_prior_base, end=v_sb_s_prior_base + atp.block_len)
                        )

                        s_active_pos = atp.block_len - extra_covered
                        v_active_reshaped_view = (
                            (bufs.v_active_reshaped)
                            .select(
                                0,
                                v_active_reshaped_batch_pos,
                            )
                            .reshape_dim(0, [cfg.s_active, cfg.d_head])
                            .slice(
                                0,
                                start=s_active_pos,
                                end=s_active_pos + atp.block_len * (num_blks_covering_s_active - 1),
                            )
                            .reshape_dim(0, [(num_blks_covering_s_active - 1), atp.block_len])
                        )
                        nisa.dma_copy(
                            dst=v_sb_view,
                            src=v_active_reshaped_view,
                            name=f"{sbm.get_name_prefix()}v_active_block_load_remaining_blocks_b{i_b}_bt{btc.batch_tile_idx}",
                        )
                else:
                    v_sb_view = (
                        (v_sb)
                        .slice(0, start=v_sb_partition_base, end=v_sb_partition_base + num_blks_covering_s_active)
                        .reshape_dim(1, [fa_tile_n_sprior, cfg.d_head])
                        .slice(1, start=v_sb_s_prior_base, end=v_sb_s_prior_base + atp.block_len)
                    )

                    v_active_reshaped_view = (
                        (bufs.v_active_reshaped)
                        .select(
                            0,
                            v_active_reshaped_batch_pos,
                        )
                        .reshape_dim(0, [cfg.s_active, cfg.d_head])
                        .slice(0, start=0, end=atp.block_len * num_blks_covering_s_active)
                        .reshape_dim(0, [num_blks_covering_s_active, atp.block_len])
                    )

                    nisa.dma_copy(
                        dst=v_sb_view,
                        src=v_active_reshaped_view,
                        name=f"{sbm.get_name_prefix()}v_active_block_load_full_b{i_b}_bt{btc.batch_tile_idx}",
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
                        (v_active).select(0, btc.global_batch_offset + i_b).slice(1, start=0, end=load1_nrows)
                    )

                    nisa.dma_copy(
                        v_sb_view,
                        v_active_view,
                        name=f"{sbm.get_name_prefix()}v_active_strided_load_partial_b{i_b}_bt{btc.batch_tile_idx}",
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
                        .select(0, btc.global_batch_offset + i_b)
                        .squeeze_dim(0)
                        .slice(0, start=load1_nrows, end=load1_nrows + load2_nrows)
                        .reshape_dim(0, [load2_nrows // fa_tile_n_sprior, fa_tile_n_sprior])
                    )

                    nisa.dma_copy(
                        v_sb_view,
                        v_active_view,
                        name=f"{sbm.get_name_prefix()}v_active_strided_load_remaining_b{i_b}_bt{btc.batch_tile_idx}",
                    )
            else:
                v_active_view = (v_active).select(0, btc.global_batch_offset + i_b).squeeze_dim(0)
                # Load to the bottom right part of last chunk of v_sb: [s_active, d_head]
                nisa.dma_copy(
                    v_sb[TC.p_max - cfg.s_active :, v_sb.shape[1] - cfg.d_head :],
                    v_active_view,
                    name=f"{sbm.get_name_prefix()}v_active_load_sequential_b{i_b}_bt{btc.batch_tile_idx}",
                )

        # Perform V^T @ exp^T, which equals to (exp @ V)^T. Recall mm1 output is transposed - KQ^T
        # For d_head tiling: compute one d_tile at a time, accumulating across sprior tiles per d_tile.
        # Array tiling on s_prior packs pv_tile_factor tiles into the array per outer iteration:
        #  - col tiling: one matmul per tile via tile_position into partition bands; PSUM [d*factor, s_qh].
        #  - dense tiling: one matmul for all tiles, both stationary and moving densely packed.
        #    Valid data in PSUM is on the diagonal blocks.
        dense_tiling = pv_dense_factor > 1
        pv_tile_factor = pv_dense_factor if dense_tiling else pv_col_tile_factor
        # Per-tile stride along the PSUM free axis between valid blocks: dense results walk the grid
        # diagonal (stride s_active_qh), col results all share the same free columns (stride 0).
        psum_free_stride = atp.s_active_qh if dense_tiling else 0
        for i_d in range(atp.n_d_tiles):
            exp_v_psum = nl.ndarray(
                (atp.d_tile_size * pv_tile_factor, atp.s_active_qh * (pv_tile_factor if dense_tiling else 1)),
                dtype=nl.float32,
                buffer=nl.psum,
                address=None
                if sbm.is_auto_alloc()
                else (0, (i_b % batch_interleave_degree_safe) * TC.psum_f_max_bytes),
            )
            for i_t in range(0, fa_tile_n_sprior, pv_tile_factor):
                n_col_tiles = min(pv_tile_factor, fa_tile_n_sprior - i_t)
                batch_s_active_qh_pos = i_b * atp.s_active_qh
                if dense_tiling:
                    # Densely pack n_col_tiles stationary [V_i_t|...] + strided moving [P_i_t|...] into
                    # one matmul; results land on the diagonal blocks of the grid PSUM.
                    v_sb_d_view = v_sb[:, i_t * cfg.d_head : (i_t + n_col_tiles) * cfg.d_head]
                    qk_io_type_view = (
                        (bufs.qk_io_type)
                        .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
                        .slice(1, start=i_t, end=i_t + n_col_tiles)
                        .slice(2, start=batch_s_active_qh_pos, end=batch_s_active_qh_pos + atp.s_active_qh)
                    )
                    nisa.nc_matmul(
                        exp_v_psum[: n_col_tiles * atp.d_tile_size, : n_col_tiles * atp.s_active_qh],
                        stationary=v_sb_d_view,
                        moving=qk_io_type_view,
                    )
                else:
                    for i_col in range(n_col_tiles):
                        v_tile_offset = (i_t + i_col) * cfg.d_head + i_d * atp.d_tile_size
                        v_sb_d_view = v_sb[:, v_tile_offset : v_tile_offset + atp.d_tile_size]
                        qk_io_type_view = (
                            (bufs.qk_io_type)
                            .reshape_dim(1, [fa_tile_n_sprior, atp.s_active_bqh])
                            .select(1, i_t + i_col)
                            .slice(1, start=batch_s_active_qh_pos, end=batch_s_active_qh_pos + atp.s_active_qh)
                        )
                        nisa.nc_matmul(
                            exp_v_psum[i_col * atp.d_tile_size : (i_col + 1) * atp.d_tile_size, :],
                            stationary=v_sb_d_view,
                            moving=qk_io_type_view,
                            tile_size=(TC.p_max, atp.d_tile_size),
                            tile_position=(0, i_col * atp.d_tile_size),
                        )

            # Fold partition slices: sum all column tiles' results together.
            # Only fold slices that were actually written (handles fa_tile_n_sprior < pv_tile_factor).
            actual_col_tiles_used = min(pv_tile_factor, fa_tile_n_sprior)
            # Copy mm2 output from psum -> sb while multiplying recip(sum)
            # When online softmax is active, we don't multiply by recip(sum) here — that's done in finalize.
            # exp_v layout: [d_tile_size, n_d_tiles * bs * s_active_qh]
            exp_v_offset = i_d * atp.bs * atp.s_active_qh + i_b * atp.s_active_qh
            exp_v_view = bufs.exp_v[: atp.d_tile_size, exp_v_offset : exp_v_offset + atp.s_active_qh]
            if atp.use_online_softmax:
                nisa.tensor_copy(exp_v_view, exp_v_psum[0 : atp.d_tile_size, 0 : atp.s_active_qh])
                for i_col in range(1, actual_col_tiles_used):
                    nisa.tensor_tensor(
                        exp_v_view,
                        exp_v_view,
                        exp_v_psum[
                            i_col * atp.d_tile_size : (i_col + 1) * atp.d_tile_size,
                            i_col * psum_free_stride : i_col * psum_free_stride + atp.s_active_qh,
                        ],
                        op=nl.add,
                    )
            else:
                exp_sum_recip_view = (
                    (bufs.exp_sum_recip)
                    .reshape_dim(1, [atp.bs, atp.s_active_qh])
                    .select(1, i_b)
                    .slice(0, start=0, end=atp.d_tile_size)
                )
                nisa.tensor_tensor(
                    exp_v_view, exp_v_psum[0 : atp.d_tile_size, 0 : atp.s_active_qh], exp_sum_recip_view, op=nl.multiply
                )
                if actual_col_tiles_used > 1:
                    sbm.open_scope()
                    exp_v_second = sbm.alloc_stack(
                        (atp.d_tile_size, atp.s_active_qh), dtype=atp.inter_type, buffer=nl.sbuf
                    )
                    for i_col in range(1, actual_col_tiles_used):
                        nisa.tensor_tensor(
                            exp_v_second,
                            exp_v_psum[
                                i_col * atp.d_tile_size : (i_col + 1) * atp.d_tile_size,
                                i_col * psum_free_stride : i_col * psum_free_stride + atp.s_active_qh,
                            ],
                            exp_sum_recip_view,
                            op=nl.multiply,
                        )
                        nisa.tensor_tensor(exp_v_view, exp_v_view, exp_v_second, op=nl.add)
                    sbm.close_scope()
        sbm.increment_section()
    sbm.close_scope()

    # When online softmax is active, accumulate the tile output into running_output and defer the
    # cross-NC gather+store to _finalize_and_store. Otherwise, the per-tile reciprocal above has
    # already normalized exp_v — gather-and-store it directly.
    if atp.use_online_softmax:
        _accumulate_output(atp, cfg, TC, sbm, bufs, fa_ctx)
    else:
        _gather_and_store_output(out, bufs.exp_v, atp, cfg, sbm, btc)


def _gather_and_store_output(
    out: nl.NkiTensor,
    res: nl.NkiTensor,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    btc: BatchTileContext,
):
    """Gather partial results from other NC if sharded, then store output to HBM/SBUF.

    Args:
        out: Output tensor in HBM/SBUF
        res: Result tensor in SBUF with shape [d_tile_size, n_d_tiles * s_active_bqh]
             (same as [d_head, s_active_bqh] when n_d_tiles == 1)
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
            dma_engine=dma_engine.gpsimd_dma if atp.exp_v_sendrecv_gpsimd else dma_engine.dma,
        )
        # Only NC0 adds partial results, unless we have out_in_sb then both cores will obtain the result
        if cfg.out_in_sb or (atp.sprior_prg_id == 0):
            nisa.tensor_tensor(res, res, res_recv, op=nl.add)

    # Store to output
    if cfg.out_in_sb:
        # exp_v and out may have different dtype, easier to just always keep this tensor copy to do the conversion
        # out is in tiled layout: [d_tile_size, n_d_tiles * total_bqh] where total_bqh = bs_full * s_active_qh
        # res is [d_tile_size, n_d_tiles * s_active_bqh]
        out_offset = btc.global_batch_offset * atp.s_active_qh
        total_bqh = atp.bs_full * atp.s_active_qh
        if total_bqh == atp.s_active_bqh:
            # No batch tiling: res and out have matching strides, single copy suffices
            total_size = atp.n_d_tiles * atp.s_active_bqh
            nisa.tensor_copy(
                out[:, out_offset : out_offset + total_size],
                src=res[:, :total_size],
            )
        else:
            for i_d in range(atp.n_d_tiles):
                d_src_offset = i_d * atp.s_active_bqh
                d_out_offset = i_d * total_bqh + out_offset
                nisa.tensor_copy(
                    out[:, d_out_offset : d_out_offset + atp.s_active_bqh],
                    src=res[:, d_src_offset : d_src_offset + atp.s_active_bqh],
                )
        if atp.bs_n_prgs > 1 and not cfg.return_cp_softmax_stats:
            # Skip sendrecv for CP: each NC keeps only its local batch portion in SBUF.
            # CP output collectives operate on each NC's local data independently.
            dst_bs_offset = ((1 - atp.bs_prg_id) * atp.bs_per_nc + btc.tile_batch_offset) * atp.s_active_qh
            if total_bqh == atp.s_active_bqh:
                # No batch tiling: d-tile sections are contiguous, single sendrecv suffices
                total_size = atp.n_d_tiles * atp.s_active_bqh
                nisa.sendrecv(
                    src=out[:, out_offset : out_offset + total_size],
                    dst=out[:, dst_bs_offset : dst_bs_offset + total_size],
                    send_to_rank=(1 - atp.bs_prg_id),
                    recv_from_rank=(1 - atp.bs_prg_id),
                    pipe_id=0,
                    dma_engine=dma_engine.gpsimd_dma if atp.exp_v_sendrecv_gpsimd else dma_engine.dma,
                )
            else:
                for i_d in range(atp.n_d_tiles):
                    d_out_src = i_d * total_bqh + out_offset
                    d_out_dst = i_d * total_bqh + dst_bs_offset
                    nisa.sendrecv(
                        src=out[:, d_out_src : d_out_src + atp.s_active_bqh],
                        dst=out[:, d_out_dst : d_out_dst + atp.s_active_bqh],
                        send_to_rank=(1 - atp.bs_prg_id),
                        recv_from_rank=(1 - atp.bs_prg_id),
                        pipe_id=0,
                        dma_engine=dma_engine.gpsimd_dma if atp.exp_v_sendrecv_gpsimd else dma_engine.dma,
                    )
    else:
        # Save exp_v (output) into DRAM (each NC writes its own batches in case of batch sharded)
        # This needs to be strided save due to different layout in SBUF and DRAM:
        #   SBUF: [d_tile_size, n_d_tiles * s_active_bqh]
        #   DRAM: [B, H, d_head, s_active]
        if atp.sprior_prg_id == 0:
            batch_pos = btc.global_batch_offset
            for i_d in range(atp.n_d_tiles):
                d_src_offset = i_d * atp.s_active_bqh
                d_dst_start = i_d * atp.d_tile_size
                res_tile_reshaped = (
                    (res)
                    .slice(1, start=d_src_offset, end=d_src_offset + atp.s_active_bqh)
                    .reshape_dim(1, [atp.bs, cfg.q_head, cfg.s_active])
                )
                out_view = (
                    (out)  # [B, H, d, S_active]
                    .slice(0, start=batch_pos, end=batch_pos + atp.bs)
                    .slice(2, start=d_dst_start, end=d_dst_start + atp.d_tile_size)
                    .permute((2, 0, 1, 3))
                )

                nisa.dma_copy(
                    dst=out_view,
                    src=res_tile_reshaped,
                    name=f"{sbm.get_name_prefix()}out_store_hbm_bt{btc.batch_tile_idx}_d{i_d}",
                )
    sbm.close_scope()


"""
Sharding Logic
"""


def _get_lnc_sharding(cfg: AttnTKGConfig) -> Tuple[int, int, int, int]:
    """
    Returns sharding parameters for context length (s_prior) and batch (bs) based on configuration.
    """
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    kernel_assert(
        n_prgs <= 2,
        f"Attention cascaded supports unsharded or LNC2 sharded; but got a spmd grid size of {n_prgs}",
    )

    sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id = (1, 0, 1, 0)
    if n_prgs > 1:
        TILE_CONSTANTS = TileConstants.get_tile_constants()
        if is_batch_sharded(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, TILE_CONSTANTS.p_max, cfg.fuse_rope):
            bs_n_prgs, bs_prg_id = (n_prgs, prg_id)
        elif is_s_prior_sharded(
            cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, TILE_CONSTANTS.p_max, cfg.fuse_rope
        ):  # If s_prior is small, and batch is not divisible by lnc
            sprior_n_prgs, sprior_prg_id = (n_prgs, prg_id)

    return sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id


def _get_interleaved_fa_tile_offset(fa_tile_idx: int, atp: AttnTileParams) -> int:
    """Compute global s_prior offset for an interleaved FA tile.

    FA tiles alternate between NC0 and NC1 for better DMA skipping load balance.
    Example with s_prior=40K, fa_tile_size=8K (each NC gets 20K = 3 tiles of 8K, 8K, 4K):

        NC0: tiles [0-8K], [16K-24K], [32K-36K]
        NC1: tiles [8K-16K], [24K-32K], [36K-40K]

    NC1 always gets the last global tile (which loads active tokens).

    Non-last tiles: offset = (2 * local_idx + prg_id) * fa_tile_size
    Last tile: offset = (num_tiles - 1) * 2 * fa_tile_size + prg_id * last_tile_size
    """
    if fa_tile_idx < atp.num_fa_tiles - 1:
        return (2 * fa_tile_idx + atp.sprior_prg_id) * atp.fa_tile_s_prior
    else:
        last_tile_size = atp.s_prior - (atp.num_fa_tiles - 1) * atp.fa_tile_s_prior
        return (atp.num_fa_tiles - 1) * 2 * atp.fa_tile_s_prior + atp.sprior_prg_id * last_tile_size


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
    cos = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name=sbm.get_name_prefix() + "name_cos")
    sin = sbm.alloc_stack((d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name=sbm.get_name_prefix() + "sin_rope")

    # Compute freqs = dot(inv_freqs, pos_ids), can be simplified to elem-wise multiply
    emb = sbm.alloc_stack(
        (d_head_half, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name=sbm.get_name_prefix() + "emb_rope"
    )
    nisa.tensor_scalar(emb, pos_ids[0:d_head_half, :], op0=nl.multiply, operand0=inv_freqs)

    # Compute ((emb + π) % 2π) - π, note that sin(θ) = sin((θ + π) % 2π - π)
    # This is to reduce emb to [-π, π] which is the restriction for Sine on ACT engine
    emb4sin = sbm.alloc_stack(
        (d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name=sbm.get_name_prefix() + "eb4sin_rope"
    )
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
    emb4cos = sbm.alloc_stack(
        (d_head, bs * s_a), dtype=nl.float32, buffer=nl.sbuf, name=sbm.get_name_prefix() + "emb4cos_rope"
    )
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
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    btc: BatchTileContext,
):
    """
    Load the attention sink and broadcast/transpose it into ``result``.

    Input ``sink_hbm`` is ``[H, 1]`` @ HBM where ``H = kv_heads * cfg.q_head``. This helper infers
    ``kv_heads`` from the shape, broadcasts over ``s_active``, and transposes the result into
    ``[s_active_bqh_tile, n_bsq_tiles]``.
    """
    sbm.open_scope()

    kv_heads = sink_hbm.shape[0] // cfg.q_head  # 1 for per-head; kv_heads when folded into batch

    # With batch sharding this core shards on a batch-folded range starting at btc.global_batch_offset.
    # When batch is odd, the kv_head idx may not be 0 at the sharded index.
    kv_offset = btc.global_batch_offset % kv_heads
    n_repeats = div_ceil(kv_offset + atp.bs, kv_heads)

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
    sink_repeated_view = sink_repeated[:, nl.ds(kv_offset * cfg.q_head * cfg.s_active, atp.s_active_bqh)]

    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_sink_transpose(
            i_bsq_tile,
            atp.s_active_bqh_tile,
            sink_hbm,
            sink_repeated_view,
            result,
            atp,
            TC,
            sbm,
        )

    if atp.s_active_bqh_remainder > 0:
        _tile_sink_transpose(
            atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, sink_hbm, sink_repeated_view, result, atp, TC, sbm
        )

    sbm.close_scope()


def _tile_sink_transpose(
    index,
    tile_size,
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
        sink_repeated[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
    )
    nisa.tensor_copy(sink_tp_repeated[:tile_size, index], sink_tp_psum)


def _fill_inactive_block_slots_with_spread(dst_u32, signed_table, num_columns, TC, sbm):
    """Replace inactive-block padding in a u32 index table with a cross-column-spread block id.

    dst_u32 is a [p_max, num_columns] uint32 gather-index table produced by a plain tensor_copy from
    signed_table, which maps every inactive-block sentinel (INACTIVE_BLOCK_IDX = -1, subdivided by
    block resize into {-rf, ..., -1}) to 0. Since the block-KV loads use dma_transpose (which cannot
    use oob_mode.skip), those padding gathers are still issued; leaving them all at 0 makes every
    inactive slot read block 0, and at partial cache the inactive columns all collide on the same
    128-block window in HBM, throttling DMA.

    Fill each inactive slot with (partition_id + column * p_max) via an iota so every slot targets a
    distinct block, spreading the padding gathers across HBM. signed_table carries the sentinels
    (negative values); dst_u32 receives the spread. Both fold-major and batch-major index tables use
    this (identical num_columns and p_max, so the spread is the same set of indices for both).

    The iota's access pattern generates values directly in [0, num_columns * p_max), which are all
    valid block indices: num_columns * p_max = batch_size * (num_folds_this_tile * p_max) is the number
    of block-slots this FA tile gathers, and a tile can gather no more blocks than the cache holds
    (num_folds_this_tile * p_max <= blocks_per_batch), so the max index stays within the gathered cache.
    """
    sbm.open_scope()
    inactive_mask = sbm.alloc_stack((TC.p_max, num_columns), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=inactive_mask, data=signed_table, op0=nl.less, operand0=0)
    spread_ids = sbm.alloc_stack((TC.p_max, num_columns), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=spread_ids, pattern=[[TC.p_max, num_columns]], offset=0, channel_multiplier=1)
    nisa.tensor_copy_predicated(dst=dst_u32, src=spread_ids, predicate=inactive_mask)
    sbm.close_scope()


_ABT_CACHE: dict = {}
_ABT_V_REARRANGED_CACHE: dict = {}


def _load_and_reshape_active_blk_table(
    active_blk_table,
    atp: AttnTileParams,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    btc: BatchTileContext,
    fa_ctx: FATileContext,
    additionally_emit_fold_major: bool = False,
):
    """
    Load active blocks table into SB for the current FA tile and batch tile.
    Only loads the folds corresponding to the current FA tile and batches for the current batch tile.
    Put every 128 consecutive blocks on the same column, spread along the partition dimension.
    If blocks per batch < 128, reduce block_len to increase blocks per batch to 128.
    Sets bufs.active_blocks_sb and atp.num_folds_per_batch.

    When additionally_emit_fold_major is True (QK-swap MM1 K-load), also exposes a fold-major uint32 table
    (bufs.active_blocks_sb_u32_fold_major). The default active_blocks_sb(_u32) is batch-major
    (required by the batched V-load), but the swap K-load gathers, per fold, the column-tile
    group's `batches` per-batch blocks as one contiguous [p_max, batches] slice — which is
    contiguous only in fold-major order (active_blk_table_sb, before the batch-major reorder).
    """
    TC = TileConstants.get_tile_constants()
    resize_factor = atp.blk_cache_resize_factor
    n_prgs = atp.sprior_n_prgs
    prg_id = atp.sprior_prg_id

    num_active_blks = active_blk_table.shape[1] * resize_factor
    kernel_assert(
        num_active_blks % (TC.p_max * n_prgs) == 0,
        (
            f"Block KV requires the number of active blocks per batch to be a multiple of (p_max * n_prgs). "
            f"Got {num_active_blks} with {n_prgs} shards. Consider using resize_cache_block_len_for_attention_tkg_kernel to get the correct resize_factor."
        ),
    )

    # Compute which folds correspond to this FA tile
    fold_s_prior = atp.block_len * TC.p_max
    is_dynamic = fa_ctx.dynamic_tile_offset_sbuf is not None

    if is_dynamic:
        # Dynamic path: num_folds_this_tile is compile-time (all non-last tiles are full-sized),
        # but fold_start is runtime. We use scalar_offset on the DMA to shift dynamically.
        num_folds_this_tile = fa_ctx.tile_s_prior // fold_s_prior
        fold_start = 0  # placeholder for NkiTensor; actual offset via scalar_offset
        # Compute dynamic fold start: dynamic_tile_offset_sbuf / fold_s_prior
        fold_s_prior_shift = int(math.log2(fold_s_prior))
        dynamic_fold_start_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=dynamic_fold_start_sbuf,
            data=fa_ctx.dynamic_tile_offset_sbuf,
            op0=nl.right_shift,
            operand0=fold_s_prior_shift,
        )
    else:
        fold_start = fa_ctx.tile_offset // fold_s_prior
        fold_end = div_ceil(fa_ctx.tile_offset + fa_ctx.tile_s_prior, fold_s_prior)
        num_folds_this_tile = fold_end - fold_start

    batch_start = btc.global_batch_offset
    batch_size = atp.bs

    # Set atp.num_folds_per_batch to the per-tile value
    atp.num_folds_per_batch = num_folds_this_tile

    # The reshape/transpose/spread chain below is a pure function of the HBM table's CONTENTS
    # and the tile geometry -- no dependency on any per-layer activation. The mega decode
    # kernel inlines this once per layer but passes only two distinct tables (SWA and full),
    # so 36 layers rebuild the same two results. Worse, the rebuild's Vector output is what
    # gates the MM1 K gather-transposes, putting redundant work directly on the critical path.
    # Cache by table identity + geometry; SWA and full tables have different ids so they
    # cannot collide. Trace-time only (tracer frontend), matching the prefetch FIFOs.
    _abt_reuse = _os.environ.get("VLLM_NEURON_MEGA_ABT_REUSE")
    _abt_key = None
    if _abt_reuse and not is_dynamic:
        _abt_key = (
            id(active_blk_table),
            fold_start,
            num_folds_this_tile,
            batch_start,
            batch_size,
            resize_factor,
            additionally_emit_fold_major,
            atp.use_dma_transpose,
            atp.use_v_dma_skipping,
            bufs.DBG_ACTIVE_TABLE is not None,
        )
        _hit = _ABT_CACHE.get(_abt_key)
        if _hit is not None:
            bufs.active_blocks_sb = _hit["active_blocks_sb"]
            if "active_blocks_sb_u32" in _hit:
                bufs.active_blocks_sb_u32 = _hit["active_blocks_sb_u32"]
            if "fold_major" in _hit:
                bufs.active_blocks_sb_u32_fold_major = _hit["fold_major"]
            return

    """
  Say active_blks has shape (B=2, blks_per_batch=4), with a reshape factor = 128/4 = 32
  [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]  # Note in reality block indices aren't sequential.

  We could load to SBUF as follows, and then do `blk_idx_sbuf * resize_factor + arange(resize_factor)`.
    par[ 0: 32]-> [0, 4,  8, 12]
    par[32: 64]-> [1, 5,  9, 13]
    par[64: 96]-> [2, 6, 10, 14]
    par[96:128]-> [3, 7, 11, 15]
  However we cannot use an affine expression with two indices on the partition dimension,
  so we cannot easily get to this state in SBUF.

  The alternative is to load to SBUF as shape (4, 128),
    [0,   0, ...,  0,   1,  1, ...,  1,   2,  2, ...,  2,   3,  3, ...,  3]
    [4,   4, ...,  4,   5,  5, ...,  5,   6,  6, ...,  6,   7,  7, ...,  7]
    [8,   8, ...,  8,   9,  9, ...,  9,  10, 10, ..., 10,  11, 11, ..., 11]
    [12, 12, ..., 12,  13, 13, ..., 13,  14, 14, ..., 14,  15, 15, ..., 15]
  And then transpose to the above desired shape.
  """
    if resize_factor == 1:
        # The code below is semantically correct for resize_factor > 1 but we cannot use
        # an affine expression with two indices on the partition dimension today.
        full_num_folds_total = num_active_blks // TC.p_max
        partition_resize = TC.p_max // resize_factor

        active_blk_table_sb = sbm.alloc_stack(
            (TC.p_max, num_folds_this_tile * batch_size),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
        )

        active_blk_table_sb_tv = (
            (active_blk_table_sb)
            # .reshape_dim(0, [partition_resize, resize_factor]) # Semantically correct, if two indices on partition were allowed
            .reshape_dim(1, [num_folds_this_tile, batch_size])
        )

        if is_dynamic:
            # Dynamic path: global fold indexing with scalar_offset for runtime fold start.
            dynamic_fold_offset_elems = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dynamic_fold_offset_elems, data=dynamic_fold_start_sbuf, op0=nl.multiply, operand0=partition_resize
            )

            active_blk_table_view = (
                active_blk_table.reshape_dim(1, [full_num_folds_total, partition_resize])
                .slice(0, batch_start, batch_start + batch_size)
                .slice(1, 0, num_folds_this_tile)
                .permute([2, 1, 0])
            )
            # Insert a 0-stride, resize_factor-wide broadcast dim into the access pattern,
            # and apply the dynamic per-fold offset via scalar_offset on dim 0 (the stride-1
            # partition-resize dim). dynamic_fold_offset_elems is in element units.
            _abt_pattern = active_blk_table_view.get_pattern()
            _abt_pattern = _abt_pattern[:1] + [[0, resize_factor]] + _abt_pattern[1:]
            active_blk_table_tv = active_blk_table_view.ap(
                pattern=_abt_pattern,
                scalar_offset=dynamic_fold_offset_elems,
                indirect_dim=0,
            )
            nisa.dma_copy(
                src=active_blk_table_tv,
                dst=active_blk_table_sb_tv,
                dge_mode=dge_mode.hwdge,
            )
        else:
            # Static path: tile_offset is global, fold indices are global.
            active_blk_table_tv = (
                (active_blk_table)
                .reshape_dim(1, [full_num_folds_total, partition_resize])
                .slice(0, batch_start, batch_start + batch_size)
                .slice(1, fold_start, fold_start + num_folds_this_tile)
                .permute([2, 1, 0])
                .expand_dim(1)
                .broadcast(1, resize_factor)
            )
            nisa.dma_copy(
                src=active_blk_table_tv,
                dst=active_blk_table_sb_tv,
                name=f"{sbm.get_name_prefix()}active_blk_table_load_resize1_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
                dge_mode=dge_mode.hwdge,
            )
    else:
        # We need to "resize" the cache blocks.
        # Only load the original blocks needed for this FA tile's folds.
        # Each expanded fold covers P_MAX sub-blocks = P_MAX/resize_factor original blocks.
        orig_blk_start = fold_start * TC.p_max // resize_factor
        orig_blk_end = (fold_start + num_folds_this_tile) * TC.p_max // resize_factor
        orig_blks_this_tile = orig_blk_end - orig_blk_start

        # tile_offset is always global for block KV, so orig_blk indices are global.

        active_blk_table_sb = sbm.alloc_stack(
            (TC.p_max, batch_size * num_folds_this_tile),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
            align=4,
        )

        sbm.open_scope()
        # Process in batch chunks of p_max to keep partition dim <= p_max
        batch_chunk = min(batch_size, TC.p_max)
        for batch_offset in range(0, batch_size, batch_chunk):
            cur_batch_chunk_sz = min(batch_chunk, batch_size - batch_offset)

            active_blk_pre_reshape = sbm.alloc_stack(
                (cur_batch_chunk_sz, orig_blks_this_tile), dtype=active_blk_table.dtype, buffer=nl.sbuf
            )
            active_blk_table_slice = (
                (active_blk_table)
                .slice(0, start=batch_start + batch_offset, end=batch_start + batch_offset + cur_batch_chunk_sz)
                .slice(1, start=orig_blk_start, end=orig_blk_end)
            )
            nisa.dma_copy(
                dst=active_blk_pre_reshape,
                src=active_blk_table_slice,
                name=f"{sbm.get_name_prefix()}active_blk_table_load_pre_reshape_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}_b{batch_offset}",
                dge_mode=nisa.dge_mode.hwdge,
            )

            # Now update the active blocks table with.  New active blocks table will be:
            #   old_blk_idx * resize_factor + arange(resize_factor)
            reshape_arange = sbm.alloc_stack(
                (cur_batch_chunk_sz, resize_factor), dtype=active_blk_table.dtype, buffer=nl.sbuf
            )
            nisa.iota(dst=reshape_arange, pattern=[[1, resize_factor]], offset=0)

            active_blk_reshaped = sbm.alloc_stack(
                (cur_batch_chunk_sz, orig_blks_this_tile, resize_factor), dtype=nl.float32, buffer=nl.sbuf
            )
            active_blk_pre_reshape_view = (active_blk_pre_reshape).expand_dim(2).broadcast(2, size=resize_factor)
            reshape_arange_view = (reshape_arange).expand_dim(1).broadcast(1, size=orig_blks_this_tile)
            nisa.scalar_tensor_tensor(
                dst=active_blk_reshaped,
                data=active_blk_pre_reshape_view,
                op0=nl.multiply,
                operand0=float(resize_factor),
                op1=nl.add,
                operand1=reshape_arange_view,
            )

            # Reshaped to flat sub-blocks: num_folds_this_tile * P_MAX sub-blocks
            active_blk_reshaped = active_blk_reshaped.reshape((cur_batch_chunk_sz, num_folds_this_tile * TC.p_max))

            # Transpose each fold into the output
            sbm.open_scope()
            for fold_rel_idx in range(num_folds_this_tile):
                active_blk_transposed = nl.ndarray(
                    (TC.p_max, cur_batch_chunk_sz),
                    dtype=active_blk_reshaped.dtype,
                    buffer=nl.psum,
                    address=None if sbm.is_auto_alloc() else (0, (fold_rel_idx % TC.psum_b_max) * TC.psum_f_max_bytes),
                )
                nisa.nc_transpose(
                    active_blk_transposed,
                    active_blk_reshaped[:, nl.ds(fold_rel_idx * TC.p_max, TC.p_max)],
                )
                nisa.tensor_copy(
                    active_blk_table_sb[:, nl.ds(fold_rel_idx * batch_size + batch_offset, cur_batch_chunk_sz)],
                    src=active_blk_transposed,
                    engine=nisa.vector_engine,
                )
            sbm.close_scope()
        sbm.close_scope()

    if additionally_emit_fold_major:
        # active_blk_table_sb is fold-major [p_max, num_folds * bs]. The swap K-load wants this
        # layout (per fold, the group's batch blocks are contiguous), so expose its u32 cast here
        # before the batch-major reorder below.
        fold_major_num_folds = num_folds_this_tile * batch_size
        bufs.active_blocks_sb_u32_fold_major = sbm.alloc_stack(
            (TC.p_max, fold_major_num_folds),
            dtype=nl.uint32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(bufs.active_blocks_sb_u32_fold_major, active_blk_table_sb, engine=nisa.vector_engine)
        _fill_inactive_block_slots_with_spread(
            bufs.active_blocks_sb_u32_fold_major, active_blk_table_sb, fold_major_num_folds, TC, sbm
        )

    # Reorder from fold-major [p_max, num_folds * bs] to batch-major [p_max, bs * num_folds]
    # so that for a fixed batch, fold indices are contiguous (required for DMA batching).
    if batch_size > 1:
        active_blk_table_reordered = sbm.alloc_stack(
            (TC.p_max, batch_size * num_folds_this_tile),
            dtype=active_blk_table_sb.dtype,
            buffer=nl.sbuf,
        )
        src_view = (active_blk_table_sb).reshape_dim(1, [num_folds_this_tile, batch_size])
        dst_view = (active_blk_table_reordered).reshape_dim(1, [batch_size, num_folds_this_tile])
        nisa.tensor_copy(dst=dst_view.permute([0, 2, 1]), src=src_view, engine=nisa.vector_engine)
        bufs.active_blocks_sb = active_blk_table_reordered
    else:
        bufs.active_blocks_sb = active_blk_table_sb

    if atp.use_dma_transpose or not atp.use_v_dma_skipping:
        # Pre-compute active_blocks_sb_u32 from the (reordered) buffer.
        total_num_folds = batch_size * num_folds_this_tile
        bufs.active_blocks_sb_u32 = sbm.alloc_stack(
            (TC.p_max, total_num_folds),
            dtype=nl.uint32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(bufs.active_blocks_sb_u32, bufs.active_blocks_sb, engine=nisa.vector_engine)
        _fill_inactive_block_slots_with_spread(
            bufs.active_blocks_sb_u32, bufs.active_blocks_sb, total_num_folds, TC, sbm
        )

    # Store to debug tensor if available (incrementally per FA tile and batch tile)
    # Use the original fold-major buffer for debug output (DBG_ACTIVE_TABLE expects fold-outer, batch-inner)
    if bufs.DBG_ACTIVE_TABLE is not None:
        dbg_fold_offset = atp.sprior_prg_id * (atp.s_prior // (atp.block_len * TC.p_max)) + fold_start
        dbg_src = active_blk_table_sb.reshape((TC.p_max, num_folds_this_tile, batch_size))
        nisa.dma_copy(
            bufs.DBG_ACTIVE_TABLE[
                :,
                nl.ds(dbg_fold_offset, num_folds_this_tile),
                nl.ds(batch_start, batch_size),
            ],
            dbg_src,
            name=f"{sbm.get_name_prefix()}dbg_active_blocks_table_store_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
        )

    if _abt_key is not None:
        _entry = {"active_blocks_sb": bufs.active_blocks_sb}
        if atp.use_dma_transpose or not atp.use_v_dma_skipping:
            _entry["active_blocks_sb_u32"] = bufs.active_blocks_sb_u32
        if additionally_emit_fold_major:
            _entry["fold_major"] = bufs.active_blocks_sb_u32_fold_major
        _ABT_CACHE[_abt_key] = _entry


### K-Active Stitching Helper


def _stitch_k_active(
    k_sb,
    bufs,
    atp,
    cfg,
    TC,
    btc,
    i_b,
    num_folds_this_tile,
    k_block_len_row_tiled,
    k_dma_batch_n_folds,
    k_dma_batch_n_batches,
    _k_sb_dtype,
):
    """Stitch k_active tokens into k_sb on the last FA tile.

    On the last FA tile, the s_active most-recent tokens (being generated) must be written into
    k_sb at the tail of the last fold. The source is bufs.k_active_sb [d_head, bs_full * s_active].

    Structure:
      - Flat KV: simple tail copy into k_sb.
      - Block KV: active tokens span num_blks_covering_s_active blocks at the end of the last fold.
        extra_covered = how many positions in the first covered block are prior cache (not active).
        Three sub-paths based on dtype and qk_row_tile_factor:

        1. fp8_packed (any qk_row_tile_factor): per-token copy into a 4D permuted fp8 view of k_sb.
           Uses _k_tile_physical_index for bh_idx to handle kbld-outer interleaving.
           Partition-half selection for qk_row_tile_factor=2 uses quarter-based addressing.

        2. Non-fp8, qk_row_tile_factor=1: bulk copy when no batching (last fold contiguous),
           per-token via _k_tile_physical_index when batching (kbld-outer interleaved).

        3. Non-fp8, qk_row_tile_factor=2: per-token copy with even/odd partition-half selection.
           Even seq positions → partitions 0:d_head, odd → d_head:128.

    All block-KV paths account for DMA batching layout (kbld-outer interleaving) when computing
    physical offsets into k_sb.
    """
    if atp.is_block_kv:
        num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
        extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active
        # Per-token stitching paths (fp8_packed and qk_row_tile_factor>1) address blocks by partition index
        # within a single fold. Bulk copy (qk_row_tile_factor==1, non-fp8) would also fail with negative
        # partition indices. In practice always true since s_active << p_max * block_len.
        kernel_assert(
            num_blks_covering_s_active <= TC.p_max,
            f"k_active stitching requires all active blocks fit in one fold (p_max={TC.p_max}), "
            f"got num_blks_covering_s_active={num_blks_covering_s_active}.",
        )

        if cfg.fp8_packed:
            # fp8_packed k_active stitching:
            # k_sb is bf16 [d_head*qk_row_tile_factor, num_folds*kblp*p_max]. Reinterpret as fp8 to get
            # [d_head*qk_row_tile_factor, num_folds*kblp*p_max*2] with interleaved even/odd layout.
            k_sb_fp8_stitch = k_sb.view(_k_sb_dtype)
            _stitch_first_dim = k_sb_fp8_stitch.shape[1] // (TC.p_max * 2)
            k_sb_fp8_4d = k_sb_fp8_stitch.reshape_dim(1, [_stitch_first_dim, TC.p_max * 2]).reshape_dim(
                2, [TC.p_max, 2]
            )
            k_sb_fp8_perm = k_sb_fp8_4d.permute([0, 2, 1, 3])

            k_active_batch = (
                (bufs.k_active_sb).reshape_dim(1, [atp.bs_full, cfg.s_active]).select(1, btc.global_batch_offset + i_b)
            )

            last_fold_idx = num_folds_this_tile - 1
            first_block_partition = TC.p_max - num_blks_covering_s_active

            for i_active in range(cfg.s_active):
                pos_in_blocks = extra_covered + i_active
                blk_idx = pos_in_blocks // atp.block_len
                seq_in_blk = pos_in_blocks % atp.block_len
                partition_idx = first_block_partition + blk_idx

                if atp.qk_row_tile_factor == 1:
                    pos_in_fold = seq_in_blk // 2
                    parity = seq_in_blk % 2
                    logical_col = last_fold_idx * k_block_len_row_tiled + pos_in_fold
                    bh_idx = _k_tile_physical_index(
                        logical_col,
                        k_block_len_row_tiled,
                        k_dma_batch_n_folds,
                        k_dma_batch_n_batches,
                        i_b,
                    )
                    dst_view = (
                        k_sb_fp8_perm.slice(1, start=partition_idx, end=partition_idx + 1)
                        .slice(2, start=bh_idx, end=bh_idx + 1)
                        .slice(3, start=parity, end=parity + 1)
                    )
                else:
                    pos_in_fold = seq_in_blk // 4
                    pos_in_quarter = seq_in_blk % 4
                    partition_half = pos_in_quarter // 2
                    parity = pos_in_quarter % 2
                    logical_col = last_fold_idx * k_block_len_row_tiled + pos_in_fold
                    bh_idx = _k_tile_physical_index(
                        logical_col,
                        k_block_len_row_tiled,
                        k_dma_batch_n_folds,
                        k_dma_batch_n_batches,
                        i_b,
                    )
                    dst_view = (
                        k_sb_fp8_perm.slice(0, start=partition_half * cfg.d_head, end=(partition_half + 1) * cfg.d_head)
                        .slice(1, start=partition_idx, end=partition_idx + 1)
                        .slice(2, start=bh_idx, end=bh_idx + 1)
                        .slice(3, start=parity, end=parity + 1)
                    )

                src_view = k_active_batch.slice(1, start=i_active, end=i_active + 1)
                nisa.tensor_copy(dst=dst_view, src=src_view)
        else:
            # Non-fp8-packed stitching. Need to handle dim_0 * blk_len + dim_1 >= extra_covered.
            # Solving: (dim_0, dim_1) in {(0, [extra_covered, blk_len)) and ([1, num_blks), [0, blk_len))}
            if atp.qk_row_tile_factor == 1:
                last_fold_idx = num_folds_this_tile - 1
                fa_tile_s_prior = k_sb.shape[1] // atp.n_d_tiles

                for i_d in range(atp.n_d_tiles):
                    k_sb_d_start = i_d * fa_tile_s_prior
                    k_sb_d_tile = (k_sb).slice(1, start=k_sb_d_start, end=k_sb_d_start + fa_tile_s_prior)
                    k_active_d_offset = i_d * atp.bs_full * cfg.s_active
                    k_active_batch = (
                        (bufs.k_active_sb)
                        .slice(1, start=k_active_d_offset, end=k_active_d_offset + atp.bs_full * cfg.s_active)
                        .reshape_dim(1, [atp.bs_full, cfg.s_active])
                        .select(1, btc.global_batch_offset + i_b)
                    )

                    if k_dma_batch_n_folds > 1 or k_dma_batch_n_batches > 1:
                        # With batching active, last fold's data is non-contiguous (kbld-outer interleaved).
                        # Use per-token addressing via _k_tile_physical_index.
                        n_tiles = fa_tile_s_prior // TC.p_max
                        k_sb_3d = k_sb_d_tile.reshape_dim(1, [n_tiles, TC.p_max]).permute([0, 2, 1])
                        first_block_partition = TC.p_max - num_blks_covering_s_active

                        for i_active in range(cfg.s_active):
                            pos_in_blocks = extra_covered + i_active
                            blk_idx = pos_in_blocks // atp.block_len
                            seq_in_blk = pos_in_blocks % atp.block_len
                            partition_idx = first_block_partition + blk_idx

                            logical_col = last_fold_idx * k_block_len_row_tiled + seq_in_blk
                            col_idx = _k_tile_physical_index(
                                logical_col,
                                k_block_len_row_tiled,
                                k_dma_batch_n_folds,
                                k_dma_batch_n_batches,
                                i_b,
                            )

                            dst_view = k_sb_3d.slice(1, start=partition_idx, end=partition_idx + 1).slice(
                                2, start=col_idx, end=col_idx + 1
                            )
                            src_view = k_active_batch.slice(1, start=i_active, end=i_active + 1)
                            nisa.tensor_copy(dst=dst_view, src=src_view)
                    else:
                        # No batching: last fold's data is contiguous in k_sb. Use bulk copy.
                        # No partition splitting needed (full d_head per partition).
                        # Need to mask as dim_0 * blk_len + dim_1 >= extra_covered
                        # Solving the above inequality with 0 <= dim_0 < num_blks_covering_s_active and 0 <= dim_1 < blk_len
                        # we get (dim_0, dim_1) in {(0,[extra_covered, blk_len)) and ([1, num_blks_covering_s_active), [0, blk_len))}
                        # Thus, if extra_covered != 0, we do an access pattern for dim_0 == 0 and dim_1 in [extra_covered, blk_len)
                        # and for main copy we don't need any restrictions.
                        last_fold_start = last_fold_idx * atp.block_len * TC.p_max
                        k_sb_last_fold = (
                            (k_sb_d_tile)
                            .slice(1, start=last_fold_start, end=last_fold_start + atp.block_len * TC.p_max)
                            .reshape_dim(1, [atp.block_len, TC.p_max])
                            .permute([0, 2, 1])
                        )

                        if extra_covered > 0:
                            if atp.block_len > extra_covered:
                                first_block_partition = TC.p_max - num_blks_covering_s_active
                                k_sb_view = k_sb_last_fold.slice(
                                    1, start=first_block_partition, end=first_block_partition + 1
                                ).slice(2, start=extra_covered, end=atp.block_len)
                                k_active_view = k_active_batch.slice(1, start=0, end=atp.block_len - extra_covered)
                                nisa.tensor_copy(dst=k_sb_view, src=k_active_view)

                            if num_blks_covering_s_active > 1:
                                k_sb_view = k_sb_last_fold.slice(
                                    1, start=TC.p_max - num_blks_covering_s_active + 1, end=TC.p_max
                                ).slice(2, start=0, end=atp.block_len)
                                k_active_start_1 = atp.block_len - extra_covered
                                k_active_view = k_active_batch.slice(
                                    1,
                                    start=k_active_start_1,
                                    end=k_active_start_1 + (num_blks_covering_s_active - 1) * atp.block_len,
                                ).reshape_dim(1, [num_blks_covering_s_active - 1, atp.block_len])
                                nisa.tensor_copy(dst=k_sb_view, src=k_active_view)
                        else:
                            k_sb_view = k_sb_last_fold.slice(
                                1, start=TC.p_max - num_blks_covering_s_active, end=TC.p_max
                            ).slice(2, start=0, end=atp.block_len)
                            k_active_view = k_active_batch.slice(
                                1, start=0, end=num_blks_covering_s_active * atp.block_len
                            ).reshape_dim(1, [num_blks_covering_s_active, atp.block_len])
                            nisa.tensor_copy(dst=k_sb_view, src=k_active_view)
            else:
                # qk_row_tile_factor > 1: per-token copy with partition half selection
                # bf16 layout: even positions (seq%2==0) → partitions 0:d_head, odd → d_head:128
                k_active_batch = (
                    (bufs.k_active_sb)
                    .reshape_dim(1, [atp.bs_full, cfg.s_active])
                    .select(1, btc.global_batch_offset + i_b)
                )
                last_fold_idx = num_folds_this_tile - 1
                first_block_partition = TC.p_max - num_blks_covering_s_active

                # Reshape full k_sb for per-token addressing: [d_head*krtf, n_tiles, p_max].
                # Use _k_tile_physical_index for col_idx to handle kbld-outer interleaving.
                n_tiles = k_sb.shape[1] // TC.p_max
                k_sb_3d = (k_sb).reshape_dim(1, [n_tiles, TC.p_max]).permute([0, 2, 1])

                for i_active in range(cfg.s_active):
                    pos_in_blocks = extra_covered + i_active
                    blk_idx = pos_in_blocks // atp.block_len
                    seq_in_blk = pos_in_blocks % atp.block_len
                    partition_half = seq_in_blk % 2
                    pos_in_packed = seq_in_blk // 2

                    logical_col = last_fold_idx * k_block_len_row_tiled + pos_in_packed
                    col_idx = _k_tile_physical_index(
                        logical_col,
                        k_block_len_row_tiled,
                        k_dma_batch_n_folds,
                        k_dma_batch_n_batches,
                        i_b,
                    )
                    partition_idx = first_block_partition + blk_idx

                    dst_view = (
                        k_sb_3d.slice(0, start=partition_half * cfg.d_head, end=(partition_half + 1) * cfg.d_head)
                        .slice(1, start=partition_idx, end=partition_idx + 1)
                        .slice(2, start=col_idx, end=col_idx + 1)
                    )
                    src_view = k_active_batch.slice(1, start=i_active, end=i_active + 1)
                    nisa.tensor_copy(dst=dst_view, src=src_view)
    else:
        # Flat KV: stitch k_active at the tail of each d_tile's section in k_sb
        for i_d in range(atp.n_d_tiles):
            k_sb_d_offset = i_d * (k_sb.shape[1] // atp.n_d_tiles)
            k_sb_d_size = k_sb.shape[1] // atp.n_d_tiles
            k_active_d_offset = i_d * atp.bs_full * cfg.s_active
            nisa.tensor_copy(
                k_sb[:, k_sb_d_offset + k_sb_d_size - cfg.s_active : k_sb_d_offset + k_sb_d_size],
                bufs.k_active_sb[
                    :, nl.ds(k_active_d_offset + (btc.global_batch_offset + i_b) * cfg.s_active, cfg.s_active)
                ],
                engine=nisa.scalar_engine,
            )


def _stitch_k_active_swap(
    k_sb,
    bufs,
    atp,
    cfg,
    TC,
    btc,
    batch_base,
    batches_per_psum,
    k_tile_pos_in_last_fold,
    n_pos_in_last_fold,
    last_fold_base_in_k_tile,
    _k_sb_dtype,
):
    """Stitch k_active tokens into k_sb on the last FA tile (QK-swap path) for swapped layout.

    k_active lands in the last fold, which may be split across several k_tiles. This k_tile covers the
    fold's positions [k_tile_pos_in_last_fold, k_tile_pos_in_last_fold + n_pos_in_last_fold), stored in
    k_sb starting at buffer column last_fold_base_in_k_tile. An active token at fold-position pos_in_fold
    is written only if it falls in this range, at buffer column
    last_fold_base_in_k_tile + (pos_in_fold - k_tile_pos_in_last_fold). Called for every k_tile
    overlapping the last fold.
    """
    num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
    extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active
    kernel_assert(
        num_blks_covering_s_active <= TC.p_max,
        f"k_active stitching requires all active blocks fit in one fold (p_max={TC.p_max}), "
        f"got num_blks_covering_s_active={num_blks_covering_s_active}.",
    )
    first_block_partition = TC.p_max - num_blks_covering_s_active

    # Row-tiled layout: the k_block_len_dma axis is folded by qk_row_tile_factor into the partition
    # halves, so the column count uses k_block_len_row_tile and a k_block_len_dma row maps to
    # (pos_in_fold = row // qk_row_tile_factor on the column axis, partition_half = row %
    # qk_row_tile_factor on the partition-half axis). The last fold's pos_in_fold maps to buffer column
    # (last_fold_base_in_k_tile + pos_in_fold - k_tile_pos_in_last_fold) * batches_per_psum + i_b_local.
    k_d_head_row_tile = cfg.d_head * atp.qk_row_tile_factor
    # k_sb (bf16 for both dtypes) free dim = (k_tile positions * batches_per_psum) * p_max.
    n_tiles = k_sb.shape[1] // TC.p_max
    if cfg.fp8_packed:
        # k_sb is bf16; reinterpret as fp8 [.., 2] to expose the packed parity pair.
        k_sb_perm = k_sb.view(_k_sb_dtype).reshape((k_d_head_row_tile, n_tiles, TC.p_max, 2)).permute([0, 2, 1, 3])
    else:
        k_sb_perm = k_sb.reshape((k_d_head_row_tile, n_tiles, TC.p_max)).permute([0, 2, 1])

    for i_b_local in range(batches_per_psum):
        i_b = batch_base + i_b_local
        k_active_batch = (
            (bufs.k_active_sb).reshape_dim(1, [atp.bs_full, cfg.s_active]).select(1, btc.global_batch_offset + i_b)
        )
        for i_active in range(cfg.s_active):
            pos_in_blocks = extra_covered + i_active
            blk_idx = pos_in_blocks // atp.block_len
            seq_in_blk = pos_in_blocks % atp.block_len
            partition_idx = first_block_partition + blk_idx
            if cfg.fp8_packed:
                # fp8 packs 2 seq positions per k_block_len_dma row -> (row, parity); row-tiling then
                # splits the row into (pos_in_fold, partition_half). partition_half picks the d_head slab.
                k_block_len_dma_row = seq_in_blk // 2
                parity = seq_in_blk % 2
                pos_in_fold = k_block_len_dma_row // atp.qk_row_tile_factor
                partition_half = k_block_len_dma_row % atp.qk_row_tile_factor
            else:
                # bf16: seq_in_blk is the k_block_len_dma row; row-tiling splits it into
                # (pos_in_fold, partition_half).
                pos_in_fold = seq_in_blk // atp.qk_row_tile_factor
                partition_half = seq_in_blk % atp.qk_row_tile_factor
            # Skip active tokens outside this k_tile's slice of the last fold (compile-time static).
            if not (k_tile_pos_in_last_fold <= pos_in_fold < k_tile_pos_in_last_fold + n_pos_in_last_fold):
                continue
            col_idx = (last_fold_base_in_k_tile + pos_in_fold - k_tile_pos_in_last_fold) * batches_per_psum + i_b_local
            if cfg.fp8_packed:
                dst_view = k_sb_perm[
                    partition_half * cfg.d_head : (partition_half + 1) * cfg.d_head,
                    partition_idx : partition_idx + 1,
                    col_idx : col_idx + 1,
                    parity : parity + 1,
                ]
            else:
                dst_view = k_sb_perm[
                    partition_half * cfg.d_head : (partition_half + 1) * cfg.d_head,
                    partition_idx : partition_idx + 1,
                    col_idx : col_idx + 1,
                ]
            src_view = k_active_batch.slice(1, start=i_active, end=i_active + 1)
            nisa.tensor_copy(dst=dst_view, src=src_view)


### QK-swap Helpers


def _build_swap_q_packed_col(bufs, atp, cfg, TC, sbm, sprior_band_factor, col_tile_width):
    """Build the QK-swap padded stationary Q when s_active_qh is small.

    Every query head is placed at the start of its own ``col_tile_width`` (== 32) slot, spaced
    col_tile_width apart with zeros elsewhere. The swap MM1 reads a 32-wide window sliding one head-width
    (s_active_qh) per s_prior band.

    Layout (each head occupies s_active_qh of its col_tile_width slot; rest of the slot is zeros):
        [ left_zeros  | Q0 | zeros | Q1 | zeros | ... | Q_last | zeros ]
    left_zeros = (sprior_band_factor - 1) * s_active_qh gives the highest band's window room to slide LEFT
    of the first head without underflowing.
    """
    k_d_head_row_tile = cfg.d_head * atp.qk_row_tile_factor
    left_zeros = (sprior_band_factor - 1) * atp.s_active_qh  # columns of left slide room
    q_pad_width = left_zeros + atp.bs_full * col_tile_width
    q_pad_sb = sbm.alloc_stack((k_d_head_row_tile, q_pad_width), dtype=bufs.q_sb.dtype, buffer=nl.sbuf)
    nisa.memset(q_pad_sb, 0.0)
    # Every head lands at the start of its col_tile_width slot (offset left_zeros + i*col_tile_width),
    q_sb_batched = bufs.q_sb.reshape((k_d_head_row_tile, atp.bs_full, atp.s_active_qh))
    q_pad_tiles = q_pad_sb[:, left_zeros : left_zeros + atp.bs_full * col_tile_width].reshape(
        (k_d_head_row_tile, atp.bs_full, col_tile_width)
    )
    nisa.tensor_copy(dst=q_pad_tiles[:, :, : atp.s_active_qh], src=q_sb_batched)
    return q_pad_sb, left_zeros


def _swap_q_packed_col_window_start(q_idx, out_band_idx, left_zeros, atp):
    """Start column of the 32-wide (== col_tile_width) stationary window for one packed-column output.

    q_idx picks which batch to read and out_band_idx picks which band to place q in the column tile.
    """
    batches_per_col_tile = 32 // atp.s_active_qh
    q_col = left_zeros + q_idx * 32
    band_col_offset = (out_band_idx % batches_per_col_tile) * atp.s_active_qh
    return q_col - band_col_offset


def _swap_band_factor(atp, TC):
    """QK-swap partition banding factor (see _compute_kq_matmul_and_max_swapped). 1 when unbanded.

    Banding folds a query's s_prior onto the partition axis (into band_factor contiguous "bands", the
    matmul's per-partition compute slots).
    """
    batches_per_psum = TC.p_max // atp.s_active_qh
    return batches_per_psum // min(atp.bs_per_nc, batches_per_psum)


def _band_combine(src, dst, op, atp, TC):
    """Combine per-band partial stats in `src` down to the un-banded s_active_bqh layout of `dst`.

    Under partition banding `src` holds s_active_bqh*sprior_band_factor partial rows: query q's bands sit
    on adjacent s_active_qh-row blocks (rows [q*bf*sqh : (q+1)*bf*sqh), band b at offset b*sqh). Reduce
    across the band axis so the result occupies `dst`'s un-banded s_active_bqh rows. `src`/`dst` are
    single-column [rows, 1] views (caller pre-slices) and may be the same buffer (in-place).
    Transpose->free-reduce->transpose because bands are on the partition dim (not directly reducible).
    Caller must guard on band_factor > 1.
    """
    bf = _swap_band_factor(atp, TC)
    sqh = atp.s_active_qh
    banded_rows = atp.s_active_bqh * bf
    # Transpose to free so the banded rows are reducible: [bf*s_active_bqh, 1] -> [1, bf*s_active_bqh].
    tp = nl.ndarray((1, banded_rows), dtype=src.dtype, buffer=nl.psum)
    nisa.nc_transpose(tp, src[:banded_rows, 0:1])
    # Free order is (q, band, p) = [bs_per_nc, bf, sqh]; band is the MIDDLE axis so tensor_reduce (which
    # only reduces trailing dims) can't take it. Accumulate the bf band slices pairwise instead into a
    # [1, bs_per_nc, sqh] = [1, s_active_bqh] combined tile.
    tp_bands = tp.reshape((1, atp.bs_per_nc, bf, sqh))
    combined = nl.ndarray((1, atp.bs_per_nc, sqh), dtype=src.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(combined, tp_bands[:, :, 0, :])
    for b in range(1, bf):
        nisa.tensor_tensor(combined, combined, tp_bands[:, :, b, :], op=op)
    # Transpose back onto the un-banded s_active_bqh partition rows. nc_transpose (PE) writes to PSUM, so
    # transpose into a PSUM tile then copy into dst.
    back_psum = nl.ndarray((atp.s_active_bqh, 1), dtype=src.dtype, buffer=nl.psum)
    nisa.nc_transpose(back_psum, combined.reshape((1, atp.s_active_bqh)))
    nisa.tensor_copy(dst[: atp.s_active_bqh, 0:1], back_psum)


def _band_broadcast(src, dst, atp, TC):
    """Broadcast an un-banded [s_active_bqh, 1] `src` column up to `dst`'s banded [s_active_bqh*bf, 1] layout.

    Replicate each band's value to all sprior_band_factor of its adjacent band rows
    (dst[q*bf*sqh + b*sqh + p] <- src[q*sqh + p] for every band b). `src` and `dst` may be the same buffer.
    Transpose->free-broadcast->transpose (partition dim broadcast). Caller must guard on band_factor > 1.
    """
    bf = _swap_band_factor(atp, TC)
    sqh = atp.s_active_qh
    banded_rows = atp.s_active_bqh * bf
    # src [s_active_bqh, 1] -> free [1, s_active_bqh].
    tp = nl.ndarray((1, atp.s_active_bqh), dtype=src.dtype, buffer=nl.psum)
    nisa.nc_transpose(tp, src[: atp.s_active_bqh, 0:1])
    # Replicate the band axis: [1, bs_per_nc, sqh] -> [1, bs_per_nc, bf, sqh] via a 0-stride band dim.
    tp_q = tp.reshape((1, atp.bs_per_nc, sqh))
    bc = nl.ndarray((1, banded_rows), dtype=src.dtype, buffer=nl.sbuf)
    bc_bands = bc.reshape((1, atp.bs_per_nc, bf, sqh))
    for b in range(bf):
        nisa.tensor_copy(bc_bands[:, :, b, :], tp_q)
    # Transpose free [1, banded_rows] back onto the banded partition rows. nc_transpose (PE) writes PSUM.
    bc_psum = nl.ndarray((banded_rows, 1), dtype=src.dtype, buffer=nl.psum)
    nisa.nc_transpose(bc_psum, bc)
    nisa.tensor_copy(dst[:banded_rows, 0:1], bc_psum)


### DMA Batching Helpers


def _compute_dma_batch_params(num_folds: int, bs: int, granularity: int, cap: int, sbm: SbufManager):
    """Compute fold-batching and batch-batching parameters for DMA calls (unswapped K-load).

    Two-step algorithm to fit n_folds * n_batches * granularity <= cap:
      1. Find largest N dividing num_folds with N * granularity <= cap.
      2. If all folds are batched (N == num_folds) and bs > 1, find largest M
         dividing bs with N * M * granularity <= cap.

    Only activates when sbm uses auto allocation (manual alloc calculations not updated for batching).
    Returns (batch_n_folds, batch_n_batches).
    """
    if not sbm.is_auto_alloc():
        return 1, 1
    batch_n_folds = 1
    for n in range(num_folds, 0, -1):
        if num_folds % n == 0 and n * granularity <= cap:
            batch_n_folds = n
            break
    batch_n_batches = 1
    if batch_n_folds == num_folds and bs > 1:
        for m in range(bs, 0, -1):
            if bs % m == 0 and batch_n_folds * m * granularity <= cap:
                batch_n_batches = m
                break
    return batch_n_folds, batch_n_batches


def _compute_k_tile_gather(
    n_mm1_grps_total: int,
    pos_per_fold: int,
    n_pos_per_mm1_grp: int,
    batches_per_psum: int,
    all_groups_full: bool,
    cap: int,
):
    """Plan the swap MM1 K gather: how many mm1 groups per k_tile, and how many k_tiles to loop over.

    A k_tile is a contiguous run of mm1 groups whose K is staged together, gathered for all
    batches_per_psum batches. The gather issues one dma_transpose per fold the k_tile spans (a fold holds
    pos_per_fold buffer positions and has its own block indices). Sizing balances two failure modes: the
    whole FA tile OOMs SBUF, while a single mm1 group throttles the DMA when n_pos_per_mm1_grp is tiny.

    Two regimes:
      - fold >= one group (pos_per_fold >= n_pos_per_mm1_grp): pick the largest count of groups that both
        fits the DGE cap and stays within one fold (divides groups-per-fold), so each k_tile is one gather.
      - fold < one group (pos_per_fold < n_pos_per_mm1_grp): one group per k_tile; its gather spans
        ceil(n_pos_per_mm1_grp / pos_per_fold) folds, each a separate dma_transpose.

    A partial last mm1 group (SWA) makes groups within a k_tile unequal, so fall back to one group per
    k_tile. is_qk_swapped gates out small s_active_qh (<= 4), so one group always fits the cap and the
    size is always >= 1.

    Returns (mm1_grps_per_k_tile, n_k_tiles).
    """
    mm1_grps_per_k_tile = 1
    if all_groups_full and pos_per_fold >= n_pos_per_mm1_grp:
        mm1_grps_per_fold = pos_per_fold // n_pos_per_mm1_grp
        for c in range(mm1_grps_per_fold, 0, -1):
            if mm1_grps_per_fold % c == 0 and c * n_pos_per_mm1_grp * batches_per_psum <= cap:
                mm1_grps_per_k_tile = c
                break
    return mm1_grps_per_k_tile, div_ceil(n_mm1_grps_total, mm1_grps_per_k_tile)


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


def _store_dbg_qk(
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx,
    btc,
):
    """Store the raw QK scores (bufs.qk) into the current FA-tile slice of DBG_QK.

    DBG_QK expects [s_prior(partition), ..., s_active_bqh(free)]. The unswapped qk is already in that
    layout and is stored directly. The swap qk is transposed [s_active_bqh(partition), s_prior(free)],
    so each (bsq tile, s_prior 128-tile) is transposed back into an s_prior-major scratch first
    (mirrors Step 3 of the exp stage, applied to raw qk).

    strided_mm1 + (FA or batch tiling) zero-fills instead: K column remapping / batch-sprior
    interleaving make per-tile slices unreliable. The swap path asserts non-strided, so it never hits
    that branch.
    """
    if not DBG_TENSORS:
        return

    dbg_tile_offset = fa_ctx.fa_tile_idx * atp.fa_n_sprior_tile
    bqh_offset = btc.tile_batch_offset * atp.s_active_qh
    dst = bufs.DBG_QK[
        :,
        atp.sprior_prg_id,
        dbg_tile_offset : dbg_tile_offset + fa_ctx.tile_n_sprior,
        atp.bs_prg_id,
        bqh_offset : bqh_offset + atp.s_active_bqh,
    ]

    if cfg.strided_mm1 and (atp.use_fa or atp.bs != atp.bs_per_nc):
        sbm.open_scope()
        dbg_zero = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk.dtype, buffer=nl.sbuf)
        nisa.memset(dbg_zero, 0.0)
        dbg_zero_bc = (
            (dbg_zero).reshape_dim(1, [1, 1, 1, 1]).broadcast(2, fa_ctx.tile_n_sprior).broadcast(4, atp.s_active_bqh)
        )
        nisa.dma_copy(
            dst,
            dbg_zero_bc,
            name=f"{sbm.get_name_prefix()}dbg_qk_store_zeros_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
        )
        sbm.close_scope()
        return

    if atp.qk_swapped:
        # qk is fp32; transpose each (bsq tile, s_prior 128-tile) directly into an fp32 s_prior-major
        # scratch, then dma_copy to the fp32 DBG_QK.
        sbm.open_scope()
        dbg_qk_sprior_major = sbm.alloc_stack(
            (TC.p_max, fa_ctx.tile_n_sprior * atp.s_active_bqh), dtype=bufs.qk.dtype, buffer=nl.sbuf
        )
        tile_free = fa_ctx.tile_n_sprior * TC.p_max
        for i_bsq in range(atp.n_bsq_tiles):
            bsq_start = i_bsq * atp.s_active_bqh_tile
            bsq_size = min(atp.s_active_bqh_tile, atp.s_active_bqh - bsq_start)
            qk_bsq = bufs.qk[:, i_bsq * tile_free : (i_bsq + 1) * tile_free]
            for i_sp in range(fa_ctx.tile_n_sprior):
                tp_psum = nl.ndarray((TC.p_max, bsq_size), dtype=bufs.qk.dtype, buffer=nl.psum)
                nisa.nc_transpose(tp_psum, qk_bsq[:bsq_size, i_sp * TC.p_max : (i_sp + 1) * TC.p_max])
                dst_view = dbg_qk_sprior_major.reshape_dim(1, [fa_ctx.tile_n_sprior, atp.s_active_bqh])[
                    :, i_sp, bsq_start : bsq_start + bsq_size
                ]
                nisa.tensor_copy(dst_view, tp_psum)
        nisa.dma_copy(
            dst,
            dbg_qk_sprior_major.reshape((TC.p_max, 1, fa_ctx.tile_n_sprior, 1, atp.s_active_bqh)),
            dge_mode=dge_mode.none,
            name=f"{sbm.get_name_prefix()}dbg_qk_store_mm1_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
        )
        sbm.close_scope()
    else:
        nisa.dma_copy(
            dst,
            bufs.qk.reshape((TC.p_max, 1, fa_ctx.tile_n_sprior, 1, atp.s_active_bqh)),
            name=f"{sbm.get_name_prefix()}dbg_qk_store_mm1_fa{fa_ctx.fa_tile_idx}_bt{btc.batch_tile_idx}",
        )


def _store_dbg_qk_exp(
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
    fa_ctx,
    btc,
):
    """Store the post-exp scores (bufs.qk_io_type) into DBG_QK_EXP. Shared by the swap and unswapped
    paths — both hold exp in [s_prior, s_active_bqh] layout at this point.

    Skipped (zero-filled) under online softmax, where exp is relative to a per-tile local max rather
    than a stable global max, and for strided_mm1 + batch tiling, where batch/sprior tiles interleave
    in the qk buffer so offset-based writes are unreliable (the swap path asserts non-strided, so that
    clause is a no-op there).
    """
    if DBG_TENSORS and not atp.use_online_softmax and not (cfg.strided_mm1 and atp.bs != atp.bs_per_nc):
        bqh_offset = btc.tile_batch_offset * atp.s_active_qh
        nisa.dma_copy(
            bufs.DBG_QK_EXP[
                :,
                atp.sprior_prg_id,
                :,
                atp.bs_prg_id,
                bqh_offset : bqh_offset + atp.s_active_bqh,
            ],
            bufs.qk_io_type.reshape((TC.p_max, 1, fa_ctx.tile_n_sprior, 1, atp.s_active_bqh)),
            name=f"{sbm.get_name_prefix()}dbg_qk_exp_store_bt{btc.batch_tile_idx}",
        )
    elif DBG_TENSORS and fa_ctx.is_last_fa_tile and btc.batch_tile_idx == 0:
        # Online softmax uses a running max so qk_io_type values aren't meaningful, but the debug
        # tensor must still be written to avoid a compiler error. Only write on the first batch tile;
        # use the debug tensor's full-batch bqh dimension.
        sbm.open_scope()
        full_bqh = bufs.DBG_QK_EXP.shape[-1]
        dbg_zero = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk_io_type.dtype, buffer=nl.sbuf)
        nisa.memset(dbg_zero, 0.0)
        dbg_zero_bc = (dbg_zero).reshape_dim(1, [1, 1, 1, 1]).broadcast(2, atp.n_sprior_tile).broadcast(4, full_bqh)
        nisa.dma_copy(
            bufs.DBG_QK_EXP[:, atp.sprior_prg_id, :, atp.bs_prg_id, :],
            dbg_zero_bc,
            name=sbm.get_name_prefix() + "dbg_qk_exp_store_zeros",
        )
        sbm.close_scope()


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
        name=sbm.get_name_prefix() + "dbg_qk_max_store_zeros_batch_tiling",
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
        name=sbm.get_name_prefix() + "dbg_exp_sum_store_zeros_batch_tiling",
    )
    sbm.close_scope()
