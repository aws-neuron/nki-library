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
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.isa import dge_mode

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.stream_shuffle_broadcast import (
    stream_shuffle_broadcast,
)
from ..utils.tensor_view import TensorView
from ..utils.tp_broadcast import tp_broadcast
from .attention_tkg_utils import (
    AttnTKGConfig,
    TileConstants,
    is_batch_sharded,
    is_fp8_e4m3,
    is_fp8_e5m2,
    resize_cache_block_len_for_attention_tkg_kernel,
)
from .gen_mask_tkg import gen_mask_tkg

_MAX_BS = 128
_MAX_Q = 16
_MAX_S_ACTIVE = 8
_MAX_S_PRIOR = 32 * 1024
_MAX_D_HEAD = 128

"""
Maps s_prior -> list of max bs vs allowed q_head * s_active configuration constraints.

Configuration rules by s_prior threshold:
- sprior <= 256: bs <= 128 and q_head * s_active <= 64
- sprior <= 2k: if bs <= 64, q_head * s_active <= 32, otherwise q_head == s_active == 1
- sprior <= 8k: if bs <= 4, q_head * s_active <= 128; if bs <= 16, q_head * s_active <= 16; otherwise q_head == s_active == 1
- sprior <= 16k: if bs <= 4, q_head * s_active <= 64, otherwise q_head == s_active == 1
- sprior <= 32k: if bs == 1, q_head * s_active <= 128, otherwise bs <= 4 and q_head * s_active <= 16
"""
_ALLOWED_CONFIGURATIONS: list[tuple[int, list[tuple[int, int]]]] = [
    (256, [(_MAX_BS, 64)]),
    (2 * 1024, [(64, 32), (_MAX_BS, 1)]),
    (
        8 * 1024,
        [(4, _MAX_Q * _MAX_S_ACTIVE), (16, 16), (_MAX_BS, 1)],
    ),
    (12 * 1024, [(8, 64), (_MAX_BS, 1)]),
    (16 * 1024, [(4, 64), (_MAX_BS, 1)]),
    (_MAX_S_PRIOR, [(1, _MAX_Q * _MAX_S_ACTIVE), (4, 16)]),
]


def attention_tkg(
    q: nl.ndarray,
    k_active: nl.ndarray,
    v_active: nl.ndarray,
    k_prior: nl.ndarray,
    v_prior: nl.ndarray,
    mask: nl.ndarray,
    out: nl.ndarray,
    cfg: AttnTKGConfig,
    sbm: SbufManager,
    inv_freqs: Optional[nl.ndarray] = None,
    rope_pos_ids: Optional[nl.ndarray] = None,
    sink: Optional[nl.ndarray] = None,
    active_blocks_table: Optional[nl.ndarray] = None,
    k_out: Optional[nl.ndarray] = None,
    DBG_TENSORS: Optional[tuple] = None,
) -> Tuple[nl.ndarray, Optional[nl.ndarray]]:
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
      v_prior: Prior value tensor from KV cache. Shape [B+, 1, s_prior, d].
               For block KV cache, shape is [B+ * block_count, block_len, d] (indexing: [b * block_count + blk, block_len, d])
      mask: Attention mask. Shape [s_active, B, H, s_active] if cfg.use_pos_id else [s_prior, B, H, s_active]
      out: Output tensor.
        Shape: if cfg.out_in_sb:
                [d, B * H * s_active] (indexing: [d, b * H * s_active+ h * s_active + s])
              else:
                [B, H, d, s_active]
      cfg: Kernel configuration with shapes and performance flags. See `AttnTKGConfig`
      sbm: SBUF memory manager for allocating temporary buffers
      inv_freqs: Inverse frequencies for RoPE. Shape [d // 2, 1]. Required when cfg.fuse_rope is True
      rope_pos_ids: Position IDs for RoPE. Shape [B, s_active]. Required when cfg.fuse_rope or cfg.use_pos_id is True
      sink: Sink attention tokens. Shape [H, 1] for streaming attention sink tokens
      active_blocks_table: Table of active blocks for block KV cache. Shape [B, num_blocks].
                          Required when using block KV cache
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
      - tp_k_prior flag indicates whether K_prior is pre-transposed in memory
      - Optimizes memory layout: [B, 1, d, s_prior] when tp_k_prior=True vs [B, 1, s_prior, d] when False
      - Reduces transpose operations during computation and improves interoperatbility with other kernels

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
      Use with s_active <= 7, and with d_head <= 128.

    Notes:
        - KV cache can have batch size larger than B (denoted B+) for garbage data buffering
        - Q is scaled with 1/sqrt(d_head) only when cfg.fuse_rope is True
        - Block KV cache requires qk_in_sb=True
        - In-kernel mask generation (use_pos_id=True) not supported with block KV or batch sharding
        - LNC2 sharding automatically selected based on tensor dimensions
        - Extended GPSIMD instructions require 16-partition alignment

    Pseudocode:
        # Setup
        TC = get_tile_constants()
        atp = compute_tile_params(cfg, TC, q, active_blocks_table)
        bufs = allocate_internal_buffers()

        # Step 0: Optional RoPE
        if cfg.fuse_rope:
            q_sb, k_active_sb = apply_rope(q, k_active, inv_freqs, rope_pos_ids)

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

        store_output(exp_v, out)
    """

    TC = TileConstants.get_tile_constants()
    atp = _compute_tile_params(cfg, TC, q, k_prior, v_prior, k_active, v_active, active_blocks_table)
    bufs = AttnInternalBuffers()

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

    _allocate_qk_buffers(atp, TC, sbm, bufs)
    bufs.one_vec = sbm.alloc_stack((TC.p_max, 1), dtype=bufs.qk_io_type.dtype, buffer=nl.sbuf, align=2)
    nisa.memset(bufs.one_vec, value=1.0)
    _load_position_ids_and_mask(rope_pos_ids, mask, atp, cfg, TC, sbm, bufs)

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

    # Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)
    _compute_qk_matmul(k_prior, k_active, DBG_TENSORS, atp, cfg, TC, sbm, bufs)

    # Step 2. Cascaded max reduce of KQ^T
    _cascaded_max_reduce(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs)

    # Step 3. Exp(KQ^T - max(KQ^T))
    _compute_exp_qk(DBG_TENSORS, atp, TC, bufs)

    # Step 4. Cascaded sum reduction of exp
    _cascaded_sum_reduction(sink, DBG_TENSORS, atp, cfg, TC, sbm, bufs)

    # Step 5. Matmult 2 of (exp @ V)^T
    _compute_pv_matmul_and_store(v_prior, v_active, out, atp, cfg, TC, sbm, bufs)

    return out, k_out


OOB_MODE_SKIP = nisa.oob_mode.skip  # FIXME: needs to be instantiated externally from kernel


@dataclass
class AttnTileParams(nl.NKIObject):
    """Computed tiling and dimension parameters for the attention kernel."""

    # Data types
    io_type = None
    inter_type = None
    k_prior_load_type = None

    # Block KV cache flag
    is_block_kv: bool = None

    # KV FP8 Quantization Flag
    is_fp8_kv: bool = None

    # Sharding parameters
    sprior_n_prgs: int = None
    sprior_prg_id: int = None
    bs_n_prgs: int = None
    bs_prg_id: int = None
    n_prgs: int = None

    # Full batch size (before sharding)
    bs_full: int = None

    # Computed dimensions per program
    bs: int = None  # batch size per program
    s_prior: int = None  # prior seq length per program
    s_active_qh: int = None  # s_active * q_head
    s_active_bqh: int = None  # bs * s_active_qh
    s_active_bqh_remainder: int = None
    n_bsq_full_tiles: int = None
    n_bsq_tiles: int = None
    s_active_bqh_tile: int = None
    n_sprior_tile: int = None

    # Matmul parameters
    batch_interleave_degree: int = None

    # Softmax reduction parameters
    softmax_final_reduction_length: int = None
    softmax_final_reduction_local_idx: int = None
    softmax_final_reduction_lnc_recv_idx: int = None
    softmax_final_reduction_sink_idx: int = None
    max_negated: bool = None

    # Block KV cache parameters (if used)
    block_len: int = None
    num_folds_per_batch: int = None


@dataclass
class AttnInternalBuffers(nl.NKIObject):
    """Internal SBUF buffers needed across multiple steps of the attention kernel."""

    # Core attention tensors
    qk: nl.ndarray = None
    qk_io_type: nl.ndarray = None
    qk_max: nl.ndarray = None
    qk_max_buf: nl.ndarray = None
    exp_sum: nl.ndarray = None
    exp_sum_recip: nl.ndarray = None
    exp_v: nl.ndarray = None

    # Preprocessed inputs
    q_sb: nl.ndarray = None
    k_active_sb: nl.ndarray = None
    pos_ids_sb: nl.ndarray = None
    mask_sb: nl.ndarray = None
    one_vec: nl.ndarray = None

    # Block KV cache buffers
    active_blocks_sb: nl.ndarray = None
    v_active_reshaped: nl.ndarray = None
    k_prior_reshaped: nl.ndarray = None
    v_prior_reshaped: nl.ndarray = None

    # Debug tensors (reshaped from DBG_TENSORS)
    DBG_QK: nl.ndarray = None
    DBG_QK_MAX: nl.ndarray = None
    DBG_QK_EXP: nl.ndarray = None
    DBG_EXP_SUM: nl.ndarray = None
    DBG_ACTIVE_TABLE: nl.ndarray = None


"""
Initialization functions
"""


def _compute_tile_params(
    cfg: AttnTKGConfig, TC: TileConstants, q, k_prior, v_prior, k_active, v_active, active_blocks_table
) -> AttnTileParams:
    """Compute tiling and dimension parameters from configuration."""
    atp = AttnTileParams()

    atp.is_block_kv = active_blocks_table is not None
    atp.block_len = 0  # Default for flat KV cache; overwritten in _setup_block_kv_cache for block KV

    # Determine FP8 KV status
    k_prior_fp8 = is_fp8_e4m3(k_prior.dtype)
    v_prior_fp8 = is_fp8_e4m3(v_prior.dtype)
    k_active_fp8 = is_fp8_e4m3(k_active.dtype)
    v_active_fp8 = is_fp8_e4m3(v_active.dtype)
    any_fp8 = k_prior_fp8 or v_prior_fp8 or k_active_fp8 or v_active_fp8
    all_fp8 = k_prior_fp8 and v_prior_fp8 and k_active_fp8 and v_active_fp8
    atp.is_fp8_kv = all_fp8

    # ========== Input validation (kernel asserts) ==========
    # Basic shape constraints
    kernel_assert(
        0 < cfg.bs <= _MAX_BS,
        f"Unsupported batch size. Got bs={cfg.bs}, must be between 1 and {_MAX_BS}, inclusive.",
    )
    kernel_assert(
        0 < cfg.q_head <= _MAX_Q,
        f"Unsupported q_head. Got q_head={cfg.q_head}, must be between 1 and {_MAX_Q}, inclusive.",
    )
    kernel_assert(
        0 < cfg.s_active <= _MAX_S_ACTIVE,
        f"Unsupported s_active. Got s_active={cfg.s_active}, must be between 1 and {_MAX_S_ACTIVE}, inclusive.",
    )
    kernel_assert(
        0 < cfg.curr_sprior <= _MAX_S_PRIOR,
        f"Unsupported curr_sprior. Got curr_sprior={cfg.curr_sprior}, must be between 1 and {_MAX_S_PRIOR}, inclusive.",
    )
    kernel_assert(
        cfg.curr_sprior <= cfg.full_sprior,
        f"curr_sprior must be <= full_sprior. Got curr_sprior={cfg.curr_sprior}, full_sprior={cfg.full_sprior}.",
    )
    kernel_assert(
        0 < cfg.d_head <= _MAX_D_HEAD,
        f"Unsupported d_head. Got d_head={cfg.d_head}, must be between 1 and {_MAX_D_HEAD}, inclusive.",
    )

    # FP8 dtype validation
    kernel_assert(
        not any_fp8 or all_fp8,
        f"FP8 KV cache requires all KV tensors to have dtype nl.float8_e4m3. "
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
    atp.io_type = q.dtype
    atp.inter_type = nl.float32
    atp.bs_full = cfg.bs
    atp.bs = atp.bs_full // atp.bs_n_prgs
    atp.s_prior = cfg.curr_sprior // atp.sprior_n_prgs  # shard prior seqlen onto each prg
    atp.s_active_qh = cfg.s_active * cfg.q_head  # flattened dim of [q_heads, s_active]
    atp.s_active_bqh = atp.bs * atp.s_active_qh  # flattened dim of [bs, q_heads, s_active]
    atp.s_active_bqh_remainder = atp.s_active_bqh % TC.p_max
    atp.n_bsq_full_tiles = atp.s_active_bqh // TC.p_max
    atp.n_bsq_tiles = atp.n_bsq_full_tiles + (atp.s_active_bqh_remainder > 0)
    atp.s_active_bqh_tile = TC.p_max if atp.n_bsq_tiles > 1 else atp.s_active_bqh
    atp.n_sprior_tile = div_ceil(atp.s_prior, TC.p_max)

    atp.batch_interleave_degree = min(atp.bs, TC.psum_b_max)

    # ========== Derived parameter validation ==========
    kernel_assert(
        atp.s_prior % TC.p_max == 0 or (atp.s_prior <= 256 and atp.sprior_n_prgs == 1),
        f"Sharded s_prior must be divisible by p_max unless s_prior <= 256. "
        f"Got sharded s_prior={atp.s_prior}, p_max={TC.p_max}.",
    )

    # Configuration combination validation
    found_supported_bqs = _ALLOWED_CONFIGURATIONS[0][1]
    for cfg_idx in range(len(_ALLOWED_CONFIGURATIONS)):
        cur_cfg = _ALLOWED_CONFIGURATIONS[cfg_idx]
        found_supported_bqs = cur_cfg[1]
        if cfg.curr_sprior <= cur_cfg[0]:
            break

    found_supported_b, found_supported_qs = None, None
    for bqs_idx in range(len(found_supported_bqs)):
        cur_supported_bqs = found_supported_bqs[bqs_idx]
        found_supported_b = cur_supported_bqs[0]
        found_supported_qs = cur_supported_bqs[1]
        if cfg.bs <= found_supported_b:
            break

    # kernel_assert(
    #     found_supported_b != None and cfg.bs <= found_supported_b and (cfg.q_head * cfg.s_active <= found_supported_qs),
    #     f"No legal configuration found for batch size {cfg.bs}, and q_head * s_active {cfg.q_head * cfg.s_active} at s_prior {cfg.curr_sprior}. "
    #     f"The current paramterization is likely to cause OOM issues. Supported configurations at this s_prior are: {found_supported_bqs} (max_batch_size, max_(q_head * s_active)).",
    # )
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
    # Check shapes
    kernel_assert(
        k_prior.shape[1] == cfg.block_len,
        f"Block KV requires k_prior input must be reshaped to have block_len as the second dimension, expected k_prior.shape[1]={cfg.block_len}, got {k_prior.shape[1]}.",
    )
    kernel_assert(
        k_prior.shape[2] == cfg.d_head,
        f"Block KV requires k_prior input must be reshaped to have d_head as the third dimension, expected k_prior.shape[2]={cfg.d_head}, got {k_prior.shape[2]}.",
    )
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
    block_len, blk_cache_resize_factor = resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_per_batch=active_blocks_table.shape[1],
        block_len=cfg.block_len,
        n_prgs=atp.n_prgs,
        p_max=TC.p_max,
    )
    # assign to atp (can't do directly because function return value cannot be assigned to object)
    atp.block_len = block_len

    new_cache_shape = (
        k_prior.shape[0] * blk_cache_resize_factor,
        atp.block_len * cfg.d_head,
    )
    bufs.k_prior_reshaped = k_prior.reshape(new_cache_shape)
    bufs.v_prior_reshaped = v_prior.reshape(new_cache_shape)

    active_blocks_sb, num_folds_per_batch = _load_and_reshape_active_blk_table(
        active_blocks_table,
        blk_cache_resize_factor,
        atp.sprior_n_prgs,
        atp.sprior_prg_id,
        cfg.bs,
        sbm,
        (atp.bs_prg_id * atp.bs, atp.bs),
    )
    bufs.active_blocks_sb = active_blocks_sb
    atp.num_folds_per_batch = num_folds_per_batch


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
        bufs.active_blocks_sb = bufs.active_blocks_sb.reshape((TC.p_max, atp.num_folds_per_batch, atp.bs))
        kernel_assert(
            bufs.DBG_ACTIVE_TABLE.shape[1] == atp.num_folds_per_batch * atp.sprior_n_prgs,
            "Active table debug tensor second dimension incorrect (needs to have shape (P_MAX, curr_sprior // block_len, batch_size)), "
            f"expected DBG_ACTIVE_TABLE.shape[1]={atp.num_folds_per_batch * atp.sprior_n_prgs}, got {bufs.DBG_ACTIVE_TABLE.shape[1]}",
        )
        kernel_assert(
            bufs.DBG_ACTIVE_TABLE.shape[2] == atp.bs_full,
            "Active table debug tensor third dimension incorrect (needs to have shape (P_MAX, curr_sprior // block_len, batch_size))"
            f"expected DBG_ACTIVE_TABLE.shape[2]={atp.bs_full}, got {bufs.DBG_ACTIVE_TABLE.shape[2]}",
        )
        nisa.dma_copy(
            bufs.DBG_ACTIVE_TABLE[
                :,
                nl.ds(atp.sprior_prg_id * atp.num_folds_per_batch, atp.num_folds_per_batch),
                nl.ds(atp.bs_prg_id * atp.bs, atp.bs),
            ],
            bufs.active_blocks_sb,
            dge_mode=dge_mode.none,
            name="dbg_active_blocks_table_store",
        )
        bufs.active_blocks_sb = bufs.active_blocks_sb.reshape((TC.p_max, atp.num_folds_per_batch * atp.bs))


def _allocate_qk_buffers(atp: AttnTileParams, TC: TileConstants, sbm: SbufManager, bufs: AttnInternalBuffers):
    """Allocate core QK buffers.

    Create KQ^T result mloc for all batches (filled with -INF for masking)
    The tensor has shape [p_max, num_sprior_t128 * bs * s_active_qh], where on the free dimension, there are
      num_sprior_t128 tiles, each tile contains bs number of subtiles, and each subtile is s_active_qh in length.
      I.e., the s_active_qh tiles are interleaved by batch on the free dimension.
    The cascaded max reduce later on will do a strided access on the free dimension.
    """
    bufs.qk = sbm.alloc_stack(
        (TC.p_max, atp.n_sprior_tile * atp.s_active_bqh),
        dtype=atp.inter_type,
    )
    nisa.memset(bufs.qk, value=-np.inf)
    bufs.qk_io_type = sbm.alloc_stack(bufs.qk.shape, dtype=atp.io_type)  # for matmults


def _load_position_ids_and_mask(
    rope_pos_ids,
    mask,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Load position IDs and mask."""
    # Load pos_id while broadcasting onto all 128 partitions (for TensorScalarPtr)
    bufs.pos_ids_sb = None
    if rope_pos_ids is None:
        # only two components that use pos ids
        kernel_assert(
            not cfg.use_pos_id and not cfg.fuse_rope,
            "To generate mask or fuse rope, rope_pos_ids tensor must be provided",
        )
    else:
        kernel_assert(
            atp.bs_n_prgs == 1,
            (
                "In-kernel cache mask generation is not supported when LNC2 shards on batch dimension. "
                "This occurs when batch_size is even and (batch_size * q_head * s_active > 128) or (curr_sprior < 256)."
            ),
        )
        rope_pos_ids = rope_pos_ids.reshape([atp.bs_n_prgs, atp.bs * cfg.s_active])
        rope_pos_ids_loaded = sbm.alloc_stack((1, atp.bs * cfg.s_active), dtype=rope_pos_ids.dtype, align=4)
        nisa.dma_copy(
            rope_pos_ids_loaded, rope_pos_ids[atp.bs_prg_id, :], dge_mode=dge_mode.none, name="rope_pos_ids_load"
        )
        bufs.pos_ids_sb = sbm.alloc_stack((TC.p_max, atp.bs * cfg.s_active), dtype=rope_pos_ids.dtype)
        stream_shuffle_broadcast(src=rope_pos_ids_loaded, dst=bufs.pos_ids_sb)

    # Load/generate mask for all batches, all heads, so should have the same layout and shape as `qk`
    bufs.mask_sb = sbm.alloc_stack(bufs.qk.shape, dtype=nl.uint8, buffer=nl.sbuf)

    # If we don't use pos_id, mask is already generated outside of kernel. Otherwise, generate prior mask in kernel and
    # load active mask at the end of the generated mask.
    if not cfg.use_pos_id:
        # FIXME: This restriction should not exist. Requires FE support.
        # kernel_assert(
        #     mask.shape == (atp.sprior_n_prgs, atp.s_prior, atp.bs_n_prgs, atp.s_active_bqh),
        #     f"Reshaping the mask in the kernel is currently not supported. Expected shape {(atp.sprior_n_prgs, atp.s_prior, atp.bs_n_prgs, atp.s_active_bqh)}.",
        # )
        mask = mask.reshape((atp.sprior_n_prgs, atp.s_prior, atp.bs_n_prgs, atp.s_active_bqh))

        src_load_offset = (atp.sprior_prg_id * atp.s_prior * atp.bs_n_prgs + atp.bs_prg_id) * atp.s_active_bqh
        mask_dst_ap = [
            [atp.n_sprior_tile * atp.s_active_bqh, TC.p_max],
            [atp.s_active_bqh, atp.n_sprior_tile],
            [1, atp.s_active_bqh],
        ]

        # The mask load needs to be strided *unless* mm1 is strided
        if cfg.strided_mm1:
            mask_src_ap = [
                [atp.n_sprior_tile * atp.bs_n_prgs * atp.s_active_bqh, TC.p_max],
                [atp.bs_n_prgs * atp.s_active_bqh, atp.n_sprior_tile],
                [1, atp.s_active_bqh],
            ]
        else:
            mask_src_ap = [
                [atp.bs_n_prgs * atp.s_active_bqh, TC.p_max],
                [TC.p_max * atp.bs_n_prgs * atp.s_active_bqh, atp.n_sprior_tile],
                [1, atp.s_active_bqh],
            ]

        nisa.dma_copy(
            dst=bufs.mask_sb.ap(pattern=mask_dst_ap, offset=0),
            src=mask.ap(pattern=mask_src_ap, offset=src_load_offset),
            dge_mode=dge_mode.none,
            name="mask_load_pregenerated",
        )
    else:
        # In-kernel mask generation supports both flat and block KV cache
        # atp.block_len is 0 for flat KV cache, adjusted block_len for block KV cache
        bufs.mask_sb = bufs.mask_sb.reshape((TC.p_max, atp.n_sprior_tile, atp.bs, cfg.q_head, cfg.s_active))
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1:
            gen_mask_tkg(
                pos_ids=bufs.pos_ids_sb,
                mask_out=bufs.mask_sb,
                bs=atp.bs,
                q_head=cfg.q_head,
                s_active=cfg.s_active,
                block_len=atp.block_len,
                strided_mm1=cfg.strided_mm1,
                active_mask=mask,
                sbm=sbm,
            )
        else:
            gen_mask_tkg(
                pos_ids=bufs.pos_ids_sb,
                mask_out=bufs.mask_sb,
                bs=atp.bs,
                q_head=cfg.q_head,
                s_active=cfg.s_active,
                block_len=atp.block_len,
                strided_mm1=cfg.strided_mm1,
                active_mask=None,
                sbm=sbm,
            )
        bufs.mask_sb = bufs.mask_sb.reshape(bufs.qk.shape)


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

    # If we fuse rope, Q and K_active would need be processed by RoPE first then be stored in Q_sb.
    bufs.q_sb = sbm.alloc_stack(
        (cfg.d_head, atp.bs_n_prgs * atp.bs * atp.s_active_qh),
        dtype=atp.io_type,
        buffer=nl.sbuf,
    )
    bufs.k_active_sb = (
        k_out
        if cfg.k_out_in_sb
        else sbm.alloc_stack(
            (cfg.d_head, atp.bs_n_prgs * atp.bs * cfg.s_active),
            dtype=atp.io_type,
            buffer=nl.sbuf,
        )
    )

    # Load inv_freqs
    inv_freqs_sb = sbm.alloc_stack(inv_freqs.shape, dtype=inv_freqs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(inv_freqs_sb, inv_freqs, dge_mode=dge_mode.none, name="inv_freqs_load")

    # Compute RoPE coefficients, then apply (while loading) onto Q and K_active (only last NC handles K_active)
    cos, sin = _rope(
        inv_freqs_sb,
        bufs.pos_ids_sb,
        bs=atp.bs,
        s_a=cfg.s_active,
        d_head=cfg.d_head,
        sbm=sbm,
    )
    _apply_rope(q, cos, sin, bufs.q_sb, cfg, sbm=sbm, name_suffix="q")
    if cfg.k_out_in_sb or (atp.sprior_prg_id == atp.sprior_n_prgs - 1):
        _apply_rope(k_active, cos, sin, bufs.k_active_sb, cfg, ignore_heads=True, sbm=sbm, name_suffix="k_active")
        # Store K to the second output if not kOutInSB; otherwise we already write to it via name alias to k_active_sb
        if not cfg.k_out_in_sb and k_out is not None:
            k_dram_ap = [
                [cfg.s_active, cfg.d_head],
                [cfg.s_active * cfg.d_head, atp.bs],
                [1, cfg.s_active],
            ]
            k_sbuf_ap = [
                [atp.bs * cfg.s_active, cfg.d_head],
                [cfg.s_active, atp.bs],
                [1, cfg.s_active],
            ]
            nisa.dma_copy(
                src=bufs.k_active_sb.ap(pattern=k_sbuf_ap, offset=0),
                dst=k_out.ap(pattern=k_dram_ap, offset=0),
                dge_mode=dge_mode.none,
                name="k_out_store_after_rope",
            )

    nisa.activation(bufs.q_sb[...], op=nl.copy, data=bufs.q_sb, scale=1 / math.sqrt(cfg.d_head))


"""
Main computation blocks
"""


def _compute_qk_matmul(
    k_prior,
    k_active,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 1. Matmult 1 of KQ^T (and optional K_prior transpose)"""

    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(atp.s_prior, atp.io_type, atp, sbm)

    # Maximum multi-buffer degree inside a batch is 8 (banks) // bs (multi-buffer degree on the current scope)
    per_batch_interleave_degree = math.floor(float(TC.psum_b_max) / batch_interleave_degree_safe)
    sbm.open_scope(interleave_degree=batch_interleave_degree_safe)
    for i_b in range(atp.bs):
        # Load entire K_prior for current batch into sbuf
        k_sb = sbm.alloc_stack((cfg.d_head, atp.s_prior), dtype=k_prior.dtype, buffer=nl.sbuf, align=32)
        if atp.is_block_kv:
            sbm.open_scope()
            for i_fold in range(atp.num_folds_per_batch):
                # FIXME: This should be a slice, but needs to be an AP because of indirect DMA lowering
                cur_blks = bufs.active_blocks_sb.ap(
                    [[atp.bs * atp.num_folds_per_batch, TC.p_max], [1, 1]],
                    offset=i_fold * atp.bs + i_b,
                )
                kernel_assert(
                    cur_blks.shape == (TC.p_max, 1),
                    f"Internal error: unexpected shape error after loading current blocks, expected {(TC.p_max, 1)}, got {cur_blks.shape}.",
                )

                k_loaded = sbm.alloc_stack(
                    (TC.p_max, atp.block_len * cfg.d_head),
                    dtype=atp.k_prior_load_type,
                    buffer=nl.sbuf,
                )

                nisa.memset(k_loaded, value=0)
                nisa.dma_copy(
                    dst=k_loaded,
                    src=bufs.k_prior_reshaped.ap(
                        [
                            [atp.block_len * cfg.d_head, TC.p_max],
                            [1, atp.block_len * cfg.d_head],
                        ],
                        offset=0,
                        vector_offset=cur_blks,
                        indirect_dim=0,
                    ),
                    oob_mode=OOB_MODE_SKIP,
                    name=f"k_prior_block_load_indirect_b{i_b}_f{i_fold}",
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
                                (tp_j_in_grp % per_batch_interleave_degree) * TC.psum_f_max_bytes,
                            ),
                        )
                        nisa.nc_transpose(
                            tp_psum,
                            k_loaded[:, nl.ds(blk_len_i * cfg.d_head, cfg.d_head)],
                        )

                        # Not balancing at the moment due to vector engine being busy with memset 'K_loaded' to zeros.
                        # When balancing disabled, scalar engine is used.
                        # FIXME: revisit balancing when the kernel is integrated because the memset by vector engine may be scheduled much ahead.
                        psum2sbuf_copy_engine_balance = False
                        if psum2sbuf_copy_engine_balance and tp_grp_i % 2 == 0:
                            engine = nisa.vector_engine
                        else:
                            engine = nisa.scalar_engine

                        cur_idx = i_fold * atp.block_len + blk_len_i
                        nisa.tensor_copy(k_sb[:, nl.ds(cur_idx * 128, 128)], tp_psum, engine=engine)
            sbm.close_scope()
        elif not cfg.tp_k_prior:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.d_head, cfg.full_sprior),
                f"k_prior[1:] expected to have shape {(1, cfg.d_head, cfg.full_sprior)=}, received {k_prior.shape[1:]=}",
            )
            # K_prior is already transposed, insert flat load
            # FIXME: This should be a simple slice, but needs to be AP to make both accesses 2d
            nisa.dma_copy(
                k_sb,
                k_prior.ap(
                    [[cfg.full_sprior, cfg.d_head], [1, atp.s_prior]],
                    offset=(i_b + atp.bs_prg_id * atp.bs) * cfg.d_head * cfg.full_sprior
                    + atp.sprior_prg_id * atp.s_prior,
                ),
                dge_mode=dge_mode.none,
                name=f"k_prior_flat_load_transposed_b{i_b}",
            )
        else:
            kernel_assert(
                k_prior.shape[1:] == (1, cfg.full_sprior, cfg.d_head),
                f"k_prior[1:] expected to have shape {(1, cfg.full_sprior, cfg.d_head)=}, received {k_prior.shape[1:]=}",
            )

            if atp.is_fp8_kv:
                # Can't do DMA transpose for FP8, so load as BF16, transpose via PSUM, copy to k_sb (casts to FP8)
                batch_offset = (i_b + atp.bs_prg_id * atp.bs) * cfg.full_sprior + atp.sprior_prg_id * atp.s_prior
                sbm.open_scope(interleave_degree=TC.psum_b_max)
                for tp_grp_i in range(atp.n_sprior_tile):
                    tile_start = tp_grp_i * TC.p_max
                    tile_size = min(TC.p_max, atp.s_prior - tile_start)
                    k_loaded = sbm.alloc_stack((TC.p_max, cfg.d_head), dtype=atp.k_prior_load_type, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=k_loaded[:tile_size, :],
                        src=k_prior.ap(
                            [[cfg.d_head, tile_size], [1, cfg.d_head]],
                            offset=(batch_offset + tile_start) * cfg.d_head,
                        ),
                    )
                    tp_psum = nl.ndarray(
                        (cfg.d_head, TC.p_max),
                        dtype=atp.k_prior_load_type,
                        buffer=nl.psum,
                        address=(0, (tp_grp_i % TC.psum_b_max) * TC.psum_f_max_bytes),
                    )
                    nisa.nc_transpose(tp_psum[:, :tile_size], k_loaded[:tile_size, :])
                    nisa.tensor_copy(k_sb[:, nl.ds(tile_start, tile_size)], tp_psum[:, :tile_size])
                sbm.close_scope()

            else:
                # FIXME: 4d access pattern required here, while simple slicing should suffice
                nisa.dma_transpose(
                    k_sb.ap([[atp.s_prior, cfg.d_head], [1, 1], [1, 1], [1, atp.s_prior]]),
                    k_prior.ap(
                        [[cfg.d_head, atp.s_prior], [1, 1], [1, 1], [1, cfg.d_head]],
                        offset=((i_b + atp.bs_prg_id * atp.bs) * cfg.full_sprior + atp.sprior_prg_id * atp.s_prior)
                        * cfg.d_head,
                    ),
                )

        # If on final NC, add K_active to the end of k_sb
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1:
            if atp.is_block_kv:
                num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
                extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active

                dst_ap = [
                    [atp.s_prior, cfg.d_head],
                    [1, num_blks_covering_s_active],
                    [TC.p_max, atp.block_len],
                ]
                dst_offset = (
                    (atp.num_folds_per_batch - 1) * atp.block_len * TC.p_max + TC.p_max - num_blks_covering_s_active
                )

                src_ap = [
                    [atp.bs_full * cfg.s_active, cfg.d_head],
                    [atp.block_len, num_blks_covering_s_active],
                    [1, atp.block_len],
                ]
                src_offset = (atp.bs_prg_id * atp.bs + i_b) * cfg.s_active - extra_covered

                # Need to mask as dim_1 * blk_len + dim_2 >= extra_covered
                # Solving the above inequality with 0 <= dim_1 < num_blks_covering_s_active and 0 <= dim_2 < blk_len
                # we get (dim_1, dim_2) in {(0,[extra_covered, blk_len)) and ([1, num_blks_covering_s_active), [0, blk_len))}
                # Thus, if extra_covered != 0, we do an access pattern for dim_1 == 0 and dim_2 in [extra_covered, blk_len)
                # and for main copy we don't need any restrictions
                if extra_covered > 0:
                    # Remove ap[1] and change ap[2] count. The corresponding increase goes into the offset.
                    if dst_ap[2][1] > extra_covered:
                        nisa.tensor_copy(
                            dst=k_sb.ap(
                                pattern=[
                                    dst_ap[0],
                                    [dst_ap[2][0], dst_ap[2][1] - extra_covered],
                                ],
                                offset=dst_offset + dst_ap[2][0] * extra_covered,
                            ),
                            src=k_active.ap(
                                pattern=[
                                    src_ap[0],
                                    [src_ap[2][0], src_ap[2][1] - extra_covered],
                                ],
                                offset=src_offset + src_ap[2][0] * extra_covered,
                            ),
                        )
                    # Start ap[1] at 1 instead of 0 (affects ap and offset)
                    if dst_ap[1][1] > 1:
                        nisa.tensor_copy(
                            dst=k_sb.ap(
                                pattern=[
                                    dst_ap[0],
                                    [dst_ap[1][0], dst_ap[1][1] - 1],
                                    dst_ap[2],
                                ],
                                offset=dst_offset + dst_ap[1][0],
                            ),
                            src=k_active.ap(
                                pattern=[
                                    src_ap[0],
                                    [src_ap[1][0], src_ap[1][1] - 1],
                                    src_ap[2],
                                ],
                                offset=src_offset + src_ap[1][0],
                            ),
                        )
                else:
                    nisa.tensor_copy(
                        dst=k_sb.ap(pattern=dst_ap, offset=dst_offset),
                        src=k_active.ap(pattern=src_ap, offset=src_offset),
                    )
            else:
                nisa.tensor_copy(
                    k_sb[:, atp.s_prior - cfg.s_active : atp.s_prior],
                    bufs.k_active_sb[
                        :,
                        nl.ds((atp.bs_prg_id * atp.bs + i_b) * cfg.s_active, cfg.s_active),
                    ],
                    engine=nisa.scalar_engine,
                )

        # Do MM1 in grps (default 4k grp size), make sure appropriate group size is selected s.t. psum free < hw limit
        mm1_grp_sz = 4 * 1024
        if (mm1_grp_sz // TC.p_max) * atp.s_active_qh > TC.psum_f_max:
            mm1_grp_sz = (TC.psum_f_max // atp.s_active_qh) * TC.p_max
        n_mm1_per_grp = mm1_grp_sz // TC.p_max

        """
        Tiling Strategy for MM1 (KQ^T computation):
        - K stationary: [d_head, s_prior] loaded per batch into k_sb
        - Q moving: [d_head, s_active_qh] per batch from q_sb
        - Tile size: mm1_grp_sz (default 4096 = 4k) to balance PSUM usage
        - PSUM allocation: [P_MAX, n_mm1_per_grp * s_active_qh]
          where n_mm1_per_grp = mm1_grp_sz / P_MAX
        - PSUM constraint: (mm1_grp_sz / P_MAX) * s_active_qh < psum_f_max
        - Output: qk [P_MAX, n_sprior_tile * s_active_bqh] with batch interleaving
        - Memory: Each tile processes P_MAX rows of K against full Q per batch
        """

        sbm.open_scope(interleave_degree=per_batch_interleave_degree)
        for i_mm1_grp in range(div_ceil(atp.s_prior, mm1_grp_sz)):
            # Perform MM for this (4k) tile in MMs of p_max (limited to p_max as K is stationary)
            # The psum can store entire output for a tile which only needs (grp_sz / p_max) * s_active_qh free dim
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

            # Do (mm1_grp_sz / p_max) matmults, note mm1_grp_sz is divisible by p_max
            for i_mm1 in range(n_mm1_per_grp):
                if (
                    cfg.strided_mm1
                ):  # optionally use strided read to K s.t. MM2 can also be strided with sequential read to V
                    k_tile_offset = i_mm1_grp * n_mm1_per_grp + i_mm1
                    num_acc = min(
                        TC.p_max,
                        (atp.s_prior - 1 - k_tile_offset) // atp.n_sprior_tile + 1,
                    )
                    k_tile = k_sb.ap(
                        pattern=[
                            [atp.s_prior, cfg.d_head],
                            [atp.n_sprior_tile, num_acc],
                        ],
                        offset=k_tile_offset,
                    )
                else:
                    k_tile_offset = i_mm1_grp * mm1_grp_sz + i_mm1 * TC.p_max
                    num_acc = min(TC.p_max, atp.s_prior - k_tile_offset)
                    if num_acc <= 0:
                        break  # k_tile_offset is strictly increasing
                    k_tile = k_sb[0 : cfg.d_head, nl.ds(k_tile_offset, num_acc)]
                qk_psum_pat = [
                    [n_mm1_per_grp * atp.s_active_qh, num_acc],
                    [1, atp.s_active_qh],
                ]

                q_sb_pat = [
                    [atp.bs_full * atp.s_active_qh, cfg.d_head],
                    [1, atp.s_active_qh],
                ]
                q_sb_offset = (atp.bs_prg_id * atp.bs + i_b) * atp.s_active_qh

                nisa.nc_matmul(
                    qk_psum.ap(qk_psum_pat, offset=i_mm1 * atp.s_active_qh),
                    stationary=k_tile,  # mask k_sb
                    moving=bufs.q_sb.ap(q_sb_pat, q_sb_offset),
                )

            # Flush psum -> sb, the write to sb needs to be strided for batch interleaving
            num_acc_cpy = min(n_mm1_per_grp, atp.s_prior // TC.p_max - i_mm1_grp * n_mm1_per_grp)
            src_ap = [
                [n_mm1_per_grp * atp.s_active_qh, TC.p_max],
                [atp.s_active_qh, num_acc_cpy],
                [1, atp.s_active_qh],
            ]
            dst_ap = [
                [atp.n_sprior_tile * atp.s_active_bqh, TC.p_max],
                [atp.s_active_bqh, num_acc_cpy],
                [1, atp.s_active_qh],
            ]
            dst_offset = i_mm1_grp * (n_mm1_per_grp * atp.s_active_bqh) + i_b * atp.s_active_qh

            nisa.tensor_copy_predicated(
                src=qk_psum.ap(pattern=src_ap, offset=0),
                dst=bufs.qk.ap(pattern=dst_ap, offset=dst_offset),
                predicate=bufs.mask_sb.ap(pattern=dst_ap, offset=dst_offset),
            )
            sbm.increment_section()
        sbm.close_scope()
        sbm.increment_section()
    sbm.close_scope()

    if DBG_TENSORS:
        nisa.dma_copy(
            bufs.DBG_QK[:, atp.sprior_prg_id, :, atp.bs_prg_id, :],
            bufs.qk.reshape((TC.p_max, 1, atp.n_sprior_tile, 1, atp.s_active_bqh)),
            dge_mode=dge_mode.none,
            name="dbg_qk_store_mm1",
        )


def _cascaded_max_reduce(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 2. Cascaded max reduce of KQ^T"""
    bufs.qk_max = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf, align=4)

    # Step 2.1. Strided reduce from [p_max, n_sprior_tile * bs * s_active_qh] -> [p_max, bs * s_active_qh]
    # This is small (e.g. if n=2, s_a=6, s_p=8192, then free dim is 64*12=768), reasonable to be done with one inst
    qk_pat = bufs.qk.ap(
        pattern=[
            [atp.n_sprior_tile * atp.s_active_bqh, TC.p_max],
            [1, atp.s_active_bqh],
            [atp.s_active_bqh, atp.n_sprior_tile],
        ],
        offset=0,
    )
    nisa.tensor_reduce(
        dst=bufs.qk_max[...], op=nl.maximum, data=qk_pat, axis=[4], keepdims=False
    )  # The axis is modified here

    # The free-dim length holding max/sum reduction values from LNC2 cores and sink.
    atp.softmax_final_reduction_length = 1 + (atp.sprior_n_prgs > 1) + (sink is not None)
    atp.softmax_final_reduction_local_idx = 0  # The reduction result from local qk goes to 1st entry.
    atp.softmax_final_reduction_lnc_recv_idx = 1  # Always 2nd entry.
    atp.softmax_final_reduction_sink_idx = atp.softmax_final_reduction_length - 1  # sink always in last entry.

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

    # Step 2.3.1  If more than one NC, add send/recv
    if atp.sprior_n_prgs > 1:
        local_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_local_idx
        recv_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_lnc_recv_idx
        nisa.sendrecv(
            src=bufs.qk_max_buf[:, nl.ds(local_offset, atp.n_bsq_tiles)],
            dst=bufs.qk_max_buf[:, nl.ds(recv_offset, atp.n_bsq_tiles)],
            send_to_rank=(1 - atp.sprior_prg_id),
            recv_from_rank=(1 - atp.sprior_prg_id),
            pipe_id=0,
            use_gpsimd_dma=cfg.use_gpsimd_sb2sb,
        )
    # Step 2.3.2  If there is sink, load with the right layout.
    if sink is not None:
        sink_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        _prep_sink(
            sink,
            bufs.qk_max_buf[: atp.s_active_bqh_tile, nl.ds(sink_offset, atp.n_bsq_tiles)],
            atp,
            cfg,
            TC,
            sbm,
        )
    # Step 2.3.3  Do the final reduction (2 or 3 reduce to 1) -> [bs * s_active_qh, 1]
    #             Negate if we are doing the reduction to save one op for sink exponential.
    atp.max_negated = False
    if atp.softmax_final_reduction_length > 1:
        atp.max_negated = True
        for i_bsq_tile in range(atp.n_bsq_tiles):
            nisa.tensor_reduce(
                bufs.qk_max_buf[: atp.s_active_bqh_tile, i_bsq_tile],
                data=bufs.qk_max_buf.ap(
                    pattern=[
                        [
                            atp.n_bsq_tiles * atp.softmax_final_reduction_length,
                            atp.s_active_bqh_tile,
                        ],
                        [atp.n_bsq_tiles, atp.softmax_final_reduction_length],
                    ],
                    offset=i_bsq_tile,
                ),
                op=nl.maximum,
                axis=1,
                negate=True,
            )

    # Step 2.4. Tranpose and broadcast along pdim -> [128, bs * s_active_qh]
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _transpose_broadcast_max(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, sbm, bufs)

    if atp.s_active_bqh_remainder > 0:
        _transpose_broadcast_max(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, sbm, bufs)

    if DBG_TENSORS:
        sbm.open_scope()
        qk_max_dbg_psum = nl.ndarray(
            (atp.n_bsq_tiles, atp.s_active_bqh_tile),
            dtype=bufs.qk_max_buf.dtype,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, 0),
        )
        qk_max_dbg = sbm.alloc_stack((atp.n_bsq_tiles, atp.s_active_bqh_tile), dtype=bufs.qk_max_buf.dtype)
        nisa.nc_transpose(
            qk_max_dbg_psum,
            bufs.qk_max_buf[
                : atp.s_active_bqh_tile,
                nl.ds(
                    atp.softmax_final_reduction_local_idx * atp.n_bsq_tiles,
                    atp.n_bsq_tiles,
                ),
            ],
        )
        if not atp.max_negated:
            nisa.tensor_scalar(qk_max_dbg, qk_max_dbg_psum, op0=nl.multiply, operand0=-1)
        else:
            nisa.tensor_copy(qk_max_dbg, qk_max_dbg_psum)
        nisa.dma_copy(
            bufs.DBG_QK_MAX.ap(
                [[atp.s_active_bqh_tile, atp.n_bsq_tiles], [1, atp.s_active_bqh_tile]],
                offset=atp.bs_prg_id * atp.n_bsq_tiles * atp.s_active_bqh_tile,
            ),
            qk_max_dbg,
            dge_mode=dge_mode.none,
            name="dbg_qk_max_store",
        )
        sbm.close_scope()


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
    # qk_max_buf_pat = qk_max_buf_pat.ap([[1, ]])
    qk_max_buf_copy = sbm.alloc_stack((tile_size, 1), dtype=bufs.qk_max_buf.dtype)
    qk_max_copy = sbm.alloc_stack((TC.p_max, tile_size), dtype=bufs.qk_max.dtype)
    nisa.tensor_copy(
        qk_max_buf_copy,
        bufs.qk_max_buf[:tile_size, nl.ds(index, 1)],
    )
    # FIXME: a hack was put into the tp_broadcast
    tp_broadcast(
        src=qk_max_buf_copy, dst=qk_max_copy, src_offset=0, psum_address=None if sbm.is_auto_alloc() else (0, 0)
    )
    nisa.tensor_copy(
        bufs.qk_max[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
        qk_max_copy,
    )
    sbm.close_scope()


def _compute_exp_qk(DBG_TENSORS, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers):
    """Step 3. Exp(KQ^T - max(KQ^T))"""

    qk_f_size = atp.n_sprior_tile * atp.s_active_bqh
    for i_s_prior in range(atp.n_sprior_tile):
        qk_pat = bufs.qk.ap(
            pattern=[
                [qk_f_size, TC.p_max],
                [atp.s_active_bqh, 1],
                [1, atp.s_active_bqh],
            ],
            offset=i_s_prior * atp.s_active_bqh,
        )

        qk_io_type_pat = bufs.qk_io_type.ap(
            pattern=[
                [qk_f_size, TC.p_max],
                [atp.s_active_bqh, 1],
                [1, atp.s_active_bqh],
            ],
            offset=i_s_prior * atp.s_active_bqh,
        )

        nisa.tensor_tensor(
            qk_pat, qk_pat, bufs.qk_max[...], op=(nl.add if atp.max_negated else nl.subtract)
        )  # rhs is broadcasted

        nisa.activation(qk_io_type_pat, op=nl.exp, data=qk_pat)

    if DBG_TENSORS:
        nisa.dma_copy(
            bufs.DBG_QK_EXP[:, atp.sprior_prg_id, :, atp.bs_prg_id, :],
            bufs.qk_io_type.reshape((TC.p_max, 1, atp.n_sprior_tile, 1, atp.s_active_bqh)),
            dge_mode=dge_mode.none,
            name="dbg_qk_exp_store",
        )


def _cascaded_sum_reduction(
    sink,
    DBG_TENSORS,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
    bufs: AttnInternalBuffers,
):
    """Step 4. Cascaded sum reduction of exp"""
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
    bufs.exp_sum_recip = sbm.alloc_stack((TC.p_max, atp.s_active_bqh), dtype=atp.inter_type, buffer=nl.sbuf)

    sbm.open_scope()
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_sum_reduction(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, bufs, sbm)
    sbm.close_scope()

    if atp.s_active_bqh_remainder > 0:
        sbm.open_scope()
        _tile_sum_reduction(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, bufs, sbm)
        sbm.close_scope()

    # If more than one NC, add send/recv and reduce again
    if atp.sprior_n_prgs > 1:
        local_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_local_idx
        recv_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_lnc_recv_idx
        nisa.sendrecv(
            src=bufs.exp_sum[:, nl.ds(local_offset, atp.n_bsq_tiles)],
            dst=bufs.exp_sum[:, nl.ds(recv_offset, atp.n_bsq_tiles)],
            send_to_rank=(1 - atp.sprior_prg_id),
            recv_from_rank=(1 - atp.sprior_prg_id),
            pipe_id=0,
        )

    if sink is not None:
        kernel_assert(
            atp.max_negated,
            "Internal error: Unexpectedly found that maximum has not been negated when using sink",
        )
        reduction_offset = atp.n_bsq_tiles * atp.softmax_final_reduction_sink_idx
        for i_bsq_tile in range(atp.n_bsq_tiles):
            nisa.tensor_scalar(
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.add,
                bufs.qk_max_buf[: atp.s_active_bqh_tile, i_bsq_tile],
            )
            nisa.activation(
                bufs.exp_sum[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
                nl.exp,
                bufs.qk_max_buf[: atp.s_active_bqh_tile, reduction_offset + i_bsq_tile],
            )

    if atp.softmax_final_reduction_length > 1:
        for i_bsq_tile in range(atp.n_bsq_tiles):
            nisa.tensor_reduce(
                bufs.exp_sum[: atp.s_active_bqh_tile, i_bsq_tile],
                data=bufs.exp_sum.ap(
                    [
                        [
                            atp.n_bsq_tiles * atp.softmax_final_reduction_length,
                            atp.s_active_bqh_tile,
                        ],
                        [atp.n_bsq_tiles, atp.softmax_final_reduction_length],
                    ],
                    offset=i_bsq_tile,
                ),
                op=nl.add,
                axis=1,
            )

    if DBG_TENSORS:
        sbm.open_scope()
        exp_sum_dbg_psum = nl.ndarray(
            (atp.n_bsq_tiles, atp.s_active_bqh_tile),
            dtype=bufs.exp_sum.dtype,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, 0),
        )
        exp_sum_dbg = sbm.alloc_stack((atp.n_bsq_tiles, atp.s_active_bqh_tile), dtype=bufs.exp_sum.dtype)
        nisa.nc_transpose(
            exp_sum_dbg_psum,
            bufs.exp_sum[
                : atp.s_active_bqh_tile,
                nl.ds(
                    atp.softmax_final_reduction_local_idx * atp.n_bsq_tiles,
                    atp.n_bsq_tiles,
                ),
            ],
        )
        nisa.tensor_copy(exp_sum_dbg, exp_sum_dbg_psum)
        nisa.dma_copy(
            dst=bufs.DBG_EXP_SUM.ap(
                [[atp.s_active_bqh_tile, atp.n_bsq_tiles], [1, atp.s_active_bqh_tile]],
                offset=atp.bs_prg_id * atp.n_bsq_tiles * atp.s_active_bqh_tile,
            ),
            src=exp_sum_dbg,
            dge_mode=dge_mode.none,
            name="dbg_exp_sum_store",
        )
        sbm.close_scope()

    # Take sum recip, transpose and broadcast on pdim
    nisa.reciprocal(
        bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
        bufs.exp_sum[: atp.s_active_bqh_tile, : atp.n_bsq_tiles],
    )

    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_reciprocal_transpose(i_bsq_tile, atp.s_active_bqh_tile, atp, TC, bufs)

    if atp.s_active_bqh_remainder > 0:
        _tile_reciprocal_transpose(atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, atp, TC, bufs)

    stream_shuffle_broadcast(src=bufs.exp_sum_recip[:1, : atp.s_active_bqh], dst=bufs.exp_sum_recip)


def _tile_sum_reduction(
    index, tile_size, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers, sbm: SbufManager
):
    """
    Step 4.1. Each of the n_sprior_tile matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    Step 4.2. Copy partial reduce output from psum -> sb while reducing the free dim (num_sprior_t128)
    """
    sum_reduce_psum = nl.ndarray(
        (tile_size, atp.n_sprior_tile),
        dtype=nl.float32,
        buffer=nl.psum,
        address=None if sbm.is_auto_alloc() else (0, (index % TC.psum_b_max) * TC.psum_f_max_bytes),
    )

    # Step 4.1. Each of the n_sprior_tile matmult reduces one tile of qk[128(P), 1, s] -> [s, 1]
    for i_exp_reduce in range(atp.n_sprior_tile):
        sum_reduce_psum_pat = [[atp.n_sprior_tile, tile_size], [1, 1]]
        qk_io_pat = [
            [atp.n_sprior_tile * atp.s_active_bqh, TC.p_max],
            [1, tile_size],
        ]
        qk_io_offset = index * atp.s_active_bqh_tile + i_exp_reduce * atp.s_active_bqh

        nisa.nc_matmul(
            sum_reduce_psum.ap(sum_reduce_psum_pat, offset=i_exp_reduce),
            stationary=bufs.qk_io_type.ap(qk_io_pat, qk_io_offset),
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


def _tile_reciprocal_transpose(index, tile_size, atp: AttnTileParams, TC: TileConstants, bufs: AttnInternalBuffers):
    base_offset = index * atp.s_active_bqh_tile
    for quadrant_idx in range(div_ceil(tile_size, TC.sbuf_quadrant_size)):
        offset = quadrant_idx * TC.sbuf_quadrant_size
        full_offset = base_offset + offset
        tp_size = min(TC.sbuf_quadrant_size, tile_size - offset)
        # Even though TP on vector engine is slower, the vector engine is not busy while the tensor engine is
        nisa.nc_transpose(
            bufs.exp_sum_recip[:1, full_offset : full_offset + tp_size],
            bufs.exp_sum[offset : offset + tp_size, index : index + 1],
            engine=nisa.vector_engine,
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
):
    """Step 5. Matmult 2 of (exp @ V)^T and store output"""
    exp_v_sendrecv_gpsimd = cfg.use_gpsimd_sb2sb and atp.sprior_n_prgs > 1 and atp.bs * atp.s_active_qh <= 256
    if exp_v_sendrecv_gpsimd:
        # Extended instructions require input/output tensors have multiple of 16 partitions
        padded_exp_v_pdim = pad_partitions_for_ext_inst(cfg.d_head)
    else:
        padded_exp_v_pdim = cfg.d_head
    bufs.exp_v = sbm.alloc_stack(
        (padded_exp_v_pdim, atp.bs, atp.s_active_qh),
        dtype=atp.inter_type,
        buffer=nl.sbuf,
    )

    batch_interleave_degree_safe = _get_safe_batch_interleave_degree(
        cfg.d_head * atp.n_sprior_tile, atp.io_type, atp, sbm
    )

    """
    Tiling Strategy for MM2 ((exp @ V)^T computation and output):
    - V stationary: [s_prior, d_head] loaded per batch into v_sb as [P_MAX, n_sprior_tile * d_head]
    - exp(QK) moving: [P_MAX, s_active_bqh] from qk_io_type (already computed and normalized)
    - Output: exp_v [d_head, bs, s_active_qh] accumulated in PSUM then copied to SBUF
    - PSUM allocation: [d_head, s_active_qh] per batch
    - Memory layout: V loaded horizontally tiled (strided if strided_mm1=False, sequential if True)
    - Batch interleaving: Uses batch_interleave_degree_safe for DMA/compute overlap
    - Final output: Gathered across cores if sprior_n_prgs > 1, then stored to HBM or kept in SBUF
    """

    sbm.open_scope(interleave_degree=batch_interleave_degree_safe)
    for i_b in range(atp.bs):
        # Load V_prior from HBM [s_prior, d_head] into SB [128, (s_prior / 128) * d_head]
        # Do strided load (horizontal tile) if not strided_mm1, otherwise load sequentially for better DMA throughput
        v_sb = sbm.alloc_stack(
            (TC.p_max, cfg.d_head * atp.n_sprior_tile),
            dtype=v_prior.dtype,
            buffer=nl.sbuf,
        )
        v_sb_pat = v_sb.ap(
            pattern=[
                [cfg.d_head * atp.n_sprior_tile, TC.p_max],
                [cfg.d_head, atp.n_sprior_tile],
                [1, cfg.d_head],
            ],
            offset=0,
        )
        if atp.is_block_kv:
            sbm.open_scope()
            for i_fold in range(atp.num_folds_per_batch):
                # FIXME: This should be a slice, but needs to be an AP because of indirect DMA lowering
                cur_blks = bufs.active_blocks_sb.ap(
                    [[atp.bs * atp.num_folds_per_batch, TC.p_max], [1, 1]],
                    offset=i_fold * atp.bs + i_b,
                )
                kernel_assert(
                    cur_blks.shape == (TC.p_max, 1),
                    f"Internal error: unexpected shape error after loading current blocks, expected {(TC.p_max, 1)}, got {cur_blks.shape}.",
                )
                nisa.dma_copy(
                    dst=v_sb[
                        :,
                        nl.ds(
                            i_fold * atp.block_len * cfg.d_head,
                            atp.block_len * cfg.d_head,
                        ),
                    ],
                    src=bufs.v_prior_reshaped.ap(
                        [
                            [atp.block_len * cfg.d_head, TC.p_max],
                            [1, atp.block_len * cfg.d_head],
                        ],
                        offset=0,
                        vector_offset=cur_blks,
                        indirect_dim=0,
                    ),
                    oob_mode=OOB_MODE_SKIP,
                    name=f"v_prior_block_load_indirect_b{i_b}_f{i_fold}",
                )
            sbm.close_scope()
        elif cfg.strided_mm1:
            v_prior_offset = (
                i_b + atp.bs_prg_id * atp.bs
            ) * cfg.full_sprior * cfg.d_head + atp.sprior_prg_id * atp.s_prior * cfg.d_head
            v_prior_pat = v_prior.ap(
                pattern=[
                    [atp.n_sprior_tile * cfg.d_head, TC.p_max],
                    [cfg.d_head, atp.n_sprior_tile],
                    [1, cfg.d_head],
                ],
                offset=v_prior_offset,
            )
            nisa.dma_copy(v_sb_pat, v_prior_pat, dge_mode=dge_mode.none, name=f"v_prior_load_strided_mm1_b{i_b}")
        else:
            v_prior_offset = (
                i_b + atp.bs_prg_id * atp.bs
            ) * cfg.full_sprior * cfg.d_head + atp.sprior_prg_id * atp.s_prior * cfg.d_head
            v_prior_pat = v_prior.ap(
                pattern=[
                    [cfg.d_head, TC.p_max],
                    [TC.p_max * cfg.d_head, atp.n_sprior_tile],
                    [1, cfg.d_head],
                ],
                offset=v_prior_offset,
            )
            nisa.dma_copy(dst=v_sb_pat, src=v_prior_pat, dge_mode=dge_mode.none, name=f"v_prior_load_sequential_b{i_b}")

        # Load V_active to the last portion if needed
        if atp.sprior_prg_id == atp.sprior_n_prgs - 1:
            if atp.is_block_kv:
                num_blks_covering_s_active = div_ceil(cfg.s_active, atp.block_len)
                extra_covered = num_blks_covering_s_active * atp.block_len - cfg.s_active

                dst_ap = [
                    [atp.n_sprior_tile * cfg.d_head, num_blks_covering_s_active],
                    [cfg.d_head, atp.block_len],
                    [1, cfg.d_head],
                ]
                dst_offset = (
                    (TC.p_max - num_blks_covering_s_active) * atp.n_sprior_tile
                    + (atp.num_folds_per_batch - 1) * atp.block_len
                ) * cfg.d_head

                src_ap = [
                    [atp.block_len * cfg.d_head, num_blks_covering_s_active],
                    [cfg.d_head, atp.block_len],
                    [1, cfg.d_head],
                ]
                src_offset = ((atp.bs_prg_id * atp.bs + i_b) * cfg.s_active - extra_covered) * cfg.d_head

                # Need to mask as dim_0 * blk_len + dim_1 >= extra_covered
                # Solving the above inequality with 0 <= dim_0 < num_blks_covering_s_active and 0 <= dim_1 < blk_len
                # we get (dim_0, dim_1) in {(0,[extra_covered, blk_len)) and ([1, num_blks_covering_s_active), [0, blk_len))}
                # Thus, if extra_covered != 0, we do an access pattern for dim_0 == 0 and dim_1 in [extra_covered, blk_len)
                # and for main copy we don't need any restrictions
                if extra_covered > 0:
                    # Remove ap[0] and change ap[1] count. The corresponding increase goes into the offset.
                    # Notice that "removing" the partition dimension implies setting count to zero. You still need it in the ap.
                    if dst_ap[1][1] > extra_covered:
                        nisa.dma_copy(
                            dst=v_sb.ap(
                                pattern=[
                                    [dst_ap[0][0], 1],
                                    [dst_ap[1][0], dst_ap[1][1] - extra_covered],
                                    dst_ap[2],
                                ],
                                offset=dst_offset + dst_ap[1][0] * extra_covered,
                            ),
                            src=bufs.v_active_reshaped.ap(
                                pattern=[
                                    [src_ap[0][0], 1],
                                    [src_ap[1][0], src_ap[1][1] - extra_covered],
                                    src_ap[2],
                                ],
                                offset=src_offset + src_ap[1][0] * extra_covered,
                            ),
                            dge_mode=dge_mode.none,
                            name=f"v_active_block_load_partial_rows_b{i_b}",
                        )
                    # Start ap[0] at 1 instead of 0 (affects ap and offset)
                    if dst_ap[0][1] > 1:
                        nisa.dma_copy(
                            dst=v_sb.ap(
                                pattern=[
                                    [dst_ap[0][0], dst_ap[0][1] - 1],
                                    dst_ap[1],
                                    dst_ap[2],
                                ],
                                offset=dst_offset + dst_ap[0][0],
                            ),
                            src=bufs.v_active_reshaped.ap(
                                pattern=[
                                    [src_ap[0][0], src_ap[0][1] - 1],
                                    src_ap[1],
                                    src_ap[2],
                                ],
                                offset=src_offset + src_ap[0][0],
                            ),
                            dge_mode=dge_mode.none,
                            name=f"v_active_block_load_remaining_blocks_b{i_b}",
                        )
                else:
                    nisa.dma_copy(
                        dst=v_sb.ap(pattern=dst_ap, offset=dst_offset),
                        src=bufs.v_active_reshaped.ap(pattern=src_ap, offset=src_offset),
                        dge_mode=dge_mode.none,
                        name=f"v_active_block_load_full_b{i_b}",
                    )
            elif cfg.strided_mm1:
                # Need to load V_active in a strided manner across the entire free dim, this requires two loads because we need
                # to load s_active rows of d_head into v_sb, which has free dim of (num_sprior_t128 * d_head).
                load1_nrows = cfg.s_active % atp.n_sprior_tile
                load2_nrows = cfg.s_active - load1_nrows

                # Load 1. Load the first (s_active % num_sprior_t128) rows of V_active (less than one row in v_sb)
                if load1_nrows > 0:
                    load1_pidx = TC.p_max - (load2_nrows // atp.n_sprior_tile) - 1

                    v_sb_offset = (
                        load1_pidx * cfg.d_head * atp.n_sprior_tile + (atp.n_sprior_tile - load1_nrows) * cfg.d_head
                    )
                    v_sb_pat = v_sb.ap(
                        pattern=[
                            [cfg.d_head * atp.n_sprior_tile, 1],
                            [cfg.d_head, load1_nrows],
                            [1, cfg.d_head],
                        ],
                        offset=v_sb_offset,
                    )

                    v_active_offset = (atp.bs_prg_id * atp.bs + i_b) * cfg.d_head * cfg.s_active
                    v_active_pat = v_active.ap(
                        pattern=[
                            [cfg.s_active * cfg.d_head, 1],
                            [cfg.d_head, load1_nrows],
                            [1, cfg.d_head],
                        ],
                        offset=v_active_offset,
                    )
                    nisa.dma_copy(
                        v_sb_pat, v_active_pat, dge_mode=dge_mode.none, name=f"v_active_strided_load_partial_b{i_b}"
                    )

                # Load 2. Load the remaining rows
                if load2_nrows > 0:
                    load2_pidx = TC.p_max - (load2_nrows // atp.n_sprior_tile)

                    v_sb_offset = load2_pidx * (cfg.d_head * atp.n_sprior_tile)
                    v_sb_pat = v_sb.ap(
                        pattern=[
                            [
                                cfg.d_head * atp.n_sprior_tile,
                                load2_nrows // atp.n_sprior_tile,
                            ],
                            [cfg.d_head, atp.n_sprior_tile],
                            [1, cfg.d_head],
                        ],
                        offset=v_sb_offset,
                    )

                    v_active_offset = (
                        atp.bs_prg_id * atp.bs + i_b
                    ) * cfg.d_head * cfg.s_active + load1_nrows * cfg.d_head
                    v_active_pat = v_active.ap(
                        pattern=[
                            [
                                atp.n_sprior_tile * cfg.d_head,
                                load2_nrows // atp.n_sprior_tile,
                            ],
                            [cfg.d_head, atp.n_sprior_tile],
                            [1, cfg.d_head],
                        ],
                        offset=v_active_offset,
                    )

                    nisa.dma_copy(
                        v_sb_pat, v_active_pat, dge_mode=dge_mode.none, name=f"v_active_strided_load_remaining_b{i_b}"
                    )
            else:
                v_active_pat = v_active.ap(
                    pattern=[
                        [cfg.d_head, cfg.s_active],
                        [1, cfg.d_head],
                    ],
                    offset=(atp.bs_prg_id * atp.bs + i_b) * cfg.s_active * cfg.d_head,
                )
                # Load to the bottom right part of last chunk of v_sb: [s_active, d_head]
                nisa.dma_copy(
                    v_sb[TC.p_max - cfg.s_active :, v_sb.shape[1] - cfg.d_head :],
                    v_active_pat,
                    dge_mode=dge_mode.none,
                    name=f"v_active_load_sequential_b{i_b}",
                )

        # Perform V^T @ exp^T, which equals to (exp @ V)^T. Recall mm1 output is transposed - KQ^T
        exp_v_psum = nl.ndarray(
            (cfg.d_head, atp.s_active_qh),
            dtype=nl.float32,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, (i_b % batch_interleave_degree_safe) * TC.psum_f_max_bytes),
        )
        for i_t in range(atp.n_sprior_tile):
            qk_io_offset = i_t * atp.s_active_bqh + i_b * atp.s_active_qh

            qk_io_pat = [
                [atp.n_sprior_tile * atp.s_active_bqh, TC.p_max],
                [1, atp.s_active_qh],
            ]
            qk_offset = qk_io_offset
            v_sb_pat = [[cfg.d_head * atp.n_sprior_tile, TC.p_max], [1, cfg.d_head]]
            v_sb_offset = i_t * cfg.d_head
            nisa.nc_matmul(
                exp_v_psum[...],
                stationary=v_sb.ap(v_sb_pat, v_sb_offset),
                moving=bufs.qk_io_type.ap(qk_io_pat, qk_offset),
            )

        # Copy mm2 output from psum -> sb while multiplying recip(sum)
        # exp_sum_recip = exp_sum_recip.reshape((p_max, bs, s_active_qh))
        exp_v_pat = bufs.exp_v.ap(
            [[atp.s_active_bqh, cfg.d_head], [1, 1], [1, atp.s_active_qh]],
            offset=i_b * atp.s_active_qh,
        )
        exp_sum_recip_pat = bufs.exp_sum_recip.ap(
            [[atp.s_active_bqh, cfg.d_head], [1, 1], [1, atp.s_active_qh]],
            offset=i_b * atp.s_active_qh,
        )
        # FIXME: SCAN ALL SLICING!!!
        nisa.tensor_tensor(exp_v_pat, exp_v_psum, exp_sum_recip_pat, op=nl.multiply)
        sbm.increment_section()
    sbm.close_scope()

    # Gather and add partial mm2 result from other NC if needed
    if atp.sprior_n_prgs > 1:
        exp_v_recv = sbm.alloc_stack(bufs.exp_v.shape, bufs.exp_v.dtype, buffer=nl.sbuf)
        nisa.sendrecv(
            src=bufs.exp_v,
            dst=exp_v_recv,
            send_to_rank=(1 - atp.sprior_prg_id),
            recv_from_rank=(1 - atp.sprior_prg_id),
            pipe_id=0,
        )

        # Only NC0 adds partial results, unless we have out_in_sb then both cores will obtain the result
        if cfg.out_in_sb or (atp.sprior_prg_id == 0):
            nisa.tensor_tensor(bufs.exp_v[...], bufs.exp_v, exp_v_recv, op=nl.add)

    # NC0 stores exp_v into HBM which has shape (bs, q_head, d_head, s_active)
    if cfg.out_in_sb:
        # exp_v and out may have different dtype, easier to just always keep this tensor copy to do the conversion
        exp_v_reshaped = bufs.exp_v.reshape((cfg.d_head, atp.s_active_bqh))

        out_offset = atp.bs_prg_id * atp.s_active_bqh
        nisa.tensor_copy(out[0 : cfg.d_head, nl.ds(out_offset, atp.s_active_bqh)], src=exp_v_reshaped)
        if atp.bs_n_prgs > 1:
            dst_bs_offset = (1 - atp.bs_prg_id) * atp.s_active_bqh
            nisa.sendrecv(
                src=out[0 : cfg.d_head, nl.ds(out_offset, atp.s_active_bqh)],
                dst=out[0 : cfg.d_head, nl.ds(dst_bs_offset, atp.s_active_bqh)],
                send_to_rank=(1 - atp.bs_prg_id),
                recv_from_rank=(1 - atp.bs_prg_id),
                pipe_id=0,
            )
    else:
        # Save exp_v (output) into DRAM, only NC0 needs to do this
        # This needs to be strided save due to different layout in SBUF and DRAM:
        #   SBUF: [d_head, s_active * n_qhead_per_kvhead]
        #   DRAM: [n_qhead_per_kvhead, d_head, s_active]
        if atp.sprior_prg_id == 0:
            # assert bs_prg_id == 0, "invalid assumption hit, update the outpat offset"
            exp_v_reshaped = bufs.exp_v.reshape((cfg.d_head, atp.bs, cfg.q_head, cfg.s_active))

            # [B, H, d, S_active]
            out_offset = atp.bs_prg_id * atp.bs * cfg.d_head * atp.s_active_qh
            out_pat = out.ap(
                pattern=[
                    [cfg.s_active, cfg.d_head],
                    [cfg.q_head * cfg.d_head * cfg.s_active, atp.bs],
                    [cfg.s_active * cfg.d_head, cfg.q_head],
                    [1, cfg.s_active],
                ],
                offset=out_offset,
            )

            nisa.dma_copy(dst=out_pat, src=exp_v_reshaped, dge_mode=dge_mode.none, name="out_store_hbm")


"""
Sharding Logic
"""


def _get_lnc_sharding(cfg: AttnTKGConfig) -> Tuple[int, int, int, int]:
    """
    Returns sharding parameters for context length (s_prior) and batch (bs) based on configuration.
    """
    n_prgs, prg_id = nl.num_programs(0), nl.program_id(0)
    kernel_assert(n_prgs == 2, "Attention currently only supports LNC2.")  # FIXME: More testing needed to support LNC1
    kernel_assert(
        n_prgs <= 2,
        f"Attention cascaded supports unsharded or LNC2 sharded; but got a spmd grid size of {n_prgs}",
    )

    sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id = (1, 0, 1, 0)
    if n_prgs > 1:
        TILE_CONSTANTS = TileConstants.get_tile_constants()
        if is_batch_sharded(cfg, TILE_CONSTANTS.p_max):
            bs_n_prgs, bs_prg_id = (n_prgs, prg_id)
        elif (
            cfg.bs * cfg.q_head * cfg.s_active <= TILE_CONSTANTS.p_max and cfg.curr_sprior >= 2 * TILE_CONSTANTS.p_max
        ):  # If s_prior is small, and batch is not divisible by lnc
            sprior_n_prgs, sprior_prg_id = (n_prgs, prg_id)

    return sprior_n_prgs, sprior_prg_id, bs_n_prgs, bs_prg_id


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
        nisa.dma_copy(x_pre_tp[...], x_inp, dge_mode=dge_mode.none, name=f"rope_x_inp_load_{name_suffix}")
        # FIXME: Add arch check
        tp_dtype = x_pre_tp.dtype
        x_tp_psum = nl.ndarray(
            (d_head, x_f), dtype=tp_dtype, buffer=nl.psum, address=None if sbm.is_auto_alloc() else (0, 0)
        )
        nisa.nc_transpose(x_tp_psum, x_pre_tp)
        nisa.tensor_copy(x[...], x_tp_psum)

    # Compute x * cos, the read to cos is repeated n_head times
    x_cos = sbm.alloc_stack(
        x_shape_expanded, dtype=nl.float32, buffer=nl.sbuf
    )  # expand dims for more convenient indexing
    # cos[i_d, s_active*i_B + i_S]
    nisa.tensor_tensor(
        dst=x_cos[...],
        data1=x.reshape(x_shape_expanded),
        data2=cos.ap(
            pattern=[
                [bs * s_active, d_head],
                [s_active, bs],
                [0, n_head],
                [1, s_active],
            ],
            offset=0,
        ),
        op=nl.multiply,
    )
    x_cos = x_cos.reshape(x.shape)

    # Compute _rotate_half(x)
    rotated_x = sbm.alloc_stack(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(rotated_x[d_head // 2 :, :], x[: d_head // 2, :])
    nisa.tensor_scalar(rotated_x[: d_head // 2, :], x[d_head // 2 :, :], op0=nl.multiply, operand0=-1.0)

    # Compute _rotate_half(x) * sin
    rotated_x = rotated_x.reshape(x_shape_expanded)
    nisa.tensor_tensor(
        dst=rotated_x[...],
        data1=rotated_x,
        data2=sin.ap(
            pattern=[
                [bs * s_active, d_head],
                [s_active, bs],
                [0, n_head],
                [1, s_active],
            ],
            offset=0,
        ),
        op=nl.multiply,
    )
    rotated_x = rotated_x.reshape(x.shape)

    # Add two intermediates
    nisa.tensor_tensor(x_embed[...], x_cos, rotated_x, op=nl.add)


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
    nisa.tensor_scalar(emb[...], pos_ids[0:d_head_half, :], op0=nl.multiply, operand0=inv_freqs)

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
    nisa.activation(sin[...], op=nl.sin, data=emb4sin)

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
    nisa.activation(cos[...], op=nl.sin, data=emb4cos)

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

    nisa.tensor_scalar(q_f32[...], x, nl.multiply, 1.0 / y, False, nl.add, -0.5, False)
    nisa.tensor_copy(q_i32[...], q_f32)

    # Compute q * y
    qy = sbm.alloc_stack((p, f), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(qy[...], q_i32, nl.multiply, y)

    # Compute x - (q * y)
    # out = x if in_place else nl.ndarray((p, f), dtype=nl.float32, buffer=nl.sbuf, name='modulo_out')
    nisa.tensor_tensor(out[...], x, qy, nl.subtract)

    return out


### Sink
def _prep_sink(
    sink_hbm,
    result,
    atp: AttnTileParams,
    cfg: AttnTKGConfig,
    TC: TileConstants,
    sbm: SbufManager,
):
    """
    Helper function that loads sink and replicate/transpose it from [1, H] to [B * H * S_active, 1].
    If B * H * S_active needs tiling, then to [p_max, ceil(B * H * S_active / p_max)]
    """
    sink_sb = sbm.alloc_stack((1, cfg.q_head), dtype=sink_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(sink_sb, sink_hbm.reshape((1, cfg.q_head)), dge_mode=dge_mode.none, name="sink_load")

    # Access pattern to 1) repeat_interleave sink for s_active times, 2) and repeat (non-interleave) for bs times.
    sink_repeated = sbm.alloc_stack((1, atp.s_active_bqh), buffer=nl.sbuf, dtype=sink_hbm.dtype)
    nisa.tensor_copy(
        dst=sink_repeated.ap(
            pattern=[
                [atp.s_active_bqh, 1],
                [cfg.q_head * cfg.s_active, atp.bs],
                [cfg.s_active, cfg.q_head],
                [1, cfg.s_active],
            ],
            offset=0,
        ),
        src=sink_sb.ap(
            pattern=[[cfg.q_head, 1], [0, atp.bs], [1, cfg.q_head], [0, cfg.s_active]],
            offset=0,
        ),
    )

    # And transpose.
    sink_tp_repeated = sbm.alloc_stack((atp.s_active_bqh_tile, atp.n_bsq_tiles), buffer=nl.sbuf, dtype=sink_hbm.dtype)

    sbm.open_scope()
    for i_bsq_tile in range(atp.n_bsq_full_tiles):
        _tile_sink_transpose(
            i_bsq_tile,
            atp.s_active_bqh_tile,
            sink_hbm,
            sink_repeated,
            sink_tp_repeated,
            atp,
            TC,
            sbm,
        )

    if atp.s_active_bqh_remainder > 0:
        _tile_sink_transpose(
            atp.n_bsq_full_tiles, atp.s_active_bqh_remainder, sink_hbm, sink_repeated, sink_tp_repeated, atp, TC, sbm
        )
    sbm.close_scope()
    nisa.tensor_copy(result, sink_tp_repeated)


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
        sink_tp_psum[...],
        sink_repeated[:, nl.ds(index * atp.s_active_bqh_tile, tile_size)],
    )
    nisa.tensor_copy(sink_tp_repeated[:tile_size, index], sink_tp_psum)


def _load_and_reshape_active_blk_table(
    active_blk_table,
    resize_factor,
    n_prgs,
    prg_id,
    batch_size,
    sbm: SbufManager,
    batch_slicing: Tuple[int, int],
):
    """
    Load active blocks table into SB.
    Put every 128 consecutive blocks on the same column, spread along the partition dimension.
    If blocks per batch < 128, reduce block_len to increase blocks per batch to 128.
    """
    TC = TileConstants.get_tile_constants()
    num_active_blks = active_blk_table.shape[1] * resize_factor
    kernel_assert(
        num_active_blks % (TC.p_max * n_prgs) == 0,
        (
            f"Block KV requires the number of active blocks per batch to a multiple of (p_max * n_prgs). "
            f"Got {num_active_blks} with {n_prgs} shards. Consider using resize_cache_block_len_for_attention_tkg_kernel to get the correct resize_factor."
        ),
    )

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
        num_128_folds_per_batch = num_active_blks // n_prgs // TC.p_max
        partition_resize = TC.p_max // resize_factor
        batch_size = batch_slicing[1]

        active_blk_table_sb = sbm.alloc_stack(
            (TC.p_max, num_128_folds_per_batch * batch_size),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
        )

        active_blk_table_sb_tv = (
            TensorView(active_blk_table_sb)
            # .reshape_dim(0, [partition_resize, resize_factor]) # Semantically correct, if two indices on partition were allowed
            .reshape_dim(1, [num_128_folds_per_batch, batch_size])
        )

        active_blk_table_tv = (
            TensorView(active_blk_table)  # [B, num_blks]
            .reshape_dim(1, [n_prgs, num_128_folds_per_batch, partition_resize])  # [B, n_prgs, folds, 128 / folds]
            .slice(0, batch_slicing[0], batch_slicing[0] + batch_size)  # [b, n_prgs, folds, 128 / folds]
            .select(1, prg_id)  # [b, folds, 128 / folds]
            .permute([2, 1, 0])  # [128 / folds, folds, b]
            .expand_dim(1)  # [128 / folds, 1, folds, b]
            .broadcast(1, resize_factor)  # [128 / folds, resize_factor, folds, b]
        )

        nisa.dma_copy(
            src=active_blk_table_tv.get_view(),
            dst=active_blk_table_sb_tv.get_view(),
            dge_mode=dge_mode.none,
            name="active_blk_table_load_resize1",
        )
    else:
        # We need to "resize" the cache blocks.
        # First load one batch per partition without repetition on free dimension.
        blks_per_prg = active_blk_table.shape[1] // n_prgs
        batch_size = batch_slicing[1]
        num_128_folds_per_batch = blks_per_prg * resize_factor // TC.p_max

        active_blk_table_sb = sbm.alloc_stack(
            (TC.p_max, batch_size * num_128_folds_per_batch),
            dtype=active_blk_table.dtype,
            buffer=nl.sbuf,
            align=4,
        )

        sbm.open_scope()
        active_blk_pre_reshape = sbm.alloc_stack(
            (batch_size, blks_per_prg), dtype=active_blk_table.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=active_blk_pre_reshape,
            src=active_blk_table[
                nl.ds(batch_slicing[0], batch_slicing[1]),
                nl.ds(prg_id * blks_per_prg, blks_per_prg),
            ],
            dge_mode=dge_mode.none,
            name="active_blk_table_load_pre_reshape",
        )

        # Now update the active blocks table with.  New active blocks table will be:
        #   old_blk_idx * resize_factor + arange(resize_factor)
        reshape_arange = sbm.alloc_stack((batch_size, resize_factor), dtype=active_blk_table.dtype, buffer=nl.sbuf)
        nisa.iota(dst=reshape_arange[...], pattern=[[1, resize_factor]], offset=0)

        active_blk_reshaped = sbm.alloc_stack(
            (batch_size, blks_per_prg, resize_factor), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.scalar_tensor_tensor(
            dst=active_blk_reshaped,
            data=active_blk_pre_reshape.ap([[blks_per_prg, batch_size], [1, blks_per_prg], [0, resize_factor]]),
            op0=nl.multiply,
            operand0=float(resize_factor),
            op1=nl.add,
            operand1=reshape_arange.ap([[resize_factor, batch_size], [0, blks_per_prg], [1, resize_factor]]),
        )

        active_blk_reshaped = active_blk_reshaped.reshape((batch_size, blks_per_prg * resize_factor))

        sbm.open_scope()
        for fold_idx in range(num_128_folds_per_batch):
            active_blk_transposed = nl.ndarray(
                (TC.p_max, batch_size),
                dtype=active_blk_reshaped.dtype,
                buffer=nl.psum,
                address=None if sbm.is_auto_alloc() else (0, (fold_idx % TC.psum_b_max) * TC.psum_f_max_bytes),
            )
            nisa.nc_transpose(
                active_blk_transposed,
                active_blk_reshaped[:, nl.ds(fold_idx * TC.p_max, TC.p_max)],
            )
            nisa.tensor_copy(
                active_blk_table_sb[:, nl.ds(fold_idx * batch_size, batch_size)],
                src=active_blk_transposed,
                dtype=active_blk_reshaped.dtype,
                engine=nisa.vector_engine,
            )
        sbm.close_scope()
        sbm.close_scope()

    return active_blk_table_sb, num_128_folds_per_batch


### Other Helpers
def _get_safe_batch_interleave_degree(space_per_leaf, dtype, atp: AttnTileParams, sbm: SbufManager):
    """
    Compute the batch interleave degree that will not overflow memory given the current memory available.
    """
    kernel_assert(atp.batch_interleave_degree > 0, "batch_interleave_degree must be greater than 0")
    space_needed_per_batch = space_per_leaf * sizeinbytes(dtype)
    space_available = sbm.get_free_space()

    result = min(atp.batch_interleave_degree, space_available // space_needed_per_batch)

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
