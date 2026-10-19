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

"""Shared configuration, tiling constants, and helpers for the token-generation attention (attention_tkg) kernel."""

import math
import os
from dataclasses import dataclass
from typing import Tuple

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert

# QK-swap (transposed-score, column-tiled) MM1 is enabled by default on compatible shapes.
# Set NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=1 to force the default (non-swap) path, e.g. for A/B testing.
_ATTN_TKG_NO_SWAP = os.environ.get("NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP", "").lower() in ("1", "true", "yes", "on")

# Flash attention: use FA when s_prior > threshold, tile size = threshold
_FA_TILE_SIZE = 8 * 1024  # 8K - serves as both threshold and tile size

# Batch tiling: SBUF memory budget for batch-dependent buffers
# Rule of thumb for SBUF memory budget: 8 * bs * q_head * s_active * fa_tile_s_prior <= _BATCH_TILE_SBUF_BUDGET
# This is based on the size for qk, qk exp and qk mask to be kept comfortably below SBUF size.
# Check attention_tkg_design_spec.md for further considerations.
_BATCH_TILE_SBUF_BUDGET = 16 * 1024 * 1024


@dataclass
class AttnTKGConfig(nl.NKIObject):
    """Configuration for token-generation attention kernel.

    This dataclass contains shape parameters and performance optimization flags
    for the attention_tkg kernel, which is optimized for small active sequence lengths.
    """

    # Tensor shapes
    bs: int = 0
    """Batch size. Number of independent sequences processed in parallel."""

    q_head: int = 0
    """Number of query heads. For MHA this equals num_heads; for GQA/MQA this may differ from KV heads."""

    s_active: int = 0
    """Active sequence length (tokens being generated this step). >1 indicates speculative decoding."""

    curr_sprior: int = 0
    """Current prior sequence length. The actual KV cache content length for this execution."""

    full_sprior: int = 0
    """Full prior sequence length. Maximum KV cache capacity (bucket size) that was allocated."""

    d_head: int = 0
    """Head dimension. Embedding size per attention head (typically 64 or 128)."""

    block_len: int = 0
    """Block length for block KV cache. Set to 0 for flat (contiguous) KV cache layout."""

    # Performance config
    tp_k_prior: bool = False
    """Whether the kernel needs to transpose k_prior during load.
    Flat KV:
        True: k_prior is [B, 1, s_prior, d] in HBM, kernel transposes to [d, s_prior] in SBUF.
        False: k_prior is [B, 1, d, s_prior] in HBM, kernel loads directly (already transposed).
    Block KV: must be True. k_prior is [num_blocks, block_len, d], kernel always transposes during block loading."""

    strided_mm1: bool = True
    """Use strided memory access pattern when loading K for first matmul (QK^T). Improves MM2 V read efficiency."""

    use_pos_id: bool = False
    """Generate attention mask in-kernel from position IDs instead of loading pre-generated mask from HBM."""

    fuse_rope: bool = False
    """Fuse RoPE (Rotary Position Embedding) computation into the kernel. Reduces memory traffic."""

    use_gpsimd_sb2sb: bool = True
    """Use GPSIMD instructions for SBUF-to-SBUF data transfers during LNC2 sharding communication."""

    qk_in_sb: bool = False
    """Query and key tensors are pre-loaded in SBUF. Required for block KV cache support."""

    k_out_in_sb: bool = False
    """Output key tensor (after RoPE) should remain in SBUF instead of being stored to HBM."""

    out_in_sb: bool = False
    """Output attention tensor should remain in SBUF instead of being stored to HBM."""

    enable_fa_s_prior_tiling: bool = True
    """Whether to enable Flash Attention (FA) with s_prior tiling.
    When enabled, tiles the attention computation across s_prior to reduce peak memory usage."""

    fp8_packed: bool = False
    """Enable packed FP8 k_prior cache layout for block KV with FP8.
    When True, k_prior has shape [num_blocks, block_len // 2, d_head, 2] with fp8 dtype,
    where the last dimension holds two consecutive sequence positions. This enables DMA transpose
    (which requires 2-byte elements) for FP8 block KV, avoiding the slower PE transpose fallback.
    Requires: block_len > 0, FP8 KV cache, block_len % 2 == 0."""

    return_cp_softmax_stats: bool = False
    """When True, return unnormalized attention output (sum of exp(QK-max)*V without dividing
    by the softmax denominator) and export local softmax stats (max, sum) via cp_softmax_stats_out.
    Used by CP for distributed softmax correction across ranks."""

    seq_packed_slot_interleave_degree: int = 0
    """Packed attention only: override the slot multi-buffering depth.

    0 keeps the default min(num_slots, TileConstants.psum_b_max). Set to 1 to disable the
    manual SBUF multi-buffering across schedule slots, so the manual allocator reuses one
    address per slot as the automatic allocator does. No effect under automatic
    allocation, where the degree is always 1. The PSUM interleave degree is derived as
    psum_b_max // this value, so lowering this raises PSUM multi-buffering."""

    batch_shard_sprior_max: int = 0
    """Override of the batch-shard s_prior threshold used by is_batch_sharded / is_s_prior_sharded.
    0 keeps the default rule (batch-shard when curr_sprior <= 2 * p_max). A positive value replaces
    that bound, so on LNC2 an even batch with curr_sprior <= batch_shard_sprior_max is batch-sharded
    regardless of bs * q_head * s_active. Intended for sliding-window layers with a small fixed prior
    (e.g. 1280 tokens) whose block count does not tile across two cores. Must be passed to every
    trace-time helper (resize_cache_block_len_for_attention_tkg_kernel, mask generation) so the
    framework and the kernel agree on the sharding and block resize."""

    partitioned_kv_load: bool = False
    """Block KV only: load K with the caller's block_len instead of the resized one.
    The stock K load stages p_max blocks per fold, so when blocks_per_batch is not a multiple of
    p_max the cache block length is reduced (gcd) and one fold becomes many small per-block DMAs.
    With partitioned_kv_load the K load issues one indirect DMA over blocks_per_load (<= p_max)
    partitions with block_len * d_head elements each, PE-transposes per position and d-tile, and
    writes k_sb with strided copies so the layout equals the stock (resized) ordering. The V side,
    mask, softmax and PV are unchanged. Requires block KV, batch sharding (sprior_n_prgs == 1) and
    fp8_packed=False; use_dma_transpose is disabled on this path.

    The flag is a literal request, not a hint: the kernel does not check whether the partitioned load
    is faster. It trades DMA descriptors for PE transposes (block_len transposes of blocks_per_load
    columns instead of resized_block_len transposes of p_max columns), so on trn2 it is measured to be
    faster only when the block resize factor exceeds the transpose inflation
    (p_max * ceil(blocks / p_max) / blocks) AND there are at least 4 batches per core; it is 3-9% slower
    when no resize is needed, 8-21% slower at resize factor 2, and up to ~25% slower at one batch per
    core. Callers should decide per (prior length, block_len, batch per core) bucket with
    partitioned_kv_load_recommended(); see attention_tkg_design_spec.md."""


### Constants
@dataclass
class TileConstants(nl.NKIObject):
    """Hardware tile constants for Trainium SBUF and PSUM dimensions.

    These constants define the hardware limits for tiling computations on Trainium.
    Use get_tile_constants() to obtain the actual hardware values at runtime.
    """

    p_max: int
    """SBUF maximum partition dimension. Number of partitions in SBUF (typically 128)."""

    psum_f_max: int
    """PSUM maximum free dimension. Maximum elements in PSUM free axis (typically 512)."""

    psum_b_max: int
    """PSUM maximum bank dimension. Number of PSUM banks available for interleaving (8)."""

    sbuf_quadrant_size: int
    """SBUF partition quadrant size. Partitions per quadrant for transpose operations (32)."""

    psum_f_max_bytes: int
    """PSUM maximum bank size in bytes. Equals psum_f_max * 4 (float32 size)."""

    @staticmethod
    def get_tile_constants():
        return TileConstants(
            p_max=nl.tile_size.pmax,
            psum_f_max=nl.tile_size.psum_fmax,
            psum_b_max=8,
            sbuf_quadrant_size=32,
            psum_f_max_bytes=nl.tile_size.psum_fmax * 4,
        )


def uses_flash_attention(cfg_flag: bool, s_prior: int) -> Tuple[bool, int]:
    """
    Returns if flash attention should be enabled, also returns the tile size.
    """

    should_enable = cfg_flag if not cfg_flag else s_prior > _FA_TILE_SIZE

    return (should_enable, _FA_TILE_SIZE)


def uses_batch_tiling(
    bs_per_nc: int,
    q_head: int,
    s_active: int,
    fa_tile_s_prior: int,
    is_auto_alloc: bool,
    dtype_size: int,
    qk_swapped: bool = False,
) -> Tuple[bool, int]:
    """
    Determine if batch tiling is needed and compute the batch tile size.

    Batch tiling is used when the full per-NC batch exceeds the SBUF memory budget.

    Per-batch SBUF cost is qk (float32=4) + qk_io_type (dtype_size) + mask (1). The QK-swap path also
    holds a bf16 pre-transpose exp buffer (qk_io_transposed) live across the nc_transpose, but it is
    intentionally not added to this estimate and encorporates it into _BATCH_TILE_SBUF_BUDGET.
    `qk_swapped` is used to round the batch tile down to a multiple of col_tiling_factor since the swap
    path needs whole column-tile groups per tile).

    Returns (use_batch_tiling, batch_tile_size) where batch_tile_size <= bs_per_nc.
    """
    # QK buffers per batch: qk (float32=4) + qk_io_type (dtype_size) + mask (1 byte)
    # Use 8 for bf16/fp8 (power-of-2 tile friendly), 10 for float32
    bytes_per_element = 10 if dtype_size == 4 else 8
    per_batch_cost = bytes_per_element * q_head * s_active * fa_tile_s_prior
    if per_batch_cost == 0:
        return (False, bs_per_nc)
    max_tile_bs = _BATCH_TILE_SBUF_BUDGET // per_batch_cost

    # If auto-alloc is enabled, allow at least 1 batch tile even if it exceeds budget.
    if is_auto_alloc:
        max_tile_bs = max(max_tile_bs, 1)

    kernel_assert(
        max_tile_bs >= 1,
        f"Cannot fit even a single batch in SBUF. "
        f"q_head={q_head}, s_active={s_active}, fa_tile_s_prior={fa_tile_s_prior} "
        f"requires {per_batch_cost} bytes per batch, exceeding budget of {_BATCH_TILE_SBUF_BUDGET}.",
    )
    max_tile_bs = min(bs_per_nc, max_tile_bs)
    # QK-swap column-tiles col_tiling_factor (= p_max // (q_head*s_active)) batches into one PSUM, so
    # a batch tile must be a whole number of those groups (bs % col_tiling_factor == 0). Round the tile
    # down to a multiple of col_tiling_factor (at least one group); bs_per_nc is already a multiple.
    if qk_swapped:
        col_tiling_factor = max(1, nl.tile_size.pmax // (q_head * s_active))
        max_tile_bs = max((max_tile_bs // col_tiling_factor) * col_tiling_factor, col_tiling_factor)
    return (max_tile_bs < bs_per_nc, max_tile_bs)


def is_fp8_e4m3(dtype) -> bool:
    """Check if dtype is FP8 E4M3 (handles both numpy dtype and compiler internal name)."""
    return dtype in [nl.float8_e4m3, nl.float8_e4m3fn] or str(dtype) == "float8e4"


def is_fp8_e5m2(dtype) -> bool:
    """Check if dtype is FP8 E5M2 (handles both numpy dtype and compiler internal name)."""
    return dtype == nl.float8_e5m2 or str(dtype) == "float8e5"


def is_batch_sharded(
    bs: int,
    q_head: int,
    s_active: int,
    curr_sprior: int,
    p_max: int,
    fuse_rope: bool = False,
    batch_shard_sprior_max: int = 0,
):
    """
    Returns true if for lnc=2, batch should be sharded given the configuration.

    Args:
      bs: Batch size.
      q_head: Number of query heads.
      s_active: Active sequence length.
      curr_sprior: Current prior sequence length.
      p_max: Number of partitions in the SBUF.
      fuse_rope: Whether RoPE is fused (incompatible with batch sharding).
      batch_shard_sprior_max: When > 0, replaces the default `curr_sprior <= 2 * p_max` bound of the
          small-s_prior disjunct (see AttnTKGConfig.batch_shard_sprior_max). 0 keeps the default rule.

    NOTE: this function is used both at trace time and also for testing infrastructure.
          Thus, it needs to take p_max as an argument.
    """
    if fuse_rope:
        return False
    LNC = 2
    sprior_max = batch_shard_sprior_max if batch_shard_sprior_max > 0 else 2 * p_max
    # Batch sharding is needed if:
    # - BQS fills (or overflows) the partition dim, so batch-sharding keeps each core's BQS within
    #   one p_max tile AND keeps the two cores symmetric (both run the same per-batch code instead
    #   of the s_prior-sharded path where only the last core loads k_active/v_active), or
    # - s_prior is too small to shard
    #   (at curr_sprior=256 we can shard on sprior but use batch sharding so we
    #    can support packed fp8 layout which requires block size >= 2 after resize)
    return (bs % LNC == 0) and (bs * q_head * s_active >= p_max or curr_sprior <= sprior_max)


def is_s_prior_sharded(
    bs: int,
    q_head: int,
    s_active: int,
    curr_sprior: int,
    p_max: int,
    fuse_rope: bool = False,
    batch_shard_sprior_max: int = 0,
):
    """
    Returns true if for lnc=2, s_prior should be sharded given the configuration.

    s_prior sharding occurs when:
    - Batch is NOT sharded (batch sharding takes priority)
    - s_prior is large enough to shard (>= 2 * p_max)

    Args:
      bs: Batch size.
      q_head: Number of query heads.
      s_active: Active sequence length.
      curr_sprior: Current prior sequence length.
      p_max: Number of partitions in the SBUF.
      fuse_rope: Whether RoPE is fused (passed through to is_batch_sharded).
      batch_shard_sprior_max: Batch-shard threshold override (passed through to is_batch_sharded).

    NOTE: this function is used both at trace time and also for testing infrastructure.
          Thus, it needs to take p_max as an argument.
    """
    # s_prior sharding requires:
    # 1. Batch is not sharded (batch sharding takes priority)
    # 2. s_prior is large enough to shard across 2 cores
    return (
        not is_batch_sharded(bs, q_head, s_active, curr_sprior, p_max, fuse_rope, batch_shard_sprior_max)
        and curr_sprior >= 2 * p_max
    )


def get_total_n_prgs(
    bs: int,
    q_head: int,
    s_active: int,
    curr_sprior: int,
    lnc: int,
    p_max: int,
    fuse_rope: bool = False,
    batch_shard_sprior_max: int = 0,
) -> int:
    """Compute the total number of programs (NCs) used, matching the kernel's sharding logic.

    Returns lnc if either batch or s_prior sharding is active, otherwise 1.

    Args:
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        curr_sprior: Current prior sequence length.
        lnc: Number of logical neuron cores (1 or 2).
        p_max: Number of partitions in the SBUF.
        fuse_rope: Whether RoPE is fused (incompatible with batch sharding).
        batch_shard_sprior_max: Batch-shard threshold override (see AttnTKGConfig.batch_shard_sprior_max).
    """
    if lnc <= 1:
        return 1
    _args = (bs, q_head, s_active, curr_sprior, p_max, fuse_rope, batch_shard_sprior_max)
    if is_batch_sharded(*_args) or is_s_prior_sharded(*_args):
        return lnc
    return 1


def is_qk_swapped(
    bs: int,
    q_head: int,
    d_head: int,
    s_active: int,
    curr_sprior: int,
    lnc: int,
    p_max: int,
    is_block_kv: bool,
    is_2byte_kv: bool,
    fp8_packed: bool,
    fuse_rope: bool,
    kv_heads: int = 1,
    batch_shard_sprior_max: int = 0,
) -> bool:
    """Whether the QK-swap (transposed-score, column-tiled) MM1 path is active for a config.

    The QK-swap path makes Q stationary and K moving, producing a transposed score tile and
    column-tiling multiple batches into one PSUM bank. It is enabled by default whenever the shape,
    layout, and KV-load requirements below are all met. Attention, gen_mask, and the tests all call
    this one function so the swap decision stays consistent across all three.

    Set ``NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=1`` to force the default (K-stationary) path.

    Args:
        bs: Full batch size before LNC sharding (per KV head, i.e. B_attn — NOT yet folded with kv_heads).
        q_head: Total number of query heads across kv_heads (q_heads_attn for GQA).
        d_head: Head dimension (embedding size per head).
        s_active: Active (query) sequence length being generated this step.
        curr_sprior: The current KV-cache length, how much context (prior) is currently live.
            This is NOT the allocated cache capacity (``full_sprior``).
        lnc: LNC value the attention kernel will run on.
        p_max: Hardware partition dimension (``nl.tile_size.pmax``, 128 on current targets).
        is_block_kv: True iff the KV cache is block-paged (block_len > 0).
        is_2byte_kv: True iff the raw KV cache element is a 2-byte type (bfloat16 / float16). False for
            1-byte (fp8) and 4-byte (float32) KV. Ignored when fp8_packed is True.
        fp8_packed: True iff fp8 KV uses the fp8 packed in bf16 layout.
        fuse_rope: Whether fused RoPE is enabled in the attention kernel.
        kv_heads: Number of KV heads (GQA) matching q_head.
        batch_shard_sprior_max: Batch-shard threshold override (see AttnTKGConfig.batch_shard_sprior_max).

    Returns:
        True iff the QK-swap MM1 path is active for this config; False for the default K-stationary path.

    Example (4-batch, 8-q_head/kv_head, d_head=128, block-KV bf16, 8192 context on LNC2):
        >>> is_qk_swapped(
        ...     bs=4,
        ...     q_head=16,
        ...     d_head=128,
        ...     s_active=1,
        ...     curr_sprior=8192,
        ...     lnc=2,
        ...     p_max=128,
        ...     is_block_kv=True,
        ...     is_2byte_kv=True,
        ...     fp8_packed=False,
        ...     fuse_rope=False,
        ...     kv_heads=2,
        ... )
    """
    if _ATTN_TKG_NO_SWAP:
        return False
    if not is_block_kv:
        return False
    if curr_sprior < 1024:  # TODO: small context sizes not evaluated
        return False
    # The swap K-load is a 2-byte indirect dma_transpose, so KV must resolve to a 2-byte SBUF tile:
    # a raw 2-byte type (bf16/fp16), or fp8 in the packed 2-per-bf16-slot layout. Unpacked fp8 (1-byte)
    # and float32 (4-byte) fall back to the non-swap path (which handles them via nc_transpose).
    if not (is_2byte_kv or fp8_packed):
        return False
    # d_head tiling not yet supported
    if d_head > p_max:
        return False

    # GQA: the attention_tkg kernel folds kv_heads into the batch (B_folded = bs * kv_heads) and runs
    # with q_head = q_per_group (= q_head // kv_heads). Match that logic here.
    bs = bs * kv_heads
    q_head_per_kv = q_head // kv_heads
    s_active_qh = s_active * q_head_per_kv

    # LNC sharding: batch shard takes priority; else s_prior shard (mirrors _get_lnc_sharding).
    _shard_args = (bs, q_head_per_kv, s_active, curr_sprior, p_max, fuse_rope, batch_shard_sprior_max)
    batch_sharded = lnc > 1 and is_batch_sharded(*_shard_args)
    sprior_sharded = lnc > 1 and not batch_sharded and is_s_prior_sharded(*_shard_args)
    bs_per_nc = bs // (lnc if batch_sharded else 1)
    s_prior_per_shard = curr_sprior // (lnc if sprior_sharded else 1)

    # Q tiling not yet supported.
    if s_active_qh > p_max:
        return False
    # Small s_active_qh (<= 4) not supported: the swap MM1 K-load would gather too many batches per
    # dma_transpose to fit the DGE cap, and cannot be sub-divided below one mm1 group.
    if s_active_qh <= 4:
        return False
    # s_active_qh must column-tile into (a whole number of, or a whole fraction of) a 32-wide tile.
    if not (s_active_qh % 32 == 0 or 32 % s_active_qh == 0):
        return False
    # s_active_qh must divide the partition dim, and there must be enough per-NC batches to fill it.
    if p_max % s_active_qh != 0 or bs_per_nc % (p_max // s_active_qh) != 0:
        return False

    # The de-interleaved swap evict reshapes each mm1 group as [.., qk_row_tile_factor, pack_factor * p_max].
    # Every FA tile's s_prior must be a whole multiple of # p_max * qk_row_tile_factor * pack_factor.
    # NOTE: the kernel gates qk_row_tile_factor==2 on sbm.is_auto_alloc() which is not accounted for in this
    # function; this function always assumes auto-alloc. This is fine as with row_tile=2 is a stricter
    # condition than row_tile=1 so this condition only ever REJECTS manually allocated configs that could
    # have swapped, making it fall back to the functional unswapped path.
    qk_row_tile_factor = 2 if d_head == 64 else 1
    pack_factor = 2 if fp8_packed else 1
    evict_unit = p_max * qk_row_tile_factor * pack_factor
    last_fa_tile = s_prior_per_shard % _FA_TILE_SIZE if s_prior_per_shard > _FA_TILE_SIZE else s_prior_per_shard
    if last_fa_tile % evict_unit != 0:
        return False

    return True


### Block KV
def resize_cache_block_len_for_attention_tkg_kernel(
    num_blocks_per_batch: int,
    block_len: int,
    lnc: int,
    p_max: int,
    bs: int,
    q_head: int,
    s_active: int,
    full_sprior: int = 0,
    enable_fa_s_prior_tiling: bool = True,
    fuse_rope: bool = False,
    batch_shard_sprior_max: int = 0,
):
    """
    Block KV in token gen attention loads p_max blocks per fold onto SBUF partitions in parallel.
    The s_prior dimension is sharded across sprior_n_prgs NCs, so the block count per shard must
    be a multiple of p_max. If not, we reduce block_len to increase num_blocks_per_batch.

    Resize block_len so that:
      1. num_blocks_per_batch (after resize) is a multiple of (sprior_n_prgs * p_max).
      2. If flash attention is active, fa_tile_size and the last FA tile remainder are both
         divisible by (reduced_block_len * p_max).


    Args:
      num_blocks_per_batch: Number of blocks in each batch. Generally the second dimension of the active blocks table.
      block_len: The size of each block.
      lnc: Number of logical neuron cores (1 or 2).
      p_max: Maximum number of partitions.
      bs: Batch size. Used to determine sharding mode.
      q_head: Number of query heads. Used to determine sharding mode.
      s_active: Active sequence length. Used to determine sharding mode.
      full_sprior: Maximum KV cache capacity (bucket size). Used for warning suggestions.
      enable_fa_s_prior_tiling: Whether flash attention tiling is enabled. Must match the value
          passed to attention_tkg / attention_block_tkg. Default True.
      fuse_rope: Whether RoPE is fused. Must match the value passed to attention_tkg.
      batch_shard_sprior_max: Batch-shard threshold override. Must match
          AttnTKGConfig.batch_shard_sprior_max so the sharding (and hence the resize) agrees with the kernel.

    NOTE: This function is used both at trace time and by testing infrastructure. Thus, it needs to take p_max as an argument.
    """
    bucket_len = num_blocks_per_batch * block_len
    sprior_n_prgs = (
        lnc
        if lnc > 1 and is_s_prior_sharded(bs, q_head, s_active, bucket_len, p_max, fuse_rope, batch_shard_sprior_max)
        else 1
    )
    min_multiple = sprior_n_prgs * p_max
    kernel_assert(
        bucket_len % min_multiple == 0,
        (
            "Cannot resize cache blocks for block KV. Number of blocks per batch must be a multiple of (sprior_n_prgs * p_max). "
            "Consider changing the bucket length (num_blocks_per_batch * block_len) to at least a multiple of (sprior_n_prgs * p_max)."
        ),
    )

    # reduced_blk_len must divide all of these (expressed as multiples of p_max):
    #   - bucket_len // (sprior_n_prgs * p_max)  [block count divisibility]
    #   - fa_tile_size // p_max                   [FA tile divisibility, if FA active]
    #   - last_fa_tile // p_max                   [last FA tile remainder, if partial]
    divisor = bucket_len // min_multiple

    s_prior_per_shard = bucket_len // sprior_n_prgs
    use_fa, fa_tile_size = uses_flash_attention(enable_fa_s_prior_tiling, s_prior_per_shard)
    if use_fa:
        kernel_assert(
            fa_tile_size % p_max == 0,
            f"FA tile size must be divisible by p_max, got fa_tile_size={fa_tile_size}, p_max={p_max}",
        )
        divisor = math.gcd(divisor, fa_tile_size // p_max)
        last_tile = s_prior_per_shard % fa_tile_size
        if last_tile > 0:
            kernel_assert(
                last_tile % p_max == 0,
                f"Last FA tile must be divisible by p_max, got last_tile={last_tile}, p_max={p_max}",
            )
            divisor = math.gcd(divisor, last_tile // p_max)

    reduced_blk_len = math.gcd(block_len, divisor)
    resize_factor = block_len // reduced_blk_len

    if resize_factor != 1:
        print(
            f"Token-gen bucket length of {num_blocks_per_batch * block_len}:",
            f"reducing block length by {resize_factor}x,",
            f"cache block length reduced from {block_len} to {reduced_blk_len}.",
            f"Number of blocks per batch increased from {num_blocks_per_batch} to {resize_factor * num_blocks_per_batch}.",
        )

    if reduced_blk_len < 8:
        no_resize_multiple = block_len * min_multiple
        min_sprior_no_resize = ((bucket_len // no_resize_multiple) + 1) * no_resize_multiple
        print(
            f"WARNING: Smaller block length (<8) results in lower DMA bandwidth utilization. Consider increasing curr_sprior to at least {min_sprior_no_resize}."
        )
        if full_sprior > 0 and min_sprior_no_resize > full_sprior:
            print(f"WARNING: This also requires increasing full_sprior to at least {min_sprior_no_resize}.")
    return reduced_blk_len, resize_factor


def partitioned_kv_load_recommended(
    bs_per_core: int,
    num_blocks_per_batch: int,
    block_len: int,
    resized_block_len: int,
    p_max: int,
    min_bs_per_core: int = 4,
) -> bool:
    """Advisory rule for when a caller should request AttnTKGConfig.partitioned_kv_load.

    The partitioned load divides the K DMA descriptor count by the block resize factor
    (block_len // resized_block_len) but multiplies the PE transposes and PSUM->SBUF copies by
    p_max * ceil(num_blocks / p_max) / num_blocks (p_max-column transposes become num_blocks-column
    ones). It pays off when the descriptor reduction exceeds the transpose inflation, and only when
    enough batches share a core for the transposes of one batch to overlap the gathers of another.
    On trn2 the second condition was measured at 4 batches per core (see
    attention_tkg_design_spec.md, "When the framework should request it"); with fewer, the
    serialized gather -> transpose -> copy chain of the partitioned load is up to ~25% slower.

    This function does not change kernel behaviour; attention_tkg takes the flag literally.

    Args:
      bs_per_core: Batches each core processes (bs // lnc under batch sharding).
      num_blocks_per_batch: Caller-sized blocks per batch (curr_sprior // block_len).
      block_len: Caller block length.
      resized_block_len: Block length after resize_cache_block_len_for_attention_tkg_kernel.
      p_max: SBUF partition count (128).
      min_bs_per_core: Batches per core below which the load is not recommended.
    """
    if bs_per_core < min_bs_per_core or num_blocks_per_batch <= 0:
        return False
    descriptor_reduction = block_len // resized_block_len
    n_chunks = -(-num_blocks_per_batch // p_max)
    transpose_inflation = p_max * n_chunks / num_blocks_per_batch
    return descriptor_reduction > transpose_inflation
