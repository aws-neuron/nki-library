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

import math
from dataclasses import dataclass
from typing import Tuple

import nki.language as nl

from ..utils.kernel_assert import kernel_assert

# Flash attention: use FA when s_prior > threshold, tile size = threshold
_FA_TILE_SIZE = 8 * 1024  # 8K - serves as both threshold and tile size

# Batch tiling: SBUF memory budget for batch-dependent buffers
# Rule of thumb for SBUF memory budget: 8 * bs * q_head * s_active * fa_tile_s_prior <= _BATCH_TILE_SBUF_BUDGET
# This is based on the size for qk, qk exp and qk mask to be kept comfortably below SBUF size.
# Check attention_tkg_design_spec.md for further considerations.
_BATCH_TILE_SBUF_BUDGET = 16 * 1024 * 1024


def uses_flash_attention(s_prior: int) -> Tuple[bool, int]:
    """
    Return True if flash attention is used for the given s_prior (per NC).
    Also returns the tile size.
    """
    return (s_prior > _FA_TILE_SIZE, _FA_TILE_SIZE)


def uses_batch_tiling(bs_per_nc: int, q_head: int, s_active: int, fa_tile_s_prior: int) -> Tuple[bool, int]:
    """
    Determine if batch tiling is needed and compute the batch tile size.

    Batch tiling is used when the full per-NC batch exceeds the SBUF memory budget.

    Returns (use_batch_tiling, batch_tile_size) where batch_tile_size <= bs_per_nc.
    """
    per_batch_cost = 8 * q_head * s_active * fa_tile_s_prior
    if per_batch_cost == 0:
        return (False, bs_per_nc)
    max_tile_bs = _BATCH_TILE_SBUF_BUDGET // per_batch_cost
    kernel_assert(
        max_tile_bs >= 1,
        f"Cannot fit even a single batch in SBUF. "
        f"q_head={q_head}, s_active={s_active}, fa_tile_s_prior={fa_tile_s_prior} "
        f"requires {per_batch_cost} bytes per batch, exceeding budget of {_BATCH_TILE_SBUF_BUDGET}.",
    )
    max_tile_bs = min(bs_per_nc, max_tile_bs)
    return (max_tile_bs < bs_per_nc, max_tile_bs)


def is_fp8_e4m3(dtype) -> bool:
    """Check if dtype is FP8 E4M3 (handles both numpy dtype and compiler internal name)."""
    return dtype == nl.float8_e4m3 or str(dtype) == "float8e4"


def is_fp8_e5m2(dtype) -> bool:
    """Check if dtype is FP8 E5M2 (handles both numpy dtype and compiler internal name)."""
    return dtype == nl.float8_e5m2 or str(dtype) == "float8e5"


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


def is_batch_sharded(cfg: AttnTKGConfig, p_max: int):
    """
    Returns true if for lnc=2, batch should be sharded given the configuration.

    Args:
      cfg: Attention kernel's configuration
      p_max: Number of parittions in the SBUF

    NOTE: this function is used both at trace time and also for testing infrastructure.
          Thus, it needs to take p_max as an argument.
    """
    LNC = 2
    # Batch sharding is needed if:
    # - BQS is large, to reduce the number of BQS tiles, or
    # - s_prior is too small to shard
    return (cfg.bs % LNC == 0) and (cfg.bs * cfg.q_head * cfg.s_active > p_max or cfg.curr_sprior < 256)


def is_s_prior_sharded(cfg: AttnTKGConfig, p_max: int):
    """
    Returns true if for lnc=2, s_prior should be sharded given the configuration.

    s_prior sharding occurs when:
    - Batch is NOT sharded (batch sharding takes priority)
    - s_prior is large enough to shard (>= 2 * p_max)

    Args:
      cfg: Attention kernel's configuration
      p_max: Number of partitions in the SBUF

    NOTE: this function is used both at trace time and also for testing infrastructure.
          Thus, it needs to take p_max as an argument.
    """
    # s_prior sharding requires:
    # 1. Batch is not sharded (batch sharding takes priority)
    # 2. s_prior is large enough to shard across 2 cores
    return not is_batch_sharded(cfg, p_max) and cfg.curr_sprior >= 2 * p_max


def get_total_n_prgs(cfg: AttnTKGConfig, lnc: int, p_max: int) -> int:
    """Compute the total number of programs (NCs) used, matching the kernel's sharding logic.

    Returns lnc if either batch or s_prior sharding is active, otherwise 1.

    Args:
        cfg: Attention kernel's configuration.
        lnc: Number of logical neuron cores (1 or 2).
        p_max: Number of partitions in the SBUF.
    """
    if lnc <= 1:
        return 1
    if is_batch_sharded(cfg, p_max) or is_s_prior_sharded(cfg, p_max):
        return lnc
    return 1


### Block KV
def resize_cache_block_len_for_attention_tkg_kernel(
    num_blocks_per_batch: int, block_len: int, n_prgs: int, p_max: int, full_sprior: int = 0
):
    """
    Block KV in token gen attention requires number of blocks per batch to be a multiple of (lnc * p_max).
    This allows loading p_max blocks onto SBUF partitions in parallel.
    If the block count is not divisible by (lnc * p_max), we will reduce block_len to increase num_blocks_per_batch.
    As long as the bucket_len is divisible by lnc * p_max, there is always a block_len (min. 1) that satisfies the requirement.

    Args:
      num_blocks_per_batch: Number of blocks in each batch. Generally the second dimension of the active blocks table.
      block_len: The size of each block.
      n_prgs: Sharding level.
      p_max: Maximum number of partitions.
      full_sprior: Maximum KV cache capacity (bucket size). Used for warning suggestions.

    NOTE: This function is used both at trace time and by testing infrastructure. Thus, it needs to take p_max as an argument.
    """
    bucket_len = num_blocks_per_batch * block_len
    min_multiple = n_prgs * p_max
    kernel_assert(
        bucket_len % min_multiple == 0,
        (
            "Cannot resize cache blocks for block KV. Number of blocks per batch must be a multiple of (lnc_size * p_max)."
            "Consider changing the bucket length (num_blocks_per_batch * block_len) to at least a multiple of (lnc_size * p_max)."
        ),
    )

    # Find the greates multiple of block_len that also divides the maximum block length.
    reduced_blk_len = math.gcd(block_len, bucket_len // min_multiple)
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
