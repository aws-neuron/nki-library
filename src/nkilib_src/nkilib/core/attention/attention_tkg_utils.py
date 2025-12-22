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

import nki.language as nl

from ..utils.kernel_assert import kernel_assert


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
    bs: int = 0  # Batch size
    q_head: int = 0  # Number of query heads
    s_active: int = 0  # Active sequence length (>1 means speculative decoding)
    curr_sprior: int = 0  # Current prior sequence length (KV cache length for this execution)
    full_sprior: int = 0  # Full prior sequence length (maximum KV cache capacity)
    d_head: int = 0  # Head dimension (embedding size per head)
    block_len: int = 0  # Block length for block KV cache (0 if not using block KV)

    # Performance config
    tp_k_prior: bool = (
        False  # Specifies that k_prior is transposed (shape [B, 1, d, s_prior] instead of [B, 1, s_prior, d])
    )
    strided_mm1: bool = True  # Use strided memory access for first matmul to improve cache locality
    use_pos_id: bool = (
        False  # Generate attention mask from position IDs in-kernel instead of loading pre-generated mask
    )
    fuse_rope: bool = False  # Fuse RoPE (Rotary Position Embedding) computation into the kernel
    use_gpsimd_sb2sb: bool = True  # Use GPSIMD instructions for SBUF-to-SBUF data transfers (LNC2 sharding)
    qk_in_sb: bool = False  # Query and key tensors are already in SBUF instead of HBM
    k_out_in_sb: bool = False  # Output key tensor after RoPE should be stored in SBUF instead of HBM
    out_in_sb: bool = False  # Output tensor should be stored in SBUF instead of HBM


### Constants
@dataclass
class TileConstants(nl.NKIObject):
    p_max: int  # sbuf max partition dim
    psum_f_max: int  # psum max free dim
    psum_b_max: int  # psum max bank dim
    sbuf_quadrant_size: int  # sbuf partition quadrant size
    psum_f_max_bytes: int  # maximum number of bytes in psum bank

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
    - BQS (batch * q_head * s_active) fits within p_max partitions
    - s_prior is large enough to shard (>= 2 * p_max)

    Args:
      cfg: Attention kernel's configuration
      p_max: Number of partitions in the SBUF

    NOTE: this function is used both at trace time and also for testing infrastructure.
          Thus, it needs to take p_max as an argument.
    """
    # s_prior sharding requires:
    # 1. Batch is not sharded (batch sharding takes priority)
    # 2. BQS is small enough to fit in partitions
    # 3. s_prior is large enough to shard across 2 cores
    return (
        not is_batch_sharded(cfg, p_max)
        and cfg.bs * cfg.q_head * cfg.s_active <= p_max
        and cfg.curr_sprior >= 2 * p_max
    )


### Block KV
def resize_cache_block_len_for_attention_tkg_kernel(num_blocks_per_batch: int, block_len: int, n_prgs: int, p_max: int):
    """
    Block KV in token gen attention requires number of blocks per batch to be a multiple of (lnc * p_max).
    This allows loading p_max blocks onto SBUF partitions in parallel.
    If the block count is not divisible by (lnc * p_max), we will reduce block_len to increase num_blocks_per_batch.
    As long as the bucket_len is divisible by lnc * p_max, there is always a block_len (min. 1) that satisfies the requirement.

    Args:
      num_blocks_per_batch: Number of blocks in each batch. Generally the second dimension of the active blocks table.
      block_len: The size of each block.
      lnc: Sharding level.
      p_max: Maximum number of partitions.

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

    print(
        f"Token-gen bucket length of {num_blocks_per_batch * block_len}:",
        f"reducing block length by {resize_factor}x,",
        f"cache block length reduced from {block_len} to {reduced_blk_len}.",
        f"Number of blocks per batch increased from {num_blocks_per_batch} to {resize_factor * num_blocks_per_batch}.",
    )
    if reduced_blk_len <= 8:
        print(
            "Smaller block length (<= 8) result in lower DMA bandwidth utilization.",
            "Consider increasing bucket length to a multiple of a greater power of 2.",
        )
    return reduced_blk_len, resize_factor
