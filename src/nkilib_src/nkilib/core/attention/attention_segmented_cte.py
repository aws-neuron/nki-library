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
Segmented attention operations with block-based KV cache support.

This module provides utilities for attention computation with segmented KV cache,
supporting iterative processing of large sequences through multiple segments.
"""

import math
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ..utils.attention_reduce import _MAX_FREE_TILES, reduce_one_batch
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.modular_allocator import ModularAllocator
from .attention_cte import (
    _D_TILE_SZ,
    _K_TILE_SZ,
    _V_D_TILE_SZ,
    _V_TILE_SZ,
    _attention_cte,
)
from .fused_segmented_attention import fused_segmented_attention_impl


def floor_nisa_kernel(src_t: nl.NkiTensor, dst_t: nl.NkiTensor, p_size: int, f_size: int, allocator: ModularAllocator):
    """
    NISA implementation for floor operation using integer casting.

    Algorithm:
        casted = (int) a
        b = (float) casted
        larger = (b > a) * (casted - 1)
        smaller = (b <= a) * casted
        floor(a) = larger + smaller

    Args:
        src_t: Source tensor to compute floor of (dtype: fp32)
        dst_t: Destination tensor for floor result (dtype: int32)
        p_size: First dimension size
        f_size: Second dimension size
        allocator: SBUF allocator for temporary tensors
    """
    orig_addr = allocator.get_current_address()

    dst_cast = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int32, align_to=4)
    dst_cast_back = allocator.alloc_sbuf_tensor((p_size, f_size), nl.float32, align_to=4)
    dst_cast_minus1 = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int32, align_to=4)

    nisa.tensor_copy(dst=dst_cast, src=src_t)
    nisa.tensor_copy(dst=dst_cast_back, src=dst_cast)
    nisa.tensor_scalar(dst=dst_cast_minus1[...], data=dst_cast[...], op0=nl.subtract, operand0=1)

    condition = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int8)
    condition_not = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int8)

    nisa.tensor_tensor(dst=condition[...], data1=dst_cast_back[...], data2=src_t[...], op=nl.greater)
    nisa.tensor_scalar(dst=condition_not[...], data=condition[...], op0=nl.logical_xor, operand0=1)

    smaller = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int32, align_to=4)
    larger = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int32, align_to=4)
    nisa.tensor_tensor(dst=smaller[...], data1=dst_cast[...], data2=condition_not[...], op=nl.multiply)
    nisa.tensor_tensor(dst=larger[...], data1=dst_cast_minus1[...], data2=condition[...], op=nl.multiply)

    nisa.tensor_tensor(dst=dst_t, data1=larger[...], data2=smaller[...], op=nl.add)

    allocator.set_current_address(address=orig_addr)


def ceil_nisa_kernel(src_t: nl.NkiTensor, dst_t: nl.NkiTensor, p_size: int, f_size: int, allocator: ModularAllocator):
    """
    NISA implementation for ceil operation using floor.

    Algorithm:
        ceil(x) = -floor(-x)

    Args:
        src_t: Source tensor to compute ceil of (dtype: fp32)
        dst_t: Destination tensor for ceil result (dtype: int32)
        p_size: First dimension size
        f_size: Second dimension size
        allocator: SBUF allocator for temporary tensors
    """
    orig_addr = allocator.get_current_address()

    # Negate input
    neg_src = allocator.alloc_sbuf_tensor((p_size, f_size), nl.float32, align_to=4)
    nisa.tensor_scalar(dst=neg_src[...], data=src_t[...], op0=nl.multiply, operand0=-1.0)

    # Compute floor(-x)
    floor_neg = allocator.alloc_sbuf_tensor((p_size, f_size), nl.int32, align_to=4)
    floor_nisa_kernel(src_t=neg_src, dst_t=floor_neg, p_size=p_size, f_size=f_size, allocator=allocator)

    # Negate result: -floor(-x)
    nisa.tensor_scalar(dst=dst_t[...], data=floor_neg[...], op0=nl.multiply, operand0=-1)

    allocator.set_current_address(address=orig_addr)


def load_kv_cache(
    k_cache,
    v_cache,
    block_tables,
    k_sbuf,
    v_sbuf,
    b_i,
    h_i,
    block_table_offset,
    num_blocks,
    allocator: ModularAllocator,
    k_pre_transposed: bool = False,
    fp8_packed: bool = False,
):
    """
    Load KV cache from block tables to SBUF for a single KV head.

    Args:
        k_cache: K cache in HBM. Shape depends on k_pre_transposed and fp8_packed:
            - k_pre_transposed=False, fp8_packed=False: (num_blocks_total, num_kv_head, block_size, head_dim)
            - k_pre_transposed=True:  (num_blocks_total * num_kv_head, head_dim, block_size)
            - fp8_packed=True: (num_blocks_total, num_kv_head, block_size // 2, head_dim, 2) fp8
        v_cache: V cache in HBM with shape (num_blocks_total, num_kv_head, block_size, head_dim).
        block_tables: Block table tensor with shape (batch_size, max_blocks_per_seq)
        k_sbuf: K SBUF tiles to load into (dtype must match k_cache.dtype)
        v_sbuf: V SBUF tiles to load into (dtype must match v_cache.dtype)
        b_i: Current sequence index in batch
        h_i: Current KV head index
        block_table_offset: SBUF tensor (1, 1) indicating the block offset for the current segment
        num_blocks: Number of blocks to load
        allocator: SBUF allocator for temporary tensor allocation
        k_pre_transposed: If True, K cache is already stored in transposed layout
            (head_dim, block_size) per block, so no transpose is needed during loading.
        fp8_packed: If True, K cache uses packed FP8 layout where 2 consecutive
            sequence positions are packed into one bf16-width element for dma_transpose.
    """
    num_kv_head = v_cache.shape[1]
    block_size = v_cache.shape[2]
    head_dim = v_cache.shape[3]
    bs, max_blocks_per_seq = block_tables.shape

    # k_sbuf/v_sbuf are 2D lists: [seq_tile][d_tile]
    num_d_tiles_k = len(k_sbuf[0])
    num_d_tiles_v = len(v_sbuf[0])
    d_tile_size_k = k_sbuf[0][0].shape[0]
    d_tile_size_v = v_sbuf[0][0].shape[1]
    K_TILE_SIZE = k_sbuf[0][0].shape[1]
    V_TILE_SIZE = v_sbuf[0][0].shape[0]

    k_needs_cast = k_cache.dtype != k_sbuf[0][0].dtype
    v_needs_cast = v_cache.dtype != v_sbuf[0][0].dtype
    kernel_assert(
        not (k_needs_cast or v_needs_cast) or k_pre_transposed or fp8_packed,
        "FP8 KV cache (dtype mismatch) requires k_pre_transposed=True",
    )

    # Store the original sbuf address
    orig_sbuf_addr = allocator.get_current_address()

    kernel_assert(
        K_TILE_SIZE >= block_size and K_TILE_SIZE % block_size == 0,
        f"K_TILE_SIZE must be >= block_size and divisible by block_size",
    )
    num_blocks_per_k_tile = K_TILE_SIZE // block_size
    num_k_tiles = num_blocks // num_blocks_per_k_tile

    # Chunk the loading into iterations of up to MAX_BLOCKS_PER_LOAD blocks each.
    # This avoids SBUF dimension constraints on block index tensors.
    MAX_BLOCKS_PER_LOAD = 128
    num_chunks = math.ceil(num_blocks / MAX_BLOCKS_PER_LOAD)

    for chunk_i in range(num_chunks):
        chunk_start = chunk_i * MAX_BLOCKS_PER_LOAD
        chunk_num_blocks = min(MAX_BLOCKS_PER_LOAD, num_blocks - chunk_start)

        # Save SBUF address so each chunk's temp allocations are freed
        chunk_sbuf_addr = allocator.get_current_address()

        # Compute the block_table_offset for this chunk
        chunk_block_table_offset = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        if chunk_start == 0:
            nisa.tensor_copy(dst=chunk_block_table_offset, src=block_table_offset)
        else:
            nisa.tensor_scalar(
                dst=chunk_block_table_offset,
                data=block_table_offset,
                op0=nl.add,
                operand0=chunk_start,
            )

        # Load block indices from block table as (1, chunk_num_blocks) for V cache / K fallback scalar_offset usage
        block_table_idx_before_tp = allocator.alloc_sbuf_tensor(shape=(1, chunk_num_blocks), dtype=nl.uint32)
        nisa.dma_copy(
            src=block_tables.ap(
                pattern=[[1, 1], [1, chunk_num_blocks]],
                offset=b_i * max_blocks_per_seq,
                scalar_offset=chunk_block_table_offset,
                indirect_dim=1,
            ),
            dst=block_table_idx_before_tp,
            oob_mode=nisa.oob_mode.skip,
        )

        chunk_num_k_tiles = chunk_num_blocks // num_blocks_per_k_tile
        chunk_k_tile_start = chunk_start // num_blocks_per_k_tile

        # Load K cache: packed FP8, pre-transposed, or original path
        if fp8_packed:
            # Head-major packed K cache: (num_blocks, num_kv_head, block_size // 2, head_dim, 2) fp8.
            # Reshape to (num_blocks * num_kv_head, block_size // 2, head_dim, 2) so the head folds
            # into the leading block dimension, then flatten + view as bf16 to get
            # (num_blocks * num_kv_head * block_size // 2, head_dim) bf16. This is structurally the
            # same per-head as the original packed layout, but with head_dim row width (instead of
            # kv_dim) and the head selected via the row index rather than a column offset.
            # Load per-block with dma_transpose (bf16 is 2-byte, so dma_transpose works),
            # then reinterpret back as fp8 to get [d_head, block_size] per block.
            block_size_half = block_size // 2

            # Flatten to (num_blocks * num_kv_head * block_size_half, head_dim * 2) fp8, then
            # reinterpret as bf16 to get (num_blocks * num_kv_head * block_size_half, head_dim) bf16.
            total_packed_rows = k_cache.shape[0] * num_kv_head * block_size_half
            k_cache_flat = k_cache.reshape((total_packed_rows, head_dim * 2))
            k_cache_bf16_view = k_cache_flat.view(nl.bfloat16)

            use_xbar_transpose = (head_dim % d_tile_size_k == 0) and (chunk_num_blocks % 16 == 0)

            if use_xbar_transpose:
                # bf16 temp for extracting non-contiguous blocks before reinterpret_cast.
                k_blk_bf16 = allocator.alloc_sbuf_tensor(
                    shape=(d_tile_size_k, block_size_half), dtype=nl.bfloat16, align_to=32
                )
                k_tp_bf16 = allocator.alloc_sbuf_tensor(
                    shape=(d_tile_size_k, 1, block_size_half, chunk_num_blocks),
                    dtype=nl.bfloat16,
                    align_to=32,
                )

                # Load block indices as (chunk_num_blocks, 1) for vector_offset
                block_table_idx_col = allocator.alloc_sbuf_tensor(shape=(chunk_num_blocks, 1), dtype=nl.uint32)
                nisa.dma_copy(
                    dst=block_table_idx_col,
                    src=block_tables.ap(
                        pattern=[[1, chunk_num_blocks], [1, 1]],
                        offset=b_i * max_blocks_per_seq,
                        scalar_offset=chunk_block_table_offset,
                        indirect_dim=1,
                    ),
                    oob_mode=nisa.oob_mode.skip,
                )
                # Head-major packed addressing: the row base for (block b, head h_i) is
                # (b * num_kv_head + h_i) * block_size_half. block_table_idx_col holds b, so
                # scale by num_kv_head * block_size_half; the h_i term is added via the static
                # offset below. Per-row stride is head_dim (not kv_dim) and there is no column
                # head offset.
                nisa.tensor_scalar(
                    dst=block_table_idx_col,
                    data=block_table_idx_col,
                    op0=nl.multiply,
                    operand0=num_kv_head * block_size_half,
                )

                free_dim_size = block_size_half * chunk_num_blocks

                for d_idx in range(num_d_tiles_k):
                    d_offset = d_idx * d_tile_size_k
                    nisa.dma_transpose(
                        src=k_cache_bf16_view.ap(
                            pattern=[
                                [head_dim * block_size_half, chunk_num_blocks],
                                [1, 1],
                                [head_dim, block_size_half],
                                [1, d_tile_size_k],
                            ],
                            offset=h_i * block_size_half * head_dim + d_offset,
                            vector_offset=block_table_idx_col,
                        ),
                        dst=k_tp_bf16,
                        axes=(3, 1, 2, 0),
                        oob_mode=nisa.oob_mode.skip,
                    )

                    for blk_idx in range(chunk_num_blocks):
                        i = blk_idx // num_blocks_per_k_tile
                        j = blk_idx % num_blocks_per_k_tile
                        nisa.tensor_copy(
                            dst=k_blk_bf16,
                            src=k_tp_bf16.ap(
                                pattern=[
                                    [free_dim_size, d_tile_size_k],
                                    [chunk_num_blocks, block_size_half],
                                ],
                                offset=blk_idx,
                            ),
                        )
                        nisa.tensor_copy(
                            dst=k_sbuf[chunk_k_tile_start + i][d_idx].ap(
                                pattern=[[K_TILE_SIZE, d_tile_size_k], [1, block_size]], offset=j * block_size
                            ),
                            src=k_blk_bf16.view(k_cache.dtype),
                        )
            else:
                # Per-block fallback: dma_copy + nc_transpose (no alignment needed).
                # vector_offset requires chunk_num_blocks divisible by 64 (SBUF partition),
                # so we use scalar_offset per block instead.
                # Head-major packed addressing: row base for (block b, head h_i) is
                # (b * num_kv_head + h_i) * block_size_half. Scale block idx by
                # num_kv_head * block_size_half; the h_i term is the static offset below.
                block_table_idx_packed = allocator.alloc_sbuf_tensor(shape=(1, chunk_num_blocks), dtype=nl.uint32)
                nisa.tensor_scalar(
                    dst=block_table_idx_packed,
                    data=block_table_idx_before_tp,
                    op0=nl.multiply,
                    operand0=num_kv_head * block_size_half,
                )

                k_blk_no_tp = allocator.alloc_sbuf_tensor(
                    shape=(block_size_half, d_tile_size_k), dtype=nl.bfloat16, align_to=32
                )
                k_blk_psum_tp = nl.ndarray(
                    (d_tile_size_k, block_size_half), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0)
                )

                for blk_idx in range(chunk_num_blocks):
                    i = blk_idx // num_blocks_per_k_tile
                    j = blk_idx % num_blocks_per_k_tile

                    for d_idx in range(num_d_tiles_k):
                        d_offset = d_idx * d_tile_size_k
                        nisa.dma_copy(
                            dst=k_blk_no_tp,
                            src=k_cache_bf16_view.ap(
                                pattern=[[head_dim, block_size_half], [1, d_tile_size_k]],
                                offset=h_i * block_size_half * head_dim + d_offset,
                                scalar_offset=block_table_idx_packed.ap(
                                    pattern=[[chunk_num_blocks, 1], [1, 1]], offset=blk_idx
                                ),
                            ),
                            dge_mode=nisa.dge_mode.hwdge,
                        )
                        nisa.nc_transpose(dst=k_blk_psum_tp, data=k_blk_no_tp)
                        nisa.tensor_copy(
                            dst=k_sbuf[chunk_k_tile_start + i][d_idx].ap(
                                pattern=[[K_TILE_SIZE, d_tile_size_k], [1, block_size]], offset=j * block_size
                            ),
                            src=k_blk_psum_tp.view(k_cache.dtype),
                        )

        elif k_pre_transposed:
            # K cache is (num_blocks_total * num_kv_head, head_dim, block_size) — already transposed.
            # Per-block dma_copy with scalar_offset.
            # For block b and head h, the index into dim-0 is b * num_kv_head + h.
            block_table_idx_tp = allocator.alloc_sbuf_tensor(shape=(1, chunk_num_blocks), dtype=nl.uint32)
            nisa.tensor_scalar(
                dst=block_table_idx_tp,
                data=block_table_idx_before_tp,
                op0=nl.multiply,
                operand0=num_kv_head,
            )
            if k_needs_cast:
                k_tp_tmp = allocator.alloc_sbuf_tensor(
                    shape=(d_tile_size_k, block_size), dtype=k_cache.dtype, align_to=32
                )
            for i in range(chunk_num_k_tiles):
                for j in range(num_blocks_per_k_tile):
                    blk_idx = i * num_blocks_per_k_tile + j

                    for d_idx in range(num_d_tiles_k):
                        d_offset = d_idx * d_tile_size_k
                        if k_needs_cast:
                            nisa.dma_copy(
                                dst=k_tp_tmp,
                                src=k_cache.ap(
                                    pattern=[[block_size, d_tile_size_k], [1, block_size]],
                                    offset=h_i * head_dim * block_size + d_offset * block_size,
                                    scalar_offset=block_table_idx_tp.ap(
                                        pattern=[[chunk_num_blocks, 1], [1, 1]], offset=blk_idx
                                    ),
                                ),
                                dge_mode=nisa.dge_mode.hwdge,
                            )
                            nisa.tensor_copy(
                                dst=k_sbuf[chunk_k_tile_start + i][d_idx].ap(
                                    pattern=[[K_TILE_SIZE, d_tile_size_k], [1, block_size]], offset=j * block_size
                                ),
                                src=k_tp_tmp,
                            )
                        else:
                            nisa.dma_copy(
                                dst=k_sbuf[chunk_k_tile_start + i][d_idx].ap(
                                    pattern=[[K_TILE_SIZE, d_tile_size_k], [1, block_size]], offset=j * block_size
                                ),
                                src=k_cache.ap(
                                    pattern=[[block_size, d_tile_size_k], [1, block_size]],
                                    offset=h_i * head_dim * block_size + d_offset * block_size,
                                    scalar_offset=block_table_idx_tp.ap(
                                        pattern=[[chunk_num_blocks, 1], [1, 1]], offset=blk_idx
                                    ),
                                ),
                                dge_mode=nisa.dge_mode.hwdge,
                            )
        else:
            # Load K cache with dma_transpose when possible (original non-transposed layout).
            # HW DGE indirect dma_transpose requires src.shape[-1] % 128 == 0, so
            # head_dim must be a multiple of 128 (observed 87.9% rel diff with
            # head_dim=64). Fall back to the per-block dma_copy + nc_transpose
            # path otherwise.
            use_dma_transpose = (head_dim % d_tile_size_k == 0) and (chunk_num_blocks % 16 == 0)

            if use_dma_transpose:
                # Load block indices directly as (chunk_num_blocks, 1)
                block_table_idx = allocator.alloc_sbuf_tensor(shape=(chunk_num_blocks, 1), dtype=nl.uint32)
                nisa.dma_copy(
                    src=block_tables.ap(
                        pattern=[[1, chunk_num_blocks], [1, 1]],
                        offset=b_i * max_blocks_per_seq,
                        scalar_offset=chunk_block_table_offset,
                        indirect_dim=1,
                    ),
                    dst=block_table_idx,
                    oob_mode=nisa.oob_mode.skip,
                )

                # dma_transpose per d-tile chunk (partition dim must be <= 128)
                k_sbuf_tmp = allocator.alloc_sbuf_tensor(
                    shape=(d_tile_size_k, 1, block_size, chunk_num_blocks), dtype=k_cache.dtype, align_to=32
                )
                for d_idx in range(num_d_tiles_k):
                    d_offset = d_idx * d_tile_size_k
                    nisa.dma_transpose(
                        src=k_cache.ap(
                            pattern=[
                                [num_kv_head * block_size * head_dim, chunk_num_blocks],
                                [1, 1],
                                [head_dim, block_size],
                                [1, d_tile_size_k],
                            ],
                            offset=h_i * block_size * head_dim + d_offset,
                            vector_offset=block_table_idx,
                        ),
                        dst=k_sbuf_tmp,
                        axes=(3, 1, 2, 0),
                        oob_mode=nisa.oob_mode.skip,
                    )

                    # Rearrange from interleaved to contiguous layout in k_sbuf tiles.
                    # Iterate per-block (flat) so chunk_num_blocks < num_blocks_per_k_tile
                    # (partial tile) still fills the first chunk_num_blocks slots of the
                    # first tile rather than silently skipping (the old nested form
                    # `for i in range(chunk_num_k_tiles)` gave 0 iterations when
                    # chunk_num_k_tiles = chunk_num_blocks // num_blocks_per_k_tile = 0).
                    # Any unwritten K-tile tail is handled by the MM1 num_f bound and
                    # the upstream memset that zeroes k_cache_sbuf.
                    for blk_idx in range(chunk_num_blocks):
                        i = blk_idx // num_blocks_per_k_tile
                        j = blk_idx % num_blocks_per_k_tile
                        nisa.tensor_copy(
                            src=k_sbuf_tmp.ap(
                                pattern=[
                                    [block_size * chunk_num_blocks, d_tile_size_k],
                                    [chunk_num_blocks, block_size],
                                ],
                                offset=blk_idx,
                            ),
                            dst=k_sbuf[chunk_k_tile_start + i][d_idx].ap(
                                pattern=[[K_TILE_SIZE, d_tile_size_k], [1, block_size]],
                                offset=j * block_size,
                            ),
                        )
            else:
                if head_dim <= 128:
                    print(
                        f"WARNING: chunk_num_blocks={chunk_num_blocks} is not a multiple of 16. "
                        f"Falling back to per-block dma_copy + nc_transpose for K cache loading."
                    )

                # Fallback: Load without transpose per block, then transpose each block.
                # Iterate per-block (flat) so partial K tiles (chunk_num_blocks <
                # num_blocks_per_k_tile) still get the first chunk_num_blocks slots
                # written. The previous nested form `for i in range(chunk_num_k_tiles)`
                # silently dropped partial-tile cases since chunk_num_k_tiles =
                # chunk_num_blocks // num_blocks_per_k_tile = 0 when chunk_num_blocks
                # < num_blocks_per_k_tile. The unwritten K-tile tail is handled by
                # the MM1 `num_f = min(seqlen_k, _K_TILE_SZ)` bound and the
                # up-front memset of the K cache (leaves padding zeroed).
                # For d>128, loads per d-tile chunk and writes to 2D k_sbuf.
                k_sbuf_no_tp = allocator.alloc_sbuf_tensor(
                    shape=(block_size, d_tile_size_k),
                    dtype=k_cache.dtype,
                    align_to=32,
                )
                k_psum_transposed = nl.ndarray(
                    (d_tile_size_k, block_size), dtype=k_cache.dtype, buffer=nl.psum, address=(0, 0)
                )

                for blk_idx in range(chunk_num_blocks):
                    i = blk_idx // num_blocks_per_k_tile
                    j = blk_idx % num_blocks_per_k_tile

                    for d_idx in range(num_d_tiles_k):
                        d_offset = d_idx * d_tile_size_k
                        nisa.dma_copy(
                            dst=k_sbuf_no_tp,
                            src=k_cache.ap(
                                pattern=[[head_dim, block_size], [1, d_tile_size_k]],
                                offset=h_i * block_size * head_dim + d_offset,
                                scalar_offset=block_table_idx_before_tp.ap(
                                    pattern=[[chunk_num_blocks, 1], [1, 1]], offset=blk_idx
                                ),
                            ),
                            dge_mode=nisa.dge_mode.hwdge,
                        )
                        nisa.nc_transpose(dst=k_psum_transposed, data=k_sbuf_no_tp)
                        nisa.tensor_copy(
                            dst=k_sbuf[chunk_k_tile_start + i][d_idx].ap(
                                pattern=[[K_TILE_SIZE, d_tile_size_k], [1, block_size]], offset=j * block_size
                            ),
                            src=k_psum_transposed,
                        )

        # Load V cache without transpose
        if v_needs_cast:
            v_tmp = allocator.alloc_sbuf_tensor(shape=(V_TILE_SIZE, d_tile_size_v), dtype=v_cache.dtype, align_to=32)
        if block_size >= V_TILE_SIZE:
            # Original path: each block has one or more V tiles
            kernel_assert(
                block_size % V_TILE_SIZE == 0,
                f"block_size must be divisible by V_TILE_SIZE when block_size >= V_TILE_SIZE",
            )
            num_v_tiles_per_block = block_size // V_TILE_SIZE

            for i in range(chunk_num_blocks):
                for j in range(num_v_tiles_per_block):
                    for d_idx in range(num_d_tiles_v):
                        d_offset = d_idx * d_tile_size_v
                        if v_needs_cast:
                            nisa.dma_copy(
                                dst=v_tmp,
                                src=v_cache.ap(
                                    pattern=[[d_tile_size_v, V_TILE_SIZE], [1, head_dim]],
                                    offset=h_i * block_size * head_dim + j * V_TILE_SIZE * head_dim + d_offset,
                                    scalar_offset=block_table_idx_before_tp.ap(
                                        pattern=[[chunk_num_blocks, 1], [1, 1]], offset=i
                                    ),
                                ),
                                dge_mode=nisa.dge_mode.hwdge,
                            )
                            nisa.tensor_copy(
                                dst=v_sbuf[(chunk_start + i) * num_v_tiles_per_block + j][d_idx].ap(
                                    pattern=[[d_tile_size_v, V_TILE_SIZE], [1, d_tile_size_v]], offset=0
                                ),
                                src=v_tmp,
                            )
                        else:
                            nisa.dma_copy(
                                dst=v_sbuf[(chunk_start + i) * num_v_tiles_per_block + j][d_idx].ap(
                                    pattern=[[d_tile_size_v, V_TILE_SIZE], [1, d_tile_size_v]], offset=0
                                ),
                                src=v_cache.ap(
                                    pattern=[[d_tile_size_v, V_TILE_SIZE], [1, head_dim]],
                                    offset=h_i * block_size * head_dim + j * V_TILE_SIZE * head_dim + d_offset,
                                    scalar_offset=block_table_idx_before_tp.ap(
                                        pattern=[[chunk_num_blocks, 1], [1, 1]], offset=i
                                    ),
                                ),
                                dge_mode=nisa.dge_mode.hwdge,
                            )
        else:
            # Small block path: each V tile spans multiple blocks
            kernel_assert(
                V_TILE_SIZE % block_size == 0,
                f"V_TILE_SIZE must be divisible by block_size when block_size < V_TILE_SIZE",
            )
            num_blocks_per_v_tile = V_TILE_SIZE // block_size
            chunk_num_v_tiles = chunk_num_blocks // num_blocks_per_v_tile
            chunk_v_tile_start = chunk_start // num_blocks_per_v_tile

            for v_tile_idx in range(chunk_num_v_tiles):
                for blk_in_tile in range(num_blocks_per_v_tile):
                    block_idx = v_tile_idx * num_blocks_per_v_tile + blk_in_tile
                    for d_idx in range(num_d_tiles_v):
                        d_offset = d_idx * d_tile_size_v
                        if v_needs_cast:
                            nisa.dma_copy(
                                dst=v_tmp[:block_size, :d_tile_size_v],
                                src=v_cache.ap(
                                    pattern=[[d_tile_size_v, block_size], [1, head_dim]],
                                    offset=h_i * block_size * head_dim + d_offset,
                                    scalar_offset=block_table_idx_before_tp.ap(
                                        pattern=[[chunk_num_blocks, 1], [1, 1]], offset=block_idx
                                    ),
                                ),
                                dge_mode=nisa.dge_mode.hwdge,
                            )
                            nisa.tensor_copy(
                                dst=v_sbuf[chunk_v_tile_start + v_tile_idx][d_idx].ap(
                                    pattern=[[d_tile_size_v, block_size], [1, d_tile_size_v]],
                                    offset=blk_in_tile * block_size * d_tile_size_v,
                                ),
                                src=v_tmp[:block_size, :d_tile_size_v],
                            )
                        else:
                            nisa.dma_copy(
                                dst=v_sbuf[chunk_v_tile_start + v_tile_idx][d_idx].ap(
                                    pattern=[[d_tile_size_v, block_size], [1, d_tile_size_v]],
                                    offset=blk_in_tile * block_size * d_tile_size_v,
                                ),
                                src=v_cache.ap(
                                    pattern=[[d_tile_size_v, block_size], [1, head_dim]],
                                    offset=h_i * block_size * head_dim + d_offset,
                                    scalar_offset=block_table_idx_before_tp.ap(
                                        pattern=[[chunk_num_blocks, 1], [1, 1]], offset=block_idx
                                    ),
                                ),
                                dge_mode=nisa.dge_mode.hwdge,
                            )

        # Free this chunk's temp allocations
        allocator.set_current_address(address=chunk_sbuf_addr)

    # Restore SBUF address to maintain callee-safe behavior
    # This allows load_kv_cache to be called multiple times without address conflicts
    allocator.set_current_address(address=orig_sbuf_addr)


def _attention_segmented_cte_swa_impl(
    q: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    v_cache: nl.NkiTensor,
    block_tables: nl.NkiTensor,
    prior_tokens: nl.NkiTensor,
    block_size: int,
    prior_seg_size: int,
    scale: float,
    tp_q: bool,
    tp_out: bool,
    sliding_window: int,
    sink: Optional[nl.NkiTensor],
    num_q_heads: int = 1,
    k_pre_transposed: bool = False,
    fp8_packed: bool = False,
    k_scale: Optional[nl.NkiTensor] = None,
    v_scale: Optional[nl.NkiTensor] = None,
):
    """
    Simplified sliding window attention implementation with single-iteration processing.

    With sliding window attention, we only need to attend to at most (sliding_window - 1)
    prior tokens, so everything can be done in a single attention_cte call per batch.

    Strategy:
    1. Load active KV from offset (prior_tokens // block_size)
    2. Always load window-sized prior KV (at most ceil((sw-1)/block_size) blocks)
    3. Call attention_cte with prefix caching; prior_used_len dynamically masks
       (0 when no prior tokens, clamped otherwise)

    Handles LNC2 sharding similar to the normal multi-segment flow.

    Args:
        q: Query tensor
        k_cache: K cache in HBM
        v_cache: V cache in HBM
        block_tables: Block table tensor
        prior_tokens: Prior tokens tensor in HBM
        block_size: Block size
        prior_seg_size: Segment size
        scale: Attention scale factor
        tp_q: Query transpose flag
        tp_out: Output transpose flag
        sliding_window: Sliding window size
        sink: Optional sink tensor

    Returns:
        result: Attention output tensor
    """
    # Extract dimensions
    if tp_q:
        bs_q, seqlen_q, head_dim = q.shape
    else:
        bs_q, head_dim, seqlen_q = q.shape

    kernel_assert(seqlen_q % 128 == 0, f"Query seqlen {seqlen_q} must be a multiple of 128")

    # D-tile parameters for 2D K/V SBUF layout
    num_d_tiles = math.ceil(head_dim / _D_TILE_SZ)
    d_tile_size_par_dim = min(head_dim, _D_TILE_SZ)
    num_d_tiles_free_dim = math.ceil(head_dim / (_V_D_TILE_SZ))
    d_tile_size_free_dim = min(head_dim, _V_D_TILE_SZ)

    # Derive num_kv_heads from v_cache shape for GQA mapping (v_cache's
    # layout is independent of k_pre_transposed).
    num_kv_heads = v_cache.shape[1]

    # Get sharding info for multi-core parallelization
    grid_ndim, num_shard, shard_id = get_verified_program_sharding_info("attention_segmented_cte", max_sharding=2)

    # Primary sharding: divide bs_q evenly across shards
    num_bs_per_shard = bs_q // num_shard
    bs_offset = shard_id * num_bs_per_shard

    # Secondary sharding: handle remainder if bs_q is odd
    has_remainder = (bs_q % num_shard) != 0
    last_batch = bs_q - 1

    # Initialize allocator
    allocator = ModularAllocator(initial_address=0)

    # Load KV dequantization scales into SBUF if provided (FP8 KV cache support)
    if k_scale is not None:
        k_scale_sb = allocator.alloc_sbuf_tensor(shape=(nl.tile_size.pmax, 1), dtype=nl.float32)
        nisa.dma_copy(dst=k_scale_sb, src=k_scale)
    else:
        k_scale_sb = None
    if v_scale is not None:
        v_scale_sb = allocator.alloc_sbuf_tensor(shape=(nl.tile_size.pmax, 1), dtype=nl.float32)
        nisa.dma_copy(dst=v_scale_sb, src=v_scale)
    else:
        v_scale_sb = None

    # Load prior_tokens to SBUF
    prior_tokens_sbuf = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.dma_copy(dst=prior_tokens_sbuf, src=prior_tokens)

    # Calculate active block offset = prior_tokens // block_size
    block_size_shift = int(math.log2(block_size))
    active_block_offset_swa = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(
        dst=active_block_offset_swa,
        data=prior_tokens_sbuf,
        op0=nl.right_shift,
        operand0=block_size_shift,
    )

    # Prior loading: load ceil((sw-1)/block_size) blocks right before active
    # num_prior_blocks_to_load is compile-time since sliding_window is compile-time
    effective_prior_size = sliding_window - 1  # Max prior tokens
    # Must load at least _K_TILE_SZ/block_size blocks so load_kv_cache can fill K tiles
    min_blocks_per_k_tile = _K_TILE_SZ // block_size
    num_prior_blocks_to_load = max(math.ceil(effective_prior_size / block_size), min_blocks_per_k_tile)

    # Calculate effective_prior_len = min(num_prior_blocks_to_load * block_size, prior_tokens)
    # We use the loaded block count (not sw-1) because attention_cte's SWA mask
    # will handle masking the extra tokens beyond the window.
    # When prior_tokens=0, effective_prior_len=0 and attention_cte masks out all prior.
    effective_prior_len_sbuf = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(
        dst=effective_prior_len_sbuf,
        data=prior_tokens_sbuf,
        op0=nl.minimum,
        operand0=num_prior_blocks_to_load * block_size,
    )

    # Re-buffer sink into private_hbm so _attention_cte's DMA load of
    # sink[batch_id, 0] has a buffer tier it can consume. The dynamic-range
    # no-op below is kept for scheduling/allocator parity with the prior>0
    # and prior=0 paths; removing it regressed prior>0 cases empirically.
    # _attention_cte correctly includes the sink in the section-0 softmax
    # even when prior_used_len=0, so no value-masking is needed here.
    if sink is not None:
        sink_masked = nl.ndarray(shape=sink.shape, dtype=sink.dtype, buffer=nl.private_hbm)
        nisa.dma_copy(dst=sink_masked, src=sink)
        no_prior_sbuf = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.tensor_scalar(dst=no_prior_sbuf, data=effective_prior_len_sbuf, op0=nl.less_equal, operand0=0)
        no_prior_reg = nisa.register_alloc()
        nisa.register_load(dst=no_prior_reg, src=no_prior_sbuf)
        sink_reload_sbuf = allocator.alloc_sbuf_tensor(shape=sink.shape, dtype=sink.dtype, align_to=4)
        nisa.dma_copy(dst=sink_reload_sbuf, src=sink)

        def _mask_sink_prior(_):
            nisa.dma_copy(dst=sink_masked, src=sink_reload_sbuf)

        nl.fori_loop(0, no_prior_reg, _mask_sink_prior)
        sink = sink_masked

    # prior_block_offset = max(0, active_block_offset - num_prior_blocks_to_load) (dynamic)
    # Clamp to 0 to avoid negative offset when prior_tokens < sliding_window
    prior_block_offset_swa = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32, align_to=4)
    nisa.tensor_scalar(
        dst=prior_block_offset_swa,
        data=active_block_offset_swa,
        op0=nl.subtract,
        operand0=num_prior_blocks_to_load,
        op1=nl.maximum,
        operand1=0,
    )
    # Allocate K/V sbuf for active segment (sized by seqlen_q, not prior_seg_size)
    num_active_blocks_swa = seqlen_q // block_size
    num_k_tiles_active_swa = math.ceil(seqlen_q / _K_TILE_SZ)
    num_v_tiles_active_swa = num_k_tiles_active_swa * (_K_TILE_SZ // _V_TILE_SZ)
    num_grps = math.ceil(seqlen_q / 128)

    k_sbuf_dtype = k_cache.dtype
    v_sbuf_dtype = v_cache.dtype
    k_cache_sbuf = allocator.alloc_sbuf_tensor(
        shape=(d_tile_size_par_dim, _K_TILE_SZ),
        dtype=k_sbuf_dtype,
        block_dim=[num_k_tiles_active_swa, num_d_tiles],
        num_free_tiles=[num_k_tiles_active_swa, num_d_tiles],
        align_to=32,
    )
    v_cache_sbuf = allocator.alloc_sbuf_tensor(
        shape=(_V_TILE_SZ, d_tile_size_free_dim),
        dtype=v_sbuf_dtype,
        block_dim=[num_v_tiles_active_swa, num_d_tiles_free_dim],
        num_free_tiles=[num_v_tiles_active_swa, num_d_tiles_free_dim],
    )

    # Allocate HBM buffers for unnormalized output and softmax stats (single batch, reused)
    # Intermediate uses tp_out=False so per-Q-position normalization (tensor_scalar with
    # 128-element partition vector) works correctly. When tp_out=True, nc_transpose at
    # final write. This constraint comes from tensor_scalar requiring the correction
    # vector to match the tile's partition dimension (128 Q positions, not head_dim).
    softmax_shape_swa = (1, 128, num_grps)
    out_o_hbm_swa = nl.ndarray(shape=(1, seqlen_q, head_dim), dtype=q.dtype, buffer=nl.private_hbm)
    out_neg_max_hbm_swa = nl.ndarray(shape=softmax_shape_swa, dtype=nl.float32, buffer=nl.private_hbm)
    out_sum_hbm_swa = nl.ndarray(shape=softmax_shape_swa, dtype=nl.float32, buffer=nl.private_hbm)

    # Allocate result
    if tp_out:
        result = nl.ndarray(shape=(bs_q, head_dim, seqlen_q), dtype=q.dtype, buffer=nl.shared_hbm)
    else:
        result = nl.ndarray(shape=(bs_q, seqlen_q, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

    # Workaround for NCC_IBIR251: Allocate Q buffer for single batch
    # Makes Q "internal" so access patterns work in dynamic loops
    if tp_q:
        q_internal = nl.ndarray(shape=(1, seqlen_q, head_dim), dtype=q.dtype, buffer=nl.private_hbm)
    else:
        q_internal = nl.ndarray(shape=(1, head_dim, seqlen_q), dtype=q.dtype, buffer=nl.private_hbm)

    # Prior KV tile counts (compile-time, used by both primary and remainder paths)
    # attention_cte derives seqlen_k_prior = len(k_prior_sbuf) * _K_TILE_SZ
    # V tiles must be consistent: num_v_tiles = num_k_tiles * (_K_TILE_SZ // _V_TILE_SZ)
    num_prior_k_tiles = math.ceil(num_prior_blocks_to_load * block_size / _K_TILE_SZ)
    num_prior_v_tiles = num_prior_k_tiles * (_K_TILE_SZ // _V_TILE_SZ)

    # Allocate prior KV SBUF once outside the loop (reused across iterations)
    k_prior_sbuf_swa = allocator.alloc_sbuf_tensor(
        shape=(d_tile_size_par_dim, _K_TILE_SZ),
        dtype=nl.bfloat16,
        block_dim=[num_prior_k_tiles, num_d_tiles],
        num_free_tiles=[num_prior_k_tiles, num_d_tiles],
        align_to=32,
    )
    v_prior_sbuf_swa = allocator.alloc_sbuf_tensor(
        shape=(_V_TILE_SZ, d_tile_size_free_dim),
        dtype=nl.bfloat16,
        block_dim=[num_prior_v_tiles, num_d_tiles_free_dim],
        num_free_tiles=[num_prior_v_tiles, num_d_tiles_free_dim],
    )

    # Process primary batches (one at a time)
    for b in range(num_bs_per_shard):
        batch_id = b + bs_offset  # Global bs_q index

        # Derive batch index and KV head index from batch_id for GQA
        b_i = batch_id // num_q_heads
        h_i = (batch_id % num_q_heads) * num_kv_heads // num_q_heads

        # Copy this batch's query data (layout matches tp_q)
        if tp_q:
            nisa.dma_copy(
                dst=q_internal[0, :, :],
                src=q.ap(
                    pattern=[[head_dim, seqlen_q], [1, head_dim]],
                    offset=batch_id * seqlen_q * head_dim,
                ),
            )
        else:
            nisa.dma_copy(
                dst=q_internal[0, :, :],
                src=q.ap(
                    pattern=[[seqlen_q, head_dim], [1, seqlen_q]],
                    offset=batch_id * head_dim * seqlen_q,
                ),
            )

        # Load active KV cache
        load_kv_cache(
            k_cache,
            v_cache,
            block_tables,
            k_cache_sbuf,
            v_cache_sbuf,
            b_i,
            h_i,
            active_block_offset_swa,
            num_active_blocks_swa,
            allocator,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
        )

        # Load at most window-sized prior KV; prior_used_len dynamically masks
        # (0 when no prior tokens, clamped to actual prior otherwise).
        load_kv_cache(
            k_cache,
            v_cache,
            block_tables,
            k_prior_sbuf_swa,
            v_prior_sbuf_swa,
            b_i,
            h_i,
            prior_block_offset_swa,
            num_prior_blocks_to_load,
            allocator,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
        )

        init_sbuf_addr = allocator.get_current_address()

        _attention_cte(
            q_internal,
            None,
            None,
            scale=scale,
            causal_mask=True,
            tp_q=tp_q,
            tp_k=False,
            tp_out=False,
            cache_softmax=True,
            skip_output_normalization=True,
            sliding_window=sliding_window,
            sink=sink[batch_id : batch_id + 1] if sink is not None else None,
            k_cache_sbuf=k_cache_sbuf,
            v_cache_sbuf=v_cache_sbuf,
            k_prior_sbuf=k_prior_sbuf_swa,
            v_prior_sbuf=v_prior_sbuf_swa,
            prior_used_len=effective_prior_len_sbuf,
            out_o_hbm=out_o_hbm_swa,
            out_neg_max_hbm=out_neg_max_hbm_swa,
            out_sum_hbm=out_sum_hbm_swa,
            init_sbuf_addr=init_sbuf_addr,
            k_scale_sb=k_scale_sb,
        )
        allocator.set_current_address(init_sbuf_addr)

        # Normalize (divide by S) and write to result
        sb_p = nl.tile_size.pmax
        sm_pat = [[num_grps, sb_p], [1, num_grps]]
        o_tile_pat = [[head_dim, sb_p], [1, head_dim]]

        norm_addr = allocator.get_current_address()
        sum_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
        nisa.dma_copy(dst=sum_sb, src=out_sum_hbm_swa.ap(pattern=sm_pat, offset=0))
        sum_recip_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
        nisa.reciprocal(sum_recip_sb, sum_sb)

        num_free = min(num_grps, _MAX_FREE_TILES)
        o_sb = allocator.alloc_sbuf_tensor(
            shape=(sb_p, d_tile_size_par_dim),
            dtype=nl.bfloat16,
            block_dim=[num_grps, num_d_tiles],
            num_free_tiles=[num_free, num_d_tiles],
        )
        if tp_out:
            o_tp_psum = nl.ndarray((d_tile_size_par_dim, sb_p), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0))
            o_tp_sb = allocator.alloc_sbuf_tensor(shape=(d_tile_size_par_dim, sb_p), dtype=nl.bfloat16)
        for grp_i in range(num_grps):
            for d_idx in range(num_d_tiles):
                d_offset = d_idx * d_tile_size_par_dim
                nisa.dma_copy(
                    dst=o_sb[grp_i][d_idx],
                    src=out_o_hbm_swa.ap(
                        pattern=[[head_dim, sb_p], [1, d_tile_size_par_dim]],
                        offset=grp_i * sb_p * head_dim + d_offset,
                    ),
                )
                # Delayed V dequant: fold v_scale multiply into normalization tensor_scalar
                if v_scale_sb is not None:
                    nisa.tensor_scalar(
                        dst=o_sb[grp_i][d_idx],
                        data=o_sb[grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=sum_recip_sb[:, grp_i],
                        op1=nl.multiply,
                        operand1=v_scale_sb,
                    )
                else:
                    nisa.tensor_scalar(
                        dst=o_sb[grp_i][d_idx],
                        data=o_sb[grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=sum_recip_sb[:, grp_i],
                    )
                if tp_out:
                    nisa.nc_transpose(dst=o_tp_psum, data=o_sb[grp_i][d_idx])
                    nisa.tensor_copy(dst=o_tp_sb, src=o_tp_psum)
                    nisa.dma_copy(
                        dst=result.ap(
                            pattern=[[seqlen_q, d_tile_size_par_dim], [1, sb_p]],
                            offset=batch_id * head_dim * seqlen_q + d_offset * seqlen_q + grp_i * sb_p,
                        ),
                        src=o_tp_sb,
                    )
                else:
                    nisa.dma_copy(
                        dst=result.ap(
                            pattern=[[head_dim, sb_p], [1, d_tile_size_par_dim]],
                            offset=batch_id * num_grps * sb_p * head_dim + grp_i * sb_p * head_dim + d_offset,
                        ),
                        src=o_sb[grp_i][d_idx],
                    )
        allocator.set_current_address(norm_addr)

    # Handle remainder batch (if bs_q is odd) with asymmetric sequence sharding.
    #
    # Core 0 handles Q[0 : ceil(num_grps/2)*128], Core 1 handles Q[ceil(num_grps/2)*128 :
    # num_grps*128]. For num_grps == 1 the split is degenerate — Core 0 does the full
    # remainder and Core 1 short-circuits.
    core0_grp_length = (num_grps + 1) // 2
    core1_grp_length = num_grps // 2
    if shard_id == 0:
        grp_length = core0_grp_length
        grp_start = 0
    else:
        grp_length = core1_grp_length
        grp_start = core0_grp_length

    run_remainder = has_remainder and grp_length > 0

    if run_remainder:
        # Per-shard active-segment sizing (Core 0's sizing also used by Core 1's
        # active-offset math to step past Core 0's active region).
        effective_q_tokens = grp_length * 128
        num_blocks_per_effective_seg = effective_q_tokens // block_size
        num_k_tiles_per_effective_seg = math.ceil(effective_q_tokens / _K_TILE_SZ)
        num_v_tiles_per_effective_seg = math.ceil(effective_q_tokens / _V_TILE_SZ)
        num_grps_effective = math.ceil(effective_q_tokens / 128)

        core0_q_tokens = core0_grp_length * 128
        core0_num_blocks_per_effective_seg = core0_q_tokens // block_size

        # Calculate sequence start position for this core's Q slice
        seq_start = grp_start * 128

        # Copy only this core's portion of Q to q_internal_remainder
        if tp_q:
            q_internal_remainder = nl.ndarray(
                shape=(1, effective_q_tokens, head_dim), dtype=q.dtype, buffer=nl.private_hbm
            )
            nisa.dma_copy(
                dst=q_internal_remainder[0, :, :],
                src=q.ap(
                    pattern=[[head_dim, effective_q_tokens], [1, head_dim]],
                    offset=last_batch * seqlen_q * head_dim + seq_start * head_dim,
                ),
            )
        else:
            q_internal_remainder = nl.ndarray(
                shape=(1, head_dim, effective_q_tokens), dtype=q.dtype, buffer=nl.private_hbm
            )
            nisa.dma_copy(
                dst=q_internal_remainder[0, :, :],
                src=q.ap(
                    pattern=[[seqlen_q, head_dim], [1, effective_q_tokens]],
                    offset=last_batch * head_dim * seqlen_q + seq_start,
                ),
            )

        # Allocate K/V sbuf for effective segment
        k_cache_sbuf_rem = allocator.alloc_sbuf_tensor(
            shape=(d_tile_size_par_dim, _K_TILE_SZ),
            dtype=k_sbuf_dtype,
            block_dim=[num_k_tiles_per_effective_seg, num_d_tiles],
            num_free_tiles=[num_k_tiles_per_effective_seg, num_d_tiles],
            align_to=32,
        )
        v_cache_sbuf_rem = allocator.alloc_sbuf_tensor(
            shape=(_V_TILE_SZ, d_tile_size_free_dim),
            dtype=v_sbuf_dtype,
            block_dim=[num_v_tiles_per_effective_seg, num_d_tiles_free_dim],
            num_free_tiles=[num_v_tiles_per_effective_seg, num_d_tiles_free_dim],
        )
        # Allocate HBM buffers for remainder batch (per-shard Q length)
        rem_softmax_shape_swa = (1, 128, num_grps_effective)
        out_o_hbm_rem = nl.ndarray(shape=(1, effective_q_tokens, head_dim), dtype=q.dtype, buffer=nl.private_hbm)
        out_neg_max_hbm_rem = nl.ndarray(shape=rem_softmax_shape_swa, dtype=nl.float32, buffer=nl.private_hbm)
        out_sum_hbm_rem = nl.ndarray(shape=rem_softmax_shape_swa, dtype=nl.float32, buffer=nl.private_hbm)

        # Adjust active_block_offset and prior parameters for each core.
        # Core 0: same as primary batch (its Q is the leading portion of the remainder).
        # Core 1: active starts after Core 0's active region; prior must cover
        # Core 0's active as part of its sliding-window prior context.
        active_block_offset_rem = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        prior_block_offset_rem = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
        effective_prior_len_rem = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)

        if shard_id == 0:
            # Core 0: Loads leading active KV, prior same as primary batch
            nisa.tensor_copy(dst=active_block_offset_rem, src=active_block_offset_swa)
            nisa.tensor_copy(dst=prior_block_offset_rem, src=prior_block_offset_swa)
            nisa.tensor_copy(dst=effective_prior_len_rem, src=effective_prior_len_sbuf)
        else:
            # Core 1: Loads trailing active KV, starting at
            # active_block_offset_swa + core0_num_blocks_per_effective_seg
            nisa.tensor_scalar(
                dst=active_block_offset_rem,
                data=active_block_offset_swa,
                op0=nl.add,
                operand0=core0_num_blocks_per_effective_seg,
            )
            # Core 1's prior: blocks right before its active segment
            # prior_block_offset = max(0, active_block_offset_rem - num_prior_blocks_to_load)
            nisa.tensor_scalar(
                dst=prior_block_offset_rem,
                data=active_block_offset_rem,
                op0=nl.subtract,
                operand0=num_prior_blocks_to_load,
                op1=nl.maximum,
                operand1=0,
            )
            # Core 1's effective_prior_len = min(num_prior_blocks_to_load * block_size,
            #                                    prior_tokens + core0_q_tokens)
            # prior_tokens + core0_q_tokens = total tokens before Core 1's active
            nisa.tensor_scalar(
                dst=effective_prior_len_rem,
                data=prior_tokens_sbuf,
                op0=nl.add,
                operand0=core0_q_tokens,
                op1=nl.minimum,
                operand1=num_prior_blocks_to_load * block_size,
            )
        # Derive batch/head indices for the remainder batch
        rem_b_i = last_batch // num_q_heads
        rem_h_i = (last_batch % num_q_heads) * num_kv_heads // num_q_heads

        # Load active KV cache for this core's portion
        load_kv_cache(
            k_cache,
            v_cache,
            block_tables,
            k_cache_sbuf_rem,
            v_cache_sbuf_rem,
            rem_b_i,
            rem_h_i,
            active_block_offset_rem,
            num_blocks_per_effective_seg,
            allocator,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
        )

        # Load at most window-sized prior KV for remainder; prior_used_len dynamically masks
        # (0 when no prior tokens, clamped to actual prior otherwise).
        k_prior_sbuf_rem = allocator.alloc_sbuf_tensor(
            shape=(d_tile_size_par_dim, _K_TILE_SZ),
            dtype=nl.bfloat16,
            block_dim=[num_prior_k_tiles, num_d_tiles],
            num_free_tiles=[num_prior_k_tiles, num_d_tiles],
            align_to=32,
        )
        v_prior_sbuf_rem = allocator.alloc_sbuf_tensor(
            shape=(_V_TILE_SZ, d_tile_size_free_dim),
            dtype=nl.bfloat16,
            block_dim=[num_prior_v_tiles, num_d_tiles_free_dim],
            num_free_tiles=[num_prior_v_tiles, num_d_tiles_free_dim],
        )
        load_kv_cache(
            k_cache,
            v_cache,
            block_tables,
            k_prior_sbuf_rem,
            v_prior_sbuf_rem,
            rem_b_i,
            rem_h_i,
            prior_block_offset_rem,
            num_prior_blocks_to_load,
            allocator,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
        )

        init_sbuf_addr = allocator.get_current_address()

        _attention_cte(
            q_internal_remainder,
            None,
            None,
            scale=scale,
            causal_mask=True,
            tp_q=tp_q,
            tp_k=False,
            tp_out=False,
            cache_softmax=True,
            skip_output_normalization=True,
            sliding_window=sliding_window,
            sink=sink[last_batch : last_batch + 1] if sink is not None else None,
            k_cache_sbuf=k_cache_sbuf_rem,
            v_cache_sbuf=v_cache_sbuf_rem,
            k_prior_sbuf=k_prior_sbuf_rem,
            v_prior_sbuf=v_prior_sbuf_rem,
            prior_used_len=effective_prior_len_rem,
            out_o_hbm=out_o_hbm_rem,
            out_neg_max_hbm=out_neg_max_hbm_rem,
            out_sum_hbm=out_sum_hbm_rem,
            init_sbuf_addr=init_sbuf_addr,
            k_scale_sb=k_scale_sb,
        )
        allocator.set_current_address(init_sbuf_addr)

        # Normalize and write results to HBM for remainder batch
        rem_sb_p = nl.tile_size.pmax
        rem_sm_pat = [[num_grps_effective, rem_sb_p], [1, num_grps_effective]]
        rem_o_tile_pat = [[head_dim, rem_sb_p], [1, head_dim]]

        rem_norm_addr = allocator.get_current_address()
        rem_sum_sb = allocator.alloc_sbuf_tensor(shape=(rem_sb_p, num_grps_effective), dtype=nl.float32)
        nisa.dma_copy(dst=rem_sum_sb, src=out_sum_hbm_rem.ap(pattern=rem_sm_pat, offset=0))
        rem_sum_recip_sb = allocator.alloc_sbuf_tensor(shape=(rem_sb_p, num_grps_effective), dtype=nl.float32)
        nisa.reciprocal(rem_sum_recip_sb, rem_sum_sb)

        rem_num_free = min(num_grps_effective, _MAX_FREE_TILES)
        rem_o_sb = allocator.alloc_sbuf_tensor(
            shape=(rem_sb_p, d_tile_size_par_dim),
            dtype=nl.bfloat16,
            block_dim=[num_grps_effective, num_d_tiles],
            num_free_tiles=[rem_num_free, num_d_tiles],
        )
        if tp_out:
            rem_o_tp_psum = nl.ndarray(
                (d_tile_size_par_dim, rem_sb_p), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0)
            )
            rem_o_tp_sb = allocator.alloc_sbuf_tensor(shape=(d_tile_size_par_dim, rem_sb_p), dtype=nl.bfloat16)
        for local_grp_i in range(num_grps_effective):
            global_grp_i = grp_start + local_grp_i
            grp_start_pos = global_grp_i * 128

            for d_idx in range(num_d_tiles):
                d_offset = d_idx * d_tile_size_par_dim
                nisa.dma_copy(
                    dst=rem_o_sb[local_grp_i][d_idx],
                    src=out_o_hbm_rem.ap(
                        pattern=[[head_dim, rem_sb_p], [1, d_tile_size_par_dim]],
                        offset=local_grp_i * rem_sb_p * head_dim + d_offset,
                    ),
                )
                # Delayed V dequant: fold v_scale multiply into normalization tensor_scalar
                if v_scale_sb is not None:
                    nisa.tensor_scalar(
                        dst=rem_o_sb[local_grp_i][d_idx],
                        data=rem_o_sb[local_grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=rem_sum_recip_sb[:, local_grp_i],
                        op1=nl.multiply,
                        operand1=v_scale_sb,
                    )
                else:
                    nisa.tensor_scalar(
                        dst=rem_o_sb[local_grp_i][d_idx],
                        data=rem_o_sb[local_grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=rem_sum_recip_sb[:, local_grp_i],
                    )
                if tp_out:
                    nisa.nc_transpose(dst=rem_o_tp_psum, data=rem_o_sb[local_grp_i][d_idx])
                    nisa.tensor_copy(dst=rem_o_tp_sb, src=rem_o_tp_psum)
                    nisa.dma_copy(
                        dst=result.ap(
                            pattern=[[seqlen_q, d_tile_size_par_dim], [1, rem_sb_p]],
                            offset=last_batch * head_dim * seqlen_q + d_offset * seqlen_q + grp_start_pos,
                        ),
                        src=rem_o_tp_sb,
                    )
                else:
                    nisa.dma_copy(
                        dst=result.ap(
                            pattern=[[head_dim, rem_sb_p], [1, d_tile_size_par_dim]],
                            offset=last_batch * num_grps * rem_sb_p * head_dim
                            + global_grp_i * rem_sb_p * head_dim
                            + d_offset,
                        ),
                        src=rem_o_sb[local_grp_i][d_idx],
                    )
        allocator.set_current_address(rem_norm_addr)

    return result


def attention_segmented_cte(
    q: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    v_cache: nl.NkiTensor,
    block_tables: nl.NkiTensor,
    prior_tokens: nl.NkiTensor,
    block_size: int,
    prior_seg_size: int,
    scale: float = 1.0,
    tp_q: bool = True,
    tp_out: bool = False,
    sliding_window: Optional[int] = None,
    sink: Optional[nl.NkiTensor] = None,
    num_q_heads: int = 1,
    k_pre_transposed: bool = False,
    fp8_packed: bool = False,
    k_scale: Optional[nl.NkiTensor] = None,
    v_scale: Optional[nl.NkiTensor] = None,
    kvp_q_offset: Optional[nl.NkiTensor] = None,
    kvp_rank_id: Optional[nl.NkiTensor] = None,
    kvp_group_size: int = 0,
    kvp_cp_offset_int: int = 0,
    kvp_seg_block_offset_int: int = 0,
    kvp_prior_load_blocks: int = 0,
    kvp_prior_fully_visible: bool = False,
):
    """
    Segmented attention computation with block-based KV cache and prefix caching.

    SEGMENTED ATTENTION OVERVIEW:
    ================================

    Case 1: Partial Prior (prior_tokens=640, prior_seg_size=512, block_size=128)
    -----------------------------------------------------------------------
    KV Cache Block Layout:
    ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
    │   0    │   1    │   2    │   3    │   4    │   5    │   6    │   7    │   8    │  Block indices
    └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
    └────Full Prior 0 (4 blks)──────────┴Partial ┴─────────Active (4 blks)───────────┘  Segments
         offset=0                        Prior             offset=5
                                         offset=4

    Iteration: Active+Partial(causal) → Prior0(no causal)

    Case 2: Full Prior Only (prior_tokens=1024, prior_seg_size=512, block_size=128)
    --------------------------------------------------------------------------
    KV Cache Block Layout:
    ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
    │   0    │   1    │   2    │   3    │   4    │   5    │   6    │   7    │   8    │   9    │   10   │   11   │  Block indices
    └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
    └────Full Prior 0 (4 blks)──────────┴────Full Prior 1 (4 blks)──────────┴──────Active (4 blks)──────────────┘  Segments
         offset=0                            offset=4                              offset=8

    Iteration: Active(causal) → Prior1(no causal) → Prior0(no causal)

    Pseudo-code algorithm:
        num_full_prior_segments = floor(prior_tokens / prior_seg_size)
        partial_prior_tokens = prior_tokens - num_full_prior_segments * prior_seg_size

        # Load active segment KV
        k_active, v_active = load_kv_cache(block_tables, offset=prior_tokens // block_size)

        # First iteration: Process active segment with causal mask
        if partial_prior_tokens > 0:
            # Partial segment prior exists
            k_prior, v_prior = load_kv_cache(block_tables, offset=num_full_prior_segments * num_blocks_per_seg)
            output = _attention_cte(q, k_prior, v_prior, k_active, v_active,
                                  causal_mask=True, prior_used_len=partial_prior_tokens)
        else:
            # Full segment prior or no prior
            output = _attention_cte(q, k_active, v_active, causal_mask=True)

        # Remaining iterations: Process full prior segments without causal mask
        for i in range(num_full_prior_segments):
            k_seg, v_seg = load_kv_cache(block_tables, offset=decremented_offset)
            seg_output = _attention_cte(q, k_seg, v_seg, causal_mask=False)
            output = reduce_one_batch(output, seg_output)  # Online softmax rescaling

        return output

    LNC2 SHARDING STRATEGY:
    ======================

    Primary Sharding (Even bs_q):
    ------------------------------
    Divides bs_q evenly across 2 cores. Example with bs_q=4:

        Core 0: Q[0], Q[1]  (num_bs_per_shard = 2)
        Core 1: Q[2], Q[3]  (num_bs_per_shard = 2)

    Secondary Sharding (Odd bs_q with Remainder):
    ----------------------------------------------
    Primary batches divided evenly, remainder batch uses 50/50 sequence sharding.
    Example with bs_q=3, prior_seg_size=2048 (16 groups of 128 tokens):

        Core 0: Q[0] (primary)
        Core 1: Q[1] (primary)

        Q[2] (remainder) - 50/50 SEQUENCE SPLIT:
        ┌─────────────────────────────────────────────────────────────┐
        │                 Q[2]: 2048 tokens (16 groups)               │
        ├──────────────────────────────┬──────────────────────────────┤
        │  Core 0: Groups [0-7]        │  Core 1: Groups [8-15]       │
        │  Tokens [0-1023]             │  Tokens [1024-2047]          │
        │  effective_prior_seg_size = 1024   │  effective_prior_seg_size = 1024   │
        └──────────────────────────────┴──────────────────────────────┘

    Prior Segment Handling (with prior_tokens > 0):
    ------------------------------------------------
    When original segment has N prior segments, effective segments double.
    Example: prior_seg_size=2048, prior=2048 (1 segment), remainder with 50/50 split:

        Original (prior_seg_size=2048):
        ┌────────────┬─────────────┐
        │ Prior Seg 0│ Active Seg 1│
        │ (2048 tok) │ (2048 tok)  │
        └────────────┴─────────────┘

        After 50/50 split (effective_prior_seg_size=1024):
        ┌──────┬──────┬──────┬──────┐
        │ P0.0 │ P0.1 │ A1.0 │ A1.1 │  (Each = 1024 tokens)
        └──────┴──────┴──────┴──────┘

        Core 0's view (segments from Core 0's perspective):
        - Seg 0: Prior tokens [0-1023]      (P0.0)
        - Seg 1: Prior tokens [1024-2047]   (P0.1)
        - Seg 2: Active tokens [0-1023]     (A1.0) ← Core 0's active
        Total: 2 prior segments (2N where N=1)

        Core 1's view (segments from Core 1's perspective):
        - Seg 0: Prior tokens [0-1023]      (P0.0)
        - Seg 1: Prior tokens [1024-2047]   (P0.1)
        - Seg 2: Active tokens [0-1023]     (A1.0) ← Core 0's active (prior for Core 1!)
        - Seg 3: Active tokens [1024-2047]  (A1.1) ← Core 1's active
        Total: 3 prior segments (2N+1)

        Implementation:
        - Both cores do 2N iterations in main loop
        - Core 1 does +1 extra iteration at block_offset=0 to process Seg 0


    Args:
        q: Query tensor with shape (batch_size, seqlen_q, d) when tp_q=True
        k_cache: K cache in HBM with shape (num_blocks, block_size, num_kv_head, head_dim)
        v_cache: V cache in HBM with shape (num_blocks, block_size, num_kv_head, head_dim)
        block_tables: Block table tensor with shape (batch_size, max_blocks_per_seq).
                     max_blocks_per_seq only needs to cover
                     ceil((prior_tokens + seqlen_q) / block_size); the kernel pads
                     internally when seqlen_q < prior_seg_size so the traced
                     partial-prior speculative read stays in-bounds.
        prior_tokens: Total number of prior (cached) tokens, shape (1, 1). Must be multiple of block_size.
        block_size: Size of each block in the KV cache
        prior_seg_size: Size of each KV segment to process iteratively
        scale: Scaling factor for attention scores (default 1.0)
        tp_q: Query tensor transpose flag (default True)
        tp_out: Output tensor transpose flag (default False)
        k_pre_transposed: If True, K cache is already stored in transposed layout
            (head_dim, block_size) per block, written by _quantize_and_store_k_transposed in qkv_cte.
        k_scale: Optional per-head-dim dequantization scale for K cache, shape (128, 1).
            When provided, Q is scaled by k_scale before QK^T matmul (delayed dequant).
        v_scale: Optional per-head-dim dequantization scale for V cache, shape (128, 1).
            When provided, the output is scaled by v_scale after PV matmul normalization.
        sliding_window: Optional sliding window size for attention (default None = disabled).
        sink: Optional sink token tensor (default None).
        num_q_heads: Number of query heads per batch item (default 1).
        kvp_q_offset: Optional [1, 1] int32, causal mask offset for KV-parallel mode.
            When set, enables KV-parallel mode and returns softmax stats for merging.
        kvp_rank_id: Optional [1, 1] int32, this rank's index within the KV-parallel group.
            Required when kvp_group_size > 0 for interleaved KV distribution.
        kvp_group_size: Number of ranks sharing KV cache in round-robin fashion (0 = disabled).
            When > 0, enables interleaved mode where rank r holds global blocks r, r+R, r+2R, ...
        kvp_cp_offset_int: Compile-time Q global offset for tile visibility checks (default 0).
        kvp_seg_block_offset_int: Compile-time segment block offset for tile visibility checks (default 0).
        kvp_prior_load_blocks: Number of blocks to load for partial prior in boundary-aligned mode (default 0).
        kvp_prior_fully_visible: If True, prior segments are fully visible and skip causal masking (default False).

    Returns:
        If kvp_q_offset is None: output tensor with attention results. Shape depends on tp_out parameter.
        If kvp_q_offset is set: tuple of (output, out_neg_max_hbm, out_sum_recip_hbm) for softmax stat
            reduction by the caller.

    Example calculations:
        prior_tokens=0:    prior_last_segment_tokens=0,   iterations=1
        prior_tokens=640:  prior_last_segment_tokens=128, iterations=2 (prior_seg_size=512)
        prior_tokens=1024: prior_last_segment_tokens=512, iterations=2 (prior_seg_size=512)
    """
    # Extract dimensions
    if tp_q:
        bs_q, seqlen_q, d = q.shape
    else:
        bs_q, d, seqlen_q = q.shape

    # Derive dims from v_cache so we don't depend on k_cache's shape, which
    # varies by layout (see k_pre_transposed argument on load_kv_cache).
    num_kv_head = v_cache.shape[1]
    head_dim = v_cache.shape[3]
    bs, max_blocks_per_seq = block_tables.shape

    # D-tile parameters for 2D K/V SBUF layout
    num_d_tiles = math.ceil(head_dim / _D_TILE_SZ)
    d_tile_size_par_dim = min(head_dim, _D_TILE_SZ)
    num_d_tiles_free_dim = math.ceil(head_dim / (_V_D_TILE_SZ))
    d_tile_size_free_dim = min(head_dim, _V_D_TILE_SZ)

    # Get sharding info for multi-core parallelization
    grid_ndim, num_shard, shard_id = get_verified_program_sharding_info("attention_segmented_cte", max_sharding=2)

    # When kvp_q_offset is set (KV-parallel mode), the KVP kernel already handles LNC sharding
    # externally. Override to single-core mode.
    if kvp_q_offset is not None:
        num_shard = 1
        shard_id = 0
    internal_buffer = nl.private_hbm
    result_buffer = nl.hbm if kvp_q_offset != None else nl.shared_hbm

    # Primary sharding: divide bs_q (batch_size * num_q_heads) evenly across shards
    num_bs_per_shard = bs_q // num_shard
    bs_offset = shard_id * num_bs_per_shard

    # Secondary sharding: handle remainder on sequence dimension if bs_q is odd
    has_remainder = (bs_q % num_shard) != 0
    last_batch = bs_q - 1

    # Validate inputs
    kernel_assert(seqlen_q % 128 == 0, f"Query seqlen {seqlen_q} must be a multiple of 128")
    kernel_assert(seqlen_q % block_size == 0, f"Query seqlen {seqlen_q} must be divisible by block_size {block_size}")
    kernel_assert(d == head_dim, f"Query head_dim {d} must match cache head_dim {head_dim}")
    kernel_assert(
        prior_seg_size % block_size == 0,
        f"prior_seg_size {prior_seg_size} must be divisible by block_size {block_size}",
    )
    kernel_assert(head_dim <= 512, f"head_dim must be <= 512 (got {head_dim}). Larger head_dim not yet supported.")
    # TODO: Fix d>256 + large prior_seg_size
    kernel_assert(
        not (head_dim > 256 and prior_seg_size > 2048),
        f"head_dim > 256 with prior_seg_size > 2048 is not supported. "
        f"Got head_dim={head_dim}, prior_seg_size={prior_seg_size}.",
    )

    num_blocks_per_seg = prior_seg_size // block_size

    # Initialize allocator
    allocator = ModularAllocator(initial_address=0)

    # Pad block_tables internally so the fused partial-prior helper's
    # compile-time-traced one-past read at
    # (num_full_prior_segments + 1) * num_blocks_per_seg stays in-bounds.
    # Unconditional: the NKI compiler's pessimistic bound check on dynamic
    # scalar_offset into block_tables makes any conditional predicate
    # incomplete. Done before the SWA dispatch so both paths benefit.
    padded_width = (max_blocks_per_seq // num_blocks_per_seg + 1) * num_blocks_per_seg

    if padded_width > max_blocks_per_seq:
        block_tables_internal = nl.ndarray(shape=(bs, padded_width), dtype=block_tables.dtype, buffer=nl.private_hbm)
        pad_addr = allocator.get_current_address()
        pad_scratch = allocator.alloc_sbuf_tensor(shape=(bs, padded_width), dtype=block_tables.dtype)
        nisa.memset(pad_scratch[...], value=0)
        nisa.dma_copy(dst=block_tables_internal, src=pad_scratch)
        nisa.dma_copy(
            dst=block_tables_internal.ap(pattern=[[padded_width, bs], [1, max_blocks_per_seq]], offset=0),
            src=block_tables,
        )
        allocator.set_current_address(pad_addr)
        block_tables = block_tables_internal
        max_blocks_per_seq = padded_width

    # Sliding window attention: use simplified single-iteration path (contiguous only).
    # For round-robin KV distribution, the normal segmented path with bound conversion handles SWA.
    if sliding_window is not None and sliding_window > 0 and kvp_group_size == 0:
        kernel_assert(
            sliding_window % block_size == 0,
            f"sliding_window {sliding_window} must be divisible by block_size {block_size}",
        )
        return _attention_segmented_cte_swa_impl(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens,
            block_size=block_size,
            prior_seg_size=prior_seg_size,
            scale=scale,
            tp_q=tp_q,
            tp_out=tp_out,
            sliding_window=sliding_window,
            sink=sink,
            num_q_heads=num_q_heads,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    num_k_tiles_per_seg = math.ceil(prior_seg_size / _K_TILE_SZ)
    num_v_tiles_per_seg = num_k_tiles_per_seg * (_K_TILE_SZ // _V_TILE_SZ)
    num_grps = math.ceil(seqlen_q / 128)

    # Active segment tile/block counts (may differ from prior when seqlen_q != prior_seg_size)
    num_active_blocks = seqlen_q // block_size
    num_k_tiles_active = math.ceil(seqlen_q / _K_TILE_SZ)
    num_v_tiles_active = num_k_tiles_active * (_K_TILE_SZ // _V_TILE_SZ)

    # K/V sbuf must be large enough for both active and prior segments
    num_k_tiles_sbuf = max(num_k_tiles_per_seg, num_k_tiles_active)
    num_v_tiles_sbuf = max(num_v_tiles_per_seg, num_v_tiles_active)

    # Load KV dequantization scales into SBUF if provided (FP8 KV cache support)
    if k_scale is not None:
        k_scale_sb = allocator.alloc_sbuf_tensor(shape=(nl.tile_size.pmax, 1), dtype=nl.float32)
        nisa.dma_copy(dst=k_scale_sb, src=k_scale)
    else:
        k_scale_sb = None
    if v_scale is not None:
        v_scale_sb = allocator.alloc_sbuf_tensor(shape=(nl.tile_size.pmax, 1), dtype=nl.float32)
        nisa.dma_copy(dst=v_scale_sb, src=v_scale)
    else:
        v_scale_sb = None

    # Compute segment offsets ONCE (both cores execute this for consistent control flow)
    prior_tokens_sbuf = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.dma_copy(dst=prior_tokens_sbuf, src=prior_tokens)

    # num_full_prior_segments = floor(prior_tokens / prior_seg_size)
    num_full_prior_segments_f32 = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.float32)
    num_full_prior_segments_i32 = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(
        dst=num_full_prior_segments_f32, data=prior_tokens_sbuf, op0=nl.multiply, operand0=1 / prior_seg_size
    )
    floor_nisa_kernel(
        src_t=num_full_prior_segments_f32, dst_t=num_full_prior_segments_i32, p_size=1, f_size=1, allocator=allocator
    )

    # Compute block offsets
    block_size_shift = int(math.log2(block_size))
    active_block_offset = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(dst=active_block_offset, data=prior_tokens_sbuf, op0=nl.right_shift, operand0=block_size_shift)

    # prior_block_offset is allocated per-batch inside the processing loop
    # (see comment below). Allocating here instead would cause cross-batch
    # compiler scheduling hazards: fused_impl decrements this tensor every
    # full-prior iteration, and the compiler can schedule later batches'
    # DMA descriptors using the previous batch's decremented value.

    # Compute partial prior and flags
    temp_seg_tokens = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(dst=temp_seg_tokens, data=num_full_prior_segments_i32, op0=nl.multiply, operand0=prior_seg_size)

    partial_prior_tokens = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_tensor(dst=partial_prior_tokens, data1=prior_tokens_sbuf, data2=temp_seg_tokens, op=nl.subtract)

    is_partial_prior_segment = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(dst=is_partial_prior_segment, data=partial_prior_tokens, op0=nl.greater, operand0=0)

    is_not_partial_prior_segment = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
    nisa.tensor_scalar(
        dst=is_not_partial_prior_segment, data=is_partial_prior_segment, op0=nl.subtract, operand0=1, reverse0=True
    )

    # Allocate K/V sbuf for max(active, prior) tiles (reused across iterations).
    # fused_segmented_attention_impl aliases k_prior_sbuf / v_prior_sbuf onto
    # these buffers via list slicing, saving ~48 KB per partition on the hot
    # non-KVP path.
    # Allocate SBUF in the cache's native dtype so dma_copy loads directly
    # without tensor_copy widening. nc_matmul accepts fp8 natively.
    k_sbuf_dtype = k_cache.dtype
    v_sbuf_dtype = v_cache.dtype
    k_cache_sbuf = allocator.alloc_sbuf_tensor(
        shape=(d_tile_size_par_dim, _K_TILE_SZ),
        dtype=k_sbuf_dtype,
        block_dim=[num_k_tiles_sbuf, num_d_tiles],
        num_free_tiles=[num_k_tiles_sbuf, num_d_tiles],
        align_to=32,
    )
    v_cache_sbuf = allocator.alloc_sbuf_tensor(
        shape=(_V_TILE_SZ, d_tile_size_free_dim),
        dtype=v_sbuf_dtype,
        block_dim=[num_v_tiles_sbuf, num_d_tiles_free_dim],
        num_free_tiles=[num_v_tiles_sbuf, num_d_tiles_free_dim],
    )

    # Allocate HBM buffers for unnormalized output and softmax stats (single batch, reused)
    # Uses tp_out=False for intermediate HBM (matches reduce_one_batch layout)
    softmax_shape = (1, 128, num_grps)
    o_prev_hbm = nl.ndarray(shape=(1, seqlen_q, head_dim), dtype=nl.float32, buffer=internal_buffer)
    neg_max_prev_hbm = nl.ndarray(shape=softmax_shape, dtype=nl.float32, buffer=internal_buffer)
    sum_prev_hbm = nl.ndarray(shape=softmax_shape, dtype=nl.float32, buffer=internal_buffer)
    o_curr_hbm = nl.ndarray(shape=(1, seqlen_q, head_dim), dtype=nl.float32, buffer=internal_buffer)
    neg_max_curr_hbm = nl.ndarray(shape=softmax_shape, dtype=nl.float32, buffer=internal_buffer)
    sum_curr_hbm = nl.ndarray(shape=softmax_shape, dtype=nl.float32, buffer=internal_buffer)

    # Copy final results to HBM (allocate for full bs_q, write only assigned portion)
    # Intermediates always use non-transposed layout (tp_out=False) for reduce_one_batch compatibility.
    # When tp_out=True, we transpose during the final normalize+write step.
    if tp_out:
        result = nl.ndarray(shape=(bs_q, head_dim, seqlen_q), dtype=q.dtype, buffer=result_buffer)
    else:
        result = nl.ndarray(shape=(bs_q, seqlen_q, head_dim), dtype=q.dtype, buffer=result_buffer)

    # Allocate softmax stats tensors for KV-parallel mode
    if kvp_q_offset != None:
        out_neg_max_hbm = nl.ndarray(shape=(bs_q, seqlen_q), dtype=nl.float32, buffer=result_buffer)
        out_sum_recip_hbm = nl.ndarray(shape=(bs_q, seqlen_q), dtype=nl.float32, buffer=result_buffer)

    # Load kvp_q_offset into SBUF once (reused per batch)
    kvp_offset_sbuf = None
    if kvp_q_offset != None:
        kvp_offset_sbuf = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.dma_copy(dst=kvp_offset_sbuf, src=kvp_q_offset)

    # Workaround for NCC_IBIR251: Allocate Q buffer once for single batch
    # Makes Q "internal" so dma_transpose/access patterns work in dynamic loops (LNC2)
    if tp_q:
        q_internal = nl.ndarray(shape=(1, seqlen_q, head_dim), dtype=q.dtype, buffer=internal_buffer)
    else:
        q_internal = nl.ndarray(shape=(1, head_dim, seqlen_q), dtype=q.dtype, buffer=internal_buffer)

    # Process primary batches one at a time using helper function (skip if no primary batches)
    for b_idx in range(num_bs_per_shard):
        batch_id = b_idx + bs_offset  # Global bs_q index

        # Derive batch index and KV head index from batch_id for GQA
        batch_b_i = batch_id // num_q_heads
        batch_h_i = (batch_id % num_q_heads) * num_kv_head // num_q_heads

        # Copy this batch's query data (layout matches tp_q)
        if tp_q:
            nisa.dma_copy(
                dst=q_internal[0, :, :],
                src=q.ap(
                    pattern=[[head_dim, seqlen_q], [1, head_dim]],
                    offset=batch_id * seqlen_q * head_dim,
                ),
            )
        else:
            nisa.dma_copy(
                dst=q_internal[0, :, :],
                src=q.ap(
                    pattern=[[seqlen_q, head_dim], [1, seqlen_q]],
                    offset=batch_id * head_dim * seqlen_q,
                ),
            )

        # Allocate a fresh prior_block_offset SBUF tensor per batch to avoid
        # cross-batch aliasing. fused_segmented_attention_impl's full-prior
        # loop decrements this value every iteration; without a fresh per-
        # batch tensor, the compiler can schedule subsequent batches'
        # compiled DMA descriptors using the decremented (zero or underflow)
        # value from the previous batch instead of the restored value. That
        # causes HW scalar-DGE OOB on multi-batch (bs_q > 1) configs with
        # >=2 full-prior iterations.
        prior_block_offset = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
        nisa.tensor_scalar(
            dst=prior_block_offset, data=num_full_prior_segments_i32, op0=nl.multiply, operand0=num_blocks_per_seg
        )

        # Process this single batch
        # fused_segmented_attention_impl handles both KVP and non-KVP via kvp_q_offset parameter.
        # Note: fused_segmented_attention_impl handles allocator reset internally
        fused_segmented_attention_impl(
            q_hbm=q_internal,
            num_batches=1,
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
            prior_tokens_sbuf=prior_tokens_sbuf,
            num_full_prior_segments_i32=num_full_prior_segments_i32,
            partial_prior_tokens=partial_prior_tokens,
            is_partial_prior_segment=is_partial_prior_segment,
            is_not_partial_prior_segment=is_not_partial_prior_segment,
            active_block_offset=active_block_offset,
            prior_block_offset=prior_block_offset,
            allocator=allocator,
            prior_seg_size=prior_seg_size,
            block_size=block_size,
            scale=scale,
            head_dim=head_dim,
            num_grps=num_grps,
            num_active_blocks=num_active_blocks,
            num_k_tiles_active=num_k_tiles_active,
            num_v_tiles_active=num_v_tiles_active,
            num_blocks_per_seg=num_blocks_per_seg,
            num_k_tiles_per_seg=num_k_tiles_per_seg,
            num_v_tiles_per_seg=num_v_tiles_per_seg,
            b_i=batch_b_i,
            h_i=batch_h_i,
            tp_q=tp_q,
            tp_out=tp_out,
            load_kv_cache_fn=load_kv_cache,
            attention_cte_fn=_attention_cte,
            sink=sink[batch_id : batch_id + 1] if sink is not None else None,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
            k_scale_sb=k_scale_sb,
            kvp_q_offset=kvp_offset_sbuf,
            kvp_rank_id=kvp_rank_id,
            kvp_group_size=kvp_group_size,
            sliding_window=sliding_window if sliding_window != None else 0,
            kvp_cp_offset_int=kvp_cp_offset_int,
            kvp_seg_block_offset_int=kvp_seg_block_offset_int,
            kvp_prior_load_blocks=kvp_prior_load_blocks,
            kvp_prior_fully_visible=kvp_prior_fully_visible,
        )

        # Normalize and write results to final output for this batch
        # o_prev_hbm[0] has unnormalized output, normalize (divide by S) and write to result[batch_id]
        sb_p = nl.tile_size.pmax
        sm_pat = [[num_grps, sb_p], [1, num_grps]]
        o_tile_pat = [[head_dim, sb_p], [1, head_dim]]

        norm_addr = allocator.get_current_address()
        sum_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
        nisa.dma_copy(dst=sum_sb, src=sum_prev_hbm.ap(pattern=sm_pat, offset=0))
        sum_recip_sb = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
        nisa.reciprocal(sum_recip_sb, sum_sb)

        num_free = min(num_grps, _MAX_FREE_TILES)
        o_sb = allocator.alloc_sbuf_tensor(
            shape=(sb_p, d_tile_size_par_dim),
            dtype=nl.bfloat16,
            block_dim=[num_grps, num_d_tiles],
            num_free_tiles=[num_free, num_d_tiles],
        )
        if tp_out:
            o_tp_psum = nl.ndarray((d_tile_size_par_dim, sb_p), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0))
            o_tp_sb = allocator.alloc_sbuf_tensor(shape=(d_tile_size_par_dim, sb_p), dtype=nl.bfloat16)
        for grp_i in range(num_grps):
            for d_idx in range(num_d_tiles):
                d_offset = d_idx * d_tile_size_par_dim
                nisa.dma_copy(
                    dst=o_sb[grp_i][d_idx],
                    src=o_prev_hbm.ap(
                        pattern=[[head_dim, sb_p], [1, d_tile_size_par_dim]],
                        offset=grp_i * sb_p * head_dim + d_offset,
                    ),
                )
                # Delayed V dequant: fold v_scale multiply into normalization tensor_scalar
                if v_scale_sb is not None:
                    nisa.tensor_scalar(
                        dst=o_sb[grp_i][d_idx],
                        data=o_sb[grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=sum_recip_sb[:, grp_i],
                        op1=nl.multiply,
                        operand1=v_scale_sb,
                    )
                else:
                    nisa.tensor_scalar(
                        dst=o_sb[grp_i][d_idx],
                        data=o_sb[grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=sum_recip_sb[:, grp_i],
                    )
                # Write to result[batch_id]
                if tp_out:
                    # nc_transpose (128, head_dim) → (head_dim, 128), then write with transposed AP
                    nisa.nc_transpose(dst=o_tp_psum, data=o_sb[grp_i][d_idx])
                    nisa.tensor_copy(dst=o_tp_sb, src=o_tp_psum)
                    nisa.dma_copy(
                        dst=result.ap(
                            pattern=[[seqlen_q, d_tile_size_par_dim], [1, sb_p]],
                            offset=batch_id * head_dim * seqlen_q + d_offset * seqlen_q + grp_i * sb_p,
                        ),
                        src=o_tp_sb,
                    )
                else:
                    nisa.dma_copy(
                        dst=result.ap(
                            pattern=[[head_dim, sb_p], [1, d_tile_size_par_dim]],
                            offset=batch_id * num_grps * sb_p * head_dim + grp_i * sb_p * head_dim + d_offset,
                        ),
                        src=o_sb[grp_i][d_idx],
                    )
        allocator.set_current_address(norm_addr)

        # Write softmax stats for KV-parallel mode
        if kvp_q_offset is not None:
            stats_addr = allocator.get_current_address()
            neg_max_sb_kvp = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            sum_sb_kvp = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            sum_recip_sb_kvp = allocator.alloc_sbuf_tensor(shape=(sb_p, num_grps), dtype=nl.float32)
            nisa.dma_copy(dst=neg_max_sb_kvp, src=neg_max_prev_hbm.ap(pattern=sm_pat, offset=0))
            nisa.dma_copy(dst=sum_sb_kvp, src=sum_prev_hbm.ap(pattern=sm_pat, offset=0))
            nisa.reciprocal(sum_recip_sb_kvp, sum_sb_kvp)
            # Write in token-ordered layout: token t = p + g * sb_p stored at offset t
            # AP pattern [[1, sb_p], [sb_p, num_grps]] stores [p, g] at offset p + g * sb_p
            tok_pat = [[1, sb_p], [sb_p, num_grps]]
            nisa.dma_copy(
                dst=out_neg_max_hbm.ap(pattern=tok_pat, offset=batch_id * sb_p * num_grps), src=neg_max_sb_kvp
            )
            nisa.dma_copy(
                dst=out_sum_recip_hbm.ap(pattern=tok_pat, offset=batch_id * sb_p * num_grps), src=sum_recip_sb_kvp
            )
            allocator.set_current_address(stats_addr)

    # Secondary sharding: handle remainder bs_q item with asymmetric sequence sharding.
    #
    # Core 0 handles Q[0 : ceil(num_grps/2)*128], Core 1 handles Q[ceil(num_grps/2)*128 :
    # num_grps*128]. For num_grps == 1 the split is degenerate — Core 0 does the full
    # remainder and Core 1 short-circuits.
    #
    # Prior-segment chunking reuses the top-level prior_seg_size on both cores so
    # num_full_prior_segments_i32 is identical across cores; this keeps the 3
    # nl.dynamic_range loops inside fused_segmented_attention_impl iterating the
    # same number of times (LNC2 sync requirement).
    core0_grp_length = (num_grps + 1) // 2
    core1_grp_length = num_grps // 2
    if shard_id == 0:
        grp_length = core0_grp_length
        grp_start = 0
    else:
        grp_length = core1_grp_length
        grp_start = core0_grp_length

    # Both cores must enter the remainder block with matching dynamic_range
    # iteration counts to satisfy LNC2 basic-block symmetry (NCC_IXGM002).
    # When num_grps == 1, Core 1's grp_length == 0 — instead of short-
    # circuiting (which would leave Core 1's IR with fewer basic blocks
    # than Core 0's), we use effective_grp_length = max(grp_length, 1) for
    # sizing and have Core 1 trace the same code paths against private_hbm
    # scratch. Its final result-write is redirected away from the shared
    # `result` at the HBM normalization step below.
    run_remainder = has_remainder
    # Static-Python predicate: Core 0 always has real work; Core 1 has real
    # work only when core1_grp_length > 0.
    run_work = grp_length > 0
    effective_grp_length = max(grp_length, 1)
    # Core 1's dummy path needs grp_start=0 so its (discarded) DMA writes
    # target a valid in-bounds offset of its scratch buffer.
    effective_grp_start = grp_start if run_work else 0

    if run_remainder:
        # Per-shard active-segment token/tile counts (drive load_kv_cache + fused_impl's
        # active processing; NOT the dynamic prior-segment loops).
        # Uses effective_grp_length so Core 1's dummy path still has valid
        # (non-zero) sizing for the allocations and dynamic_range parameters.
        effective_q_tokens = effective_grp_length * 128
        num_blocks_per_effective_seg = effective_q_tokens // block_size
        num_k_tiles_per_effective_seg = math.ceil(effective_q_tokens / _K_TILE_SZ)
        num_v_tiles_per_effective_seg = num_k_tiles_per_effective_seg * (_K_TILE_SZ // _V_TILE_SZ)

        # Core 0's active-segment sizing — used by Core 1's "+1 extra iteration" to
        # attend over Core 0's active KV region, and as the shared effective
        # segment size below (ceil so it's >= core1's).
        core0_q_tokens = core0_grp_length * 128
        core0_num_blocks_per_effective_seg = core0_q_tokens // block_size
        core0_num_k_tiles_per_effective_seg = math.ceil(core0_q_tokens / _K_TILE_SZ)
        core0_num_v_tiles_per_effective_seg = core0_num_k_tiles_per_effective_seg * (_K_TILE_SZ // _V_TILE_SZ)

        # Shared effective prior-segment size for the remainder's fused_impl call.
        # Both cores pass the same value so num_full_prior_segments (computed from
        # prior_tokens / effective_prior_seg_size_shared) is identical across
        # cores and the 3 inner nl.dynamic_range loops iterate in lockstep
        # (LNC2 sync requirement).
        #
        # Use the top-level prior_seg_size so the remainder path iterates the
        # SAME number of prior segments as the primary path (e.g. 3 iterations
        # for prior_tokens=32512, prior_seg_size=8192). Using a smaller size
        # here (e.g. core0_q_tokens=128) would balloon the iteration count to
        # prior_tokens/core0_q_tokens ≈ 254, amplifying bf16 accumulation
        # drift in the flash-attention online softmax to ~15% rel error. The
        # k_cache_sbuf / v_cache_sbuf allocations above (lines 1708–1722) are
        # already sized for max(num_k_tiles_per_seg, num_k_tiles_active), so
        # using prior_seg_size fits.
        effective_prior_seg_size_shared = prior_seg_size
        num_blocks_per_effective_seg_shared = effective_prior_seg_size_shared // block_size
        num_k_tiles_per_effective_seg_shared = math.ceil(effective_prior_seg_size_shared / _K_TILE_SZ)
        num_v_tiles_per_effective_seg_shared = num_k_tiles_per_effective_seg_shared * (_K_TILE_SZ // _V_TILE_SZ)

        # Sequence start position for this core's Q slice. Core 1's dummy
        # path (run_work=False) clamps to 0 so the DMA source offset
        # below stays in-bounds (reads bytes overlapping Core 0's Q slice;
        # result is discarded downstream).
        seq_start = grp_start * 128 if run_work else 0

        # Copy only this core's portion of Q to q_internal_remainder.
        if tp_q:
            q_internal_remainder = nl.ndarray(
                shape=(1, effective_q_tokens, head_dim), dtype=q.dtype, buffer=nl.private_hbm
            )
            nisa.dma_copy(
                dst=q_internal_remainder[0, :, :],
                src=q.ap(
                    pattern=[[head_dim, effective_q_tokens], [1, head_dim]],
                    offset=last_batch * seqlen_q * head_dim + seq_start * head_dim,
                ),
            )
        else:
            q_internal_remainder = nl.ndarray(
                shape=(1, head_dim, effective_q_tokens), dtype=q.dtype, buffer=nl.private_hbm
            )
            nisa.dma_copy(
                dst=q_internal_remainder[0, :, :],
                src=q.ap(
                    pattern=[[seqlen_q, head_dim], [1, effective_q_tokens]],
                    offset=last_batch * head_dim * seqlen_q + seq_start,
                ),
            )

        # Allocate HBM buffers for remainder (per-shard Q length).
        # Use effective_grp_length so Core 1's dummy path gets valid
        # non-zero shapes; its writes are isolated in private_hbm.
        #
        # Intermediate unnormalized output buffers are f32 to match the
        # primary batch path (lines 1732/1735) and avoid bf16 quantization
        # on every flash-attention online-softmax combine. With many prior
        # segments (num_full >= 128), accumulated bf16 quantization error
        # on each segment's o_rem_prev_hbm round-trip causes systematic
        # drift (observed 7–22% rel diff for num_full >= 128).
        rem_softmax_shape = (1, 128, effective_grp_length)
        o_rem_prev_hbm = nl.ndarray(shape=(1, effective_q_tokens, head_dim), dtype=nl.float32, buffer=nl.private_hbm)
        neg_max_rem_prev_hbm = nl.ndarray(shape=rem_softmax_shape, dtype=nl.float32, buffer=nl.private_hbm)
        sum_rem_prev_hbm = nl.ndarray(shape=rem_softmax_shape, dtype=nl.float32, buffer=nl.private_hbm)
        o_rem_curr_hbm = nl.ndarray(shape=(1, effective_q_tokens, head_dim), dtype=nl.float32, buffer=nl.private_hbm)
        neg_max_rem_curr_hbm = nl.ndarray(shape=rem_softmax_shape, dtype=nl.float32, buffer=nl.private_hbm)
        sum_rem_curr_hbm = nl.ndarray(shape=rem_softmax_shape, dtype=nl.float32, buffer=nl.private_hbm)

        # Recompute num_full_prior_segments, partial_prior_tokens, and the
        # associated flags using the SHARED effective_prior_seg_size so both
        # cores see identical values → dynamic loops inside fused_impl iterate
        # the same count.
        num_full_prior_segments_remainder_f32 = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.float32)
        num_full_prior_segments_remainder_i32 = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.tensor_scalar(
            dst=num_full_prior_segments_remainder_f32,
            data=prior_tokens_sbuf,
            op0=nl.multiply,
            operand0=1.0 / effective_prior_seg_size_shared,
        )
        floor_nisa_kernel(
            src_t=num_full_prior_segments_remainder_f32,
            dst_t=num_full_prior_segments_remainder_i32,
            p_size=1,
            f_size=1,
            allocator=allocator,
        )

        # prior_block_offset_remainder = num_full_prior_segments_remainder * num_blocks_per_effective_seg_shared
        prior_block_offset_remainder = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
        nisa.tensor_scalar(
            dst=prior_block_offset_remainder,
            data=num_full_prior_segments_remainder_i32,
            op0=nl.multiply,
            operand0=num_blocks_per_effective_seg_shared,
        )

        # partial_prior_tokens_remainder = prior_tokens - num_full_prior_segments_remainder * effective_prior_seg_size_shared
        temp_seg_tokens_remainder = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.tensor_scalar(
            dst=temp_seg_tokens_remainder,
            data=num_full_prior_segments_remainder_i32,
            op0=nl.multiply,
            operand0=effective_prior_seg_size_shared,
        )
        partial_prior_tokens_remainder = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.tensor_tensor(
            dst=partial_prior_tokens_remainder,
            data1=prior_tokens_sbuf,
            data2=temp_seg_tokens_remainder,
            op=nl.subtract,
        )

        is_partial_prior_segment_remainder = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.tensor_scalar(
            dst=is_partial_prior_segment_remainder,
            data=partial_prior_tokens_remainder,
            op0=nl.greater,
            operand0=0,
        )
        is_not_partial_prior_segment_remainder = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        nisa.tensor_scalar(
            dst=is_not_partial_prior_segment_remainder,
            data=is_partial_prior_segment_remainder,
            op0=nl.subtract,
            operand0=1,
            reverse0=True,
        )

        # Adjust active_block_offset per core: Core 0 uses the primary offset,
        # Core 1 steps past Core 0's active region (size = core0_num_blocks_per_effective_seg).
        # When run_work=False (Core 1's dummy path at num_grps=1), use the
        # primary offset so the KV cache read stays in-bounds; Core 1's
        # output is discarded via the private_hbm scratch redirect below.
        active_block_offset_remainder = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32)
        if shard_id == 0 or not run_work:
            nisa.tensor_copy(dst=active_block_offset_remainder, src=active_block_offset)
        else:
            nisa.tensor_scalar(
                dst=active_block_offset_remainder,
                data=active_block_offset,
                op0=nl.add,
                operand0=core0_num_blocks_per_effective_seg,
            )

        # Derive batch/head indices for remainder batch
        rem_b_i = last_batch // num_q_heads
        rem_h_i = (last_batch % num_q_heads) * num_kv_head // num_q_heads

        # Process this core's portion. Pass the full k/v_cache_sbuf (allocated
        # earlier with max(num_k_tiles_per_seg, num_k_tiles_active) tiles) so
        # fused_impl can address num_k_tiles_per_seg_shared tiles during the
        # prior loop and num_k_tiles_per_effective_seg tiles during the
        # per-shard active segment.
        # fused_segmented_attention_impl handles both KVP and non-KVP via kvp_q_offset parameter.
        fused_segmented_attention_impl(
            q_hbm=q_internal_remainder,
            num_batches=1,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            k_cache_sbuf=k_cache_sbuf,
            v_cache_sbuf=v_cache_sbuf,
            o_prev_hbm=o_rem_prev_hbm,
            neg_max_prev_hbm=neg_max_rem_prev_hbm,
            sum_prev_hbm=sum_rem_prev_hbm,
            o_curr_hbm=o_rem_curr_hbm,
            neg_max_curr_hbm=neg_max_rem_curr_hbm,
            sum_curr_hbm=sum_rem_curr_hbm,
            prior_tokens_sbuf=prior_tokens_sbuf,
            # Prior-driven params use the SHARED effective values so
            # num_full_prior_segments is identical on both cores.
            num_full_prior_segments_i32=num_full_prior_segments_remainder_i32,
            partial_prior_tokens=partial_prior_tokens_remainder,
            is_partial_prior_segment=is_partial_prior_segment_remainder,
            is_not_partial_prior_segment=is_not_partial_prior_segment_remainder,
            active_block_offset=active_block_offset_remainder,
            prior_block_offset=prior_block_offset_remainder,
            allocator=allocator,
            prior_seg_size=effective_prior_seg_size_shared,
            block_size=block_size,
            scale=scale,
            head_dim=head_dim,
            num_grps=effective_grp_length,
            # Active-driven params use per-shard Q length.
            num_active_blocks=num_blocks_per_effective_seg,
            num_k_tiles_active=num_k_tiles_per_effective_seg,
            num_v_tiles_active=num_v_tiles_per_effective_seg,
            # Prior-segment chunk sizing uses the shared effective value.
            num_blocks_per_seg=num_blocks_per_effective_seg_shared,
            num_k_tiles_per_seg=num_k_tiles_per_effective_seg_shared,
            num_v_tiles_per_seg=num_v_tiles_per_effective_seg_shared,
            b_i=rem_b_i,
            h_i=rem_h_i,
            tp_q=tp_q,
            tp_out=tp_out,
            load_kv_cache_fn=load_kv_cache,
            attention_cte_fn=_attention_cte,
            sink=sink[last_batch : last_batch + 1] if sink is not None else None,
            k_pre_transposed=k_pre_transposed,
            fp8_packed=fp8_packed,
            k_scale_sb=k_scale_sb,
            kvp_q_offset=kvp_offset_sbuf,
            kvp_rank_id=kvp_rank_id,
            kvp_group_size=kvp_group_size,
            sliding_window=sliding_window if sliding_window != None else 0,
            kvp_cp_offset_int=kvp_cp_offset_int,
            kvp_seg_block_offset_int=kvp_seg_block_offset_int,
            kvp_prior_load_blocks=kvp_prior_load_blocks,
            kvp_prior_fully_visible=kvp_prior_fully_visible,
        )

        # Core 1 does one extra iteration to process Core 0's active segment.
        # Core 0's active region covers active_block_offset .. active_block_offset +
        # core0_num_blocks_per_effective_seg blocks (the first core0_q_tokens tokens
        # of active K). Core 1 needs to attend to these positions (they're
        # causally before Core 1's own Q) — fused_impl's active segment only
        # covered Core 1's own slice (blocks starting at active_block_offset +
        # core0_num_blocks_per_effective_seg).
        if shard_id == 1:
            init_sbuf_addr = allocator.get_current_address()
            # Block offset = active_block_offset (start of all active blocks);
            # load_kv_cache will load core0_num_blocks_per_effective_seg blocks
            # starting from there, which is exactly Core 0's active region.
            extra_offset = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
            nisa.tensor_copy(dst=extra_offset, src=active_block_offset)

            # Zero ALL K/V tiles that the extra iteration will read.
            # k_cache_sbuf / v_cache_sbuf still hold data from the prior-segment
            # loop. load_kv_cache writes core0_num_blocks_per_effective_seg blocks
            # starting at tile 0 — when this doesn't fill all tiles (e.g. 1 block
            # into a 4-block K tile), the unwritten portions remain stale.
            # Zero all consumed tiles so any unwritten tail is 0, not NaN/stale.
            for k_idx in range(core0_num_k_tiles_per_effective_seg):
                for d_idx in range(len(k_cache_sbuf[0])):
                    nisa.memset(k_cache_sbuf[k_idx][d_idx][...], value=0.0)
            for v_idx in range(core0_num_v_tiles_per_effective_seg):
                for d_idx in range(len(v_cache_sbuf[0])):
                    nisa.memset(v_cache_sbuf[v_idx][d_idx][...], value=0.0)

            # Load Core 0's active KV segment.
            load_kv_cache(
                k_cache,
                v_cache,
                block_tables,
                k_cache_sbuf,
                v_cache_sbuf,
                rem_b_i,
                rem_h_i,
                extra_offset,
                core0_num_blocks_per_effective_seg,
                allocator,
                k_pre_transposed=k_pre_transposed,
                fp8_packed=fp8_packed,
            )
            allocator.set_current_address(init_sbuf_addr)

            extra_kv_used_len = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.float32)
            nisa.memset(extra_kv_used_len, value=float(core0_q_tokens))
            inner_init_sbuf_addr = allocator.get_current_address()

            # Compute attention for this extra segment
            _attention_cte(
                q_internal_remainder,
                None,
                None,
                scale=scale,
                causal_mask=False,
                tp_q=tp_q,
                tp_k=False,
                tp_out=False,
                cache_softmax=True,
                skip_output_normalization=True,
                k_cache_sbuf=k_cache_sbuf[:core0_num_k_tiles_per_effective_seg],
                v_cache_sbuf=v_cache_sbuf[:core0_num_v_tiles_per_effective_seg],
                kv_used_len=extra_kv_used_len,
                out_o_hbm=o_rem_curr_hbm,
                out_neg_max_hbm=neg_max_rem_curr_hbm,
                out_sum_hbm=sum_rem_curr_hbm,
                init_sbuf_addr=inner_init_sbuf_addr,
                k_scale_sb=k_scale_sb,
            )
            allocator.set_current_address(init_sbuf_addr)

            # HBM-based reduction: combine extra segment into accumulated results.
            # Uses effective_grp_length so Core 1's dummy path (num_grps=1) has
            # valid non-zero shapes/iteration counts.
            rem_sb_p = nl.tile_size.pmax
            rem_softmax_pat = [[effective_grp_length, rem_sb_p], [1, effective_grp_length]]
            rem_o_pat = [[head_dim, rem_sb_p], [1, head_dim]]
            rem_num_free = min(effective_grp_length, _MAX_FREE_TILES)

            rem_neg_max_prev_sb = allocator.alloc_sbuf_tensor(shape=(rem_sb_p, effective_grp_length), dtype=nl.float32)
            rem_sum_prev_sb = allocator.alloc_sbuf_tensor(shape=(rem_sb_p, effective_grp_length), dtype=nl.float32)
            rem_neg_max_curr_sb = allocator.alloc_sbuf_tensor(shape=(rem_sb_p, effective_grp_length), dtype=nl.float32)
            rem_sum_curr_sb = allocator.alloc_sbuf_tensor(shape=(rem_sb_p, effective_grp_length), dtype=nl.float32)
            rem_o_prev_sb = allocator.alloc_sbuf_tensor(
                shape=(rem_sb_p, head_dim),
                dtype=nl.float32,
                block_dim=[effective_grp_length],
                num_free_tiles=[rem_num_free],
            )
            rem_o_curr_sb = allocator.alloc_sbuf_tensor(
                shape=(rem_sb_p, head_dim),
                dtype=nl.float32,
                block_dim=[effective_grp_length],
                num_free_tiles=[rem_num_free],
            )
            rem_o_new_sb = allocator.alloc_sbuf_tensor(
                shape=(rem_sb_p, head_dim),
                dtype=nl.float32,
                block_dim=[effective_grp_length],
                num_free_tiles=[rem_num_free],
            )
            rem_batch_loop_addr = allocator.get_current_address()

            reduce_one_batch(
                o_rem_prev_hbm,
                neg_max_rem_prev_hbm,
                sum_rem_prev_hbm,
                o_rem_curr_hbm,
                neg_max_rem_curr_hbm,
                sum_rem_curr_hbm,
                0,
                0,
                effective_grp_length,
                head_dim,
                effective_grp_length,
                rem_sb_p,
                rem_softmax_pat,
                rem_o_pat,
                rem_neg_max_prev_sb,
                rem_sum_prev_sb,
                rem_neg_max_curr_sb,
                rem_sum_curr_sb,
                rem_o_prev_sb,
                rem_o_curr_sb,
                rem_o_new_sb,
                rem_batch_loop_addr,
                allocator,
            )
            allocator.set_current_address(init_sbuf_addr)
        else:
            # Core 0 does dummy ops for control flow consistency
            rng_seeds_sb = allocator.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.uint32)
            nisa.memset(rng_seeds_sb, 0.0)
            nisa.set_rng_seed(rng_seeds_sb)

        # Normalize and write each core's portion to result[last_batch].
        # Core 1's dummy path (run_work=False, i.e. num_grps=1) redirects
        # the result write to a private_hbm scratch tensor so it doesn't
        # corrupt the shared `result`. Both cores trace the same static
        # Python for-loop so the IR stays symmetric for LNC2.
        rem_norm_addr = allocator.get_current_address()
        rem_sb_p2 = nl.tile_size.pmax
        rem_sm_pat = [[effective_grp_length, rem_sb_p2], [1, effective_grp_length]]
        rem_o_tile_pat = [[head_dim, rem_sb_p2], [1, head_dim]]

        # Scratch result buffer for Core 1's dummy path. Core 0 points to
        # the shared `result`; Core 1 points to a fresh private_hbm with
        # matching shape so the DMA writes don't corrupt shared state.
        if run_work:
            result_write_target = result
        else:
            result_write_target = nl.ndarray(shape=result.shape, dtype=result.dtype, buffer=nl.private_hbm)

        rem_sum_sb2 = allocator.alloc_sbuf_tensor(shape=(rem_sb_p2, effective_grp_length), dtype=nl.float32)
        nisa.dma_copy(dst=rem_sum_sb2, src=sum_rem_prev_hbm.ap(pattern=rem_sm_pat, offset=0))
        rem_sum_recip_sb2 = allocator.alloc_sbuf_tensor(shape=(rem_sb_p2, effective_grp_length), dtype=nl.float32)
        nisa.reciprocal(rem_sum_recip_sb2, rem_sum_sb2)

        rem_num_free2 = min(effective_grp_length, _MAX_FREE_TILES)
        rem_o_sb2 = allocator.alloc_sbuf_tensor(
            shape=(rem_sb_p2, d_tile_size_par_dim),
            dtype=nl.bfloat16,
            block_dim=[effective_grp_length, num_d_tiles],
            num_free_tiles=[rem_num_free2, num_d_tiles],
        )
        if tp_out:
            rem_o_tp_psum2 = nl.ndarray(
                (d_tile_size_par_dim, rem_sb_p2), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0)
            )
            rem_o_tp_sb2 = allocator.alloc_sbuf_tensor(shape=(d_tile_size_par_dim, rem_sb_p2), dtype=nl.bfloat16)
        for local_grp_i in range(effective_grp_length):
            global_grp_i = effective_grp_start + local_grp_i
            grp_start_pos = global_grp_i * 128

            for d_idx in range(num_d_tiles):
                d_offset = d_idx * d_tile_size_par_dim
                nisa.dma_copy(
                    dst=rem_o_sb2[local_grp_i][d_idx],
                    src=o_rem_prev_hbm.ap(
                        pattern=[[head_dim, rem_sb_p2], [1, d_tile_size_par_dim]],
                        offset=local_grp_i * rem_sb_p2 * head_dim + d_offset,
                    ),
                )
                # Delayed V dequant: fold v_scale multiply into normalization tensor_scalar
                if v_scale_sb is not None:
                    nisa.tensor_scalar(
                        dst=rem_o_sb2[local_grp_i][d_idx],
                        data=rem_o_sb2[local_grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=rem_sum_recip_sb2[:, local_grp_i],
                        op1=nl.multiply,
                        operand1=v_scale_sb,
                    )
                else:
                    nisa.tensor_scalar(
                        dst=rem_o_sb2[local_grp_i][d_idx],
                        data=rem_o_sb2[local_grp_i][d_idx],
                        op0=nl.multiply,
                        operand0=rem_sum_recip_sb2[:, local_grp_i],
                    )
                # Write to result[last_batch] at global group position (or scratch
                # on Core 1's dummy path). For Core 1's dummy path, grp_start=0
                # and local_grp_i=0, so grp_start_pos=0 — writes to the scratch
                # buffer's beginning, which is safe.
                if tp_out:
                    nisa.nc_transpose(dst=rem_o_tp_psum2, data=rem_o_sb2[local_grp_i][d_idx])
                    nisa.tensor_copy(dst=rem_o_tp_sb2, src=rem_o_tp_psum2)
                    nisa.dma_copy(
                        dst=result_write_target.ap(
                            pattern=[[seqlen_q, d_tile_size_par_dim], [1, rem_sb_p2]],
                            offset=last_batch * head_dim * seqlen_q + d_offset * seqlen_q + grp_start_pos,
                        ),
                        src=rem_o_tp_sb2,
                    )
                else:
                    nisa.dma_copy(
                        dst=result_write_target.ap(
                            pattern=[[head_dim, rem_sb_p2], [1, d_tile_size_par_dim]],
                            offset=last_batch * num_grps * rem_sb_p2 * head_dim
                            + global_grp_i * rem_sb_p2 * head_dim
                            + d_offset,
                        ),
                        src=rem_o_sb2[local_grp_i][d_idx],
                    )
        allocator.set_current_address(rem_norm_addr)

    if kvp_q_offset is not None:
        return result, out_neg_max_hbm, out_sum_recip_hbm
    return result
