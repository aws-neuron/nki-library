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
"""KV-parallel segmented prefill attention kernel.

This kernel enables context parallelism by distributing the KV cache across multiple
ranks. Each rank computes attention over its local KV shard, then results are merged
using online softmax.

See README.md for detailed documentation.
"""

from typing import Optional

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..utils.modular_allocator import ModularAllocator
from .attention_segmented_cte import attention_segmented_cte

P_MAX = nl.tile_size.pmax


@nki.jit
def attention_kv_parallel_segmented_cte(
    q: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    v_cache: nl.NkiTensor,
    block_tables: nl.NkiTensor,
    kvp_q_offset: nl.NkiTensor,
    replica_groups: ReplicaGroup,
    group_size: int,
    block_size: int,
    seg_size: int,
    scale: float = 1.0,
    global_q_offset: int = 0,
    tp_out: bool = False,
    sliding_window: int = 0,
    kvp_rank_id: Optional[nl.NkiTensor] = None,
    kvp_group_size: int = 0,
    apc_mode: bool = False,
    valid_num_prior_tokens: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """
    KV-parallel segmented prefill attention.

    Distributes attention computation across ranks, where each rank holds a shard
    of the KV cache. Uses online softmax to merge partial results.

    Dimensions:
        q_heads_per_rank: Q heads per physical rank (= q.shape[0])
        S: Sequence length
        D: Head dimension
        G: Group size (number of logical ranks in the replica group = len(replica_groups))
        total_heads: q_heads_per_rank * group_size (total Q heads after all-gather)

    Args:
        q (nl.NkiTensor): [q_heads_per_rank, S, D], This rank's Q heads.
        k_cache (nl.NkiTensor): [num_blocks, num_kv_heads, block_size, D], Local KV cache (K).
        v_cache (nl.NkiTensor): [num_blocks, num_kv_heads, block_size, D], Local KV cache (V).
        block_tables (nl.NkiTensor): [1, max_blocks] int32, Block indices for paged KV.
        kvp_q_offset (nl.NkiTensor): [1, 1] int32, Causal mask offset.
            For contiguous KV sharding: -rank_id * local_kv_len + global_q_offset.
            For round-robin KV distribution: the actual global Q position (runtime),
            since the kernel uses kvp_rank_id to handle K-side positioning internally.
        replica_groups (ReplicaGroup): ReplicaGroup for collective operations.
        group_size (int): Number of logical ranks in the replica group (= len(replica_groups)).
        block_size (int): KV cache block size.
        seg_size (int): Segment size for attention iteration.
        scale (float): Attention scale factor (default 1.0).
        global_q_offset (int): Compile-time global position of Q token 0 (default 0).
            In non-APC mode, this is required for correctness — it determines how many prior
            KV segments are loaded per chunk. In APC mode, set to 0 since the prior token
            count is derived at runtime from kvp_q_offset instead.
        tp_out (bool): If True, output is transposed to [q_heads_per_rank, D, S] (default False).
        sliding_window (int): Sliding window size for attention (0 = disabled).
        kvp_rank_id (nl.NkiTensor): [1, 1] int32, This rank's index within the KV-parallel group.
            Required for interleaved (round-robin) KV distribution to convert global K
            positions to segment-local positions.
        kvp_group_size (int): Number of ranks sharing the KV cache in round-robin fashion.
            When > 0, enables interleaved KV mode where rank r holds global blocks
            r, r+R, r+2R, ... (R = kvp_group_size).
        apc_mode (bool): Automated prefix caching mode (default False). When True,
            prior token count is taken from valid_num_prior_tokens (required).
            Compile-time tile-visibility optimizations are disabled since global_q_offset
            may not reflect the actual prefix length. Set global_q_offset=0.
        valid_num_prior_tokens (nl.NkiTensor): [1, 1] int32. Required when apc_mode is True.
            The number of fully-visible local prior tokens for Q chunk 0 on this rank.
            Must be a multiple of block_size. The kernel increments this per chunk by
            seg_size // kvp_group_size as later chunks see more local KV.

    Returns:
        out (nl.NkiTensor): [q_heads_per_rank, S, D], Merged attention output for this rank's Q heads.

    Pseudocode:
        # Step 1: All-gather Q across ranks
        q_full = all_gather(q)  # [total_heads, S, D]

        # Step 2: For each Q chunk, compute local attention
        for q_chunk_idx in range(num_q_chunks):
            q_chunk = q_full[heads_for_this_nc, q_start:q_end, :]
            chunk_out, chunk_neg_max, chunk_sum_recip = attention_segmented_cte(q_chunk, k_cache, v_cache)
            partial_out[heads_for_this_nc, q_start:q_end] = chunk_out

        # Step 3: Pack softmax stats + partial outputs, exchange via all-to-all
        send_packed = pack(partial_out, neg_max, sum_recip)
        recv_packed = all_to_all(send_packed)

        # Step 4: Merge partials using online softmax
        for head_idx in range(q_heads_per_rank // lnc_degree):
            for tile_idx in range(num_tiles):
                global_neg_max = min(recv_packed[:, neg_max_channel])
                factors = exp(global_neg_max - neg_max_per_rank) / sum_recip_per_rank
                factors = factors / sum(factors)
                out[head_idx, tile] = sum(factors * recv_packed[:, out_channel])
    """
    q_heads_per_rank, seq_len, head_dim = q.shape
    num_q_chunks = seq_len // seg_size

    shard_id = nl.program_id(0)
    lnc_degree = nl.num_programs(0)
    total_heads = q_heads_per_rank * group_size
    heads_per_nc = total_heads // lnc_degree

    # Input validation
    kernel_assert(seq_len % seg_size == 0, f"seq_len ({seq_len}) must be divisible by seg_size ({seg_size})")
    kernel_assert(
        total_heads % lnc_degree == 0,
        f"total_heads ({q_heads_per_rank} * {group_size}) must be divisible by lnc_degree",
    )
    kernel_assert(
        q.shape[2] == k_cache.shape[3],
        f"head_dim mismatch: q has {q.shape[2]}, k_cache has {k_cache.shape[3]}",
    )
    kernel_assert(
        k_cache.shape[2] == block_size,
        f"k_cache block_size dim ({k_cache.shape[2]}) must match block_size ({block_size})",
    )

    # All-gather Q across ranks.
    # Collectives cannot read/write I/O tensors directly, so each NC DMAs its slice into shared_hbm first.
    # Parallelize the copy: each NC copies its chunk, with remainder distributed round-robin.
    q_src = nl.ndarray((q_heads_per_rank, seq_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm, name="q_src")
    heads_per_nc_copy = q_heads_per_rank // lnc_degree
    leftover_heads = q_heads_per_rank % lnc_degree
    for nc_idx in range(lnc_degree):
        copy_start = nc_idx * heads_per_nc_copy + min(nc_idx, leftover_heads)
        copy_end = copy_start + heads_per_nc_copy + (1 if nc_idx < leftover_heads else 0)
        if copy_end > copy_start and shard_id == nc_idx:
            nisa.dma_copy(dst=q_src[copy_start:copy_end, :, :], src=q[copy_start:copy_end, :, :])

    q_full = nl.ndarray(
        (total_heads, seq_len, head_dim),
        dtype=q.dtype,
        buffer=nl.shared_hbm,
        name="q_full",
    )
    ncc.all_gather(dsts=[q_full], srcs=[q_src], replica_group=replica_groups, collective_dim=0)

    partial_out = nl.ndarray(
        (total_heads, seq_len, head_dim), dtype=nl.float32, buffer=nl.shared_hbm, name="partial_out"
    )
    neg_max = nl.ndarray((total_heads, seq_len), dtype=nl.float32, buffer=nl.shared_hbm, name="neg_max")
    sum_recip = nl.ndarray((total_heads, seq_len), dtype=nl.float32, buffer=nl.shared_hbm, name="sum_recip")

    chunk_allocator = ModularAllocator()
    kvp_offset_chunk_sbuf = chunk_allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    kvp_offset_chunk_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="kvp_offset_chunk_hbm")
    prior_tokens_chunk_sbuf = chunk_allocator.alloc_sbuf_tensor((1, 1), nl.int32)
    prior_tokens_chunk_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.private_hbm, name="prior_tokens_chunk_hbm")
    if apc_mode:
        kernel_assert(valid_num_prior_tokens is not None, "valid_num_prior_tokens is required when apc_mode=True")

    num_kv_blocks = k_cache.shape[0]
    local_kv_len = num_kv_blocks * block_size
    max_prior_tokens = local_kv_len - seg_size

    # Compute local attention for each Q chunk.
    # Each NC processes heads_per_nc Q heads starting at shard_id * heads_per_nc.
    for q_chunk_idx in range(num_q_chunks):
        q_start = q_chunk_idx * seg_size
        q_end = q_start + seg_size

        q_chunk = nl.ndarray(
            (heads_per_nc, seg_size, head_dim), dtype=q.dtype, buffer=nl.private_hbm, name=f"q_chunk_{q_chunk_idx}"
        )
        nisa.dma_copy(
            dst=q_chunk,
            src=q_full[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end, :],
        )

        prior_tokens_for_chunk = min(q_start + global_q_offset, max_prior_tokens)

        # For interleaved KV: set active_block_offset so the active segment covers all
        # local blocks that Q can see. Prior segments become fully visible (no masking).
        kvp_prior_load_blocks = 0  # 0 = use default (num_blocks_per_seg)
        # APC: prior tokens are guaranteed fully visible by contract (FAL passes correct
        # valid_num_prior_tokens), so set prior_fully_visible=True to use the causal_mask=False
        # path in the inner kernel and avoid double-counting with the active segment.
        prior_fully_visible = apc_mode and kvp_group_size > 0
        if kvp_group_size > 0 and not apc_mode:
            stride = group_size * block_size
            kvp_cp_offset_int = q_start + global_q_offset
            num_blocks_per_seg = seg_size // block_size
            num_active_blocks = seg_size // block_size
            blocks_per_k_tile = 512 // block_size  # _K_TILE_SZ // block_size
            # Last local block Q can see (any rank)
            last_needed_block = (kvp_cp_offset_int + seg_size - 1) // stride
            # Active must contain last_needed_block
            min_active_start = max(0, last_needed_block - num_active_blocks + 1)
            # Round UP to K-tile boundary so partial prior fills complete tiles
            active_block_offset_int = (
                (min_active_start + blocks_per_k_tile - 1) // blocks_per_k_tile
            ) * blocks_per_k_tile
            # Ensure at least 1 full prior segment (required for reduce_one_batch path)
            active_block_offset_int = max(active_block_offset_int, num_blocks_per_seg)
            # Verify prior is fully visible: max global in prior < kvp_cp_offset_int
            # max_global_in_prior = active_block_offset_int * stride - 1
            prior_fully_visible = (active_block_offset_int * stride - 1) < kvp_cp_offset_int
            # Verify active covers all needed blocks
            covers_needed = (active_block_offset_int + num_active_blocks - 1) >= last_needed_block
            # Only apply if both constraints are met and it reduces prior_tokens
            if prior_fully_visible and covers_needed and active_block_offset_int < prior_tokens_for_chunk // block_size:
                # Compute partial prior load count
                num_full_prior_segs = active_block_offset_int // num_blocks_per_seg
                partial_blocks = active_block_offset_int - num_full_prior_segs * num_blocks_per_seg
                if partial_blocks > 0:
                    kvp_prior_load_blocks = partial_blocks
                prior_tokens_for_chunk = active_block_offset_int * block_size

        nisa.dma_copy(dst=kvp_offset_chunk_sbuf, src=kvp_q_offset)
        nisa.tensor_scalar(dst=kvp_offset_chunk_sbuf, data=kvp_offset_chunk_sbuf, op0=nl.add, operand0=q_start)
        nisa.dma_copy(dst=kvp_offset_chunk_hbm, src=kvp_offset_chunk_sbuf)
        if kvp_group_size > 0:
            if apc_mode:
                # APC: prior_tokens grows per chunk as Q advances. Each chunk sees
                # seg_size // kvp_group_size more local prior tokens than the previous.
                apc_prior_increment = q_chunk_idx * (seg_size // kvp_group_size)
                nisa.dma_copy(dst=prior_tokens_chunk_sbuf, src=valid_num_prior_tokens)
                nisa.tensor_scalar(
                    dst=prior_tokens_chunk_sbuf, data=prior_tokens_chunk_sbuf, op0=nl.add, operand0=apc_prior_increment
                )
            else:
                nisa.tensor_scalar(
                    dst=prior_tokens_chunk_sbuf,
                    data=kvp_offset_chunk_sbuf,
                    op0=nl.minimum,
                    operand0=prior_tokens_for_chunk,
                )
        else:
            nisa.memset(prior_tokens_chunk_sbuf[...], value=prior_tokens_for_chunk)
        nisa.dma_copy(dst=prior_tokens_chunk_hbm, src=prior_tokens_chunk_sbuf)

        # APC: compile-time offset unknown, disable tile-visibility optimization
        chunk_kvp_cp_offset_int = 0 if apc_mode else q_start + global_q_offset
        chunk_kvp_seg_block_offset_int = 0 if apc_mode else prior_tokens_for_chunk // block_size

        chunk_out, chunk_neg_max, chunk_sum_recip = attention_segmented_cte(
            q=q_chunk,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens_chunk_hbm,
            block_size=block_size,
            prior_seg_size=seg_size,
            scale=scale,
            tp_q=True,
            tp_out=False,  # Always False internally; tp_out handled at final output write
            kvp_q_offset=kvp_offset_chunk_hbm,
            kvp_rank_id=kvp_rank_id,
            kvp_group_size=kvp_group_size,
            sliding_window=sliding_window,
            kvp_cp_offset_int=chunk_kvp_cp_offset_int,
            kvp_seg_block_offset_int=chunk_kvp_seg_block_offset_int,
            kvp_prior_load_blocks=kvp_prior_load_blocks,
            kvp_prior_fully_visible=prior_fully_visible,
        )

        nisa.dma_copy(
            dst=partial_out[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end, :],
            src=chunk_out,
        )
        nisa.dma_copy(
            dst=neg_max[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end],
            src=chunk_neg_max,
        )
        nisa.dma_copy(
            dst=sum_recip[nl.ds(shard_id * heads_per_nc, heads_per_nc), q_start:q_end],
            src=chunk_sum_recip,
        )
    # Exchange partial outputs and softmax stats via coalesced all-to-all (3 separate tensors).
    recv_out = nl.ndarray(
        (group_size, q_heads_per_rank, seq_len, head_dim),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_out",
    )
    recv_neg_max = nl.ndarray(
        (group_size, q_heads_per_rank, seq_len),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_neg_max",
    )
    recv_sum_recip = nl.ndarray(
        (group_size, q_heads_per_rank, seq_len),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
        name="recv_sum_recip",
    )
    ncc.all_to_all(
        dsts=[recv_out, recv_neg_max, recv_sum_recip],
        srcs=[partial_out, neg_max, sum_recip],
        replica_group=replica_groups,
        collective_dim=0,
    )

    # Merge partial attention outputs using online softmax.
    allocator = ModularAllocator()
    out = nl.ndarray(
        (q_heads_per_rank, head_dim, seq_len) if tp_out else (q_heads_per_rank, seq_len, head_dim),
        dtype=q.dtype,
        buffer=nl.shared_hbm,
        name="out",
    )
    _merge_partial_attention_outputs(
        recv_out,
        recv_neg_max,
        recv_sum_recip,
        out,
        shard_id,
        group_size,
        q_heads_per_rank,
        lnc_degree,
        seq_len,
        head_dim,
        q.dtype,
        allocator,
        tp_out,
    )
    return out


def _merge_partial_attention_outputs(
    recv_out: nl.NkiTensor,
    recv_neg_max: nl.NkiTensor,
    recv_sum_recip: nl.NkiTensor,
    out: nl.NkiTensor,
    shard_id: int,
    group_size: int,
    q_heads_per_rank: int,
    lnc_degree: int,
    seq_len: int,
    head_dim: int,
    out_dtype,
    allocator: ModularAllocator,
    tp_out: bool = False,
) -> None:
    """
    Merge partial attention outputs from all ranks using online softmax rescaling.

    recv_out: [group_size, q_heads_per_rank, seq_len, head_dim]
    recv_neg_max: [group_size, q_heads_per_rank, seq_len]
    recv_sum_recip: [group_size, q_heads_per_rank, seq_len]

    Each NC merges its share of q_heads_per_rank heads.
    """
    local_heads_per_nc = div_ceil(q_heads_per_rank, lnc_degree)
    neg_max_sbuf = allocator.alloc_sbuf_tensor((P_MAX, group_size), nl.float32)
    sum_recip_sbuf = allocator.alloc_sbuf_tensor((P_MAX, group_size), nl.float32)
    global_neg_max = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    factors = allocator.alloc_sbuf_tensor((P_MAX, group_size), nl.float32)
    exp_term = allocator.alloc_sbuf_tensor((P_MAX, group_size), nl.float32)
    recip = allocator.alloc_sbuf_tensor((P_MAX, group_size), nl.float32)
    factor_sum = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    factor_sum_recip = allocator.alloc_sbuf_tensor((P_MAX, 1), nl.float32)
    out_tile = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    partial_tile = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    scaled = allocator.alloc_sbuf_tensor((P_MAX, head_dim), nl.float32)
    out_tile_cast = allocator.alloc_sbuf_tensor((P_MAX, head_dim), out_dtype)
    # tp_out: extra SBUF buffers for transposing merged tile (P_MAX, head_dim) → (head_dim, P_MAX).
    out_tile_tp_psum = nl.ndarray((head_dim, P_MAX), dtype=nl.float32, buffer=nl.psum) if tp_out else None
    out_tile_tp_sbuf = allocator.alloc_sbuf_tensor((head_dim, P_MAX), out_dtype) if tp_out else None

    num_tiles = div_ceil(seq_len, P_MAX)
    for head_idx in range(local_heads_per_nc):
        pos = shard_id * local_heads_per_nc + head_idx
        if pos >= q_heads_per_rank:
            break

        for tile_idx in range(num_tiles):
            tile_start = tile_idx * P_MAX
            tile_end = min(tile_start + P_MAX, seq_len)
            tile_size = tile_end - tile_start

            # Load neg_max and sum_recip for all ranks — contiguous layout.
            # recv_neg_max: [group_size, q_heads_per_rank, seq_len]
            # Want: neg_max_sbuf[tile, rank] — partition=tile (P_MAX), free=rank
            stats_p_stride = 1  # consecutive tokens
            stats_f_stride = q_heads_per_rank * seq_len  # stride between ranks
            stats_offset = pos * seq_len + tile_start
            stats_ap = [[stats_p_stride, P_MAX], [stats_f_stride, group_size]]
            nisa.dma_copy(dst=neg_max_sbuf, src=recv_neg_max.ap(pattern=stats_ap, offset=stats_offset))
            nisa.dma_copy(dst=sum_recip_sbuf, src=recv_sum_recip.ap(pattern=stats_ap, offset=stats_offset))

            # Compute per-rank rescaling factors using online softmax, then normalize.
            nisa.tensor_reduce(
                dst=global_neg_max[:tile_size, :],
                op=nl.minimum,
                data=neg_max_sbuf[:tile_size, :],
                axis=[1],
                keepdims=True,
            )
            nisa.activation(
                dst=exp_term[:tile_size, :],
                op=nl.exp,
                data=neg_max_sbuf[:tile_size, :],
                scale=-1.0,
                bias=global_neg_max[:tile_size, :],
            )
            nisa.activation(dst=recip[:tile_size, :], op=nl.reciprocal, data=sum_recip_sbuf[:tile_size, :])
            nisa.tensor_tensor(
                dst=factors[:tile_size, :], data1=exp_term[:tile_size, :], data2=recip[:tile_size, :], op=nl.multiply
            )
            nisa.tensor_reduce(
                dst=factor_sum[:tile_size, :], op=nl.add, data=factors[:tile_size, :], axis=[1], keepdims=True
            )
            nisa.activation(dst=factor_sum_recip[:tile_size, :], op=nl.reciprocal, data=factor_sum[:tile_size, :])
            nisa.activation(
                dst=factors[:tile_size, :],
                op=nl.copy,
                data=factors[:tile_size, :],
                scale=factor_sum_recip[:tile_size, :],
            )

            # Weighted sum: out = sum_r(factors[:, r] * partial[:, r, :]).
            nisa.memset(out_tile, 0)
            for rank_idx in range(group_size):
                nisa.dma_copy(dst=partial_tile[:tile_size, :], src=recv_out[rank_idx, pos, tile_start:tile_end, :])
                nisa.tensor_scalar(
                    dst=scaled[:tile_size, :],
                    data=partial_tile[:tile_size, :],
                    op0=nl.multiply,
                    operand0=factors[:tile_size, rank_idx : rank_idx + 1],
                )
                nisa.tensor_tensor(
                    dst=out_tile[:tile_size, :],
                    data1=out_tile[:tile_size, :],
                    data2=scaled[:tile_size, :],
                    op=nl.add,
                )

            nisa.tensor_copy(dst=out_tile_cast[:tile_size, :], src=out_tile[:tile_size, :])
            if tp_out:
                nisa.nc_transpose(out_tile_tp_psum[:, :tile_size], out_tile[:tile_size, :])
                nisa.tensor_copy(dst=out_tile_tp_sbuf[:, :tile_size], src=out_tile_tp_psum[:, :tile_size])
                nisa.dma_copy(dst=out[pos, :, tile_start:tile_end], src=out_tile_tp_sbuf[:, :tile_size])
            else:
                nisa.dma_copy(dst=out[pos, tile_start:tile_end, :], src=out_tile_cast[:tile_size, :])
