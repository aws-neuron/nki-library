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

"""DeepSeek-V3.2 MLA decode QKV front-fold: Q and KV projections, RoPE, absorb, and cache scatter."""

from typing import NamedTuple

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.entry_trace import trace_kernel_entry
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ....core.utils.tiled_range import TiledRange
from . import F_MAX, P_MAX, MlaDecodeQkvShape, get_mla_decode_qkv_shape


class MlaDecodeQkvResult(NamedTuple):
    """What the front-fold hands the attention core: the two Q tiles stay in SBUF."""

    q_absorbed_sbuf: nl.NkiTensor  # [P_MAX, n_kv_tiles, n_tokens, n_heads]  feature-major
    q_pe_sbuf: nl.NkiTensor  # [P_MAX, n_tokens, n_heads]              feature-major
    kv_row_cur_hbm: nl.NkiTensor  # [n_tokens, kv_row_dim]                  key-major cache row


@nki.jit
def mla_decode_qkv(
    hidden_states: nl.NkiTensor,  # [n_tokens, hidden]
    wq_a: nl.NkiTensor,  # [hidden, q_lora_rank]                       query down-projection
    q_norm_w: nl.NkiTensor,  # [1, q_lora_rank]                            RMSNorm gamma, fp32
    wq_b: nl.NkiTensor,  # [q_lora_rank, qk_packed_dim]                query up-projection
    wkv_a: nl.NkiTensor,  # [hidden, kv_row_dim]                        KV down-projection
    kv_norm_w: nl.NkiTensor,  # [1, kv_lora_rank]                           RMSNorm gamma, fp32
    cos: nl.NkiTensor,  # [n_tokens, qk_rope_dim // 2]                interleaved RoPE table
    sin: nl.NkiTensor,  # [n_tokens, qk_rope_dim // 2]
    q_absorb_w: nl.NkiTensor,  # [n_heads, qk_nope_dim, kv_lora_rank]        wkv_b nope half
    kv_cache: nl.NkiTensor,  # [n_blocks, 1, block_size, kv_row_dim]       paged, mutated in place
    slot_mapping: nl.NkiTensor,  # [n_tokens] int32                            destination row per token
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    norm_eps: float = 1e-6,
    input_norm_w: nl.NkiTensor | None = None,  # [1, hidden] fp32, fold the layer's input RMSNorm
) -> MlaDecodeQkvResult:
    """
    Computes the DeepSeek-V3.2 MLA query and key/value projections for one decode step, applies
    interleaved RoPE, absorbs the query into the cache's latent space, and scatters this step's KV
    row into the paged cache.

    This is the front-fold of an MLA decode layer. It is designed to be traced inline into an
    attention core rather than run standalone: the two query tiles are returned **in SBUF**, already
    feature-major in the layout the core's score matmuls want. Returning them through HBM was
    measured as pure latency, an 18 KB round trip into an identical SBUF layout. A standalone caller
    must therefore evict them itself.

    Both RMSNorms are fused into the projections that feed them, and the KV row is built directly in
    cache-row order so that no transpose is needed before the scatter.

    Dimensions:
        n_tokens: Decode tokens in the batch, one per sequence. Must be <= P_MAX.
        hidden: Model hidden width, the contraction dimension of both down-projections.
        q_lora_rank: LoRA rank of the query path, derived from ``wq_a``.
        kv_lora_rank: LoRA rank of the KV path. Must be a multiple of P_MAX.
        n_heads: Attention heads.
        qk_nope_dim: Per-head QK width that RoPE does not rotate. Must be <= P_MAX.
        qk_rope_dim: Per-head QK width that RoPE rotates. Must be even.
        kv_row_dim: One cache row, ``qk_rope_dim + kv_lora_rank``.
        n_blocks, block_size: Paged KV cache geometry.

    Args:
        hidden_states: ``[n_tokens, hidden]``. This step's hidden states.
        wq_a: ``[hidden, q_lora_rank]``. Query down-projection.
        q_norm_w: ``[1, q_lora_rank]`` fp32. RMSNorm gamma for the query latent.
        wq_b: ``[q_lora_rank, qk_packed_dim]``. Query up-projection, emitting every head's
            ``[q_nope | q_pe]`` concatenated.
        wkv_a: ``[hidden, kv_row_dim]``. KV down-projection, emitting one cache row per token.
        kv_norm_w: ``[1, kv_lora_rank]`` fp32. RMSNorm gamma for the KV latent.
        cos: ``[n_tokens, qk_rope_dim // 2]``. Cosine table for interleaved RoPE.
        sin: ``[n_tokens, qk_rope_dim // 2]``. Sine table for interleaved RoPE.
        q_absorb_w: ``[n_heads, qk_nope_dim, kv_lora_rank]``. The nope half of ``wkv_b``, which maps
            a head's nope query into the cache's latent space.
        kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Paged cache, **mutated in place**.
        slot_mapping: ``[n_tokens]`` int32. Destination cache row for each token.
        n_heads: Attention heads.
        qk_nope_head_dim: Per-head QK width RoPE does not rotate.
        qk_rope_head_dim: Per-head QK width RoPE rotates.
        kv_lora_rank: LoRA rank of the KV path.
        norm_eps: Epsilon for every RMSNorm in this kernel.
        input_norm_w: ``[1, hidden]`` fp32, optional. When given, the layer's input RMSNorm is folded
            into the hidden load.

    Returns:
        MlaDecodeQkvResult with ``q_absorbed_sbuf`` ``[P_MAX, n_kv_tiles, n_tokens, n_heads]`` and
        ``q_pe_sbuf`` ``[P_MAX, n_tokens, n_heads]``, both SBUF-resident and feature-major, plus
        ``kv_row_cur_hbm`` ``[n_tokens, kv_row_dim]``, this step's cache row published key-major for
        the core's key contraction.

    Notes:
        The cache row is laid out ``[k_pe | kv_latent]``, which **inverts** DeepSeek's reference
        order of ``[k_latent | k_pe]``. Building it inverted costs nothing here, because the two
        fields land in separate PSUM banks anyway, and it saves a transpose later.

        ``F_MAX == kv_row_latent_dim`` is asserted so the PSUM n-tile boundary falls exactly on the
        field boundary, which is what lets the two fields be evicted independently.

        RoPE here is **interleaved**: element ``2i`` pairs with element ``2i + 1``. This differs from
        the sparse indexer, where feature ``i`` pairs with ``i + rope_dim_half``.

    Pseudocode:
        hidden = RMSNorm(hidden_states, input_norm_w) if input_norm_w else hidden_states
        kv_row = hidden @ wkv_a                     # built as [k_pe | kv_latent]
        kv_row.kv_latent = RMSNorm(kv_row.kv_latent, kv_norm_w)
        kv_row.k_pe = RoPE(kv_row.k_pe)
        q = RMSNorm(hidden @ wq_a, q_norm_w) @ wq_b
        q.q_pe = RoPE(q.q_pe)
        q_absorbed = einsum("thd,hdc->cth", q.q_nope, q_absorb_w)
        kv_cache[slot_mapping] = kv_row
        return q_absorbed, q.q_pe, kv_row
    """

    trace_kernel_entry("mla_decode_qkv", locals())

    shapes = get_mla_decode_qkv_shape(
        hidden_states, wq_a, wq_b, wkv_a, kv_cache, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank
    )

    hidden_sbuf = _load_hidden(hidden_states, input_norm_w, norm_eps, shapes)
    cos_sbuf, sin_sbuf = _load_rope_tables(cos, sin, shapes)

    # Move hidden onto partitions so both projections can contract it.
    hidden_tiles_sbuf = _transpose_into_k_tiles(hidden_sbuf, shapes.n_tokens)

    # The KV row arrives in cache-row order, normalized and RoPE'd, ready to scatter.
    kv_cache_row_sbuf = _kv_projection_to_cache_row(
        hidden_tiles_sbuf, wkv_a, kv_norm_w, cos_sbuf, sin_sbuf, norm_eps, shapes
    )

    q_sbuf = _q_down_up_projection(hidden_tiles_sbuf, wq_a, wq_b, q_norm_w, norm_eps, shapes)
    _rope_inplace(q_sbuf, cos_sbuf, sin_sbuf, shapes.qk_nope_dim, shapes.qk_head_dim, shapes.n_heads, shapes.n_tokens)
    q_absorbed_sbuf, q_pe_sbuf = _split_absorb_q(q_sbuf, q_absorb_w, shapes)

    kv_row_cur_hbm = _scatter_current_kv(kv_cache, kv_cache_row_sbuf, slot_mapping, shapes)

    return MlaDecodeQkvResult(q_absorbed_sbuf, q_pe_sbuf, kv_row_cur_hbm)


def _q_down_up_projection(
    hidden_tiles_sbuf: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    q_norm_w: nl.NkiTensor,
    eps: float,
    shapes: MlaDecodeQkvShape,
) -> nl.NkiTensor:
    """
    Projects hidden states down to the query latent, normalizes, and projects back up.

    Returns ``[P_MAX, qk_packed_dim]``, packed ``[q_nope | q_pe]`` per head, and computes
    ``q = RMSNorm(hidden @ wq_a, q_norm_w) @ wq_b``.
    """

    n_tokens = shapes.n_tokens

    # [n_tokens, hidden] @ [hidden, q_lora_rank] -> [n_tokens, q_lora_rank], hidden on partitions
    q_latent_sbuf = _linear(hidden_tiles_sbuf, wq_a, n_tokens)

    q_gamma_sbuf = _load_gamma_broadcast(q_norm_w, n_tokens)
    _rmsnorm_inplace(q_latent_sbuf, q_gamma_sbuf, n_tokens, eps)

    # [n_tokens, q_lora_rank] @ [q_lora_rank, qk_packed_dim] -> [n_tokens, qk_packed_dim]
    q_latent_tiles_sbuf = _transpose_into_k_tiles(q_latent_sbuf, n_tokens)

    return _linear(q_latent_tiles_sbuf, wq_b, n_tokens)


def _kv_projection_to_cache_row(
    hidden_tiles_sbuf: nl.NkiTensor,
    wkv_a: nl.NkiTensor,
    kv_norm_w: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    eps: float,
    shapes: MlaDecodeQkvShape,
) -> nl.NkiTensor:
    """
    Projects hidden states into one complete KV cache row per token, normalized and RoPE'd.

    The row is built in the kernel's own field order, ``[k_pe | kv_latent]``, which inverts
    DeepSeek's reference order of ``[k_latent | k_pe]``. The two fields accumulate into separate PSUM
    banks regardless, so writing them out swapped is free here and saves transposing the row later.

    Computes ``kv_row = hidden @ wkv_a``.
    """

    n_tokens, kv_row_dim, dtype = shapes.n_tokens, shapes.kv_row_dim, shapes.dtype
    k_pe_dim, latent_dim = shapes.kv_row_k_pe_dim, shapes.kv_row_latent_dim
    hidden = wkv_a.shape[0]

    kernel_assert(
        F_MAX == latent_dim,
        f"cache-row eviction needs the n-tile boundary on the field boundary, "
        f"but F_MAX={F_MAX} != kv_row_latent_dim={latent_dim}",
    )

    acc_psums = []
    for _ in TiledRange(kv_row_dim, F_MAX):
        acc_psums.append(nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum))

    is_first_k = True
    for k_tile in TiledRange(hidden, P_MAX):
        k_start, k_end, k_size = k_tile.start_offset, k_tile.end_offset, k_tile.size

        w_moving_sbuf = nl.ndarray((P_MAX, kv_row_dim), dtype=wkv_a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=w_moving_sbuf[0:k_size, 0:kv_row_dim],
            src=wkv_a[k_start:k_end, 0:kv_row_dim],
            dge_mode=nisa.dge_mode.none,
        )

        stationary_tile_start = k_tile.index * n_tokens
        for n_tile in TiledRange(kv_row_dim, F_MAX):
            n_start, n_end, n_size = n_tile.start_offset, n_tile.end_offset, n_tile.size

            nisa.nc_matmul(
                dst=acc_psums[n_tile.index][0:n_tokens, 0:n_size],
                stationary=hidden_tiles_sbuf[0:k_size, stationary_tile_start : stationary_tile_start + n_tokens],
                moving=w_moving_sbuf[0:k_size, n_start:n_end],
                accumulate=not is_first_k,
            )

        is_first_k = False

    # Evict the two PSUM banks into swapped field positions, which is where the inversion happens.
    kv_cache_row_sbuf = nl.ndarray((P_MAX, kv_row_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=kv_cache_row_sbuf[0:n_tokens, k_pe_dim:kv_row_dim],
        src=acc_psums[0][0:n_tokens, 0:latent_dim],
    )
    nisa.tensor_copy(
        dst=kv_cache_row_sbuf[0:n_tokens, 0:k_pe_dim],
        src=acc_psums[1][0:n_tokens, 0:k_pe_dim],
    )

    kv_gamma_sbuf = _load_gamma_broadcast(kv_norm_w, n_tokens)
    _rmsnorm_inplace(kv_cache_row_sbuf[0:P_MAX, k_pe_dim:kv_row_dim], kv_gamma_sbuf, n_tokens, eps)

    # One rope block covering the k_pe field, which sits at the start of the row.
    _rope_inplace(kv_cache_row_sbuf, cos_sbuf, sin_sbuf, 0, k_pe_dim, 1, n_tokens)

    return kv_cache_row_sbuf


def _rope_inplace(
    x_sbuf: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    col_offset: int,
    block_stride: int,
    n_rope_blocks: int,
    P: int,
) -> None:
    """
    Applies interleaved RoPE in place across ``n_rope_blocks`` equally strided blocks at once.

    Interleaved means element ``2i`` pairs with element ``2i + 1``, so the even and odd lanes are
    reached with one stride-2 access pattern each, and the rotation is
    ``even' = even * cos - odd * sin`` and ``odd' = odd * cos + even * sin``.
    """

    rope_dim_half = cos_sbuf.shape[1]
    partition_stride = x_sbuf.shape[1]

    # [stride, count] per axis: partition, then block, then the stride-2 even or odd lane.
    block_pattern = [[partition_stride, P], [block_stride, n_rope_blocks], [2, rope_dim_half]]

    even = x_sbuf.ap(pattern=block_pattern, offset=col_offset)
    odd = x_sbuf.ap(pattern=block_pattern, offset=col_offset + 1)

    cos_blocks = cos_sbuf[0:P, 0:rope_dim_half].expand_dim(1).broadcast(1, n_rope_blocks)
    sin_blocks = sin_sbuf[0:P, 0:rope_dim_half].expand_dim(1).broadcast(1, n_rope_blocks)

    even_out = nl.ndarray((P_MAX, n_rope_blocks, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    odd_out = nl.ndarray((P_MAX, n_rope_blocks, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    scratch = nl.ndarray((P_MAX, n_rope_blocks, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_tensor(even_out[0:P], even, cos_blocks, nl.multiply)
    nisa.tensor_tensor(scratch[0:P], odd, sin_blocks, nl.multiply)
    nisa.tensor_tensor(even_out[0:P], even_out[0:P], scratch[0:P], nl.subtract)

    nisa.tensor_tensor(odd_out[0:P], odd, cos_blocks, nl.multiply)
    nisa.tensor_tensor(scratch[0:P], even, sin_blocks, nl.multiply)
    nisa.tensor_tensor(odd_out[0:P], odd_out[0:P], scratch[0:P], nl.add)

    nisa.tensor_copy(dst=even, src=even_out[0:P])
    nisa.tensor_copy(dst=odd, src=odd_out[0:P])


def _split_absorb_q(
    q_sbuf: nl.NkiTensor,
    q_absorb_w: nl.NkiTensor,
    shapes: MlaDecodeQkvShape,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Splits the packed query into its nope and rope halves, and maps the nope half into the cache's
    latent space so the attention core can score against cached latents directly.

    Both outputs are left feature-major, which is the layout the core's score matmuls want, so the
    per-head transposes here are what make the hand-over free.

    Computes ``q_absorbed = einsum("thd,hdc->cth", q_nope, q_absorb_w)``.
    """

    qk_nope_dim, qk_rope_dim = shapes.qk_nope_dim, shapes.qk_rope_dim
    qk_head_dim, kv_lora_rank = shapes.qk_head_dim, shapes.kv_lora_rank
    n_tokens, n_heads, dtype = shapes.n_tokens, shapes.n_heads, shapes.dtype
    n_kv_tiles = div_ceil(kv_lora_rank, P_MAX)

    q_absorbed_sbuf = nl.ndarray((P_MAX, n_kv_tiles, n_tokens, n_heads), dtype=dtype, buffer=nl.sbuf)
    q_pe_sbuf = nl.ndarray((P_MAX, n_tokens, n_heads), dtype=dtype, buffer=nl.sbuf)

    for head in range(n_heads):
        q_nope_start = head * qk_head_dim
        q_pe_start = q_nope_start + qk_nope_dim

        q_absorb_w_head_sbuf = nl.ndarray((P_MAX, kv_lora_rank), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_absorb_w_head_sbuf[0:qk_nope_dim, 0:kv_lora_rank],
            src=q_absorb_w[head, 0:qk_nope_dim, 0:kv_lora_rank],
        )

        # Transpose this head's q_nope so the absorb contracts it on partitions.
        q_nope_head_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(
            q_nope_head_psum[0:qk_nope_dim, 0:n_tokens],
            q_sbuf[0:n_tokens, q_nope_start : q_nope_start + qk_nope_dim],
        )

        q_nope_head_sbuf = nl.ndarray((P_MAX, n_tokens), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=q_nope_head_sbuf[0:qk_nope_dim, 0:n_tokens],
            src=q_nope_head_psum[0:qk_nope_dim, 0:n_tokens],
        )

        q_absorbed_psum = nl.ndarray((P_MAX, n_kv_tiles, n_tokens), dtype=nl.float32, buffer=nl.psum)
        for kv_tile in TiledRange(kv_lora_rank, P_MAX):
            kv_start, kv_end, kv_size = kv_tile.start_offset, kv_tile.end_offset, kv_tile.size

            nisa.nc_matmul(
                dst=q_absorbed_psum[0:kv_size, kv_tile.index, 0:n_tokens],
                stationary=q_absorb_w_head_sbuf[0:qk_nope_dim, kv_start:kv_end],
                moving=q_nope_head_sbuf[0:qk_nope_dim, 0:n_tokens],
            )

        nisa.tensor_copy(
            dst=q_absorbed_sbuf[0:P_MAX, 0:n_kv_tiles, 0:n_tokens, head],
            src=q_absorbed_psum[0:P_MAX, 0:n_kv_tiles, 0:n_tokens],
        )

        # q_pe needs no absorb, only the same transpose into feature-major.
        q_pe_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=q_pe_psum[0:qk_rope_dim, 0:n_tokens],
            data=q_sbuf[0:n_tokens, q_pe_start : q_pe_start + qk_rope_dim],
        )
        nisa.tensor_copy(dst=q_pe_sbuf[0:qk_rope_dim, 0:n_tokens, head], src=q_pe_psum[0:qk_rope_dim, 0:n_tokens])

    return q_absorbed_sbuf, q_pe_sbuf


def _scatter_current_kv(
    kv_cache: nl.NkiTensor,
    kv_cache_row_sbuf: nl.NkiTensor,
    slot_mapping: nl.NkiTensor,
    shapes: MlaDecodeQkvShape,
) -> nl.NkiTensor:
    """
    Scatters this step's cache rows into ``kv_cache`` at ``slot_mapping``, and publishes the same
    rows key-major through HBM for the attention core's key contraction.

    The row arrives complete from ``_kv_projection_to_cache_row``, normalized and RoPE'd, so there is
    nothing to assemble here.

    Handing the row over in SBUF instead was measured twice, before and after the key-major work,
    and sat inside the run-to-run noise band both times (516.3/517.2 against 511.4, then 368.5/370.4
    against a 369.4-373.3 baseline). The only repeatable signal was DMA active time at -2.8 us, which
    does not justify coupling the core to this tile's partition layout, where token ``t`` lives on
    partition ``t`` and the core would have to cross it with 16 SBUF DMAs.
    """

    n_tokens, kv_row_dim = shapes.n_tokens, shapes.kv_row_dim

    # Reshaped to [n_tokens, 1] so the indirect DMA reads one slot index per partition.
    slot_mapping_sbuf = nl.ndarray((n_tokens, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_mapping_sbuf[0:n_tokens, 0:1], src=slot_mapping[0:n_tokens].reshape((n_tokens, 1)))

    # Flatten the paged cache so a slot index addresses a row directly.
    kv_cache_rows = kv_cache.reshape((shapes.n_cache_rows, kv_row_dim))
    nisa.dma_copy(
        dst=kv_cache_rows.ap(
            pattern=[[kv_row_dim, n_tokens], [1, kv_row_dim]],
            offset=0,
            vector_offset=slot_mapping_sbuf.ap(pattern=[[1, n_tokens], [1, 1]], offset=0),
            indirect_dim=0,
        ),
        src=kv_cache_row_sbuf[0:n_tokens, 0:kv_row_dim],
        oob_mode=nisa.oob_mode.skip,
    )

    kv_cache_row_hbm = nl.ndarray((n_tokens, kv_row_dim), dtype=shapes.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=kv_cache_row_hbm[0:n_tokens, 0:kv_row_dim], src=kv_cache_row_sbuf[0:n_tokens, 0:kv_row_dim])

    return kv_cache_row_hbm


def _linear(stationary_tiles_sbuf: nl.NkiTensor, w_hbm: nl.NkiTensor, P: int) -> nl.NkiTensor:
    """
    Applies a linear transformation from an already K-tiled stationary operand.

    Computes ``y[P, N] = x[P, K] @ w_hbm[K, N]``.
    """

    K, N = w_hbm.shape[0], w_hbm.shape[1]
    kernel_assert(
        stationary_tiles_sbuf.shape[1] == div_ceil(K, P_MAX) * P,
        "stationary packing must match w_hbm's K and P",
    )

    acc_psums = []
    for _ in TiledRange(N, F_MAX):
        acc_psums.append(nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum))

    is_first_k = True
    for k_tile in TiledRange(K, P_MAX):
        k_start, k_end, k_size = k_tile.start_offset, k_tile.end_offset, k_tile.size

        w_moving_sbuf = nl.ndarray((P_MAX, N), dtype=w_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_moving_sbuf[0:k_size, 0:N], src=w_hbm[k_start:k_end, 0:N], dge_mode=nisa.dge_mode.none)

        stationary_tile_start = k_tile.index * P
        for n_tile in TiledRange(N, F_MAX):
            n_start, n_end, n_size = n_tile.start_offset, n_tile.end_offset, n_tile.size
            nisa.nc_matmul(
                dst=acc_psums[n_tile.index][0:P, 0:n_size],
                stationary=stationary_tiles_sbuf[0:k_size, stationary_tile_start : stationary_tile_start + P],
                moving=w_moving_sbuf[0:k_size, n_start:n_end],
                accumulate=not is_first_k,
            )

        is_first_k = False

    out_sbuf = nl.ndarray((P_MAX, N), dtype=stationary_tiles_sbuf.dtype, buffer=nl.sbuf)
    for n_tile in TiledRange(N, F_MAX):
        n_start, n_end, n_size = n_tile.start_offset, n_tile.end_offset, n_tile.size
        nisa.tensor_copy(dst=out_sbuf[0:P, n_start:n_end], src=acc_psums[n_tile.index][0:P, 0:n_size])

    return out_sbuf


def _rmsnorm_inplace(x_sbuf: nl.NkiTensor, gamma_sbuf: nl.NkiTensor, P: int, eps: float) -> None:
    """
    Applies RMSNorm in place over ``x_sbuf`` ``[P, F]``.

    Computes ``x *= rsqrt(mean(x^2) + eps) * gamma``.
    """

    F = x_sbuf.shape[1]

    squares_discard = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=squares_discard[0:P, 0:F],
        op=nl.square,
        data=x_sbuf[0:P, 0:F],
        reduce_op=nl.add,
        reduce_res=inv_rms[0:P, 0:1],
    )

    # The mean's 1/F and the eps fold into the activation's scale and bias: rsqrt(sum/F + eps).
    nisa.activation(inv_rms[0:P, 0:1], op=nl.rsqrt, data=inv_rms[0:P, 0:1], scale=1.0 / F, bias=eps)

    nisa.scalar_tensor_tensor(
        dst=x_sbuf[0:P, 0:F],
        data=x_sbuf[0:P, 0:F],
        op0=nl.multiply,
        operand0=inv_rms[0:P, 0:1],
        op1=nl.multiply,
        operand1=gamma_sbuf[0:P, 0:F],
    )


def _load_hidden(
    hidden_states: nl.NkiTensor,
    input_norm_w: nl.NkiTensor | None,
    eps: float,
    shapes: MlaDecodeQkvShape,
) -> nl.NkiTensor:
    """Loads hidden states from HBM to SBUF, applying RMSNorm eagerly when ``input_norm_w`` is given."""

    n_tokens, hidden, dtype = shapes.n_tokens, shapes.hidden, shapes.dtype

    hidden_sbuf = nl.ndarray((P_MAX, hidden), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=hidden_sbuf[0:n_tokens, 0:hidden], src=hidden_states[0:n_tokens, 0:hidden])

    if input_norm_w is not None:
        gamma_sbuf = _load_gamma_broadcast(input_norm_w, n_tokens)
        _rmsnorm_inplace(hidden_sbuf, gamma_sbuf, n_tokens, eps)

    return hidden_sbuf


def _load_rope_tables(
    cos: nl.NkiTensor, sin: nl.NkiTensor, shapes: MlaDecodeQkvShape
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Loads the ``cos`` and ``sin`` tables used by RoPE from HBM to SBUF."""

    n_tokens, qk_rope_dim_half = shapes.n_tokens, shapes.qk_rope_dim // 2

    cos_sbuf = nl.ndarray((P_MAX, qk_rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    sin_sbuf = nl.ndarray((P_MAX, qk_rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sbuf[0:n_tokens, 0:qk_rope_dim_half], src=cos[0:n_tokens, 0:qk_rope_dim_half])
    nisa.dma_copy(dst=sin_sbuf[0:n_tokens, 0:qk_rope_dim_half], src=sin[0:n_tokens, 0:qk_rope_dim_half])

    return cos_sbuf, sin_sbuf


def _load_gamma_broadcast(gamma_hbm: nl.NkiTensor, P: int) -> nl.NkiTensor:
    """Loads an RMSNorm gamma of shape ``[1, D]`` broadcast to ``[P, D]``."""

    D = gamma_hbm.shape[1]

    gamma_sbuf = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sbuf[0:P, 0:D], src=gamma_hbm[0:1, 0:D].broadcast(0, P))

    return gamma_sbuf


def _transpose_into_k_tiles(x_sbuf: nl.NkiTensor, P: int) -> nl.NkiTensor:
    """
    Tiles a ``[P, K]`` tensor across its contraction dimension, transposed so K lands on partitions.

    Returns ``[P_MAX, div_ceil(K, P_MAX) * P]``, which is the packing ``_linear`` expects.
    """

    dtype, K = x_sbuf.dtype, x_sbuf.shape[1]
    n_k_tiles = div_ceil(K, P_MAX)

    tiles_sbuf = nl.ndarray((P_MAX, n_k_tiles * P), dtype=dtype, buffer=nl.sbuf)

    for tile in TiledRange(K, P_MAX):
        start_offset, end_offset, size = tile.start_offset, tile.end_offset, tile.size

        tile_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(
            tile_psum[0:size, 0:P],
            x_sbuf[0:P, start_offset:end_offset],
        )

        tile_start = tile.index * P
        nisa.tensor_copy(dst=tiles_sbuf[0:size, tile_start : tile_start + P], src=tile_psum[0:size, 0:P])

    return tiles_sbuf
