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

"""DeepSeek-V3.2 MLA decode attention core: scores, softmax, out-absorb, and o-projection."""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.entry_trace import trace_kernel_entry
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.tiled_range import TiledRange
from . import F_MAX, P_MAX, MlaDecodeAttentionShape, get_mla_decode_attention_shape

_NEG_INF = -30000.0
"""
Score written into the key slots that no token ever fills, so they drop out of every softmax.

Finite rather than ``-inf``, because these slots are also multiplied and summed.
"""

_MASK_BOUND = 1.0e30
"""
Magnitude the causal mask clamps a score to.

``min(score, +_MASK_BOUND)`` is the score and ``min(score, -_MASK_BOUND)`` is ``-_MASK_BOUND`` for
any finite score, which makes the clamp a true select that a large-magnitude dropped score cannot
defeat. The bound is built as ``keep * 2 * _MASK_BOUND - _MASK_BOUND``, so ``2 * _MASK_BOUND`` has to
stay representable in fp32: at 3e38 it overflows to infinity, which happens to give the right answer
for a kept key and the wrong one for every dropped key.
"""

_DMA_TRANSPOSED_LATENT_CHUNKS = 1
"""
How many latent chunks arrive feature-major straight from HBM, the rest transposing on the PE.

This balances two bottlenecks. Each DMA-transposed chunk takes roughly 18 us of PE and eviction work
off the compute engines, but costs about 16x its logical bytes in DMA: 8.4 MB of latent per chunk
becomes ~134 MB of traffic. The cost is the destination scatter, not the source, because each
gathered row's 128 values land on 128 different partitions and every partition write carries a
minimum granule; read bytes stay at exactly 8.4 MB. Moving all four chunks drives DMA to 75%
occupancy and is slower than moving one, so the response is a U-curve and 1 captures 91% of the win
for a third of the extra traffic. That default is also deliberate for the whole model: this kernel is
one of 61 layers sharing the same DMA, and software-DGE pressure is what made the full model hang
when this transpose was first tried. A feature-chunked cache layout was prototyped and does not help,
since the amplification is on the destination side.
"""

_SOFTMAX_GROUP_SIZE = 4
"""
Tokens that share one batched softmax.

The softmax tail is a serial chain of about ten operations producing a couple of scalars, and this
kernel is bound by the latency of its per-token chain rather than by engine occupancy, so collapsing
several tails into one is worth more than relocating work. A whole-batch softmax needed a re-gather
that cost more DMA than the batching saved, so the group is instead sized to keep every member's
token-major latent resident: ``group * n_key_tiles * kv_row_dim * 2`` bytes per partition, which at 4
is about 78 KB of the 240 KB SBUF.
"""

_N_KEY_BUFFERS = 2
"""
Slots in the feature-major key buffers, so consecutive tokens do not share an address.

The allocator folds a loop-local tile's lifetimes onto one address, which makes token ``t + 1``'s
transposes wait on token ``t``'s last score matmul and serializes the per-token chain. Two slots
break that cross-token anti-dependency.
"""


@nki.jit
def mla_decode_attention(
    q_absorbed_sbuf: nl.NkiTensor,  # [P_MAX, n_latent_tiles, n_tokens, n_heads] SBUF, feature-major
    q_pe_sbuf: nl.NkiTensor,  # [P_MAX, n_tokens, n_heads]                 SBUF, feature-major
    kv_cache: nl.NkiTensor,  # [n_blocks, 1, block_size, kv_row_dim]      paged, read only
    kv_row_cur: nl.NkiTensor,  # [n_tokens, kv_row_dim]                     this step's cache row
    prior_rows: nl.NkiTensor,  # [P_MAX, n_tokens, n_prior_tiles] int32     gather order
    prior_key_pos: nl.NkiTensor,  # [P_MAX, n_tokens, n_prior_tiles] fp32      key-major positions
    prior_rows_natural: nl.NkiTensor,  # [P_MAX, n_tokens, n_prior_tiles] uint32    natural key order
    positions: nl.NkiTensor,  # [n_tokens] int32
    out_absorb_w: nl.NkiTensor,  # [n_heads, v_head_dim, kv_lora_rank]        wkv_b value half
    o_proj_w: nl.NkiTensor,  # [n_heads * v_head_dim, hidden]             per-rank o-projection
    n_prior: int,
    softmax_scale: float,
) -> nl.NkiTensor:
    """
    Computes one DeepSeek-V3.2 MLA decode step's attention over the prior keys an indexer selected,
    then projects the result out of the cache's latent space and through the o-projection, returning
    this rank's attention-block output.

    This is the back half of an MLA decode layer, and the counterpart of the QKV front-fold. It is
    designed to be traced inline after that front-fold rather than run standalone: the two query
    tiles arrive **in SBUF**, already feature-major in the layout the score matmuls index, so a
    standalone caller has to stage them itself.

    Selection has already happened, so there is no sparse score bias here. What the selection leaves
    behind is a per-token map from key slot to absolute position, and that map drives the causal mask.

    Rather than read a pre-gathered feature-major prior out of HBM, the kernel gathers each token's
    selected rows straight from ``kv_cache`` in the cache's native token-major layout, and then
    transposes once to feature-major for the score matmul. The weighted sum contracts over keys and
    so wants the token-major tile, which is what the gather already produced, and reads it directly.
    That deletes a 33.5 MB HBM round trip and one of two mirror-image transposes.

    Dimensions:
        n_tokens: Decode tokens in the batch, one per sequence. Must be <= P_MAX.
        n_heads: Attention heads on this rank. Must be <= P_MAX.
        n_prior: Prior keys the indexer selected per token.
        n_keys: Keys scored per token, ``n_prior + 1``, the selected prior plus the current token.
        kv_lora_rank: LoRA rank of the KV path, and the width of a cache row's latent field. Must be
            a multiple of P_MAX and fit one PSUM bank.
        qk_rope_dim: Width of a cache row's ``k_pe`` field. Must be <= P_MAX.
        v_head_dim: Per-head width the out-absorb produces. Must be <= P_MAX.
        hidden: Model hidden width, which the o-projection emits.
        kv_row_dim: One cache row, ``qk_rope_dim + kv_lora_rank``.
        n_blocks, block_size: Paged KV cache geometry.

    Args:
        q_absorbed_sbuf: ``[P_MAX, n_latent_tiles, n_tokens, n_heads]``. The query already mapped into
            the cache's latent space by the front-fold's absorb.
        q_pe_sbuf: ``[P_MAX, n_tokens, n_heads]``. The query's RoPE'd positional half.
        kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Paged cache, read only here. The current
            token's row must already have been scattered into it by the front-fold.
        kv_row_cur: ``[n_tokens, kv_row_dim]``. This step's cache row per token, laid out
            ``[k_pe | kv_latent]``.
        prior_rows: ``[P_MAX, n_tokens, n_prior_tiles]`` int32. Flat cache row per selected key, in
            the batched gather's consumption order. From ``build_prior_index``.
        prior_key_pos: ``[P_MAX, n_tokens, n_prior_tiles]`` fp32. Absolute position of each selected
            key, key-major. From ``build_prior_index``.
        prior_rows_natural: ``[P_MAX, n_tokens, n_prior_tiles]`` uint32. The same rows in natural key
            order, which is what the indirect ``dma_transpose`` needs. From ``build_prior_index``.
        positions: ``[n_tokens]`` int32. This step's decode position per sequence.
        out_absorb_w: ``[n_heads, v_head_dim, kv_lora_rank]``. The value half of ``wkv_b``, which maps
            a head's latent attention output back out into ``v_head_dim``.
        o_proj_w: ``[n_heads * v_head_dim, hidden]``. This rank's o-projection.
        n_prior: Prior keys the indexer selected per token.
        softmax_scale: Scale applied to the scores before masking and softmax.

    Returns:
        ``[n_tokens, hidden]``, this rank's attention-block output before the cross-rank all-reduce.

    Notes:
        Scores and softmax weights are held **key-major**: key ``tile * P_MAX + p`` of head ``h`` sits
        at ``[p, tile, h]``. Swapping which matmul operand is stationary transposes the output for
        free, and putting keys on partitions pays off three times over. The softmax then runs 128 lanes
        wide instead of on the single partition a ``[n_heads, n_keys]`` layout leaves at ``n_heads ==
        1``; the causal mask becomes a plain per-partition column instead of a broadcast matmul; and
        the weighted sum's stationary operand is already oriented correctly, deleting a transpose.
        This was the single largest win in this kernel's history, 521.5 -> 402.2 us, and most of it
        came from lane utilization rather than from the transpose removal.

        The out-absorb and o-projection weights do not depend on the token, so they are loaded once
        above the token loop. Both tails are also batched across tokens: they are per-token GEMVs
        against shared weights, and a ``128x1`` matmul costs about 360 ns against 179 ns for a
        ``128x128``, because the cost is set by the moving operand and a width-1 stationary pays full
        price for 1/128 of the array.

        Emitting the weighted sum for the previous token, so the PE could run ahead of a softmax, was
        measured and is exactly neutral: the post-scheduler reorders globally, so NKI emission order
        does not control per-engine issue order. Do not expect a win from hand-scheduling here.

        Giving each latent tile its own 32-partition band via ``tile_position``, so the tiles issue
        concurrently instead of chaining through one PSUM bank, was measured at 21% slower
        (433.8 -> 525.7 us). It band-sums per token and key chunk rather than once per kernel, and
        each sum is three extra full-width Vector operations on what is already the bottleneck engine.

    Pseudocode:
        for token in range(n_tokens):
            keys = cat([kv_cache[prior_rows[token]], kv_row_cur[token]])   # [n_keys, kv_row_dim]
            scores = softmax_scale * (q_absorbed[token] @ keys.kv_latent.T
                                      + q_pe[token] @ keys.k_pe.T)         # [n_heads, n_keys]
            scores[:, prior_key_pos[token] >= positions[token]] = -inf
            x = softmax(scores, axis=-1) @ keys.kv_latent                  # [n_heads, kv_lora_rank]
            attn = einsum("hc,hdc->hd", x, out_absorb_w)                   # [n_heads, v_head_dim]
            output[token] = attn.reshape(n_heads * v_head_dim) @ o_proj_w
    """

    trace_kernel_entry("mla_decode_attention", locals())

    shapes = get_mla_decode_attention_shape(kv_cache, positions, out_absorb_w, o_proj_w, n_prior)
    kernel_assert(
        q_absorbed_sbuf.dtype == shapes.dtype,
        f"q_absorbed_sbuf dtype {q_absorbed_sbuf.dtype} must match out_absorb_w dtype {shapes.dtype}",
    )

    n_tokens, n_heads, dtype = shapes.n_tokens, shapes.n_heads, shapes.dtype
    n_keys, n_key_tiles = shapes.n_keys, shapes.n_key_tiles
    n_latent_tiles, kv_row_dim = shapes.n_latent_tiles, shapes.kv_row_dim

    """
    The indirect dma_transpose requires a 2-byte source dtype, so fp32 callers transpose every latent chunk on the PE
    instead. This is a dtype, hence a compile-time constant: the tracer emits exactly one path per specialization and no
    runtime branch survives.
    """
    dma_transposed_chunks = _DMA_TRANSPOSED_LATENT_CHUNKS if str(dtype) in ("bfloat16", "float16") else 0

    output = nl.ndarray((n_tokens, shapes.hidden), dtype=dtype, buffer=nl.shared_hbm)

    out_absorb_t_sbuf = _load_out_absorb_transposed(out_absorb_w, shapes)
    o_proj_sbuf = _load_o_proj(o_proj_w, shapes)
    positions_sbuf = _load_positions_broadcast(positions, shapes)

    """
    Ones vectors for the two cross-partition steps the key-major layout needs: a [1, P_MAX] stationary broadcasts a
    per-head scalar down every partition, a [P_MAX, 1] stationary sums over them. Both are constants, so they are built
    once.
    """
    ones_row_sbuf = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row_sbuf[0:1, 0:P_MAX], value=1.0)
    ones_partition_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_partition_sbuf[0:P_MAX, 0:1], value=1.0)

    """
    The gathered rows are the longest-lived per-token buffer, since the weighted sum reads them at the very end of a
    token's chain, and the softmax group re-reads what the score phase gathered. One slot per group member is therefore
    both what the grouping needs and what keeps token t + 1's gather from waiting on token t's last matmul.
    """
    group_capacity = min(_SOFTMAX_GROUP_SIZE, n_tokens)
    key_rows_sbuf = nl.ndarray((P_MAX, group_capacity, n_key_tiles, kv_row_dim), dtype=dtype, buffer=nl.sbuf)

    latent_keys_sbuf = nl.ndarray(
        (P_MAX, _N_KEY_BUFFERS, n_latent_tiles * shapes.latent_chunk_stride), dtype=dtype, buffer=nl.sbuf
    )
    pe_keys_sbuf = nl.ndarray((P_MAX, _N_KEY_BUFFERS, n_keys), dtype=dtype, buffer=nl.sbuf)

    scores_sbuf = nl.ndarray((P_MAX, n_tokens, n_key_tiles, n_heads), dtype=nl.float32, buffer=nl.sbuf)
    """
    The last key tile holds only the current token when n_prior is a multiple of P_MAX, so most of its partitions are
    not keys at all. No token ever writes them, so one memset here keeps them out of every token's softmax sum for the
    whole kernel.
    """
    nisa.memset(dst=scores_sbuf[0:P_MAX, 0:n_tokens, 0:n_key_tiles, 0:n_heads], value=_NEG_INF)
    weights_sbuf = nl.ndarray((P_MAX, n_tokens, n_key_tiles, n_heads), dtype=dtype, buffer=nl.sbuf)

    # Softmax state, all [*, n_tokens * n_heads] so one group's tokens are a contiguous slice. Every
    # token's scores are n_key_tiles * n_heads fp32 per partition, so the whole batch fits ~1.1 KB.
    partition_max_sbuf = nl.ndarray((P_MAX, n_tokens * n_heads), dtype=nl.float32, buffer=nl.sbuf)
    partition_sum_sbuf = nl.ndarray((P_MAX, n_tokens * n_heads), dtype=nl.float32, buffer=nl.sbuf)
    neg_max_sbuf = nl.ndarray((P_MAX, n_tokens * n_heads), dtype=nl.float32, buffer=nl.sbuf)
    inv_denom_sbuf = nl.ndarray((1, n_tokens * n_heads), dtype=nl.float32, buffer=nl.sbuf)

    # The latent attention output, transposed, for every token: [latent_tile, chunk, head, token] with
    # token innermost so the batched tail reads all tokens of one (head, chunk) as one slice.
    latent_out_t_sbuf = nl.ndarray((P_MAX, n_latent_tiles * n_heads * n_tokens), dtype=dtype, buffer=nl.sbuf)

    for group_start in range(0, n_tokens, _SOFTMAX_GROUP_SIZE):
        # The last group is short when the group size does not divide n_tokens.
        group_size = min(_SOFTMAX_GROUP_SIZE, n_tokens - group_start)

        for token in range(group_start, group_start + group_size):
            slot = token - group_start
            key_buf = token % _N_KEY_BUFFERS

            _gather_key_rows(key_rows_sbuf, slot, kv_cache, kv_row_cur, prior_rows, token, shapes)
            _transpose_keys_feature_major(
                latent_keys_sbuf[0:P_MAX, key_buf],
                pe_keys_sbuf[0:P_MAX, key_buf],
                key_rows_sbuf,
                slot,
                dma_transposed_chunks,
                shapes,
            )
            _dma_transpose_prior_latent(
                latent_keys_sbuf[0:P_MAX, key_buf], kv_cache, prior_rows_natural, token, dma_transposed_chunks, shapes
            )

            mask_bound_sbuf = _causal_mask_bound(prior_key_pos, positions_sbuf, token, shapes)
            _score_keys(
                scores_sbuf,
                latent_keys_sbuf[0:P_MAX, key_buf],
                pe_keys_sbuf[0:P_MAX, key_buf],
                q_absorbed_sbuf,
                q_pe_sbuf,
                mask_bound_sbuf,
                token,
                softmax_scale,
                shapes,
            )
            _reduce_key_tile_max(partition_max_sbuf, scores_sbuf, token, shapes)

        _group_neg_max(neg_max_sbuf, partition_max_sbuf, ones_row_sbuf, group_start, group_size, shapes)
        _group_exp_and_inv_denominator(
            weights_sbuf,
            inv_denom_sbuf,
            partition_sum_sbuf,
            scores_sbuf,
            neg_max_sbuf,
            ones_partition_sbuf,
            group_start,
            group_size,
            shapes,
        )

        # The weighted sum reuses what the score phase already gathered, so nothing is re-read.
        for token in range(group_start, group_start + group_size):
            _weighted_latent(
                latent_out_t_sbuf, weights_sbuf, key_rows_sbuf, inv_denom_sbuf, token, token - group_start, shapes
            )

    attn_sbuf = _out_absorb(latent_out_t_sbuf, out_absorb_t_sbuf, shapes)
    _o_proj(output, attn_sbuf, o_proj_sbuf, shapes)

    return output


def _load_out_absorb_transposed(out_absorb_w: nl.NkiTensor, shapes: MlaDecodeAttentionShape) -> nl.NkiTensor:
    """
    Loads every head's out-absorb weight transposed, so the latent axis lands on partitions.

    Returns ``[P_MAX, n_latent_tiles * n_heads * v_head_dim]``, holding chunk ``chunk`` of head
    ``head`` at column ``(head * n_latent_tiles + chunk) * v_head_dim``. The weight does not depend on
    the token, and loading and re-transposing it per token cost ``n_tokens`` times the same work.
    """

    n_heads, v_head_dim, dtype = shapes.n_heads, shapes.v_head_dim, shapes.dtype
    n_latent_tiles = shapes.n_latent_tiles

    out_absorb_t_sbuf = nl.ndarray((P_MAX, n_latent_tiles * n_heads * v_head_dim), dtype=dtype, buffer=nl.sbuf)
    for head in range(n_heads):
        for latent_tile in TiledRange(shapes.kv_lora_rank, P_MAX):
            latent_start, latent_end, latent_size = (
                latent_tile.start_offset,
                latent_tile.end_offset,
                latent_tile.size,
            )

            head_chunk_sbuf = nl.ndarray((P_MAX, v_head_dim), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=head_chunk_sbuf[0:v_head_dim, 0:latent_size],
                src=out_absorb_w[head, 0:v_head_dim, latent_start:latent_end],
            )

            head_chunk_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(
                head_chunk_psum[0:latent_size, 0:v_head_dim], head_chunk_sbuf[0:v_head_dim, 0:latent_size]
            )

            base = (head * n_latent_tiles + latent_tile.index) * v_head_dim
            nisa.tensor_copy(
                dst=out_absorb_t_sbuf[0:latent_size, base : base + v_head_dim],
                src=head_chunk_psum[0:latent_size, 0:v_head_dim],
            )

    return out_absorb_t_sbuf


def _load_o_proj(o_proj_w: nl.NkiTensor, shapes: MlaDecodeAttentionShape) -> nl.NkiTensor:
    """
    Loads the o-projection weight as ``[v_head_dim, hidden]`` per head, the o-proj's moving operand.

    Returns ``[P_MAX, n_heads * hidden]``. Like the out-absorb this is token-invariant, and re-loading
    it per token cost 29.4 MB of redundant DMA at the production decode shape.
    """

    n_heads, v_head_dim, hidden = shapes.n_heads, shapes.v_head_dim, shapes.hidden

    o_proj_sbuf = nl.ndarray((P_MAX, n_heads * hidden), dtype=shapes.dtype, buffer=nl.sbuf)
    for head in range(n_heads):
        nisa.dma_copy(
            dst=o_proj_sbuf[0:v_head_dim, head * hidden : (head + 1) * hidden],
            src=o_proj_w[head * v_head_dim : (head + 1) * v_head_dim, 0:hidden],
        )

    return o_proj_sbuf


def _load_positions_broadcast(positions: nl.NkiTensor, shapes: MlaDecodeAttentionShape) -> nl.NkiTensor:
    """
    Loads ``positions`` replicated down every partition, as fp32.

    Returns ``[P_MAX, n_tokens]``. The causal predicate is per key and keys are partitions, so the
    current position has to be available on all of them; a stride-0 partition pattern gets it there in
    one DMA instead of a per-token broadcast.
    """

    n_tokens = shapes.n_tokens

    positions_int_sbuf = nl.ndarray((P_MAX, n_tokens), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=positions_int_sbuf[0:P_MAX, 0:n_tokens],
        src=positions[0:n_tokens].reshape((1, n_tokens)).ap(pattern=[[0, P_MAX], [1, n_tokens]], offset=0),
    )

    positions_sbuf = nl.ndarray((P_MAX, n_tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=positions_sbuf[0:P_MAX, 0:n_tokens], src=positions_int_sbuf[0:P_MAX, 0:n_tokens])

    return positions_sbuf


def _gather_key_rows(
    key_rows_sbuf: nl.NkiTensor,
    slot: int,
    kv_cache: nl.NkiTensor,
    kv_row_cur: nl.NkiTensor,
    prior_rows: nl.NkiTensor,
    token: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Gathers one token's selected prior rows plus its current row into ``key_rows_sbuf[:, slot]``.

    The rows land in the cache's native token-major order, key ``tile * P_MAX + p`` on partition ``p``,
    which is what the weighted sum wants and what the score matmul transposes once.

    ``prior_rows`` is already in this gather's consumption order, so the token's slice is used as-is:
    one indirect DMA covers every selected key rather than one per key tile.
    """

    n_prior, kv_row_dim = shapes.n_prior, shapes.kv_row_dim
    n_prior_tiles = shapes.n_prior_tiles

    cache_rows = kv_cache.reshape((shapes.n_cache_rows, kv_row_dim))
    nisa.dma_copy(
        dst=key_rows_sbuf[0:P_MAX, slot, 0:n_prior_tiles, 0:kv_row_dim],
        src=cache_rows.ap(
            pattern=[[kv_row_dim, P_MAX * n_prior_tiles], [1, kv_row_dim]],
            offset=0,
            vector_offset=prior_rows[0:P_MAX, token, 0:n_prior_tiles],
            indirect_dim=0,
        ),
        oob_mode=nisa.oob_mode.skip,
    )

    # The current token is key n_prior, so it lands at partition n_prior % P_MAX of the tile that
    # index falls in, which is a tile the prior shares whenever n_prior is ragged.
    cur_partition, cur_tile = n_prior % P_MAX, n_prior // P_MAX
    nisa.dma_copy(
        dst=key_rows_sbuf[cur_partition : cur_partition + 1, slot, cur_tile, 0:kv_row_dim],
        src=kv_row_cur[token : token + 1, 0:kv_row_dim],
    )


def _transpose_keys_feature_major(
    latent_keys_sbuf: nl.NkiTensor,
    pe_keys_sbuf: nl.NkiTensor,
    key_rows_sbuf: nl.NkiTensor,
    slot: int,
    dma_transposed_chunks: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Transposes the gathered rows to feature-major on the PE, which is what the score matmul contracts.

    Writes ``pe_keys_sbuf`` ``[qk_rope_dim, n_keys]`` and, for the latent chunks the DMA does not
    deliver itself, ``latent_keys_sbuf`` ``[P_MAX, chunk * latent_chunk_stride + key]``.

    ``k_pe`` always transposes here. The indirect ``dma_transpose`` needs a source width that is a
    multiple of P_MAX rather than merely at most P_MAX, and a 64-wide ``k_pe`` slice silently returns
    wrong numbers instead of failing, which production testing saw as an 87.9% relative difference.

    The last key tile also always transposes here, for every chunk: the DMA gather only covers the
    ``n_prior`` selected rows, so the current token's key always arrives token-major. When ``n_prior``
    is ragged that tile straddles both paths and the PE rewrites the values the DMA already wrote,
    which is redundant but not wrong, since both orders place a given key in the same column.
    """

    qk_rope_dim, n_keys = shapes.qk_rope_dim, shapes.n_keys
    latent_chunk_stride = shapes.latent_chunk_stride

    for key_tile in TiledRange(n_keys, P_MAX):
        key_start, key_end, key_size = key_tile.start_offset, key_tile.end_offset, key_tile.size
        is_last_key_tile = key_tile.index == shapes.n_key_tiles - 1

        pe_psum = nl.ndarray((P_MAX, P_MAX), dtype=shapes.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            pe_psum[0:qk_rope_dim, 0:key_size],
            key_rows_sbuf[0:key_size, slot, key_tile.index, 0:qk_rope_dim],
        )
        nisa.tensor_copy(dst=pe_keys_sbuf[0:qk_rope_dim, key_start:key_end], src=pe_psum[0:qk_rope_dim, 0:key_size])

        for latent_tile in TiledRange(shapes.kv_lora_rank, P_MAX):
            latent_start, latent_end, latent_size = (
                latent_tile.start_offset,
                latent_tile.end_offset,
                latent_tile.size,
            )
            if latent_tile.index < dma_transposed_chunks and not is_last_key_tile:
                continue

            k_pe_dim = shapes.kv_row_k_pe_dim
            latent_psum = nl.ndarray((P_MAX, P_MAX), dtype=shapes.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                latent_psum[0:latent_size, 0:key_size],
                key_rows_sbuf[0:key_size, slot, key_tile.index, k_pe_dim + latent_start : k_pe_dim + latent_end],
            )

            base = latent_tile.index * latent_chunk_stride
            nisa.tensor_copy(
                dst=latent_keys_sbuf[0:latent_size, base + key_start : base + key_end],
                src=latent_psum[0:latent_size, 0:key_size],
            )


def _dma_transpose_prior_latent(
    latent_keys_sbuf: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    prior_rows_natural: nl.NkiTensor,
    token: int,
    dma_transposed_chunks: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Gathers the leading latent chunks feature-major straight from HBM, transposing inside the DMA.

    One indirect ``dma_transpose`` per chunk both gathers the token's selected rows and transposes
    them, so it replaces ``n_prior_tiles`` PE transposes and their PSUM evictions per chunk per token
    and never touches the compute engines. See ``_DMA_TRANSPOSED_LATENT_CHUNKS`` for why only the
    leading chunks take this path.

    The chunk width must be exactly P_MAX: an indirect ``dma_transpose`` whose source width is not a
    multiple of 128 silently corrupts the gather, which the ``kv_lora_rank`` multiple-of-P_MAX
    assertion guarantees against. The hardware flattens the index table column-major, which is why the
    natural-order table is the one that yields destination column ``tile * P_MAX + p`` for key
    ``tile * P_MAX + p``, matching both the PE path's layout and ``prior_key_pos``'s order.
    """

    n_prior, k_pe_dim = shapes.n_prior, shapes.kv_row_k_pe_dim
    latent_chunk_stride = shapes.latent_chunk_stride

    cache_rows = kv_cache.reshape((shapes.n_cache_rows, shapes.kv_row_dim))
    for chunk in range(dma_transposed_chunks):
        base = chunk * latent_chunk_stride
        nisa.dma_transpose(
            dst=latent_keys_sbuf[0:P_MAX, base : base + n_prior],
            src=cache_rows.ap(
                pattern=[[shapes.kv_row_dim, n_prior], [1, P_MAX]],
                offset=k_pe_dim + chunk * P_MAX,
                vector_offset=prior_rows_natural[0:P_MAX, token, 0 : shapes.n_prior_tiles],
                indirect_dim=0,
            ),
            axes=(1, 0),
            oob_mode=nisa.oob_mode.skip,
        )


def _causal_mask_bound(
    prior_key_pos: nl.NkiTensor,
    positions_sbuf: nl.NkiTensor,
    token: int,
    shapes: MlaDecodeAttentionShape,
) -> nl.NkiTensor:
    """
    Builds the per-key clamp bound that drops prior keys at or beyond this token's position.

    Returns ``[P_MAX, n_prior_tiles]`` fp32, holding ``+_MASK_BOUND`` where a key is kept and
    ``-_MASK_BOUND`` where it is dropped, so the score loop can fold masking into its scale as a
    single ``minimum``.

    Two instructions cover every selected key of the token, because the predicate is per key and keys
    are partitions. Computes ``keep = prior_key_pos < positions[token]`` then
    ``bound = keep * 2 * _MASK_BOUND - _MASK_BOUND``.
    """

    n_prior_tiles = shapes.n_prior_tiles

    keep_sbuf = nl.ndarray((P_MAX, n_prior_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=keep_sbuf[0:P_MAX, 0:n_prior_tiles],
        data=prior_key_pos[0:P_MAX, token, 0:n_prior_tiles],
        op0=nl.subtract,
        operand0=positions_sbuf[0:P_MAX, token : token + 1],
        op1=nl.less,
        operand1=0.0,
    )

    bound_sbuf = nl.ndarray((P_MAX, n_prior_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=bound_sbuf[0:P_MAX, 0:n_prior_tiles],
        data=keep_sbuf[0:P_MAX, 0:n_prior_tiles],
        op0=nl.multiply,
        operand0=2.0 * _MASK_BOUND,
        op1=nl.subtract,
        operand1=_MASK_BOUND,
    )

    return bound_sbuf


def _score_keys(
    scores_sbuf: nl.NkiTensor,
    latent_keys_sbuf: nl.NkiTensor,
    pe_keys_sbuf: nl.NkiTensor,
    q_absorbed_sbuf: nl.NkiTensor,
    q_pe_sbuf: nl.NkiTensor,
    mask_bound_sbuf: nl.NkiTensor,
    token: int,
    softmax_scale: float,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Scores one token's keys against its query, key-major, scaled and causally masked.

    Writes ``scores_sbuf[p, token, tile, head]`` for key ``tile * P_MAX + p``. The latent and
    positional contractions accumulate into the same PSUM region, so their sum is free.

    Taking the key tile as the stationary operand and the query as the moving one is what lands keys
    on partitions: ``dst[M, N] = stationary[K, M]^T . moving[K, N]``, so ``M`` being the key tile and
    ``N`` the heads transposes the output at no cost.

    Dropped keys are overwritten with ``-_MASK_BOUND`` rather than having a bias added, which is a
    true select and cannot be defeated by a garbage-magnitude score.

    Computes ``scores = softmax_scale * (q_absorbed . kv_latent + q_pe . k_pe)``, clamped.
    """

    n_heads, qk_rope_dim = shapes.n_heads, shapes.qk_rope_dim
    n_prior_tiles, latent_chunk_stride = shapes.n_prior_tiles, shapes.latent_chunk_stride

    for key_tile in TiledRange(shapes.n_keys, P_MAX):
        key_start, key_end, key_size = key_tile.start_offset, key_tile.end_offset, key_tile.size

        score_psum = nl.ndarray((P_MAX, n_heads), dtype=nl.float32, buffer=nl.psum)
        is_first_matmul = True
        for latent_tile in TiledRange(shapes.kv_lora_rank, P_MAX):
            latent_size = latent_tile.size
            base = latent_tile.index * latent_chunk_stride

            nisa.nc_matmul(
                dst=score_psum[0:key_size, 0:n_heads],
                stationary=latent_keys_sbuf[0:latent_size, base + key_start : base + key_end],
                moving=q_absorbed_sbuf[0:latent_size, latent_tile.index, token, 0:n_heads],
                accumulate=not is_first_matmul,
            )
            is_first_matmul = False

        # qk_rope_dim is at most P_MAX, so the positional half is a single contraction.
        nisa.nc_matmul(
            dst=score_psum[0:key_size, 0:n_heads],
            stationary=pe_keys_sbuf[0:qk_rope_dim, key_start:key_end],
            moving=q_pe_sbuf[0:qk_rope_dim, token, 0:n_heads],
            accumulate=True,
        )

        # Tiles beyond the selected prior hold only the current token's key, which is never masked.
        is_masked_tile = key_tile.index < n_prior_tiles
        for head in range(n_heads):
            if is_masked_tile:
                nisa.scalar_tensor_tensor(
                    dst=scores_sbuf[0:key_size, token, key_tile.index, head : head + 1],
                    data=score_psum[0:key_size, head : head + 1],
                    op0=nl.multiply,
                    operand0=softmax_scale,
                    op1=nl.minimum,
                    operand1=mask_bound_sbuf[0:key_size, key_tile.index : key_tile.index + 1],
                )
            else:
                nisa.tensor_scalar(
                    dst=scores_sbuf[0:key_size, token, key_tile.index, head : head + 1],
                    data=score_psum[0:key_size, head : head + 1],
                    op0=nl.multiply,
                    operand0=softmax_scale,
                )


def _reduce_key_tile_max(
    partition_max_sbuf: nl.NkiTensor,
    scores_sbuf: nl.NkiTensor,
    token: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Reduces one token's scores over the key-tile axis, leaving a per-partition max per head.

    The cross-partition half of the max is deferred to the group softmax, which is what lets the
    ten-operation reduction chain run once per group instead of once per token.
    """

    n_heads, n_key_tiles = shapes.n_heads, shapes.n_key_tiles

    for head in range(n_heads):
        column = token * n_heads + head
        nisa.tensor_reduce(
            partition_max_sbuf[0:P_MAX, column : column + 1],
            nl.maximum,
            scores_sbuf[0:P_MAX, token, 0:n_key_tiles, head],
            1,
        )


def _group_neg_max(
    neg_max_sbuf: nl.NkiTensor,
    partition_max_sbuf: nl.NkiTensor,
    ones_row_sbuf: nl.NkiTensor,
    group_start: int,
    group_size: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Finishes the softmax max across partitions for a whole group, and broadcasts it back negated.

    Writes ``neg_max_sbuf[p, token * n_heads + head] = -max(scores[:, token, :, head])``, which the
    exponential then consumes as a bias. Runs once per group: the chain is a transpose, an eviction, a
    reduce, a transpose, an eviction, a broadcast matmul and an eviction, all to produce a handful of
    scalars, and this kernel's cost is the latency of that serial chain.
    """

    n_heads = shapes.n_heads
    group_cols, group_base = group_size * n_heads, group_start * n_heads

    max_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(
        max_psum[0:group_cols, 0:P_MAX], partition_max_sbuf[0:P_MAX, group_base : group_base + group_cols]
    )
    max_rows_sbuf = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=max_rows_sbuf[0:group_cols, 0:P_MAX], src=max_psum[0:group_cols, 0:P_MAX])

    neg_max_col_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        neg_max_col_sbuf[0:group_cols, 0:1], nl.maximum, max_rows_sbuf[0:group_cols, 0:P_MAX], 1, negate=True
    )

    # Back onto the free axis, so the broadcast matmul can replicate it down every partition.
    neg_max_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(neg_max_psum[0:1, 0:group_cols], neg_max_col_sbuf[0:group_cols, 0:1])
    neg_max_row_sbuf = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=neg_max_row_sbuf[0:1, 0:group_cols], src=neg_max_psum[0:1, 0:group_cols])

    broadcast_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=broadcast_psum[0:P_MAX, 0:group_cols],
        stationary=ones_row_sbuf[0:1, 0:P_MAX],
        moving=neg_max_row_sbuf[0:1, 0:group_cols],
    )
    nisa.tensor_copy(
        dst=neg_max_sbuf[0:P_MAX, group_base : group_base + group_cols], src=broadcast_psum[0:P_MAX, 0:group_cols]
    )


def _group_exp_and_inv_denominator(
    weights_sbuf: nl.NkiTensor,
    inv_denom_sbuf: nl.NkiTensor,
    partition_sum_sbuf: nl.NkiTensor,
    scores_sbuf: nl.NkiTensor,
    neg_max_sbuf: nl.NkiTensor,
    ones_partition_sbuf: nl.NkiTensor,
    group_start: int,
    group_size: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Exponentiates a group's scores and builds each token's reciprocal softmax denominator.

    Writes the unnormalized weights and ``inv_denom_sbuf[0, token * n_heads + head]``. The denominator
    is deliberately not applied here: folding it into the latent output afterwards touches
    ``n_heads * kv_lora_rank`` values instead of the ``n_heads * n_keys`` weights.

    Computes ``weights = exp(scores - max)`` and ``inv_denom = 1 / sum(weights)``.
    """

    n_heads, n_key_tiles = shapes.n_heads, shapes.n_key_tiles
    group_cols, group_base = group_size * n_heads, group_start * n_heads

    # Fused exponential and per-partition sum, one instruction per token and head.
    for token in range(group_start, group_start + group_size):
        for head in range(n_heads):
            column = token * n_heads + head
            nisa.activation(
                weights_sbuf[0:P_MAX, token, 0:n_key_tiles, head],
                nl.exp,
                scores_sbuf[0:P_MAX, token, 0:n_key_tiles, head],
                bias=neg_max_sbuf[0:P_MAX, column : column + 1],
                reduce_op=nl.add,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                reduce_res=partition_sum_sbuf[0:P_MAX, column : column + 1],
            )

    # One matmul sums across partitions for every token and head in the group at once.
    sum_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=sum_psum[0:1, 0:group_cols],
        stationary=ones_partition_sbuf[0:P_MAX, 0:1],
        moving=partition_sum_sbuf[0:P_MAX, group_base : group_base + group_cols],
    )
    sum_row_sbuf = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sum_row_sbuf[0:1, 0:group_cols], src=sum_psum[0:1, 0:group_cols])

    nisa.reciprocal(inv_denom_sbuf[0:1, group_base : group_base + group_cols], sum_row_sbuf[0:1, 0:group_cols])


def _weighted_latent(
    latent_out_t_sbuf: nl.NkiTensor,
    weights_sbuf: nl.NkiTensor,
    key_rows_sbuf: nl.NkiTensor,
    inv_denom_sbuf: nl.NkiTensor,
    token: int,
    slot: int,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Sums one token's cached latents weighted by its softmax weights, then stashes the result
    transposed for the batched tail.

    This contraction is over keys, so it wants the latent token-major, which is exactly how the gather
    already holds it: a cache row is ``[k_pe | kv_latent]``, so the latent is a column slice of
    ``key_rows_sbuf``. Feeding it directly is the fused-layout win, and the weights need no transpose
    either because the softmax already produced them key-major.

    Computes ``x = (weights . kv_latent) / denominator``, then writes ``x^T`` into
    ``latent_out_t_sbuf``.
    """

    n_heads, kv_lora_rank = shapes.n_heads, shapes.kv_lora_rank
    n_tokens, k_pe_dim, dtype = shapes.n_tokens, shapes.kv_row_k_pe_dim, shapes.dtype

    latent_out_psum = nl.ndarray((P_MAX, kv_lora_rank), dtype=nl.float32, buffer=nl.psum)
    is_first_matmul = True
    for key_tile in TiledRange(shapes.n_keys, P_MAX):
        key_size = key_tile.size

        nisa.nc_matmul(
            dst=latent_out_psum[0:n_heads, 0:kv_lora_rank],
            stationary=weights_sbuf[0:key_size, token, key_tile.index, 0:n_heads],
            moving=key_rows_sbuf[0:key_size, slot, key_tile.index, k_pe_dim : k_pe_dim + kv_lora_rank],
            accumulate=not is_first_matmul,
        )
        is_first_matmul = False

    """
    The batched softmax leaves the denominators as a row, but a tensor_scalar operand needs one element per partition.
    At n_heads == 1 the slice is already shaped right; above that the head axis has to be flipped onto partitions.
    """
    if n_heads == 1:
        inv_denom_operand = inv_denom_sbuf[0:1, token : token + 1]
    else:
        inv_denom_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(inv_denom_psum[0:n_heads, 0:1], inv_denom_sbuf[0:1, token * n_heads : (token + 1) * n_heads])
        inv_denom_col_sbuf = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=inv_denom_col_sbuf[0:n_heads, 0:1], src=inv_denom_psum[0:n_heads, 0:1])
        inv_denom_operand = inv_denom_col_sbuf[0:n_heads, 0:1]

    latent_out_sbuf = nl.ndarray((P_MAX, kv_lora_rank), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=latent_out_sbuf[0:n_heads, 0:kv_lora_rank],
        data=latent_out_psum[0:n_heads, 0:kv_lora_rank],
        op0=nl.multiply,
        operand0=inv_denom_operand,
    )

    for latent_tile in TiledRange(kv_lora_rank, P_MAX):
        latent_start, latent_end, latent_size = latent_tile.start_offset, latent_tile.end_offset, latent_tile.size

        latent_out_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(
            latent_out_t_psum[0:latent_size, 0:n_heads], latent_out_sbuf[0:n_heads, latent_start:latent_end]
        )

        # Token innermost, so the tail reads all tokens of one (chunk, head) as one contiguous slice.
        for head in range(n_heads):
            column = (latent_tile.index * n_heads + head) * n_tokens + token
            nisa.tensor_copy(
                dst=latent_out_t_sbuf[0:latent_size, column : column + 1],
                src=latent_out_t_psum[0:latent_size, head : head + 1],
            )


def _out_absorb(
    latent_out_t_sbuf: nl.NkiTensor,
    out_absorb_t_sbuf: nl.NkiTensor,
    shapes: MlaDecodeAttentionShape,
) -> nl.NkiTensor:
    """
    Projects every token's latent attention output back out of the cache's latent space.

    Returns ``[P_MAX, n_tokens * n_heads]``, holding head ``head``'s output for all tokens at column
    ``head * n_tokens``. This is the mirror of what the front-fold's absorb did to the query: the
    query expands into latent space on the way in, the value contracts out of it on the way out.

    Batched over tokens, which turns ``n_tokens`` matmuls of stationary width 1 into one of stationary
    width ``n_tokens`` per head and chunk. Computes ``attn = einsum("hc,hdc->hd", x, out_absorb_w)``.
    """

    n_tokens, n_heads = shapes.n_tokens, shapes.n_heads
    v_head_dim, n_latent_tiles = shapes.v_head_dim, shapes.n_latent_tiles

    attn_sbuf = nl.ndarray((P_MAX, n_tokens * n_heads), dtype=shapes.dtype, buffer=nl.sbuf)
    for head in range(n_heads):
        attn_psum = nl.ndarray((P_MAX, n_tokens), dtype=nl.float32, buffer=nl.psum)
        is_first_matmul = True
        for latent_tile in TiledRange(shapes.kv_lora_rank, P_MAX):
            latent_size = latent_tile.size
            weight_base = (head * n_latent_tiles + latent_tile.index) * v_head_dim
            latent_base = (latent_tile.index * n_heads + head) * n_tokens

            nisa.nc_matmul(
                dst=attn_psum[0:v_head_dim, 0:n_tokens],
                stationary=out_absorb_t_sbuf[0:latent_size, weight_base : weight_base + v_head_dim],
                moving=latent_out_t_sbuf[0:latent_size, latent_base : latent_base + n_tokens],
                accumulate=not is_first_matmul,
            )
            is_first_matmul = False

        nisa.tensor_copy(
            dst=attn_sbuf[0:v_head_dim, head * n_tokens : (head + 1) * n_tokens],
            src=attn_psum[0:v_head_dim, 0:n_tokens],
        )

    return attn_sbuf


def _o_proj(
    output: nl.NkiTensor,
    attn_sbuf: nl.NkiTensor,
    o_proj_sbuf: nl.NkiTensor,
    shapes: MlaDecodeAttentionShape,
) -> None:
    """
    Contracts every head's attention output through the o-projection and evicts the result.

    Accumulates over heads into one PSUM bank per ``hidden`` tile, and is batched over tokens for the
    same reason the out-absorb is. Computes ``output = attn.reshape(n_tokens, attn_packed_dim) @
    o_proj_w``.
    """

    n_tokens, n_heads = shapes.n_tokens, shapes.n_heads
    v_head_dim, hidden = shapes.v_head_dim, shapes.hidden

    out_sbuf = nl.ndarray((P_MAX, hidden), dtype=shapes.dtype, buffer=nl.sbuf)
    for hidden_tile in TiledRange(hidden, F_MAX):
        hidden_start, hidden_end, hidden_size = (
            hidden_tile.start_offset,
            hidden_tile.end_offset,
            hidden_tile.size,
        )

        o_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
        is_first_matmul = True
        for head in range(n_heads):
            nisa.nc_matmul(
                dst=o_psum[0:n_tokens, 0:hidden_size],
                stationary=attn_sbuf[0:v_head_dim, head * n_tokens : (head + 1) * n_tokens],
                moving=o_proj_sbuf[0:v_head_dim, head * hidden + hidden_start : head * hidden + hidden_end],
                accumulate=not is_first_matmul,
            )
            is_first_matmul = False

        nisa.tensor_copy(dst=out_sbuf[0:n_tokens, hidden_start:hidden_end], src=o_psum[0:n_tokens, 0:hidden_size])

    nisa.dma_copy(dst=output[0:n_tokens, 0:hidden], src=out_sbuf[0:n_tokens, 0:hidden])
