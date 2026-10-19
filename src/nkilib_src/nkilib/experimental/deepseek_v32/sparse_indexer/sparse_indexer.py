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
DeepSeek-V3.2 sparse indexer decode kernel.

Scores every cached key slot against the current query and selects the highest scoring positions
per token, so that sparse attention reads only those.
"""

from typing import NamedTuple

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.entry_trace import trace_kernel_entry
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ....core.utils.tiled_range import TiledRange
from . import (
    F_MAX,
    K_PACK,
    MASK_BOUND,
    MX_BLOCK,
    P_MAX,
    SCALE_QUADRANT_SIZE,
    SCALES_PER_QUADRANT,
    TOPK_GROUP,
    TOPK_GROUP_SHIFT,
    TOPK_GROUPS_PER_CALL,
    DeepseekV32SparseIndexerShape,
    get_deepseek_v32_sparse_indexer_shape,
)


class DeepseekV32SparseIndexerOutput(NamedTuple):
    scores: nl.NkiTensor
    topk_indices: nl.NkiTensor


@nki.jit
def deepseek_v32_sparse_indexer(
    hidden_states: nl.NkiTensor,  # [n_tokens, hidden]
    qr: nl.NkiTensor,  # [n_tokens, q_lora_rank]   q_norm(wq_a(x)) from the MLA block
    wq_b: nl.NkiTensor,  # [q_lora_rank//K_PACK, n_heads*head_dim] fp8x4, MX-packed along K
    wq_b_scale: nl.NkiTensor,  # [q_lora_rank//MX_BLOCK, n_heads*head_dim] uint8 e8m0 MX scale
    wk: nl.NkiTensor,  # [hidden, head_dim]
    k_norm_weight: nl.NkiTensor,  # [1, head_dim] fp32   LayerNorm gain
    k_norm_bias: nl.NkiTensor,  # [1, head_dim] fp32   LayerNorm bias
    weights_proj: nl.NkiTensor,  # [hidden, n_heads]
    hadamard: nl.NkiTensor,  # [head_dim, head_dim]  1/sqrt(head_dim) pre-baked
    cos: nl.NkiTensor,  # [n_tokens, rope_dim_half]
    sin: nl.NkiTensor,  # [n_tokens, rope_dim_half]
    key_cache: nl.NkiTensor,  # [n_tokens, max_seq_len, head_dim]  caller-owned layout
    positions: nl.NkiTensor,  # [n_tokens] int32   current absolute position per sequence
    index_topk: int,
    valid_len: int | None = None,
) -> DeepseekV32SparseIndexerOutput:
    """
    Score every cached key slot against the current query and select the highest scoring ones.

    This is the DSA lightning indexer for one decode step, optimized for the token-generation
    regime where each sequence contributes a single token: ``n_tokens <= P_MAX``, with 16 at the
    shipping shape. The query projection is always MX-quantized, so it requires trn3 or later.

    Dimensions:
        n_tokens: Decode tokens in the batch, one per sequence
        n_heads: Indexer heads
        head_dim: Per-head feature width, which must equal P_MAX
        hidden: Model hidden width, the contraction dimension of the key projection
        q_lora_rank: LoRA rank of the query path, the contraction dimension of the query projection
        rope_dim: Leading feature span that RoPE rotates
        max_seq_len: Key-cache capacity per sequence, in slots
        index_topk: Slots selected per token

    Args:
        hidden_states (nl.NkiTensor): [n_tokens, hidden], Hidden states for this step
        qr (nl.NkiTensor): [n_tokens, q_lora_rank], ``q_norm(wq_a(x))`` from the MLA block
        wq_b (nl.NkiTensor): [q_lora_rank // K_PACK, n_heads * head_dim], Query projection weight,
            fp8x4 MX-packed along the contraction dimension
        wq_b_scale (nl.NkiTensor): [q_lora_rank // MX_BLOCK, n_heads * head_dim], uint8 e8m0 MX
            scale paired with ``wq_b``
        wk (nl.NkiTensor): [hidden, head_dim], Key projection weight
        k_norm_weight (nl.NkiTensor): [1, head_dim], fp32 LayerNorm gain for the key
        k_norm_bias (nl.NkiTensor): [1, head_dim], fp32 LayerNorm bias for the key
        weights_proj (nl.NkiTensor): [hidden, n_heads], Per-head weight projection
        hadamard (nl.NkiTensor): [head_dim, head_dim], Scaled Hadamard matrix with ``1/sqrt(head_dim)``
            pre-baked
        cos (nl.NkiTensor): [n_tokens, rope_dim // 2], RoPE cosine table
        sin (nl.NkiTensor): [n_tokens, rope_dim // 2], RoPE sine table
        key_cache (nl.NkiTensor): [n_tokens, max_seq_len, head_dim], Caller-owned key cache, written
            in place at each token's current position
        positions (nl.NkiTensor): [n_tokens], int32 absolute position of each sequence's current token
        shapes (DeepseekV32SparseIndexerShape): Input shapes and derived tiling constants

    Returns:
        scores (nl.NkiTensor): [n_tokens, max_seq_len], fp32 per-slot relevance, with slots the
            token cannot attend to clamped to the mask bound
        topk_indices (nl.NkiTensor): [n_tokens, index_topk], int32 selected slot positions

    Notes:
        - ``topk_indices`` is an unordered set: column order carries no meaning.
        - The mask bound is finite rather than ``-inf`` because the top-k consumer reads bfloat16,
          where ``-inf`` makes the hardware max-select produce NaN.
        - ``q_lora_rank`` must be a multiple of ``P_MAX * K_PACK``, the MX K-tile.

    Pseudocode:
        query = hadamard.T @ rope(qr @ wq_b)
        key = rope(layer_norm(hidden_states @ wk)) @ hadamard
        key_cache[token, positions[token]] = key[token]
        head_weight = (weights_proj.T @ hidden_states) * n_heads_scale * softmax_scale
        for token in range(n_tokens):
            raw = key_cache[token] @ query[token]
            scores[token] = sum_head(relu(raw) * head_weight[token])
            scores[token] = mask(scores[token], slot > positions[token])
        topk_indices = topk(scores, index_topk)
    """

    trace_kernel_entry("deepseek_v32_sparse_indexer", locals())

    shapes = get_deepseek_v32_sparse_indexer_shape(
        hidden_states, qr, wk, weights_proj, cos, key_cache, index_topk, valid_len
    )

    n_tokens, head_dim, n_heads, q_lora_rank, rope_dim = (
        shapes.n_tokens,
        shapes.head_dim,
        shapes.n_heads,
        shapes.q_lora_rank,
        shapes.rope_dim,
    )

    kernel_assert(head_dim == P_MAX, f"head_dim={head_dim} must be {P_MAX}")
    kernel_assert(n_heads <= P_MAX, f"n_heads={n_heads} must be <= {P_MAX}")
    kernel_assert(n_tokens <= P_MAX, f"n_tokens={n_tokens} must be <= {P_MAX}")
    kernel_assert(rope_dim <= head_dim, f"rope_dim={rope_dim} must be <= head_dim={head_dim}")
    kernel_assert(
        q_lora_rank % (P_MAX * K_PACK) == 0,
        f"q_lora_rank={q_lora_rank} must be a multiple of the MX K-tile {P_MAX * K_PACK}",
    )

    wk_tiled_sbuf = _load_k_tiled(wk)
    weights_proj_tiled_sbuf = _load_k_tiled(weights_proj)
    hadamard_sbuf = _load_hadamard(hadamard)
    k_norm_weight_bcast_sbuf = _load_broadcast_row_to_tokens(k_norm_weight, n_tokens)
    k_norm_bias_bcast_sbuf = _load_broadcast_row_to_tokens(k_norm_bias, n_tokens)
    cos_sbuf, sin_sbuf = _load_rope_tables(cos, sin, n_heads)
    cos_token_major_sbuf, sin_token_major_sbuf = _load_rope_tables_token_major(cos, sin)
    hidden_sbuf = _load_m_major_then_transpose(hidden_states)

    query_sbuf = _project_and_rotate_query(qr, wq_b, wq_b_scale, hadamard_sbuf, cos_sbuf, sin_sbuf, shapes)
    cache_rows = _project_and_rotate_key(
        hidden_sbuf,
        wk_tiled_sbuf,
        k_norm_weight_bcast_sbuf,
        k_norm_bias_bcast_sbuf,
        cos_token_major_sbuf,
        sin_token_major_sbuf,
        hadamard_sbuf,
        key_cache,
        positions,
        shapes,
    )
    scores = _score_all_tokens(cache_rows, query_sbuf, hidden_sbuf, weights_proj_tiled_sbuf, positions, shapes)
    topk_indices = _select_topk_indices(scores, shapes)
    return DeepseekV32SparseIndexerOutput(scores=scores, topk_indices=topk_indices)


# Query Step


def _project_and_rotate_query(
    qr: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    wq_b_scale: nl.NkiTensor,
    hadamard_sbuf: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    shapes: DeepseekV32SparseIndexerShape,
) -> nl.NkiTensor:
    """
    Project ``qr`` into the indexer query, rotate it with RoPE and apply the Hadamard transform,
    returning a feature-major query in SBUF whose columns are ordered ``head * n_tokens + token``.
    """

    n_tokens, dtype = shapes.n_tokens, shapes.dtype
    head_dim, rope_dim, n_tokens_across_heads = shapes.head_dim, shapes.rope_dim, shapes.n_tokens_across_heads

    qr_quantized_sbuf, qr_scale_sbuf = _quantize_qr_for_mx(qr)
    query_sbuf = _project_query_mx(qr_quantized_sbuf, qr_scale_sbuf, wq_b, wq_b_scale, head_dim, n_tokens)
    _rope_inplace(query_sbuf, cos_sbuf, sin_sbuf, n_tokens_across_heads, rope_dim)
    return _apply_hadamard(query_sbuf, hadamard_sbuf, head_dim, n_tokens_across_heads, dtype)


def _quantize_qr_for_mx(qr: nl.NkiTensor) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Quantize a ``[M, K]`` tensor to block-32 fp8 for ``nki.isa.nc_matmul_mx``, returning the packed
    ``fp8x4`` values and their scales in SBUF with the contraction dimension on partitions. One
    e8m0 exponent is shared per ``MX_BLOCK`` contraction elements.
    """

    M, K = qr.shape[0], qr.shape[1]
    mx_k_tiles = K // (P_MAX * K_PACK)

    # Load ``[M, K]`` as bf16 directly, since the DMA casts f32 to bf16 on the wire.
    qr_sbuf = nl.ndarray((P_MAX, K), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qr_sbuf[0:M, 0:K], src=qr[0:M, 0:K])

    """
    Transpose and swizzle so each group of K_PACK free elements holds the sub-elements of one
    column, matching quantize_mx's 32-partition by 4-free MX block. The access pattern is
    [stride, count] per axis: [K, P_MAX] steps one full row of ``qr`` per partition, and
    [K_PACK, P_MAX] steps K_PACK elements along the free axis. Source and destination walk the
    same pattern, so only the transpose moves data.
    """
    swizzle = [[K, P_MAX], [K_PACK, P_MAX]]
    swizzled_sbuf = nl.ndarray((P_MAX, mx_k_tiles, P_MAX * K_PACK), dtype=nl.bfloat16, buffer=nl.sbuf)

    # One PSUM bank per sub-element, so the K_PACK transposes issue back to back instead of each
    # waiting on the previous eviction.
    transpose_psum = []
    for _ in range(K_PACK):
        transpose_psum.append(nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum))

    for k_tile in range(mx_k_tiles):
        for k_sub in range(K_PACK):
            nisa.nc_transpose(
                dst=transpose_psum[k_sub][0:P_MAX, 0:P_MAX],
                data=qr_sbuf.ap(pattern=swizzle, offset=k_tile * P_MAX * K_PACK + k_sub),
            )
        for k_sub in range(K_PACK):
            nisa.tensor_copy(
                dst=swizzled_sbuf.ap(pattern=swizzle, offset=k_tile * P_MAX * K_PACK + k_sub),
                src=transpose_psum[k_sub][0:P_MAX, 0:P_MAX],
            )

    quantized_sbuf = nl.ndarray((P_MAX, mx_k_tiles, P_MAX), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    scale_sbuf = nl.ndarray((P_MAX, mx_k_tiles, P_MAX), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(
        src=swizzled_sbuf[0:P_MAX, 0:mx_k_tiles, 0 : P_MAX * K_PACK],
        dst=quantized_sbuf[0:P_MAX, 0:mx_k_tiles, 0:P_MAX],
        dst_scale=scale_sbuf[0:P_MAX, 0:mx_k_tiles, 0:P_MAX],
    )

    return quantized_sbuf, scale_sbuf


def _project_query_mx(
    qr_quantized_sbuf: nl.NkiTensor,
    qr_scale_sbuf: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    wq_b_scale: nl.NkiTensor,
    head_dim: int,
    n_tokens: int,
) -> nl.NkiTensor:
    """
    Project the quantized ``qr`` through the MX-packed weight ``wq_b`` into a feature-major query of
    shape ``[P_MAX, n_tokens_across_heads]``. MX packs along the contraction dimension only, so
    ``wq_b.shape[1]`` is the unpacked ``n_heads * head_dim``.

    ``query = (qr @ wq_b).T`` per head
    """

    total_out = wq_b.shape[1]
    mx_k_tiles = qr_quantized_sbuf.shape[1]
    n_tokens_across_heads = (total_out // head_dim) * n_tokens

    """
    ``nc_matmul_mx`` consumes the stationary operand in packed element pairs, so its free dimension
    must be even. Round the token count up for the matmul and drop the extra PSUM row when
    evicting.
    """
    padded_tokens = n_tokens + (n_tokens % 2)

    """
    Weights land one K-tile per DMA rather than in one 3-D transfer, so the first matmul depends
    only on tile 0 and compute starts after roughly 1/mx_k_tiles of the weight bytes instead of
    all of them. The access pattern is [stride, count] per axis: [total_out, P_MAX] puts one packed
    K row on each partition, and [1, total_out] reads total_out contiguous fp8x4 words from it.
    The offset advances a whole K-tile of P_MAX rows.
    """
    wq_b_sbuf = nl.ndarray((P_MAX, mx_k_tiles, total_out), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    for k_tile in range(mx_k_tiles):
        nisa.dma_copy(
            dst=wq_b_sbuf[0:P_MAX, k_tile, 0:total_out],
            src=wq_b.ap(
                pattern=[[total_out, P_MAX], [1, total_out]],
                offset=k_tile * P_MAX * total_out,
                dtype=nl.float8_e4m3fn_x4,
            ),
        )

    wq_b_scale_sbuf = nl.ndarray((P_MAX, mx_k_tiles, total_out), dtype=nl.uint8, buffer=nl.sbuf)
    _load_mx_weight_scale(wq_b_scale, wq_b_scale_sbuf, mx_k_tiles, total_out)

    query_sbuf = nl.ndarray((P_MAX, n_tokens_across_heads), dtype=nl.float32, buffer=nl.sbuf)
    query_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
    query_tile_sbuf = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.sbuf)
    query_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)

    for out_tile in TiledRange(total_out, F_MAX):
        out_start, out_size = out_tile.start_offset, out_tile.size

        for k_tile in range(mx_k_tiles):
            nisa.nc_matmul_mx(
                dst=query_psum[0:padded_tokens, 0:out_size],
                stationary=qr_quantized_sbuf[0:P_MAX, k_tile, 0:padded_tokens],
                moving=wq_b_sbuf[0:P_MAX, k_tile, out_start : out_start + out_size],
                stationary_scale=qr_scale_sbuf[0:P_MAX, k_tile, 0:padded_tokens],
                moving_scale=wq_b_scale_sbuf[0:P_MAX, k_tile, out_start : out_start + out_size],
                accumulate=(k_tile != 0),
            )
        nisa.tensor_copy(dst=query_tile_sbuf[0:n_tokens, 0:out_size], src=query_psum[0:n_tokens, 0:out_size])

        # Transpose each head into feature-major columns. ``out_start`` already encodes the tile's
        # first head, so no separate heads-per-tile term is needed.
        for head_in_tile in range(out_size // head_dim):
            head_start = head_in_tile * head_dim
            col = (out_start // head_dim + head_in_tile) * n_tokens
            nisa.nc_transpose(
                query_t_psum[0:head_dim, 0:n_tokens],
                query_tile_sbuf[0:n_tokens, head_start : head_start + head_dim],
            )
            nisa.tensor_copy(
                dst=query_sbuf[0:head_dim, col : col + n_tokens],
                src=query_t_psum[0:head_dim, 0:n_tokens],
            )

    return query_sbuf


def _load_mx_weight_scale(scale: nl.NkiTensor, scale_sbuf: nl.NkiTensor, mx_k_tiles: int, N: int) -> None:
    """Load the e8m0 MX weight scales from HBM to SBUF of shape ``[P_MAX, mx_k_tiles, N]``."""

    scale_rows_per_k_tile = (P_MAX * K_PACK) // MX_BLOCK
    n_quadrants = P_MAX // SCALE_QUADRANT_SIZE

    nisa.memset(dst=scale_sbuf[0:P_MAX, 0:mx_k_tiles, 0:N], value=0)

    """
    Split per K-tile so the first matmul waits only on tile 0's scales. A transfer spanning every
    K-tile made it wait on all of them, and the weights already arrive one K-tile at a time. The
    quadrant loop stays because SBUF access patterns require the partition stride to equal the
    free size, so the four quadrant heads (partitions 0, 32, 64, 96) cannot be walked in one
    transfer. The access pattern is [stride, count] per axis: [N, SCALES_PER_QUADRANT] puts
    SCALES_PER_QUADRANT scale rows on consecutive partitions, and [1, N] reads N contiguous scale
    bytes from each. The offset selects this K-tile's quadrant head out of the flat scale rows.
    """
    for k_tile in range(mx_k_tiles):
        for quadrant in range(n_quadrants):
            part_start = quadrant * SCALE_QUADRANT_SIZE
            nisa.dma_copy(
                dst=scale_sbuf[part_start : part_start + SCALES_PER_QUADRANT, k_tile, 0:N],
                src=scale.ap(
                    pattern=[[N, SCALES_PER_QUADRANT], [1, N]],
                    offset=(k_tile * scale_rows_per_k_tile + quadrant * SCALES_PER_QUADRANT) * N,
                    dtype=nl.uint8,
                ),
            )


def _rope_inplace(
    feat_major_sbuf: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    n_cols: int,
    rope_dim: int,
) -> None:
    """
    Apply non-interleaved RoPE to the rope span, in place, feature-major. Feature ``i`` pairs with
    feature ``i + rope_dim_half``, so the high half starts on partition ``rope_dim_half`` and is
    staged down to partition 0 first.

    ``low, high = low * cos - high * sin, high * cos + low * sin``
    """

    rope_dim_half = rope_dim // 2
    low = feat_major_sbuf[0:rope_dim_half, 0:n_cols]
    high = feat_major_sbuf[rope_dim_half:rope_dim, 0:n_cols]
    cos_cols = cos_sbuf[0:rope_dim_half, 0:n_cols]
    sin_cols = sin_sbuf[0:rope_dim_half, 0:n_cols]

    high_at0 = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    low_cos = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    high_sin = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    rotated_high = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
    high_v = high_at0[0:rope_dim_half, 0:n_cols]
    low_cos_v = low_cos[0:rope_dim_half, 0:n_cols]
    high_sin_v = high_sin[0:rope_dim_half, 0:n_cols]
    rotated_high_v = rotated_high[0:rope_dim_half, 0:n_cols]

    nisa.tensor_copy(dst=high_v, src=high)  # stage the high half to partition 0
    nisa.tensor_tensor(dst=low_cos_v, data1=low, data2=cos_cols, op=nl.multiply)
    nisa.tensor_tensor(dst=high_sin_v, data1=high_v, data2=sin_cols, op=nl.multiply)
    nisa.tensor_tensor(dst=rotated_high_v, data1=high_v, data2=cos_cols, op=nl.multiply)

    # high_v is dead now, so it carries low * sin
    nisa.tensor_tensor(dst=high_v, data1=low, data2=sin_cols, op=nl.multiply)
    nisa.tensor_tensor(dst=rotated_high_v, data1=rotated_high_v, data2=high_v, op=nl.add)
    nisa.tensor_tensor(dst=low, data1=low_cos_v, data2=high_sin_v, op=nl.subtract)
    nisa.tensor_copy(dst=high, src=rotated_high_v)


def _apply_hadamard(
    query_sbuf: nl.NkiTensor,
    hadamard_sbuf: nl.NkiTensor,
    head_dim: int,
    n_tokens_across_heads: int,
    dtype: nl.DType,
) -> nl.NkiTensor:
    """
    Apply the Hadamard transform to the feature-major query, tiled over its columns.

    ``query = hadamard.T @ query``
    """

    out_sbuf = nl.ndarray((P_MAX, n_tokens_across_heads), dtype=dtype, buffer=nl.sbuf)
    out_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
    for col_tile in TiledRange(n_tokens_across_heads, F_MAX):
        col_start, col_size = col_tile.start_offset, col_tile.size
        nisa.nc_matmul(
            dst=out_psum[0:head_dim, 0:col_size],
            stationary=hadamard_sbuf[0:head_dim, 0:head_dim],
            moving=query_sbuf[0:head_dim, col_start : col_start + col_size],
            accumulate=False,
        )
        nisa.tensor_copy(
            dst=out_sbuf[0:head_dim, col_start : col_start + col_size],
            src=out_psum[0:head_dim, 0:col_size],
        )

    return out_sbuf


# Key Step


def _project_and_rotate_key(
    hidden_sbuf: nl.NkiTensor,
    wk_tiled_sbuf: nl.NkiTensor,
    k_norm_weight_bcast_sbuf: nl.NkiTensor,
    k_norm_bias_bcast_sbuf: nl.NkiTensor,
    cos_token_major_sbuf: nl.NkiTensor,
    sin_token_major_sbuf: nl.NkiTensor,
    hadamard_sbuf: nl.NkiTensor,
    key_cache: nl.NkiTensor,
    positions: nl.NkiTensor,
    shapes: DeepseekV32SparseIndexerShape,
) -> nl.NkiTensor:
    """
    Project the hidden states into the indexer key, normalize and rotate it, apply the Hadamard
    transform, then scatter it into the caller's key cache. The key stays token-major throughout.
    """

    n_tokens, max_seq_len, head_dim = shapes.n_tokens, shapes.max_seq_len, shapes.head_dim
    hidden, rope_dim, dtype = shapes.hidden, shapes.rope_dim, shapes.dtype

    key_sbuf = _project_key(hidden_sbuf, wk_tiled_sbuf, hidden)
    _layernorm_key_inplace(key_sbuf, k_norm_weight_bcast_sbuf, k_norm_bias_bcast_sbuf, n_tokens)
    _rope_key_inplace(key_sbuf, cos_token_major_sbuf, sin_token_major_sbuf, n_tokens, rope_dim)
    key_sbuf = _apply_hadamard_token_major(key_sbuf, hadamard_sbuf, head_dim, n_tokens, dtype)

    """
    Flatten the cache to rows so that a slot index is a row index. The same flattened handle must
    serve both this write and the score's reads, otherwise the compiler cannot see the dependency
    and the score reads a stale slot for the current token.
    """
    cache_rows = key_cache.reshape((n_tokens * max_seq_len, head_dim))
    _write_key_cache(cache_rows, key_sbuf, positions, n_tokens, max_seq_len)

    return cache_rows


def _project_key(hidden_sbuf: nl.NkiTensor, wk_tiled_sbuf: nl.NkiTensor, K: int) -> nl.NkiTensor:
    """
    Project the hidden states through ``wk`` into a token-major key of shape ``[M, N]``, where ``M``
    is the token count and ``N`` the head dimension.

    Both operands arrive contraction-tiled as ``[P_MAX, n_tiles, free]`` with the contraction
    dimension on partitions, so ``hidden_sbuf`` supplies the ``M`` output partitions and ``wk_tiled_sbuf``
    the ``N`` output columns, matching ``nki.isa.nc_matmul``'s ``stationary.T @ moving``. ``K`` is
    the contraction extent and must be passed: the tiled operands only carry ``n_tiles``, which
    rounds ``K`` up to a multiple of ``P_MAX``.

    ``key = hidden.T @ wk``
    """

    M, N = hidden_sbuf.shape[2], wk_tiled_sbuf.shape[2]

    key_sbuf = nl.ndarray((P_MAX, N), dtype=nl.float32, buffer=nl.sbuf)
    key_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)

    first_tile = True
    for tile in TiledRange(K, P_MAX):
        size = tile.size
        nisa.nc_matmul(
            dst=key_psum[0:M, 0:N],
            stationary=hidden_sbuf[0:size, tile.index, 0:M],
            moving=wk_tiled_sbuf[0:size, tile.index, 0:N],
            accumulate=not first_tile,
        )
        first_tile = False

    nisa.tensor_copy(
        dst=key_sbuf[0:M, 0:N],
        src=key_psum[0:M, 0:N],
    )

    return key_sbuf


def _layernorm_key_inplace(
    key_sbuf: nl.NkiTensor,
    key_norm_weight_sbuf: nl.NkiTensor,
    bias_sbuf: nl.NkiTensor,
    n_tokens: int,
    eps: float = 1e-6,
) -> None:
    """
    Apply an fp32 LayerNorm over the head dimension, in place.

    This is a LayerNorm and not an RMSNorm: it subtracts the mean and adds a bias. The mean needs
    the ``1 / head_dim`` factor that the variance folds into its rsqrt scale.

    ``key = (key - mean(key)) * rsqrt(var(key) + eps) * gain + bias``
    """

    head_dim = key_sbuf.shape[1]
    scratch = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    stat = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=scratch[0:n_tokens, 0:head_dim],
        op=nl.copy,
        data=key_sbuf[0:n_tokens, 0:head_dim],
        reduce_op=nl.add,
        reduce_res=stat[0:n_tokens, 0:1],
    )

    # activation_reduce returns the sum, so scale it to the mean here. The variance's 1 / head_dim
    # rides the rsqrt scale below.
    nisa.tensor_scalar(
        dst=stat[0:n_tokens, 0:1],
        data=stat[0:n_tokens, 0:1],
        op0=nl.multiply,
        operand0=1.0 / head_dim,
    )
    nisa.tensor_scalar(
        dst=key_sbuf[0:n_tokens, 0:head_dim],
        data=key_sbuf[0:n_tokens, 0:head_dim],
        op0=nl.subtract,
        operand0=stat[0:n_tokens, 0:1],
    )
    nisa.activation_reduce(
        dst=scratch[0:n_tokens, 0:head_dim],
        op=nl.square,
        data=key_sbuf[0:n_tokens, 0:head_dim],
        reduce_op=nl.add,
        reduce_res=stat[0:n_tokens, 0:1],
    )
    nisa.activation(
        stat[0:n_tokens, 0:1],
        op=nl.rsqrt,
        data=stat[0:n_tokens, 0:1],
        scale=1.0 / head_dim,
        bias=eps,
    )
    nisa.scalar_tensor_tensor(
        dst=key_sbuf[0:n_tokens, 0:head_dim],
        data=key_sbuf[0:n_tokens, 0:head_dim],
        op0=nl.multiply,
        operand0=stat[0:n_tokens, 0:1],
        op1=nl.multiply,
        operand1=key_norm_weight_sbuf[0:n_tokens, 0:head_dim],
    )
    nisa.tensor_tensor(
        dst=key_sbuf[0:n_tokens, 0:head_dim],
        data1=key_sbuf[0:n_tokens, 0:head_dim],
        data2=bias_sbuf[0:n_tokens, 0:head_dim],
        op=nl.add,
    )


def _rope_key_inplace(
    key_sbuf: nl.NkiTensor,
    cos_sbuf: nl.NkiTensor,
    sin_sbuf: nl.NkiTensor,
    n_tokens: int,
    rope_dim: int,
) -> None:
    """
    Apply non-interleaved RoPE to the key's rope span, in place, token-major. Feature ``i`` pairs
    with feature ``i + rope_dim_half``.

    ``low, high = low * cos - high * sin, high * cos + low * sin``
    """

    rope_dim_half = rope_dim // 2
    low = key_sbuf[0:n_tokens, 0:rope_dim_half]
    high = key_sbuf[0:n_tokens, rope_dim_half:rope_dim]
    cos_cols = cos_sbuf[0:n_tokens, 0:rope_dim_half]
    sin_cols = sin_sbuf[0:n_tokens, 0:rope_dim_half]

    low_cos = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    high_sin = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    rotated_high = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    low_cos_v = low_cos[0:n_tokens, 0:rope_dim_half]
    high_sin_v = high_sin[0:n_tokens, 0:rope_dim_half]
    rotated_high_v = rotated_high[0:n_tokens, 0:rope_dim_half]

    # Token-major needs no staging copy: both halves already start on partition 0, unlike the
    # feature-major query where the high half sits on partitions rope_dim_half and up.
    nisa.tensor_tensor(dst=low_cos_v, data1=low, data2=cos_cols, op=nl.multiply)
    nisa.tensor_tensor(dst=high_sin_v, data1=high, data2=sin_cols, op=nl.multiply)
    nisa.tensor_tensor(dst=rotated_high_v, data1=high, data2=cos_cols, op=nl.multiply)

    # high is dead now, so it carries low * sin
    nisa.tensor_tensor(dst=high, data1=low, data2=sin_cols, op=nl.multiply)
    nisa.tensor_tensor(dst=high, data1=rotated_high_v, data2=high, op=nl.add)
    nisa.tensor_tensor(dst=low, data1=low_cos_v, data2=high_sin_v, op=nl.subtract)


def _apply_hadamard_token_major(
    key_sbuf: nl.NkiTensor,
    hadamard_sbuf: nl.NkiTensor,
    head_dim: int,
    n_tokens: int,
    dtype: nl.DType,
) -> nl.NkiTensor:
    """
    Apply the Hadamard transform to the token-major key, returning it token-major.

    The contraction runs over features, so the key is transposed first, and feeding the transposed
    key as the stationary operand returns the result token-major without a second transpose.

    ``key = key @ hadamard``
    """

    # Transpose the keys.
    key_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    key_t_sbuf = nl.ndarray((P_MAX, n_tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(key_t_psum[0:head_dim, 0:n_tokens], key_sbuf[0:n_tokens, 0:head_dim])
    nisa.tensor_copy(
        dst=key_t_sbuf[0:head_dim, 0:n_tokens],
        src=key_t_psum[0:head_dim, 0:n_tokens],
    )

    # Matmul the transposed keys with the hadamard tensor.
    out_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=out_psum[0:n_tokens, 0:head_dim],
        stationary=key_t_sbuf[0:head_dim, 0:n_tokens],
        moving=hadamard_sbuf[0:head_dim, 0:head_dim],
        accumulate=False,
    )

    out_sbuf = nl.ndarray((P_MAX, head_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sbuf[0:n_tokens, 0:head_dim], src=out_psum[0:n_tokens, 0:head_dim])
    return out_sbuf


def _write_key_cache(
    cache_rows: nl.NkiTensor,
    key_sbuf: nl.NkiTensor,
    positions: nl.NkiTensor,
    n_tokens: int,
    max_seq_len: int,
) -> None:
    """
    Scatter each token's key into ``key_cache[token, positions[token], :]``. The cache arrives
    flattened to rows so that a slot index is a row index, which makes the flat row for a token
    ``token * max_seq_len + positions[token]``. That row index is built on chip with an iota, so a
    single indirect DMA covers the whole batch.
    """

    head_dim = key_sbuf.shape[1]

    flat_row = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    token_base = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=flat_row[0:n_tokens, 0:1], src=positions[0:n_tokens].reshape((n_tokens, 1)))
    nisa.iota(token_base[0:n_tokens, 0:1], pattern=[[1, 1]], channel_multiplier=max_seq_len, offset=0)
    nisa.tensor_tensor(
        dst=flat_row[0:n_tokens, 0:1],
        data1=flat_row[0:n_tokens, 0:1],
        data2=token_base[0:n_tokens, 0:1],
        op=nl.add,
    )

    """
    The access pattern [head_dim, n_tokens] places one token on each partition, and [1, head_dim] writes
    head_dim contiguous features from it. The ``vector_offset`` supplies the per-partition destination row,
    which is what makes this a scatter, and ``indirect_dim=0`` applies it to the row axis. Rows past the end
    of the cache are skipped rather than clamped.
    """
    nisa.dma_copy(
        dst=cache_rows.ap(
            pattern=[[head_dim, n_tokens], [1, head_dim]],
            offset=0,
            vector_offset=flat_row[0:n_tokens, 0:1].ap(pattern=[[1, n_tokens], [1, 1]], offset=0),
            indirect_dim=0,
        ),
        src=key_sbuf[0:n_tokens, 0:head_dim],
        oob_mode=nisa.oob_mode.skip,
    )


# Score Step


def _score_all_tokens(
    cache_rows: nl.NkiTensor,
    query_sbuf: nl.NkiTensor,
    hidden_sbuf: nl.NkiTensor,
    weights_proj_sbuf: nl.NkiTensor,
    positions: nl.NkiTensor,
    shapes: DeepseekV32SparseIndexerShape,
) -> nl.NkiTensor:
    """
    Score every cache slot of every sequence against that sequence's query, returning a
    ``[n_tokens, max_seq_len]`` score tensor in HBM.

    ``cache_rows`` must be the same flattened handle the key write used, otherwise the compiler
    cannot see the dependency and the current token's slot reads stale.
    """

    n_tokens, max_seq_len, n_key_tiles = shapes.n_tokens, shapes.max_seq_len, shapes.n_key_tiles

    head_weights_sbuf = _project_head_weights(
        hidden_sbuf, weights_proj_sbuf, shapes.hidden, shapes.n_heads_scale * shapes.softmax_scale
    )
    slot_id_sbuf = _load_slot_ids(n_key_tiles)
    positions_bcast_sbuf = _broadcast_positions(positions, n_tokens)

    score = nl.ndarray((n_tokens, max_seq_len), dtype=nl.float32, buffer=nl.shared_hbm)
    score_psum = nl.ndarray((P_MAX, n_key_tiles), dtype=nl.float32, buffer=nl.psum)

    # Every token's transposed score, so the whole batch leaves in one DMA.
    score_t_all = nl.ndarray((P_MAX, n_tokens, P_MAX), dtype=nl.float32, buffer=nl.sbuf)

    for token in range(n_tokens):
        _score_one_token(cache_rows, query_sbuf, head_weights_sbuf, score_psum, token, shapes)
        _mask_score(score_psum, score_t_all, slot_id_sbuf, positions_bcast_sbuf, token, shapes.n_scored_tiles)

    """
    Slot ``tile * P_MAX + p`` of token ``t`` belongs at ``score[t, tile * P_MAX + p]``. The access
    pattern is [stride, count] per axis: the partition axis is the tile, then tokens stride by
    max_seq_len, then the P_MAX slots within a tile are contiguous.
    """
    nisa.dma_copy(
        dst=score.reshape_dim(1, (n_key_tiles, P_MAX)).permute((1, 0, 2)),
        src=score_t_all[0:n_key_tiles, 0:n_tokens, 0:P_MAX],
    )

    return score


def _project_head_weights(
    hidden_sbuf: nl.NkiTensor,
    weights_proj_sbuf: nl.NkiTensor,
    K: int,
    scale: float,
) -> nl.NkiTensor:
    """
    Project the hidden states through ``weights_proj`` into per-head weights of shape ``[M, N]``,
    where ``M`` is the head count and ``N`` the token count.

    Both operands arrive contraction-tiled as ``[P_MAX, n_tiles, free]`` with the contraction
    dimension on partitions. ``weights_proj_sbuf`` is the stationary operand here, so it supplies the
    ``M`` output partitions and the result comes out head-major, transposed relative to the key
    projection. ``scale`` folds into the eviction rather than costing a second pass.

    ``head_weight = (weights_proj.T @ hidden) * scale``
    """

    M, N = weights_proj_sbuf.shape[2], hidden_sbuf.shape[2]

    weights_sbuf = nl.ndarray((P_MAX, N), dtype=nl.float32, buffer=nl.sbuf)
    weights_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)

    first_tile = True
    for tile in TiledRange(K, P_MAX):
        size = tile.size
        nisa.nc_matmul(
            dst=weights_psum[0:M, 0:N],
            stationary=weights_proj_sbuf[0:size, tile.index, 0:M],
            moving=hidden_sbuf[0:size, tile.index, 0:N],
            accumulate=not first_tile,
        )
        first_tile = False

    nisa.tensor_scalar(
        dst=weights_sbuf[0:M, 0:N],
        data=weights_psum[0:M, 0:N],
        op0=nl.multiply,
        operand0=scale,
    )

    return weights_sbuf


def _score_one_token(
    cache_rows: nl.NkiTensor,
    query_sbuf: nl.NkiTensor,
    head_weights_sbuf: nl.NkiTensor,
    score_psum: nl.NkiTensor,
    token: int,
    shapes: DeepseekV32SparseIndexerShape,
) -> None:
    """
    Score one sequence's cache slots against its query, accumulating into ``score_psum``.

    Slots stay on partitions so that the mask downstream runs on all P_MAX lanes, and the weighted
    head-sum is a free-axis reduction on the Vector engine rather than a matmul on the busier Tensor
    engine.

    ``score[slot] = sum_head relu(<query[head], key_cache[slot]>) * head_weight[head]``
    """

    n_heads, head_dim, dtype = shapes.n_heads, shapes.head_dim, shapes.dtype
    max_seq_len, n_tokens = shapes.max_seq_len, shapes.n_tokens
    n_scored_tiles, n_tokens_across_heads = shapes.n_scored_tiles, shapes.n_tokens_across_heads

    # Slice the flattened handle rather than re-deriving one, so the read-after-write on this
    # token's key stays visible to the compiler.
    slab = cache_rows[token * max_seq_len : (token + 1) * max_seq_len]

    # [head_dim, P_MAX] puts one slot on each partition, [P_MAX * head_dim, n_scored_tiles] advances a
    # whole tile of slots, and [1, head_dim] reads the slot's features contiguously.
    slots_sbuf = nl.ndarray((P_MAX, n_scored_tiles, head_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=slots_sbuf[0:P_MAX, 0:n_scored_tiles, 0:head_dim],
        src=slab[0 : n_scored_tiles * P_MAX].reshape_dim(0, (n_scored_tiles, P_MAX)).permute((1, 0, 2)),
    )

    raw_score_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)

    # Chunk-wide, so each sub-tile transposes into its own columns.
    slots_t_psum = nl.ndarray((P_MAX, F_MAX), dtype=dtype, buffer=nl.psum)
    slots_t_sbuf = nl.ndarray((P_MAX, F_MAX), dtype=dtype, buffer=nl.sbuf)
    relu_sbuf = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.sbuf)
    weighted_sbuf = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)

    # weight_bcast[slot, head] = head_weight[head, token], identical for every slot, so the head-sum can multiply
    # relu[slot, head] by it and reduce over the free (head) axis.
    weight_col_sbuf = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    weight_bcast_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    weight_bcast_sbuf = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)

    nisa.tensor_copy(
        dst=weight_col_sbuf[0:n_heads, 0:P_MAX],
        src=head_weights_sbuf[0:n_heads, token : token + 1].broadcast(1, P_MAX),
    )
    nisa.nc_transpose(weight_bcast_psum[0:P_MAX, 0:n_heads], weight_col_sbuf[0:n_heads, 0:P_MAX])
    nisa.tensor_copy(dst=weight_bcast_sbuf[0:P_MAX, 0:n_heads], src=weight_bcast_psum[0:P_MAX, 0:n_heads])

    # Query columns are head-major (``head * n_tokens + token``), so this token's heads are strided
    # by n_tokens rather than contiguous.
    head_cols = query_sbuf[0:head_dim].reshape_dim(1, (n_heads, n_tokens)).select(2, token)

    scored_len = n_scored_tiles * P_MAX
    for chunk_tile in TiledRange(scored_len, F_MAX):
        chunk = chunk_tile.size

        for sub in TiledRange(chunk_tile, P_MAX):
            # start_offset stays absolute when nested, so it names the slab tile directly, while
            # slots_t_psum is chunk-sized and needs the chunk-relative offset.
            sub_len, sub_off = sub.size, sub.start_offset - chunk_tile.start_offset
            nisa.nc_transpose(
                slots_t_psum[0:head_dim, sub_off : sub_off + sub_len],
                slots_sbuf[0:sub_len, sub.start_offset // P_MAX, 0:head_dim],
            )

        # One eviction for the whole chunk rather than one per sub-tile.
        nisa.tensor_copy(
            dst=slots_t_sbuf[0:head_dim, 0:chunk],
            src=slots_t_psum[0:head_dim, 0:chunk],
        )

        for sub in TiledRange(chunk_tile, P_MAX):
            sub_len, sub_off = sub.size, sub.start_offset - chunk_tile.start_offset
            tile = sub.start_offset // P_MAX

            # dst[slot, head] = <key[slot], query[head]>, with accumulate=False as raw_score_psum is
            # reused by every sub-tile.
            nisa.nc_matmul(
                dst=raw_score_psum[0:sub_len, 0:n_heads],
                stationary=slots_t_sbuf[0:head_dim, sub_off : sub_off + sub_len],
                moving=head_cols,
                accumulate=False,
            )
            nisa.activation(relu_sbuf[0:sub_len, 0:n_heads], op=nl.relu, data=raw_score_psum[0:sub_len, 0:n_heads])

            # Weight, then reduce over the free (head) axis. This keeps the head-sum on the Vector
            # engine and off the busier Tensor engine.
            nisa.tensor_tensor(
                dst=weighted_sbuf[0:sub_len, 0:n_heads],
                data1=relu_sbuf[0:sub_len, 0:n_heads],
                data2=weight_bcast_sbuf[0:sub_len, 0:n_heads],
                op=nl.multiply,
            )
            nisa.tensor_reduce(
                score_psum[0:sub_len, tile : tile + 1],
                op=nl.add,
                data=weighted_sbuf[0:sub_len, 0:n_heads],
                axis=1,
            )


def _mask_score(
    score_psum: nl.NkiTensor,
    score_t_all: nl.NkiTensor,
    slot_id_sbuf: nl.NkiTensor,
    positions_bcast_sbuf: nl.NkiTensor,
    token: int,
    n_scored_tiles: int,
) -> None:
    """
    Clamp slots the token cannot attend to, then park its score transposed so that the whole batch
    leaves in one store. The comparison is ``slot > position``, so the current position stays
    selectable. Tiles past ``n_scored_tiles`` are never scored and are written as the bound directly.

    ``score[slot] = min(score[slot], MASK_BOUND if slot <= position else -MASK_BOUND)``
    """

    n_key_tiles = score_psum.shape[1]

    score_sbuf = nl.ndarray((P_MAX, n_key_tiles), dtype=nl.float32, buffer=nl.sbuf)
    keep_sbuf = nl.ndarray((P_MAX, n_key_tiles), dtype=nl.float32, buffer=nl.sbuf)
    if n_scored_tiles < n_key_tiles:
        nisa.memset(dst=score_sbuf[0:P_MAX, n_scored_tiles:n_key_tiles], value=-MASK_BOUND)

    nisa.tensor_copy(dst=score_sbuf[0:P_MAX, 0:n_scored_tiles], src=score_psum[0:P_MAX, 0:n_scored_tiles])

    # keep = (slot_id - position <= 0), giving 1.0 for a visible slot and 0.0 otherwise.
    nisa.tensor_scalar(
        dst=keep_sbuf[0:P_MAX, 0:n_scored_tiles],
        data=slot_id_sbuf[0:P_MAX, 0:n_scored_tiles],
        op0=nl.subtract,
        operand0=positions_bcast_sbuf[0:P_MAX, token : token + 1],
        op1=nl.less_equal,
        operand1=0.0,
    )

    # Map that 0/1 to -MASK_BOUND/+MASK_BOUND so a minimum against it either passes the score
    # through or clamps it to the bound.
    nisa.tensor_scalar(
        dst=keep_sbuf[0:P_MAX, 0:n_scored_tiles],
        data=keep_sbuf[0:P_MAX, 0:n_scored_tiles],
        op0=nl.multiply,
        operand0=2.0 * MASK_BOUND,
        op1=nl.subtract,
        operand1=MASK_BOUND,
    )
    nisa.tensor_tensor(
        dst=score_sbuf[0:P_MAX, 0:n_scored_tiles],
        data1=score_sbuf[0:P_MAX, 0:n_scored_tiles],
        data2=keep_sbuf[0:P_MAX, 0:n_scored_tiles],
        op=nl.minimum,
    )

    score_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(score_t_psum[0:n_key_tiles, 0:P_MAX], score_sbuf[0:P_MAX, 0:n_key_tiles])
    nisa.tensor_copy(dst=score_t_all[0:n_key_tiles, token, 0:P_MAX], src=score_t_psum[0:n_key_tiles, 0:P_MAX])


def _load_slot_ids(n_key_tiles: int) -> nl.NkiTensor:
    """Build the slot id of partition ``p``, column ``tile``, which is ``p + P_MAX * tile``."""

    slot_id_sbuf = nl.ndarray((P_MAX, n_key_tiles), dtype=nl.float32, buffer=nl.sbuf)
    slot_id_int = nl.ndarray((P_MAX, n_key_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(
        slot_id_int[0:P_MAX, 0:n_key_tiles],
        pattern=[[P_MAX, n_key_tiles]],
        channel_multiplier=1,
        offset=0,
    )
    nisa.tensor_copy(dst=slot_id_sbuf[0:P_MAX, 0:n_key_tiles], src=slot_id_int[0:P_MAX, 0:n_key_tiles])
    return slot_id_sbuf


def _broadcast_positions(positions: nl.NkiTensor, n_tokens: int) -> nl.NkiTensor:
    """Replicate ``positions`` down every partition, so that the mask operand is per-partition."""

    positions_int = nl.ndarray((P_MAX, n_tokens), dtype=nl.int32, buffer=nl.sbuf)
    positions_bcast_sbuf = nl.ndarray((P_MAX, n_tokens), dtype=nl.float32, buffer=nl.sbuf)

    # [0, P_MAX] is a stride-0 broadcast down the partitions, then [1, n_tokens] walks the row.
    nisa.dma_copy(
        dst=positions_int[0:P_MAX, 0:n_tokens],
        src=positions[0:n_tokens].reshape((1, n_tokens)).broadcast(0, P_MAX),
    )
    nisa.tensor_copy(dst=positions_bcast_sbuf[0:P_MAX, 0:n_tokens], src=positions_int[0:P_MAX, 0:n_tokens])
    return positions_bcast_sbuf


# TopK Step


def _select_topk_indices(scores: nl.NkiTensor, shapes: DeepseekV32SparseIndexerShape) -> nl.NkiTensor:
    """
    Select the highest scoring cache positions for each token with the hardware ``nki.isa.topk``,
    returning a ``[n_tokens, index_topk]`` index tensor in HBM.

    ``topk`` ranks one ``TOPK_GROUP``-partition snake group at a time, so each call covers
    ``TOPK_GROUPS_PER_CALL`` sequences and a token's ``max_seq_len`` slots are laid across
    ``TOPK_GROUP`` partitions. The result is an unordered set, so column order is free.

    ``indices[token] = the index_topk slots with the largest scores[token]``
    """

    n_tokens, max_seq_len = scores.shape[0], scores.shape[1]
    index_topk = shapes.index_topk

    # Columns per partition, for the score being ranked and for the indices coming back.
    score_cols = max_seq_len // TOPK_GROUP
    index_cols = index_topk // TOPK_GROUP

    indices = nl.ndarray((n_tokens, index_topk), dtype=nl.int32, buffer=nl.shared_hbm)

    for group in TiledRange(n_tokens, TOPK_GROUPS_PER_CALL):
        base, group_count = group.start_offset, group.size

        """
        Blocked contiguous load, so each partition reads one unbroken run. The placement that
        makes the snake position equal the slot position needs no remap but scatters the read.
        All group_count tokens move in one transfer: the access pattern is [stride, count] per
        axis, so the outer level walks token rows and the inner two lay each row across
        TOPK_GROUP partitions. One DMA per token instead paid a fixed cost of roughly 600 ns
        against only ~10 KB of payload, in the tail where nothing hides it.
        """
        score_sbuf = nl.ndarray((P_MAX, score_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=score_sbuf.ap(pattern=[[score_cols, group_count * TOPK_GROUP], [1, score_cols]]),
            src=scores[base : base + group_count].reshape_dim(1, (TOPK_GROUP, score_cols)),
        )

        value_sbuf = nl.ndarray((P_MAX, index_topk), dtype=nl.bfloat16, buffer=nl.sbuf)
        snake_index_sbuf = nl.ndarray((P_MAX, index_topk), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(
            val_dst=value_sbuf[0:P_MAX, 0:index_topk],
            idx_dst=snake_index_sbuf[0:P_MAX, 0:index_topk],
            src=score_sbuf[0:P_MAX, 0:score_cols],
            n=max_seq_len,
        )

        """
        Remap the snake position to a slot position. The element at (partition r, column c) held
        scores[r * score_cols + c], and snake position i sits at r = i % TOPK_GROUP,
        c = i // TOPK_GROUP, so slot = (i % TOPK_GROUP) * score_cols + (i // TOPK_GROUP). The
        modulo is a bitwise and, and the divide a right shift, since TOPK_GROUP is a power of two.
        """
        snake_sbuf = nl.ndarray((P_MAX, index_cols), dtype=nl.int32, buffer=nl.sbuf)
        partition_sbuf = nl.ndarray((P_MAX, index_cols), dtype=nl.int32, buffer=nl.sbuf)
        slot_sbuf = nl.ndarray((P_MAX, index_cols), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=snake_sbuf[0:P_MAX, 0:index_cols], src=snake_index_sbuf[0:P_MAX, 0:index_cols])
        nisa.tensor_scalar(
            dst=partition_sbuf[0:P_MAX, 0:index_cols],
            data=snake_sbuf[0:P_MAX, 0:index_cols],
            op0=nl.bitwise_and,
            operand0=TOPK_GROUP - 1,
        )
        nisa.tensor_scalar(
            dst=snake_sbuf[0:P_MAX, 0:index_cols],
            data=snake_sbuf[0:P_MAX, 0:index_cols],
            op0=nl.right_shift,
            operand0=TOPK_GROUP_SHIFT,
        )
        nisa.scalar_tensor_tensor(
            dst=slot_sbuf[0:P_MAX, 0:index_cols],
            data=partition_sbuf[0:P_MAX, 0:index_cols],
            op0=nl.multiply,
            operand0=score_cols,
            op1=nl.add,
            operand1=snake_sbuf[0:P_MAX, 0:index_cols],
        )

        # The output is an unordered set, so column order is free: keep the store contiguous and
        # batch it the same way as the load.
        nisa.dma_copy(
            dst=indices[base : base + group_count].reshape_dim(1, (TOPK_GROUP, index_cols)),
            src=slot_sbuf.ap(pattern=[[index_cols, group_count * TOPK_GROUP], [1, index_cols]]),
        )

    return indices


# Tensor Loading and Preprocessing


def _load_k_tiled(weight: nl.NkiTensor) -> nl.NkiTensor:
    """
    Loads a ``[K, N]`` weight tensor from HBM to SBUF, tiled across its contraction
    dimension of shape ``[P_MAX, n_tiles, N]``.
    """

    K, N = weight.shape[0], weight.shape[1]

    n_tiles = div_ceil(K, P_MAX)
    weight_sbuf = nl.ndarray((P_MAX, n_tiles, N), dtype=weight.dtype, buffer=nl.sbuf)

    """
    Even tiling means every tile is full, so one access pattern covers the whole weight: the
    counts in an access pattern are per-axis constants and cannot express a short final tile.
    That collapses n_tiles transfers into one.
    """
    if K % P_MAX == 0:
        nisa.dma_copy(
            dst=weight_sbuf[0:P_MAX, 0:n_tiles, 0:N],
            src=weight.reshape_dim(0, (n_tiles, P_MAX)).permute((1, 0, 2)),
        )
        return weight_sbuf

    for tile in TiledRange(K, P_MAX):
        start, size = tile.start_offset, tile.size
        nisa.dma_copy(
            dst=weight_sbuf[0:size, tile.index, 0:N],
            src=weight[start : start + size, 0:N],
        )

    return weight_sbuf


def _load_hadamard(hadamard: nl.NkiTensor) -> nl.NkiTensor:
    """Load a Hadamard tensor from HBM to SBUF."""

    head_dim = hadamard.shape[0]
    hadamard_sbuf = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hadamard_sbuf[0:head_dim, 0:head_dim],
        src=hadamard[0:head_dim, 0:head_dim],
    )
    return hadamard_sbuf


def _load_rope_tables(cos: nl.NkiTensor, sin: nl.NkiTensor, n_heads: int) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Loads the rope tables ``cos`` and ``sin`` from HBM to SBUF."""

    n_tokens, rope_dim_half = cos.shape[0], cos.shape[1]
    pattern = [[1, rope_dim_half], [rope_dim_half, n_tokens]]
    tables = []

    for table in (cos, sin):
        # Keep a separate base buffer to pipeline copies.
        table_sbuf = nl.ndarray((P_MAX, n_heads * n_tokens), dtype=nl.float32, buffer=nl.sbuf)
        base_sbuf = nl.ndarray((P_MAX, n_tokens), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=base_sbuf[0:rope_dim_half, 0:n_tokens], src=table.ap(pattern=pattern, offset=0))

        for head in range(n_heads):
            col = head * n_tokens
            nisa.tensor_copy(
                dst=table_sbuf[0:rope_dim_half, col : col + n_tokens],
                src=base_sbuf[0:rope_dim_half, 0:n_tokens],
            )

        tables.append(table_sbuf)

    return tables[0], tables[1]  # (cos, sin)


def _load_rope_tables_token_major(cos: nl.NkiTensor, sin: nl.NkiTensor) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Loads the rope tables ``cos`` and ``sin`` from HBM to SBUF, keyed token major."""

    n_tokens, rope_dim_half = cos.shape[0], cos.shape[1]

    cos_sbuf = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    sin_sbuf = nl.ndarray((P_MAX, rope_dim_half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sbuf[0:n_tokens, 0:rope_dim_half], src=cos[0:n_tokens, 0:rope_dim_half])
    nisa.dma_copy(dst=sin_sbuf[0:n_tokens, 0:rope_dim_half], src=sin[0:n_tokens, 0:rope_dim_half])
    return cos_sbuf, sin_sbuf


def _load_broadcast_row_to_tokens(src: nl.NkiTensor, n_tokens: int) -> nl.NkiTensor:
    """Load a ``[1, width]`` row to SBUF and broadcast it down to ``[n_tokens, width]``."""

    width = src.shape[1]
    out_sbuf = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.sbuf)

    nisa.dma_copy(dst=out_sbuf[0:n_tokens, 0:width], src=src.broadcast(0, n_tokens))
    return out_sbuf


def _load_m_major_then_transpose(src: nl.NkiTensor) -> nl.NkiTensor:
    """
    Load an ``[M, K]`` tensor from HBM to SBUF and transpose it into ``[P_MAX, n_tiles, M]``, the
    same contraction-on-partitions layout that ``_load_k_tiled`` produces.

    The input is contraction minor, which is why a transpose is needed at all: ``_load_k_tiled``
    takes ``[K, N]`` and reaches the same layout with a strided DMA alone.
    """

    M, K = src.shape[0], src.shape[1]
    dtype = src.dtype

    n_tiles = div_ceil(K, P_MAX)

    # Load ``src`` into SBUF.
    src_sbuf = nl.ndarray((P_MAX, K), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src_sbuf[0:M, 0:K], src=src[0:M, 0:K])

    # Allocate output buffers, then transpose.
    out_sbuf = nl.ndarray((P_MAX, n_tiles, M), dtype=dtype, buffer=nl.sbuf)
    out_psum = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.psum)
    for tile in TiledRange(K, P_MAX):
        start, size = tile.start_offset, tile.size
        nisa.nc_transpose(out_psum[0:size, 0:M], src_sbuf[0:M, start : start + size])
        nisa.tensor_copy(dst=out_sbuf[0:size, tile.index, 0:M], src=out_psum[0:size, 0:M])

    return out_sbuf
