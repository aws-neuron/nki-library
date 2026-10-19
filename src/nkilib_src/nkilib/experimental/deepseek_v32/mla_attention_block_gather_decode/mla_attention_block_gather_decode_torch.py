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

"""Torch reference for the DeepSeek-V3.2 MLA decode attention block."""

import torch

from ..mla_decode_attention import P_MAX, mla_decode_attention_torch_ref
from ..mla_decode_qkv import mla_decode_qkv_torch_ref


def mla_attention_block_gather_decode_torch_ref(
    hidden_states: torch.Tensor,  # [n_tokens, hidden]
    input_norm_w: torch.Tensor,  # [1, hidden] fp32
    wq_a: torch.Tensor,  # [hidden, q_lora_rank]
    q_norm_w: torch.Tensor,  # [1, q_lora_rank] fp32
    wq_b: torch.Tensor,  # [q_lora_rank, qk_packed_dim]
    wkv_a: torch.Tensor,  # [hidden, kv_row_dim]
    kv_norm_w: torch.Tensor,  # [1, kv_lora_rank] fp32
    q_absorb_w: torch.Tensor,  # [n_heads, qk_nope_dim, kv_lora_rank]
    out_absorb_w: torch.Tensor,  # [n_heads, v_head_dim, kv_lora_rank]
    o_proj_w: torch.Tensor,  # [n_heads * v_head_dim, hidden]
    cos: torch.Tensor,  # [n_tokens, qk_rope_dim // 2]
    sin: torch.Tensor,  # [n_tokens, qk_rope_dim // 2]
    kv_cache: torch.Tensor,  # [n_blocks, 1, block_size, kv_row_dim]
    slot_mapping: torch.Tensor,  # [n_tokens] int32
    block_table: torch.Tensor,  # [n_tokens, max_blocks_per_seq] int32
    positions: torch.Tensor,  # [n_tokens] int32
    gather_indices: torch.Tensor,  # [n_tokens, n_prior] int32
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    softmax_scale: float,
    norm_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    Reference for the MLA decode attention block over indexer-selected prior keys.

    Composes the two references the kernel composes, and introduces no numerics of its own. What it
    does add is the wiring between them, which is the only thing this block contributes: the query
    tiles are tiled onto partitions the way the front-fold leaves them in SBUF, and the selected
    prior is resolved to cache rows the way ``build_prior_index`` resolves it.

    The reference for the front-fold does not write the cache, so this one applies the scatter
    itself before the core reads it. The kernel gets that for free, the front-fold having written
    the row in place, and the core then reading the current row from the same cache.

    Returns:
        ``output_0``, ``[n_tokens, hidden]``, this rank's attention-block output before the
        cross-rank all-reduce.
    """

    front_fold = mla_decode_qkv_torch_ref(
        hidden_states,
        wq_a,
        q_norm_w.reshape(-1),
        wq_b,
        wkv_a,
        kv_norm_w.reshape(-1),
        cos,
        sin,
        q_absorb_w,
        kv_cache,
        slot_mapping,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        norm_eps,
        input_norm_w.reshape(-1),
    )

    kv_row_cur = front_fold["output_2"]
    block_size = kv_cache.shape[2]

    """
    The kernel reads the current row back out of the cache the front-fold just wrote, so the
    reference has to write it too or the two disagree whenever a selected prior lands on the
    row this step is writing.
    """
    kv_cache = kv_cache.clone()
    kv_cache.reshape(-1, kv_cache.shape[3])[slot_mapping.to(torch.int64)] = kv_row_cur.to(kv_cache.dtype)

    n_prior = gather_indices.shape[1]
    rows = _resolve_cache_rows(gather_indices, block_table, block_size)

    return mla_decode_attention_torch_ref(
        _tile_features(front_fold["output_0"]),
        _pad_partitions(front_fold["output_1"]),
        kv_cache,
        kv_row_cur,
        _tile_keys(rows),
        _tile_keys(gather_indices.float()),
        _tile_keys(rows),
        positions,
        out_absorb_w,
        o_proj_w,
        n_prior,
        softmax_scale,
    )


def _resolve_cache_rows(
    gather_indices: torch.Tensor,  # [n_tokens, n_prior] int32
    block_table: torch.Tensor,  # [n_tokens, max_blocks_per_seq] int32
    block_size: int,
) -> torch.Tensor:
    """
    Resolves each selected prior position to its row in the flattened paged cache.

    Position ``p`` of token ``t`` lives in block ``block_table[t, p // block_size]`` at offset
    ``p % block_size``, so its flat row is ``block * block_size + offset``.
    """

    positions = gather_indices.to(torch.int64)
    blocks = torch.gather(block_table.to(torch.int64), 1, positions // block_size)
    return blocks * block_size + positions % block_size


def _tile_features(tile: torch.Tensor) -> torch.Tensor:
    """
    Tiles a feature-major query onto partitions, ``[feature, ...]`` to ``[P_MAX, n_tiles, ...]``.

    Feature ``tile * P_MAX + p`` lives on partition ``p`` of tile ``tile``, which is the layout the
    front-fold leaves in SBUF and the score matmul contracts over.
    """

    n_features = tile.shape[0]
    return tile.reshape(n_features // P_MAX, P_MAX, *tile.shape[1:]).transpose(0, 1).contiguous()


def _pad_partitions(tile: torch.Tensor) -> torch.Tensor:
    """
    Pads a feature-major tile's leading axis up to ``P_MAX``, the partition count of an SBUF tile.

    The consumer slices the real width back off, so the padding is never read.
    """

    pad = torch.zeros((P_MAX - tile.shape[0], *tile.shape[1:]), dtype=tile.dtype, device=tile.device)
    return torch.cat([tile, pad], dim=0)


def _tile_keys(table: torch.Tensor) -> torch.Tensor:
    """
    Tiles a per-token key table onto partitions, ``[n_tokens, n_prior]`` to ``[P_MAX, n_tokens, n_tiles]``.

    Key ``tile * P_MAX + p`` of token ``t`` lands at ``[p, t, tile]``, inverting the flattening the
    attention reference applies to read it back in plain key order. A ragged last tile is padded,
    and those entries are dropped on the way back out.
    """

    n_tokens, n_prior = table.shape
    n_tiles = (n_prior + P_MAX - 1) // P_MAX
    padded = torch.zeros((n_tokens, n_tiles * P_MAX), dtype=table.dtype, device=table.device)
    padded[:, 0:n_prior] = table
    return padded.reshape(n_tokens, n_tiles, P_MAX).permute(2, 0, 1).contiguous()
