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

"""Torch reference for the DeepSeek-V3.2 MLA decode attention core."""

import torch


def mla_decode_attention_torch_ref(
    q_absorbed_sbuf: torch.Tensor,  # [P_MAX, n_latent_tiles, n_tokens, n_heads]
    q_pe_sbuf: torch.Tensor,  # [P_MAX, n_tokens, n_heads]
    kv_cache: torch.Tensor,  # [n_blocks, 1, block_size, kv_row_dim]
    kv_row_cur: torch.Tensor,  # [n_tokens, kv_row_dim]
    prior_rows: torch.Tensor,  # [P_MAX, n_tokens, n_prior_tiles] int32
    prior_key_pos: torch.Tensor,  # [P_MAX, n_tokens, n_prior_tiles] fp32
    prior_rows_natural: torch.Tensor,  # [P_MAX, n_tokens, n_prior_tiles] uint32
    positions: torch.Tensor,  # [n_tokens] int32
    out_absorb_w: torch.Tensor,  # [n_heads, v_head_dim, kv_lora_rank]
    o_proj_w: torch.Tensor,  # [n_heads * v_head_dim, hidden]
    n_prior: int,
    softmax_scale: float,
) -> dict[str, torch.Tensor]:
    """
    Reference for the MLA decode attention core over indexer-selected prior keys.

    Signature-matched to the kernel, so the query tiles arrive in the partition-tiled feature-major
    layout the front-fold leaves in SBUF, and the selected prior arrives already resolved to cache rows
    by ``build_prior_index``. Selection and resolution therefore are not repeated here: this reference
    covers only what the core itself computes.

    Every intermediate is computed in fp32 and only the result is cast back, which is what the kernel
    does too, since its scores and softmax are fp32 regardless of the operand dtype.

    Returns:
        ``output_0``, ``[n_tokens, hidden]``, this rank's attention-block output before the cross-rank
        all-reduce.
    """

    """
    prior_rows holds the same cache rows as prior_rows_natural, permuted into the order the kernel's batched indirect
    gather consumes them in. Only the natural order reads back as plain key order, so that is the one this reference
    indexes with.
    """
    del prior_rows

    p_max, n_latent_tiles, n_tokens, n_heads = q_absorbed_sbuf.shape
    kv_lora_rank = out_absorb_w.shape[2]
    qk_rope_dim = kv_cache.shape[3] - kv_lora_rank
    out_dtype = q_absorbed_sbuf.dtype

    # [P_MAX, n_latent_tiles, ...] -> [kv_lora_rank, ...], since feature tile * P_MAX + p sits at
    # [p, tile]. Then to [n_tokens, n_heads, feature], which is what the einsums below want.
    q_absorbed = q_absorbed_sbuf.permute(1, 0, 2, 3).reshape(n_latent_tiles * p_max, n_tokens, n_heads)
    q_absorbed = q_absorbed[0:kv_lora_rank].permute(1, 2, 0).float()
    q_pe = q_pe_sbuf[0:qk_rope_dim].permute(1, 2, 0).float()

    rows = _to_key_order(prior_rows_natural, n_prior).to(torch.int64)
    key_pos = _to_key_order(prior_key_pos, n_prior).float()

    cache_rows = kv_cache.reshape(-1, kv_cache.shape[3]).float()
    keys = torch.cat([cache_rows[rows], kv_row_cur.float().unsqueeze(1)], dim=1)
    k_pe, kv_latent = keys[..., :qk_rope_dim], keys[..., qk_rope_dim:]

    scores = softmax_scale * (
        torch.einsum("thc,tsc->ths", q_absorbed, kv_latent) + torch.einsum("thr,tsr->ths", q_pe, k_pe)
    )

    # A prior key is kept only from strictly before this token's position. The current token's key is
    # the last one and is never masked.
    keep_prior = key_pos < positions.float().unsqueeze(1)
    keep_current = torch.ones((n_tokens, 1), dtype=torch.bool, device=keep_prior.device)
    keep = torch.cat([keep_prior, keep_current], dim=1)
    scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))

    latent_out = torch.einsum("ths,tsc->thc", torch.softmax(scores, dim=-1), kv_latent)
    attn = torch.einsum("thc,hdc->thd", latent_out, out_absorb_w.float())
    output = attn.reshape(n_tokens, -1) @ o_proj_w.float()

    return {"output_0": output.to(out_dtype)}


def _to_key_order(table: torch.Tensor, n_prior: int) -> torch.Tensor:
    """
    Flattens a key-tiled index table into plain key order.

    Takes ``[P_MAX, n_tokens, n_prior_tiles]``, where key ``tile * P_MAX + p`` of token ``t`` lives at
    ``[p, t, tile]``, and returns ``[n_tokens, n_prior]``. The trailing entries of a ragged last tile
    are dropped, since they hold no selected key.
    """

    n_tokens = table.shape[1]
    return table.permute(1, 2, 0).reshape(n_tokens, -1)[:, 0:n_prior]
