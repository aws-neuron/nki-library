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

"""DeepSeek-V3.2 MLA decode attention block over indexer-selected keys."""

import nki
import nki.language as nl

from ....core.utils.entry_trace import trace_kernel_entry
from ..mla_decode_attention import build_prior_index, mla_decode_attention
from ..mla_decode_qkv import mla_decode_qkv


@nki.jit
def mla_attention_block_gather_decode(
    hidden_states: nl.NkiTensor,  # [n_tokens, hidden]
    input_norm_w: nl.NkiTensor,  # [1, hidden]                                 RMSNorm gamma, fp32
    wq_a: nl.NkiTensor,  # [hidden, q_lora_rank]                       query down-projection
    q_norm_w: nl.NkiTensor,  # [1, q_lora_rank]                            RMSNorm gamma, fp32
    wq_b: nl.NkiTensor,  # [q_lora_rank, qk_packed_dim]                query up-projection
    wkv_a: nl.NkiTensor,  # [hidden, kv_row_dim]                        KV down-projection
    kv_norm_w: nl.NkiTensor,  # [1, kv_lora_rank]                           RMSNorm gamma, fp32
    q_absorb_w: nl.NkiTensor,  # [n_heads, qk_nope_dim, kv_lora_rank]        wkv_b nope half
    out_absorb_w: nl.NkiTensor,  # [n_heads, v_head_dim, kv_lora_rank]         wkv_b value half
    o_proj_w: nl.NkiTensor,  # [n_heads * v_head_dim, hidden]              per-rank o-projection
    cos: nl.NkiTensor,  # [n_tokens, qk_rope_dim // 2]                interleaved RoPE table
    sin: nl.NkiTensor,  # [n_tokens, qk_rope_dim // 2]
    kv_cache: nl.NkiTensor,  # [n_blocks, 1, block_size, kv_row_dim]       paged, mutated in place
    slot_mapping: nl.NkiTensor,  # [n_tokens] int32                            destination row per token
    block_table: nl.NkiTensor,  # [n_tokens, max_blocks_per_seq] int32
    positions: nl.NkiTensor,  # [n_tokens] int32                            current decode positions
    gather_indices: nl.NkiTensor,  # [n_tokens, n_prior] int32                   indexer-selected priors
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    softmax_scale: float,
    norm_eps: float = 1e-6,
) -> nl.NkiTensor:
    """
    Computes one DeepSeek-V3.2 MLA decode step end to end, from the residual stream to this rank's
    attention-block output, attending over only the prior keys the Lightning Indexer selected.

    This is the whole attention block: it traces the QKV front-fold and the attention core into a
    single NEFF rather than calling them as separate kernels. That is what makes the composition
    worth having, because the front-fold's two query tiles stay **in SBUF** and are handed straight
    to the core. Run standalone the tiles would have to round-trip through HBM, and the core's
    signature is built for the inline hand-off, not for a caller staging them itself.

    The block owns no numerics of its own. Every value it returns is computed by one of the two
    kernels it composes, so its correctness is theirs plus the wiring, and its reference composes
    their references for the same reason.

    Dimensions:
        n_tokens: Decode tokens in the batch, one per sequence.
        n_heads: Attention heads on this rank.
        n_prior: Prior keys the indexer selected per token, taken from ``gather_indices``.
        hidden: Model hidden width, which the o-projection emits.

    Args:
        hidden_states: ``[n_tokens, hidden]``. This step's residual-stream hidden states.
        input_norm_w: ``[1, hidden]`` fp32. RMSNorm gamma for the layer input, folded in by the
            front-fold rather than applied by the caller.
        wq_a: ``[hidden, q_lora_rank]``. Query down-projection.
        q_norm_w: ``[1, q_lora_rank]`` fp32. RMSNorm gamma for the query latent.
        wq_b: ``[q_lora_rank, qk_packed_dim]``. Query up-projection, emitting every head's
            ``[q_nope | q_pe]`` concatenated.
        wkv_a: ``[hidden, kv_row_dim]``. KV down-projection, emitting one cache row per token.
        kv_norm_w: ``[1, kv_lora_rank]`` fp32. RMSNorm gamma for the KV latent.
        q_absorb_w: ``[n_heads, qk_nope_dim, kv_lora_rank]``. The nope half of ``wkv_b``, which maps
            a head's nope query into the cache's latent space.
        out_absorb_w: ``[n_heads, v_head_dim, kv_lora_rank]``. The value half of ``wkv_b``, which
            maps the latent attention output back out to a head's value width.
        o_proj_w: ``[n_heads * v_head_dim, hidden]``. This rank's shard of the o-projection.
        cos: ``[n_tokens, qk_rope_dim // 2]``. Cosine table for interleaved RoPE.
        sin: ``[n_tokens, qk_rope_dim // 2]``. Sine table for interleaved RoPE.
        kv_cache: ``[n_blocks, 1, block_size, kv_row_dim]``. Paged cache, **mutated in place**: the
            front-fold writes this step's row, and the core reads the selected prior rows back.
        slot_mapping: ``[n_tokens]`` int32. Flat cache row each token's own key is written to.
        block_table: ``[n_tokens, max_blocks_per_seq]`` int32. Per-sequence logical-to-physical
            block map, which resolves a selected position to a cache row.
        positions: ``[n_tokens]`` int32. This step's absolute position per token, which the causal
            mask compares each selected key against.
        gather_indices: ``[n_tokens, n_prior]`` int32. The absolute prior positions the indexer
            selected, unsorted. Its width fixes ``n_prior``.
        n_heads: Attention heads on this rank.
        qk_nope_head_dim: Per-head query/key width that RoPE does not rotate.
        qk_rope_head_dim: Per-head query/key width that RoPE rotates.
        kv_lora_rank: LoRA rank of the KV path, and the latent width of a cache row.
        softmax_scale: Score scale, conventionally the inverse square root of the full per-head
            query/key width.
        norm_eps: Epsilon for every RMSNorm in the front-fold.

    Returns:
        ``[n_tokens, hidden]``, this rank's attention-block output before the cross-rank all-reduce.
        ``kv_cache`` is also mutated in place, the front-fold having written this step's row.

    Notes:
        The index build is issued **first**, ahead of the front-fold that does not consume it. Its
        two inputs are under 1 MB together, but queued behind the projection weights they sit behind
        roughly 31 MB on the same DMA queue and do not land until about 97 us, and no per-token
        gather in the core can start until they do.

        Fusing the two halves is worth about 45 us at the shipping shape, measured as 385.8 us for
        the two run separately against 340.2 us here. The saving is the HBM round trip the query
        tiles would otherwise make, which is why the core takes them in SBUF.

    Pseudocode:
        prior = build_prior_index(gather_indices, block_table, kv_cache)
        q_absorbed, q_pe, kv_row_cur = mla_decode_qkv(...)     # writes kv_cache[slot_mapping]
        return mla_decode_attention(q_absorbed, q_pe, kv_cache, kv_row_cur, prior, ...)
    """

    trace_kernel_entry("mla_attention_block_gather_decode", locals())

    """
    Resolve every selected prior position to a flat cache row FIRST, ahead of the front-fold that
    does not consume it. block_table and gather_indices together are under 1 MB, but issued after
    the projection weights they queue behind roughly 31 MB on the same DMA queue and do not land
    until about 97 us, and the core cannot start any per-token gather until they do.
    """
    prior = build_prior_index(gather_indices, block_table, kv_cache)

    # The front-fold leaves both query tiles in SBUF and writes this step's row into kv_cache. The
    # cache mutation reaches the caller through its operand_output_aliases.
    qkv = mla_decode_qkv(
        hidden_states,
        wq_a,
        q_norm_w,
        wq_b,
        wkv_a,
        kv_norm_w,
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
        input_norm_w,
    )

    # The core gathers the selected rows itself, straight from kv_cache in its native token-major
    # layout, so the front-fold hands over only the query tiles and this step's row.
    return mla_decode_attention(
        qkv.q_absorbed_sbuf,
        qkv.q_pe_sbuf,
        kv_cache,
        qkv.kv_row_cur_hbm,
        prior.rows,
        prior.key_pos,
        prior.rows_natural,
        positions,
        out_absorb_w,
        o_proj_w,
        gather_indices.shape[1],
        softmax_scale,
    )
