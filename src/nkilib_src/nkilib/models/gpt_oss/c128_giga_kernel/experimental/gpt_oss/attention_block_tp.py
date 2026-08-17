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

"""GPT-OSS attention decode block with its surrounding TP collectives.

Wraps ``attention_block_tkg`` with the tensor-parallel dispatch/combine
collectives GPT-OSS decode uses (``model_mxfp4.py``): an ``AllGather[TP]`` on the
sequence-parallel (token-sharded) input (``:779``) and a ``ReduceScatter[TP]`` on
the row-parallel O-projection output (``:911-912``).

Sharding (TP = ``num_ranks`` ranks per group; DP replicas are independent groups):
  * Each TP rank owns ``q_heads`` query heads (a 1/TP slice of the model's heads)
    and its row-parallel ``W_out`` slice ``[q_heads*d_head, H]``. The single KV
    head (GQA, ``kv_heads=1``) and its paged cache are replicated on every TP rank.
  * Layer input is **token-SP-sharded**: rank holds ``[B*S_tkg / num_ranks, H]``.
  * **Dispatch — AllGather[TP] on dim 0**: reconstruct the full ``[B*S_tkg, H]``
    tokens so the column-parallel QKV projection sees every token.
  * **Compute**: ``attention_block_tkg`` runs RMSNorm(optional) -> QKV -> RoPE ->
    paged attention with per-head sinks -> in-place KV write -> O-proj. Because
    ``W_out`` is row-parallel over this rank's heads, the O-proj output is a
    **per-rank partial** ``[B*S_tkg, H]`` (the contribution of this rank's heads).
  * **Combine — ReduceScatter[TP] on dim 0**: sum the ``num_ranks`` head-partials
    (yielding the true multi-head attention output) and re-shard over tokens ->
    ``[B*S_tkg / num_ranks, H]`` per rank, back in SP layout for the residual add
    and the downstream MoE — matching the production ReduceScatter[TP] convention.

DP is pure batch data-parallelism: each DP replica is an independent TP group
(disjoint requests, private KV), so there is no cross-DP collective — DP>1 is
expressed purely as extra subgroups in the ``replica_group``.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.common_types import QuantizationType
from ..transformer.attention_block_tkg import attention_block_tkg


@nki.jit
def attention_block_tp_kernel(
    X_shard: nl.ndarray,
    W_qkv: nl.ndarray,
    bias_qkv: nl.ndarray,
    W_out: nl.ndarray,
    bias_out: nl.ndarray,
    cos: nl.ndarray,
    sin: nl.ndarray,
    sink: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    active_blocks_table: nl.ndarray,
    attention_mask: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    pos_ids: nl.ndarray,
    swa_start_pos_ids: nl.ndarray,
    replica_group: ReplicaGroup,
    num_ranks: int,
    B: int,
    S_tkg: int,
    X_hidden_dim_actual: int,
    softmax_scale: float,
):
    """One TP rank's attention decode block: AllGather -> attention -> ReduceScatter.

    Args:
        X_shard: ``[B*S_tkg // num_ranks, H]`` this rank's token shard (SP layout).
        W_qkv: ``[H, (q_heads + 2*kv_heads)*d_head]`` this rank's column-parallel
            QKV weight (its head slice).
        bias_qkv: ``[1, (q_heads + 2*kv_heads)*d_head]`` QKV bias.
        W_out: ``[q_heads*d_head, H]`` this rank's row-parallel O-proj weight.
        bias_out: ``[1, H]`` O-proj bias.
        cos, sin: RoPE tables (contiguous half-split layout).
        sink: ``[q_heads, 1]`` per-head attention sink for this rank's heads.
        K_cache, V_cache: this rank's paged KV cache (replicated KV head).
        active_blocks_table, attention_mask, kv_cache_update_idx: paged-attn args.
        pos_ids, swa_start_pos_ids: in-kernel mask generation (SWA if provided).
        replica_group: TP replica group (its subgroup for this DP replica).
        num_ranks: TP degree.
        B, S_tkg: batch and decode-token count (B*S_tkg == full token count).
        X_hidden_dim_actual: unpadded hidden width for RMSNorm (None-safe int).
        softmax_scale: attention softmax scale.

    Returns:
        ``[B*S_tkg // num_ranks, H]`` this rank's token shard of the summed
        multi-head attention output (SP layout).
    """
    T_shard, H = X_shard.shape
    T = T_shard * num_ranks
    dtype = X_shard.dtype

    # ── Dispatch: AllGather[TP] the SP token shards -> full [T, H] on every rank.
    ag_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="ag_src")
    ag_dst = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm, name="ag_dst")
    nisa.dma_copy(dst=ag_src, src=X_shard)
    ncc.all_gather(dsts=[ag_dst], srcs=[ag_src], replica_group=replica_group, collective_dim=0)

    X = ag_dst.reshape((B, S_tkg, H))

    # ── Compute this rank's head-shard attention (O-proj emits a per-rank partial).
    # update_cache=True writes K/V in place into this rank's (replicated-KV) cache.
    out, _, _ = attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=False,
        rmsnorm_X_eps=None,
        rmsnorm_X_gamma=None,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        # update_cache=False: this test validates the collective + attention output,
        # not KV-cache persistence. The current token's freshly-computed K/V is still
        # used in the softmax; only the write-back is skipped — which avoids the
        # in-place KV aliasing protocol (the wrapper returns only the attn output).
        update_cache=False,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        transposed_in=False,
        softmax_scale=softmax_scale,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
    )  # [T, H] per-rank O-proj partial (this rank's head contribution)

    # ── Combine: ReduceScatter[TP] sums head-partials across ranks and re-shards
    # over tokens -> [T // num_ranks, H] per rank (SP layout).
    rs_src = nl.ndarray((T, H), dtype=out.dtype, buffer=nl.shared_hbm, name="rs_src")
    rs_dst = nl.ndarray((T_shard, H), dtype=out.dtype, buffer=nl.shared_hbm, name="rs_dst")
    result = nl.ndarray((T_shard, H), dtype=out.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=rs_src, src=out)
    ncc.reduce_scatter(dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=result, src=rs_dst)
    return result
