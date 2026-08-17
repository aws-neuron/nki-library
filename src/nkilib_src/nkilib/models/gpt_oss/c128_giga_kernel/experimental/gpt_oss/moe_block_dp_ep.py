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

"""GPT-OSS MXFP4 MoE decode block with its surrounding EP collectives (LNC=1).

This wraps the ``moe_block_tkg`` all-expert MXFP4 kernel with the *dense cross-DP
EP* dispatch/combine collectives — the path GPT-OSS decode uses when the sparse
AllToAllV backend is unavailable (which it is on LNC=1). It mirrors the
``cross_dp_ep`` branch of ``vllm_neuron/model/gpt_oss/model_mxfp4.py`` (dispatch
``AllGather`` at ``:1398-1401``, combine ``ReduceScatter`` at ``:1414-1415``),
collapsed to a single expert-parallel group so it can be exercised as one
standalone multi-rank unit.

Sharding (EP = ``num_ranks`` ranks, one expert per rank):
  * Each rank ``r`` owns global expert ``r`` (``num_local_experts=1``,
    ``num_global_experts=num_ranks``, ``rank_id=r``).
  * Layer input is **token-sharded (SP)**: rank ``r`` holds ``[T/num_ranks, H]``.
  * **Dispatch — AllGather[EP] on dim 0**: every rank gathers all ranks' token
    shards -> full ``[T, H]``, so each rank can run its one expert over every
    token (the router selects which tokens actually route to it).
  * **Compute**: ``moe_block_tkg(is_all_expert=True, rank_id=r)`` -> this rank's
    expert contribution, affinity-masked, for all ``T`` tokens -> ``[T, H]``.
  * **Combine — ReduceScatter[EP] on dim 0**: sum the per-expert partials across
    ranks (a token's top-k contributions live on k different ranks) and re-shard
    over tokens -> each rank gets ``[T/num_ranks, H]`` of the final MoE output,
    still token-sharded (ready for the next layer's attention, matching the
    ReduceScatter[World] rsag-combine convention).

This is the whole point of ReduceScatter (rather than AllReduce) for the combine:
the output stays token-sharded end-to-end, so no redundant re-shard is needed
between the MoE and the next attention block.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.moe_block.moe_block_tkg import moe_block_tkg
from ...core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)


@nki.jit
def moe_block_dp_ep_kernel(
    inp_shard: nl.ndarray,
    gamma: nl.ndarray,
    router_weights: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    expert_gate_up_weights_scale: nl.ndarray,
    expert_down_weights_scale: nl.ndarray,
    router_bias: nl.ndarray,
    expert_gate_up_bias: nl.ndarray,
    expert_down_bias: nl.ndarray,
    rank_id: nl.ndarray,
    replica_group: ReplicaGroup,
    num_ranks: int,
    top_k: int,
    eps: float,
    hidden_actual: int,
    gate_clamp_upper_limit: float,
    up_clamp_upper_limit: float,
    up_clamp_lower_limit: float,
):
    """One EP rank's MoE decode block: AllGather -> moe_block_tkg -> ReduceScatter.

    Args:
        inp_shard: ``[T // num_ranks, H]`` this rank's token shard of the
            residual-stream hidden states (SP layout), MXFP4-model dtype.
        gamma: ``[1, H]`` post-attention RMSNorm weight (replicated).
        router_weights: ``[H, num_global_experts]`` router matmul weight
            (replicated across ranks; each rank runs the full router).
        expert_gate_up_weights: ``[1, 128, 2, n_H512, I]`` MXFP4 (float4_e2m1fn_x4)
            fused gate/up weight for this rank's single local expert.
        expert_down_weights: ``[1, 128, n_I512, H]`` MXFP4 down weight.
        expert_gate_up_weights_scale / expert_down_weights_scale: uint8 E8M0 block
            scales for the above.
        router_bias: ``[1, num_global_experts]`` router bias (replicated).
        expert_gate_up_bias / expert_down_bias: this rank's expert bias (the up
            half of gate/up carries the GPT-OSS ``+1``, baked in offline).
        rank_id: ``[1, 1]`` uint32 = this rank's expert index (== EP rank).
        replica_group: EP replica group spanning all ``num_ranks`` ranks.
        num_ranks: EP degree (number of ranks / global experts).
        top_k, eps, hidden_actual, gate/up clamps: GPT-OSS MoE hyperparameters.

    Returns:
        ``[T // num_ranks, H]`` this rank's token shard of the summed MoE output.
    """
    T_shard, H = inp_shard.shape
    T = T_shard * num_ranks
    dtype = inp_shard.dtype

    # ── Dispatch: AllGather[EP] the SP token shards -> full [T, H] on every rank.
    # (name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure.)
    ag_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="ag_src")
    ag_dst = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm, name="ag_dst")
    nisa.dma_copy(dst=ag_src, src=inp_shard)
    ncc.all_gather(dsts=[ag_dst], srcs=[ag_src], replica_group=replica_group, collective_dim=0)

    # moe_block_tkg wants [B, S, H]; feed the gathered tokens as B=1, S=T.
    inp_full = ag_dst.reshape((1, T, H))

    # ── Compute this rank's single local expert over all T tokens.
    (expert_out,) = moe_block_tkg(
        inp=inp_full,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        router_pre_norm=False,  # softmax AFTER top-k (GPT-OSS)
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,  # x * sigmoid(1.702 x)
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=nl.float16,  # match the float16 router weights / hidden
        hidden_actual=hidden_actual,
        skip_router_logits=True,
        is_all_expert=True,
        rank_id=rank_id,
    )  # [T, H] partial (this expert's affinity-scaled contribution per token)

    # ── Combine: ReduceScatter[EP] sums per-expert partials across ranks and
    # re-shards over tokens -> [T // num_ranks, H] on each rank (SP layout).
    rs_src = nl.ndarray((T, H), dtype=expert_out.dtype, buffer=nl.shared_hbm, name="rs_src")
    rs_dst = nl.ndarray((T_shard, H), dtype=expert_out.dtype, buffer=nl.shared_hbm, name="rs_dst")
    out = nl.ndarray((T_shard, H), dtype=expert_out.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=rs_src, src=expert_out)
    ncc.reduce_scatter(dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=rs_dst)
    return out
