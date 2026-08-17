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

"""Torch reference for ``moe_block_dp_ep_kernel``.

Same signature as the kernel. Mirrors the kernel's collective structure using
``torch.distributed`` (backed by the test framework's SimDistAdapter in sim, or
a real backend on hardware) via ``get_pg``:

    AllGather[EP] token shards -> moe_block_tkg_torch_ref(this rank's expert)
                               -> ReduceScatter[EP] the partials

The per-expert math is delegated to the validated ``moe_block_tkg_torch_ref``
(RMSNorm -> router top-k -> per-expert SwiGLU), run in all-expert mode with this
rank's ``rank_id`` so it applies only the local expert's affinity-masked
contribution. Summing those partials over ranks (the ReduceScatter) reconstructs
the full top-k mixture.
"""

import nki.language as nl
import numpy as np
import torch
import torch.distributed as dist

from ...core.moe_block.moe_block_tkg_torch import moe_block_tkg_torch_ref
from ...core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from ...core.utils.torch_ref_wrapper import torch_ref_wrapper
from ..collectives.distributed_adapter import get_pg


def moe_block_dp_ep_torch_ref(
    inp_shard: np.ndarray,
    gamma: np.ndarray,
    router_weights: np.ndarray,
    expert_gate_up_weights: np.ndarray,
    expert_down_weights: np.ndarray,
    expert_gate_up_weights_scale: np.ndarray,
    expert_down_weights_scale: np.ndarray,
    router_bias: np.ndarray,
    expert_gate_up_bias: np.ndarray,
    expert_down_bias: np.ndarray,
    rank_id: np.ndarray,
    replica_group,
    num_ranks: int,
    top_k: int,
    eps: float,
    hidden_actual: int,
    gate_clamp_upper_limit: float,
    up_clamp_upper_limit: float,
    up_clamp_lower_limit: float,
) -> dict:
    """Reference matching ``moe_block_dp_ep_kernel`` (see module docstring)."""
    pg = get_pg(replica_group)
    dtype = inp_shard.dtype
    T_shard, H = inp_shard.shape

    # ── Dispatch: AllGather[EP] the SP token shards -> full [T, H].
    shard_t = torch.from_numpy(inp_shard.astype(np.float32))
    gathered = [torch.zeros_like(shard_t) for _ in range(num_ranks)]
    dist.all_gather(gathered, shard_t, group=pg)
    inp_full = torch.cat(gathered, dim=0)  # [T, H] fp32
    inp_full_np = inp_full.numpy().astype(dtype)

    # ── Compute this rank's single local expert over all T tokens via the
    # validated MoE torch ref (all-expert mode, rank_id selects the local expert).
    # torch_ref_wrapper handles numpy<->torch incl. the MXFP4 x4 weight passthrough.
    wrapped = torch_ref_wrapper(moe_block_tkg_torch_ref, preserve_lower_precision=True)
    expert_out = wrapped(
        inp=inp_full_np.reshape(1, inp_full_np.shape[0], H),
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
        router_pre_norm=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=nl.float16,
        hidden_actual=hidden_actual,
        skip_router_logits=True,
        is_all_expert=True,
        rank_id=rank_id,
    )["out"]  # [T, H] in the model dtype

    # ── Combine: ReduceScatter[EP] the per-expert partials -> [T_shard, H].
    partial = torch.from_numpy(expert_out.astype(np.float32))  # [T, H]
    chunks = list(partial.chunk(num_ranks, dim=0))
    out = torch.zeros_like(chunks[0])
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=pg)
    return {"out": out.numpy().astype(dtype)}
