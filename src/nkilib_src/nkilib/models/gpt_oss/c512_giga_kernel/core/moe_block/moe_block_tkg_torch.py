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

"""PyTorch reference implementation for MoE Block TKG kernel.

Composes rmsnorm, router_topk, and moe_tkg subkernel torch refs.
"""

from typing import Optional

import numpy as np
import torch

from ..moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref
from ..router_topk.router_topk_torch import router_topk_torch_ref
from ..subkernels.rmsnorm_torch import rms_norm_torch_ref
from ..utils.common_types import ExpertAffinityScaleMode, RouterActFnType

# NKI dtype -> torch dtype for the low-precision router-matmul types (str(nl.bfloat16) == "bfloat16").
_NL_TO_TORCH_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def moe_block_tkg_torch_ref(
    inp: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    expert_gate_up_weights,
    expert_down_weights,
    shared_expert_gate_w=None,
    shared_expert_up_w=None,
    shared_expert_down_w=None,
    expert_gate_up_weights_scale=None,
    expert_down_weights_scale=None,
    router_bias=None,
    expert_gate_up_bias=None,
    expert_down_bias=None,
    shared_expert_gate_bias=None,
    shared_expert_up_bias=None,
    shared_expert_down_bias=None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    hidden_act_fn=None,
    hidden_act_scale_factor=None,
    hidden_act_bias=None,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    router_mm_dtype=None,
    hidden_actual: Optional[int] = None,
    skip_router_logits: bool = False,
    is_all_expert: bool = False,
    rank_id=None,
    residual=None,
    expert_gate_up_input_scale=None,
    expert_down_input_scale=None,
    is_all_expert_dynamic=False,
    block_size=None,
    inp_layout=None,
    outp_layout=None,
) -> dict:
    """Composite torch ref for moe_block_tkg kernel.

    Signature matches the moe_block_tkg kernel exactly.
    Composes: RMSNorm -> RouterTopK -> MoE TKG.
    """
    from ..utils.common_types import MoEBlockIOLayout

    _T_LAYOUT = MoEBlockIOLayout._128_Nprgs_Hfree_T

    if inp_layout == _T_LAYOUT and len(inp.shape) == 4:
        H0, n_prgs, H1_shard, BxS = inp.shape
        inp = inp.permute(3, 1, 0, 2).reshape(BxS, n_prgs * H0 * H1_shard).unsqueeze(0)
    B, S, H = inp.shape
    T = B * S
    dtype = inp.dtype

    # Step 1: RMSNorm
    rmsnorm_out = rms_norm_torch_ref(inp, gamma, eps=eps, hidden_actual=hidden_actual)
    rmsnorm_out = rmsnorm_out.to(dtype).reshape(T, H)

    # Step 2: Router TopK
    # The kernel runs the router matmul on the bf16 (router_mm_dtype) RMSNorm
    # activation. Round the router's view of rmsnorm_out to that dtype so a sub-ulp
    # top-k logit tie breaks the same way as the kernel; the full-precision
    # rmsnorm_out is still used for the MLP below.
    _, E = router_weights.shape
    router_x = rmsnorm_out
    router_torch_dtype = _NL_TO_TORCH_DTYPE.get(str(router_mm_dtype))
    if router_torch_dtype is not None:
        router_x = rmsnorm_out.to(router_torch_dtype).to(rmsnorm_out.dtype)
    router_outputs = router_topk_torch_ref(
        x=router_x,
        w=router_weights,
        w_bias=router_bias,
        router_logits=torch.zeros(T, E, dtype=dtype),
        expert_affinities=torch.zeros(T, E, dtype=dtype),
        expert_index=torch.zeros(T, top_k, dtype=torch.int32),
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=1,
        x_sb_layout=0,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
    )

    # Step 3: MoE TKG
    # For all-expert mode, slice affinities to local experts and apply masking
    expert_affinities_out = router_outputs["expert_affinities"]
    expert_index_out = router_outputs["expert_index"]
    if is_all_expert and rank_id is not None:
        E_L = (
            expert_gate_up_weights.shape[0] if hasattr(expert_gate_up_weights, 'shape') else len(expert_gate_up_weights)
        )
        rid = rank_id[0, 0].item() if isinstance(rank_id, (torch.Tensor, np.ndarray)) else rank_id
        expert_offset = int(rid) * E_L
        expert_affinities_out = expert_affinities_out[:, expert_offset : expert_offset + E_L].clone()
        if router_pre_norm:  # maps to mask_unselected_experts
            for e in range(E_L):
                mask = (expert_index_out == expert_offset + e).any(dim=1).to(expert_affinities_out.dtype)
                expert_affinities_out[:, e] *= mask

    moe_outputs = moe_tkg_torch_ref(
        hidden_input=rmsnorm_out,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities_out,
        expert_index=expert_index_out,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
        mask_unselected_experts=router_pre_norm,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_fn=hidden_act_fn,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )

    result = {"out": moe_outputs["out"]}
    if outp_layout == _T_LAYOUT:
        out_np = result["out"]
        if isinstance(out_np, torch.Tensor):
            out_np = out_np.float().numpy()
        H0 = 128
        n_prgs_out = 2  # LNC=2
        H1_shard_out = H // (H0 * n_prgs_out)
        # [T, H] -> [T, n_prgs, H0, H1_shard] -> [H0, n_prgs, H1_shard, T]
        result["out"] = out_np.reshape(T, n_prgs_out, H0, H1_shard_out).transpose(2, 1, 3, 0)
    if not skip_router_logits:
        result["router_logits"] = router_outputs["router_logits"]

    return result
