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

"""PyTorch reference implementation for MoE CTE blockwise matrix multiplication kernels."""

from typing import Optional

import torch

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from .moe_cte_torch_utils import torch_act_fn
from .moe_cte_utils import SkipMode


def moe_cte_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    block_size: int,
    bwmm_func,
    lnc_degree: int = 2,
    conditions: Optional[torch.Tensor] = None,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    n_block_per_iter: int = 1,
    block_sharding_strategy=None,
    num_static_block: Optional[int] = None,
    gate_up_activations_T: Optional[torch.Tensor] = None,
    down_activations: Optional[torch.Tensor] = None,
    top_k: int = 1,
) -> dict:
    """
    PyTorch reference implementation of blockwise MoE matrix multiplication.

    This is a line-by-line port of the numpy golden (generate_blockwise_numpy_golden)
    to PyTorch, used for testing the NKI MoE CTE kernels.

    Args:
        hidden_states: [T+1, H], Input token embeddings (T+1 includes padding token)
        expert_affinities_masked: [(T+1)*E, 1], Expert routing weights
        gate_up_proj_weight: [E, H, 2, I_TP], Gate and up projection weights
        down_proj_weight: [E, I_TP_padded, H], Down projection weights (may be padded on I dimension)
        token_position_to_id: [N*B], Token to block position mapping
        block_to_expert: [N], Expert assignment per block
        block_size: Tokens per block
        bwmm_func: BWMMFunc enum indicating kernel variant
        lnc_degree: LNC degree
        top_k: Number of top experts per token

    Returns:
        dict with 'output' and optionally 'gate_up_activations_T', 'down_activations'
    """
    from test.integration.nkilib.core.moe.moe_cte.test_moe_cte_common import BWMMFunc

    E = gate_up_proj_weight.shape[0]
    H = hidden_states.shape[-1]
    I_TP = gate_up_proj_weight.shape[-1]
    B = block_size

    is_block_parallel = bwmm_func == BWMMFunc.SHARD_ON_BLOCK
    is_dropping = bwmm_func == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING
    is_shard_block = bwmm_func == BWMMFunc.SHARD_ON_BLOCK
    do_checkpoint = checkpoint_activation or is_dropping

    has_quantize = gate_up_proj_scale is not None and gate_up_proj_scale.numel() > 0
    quantize_strategy = 6 if has_quantize else 0

    block_to_expert_flat = block_to_expert.flatten().long()
    N = block_to_expert_flat.shape[0]
    separate_outputs = is_block_parallel and top_k > 1

    # Match numpy golden: T is derived from hidden_states shape
    # When skip_token=False: hidden_states is [T+1, H], so T = shape[0] - 1
    # When skip_token=True: hidden_states is [T, H], so T = shape[0]
    T = hidden_states.shape[0] - 1 if not skip_dma.skip_token else hidden_states.shape[0]

    # Output shape matches numpy golden: always [T+1, ...]
    if is_shard_block:
        output_shape = [T + 1, lnc_degree, H] if separate_outputs else [T + 1, H]
    else:
        output_shape = [lnc_degree, T + 1, H] if separate_outputs else [T + 1, H]

    # Use bfloat16 for output accumulation to match numpy golden behavior
    output = torch.zeros(output_shape, dtype=torch.float32)
    token_pos_2d = token_position_to_id.long().reshape(N, B)

    if do_checkpoint:
        ckpt_gate_up = torch.zeros(N, 2, I_TP, B, dtype=torch.float32)
        ckpt_down = torch.zeros(N, B, H, dtype=torch.float32)

    # Reshape weights: [E, H, 2*I_TP]
    gate_up_w = gate_up_proj_weight.float().reshape(E, H, 2 * I_TP)
    down_w = down_proj_weight.float()

    # Handle quantize_strategy == 5 scale transpose (matching numpy golden)
    down_scale_work = None
    if has_quantize and quantize_strategy == 5 and down_proj_scale is not None:
        ds = down_proj_scale.float()
        if ds.shape[0] == E:
            down_scale_work = ds.reshape(E, 128, H // 128).permute(0, 2, 1).reshape(E, 1, H)
        else:
            down_scale_work = ds
    elif has_quantize and down_proj_scale is not None:
        down_scale_work = down_proj_scale.float()

    # Working copies that grow for skip_token (matching numpy golden)
    hidden_work = hidden_states.float()
    affinities_2d = expert_affinities_masked.float().reshape(-1, E)

    gup_scale = gate_up_proj_scale.float() if has_quantize and gate_up_proj_scale is not None else None

    for b_idx in range(N):
        if conditions is not None and conditions[b_idx].item() == 0:
            break

        local_ids = token_pos_2d[b_idx]
        expert_idx = block_to_expert_flat[b_idx].item()

        # For skip_token, append zero row each iteration (matching numpy golden exactly)
        if skip_dma.skip_token:
            hidden_work = torch.cat([hidden_work, torch.zeros(1, H)], dim=0)
            affinities_2d = torch.cat([affinities_2d, torch.zeros(1, E)], dim=0)

        local_hidden = hidden_work[local_ids].float()
        local_affinities = affinities_2d[local_ids, expert_idx].unsqueeze(1).to(hidden_states.dtype)

        if expert_affinities_scaling_mode in [
            ExpertAffinityScaleMode.PRE_SCALE,
            ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        ]:
            local_hidden = local_affinities * local_hidden

        if expert_idx >= E:
            continue

        # Gate-up projection: [B, H] @ [H, 2*I_TP] -> [B, 2, I_TP]
        gate_up_act = torch.matmul(local_hidden, gate_up_w[expert_idx]).reshape(B, 2, I_TP)
        gate_act = gate_up_act[:, 0, :].clone()
        up_act = gate_up_act[:, 1, :].clone()

        # Apply quantization scales
        if has_quantize and gup_scale is not None:
            if gup_scale.shape[0] == 1:
                gate_act *= gup_scale.squeeze()[:I_TP]
                up_act *= gup_scale.squeeze()[I_TP:]
            elif gup_scale.shape[0] == E:
                gate_act *= gup_scale[expert_idx, 0, :I_TP]
                up_act *= gup_scale[expert_idx, 0, I_TP:]

        # Apply bias
        if gate_and_up_proj_bias is not None:
            gate_act += gate_and_up_proj_bias[expert_idx, 0, :]
            up_act += gate_and_up_proj_bias[expert_idx, 1, :]

        # Apply clamping
        if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
            gate_act = torch.clamp(gate_act, min=gate_clamp_lower_limit, max=gate_clamp_upper_limit)
        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            up_act = torch.clamp(up_act, min=up_clamp_lower_limit, max=up_clamp_upper_limit)

        if do_checkpoint:
            ckpt_gate_up[b_idx] = gate_up_act.permute(1, 2, 0)

        # Activation + element-wise multiply
        intermediate = torch_act_fn(gate_act, activation_function) * up_act

        # Down projection: [B, I_TP] @ [I_TP_padded, H] -> [B, H]
        down_act = torch.matmul(intermediate, down_w[expert_idx])

        # Apply down quantization scale
        if has_quantize and down_scale_work is not None:
            if down_scale_work.shape[0] == 1:
                down_act = down_act * down_scale_work.squeeze()
            elif down_scale_work.shape[0] == E:
                down_act = down_act * down_scale_work[expert_idx, 0, :]

        # Apply down bias
        if down_proj_bias is not None:
            down_act += down_proj_bias[expert_idx]

        if do_checkpoint:
            ckpt_down[b_idx] = down_act

        # Apply expert affinity scaling
        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            scaled = down_act * local_affinities
        else:
            scaled = down_act

        # Accumulate output (cast to bfloat16 then back, matching numpy golden .astype(dtype))
        scaled_bf16 = scaled.to(torch.bfloat16).to(torch.float32)
        if separate_outputs:
            if is_shard_block:
                output[local_ids, 0, :] += scaled_bf16
            else:
                output[0, local_ids, :] += scaled_bf16
        else:
            output[local_ids, :] += scaled_bf16

    # Slice for skip_token (matching numpy golden)
    if skip_dma.skip_token:
        if separate_outputs:
            if is_shard_block:
                output = output[:T, :, :]
            else:
                output = output[:, :T, :]
        else:
            output = output[:T, :]

    result = {"output": output}
    if do_checkpoint:
        result["gate_up_activations_T"] = ckpt_gate_up
        if not expert_affinity_multiply_on_I:
            result["down_activations"] = ckpt_down
    return result
