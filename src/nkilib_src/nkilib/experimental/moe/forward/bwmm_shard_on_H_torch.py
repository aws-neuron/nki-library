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

"""PyTorch reference implementation for blockwise_mm_baseline_shard_hidden kernel."""

import nki.language as nl
import torch

from ....core.utils.common_types import ExpertAffinityScaleMode
from .bwmm_shard_on_H import SkipMode


def _silu(x):
    """SiLU activation for torch tensors."""
    return x * torch.sigmoid(x)


def blockwise_mm_baseline_shard_hidden_torch_ref(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    block_size,
    token_position_to_id,
    block_to_expert,
    gate_up_activations_T=None,
    down_activations=None,
    skip_dma=SkipMode(),
    compute_dtype=nl.bfloat16,
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    activation_dtype=nl.bfloat16,
    accum_dtype=nl.bfloat16,
):
    """Torch reference for blockwise_mm_baseline_shard_hidden."""
    T, H = hidden_states.shape
    B = block_size
    E = down_proj_weight.shape[0]
    I_TP = gate_up_proj_weight.shape[3]
    N = token_position_to_id.shape[0] // B

    expert_affinities = expert_affinities_masked.reshape(-1, E)
    down_proj_weights = down_proj_weight[:, :I_TP, :]
    gate_up_weights = gate_up_proj_weight.reshape(E, H, 2 * I_TP)

    token_position_to_id_reshaped = token_position_to_id.reshape(N, B)
    block_to_expert_flat = block_to_expert.flatten()

    checkpoint_activation = gate_up_activations_T is not None

    output = torch.zeros((T, H), dtype=hidden_states.dtype)

    if checkpoint_activation:
        gate_up_act_T = torch.zeros(N, 2, I_TP, B, dtype=hidden_states.dtype)
        down_act = torch.zeros(N, B, H, dtype=hidden_states.dtype)

    for b in range(N):
        local_token_ids = token_position_to_id_reshaped[b, :]
        local_hidden = hidden_states[local_token_ids.long(), :]

        expert_idx = block_to_expert_flat[b]
        local_affinities = expert_affinities[local_token_ids.long(), expert_idx].reshape(-1, 1)

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
            local_hidden = local_affinities * local_hidden

        gate_up = torch.matmul(local_hidden, gate_up_weights[expert_idx]).reshape(B, 2, I_TP)
        gate = gate_up[:, 0, :]
        up = gate_up[:, 1, :]

        if checkpoint_activation:
            gate_up_act_T[b] = gate_up.permute(1, 2, 0).to(hidden_states.dtype)

        intermediate = _silu(gate) * up

        down = torch.matmul(intermediate, down_proj_weights[expert_idx])

        if checkpoint_activation:
            down_act[b] = down.to(hidden_states.dtype)

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            down = down * local_affinities

        output[local_token_ids.long(), :] += down.to(hidden_states.dtype)

    return (
        {"output": output} if not checkpoint_activation else {"output": output, "gate_up_activations_T": gate_up_act_T}
    )
