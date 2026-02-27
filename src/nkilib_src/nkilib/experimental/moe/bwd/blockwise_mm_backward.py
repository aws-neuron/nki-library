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

"""Backward pass kernel for blockwise matrix multiplication in Mixture of Experts."""

import nki
import nki.language as nl

from .bwmm_bwd_dropless import blockwise_mm_bwd_dropless
from .moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    KernelTypeOption,
    MOEBwdParameters,
    ShardOption,
    SkipMode,
)


@nki.jit
def blockwise_mm_bwd(
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    gate_up_proj_act_checkpoint_T: nl.ndarray,
    down_proj_act_checkpoint: nl.ndarray,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    output_hidden_states_grad: nl.ndarray,
    block_size: int,
    skip_dma: SkipMode = None,
    compute_dtype: nki.dtype = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    shard_option: ShardOption = ShardOption.SHARD_ON_HIDDEN,
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H,
    kernel_type_option: KernelTypeOption = KernelTypeOption.DROPLESS,
    clamp_limits: ClampLimits = None,
    bias: bool = False,
    activation_type: ActFnType = ActFnType.SiLU,
    block_tile_size: int = None,
) -> tuple:
    """
    Compute backward pass for blockwise MoE layer.

    This kernel computes gradients for all parameters in a Mixture of Experts layer
    using blockwise matrix multiplication. Optimized for dropless MoE with variable
    block assignments per expert.

    TODO: Specify intended usage range (e.g., block sizes, hidden dimensions)

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size
        I_TP: Intermediate size / tensor parallel degree
        E: Number of experts
        B: Block size (tokens per block)
        N: Number of blocks

    Args:
        hidden_states (nl.ndarray): [T, H], Input hidden states on HBM.
        expert_affinities_masked (nl.ndarray): [T * E, 1], Expert affinities on HBM.
        gate_up_proj_weight (nl.ndarray): [E, H, 2, I_TP], Gate/up projection weights on HBM.
        down_proj_weight (nl.ndarray): [E, I_TP, H], Down projection weights on HBM.
        gate_up_proj_act_checkpoint_T (nl.ndarray): [N, 2, I_TP, B], Checkpointed gate/up activations.
        down_proj_act_checkpoint (nl.ndarray): [N, B, H], Checkpointed down projection activations.
        token_position_to_id (nl.ndarray): [N * B], Token position to block mapping.
        block_to_expert (nl.ndarray): [N, 1], Expert index per block.
        output_hidden_states_grad (nl.ndarray): [T, H], Upstream gradient from output.
        block_size (int): Number of tokens per block (128, 256, 512, or 1024).
        skip_dma (SkipMode): OOB handling mode for DMA operations.
        compute_dtype (nki.dtype): Computation dtype (default: nl.bfloat16).
        is_tensor_update_accumulating (bool): Whether to accumulate into existing gradients.
        shard_option (ShardOption): Sharding strategy selection.
        affinity_option (AffinityOption): Affinity scaling dimension.
        kernel_type_option (KernelTypeOption): Token dropping strategy.
        clamp_limits (ClampLimits): Gradient clamping limits.
        bias (bool): Whether to compute bias gradients.
        activation_type (ActFnType): Activation function type.
        block_tile_size (int): Optional tile size override.

    Returns:
        tuple: Gradient tensors:
            - hidden_states_grad (nl.ndarray): [T, H], Gradient for hidden states.
            - expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Gradient for affinities.
            - gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Gradient for gate/up weights.
            - down_proj_weight_grad (nl.ndarray): [E, I_TP, H], Gradient for down weights.
            - gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP], Bias gradients if bias=True.
            - down_proj_bias_grad (nl.ndarray, optional): [E, H], Down bias gradients if bias=True.

    Notes:
        - block_size must be one of: 128, 256, 512, 1024.
        - H must be divisible by num_shards for LNC sharding.
        - Currently only supports DROPLESS kernel type.

    Pseudocode:
        TODO: Add pseudocode description
    """
    if skip_dma is None:
        skip_dma = SkipMode(False, False)
    if clamp_limits is None:
        clamp_limits = ClampLimits()

    hidden_states_grad = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    expert_affinities_masked_grad = nl.ndarray(
        expert_affinities_masked.shape, dtype=expert_affinities_masked.dtype, buffer=nl.shared_hbm
    )

    gate_up_proj_weight_grad = nl.ndarray(
        gate_up_proj_weight.shape, dtype=gate_up_proj_weight.dtype, buffer=nl.shared_hbm
    )

    down_proj_weight_grad = nl.ndarray(down_proj_weight.shape, dtype=down_proj_weight.dtype, buffer=nl.shared_hbm)

    gate_and_up_proj_bias_grad = None
    down_proj_bias_grad = None
    if bias:
        expert_count, hidden_dim, _, intermediate_dim = gate_up_proj_weight_grad.shape
        gate_and_up_proj_bias_grad = nl.ndarray(
            shape=(expert_count, 2, intermediate_dim), dtype=compute_dtype, buffer=nl.shared_hbm
        )
        down_proj_bias_grad = nl.ndarray(shape=(expert_count, hidden_dim), dtype=compute_dtype, buffer=nl.shared_hbm)

    params = MOEBwdParameters(
        hidden_states=hidden_states,
        hidden_states_grad=hidden_states_grad,
        expert_affinities_masked=expert_affinities_masked,
        expert_affinities_masked_grad=expert_affinities_masked_grad,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_proj_weight_grad=gate_up_proj_weight_grad,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        down_proj_weight=down_proj_weight,
        down_proj_weight_grad=down_proj_weight_grad,
        down_proj_act_checkpoint=down_proj_act_checkpoint,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        clamp_limits=clamp_limits,
        gate_and_up_proj_bias_grad=gate_and_up_proj_bias_grad,
        down_proj_bias_grad=down_proj_bias_grad,
        activation_type=activation_type,
    )

    params.validate()
    params.validate_sharding(nl.num_programs(axes=0))

    blockwise_mm_bwd_dropless(params)

    if bias:
        return (
            hidden_states_grad,
            expert_affinities_masked_grad,
            gate_up_proj_weight_grad,
            down_proj_weight_grad,
            gate_and_up_proj_bias_grad,
            down_proj_bias_grad,
        )
    return hidden_states_grad, expert_affinities_masked_grad, gate_up_proj_weight_grad, down_proj_weight_grad
