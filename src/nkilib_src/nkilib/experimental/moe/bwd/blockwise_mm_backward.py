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
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from .bwmm_bwd_dropless import blockwise_mm_bwd_dropless
from .moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    KernelTypeOption,
    MOEBwdDroplessBlockingParams,
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
    skip_grad_initialization: bool = False,
    shard_option: ShardOption = ShardOption.SHARD_ON_HIDDEN,
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H,
    kernel_type_option: KernelTypeOption = KernelTypeOption.DROPLESS,
    clamp_limits: ClampLimits = None,
    bias: bool = False,
    activation_type: ActFnType = ActFnType.SiLU,
    block_tile_size: int = None,
    blocking_params: MOEBwdDroplessBlockingParams = None,
    hidden_states_grad_out: nl.ndarray = None,
    expert_affinities_masked_grad_out: nl.ndarray = None,
    gate_up_proj_weight_grad_out: nl.ndarray = None,
    down_proj_weight_grad_out: nl.ndarray = None,
    accumulation_dtype: nki.dtype = None,
    skip_gate_proj: bool = False,
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
        blocking_params (MOEBwdDroplessBlockingParams): Optional blocking hyperparameters
            The kernel consists of 4 matrix multiplications, all using blocked matrix multiplication. This parameter
            controls the number of tiles packed per LHS, RHS, and output block in each matmul. Increasing the number of tiles
            for any dimension increases the amount of data loaded into SBUF before the matmul begins execution. This allows
            more compute per load but also increases SBUF memory consumption. If None, uses defaults. It is highly recommended
            to tune this parameter to maximize kernel performance.
        hidden_states_grad_out (nl.ndarray, optional): Pre-allocated [T, H] output tensor for hidden states
            gradient. If None, allocated internally.
        expert_affinities_masked_grad_out (nl.ndarray, optional): Pre-allocated [T*E, 1] output tensor for
            expert affinity gradient. If None, allocated internally.
        gate_up_proj_weight_grad_out (nl.ndarray, optional): Pre-allocated [E, H, 2, I_TP] output tensor for
            gate/up projection weight gradient. If None, allocated internally.
        down_proj_weight_grad_out (nl.ndarray, optional): Pre-allocated [E, I_TP, H] output tensor for down
            projection weight gradient. If None, allocated internally.
        accumulation_dtype (nki.dtype, optional): Opt-in high-precision dtype for the gradient
            accumulators (hidden/affinity/weight/bias grads). Default None = compute_dtype (baseline,
            unchanged). Set to nl.float32 to accumulate all gradients in fp32 (reduces bf16 accumulation
            error). fp32 accumulation is also auto-enabled if any *_grad_out buffer is fp32. When fp32
            accumulation is requested but the output buffer is bf16, the kernel accumulates into an
            internal fp32 scratch and downcasts to the bf16 buffer on return. Passing a lower-precision
            accumulation_dtype together with fp32 grad-out buffers is contradictory and raises ValueError.

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
    if skip_dma == None:
        skip_dma = SkipMode(False, False)
    if clamp_limits == None:
        clamp_limits = ClampLimits()

    # ------------------------------------------------------------------
    # Resolve grad-output buffers + the effective accumulation dtype.
    #
    # fp32 accumulation is enabled if EITHER the caller allocates fp32 grad-out buffers OR passes
    # accumulation_dtype=float32. When enabled but the output buffer is bf16, the kernel accumulates
    # into an internal fp32 scratch and downcasts to the caller's bf16 buffer at the end -- so callers
    # can keep bf16 grads and still get full-precision (fp32) accumulation. Passing a lower-precision
    # accumulation_dtype together with fp32 grad buffers is contradictory and rejected.
    # ------------------------------------------------------------------
    fp32 = nl.float32
    any_fp32_grad_buffer = (
        (hidden_states_grad_out != None and hidden_states_grad_out.dtype == fp32)
        or (expert_affinities_masked_grad_out != None and expert_affinities_masked_grad_out.dtype == fp32)
        or (gate_up_proj_weight_grad_out != None and gate_up_proj_weight_grad_out.dtype == fp32)
        or (down_proj_weight_grad_out != None and down_proj_weight_grad_out.dtype == fp32)
    )

    # Row-6: a lower-precision accumulation_dtype together with fp32 grad-out buffers is contradictory.
    kernel_assert(
        not (any_fp32_grad_buffer and accumulation_dtype != None and accumulation_dtype != fp32),
        "accumulation_dtype conflicts with fp32 grad-out buffers: pass fp32 grad buffers OR "
        "accumulation_dtype=float32 (not a lower-precision accumulation_dtype together with fp32 buffers).",
    )

    if accumulation_dtype == fp32 or any_fp32_grad_buffer:
        effective_accum_dtype = fp32
    else:
        effective_accum_dtype = compute_dtype

    # Single switch for the whole kernel: when True, accumulate every gradient into an internal fp32
    # scratch buffer and downcast back to the caller's bf16 grad buffers at the end. True only when fp32
    # accumulation is requested via accumulation_dtype while the grad-out buffers are bf16. (If the
    # buffers are already fp32 we accumulate into them directly; bf16 with no opt-in = bf16 baseline.)
    need_fp32_scratch = effective_accum_dtype == fp32 and not any_fp32_grad_buffer

    # Return (output) buffers: caller's if provided, else internally allocated at the io dtype.
    if hidden_states_grad_out != None:
        hidden_states_grad = hidden_states_grad_out
    else:
        hidden_states_grad = nl.ndarray(hidden_states.shape, dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    if expert_affinities_masked_grad_out != None:
        expert_affinities_masked_grad = expert_affinities_masked_grad_out
    else:
        expert_affinities_masked_grad = nl.ndarray(
            expert_affinities_masked.shape, dtype=expert_affinities_masked.dtype, buffer=nl.shared_hbm
        )
    if gate_up_proj_weight_grad_out != None:
        gate_up_proj_weight_grad = gate_up_proj_weight_grad_out
    else:
        gate_up_proj_weight_grad = nl.ndarray(
            gate_up_proj_weight.shape, dtype=gate_up_proj_weight.dtype, buffer=nl.shared_hbm
        )
    if down_proj_weight_grad_out != None:
        down_proj_weight_grad = down_proj_weight_grad_out
    else:
        down_proj_weight_grad = nl.ndarray(down_proj_weight.shape, dtype=down_proj_weight.dtype, buffer=nl.shared_hbm)

    # Kernel (accumulation) buffers: fp32 scratch when need_fp32_scratch, else the output buffers.
    if need_fp32_scratch:
        hidden_states_grad_kbuf = nl.ndarray(hidden_states.shape, dtype=fp32, buffer=nl.shared_hbm)
        expert_affinities_masked_grad_kbuf = nl.ndarray(
            expert_affinities_masked.shape, dtype=fp32, buffer=nl.shared_hbm
        )
        gate_up_proj_weight_grad_kbuf = nl.ndarray(gate_up_proj_weight.shape, dtype=fp32, buffer=nl.shared_hbm)
        down_proj_weight_grad_kbuf = nl.ndarray(down_proj_weight.shape, dtype=fp32, buffer=nl.shared_hbm)
    else:
        hidden_states_grad_kbuf = hidden_states_grad
        expert_affinities_masked_grad_kbuf = expert_affinities_masked_grad
        gate_up_proj_weight_grad_kbuf = gate_up_proj_weight_grad
        down_proj_weight_grad_kbuf = down_proj_weight_grad

    gate_and_up_proj_bias_grad = None
    down_proj_bias_grad = None
    gate_and_up_proj_bias_grad_kbuf = None
    down_proj_bias_grad_kbuf = None
    if bias:
        expert_count, hidden_dim, _, intermediate_dim = gate_up_proj_weight.shape
        # Bias grads are returned at their paired weight-grad output dtype; accumulate via fp32 scratch
        # when need_fp32_scratch, else directly into the output buffer.
        gate_and_up_proj_bias_grad = nl.ndarray(
            (expert_count, 2, intermediate_dim), dtype=gate_up_proj_weight_grad.dtype, buffer=nl.shared_hbm
        )
        down_proj_bias_grad = nl.ndarray(
            (expert_count, hidden_dim), dtype=down_proj_weight_grad.dtype, buffer=nl.shared_hbm
        )
        if need_fp32_scratch:
            gate_and_up_proj_bias_grad_kbuf = nl.ndarray(
                (expert_count, 2, intermediate_dim), dtype=fp32, buffer=nl.shared_hbm
            )
            down_proj_bias_grad_kbuf = nl.ndarray((expert_count, hidden_dim), dtype=fp32, buffer=nl.shared_hbm)
        else:
            gate_and_up_proj_bias_grad_kbuf = gate_and_up_proj_bias_grad
            down_proj_bias_grad_kbuf = down_proj_bias_grad

    params = MOEBwdParameters(
        hidden_states=hidden_states,
        hidden_states_grad=hidden_states_grad_kbuf,
        expert_affinities_masked=expert_affinities_masked,
        expert_affinities_masked_grad=expert_affinities_masked_grad_kbuf,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_proj_weight_grad=gate_up_proj_weight_grad_kbuf,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        down_proj_weight=down_proj_weight,
        down_proj_weight_grad=down_proj_weight_grad_kbuf,
        down_proj_act_checkpoint=None if affinity_option == AffinityOption.AFFINITY_ON_I else down_proj_act_checkpoint,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output_hidden_states_grad=output_hidden_states_grad,
        block_size=block_size,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        skip_grad_initialization=skip_grad_initialization,
        clamp_limits=clamp_limits,
        gate_and_up_proj_bias_grad=gate_and_up_proj_bias_grad_kbuf,
        down_proj_bias_grad=down_proj_bias_grad_kbuf,
        activation_type=activation_type,
        affinity_option=affinity_option,
        blocking_params=blocking_params,
        shard_option=shard_option,
        accumulation_dtype=effective_accum_dtype,
        skip_gate_proj=skip_gate_proj,
    )

    params.validate()
    params.validate_sharding(nl.num_programs(axes=0))

    blockwise_mm_bwd_dropless(params)

    # Downcast the fp32 scratch accumulators back to the caller's bf16 grad buffers (HBM->HBM DMA cast).
    if need_fp32_scratch:
        nisa.dma_copy(dst=hidden_states_grad, src=hidden_states_grad_kbuf)
        nisa.dma_copy(dst=expert_affinities_masked_grad, src=expert_affinities_masked_grad_kbuf)
        nisa.dma_copy(dst=gate_up_proj_weight_grad, src=gate_up_proj_weight_grad_kbuf)
        nisa.dma_copy(dst=down_proj_weight_grad, src=down_proj_weight_grad_kbuf)
        if bias:
            nisa.dma_copy(dst=gate_and_up_proj_bias_grad, src=gate_and_up_proj_bias_grad_kbuf)
            nisa.dma_copy(dst=down_proj_bias_grad, src=down_proj_bias_grad_kbuf)

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
