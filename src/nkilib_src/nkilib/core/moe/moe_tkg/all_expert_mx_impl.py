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

"""All-expert MoE token generation implementation with MX (microscaling) quantization support."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from ...mlp.mlp_parameters import MLPParameters
from ...mlp.mlp_tkg.projection_mx_constants import (
    GATE_FUSED_IDX,
    MX_SCALE_DTYPE,
    SUPPORTED_QMX_OUTPUT_DTYPES,
    UP_FUSED_IDX,
    _q_width,
)
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from .all_expert_mx_utils import (
    _NONZERO_WITH_COUNT_PAD_VAL,
    _NUM_H4_FOLDS_PER_COLUMN,
    AllExpertMXDimensions,
    AllExpertMXDynamismConfig,
    AllExpertMXInputTensors,
    AllExpertMXKernelConfig,
    ExpertWeightsSBUF,
    init_all_expert_mx_configs,
    validate_all_expert_mx_inputs,
)
from .down_projection_mx_shard_I import (
    down_projection_mx_shard_I,
    load_broadcast_down_weight_scale_bias,
)
from .gate_up_projection_mx_shard_I import (
    gate_up_projection_mx_shard_I,
    load_gate_up_weight_scale_bias,
)


@nki.jit
def _all_expert_moe_tkg_mx(
    mlp_params: MLPParameters,
    output: nl.ndarray,
    activation_compute_dtype: nki.dtype = nl.bfloat16,
    is_all_expert_dynamic: bool = False,
    block_size: int = None,
) -> nl.ndarray:
    """
    Perform all-expert MoE MLP on input using microscaling format (MX) weights.

    Shards compute on intermediate dimension when run with LNC=2.

    Dimensions:
        B: Batch size
        S: Sequence length
        T: Total number of input tokens (equivalent to B*S)
        H: Hidden dimension size of the model
        I: Intermediate dimension size of the model after tensor parallelism
        E_L: Number of local experts after expert parallelism

    Args:
        mlp_params (MLPParameters): MLPParameters containing all input tensors and configuration, including:
        output (nl.ndarray): [min(T, 128), ⌈T/128⌉, H] in SBUF or [T, H] in HBM output tensor.
        activation_compute_dtype: Compute dtype for activations.
        is_all_expert_dynamic: Whether to use dynamic control flow. Improves performance when T is large.
        block_size (int): Block size for dynamic control flow. Required when is_all_expert_dynamic=True.

    Returns:
        output (nl.ndarray): [T, H] in HBM or [min(T, 128), ⌈T/128⌉, H] in SBUF, Output tensor with MoE results.

    Pseudocode:
        # Step 1: Load and quantize input (skipped if hidden_input_scale provided)
        input_quant, input_scale = layout_adapter(input)

        # Step 2: Process each expert sequentially
        for expert_idx in range(E_L):
            # Load expert weights
            gate_w, up_w, down_w = load_one_expert(expert_idx)

            # --- Static algorithm (is_all_expert_dynamic=False) ---
            # Compute gate/up projection and activation
            act = gate_up_projection(input_quant, gate_w, up_w)

            # Compute down projection with affinity scaling
            expert_out = down_projection(act, down_w)
            if affinity_scaling_mode == POST_SCALE:
                expert_out *= expert_affinities[expert_idx]

            # Accumulate results
            if expert_idx == 0:
                output = expert_out
            else:
                output += expert_out

            # --- Dynamic algorithm (is_all_expert_dynamic=True) ---
            # Find indices of tokens routed to this expert
            routed_indices, dynamic_decision = nonzero_with_count(expert_affinities[expert_idx])

            # Compute static blocks (always executed)
            for block_idx in range(n_static_blocks):
                block_input = gather(input, routed_indices[block_idx])
                block_out = expert_mlp(block_input, gate_w, up_w, down_w)
                scatter(output, routed_indices[block_idx], block_out)

            # Compute dynamic blocks (skipped at runtime if no routed tokens)
            dynamic_block_idx = 0
            while dynamic_decision[dynamic_block_idx]:
                block_input = gather(input, routed_indices[n_static_blocks + dynamic_block_idx])
                block_out = expert_mlp(block_input, gate_w, up_w, down_w)
                scatter(output, routed_indices[n_static_blocks + dynamic_block_idx], block_out)
                dynamic_block_idx += 1
    """

    # Initialize configs and validate inputs
    input_tensors, kernel_cfg, dims, dynamism_cfg = init_all_expert_mx_configs(
        mlp_params=mlp_params,
        output=output,
        activation_compute_dtype=activation_compute_dtype,
        is_all_expert_dynamic=is_all_expert_dynamic,
        block_size=block_size,
    )
    validate_all_expert_mx_inputs(input_tensors, kernel_cfg, dims, dynamism_cfg)

    # Dispatch to expert MLP implementation
    if dynamism_cfg.is_all_expert_dynamic:
        _all_expert_mx_dynamic(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            dynamism_cfg=dynamism_cfg,
        )
    else:
        _all_expert_mx_static(input_tensors=input_tensors, kernel_cfg=kernel_cfg, dims=dims)

    return output


@nki.jit
def _all_expert_mx_static(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
) -> nl.ndarray:
    """
    Static all-expert MoE computation without dynamic loop on chip (DLoC).

    Processes all experts sequentially, computing MLP(all tokens) for each expert
    before moving to the next. This is optimal when DLoC overhead exceeds benefits.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.

    Returns:
        nl.ndarray: Output tensor with MoE computation results.
    """

    # Step 1: Process inputs
    # Step 1.2: Optional load + swizzle + QMX hidden states
    if kernel_cfg.input_in_sbuf:
        # Input must be swizzled + MX quantized upstream when it is already in SBUF
        kernel_assert(
            input_tensors.hidden_input_scale != None,
            f"Expected pre-quantized input when input is in SBUF, "
            f"got {input_tensors.hidden_input.dtype=} {input_tensors.hidden_input_scale=}",
        )
        if dims.shard_on_T:
            input_quant_sb = input_tensors.hidden_input[:, :, nl.ds(dims.T_offset, dims.T_local)]
            input_scale_sb = input_tensors.hidden_input_scale[:, :, nl.ds(dims.T_offset, dims.T_local)]
        else:
            input_quant_sb, input_scale_sb = input_tensors.hidden_input, input_tensors.hidden_input_scale
    else:
        input_quant_sb, input_scale_sb = _layout_adapter_qmx_hbm(
            input=input_tensors.hidden_input,
            dims=dims,
        )

    # Step 1.2: View expert_affinities_masked and output_hbm based on sharding decision
    if dims.shard_on_T:
        T_eff = dims.T_local
        T_hbm_offset = dims.T_offset
        output_hbm_view = input_tensors.output[nl.ds(dims.T_offset, dims.T_local), :]
        if len(input_tensors.expert_affinities_masked.shape) == 3:
            # 3D tiled layout [pmax, n_T128_tiles, E_L]
            tile_start = T_hbm_offset // dims.pmax
            n_local_tiles = div_ceil(T_eff, dims.pmax)
            expert_affinities_masked_sb = input_tensors.expert_affinities_masked[:, nl.ds(tile_start, n_local_tiles), :]
        else:
            # 2D layout [T, E_L]
            expert_affinities_masked_sb = input_tensors.expert_affinities_masked[nl.ds(T_hbm_offset, T_eff), :]
    else:
        expert_affinities_masked_sb = input_tensors.expert_affinities_masked
        output_hbm_view = input_tensors.output

    # Step 2: Allocate output
    output_shape = (dims.tile_T, dims.n_tiles_in_T, dims.H)
    output_sb = nl.ndarray(output_shape, dtype=kernel_cfg.activation_compute_dtype, buffer=nl.sbuf)

    # Step 3: Compute expert MLPs sequentially
    for expert_idx in nl.sequential_range(dims.E_L):
        # Step 3.1: Load weights for this expert
        weights = _load_expert(input_tensors=input_tensors, kernel_cfg=kernel_cfg, dims=dims, expert_idx=expert_idx)

        # Step 3.2: Compute MLP for this expert
        _compute_expert_mlp(
            input_quant=input_quant_sb,
            input_scale=input_scale_sb,
            weights=weights,
            kernel_cfg=kernel_cfg,
            expert_affinities_masked=expert_affinities_masked_sb,
            output_sb=output_sb[...],
            output_hbm=output_hbm_view[...] if (not kernel_cfg.output_in_sbuf) else None,
            expert_idx=expert_idx,
            is_first_expert=(expert_idx == 0),
            is_last_expert=(expert_idx == dims.E_L - 1),
            shard_on_I=dims.shard_on_I,
            shard_on_T=dims.shard_on_T,
        )

    return input_tensors.output


@nki.jit
def _all_expert_mx_dynamic(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig,
) -> nl.ndarray:
    """
    All-expert MoE computation with dynamic control flow (DLoC).

    Processes all experts sequentially, with tokens split into blocks for each expert.
    When a block contains routed tokens, we compute MLP(block tokens).
    A portion of the blocks for each expert are dynamically skipped at runtime if none
    of the tokens in a dynamic block are routed to the expert.
    Dynamism can provide performance improvements relative to _all_expert_mx_static when T is large.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.

    Returns:
        nl.ndarray: Output tensor with MoE computation results.
    """

    # Step 1: Arange [0, 1, 2, 3] for token indices broadcast
    arange_4H = nl.ndarray((1, _q_width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(arange_4H, [[1, _q_width]], offset=0)

    # Step 2: Compute expert MLPs sequentially
    for expert_idx in nl.sequential_range(dims.E_L):
        # Step 2.1: Load weights for current expert
        weights = _load_expert(input_tensors=input_tensors, kernel_cfg=kernel_cfg, dims=dims, expert_idx=expert_idx)

        # Step 2.2: Find indices of tokens routed to current expert, build dynamic block decision vector
        routed_token_indices_with_count_sb, dynamic_decision_sb = _find_expert_routed_tokens(
            input_tensors=input_tensors, dims=dims, dynamism_cfg=dynamism_cfg, expert_idx=expert_idx
        )

        # Step 2.3: Compute static blocks
        for static_block_idx in nl.sequential_range(dynamism_cfg.n_static_blocks):
            _compute_block(
                input_tensors=input_tensors,
                kernel_cfg=kernel_cfg,
                dims=dims,
                dynamism_cfg=dynamism_cfg,
                weights=weights,
                routed_token_indices=routed_token_indices_with_count_sb,
                arange_4H=arange_4H,
                expert_idx=expert_idx,
                block_idx=static_block_idx,
                is_dynamic_block=False,
            )

        # Step 2.4: Compute dynamic blocks
        # Step 2.4.1: Move index/decision vectors to HBM, so that each dynamic block can indirect gather its corresponding data
        # TODO: utilize SBUF->SBUF indirection instead of HBM->SBUF indirection
        dynamic_block_token_indices_hbm, dynamic_decision_hbm = _init_dynamic_block_indices_hbm(
            dims=dims,
            dynamism_cfg=dynamism_cfg,
            routed_token_indices_with_count_sb=routed_token_indices_with_count_sb,
            dynamic_decision_sb=dynamic_decision_sb,
        )

        # Step 2.4.2: Initialize dynamic register + loop iteration counter
        compute_next_dynamic_block = nisa.register_alloc(dynamic_decision_sb[0, 0])
        dynamic_block_idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dynamic_block_idx, 0)

        # Step 2.4.3: Dynamic loop over dynamic blocks
        while compute_next_dynamic_block:
            _compute_block(
                input_tensors=input_tensors,
                kernel_cfg=kernel_cfg,
                dims=dims,
                dynamism_cfg=dynamism_cfg,
                weights=weights,
                routed_token_indices=dynamic_block_token_indices_hbm,
                arange_4H=arange_4H,
                expert_idx=expert_idx,
                block_idx=dynamic_block_idx,
                is_dynamic_block=True,
            )

            # Step 2.4.4: Update dynamic register
            nisa.tensor_scalar(
                data=dynamic_block_idx,
                op0=nl.add,
                operand0=1,
                dst=dynamic_block_idx,
            )
            next_decision = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                src=dynamic_decision_hbm.ap(
                    pattern=[[dynamism_cfg.n_dynamic_blocks_plus_1, 1], [1, 1]],
                    offset=0,
                    scalar_offset=dynamic_block_idx,
                    indirect_dim=1,
                ),
                dst=next_decision,
            )
            nisa.register_load(
                src=next_decision,
                dst=compute_next_dynamic_block,
            )

    # FIXME: the following code block avoids a NKI failure when no ops follow
    # a dynamic control flow block; remove when NKI fixes bug
    rng_seeds_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(rng_seeds_sb, 0.0)
    nisa.set_rng_seed(rng_seeds_sb)

    return input_tensors.output


@nki.jit
def _find_expert_routed_tokens(input_tensors, dims, dynamism_cfg, expert_idx):
    """
    Find indices of tokens routed to a specific expert and build dynamic block decision vector.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        expert_idx (int): Index of the expert to find routed tokens for.

    Returns:
        routed_token_indices_with_count_sb (nl.ndarray): [pmax, T+1], Output from nonzero_with_count in SBUF.
            Partition 0 contains routed token indices with count in final element. Partitions 1..pmax are padding.
        dynamic_conditions_sb (nl.ndarray): [1, n_dynamic_blocks+1], Decision vector indicating
            which dynamic blocks need computation.
    """

    # Allocations
    expert_affinities_masked_T_f32_sb = nl.ndarray((dims.pmax, dims.T), dtype=nl.float32, buffer=nl.sbuf)
    routed_token_indices_with_count_sb = nl.ndarray((dims.pmax, dynamism_cfg.T_plus_1), dtype=nl.int32, buffer=nl.sbuf)
    dynamic_conditions_sb = nl.ndarray((1, dynamism_cfg.n_dynamic_blocks_plus_1), dtype=nl.int32, buffer=nl.sbuf)
    count_nonzero_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)

    # DMA transpose pre-sliced expert affinities from [T, E_L] -> [1, T] for index calculation
    # nonzero_with_count requires 32bit input (s32 or f32)
    needs_cast = input_tensors.expert_affinities_masked.dtype != nl.float32
    if needs_cast:
        expert_affinities_masked_T_sb = nl.ndarray(
            (1, dims.T), dtype=input_tensors.expert_affinities_masked.dtype, buffer=nl.sbuf
        )
    load_dst = expert_affinities_masked_T_sb if needs_cast else expert_affinities_masked_T_f32_sb
    # TODO: support loading multiple experts' affinities into every 16th partition for consumption by nonzero_with_count
    nisa.dma_transpose(
        src=input_tensors.expert_affinities_masked.ap(
            pattern=[[dims.E_L, dims.T], [1, 1], [1, 1], [1, 1]],
            offset=expert_idx,
        ),
        dst=load_dst.ap(
            pattern=[[dims.T, 1], [1, 1], [1, 1], [1, dims.T]],
            offset=0,
        ),
    )
    if needs_cast:
        nisa.tensor_copy(
            src=expert_affinities_masked_T_sb[...],
            dst=expert_affinities_masked_T_f32_sb[0, :],
        )

    # Find nonzero indices, with count
    # NOTE: partitions 1, ..., pmax are padding from nonzero_with_count output shape requirement
    nisa.nonzero_with_count(
        src=expert_affinities_masked_T_f32_sb[...],
        padding_val=_NONZERO_WITH_COUNT_PAD_VAL,
        dst=routed_token_indices_with_count_sb[...],
    )

    # Build boolean dynamic block decision vector, with final element 0 to ensure loop terminates when all dynamic blocks are computed
    # Example: iota=[129, 257, 385, 513], count=[483], less(iota, count)=[1, 1, 1, 0]
    nisa.tensor_copy(
        src=routed_token_indices_with_count_sb[0, dims.T],
        dst=count_nonzero_f32,  # tensor_scalar operand must be f32
    )
    nisa.iota(
        dst=dynamic_conditions_sb[...],
        pattern=[[dynamism_cfg.block_size, dynamism_cfg.n_dynamic_blocks_plus_1]],
        offset=dynamism_cfg.n_static_blocks * dynamism_cfg.block_size + 1,
    )
    nisa.tensor_scalar(
        data=dynamic_conditions_sb[...],
        op0=nl.less,
        operand0=count_nonzero_f32,
        dst=dynamic_conditions_sb[...],
    )

    return routed_token_indices_with_count_sb, dynamic_conditions_sb


@nki.jit
def _init_dynamic_block_indices_hbm(dims, dynamism_cfg, routed_token_indices_with_count_sb, dynamic_decision_sb):
    """
    Move routed token indices corresponding to dynamic blocks and dynamic block decision vector to HBM. Data from these buffers
    will be reloaded with an indirect DMA inside the dynamic block loop.

    Args:
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        routed_token_indices_with_count_sb (nl.ndarray): [pmax, T+1], Output from nonzero_with_count in SBUF.
            Partition 0 contains routed token indices with count in final element. Partitions 1..pmax are padding.
        dynamic_decision_sb (nl.ndarray): [1, n_dynamic_blocks+1], Dynamic block decision vector in SBUF.

    Returns:
        dynamic_block_token_indices_hbm (nl.ndarray): [n_dynamic_blocks, block_size], Token indices in shared HBM.
        dynamic_decision_hbm (nl.ndarray): [1, n_dynamic_blocks+1], Decision vector in private HBM.
    """

    # Allocations
    n_static_tokens = dynamism_cfg.n_static_blocks * dynamism_cfg.block_size
    n_dynamic_tokens = dynamism_cfg.n_dynamic_blocks * dynamism_cfg.block_size
    n_dynamic_tokens_local = n_dynamic_tokens // 2 if dims.n_prgs > 1 else n_dynamic_tokens
    dynamic_token_offset = n_dynamic_tokens_local if dims.prg_id > 0 else 0
    dynamic_block_token_indices_hbm = nl.ndarray(
        (1, n_dynamic_tokens),
        dtype=routed_token_indices_with_count_sb.dtype,
        buffer=nl.shared_hbm,
        name='dynamic_block_token_indices_hbm',
    )
    dynamic_decision_hbm = nl.ndarray(
        dynamic_decision_sb.shape,
        dtype=dynamic_decision_sb.dtype,
        buffer=nl.private_hbm,
    )

    # Save buffers to HBM
    # TODO: use private HBM for dynamic_block_token_indices_hbm to skip CB
    nisa.dma_copy(
        src=routed_token_indices_with_count_sb[
            0, nl.ds(n_static_tokens + dynamic_token_offset, n_dynamic_tokens_local)
        ],
        dst=dynamic_block_token_indices_hbm[0, nl.ds(dynamic_token_offset, n_dynamic_tokens_local)],
    )
    dynamic_block_token_indices_hbm = dynamic_block_token_indices_hbm.reshape(
        (dynamism_cfg.n_dynamic_blocks, dynamism_cfg.block_size)
    )
    nisa.core_barrier(dynamic_block_token_indices_hbm, [0, 1])
    nisa.dma_copy(dynamic_decision_hbm, dynamic_decision_sb)

    return dynamic_block_token_indices_hbm, dynamic_decision_hbm


@nki.jit
def _get_block_token_position_to_id(dynamism_cfg, routed_token_indices, arange_4H, block_idx, is_dynamic_block):
    """
    Build token position-to-ID mapping vectors for a specific block.

    Creates index vectors used for indirect DMAs to load hidden states and spill expert outputs for the current block.

    Args:
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        routed_token_indices (nl.ndarray): Token indices from nonzero_with_count. Shape depends on context:
            - Static blocks: [pmax, T+1] in SBUF, with count in final element
            - Dynamic blocks: [n_dynamic_blocks, block_size] in HBM
        arange_4H (nl.ndarray): [1, 4], Arange vector for 4_H broadcast.
        block_idx: Block index. Static: int literal. Dynamic: [1, 1] SBUF tensor.
        is_dynamic_block (bool): Whether this is a dynamic block (affects indexing pattern).

    Returns:
        token_position_to_id_4_H_T_sb (nl.ndarray): [blk_tile_T_x4, blk_n_T_x4_tiles], Transposed indices
            with 4_H broadcast for hidden state loading.
        token_position_to_id_T_sb (nl.ndarray): [blk_tile_T, blk_n_T_tiles], Transposed indices
            for expert affinity loading and output spilling.
    """
    # Token position to id with 4_H broadcast (hidden load)
    token_position_to_id_4_H_sb = nl.ndarray((1, dynamism_cfg.block_size, _q_width), dtype=nl.int32, buffer=nl.sbuf)
    token_position_to_id_4_H_f32_sb = nl.ndarray((1, dynamism_cfg.blk_T_x4), dtype=nl.float32, buffer=nl.sbuf)
    token_position_to_id_4_H_T_psum = nl.ndarray(
        (dynamism_cfg.blk_tile_T_x4, dynamism_cfg.blk_n_T_x4_tiles), dtype=nl.float32, buffer=nl.psum
    )
    token_position_to_id_4_H_T_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T_x4, dynamism_cfg.blk_n_T_x4_tiles), dtype=nl.int32, buffer=nl.sbuf
    )

    # Token position to id (expert affinity load + expert MLP out spill)
    token_position_to_id_f32_sb = nl.ndarray((1, dynamism_cfg.block_size), dtype=nl.float32, buffer=nl.sbuf)
    token_position_to_id_T_psum = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.float32, buffer=nl.psum
    )
    token_position_to_id_T_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.int32, buffer=nl.sbuf
    )

    # Dynamic: indirect load indices from HBM [n_dynamic_blocks, block_size] -> SBUF [1, block_size]
    if is_dynamic_block:
        token_position_to_id_sb = nl.ndarray((1, dynamism_cfg.block_size), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            src=routed_token_indices.ap(
                pattern=[[dynamism_cfg.block_size, 1], [1, dynamism_cfg.block_size]],
                offset=0,
                scalar_offset=block_idx,
                indirect_dim=0,
            ),
            dst=token_position_to_id_sb,
        )
        indices_src = token_position_to_id_sb
        indices_pattern = [[dynamism_cfg.block_size, 1], [1, dynamism_cfg.block_size], [0, _q_width]]
        indices_offset = 0
    # Static: directly index indices in SBUF
    else:
        indices_src = routed_token_indices
        indices_pattern = [[dynamism_cfg.T_plus_1, 1], [1, dynamism_cfg.block_size], [0, _q_width]]
        indices_offset = dynamism_cfg.block_size * block_idx

    # Broadcast indices from [1, block_size] -> [1, block_size, 4] to load interleaved T * 4_H dim
    nisa.scalar_tensor_tensor(
        data=indices_src.ap(
            pattern=indices_pattern,  # step=0 broadcasts across the _q_width dim
            offset=indices_offset,
        ),
        op0=nl.multiply,
        operand0=float(_q_width),  # TensorScalar operand must be f32
        op1=nl.add,
        operand1=arange_4H.ap(
            pattern=[
                [_q_width, 1],
                [0, dynamism_cfg.block_size],
                [1, _q_width],
            ],  # step=0 broadcasts across the block_size dim
            offset=0,
        ),
        dst=token_position_to_id_4_H_sb,
    )

    # Flatten to [1, block_size * _q_width]
    token_position_to_id_4_H_sb = token_position_to_id_4_H_sb.reshape((1, dynamism_cfg.blk_T_x4))

    # Cast indices to f32 for PE transpose
    # FIXME: utilize bitcast for better performance when NKI fixes reinterpret casting
    nisa.tensor_copy(token_position_to_id_4_H_f32_sb, token_position_to_id_4_H_sb)
    if is_dynamic_block:
        nisa.tensor_copy(src=token_position_to_id_sb, dst=token_position_to_id_f32_sb)
    else:
        nisa.tensor_copy(
            dst=token_position_to_id_f32_sb,
            src=routed_token_indices[0, nl.ds(dynamism_cfg.block_size * block_idx, dynamism_cfg.block_size)],
        )

    # Transpose indices
    for tile_in in range(dynamism_cfg.blk_n_T_x4_tiles):
        nisa.nc_transpose(
            data=token_position_to_id_4_H_f32_sb[
                0, nl.ds(dynamism_cfg.blk_tile_T_x4 * tile_in, dynamism_cfg.blk_tile_T_x4)
            ],
            dst=token_position_to_id_4_H_T_psum[:, tile_in],
        )
    nisa.tensor_copy(
        src=token_position_to_id_4_H_T_psum[...],
        dst=token_position_to_id_4_H_T_sb[...],
    )
    for tile_out in range(dynamism_cfg.blk_n_T_tiles):
        nisa.nc_transpose(
            data=token_position_to_id_f32_sb[0, nl.ds(dynamism_cfg.blk_tile_T * tile_out, dynamism_cfg.blk_tile_T)],
            dst=token_position_to_id_T_psum[:, tile_out],
        )
    nisa.tensor_copy(
        src=token_position_to_id_T_psum[...],
        dst=token_position_to_id_T_sb[...],
    )

    return token_position_to_id_4_H_T_sb, token_position_to_id_T_sb


@nki.jit
def _layout_adapter_qmx_hbm(
    input: nl.ndarray,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig = None,
    input_indices_T_sb: nl.ndarray = None,
    output_dtype: nki.dtype = nl.float8_e4m3fn_x4,
) -> tuple[nl.ndarray, nl.ndarray]:
    """
    Load input from HBM, transform tensor into swizzled layout, and perform quantization to MXFP8.

    Args:
        input (nl.ndarray): [T, 4_H * H/512 * 16_H * 8_H], Input tensor in HBM.
        dims (AllExpertMXDimensions): Dimension parameters. Uses full-T tiling when input_indices_T_sb is None,
            otherwise uses per-block tiling. Uses dims.t32_tile_offset for T-sharding.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters. Required when input_indices_T_sb is provided.
        input_indices_T_sb (nl.ndarray): [32_T * 4_H, T/32] Optional indices for indirect load from HBM.
        output_dtype (nki.dtype): MXFP8 dtype to quantize to.

    Returns:
        output_quant_sb (nl.ndarray): [16_H * 8_H, H/512, T], Quantized output in SBUF
            (4_H packed in x4 dtype).
        output_scale_sb (nl.ndarray): [16_H * 8_H, H/512, T], Scales in SBUF
            (located in leading 4P of each SBUF quadrant).
    """

    # Validate inputs, extract shapes
    is_blockwise = input_indices_T_sb != None
    n_T32_load_tiles = dynamism_cfg.blk_n_T32_tiles if is_blockwise else dims.n_T32_tiles
    T_load = dynamism_cfg.block_size if is_blockwise else dims.T_local
    T_x4_load = dynamism_cfg.blk_tile_T_x4 if is_blockwise else dims.T32_H4
    kernel_assert(
        output_dtype in SUPPORTED_QMX_OUTPUT_DTYPES,
        f"Got {output_dtype=}, expected output_x4_dtype in {SUPPORTED_QMX_OUTPUT_DTYPES}",
    )

    # Shapes + allocations
    # If using blockwise algorithm, flatten T * 4_H dim for indirect load
    n_T32_tiles_global = div_ceil(dims.T, _NUM_H4_FOLDS_PER_COLUMN)
    input_hbm_shape = (
        (n_T32_tiles_global * dims.T32_H4, dims.n_H512_tiles, dims.tile_H)
        if is_blockwise
        else (n_T32_tiles_global, dims.T32_H4, dims.n_H512_tiles, dims.tile_H)
    )
    input_sb_shape = (T_x4_load, n_T32_load_tiles, dims.n_H512_tiles, dims.tile_H)
    swizzle_shape = (dims.tile_H, dims.n_H512_tiles, n_T32_load_tiles, T_x4_load)
    out_quantized_shape = (dims.tile_H, dims.n_H512_tiles, T_load)
    input_sb = nl.ndarray(input_sb_shape, dtype=input.dtype, buffer=nl.sbuf)
    input_swizzled_sb = nl.ndarray(swizzle_shape, dtype=input_sb.dtype, buffer=nl.sbuf)
    output_quant_sb = nl.ndarray(out_quantized_shape, dtype=output_dtype, buffer=nl.sbuf)
    output_scale_sb = nl.ndarray(out_quantized_shape, dtype=MX_SCALE_DTYPE, buffer=nl.sbuf)

    # Reshape input
    input = input.reshape(input_hbm_shape)

    # Load interleaved T * 4_H, then transpose to achieve swizzled layout
    for t32_tile_idx in nl.affine_range(n_T32_load_tiles):
        if is_blockwise:
            nisa.dma_copy(
                src=input.ap(
                    pattern=[
                        [dims.n_H512_tiles * dims.tile_H, T_x4_load],
                        [1, 1],
                        [dims.tile_H, dims.n_H512_tiles],
                        [1, dims.tile_H],
                    ],
                    offset=0,
                    vector_offset=input_indices_T_sb.ap(
                        pattern=[[n_T32_load_tiles, T_x4_load], [1, 1]],
                        offset=t32_tile_idx,
                    ),
                    indirect_dim=0,
                ),
                dst=input_sb[:, t32_tile_idx, :, :],
                # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
                oob_mode=oob_mode.skip,
            )
        else:
            nisa.dma_copy(
                src=input[t32_tile_idx + dims.t32_tile_offset, :, :, :],
                dst=input_sb[:, t32_tile_idx, :, :],
            )
        for h512_tile_idx in nl.affine_range(dims.n_H512_tiles):
            input_transposed_psum = nl.ndarray((dims.tile_H, T_x4_load), dtype=input_sb.dtype, buffer=nl.psum)
            nisa.nc_transpose(data=input_sb[:, t32_tile_idx, h512_tile_idx, :], dst=input_transposed_psum[...])
            nisa.tensor_copy(src=input_transposed_psum[...], dst=input_swizzled_sb[:, h512_tile_idx, t32_tile_idx, :])

    # Quantize to MXFP8
    nisa.quantize_mx(
        src=input_swizzled_sb,
        dst=output_quant_sb,
        dst_scale=output_scale_sb,
    )

    return output_quant_sb, output_scale_sb


@nki.jit
def _load_block_expert_affinities(input_tensors, dims, dynamism_cfg, token_position_to_id_T_sb, expert_idx):
    """
    Load expert affinities for tokens in a block using indirect DMA.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        token_position_to_id_T_sb (nl.ndarray): [blk_tile_T, blk_n_T_tiles], Token position-to-ID mapping.
        expert_idx (int): Index of the expert to load affinities for.

    Returns:
        expert_affinities_masked_sb (nl.ndarray): [blk_tile_T, blk_n_T_tiles, 1], Expert affinities
            for the block's tokens in SBUF.
    """
    # Expert affinities + index calc
    expert_affinities_masked_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles, 1),
        dtype=input_tensors.expert_affinities_masked.dtype,
        buffer=nl.sbuf,
    )

    # Step 3: Load expert affinities for this block
    for tile_T in range(dynamism_cfg.blk_n_T_tiles):
        nisa.dma_copy(
            src=input_tensors.expert_affinities_masked.ap(
                pattern=[[dims.E_L, dynamism_cfg.blk_tile_T], [1, 1], [1, 1]],
                offset=expert_idx,
                vector_offset=token_position_to_id_T_sb.ap(
                    pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                    offset=tile_T,
                ),
                indirect_dim=0,
            ),
            # Always use 0 for innermost dim because we load 1x expert's affinities at a time
            dst=expert_affinities_masked_sb[:, tile_T, 0],
            # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
            oob_mode=oob_mode.skip,
        )

    return expert_affinities_masked_sb


@nki.jit
def _load_expert(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    expert_idx: int,
) -> ExpertWeightsSBUF:
    """
    Load gate, up, and down projection weight, scale, and bias input_tensors for one expert.

    When LNC=2, the loaded input_tensors are sharded on I dimension (tile-based sharding).
    This ensures gate_up and down projections use the same I-sharding strategy.
    For down_bias, we broadcast to [tile_T, H]. When LNC=2, the first half of H is full
    of bias and second half of H is full of zeros on NC0; NC1 is the inverse.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        expert_idx (int): Expert index to load.

    Returns:
        ExpertWeightsSBUF: Expert weights, scales, and biases in SBUF.
    """

    # Load gate projection with tile-based I sharding
    gate_weight_sb, gate_weight_scale_sb, gate_bias_sb = load_gate_up_weight_scale_bias(
        weight=input_tensors.gate_up_weights,
        scale=input_tensors.gate_up_weights_scale,
        bias=input_tensors.gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=GATE_FUSED_IDX,
        H=dims.H,
        n_I512_tiles_local=dims.n_I512_tiles_local,
        I_local=dims.I_local,
        I_offset=dims.I_offset,
        I_local_padded=dims.I_local_padded,
    )

    # Load up projection with tile-based I sharding
    up_weight_sb, up_weight_scale_sb, up_bias_sb = load_gate_up_weight_scale_bias(
        weight=input_tensors.gate_up_weights,
        scale=input_tensors.gate_up_weights_scale,
        bias=input_tensors.gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=UP_FUSED_IDX,
        H=dims.H,
        n_I512_tiles_local=dims.n_I512_tiles_local,
        I_local=dims.I_local,
        I_offset=dims.I_offset,
        I_local_padded=dims.I_local_padded,
    )

    # Load down projection, broadcast down projection bias
    # Pass pre-computed tile_start to ensure alignment with gate_up projection
    down_weight_sb, down_weight_scale_sb, down_bias_sb = load_broadcast_down_weight_scale_bias(
        weight=input_tensors.down_weights,
        scale=input_tensors.down_weights_scale,
        bias=input_tensors.down_weights_bias,
        expert_idx=expert_idx,
        H=dims.H,
        tile_I=nl.tile_size.pmax,
        n_I512_tiles=dims.n_I512_tiles_local,
        tile_offset=dims.tile_start,
        tile_T=dims.tile_T,
        activation_compute_dtype=kernel_cfg.activation_compute_dtype,
        use_PE_bias_broadcast=False,  # FIXME: PE bias broadcast leads to inaccuracy
        shard_on_T=dims.shard_on_T,
    )

    return ExpertWeightsSBUF(
        gate_weight_sb=gate_weight_sb,
        up_weight_sb=up_weight_sb,
        down_weight_sb=down_weight_sb,
        gate_weight_scale_sb=gate_weight_scale_sb,
        up_weight_scale_sb=up_weight_scale_sb,
        down_weight_scale_sb=down_weight_scale_sb,
        gate_bias_sb=gate_bias_sb,
        up_bias_sb=up_bias_sb,
        down_bias_sb=down_bias_sb,
    )


@nki.jit
def _compute_expert_mlp(
    input_quant: nl.ndarray,
    input_scale: nl.ndarray,
    weights: ExpertWeightsSBUF,
    kernel_cfg: AllExpertMXKernelConfig,
    expert_affinities_masked: nl.ndarray,
    output_sb: nl.ndarray,
    output_hbm: nl.ndarray,
    expert_idx: int,
    is_first_expert: bool,
    is_last_expert: bool,
    shard_on_I: bool = True,
    shard_on_T: bool = False,
    T_offset: int = 0,
    token_position_to_id_T: nl.ndarray = None,
) -> nl.ndarray:
    """
    Compute expert MLP for one block of input.

    Args:
        input_quant (nl.ndarray): Quantized input tensor.
        input_scale (nl.ndarray): Input scale tensor.
        weights (ExpertWeightsSBUF): Expert weights, scales, and biases in SBUF.
        kernel_cfg (AllExpertMXKernelConfig): Kernel config parameters.
        expert_affinities_masked (nl.ndarray): Masked expert affinities.
        output_sb (nl.ndarray): Output tensor in SBUF.
        output_hbm (nl.ndarray): Output tensor in HBM.
        expert_idx (int): Expert index.
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.
        shard_on_I (bool): Whether I dimension is sharded across NCs.
        shard_on_T (bool): Whether T dimension is sharded across NCs.
        T_offset (int): Offset for T dimension in HBM output.
        token_position_to_id_T (nl.ndarray): Token position to ID mapping for blockwise DMA.

    Returns:
        output_sb: Output tensor in SBUF.
    """

    # Step 1: Compute gate/up projection, projection clamping, activation function, and QMX
    act_quant_sb, act_scale_sb = gate_up_projection_mx_shard_I(
        input_quant_sb=input_quant,
        input_scale_sb=input_scale,
        gate_weight_sb=weights.gate_weight_sb,
        up_weight_sb=weights.up_weight_sb,
        gate_weight_scale_sb=weights.gate_weight_scale_sb,
        up_weight_scale_sb=weights.up_weight_scale_sb,
        gate_bias_sb=weights.gate_bias_sb,
        up_bias_sb=weights.up_bias_sb,
        gate_clamp_upper_limit=kernel_cfg.gate_clamp_upper_limit,
        gate_clamp_lower_limit=kernel_cfg.gate_clamp_lower_limit,
        up_clamp_upper_limit=kernel_cfg.up_clamp_upper_limit,
        up_clamp_lower_limit=kernel_cfg.up_clamp_lower_limit,
        hidden_act_fn=kernel_cfg.hidden_act_fn,
        activation_compute_dtype=kernel_cfg.activation_compute_dtype,
    )

    # Step 3: Compute down projection, expert affinity scaling, expert add, LNC reduction, and SB->HBM spill
    down_projection_mx_shard_I(
        act_sb=act_quant_sb[...],
        act_scale_sb=act_scale_sb[...],
        weight_sb=weights.down_weight_sb,
        weight_scale_sb=weights.down_weight_scale_sb,
        bias_sb=weights.down_bias_sb,
        expert_affinities_masked_sb=expert_affinities_masked,
        expert_idx=expert_idx,
        out_sb=output_sb,
        out_hbm=output_hbm,
        expert_affinities_scaling_mode=kernel_cfg.expert_affinities_scaling_mode,
        activation_compute_dtype=kernel_cfg.activation_compute_dtype,
        is_first_expert=is_first_expert,
        is_last_expert=is_last_expert,
        shard_on_I=shard_on_I,
        shard_on_T=shard_on_T,
        T_offset=T_offset,
        token_position_to_id_T=token_position_to_id_T,
    )

    return output_sb


@nki.jit
def _compute_block(
    input_tensors,
    kernel_cfg,
    dims,
    dynamism_cfg,
    weights,
    routed_token_indices,
    arange_4H,
    expert_idx,
    block_idx,
    is_dynamic_block,
):
    """
    Compute expert MLP for a single block of tokens.

    Builds block index mapping, loads and quantizes hidden states, loads expert affinities, and computes the expert MLP.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        weights (ExpertWeightsSBUF): Expert weights, scales, and biases in SBUF.
        routed_token_indices (nl.ndarray): Token indices from nonzero_with_count.
            - Static blocks: [pmax, T+1] in SBUF, with count in final element
            - Dynamic blocks: [n_dynamic_blocks, block_size] in HBM
        arange_4H (nl.ndarray): [1, 4], Arange vector for 4_H broadcast.
        expert_idx (int): Index of the current expert.
        block_idx: Block index. Static: int literal. Dynamic: [1, 1] SBUF tensor.
        is_dynamic_block (bool): Whether this is a dynamic block (affects indexing pattern).
    """
    # Build token_position_to_id vectors for load/spill for this block
    token_position_to_id_4_H_T_sb, token_position_to_id_T_sb = _get_block_token_position_to_id(
        dynamism_cfg=dynamism_cfg,
        routed_token_indices=routed_token_indices,
        arange_4H=arange_4H,
        block_idx=block_idx,
        is_dynamic_block=is_dynamic_block,
    )

    # Load + quantize hidden states for this block
    input_quant_sb, input_scale_sb = _layout_adapter_qmx_hbm(
        input=input_tensors.hidden_input,
        dims=dims,
        dynamism_cfg=dynamism_cfg,
        input_indices_T_sb=token_position_to_id_4_H_T_sb,
    )

    # Load expert affinities for this block
    expert_affinities_masked_sb = _load_block_expert_affinities(
        input_tensors=input_tensors,
        dims=dims,
        dynamism_cfg=dynamism_cfg,
        token_position_to_id_T_sb=token_position_to_id_T_sb,
        expert_idx=expert_idx,
    )

    # Allocate SBUF result buffer for MLP(block)
    output_shape = (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles, dims.H)
    output_sb = nl.ndarray(output_shape, dtype=kernel_cfg.activation_compute_dtype, buffer=nl.sbuf)

    # Compute expert MLP for this block
    _compute_expert_mlp(
        input_quant=input_quant_sb,
        input_scale=input_scale_sb,
        weights=weights,
        kernel_cfg=kernel_cfg,
        expert_affinities_masked=expert_affinities_masked_sb,
        output_sb=output_sb,
        output_hbm=input_tensors.output if (not kernel_cfg.output_in_sbuf) else None,
        expert_idx=expert_idx,
        is_first_expert=(expert_idx == 0),
        is_last_expert=(expert_idx == dims.E_L - 1),
        token_position_to_id_T=token_position_to_id_T_sb,
    )
