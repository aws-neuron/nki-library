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

from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode, oob_mode

from ...moe_block.moe_block_tkg_utils import _SBUF_USABLE_PER_PARTITION, _dtype_size
from ...quantization.fp8_quantize import pre_combine_dequant_scales, row_quantization, static_quantization
from ...utils.common_types import MoEAllToAllVStrategy, MoELNCShardingStrategy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .all_expert_mx_utils import (
    BF16_PER_FP32,
    BF16_PER_INT32,
    FP8_PER_BF16,
    FP8_PER_FP8X4,
    FP8_PER_FP32,
    FP8_PER_INT32,
    FP8X4_TP_VIEW_DTYPE,
    NONZERO_WITH_COUNT_PAD_VAL,
    NUM_H4_FOLDS_PER_COLUMN,
    UINT8_TP_VIEW_DTYPE,
    AllExpertMXDimensions,
    AllExpertMXDynamismConfig,
    AllExpertMXInputTensors,
    AllExpertMXKernelConfig,
    ExpertWeightsSBUF,
    alloc_dummy_scale_tile,
    init_all_expert_mx_configs,
    validate_all_expert_mx_inputs,
)
from .down_projection_mx import (
    down_projection_mx,
    load_broadcast_down_weight_scale_bias,
)
from .gate_up_projection_mx import (
    gate_up_projection_mx,
    load_gate_up_weight_scale_bias,
)
from .mlp_parameters import MLPParameters
from .projection_mx_constants import (
    GATE_FUSED_IDX,
    MX_SCALE_DTYPE,
    MXFP8_UNPACKED_PACKED_MAP,
    SUPPORTED_QMX_OUTPUT_DTYPES,
    UP_FUSED_IDX,
    _q_width,
)


@nki.jit
def _all_expert_moe_tkg_mx(
    mlp_params: MLPParameters,
    output: nl.NkiTensor,
    output_t_offset: int = 0,
) -> nl.NkiTensor:
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
        output (nl.NkiTensor): [min(T, 128), ⌈T/128⌉, H] in SBUF or [T, H] in HBM output tensor.

    Returns:
        output (nl.NkiTensor): [T, H] in HBM or [min(T, 128), ⌈T/128⌉, H] in SBUF, Output tensor with MoE results.

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
    )
    validate_all_expert_mx_inputs(input_tensors, kernel_cfg, dims, dynamism_cfg)

    # Dispatch to expert MLP implementation
    if dynamism_cfg.is_all_expert_dynamic:
        kernel_assert(not kernel_cfg.is_static_quant, "STATIC_MX is not supported with dynamic all-expert mode")
        kernel_assert(not kernel_cfg.is_row_quant, "ROW_MX is not supported with dynamic all-expert mode")
        """
        The DLoC currently have the restriction that both PNC in LNC=2 must
        execute the same control flow logic.

        When shard on I, we would need to pay the DLoC overhead for each expert, while
        shard on E, we would need to pay the DLoC overhead E_L // 2, half that of shard-on-I.
        The trade off is that each core would load twice the amount of weight than shard-on-I.

        The DLoC loop's overhead is around 15us at the moment, making shard on E more efficient.

        SHARD_E is a cross-core (LNC=2) optimization: it splits experts across
        the two PNCs. At LNC=1 (n_prgs==1) there is no second core to split
        across, and the SHARD_E path emits cross-core sendrecv/core_barrier that
        are invalid on a single core. So gate SHARD_E on n_prgs>1 and route the
        single-core case through SHARD_I (the LNC=1-validated dynamic path).
        """
        _, n_prgs, _ = get_verified_program_sharding_info()

        if n_prgs > 1 and (dims.E_L % 2 == 0) and dims.T >= 128:
            # Force reinitialize with SHARD_E
            input_tensors, kernel_cfg, dims, dynamism_cfg = init_all_expert_mx_configs(
                mlp_params=mlp_params, output=output, sharding_strategy=MoELNCShardingStrategy.SHARD_E
            )
            _all_expert_mx_dynamic_shard_on_E(
                input_tensors=input_tensors,
                kernel_cfg=kernel_cfg,
                dims=dims,
                dynamism_cfg=dynamism_cfg,
            )
        else:
            _all_expert_mx_dynamic_shard_on_I(
                input_tensors=input_tensors,
                kernel_cfg=kernel_cfg,
                dims=dims,
                dynamism_cfg=dynamism_cfg,
            )
    elif kernel_cfg.is_static_quant:
        # STATIC_MX keeps bf16 in SBUF, per-expert quantization in the loop (no pre-quantization).
        # TODO: For E_L=1, optimize by using pre-quantization.

        # If T is too large for SHARD_I, force SHARD_T.
        needs_shard_on_T = _static_mx_needs_shard_t(
            dims.T,
            dims.H,
            dims.E_L,
            dims.I,
            input_tensors.gate_up_weights.dtype,
            input_tensors.hidden_input.dtype,
        )
        if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_I and needs_shard_on_T:
            input_tensors, kernel_cfg, dims, _ = init_all_expert_mx_configs(
                mlp_params=mlp_params, output=output, sharding_strategy=MoELNCShardingStrategy.SHARD_T
            )
        _all_expert_static_mx(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            output_t_offset=output_t_offset,
        )
    else:
        _all_expert_mx_and_row_mx(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            output_t_offset=output_t_offset,
        )

    return output


def _all_expert_mx_and_row_mx(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    output_t_offset: int = 0,
) -> nl.NkiTensor:
    """
    Static all-expert MoE computation without dynamic loop on chip (DLoC).

    Processes all experts sequentially, computing MLP(all tokens) for each expert
    before moving to the next. This is preferred for small batch sizes where
    dynamic control flow overhead (nonzero_with_count, indirect DMAs, and dynamic
    branch checking) would exceed the potential savings from skipping unrouted blocks.
    For reference, with GPT-OSS and EP-only sharding, the crossover is around
    T=128-256, but the optimal threshold is model-dependent and should be tuned
    for peak performance.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.

    Returns:
        nl.NkiTensor: Output tensor with MoE computation results.
    """

    # Step 1: Process inputs
    # Step 1.1: Optional load + swizzle + QMX hidden states
    if kernel_cfg.input_in_sbuf:
        # Input is already swizzled + MX quantized upstream (SBUF or pre-quantized HBM)
        if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
            T_load = dims.T_local
            if input_tensors.hidden_input.buffer != nl.sbuf:
                # Pre-quantized HBM: DMA copy T-shard to SBUF
                n_h512 = input_tensors.hidden_input.shape[1]
                input_quant_sb = nl.ndarray(
                    (dims.pmax, n_h512, T_load), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
                )
                input_scale_sb = nl.ndarray((dims.pmax, n_h512, T_load), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.dma_copy(dst=input_quant_sb, src=input_tensors.hidden_input[:, :, nl.ds(dims.T_offset, T_load)])
                nisa.dma_copy(
                    dst=input_scale_sb, src=input_tensors.hidden_input_scale[:, :, nl.ds(dims.T_offset, T_load)]
                )
            else:
                input_quant_sb = input_tensors.hidden_input[:, :, nl.ds(dims.T_offset, T_load)]
                input_scale_sb = input_tensors.hidden_input_scale[:, :, nl.ds(dims.T_offset, T_load)]
        else:
            if input_tensors.hidden_input.buffer != nl.sbuf:
                # Pre-quantized HBM: DMA copy T_physical to padded SBUF buffer
                n_h512 = input_tensors.hidden_input.shape[1]
                input_quant_sb = nl.ndarray(
                    (dims.pmax, n_h512, dims.T), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
                )
                input_scale_sb = nl.ndarray(
                    (dims.pmax, n_h512, dims.T),
                    dtype=input_tensors.hidden_input_scale.dtype,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(dst=input_quant_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input)
                nisa.dma_copy(dst=input_scale_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input_scale)
            else:
                if dims.T_physical < dims.T:
                    n_h512 = input_tensors.hidden_input.shape[1]
                    input_quant_sb = nl.ndarray(
                        (dims.pmax, n_h512, dims.T), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
                    )
                    input_scale_sb = nl.ndarray(
                        (dims.pmax, n_h512, dims.T), dtype=input_tensors.hidden_input_scale.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=input_quant_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input)
                    nisa.tensor_copy(dst=input_scale_sb[:, :, : dims.T_physical], src=input_tensors.hidden_input_scale)
                else:
                    input_quant_sb = input_tensors.hidden_input
                    input_scale_sb = input_tensors.hidden_input_scale

            # Zero-pad trailing T rows if T was padded
            if dims.T_physical < dims.T:
                nisa.memset(input_quant_sb[:, :, dims.T_physical :], 0)
                nisa.memset(input_scale_sb[:, :, dims.T_physical :], 0)
    else:
        input_quant_sb, input_scale_sb = _layout_adapter_qmx_hbm(
            input=input_tensors.hidden_input,
            dims=dims,
        )

    # ROW_MX: hoist a single 2D dummy scale tile [128, free_dim] outside the expert loop
    # instead of allocating per-expert 3D scale buffers [P, n_tiles, F]
    scale_sb = None
    if kernel_cfg.is_row_quant:
        scale_sb = alloc_dummy_scale_tile(free_dim=nl.tile_size.psum_fmax * 2)

    # Step 1.2: View expert_affinities_masked and output_hbm based on sharding decision
    # When output_t_offset > 0 (tiling), pass the full output tensor and let down_projection_mx
    # handle the absolute T offset. When output_t_offset == 0, use the original sliced view.
    hbm_t_offset_for_down_proj = output_t_offset
    if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
        T_eff = dims.T_local
        T_hbm_offset = dims.T_offset
        if output_t_offset > 0:
            output_hbm_view = input_tensors.output
            hbm_t_offset_for_down_proj = output_t_offset + dims.T_offset
        else:
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
        output_hbm_view = input_tensors.output
        if dims.T_physical < dims.T:
            affinities_padded = nl.ndarray(
                (dims.T, input_tensors.expert_affinities_masked.shape[1]),
                dtype=input_tensors.expert_affinities_masked.dtype,
                buffer=nl.sbuf,
            )
            nisa.memset(affinities_padded, 0)
            nisa.dma_copy(dst=affinities_padded[: dims.T_physical, :], src=input_tensors.expert_affinities_masked)
            expert_affinities_masked_sb = affinities_padded
        else:
            expert_affinities_masked_sb = input_tensors.expert_affinities_masked

    # Step 2: Allocate output
    output_shape = (dims.tile_T, dims.n_tiles_in_T, dims.H)
    output_sb = nl.ndarray(output_shape, dtype=kernel_cfg.activation_compute_dtype, buffer=nl.sbuf)

    # Step 3: Compute expert MLPs sequentially
    for expert_idx in nl.sequential_range(dims.E_L):
        # Step 3.1: Load weights for this expert
        weights = _load_expert(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            expert_idx=expert_idx,
        )

        # ROW_MX: override per-expert 3D scale tensors with the shared 2D dummy tile [128, F].
        # matmul callers index as [:, :F] instead of [:, tile_h, slice].
        if kernel_cfg.is_row_quant:
            weights.gate_weight_scale_sb = scale_sb
            weights.up_weight_scale_sb = scale_sb
            weights.down_weight_scale_sb = scale_sb
            weights.dummy_scale_tile_sb = scale_sb

        # Step 3.2: Compute MLP for this expert
        _compute_expert_mlp(
            input_quant=input_quant_sb,
            input_scale=input_scale_sb,
            weights=weights,
            kernel_cfg=kernel_cfg,
            expert_affinities_masked=expert_affinities_masked_sb,
            output_sb=output_sb[...],
            output_hbm=output_hbm_view if (not kernel_cfg.output_in_sbuf) else None,
            expert_idx=expert_idx,
            is_first_expert=(expert_idx == 0),
            is_last_expert=(expert_idx == dims.E_L - 1),
            sharding_strategy=dims.sharding_strategy,
            input_dequant_scale_sb=input_tensors.input_dequant_scale,
            output_t_offset=hbm_t_offset_for_down_proj,
            is_software_quant=kernel_cfg.is_row_quant,
            T_physical=dims.T_physical if dims.T_physical < dims.T else None,
        )

    return input_tensors.output


def _all_expert_static_mx(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    output_t_offset: int = 0,
) -> nl.ndarray:
    """
    Static all-expert MoE computation with per-expert STATIC_MX quantization.

    Unlike _all_expert_mx_static which receives pre-quantized fp8_x4 input,
    this function receives bf16 input and performs swizzle + static_quantization
    per expert inside the loop using that expert's gate_up_in_scale.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
            hidden_input is bf16 [pmax, T, H_free] in SBUF (not pre-quantized).
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        output_t_offset (int): T offset for output writes (used in tiling).

    Returns:
        nl.ndarray: Output tensor with MoE computation results.
    """
    # Compute dimensions for swizzle
    pmax = dims.pmax
    H_free = dims.H // pmax
    n_H512_tiles = dims.H // (pmax * _q_width)

    # Step 1: Load bf16 input into SBUF [pmax, T_load, H_free]
    # Input arrives as bf16 [pmax, T, H_free] in SBUF or [T, H] in HBM
    if input_tensors.hidden_input.buffer == nl.sbuf:
        if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
            T_load = dims.T_local
            input_bf16_sb = input_tensors.hidden_input[:, nl.ds(dims.T_offset, T_load), :]
        else:
            T_load = dims.T
            if dims.T_physical < dims.T:
                input_bf16_sb = nl.ndarray(
                    (pmax, dims.T, H_free), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=input_bf16_sb[:, : dims.T_physical, :], src=input_tensors.hidden_input)
                nisa.memset(dst=input_bf16_sb[:, dims.T_physical :, :], value=0)
            else:
                input_bf16_sb = input_tensors.hidden_input
    else:
        # HBM input [T, H]: DMA copy to SBUF as [pmax, T_load, H_free]
        if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
            T_load = dims.T_local
            input_bf16_sb = nl.ndarray((pmax, T_load, H_free), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf)
            # dst[p, t, f] = hidden[t, f*pmax + p]
            nisa.dma_copy(
                dst=input_bf16_sb,
                src=input_tensors.hidden_input.ap(
                    pattern=[[1, pmax], [dims.H, T_load], [pmax, H_free]],
                    offset=dims.T_offset * dims.H,  # start at the T-shard
                ),
            )
        else:
            T_load = dims.T
            input_bf16_sb = nl.ndarray((pmax, dims.T, H_free), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf)
            # dst[p, t, f] = hidden[t, f*pmax + p]
            nisa.dma_copy(
                dst=input_bf16_sb[:, : dims.T_physical, :],
                src=input_tensors.hidden_input.ap(
                    pattern=[[1, pmax], [dims.H, dims.T_physical], [pmax, H_free]],
                    offset=0,
                ),
            )
            if dims.T_physical < dims.T:
                nisa.memset(dst=input_bf16_sb[:, dims.T_physical :, :], value=0)

    # Hoist a single 2D dummy scale tile [128, free_dim] outside the expert loop
    scale_sb = alloc_dummy_scale_tile(free_dim=nl.tile_size.psum_fmax * 2)

    # Step 1.2: View expert_affinities_masked and output_hbm based on sharding decision
    hbm_t_offset_for_down_proj = output_t_offset
    if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
        T_eff = dims.T_local
        T_hbm_offset = dims.T_offset
        if output_t_offset > 0:
            output_hbm_view = input_tensors.output
            hbm_t_offset_for_down_proj = output_t_offset + dims.T_offset
        else:
            output_hbm_view = input_tensors.output[nl.ds(dims.T_offset, dims.T_local), :]
        if len(input_tensors.expert_affinities_masked.shape) == 3:
            tile_start = T_hbm_offset // dims.pmax
            n_local_tiles = div_ceil(T_eff, dims.pmax)
            expert_affinities_masked_sb = input_tensors.expert_affinities_masked[:, nl.ds(tile_start, n_local_tiles), :]
        else:
            expert_affinities_masked_sb = input_tensors.expert_affinities_masked[nl.ds(T_hbm_offset, T_eff), :]
    else:
        output_hbm_view = input_tensors.output
        if dims.T_physical < dims.T:
            affinities_padded = nl.ndarray(
                (dims.T, input_tensors.expert_affinities_masked.shape[1]),
                dtype=input_tensors.expert_affinities_masked.dtype,
                buffer=nl.sbuf,
            )
            nisa.memset(affinities_padded, 0)
            nisa.dma_copy(dst=affinities_padded[: dims.T_physical, :], src=input_tensors.expert_affinities_masked)
            expert_affinities_masked_sb = affinities_padded
        else:
            expert_affinities_masked_sb = input_tensors.expert_affinities_masked

    # Step 2: Allocate output
    output_shape = (dims.tile_T, dims.n_tiles_in_T, dims.H)
    output_sb = nl.ndarray(output_shape, dtype=kernel_cfg.activation_compute_dtype, buffer=nl.sbuf)

    # Step 3: Compute expert MLPs sequentially
    for expert_idx in nl.sequential_range(dims.E_L):
        # Step 3.1: Load weights for this expert
        weights = _load_expert(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            expert_idx=expert_idx,
        )

        # Override per-expert 3D scale tensors with the shared 2D dummy tile [128, F]
        weights.gate_weight_scale_sb = scale_sb
        weights.up_weight_scale_sb = scale_sb
        weights.down_weight_scale_sb = scale_sb
        weights.dummy_scale_tile_sb = scale_sb

        # Step 3.2: Swizzle bf16 input from [pmax, T_load, H_free] → [pmax, n_H512, T_load, q_width]
        swizzled_sb = nl.ndarray((pmax, n_H512_tiles, T_load, _q_width), dtype=input_bf16_sb.dtype, buffer=nl.sbuf)
        for h512_tile_idx in nl.affine_range(n_H512_tiles):
            for q_idx in nl.affine_range(_q_width):
                nisa.tensor_copy(
                    dst=swizzled_sb[:, h512_tile_idx, :, q_idx],
                    src=input_bf16_sb[:, :, q_idx * n_H512_tiles + h512_tile_idx],
                )

        # Step 3.3: Load per-expert gate_up_in_scale and quantize
        gate_up_in_scale_sb = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        gate_up_in_view = (
            input_tensors.gate_up_in_scale.select(dim=0, index=expert_idx)
            .broadcast(dim=0, size=pmax)
            .reshape_dim(dim=0, shape=(pmax, 1))
        )
        nisa.dma_copy(dst=gate_up_in_scale_sb, src=gate_up_in_view)

        # Flatten swizzled bf16 and apply static_quantization (modifies in-place)
        total_free = n_H512_tiles * T_load * _q_width
        swizzled_flat = swizzled_sb.reshape((pmax, total_free))
        quantized_flat, input_dequant_scale = static_quantization(swizzled_flat, gate_up_in_scale_sb)

        # Cast bf16 → fp8, then reinterpret as fp8_x4
        quantized_fp8 = nl.ndarray(quantized_flat.shape, dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        nisa.tensor_copy(dst=quantized_fp8, src=quantized_flat)
        total_x4 = n_H512_tiles * T_load
        input_quant_sb = nl.ndarray((pmax, total_x4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=input_quant_sb.view(nl.uint32), src=quantized_fp8.view(nl.uint32), engine=nisa.vector_engine
        )
        input_quant_sb = input_quant_sb.reshape((pmax, n_H512_tiles, T_load))

        # Dummy 127 MX scales
        input_scale_sb = nl.ndarray((pmax, n_H512_tiles, T_load), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=input_scale_sb, value=127)

        # Step 3.4: Compute combined dequant scales (input_dequant * weight_dequant)
        gate_w_dequant_sb = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        up_w_dequant_sb = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        down_w_dequant_sb = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        gate_w_view = input_tensors.gate_up_weights_scale.select(dim=0, index=expert_idx)
        nisa.dma_copy(dst=gate_w_dequant_sb, src=gate_w_view.slice(dim=0, start=0, end=1).broadcast(dim=0, size=pmax))
        nisa.dma_copy(dst=up_w_dequant_sb, src=gate_w_view.slice(dim=0, start=1, end=2).broadcast(dim=0, size=pmax))
        down_w_view = (
            input_tensors.down_weights_scale.select(dim=0, index=expert_idx)
            .broadcast(dim=0, size=pmax)
            .reshape_dim(dim=0, shape=(pmax, 1))
        )
        nisa.dma_copy(dst=down_w_dequant_sb, src=down_w_view)

        weights.gate_dequant_scale_sb = pre_combine_dequant_scales(input_dequant_scale, gate_w_dequant_sb)
        weights.up_dequant_scale_sb = pre_combine_dequant_scales(input_dequant_scale, up_w_dequant_sb)

        # down: down_in_scale[expert_idx] * down_w_dequant
        down_in_scale_sb = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        down_in_view = (
            input_tensors.down_in_scale.select(dim=0, index=expert_idx)
            .broadcast(dim=0, size=pmax)
            .reshape_dim(dim=0, shape=(pmax, 1))
        )
        nisa.dma_copy(dst=down_in_scale_sb, src=down_in_view)
        weights.down_dequant_scale_sb = pre_combine_dequant_scales(down_in_scale_sb, down_w_dequant_sb)

        # Step 3.5: Compute MLP for this expert
        _compute_expert_mlp(
            input_quant=input_quant_sb,
            input_scale=input_scale_sb,
            weights=weights,
            kernel_cfg=kernel_cfg,
            expert_affinities_masked=expert_affinities_masked_sb,
            output_sb=output_sb[...],
            output_hbm=output_hbm_view if (not kernel_cfg.output_in_sbuf) else None,
            expert_idx=expert_idx,
            is_first_expert=(expert_idx == 0),
            is_last_expert=(expert_idx == dims.E_L - 1),
            sharding_strategy=dims.sharding_strategy,
            input_dequant_scale_sb=input_dequant_scale,
            output_t_offset=hbm_t_offset_for_down_proj,
            is_software_quant=True,
            T_physical=dims.T_physical if dims.T_physical < dims.T else None,
            down_in_scale_sb=down_in_scale_sb,
        )

    return input_tensors.output


def _all_expert_mx_dynamic_shard_on_I(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig,
) -> nl.NkiTensor:
    """
    All-expert MoE computation with dynamic control flow (DLoC), shard on I dimension.

    Please see the comment at the dispatch logic between shard on I and E for the trade
    offs of different sharding schemes.

    Processes all experts sequentially, with tokens split into blocks for each expert.
    When a block contains routed tokens, we compute MLP(block tokens).
    A portion of the blocks for each expert are dynamically skipped at runtime if none
    of the tokens in a dynamic block are routed to the expert.
    Dynamism can provide performance improvements relative to _all_expert_mx_static
    when T is large. For reference, with GPT-OSS and EP-only sharding, benefits start
    around T=128-256, but the optimal threshold is model-dependent and should be tuned
    for peak performance.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.

    Returns:
        nl.NkiTensor: Output tensor with MoE computation results.
    """

    # Step 1: Prepare buffers shared across all experts
    # Step 1.1: Memset output to 0, using u32 memset for 2x perf
    # FIXME[perf]: When we have K=1 or E_L=1, we can skip K reduction and reduce I on-chip.
    _, n_prgs, prg_id = get_verified_program_sharding_info()
    zero_sb = nl.ndarray((dims.tile_T, dims.H // BF16_PER_INT32), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0, engine=nisa.vector_engine)
    # On the profile, the dma_copy is being moved around, causing ineffiency,
    # might need to remove this when running E2E.
    with nl.no_reorder():
        for t_tile in nl.sequential_range(dims.n_tiles_in_T):
            tile_T_actual = min(dims.tile_T, dims.T_local - dims.tile_T * t_tile)
            nisa.dma_copy(
                src=zero_sb[:tile_T_actual, :],
                dst=input_tensors.output.ap(
                    [[dims.H // 2, tile_T_actual], [1, dims.H // 2]],
                    offset=t_tile * dims.tile_T * (dims.H // 2),
                    dtype=nl.uint32,
                ),
                dge_mode=dge_mode.none,
            )

    # Step 1.2: Arange [0, 1, 2, 3] for token indices broadcast, when input is not prequantized
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
        arange_4H = nl.ndarray((1, _q_width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(arange_4H, [[1, _q_width]], offset=0)
    else:
        arange_4H = None

    # Step 1.3: When using A2A-v, copy token indices from input buffer to output buffer
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS:
        # When packing output rows, indices are copied with indirection, during index mapping
        output_indices_hbm = _build_output_indices(input_tensors, dims, dynamism_cfg)
    elif dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PRESERVE_ROW_ORDER:
        # When preserving row order, direct copy with sharding on T
        T_local = dims.T // dims.n_prgs
        T_offset = dims.prg_id * T_local
        num_input_fp8_cols = input_tensors.hidden_input.shape[1]
        input_int32_view = (
            input_tensors.hidden_input.slice(dim=0, start=T_offset, end=T_offset + T_local)
            .slice(dim=1, start=num_input_fp8_cols - FP8_PER_INT32, end=num_input_fp8_cols)
            .view(nl.bfloat16)
        )
        output_int32_view = input_tensors.output.slice(dim=0, start=T_offset, end=T_offset + T_local).slice(
            dim=1, start=dims.H, end=dims.H + BF16_PER_INT32
        )
        nisa.dma_copy(
            src=input_int32_view,
            dst=output_int32_view,
            dge_mode=dge_mode.none,
        )
        output_indices_hbm = None
    else:
        output_indices_hbm = None

    # Process experts in groups of 4 to vectorize the _find_expert_routed_tokens
    for expert_grp in range(div_ceil(dims.E_L, 4)):
        # Step 2.2: (Vectorized) Find indices of tokens routed to current expert group
        # partition 0, 32, 64, 96 contains the results for experts 0-3 respectively.
        routed_token_indices_with_count_sb, dynamic_decision_sb = _find_expert_routed_tokens(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            dynamism_cfg=dynamism_cfg,
            expert_4_tile_idx=expert_grp,
        )

        # Step 2: Compute expert MLPs sequentially
        for expert_idx_local in nl.sequential_range(min(4, dims.E_L - 4 * expert_grp)):
            expert_idx = 4 * expert_grp + expert_idx_local
            # Step 2.1: Load weights for current expert
            weights = _load_expert(input_tensors=input_tensors, kernel_cfg=kernel_cfg, dims=dims, expert_idx=expert_idx)
            local_routed_token_indices_with_count_sb = nl.ndarray(
                (dims.pmax, dynamism_cfg.T_plus_1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                local_routed_token_indices_with_count_sb[0, :],
                routed_token_indices_with_count_sb[expert_idx_local * 32, :],
            )

            # Step 2.3: Compute static blocks
            for static_block_idx in nl.sequential_range(dynamism_cfg.n_static_blocks):
                _compute_block(
                    input_tensors=input_tensors,
                    kernel_cfg=kernel_cfg,
                    dims=dims,
                    dynamism_cfg=dynamism_cfg,
                    weights=weights,
                    output_local=input_tensors.output,
                    routed_token_indices=local_routed_token_indices_with_count_sb,
                    arange_4H=arange_4H,
                    expert_idx=expert_idx,
                    block_idx=static_block_idx,
                    is_dynamic_block=False,
                    output_indices_hbm=output_indices_hbm,
                )

            # Step 2.4: Compute dynamic blocks.
            # Step 2.4.1: Move index vector to HBM, so that each dynamic block's indices can be indirect reloaded
            n_static_tokens = dynamism_cfg.n_static_blocks * dynamism_cfg.block_size
            n_dynamic_tokens = dynamism_cfg.n_dynamic_blocks * dynamism_cfg.block_size
            dynamic_block_token_indices_hbm = nl.ndarray(
                (1, n_dynamic_tokens),
                dtype=routed_token_indices_with_count_sb.dtype,
                buffer=nl.private_hbm,
                name=f'dynamic_block_token_indices_hbm_e{expert_idx}',
            )
            nisa.dma_copy(
                src=local_routed_token_indices_with_count_sb[0, nl.ds(n_static_tokens, n_dynamic_tokens)],
                dst=dynamic_block_token_indices_hbm[0, :],
                dge_mode=dge_mode.none,
            )
            dynamic_block_token_indices_hbm = dynamic_block_token_indices_hbm.reshape(
                (dynamism_cfg.n_dynamic_blocks, dynamism_cfg.block_size)
            )

            # Step 2.4.2: Initialize dynamic register + loop iteration counter
            compute_next_dynamic_block = nisa.register_alloc()
            nisa.register_load(src=dynamic_decision_sb[expert_idx_local * 32, 0], dst=compute_next_dynamic_block)
            dynamic_block_idx = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.memset(dynamic_block_idx, 0)

            # Step 2.4.3: Dynamic loop over dynamic blocks
            def _dynamic_block_body(compute_next_dynamic_block):
                _compute_block(
                    input_tensors=input_tensors,
                    kernel_cfg=kernel_cfg,
                    dims=dims,
                    dynamism_cfg=dynamism_cfg,
                    weights=weights,
                    output_local=input_tensors.output,
                    routed_token_indices=dynamic_block_token_indices_hbm,
                    arange_4H=arange_4H,
                    expert_idx=expert_idx,
                    block_idx=dynamic_block_idx,
                    is_dynamic_block=True,
                    output_indices_hbm=output_indices_hbm,
                )

                # Step 2.4.4: Update dynamic register
                nisa.tensor_scalar(data=dynamic_block_idx, op0=nl.add, operand0=1, dst=dynamic_block_idx)
                next_decision_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    src=dynamic_decision_sb.ap(
                        pattern=[[dynamism_cfg.n_dynamic_blocks_plus_1, 1], [1, 1]],
                        offset=expert_idx_local * 32 * dynamism_cfg.n_dynamic_blocks_plus_1,
                        scalar_offset=dynamic_block_idx,
                        indirect_dim=1,
                    ),
                    dst=next_decision_sb,
                )
                nisa.register_load(src=next_decision_sb, dst=compute_next_dynamic_block)
                return compute_next_dynamic_block

            nl.while_loop(compute_next_dynamic_block, _dynamic_block_body)

    return input_tensors.output


def _all_expert_mx_dynamic_shard_on_E(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig,
) -> nl.NkiTensor:
    """
    All-expert MoE computation with dynamic control flow (DLoC), shard on expert
    to reduce the number of DLoC loops are invoked to E_L // 2.

    Please see the comment at the dispatch logic between shard on I and E for the trade
    offs of different sharding schemes.

    Processes all experts sequentially, with tokens split into blocks for each expert.
    When a block contains routed tokens, we compute MLP(block tokens).
    A portion of the blocks for each expert are dynamically skipped at runtime if none
    of the tokens in a dynamic block are routed to the expert.
    Dynamism can provide performance improvements relative to _all_expert_mx_static
    when T is large. For reference, with GPT-OSS and EP-only sharding, benefits start
    around T=128-256, but the optimal threshold is model-dependent and should be tuned
    for peak performance.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.

    Returns:
        nl.NkiTensor: Output tensor with MoE computation results.
    """
    kernel_assert(dims.E_L % 2 == 0, "DLoC shard-on-E requires even number of experts")
    kernel_assert(
        dims.sharding_strategy == MoELNCShardingStrategy.SHARD_E,
        "LOGIC FAULT: sharding config is not set to shard_E while shard_on_E is invoked",
    )

    # Step 1: Prepare buffers shared across all experts
    # Step 1.1: Memset output to 0, using u32 memset for 2x perf
    _, n_prgs, prg_id = get_verified_program_sharding_info()
    # FIXME[perf]: the following buffer zeroing is a conservative, unsharded zeroing for correctness,
    # we might want to shard this to 2 LNC cores if necessary.
    zero_sb = nl.ndarray((dims.tile_T, dims.H // BF16_PER_INT32), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0, engine=nisa.vector_engine)
    with nl.no_reorder():
        for t_tile in nl.sequential_range(dims.n_tiles_in_T):
            tile_T_actual = min(dims.tile_T, dims.T_local - dims.tile_T * t_tile)
            nisa.dma_copy(
                src=zero_sb[:tile_T_actual, :],
                dst=input_tensors.output.ap(
                    [[dims.H // 2, tile_T_actual], [1, dims.H // 2]],
                    offset=t_tile * dims.tile_T * (dims.H // 2),
                    dtype=nl.uint32,
                ),
                dge_mode=dge_mode.none,
            )
    # LNC=1 (n_prgs==1): a single core zeroed the whole output; no cross-core
    # ordering needed. core_barrier requires LNC degree >= 2.
    if n_prgs > 1:
        nisa.core_barrier(input_tensors.output, cores=[0, 1])

    # Step 1.2: Arange [0, 1, 2, 3] for token indices broadcast, when input is not prequantized
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
        arange_4H = nl.ndarray((1, _q_width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(arange_4H, [[1, _q_width]], offset=0)
    else:
        arange_4H = None

    # Step 1.3: When using A2A-v, copy token indices from input buffer to output buffer
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS:
        # When packing output rows, indices are copied with indirection, during index mapping
        output_indices_hbm = _build_output_indices(input_tensors, dims, dynamism_cfg)
    elif dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PRESERVE_ROW_ORDER:
        # When preserving row order, direct copy with sharding on T
        T_local = dims.T // dims.n_prgs
        T_offset = dims.prg_id * T_local
        num_input_fp8_cols = input_tensors.hidden_input.shape[1]
        input_int32_view = (
            input_tensors.hidden_input.slice(dim=0, start=T_offset, end=T_offset + T_local)
            .slice(dim=1, start=num_input_fp8_cols - FP8_PER_INT32, end=num_input_fp8_cols)
            .view(nl.bfloat16)
        )
        output_int32_view = input_tensors.output.slice(dim=0, start=T_offset, end=T_offset + T_local).slice(
            dim=1, start=dims.H, end=dims.H + BF16_PER_INT32
        )
        nisa.dma_copy(
            src=input_int32_view,
            dst=output_int32_view,
            dge_mode=dge_mode.none,
        )
        output_indices_hbm = None
    else:
        output_indices_hbm = None

    kernel_assert(dims.E_L % n_prgs == 0, "Shard on E DLoC MoE must have E divisible by core count")
    # Process experts in groups of 4 to vectorize the _find_expert_routed_tokens
    for expert_grp in range(div_ceil(dims.E_L, 4)):
        # Step 2.2: (Vectorized) Find indices of tokens routed to current expert group
        # partition 0, 32, 64, 96 contains the results for experts 0-3 respectively.
        routed_token_indices_with_count_sb, dynamic_decision_sb = _find_expert_routed_tokens(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            dynamism_cfg=dynamism_cfg,
            expert_4_tile_idx=expert_grp,
        )

        # Compute the number of dynamic iterations
        dynamic_iteration_count_sb = nl.ndarray((dims.pmax, 1), dtype=dynamic_decision_sb.dtype, buffer=nl.sbuf)
        nisa.memset(dynamic_iteration_count_sb, 0)
        nisa.tensor_reduce(dynamic_iteration_count_sb, nl.add, dynamic_decision_sb, axis=1)
        # Find the maximum number of the iterations.
        # 1. [TODO]perf: this could be done in one instruction for both 2 expert group. We need to reorder the
        # expert to be 0, 2, 1, 3 in the _find_expert_routed_tokens. This is probably not an perf bottleneck.
        # using 2 instructions for now.
        # 2. In addition: tensor_tensor requires the start partiton of operand to align. It is not possible to vectorize the following
        # unless we do split the dynamic_decision_sb into 2 tensors.
        # 3. tensor_scalar does not support Int32
        nisa.tensor_partition_reduce(dynamic_iteration_count_sb[0, 0], nl.max, dynamic_iteration_count_sb[0:33, 0])
        nisa.tensor_partition_reduce(dynamic_iteration_count_sb[64, 0], nl.max, dynamic_iteration_count_sb[0:97, 0])

        # Step 2: Compute expert MLPs sequentially.
        # We are sharding on E, in LNC=2, PNC core would take consecutive experts
        current_expert_grp_LNC2_total_count = min(4, dims.E_L - 4 * expert_grp)
        for expert_lnc_grp in nl.sequential_range(current_expert_grp_LNC2_total_count // 2):
            expert_idx = expert_lnc_grp * 2 + prg_id
            global_expert_idx = expert_grp * 4 + expert_idx

            # Step 2.1: Load weights for current expert
            weights = _load_expert(
                input_tensors=input_tensors, kernel_cfg=kernel_cfg, dims=dims, expert_idx=global_expert_idx
            )
            local_routed_token_indices_with_count_sb = nl.ndarray(
                (dims.pmax, dynamism_cfg.T_plus_1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                local_routed_token_indices_with_count_sb[0, :],
                routed_token_indices_with_count_sb[expert_idx * 32, :],
            )

            # Step 2.3: Compute static blocks
            for static_block_idx in nl.sequential_range(dynamism_cfg.n_static_blocks):
                _compute_block(
                    input_tensors=input_tensors,
                    kernel_cfg=kernel_cfg,
                    dims=dims,
                    dynamism_cfg=dynamism_cfg,
                    weights=weights,
                    output_local=input_tensors.output,
                    routed_token_indices=local_routed_token_indices_with_count_sb,
                    arange_4H=arange_4H,
                    expert_idx=global_expert_idx,
                    block_idx=static_block_idx,
                    is_dynamic_block=False,
                    output_indices_hbm=output_indices_hbm,
                    is_first_expert=(global_expert_idx == 0),
                    is_last_expert=((global_expert_idx // n_prgs) == (dims.E_L // 2 - 1)),
                )

            # Step 2.4: Compute dynamic blocks.
            # Step 2.4.1: Move index vector to HBM, so that each dynamic block's indices can be indirect reloaded
            n_static_tokens = dynamism_cfg.n_static_blocks * dynamism_cfg.block_size
            n_dynamic_tokens = dynamism_cfg.n_dynamic_blocks * dynamism_cfg.block_size
            dynamic_block_token_indices_hbm = nl.ndarray(
                (1, n_dynamic_tokens),
                dtype=routed_token_indices_with_count_sb.dtype,
                buffer=nl.private_hbm,
                name=f'dynamic_block_token_indices_hbm_e{global_expert_idx}',
            )
            nisa.dma_copy(
                src=local_routed_token_indices_with_count_sb[0, nl.ds(n_static_tokens, n_dynamic_tokens)],
                dst=dynamic_block_token_indices_hbm[0, :],
                dge_mode=dge_mode.none,
            )
            dynamic_block_token_indices_hbm = dynamic_block_token_indices_hbm.reshape(
                (dynamism_cfg.n_dynamic_blocks, dynamism_cfg.block_size)
            )

            dynamic_iteration_count_reg = nisa.register_alloc()
            nisa.register_load(src=dynamic_iteration_count_sb[expert_lnc_grp * 64, 0], dst=dynamic_iteration_count_reg)

            # Step 2.4.3: Dynamic loop over dynamic blocks
            def _dynamic_block_body(dynamic_block_idx):
                _compute_block(
                    input_tensors=input_tensors,
                    kernel_cfg=kernel_cfg,
                    dims=dims,
                    dynamism_cfg=dynamism_cfg,
                    weights=weights,
                    output_local=input_tensors.output,
                    routed_token_indices=dynamic_block_token_indices_hbm,
                    arange_4H=arange_4H,
                    expert_idx=global_expert_idx,
                    block_idx=dynamic_block_idx,
                    is_dynamic_block=True,
                    output_indices_hbm=output_indices_hbm,
                    is_first_expert=(global_expert_idx == 0),
                    is_last_expert=((global_expert_idx // n_prgs) == (dims.E_L // 2 - 1)),
                )

            nl.fori_loop(0, dynamic_iteration_count_reg, _dynamic_block_body)
    return input_tensors.output


def _build_output_indices(input_tensors, dims, dynamism_cfg):
    """
    Build unpermute index mapping and gather sparse input token ids into packed rows in output buffer.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.

    Returns:
        output_indices_hbm (nl.NkiTensor): [T, 1], Maps input_position -> unpermuted output position.
    """

    # Step 1: Allocations
    token_indices_sb = nl.ndarray((dims.pmax, dims.T), dtype=nl.int32, buffer=nl.sbuf)
    routed_indices_sb = nl.ndarray((dims.pmax, dynamism_cfg.T_plus_1), dtype=nl.int32, buffer=nl.sbuf)
    iota_sb = nl.ndarray((dims.tile_T, dims.n_tiles_in_T), dtype=nl.int32, buffer=nl.sbuf)
    routed_indices_T_sb = nl.ndarray((dims.tile_T, dims.n_tiles_in_T), dtype=nl.float32, buffer=nl.sbuf)
    routed_indices_T_psum = nl.ndarray((dims.tile_T, dims.n_tiles_in_T), dtype=nl.float32, buffer=nl.psum)
    output_indices_hbm = nl.ndarray((dims.T, 1), dtype=nl.int32, buffer=nl.private_hbm)

    # Step 2: Find indices of routed tokens
    # Step 2.1: Load token_idx from end of input buffer: bitcast fp8[T,4] -> bf1632[T,2] -> reshape [1, T, 2]
    # NOTE: we use bf16 reinterpret because input is not always 4B aligned
    token_idx_col_offset_fp8 = dims.H + dims.H // _q_width + dims.E_L * FP8_PER_BF16
    token_idx_col_offset_bf16 = token_idx_col_offset_fp8 // FP8_PER_BF16
    src_view = (
        input_tensors.hidden_input.view(nl.bfloat16)
        .slice(dim=1, start=token_idx_col_offset_bf16, end=token_idx_col_offset_bf16 + BF16_PER_INT32)
        .expand_dim(dim=0)
    )  # shape (1, T, BF16_PER_INT32) bf16
    dst_view = (
        token_indices_sb.view(nl.bfloat16)
        .reshape_dim(dim=1, shape=(dims.T, BF16_PER_INT32))
        .slice(dim=0, start=0, end=1)
    )  # shape (1, T, BF16_PER_INT32) bf16
    nisa.dma_copy(src=src_view, dst=dst_view, dge_mode=dge_mode.none)

    # Step 2.2: Find routed token indices using NonzeroWithCount
    nisa.nonzero_with_count(
        src=token_indices_sb[...],
        padding_val=NONZERO_WITH_COUNT_PAD_VAL,
        dst=routed_indices_sb[...],
    )

    # Step 3: Build output indicies buffer in private HBM
    # Step 3.1: Build iota [pmax, n_free_tiles]: value[p, f] = p + f * pmax = sequential index 0..T-1
    nisa.iota(dst=iota_sb, pattern=[[dims.tile_T, dims.n_tiles_in_T]], offset=0, channel_multiplier=1)

    # Step 3.2: Transpose routed_indices_sb[0, 0:T] from [1, T] to [pmax, n_free_tiles] for use as vector_offset
    for tile_T_idx in range(dims.n_tiles_in_T):
        nisa.nc_transpose(
            data=routed_indices_sb.view(nl.float32)[0, nl.ds(dims.tile_T * tile_T_idx, dims.tile_T)],
            dst=routed_indices_T_psum[:, tile_T_idx],
        )
    nisa.tensor_copy(src=routed_indices_T_psum, dst=routed_indices_T_sb)
    routed_indices_T_int32_sb = routed_indices_T_sb.view(nl.int32)

    # Step 3.3: Scatter iota into output_indices_hbm using routed_positions as vector_offset
    for tile_T_idx in range(dims.n_tiles_in_T):
        nisa.dma_copy(
            src=iota_sb[:, tile_T_idx : tile_T_idx + 1],
            dst=output_indices_hbm.ap(
                pattern=[[1, dims.tile_T], [1, 1]],
                offset=0,
                vector_offset=routed_indices_T_int32_sb.ap(
                    pattern=[[dims.n_tiles_in_T, dims.tile_T], [1, 1]],
                    offset=tile_T_idx,
                ),
                indirect_dim=0,
            ),
            # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
            oob_mode=oob_mode.skip,
            dge_mode=dge_mode.swdge,
        )

    # Step 4: Gather token indices from input buffer and pack into output buffer
    # Tiles are sharded on T with ping-pong style across NCs for load balancing
    _, n_prgs, prg_id = get_verified_program_sharding_info()
    n_tiles_per_nc = div_ceil(dims.n_tiles_in_T, n_prgs)
    for local_tile_idx in range(n_tiles_per_nc):
        global_tile_idx = n_prgs * local_tile_idx + prg_id
        token_idx_tile_sb = nl.ndarray(
            (dims.tile_T, FP8_PER_INT32), dtype=input_tensors.hidden_input.dtype, buffer=nl.sbuf
        )
        if global_tile_idx < dims.n_tiles_in_T:
            # Initialize to 0 so skipped positions (padding) write 0 to output trailing columns
            # FIXME[perf]: We can potentially skip this memset or move it to GPSIMD
            nisa.memset(token_idx_tile_sb, 0, engine=nisa.vector_engine)
            # Gather token indices from input at routed positions for this tile
            nisa.dma_copy(
                src=input_tensors.hidden_input.ap(
                    pattern=[[FP8_PER_INT32, dims.tile_T], [1, FP8_PER_INT32]],
                    offset=token_idx_col_offset_fp8,
                    vector_offset=routed_indices_T_int32_sb.ap(
                        pattern=[[dims.n_tiles_in_T, dims.tile_T], [1, 1]],
                        offset=global_tile_idx,
                    ),
                    indirect_dim=0,
                ),
                dst=token_idx_tile_sb,
                oob_mode=oob_mode.skip,
                dge_mode=dge_mode.swdge,
            )
            # Spill token indices into contiguous rows in output
            nisa.dma_copy(
                src=token_idx_tile_sb.ap(
                    pattern=[[BF16_PER_INT32, dims.tile_T], [1, BF16_PER_INT32]],
                    offset=0,
                    dtype=input_tensors.output.dtype,
                ),
                dst=input_tensors.output[
                    nl.ds(global_tile_idx * dims.tile_T, dims.tile_T), nl.ds(dims.H, BF16_PER_INT32)
                ],
            )

    return output_indices_hbm


def _find_expert_routed_tokens(input_tensors, kernel_cfg, dims, dynamism_cfg, expert_4_tile_idx):
    """
    Find indices of tokens routed to a specific expert and build dynamic block decision vector,
    in 4 expert increment, i.e. find expert expert_4_tile_idx*4:min(dims.E_L, expert_4_tile_idx+4)

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        expert_4_tile_idx (int): Index of the tile of the expert to find routed tokens for.

    Returns:
        routed_token_indices_with_count_sb (nl.NkiTensor): [pmax, T+1], Output from nonzero_with_count in SBUF.
            Partition 0  contains routed token indices with count in final element for expert 0.
            Partition 32 contains routed token indices with count in final element for expert 1.
            Partition 64 and 96 contain results for experts 2 and 3 respectively.
        dynamic_conditions_sb (nl.NkiTensor): [pmax, n_dynamic_blocks+1], Decision vector indicating
            which dynamic blocks need computation. Expert e_id's decision is at partition e_id*32.
    """

    # Allocations
    expert_affinities_masked_T_f32_sb = nl.ndarray((dims.pmax, dims.T), dtype=nl.float32, buffer=nl.sbuf)
    # Zero-init: only partitions e_id*32 are loaded below. The shard-on-E path reduces the
    # per-partition routed-token counts across partition ranges (tensor_partition_reduce over
    # [0:33]/[0:97]) to compute the DLoC iteration count. Unloaded partitions must read 0 so
    # they contribute count=0 to the max; otherwise garbage differs per LNC core, producing
    # divergent dynamic-loop trip counts and an out-of-bounds scalar-DGE indirect access.
    nisa.memset(expert_affinities_masked_T_f32_sb, 0)
    routed_token_indices_with_count_sb = nl.ndarray((dims.pmax, dynamism_cfg.T_plus_1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(routed_token_indices_with_count_sb[:, dims.T], 0)
    dynamic_conditions_sb = nl.ndarray(
        (dims.pmax, dynamism_cfg.n_dynamic_blocks_plus_1), dtype=nl.int32, buffer=nl.sbuf
    )
    count_nonzero_f32 = nl.ndarray((dims.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)

    actual_iteration = min(4, dims.E_L - expert_4_tile_idx * 4)

    # Load expert affinities from [T, E_L] -> [1, T] with cast to fp32
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
        for e_id in range(actual_iteration):
            affinity_view = input_tensors.expert_affinities_masked.slice(
                dim=1, start=expert_4_tile_idx * 4 + e_id, end=expert_4_tile_idx * 4 + e_id + 1
            ).reshape((1, dims.T))
            # Load with cast to fp32
            nisa.dma_copy(src=affinity_view, dst=expert_affinities_masked_T_f32_sb[nl.ds(e_id * 32, 1), :])
    else:
        # A2A-v: expert affinities are bitcast as fp8 and located at col offset H + H/4
        # performance note: it is not faster to write a strided access pattern in this case because the stride is too large.
        for e_id in range(actual_iteration):
            affinities_col_offset_bf16 = (dims.H + dims.H // _q_width) // FP8_PER_BF16 + expert_4_tile_idx * 4 + e_id
            affinity_view = (
                input_tensors.hidden_input.view(kernel_cfg.expert_affinities_dtype)
                .slice(dim=1, start=affinities_col_offset_bf16, end=affinities_col_offset_bf16 + 1)
                .reshape((1, dims.T))
            )
            # Load with cast to fp32
            nisa.dma_copy(src=affinity_view, dst=expert_affinities_masked_T_f32_sb[nl.ds(e_id * 32, 1), :])

    # Find nonzero indices, with count
    # NOTE: partitions 1, ..., pmax are padding from nonzero_with_count output shape requirement
    nisa.nonzero_with_count(
        src=expert_affinities_masked_T_f32_sb[...],
        padding_val=NONZERO_WITH_COUNT_PAD_VAL,
        dst=routed_token_indices_with_count_sb[...],
    )

    """
    Build boolean dynamic block decision vector per expert, with final element 0 to ensure loop terminates
    when all dynamic blocks are computed. Uses partitions 0, 32, 64, 96 (valid engine start partitions).
    Example: iota=[129, 257, 385, 513], count=[483], less(iota, count)=[1, 1, 1, 0]
    """
    nisa.tensor_copy(
        src=routed_token_indices_with_count_sb[:, dims.T],
        dst=count_nonzero_f32,
    )
    nisa.iota(
        dst=dynamic_conditions_sb[...],
        pattern=[[dynamism_cfg.block_size, dynamism_cfg.n_dynamic_blocks_plus_1]],
        offset=dynamism_cfg.n_static_blocks * dynamism_cfg.block_size,
    )
    nisa.tensor_scalar(
        data=dynamic_conditions_sb[...],
        op0=nl.less,
        operand0=count_nonzero_f32,
        dst=dynamic_conditions_sb[...],
    )

    return routed_token_indices_with_count_sb, dynamic_conditions_sb


def _get_block_token_position_to_id(
    dynamism_cfg,
    routed_token_indices,
    arange_4H,
    block_idx,
    is_dynamic_block,
):
    """
    Build token position-to-ID mapping vectors for a specific block.

    Creates index vectors used for indirect DMAs to load hidden states and spill expert outputs for the current block.

    Args:
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        routed_token_indices (nl.NkiTensor): Token indices from nonzero_with_count. Shape depends on context:
            - Static blocks: [pmax, T+1] in SBUF, with count in final element
            - Dynamic blocks: [n_dynamic_blocks, block_size] in HBM
        arange_4H (nl.NkiTensor): [1, 4], Arange vector for 4_H broadcast.
        block_idx: Block index. Static: int literal. Dynamic: [1, 1] SBUF tensor.
        is_dynamic_block (bool): Whether this is a dynamic block (affects indexing pattern).

    Returns:
        token_position_to_id_4_H_T_sb (nl.NkiTensor): [blk_tile_T_x4, blk_n_T_x4_tiles], Transposed indices
            with 4_H broadcast for hidden state loading.
        token_position_to_id_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles], Transposed indices
            for expert affinity loading and output spilling.
    """

    # Step 1: Allocations
    # Token position to id with 4_H broadcast (hidden load)
    token_position_to_id_4_H_sb = nl.ndarray((1, dynamism_cfg.block_size, _q_width), dtype=nl.int32, buffer=nl.sbuf)
    token_position_to_id_4_H_T_psum = nl.ndarray(
        (dynamism_cfg.blk_tile_T_x4, dynamism_cfg.blk_n_T_x4_tiles), dtype=nl.float32, buffer=nl.psum
    )
    token_position_to_id_4_H_T_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T_x4, dynamism_cfg.blk_n_T_x4_tiles), dtype=nl.float32, buffer=nl.sbuf
    )

    # Token position to id (expert affinity load + expert MLP out spill)
    token_position_to_id_T_psum = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.float32, buffer=nl.psum
    )
    token_position_to_id_T_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.float32, buffer=nl.sbuf
    )

    # Step 2: Get indices
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
            dge_mode=nisa.dge_mode.hwdge,
        )
        indices_src = token_position_to_id_sb
        indices_pattern = [[dynamism_cfg.block_size, 1], [1, dynamism_cfg.block_size], [0, _q_width]]
        indices_offset = 0
    # Static: directly index indices in SBUF
    else:
        indices_src = routed_token_indices
        indices_pattern = [[dynamism_cfg.T_plus_1, 1], [1, dynamism_cfg.block_size], [0, _q_width]]
        indices_offset = dynamism_cfg.block_size * block_idx

    # Step 3: Broadcast indices from [1, block_size] -> [1, block_size, 4] to load interleaved T * 4_H dim
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

    # Step 4: Transpose 4H broadcast indices
    # Step 4.1: Flatten to [1, block_size * _q_width], reinterpret to fp32 for tranpose
    token_position_to_id_4_H_sb = token_position_to_id_4_H_sb.reshape((1, dynamism_cfg.blk_T_x4))
    token_position_to_id_4_H_sb = token_position_to_id_4_H_sb.view(nl.float32)

    # Step 4.2: PE transpose in fp32, then reinterpret to int32. We do this because PE does not support int32 dtypes.
    for tile_in in range(dynamism_cfg.blk_n_T_x4_tiles):
        nisa.nc_transpose(
            data=token_position_to_id_4_H_sb[
                0, nl.ds(dynamism_cfg.blk_tile_T_x4 * tile_in, dynamism_cfg.blk_tile_T_x4)
            ],
            dst=token_position_to_id_4_H_T_psum[:, tile_in],
        )
    nisa.tensor_copy(
        src=token_position_to_id_4_H_T_psum[...],
        dst=token_position_to_id_4_H_T_sb[...],
    )
    token_position_to_id_4_H_T_sb = token_position_to_id_4_H_T_sb.view(nl.int32)

    # Step 5: Transpose non-broadcast indices
    # Step 5.1: Reinterpret to fp32 for transpose
    if not is_dynamic_block:
        token_position_to_id_sb = routed_token_indices[
            0, nl.ds(dynamism_cfg.block_size * block_idx, dynamism_cfg.block_size)
        ]
    token_position_to_id_sb = token_position_to_id_sb.view(nl.float32)

    # Step 5.2: PE transpose in fp32, then reinterpret to int32. We do this because PE does not support int32 dtypes.
    for tile_out in range(dynamism_cfg.blk_n_T_tiles):
        nisa.nc_transpose(
            data=token_position_to_id_sb[0, nl.ds(dynamism_cfg.blk_tile_T * tile_out, dynamism_cfg.blk_tile_T)],
            dst=token_position_to_id_T_psum[:, tile_out],
        )
    nisa.tensor_copy(
        src=token_position_to_id_T_psum[...],
        dst=token_position_to_id_T_sb[...],
    )
    token_position_to_id_T_sb = token_position_to_id_T_sb.view(nl.int32)

    return token_position_to_id_4_H_T_sb, token_position_to_id_T_sb


def _get_block_token_position_to_id_a2av(
    dynamism_cfg,
    routed_token_indices,
    block_idx,
    is_dynamic_block,
):
    """
    Build token position-to-ID mapping vector for a specific block (A2A-v variant).

    Like _get_block_token_position_to_id but only creates the T-dimension index vector
    (no 4_H broadcast), since A2A-v loads the concatenated buffer without interleaved T * 4_H layout.

    Args:
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        routed_token_indices (nl.NkiTensor): Token indices from nonzero_with_count.
        block_idx: Block index. Static: int literal. Dynamic: [1, 1] SBUF tensor.
        is_dynamic_block (bool): Whether this is a dynamic block.

    Returns:
        token_position_to_id_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles], Transposed indices
            for indirect loading and output spilling.
    """
    # Token position to id (load + spill)
    token_position_to_id_T_psum = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.float32, buffer=nl.psum
    )
    token_position_to_id_T_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.float32, buffer=nl.sbuf
    )

    # Load indices to SBUF
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
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        token_position_to_id_sb = routed_token_indices[
            0, nl.ds(dynamism_cfg.block_size * block_idx, dynamism_cfg.block_size)
        ]

    # Transpose indices, reinterpreting to fp32 since PE doesn't support int32 input
    token_position_to_id_sb = token_position_to_id_sb.view(nl.float32)
    for tile_out in range(dynamism_cfg.blk_n_T_tiles):
        nisa.nc_transpose(
            data=token_position_to_id_sb[0, nl.ds(dynamism_cfg.blk_tile_T * tile_out, dynamism_cfg.blk_tile_T)],
            dst=token_position_to_id_T_psum[:, tile_out],
        )
    nisa.tensor_copy(
        src=token_position_to_id_T_psum[...],
        dst=token_position_to_id_T_sb[...],
    )
    token_position_to_id_T_sb = token_position_to_id_T_sb.view(nl.int32)

    return token_position_to_id_T_sb


def _layout_adapter_qmx_hbm(
    input: nl.NkiTensor,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig = None,
    input_indices_T_sb: nl.NkiTensor = None,
    output_dtype: nki.dtype = nl.float8_e4m3fn_x4,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Load input from HBM, transform tensor into swizzled layout, and perform quantization to MXFP8.

    Args:
        input (nl.NkiTensor): [T, 4_H * H/512 * 16_H * 8_H], Input tensor in HBM.
        dims (AllExpertMXDimensions): Dimension parameters. Uses full-T tiling when input_indices_T_sb is None,
            otherwise uses per-block tiling. Uses dims.t32_tile_offset for T-sharding.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters. Required when input_indices_T_sb is provided.
        input_indices_T_sb (nl.NkiTensor): [32_T * 4_H, T/32] Optional indices for indirect load from HBM.
        output_dtype (nki.dtype): MXFP8 dtype to quantize to.

    Returns:
        output_quant_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Quantized output in SBUF
            (4_H packed in x4 dtype).
        output_scale_sb (nl.NkiTensor): [16_H * 8_H, H/512, T], Scales in SBUF
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
    n_T32_tiles_global = div_ceil(dims.T, NUM_H4_FOLDS_PER_COLUMN)
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
                dge_mode=nisa.dge_mode.swdge,
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


def _layout_adapter_a2av_hbm(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig,
    input_indices_T_sb: nl.NkiTensor,
    expert_idx: Optional[Union[int, nl.NkiTensor]],
):
    """
    Load concatenated A2A-v input from HBM, unpack into separate SBUF tensors for expert MLP.

    Input is a concatenated fp8 buffer [T, H_concat] where H_concat = H + H/4 + E_L*2 + 4,
    representing [hidden_quant | hidden_scale | expert_affinities | token_index] bitcast to fp8.

    Args:
        input_tensors (AllExpertMXInputTensors): Contains hidden_input [T, H_concat] in HBM.
        kernel_cfg (AllExpertMXKernelConfig): Kernel configuration (expert_affinities_dtype).
        dims (AllExpertMXDimensions): Dimension parameters (H, H_concat, E_L, tile_H, n_H512_tiles).
        dynamism_cfg (AllExpertMXDynamismConfig): Block tiling parameters (blk_tile_T, blk_n_T_tiles, block_size).
        input_indices_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles], Token indices for indirect DMA gather.
        expert_idx (int or nl.NkiTensor): Index of the current expert for affinity extraction.

    Returns:
        input_quant_sb (nl.NkiTensor): [128_H, H/512, T], Quantized hidden states in SBUF (4_H packed in x4 dtype).
        input_scale_sb (nl.NkiTensor): [128_H, H/512, T], MX scales in SBUF (uint8).
        expert_affinities_masked_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles, 1], Per-token expert affinities in fp32.
    """
    # Step 1: shapes, allocations
    input_quant_x4_dtype = MXFP8_UNPACKED_PACKED_MAP[input_tensors.hidden_input.dtype]

    input_quant_sb = nl.ndarray(
        (dims.tile_H, dims.n_H512_tiles, dynamism_cfg.block_size),
        dtype=FP8X4_TP_VIEW_DTYPE,
        buffer=nl.sbuf,
    )
    input_scale_sb = nl.ndarray(
        (dims.tile_H, dims.n_H512_tiles, dynamism_cfg.block_size),
        dtype=UINT8_TP_VIEW_DTYPE,
        buffer=nl.sbuf,
    )
    expert_affinities_masked_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )

    # Step 2: Load entire concatenated buffer, except for global token indices, in tiles of [128_T, H_concat - 4]
    # NOTE[perf]: may be better to do load as a loop and then compute as a loop due to strided TensorCopy
    for tile_T_idx in range(dynamism_cfg.blk_n_T_tiles):
        # Step 2.1: Load indirect
        # NOTE: we need to pad to multiple of 4 for f8->f32 reinterpret
        H_concat_without_indices = dims.H_concat - FP8_PER_INT32
        H_concat_4B_aligned = div_ceil(H_concat_without_indices, FP8_PER_FP32) * FP8_PER_FP32
        input_concat_tile_sb = nl.ndarray(
            (dynamism_cfg.blk_tile_T, H_concat_4B_aligned),
            dtype=input_tensors.hidden_input.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(
            src=input_tensors.hidden_input.ap(
                pattern=[[dims.H_concat, dynamism_cfg.blk_tile_T], [1, H_concat_4B_aligned]],
                offset=0,
                vector_offset=input_indices_T_sb.ap(
                    pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                    offset=tile_T_idx,
                ),
                indirect_dim=0,
            ),
            dst=input_concat_tile_sb,
            # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
        )

        # Step 2.2: Transpose input_quant from [T, H/512 * 128_H * 4_H] to [128_H, H/512, T] with 4_H packed in dtype
        # Reinterpret f8 -> f32 to preserve 4_H as innermost dim. We reinterpret to f32 since PE transpose does not allow f8x4.
        n_H512_fp32_tiles_per_group = nl.tile_size.psum_fmax // dims.tile_H
        n_H512_fp32_groups = div_ceil(dims.n_H512_tiles, n_H512_fp32_tiles_per_group)
        for H512_group_idx in range(n_H512_fp32_groups):
            n_H512_tiles_actual = min(
                n_H512_fp32_tiles_per_group, dims.n_H512_tiles - (n_H512_fp32_tiles_per_group * H512_group_idx)
            )
            input_quant_tile_psum = nl.ndarray(
                (dims.tile_H, n_H512_tiles_actual, dynamism_cfg.blk_tile_T), dtype=nl.float32, buffer=nl.psum
            )
            for H512_tile_idx in range(n_H512_tiles_actual):
                global_H512_tile_idx = H512_group_idx * n_H512_fp32_tiles_per_group + H512_tile_idx
                nisa.nc_transpose(
                    data=input_concat_tile_sb.ap(
                        pattern=[[H_concat_4B_aligned // FP8_PER_FP8X4, dynamism_cfg.blk_tile_T], [1, dims.tile_H]],
                        offset=global_H512_tile_idx * dims.tile_H,
                        dtype=FP8X4_TP_VIEW_DTYPE,
                    ),
                    dst=input_quant_tile_psum[:, H512_tile_idx, :],
                )
            nisa.tensor_copy(
                src=input_quant_tile_psum,
                dst=input_quant_sb[
                    :,
                    nl.ds(H512_group_idx * n_H512_fp32_tiles_per_group, n_H512_tiles_actual),
                    nl.ds(tile_T_idx * dynamism_cfg.blk_tile_T, dynamism_cfg.blk_tile_T),
                ],
            )

        # Step 2.3: Transpose input_scale from [T, H/512 * 128_H] to [128_H, H/512, T]
        # Reinterpret u8 -> f8_e5 since PE transpose does not allow u8.
        fp8_transpose_interleave = FP8_PER_BF16
        n_H512_fp8_tiles_per_group = nl.tile_size.psum_fmax * BF16_PER_FP32 // dims.tile_H
        n_H512_fp8_groups = div_ceil(dims.n_H512_tiles, n_H512_fp8_tiles_per_group)
        for H512_group_idx in range(n_H512_fp8_groups):
            n_H512_tiles_actual = min(
                n_H512_fp8_tiles_per_group, dims.n_H512_tiles - (n_H512_fp8_tiles_per_group * H512_group_idx)
            )
            input_scale_tile_psum = nl.ndarray(
                (dims.tile_H, n_H512_tiles_actual, dynamism_cfg.blk_tile_T, fp8_transpose_interleave),
                dtype=UINT8_TP_VIEW_DTYPE,
                buffer=nl.psum,
            )
            for H512_tile_idx in range(n_H512_tiles_actual):
                global_H512_tile_idx = H512_group_idx * n_H512_fp8_tiles_per_group + H512_tile_idx
                nisa.nc_transpose(
                    data=input_concat_tile_sb.ap(
                        pattern=[[H_concat_4B_aligned, dynamism_cfg.blk_tile_T], [1, dims.tile_H]],
                        offset=dims.H + global_H512_tile_idx * dims.tile_H,
                        dtype=UINT8_TP_VIEW_DTYPE,
                    ),
                    dst=input_scale_tile_psum[:, H512_tile_idx, :, 0],
                )
            nisa.tensor_copy(
                src=input_scale_tile_psum[:, :, :, 0],
                dst=input_scale_sb[
                    :,
                    nl.ds(H512_group_idx * n_H512_fp8_tiles_per_group, n_H512_tiles_actual),
                    nl.ds(tile_T_idx * dynamism_cfg.blk_tile_T, dynamism_cfg.blk_tile_T),
                ],
            )

        # Step 2.4: TensorCopy expert affinities, with upcast from bf16 -> fp32. Fp32 is used in down
        # projection to run affinity scaling on ACT.
        hidden_quant_scale_col_offset_fp8 = dims.H + dims.H // _q_width
        affinities_offset_fp8 = dims.E_L * FP8_PER_BF16
        nisa.tensor_copy(
            src=input_concat_tile_sb.ap(
                pattern=[[H_concat_4B_aligned // FP8_PER_BF16, dynamism_cfg.blk_tile_T], [1, 1]],
                offset=hidden_quant_scale_col_offset_fp8 // FP8_PER_BF16 + expert_idx,
                dtype=kernel_cfg.expert_affinities_dtype,
            ),
            dst=expert_affinities_masked_sb[:, tile_T_idx, :],
        )

    # Reinterpret quantized hidden states to f8x4 and u8
    input_quant_sb = input_quant_sb.view(input_quant_x4_dtype)
    input_scale_sb = input_scale_sb.view(MX_SCALE_DTYPE)

    return input_quant_sb, input_scale_sb, expert_affinities_masked_sb


def _load_block_expert_affinities(input_tensors, dims, dynamism_cfg, token_position_to_id_T_sb, expert_idx):
    """
    Load expert affinities for tokens in a block using indirect DMA.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
        token_position_to_id_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles], Token position-to-ID mapping.
        expert_idx (int): Index of the expert to load affinities for.

    Returns:
        expert_affinities_masked_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles, 1], Expert affinities
            for the block's tokens in SBUF.
    """
    # Allocation
    expert_affinities_masked_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles, 1),
        dtype=input_tensors.expert_affinities_masked.dtype,
        buffer=nl.sbuf,
    )

    # Load expert affinities for this block
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
            dge_mode=nisa.dge_mode.swdge,
        )

    return expert_affinities_masked_sb


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

    is_software_quant = kernel_cfg.is_static_quant or kernel_cfg.is_row_quant

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
        skip_scale_load=is_software_quant,
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
        skip_scale_load=is_software_quant,
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
        use_PE_bias_broadcast=True,
        sharding_strategy=dims.sharding_strategy,
        skip_scale_load=is_software_quant,
    )

    # ROW_MX: compute dequant scales per expert
    gate_dequant_scale_sb = None
    up_dequant_scale_sb = None
    down_dequant_scale_sb = None
    if kernel_cfg.is_row_quant:
        # ROW_MX: load per-row weight dequant scales per expert
        # Reuse gate_dequant_scale_sb/up_dequant_scale_sb for weight scales (dispatch on shape downstream)
        pmax = nl.tile_size.pmax

        # gate/up weight scales from [E_L, 2, n_I512*4]
        gate_up_scale_cols = input_tensors.gate_up_weights_scale.shape[2]
        gate_dequant_scale_sb = nl.ndarray((pmax, gate_up_scale_cols), dtype=nl.float32, buffer=nl.sbuf)
        up_dequant_scale_sb = nl.ndarray((pmax, gate_up_scale_cols), dtype=nl.float32, buffer=nl.sbuf)

        # Load gate/up weight dequant scales with partition-dim broadcast
        gate_up_w_view = input_tensors.gate_up_weights_scale.select(dim=0, index=expert_idx)
        nisa.dma_copy(
            dst=gate_dequant_scale_sb,
            src=gate_up_w_view.slice(dim=0, start=0, end=1).broadcast(dim=0, size=pmax),
        )
        nisa.dma_copy(
            dst=up_dequant_scale_sb,
            src=gate_up_w_view.slice(dim=0, start=1, end=2).broadcast(dim=0, size=pmax),
        )

        # Load down weight scale with partition-dim broadcast from [E_L, H//_pmax]
        down_scale_cols = input_tensors.down_weights_scale.shape[1]
        down_dequant_scale_sb = nl.ndarray((dims.tile_T, down_scale_cols), dtype=nl.float32, buffer=nl.sbuf)
        down_w_view = (
            input_tensors.down_weights_scale.select(dim=0, index=expert_idx)  # [down_scale_cols]
            .reshape_dim(dim=0, shape=(1, down_scale_cols))  # [1, down_scale_cols]
            .broadcast(dim=0, size=dims.tile_T)  # [tile_T, down_scale_cols]
        )
        nisa.dma_copy(dst=down_dequant_scale_sb, src=down_w_view)

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
        gate_dequant_scale_sb=gate_dequant_scale_sb,
        up_dequant_scale_sb=up_dequant_scale_sb,
        down_dequant_scale_sb=down_dequant_scale_sb,
    )


def _compute_expert_mlp(
    input_quant: nl.NkiTensor,
    input_scale: nl.NkiTensor,
    weights: ExpertWeightsSBUF,
    kernel_cfg: AllExpertMXKernelConfig,
    expert_affinities_masked: nl.NkiTensor,
    output_sb: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    expert_idx: int,
    is_first_expert: bool,
    is_last_expert: bool,
    sharding_strategy: MoELNCShardingStrategy = MoELNCShardingStrategy.SHARD_I,
    T_offset: int = 0,
    token_position_to_id_T: nl.NkiTensor = None,
    input_dequant_scale_sb: nl.NkiTensor = None,
    output_t_offset: int = 0,
    is_software_quant: bool = False,
    T_physical: int = None,
    down_in_scale_sb: nl.ndarray = None,
) -> nl.ndarray:
    """
    Compute expert MLP for one block of input.

    Args:
        input_quant (nl.NkiTensor): Quantized input tensor.
        input_scale (nl.NkiTensor): Input scale tensor.
        weights (ExpertWeightsSBUF): Expert weights, scales, and biases in SBUF.
        kernel_cfg (AllExpertMXKernelConfig): Kernel config parameters.
        expert_affinities_masked (nl.NkiTensor): Masked expert affinities.
        output_sb (nl.NkiTensor): Output tensor in SBUF.
        output_hbm (nl.NkiTensor): Output tensor in HBM.
        expert_idx (int): Expert index.
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.
        sharding_strategy (MoELNCShardingStrategy): LNC sharding strategy.
        T_offset (int): Offset for T dimension in HBM output.
        token_position_to_id_T (nl.NkiTensor): Token position to ID mapping for blockwise DMA.
        input_dequant_scale_sb (nl.NkiTensor): Optional input dequantization scales for ROW_MX mode.
        is_software_quant (bool): When True, weight scales are 2D [128, F] dummy tiles indexed
            as [:, :slice] instead of the normal 3D [:, tile_idx, slice].
    Returns:
        output_sb: Output tensor in SBUF.
    """

    is_row_quant = kernel_cfg.is_row_quant
    is_static_mx = kernel_cfg.is_static_quant

    # Step 1: Compute gate/up projection, projection clamping, activation function, and QMX
    act_quant_sb, act_scale_sb = gate_up_projection_mx(
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
        gate_dequant_scale=weights.gate_dequant_scale_sb,
        up_dequant_scale=weights.up_dequant_scale_sb,
        input_dequant_scale=input_dequant_scale_sb if is_row_quant else None,
        is_software_quant=is_software_quant,
    )

    # Step 2 (ROW_MX only): row-quantize intermediate gate*up output for down projection
    if is_row_quant:
        # act_quant_sb is bf16 [TILE_I, n_I512_tiles, T, I_4] for ROW_MX
        TILE_I = act_quant_sb.shape[0]
        n_I512_tiles = act_quant_sb.shape[1]
        T_act = act_quant_sb.shape[2]
        I_4 = act_quant_sb.shape[3]

        # Permute [TILE_I, n_I512, T, I_4] → [TILE_I, T, n_I512 * I_4] for per-token quantization
        act_permuted = nl.ndarray((TILE_I, T_act, n_I512_tiles * I_4), dtype=act_quant_sb.dtype, buffer=nl.sbuf)
        for i_tile in nl.affine_range(n_I512_tiles):
            for i_q in nl.affine_range(I_4):
                col = i_tile * I_4 + i_q
                if col % 2 == 0:
                    nisa.activation(
                        dst=act_permuted[:, :T_act, col], data=act_quant_sb[:, i_tile, :T_act, i_q], op=nl.copy
                    )
                else:
                    nisa.tensor_copy(
                        dst=act_permuted[:, :T_act, col],
                        src=act_quant_sb[:, i_tile, :T_act, i_q],
                        engine=nisa.vector_engine,
                    )

        quantized_inter, inter_dequant_scale = row_quantization(
            act_permuted,
            output_dtype=nl.float8_e4m3fn,
        )
        # inter_dequant_scale: [TILE_I, T, 1]

        # Swizzle fp8 back to [TILE_I, n_I512, T, I_4] layout, then reinterpret as fp8_x4
        swizzled_fp8 = nl.ndarray((TILE_I, n_I512_tiles, T_act, I_4), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        for i_tile in nl.affine_range(n_I512_tiles):
            for i_q in nl.affine_range(I_4):
                col = i_tile * I_4 + i_q
                if col % 2 == 0:
                    nisa.activation(
                        dst=swizzled_fp8[:, i_tile, :T_act, i_q], data=quantized_inter[:, :T_act, col], op=nl.copy
                    )
                else:
                    nisa.tensor_copy(
                        dst=swizzled_fp8[:, i_tile, :T_act, i_q],
                        src=quantized_inter[:, :T_act, col],
                        engine=nisa.vector_engine,
                    )

        # Flatten to 2D, view as uint32 (packs 4 fp8 → 1 uint32), then view as fp8_x4
        total_fp8 = n_I512_tiles * T_act * I_4
        total_x4 = n_I512_tiles * T_act
        swizzled_flat = swizzled_fp8.reshape((TILE_I, total_fp8))
        temp_quant = nl.ndarray((TILE_I, total_x4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=temp_quant.view(nl.uint32),
            src=swizzled_flat.view(nl.uint32),
            engine=nisa.vector_engine,
        )
        act_quant_sb = temp_quant.reshape((TILE_I, n_I512_tiles, T_act))

        # SW quant: reuse the hoisted 2D dummy scale [128, F] instead of per-tile 3D scale
        act_scale_sb = weights.dummy_scale_tile_sb

    # Step 2 (STATIC_MX only): static-quantize intermediate with down_in_scale before down projection
    elif is_static_mx:
        # act_quant_sb is bf16 [TILE_I, n_I512_tiles, T, I_4] from gate_up_projection_mx
        TILE_I = act_quant_sb.shape[0]
        n_I512_tiles = act_quant_sb.shape[1]
        T_act = act_quant_sb.shape[2]
        I_4 = act_quant_sb.shape[3]

        # Flatten and apply static_quantization (divides by down_in_scale, clips to FP8 range)
        total_free = n_I512_tiles * T_act * I_4
        act_flat = act_quant_sb.reshape((TILE_I, total_free))
        quantized_flat, _ = static_quantization(act_flat, down_in_scale_sb)

        # Cast bf16 to fp8, then reinterpret as fp8_x4
        quantized_fp8 = nl.ndarray(quantized_flat.shape, dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        nisa.tensor_copy(dst=quantized_fp8, src=quantized_flat)
        total_x4 = n_I512_tiles * T_act
        temp_quant = nl.ndarray((TILE_I, total_x4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        nisa.tensor_copy(dst=temp_quant.view(nl.uint32), src=quantized_fp8.view(nl.uint32), engine=nisa.vector_engine)
        act_quant_sb = temp_quant.reshape((TILE_I, n_I512_tiles, T_act))

        # Dummy 127 MX scales
        act_scale_sb = weights.dummy_scale_tile_sb

    # Step 3: Compute down projection, expert affinity scaling, expert add, LNC reduction, and SB->HBM spill
    down_projection_mx(
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
        sharding_strategy=sharding_strategy,
        T_offset=T_offset,
        token_position_to_id_T=token_position_to_id_T,
        down_dequant_scale=weights.down_dequant_scale_sb,
        down_input_dequant_scale=inter_dequant_scale if is_row_quant else None,
        output_t_offset=output_t_offset,
        is_software_quant=is_software_quant,
        T_physical=T_physical,
    )

    return output_sb


def _compute_block(
    input_tensors,
    kernel_cfg,
    dims,
    dynamism_cfg,
    output_local,
    weights,
    routed_token_indices,
    arange_4H,
    expert_idx,
    block_idx,
    is_dynamic_block,
    output_indices_hbm=None,
    is_first_expert=None,
    is_last_expert=None,
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
        routed_token_indices (nl.NkiTensor): Token indices from nonzero_with_count.
            - Static blocks: [pmax, T+1] in SBUF, with count in final element
            - Dynamic blocks: [n_dynamic_blocks, block_size] in HBM
        arange_4H (nl.NkiTensor): [1, 4], Arange vector for 4_H broadcast.
        expert_idx (int): Index of the current expert.
        block_idx: Block index. Static: int literal. Dynamic: [1, 1] SBUF tensor.
        is_dynamic_block (bool): Whether this is a dynamic block (affects indexing pattern).
        is_first_expert (Optional[bool]): Whether it is the first expert, affect accumulation behaviour,
            computed by is_first_expert=(expert_idx == 0) if not specified
        is_last_expert (Optional[bool]): Whether it is the last expert, affect accumulation behaviour,
            computed by is_last_expert=(expert_idx == dims.E_L - 1) if not specified


    Returns:
        None. Results are accumulated into input_tensors.output via _compute_expert_mlp.
    """
    if is_first_expert == None:
        is_first_expert = expert_idx == 0
    if is_last_expert == None:
        is_last_expert = expert_idx == dims.E_L - 1

    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
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

    else:
        # Build token_position_to_id vector for load for this block
        token_position_to_id_T_sb = _get_block_token_position_to_id_a2av(
            dynamism_cfg=dynamism_cfg,
            routed_token_indices=routed_token_indices,
            block_idx=block_idx,
            is_dynamic_block=is_dynamic_block,
        )

        # Load and transpose pre-quantized hidden states for this block
        input_quant_sb, input_scale_sb, expert_affinities_masked_sb = _layout_adapter_a2av_hbm(
            input_tensors=input_tensors,
            kernel_cfg=kernel_cfg,
            dims=dims,
            dynamism_cfg=dynamism_cfg,
            input_indices_T_sb=token_position_to_id_T_sb,
            expert_idx=expert_idx,
        )

        # Load output indices when output layout must be packed
        if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS:
            spill_indices_sb = nl.ndarray(
                (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.int32, buffer=nl.sbuf
            )
            # Initialize to -1 so that unrouted tokens get skipped when spilling expert MLP output
            nisa.memset(spill_indices_sb, -1)
            for tile_T_idx in range(dynamism_cfg.blk_n_T_tiles):
                nisa.dma_copy(
                    src=output_indices_hbm.ap(
                        pattern=[[1, dynamism_cfg.blk_tile_T], [1, 1]],
                        offset=0,
                        vector_offset=token_position_to_id_T_sb.ap(
                            pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                            offset=tile_T_idx,
                        ),
                        indirect_dim=0,
                    ),
                    dst=spill_indices_sb[:, tile_T_idx : tile_T_idx + 1],
                    # When a token is not routed to a given expert, vector_offset[token] = -1 and we skip DMA
                    oob_mode=oob_mode.skip,
                    dge_mode=dge_mode.swdge,
                )
            token_position_to_id_T_sb = spill_indices_sb

    # Allocate SBUF result buffer for MLP(block)
    H_pad = 0 if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED else BF16_PER_INT32
    output_shape = (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles, dims.H + H_pad)
    output_sb = nl.ndarray(output_shape, dtype=kernel_cfg.activation_compute_dtype, buffer=nl.sbuf)
    # Zero output buffer to prevent uninitialized SBUF memory from leaking to HBM
    # for unrouted token positions via the indirect DMA scatter (oob_mode.skip does
    # not fully suppress writes when the source is a sub-tensor view).
    nisa.memset(output_sb, 0, engine=nisa.vector_engine)

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
        is_first_expert=is_first_expert,
        is_last_expert=is_last_expert,
        sharding_strategy=dims.sharding_strategy,
        token_position_to_id_T=token_position_to_id_T_sb,
    )


def _static_mx_needs_shard_t(
    T: int, H: int, E_L: int, I: int, weight_dtype: "nki.dtype", input_dtype: "nki.dtype"
) -> bool:
    """Check if STATIC_MX all-expert bf16 input exceeds SBUF capacity with SHARD_I.

    STATIC_MX keeps bf16 input in SBUF for per-expert quantization. When T is large,
    the bf16 footprint (input + swizzled copy) may exceed SBUF capacity alongside
    weights and output. Returns True if SHARD_T is needed to halve T_local.

    Args:
        T: Total number of tokens (padded to multiple of 4).
        H: Hidden dimension size.
        E_L: Number of local experts.
        I: Intermediate dimension size.
        weight_dtype: Expert weight dtype (e.g., nl.float8_e4m3fn_x4).
        input_dtype: Input/activation dtype (e.g., nl.bfloat16).

    Returns:
        True if the estimated SBUF footprint exceeds capacity and SHARD_T is needed.
    """
    pmax = nl.tile_size.pmax
    H_free = H // pmax
    input_bytes = _dtype_size(input_dtype)
    # STATIC_MX requires trn3 (MX matmul engine), so use trn3 capacity
    sbuf_capacity = _SBUF_USABLE_PER_PARTITION["trn3"]

    # Weight footprint (same as _get_tile_size)
    n_I512 = div_ceil(I, 512)
    n_I512_local = div_ceil(n_I512, 2)  # LNC-2 sharding
    weight_bytes = _dtype_size(weight_dtype)
    weight_per_part = n_I512_local * H * weight_bytes + n_I512_local * H

    # bf16 input + swizzled copy (both live simultaneously during static_quantization)
    _NUM_CONCURRENT_INPUT_BUFFERS = 2
    input_per_part = T * H_free * input_bytes * _NUM_CONCURRENT_INPUT_BUFFERS

    # Affinities (float32) + output accumulator
    n_T128 = div_ceil(T, pmax)
    affinities_per_part = n_T128 * E_L * _dtype_size(nl.float32)
    output_per_part = n_T128 * H * input_bytes

    total_required_mem = weight_per_part + input_per_part + affinities_per_part + output_per_part
    return total_required_mem > sbuf_capacity
