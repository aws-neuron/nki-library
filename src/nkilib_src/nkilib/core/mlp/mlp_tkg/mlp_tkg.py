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

"""MLP TKG kernel implementation for token generation scenarios with optional normalization and fused add."""

from typing import Optional

import nki  # noqa: F401
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import BufferManager, SbufManager
from ...utils.common_types import ActFnType, NormType, QuantizationType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...utils.logging import Logger, get_logger
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import (
    BS_TILE_SIZE,
    MLPParameters,
    mlpp_has_fused_add,
    mlpp_has_normalization,
    mlpp_has_normalization_bias,
    mlpp_has_projection_bias,
    mlpp_store_fused_add,
)
from .llama3_70b_high_batch import mlp_tkg_llama3_70b_high_batch
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import process_gate_up_projection
from .mlp_tkg_utils import (
    input_fused_add,
    input_norm_load,
    transpose_store_sbuf_copy,
)


def _mlp_tkg_impl(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
    sbm: Optional[BufferManager] = None,
) -> list[nl.NkiTensor]:
    """
    Allocated kernel that computes Norm(hidden) @ wMLP for token generation.

    This kernel performs the MLP forward pass with optional normalization and fused
    add operations. It is optimized for token generation (TKG/decode) scenarios.
    TODO: Specify intended usage range (e.g., sequence length, batch size)

    Dimensions:
        B: Batch size
        S: Sequence length
        T: Total tokens (B * S)
        H: Hidden dimension size
        I: Intermediate dimension size
        H0: Partition dimension (128)
        H1: H // H0

    Args:
        params (MLPParameters): MLP configuration containing:
            - hidden_tensor: Input tensor [B, S, H] or [H0, T, H1] in SBUF
            - gate_proj_weights_tensor: Gate projection weights [H, I]
            - up_proj_weights_tensor: Up projection weights [H, I]
            - down_proj_weights_tensor: Down projection weights [I, H]
            - activation_fn: Activation function type (e.g., SiLU)
            - output_dtype: Output data type
            - fused_add_params: Fused add configuration
            - norm_params: Normalization configuration
            - bias_params: Bias tensors for projections
            - quant_params: Quantization configuration
            - eps: Epsilon for normalization
            - store_output_in_sbuf: If True, keep output in SBUF
            - skip_gate_proj: If True, skip gate projection and only use up projection
            - use_tkg_gate_up_proj_column_tiling: Matmul mode for gate/up projection
            - use_tkg_down_proj_column_tiling: Matmul mode for down projection
            - use_tkg_down_proj_optimized_layout: Use optimized weight layout for down projection, only applicable when use_tkg_down_proj_column_tiling is off
            - gate_clamp_lower_limit: Lower clamp limit for gate projection output
            - gate_clamp_upper_limit: Upper clamp limit for gate projection output
            - up_clamp_lower_limit: Lower clamp limit for up projection output
            - up_clamp_upper_limit: Upper clamp limit for up projection output
        output_tensor_hbm (nl.NkiTensor): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.NkiTensor): [B, S, H], Optional fused add result storage
        sbm (BufferManager): Optional BufferManager for SBUF allocation with consistent naming.

    Returns:
        list[nl.NkiTensor]: List containing:
            - output_tensor_hbm or down_sb: MLP output tensor
            - output_stored_add_tensor_hbm: (optional) Fused add result if store_fused_add_result=True

    Notes:
        - Supports RMSNorm and LayerNorm normalization
        - Supports static and row-wise quantization
        - Uses BufferManager for memory allocation
        - When skip_gate_proj=True, only up projection with activation is performed
        - Matmul projection modes:
            - Column tiling: hidden is stationary tensor, weight is moving tensor
            - LHS/RHS swap: weight is stationary tensor, hidden is moving tensor

    Pseudocode:
        # Step 1: Optional fused add
        if has_fused_add:
            hidden = hidden + fused_add_tensor

        # Step 2: Optional normalization
        if has_normalization:
            hidden = normalize(hidden)

        # Step 3: Gate/Up projection with activation
        if skip_gate_proj:
            up_out = hidden @ up_weight + up_bias
            intermediate = activation(up_out)
        else:
            gate_out = hidden @ gate_weight + gate_bias
            up_out = hidden @ up_weight + up_bias
            intermediate = activation(gate_out) * up_out

        # Step 4: Down projection
        output = intermediate @ down_weight + down_bias
    """

    io_dtype = params.hidden_tensor.dtype

    # ---------------- Compute Kernel Dimensions & SBUF Manager ----------------
    dims = MLPTKGConstants.calculate_constants(params)

    # Always create internal BufferManager for SBUF allocations.
    if sbm is None:
        sbm = SbufManager(0, 200 * 1024, get_logger("mlp_tkg"))
        sbm.set_name_prefix("mlp_")
    else:
        kernel_assert(
            not sbm.is_auto_alloc(),
            "If the MLP kernel receives a SBM from the caller, that SBM must be manually allocated",
        )
        kernel_assert(
            sbm.get_free_space() >= 200 * 1024,
            f"If the MLP kernel receives a SBM from the caller, the SBM must have the free space of at least 200 * 1024 elements",
        )
    sbm.open_scope()  # Start SBUF allocation scope

    # ---------------- Fused Add ----------------
    # Apply fused add if present (hidden + attention output)
    hidden = params.hidden_tensor
    if mlpp_has_fused_add(params):
        if not params.fused_add_params.store_fused_add_result:
            fused_add_output = sbm.alloc(
                (params.batch_size, params.sequence_len, params.hidden_size),
                dtype=params.output_dtype,
                buffer=nl.shared_hbm,
                name="output_stored_add_tensor_hbm",
            )
        else:
            fused_add_output = output_stored_add_tensor_hbm

        input_fused_add(
            input=hidden,
            fused_add_tensor=params.fused_add_params.fused_add_tensor,
            fused_output=fused_add_output,
            normtype=params.norm_params.normalization_type,
            sbm=sbm,
            dims=dims,
        )
        hidden = fused_add_output  # Use fused result as hidden input

    # ---------------- Norm / Input Load ----------------
    if mlpp_has_normalization(params) or (not (hidden.buffer == nl.sbuf)):
        input_sb = input_norm_load(hidden, params, dims, sbm)
    else:
        # Hidden already in SBUF — use dims.hidden_layout (defaults to H0_T_H1 if caller didn't set it)
        input_sb = hidden

    # ---------- Process gate/up projection, silu, gate/up multiplication ----------
    # Allocate SBUF tile for gate/up projection output
    gate_up_sb = sbm.alloc_stack(
        (dims.I0, div_ceil(dims.I, dims.I0), dims.T),
        dtype=io_dtype,
        buffer=nl.sbuf,
        name="gate_up_sbuf",
    )
    sbm.open_scope()
    gate_tile_info = process_gate_up_projection(
        hidden=input_sb,
        output=gate_up_sb,
        params=params,
        dims=dims,
        sbm=sbm,
    )
    sbm.close_scope()

    # dealloc input_sb
    sbm.pop_heap()

    # ---------- Process down projection ----------
    # Allocate SBUF tile for down projection output
    if params.use_tkg_down_proj_column_tiling:
        down_sb = sbm.alloc_stack(
            (dims.T, dims.H_per_shard),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="down_sbuf",
        )
    else:
        down_sb = sbm.alloc_stack(
            (dims.H0, dims.H1_shard, dims.T),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="down_sbuf",
        )
    sbm.open_scope()
    process_down_projection(
        hidden=gate_up_sb,
        output=down_sb,
        params=params,
        dims=dims,
        gate_tile_info=gate_tile_info,
        sbm=sbm,
    )
    sbm.close_scope()

    # ---------- Return output ----------
    if not params.store_output_in_sbuf:
        if params.transposed_out:
            # Transposed output: down_sb is [H0, H1_shard, T], DMA to [H0, n_prgs, H1_shard, T]
            nc_size = dims.H1_shard * dims.T
            nisa.dma_copy(
                dst=output_tensor_hbm.reshape((dims.H0, dims.num_shards * nc_size))[
                    :, dims.shard_id * nc_size : (dims.shard_id + 1) * nc_size
                ],
                src=down_sb.reshape((dims.H0, nc_size)),
            )
            sbm.close_scope()
            return (
                [output_tensor_hbm, output_stored_add_tensor_hbm]
                if mlpp_store_fused_add(params)
                else [output_tensor_hbm]
            )
        else:
            # reshape to 2D tensor
            B, S, H = output_tensor_hbm.shape
            output_tensor_hbm = output_tensor_hbm.reshape((B * S, H))

            if params.use_tkg_down_proj_column_tiling:
                nisa.dma_copy(
                    dst=output_tensor_hbm.slice(
                        dim=1,
                        start=dims.shard_id * dims.H_per_shard,
                        end=(dims.shard_id + 1) * dims.H_per_shard,
                    ),
                    src=down_sb.slice(dim=1, start=0, end=dims.H_per_shard),
                )
            else:
                # Transpose output[H0, H1, T] to [T, H]
                output_nd = output_tensor_hbm.reshape((B * S, H))
                transpose_store_sbuf_copy(down_sb, output_nd, dims, io_dtype, sbm)

            # reshape back to 3D tensor
            output_tensor_hbm = output_tensor_hbm.reshape((B, S, H))
            sbm.close_scope()  # Close SBUF allocation scope

            return (
                [output_tensor_hbm, output_stored_add_tensor_hbm]
                if mlpp_store_fused_add(params)
                else [output_tensor_hbm]
            )

    else:
        sbm.close_scope()

        return [down_sb, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [down_sb]


def mlp_tkg(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
    sbm: Optional[BufferManager] = None,
) -> list[nl.NkiTensor]:
    """Wrapper that converts to NkiTensor, tiles along BxS, and calls _mlp_tkg_impl per tile."""

    # Dispatch to specialized Llama3-70B high-batch kernel for matching configs
    if _is_llama3_70b_specialized_config(params):
        return mlp_tkg_llama3_70b_high_batch(params, output_tensor_hbm, output_stored_add_tensor_hbm, sbm=sbm)

    # SBM setup
    if sbm is None:
        sbm = SbufManager(0, 200 * 1024, Logger("mlp_tkg"))
        sbm.set_name_prefix("mlp_")

    T = params.batch_size * params.sequence_len
    H = params.hidden_size

    # --- T-sharding: each LNC core processes T/lnc tokens with full H ---
    _, lnc, shard_id = get_verified_program_sharding_info("mlp_tkg", (0, 1))
    use_t_sharding = T >= BS_TILE_SIZE * lnc and lnc >= 2 and T % lnc == 0 and not params.shard_on_h_disabled

    if use_t_sharding:
        T_shard = T // lnc
        if params.input_in_sbuf:
            # SBUF input: [H0, T, H1]
            params.hidden_tensor = params.hidden_tensor.slice(
                dim=1, start=shard_id * T_shard, end=(shard_id + 1) * T_shard
            )
        elif params.transposed_in:
            # Transposed input: [H0, n_prgs, H1_shard, T]
            params.hidden_tensor = params.hidden_tensor.slice(
                dim=3, start=shard_id * T_shard, end=(shard_id + 1) * T_shard
            )
        else:
            # HBM input: [B, S, H] -> flatten to [T, H] then slice
            hidden = params.hidden_tensor.flatten_dims(start_dim=0, end_dim=1)
            params.hidden_tensor = hidden.slice(
                dim=0, start=shard_id * T_shard, end=(shard_id + 1) * T_shard
            ).expand_dim(dim=0)
        params.batch_size = 1
        params.sequence_len = T_shard
        params.shard_on_h_disabled = True
        T = T_shard

    # --- Default path: tile along BxS and H-shard across cores ---
    tile_size = min(BS_TILE_SIZE, T)

    # Pass full output tensors + T_offset so _mlp_tkg_impl writes to the correct position.

    original_hidden = params.hidden_tensor
    original_fused_add = params.fused_add_params.fused_add_tensor

    # Flatten hidden to 2D (T, H) for contiguous slicing
    if not params.input_in_sbuf:
        if params.transposed_in:
            hidden = original_hidden  # [H0, n_prgs, H1_shard, BxS] — no flatten needed
        else:
            hidden = original_hidden.flatten_dims(start_dim=0, end_dim=1)  # (B, S, H) -> #(T, H)
    else:
        hidden = original_hidden  # (H0, T, H1)

    fused_add = None
    if original_fused_add is not None:
        fused_add = original_fused_add.flatten_dims(start_dim=0, end_dim=1)  # (B, S, H) -> #(T, H)

    if not params.store_output_in_sbuf:
        if params.transposed_out:
            output_hbm_view = output_tensor_hbm  # 4D [H0, n_prgs, H1_shard, T]
        else:
            B, S, H = output_tensor_hbm.shape
            output_hbm_full = output_tensor_hbm.reshape((B * S, H))
            if use_t_sharding:
                output_hbm_view = output_hbm_full.slice(dim=0, start=shard_id * T_shard, end=(shard_id + 1) * T_shard)
            else:
                output_hbm_view = output_hbm_full

    for bxs_tile in TiledRange(T, tile_size):
        params.batch_size = 1
        params.sequence_len = bxs_tile.size

        # Slice hidden per tile
        if not params.input_in_sbuf:
            if params.transposed_in:
                # Transposed input: [H0, n_prgs, H1_shard, BxS] — slice along last dim
                params.hidden_tensor = hidden.slice(dim=3, start=bxs_tile.start_offset, end=bxs_tile.end_offset)
            else:
                params.hidden_tensor = hidden.slice(
                    dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset
                ).expand_dim(dim=0)
        else:
            params.hidden_tensor = hidden.slice(dim=1, start=bxs_tile.start_offset, end=bxs_tile.end_offset)

        if fused_add is not None:
            params.fused_add_params.fused_add_tensor = fused_add.slice(
                dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset
            ).expand_dim(dim=0)

        # Slice output HBM so _mlp_tkg_impl sees a tile-sized dst
        if not params.store_output_in_sbuf:
            if params.transposed_out:
                output_tile_hbm = output_tensor_hbm  # Pass full tensor; _mlp_tkg_impl writes to NC offset
            else:
                output_tile_hbm = output_hbm_view.slice(
                    dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset
                ).expand_dim(dim=0)
        else:
            output_tile_hbm = None

        prev_prefix = sbm.get_name_prefix()
        sbm.set_name_prefix(f"{prev_prefix}bxs_{bxs_tile.index}_")
        tile_result = _mlp_tkg_impl(params, output_tile_hbm, output_stored_add_tensor_hbm, sbm=sbm)
        sbm.set_name_prefix(prev_prefix)

    if params.store_output_in_sbuf:
        return tile_result
    # Return the full output tensors (reshape back to original shape)
    if params.transposed_out:
        output_tensors = [output_tensor_hbm]
    else:
        output_tensors = [output_hbm_full.reshape((B, S, H))]
    if mlpp_store_fused_add(params):
        output_tensors.append(output_stored_add_tensor_hbm)
    return output_tensors


def _is_llama3_70b_specialized_config(params: MLPParameters) -> bool:
    """Check if config matches the Llama3-70B static FP8 specialization constraints."""
    if params.hidden_size != 8192:
        return False
    if params.intermediate_size != 3584:
        return False
    if params.output_dtype != nl.bfloat16:
        return False
    if params.quant_params.quantization_type != QuantizationType.STATIC:
        return False
    if params.norm_params.normalization_type != NormType.RMS_NORM:
        return False
    if params.activation_fn != ActFnType.SiLU:
        return False
    if mlpp_has_fused_add(params):
        return False
    if params.fused_add_params.store_fused_add_result:
        return False
    if mlpp_has_projection_bias(params):
        return False
    if mlpp_has_normalization_bias(params):
        return False
    if params.skip_gate_proj:
        return False
    if params.store_output_in_sbuf:
        return False
    if params.input_in_sbuf:
        return False
    if params.transposed_in or params.transposed_out:
        return False
    if params.use_tkg_down_proj_optimized_layout:
        return False

    # Runtime guards for the dispatch kernel's hardcoded layout assumptions
    T = params.batch_size * params.sequence_len
    _, lnc, _ = get_verified_program_sharding_info("mlp_tkg", (0, 1))
    if params.shard_on_h_disabled:
        return False
    if lnc < 2 or T % lnc != 0:
        return False
    if T // lnc != BS_TILE_SIZE:
        return False
    return True
