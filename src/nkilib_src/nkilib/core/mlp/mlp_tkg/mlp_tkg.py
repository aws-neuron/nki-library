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

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.logging import get_logger
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_fused_add,
    mlpp_has_normalization,
    mlpp_store_fused_add,
)
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import process_gate_up_projection
from .mlp_tkg_utils import (
    convert_weight_scale_params_to_views,
    input_fused_add,
    input_norm_load,
    transpose_store,
)


def mlp_tkg(
    params: MLPParameters,
    output_tensor_hbm: nl.ndarray,
    output_stored_add_tensor_hbm: nl.ndarray,
) -> list[nl.ndarray]:
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
        output_tensor_hbm (nl.ndarray): [B, S, H], Output tensor in HBM
        output_stored_add_tensor_hbm (nl.ndarray): [B, S, H], Optional fused add result storage

    Returns:
        list[nl.ndarray]: List containing:
            - output_tensor_hbm or down_sb: MLP output tensor
            - output_stored_add_tensor_hbm: (optional) Fused add result if store_fused_add_result=True

    Notes:
        - Supports RMSNorm and LayerNorm normalization
        - Supports static and row-wise quantization
        - Uses SbufManager for memory allocation
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

    sbm = SbufManager(0, 200 * 1024, get_logger("mlp_tkg"))
    sbm.open_scope()  # Start SBUF allocation scope

    # ---------------- Fused Add ----------------
    # Apply fused add if present (hidden + attention output)
    hidden = params.hidden_tensor
    if mlpp_has_fused_add(params):
        if not params.fused_add_params.store_fused_add_result:
            H1_dim = dims.H1 if mlpp_has_normalization(params) else dims.H1_shard
            fused_add_output_sb = sbm.alloc_heap(
                (dims.H0, dims.T, H1_dim),
                dtype=io_dtype,
                buffer=nl.sbuf,
                name="fused_add_output_sb",
            )
            fused_add_output = fused_add_output_sb
        else:
            fused_add_output = output_stored_add_tensor_hbm

        input_fused_add(
            input=hidden,
            fused_add_tensor=params.fused_add_params.fused_add_tensor,
            fused_output=fused_add_output,
            normtype=params.norm_params.normalization_type,
            store_fused_add_result=params.fused_add_params.store_fused_add_result,
            sbm=sbm,
            dims=dims,
        )
        hidden = fused_add_output  # Use fused result as hidden input

    # ---------------- Norm / Input Load ----------------
    if mlpp_has_normalization(params) or (hidden.buffer != nl.sbuf):
        # Allocate heap tile for normalized or loaded input
        input_sb = sbm.alloc_heap(
            (dims.H0, dims.T, dims.H1_shard),
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="input_sbuf",
        )
        input_norm_load(hidden, input_sb, params, dims, sbm)  # Norm or direct load

        if hidden.buffer == nl.sbuf:
            sbm.pop_heap()  # dealloc fused_add_output_sb
    else:
        input_sb = hidden

    # Convert to view when required (TODO: make this consistent)
    convert_weight_scale_params_to_views(params)

    # ---------- Process gate/up projection, silu, gate/up multiplication ----------
    # Allocate SBUF tile for gate/up projection output
    gate_up_sb = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
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
        # reshape to 2D tensor
        B, S, H = output_tensor_hbm.shape
        output_tensor_hbm = output_tensor_hbm.reshape((B * S, H))

        if params.use_tkg_down_proj_column_tiling:
            nisa.dma_copy(
                dst=output_tensor_hbm[
                    :,
                    nl.ds(
                        dims.shard_id * dims.H_per_shard,
                        dims.H_per_shard,
                    ),
                ],
                src=down_sb[:, 0 : dims.H_per_shard],
            )
        else:
            # Transpose output[H0, H1, T] to [T, H]
            transpose_store(down_sb, output_tensor_hbm, dims, io_dtype, sbm)

        # reshape back to 3D tensor
        output_tensor_hbm = output_tensor_hbm.reshape((B, S, H))
        sbm.close_scope()  # Close SBUF allocation scope

        return (
            [output_tensor_hbm, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [output_tensor_hbm]
        )

    else:
        sbm.close_scope()  # Close SBUF allocation scope

        return [down_sb, output_stored_add_tensor_hbm] if mlpp_store_fused_add(params) else [down_sb]
