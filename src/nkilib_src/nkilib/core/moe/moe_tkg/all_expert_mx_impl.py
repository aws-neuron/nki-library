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

from typing import Optional

import nki.isa as nisa
import nki.language as nl

# MLP parameters
from ...mlp.mlp_parameters import MLPParameters
from ...mlp.mlp_tkg.projection_mx_constants import (
    GATE_FUSED_IDX,
    MX_DTYPES,
    MX_SCALE_DTYPE,
    SUPPORTED_QMX_INPUT_DTYPES,
    SUPPORTED_QMX_OUTPUT_DTYPES,
    UP_FUSED_IDX,
    _q_width,
)

# Common utils
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .down_projection_mx_shard_I import (
    down_projection_mx_shard_I,
    load_broadcast_down_weight_scale_bias,
)

# MLP TKG projection sub-kernels (I-sharding)
from .gate_up_mx_shard_I import (
    gate_up_projection_mx_shard_I,
    load_gate_up_weight_scale_bias,
)

# Constants
DYNAMIC_LOOP_TOKEN_THRESHOLD = 256

# FIXME: add @nki.jit decorator to all sub-kernels when NKIFE-557 is resolved


def get_is_dynamic_while(E_L, T):
    """
    Determine whether all-expert MX kernel should use dynamic loop on chip (DLoC).

    DLoC can improve performance with high concurrency (T), but may reduce performance
    for low concurrency due to DLoC overheads. Currently supports DLoC for E_L = 1 only,
    and uses DYNAMIC_LOOP_TOKEN_THRESHOLD as a heuristic.

    Args:
        E_L (int): Number of local experts.
        T (int): Number of tokens.

    Returns:
        bool: Whether to use dynamic loop (currently always False).
    """
    # TODO: uncomment below when we add support for dynamic loop
    # if E_L > 1:
    #     return False
    # else:
    #     return T >= DYNAMIC_LOOP_TOKEN_THRESHOLD

    return False


def get_block_size(T, is_dynamic_while):
    """
    Determine the block size to use with dynamic loop on chip (DLoC).

    When DLoC is not used, we use B=T.

    Args:
        T (int): Number of tokens.
        is_dynamic_while (bool): Whether dynamic loop is enabled.

    Returns:
        int: Block size (currently always T).
    """
    return T


def layout_adapter_qmx_hbm(input, T32_H4, TILE_H, n_T32_tiles, n_H512_tiles):
    """
    Load input from HBM, transform tensor into swizzled layout, and perform quantization to MXFP8.

    Args:
        input (nl.ndarray): [T, 4_H * H/512 * 16_H * 8_H], Input tensor in HBM.
        T32_H4 (int): Tile size for T dimension (32 * 4).
        TILE_H (int): Tile size for H dimension.
        n_T32_tiles (int): Number of T32 tiles.
        n_H512_tiles (int): Number of H512 tiles.

    Returns:
        output_quant_sb (nl.ndarray): [16_H * 8_H, H/512, T], Quantized output in SBUF (4_H packed in x4 dtype).
        output_scale_sb (nl.ndarray): [16_H * 8_H, H/512, T], Scales in SBUF (located in leading 4P of each SBUF quadrant).
    """

    # TODO: add output_x4_dtype as optional arg to API once NKI supports simulating w/ x4 dtypes as kernel args / if we want to use float8_e5m2_x4
    output_x4_dtype = nl.float8_e4m3fn_x4
    kernel_assert(
        output_x4_dtype in [nl.float8_e4m3fn_x4],
        f"Got {output_x4_dtype=}, expected output_x4_dtype in [nl.float8_e4m3fn_x4]",
    )

    # Load from HBM
    # [T/32 * 32_T, 4_H * H/512 * 16_H * 8_H]@HBM -> [T/32, T_32 * 4_H, H/512, 16_H * 8_H]@HBM
    input = input.reshape((n_T32_tiles, T32_H4, n_H512_tiles, TILE_H))
    # [32_T * 4_H, T/32, H/512, 16_H * 8_H]@SB
    input_sb = nl.ndarray((T32_H4, n_T32_tiles, n_H512_tiles, TILE_H), dtype=input.dtype, buffer=nl.sbuf)

    # Transpose T/32, 32_T * 4_H dims of HBM tensor during load, perform T/32 * H/512 transposes to achieve swizzled layout
    SWIZZLE_SHAPE = (TILE_H, n_H512_tiles, n_T32_tiles, T32_H4)
    input_swizzled_sb = nl.ndarray(SWIZZLE_SHAPE, dtype=input_sb.dtype, buffer=nl.sbuf)
    for t32_tile_idx in nl.affine_range(n_T32_tiles):
        nisa.dma_copy(
            src=input[t32_tile_idx, :, :, :],
            dst=input_sb[:, t32_tile_idx, :, :],
        )
        for h512_tile_idx in nl.affine_range(n_H512_tiles):
            input_transposed_psum = nl.ndarray((TILE_H, T32_H4), dtype=input_sb.dtype, buffer=nl.psum)
            input_transposed_psum[...] = nisa.nc_transpose(input_sb[:, t32_tile_idx, h512_tile_idx, :])
            input_swizzled_sb[:, h512_tile_idx, t32_tile_idx, :] = nisa.tensor_copy(input_transposed_psum[...])

    # View swizzled shape as [16_H * 8_H, H/512 * T * 4_H]
    T_H4 = n_T32_tiles * T32_H4
    T = T_H4 // 4
    input_swizzled_sb = input_swizzled_sb.reshape((TILE_H, n_H512_tiles * n_T32_tiles * T32_H4))

    # Allocate [16_H * 8_H, H/512 * T] QMX output buffers, 4_H is x4 packed in dtype
    out_qmx_flat_shape = (TILE_H, n_H512_tiles * T)
    output_quant_sb = nl.ndarray(out_qmx_flat_shape, dtype=output_x4_dtype, buffer=nl.sbuf)
    output_scale_sb = nl.ndarray(out_qmx_flat_shape, dtype=MX_SCALE_DTYPE, buffer=nl.sbuf)

    # Quantize to MXFP8
    nisa.quantize_mx(
        src=input_swizzled_sb,
        dst=output_quant_sb,
        dst_scale=output_scale_sb,
    )

    # Reshape outputs to [16_H * 8_H, H/512, T], 4_H is x4 packed in dtype
    out_qmx_3D_shape = (TILE_H, n_H512_tiles, T)
    output_quant_sb = output_quant_sb.reshape(out_qmx_3D_shape)
    output_scale_sb = output_scale_sb.reshape(out_qmx_3D_shape)

    return output_quant_sb, output_scale_sb


def layout_adapter_qmx_sb(input_sb, T32_H4, TILE_H, n_T32_tiles, n_H512_tiles):
    """
    Transform SB input tensor into swizzled layout and perform quantization to MXFP8.

    Args:
        input_sb (nl.ndarray): [16_H * 8_H, T, 4_H * H/512], Input tensor in SBUF.
        T32_H4 (int): Tile size for T dimension (32 * 4).
        TILE_H (int): Tile size for H dimension.
        n_T32_tiles (int): Number of T32 tiles.
        n_H512_tiles (int): Number of H512 tiles.

    Returns:
        output_quant_sb (nl.ndarray): [16_H * 8_H, H/512, T], Quantized output in SBUF (4_H packed in x4 dtype).
        output_scale_sb (nl.ndarray): [16_H * 8_H, H/512, T], Scales in SBUF (located in leading 4P of each SBUF quadrant).
    """
    # TODO: migrate SB layout adapter to the new FE when we migrate the top-level MK that hits this code path. See _pre_prod_kernels/mlp_tkg/expert_mlp_tkg_all_expert_mx_impl.py.
    kernel_assert(False, "layout_adapter_qmx_sb not yet migrated to new NKI FE")


def load_one_expert(
    gate_up_weights,
    down_weights,
    gate_up_weights_scale,
    down_weights_scale,
    gate_up_weights_bias,
    down_weights_bias,
    expert_idx,
    H,
    I,
    T,
    n_prgs,
    prg_id,
    activation_compute_dtype,
    use_PE_bias_broadcast,
):
    """
    Load gate, up, and down projection weight, scale, and bias tensors for one expert.

    When LNC=2, the loaded tensors are sharded on I dimension, except for down_bias.
    For down_bias, we broadcast to [tile_T, H]. When LNC=2, the first half of H is full
    of bias and second half of H is full of zeros on NC0; NC1 is the inverse.

    Args:
        gate_up_weights: Gate/up projection weights tensor.
        down_weights: Down projection weights tensor.
        gate_up_weights_scale: Gate/up projection scales tensor.
        down_weights_scale: Down projection scales tensor.
        gate_up_weights_bias: Gate/up projection bias tensor.
        down_weights_bias: Down projection bias tensor.
        expert_idx (int): Expert index to load.
        H (int): Hidden dimension.
        I (int): Intermediate dimension.
        T (int): Number of tokens.
        n_prgs (int): Number of programs (LNC).
        prg_id (int): Program ID.
        activation_compute_dtype: Compute dtype for activations.
        use_PE_bias_broadcast (bool): Whether to use PE bias broadcast.

    Returns:
        tuple: (gate_weight_sb, up_weight_sb, down_weight_sb, gate_weight_scale_sb,
                up_weight_scale_sb, down_weight_scale_sb, gate_bias_sb, up_bias_sb, down_bias_sb)
    """

    # Calculate constants
    I_local = I // 2 if n_prgs > 1 else I
    n_I512_tiles = I_local // 512
    tile_I = down_weights.shape[1]
    tile_T = min(T, nl.tile_size.pmax)

    # Load gate projection
    gate_weight_sb, gate_weight_scale_sb, gate_bias_sb = load_gate_up_weight_scale_bias(
        weight=gate_up_weights,
        scale=gate_up_weights_scale,
        bias=gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=GATE_FUSED_IDX,
        H=H,
        I_local=I_local,
        n_I512_tiles=n_I512_tiles,
        prg_id=prg_id,
    )

    # Load up projection
    up_weight_sb, up_weight_scale_sb, up_bias_sb = load_gate_up_weight_scale_bias(
        weight=gate_up_weights,
        scale=gate_up_weights_scale,
        bias=gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=UP_FUSED_IDX,
        H=H,
        I_local=I_local,
        n_I512_tiles=n_I512_tiles,
        prg_id=prg_id,
    )

    # Load down projection, broadcast down projection bias
    down_weight_sb, down_weight_scale_sb, down_bias_sb = load_broadcast_down_weight_scale_bias(
        weight=down_weights,
        scale=down_weights_scale,
        bias=down_weights_bias,
        expert_idx=expert_idx,
        H=H,
        tile_I=tile_I,
        n_I512_tiles=n_I512_tiles,
        tile_T=tile_T,
        activation_compute_dtype=activation_compute_dtype,
        use_PE_bias_broadcast=use_PE_bias_broadcast,
    )

    return (
        gate_weight_sb,
        up_weight_sb,
        down_weight_sb,
        gate_weight_scale_sb,
        up_weight_scale_sb,
        down_weight_scale_sb,
        gate_bias_sb,
        up_bias_sb,
        down_bias_sb,
    )


def compute_one_block(
    input_quant,
    input_scale,
    gate_weights_sb,
    up_weights_sb,
    down_weights_sb,
    gate_weights_scale_sb,
    up_weights_scale_sb,
    down_weights_scale_sb,
    gate_bias_sb,
    up_bias_sb,
    down_bias_sb,
    expert_affinities_masked,
    output_sb,
    output_hbm,
    expert_idx,
    # Placeholder DLoC args
    block_size,
    token_position_to_id,
    input_in_sbuf,
    # Placeholder argument
    output_in_sbuf,
    lhs_rhs_swap,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    hidden_act_fn,
    expert_affinities_scaling_mode,
    activation_compute_dtype,
    is_first_expert,
    is_last_expert,
):
    """
    Compute expert MLP for one block of input.

    Args:
        input_quant: Quantized input tensor.
        input_scale: Input scale tensor.
        gate_weights_sb: Gate projection weights in SBUF.
        up_weights_sb: Up projection weights in SBUF.
        down_weights_sb: Down projection weights in SBUF.
        gate_weights_scale_sb: Gate projection scales in SBUF.
        up_weights_scale_sb: Up projection scales in SBUF.
        down_weights_scale_sb: Down projection scales in SBUF.
        gate_bias_sb: Gate projection bias in SBUF.
        up_bias_sb: Up projection bias in SBUF.
        down_bias_sb: Down projection bias in SBUF.
        expert_affinities_masked: Masked expert affinities.
        output_sb: Output tensor in SBUF.
        output_hbm: Output tensor in HBM.
        expert_idx (int): Expert index.
        block_size: Block size for dynamic loop (placeholder).
        token_position_to_id: Token position mapping (placeholder).
        input_in_sbuf (bool): Whether input is in SBUF.
        output_in_sbuf (bool): Whether output is in SBUF.
        lhs_rhs_swap (bool): Whether to swap LHS/RHS in matmul.
        gate_clamp_upper_limit (float): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (float): Lower clamp limit for gate projection.
        up_clamp_upper_limit (float): Upper clamp limit for up projection.
        up_clamp_lower_limit (float): Lower clamp limit for up projection.
        hidden_act_fn: Activation function type.
        expert_affinities_scaling_mode: Expert affinity scaling mode.
        activation_compute_dtype: Compute dtype for activations.
        is_first_expert (bool): Whether this is the first expert.
        is_last_expert (bool): Whether this is the last expert.

    Returns:
        output_sb: Output tensor in SBUF.

    Notes:
        TODO[DLoC]: Explain what we are doing for dynamic loop.
    """

    # Step 1: Process inputs
    if input_in_sbuf:
        # TODO: move input handling logic here
        pass
    else:
        # TODO[DLoC]: Indirect load input_quant, input_scale, expert_affinities_masked
        pass

    # Step 2: Compute gate/up projection, projection clamping, activation function, and QMX
    act_quant_sb, act_scale_sb = gate_up_projection_mx_shard_I(
        input_quant_sb=input_quant,
        input_scale_sb=input_scale,
        gate_weight_sb=gate_weights_sb,
        up_weight_sb=up_weights_sb,
        gate_weight_scale_sb=gate_weights_scale_sb,
        up_weight_scale_sb=up_weights_scale_sb,
        gate_bias_sb=gate_bias_sb,
        up_bias_sb=up_bias_sb,
        lhs_rhs_swap=lhs_rhs_swap,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        hidden_act_fn=hidden_act_fn,
        activation_compute_dtype=activation_compute_dtype,
    )

    # Step 3: Compute down projection, expert affinity scaling, expert add, LNC reduction, and SB->HBM spill
    down_projection_mx_shard_I(
        act_sb=act_quant_sb[...],
        act_scale_sb=act_scale_sb[...],
        weight_sb=down_weights_sb,
        weight_scale_sb=down_weights_scale_sb,
        bias_sb=down_bias_sb,
        expert_affinities_masked_sb=expert_affinities_masked,
        expert_idx=expert_idx,
        out_sb=output_sb,
        out_hbm=output_hbm,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_compute_dtype=activation_compute_dtype,
        is_first_expert=is_first_expert,
        is_last_expert=is_last_expert,
    )

    return output_sb


def _all_expert_moe_tkg_mx(
    mlp_params: MLPParameters,
    output: nl.ndarray,
    input_scale: nl.ndarray = None,
    input_in_sbuf: bool = False,
    output_in_sbuf: bool = False,
    lhs_rhs_swap: bool = True,
    activation_compute_dtype=nl.bfloat16,
):
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
        mlp_params (MLPParameters): MLPParameters containing all input tensors and configuration.
        output (nl.ndarray): [min(T, 128), ⌈T/128⌉, H] in SBUF or [T, H] in HBM, Output tensor.
        input_scale (nl.ndarray, optional): Quantization scale for input. Expected in SBUF.
        input_in_sbuf (bool): Indicates whether inputs are in SBUF or HBM.
        output_in_sbuf (bool): Indicates desired output buffer location (SBUF or HBM).
        lhs_rhs_swap (bool): Indicates whether to swap LHS and RHS of gate and up projection matmuls.
        activation_compute_dtype: Compute dtype for activations.

    Returns:
        output (nl.ndarray): [T, H] in HBM or [min(T, 128), ⌈T/128⌉, H] in SBUF, Output tensor with MoE results.

    Notes:
        - More details on input & weight layout in doc `YFIQAmI1p2nr`

    Pseudocode:
        # Step 1: Load and quantize input
        input_quant, input_scale = layout_adapter(input)

        # Step 2: Process each expert sequentially
        for expert_idx in range(E_L):
            # Load expert weights
            gate_w, up_w, down_w = load_one_expert(expert_idx)

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
    """

    # Unpack from MLPParameters
    input = mlp_params.hidden_tensor
    gate_up_weights = mlp_params.gate_proj_weights_tensor
    down_weights = mlp_params.down_proj_weights_tensor
    gate_up_weights_scale = mlp_params.quant_params.gate_w_scale
    down_weights_scale = mlp_params.quant_params.down_w_scale
    gate_up_weights_bias = mlp_params.bias_params.gate_proj_bias_tensor if mlp_params.bias_params else None
    down_weights_bias = mlp_params.bias_params.down_proj_bias_tensor if mlp_params.bias_params else None
    expert_affinities_masked = mlp_params.expert_params.expert_affinities
    expert_index = mlp_params.expert_params.expert_index
    expert_affinities_scaling_mode = mlp_params.expert_params.expert_affinities_scaling_mode
    gate_clamp_upper_limit = mlp_params.gate_clamp_upper_limit
    gate_clamp_lower_limit = mlp_params.gate_clamp_lower_limit
    up_clamp_upper_limit = mlp_params.up_clamp_upper_limit
    up_clamp_lower_limit = mlp_params.up_clamp_lower_limit
    hidden_act_fn = mlp_params.activation_fn

    # Step 1: Check shapes and types, prep inputs
    if input_scale == None:
        kernel_assert(
            input.dtype in SUPPORTED_QMX_INPUT_DTYPES,
            f"Expected input dtype in {SUPPORTED_QMX_INPUT_DTYPES}, got {input.dtype=}.",
        )
    else:
        kernel_assert(input.dtype in MX_DTYPES, f"Expected quantized input dtype in {MX_DTYPES}, got {input.dtype=}")
        kernel_assert(input_scale.dtype == nl.uint8, f"Expected input_scale dtype = nl.uint8, got {input_scale.dtype=}")
    kernel_assert(lhs_rhs_swap, "lhs_rhs_swap=False is not yet supported!")

    # TODO: move constant extraction and shape validation to helper func / shape management object
    pmax = nl.tile_size.pmax
    if input_in_sbuf:
        if input_scale != None:
            # quantized input shape expected to be [16_H * 8_H, H/512, T]
            T = input.shape[-1]
        else:
            # input shape expected to be [16_H * 8_H, T, 4_H * H/512]
            T = input.shape[1]
    else:
        # [T, 4_H * H/512 * 16_H * 8_H]@HBM
        T, _ = input.shape

    # Shape extraction
    E_L = gate_up_weights.shape[0]
    I = gate_up_weights.shape[-1]
    H = down_weights.shape[-1]

    if not input_in_sbuf or input_scale == None:
        # Layout adapters only suppoort T divisible by 32
        kernel_assert(
            T % 32 == 0,
            f"Expected T divisible by 32, got {T=}. To use T divisible by 4, provide prequantized input and input_scale.",
        )
    else:
        # T must be divisible by 4 to meet MatmultMx alignment constraints
        kernel_assert(T % 4 == 0, f"Expected T divisible by 4, got {T=}")

    kernel_assert(
        len(expert_affinities_masked.shape) in (2, 3),
        f"expected 2D or 3D expert_affinities_masked, got {expert_affinities_masked.shape=}",
    )
    kernel_assert(not output_in_sbuf, f"all-expert MX kernel does not yet support SBUF output, got {output_in_sbuf=}")

    # LNC config
    _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx_shard_I", (0, 1))

    # Tiling strategy
    NUM_TILES_IN_T = div_ceil(T, pmax)
    n_T32_tiles = div_ceil(T, 32)
    n_H512_tiles = div_ceil(H, 512)
    TILE_T = min(T, pmax)
    TILE_H = H // n_H512_tiles // 4
    T32_H4 = pmax  # always pad to 128 because of above conditions

    # Step 2: Optional load + swizzle + QMX input
    if input_in_sbuf:
        if input_scale == None:
            input_quant_sb, input_scale_sb = layout_adapter_qmx_sb(input, T32_H4, TILE_H, n_T32_tiles, n_H512_tiles)
        else:
            # Input has been swizzled + MX quantized upstream
            input_quant_sb, input_scale_sb = input, input_scale
    else:
        input_quant_sb, input_scale_sb = layout_adapter_qmx_hbm(input, T32_H4, TILE_H, n_T32_tiles, n_H512_tiles)

    # Handle expert_affinities_masked based on its buffer type
    if expert_affinities_masked.buffer == nl.sbuf:
        kernel_assert(
            expert_affinities_masked.shape[0] <= pmax,
            f"expected expert_affinities_masked shape [128_T, T/128, E_L] when T>128, got {expert_affinities_masked.shape=}",
        )
        expert_affinities_masked_sb = expert_affinities_masked
    else:
        # Load from HBM to SBUF
        expert_affinities_masked_shape = (T, E_L) if T <= 128 else (128, T // 128, E_L)
        expert_affinities_masked_sb = nl.ndarray(
            expert_affinities_masked_shape, dtype=expert_affinities_masked.dtype, buffer=nl.sbuf
        )
        if T <= 128:
            nisa.dma_copy(
                src=expert_affinities_masked[...],
                dst=expert_affinities_masked_sb[...],
            )
        else:
            for t128_tile_idx in nl.affine_range(T // TILE_T):
                nisa.dma_copy(
                    src=expert_affinities_masked[nl.ds(TILE_T * t128_tile_idx, TILE_T), :],
                    dst=expert_affinities_masked_sb[:, t128_tile_idx, :],
                )

    # Step 3: Compute expert MLPs sequentially
    # Step 3.1: Allocate output
    OUTPUT_SHAPE = (TILE_T, NUM_TILES_IN_T, H)
    if output_in_sbuf:
        output_sb = output
    else:
        output_sb = nl.ndarray(OUTPUT_SHAPE, dtype=activation_compute_dtype, buffer=nl.sbuf)

    # Step 3.2: Blockwise compute strategy
    # NOTE: right now this section is a no-op
    is_dynamic_while = get_is_dynamic_while(E_L, T)
    block_size = get_block_size(T, is_dynamic_while)
    num_blocks = T // block_size
    num_static_blocks = 1
    num_dynamic_blocks = num_blocks - num_static_blocks
    for expert_idx in nl.sequential_range(E_L):
        # Step 3.2: Compute fused gate projection, up projection, activation, and MX quantization
        (
            gate_weight_sb,
            up_weight_sb,
            down_weight_sb,
            gate_weight_scale_sb,
            up_weight_scale_sb,
            down_weight_scale_sb,
            gate_bias_sb,
            up_bias_sb,
            down_bias_sb,
        ) = load_one_expert(
            gate_up_weights=gate_up_weights,
            down_weights=down_weights,
            gate_up_weights_scale=gate_up_weights_scale,
            down_weights_scale=down_weights_scale,
            gate_up_weights_bias=gate_up_weights_bias,
            down_weights_bias=down_weights_bias,
            expert_idx=expert_idx,
            H=H,
            I=I,
            T=T,
            n_prgs=n_prgs,
            prg_id=prg_id,
            activation_compute_dtype=activation_compute_dtype,
            # FIXME: PE bias broadcast is leads to inaccuracy, fix this and turn it on for all configs
            use_PE_bias_broadcast=False,
        )

        # Step 3.4: Compute static block. When we are not using DLoC, it is most efficient to use one block for the entire expert MLP.
        compute_one_block(
            input_quant=input_quant_sb,
            input_scale=input_scale_sb,
            gate_weights_sb=gate_weight_sb,
            up_weights_sb=up_weight_sb,
            down_weights_sb=down_weight_sb,
            gate_weights_scale_sb=gate_weight_scale_sb,
            up_weights_scale_sb=up_weight_scale_sb,
            down_weights_scale_sb=down_weight_scale_sb,
            gate_bias_sb=gate_bias_sb,
            up_bias_sb=up_bias_sb,
            down_bias_sb=down_bias_sb,
            expert_affinities_masked=expert_affinities_masked_sb[...],
            output_sb=output_sb[...],
            output_hbm=output[...] if (not output_in_sbuf) else None,
            expert_idx=expert_idx,
            # Placeholder DLoC args
            block_size=None,
            token_position_to_id=None,
            input_in_sbuf=input_in_sbuf,
            output_in_sbuf=output_in_sbuf,
            lhs_rhs_swap=lhs_rhs_swap,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            hidden_act_fn=hidden_act_fn,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            activation_compute_dtype=activation_compute_dtype,
            is_first_expert=(expert_idx == 0),
            is_last_expert=(expert_idx == E_L - 1),
        )

        # Step 3.5: Compute dynamic block(s)
        # TODO[DLoC]: Implement

    return output
