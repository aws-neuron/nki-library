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

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

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
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .down_projection_mx_shard_I import (
    down_projection_mx_shard_I,
    load_broadcast_down_weight_scale_bias,
)
from .gate_up_projection_mx_shard_I import (
    gate_up_projection_mx_shard_I,
    load_gate_up_weight_scale_bias,
)

# Constants
DYNAMIC_LOOP_TOKEN_THRESHOLD = 256

# FIXME: add @nki.jit decorator to all sub-kernels when NKIFE-557 is resolved


@dataclass
class AllExpertMXDims(nl.NKIObject):
    """
    Dimension sizes and tiling constants for all-expert MX kernel.

    Stores extracted shapes from input tensors and derived tiling parameters.
    """

    T: int  # Total number of input tokens
    E_L: int  # Number of local experts
    I: int  # Intermediate dimension size
    H: int  # Hidden dimension size
    down_proj_tile_I: int  # Down projection intermediate dimension tile size
    pmax: int  # Partition max size
    n_prgs: int  # Number of programs (LNC shards)
    prg_id: int  # Current program ID

    def __post_init__(self):
        """Derive tiling constants from base dimensions."""
        self.num_tiles_in_T = div_ceil(self.T, self.pmax)
        self.n_T32_tiles = div_ceil(self.T, 32)
        self.n_H512_tiles = div_ceil(self.H, 512)
        self.tile_T = min(self.T, self.pmax)
        self.tile_H = self.H // self.n_H512_tiles // _q_width
        self.T32_H4 = self.pmax  # Always pad to 128 due to alignment constraints

        # Derived dimensions for _load_expert
        self.I_local = self.I // 2 if self.n_prgs > 1 else self.I
        self.n_I512_tiles = self.I_local // 512

    @staticmethod
    def extract_dims(
        input_tensor: nl.ndarray,
        gate_up_weights: nl.ndarray,
        down_weights: nl.ndarray,
        hidden_input_scale: nl.ndarray,
    ) -> "AllExpertMXDims":
        """
        Extract dimension sizes and tiling constants from input tensors.

        Args:
            input_tensor (nl.ndarray): Input hidden states tensor.
            gate_up_weights (nl.ndarray): [E_L, H, 2, I], Gate and up projection weights.
            down_weights (nl.ndarray): [E_L, I, H], Down projection weights.
            hidden_input_scale (nl.ndarray): Optional MX quantization scale for pre-quantized hidden_input.

        Returns:
            AllExpertMXDims: Dataclass containing all dimension sizes and tiling constants.
        """
        pmax = nl.tile_size.pmax

        # Extract T based on input location and quantization state
        if input_tensor.buffer == nl.sbuf:
            if hidden_input_scale is not None:
                # Quantized input shape: [16_H * 8_H, H/512, T]
                T = input_tensor.shape[-1]
            else:
                # Non-quantized input shape: [16_H * 8_H, T, 4_H * H/512]
                T = input_tensor.shape[1]
        else:
            # HBM input shape: [T, 4_H * H/512 * 16_H * 8_H]
            T, _ = input_tensor.shape

        # Extract other dimensions from weight shapes
        E_L = gate_up_weights.shape[0]
        I = gate_up_weights.shape[-1]
        H = down_weights.shape[-1]
        down_proj_tile_I = down_weights.shape[1]

        # Get LNC config
        _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx_shard_I", (0, 1))

        return AllExpertMXDims(
            T=T,
            E_L=E_L,
            I=I,
            H=H,
            down_proj_tile_I=down_proj_tile_I,
            pmax=pmax,
            n_prgs=n_prgs,
            prg_id=prg_id,
        )

    @staticmethod
    def validate_inputs(
        input_tensor: nl.ndarray,
        hidden_input_scale: nl.ndarray,
        expert_affinities_masked: nl.ndarray,
        dims: "AllExpertMXDims",
        lhs_rhs_swap: bool,
        output_in_sbuf: bool,
    ) -> None:
        """
        Validate input tensors and configuration for all-expert MX kernel.

        Args:
            input_tensor (nl.ndarray): Input hidden states tensor.
            hidden_input_scale (nl.ndarray): Optional MX quantization scale for pre-quantized hidden_input.
            expert_affinities_masked (nl.ndarray): Expert affinity scores.
            dims (AllExpertMXDims): Extracted dimension sizes.
            lhs_rhs_swap (bool): Whether to swap LHS/RHS in matmuls.
            output_in_sbuf (bool): Whether output should be in SBUF.

        Notes:
            - Raises AssertionError if any validation fails.
        """
        # Validate input dtype based on quantization state
        if hidden_input_scale is None:
            kernel_assert(
                input_tensor.dtype in SUPPORTED_QMX_INPUT_DTYPES,
                f"Expected input dtype in {SUPPORTED_QMX_INPUT_DTYPES}, got {input_tensor.dtype=}.",
            )
        else:
            kernel_assert(
                input_tensor.dtype in MX_DTYPES,
                f"Expected quantized input dtype in {MX_DTYPES}, got {input_tensor.dtype=}",
            )
            kernel_assert(
                hidden_input_scale.dtype == nl.uint8,
                f"Expected hidden_input_scale dtype = nl.uint8, got {hidden_input_scale.dtype=}",
            )

        kernel_assert(lhs_rhs_swap, "lhs_rhs_swap=False is not yet supported!")

        # Validate T alignment based on input state
        if hidden_input_scale is None:
            # Layout adapters only support T divisible by 32
            kernel_assert(
                dims.T % 32 == 0,
                f"Expected T divisible by 32, got T={dims.T}. "
                "To use T divisible by 4, provide prequantized input and hidden_input_scale.",
            )
        else:
            # T must be divisible by 4 for MatmultMx alignment
            kernel_assert(dims.T % 4 == 0, f"Expected T divisible by 4, got T={dims.T}")

        # Validate expert affinities shape
        kernel_assert(
            len(expert_affinities_masked.shape) in (2, 3),
            f"Expected 2D or 3D expert_affinities_masked, got {expert_affinities_masked.shape=}",
        )

        # Validate output location
        kernel_assert(
            not output_in_sbuf,
            f"All-expert MX kernel does not yet support SBUF output, got {output_in_sbuf=}",
        )


@dataclass
class AllExpertMXParams(nl.NKIObject):
    """
    Parameters for all-expert MX kernel extracted from MLPParameters.
    """

    # TODO: consolidate with MLPParameters rather than using a separate dataclass.

    input: nl.ndarray
    gate_up_weights: nl.ndarray
    down_weights: nl.ndarray
    output: nl.ndarray
    expert_affinities_masked: nl.ndarray
    gate_up_weights_scale: nl.ndarray
    down_weights_scale: nl.ndarray
    hidden_input_scale: nl.ndarray
    gate_up_weights_bias: nl.ndarray
    down_weights_bias: nl.ndarray
    expert_affinities_scaling_mode: ExpertAffinityScaleMode
    hidden_act_fn: ActFnType
    gate_clamp_lower_limit: float
    gate_clamp_upper_limit: float
    up_clamp_lower_limit: float
    up_clamp_upper_limit: float
    input_in_sbuf: bool
    output_in_sbuf: bool
    lhs_rhs_swap: bool
    activation_compute_dtype: nki.dtype = nl.bfloat16

    @staticmethod
    def from_mlp_params(
        mlp_params: MLPParameters,
        output: nl.ndarray,
        output_in_sbuf: bool = False,
        lhs_rhs_swap: bool = True,
        activation_compute_dtype=nl.bfloat16,
    ) -> "AllExpertMXParams":
        """
        Create AllExpertMXParams from MLPParameters.

        Args:
            mlp_params (MLPParameters): Source parameters.
            output (nl.ndarray): Output tensor.
            output_in_sbuf (bool): Whether output should be in SBUF.
            lhs_rhs_swap (bool): Whether to swap LHS/RHS in matmuls.
            activation_compute_dtype: Compute dtype for activations.

        Returns:
            AllExpertMXParams: Flattened parameters for this kernel.
        """
        return AllExpertMXParams(
            input=mlp_params.hidden_tensor,
            gate_up_weights=mlp_params.gate_proj_weights_tensor,
            down_weights=mlp_params.down_proj_weights_tensor,
            output=output,
            expert_affinities_masked=mlp_params.expert_params.expert_affinities,
            gate_up_weights_scale=mlp_params.quant_params.gate_w_scale,
            down_weights_scale=mlp_params.quant_params.down_w_scale,
            hidden_input_scale=mlp_params.hidden_input_scale,
            gate_up_weights_bias=(mlp_params.bias_params.gate_proj_bias_tensor if mlp_params.bias_params else None),
            down_weights_bias=(mlp_params.bias_params.down_proj_bias_tensor if mlp_params.bias_params else None),
            expert_affinities_scaling_mode=mlp_params.expert_params.expert_affinities_scaling_mode,
            hidden_act_fn=mlp_params.activation_fn,
            gate_clamp_lower_limit=mlp_params.gate_clamp_lower_limit,
            gate_clamp_upper_limit=mlp_params.gate_clamp_upper_limit,
            up_clamp_lower_limit=mlp_params.up_clamp_lower_limit,
            up_clamp_upper_limit=mlp_params.up_clamp_upper_limit,
            input_in_sbuf=mlp_params.input_in_sbuf,
            output_in_sbuf=output_in_sbuf,
            lhs_rhs_swap=lhs_rhs_swap,
            activation_compute_dtype=activation_compute_dtype,
        )


@dataclass
class ExpertWeightsSBUF(nl.NKIObject):
    """Expert weights, scales, and biases loaded in SBUF for one expert."""

    gate_weight_sb: nl.ndarray
    up_weight_sb: nl.ndarray
    down_weight_sb: nl.ndarray
    gate_weight_scale_sb: nl.ndarray
    up_weight_scale_sb: nl.ndarray
    down_weight_scale_sb: nl.ndarray
    gate_bias_sb: nl.ndarray
    up_bias_sb: nl.ndarray
    down_bias_sb: nl.ndarray


def _all_expert_moe_tkg_mx(
    mlp_params: MLPParameters,
    output: nl.ndarray,
    output_in_sbuf: bool = False,
    lhs_rhs_swap: bool = True,
    activation_compute_dtype: nki.dtype = nl.bfloat16,
    is_all_expert_dynamic: bool = False,
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
            - hidden_tensor: Input tensor in HBM [T, H] or SBUF [H0, H/512, T] (if pre-quantized)
            - hidden_input_scale: Optional MX quantization scale for pre-quantized input [H0, H/512, T]
            - input_in_sbuf: Whether input is in SBUF
        output (nl.ndarray): [min(T, 128), ⌈T/128⌉, H] in SBUF or [T, H] in HBM, Output tensor.
        output_in_sbuf (bool): Indicates desired output buffer location (SBUF or HBM).
        lhs_rhs_swap (bool): Indicates whether to swap LHS and RHS of gate and up projection matmuls.
        activation_compute_dtype: Compute dtype for activations.
        is_all_expert_dynamic: TODO: document experimental flag

    Returns:
        output (nl.ndarray): [T, H] in HBM or [min(T, 128), ⌈T/128⌉, H] in SBUF, Output tensor with MoE results.

    Notes:
        - More details on input & weight layout in doc `YFIQAmI1p2nr`
        - When mlp_params.hidden_input_scale is provided, input is expected to be pre-quantized in SBUF

    Pseudocode:
        # Step 1: Load and quantize input (skipped if hidden_input_scale provided)
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

    # Initialize parameters manager, extract dims, validate inputs
    params = AllExpertMXParams.from_mlp_params(
        mlp_params=mlp_params,
        output=output,
        output_in_sbuf=output_in_sbuf,
        lhs_rhs_swap=lhs_rhs_swap,
        activation_compute_dtype=activation_compute_dtype,
    )
    dims = AllExpertMXDims.extract_dims(
        params.input,
        params.gate_up_weights,
        params.down_weights,
        params.hidden_input_scale,
    )
    AllExpertMXDims.validate_inputs(
        params.input,
        params.hidden_input_scale,
        params.expert_affinities_masked,
        dims,
        params.lhs_rhs_swap,
        params.output_in_sbuf,
    )

    # Dispatch to expert MLP implementation
    if is_all_expert_dynamic:
        _all_expert_mx_dynamic(params=params, dims=dims)
    else:
        _all_expert_mx_static(params=params, dims=dims)

    return output


def _all_expert_mx_static(
    params: AllExpertMXParams,
    dims: AllExpertMXDims,
) -> nl.ndarray:
    """
    Static all-expert MoE computation without dynamic loop on chip (DLoC).

    Processes all experts sequentially, computing the full token batch for each expert
    before moving to the next. This is optimal when DLoC overhead exceeds benefits.

    Args:
        params (AllExpertMXParams): All kernel parameters including tensors and config.
        dims (AllExpertMXDims): Dimension information for tiling and sharding.

    Returns:
        nl.ndarray: Output tensor with MoE computation results.
    """

    # Step 1: Optional load + swizzle + QMX input
    if params.input_in_sbuf:
        if params.hidden_input_scale is None:
            input_quant_sb, input_scale_sb = _layout_adapter_qmx_sb(
                params.input, dims.T32_H4, dims.tile_H, dims.n_T32_tiles, dims.n_H512_tiles
            )
        else:
            # Input has been swizzled + MX quantized upstream
            input_quant_sb, input_scale_sb = params.input, params.hidden_input_scale
    else:
        input_quant_sb, input_scale_sb = _layout_adapter_qmx_hbm(
            params.input, dims.T32_H4, dims.tile_H, dims.n_T32_tiles, dims.n_H512_tiles
        )

    # Handle expert_affinities_masked based on its buffer type
    if params.expert_affinities_masked.buffer == nl.sbuf:
        kernel_assert(
            params.expert_affinities_masked.shape[0] <= dims.pmax,
            f"expected expert_affinities_masked shape [pmax_T, T/pmax, E_L] when T>pmax, got {params.expert_affinities_masked.shape=}",
        )
        expert_affinities_masked_sb = params.expert_affinities_masked
    else:
        # Load from HBM to SBUF
        expert_affinities_masked_shape = (
            (dims.T, dims.E_L) if dims.T <= dims.pmax else (dims.pmax, dims.T // dims.pmax, dims.E_L)
        )
        expert_affinities_masked_sb = nl.ndarray(
            expert_affinities_masked_shape, dtype=params.expert_affinities_masked.dtype, buffer=nl.sbuf
        )
        if dims.T <= dims.pmax:
            nisa.dma_copy(
                src=params.expert_affinities_masked[...],
                dst=expert_affinities_masked_sb[...],
            )
        else:
            for t128_tile_idx in nl.affine_range(dims.T // dims.tile_T):
                nisa.dma_copy(
                    src=params.expert_affinities_masked[nl.ds(dims.tile_T * t128_tile_idx, dims.tile_T), :],
                    dst=expert_affinities_masked_sb[:, t128_tile_idx, :],
                )

    # Step 2: Allocate output
    OUTPUT_SHAPE = (dims.tile_T, dims.num_tiles_in_T, dims.H)
    if params.output_in_sbuf:
        output_sb = params.output
    else:
        output_sb = nl.ndarray(OUTPUT_SHAPE, dtype=params.activation_compute_dtype, buffer=nl.sbuf)

    # Step 3: Compute expert MLPs sequentially
    for expert_idx in nl.sequential_range(dims.E_L):
        # Step 3.1: Load weights for this expert
        weights = _load_expert(params=params, dims=dims, expert_idx=expert_idx)

        # Step 3.2: Compute MLP for this expert
        _compute_expert_mlp(
            input_quant=input_quant_sb,
            input_scale=input_scale_sb,
            weights=weights,
            params=params,
            expert_affinities_masked=expert_affinities_masked_sb[...],
            output_sb=output_sb[...],
            output_hbm=params.output[...] if (not params.output_in_sbuf) else None,
            expert_idx=expert_idx,
            is_first_expert=(expert_idx == 0),
            is_last_expert=(expert_idx == dims.E_L - 1),
        )

    return params.output


def _all_expert_mx_dynamic(
    params: AllExpertMXParams,
    dims: AllExpertMXDims,
) -> nl.ndarray:
    """
    All-expert MoE computation with dynamic control flow (DLoC).

    Processes all experts sequentially, with tokens split into blocks for each expert. If any of the tokens in a block is
    routed to the expert, the block is computed. Otherwise, the block is skipped. This is optimal when T is very large.
    """

    # TODO: implement

    pass


def _is_dynamic_while(E_L: int, T: int) -> bool:
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


def _get_block_size(T: int, is_dynamic_while: bool) -> int:
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


def _layout_adapter_qmx_hbm(
    input: nl.ndarray,
    T32_H4: int,
    TILE_H: int,
    n_T32_tiles: int,
    n_H512_tiles: int,
) -> tuple[nl.ndarray, nl.ndarray]:
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
            nisa.nc_transpose(data=input_sb[:, t32_tile_idx, h512_tile_idx, :], dst=input_transposed_psum[...])
            nisa.tensor_copy(src=input_transposed_psum[...], dst=input_swizzled_sb[:, h512_tile_idx, t32_tile_idx, :])

    # View swizzled shape as [16_H * 8_H, H/512 * T * 4_H]
    T_H4 = n_T32_tiles * T32_H4
    T = T_H4 // _q_width
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


def _layout_adapter_qmx_sb(
    input_sb: nl.ndarray,
    T32_H4: int,
    TILE_H: int,
    n_T32_tiles: int,
    n_H512_tiles: int,
) -> tuple[nl.ndarray, nl.ndarray]:
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
    kernel_assert(False, "_layout_adapter_qmx_sb not yet migrated to new NKI FE")


def _load_expert(
    params: AllExpertMXParams,
    dims: AllExpertMXDims,
    expert_idx: int,
) -> ExpertWeightsSBUF:
    """
    Load gate, up, and down projection weight, scale, and bias tensors for one expert.

    When LNC=2, the loaded tensors are sharded on I dimension, except for down_bias.
    For down_bias, we broadcast to [tile_T, H]. When LNC=2, the first half of H is full
    of bias and second half of H is full of zeros on NC0; NC1 is the inverse.

    Args:
        params (AllExpertMXParams): All kernel parameters.
        dims (AllExpertMXDims): Dimension information.
        expert_idx (int): Expert index to load.

    Returns:
        ExpertWeightsSBUF: Expert weights, scales, and biases in SBUF.
    """

    # Load gate projection
    gate_weight_sb, gate_weight_scale_sb, gate_bias_sb = load_gate_up_weight_scale_bias(
        weight=params.gate_up_weights,
        scale=params.gate_up_weights_scale,
        bias=params.gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=GATE_FUSED_IDX,
        H=dims.H,
        I_local=dims.I_local,
        n_I512_tiles=dims.n_I512_tiles,
        prg_id=dims.prg_id,
    )

    # Load up projection
    up_weight_sb, up_weight_scale_sb, up_bias_sb = load_gate_up_weight_scale_bias(
        weight=params.gate_up_weights,
        scale=params.gate_up_weights_scale,
        bias=params.gate_up_weights_bias,
        expert_idx=expert_idx,
        gate_or_up_idx=UP_FUSED_IDX,
        H=dims.H,
        I_local=dims.I_local,
        n_I512_tiles=dims.n_I512_tiles,
        prg_id=dims.prg_id,
    )

    # Load down projection, broadcast down projection bias
    down_weight_sb, down_weight_scale_sb, down_bias_sb = load_broadcast_down_weight_scale_bias(
        weight=params.down_weights,
        scale=params.down_weights_scale,
        bias=params.down_weights_bias,
        expert_idx=expert_idx,
        H=dims.H,
        tile_I=dims.down_proj_tile_I,
        n_I512_tiles=dims.n_I512_tiles,
        tile_T=dims.tile_T,
        activation_compute_dtype=params.activation_compute_dtype,
        use_PE_bias_broadcast=False,  # FIXME: PE bias broadcast leads to inaccuracy
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


def _compute_expert_mlp(
    input_quant: nl.ndarray,
    input_scale: nl.ndarray,
    weights: ExpertWeightsSBUF,
    params: AllExpertMXParams,
    expert_affinities_masked: nl.ndarray,
    output_sb: nl.ndarray,
    output_hbm: nl.ndarray,
    expert_idx: int,
    is_first_expert: bool,
    is_last_expert: bool,
) -> nl.ndarray:
    """
    Compute expert MLP for one block of input.

    Args:
        input_quant (nl.ndarray): Quantized input tensor.
        input_scale (nl.ndarray): Input scale tensor.
        weights (ExpertWeightsSBUF): Expert weights, scales, and biases in SBUF.
        params (AllExpertMXParams): All kernel parameters.
        expert_affinities_masked (nl.ndarray): Masked expert affinities.
        output_sb (nl.ndarray): Output tensor in SBUF.
        output_hbm (nl.ndarray): Output tensor in HBM.
        expert_idx (int): Expert index.
        is_first_expert (bool): Whether the current expert is the first expert.
        is_last_expert (bool): Whether the current expert is the last expert.

    Returns:
        output_sb: Output tensor in SBUF.

    Notes:
        TODO[DLoC]: Explain what we are doing for dynamic loop.
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
        lhs_rhs_swap=params.lhs_rhs_swap,
        gate_clamp_upper_limit=params.gate_clamp_upper_limit,
        gate_clamp_lower_limit=params.gate_clamp_lower_limit,
        up_clamp_upper_limit=params.up_clamp_upper_limit,
        up_clamp_lower_limit=params.up_clamp_lower_limit,
        hidden_act_fn=params.hidden_act_fn,
        activation_compute_dtype=params.activation_compute_dtype,
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
        expert_affinities_scaling_mode=params.expert_affinities_scaling_mode,
        activation_compute_dtype=params.activation_compute_dtype,
        is_first_expert=is_first_expert,
        is_last_expert=is_last_expert,
    )

    return output_sb
