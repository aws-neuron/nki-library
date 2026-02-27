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

"""Parameter dataclasses, config initialization, and validation for all-expert MoE MX kernel."""

from dataclasses import dataclass

import nki
import nki.language as nl

from ...mlp.mlp_parameters import MLPParameters
from ...mlp.mlp_tkg.projection_mx_constants import (
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    MIN_MATMULT_MX_P_DIM,
    MX_DTYPES,
    SUPPORTED_QMX_INPUT_DTYPES,
    _q_width,
)
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

# Constants
_NUM_H4_FOLDS_PER_COLUMN = 32
_NUM_DYNAMIC_ALGO_STATIC_BLOCKS = 1
_NONZERO_WITH_COUNT_PAD_VAL = -1  # We pad indices with -1s to utilize DMA skipping


@dataclass
class AllExpertMXInputTensors(nl.NKIObject):
    """Input tensors for all-expert MX kernel."""

    hidden_input: nl.ndarray
    gate_up_weights: nl.ndarray
    down_weights: nl.ndarray
    output: nl.ndarray
    expert_affinities_masked: nl.ndarray
    gate_up_weights_scale: nl.ndarray
    down_weights_scale: nl.ndarray
    hidden_input_scale: nl.ndarray
    gate_up_weights_bias: nl.ndarray
    down_weights_bias: nl.ndarray


@dataclass
class AllExpertMXKernelConfig(nl.NKIObject):
    """Kernel-level hyperparameters for all-expert MX kernel."""

    expert_affinities_scaling_mode: ExpertAffinityScaleMode
    hidden_act_fn: ActFnType
    gate_clamp_lower_limit: float
    gate_clamp_upper_limit: float
    up_clamp_lower_limit: float
    up_clamp_upper_limit: float
    input_in_sbuf: bool
    output_in_sbuf: bool
    activation_compute_dtype: nki.dtype = nl.bfloat16


@dataclass
class AllExpertMXDimensions(nl.NKIObject):
    """Tensor dimensions and derived tiling/sharding constants for all-expert MX kernel."""

    # Base dimensions
    T: int  # Total number of input tokens
    E_L: int  # Number of local experts
    I: int  # Intermediate dimension size
    H: int  # Hidden dimension size

    def __post_init__(self):
        """Derive tiling strategy from tensor dimensions."""
        # Hardware and sharding constants
        self.pmax = nl.tile_size.pmax
        _, n_prgs, prg_id = get_verified_program_sharding_info("down_projection_mx_shard_I", (0, 1))
        self.n_prgs = n_prgs
        self.prg_id = prg_id

        # Shared tiling strategy
        self.n_tiles_in_T = div_ceil(self.T, self.pmax)
        self.n_T32_tiles = div_ceil(self.T, _NUM_H4_FOLDS_PER_COLUMN)
        self.n_H512_tiles = div_ceil(self.H, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
        self.tile_T = min(self.T, self.pmax)
        self.tile_H = self.H // self.n_H512_tiles // _q_width
        self.T32_H4 = self.pmax

        # LNC sharding strategy
        total_I512_tiles = div_ceil(self.I, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
        self.shard_on_I = total_I512_tiles >= self.n_prgs

        if self.shard_on_I:
            tiles_per_nc = div_ceil(total_I512_tiles, self.n_prgs) if self.n_prgs > 1 else total_I512_tiles
            self.n_I512_tiles_local = (
                min(tiles_per_nc, total_I512_tiles - self.prg_id * tiles_per_nc)
                if self.n_prgs > 1
                else total_I512_tiles
            )
            self.tile_start = self.prg_id * tiles_per_nc if self.n_prgs > 1 else 0
            self.I_offset = self.tile_start * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            self.I_local = min(self.I - self.I_offset, self.n_I512_tiles_local * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
            self.I_local_padded = div_ceil(self.I_local, MIN_MATMULT_MX_P_DIM) * MIN_MATMULT_MX_P_DIM
            self.shard_on_T = False
            self.T_local = self.T
            self.T_offset = 0
            self.t32_tile_offset = 0
        else:
            self.n_I512_tiles_local = total_I512_tiles
            self.tile_start = 0
            self.I_offset = 0
            self.I_local = self.I
            self.I_local_padded = div_ceil(self.I_local, MIN_MATMULT_MX_P_DIM) * MIN_MATMULT_MX_P_DIM

            T_local_candidate = self.T // self.n_prgs if self.n_prgs > 1 else self.T
            if (
                self.n_prgs > 1
                and self.T % self.n_prgs == 0
                and T_local_candidate >= _NUM_H4_FOLDS_PER_COLUMN  # HBM layout adapter requires T_local >= 32
            ):
                self.shard_on_T = True
                self.T_local = T_local_candidate
                self.T_offset = self.prg_id * self.T_local
                self.t32_tile_offset = self.T_offset // _NUM_H4_FOLDS_PER_COLUMN
                self.n_tiles_in_T = div_ceil(self.T_local, self.pmax)
                self.n_T32_tiles = div_ceil(self.T_local, _NUM_H4_FOLDS_PER_COLUMN)
                self.tile_T = min(self.T_local, self.pmax)
            else:
                self.shard_on_T = False
                self.T_local = self.T
                self.T_offset = 0
                self.t32_tile_offset = 0


@dataclass
class AllExpertMXDynamismConfig(nl.NKIObject):
    """Dynamic control flow config for all-expert MX kernel."""

    is_all_expert_dynamic: bool = False
    block_size: int = None

    def __post_init__(self):
        """Derive dynamic algorithm constants from block_size."""
        self.n_blocks = 0
        self.n_static_blocks = _NUM_DYNAMIC_ALGO_STATIC_BLOCKS
        self.n_dynamic_blocks = 0
        self.T_plus_1 = 0
        self.n_dynamic_blocks_plus_1 = 0
        self.blk_T_x4 = 0
        self.blk_tile_T_x4 = 0
        self.blk_tile_T = 0
        self.blk_n_T_x4_tiles = 0
        self.blk_n_T_tiles = 0
        self.blk_n_T32_tiles = 0

    def derive_from_dims(self, dims: AllExpertMXDimensions):
        """Derive dynamic tiling constants that depend on dimension parameters."""
        pmax = nl.tile_size.pmax

        # Block sizes
        self.n_blocks = dims.T // self.block_size
        self.n_dynamic_blocks = self.n_blocks - self.n_static_blocks

        # Padding
        self.T_plus_1 = dims.T + 1
        self.n_dynamic_blocks_plus_1 = self.n_dynamic_blocks + 1

        # Tiling
        self.blk_T_x4 = self.block_size * _q_width
        self.blk_tile_T_x4 = min(pmax, self.blk_T_x4)
        self.blk_tile_T = min(pmax, self.block_size)
        self.blk_n_T_x4_tiles = div_ceil(self.blk_T_x4, pmax)
        self.blk_n_T_tiles = div_ceil(self.block_size, pmax)
        self.blk_n_T32_tiles = div_ceil(self.block_size, _NUM_H4_FOLDS_PER_COLUMN)

        # Misc - use _NONZERO_WITH_COUNT_PAD_VAL for DMA skipping
        self.nonzero_with_count_pad_val = _NONZERO_WITH_COUNT_PAD_VAL


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


def init_all_expert_mx_configs(
    mlp_params: MLPParameters,
    output: nl.ndarray,
    activation_compute_dtype: nki.dtype = nl.bfloat16,
    is_all_expert_dynamic: bool = False,
    block_size: int = None,
) -> tuple[AllExpertMXInputTensors, AllExpertMXKernelConfig, AllExpertMXDimensions, AllExpertMXDynamismConfig]:
    """
    Initialize all sub-configs for the all-expert MX kernel from MLPParameters.

    Args:
        mlp_params (MLPParameters): Source parameters.
        output (nl.ndarray): Output tensor.
        activation_compute_dtype: Compute dtype for activations.
        is_all_expert_dynamic: Whether to use dynamic control flow.
        block_size: Block size for dynamic control flow algorithm.

    Returns:
        tuple: (AllExpertMXInputTensors, AllExpertMXKernelConfig, AllExpertMXDimensions, AllExpertMXDynamismConfig)
    """
    hidden_input = mlp_params.hidden_tensor
    hidden_input_scale = mlp_params.hidden_input_scale
    gate_up_weights = mlp_params.gate_proj_weights_tensor
    down_weights = mlp_params.down_proj_weights_tensor

    # Extract T based on input location and quantization state
    if hidden_input.buffer == nl.sbuf:
        if hidden_input_scale != None:
            T = hidden_input.shape[-1]
        else:
            T = hidden_input.shape[1]
    else:
        T, _ = hidden_input.shape

    # Extract dimensions from weight shapes
    E_L = gate_up_weights.shape[0]
    I = gate_up_weights.shape[-1]
    H = down_weights.shape[-1]

    # Set block_size to T if not in all-expert dynamic mode
    effective_block_size = block_size if block_size != None else T

    # Derive output_in_sbuf from output buffer location
    output_in_sbuf = output.buffer == nl.sbuf

    input_tensors = AllExpertMXInputTensors(
        hidden_input=hidden_input,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        output=output,
        expert_affinities_masked=mlp_params.expert_params.expert_affinities,
        gate_up_weights_scale=mlp_params.quant_params.gate_w_scale,
        down_weights_scale=mlp_params.quant_params.down_w_scale,
        hidden_input_scale=hidden_input_scale,
        gate_up_weights_bias=(mlp_params.bias_params.gate_proj_bias_tensor if mlp_params.bias_params else None),
        down_weights_bias=(mlp_params.bias_params.down_proj_bias_tensor if mlp_params.bias_params else None),
    )
    kernel_cfg = AllExpertMXKernelConfig(
        expert_affinities_scaling_mode=mlp_params.expert_params.expert_affinities_scaling_mode,
        hidden_act_fn=mlp_params.activation_fn,
        gate_clamp_lower_limit=mlp_params.gate_clamp_lower_limit,
        gate_clamp_upper_limit=mlp_params.gate_clamp_upper_limit,
        up_clamp_lower_limit=mlp_params.up_clamp_lower_limit,
        up_clamp_upper_limit=mlp_params.up_clamp_upper_limit,
        input_in_sbuf=mlp_params.input_in_sbuf,
        output_in_sbuf=output_in_sbuf,
        activation_compute_dtype=activation_compute_dtype,
    )
    dims = AllExpertMXDimensions(
        T=T,
        E_L=E_L,
        I=I,
        H=H,
    )
    dynamism_cfg = AllExpertMXDynamismConfig(
        is_all_expert_dynamic=is_all_expert_dynamic,
        block_size=effective_block_size,
    )
    dynamism_cfg.derive_from_dims(dims)

    return input_tensors, kernel_cfg, dims, dynamism_cfg


def validate_all_expert_mx_inputs(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig,
) -> None:
    """
    Validate input input_tensors and configuration for all-expert MX kernel.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.
    """

    # Validate input dtype based on quantization state
    if input_tensors.hidden_input_scale == None:
        kernel_assert(
            input_tensors.hidden_input.dtype in SUPPORTED_QMX_INPUT_DTYPES,
            f"Expected input dtype in {SUPPORTED_QMX_INPUT_DTYPES}, got {input_tensors.hidden_input.dtype=}.",
        )
    else:
        kernel_assert(
            input_tensors.hidden_input.dtype in MX_DTYPES,
            f"Expected quantized input dtype in {MX_DTYPES}, got {input_tensors.hidden_input.dtype=}",
        )
        kernel_assert(
            input_tensors.hidden_input_scale.dtype == nl.uint8,
            f"Expected hidden_input_scale dtype = nl.uint8, got {input_tensors.hidden_input_scale.dtype=}",
        )

    # Validate T size based on input state
    if input_tensors.hidden_input_scale == None:
        kernel_assert(
            dims.T % 32 == 0,
            f"Expected T divisible by 32, got T={dims.T}. "
            "To use T divisible by 4, provide prequantized input and hidden_input_scale.",
        )
        if dims.shard_on_T:
            kernel_assert(
                dims.T_local % 32 == 0,
                f"Expected T_local divisible by 32 for shard_on_T with HBM input, " f"got T_local={dims.T_local}.",
            )
    else:
        kernel_assert(dims.T % 4 == 0, f"Expected T divisible by 4, got T={dims.T}")
        if dims.shard_on_T:
            kernel_assert(
                dims.T_local % 4 == 0,
                f"Expected T_local divisible by 4 for shard_on_T with pre-quantized input, "
                f"got T_local={dims.T_local}.",
            )

    # Validate expert affinities shape
    kernel_assert(
        len(input_tensors.expert_affinities_masked.shape) in (2, 3),
        f"Expected 2D or 3D expert_affinities_masked, got {input_tensors.expert_affinities_masked.shape=}",
    )

    # Validate output location
    kernel_assert(
        not kernel_cfg.output_in_sbuf,
        f"All-expert MX kernel does not yet support SBUF output, got {kernel_cfg.output_in_sbuf=}",
    )

    # Algorithm-specific constraints
    if dynamism_cfg.is_all_expert_dynamic:
        kernel_assert(
            input_tensors.expert_affinities_masked.buffer != nl.sbuf,
            f"Expected expert_affinities_masked in HBM, got {input_tensors.expert_affinities_masked.buffer=}",
        )
        kernel_assert(
            dims.E_L == 1,
            f"All-expert MX kernel does not support E_L>1 with is_all_expert_dynamic=True, but got {dims.E_L=}",
        )
        kernel_assert(
            dynamism_cfg.block_size != None and _is_valid_block_size(dims.T, dynamism_cfg.block_size),
            f"Invalid block_size: expected (1) nonzero block_size (2) block_size that evenly divides T, (3) block_size at most T/2, "
            f"and (4) block_size<32 and divisible by 8, block_size<128 and divisible by 32, or block_size divisible by 128; "
            f"but got {dynamism_cfg.block_size=}, {dims.T=}",
        )


def _is_valid_block_size(T, block_size):
    """
    Helper function to validate that block_size is valid for a given T. Block size is expected to:
    - Be nonzero
    - Evenly divide T
    - Be at max T/2, resulting in at least 2 blocks
    - Be one of <32 and divisible by 8, <128 and divisible by 32, or divisible by 128
    """
    if block_size == 0:
        return False
    if T % block_size != 0:
        return False
    if block_size > T // 2:
        return False
    if block_size < 32:
        return block_size % 8 == 0
    elif block_size < 128:
        return block_size % 32 == 0
    return block_size % 128 == 0
