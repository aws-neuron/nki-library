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
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import sizeinbytes
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode, MoEAllToAllVStrategy, MoELNCShardingStrategy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .mlp_parameters import MLPParameters
from .projection_mx_constants import (
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    MIN_MATMULT_MX_P_DIM,
    MX_PACKED_DTYPES,
    MX_SCALE_DTYPE,
    MX_UNPACKED_DTYPES,
    SUPPORTED_QMX_INPUT_DTYPES,
    _q_width,
)

# Constants
NUM_H4_FOLDS_PER_COLUMN = 32
NUM_DYNAMIC_ALGO_STATIC_BLOCKS = 1
NONZERO_WITH_COUNT_PAD_VAL = -1  # We pad indices with -1s to utilize DMA skipping
NONZERO_WITH_COUNT_P_STRIDE = (
    32  # Partition stride for parallel nonzero_with_count (stride 32 for SBUF access compatibility)
)
NONZERO_WITH_COUNT_EXPERT_GROUP_SIZE = nl.tile_size.pmax // NONZERO_WITH_COUNT_P_STRIDE  # 4 experts per call
UINT8_TP_VIEW_DTYPE = nl.float8_e5m2  # Used with bitcast for uint8 transposes, since PE does not support uint8 input
FP8X4_TP_VIEW_DTYPE = (
    nl.float32
)  # Used with bitcast for float8_x4 transposes, since PE does not support float8_x4 input

# Dummy MX scale value: uint32 representation of 4 × uint8(127) = 0x7F7F7F7F
_DUMMY_SCALE_U32 = 2139062143

# Dtype packing conversions
FP8_PER_BF16 = 2
FP8_PER_FP32 = 4
FP8_PER_INT32 = 4
BF16_PER_INT32 = 2
FP8_PER_FP8X4 = 4
BF16_PER_FP32 = 2

# Supported sharding strategies for all-expert MX kernel
SUPPORTED_MOE_SHARDING_STRATEGIES = [
    MoELNCShardingStrategy.NO_SHARD,
    MoELNCShardingStrategy.SHARD_I,
    MoELNCShardingStrategy.SHARD_T,
]


@dataclass
class AllExpertMXInputTensors(nl.NKIObject):
    """Input tensors for all-expert MX kernel."""

    hidden_input: nl.NkiTensor
    gate_up_weights: nl.NkiTensor
    down_weights: nl.NkiTensor
    output: nl.NkiTensor
    expert_affinities_masked: nl.NkiTensor
    gate_up_weights_scale: nl.NkiTensor
    down_weights_scale: nl.NkiTensor
    hidden_input_scale: nl.NkiTensor
    gate_up_weights_bias: nl.NkiTensor
    down_weights_bias: nl.NkiTensor
    # STATIC_MX: per-tensor FP8 dequant scales for activations (float32)
    gate_up_in_scale: nl.NkiTensor = None  # [E_L, 1] or [1, 1]
    down_in_scale: nl.NkiTensor = None  # [E_L, 1] or [1, 1]
    input_dequant_scale: nl.NkiTensor = None  # [pmax, 1] in SBUF, broadcast input dequant scale


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
    expert_affinities_dtype: nki.dtype = nl.bfloat16
    is_static_quant: bool = False
    is_row_quant: bool = False


@dataclass
class AllExpertMXDimensions(nl.NKIObject):
    """Tensor dimensions and derived tiling/sharding constants for all-expert MX kernel."""

    # Base dimensions
    T: int  # Total number of input tokens
    E_L: int  # Number of local experts
    I: int  # Intermediate dimension size
    H: int  # Hidden dimension size
    H_concat: int  # Size of dim1 of input tensor when [input_hidden, input_scale, expert_affinities, token_idx] is concatenated
    sharding_strategy: MoELNCShardingStrategy = None

    def __post_init__(self):
        """Derive tiling strategy from tensor dimensions."""
        # Hardware and sharding constants
        self.pmax = nl.tile_size.pmax
        _, n_prgs, prg_id = get_verified_program_sharding_info("_all_expert_moe_tkg_mx", (0, 1))
        self.n_prgs = n_prgs
        self.prg_id = prg_id

        # Pad T to the next multiple of 4 to satisfy nc_matmul_mx even free-dim constraint.
        # T_physical retains the original (unpadded) token count for output writes.
        self.T_physical = self.T
        self.T = div_ceil(self.T, _q_width) * _q_width

        # Shared tiling strategy
        self.n_tiles_in_T = div_ceil(self.T, self.pmax)
        self.n_T32_tiles = div_ceil(self.T, NUM_H4_FOLDS_PER_COLUMN)
        self.n_H512_tiles = div_ceil(self.H, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
        self.tile_T = min(self.T, self.pmax)
        self.tile_H = self.H // self.n_H512_tiles // _q_width
        self.T32_H4 = self.pmax

        # LNC sharding strategy decision
        total_I512_tiles = div_ceil(self.I, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
        can_shard_on_I = total_I512_tiles >= self.n_prgs
        T_local_candidate = self.T // self.n_prgs if self.n_prgs > 1 else self.T
        can_shard_on_T = (
            self.n_prgs > 1
            and self.T % self.n_prgs == 0
            and T_local_candidate >= NUM_H4_FOLDS_PER_COLUMN  # HBM layout adapter requires T_local >= 32
            and self.T_physical % _q_width == 0  # not allow shard on T for T_physical not divisible by 4
        )

        # Determine sharding strategy (I-sharding takes priority over T-sharding)
        if self.sharding_strategy != None:
            # Use user specified sharding strategy is already specified.
            pass
        elif can_shard_on_I:
            self.sharding_strategy = MoELNCShardingStrategy.SHARD_I
        elif can_shard_on_T:
            self.sharding_strategy = MoELNCShardingStrategy.SHARD_T
        else:
            self.sharding_strategy = MoELNCShardingStrategy.NO_SHARD

        # Apply configuration based on sharding strategy
        if self.sharding_strategy == MoELNCShardingStrategy.SHARD_I:
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
            self.T_local = self.T
            self.T_offset = 0
            self.t32_tile_offset = 0
        elif self.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
            self.n_I512_tiles_local = total_I512_tiles
            self.tile_start = 0
            self.I_offset = 0
            self.I_local = self.I
            self.I_local_padded = div_ceil(self.I_local, MIN_MATMULT_MX_P_DIM) * MIN_MATMULT_MX_P_DIM
            self.T_local = T_local_candidate
            self.T_offset = self.prg_id * self.T_local
            self.t32_tile_offset = self.T_offset // NUM_H4_FOLDS_PER_COLUMN
            self.n_tiles_in_T = div_ceil(self.T_local, self.pmax)
            self.n_T32_tiles = div_ceil(self.T_local, NUM_H4_FOLDS_PER_COLUMN)
            self.tile_T = min(self.T_local, self.pmax)
        else:  # NO_SHARD or SHARD_E
            self.n_I512_tiles_local = total_I512_tiles
            self.tile_start = 0
            self.I_offset = 0
            self.I_local = self.I
            self.I_local_padded = div_ceil(self.I_local, MIN_MATMULT_MX_P_DIM) * MIN_MATMULT_MX_P_DIM
            self.T_local = self.T
            self.T_offset = 0
            self.t32_tile_offset = 0


@dataclass
class AllExpertMXDynamismConfig(nl.NKIObject):
    """Dynamic control flow config for all-expert MX kernel."""

    is_all_expert_dynamic: bool = False
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED
    block_size: int = None

    def __post_init__(self):
        """Derive dynamic algorithm constants from block_size."""
        self.n_blocks = 0
        self.n_static_blocks = NUM_DYNAMIC_ALGO_STATIC_BLOCKS
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
        self.blk_n_T32_tiles = div_ceil(self.block_size, NUM_H4_FOLDS_PER_COLUMN)

        # Misc - use NONZERO_WITH_COUNT_PAD_VAL for DMA skipping
        self.nonzero_with_count_pad_val = NONZERO_WITH_COUNT_PAD_VAL


@dataclass
class ExpertWeightsSBUF(nl.NKIObject):
    """Expert weights, scales, and biases loaded in SBUF for one expert."""

    gate_weight_sb: nl.NkiTensor
    up_weight_sb: nl.NkiTensor
    down_weight_sb: nl.NkiTensor
    gate_weight_scale_sb: nl.NkiTensor
    up_weight_scale_sb: nl.NkiTensor
    down_weight_scale_sb: nl.NkiTensor
    gate_bias_sb: nl.NkiTensor
    up_bias_sb: nl.NkiTensor
    down_bias_sb: nl.NkiTensor
    # STATIC_MX: per-expert combined dequant scales (input_scale * weight_scale)
    gate_dequant_scale_sb: nl.NkiTensor = None
    up_dequant_scale_sb: nl.NkiTensor = None
    down_dequant_scale_sb: nl.NkiTensor = None
    dummy_scale_tile_sb: nl.NkiTensor = None  # SW quant: shared 2D [128, F] dummy scale (all-127)


def alloc_dummy_scale_tile(_pmax=128, free_dim=512):
    """Allocate a 2D dummy MX scale tile [_pmax, free_dim] (all-127, i.e. scale=1.0).

    SW quant path uses this shared 2D tile in place of the normal per-tile 3D scale
    [P, n_tiles, F]. Matmul callers index it as [:, :slice] instead of [:, tile_idx, slice].

    Internally allocates [_pmax, free_dim // 4] uint32, memsets to 0x7F7F7F7F, and
    returns a uint8 view of shape [_pmax, free_dim].

    Args:
        _pmax (int): Partition dimension (default 128).
        free_dim (int): Free dimension of the returned tile (default 512).

    Returns:
        nl.NkiTensor: uint8 tensor of shape [_pmax, free_dim].
    """
    n_u32 = free_dim // _q_width
    n_part = _pmax
    u32_buf = nl.ndarray((n_part, n_u32), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(dst=u32_buf, value=_DUMMY_SCALE_U32)
    return u32_buf.view(nl.uint8)


def init_all_expert_mx_configs(
    mlp_params: MLPParameters,
    output: nl.NkiTensor,
    activation_compute_dtype: nki.dtype = nl.bfloat16,
    sharding_strategy: MoELNCShardingStrategy = None,
) -> tuple[AllExpertMXInputTensors, AllExpertMXKernelConfig, AllExpertMXDimensions, AllExpertMXDynamismConfig]:
    """
    Initialize all sub-configs for the all-expert MX kernel from MLPParameters.

    Args:
        mlp_params (MLPParameters): Source parameters.
        output (nl.NkiTensor): Output tensor.
        activation_compute_dtype: Compute dtype for activations.

    Returns:
        tuple: (AllExpertMXInputTensors, AllExpertMXKernelConfig, AllExpertMXDimensions, AllExpertMXDynamismConfig)
    """
    hidden_input = mlp_params.hidden_tensor
    hidden_input_scale = mlp_params.hidden_input_scale
    gate_up_weights = mlp_params.gate_proj_weights_tensor
    down_weights = mlp_params.down_proj_weights_tensor

    # Extract T based on input location and quantization state
    if hidden_input_scale != None:
        # Pre-quantized input (SBUF or HBM): T is the last dimension
        T = hidden_input.shape[-1]
    elif hidden_input.buffer == nl.sbuf:
        T = hidden_input.shape[1]
    else:
        T, _ = hidden_input.shape

    # Extract dimensions from weight shapes
    E_L = gate_up_weights.shape[0]
    I = gate_up_weights.shape[-1]
    H = down_weights.shape[-1]
    H_concat = (
        H if mlp_params.expert_params.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED else hidden_input.shape[-1]
    )

    # Set block_size to T if not in all-expert dynamic mode
    effective_block_size = mlp_params.expert_params.block_size if mlp_params.expert_params.block_size != None else T

    # Derive output_in_sbuf from output buffer location
    output_in_sbuf = output.buffer == nl.sbuf

    is_static_quant = mlp_params.quant_params.is_quant_static_mx()
    is_row_quant = mlp_params.quant_params.is_quant_row_mx()

    # When using all_to_all_v, affinity dtype is hardcoded to bf16
    expert_affinities_dtype = (
        mlp_params.expert_params.expert_affinities.dtype
        if mlp_params.expert_params.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED
        else nl.bfloat16
    )

    input_tensors = AllExpertMXInputTensors(
        hidden_input=hidden_input,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        output=output,
        expert_affinities_masked=mlp_params.expert_params.expert_affinities,
        gate_up_weights_scale=mlp_params.quant_params.gate_w_scale,
        down_weights_scale=mlp_params.quant_params.down_w_scale,
        hidden_input_scale=hidden_input_scale,
        gate_up_weights_bias=mlp_params.bias_params.gate_proj_bias_tensor if mlp_params.bias_params else None,
        down_weights_bias=mlp_params.bias_params.down_proj_bias_tensor if mlp_params.bias_params else None,
        gate_up_in_scale=mlp_params.quant_params.gate_up_in_scale if is_static_quant else None,
        down_in_scale=mlp_params.quant_params.down_in_scale if is_static_quant else None,
        input_dequant_scale=mlp_params.input_dequant_scale,
    )
    kernel_cfg = AllExpertMXKernelConfig(
        expert_affinities_scaling_mode=mlp_params.expert_params.expert_affinities_scaling_mode,
        hidden_act_fn=mlp_params.activation_fn,
        gate_clamp_lower_limit=mlp_params.gate_clamp_lower_limit,
        gate_clamp_upper_limit=mlp_params.gate_clamp_upper_limit,
        up_clamp_lower_limit=mlp_params.up_clamp_lower_limit,
        up_clamp_upper_limit=mlp_params.up_clamp_upper_limit,
        # input_in_sbuf=True means "input is already in quantized layout, skip _layout_adapter_qmx_hbm".
        # Pre-quantized HBM input (hidden_input_scale != None) also skips the adapter.
        input_in_sbuf=mlp_params.input_in_sbuf or mlp_params.hidden_input_scale is not None,
        output_in_sbuf=output_in_sbuf,
        activation_compute_dtype=activation_compute_dtype,
        expert_affinities_dtype=expert_affinities_dtype,
        is_static_quant=is_static_quant,
        is_row_quant=is_row_quant,
    )
    dims = AllExpertMXDimensions(
        T=T,
        E_L=E_L,
        I=I,
        H=H,
        H_concat=H_concat,
        sharding_strategy=sharding_strategy,
    )
    dynamism_cfg = AllExpertMXDynamismConfig(
        is_all_expert_dynamic=mlp_params.expert_params.is_all_expert_dynamic,
        all_to_all_v_strategy=mlp_params.expert_params.all_to_all_v_strategy,
        block_size=effective_block_size,
    )
    dynamism_cfg.derive_from_dims(dims)

    return input_tensors, kernel_cfg, dims, dynamism_cfg


# Validation helpers


def validate_all_expert_mx_inputs(
    input_tensors: AllExpertMXInputTensors,
    kernel_cfg: AllExpertMXKernelConfig,
    dims: AllExpertMXDimensions,
    dynamism_cfg: AllExpertMXDynamismConfig,
) -> None:
    """
    Validate input tensors and configuration for all-expert MX kernel.

    Args:
        input_tensors (AllExpertMXInputTensors): Tensor parameters.
        kernel_cfg (AllExpertMXKernelConfig): Scalar parameters.
        dims (AllExpertMXDimensions): Dimension parameters.
        dynamism_cfg (AllExpertMXDynamismConfig): Dynamism parameters.

    Returns:
        None. Raises AssertionError via kernel_assert if validation fails.
    """

    # Validate input dtype based on quantization state
    # Not pre-quantized, no A2A-v
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED and input_tensors.hidden_input_scale == None:
        kernel_assert(
            input_tensors.hidden_input.dtype in SUPPORTED_QMX_INPUT_DTYPES,
            f"Expected input dtype in {SUPPORTED_QMX_INPUT_DTYPES}, got {input_tensors.hidden_input.dtype=}.",
        )
    # Pre-quantized, no A2A-v
    elif dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
        kernel_assert(
            input_tensors.hidden_input.dtype in MX_PACKED_DTYPES,
            f"Expected quantized input dtype in {MX_PACKED_DTYPES}, got {input_tensors.hidden_input.dtype=}",
        )
        kernel_assert(
            input_tensors.hidden_input_scale.dtype == MX_SCALE_DTYPE,
            f"Expected hidden_input_scale.dtype={MX_SCALE_DTYPE} with all_to_all_v_strategy=DISABLED, got {input_tensors.hidden_input_scale.dtype=}",
        )
    # A2A-v must have pre-quantized hidden input
    else:
        kernel_assert(
            input_tensors.hidden_input.dtype in MX_UNPACKED_DTYPES,
            f"Expected quantized input dtype in {MX_UNPACKED_DTYPES} with all_to_all_v_strategy!=DISABLED, got {input_tensors.hidden_input.dtype=}, {dynamism_cfg.all_to_all_v_strategy=}",
        )

    # Validate T size based on quantization.
    # When quantized, T can be divisible by 4; otherwise T must be divisible by 32. When using SHARD_T, these constraints double.
    is_quantized = sizeinbytes(input_tensors.hidden_input.dtype) < 2 or input_tensors.hidden_input_scale is not None
    is_static_mx = kernel_cfg.is_static_quant
    if is_quantized:
        kernel_assert(dims.T % 4 == 0, f"Expected T divisible by 4 with quantized input, got T={dims.T}")
        if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
            kernel_assert(
                dims.T_local % 4 == 0,
                f"Expected T_local divisible by 4 for SHARD_T with sub-2-byte input, got T_local={dims.T_local}.",
            )
    # all-expert STATIC_MX expects non pre-quantized inputs
    elif not is_static_mx:
        kernel_assert(
            dims.T_physical % 32 == 0,
            f"Expected T divisible by 32, got T={dims.T}. "
            "To use T divisible by 4, provide prequantized input and hidden_input_scale.",
        )
        if dims.sharding_strategy == MoELNCShardingStrategy.SHARD_T:
            kernel_assert(
                dims.T_local % 32 == 0,
                f"Expected T_local divisible by 32 for SHARD_T with HBM input, got T_local={dims.T_local}.",
            )

    # Validate expert affinities shape (affinities are packed when using all_to_all_v)
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
        kernel_assert(
            len(input_tensors.expert_affinities_masked.shape) in (2, 3),
            f"Expected 2D or 3D expert_affinities_masked, got {input_tensors.expert_affinities_masked.shape=}",
        )

    # Validate output location
    kernel_assert(
        not kernel_cfg.output_in_sbuf,
        f"All-expert MX kernel does not yet support SBUF output, got {kernel_cfg.output_in_sbuf=}",
    )

    # Validate input_in_sbuf requires pre-quantized input (except STATIC_MX which receives bf16 for per-expert quantization)
    if kernel_cfg.input_in_sbuf and not kernel_cfg.is_static_quant:
        kernel_assert(
            input_tensors.hidden_input_scale != None,
            f"Expected pre-quantized input when input is in SBUF, "
            f"got {input_tensors.hidden_input.dtype=} {input_tensors.hidden_input_scale=}",
        )

    # Dynamism constraints
    if dynamism_cfg.is_all_expert_dynamic:
        if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
            kernel_assert(
                input_tensors.expert_affinities_masked.buffer != nl.sbuf,
                f"Expected expert_affinities_masked in HBM, got {input_tensors.expert_affinities_masked.buffer=}",
            )
        kernel_assert(
            dynamism_cfg.block_size != None and _is_valid_block_size(dims.T, dynamism_cfg.block_size),
            f"Invalid block_size: expected (1) nonzero block_size (2) block_size that evenly divides T, (3) block_size at most T/2, "
            f"and (4) block_size≤32 and divisible by 4, block_size≤128 and divisible by 32, or block_size divisible by 128; "
            f"but got {dynamism_cfg.block_size=}, {dims.T=}",
        )

    # all_to_all_v requires is_all_expert_dynamic
    else:
        kernel_assert(
            dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED,
            f"all_to_all_v_strategy != DISABLED is only supported with is_all_expert_dynamic=True, but got {dynamism_cfg.is_all_expert_dynamic=}, {dynamism_cfg.all_to_all_v_strategy=}",
        )

    # Validate all_to_all_v input shape
    if dynamism_cfg.all_to_all_v_strategy != MoEAllToAllVStrategy.DISABLED:
        kernel_assert(
            dims.H_concat == dims.H + dims.H // _q_width + dims.E_L * FP8_PER_BF16 + FP8_PER_INT32,
            f"Expected {dims.H + dims.H // _q_width + dims.E_L * FP8_PER_BF16 + FP8_PER_INT32=} columns in hidden_input with all_to_all_v_strategy!=DISABLED, but got {input_tensors.hidden_input.shape=}, {dynamism_cfg.all_to_all_v_strategy=}",
        )
        kernel_assert(
            input_tensors.hidden_input.dtype in MX_UNPACKED_DTYPES,
            f"Expected hidden_input.dtype in {MX_UNPACKED_DTYPES}, but got {input_tensors.hidden_input.dtype=}",
        )
        kernel_assert(
            input_tensors.expert_affinities_masked == None,
            f"Expected expert affinities packed into hidden_input and expert_affinities_masked=None when all_to_all_v_strategy!=DISABLED, but got {input_tensors.expert_affinities_masked=}, {dynamism_cfg.all_to_all_v_strategy=}",
        )


def _is_valid_block_size(T: int, block_size: int) -> bool:
    """
    Validate that block_size is valid for a given T.

    Block size must be nonzero, evenly divide T, be at most T/2 (resulting in at least 2 blocks),
    and satisfy: block_size≤32 and divisible by 4, block_size≤128 and divisible by 32, or
    block_size divisible by 128.

    Args:
        T (int): Total number of tokens.
        block_size (int): Block size to validate.

    Returns:
        bool: True if block_size is valid, False otherwise.
    """
    if block_size == 0:
        return False
    if T % block_size != 0:
        return False
    if block_size > T // 2:
        return False
    if block_size <= 32:
        return block_size % 4 == 0
    elif block_size <= 128:
        return block_size % 32 == 0
    return block_size % 128 == 0
