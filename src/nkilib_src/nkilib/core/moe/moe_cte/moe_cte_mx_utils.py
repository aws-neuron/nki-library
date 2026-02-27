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

"""Shared utilities and dataclasses for MXFP4/MXFP8 MoE kernels with intermediate dimension sharding.

This module provides configuration classes, buffer containers, and helper functions for
MXFP4/MXFP8-quantized MoE operations. Key functionality includes:
- Hidden state loading with H-dimension folding for efficient vector DGE
- Layout adaptation (transpose) for MXFP4 quantization alignment
- Index vector computation for indirect DGE token gathering
- Online quantization of activations to FP8 format
"""

from dataclasses import dataclass
from typing import Any, Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_program_sharding_info, reduce
from .moe_cte_utils import SkipMode, div_ceil

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# SBUF partition dimension maximum (128 partitions)
_pmax = 128

# PSUM free dimension maximum (512 elements in fp32)
_psum_fmax = 512

# PSUM bank dimension maximum
_psum_bmax = 8

# TRN3 MXFP4/8 quantization block dimensions
_q_height = 8  # Partitions per quantization block
_q_width = 4  # Free dimension elements per quantization block

# SBUF quadrant size (32 partitions per quadrant)
SBUF_QUADRANT_SIZE = 32

# Alternative dtypes for MXFP weights (torch/xla doesn't support passing mxfp tensors directly)
# Mapping from alternative dtypes to their MXFP target dtype
# (torch/xla doesn't support passing MXFP tensors directly)
_ALTERNATIVE_DTYPE_TO_MXFP = {
    nl.uint16: nl.float4_e2m1fn_x4,
    nl.int16: nl.float4_e2m1fn_x4,
    nl.float16: nl.float4_e2m1fn_x4,
    nl.uint32: nl.float8_e4m3fn_x4,
    nl.int32: nl.float8_e4m3fn_x4,
    nl.float32: nl.float8_e4m3fn_x4,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DTYPE CONVERSION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def convert_to_mxfp_dtype(tensor: nl.ndarray, weight_dtype: Any = None) -> tuple:
    """
    Convert weight tensor to appropriate MXFP dtype.

    torch/xla doesn't support passing MXFP tensors to kernels directly, so weights
    are passed as alternative dtypes (uint16/int16/float16 for MXFP4, uint32/int32/float32
    for MXFP8) and converted here.

    Args:
        tensor (nl.ndarray): Weight tensor that may need dtype conversion.
        weight_dtype: Target MXFP dtype. If None, auto-detects based on input dtype:
            - MXFP4 alternatives -> nl.float4_e2m1fn_x4
            - MXFP8 alternatives -> nl.float8_e4m3fn_x4

    Returns:
        tuple: (converted_tensor, target_dtype)
            - converted_tensor: Tensor with view cast to target MXFP dtype
            - target_dtype: The MXFP dtype used for conversion
    """
    mxfp_target = _ALTERNATIVE_DTYPE_TO_MXFP.get(tensor.dtype)

    if weight_dtype is None:
        target_dtype = mxfp_target if mxfp_target is not None else tensor.dtype
    else:
        target_dtype = weight_dtype

    if mxfp_target is not None:
        tensor = tensor.view(target_dtype, shape=tensor.shape)

    return tensor, target_dtype


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InputTensors(nl.NKIObject):
    """Container for all input tensors passed to MXFP4 MoE kernels.

    Groups HBM tensors and SBUF buffers used throughout the MoE computation pipeline.
    Tensors are organized by their role: token mapping, weights, scales, and biases.

    Attributes:
        token_position_to_id (nl.ndarray): [N*B], Maps block positions to token IDs in HBM.
        block_to_expert (nl.ndarray): [N, 1], Expert assignment per block in HBM.
        hidden_states (nl.ndarray): [T, H], Input hidden states in HBM.
        gate_up_proj_weight (nl.ndarray): [E, 128, 2, n_H512_tile, I], MXFP4 gate/up weights.
        gate_and_up_proj_bias (nl.ndarray): [E, 128, 2, n_I512_tile, 4], Gate/up biases.
        down_proj_bias (nl.ndarray): [E, H], Down projection biases.
        down_proj_weight (nl.ndarray): [E, p_I, n_I512_tile, H], MXFP4 down weights.
        expert_affinities_masked (nl.ndarray): [(T+1)*E, 1], Expert affinities per token.
        gate_up_proj_scale (nl.ndarray): [E, 16, 2, n_H512_tile, I], Gate/up dequant scales.
        down_proj_scale (nl.ndarray): [E, 16, n_I512_tile, H], Down projection dequant scales.
        p_gup_idx_vector (nl.ndarray): [128, 1], Reusable index vector for gate/up scale DGE.
        p_down_idx_vector (nl.ndarray): [128, 1], Reusable index vector for down scale DGE.
        gup_scales_sb (nl.ndarray): [128, 2, n_H512_tile, I], Gate/up scales buffer in SBUF.
        activation_bias (nl.ndarray): [128, 1], Activation bias buffer in SBUF.
        conditions (nl.ndarray): [N+1], Block validity conditions for dynamic loops.
    """

    token_position_to_id: nl.ndarray
    block_to_expert: nl.ndarray
    hidden_states: nl.ndarray
    gate_up_proj_weight: nl.ndarray
    gate_and_up_proj_bias: nl.ndarray
    down_proj_bias: nl.ndarray
    down_proj_weight: nl.ndarray
    expert_affinities_masked: nl.ndarray
    gate_up_proj_scale: nl.ndarray
    down_proj_scale: nl.ndarray
    p_gup_idx_vector: nl.ndarray = None
    p_down_idx_vector: nl.ndarray = None
    gup_scales_sb: nl.ndarray = None
    activation_bias: nl.ndarray = None
    conditions: nl.ndarray = None


@dataclass
class SharedBuffers(nl.NKIObject):
    """Container for SBUF buffers shared across MoE computation stages.

    These buffers are allocated once and reused across blocks to minimize
    memory allocation overhead. Includes buffers for hidden state processing,
    quantization, and intermediate results.

    Attributes:
        block_hidden_states (nl.ndarray): [128, B/32, n_H512_tile, 128], Pre-transpose hidden states.
        block_hidden_states_T (nl.ndarray): [128, n_H512_tile, B/32, 128], Transposed hidden states.
        token_4_H_indices_on_p (nl.ndarray): [128, B/32], Token indices with H-folding on partitions.
        hidden_qtz_sb (nl.ndarray): [128, n_H512_tile, B], Quantized hidden states (FP8).
        hidden_scale_sb (nl.ndarray): [128, n_H512_tile, B], Quantization scales (uint8).
        down_weight_qtz (nl.ndarray): [128, n_I512_tile, H], Down projection weights in SBUF.
        block_old (nl.ndarray): [128, n_B128_tiles, H], Previous output for accumulation.
        cond (nl.ndarray): Condition buffer for dynamic loop control.
        index (nl.ndarray): Index buffer for dynamic loop control.
        down_scale_sb (nl.ndarray): Down projection scales in SBUF.
    """

    block_hidden_states: nl.ndarray
    block_hidden_states_T: nl.ndarray
    token_4_H_indices_on_p: nl.ndarray
    hidden_qtz_sb: nl.ndarray
    hidden_scale_sb: nl.ndarray
    down_weight_qtz: nl.ndarray
    block_old: nl.ndarray
    cond: nl.ndarray = None
    index: nl.ndarray = None
    down_scale_sb: nl.ndarray = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProjConfig(nl.NKIObject):
    """Configuration for MXFP4 projection sub-kernels.

    Computes tiling parameters and validates constraints for gate/up and down
    projections. Supports both I-sharding (intermediate dimension) and H-sharding
    (hidden dimension) configurations.

    Attributes:
        H (int): Hidden dimension size.
        I (int): Intermediate dimension size (before TP sharding).
        BxS (int): Batch * Sequence length (tokens per block).
        n_prgs (int): Number of programs/shards.
        prg_id (int): Current program/shard ID.
        force_lnc1 (bool): Force single-core mode when True.
        bias_t_shared_between_gate_up (bool): Share bias tensor between gate and up projections.
        bias_t_shared_base_offset (int): Base offset when sharing bias tensor.
        out_p_offset (int): Output partition offset for TKG projections (BxS <= 128).
        dbg_hidden (bool): Debug flag to return hidden states early.
        dbg_weight (bool): Debug flag to return weights early.
        sharding_config (str): Sharding mode - "I" for intermediate, "H" for hidden.

    Computed Attributes (set in __post_init__):
        H0 (int): Partition dimension size (always 128).
        H1 (int): H // 128, elements per partition.
        n_H512_tile (int): Number of 512-element tiles in H dimension.
        n_I512_tile (int): Number of 512-element tiles in I dimension.
        n_I512_tile_lnc_sharded (int): I512 tiles per shard.
        I_lnc_sharded (int): I dimension size per shard.
        BxS_tile_sz (int): Tile size for BxS dimension processing.
    """

    # Tensor shapes / dims
    H: int
    I: int
    BxS: int

    # LNC info
    n_prgs: int
    prg_id: int
    force_lnc1: bool = False

    # Bias sharing configuration
    bias_t_shared_between_gate_up: bool = False
    bias_t_shared_base_offset: int = 0

    # Down projection output partition offset (for TKG with BxS <= 128)
    out_p_offset: int = 0

    # Bias broadcast method: True = stream shuffle broadcast (default), False = PE broadcast via matmul
    use_stream_shuffle_broadcast: bool = True

    # Debug flags
    dbg_hidden: bool = False
    dbg_weight: bool = False
    sharding_config: str = "I"

    def _check_shapes_H_sharded(self):
        """Validate H-sharding configuration constraints."""
        kernel_assert(self.H % _pmax == 0, f"H={self.H} must be divisible by num partitions ({_pmax})")
        kernel_assert(self.H1 % self.n_prgs == 0, f"H1={self.H1} must be divisible by num shards ({self.n_prgs})")
        kernel_assert(
            self.H1_sharded % _q_width == 0,
            f"H1_sharded={self.H1_sharded} must be divisible by quantization width ({_q_width})",
        )
        kernel_assert(
            self.r_I512_tile % (_q_width * _q_height) == 0,
            f"I512 tile remainder ({self.r_I512_tile}) must be divisible by "
            f"quantization block size ({_q_width * _q_height})",
        )

        if self.out_p_offset != 0:
            kernel_assert(0 <= self.out_p_offset < _pmax, f"out_p_offset={self.out_p_offset} must be in [0, {_pmax})")
            kernel_assert(
                self.out_p_offset % SBUF_QUADRANT_SIZE == 0,
                f"out_p_offset={self.out_p_offset} must align to SBUF quadrant ({SBUF_QUADRANT_SIZE})",
            )
            kernel_assert(
                self.out_p_offset + self.BxS <= _pmax,
                f"out_p_offset + BxS = {self.out_p_offset + self.BxS} exceeds partition limit ({_pmax})",
            )

    def __post_init__(self):
        """Initialize derived tiling parameters based on sharding configuration."""
        if self.sharding_config == "I":
            self._generate_I_shard_config()
        elif self.sharding_config == "H":
            self._generate_H_shard_config()
        else:
            kernel_assert(False, f"Unsupported sharding type: {self.sharding_config}")

    def _generate_I_shard_config(self):
        """
        Generate tiling configuration for I-dimension sharding.

        In I-sharding mode, the intermediate dimension is split across cores.
        Each core processes I/n_prgs elements of the intermediate dimension.
        """
        self.H0 = _pmax
        self.H1 = self.H // _pmax
        self.I_lnc_sharded = self.I // self.n_prgs

        # H dimension tiling: 512 = 128 partitions * 4 (q_width)
        self.n_H512_tile = self.H1 // _q_width

        # I dimension tiling
        I512_tiling_info = divmod(self.I, _pmax * _q_width)
        self.n_I512_tile = I512_tiling_info[0]
        n_I512_tile_lnc_sharded, n_I512_tile_lnc_sharded_r = divmod(self.n_I512_tile, self.n_prgs)
        self.n_I512_tile_lnc_sharded = n_I512_tile_lnc_sharded
        kernel_assert(n_I512_tile_lnc_sharded_r == 0, "I need to be multiple of 1024")
        self.r_I512_tile = I512_tiling_info[1]
        kernel_assert(self.r_I512_tile == 0, "I need to be multiple of 1024")
        self.n_total_I512_tile = self.n_I512_tile + (self.r_I512_tile > 0)
        n_total_I512_tile_lnc_sharded, _ = divmod(self.n_total_I512_tile, self.n_prgs)
        self.n_total_I512_tile_lnc_sharded = n_total_I512_tile_lnc_sharded

        # BxS tile size: limited by PSUM capacity (bf16 output = 2x fp32 elements)
        self.BxS_tile_sz = min(self.BxS, _psum_fmax * 2 // _q_width)

        if self.force_lnc1:
            self.n_prgs = 1
            self.prg_id = 0

    def _generate_H_shard_config(self):
        """
        Generate tiling configuration for H-dimension sharding.

        In H-sharding mode, the hidden dimension is split across cores.
        Each core processes H/n_prgs elements of the hidden dimension.
        """
        self.H0 = _pmax
        self.H1 = self.H // _pmax
        self.H1_sharded = self.H1 // self.n_prgs
        self.H_sharded = self.H // self.n_prgs

        # Sharded H dimension tiling
        self.n_H512_tile_sharded = self.H1_sharded // _q_width
        self.n_H512_tile = self.n_H512_tile_sharded * self.n_prgs

        # I dimension tiling
        I512_tiling_info = divmod(self.I, _pmax * _q_width)
        self.n_I512_tile = I512_tiling_info[0]
        self.r_I512_tile = I512_tiling_info[1]
        self.n_total_I512_tile = self.n_I512_tile + (self.r_I512_tile > 0)
        self.n_par_r_I512_tile = self.r_I512_tile // _q_width

        # BxS tile size
        self.BxS_tile_sz = min(self.BxS, _psum_fmax * 2 // _q_width)

        self.H_tile_size = 2 * _psum_fmax if self.H_sharded % (2 * _psum_fmax) == 0 else _psum_fmax
        self.n_H_tile_sharded = self.H_sharded // self.H_tile_size
        if self.force_lnc1:
            self.n_prgs = 1
            self.prg_id = 0

        self._check_shapes_H_sharded()


@dataclass
class BWMMMXDimensionSizes(nl.NKIObject):
    """Dimension sizes and derived tiling parameters for MXFP4 blockwise matmul.

    Captures all dimension information needed for the MoE kernel and computes
    derived values like tile counts and sharding parameters.

    Attributes:
        B (int): Block size (tokens per block).
        H (int): Hidden dimension size.
        T (int): Total number of tokens.
        E (int): Number of experts.
        N (int): Number of blocks.
        I (int): Intermediate dimension size.
        cond_vec_len (int): Length of conditions vector for dynamic loops.
        TILESIZE (int): Base tile size (default 512).

    Computed Attributes:
        num_shards (int): Number of LNC shards.
        shard_id (int): Current shard ID.
        n_B128_tiles (int): Number of 128-token tiles in block.
        n_B128_tiles_sharded (int): B128 tiles per shard.
        p_I (int): Partition dimension for I (128 if I > 512, else I//4).
        I_lnc_sharded (int): I dimension per shard.
    """

    B: int
    H: int
    T: int
    E: int
    N: int
    I: int
    cond_vec_len: int
    TILESIZE: int = _psum_fmax

    def __post_init__(self):
        """Compute derived dimension values and sharding parameters."""
        _, num_shards, shard_id = get_program_sharding_info()
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.MODULO_FACTOR = self.B // self.TILESIZE
        self.n_B128_tiles = div_ceil(self.B, _pmax)
        self.n_B128_tiles_sharded = div_ceil(self.n_B128_tiles, self.num_shards)
        self.hidden_sbuf_expected_shape = (
            SBUF_QUADRANT_SIZE * _q_width,
            self.B // SBUF_QUADRANT_SIZE,
            div_ceil(self.H, 512),
            16 * 8,
        )
        self.p_I = _pmax if self.I > 512 else self.I // _q_width
        self.I_lnc_sharded = self.I // num_shards


@dataclass
class BWMMMXConfigs(nl.NKIObject):
    """Runtime configuration for MXFP4 blockwise matmul kernel.

    Contains all configuration flags and parameters that control kernel behavior
    including quantization settings, activation functions, and loop control.

    Attributes:
        scaling_mode (ExpertAffinityScaleMode): When to apply expert affinity scaling.
        skip_dma (SkipMode): DMA skip configuration for debugging.
        compute_dtype: Data type for intermediate computations.
        weight_dtype: Data type for quantized weights.
        io_dtype: Data type for input/output tensors.
        is_tensor_update_accumulating (bool): Accumulate output for TopK > 1.
        use_dynamic_while (bool): Use dynamic loop control for padded blocks.
        n_static_blocks (int): Number of non-padded blocks for static loop.
        linear_bias (bool): Whether bias tensors are provided.
        activation_function (ActFnType): Activation function (SiLU, SwiGLU, etc.).
        fuse_gate_and_up_load (bool): Load gate and up weights together.
        gate_clamp_upper_limit (float): Upper clamp for gate projection output.
        gate_clamp_lower_limit (float): Lower clamp for gate projection output.
        up_clamp_upper_limit (float): Upper clamp for up projection output.
        up_clamp_lower_limit (float): Lower clamp for up projection output.
        qtz_dtype: Quantization data type for activations (e.g., float8_e4m3fn_x4).
    """

    scaling_mode: ExpertAffinityScaleMode
    skip_dma: SkipMode
    compute_dtype: Any
    weight_dtype: Any
    io_dtype: Any
    is_tensor_update_accumulating: bool
    use_dynamic_while: bool
    n_static_blocks: int
    linear_bias: bool
    activation_function: ActFnType
    fuse_gate_and_up_load: bool
    gate_clamp_upper_limit: Optional[float]
    gate_clamp_lower_limit: Optional[float]
    up_clamp_upper_limit: Optional[float]
    up_clamp_lower_limit: Optional[float]
    qtz_dtype: Any


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN STATE LOADING AND TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════


def load_hidden_states_mx(
    inps: InputTensors,
    dims: BWMMMXDimensionSizes,
    skip_dma: SkipMode,
    block_idx: int = None,
    token_4_H_indices_on_p: nl.ndarray = None,
    block_hidden_states: nl.ndarray = None,
    block_hidden_states_T: nl.ndarray = None,
    use_dma_transpose: bool = False,
):
    """
    Load hidden states from HBM to SBUF with optional DMA transpose.

    This function gathers hidden states for tokens in the current block using
    indirect vector DGE with H-dimension folding. Supports two modes:
    - PE mode (use_dma_transpose=False): Loads to intermediate buffer, requires separate transpose
    - DMA mode (use_dma_transpose=True): Loads and transposes in one step using hardware DMA

    Memory Layout Transformation:
        HBM: hidden_states [T, H]
        View as: [T*4, H/512, 128] where H is folded 4x onto the T dimension
        PE mode output: block_hidden_states [128, B/32, H/512, 128]
        DMA mode output: block_hidden_states_T [128, H/512, B/32, 128] (transposed)

    Args:
        inps (InputTensors): Input tensors containing hidden_states [T, H] in HBM.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        skip_dma (SkipMode): DMA skip configuration for debugging.
        block_idx (int): Current block index (required for DMA mode).
        token_4_H_indices_on_p (nl.ndarray): Precomputed indices [128, B/32] with
            H-folded token positions (required for PE mode).
        block_hidden_states (nl.ndarray): Destination buffer [128, B/32, H/512, 128]
            in SBUF (required for PE mode).
        block_hidden_states_T (nl.ndarray): Destination buffer [128, H/512, B/32, 128]
            in SBUF for transposed output (required for DMA mode).
        use_dma_transpose (bool): If True, use DMA transpose; if False, use PE mode.

    Returns:
        None: Modifies block_hidden_states or block_hidden_states_T in-place.

    Notes:
        - PE mode: Processes B/32 tiles sequentially using precomputed indices
        - DMA mode: Uses nisa.dma_transpose with axes=(2,1,0)
        - Uses oob_mode.skip when skip_dma.skip_token is True for padding tokens

    Pseudocode:
        if use_dma_transpose:
            token_indices = load token_position_to_id[block_idx*B:(block_idx+1)*B]
            all_token_4_H_indices = token_indices * 4 + [0, 1, 2, 3]
            for b_tile_idx in range(B/32):
                indices_on_p = transpose(all_token_4_H_indices[:, b_tile_idx*128:(b_tile_idx+1)*128])
                block_hidden_states_T[:, :, b_tile_idx, :] = dma_transpose(hidden_states[indices_on_p], axes=(2,1,0))
        else:
            hidden_view = reshape hidden_states to [T*4, H/512, 128]
            for b_tile_idx in range(B/32):
                indices = token_4_H_indices_on_p[:, b_tile_idx]
                block_hidden_states[:, b_tile_idx, :, :] = hidden_view[indices, :, :]
    """
    B = dims.B
    H_div_512 = dims.H // 512
    B_div_32 = B // SBUF_QUADRANT_SIZE

    if use_dma_transpose:
        kernel_assert(block_idx is not None, "block_idx required for DMA transpose mode")
        kernel_assert(block_hidden_states_T is not None, "block_hidden_states_T required for DMA transpose mode")

        hidden_states = inps.hidden_states
        token_position_to_id = inps.token_position_to_id

        # Load token indices for current block
        token_indices = nl.ndarray((1, dims.B), dtype=token_position_to_id.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=token_indices,
            src=token_position_to_id.reshape((1, token_position_to_id.shape[0]))[
                :, block_idx * dims.B : dims.B * (block_idx + 1)
            ],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
        kernel_assert(token_indices.shape == (1, dims.B), f"token_indices.shape = {token_indices.shape}")

        # Create [0, 1, 2, 3] offset vector for H-folding
        arange_4H = nl.ndarray((1, _q_width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(arange_4H, [[1, _q_width]], offset=0)

        # Compute 4H-folded indices: token_indices * 4 + [0,1,2,3]
        all_token_4_H_indices = nl.ndarray((1, dims.B, _q_width), dtype=nl.float32, buffer=nl.sbuf)

        nisa.scalar_tensor_tensor(
            dst=all_token_4_H_indices,
            data=token_indices.ap(pattern=[[dims.B, 1], [1, dims.B], [0, _q_width]], offset=0),
            op0=nl.multiply,
            operand0=float(_q_width),
            op1=nl.add,
            operand1=arange_4H.ap(pattern=[[_q_width, 1], [0, dims.B], [1, _q_width]], offset=0),
        )

        all_token_4_H_indices = all_token_4_H_indices.reshape((1, dims.B * _q_width))
        kernel_assert(all_token_4_H_indices.shape == (1, dims.B * _q_width), f"got {all_token_4_H_indices.shape}")

        # Process each B/32 tile with DMA transpose
        for b_tile_idx in range(B_div_32):
            token_4_H_indices = all_token_4_H_indices[:, b_tile_idx * _pmax, b_tile_idx * _pmax + _pmax]
            token_4_H_indices_psum = nl.ndarray(
                (token_4_H_indices.shape[1], token_4_H_indices.shape[0]), dtype=token_4_H_indices.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(dst=token_4_H_indices_psum, data=token_4_H_indices)

            token_4_H_indices_on_p_local = nl.ndarray(token_4_H_indices_psum.shape, dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=token_4_H_indices_on_p_local, src=token_4_H_indices_psum, engine=nisa.scalar_engine)

            kernel_assert(
                token_4_H_indices_on_p_local.shape == (_pmax, 1),
                f"token_4_H_indices.shape = {token_4_H_indices_on_p_local.shape}",
            )

            nisa.dma_transpose(
                dst=block_hidden_states_T[: (SBUF_QUADRANT_SIZE * _q_width), :H_div_512, b_tile_idx, : (16 * 8)],
                src=hidden_states.reshape((dims.T * _q_width, H_div_512, _pmax))[
                    token_4_H_indices_on_p_local[:, 0], :H_div_512, : (16 * 8)
                ],
                axes=(2, 1, 0),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
    else:
        kernel_assert(token_4_H_indices_on_p is not None, "token_4_H_indices_on_p required for PE mode")
        kernel_assert(block_hidden_states is not None, "block_hidden_states required for PE mode")

        hidden_states_view = inps.hidden_states.reshape((dims.T * _q_width, H_div_512, _pmax))

        for b_tile_idx in range(B_div_32):
            nisa.dma_copy(
                src=hidden_states_view.ap(
                    pattern=[
                        [H_div_512 * _pmax, SBUF_QUADRANT_SIZE * _q_width],
                        [1, 1],
                        [_pmax, H_div_512],
                        [1, 16 * 8],
                    ],
                    offset=0,
                    vector_offset=token_4_H_indices_on_p.ap(
                        [[B_div_32, _pmax], [1, 1]],
                        offset=b_tile_idx,
                    ),
                    indirect_dim=0,
                ),
                dst=block_hidden_states.ap(
                    pattern=[
                        [B_div_32 * H_div_512 * _pmax, SBUF_QUADRANT_SIZE * _q_width],
                        [H_div_512 * _pmax, 1],
                        [_pmax, H_div_512],
                        [1, 16 * 8],
                    ],
                    offset=b_tile_idx * H_div_512 * _pmax,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )


def compute_hidden_index_vector(
    inps: InputTensors,
    buffers: SharedBuffers,
    block_idx,
    dims: BWMMMXDimensionSizes,
    skip_dma: SkipMode,
    is_block_idx_dynamic: bool = False,
):
    """
    Compute token-to-hidden-state index mapping for indirect DGE loading.

    Transforms token indices into 4H-folded indices suitable for vector DGE.
    The transformation enables loading hidden states with the H dimension
    folded 4x onto partitions for efficient memory access.

    Index Transformation:
        Input: token_indices [B] with values in range [0, T)
        Output: token_4_H_indices [128, B/32] with values in range [0, T*4)

        For each token index t, generates 4 indices: [4*t, 4*t+1, 4*t+2, 4*t+3]
        These are transposed to partition dimension for vector DGE.

    Args:
        inps (InputTensors): Input tensors with token_position_to_id mapping.
        buffers (SharedBuffers): Buffers including token_4_H_indices_on_p output.
        block_idx: Current block index (int for static, nl.ndarray for dynamic).
        dims (BWMMMXDimensionSizes): Dimension configuration.
        skip_dma (SkipMode): DMA skip configuration.
        is_block_idx_dynamic (bool): True if block_idx is from dynamic loop.

    Returns:
        None: Modifies buffers.token_4_H_indices_on_p in-place.

    Notes:
        - Uses scalar_tensor_tensor for efficient broadcast: indices * 4 + [0,1,2,3]
        - Transposes result to partition dimension via nc_transpose
        - Handles both static (compile-time) and dynamic (runtime) block indices

    Pseudocode:
        token_indices = load token_position_to_id[block_idx*B : (block_idx+1)*B]
        arange_4H = [0, 1, 2, 3]
        all_indices = token_indices * 4 + arange_4H  # broadcast
        all_indices = reshape to [1, B*4]

        for tile_idx in range(B/32):
            indices_128 = all_indices[:, tile_idx*128 : (tile_idx+1)*128]
            token_4_H_indices_on_p[:, tile_idx] = transpose(indices_128)
    """
    num_tokens = dims.B
    num_token_tiles = num_tokens // SBUF_QUADRANT_SIZE

    token_position_to_id = inps.token_position_to_id

    # Load token indices for current block using appropriate DGE mode
    if is_block_idx_dynamic:
        total_size = reduce(op='mul', input=token_position_to_id.shape, initial_value=1)
        kernel_assert(total_size % dims.B == 0, "token_position_to_id shape must be divisible by B")
        token_indices = nl.ndarray((1, dims.B), buffer=nl.sbuf, dtype=nl.int32)
        reshaped = token_position_to_id.reshape((total_size // dims.B, dims.B))
        nisa.dma_copy(
            src=reshaped.ap(pattern=[[dims.B, 1], [1, dims.B]], offset=0, scalar_offset=block_idx, indirect_dim=0),
            dst=token_indices[:, : dims.B],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )
    else:
        token_indices = nl.ndarray((1, num_tokens), buffer=nl.sbuf, dtype=nl.int32)
        nisa.dma_copy(
            src=token_position_to_id.reshape((1, token_position_to_id.shape[0]))[
                :, block_idx * dims.B : dims.B * (block_idx + 1)
            ],
            dst=token_indices[:, :num_tokens],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )

    kernel_assert(token_indices.shape == (1, num_tokens), f"token_indices.shape = {token_indices.shape}")

    # Create [0, 1, 2, 3] offset vector for H-folding
    arange_4H = nl.ndarray((1, _q_width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(arange_4H, [[1, _q_width]], offset=0)

    # Compute 4H-folded indices: token_indices * 4 + [0,1,2,3]
    # This broadcasts token_indices across the 4 H-fold dimension
    all_token_4_H_indices = nl.ndarray((1, num_tokens, _q_width), dtype=nl.float32, buffer=nl.sbuf)

    nisa.scalar_tensor_tensor(
        dst=all_token_4_H_indices,
        data=token_indices.ap(
            pattern=[[num_tokens, 1], [1, num_tokens], [0, _q_width]],  # step=0 broadcasts across q_width dim
            offset=0,
        ),
        op0=nl.multiply,
        operand0=float(_q_width),
        op1=nl.add,
        operand1=arange_4H.ap(
            pattern=[[_q_width, 1], [0, num_tokens], [1, _q_width]],  # step=0 broadcasts across num_tokens dim
            offset=0,
        ),
    )

    all_token_4_H_indices = all_token_4_H_indices.reshape((1, num_tokens * _q_width))

    # Transpose indices to partition dimension for vector DGE (128 indices per tile)
    for tile_idx in range(num_token_tiles):
        token_4_H_indices = all_token_4_H_indices[:, tile_idx * _pmax : tile_idx * _pmax + _pmax]

        token_4_H_indices_psum = nl.ndarray(
            (token_4_H_indices.shape[1], token_4_H_indices.shape[0]), dtype=token_4_H_indices.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(dst=token_4_H_indices_psum, data=token_4_H_indices)

        nisa.tensor_copy(
            dst=buffers.token_4_H_indices_on_p[:, tile_idx], src=token_4_H_indices_psum, engine=nisa.scalar_engine
        )


def quantize_block_hidden_state_T(buffers: SharedBuffers, prj_cfg: ProjConfig, dims: BWMMMXDimensionSizes):
    """
    Quantize transposed block hidden states to FP8 format for MXFP4/MXFP8 matmul.

    Performs online quantization of hidden states from BF16/FP32 to FP8 (float8_e4m3fn)
    with per-block scaling factors. The quantization is required because nc_matmul_mx
    expects FP8 activations when using MXFP4/MXFP8 weights.

    Quantization Block Structure:
        - MXFP8 uses 8x4 quantization blocks (8 partitions x 4 free elements)
        - Each block shares a single uint8 scale factor
        - Input is 4x larger than output due to 4:1 compression ratio

    Args:
        buffers (SharedBuffers): Buffers containing:
            - block_hidden_states_T: Input [128, n_H512_tile, B/32, 128] in BF16
            - hidden_qtz_sb: Output [128, n_H512_tile, B/32, 32] in FP8
            - hidden_scale_sb: Output [128, n_H512_tile, B/32, 32] in uint8
        prj_cfg (ProjConfig): Projection configuration with n_H512_tile.
        dims (BWMMMXDimensionSizes): Dimension configuration with B.

    Returns:
        None: Modifies buffers.hidden_qtz_sb and buffers.hidden_scale_sb in-place.

    Notes:
        - Uses hardware-accelerated quantize_mx instruction
        - Output free dimension is 1/4 of input due to FP8 packing
        - Scales are stored separately for dequantization during matmul

    Pseudocode:
        quantize_mx(
            src=block_hidden_states_T[:128, :n_H512_tile, :B//32, :128],
            dst=hidden_qtz_sb[:128, :n_H512_tile, :B//32, :32],
            dst_scale=hidden_scale_sb[:128, :n_H512_tile, :B//32, :32]
        )
    """
    num_b_tiles = dims.B // SBUF_QUADRANT_SIZE
    nisa.quantize_mx(
        src=buffers.block_hidden_states_T[:_pmax, : prj_cfg.n_H512_tile, :num_b_tiles, :_pmax],
        dst=buffers.hidden_qtz_sb[:_pmax, : prj_cfg.n_H512_tile, :num_b_tiles, :SBUF_QUADRANT_SIZE],
        dst_scale=buffers.hidden_scale_sb[:_pmax, : prj_cfg.n_H512_tile, :num_b_tiles, :SBUF_QUADRANT_SIZE],
    )


def sbuf_layout_adapter(
    src: nl.ndarray,
    dst: nl.ndarray,
    dims: BWMMMXDimensionSizes,
    use_dma_tp: bool = False,
):
    """
    Transpose tensor layout in SBUF for MXFP4 quantization alignment.

    Performs layout transformation to prepare hidden states for quantization.
    The transpose swaps the token and H-slice dimensions to align data for
    efficient MXFP4 matmul operations.

    Layout Transformation:
        Input:  [32*4 (P), B/32, H/512, 16*8] - tokens interleaved with H-slices on P
        Output: [128 (P), H/512, B/32, 128]   - H-slices on P, tokens on free dim

    The transformation is necessary because:
    1. Hidden states are loaded with H folded 4x onto partitions for efficient DGE
    2. MXFP4 matmul expects H on partition dimension, tokens on free dimension
    3. This transpose reorders data to match matmul input requirements

    Args:
        src (nl.ndarray): Source tensor [32*4, B/32, H/512, 16*8] in SBUF.
        dst (nl.ndarray): Destination tensor [128, H/512, B/32, 128] in SBUF.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        use_dma_tp (bool): If True, use DMA transpose; if False, use nc_transpose.

    Returns:
        None: Modifies dst tensor in-place.

    Notes:
        - nc_transpose method uses PSUM multi-buffering (8 transposes per bank)
        - DMA transpose is faster but may have hardware limitations
        - Each 128x128 transpose moves one B/32 tile for one H/512 tile

    Pseudocode:
        B_div_32 = B // 32
        H_div_512 = H // 512
        transposes_per_bank = min(8, B_div_32)
        num_banks = B_div_32 // transposes_per_bank

        if use_dma_tp:
            tmp = dma_transpose(src, axes=(3,1,2,0))
            copy tmp to dst with dimension reordering
        else:
            for h_tile_idx in range(H_div_512):
                for bank_idx in range(num_banks):
                    psum_buffer = allocate PSUM
                    for tp_idx in range(transposes_per_bank):
                        b_tile_idx = bank_idx * transposes_per_bank + tp_idx
                        psum_buffer[:, tp_idx*128:(tp_idx+1)*128] = nc_transpose(src[:, b_tile_idx, h_tile_idx, :])
                    dst[:, h_tile_idx, bank_idx, :] = psum_buffer
    """
    src_sbuf = src

    B_div_32 = dims.B // SBUF_QUADRANT_SIZE
    H_div_512 = dims.H // 512

    # Multi-buffering: 8 transposes fit in one PSUM bank (8 * 128 * 128 * 2 bytes = 256KB)
    transposes_per_bank = min(_psum_bmax, B_div_32)
    num_banks = B_div_32 // transposes_per_bank

    if use_dma_tp:
        tmp_sbuf = nl.ndarray((_pmax, B_div_32, H_div_512, _pmax), dtype=src_sbuf.dtype)
        nisa.dma_transpose(dst=tmp_sbuf[:, :, :, :], src=src_sbuf[:, :, :, :], axes=(3, 1, 2, 0))

        for h_tile_idx in range(H_div_512):
            for b_tile_idx in range(B_div_32):
                dst[0:_pmax, h_tile_idx, b_tile_idx, :_pmax] = tmp_sbuf[0:_pmax, b_tile_idx, h_tile_idx, :_pmax]
    else:
        # Reshape dst for efficient bank-based access
        dst = dst.reshape((_pmax, H_div_512, num_banks, transposes_per_bank * SBUF_QUADRANT_SIZE * _q_width))

        for h_tile_idx in range(H_div_512):
            for bank_idx in range(num_banks):
                # Allocate PSUM buffer for multiple transposes
                psum_buffer = nl.ndarray((_pmax, transposes_per_bank * _pmax), dtype=src.dtype, buffer=nl.psum)

                for tp_idx in range(transposes_per_bank):
                    b_tile_idx = bank_idx * transposes_per_bank + tp_idx
                    # Transpose [32*4, 16*8] -> [128, 128]
                    nisa.nc_transpose(
                        dst=psum_buffer[:_pmax, tp_idx * _pmax : (tp_idx + 1) * _pmax],
                        data=src_sbuf[0:_pmax, b_tile_idx, h_tile_idx, 0:_pmax],
                    )

                # Evict full bank to SBUF
                nisa.tensor_copy(
                    dst[0:_pmax, h_tile_idx, bank_idx, 0 : transposes_per_bank * _pmax],
                    psum_buffer[:_pmax, : transposes_per_bank * _pmax],
                    engine=nisa.scalar_engine,
                )

        # Restore original shape
        dst = dst.reshape((_pmax, H_div_512, B_div_32, SBUF_QUADRANT_SIZE * _q_width))


def load_and_quantize_hidden_states(
    inps: InputTensors,
    block_idx,
    buffers: SharedBuffers,
    dims: BWMMMXDimensionSizes,
    kernel_cfg: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    is_block_idx_dynamic: bool = False,
    use_dma_transpose: bool = False,
):
    """
    Load, transpose, and quantize hidden states for current block.

    Orchestrates the complete pipeline: load from HBM, transpose layout,
    and quantize to MXFP4 format.

    Args:
        inps (InputTensors): Input tensors.
        block_idx (int): Current block index.
        buffers (SharedBuffers): Shared buffers for intermediate results.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        kernel_cfg (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        is_block_idx_dynamic (bool): Whether using dynamic block indexing.
        use_dma_transpose (bool): Use DMA transpose instead of PE transpose.

    Returns:
        None: Modifies buffers in-place with quantized hidden states.

    Notes:
        - Supports both DMA transpose and PE transpose methods
        - Reshapes buffers for quantization
        - Called once per block during processing

    Pseudocode:
        if use_dma_transpose:
            load_hidden_states_mx(inps, dims, skip_dma, block_idx=block_idx,
                                  block_hidden_states_T=buffers.block_hidden_states_T, use_dma_transpose=True)
        else:
            compute_hidden_index_vector(inps, buffers, block_idx, dims, skip_dma, is_block_idx_dynamic)
            load_hidden_states_mx(inps, dims, skip_dma, token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                                  block_hidden_states=buffers.block_hidden_states, use_dma_transpose=False)
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        reshape buffers.hidden_qtz_sb to [128, n_H512_tile, B//32, 32]
        reshape buffers.hidden_scale_sb to [128, n_H512_tile, B//32, 32]
        quantize_block_hidden_state_T(buffers, prj_cfg, dims)
    """
    if use_dma_transpose:
        load_hidden_states_mx(
            inps,
            dims,
            kernel_cfg.skip_dma,
            block_idx=block_idx,
            block_hidden_states_T=buffers.block_hidden_states_T,
            use_dma_transpose=True,
        )
    else:
        compute_hidden_index_vector(inps, buffers, block_idx, dims, kernel_cfg.skip_dma, is_block_idx_dynamic)
        load_hidden_states_mx(
            inps,
            dims,
            kernel_cfg.skip_dma,
            token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
            block_hidden_states=buffers.block_hidden_states,
            use_dma_transpose=False,
        )
        sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

    buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
    buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # quantize. Note that online quantize can only quantize to fp8
    quantize_block_hidden_state_T(buffers, prj_cfg, dims)


def _generate_expert_index_vector(
    expert_index: nl.ndarray,
    dst_idx_vector: nl.ndarray,
    scale_factor: int,
    n_quadrants_needed: int,
    n_remaining_partition: int = 0,
) -> nl.ndarray:
    """
    Generate padded index vector for expert-based DGE scale loading.

    Constructs a partition-dimension index vector that maps expert indices to
    scale tensor rows. Indices are scattered across quadrants (32-element blocks)
    with -1 padding for unused positions.

    Args:
        expert_index (nl.ndarray): Expert index scalar, shape [1, 1].
        dst_idx_vector (nl.ndarray): Destination buffer for index vector, shape [128, 1].
        scale_factor (int): Multiplier for expert index (typically 16 for E*16 folding).
        n_quadrants_needed (int): Number of full 32-element quadrants to fill.
        n_remaining_partition (int): Extra partitions beyond full quadrants (0-3).

    Returns:
        nl.ndarray: Index vector in SBUF with shape [128, 1], dtype int32.

    Notes:
        - Output pattern for expert=0: [0,1,2,3,-1...,-1, 4,5,6,7,-1...,-1, ...]
        - Output pattern for expert=3: [48,49,50,51,-1..., 52,53,54,55,-1..., ...]
        - Remainder partitions are placed at quadrant n_quadrants_needed
    """
    n_quadrants = n_quadrants_needed + (n_remaining_partition > 0)
    kernel_assert(n_quadrants <= 4, f"n_quadrants must be <= 4, got {n_quadrants}")
    kernel_assert(n_remaining_partition < 4, f"n_remaining_partition must be < 4, got {n_remaining_partition}")

    arange_f = nl.ndarray((1, n_quadrants * 4), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=arange_f, pattern=[[1, n_quadrants * 4]])

    padded_index = nl.ndarray((1, _pmax), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=padded_index, value=-1.0)

    nisa.scalar_tensor_tensor(
        dst=padded_index.ap([[_pmax, 1], [SBUF_QUADRANT_SIZE, n_quadrants], [1, 4]]),
        data=expert_index.ap([[1, 1], [0, n_quadrants * 4]]),
        op0=nl.multiply,
        operand0=float(scale_factor),
        op1=nl.add,
        operand1=arange_f,
    )

    if n_remaining_partition > 0:
        offset = (n_quadrants - 1) * SBUF_QUADRANT_SIZE + n_remaining_partition
        extra = 4 - n_remaining_partition
        nisa.memset(dst=padded_index.ap([[_pmax, 1], [1, extra]], offset=offset), value=-1.0)

    padded_index_psum = nl.ndarray((_pmax, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=padded_index_psum, data=padded_index)
    nisa.tensor_copy(dst=dst_idx_vector, src=padded_index_psum)

    token_indices_on_p = nl.ndarray(dst_idx_vector.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=token_indices_on_p, src=dst_idx_vector, engine=nisa.scalar_engine)
    return token_indices_on_p
