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
from ...utils.kernel_helpers import (
    NUM_HW_PSUM_BANKS,
    PSUM_BANK_SIZE,
    _psum_alloc,
    _sbm_alloc,
    get_program_sharding_info,
    reduce,
)
from .moe_cte_utils import SkipMode, div_ceil

# --- HARDWARE CONSTANTS ---

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

# PE-transpose view dtypes for sub-32-bit data. The PE (nc_transpose) cannot take fp8x4 or
# uint8 inputs directly, so we bitcast to a 32-bit / 8-bit float view that the engine accepts:
#   - fp8x4 hidden is viewed as fp32 (4 fp8 bytes ride inside one fp32 element, preserving 4_H).
#   - uint8 MX scales are viewed as fp8_e5m2 (same byte width, transposes element-wise).
_FP8X4_TP_VIEW_DTYPE = nl.float32
_UINT8_TP_VIEW_DTYPE = nl.float8_e5m2
_FP8_PER_FP32 = 4
# PE FP8 transpose mode requires the output to have an element step of 2; the uint8 scale
# transpose writes into a PSUM buffer with this trailing interleave dim and reads back slot 0.
_FP8_TP_OUT_STEP = 2
# Packed MX scales: rmsnorm_mx_prefill folds 4 H512 tiles' scales into one 128-wide block (one
# scale per 32-element quadrant occupies only 4 of 32 columns, so 4 tiles share a 128-wide block).
_SCALES_PER_PACK = 4


# Alternative dtypes for MXFP weights — torch/xla doesn't support passing mxfp tensors directly
# Mapping from alternative dtypes to their MXFP target dtype.
_ALTERNATIVE_DTYPE_TO_MXFP = {
    nl.uint16: nl.float4_e2m1fn_x4,
    nl.int16: nl.float4_e2m1fn_x4,
    nl.uint32: nl.float8_e4m3fn_x4,
    nl.int32: nl.float8_e4m3fn_x4,
    nl.float8_e4m3fn: nl.float8_e4m3fn_x4,
    nl.float8_e5m2: nl.float8_e5m2_x4,
}


# --- DTYPE CONVERSION UTILITIES ---
def convert_to_mxfp_dtype(tensor: nl.NkiTensor, weight_dtype: Any = None) -> tuple:
    """
    Convert weight tensor to appropriate MXFP dtype.

    torch/xla doesn't support passing MXFP tensors to kernels directly, so weights
    are passed as alternative dtypes (uint16/int16/float16 for MXFP4, uint32/int32/float32
    for MXFP8) and converted here.

    Args:
        tensor (nl.NkiTensor): Weight tensor that may need dtype conversion.
        weight_dtype: Target MXFP dtype. If None, auto-detects based on input dtype:
            - MXFP4 alternatives -> nl.float4_e2m1fn_x4
            - MXFP8 alternatives -> nl.float8_e4m3fn_x4

    Returns:
        tuple: (converted_tensor, target_dtype)
            - converted_tensor: Tensor with view cast to target MXFP dtype
            - target_dtype: The MXFP dtype used for conversion
    """
    mxfp_target = _ALTERNATIVE_DTYPE_TO_MXFP.get(tensor.dtype)

    if weight_dtype == None:
        target_dtype = mxfp_target if mxfp_target != None else tensor.dtype
    else:
        target_dtype = weight_dtype

    if mxfp_target != None:
        # view as the target MXFP dtype
        tensor = tensor.view(target_dtype)

    return tensor, target_dtype


# --- DATA CONTAINERS ---


@dataclass
class InputTensors(nl.NKIObject):
    """Container for all input tensors passed to MXFP4 MoE kernels.

    Groups HBM tensors and SBUF buffers used throughout the MoE computation pipeline.
    Tensors are organized by their role: token mapping, weights, scales, and biases.

    Attributes:
        token_position_to_id (nl.NkiTensor): [N*B], Maps block positions to token IDs in HBM.
        block_to_expert (nl.NkiTensor): [N, 1], Expert assignment per block in HBM.
        hidden_states (nl.NkiTensor): [T, H], Input hidden states in HBM.
        gate_up_proj_weight (nl.NkiTensor): [E, 128, 2, n_H512_tile, I], MXFP4 gate/up weights.
        gate_and_up_proj_bias (nl.NkiTensor): [E, 128, 2, n_I512_tile, 4], Gate/up biases.
        down_proj_bias (nl.NkiTensor): [E, H], Down projection biases.
        down_proj_weight (nl.NkiTensor): [E, p_I, n_I512_tile, H], MXFP4 down weights.
        expert_affinities_masked (nl.NkiTensor): [(T+1)*E, 1], Expert affinities per token.
        gate_up_proj_scale (nl.NkiTensor): [E, 16, 2, n_H512_tile, I], Gate/up dequant scales.
        down_proj_scale (nl.NkiTensor): [E, 16, n_I512_tile, H], Down projection dequant scales.
        local_expert_start_idx (nl.NkiTensor): [1, 1] int32 SBUF, this instance's first global expert id, derived on-device as ep_rank * E_local; packed-affinity path only.
        p_gup_idx_vector (nl.NkiTensor): [128, 1], Reusable index vector for gate/up scale DGE.
        p_gup_idx_vector_int32 (nl.NkiTensor): [128, 1], Persistent int32 cache of gate/up index vector.
        p_down_idx_vector (nl.NkiTensor): [128, 1], Reusable index vector for down scale DGE.
        gup_scales_sb (nl.NkiTensor): [128, 2, n_H512_tile, I], Gate/up scales buffer in SBUF.
        activation_bias (nl.NkiTensor): [128, 1], Activation bias buffer in SBUF.
        conditions (nl.NkiTensor): [N+1], Block validity conditions for dynamic loops.
        arange_4H (nl.NkiTensor): [1, 4], Persistent [0,1,2,3] H-fold offset vector, hoisted once at entry.
        gup_scale_lut_sb (nl.NkiTensor): fp32 [128, E, 3], STATIC_MX per-expert gate/up table (1/in_scale, in_scale*gate_w_scale, in_scale*up_w_scale), broadcast across partitions.
        down_scale_lut_sb (nl.NkiTensor): fp32 [128, E, 2], STATIC_MX per-expert down table (1/down_in_scale, down_in_scale*down_w_scale), broadcast across partitions.
    """

    token_position_to_id: nl.NkiTensor
    block_to_expert: nl.NkiTensor
    hidden_states: nl.NkiTensor
    gate_up_proj_weight: nl.NkiTensor
    gate_and_up_proj_bias: nl.NkiTensor
    down_proj_bias: nl.NkiTensor
    down_proj_weight: nl.NkiTensor
    expert_affinities_masked: nl.NkiTensor
    gate_up_proj_scale: nl.NkiTensor
    down_proj_scale: nl.NkiTensor
    local_expert_start_idx: nl.NkiTensor = None
    p_gup_idx_vector: nl.NkiTensor = None
    p_gup_idx_vector_int32: nl.NkiTensor = None
    p_down_idx_vector: nl.NkiTensor = None
    gup_scales_sb: nl.NkiTensor = None
    activation_bias: nl.NkiTensor = None
    conditions: nl.NkiTensor = None
    arange_4H: nl.NkiTensor = None
    gup_scale_lut_sb: nl.NkiTensor = None
    down_scale_lut_sb: nl.NkiTensor = None


@dataclass
class SharedBuffers(nl.NKIObject):
    """Container for SBUF buffers shared across MoE computation stages.

    These buffers are allocated once and reused across blocks to minimize
    memory allocation overhead. Includes buffers for hidden state processing,
    quantization, and intermediate results.

    Attributes:
        block_hidden_states (nl.NkiTensor): [128, B/32, n_H512_tile, 128], Pre-transpose hidden states.
        block_hidden_states_T (nl.NkiTensor): [128, n_H512_tile, B/32, 128], Transposed hidden states.
        token_4_H_indices_on_p (nl.NkiTensor): [128, B/32], Token indices with H-folding on partitions.
        hidden_qtz_sb (nl.NkiTensor): [128, n_H512_tile, B], Quantized hidden states (FP8).
        hidden_scale_sb (nl.NkiTensor): [128, n_H512_tile, B], Quantization scales (uint8).
        down_weight_qtz (nl.NkiTensor): [128, n_I512_tile, H], Down projection weights in SBUF.
        block_old (nl.NkiTensor): [128, n_B128_tiles, H], Previous output for accumulation.
        cond (nl.NkiTensor): Condition buffer for dynamic loop control.
        index (nl.NkiTensor): Index buffer for dynamic loop control.
        down_scale_sb (nl.NkiTensor): Down projection scales in SBUF.
    """

    block_hidden_states: nl.NkiTensor
    block_hidden_states_T: nl.NkiTensor
    token_4_H_indices_on_p: nl.NkiTensor
    hidden_qtz_sb: nl.NkiTensor
    hidden_scale_sb: nl.NkiTensor
    down_weight_qtz: nl.NkiTensor
    block_old: nl.NkiTensor
    cond: nl.NkiTensor = None
    index: nl.NkiTensor = None
    down_scale_sb: nl.NkiTensor = None
    # fp8 pre-quantized path: gathered [_pmax, B//_pmax, H_load] concat rows
    # ([hidden_quant | hidden_scale]) awaiting transpose into hidden_qtz_sb / hidden_scale_sb.
    block_hidden_concat: nl.NkiTensor = None
    broadcast_stationary: nl.NkiTensor = None
    gup_tile_buf_a: nl.NkiTensor = None
    dummy_inter_scale_sb: nl.NkiTensor = None


# --- CONFIGURATION CLASSES ---


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

    # Zero unused partitions in gate/up projection output when I % 512 != 0
    zero_unused_partitions: bool = True

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
    # When True, gate_up_proj_scale and down_proj_scale are passed in PACKED HBM layout:
    #   gate_up_proj_scale: uint8[E, _pmax=128, n_packed_gup, 2, I]
    #   down_proj_scale:    uint8[E, _pmax=128, n_packed_down, H]
    # The kernel dispatches to packed code paths in the projection sub-kernels.
    # The flag is supplied by the caller; the kernel does not infer it from shape.
    use_packed_scales: bool = False
    # When True, gate_up_proj_scale and down_proj_scale are passed in DeepSeek-style
    # block-128 HBM layout (one uint8 per 128(K) x 128(N) weight block):
    #   gate_up_proj_scale: uint8[E, 4, 2, n_H512_tile, ceil(I / 128)]
    #   down_proj_scale:    uint8[E, 4, n_I512_tile, ceil(H / 128)]
    # The scale loaders materialize this into the same SBUF layout nc_matmul_mx
    # consumes (one scale per 8(p) x 4(f) block) via a two-stage broadcast load,
    # so the matmul and projection compute are identical to the native MX path.
    # This is a distinct option from use_packed_scales (partition-slot repacking).
    use_block128_scales: bool = False
    is_static_quant: bool = False
    # When True (only relevant under is_static_quant), silu cannot be fused into the gate
    # projection's STATIC_MX activation — clamp must run between (dequant+bias) and silu.
    has_gate_clamp: bool = False
    # When True, the persistent gup buffer holds the FULL gate+up weights (full I dim)
    # resident across blocks. The tiled compute reads I-tile slices from it directly
    # instead of streaming each tile. Cross-block prefetch OOB-skips on same-expert →
    # 100% gup DMAs avoided. Activated only under STATIC_MX with skip_weight enabled
    # and dims.I > _I_TILE_SZ.
    gup_full_persistent: bool = False
    # When True, hidden states arrive PRE-QUANTIZED (fp8 + uint8 MX scales) in a concatenated
    # [T, H_concat] tensor. The kernel gathers + PE-transposes both regions (no on-device
    # quantize_mx) via load_fp8_hidden_states_mx. Only supported with QuantizationType.MX.
    is_fp8_hidden: bool = False
    # When True (only with is_fp8_hidden), the concat row also carries the dense [T, E] expert
    # affinities after the scale region, in expert_affinities_dtype. Each block extracts its expert's
    # affinity column from the packed hidden.
    # affinities_col_offset is the fp8 column where the affinity region begins.
    is_affinities_packed: bool = False
    affinities_col_offset: int = 0
    # dtype the packed affinity tail is stored in (== the incoming hidden_states dtype). Used to
    # reinterpret the tail: sizeinbytes(this) fp8 columns per affinity element.
    expert_affinities_dtype: Any = nl.bfloat16


# --- HIDDEN STATE LOADING AND TRANSFORMATION ---


def load_fp8_hidden_states_mx(
    inps: InputTensors,
    dims: BWMMMXDimensionSizes,
    skip_dma: SkipMode,
    token_indices_on_p: nl.NkiTensor,
    block_hidden_concat: nl.NkiTensor,
):
    """Prefetch (gather only) PRE-QUANTIZED fp8 hidden states for one block.

    Indirect-gathers the per-block tokens' concatenated rows from HBM into block_hidden_concat.
    This is a pure DMA stage (no PE), so it overlaps the prior block's compute — mirroring the
    bf16 load_hidden_states_mx prefetch. The matching transpose_fp8_hidden_states does the PE work.

    HBM layout: hidden_states [T, H_concat], each row = [hidden_quant (H fp8) | hidden_scale
    (packed scale_region uint8)]; scale_region = ceil((H/512)/4)*128.

    Args:
        inps (InputTensors): inps.hidden_states is the concat HBM tensor [T, H_concat].
        dims (BWMMMXDimensionSizes): Dimension configuration (H, B).
        skip_dma (SkipMode): Padding tokens are skipped via oob_mode.skip.
        token_indices_on_p (nl.NkiTensor): [_pmax, B//_pmax] int32 token ids on partitions for this block.
        block_hidden_concat (nl.NkiTensor): Dest [_pmax, B//_pmax, H_load] gathered concat rows.
    """
    B = dims.B
    H_concat = inps.hidden_states.shape[-1]
    n_B_tiles = div_ceil(B, _pmax)
    blk_T = min(B, _pmax)
    # Gather the FULL concat row. H_concat = H + scale_region [+ affin_region when the producer fused
    # expert affinities into the row]. Loading the whole row pulls the (optional) affinity tail with
    # hidden+scale in one indirect DMA. Affinities region is extracted later.
    row_load = H_concat

    for b_tile_idx in range(n_B_tiles):
        nisa.dma_copy(
            src=inps.hidden_states.ap(
                pattern=[[H_concat, blk_T], [1, row_load]],
                offset=0,
                vector_offset=token_indices_on_p.ap(
                    pattern=[[n_B_tiles, blk_T], [1, 1]],
                    offset=b_tile_idx,
                ),
                indirect_dim=0,
            ),
            dst=block_hidden_concat[0:blk_T, b_tile_idx, 0:row_load],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            dge_mode=dge_mode.swdge,
        )


def transpose_fp8_hidden_states(
    dims: BWMMMXDimensionSizes,
    prj_cfg,
    block_hidden_concat: nl.NkiTensor,
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scale_sb: nl.NkiTensor,
    sbm=None,
):
    """PE-transpose prefetched fp8 hidden + uint8 MX scales into the matmul-ready layout.

    Reads the gathered block_hidden_concat (filled by load_fp8_hidden_states_mx) and transposes
    the quant region into hidden_qtz_sb [128_H, n_H512_tile, B] and the PACKED scale region into
    hidden_scale_sb [128_H, n_packed, B] (n_packed = ceil(n_H512_tile/4); 4 H512 tiles fold into one
    128-wide block). No quantization — the data is already MX-quantized. This is the deferred PE stage
    (mirrors the bf16 sbuf_layout_adapter), run after the projections so nc_transpose doesn't contend
    with the matmuls. The matmul consumes the packed scales via is_packed_moving_scale addressing.

    PE transpose cannot take fp8x4/uint8 directly: the fp8 region rides through an fp32 view (4_H
    packed inside one fp32) and the uint8 scales through an fp8_e5m2 view
    """
    H = dims.H
    B = dims.B
    n_H512_tile = prj_cfg.n_H512_tile
    n_packed = div_ceil(n_H512_tile, _SCALES_PER_PACK)  # packed scale blocks (4 H512 tiles each)
    tile_H = _pmax  # 128 H-partitions per H512 tile
    H_load = block_hidden_concat.shape[-1]
    n_B_tiles = div_ceil(B, _pmax)
    blk_T = min(B, _pmax)

    # Per-partition (per-token-row) free size of block_hidden_concat, in fp8 bytes and fp32 elements.
    row_stride_fp8 = n_B_tiles * H_load
    row_stride_fp32 = row_stride_fp8 // _FP8_PER_FP32

    # Batch the copy-out: pack several H512 transposes into one PSUM bank at distinct free offsets,
    # then do ONE strided tensor_copy per group.
    # A PSUM bank holds 512 (_psum_fmax) fp32 elements; the scale view is uint8 with a x2 interleave (HW constraint),
    # so it packs 2x as many tiles per bank.
    quant_tiles_per_group = max(1, _psum_fmax // blk_T)
    scale_tiles_per_group = max(1, _psum_fmax * _FP8_TP_OUT_STEP // blk_T)
    n_quant_groups = div_ceil(n_H512_tile, quant_tiles_per_group)
    n_scale_groups = div_ceil(n_packed, scale_tiles_per_group)

    # Double-buffer PSUM (like sbuf_layout_adapter): one bank copies out (DVE) while the next fills
    # (TE). Separate bank pairs for quant (fp32-view) and scale (fp8-view) so they don't serialize.
    quant_psum_bufs = [
        _psum_alloc((tile_H, quant_tiles_per_group, blk_T), _FP8X4_TP_VIEW_DTYPE, sbm, 0 * PSUM_BANK_SIZE),
        _psum_alloc((tile_H, quant_tiles_per_group, blk_T), _FP8X4_TP_VIEW_DTYPE, sbm, 1 * PSUM_BANK_SIZE),
    ]
    scale_psum_bufs = [
        _psum_alloc(
            (tile_H, scale_tiles_per_group, blk_T, _FP8_TP_OUT_STEP), _UINT8_TP_VIEW_DTYPE, sbm, 2 * PSUM_BANK_SIZE
        ),
        _psum_alloc(
            (tile_H, scale_tiles_per_group, blk_T, _FP8_TP_OUT_STEP), _UINT8_TP_VIEW_DTYPE, sbm, 3 * PSUM_BANK_SIZE
        ),
    ]

    for b_tile_idx in range(n_B_tiles):
        # Transpose fp8 hidden region [blk_T, H] -> [128_H, n_H512_tile, blk_T] via fp32 view.
        for group_idx in range(n_quant_groups):  # how many psum bank fills we will have
            first_tile = group_idx * quant_tiles_per_group
            tiles_in_group = min(quant_tiles_per_group, n_H512_tile - first_tile)
            quant_psum = quant_psum_bufs[group_idx % 2]
            for tile_in_group in range(tiles_in_group):
                nisa.nc_transpose(
                    data=block_hidden_concat.ap(
                        pattern=[[row_stride_fp32, blk_T], [1, tile_H]],
                        offset=(b_tile_idx * H_load) // _FP8_PER_FP32
                        + (first_tile + tile_in_group)
                        * tile_H,  # (start of the B-tile) + (H512-tile offset inside hidden)
                        dtype=_FP8X4_TP_VIEW_DTYPE,
                    ),
                    dst=quant_psum[:, tile_in_group, :],
                )
            # One strided copy lands the whole group's tiles into hidden_qtz_sb.
            nisa.tensor_copy(
                src=quant_psum[:, :tiles_in_group, :],
                dst=hidden_qtz_sb.ap(
                    pattern=[[n_H512_tile * B, tile_H], [B, tiles_in_group], [1, blk_T]],
                    offset=first_tile * B + b_tile_idx * blk_T,
                    dtype=_FP8X4_TP_VIEW_DTYPE,
                ),
            )

        """
        Transpose the PACKED uint8 scale region [blk_T, n_packed*128] -> [128_H, n_packed, blk_T]
        via fp8_e5m2 view. Each pack block is a contiguous 128-wide region in HBM (4 H512 tiles
        already folded by the producer), so we transpose n_packed contiguous-128 blocks. FP8
        transpose mode requires the PE output to have an element step of 2, so the PSUM dst carries
        a trailing interleave dim of size 2 and we read back slot 0
        """
        for group_idx in range(n_scale_groups):  # how many psum bank fill iterations we need
            first_tile = group_idx * scale_tiles_per_group
            tiles_in_group = min(scale_tiles_per_group, n_packed - first_tile)
            scale_psum = scale_psum_bufs[group_idx % 2]  # double buffered
            for tile_in_group in range(tiles_in_group):  # how many tiles fit into one psum bank to evict out at once
                nisa.nc_transpose(
                    data=block_hidden_concat.ap(
                        pattern=[[row_stride_fp8, blk_T], [1, tile_H]],
                        offset=b_tile_idx * H_load + H + (first_tile + tile_in_group) * tile_H,
                        dtype=_UINT8_TP_VIEW_DTYPE,
                    ),
                    dst=scale_psum[:, tile_in_group, :, 0],
                )
            nisa.tensor_copy(
                src=scale_psum[:, :tiles_in_group, :, 0],
                dst=hidden_scale_sb.ap(
                    pattern=[[n_packed * B, tile_H], [B, tiles_in_group], [1, blk_T]],
                    offset=first_tile * B + b_tile_idx * blk_T,
                    dtype=_UINT8_TP_VIEW_DTYPE,  # Scale Dtype fp8_e5m2
                ),
            )


def load_hidden_states_mx(
    inps: InputTensors,
    dims: BWMMMXDimensionSizes,
    skip_dma: SkipMode,
    block_idx: int = None,
    token_4_H_indices_on_p: nl.NkiTensor = None,
    block_hidden_states: nl.NkiTensor = None,
    block_hidden_states_T: nl.NkiTensor = None,
    use_dma_transpose: bool = False,
    sbm=None,
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

    Both modes consume token_4_H_indices_on_p [128, B/32] (uint32), which must be
    populated by compute_hidden_index_vector() prior to calling this function.

    Args:
        inps (InputTensors): Input tensors containing hidden_states [T, H] in HBM.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        skip_dma (SkipMode): DMA skip configuration for debugging.
        block_idx (int): Unused (kept for backward compatibility).
        token_4_H_indices_on_p (nl.NkiTensor): Precomputed indices [128, B/32] (uint32)
            with H-folded token positions on partitions. Required for both modes.
        block_hidden_states (nl.NkiTensor): Destination buffer [128, B/32, H/512, 128]
            in SBUF (required for PE mode).
        block_hidden_states_T (nl.NkiTensor): Destination buffer [128, H/512, B/32, 128]
            in SBUF for transposed output (required for DMA mode).
        use_dma_transpose (bool): If True, use DMA transpose; if False, use PE mode.

    Returns:
        None: Modifies block_hidden_states or block_hidden_states_T in-place.

    Notes:
        - PE mode: Uses vector_offset-based dma_copy per B/32 tile
        - DMA mode: Single vector_offset-based dma_transpose with axes=(3,1,2,0)
        - Uses oob_mode.skip when skip_dma.skip_token == True for padding tokens

    Pseudocode:
        if use_dma_transpose:
            # Single DMA transpose with gather via vector_offset
            src = hidden_states.reshape(T*4, 1, H/512, 128).ap(
                      pattern=..., vector_offset=token_4_H_indices_on_p[:128, :N])
            dst = block_hidden_states_T.reshape(128, 1, H/512, N*128).ap(pattern=...)
            dma_transpose(src, dst, axes=(3,1,2,0))
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
        kernel_assert(block_hidden_states_T != None, "block_hidden_states_T required for DMA transpose mode")
        kernel_assert(token_4_H_indices_on_p != None, "token_4_H_indices_on_p required for DMA transpose mode")

        hidden_states = inps.hidden_states
        N = B_div_32  # N = (B * 4) // 128 = B // 32

        # Single dma_transpose with vector_offset gather: write all N B/32-tiles at once.
        # _pmax*N rows are gathered from hidden_states via token_4_H_indices_on_p[:_pmax, :N];
        # the transpose places those rows on the free axis of block_hidden_states_T while the
        # original 128-element free axis becomes the partition axis.
        #
        # 4D reshape with dummy dim=1 satisfies the HW constraint that one of the
        # non-partition transpose dimensions must be padding (size 1).
        # src (_pmax*N, 1, H_div_512, 128) --axes=(3,1,2,0)--> dst (128, 1, H_div_512, _pmax*N)
        nisa.dma_transpose(
            src=hidden_states.reshape((dims.T * _q_width, 1, H_div_512, _pmax)).ap(
                pattern=[
                    [H_div_512 * _pmax, _pmax * N],  # dim0: _pmax*N rows gathered via vector_offset
                    [1, 1],  # dim1: dummy padding dimension
                    [_pmax, H_div_512],  # dim2: H_div_512 tiles
                    [1, _pmax],  # dim3: 128 free elements per tile
                ],
                offset=0,
                vector_offset=token_4_H_indices_on_p[:_pmax, :N].view(nl.uint32),
            ),
            dst=block_hidden_states_T.reshape((_pmax, 1, H_div_512, N * _pmax)).ap(
                pattern=[
                    [H_div_512 * N * _pmax, _pmax],  # dim0: _pmax partitions
                    [1, 1],  # dim1: dummy padding dimension
                    [N * _pmax, H_div_512],  # dim2: H_div_512 tiles
                    [1, N * _pmax],  # dim3: N*128 free elements (all N tiles)
                ],
            ),
            axes=(3, 1, 2, 0),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            priority=0,
        )

    else:
        kernel_assert(token_4_H_indices_on_p != None, "token_4_H_indices_on_p required for PE mode")
        kernel_assert(block_hidden_states != None, "block_hidden_states required for PE mode")

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
    sbm=None,
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
        block_idx: Current block index (int for static, nl.NkiTensor for dynamic).
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
        token_indices = _sbm_alloc(
            sbm, (1, dims.B), dtype=nl.int32, name=f"hiv_token_indices_dyn{block_idx}", align=SBUF_QUADRANT_SIZE
        )
        reshaped = token_position_to_id.reshape((total_size // dims.B, dims.B))
        nisa.dma_copy(
            src=reshaped.ap(pattern=[[dims.B, 1], [1, dims.B]], offset=0, scalar_offset=block_idx, indirect_dim=0),
            dst=token_indices[:, : dims.B],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )
    else:
        token_indices = _sbm_alloc(
            sbm, (1, num_tokens), dtype=nl.int32, name=f"hiv_token_indices{block_idx}", align=SBUF_QUADRANT_SIZE
        )
        nisa.dma_copy(
            src=token_position_to_id.reshape((1, token_position_to_id.shape[0]))[
                :, block_idx * dims.B : dims.B * (block_idx + 1)
            ],
            dst=token_indices[:, :num_tokens],
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )

    kernel_assert(token_indices.shape == (1, num_tokens), f"token_indices.shape = {token_indices.shape}")

    # [0, 1, 2, 3] offset vector for H-folding; hoisted once at kernel entry
    # (inps.arange_4H) to avoid per-block iota allocation.
    arange_4H = inps.arange_4H

    # Compute 4H-folded indices: token_indices * 4 + [0,1,2,3]
    # This broadcasts token_indices across the 4 H-fold dimension
    all_token_4_H_indices = _sbm_alloc(
        sbm, (1, num_tokens, _q_width), dtype=nl.float32, name=f"hiv_all_token_4H_{block_idx}", align=SBUF_QUADRANT_SIZE
    )

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

        token_4_H_indices_psum = _psum_alloc(
            (token_4_H_indices.shape[1], token_4_H_indices.shape[0]),
            token_4_H_indices.dtype,
            sbm,
            (tile_idx % NUM_HW_PSUM_BANKS) * PSUM_BANK_SIZE,
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


def quantize_block_hidden_state_T_static_mx(
    buffers: SharedBuffers,
    prj_cfg: ProjConfig,
    dims: BWMMMXDimensionSizes,
    in_quant_recip: nl.NkiTensor,
):
    """STATIC_MX variant: software-quant block_hidden_states_T (bf16) → hidden_qtz_sb (fp8_x4).

    Caller passes the per-block reciprocal (= 1 / gate_up_in_scale[block_expert]) gathered
    from the [_pmax, E, 3] gup table. The matmul-side moving_scale reuses buffers.hidden_scale_sb,
    which the kernel pre-fills with 0x7F7F7F7F (uint8 = all 127) at top of kernel for STATIC_MX.

    Mirrors MoE TKG selective_expert_mx_impl.py STATIC_MX pattern:
      1. tensor_scalar to scale bf16 hidden, writing into a separately-allocated fp8 buffer
         (saturation cast handles the ±448 clip).
      2. tensor_copy via uint32 view to land bytes in hidden_qtz_sb (fp8_x4).
    """
    num_b_tiles = dims.B // SBUF_QUADRANT_SIZE

    hidden_qtz_fp8 = buffers.hidden_qtz_sb.view(nl.float8_e4m3fn)
    nisa.tensor_scalar(
        dst=hidden_qtz_fp8,
        data=buffers.block_hidden_states_T[:_pmax, : prj_cfg.n_H512_tile, :num_b_tiles, :_pmax],
        op0=nl.multiply,
        operand0=in_quant_recip,
    )


def sbuf_layout_adapter(
    src: nl.NkiTensor,
    dst: nl.NkiTensor,
    dims: BWMMMXDimensionSizes,
    sbm=None,
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
        src (nl.NkiTensor): Source tensor [32*4, B/32, H/512, 16*8] in SBUF.
        dst (nl.NkiTensor): Destination tensor [128, H/512, B/32, 128] in SBUF.
        dims (BWMMMXDimensionSizes): Dimension configuration.

    Returns:
        None: Modifies dst tensor in-place.

    Notes:
        - Uses nc_transpose with PSUM multi-buffering (8 transposes per bank)
        - Each 128x128 transpose moves one B/32 tile for one H/512 tile

    Pseudocode:
        B_div_32 = B // 32
        H_div_512 = H // 512
        transposes_per_bank = min(8, B_div_32)
        num_banks = B_div_32 // transposes_per_bank

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

    # Reshape dst for efficient bank-based access
    dst = dst.reshape((_pmax, H_div_512, num_banks, transposes_per_bank * SBUF_QUADRANT_SIZE * _q_width))

    # Double-buffer: 2 pre-allocated PSUM banks, alternate across H tiles
    # While one bank copies out (DVE), the next fills with transposes (TE)

    if sbm != None:
        psum_bufs = [
            _psum_alloc((_pmax, transposes_per_bank * _pmax), src_sbuf.dtype, sbm, 0 * PSUM_BANK_SIZE),
            _psum_alloc((_pmax, transposes_per_bank * _pmax), src_sbuf.dtype, sbm, 1 * PSUM_BANK_SIZE),
        ]
    else:
        psum_bufs = [
            _psum_alloc((_pmax, transposes_per_bank * _pmax), src_sbuf.dtype, None),
            _psum_alloc((_pmax, transposes_per_bank * _pmax), src_sbuf.dtype, None),
        ]

    for h_tile_idx in nl.affine_range(H_div_512):
        for bank_idx in range(num_banks):
            psum_buffer = psum_bufs[(h_tile_idx * num_banks + bank_idx) % 2]

            for tp_idx in range(transposes_per_bank):
                b_tile_idx = bank_idx * transposes_per_bank + tp_idx
                nisa.nc_transpose(
                    dst=psum_buffer[:_pmax, tp_idx * _pmax : (tp_idx + 1) * _pmax],
                    data=src_sbuf[0:_pmax, b_tile_idx, h_tile_idx, 0:_pmax],
                )

            nisa.tensor_copy(
                dst[0:_pmax, h_tile_idx, bank_idx, 0 : transposes_per_bank * _pmax],
                psum_buffer[:_pmax, : transposes_per_bank * _pmax],
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
    use_fp32_transpose: bool = False,
    sbm=None,
    in_quant_recip: Optional[nl.NkiTensor] = None,
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
        compute_hidden_index_vector(inps, buffers, block_idx, dims, skip_dma, is_block_idx_dynamic)
        if use_dma_transpose:
            load_hidden_states_mx(inps, dims, skip_dma, token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                                  block_hidden_states_T=buffers.block_hidden_states_T, use_dma_transpose=True)
        else:
            load_hidden_states_mx(inps, dims, skip_dma, token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                                  block_hidden_states=buffers.block_hidden_states, use_dma_transpose=False)
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims)

        reshape buffers.hidden_qtz_sb to [128, n_H512_tile, B//32, 32]
        reshape buffers.hidden_scale_sb to [128, n_H512_tile, B//32, 32]
        quantize_block_hidden_state_T(buffers, prj_cfg, dims)
    """
    # Both DMA and PE paths consume buffers.token_4_H_indices_on_p via vector_offset.
    compute_hidden_index_vector(inps, buffers, block_idx, dims, kernel_cfg.skip_dma, is_block_idx_dynamic, sbm=sbm)
    if use_dma_transpose:
        load_hidden_states_mx(
            inps,
            dims,
            kernel_cfg.skip_dma,
            token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
            block_hidden_states_T=buffers.block_hidden_states_T,
            use_dma_transpose=True,
            sbm=sbm,
        )
    else:
        load_hidden_states_mx(
            inps,
            dims,
            kernel_cfg.skip_dma,
            token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
            block_hidden_states=buffers.block_hidden_states,
            use_dma_transpose=False,
            sbm=sbm,
        )
        sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims, sbm=sbm)

    buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
    if not kernel_cfg.is_static_quant:
        buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # Quantize block_hidden_states_T → hidden_qtz_sb. STATIC_MX uses software tensor_scalar
    # (saturation cast handles the fp8 clip); MX uses HW quantize_mx.
    if kernel_cfg.is_static_quant:
        kernel_assert(in_quant_recip is not None, "STATIC_MX path requires caller-supplied in_quant_recip")
        quantize_block_hidden_state_T_static_mx(buffers, prj_cfg, dims, in_quant_recip)
    else:
        quantize_block_hidden_state_T(buffers, prj_cfg, dims)


def _generate_expert_index_vector(
    expert_index: nl.NkiTensor,
    dst_idx_vector: nl.NkiTensor,
    scale_factor: int,
    n_quadrants_needed: int,
    n_remaining_partition: int = 0,
    name_prefix: str = "eiv",
    sbm=None,
    dst_int32: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """
    Generate padded index vector for expert-based DGE scale loading.

    Constructs a partition-dimension index vector that maps expert indices to
    scale tensor rows. Indices are scattered across quadrants (32-element blocks)
    with -1 padding for unused positions.

    Args:
        expert_index (nl.NkiTensor): Expert index scalar, shape [1, 1].
        dst_idx_vector (nl.NkiTensor): Destination buffer for index vector, shape [128, 1].
        scale_factor (int): Multiplier for expert index (typically 16 for E*16 folding).
        n_quadrants_needed (int): Number of full 32-element quadrants to fill.
        n_remaining_partition (int): Extra partitions beyond full quadrants (0-3).

    Returns:
        nl.NkiTensor: Index vector in SBUF with shape [128, 1], dtype int32.

    Notes:
        - Output pattern for expert=0: [0,1,2,3,-1...,-1, 4,5,6,7,-1...,-1, ...]
        - Output pattern for expert=3: [48,49,50,51,-1..., 52,53,54,55,-1..., ...]
        - Remainder partitions are placed at quadrant n_quadrants_needed
    """
    n_quadrants = n_quadrants_needed + (n_remaining_partition > 0)
    kernel_assert(n_quadrants <= 4, f"n_quadrants must be <= 4, got {n_quadrants}")
    kernel_assert(n_remaining_partition < 4, f"n_remaining_partition must be < 4, got {n_remaining_partition}")

    arange_f = _sbm_alloc(
        sbm, (1, n_quadrants * 4), dtype=nl.int32, name=f"{name_prefix}_arange_f", align=SBUF_QUADRANT_SIZE
    )
    nisa.iota(dst=arange_f, pattern=[[1, n_quadrants * 4]])

    padded_index = _sbm_alloc(
        sbm, (1, _pmax), dtype=nl.float32, name=f"{name_prefix}_padded_index", align=SBUF_QUADRANT_SIZE
    )
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

    padded_index_psum = _psum_alloc((_pmax, 1), nl.float32, sbm)
    nisa.nc_transpose(dst=padded_index_psum, data=padded_index)
    nisa.tensor_copy(dst=dst_idx_vector, src=padded_index_psum)

    token_indices_on_p = (
        dst_int32
        if dst_int32 != None
        else _sbm_alloc(
            sbm,
            dst_idx_vector.shape,
            dtype=nl.int32,
            name=f"{name_prefix}_token_indices_on_p",
            align=SBUF_QUADRANT_SIZE,
        )
    )
    nisa.tensor_copy(dst=token_indices_on_p, src=dst_idx_vector, engine=nisa.scalar_engine)
    return token_indices_on_p


def apply_clamp(tensor, upper_limit, lower_limit):
    """Apply optional upper and/or lower clamping to a tensor in-place. No-op if both limits are None."""
    if upper_limit != None and lower_limit != None:
        nisa.tensor_scalar(tensor, tensor, op0=nl.minimum, operand0=upper_limit, op1=nl.maximum, operand1=lower_limit)
    elif upper_limit != None:
        nisa.tensor_scalar(tensor, tensor, op0=nl.minimum, operand0=upper_limit)
    elif lower_limit != None:
        nisa.tensor_scalar(tensor, tensor, op0=nl.maximum, operand0=lower_limit)
