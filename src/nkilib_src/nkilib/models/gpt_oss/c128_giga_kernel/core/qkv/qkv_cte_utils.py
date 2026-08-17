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

"""
This module contains utility classes and functions for the QKV CTE kernel including configuration dataclasses, dimension management, and input validation.
"""

# Standard Library
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import nki
import nki.language as nl

from ..utils.allocator import SbufManager

# NKI Library
from ..utils.common_types import (
    DtypeMode,
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
    StridedInputConfig,
)
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import (
    get_max_positive_value_for_dtype,
    get_program_sharding_info,
    resolve_fp8_e4m3_dtype,
)
from ..utils.logging import get_logger

_logger = get_logger("qkv_cte")

ROW_MX_TAIL_SCALE_BYTES = 4  # float32 per-row scale appended to each input row


# Represents unmodified user inputs, no additional data members.
# Used for initial processing for input validation, and to build
# QKV_CTE_Config and QKV_CTE_Dims.
@dataclass
class QKV_CTE_UserInput(nl.NKIObject):
    """
    Container for unmodified user inputs to QKV CTE kernel.

    This dataclass captures all user-provided parameters without modification.
    Used for initial input validation and to construct QKV_CTE_Config and QKV_CTE_Dims objects.

    Attributes:
        input (nl.ndarray): [B, S, H], Input hidden states tensor
        fused_qkv_weights (nl.ndarray): [H, I], Fused QKV weight matrix
        output_layout (QKVOutputLayout): Desired output tensor layout
        bias (Optional[nl.ndarray]): [1, I], Optional bias tensor
        fused_residual_add (Optional[bool]): Whether to perform residual addition
        mlp_prev (Optional[nl.ndarray]): [B, S, H], Previous MLP output for residual
        attention_prev (Optional[nl.ndarray]): [B, S, H], Previous attention output for residual
        fused_norm_type (NormType): Type of normalization to apply
        gamma_norm_weights (Optional[nl.ndarray]): [1, H], Normalization gamma weights
        layer_norm_bias (Optional[nl.ndarray]): [1, H], Layer norm beta weights
        norm_eps (Optional[float]): Epsilon for normalization stability
        hidden_actual (Optional[int]): Actual hidden dimension if H is padded
        fused_rope (Optional[bool]): Whether to apply RoPE
        cos_cache (Optional[nl.ndarray]): [B, S, d_head], RoPE cosine cache
        sin_cache (Optional[nl.ndarray]): [B, S, d_head], RoPE sine cache
        d_head (Optional[int]): Dimension per attention head
        num_q_heads (Optional[int]): Number of query heads
        num_kv_heads (Optional[int]): Number of key/value heads
        store_output_in_sbuf (bool): Whether to store output in SBUF
        sbm (Optional[SbufManager]): Optional SBUF manager
        use_auto_allocation (bool): Whether to use automatic SBUF allocation
        load_input_with_DMA_transpose (bool): Whether to use DMA transpose
        quantization_type (QuantizationType): Quantization type for QKV projection
        qkv_w_scale (Optional[nl.ndarray]): Quantization scale for QKV weights
        qkv_in_scale (Optional[nl.ndarray]): Quantization scale for QKV input
        is_input_swizzled (bool): If input tensor is swizzled for MX
        weight_layout (QKVWeightLayout): Layout of fused_qkv_weights
    """

    input: nl.ndarray
    fused_qkv_weights: nl.ndarray
    output_layout: QKVOutputLayout
    # -- Bias
    bias: Optional[nl.ndarray]
    # -- Fused Residual Add
    fused_residual_add: Optional[bool]
    mlp_prev: Optional[nl.ndarray]
    attention_prev: Optional[nl.ndarray]
    # --- Fused Norm Related
    fused_norm_type: NormType
    gamma_norm_weights: Optional[nl.ndarray]
    layer_norm_bias: Optional[nl.ndarray]
    norm_eps: Optional[float]
    hidden_actual: Optional[int]
    # --- Fused RoPE Related
    fused_rope: Optional[bool]
    cos_cache: Optional[nl.ndarray]
    sin_cache: Optional[nl.ndarray]
    d_head: Optional[int]
    num_q_heads: Optional[int]
    num_kv_heads: Optional[int]
    # --- KV Cache Related
    k_cache: Optional[nl.ndarray]
    v_cache: Optional[nl.ndarray]
    k_scale: Optional[nl.ndarray]
    v_scale: Optional[nl.ndarray]
    fp8_max: Optional[float]
    fp8_min: Optional[float]
    kv_dtype: Optional[Any]
    # --- Block KV Cache Related
    use_block_kv: bool
    transpose_k_cache: bool
    fp8_packed: bool
    block_size: Optional[int]
    slot_mapping: Optional[nl.ndarray]
    # --- Performance Related
    store_output_in_sbuf: bool
    sbm: Optional[SbufManager]
    use_auto_allocation: bool
    load_input_with_DMA_transpose: bool
    # --- Quantization Related
    quantization_type: QuantizationType
    qkv_w_scale: Optional[nl.ndarray]
    qkv_in_scale: Optional[nl.ndarray]
    is_input_swizzled: bool
    weight_layout: QKVWeightLayout
    # --- QK-Norm Related
    qk_norm_pre_rope: Optional[QKNormConfig]
    qk_norm_post_rope: Optional[QKNormConfig]
    # --- Strided Input
    strided_input_config: Optional[StridedInputConfig]
    output_hbm: Optional[nl.ndarray]
    # --- Optional per-segment sum-of-squares outputs (each [B, S, 1])
    q_squared_sum_out: Optional[nl.ndarray] = None
    k_squared_sum_out: Optional[nl.ndarray] = None
    v_squared_sum_out: Optional[nl.ndarray] = None
    # --- FP8 E4M3 dtype mode. See DtypeMode enum.
    dtype_mode: DtypeMode = DtypeMode.NON_OCP


# Represent quantization config
@dataclass
class QKV_Quant_Config(nl.NKIObject):
    quantization_type: QuantizationType
    qkv_w_scale: Optional[nl.ndarray] = None  # weight quant scale for qkv projection
    qkv_in_scale: Optional[nl.ndarray] = None  # in_scale are same for q, k, v
    quant_dtype: Optional[Any] = None  # Follow _fp8_e4m3_dtype for fp8 and fused_qkv_weights.dtype for other cases
    has_mx_static_dequant_scales: bool = False
    has_row_mx_dequant: bool = False
    has_mx_native_row_input_dequant: bool = False


# Represents kernel config.
@dataclass
class QKV_CTE_Config(nl.NKIObject):
    """
    Kernel configuration for QKV CTE.

    Contains both user-requested configuration and internally-derived settings
    that control kernel behavior, data types, and optimizations.

    Attributes:
        output_layout (QKVOutputLayout): User-requested output tensor layout
        add_bias (bool): Whether to add bias to QKV projection
        fused_residual_add (bool): Whether to perform residual addition
        fused_norm_type (NormType): Type of normalization to apply
        add_layer_norm_bias (bool): Whether to add layer norm bias
        fused_rope (bool): Whether to apply RoPE
        use_auto_allocation (bool): Whether to use automatic SBUF allocation
        load_input_with_DMA_transpose (bool): Whether to use DMA transpose for input loading
        compute_mm_dtype (Any): Data type for matrix multiplication computation
        act_dtype (Any): Data type for activations in normalization
        psum_transpose_dtype (Any): Data type for PE array transpose (BF16 on >=Trn2)
        use_BxS_input_reshape (bool): Whether to collapse B and S to BxS for performance
        total_available_sbuf_space_to_this_kernel (int): Total SBUF space available per partition
    """

    # User Requested
    output_layout: QKVOutputLayout
    weight_layout: QKVWeightLayout
    add_bias: bool
    fused_residual_add: bool
    fused_norm_type: NormType
    add_layer_norm_bias: bool
    fused_rope: bool
    use_auto_allocation: bool  # functional
    # KV Cache
    use_kv_cache: bool
    use_kv_quantization: bool
    kv_dtype: Any
    fp8_max: Optional[float]
    fp8_min: Optional[float]
    # Block KV Cache
    use_block_kv: bool
    transpose_k_cache: bool
    fp8_packed: bool
    block_size: Optional[int]
    # Additional Internal Config
    load_input_with_DMA_transpose: bool
    compute_mm_dtype: Any
    act_dtype: Any  # Used for activations in normalization.
    weights_dtype: Any  # Used for the matrix multiplication
    psum_transpose_dtype: Any  # On >=Trn2, PE array supports BF16 transpose.
    use_BxS_input_reshape: bool  # Collapse B and S to BxS for performance.
    total_available_sbuf_space_to_this_kernel: int  # If SbufManger is provided, we need to restrict it.
    input_dtype: Any
    quantization_config: QKV_Quant_Config
    is_input_swizzled: bool
    # QK-Norm
    qk_norm_pre_rope: Optional[QKNormConfig]
    qk_norm_post_rope: Optional[QKNormConfig]
    # Strided Input
    strided_input_config: Optional[StridedInputConfig]

    def print(self):
        """
        Print all data members of the QKV_CTE_Config class.
        Useful for Debug
        """
        print(f"")
        print("QKV_CTE_Config Data Members:")
        print("User Requested:")
        print(f"  output_layout:        {self.output_layout}")
        print(f"  weight_layout         {self.weight_layout}")
        print(f"  add_bias:             {self.add_bias}")
        print(f"  fused_residual_add:   {self.fused_residual_add}")
        print(f"  fused_norm_type:      {self.fused_norm_type}")
        print(f"  add_layer_norm_bias:  {self.add_layer_norm_bias}")
        print(f"  fused_rope:           {self.fused_rope}")
        print(f"  use_auto_allocation:  {self.use_auto_allocation}")
        print("Additional Internal Config:")
        print(f"  load_input_with_DMA_transpose: {self.load_input_with_DMA_transpose}")
        print(f"  compute_mm_dtype:     {self.compute_mm_dtype}")
        print(f"  act_dtype:            {self.act_dtype}")
        print(f"  psum_transpose_dtype:  {self.psum_transpose_dtype}")
        print(f"  use_BxS_input_reshape: {self.use_BxS_input_reshape}")
        print(f"  total_available_sbuf_space_to_this_kernel {self.total_available_sbuf_space_to_this_kernel}")
        print(f"  quantization_config:  {self.quantization_config}")
        print(f"")


# Represents tensor dimensions of input tensors.
@dataclass
class QKV_CTE_Dims(nl.NKIObject):
    """
    Tensor dimensions for QKV CTE kernel.

    Stores all dimension-related information including original tensor shapes,
    potentially reshaped dimensions, sharding information, and tiling parameters.

    Attributes:
        B_orig (int): Original batch size before any reshaping
        S_orig (int): Original sequence length before any reshaping
        BxS (int): Product B_orig * S_orig (used if use_BxS_input_reshape is True)
        B (int): Batch size used by kernel (could be 1 if reshaped)
        S (int): Sequence length used by kernel (could be BxS if reshaped)
        S_shard (int): Chunk of S each LNC core processes
        S_shard_offset (int): Offset into S for current shard
        H (int): Hidden dimension
        I (int): Fused QKV dimension = (num_q_heads + 2*num_kv_heads) * d_head
        H_actual (Optional[int]): Actual hidden dimension if H is padded
        d_head (Optional[int]): Dimension per attention head
        num_q_heads (Optional[int]): Number of query heads
        num_kv_heads (Optional[int]): Number of key/value heads
        num_heads (Optional[int]): Total number of heads
        num_128_tiles_per_H (int): Number of 128-sized tiles in H dimension
        num_512_tiles_per_H (int): Number of 512-sized tiles in H dimension
        num_512_tiles_per_I (int): Number of 512-sized tiles in I dimension
        NUM_WEIGHT_BUFFERS_DEFAULT (int): Default number of weight buffers for multi-buffering
        WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT (int): Default block size for loading weights along H
        MAX_S_MULTI_BUFFER_DEGREE (int): Maximum multi-buffering degree for sequence dimension
    """

    B_orig: int  # Batch Size of the orginal input tensor, non-reshaped.
    S_orig: int  # Sequence Length of the orginal input tensor, non-reshaped.
    BxS: int  # B_orig * S_orig (may or not be used)
    B: int  # Batch Size used by kernel implementation. Could be 1, if use_BxS reshape is True.
    S: int  # Sequence Length used by kernel implementation. Could be BxS, if use_BxS reshape is True.
    S_shard: int  # Chunk of S each lnc-core is processing.
    S_shard_offset: int
    H: int  # Hidden Dimension
    I: int  # Fused QKV Dimension = heads * num_heads, 2nd dimension of the weight matrix.
    H_actual: Optional[int]
    d_head: Optional[int]
    num_q_heads: Optional[int]
    num_kv_heads: Optional[int]
    num_heads: Optional[int]
    # -- QKV Dimension Info (for KV quantization) -- #
    q_dim: Optional[int]  # num_q_heads * d_head
    kv_dim: Optional[int]  # num_kv_heads * d_head
    # -- Additional Tile Info -- #
    num_128_tiles_per_H: int
    num_512_tiles_per_H: int
    num_512_tiles_per_I: int
    # Weights SBUF Related #
    NUM_WEIGHT_BUFFERS_DEFAULT: int  # If we do not prefetch weights, we use multi-buffering.
    WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT: int  # If we do not prefetch weights, we load them in BLOCKS, e.g. 1024 per H.
    # S Tiling Related
    MAX_S_MULTI_BUFFER_DEGREE: int  # Depending on SBUF space, try to multi-buffer as much, up to this constant.

    def print(self):
        """
        Print all data members of the QKV_CTE_Dims class.
        Useful for Debug
        """
        print(f"")
        print("QKV_CTE_Dims Data Members:")
        print(f"  B_orig:               {self.B_orig}")
        print(f"  S_orig:               {self.S_orig}")
        print(f"  BxS:                  {self.BxS}")
        print(f"  B:                    {self.B}")
        print(f"  S:                    {self.S}")
        print(f"  S_shard:              {self.S_shard}")
        print(f"  S_shard_offset:       {self.S_shard_offset}")
        print(f"  H:                    {self.H}")
        print(f"  I:                    {self.I}")
        print(f"  H_actual:             {self.H_actual}")
        print(f"  d_head:               {self.d_head}")
        print(f"  num_q_heads:          {self.num_q_heads}")
        print(f"  num_kv_heads:         {self.num_kv_heads}")
        print(f"  num_heads:            {self.num_heads}")
        print(f"  num_128_tiles_per_H:   {self.num_128_tiles_per_H}")
        print(f"  num_512_tiles_per_H:   {self.num_512_tiles_per_H}")
        print(f"  num_512_tiles_per_I:   {self.num_512_tiles_per_I}")
        print(f"  NUM_WEIGHT_BUFFERS_DEFAULT: {self.NUM_WEIGHT_BUFFERS_DEFAULT}")
        print(f"  WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT: {self.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT}")
        print(f"  MAX_S_MULTI_BUFFER_DEGREE: {self.MAX_S_MULTI_BUFFER_DEGREE}")
        print(f"")


def _is_enum_like(x):
    """Duck-type check for an Enum member: has str .name and int .value.

    Used to validate ``QKNormConfig.q_norm``/``k_norm``. A strict
    ``isinstance(x, NormType)`` would reject custom enum types registered by
    callers, so we only require the enum-member shape (.name: str, .value:
    int) and let the norm registry reject unknown members downstream.

    Also avoids ``isinstance(x, Enum)`` directly because the NKI tracer can't
    resolve the ``enum.Enum`` name when this validator runs inside a traced
    kernel.
    """
    return hasattr(x, "name") and isinstance(x.name, str) and hasattr(x, "value") and isinstance(x.value, int)


def _validate_qk_norm_config(args, config_name, config):
    """
    Validate norm func values, beta rejection, and gamma/beta weight shapes for a single QKNormConfig.
    """
    if config is None:
        return
    kernel_assert(
        config.q_norm is None or _is_enum_like(config.q_norm),
        f"[QKV CTE Kernel] {config_name}.q_norm must be an Enum or None, got {type(config.q_norm)}.",
    )
    kernel_assert(
        config.k_norm is None or _is_enum_like(config.k_norm),
        f"[QKV CTE Kernel] {config_name}.k_norm must be an Enum or None, got {type(config.k_norm)}.",
    )
    kernel_assert(
        config.q_beta_norm_weights is None and config.k_beta_norm_weights is None,
        f"[QKV CTE Kernel] {config_name} beta weights are not yet implemented.",
    )
    if config.q_gamma_norm_weights is not None:
        kernel_assert(
            config.q_gamma_norm_weights.shape == (1, args.d_head),
            f"[QKV CTE Kernel] {config_name}.q_gamma_norm_weights must have shape"
            f" (1, d_head)=(1, {args.d_head}), but got {config.q_gamma_norm_weights.shape}.",
        )
    if config.k_gamma_norm_weights is not None:
        kernel_assert(
            config.k_gamma_norm_weights.shape == (1, args.d_head),
            f"[QKV CTE Kernel] {config_name}.k_gamma_norm_weights must have shape"
            f" (1, d_head)=(1, {args.d_head}), but got {config.k_gamma_norm_weights.shape}.",
        )


def _validate_row_mx_inputs(
    args: QKV_CTE_UserInput,
    H: int,
    I: int,
    S: int,
    num_shards: int,
    _H: int,
) -> None:
    """Validate ROW_MX-specific input constraints.

    Args:
        args (QKV_CTE_UserInput): User inputs.
        H (int): True hidden dimension (after tail-scale removal).
        I (int): Output dimension (Q+K+V heads * d_head).
        S (int): Sequence length.
        num_shards (int): Number of LNC shards.
        _H (int): Weight first dimension (H // 4 for MX-packed weights).
    """
    kernel_assert(
        _H == H // 4,
        f"[QKV CTE Kernel] Hidden dimensions of 'input' must be 4 * 'fused_qkv_weights' when weights are in MX,"
        f" input.shape[2] = {H}, but fused_qkv_weights[0] = {_H}).",
    )
    kernel_assert(
        (S // num_shards) % 2 == 0,
        f"[QKV CTE Kernel] S_Shard needs to be even for mx matrix multiplication,S_shard = {S // num_shards}.",
    )
    kernel_assert(
        args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn],
        f"[QKV CTE Kernel] ROW_MX requires FP8 input, got {args.input.dtype}.",
    )
    kernel_assert(
        args.qkv_in_scale is None,
        "[QKV CTE Kernel] ROW_MX packs per-row scales in the input tensor; qkv_in_scale must be None.",
    )
    kernel_assert(
        args.qkv_w_scale is not None,
        "[QKV CTE Kernel] ROW_MX requires qkv_w_scale (per-channel weight scale).",
    )
    kernel_assert(
        args.qkv_w_scale.shape == (1, I) or args.qkv_w_scale.shape == (nl.tile_size.pmax, I),
        f"[QKV CTE Kernel] ROW_MX qkv_w_scale must be [1, I] or [128, I], got {args.qkv_w_scale.shape}.",
    )
    kernel_assert(
        args.weight_layout == QKVWeightLayout.MX_CONTIGUOUS,
        f"[QKV CTE Kernel] ROW_MX requires MX_CONTIGUOUS weight layout "
        f"(float32 DMA transpose uses consecutive-4 packing), got {args.weight_layout}.",
    )
    kernel_assert(
        args.load_input_with_DMA_transpose,
        "[QKV CTE Kernel] ROW_MX requires load_input_with_DMA_transpose=True (no PE transpose fallback).",
    )
    kernel_assert(
        args.fused_norm_type == NormType.NO_NORM,
        f"[QKV CTE Kernel] ROW_MX does not support fused normalization.",
    )
    kernel_assert(
        args.fused_residual_add == False,
        f"[QKV CTE Kernel] ROW_MX does not support fused residual add.",
    )
    kernel_assert(
        not args.is_input_swizzled,
        f"[QKV CTE Kernel] ROW_MX does not support swizzled input.",
    )
    kernel_assert(
        args.d_head is not None and args.num_q_heads is not None and args.num_kv_heads is not None,
        "[QKV CTE Kernel] ROW_MX requires d_head, num_q_heads, and num_kv_heads to be specified.",
    )


def _validate_user_inputs(args: QKV_CTE_UserInput):
    """
    Validate all user inputs to the QKV CTE kernel.

    Performs comprehensive validation of tensor shapes, dimensions, and configuration
    parameters to ensure they meet kernel requirements and are mutually consistent.

    Args:
        args (QKV_CTE_UserInput): Container with all user-provided inputs

    Raises:
        AssertionError: If any validation check fails with descriptive error message

    Notes:
        - H must be <= 24576 and divisible by 128
        - I must be <= 4096
        - Validates consistency between input/weight shapes
        - Validates fused operation requirements (residual add, normalization, RoPE)
        - Validates output layout requirements
    """

    B, S, H = args.input.shape
    _, num_shards, _ = get_program_sharding_info()

    # MX weights: 3D [H//4, I, 4] with unpacked fp8 dtype
    if args.quantization_type.is_mx():
        kernel_assert(
            len(args.fused_qkv_weights.shape) == 3,
            f"[QKV CTE Kernel] MX weights must be 3D [H//4, I, 4], got shape {args.fused_qkv_weights.shape}.",
        )
        _H, I, _pack_dim = args.fused_qkv_weights.shape
        kernel_assert(
            _pack_dim == 4,
            f"[QKV CTE Kernel] MX weights must have innermost dimension of 4, got {_pack_dim}.",
        )
        kernel_assert(
            args.fused_qkv_weights.dtype in [nl.float8_e4m3, nl.float8_e4m3fn],
            f"[QKV CTE Kernel] MX weights must have fp8 dtype, got {args.fused_qkv_weights.dtype}.",
        )
    else:
        _H, I = args.fused_qkv_weights.shape

    # ROW_MX: input is [B, S, H + ROW_MX_TAIL_SCALE_BYTES] with tail-packed float32 scale. Derive true H.
    if args.quantization_type == QuantizationType.ROW_MX:
        kernel_assert(
            H > ROW_MX_TAIL_SCALE_BYTES and (H - ROW_MX_TAIL_SCALE_BYTES) % 128 == 0,
            f"[QKV CTE Kernel] ROW_MX input last dimension must be H+{ROW_MX_TAIL_SCALE_BYTES} where H is a"
            f" multiple of 128, but got input.shape[-1]={H} (H={H - ROW_MX_TAIL_SCALE_BYTES}).",
        )
        H = H - ROW_MX_TAIL_SCALE_BYTES

    # Native MX + row-quantized FP8 input: same tail-packed format as ROW_MX.
    _is_mx_native_row_input = (
        args.quantization_type == QuantizationType.MX
        and args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
        and H == _H * 4 + ROW_MX_TAIL_SCALE_BYTES
    )
    if _is_mx_native_row_input:
        H = H - ROW_MX_TAIL_SCALE_BYTES

    # H
    kernel_assert(
        H <= 24576,
        f"[QKV CTE Kernel] Hidden dimension must be <= 24576, but got {H}."
        f" Kernel may go out of space for larger hidden dimensions",
    )

    kernel_assert(
        H % 128 == 0,
        f"[QKV CTE Kernel] Hidden dimension must be a multiple of 128, but got {H}."
        f" Limitation of the current kernel implementation. ",
    )

    # I
    if not args.quantization_type.is_mx():
        kernel_assert(
            I <= 4096,
            f"[QKV CTE Kernel] weights.shape[1] must be <= 4096, but got {I}."
            f" Kernel matrix multiplication is optimized for performance for I <= 4096, and does not provide support for "
            f" larger weights.shape[1] at the moment.",
        )

    if not args.quantization_type.is_mx():
        kernel_assert(
            _H == H,
            f"[QKV CTE Kernel] Hidden dimensions of 'input' and 'fused_qkv_weights' must match,"
            f" input.shape[2] = {H}, but fused_qkv_weights[0] = {_H}).",
        )

    # Validate Output Layout.
    kernel_assert(
        args.output_layout == QKVOutputLayout.BSD or args.output_layout == QKVOutputLayout.NBSd,
        f"[QKV CTE Kernel] Unsupported output layout, output_layout must be 'QKVOutputLayout.BSD' or 'QKVOutputLayout.NBSd',"
        f" but got output_layout = {args.output_layout}.",
    )

    if args.output_layout == QKVOutputLayout.NBSd:
        kernel_assert(
            args.d_head != None,
            f"[QKV CTE Kernel] For NBSd output_layout, d_head must be specified (and must be =128), but got d_head = {args.d_head}.",
        )
        kernel_assert(
            args.d_head == 128,
            f"[QKV CTE Kernel] For NBSd output_layout, d_head=128 is only supported at the moment, but got d_head = {args.d_head}.",
        )

    # Bias Validation.
    kernel_assert(
        args.bias == None or args.bias.shape == (1, I) or args.bias.shape == (nl.tile_size.pmax, I),
        f"[QKV CTE Kernel] Bias shape must be [1, I] or [pmax, I] where I=fused_qkv_weights.shape[1]={I},"
        f" but got {args.bias.shape if args.bias != None else args.bias}",
    )

    # Fused Residual Add Validation.
    if args.fused_residual_add:
        kernel_assert(
            args.mlp_prev != None and args.attention_prev != None,
            f"[QKV CTE Kernel] Fused residual add requires both mlp_prev and attention_prev to be provided.",
        )
        kernel_assert(
            args.mlp_prev.shape == args.attention_prev.shape == args.input.shape,
            f"[QKV CTE Kernel] Fused residual add requires mlp_prev, attention_prev and input to have the same shape,"
            f" but got args.input.shape = {args.input.shape}, mlp_prev.shape = {args.mlp_prev.shape}, attention_prev.shape = {args.attention_prev.shape}.",
        )
    if args.mlp_prev != None or args.attention_prev != None:
        kernel_assert(
            args.fused_residual_add == True,
            f"[QKV CTE Kernel] mlp_prev or attention_prev provided without setting fused_residual_add to True.",
        )

    # Fused Normalization Validation.
    # Note: NormType.RMS_NORM_SKIP_GAMMA, does not require Gamma tensor.
    if (args.fused_norm_type == NormType.RMS_NORM) or (args.fused_norm_type == NormType.LAYER_NORM):
        kernel_assert(
            args.gamma_norm_weights != None,
            f"[QKV CTE Kernel] Fused normalization requires gamma_norm_weights to be provided.",
        )
        kernel_assert(
            args.gamma_norm_weights.shape == (1, H),
            f"[QKV CTE Kernel] Fused normalization requires gamma_norm_weights to be of shape (1, H),"
            f" but got gamma_norm_weights.shape = {args.gamma_norm_weights.shape}.",
        )

    if (
        (args.fused_norm_type == NormType.RMS_NORM)
        or (args.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA)
        or (args.fused_norm_type == NormType.LAYER_NORM)
    ):
        kernel_assert(
            args.norm_eps != None,
            f"[QKV CTE Kernel] Fused normalization requires norm_eps to be provided.",
        )

    if args.layer_norm_bias != None:
        kernel_assert(
            args.fused_norm_type == NormType.LAYER_NORM,
            f"[QKV CTE Kernel] Beta normalization bias is only supported for fused LAYER_NORM.",
        )
        kernel_assert(
            args.layer_norm_bias.shape == (1, H),
            f"[QKV CTE Kernel] Layer norm bias must be of shape (1, H), but got layer_norm_bias.shape = {args.layer_norm_bias.shape}.",
        )

    if args.gamma_norm_weights != None:
        kernel_assert(
            args.fused_norm_type != NormType.NO_NORM,
            f"[QKV CTE Kernel] gamma_norm_weights are provided, but requested fused normalization (RMSNorm or LayerNorm).",
        )

        kernel_assert(
            args.fused_norm_type != NormType.RMS_NORM_SKIP_GAMMA,
            f"[QKV CTE Kernel] gamma_norm_weights are provided, but fused_norm_type is RMS_NORM_SKIP_GAMMA.",
        )

    if args.hidden_actual != None:
        kernel_assert(
            args.hidden_actual <= H,
            f"[QKV CTE Kernel] hidden_actual is expected to be <= H ( e.g. H is infered from (potentially) padded tensors),"
            f" but got hidden_actual = {args.hidden_actual}, H = {H}.",
        )

    # Fused RoPE Validation, and heads validation.
    if args.d_head != None and args.num_q_heads != None and args.num_kv_heads != None:
        kernel_assert(
            (args.num_q_heads + 2 * args.num_kv_heads) * args.d_head == I,
            f"[QKV CTE Kernel] (num_q_heads + 2 * num_kv_heads)*d_head must equal to fused_qkv_weights.shape[1].",
        )

    if args.fused_rope:
        kernel_assert(
            args.cos_cache != None
            and args.sin_cache != None
            and args.num_q_heads != None
            and args.num_kv_heads != None,
            f"[QKV CTE Kernel] Fused RoPE requires cos_cache, sin_cache, num_q_heads and num_kv_heads to be provided.",
        )
        d_head = args.d_head if args.d_head != None else I // (args.num_q_heads + 2 * args.num_kv_heads)
        # Under strided input, cos/sin caches must be pre-gathered to the output S
        # (= num_local_tokens); otherwise they match the full input S.
        expected_S = args.strided_input_config.num_local_tokens if args.strided_input_config != None else S
        kernel_assert(
            args.cos_cache.shape == args.sin_cache.shape == (B, expected_S, d_head),
            f"[QKV CTE Kernel] cos_cache and sin_cache must have the shape of (B, S, d_head)"
            f" where S = {expected_S}"
            f"{' (= strided num_local_tokens)' if args.strided_input_config != None else ''},"
            f" but got cos_cache.shape = {args.cos_cache.shape},"
            f" sin_cache.shape = {args.sin_cache.shape}.",
        )

    if args.cos_cache != None or args.sin_cache != None:
        kernel_assert(
            args.fused_rope == True,
            f"[QKV CTE Kernel] cos_cache or sin_cache have been provided, but fused_rope=False.",
        )

    use_kv_cache = args.k_cache is not None or args.v_cache is not None
    use_kv_quantization = args.k_scale is not None and args.v_scale is not None
    if use_kv_cache:
        kernel_assert(
            args.k_cache is not None and args.v_cache is not None,
            "[QKV CTE Kernel] Both k_cache and v_cache must be provided together.",
        )
        kernel_assert(
            args.output_layout == QKVOutputLayout.BSD,
            f"[QKV CTE Kernel] KV cache is only supported for BSD output layout, got {args.output_layout}",
        )
        kernel_assert(
            args.num_q_heads is not None and args.num_kv_heads is not None and args.d_head is not None,
            "[QKV CTE Kernel] Must specify num_q_heads, num_kv_heads, and d_head when KV cache is enabled",
        )
        q_dim = args.num_q_heads * args.d_head
        kv_dim = args.num_kv_heads * args.d_head
        kernel_assert(
            I == q_dim + 2 * kv_dim,
            f"[QKV CTE Kernel] fused_qkv_dim {I} must equal q_dim {q_dim} + 2 * kv_dim {2 * kv_dim}",
        )

    kernel_assert(
        not args.fp8_packed or args.use_block_kv,
        "[QKV CTE Kernel] fp8_packed requires use_block_kv=True.",
    )

    if args.use_block_kv:
        kernel_assert(
            args.slot_mapping is not None,
            "[QKV CTE Kernel] slot_mapping required for block KV cache",
        )
        kernel_assert(
            args.block_size is not None,
            "[QKV CTE Kernel] block_size required for block KV cache",
        )
        if args.transpose_k_cache == True:
            kernel_assert(
                args.d_head <= 128,
                f"[QKV CTE Kernel] transpose_k_cache requires d_head <= 128 (NKI partition dimension limit), but got d_head = {args.d_head}.",
            )
            kernel_assert(
                (args.block_size & (args.block_size - 1)) == 0,
                f"[QKV CTE Kernel] transpose_k_cache requires power-of-2 block_size for block kv cache, got {args.block_size}",
            )
        if args.fp8_packed:
            kernel_assert(
                args.k_cache is not None,
                "[QKV CTE Kernel] fp8_packed requires k_cache to be provided.",
            )
            kernel_assert(
                not args.transpose_k_cache,
                "[QKV CTE Kernel] fp8_packed is mutually exclusive with transpose_k_cache.",
            )
            kernel_assert(
                args.k_scale is not None and args.v_scale is not None,
                "[QKV CTE Kernel] fp8_packed requires FP8 KV quantization (k_scale and v_scale must be provided).",
            )
            kernel_assert(
                args.block_size % 2 == 0,
                f"[QKV CTE Kernel] fp8_packed requires even block_size, got {args.block_size}.",
            )
            kernel_assert(
                args.d_head <= 128,
                f"[QKV CTE Kernel] fp8_packed requires d_head <= 128 (NKI partition dimension limit), but got d_head = {args.d_head}.",
            )
            # fp8_packed pairs consecutive tokens, so every tile must have an even token count.
            _S = args.input.shape[1]
            kernel_assert(
                _S % 2 == 0,
                f"[QKV CTE Kernel] fp8_packed requires S to be even, got S = {_S}.",
            )

    kernel_assert(
        args.store_output_in_sbuf == False,
        f"[QKV CTE Kernel] store_output_in_sbuf is unsupported in the CTE version of qkv kernel.",
    )

    if args.sbm != None:
        kernel_assert(
            args.sbm.is_auto_alloc() == args.use_auto_allocation,
            f"[QKV CTE Kernel] If SbufManager is provided then args.sbm.is_auto_alloc() == args.use_auto_allocation, however"
            f" use_auto_allocation = {args.use_auto_allocation}, but sbm.is_auto_alloc = {args.sbm.is_auto_alloc()}",
        )
    # MX Quantization and is_input_swizzled validation
    if args.quantization_type == QuantizationType.ROW_MX:
        _validate_row_mx_inputs(args, H, I, S, num_shards, _H)
    elif args.quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX):
        kernel_assert(
            _H == H // 4,
            f"[QKV CTE Kernel] Hidden dimensions of 'input' must be 4 * 'fused_qkv_weights' when weights are in MX,"
            f" input.shape[2] = {H}, but fused_qkv_weights[0] = {_H}).",
        )

        if args.is_input_swizzled:
            kernel_assert(
                H % 512 == 0,
                f"[QKV CTE Kernel] Hidden dimensions of 'input' must be divisible by 512 for swizzled MX input, Hidden shape = {H}.",
            )

        kernel_assert(
            (S // num_shards) % 2 == 0,
            f"[QKV CTE Kernel] S_Shard needs to be even for mx matrix multiplication,S_shard = {S // num_shards}.",
        )

    if args.is_input_swizzled:
        kernel_assert(
            args.quantization_type == QuantizationType.MX,
            f"[QKV CTE Kernel] is_input_swizzled is only supported for MX Quantization.",
        )
        kernel_assert(
            args.fused_norm_type == NormType.NO_NORM,
            f"[QKV CTE Kernel] is_input_swizzled does not support input Normalization.",
        )
        kernel_assert(
            args.fused_residual_add == False,
            f"[QKV CTE Kernel] is_input_swizzled does not support fused residual add.",
        )

    # FP8 input validation for MX path
    if args.quantization_type in (
        QuantizationType.MX,
        QuantizationType.STATIC_MX,
        QuantizationType.STATIC,
    ) and args.input.dtype in [
        nl.float8_e4m3,
        nl.float8_e4m3fn,
    ]:
        kernel_assert(
            args.fused_norm_type == NormType.NO_NORM,
            f"[QKV CTE Kernel] FP8 input with MX quantization does not support normalization.",
        )
        kernel_assert(
            args.fused_residual_add == False,
            f"[QKV CTE Kernel] FP8 input with MX quantization does not support fused residual add.",
        )

    # Native MX + FP8 input with row-quantized tail-packed scale:
    # input is [B, S, H+4] where last 4 bytes per row encode a float32 per-row dequant scale.
    if _is_mx_native_row_input:
        kernel_assert(
            not args.is_input_swizzled,
            "[QKV CTE Kernel] Native MX with row-quantized FP8 input does not support swizzled input.",
        )
        kernel_assert(
            args.qkv_in_scale is None,
            "[QKV CTE Kernel] Native MX with row-quantized FP8 input packs per-row scales in the input tensor; qkv_in_scale must be None.",
        )
        kernel_assert(
            args.load_input_with_DMA_transpose,
            "[QKV CTE Kernel] Native MX with row-quantized FP8 input requires load_input_with_DMA_transpose=True.",
        )
        kernel_assert(
            args.weight_layout == QKVWeightLayout.MX_CONTIGUOUS,
            f"[QKV CTE Kernel] Native MX with row-quantized FP8 input requires MX_CONTIGUOUS weight layout, got {args.weight_layout}.",
        )
        kernel_assert(
            args.d_head is not None and args.num_q_heads is not None and args.num_kv_heads is not None,
            "[QKV CTE Kernel] Native MX with row-quantized FP8 input requires d_head, num_q_heads, and num_kv_heads.",
        )

    # Weight layout validation
    if args.quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX):
        _is_fp8_input = args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
        _is_static_mx = args.quantization_type == QuantizationType.STATIC_MX
        _use_dma_xpose = (_is_fp8_input or _is_static_mx) and args.load_input_with_DMA_transpose
        if _use_dma_xpose:
            kernel_assert(
                args.weight_layout in (QKVWeightLayout.MX_INTERLEAVED, QKVWeightLayout.MX_CONTIGUOUS),
                f"[QKV CTE Kernel] DMA transpose MX path requires MX_INTERLEAVED or MX_CONTIGUOUS weight layout, got {args.weight_layout}.",
            )
        else:
            kernel_assert(
                args.weight_layout == QKVWeightLayout.MX_CONTIGUOUS,
                f"[QKV CTE Kernel] Non-DMA-transpose MX path requires MX_CONTIGUOUS weight layout, got {args.weight_layout}.",
            )
    else:
        if args.quantization_type != QuantizationType.ROW_MX:
            kernel_assert(
                args.weight_layout == QKVWeightLayout.CONTIGUOUS,
                f"[QKV CTE Kernel] Non-MX quantization requires CONTIGUOUS weight layout, got {args.weight_layout}.",
            )

    # STATIC_MX dequantization scales validation
    if args.quantization_type == QuantizationType.STATIC_MX:
        kernel_assert(
            args.qkv_in_scale is not None,
            f"[QKV CTE Kernel] qkv_in_scale must be provided for STATIC_MX quantization.",
        )
        kernel_assert(
            args.qkv_w_scale is not None,
            f"[QKV CTE Kernel] qkv_w_scale must be provided for STATIC_MX quantization.",
        )
        kernel_assert(
            args.qkv_in_scale.shape == (1, 1) or args.qkv_in_scale.shape == (nl.tile_size.pmax, 1),
            f"[QKV CTE Kernel] qkv_in_scale shape must be [1, 1] or [128, 1], got {args.qkv_in_scale.shape}.",
        )
        kernel_assert(
            args.qkv_w_scale.shape == (1, 3) or args.qkv_w_scale.shape == (nl.tile_size.pmax, 3),
            f"[QKV CTE Kernel] qkv_w_scale shape must be [1, 3] or [128, 3] for STATIC_MX, got {args.qkv_w_scale.shape}.",
        )
        kernel_assert(
            args.d_head != None and args.num_q_heads != None and args.num_kv_heads != None,
            f"[QKV CTE Kernel] d_head, num_q_heads, and num_kv_heads must be specified for STATIC_MX quantization.",
        )

    # Quantization validation
    if args.quantization_type == QuantizationType.STATIC:
        kernel_assert(
            args.fused_qkv_weights.dtype in (nl.float8_e4m3, nl.float8_e4m3fn)
            or str(args.fused_qkv_weights.dtype) in ("float8e4", "float8_e4m3fn"),
            f"[QKV CTE Kernel] When quantization_type is STATIC, currently only fp8 is supported as the qkv_weights dtype, "
            f"but got dtype={args.fused_qkv_weights.dtype}.",
        )
        kernel_assert(
            args.qkv_w_scale is not None and args.qkv_in_scale is not None,
            f"[QKV CTE Kernel] When quantization_type is STATIC, both qkv_w_scale and qkv_in_scale must be provided, "
            f"but got qkv_w_scale={args.qkv_w_scale}, qkv_in_scale={args.qkv_in_scale}.",
        )
        kernel_assert(
            args.num_q_heads is not None and args.num_kv_heads is not None,
            f"[QKV CTE Kernel] When quantization_type is STATIC, both num_q_heads and num_kv_heads must be specified, "
            f"but got num_q_heads={args.num_q_heads}, num_kv_heads={args.num_kv_heads}.",
        )

    # QK-norm validation
    if args.qk_norm_pre_rope is not None or args.qk_norm_post_rope is not None:
        kernel_assert(
            args.d_head is not None and args.num_q_heads is not None and args.num_kv_heads is not None,
            "[QKV CTE Kernel] QK-norm requires d_head, num_q_heads, and num_kv_heads to be specified.",
        )

    _validate_qk_norm_config(args, "qk_norm_pre_rope", args.qk_norm_pre_rope)
    _validate_qk_norm_config(args, "qk_norm_post_rope", args.qk_norm_post_rope)

    # Squared-sum output validation
    _any_squared_sum = (
        args.q_squared_sum_out is not None or args.k_squared_sum_out is not None or args.v_squared_sum_out is not None
    )
    if _any_squared_sum:
        kernel_assert(
            args.output_layout in (QKVOutputLayout.BSD, QKVOutputLayout.NBSd),
            "[QKV CTE Kernel] q/k/v_squared_sum_out are only supported with the BSD or NBSd output layout, "
            f"but got {args.output_layout}.",
        )
        kernel_assert(
            args.quantization_type not in (QuantizationType.MX, QuantizationType.ROW_MX),
            "[QKV CTE Kernel] q/k/v_squared_sum_out are not supported with MX quantization.",
        )
        kernel_assert(
            args.k_cache is None and args.v_cache is None,
            "[QKV CTE Kernel] q/k/v_squared_sum_out are not supported together with the KV-cache path.",
        )
        kernel_assert(
            args.d_head is not None and args.num_q_heads is not None and args.num_kv_heads is not None,
            "[QKV CTE Kernel] q/k/v_squared_sum_out require d_head, num_q_heads, and num_kv_heads to be specified.",
        )
        if args.q_squared_sum_out is not None:
            kernel_assert(
                args.q_squared_sum_out.shape == (B, S, 1),
                f"[QKV CTE Kernel] q_squared_sum_out must have shape [B, S, 1] = {(B, S, 1)}, "
                f"but got {args.q_squared_sum_out.shape}.",
            )
        if args.k_squared_sum_out is not None:
            kernel_assert(
                args.k_squared_sum_out.shape == (B, S, 1),
                f"[QKV CTE Kernel] k_squared_sum_out must have shape [B, S, 1] = {(B, S, 1)}, "
                f"but got {args.k_squared_sum_out.shape}.",
            )
        if args.v_squared_sum_out is not None:
            kernel_assert(
                args.v_squared_sum_out.shape == (B, S, 1),
                f"[QKV CTE Kernel] v_squared_sum_out must have shape [B, S, 1] = {(B, S, 1)}, "
                f"but got {args.v_squared_sum_out.shape}.",
            )

    # gamma_fused_in_rope_caches validation
    if args.qk_norm_pre_rope is not None and args.qk_norm_pre_rope.gamma_fused_in_rope_caches:
        kernel_assert(
            args.fused_rope,
            "[QKV CTE Kernel] qk_norm_pre_rope.gamma_fused_in_rope_caches=True requires fused_rope=True.",
        )
        kernel_assert(
            args.qk_norm_pre_rope.q_norm == NormType.RMS_NORM and args.qk_norm_pre_rope.k_norm == NormType.RMS_NORM,
            "[QKV CTE Kernel] gamma_fused_in_rope_caches requires both q_norm and k_norm to be NormType.RMS_NORM.",
        )
        kernel_assert(
            args.qk_norm_pre_rope.q_gamma_norm_weights is None and args.qk_norm_pre_rope.k_gamma_norm_weights is None,
            "[QKV CTE Kernel] qk_norm_pre_rope.gamma_fused_in_rope_caches=True requires"
            " q_gamma_norm_weights=None and k_gamma_norm_weights=None"
            " (gamma must be pre-multiplied into cos/sin caches, not passed separately).",
        )
    if args.qk_norm_post_rope is not None:
        kernel_assert(
            not args.qk_norm_post_rope.gamma_fused_in_rope_caches,
            "[QKV CTE Kernel] gamma_fused_in_rope_caches is only valid on qk_norm_pre_rope, not qk_norm_post_rope.",
        )

    # Strided Input validation.
    if args.strided_input_config != None:
        _validate_strided_input_config(args)


def _validate_strided_input_config(args: QKV_CTE_UserInput, dims: Optional[QKV_CTE_Dims] = None):
    """
    Validate StridedInputConfig constraints.

    Cross-checks strided_input_config against other user inputs. Called from
    _validate_user_inputs when args.strided_input_config != None, and again
    from qkv_cte after dims are computed (with dims= set) to validate
    S-shard alignment.
    """
    si = args.strided_input_config
    _, S_full, _ = args.input.shape

    # block_len must divide pmax (packed regime: multiple blocks per DMA tile)
    # OR be a multiple of pmax (contiguous-tile regime: one block spans multiple DMA tiles).
    kernel_assert(
        si.block_len > 0 and (nl.tile_size.pmax % si.block_len == 0 or si.block_len % nl.tile_size.pmax == 0),
        f"[QKV CTE Kernel] strided_input_config.block_len must divide pmax ({nl.tile_size.pmax})"
        f" or be a multiple of pmax ({nl.tile_size.pmax}), but got block_len={si.block_len}.",
    )
    # block_stride must be a whole number of blocks, >= block_len (no overlap)
    kernel_assert(
        si.block_stride >= si.block_len and si.block_stride % si.block_len == 0,
        f"[QKV CTE Kernel] strided_input_config.block_stride must be >= block_len and divisible by block_len,"
        f" but got block_stride={si.block_stride}, block_len={si.block_len}.",
    )
    # num_local_tokens must be a whole number of blocks
    kernel_assert(
        si.num_local_tokens > 0 and si.num_local_tokens % si.block_len == 0,
        f"[QKV CTE Kernel] strided_input_config.num_local_tokens must be a positive multiple of block_len,"
        f" but got num_local_tokens={si.num_local_tokens}, block_len={si.block_len}.",
    )
    # Last block must be within input S bounds
    num_blocks = si.num_local_tokens // si.block_len
    last_block_end = si.block_offset + (num_blocks - 1) * si.block_stride + si.block_len
    kernel_assert(
        si.block_offset >= 0 and last_block_end <= S_full,
        f"[QKV CTE Kernel] strided_input_config reads beyond input S: block_offset={si.block_offset},"
        f" num_blocks={num_blocks}, block_stride={si.block_stride}, block_len={si.block_len},"
        f" last_block_end={last_block_end}, input S={S_full}.",
    )
    """
    Caller-provided output_hbm required for the standard output path (output S != input S).
    KV cache path is an exception: the kernel allocates q_tensor_hbm itself (sized with
    num_local_tokens), and k_cache/v_cache are caller-provided.
    """
    _uses_kv_cache = args.k_cache is not None or args.v_cache is not None
    if not _uses_kv_cache:
        kernel_assert(
            args.output_hbm != None,
            "[QKV CTE Kernel] strided_input_config requires a caller-provided output_hbm"
            " (output S = num_local_tokens differs from input S).",
        )
    # Output layout
    kernel_assert(
        args.output_layout in (QKVOutputLayout.BSD, QKVOutputLayout.NBSd),
        f"[QKV CTE Kernel] strided_input_config supports output_layout BSD or NBSd, got {args.output_layout}.",
    )
    # Incompatible fusions
    kernel_assert(
        not args.fused_residual_add,
        "[QKV CTE Kernel] strided_input_config is incompatible with fused_residual_add"
        " (mlp_prev/attention_prev are not strided).",
    )
    # Requires DMA transpose input load path
    kernel_assert(
        args.load_input_with_DMA_transpose,
        "[QKV CTE Kernel] strided_input_config requires load_input_with_DMA_transpose=True.",
    )
    # DMA transpose path is only used when fused_norm_type == NO_NORM (see _build_config).
    # Other norm types fall back to the PE-transpose input load which does not implement
    # strided gather.
    kernel_assert(
        args.fused_norm_type == NormType.NO_NORM,
        f"[QKV CTE Kernel] strided_input_config requires fused_norm_type=NO_NORM"
        f" (DMA-transpose path only), got {args.fused_norm_type}.",
    )
    # Non-MX quantization only for now (scoped; MX path is a separate DMA-transpose helper)
    kernel_assert(
        args.quantization_type in (QuantizationType.NONE, QuantizationType.STATIC),
        f"[QKV CTE Kernel] strided_input_config currently supports only non-MX quantization"
        f" (NONE, STATIC), got {args.quantization_type}.",
    )
    # Validate output_hbm shape matches the chosen layout with S = num_local_tokens.
    # Skipped for KV-cache path (kernel allocates q_tensor_hbm; output_hbm is unused).
    if not _uses_kv_cache:
        B_orig = args.input.shape[0]
        I = args.fused_qkv_weights.shape[1]
        if args.output_layout == QKVOutputLayout.BSD:
            expected_out_shape = (B_orig, si.num_local_tokens, I)
        else:  # NBSd
            # num_heads is derived the same way as _get_tensor_dimensions: I // d_head.
            # d_head is validated elsewhere to be set for NBSd.
            num_heads = I // args.d_head
            expected_out_shape = (num_heads, B_orig, si.num_local_tokens, args.d_head)
        kernel_assert(
            tuple(args.output_hbm.shape) == expected_out_shape,
            f"[QKV CTE Kernel] strided_input_config requires output_hbm.shape == {expected_out_shape}"
            f" for output_layout={args.output_layout}, got {tuple(args.output_hbm.shape)}.",
        )

    # S-shard alignment: requires dims, so only checked on the second call (after _get_tensor_dimensions).
    if dims is not None:
        _bl = si.block_len
        kernel_assert(
            dims.S_shard_offset % _bl == 0 and dims.S_shard % _bl == 0,
            f"[QKV CTE Kernel] strided_input_config requires block-aligned S window,"
            f" got S_shard_offset={dims.S_shard_offset}, S_shard={dims.S_shard},"
            f" block_len={_bl}.",
        )


def _build_config(args: QKV_CTE_UserInput) -> QKV_CTE_Config:
    """
    Build QKV_CTE_Config object from validated user inputs.

    Constructs kernel configuration by deriving internal settings from user inputs,
    including data types, optimization flags, and memory allocation parameters.

    Args:
        args (QKV_CTE_UserInput): Validated user inputs

    Returns:
        QKV_CTE_Config: Kernel configuration object with all settings

    Notes:
        - Assumes user inputs have already been validated via _validate_user_inputs
        - Determines compute dtype (converts fp32 to bf16)
        - Decides whether to use DMA transpose based on hardware and fusion settings
        - Determines whether to reshape B,S to BxS for performance
        - Sets available SBUF space based on provided SbufManager or uses maximum
    """
    _, S, _ = args.input.shape

    add_bias = args.bias != None
    add_layer_norm_bias = args.layer_norm_bias != None

    # Load input with transpose when the three conditions are all met:
    # If the input is load_input_with_DMA_transpose=True, but these conditions are not met, use PE transpose instead.
    #  - 1. The hardware architecture is trn2 or higher; trn1 lacks support for this feature.
    #  - 2. fusedAdd is disabled, as it requires DMA FMA mode that is not supported yet with DMA transpose.
    #  - 3. There is no normalization, as normalization requires input to be in non-transposed layout.
    #  - 4. The input dtype is 2-byte dtype, i.e. BF16/FP16, or FP8 with MX quantization
    _is_fp8_input = args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
    _is_2byte_input = args.input.dtype == nl.bfloat16 or args.input.dtype == nl.float16
    _is_row_mx = args.quantization_type == QuantizationType.ROW_MX
    _is_static_mx = args.quantization_type == QuantizationType.STATIC_MX
    _is_mx_or_static_mx = args.quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX)
    load_input_with_DMA_transpose = (
        args.load_input_with_DMA_transpose
        and nki.isa.get_nc_version() >= nki.isa.nc_version.gen3
        and (args.fused_norm_type == NormType.NO_NORM)
        and args.fused_residual_add == False
        and (_is_2byte_input or (_is_fp8_input and (_is_mx_or_static_mx or _is_row_mx)))
    )

    if _is_fp8_input and _is_mx_or_static_mx:
        if load_input_with_DMA_transpose:
            _logger.info("QKV CTE: Using FP8 MX DMA transpose path (pre-scaled FP8 input with neutral MX input scales)")
        else:
            _logger.info(
                "QKV CTE: FP8 MX input detected but DMA transpose disabled — falling back to PE transpose path"
            )

    if _is_2byte_input and _is_static_mx:
        if load_input_with_DMA_transpose:
            _logger.info(
                "QKV CTE: Using BF16 STATIC_MX DMA transpose path (pre-scaled BF16 input with static dequant scales)"
            )
        else:
            _logger.info("QKV CTE: BF16 STATIC_MX input but DMA transpose disabled — falling back to PE transpose path")

    # Compute dtype used in the kernel. Even if inputs are fp32 or fp8, computations will be done with bf16.
    compute_mm_dtype = nl.bfloat16 if (args.input.dtype == nl.float32 or _is_fp8_input) else args.input.dtype

    act_dtype = nl.float32

    _fp8_e4m3_dtype = resolve_fp8_e4m3_dtype(args.dtype_mode)
    if args.quantization_type == QuantizationType.STATIC:
        weight_dtype_str = str(args.fused_qkv_weights.dtype)
        weights_dtype = _fp8_e4m3_dtype if weight_dtype_str == "float8e4" else args.fused_qkv_weights.dtype
    else:
        weights_dtype = compute_mm_dtype

    # Instances after >=Trn2 support BF16 transpose mode on PE Array.
    if args.quantization_type == QuantizationType.STATIC:
        if _is_fp8_input:
            psum_transpose_dtype = _fp8_e4m3_dtype
        else:
            psum_transpose_dtype = nl.bfloat16 if nki.isa.get_nc_version() >= nki.isa.nc_version.gen3 else nl.float32
    else:
        psum_transpose_dtype = compute_mm_dtype if nki.isa.get_nc_version() >= nki.isa.nc_version.gen3 else nl.float32

    # Decide whether to reshape input [B, S, H] -> [BxS, H] for the performance benefits, High batch tests benfit -30% or more.
    # If S is small, reshaping will increase the blocking/multi-buffering degree in the kernel,
    # and give better performance due to better allocation (especially if args.use_auto_allocation == False).
    # Note #1: S_THRESHOLD_FOR_RESHAPE_DEFAULT was derived empirically because of few regressed tests.
    #           This should be revisted, maybe it is not required anymore. Large S won't get much perf benefit anyways.
    # Note #2: Even if the threshold requirement is removed, it important to keep "use_BxS_input_reshape" in the config,
    #          as some output_layouts like "NBdS" (potentially added in future), cannot be reshaped.
    _, num_shards, _ = get_program_sharding_info()
    S_THRESHOLD_FOR_RESHAPE_DEFAULT = (5 * 128) * num_shards
    use_BxS_input_reshape = (S < S_THRESHOLD_FOR_RESHAPE_DEFAULT) and (
        args.output_layout == QKVOutputLayout.BSD or args.output_layout == QKVOutputLayout.NBSd
    )  # In case "NBdS" gets added.
    # Strided input requires output S != input S; BxS reshape is invalid.
    if args.strided_input_config != None:
        use_BxS_input_reshape = False

    # Set to maximum, unless we are restricted by the provided 'sbm'.
    total_available_sbuf_space_to_this_kernel = nl.tile_size.total_available_sbuf_size
    if args.sbm != None:
        # In auto_allocation mode, "sbm.get_free_space" does not work (if user provides sbm with auto_alloc set to True).
        if args.use_auto_allocation:
            total_available_sbuf_space_to_this_kernel = nl.tile_size.total_available_sbuf_size
        else:
            total_available_sbuf_space_to_this_kernel = args.sbm.get_free_space()

    # KV Cache config
    use_kv_cache = args.k_cache is not None or args.v_cache is not None
    use_kv_quantization = args.k_scale is not None and args.v_scale is not None
    # FP8 KV cache: use caller's concrete dtype when explicit; resolve from
    # dtype_mode only for the opaque "float8e4" sentinel. bf16 and other
    # non-FP8 dtypes pass through unchanged.
    if args.kv_dtype is None:
        kv_dtype = args.input.dtype
    elif str(args.kv_dtype) == "float8e4":
        kv_dtype = resolve_fp8_e4m3_dtype(args.dtype_mode)
    else:
        kv_dtype = args.kv_dtype

    if use_kv_quantization:
        max_val = get_max_positive_value_for_dtype(kv_dtype)
        fp8_max = args.fp8_max if args.fp8_max is not None else max_val
        fp8_min = args.fp8_min if args.fp8_min is not None else -max_val
    else:
        fp8_max = None
        fp8_min = None

    # Set quantization config
    quant_config = QKV_Quant_Config(quantization_type=args.quantization_type)
    if args.quantization_type != QuantizationType.NONE:
        quant_config.qkv_w_scale = args.qkv_w_scale
        quant_config.qkv_in_scale = args.qkv_in_scale
    # Mirror the FP8 dtype onto the quant config for downstream sites.
    # Use the caller's concrete weight dtype when available; resolve from
    # dtype_mode only for the opaque "float8e4" sentinel.
    if args.quantization_type in (QuantizationType.STATIC, QuantizationType.ROW):
        weight_dtype_str = str(args.fused_qkv_weights.dtype)
        if weight_dtype_str == "float8e4":
            quant_config.quant_dtype = _fp8_e4m3_dtype
        else:
            quant_config.quant_dtype = args.fused_qkv_weights.dtype
    quant_config.has_mx_static_dequant_scales = args.quantization_type == QuantizationType.STATIC_MX
    quant_config.has_row_mx_dequant = args.quantization_type == QuantizationType.ROW_MX
    _H_weights = args.fused_qkv_weights.shape[0]
    quant_config.has_mx_native_row_input_dequant = (
        args.quantization_type == QuantizationType.MX
        and args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
        and args.input.shape[-1] == _H_weights * 4 + ROW_MX_TAIL_SCALE_BYTES
    )

    return QKV_CTE_Config(
        output_layout=args.output_layout,
        weight_layout=args.weight_layout,
        add_bias=add_bias,
        fused_residual_add=args.fused_residual_add,
        fused_norm_type=args.fused_norm_type,
        add_layer_norm_bias=add_layer_norm_bias,
        fused_rope=args.fused_rope,
        use_auto_allocation=args.use_auto_allocation,
        use_kv_cache=use_kv_cache,
        use_kv_quantization=use_kv_quantization,
        kv_dtype=kv_dtype,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        use_block_kv=args.use_block_kv,
        transpose_k_cache=args.transpose_k_cache,
        fp8_packed=args.fp8_packed,
        block_size=args.block_size,
        # Internal Config
        load_input_with_DMA_transpose=load_input_with_DMA_transpose,
        compute_mm_dtype=compute_mm_dtype,
        act_dtype=act_dtype,
        weights_dtype=weights_dtype,
        psum_transpose_dtype=psum_transpose_dtype,
        use_BxS_input_reshape=use_BxS_input_reshape,
        total_available_sbuf_space_to_this_kernel=total_available_sbuf_space_to_this_kernel,
        input_dtype=args.input.dtype,
        quantization_config=quant_config,
        is_input_swizzled=args.is_input_swizzled,
        qk_norm_pre_rope=args.qk_norm_pre_rope,
        qk_norm_post_rope=args.qk_norm_post_rope,
        strided_input_config=args.strided_input_config,
    )


def _get_tensor_dimensions(args: QKV_CTE_UserInput, cfg: QKV_CTE_Config) -> QKV_CTE_Dims:
    """
    Build QKV_CTE_Dims object containing all tensor dimension information.

    Extracts and computes tensor dimensions from user inputs and configuration,
    including sharding calculations for LNC parallelism and tiling parameters.

    Args:
        args (QKV_CTE_UserInput): Validated user inputs
        cfg (QKV_CTE_Config): Kernel configuration

    Returns:
        QKV_CTE_Dims: Object containing all dimension information

    Notes:
        - Assumes user inputs have already been validated
        - Handles B,S to BxS reshaping based on cfg.use_BxS_input_reshape
        - Computes S_shard and S_shard_offset for LNC sharding
        - Pre-calculates tiling parameters for H and I dimensions
        - Infers d_head from num_q_heads and num_kv_heads if not provided
    """
    B_orig, S_orig, H = args.input.shape
    # ROW_MX / native MX row input: input is [B, S, H + ROW_MX_TAIL_SCALE_BYTES] with tail-packed float32 scale.
    if args.quantization_type == QuantizationType.ROW_MX or cfg.quantization_config.has_mx_native_row_input_dequant:
        H = H - ROW_MX_TAIL_SCALE_BYTES
    BxS = B_orig * S_orig
    I = args.fused_qkv_weights.shape[1]
    H_actual = args.hidden_actual if args.hidden_actual else H
    d_head = args.d_head

    if args.num_q_heads != None and args.num_kv_heads != None:
        d_head_infer = I // (args.num_q_heads + 2 * args.num_kv_heads)
        if d_head == None:
            d_head = d_head_infer

    num_heads = I // d_head if d_head != None else None

    # For LNC1, num_shards = 1, shard_id = 0
    # For LNC2, num_shards = 2, shard_id = 0,1
    _, num_shards, shard_id = get_program_sharding_info()

    # When sharded, we only process a portion of S.
    # If LNC=1, S_shard == S.
    if cfg.use_BxS_input_reshape:
        S = BxS
        B = 1
    elif args.strided_input_config != None:
        # Strided input: kernel loop bounds and output sizing use num_local_tokens.
        # S_orig is preserved above for HBM offset calculations in the strided DMA path.
        S = args.strided_input_config.num_local_tokens
        B = B_orig
    else:
        S = S_orig
        B = B_orig

    S_shard_base = S // num_shards  # Size of S_shard for shard_id = 0.
    S_shard = S_shard_base
    if S % num_shards != 0 and shard_id == num_shards - 1:
        # If S cannot be evenly divided by num_shards (at most 2), we add 1 to the last shard.
        # Note: Right now LNC can only 2, but hypthetically if num_shards > 2, this logic would need adjustment
        S_shard = S // num_shards + 1

    # For sharded kernel, calculate S_shard_offset based on shard index.
    S_shard_offset = shard_id * S_shard_base

    num_128_tiles_per_H = math.ceil(H / 128)
    num_512_tiles_per_H = math.ceil(H / 512)
    num_512_tiles_per_I = math.ceil(I / 512)
    NUM_WEIGHT_BUFFERS_DEFAULT = 4
    WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT = 1024
    MAX_S_MULTI_BUFFER_DEGREE = 5

    # Compute q_dim and kv_dim for KV quantization
    q_dim = None
    kv_dim = None
    if args.num_q_heads is not None and args.num_kv_heads is not None and d_head is not None:
        q_dim = args.num_q_heads * d_head
        kv_dim = args.num_kv_heads * d_head

    return QKV_CTE_Dims(
        B_orig=B_orig,
        S_orig=S_orig,
        BxS=BxS,
        B=B,
        S=S,
        S_shard=S_shard,
        S_shard_offset=S_shard_offset,
        H=H,
        I=I,
        H_actual=H_actual,
        d_head=d_head,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        num_heads=num_heads,
        q_dim=q_dim,
        kv_dim=kv_dim,
        num_128_tiles_per_H=num_128_tiles_per_H,
        num_512_tiles_per_H=num_512_tiles_per_H,
        num_512_tiles_per_I=num_512_tiles_per_I,
        NUM_WEIGHT_BUFFERS_DEFAULT=NUM_WEIGHT_BUFFERS_DEFAULT,
        WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT=WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT,
        MAX_S_MULTI_BUFFER_DEGREE=MAX_S_MULTI_BUFFER_DEGREE,
    )


def _mx_partition_splits(h_packed: int) -> Tuple[List[int], List[int]]:
    """Split a partition count into valid nc_matmul_mx sizes (32, 64, or 128).

    Returns (starts, sizes) as two separate lists.
    For 96 (3 sub-tiles), returns ([0, 64], [64, 32]).

    h_packed is always min(P_MAX, H_PACKED - tile * P_MAX) where H_PACKED = H // 4
    and H % 128 == 0, so the only possible values are 32, 64, 96, and 128.
    """
    if h_packed == 32:
        return [0], [32]
    if h_packed == 64:
        return [0], [64]
    if h_packed == 96:
        return [0, 64], [64, 32]
    if h_packed == 128:
        return [0], [128]
    kernel_assert(False, f"h_packed must be 32, 64, 96, or 128, got {h_packed}")
