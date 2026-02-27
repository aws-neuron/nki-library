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
from typing import Any, Optional

import nki
import nki.language as nl

from ..utils.allocator import SbufManager

# NKI Library
from ..utils.common_types import NormType, QKVOutputLayout, QKVWeightLayout, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import (
    get_max_positive_value_for_dtype,
    get_program_sharding_info,
)
from ..utils.logging import get_logger

_logger = get_logger("qkv_cte")


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
    # --- FP8 KV Cache Quantization Related
    k_cache: Optional[nl.ndarray]
    v_cache: Optional[nl.ndarray]
    k_scale: Optional[nl.ndarray]
    v_scale: Optional[nl.ndarray]
    fp8_max: Optional[float]
    fp8_min: Optional[float]
    kv_dtype: Optional[Any]
    # --- Block KV Cache Related
    use_block_kv: bool
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


# Represent quantization config
@dataclass
class QKV_Quant_Config(nl.NKIObject):
    quantization_type: QuantizationType
    qkv_w_scale: Optional[nl.ndarray] = None  # weight quant scale for qkv projection
    qkv_in_scale: Optional[nl.ndarray] = None  # in_scale are same for q, k, v
    quant_dtype: str = nl.float8_e4m3
    has_mx_static_dequant_scales: bool = False


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
    add_bias: bool
    fused_residual_add: bool
    fused_norm_type: NormType
    add_layer_norm_bias: bool
    fused_rope: bool
    use_auto_allocation: bool  # functional
    # FP8 KV Cache Quantization
    use_kv_quantization: bool
    kv_dtype: Any
    fp8_max: Optional[float]
    fp8_min: Optional[float]
    # Block KV Cache
    use_block_kv: bool
    block_size: Optional[int]
    # Additional Internal Config
    load_input_with_DMA_transpose: bool
    compute_mm_dtype: Any
    act_dtype: Any  # Used for activations in normalization.
    psum_transpose_dtype: Any  # On >=Trn2, PE array supports BF16 transpose.
    use_BxS_input_reshape: bool  # Collapse B and S to BxS for performance.
    total_available_sbuf_space_to_this_kernel: int  # If SbufManger is provided, we need to restrict it.
    input_dtype: Any
    quantization_config: QKV_Quant_Config
    is_input_swizzled: bool

    def print(self):
        """
        Print all data members of the QKV_CTE_Config class.
        Useful for Debug
        """
        print(f"")
        print("QKV_CTE_Config Data Members:")
        print("User Requested:")
        print(f"  output_layout:        {self.output_layout}")
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
    _H, I = args.fused_qkv_weights.shape
    _, num_shards, _ = get_program_sharding_info()

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
    kernel_assert(
        I <= 4096,
        f"[QKV CTE Kernel] weights.shape[1] must be <= 4096, but got {I}."
        f" Kernel matrix multiplication is optimized for performance for I <= 4096, and does not provide support for "
        f" larger weights.shape[1] at the moment.",
    )

    if args.quantization_type != QuantizationType.MX:
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
        args.bias == None or args.bias.shape == (1, I),
        f"[QKV CTE Kernel] Bias shape must be [1, fused_qkv_weights.shape[1] = heads * d_head ],"
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
        kernel_assert(
            args.cos_cache.shape == args.sin_cache.shape == (B, S, d_head),
            f"[QKV CTE Kernel] cos_cache and sin_cache must have the shape of (B, S, d_head),"
            f" but got cos_cache.shape = {args.cos_cache.shape}, sin_cache.shape = {args.sin_cache.shape}.",
        )

    if args.cos_cache != None or args.sin_cache != None:
        kernel_assert(
            args.fused_rope == True,
            f"[QKV CTE Kernel] cos_cache or sin_cache have been provided, but fused_rope=False.",
        )

    use_kv_quantization = args.k_scale is not None and args.v_scale is not None
    if use_kv_quantization:
        kernel_assert(
            args.output_layout == QKVOutputLayout.BSD,
            f"[QKV CTE Kernel] KV output quantization is only supported for BSD output layout, got {args.output_layout}",
        )
        kernel_assert(
            args.num_q_heads is not None and args.num_kv_heads is not None and args.d_head is not None,
            "[QKV CTE Kernel] Must specify num_q_heads, num_kv_heads, and d_head when KV quantization is enabled",
        )
        kernel_assert(
            args.k_cache is not None and args.v_cache is not None,
            "[QKV CTE Kernel] k_cache and v_cache must be specified for KV output quantization",
        )
        q_dim = args.num_q_heads * args.d_head
        kv_dim = args.num_kv_heads * args.d_head
        kernel_assert(
            I == q_dim + 2 * kv_dim,
            f"[QKV CTE Kernel] fused_qkv_dim {I} must equal q_dim {q_dim} + 2 * kv_dim {2 * kv_dim}",
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
    if args.quantization_type == QuantizationType.MX:
        kernel_assert(
            _H == H // 4,
            f"[QKV CTE Kernel] Hidden dimensions of 'input' must be 4 * 'fused_qkv_weights' when weights are in MX,"
            f" input.shape[2] = {H}, but fused_qkv_weights[0] = {_H}).",
        )

        kernel_assert(
            H % 512 == 0,
            f"[QKV CTE Kernel] Hidden dimensions of 'input' must be divisible by 512, Hidden shape = {H}.",
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
    if args.quantization_type == QuantizationType.MX and args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]:
        kernel_assert(
            args.fused_norm_type == NormType.NO_NORM,
            f"[QKV CTE Kernel] FP8 input with MX quantization does not support normalization.",
        )
        kernel_assert(
            args.fused_residual_add == False,
            f"[QKV CTE Kernel] FP8 input with MX quantization does not support fused residual add.",
        )

    # Weight layout validation
    if args.quantization_type == QuantizationType.MX:
        _is_fp8_input = args.input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
        _has_dequant = args.qkv_in_scale != None and args.weight_layout == QKVWeightLayout.MX_INTERLEAVED
        _use_dma_xpose = (_is_fp8_input or (not _is_fp8_input and _has_dequant)) and args.load_input_with_DMA_transpose
        if _use_dma_xpose:
            kernel_assert(
                args.weight_layout == QKVWeightLayout.MX_INTERLEAVED,
                f"[QKV CTE Kernel] DMA transpose MX path requires MX_INTERLEAVED weight layout, got {args.weight_layout}.",
            )
        else:
            kernel_assert(
                args.weight_layout == QKVWeightLayout.MX_CONTIGUOUS,
                f"[QKV CTE Kernel] Non-DMA-transpose MX path requires MX_CONTIGUOUS weight layout, got {args.weight_layout}.",
            )
    else:
        kernel_assert(
            args.weight_layout == QKVWeightLayout.CONTIGUOUS,
            f"[QKV CTE Kernel] Non-MX quantization requires CONTIGUOUS weight layout, got {args.weight_layout}.",
        )

    # MX static dequantization scales validation
    _is_mx_static_dequant = (
        args.quantization_type == QuantizationType.MX
        and args.weight_layout == QKVWeightLayout.MX_INTERLEAVED
        and args.qkv_in_scale != None
    )
    if _is_mx_static_dequant:
        kernel_assert(
            args.qkv_w_scale is not None,
            f"[QKV CTE Kernel] qkv_w_scale must be provided when qkv_in_scale is set with MX_INTERLEAVED layout.",
        )
        kernel_assert(
            args.qkv_in_scale.shape == (1, 1) or args.qkv_in_scale.shape == (nl.tile_size.pmax, 1),
            f"[QKV CTE Kernel] qkv_in_scale shape must be [1, 1] or [128, 1], got {args.qkv_in_scale.shape}.",
        )
        kernel_assert(
            args.qkv_w_scale.shape == (1, 3) or args.qkv_w_scale.shape == (nl.tile_size.pmax, 3),
            f"[QKV CTE Kernel] qkv_w_scale shape must be [1, 3] or [128, 3] for MX static dequant, got {args.qkv_w_scale.shape}.",
        )
        kernel_assert(
            args.d_head != None and args.num_q_heads != None and args.num_kv_heads != None,
            f"[QKV CTE Kernel] d_head, num_q_heads, and num_kv_heads must be specified when MX static dequant scales are provided.",
        )

    # Quantization validation
    if args.quantization_type == QuantizationType.STATIC:
        kernel_assert(
            args.fused_qkv_weights.dtype == nl.float8_e4m3 or str(args.fused_qkv_weights.dtype) == "float8e4",
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
    load_input_with_DMA_transpose = (
        args.load_input_with_DMA_transpose
        and nki.isa.get_nc_version() >= nki.isa.nc_version.gen3
        and (args.fused_norm_type == NormType.NO_NORM)
        and args.fused_residual_add == False
        and (_is_2byte_input or (_is_fp8_input and args.quantization_type == QuantizationType.MX))
    )

    if _is_fp8_input and args.quantization_type == QuantizationType.MX:
        if load_input_with_DMA_transpose:
            _logger.info("QKV CTE: Using FP8 MX DMA transpose path (pre-scaled FP8 input with neutral MX input scales)")
        else:
            _logger.info(
                "QKV CTE: FP8 MX input detected but DMA transpose disabled — falling back to PE transpose path"
            )

    _has_mx_static_dequant = (
        args.quantization_type == QuantizationType.MX
        and args.weight_layout == QKVWeightLayout.MX_INTERLEAVED
        and args.qkv_in_scale != None
    )
    if _is_2byte_input and _has_mx_static_dequant and args.quantization_type == QuantizationType.MX:
        if load_input_with_DMA_transpose:
            _logger.info("QKV CTE: Using BF16 MX DMA transpose path (pre-scaled BF16 input with static dequant scales)")
        else:
            _logger.info(
                "QKV CTE: BF16 MX input with static dequant scales but DMA transpose disabled — falling back to PE transpose path"
            )

    if args.quantization_type == QuantizationType.STATIC:
        compute_mm_dtype = nl.float8_e4m3
    else:
        # Compute dtype used in the kernel. Even if inputs are fp32 or fp8, computations will be done with bf16.
        compute_mm_dtype = nl.bfloat16 if (args.input.dtype == nl.float32 or _is_fp8_input) else args.input.dtype

    act_dtype = nl.float32

    # Instances after >=Trn2 support BF16 transpose mode on PE Array.
    if args.quantization_type == QuantizationType.STATIC:
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

    # Set to maximum, unless we are restricted by the provided 'sbm'.
    total_available_sbuf_space_to_this_kernel = nl.tile_size.total_available_sbuf_size
    if args.sbm != None:
        # In auto_allocation mode, "sbm.get_free_space" does not work (if user provides sbm with auto_alloc set to True).
        if args.use_auto_allocation:
            total_available_sbuf_space_to_this_kernel = nl.tile_size.total_available_sbuf_size
        else:
            total_available_sbuf_space_to_this_kernel = args.sbm.get_free_space()

    # FP8 KV Cache Quantization config
    use_kv_quantization = args.k_scale is not None and args.v_scale is not None
    kv_dtype = args.kv_dtype if args.kv_dtype is not None else args.input.dtype

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
    quant_config.has_mx_static_dequant_scales = (
        args.quantization_type == QuantizationType.MX
        and args.weight_layout == QKVWeightLayout.MX_INTERLEAVED
        and args.qkv_in_scale != None
    )

    return QKV_CTE_Config(
        output_layout=args.output_layout,
        add_bias=add_bias,
        fused_residual_add=args.fused_residual_add,
        fused_norm_type=args.fused_norm_type,
        add_layer_norm_bias=add_layer_norm_bias,
        fused_rope=args.fused_rope,
        use_auto_allocation=args.use_auto_allocation,
        use_kv_quantization=use_kv_quantization,
        kv_dtype=kv_dtype,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        use_block_kv=args.use_block_kv,
        block_size=args.block_size,
        # Internal Config
        load_input_with_DMA_transpose=load_input_with_DMA_transpose,
        compute_mm_dtype=compute_mm_dtype,
        act_dtype=act_dtype,
        psum_transpose_dtype=psum_transpose_dtype,
        use_BxS_input_reshape=use_BxS_input_reshape,
        total_available_sbuf_space_to_this_kernel=total_available_sbuf_space_to_this_kernel,
        input_dtype=args.input.dtype,
        quantization_config=quant_config,
        is_input_swizzled=args.is_input_swizzled,
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
    BxS = B_orig * S_orig
    _, I = args.fused_qkv_weights.shape
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
