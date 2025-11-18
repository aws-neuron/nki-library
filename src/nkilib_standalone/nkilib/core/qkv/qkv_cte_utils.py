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
QKV_CTE kernel Utils

"""

# Standard Library
import math
from dataclasses import dataclass
from typing import Any, Optional

# Neuron Kernel Interface
import nki as nki
import nki.language as nl

# NKI Library
from ..utils.common_types import NormType, QKVOutputLayout
from ..utils.allocator import SbufManager
from ..utils.kernel_helpers import get_program_sharding_info, is_launched_as_spmd
from ..utils.kernel_assert import kernel_assert


# Represents unmodified user inputs, no additional data members.
# Used for initial processing for input validation, and to build
# QKV_CTE_Config and QKV_CTE_Dims.
@dataclass
class QKV_CTE_UserInput(nl.NKIObject):
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
    # --- Performance Related
    store_output_in_sbuf: bool
    sbm: Optional[SbufManager]
    use_auto_allocation: bool
    load_input_with_DMA_transpose: bool


# Represents kernel config.
@dataclass
class QKV_CTE_Config(nl.NKIObject):
    # User Requested
    output_layout: QKVOutputLayout
    add_bias: bool
    fused_residual_add: bool
    fused_norm_type: NormType
    add_layer_norm_bias: bool
    fused_rope: bool
    use_auto_allocation: bool  # functional
    # Additional Internal Config
    load_input_with_DMA_transpose: bool
    compute_mm_dtype: Any
    act_dtype: Any  # Used for activations in normalization.
    psum_transpose_dtype: Any  # On >=Trn2, PE array supports BF16 transpose.
    use_BxS_input_reshape: bool  # Collapse B and S to BxS for performance.
    total_available_sbuf_space_to_this_kernel: int  # If SbufManger is provided, we need to restrict it.

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
        print(f"  psum_transpose_dtype  {self.psum_transpose_dtype}")
        print(f"  use_BxS_input_reshape: {self.use_BxS_input_reshape}")
        print(f"  total_available_sbuf_space_to_this_kernel {self.total_available_sbuf_space_to_this_kernel}")
        print(f"")


# Represents tensor dimensions of input tensors.
@dataclass
class QKV_CTE_Dims(nl.NKIObject):
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
    Validate the inputs to the QKV_CTE kernel.
    """

    B, S, H = args.input.shape
    _H, I = args.fused_qkv_weights.shape

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


def _build_config(args: QKV_CTE_UserInput) -> QKV_CTE_Config:
    """
    Build QKV_CTE_Config object used throughout the kernel.
    Assumes user inputs have already been validated.
    """
    _, S, _ = args.input.shape

    add_bias = args.bias != None
    add_layer_norm_bias = args.layer_norm_bias != None

    # Load input with transpose when the three conditions are all met:
    # If the input is load_input_with_DMA_transpose=True, but these conditions are not met, use PE transpose instead.
    #  - 1. The hardware architecture is trn2 or higher; trn1 lacks support for this feature.
    #  - 2. fusedAdd is disabled, as it requires DMA FMA mode that is not supported yet with DMA transpose.
    #  - 3. There is no normalization, as normalization requires input to be in non-transposed layout.
    #  - 4. The input dtype is 2-byte dtype, i.e. BF16/BF16
    load_input_with_DMA_transpose = (
        args.load_input_with_DMA_transpose
        and nki.isa.get_nc_version() >= nki.isa.nc_version.gen3
        and (args.fused_norm_type == NormType.NO_NORM)
        and args.fused_residual_add == False
        and (args.input.dtype == nl.bfloat16 or args.input.dtype == nl.float16)
    )

    # Compute dtype used in the kernel. Even if inputs are fp32, computations will be done with bf16.
    compute_mm_dtype = args.input.dtype if args.input.dtype != nl.float32 else nl.bfloat16
    act_dtype = nl.float32
    # Instances after >=Trn2 support BF16 transpose mode on PE Array.
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

    return QKV_CTE_Config(
        output_layout=args.output_layout,
        add_bias=add_bias,
        fused_residual_add=args.fused_residual_add,
        fused_norm_type=args.fused_norm_type,
        add_layer_norm_bias=add_layer_norm_bias,
        fused_rope=args.fused_rope,
        use_auto_allocation=args.use_auto_allocation,
        load_input_with_DMA_transpose=load_input_with_DMA_transpose,
        compute_mm_dtype=compute_mm_dtype,
        act_dtype=act_dtype,
        psum_transpose_dtype=psum_transpose_dtype,
        use_BxS_input_reshape=use_BxS_input_reshape,
        total_available_sbuf_space_to_this_kernel=total_available_sbuf_space_to_this_kernel,
    )


def _get_tensor_dimensions(args: QKV_CTE_UserInput, cfg: QKV_CTE_Config) -> QKV_CTE_Dims:
    """
    Build QKV_CTE_Dims object to store tensor dimensions used throughout the kernel.
    Some are pre-calculated ahead of time, e.g "S_shard".
    Assumes user inputs have already been validated.
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
        num_128_tiles_per_H=num_128_tiles_per_H,
        num_512_tiles_per_H=num_512_tiles_per_H,
        num_512_tiles_per_I=num_512_tiles_per_I,
        NUM_WEIGHT_BUFFERS_DEFAULT=NUM_WEIGHT_BUFFERS_DEFAULT,
        WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT=WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT,
        MAX_S_MULTI_BUFFER_DEGREE=MAX_S_MULTI_BUFFER_DEGREE,
    )
