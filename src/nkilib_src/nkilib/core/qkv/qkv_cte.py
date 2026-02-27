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
This kernel implements QKV (Query, Key, Value) projection optimized for Context Encoding (CTE) with support for multiple fused operations including normalization, residual addition, bias, and RoPE.
"""

# Standard Library
import math
from typing import List, Optional, Tuple, cast

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ..utils import stream_shuffle_broadcast
from ..utils.allocator import Logger, SbufManager, sizeinbytes

# NKI Library
from ..utils.common_types import NormType, QKVOutputLayout, QKVWeightLayout, QuantizationType
from ..utils.logging import get_logger

# QKV CTE
from .qkv_cte_utils import (
    QKV_CTE_Config,
    QKV_CTE_Dims,
    QKV_CTE_UserInput,
    _build_config,
    _get_tensor_dimensions,
    _validate_user_inputs,
)

# HARDWARE CONSTANTS
NUM_HW_PSUM_BANKS = 8
MAX_STREAM_SHUFFLE_PARTITIONS = 32
NUM_MX_WEIGHT_BUFFERS = 2
MX_NEUTRAL_SCALE = 127  # MX scale exponent bias: 2^(127-127) = 2^0 = 1.0 (no scaling)
NUM_QKV_SEGMENTS = 3  # Q, K, V


def _get_psum_bank_size() -> int:
    """
    Calculate PSUM bank size in bytes.

    Returns:
        int: Size of a single PSUM bank in bytes
    """
    return sizeinbytes(nl.float32) * nl.tile_size.psum_fmax


# FP8 QUANTIZATION CONSTANT
def _get_fp8_e4m3_max_pos_val() -> float:
    """
    Get the maximum positive value for FP8 E4M3 quantization based on hardware version.

    Returns:
        448.0 for trn3 (gen4+), 240.0 for earlier hardware (trn1/trn2)

    TODO: switch to nki.isa.get_nc_version() >= nki.isa.nc_version.gen4 after the comparison support
    """
    return 448.0 if nki.isa.get_nc_version() == nki.isa.nc_version.gen4 else 240.0


def qkv_cte(
    input: nl.ndarray,
    fused_qkv_weights: nl.ndarray,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.ndarray] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attention_prev: Optional[nl.ndarray] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.ndarray] = None,
    layer_norm_bias: Optional[nl.ndarray] = None,
    norm_eps: Optional[float] = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[nl.ndarray] = None,
    sin_cache: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # --- FP8 KV Cache Quantization Related
    k_cache: Optional[nl.ndarray] = None,
    v_cache: Optional[nl.ndarray] = None,
    k_scale: Optional[nl.ndarray] = None,
    v_scale: Optional[nl.ndarray] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[type] = None,
    # --- Block KV Cache Related
    use_block_kv: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[nl.ndarray] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
    # --- Quantization Related
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[nl.ndarray] = None,
    qkv_in_scale: Optional[nl.ndarray] = None,
    # ----------------------------------------
    is_input_swizzled: bool = False,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
) -> nl.ndarray:
    """
    QKV (Query, Key, Value) projection kernel with multiple (optional) fused operations.

    This kernel is optimized for large B x S, which commonly appear in prefill/context-encoding.
    Ideally, use this kernel when B x S >= 128.

    Performs matrix multiplication between hidden states (input) and fused QKV weights matrix,
    with optional fused operations including:
    - Residual addition (input + mlp_prev + attention_prev)
    - Layer normalization (LayerNorm) or RMS normalization
    - Bias addition to QKV projection output
    - RoPE (Rotary Position Embedding) rotation applied to Query and Key heads

    Core operation:
    1. Optional residual addition: input = input + mlp_prev + attention_prev
    2. Optional normalization: input = norm(input)
    3. QKV projection: qkv = input @ fused_qkv_weights + bias
    4. Optional RoPE: apply rotary position embedding to Q and K heads in qkv

    Formulas for fused operators:
    -----------------------------
    RMS Norm:
        RMSNorm(x) = x * gamma / sqrt(mean(x²) + eps)

    Layer Norm:
        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
        where var(x) = mean((x - mean(x))²)

    Both normalizations operate along the hidden dimension for each sequence position.

    RoPE (Rotary Position Embedding):
        For each Query/Key head X = [X1, X2] (where X1, X2 are first/second half of head):
            RoPE(X) = [X1, X2] * cos_cache + [-X2, X1] * sin_cache

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        I: Fused QKV dimension = (num_q_heads + 2*num_kv_heads) * d_head
        d_head: Dimension per attention head
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        num_heads: Total number of heads = num_q_heads + 2*num_kv_heads

    Args:
        input (nl.ndarray): [B, S, H], Input hidden states tensor where B=batch, S=sequence_length, H=hidden_dim.
            We name it 'input' and not 'hidden' to avoid ambiguity with the size of "hidden dimension".
        fused_qkv_weights (nl.ndarray): [H, I],  or [H//4, I] for MX, Fused QKV weight matrix where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head
        output_layout (QKVOutputLayout): Output tensor layout: QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]. Default: QKVOutputLayout.BSD
        bias (Optional[nl.ndarray]): [1, I], Bias tensor to add to QKV projection output. Default: None
        fused_residual_add (Optional[bool]): Whether to perform residual addition: input = input + mlp_prev + attention_prev. Default: False
        mlp_prev (Optional[nl.ndarray]): [B, S, H], Previous MLP output tensor for residual addition. Default: None
        attention_prev (Optional[nl.ndarray]): [B, S, H], Previous attention output tensor for residual addition. Default: None
        fused_norm_type (NormType): Type of normalization: NormType.NO_NORM, NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA, or NormType.LAYER_NORM.
            NormType.RMS_NORM_SKIP_GAMMA assumes fused_qkv_weights have been pre-multiplied with gamma vector, so its skipped here. Default: NormType.NO_NORM
        gamma_norm_weights (Optional[nl.ndarray]): [1, H], Normalization gamma/scale weights (required for NormType.RMS_NORM and NormType.LAYER_NORM). Default: None
        layer_norm_bias (Optional[nl.ndarray]): [1, H], Layer normalization beta/bias weights (only for NormType.LAYER_NORM). Using layer norm bias is optional. Default: None
        norm_eps (Optional[float]): Epsilon value for numerical stability in normalization. Default: 1e-6
        hidden_actual (Optional[int]): Actual hidden dimension for padded tensors (if H contains padding). Default: None
        fused_rope (Optional[bool]): Whether to apply RoPE rotation to Query and Key heads after QKV projection. Default: False
        cos_cache (Optional[nl.ndarray]): [B, S, d_head], Cosine cache for RoPE (required if fused_rope=True). Default: None
        sin_cache (Optional[nl.ndarray]): [B, S, d_head], Sine cache for RoPE (required if fused_rope=True). Default: None
        d_head (Optional[int]): Dimension per attention head (required for QKVOutputLayout.NBSd and RoPE). Default: None
        num_q_heads (Optional[int]): Number of query heads (required for RoPE). Default: None
        num_kv_heads (Optional[int]): Number of key/value heads (required for RoPE). Default: None
        store_output_in_sbuf (bool): Whether to store output in SBUF (currently unsupported, must be False). Default: False
        sbm (Optional[SbufManager]): Optional SBUF manager for memory allocation control, with pre-specified bounds for SBUF usage.
            If sbm is not provided, kernel will by default be allocated and use all of the available SBUF space. Default: None
        use_auto_allocation (bool): Whether to use automatic SBUF allocation, by default kernel is manually allocated and it creates its own SbufManager.
            If 'sbm' is provided by user, user has the responsibility to set use_auto_allocation=True in the provided SbufManager. Default: False
        load_input_with_DMA_transpose (bool): Whether to use DMA transpose optimization. Default: True
        quantization_type: QuantizationType, default=QuantizationType.NONE
        qkv_w_scale: Optional[nl.ndarray], default=None The weight quantization scale for qkv projection,
            Shape: [H//32, I] for MX
        qkv_in_scale: Optional[nl.ndarray], default=None The input quantization scale for qkv projection, currently assume the input quantization scales are the scale for q, k, v projections
        is_input_swizzled: bool, default=False
            Whether the input tensor is swizzled (only applicable with MX Quantization).
            If not swizzled, input has shape [B, S, H].
            If swizzled, input has shape [B, S, H] but is preswizzled from
            [B, S, H//512, 128, 4] -> [B, S, 4, H//512, 128] and flattened to [B, S, H].
        weight_layout (QKVWeightLayout): Layout of fused_qkv_weights. See QKVWeightLayout
            docstring for packing instructions. Default: QKVWeightLayout.CONTIGUOUS

    Returns:
        output (nl.ndarray): QKV projection output tensor:
            - If output_layout=QKVOutputLayout.BSD: shape [B, S, I]
            - If output_layout=QKVOutputLayout.NBSd: shape [num_heads, B, S, d_head]

    Notes:
        Tensor Shape Requirements:
        - H must be ≤ 24576 and divisible by 128
        - I must be ≤ 4096
        - For QKVOutputLayout.NBSd output: d_head must be specified and equal to 128

        Dimension Consistency:
        - input.shape[2] must equal fused_qkv_weights.shape[0] (H dimension)
        - If heads are specified: (num_q_heads + 2*num_kv_heads) * d_head must equal I

        Fused Operation Requirements:
        - fused_residual_add=True requires both mlp_prev and attention_prev tensors
        - NormType.RMS_NORM/NormType.LAYER_NORM require gamma_norm_weights and norm_eps
        - fused_rope=True requires cos_cache, sin_cache, num_q_heads, and num_kv_heads

        Hardware Compatibility:
        - Loading input with dma transpose may be ignored internally if current implementation
          or hardware does not allow it.

        Supported Data Types:
        - bf16, fp16, fp32 (fp32 inputs are internally converted to bf16 for computation)
        - mxfp8/int32 weights for MX quantization

    Pseudocode:
        # Step 1: Optional fused residual addition
        if fused_residual_add:
            x = input + mlp_prev + attention_prev
        else:
            x = input

        # Step 2: Optional normalization
        if fused_norm_type == RMS_NORM:
            x = x * gamma / sqrt(mean(x^2) + eps)
        elif fused_norm_type == LAYER_NORM:
            x = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

        # Step 3: QKV projection with optional bias
        qkv = x @ fused_qkv_weights
        if bias is not None:
            qkv = qkv + bias

        # Step 4: Optional RoPE rotation on Q and K heads
        if fused_rope:
            Q, K, V = split_qkv(qkv, num_q_heads, num_kv_heads, d_head)
            Q = apply_rope(Q, cos_cache, sin_cache)
            K = apply_rope(K, cos_cache, sin_cache)
            qkv = concat(Q, K, V)

        # Step 5: Reshape output based on layout
        if output_layout == NBSd:
            output = reshape(qkv, [num_heads, B, S, d_head])
        else:  # BSD
            output = qkv  # [B, S, I]

        return output
    """

    # Build object of user inputs.
    user_inputs = QKV_CTE_UserInput(
        input=input,
        fused_qkv_weights=fused_qkv_weights,
        output_layout=output_layout,
        bias=bias,
        fused_residual_add=fused_residual_add,
        mlp_prev=mlp_prev,
        attention_prev=attention_prev,
        fused_norm_type=fused_norm_type,
        gamma_norm_weights=gamma_norm_weights,
        layer_norm_bias=layer_norm_bias,
        norm_eps=norm_eps,
        hidden_actual=hidden_actual,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        # FP8 KV Cache Quantization
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        kv_dtype=kv_dtype,
        # Block KV Cache
        use_block_kv=use_block_kv,
        block_size=block_size,
        slot_mapping=slot_mapping,
        # Performance
        store_output_in_sbuf=store_output_in_sbuf,
        sbm=sbm,
        use_auto_allocation=use_auto_allocation,
        load_input_with_DMA_transpose=load_input_with_DMA_transpose,
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        is_input_swizzled=is_input_swizzled,
        weight_layout=weight_layout,
    )

    _validate_user_inputs(args=user_inputs)
    # Build 'cfg' object, to store kernel configuration.
    cfg = _build_config(args=user_inputs)
    # Build 'dims' object to store tensor dimensions used throughout the kernel.
    dims = _get_tensor_dimensions(args=user_inputs, cfg=cfg)

    # Create output tensor with original dimensions
    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
        # Keep the output in bf16 when using static quantization
        output_dtype = nl.bfloat16
    elif input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]:
        # FP8 input produces bf16 output (matmul accumulates in f32, stores as bf16)
        output_dtype = nl.bfloat16
    else:
        output_dtype = input.dtype

    if cfg.use_kv_quantization:
        q_tensor_hbm = nl.ndarray((dims.B_orig, dims.S_orig, dims.q_dim), dtype=input.dtype, buffer=nl.shared_hbm)
        output_hbm = None
    elif cfg.output_layout == QKVOutputLayout.BSD:
        output_hbm = nl.ndarray((dims.B_orig, dims.S_orig, dims.I), dtype=output_dtype, buffer=nl.shared_hbm)
        q_tensor_hbm = None
    else:  # QKVOutputLayout.NBSd
        output_hbm = nl.ndarray(
            (dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head),
            dtype=output_dtype,
            buffer=nl.shared_hbm,
        )
        q_tensor_hbm = None

    # Potentially reshape B,S to BxS, for performance benefits.
    if cfg.use_BxS_input_reshape:
        input = input.reshape((1, dims.BxS, dims.H))
        if cfg.fused_residual_add:
            mlp_prev = mlp_prev.reshape((1, dims.BxS, dims.H))
            attention_prev = attention_prev.reshape((1, dims.BxS, dims.H))
        if cfg.fused_rope:
            cos_cache = cos_cache.reshape((1, dims.BxS, dims.d_head))
            sin_cache = sin_cache.reshape((1, dims.BxS, dims.d_head))
        if cfg.use_kv_quantization:
            q_tensor_hbm = q_tensor_hbm.reshape((1, dims.BxS, dims.q_dim))
            if not use_block_kv:
                # Cache layout: [B, max_seq_len, kv_dim] -> [1, max_seq_len, kv_dim] (B folded)
                k_cache = k_cache.reshape((1, k_cache.shape[1], dims.kv_dim))
                v_cache = v_cache.reshape((1, v_cache.shape[1], dims.kv_dim))
            if use_block_kv:
                slot_mapping = slot_mapping.reshape((1, dims.BxS))
        elif cfg.output_layout == QKVOutputLayout.BSD:
            output_hbm = output_hbm.reshape((1, dims.BxS, dims.I))
        elif cfg.output_layout == QKVOutputLayout.NBSd:
            output_hbm = output_hbm.reshape((dims.num_heads, 1, dims.BxS, dims.d_head))

    # Pass values and directly, and keep 'cfg' and 'dims'
    # object separate for clarity.
    if quantization_type == QuantizationType.MX:
        _qkv_cte_mx_impl(
            input_hbm=input,
            fused_qkv_weights_hbm=fused_qkv_weights,
            output_hbm=output_hbm,
            cfg=cfg,
            dims=dims,
            sbm=sbm,
            bias_hbm=bias,
            mlp_prev_hbm=mlp_prev,
            attention_prev_hbm=attention_prev,
            gamma_norm_weights_hbm=gamma_norm_weights,
            layer_norm_bias_hbm=layer_norm_bias,
            norm_eps=norm_eps,
            cos_cache_hbm=cos_cache,
            sin_cache_hbm=sin_cache,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
        )
    else:
        _qkv_cte_impl(
            input_hbm=input,
            fused_qkv_weights_hbm=fused_qkv_weights,
            output_hbm=output_hbm,
            cfg=cfg,
            dims=dims,
            sbm=sbm,
            bias_hbm=bias,
            mlp_prev_hbm=mlp_prev,
            attention_prev_hbm=attention_prev,
            gamma_norm_weights_hbm=gamma_norm_weights,
            layer_norm_bias_hbm=layer_norm_bias,
            norm_eps=norm_eps,
            cos_cache_hbm=cos_cache,
            sin_cache_hbm=sin_cache,
            q_tensor_hbm=q_tensor_hbm,
            k_cache_hbm=k_cache,
            v_cache_hbm=v_cache,
            k_scale_hbm=k_scale,
            v_scale_hbm=v_scale,
            slot_mapping_hbm=slot_mapping,
            qkv_in_scale=qkv_in_scale,
            qkv_w_scale=qkv_w_scale,
        )

    # Revert BxS to B,S as it is required by the user provided output_layout.
    if cfg.use_BxS_input_reshape:
        input = input.reshape((dims.B_orig, dims.S_orig, dims.H))
        if cfg.fused_residual_add:
            mlp_prev = mlp_prev.reshape((dims.B_orig, dims.S_orig, dims.H))
            attention_prev = attention_prev.reshape((dims.B_orig, dims.S_orig, dims.H))
        if cfg.fused_rope:
            cos_cache = cos_cache.reshape((dims.B_orig, dims.S_orig, dims.d_head))
            sin_cache = sin_cache.reshape((dims.B_orig, dims.S_orig, dims.d_head))
        if cfg.use_kv_quantization:
            q_tensor_hbm = q_tensor_hbm.reshape((dims.B_orig, dims.S_orig, dims.q_dim))
            if not use_block_kv:
                # Cache layout: [B, max_seq_len, kv_dim] - restore original shape
                k_cache = k_cache.reshape((dims.B_orig, k_cache.shape[1], dims.kv_dim))
                v_cache = v_cache.reshape((dims.B_orig, v_cache.shape[1], dims.kv_dim))
            if use_block_kv:
                slot_mapping = slot_mapping.reshape((dims.B_orig, dims.S_orig))
        elif cfg.output_layout == QKVOutputLayout.BSD:
            output_hbm = output_hbm.reshape((dims.B_orig, dims.S_orig, dims.I))
        elif cfg.output_layout == QKVOutputLayout.NBSd:
            output_hbm = output_hbm.reshape((dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head))

    if cfg.use_kv_quantization:
        return q_tensor_hbm, k_cache, v_cache
    return output_hbm


def _quantize_and_store_kv(
    output_sb: nl.ndarray,
    scale_sb: nl.ndarray,
    cache_hbm: nl.ndarray,
    kv_offset: int,
    i_batch: int,
    s_tile_global_offset: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    slot_mapping_sb: Optional[nl.ndarray] = None,
):
    """
    Scale values and clamp to FP8 range, then store to cache.
    Quantization formula: quantized_value = clamp(value / scale, fp8_min, fp8_max)

    Uses NBSd-style .ap() pattern for multi-head output.
    """
    kv_dim = dims.kv_dim
    d_head = dims.d_head
    num_kv_heads = dims.num_kv_heads
    max_seq_len = cache_hbm.shape[1] if not cfg.use_block_kv else None

    # Compute reciprocal of scale once
    inv_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=scale_sb.dtype, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_scale_sb[0:s_tile_sz, 0:1], data=scale_sb[0:s_tile_sz, 0:1])

    # Allocate buffer for full kv_dim, scale and clamp all heads at once
    clamped_sb = sbm.alloc_stack((nl.tile_size.pmax, kv_dim), dtype=cfg.kv_dtype, buffer=nl.sbuf)

    nisa.tensor_scalar(
        dst=clamped_sb[0:s_tile_sz, 0:kv_dim],
        data=output_sb[0:s_tile_sz, kv_offset : kv_offset + kv_dim],
        op0=nl.multiply,
        operand0=inv_scale_sb[0:s_tile_sz, 0:1],
    )
    nisa.tensor_scalar(
        dst=clamped_sb[0:s_tile_sz, 0:kv_dim],
        data=clamped_sb[0:s_tile_sz, 0:kv_dim],
        op0=nl.maximum,
        operand0=cfg.fp8_min,
        op1=nl.minimum,
        operand1=cfg.fp8_max,
    )

    if cfg.use_block_kv:
        # Block KV layout: [num_blocks, block_size, kv_dim]
        num_blocks = cache_hbm.shape[0]
        block_size = cfg.block_size
        total_slots = num_blocks * block_size
        cache_2d = cache_hbm.reshape((total_slots, kv_dim))

        nisa.dma_copy(
            dst=cache_2d.ap(
                pattern=[[kv_dim, s_tile_sz], [1, kv_dim]],
                offset=0,
                vector_offset=slot_mapping_sb.ap(pattern=[[1, s_tile_sz], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            src=clamped_sb[0:s_tile_sz, 0:kv_dim],
            dge_mode=dge_mode.swdge,
        )
    else:
        # Cache layout: [B, max_seq_len, kv_dim] - same pattern as BSD output store
        nisa.dma_copy(
            dst=cache_hbm.ap(
                pattern=[[kv_dim, s_tile_sz], [1, kv_dim]],
                offset=i_batch * max_seq_len * kv_dim + s_tile_global_offset * kv_dim,
            ),
            src=clamped_sb[0:s_tile_sz, 0:kv_dim],
            dge_mode=dge_mode.swdge,
        )


def _qkv_cte_impl(
    input_hbm,
    fused_qkv_weights_hbm,
    output_hbm,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    bias_hbm: Optional[nl.ndarray] = None,
    # Fused Residual Add Related
    mlp_prev_hbm: Optional[nl.ndarray] = None,
    attention_prev_hbm: Optional[nl.ndarray] = None,
    # Fused Normalization Related
    gamma_norm_weights_hbm: Optional[nl.ndarray] = None,
    layer_norm_bias_hbm: Optional[nl.ndarray] = None,
    norm_eps: Optional[float] = 1e-6,
    # Fused RoPE Related
    cos_cache_hbm: Optional[nl.ndarray] = None,
    sin_cache_hbm: Optional[nl.ndarray] = None,
    # FP8 KV Cache Quantization Related
    q_tensor_hbm: Optional[nl.ndarray] = None,
    k_cache_hbm: Optional[nl.ndarray] = None,
    v_cache_hbm: Optional[nl.ndarray] = None,
    k_scale_hbm: Optional[nl.ndarray] = None,
    v_scale_hbm: Optional[nl.ndarray] = None,
    # Block KV Cache Related
    slot_mapping_hbm: Optional[nl.ndarray] = None,
    # Quantization Related
    qkv_in_scale: Optional[nl.ndarray] = None,
    qkv_w_scale: Optional[nl.ndarray] = None,
):
    """
    Core QKV CTE kernel implementation.

    Performs the main computation including optional normalization, QKV projection,
    and optional RoPE application. Handles memory allocation, tiling, and data movement
    between HBM, SBUF, and PSUM.

    Args:
        input_hbm (nl.ndarray): [dims.B, dims.S, dims.H], Input tensor on HBM
        fused_qkv_weights_hbm (nl.ndarray): [dims.H, dims.I], Weight matrix on HBM
        output_hbm (nl.ndarray): Output tensor on HBM with shape determined by cfg.output_layout
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager
        bias_hbm (Optional[nl.ndarray]): [1, I], Optional bias tensor on HBM
        mlp_prev_hbm (Optional[nl.ndarray]): [dims.B, dims.S, dims.H], Optional MLP residual on HBM
        attention_prev_hbm (Optional[nl.ndarray]): [dims.B, dims.S, dims.H], Optional attention residual on HBM
        gamma_norm_weights_hbm (Optional[nl.ndarray]): [1, H], Optional normalization weights on HBM
        layer_norm_bias_hbm (Optional[nl.ndarray]): [1, H], Optional layer norm bias on HBM
        norm_eps (Optional[float]): Epsilon for normalization stability
        cos_cache_hbm (Optional[nl.ndarray]): [B, S, d_head], Optional RoPE cosine cache on HBM
        sin_cache_hbm (Optional[nl.ndarray]): [B, S, d_head], Optional RoPE sine cache on HBM

    Returns:
        nl.ndarray: Output tensor (same as output_hbm parameter)

    Notes:
        - Processes only dims.S_shard portion of sequence dimension when sharded
        - Uses multi-buffering for sequence tiles to improve performance
        - Supports weight prefetching when SBUF space allows
    """
    # Uncomment for debug.
    # cfg.print()
    # dims.print()

    """
    Input tensor shape: [dims.B, dims.S, dims.H]
    Weight tensor shape: [dims.H, dims.I]
    
    We apply QKV projection only on dims.S_shard part of input_hbm (with dims.S_shard_offset).
    """

    S_shard = dims.S_shard
    H = dims.H
    I = dims.I
    if dims.S_shard == 0:
        return output_hbm

    """
    If user provided SbufManager (with more restricted sb_lower_bound and sb_upper_bound),
    use that (likely at the expense of performance). Otherwise, use most sbuf space available.
    """
    if sbm == None:
        sbm_logger = get_logger(name="qkv_cte")
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=cfg.total_available_sbuf_space_to_this_kernel,
            use_auto_alloc=cfg.use_auto_allocation,
            logger=sbm_logger,
        )
    sbm.open_scope()

    ######################### Global SBUF Allocations ######################################
    # Allocate zero_bias_sb, norm_eps_sb, bias_sb, gamma_norm_weights_sb, layer_norm_bias_sb

    zero_bias_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_bias_sb, value=0)

    norm_eps_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.memset(dst=norm_eps_sb, value=norm_eps)

    if cfg.add_bias:
        # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    """
    Load gamma_norm_weights_hbm (1,H) to sbuf.
    
    Mathematically, we (later) need to apply elementwise multiplication: (input) [S, H] * (gamma) [1, H] for each row.
    Note: NormType.RMS_NORM_SKIP_GAMMA skips this step.
    """
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        # We load gamma weights into SBUF as a 2-D tensor of shape [128, ceil(H / nl.tile_size.pmax) ].
        gamma_norm_weights_sb = _load_norm_weights(norm_weights_hbm=gamma_norm_weights_hbm, cfg=cfg, dims=dims, sbm=sbm)

        """
        gamma_norm_weights_sb is used for both RMS_NORM and LAYER_NORM.
        But, LAYER_NORM may have an additional Beta bias term provided: layer_norm_bias_hbm
        """
        if cfg.add_layer_norm_bias:
            # Shape: [128, ceil(H / nl.tile_size.pmax)]
            layer_norm_bias_sb = _load_norm_weights(norm_weights_hbm=layer_norm_bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    # Load KV quantization scales if enabled
    if cfg.use_kv_quantization:
        k_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        v_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_scale_sb[0 : nl.tile_size.pmax, 0:1],
            src=k_scale_hbm[0 : nl.tile_size.pmax, 0:1],
            dge_mode=dge_mode.swdge,
        )
        nisa.dma_copy(
            dst=v_scale_sb[0 : nl.tile_size.pmax, 0:1],
            src=v_scale_hbm[0 : nl.tile_size.pmax, 0:1],
            dge_mode=dge_mode.swdge,
        )

    # Load quantization scales
    w_scale_tile, in_scale_tile = None, None
    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
        w_scale_tile = sbm.alloc_stack(shape=(nl.tile_size.pmax, 3), dtype=qkv_w_scale.dtype, buffer=nl.sbuf)
        # Load and broadcast weight scale
        if qkv_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=w_scale_tile[0, :], src=qkv_w_scale[0, :])
            stream_shuffle_broadcast(w_scale_tile, w_scale_tile)
        else:
            nisa.dma_copy(dst=w_scale_tile, src=qkv_w_scale)

        in_scale_tile = sbm.alloc_stack(shape=(nl.tile_size.pmax, 1), dtype=qkv_in_scale.dtype)
        if qkv_in_scale.shape[0] == 1:
            nisa.dma_copy(dst=in_scale_tile[0, :], src=qkv_in_scale[0, :])
            stream_shuffle_broadcast(in_scale_tile, in_scale_tile)
        else:
            nisa.dma_copy(dst=in_scale_tile, src=qkv_in_scale)
        # pre-apply input scales onto the weight scaling
        nisa.activation(dst=w_scale_tile, op=nl.copy, data=w_scale_tile, scale=in_scale_tile)

        # Compute reciprocal once for quantization: 1/scale
        nisa.reciprocal(dst=in_scale_tile, data=in_scale_tile)

    ######################## Choose Multi-Buffering Degree ###################################

    """
    Multi-Buffering S: Choose max multi-buffering degree for sequence length, without spilling SBUF and PSUM space.
    
    WARNING: This function needs to be updated if any new tensors get added to the kernel.
    It assumes current tensor shapes, and its "look-ahead", e.g. pre-calculates SBUF space ahead of time.
    Note: In auto-allocation mode, sbuf space calculations do not make sense, but they do not break the kernel correctness.
    """
    s_multi_buffer_degree, projected_sbuf_taken_space_after_multi_buffer = _multi_buffering_degree_for_seqlen(
        cfg=cfg, dims=dims, sbm=sbm, qkv_in_scale=qkv_in_scale
    )

    # Block is PMAX * multi_buffer_degree, e.g.  process [128 * 4, H] elements of S at once.
    S_BLOCK_SIZE = s_multi_buffer_degree * min(dims.S_shard, nl.tile_size.pmax)
    num_blocks_per_S_shard = math.ceil(dims.S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0

    ######################## Weight Prefetching: Enough Space Left ?  #########################

    use_weight_prefetch = _use_weight_prefetch(
        projected_sbuf_taken_space_after_multi_buffer, cfg=cfg, dims=dims, sbm=sbm
    )

    if use_weight_prefetch:
        """
        If rhs (weights) is small enough, weights can be pre-loaded before QKV projection.

        NOTE: In both prefetched and non-pretched case, we append allocated weight tensor to the same "weights_sb" list,
        to keep later changes to the minimum. To keep indexing differences in QKV projection to the minimum between the two cases,
        keep the shape of the allocated tensor the same:
        weights_allocated = (128, num_allocated_H_subtiles_per_weight_load, I), and
        weights_sb[...] may multi-buffer/allocate multiple of above tensors

        In the case of weight prefetching, set the following variables:
        """
        num_weight_buffers = 1  # Since we allocate all weights at once, so there is no need for multiple-buffers. We still need to set it to 1 as the kernel uses this constant later.
        weight_load_block_size_per_H = (
            H  # We pre-load the entire weight matrix. In non-prefetched case, we can load e.g 1024 H blocks at a time.
        )
        num_weight_load_blocks_per_H = 1  # In pre-fetch case, H / weight_load_block_size_per_H = 1
        max_num_128_H_subtiles_per_weight_block = math.ceil(weight_load_block_size_per_H / 128)  # = math.ceil(H / 128).

        weights_sb = []
        weights_prefetched_sb = sbm.alloc_stack(
            (nl.tile_size.pmax, max_num_128_H_subtiles_per_weight_block, I),
            dtype=cfg.compute_mm_dtype,
            buffer=nl.sbuf,
        )

        for i_tile_H in range(max_num_128_H_subtiles_per_weight_block):
            h_tile_sz = min(nl.tile_size.pmax, H - (i_tile_H * nl.tile_size.pmax))
            dst_pattern = [
                [max_num_128_H_subtiles_per_weight_block * I, h_tile_sz],
                [1, I],
            ]
            src_pattern = [[I, h_tile_sz], [1, I]]
            nisa.dma_copy(
                dst=weights_prefetched_sb.ap(pattern=dst_pattern, offset=i_tile_H * I),
                src=fused_qkv_weights_hbm.ap(pattern=src_pattern, offset=i_tile_H * nl.tile_size.pmax * I),
                dge_mode=dge_mode.swdge,
            )
        weights_sb.append(weights_prefetched_sb)

    for i_batch in range(dims.B):
        for i_block_S in nl.affine_range(num_blocks_per_S_shard):
            sbm.open_scope()
            # Adjust for the last loop iteration.
            s_block_sz = min(S_BLOCK_SIZE, S_shard - S_BLOCK_SIZE * i_block_S)
            num_S_tiles_in_block = math.ceil(s_block_sz / nl.tile_size.pmax)

            #################### Start of Allocations for Multi-Buffered tensors ##############################
            input_sb = []
            if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                input_quantized_sb = []
            for _ in range(num_S_tiles_in_block):
                align = 32 if cfg.load_input_with_DMA_transpose else 1  # DMA_transpose requires align=32.
                input_dtype = (
                    cfg.compute_mm_dtype
                    if cfg.quantization_config.quantization_type != QuantizationType.STATIC
                    else input_hbm.dtype
                )
                input_sb.append(
                    sbm.alloc_stack(
                        (nl.tile_size.pmax, H),
                        dtype=input_dtype,
                        buffer=nl.sbuf,
                        align=align,
                    )
                )
                if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                    # assume q, k, v projections have same input scale, therefore no need to have multiple input copies.
                    input_quantized_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, H),
                            dtype=cfg.quantization_config.quant_dtype,
                            buffer=nl.sbuf,
                            align=align,
                        )
                    )

            output_sb = []
            for _ in range(num_S_tiles_in_block):
                output_dtype = (
                    cfg.compute_mm_dtype
                    if cfg.quantization_config.quantization_type != QuantizationType.STATIC
                    else nl.bfloat16
                )
                output_sb.append(sbm.alloc_stack((nl.tile_size.pmax, I), dtype=output_dtype, buffer=nl.sbuf))

            if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                # Allocate rms_norm tensor outside of tiled S buffer loop, as square_sum has NUM_128_TILES_PER_S_BUFFER in its shape.
                # Write RMS and RMS Reciprocal tensors here, in-place.
                square_sum_sb = []
                for _ in range(num_S_tiles_in_block):
                    square_sum_sb.append(sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.act_dtype, buffer=nl.sbuf))

            elif cfg.fused_norm_type == NormType.LAYER_NORM:
                # Used for the result of layer_norm, stores mean and variance.
                NUM_AGGR_STATS = 2  # mean and variance.
                bn_aggr_result_sb = []
                for _ in range(num_S_tiles_in_block):
                    bn_aggr_result_sb.append(
                        sbm.alloc_stack((nl.tile_size.pmax, NUM_AGGR_STATS), dtype=cfg.act_dtype, buffer=nl.sbuf)
                    )

            if cfg.fused_rope:
                # For the input head X = [X1, X2] , RoPE does the following:
                # X = [X1, X2] * cos_cache + [-X2, X1] * sin_cache
                # X = [X1, X2] * cos_cache + [-X2*sin_cache_1, X1*sin_cache_2]
                # sin_cache_1 = sin_cache_2. Therefore, we can keep only half of the sin_cache.
                cos_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    cos_buffer_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, dims.d_head),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )

                sin_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    sin_buffer_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, dims.d_head // 2),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )

                rope_intermediate_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    rope_intermediate_buffer_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, dims.d_head * 2),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )
            #########################  End of Allocations for Multi-Buffered tensors ##################################

            # In this case, we will transpose the input buffer using PE array.
            if not cfg.load_input_with_DMA_transpose:
                for i_tile_S in range(num_S_tiles_in_block):
                    sbm.open_scope()
                    # i_tile_S is used to index input_sb, e.g., a single tile of input row [128, H].
                    ######################################################################################################
                    # Step 1: Load the row of input tensor: [nl.tile_size.pmax, H]
                    #         (optionally apply fused_residual_add with mlp_prev, and attention_prev).
                    #          If load_input_with_DMA_transpose is applicable, this step is skipped, and moved later.
                    ######################################################################################################
                    S_TILE_SIZE = nl.tile_size.pmax
                    s_tile_local_offset = (
                        i_block_S * S_BLOCK_SIZE + i_tile_S * S_TILE_SIZE
                    )  # s_tile offset within S_shard.
                    s_tile_sz = min(
                        nl.tile_size.pmax, S_shard - s_tile_local_offset
                    )  # tile_size adjusted for the last loop iteration.
                    if cfg.fused_residual_add:
                        # Load row of input, and apply fused residual add.
                        s_tile_global_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                        nisa.dma_compute(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            srcs=[
                                input_hbm.ap(
                                    pattern=[[H, s_tile_sz], [1, H]],
                                    offset=s_tile_global_offset,
                                ),
                                mlp_prev_hbm.ap(
                                    pattern=[[H, s_tile_sz], [1, H]],
                                    offset=s_tile_global_offset,
                                ),
                                attention_prev_hbm.ap(
                                    pattern=[[H, s_tile_sz], [1, H]],
                                    offset=s_tile_global_offset,
                                ),
                            ],
                            scales=[1.0, 1.0, 1.0],
                            reduce_op=nl.add,
                        )
                    else:  # Do a regular input load without the fused residual add.
                        s_tile_global_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                        nisa.dma_copy(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            src=input_hbm.ap(
                                pattern=[[H, s_tile_sz], [1, H]],
                                offset=s_tile_global_offset,
                            ),
                            dge_mode=dge_mode.swdge,
                        )

                    ######################################################################################################
                    # Step 2: Apply (partial) RMS_NORM / LAYER_NORM to the input row.
                    ######################################################################################################
                    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                        # Multiply input_sb[i_tile_S] by 1 / RMS(x) =  1 / sqrt(eps + (1 / hidden_actual) * (x1^2 + x2^2 + ... + xn^2)).
                        # Gamma weights multiply is yet to be done.
                        _apply_rms_normalization(
                            input_sb[i_tile_S],
                            square_sum_sb[i_tile_S],
                            zero_bias_sb,
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )

                    elif cfg.fused_norm_type == NormType.LAYER_NORM:
                        """
                        Compute LayerNorm statistics for row of input_sb and store it bn_aggr_result_tile.
                            mean = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 0:1]
                            rvar = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 1:2] #rvar(var + eps)
                        """
                        _compute_layer_norm_stats(
                            input_sb[i_tile_S],
                            bn_aggr_result_sb[i_tile_S],
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )

                    #######################################################################################################
                    # Step 3: Transpose input_sb using PE array, and apply already calculated NORM to each tile if needed.
                    #######################################################################################################
                    # Transposed tiles will be written back to input_sb.

                    # This is PE transpose loop, multiplying input with identity tensor.
                    # Also finalize applying normalization to the input tensor inside the loop.
                    for i_tile_H in range(dims.num_512_tiles_per_H):
                        # In this 512 loop we need a mask for H, we require H to be divisible by 128, but not 512.
                        H_TILE_SIZE = nl.tile_size.psum_fmax
                        h_tile_offset = i_tile_H * H_TILE_SIZE
                        h_tile_sz = min(H_TILE_SIZE, H - h_tile_offset)

                        # Note: In some cases, applying RMSNorm might give us better pipelining if tensor_scalar was placed here.
                        if cfg.fused_norm_type == NormType.LAYER_NORM:
                            # Compute (x - mean) * rvar  (multiply with Gamma and add Beta later).
                            mean = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 0:1]
                            rvar = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 1:2]  # rvar(var + eps)

                            nisa.tensor_scalar(
                                dst=input_sb[i_tile_S][0:s_tile_sz, nl.ds(h_tile_offset, h_tile_sz)],
                                data=input_sb[i_tile_S][0:s_tile_sz, nl.ds(h_tile_offset, h_tile_sz)],
                                op0=nl.subtract,
                                operand0=mean,
                                op1=nl.multiply,
                                operand1=rvar,
                            )

                        # Transpose each [S (128), H (512)] tile of input buffer -> PSUM.
                        psum_bank_size = _get_psum_bank_size()
                        input_transposed_psum = []
                        for bank_id in range(NUM_HW_PSUM_BANKS):
                            # Note: PSUM tensors do not have "use_auto_allocation" flag like SbufManager to ignore the allocation.
                            if cfg.use_auto_allocation:
                                input_transposed_psum.append(
                                    nl.ndarray(
                                        (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                                        dtype=cfg.psum_transpose_dtype,
                                        buffer=nl.psum,
                                    )
                                )
                            else:
                                input_transposed_psum.append(
                                    nl.ndarray(
                                        (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                                        dtype=cfg.psum_transpose_dtype,
                                        buffer=nl.psum,
                                        address=(0, bank_id * psum_bank_size),
                                    )
                                )

                        tp_psum_bank_idx = (i_tile_S * dims.num_512_tiles_per_H + i_tile_H) % NUM_HW_PSUM_BANKS

                        # (Transpose) nisa.nc_matmul(...) returns result of [128, 128] shape, we need FOUR of these for a single [128, 512] tile.
                        num_128_subtiles_per_H_tile = math.ceil(h_tile_sz / 128)  # At most FMAX/PMAX = 4
                        for j_subtile_H in nl.affine_range(num_128_subtiles_per_H_tile):
                            H_SUBTILE_SIZE = nl.tile_size.pmax
                            h_subtile_offset_src = h_tile_offset + j_subtile_H * H_SUBTILE_SIZE  # src SBUF
                            h_subtile_offset_dst = j_subtile_H * H_SUBTILE_SIZE  # dst PSUM
                            h_subtile_sz = min(H_SUBTILE_SIZE, H - h_subtile_offset_src)

                            nisa.nc_transpose(
                                data=input_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(h_subtile_offset_src, h_subtile_sz),
                                ],
                                dst=input_transposed_psum[tp_psum_bank_idx][
                                    0:h_subtile_sz,
                                    nl.ds(h_subtile_offset_dst, s_tile_sz),
                                ],
                            )

                        # Copy transposed [128 (H), 512 (S)] tile from PSUM -> SBUF, and apply gamma_weights.
                        if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
                            # Note: NormType.RMS_NORM_SKIP_GAMMA skips this step.

                            # -- INPUT_TRANSPOSED * GAMMA_WEIGHTS elementwise-multiply ---
                            # Multiply each PSUM sub-tile [128(H), 128 (S)] with [128 (H), 1] column of gamma (with offsets).
                            # Recall that gamma_weights_hbm (H) was broadcasted to [128, H // nl.tile_size.pmax].
                            for j_subtile_H in nl.affine_range(num_128_subtiles_per_H_tile):
                                h_subtile_offset_sbuf = h_tile_offset + j_subtile_H * nl.tile_size.pmax
                                h_subtile_sz = min(nl.tile_size.pmax, H - h_subtile_offset_sbuf)

                                s_tile_offset_psum = j_subtile_H * nl.tile_size.pmax  # src
                                s_tile_offset_sbuf = h_subtile_offset_sbuf  # dst

                                # Index the right column of gamma_norm_weights_sb.
                                # i_tile_H is current 512th tile of H, we need current 128th tile of H for gamma column index.
                                gamma_tile_index = (
                                    i_tile_H * (nl.tile_size.psum_fmax // nl.tile_size.pmax) + j_subtile_H
                                )
                                if not cfg.add_layer_norm_bias:  # Multiply gamma weights only.
                                    nisa.tensor_scalar(
                                        dst=input_sb[i_tile_S][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_sbuf, s_tile_sz),
                                        ],
                                        data=input_transposed_psum[tp_psum_bank_idx][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_psum, s_tile_sz),
                                        ],
                                        op0=nl.multiply,
                                        operand0=gamma_norm_weights_sb[0:h_subtile_sz, nl.ds(gamma_tile_index, 1)],
                                    )
                                else:  # In addition to gamma weights, multiply with beta weights as well (layer_norm_bias).
                                    beta_tile_index = gamma_tile_index
                                    nisa.tensor_scalar(
                                        dst=input_sb[i_tile_S][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_sbuf, s_tile_sz),
                                        ],
                                        data=input_transposed_psum[tp_psum_bank_idx][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_psum, s_tile_sz),
                                        ],
                                        op0=nl.multiply,
                                        operand0=gamma_norm_weights_sb[0:h_subtile_sz, nl.ds(gamma_tile_index, 1)],
                                        op1=nl.add,
                                        operand1=layer_norm_bias_sb[0:h_subtile_sz, nl.ds(beta_tile_index, 1)],
                                    )

                                # Input quantization after norm
                                if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                                    nisa.tensor_scalar(
                                        dst=input_quantized_sb[i_tile_S][
                                            0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)
                                        ],
                                        data=input_sb[i_tile_S][0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)],
                                        op0=nl.multiply,
                                        operand0=in_scale_tile,
                                    )

                                    fp8_max_val = _get_fp8_e4m3_max_pos_val()
                                    nisa.tensor_scalar(
                                        dst=input_quantized_sb[i_tile_S][
                                            0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)
                                        ],
                                        data=input_quantized_sb[i_tile_S][
                                            0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)
                                        ],
                                        op0=nl.minimum,
                                        operand0=fp8_max_val,
                                        op1=nl.maximum,
                                        operand1=-fp8_max_val,
                                    )

                        else:  # In NO_NORM, RMS_NORM_SKIP_GAMMA, we just copy the transposed PSUM tile to SBUF.
                            # We copy [128, 512] elements in single tensor copy for performance.
                            # -> for S % 128 !=0, we'll copy some garbage memory (to be masked later).
                            # Otherwise, we need extra nl.tile_size.psum_fmax/nl.tile_size.pmax loop here.
                            nisa.tensor_copy(
                                dst=input_sb[i_tile_S][0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)],
                                src=input_transposed_psum[tp_psum_bank_idx][0 : nl.tile_size.pmax, 0:h_tile_sz],
                            )

                            if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                                # perform quantization for the input
                                # input_quantized = clip(input / in_scale_tile, _FP8_E4M3_MAX_POS_VAL, -_FP8_E4M3_MAX_POS_VAL)
                                nisa.tensor_scalar(
                                    dst=input_quantized_sb[i_tile_S][
                                        0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)
                                    ],
                                    data=input_transposed_psum[tp_psum_bank_idx][0 : nl.tile_size.pmax, 0:h_tile_sz],
                                    op0=nl.multiply,
                                    operand0=in_scale_tile,
                                )

                                fp8_max_val = _get_fp8_e4m3_max_pos_val()
                                nisa.tensor_scalar(
                                    dst=input_quantized_sb[i_tile_S][
                                        0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)
                                    ],
                                    data=input_quantized_sb[i_tile_S][
                                        0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)
                                    ],
                                    op0=nl.minimum,
                                    operand0=fp8_max_val,
                                    op1=nl.maximum,
                                    operand1=-fp8_max_val,
                                )

                        # End of i_tile_H loop.
                    # End of i_tile_S loop.
                    sbm.close_scope()  # act and bn_stats_result are the only allocated tensors in this scope.
                # End of cfg.load_input_with_DMA_transpose conditional

            #######################################################################################################
            # Step 4: (QKV Projection) Multiply transposed input buffer (potentially with norm pre-applied) with weights.
            #######################################################################################################

            # PSUM accumulation buffer for QKV matmult results.
            # Each column of 512 size of rhs (weights) is accumulated to a distinct PSUM bank.
            # Since kernel assumes I <= 4096 (=psum_banks * 512), we have enough banks to accumulate all columns without COPYs of intermediate results.
            psum_bank_size = _get_psum_bank_size()
            qkv_MM_num_psum_banks_needed = dims.num_512_tiles_per_I * num_S_tiles_in_block
            qkv_MM_output_psum = []
            for bank_id in nl.affine_range(qkv_MM_num_psum_banks_needed):
                if cfg.use_auto_allocation:
                    qkv_MM_output_psum.append(
                        nl.ndarray((nl.tile_size.pmax, nl.tile_size.psum_fmax), dtype=nl.float32, buffer=nl.psum)
                    )
                else:
                    qkv_MM_output_psum.append(
                        nl.ndarray(
                            (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                            dtype=nl.float32,
                            buffer=nl.psum,
                            address=(0, bank_id * psum_bank_size),
                        )
                    )

            """
            Multiply transposed input_sb @ weight tensor.
            
            The following loop reads from transposed input_sb(i_tile_S, 128, H),
            and outputs to psum_buffer(psum_banks_used, 128, 512).
            
            * Loop Structure of QKV Projection (in the case of non-prefetched weights):
            * Here,  weight_load_block_size_per_H = 1024.
            for each WEIGHT_BLOCK of 1024 size (along H):
                Load [1024, I] of weights to SBUF at once.  
                * We have 8 * [128, I] sub-tiles of H in a single load.           
                
                for each row of S buffer:                  ( e.g. 1, [128, H] sized rows)
                    for jth_subtile 0 to 8:                       (1024 / 128 = 8)
                        for each 512 column tile of weights (along I)
                            Multiply (weights) tile [128, 512] with the corresponding (input) tile in transposes_input_row [128, 128].
                            * Each of 512 tiles (columns in I) is accumulated to a different PSUM bank. 
            """

            # Allocate weights here, if not prefetched already.
            if not use_weight_prefetch:
                # Load weights [weight_load_block_size_per_H=1024, I] at a time.
                # Default weight constants are meant for non-prefetches case (they are over-written in case of prefetching)
                # Note: Projection uses num_weight_buffers and weight_load_block_size_per_H for indexing regardless of use_weight_prefetch.
                num_weight_buffers = dims.NUM_WEIGHT_BUFFERS_DEFAULT  # 4
                weight_load_block_size_per_H = dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT  # 1024
                num_weight_load_blocks_per_H = math.ceil(H / weight_load_block_size_per_H)
                max_num_128_H_subtiles_per_weight_block = math.ceil(
                    weight_load_block_size_per_H / 128
                )  # e.g 1024 / 128 = 8.

                # We load weights using a strided access pattern with only a single DMA ISA.
                weights_sb = []
                for _ in range(num_weight_buffers):
                    weights_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, max_num_128_H_subtiles_per_weight_block, I),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )

            # If use_weight_prefetch, then NUM_WEIGHT_LOADS_PER_H == 1.
            for i_weight_load in nl.affine_range(num_weight_load_blocks_per_H):
                weight_load_offset = i_weight_load * weight_load_block_size_per_H
                curr_num_128_H_subtiles_per_weight_block = min(
                    max_num_128_H_subtiles_per_weight_block,
                    math.ceil((H - weight_load_offset) / 128),
                )

                if not use_weight_prefetch:
                    # If we did not prefetch weights on top, load a section of weights -> compute -> load a section of weights -> etc.
                    # Example of the strided weight load in the comment below.
                    weight_buf = weights_sb[i_weight_load % num_weight_buffers]
                    nisa.dma_copy(
                        dst=weight_buf.ap(
                            pattern=[
                                [max_num_128_H_subtiles_per_weight_block * I, nl.tile_size.pmax],
                                [I, curr_num_128_H_subtiles_per_weight_block],
                                [1, I],
                            ],
                            offset=0,
                        ),
                        src=fused_qkv_weights_hbm.ap(
                            pattern=[
                                [I, min(nl.tile_size.pmax, H - weight_load_offset)],
                                [128 * I, curr_num_128_H_subtiles_per_weight_block],
                                [1, I],
                            ],
                            offset=weight_load_offset * I,
                        ),
                        dge_mode=dge_mode.swdge,
                    )

                    """
                    Strided HBM->SBUF weights load example, if loading 1024 x I weights at a time.
                    Here, weight_load_block_size_per_H  = 1024.
                    
                    HBM Weights
                    ------------
                                        I
                             -----------------------------
                        128 |       H_1                  |
                    1024 128|       H_2                  | H
                        ...                            ...
                            |       H_8                  |
                             -----------------------------
                                        ....
                            
                    SBUF Weights
                    ------------
                                                8 * I
                            -------------------------------------------------
                        128|  H_1   |  H_2 |      ....              |  H_8   |
                            -------------------------------------------------
                        
                    Note: Access pattern on HBM side is strided, we are skipping 128 * I elements each time.
                        Order:
                            [0, 0:I], [128, 0:I], [256, 0:I], ...   ( 8 rows of I elements)
                            [1, 0:I], [129, 0:I], [257, 0:I], ...   ( 8 rows of I elements)
                        
                    On SBUF side,
                            1st row of H_1, and 1st row H_2 will be both partition=0, etc.                 
                    """

                for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                    if cfg.load_input_with_DMA_transpose:
                        # Recall we did not use PE Array to transpose input buffer in case of loadWithTranspose.
                        # Use nisa.dma_transpose(...) to load/transpose just enough of input for this round of matmult.
                        # Load/transpose only [128, 1024] elements of input.

                        # NOTE: To drop H divisible by 128 constraint, update AP below with valid "num_h" for the last iteration.
                        src_offset = (
                            i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H + weight_load_offset
                        )
                        nisa.dma_transpose(
                            dst=input_sb[i_tile_S].ap(
                                pattern=[
                                    [H, nl.tile_size.pmax],
                                    [1, 1],
                                    [nl.tile_size.pmax, curr_num_128_H_subtiles_per_weight_block],
                                    [1, s_tile_sz],
                                ],
                                offset=weight_load_offset,
                            ),
                            src=input_hbm.ap(
                                pattern=[
                                    [H, s_tile_sz],
                                    [1, 1],
                                    [nl.tile_size.pmax, curr_num_128_H_subtiles_per_weight_block],
                                    [1, nl.tile_size.pmax],
                                ],
                                offset=src_offset,
                            ),
                        )

                        if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                            # perform quantization for the input
                            # input_quantized = clip(input / in_scale_tile, _FP8_E4M3_MAX_POS_VAL, -_FP8_E4M3_MAX_POS_VAL)
                            # qkv proj input quantization

                            nisa.tensor_scalar(
                                dst=input_quantized_sb[i_tile_S].ap(
                                    pattern=[
                                        [H, nl.tile_size.pmax],
                                        [nl.tile_size.pmax, curr_num_128_H_subtiles_per_weight_block],
                                        [1, s_tile_sz],
                                    ],
                                    offset=weight_load_offset,
                                ),
                                data=input_sb[i_tile_S].ap(
                                    pattern=[
                                        [H, nl.tile_size.pmax],
                                        [nl.tile_size.pmax, curr_num_128_H_subtiles_per_weight_block],
                                        [1, s_tile_sz],
                                    ],
                                    offset=weight_load_offset,
                                ),
                                op0=nl.multiply,
                                operand0=in_scale_tile,
                            )

                            fp8_max_val = _get_fp8_e4m3_max_pos_val()
                            nisa.tensor_scalar(
                                dst=input_quantized_sb[i_tile_S].ap(
                                    pattern=[
                                        [H, nl.tile_size.pmax],
                                        [nl.tile_size.pmax, curr_num_128_H_subtiles_per_weight_block],
                                        [1, s_tile_sz],
                                    ],
                                    offset=weight_load_offset,
                                ),
                                data=input_quantized_sb[i_tile_S].ap(
                                    pattern=[
                                        [H, nl.tile_size.pmax],
                                        [nl.tile_size.pmax, curr_num_128_H_subtiles_per_weight_block],
                                        [1, s_tile_sz],
                                    ],
                                    offset=weight_load_offset,
                                ),
                                op0=nl.minimum,
                                operand0=fp8_max_val,
                                op1=nl.maximum,
                                operand1=-fp8_max_val,
                            )

                    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                        # NOTE: for fp8 input and weights, we need to run double row mode for matmul
                        curr_num_of_256_H_subtiles_per_weight_block = math.ceil(
                            curr_num_128_H_subtiles_per_weight_block / 2
                        )
                        for j_256_subtile_of_weight_load in nl.affine_range(
                            curr_num_of_256_H_subtiles_per_weight_block
                        ):
                            for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                                # Each nl.tile_size.psum_fmax column of I is accumulated to a different PSUM bank.
                                psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                                h_subtile_offset = (
                                    weight_load_block_size_per_H * i_weight_load
                                    + nl.tile_size.pmax * j_256_subtile_of_weight_load
                                )
                                i_offset = nl.tile_size.psum_fmax * k_tile_I

                                h_subtile_sz = min(nl.tile_size.pmax, H - h_subtile_offset)
                                i_tile_sz = min(nl.tile_size.psum_fmax, I - i_offset)

                                input_offset = (
                                    max_num_128_H_subtiles_per_weight_block * i_weight_load
                                    + j_256_subtile_of_weight_load * 2
                                ) * nl.tile_size.pmax
                                weight_offset = j_256_subtile_of_weight_load * 2 * I + k_tile_I * nl.tile_size.psum_fmax
                                # Stationary PSUM tile is input tile:  offset_in_transposed_input_row   + [nl.tile_size.pmax,nl.tile_size.pmax].
                                # Moving PSUM tile is weights_sb tile: offset_in_weights_sbuf           + [nl.tile_size.pmax,nl.tile_size.psum_fmax].

                                nisa.nc_matmul(
                                    stationary=input_quantized_sb[i_tile_S].ap(
                                        pattern=[[H, h_subtile_sz], [nl.tile_size.pmax, 2], [1, s_tile_sz]],
                                        offset=input_offset,
                                    ),
                                    moving=weights_sb[i_weight_load % num_weight_buffers].ap(
                                        pattern=[
                                            [
                                                max_num_128_H_subtiles_per_weight_block * I,
                                                h_subtile_sz,
                                            ],
                                            [I, 2],
                                            [1, i_tile_sz],
                                        ],
                                        offset=weight_offset,
                                    ),
                                    dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                                    perf_mode="double_row",
                                )
                    else:
                        for j_128_subtile_of_weight_load in nl.affine_range(curr_num_128_H_subtiles_per_weight_block):
                            for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                                # Each nl.tile_size.psum_fmax column of I is accumulated to a different PSUM bank.
                                psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                                h_subtile_offset = (
                                    weight_load_block_size_per_H * i_weight_load
                                    + nl.tile_size.pmax * j_128_subtile_of_weight_load
                                )
                                i_offset = nl.tile_size.psum_fmax * k_tile_I

                                h_subtile_sz = min(nl.tile_size.pmax, H - h_subtile_offset)
                                i_tile_sz = min(nl.tile_size.psum_fmax, I - i_offset)

                                # Stationary PSUM tile is input tile:  offset_in_transposed_input_row   + [nl.tile_size.pmax,nl.tile_size.pmax].
                                # Moving PSUM tile is weights_sb tile: offset_in_weights_sbuf           + [nl.tile_size.pmax,nl.tile_size.psum_fmax].
                                nisa.nc_matmul(
                                    stationary=input_sb[i_tile_S][
                                        0:h_subtile_sz, nl.ds(h_subtile_offset, s_tile_sz)
                                    ],  # Use h_subtile_offset as [S,H]->[H,S] was transposed in-place.
                                    moving=weights_sb[i_weight_load % num_weight_buffers][
                                        0:h_subtile_sz,
                                        j_128_subtile_of_weight_load,
                                        nl.ds(nl.tile_size.psum_fmax * k_tile_I, i_tile_sz),
                                    ],
                                    dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                                )
                # End of i_weight_load loop

            #######################################################################################################
            # Step 5: Copy PSUM results from matmult back to SBUF, and optionally apply fused RoPE.
            #######################################################################################################
            # Store results to SBUF before copying them to HBM output_tensor.

            # We have one matmult result per each 512 tile/column of weights stored in psum_buffer[bank_index].
            for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                # Copy results PSUM -> SBUF, apply RoPE fusion, (optionally) add_bias.
                if cfg.fused_rope:
                    _copy_psum_to_sbuf_apply_rope_and_bias(
                        qkv_MM_output_psum=qkv_MM_output_psum,
                        output_sb=output_sb,
                        cos_buffer_sb=cos_buffer_sb,
                        sin_buffer_sb=sin_buffer_sb,
                        rope_intermediate_buffer_sb=rope_intermediate_buffer_sb,
                        cos_cache_hbm=cos_cache_hbm,
                        sin_cache_hbm=sin_cache_hbm,
                        i_tile_S=i_tile_S,
                        s_tile_sz=s_tile_sz,
                        i_batch=i_batch,
                        s_tile_local_offset=s_tile_local_offset,
                        cfg=cfg,
                        dims=dims,
                        bias_sb=bias_sb if cfg.add_bias else None,
                        w_scale_tile=w_scale_tile,
                    )
                # Copy results PSUM -> SBUF, (optionally) add_bias.
                else:
                    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                        for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                            psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I
                            num_i = min(nl.tile_size.psum_fmax, I - nl.tile_size.psum_fmax * k_tile_I)

                            nisa.tensor_copy(
                                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)],
                                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                            )

                        if cfg.add_bias:
                            # dequantize the qkv output
                            nisa.scalar_tensor_tensor(
                                dst=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                data=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 0],
                                op1=nl.add,
                                operand1=bias_sb[0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                            )

                            nisa.scalar_tensor_tensor(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 1],
                                op1=nl.add,
                                operand1=bias_sb[
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                            )

                            nisa.scalar_tensor_tensor(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 2],
                                op1=nl.add,
                                operand1=bias_sb[
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                            )

                        else:
                            # dequantize the qkv output
                            nisa.tensor_scalar(
                                dst=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                data=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 0],
                            )

                            nisa.tensor_scalar(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 1],
                            )

                            nisa.tensor_scalar(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 2],
                            )

                    else:
                        for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                            psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I
                            num_i = min(nl.tile_size.psum_fmax, I - nl.tile_size.psum_fmax * k_tile_I)

                            if cfg.add_bias:
                                nisa.tensor_tensor(
                                    dst=output_sb[i_tile_S][
                                        0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)
                                    ],
                                    data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                                    data2=bias_sb[0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)],
                                    op=nl.add,
                                )
                            else:
                                nisa.tensor_copy(
                                    dst=output_sb[i_tile_S][
                                        0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)
                                    ],
                                    src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                                )

                # End of i_tile_S loop.

            #######################################################################################################
            # Step 6: Store SBUF results back to HBM, using given output layout.
            #######################################################################################################
            # This parts reads from output_matmult_sbuf and writes to out_tensor.

            if cfg.use_kv_quantization and cfg.output_layout == QKVOutputLayout.BSD:
                # KV quantization mode: store Q separately, quantize and store K/V to caches
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                    s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset

                    nisa.dma_copy(
                        dst=q_tensor_hbm.ap(
                            pattern=[[dims.q_dim, s_tile_sz], [1, dims.q_dim]],
                            offset=i_batch * dims.S * dims.q_dim + s_tile_global_offset * dims.q_dim,
                        ),
                        src=output_sb[i_tile_S][0:s_tile_sz, 0 : dims.q_dim],
                        dge_mode=dge_mode.swdge,
                    )

                    # Load slot_mapping for block KV cache
                    if cfg.use_block_kv:
                        slot_mapping_tile_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
                        # slot_mapping shape: (batch, seqlen)
                        # Access continuous elements: slot_mapping[i_batch, s_tile_global_offset:s_tile_global_offset+s_tile_sz]
                        nisa.dma_copy(
                            dst=slot_mapping_tile_sb[0:s_tile_sz, 0:1],
                            src=slot_mapping_hbm.ap(
                                pattern=[[1, s_tile_sz]],
                                offset=i_batch * dims.S + s_tile_global_offset,
                            ),
                            dge_mode=dge_mode.swdge,
                        )

                    _quantize_and_store_kv(
                        output_sb=output_sb[i_tile_S],
                        scale_sb=k_scale_sb,
                        cache_hbm=k_cache_hbm,
                        kv_offset=dims.q_dim,
                        i_batch=i_batch,
                        s_tile_global_offset=s_tile_global_offset,
                        s_tile_sz=s_tile_sz,
                        cfg=cfg,
                        dims=dims,
                        sbm=sbm,
                        slot_mapping_sb=slot_mapping_tile_sb if cfg.use_block_kv else None,
                    )

                    _quantize_and_store_kv(
                        output_sb=output_sb[i_tile_S],
                        scale_sb=v_scale_sb,
                        cache_hbm=v_cache_hbm,
                        kv_offset=dims.q_dim + dims.kv_dim,
                        i_batch=i_batch,
                        s_tile_global_offset=s_tile_global_offset,
                        s_tile_sz=s_tile_sz,
                        cfg=cfg,
                        dims=dims,
                        sbm=sbm,
                        slot_mapping_sb=slot_mapping_tile_sb if cfg.use_block_kv else None,
                    )

            elif cfg.output_layout == QKVOutputLayout.BSD:
                # output_tensor shape: [B, S, I].
                # output_matmult_sbuf contains [128 (S), I].
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                    nisa.dma_copy(
                        dst=output_hbm.ap(
                            pattern=[[I, s_tile_sz], [1, I]],
                            offset=i_batch * dims.S * I + (dims.S_shard_offset + s_tile_local_offset) * I,
                        ),
                        src=output_sb[i_tile_S][0:s_tile_sz, 0:I],
                        dge_mode=dge_mode.swdge,
                    )

            else:  # NBSd = [heads, B, S, head_dim], I = heads * head_dim
                d_head = cast(int, dims.d_head)  # Safe due to validation
                for i_head in range(dims.num_heads):
                    for i_tile_S in range(num_S_tiles_in_block):
                        s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                        s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                        num_d = min(d_head, I - (i_head * d_head))

                        nisa.dma_copy(
                            dst=output_hbm.ap(
                                pattern=[[d_head, s_tile_sz], [1, num_d]],
                                offset=i_head * dims.B * dims.S * d_head
                                + i_batch * dims.S * d_head
                                + (dims.S_shard_offset + s_tile_local_offset) * d_head,
                            ),
                            src=output_sb[i_tile_S].ap(
                                pattern=[[I, s_tile_sz], [1, num_d]],
                                offset=i_head * d_head,
                            ),
                            dge_mode=dge_mode.swdge,
                        )
            sbm.close_scope()  # Deallocate all multi-buffered tensors.
            # End of i_buffer_s loop
        # End of batch loop
    sbm.close_scope()
    return output_hbm


def _qkv_cte_mx_impl(
    input_hbm: nl.ndarray,
    fused_qkv_weights_hbm: nl.ndarray,
    output_hbm: nl.ndarray,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    bias_hbm: Optional[nl.ndarray] = None,
    mlp_prev_hbm: Optional[nl.ndarray] = None,
    attention_prev_hbm: Optional[nl.ndarray] = None,
    gamma_norm_weights_hbm: Optional[nl.ndarray] = None,
    layer_norm_bias_hbm: Optional[nl.ndarray] = None,
    norm_eps: Optional[float] = 1e-6,
    cos_cache_hbm: Optional[nl.ndarray] = None,
    sin_cache_hbm: Optional[nl.ndarray] = None,
    qkv_w_scale: Optional[nl.ndarray] = None,
    qkv_in_scale: Optional[nl.ndarray] = None,
) -> nl.ndarray:
    """
    MX Quantization implementation of QKV CTE kernel.

    Performs QKV projection using MX (Microscaling) quantized weights and activations.
    Supports optional fused operations including normalization, residual addition,
    bias, and RoPE. Uses nisa.quantize_mx for activation quantization and
    nisa.nc_matmul_mx for the matrix multiplication.

    Optionally supports static-quant-via-MX: when a statically-quantized FP8 model is routed
    through the MX engine for quadrow performance, per-tensor dequantization scales
    (qkv_in_scale and qkv_w_scale) are applied post-matmul.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        I: Fused QKV dimension = (num_q_heads + 2*num_kv_heads) * d_head
        H_128_tiles: Number of 128-element tiles in H dimension (H // 512)
        H_pack: Packing factor for MX format (4)

    Args:
        input_hbm (nl.ndarray): [B, S, H], Input hidden states tensor on HBM
        fused_qkv_weights_hbm (nl.ndarray): [H//4, I], MX-quantized fused QKV weights on HBM
        output_hbm (nl.ndarray): Output tensor on HBM, shape depends on cfg.output_layout
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager
        bias_hbm (Optional[nl.ndarray]): [1, I], Optional bias tensor on HBM
        mlp_prev_hbm (Optional[nl.ndarray]): [B, S, H], Optional MLP residual on HBM
        attention_prev_hbm (Optional[nl.ndarray]): [B, S, H], Optional attention residual on HBM
        gamma_norm_weights_hbm (Optional[nl.ndarray]): [1, H], Optional normalization weights on HBM
        layer_norm_bias_hbm (Optional[nl.ndarray]): [1, H], Optional layer norm bias on HBM
        norm_eps (Optional[float]): Epsilon for normalization stability
        cos_cache_hbm (Optional[nl.ndarray]): [B, S, d_head], Optional RoPE cosine cache on HBM
        sin_cache_hbm (Optional[nl.ndarray]): [B, S, d_head], Optional RoPE sine cache on HBM
        qkv_w_scale (Optional[nl.ndarray]): MX weight scales on HBM. Interpretation
            depends on the quantization path:
            - MX per-block scales: [H//32, I], used in nc_matmul_mx for per-block dequantization.
              If None, uses neutral scaling (MX_NEUTRAL_SCALE=127, i.e. 2^0=1.0).
            - MX static dequant: [1, 3] or [128, 3], per-tensor weight dequantization scale
              (one per Q/K/V), applied post-matmul. Used when qkv_in_scale is also provided.
        qkv_in_scale (Optional[nl.ndarray]): [1, 1] or [128, 1], Per-tensor input dequantization
            scale for static-quant FP8 models routed through MX engine. When provided,
            qkv_w_scale is interpreted as per-tensor dequant w_scale ([1,3] or [128,3])
            and the combined scale (in_scale * w_scale) is applied post-matmul.

    Returns:
        nl.ndarray: Output tensor (same as output_hbm parameter)

    Notes:
        - Input can be pre-swizzled (cfg.is_input_swizzled) for optimized loading
        - Uses S_TILE_SIZE=32 for swizzled input, P_MAX=128 otherwise
        - Weight prefetching is used when SBUF space allows
        - Supports both BSD and NBSd output layouts

    Pseudocode:
        # Step 1: Load input and apply optional normalization
        for each S_tile:
            load input_tile from HBM
            if fused_residual_add:
                input_tile += mlp_prev + attention_prev
            if fused_norm:
                input_tile = normalize(input_tile)

        # Step 2: Transpose and swizzle for MX format
        transposed = transpose_for_mx(input_tiles)

        # Step 3: Quantize activations to MX format
        hidden_qtz, hidden_scale = quantize_mx(transposed)

        # Step 4: MX matrix multiplication
        for each weight_block:
            output_psum += nc_matmul_mx(hidden_qtz, weights, scales)

        # Step 5: Apply optional RoPE and bias, copy to output
        for each S_tile:
            if fused_rope:
                apply_rope(output_tile)
            if add_bias:
                output_tile += bias
            store output_tile to HBM
    """

    S_shard = dims.S_shard
    H = dims.H
    H_PACKED = H // 4  # MX packs 4 elements, so weight H dimension is H/4
    I = dims.I

    if dims.S_shard == 0:
        return output_hbm

    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    S_TILE_SIZE = 32 if cfg.is_input_swizzled else P_MAX
    PSUM_BANK_SIZE = _get_psum_bank_size()

    if sbm is None:
        sbm_logger = Logger(name="logger")
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=cfg.total_available_sbuf_space_to_this_kernel,
            use_auto_alloc=cfg.use_auto_allocation,
            logger=sbm_logger,
        )
    sbm.open_scope()

    if cfg.add_bias:
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    # MX constants
    SCALE_P_PER_QUAD = H_pack = 4
    H_128_tiles = H // (P_MAX * H_pack)

    # Load norm weights only for non-swizzled path
    if not cfg.is_input_swizzled:
        zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
        nisa.memset(dst=zero_bias_sb, value=0)

        norm_eps_sb = sbm.alloc_stack((P_MAX, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
        nisa.memset(dst=norm_eps_sb, value=norm_eps)
    else:
        input_view = input_hbm.reshape((dims.B, dims.S * H_pack, H_128_tiles, P_MAX))

    _is_fp8_input = input_hbm.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
    _use_static_dequant = qkv_in_scale is not None
    _is_bf16_with_static_quant = input_hbm.dtype == nl.bfloat16 and _use_static_dequant
    _use_dma_xpose_mx_path = (_is_fp8_input or _is_bf16_with_static_quant) and cfg.load_input_with_DMA_transpose

    # When on static dequant path, qkv_w_scale carries the per-tensor dequant w_scale
    # ([1,3] or [128,3]), not per-block MX scales. Extract it and clear so per-block logic sees None.
    mx_static_dequant_w_scale = None
    if _use_static_dequant:
        mx_static_dequant_w_scale = qkv_w_scale
        qkv_w_scale = None

    # DMA transpose path: pack pairs of elements into a wider type for x4-interleaved layout.
    # FP8 (1B)→FP16 (2B), BF16 (2B)→FP32 (4B); reshape [B, S, H] → [B*S*2, H//2] viewed as wider type.
    if _use_dma_xpose_mx_path:
        _dma_xpose_packed_2d = input_hbm.reshape((dims.B * dims.S * 2, H // 2))
        _dma_xpose_view_dtype = nl.float32 if _is_bf16_with_static_quant else nl.float16

    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        gamma_norm_weights_sb = _load_norm_weights_mx(
            norm_weights_hbm=gamma_norm_weights_hbm, cfg=cfg, dims=dims, sbm=sbm
        )
        if cfg.add_layer_norm_bias:
            layer_norm_bias_sb = _load_norm_weights_mx(
                norm_weights_hbm=layer_norm_bias_hbm, cfg=cfg, dims=dims, sbm=sbm
            )

    # Replace the manual multi-buffering calculation with:
    s_multi_buffer_degree, projected_sbuf_taken_space = _multi_buffering_degree_for_seqlen_mx(
        cfg=cfg, dims=dims, sbm=sbm
    )

    S_BLOCK_SIZE = s_multi_buffer_degree * min(S_shard, P_MAX)
    num_blocks_per_S_shard = math.ceil(S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0

    # Weight prefetch decision
    use_weight_prefetch_mx = _use_weight_prefetch_mx(projected_sbuf_taken_space, cfg=cfg, dims=dims)

    # Pre-allocate neutral MX scale tile: used wherever scales are all 1.0 (exponent=0)
    _needs_neutral_scale = _use_dma_xpose_mx_path or qkv_w_scale is None
    if _needs_neutral_scale:
        neutral_scale_sb = sbm.alloc_stack((P_MAX, 1, F_MAX), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=neutral_scale_sb, value=MX_NEUTRAL_SCALE)

    if use_weight_prefetch_mx:
        num_weight_load_blocks_mx = H_128_tiles
        h_tiles_per_block = H_128_tiles

        weight_scale_sb = []
        if qkv_w_scale is not None:
            mx_weight_scale_sb = sbm.alloc_stack((P_MAX, H_128_tiles, I), dtype=nl.uint8, buffer=nl.sbuf)
            for h_tile_idx in nl.affine_range(H_128_tiles):
                for quad_idx in nl.affine_range(4):
                    # HBM layout: [H/32, I] = [64, 512], rows are interleaved by quadrant per H_128_tile
                    # Row index = h_tile_idx * 16 + quad_idx * 4 ... + 4
                    hbm_row_offset = (h_tile_idx * 16 + quad_idx * SCALE_P_PER_QUAD) * I
                    nisa.dma_copy(
                        dst=mx_weight_scale_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), h_tile_idx, :],
                        src=qkv_w_scale.ap(
                            pattern=[[I, SCALE_P_PER_QUAD], [1, I]], offset=hbm_row_offset, dtype=nl.uint8
                        ),
                        dge_mode=dge_mode.swdge,
                    )
            weight_scale_sb.append(mx_weight_scale_sb)

        # Load MX weights
        weights_sb = []
        mx_weights_sb = sbm.alloc_stack(
            (P_MAX, dims.num_128_tiles_per_H // 4, I), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf
        )
        for h_tile_idx in nl.affine_range(dims.num_128_tiles_per_H // 4):
            h_tile_sz = min(P_MAX, H_PACKED - h_tile_idx * P_MAX)
            nisa.dma_copy(
                dst=mx_weights_sb[0:h_tile_sz, h_tile_idx, 0:I],
                src=fused_qkv_weights_hbm.ap(
                    pattern=[[I, h_tile_sz], [1, I]], offset=h_tile_idx * P_MAX * I, dtype=nl.float8_e4m3fn_x4
                ),
                dge_mode=dge_mode.swdge,
            )
        weights_sb.append(mx_weights_sb)
    else:
        # Chunked loading: allocate smaller buffers
        num_weight_load_blocks_mx = H_128_tiles  # Load 1 H_128_tile at a time
        h_tiles_per_block = 1

        weight_scale_sb = []
        weights_sb = []
        for _ in range(NUM_MX_WEIGHT_BUFFERS):
            if qkv_w_scale is not None:
                weight_scale_sb.append(sbm.alloc_stack((P_MAX, h_tiles_per_block, I), dtype=nl.uint8, buffer=nl.sbuf))
            weights_sb.append(sbm.alloc_stack((P_MAX, h_tiles_per_block, I), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf))

    # Load and pre-fuse MX static dequantization scales: combined = w_scale * in_scale
    mx_dequant_sb = None
    if _use_static_dequant:
        mx_dequant_sb = sbm.alloc_stack(shape=(P_MAX, 3), dtype=qkv_in_scale.dtype, buffer=nl.sbuf)
        if mx_static_dequant_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=mx_dequant_sb[0, :], src=mx_static_dequant_w_scale[0, :])
            stream_shuffle_broadcast(mx_dequant_sb, mx_dequant_sb)
        else:
            nisa.dma_copy(dst=mx_dequant_sb, src=mx_static_dequant_w_scale)

        in_scale_sb = sbm.alloc_stack(shape=(P_MAX, 1), dtype=qkv_in_scale.dtype, buffer=nl.sbuf)
        if qkv_in_scale.shape[0] == 1:
            nisa.dma_copy(dst=in_scale_sb[0, :], src=qkv_in_scale[0, :])
            stream_shuffle_broadcast(in_scale_sb, in_scale_sb)
        else:
            nisa.dma_copy(dst=in_scale_sb, src=qkv_in_scale)

        # Pre-fuse: mx_dequant_sb = w_scale * in_scale (single post-matmul multiply)
        nisa.activation(dst=mx_dequant_sb, op=nl.copy, data=mx_dequant_sb, scale=in_scale_sb)

        # BF16 path needs 1/in_scale for static quantization before quantize_mx
        inv_in_scale_sb = None
        if _is_bf16_with_static_quant:
            inv_in_scale_sb = sbm.alloc_stack(shape=(P_MAX, 1), dtype=qkv_in_scale.dtype, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_in_scale_sb, data=in_scale_sb)

    for i_batch in range(dims.B):
        for i_block_S in nl.affine_range(num_blocks_per_S_shard):
            sbm.open_scope()

            s_block_sz = min(S_BLOCK_SIZE, S_shard - S_BLOCK_SIZE * i_block_S)
            num_S_tiles_in_block = math.ceil(s_block_sz / S_TILE_SIZE)

            if _use_dma_xpose_mx_path:
                # DMA transpose path: allocate buffers, loading happens per-weight-block below
                quant_s_tiles = num_S_tiles_in_block * S_TILE_SIZE
                hidden_qtz_sb = sbm.alloc_stack(
                    (P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, align=32
                )

                hidden_scale_sb = None
                # BF16 path: allocate staging buffer for DMA transpose output before quantize_mx
                if _is_bf16_with_static_quant:
                    hidden_scale_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.uint8, buffer=nl.sbuf
                    )
                    bf16_xpose_staging_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, quant_s_tiles * 2), dtype=nl.float32, buffer=nl.sbuf, align=32
                    )

            _use_bf16_mx_path = not _use_dma_xpose_mx_path

            # BF16/swizzled path: load → norm → transpose → swizzle → quantize_mx
            if _use_bf16_mx_path:
                input_sb = []
                for _ in range(num_S_tiles_in_block):
                    if cfg.is_input_swizzled:
                        input_sb.append(
                            sbm.alloc_stack((P_MAX, H_128_tiles, P_MAX), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
                        )
                    else:
                        input_sb.append(sbm.alloc_stack((P_MAX, H), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))

                if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                    square_sum_sb = []
                    for _ in range(num_S_tiles_in_block):
                        square_sum_sb.append(sbm.alloc_stack((P_MAX, 1), dtype=cfg.act_dtype, buffer=nl.sbuf))
                elif cfg.fused_norm_type == NormType.LAYER_NORM:
                    NUM_AGGR_STATS = 2
                    bn_aggr_result_sb = []
                    for _ in range(num_S_tiles_in_block):
                        bn_aggr_result_sb.append(
                            sbm.alloc_stack((P_MAX, NUM_AGGR_STATS), dtype=cfg.act_dtype, buffer=nl.sbuf)
                        )

                # Step 1: Load input and apply normalization
                for i_tile_S in range(num_S_tiles_in_block):
                    sbm.open_scope()

                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * S_TILE_SIZE
                    s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)

                    if cfg.is_input_swizzled:
                        # Swizzled input loading
                        s_tile_global_offset = (
                            i_batch * dims.S * H_pack + (dims.S_shard_offset + s_tile_local_offset) * H_pack
                        )
                        nisa.dma_copy(
                            dst=input_sb[i_tile_S][0 : s_tile_sz * H_pack, 0:H_128_tiles, 0:P_MAX],
                            src=input_view.ap(
                                pattern=[[H_128_tiles * P_MAX, s_tile_sz * H_pack], [P_MAX, H_128_tiles], [1, P_MAX]],
                                offset=s_tile_global_offset * H_128_tiles * P_MAX,
                            ),
                        )
                    else:
                        if cfg.fused_residual_add:
                            s_tile_global_offset = (
                                i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                            )
                            nisa.dma_compute(
                                dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                                srcs=[
                                    input_hbm.ap(pattern=[[H, s_tile_sz], [1, H]], offset=s_tile_global_offset),
                                    mlp_prev_hbm.ap(pattern=[[H, s_tile_sz], [1, H]], offset=s_tile_global_offset),
                                    attention_prev_hbm.ap(
                                        pattern=[[H, s_tile_sz], [1, H]], offset=s_tile_global_offset
                                    ),
                                ],
                                scales=[1.0, 1.0, 1.0],
                                reduce_op=nl.add,
                            )
                        else:
                            s_tile_global_offset = (
                                i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                            )
                            nisa.dma_copy(
                                dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                                src=input_hbm.ap(pattern=[[H, s_tile_sz], [1, H]], offset=s_tile_global_offset),
                                dge_mode=dge_mode.swdge,
                            )

                    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                        _apply_rms_normalization(
                            input_sb[i_tile_S],
                            square_sum_sb[i_tile_S],
                            zero_bias_sb,
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )
                    elif cfg.fused_norm_type == NormType.LAYER_NORM:
                        _compute_layer_norm_stats(
                            input_sb[i_tile_S],
                            bn_aggr_result_sb[i_tile_S],
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )

                        # Apply (x - mean) * rvar * gamma [+ beta]
                        mean = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 0:1]
                        rvar = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 1:2]

                        # (x - mean) * rvar
                        nisa.tensor_scalar(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            data=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            op0=nl.subtract,
                            operand0=mean,
                            op1=nl.multiply,
                            operand1=rvar,
                        )

                    sbm.close_scope()

                # Step 2: Swizzle + Transpose for MX
                # Allocate transposed buffer with correct layout for quantize_mx
                if cfg.is_input_swizzled:
                    transposed_buffer_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, num_S_tiles_in_block * P_MAX), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf
                    )

                    n_psum_banks = NUM_HW_PSUM_BANKS  # Always use all 8 banks
                    n_transposes_per_bank = math.ceil(H_128_tiles / n_psum_banks)

                    # Each bank holds multiple [128,128] transposes concatenated in free dim
                    transpose_psum = []
                    for bank_id in range(n_psum_banks):
                        transpose_psum.append(
                            nl.ndarray(
                                (P_MAX, n_transposes_per_bank * P_MAX),
                                dtype=nl.bfloat16,
                                buffer=nl.psum,
                                address=(0, bank_id * PSUM_BANK_SIZE),
                            )
                        )

                    for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                        for bank in range(n_psum_banks):
                            # Transpose all H tiles that map to this bank
                            for idx in range(n_transposes_per_bank):
                                h_tile = bank * n_transposes_per_bank + idx
                                if h_tile < H_128_tiles:
                                    nisa.nc_transpose(
                                        data=input_sb[i_tile_S][0:P_MAX, h_tile, 0:P_MAX],
                                        dst=transpose_psum[bank][0:P_MAX, idx * P_MAX : (idx + 1) * P_MAX],
                                    )

                            # Evict full bank - copy all transposes at once
                            for idx in range(n_transposes_per_bank):
                                h_tile = bank * n_transposes_per_bank + idx
                                if h_tile < H_128_tiles:
                                    nisa.tensor_copy(
                                        dst=transposed_buffer_sb[0:P_MAX, h_tile, nl.ds(i_tile_S * P_MAX, P_MAX)],
                                        src=transpose_psum[bank][0:P_MAX, idx * P_MAX : (idx + 1) * P_MAX],
                                    )

                else:
                    transposed_buffer_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, num_S_tiles_in_block * P_MAX * H_pack),
                        dtype=cfg.compute_mm_dtype,
                        buffer=nl.sbuf,
                    )

                    # Pre-allocate PSUM banks ONCE outside all loops
                    transpose_psum = []
                    for bank_id in range(H_pack):
                        if cfg.use_auto_allocation:
                            transpose_psum.append(nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum))
                        else:
                            transpose_psum.append(
                                nl.ndarray(
                                    (P_MAX, P_MAX),
                                    dtype=nl.bfloat16,
                                    buffer=nl.psum,
                                    address=(0, bank_id * PSUM_BANK_SIZE),
                                )
                            )

                    for h_tile in nl.affine_range(H_128_tiles):
                        for s_tile in nl.affine_range(num_S_tiles_in_block):
                            # Transpose all 4 sub-tiles
                            for h_sub in nl.affine_range(H_pack):
                                src_h_base = h_tile * P_MAX * H_pack + h_sub
                                nisa.nc_transpose(
                                    data=input_sb[s_tile].ap(pattern=[[H, P_MAX], [H_pack, P_MAX]], offset=src_h_base),
                                    dst=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                )

                            # Copy to output with interleaved pattern, optionally applying gamma
                            for h_sub in nl.affine_range(H_pack):
                                out_base = s_tile * P_MAX * H_pack + h_sub
                                dst_ap = transposed_buffer_sb.ap(
                                    pattern=[
                                        [H_128_tiles * num_S_tiles_in_block * P_MAX * H_pack, P_MAX],
                                        [H_pack, P_MAX],
                                    ],
                                    offset=h_tile * num_S_tiles_in_block * P_MAX * H_pack + out_base,
                                )

                                if (
                                    cfg.fused_norm_type == NormType.RMS_NORM
                                    or cfg.fused_norm_type == NormType.LAYER_NORM
                                ):
                                    gamma_tile_index = h_tile * H_pack + h_sub
                                    if not cfg.add_layer_norm_bias:
                                        nisa.tensor_scalar(
                                            dst=dst_ap,
                                            data=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                            op0=nl.multiply,
                                            operand0=gamma_norm_weights_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                                        )
                                    else:
                                        nisa.tensor_scalar(
                                            dst=dst_ap,
                                            data=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                            op0=nl.multiply,
                                            operand0=gamma_norm_weights_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                                            op1=nl.add,
                                            operand1=layer_norm_bias_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                                        )
                                else:  # NO_NORM or RMS_NORM_SKIP_GAMMA
                                    nisa.tensor_copy(
                                        dst=dst_ap,
                                        src=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                    )

                # Step 3: Quantize using nisa.quantize_mx
                quant_s_tiles = num_S_tiles_in_block * S_TILE_SIZE
                hidden_qtz_sb = sbm.alloc_stack(
                    (P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf
                )
                hidden_scale_sb = sbm.alloc_stack((P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.uint8, buffer=nl.sbuf)

                quant_src_size = quant_s_tiles * H_pack
                nisa.quantize_mx(
                    src=transposed_buffer_sb[0:P_MAX, 0:H_128_tiles, 0:quant_src_size],
                    dst=hidden_qtz_sb[0:P_MAX, 0:H_128_tiles, 0:quant_s_tiles],
                    dst_scale=hidden_scale_sb[0:P_MAX, 0:H_128_tiles, 0:quant_s_tiles],
                )

            # RoPE buffers: allocated for both BF16 and FP8 paths (used in Step 5 post-matmul)
            if cfg.fused_rope:
                cos_buffer_sb = []
                sin_buffer_sb = []
                rope_intermediate_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    cos_buffer_sb.append(
                        sbm.alloc_stack((P_MAX, dims.d_head), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
                    )
                    sin_buffer_sb.append(
                        sbm.alloc_stack((P_MAX, dims.d_head // 2), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
                    )
                    rope_intermediate_buffer_sb.append(
                        sbm.alloc_stack((P_MAX, dims.d_head * 2), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
                    )

            # Step 4: MX Matrix Multiplication
            num_output_s_tiles = max(1, num_S_tiles_in_block // 4) if cfg.is_input_swizzled else num_S_tiles_in_block
            output_sb = []
            for _ in range(num_output_s_tiles):
                output_sb.append(sbm.alloc_stack((P_MAX, I), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))

            qkv_MM_num_psum_banks_needed = dims.num_512_tiles_per_I * num_output_s_tiles
            qkv_MM_output_psum = []
            for bank_id in nl.affine_range(qkv_MM_num_psum_banks_needed):
                if cfg.use_auto_allocation:
                    qkv_MM_output_psum.append(nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum))
                else:
                    qkv_MM_output_psum.append(
                        nl.ndarray(
                            (P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum, address=(0, bank_id * PSUM_BANK_SIZE)
                        )
                    )

            for i_weight_block in nl.affine_range(num_weight_load_blocks_mx):
                buf_idx = i_weight_block % NUM_MX_WEIGHT_BUFFERS if not use_weight_prefetch_mx else 0

                if not use_weight_prefetch_mx:
                    # Load this chunk: weights and scales for H_128_tile = i_weight_block
                    h_tile_idx = i_weight_block
                    h_tile_sz = min(P_MAX, H_PACKED - h_tile_idx * P_MAX)

                    if qkv_w_scale is not None:
                        for quad_idx in nl.affine_range(4):
                            hbm_row_offset = (h_tile_idx * 16 + quad_idx * SCALE_P_PER_QUAD) * I
                            nisa.dma_copy(
                                dst=weight_scale_sb[buf_idx][nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), 0, :],
                                src=qkv_w_scale.ap(
                                    pattern=[[I, SCALE_P_PER_QUAD], [1, I]], offset=hbm_row_offset, dtype=nl.uint8
                                ),
                                dge_mode=dge_mode.swdge,
                            )

                    # Load weights for this h_tile
                    nisa.dma_copy(
                        dst=weights_sb[buf_idx][0:h_tile_sz, 0, 0:I],
                        src=fused_qkv_weights_hbm.ap(
                            pattern=[[I, h_tile_sz], [1, I]], offset=h_tile_idx * P_MAX * I, dtype=nl.float8_e4m3fn_x4
                        ),
                        dge_mode=dge_mode.swdge,
                    )

                # DMA transpose: load one H_128_tile for each S tile
                for i_tile_S in nl.affine_range(num_output_s_tiles):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                    s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                    if _use_dma_xpose_mx_path:
                        s_global = dims.S_shard_offset + s_tile_local_offset
                        src_offset = i_batch * dims.S * 2 * H_PACKED + s_global * 2 * H_PACKED + i_weight_block * P_MAX

                        # FP8: transpose directly into hidden_qtz_sb (no quantize_mx needed)
                        # BF16: transpose into staging buffer (quantize_mx runs after all H tiles)
                        xpose_dst_buf = bf16_xpose_staging_sb if _is_bf16_with_static_quant else hidden_qtz_sb
                        dst_offset = i_weight_block * quant_s_tiles * 2 + i_tile_S * S_TILE_SIZE * 2
                        nisa.dma_transpose(
                            src=_dma_xpose_packed_2d.ap(
                                pattern=[[H_PACKED, s_tile_sz * 2], [1, 1], [1, 1], [1, P_MAX]],
                                offset=src_offset,
                                dtype=_dma_xpose_view_dtype,
                            ),
                            dst=xpose_dst_buf.ap(
                                pattern=[[quant_s_tiles * 2 * H_128_tiles, P_MAX], [1, 1], [1, 1], [1, s_tile_sz * 2]],
                                offset=dst_offset,
                                dtype=_dma_xpose_view_dtype,
                            ),
                        )

                        # BF16: quantize the transposed staging data for this (H_128_tile, S_tile)
                        if _is_bf16_with_static_quant:
                            _static_mx_quantize_bf16_hidden_tile(
                                bf16_xpose_staging_sb=bf16_xpose_staging_sb,
                                hidden_qtz_sb=hidden_qtz_sb,
                                hidden_scale_sb=hidden_scale_sb,
                                inv_in_scale_sb=inv_in_scale_sb,
                                i_weight_block=i_weight_block,
                                i_tile_S=i_tile_S,
                                s_tile_sz=s_tile_sz,
                                quant_s_tiles=quant_s_tiles,
                                H_128_tiles=H_128_tiles,
                                H_pack=H_pack,
                                S_TILE_SIZE=S_TILE_SIZE,
                            )

                    for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                        i_tile_sz = min(F_MAX, I - k_tile_I * F_MAX)
                        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                        # Key mapping: h_idx in buffer depends on prefetch mode
                        h_idx_in_buf = i_weight_block if use_weight_prefetch_mx else 0

                        # Stationary scale: neutral (pre-quantized FP8) or computed (BF16 quantize_mx)
                        if _is_fp8_input:
                            stat_scale = neutral_scale_sb[0:P_MAX, 0, 0:s_tile_sz]
                        else:
                            stat_scale = hidden_scale_sb[0:P_MAX, i_weight_block, nl.ds(i_tile_S * P_MAX, s_tile_sz)]

                        # Moving scale: neutral (no weight scales) or loaded from HBM
                        if qkv_w_scale is None:
                            mov_scale = neutral_scale_sb[0:P_MAX, 0, 0:i_tile_sz]
                        else:
                            mov_scale = weight_scale_sb[buf_idx][
                                0:P_MAX, h_idx_in_buf, nl.ds(k_tile_I * F_MAX, i_tile_sz)
                            ]

                        nisa.nc_matmul_mx(
                            dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                            stationary=hidden_qtz_sb[0:P_MAX, i_weight_block, nl.ds(i_tile_S * P_MAX, s_tile_sz)],
                            moving=weights_sb[buf_idx][0:P_MAX, h_idx_in_buf, nl.ds(k_tile_I * F_MAX, i_tile_sz)],
                            stationary_scale=stat_scale,
                            moving_scale=mov_scale,
                        )

            # Step 5: Copy PSUM to SBUF, apply RoPE/bias
            for i_tile_S in nl.affine_range(num_output_s_tiles):
                s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                # RoPE with optional static dequant and optional bias.
                if cfg.fused_rope:
                    _copy_psum_to_sbuf_apply_rope_and_bias(
                        qkv_MM_output_psum=qkv_MM_output_psum,
                        output_sb=output_sb,
                        cos_buffer_sb=cos_buffer_sb,
                        sin_buffer_sb=sin_buffer_sb,
                        rope_intermediate_buffer_sb=rope_intermediate_buffer_sb,
                        cos_cache_hbm=cos_cache_hbm,
                        sin_cache_hbm=sin_cache_hbm,
                        i_tile_S=i_tile_S,
                        s_tile_sz=s_tile_sz,
                        i_batch=i_batch,
                        s_tile_local_offset=s_tile_local_offset,
                        cfg=cfg,
                        dims=dims,
                        bias_sb=bias_sb if cfg.add_bias else None,
                        w_scale_tile=mx_dequant_sb,
                    )
                # Static dequant with optional bias.
                elif _use_static_dequant:
                    _evict_psum_to_sbuf_static_dequant_apply_bias(
                        output_sb=output_sb[i_tile_S],
                        qkv_MM_output_psum=qkv_MM_output_psum,
                        s_tile_sz=s_tile_sz,
                        i_tile_S=i_tile_S,
                        mx_dequant_sb=mx_dequant_sb,
                        bias_sb=bias_sb if cfg.add_bias else None,
                        q_dim=dims.num_q_heads * dims.d_head,
                        kv_dim=dims.num_kv_heads * dims.d_head,
                        I=I,
                        num_512_tiles_per_I=dims.num_512_tiles_per_I,
                    )
                # No RoPE, no static dequant with optional bias.
                else:
                    _evict_psum_to_sbuf_apply_bias(
                        output_sb=output_sb[i_tile_S],
                        qkv_MM_output_psum=qkv_MM_output_psum,
                        s_tile_sz=s_tile_sz,
                        i_tile_S=i_tile_S,
                        bias_sb=bias_sb if cfg.add_bias else None,
                        I=I,
                        num_512_tiles_per_I=dims.num_512_tiles_per_I,
                    )

            # Step 6: Store output to HBM
            if cfg.output_layout == QKVOutputLayout.BSD:
                for i_tile_S in range(num_output_s_tiles):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                    s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                    nisa.dma_copy(
                        dst=output_hbm.ap(
                            pattern=[[I, s_tile_sz], [1, I]],
                            offset=i_batch * dims.S * I + (dims.S_shard_offset + s_tile_local_offset) * I,
                        ),
                        src=output_sb[i_tile_S][0:s_tile_sz, 0:I],
                        dge_mode=dge_mode.swdge,
                    )
            else:  # NBSd = [heads, B, S, head_dim]
                d_head = cast(int, dims.d_head)
                for i_head in range(dims.num_heads):
                    for i_tile_S in range(num_output_s_tiles):
                        s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                        s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)
                        num_d = min(d_head, I - (i_head * d_head))

                        nisa.dma_copy(
                            dst=output_hbm.ap(
                                pattern=[[d_head, s_tile_sz], [1, num_d]],
                                offset=i_head * dims.B * dims.S * d_head
                                + i_batch * dims.S * d_head
                                + (dims.S_shard_offset + s_tile_local_offset) * d_head,
                            ),
                            src=output_sb[i_tile_S].ap(
                                pattern=[[I, s_tile_sz], [1, num_d]],
                                offset=i_head * d_head,
                            ),
                            dge_mode=dge_mode.swdge,
                        )
            sbm.close_scope()
    sbm.close_scope()
    return output_hbm


def _evict_psum_to_sbuf_apply_bias(
    output_sb: nl.ndarray,
    qkv_MM_output_psum: list,
    s_tile_sz: int,
    i_tile_S: int,
    bias_sb: Optional[nl.ndarray],
    I: int,
    num_512_tiles_per_I: int,
) -> None:
    """Plain PSUM eviction: copy PSUM banks to SBUF, optionally fusing bias add.

    No dequant scaling, no RoPE — just a straight copy (± bias) across all banks.

    Args:
        output_sb: Destination buffer in SBUF for this S tile
        qkv_MM_output_psum: PSUM bank list from matmul accumulation
        s_tile_sz: Active rows in the S tile
        i_tile_S: S tile index (for PSUM bank mapping)
        bias_sb: [P_MAX, I] bias in SBUF, or None
        I: Total fused QKV dimension
        num_512_tiles_per_I: Number of PSUM banks per S tile
    """
    F_MAX = 512
    for k_tile_I in nl.affine_range(num_512_tiles_per_I):
        psum_accumulation_bank_id = i_tile_S * num_512_tiles_per_I + k_tile_I
        num_i = min(F_MAX, I - k_tile_I * F_MAX)

        if bias_sb is not None:
            nisa.tensor_tensor(
                dst=output_sb[0:s_tile_sz, nl.ds(k_tile_I * F_MAX, num_i)],
                data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                data2=bias_sb[0:s_tile_sz, nl.ds(k_tile_I * F_MAX, num_i)],
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=output_sb[0:s_tile_sz, nl.ds(k_tile_I * F_MAX, num_i)],
                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
            )


def _evict_psum_to_sbuf_static_dequant_apply_bias(
    output_sb: nl.ndarray,
    qkv_MM_output_psum: list,
    s_tile_sz: int,
    i_tile_S: int,
    mx_dequant_sb: nl.ndarray,
    bias_sb: Optional[nl.ndarray],
    q_dim: int,
    kv_dim: int,
    I: int,
    num_512_tiles_per_I: int,
) -> None:
    """Single-pass PSUM eviction with fused dequant scale (± bias) per Q/K/V segment.

    Banks straddling segment boundaries are split to apply the correct per-segment scale.
    Fuses the dequant multiply (and optional bias add) into the PSUM read, eliminating
    a separate copy + scale pass.

    Args:
        output_sb: Destination buffer in SBUF for this S tile
        qkv_MM_output_psum: PSUM bank list from matmul accumulation
        s_tile_sz: Active rows in the S tile
        i_tile_S: S tile index (for PSUM bank mapping)
        mx_dequant_sb: [P_MAX, 3], Pre-fused combined dequant scale per Q/K/V
        bias_sb: [P_MAX, I] bias in SBUF, or None
        q_dim: Q segment size (num_q_heads * d_head)
        kv_dim: KV segment size (num_kv_heads * d_head)
        I: Total fused QKV dimension
        num_512_tiles_per_I: Number of PSUM banks per S tile
    """
    F_MAX = 512
    seg_starts = [0, q_dim, q_dim + kv_dim]
    seg_ends = [q_dim, q_dim + kv_dim, I]

    for seg_idx in range(NUM_QKV_SEGMENTS):
        seg_start = seg_starts[seg_idx]
        seg_end = seg_ends[seg_idx]
        first_bank = seg_start // F_MAX
        last_bank = (seg_end - 1) // F_MAX
        for k_tile_I in range(first_bank, last_bank + 1):
            bank_start = k_tile_I * F_MAX
            col_start = max(seg_start, bank_start)
            col_end = min(seg_end, min(bank_start + F_MAX, I))
            num_cols = col_end - col_start
            psum_offset = col_start - bank_start
            psum_accumulation_bank_id = i_tile_S * num_512_tiles_per_I + k_tile_I

            if bias_sb is not None:
                nisa.scalar_tensor_tensor(
                    dst=output_sb[0:s_tile_sz, nl.ds(col_start, num_cols)],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_offset, num_cols)],
                    op0=nl.multiply,
                    operand0=mx_dequant_sb[0:s_tile_sz, seg_idx],
                    op1=nl.add,
                    operand1=bias_sb[0:s_tile_sz, nl.ds(col_start, num_cols)],
                )
            else:
                nisa.tensor_scalar(
                    dst=output_sb[0:s_tile_sz, nl.ds(col_start, num_cols)],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_offset, num_cols)],
                    op0=nl.multiply,
                    operand0=mx_dequant_sb[0:s_tile_sz, seg_idx],
                )


def _static_mx_quantize_bf16_hidden_tile(
    bf16_xpose_staging_sb: nl.ndarray,
    hidden_qtz_sb: nl.ndarray,
    hidden_scale_sb: nl.ndarray,
    inv_in_scale_sb: nl.ndarray,
    i_weight_block: int,
    i_tile_S: int,
    s_tile_sz: int,
    quant_s_tiles: int,
    H_128_tiles: int,
    H_pack: int,
    S_TILE_SIZE: int,
) -> None:
    """Static-quantize and MX-quantize one (H_128_tile, S_tile) from the BF16 DMA transpose staging buffer.

    Applies static quantization in-place on the bfloat16 AP view (divide by in_scale,
    clamp to FP8 range), then runs quantize_mx to produce fp8_x4 data and uint8 scales
    for nc_matmul_mx.

    Args:
        bf16_xpose_staging_sb: Float32 staging buffer holding transposed BF16 data
        hidden_qtz_sb: Destination for quantized fp8_x4 data
        hidden_scale_sb: Destination for MX uint8 scales
        inv_in_scale_sb: [P_MAX, 1], reciprocal of static input dequant scale
        i_weight_block: Current H_128_tile index
        i_tile_S: Current S tile index
        s_tile_sz: Active rows in the S tile
        quant_s_tiles: Total S tiles allocated in buffers
        H_128_tiles: Number of H_128 tiles
        H_pack: Packing factor (4)
        S_TILE_SIZE: S tile size constant
    """
    P_MAX = nl.tile_size.pmax
    quant_src_size = s_tile_sz * H_pack

    src_ap = bf16_xpose_staging_sb.ap(
        pattern=[
            [quant_s_tiles * H_pack * H_128_tiles, P_MAX],
            [1, 1],
            [1, quant_src_size],
        ],
        offset=i_weight_block * quant_s_tiles * H_pack + i_tile_S * S_TILE_SIZE * H_pack,
        dtype=nl.bfloat16,
    )

    # Static quantize in-place: clamp(data / in_scale, ±fp8_max)
    fp8_max = _get_fp8_e4m3_max_pos_val()
    nisa.tensor_scalar(
        dst=src_ap,
        data=src_ap,
        op0=nl.multiply,
        operand0=inv_in_scale_sb[0:P_MAX, 0:1],
        op1=nl.minimum,
        operand1=fp8_max,
    )
    nisa.tensor_scalar(
        dst=src_ap,
        data=src_ap,
        op0=nl.maximum,
        operand0=-fp8_max,
    )

    nisa.quantize_mx(
        src=src_ap,
        dst=hidden_qtz_sb[0:P_MAX, i_weight_block, nl.ds(i_tile_S * P_MAX, s_tile_sz)],
        dst_scale=hidden_scale_sb[0:P_MAX, i_weight_block, nl.ds(i_tile_S * P_MAX, s_tile_sz)],
    )


def _load_and_broadcast_bias(
    bias_hbm: nl.ndarray, cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager
) -> nl.ndarray:
    """
    Loads bias with shape [1,I] to SBUF and broadcasts it to [nl.tile_size.pmax, I], using stream_shuffle.

    Returns allocated SBUF bias tensor.
    Note: User is responsible for deallocating SBUF tensor.
    """
    # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
    bias_sb = sbm.alloc_stack((nl.tile_size.pmax, dims.I), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias_sb[0:1, 0 : dims.I],
        src=bias_hbm[0:1, 0 : dims.I],
        dge_mode=dge_mode.swdge,
    )
    # Stream Shuffle works on 32 partitions only, apply it nl.tile_size.pmax // 32 = 4 times.
    NUM_BROADCASTS = nl.tile_size.pmax // MAX_STREAM_SHUFFLE_PARTITIONS
    for broadcast_idx in nl.affine_range(NUM_BROADCASTS):
        nisa.nc_stream_shuffle(
            dst=bias_sb[
                nl.ds(broadcast_idx * MAX_STREAM_SHUFFLE_PARTITIONS, MAX_STREAM_SHUFFLE_PARTITIONS),
                0 : dims.I,
            ],
            src=bias_sb[0:1, 0 : dims.I],
            shuffle_mask=[0] * MAX_STREAM_SHUFFLE_PARTITIONS,
        )
    return bias_sb


def _load_norm_weights(
    norm_weights_hbm: nl.ndarray,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Loads norm_weights with shape [H] to SBUF as [nl.tile_size.pmax, H // nl.tile_size.pmax].

    Returns allocated SBUF norm_weights tensor.
    Note: User is responsible for deallocating SBUF tensor.

    Used by RMS_NORM and LAYER_NORM to load gamma_weights_hbm to SBUF.
    In addition, may be used in LAYER_NORM to load layer_norm_bias to SBUF.
    """
    # norm_weights_hbm have 1D shape [H], make it 2-D to make NKI loads easier.
    norm_weights_hbm = norm_weights_hbm.reshape((dims.H, 1))
    # We load norm_weights into SBUF as a 2-D tensor of shape [128, H // nl.tile_size.pmax].
    # Note: We later do the multiplication on transposed input [H, S], so the math works out.
    norm_elements_in_free_dim = dims.num_128_tiles_per_H
    norm_weights_sb = sbm.alloc_stack(
        (nl.tile_size.pmax, norm_elements_in_free_dim), dtype=cfg.act_dtype, buffer=nl.sbuf
    )

    # Load in tiles of [128,1] to SBUF, now H is the first dimension (DMA broadcasted).
    # It is loaded in a way that a single norm tile uses all nl.tile_size.pmax partitions, and has 1 element per partition.
    for i_gamma_tile in range(norm_elements_in_free_dim):
        nisa.dma_copy(
            dst=norm_weights_sb[0 : nl.tile_size.pmax, nl.ds(i_gamma_tile, 1)],
            src=norm_weights_hbm[nl.ds(i_gamma_tile * nl.tile_size.pmax, nl.tile_size.pmax), 0:1],
            dge_mode=dge_mode.swdge,
        )
    return norm_weights_sb


def _load_norm_weights_mx(
    norm_weights_hbm: nl.ndarray,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Load normalization weights to SBUF in swizzled format for MX path.

    Loads norm_weights from HBM with shape [1, H] and rearranges them into
    [P_MAX, H_128_tiles * H_pack] format suitable for MX quantized computation.
    Uses stride-4 gather pattern to match MX data layout.

    Args:
        norm_weights_hbm (nl.ndarray): [1, H], Normalization weights on HBM
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager

    Returns:
        nl.ndarray: [P_MAX, H_128_tiles * H_pack], Swizzled norm weights in SBUF

    Notes:
        - H_pack = 4 for MX format
        - H_128_tiles = H // (P_MAX * H_pack)
        - User is responsible for deallocating SBUF tensor
    """
    P_MAX = nl.tile_size.pmax
    H_pack = 4
    H_128_tiles = dims.H // (P_MAX * H_pack)

    # Reshape to [1, H] so stride-4 gather works on free dimension
    norm_weights_hbm = norm_weights_hbm.reshape((1, dims.H))
    gamma_sb = sbm.alloc_stack((P_MAX, H_128_tiles * H_pack), dtype=cfg.act_dtype, buffer=nl.sbuf)

    for h_tile_idx in range(H_128_tiles):
        for h_sub_idx in range(H_pack):
            src_offset = h_tile_idx * P_MAX * H_pack + h_sub_idx
            dst_col = h_tile_idx * H_pack + h_sub_idx
            # Load 128 elements with stride-4: gamma[src_offset + 0*4], gamma[src_offset + 1*4], ...
            nisa.dma_copy(
                dst=gamma_sb[0:P_MAX, nl.ds(dst_col, 1)],
                src=norm_weights_hbm.ap(
                    pattern=[[H_pack, P_MAX], [1, 1]],  # stride-4 on partition dim, 1 element free dim
                    offset=src_offset,
                ),
                dge_mode=dge_mode.swdge,
            )
    return gamma_sb


def _multi_buffering_degree_for_seqlen(
    cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager, qkv_in_scale: Optional[nl.ndarray] = None
) -> Tuple[int, int]:
    """
    Compute maximum multi-buffering degree that we can use for SEQLEN without over-flowing SBUF or PSUM space.

    WARNING: This is not independently useful function, its correctness is based on the tensor allocation that comes after it.
    This is a 'lookahead' function.
    NOTE: If any additional tensors are added in the kernel, this function needs to be updated.

    Goal is to find the MAX "multi_buffer_degree" such that:
    (multi_buffer_degree * X) + Y < sbuf_space (per_partition), where
    X = sbuf_space_taken_by_tensors_about_to_be_multi_buffered (per_partition)
    Y = sbuf_space_taken_by_live_non_buffered_tensors (per_partition)

    Note: cfg.total_available_sbuf_space_to_this_kernel gives SBUF space PER_PARTITION.

    Assumes:
        * Weight prefetching decision is made after we choose multi-buffering degree.
        * For SBUF space calculations, we take into account the space taken by non-prefeched weights.
        * All globally allocated tensors have already been allocated, so that we can use sbm.get_free_space().
            Note: Still need to do look-ahead calculation for the tensors after call to this function is made.

    Returns: multi_buffer_degree, projected_total_sbuf_space_taken (including all tensors).
    """

    # Cannot multi-buffer more than dims.S_shard / nl.tile_size.pmax, e.g. if S_shard=256, best we can do is 2.
    s_multi_buffer_degree = 1
    s_multi_buffer_degree = min(math.ceil(dims.S_shard / nl.tile_size.pmax), dims.MAX_S_MULTI_BUFFER_DEGREE)

    # ------------------- Make sure multi-buffering does not cause SBUF overflow --------------------#

    # ------------------------ SBUF Space Taken by Non Buffered Tensors ----------------------------#
    #     This is the space that will be consumed by tensors we will not multi-buffer
    #     This calculation assumes we are not pre-fetching weights (this can be decided after buffering)
    #     These same constants are used in the allocation of weight tensor.

    # Sum up sizes of: # zero_bias_sb, norm_eps_sb, bias_sb, gamma_weights_sb, l
    #   layer_norm_bias_sb, act_reduce_sum, bn_stats_result, and weights_sb (non-prefetched)
    sbuf_tile_space_non_buffered = 0
    # zero_bias_sb, norm_eps_sb, bias_sb, gamma_weights_sb, layer_norm_bias_sb, act_reduce_sum, bn_stats_result.
    sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # zero_bias_sb (nl.tile_size.pmax, 1)
    sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # norm_eps_sb (nl.tile_size.pmax, 1)
    if cfg.add_bias:
        sbuf_tile_space_non_buffered += dims.I * sizeinbytes(
            cfg.compute_mm_dtype
        )  # bias_sb (nl.tile_size.pmax, dims.I)
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        # gamma_weights_sb (nl.tile_size.pmax, num_128_tiles_per_H)
        sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)
        if cfg.add_layer_norm_bias:
            # layer_norm_bias_sb (nl.tile_size.pmax, num_128_tiles_per_H)
            sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)

    # act_reduce_sum and bn_stats_result_sb appear in the loop.
    # sbuf_tile_space_non_buffered = nl.tile_size.total_available_sbuf_size - sbm.get_free_space() # space taken so far.
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.act_dtype)  # act_reduce_sum (nl.tile_size.pmax, 1)
    if cfg.fused_norm_type == NormType.LAYER_NORM:
        # bn_stats_result_sb (nl.tile_size.pmax, 6*NUM_512_BN_STATS_TILES_H)
        BN_STATS_FMAX = 512  # nl.tile_size.bn_stats_fmax  # 512
        NUM_512_BN_STATS_TILES_H = math.ceil(dims.H / BN_STATS_FMAX)
        sbuf_tile_space_non_buffered += 6 * NUM_512_BN_STATS_TILES_H * sizeinbytes(cfg.act_dtype)

    weights_space_per_partition = (
        dims.NUM_WEIGHT_BUFFERS_DEFAULT
        * (dims.I * math.ceil(dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT / 128))
        * sizeinbytes(cfg.compute_mm_dtype)
    )
    sbuf_tile_space_non_buffered += weights_space_per_partition

    # --------------------SBUF Space Taken By Tensors We Will be Multi-Buffering ---------------------#

    sbuf_tile_space_pre_buffering = _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
        cfg=cfg, dims=dims, sbm=sbm, qkv_in_scale=qkv_in_scale
    )

    # Note: cfg.total_available_sbuf_space_to_this_kernel is total_available_sbuf_space PER PARTITION.
    max_s_buffer_without_exceeding_sbuf = (
        cfg.total_available_sbuf_space_to_this_kernel - sbuf_tile_space_non_buffered
    ) // sbuf_tile_space_pre_buffering
    s_multi_buffer_degree = min(s_multi_buffer_degree, max_s_buffer_without_exceeding_sbuf)

    # Step (3) Ensure multi-buffering does not exceed number of PSUM banks.
    # Later we use NUM_512_TILES_PER_H * s_multi_buffer_degree for psum_banks. (NUM_512_TILES_PER_H <= 4, since I <= 4096)
    # Ensure NUM_512_TILES_PER_H * s_multi_buffer_degree <= 8
    MAX_PSUM_TILING_GROUPS = NUM_HW_PSUM_BANKS // dims.num_512_tiles_per_I
    s_multi_buffer_degree = min(s_multi_buffer_degree, MAX_PSUM_TILING_GROUPS)

    projected_sbuf_taken_space = s_multi_buffer_degree * sbuf_tile_space_pre_buffering + sbuf_tile_space_non_buffered
    return s_multi_buffer_degree, projected_sbuf_taken_space


def _multi_buffering_degree_for_seqlen_mx(cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager) -> Tuple[int, int]:
    """
    Compute multi-buffering degree for MX path, accounting for MX-specific buffers.

    Calculates the maximum number of sequence tiles that can be processed simultaneously
    without exceeding SBUF or PSUM space constraints. Accounts for MX-specific buffers
    including transposed input, quantized activations, and scales.

    Args:
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager

    Returns:
        Tuple[int, int]: (multi_buffer_degree, projected_sbuf_space)
            - multi_buffer_degree: Number of S tiles to process simultaneously
            - projected_sbuf_space: Estimated total SBUF space usage

    Notes:
        - Considers MX weight buffers in chunked mode
        - Accounts for transposed_buffer, hidden_qtz, and hidden_scale per S tile
        - Respects PSUM bank constraints (NUM_HW_PSUM_BANKS // num_512_tiles_per_I)
    """
    P_MAX = nl.tile_size.pmax
    H_pack = 4
    H_128_tiles = dims.H // (P_MAX * H_pack)

    s_multi_buffer_degree = min(math.ceil(dims.S_shard / P_MAX), dims.MAX_S_MULTI_BUFFER_DEGREE)

    # Non-buffered space: global allocations + MX weight buffers (chunked)
    NUM_MX_WEIGHT_BUFFERS_LOCAL = 2
    sbuf_tile_space_non_buffered = 0
    if not cfg.is_input_swizzled:
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # zero_bias_sb
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # norm_eps_sb

    if cfg.add_bias:
        sbuf_tile_space_non_buffered += dims.I * sizeinbytes(cfg.compute_mm_dtype)
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)
        if cfg.add_layer_norm_bias:
            sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)

    # MX weight buffers (chunked mode - worst case)
    sbuf_tile_space_non_buffered += NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.float8_e4m3fn_x4)
    sbuf_tile_space_non_buffered += NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.uint8)

    _is_fp8_input = cfg.input_dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
    _is_dma_xpose_mx = (
        _is_fp8_input or (cfg.input_dtype == nl.bfloat16 and cfg.quantization_config.has_mx_static_dequant_scales)
    ) and cfg.load_input_with_DMA_transpose

    # Per-S-tile space: input_sb + output_sb + norm buffers + rope buffers
    sbuf_tile_space_per_s_tile = _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
        cfg=cfg, dims=dims, sbm=sbm, is_fp8_dma_xpose=_is_dma_xpose_mx
    )

    # MX-specific per-S-tile space (scales with num_s_tiles)
    mx_space_per_s_tile = H_128_tiles * P_MAX * sizeinbytes(nl.float8_e4m3fn_x4)  # hidden_qtz
    if not (_is_fp8_input and _is_dma_xpose_mx):
        mx_space_per_s_tile += (
            H_128_tiles * P_MAX * sizeinbytes(nl.uint8)
        )  # hidden_scale (not needed for FP8 DMA xpose)
    if not _is_dma_xpose_mx:
        mx_space_per_s_tile += H_128_tiles * P_MAX * H_pack * sizeinbytes(cfg.compute_mm_dtype)  # transposed_buffer

    # BF16 DMA transpose path: staging buffer for DMA transpose output before quantize_mx
    _is_bf16_dma_xpose_mx = (
        cfg.input_dtype == nl.bfloat16
        and cfg.quantization_config.has_mx_static_dequant_scales
        and cfg.load_input_with_DMA_transpose
    )
    if _is_bf16_dma_xpose_mx:
        mx_space_per_s_tile += H_128_tiles * P_MAX * 2 * sizeinbytes(nl.float32)  # bf16_xpose_staging_sb

    total_space_per_s_tile = sbuf_tile_space_per_s_tile + mx_space_per_s_tile

    max_s_buffer = (
        cfg.total_available_sbuf_space_to_this_kernel - sbuf_tile_space_non_buffered
    ) // total_space_per_s_tile
    s_multi_buffer_degree = min(s_multi_buffer_degree, max(1, max_s_buffer))

    # PSUM constraint
    MAX_PSUM_TILING_GROUPS = NUM_HW_PSUM_BANKS // dims.num_512_tiles_per_I
    s_multi_buffer_degree = min(s_multi_buffer_degree, MAX_PSUM_TILING_GROUPS)

    projected_space = s_multi_buffer_degree * total_space_per_s_tile + sbuf_tile_space_non_buffered
    return s_multi_buffer_degree, projected_space


def _use_weight_prefetch_mx(
    projected_sbuf_taken_space: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
) -> bool:
    """
    Determine if MX path can afford weight prefetching.

    Compares the space needed for full weight prefetching against available SBUF space
    after accounting for other allocations. Prefetching loads all weights upfront
    instead of chunked loading during computation.

    Args:
        projected_sbuf_taken_space (int): Estimated SBUF space already allocated
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object

    Returns:
        bool: True if weight prefetching is feasible, False otherwise

    Notes:
        - Prefetch space = H_128_tiles * I * (sizeof(float8) + sizeof(uint8))
        - Chunked space = NUM_MX_WEIGHT_BUFFERS * 1 * I * (sizeof(float8) + sizeof(uint8))
    """
    P_MAX = nl.tile_size.pmax
    H_pack = 4
    H_128_tiles = dims.H // (P_MAX * H_pack)
    NUM_MX_WEIGHT_BUFFERS_LOCAL = 2

    # Prefetch space needed
    weights_prefetch_space = H_128_tiles * dims.I * sizeinbytes(nl.float8_e4m3fn_x4)
    scales_prefetch_space = H_128_tiles * dims.I * sizeinbytes(nl.uint8)

    # Chunked space (already accounted in projected_space)
    weights_chunk_space = NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.float8_e4m3fn_x4)
    scales_chunk_space = NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.uint8)

    can_prefetch = (
        projected_sbuf_taken_space
        - weights_chunk_space
        - scales_chunk_space
        + weights_prefetch_space
        + scales_prefetch_space
    ) < cfg.total_available_sbuf_space_to_this_kernel

    return can_prefetch


def _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    is_fp8_dma_xpose: bool = False,
    qkv_in_scale: Optional[nl.ndarray] = None,
) -> int:
    """
    Compute the total SBUF space taken (per partition) by simultaneously live tensors that will be multi-buffered in the kernel.

    WARNING: This is not independently useful function, its correctness is based on the tensor allocation that comes after it.
    This is a 'lookahead' function.
    NOTE: If any additional tensors are added in the kernel, this function needs to be updated.

    Current tensors inside a loop that will get buffered are:
    'input_sb', 'output_sb'                                      (unless FP8 DMA transpose: no input_sb)
    'square_sum_sb',                                             (if cfg.fused_norm_type.RMS_NORM or cfg.fused_norm_type.RMS_NORM_GAMMA)
    'bn_aggr_result_sb'                                          (if cfg.fused_norm_type.LAYER_NORM)
    'cos_buffer_sb', 'sin_buffer_sb', 'rope_intermediate_buffer' (if cfg.fused_rope)
    """

    pre_buffer_tile_space_per_partition = 0

    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
        # 'input_sb  [nl.tile_size.pmax, H]' + 'input_quantized_sb [nl.tile_size.pmax, H]'
        pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.input_dtype)
        pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.quantization_config.quant_dtype)

        # 'in_scale_tile' + 'w_scale_tile'
        pre_buffer_tile_space_per_partition += 3 * sizeinbytes(cfg.quantization_config.qkv_w_scale.dtype) + sizeinbytes(
            qkv_in_scale.dtype
        )
    else:
        # 'input_sb  [nl.tile_size.pmax, H]' — not allocated on FP8 DMA transpose path
        if not is_fp8_dma_xpose:
            pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.compute_mm_dtype)

    # 'output_sb [nl.tile_size.pmax, I]'
    pre_buffer_tile_space_per_partition += dims.I * sizeinbytes(cfg.compute_mm_dtype)

    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        pre_buffer_tile_space_per_partition += 1 * sizeinbytes(cfg.act_dtype)  # 'square_sum_sb [nl.tile_size.pmax, 1]'

    if cfg.fused_norm_type == NormType.LAYER_NORM:
        NUM_AGGR_STATS = 2
        # 'bn_aggr_result_sb [nl.tile_size.pmax, NUM_AGGR_STATS]'
        pre_buffer_tile_space_per_partition += NUM_AGGR_STATS * sizeinbytes(cfg.act_dtype)

    if cfg.fused_rope:
        # 'cos_buffer_sb [nl.tile_size.pmax, d_head]'
        pre_buffer_tile_space_per_partition += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
        # 'sin_buffer_sb [nl.tile_size.pmax, d_head // 2]'
        pre_buffer_tile_space_per_partition += dims.d_head // 2 * sizeinbytes(cfg.compute_mm_dtype)
        # 'rope_intermediate_buffer [nl.tile_size.pmax, d_head * 2]'
        pre_buffer_tile_space_per_partition += dims.d_head * 2 * sizeinbytes(cfg.compute_mm_dtype)

    return pre_buffer_tile_space_per_partition


def _use_weight_prefetch(
    projected_sbuf_taken_space_after_multi_buffer: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> bool:
    """
    Returns True if we can afford weight prefetching, given projected space requirements post multi-buffering.
    """
    # This is how much space we need to prefetch weights, and keep them on SBUF through the entire kernel.
    weights_NEW_space_needed = (dims.I * dims.num_128_tiles_per_H) * sizeinbytes(cfg.compute_mm_dtype)
    # Subtract the weights_OLD_space (non-prefetched), which was taken into account by multi-buffering space calculation.
    weights_OLD_space_taken = (
        dims.NUM_WEIGHT_BUFFERS_DEFAULT
        * (dims.I * math.ceil(dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT / nl.tile_size.pmax))
        * sizeinbytes(cfg.compute_mm_dtype)
    )
    # Note: In auto-allocation mode, sbuf space calculations do not make sense, but they do not break kernel correctness.
    can_weight_prefetch = (
        projected_sbuf_taken_space_after_multi_buffer - weights_OLD_space_taken
    ) + weights_NEW_space_needed < cfg.total_available_sbuf_space_to_this_kernel

    # Note: S >= 1024 should be investigated further. For small S, prefetching causes degradation in some cases.
    weight_prefetch_heuristic = (dims.S_shard >= 1024) or (dims.I >= 1024)
    use_weight_prefetch = can_weight_prefetch and weight_prefetch_heuristic
    return use_weight_prefetch


def _apply_rms_normalization(
    input_row_sb: nl.ndarray,
    square_sum_row_sb: nl.ndarray,
    zero_bias: nl.ndarray,
    norm_eps: nl.ndarray,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> None:
    """
    Apply RMS normalization to input tile in-place (no gamma weights applied yet).
    Multiply input_row_sb by 1 / RMS(x) =  1 / sqrt(eps + (1 / hidden_actual) * (x1^2 + x2^2 + ... + xn^2)).
    Src tensors: input_row_sb (+ norm_eps) [s_tile_sz, dims.H]
                 square_sum_row_sb         [s_tile_sz, dims.H] , temporary buffer pre-allocated.
    Dst tensors: input_row_sb (+ norm_eps) [s_tile_sz, dims.H] (multiplied by 1/RMS(x) ).
    """

    # NOTE: "act" tensor is not used, we only use reduce_res=square_sum from activation_reduce output, but NKI ISA requires dst tensor.
    # nisa.activation_reduce(...) requires src and dst APs shapes to be of equal size, but (input_sb[i_tile_S] has shape [128, H]).
    # To get around this, we use 0-Step AccessPattern "dst=act.ap(pattern=[[1, num_s],[0,H]])".
    # Keeping "act" shape as [pmax, H] would waste valuable SBUF space and potentally limit multi-buffering and weight-prefetching.

    # Temporary tensor allocation
    act_reduce_sbm = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.act_dtype, buffer=nl.sbuf)

    # Sum of squares: x1^2 + x2^2 + ... + xn^2 ( sum of squares of the input row ).
    nisa.activation_reduce(
        dst=act_reduce_sbm.ap(pattern=[[1, s_tile_sz], [0, dims.H]]),
        op=nl.square,
        data=input_row_sb[0:s_tile_sz, 0 : dims.H],
        reduce_op=nl.add,
        reduce_res=square_sum_row_sb[0:s_tile_sz, 0:1],
        bias=zero_bias[0:s_tile_sz, 0:1],
    )

    # Reciprocal square root: 1 / RMS(x) =  1 / sqrt(eps + (1 / hidden_actual) * (x1^2 + x2^2 + ... + xn^2)).
    nisa.activation(
        dst=square_sum_row_sb[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
        data=square_sum_row_sb[0:s_tile_sz, 0:1],
        bias=norm_eps[0:s_tile_sz, 0:1],
        scale=float(1.0 / dims.H_actual),
    )

    # Apply normalization: Multiply input_sb by 1 / RMS(x).
    nisa.tensor_scalar(
        dst=input_row_sb[0:s_tile_sz, :],
        data=input_row_sb[0:s_tile_sz, :],
        op0=nl.multiply,
        operand0=square_sum_row_sb[0:s_tile_sz, 0:1],
        engine=nisa.vector_engine,
    )


def _compute_layer_norm_stats(
    input_row_sb: nl.ndarray,
    bn_aggr_result_tile: nl.ndarray,
    norm_eps: nl.ndarray,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> None:
    """
    Compute LayerNorm statistics for row of input_sb and store it bn_aggr_result_tile.
        mean = bn_aggr_result_tile[0:s_tile_sz, 0:1]
        rvar = bn_aggr_result_tile[0:s_tile_sz, 1:2] #rvar(var + eps)

    Src tensor: input_row_sb (+ norm_eps) [s_tile_sz, dims.H]
    Dst tensor: bn_aggr_result_tile       [s_tile_sz, 2]
    """
    # LayerNorm constants
    BN_STATS_TILE_SIZE = 512
    BN_STATS_DST_SIZE = 6
    NUM_512_BN_STATS_TILES_H = math.ceil(dims.H / BN_STATS_TILE_SIZE)

    # Allocate temporary tensor for bn_stats results
    # nisa.bn_stats(...) outputs 6 different metrics per tile (to later be aggregated to mean/varaince by nisa.bn_aggr(...)).
    bn_stats_result_sb = sbm.alloc_stack(
        (nl.tile_size.pmax, BN_STATS_DST_SIZE * NUM_512_BN_STATS_TILES_H),
        dtype=cfg.act_dtype,
        buffer=nl.sbuf,
    )

    # Compute bn_stats for each 512-sized tile along H dimension
    # Note: All nisa.bn_stats(..) computation is done on float32.
    for i_bn_tile in nl.affine_range(NUM_512_BN_STATS_TILES_H):
        # Calculate valid H elements for this tile
        bn_tile_offset = i_bn_tile * BN_STATS_TILE_SIZE
        bn_tile_sz = min(BN_STATS_TILE_SIZE, dims.H - bn_tile_offset)

        nisa.bn_stats(
            dst=bn_stats_result_sb[0:s_tile_sz, nl.ds(i_bn_tile * BN_STATS_DST_SIZE, BN_STATS_DST_SIZE)],
            data=input_row_sb[0:s_tile_sz, nl.ds(bn_tile_offset, bn_tile_sz)],
        )

    # Aggregate 6 bn_stats metrics into mean and variance
    NUM_AGGR_STATS = 2
    nisa.bn_aggr(
        dst=bn_aggr_result_tile[0:s_tile_sz, 0:NUM_AGGR_STATS],
        data=bn_stats_result_sb[0:s_tile_sz, 0 : BN_STATS_DST_SIZE * NUM_512_BN_STATS_TILES_H],
    )

    # Compute reciprocal square root of variance for normalization
    nisa.activation(
        dst=bn_aggr_result_tile[0:s_tile_sz, 1:NUM_AGGR_STATS],
        data=bn_aggr_result_tile[0:s_tile_sz, 1:NUM_AGGR_STATS],
        bias=norm_eps[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
    )


def _copy_psum_to_sbuf_apply_rope_and_bias(
    qkv_MM_output_psum: List[nl.ndarray],
    output_sb: List[nl.ndarray],
    cos_buffer_sb: List[nl.ndarray],
    sin_buffer_sb: List[nl.ndarray],
    rope_intermediate_buffer_sb: List[nl.ndarray],
    cos_cache_hbm: nl.ndarray,
    sin_cache_hbm: nl.ndarray,
    i_tile_S: int,
    s_tile_sz: int,
    i_batch: int,
    s_tile_local_offset: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    bias_sb: Optional[nl.ndarray],
    # Quantization Related
    w_scale_tile: Optional[nl.ndarray],
) -> None:
    """
    Apply RoPE rotation to Q/K heads and copy V heads from PSUM matmul results to output buffer.
    Performs the copy only for "i_tile_S" contracted row.

    w_scale_tile: shape of [128, 3] contains the dequant scale for q, k, v

    Src: * qkv_MM_output_psum (QKV Projection Results)
         * Pre-allocated RoPE buffers: cos_buffer_sb, sin_buffer_sb, rope_intermediate_buffer_sb.
            and corresponding HBM tensors: cos_buffer_hbm, sin_buffer_hbm

    Dst: Store results to output_matmult_sb[i_tile_S]

    - Each element is a PSUM bank [128, 512] storing results for specific (S_tile, I_tile)
    - Bank indexing: i_tile_S * dims.num_512_tiles_per_I + k_tile_I
    - Contains Q, K, V head data across different banks based on head_offset
    """

    d_head = dims.d_head
    d_head_half = d_head // 2

    NUM_HEADS_PER_PSUM_BANK = 512 // d_head

    # Load RoPE tensors if RoPE fusion is enabled.
    cos_src_offset = i_batch * dims.S * d_head + (dims.S_shard_offset + s_tile_local_offset) * d_head
    nisa.dma_copy(
        dst=cos_buffer_sb[i_tile_S].ap(pattern=[[d_head, s_tile_sz], [1, d_head]], offset=0),
        src=cos_cache_hbm.ap(pattern=[[d_head, s_tile_sz], [1, d_head]], offset=cos_src_offset),
        dge_mode=dge_mode.swdge,
    )

    sin_src_offset = i_batch * dims.S * d_head + (dims.S_shard_offset + s_tile_local_offset) * d_head
    nisa.dma_copy(
        dst=sin_buffer_sb[i_tile_S].ap(pattern=[[d_head_half, s_tile_sz], [1, d_head_half]], offset=0),
        src=sin_cache_hbm.ap(pattern=[[d_head, s_tile_sz], [1, d_head_half]], offset=sin_src_offset),
        dge_mode=dge_mode.swdge,
    )

    # For each head, RoPE([X1, X2]) = [X1, X2] * cos + [-X2 * sin, X1 * sin]
    for i_head in nl.sequential_range(dims.num_q_heads + dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        num_d_half = num_d // 2

        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + i_head // NUM_HEADS_PER_PSUM_BANK
        psum_head_offset = (i_head % NUM_HEADS_PER_PSUM_BANK) * d_head

        if w_scale_tile is not None:
            if i_head < dims.num_q_heads:
                current_head_w_scale_tile = w_scale_tile[0:s_tile_sz, 0]
            else:
                current_head_w_scale_tile = w_scale_tile[0:s_tile_sz, 1]

        # Copy the current head from psum to sbuf first. we maintain two copy of the head, the first copy is for cos * X and the second for sin * rotate_half(X)
        if cfg.add_bias:
            if w_scale_tile is not None:
                nisa.scalar_tensor_tensor(
                    dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                    op0=nl.multiply,
                    operand0=current_head_w_scale_tile,
                    op1=nl.add,
                    operand1=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                )
            else:
                nisa.tensor_tensor(
                    dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
                    data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                    data2=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                    op=nl.add,
                )
        else:
            # Copy the current head from psum to sbuf first. we maintain two copy of the head, the first copy is for cos * X and the second for sin * rotate_half(X)
            if w_scale_tile is not None:
                nisa.tensor_scalar(
                    dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                    op0=nl.multiply,
                    operand0=current_head_w_scale_tile,
                )
            else:
                nisa.tensor_copy(
                    dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
                    src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                )

            # -X2 * sin
        nisa.tensor_tensor(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d_half)],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head_half, num_d_half)],
            data2=sin_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d_half],
            op=nl.multiply,
        )

        nisa.tensor_scalar(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d_half)],
            data=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d_half)],
            op0=nl.multiply,
            operand0=-1,
        )

        # X1 * sin
        nisa.tensor_tensor(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head + d_head_half, num_d_half)],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d_half],
            data2=sin_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d_half],
            op=nl.multiply,
        )

        # X * cos
        nisa.tensor_tensor(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            data2=cos_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            op=nl.multiply,
        )

        # Copy X * cos + [-X2 * sin, X1 * sin] to output sbuf
        nisa.tensor_tensor(
            dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            data2=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d)],
            op=nl.add,
        )

    # Copy V
    for i_head in range(dims.num_q_heads + dims.num_kv_heads, dims.num_q_heads + 2 * dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + i_head // NUM_HEADS_PER_PSUM_BANK
        psum_head_offset = (i_head % NUM_HEADS_PER_PSUM_BANK) * d_head

        if w_scale_tile is not None:
            current_head_w_scale_tile = w_scale_tile[0:s_tile_sz, 2]

        if cfg.add_bias:
            if w_scale_tile is not None:
                nisa.scalar_tensor_tensor(
                    dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                    op0=nl.multiply,
                    operand0=current_head_w_scale_tile,
                    op1=nl.add,
                    operand1=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                )
            else:
                nisa.tensor_tensor(
                    dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                    data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                    data2=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                    op=nl.add,
                )
        else:
            if w_scale_tile is not None:
                nisa.tensor_scalar(
                    dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                    op0=nl.multiply,
                    operand0=current_head_w_scale_tile,
                )
            else:
                nisa.tensor_copy(
                    dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                    src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                )
