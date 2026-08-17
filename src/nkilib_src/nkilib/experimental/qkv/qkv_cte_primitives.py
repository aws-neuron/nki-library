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

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

# QKV CTE
from ...core.qkv.qkv_cte_utils import (
    QKV_CTE_Config,
    QKV_CTE_Dims,
    QKV_CTE_UserInput,
    _build_config,
    _get_tensor_dimensions,
    _validate_user_inputs,
)
from ...core.utils.allocator import SbufManager, sizeinbytes

# NKI Library
from ...core.utils.common_types import NormType, QKVOutputLayout, QKVWeightLayout, QuantizationType
from ...core.utils.logging import get_logger

# Primitives
from ...experimental.primitives import ColMajor, RowMajor, blas, dma, tile_stream

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


def qkv_cte(
    input: nl.NkiTensor,
    fused_qkv_weights: nl.NkiTensor,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.NkiTensor] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[nl.NkiTensor] = None,
    attention_prev: Optional[nl.NkiTensor] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.NkiTensor] = None,
    layer_norm_bias: Optional[nl.NkiTensor] = None,
    norm_eps: Optional[float] = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[nl.NkiTensor] = None,
    sin_cache: Optional[nl.NkiTensor] = None,
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # --- FP8 KV Cache Quantization Related
    k_cache: Optional[nl.NkiTensor] = None,
    v_cache: Optional[nl.NkiTensor] = None,
    k_scale: Optional[nl.NkiTensor] = None,
    v_scale: Optional[nl.NkiTensor] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[type] = None,
    # --- Block KV Cache Related
    use_block_kv: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[nl.NkiTensor] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # --- Quantization Related
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[nl.NkiTensor] = None,
    qkv_in_scale: Optional[nl.NkiTensor] = None,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
    # ----------------------------------------
    is_input_swizzled: bool = False,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
) -> nl.NkiTensor:
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
        input (nl.NkiTensor): [B, S, H], Input hidden states tensor where B=batch, S=sequence_length, H=hidden_dim.
            We name it 'input' and not 'hidden' to avoid ambiguity with the size of "hidden dimension".
        fused_qkv_weights (nl.NkiTensor): [H, I],  or [H//4, I] for MX, Fused QKV weight matrix where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head
        output_layout (QKVOutputLayout): Output tensor layout: QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]. Default: QKVOutputLayout.BSD
        bias (Optional[nl.NkiTensor]): [1, I], Bias tensor to add to QKV projection output. Default: None
        fused_residual_add (Optional[bool]): Whether to perform residual addition: input = input + mlp_prev + attention_prev. Default: False
        mlp_prev (Optional[nl.NkiTensor]): [B, S, H], Previous MLP output tensor for residual addition. Default: None
        attention_prev (Optional[nl.NkiTensor]): [B, S, H], Previous attention output tensor for residual addition. Default: None
        fused_norm_type (NormType): Type of normalization: NormType.NO_NORM, NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA, or NormType.LAYER_NORM.
            NormType.RMS_NORM_SKIP_GAMMA assumes fused_qkv_weights have been pre-multiplied with gamma vector, so its skipped here. Default: NormType.NO_NORM
        gamma_norm_weights (Optional[nl.NkiTensor]): [1, H], Normalization gamma/scale weights (required for NormType.RMS_NORM and NormType.LAYER_NORM). Default: None
        layer_norm_bias (Optional[nl.NkiTensor]): [1, H], Layer normalization beta/bias weights (only for NormType.LAYER_NORM). Using layer norm bias is optional. Default: None
        norm_eps (Optional[float]): Epsilon value for numerical stability in normalization. Default: 1e-6
        hidden_actual (Optional[int]): Actual hidden dimension for padded tensors (if H contains padding). Default: None
        fused_rope (Optional[bool]): Whether to apply RoPE rotation to Query and Key heads after QKV projection. Default: False
        cos_cache (Optional[nl.NkiTensor]): [B, S, d_head], Cosine cache for RoPE (required if fused_rope=True). Default: None
        sin_cache (Optional[nl.NkiTensor]): [B, S, d_head], Sine cache for RoPE (required if fused_rope=True). Default: None
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
        qkv_w_scale: Optional[nl.NkiTensor], default=None The weight quantization scale for qkv projection,
            Shape: [H//32, I] for MX
        qkv_in_scale: Optional[nl.NkiTensor], default=None The input quantization scale for qkv projection, currently assume the input quantization scales are the scale for q, k, v projections
        is_input_swizzled: bool, default=False
            Whether the input tensor is swizzled (only applicable with MX Quantization).
            If not swizzled, input has shape [B, S, H].
            If swizzled, input has shape [B, S, H] but is preswizzled from
            [B, S, H//512, 128, 4] -> [B, S, 4, H//512, 128] and flattened to [B, S, H].
        weight_layout (QKVWeightLayout): Layout of fused_qkv_weights. See QKVWeightLayout
            docstring for packing instructions. Default: QKVWeightLayout.CONTIGUOUS

    Returns:
        output (nl.NkiTensor): QKV projection output tensor:
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
    output_dtype = input.dtype

    if cfg.output_layout == QKVOutputLayout.BSD:
        output_hbm = nl.ndarray((dims.B_orig, dims.S_orig, dims.I), dtype=output_dtype, buffer=nl.shared_hbm)
        q_tensor_hbm = None
    else:  # QKVOutputLayout.NBSd
        output_hbm = nl.ndarray(
            (dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head),
            dtype=output_dtype,
            buffer=nl.shared_hbm,
        )
        q_tensor_hbm = None

    # Pass values and directly, and keep 'cfg' and 'dims'
    # object separate for clarity.
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

    return output_hbm


def _qkv_cte_impl(
    input_hbm,
    fused_qkv_weights_hbm,
    output_hbm,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    bias_hbm: Optional[nl.NkiTensor] = None,
    # Fused Residual Add Related
    mlp_prev_hbm: Optional[nl.NkiTensor] = None,
    attention_prev_hbm: Optional[nl.NkiTensor] = None,
    # Fused Normalization Related
    gamma_norm_weights_hbm: Optional[nl.NkiTensor] = None,
    layer_norm_bias_hbm: Optional[nl.NkiTensor] = None,
    norm_eps: Optional[float] = 1e-6,
    # Fused RoPE Related
    cos_cache_hbm: Optional[nl.NkiTensor] = None,
    sin_cache_hbm: Optional[nl.NkiTensor] = None,
    # FP8 KV Cache Quantization Related
    q_tensor_hbm: Optional[nl.NkiTensor] = None,
    k_cache_hbm: Optional[nl.NkiTensor] = None,
    v_cache_hbm: Optional[nl.NkiTensor] = None,
    k_scale_hbm: Optional[nl.NkiTensor] = None,
    v_scale_hbm: Optional[nl.NkiTensor] = None,
    # Block KV Cache Related
    slot_mapping_hbm: Optional[nl.NkiTensor] = None,
    # Quantization Related
    qkv_in_scale: Optional[nl.NkiTensor] = None,
    qkv_w_scale: Optional[nl.NkiTensor] = None,
):
    """
    Core QKV CTE kernel implementation.

    Performs the main computation including optional normalization, QKV projection,
    and optional RoPE application. Handles memory allocation, tiling, and data movement
    between HBM, SBUF, and PSUM.

    Args:
        input_hbm (nl.NkiTensor): [dims.B, dims.S, dims.H], Input tensor on HBM
        fused_qkv_weights_hbm (nl.NkiTensor): [dims.H, dims.I], Weight matrix on HBM
        output_hbm (nl.NkiTensor): Output tensor on HBM with shape determined by cfg.output_layout
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager
        bias_hbm (Optional[nl.NkiTensor]): [1, I], Optional bias tensor on HBM
        mlp_prev_hbm (Optional[nl.NkiTensor]): [dims.B, dims.S, dims.H], Optional MLP residual on HBM
        attention_prev_hbm (Optional[nl.NkiTensor]): [dims.B, dims.S, dims.H], Optional attention residual on HBM
        gamma_norm_weights_hbm (Optional[nl.NkiTensor]): [1, H], Optional normalization weights on HBM
        layer_norm_bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional layer norm bias on HBM
        norm_eps (Optional[float]): Epsilon for normalization stability
        cos_cache_hbm (Optional[nl.NkiTensor]): [B, S, d_head], Optional RoPE cosine cache on HBM
        sin_cache_hbm (Optional[nl.NkiTensor]): [B, S, d_head], Optional RoPE sine cache on HBM

    Returns:
        nl.NkiTensor: Output tensor (same as output_hbm parameter)

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

    zero_bias_sb = tile_stream.alloc_logical(
        (nl.tile_size.pmax, 1), nl.tile_size.pmax, cfg.compute_mm_dtype, "zero_bias_sb", sbm
    )
    nisa.memset(dst=zero_bias_sb, value=0)

    norm_eps_sb = tile_stream.alloc_logical(
        (nl.tile_size.pmax, 1), nl.tile_size.pmax, cfg.compute_mm_dtype, "norm_eps_sb", sbm
    )
    nisa.memset(dst=norm_eps_sb, value=norm_eps)

    if cfg.add_bias:
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    ######################## Choose Multi-Buffering Degree ###################################

    s_multi_buffer_degree, projected_sbuf_taken_space_after_multi_buffer = _multi_buffering_degree_for_seqlen(
        cfg=cfg, dims=dims, sbm=sbm, qkv_in_scale=qkv_in_scale
    )

    S_BLOCK_SIZE = s_multi_buffer_degree * min(dims.S_shard, nl.tile_size.pmax)
    num_blocks_per_S_shard = math.ceil(dims.S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0

    ######################## Weight Prefetching: Enough Space Left ?  #########################

    use_weight_prefetch = _use_weight_prefetch(
        projected_sbuf_taken_space_after_multi_buffer, cfg=cfg, dims=dims, sbm=sbm
    )
    # TODO: remove this
    use_weight_prefetch = True

    if use_weight_prefetch:
        num_weight_buffers = 1
        weight_load_block_size_per_H = H
        num_weight_load_blocks_per_H = 1
        max_num_128_H_subtiles_per_weight_block = math.ceil(weight_load_block_size_per_H / 128)

        weights_sb = []
        weights_prefetched_sb = tile_stream.alloc_logical(
            (weight_load_block_size_per_H, I),
            nl.tile_size.pmax,
            cfg.compute_mm_dtype,
            "weights_prefetched_sb",
            sbm,
        )

        dma.Load(
            tile_stream.tile(weights_prefetched_sb, (nl.tile_size.pmax, I), iter_order=RowMajor()),
            tile_stream.tile_hbm(fused_qkv_weights_hbm, (nl.tile_size.pmax, I), iter_order=RowMajor()),
        ).execute()
        weights_sb.append(weights_prefetched_sb)

    for i_batch in range(dims.B):
        for i_block_S in nl.affine_range(num_blocks_per_S_shard):
            sbm.open_scope()
            # Adjust for the last loop iteration.
            s_block_sz = min(S_BLOCK_SIZE, S_shard - S_BLOCK_SIZE * i_block_S)
            num_S_tiles_in_block = math.ceil(s_block_sz / nl.tile_size.pmax)

            #################### Start of Allocations for Multi-Buffered tensors ##############################
            input_sb = []
            for _ in range(num_S_tiles_in_block):
                align = 32 if cfg.load_input_with_DMA_transpose else 1  # DMA_transpose requires align=32.
                input_dtype = cfg.compute_mm_dtype
                input_sb.append(
                    tile_stream.alloc_logical(
                        (H, s_block_sz),
                        nl.tile_size.pmax,
                        input_dtype,
                        "input_sb",
                        sbm,
                        align=32,
                    )
                )

            output_sb = []
            for _ in range(num_S_tiles_in_block):
                output_dtype = cfg.compute_mm_dtype
                output_sb.append(
                    tile_stream.alloc_logical(
                        (nl.tile_size.pmax, I),
                        nl.tile_size.pmax,
                        output_dtype,
                        "output_sb",
                        sbm,
                    )
                )

            if cfg.fused_rope:
                cos_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    cos_buffer_sb.append(
                        tile_stream.alloc_logical(
                            (nl.tile_size.pmax, dims.d_head),
                            nl.tile_size.pmax,
                            cfg.compute_mm_dtype,
                            "cos_buffer_sb",
                            sbm,
                        )
                    )

                sin_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    sin_buffer_sb.append(
                        tile_stream.alloc_logical(
                            (nl.tile_size.pmax, dims.d_head // 2),
                            nl.tile_size.pmax,
                            cfg.compute_mm_dtype,
                            "sin_buffer_sb",
                            sbm,
                        )
                    )

                rope_intermediate_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    rope_intermediate_buffer_sb.append(
                        tile_stream.alloc_logical(
                            (nl.tile_size.pmax, dims.d_head * 2),
                            nl.tile_size.pmax,
                            cfg.compute_mm_dtype,
                            "rope_intermediate_buffer_sb",
                            sbm,
                        )
                    )

            #########################  End of Allocations for Multi-Buffered tensors ##################################

            #######################################################################################################
            # Step 4: (QKV Projection) Multiply transposed input buffer (potentially with norm pre-applied) with weights.
            #######################################################################################################

            # PSUM accumulation buffer for QKV matmult results.
            # One buffer per S tile, each holding the full I dimension.
            qkv_MM_output_psum = []
            for _ in nl.affine_range(num_S_tiles_in_block):
                qkv_MM_output_psum.append(
                    tile_stream.alloc_logical(
                        (nl.tile_size.pmax, I),
                        nl.tile_size.pmax,
                        nl.float32,
                        "qkv_MM_output_psum",
                        buffer=nl.psum,
                    )
                )

            # If use_weight_prefetch, then NUM_WEIGHT_LOADS_PER_H == 1.
            for i_weight_load in nl.affine_range(num_weight_load_blocks_per_H):
                weight_load_offset = i_weight_load * weight_load_block_size_per_H
                curr_num_128_H_subtiles_per_weight_block = min(
                    max_num_128_H_subtiles_per_weight_block,
                    math.ceil((H - weight_load_offset) / 128),
                )

                #    """
                #    Strided HBM->SBUF weights load example, if loading 1024 x I weights at a time.
                #    Here, weight_load_block_size_per_H  = 1024.
                #
                #    HBM Weights
                #    ------------
                #                        I
                #             -----------------------------
                #        128 |       H_1                  |
                #    1024 128|       H_2                  | H
                #        ...                            ...
                #            |       H_8                  |
                #             -----------------------------
                #                        ....
                #
                #    SBUF Weights
                #    ------------
                #                                8 * I
                #            -------------------------------------------------
                #        128|  H_1   |  H_2 |      ....              |  H_8   |
                #            -------------------------------------------------
                #
                #    Note: Access pattern on HBM side is strided, we are skipping 128 * I elements each time.
                #        Order:
                #            [0, 0:I], [128, 0:I], [256, 0:I], ...   ( 8 rows of I elements)
                #            [1, 0:I], [129, 0:I], [257, 0:I], ...   ( 8 rows of I elements)
                #
                #    On SBUF side,
                #            1st row of H_1, and 1st row H_2 will be both partition=0, etc.
                #    """

                for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                    # Recall we did not use PE Array to transpose input buffer in case of loadWithTranspose.
                    # Use nisa.dma_transpose(...) to load/transpose just enough of input for this round of matmult.
                    # Load/transpose only [128, 1024] elements of input.

                    # NOTE: To drop H divisible by 128 constraint, update AP below with valid "num_h" for the last iteration.
                    dma.Load(
                        dst=tile_stream.tile(
                            input_sb[i_tile_S],
                            (nl.tile_size.pmax, s_tile_sz),
                            iter_order=RowMajor(),
                            logical_p=H,
                        ),
                        src=tile_stream.tile_hbm(
                            input_hbm.select(0, i_batch).slice(
                                0, s_tile_global_offset, s_tile_global_offset + s_tile_sz
                            ),
                            (s_tile_sz, nl.tile_size.pmax),
                            iter_order=RowMajor(),
                        ),
                        transpose=True,
                    ).execute()

                    # QKV matmul: output = input^T @ weights (+ optional bias)
                    # stationary: input_sb[i_tile_S], logical (H, s_tile_sz), tiled (K=128, P=s_tile_sz)
                    # moving: weights_prefetched_sb, logical (H, I), tiled (K=128, F=512)
                    # dst: qkv_MM_output_psum[i_tile_S], logical (s_tile_sz, I), tiled (P=s_tile_sz, F=512)
                    blas.Matmul(
                        dst=tile_stream.tile(
                            qkv_MM_output_psum[i_tile_S],
                            (s_tile_sz, min(nl.tile_size.psum_fmax, I)),
                            iter_order=RowMajor(),
                        ),
                        stationary=tile_stream.tile(
                            input_sb[i_tile_S],
                            (nl.tile_size.pmax, s_tile_sz),
                            iter_order=ColMajor(),
                            logical_p=H,
                        ),
                        moving=tile_stream.tile(
                            weights_prefetched_sb,
                            (nl.tile_size.pmax, min(nl.tile_size.psum_fmax, I)),
                            iter_order=ColMajor(),
                        ),
                        skip_evict=True,
                    ).execute()
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
                    )
                # Copy results PSUM -> SBUF, (optionally) add_bias.
                else:
                    for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                        i_offset = nl.tile_size.psum_fmax * k_tile_I
                        num_i = min(nl.tile_size.psum_fmax, I - i_offset)

                        psum_slice = _sv(qkv_MM_output_psum[i_tile_S], s_tile_sz, i_offset, num_i)
                        out_slice = _sv(output_sb[i_tile_S], s_tile_sz, i_offset, num_i)

                        if cfg.add_bias:
                            bias_slice = _sv(bias_sb, s_tile_sz, i_offset, num_i)
                            nisa.tensor_tensor(
                                dst=out_slice,
                                data1=psum_slice,
                                data2=bias_slice,
                                op=nl.add,
                            )
                        else:
                            nisa.tensor_copy(
                                dst=out_slice,
                                src=psum_slice,
                            )

                # End of i_tile_S loop.

            #######################################################################################################
            # Step 6: Store SBUF results back to HBM, using given output layout.
            #######################################################################################################
            # This parts reads from output_matmult_sbuf and writes to out_tensor.

            if cfg.output_layout == QKVOutputLayout.BSD:
                # output_tensor shape: [B, S, I].
                # output_matmult_sbuf contains [128 (S), I].
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                    s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset

                    dma.Store(
                        dst=tile_stream.tile_hbm(
                            output_hbm.select(0, i_batch).slice(
                                0, s_tile_global_offset, s_tile_global_offset + s_tile_sz
                            ),
                            (s_tile_sz, I),
                            iter_order=RowMajor(),
                        ),
                        src=tile_stream.tile(
                            output_sb[i_tile_S],
                            (s_tile_sz, I),
                            iter_order=RowMajor(),
                            logical_p=s_tile_sz,
                        ),
                    ).execute()

            else:  # NBSd = [heads, B, S, head_dim], I = heads * head_dim
                d_head = cast(int, dims.d_head)  # Safe due to validation
                for i_head in range(dims.num_heads):
                    for i_tile_S in range(num_S_tiles_in_block):
                        s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                        s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                        s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset
                        num_d = min(d_head, I - (i_head * d_head))

                        dma.Store(
                            dst=tile_stream.tile_hbm(
                                output_hbm.select(0, i_head)
                                .select(0, i_batch)
                                .slice(0, s_tile_global_offset, s_tile_global_offset + s_tile_sz)
                                .slice(1, 0, num_d),
                                (s_tile_sz, num_d),
                                iter_order=RowMajor(),
                            ),
                            src=tile_stream.tile(
                                output_sb[i_tile_S].slice(2, i_head * d_head, i_head * d_head + num_d),
                                (s_tile_sz, num_d),
                                iter_order=RowMajor(),
                                logical_p=s_tile_sz,
                            ),
                        ).execute()
            sbm.close_scope()  # Deallocate all multi-buffered tensors.
            # End of i_buffer_s loop
        # End of batch loop
    sbm.close_scope()
    return output_hbm


def _load_and_broadcast_bias(
    bias_hbm: nl.NkiTensor, cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager
) -> nl.NkiTensor:
    """
    Loads bias with shape [1,I] to SBUF and broadcasts it to [nl.tile_size.pmax, I], using stream_shuffle.

    Returns allocated SBUF bias tensor.
    Note: User is responsible for deallocating SBUF tensor.
    """
    # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
    bias_sb = tile_stream.alloc_logical(
        (nl.tile_size.pmax, dims.I), nl.tile_size.pmax, cfg.compute_mm_dtype, "bias_sb", sbm
    )
    dma.Load(
        tile_stream.tile(bias_sb.slice(0, 0, 1), (1, dims.I), iter_order=RowMajor()),
        tile_stream.tile_hbm(bias_hbm, (1, dims.I), iter_order=RowMajor()),
    ).execute()
    blas.Broadcast(
        tile_stream.tile(bias_sb, (nl.tile_size.pmax, dims.I), iter_order=RowMajor()),
        tile_stream.tile(bias_sb, (nl.tile_size.pmax, dims.I), iter_order=RowMajor()),
    ).execute()
    return bias_sb


def _load_norm_weights(
    norm_weights_hbm: nl.NkiTensor,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> nl.NkiTensor:
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


def _multi_buffering_degree_for_seqlen(
    cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager, qkv_in_scale: Optional[nl.NkiTensor] = None
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
    # TODO: revive this back

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

    # act_reduce_sum and bn_stats_result_sb appear in the loop.
    # sbuf_tile_space_non_buffered = nl.tile_size.total_available_sbuf_size - sbm.get_free_space() # space taken so far.
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


def _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    is_fp8_dma_xpose: bool = False,
    qkv_in_scale: Optional[nl.NkiTensor] = None,
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

    # 'input_sb  [nl.tile_size.pmax, H]' — not allocated on FP8 DMA transpose path
    pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.compute_mm_dtype)

    # 'output_sb [nl.tile_size.pmax, I]'
    pre_buffer_tile_space_per_partition += dims.I * sizeinbytes(cfg.compute_mm_dtype)

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


def _sv(tv, s_tile_sz, f_offset, f_size):
    """Slice a nl.NkiTensor to (s_tile_sz, f_size) at free-dim offset, returning nl.NkiTensor for nisa calls.
    Assumes container shape (pdim, 1, F) from alloc_logical with n_p_tiles=1."""
    return tv.slice(0, 0, s_tile_sz).select(1, 0).slice(1, f_offset, f_offset + f_size)


def _copy_psum_to_sbuf_apply_rope_and_bias(
    qkv_MM_output_psum: List[nl.NkiTensor],
    output_sb: List[nl.NkiTensor],
    cos_buffer_sb: List[nl.NkiTensor],
    sin_buffer_sb: List[nl.NkiTensor],
    rope_intermediate_buffer_sb: List[nl.NkiTensor],
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    i_tile_S: int,
    s_tile_sz: int,
    i_batch: int,
    s_tile_local_offset: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    bias_sb: Optional[nl.NkiTensor],
) -> None:
    """
    Apply RoPE rotation to Q/K heads and copy V heads from PSUM matmul results to output buffer.
    Performs the copy only for "i_tile_S" contracted row.

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

    # Get nl.NkiTensor for this S tile's PSUM buffer (pmax, I)
    psum_tv = qkv_MM_output_psum[i_tile_S]
    rope_tv = rope_intermediate_buffer_sb[i_tile_S]
    cos_tv = cos_buffer_sb[i_tile_S]
    sin_tv = sin_buffer_sb[i_tile_S]
    out_tv = output_sb[i_tile_S]

    # Load RoPE tensors if RoPE fusion is enabled.
    s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset
    dma.Load(
        tile_stream.tile(cos_tv, (s_tile_sz, d_head), iter_order=RowMajor(), logical_p=s_tile_sz),
        tile_stream.tile_hbm(
            cos_cache_hbm.select(0, i_batch).slice(0, s_tile_global_offset, s_tile_global_offset + s_tile_sz),
            (s_tile_sz, d_head),
            iter_order=RowMajor(),
        ),
    ).execute()

    dma.Load(
        tile_stream.tile(sin_tv, (s_tile_sz, d_head_half), iter_order=RowMajor(), logical_p=s_tile_sz),
        tile_stream.tile_hbm(
            sin_cache_hbm.select(0, i_batch)
            .slice(0, s_tile_global_offset, s_tile_global_offset + s_tile_sz)
            .slice(1, 0, d_head_half),
            (s_tile_sz, d_head_half),
            iter_order=RowMajor(),
        ),
    ).execute()

    # For each head, RoPE([X1, X2]) = [X1, X2] * cos + [-X2 * sin, X1 * sin]
    for i_head in nl.sequential_range(dims.num_q_heads + dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        num_d_half = num_d // 2

        # Copy the current head from psum to sbuf first. we maintain two copy of the head, the first copy is for cos * X and the second for sin * rotate_half(X)
        if cfg.add_bias:
            nisa.tensor_tensor(
                dst=_sv(rope_tv, s_tile_sz, 0, num_d),
                data1=_sv(psum_tv, s_tile_sz, head_offset, num_d),
                data2=_sv(bias_sb, s_tile_sz, head_offset, num_d),
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=_sv(rope_tv, s_tile_sz, 0, num_d),
                src=_sv(psum_tv, s_tile_sz, head_offset, num_d),
            )

        # -X2 * sin
        nisa.tensor_tensor(
            dst=_sv(rope_tv, s_tile_sz, d_head, num_d_half),
            data1=_sv(rope_tv, s_tile_sz, d_head_half, num_d_half),
            data2=_sv(sin_tv, s_tile_sz, 0, num_d_half),
            op=nl.multiply,
        )

        nisa.tensor_scalar(
            dst=_sv(rope_tv, s_tile_sz, d_head, num_d_half),
            data=_sv(rope_tv, s_tile_sz, d_head, num_d_half),
            op0=nl.multiply,
            operand0=-1,
        )

        # X1 * sin
        nisa.tensor_tensor(
            dst=_sv(rope_tv, s_tile_sz, d_head + d_head_half, num_d_half),
            data1=_sv(rope_tv, s_tile_sz, 0, num_d_half),
            data2=_sv(sin_tv, s_tile_sz, 0, num_d_half),
            op=nl.multiply,
        )

        # X * cos
        nisa.tensor_tensor(
            dst=_sv(rope_tv, s_tile_sz, 0, num_d),
            data1=_sv(rope_tv, s_tile_sz, 0, num_d),
            data2=_sv(cos_tv, s_tile_sz, 0, num_d),
            op=nl.multiply,
        )

        # Copy X * cos + [-X2 * sin, X1 * sin] to output sbuf
        nisa.tensor_tensor(
            dst=_sv(out_tv, s_tile_sz, head_offset, num_d),
            data1=_sv(rope_tv, s_tile_sz, 0, num_d),
            data2=_sv(rope_tv, s_tile_sz, d_head, num_d),
            op=nl.add,
        )

    # Copy V
    for i_head in range(dims.num_q_heads + dims.num_kv_heads, dims.num_q_heads + 2 * dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)

        if cfg.add_bias:
            nisa.tensor_tensor(
                dst=_sv(out_tv, s_tile_sz, head_offset, num_d),
                data1=_sv(psum_tv, s_tile_sz, head_offset, num_d),
                data2=_sv(bias_sb, s_tile_sz, head_offset, num_d),
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=_sv(out_tv, s_tile_sz, head_offset, num_d),
                src=_sv(psum_tv, s_tile_sz, head_offset, num_d),
            )
