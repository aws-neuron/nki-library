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
QKV kernel.

"""

# Standard Library
from typing import Optional

# Neuron Kernel Interface
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

# QKV
from .qkv_cte import qkv_cte
from .qkv_tkg import qkv_tkg

# Sequence length threshold for CTE dispatch.
# Note: This should be updated, and be based on BxS.
SEQLEN_THRESHOLD_FOR_QKV_CTE = 96


def _attach_qk_norm_weights(cfg, q_gamma, k_gamma, q_beta, k_beta):
    """Attach tensor weights to a QKNormConfig, if the config is not None."""
    if cfg is None:
        return
    cfg.q_gamma_norm_weights = q_gamma
    cfg.k_gamma_norm_weights = k_gamma
    cfg.q_beta_norm_weights = q_beta
    cfg.k_beta_norm_weights = k_beta


@nki.jit
def qkv(
    input: nl.NkiTensor,
    fused_qkv_weights: nl.NkiTensor,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.NkiTensor] = None,
    # -- Quantization
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[nl.NkiTensor] = None,
    qkv_in_scale: Optional[nl.NkiTensor] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[nl.NkiTensor] = None,
    attention_prev: Optional[nl.NkiTensor] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.NkiTensor] = None,
    layer_norm_bias: Optional[nl.NkiTensor] = None,
    norm_eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[nl.NkiTensor] = None,
    sin_cache: Optional[nl.NkiTensor] = None,
    # Fused RoPE + QK-norm: K-specific gamma-fused caches
    k_cos_cache: Optional[nl.NkiTensor] = None,
    k_sin_cache: Optional[nl.NkiTensor] = None,
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
    transpose_k_cache: bool = False,
    fp8_packed: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[nl.NkiTensor] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
    # ----------------------------------------
    is_h_dim_4h_transposed: bool = False,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    # --- Transposed Input
    transposed_in: bool = False,
    # --- QK-Norm Related
    qk_norm_pre_rope: Optional[QKNormConfig] = None,
    qk_norm_post_rope: Optional[QKNormConfig] = None,
    qk_norm_pre_rope_q_gamma: Optional[nl.NkiTensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[nl.NkiTensor] = None,
    qk_norm_post_rope_q_gamma: Optional[nl.NkiTensor] = None,
    qk_norm_post_rope_k_gamma: Optional[nl.NkiTensor] = None,
    qk_norm_pre_rope_q_beta: Optional[nl.NkiTensor] = None,
    qk_norm_pre_rope_k_beta: Optional[nl.NkiTensor] = None,
    qk_norm_post_rope_q_beta: Optional[nl.NkiTensor] = None,
    qk_norm_post_rope_k_beta: Optional[nl.NkiTensor] = None,
    # --- Strided Input
    strided_input_config: Optional[StridedInputConfig] = None,
    # --- Output
    output_hbm: Optional[nl.NkiTensor] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    # --- Optional per-segment sum-of-squares outputs (each [B, S, 1]); qkv_cte only
    q_squared_sum_out: Optional[nl.NkiTensor] = None,
    k_squared_sum_out: Optional[nl.NkiTensor] = None,
    v_squared_sum_out: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    QKV (Query, Key, Value) projection kernel with multiple (optional) fused operations.

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


    Parameters:
    -----------
    input : nl.NkiTensor
        Input hidden states tensor of shape [B, S, H] where B=batch, S=sequence_length, H=hidden_dim.
        We name it 'input' and not 'hidden' to avoid ambiguity with the size of "hidden dimension".
    fused_qkv_weights : nl.NkiTensor
        Fused QKV weight matrix of shape [H, I] or [H//4, I, 4] for MX where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head
    output_layout : QKVOutputLayout, default=QKVOutputLayout.BSD
        Output tensor layout: QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]
    bias : Optional[nl.NkiTensor], default=None
        Bias tensor of shape [1, I] to add to QKV projection output
    quantization_type (QuantizationType):
        Type of quantization to apply (NONE, ROW, STATIC, MX). Default: QuantizationType.NONE.
        Note: For MX quantization, is only supported in QKV CTE kernel.
    qkv_w_scale (nl.NkiTensor, optional):
        QKV weight scale tensor in HBM for QKV projection.
        Shape:    [1, I] or [128, I] if row quantization, [1, 3] or [128, 3] if static quantization, [H//32, I] if MX quantization
    qkv_in_scale (nl.NkiTensor, optional):
        QKV input scale tensor in HBM for QKV projection. Only required for static quantization.
        Shape:    [1, 1] or [128, 1]
    fused_residual_add : Optional[bool], default=False
        Whether to perform residual addition: input = input + mlp_prev + attention_prev
    mlp_prev : Optional[nl.NkiTensor], default=None
        Previous MLP output tensor of shape [B, S, H] for residual addition
    attention_prev : Optional[nl.NkiTensor], default=None
        Previous attention output tensor of shape [B, S, H] for residual addition
    fused_norm_type : NormType, default=NormType.NO_NORM
        Type of normalization: NormType.NO_NORM, NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA, or NormType.LAYER_NORM
        NormType.RMS_NORM_SKIP_GAMMA assumes fused_qkv_weights have been pre-multiplied with gamma vector, so its skipped here.
    gamma_norm_weights : Optional[nl.NkiTensor], default=None
        Normalization gamma/scale weights of shape [1, H] (required for NormType.RMS_NORM and NormType.LAYER_NORM)
    layer_norm_bias : Optional[nl.NkiTensor], default=None
        Layer normalization beta/bias weights of shape [1, H] (only for NormType.LAYER_NORM)
        Using layer norm bias is optional.
    norm_eps : float, default=1e-6
        Epsilon value for numerical stability in normalization
    hidden_actual : Optional[int], default=None
        Actual hidden dimension for padded tensors (if H contains padding)
    fused_rope : Optional[bool], default=False
        Whether to apply RoPE rotation to Query and Key heads after QKV projection
    cos_cache : Optional[nl.NkiTensor], default=None
        Cosine cache for RoPE of shape [B, S, d_head] (required if fused_rope=True)
    sin_cache : Optional[nl.NkiTensor], default=None
        Sine cache for RoPE of shape [B, S, d_head] (required if fused_rope=True)
    d_head : Optional[int], default=None
        Dimension per attention head (required for QKVOutputLayout.NBSd and RoPE)
    num_q_heads : Optional[int], default=None
        Number of query heads (required for RoPE)
    num_kv_heads : Optional[int], default=None
        Number of key/value heads (required for RoPE)
    transpose_k_cache (bool), Default: False
        Whether to store K in transposed layout [num_blocks*num_kv_heads, d_head, block_size]
        in the block KV cache or [B, kv_dim, max_seq_len] for flat KV cache.
    fp8_packed (bool), Default: False
        Enable packed FP8 K cache layout for block KV. Packs 2 consecutive FP8 sequence
        positions into one row: k_cache shape [num_blocks, num_kv_heads, block_size // 2, d_head, 2] fp8,
        where dim 4 index 0 = even positions and index 1 = odd positions. Enables DMA
        transpose on decode. Requires block KV, FP8 quantization, even block_size, and
        d_head <= 128. Mutually exclusive with transpose_k_cache.
    store_output_in_sbuf : bool, default=False
        Whether to store output in SBUF (currently unsupported, must be False)
    sbm : Optional[SbufManager], default=None
        Optional SBUF manager for memory allocation control, with pre-specified bounds for SBUF usage.
        If sbm is not provided, kernel will by default be allocated and use all of the available SBUF space.
    use_auto_allocation : bool, default=False
        Whether to use automatic SBUF allocation, by default kernel is manually allocated and it creates its own SbufManager.
        If 'sbm' is provided by user, user has the responsibility to set use_auto_allocation=True in the provided SbufManager.
    load_input_with_DMA_transpose : bool, default=True
        Whether to use DMA transpose optimization.
    is_h_dim_4h_transposed: bool, default=False
            Whether the H-dim (in input and gamma) has been pre-transposed by 4 (only applicable with MX Quantization).
            If is_h_dim_4h_transposed = False,
                * input has typical shape [B, S, H], viewed as [B, S, H//512, 128_H, 4_H].
            If is_h_dim_4h_transposed = True,
                * input has shape [B, S, H] but is pre-shuffled from
                  [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] and flattened to [B, S, H].
                * IMPORTANT: H-dim in both input and gamma weights (for RMSNorm) must be pre-shuffled.
                    * For input, this is achieved by offline pre-shuffling weights of upstream projection (in real model).
                    * For gamma, this is achieved by offline pre-shuffling of gamma tensor.
                Purpose: More efficent for obtaining the required swizzled layout for quantize_mx instruction.
    weight_layout : QKVWeightLayout, default=QKVWeightLayout.CONTIGUOUS
        Layout of fused_qkv_weights. See QKVWeightLayout docstring for packing instructions.
    transposed_in : bool, default=False
        When True, input is in transposed HBM layout [H0, n_prgs, H1_shard, BxS] instead of [B, S, H].
        Used for zero-conversion inter-layer data flow in the transformer megakernel.
        Only supported with qkv_tkg (small batch, B*S <= pmax). Requires LNC >= 2.
    dtype_mode : DtypeMode, default=DtypeMode.NON_OCP
        Quantization dtype policy for STATIC/ROW weight tiles. Also used for
        the FP8 KV cache when ``kv_dtype`` is the opaque ``"float8e4"``
        sentinel; a concrete ``kv_dtype`` (e.g. ``nl.float8_e4m3fn``) is
        honored as-is.
        - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
        - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
        - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, else ``nl.float8_e4m3``.
    Returns:
    --------
    nl.NkiTensor
        QKV projection output tensor:
        - If output_layout=QKVOutputLayout.BSD: shape [B, S, I]
        - If output_layout=QKVOutputLayout.NBSd: shape [num_heads, B, S, d_head]

    Constraints:
    ------------
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
    - MXFP8, bf16, fp16, fp32 (fp32 inputs are internally converted to bf16 for computation)
    """
    # Note: Detailed asserts done inside qkv_cte and qkv_tkg kernels.
    # Some features like fused_rope only
    is_input_config_only_available_in_qkv_cte_kernel = False

    # In-kernel KV cache update is only supported by qkv_cte (qkv_tkg has no cache write path).
    in_kernel_cache_update = k_cache is not None or v_cache is not None

    # Extend this variable if future configs become available only in one of the sub-kernels.
    is_input_config_only_available_in_qkv_cte_kernel = (
        fused_rope or (strided_input_config != None) or in_kernel_cache_update
    )

    input_in_sbuf = input.buffer == nl.sbuf
    if transposed_in:
        # Transposed HBM input: [H0, n_prgs, H1_shard, BxS]
        BxS = input.shape[3]
    elif not input_in_sbuf:
        B, S, _ = input.shape
        is_input_config_only_available_in_qkv_cte_kernel = (
            is_input_config_only_available_in_qkv_cte_kernel
            or S > SEQLEN_THRESHOLD_FOR_QKV_CTE
            or B * S > nl.tile_size.pmax  # Large batch: qkv_tkg requires B*S <= pmax for SBUF output
        )
    else:
        _, BxS, _ = input.shape
        # Added a guard here to prevent selection of QKV TKG when BxS is too large. This needs to be updated
        # once we have a threshold based on BxS
        kernel_assert(
            not is_input_config_only_available_in_qkv_cte_kernel and BxS <= SEQLEN_THRESHOLD_FOR_QKV_CTE,
            "input in sbuf for QKV CTE is not supported yet",
        )

    # TODO: Once qkv_tkg is merged, uncomment if statement.
    _attach_qk_norm_weights(
        qk_norm_pre_rope,
        qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma,
        qk_norm_pre_rope_q_beta,
        qk_norm_pre_rope_k_beta,
    )
    _attach_qk_norm_weights(
        qk_norm_post_rope,
        qk_norm_post_rope_q_gamma,
        qk_norm_post_rope_k_gamma,
        qk_norm_post_rope_q_beta,
        qk_norm_post_rope_k_beta,
    )

    if is_input_config_only_available_in_qkv_cte_kernel:
        output = qkv_cte(
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
            k_cos_cache=k_cos_cache,
            k_sin_cache=k_sin_cache,
            d_head=d_head,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            kv_dtype=kv_dtype,
            use_block_kv=use_block_kv,
            transpose_k_cache=transpose_k_cache,
            fp8_packed=fp8_packed,
            block_size=block_size,
            slot_mapping=slot_mapping,
            store_output_in_sbuf=store_output_in_sbuf,
            sbm=sbm,
            use_auto_allocation=use_auto_allocation,
            load_input_with_DMA_transpose=load_input_with_DMA_transpose,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            is_input_swizzled=is_h_dim_4h_transposed,
            weight_layout=weight_layout,
            qk_norm_pre_rope=qk_norm_pre_rope,
            qk_norm_post_rope=qk_norm_post_rope,
            strided_input_config=strided_input_config,
            output_hbm=output_hbm,
            dtype_mode=dtype_mode,
            q_squared_sum_out=q_squared_sum_out,
            k_squared_sum_out=k_squared_sum_out,
            v_squared_sum_out=v_squared_sum_out,
        )

    else:
        output = qkv_tkg(
            hidden=input,
            qkv_w=fused_qkv_weights,
            qkv_bias=bias,
            fused_add=fused_residual_add,
            mlp_prev=mlp_prev,
            attn_prev=attention_prev,
            norm_type=fused_norm_type,
            norm_w=gamma_norm_weights,
            norm_bias=layer_norm_bias,
            quantization_type=quantization_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            eps=norm_eps,
            hidden_actual=hidden_actual,
            output_layout=output_layout,
            output_in_sbuf=store_output_in_sbuf,
            d_head=d_head,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            sbm=sbm,
            transposed_in=transposed_in,
        )

    return output
