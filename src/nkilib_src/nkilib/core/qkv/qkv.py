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
from ..utils.common_types import NormType, QKVOutputLayout, QuantizationType
from ..utils.kernel_assert import kernel_assert

# QKV
from .qkv_cte import qkv_cte
from .qkv_tkg import qkv_tkg


@nki.jit(
    mode="auto",
    debug_kernel=True,
    show_compiler_tb=True,
    experimental_flags="skip-non-top-level-shared-hbm-check",
)
def qkv(
    input: nl.ndarray,
    fused_qkv_weights: nl.ndarray,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.ndarray] = None,
    # -- Quantization
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[nl.ndarray] = None,
    qkv_in_scale: Optional[nl.ndarray] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attention_prev: Optional[nl.ndarray] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.ndarray] = None,
    layer_norm_bias: Optional[nl.ndarray] = None,
    norm_eps: float = 1e-6,
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
    # ----------------------------------------
    is_input_swizzled: bool = False,
) -> nl.ndarray:
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
    input : nl.ndarray
        Input hidden states tensor of shape [B, S, H] where B=batch, S=sequence_length, H=hidden_dim.
        We name it 'input' and not 'hidden' to avoid ambiguity with the size of "hidden dimension".
    fused_qkv_weights : nl.ndarray
        Fused QKV weight matrix of shape [H, I] or [H//4, I] for MX where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head
    output_layout : QKVOutputLayout, default=QKVOutputLayout.BSD
        Output tensor layout: QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]
    bias : Optional[nl.ndarray], default=None
        Bias tensor of shape [1, I] to add to QKV projection output
    quantization_type (QuantizationType):
        Type of quantization to apply (NONE, ROW, STATIC, MX). Default: QuantizationType.NONE.
        Note: For MX quantization, is only supported in QKV CTE kernel.
    qkv_w_scale (nl.ndarray, optional):
        QKV weight scale tensor in HBM for QKV projection.
        Shape:    [1, I] or [128, I] if row quantization, [1, 3] or [128, 3] if static quantization, [H//32, I] if MX quantization
    qkv_in_scale (nl.ndarray, optional):
        QKV input scale tensor in HBM for QKV projection. Only required for static quantization.
        Shape:    [1, 1] or [128, 1]
    fused_residual_add : Optional[bool], default=False
        Whether to perform residual addition: input = input + mlp_prev + attention_prev
    mlp_prev : Optional[nl.ndarray], default=None
        Previous MLP output tensor of shape [B, S, H] for residual addition
    attention_prev : Optional[nl.ndarray], default=None
        Previous attention output tensor of shape [B, S, H] for residual addition
    fused_norm_type : NormType, default=NormType.NO_NORM
        Type of normalization: NormType.NO_NORM, NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA, or NormType.LAYER_NORM
        NormType.RMS_NORM_SKIP_GAMMA assumes fused_qkv_weights have been pre-multiplied with gamma vector, so its skipped here.
    gamma_norm_weights : Optional[nl.ndarray], default=None
        Normalization gamma/scale weights of shape [1, H] (required for NormType.RMS_NORM and NormType.LAYER_NORM)
    layer_norm_bias : Optional[nl.ndarray], default=None
        Layer normalization beta/bias weights of shape [1, H] (only for NormType.LAYER_NORM)
        Using layer norm bias is optional.
    norm_eps : float, default=1e-6
        Epsilon value for numerical stability in normalization
    hidden_actual : Optional[int], default=None
        Actual hidden dimension for padded tensors (if H contains padding)
    fused_rope : Optional[bool], default=False
        Whether to apply RoPE rotation to Query and Key heads after QKV projection
    cos_cache : Optional[nl.ndarray], default=None
        Cosine cache for RoPE of shape [B, S, d_head] (required if fused_rope=True)
    sin_cache : Optional[nl.ndarray], default=None
        Sine cache for RoPE of shape [B, S, d_head] (required if fused_rope=True)
    d_head : Optional[int], default=None
        Dimension per attention head (required for QKVOutputLayout.NBSd and RoPE)
    num_q_heads : Optional[int], default=None
        Number of query heads (required for RoPE)
    num_kv_heads : Optional[int], default=None
        Number of key/value heads (required for RoPE)
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
    is_input_swizzled: bool, default=False
        Whether the input tensor is swizzled (only applicable with MX Quantization).
    Returns:
    --------
    nl.ndarray
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

    # QKV_CTE is used for S bigger than this number.
    # Note: This should be updated, and be based on BxS.
    SEQLEN_THRESHOLD_FOR_QKV_CTE = 96
    # Some features like fused_rope only
    is_input_config_only_available_in_qkv_cte_kernel = False

    # Extend this variable if future configs become available only in one of the sub-kernels.
    is_input_config_only_available_in_qkv_cte_kernel = fused_rope

    input_in_sbuf = input.buffer == nl.sbuf
    if not input_in_sbuf:
        _, S, _ = input.shape
        is_input_config_only_available_in_qkv_cte_kernel = (
            is_input_config_only_available_in_qkv_cte_kernel or S > SEQLEN_THRESHOLD_FOR_QKV_CTE
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
            block_size=block_size,
            slot_mapping=slot_mapping,
            store_output_in_sbuf=store_output_in_sbuf,
            sbm=sbm,
            use_auto_allocation=use_auto_allocation,
            load_input_with_DMA_transpose=load_input_with_DMA_transpose,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            is_input_swizzled=is_input_swizzled,
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
        )

    return output
