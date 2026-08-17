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
PyTorch reference for qkv kernel
"""

from typing import Dict, Optional

import torch

from ..utils.common_types import DtypeMode, NormType, QKVOutputLayout, QKVWeightLayout, QuantizationType
from .qkv import SEQLEN_THRESHOLD_FOR_QKV_CTE, _attach_qk_norm_weights
from .qkv_cte_torch import qkv_cte_torch_ref
from .qkv_tkg_torch import qkv_tkg_torch_ref

# Hardware partition dimension size (nl.tile_size.pmax resolves to -1 on host)
_PMAX = 128


def qkv_torch_ref(
    input: torch.Tensor,
    fused_qkv_weights: torch.Tensor,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[torch.Tensor] = None,
    # -- Quantization
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[torch.Tensor] = None,
    qkv_in_scale: Optional[torch.Tensor] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[torch.Tensor] = None,
    attention_prev: Optional[torch.Tensor] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[torch.Tensor] = None,
    layer_norm_bias: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    k_cos_cache: Optional[torch.Tensor] = None,  # Accepted for interface compatibility; not used in torch ref
    k_sin_cache: Optional[torch.Tensor] = None,  # Accepted for interface compatibility; not used in torch ref
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # --- FP8 KV Cache Quantization Related
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[type] = None,
    # --- Block KV Cache Related
    use_block_kv: bool = False,
    transpose_k_cache: bool = False,
    fp8_packed: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[torch.Tensor] = None,
    store_output_in_sbuf: bool = False,
    sbm=None,
    use_auto_allocation: bool = False,
    load_input_with_DMA_transpose: bool = True,
    is_h_dim_4h_transposed: bool = False,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    # --- QK-Norm Related
    qk_norm_pre_rope=None,
    qk_norm_post_rope=None,
    qk_norm_pre_rope_q_gamma: Optional[torch.Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[torch.Tensor] = None,
    qk_norm_post_rope_q_gamma: Optional[torch.Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[torch.Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[torch.Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[torch.Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[torch.Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[torch.Tensor] = None,
    transposed_in: bool = False,
    # --- Strided Input (accepted for signature parity; handled by test wrapper)
    strided_input_config=None,
    # --- Output
    output_hbm: Optional[torch.Tensor] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    q_squared_sum_out: Optional[torch.Tensor] = None,
    k_squared_sum_out: Optional[torch.Tensor] = None,
    v_squared_sum_out: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Torch reference matching the qkv() kernel entry signature.

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

    Dispatches to the appropriate path-specific torch ref based on input
    configuration, mirroring the qkv() kernel's dispatch logic:
    - CTE path: S > SEQLEN_THRESHOLD or CTE-only features (fused_rope, FP8 KV cache,
      block KV, MX quantization, input swizzling)
    - TKG path: everything else

    Args:
        input (torch.Tensor): Input hidden states [B, S, H].
        fused_qkv_weights (torch.Tensor): Fused QKV weight matrix [H, I] or [H//4, I]
            for MX, where I = (num_q_heads + 2*num_kv_heads) * d_head.
        output_layout (QKVOutputLayout): Output layout: BSD [B,S,I], NBSd [N,B,S,d_head],
            or NBdS [N,B,d_head,S]. Default: BSD.
        bias (Optional[torch.Tensor]): QKV bias [1, I]. Default: None.
        quantization_type (QuantizationType): Quantization mode (NONE, ROW, STATIC, MX).
            Default: NONE.
        qkv_w_scale (Optional[torch.Tensor]): Weight quantization scale.
        qkv_in_scale (Optional[torch.Tensor]): Input quantization scale.
        fused_residual_add (Optional[bool]): Compute input + mlp_prev + attention_prev.
            Default: False.
        mlp_prev (Optional[torch.Tensor]): Previous MLP output [B, S, H].
        attention_prev (Optional[torch.Tensor]): Previous attention output [B, S, H].
        fused_norm_type (NormType): Normalization type (NO_NORM, RMS_NORM,
            RMS_NORM_SKIP_GAMMA, LAYER_NORM). Default: NO_NORM.
        gamma_norm_weights (Optional[torch.Tensor]): Norm gamma/scale [1, H].
        layer_norm_bias (Optional[torch.Tensor]): LayerNorm beta [1, H].
        norm_eps (float): Normalization epsilon. Default: 1e-6.
        hidden_actual (Optional[int]): Actual hidden dim for padded tensors.
        fused_rope (Optional[bool]): Apply RoPE to Q and K heads. Default: False.
        cos_cache (Optional[torch.Tensor]): RoPE cosine cache [B, S, d_head].
        sin_cache (Optional[torch.Tensor]): RoPE sine cache [B, S, d_head].
        d_head (Optional[int]): Head dimension.
        num_q_heads (Optional[int]): Number of query heads.
        num_kv_heads (Optional[int]): Number of key/value heads.
        k_cache (Optional[torch.Tensor]): FP8 K cache tensor (mutable).
        v_cache (Optional[torch.Tensor]): FP8 V cache tensor (mutable).
        k_scale (Optional[torch.Tensor]): K quantization scale [128, 1].
        v_scale (Optional[torch.Tensor]): V quantization scale [128, 1].
        fp8_max (Optional[float]): FP8 clamp upper bound.
        fp8_min (Optional[float]): FP8 clamp lower bound.
        kv_dtype (Optional[type]): KV cache dtype.
        use_block_kv (bool): Use block KV cache layout. Default: False.
        block_size (Optional[int]): Block size for block KV cache.
        slot_mapping (Optional[torch.Tensor]): Slot mapping [B, S] for block KV.
        store_output_in_sbuf (bool): Hardware-only, ignored. Default: False.
        sbm: Hardware-only, ignored.
        use_auto_allocation (bool): Hardware-only, ignored. Default: False.
        load_input_with_DMA_transpose (bool): Hardware-only, ignored. Default: True.
        is_h_dim_4h_transposed (bool): Whether input H-dim is pre-transposed by 4 (MX only). Default: False.
        weight_layout (QKVWeightLayout): Weight layout. Default: CONTIGUOUS.
        transposed_in (bool): When True, input is [H0, n_prgs, H1_shard, BxS]; converted to [B, S, H] internally. Default: False.
        dtype_mode (DtypeMode): Quantization dtype policy for STATIC/ROW
            weight tiles. Caller must pre-resolve ``DtypeMode.AUTO`` via
            ``resolve_dtype_mode_for_torch_ref``.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.

    Returns:
        Dict with "out" tensor, and "fused_hidden" when fused_residual_add=True.
        For FP8 KV cache path: dict with "q_tensor_hbm", "k_cache", "v_cache".
    """
    if transposed_in:
        # Convert [H0, n_prgs, H1_shard, BxS] back to [B, S, H]
        H0, n_prgs, H1_shard, BxS = input.shape
        H = H0 * n_prgs * H1_shard
        input = input.permute(3, 1, 0, 2).reshape(BxS, 1, H)
    B, S, _H = input.shape
    use_cte = (
        fused_rope
        or k_cache is not None
        or use_block_kv
        or S > SEQLEN_THRESHOLD_FOR_QKV_CTE
        or B * S > _PMAX  # Large batch: qkv_tkg requires B*S <= pmax for SBUF output
    )

    if use_cte:
        # Attach tensor weights to QKNormConfig before passing to CTE torch ref
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
        return qkv_cte_torch_ref(
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
            output_hbm=output_hbm,
            strided_input_config=strided_input_config,
            dtype_mode=dtype_mode,
            q_squared_sum_out=q_squared_sum_out,
            k_squared_sum_out=k_squared_sum_out,
            v_squared_sum_out=v_squared_sum_out,
        )

    return qkv_tkg_torch_ref(
        hidden=input,
        qkv_w=fused_qkv_weights,
        norm_w=gamma_norm_weights,
        fused_add=fused_residual_add,
        mlp_prev=mlp_prev,
        attn_prev=attention_prev,
        d_head=d_head,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        output_layout=output_layout,
        eps=norm_eps,
        norm_type=fused_norm_type,
        quantization_type=quantization_type,
        is_h_dim_4h_transposed=is_h_dim_4h_transposed,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        output_in_sbuf=store_output_in_sbuf,
        qkv_bias=bias,
        norm_bias=layer_norm_bias,
        hidden_actual=hidden_actual,
        dtype_mode=dtype_mode,
    )
