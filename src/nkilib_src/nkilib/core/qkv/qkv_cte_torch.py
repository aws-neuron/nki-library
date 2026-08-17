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

"""PyTorch reference implementation for qkv_cte kernel."""

from typing import Any, Dict, Optional

import nki.language as nl
import numpy as np
import torch

from ..subkernels.norm_torch_dispatch import norm_name2func_torch
from ..utils.common_types import DtypeMode, NormType, QKVOutputLayout, QKVWeightLayout, QuantizationType
from ..utils.kernel_helpers import get_max_positive_value_for_dtype
from ..utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx,
    unpack_float8_e4m3fn_x4,
)

# MX quantization constants
_Q_WIDTH = 4  # MX quantization group width (number of elements packed together)
_PMAX = 128  # Hardware partition dimension size (nl.tile_size.pmax resolves to -1 on host)


def _apply_qk_norm(
    qkv_out: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    d_head: int,
    eps: float,
    q_gamma: Optional[torch.Tensor],
    k_gamma: Optional[torch.Tensor],
    q_norm: Optional[NormType],
    k_norm: Optional[NormType],
) -> None:
    """Apply per-head norm to Q and K heads in-place. V heads are unaltered."""
    for i_head in range(num_q_heads + num_kv_heads):
        is_q = i_head < num_q_heads
        norm_func = q_norm if is_q else k_norm
        if norm_func is None:
            continue
        start = i_head * d_head
        end = start + d_head
        head = qkv_out[:, :, start:end]
        if norm_func == NormType.RMS_NORM:
            rms = torch.sqrt(torch.mean(head**2, dim=-1, keepdim=True) + eps)
            normed = head / rms
        else:
            raise ValueError(f"Unsupported QK norm type: {norm_func}")
        gamma = q_gamma if is_q else k_gamma
        if gamma is not None:
            normed = normed * gamma.to(torch.float32)
        qkv_out[:, :, start:end] = normed


def qkv_cte_torch_ref(
    input: torch.Tensor,
    fused_qkv_weights: torch.Tensor,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[torch.Tensor] = None,
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
    # K-specific gamma-fused caches; accepted for parity. The test framework's
    # gamma_unfused_ref wrapper unfuses gamma before invoking this ref, so
    # k_cos_cache / k_sin_cache are unused here.
    k_cos_cache: Optional[torch.Tensor] = None,  # noqa: ARG001
    k_sin_cache: Optional[torch.Tensor] = None,  # noqa: ARG001
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # --- KV Cache Related
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
    # --- Hardware-only — accepted for kernel signature parity, no-op on CPU
    store_output_in_sbuf: bool = False,  # noqa: ARG001
    sbm: Any = None,  # noqa: ARG001
    use_auto_allocation: bool = False,  # noqa: ARG001
    load_input_with_DMA_transpose: bool = True,  # noqa: ARG001
    # --- Quantization Related
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[torch.Tensor] = None,
    qkv_in_scale: Optional[torch.Tensor] = None,
    # --- Input Swizzle (MX only)
    is_input_swizzled: bool = False,
    # Weight layout; torch ref assumes CONTIGUOUS (default). Other layouts
    # would need an unswizzle in the framework wrapper before calling.
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    # --- QK-Norm Related
    qk_norm_pre_rope=None,
    qk_norm_post_rope=None,
    # --- Output / Input layout — accepted for kernel signature parity, no-op on CPU
    output_hbm: Any = None,  # noqa: ARG001
    strided_input_config: Any = None,  # noqa: ARG001
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    q_squared_sum_out: Any = None,
    k_squared_sum_out: Any = None,
    v_squared_sum_out: Any = None,
) -> Dict[str, torch.Tensor]:
    """PyTorch reference implementation for the QKV CTE (Context Encoding) kernel.

    Implements the same mathematical operations as the ``qkv_cte`` NKI kernel using
    pure PyTorch, serving as ground truth for UnitTestFramework validation.

    The computation pipeline (matching the kernel order) is:
        1. Convert all inputs to float32
        2. If ``is_input_swizzled`` and MX: unswizzle input
        3. If ``fused_residual_add``: input = input + mlp_prev + attention_prev
        4. Apply normalization (NO_NORM / RMS_NORM / RMS_NORM_SKIP_GAMMA / LAYER_NORM)
        5. QKV matmul (NONE/ROW/STATIC/MX paths)
        6. Add bias if provided
        7. If ``fused_rope``: apply RoPE to Q and K heads
        8. If KV cache: split Q/K/V, optionally quantize K/V, scatter into caches
        9. Reshape output per ``output_layout``

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        I: Fused QKV dim = (num_q_heads + 2 * num_kv_heads) * d_head
        d_head: Head dimension
        N: Total number of heads = num_q_heads + 2 * num_kv_heads

    Args:
        input (torch.Tensor): Input hidden states [B, S, H].
        fused_qkv_weights (torch.Tensor): Fused QKV weight matrix [H, I], or
            [H//4, I, 4] fp8 for MX/ROW_MX quantization.
        output_layout (QKVOutputLayout): Output layout: BSD [B,S,I],
            NBSd [N,B,S,d_head], or NBdS [N,B,d_head,S]. Default: BSD.
        bias (Optional[torch.Tensor]): QKV bias [1, I]. Default: None.
        fused_residual_add (Optional[bool]): Compute input + mlp_prev +
            attention_prev. Default: False.
        mlp_prev (Optional[torch.Tensor]): Previous MLP output [B, S, H].
        attention_prev (Optional[torch.Tensor]): Previous attention output [B, S, H].
        fused_norm_type (NormType): Normalization type. Default: NO_NORM.
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
        k_cache (Optional[torch.Tensor]): K cache tensor (mutable). FP8 or BF16.
        v_cache (Optional[torch.Tensor]): V cache tensor (mutable). FP8 or BF16.
        k_scale (Optional[torch.Tensor]): K quantization scale [128, 1]. None for BF16.
        v_scale (Optional[torch.Tensor]): V quantization scale [128, 1]. None for BF16.
        fp8_max (Optional[float]): FP8 clamp upper bound.
        fp8_min (Optional[float]): FP8 clamp lower bound.
        kv_dtype (Optional[type]): KV cache dtype.
        use_block_kv (bool): Use block KV cache layout. Default: False.
        block_size (Optional[int]): Block size for block KV cache.
        slot_mapping (Optional[torch.Tensor]): Slot mapping [B, S] for block KV.
        quantization_type (QuantizationType): Quantization mode. Default: NONE.
        qkv_w_scale (Optional[torch.Tensor]): Weight quantization scale.
        qkv_in_scale (Optional[torch.Tensor]): Input quantization scale.
        is_input_swizzled (bool): Whether input is MX-swizzled. Default: False.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with keys:
            - ``"out"``: QKV output shaped per ``output_layout``.
            For KV cache path, returns:
            - ``"q_tensor_hbm"``: Q projection [B, S, q_dim].
            - ``"k_cache"``: K cache (quantized FP8 or BF16).
            - ``"v_cache"``: V cache (quantized FP8 or BF16).

    Raises:
        ValueError: If required parameters are missing for the requested mode.

    Notes:
        - This is a reference implementation for testing; it prioritises clarity
          over performance.
        - Hardware-only parameters (``sbm``, ``use_auto_allocation``,
          ``store_output_in_sbuf``, ``load_input_with_DMA_transpose``) are
          excluded from the signature as they have no effect on mathematical output.
    """
    # Convert all inputs to float32 because CPU doesn't support half precision
    input_original_dtype = input.dtype
    input = input.to(torch.float32)
    # AUTO must be pre-resolved by the caller (see resolve_dtype_mode_for_torch_ref);
    # the torch ref runs on CPU and can't query hardware.
    assert dtype_mode != DtypeMode.AUTO, (  # noqa: S101
        "qkv_cte_torch_ref requires DtypeMode.AUTO to be pre-resolved by the caller."
    )
    _fp8_e4m3_dtype = nl.float8_e4m3fn if dtype_mode == DtypeMode.OCP else nl.float8_e4m3
    fp8_clip_value = get_max_positive_value_for_dtype(_fp8_e4m3_dtype)
    # Resolve the torch dtype that mirrors ``_fp8_e4m3_dtype`` for the
    # post-clip FP8 round-trip. Torch ships ``float8_e4m3fn`` but historically
    # did not expose the non-FN ``float8_e4m3`` (TRN2's clipped-at-240 variant)
    # on CPU; fall back to clamp-only there by leaving this ``None``.
    _fp8_torch_dtype = torch.float8_e4m3fn if dtype_mode == DtypeMode.OCP else getattr(torch, "float8_e4m3", None)
    if not quantization_type.is_mx():
        fused_qkv_weights = fused_qkv_weights.to(torch.float32)
    mlp_prev = mlp_prev.to(torch.float32) if mlp_prev is not None else None
    attention_prev = attention_prev.to(torch.float32) if attention_prev is not None else None
    gamma_norm_weights = gamma_norm_weights.to(torch.float32) if gamma_norm_weights is not None else None
    layer_norm_bias = layer_norm_bias.to(torch.float32) if layer_norm_bias is not None else None
    bias = bias.to(torch.float32) if bias is not None else None
    cos_cache = cos_cache.to(torch.float32) if cos_cache is not None else None
    sin_cache = sin_cache.to(torch.float32) if sin_cache is not None else None

    if quantization_type == QuantizationType.STATIC:
        if qkv_w_scale is None:
            raise ValueError("qkv_w_scale required for STATIC quantization")
        if qkv_in_scale is None:
            raise ValueError("qkv_in_scale required for STATIC quantization")
        if d_head is None:
            raise ValueError("d_head required for STATIC quantization")
        if num_q_heads is None:
            raise ValueError("num_q_heads required for STATIC quantization")
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required for STATIC quantization")
        qkv_w_scale = qkv_w_scale[0, :].to(torch.float32)
        qkv_in_scale = qkv_in_scale[0, 0].to(torch.float32)

    # Unswizzle input (and gamma) if MX + swizzled.
    # The test generator swizzles both input and gamma_norm_weights when
    # is_h_dim_4h_transposed=True, so we must unswizzle both before norm.
    if is_input_swizzled and quantization_type == QuantizationType.MX:
        B, S, H = input.shape
        input = input.reshape(B * S, _Q_WIDTH, H // (_PMAX * _Q_WIDTH), _PMAX).permute(0, 2, 3, 1).reshape(B, S, H)
        if gamma_norm_weights is not None:
            gamma_norm_weights = (
                gamma_norm_weights.reshape(1, _Q_WIDTH, H // (_PMAX * _Q_WIDTH), _PMAX)
                .permute(0, 2, 3, 1)
                .reshape(1, H)
            )

    # Fused residual addition
    if fused_residual_add:
        if mlp_prev is None:
            raise ValueError("mlp_prev required when fused_residual_add is True")
        if attention_prev is None:
            raise ValueError("attention_prev required when fused_residual_add is True")
        input = input + mlp_prev + attention_prev

    # Normalization dispatch
    if fused_norm_type == NormType.LAYER_NORM:
        input = norm_name2func_torch[fused_norm_type](input, gamma_norm_weights, eps=norm_eps, norm_b=layer_norm_bias)
    elif fused_norm_type in (NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA):
        # RMS_NORM_SKIP_GAMMA: gamma_norm_weights will be None, and
        # rms_norm_torch_ref skips the multiply when gamma is None
        input = norm_name2func_torch[fused_norm_type](
            input, gamma_norm_weights, eps=norm_eps, hidden_actual=hidden_actual
        )

    # QKV matmul

    # ROW_MX: per-row dynamic FP8 quantization with MX_CONTIGUOUS weights
    if quantization_type == QuantizationType.ROW_MX:
        B, S, H = input.shape
        weights_np = fused_qkv_weights if isinstance(fused_qkv_weights, np.ndarray) else fused_qkv_weights.numpy()
        H_quarter, I, _ = weights_np.shape  # [H//4, I, 4] fp8

        # Reverse MX_CONTIGUOUS packing: (H//4, I, 4) -> (H//4, 4, I) -> (H, I)
        w_f32 = torch.from_numpy(weights_np.transpose(0, 2, 1).reshape(H, I).astype(np.float32))

        # Matmul: input [B, S, H] @ weights [H, I] -> [B, S, I]
        qkv_out = input.reshape(B * S, H) @ w_f32
        qkv_out = qkv_out.reshape(B, S, I)

        # Dequant: output *= per_row_input_scale * per_channel_weight_scale
        # qkv_in_scale is [B, S, 1] (passed by test harness to torch ref only)
        in_scale = (
            qkv_in_scale.to(torch.float32)
            if isinstance(qkv_in_scale, torch.Tensor)
            else torch.from_numpy(np.array(qkv_in_scale, dtype=np.float32))
        )
        w_scale = (
            qkv_w_scale.to(torch.float32)
            if isinstance(qkv_w_scale, torch.Tensor)
            else torch.from_numpy(np.array(qkv_w_scale, dtype=np.float32))
        )
        # Broadcast w_scale [1, I] or [128, I] -> use first row for torch ref.
        # [128, I] is assumed to be a uniform broadcast (all rows identical).
        if w_scale.shape[0] == 1:
            w_scale_broadcast = w_scale  # [1, I] broadcasts naturally
        else:
            w_scale_broadcast = w_scale[0:1, :]  # Take first row, [1, I]
        qkv_out = qkv_out * in_scale * w_scale_broadcast

    # STATIC_MX: per-tensor static dequant via MX engine
    elif quantization_type == QuantizationType.STATIC_MX:
        B, S, H = input.shape
        weights_np = fused_qkv_weights if isinstance(fused_qkv_weights, np.ndarray) else fused_qkv_weights.numpy()
        H_quarter, I, _ = weights_np.shape  # [H//4, I, 4] fp8

        # Reverse packing: (H//4, I, 4) -> (H//4, 4, I) -> (H, I)
        w_f32_reordered = weights_np.transpose(0, 2, 1).reshape(H, I).astype(np.float32)

        if weight_layout == QKVWeightLayout.MX_INTERLEAVED:
            h_idx = np.empty(H, dtype=np.int64)
            for p in range(H // 4):
                h_idx[4 * p] = 2 * p
                h_idx[4 * p + 1] = 2 * p + 1
                h_idx[4 * p + 2] = H // 2 + 2 * p
                h_idx[4 * p + 3] = H // 2 + 2 * p + 1
            inv_idx = np.argsort(h_idx)
            w_f32 = torch.from_numpy(w_f32_reordered[inv_idx, :])
        elif weight_layout == QKVWeightLayout.MX_CONTIGUOUS:
            w_f32 = torch.from_numpy(w_f32_reordered)

        in_scale = (
            qkv_in_scale[0, 0].to(torch.float32)
            if isinstance(qkv_in_scale, torch.Tensor)
            else float(qkv_in_scale.flat[0])
        )
        w_scale = (
            qkv_w_scale[0, :].to(torch.float32)
            if isinstance(qkv_w_scale, torch.Tensor)
            else torch.from_numpy(qkv_w_scale[0, :].astype(np.float32))
        )

        is_bf16_input = input_original_dtype == torch.bfloat16
        input_f32 = input.to(torch.float32)

        if is_bf16_input:
            input_f32 = (input_f32 / in_scale).clamp(-fp8_clip_value, fp8_clip_value)
            if _fp8_torch_dtype is not None:
                input_f32 = input_f32.to(_fp8_torch_dtype).to(torch.float32)

        qkv_out = input_f32.reshape(B * S, H) @ w_f32
        qkv_out = qkv_out.reshape(B, S, I)

        # Apply per-Q/K/V dequant scaling
        q_end_idx = num_q_heads * d_head
        k_end_idx = (num_q_heads + num_kv_heads) * d_head
        v_end_idx = (num_q_heads + 2 * num_kv_heads) * d_head
        combined_scale = in_scale * w_scale
        qkv_out[:, :, :q_end_idx] *= combined_scale[0]
        qkv_out[:, :, q_end_idx:k_end_idx] *= combined_scale[1]
        qkv_out[:, :, k_end_idx:v_end_idx] *= combined_scale[2]

    # Native MX path: per-block e8m0 scales
    elif quantization_type == QuantizationType.MX:
        B, S, H = input.shape
        weights_np = fused_qkv_weights if isinstance(fused_qkv_weights, np.ndarray) else fused_qkv_weights.numpy()
        _, I, _ = weights_np.shape

        # Reshape normalized hidden to [P, F] layout for MX block quantization.
        # P = H // _Q_WIDTH, F = _Q_WIDTH * B * S
        hidden_np = input.reshape(B * S, H).T.numpy()  # [H, B*S]
        hidden_np = (
            hidden_np.reshape(H // _Q_WIDTH, _Q_WIDTH, B * S)
            .transpose(0, 2, 1)
            .reshape(H // _Q_WIDTH, _Q_WIDTH * B * S)
            .astype(np.float32)
        )

        # Quantize hidden to MX FP8
        hidden_mx, hidden_scale = quantize_to_mx(hidden_np, nl.float8_e4m3fn_x4)

        # Convert weights [H//4, I, 4] fp8 to float32 [H//4, I*4] for mx_matmul
        weights_unpacked = torch.from_numpy(weights_np.reshape(weights_np.shape[0], I * _Q_WIDTH).astype(np.float32))
        hidden_mx_torch = unpack_float8_e4m3fn_x4(hidden_mx)

        # Prepare scales as float64 torch tensors for mx_matmul
        hidden_scale_torch = torch.from_numpy(hidden_scale.astype(np.float64))
        if isinstance(qkv_w_scale, torch.Tensor):
            qkv_w_scale_torch = qkv_w_scale.to(torch.float64)
        else:
            qkv_w_scale_torch = torch.from_numpy(qkv_w_scale.astype(np.float64))

        qkv_out = mx_matmul(
            stationary=hidden_mx_torch,
            moving=weights_unpacked,
            stationary_scale=hidden_scale_torch,
            moving_scale=qkv_w_scale_torch,
        )

        # Reshape result back to [B, S, I]
        qkv_out = qkv_out.reshape(B, S, I)
    else:
        # NONE, ROW, and STATIC: standard matmul
        if quantization_type == QuantizationType.STATIC:
            # Quantize input: clip(input / in_scale, -fp8_max, fp8_max),
            # then round-trip through FP8 to apply the actual grid
            # rounding the kernel does on hardware. Without the cast the
            # reference is only an approximation: values in the
            # high-magnitude band (snapped to FP8 grid points by the
            # hardware) stay continuous here, drifting from the kernel
            # output on real-model activations where those values are
            # common.
            input = input / qkv_in_scale
            input = input.clip(-fp8_clip_value, fp8_clip_value)
            if _fp8_torch_dtype is not None:
                input = input.to(_fp8_torch_dtype).to(torch.float32)

        qkv_out = input @ fused_qkv_weights

        if quantization_type == QuantizationType.STATIC:
            # Per-head weight scaling
            q_end_idx = num_q_heads * d_head
            k_end_idx = (num_q_heads + num_kv_heads) * d_head
            v_end_idx = (num_q_heads + 2 * num_kv_heads) * d_head
            qkv_out[:, :, :q_end_idx] *= qkv_w_scale[0]
            qkv_out[:, :, q_end_idx:k_end_idx] *= qkv_w_scale[1]
            qkv_out[:, :, k_end_idx:v_end_idx] *= qkv_w_scale[2]
            qkv_out *= qkv_in_scale
        elif quantization_type == QuantizationType.ROW:
            # Per-column weight scaling
            w_scale = (
                qkv_w_scale[0:1, :].to(torch.float32)
                if isinstance(qkv_w_scale, torch.Tensor)
                else torch.from_numpy(qkv_w_scale[0:1, :].astype(np.float32))
            )
            qkv_out = qkv_out * w_scale

    # Add bias
    if bias is not None:
        qkv_out = qkv_out + bias

    # Pre-RoPE QK-norm
    if qk_norm_pre_rope is not None:
        _apply_qk_norm(
            qkv_out,
            num_q_heads,
            num_kv_heads,
            d_head,
            qk_norm_pre_rope.eps,
            qk_norm_pre_rope.q_gamma_norm_weights,
            qk_norm_pre_rope.k_gamma_norm_weights,
            qk_norm_pre_rope.q_norm,
            qk_norm_pre_rope.k_norm,
        )

    # RoPE – first-second-half interleave rotation on Q and K heads
    if fused_rope:
        if cos_cache is None:
            raise ValueError("cos_cache required when fused_rope is True")
        if sin_cache is None:
            raise ValueError("sin_cache required when fused_rope is True")
        if num_q_heads is None:
            raise ValueError("num_q_heads required when fused_rope is True")
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required when fused_rope is True")
        if d_head is None:
            raise ValueError("d_head required when fused_rope is True")

        B, S, I = qkv_out.shape
        q_dim = num_q_heads * d_head
        kv_dim = num_kv_heads * d_head
        half = d_head // 2

        # Both halves of cos are needed because they may differ when gamma is
        # pre-multiplied into the caches (gamma_fused_in_rope_caches).
        cos_first = cos_cache[:, :, :half]  # [B, S, d_head//2]
        cos_second = cos_cache[:, :, half:]  # [B, S, d_head//2]
        sin = sin_cache[:, :, :half]  # [B, S, d_head//2]

        # Split QKV into Q [B, S, q_dim], K [B, S, kv_dim], V [B, S, kv_dim]
        q = qkv_out[:, :, :q_dim]
        k = qkv_out[:, :, q_dim : q_dim + kv_dim]
        v = qkv_out[:, :, q_dim + kv_dim :]

        # Reshape to per-head: [B, S, num_heads, d_head]
        q = q.reshape(B, S, num_q_heads, d_head)
        k = k.reshape(B, S, num_kv_heads, d_head)

        # Split each head into first/second halves along d_head
        q_first = q[:, :, :, :half]  # [B, S, num_q_heads, d_head//2]
        q_second = q[:, :, :, half:]  # [B, S, num_q_heads, d_head//2]
        k_first = k[:, :, :, :half]  # [B, S, num_kv_heads, d_head//2]
        k_second = k[:, :, :, half:]  # [B, S, num_kv_heads, d_head//2]

        # Broadcast cos/sin to [B, S, 1, d_head//2] for head dimension
        cos_first = cos_first.unsqueeze(2)
        cos_second = cos_second.unsqueeze(2)
        sin = sin.unsqueeze(2)

        # Apply rotation (first-second-half interleave layout):
        q_rot_first = q_first * cos_first - q_second * sin
        q_rot_second = q_second * cos_second + q_first * sin
        k_rot_first = k_first * cos_first - k_second * sin
        k_rot_second = k_second * cos_second + k_first * sin

        # Concatenate rotated halves and flatten back to [B, S, dim]
        q = torch.cat([q_rot_first, q_rot_second], dim=-1).reshape(B, S, q_dim)
        k = torch.cat([k_rot_first, k_rot_second], dim=-1).reshape(B, S, kv_dim)

        # Recombine Q, K, V (V is unrotated)
        qkv_out = torch.cat([q, k, v], dim=-1)

    # Post-RoPE QK-norm
    if qk_norm_post_rope is not None:
        _apply_qk_norm(
            qkv_out,
            num_q_heads,
            num_kv_heads,
            d_head,
            qk_norm_post_rope.eps,
            qk_norm_post_rope.q_gamma_norm_weights,
            qk_norm_post_rope.k_gamma_norm_weights,
            qk_norm_post_rope.q_norm,
            qk_norm_post_rope.k_norm,
        )

    # KV cache: FP8 quantized or BF16 direct store
    _kv_cache_provided = k_cache is not None and v_cache is not None
    _kv_scale_provided = k_scale is not None and v_scale is not None
    if _kv_scale_provided and not _kv_cache_provided:
        raise ValueError(
            "KV cache tensors must be provided when scales are provided: "
            "k_cache, v_cache are required with k_scale, v_scale"
        )
    use_kv_quantization = _kv_cache_provided
    _is_bf16_kv = _kv_cache_provided and not _kv_scale_provided

    if use_kv_quantization:
        B, S, I = qkv_out.shape
        q_dim = num_q_heads * d_head
        kv_dim = num_kv_heads * d_head

        # Split QKV into Q, K, V sections
        q = qkv_out[:, :, :q_dim]
        k = qkv_out[:, :, q_dim : q_dim + kv_dim]
        v = qkv_out[:, :, q_dim + kv_dim :]

        if _is_bf16_kv:
            # BF16 path: store directly without quantization
            k_processed = k.numpy()
            v_processed = v.numpy()
        else:
            # FP8 path: quantize K and V
            k_scale_f32 = k_scale.to(torch.float32).flatten()[0]
            v_scale_f32 = v_scale.to(torch.float32).flatten()[0]
            k_processed = (k / k_scale_f32).clamp(fp8_min, fp8_max).numpy()
            v_processed = (v / v_scale_f32).clamp(fp8_min, fp8_max).numpy()

        cache_shape = k_cache.shape
        if use_block_kv:
            if fp8_packed:
                # FP8 packed layout: [num_blocks, num_kv_heads, block_size//2, d_head, 2] fp8
                # where [:,:,:,0] = even seq positions, [:,:,:,1] = odd seq positions.
                num_blocks_actual = k_cache.shape[0]
                k_fp8_full = np.zeros((num_blocks_actual, num_kv_heads, block_size, d_head), dtype=kv_dtype)
                v_cache_out = np.zeros(v_cache.shape, dtype=kv_dtype)
                for b in range(B):
                    for s in range(S):
                        slot = slot_mapping[b, s].item()
                        block_idx = slot // block_size
                        pos_in_block = slot % block_size
                        for h in range(num_kv_heads):
                            k_fp8_full[block_idx, h, pos_in_block, :] = k_processed[b, s, h * d_head : (h + 1) * d_head]
                            v_cache_out[block_idx, h, pos_in_block, :] = v_processed[
                                b, s, h * d_head : (h + 1) * d_head
                            ]
                # Reshape to [nb, num_kv_heads, block_size//2, 2, d_head] then transpose to [nb, num_kv_heads, block_size//2, d_head, 2]
                k_cache_out = k_fp8_full.reshape(num_blocks_actual, num_kv_heads, block_size // 2, 2, d_head).transpose(
                    0, 1, 2, 4, 3
                )
            elif transpose_k_cache:
                k_cache_out = np.zeros(k_cache.shape, dtype=kv_dtype)
                v_cache_out = np.zeros(v_cache.shape, dtype=kv_dtype)

                for b in range(B):
                    for s in range(S):
                        slot = slot_mapping[b, s].item()
                        block_idx = slot // block_size
                        pos_in_block = slot % block_size

                        # K: store transposed [num_blocks * num_kv_heads, d_head, block_size]
                        k_val = k_processed[b, s, :]  # [kv_dim]
                        for h in range(num_kv_heads):
                            row = block_idx * num_kv_heads + h
                            head_start = h * d_head
                            head_end = head_start + d_head
                            k_cache_out[row, :, pos_in_block] = k_val[head_start:head_end]

                        # V: store non-transposed [num_blocks, block_size, kv_dim]
                        v_cache_out[block_idx, pos_in_block, :] = v_processed[b, s, :]
            else:
                # Block KV layout: scatter using slot_mapping
                k_cache_out = np.zeros(cache_shape, dtype=kv_dtype)
                v_cache_out = np.zeros(cache_shape, dtype=kv_dtype)
                for b in range(B):
                    for s in range(S):
                        slot = slot_mapping[b, s].item()
                        block_idx = slot // block_size
                        pos_in_block = slot % block_size
                        k_cache_out[block_idx, pos_in_block, :] = k_processed[b, s, :]
                        v_cache_out[block_idx, pos_in_block, :] = v_processed[b, s, :]
        else:
            if transpose_k_cache:
                # Flat transposed K cache layout: [B, kv_dim, max_seq_len]
                k_cache_out = np.zeros(k_cache.shape, dtype=kv_dtype)
                k_cache_out[:, :, :S] = np.transpose(k_processed, (0, 2, 1)).astype(kv_dtype)
                v_cache_out = np.zeros(v_cache.shape, dtype=kv_dtype)
                v_cache_out[:, :S, :] = v_processed.astype(kv_dtype)
            else:
                # Non-block KV layout: write directly at [batch, :S, :]
                k_cache_out = np.zeros(cache_shape, dtype=kv_dtype)
                v_cache_out = np.zeros(v_cache.shape, dtype=kv_dtype)
                k_cache_out[:, :S, :] = k_processed.astype(kv_dtype)
                v_cache_out[:, :S, :] = v_processed.astype(kv_dtype)

        # Return numpy arrays directly (torch_ref_wrapper handles dict values)
        return {
            "q_tensor_hbm": q,
            "k_cache": k_cache_out,
            "v_cache": v_cache_out,
        }

    # Reshape output per output_layout
    B, S, I = qkv_out.shape

    # Optional per-segment sum of squares over the head dimension, computed from
    # the [B, S, I] projection output before it is reshaped to the requested
    # output layout, while the Q/K/V segments are still contiguous along the
    # last dimension.
    squared_sums: Dict[str, torch.Tensor] = {}
    if q_squared_sum_out is not None or k_squared_sum_out is not None or v_squared_sum_out is not None:
        q_dim = num_q_heads * d_head
        kv_dim = num_kv_heads * d_head
        if q_squared_sum_out is not None:
            squared_sums["q_squared_sum_out"] = torch.sum(qkv_out[:, :, :q_dim] ** 2, dim=-1, keepdim=True)
        if k_squared_sum_out is not None:
            squared_sums["k_squared_sum_out"] = torch.sum(
                qkv_out[:, :, q_dim : q_dim + kv_dim] ** 2, dim=-1, keepdim=True
            )
        if v_squared_sum_out is not None:
            squared_sums["v_squared_sum_out"] = torch.sum(qkv_out[:, :, q_dim + kv_dim :] ** 2, dim=-1, keepdim=True)

    if output_layout in (QKVOutputLayout.NBSd, QKVOutputLayout.NBdS):
        if d_head is None:
            raise ValueError(f"d_head required for {output_layout} output layout")
        num_heads = I // d_head

    if output_layout == QKVOutputLayout.NBSd:
        qkv_out = qkv_out.reshape(B, S, num_heads, d_head)
        qkv_out = qkv_out.permute(2, 0, 1, 3)
    elif output_layout == QKVOutputLayout.NBdS:
        qkv_out = qkv_out.reshape(B, S, num_heads, d_head)
        qkv_out = qkv_out.permute(2, 0, 3, 1)

    result = {"out": qkv_out}
    result.update(squared_sums)
    return result
