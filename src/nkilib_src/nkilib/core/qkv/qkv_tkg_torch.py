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

"""PyTorch reference implementation for qkv_tkg kernel."""

from typing import Any, Dict, Optional

import nki.language as nl
import numpy as np
import torch

from ..subkernels.norm_torch_dispatch import norm_name2func_torch
from ..utils.common_types import DtypeMode, NormType, QKVOutputLayout, QuantizationType
from ..utils.kernel_helpers import get_max_positive_value_for_dtype

P_MAX = 128
_Q_WIDTH = 4


def qkv_tkg_torch_ref(
    hidden: torch.Tensor,
    qkv_w: torch.Tensor,
    norm_w: Optional[torch.Tensor] = None,
    fused_add: bool = False,
    mlp_prev: Optional[torch.Tensor] = None,
    attn_prev: Optional[torch.Tensor] = None,
    d_head: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    is_h_dim_4h_transposed: bool = False,
    qkv_w_scale: Optional[torch.Tensor] = None,
    qkv_in_scale: Optional[torch.Tensor] = None,
    output_in_sbuf: bool = False,
    qkv_bias: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    hidden_actual: Optional[int] = None,
    sbm: Any = None,  # noqa: ARG001 — SbufManager is hardware-only; accepted for signature parity with kernel
    transposed_in: bool = False,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> Dict[str, torch.Tensor]:
    """
    PyTorch reference implementation for qkv_tkg kernel.

    This is a reference implementation for testing the qkv_tkg kernel.
    Implements the same mathematical operation using PyTorch operations.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        I: QKV projection output dimension (num_q_heads + 2 * num_kv_heads) * d_head
        D: Head dimension (d_head)
        N: Number of heads

    Args:
        hidden (torch.Tensor): Input hidden states tensor. Shape: [B, S, H] when in HBM,
            [H0=128, BxS, H1] when in SBUF.
        qkv_w (torch.Tensor): QKV projection weight tensor. Shape: [H, I].
        norm_w (torch.Tensor, optional): Normalization weight tensor. Required when
            norm_type is RMS_NORM or LAYER_NORM. Shape: [1, H].
        fused_add (bool): Enable fused residual addition (hidden + attn_prev + mlp_prev).
            Default: False.
        mlp_prev (torch.Tensor, optional): Previous MLP residual tensor. Required when
            fused_add is True. Shape: [B, S, H].
        attn_prev (torch.Tensor, optional): Previous attention residual tensor. Required
            when fused_add is True. Shape: [B, S, H].
        d_head (int, optional): Head dimension size D. Required for static quantization and NBSd and NBdS output layouts.
        num_kv_heads (int, optional): Number of key/value heads. Required for FP8
            quantization.
        num_q_heads (int, optional): Number of query heads. Required for FP8 quantization.
        output_layout (QKVOutputLayout): Output tensor layout format. BSD: [B, S, I] or
            NBSd: [N, B, S, D]. Default: QKVOutputLayout.BSD.
        eps (float): Epsilon value for numerical stability in normalization. Default: 1e-6.
        norm_type (NormType): Type of normalization (NO_NORM, RMS_NORM, or LAYER_NORM).
            Default: NormType.RMS_NORM.
        quantization_type (QuantizationType): Type of quantization (NONE, ROW, STATIC, STATIC_MX, ROW_MX).
            Default: QuantizationType.NONE.
        qkv_w_scale (torch.Tensor, optional): QKV weight scale tensor for quantization.
            Shape: [1, I] or [128, I] for row quantization, [1, 3] or [128, 3] for static.
        qkv_in_scale (torch.Tensor, optional): QKV input scale tensor. Only required for
            static quantization. Shape: [1, 1] or [128, 1].
        output_in_sbuf (bool): If True, output is kept in SBUF; otherwise stored to HBM.
            Default: False. Only supports single I-shard when True.
        qkv_bias (torch.Tensor, optional): Bias tensor for QKV projection. Shape: [1, I].
        norm_bias (torch.Tensor, optional): LayerNorm beta parameter tensor. Required when
            norm_type is LAYER_NORM. Shape: [1, H].
        hidden_actual (int, optional): Actual hidden dimension for padded input tensors.
            If specified, normalization uses this value instead of H for mean calculation.
        transposed_in (bool): When True, hidden is [H0, n_prgs, H1_shard, BxS]; converted
            to [B, S, H] internally. Default: False.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with the following keys:
            - "out": QKV projection output tensor. Shape: [B, S, I] for BSD layout,
              [N, B, S, D] for NBSd layout.
            - "fused_hidden" (only when fused_add=True): Result of the fused residual
              addition (hidden + mlp_prev + attn_prev). Shape: [B, S, H].

    Notes:
        - This implementation prioritizes clarity over performance
        - Hardware-specific parameters are ignored as they don't affect the mathematical result
    """

    # Convert to float32 because CPU doesn't support half precision
    hidden = hidden.to(torch.float32)
    if transposed_in:
        # Convert [H0, n_prgs, H1_shard, BxS] back to [B, S, H]
        H0, n_prgs, H1_shard, BxS = hidden.shape
        H = H0 * n_prgs * H1_shard
        hidden = hidden.permute(3, 1, 0, 2).reshape(BxS, 1, H)
    mlp_prev = mlp_prev.to(torch.float32) if mlp_prev is not None else None
    attn_prev = attn_prev.to(torch.float32) if attn_prev is not None else None
    norm_w = norm_w.to(torch.float32) if norm_w is not None else None
    norm_bias = norm_bias.to(torch.float32) if norm_bias is not None else None
    qkv_bias = qkv_bias.to(torch.float32) if qkv_bias is not None else None

    if quantization_type == QuantizationType.MX:
        return _qkv_tkg_mx_torch_ref(
            hidden=hidden,
            qkv_w=qkv_w,
            norm_w=norm_w,
            fused_add=fused_add,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            qkv_bias=qkv_bias,
            norm_bias=norm_bias,
            hidden_actual=hidden_actual,
        )

    is_static_mx = quantization_type == QuantizationType.STATIC_MX
    is_row_mx = quantization_type == QuantizationType.ROW_MX
    is_static = quantization_type == QuantizationType.STATIC

    # STATIC_MX / ROW_MX: weights are unpacked fp8 [H//4, I, 4] -> float32 [H, I]
    if is_static_mx or is_row_mx:
        weights_np = qkv_w if isinstance(qkv_w, np.ndarray) else qkv_w.numpy()
        # weights_np is [H//4, I, 4] unpacked fp8
        H_quarter = weights_np.shape[0]
        I = weights_np.shape[1]
        qkv_w = torch.from_numpy(weights_np.transpose(0, 2, 1).reshape(H_quarter * _Q_WIDTH, I).astype(np.float32))
    else:
        qkv_w = qkv_w.to(torch.float32)

    if is_static or is_static_mx:
        qkv_in_scale = (
            float(np.asarray(qkv_in_scale).flat[0])
            if not isinstance(qkv_in_scale, (int, float))
            else float(qkv_in_scale)
        )
        qkv_w_scale = torch.from_numpy(np.asarray(qkv_w_scale).reshape(-1).astype(np.float32))
    elif quantization_type == QuantizationType.ROW or is_row_mx:
        qkv_w_scale = (
            qkv_w_scale[0, :].to(torch.float32)
            if isinstance(qkv_w_scale, torch.Tensor)
            else torch.from_numpy(np.asarray(qkv_w_scale).reshape(-1).astype(np.float32))
        )

    B, S, H = hidden.shape
    fused_hidden = None

    if fused_add:
        if mlp_prev is None:
            raise ValueError("mlp_prev required when fused_add is True")
        if attn_prev is None:
            raise ValueError("attn_prev required when fused_add is True")
        hidden = hidden + mlp_prev + attn_prev
        fused_hidden = hidden

    # STATIC_MX / ROW_MX: unswizzle input and gamma (undo 4H pre-shuffle)
    if (is_static_mx or is_row_mx) and is_h_dim_4h_transposed:
        hidden = hidden.reshape(B * S, _Q_WIDTH, H // (P_MAX * _Q_WIDTH), P_MAX).permute(0, 2, 3, 1).reshape(B, S, H)
        if norm_w is not None:
            norm_w = norm_w.reshape(1, _Q_WIDTH, H // (P_MAX * _Q_WIDTH), P_MAX).permute(0, 2, 3, 1).reshape(1, H)

    if norm_type == NormType.RMS_NORM:
        hidden = norm_name2func_torch[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias, hidden_actual=hidden_actual)
    else:
        hidden = norm_name2func_torch[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias)

    if is_static or is_static_mx:
        if d_head is None:
            raise ValueError("d_head required for STATIC/STATIC_MX quantization")
        if num_q_heads is None:
            raise ValueError("num_q_heads required for STATIC/STATIC_MX quantization")
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required for STATIC/STATIC_MX quantization")
        # STATIC_MX is always OCP. STATIC follows dtype_mode (caller pre-resolves AUTO).
        assert dtype_mode != DtypeMode.AUTO, (  # noqa: S101
            "qkv_tkg_torch_ref requires DtypeMode.AUTO to be pre-resolved by the caller."
        )
        if is_static_mx or dtype_mode == DtypeMode.OCP:
            _fp8_e4m3_dtype = nl.float8_e4m3fn
        else:
            _fp8_e4m3_dtype = nl.float8_e4m3
        clip_value = get_max_positive_value_for_dtype(_fp8_e4m3_dtype)
        hidden = (hidden / qkv_in_scale).clamp(-clip_value, clip_value)

    if is_row_mx:
        # ROW_MX is TRN3-only, always OCP.
        clip_value = get_max_positive_value_for_dtype(nl.float8_e4m3fn)
        absmax = hidden.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        row_dequant_scale = absmax / clip_value
        hidden = (hidden / row_dequant_scale).clamp(-clip_value, clip_value)

    # Main qkv matmul
    qkv_out = hidden @ qkv_w

    if is_static or is_static_mx:
        # Per-Q/K/V dequant: combined_scale = in_scale * w_scale
        combined_scale = qkv_in_scale * qkv_w_scale  # [3]
        q_end_idx = num_q_heads * d_head
        k_end_idx = (num_q_heads + num_kv_heads) * d_head
        v_end_idx = (num_q_heads + 2 * num_kv_heads) * d_head
        qkv_out[:, :, :q_end_idx] *= combined_scale[0]
        qkv_out[:, :, q_end_idx:k_end_idx] *= combined_scale[1]
        qkv_out[:, :, k_end_idx:v_end_idx] *= combined_scale[2]
    elif quantization_type == QuantizationType.ROW:
        qkv_out = qkv_out * qkv_w_scale
    elif is_row_mx:
        # ROW_MX: per-token input dequant * per-column weight dequant
        qkv_out = qkv_out * row_dequant_scale * qkv_w_scale

    if qkv_bias is not None:
        qkv_out += qkv_bias

    B, S, d_heads = qkv_out.shape

    if output_layout in (QKVOutputLayout.NBSd, QKVOutputLayout.NBdS):
        if d_head is None:
            raise ValueError(f"d_head required for {output_layout} output layout")
        num_heads = d_heads // d_head

    if output_layout == QKVOutputLayout.NBdS:
        qkv_out = torch.reshape(qkv_out, (B, S, num_heads, d_head))
        qkv_out = torch.permute(qkv_out, (2, 0, 3, 1))
    elif output_layout == QKVOutputLayout.NBSd:
        qkv_out = torch.reshape(qkv_out, (B, S, num_heads, d_head))
        qkv_out = torch.permute(qkv_out, (2, 0, 1, 3))

    # When transposed_in, input was reshaped to (BxS, 1, H), so output is (BxS, 1, I).
    # Squeeze to (BxS, I) to match kernel output (which skips the B,S reshape).
    if transposed_in:
        qkv_out = qkv_out.squeeze(1)

    if fused_add:
        return {"out": qkv_out, "fused_hidden": fused_hidden}
    else:
        return {"out": qkv_out}


def _qkv_tkg_mx_torch_ref(
    hidden: torch.Tensor,
    qkv_w,  # numpy fp8 [H//4, I, 4]
    norm_w: Optional[torch.Tensor],
    fused_add: bool,
    mlp_prev: Optional[torch.Tensor],
    attn_prev: Optional[torch.Tensor],
    d_head: Optional[int],
    num_kv_heads: Optional[int],
    num_q_heads: Optional[int],
    output_layout: QKVOutputLayout,
    eps: float,
    norm_type: NormType,
    is_h_dim_4h_transposed: bool,
    qkv_w_scale,
    qkv_in_scale,
    qkv_bias: Optional[torch.Tensor],
    norm_bias: Optional[torch.Tensor],
    hidden_actual: Optional[int],
) -> Dict[str, torch.Tensor]:
    """MX torch reference for QKV TKG.

    Mirrors the kernel: unswizzle input/gamma, norm, MX block quantize,
    mx_matmul with real MX scales, bias.
    """
    from ..utils.mx_torch_common import (
        mx_matmul,
        quantize_to_mx,
        unpack_float8_e4m3fn_x4,
    )

    B, S, H = hidden.shape
    fused_hidden = None

    # Fused residual add
    if fused_add:
        hidden = hidden + mlp_prev + attn_prev
        fused_hidden = hidden

    # Unswizzle input and gamma if is_h_dim_4h_transposed
    if is_h_dim_4h_transposed:
        hidden = hidden.reshape(B * S, _Q_WIDTH, H // (P_MAX * _Q_WIDTH), P_MAX).permute(0, 2, 3, 1).reshape(B, S, H)
        if norm_w != None:
            norm_w = norm_w.reshape(1, _Q_WIDTH, H // (P_MAX * _Q_WIDTH), P_MAX).permute(0, 2, 3, 1).reshape(1, H)

    # Normalization
    if norm_type == NormType.RMS_NORM:
        hidden = norm_name2func_torch[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias, hidden_actual=hidden_actual)
    elif norm_type != NormType.NO_NORM:
        hidden = norm_name2func_torch[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias)

    # MX block quantization of input
    weights_np = qkv_w if isinstance(qkv_w, np.ndarray) else qkv_w.numpy()
    # weights_np is [H//4, I, 4] unpacked fp8
    H_quarter, I, _pack = weights_np.shape

    hidden_np = hidden.reshape(B * S, H).T.numpy()  # [H, B*S]
    hidden_np = (
        hidden_np.reshape(H // _Q_WIDTH, _Q_WIDTH, B * S)
        .transpose(0, 2, 1)
        .reshape(H // _Q_WIDTH, _Q_WIDTH * B * S)
        .astype(np.float32)
    )
    hidden_mx, hidden_scale = quantize_to_mx(hidden_np, nl.float8_e4m3fn_x4)

    # Unpack hidden from x4 packed format; weights are already unpacked [H//4, I, 4]
    weights_unpacked = torch.from_numpy(weights_np.reshape(H_quarter, I * _Q_WIDTH).astype(np.float32))
    hidden_mx_torch = unpack_float8_e4m3fn_x4(hidden_mx)

    # Prepare scales
    hidden_scale_torch = torch.from_numpy(hidden_scale.astype(np.float64))
    if isinstance(qkv_w_scale, torch.Tensor):
        qkv_w_scale_torch = qkv_w_scale.to(torch.float64)
    else:
        qkv_w_scale_torch = torch.from_numpy(np.asarray(qkv_w_scale).astype(np.float64))

    qkv_out = mx_matmul(
        stationary=hidden_mx_torch,
        moving=weights_unpacked,
        stationary_scale=hidden_scale_torch,
        moving_scale=qkv_w_scale_torch,
    )
    qkv_out = qkv_out.reshape(B, S, I).to(torch.float32)

    # Bias
    if qkv_bias != None:
        qkv_out += qkv_bias

    # Output layout
    if output_layout in (QKVOutputLayout.NBSd, QKVOutputLayout.NBdS):
        num_heads = I // d_head
        qkv_out = qkv_out.reshape(B, S, num_heads, d_head)
        if output_layout == QKVOutputLayout.NBSd:
            qkv_out = qkv_out.permute(2, 0, 1, 3)
        else:
            qkv_out = qkv_out.permute(2, 0, 3, 1)

    if fused_add:
        return {"out": qkv_out, "fused_hidden": fused_hidden}
    return {"out": qkv_out}
