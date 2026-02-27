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

from test.integration.nkilib.utils.test_kernel_common import norm_name2func_torch
from typing import Dict, Optional

import torch

from ..utils.common_types import NormType, QKVOutputLayout, QuantizationType


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
    qkv_w_scale: Optional[torch.Tensor] = None,
    qkv_in_scale: Optional[torch.Tensor] = None,
    output_in_sbuf: bool = False,
    qkv_bias: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    hidden_actual: Optional[int] = None,
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
        quantization_type (QuantizationType): Type of quantization (NONE, ROW, STATIC).
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
    mlp_prev = mlp_prev.to(torch.float32) if mlp_prev is not None else None
    attn_prev = attn_prev.to(torch.float32) if attn_prev is not None else None
    qkv_w = qkv_w.to(torch.float32)
    norm_w = norm_w.to(torch.float32) if norm_w is not None else None
    norm_bias = norm_bias.to(torch.float32) if norm_bias is not None else None
    qkv_bias = qkv_bias.to(torch.float32) if qkv_bias is not None else None
    if quantization_type == QuantizationType.STATIC:
        qkv_w_scale = qkv_w_scale[0, :].to(torch.float32) if qkv_w_scale is not None else None
        qkv_in_scale = qkv_in_scale[0, 0].to(torch.float32) if qkv_in_scale is not None else None

    if fused_add:
        if mlp_prev is None:
            raise ValueError("mlp_prev required when fused_add is True")
        if attn_prev is None:
            raise ValueError("attn_prev required when fused_add is True")
        hidden = hidden + mlp_prev + attn_prev
        fused_hidden = hidden  # Save pre-norm sum for output

    if norm_type == NormType.RMS_NORM:
        hidden = norm_name2func_torch[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias, hidden_actual=hidden_actual)
    else:
        hidden = norm_name2func_torch[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias)

    # FP8 quantization clipping constant
    FP8_CLIP_VALUE = 240

    if quantization_type == QuantizationType.STATIC:
        if d_head is None:
            raise ValueError("d_head required for STATIC quantization")
        if num_q_heads is None:
            raise ValueError("num_q_heads required for STATIC quantization")
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required for STATIC quantization")
        if qkv_w_scale is None:
            raise ValueError("qkv_w_scale required for STATIC quantization")
        if qkv_in_scale is None:
            raise ValueError("qkv_in_scale required for STATIC quantization")
        hidden /= qkv_in_scale
        hidden = hidden.clip(-FP8_CLIP_VALUE, FP8_CLIP_VALUE)
    elif quantization_type == QuantizationType.ROW:
        raise NotImplementedError("ROW quantization not supported")

    # Main qkv matmul.
    qkv_out = hidden @ qkv_w

    if quantization_type == QuantizationType.STATIC:
        # Apply per-head scaling for Q, K, V sections
        # qkv_out shape is [B, S, num_heads * d_head], where the q, k, v heads are laid out sequentially
        q_end_idx = num_q_heads * d_head
        k_end_idx = (num_q_heads + num_kv_heads) * d_head
        v_end_idx = (num_q_heads + 2 * num_kv_heads) * d_head
        qkv_out[:, :, :q_end_idx] *= qkv_w_scale[0]
        qkv_out[:, :, q_end_idx:k_end_idx] *= qkv_w_scale[1]
        qkv_out[:, :, k_end_idx:v_end_idx] *= qkv_w_scale[2]
        qkv_out *= qkv_in_scale

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

    if fused_add:
        return {"out": qkv_out, "fused_hidden": fused_hidden}
    else:
        return {"out": qkv_out}
