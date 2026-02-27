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

"""PyTorch reference implementation for output projection CTE kernel."""

import math
from typing import Optional

import numpy as np
import torch

from ...utils.common_types import QuantizationType
from ...utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx_fp8,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
)

# FP8 max values for different formats
_FP8_E4M3_MAX = 240.0
_FP8_E4M3FN_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


def _get_min_max_for_dtype(dtype) -> tuple[float, float]:
    """
    Get min and max representable values for FP8 dtypes.

    Args:
        dtype: PyTorch or string dtype to check.

    Returns:
        tuple[float, float]: (min_val, max_val) for the dtype.
    """
    dtype_str = str(dtype)
    if "float8_e4m3fn" in dtype_str:
        return (-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX)
    elif "float8_e4m3" in dtype_str:
        return (-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    elif "float8_e5m2" in dtype_str:
        return (-_FP8_E5M2_MAX, _FP8_E5M2_MAX)
    return (-_FP8_E4M3_MAX, _FP8_E4M3_MAX)


def _scale_with_broadcast(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply scale with broadcasting for FP8 quantization.

    Tiles the scale tensor to match the input tensor dimensions and applies
    element-wise multiplication.

    Args:
        tensor (torch.Tensor): Input tensor to scale (2D or 3D).
        scale (torch.Tensor): Scale tensor to broadcast and apply.

    Returns:
        torch.Tensor: Scaled tensor in float32.
    """
    if len(tensor.shape) == 3 and len(scale.shape) == 2:
        tiled_scale = scale.repeat(
            tensor.shape[0],
            math.ceil(tensor.shape[1] / scale.shape[0]),
            math.ceil(tensor.shape[2] / scale.shape[1]),
        )
        tiled_scale = tiled_scale[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]]
        return tensor.float() * tiled_scale
    elif len(tensor.shape) == 3 and len(scale.shape) == 3:
        return tensor.float() * scale
    else:
        return tensor.float() * scale.repeat(
            math.ceil(tensor.shape[0] / scale.shape[0]),
            math.ceil(tensor.shape[1] / scale.shape[1]),
        )


def _perform_static_quant(
    input_tensor: torch.Tensor,
    quant_scale: torch.Tensor,
    min_val: float,
    max_val: float,
) -> torch.Tensor:
    """
    Perform static quantization by scaling and clamping.

    Args:
        input_tensor (torch.Tensor): Input tensor to quantize.
        quant_scale (torch.Tensor): Quantization scale tensor.
        min_val (float): Minimum value for clamping (FP8 range).
        max_val (float): Maximum value for clamping (FP8 range).

    Returns:
        torch.Tensor: Quantized tensor clamped to [min_val, max_val].
    """
    scaled_tensor = _scale_with_broadcast(input_tensor, 1 / quant_scale)
    return torch.clamp(scaled_tensor, min_val, max_val)


def _perform_projection(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Perform quantized projection with dequantization.

    Computes matmul and applies combined scale for dequantization.

    Args:
        input_tensor (torch.Tensor): Quantized input tensor [B, S, N*D].
        weight (torch.Tensor): Weight tensor [N*D, H].
        weight_scale (torch.Tensor): Weight quantization scale.
        input_scale (torch.Tensor): Input quantization scale.

    Returns:
        torch.Tensor: Dequantized projection result [B, S, H].
    """
    return _scale_with_broadcast(input_tensor @ weight, weight_scale * input_scale)


def output_projection_cte_torch_ref(
    attention: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scales: Optional[torch.Tensor] = None,
    weight_scales: Optional[torch.Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    output_dtype=torch.float32,
) -> torch.Tensor:
    """
    PyTorch reference implementation of output projection for CTE.

    This is a reference implementation for testing the NKI output_projection_cte kernel.
    It implements the same mathematical operation using PyTorch operations.

    Dimensions:
        B: Batch size
        N: Number of heads
        D: Head dimension
        S: Sequence length
        H: Hidden dimension

    Args:
        attention (torch.Tensor): [B, N, D, S], Input tensor from attention block.
        weight (torch.Tensor): [N * D, H], Weight tensor.
        bias (Optional[torch.Tensor]): [1, H], Optional bias tensor.
        input_scales (Optional[torch.Tensor]): [128, 1], Input quantization scale (for STATIC).
        weight_scales (Optional[torch.Tensor]): [128, 1], Weight quantization scale (for STATIC).
        quantization_type (QuantizationType): Type of quantization (NONE, STATIC).
        dtype: Output data type.

    Returns:
        torch.Tensor: [B, S, H], Output tensor.

    Notes:
        - Hardware-specific parameters (LNC sharding) are not included as they
          don't affect the mathematical result.
        - This implementation prioritizes clarity over performance.

    Pseudocode:
        attn_reshaped = attention.permute(0, 3, 1, 2).reshape(B, S, N*D)
        if quantization_type == STATIC:
            quantized_input = scale_and_clamp(attn_reshaped, input_scales)
            out = dequantize(quantized_input @ weight, weight_scales, input_scales)
        else:
            out = attn_reshaped @ weight
        out = out + bias if bias else out
        return out
    """
    batch_size, num_heads, head_dim, seq_len = attention.shape

    # Convert to float32 for computation
    attention = attention.float()
    weight = weight.float()

    # STATIC_MX weight is pre-shuffled for kernel; reverse to logical layout
    if quantization_type == QuantizationType.STATIC_MX:
        nd = weight.shape[0]
        if nd % 4 == 0:
            w = weight.reshape(nd // 4, weight.shape[1], 4)
            weight = w.permute(0, 2, 1).reshape(nd, weight.shape[1])

    # Reshape attention from [B, N, D, S] to [B, S, N*D]
    attn_reshaped = attention.permute(0, 3, 1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    if quantization_type == QuantizationType.STATIC:
        min_val, max_val = _get_min_max_for_dtype(weight.dtype)
        quantized_input = _perform_static_quant(attn_reshaped, input_scales, min_val, max_val)
        out = _perform_projection(quantized_input, weight, weight_scales, input_scales)
    elif quantization_type == QuantizationType.STATIC_MX:
        min_val, max_val = _get_min_max_for_dtype(torch.float8_e4m3fn)
        quantized_input = _perform_static_quant(attn_reshaped, input_scales, min_val, max_val)
        out = _perform_projection(quantized_input, weight, weight_scales, input_scales)
    else:
        out = attn_reshaped @ weight

    if bias != None:
        out = out + bias.float()

    return out.to(output_dtype)


def output_projection_cte_mx_torch_ref(
    attention,
    weight,
    bias=None,
    quantization_type: QuantizationType = QuantizationType.MX,
    input_scales=None,
    weight_scales=None,
    output_dtype=None,
) -> dict[str, np.ndarray]:
    """PyTorch reference for MX FP4 output projection CTE.

    Handles both online quantization (bf16 attention) and pre-quantized
    (float8_e4m3fn_x4 attention) paths.

    Args:
        attention: Online: numpy bf16 [B, N, D, S]. Pre-quantized: numpy uint8 [B, 1, D_packed, S].
        weight: numpy float4_e2m1fn_x4 [N*D//4, H].
        bias: Optional[numpy bf16 [1, H]], bias tensor.
        quantization_type: must be QuantizationType.MX.
        input_scales: Pre-quantized only: numpy uint8 [B, D_packed//8, S]. None for online.
        weight_scales: numpy uint8 [N*D//32, H].
        output_dtype: unused.

    Returns:
        dict with "out": numpy float32 [B, S, H].
    """
    # Convert to numpy if torch tensors (torch_ref_wrapper may pass tensors)
    if isinstance(attention, torch.Tensor):
        attention = attention.numpy()
    if isinstance(weight, torch.Tensor):
        weight = weight.numpy()
    if input_scales is not None and isinstance(input_scales, torch.Tensor):
        input_scales = input_scales.numpy()
    if isinstance(weight_scales, torch.Tensor):
        weight_scales = weight_scales.numpy()

    batch = attention.shape[0]
    hidden = weight.shape[1]
    seqlen = attention.shape[3]
    prequantized = input_scales is not None

    if not prequantized:
        n_head, d_head = attention.shape[1], attention.shape[2]

    # Unpack weight: float4_e2m1fn_x4 [N*D//4, H] -> float32 [N*D, H]
    w_unpacked = unpack_float4_x4(weight.reshape(-1, hidden))
    w_scale = torch.from_numpy(weight_scales.reshape(-1, hidden)).float()

    results = []
    for b in range(batch):
        if prequantized:
            # attention[b] is [1, D_packed, S], reshape to [D_packed, S]
            inp_packed = attention[b].reshape(-1, seqlen)
            inp_unpacked = unpack_float8_e4m3fn_x4(inp_packed)
            inp_scale = torch.from_numpy(input_scales[b].reshape(-1, seqlen)).float()
        else:
            # bf16 [N, D, S] -> float32 [N*D//4, 4, S] -> [N*D//4, S*4] -> quantize
            attn_b = attention[b].astype(np.float32)
            attn_b = attn_b.reshape(n_head * d_head // 4, 4, seqlen)
            attn_b = np.transpose(attn_b, (0, 2, 1)).reshape(-1, seqlen * 4)
            inp_packed, inp_scale_np = quantize_to_mx_fp8(attn_b)
            inp_unpacked = unpack_float8_e4m3fn_x4(inp_packed)
            inp_scale = torch.from_numpy(inp_scale_np).float()

        # mx_matmul: stationary=[D, S], moving=[D, H] -> [S, H]
        result_b = mx_matmul(inp_unpacked, w_unpacked, inp_scale, w_scale)
        results.append(result_b)

    out = torch.stack(results, dim=0)
    if bias is not None:
        bias_np = bias.numpy() if isinstance(bias, torch.Tensor) else bias
        out = out + torch.from_numpy(bias_np.astype(np.float32))

    return {"out": out.numpy()}
