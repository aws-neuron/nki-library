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

import torch

from ...utils.common_types import QuantizationType

# FP8 max values for different formats
_FP8_E4M3_MAX = 240.0
_FP8_E5M2_MAX = 57344.0


def _get_max_value_for_dtype(dtype) -> float:
    """
    Get maximum representable value for FP8 dtypes.

    Args:
        dtype: PyTorch or string dtype to check.

    Returns:
        float: Maximum value for the dtype (240.0 for e4m3, 57344.0 for e5m2).
    """
    dtype_str = str(dtype)
    if "float8_e4m3" in dtype_str:
        return _FP8_E4M3_MAX
    elif "float8_e5m2" in dtype_str:
        return _FP8_E5M2_MAX
    return _FP8_E4M3_MAX


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
    max_val: float,
) -> torch.Tensor:
    """
    Perform static quantization by scaling and clamping.

    Args:
        input_tensor (torch.Tensor): Input tensor to quantize.
        quant_scale (torch.Tensor): Quantization scale tensor.
        max_val (float): Maximum value for clamping (FP8 range).

    Returns:
        torch.Tensor: Quantized tensor clamped to [-max_val, max_val].
    """
    scaled_tensor = _scale_with_broadcast(input_tensor, 1 / quant_scale)
    return torch.clamp(scaled_tensor, -max_val, max_val)


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
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    dtype=torch.float32,
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
        input_scale (Optional[torch.Tensor]): [128, 1], Input quantization scale (for STATIC).
        weight_scale (Optional[torch.Tensor]): [128, 1], Weight quantization scale (for STATIC).
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
            quantized_input = scale_and_clamp(attn_reshaped, input_scale)
            out = dequantize(quantized_input @ weight, weight_scale, input_scale)
        else:
            out = attn_reshaped @ weight
        out = out + bias if bias else out
        return out
    """
    batch_size, num_heads, head_dim, seq_len = attention.shape

    # Convert to float32 for computation
    attention = attention.float()
    weight = weight.float()

    # Reshape attention from [B, N, D, S] to [B, S, N*D]
    attn_reshaped = attention.permute(0, 3, 1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    if quantization_type == QuantizationType.STATIC:
        max_val = _get_max_value_for_dtype(weight.dtype)
        quantized_input = _perform_static_quant(attn_reshaped, input_scale, max_val)
        out = _perform_projection(quantized_input, weight, weight_scale, input_scale)
    else:
        out = attn_reshaped @ weight

    if bias is not None:
        out = out + bias.float()

    return out.to(dtype)
