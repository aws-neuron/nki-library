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

import nki.language as nl
import numpy as np
import torch

from ...utils.common_types import DtypeMode, QuantizationType
from ...utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
)

_MXFP4_X4_ITEMSIZE = 2  # bytes; float4_e2m1fn_x4 or its uint16 container label

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
    fp8_dtype=None,
) -> torch.Tensor:
    """
    Perform static quantization: scale, clamp, and (optionally) round-trip
    through FP8 to match what FP8 hardware does.

    Without the FP8 round-trip the result is only an approximation of static
    FP8 quantization. Values in the high-magnitude band that hardware rounds
    to the nearest FP8 grid point stay continuous here, so the reference
    drifts from FP8 hardware on real-model activation distributions where
    those values are common.

    Args:
        input_tensor (torch.Tensor): Input tensor to quantize.
        quant_scale (torch.Tensor): Quantization scale tensor.
        min_val (float): Minimum value for clamping (FP8 range).
        max_val (float): Maximum value for clamping (FP8 range).
        fp8_dtype: FP8 ``torch.dtype`` to cast through after clamping.
            When provided (e.g. ``torch.float8_e4m3fn``), the result is
            cast to FP8 and back to fp32 to apply the actual grid
            rounding. When ``None``, only clamping is applied — kept
            for callers that have not yet been migrated.

    Returns:
        torch.Tensor: Quantized tensor in fp32, clamped to
        ``[min_val, max_val]`` and (if ``fp8_dtype`` is set) round-tripped
        through FP8.
    """
    scaled_tensor = _scale_with_broadcast(input_tensor, 1 / quant_scale)
    clamped = torch.clamp(scaled_tensor, min_val, max_val)
    if fp8_dtype is not None:
        return clamped.to(fp8_dtype).to(torch.float32)
    return clamped


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
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    compact_weight_scales: bool = False,  # noqa: ARG001 — accepted for kernel signature parity (only used by MX-compact ref)
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
    # AUTO must be pre-resolved by the caller (see resolve_dtype_mode_for_torch_ref);
    # the torch ref runs on CPU and can't query hardware.
    assert dtype_mode != DtypeMode.AUTO, (  # noqa: S101
        "output_projection_cte_torch_ref requires DtypeMode.AUTO to be pre-resolved by the caller."
    )
    # Preserve the caller-supplied weight dtype BEFORE casting to float32 for compute.
    # `_get_min_max_for_dtype(weight.dtype)` below would otherwise always see float32.
    weight_orig_dtype = weight.dtype

    # Convert to float32 for computation
    attention = attention.float()
    weight = weight.float()

    if quantization_type == QuantizationType.ROW:
        # ROW: attention is [B, S, N, D]
        batch_size, seq_len, num_heads, head_dim = attention.shape
        attn_reshaped = attention.reshape(batch_size, seq_len, num_heads * head_dim)
    else:
        # All other paths: attention is [B, N, D, S]
        batch_size, num_heads, head_dim, seq_len = attention.shape
        # Reshape attention from [B, N, D, S] to [B, S, N*D]
        attn_reshaped = attention.permute(0, 3, 1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    # STATIC_MX weight is pre-shuffled for kernel; reverse to logical layout
    if quantization_type == QuantizationType.STATIC_MX:
        nd = weight.shape[0]
        if nd % 4 == 0:
            w = weight.reshape(nd // 4, weight.shape[1], 4)
            weight = w.permute(0, 2, 1).reshape(nd, weight.shape[1])

    if quantization_type == QuantizationType.STATIC:
        # STATIC clip mirrors the kernel-allocated FP8 dtype:
        # caller's concrete OCP weight or dtype_mode=OCP → 448, otherwise → 240.
        if dtype_mode == DtypeMode.OCP or str(weight_orig_dtype) in ("float8_e4m3fn", str(nl.float8_e4m3fn)):
            min_val, max_val = (-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX)
        else:
            min_val, max_val = (-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        # Pass the FP8 dtype so the reference includes the grid rounding,
        # not just the clamp. The dtype mirrors which FP8 variant the kernel
        # would allocate (e4m3fn for OCP / max=448, e4m3 otherwise). Some
        # torch versions don't expose the non-FN ``torch.float8_e4m3``
        # dtype on CPU; in that case fall back to clamp-only.
        if max_val == _FP8_E4M3FN_MAX:
            fp8_dtype = torch.float8_e4m3fn
        else:
            fp8_dtype = getattr(torch, "float8_e4m3", None)
        quantized_input = _perform_static_quant(attn_reshaped, input_scales, min_val, max_val, fp8_dtype=fp8_dtype)
        out = _perform_projection(quantized_input, weight, weight_scales, input_scales)
    elif quantization_type == QuantizationType.STATIC_MX:
        min_val, max_val = _get_min_max_for_dtype(torch.float8_e4m3fn)
        quantized_input = _perform_static_quant(
            attn_reshaped,
            input_scales,
            min_val,
            max_val,
            fp8_dtype=torch.float8_e4m3fn,
        )
        out = _perform_projection(quantized_input, weight, weight_scales, input_scales)
    elif quantization_type == QuantizationType.ROW_MX:
        # ROW_MX: MXFP4 quantize input on-device, plain matmul with FP8 weight, apply per-row weight dequant
        nd = num_heads * head_dim
        w_mx = weight.reshape(nd // 4, weight.shape[1] * 4)
        w_dummy_scale = torch.full((nd // 32, weight.shape[1]), 127, dtype=torch.float32)
        results = []
        for batch_idx in range(batch_size):
            attn_b = attention[batch_idx].numpy()
            attn_b = attn_b.reshape(nd // 4, 4, seq_len)
            attn_b = np.transpose(attn_b, (0, 2, 1)).reshape(-1, seq_len * 4)
            inp_packed, inp_scale_np = quantize_to_mx(attn_b, nl.float8_e4m3fn_x4)
            inp_unpacked = unpack_float8_e4m3fn_x4(inp_packed)
            inp_scale = torch.from_numpy(inp_scale_np).float()
            results.append(mx_matmul(inp_unpacked, w_mx, inp_scale, w_dummy_scale))
        out = torch.stack(results, dim=0)
        out = _scale_with_broadcast(out, weight_scales)
    elif quantization_type == QuantizationType.ROW:
        # ROW: dynamic per-token FP8 quantization, matmul, two-step dequant.
        # Clip must match the kernel's dtype: OCP → 448, NON_OCP → 240.
        _fp8_max = _FP8_E4M3FN_MAX if dtype_mode == DtypeMode.OCP else _FP8_E4M3_MAX
        min_val, max_val = (-_fp8_max, _fp8_max)
        # Per-token row quantize: absmax over N*D, scale, clamp
        absmax = attn_reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        dequant_scale = absmax / max_val
        quant_scale = 1.0 / dequant_scale
        quantized_input = torch.clamp(attn_reshaped * quant_scale, min_val, max_val)
        out = _scale_with_broadcast(quantized_input @ weight, weight_scales)
        out = out * dequant_scale
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
    output_dtype=None,  # noqa: ARG001 — unused (output is bf16/fp32 to match kernel)
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,  # noqa: ARG001 — accepted for framework signature parity (MX is structurally OCP)
    compact_weight_scales: bool = False,
) -> dict[str, np.ndarray]:
    """PyTorch reference for MX output projection CTE.

    Handles three weight/scale combinations:

    - Standard MX FP4 with dense block-32 scales (online or pre-quantized input).
      ``weight`` is ``float4_e2m1fn_x4 [N*D//4, H]`` and ``weight_scales`` is
      ``[N*D//32, H]``.
    - MX FP8 with compact block-128 scales (online input only).
      ``weight`` is ``float8_e4m3fn_x4 [N*D//4, H]`` and ``weight_scales`` is
      ``[N*D//128, H//128]``. Selected via ``compact_weight_scales=True``.

    Args:
        attention: Online: numpy bf16 [B, N, D, S]. Pre-quantized: numpy uint8 [B, 1, D_packed, S].
        weight: numpy float4_e2m1fn_x4 or float8_e4m3fn_x4 [N*D//4, H]. May arrive
            labeled with the same-width torch container dtype (uint16/uint32).
        bias: Optional[numpy bf16 [1, H]], bias tensor.
        quantization_type: must be QuantizationType.MX.
        input_scales: Pre-quantized only: numpy uint8 [B, D_packed//8, S]. None for online.
        weight_scales: dense [N*D//32, H] (block-32) or compact [N*D//128, H//128] (block-128).
        compact_weight_scales: True selects FP8 + compact block-128 scales.

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

    # Unpack weight + materialize the dense [N*D//32, H] block-32 scale that
    # mx_matmul consumes. Compact block-128 expands to dense via factors
    # (128/32, 128) = (4, 128).
    if compact_weight_scales:
        DS_SCALE_BLOCK = 128
        w_unpacked = unpack_float8_e4m3fn_x4(weight.reshape(-1, hidden))
        w_scale_dense = np.repeat(
            np.repeat(weight_scales, DS_SCALE_BLOCK // 32, axis=0),
            DS_SCALE_BLOCK,
            axis=1,
        )
        w_scale = torch.from_numpy(w_scale_dense).float()
    else:
        weight_reshaped = weight.reshape(-1, hidden)
        if weight_reshaped.dtype.itemsize == _MXFP4_X4_ITEMSIZE:
            w_unpacked = unpack_float4_x4(weight_reshaped)
        else:
            w_unpacked = unpack_float8_e4m3fn_x4(weight_reshaped)
        w_scale = torch.from_numpy(weight_scales.reshape(-1, hidden)).float()

    results = []
    for b in range(batch):
        if prequantized:
            # attention[b] is [1, D_packed, S], reshape to [D_packed, S]
            inp_packed = attention[b].reshape(-1, seqlen)
            inp_unpacked = unpack_float8_e4m3fn_x4(inp_packed)
            inp_scale = torch.from_numpy(input_scales[b].reshape(-1, seqlen)).float()
        else:
            # bf16 [N, D, S] -> [N*D//4, 4, S] -> [N*D//4, S*4] -> quantize
            attn_b = attention[b]
            attn_b = attn_b.reshape(n_head * d_head // 4, 4, seqlen)
            attn_b = np.transpose(attn_b, (0, 2, 1)).reshape(-1, seqlen * 4)
            inp_packed, inp_scale_np = quantize_to_mx(attn_b, nl.float8_e4m3fn_x4)
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
