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

"""PyTorch reference implementation for output projection TKG kernel."""

from typing import Optional

import nki.language as nl
import numpy as np
import torch

from ..utils.common_types import DtypeMode, QuantizationType
from ..utils.kernel_helpers import get_max_positive_value_for_dtype
from ..utils.mx_torch_common import mx_matmul, quantize_to_mx, unpack_float8_e4m3fn_x4

_FP8_E4M3_MAX = get_max_positive_value_for_dtype(nl.float8_e4m3)
_FP8_E4M3FN_MAX = get_max_positive_value_for_dtype(nl.float8_e4m3fn)

_Q_WIDTH = 4


def output_projection_tkg_torch_ref(
    attention: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    sbm=None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> dict:
    """PyTorch reference implementation of output projection for TKG (token generation).

    Computes: out = attention @ weight + bias

    Dimensions:
        D: Head dimension
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension

    Args:
        attention: [D, B, N, S] input tensor from attention block.
        weight: [N*D, H] weight tensor.
        bias: [1, H] optional bias tensor.
        quantization_type: Type of quantization (NONE, STATIC, ROW).
        weight_scale: [128, 1] weight quantization scale (for STATIC),
                      [128, H] weight quantization scale (for ROW).
        input_scale: [128, 1] input quantization scale (for STATIC).
        TRANSPOSE_OUT: Whether to produce transposed output layout.
        OUT_IN_SB: Whether output is in SBUF (does not affect math).

    Returns:
        dict with "out" tensor. Shape depends on TRANSPOSE_OUT:
            False: [B*S, H]
            True: [128, lnc, H//(lnc*128), B*S] where lnc is inferred from H.
    """
    # AUTO must be pre-resolved by the caller (see resolve_dtype_mode_for_torch_ref);
    # the torch ref runs on CPU and can't query hardware.
    assert dtype_mode != DtypeMode.AUTO, (  # noqa: S101
        "output_projection_tkg_torch_ref requires DtypeMode.AUTO to be pre-resolved by the caller."
    )

    if quantization_type == QuantizationType.MX:
        return output_projection_tkg_mx_torch_ref(
            attention=attention,
            weight=weight,
            bias=bias,
            quantization_type=quantization_type,
            weight_scale=weight_scale,
            input_scale=input_scale,
            TRANSPOSE_OUT=TRANSPOSE_OUT,
            OUT_IN_SB=OUT_IN_SB,
            sbm=sbm,
        )

    is_static_mx = quantization_type == QuantizationType.STATIC_MX
    is_static = quantization_type == QuantizationType.STATIC

    D, B, N, S = attention.shape

    attention = attention.float()
    # Reshape from [D, B, N, S] to [B*S, N*D]
    attn = attention.permute(1, 3, 2, 0).reshape(B * S, N * D)

    # STATIC_MX: weights are unpacked fp8 [N*D//4, H, 4] -> float32 [N*D, H]
    if is_static_mx:
        weights_np = weight if isinstance(weight, np.ndarray) else weight.numpy()
        ND_packed = weights_np.shape[0]
        H = weights_np.shape[1]
        weight = torch.from_numpy(weights_np.transpose(0, 2, 1).reshape(ND_packed * _Q_WIDTH, H).astype(np.float32))
    else:
        H = weight.shape[1]
        weight = weight.float()

    if is_static or is_static_mx:
        weight_scale_value = (
            weight_scale[0, 0].float()
            if isinstance(weight_scale, torch.Tensor)
            else float(np.asarray(weight_scale).flat[0])
        )
        input_scale_value = (
            input_scale[0, 0].float()
            if isinstance(input_scale, torch.Tensor)
            else float(np.asarray(input_scale).flat[0])
        )
        # STATIC_MX is always OCP; STATIC follows dtype_mode (OCP → 448, NON_OCP → 240).
        if is_static_mx:
            clip_value = _FP8_E4M3FN_MAX
        elif dtype_mode == DtypeMode.OCP:
            clip_value = _FP8_E4M3FN_MAX
        else:
            clip_value = _FP8_E4M3_MAX
        attn = torch.clamp(attn / input_scale_value, -clip_value, clip_value)
    elif quantization_type == QuantizationType.ROW:
        weight_scale_value = weight_scale[0, :].float()

    out = attn @ weight

    if is_static or is_static_mx:
        combined_scale = weight_scale_value * input_scale_value
        out = out * combined_scale
    elif quantization_type == QuantizationType.ROW:
        out = out * weight_scale_value

    if bias is not None:
        out = out + bias.float()

    if TRANSPOSE_OUT:
        # Infer lnc from H: try lnc=2 first, fall back to lnc=1
        H0 = 128
        lnc = 2 if H % (2 * H0) == 0 else 1
        out = out.reshape(B * S, lnc, H0, H // (lnc * H0)).permute(2, 1, 3, 0)

    return {"out": out}


def output_projection_tkg_mx_torch_ref(
    attention: torch.Tensor,
    weight: np.ndarray,
    bias: Optional[torch.Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.MX,
    weight_scale: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    sbm=None,
) -> dict:
    """PyTorch reference implementation of output projection for TKG MXFP8 (token generation).

    Computes: out = attention (MXFP8) @ weight (MXFP8) + bias (with weights pre-quantized)
              attention comes (BF16) and will be quantized online.

    Dimensions:
        D: Head dimension
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension

    Args:
        attention: [D, B, N, S] input tensor from attention block (BF16, quantized online), dtype=nl.bfloat16
        weight: [N*D // 4, H, 4] weight tensor (pre-quantized, unpacked fp8), dtype=nl.float8_e4m3fn
                NOTE: weight is np.ndarray dtype because of MX dtype.
        bias: [1, H] optional bias tensor.
        quantization_type: Type of quantization (must be QuantizationType.MX)
        weight_scale: [N*D // 32, H] weight quantization scale, dtype nl.uint8
        input_scale: MX version does not accept input scales.
        TRANSPOSE_OUT: Whether to produce transposed output layout.
        OUT_IN_SB: Whether output is in SBUF (does not affect math).

    Returns:
        dict with "out" tensor. Shape depends on TRANSPOSE_OUT:
            False: [B*S, H]
            True: [128, lnc, H//(lnc*128), B*S] where lnc is inferred from H.
    """

    if isinstance(attention, torch.Tensor):
        attention = attention.numpy()
    if isinstance(weight, torch.Tensor):
        weight = weight.numpy()
    if isinstance(weight_scale, torch.Tensor):
        weight_scale = weight_scale.numpy()

    D, B, N, S = attention.shape
    BxS = B * S
    N_D = N * D
    # weight is [N*D//4, H, 4] unpacked fp8
    ND_packed = weight.shape[0]
    H = weight.shape[1]

    # Goal: We need multiply attn_qtz @ weight_qtz = [N*D//4, BxS] @ [N*D//4, H]
    ####################### Quantize attention ########################################
    # Permute attention: [D, B, N, S] -> [N*D, B*S] -> [N*D//4, B*S*4] for MX quantization
    attn_permuted = np.transpose(attention, (2, 0, 1, 3)).reshape(N_D, BxS)
    attn_for_quant = attn_permuted.reshape(N_D // 4, 4, BxS).transpose(0, 2, 1).reshape(N_D // 4, BxS * 4)
    # Quantize attention online
    attn_qtz, attn_scale = quantize_to_mx(attn_for_quant, nl.float8_e4m3fn_x4)

    ####################  Convert to torch tensors needed for mx_matmul ############
    # Unpack quantized attention from x4; weights are already unpacked fp8
    attn_unpacked = unpack_float8_e4m3fn_x4(attn_qtz)
    weight_unpacked = torch.from_numpy(weight.reshape(ND_packed, H * _Q_WIDTH).astype(np.float32))
    # Scales are numpy uint8 tensors, convert them to torch
    attn_scale_torch = torch.from_numpy(attn_scale).float()
    weight_scale_torch = torch.from_numpy(weight_scale).float()

    #################### MX matmul: [N*D, B*S] @ [N*D, H] -> [B*S, H] #################
    out = mx_matmul(attn_unpacked, weight_unpacked, attn_scale_torch, weight_scale_torch)

    if bias is not None:
        if isinstance(bias, np.ndarray):
            bias = torch.from_numpy(bias.astype(np.float32))
        out = out + bias.float()

    if TRANSPOSE_OUT:
        H0 = 128
        lnc = 2 if H % (2 * H0) == 0 else 1
        out = out.reshape(BxS, lnc, H0, H // (lnc * H0)).permute(2, 1, 3, 0)

    return {"out": out}
