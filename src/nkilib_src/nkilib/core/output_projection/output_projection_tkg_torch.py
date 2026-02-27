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

import torch

from ..utils.common_types import QuantizationType

_FP8_E4M3_MAX = 240.0


def output_projection_tkg_torch_ref(
    attention: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
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
        quantization_type: Type of quantization (NONE, STATIC).
        weight_scale: [128, 1] weight quantization scale (for STATIC).
        input_scale: [128, 1] input quantization scale (for STATIC).
        TRANSPOSE_OUT: Whether to produce transposed output layout.
        OUT_IN_SB: Whether output is in SBUF (does not affect math).

    Returns:
        dict with "out" tensor. Shape depends on TRANSPOSE_OUT:
            False: [B*S, H]
            True: [128, lnc, H//(lnc*128), B*S] where lnc is inferred from H.
    """
    D, B, N, S = attention.shape
    H = weight.shape[1]

    attention = attention.float()
    weight = weight.float()

    # Reshape from [D, B, N, S] to [B*S, N*D]
    attn = attention.permute(1, 3, 2, 0).reshape(B * S, N * D)

    if quantization_type == QuantizationType.STATIC:
        ws = weight_scale[0, 0].float()
        ins = input_scale[0, 0].float()
        attn = torch.clamp(attn / ins, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)

    out = attn @ weight

    if quantization_type == QuantizationType.STATIC:
        out = out * (ws * ins)

    if bias is not None:
        out = out + bias.float()

    if TRANSPOSE_OUT:
        # Infer lnc from H: try lnc=2 first, fall back to lnc=1
        H0 = 128
        lnc = 2 if H % (2 * H0) == 0 else 1
        out = out.reshape(B * S, lnc, H0, H // (lnc * H0)).permute(2, 1, 3, 0)

    return {"out": out}
