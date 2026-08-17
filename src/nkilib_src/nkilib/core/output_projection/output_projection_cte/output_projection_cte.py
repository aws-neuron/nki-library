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

"""Output projection CTE kernel for context encoding scenarios with LNC sharding support."""

from typing import Optional

import nki
import nki.language as nl

from ...utils.common_types import DtypeMode, OProjAttentionLayout, QuantizationType
from ...utils.kernel_helpers import get_program_sharding_info
from .output_projection_cte_float import perform_float_projection
from .output_projection_cte_parameters import (
    build_quantization_config,
    build_tiling_config,
    resolve_attention_layout,
    unpack_attention_shape,
    validate_output_projection_inputs,
)
from .output_projection_cte_quantization import (
    perform_mx_quantized_projection,
    perform_row_mx_quantized_projection,
    perform_row_quantized_projection,
    perform_static_mx_quantized_projection,
    perform_static_quantized_projection,
)

# pylint: disable=too-many-locals


@nki.jit
def output_projection_cte(
    attention: nl.NkiTensor,
    weight: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[nl.NkiTensor] = None,
    weight_scales: Optional[nl.NkiTensor] = None,
    output_dtype: Optional[type] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    compact_weight_scales: bool = False,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> nl.NkiTensor:
    """
    Output projection kernel optimized for Context Encoding (CTE/Prefill) scenarios.

    Computes out = attention @ weight + bias, typically used to project output scores
    after attention blocks in transformer models. Optimized for large sequence lengths
    (S >= 512). Using this kernel with S < 512 may result in degraded performance.

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        attention (nl.NkiTensor): [B, N, D, S], Input tensor in HBM from attention block.
        weight (nl.NkiTensor): [N * D, H], Weight tensor in HBM.
        bias (Optional[nl.NkiTensor]): [1, H], Optional bias tensor in HBM.
        quantization_type (QuantizationType): Type of quantization (NONE, STATIC for FP8, or MX).
        input_scales (Optional[nl.NkiTensor]): [128, 1], Input scale tensor for FP8 quantization.
        weight_scales (Optional[nl.NkiTensor]): [128, 1], Weight scale tensor for FP8 quantization.
        output_dtype (Optional[type]): Output data type. Defaults to attention.dtype for non-MX,
            or nl.bfloat16 for MX quantization. Can be set to nl.float16 for higher precision.
        dtype_mode (DtypeMode): Quantization dtype policy for STATIC/ROW
            attention and weight-tile allocation.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
            - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, else ``nl.float8_e4m3``.
        compact_weight_scales (bool): When True (and quantization_type is MX),
            ``weight_scales`` is interpreted as the block-128 compact
            layout ``[N*D // 128, H // 128]`` uint8, with one scale per 128x128
            weight block. The kernel expands the compact scales to the hardware
            MX layout on-device. Defaults to False (block-32 dense layout).
        attention_layout (Optional[OProjAttentionLayout]): Dimension order of ``attention``.
            When declared it is taken as-is, asserted to be supported by ``quantization_type``.
            Every layout is 4D, so a tag that does not describe the tensor passed in
            permutes N, D, and S rather than failing; the caller owns getting this right.
            - ``None`` (default): the layout ``quantization_type`` implies — [B, S, N, D]
              for ``QuantizationType.ROW``, [B, N, D, S] for every other type.
            - ``OProjAttentionLayout.BNdS``: ``attention`` is [B, N, D, S]. Supported by
              every ``quantization_type`` except ROW.
            - ``OProjAttentionLayout.BSNd``: ``attention`` is [B, S, N, D]. Supported by
              ``QuantizationType.NONE`` and ``QuantizationType.ROW``.
            - ``OProjAttentionLayout.BNSd``: ``attention`` is [B, N, S, D] — the attention
              CTE output with heads folded into batch and ``tp_out=False``. Supported by
              ``QuantizationType.NONE``.

    Returns:
        out (nl.NkiTensor): [B, S, H], Output tensor in HBM.

    Notes:
        - Product B * S must not exceed 131072.
        - Head dimension D > 128 is supported by folding D back into N (D must have a divisor that brings it to <= 128).
        - Hidden dimension H must not exceed 20705 (not fully tested beyond).
        - Number of heads N must not exceed 17 (not fully tested beyond).
        - Hidden dimension H must be divisible by LNC (1 or 2).
        - FP8 static quantization requires N or D to be even for double row matmul.

    Pseudocode:
        out = zeros([B, S, H])
        h_sharded = H // LNC
        for h_block in range(num_h_blocks):
            w_sbuf = load_weights(weight, h_block)
            bias_sbuf = load_bias(bias, h_block) if bias else None
            for b in range(B):
                for s_block in range(num_s_blocks):
                    attn_sbuf = load_attention(attention, b, s_block)
                    for s_subtile in range(s_subtiles):
                        for h_subtile in range(h_subtiles):
                            res_psum = zeros()
                            for n in range(N):
                                res_psum += attn_sbuf[n] @ w_sbuf[n]
                            out[b, s_block, h_block] = res_psum + bias_sbuf
        return out
    """
    attention_layout = resolve_attention_layout(attention_layout, quantization_type)
    b_size, n_size, d_size, s_size = unpack_attention_shape(attention, attention_layout)
    _, h_size = weight.shape

    _, n_prgs, prg_id = get_program_sharding_info()

    # Default to 1 program if not in SPMD context (e.g., simulation)
    if n_prgs == None:
        n_prgs = 1
        prg_id = 0

    # Validation
    validate_output_projection_inputs(
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        n_prgs=n_prgs,
        attention_dtype=attention.dtype,
        weight_dtype=weight.dtype,
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
        compact_weight_scales=compact_weight_scales,
    )

    # Configuration
    quant_config = build_quantization_config(
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
        input_data_type=attention.dtype,
        weight_data_type=weight.dtype,
        dtype_mode=dtype_mode,
        compact_weight_scales=compact_weight_scales,
    )

    tiling_config = build_tiling_config(
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        n_prgs=n_prgs,
        quant_config=quant_config,
        weight_dtype=weight.dtype,
    )

    # Execution
    if output_dtype != None:
        out_dtype = output_dtype
    elif quantization_type in (QuantizationType.MX, QuantizationType.ROW_MX):
        out_dtype = nl.bfloat16
    else:
        out_dtype = attention.dtype
    out = nl.ndarray((b_size, s_size, h_size), dtype=out_dtype, buffer=nl.shared_hbm)

    if quant_config.is_enabled and quantization_type == QuantizationType.STATIC:
        perform_static_quantized_projection(
            attention_hbm=attention,
            weight_hbm=weight,
            output_hbm=out,
            bias_hbm=bias,
            input_scale_hbm=input_scales,
            weight_scale_hbm=weight_scales,
            prg_id=prg_id,
            cfg=tiling_config,
            quant_config=quant_config,
        )
    elif quant_config.is_enabled and quantization_type == QuantizationType.STATIC_MX:
        perform_static_mx_quantized_projection(
            attention_hbm=attention,
            weight_hbm=weight,
            output_hbm=out,
            bias_hbm=bias,
            input_scale_hbm=input_scales,
            weight_scale_hbm=weight_scales,
            prg_id=prg_id,
            cfg=tiling_config,
            quant_config=quant_config,
        )
    elif quant_config.is_enabled and quantization_type == QuantizationType.ROW_MX:
        perform_row_mx_quantized_projection(
            attention_hbm=attention,
            weight_hbm=weight,
            output_hbm=out,
            bias_hbm=bias,
            input_scale_hbm=input_scales,
            weight_scale_hbm=weight_scales,
            prg_id=prg_id,
            cfg=tiling_config,
            quant_config=quant_config,
        )
    elif quant_config.is_enabled and quantization_type == QuantizationType.MX:
        perform_mx_quantized_projection(
            attention_hbm=attention,
            weight_hbm=weight,
            output_hbm=out,
            bias_hbm=bias,
            weight_scale_hbm=weight_scales,
            input_scale_hbm=input_scales,
            prg_id=prg_id,
            cfg=tiling_config,
            quant_config=quant_config,
        )
    elif quant_config.is_enabled and quantization_type == QuantizationType.ROW:
        perform_row_quantized_projection(
            attention_hbm=attention,
            weight_hbm=weight,
            output_hbm=out,
            bias_hbm=bias,
            weight_scale_hbm=weight_scales,
            prg_id=prg_id,
            cfg=tiling_config,
            quant_config=quant_config,
        )
    else:
        perform_float_projection(
            attention_hbm=attention,
            weight_hbm=weight,
            bias_hbm=bias,
            out_hbm=out,
            cfg=tiling_config,
            prg_id=prg_id,
            attention_layout=attention_layout,
        )

    return out
