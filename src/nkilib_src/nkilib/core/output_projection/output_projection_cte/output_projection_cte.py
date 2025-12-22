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

from ...utils.common_types import QuantizationType
from ...utils.kernel_helpers import get_program_sharding_info
from .output_projection_cte_float import perform_float_projection
from .output_projection_cte_parameters import (
    build_quantization_config,
    build_tiling_config,
    validate_output_projection_inputs,
)
from .output_projection_cte_quantization import perform_static_quantized_projection

# pylint: disable=too-many-locals


@nki.jit
def output_projection_cte(
    attention: nl.ndarray,
    weight: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[nl.ndarray] = None,
    weight_scales: Optional[nl.ndarray] = None,
) -> nl.ndarray:
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
        attention (nl.ndarray): [B, N, D, S], Input tensor in HBM from attention block.
        weight (nl.ndarray): [N * D, H], Weight tensor in HBM.
        bias (Optional[nl.ndarray]): [1, H], Optional bias tensor in HBM.
        quantization_type (QuantizationType): Type of quantization (NONE or STATIC for FP8).
        input_scales (Optional[nl.ndarray]): [128, 1], Input scale tensor for FP8 quantization.
        weight_scales (Optional[nl.ndarray]): [128, 1], Weight scale tensor for FP8 quantization.

    Returns:
        out (nl.ndarray): [B, S, H], Output tensor in HBM.

    Notes:
        - Product B * S must not exceed 131072.
        - Head dimension D must not exceed 128.
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
    b_size, n_size, d_size, s_size = attention.shape
    _, h_size = weight.shape

    _, n_prgs, prg_id = get_program_sharding_info()

    # Default to 1 program if not in SPMD context (e.g., simulation)
    if n_prgs is None:
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
    )

    # Configuration
    quant_config = build_quantization_config(
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
        input_data_type=attention.dtype,
        weight_data_type=weight.dtype,
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
    out = nl.ndarray((b_size, s_size, h_size), dtype=attention.dtype, buffer=nl.shared_hbm)

    weight = weight.reshape((tiling_config.n_size, tiling_config.d_size, tiling_config.h_size))

    if tiling_config.group_size > 1 or quant_config.use_double_row:
        attention = attention.reshape(
            (
                tiling_config.b_size,
                tiling_config.n_size,
                tiling_config.d_size,
                tiling_config.s_size,
            )
        )

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
    else:
        perform_float_projection(
            attention=attention,
            weight=weight,
            bias=bias,
            out=out,
            tiling_config=tiling_config,
            prg_id=prg_id,
        )

    return out
