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


from typing import Optional

import nki
import nki.language as nl

# common utils
from ..utils.common_types import ActFnType, NormType, QuantizationType

# MLP utils
from .mlp_cte.mlp_cte import mlp_cte
from .mlp_parameters import (
    MLPParameters,
    is_mlp_tkg,
    mlpp_store_fused_add,
    validate_mlp_arguments,
)
from .mlp_tkg.mlp_tkg import mlp_tkg
from .mlp_tkg.mlp_tkg_mx import mlp_tkg_mx

#
# **********************
# MLP Kernel ISA
# **********************
#


@nki.jit
def mlp(
    hidden_tensor: nl.ndarray,
    gate_proj_weights_tensor: nl.ndarray,
    up_proj_weights_tensor: nl.ndarray,
    down_proj_weights_tensor: nl.ndarray,
    normalization_weights_tensor: Optional[nl.ndarray] = None,
    gate_proj_bias_tensor: Optional[nl.ndarray] = None,
    up_proj_bias_tensor: Optional[nl.ndarray] = None,
    down_proj_bias_tensor: Optional[nl.ndarray] = None,
    normalization_bias_tensor: Optional[nl.ndarray] = None,
    fused_add_tensor: Optional[nl.ndarray] = None,
    store_fused_add_result: bool = False,
    activation_fn: ActFnType = ActFnType.SiLU,
    normalization_type: NormType = NormType.NO_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale: Optional[nl.ndarray] = None,
    up_w_scale: Optional[nl.ndarray] = None,
    down_w_scale: Optional[nl.ndarray] = None,
    gate_up_in_scale: Optional[nl.ndarray] = None,
    down_in_scale: Optional[nl.ndarray] = None,
    quant_clipping_bound: float = 0.0,
    output_dtype=None,
    store_output_in_sbuf: bool = False,
    eps: float = 1e-6,
    skip_gate_proj: bool = False,
    use_tkg_gate_up_proj_column_tiling: bool = True,
    use_tkg_down_proj_column_tiling: bool = True,
    use_tkg_down_proj_optimized_layout: bool = False,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    force_cte_mode: bool = False,
) -> list[nl.ndarray]:
    """
    MLP (Multi-Layer Perceptron) Kernel implementation.

    Performs the standard MLP computation with support for both context encoding (CTE) and
    token generation (TKG) modes. Automatically selects the appropriate implementation based
    on input dimensions and supports various optimizations including FP8 and MXFP quantization.

    Supported input data types: bfloat16, float16, float32.
    Supported quantization: FP8 (tensor-wise/row-wise), MXFP4, MXFP8.

    Computation flow:
        if fused_add is applied:
            hidden_states = hidden_states + fused_add_tensor

        if normalization is applied:
            hidden_states = normalization_type(hidden_states)

        gate_proj_out = hidden_states @ gate_proj_weights_tensor
        act_gate_proj = activation_fn(gate_proj_out)

        up_proj_out = hidden_states @ up_proj_weights_tensor
        hidden_states = Multiply(act_gate_proj, up_proj_out)

        down_proj_out = hidden_states @ down_proj_weights_tensor
        output = down_proj_out

    Args:
        hidden_tensor (nl.ndarray): Input hidden states tensor with shape [B, S, H] or SBUF layout.
        gate_proj_weights_tensor (nl.ndarray): Gate projection weight matrix with shape [H, I].
        up_proj_weights_tensor (nl.ndarray): Up projection weight matrix with shape [H, I].
        down_proj_weights_tensor (nl.ndarray, optional): Down projection weight matrix with shape [I, H].
        normalization_weights_tensor (nl.ndarray, optional): Normalization weights with shape [1, H].
        gate_proj_bias_tensor (nl.ndarray, optional): Bias tensor for gate projection with shape [1, I].
        up_proj_bias_tensor (nl.ndarray, optional): Bias tensor for up projection with shape [1, I].
        down_proj_bias_tensor (nl.ndarray, optional): Bias tensor for down projection with shape [1, H].
        normalization_bias_tensor (nl.ndarray, optional): Bias tensor for normalization with shape [1, H].
            Only applicable for layer normalization.
        fused_add_tensor (nl.ndarray, optional): tensor to fuse for the residual connection..
        store_fused_add_result (bool): If True, stores the fused_add output to HBM, and
            the kernel returns both the fused_add output and the MLP output.
            (default: False)
        activation_fn (ActFnType): Activation function type.
        normalization_type (NormType): Type of normalization.
        quantization_type (QuantizationType): Quantization type to use (default: QuantizationType.NONE).
            Supported values:
            - QuantizationType.NONE: No quantization
            - QuantizationType.STATIC: FP8 tensor-wise quantization with 2x perf mode (CTE and TKG)
            - QuantizationType.STATIC_MX: FP8 tensor-wise quantization with 4x perf mode (CTE)
                - If using STATIC_MX in CTE mode, the up and gate weights should be swizzled as follows:
                    H, I = w.shape
                    w.reshape(
                        (2, ceil(H / 512), 128, 2, ceil(I / 512), 128, 4)
                    ).transpose(2, 1, 4, 6, 5, 0, 3).reshape((H, I))
                - It expects the down weights to be swizzled as follows:
                    I, H = w.shape
                    w.reshape((ceil(I / 512), 128, 4, H)).transpose(1, 0, 3, 2).reshape((I, H))
            - QuantizationType.ROW: FP8 row-wise quantization (CTE and TKG)
            - QuantizationType.MX: MXFP quantization (MXFP4/MXFP8, TKG only)
        gate_w_scale (nl.ndarray, optional): Dequantization scales for gate weights.
            - FP8: Shape [128, I] for row-wise, [128, 1] for tensor-wise quantization
            - MXFP (TKG only): Scale factors for MXFP quantized weights
        up_w_scale (nl.ndarray, optional): Dequantization scales for up weights.
            - FP8: Shape [128, I] for row-wise, [128, 1] for tensor-wise quantization
            - MXFP (TKG only): Scale factors for MXFP quantized weights
        down_w_scale (nl.ndarray, optional): Dequantization scales for down weights.
            - FP8: Shape [128, I] for row-wise, [128, 1] for tensor-wise quantization
            - MXFP (TKG only): Scale factors for MXFP quantized weights
        gate_up_in_scale (nl.ndarray, optional): FP8 dequantization scales for gate and up input.
            Used for tensor-wise quantization with shape [128, 1]. Defaults to None.
        down_in_scale (nl.ndarray, optional): FP8 dequantization scales for down input.
            Used for tensor-wise quantization with shape [128, 1]. Defaults to None.
        quant_clipping_bound (float): Clipping boundary for context encoding FP8 row quantization (default: 0.0)
        output_dtype: Output tensor data type. Defaults to None; if None, the hidden tensor’s dtype is used.
        store_output_in_sbuf (bool): If True, stores the output in SBUF instead of HBM,
            allowing the next layer to read it directly without an additional load operation.
            This option is only available in TKG mode where output tensor is small enough to fit in SBUF.
            (default: False)
        eps (float): Epsilon value for numerical stability.
        skip_gate_proj (bool): Skip gate projection
        use_tkg_gate_up_proj_column_tiling (bool): If True, uses column tiling for the gate
            and up projection in TKG mode. (default: True)
        use_tkg_down_proj_column_tiling (bool): If True, uses column tiling for the down projection in TKG mode.
            (default: True)
        use_tkg_down_proj_optimized_layout (bool): If True, the standard down_weight tensor (shape [I, H])
            is reinterpreted as [I, lnc, 128, H // (128 * lnc)], then transposed to
            [I, lnc, H // (128 * lnc), 128]. This layout provides unit-stride weight loading,
            reducing the matrix multiplication initiation interval. Only applied when
            `use_tkg_down_proj_column_tiling` is False. (default: False)
        gate_clamp_upper_limit (float): upper bound value to clamp on gate projection results, does not perform clamping if the value is set to None
        gate_clamp_lower_limit (float): lower bound value to clamp on gate projection results, does not perform clamping if the value is set to None
        up_clamp_upper_limit (float): upper bound value to clamp on up projection results, does not perform clamping if the value is set to None
        up_clamp_lower_limit (float): lower bound value to clamp on up projection results, does not perform clamping if the value is set to None
        force_cte_mode (bool): If True, forces the use of CTE mode. (default: False)

    Returns:
        list:
            The MLP output tensor(s):
            - HBM output: Tensor with shape [B, S, H].
            - SBUF output: Shape depends on the mode setting.
                - CTE : Not applicable
                - TKG when `use_tkg_down_proj_column_tiling` is True = [BxS, H]
                - TKG when `use_tkg_down_proj_column_tiling` is False = [128(p_max), H/128, BxS]
            - If `store_fused_add_result` is True, returns a list containing both the output
            and the stored fused output.

    Notes:
        Automatically dispatches to either CTE or TKG implementation based on batch size and
        sequence length. Token generation mode (TKG) is used for small batch/sequence dimensions
        (B×S ≤ 96), while context encoding (CTE) handles larger inputs.

        TKG mode supports:
        - FP8 quantization (tensor-wise and row-wise)
        - MXFP quantization (MXFP4 and MXFP8) - TKG only
        - Column tiling optimizations
        - Tensor layout optimization for down projection
        - Input in SBUF for kernel fusion

        CTE mode supports:
        - FP8 quantization (tensor-wise and row-wise)
        - Standard matrix multiplication layouts
    """

    # Build MLP parameter object with all relevant weights, biases, and config
    mlp_params = MLPParameters(
        hidden_tensor=hidden_tensor,
        gate_proj_weights_tensor=gate_proj_weights_tensor,
        up_proj_weights_tensor=up_proj_weights_tensor,
        down_proj_weights_tensor=down_proj_weights_tensor,
        normalization_weights_tensor=normalization_weights_tensor,
        fused_add_tensor=fused_add_tensor,
        store_fused_add_result=store_fused_add_result,
        activation_fn=activation_fn,
        normalization_type=normalization_type,
        gate_proj_bias_tensor=gate_proj_bias_tensor,
        up_proj_bias_tensor=up_proj_bias_tensor,
        down_proj_bias_tensor=down_proj_bias_tensor,
        normalization_bias_tensor=normalization_bias_tensor,
        quantization_type=quantization_type,
        gate_w_scale=gate_w_scale,
        up_w_scale=up_w_scale,
        down_w_scale=down_w_scale,
        gate_up_in_scale=gate_up_in_scale,
        down_in_scale=down_in_scale,
        quant_clipping_bound=quant_clipping_bound,
        output_dtype=output_dtype,
        store_output_in_sbuf=store_output_in_sbuf,
        eps=eps,
        skip_gate_proj=skip_gate_proj,
        use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        force_cte_mode=force_cte_mode,
    )

    # Validate MLP arguments
    validate_mlp_arguments(mlp_params)

    # Allocate output tensor in shared HBM memory
    output_tensors = []
    out = None
    if not store_output_in_sbuf:
        out = nl.ndarray(
            (mlp_params.batch_size, mlp_params.sequence_len, mlp_params.hidden_size),
            dtype=mlp_params.output_dtype,
            buffer=nl.shared_hbm,
            name="output_tensor_hbm",
        )
    output_tensors.append(out)

    # Optionally allocate an additional tensor to store fused addition results
    fused_add_out = None
    if mlpp_store_fused_add(mlp_params):
        fused_add_out = nl.ndarray(
            (mlp_params.batch_size, mlp_params.sequence_len, mlp_params.hidden_size),
            dtype=mlp_params.output_dtype,
            buffer=nl.shared_hbm,
            name="output_stored_add_tensor_hbm",
        )
        output_tensors.append(fused_add_out)

    # Determine if MLP should be invoked in token-generation (TKG) mode or context encoding (CTE) mode
    # If batch size × sequence length <= TKG_BS_SEQLEN_THRESHOLD(currently at 96), the kernel runs in TKG mode.
    # TODO: update TKG_BS_SEQLEN_THRESHOLD to 128
    if is_mlp_tkg(mlp_params):
        if mlp_params.quant_params.is_quant_mx():
            return mlp_tkg_mx(mlp_params, out, fused_add_out)
        else:
            return mlp_tkg(mlp_params, out, fused_add_out)
    else:
        mlp_cte(mlp_params, out, fused_add_out)
        # Return all output tensors (mlp output and optionally fused add)
        return output_tensors
