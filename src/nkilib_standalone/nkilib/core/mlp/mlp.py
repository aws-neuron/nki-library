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
from ..utils.common_types import ActFnType, NormType

# MLP utils
from .mlp_cte.mlp_cte import mlp_cte_invoke_kernel
from .mlp_parameters import (
    MLPParameters,
    is_mlp_tkg,
    mlpp_store_fused_add,
    validate_mlp_arguments,
)
from .mlp_tkg.mlp_tkg import mlp_tkg_invoke_kernel

#
# **********************
# MLP Kernel ISA
# **********************
#


@nki.jit
def mlp_kernel(
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
    output_dtype=None,
    store_output_in_sbuf: bool = False,
    eps: float = 1e-6,
    use_tkg_gate_up_proj_column_tiling: bool = True,
    use_tkg_down_proj_column_tiling: bool = True,
    use_tkg_down_proj_optimized_layout: bool = False,
    force_cte_mode: bool = False,
) -> list[nl.ndarray]:
    """
    MLP(Multi-Layer Perceptron) Kernel implementation.

    Performs the standard MLP computation with support for both context encoding (CTE) and
    token generation (TKG) modes. Automatically selects the appropriate implementation based
    on input dimensions and supports various optimizations

    Supported input data types: bfloat16, float16, float32.

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
        output_dtype: Output tensor data type. Defaults to None; if None, the hidden tensor’s dtype is used.
        store_output_in_sbuf (bool): If True, stores the output in SBUF instead of HBM,
            allowing the next layer to read it directly without an additional load operation.
            This option is only available in TKG mode where output tensor is small enough to fit in SBUF.
            (default: False)
        eps (float): Epsilon value for numerical stability.
        use_tkg_gate_up_proj_column_tiling (bool): If True, uses column tiling for the gate
            and up projection in TKG mode. (default: True)
        use_tkg_down_proj_column_tiling (bool): If True, uses column tiling for the down projection in TKG mode.
            (default: True)
        use_tkg_down_proj_optimized_layout (bool): If True, the standard down_weight tensor (shape [I, H])
            is reinterpreted as [I, lnc, 128, H // (128 * lnc)], then transposed to
            [I, lnc, H // (128 * lnc), 128]. This layout provides unit-stride weight loading,
            reducing the matrix multiplication initiation interval. Only applied when
            `use_tkg_down_proj_column_tiling` is False. (default: False)
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
        sequence length. Token generation mode (TKG) is used for small batch/sequence dimensions,
        while context encoding (CTE) handles larger inputs. Column tiling and tensor layout
        optimization (`use_down_proj_layout_optimization`) are valid only in TKG mode.
    """

    # If output_dtype is not provided, use the same dtype as the hidden tensor
    if output_dtype == None:
        output_dtype = hidden_tensor.dtype

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
        output_dtype=output_dtype,
        store_output_in_sbuf=store_output_in_sbuf,
        eps=eps,
        use_tkg_gate_up_proj_column_tiling=use_tkg_gate_up_proj_column_tiling,
        use_tkg_down_proj_column_tiling=use_tkg_down_proj_column_tiling,
        use_tkg_down_proj_optimized_layout=use_tkg_down_proj_optimized_layout,
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
    if is_mlp_tkg(mlp_params) and not force_cte_mode:
        return mlp_tkg_invoke_kernel(mlp_params, out, fused_add_out)
    else:
        mlp_cte_invoke_kernel(mlp_params, out, fused_add_out)
        # Return all output tensors (mlp output and optionally fused add)
        return output_tensors
