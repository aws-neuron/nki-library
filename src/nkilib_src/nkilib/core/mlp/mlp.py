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
from ..utils.allocator import BufferManager, SbufManager
from ..utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    MLPGateUpWeightLayout,
    NormType,
    QuantizationType,
)
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import get_logger

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
    hidden_tensor: nl.NkiTensor,
    gate_proj_weights_tensor: nl.NkiTensor,
    up_proj_weights_tensor: nl.NkiTensor,
    down_proj_weights_tensor: nl.NkiTensor,
    normalization_weights_tensor: Optional[nl.NkiTensor] = None,
    gate_proj_bias_tensor: Optional[nl.NkiTensor] = None,
    up_proj_bias_tensor: Optional[nl.NkiTensor] = None,
    down_proj_bias_tensor: Optional[nl.NkiTensor] = None,
    normalization_bias_tensor: Optional[nl.NkiTensor] = None,
    fused_add_tensor: Optional[nl.NkiTensor] = None,
    store_fused_add_result: bool = False,
    activation_fn: ActFnType = ActFnType.SiLU,
    normalization_type: NormType = NormType.NO_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale: Optional[nl.NkiTensor] = None,
    up_w_scale: Optional[nl.NkiTensor] = None,
    down_w_scale: Optional[nl.NkiTensor] = None,
    gate_up_in_scale: Optional[nl.NkiTensor] = None,
    down_in_scale: Optional[nl.NkiTensor] = None,
    quant_clipping_bound: float = 0.0,
    output_dtype=None,
    store_output_in_sbuf: bool = False,
    eps: float = 1e-6,
    skip_gate_proj: bool = False,
    use_tkg_gate_up_proj_column_tiling: bool = True,
    use_tkg_down_proj_column_tiling: bool = True,
    use_tkg_down_proj_optimized_layout: bool = False,
    use_contiguous_x4_gate_up: bool = False,  # TODO: Remove once all callers migrate to gate_up_w_layout
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    force_cte_mode: bool = False,
    mode: ComputationMode = ComputationMode.AUTO,
    sbm: Optional[BufferManager] = None,
    mx_dummy_scale_hbm: Optional[nl.NkiTensor] = None,
    transposed_in: bool = False,
    transposed_out: bool = False,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    gate_up_w_layout: MLPGateUpWeightLayout = MLPGateUpWeightLayout.CONTIGUOUS,
) -> list[nl.NkiTensor]:
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
        hidden_tensor (nl.NkiTensor): Input hidden states tensor with shape [B, S, H], SBUF layout,
            or transposed HBM layout [H0, n_prgs, H1_shard, BxS] when transposed_in=True.
        gate_proj_weights_tensor (nl.NkiTensor): Gate projection weight matrix with shape [H, I].
        up_proj_weights_tensor (nl.NkiTensor): Up projection weight matrix with shape [H, I].
        down_proj_weights_tensor (nl.NkiTensor, optional): Down projection weight matrix with shape [I, H].
        normalization_weights_tensor (nl.NkiTensor, optional): Normalization weights with shape [1, H].
        gate_proj_bias_tensor (nl.NkiTensor, optional): Bias tensor for gate projection with shape [1, I].
        up_proj_bias_tensor (nl.NkiTensor, optional): Bias tensor for up projection with shape [1, I].
        down_proj_bias_tensor (nl.NkiTensor, optional): Bias tensor for down projection with shape [1, H].
        normalization_bias_tensor (nl.NkiTensor, optional): Bias tensor for normalization with shape [1, H].
            Only applicable for layer normalization.
        fused_add_tensor (nl.NkiTensor, optional): tensor to fuse for the residual connection..
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
                - If using STATIC_MX in CTE mode, the down weights should be swizzled as follows:
                    I, H = w.shape
                    w.reshape((ceil(I / 512), 128, 4, H)).transpose(1, 0, 3, 2).to_x4_dtype()
                - It expects up and gate weights to be swizzled as well. See gate_up_w_layout for details.
            - QuantizationType.ROW: FP8 row-wise quantization (CTE and TKG)
            - QuantizationType.MX: MXFP quantization (MXFP4/MXFP8, TKG only)
        gate_w_scale (nl.NkiTensor, optional): Dequantization scales for gate weights.
            - FP8: Shape [128, I] for row-wise, [128, 1] for tensor-wise quantization
            - MXFP (TKG only): Scale factors for MXFP quantized weights
        up_w_scale (nl.NkiTensor, optional): Dequantization scales for up weights.
            - FP8: Shape [128, I] for row-wise, [128, 1] for tensor-wise quantization
            - MXFP (TKG only): Scale factors for MXFP quantized weights
        down_w_scale (nl.NkiTensor, optional): Dequantization scales for down weights.
            - FP8: Shape [128, I] for row-wise, [128, 1] for tensor-wise quantization
            - MXFP (TKG only): Scale factors for MXFP quantized weights
        gate_up_in_scale (nl.NkiTensor, optional): FP8 dequantization scales for gate and up input.
            Used for tensor-wise quantization with shape [128, 1]. Defaults to None.
        down_in_scale (nl.NkiTensor, optional): FP8 dequantization scales for down input.
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
        mode (ComputationMode): Computation mode to use (default = AUTO)
            Supported values:
            - AUTO (default): automatically choose the more performant computation mode. Does not support fused normalization and quantization
            - PREFILL: use prefill computation mode, does not support fused quantization, but support quantized input.
            - DECDOE: use decode computation mode, not performant for large T, supported fused normalization and quantization.
        sbm (BufferManager): Optional BufferManager for HBM tensor allocation with consistent naming.
        transposed_in (bool): When True, input is in transposed HBM layout [H0, n_prgs, H1_shard, BxS]
            instead of [B, S, H]. Enables contiguous per-NC DMA loads. Only supported in TKG mode
            with NONE, STATIC, or ROW quantization. Not compatible with fused add or SBUF input.
            (default: False)
        transposed_out (bool): When True, output is in transposed HBM layout [H0, n_prgs, H1_shard, BxS]
            instead of [B, S, H]. Each NC DMAs its shard directly without transpose_store.
            Only supported in TKG mode. Not compatible with down_proj column tiling or SBUF output.
            (default: False)
        dtype_mode (DtypeMode): Explicit FP8 E4M3 dtype selection for TKG STATIC/ROW
            quantization weight tiles.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
            - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3``
              elsewhere.
            (default: DtypeMode.NON_OCP)
        gate_up_w_layout (MLPGateUpWeightLayout): The layout of gate_proj_weights_tensor and up_proj_weights_tensor (default: CONTIGUOUS)
            This parameter is required if quantization_type is either MX, ROW_MX, or STATIC_MX.
            Supported values:
            - CONTIGUOUS: Use if quantization_type is not one of MX, ROW_MX, or STATIC_MX
            - H_X4_INNERMOST: One of two alternative MX weight layouts. More optimal if hidden_tensor is already quantized.
            - H_X4_MIDDLE: One of two alternative MX weight layouts. More optimal if hidden_tensor is not quantized.

    Returns:
        list:
            The MLP output tensor(s):
            - HBM output: Tensor with shape [B, S, H].
            - HBM transposed output (transposed_out=True): Tensor with shape [H0, n_prgs, H1_shard, BxS].
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
        use_contiguous_x4_gate_up=use_contiguous_x4_gate_up,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        force_cte_mode=force_cte_mode,
        mx_dummy_scale_hbm=mx_dummy_scale_hbm,
        mode=mode,
        transposed_in=transposed_in,
        transposed_out=transposed_out,
        dtype_mode=dtype_mode,
        gate_up_w_layout=gate_up_w_layout,
    )

    # Validate MLP arguments
    validate_mlp_arguments(mlp_params)

    # Create local sbm if not provided
    if sbm is None:
        sbm = SbufManager(0, 200 * 1024, get_logger("mlp"))
        sbm.set_name_prefix("mlp_")
    # Allocate output tensor in shared HBM memory
    output_tensors = []
    out = None
    if not store_output_in_sbuf and not transposed_out:
        out = sbm.alloc(
            (mlp_params.batch_size, mlp_params.sequence_len, mlp_params.hidden_size),
            dtype=mlp_params.output_dtype,
            buffer=nl.shared_hbm,
            name="output_tensor_hbm",
        )
    elif transposed_out:
        # Allocate shared transposed output: [H0, n_prgs, H1_shard, T]
        _, n_prgs, _ = get_verified_program_sharding_info("mlp", (0, 1))
        H0 = 128
        H1_shard = mlp_params.hidden_size // (H0 * n_prgs)
        T = mlp_params.batch_size * mlp_params.sequence_len
        out = sbm.alloc(
            (H0, n_prgs, H1_shard, T),
            dtype=mlp_params.output_dtype,
            buffer=nl.shared_hbm,
            name="output_tensor_hbm_transposed",
        )
    output_tensors.append(out)

    # Optionally allocate an additional tensor to store fused addition results
    fused_add_out = None
    if mlpp_store_fused_add(mlp_params):
        fused_add_out = sbm.alloc(
            (mlp_params.batch_size, mlp_params.sequence_len, mlp_params.hidden_size),
            dtype=mlp_params.output_dtype,
            buffer=nl.shared_hbm,
            name="output_stored_add_tensor_hbm",
        )
        output_tensors.append(fused_add_out)

    # Determine if MLP should be invoked in token-generation (TKG) mode or context encoding (CTE) mode
    if is_mlp_tkg(mlp_params):
        if mlp_params.quant_params.is_dtype_mx():
            # mlp_tkg_mx does not use sbm
            return mlp_tkg_mx(mlp_params, out, fused_add_out)
        else:
            return mlp_tkg(mlp_params, out, fused_add_out, sbm=sbm)
    else:
        # mlp_cte doesn't accept a passed-in SBM
        mlp_cte(mlp_params, out, fused_add_out)
        # Return all output tensors (mlp output and optionally fused add)
        return output_tensors
