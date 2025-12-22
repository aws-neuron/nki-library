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


from dataclasses import dataclass
from typing import Optional

import nki
import nki.language as nl
from nki.language import NKIObject

# common utils
from ..utils.common_types import ActFnType, ExpertAffinityScaleMode, NormType, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import is_rms_normalization, normalization_uses_weights
from ..utils.tensor_view import TensorView

SUPPORTED_DTYPES = [nl.bfloat16, nl.float8_e4m3, 'float8e4']
SUPPORTED_QUANT_TYPES = [QuantizationType.NONE, QuantizationType.STATIC, QuantizationType.ROW, QuantizationType.MX]

# Threshold currently set to 96 based on existing tuning; subject to future refinement.
TKG_BS_SEQLEN_THRESHOLD = 96


#
#
# ****************************
# Quantization params and method
# ****************************
#
@dataclass
class MLPQuantizationParameters(NKIObject):
    quantization_type: QuantizationType
    gate_w_scale: Optional[nl.ndarray]
    up_w_scale: Optional[nl.ndarray]
    down_w_scale: Optional[nl.ndarray]
    gate_up_in_scale: Optional[nl.ndarray]
    down_in_scale: Optional[nl.ndarray]
    clipping_bound: float

    def __init__(
        self,
        quantization_type: QuantizationType,
        gate_w_scale: Optional[nl.ndarray],
        up_w_scale: Optional[nl.ndarray],
        down_w_scale: Optional[nl.ndarray],
        gate_up_in_scale: Optional[nl.ndarray],
        down_in_scale: Optional[nl.ndarray],
        clipping_bound: float,
    ):
        self.quantization_type = quantization_type
        self.gate_w_scale = gate_w_scale
        self.up_w_scale = up_w_scale
        self.down_w_scale = down_w_scale
        self.gate_up_in_scale = gate_up_in_scale
        self.down_in_scale = down_in_scale
        self.clipping_bound = clipping_bound

    def _validate_dtype(self):
        kernel_assert(
            self.quantization_type == QuantizationType.NONE
            or self.quantization_type == QuantizationType.ROW
            or self.quantization_type == QuantizationType.STATIC
            or self.quantization_type == QuantizationType.MX,
            f"Unsupported quantization_type: got {self.quantization_type},"
            f"expected one of the values:{SUPPORTED_QUANT_TYPES}.",
        )

        if self.quantization_type in (QuantizationType.ROW, QuantizationType.STATIC):
            kernel_assert(
                self.gate_w_scale != None and self.gate_w_scale.dtype == nl.float32,
                f"Unsupported gate_w_scale dtype: got {self.gate_w_scale.dtype}, expected nl.float32.",
            )

            kernel_assert(
                self.up_w_scale != None and self.up_w_scale.dtype == nl.float32,
                f"Unsupported up_w_scale dtype: got {self.up_w_scale.dtype}, expected nl.float32.",
            )

            kernel_assert(
                self.down_w_scale != None and self.down_w_scale.dtype == nl.float32,
                f"Unsupported down_w_scale dtype: got {self.down_w_scale.dtype}, expected nl.float32.",
            )

        if self.quantization_type == QuantizationType.STATIC:
            kernel_assert(
                self.gate_up_in_scale != None and self.gate_up_in_scale.dtype == nl.float32,
                f"Unsupported gate_up_in_scale dtype: got {self.gate_up_in_scale.dtype}, expected nl.float32.",
            )

            kernel_assert(
                self.down_in_scale != None and self.down_in_scale.dtype == nl.float32,
                f"Unsupported down_in_scale dtype: got {self.down_in_scale.dtype}, expected nl.float32.",
            )

        if self.quantization_type == QuantizationType.MX:
            kernel_assert(
                self.gate_w_scale != None and self.gate_w_scale.dtype == nl.uint8,
                f"Unsupported gate_w_scale dtype: got {self.gate_w_scale}, expected nl.uint8.",
            )

            kernel_assert(
                self.up_w_scale != None and self.up_w_scale.dtype == nl.uint8,
                f"Unsupported up_w_scale dtype: got {self.up_w_scale}, expected nl.uint8.",
            )

            kernel_assert(
                self.down_w_scale != None and self.down_w_scale.dtype == nl.uint8,
                f"Unsupported down_w_scale dtype: got {self.down_w_scale}, expected nl.uint8.",
            )

    def _validate_shapes(self, params):
        # Extract input tensor shapes
        H = params.up_proj_weights_tensor.shape[0]
        I = params.up_proj_weights_tensor.shape[1]
        if self.quantization_type == QuantizationType.STATIC:
            kernel_assert(
                self.gate_up_in_scale != None and self.gate_up_in_scale.shape == (128, 1),
                f"Unsupported gate_up_in_scale shape: got {self.gate_up_in_scale.shape}, expected (128, 1).",
            )
            kernel_assert(
                self.down_in_scale != None and self.down_in_scale.shape == (128, 1),
                f"Unsupported down_in_scale shape: got {self.down_in_scale.shape}, expected (128, 1).",
            )
            kernel_assert(
                self.gate_w_scale == None or self.gate_w_scale.shape == (128, 1),
                f"Unsupported gate_w_scale shape: got {self.gate_w_scale.shape}, expected (128, 1).",
            )
            kernel_assert(
                self.up_w_scale == None or self.up_w_scale.shape == (128, 1),
                f"Unsupported up_w_scale shape: got {self.up_w_scale.shape}, expected (128, 1).",
            )
            kernel_assert(
                self.down_w_scale == None or self.down_w_scale.shape == (128, 1),
                f"Unsupported down_w_scale shape: got {self.down_w_scale.shape}, expected (128, 1).",
            )
        elif self.quantization_type == QuantizationType.ROW:
            kernel_assert(
                self.gate_w_scale == None or self.gate_w_scale.shape == (128, I),
                f"Unsupported gate_w_scale shape: got {self.gate_w_scale.shape}, expected (128, {I}).",
            )
            kernel_assert(
                self.up_w_scale == None or self.up_w_scale.shape == (128, I),
                f"Unsupported up_w_scale shape: got {self.up_w_scale.shape}, expected (128, {I}).",
            )
            kernel_assert(
                self.down_w_scale == None or self.down_w_scale.shape == (128, H),
                f"Unsupported down_w_scale shape: got {self.down_w_scale.shape}, expected (128, {H}).",
            )

    def is_no_quant(self):
        return self.quantization_type == QuantizationType.NONE

    def is_quant(self):
        return self.quantization_type != QuantizationType.NONE

    def is_quant_static(self):
        return self.quantization_type == QuantizationType.STATIC

    def is_quant_row(self):
        return self.quantization_type == QuantizationType.ROW

    def is_quant_mx(self):
        return self.quantization_type == QuantizationType.MX

    def has_clipping_bound(self):
        return self.clipping_bound > 0.0

    def convert_to_view(self):
        if self.gate_w_scale is not None:
            self.gate_w_scale = TensorView(self.gate_w_scale)
        if self.up_w_scale is not None:
            self.up_w_scale = TensorView(self.up_w_scale)
        if self.down_w_scale is not None:
            self.down_w_scale = TensorView(self.down_w_scale)
        if self.gate_up_in_scale is not None:
            self.gate_up_in_scale = TensorView(self.gate_up_in_scale)
        if self.down_in_scale is not None:
            self.down_in_scale = TensorView(self.down_in_scale)


#
#
# ****************************
# Fused add params and methods
# ****************************
#


@dataclass
class MLPFusedAddParameters(NKIObject):
    fused_add_tensor: Optional[nl.ndarray]
    store_fused_add_result: bool

    def __init__(self, fused_add_tensor: Optional[nl.ndarray], store_fused_add_result: bool):
        self.fused_add_tensor = fused_add_tensor if fused_add_tensor != None else None
        self.store_fused_add_result = store_fused_add_result

    def _validate_dtype(self):
        if self.fused_add_tensor != None:
            kernel_assert(
                self.fused_add_tensor.dtype in SUPPORTED_DTYPES,
                f"Unsupported fused_add_tensor dtype: got {self.fused_add_tensor.dtype}, "
                f"expected one of {SUPPORTED_DTYPES}.",
            )


#
# ********************************
# Normalization params and methods
# ********************************
#


@dataclass
class MLPNormalizationParameters(NKIObject):
    normalization_type: NormType
    normalization_weights_tensor: Optional[nl.ndarray]
    normalization_bias_tensor: Optional[nl.ndarray]

    def __init__(
        self,
        normalization_type: NormType,
        normalization_weights_tensor: Optional[nl.ndarray],
        normalization_bias_tensor: Optional[nl.ndarray],
    ):
        # If NO_NORM, set all fields to None
        if normalization_type == NormType.NO_NORM:
            self.normalization_type = NormType.NO_NORM
            self.normalization_weights_tensor = None
            self.normalization_bias_tensor = None
        else:
            self.normalization_type = normalization_type
            self.normalization_weights_tensor = normalization_weights_tensor
            self.normalization_bias_tensor = normalization_bias_tensor

    def _validate_dtype(self):
        if self.normalization_weights_tensor != None:
            kernel_assert(
                self.normalization_weights_tensor.dtype in SUPPORTED_DTYPES,
                f"Unsupported normalization_weights_tensor dtype: got {self.normalization_weights_tensor.dtype}, "
                f"expected one of {SUPPORTED_DTYPES}.",
            )
        if self.normalization_bias_tensor != None:
            kernel_assert(
                self.normalization_bias_tensor.dtype in SUPPORTED_DTYPES,
                f"Unsupported normalization_bias_tensor dtype: got {self.normalization_bias_tensor.dtype}, "
                f"expected one of {SUPPORTED_DTYPES}.",
            )


@dataclass
class MLPExpertParameters(NKIObject):
    expert_affinities: nl.ndarray
    expert_index: nl.ndarray
    expert_affinities_eager: Optional[nl.ndarray]
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE


#
# ***********************
# Bias params and methods
# ***********************
#


@dataclass
class MLPBiasParameters(NKIObject):
    gate_proj_bias_tensor: Optional[nl.ndarray]
    up_proj_bias_tensor: Optional[nl.ndarray]
    down_proj_bias_tensor: Optional[nl.ndarray]

    def __init__(
        self,
        gate_proj_bias_tensor: Optional[nl.ndarray],
        up_proj_bias_tensor: Optional[nl.ndarray],
        down_proj_bias_tensor: Optional[nl.ndarray],
    ):
        self.gate_proj_bias_tensor = gate_proj_bias_tensor
        self.up_proj_bias_tensor = up_proj_bias_tensor
        self.down_proj_bias_tensor = down_proj_bias_tensor

    def _validate_dtype(self):
        if self.gate_proj_bias_tensor != None:
            kernel_assert(
                self.gate_proj_bias_tensor.dtype in SUPPORTED_DTYPES,
                f"Unsupported gate_proj_bias_tensor dtype: got {self.gate_proj_bias_tensor.dtype}, "
                f"expected one of {SUPPORTED_DTYPES}.",
            )
        if self.up_proj_bias_tensor != None:
            kernel_assert(
                self.up_proj_bias_tensor.dtype in SUPPORTED_DTYPES,
                f"Unsupported up_proj_bias_tensor dtype: got {self.up_proj_bias_tensor.dtype}, "
                f"expected one of {SUPPORTED_DTYPES}.",
            )
        if self.down_proj_bias_tensor != None:
            kernel_assert(
                self.down_proj_bias_tensor.dtype in SUPPORTED_DTYPES,
                f"Unsupported down_proj_bias_tensor dtype: got {self.down_proj_bias_tensor.dtype}, "
                f"expected one of {SUPPORTED_DTYPES}.",
            )


#
# ***********************
# MLP params and methods
# ***********************
#


@dataclass
class MLPParameters(NKIObject):
    hidden_tensor: nl.ndarray
    gate_proj_weights_tensor: nl.ndarray
    up_proj_weights_tensor: nl.ndarray
    down_proj_weights_tensor: nl.ndarray
    activation_fn: ActFnType
    output_dtype: nki.dtype
    fused_add_params: Optional[MLPFusedAddParameters]
    norm_params: Optional[MLPNormalizationParameters]
    bias_params: Optional[MLPBiasParameters]
    quant_params: Optional[MLPQuantizationParameters]
    expert_params: Optional[MLPExpertParameters]
    eps: float
    batch_size: int
    sequence_len: int
    hidden_size: int
    intermediate_size: int
    input_in_sbuf: bool
    store_output_in_sbuf: bool
    skip_gate_proj: bool
    use_tkg_gate_up_proj_column_tiling: bool
    use_tkg_down_proj_column_tiling: bool
    use_tkg_down_proj_optimized_layout: bool
    shard_on_k: bool
    gate_clamp_lower_limit: Optional[float]
    gate_clamp_upper_limit: Optional[float]
    up_clamp_lower_limit: Optional[float]
    up_clamp_upper_limit: Optional[float]

    def __init__(
        self,
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
        output_dtype: nki.dtype = nl.bfloat16,
        store_output_in_sbuf: bool = False,
        eps: float = 1e-6,
        skip_gate_proj: bool = False,
        use_tkg_gate_up_proj_column_tiling: bool = False,
        use_tkg_down_proj_column_tiling: bool = False,
        use_tkg_down_proj_optimized_layout: bool = False,
        shard_on_k: bool = False,
        gate_clamp_lower_limit: Optional[float] = None,
        gate_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        expert_params: Optional[MLPExpertParameters] = None,
    ):
        self.input_in_sbuf = hidden_tensor.buffer == nl.sbuf
        if self.input_in_sbuf:
            # SBUF input shape: [H0, T, H1]
            kernel_assert(len(hidden_tensor.shape) == 3, "SBUF input must have 3D shape [H0, T, H1]")
            # Might be sharded so get hidden_size from weights tensor
            _, T, _ = hidden_tensor.shape
            self.batch_size = 1
            self.sequence_len = T
            self.hidden_size = down_proj_weights_tensor.shape[-1]
        elif len(hidden_tensor.shape) == 3:  # B, S, H
            self.batch_size = hidden_tensor.shape[0]
            self.sequence_len = hidden_tensor.shape[1]
            self.hidden_size = hidden_tensor.shape[2]
        else:  # T, H
            self.batch_size = 1
            self.sequence_len = hidden_tensor.shape[0]
            self.hidden_size = hidden_tensor.shape[1]

        if len(down_proj_weights_tensor.shape) == 3:  # E, I, H
            self.intermediate_size = down_proj_weights_tensor.shape[1]
            kernel_assert(
                down_proj_weights_tensor.shape[2] == self.hidden_size,
                "unexpected down project weight shape {down_proj_weights_tensor.shape}",
            )
        elif len(down_proj_weights_tensor.shape) == 2:  # I, H
            self.intermediate_size = down_proj_weights_tensor.shape[0]
            kernel_assert(
                down_proj_weights_tensor.shape[1] == self.hidden_size,
                "unexpected down project weight shape {down_proj_weights_tensor.shape}",
            )
        self.hidden_tensor = hidden_tensor
        self.gate_proj_weights_tensor = gate_proj_weights_tensor
        self.up_proj_weights_tensor = up_proj_weights_tensor
        self.down_proj_weights_tensor = down_proj_weights_tensor
        self.activation_fn = activation_fn
        self.output_dtype = output_dtype
        self.eps = eps
        self.store_output_in_sbuf = store_output_in_sbuf
        self.skip_gate_proj = skip_gate_proj
        self.use_tkg_gate_up_proj_column_tiling = use_tkg_gate_up_proj_column_tiling
        self.use_tkg_down_proj_column_tiling = use_tkg_down_proj_column_tiling
        self.use_tkg_down_proj_optimized_layout = use_tkg_down_proj_optimized_layout
        self.shard_on_k = shard_on_k
        self.gate_clamp_lower_limit = gate_clamp_lower_limit
        self.gate_clamp_upper_limit = gate_clamp_upper_limit
        self.up_clamp_lower_limit = up_clamp_lower_limit
        self.up_clamp_upper_limit = up_clamp_upper_limit

        self.fused_add_params = MLPFusedAddParameters(fused_add_tensor, store_fused_add_result)
        self.norm_params = MLPNormalizationParameters(
            normalization_type, normalization_weights_tensor, normalization_bias_tensor
        )
        self.bias_params = MLPBiasParameters(gate_proj_bias_tensor, up_proj_bias_tensor, down_proj_bias_tensor)
        self.expert_params = expert_params

        self.quant_params = MLPQuantizationParameters(
            quantization_type,
            gate_w_scale,
            up_w_scale,
            down_w_scale,
            gate_up_in_scale,
            down_in_scale,
            quant_clipping_bound,
        )


def is_mlp_tkg(params: MLPParameters) -> bool:
    return params.batch_size * params.sequence_len <= TKG_BS_SEQLEN_THRESHOLD


def mlpp_has_quantized_weights(params: MLPParameters) -> bool:
    return params.quant_params.is_quant()


def mlpp_has_quantized_input(params: MLPParameters) -> bool:
    return params.hidden_tensor.dtype in [nl.float8_e4m3, 'float8e4']


def mlpp_input_has_packed_scale(params: MLPParameters) -> bool:
    return mlpp_has_quantized_input(params) and params.quant_params.is_quant_row()


def mlpp_has_fused_add(params: MLPParameters) -> bool:
    return params.fused_add_params.fused_add_tensor != None


def mlpp_store_fused_add(params: MLPParameters) -> bool:
    return mlpp_has_fused_add(params) and params.fused_add_params.store_fused_add_result


def mlpp_has_normalization(params: MLPParameters) -> bool:
    return params.norm_params.normalization_type != NormType.NO_NORM


def mlpp_has_rms_normalization(params: MLPParameters) -> bool:
    return is_rms_normalization(params.norm_params.normalization_type)


def mlpp_has_layer_normalization(params: MLPParameters) -> bool:
    return params.norm_params.normalization_type == NormType.LAYER_NORM


def mlpp_has_normalization_weights(params: MLPParameters) -> bool:
    return mlpp_has_normalization(params) and normalization_uses_weights(params.norm_params.normalization_type)


def mlpp_has_gate_projection(params: MLPParameters) -> bool:
    return params.gate_proj_weights_tensor != None


def mlpp_has_gate_projection_bias(params: MLPParameters) -> bool:
    return params.bias_params.gate_proj_bias_tensor != None


def mlpp_has_up_projection_bias(params: MLPParameters) -> bool:
    return params.bias_params.up_proj_bias_tensor != None


def mlpp_has_down_projection_bias(params: MLPParameters) -> bool:
    return params.bias_params.down_proj_bias_tensor != None


def mlpp_has_projection_bias(params: MLPParameters) -> bool:
    return (
        mlpp_has_up_projection_bias(params)
        or mlpp_has_down_projection_bias(params)
        or mlpp_has_gate_projection_bias(params)
    )


def mlpp_has_normalization_bias(params: MLPParameters) -> bool:
    return mlpp_has_normalization(params) and params.norm_params.normalization_bias_tensor != None


def override_seq_len(mlp_params: MLPParameters, seq_len: int) -> MLPParameters:
    kernel_assert(
        seq_len > 0 and seq_len <= mlp_params.sequence_len,
        f"Internal error: Sequence length override of {seq_len} is outside the bounds of the legal range [1-{mlp_params.sequence_len}].",
    )
    mlp_params.seq_len = seq_len
    return mlp_params


def override_inter_size(mlp_params: MLPParameters, inter_sz: int) -> MLPParameters:
    kernel_assert(
        inter_sz > 0 and inter_sz <= mlp_params.intermediate_size,
        f"Internal error: Intermediate size override of {inter_sz} is outside the bounds of the legal range [1-{mlp_params.intermediate_size}].",
    )
    mlp_params.intermediate_size = inter_sz
    return mlp_params


def _validate_mlp_required_arguments(params: MLPParameters):
    kernel_assert(params.hidden_tensor != None, "Hidden tensor is a required argument")
    kernel_assert(
        params.gate_proj_weights_tensor != None,
        "Gate projection tensor is a required argument",
    )
    kernel_assert(
        params.up_proj_weights_tensor != None,
        "Up projection tensor is a required argument",
    )
    kernel_assert(
        params.down_proj_weights_tensor != None,
        "Down projection tensor is a required argument",
    )


def _validate_mlp_arguments_shapes(params: MLPParameters):
    # Extract input tensor shapes and data type
    BxS = params.batch_size * params.sequence_len
    H = params.gate_proj_weights_tensor.shape[0]
    _dim = params.hidden_tensor.shape[2]
    I = params.gate_proj_weights_tensor.shape[1]

    # Determine if we are in token-generation (TKG) mode
    is_tkg = is_mlp_tkg(params)

    kernel_assert(H % 128 == 0, f"Unsupported hidden dimension {H}; expected H % 128 == 0.")

    kernel_assert(BxS > 0, f'Unsupported batch by sequence dimension {BxS}; expected BxS to be positive.')
    kernel_assert(H > 0, f'Unsupported hidden dimension {H}; expected H to be positive.')
    kernel_assert(I > 0, f'Unsupported intermediate dimension {I}; expected I to be positive.')

    if is_tkg or not mlpp_input_has_packed_scale(params):
        kernel_assert(
            _dim == H,
            f"Reduction dimension mismatch: got {_dim}, expected {_dim} == {H}.",
        )

    if mlpp_has_gate_projection_bias(params):
        expected = (1, I)
        actual = params.bias_params.gate_proj_bias_tensor.shape
        kernel_assert(
            actual == expected,
            f"Gate projection bias shape mismatch: expected {expected}, got {actual}.",
        )

    if mlpp_has_up_projection_bias(params):
        expected = (1, I)
        actual = params.bias_params.up_proj_bias_tensor.shape
        kernel_assert(
            actual == expected,
            f"Up projection bias shape mismatch: expected {expected}, got {actual}.",
        )

    if mlpp_has_down_projection_bias(params):
        expected = (1, H)
        actual = params.bias_params.down_proj_bias_tensor.shape
        kernel_assert(
            actual == expected,
            f"Down projection bias shape mismatch: expected {expected}, got {actual}.",
        )

    params.quant_params._validate_shapes(params)


def _validate_mlp_arguments_dtype(params):
    kernel_assert(
        params.hidden_tensor.dtype in SUPPORTED_DTYPES,
        f"Unsupported hidden_tensor dtype: got {params.hidden_tensor.dtype}, expected one of {SUPPORTED_DTYPES}.",
    )
    kernel_assert(
        params.gate_proj_weights_tensor.dtype in SUPPORTED_DTYPES
        or str(params.gate_proj_weights_tensor.dtype) in SUPPORTED_DTYPES,
        f"Unsupported gate_proj_weights_tensor dtype: got {params.gate_proj_weights_tensor.dtype}, "
        f"expected one of {SUPPORTED_DTYPES}.",
    )
    kernel_assert(
        params.up_proj_weights_tensor.dtype in SUPPORTED_DTYPES
        or str(params.up_proj_weights_tensor.dtype) in SUPPORTED_DTYPES,
        f"Unsupported up_proj_weights_tensor dtype: got {params.up_proj_weights_tensor.dtype}, "
        f"expected one of {SUPPORTED_DTYPES}.",
    )
    kernel_assert(
        params.down_proj_weights_tensor.dtype in SUPPORTED_DTYPES
        or str(params.down_proj_weights_tensor.dtype) in SUPPORTED_DTYPES,
        f"Unsupported down_proj_weights_tensor dtype: got {params.down_proj_weights_tensor.dtype}, "
        f"expected one of {SUPPORTED_DTYPES}.",
    )
    params.fused_add_params._validate_dtype()
    params.norm_params._validate_dtype()
    params.bias_params._validate_dtype()
    params.quant_params._validate_dtype()


def _validate_mlp_arguments_restrictions(params: MLPParameters):
    kernel_assert(
        nl.program_ndim() == 0 or nl.program_ndim() == 1,
        "kernel only supports no specialization or specialization along one axis",
    )

    kernel_assert(
        not mlpp_has_normalization(params)
        or params.norm_params.normalization_type == NormType.LAYER_NORM
        or not params.norm_params.normalization_bias_tensor,
        "Normalization bias is only supported for LAYER_NORM",
    )

    is_tkg = is_mlp_tkg(params)

    if is_tkg:  # TKG mode
        if params.use_tkg_down_proj_optimized_layout:
            kernel_assert(
                not params.use_tkg_down_proj_column_tiling,
                "Optimized layout for down_proj is only supported in TKG mode without column tiling. "
                "Please disable use_tkg_down_proj_column_tiling to enable this.",
            )

        if params.input_in_sbuf:
            kernel_assert(
                not params.store_output_in_sbuf,
                "Storing fused_add_result is not supported when input is in SBUF",
            )

    else:  # CTE mode
        kernel_assert(
            not params.skip_gate_proj,
            "Skipping gate projection is only supported in TKG mode.",
        )

        kernel_assert(
            params.gate_clamp_lower_limit is None and params.gate_clamp_upper_limit is None,
            "Gate projection clamp is only supported in TKG mode.",
        )

        kernel_assert(
            params.up_clamp_lower_limit is None and params.up_clamp_upper_limit is None,
            "Up projection clamp is only supported in TKG mode.",
        )

        kernel_assert(
            not params.store_output_in_sbuf,
            "Storing output in SBUF is only supported in TKG mode due to SBUF size limitations.",
        )

        kernel_assert(
            not params.input_in_sbuf,
            "Taking input in SBUF is only supported in TKG mode due to SBUF size limitations.",
        )

        kernel_assert(
            not params.use_tkg_down_proj_optimized_layout,
            "Down projection layout optimization is only supported in TKG mode.",
        )


def validate_mlp_arguments(params: MLPParameters):
    _validate_mlp_required_arguments(params)
    _validate_mlp_arguments_shapes(params)
    _validate_mlp_arguments_dtype(params)
    _validate_mlp_arguments_restrictions(params)
