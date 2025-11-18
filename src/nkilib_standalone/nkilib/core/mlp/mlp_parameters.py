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
from ..utils.common_types import ActFnType, NormType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import is_rms_normalization, normalization_uses_weights

SUPPORTED_DTYPES = [nl.bfloat16]

# Threshold currently set to 96 based on existing tuning; subject to future refinement.
TKG_BS_SEQLEN_THRESHOLD = 96

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
    fused_add_params: Optional[MLPFusedAddParameters]  # if this is None then there is no fused add
    norm_params: Optional[MLPNormalizationParameters]  # if this is None then there is no normalization
    bias_params: Optional[MLPBiasParameters]
    eps: float
    batch_size: int
    sequence_len: int
    hidden_size: int
    intermediate_size: int
    store_output_in_sbuf: bool
    use_tkg_gate_up_proj_column_tiling: bool
    use_tkg_down_proj_column_tiling: bool
    use_tkg_down_proj_optimized_layout: bool

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
        output_dtype: nki.dtype = nl.bfloat16,
        store_output_in_sbuf: bool = False,
        eps: float = 1e-6,
        use_tkg_gate_up_proj_column_tiling: bool = False,
        use_tkg_down_proj_column_tiling: bool = False,
        use_tkg_down_proj_optimized_layout: bool = False,
    ):
        self.batch_size = hidden_tensor.shape[0]
        self.sequence_len = hidden_tensor.shape[1]
        self.hidden_size = hidden_tensor.shape[2]
        self.intermediate_size = up_proj_weights_tensor.shape[1]
        self.hidden_tensor = hidden_tensor
        self.gate_proj_weights_tensor = gate_proj_weights_tensor
        self.up_proj_weights_tensor = up_proj_weights_tensor
        self.down_proj_weights_tensor = down_proj_weights_tensor
        self.activation_fn = activation_fn
        self.output_dtype = output_dtype
        self.eps = eps
        self.store_output_in_sbuf = store_output_in_sbuf
        self.use_tkg_gate_up_proj_column_tiling = use_tkg_gate_up_proj_column_tiling
        self.use_tkg_down_proj_column_tiling = use_tkg_down_proj_column_tiling
        self.use_tkg_down_proj_optimized_layout = use_tkg_down_proj_optimized_layout

        self.fused_add_params = MLPFusedAddParameters(fused_add_tensor, store_fused_add_result)
        self.norm_params = MLPNormalizationParameters(
            normalization_type, normalization_weights_tensor, normalization_bias_tensor
        )
        self.bias_params = MLPBiasParameters(gate_proj_bias_tensor, up_proj_bias_tensor, down_proj_bias_tensor)


def is_mlp_tkg(params: MLPParameters) -> bool:
    return params.batch_size * params.sequence_len <= TKG_BS_SEQLEN_THRESHOLD


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
    B, S, H = params.hidden_tensor.shape
    _dim, I = params.gate_proj_weights_tensor.shape

    # Determine if we are in token-generation (TKG) mode
    is_tkg = is_mlp_tkg(params)

    if is_tkg:
        kernel_assert(H % 128 == 0, f"Unsupported hidden dimension {H}; expected H % 128 == 0.")

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


def _validate_mlp_arguments_dtype(params):
    kernel_assert(
        params.hidden_tensor.dtype in SUPPORTED_DTYPES,
        f"Unsupported hidden_tensor dtype: got {params.hidden_tensor.dtype}, " f"expected one of {SUPPORTED_DTYPES}.",
    )
    kernel_assert(
        params.gate_proj_weights_tensor.dtype in SUPPORTED_DTYPES,
        f"Unsupported gate_proj_weights_tensor dtype: got {params.gate_proj_weights_tensor.dtype}, "
        f"expected one of {SUPPORTED_DTYPES}.",
    )
    kernel_assert(
        params.up_proj_weights_tensor.dtype in SUPPORTED_DTYPES,
        f"Unsupported up_proj_weights_tensor dtype: got {params.up_proj_weights_tensor.dtype}, "
        f"expected one of {SUPPORTED_DTYPES}.",
    )
    kernel_assert(
        params.down_proj_weights_tensor.dtype in SUPPORTED_DTYPES,
        f"Unsupported down_proj_weights_tensor dtype: got {params.down_proj_weights_tensor.dtype}, "
        f"expected one of {SUPPORTED_DTYPES}.",
    )
    params.fused_add_params._validate_dtype()
    params.norm_params._validate_dtype()
    params.bias_params._validate_dtype()


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

    else:  # CTE mode
        kernel_assert(
            not params.store_output_in_sbuf,
            "Storing output in SBUF is only supported in TKG mode due to SBUF size limitations.",
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
