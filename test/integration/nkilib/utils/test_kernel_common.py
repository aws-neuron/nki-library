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
from typing import Any, Optional

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch
from nkilib_src.nkilib.core.subkernels.layernorm_torch import layer_norm_torch_ref
from nkilib_src.nkilib.core.subkernels.rmsnorm_torch import rms_norm_torch_ref
from nkilib_src.nkilib.core.utils.common_types import ActFnType, NormType
from scipy.special import erf, expit


def rms_norm(hidden, gamma, eps=1e-6, hidden_actual=None, **_):
    # All intermediates need to happen in FP32 for numerical precision
    hidden = hidden.astype(np.float32)

    if hidden_actual is not None:
        sum_squares = np.sum(np.square(hidden), axis=-1, keepdims=True)
        rms = np.sqrt(sum_squares / hidden_actual + eps).astype(hidden.dtype)
    else:
        rms = np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True) + eps).astype(hidden.dtype)

    norm = hidden * np.reciprocal(rms).astype(hidden.dtype)
    if gamma is not None:
        gamma = gamma.astype(np.float32)
        norm *= gamma
    return norm


# D = 1 LayerNorm
def layer_norm(hidden, gamma, norm_b=None, eps=1e-6, **_):
    mean = np.mean(hidden.astype(np.float32), axis=-1, keepdims=True)
    var = np.var(hidden.astype(np.float32), axis=-1, keepdims=True)
    norm = ((hidden - mean) * np.reciprocal(np.sqrt(var + eps))).astype(hidden.dtype)
    if gamma is not None:
        norm *= gamma
    if norm_b is not None:
        norm += norm_b
    return norm


norm_name2func = {
    NormType.NO_NORM: lambda *x, **_: x[0],
    NormType.RMS_NORM: rms_norm,
    NormType.LAYER_NORM: layer_norm,
    NormType.RMS_NORM_SKIP_GAMMA: rms_norm,
}


norm_name2func_torch = {
    NormType.NO_NORM: lambda *x, **_: x[0],
    NormType.RMS_NORM: rms_norm_torch_ref,
    NormType.LAYER_NORM: layer_norm_torch_ref,
    NormType.RMS_NORM_SKIP_GAMMA: rms_norm_torch_ref,
}


def gelu(x: np.ndarray):
    # 0.5x(1+erf(x/√2))
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def silu(x: np.ndarray):
    # silu(x) = x * sigmoid(x)
    return x * expit(x)


def gelu_apprx_tanh(x: np.ndarray):
    # REFERENCE: https://github.com/hendrycks/GELUs
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def gelu_apprx_sigmoid(x: np.ndarray):
    # REFERENCE: https://github.com/hendrycks/GELUs
    return x * (1 / (1 + np.exp(-1.702 * x)))


def gelu_apprx_sigmoid_dx(x: np.ndarray):
    """
    Nuemrically stable form of gelu_apprx_sigmoid_dx
    """
    X = 1.702 * x
    S0 = expit(X)
    S1 = expit(-X)
    return S0 * (1.0 + X * S1)


act_fn_type2func = {
    ActFnType.SiLU: silu,
    ActFnType.GELU: gelu,
    ActFnType.GELU_Tanh_Approx: gelu_apprx_tanh,
    ActFnType.Swish: gelu_apprx_sigmoid,
}


def convert_to_torch(tensor: Optional[np.ndarray]) -> Any:
    """
    Convert a numpy tensor to torch.
    """
    if tensor is None:
        return None

    # Torch cannot directly convert some types such as bf16
    try:
        result = torch.from_numpy(tensor)
    except:
        if tensor.dtype == nl.float8_e4m3:
            result = torch.from_numpy(dt.static_cast(tensor, np.float32))
        elif tensor.dtype == nl.bfloat16:
            result = torch.from_numpy(tensor.astype(np.float32)).to(torch.bfloat16)
        elif tensor.dtype == np.uint32:
            result = torch.from_numpy(tensor.view(np.int32))
        else:
            raise
    return result


def convert_to_numpy(tensor: Optional[torch.Tensor], dtype=None) -> Optional[np.ndarray]:
    """
    Convert a torch tensor to numpy.
    """
    if tensor is None:
        return None

    if dtype is not None:
        return dt.static_cast(tensor.numpy(), dtype)
    return tensor.numpy()


def is_dtype_mx(dtype):
    """Check if dtype is an MX quantization format."""
    return dtype in (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4)


def is_dtype_fp8(dtype):
    return dtype in (nl.float8_e4m3, nl.float8_e5m2)


def is_dtype_low_precision(dtype):
    return is_dtype_mx(dtype) or is_dtype_fp8(dtype)
