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

"""PyTorch reference implementations for foreach elementwise kernels."""

import torch


def _to_scalar(x):
    """Extract scalar value from tensor or pass through if already scalar."""
    if isinstance(x, torch.Tensor):
        return x.flatten()[0].item()
    return x


def add_scalar_torch_ref(data: torch.Tensor, scalar_tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return data + _to_scalar(scalar_tensor)


def sub_scalar_torch_ref(data: torch.Tensor, scalar_tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return data - _to_scalar(scalar_tensor)


def mul_scalar_torch_ref(data: torch.Tensor, scalar_tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return data * _to_scalar(scalar_tensor)


def div_scalar_torch_ref(data: torch.Tensor, scalar_tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return data / _to_scalar(scalar_tensor)


def add_tensor_torch_ref(
    data1: torch.Tensor, data2: torch.Tensor, alpha_tensor=None, numel: int = None
) -> torch.Tensor:  # noqa: ARG001
    alpha = _to_scalar(alpha_tensor) if alpha_tensor is not None else 1.0
    return data1 + alpha * data2


def sub_tensor_torch_ref(
    data1: torch.Tensor, data2: torch.Tensor, alpha_tensor=None, numel: int = None
) -> torch.Tensor:  # noqa: ARG001
    alpha = _to_scalar(alpha_tensor) if alpha_tensor is not None else 1.0
    return data1 - alpha * data2


def mul_tensor_torch_ref(data1: torch.Tensor, data2: torch.Tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return data1 * data2


def div_tensor_torch_ref(data1: torch.Tensor, data2: torch.Tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return data1 / data2


def addcdiv_torch_ref(
    data: torch.Tensor, data1: torch.Tensor, data2: torch.Tensor, value_tensor=None, numel: int = None
) -> torch.Tensor:  # noqa: ARG001
    value = _to_scalar(value_tensor) if value_tensor is not None else 1.0
    return data + value * (data1 / data2)


def addcmul_torch_ref(
    data: torch.Tensor, data1: torch.Tensor, data2: torch.Tensor, value_tensor=None, numel: int = None
) -> torch.Tensor:  # noqa: ARG001
    value = _to_scalar(value_tensor) if value_tensor is not None else 1.0
    return data + value * (data1 * data2)


def lerp_torch_ref(data: torch.Tensor, end: torch.Tensor, weight_tensor=None, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    weight = _to_scalar(weight_tensor) if weight_tensor is not None else 0.0
    return data + weight * (end - data)


def sqrt_torch_ref(data: torch.Tensor, numel: int = None) -> torch.Tensor:  # noqa: ARG001
    return torch.sqrt(data)
