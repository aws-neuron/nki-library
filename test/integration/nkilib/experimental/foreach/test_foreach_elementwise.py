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

"""Integration tests for foreach elementwise kernels (add, sub, mul, div)."""

import ml_dtypes
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.foreach.foreach_elementwise import (
    add_scalar_kernel,
    add_tensor_kernel,
    addcdiv_kernel,
    addcmul_kernel,
    div_scalar_kernel,
    div_tensor_kernel,
    lerp_kernel,
    mul_scalar_kernel,
    mul_tensor_kernel,
    sqrt_kernel,
    sub_scalar_kernel,
    sub_tensor_kernel,
)
from nkilib_src.nkilib.experimental.foreach.foreach_elementwise_torch import (
    add_scalar_torch_ref,
    add_tensor_torch_ref,
    addcdiv_torch_ref,
    addcmul_torch_ref,
    div_scalar_torch_ref,
    div_tensor_torch_ref,
    lerp_torch_ref,
    mul_scalar_torch_ref,
    mul_tensor_torch_ref,
    sqrt_torch_ref,
    sub_scalar_torch_ref,
    sub_tensor_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_SCALAR_VALUE = 2.5
_ALPHA_VALUE = 1.5
_VALUE = 0.5  # for addcdiv/addcmul value param
_WEIGHT = 0.3  # for lerp weight param


TEST_SHAPES = [(128,), (256,), (1024,), (32, 32), (4, 8, 32)]
BOUNDARY_SHAPES = [
    (127,),
    (129,),
    (255,),
    (257,),
    # Single tile boundary
    (128, 2048 - 1),
    (128, 2048 + 1),
    # Two tile boundary
    (256, 2048 - 1),
    (256, 2048 + 1),
    # Multi-tile (K > 2)
    (128, 2048 * 3),
    (128, 2048 * 4),
]


def _scalar_inputs(shape):
    np.random.seed(42)
    data = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    scalar_tensor = np.full((128, 1), _SCALAR_VALUE, dtype=np.float32)
    return {"data": data, "scalar_tensor": scalar_tensor, "numel": int(np.prod(shape))}


def _tensor_inputs(shape):
    np.random.seed(42)
    data1 = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    data2 = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    return {"data1": data1, "data2": data2, "numel": int(np.prod(shape))}


def _tensor_alpha_inputs(shape):
    np.random.seed(42)
    data1 = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    data2 = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    alpha_tensor = np.full((128, 1), _ALPHA_VALUE, dtype=np.float32)
    return {"data1": data1, "data2": data2, "alpha_tensor": alpha_tensor, "numel": int(np.prod(shape))}


def _three_tensor_value_inputs(shape):
    np.random.seed(42)
    data = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    data1 = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    # Avoid near-zero denominators for addcdiv
    data2 = (np.random.randn(*shape) + 2.0).astype(ml_dtypes.bfloat16)
    value_tensor = np.full((128, 1), _VALUE, dtype=np.float32)
    return {"data": data, "data1": data1, "data2": data2, "value_tensor": value_tensor, "numel": int(np.prod(shape))}


def _lerp_inputs(shape):
    np.random.seed(42)
    data = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    end = np.random.randn(*shape).astype(ml_dtypes.bfloat16)
    weight_tensor = np.full((128, 1), _WEIGHT, dtype=np.float32)
    return {"data": data, "end": end, "weight_tensor": weight_tensor, "numel": int(np.prod(shape))}


def _sqrt_inputs(shape):
    np.random.seed(42)
    data = np.abs(np.random.randn(*shape)).astype(ml_dtypes.bfloat16)
    return {"data": data, "numel": int(np.prod(shape))}


def _output_scalar(kernel_input):
    return {"out": np.zeros(kernel_input["data"].shape, dtype=ml_dtypes.bfloat16)}


def _output_tensor(kernel_input):
    return {"out": np.zeros(kernel_input["data1"].shape, dtype=ml_dtypes.bfloat16)}


def _output_three_tensor(kernel_input):
    return {"out": np.zeros(kernel_input["data"].shape, dtype=ml_dtypes.bfloat16)}


# Torch ref wrappers matching kernel signatures
def _wrap_scalar_ref(ref_fn):
    def wrapped(data: torch.Tensor, scalar_tensor: torch.Tensor, numel: int) -> torch.Tensor:
        return ref_fn(data.float(), _SCALAR_VALUE).to(data.dtype)

    return wrapped


def _wrap_tensor_ref(ref_fn):
    def wrapped(data1: torch.Tensor, data2: torch.Tensor, numel: int) -> torch.Tensor:
        return ref_fn(data1.float(), data2.float()).to(data1.dtype)

    return wrapped


def _wrap_tensor_alpha_ref(ref_fn):
    def wrapped(data1: torch.Tensor, data2: torch.Tensor, alpha_tensor: torch.Tensor, numel: int) -> torch.Tensor:
        return ref_fn(data1.float(), data2.float(), alpha_tensor=_ALPHA_VALUE).to(data1.dtype)

    return wrapped


def _wrap_addcdiv_ref():
    def wrapped(
        data: torch.Tensor, data1: torch.Tensor, data2: torch.Tensor, value_tensor: torch.Tensor, numel: int
    ) -> torch.Tensor:
        return addcdiv_torch_ref(data.float(), data1.float(), data2.float(), value_tensor=_VALUE).to(data.dtype)

    return wrapped


def _wrap_addcmul_ref():
    def wrapped(
        data: torch.Tensor, data1: torch.Tensor, data2: torch.Tensor, value_tensor: torch.Tensor, numel: int
    ) -> torch.Tensor:
        return addcmul_torch_ref(data.float(), data1.float(), data2.float(), value_tensor=_VALUE).to(data.dtype)

    return wrapped


def _wrap_lerp_ref():
    def wrapped(data: torch.Tensor, end: torch.Tensor, weight_tensor: torch.Tensor, numel: int) -> torch.Tensor:
        return lerp_torch_ref(data.float(), end.float(), weight_tensor=_WEIGHT).to(data.dtype)

    return wrapped


def _wrap_sqrt_ref():
    def wrapped(data: torch.Tensor, numel: int) -> torch.Tensor:
        return sqrt_torch_ref(data.float()).to(data.dtype)

    return wrapped


@pytest_test_metadata(
    name="ForeachElementwise",
    pytest_marks=["foreach_elementwise"],
)
class TestForeachElementwise:
    """Integration tests for foreach elementwise NKI kernels."""

    @pytest.mark.parametrize(
        "kernel,ref,shape",
        [
            pytest.param(k, r, s, marks=pytest.mark.fast)
            if (k, s)
            in {
                (div_scalar_kernel, (256,)),
                (sub_scalar_kernel, (256,)),
                (add_scalar_kernel, (1024,)),
            }
            else pytest.param(k, r, s)
            for k, r in [
                (add_scalar_kernel, add_scalar_torch_ref),
                (sub_scalar_kernel, sub_scalar_torch_ref),
                (mul_scalar_kernel, mul_scalar_torch_ref),
                (div_scalar_kernel, div_scalar_torch_ref),
            ]
            for s in TEST_SHAPES
        ],
    )
    def test_scalar_ops(self, test_manager: Orchestrator, platform_target: Platforms, shape, kernel, ref):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(_wrap_scalar_ref(ref)),
            kernel_input_generator=lambda _: _scalar_inputs(shape),
            output_tensor_descriptor=_output_scalar,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "kernel,ref,uses_alpha,shape",
        [
            pytest.param(k, r, ua, s, marks=pytest.mark.fast)
            if (k, s)
            in {
                (div_tensor_kernel, (1024,)),
                (mul_tensor_kernel, (4, 8, 32)),
                (sub_tensor_kernel, (4, 8, 32)),
                (div_tensor_kernel, (128,)),
                (mul_tensor_kernel, (128,)),
            }
            else pytest.param(k, r, ua, s)
            for k, r, ua in [
                (add_tensor_kernel, add_tensor_torch_ref, True),
                (sub_tensor_kernel, sub_tensor_torch_ref, True),
                (mul_tensor_kernel, mul_tensor_torch_ref, False),
                (div_tensor_kernel, div_tensor_torch_ref, False),
            ]
            for s in TEST_SHAPES
        ],
    )
    def test_tensor_ops(self, test_manager: Orchestrator, platform_target: Platforms, shape, kernel, ref, uses_alpha):
        if uses_alpha:

            def gen(_):
                return _tensor_alpha_inputs(shape)

            wrapped_ref = _wrap_tensor_alpha_ref(ref)
        else:

            def gen(_):
                return _tensor_inputs(shape)

            wrapped_ref = _wrap_tensor_ref(ref)
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(wrapped_ref),
            kernel_input_generator=gen,
            output_tensor_descriptor=_output_tensor,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "shape",
        [pytest.param(s, marks=pytest.mark.fast) if s == (257,) else pytest.param(s) for s in BOUNDARY_SHAPES],
    )
    def test_scalar_boundary(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        """Boundary sizes: non-multiples of P_MAX to exercise tail path."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mul_scalar_kernel,
            torch_ref=torch_ref_wrapper(_wrap_scalar_ref(mul_scalar_torch_ref)),
            kernel_input_generator=lambda _: _scalar_inputs(shape),
            output_tensor_descriptor=_output_scalar,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "shape",
        [pytest.param(s, marks=pytest.mark.fast) if s == (255,) else pytest.param(s) for s in BOUNDARY_SHAPES],
    )
    def test_tensor_boundary(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        """Boundary sizes: non-multiples of P_MAX to exercise tail path for tensor ops."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=add_tensor_kernel,
            torch_ref=torch_ref_wrapper(_wrap_tensor_alpha_ref(add_tensor_torch_ref)),
            kernel_input_generator=lambda _: _tensor_alpha_inputs(shape),
            output_tensor_descriptor=_output_tensor,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "kernel,ref_wrapper,shape",
        [
            pytest.param(k, rw, s, marks=pytest.mark.fast)
            if (k, s)
            in {
                (addcdiv_kernel, (256,)),
                (addcmul_kernel, (256,)),
                (addcdiv_kernel, (128,)),
                (addcmul_kernel, (128,)),
            }
            else pytest.param(k, rw, s)
            for k, rw in [
                (addcdiv_kernel, _wrap_addcdiv_ref()),
                (addcmul_kernel, _wrap_addcmul_ref()),
            ]
            for s in TEST_SHAPES
        ],
    )
    def test_addcd_ops(self, test_manager: Orchestrator, platform_target: Platforms, shape, kernel, ref_wrapper):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(ref_wrapper),
            kernel_input_generator=lambda _: _three_tensor_value_inputs(shape),
            output_tensor_descriptor=_output_three_tensor,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "shape",
        [pytest.param(s, marks=pytest.mark.fast) if s in {(32, 32), (128,)} else pytest.param(s) for s in TEST_SHAPES],
    )
    def test_lerp(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=lerp_kernel,
            torch_ref=torch_ref_wrapper(_wrap_lerp_ref()),
            kernel_input_generator=lambda _: _lerp_inputs(shape),
            output_tensor_descriptor=lambda ki: {"out": np.zeros(ki["data"].shape, dtype=ml_dtypes.bfloat16)},
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "shape",
        [pytest.param(s, marks=pytest.mark.fast) if s in {(1024,), (128,)} else pytest.param(s) for s in TEST_SHAPES],
    )
    def test_sqrt(self, test_manager: Orchestrator, platform_target: Platforms, shape):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=sqrt_kernel,
            torch_ref=torch_ref_wrapper(_wrap_sqrt_ref()),
            kernel_input_generator=lambda _: _sqrt_inputs(shape),
            output_tensor_descriptor=_output_scalar,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )
