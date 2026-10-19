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

"""Tests for the 3D max pooling kernel using UnitTestFramework."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.pooling.max_pooling_3d import (
    max_pooling_3d,
)
from nkilib_src.nkilib.experimental.pooling.max_pooling_3d_torch import (
    max_pooling_3d_torch_ref,
)

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, LazyGoldenGenerator, Platforms, ValidationArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Per-dtype validation tolerances
_TOL = {
    nl.float32: (1e-5, 1e-6),
    nl.bfloat16: (2e-2, 1e-2),
    nl.float16: (5e-3, 1e-3),
}


def _input_shape(b, c, spatial, data_format):
    """5-D input shape for the given layout: NCDHW (B,C,D,H,W) or NDHWC (B,D,H,W,C)."""
    D, H, W = spatial[0], spatial[1], spatial[2]
    if data_format == "NDHWC":
        return (b, D, H, W, c)
    return (b, c, D, H, W)


def _output_shape(b, c, out_spatial, output_format):
    """5-D output shape for the given layout."""
    Do, Ho, Wo = out_spatial[0], out_spatial[1], out_spatial[2]
    if output_format == "NDHWC":
        return (b, Do, Ho, Wo, c)
    return (b, c, Do, Ho, Wo)


def generate_max_pooling_3d_inputs(b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format):
    """Generate a single 5-D input for the 3D max pool test in the given layout."""
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()
    shape = _input_shape(b, c, spatial, data_format)
    value = generate_tensor(name="src_tensor", shape=shape, dtype=dtype)
    return {
        "src_tensor": value,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "data_format": data_format,
        "output_format": output_format,
    }


# fmt: off
# (b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format)
MAX_POOL_3D_PARAM_NAMES = "b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format"

# Basic tests
MAX_POOL_3D_BASIC_ALL_PARAMS = [
    # k2s2, unpadded
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.bfloat16, "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float16,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    # k3s1 pad1
    pytest.param(8, 16, (6, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (6, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), nl.bfloat16, "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (6, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), nl.float16,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    # C > 128, NCDHW
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.bfloat16, "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float16,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    # k3s2 pad1
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.bfloat16, "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.float16,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    # k2s2, unpadded
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float32,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.bfloat16, "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float16,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    # k3s1 pad1
    pytest.param(8, 16, (6, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), nl.float32,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (6, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), nl.bfloat16, "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (6, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), nl.float16,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    # C > 128, NDHWC
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float32,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.bfloat16, "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float16,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    # k3s2 pad1
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.float32,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.bfloat16, "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.float16,  "NDHWC", "NDHWC", marks=pytest.mark.fast),
    # NCDHW in -> NDHWC out
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float32,  "NCDHW", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.bfloat16, "NCDHW", "NDHWC", marks=pytest.mark.fast),
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.bfloat16, "NCDHW", "NDHWC", marks=pytest.mark.fast),
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float16,  "NCDHW", "NDHWC", marks=pytest.mark.fast),
    # NDHWC in -> NCDHW out
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float32,  "NDHWC", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (4, 8, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.bfloat16, "NDHWC", "NCDHW", marks=pytest.mark.fast),
    pytest.param(8, 16, (5, 7, 7), (3, 3, 3), (2, 2, 2), (1, 1, 1), nl.bfloat16, "NDHWC", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 130, (4, 6, 6), (2, 2, 2), (2, 2, 2), (0, 0, 0), nl.float16,  "NDHWC", "NCDHW", marks=pytest.mark.fast),
    # Spatially tiled
    pytest.param(1, 4, (1, 96, 96), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 4, (1, 128, 128), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 4, (4, 96, 96),   (2, 3, 3), (2, 2, 2), (0, 1, 1), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 4, (1, 1, 16384), (1, 1, 3), (1, 1, 2), (0, 0, 1), nl.float32,  "NCDHW", "NCDHW", marks=pytest.mark.fast),
    pytest.param(1, 4, (1, 176, 176), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NDHWC", "NDHWC", marks=pytest.mark.fast),
    pytest.param(1, 4, (1, 176, 176), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NCDHW", "NDHWC", marks=pytest.mark.fast),
    pytest.param(1, 4, (1, 176, 176), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NDHWC", "NCDHW", marks=pytest.mark.fast),
]

# ResNet tests
MAX_POOL_3D_RESNET_PARAMS = [
    (2, 64, (1, 112, 112), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NCDHW", "NCDHW"),
    (2, 64, (1, 216, 384), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NCDHW", "NCDHW"),
    (32, 64, (1, 112, 112), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NCDHW", "NCDHW"),
    (32, 64, (1, 216, 384), (1, 3, 3), (1, 2, 2), (0, 1, 1), nl.bfloat16, "NCDHW", "NCDHW"),
]
# fmt: on


def _make_test_body(b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format):
    """Build the (input_generator, output_tensors, validation) trio for one config."""
    out_spatial = tuple((spatial[a] + 2 * padding[a] - kernel_size[a]) // stride[a] + 1 for a in range(3))
    out_shape = _output_shape(b, c, out_spatial, output_format)

    def input_generator(test_config):
        return generate_max_pooling_3d_inputs(
            b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format
        )

    def output_tensors(kernel_input):
        return {"out": np.zeros(out_shape, dtype=dtype)}

    def golden_generator():
        kernel_input = input_generator(None)
        return torch_ref_wrapper(max_pooling_3d_torch_ref)(**kernel_input)

    rtol, atol = _TOL[dtype]
    validation = ValidationArgs(
        golden_output=LazyGoldenGenerator(
            output_ndarray=output_tensors(None),
            lazy_golden_generator=golden_generator,
        ),
        relative_accuracy=rtol,
        absolute_accuracy=atol,
    )
    return input_generator, output_tensors, validation


@pytest.mark.platforms(exclude=[Platforms.TRN1])
@pytest_test_metadata(name="Max Pooling 3D")
@pytest_marks(["pooling", "max_pooling_3d"])
class TestMaxPooling3D:
    @pytest_marks(["basic"])
    @pytest.mark.parametrize(MAX_POOL_3D_PARAM_NAMES, MAX_POOL_3D_BASIC_ALL_PARAMS)
    def test_max_pooling_3d_basic(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        b,
        c,
        spatial,
        kernel_size,
        stride,
        padding,
        dtype,
        data_format,
        output_format,
    ):
        """Basic max pooling 3D tests."""
        input_generator, output_tensors, validation = _make_test_body(
            b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format
        )
        test_framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=max_pooling_3d,
            torch_ref=torch_ref_wrapper(max_pooling_3d_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        test_framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            custom_validation_args=validation,
        )

    @pytest_marks(["resnet"])
    @pytest.mark.parametrize(MAX_POOL_3D_PARAM_NAMES, MAX_POOL_3D_RESNET_PARAMS)
    def test_max_pooling_3d_resnet(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        b,
        c,
        spatial,
        kernel_size,
        stride,
        padding,
        dtype,
        data_format,
        output_format,
    ):
        """ResNet stem max pooling tests."""
        input_generator, output_tensors, validation = _make_test_body(
            b, c, spatial, kernel_size, stride, padding, dtype, data_format, output_format
        )
        test_framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=max_pooling_3d,
            torch_ref=torch_ref_wrapper(max_pooling_3d_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        test_framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            custom_validation_args=validation,
        )
