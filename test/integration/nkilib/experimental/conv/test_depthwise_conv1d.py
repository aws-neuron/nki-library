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

"""Tests for depthwise_conv1d_implicit_gemm kernel using UnitTestFramework."""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.conv.depthwise_conv1d import depthwise_conv1d_implicit_gemm
from nkilib_src.nkilib.experimental.conv.depthwise_conv1d_torch import depthwise_conv1d_implicit_gemm_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_inputs(batch: int, channels: int, width: int, kernel_size: int, dtype, padding, stride):
    """Generate kernel inputs for depthwise conv1d."""
    np.random.seed(42)
    img_ref = np.random.randn(batch, channels, 1, width).astype(dtype)
    filter_ref = np.random.randn(channels, 1, 1, kernel_size).astype(dtype)
    return {
        "img_ref": img_ref,
        "filter_ref": filter_ref,
        "padding": padding,
        "stride": stride,
        "rhs_dilation": (1, 1),
        "lhs_dilation": (1, 1),
        "feature_group_count": channels,
        "batch_group_count": 1,
    }


# fmt: off
PARAMS = "batch, channels, width, kernel_size, stride, padding, dtype"
_ABBREVS = {
    "batch": "b", "channels": "c", "width": "w", "kernel_size": "ks",
    "stride": "st", "padding": "pad", "dtype": "dt",
}

FAST_TEST_CASES = [
    # No padding, no stride
    (1, 16, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    # With padding
    (1, 256, 512, 64, (1, 1), ((0, 0), (10, 10)), nl.bfloat16),
    # With stride
    (1, 256, 512, 64, (1, 2), ((0, 0), (0, 0)), nl.bfloat16),
    # With stride and padding
    (1, 256, 512, 64, (1, 2), ((0, 0), (5, 5)), nl.bfloat16),
    # FS-ASR config
    (1, 512, 5000, 8, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
]

ALL_TEST_CASES = [
    # Small kernels
    (1, 16, 512, 3, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 16, 512, 7, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 16, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),

    # Large kernels (S > 128)
    (1, 16, 512, 200, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 512, 256, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 1024, 512, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),

    # Large channels
    (1, 1024, 1024, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 2048, 2048, 256, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),

    # Large width (Q > 512)
    (1, 256, 2048, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 4096, 256, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),

    # Batch variations
    (2, 256, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (4, 256, 512, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),

    # Edge cases
    (1, 2, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 256, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 129, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),

    # Different dtypes
    (1, 256, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.float32),
    (1, 256, 512, 128, (1, 1), ((0, 0), (0, 0)), nl.float16),

    # With padding
    (1, 16, 512, 3, (1, 1), ((0, 0), (1, 1)), nl.bfloat16),
    (1, 16, 512, 7, (1, 1), ((0, 0), (3, 3)), nl.bfloat16),
    (1, 256, 512, 64, (1, 1), ((0, 0), (10, 10)), nl.bfloat16),
    (1, 16, 512, 200, (1, 1), ((0, 0), (10, 10)), nl.bfloat16),
    pytest.param(
        1, 1024, 1024, 128, (1, 1), ((0, 0), (5, 5)), nl.bfloat16,
        marks=pytest.mark.skip(reason="Skipped: neuron-explorer view OOM on 128GB TRN instance during JSON generation"),
        id="1-1024-1024-128-1_1-0_0_5_5-bfloat16-SKIP_OOM",
    ),
    (1, 256, 2048, 128, (1, 1), ((0, 0), (10, 10)), nl.bfloat16),
    (2, 256, 512, 64, (1, 1), ((0, 0), (5, 5)), nl.bfloat16),
    (1, 256, 128, 128, (1, 1), ((0, 0), (1, 1)), nl.bfloat16),

    # With stride
    (1, 16, 512, 3, (1, 2), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 512, 64, (1, 4), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 16, 512, 200, (1, 2), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 1024, 256, (1, 3), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 2048, 2048, 256, (1, 4), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 4096, 256, (1, 16), ((0, 0), (0, 0)), nl.bfloat16),
    (1, 256, 512, 64, (1, 2), ((0, 0), (0, 0)), nl.float32),

    # With stride and padding
    (1, 16, 512, 3, (1, 2), ((0, 0), (1, 1)), nl.bfloat16),
    (1, 256, 512, 64, (1, 2), ((0, 0), (10, 10)), nl.bfloat16),
    (1, 16, 512, 200, (1, 2), ((0, 0), (10, 10)), nl.bfloat16),
    (1, 1024, 1024, 128, (1, 2), ((0, 0), (5, 5)), nl.bfloat16),
    
    # Qwen3 config
    (1, 8192, 4096, 4093, (1, 1), ((0, 0), (0, 0)), nl.bfloat16),
]
# fmt: on


@pytest_test_metadata(name="Depthwise Conv1D Explicit GEMM")
@pytest_marks(["conv", "depthwise"])
@final
class TestDepthwiseConv1d:
    """Test class for depthwise conv1d kernel."""

    def _run_test(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        channels: int,
        width: int,
        kernel_size: int,
        stride: tuple,
        padding: tuple,
        dtype,
    ):
        W_padding_l, W_padding_r = padding[1]
        stride_w = stride[1]
        Q = (width + W_padding_l + W_padding_r - kernel_size) // stride_w + 1

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=depthwise_conv1d_implicit_gemm,
            torch_ref=torch_ref_wrapper(depthwise_conv1d_implicit_gemm_torch_ref),
            kernel_input_generator=lambda _: generate_inputs(
                batch, channels, width, kernel_size, dtype, padding, stride
            ),
            output_tensor_descriptor=lambda ki: {
                "output": np.zeros((batch, channels, 1, Q), dtype=ki["img_ref"].dtype)
            },
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(enable_birsim=False, logical_nc_config=2, platform_target=platform_target),
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.fast
    @pytest_parametrize(PARAMS, FAST_TEST_CASES, abbrevs=_ABBREVS)
    def test_depthwise_conv1d_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        channels,
        width,
        kernel_size,
        stride,
        padding,
        dtype,
    ):
        """Fast test covering all conv1d scenarios."""
        self._run_test(test_manager, platform_target, batch, channels, width, kernel_size, stride, padding, dtype)

    @pytest_parametrize(PARAMS, ALL_TEST_CASES, abbrevs=_ABBREVS)
    def test_depthwise_conv1d_all(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        channels,
        width,
        kernel_size,
        stride,
        padding,
        dtype,
    ):
        """Comprehensive test for all conv1d configurations."""
        self._run_test(test_manager, platform_target, batch, channels, width, kernel_size, stride, padding, dtype)
