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

"""
Tests for depthwise_conv1d_implicit_gemm kernel.
"""

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.conv.depthwise_conv1d import depthwise_conv1d_implicit_gemm


def generate_kernel_inputs(
    batch: int, channels: int, width: int, kernel_size: int, dtype, padding=((0, 0), (0, 0)), stride=(1, 1)
):
    """Generate kernel inputs for depthwise conv1d."""
    generate_tensor = gaussian_tensor_generator()
    img_ref = generate_tensor(name="img_ref", shape=(batch, channels, 1, width), dtype=dtype)
    filter_ref = generate_tensor(name="filter_ref", shape=(channels, 1, 1, kernel_size), dtype=dtype)
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


def golden_depthwise_conv1d(inp_np):
    """Golden function using PyTorch depthwise conv2d."""
    input_tensor = torch.from_numpy(inp_np["img_ref"].astype(np.float32))
    kernel = torch.from_numpy(inp_np["filter_ref"].astype(np.float32))
    C = input_tensor.shape[1]
    padding = inp_np["padding"]
    stride = inp_np["stride"]
    # Convert padding from ((0,0), (p_l, p_r)) to (0, p) for PyTorch
    padding_pytorch = (0, padding[1][0])
    output = torch.nn.functional.conv2d(
        input_tensor, kernel, bias=None, stride=stride, padding=padding_pytorch, groups=C
    )
    return {"output": output.numpy().astype(inp_np["img_ref"].dtype)}


@final
@pytest_test_metadata(
    name="Depthwise Conv1D Explicit GEMM",
    pytest_marks=["conv", "depthwise"],
)
class TestDepthwiseConv1d:
    """Test class for depthwise conv1d kernel."""

    # fmt: off
    # Fast test cases - representative configs covering all scenarios
    fast_params = "batch, channels, width, kernel_size, stride, padding, dtype"
    fast_test_cases = [
        # No padding, no stride
        [1, 16, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        # With padding
        [1, 256, 512, 64, (1, 1), ((0, 0), (10, 10)), nl.bfloat16],
        # With stride
        [1, 256, 512, 64, (1, 2), ((0, 0), (0, 0)), nl.bfloat16],
        # With stride and padding
        [1, 256, 512, 64, (1, 2), ((0, 0), (5, 5)), nl.bfloat16],
        # Qwen3 config
        [1, 8192, 4096, 4093, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        # FS-ASR config
        [1, 512, 5000, 8, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
    ]

    # Comprehensive test cases - all variations
    all_params = "batch, channels, width, kernel_size, stride, padding, dtype"
    all_test_cases = [
        # Small kernels
        [1, 16, 512, 3, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 16, 512, 7, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 16, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 256, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        
        # Large kernels (S > 128)
        [1, 16, 512, 200, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 256, 512, 256, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 256, 1024, 512, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        
        # Large channels
        [1, 1024, 1024, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 2048, 2048, 256, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        
        # Large width (Q > 512)
        [1, 256, 2048, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 256, 4096, 256, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        
        # Batch variations
        [2, 256, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [4, 256, 512, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        
        # Edge cases
        [1, 2, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 256, 256, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        [1, 256, 129, 128, (1, 1), ((0, 0), (0, 0)), nl.bfloat16],
        
        # Different dtypes
        [1, 256, 512, 64, (1, 1), ((0, 0), (0, 0)), nl.float32],
        [1, 256, 512, 128, (1, 1), ((0, 0), (0, 0)), nl.float16],
        
        # With padding
        [1, 16, 512, 3, (1, 1), ((0, 0), (1, 1)), nl.bfloat16],
        [1, 16, 512, 7, (1, 1), ((0, 0), (3, 3)), nl.bfloat16],
        [1, 256, 512, 64, (1, 1), ((0, 0), (10, 10)), nl.bfloat16],
        [1, 16, 512, 200, (1, 1), ((0, 0), (10, 10)), nl.bfloat16],
        pytest.param(
            1, 1024, 1024, 128, (1, 1), ((0, 0), (5, 5)), nl.bfloat16,
            marks=pytest.mark.skip(reason="Skipped: neuron-profile view OOM on 128GB TRN instance during JSON generation"),
            id="1-1024-1024-128-1_1-0_0_5_5-bfloat16-SKIP_OOM",
        ),
        [1, 256, 2048, 128, (1, 1), ((0, 0), (10, 10)), nl.bfloat16],
        [2, 256, 512, 64, (1, 1), ((0, 0), (5, 5)), nl.bfloat16],
        [1, 256, 128, 128, (1, 1), ((0, 0), (1, 1)), nl.bfloat16],
        
        # With stride
        [1, 16, 512, 3, (1, 2), ((0, 0), (0, 0)), nl.bfloat16],  # Small kernel + stride=2
        [1, 256, 512, 64, (1, 4), ((0, 0), (0, 0)), nl.bfloat16],  # Medium kernel + stride=4
        [1, 16, 512, 200, (1, 2), ((0, 0), (0, 0)), nl.bfloat16],  # Large kernel (S>128) + stride
        [1, 256, 1024, 256, (1, 3), ((0, 0), (0, 0)), nl.bfloat16],  # Large kernel (S>128) + stride
        [1, 2048, 2048, 256, (1, 4), ((0, 0), (0, 0)), nl.bfloat16],  # Large channels + stride
        [1, 256, 4096, 256, (1, 16), ((0, 0), (0, 0)), nl.bfloat16],  # Large width + large stride
        [1, 256, 512, 64, (1, 2), ((0, 0), (0, 0)), nl.float32],  # Stride + float32
        
        # With stride and padding
        [1, 16, 512, 3, (1, 2), ((0, 0), (1, 1)), nl.bfloat16],  # Small kernel + stride + padding
        [1, 256, 512, 64, (1, 2), ((0, 0), (10, 10)), nl.bfloat16],  # Medium kernel + stride + padding
        [1, 16, 512, 200, (1, 2), ((0, 0), (10, 10)), nl.bfloat16],  # Large kernel + stride + padding
        [1, 1024, 1024, 128, (1, 2), ((0, 0), (5, 5)), nl.bfloat16],  # Large channels + stride + padding
    ]
    # fmt: on

    @staticmethod
    def idfn(val):
        """Generate readable test IDs for parametrized tests."""
        if hasattr(val, "name"):
            return val.name
        if isinstance(val, type):
            return val.__name__
        if isinstance(val, tuple):
            return str(val).replace("(", "").replace(")", "").replace(", ", "_").replace(" ", "")
        return str(val)

    @pytest.mark.fast
    @pytest.mark.parametrize(fast_params, fast_test_cases, ids=idfn)
    def test_depthwise_conv1d_fast(
        self,
        test_manager: Orchestrator,
        batch: int,
        channels: int,
        width: int,
        kernel_size: int,
        stride: tuple,
        padding: tuple,
        dtype,
    ):
        """Fast test covering all conv1d scenarios."""
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=2)
        kernel_input = generate_kernel_inputs(batch, channels, width, kernel_size, dtype, padding, stride)

        W_padding_l, W_padding_r = padding[1]
        stride_w = stride[1]
        Q = (width + W_padding_l + W_padding_r - kernel_size) // stride_w + 1
        output_placeholder = {"output": np.zeros((batch, channels, 1, Q), dtype=kernel_input["img_ref"].dtype)}

        def create_golden():
            return golden_depthwise_conv1d(kernel_input)

        test_manager.execute(
            KernelArgs(
                kernel_func=depthwise_conv1d_implicit_gemm,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_golden,
                        output_ndarray=output_placeholder,
                    ),
                    absolute_accuracy=1e-2,
                    relative_accuracy=1e-2,
                ),
            )
        )

    @pytest.mark.parametrize(all_params, all_test_cases, ids=idfn)
    def test_depthwise_conv1d_all(
        self,
        test_manager: Orchestrator,
        batch: int,
        channels: int,
        width: int,
        kernel_size: int,
        stride: tuple,
        padding: tuple,
        dtype,
    ):
        """Comprehensive test for all conv1d configurations."""
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=2)
        kernel_input = generate_kernel_inputs(batch, channels, width, kernel_size, dtype, padding, stride)

        W_padding_l, W_padding_r = padding[1]
        stride_w = stride[1]
        Q = (width + W_padding_l + W_padding_r - kernel_size) // stride_w + 1
        output_placeholder = {"output": np.zeros((batch, channels, 1, Q), dtype=kernel_input["img_ref"].dtype)}

        def create_golden():
            return golden_depthwise_conv1d(kernel_input)

        test_manager.execute(
            KernelArgs(
                kernel_func=depthwise_conv1d_implicit_gemm,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_golden,
                        output_ndarray=output_placeholder,
                    ),
                    absolute_accuracy=1e-2,
                    relative_accuracy=1e-2,
                ),
            )
        )
