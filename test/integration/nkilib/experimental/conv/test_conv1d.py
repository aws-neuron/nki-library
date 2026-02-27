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

"""Tests for Conv1D kernel using UnitTestFramework."""

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import Optional, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import ActFnType
from nkilib_src.nkilib.experimental.conv.conv1d import conv1d
from nkilib_src.nkilib.experimental.conv.conv1d_torch import conv1d_torch_ref


def generate_conv1d_inputs(
    batch: int,
    in_channels: int,
    out_channels: int,
    sequence_length: int,
    filter_size: int,
    stride: int,
    pad_left: int,
    pad_right: int,
    dilation: int,
    use_bias: bool,
    activation_fn: Optional[ActFnType],
    lnc_shard: bool,
    dtype,
):
    """
    Generate inputs for conv1d kernel test.

    Args:
        batch: Batch size (B)
        in_channels: Number of input channels (C_in)
        out_channels: Number of output channels (C_out)
        sequence_length: Input sequence length (L)
        filter_size: Convolution filter/kernel size (K)
        stride: Convolution stride
        pad_left: Left padding size
        pad_right: Right padding size
        dilation: Dilation factor
        use_bias: Whether to include bias in convolution
        activation_fn: Activation function to apply (or None)
        lnc_shard: Whether to enable LNC sharding
        dtype: Data type for tensors

    Returns:
        dict: Dictionary of input tensors and parameters for kernel
    """
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()

    x_in = generate_tensor(name="x_in", shape=(batch, in_channels, sequence_length), dtype=dtype)
    filters = generate_tensor(name="filters", shape=(filter_size, in_channels, out_channels), dtype=dtype)
    bias = generate_tensor(name="bias", shape=(out_channels,), dtype=dtype) if use_bias else None

    return {
        "x_in": x_in,
        "filters": filters,
        "bias": bias,
        "stride": stride,
        "padding": (pad_left, pad_right),
        "dilation": dilation,
        "activation_fn": activation_fn,
        "lnc_shard": lnc_shard,
    }


# fmt: off
# Parameter names for pytest.mark.parametrize
CONV1D_PARAM_NAMES = (
    "batch, in_channels, out_channels, sequence_length, filter_size, stride, "
    "pad_left, pad_right, dilation, use_bias, activation_fn, lnc_shard, dtype"
)

# Basic test parameters
CONV1D_BASIC_PARAMS = [
    # Small
    (1, 16, 32, 20, 3, 1, 0, 0, 1, False, None, False, nl.float32),
    (1, 16, 32, 20, 3, 1, 0, 0, 1, True, ActFnType.SiLU, False, nl.float32),
    (2, 64, 128, 40, 5, 2, 2, 2, 2, True, ActFnType.GELU, False, nl.float32),

    # Medium
    (1, 48, 96, 64, 3, 1, 1, 1, 1, False, None, False, nl.float32),
    (1, 64, 128, 100, 5, 1, 2, 2, 1, True, ActFnType.GELU, False, nl.float32),
    (2, 96, 192, 80, 7, 2, 3, 3, 1, False, ActFnType.SiLU, False, nl.float32),

    # Large
    (1, 256, 256, 200, 3, 1, 1, 1, 1, False, None, False, nl.float32),
    (1, 192, 384, 300, 5, 1, 3, 3, 1, False, ActFnType.SiLU, False, nl.float32),
    (1, 256, 256, 512, 3, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.float32),

    # Kernel size
    (1, 32, 64, 50, 1, 1, 0, 0, 1, False, None, False, nl.float32),
    (1, 32, 64, 50, 8, 1, 3, 3, 1, True, ActFnType.GELU, False, nl.float32),
    (1, 48, 64, 128, 100, 1, 1, 1, 1, True, None, False, nl.float32),

    # Stride
    (1, 64, 128, 100, 3, 2, 1, 1, 1, False, None, False, nl.float32),
    (1, 64, 128, 100, 3, 3, 1, 1, 1, False, None, False, nl.float32),
    (1, 64, 128, 100, 5, 10, 2, 2, 1, True, ActFnType.GELU, False, nl.float32),

    # Dilation
    (1, 32, 64, 50, 3, 1, 2, 2, 2, False, None, False, nl.float32),
    (1, 32, 64, 100, 3, 1, 3, 3, 3, True, ActFnType.GELU, False, nl.float32),
    (1, 64, 128, 150, 5, 2, 4, 6, 2, False, ActFnType.SiLU, False, nl.float32),

    # Padding
    (1, 32, 64, 50, 3, 1, 0, 2, 1, False, None, False, nl.float32),
    (1, 64, 128, 80, 5, 2, 1, 3, 1, True, ActFnType.GELU, False, nl.float32),
    (1, 48, 96, 60, 7, 1, 2, 4, 1, False, ActFnType.SiLU, False, nl.float32),

    # Batch size
    (1, 64, 128, 100, 3, 1, 1, 1, 1, False, None, False, nl.float32),
    (2, 64, 128, 100, 3, 1, 1, 1, 1, False, None, False, nl.float32),
    (8, 32, 64, 50, 3, 1, 1, 1, 1, False, ActFnType.SiLU, False, nl.float32),

    # LNC sharding
    (1, 16, 32, 20, 3, 1, 0, 0, 1, False, None, True, nl.float32),
    (1, 128, 256, 200, 5, 2, 2, 2, 1, True, ActFnType.GELU, True, nl.float32),
    (2, 96, 192, 150, 7, 1, 3, 3, 1, False, ActFnType.SiLU, True, nl.float32),

    # bfloat16
    (1, 16, 32, 20, 3, 1, 0, 0, 1, False, None, False, nl.bfloat16),
    (1, 16, 32, 20, 3, 1, 0, 0, 1, True, ActFnType.SiLU, False, nl.bfloat16),
    (1, 64, 128, 100, 5, 1, 2, 2, 1, True, ActFnType.GELU, False, nl.bfloat16),
]

# Whisper model test parameters
CONV1D_WHISPER_PARAMS = [
    # Conv1
    (2, 80, 384, 1500, 3, 1, 1, 1, 1, True, ActFnType.GELU, False, nl.float32),
    (1, 128, 1280, 1500, 3, 1, 1, 1, 1, True, ActFnType.GELU, True, nl.bfloat16),

    # Conv2
    (2, 384, 384, 1500, 3, 2, 1, 1, 1, True, ActFnType.GELU, False, nl.float32),
    (1, 1280, 1280, 1500, 3, 2, 1, 1, 1, True, ActFnType.GELU, True, nl.bfloat16),
]

# All test parameters combined
CONV1D_ALL_PARAMS = CONV1D_BASIC_PARAMS + CONV1D_WHISPER_PARAMS
# fmt: on


@pytest_test_metadata(
    name="Conv1D",
    pytest_marks=["conv", "conv1d"],
)
@final
class TestConv1DKernel:
    """Test class for Conv1D kernel validation using UnitTestFramework."""

    @pytest.mark.fast
    @pytest.mark.parametrize(CONV1D_PARAM_NAMES, CONV1D_BASIC_PARAMS)
    def test_conv1d_basic(
        self,
        test_manager: Orchestrator,
        batch: int,
        in_channels: int,
        out_channels: int,
        sequence_length: int,
        filter_size: int,
        stride: int,
        pad_left: int,
        pad_right: int,
        dilation: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run basic Conv1D tests covering various convolution configurations."""
        self._run_conv1d_test(
            test_manager=test_manager,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            sequence_length=sequence_length,
            filter_size=filter_size,
            stride=stride,
            pad_left=pad_left,
            pad_right=pad_right,
            dilation=dilation,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(CONV1D_PARAM_NAMES, CONV1D_WHISPER_PARAMS)
    def test_conv1d_whisper(
        self,
        test_manager: Orchestrator,
        batch: int,
        in_channels: int,
        out_channels: int,
        sequence_length: int,
        filter_size: int,
        stride: int,
        pad_left: int,
        pad_right: int,
        dilation: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run Whisper model Conv1D tests for Conv1 and Conv2 layer configurations."""
        self._run_conv1d_test(
            test_manager=test_manager,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            sequence_length=sequence_length,
            filter_size=filter_size,
            stride=stride,
            pad_left=pad_left,
            pad_right=pad_right,
            dilation=dilation,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    @pytest.mark.parametrize(CONV1D_PARAM_NAMES, CONV1D_ALL_PARAMS)
    def test_conv1d_all(
        self,
        test_manager: Orchestrator,
        batch: int,
        in_channels: int,
        out_channels: int,
        sequence_length: int,
        filter_size: int,
        stride: int,
        pad_left: int,
        pad_right: int,
        dilation: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """Run all Conv1D tests combining basic and Whisper configurations."""
        self._run_conv1d_test(
            test_manager=test_manager,
            batch=batch,
            in_channels=in_channels,
            out_channels=out_channels,
            sequence_length=sequence_length,
            filter_size=filter_size,
            stride=stride,
            pad_left=pad_left,
            pad_right=pad_right,
            dilation=dilation,
            use_bias=use_bias,
            activation_fn=activation_fn,
            lnc_shard=lnc_shard,
            dtype=dtype,
        )

    def _run_conv1d_test(
        self,
        test_manager: Orchestrator,
        batch: int,
        in_channels: int,
        out_channels: int,
        sequence_length: int,
        filter_size: int,
        stride: int,
        pad_left: int,
        pad_right: int,
        dilation: int,
        use_bias: bool,
        activation_fn: Optional[ActFnType],
        lnc_shard: bool,
        dtype,
    ) -> None:
        """
        Execute a Conv1D kernel test with specified parameters using UnitTestFramework.

        Args:
            test_manager: Test orchestrator for kernel execution
            batch: Batch size (B)
            in_channels: Number of input channels (C_in)
            out_channels: Number of output channels (C_out)
            sequence_length: Input sequence length (L)
            filter_size: Convolution filter/kernel size (K)
            stride: Convolution stride
            pad_left: Left padding size
            pad_right: Right padding size
            dilation: Dilation factor
            use_bias: Whether to include bias in convolution
            activation_fn: Activation function to apply (or None)
            lnc_shard: Whether to enable LNC sharding
            dtype: Data type for tensors
        """
        # Calculate output shape
        L_out = (sequence_length + pad_left + pad_right - dilation * (filter_size - 1) - 1) // stride + 1

        def input_generator(test_config):
            return generate_conv1d_inputs(
                batch=batch,
                in_channels=in_channels,
                out_channels=out_channels,
                sequence_length=sequence_length,
                filter_size=filter_size,
                stride=stride,
                pad_left=pad_left,
                pad_right=pad_right,
                dilation=dilation,
                use_bias=use_bias,
                activation_fn=activation_fn,
                lnc_shard=lnc_shard,
                dtype=dtype,
            )

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((batch, out_channels, L_out), dtype=dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=conv1d,
            torch_ref=torch_ref_wrapper(conv1d_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        # Use higher tolerance for bfloat16
        rtol = 5e-2 if dtype == nl.bfloat16 else 1e-5
        atol = 5e-2 if dtype == nl.bfloat16 else 1e-5

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(),
            rtol=rtol,
            atol=atol,
        )
